"""D4 capsules v2 (line A math; runtime wiring waits for |R|).

Three fixed-budget capsule encoders for one block's (k, v) — (H_kv, L, D):
- reskv: r centroid (K, V) pairs from per-(layer,head) k-means on the
  KEYS (v1 bucketed uniformly); the token count enters as a LOGIT BIAS
  log(count) via d_attn_ext — NOT as a V scale (v1's fatal error: V
  scaling changes only the numerator).
- keepkv: attention-importance-weighted MERGE — imp[i] = sum over the
  block's own queries of attn[q, i] (v1 ignored Q and scored ||k||);
  repeatedly merge the most similar pair, votes_a+votes_b, until r slots.
- resa: positive-feature ledgers H = Σψ(k)^T v, z = Σψ(k) with ψ=elu+1
  (identity makes z sign-indefinite — v1's denominator was undefined);
  folded at runtime as extra_num/extra_den through d_attn_ext.

All three report exact capsule bytes via d_payload and can be equalized
to a common budget ±2% by solving r (reskv/keepkv) against resa's fixed
size — the premise of the same-bytes comparison.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch


def _kmeans(x: torch.Tensor, r: int, iters: int = 8, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """(n, d) -> centroids (r, d), assignments (n,) — deterministic init."""
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    r = min(r, n)
    idx = torch.randperm(n, generator=g)[:r]
    cent = x[idx].clone()
    assign = torch.zeros(n, dtype=torch.long)
    for _ in range(iters):
        d2 = torch.cdist(x, cent)
        assign = d2.argmin(dim=1)
        for j in range(r):
            sel = x[assign == j]
            if sel.numel():
                cent[j] = sel.mean(dim=0)
    return cent, assign


def encode_reskv(k: torch.Tensor, v: torch.Tensor, r: int) -> Dict:
    """k-means centroid pairs + counts; decode carries log(count) as a
    LOGIT bias (never a V scale).  EMPTY clusters are dropped from the
    capsule entirely — v1.1 kept dead centroids with V=0 and bias=0, which
    still soaked up attention mass in the shared softmax (S0.5a)."""
    H, L, D = k.shape
    caps = []
    actual_rs = []
    for h in range(H):
        cent, assign = _kmeans(k.float()[h], r, seed=h)
        counts = torch.bincount(assign, minlength=cent.shape[0]).float()
        keep = [j for j in range(cent.shape[0]) if counts[j] > 0]
        vcent = torch.stack([
            v.float()[h][assign == j].mean(dim=0) for j in keep
        ])
        caps.append({"k": cent[keep], "v": vcent, "counts": counts[keep]})
        actual_rs.append(len(keep))
    # record the ACTUAL non-empty cluster count (v1.1 recorded the
    # requested r — a lying bytes/capacity ledger, S0.5b)
    return {"kind": "reskv", "r_requested": r, "r": int(min(actual_rs)),
            "r_per_head": actual_rs, "caps": caps}


def decode_reskv(cap: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(K, V, logit_bias=log c) stacked (H, r, D) / (H, r)."""
    ks = torch.stack([c["k"] for c in cap["caps"]])
    vs = torch.stack([c["v"] for c in cap["caps"]])
    bias = torch.stack([torch.log(c["counts"].clamp_min(1.0)) for c in cap["caps"]])
    return ks, vs, bias


def _attn_importance(k: torch.Tensor, q: torch.Tensor, n_rep: int, scale: float) -> torch.Tensor:
    """imp[i] = sum over the block's own queries of attn[q, i] (causal)."""
    H, L, D = k.shape
    imp = torch.zeros(H, L)
    for h in range(H):
        qh = q.float()[h * n_rep:(h + 1) * n_rep]
        logits = torch.matmul(qh, k.float()[h].transpose(-1, -2)) * scale
        mask = torch.ones(logits.shape[-2:], dtype=torch.bool, device=k.device).tril()
        attn = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
        imp[h] = attn.sum(dim=(0, 1))
    return imp


def encode_keepkv(k: torch.Tensor, v: torch.Tensor, q: torch.Tensor, r: int,
                  n_rep: int = 4) -> Dict:
    """Merge (not select): importance-weighted votes, most-similar-first,
    until r slots remain.  BOTH K and V merge votes-weighted — v1.1 kept
    the surviving token's ORIGINAL V while the votes bias claims the
    merged pair's attention mass, so the numerator was wrong by
    construction (S0.4).  Inputs are cloned (float() on a float32 input
    returns a VIEW; the old code mutated the caller's tensor)."""
    H, L, D = k.shape
    scale = 1.0 / math.sqrt(D)
    votes0 = _attn_importance(k, q, n_rep, scale)
    caps = []
    for h in range(H):
        keys = list(range(L))
        votes = votes0[h].clone()
        kmat = k.float()[h].clone()
        vmat = v.float()[h].clone()
        while len(keys) > r:
            mat = torch.nn.functional.normalize(kmat[keys], dim=-1)
            sim = mat @ mat.T
            sim.fill_diagonal_(-2.0)
            i, j = divmod(int(sim.argmax()), len(keys))
            a, b = min(i, j), max(i, j)
            wa, wb = votes[keys[a]], votes[keys[b]]
            wsum = (wa + wb).clamp_min(1e-9)
            kmat[keys[a]] = (kmat[keys[a]] * wa + kmat[keys[b]] * wb) / wsum
            vmat[keys[a]] = (vmat[keys[a]] * wa + vmat[keys[b]] * wb) / wsum
            votes[keys[a]] = wa + wb
            keys.pop(b)
        caps.append({
            "k": kmat[keys],
            "v": vmat[keys],
            "votes": votes[keys],
        })
    return {"kind": "keepkv", "r": r, "caps": caps}


def encode_less(k: torch.Tensor, v: torch.Tensor) -> Dict:
    """Positive-feature ledgers (LESS §3.1 Eq. 9-11 correspondence; the
    name 'resa' was unverifiable — no arXiv match — renamed, S1.5):
    H = Σψ(k)^T v (H_kv, D, Dv), z = Σψ(k) (H_kv, D).  Runtime folds them
    as extra_num=φ(q)H / extra_den=φ(q)z through d_attn_ext.

    NOTE: ψ is FIXED elu+1 here; LESS uses |GELU-MLP| and RMA a ReZero
    MLP, both TRAINED.  This capsule is therefore the UNTRAINED LOWER
    BOUND of both, not an implementation of either paper's method."""
    psi = torch.nn.functional.elu(k.float()) + 1.0     # positive
    H_ledger = torch.einsum("hld,hle->hde", psi, v.float())
    z = psi.sum(dim=1)                                  # (H_kv, D)
    return {"kind": "less", "H": H_ledger, "z": z}


# back-compat alias (S1.5 rename)
encode_resa = encode_less


def capsule_bytes(cap: Dict) -> int:
    """Exact bytes of the stored capsule (f16 arrays; no pickle).

    AXIS SEMANTICS (S0.5d, declared): the LESS/RESA ledger bytes are
    INDEPENDENT of block length (fixed H_ledger (H,D,D) + z (H,D)), while
    reskv/keepkv scale with r.  The same-bytes equalization therefore
    holds at fixed r for a GIVEN budget, not across block lengths — the
    report must state this instead of calling it a clean same-bytes trio.
    """
    total = 8  # kind tag
    if cap["kind"] == "reskv":
        for c in cap["caps"]:
            total += c["k"].numel() * 2 + c["v"].numel() * 2 + c["counts"].numel() * 4
    elif cap["kind"] == "keepkv":
        for c in cap["caps"]:
            total += c["k"].numel() * 2 + c["v"].numel() * 2 + c["votes"].numel() * 2
    elif cap["kind"] == "less":
        total += cap["H"].numel() * 2 + cap["z"].numel() * 2
    return total


def equalize_r(k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
               tol: float = 0.02, r_min: int = 2, r_max: int = 64) -> Dict:
    """Solve reskv/keepkv r so all three capsules land within ±tol of the
    resa budget (the same-bytes premise; v1 had no such code)."""
    target = capsule_bytes(encode_resa(k, v))
    out = {"resa_bytes": target, "reskv_r": None, "keepkv_r": None}
    for name, enc in (
        ("reskv_r", lambda r: capsule_bytes(encode_reskv(k, v, r))),
        ("keepkv_r", lambda r: capsule_bytes(encode_keepkv(k, v, q, r))),
    ):
        for r in range(r_min, r_max + 1):
            b = enc(r)
            if abs(b - target) <= tol * target or b >= target:
                out[name] = r
                out[f"{name}_bytes"] = b
                break
    return out
