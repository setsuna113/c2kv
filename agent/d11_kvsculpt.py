"""D11 KVSculpt (offline, no cards; S2 of the 2026-08-31 plan).

Directly OPTIMIZE r synthetic KV slots against the block's own raw
teacher — the ceiling that makes D4's empty results attributable
("capsule form insufficient" vs "our capsules badly built").  Reuses the
GRKV ridge step (grkv_v_edit IS the closed-form V half of KVSculpt) and
evaluates on the distortion bench's attention-output error.

Per (layer, kv head):
  teacher  O = softmax_causal(q K_raw/√d) V_raw
  student  A_s = softmax(q S/√d)          (S: r free keys, NO causal —
           tokens never attend gists in the real forward, counterfactual
           either way, declared in prereg v2.10)
  V half   ridge solve A_s V = O          (grkv step)
  K half   Adam on S over ||A_s V̂ − O||²  + λ_lse · logsumexp-margin term
           (mass-shaping: keep the student's attention mass from
           collapsing onto one slot — the "LSE term" in the plan)

Everything runs on dumped sidecar tensors (want_q=True) — GPU-minutes on
CPU-scale tensors, independent of the |R| gate.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch


def _causal_teacher(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    Lq = q.shape[-2]
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    mask = torch.ones(Lq, logits.shape[-1], dtype=torch.bool, device=q.device).tril()
    logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(logits, dim=-1) @ v


def _ridge_V(A: torch.Tensor, O: torch.Tensor, ridge: float) -> torch.Tensor:
    """A: (n, r), O: (n, Dv) -> V: (r, Dv), fp32 ridge solve."""
    AtA = A.T @ A
    lam = ridge * torch.diagonal(AtA).mean().clamp_min(1e-8)
    eye = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)
    return torch.linalg.solve(AtA + lam * eye, A.T @ O)


def sculpt_head(
    q: torch.Tensor,          # (n_rep*L, D) teacher queries (one kv head)
    k_raw: torch.Tensor,      # (L, D)
    v_raw: torch.Tensor,      # (L, Dv)
    r: int,
    iters: int = 120,
    lr: float = 0.05,
    ridge: float = 1e-2,
    lam_lse: float = 0.01,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Optimize (S, V) for ONE kv head.  Returns S (r, D), V (r, Dv), info."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    O = _causal_teacher(q, k_raw, v_raw, scale)
    g = torch.Generator().manual_seed(seed)
    S = (0.1 * torch.randn(r, q.shape[-1], generator=g)).requires_grad_(True)
    with torch.no_grad():
        A0 = torch.softmax(torch.matmul(q, S.T) * scale, dim=-1)
        V = _ridge_V(A0, O, ridge)
    opt = torch.optim.Adam([S], lr=lr)
    hist = []
    for it in range(iters):
        opt.zero_grad()
        A = torch.softmax(torch.matmul(q, S.T) * scale, dim=-1)
        V = _ridge_V(A, O, ridge)
        out = A @ V
        err = ((out - O) ** 2).sum()
        # mass-shaping LSE term: penalize one-slot collapse (row logsumexp
        # of logits vs uniform reference)
        logits = torch.matmul(q, S.T) * scale
        lse = torch.logsumexp(logits, dim=-1).mean()
        loss = err + lam_lse * lse
        loss.backward()
        opt.step()
        if it % 20 == 0 or it == iters - 1:
            hist.append(round(float(err.sqrt() / O.norm().clamp_min(1e-9)), 4))
    with torch.no_grad():
        A = torch.softmax(torch.matmul(q, S.T) * scale, dim=-1)
        V = _ridge_V(A, O, ridge)
        rel = float(((A @ V - O).norm() / O.norm().clamp_min(1e-9)))
    return S.detach(), V.detach(), {"rel_err": round(rel, 4), "trace": hist}


def sculpt_block(
    k_raw: torch.Tensor,      # (H_kv, L, D) pre-RoPE raw keys (dump)
    v_raw: torch.Tensor,      # (H_kv, L, Dv)
    q_raw: torch.Tensor,      # (H_q, L, D)
    r: int,
    n_rep: int = 4,
    **kw,
) -> Dict:
    """Sculpt per kv head; returns the capsule {'k','v'} + diagnostics."""
    H, L, D = k_raw.shape
    Ss, Vs, infos = [], [], []
    for h in range(H):
        qh = q_raw.float()[h * n_rep:(h + 1) * n_rep].reshape(-1, D)
        S, V, info = sculpt_head(qh, k_raw.float()[h], v_raw.float()[h], r, seed=h, **kw)
        Ss.append(S)
        Vs.append(V)
        infos.append(info)
    return {
        "kind": "kvsculpt", "r": r,
        "k": torch.stack(Ss), "v": torch.stack(Vs),
        "rel_err_mean": sum(i["rel_err"] for i in infos) / len(infos),
        "trace_first": infos[0]["trace"], "trace_last": infos[-1]["trace"],
    }


def sculpt_bytes(cap: Dict) -> int:
    return 8 + cap["k"].numel() * 2 + cap["v"].numel() * 2
