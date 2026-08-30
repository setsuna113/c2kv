"""D3 codec tournament v2 (line A, 2026-08-31) — real bytes, real packing.

Replaces the deprecated d3_codecs.py (v1: int16 "4-bit", per-block refit
PCA billed per block, aatc that crashed on first call).  Every codec here
encodes one LAYER's sidecar block pair (k, v) — (H_kv, L, D) pre-RoPE —
into a ``d_payload.Payload`` whose nbytes are exact, and decodes back.

Shared artifacts (PCA basis, regression W) are fitted OFFLINE on held-out
blocks and passed in; their bytes ride in ``SharedArtifacts`` (session
amortized), never in the per-block payload.  In-block self-fit variants
are exposed explicitly as DIAGNOSTIC UPPER BOUNDS, never as the arm.

Quantization scheme (uniform): per-CHANNEL scalar min/delta (channel =
last dim), codes packed with ``bits.py`` grouped by width — header stays
small and the width map ships in it.

Conventions (spec): GQA maps kv_head = q_head // n_rep; attention math is
1/sqrt(D) with causal mask and fp32 softmax (d_attn_ext).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch

from d_payload import Payload


# ---------------------------------------------------------------------------
# quantization / packing helpers
# ---------------------------------------------------------------------------

def _asym_quant(x: torch.Tensor, bits: int):
    """Whole-tensor asymmetric uniform quantization: x ≈ mn + codes*delta."""
    mn = x.min()
    mx = x.max()
    delta = (mx - mn) / ((1 << bits) - 1) if mx > mn else torch.tensor(1.0, dtype=x.dtype)
    codes = torch.clamp(torch.round((x - mn) / delta), 0, (1 << bits) - 1).to(torch.int64)
    return codes, mn.detach().reshape(1).to(torch.float16), delta.detach().reshape(1).to(torch.float16)


def _asym_dequant(codes: torch.Tensor, mn: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    return codes.to(torch.float32) * delta.to(torch.float32) + mn.to(torch.float32)


def _pack_qchannels(p: Payload, name: str, t: torch.Tensor, bits) -> None:
    """Quantize the LAST dim of t channel-by-channel (per-channel scalar
    scale) and pack channels grouped by bit width."""
    C = t.shape[-1]
    widths = [int(b) for b in bits]
    p.header[f"{name}_widths"] = widths
    mns, dls = [], []
    by_width = {}
    for c in range(C):
        w = widths[c]
        codes, mn, dl = _asym_quant(t[..., c].float(), w)
        mns.append(mn)
        dls.append(dl)
        by_width.setdefault(w, []).append(codes.reshape(-1))
    for w, chunks in sorted(by_width.items()):
        p.add_packed(f"{name}_w{w}", torch.cat(chunks), w)
    p.add_floats(f"{name}_mn", torch.cat(mns))
    p.add_floats(f"{name}_dl", torch.cat(dls))


def _read_qchannels(p: Payload, name: str, lead_shape) -> torch.Tensor:
    widths = p.header[f"{name}_widths"]
    C = len(widths)
    mn = p.read_floats(f"{name}_mn")
    dl = p.read_floats(f"{name}_dl")
    n_per = max(1, math.prod(lead_shape))
    outs: List[torch.Tensor] = [None] * C
    for w in sorted(set(widths)):
        codes = p.read_packed(f"{name}_w{w}")
        idxs = [c for c in range(C) if widths[c] == w]
        for j, c in enumerate(idxs):
            outs[c] = _asym_dequant(
                codes[j * n_per:(j + 1) * n_per].reshape(*lead_shape), mn[c], dl[c]
            )
    return torch.stack(outs, dim=-1)


def waterfill_bits(weights: torch.Tensor, total_bits: float, lo: int = 2, hi: int = 8) -> torch.Tensor:
    """Deterministic bit allocation proportional to weights, clamped to
    [lo, hi], greedily repaired to meet the total budget."""
    w = weights.to(torch.float64)
    w = w / w.sum().clamp_min(1e-12)
    b = torch.clamp(torch.round(w * total_bits), lo, hi).to(torch.int64)
    target = int(round(total_bits))
    for _ in range(4):
        diff = target - int(b.sum())
        if diff == 0:
            break
        step = 1 if diff > 0 else -1
        order = torch.argsort(w, descending=(step > 0))
        i = 0
        while diff != 0 and i < 200 * len(b):
            idx = int(order[i % len(b)])
            nb = int(b[idx]) + step
            if lo <= nb <= hi:
                b[idx] = nb
                diff -= step
            i += 1
    return b


# ---------------------------------------------------------------------------
# codecs: raw baseline
# ---------------------------------------------------------------------------

def enc_raw_bf16(k: torch.Tensor, v: torch.Tensor) -> Payload:
    p = Payload("raw_bf16_v2")
    p.add_floats("k", k, dtype="float16")
    p.add_floats("v", v, dtype="float16")
    return p


def dec_raw_bf16(p: Payload) -> Tuple[torch.Tensor, torch.Tensor]:
    return p.read_floats("k"), p.read_floats("v")


def enc_raw_q4(k: torch.Tensor, v: torch.Tensor) -> Payload:
    """Asymmetric 4-bit with TRUE bit packing (v1 used one per-tensor scale
    and stored int16 codes; here scales are per head)."""
    p = Payload("raw_q4_v2")
    p.header["block_shape"] = list(k.shape)
    H = k.shape[0]
    for name, t in (("k", k), ("v", v)):
        codes = torch.zeros(H, t.shape[1] * t.shape[2], dtype=torch.int64)
        mns, dls = [], []
        for h in range(H):
            c, mn, dl = _asym_quant(t[h].float().reshape(-1), 4)
            codes[h] = c
            mns.append(mn)
            dls.append(dl)
        p.add_packed(f"{name}_codes", codes, 4)
        p.add_floats(f"{name}_mn", torch.cat(mns))
        p.add_floats(f"{name}_dl", torch.cat(dls))
    return p


def dec_raw_q4(p: Payload) -> Tuple[torch.Tensor, torch.Tensor]:
    H, L, D = p.header["block_shape"]
    out = []
    for name in ("k", "v"):
        codes = p.read_packed(f"{name}_codes")
        mn = p.read_floats(f"{name}_mn")
        dl = p.read_floats(f"{name}_dl")
        parts = [_asym_dequant(codes[h], mn[h], dl[h]).reshape(L, D) for h in range(H)]
        out.append(torch.stack(parts, dim=0))
    return out[0], out[1]


# ---------------------------------------------------------------------------
# vector_konly: K stored, V = K_hat @ W (W fitted offline, shared)
# ---------------------------------------------------------------------------

def _fit_v_regression(k: torch.Tensor, v: torch.Tensor, ridge: float = 1e-2) -> torch.Tensor:
    """Per-head ridge least-squares W: argmin ||K W - V||_F^2 + lam ||W||^2."""
    kf, vf = k.float(), v.float()
    A = torch.einsum("hld,hle->hde", kf, kf)
    B = torch.einsum("hld,hle->hde", kf, vf)
    trace = A.diagonal(dim1=1, dim2=2).sum(-1).mean()
    eye = torch.eye(A.shape[-1], dtype=A.dtype)
    lam = ridge * trace / A.shape[-1]
    return torch.linalg.solve(A + lam * eye, B)


def fit_v_regression_heldout(blocks: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Shared-artifact W fitted over held-out blocks (sum the normal
    equations across blocks, then solve once per head)."""
    A = None
    B = None
    for k, v in blocks:
        kf, vf = k.float(), v.float()
        A_b = torch.einsum("hld,hle->hde", kf, kf)
        B_b = torch.einsum("hld,hle->hde", kf, vf)
        A = A_b if A is None else A + A_b
        B = B_b if B is None else B + B_b
    trace = A.diagonal(dim1=1, dim2=2).sum(-1).mean()
    eye = torch.eye(A.shape[-1], dtype=A.dtype)
    return torch.linalg.solve(A + (1e-2 * trace / A.shape[-1]) * eye, B)


def enc_vector_konly(k: torch.Tensor, v: torch.Tensor, W: Optional[torch.Tensor] = None,
                     k_bits: int = 8, self_fit: bool = False) -> Payload:
    """K-only storage; V reconstructed as K_hat @ W (per head).

    W: (H, D, Dv) shared regression from fit_v_regression_heldout (shared
    artifact — NOT billed here).  self_fit=True fits on THIS block and
    bills it: diagnostic upper bound only."""
    fitted_here = self_fit or W is None
    if fitted_here:
        W = _fit_v_regression(k, v)
    p = Payload("vector_konly_v2_selffit" if fitted_here else "vector_konly_v2")
    p.header["block_shape"] = list(k.shape)
    H = k.shape[0]
    codes = torch.zeros(H, k.shape[1] * k.shape[2], dtype=torch.int64)
    mns, dls = [], []
    for h in range(H):
        c, mn, dl = _asym_quant(k[h].float().reshape(-1), k_bits)
        codes[h] = c
        mns.append(mn)
        dls.append(dl)
    p.add_packed("k_codes", codes, k_bits)
    p.add_floats("k_mn", torch.cat(mns))
    p.add_floats("k_dl", torch.cat(dls))
    if fitted_here:
        p.add_floats("W", W, dtype="float16")
    p.header["self_fit"] = fitted_here
    return p


def dec_vector_konly(p: Payload, W: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    H, L, D = p.header["block_shape"]
    codes = p.read_packed("k_codes")
    mn = p.read_floats("k_mn")
    dl = p.read_floats("k_dl")
    k = torch.stack([_asym_dequant(codes[h], mn[h], dl[h]).reshape(L, D) for h in range(H)])
    Wt = p.read_floats("W").to(torch.float32) if p.header.get("self_fit") else W.to(torch.float32)
    v = torch.einsum("hld,hde->hle", k, Wt)
    return k, v


# ---------------------------------------------------------------------------
# kvtc: offline PCA basis + water-filled coefficient bits
# ---------------------------------------------------------------------------

def fit_pca_basis(blocks: List[torch.Tensor], rank: int = 32) -> torch.Tensor:
    """Shared PCA basis per layer over held-out blocks (top-r right
    singular vectors of the concatenated token matrix)."""
    flat = torch.cat([b.float().reshape(-1, b.shape[-1]) for b in blocks], dim=0)
    U, S, Vh = torch.linalg.svd(flat, full_matrices=False)
    return Vh[:rank]


def enc_kvtc(k: torch.Tensor, v: torch.Tensor,
             basis_k: Optional[torch.Tensor] = None, basis_v: Optional[torch.Tensor] = None,
             budget_bytes: Optional[int] = None) -> Payload:
    """PCA-basis codec: basis fitted OFFLINE (shared artifact).  Per block:
    means + quantized coefficients, bits water-filled by basis mass."""
    H, L, D = k.shape
    self_fit = basis_k is None
    if self_fit:
        basis_k = fit_pca_basis([k], rank=32)
        basis_v = fit_pca_basis([v], rank=32)
    if budget_bytes is None:
        budget_bytes = (H * L * D * 2) // 4
    p = Payload("kvtc_v2_selffit" if self_fit else "kvtc_v2")
    # bits are PER COEFFICIENT (H*L of them per component): the width
    # vector's TOTAL budget is budget_bytes/(H*L) bits, split k/v
    per_elem_bits = budget_bytes * 8 * 0.75 / (H * L)
    bits_k = waterfill_bits(basis_k.float().norm(dim=-1), per_elem_bits / 2)
    bits_v = waterfill_bits(basis_v.float().norm(dim=-1), per_elem_bits / 2)
    for name, t, basis, bits in (("k", k, basis_k, bits_k), ("v", v, basis_v, bits_v)):
        mean = t.float().mean()  # single block mean (scalar per name)
        coef = torch.einsum("hld,rd->hlr", t.float() - mean, basis.float())
        _pack_qchannels(p, f"{name}_coef", coef, bits)
        p.add_floats(f"{name}_mean", mean.reshape(1))
    p.header["self_fit"] = self_fit
    return p


def dec_kvtc(p: Payload, basis_k: torch.Tensor, basis_v: torch.Tensor,
             lead_shape) -> Tuple[torch.Tensor, torch.Tensor]:
    out = []
    for name, basis in (("k", basis_k), ("v", basis_v)):
        coef = _read_qchannels(p, f"{name}_coef", lead_shape)
        mean = p.read_floats(f"{name}_mean")
        t = torch.einsum("hlr,rd->hld", coef, basis.float()) + mean.item()
        out.append(t)
    return out[0], out[1]


# ---------------------------------------------------------------------------
# aatc: attention-distortion channel bits, budget-matched to kvtc
# ---------------------------------------------------------------------------

def channel_sensitivity(k: torch.Tensor, q: torch.Tensor, n_rep: int = 4) -> torch.Tensor:
    """(H_kv, D) sensitivity per spec: for kv head h, sum the block's OWN
    raw queries (q heads h*n_rep..(h+1)*n_rep-1) attention-weighted channel
    variance:  sens[h, d] = sum_{qh,i} attn[qh, i, :] * (k[h, :, d] - mean)^2
    with causal attention at 1/sqrt(D), fp32."""
    H, L, D = k.shape
    kf = k.float()
    kbar = kf.mean(dim=1, keepdim=True)
    var = (kf - kbar) ** 2                        # (H, L, D)
    sens = torch.zeros(H, D, dtype=torch.float64)
    for h in range(H):
        qh = q.float()[h * n_rep:(h + 1) * n_rep]  # (n_rep, Lq, D)
        logits = torch.matmul(qh, kf[h].transpose(-1, -2)) / (D ** 0.5)
        mask = torch.ones(logits.shape[-2:], dtype=torch.bool, device=k.device).tril()
        attn = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1)
        sens[h] = (attn.sum(dim=(0, 1)) @ var[h]).to(torch.float64)
    return sens


def enc_aatc(k: torch.Tensor, v: torch.Tensor, q: torch.Tensor,
             target_bytes: int, n_rep: int = 4) -> Payload:
    """Attention-distortion bit allocation: channels water-filled by mean
    sensitivity, TOTAL budget matched to kvtc's measured bytes
    (target_bytes is actually read).  The width map ships in the header."""
    sens = channel_sensitivity(k, q, n_rep=n_rep).mean(dim=0)   # (D,) across heads
    # bits are PER ELEMENT (H*L rows per channel): the width vector's TOTAL
    # budget is target_bytes/(H*L) bits — the v1 code passed the raw byte
    # budget and every channel clamped to 8 bits
    H, L, D = k.shape
    # budget split evenly between k and v (each channel's width applies to
    # its H*L elements in BOTH tensors)
    per_elem_bits = target_bytes * 8 * 0.9 / (H * L) / 2
    bits = waterfill_bits(sens, per_elem_bits)
    p = Payload("aatc_v2")
    for name, t in (("k", k), ("v", v)):
        _pack_qchannels(p, name, t, bits)
    p.header["sens_bits"] = [int(b) for b in bits.tolist()]
    return p


def dec_aatc(p: Payload, lead_shape) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        _read_qchannels(p, "k", lead_shape),
        _read_qchannels(p, "v", lead_shape),
    )
