"""D6 GRKV + D7 SelKV, v2 (line A math; runtime wiring waits for |R|).

D6 grkv_v_edit — closed-form ΔV on the block's gist tokens:
  teacher  O_raw = softmax_causal(q_raw K_raw/√d) V_raw          (per kv head)
  student  A_gist = softmax(q_raw' K_gist/√d)  (q at the SAME absolute
           positions via apply_abs_rope; K_gist sliced from the live cache
           via gist_doc_spans — post-RoPE at blend positions)
  solve    A_gist ΔV = O_raw − A_gist V_gist   per head, fp32, ridge
  write    cache.values[..., phys_start:phys_end, :] += ΔV
(v1 defects fixed: bf16 lstsq -> fp32 solve with ridge; GQA uses ALL
n_rep teacher heads per kv head; scale is 1/√D DIVISION; A_raw is causal.)

D7 selkv_mass_ratio — the unnormalized mass ratio, the quantity v1 could
not express (its two separate softmaxes each summed to 1):
  R_g = ( Σ_{i in block tokens} e^{q·k_i/√d} ) / e^{q·k_g/√d}
averaged over the block's raw queries; injected as α·log R per-key logit
bias (d_attn_ext), control arm α·log(token_count).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch


def _causal_attn(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    Lq, Lk = q.shape[-2], k.shape[-2]
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    if Lq == Lk:
        mask = torch.ones(Lq, Lk, dtype=torch.bool, device=q.device).tril()
        logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def grkv_v_edit(
    k_gist: torch.Tensor,      # (H_kv, g, D) gist keys, post-RoPE at absolute positions
    v_gist: torch.Tensor,      # (H_kv, g, D) gist values
    q_raw: torch.Tensor,       # (H_q, L, D) raw queries at the SAME absolute positions
    k_raw: torch.Tensor,       # (H_kv, L, D) raw keys (post-RoPE, same positions)
    v_raw: torch.Tensor,       # (H_kv, L, D) raw values
    n_rep: int = 4,
    ridge: float = 1e-2,
) -> torch.Tensor:
    """ΔV (H_kv, g, D) in fp32; caller adds it to the cache's gist span."""
    H_kv, g, D = k_gist.shape
    scale = 1.0 / math.sqrt(D)          # DIVISION, not multiplication
    delta = torch.zeros(H_kv, g, D, dtype=torch.float32)
    for h in range(H_kv):
        # teacher: ALL n_rep query heads of this kv head, stacked
        qh = q_raw.float()[h * n_rep:(h + 1) * n_rep]           # (n_rep, L, D)
        o_raw = torch.matmul(
            _causal_attn(qh, k_raw.float()[h], scale), v_raw.float()[h]
        )                                                        # (n_rep, L, D)
        # student: same queries over the gist tokens (no causal constraint)
        a_gist = torch.softmax(
            torch.matmul(qh, k_gist.float()[h].transpose(-1, -2)) * scale, dim=-1
        )                                                        # (n_rep, L, g)
        A = a_gist.reshape(-1, g)                                # (n_rep*L, g)
        R = (o_raw - torch.matmul(a_gist, v_gist.float()[h])).reshape(-1, D)
        # ridge solve (g is small; A^T A near-collinear) in fp32
        AtA = A.T @ A
        lam = ridge * torch.diagonal(AtA).mean().clamp_min(1e-8)
        delta[h] = torch.linalg.solve(
            AtA + lam * torch.eye(g, dtype=AtA.dtype, device=AtA.device), A.T @ R)
    return delta


def grkv_writeback(cache_layer_values: torch.Tensor, phys_start: int, phys_end: int,
                   delta_v: torch.Tensor) -> None:
    """Apply the edit to a live cache layer's values (the v1 module never
    wrote anything back)."""
    cache_layer_values[..., phys_start:phys_end, :] += delta_v.to(cache_layer_values.dtype)


def selkv_mass_ratio(
    k_gist: torch.Tensor,      # (H_kv, g, D) the block's gist keys at real positions
    k_raw: torch.Tensor,       # (H_kv, L, D) the block's raw keys at the same positions
    q_raw: torch.Tensor,       # (H_q, L, D) raw queries at the same positions
    n_rep: int = 4,
) -> torch.Tensor:
    """log R (H_kv, g) — the LOG of the unnormalized mass ratio, averaged
    over the block's own queries IN LOG SPACE (geometric mean).

    log R_g = mean_q [ log Σ_i exp(q·k_i/√d) − q·k_g/√d ]

    Review G: the ratio of exponentials is heavy-tailed; an arithmetic
    mean would be dominated by whichever query sees a tiny gist mass,
    while the bias consumes α·log R — so average the logs directly.
    (logsumexp keeps the raw mass term overflow-safe.)
    """
    H_kv, g, D = k_gist.shape
    scale = 1.0 / math.sqrt(D)
    log_R = torch.zeros(H_kv, g, dtype=torch.float64)
    for h in range(H_kv):
        qh = q_raw.double()[h * n_rep:(h + 1) * n_rep]           # (n_rep, L, D)
        raw_logits = torch.matmul(qh, k_raw.double()[h].transpose(-1, -2)) * scale
        log_raw_mass = torch.logsumexp(raw_logits, dim=-1, keepdim=True)   # (n_rep, L, 1)
        gist_logits = torch.matmul(qh, k_gist.double()[h].transpose(-1, -2)) * scale
        log_R[h] = (log_raw_mass - gist_logits).mean(dim=(0, 1))
    return log_R.to(torch.float32)


def selkv_logit_bias(log_mass_ratio: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-key logit bias α·log R from the (already log-space) ratio."""
    return alpha * log_mass_ratio


def selkv_count_bias(token_count: int, g: int, alpha: float) -> torch.Tensor:
    """Control arm: α·log(token_count) uniform over the g gist tokens."""
    return torch.full((g,), alpha * math.log(max(token_count, 1)), dtype=torch.float32)


# ---------------------------------------------------------------------------
# D6 runtime caller: rotate the sidecar's pre-RoPE K/Q to absolute
# positions, slice the live gist span, solve, and write the edit back.
# (Review D/I: the student a_gist is deliberately NOT causal — in the real
# compression forward tokens never attend gists, so the student is
# counterfactual either way; declared in the D6 report.)
# ---------------------------------------------------------------------------

def grkv_edit_cache(
    cache,                       # live prefix cache (layers with .keys/.values)
    store,                       # SidecarStore holding doc k*'s raw K/V/Q (want_q=True)
    qid: str,
    k_star: int,
    gist_span,                   # (gs, ge) physical gist span of doc k* (gist_doc_spans + system_length)
    abs_start: int,              # doc k*'s absolute logical offset (offsets[k_star])
    rotary_emb,                  # model rotary for apply_abs_rope
    n_rep: int = 4,
    ridge: float = 1e-2,
):
    """Per-layer: ΔV solved against the block's OWN raw teacher and written
    back into the cache's gist span.  Returns the per-layer residual norms
    (diagnostic)."""
    from inference.abs_rope import apply_abs_rope

    gs, ge = gist_span
    layer0 = cache.layers[0]
    device, dtype = layer0.keys.device, layer0.keys.dtype
    k_raw = store.get(qid, k_star, "k", device=device, dtype=torch.float32)
    v_raw = store.get(qid, k_star, "v", device=device, dtype=torch.float32)
    q_raw = store.get(qid, k_star, "q", device=device, dtype=torch.float32)
    residuals = []
    for li, layer in enumerate(cache.layers):
        k_gist = layer.keys[0, :, gs:ge, :]                 # post-RoPE at blend positions
        v_gist = layer.values[0, :, gs:ge, :]
        k_rot = apply_abs_rope(k_raw[li], abs_start, rotary_emb)
        q_rot = apply_abs_rope(q_raw[li], abs_start, rotary_emb)
        delta = grkv_v_edit(k_gist.float(), v_gist.float(), q_rot, k_rot,
                            v_raw[li], n_rep=n_rep, ridge=ridge)
        before = layer.values[0, :, gs:ge, :].float().norm().item()
        layer.values[0, :, gs:ge, :] += delta.to(layer.values.dtype)
        residuals.append(before)
    return residuals
