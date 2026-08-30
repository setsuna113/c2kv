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
        delta[h] = torch.linalg.solve(AtA + lam * torch.eye(g, dtype=AtA.dtype), A.T @ R)
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
    """R (H_kv, g): unnormalized mass the gist tokens SHOULD carry relative
    to what one raw token carries, averaged over the block's own queries.

    R_g = mean_q ( Σ_i exp(q·k_i/√d) ) / exp(q·k_g/√d)
    """
    H_kv, g, D = k_gist.shape
    scale = 1.0 / math.sqrt(D)
    R = torch.zeros(H_kv, g, dtype=torch.float64)
    for h in range(H_kv):
        qh = q_raw.double()[h * n_rep:(h + 1) * n_rep]           # (n_rep, L, D)
        raw_mass = torch.exp(
            torch.matmul(qh, k_raw.double()[h].transpose(-1, -2)) * scale
        ).sum(dim=-1, keepdim=True)                               # (n_rep, L, 1)
        gist_mass = torch.exp(
            torch.matmul(qh, k_gist.double()[h].transpose(-1, -2)) * scale
        )                                                          # (n_rep, L, g)
        R[h] = (raw_mass / gist_mass).mean(dim=(0, 1))
    return R.to(torch.float32)


def selkv_logit_bias(mass_ratio: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-key logit bias α·log R (inject through d_attn_ext / the live path)."""
    return alpha * torch.log(mass_ratio.clamp_min(1e-12))


def selkv_count_bias(token_count: int, g: int, alpha: float) -> torch.Tensor:
    """Control arm: α·log(token_count) uniform over the g gist tokens."""
    return torch.full((g,), alpha * math.log(max(token_count, 1)), dtype=torch.float32)
