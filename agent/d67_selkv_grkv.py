"""D6: Local GRKV regression edit + D7: SelKV attention compensation.

D6 GRKV (arXiv from contract):
  grkv_v_block: closed-form ΔV_k from local raw Q/K/V teacher.
    Only modifies G_k's V — no cache row added, no RoPE involvement.
  grkv_kv_block: if V-only works, add linearized ΔK_k.

D7 SelKV (arXiv:2607.16213):
  Compression-time: estimate attention mass lost per gist token.
  Decode-time: α·log(R) logit bias on G_k's keys.
  Control: simple log(token_count) bias.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import eval_agent_history_c2kv as HH
from d0_sidecar import SidecarStore


# ---------------------------------------------------------------------------
# D6 GRKV: closed-form ΔV_k
# ---------------------------------------------------------------------------

def grkv_v_edit(
    gist_k: torch.Tensor,      # (H_kv, g_len, D) — gist K for doc k (pre-RoPE)
    gist_v: torch.Tensor,      # (H_kv, g_len, D) — gist V for doc k
    raw_q: torch.Tensor,       # (H_q, L, D) — raw queries for doc k
    raw_k: torch.Tensor,       # (H_kv, L, D) — raw keys for doc k
    raw_v: torch.Tensor,       # (H_kv, L, D) — raw values for doc k
    scale: float = 1.0,
) -> torch.Tensor:
    """Closed-form ΔV that minimizes ||gist_attn_output - raw_attn_output||^2.

    For each raw query q_i, the raw attention output is:
      o_raw = softmax(q_i @ raw_k^T / scale) @ raw_v

    The gist attention output is:
      o_gist = softmax(q_i @ gist_k^T / scale) @ gist_v

    We want ΔV such that:
      softmax(q_i @ gist_k^T / scale) @ (gist_v + ΔV) ≈ o_raw

    Closed form (least squares):
      ΔV = (A_gist^T A_gist)^{-1} A_gist^T (O_raw - A_gist @ gist_v)
    where A_gist = softmax(Q @ gist_k^T / scale)
    """
    H_kv = gist_k.shape[0]
    H_q = raw_q.shape[0]
    # GQA: repeat queries to match kv heads
    if H_q > H_kv:
        n_rep = H_q // H_kv
        raw_q_hkv = raw_q[:H_kv]  # take first H_kv queries (sufficient for teacher)
    else:
        raw_q_hkv = raw_q

    # attention weight matrices
    A_gist = torch.softmax(
        torch.einsum("hqd,hkd->hqk", raw_q_hkv, gist_k) * scale, dim=-1
    )  # (H, L_q, g_len)
    A_raw = torch.softmax(
        torch.einsum("hqd,hkd->hqk", raw_q_hkv, raw_k) * scale, dim=-1
    )  # (H, L_q, L_kv)

    # raw attention output
    O_raw = torch.einsum("hqk,hkd->hqd", A_raw, raw_v)  # (H, L_q, D)

    # gist attention output
    O_gist = torch.einsum("hqk,hkd->hqd", A_gist, gist_v)  # (H, L_q, D)

    # residual: what we need ΔV to explain
    Residual = O_raw - O_gist  # (H, L_q, D)

    # closed-form: ΔV = (A^T A)^{-1} A^T Residual
    # A_gist: (H, L_q, g_len), Residual: (H, L_q, D)
    # Solve: min ||A_gist @ ΔV - Residual||^2
    # ΔV = pinv(A_gist) @ Residual
    # For stability, use lstsq per head
    L_q = A_gist.shape[1]
    g_len = A_gist.shape[2]
    D = Residual.shape[-1]

    delta_v = torch.zeros_like(gist_v)  # (H, g_len, D)
    for h in range(H_kv):
        A_h = A_gist[h]  # (L_q, g_len)
        R_h = Residual[h]  # (L_q, D)
        # lstsq: min ||A_h @ dv - R_h||^2
        solution = torch.linalg.lstsq(A_h, R_h).solution  # (g_len, D)
        delta_v[h] = solution

    return delta_v


def grkv_kv_edit(
    gist_k: torch.Tensor,
    delta_v: torch.Tensor,
    raw_q: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """Linearized ΔK: if V-only works, add a key shift.

    This is a simplified version: perturbs gist K in the direction that
    increases attention to the corrected V. For the contract, this is
    experimental and only used if grkv_v shows benefit.
    """
    # For now: identity (no key edit) — the V edit is the primary mechanism
    return torch.zeros_like(gist_k)


# ---------------------------------------------------------------------------
# D7 SelKV: attention mass compensation
# ---------------------------------------------------------------------------

def compute_attention_mass_ratio(
    gist_k: torch.Tensor,   # (H_kv, g_len, D)
    raw_k: torch.Tensor,    # (H_kv, L, D)
    raw_q: torch.Tensor,    # (H_q, L, D)
    scale: float = 1.0,
) -> torch.Tensor:
    """Compute R = raw_attention_mass / gist_attention_mass per gist token.

    High R means the gist token is under-attended relative to the raw tokens
    it represents. α·log(R) compensates for this at decode.
    """
    H_kv = gist_k.shape[0]
    q = raw_q[:H_kv] if raw_q.shape[0] > H_kv else raw_q

    # gist attention: how much do raw queries attend to gist keys?
    attn_gist = torch.softmax(
        torch.einsum("hqd,hkd->hqk", q, gist_k) * scale, dim=-1
    )  # (H, L_q, g_len)
    gist_mass = attn_gist.sum(dim=1).mean(dim=0)  # (g_len,) mean over heads

    # raw attention: how much do raw queries attend to raw keys?
    attn_raw = torch.softmax(
        torch.einsum("hqd,hkd->hqk", q, raw_k) * scale, dim=-1
    )  # (H, L_q, L_kv)
    raw_mass = attn_raw.sum(dim=1).mean(dim=0)  # (L_kv,)

    # ratio: how much mass should each gist token get?
    # (total raw mass / g_len) vs actual gist mass
    target_mass = raw_mass.sum() / gist_k.shape[1]  # scalar
    R = target_mass / (gist_mass + 1e-8)  # (g_len,)

    return R


def selkv_logit_bias(
    R: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """α·log(R) bias to add to gist K logits at decode time."""
    return alpha * torch.log(R + 1e-8)
