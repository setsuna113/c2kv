"""D5: LESS / Residual-Mass Accounting merged structural arm.
DEPRECATED / FROZEN (2026-08-30, prereg v2 / handoff §2.6): this module
was written before the D1 upper-bound verdict and has ZERO call sites; the
downstream review verified fatal defects in its core algorithm (see
docs/research/ and the handoff list).  Do NOT run, do NOT patch — the v2
plan rewrites these arms from scratch AFTER the D1 verdict.  Kept verbatim
for the record of what was tried.



Source: Get More with LESS (arXiv:2402.09398) + Residual-Mass Accounting.

Mechanism: instead of selecting representative tokens, store the ENTIRE
block's attention "numerator ledger" and "denominator ledger":

  H_k = Σ_i ψ(k_i)^T v_i    (D×D numerator matrix per head)
  z_k = Σ_i ψ(k_i)           (D denominator vector per head)

At repair: remove G_k, compute φ(q) @ H_k and φ(q) @ z_k for the current
query, and add the result to the remaining cache's numerator/denominator.

Note: φ and ψ are feature maps (identity for standard attention, or the
nonlinear feature map for softmax attention via random features).
For this implementation we use the identity feature map (linear attention
approximation), which is the simplest valid instantiation.

The two papers are the SAME structural arm in the block-replacement setting;
feature map and training objective are config ablations, not separate arms.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))


def encode_less_block(
    keys_per_layer: List[torch.Tensor],    # (H_kv, L, D)
    values_per_layer: List[torch.Tensor],  # (H_kv, L, D)
) -> List[Dict[str, torch.Tensor]]:
    """Encode H_k and z_k ledgers for one block.

    H_k = Σ_i k_i^T v_i   (H, D, D) — attention numerator
    z_k = Σ_i k_i          (H, D)    — attention denominator
    """
    ledgers = []
    for k, v in zip(keys_per_layer, values_per_layer):
        H = torch.einsum("hld,hle->hde", k, v)  # (H, D, D)
        z = k.sum(dim=1)  # (H, D)
        ledgers.append({"H": H, "z": z})
    return ledgers


@torch.inference_mode()
def apply_less_to_query(
    ledgers: List[Dict[str, torch.Tensor]],
    query: torch.Tensor,         # (1, H_q, 1, D) — current query
    scale: float = 1.0,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Compute the block's contribution for a given query.

    Returns (numerator_contribution, denominator_contribution) per layer:
      num = φ(q) @ H_k   → (1, H, 1, D)
      den = φ(q) @ z_k   → (1, H, 1, 1)
    """
    nums, dens = [], []
    H_q = query.shape[1]
    for ledger in ledgers:
        H_kv = ledger["H"].shape[0]
        # GQA: average query heads to kv head count
        if H_q > H_kv:
            q = query.view(1, H_kv, H_q // H_kv, -1).mean(dim=2)  # (1, H_kv, 1, D)
        else:
            q = query[:, :H_kv]
        num = torch.einsum("hqd,hde->hqe", q[0], ledger["H"]).unsqueeze(0) * scale
        den = torch.einsum("hqd,hd->hq", q[0], ledger["z"]).unsqueeze(0).unsqueeze(-1) * scale
        nums.append(num)
        dens.append(den)
    return nums, dens


@torch.inference_mode()
def splice_less_summary(
    cache: Any,
    ledgers: List[Dict[str, torch.Tensor]],
    logical_start: int,
    rope_theta: float,
    rope_type: Optional[str],
) -> Any:
    """Splice a single summary slot per layer (rank-1 reconstruction).

    The summary slot's K = z_k (denominator vector, captures "what keys"),
    V = H_k @ (z_k / ||z_k||^2) (the "average value" weighted by key mass).
    This is a rank-1 approximation of the full block's attention.
    """
    from inference.rope_reposition import rotate_k_cache_rope

    for layer, ledger in zip(cache.layers, ledgers):
        z = ledger["z"]  # (H, D)
        H_mat = ledger["H"]  # (H, D, D)

        # summary K: the dominant key direction
        k_summary = z / (z.norm(dim=-1, keepdim=True) + 1e-8)  # (H, D)

        # summary V: H @ k_hat (projection of numerator onto key direction)
        v_summary = torch.einsum("hde,hd->he", H_mat, k_summary)  # (H, D)

        # add seq dim
        k_summary = k_summary.unsqueeze(1)  # (H, 1, D)
        v_summary = v_summary.unsqueeze(1)  # (H, 1, D)

        # rotate K to logical position
        k_rotated = rotate_k_cache_rope(k_summary, logical_start, rope_theta, rope_type)

        layer.keys = torch.cat([layer.keys, k_rotated.unsqueeze(0)], dim=-2)
        layer.values = torch.cat([layer.values, v_summary.unsqueeze(0)], dim=-2)
    return cache
