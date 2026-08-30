"""D4: Fixed-budget pseudo-KV / residual capsule arms.
DEPRECATED / FROZEN (2026-08-30, prereg v2 / handoff §2.6): this module
was written before the D1 upper-bound verdict and has ZERO call sites; the
downstream review verified fatal defects in its core algorithm (see
docs/research/ and the handoff list).  Do NOT run, do NOT patch — the v2
plan rewrites these arms from scratch AFTER the D1 verdict.  Kept verbatim
for the record of what was tried.



Three arms at SAME capsule bytes (r slots per block):
  reskv_block_r      ResKV: r centroid K/V pairs + log token count
  keepkv_zip_block_r KeepKV: merge by attention score, Electoral Votes EMA
  resa_rank1_block   RESA: rank-1 prior output + normalizer + mean stats

All capsules cover the entire block; repair removes G_k by default
(double-counting prevention), with a keepG ablation.

Source papers:
  ResKV (arXiv:2607.29591): fixed budget b = m + r, residual cache
  KeepKV: attention-score merge with Electoral Votes
  RESA (ICLR 2026): rank-1 attention prior

Timing: capsule encode counts as T_capture (during compression);
capsule load+apply counts as T_edit (repair).
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

logger = __name__ if __name__ != "__main__ else" else __name__


# ---------------------------------------------------------------------------
# Capsule encoders: raw block KV -> r-slot compressed representation
# ---------------------------------------------------------------------------

def encode_reskv_block(
    keys_per_layer: List[torch.Tensor],    # each (kv_heads, L, D)
    values_per_layer: List[torch.Tensor],
    r: int,
) -> List[Dict[str, torch.Tensor]]:
    """ResKV-style: r centroid K/V pairs + per-centroid token count.

    Clustering is per-layer K-means (spherical, since K is post-RoPE on a
    sphere); V centroid is the mean of assigned V. Count is used at decode
    for log-count attention weighting.
    """
    capsules = []
    for k, v in zip(keys_per_layer, values_per_layer):
        # k: (H, L, D), v: (H, L, D)
        H, L, D = k.shape
        if L <= r:
            # not enough tokens to compress; store as-is
            capsules.append({"k": k, "v": v, "count": torch.ones(H, L, device=k.device)})
            continue
        # simple uniform-bucket centroids (k-means is overkill for oracle)
        # split L tokens into r roughly equal groups
        boundaries = torch.linspace(0, L, r + 1, dtype=torch.long, device=k.device)
        cent_k = []
        cent_v = []
        counts = []
        for i in range(r):
            lo, hi = int(boundaries[i]), int(boundaries[i + 1])
            if hi <= lo:
                hi = lo + 1
            cent_k.append(k[:, lo:hi, :].mean(dim=1))  # (H, D)
            cent_v.append(v[:, lo:hi, :].mean(dim=1))  # (H, D)
            counts.append(hi - lo)
        cent_k = torch.stack(cent_k, dim=1)  # (H, r, D)
        cent_v = torch.stack(cent_v, dim=1)  # (H, r, D)
        counts = torch.tensor(counts, dtype=torch.float32, device=k.device)
        capsules.append({"k": cent_k, "v": cent_v, "count": counts})
    return capsules


def encode_keepkv_block(
    keys_per_layer: List[torch.Tensor],
    values_per_layer: List[torch.Tensor],
    queries_per_layer: List[torch.Tensor],  # (q_heads, L, D) — for attention scoring
    r: int,
) -> List[Dict[str, torch.Tensor]]:
    """KeepKV-style: merge by attention score with Electoral Votes.

    Simplified: score each token by mean attention from other tokens in the
    block (proxy for importance), keep top-r as pseudo-KV, attach vote counts.
    """
    capsules = []
    for k, v, q in zip(keys_per_layer, values_per_layer, queries_per_layer):
        H_kv, L, D = k.shape
        if L <= r:
            capsules.append({"k": k, "v": v, "votes": torch.ones(H_kv, L, device=k.device)})
            continue
        # attention proxy: ||k||^2 (tokens with large keys attract more attention)
        scores = k.norm(dim=-1).mean(dim=0)  # (L,) mean over heads
        top_idx = scores.topk(min(r, L)).indices.sort().values
        sel_k = k[:, top_idx, :]  # (H, r, D)
        sel_v = v[:, top_idx, :]
        # votes: uniform for oracle (EMA would be used online)
        votes = torch.ones(H_kv, len(top_idx), device=k.device) * (L / len(top_idx))
        capsules.append({"k": sel_k, "v": sel_v, "votes": votes})
    return capsules


def encode_resa_block(
    keys_per_layer: List[torch.Tensor],
    values_per_layer: List[torch.Tensor],
    queries_per_layer: List[torch.Tensor],
) -> List[Dict[str, torch.Tensor]]:
    """RESA rank-1: store the block's aggregate attention numerator and normalizer.

    H_k ≈ Σ_i φ(k_i)^T v_i  (numerator matrix, D×D)
    z_k ≈ Σ_i φ(k_i)        (denominator vector, D)
    At decode: contribution = φ(q) @ H_k, normalized by φ(q) @ z_k.
    """
    capsules = []
    for k, v, q in zip(keys_per_layer, values_per_layer, queries_per_layer):
        # Use only kv heads (queries have more heads due to GQA)
        H_kv = k.shape[0]
        q_reduced = q[:H_kv] if q.shape[0] > H_kv else q  # (H_kv, L, D)
        # numerator: sum over tokens of outer(k_i, v_i) — (H, D, D)
        # For memory: use low-rank approximation (just sum k_i^T v_i)
        numerator = torch.einsum("hld,hle->hde", k, v)  # (H, D, D)
        # denominator: sum of k_i — (H, D)
        denominator = k.sum(dim=1)  # (H, D)
        # mean stats
        k_mean = k.mean(dim=1)  # (H, D)
        v_mean = v.mean(dim=1)  # (H, D)
        capsules.append({
            "H": numerator, "z": denominator,
            "k_mean": k_mean, "v_mean": v_mean,
        })
    return capsules


# ---------------------------------------------------------------------------
# Capsule -> cache splice
# ---------------------------------------------------------------------------

def _log_count_scaling(counts: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """ResKV-style: log(count) scaling for attention."""
    return torch.log1p(counts / temperature)


@torch.inference_mode()
def splice_capsule_to_cache(
    cache: Any,
    capsule: List[Dict[str, torch.Tensor]],
    capsule_type: str,
    logical_start: int,
    rope_theta: float,
    rope_type: Optional[str],
    position_offset: int = 0,
) -> Any:
    """Splice capsule slots into cache at the target block's position.

    For reskv/keepkv: append r pseudo-KV slots (K rotated, V as-is),
    with log-count/vote scaling applied to V (proxy for attention mass).
    For resa: append k_mean/v_mean as a single summary slot (rank-1 approx).
    """
    from inference.rope_reposition import rotate_k_cache_rope

    for layer, cap in zip(cache.layers, capsule):
        if capsule_type == "reskv":
            k = cap["k"]  # (H, r, D)
            v = cap["v"]
            counts = cap["count"]  # (r,) or (H, r)
            # rotate K to logical positions (uniformly spread across block span)
            r_slots = k.shape[-2]
            rotated_k = rotate_k_cache_rope(k, logical_start, rope_theta, rope_type)
            # apply log-count scaling to V
            if counts.dim() == 1:
                scale = _log_count_scaling(counts)  # (r,)
                v = v * scale.unsqueeze(0).unsqueeze(-1)  # (H, r, D)
            else:
                scale = _log_count_scaling(counts)
                v = v * scale.unsqueeze(-1)
        elif capsule_type == "keepkv":
            k = cap["k"]
            v = cap["v"]
            votes = cap["votes"]
            rotated_k = rotate_k_cache_rope(k, logical_start, rope_theta, rope_type)
            if votes.dim() == 1:
                v = v * _log_count_scaling(votes).unsqueeze(0).unsqueeze(-1)
            else:
                v = v * _log_count_scaling(votes).unsqueeze(-1)
        elif capsule_type == "resa":
            k = cap["k_mean"].unsqueeze(1)  # (H, 1, D)
            v = cap["v_mean"].unsqueeze(1)  # (H, 1, D)
            rotated_k = rotate_k_cache_rope(k, logical_start, rope_theta, rope_type)
        else:
            raise ValueError(f"unknown capsule type {capsule_type}")

        # add batch dim and append
        layer.keys = torch.cat([layer.keys, rotated_k.unsqueeze(0)], dim=-2)
        layer.values = torch.cat([layer.values, v.unsqueeze(0)], dim=-2)
    return cache
