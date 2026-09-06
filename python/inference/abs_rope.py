"""Absolute-position RoPE for PRE-RoPE K (sidecar release path).

`rotate_k_cache_rope` (rope_reposition.py) rotates an already-RoPE'd K cache
by a position delta: one angle ``inv_freq * delta_pos`` broadcast over all
sequence positions, which is only legal for K that already carries RoPE at
positions 0..L-1 (token i moves i -> i + delta).  The sidecar stores K
PRE-RoPE (position-free), so releasing it to absolute position ``start``
requires the full per-token RoPE with ``position_ids = arange(start,
start+L)`` — exactly the path ``models.gist_utils._concat_gist_key_values``
uses to place gist keys (rotary_emb + apply_rotary_pos_emb).  This module
mirrors that path and does not touch ``rotate_k_cache_rope``.
"""
from __future__ import annotations

import torch

from models.gist_utils import apply_rotary_pos_emb


def apply_abs_rope(
    k_pre_rope: torch.Tensor,
    start: int,
    rotary_emb,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """RoPE a PRE-RoPE K block onto absolute positions start..start+L-1.

    Args:
        k_pre_rope: ``(heads, L, head_dim)`` or ``(1, heads, L, head_dim)``,
            pre-RoPE (post-k_norm) keys as stored by the sidecar.
        start: absolute logical start position of the block.
        rotary_emb: the model's rotary module, called as
            ``rotary_emb(x, position_ids) -> (cos, sin)`` (the same callable
            `_concat_gist_key_values` uses).
        dtype / device: optional cast before rotation (splice-time cache
            alignment; the sidecar hands back CPU tensors).

    Returns the rotated K with the same layout as the input.
    """
    squeeze = k_pre_rope.dim() == 3
    k = k_pre_rope.unsqueeze(0) if squeeze else k_pre_rope
    if device is not None:
        k = k.to(device)
    if dtype is not None:
        k = k.to(dtype)
    seq_len = k.shape[-2]
    position_ids = torch.arange(start, start + seq_len, device=k.device).unsqueeze(0)
    cos, sin = rotary_emb(k, position_ids)
    rotated = apply_rotary_pos_emb(k, cos, sin)
    return rotated.squeeze(0) if squeeze else rotated
