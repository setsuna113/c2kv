# -*- coding: utf-8 -*-
"""CPU unit test for python/inference/abs_rope.py (acceptance criterion #1).

The contract identity: for PRE-RoPE K and any start s,

    apply_abs_rope(k, s)  ==  rotate_k_cache_rope(apply_abs_rope(k, 0), s)

i.e. "RoPE at absolute positions s..s+L-1 in one shot" equals "full RoPE at
0..L-1, then delta-rotate by s" — the composition law that makes both the
sidecar release path and the existing post-RoPE callers correct.  If this
identity fails, the release path has a wrong understanding of RoPE.

Also pins the B7 regression: the OLD wrong path (delta-rotating PRE-RoPE K
directly) collapses all L tokens onto one absolute position; apply_abs_rope
must place them at distinct positions.

Run:  pytest metrology/test_abs_rope.py -v   (needs torch; skips without)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")  # models.gist_utils imports transformers

import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "python", _REPO_ROOT / "python" / "inference"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from inference.abs_rope import apply_abs_rope  # noqa: E402
from inference.rope_reposition import rotate_k_cache_rope  # noqa: E402

THETA = 10000.0
HEAD_DIM = 64


class FakeRotary:
    """Mirrors Qwen3RotaryEmbedding.forward: fp32 outer(pos, inv_freq),
    cos/sin cast to x.dtype, shape (B, S, D), attention_scaling = 1."""

    def __init__(self, theta: float = THETA, head_dim: int = HEAD_DIM):
        self.inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )

    def __call__(self, x: torch.Tensor, position_ids: torch.Tensor):
        freqs = torch.outer(position_ids[0].to(torch.float32), self.inv_freq)  # (S, D/2)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)  # (1, S, D)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


def _rand_k(heads: int = 4, length: int = 10, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(heads, length, HEAD_DIM, generator=g, dtype=torch.float32)


def test_identity_one_shot_equals_full_rope_then_delta():
    rot = FakeRotary()
    k = _rand_k()
    for start in (0, 1, 37, 512):
        lhs = apply_abs_rope(k, start, rot)
        full_at_zero = apply_abs_rope(k, 0, rot)
        rhs = rotate_k_cache_rope(full_at_zero, start, THETA, "default")
        assert lhs.shape == k.shape
        assert torch.equal(lhs, rhs), f"identity broke at start={start}: max|d|={(lhs-rhs).abs().max()}"


def test_positions_are_distinct_b7_regression():
    rot = FakeRotary()
    k = _rand_k(heads=2, length=3, seed=1)
    out = apply_abs_rope(k, 100, rot)
    # distinct absolute positions -> no two rows of the rotated block coincide
    rows = [out[0, i] for i in range(out.shape[-2])]
    assert not torch.equal(rows[0], rows[1])
    assert not torch.equal(rows[1], rows[2])
    # the OLD wrong path (delta-rotating PRE-RoPE K) collapses onto one position
    collapsed = rotate_k_cache_rope(k, 100, THETA, "default")
    assert torch.equal(collapsed[0, 0], collapsed[0, 1])  # the bug, pinned


def test_matches_real_prefill_rotation_math():
    # direct spot-check of the rotation formula on one token/one pair of dims:
    # RoPE at absolute position p rotates pair (x0, x1) by angle p * inv_freq[j]
    rot = FakeRotary()
    k = torch.zeros(1, 1, HEAD_DIM, dtype=torch.float32)
    j, p = 5, 97
    k[0, 0, 2 * j], k[0, 0, 2 * j + 1] = 0.6, -0.8
    out = apply_abs_rope(k, p, rot)[0, 0]
    ang = p * rot.inv_freq[j].item()
    expect0 = 0.6 * math.cos(ang) - (-0.8) * math.sin(ang)
    expect1 = (-0.8) * math.cos(ang) + 0.6 * math.sin(ang)
    assert abs(out[2 * j].item() - expect0) < 1e-5
    assert abs(out[2 * j + 1].item() - expect1) < 1e-5
