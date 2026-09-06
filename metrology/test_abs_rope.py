# -*- coding: utf-8 -*-
"""CPU unit test for python/inference/abs_rope.py (acceptance criterion #1).

The contract identity (mathematical, float32-tolerant — the two sides
compute the angle by different associativity, direct (s+i)*inv_freq vs
i*inv_freq + s*inv_freq, so 1-ulp drift is expected and allowed):

    apply_abs_rope(k, s)  ~=  rotate_k_cache_rope(apply_abs_rope(k, 0), s)

Also pins the B7 regression STRUCTURALLY: the old wrong path (delta-rotating
PRE-RoPE K directly) lands EVERY row at the same absolute position s —
wrong[:, i] == rotate(row_i as position-0, s) for every i — whereas
apply_abs_rope places row i at s+i.

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
# conftest stubs the `inference` package as a plain module whenever
# inference.mdocdataset's heavy deps are missing; pop the stub so the REAL
# namespace package (python/inference) resolves for the imports below
for _stub in ("inference.mdocdataset", "inference"):
    sys.modules.pop(_stub, None)

from inference.abs_rope import apply_abs_rope  # noqa: E402
from inference.rope_reposition import rotate_k_cache_rope  # noqa: E402

THETA = 10000.0
HEAD_DIM = 64
TOL = 1e-4  # fp32 angle associativity drift grows with position (~4e-5 @512)


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
        assert torch.allclose(lhs, rhs, atol=TOL, rtol=0), (
            f"identity broke at start={start}: max|d|={(lhs - rhs).abs().max()}"
        )


def test_positions_are_distinct_b7_regression():
    rot = FakeRotary()
    k = _rand_k(heads=2, length=4, seed=1)
    L = k.shape[1]
    start = 100

    correct = apply_abs_rope(k, start, rot)
    at_zero = apply_abs_rope(k, 0, rot)  # row i carries position i (0..L-1)
    for i in range(L):
        # correct: row i sits at absolute position start+i — delta from its
        # position-i anchor is exactly `start`
        expect = rotate_k_cache_rope(at_zero[:, i:i + 1, :], start, THETA, "default")
        assert torch.allclose(correct[:, i:i + 1, :], expect, atol=TOL, rtol=0), (
            f"row {i} not at absolute position {start + i}"
        )

    # the OLD wrong path (B7): delta-rotating PRE-RoPE K lands EVERY row at
    # the SAME absolute position start, erasing intra-block relative position
    wrong = rotate_k_cache_rope(k, start, THETA, "default")
    for i in range(L):
        landed = rotate_k_cache_rope(k[:, i:i + 1, :], start, THETA, "default")
        assert torch.allclose(wrong[:, i:i + 1, :], landed, atol=TOL, rtol=0), (
            f"row {i} collapsed onto absolute position {start} — the B7 bug, pinned"
        )
        if i > 0:
            assert not torch.allclose(
                correct[:, i:i + 1, :], rotate_k_cache_rope(k[:, i:i + 1, :], start, THETA, "default"),
                atol=1e-3, rtol=0,
            ), "correct path must NOT equal the collapsed path"


def test_matches_real_prefill_rotation_math():
    # spot-check on one token / one dim pair.  The half-split convention
    # (rotate_half chunks the LAST dim in halves) pairs dim j with dim
    # j+D/2; both rotate by angle p * inv_freq[j]:
    #   out[j] = x[j]*cos - x[j+D/2]*sin ;  out[j+D/2] = x[j+D/2]*cos + x[j]*sin
    rot = FakeRotary()
    k = torch.zeros(1, 1, HEAD_DIM, dtype=torch.float32)
    j, p = 5, 97
    a, b = 0.6, -0.8
    k[0, 0, j], k[0, 0, j + HEAD_DIM // 2] = a, b
    out = apply_abs_rope(k, p, rot)[0, 0]
    ang = p * float(rot.inv_freq[j])
    expect_a = a * math.cos(ang) - b * math.sin(ang)
    expect_b = b * math.cos(ang) + a * math.sin(ang)
    assert abs(out[j].item() - expect_a) < 1e-4
    assert abs(out[j + HEAD_DIM // 2].item() - expect_b) < 1e-4
