"""Tensor contracts for native eviction attention scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from benchmarks.memory_runtime import native_eviction


def test_chunked_causal_gqa_scores_match_dense_repeat_kv_softmax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(native_eviction, "_SCORE_CHUNK_SIZE", 3)

    generator = torch.Generator().manual_seed(20260913)
    batch, kv_heads, groups = 1, 2, 2
    query_tokens, prefix_tokens, head_dim = 3, 8, 4
    query_heads = kv_heads * groups
    query_start = 4
    scaling = head_dim**-0.5

    query = torch.randn(
        batch, query_heads, query_tokens, head_dim, generator=generator
    ) * 0.2
    keys = torch.randn(
        batch, kv_heads, prefix_tokens, head_dim, generator=generator
    ) * 0.2
    # Give protected-prefix keys substantial mass so omitting them from the
    # softmax denominator would also change scores for later candidate keys.
    query[..., 0] += 1.5
    keys[:, :, 0, 0] += 5.0
    keys[:, :, 1, 0] += 4.0

    actual = native_eviction._recent_attention_scores(
        torch,
        query,
        keys,
        query_start=query_start,
        scaling=scaling,
    )

    repeated_keys = (
        keys[:, :, None, :, :]
        .expand(batch, kv_heads, groups, prefix_tokens, head_dim)
        .reshape(batch, query_heads, prefix_tokens, head_dim)
    )
    logits = torch.matmul(query, repeated_keys.transpose(-2, -1)).float()
    logits.mul_(scaling)
    query_positions = torch.arange(query_start, query_start + query_tokens)
    key_positions = torch.arange(prefix_tokens)
    allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
    logits.masked_fill_(~allowed[None, None, :, :], -torch.inf)
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
    expected = probabilities.reshape(
        batch, kv_heads, groups, query_tokens, prefix_tokens
    ).mean(dim=(0, 2, 3))

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert actual[:, 7].equal(torch.zeros_like(actual[:, 7]))

    without_protected = logits.clone()
    without_protected[..., :2] = -torch.inf
    wrong_probabilities = torch.softmax(
        without_protected, dim=-1, dtype=torch.float32
    )
    wrong_scores = wrong_probabilities.reshape(
        batch, kv_heads, groups, query_tokens, prefix_tokens
    ).mean(dim=(0, 2, 3))
    assert not torch.allclose(
        actual[:, 2:7], wrong_scores[:, 2:7], rtol=1e-3, atol=1e-4
    )
