"""Regression tests for the forced-prefix logp cache-pollution fix.

Round-1 bug: `_generate_one` ran `_generate_with_prefix` (model.generate with
use_cache=True, appending prompt+generated KV into prefix["cache"] in place)
BEFORE `_prefix_continuation_logp`, so the scored prefix contained the model's
own generated answer. All round-1 logp_prefix_* / delta_logp_prefix fields were
voided. The fix moves logp scoring ahead of generation.

These tests guard the two invariants that make the fix sound:
  1. `_prefix_continuation_logp` itself leaves the prefix cache pristine
     (cache.get_seq_length() identical before/after the scoring forward) and
     scores against exactly the original prefix length (deepcopy works).
  2. If the cache HAS been appended to (the round-1 pollution pattern), the
     scoring forward observes the larger cache — i.e. ordering matters and the
     pre-generation call site is load-bearing.

Runs on CPU with a stub model: `python tests/test_prefix_logp_cache_pristine.py`
or `pytest tests/test_prefix_logp_cache_pristine.py`.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(REPO_ROOT / "python" / "inference"))
sys.path.insert(0, str(REPO_ROOT / "agent"))

from transformers import DynamicCache  # noqa: E402

from eval_agent_history_c2kv import _prefix_continuation_logp  # noqa: E402


class _StubModel:
    """Minimal stand-in: returns fixed-seed random logits, records the cache
    length it was called with. Carries the `.model.config._attn_implementation`
    handle that _prefix_continuation_logp toggles."""

    def __init__(self, vocab_size: int = 64):
        self.model = types.SimpleNamespace(
            config=types.SimpleNamespace(_attn_implementation="eager")
        )
        self.device = torch.device("cpu")
        self.vocab_size = vocab_size
        self.seen_cache_length = None

    def __call__(self, input_ids, attention_mask=None, position_ids=None,
                 past_key_values=None, use_cache=False, **kwargs):
        self.seen_cache_length = past_key_values.get_seq_length()
        batch, seq = input_ids.shape
        gen = torch.Generator().manual_seed(1234)
        logits = torch.randn(batch, seq, self.vocab_size, generator=gen)
        return types.SimpleNamespace(logits=logits)


def _make_cache(num_layers: int = 2, kv_heads: int = 2, seq_len: int = 7,
                head_dim: int = 4) -> DynamicCache:
    cache = DynamicCache()
    gen = torch.Generator().manual_seed(42)
    for layer in range(num_layers):
        k = torch.randn(1, kv_heads, seq_len, head_dim, generator=gen)
        v = torch.randn(1, kv_heads, seq_len, head_dim, generator=gen)
        cache.update(k, v, layer)
    return cache


def _make_prefix(seq_len: int = 7) -> dict:
    return {
        "cache": _make_cache(seq_len=seq_len),
        "system_length": 3,
        "history_length": seq_len - 3,
        "use_gist": False,
    }


def test_logp_leaves_cache_pristine():
    prefix = _make_prefix()
    cache = prefix["cache"]
    before = cache.get_seq_length()
    model = _StubModel()
    out = _prefix_continuation_logp(
        model, prefix, prompt_ids=[5, 6, 7], continuation_ids=[8, 9], attn_impl="eager"
    )
    after = cache.get_seq_length()
    assert before == after, f"cache length changed: {before} -> {after}"
    assert model.seen_cache_length == before, (
        f"scoring forward saw cache length {model.seen_cache_length}, expected {before}"
    )
    assert out is not None


def test_appended_cache_changes_what_logp_sees():
    """Documents why call order matters: if generation has appended to the
    cache in place (the round-1 bug), the scoring forward observes the grown
    cache. The fixed call site scores before generation."""
    prefix = _make_prefix()
    cache = prefix["cache"]
    before = cache.get_seq_length()
    # Simulate a generation pass appending prompt+generated KV in place.
    gen = torch.Generator().manual_seed(7)
    for layer in range(2):
        k = torch.randn(1, 2, 5, 4, generator=gen)
        v = torch.randn(1, 2, 5, 4, generator=gen)
        cache.update(k, v, layer)
    model = _StubModel()
    _prefix_continuation_logp(
        model, prefix, prompt_ids=[5, 6, 7], continuation_ids=[8, 9], attn_impl="eager"
    )
    assert model.seen_cache_length == before + 5


def main() -> int:
    test_logp_leaves_cache_pristine()
    print("PASS test_logp_leaves_cache_pristine")
    test_appended_cache_changes_what_logp_sees()
    print("PASS test_appended_cache_changes_what_logp_sees")
    return 0


if __name__ == "__main__":
    sys.exit(main())
