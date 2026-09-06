"""Pure-part tests for agent/t34_contextcite_bind.py (Binding itself is server-only)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t34_contextcite_bind as B  # noqa: E402


def test_blocks_from_v_is_ascending_restore_set():
    assert B.blocks_from_v((0, 0, 0)) == []
    assert B.blocks_from_v((1, 0, 1, 1)) == [0, 2, 3]
    assert B.blocks_from_v([1.0, 0.0]) == [0]


def test_scored_slice_payload_then_fallback_then_full():
    spans = {"payload_first": 3, "payload_last": 9}
    assert B.scored_slice(spans, 12) == (3, 9, "payload")
    assert B.scored_slice({"payload_first": None, "payload_last": None}, 12) == (0, 11, "full_fallback")
    assert B.scored_slice({"payload_first": 5, "payload_last": 40}, 12) == (0, 11, "full_fallback")
    assert B.scored_slice(spans, 12, mode="full") == (0, 11, "full")
    with pytest.raises(ValueError):
        B.scored_slice(spans, 0)
    with pytest.raises(ValueError):
        B.scored_slice(spans, 12, mode="weird")


def test_neutral_span_ids_is_equal_length_deterministic_and_block_specific():
    corpus = list(range(100, 137))          # 37-token ring
    a = B.neutral_span_ids(corpus, "q:1", 0, 50)
    b = B.neutral_span_ids(corpus, "q:1", 0, 50)
    c = B.neutral_span_ids(corpus, "q:1", 1, 50)
    assert a == b and len(a) == 50 and set(a) <= set(corpus)
    assert a != c                            # per-block offset, not the plan's per-qid one
    # the ring rule is the sham plan's own primitive, keyed by "qid:k"
    from d_sham_plan import corpus_offset, ring_slice
    assert a == ring_slice(corpus, corpus_offset(B.SHAM_SEED, "q:1:0", 37), 50)
    with pytest.raises(ValueError):
        B.neutral_span_ids([], "q:1", 0, 3)


def test_env_config_defaults_and_overrides():
    cfg = B.env_config({})
    assert cfg["attn_impl"] == "eager" and cfg["device_type"] == "npu" and cfg["ratio"] == 8
    assert cfg["corpus"].endswith("d_neutral_corpus.txt")
    assert (cfg["max_doc_length"], cfg["max_doc_num"], cfg["max_new_tokens"]) == (768, 16, 128)
    cfg2 = B.env_config({"T34_CC_RATIO": "4", "T34_CC_SPAN": "full", "T34_CC_ROOT": "/w"})
    assert cfg2["ratio"] == 4 and cfg2["span"] == "full" and cfg2["root"] == "/w"


def test_seed_matches_the_frozen_sham_plan_seed():
    import d_sham_plan
    assert B.SHAM_SEED == d_sham_plan.SEED


def test_factories_are_loadable_by_the_contextcite_runner():
    """t34_contextcite._load_score_fn resolves 'pkg.mod:factory'; the names must exist."""
    assert callable(B.factory) and callable(B.sham_factory)
    import t34_contextcite as CC
    assert hasattr(CC, "_load_score_fn")
