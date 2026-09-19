from __future__ import annotations

import pytest
from types import SimpleNamespace

pytest.importorskip("torch")

from benchmarks.tool_definition.evaluate import (_accumulate, _finalize_summaries,
                                                 _record_result, _validate_t0_generation_contract)


class Tokenizer:
    def decode(self, _tokens, **_kwargs):
        return "<tool_call>{\"name\":\"search\",\"arguments\":{}}</tool_call>"


def test_tool_compression_factor_uses_full_over_resident():
    record = {
        "decision_id": "recorded", "source": "fixture", "ratio": 8,
        "k": 3, "seed": 42, "prompt_sha256": "abc",
        "gold_tool_calls": [{"name": "search", "arguments": {}}],
        "native_indices": [], "base_prompt_tokens_without_tool_protocol": 50,
    }
    row = _record_result(
        record, method="c2kv", layout="uniform", tokens=(1,),
        finish_reason="eos", tokenizer=Tokenizer(), per_layer_kv=(75, 75),
        allowance=80, replayed=False, full_resident_tokens=150,
    )
    assert row["R_tool_numerator_tokens"] == 200
    assert row["R_tool_denominator_tokens"] == 50
    assert row["R_tool"] == 4
    assert row["tool_kv_retention_fraction"] == 0.25
    assert row["outcome"]["strict_ordered_call_correct"]
    groups = {}
    _accumulate(groups, row)
    summary, = _finalize_summaries(groups)
    assert summary["R_tool"] == 4
    assert summary["gold_tool_rows"] == 1
    assert summary["gold_no_call_rows"] == 0


def test_t0_generation_accepts_next_compression_profile_only():
    config = SimpleNamespace(
        gist_type="dynamic-interleave", gist_param="qkv",
        gist_residual_type="embed-mean", history_memory_normal_query="base",
        history_memory_training_profile="next-compression-base-query-v1",
        history_memory_variant="T0", history_memory_compression_domain="tool",
        history_memory_render_profile="next-compression-tool-explicit-protocol-v2",
        history_memory_supported_ratios=[8, 12],
    )
    _validate_t0_generation_contract(config, 8)
    config.history_memory_variant = "H0"
    with pytest.raises(ValueError, match="variant"):
        _validate_t0_generation_contract(config, 8)
