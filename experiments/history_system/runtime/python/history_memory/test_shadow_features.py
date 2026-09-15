"""Focused CPU tests for shadow-only event-native generation features."""

from __future__ import annotations

import copy
import json
import math

import pytest

torch = pytest.importorskip("torch")

from history_memory.evidence import EVIDENCE_VERSION
from history_memory.inference import EventNativeGenerator
from history_memory.packing import PACKING_VERSION, RAW_LAYOUT_PROFILE, MemoryView, PackedMemory
from history_memory.runtime import HistoryMemoryModel
from history_memory.shadow_features import (
    SHADOW_FEATURE_SCHEMA,
    ShadowFeatureConfig,
    distribution_features,
    locate_tool_name_tokens,
)
from models.qwen3 import Qwen3Config, Qwen3ForCausalLM


_TOOL_CALL_PIECES = (
    "<tool_call>",
    '{"name":"',
    "weather",
    '","arguments":{}}',
    "</tool_call>",
)


def _decode_by_length(token_ids) -> str:
    return "".join(_TOOL_CALL_PIECES[: len(token_ids)])


def _decode_protocol_tokens(token_ids) -> str:
    pieces = {
        101: "<tool_call>",
        102: '{"name":"',
        103: "weather",
        104: '","arguments":{}}',
        105: "</tool_call>",
        0: "<|endoftext|>",
    }
    return "".join(pieces[token_id] for token_id in token_ids)


def _tiny_qwen(seed: int = 0) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=48,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=256,
        pad_token_id=None,
        eos_token_id=None,
        gist_type="dynamic-interleave",
        gist_param="qkv",
        gist_residual_type="embed-mean",
        gist_extra_embed_num=1,
        gist_token_id=0,
        attention_dropout=0.0,
        attn_implementation="eager",
        history_memory_training_profile="history-event-base-query-v1",
        history_memory_normal_query="base",
        history_memory_raw_layout=RAW_LAYOUT_PROFILE,
        history_memory_packing_version=PACKING_VERSION,
        history_memory_evidence_version=EVIDENCE_VERSION,
        history_memory_supported_ratios=[4, 8],
    )
    model = Qwen3ForCausalLM(config)
    with torch.no_grad():
        for layer in model.model.layers:
            attention = layer.self_attn
            attention.gist_q_proj.weight.copy_(attention.q_proj.weight)
            attention.gist_k_proj.weight.copy_(attention.k_proj.weight)
            attention.gist_v_proj.weight.copy_(attention.v_proj.weight)
        model.model.gist_embed_tokens.weight.normal_(mean=0.0, std=0.02)
    return model.eval()


def _memory() -> PackedMemory:
    return PackedMemory(
        view=MemoryView(gist_event_ids=(), raw_event_ids=("raw",)),
        system_input_ids=(1, 2, 3),
        workspace_input_ids=(21, 22, 23),
        raw_source_indices=(0,),
        chunks=(),
    )


def _assert_same_generation(left, right) -> None:
    assert left.token_ids == right.token_ids
    assert left.finish_reason == right.finish_reason
    torch.testing.assert_close(
        torch.tensor(left.token_logprobs),
        torch.tensor(right.token_logprobs),
        rtol=0.0,
        atol=0.0,
    )


def test_enabled_config_requires_native_decoder():
    with pytest.raises(ValueError, match="decode_token_ids"):
        ShadowFeatureConfig(enabled=True)


def test_strict_tool_name_locator_uses_continuation_token_indices():
    located = locate_tool_name_tokens(_decode_by_length, [7, 8, 9, 10, 11])

    assert located == {
        "status": "located",
        "reason": None,
        "first_generated_token_index": 2,
        "last_generated_token_index": 2,
    }


def test_protocol_tokens_are_preserved_and_trailing_eos_is_outside_tool_name():
    located = locate_tool_name_tokens(
        _decode_protocol_tokens,
        [101, 102, 103, 104, 105, 0],
    )

    assert located["status"] == "located"
    assert located["first_generated_token_index"] == 2
    assert located["last_generated_token_index"] == 2


def test_malformed_tool_call_is_unavailable_instead_of_safe():
    located = locate_tool_name_tokens(_decode_by_length, [7, 8, 9])

    assert located == {"status": "unavailable", "reason": "unclosed_tool_call"}


def test_distribution_features_match_full_vocab_definition():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0])
    log_probs = torch.log_softmax(logits, dim=-1)

    row = distribution_features(log_probs, selected_token_id=0)

    expected_entropy = float((-(log_probs.exp() * log_probs).sum()).item())
    assert row["top1_token_id"] == 0
    assert row["top2_token_id"] == 1
    assert row["top2_logprob_margin"] == pytest.approx(1.0)
    assert row["full_vocab_entropy_nats"] == pytest.approx(expected_entropy)
    assert row["vocab_size"] == 4


def test_default_generator_has_no_shadow_trace():
    runtime = HistoryMemoryModel(_tiny_qwen(seed=10)).eval()

    result = EventNativeGenerator(runtime).generate(
        _memory(), ratio=4, max_new_tokens=2
    )

    assert "shadow_features" not in result.stats


def test_enabled_shadow_trace_preserves_incremental_generation_and_actions():
    model = _tiny_qwen(seed=11)
    baseline = EventNativeGenerator(
        HistoryMemoryModel(copy.deepcopy(model)).eval(),
        prefill_chunk_size=2,
    ).generate(_memory(), ratio=8, max_new_tokens=5)
    observed = EventNativeGenerator(
        HistoryMemoryModel(copy.deepcopy(model)).eval(),
        prefill_chunk_size=2,
        shadow_feature_config=ShadowFeatureConfig(
            enabled=True,
            decode_token_ids=_decode_by_length,
            prefill_layer=-2,
            memgen_layer=-2,
            model_binding="test-checkpoint",
            tokenizer_binding="test-tokenizer",
        ),
    ).generate(_memory(), ratio=8, max_new_tokens=5)

    _assert_same_generation(baseline, observed)
    trace = observed.stats["shadow_features"]
    assert trace["schema"] == SHADOW_FEATURE_SCHEMA
    assert trace["gold_used"] is False
    assert trace["tool_name"]["status"] == "located"
    first = trace["tool_name"]["first_token"]
    assert first["generated_token_index"] == 2
    assert first["logical_position"] == _memory().workspace_position_start + 3 + 2
    assert first["top1_token_id"] == observed.token_ids[2]
    assert first["top2_logprob_margin"] >= 0.0
    assert math.isfinite(first["full_vocab_entropy_nats"])
    assert trace["signals"] == {
        "first_name_top2_logprob_margin": first["top2_logprob_margin"],
        "first_name_full_vocab_entropy_nats": first["full_vocab_entropy_nats"],
    }
    assert trace["prefill"]["status"] == "captured"
    assert trace["prefill"]["layer"] == 0
    assert len(trace["prefill"]["hidden"]) == 16
    assert trace["memgen"]["status"] == "captured"
    assert trace["memgen"]["layer"] == 0
    assert len(trace["memgen"]["hidden"]) == 16
    json.dumps(trace)


def test_full_recompute_exports_same_scalar_contract_without_hidden_hooks():
    model = _tiny_qwen(seed=12)
    baseline = EventNativeGenerator(
        HistoryMemoryModel(copy.deepcopy(model)).eval(),
        decode_strategy="full_recompute",
    ).generate(_memory(), ratio=4, max_new_tokens=5)
    observed = EventNativeGenerator(
        HistoryMemoryModel(copy.deepcopy(model)).eval(),
        decode_strategy="full_recompute",
        shadow_feature_config=ShadowFeatureConfig(
            enabled=True,
            decode_token_ids=_decode_by_length,
        ),
    ).generate(_memory(), ratio=4, max_new_tokens=5)

    _assert_same_generation(baseline, observed)
    trace = observed.stats["shadow_features"]
    assert trace["tool_name"]["status"] == "located"
    assert trace["signals"]["first_name_top2_logprob_margin"] is not None
    assert trace["signals"]["first_name_full_vocab_entropy_nats"] is not None
    assert trace["prefill"]["status"] == "disabled"
    assert trace["memgen"]["status"] == "disabled"
