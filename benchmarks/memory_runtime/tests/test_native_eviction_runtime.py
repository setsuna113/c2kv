"""CPU contracts for full-system native history-boundary reselection."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from benchmarks.memory_runtime.native_eviction_runtime import (
    NativeEvictionRunner,
    plan_native_boundary_input,
    select_boundary_history_indices,
)
from benchmarks.memory_runtime import native_eviction_runtime


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tools=None, add_generation_prompt=False, **_kwargs
    ):
        text = "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>" if tools else ""
        for message in messages:
            text += "<" + message["role"] + ">" + json.dumps(
                message, sort_keys=True
            ) + "</end>"
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(character) for character in text]

    def decode(self, ids, **_kwargs):
        return "".join(chr(value) for value in ids)


def _tool_call(call_id="call-1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"key":"alpha"}'},
    }


def test_boundary_matches_s0_common_accounting_and_charges_protected_history():
    payload = {
        "session_id": "s",
        "decision_key": "d0",
        "tools": [],
        "messages": [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Look up alpha."},
            {"role": "assistant", "content": None, "tool_calls": [_tool_call()]},
            {"role": "tool", "tool_call_id": "call-1", "content": "alpha=7"},
            {"role": "assistant", "content": "The lookup completed."},
        ],
    }

    boundary = plan_native_boundary_input(Tokenizer(), payload, recent_window=8)

    assert boundary.source_cutoff == 3
    assert boundary.common_source_indices == (0, 3, 4)
    assert len(boundary.history_indices) == (
        len(boundary.full_ids) - boundary.common_baseline_tokens
    )
    # The latest user and the assistant half of the cross-cutoff tool event are
    # protected, but their early-source tokens still consume the history budget.
    assert boundary.protected_history_indices
    assert set(boundary.protected_history_indices) <= set(boundary.history_indices)
    assert set(boundary.query_indices) <= set(boundary.history_indices)
    assert max(boundary.query_indices) < boundary.history_boundary
    assert min(boundary.current_suffix_indices) == boundary.history_boundary
    assert not set(boundary.current_suffix_indices) & set(boundary.history_indices)


def test_boundary_selection_obeys_budget_and_keeps_nonfree_mandatory_tokens():
    history = tuple(range(10, 20))
    mandatory = (11, 18, 19)
    selected = select_boundary_history_indices(
        [0.1, 0.9, 0.2, 0.3, 0.8, 0.4, 0.7, 0.5, 0.0, 0.0],
        history_indices=history,
        mandatory_indices=mandatory,
        history_budget_tokens=5,
        method="snapkv_style",
    )

    assert len(selected) == 5
    assert tuple(sorted(selected)) == selected
    assert set(mandatory) <= set(selected)
    assert set(selected) <= set(history)

    with pytest.raises(ValueError, match="protected current/recent history"):
        select_boundary_history_indices(
            [0.0] * len(history),
            history_indices=history,
            mandatory_indices=mandatory,
            history_budget_tokens=2,
            method="snapkv_style",
        )


class FakeGenerator:
    decode_strategy = "incremental"
    prefill_chunk_size = 16

    def __init__(self):
        config = SimpleNamespace(num_hidden_layers=2, num_key_value_heads=2)
        self.runtime = SimpleNamespace(base_model=SimpleNamespace(config=config))
        self.generate_calls = 0
        self.closed = 0
        self.seen_memory = None

    def kv_bytes_per_token(self):
        return 4

    def _model_context_length(self):
        return 100000

    def close_session(self):
        self.closed += 1

    def generate(self, memory, *, ratio, max_new_tokens):
        self.generate_calls += 1
        self.seen_memory = memory
        assert ratio == 4 and max_new_tokens == 2
        return SimpleNamespace(
            token_ids=(ord("O"), ord("K")),
            finish_reason="length",
            token_logprobs=(-0.1, -0.2),
            stats={
                "eos_token_ids": [],
                "system_prefill_calls": 1,
                "system_prefill_tokens": len(memory.system_input_ids),
                "raw_prefill_forward_calls": 2,
                "raw_prefill_input_tokens": len(memory.workspace_input_ids),
                "one_token_decode_forward_calls": 1,
                "one_token_decode_input_tokens": 1,
                "resident_kv_tokens_final": (
                    len(memory.system_input_ids + memory.workspace_input_ids) + 1
                ),
                "resident_kv_logical_bytes_final": (
                    len(memory.system_input_ids + memory.workspace_input_ids) + 1
                ) * self.kv_bytes_per_token(),
            },
        )


def test_b500_default_budget_is_exactly_768_history_tokens():
    generator = FakeGenerator()
    generator.kv_bytes_per_token = lambda: 147456
    generator.prefill_chunk_size = 256

    runner = NativeEvictionRunner(generator, Tokenizer(), method="snapkv_style")

    assert runner.history_budget_bytes == 113246208
    assert runner.kv_bytes_per_token == 147456
    assert runner.history_budget_tokens == 768
    assert runner.history_budget_remainder_bytes == 0


def test_runner_full_identity_path_is_api_compatible_and_idempotent():
    generator = FakeGenerator()
    runner = NativeEvictionRunner(
        generator,
        Tokenizer(),
        method="snapkv_style",
        history_budget_bytes=100000,
        max_new_tokens=2,
        max_generation_calls=1,
        max_sequence_tokens=100000,
        prefill_chunk_size=16,
    )
    payload = {
        "session_id": "s-full",
        "decision_key": "d0",
        "tools": [],
        "messages": [
            {"role": "system", "content": "Answer exactly."},
            {"role": "user", "content": "Say OK."},
        ],
    }

    first = runner.run(copy.deepcopy(payload))
    second = runner.run(copy.deepcopy(payload))

    assert first == second
    assert first["status"] == "ok"
    assert first["response"]["content"] == "OK"
    assert first["generation_usage_total"]["completion_tokens"] == 2
    assert first["generation_usage_total"]["prompt_tokens"] == len(
        generator.seen_memory.system_input_ids + generator.seen_memory.workspace_input_ids
    )
    assert first["eviction"]["stats"]["full_identity_generator_path"] is True
    assert first["eviction"]["stats"]["full_history_prefill_tokens"] == first["boundary"]["history_boundary"]
    assert first["eviction"]["stats"]["prefill_input_tokens"] == len(
        generator.seen_memory.system_input_ids + generator.seen_memory.workspace_input_ids
    )
    assert first["post_draft_exact_recovery_applied"] is False
    assert type(first["decision_runtime_seconds"]) is float
    assert runner.generation_calls == generator.generate_calls == 1


def test_runner_rejects_decode_or_prefill_contract_drift():
    generator = FakeGenerator()
    generator.decode_strategy = "full_recompute"
    with pytest.raises(ValueError, match="incremental decode"):
        NativeEvictionRunner(
            generator, Tokenizer(), method="snapkv_style", prefill_chunk_size=16
        )

    generator.decode_strategy = "incremental"
    with pytest.raises(ValueError, match="prefill_chunk_size"):
        NativeEvictionRunner(
            generator, Tokenizer(), method="snapkv_style", prefill_chunk_size=32
        )


def test_boundary_rejects_non_prefix_stable_tokenizer_without_guessing_offsets():
    class UnstableTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            ids = super().apply_chat_template(messages, **kwargs)
            if len(messages) > 1 and not kwargs.get("add_generation_prompt", False):
                ids[0] += 1
            return ids

    payload = {
        "session_id": "unstable",
        "decision_key": "d0",
        "messages": [
            {"role": "system", "content": "System."},
            {"role": "user", "content": "Old."},
            {"role": "assistant", "content": "Done."},
            {"role": "user", "content": "Current."},
        ],
        "tools": [],
    }
    with pytest.raises(ValueError, match="prefix"):
        plan_native_boundary_input(UnstableTokenizer(), payload)


def test_boundary_uses_exact_lcp_inside_consecutive_tool_response_group():
    class GroupedToolTokenizer(Tokenizer):
        def apply_chat_template(
            self, messages, *, tools=None, add_generation_prompt=False, **_kwargs
        ):
            text = (
                "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>"
                if tools
                else ""
            )
            for index, message in enumerate(messages):
                if message["role"] != "tool":
                    text += "<" + message["role"] + ">" + json.dumps(
                        message, sort_keys=True
                    ) + "</end>"
                    continue
                if index == 0 or messages[index - 1]["role"] != "tool":
                    text += "<user>"
                text += "<tool_response>" + message["content"] + "</tool_response>"
                if index + 1 == len(messages) or messages[index + 1]["role"] != "tool":
                    text += "</end>"
            if add_generation_prompt:
                text += "<assistant>"
            return [ord(character) for character in text]

    payload = {
        "session_id": "grouped-tools",
        "decision_key": "d0",
        "messages": [
            {"role": "user", "content": "Inspect both values."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("call-a"), _tool_call("call-b")],
            },
            {"role": "tool", "tool_call_id": "call-a", "content": "alpha=7"},
            {"role": "tool", "tool_call_id": "call-b", "content": "beta=9"},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }

    boundary = plan_native_boundary_input(GroupedToolTokenizer(), payload)

    assert boundary.source_cutoff == 2
    assert boundary.common_source_indices == (2, 3)
    assert len(boundary.history_indices) == (
        len(boundary.full_ids) - boundary.common_baseline_tokens
    )
    assert tuple(boundary.current_suffix_indices) == tuple(
        range(boundary.history_boundary, len(boundary.full_ids))
    )


def test_explicit_history_query_positions_match_dense_causal_attention():
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(20260913)
    query = torch.randn(1, 4, 2, 3, generator=generator)
    keys = torch.randn(1, 2, 7, 3, generator=generator)
    query_positions = torch.tensor([2, 5])
    scaling = 3**-0.5

    actual = native_eviction_runtime._attention_scores_for_positions(
        torch,
        query,
        keys,
        query_positions=query_positions,
        scaling=scaling,
    )

    repeated = (
        keys[:, :, None, :, :]
        .expand(1, 2, 2, 7, 3)
        .reshape(1, 4, 7, 3)
    )
    logits = torch.matmul(query, repeated.transpose(-2, -1)).float() * scaling
    allowed = torch.arange(7).unsqueeze(0) <= query_positions.unsqueeze(1)
    logits.masked_fill_(~allowed[None, None, :, :], -torch.inf)
    expected = torch.softmax(logits, dim=-1).reshape(1, 2, 2, 2, 7).mean(
        dim=(0, 2, 3)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_runtime_gather_keeps_head_specific_common_and_history_coordinates():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    keys = torch.tensor(
        [[
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]],
        ]]
    )
    cache = transformers.DynamicCache([(keys, keys + 100.0)])
    selected = native_eviction_runtime._gather_prefix_cache(
        torch,
        transformers.DynamicCache,
        cache,
        [[(0, 2, 5), (0, 3, 5)]],
        config=None,
    )

    assert selected.layers[0].keys[0, 0, :, 0].tolist() == [0.0, 2.0, 5.0]
    assert selected.layers[0].keys[0, 1, :, 0].tolist() == [10.0, 13.0, 15.0]
