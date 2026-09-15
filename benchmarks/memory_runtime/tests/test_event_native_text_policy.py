"""Focused contracts for the event-native Text summary controller."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from benchmarks.memory_runtime.adapter import raw_source_cutoff
from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from benchmarks.memory_runtime.event_native_always import NATIVE_TEXT_S0_MODE
from benchmarks.memory_runtime.event_native_controls import describe_event_native_route
from benchmarks.memory_runtime.event_native_text_policy import (
    EVENT_NATIVE_TEXT_SUMMARY_MODE,
    EventNativeTextSummaryController,
    PreparedEventNativeTextSummary,
)
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.tests.test_event_native_step import (
    Generator,
    Tokenizer as DecodeTokenizer,
)
from benchmarks.memory_runtime.failed_operation import INSTRUCTION
from benchmarks.memory_runtime.policy import PolicyInputError
from benchmarks.memory_runtime.source_needs_runtime import SourceNeedsRuntime
from history_memory.events import EventStore
from history_memory.packing import (
    PackingBudgetError,
    native_ids,
    raw_workspace_messages,
)


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tools=None, add_generation_prompt=False, **_kwargs
    ):
        text = (
            "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>"
            if tools
            else ""
        )
        for message in messages:
            text += (
                "<"
                + message["role"]
                + ">"
                + json.dumps(message, sort_keys=True)
                + "</end>"
            )
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]


class Renderer:
    def __init__(self):
        self.calls = []

    def render(self, source, context):
        self.calls.append((copy.deepcopy(source), copy.deepcopy(context)))
        cutoff = raw_source_cutoff(source)
        records = []
        fragments = []
        for index, message in enumerate(source[:cutoff]):
            if message["role"] == "system":
                continue
            fragment_id = len(fragments)
            tokens = 20 + fragment_id
            fragment = {
                "fragment_id": fragment_id,
                "source_indices": [index],
                "encoder_input_tokens": tokens,
            }
            fragments.append(fragment)
            records.append(
                {
                    "summary_key": f"summary-{fragment_id}",
                    "packing_fragment_id": fragment_id,
                    "source_indices": [index],
                    "encoder_input_tokens": tokens,
                    "source_content_sha256": hashlib.sha256(
                        str(index).encode()
                    ).hexdigest(),
                    "message": {
                        "role": "user",
                        "content": f"Historical summary {fragment_id}.",
                    },
                    "completion_cap": 16,
                    "finish_reason": "stop",
                }
            )
        return {
            "version": "normalized-turn-text-summary-test-v1",
            "records": records,
            "history_packing_fragments": fragments,
            "dropped_docs": 0,
            "lookups": [],
            "producer_calls": len(records),
            "wall_sec": 0.25,
            "source_scope": "preceding observed prefix",
            "coverage_scope": "source fragment accounting only",
        }


def _packing(**overrides):
    result = {
        "ratios": [4, 8],
        "recent_tool_events": 1,
        "max_chunk_tokens": 768,
        "chunk_overlap": 64,
        "max_chunks": 48,
        "max_encoder_tokens": 100_000,
        "max_system_tokens": 20_000,
        "max_workspace_tokens": 50_000,
        "max_target_tokens": 4096,
        "max_sequence_tokens": 60_000,
    }
    result.update(overrides)
    return result


def _policy(*, history_budget=1_000_000, workspace_budget=1_000_000):
    return {
        "mode": "persistent",
        "history_budget_bytes": history_budget,
        "workspace_budget_bytes": workspace_budget,
        "lease_decisions": 0,
        "max_retrieved_events": 2,
        "kv_bytes_per_token": 1,
        "source_commit": POLICY_SOURCE_COMMIT,
        "history_budget_definition": HISTORY_BUDGET_DEFINITION,
        "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
        "current_input_baseline": CURRENT_INPUT_BASELINE,
    }


def _controller(renderer, *, packing=None, policy=None, model_context=None):
    return EventNativeTextSummaryController(
        Tokenizer(),
        packing=packing or _packing(),
        policy=policy or _policy(),
        model_context=model_context,
        run_id="native-text-test",
        summary_renderer=renderer,
    )


def _tool_event(call_id, result, *, name="lookup"):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps({"key": call_id}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result),
        },
    ]


def _messages(*, failed=False):
    result = [
        {"role": "system", "content": "Use the visible records."},
        {"role": "user", "content": "Collect alpha and beta."},
        *_tool_event("alpha", {"value": "ALPHA"}),
        {"role": "assistant", "content": "Alpha recorded."},
        *_tool_event("beta", {"value": "BETA"}),
        {"role": "assistant", "content": "Beta recorded."},
        {"role": "user", "content": "Use the records for the current action."},
    ]
    if failed:
        result += _tool_event(
            "current", {"success": False, "error": "conflict"}, name="update"
        )
    return result


def _payload(messages, *, key="d1"):
    return {
        "session_id": "bfcl/native-text/attempt-0",
        "decision_key": key,
        "messages": copy.deepcopy(messages),
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a record.",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update",
                    "description": "Update a record.",
                    "parameters": {"type": "object"},
                },
            },
        ],
    }


def test_reuses_allocator_and_native_packing_reproduces_order_tokens_and_sources(
    monkeypatch,
):
    renderer = Renderer()
    allocator_results = []
    original_apply = SourceNeedsRuntime.apply

    def observed_apply(self, *args, **kwargs):
        result = original_apply(self, *args, **kwargs)
        allocator_results.append(copy.deepcopy(result))
        return result

    monkeypatch.setattr(SourceNeedsRuntime, "apply", observed_apply)
    messages = _messages()
    original = copy.deepcopy(messages)
    controller = _controller(renderer)
    prepared = controller.prepare(_payload(messages), ratio=4, max_new_tokens=4096)

    assert isinstance(prepared, PreparedEventNativeTextSummary)
    assert len(allocator_results) == len(renderer.calls) == 1
    assert messages == original
    assert prepared.memory.chunks == prepared.eligible_chunks == ()
    assert prepared.memory.view.gist_event_ids == ()

    allocator_messages, allocator_counts = allocator_results[0]
    allocator_meta = allocator_counts["memory_runtime"]
    derived = prepared.metadata["derived_workspace_prefix_messages"]
    rebuilt = list(raw_workspace_messages(
        EventStore.from_messages("bfcl/native-text/attempt-0", messages),
        prepared.memory.view,
    ))
    prefix = 0
    while prefix < len(rebuilt) and rebuilt[prefix]["role"] == "system":
        prefix += 1
    rebuilt[prefix:prefix] = derived
    assert rebuilt == allocator_messages
    assert prepared.memory.system_input_ids + prepared.memory.workspace_input_ids == native_ids(
        Tokenizer(), allocator_messages, tools=tuple(_payload(messages)["tools"]),
        generation=True,
    )
    assert (
        len(prepared.memory.system_input_ids)
        + len(prepared.memory.workspace_input_ids)
        == allocator_meta["total_raw_prompt_tokens"]
        == prepared.metadata["raw_prompt_tokens"]
    )

    store = EventStore.from_messages("bfcl/native-text/attempt-0", messages)
    whole_event_sources = sorted(
        index
        for event_id in prepared.memory.view.raw_event_ids
        for index in store.event(event_id).source_indices
    )
    assert whole_event_sources == prepared.metadata["raw_source_indices"]
    assert whole_event_sources == allocator_meta["selected_source_indices"]
    assert prepared.metadata["native_allocator_receipt"] == {
        **prepared.metadata["native_allocator_receipt"],
        "message_order_reproduced": True,
        "token_ids_reproduced": True,
        "source_indices_reproduced": True,
        "zero_gist_chunks": True,
    }
    assert prepared.metadata["compression_policy"] is None
    assert (
        prepared.metadata["native_allocator_receipt"]["compression_policy"]
        == "always-compress-v1"
    )
    assert prepared.metadata["source_coverage"]["semantic_fidelity"] == "unknown"
    assert prepared.metadata["source_coverage"]["complete_history_coverage"] is None
    assert prepared.metadata["auxiliary_summary_receipt"]["actor_generation"] is False
    assert prepared.metadata["actual_gist_tokens"] == 0
    assert prepared.metadata["actual_history_bytes"] <= min(
        _policy()["history_budget_bytes"], _policy()["workspace_budget_bytes"]
    )
    assert renderer.calls[0][1] == {
        "session_id": "bfcl/native-text/attempt-0",
        "task_id": "native-text",
        "attempt_id": 0,
        "decision_id": "d1",
        "run_id": "native-text-test",
        "parent_request_id": '["bfcl/native-text/attempt-0","d1"]',
    }


def test_summary_and_failed_cue_are_derived_workspace_with_no_false_source_ids(
    monkeypatch,
):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.failed_operation.PROMPT_CAP", 2_000
    )
    renderer = Renderer()
    messages = _messages(failed=True)
    controller = _controller(renderer)
    prepared = controller.prepare(_payload(messages), ratio=8, max_new_tokens=32)
    metadata = prepared.metadata
    derived = metadata["derived_workspace_prefix_messages"]

    assert metadata["derived_summary_message_count"] > 0
    assert metadata["derived_failed_operation_cue_count"] == 1
    assert derived[-1]["content"].startswith(INSTRUCTION)
    assert metadata["derived_workspace_source_indices"] == []
    assert metadata["summary_records"]
    assert all(record["source_indices"] for record in metadata["summary_records"])
    assert all(index < len(messages) for index in metadata["raw_source_indices"])
    assert not any(message["content"].startswith("Historical summary")
                   for message in messages if isinstance(message.get("content"), str))
    assert metadata["failed_operation_cue"]["status"] == "admitted"
    assert metadata["source_coverage"]["semantic_fidelity"] == "unknown"
    assert metadata["eligible_extraction"]["backend_execution_required"] is False


def test_repeated_decision_is_cached_and_reconsider_is_single_generation_no_op():
    renderer = Renderer()
    controller = _controller(renderer)
    payload = _payload(_messages())
    prepared = controller.prepare(payload, ratio=4, max_new_tokens=64)
    assert controller.prepare(
        copy.deepcopy(payload), ratio=4, max_new_tokens=64
    ) is prepared
    assert len(renderer.calls) == 1

    result = controller.reconsider(
        prepared, [], draft_text="done", parse_error=None
    )
    assert result["regenerate"] is False
    assert result["memory"] is prepared.memory
    assert result["decision"]["reason"] == "native_text_summary_single_generation"
    assert result["decision"]["upgrade_count"] == 0
    assert result["metadata"]["post_draft_exact_recovery_applied"] is False
    assert controller.reconsider(
        prepared, [], draft_text="done", parse_error=None
    ) == result
    with pytest.raises(PolicyInputError, match="second different draft"):
        controller.reconsider(
            prepared, [], draft_text="changed", parse_error=None
        )


def test_decision_runner_receives_empty_extraction_plan_and_generates_once(tmp_path):
    renderer = Renderer()
    controller = _controller(renderer)
    payload = _payload(_messages())
    path = tmp_path / "attempts.jsonl"

    class CapturingGenerator(Generator):
        def generate(self, memory, **kwargs):
            assert kwargs["compression_chunks"] == ()
            assert memory.chunks == ()
            return super().generate(memory, **kwargs)

    runner = EventNativeDecisionRunner(
        controller,
        CapturingGenerator(path, ["Done."]),
        DecodeTokenizer(),
        ratio=4,
        max_new_tokens=64,
        max_generation_calls=1,
        journal=AttemptJournal(path),
    )
    result = runner.run(payload)

    assert result["status"] == "ok"
    assert result["generation_attempts"] == 1
    assert result["exact_recovery"]["reason"] == (
        "native_text_summary_single_generation"
    )
    assert len(renderer.calls) == 1


@pytest.mark.parametrize("limit_kind", ["physical", "logical"])
def test_final_native_limits_reserve_target_tokens(limit_kind):
    baseline = _controller(Renderer()).prepare(
        _payload(_messages()), ratio=4, max_new_tokens=4096
    )
    prompt_tokens = (
        len(baseline.memory.system_input_ids)
        + len(baseline.memory.workspace_input_ids)
    )
    if limit_kind == "physical":
        packing = _packing(max_sequence_tokens=prompt_tokens + 4095)
        model_context = None
        expected = "Physical sequence"
    else:
        packing = _packing()
        model_context = prompt_tokens + 4095
        expected = "Logical sequence"
    with pytest.raises(PackingBudgetError, match=expected):
        _controller(
            Renderer(), packing=packing, model_context=model_context
        ).prepare(_payload(_messages()), ratio=4, max_new_tokens=4096)


def test_final_native_workspace_limit_is_not_silently_truncated():
    baseline = _controller(Renderer()).prepare(
        _payload(_messages()), ratio=4, max_new_tokens=32
    )
    workspace_tokens = len(baseline.memory.workspace_input_ids)
    with pytest.raises(PackingBudgetError, match="workspace needs"):
        _controller(
            Renderer(),
            packing=_packing(max_workspace_tokens=workspace_tokens - 1),
        ).prepare(_payload(_messages()), ratio=4, max_new_tokens=32)


def test_legacy_allocator_uses_min_budget_and_keeps_first_plus_tail_summary_fill():
    prepared = _controller(
        Renderer(),
        policy=_policy(history_budget=2_000, workspace_budget=700),
    ).prepare(_payload(_messages()), ratio=4, max_new_tokens=32)
    metadata = prepared.metadata

    assert metadata["shared_allocation_budget_bytes"] == 700
    assert metadata["actual_history_bytes"] <= 700
    assert [
        record["packing_fragment_id"] for record in metadata["representation_refs"]
    ] == [0, 2, 3, 4, 5, 6]
    assert len(metadata["source_needs"]["requested_event_ids"]) <= 2
    assert metadata["latest_complete_tool_protection"]["policy"] == "budgeted"


def test_callable_renderer_and_no_history_skip_auxiliary_generation():
    calls = []

    def renderer(source, context):
        calls.append((source, context))
        raise AssertionError("No eligible history must not call the renderer")

    controller = _controller(renderer)
    prepared = controller.prepare(
        _payload([{"role": "user", "content": "Start."}]),
        ratio=4,
        max_new_tokens=16,
    )

    assert calls == []
    assert prepared.metadata["no_eligible_history"] is True
    assert prepared.metadata["derived_workspace_prefix_messages"] == []
    assert prepared.metadata["auxiliary_summary_receipt"]["producer_calls"] == 0
    assert prepared.metadata["source_coverage"]["semantic_fidelity"] == "unknown"
    assert prepared.memory.raw_source_indices == (0,)
    assert prepared.metadata["mode"] == EVENT_NATIVE_TEXT_SUMMARY_MODE
    assert prepared.metadata["route"] == describe_event_native_route(
        NATIVE_TEXT_S0_MODE
    )
