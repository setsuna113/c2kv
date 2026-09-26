"""Focused contracts for the native S0 controller."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from benchmarks.memory_runtime.always_compress import CapacityInfeasible
from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from benchmarks.memory_runtime.event_native_s0_policy import (
    EventNativeS0Controller,
    PreparedEventNativeS0,
)
from benchmarks.memory_runtime.policy import PolicyInputError
from history_memory.events import EventStore


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tools=None, add_generation_prompt=False, **kwargs
    ):
        text = "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>" if tools else ""
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


def _packing():
    return {
        "ratios": [4],
        "recent_tool_events": 1,
        "max_chunk_tokens": 768,
        "chunk_overlap": 64,
        "max_chunks": 48,
        "max_encoder_tokens": 100_000,
        "max_system_tokens": 20_000,
        "max_workspace_tokens": 50_000,
        "max_target_tokens": 32,
        "max_sequence_tokens": 100_000,
    }


def _policy(budget=1_000_000):
    return {
        "mode": "persistent",
        "history_budget_bytes": budget,
        "workspace_budget_bytes": budget,
        "lease_decisions": 0,
        "max_retrieved_events": 2,
        "kv_bytes_per_token": 1,
        "source_commit": POLICY_SOURCE_COMMIT,
        "history_budget_definition": HISTORY_BUDGET_DEFINITION,
        "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
        "current_input_baseline": CURRENT_INPUT_BASELINE,
    }


def _controller(*, budget=1_000_000):
    return EventNativeS0Controller(
        Tokenizer(), packing=_packing(), policy=_policy(budget)
    )


def _payload(messages, key="d1", tools=None):
    return {
        "session_id": "native-s0/session",
        "decision_key": key,
        "messages": copy.deepcopy(messages),
        "tools": copy.deepcopy(tools or []),
    }


def _tool_schema(name, description=""):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_event(call_id, name, arguments, result):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)},
    ]


def test_api_keeps_cross_cutoff_complete_event_raw_and_extracts_it() -> None:
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Find alpha."},
        *_tool_event("old", "lookup", {"query": "alpha"}, {"value": "alpha"}),
        {"role": "assistant", "content": "I will verify it."},
        {"role": "user", "content": "Verify alpha now."},
        *_tool_event("current", "verify", {"value": "alpha"}, {"ok": True}),
    ]
    original = copy.deepcopy(messages)
    prepared = _controller().prepare(_payload(messages), ratio=4, max_new_tokens=8)
    store = EventStore.from_messages("native-s0/session", messages)
    current_event = next(
        event for event in store.events if "current" in event.tool_call_ids
    )

    assert isinstance(prepared, PreparedEventNativeS0)
    assert messages == original
    assert prepared.metadata["raw_source_cutoff"] == 7
    assert set(current_event.source_indices) <= set(prepared.memory.raw_source_indices)
    assert current_event.event_id in prepared.memory.view.raw_event_ids
    assert current_event.event_id in prepared.memory.view.gist_event_ids
    assert current_event.event_id in {
        chunk.event_id for chunk in prepared.eligible_chunks
    }
    assert prepared.metadata["source_coverage"]["eligible_source_indices"] == [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    assert prepared.metadata["source_coverage"][
        "whole_event_extra_source_indices"
    ] == [7]
    assert prepared.metadata["lease_decisions"] == 0
    assert prepared.metadata["route"]["max_generations_per_decision"] == 1

    controller = _controller()
    owned = controller.prepare(_payload(messages), ratio=4, max_new_tokens=8)
    result = controller.reconsider(
        owned, [], draft_text="done", parse_error=None
    )
    assert result["regenerate"] is False
    assert result["memory"] is owned.memory
    assert result["decision"]["status"] == "no_op"
    assert result["decision"]["reason"] == "native_s0_single_generation"


def test_budgeted_latest_drop_recomputes_lexical_index() -> None:
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Find the archive."},
        *_tool_event("small", "lookup", {"query": "archive"}, {"value": "archive"}),
        {"role": "assistant", "content": "I found the archive."},
        *_tool_event(
            "large",
            "inspect",
            {"query": "work-marker"},
            {"payload": "X" * 5_000},
        ),
        {"role": "assistant", "content": "I recorded work-marker."},
        {"role": "user", "content": "Use work-marker for the current answer."},
    ]
    prepared = _controller(budget=1_000).prepare(
        _payload(messages), ratio=4, max_new_tokens=8
    )
    store = EventStore.from_messages("native-s0/session", messages)
    latest = next(event for event in store.events if "large" in event.tool_call_ids)
    receipt = prepared.metadata["latest_complete_tool_protection"]

    assert receipt["status"] == "skipped"
    assert receipt["reason"] == "protected_native_over_budget"
    assert receipt["candidate_required_native_bytes"] > receipt["budget_bytes"]
    assert receipt["selector_recomputed_after_skip"] is True
    assert receipt["indexed_after_skip"] is True
    assert latest.event_id in prepared.metadata["source_needs"]["candidate_event_ids"]
    assert latest.event_id in prepared.metadata["source_needs"]["requested_event_ids"]
    assert latest.event_id not in prepared.metadata["protected_event_ids"]
    assert latest.event_id not in prepared.memory.view.raw_event_ids
    assert prepared.metadata["actual_history_bytes"] <= 1_000


def test_lexical_raw_overlap_and_one_newest_spare_event(monkeypatch) -> None:
    messages = [{"role": "user", "content": "Collect values."}]
    messages += _tool_event("a", "lookup", {"query": "alpha"}, {"value": "alpha"})
    messages += [{"role": "assistant", "content": "Alpha recorded."}]
    messages += _tool_event("b", "lookup", {"query": "beta"}, {"value": "beta"})
    messages += [{"role": "assistant", "content": "Beta recorded."}]
    messages += _tool_event("c", "lookup", {"query": "gamma"}, {"value": "gamma"})
    messages += [
        {"role": "assistant", "content": "Gamma recorded."},
        {"role": "user", "content": "Use alpha for the answer."},
    ]
    store = EventStore.from_messages("native-s0/session", messages)
    events = {
        call_id: next(event for event in store.events if call_id in event.tool_call_ids)
        for call_id in ("a", "b", "c")
    }
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.lexical_source_ids",
        lambda store, context, max_sources=2: (events["a"].event_id,),
    )
    prepared = _controller().prepare(_payload(messages), ratio=4, max_new_tokens=8)

    assert prepared.metadata["retrieved_event_ids"] == [events["a"].event_id]
    assert events["a"].event_id in prepared.memory.view.raw_event_ids
    assert events["a"].event_id in prepared.memory.view.gist_event_ids
    reserve = prepared.metadata["raw_reserve"]
    assert reserve["status"] == "extra_event_admitted"
    assert reserve["admitted_event_id"] == events["b"].event_id
    assert prepared.metadata["recency_selected_event_ids"] == [events["b"].event_id]
    assert events["c"].event_id in prepared.metadata["protected_event_ids"]
    assert prepared.memory.view.evidence_event_ids == ()


def test_failed_operation_cue_is_derived_and_source_bound(monkeypatch) -> None:
    monkeypatch.setattr(
        "benchmarks.memory_runtime.event_native_s0_policy.PROMPT_CAP", 2_000
    )
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Complete account-7."},
        *_tool_event(
            "failed",
            "update_account",
            {"account": "account-7"},
            {"success": False, "error": "conflict"},
        ),
    ]
    prepared = _controller().prepare(_payload(messages), ratio=4, max_new_tokens=8)
    cue = prepared.metadata["failed_operation_cue"]

    assert cue["status"] == "admitted"
    assert cue["selected_record"]["result_source_index"] == 3
    assert cue["extra_bytes"] > 0
    assert len(prepared.metadata["derived_workspace_prefix_messages"]) == 1
    assert prepared.metadata["derived_workspace_source_indices"] == []
    assert prepared.memory.raw_source_indices == (0, 1, 2, 3)
    assert cue["source_indices_are_not_raw_copies"] is True


def test_minimum_whole_event_gist_obeys_exact_b0_boundary() -> None:
    messages = [
        {"role": "system", "content": "Use the record."},
        {"role": "assistant", "content": "historical value " + "x" * 80},
        {"role": "user", "content": "Answer now."},
    ]
    roomy = _controller().prepare(_payload(messages), ratio=4, max_new_tokens=8)
    boundary = roomy.metadata["gist_reservation"]["minimum_active_history_bytes"]
    assert boundary > 0

    exact = _controller(budget=boundary).prepare(
        _payload(messages), ratio=4, max_new_tokens=8
    )
    assert exact.metadata["min_gist_reservation_met"] is True
    assert exact.metadata["actual_history_bytes"] == boundary
    with pytest.raises(CapacityInfeasible):
        _controller(budget=boundary - 1).prepare(
            _payload(messages), ratio=4, max_new_tokens=8
        )


def test_no_eligible_history_has_empty_pre_extraction_and_idempotent_api() -> None:
    messages = [{"role": "user", "content": "Start."}]
    controller = _controller()
    first = controller.prepare(_payload(messages), ratio=4, max_new_tokens=8)
    second = controller.prepare(_payload(messages), ratio=4, max_new_tokens=8)

    assert second is first
    assert first.eligible_chunks == ()
    assert first.memory.chunks == ()
    assert first.metadata["no_eligible_history"] is True
    assert first.metadata["gist_reservation"]["required"] is False
    changed = _payload(
        [
            {"role": "user", "content": "Start."},
            {"role": "assistant", "content": "Changed."},
        ]
    )
    with pytest.raises(PolicyInputError, match="reused with different input"):
        controller.prepare(changed, ratio=4, max_new_tokens=8)


def test_append_only_tool_reveal_preserves_session_and_recomputes_prefix() -> None:
    controller = _controller()
    lookup = _tool_schema("lookup")
    set_budget = _tool_schema("set_budget_limit")
    first_messages = [{"role": "user", "content": "Inspect the budget."}]
    first = controller.prepare(
        _payload(first_messages, tools=[lookup]), ratio=4, max_new_tokens=8
    )
    controller.reconsider(first, [], draft_text="The setter is unavailable.")

    second_messages = [
        *first_messages,
        {"role": "assistant", "content": "The setter is unavailable."},
        {"role": "user", "content": "A new tool is now available."},
    ]
    second = controller.prepare(
        _payload(second_messages, key="d2", tools=[lookup, set_budget]),
        ratio=4,
        max_new_tokens=8,
    )
    result = controller.reconsider(
        second, [], draft_text="I can set the budget now."
    )

    assert second.metadata["decision_index"] == 2
    assert len(second.memory.system_input_ids) > len(first.memory.system_input_ids)
    assert controller._sessions["native-s0/session"].message_json[:1] == (
        json.dumps(first_messages[0], ensure_ascii=False, allow_nan=False),
    )
    assert result["decision"]["status"] == "no_op"

    with pytest.raises(PolicyInputError, match="reused with different input"):
        controller.prepare(
            _payload(
                second_messages,
                key="d2",
                tools=[lookup, set_budget, _tool_schema("newer")],
            ),
            ratio=4,
            max_new_tokens=8,
        )


@pytest.mark.parametrize(
    "changed_tools",
    [
        [_tool_schema("lookup")],
        [_tool_schema("set_budget_limit"), _tool_schema("lookup")],
        [
            _tool_schema("lookup", "rewritten"),
            _tool_schema("set_budget_limit"),
        ],
    ],
)
def test_tool_catalog_rejects_removal_reordering_and_rewrite(changed_tools) -> None:
    controller = _controller()
    initial = [_tool_schema("lookup"), _tool_schema("set_budget_limit")]
    controller.prepare(
        _payload([{"role": "user", "content": "Start."}], tools=initial),
        ratio=4,
        max_new_tokens=8,
    )
    with pytest.raises(PolicyInputError, match="only append-only"):
        controller.prepare(
            _payload(
                [
                    {"role": "user", "content": "Start."},
                    {"role": "assistant", "content": "Continuing."},
                    {"role": "user", "content": "Next."},
                ],
                key="d2",
                tools=changed_tools,
            ),
            ratio=4,
            max_new_tokens=8,
        )


def test_tool_catalog_extension_still_rejects_duplicate_names() -> None:
    controller = _controller()
    lookup = _tool_schema("lookup")
    with pytest.raises(PolicyInputError, match="unique tool names"):
        controller.prepare(
            _payload(
                [{"role": "user", "content": "Start."}],
                tools=[lookup, copy.deepcopy(lookup)],
            ),
            ratio=4,
            max_new_tokens=8,
        )


def test_constructor_enforces_frozen_geometry_and_s0_policy_shape() -> None:
    packing = _packing()
    packing["max_chunk_tokens"] = 512
    with pytest.raises(ValueError, match="max_chunk_tokens=768"):
        EventNativeS0Controller(Tokenizer(), packing=packing, policy=_policy())
    with pytest.raises(ValueError, match="lease_decisions=0"):
        EventNativeS0Controller(
            Tokenizer(),
            packing=_packing(),
            policy={**_policy(), "lease_decisions": 1},
        )
