"""CPU contracts for distinct first-draft candidate allocations."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.events import EventStore
from history_memory.packing import select_view

from benchmarks.memory_runtime.always_compress import CapacityInfeasible
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.candidate_algorithms.allocation import CandidateAllocator
from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tools=None, add_generation_prompt=False, **kwargs
    ):
        text = ""
        if tools:
            text += "<tools> " + json.dumps(tools, sort_keys=True) + " </tools> "
        for message in messages:
            text += (
                "<" + message["role"] + "> "
                + json.dumps(message, sort_keys=True, ensure_ascii=False)
                + " </end> "
            )
        if add_generation_prompt:
            text += "<assistant> "
        return [
            sum((index + 1) * ord(char) for index, char in enumerate(word))
            for word in text.split()
        ]


def packing():
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


def policy(budget=1_000_000):
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


def tool_pair(number, *, result):
    call_id = f"call-{number}"
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": json.dumps({"key": number}),
                },
            }],
        },
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)},
    ]


def messages():
    return [
        {"role": "system", "content": "Use tool observations."},
        {"role": "user", "content": "Remember the old project."},
        *tool_pair(1, result={"value": "old"}),
        {"role": "user", "content": "Continue the current project."},
        *tool_pair(2, result={"error": "retry required"}),
        *tool_pair(3, result={"value": "still working"}),
    ]


def payload(name="allocation"):
    return {
        "session_id": name,
        "decision_key": "d1",
        "messages": copy.deepcopy(messages()),
        "tools": [],
    }


def allocator(variant, *, budget=1_000_000):
    return CandidateAllocator(
        Tokenizer(), packing=packing(), policy=policy(budget), variant=variant
    )


def test_static_first_view_uses_select_view_without_s0_extras():
    controller = allocator("static_t02")
    request = payload("static")
    prepared = controller.prepare(request, ratio=4, max_new_tokens=8)
    selected = select_view(
        EventStore.from_messages("static", request["messages"]),
        recent_tool_events=1,
    )
    assert set(prepared.memory.view.raw_event_ids) == set(selected.raw_event_ids)
    assert set(prepared.memory.view.gist_event_ids) == set(selected.gist_event_ids)
    assert not set(prepared.memory.view.raw_event_ids) & set(
        prepared.memory.view.gist_event_ids
    )
    assert prepared.metadata["source_needs"]["prediction_status"] == "not_run"
    assert prepared.metadata["raw_reserve"]["status"] == "disabled_for_variant"
    assert prepared.metadata["failed_operation_cue"]["status"] == (
        "disabled_for_variant"
    )
    assert history_budget_receipt(
        prepared.memory, prepared.metadata, controller, ratio=4, phase="draft"
    )["status"] == "passed"
    assert controller.prepare(request, ratio=4, max_new_tokens=8) is prepared


def test_static_can_fit_when_s0_minimum_gist_cannot():
    request = payload("tight")
    feasible = None
    for budget in range(1, 100):
        candidate = allocator("static_t02", budget=budget)
        try:
            prepared = candidate.prepare(request, ratio=4, max_new_tokens=8)
        except CapacityInfeasible:
            continue
        if not prepared.memory.view.gist_event_ids:
            feasible = budget
            break
    assert feasible is not None
    with pytest.raises(CapacityInfeasible):
        EventNativeS0Controller(
            Tokenizer(), packing=packing(), policy=policy(feasible)
        ).prepare(request, ratio=4, max_new_tokens=8)


def test_turn_first_view_prioritizes_current_complete_calls_and_failed_cue():
    controller = allocator("turn_c1")
    prepared = controller.prepare(payload("turn"), ratio=4, max_new_tokens=8)
    raw = set(prepared.memory.view.raw_event_ids)
    gist = set(prepared.memory.view.gist_event_ids)
    assert "turn:m5" in raw
    assert "turn:m7" in raw
    assert prepared.metadata["current_turn_raw_event_ids"] == ["turn:m5", "turn:m7"]
    assert "turn:m2" not in raw
    assert "turn:m2" in gist
    assert not raw & gist
    assert prepared.metadata["source_needs"]["prediction_status"] == "not_run"
    assert prepared.metadata["failed_operation_cue"]["status"] == "admitted"
    assert len(prepared.metadata["derived_workspace_prefix_messages"]) == 1
    assert history_budget_receipt(
        prepared.memory, prepared.metadata, controller, ratio=4, phase="draft"
    )["status"] == "passed"


def test_goal_rescue_keeps_exact_s0_initial_memory():
    request = payload("goal")
    candidate = allocator("goal_rescue").prepare(
        request, ratio=4, max_new_tokens=8
    )
    baseline = EventNativeS0Controller(
        Tokenizer(), packing=packing(), policy=policy()
    ).prepare(request, ratio=4, max_new_tokens=8)
    assert candidate.memory == baseline.memory
    assert candidate.metadata["candidate_algorithm"] == "goal_rescue"
    assert candidate.metadata["source_needs"] == baseline.metadata["source_needs"]


def test_dependency_first_admits_source_bound_packets_under_b0():
    controller = allocator("dependency_first")
    request = {
        "session_id": "dependency",
        "decision_key": "d1",
        "messages": [
            {"role": "system", "content": "Use observed results."},
            {"role": "user", "content": "Create a project and look it up."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "make-1", "type": "function",
                "function": {"name": "create_project", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "make-1", "content": (
                '{"project_id":"project-1234"}'
            )},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "lookup-1", "type": "function",
                "function": {"name": "lookup_project", "arguments": (
                    '{"project_id":"project-1234"}'
                )},
            }]},
            {"role": "tool", "tool_call_id": "lookup-1", "content": (
                '{"status":"active"}'
            )},
            {"role": "user", "content": "What is the project status?"},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup_project",
                "parameters": {
                    "type": "object",
                    "properties": {"project_id": {"type": "string"}},
                },
            },
        }],
    }
    prepared = controller.prepare(request, ratio=4, max_new_tokens=8)
    receipt = prepared.metadata["dependency_packet"]
    assert receipt["b0_rechecked"] is True
    assert receipt["b0_admitted_packet_count"] > 0
    assert receipt["source_indices"]
    assert receipt["b0_admitted_source_indices"] == receipt["source_indices"]
    assert prepared.metadata["derived_workspace_prefix_messages"]
    assert not set(prepared.memory.view.raw_event_ids) & set(
        prepared.memory.view.gist_event_ids
    )
    assert history_budget_receipt(
        prepared.memory, prepared.metadata, controller, ratio=4, phase="draft"
    )["status"] == "passed"


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown candidate allocator variant"):
        allocator("unknown")
