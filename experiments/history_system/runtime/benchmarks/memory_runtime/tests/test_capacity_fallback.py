"""CPU contracts for the bounded native S0 capacity fallback."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.events import EventStore
from history_memory.packing import raw_workspace_messages

from benchmarks.memory_runtime.always_compress import CapacityInfeasible
from benchmarks.memory_runtime.candidate_algorithms.capacity_fallback import (
    POLICY_VERSION,
    CapacityFallbackAllocator,
    _bounded_projection_sets,
    build_capacity_fallback_allocator,
)
from benchmarks.memory_runtime.candidate_algorithms.repacking import repack
from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from benchmarks.memory_runtime.event_native_s0_policy import (
    EventNativeS0Controller,
    S0_CONFIG_DEFAULTS,
)
from benchmarks.memory_runtime.recovery.orchestrator import (
    EventNativeRecoveryController,
)
from benchmarks.memory_runtime.recovery.config import E1_RECOVERY_VERSION
from benchmarks.memory_runtime.same_event_bridge_only import (
    SAME_EVENT_BRIDGE_ONLY_POLICY,
    SameEventBridgeOnlyS0Controller,
)


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


def packing(ratios=(4, 8)):
    return {
        "ratios": list(ratios),
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


def tool_pair(number, *, narrative=None):
    call_id = f"call-{number}"
    return [
        {
            "role": "assistant",
            "content": narrative,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": json.dumps({"key": number}),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps({"value": number}),
        },
    ]


def payload(name="capacity", *, latest_narrative=None, old_narrative=None):
    return {
        "session_id": name,
        "decision_key": "d1",
        "messages": [
            {"role": "system", "content": "Use tool observations."},
            {"role": "user", "content": "Remember the old project."},
            *tool_pair(1, narrative=old_narrative),
            {"role": "user", "content": "Continue the current project."},
            *tool_pair(2),
            *tool_pair(3, narrative=latest_narrative),
        ],
        "tools": [],
    }


def s0(budget, *, ratios=(4, 8)):
    return EventNativeS0Controller(
        Tokenizer(), packing=packing(ratios), policy=policy(budget)
    )


def fallback(budget):
    return CapacityFallbackAllocator(s0(budget))


def test_legacy_feasible_path_is_the_exact_incumbent_object_and_behavior():
    request = payload("legacy")
    base = s0(1_000_000)
    controller = CapacityFallbackAllocator(base)
    prepared = controller.prepare(request, ratio=8, max_new_tokens=8)
    reference = s0(1_000_000).prepare(
        request, ratio=8, max_new_tokens=8
    )

    assert prepared.memory == reference.memory
    assert prepared.metadata == reference.metadata
    assert "capacity_fallback" not in prepared.metadata
    assert base._sessions["legacy"].decisions["d1"][1] is prepared
    expected = base.reconsider(prepared, [], draft_text="done")
    assert controller.reconsider(prepared, [], draft_text="done") == expected


def test_ratio4_only_rejection_is_rescued_by_actual_requested_ratio8():
    request = payload("ratio8")
    with pytest.raises(CapacityInfeasible):
        s0(27).prepare(request, ratio=8, max_new_tokens=8)

    prepared = fallback(27).prepare(request, ratio=8, max_new_tokens=8)
    receipt = prepared.metadata["capacity_fallback"]
    assert receipt["version"] == POLICY_VERSION
    assert receipt["stage"] == "requested_ratio_only"
    assert receipt["measurement_ratios"] == [8]
    assert set(prepared.metadata["per_ratio"]) == {"8"}
    assert prepared.metadata["actual_history_bytes"] <= 27


def test_true_overbudget_keeps_the_original_typed_failure():
    request = payload(
        "overbudget", latest_narrative="unavoidable complete source " * 500
    )
    with pytest.raises(CapacityInfeasible) as expected:
        s0(1).prepare(request, ratio=8, max_new_tokens=8)
    with pytest.raises(CapacityInfeasible) as actual:
        fallback(1).prepare(request, ratio=8, max_new_tokens=8)
    assert str(actual.value) == str(expected.value)


def test_duplicate_gist_is_removed_only_for_a_complete_full_raw_event():
    request = payload("dedupe")
    prepared = fallback(26).prepare(request, ratio=8, max_new_tokens=8)
    receipt = prepared.metadata["capacity_fallback"]
    assert receipt["stage"] == "remove_redundant_gist"
    assert receipt["narrative_projections"] == []
    store = EventStore.from_messages("dedupe", request["messages"])
    raw = set(prepared.memory.view.raw_event_ids)
    for event_id in receipt["removed_duplicate_gist_event_ids"]:
        assert event_id in raw
        assert store.event(event_id).complete
        assert set(store.event(event_id).source_indices) <= set(
            prepared.memory.raw_source_indices
        )


def test_projection_preserves_source_calls_results_gist_and_common_baseline():
    narrative = "retain this immutable narrative " * 40
    request = payload("projection", latest_narrative=narrative)
    source_snapshot = copy.deepcopy(request["messages"])
    common_reference = s0(1_000_000).prepare(
        payload("common", latest_narrative=narrative),
        ratio=8,
        max_new_tokens=8,
    ).metadata["common_raw_prompt_tokens"]
    controller = fallback(47)
    prepared = controller.prepare(request, ratio=8, max_new_tokens=8)
    receipt = prepared.metadata["capacity_fallback"]

    assert receipt["stage"] == "omit_complete_tool_event_narrative"
    assert receipt["common_raw_prompt_tokens"] == common_reference
    assert request["messages"] == source_snapshot
    projection = receipt["narrative_projections"][0]
    event_id = projection["event_id"]
    assert event_id in prepared.memory.view.raw_event_ids
    assert event_id in prepared.memory.view.gist_event_ids
    assert projection["call_ids"] == ["call-3"]
    assert projection["result_source_indices"] == [8]

    store = EventStore.from_messages("projection", request["messages"])
    assert narrative in store.messages[projection["source_index"]].json_text
    projected_message = copy.deepcopy(
        store.messages[projection["source_index"]].to_dict()
    )
    projected_message["content"] = None
    rendered = raw_workspace_messages(
        store,
        prepared.memory.view,
        source_message_overrides={projection["source_index"]: projected_message},
    )
    assert any(row["role"] == "user" for row in rendered)
    assistant = next(
        row for row in rendered
        if row.get("tool_calls")
        and row["tool_calls"][0].get("id") == "call-3"
    )
    assert assistant["content"] is None
    assert assistant["tool_calls"][0]["function"]["arguments"] == {"key": 3}
    result = next(row for row in rendered if row.get("tool_call_id") == "call-3")
    assert result["content"] == json.dumps({"value": 3})


def test_gist_only_large_prose_cannot_displace_the_measured_raw_candidate():
    request = payload(
        "candidate-pool",
        old_narrative="old gist-only prose " * 500,
        latest_narrative="latest raw prose " * 20,
    )
    controller = fallback(1_000_000)
    rows = controller._projection_candidates(
        request, allowed_source_indices={7, 8}
    )
    assert [row["source_index"] for row in rows] == [7]


def test_three_projection_prefix_is_reachable_within_the_bound():
    candidates = tuple({
        "source_index": index,
        "standalone_rendered_token_delta": saving,
    } for index, saving in enumerate((5, 9, 7)))
    choices = list(_bounded_projection_sets(candidates))
    assert any(len(choice) == 3 for choice in choices)
    assert len(choices) <= 2 * len(candidates) - 1
    assert [len(choice) for choice in choices[:3]] == [1, 1, 1]


def test_repack_reuses_projection_and_ratio_context_without_leaking_state():
    narrative = "capacity-bound narrative " * 40
    controller = fallback(37)
    recovery = EventNativeRecoveryController(
        controller, {"schema": E1_RECOVERY_VERSION, "gate": "disabled"}
    )
    prepared = recovery.prepare(
        payload("repack", latest_narrative=narrative),
        ratio=8,
        max_new_tokens=8,
    )
    measure, metadata, receipt = repack(controller, prepared)
    assert measure is not None
    assert receipt["status"] == "admitted"
    assert receipt["raw_gist_exclusive"] is False
    assert receipt["projected_gist_event_ids"]
    assert measure.raw_prompt_tokens == prepared.metadata["raw_prompt_tokens"]
    assert set(measure.per_ratio) == {"8"}
    assert metadata["actual_history_bytes"] <= 37
    assert metadata["gist_reservation"]["status"] == (
        "retained_for_projected_raw_source_coverage"
    )
    assert controller._measurement_context.get() is None

    projected = {
        row["event_id"]
        for row in prepared.metadata["capacity_fallback"]["narrative_projections"]
    }
    blocked, _, failed = repack(controller, prepared, derived_messages=[
        {"role": "user", "content": "charged extra evidence " * 500}
    ])
    assert blocked is None
    assert not projected.intersection(failed["released_gist_event_ids"])

    with pytest.raises(RuntimeError):
        with controller.measurement_context(prepared):
            assert controller._measurement_context.get() is not None
            raise RuntimeError("probe")
    assert controller._measurement_context.get() is None


def test_builder_preserves_the_frozen_same_event_bridge_base():
    controller = build_capacity_fallback_allocator(
        Tokenizer(),
        packing=packing(),
        policy=policy(),
        model_context=None,
        s0_config={
            **S0_CONFIG_DEFAULTS,
            "observed_entity_slot_policy": SAME_EVENT_BRIDGE_ONLY_POLICY,
        },
        benchmark="bfcl",
    )
    assert isinstance(controller.base, SameEventBridgeOnlyS0Controller)
