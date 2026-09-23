"""CPU contracts for a bounded, source-preserving native tool argument rescue."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.events import EventStore
from history_memory.packing import encode_event_chunks, raw_workspace_messages

from benchmarks.memory_runtime.always_compress import CapacityInfeasible
from benchmarks.memory_runtime.candidate_algorithms.capacity_fallback import (
    CapacityFallbackAllocator,
)
from benchmarks.memory_runtime.candidate_algorithms.c1_v2 import (
    C1V2VerifiedController,
    c1_v2_fields,
)
from benchmarks.memory_runtime.candidate_algorithms.repacking import repack
from benchmarks.memory_runtime.candidate_algorithms.tool_event_rescue import (
    POLICY_VERSION,
    ToolEventRescueAllocator,
)
from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.recovery.config import E1_RECOVERY_VERSION
from benchmarks.memory_runtime.recovery.orchestrator import EventNativeRecoveryController
from benchmarks.memory_runtime.recovery.set_protocol import context_from_prepared
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.tests.test_goal_composition import call


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
        "ratios": [4, 8],
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


def policy(budget):
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


def s0(budget):
    return EventNativeS0Controller(Tokenizer(), packing=packing(), policy=policy(budget))


def with_recovery(controller):
    return EventNativeRecoveryController(
        controller, {"schema": E1_RECOVERY_VERSION, "gate": "disabled"}
    )


def payload(name, *, large_arguments=False, narrative=None, pending=False):
    arguments = {"key": 3}
    if large_arguments:
        arguments["details"] = "original argument token " * 48
    messages = [
        {"role": "system", "content": "Use the declared tools and observations."},
        {"role": "user", "content": "Remember the old project."},
        {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": "old-call", "type": "function",
                "function": {"name": "lookup", "arguments": '{"key": 1}'},
            }],
        },
        {"role": "tool", "tool_call_id": "old-call", "content": '{"value": 1}'},
        {"role": "user", "content": "Continue the current project."},
        {
            "role": "assistant", "content": None,
            "tool_calls": [{
                "id": "middle-call", "type": "function",
                "function": {"name": "lookup", "arguments": '{"key": 2}'},
            }],
        },
        {"role": "tool", "tool_call_id": "middle-call", "content": '{"value": 2}'},
        {
            "role": "assistant", "content": narrative,
            "tool_calls": [{
                "id": "latest-a", "type": "function",
                "function": {"name": "lookup", "arguments": json.dumps(arguments)},
            }, {
                "id": "latest-b", "type": "function",
                "function": {"name": "verify", "arguments": '{"key": 4}'},
            }],
        },
        {"role": "tool", "tool_call_id": "latest-b", "content": '{"verified": true}'},
    ]
    if not pending:
        messages.append({
            "role": "tool", "tool_call_id": "latest-a", "content": '{"value": 3}',
        })
    return {
        "session_id": name,
        "decision_key": "d1",
        "messages": messages,
        "tools": [{
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }],
    }


def _rendered(controller, prepared):
    with controller.measurement_context(prepared):
        context = controller._measurement_context.get()
        assert context is not None
        return raw_workspace_messages(
            prepared._store,
            prepared.memory.view,
            source_message_overrides=context["raw_message_overrides"],
        )


def test_unchanged_native_and_existing_fallback_success_paths():
    request = payload("native")
    base = s0(1_000_000)
    controller = ToolEventRescueAllocator(base)
    prepared = controller.prepare(request, ratio=8, max_new_tokens=8)
    reference = s0(1_000_000).prepare(request, ratio=8, max_new_tokens=8)
    assert prepared.memory == reference.memory
    assert prepared.metadata == reference.metadata
    assert "capacity_fallback" not in prepared.metadata
    assert base._sessions["native"].decisions["d1"][1] is prepared

    request = payload("legacy-fallback")
    rescued_wrapper = ToolEventRescueAllocator(s0(37))
    existing_wrapper = CapacityFallbackAllocator(s0(37))
    new_prepared = rescued_wrapper.prepare(request, ratio=8, max_new_tokens=8)
    old_prepared = existing_wrapper.prepare(request, ratio=8, max_new_tokens=8)
    assert new_prepared.memory == old_prepared.memory
    assert new_prepared.metadata == old_prepared.metadata
    assert new_prepared.metadata["capacity_fallback"]["stage"] == "requested_ratio_only"
    assert "argument_projections" not in new_prepared.metadata["capacity_fallback"]
    assert rescued_wrapper.base._sessions["legacy-fallback"].decisions["d1"][1] is new_prepared


def test_complete_tool_arguments_rescue_preserves_source_gist_and_live_bindings():
    request = payload("argument-rescue", large_arguments=True)
    original_messages = copy.deepcopy(request["messages"])
    with pytest.raises(CapacityInfeasible):
        CapacityFallbackAllocator(s0(60)).prepare(
            request, ratio=8, max_new_tokens=8
        )

    controller = ToolEventRescueAllocator(s0(60))
    prepared = with_recovery(controller).prepare(
        request, ratio=8, max_new_tokens=8
    )
    receipt = prepared.metadata["capacity_fallback"]
    assert receipt["terminal_rescue_version"] == POLICY_VERSION
    assert receipt["stage"] == "compress_complete_tool_arguments"
    assert receipt["argument_projections"]
    assert receipt["argument_projections"][0]["projected_call_ids"] == ["latest-a"]
    assert receipt["source_archive_unchanged"] is True
    assert receipt["encoder_gist_inputs_unchanged"] is True
    assert receipt["current_user_text_unchanged"] is True
    assert request["messages"] == original_messages
    assert prepared._store.messages[7].to_dict() == original_messages[7]
    assert prepared.metadata["actual_history_bytes"] <= 60
    assert prepared.metadata["per_ratio"]["8"]["history_bytes"] <= 60
    assert controller.policy_config.history_budget_bytes == 60
    assert controller.policy_config.workspace_budget_bytes == 60

    store = EventStore.from_messages("argument-rescue", original_messages)
    event = next(event for event in store.events if event.source_indices[0] == 7)
    assert event.complete and event.kind == "tool_event"
    assert event.event_id in prepared.memory.view.raw_event_ids
    assert event.event_id in prepared.memory.view.gist_event_ids
    assert 7 in prepared.metadata["partial_raw_source_indices"]
    assert 7 not in prepared.metadata["exact_raw_source_indices"]
    actual_chunks = tuple(
        chunk.token_ids for chunk in prepared.memory.chunks
        if chunk.event_id == event.event_id
    )
    source_chunks = tuple(
        chunk.token_ids for chunk in encode_event_chunks(store, event.event_id, Tokenizer())
    )
    assert actual_chunks == source_chunks
    assert actual_chunks

    rendered = _rendered(controller, prepared)
    assert rendered[0] == original_messages[0]
    assert original_messages[4] in rendered
    projected = next(row for row in rendered if row.get("tool_calls") and
                     row["tool_calls"][0]["id"] == "latest-a")
    source_calls = original_messages[7]["tool_calls"]
    assert [(call["id"], call["type"], call["function"]["name"])
            for call in projected["tool_calls"]] == [
                (call["id"], call["type"], call["function"]["name"])
                for call in source_calls
            ]
    assert projected["tool_calls"][0]["function"]["arguments"] == {
        "__c2kv_gist__": "arguments"
    }
    assert projected["tool_calls"][1]["function"]["arguments"] == {"key": 4}
    assert [(row["tool_call_id"], row["content"])
            for row in rendered if row["role"] == "tool"
            and row["tool_call_id"] in {"latest-a", "latest-b"}] == [
                ("latest-b", '{"verified": true}'),
                ("latest-a", '{"value": 3}'),
            ]
    assert list(prepared._tools) == request["tools"]
    context = context_from_prepared(prepared, [], "held draft")
    context_call = next(
        row for row in context["raw_visible"] if row.get("tool_calls")
        and row["tool_calls"][0]["id"] == "latest-a"
    )
    assert context_call["tool_calls"][0]["function"]["arguments"] == {
        "__c2kv_gist__": "arguments"
    }
    assert "original argument token" not in json.dumps(context["raw_visible"])
    assert event.event_id not in context["raw_source_ids"]


def test_pending_tool_event_is_not_projected():
    request = payload("pending", large_arguments=True, pending=True)
    request["messages"].append({"role": "user", "content": "Wait for the missing result."})
    snapshot = copy.deepcopy(request["messages"])
    controller = ToolEventRescueAllocator(s0(1))
    store = EventStore.from_messages("pending", request["messages"])
    assert not next(event for event in store.events if event.source_indices[0] == 7).complete
    assert controller._argument_candidates(request, {7}) == ()
    with pytest.raises(CapacityInfeasible):
        controller.prepare(request, ratio=8, max_new_tokens=8)
    assert request["messages"] == snapshot
    assert controller._measurement_context.get() is None


def test_narrative_and_argument_projection_compose_on_the_same_event():
    narrative = "long narrative token " * 40
    request = payload(
        "combined-projection", large_arguments=True, narrative=narrative
    )
    with pytest.raises(CapacityInfeasible):
        CapacityFallbackAllocator(s0(70)).prepare(
            request, ratio=8, max_new_tokens=8
        )
    controller = ToolEventRescueAllocator(s0(70))
    prepared = with_recovery(controller).prepare(
        request, ratio=8, max_new_tokens=8
    )
    receipt = prepared.metadata["capacity_fallback"]
    assert receipt["stage"] == "compress_complete_tool_arguments"
    assert [row["source_index"] for row in receipt["argument_projections"]] == [7]
    assert [row["source_index"] for row in receipt["narrative_projections"]] == [7]
    rendered = _rendered(controller, prepared)
    call = next(
        row for row in rendered if row.get("tool_calls")
        and row["tool_calls"][0]["id"] == "latest-a"
    )
    assert call["content"] is None
    assert call["tool_calls"][0]["function"]["arguments"] == {"__c2kv_gist__": "arguments"}
    assert call["tool_calls"][1]["function"]["arguments"] == {"key": 4}
    assert prepared._store.messages[7].to_dict()["content"] == narrative
    measure, _, repack_receipt = repack(controller, prepared)
    assert measure is not None and repack_receipt["status"] == "admitted"
    assert measure.raw_prompt_tokens == prepared.metadata["raw_prompt_tokens"]
    assert controller._measurement_context.get() is None


def test_true_overbudget_retains_the_original_typed_failure():
    request = payload("too-small", large_arguments=True)
    with pytest.raises(CapacityInfeasible) as expected:
        CapacityFallbackAllocator(s0(1)).prepare(
            request, ratio=8, max_new_tokens=8
        )
    with pytest.raises(CapacityInfeasible) as actual:
        ToolEventRescueAllocator(s0(1)).prepare(
            request, ratio=8, max_new_tokens=8
        )
    assert str(actual.value) == str(expected.value)


def test_repack_replays_argument_projection_and_protects_its_gist():
    controller = ToolEventRescueAllocator(s0(60))
    prepared = with_recovery(controller).prepare(
        payload("repack-arguments", large_arguments=True),
        ratio=8, max_new_tokens=8,
    )
    projected = {
        row["event_id"]
        for row in prepared.metadata["capacity_fallback"]["argument_projections"]
    }
    measure, metadata, receipt = repack(controller, prepared)
    assert measure is not None
    assert receipt["status"] == "admitted"
    assert projected <= set(measure.memory.view.gist_event_ids)
    assert measure.raw_prompt_tokens == prepared.metadata["raw_prompt_tokens"]
    assert metadata["actual_history_bytes"] <= 60
    assert controller._measurement_context.get() is None

    blocked, _, failed = repack(controller, prepared, derived_messages=[
        {"role": "user", "content": "charged extra evidence " * 500}
    ])
    assert blocked is None
    assert not projected.intersection(failed["released_gist_event_ids"])
    assert controller._measurement_context.get() is None

    with pytest.raises(RuntimeError):
        with controller.measurement_context(prepared):
            assert controller._measurement_context.get() is not None
            raise RuntimeError("probe")
    assert controller._measurement_context.get() is None


def test_non_capacity_error_propagates_without_a_rescue_attempt(monkeypatch):
    base = s0(1_000_000)
    controller = ToolEventRescueAllocator(base)

    def broken_prepare(*args, **kwargs):
        raise RuntimeError("unrelated preparation failure")

    monkeypatch.setattr(base, "prepare", broken_prepare)
    with pytest.raises(RuntimeError, match="unrelated preparation failure"):
        controller.prepare(payload("non-capacity"), ratio=8, max_new_tokens=8)
    assert controller._measurement_context.get() is None


@pytest.mark.parametrize("scope", ["record", "record_bound", "record_bound_structural", "adjacent_pair"])
def test_non_whole_event_scopes_do_not_claim_arguments_are_gist_backed(scope):
    controller = ToolEventRescueAllocator(s0(60))
    controller.encoding_scope = scope
    assert controller._argument_candidates(payload("scope", large_arguments=True), {7}) == ()


@pytest.mark.parametrize(
    ("score", "budget", "regenerate"),
    [(0.1, 70, False), (0.9, 70, True), (0.9, 60, False)],
)
def test_c1_v2_commit_keeps_original_draft_after_argument_rescue(
    score, budget, regenerate
):
    config = {
        "variant": "c1_v2_verified",
        "risk_artifact": {"fixture": True},
        "risk_threshold": 0.5,
        **c1_v2_fields("c1_v2_verified"),
    }
    base = ToolEventRescueAllocator(s0(budget))
    controller = C1V2VerifiedController(base, config, risk_model=Risk(score))
    request = payload(f"c1-integration-{score}-{budget}", large_arguments=True)
    prepared = controller.prepare(request, ratio=8, max_new_tokens=8)
    assert controller.prepare(request, ratio=8, max_new_tokens=8) is prepared
    assert prepared.metadata["capacity_fallback"]["stage"] == (
        "compress_complete_tool_arguments"
    )
    projected_event = prepared.metadata["capacity_fallback"][
        "argument_projections"
    ][0]["event_id"]
    source_arguments = request["messages"][7]["tool_calls"][0]["function"]["arguments"]
    assert "original argument token" in source_arguments

    draft = [call(arguments={"key": 99, "request": "fresh action"})]
    original_draft = copy.deepcopy(draft)
    result = controller.reconsider(prepared, draft, draft_text="lookup")
    assert result["regenerate"] is regenerate
    assert result["decision"]["gate"]["triggered"] is (score > 0.5)
    assert draft == original_draft
    if regenerate:
        assert result["decision"]["reason"] == (
            "risk_triggered_complete_event_replaced"
        )
        assert projected_event in result["memory"].view.raw_event_ids
        assert projected_event in result["memory"].view.gist_event_ids
        assert result["metadata"]["capacity_fallback"]["argument_projections"]
        assert result["metadata"]["actual_history_bytes"] <= budget
        assert tuple(
            chunk.token_ids for chunk in result["memory"].chunks
            if chunk.event_id == projected_event
        ) == tuple(
            chunk.token_ids for chunk in prepared.memory.chunks
            if chunk.event_id == projected_event
        )
    else:
        assert result["memory"] == prepared.memory
        assert result["metadata"]["capacity_fallback"]["argument_projections"]
        if score > 0.5:
            assert result["decision"]["reason"] == (
                "no_feasible_new_complete_event"
            )
        else:
            assert result["decision"]["reason"] == "risk_not_above_threshold"

    verdict = controller.validate_commit(prepared, draft, draft_text="lookup")
    assert verdict["accepted"] is True
    committed, receipt = controller.finalize_commit(prepared, draft)
    assert list(committed) == original_draft
    assert json.loads(committed[0]["function"]["arguments"]) == {
        "key": 99, "request": "fresh action"
    }
    assert "__c2kv_gist__" not in json.dumps(committed)
    assert receipt["changed"] is False
    assert base._measurement_context.get() is None

    next_prepared = controller.prepare(
        payload(f"plain-after-{score}-{budget}"), ratio=8, max_new_tokens=8
    )
    assert not next_prepared.metadata.get("capacity_fallback", {}).get(
        "argument_projections"
    )
    assert "__c2kv_gist__" not in json.dumps(
        context_from_prepared(next_prepared, [], "next draft")["raw_visible"]
    )
    assert base._measurement_context.get() is None
