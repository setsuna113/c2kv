"""Contracts for candidate-first D3 raw-event recovery."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.events import EventStore

from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
    PolicyInputError,
)
from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import (
    build_event_native_controller,
)
from benchmarks.memory_runtime.event_native_s0_policy import (
    S0_CONFIG_DEFAULTS,
    EventNativeS0Controller,
)
from benchmarks.memory_runtime.recovery.config import E1_RECOVERY_VERSION
from benchmarks.memory_runtime.recovery.hybrid import D3HybridRecoveryController
from benchmarks.memory_runtime.recovery.source import select_source_event


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tools=None, add_generation_prompt=False, **kwargs
    ):
        text = (
            "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>"
            if tools
            else ""
        )
        for message in messages:
            text += (
                "<" + message["role"] + ">"
                + json.dumps(message, sort_keys=True)
                + "</end>"
            )
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]


def detector_config():
    return {
        "schema": E1_RECOVERY_VERSION,
        "gate": "seeded_random",
        "random_seed": 0,
        "random_probability": {"numerator": 1, "denominator": 5},
        "quota": {"numerator": 1, "denominator": 5},
        "task_generation_limit": 96,
    }


def messages():
    result = [
        {"role": "system", "content": "Use observed tool results."},
        {"role": "user", "content": "Remember project facts."},
    ]
    for index, (key, value) in enumerate(
        (
            ("alpha", "violet"),
            ("beta", "orange"),
            ("gamma", "green"),
            ("status", "continue alpha using violet"),
        ),
        1,
    ):
        result.extend(
            (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call-{index}",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": json.dumps({"key": key}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": f"call-{index}",
                    "content": json.dumps({"value": value}),
                },
            )
        )
    result.append({"role": "user", "content": "Actually continue."})
    return result


def packing_config():
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


def policy_config(*, budget=1_000_000):
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


def base_controller(*, budget=1_000_000):
    return EventNativeS0Controller(
        Tokenizer(),
        packing=packing_config(),
        policy=policy_config(budget=budget),
    )


def payload(session_id="bfcl/hybrid"):
    return {
        "session_id": session_id,
        "decision_key": "turn-1/decision-1",
        "messages": copy.deepcopy(messages()),
        "tools": [],
    }


def test_source_options_add_latest_observation_without_changing_native_defaults():
    store = EventStore.from_messages("bfcl/source", messages())
    tools = [event for event in store.events if event.kind == "tool_event"]
    latest = tools[-1]
    prepared = SimpleNamespace(
        _store=store,
        memory=SimpleNamespace(
            view=SimpleNamespace(
                raw_event_ids=(latest.event_id,),
                mandatory_raw_event_ids=(latest.event_id,),
            )
        ),
        metadata={
            "eligible_extraction": {
                "eligible_event_ids": [event.event_id for event in store.events]
            },
            "revision_cancelled_event_ids": [],
        },
    )

    native, native_receipt = select_source_event(
        prepared, [], draft_text=""
    )
    hybrid, hybrid_receipt = select_source_event(
        prepared,
        [],
        draft_text="",
        include_latest_complete_observation=True,
        explicit_revision_abstain=False,
        allow_empty_draft_query=True,
    )

    assert native is None
    assert native_receipt["reason"] == "no_held_draft_query"
    assert hybrid == tools[0].event_id
    assert hybrid_receipt["latest_complete_observation_event_id"] == latest.event_id
    assert hybrid_receipt["staleness_policy"] == "revision-cancelled-events-only-v1"

    prepared.metadata["revision_cancelled_event_ids"] = [tools[0].event_id]
    cancelled, cancelled_receipt = select_source_event(
        prepared,
        [],
        draft_text="",
        include_latest_complete_observation=True,
        explicit_revision_abstain=False,
        allow_empty_draft_query=True,
    )
    assert cancelled != tools[0].event_id
    assert tools[0].event_id not in cancelled_receipt[
        "ranked_candidate_event_ids"
    ]


def test_real_s0_recovery_restores_complete_event_with_native_b0_metadata():
    controller = D3HybridRecoveryController(
        base_controller(budget=900), detector_config()
    )
    controller._gate = lambda prepared: {
        "type": "prefill_linear_head",
        "triggered": True,
        "reason": "prefill_score_at_or_above_threshold",
        "score": 1.0,
    }
    prepared = controller.prepare(payload(), ratio=4, max_new_tokens=8)
    before = copy.deepcopy(prepared.memory.view)
    store = prepared._store
    oldest_tool = [
        event for event in store.events if event.kind == "tool_event"
    ][0]
    assert oldest_tool.event_id not in before.raw_event_ids

    result = controller.reconsider(
        prepared, [], draft_text="Use the observed violet value."
    )

    assert result["regenerate"] is True
    assert result["decision"]["upgraded_event_id"] == oldest_tool.event_id
    assert oldest_tool.event_id in result["memory"].view.raw_event_ids
    assert set(before.mandatory_raw_event_ids) <= set(
        result["memory"].view.raw_event_ids
    )
    assert result["metadata"]["post_draft_exact_recovery_applied"] is True
    assert result["metadata"]["post_draft_recovery_allocation"][
        "mandatory_raw_event_ids_unchanged"
    ] is True
    allocation = result["metadata"]["post_draft_recovery_allocation"]
    assert allocation["demoted_raw_event_ids"]
    assert set(allocation["demoted_raw_event_ids"]) <= set(
        before.gist_event_ids
    )
    assert allocation["minimum_gist_preserved"] is True
    restored = result["decision"]["restored_event"]
    assert restored["event_id"] == oldest_tool.event_id
    assert restored["representation"] == "native_raw_event"
    assert restored["marginal_raw_prompt_tokens"] > 0
    assert result["decision"]["limits"]["legacy_e1_quota_applied"] is False
    assert "quota" not in result["decision"]

    cached = controller.reconsider(
        prepared, [], draft_text="Use the observed violet value."
    )
    assert cached == result
    with pytest.raises(PolicyInputError, match="two different drafts"):
        controller.reconsider(prepared, [], draft_text="Different draft")

    second_payload = payload()
    second_payload["decision_key"] = "turn-1/decision-2"
    second = controller.prepare(second_payload, ratio=4, max_new_tokens=8)
    second_result = controller.reconsider(
        second, [], draft_text="Use the observed violet value."
    )
    assert second_result["regenerate"] is True
    assert second_result["decision"]["limits"]["recovery_count_before"] == 1
    assert second_result["decision"]["task_recovery_count"] == 2


def test_first_infeasible_ranked_event_falls_back_to_next_complete_event():
    controller = D3HybridRecoveryController(base_controller(), detector_config())
    controller._gate = lambda prepared: {
        "type": "prefill_linear_head",
        "triggered": True,
        "reason": "prefill_score_at_or_above_threshold",
        "score": 1.0,
    }
    prepared = controller.prepare(
        payload("bfcl/fallback"), ratio=4, max_new_tokens=8
    )
    candidates = [
        event.event_id
        for event in prepared._store.events
        if event.kind == "tool_event"
        and event.event_id not in prepared.memory.view.raw_event_ids
    ]
    assert len(candidates) >= 2
    controller._select_ranked_events = lambda *args, **kwargs: (
        candidates[0],
        {
            "policy": "test-ranked-events",
            "reason": "ranked",
            "ranked_candidate_event_ids": candidates[:2],
            "selected_event": None,
        },
    )
    native_admit = controller._admit

    def admit(context, event_id):
        if event_id == candidates[0]:
            return {
                "measure": None,
                "receipt": {
                    "policy": "r-event-b0-repack-v1",
                    "status": "abstained",
                    "candidate_event_id": event_id,
                    "demoted_raw_event_ids": [],
                    "released_gist_event_ids": [],
                },
            }
        return native_admit(context, event_id)

    controller._admit = admit
    result = controller.reconsider(prepared, [], draft_text="Use history.")

    trials = result["decision"]["candidate_feasibility"]["trials"]
    assert [trial["admitted"] for trial in trials] == [False, True]
    assert result["decision"]["upgraded_event_id"] == candidates[1]


def test_shared_task_generation_limit_still_abstains_without_quota():
    controller = D3HybridRecoveryController(base_controller(), detector_config())
    controller._gate = lambda prepared: {
        "type": "prefill_linear_head",
        "triggered": True,
        "reason": "prefill_score_at_or_above_threshold",
        "score": 1.0,
    }
    prepared = controller.prepare(
        payload("bfcl/task-cap"), ratio=4, max_new_tokens=8
    )
    prepared.metadata["decision_index"] = 96

    result = controller.reconsider(
        prepared, [], draft_text="Use the observed violet value."
    )

    assert result["regenerate"] is False
    assert result["decision"]["reason"] == "shared_task_generation_limit"
    assert result["decision"]["limits"][
        "online_cumulative_quota_applied"
    ] is False


def test_feasibility_is_side_effect_free_when_prefill_gate_abstains():
    controller = D3HybridRecoveryController(base_controller(), detector_config())
    gate_calls = []

    def gate(prepared):
        gate_calls.append(True)
        return {
            "type": "prefill_linear_head",
            "triggered": False,
            "reason": "prefill_score_below_threshold",
            "score": 0.0,
        }

    controller._gate = gate
    prepared = controller.prepare(
        payload("bfcl/no-mutation"), ratio=4, max_new_tokens=8
    )
    before_memory = prepared.memory
    before_view = copy.deepcopy(prepared.memory.view)

    result = controller.reconsider(
        prepared, [], draft_text="Use the observed violet value."
    )

    assert gate_calls == [True]
    assert result["regenerate"] is False
    assert result["memory"] is before_memory
    assert result["memory"].view == before_view
    assert result["decision"]["candidate_feasibility"][
        "state_mutated_before_gate"
    ] is False
    assert result["metadata"]["route"]["baseline_identity"].endswith(
        "+d3-hybrid-post-draft-event-recovery"
    )


def test_empty_draft_guard_runs_after_candidates_and_before_detector():
    controller = D3HybridRecoveryController(base_controller(), detector_config())
    gate_calls = []
    controller._gate = lambda prepared: gate_calls.append(True)
    prepared = controller.prepare(
        payload("bfcl/empty-draft"), ratio=4, max_new_tokens=8
    )

    result = controller.reconsider(prepared, [], draft_text="")

    assert result["regenerate"] is False
    assert result["decision"]["reason"] == (
        "no_held_draft_text_or_valid_tool_call"
    )
    assert result["decision"]["candidate_feasibility"][
        "tried_candidate_count"
    ] >= 1
    assert "gate" not in result["decision"]
    assert gate_calls == []


def test_top_level_routing_flag_builds_hybrid_without_gp_overlay():
    controller = build_event_native_controller(
        Tokenizer(),
        packing=packing_config(),
        policy=policy_config(),
        view_mode=NATIVE_S0_MODE,
        compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config={
            **copy.deepcopy(S0_CONFIG_DEFAULTS),
            "d3_hybrid_recovery": True,
            "post_draft_recovery": detector_config(),
        },
    )

    assert isinstance(controller, D3HybridRecoveryController)
    assert isinstance(controller.base, EventNativeS0Controller)
    assert not hasattr(controller, "gp")

    with pytest.raises(ValueError, match="cannot be combined with gp_experiments"):
        build_event_native_controller(
            Tokenizer(),
            packing=packing_config(),
            policy=policy_config(),
            view_mode=NATIVE_S0_MODE,
            compression_policy=ALWAYS_COMPRESSION_POLICY,
            s0_config={
                **copy.deepcopy(S0_CONFIG_DEFAULTS),
                "d3_hybrid_recovery": True,
                "post_draft_recovery": detector_config(),
                "gp_experiments": {},
            },
        )
