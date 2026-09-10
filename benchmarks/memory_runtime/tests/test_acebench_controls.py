"""Controller composition checks for receipt-backed ACEBench history."""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.memory_runtime.acebench_controls import (
    ACEBENCH_VIEW_MODES,
    AceEventNativeExactController,
    AceEventNativeOnePassController,
    build_acebench_controller,
)
from benchmarks.memory_runtime.acebench_source import build_ace_event_store, parse_ace_draft
from benchmarks.memory_runtime.exact_gap import detect_exact_source_gap
from benchmarks.memory_runtime.policy import PolicyInputError
from benchmarks.memory_runtime.tests.test_event_native_policy import Tokenizer, contracts
from history_memory.packing import PackingBudgetError


RATIO = 4
MAX_NEW_TOKENS = 2


def _receipt(
    *, count=1, execution_index=3, agent_history_index=1,
    decoded_calls=None,
):
    return {
        "execution_message_index": execution_index,
        "version": "acebench-execution-receipt-v1",
        "agent_history_index": agent_history_index,
        "decode_status": "ok",
        "decoded_calls": decoded_calls or ["Reserve(city='Paris')"],
        "executor_status": "returned",
        "executor_return_shape": "list",
        "executor_return_count": count,
    }


def _messages(*, padding=1600):
    return [
        {"role": "system", "content": "Use observable API results."},
        {"role": "user", "content": "Reserve Paris."},
        {"role": "assistant", "content": "[Reserve(city='Paris')]"},
        {
            "role": "tool",
            "content": json.dumps(
                [{"reservation_id": "R-7", "detail": "Z" * padding}]
            ),
        },
        {"role": "assistant", "content": "The reservation was created."},
        {"role": "user", "content": "Check an unrelated detail."},
        {"role": "assistant", "content": "Unrelated " + "Q" * padding},
        {"role": "user", "content": "Cancel the reservation."},
    ]


def _payload(*, decision_key="d1", messages=None, receipt=None):
    return {
        "session_id": "ace/controller",
        "decision_key": decision_key,
        "messages": copy.deepcopy(messages or _messages()),
        "tools": [],
        "c2kv_ace_source": {
            "version": "acebench-text-actions-v1",
            "receipts": [copy.deepcopy(receipt or _receipt())],
        },
    }


def _two_execution_payload():
    messages = _messages(padding=0)[:6] + [
        {"role": "assistant", "content": "[Clock(city='London')]"},
        {"role": "tool", "content": '[{"time":"09:30"}]'},
        {"role": "assistant", "content": "Unrelated archive: " + "Q" * 3000},
        {"role": "user", "content": "Cancel the reservation."},
    ]
    payload = _payload(messages=messages)
    payload["c2kv_ace_source"]["receipts"].append(
        _receipt(
            execution_index=7,
            agent_history_index=5,
            decoded_calls=["Clock(city='London')"],
        )
    )
    return payload


def _controller(view_mode, *, packing=None, policy=None):
    default_packing, default_policy = contracts()
    default_packing.update(
        max_chunk_tokens=96,
        chunk_overlap=8,
        max_chunks=256,
        max_encoder_tokens=200000,
        max_workspace_tokens=200000,
        max_sequence_tokens=200000,
    )
    default_policy.update(workspace_budget_bytes=10_000_000)
    return build_acebench_controller(
        Tokenizer(),
        packing=copy.deepcopy(packing or default_packing),
        policy=copy.deepcopy(policy or default_policy),
        view_mode=view_mode,
    )


def _activated_contracts(payload=None):
    packing, policy = contracts()
    packing.update(
        max_chunk_tokens=96,
        chunk_overlap=8,
        max_chunks=256,
        max_encoder_tokens=200000,
        max_workspace_tokens=200000,
        max_sequence_tokens=200000,
    )
    policy.update(workspace_budget_bytes=10_000_000)
    probe = build_acebench_controller(
        Tokenizer(), packing=copy.deepcopy(packing), policy=copy.deepcopy(policy),
        view_mode="capacity_exact_once",
    ).prepare(
        payload or _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    policy["history_budget_bytes"] = (
        probe.metadata["capacity_gate"]["full_history_bytes"] - 1
    )
    return packing, policy


def test_factory_routes_only_supported_acebench_views() -> None:
    assert ACEBENCH_VIEW_MODES == {
        "capacity_protect",
        "capacity_exact_once",
        "capacity_exact_persistent",
        "full_exact_shared",
        "capacity_exact_no_gist",
        "full_original",
    }
    assert isinstance(_controller("full_original"), AceEventNativeOnePassController)
    for mode in sorted(ACEBENCH_VIEW_MODES - {"full_original"}):
        assert isinstance(_controller(mode), AceEventNativeExactController)
    with pytest.raises(ValueError, match="ACEBench view_mode"):
        _controller("static")
    packing, policy = contracts()
    with pytest.raises(ValueError, match="must be full_original"):
        AceEventNativeOnePassController(
            Tokenizer(), packing=packing, policy=policy, view_mode="static"
        )


def test_full_original_preserves_exact_messages_and_reports_classification() -> None:
    payload = _payload()
    prepared = _controller("full_original").prepare(
        payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    assert prepared.metadata["source_profile"] == "acebench-text-actions-v1"
    assert prepared.metadata["acebench_events"][2] == {
        "event_id": "ace/controller:m2",
        "kind": "tool_event",
        "complete": True,
        "source_indices": [2, 3],
        "submitted_call_count": 1,
    }
    store = build_ace_event_store(
        payload["session_id"], payload["messages"], payload["c2kv_ace_source"]
    )
    assert prepared.memory.raw_source_indices == tuple(range(len(store.messages)))


def test_append_receipt_expands_action_event_without_rewriting_prefix() -> None:
    controller = _controller("full_original")
    messages = _messages()[:3]
    first = _payload(decision_key="d1", messages=messages)
    first["c2kv_ace_source"]["receipts"] = []
    initial = controller.prepare(first, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS)
    assert initial.metadata["acebench_events"][2]["kind"] == (
        "acebench_execution_opaque"
    )

    later = _payload(decision_key="d2", messages=_messages()[:4])
    completed = controller.prepare(later, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS)
    assert completed.metadata["acebench_events"][2]["event_id"] == (
        initial.metadata["acebench_events"][2]["event_id"]
    )
    assert completed.metadata["acebench_events"][2]["kind"] == "tool_event"
    assert completed.metadata["acebench_events"][2]["complete"] is True


def test_receipt_mutation_is_part_of_monotone_prefix_identity() -> None:
    controller = _controller("full_original")
    controller.prepare(_payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS)
    changed = _payload(decision_key="d2", receipt=_receipt(count=2))
    with pytest.raises(PolicyInputError, match="truncated or rewritten"):
        controller.prepare(changed, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS)


def test_hidden_complete_aggregate_observation_recovers_exact_source() -> None:
    payload = _two_execution_payload()
    packing, policy = _activated_contracts(payload)
    controller = _controller(
        "capacity_exact_once", packing=packing, policy=policy
    )
    prepared = controller.prepare(
        payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    target_id = "ace/controller:m2"
    assert prepared.metadata["capacity_gate"]["activated"] is True
    assert target_id in prepared.memory.view.gist_event_ids
    assert not ({2, 3} & set(prepared.memory.raw_source_indices))

    draft = parse_ace_draft(
        "[Cancel(reservation_id='R-7')]", call_id_prefix="d1_r0"
    )
    result = controller.reconsider(
        prepared,
        list(draft.tool_calls),
        draft_text=draft.text,
    )
    assert result["regenerate"] is True
    assert result["decision"]["candidate_event_id"] == target_id
    assert result["decision"]["status"] == "gap"
    assert target_id in result["memory"].view.evidence_event_ids
    assert {2, 3} <= set(result["memory"].raw_source_indices)
    evidence_messages = [
        message.to_dict() for message in controller._sessions["ace/controller"]
        .decisions["d1"][1]._store.event_messages(target_id)
    ]
    assert evidence_messages == payload["messages"][2:4]


def test_same_receipt_group_as_opaque_is_ineligible_for_exact_recovery() -> None:
    payload = _payload(receipt=_receipt(count=2))
    store = build_ace_event_store(
        payload["session_id"], payload["messages"], payload["c2kv_ace_source"]
    )
    target = store.event("ace/controller:m2")
    assert target.complete is False
    draft = parse_ace_draft(
        "[Cancel(reservation_id='R-7')]", call_id_prefix="d1_r0"
    )
    decision = detect_exact_source_gap(
        store,
        visible_source_indices={0, 4},
        source_cutoff=len(store.messages),
        draft_tool_calls=list(draft.tool_calls),
    )
    assert decision.status == "abstain"
    assert decision.reason == "missing_source"
    assert decision.event_id is None


def test_opaque_execution_is_mandatory_raw_and_fails_closed_under_small_budget() -> None:
    packing, policy = contracts()
    packing.update(max_workspace_tokens=100, max_sequence_tokens=100)
    payload = _payload(receipt=_receipt(count=2))
    with pytest.raises(PackingBudgetError):
        build_acebench_controller(
            Tokenizer(), packing=packing, policy=policy,
            view_mode="capacity_exact_once",
        ).prepare(payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS)


def test_unknown_fields_and_missing_source_are_rejected_by_controller() -> None:
    payload = _payload()
    payload["target_action"] = "forbidden"
    with pytest.raises(PolicyInputError, match="privileged"):
        _controller("full_original").prepare(
            payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
        )
    payload = _payload()
    del payload["c2kv_ace_source"]
    with pytest.raises(PolicyInputError, match="version and receipts"):
        _controller("full_original").prepare(
            payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
        )
