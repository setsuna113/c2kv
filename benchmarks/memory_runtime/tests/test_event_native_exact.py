"""Pure-tokenizer contracts for event-native exact-source recovery."""

from __future__ import annotations

import copy
import math

import pytest

from benchmarks.memory_runtime import event_native_exact_policy as exact_policy_module
from benchmarks.memory_runtime.event_native_exact_policy import (
    EventNativeExactController,
)
from benchmarks.memory_runtime.policy import PolicyInputError
from benchmarks.memory_runtime.tests.test_event_native_policy import (
    Tokenizer,
    contracts,
)
from history_memory.events import EventStore
from history_memory.packing import MemoryView, PackingBudgetError, pack_memory


RATIO = 4
MAX_NEW_TOKENS = 2
TARGET_EVENT_ID = "exact/session:m1"
PROTECTED_EVENT_ID = "exact/session:m4"


def _messages() -> list[dict]:
    return [
        {"role": "system", "content": "Use only observable conversation records."},
        {
            "role": "user",
            "content": "Archived record item-17 contains " + "Z" * 1800,
        },
        {"role": "assistant", "content": "The archived record was noted."},
        {"role": "user", "content": "Check an unrelated clock value."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "clock-call",
                    "type": "function",
                    "function": {
                        "name": "clock",
                        "arguments": '{"city":"London"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "clock-call", "content": "09:30"},
        {"role": "assistant", "content": "The clock result was recorded."},
        {
            "role": "assistant",
            "content": "Unrelated archived notes: " + "Q" * 1400,
        },
        {"role": "user", "content": "Continue with the archived record."},
    ]


def _payload(*, decision_key: str = "d1", messages: list[dict] | None = None) -> dict:
    return {
        "session_id": "exact/session",
        "decision_key": decision_key,
        "messages": copy.deepcopy(messages if messages is not None else _messages()),
        "tools": [],
    }


def _draft_calls() -> list[dict]:
    return [
        {
            "id": "unsubmitted-use",
            "type": "function",
            "function": {
                "name": "use_record",
                "arguments": '{"record_id":"item-17"}',
            },
        },
        {
            "id": "unsubmitted-count",
            "type": "function",
            "function": {"name": "count", "arguments": '{"count":1}'},
        },
    ]


def _settings(
    *,
    history_budget_bytes: int | None = None,
    lease_decisions: int = 2,
    max_chunk_tokens: int = 96,
    max_sequence_tokens: int | None = None,
) -> tuple[dict, dict]:
    packing, policy = contracts()
    packing.update(
        max_chunk_tokens=max_chunk_tokens,
        chunk_overlap=min(8, max_chunk_tokens - 1),
        max_chunks=256,
        max_encoder_tokens=200000,
        max_workspace_tokens=200000,
        max_sequence_tokens=max_sequence_tokens or 200000,
    )
    policy.update(
        workspace_budget_bytes=10_000_000,
        lease_decisions=lease_decisions,
    )
    if history_budget_bytes is not None:
        policy["history_budget_bytes"] = history_budget_bytes
    return packing, policy


def _controller(
    mode: str,
    packing: dict,
    policy: dict,
    *,
    model_context: int | None = None,
) -> EventNativeExactController:
    return EventNativeExactController(
        Tokenizer(),
        packing=copy.deepcopy(packing),
        policy=copy.deepcopy(policy),
        mode=mode,
        model_context=model_context,
    )


def _activated_settings(
    *,
    lease_decisions: int = 2,
    max_chunk_tokens: int = 96,
) -> tuple[dict, dict]:
    packing, policy = _settings(
        lease_decisions=lease_decisions,
        max_chunk_tokens=max_chunk_tokens,
    )
    probe = _controller("capacity_exact_once", packing, policy).prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    full_bytes = probe.metadata["capacity_gate"]["full_history_bytes"]
    assert full_bytes > 0
    policy["history_budget_bytes"] = full_bytes - 1
    return packing, policy


def _prepare_activated(
    mode: str = "capacity_exact_once",
    *,
    lease_decisions: int = 2,
) -> tuple[EventNativeExactController, object, dict, dict]:
    packing, policy = _activated_settings(lease_decisions=lease_decisions)
    controller = _controller(mode, packing, policy)
    prepared = controller.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    assert prepared.metadata["capacity_gate"]["activated"] is True
    assert prepared.metadata["capacity_gate"]["full_identity_bypass"] is False
    return controller, prepared, packing, policy


def _reconsider(controller, prepared, calls=None):
    return controller.reconsider(
        prepared,
        copy.deepcopy(calls if calls is not None else _draft_calls()),
        draft_text="unsubmitted native draft",
    )


def _max_cost(metadata: dict, key: str) -> int:
    return max(row[key] for row in metadata["per_ratio"].values())


def test_unique_hidden_source_allows_one_same_decision_regeneration() -> None:
    controller, prepared, _, _ = _prepare_activated()

    assert prepared.metadata["pre_draft_retrieval"] is False
    assert prepared.metadata["decision_index"] == 1
    assert prepared.metadata["protected_event_ids"] == [PROTECTED_EVENT_ID]
    assert prepared.metadata["retrieved_event_ids"] == []
    assert TARGET_EVENT_ID in prepared.metadata["gist_event_ids"]
    assert set((1,)).isdisjoint(prepared.metadata["actual_visible_source_indices"])

    result = _reconsider(controller, prepared)

    assert result["regenerate"] is True
    assert result["decision"] == {
        **result["decision"],
        "status": "gap",
        "reason": "missing_unique_complete_source",
        "gap_type": "binding",
        "candidate_event_id": TARGET_EVENT_ID,
        "judges_action_correctness": False,
        "decision_index": 1,
        "upgrade_count": 1,
        "regeneration_allowed": True,
        "upgraded_event_id": TARGET_EVENT_ID,
    }
    assert result["metadata"]["decision_index"] == 1
    assert result["metadata"]["retrieved_event_ids"] == [TARGET_EVENT_ID]
    assert TARGET_EVENT_ID in result["metadata"]["evidence_event_ids"]
    assert TARGET_EVENT_ID in result["metadata"]["raw_event_ids"]
    assert TARGET_EVENT_ID not in result["metadata"]["gist_event_ids"]
    assert TARGET_EVENT_ID not in result["metadata"]["omitted_event_ids"]
    assert result["metadata"]["post_draft_exact_recovery_applied"] is True


def test_reconsider_is_idempotent_but_binds_the_complete_draft_identity() -> None:
    controller, prepared, _, _ = _prepare_activated()
    calls = _draft_calls()
    first = _reconsider(controller, prepared, calls)

    assert _reconsider(controller, prepared, copy.deepcopy(calls)) == first

    variants = []
    changed_id = copy.deepcopy(calls)
    changed_id[0]["id"] = "different-id"
    variants.append(changed_id)
    changed_name = copy.deepcopy(calls)
    changed_name[0]["function"]["name"] = "different_tool"
    variants.append(changed_name)
    changed_arguments = copy.deepcopy(calls)
    changed_arguments[0]["function"]["arguments"] = '{"record_id":"item-18"}'
    variants.append(changed_arguments)
    variants.append(list(reversed(copy.deepcopy(calls))))

    for different_draft in variants:
        with pytest.raises(PolicyInputError, match="second different draft"):
            _reconsider(controller, prepared, different_draft)


def test_privileged_request_is_rejected_without_advancing_state_and_no_source_abstains() -> None:
    packing, policy = _activated_settings()
    controller = _controller("capacity_exact_once", packing, policy)
    privileged = _payload()
    privileged["gold_action"] = {"record_id": "item-17"}

    with pytest.raises(PolicyInputError, match="privileged request fields"):
        controller.prepare(
            privileged, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
        )

    prepared = controller.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    missing = _draft_calls()
    missing[0]["function"]["arguments"] = '{"record_id":"never-recorded"}'
    result = _reconsider(controller, prepared, missing)

    assert prepared.metadata["decision_index"] == 1
    assert result["regenerate"] is False
    assert result["memory"] == prepared.memory
    assert result["decision"]["status"] == "abstain"
    assert result["decision"]["reason"] == "missing_source"
    assert result["decision"]["candidate_event_id"] is None
    assert result["decision"]["judges_action_correctness"] is False
    assert result["decision"]["upgrade_count"] == 0


def test_full_shared_and_no_gist_share_initial_e_but_use_actual_visibility() -> None:
    packing, policy = _activated_settings()
    full = _controller("full_exact_shared", packing, policy)
    no_gist = _controller("capacity_exact_no_gist", packing, policy)
    full_prepared = full.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    no_gist_prepared = no_gist.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )

    shared_keys = (
        "selected_event_ids",
        "protected_event_ids",
        "retrieved_event_ids",
        "retained_event_ids",
        "expired_lease_event_ids",
        "shared_evidence_event_ids",
    )
    for key in shared_keys:
        assert full_prepared.metadata[key] == no_gist_prepared.metadata[key]
    assert full_prepared.metadata["evidence_event_ids"] == [PROTECTED_EVENT_ID]
    assert no_gist_prepared.metadata["evidence_event_ids"] == [PROTECTED_EVENT_ID]
    assert full_prepared.metadata["omitted_event_ids"] == []
    assert full_prepared.metadata["actual_visible_source_indices"] == list(
        range(len(_messages()))
    )
    assert TARGET_EVENT_ID in no_gist_prepared.metadata["omitted_event_ids"]
    assert 1 not in no_gist_prepared.metadata["actual_visible_source_indices"]

    full_result = _reconsider(full, full_prepared)
    no_gist_result = _reconsider(no_gist, no_gist_prepared)

    assert full_result["regenerate"] is False
    assert full_result["decision"]["reason"] == "all_bindings_visible"
    assert no_gist_result["regenerate"] is True
    assert no_gist_result["decision"]["candidate_event_id"] == TARGET_EVENT_ID
    assert TARGET_EVENT_ID in no_gist_result["metadata"]["evidence_event_ids"]
    assert TARGET_EVENT_ID not in no_gist_result["metadata"]["omitted_event_ids"]


def test_persistent_upgrade_and_reentry_do_not_double_age_the_lease() -> None:
    controller, first, _, _ = _prepare_activated(
        "capacity_exact_persistent", lease_decisions=2
    )
    acquired = _reconsider(controller, first)
    assert acquired["regenerate"] is True
    assert acquired["metadata"]["selection"]["upgrade"][
        "lease_expires_at_decision"
    ] == 3

    second_payload = _payload(decision_key="d2")
    second = controller.prepare(
        second_payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    assert second.metadata["decision_index"] == 2
    assert second.metadata["retained_event_ids"] == [TARGET_EVENT_ID]
    assert controller.prepare(
        copy.deepcopy(second_payload), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    ) is second
    visible = _reconsider(controller, second)
    assert visible["regenerate"] is False
    assert visible["decision"]["reason"] == "all_bindings_visible"
    assert _reconsider(controller, second) == visible

    third = controller.prepare(
        _payload(decision_key="d3"),
        ratio=RATIO,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    assert third.metadata["decision_index"] == 3
    assert third.metadata["expired_lease_event_ids"] == [TARGET_EVENT_ID]
    assert TARGET_EVENT_ID not in third.metadata["retained_event_ids"]
    assert TARGET_EVENT_ID not in third.metadata["selected_event_ids"]


def test_injected_final_no_gist_render_failure_rolls_back_upgrade_and_lease(
    monkeypatch,
) -> None:
    packing, policy = _activated_settings()
    original_build = exact_policy_module.build_raw_control
    rendered_evidence: list[tuple[str, ...]] = []

    def fail_only_after_target_upgrade(*args, **kwargs):
        evidence = tuple(kwargs.get("evidence_event_ids", ()))
        rendered_evidence.append(evidence)
        if TARGET_EVENT_ID in evidence:
            raise PackingBudgetError("injected final raw rendering admission failure")
        return original_build(*args, **kwargs)

    # NoGist normally reflows R to fit a tighter cap. Inject at the final raw
    # renderer seam to isolate the transaction guarantee after selection passed.
    monkeypatch.setattr(
        exact_policy_module, "build_raw_control", fail_only_after_target_upgrade
    )
    controller = _controller("capacity_exact_no_gist", packing, policy)
    prepared = controller.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    result = _reconsider(controller, prepared)

    assert result["regenerate"] is False
    assert result["memory"] == prepared.memory
    assert result["decision"]["status"] == "abstain"
    assert result["decision"]["reason"] == "budget_exhausted"
    assert result["decision"]["candidate_event_id"] == TARGET_EVENT_ID
    assert result["decision"]["upgrade_count"] == 0
    assert result["metadata"]["selected_event_ids"] == prepared.metadata[
        "selected_event_ids"
    ]
    assert result["metadata"]["retrieved_event_ids"] == []
    assert result["metadata"]["post_draft_exact_recovery_applied"] is False
    assert rendered_evidence == [
        (PROTECTED_EVENT_ID,),
        (TARGET_EVENT_ID, PROTECTED_EVENT_ID),
    ]

    next_prepared = controller.prepare(
        _payload(decision_key="d2"),
        ratio=RATIO,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    assert next_prepared.metadata["decision_index"] == 2
    assert TARGET_EVENT_ID not in next_prepared.metadata["retained_event_ids"]
    assert TARGET_EVENT_ID not in next_prepared.metadata["selected_event_ids"]
    assert rendered_evidence[-1] == (PROTECTED_EVENT_ID,)


def test_gist_refill_packing_failure_has_stable_code_and_keeps_fallback(
    monkeypatch,
) -> None:
    packing, policy = _activated_settings()
    controller = _controller("capacity_exact_once", packing, policy)
    original_measure = controller._measure_view
    rejected: list[str] = []

    def reject_first_candidate(store, static_view, view, tools, max_new_tokens):
        if (
            view.gist_event_ids
            and view.evidence_event_ids == (PROTECTED_EVENT_ID,)
            and not rejected
        ):
            rejected.append(view.gist_event_ids[0])
            raise PackingBudgetError("injected candidate packing failure")
        return original_measure(store, static_view, view, tools, max_new_tokens)

    monkeypatch.setattr(controller, "_measure_view", reject_first_candidate)
    prepared = controller.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )

    assert rejected == [prepared.metadata["gist_refill_priority_event_ids"][0]]
    assert prepared.metadata["skipped_gist_refill_events"] == [
        {
            "event_id": rejected[0],
            "reasons": ["packing_budget_exceeded"],
            "detail": "injected candidate packing failure",
        }
    ]
    assert rejected[0] not in prepared.metadata["gist_refilled_event_ids"]
    assert prepared.metadata["gist_refilled_event_ids"]


def test_evidence_admission_evicts_gist_and_never_splits_event_chunks() -> None:
    packing, loose_policy = _activated_settings(max_chunk_tokens=64)
    loose = _controller("capacity_exact_once", packing, loose_policy)
    loose_initial = loose.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    loose_upgraded = _reconsider(loose, loose_initial)
    assert loose_upgraded["regenerate"] is True
    boundary = max(
        _max_cost(loose_initial.metadata, "history_bytes"),
        _max_cost(loose_upgraded["metadata"], "history_bytes"),
    )
    assert boundary < loose_initial.metadata["capacity_gate"]["full_history_bytes"]

    policy = copy.deepcopy(loose_policy)
    policy["history_budget_bytes"] = boundary
    controller = _controller("capacity_exact_once", packing, policy)
    prepared = controller.prepare(
        _payload(), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    target_chunks = [
        chunk for chunk in prepared.memory.chunks if chunk.event_id == TARGET_EVENT_ID
    ]
    assert len(target_chunks) > 1
    assert TARGET_EVENT_ID in prepared.metadata["gist_event_ids"]

    result = _reconsider(controller, prepared)
    assert result["regenerate"] is True
    upgraded = result["metadata"]
    pre_gist_tokens = sum(
        math.ceil(len(chunk.token_ids) / RATIO) for chunk in prepared.memory.chunks
    )
    hypothetical_all_gist_bytes = (
        upgraded["per_ratio"][str(RATIO)]["history_raw_tokens"]
        + pre_gist_tokens
    ) * policy["kv_bytes_per_token"]

    assert hypothetical_all_gist_bytes > policy["history_budget_bytes"]
    assert _max_cost(upgraded, "history_bytes") <= policy["history_budget_bytes"]
    assert set(prepared.metadata["gist_event_ids"]) - set(
        upgraded["gist_event_ids"]
    )
    assert TARGET_EVENT_ID in upgraded["evidence_event_ids"]
    assert TARGET_EVENT_ID not in upgraded["gist_event_ids"]
    assert TARGET_EVENT_ID not in upgraded["omitted_event_ids"]
    assert not any(
        chunk.event_id == TARGET_EVENT_ID for chunk in result["memory"].chunks
    )
    assert upgraded["atomic_packing_unit"] == "whole_event_all_encoder_chunks"


def test_full_identity_bypass_preserves_native_tokens_and_ticks_decisions() -> None:
    packing, policy = _settings()
    controller = _controller("full_exact_shared", packing, policy)
    payload = _payload()
    first = controller.prepare(
        payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    )
    store = EventStore.from_messages(payload["session_id"], payload["messages"])
    all_event_ids = tuple(event.event_id for event in store.events)
    expected = pack_memory(
        store,
        MemoryView(gist_event_ids=(), raw_event_ids=all_event_ids),
        Tokenizer(),
        tools=(),
    )

    assert first.metadata["capacity_gate"]["activated"] is False
    assert first.metadata["capacity_gate"]["full_identity_bypass"] is True
    assert first.metadata["decision_index"] == 1
    assert first.metadata["selected_event_ids"] == []
    assert first.metadata["evidence_event_ids"] == []
    assert first.memory.system_input_ids == expected.system_input_ids
    assert first.memory.workspace_input_ids == expected.workspace_input_ids
    assert first.memory.raw_source_indices == expected.raw_source_indices
    assert first.memory.chunks == expected.chunks == ()
    assert first.memory.view.evidence_event_ids == ()
    assert first.memory.raw_source_indices == tuple(range(len(payload["messages"])))
    assert controller.prepare(
        copy.deepcopy(payload), ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
    ) is first

    second = controller.prepare(
        _payload(decision_key="d2"),
        ratio=RATIO,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    assert second.metadata["capacity_gate"]["full_identity_bypass"] is True
    assert second.metadata["decision_index"] == 2
