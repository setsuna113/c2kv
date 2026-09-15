from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "python"))

from history_memory.events import EventStore  # noqa: E402
from memory_runtime.exact_policy import (  # noqa: E402
    EXACT_POLICY_VERSION,
    ExactRecoveryMemory,
)
from memory_runtime.policy import (  # noqa: E402
    BudgetExceeded,
    ConversationMemory,
    PolicyInputError,
    RuntimeConfig,
)


def _call(call_id: str, name: str = "read") -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": '{"id":"old-id"}'},
        }],
    }


def _result(call_id: str, content: str = "done") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _store(messages: list[dict], session_id: str = "s") -> EventStore:
    return EventStore.from_messages(session_id, messages)


def _cost(event_ids: tuple[str, ...]) -> int:
    return len(event_ids)


def _config(
    mode: str,
    *,
    budget: int = 8,
    lease_decisions: int = 3,
) -> RuntimeConfig:
    return RuntimeConfig(
        mode=mode,
        history_budget_bytes=budget,
        workspace_budget_bytes=budget,
        lease_decisions=lease_decisions,
        max_retrieved_events=1,
    )


def test_accepts_only_first_stage_exact_recovery_modes() -> None:
    ExactRecoveryMemory("s", _config("recover_once"))
    ExactRecoveryMemory("s", _config("persistent"))

    for mode in ("protect", "no_gist", "full_shared"):
        with pytest.raises(ValueError, match="recover_once or persistent"):
            ExactRecoveryMemory("s", _config(mode))


def test_initial_selection_composes_protect_without_pre_draft_retrieval() -> None:
    messages = [
        {"role": "user", "content": "old binding"},
        _call("c1"),
        _result("c1"),
        {"role": "user", "content": "current request"},
    ]
    store = _store(messages)
    visible = {"s:m3"}
    config = _config("recover_once")
    expected = ConversationMemory(
        "s", replace(config, mode="protect")
    ).prepare(store, _cost, visible, "d1")

    handle = ExactRecoveryMemory("s", config).prepare_decision(
        store, _cost, visible, "d1"
    )

    assert handle.selection.protected_event_ids == expected.protected_event_ids
    assert handle.selection.selected_event_ids == expected.selected_event_ids
    assert handle.selection.retrieved_event_ids == ()
    assert handle.selection.retained_event_ids == ()
    assert handle.selection.metadata["pre_draft_retrieval"] is False
    assert handle.selection.metadata["exact_policy_version"] == EXACT_POLICY_VERSION
    assert handle.store is store
    assert handle.visible_event_ids == frozenset(visible)
    assert handle.decision_index == 1


def test_upgrade_is_source_ordered_idempotent_and_limited_to_one_event() -> None:
    messages = [
        {"role": "user", "content": "older exact source"},
        _call("c1"),
        _result("c1"),
        {"role": "user", "content": "current request"},
    ]
    memory = ExactRecoveryMemory("s", _config("recover_once"))
    handle = memory.prepare_decision(_store(messages), _cost, {"s:m3"}, "d1")

    upgraded = memory.upgrade_decision(handle, "s:m0")
    repeated = memory.upgrade_decision(handle, "s:m0")

    assert repeated is upgraded
    assert upgraded.selected_event_ids == ("s:m0", "s:m1")
    assert upgraded.protected_event_ids == ("s:m1",)
    assert upgraded.retrieved_event_ids == ("s:m0",)
    assert upgraded.metadata["decision_index"] == handle.decision_index == 1
    assert upgraded.metadata["upgrade"] == {
        "status": "admitted",
        "event_id": "s:m0",
        "upgrade_count": 1,
        "decision_index": 1,
        "lease_expires_at_decision": None,
    }
    with pytest.raises(PolicyInputError, match="second event"):
        memory.upgrade_decision(handle, "s:m3")


def test_upgrade_requires_an_existing_complete_hidden_unselected_event() -> None:
    messages = [
        {"role": "user", "content": "visible source"},
        _call("pending"),
        {"role": "user", "content": "current request"},
    ]
    memory = ExactRecoveryMemory("s", _config("recover_once"))
    handle = memory.prepare_decision(
        _store(messages), _cost, {"s:m0", "s:m2"}, "d1"
    )

    with pytest.raises(PolicyInputError, match="complete"):
        memory.upgrade_decision(handle, "s:m1")
    with pytest.raises(PolicyInputError, match="raw visibility"):
        memory.upgrade_decision(handle, "s:m0")
    with pytest.raises(PolicyInputError, match="outside the observable prefix"):
        memory.upgrade_decision(handle, "s:m99")


def test_budget_failure_does_not_consume_or_pollute_the_upgrade() -> None:
    messages = [
        {"role": "user", "content": "too expensive"},
        {"role": "user", "content": "fits"},
        {"role": "user", "content": "current"},
    ]
    memory = ExactRecoveryMemory("s", _config("persistent", budget=1))

    def nonuniform_cost(event_ids: tuple[str, ...]) -> int:
        return 2 if "s:m0" in event_ids else len(event_ids)

    handle = memory.prepare_decision(
        _store(messages), nonuniform_cost, {"s:m2"}, "d1"
    )
    with pytest.raises(BudgetExceeded) as error:
        memory.upgrade_decision(handle, "s:m0")
    assert error.value.required_event_ids == ("s:m0",)
    assert error.value.required_cost == 2
    assert handle.selection.metadata["upgrade"]["status"] == "not_requested"

    admitted = memory.upgrade_decision(handle, "s:m1")
    assert admitted.retrieved_event_ids == ("s:m1",)
    assert admitted.metadata["upgrade"]["lease_expires_at_decision"] == 4


def test_duplicate_key_owner_and_stale_handle_guards() -> None:
    messages = [
        {"role": "user", "content": "old"},
        {"role": "user", "content": "current"},
    ]
    memory = ExactRecoveryMemory("s", _config("recover_once"))
    store = _store(messages)
    handle = memory.prepare_decision(store, _cost, {"s:m1"}, "same")
    repeated = memory.prepare_decision(
        store, lambda _: 999, {"s:m1"}, "same"
    )
    assert repeated is handle
    assert repeated.decision_index == 1

    changed = _store(messages + [{"role": "assistant", "content": "changed"}])
    with pytest.raises(PolicyInputError, match="reused"):
        memory.prepare_decision(changed, _cost, {"s:m1"}, "same")

    other = ExactRecoveryMemory("s", _config("recover_once"))
    with pytest.raises(PolicyInputError, match="different memory"):
        other.upgrade_decision(handle, "s:m0")

    next_handle = memory.prepare_decision(
        changed, _cost, {event.event_id for event in changed.events}, "next"
    )
    assert next_handle.decision_index == 2
    with pytest.raises(PolicyInputError, match="stale"):
        memory.upgrade_decision(handle, "s:m0")


def test_once_and_persistent_share_admission_then_only_persistent_retains() -> None:
    first_messages = [
        {"role": "user", "content": "exact old source"},
        {"role": "user", "content": "current request"},
    ]
    once = ExactRecoveryMemory("s", _config("recover_once"))
    persistent = ExactRecoveryMemory("s", _config("persistent"))
    once_handle = once.prepare_decision(
        _store(first_messages), _cost, {"s:m1"}, "d1"
    )
    persistent_handle = persistent.prepare_decision(
        _store(first_messages), _cost, {"s:m1"}, "d1"
    )

    once_upgrade = once.upgrade_decision(once_handle, "s:m0")
    persistent_upgrade = persistent.upgrade_decision(persistent_handle, "s:m0")
    assert (
        once_handle.selection.protected_event_ids,
        once_handle.selection.retrieved_event_ids,
        once_handle.selection.retained_event_ids,
        once_handle.selection.selected_event_ids,
        once_handle.selection.metadata["selected_cost_bytes"],
    ) == (
        persistent_handle.selection.protected_event_ids,
        persistent_handle.selection.retrieved_event_ids,
        persistent_handle.selection.retained_event_ids,
        persistent_handle.selection.selected_event_ids,
        persistent_handle.selection.metadata["selected_cost_bytes"],
    )
    assert once_upgrade.selected_event_ids == persistent_upgrade.selected_event_ids
    assert once_upgrade.retrieved_event_ids == persistent_upgrade.retrieved_event_ids

    second_messages = first_messages + [
        {"role": "assistant", "content": "first final"},
        {"role": "user", "content": "continue"},
    ]
    second_store = _store(second_messages)
    visible = {"s:m1", "s:m2", "s:m3"}
    once_second = once.prepare_decision(second_store, _cost, visible, "d2")
    persistent_second = persistent.prepare_decision(
        second_store, _cost, visible, "d2"
    )

    assert once_second.selection.retained_event_ids == ()
    assert "s:m0" not in once_second.selection.selected_event_ids
    assert persistent_second.selection.retained_event_ids == ("s:m0",)
    assert "s:m0" in persistent_second.selection.selected_event_ids


def test_full_visible_decisions_tick_and_expire_without_duplicate_or_refresh() -> None:
    messages = [
        {"role": "user", "content": "leased source"},
        {"role": "user", "content": "current request"},
    ]
    memory = ExactRecoveryMemory(
        "s", _config("persistent", lease_decisions=3)
    )
    first = memory.prepare_decision(_store(messages), _cost, {"s:m1"}, "d1")
    upgraded = memory.upgrade_decision(first, "s:m0")
    assert upgraded.metadata["upgrade"]["lease_expires_at_decision"] == 4

    handles = []
    for decision in range(2, 5):
        messages = messages + [{
            "role": "assistant", "content": f"full-visible decision {decision}"
        }]
        store = _store(messages)
        visible = {event.event_id for event in store.events}
        handle = memory.prepare_decision(store, _cost, visible, f"d{decision}")
        handles.append(handle)
        assert handle.decision_index == decision
        assert "s:m0" not in handle.selection.selected_event_ids
        assert handle.selection.retained_event_ids == ()
        if decision == 2:
            assert memory.prepare_decision(
                store, lambda _: 999, visible, "d2"
            ) is handle
        if decision < 4:
            assert handle.selection.metadata["expired_lease_event_ids"] == ()

    assert handles[-1].selection.metadata["expired_lease_event_ids"] == ("s:m0",)


def test_explicit_revision_clears_exact_lease() -> None:
    first_messages = [
        {"role": "user", "content": "old invoice INV-1"},
        {"role": "user", "content": "use it"},
    ]
    memory = ExactRecoveryMemory(
        "s", _config("persistent", lease_decisions=5)
    )
    first = memory.prepare_decision(
        _store(first_messages), _cost, {"s:m1"}, "d1"
    )
    memory.upgrade_decision(first, "s:m0")

    revised_messages = first_messages + [
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "Actually, change to a new invoice."},
    ]
    revised = memory.prepare_decision(
        _store(revised_messages), _cost, {"s:m1", "s:m2", "s:m3"}, "d2"
    )

    assert revised.selection.metadata["revision_cancelled_event_ids"] == ("s:m0",)
    assert revised.selection.retained_event_ids == ()
    assert "s:m0" not in revised.selection.selected_event_ids


def test_persistent_retention_is_after_protection_and_can_be_skipped() -> None:
    first_messages = [
        {"role": "user", "content": "leased source"},
        {"role": "user", "content": "current request"},
    ]
    memory = ExactRecoveryMemory("s", _config("persistent", budget=1))
    first = memory.prepare_decision(
        _store(first_messages), _cost, {"s:m1"}, "d1"
    )
    memory.upgrade_decision(first, "s:m0")

    second_messages = first_messages + [
        _call("new"),
        _result("new"),
        {"role": "user", "content": "continue"},
    ]
    second = memory.prepare_decision(
        _store(second_messages), _cost, {"s:m1", "s:m4"}, "d2"
    )

    assert second.selection.protected_event_ids == ("s:m2",)
    assert second.selection.selected_event_ids == ("s:m2",)
    assert second.selection.retained_event_ids == ()
    assert second.selection.metadata["skipped_for_budget"] == ({
        "event_id": "s:m0",
        "reason": "active_exact_lease",
        "candidate_cost_bytes": 2,
    },)
