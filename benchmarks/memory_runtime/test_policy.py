from __future__ import annotations

import sys
from pathlib import Path

import pytest


RUNTIME_ROOT = Path(__file__).resolve().parents[2]
HISTORY_PYTHON = RUNTIME_ROOT / "python"
_ADDED_HISTORY_PATH = False
if str(HISTORY_PYTHON) not in sys.path:
    sys.path.insert(0, str(HISTORY_PYTHON))
    _ADDED_HISTORY_PATH = True

try:
    from history_memory.events import EventStore  # noqa: E402

    from .policy import (  # noqa: E402
        BudgetExceeded,
        ConversationMemory,
        PolicyInputError,
        RuntimeConfig,
    )
finally:
    if _ADDED_HISTORY_PATH:
        sys.path.remove(str(HISTORY_PYTHON))


def _call(call_id: str, name: str, arguments: str) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _result(call_id: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _store(session_id: str, messages: list[dict]) -> EventStore:
    return EventStore.from_messages(session_id, messages)


def _count_cost(event_ids: tuple[str, ...]) -> int:
    return len(event_ids)


def _retrieval_trace(final_request: str = "Use receipt INV-2026-0042 again") -> list[dict]:
    return [
        {"role": "user", "content": "Fetch invoice INV-2026-0042"},
        _call("old", "get_invoice", '{"invoice_id":"INV-2026-0042"}'),
        _result("old", "INV-2026-0042 total is GBP 81"),
        {"role": "assistant", "content": "Recorded."},
        _call("new", "get_clock", '{"city":"London"}'),
        _result("new", "09:30"),
        {"role": "user", "content": final_request},
    ]


def test_completed_tool_event_is_selected_as_one_bound_event() -> None:
    store = _store("s", _retrieval_trace())
    memory = ConversationMemory(
        "s",
        RuntimeConfig(
            mode="recover_once",
            history_budget_bytes=4,
            workspace_budget_bytes=4,
            max_retrieved_events=1,
        ),
    )

    selection = memory.prepare(store, _count_cost, set(), "d1")

    old_tool = "s:m1"
    assert old_tool in selection.retrieved_event_ids
    assert store.event(old_tool).source_indices == (1, 2)
    assert store.event(old_tool).tool_call_ids == ("old",)
    assert all(
        store.event(event_id).complete for event_id in selection.selected_event_ids
    )


def test_recent_tool_arguments_retrieve_an_older_direct_source() -> None:
    store = _store(
        "s",
        [
            {"role": "user", "content": "Load the customer record."},
            _call("old", "get_customer", '{"customer":"客户甲"}'),
            _result("old", "客户甲 credit limit is 50"),
            {"role": "user", "content": "Continue."},
            _call("new", "make_order", '{"customer":"客户甲"}'),
            _result("new", "Order prepared"),
        ],
    )
    selection = ConversationMemory(
        "s",
        RuntimeConfig(
            mode="recover_once",
            history_budget_bytes=3,
            workspace_budget_bytes=3,
            max_retrieved_events=1,
        ),
    ).prepare(store, _count_cost, set(), "d1")

    assert selection.retrieved_event_ids == ("s:m1",)


def test_fully_raw_visible_pending_tool_context_is_not_duplicated() -> None:
    store = _store(
        "s",
        [
            {"role": "user", "content": "Run it"},
            _call("pending", "write_file", '{"path":"a.txt"}'),
        ],
    )
    memory = ConversationMemory(
        "s", RuntimeConfig(history_budget_bytes=2, workspace_budget_bytes=2)
    )

    selection = memory.prepare(store, _count_cost, {"s:m0", "s:m1"}, "d1")

    assert selection.selected_event_ids == ()
    assert selection.metadata["raw_pending_event_ids"] == ("s:m1",)


def test_nonvisible_pending_tool_context_is_mandatory() -> None:
    store = _store(
        "s",
        [
            {"role": "user", "content": "Run it"},
            _call("pending", "write_file", '{"path":"a.txt"}'),
        ],
    )
    selection = ConversationMemory(
        "s", RuntimeConfig(history_budget_bytes=1, workspace_budget_bytes=1)
    ).prepare(store, _count_cost, {"s:m0"}, "d1")

    assert selection.selected_event_ids == ("s:m1",)
    assert selection.metadata["mandatory_event_ids"] == ("s:m1",)


def test_full_shared_current_user_and_pending_call_are_identity_raw() -> None:
    store = _store(
        "s",
        [
            {"role": "user", "content": "Run it"},
            _call("pending", "write_file", '{"path":"a.txt"}'),
        ],
    )
    selection = ConversationMemory(
        "s",
        RuntimeConfig(
            mode="full_shared", history_budget_bytes=0, workspace_budget_bytes=0
        ),
    ).prepare(store, _count_cost, {"s:m0", "s:m1"}, "d1")

    assert selection.selected_event_ids == ()
    assert selection.metadata["selected_cost_bytes"] == 0


def test_large_recent_completed_tool_is_skipped_instead_of_failing() -> None:
    store = _store(
        "s",
        [
            _call("large", "read", "{}"),
            _result("large", "large observation"),
            {"role": "user", "content": "continue"},
        ],
    )

    def cost(event_ids: tuple[str, ...]) -> int:
        return sum(10 if event_id == "s:m0" else 1 for event_id in event_ids)

    selection = ConversationMemory(
        "s", RuntimeConfig(history_budget_bytes=2, workspace_budget_bytes=2)
    ).prepare(store, cost, set(), "d1")

    assert selection.selected_event_ids == ("s:m2",)
    assert selection.metadata["skipped_for_budget"] == (
        {
            "event_id": "s:m0",
            "reason": "recent_complete_tool",
            "candidate_cost_bytes": 11,
        },
    )


def test_mandatory_events_raise_with_whole_set_cost() -> None:
    store = _store(
        "s",
        [
            {"role": "user", "content": "Run it"},
            _call("pending", "write_file", '{"path":"a.txt"}'),
        ],
    )
    calls: list[tuple[str, ...]] = []

    def nonadditive_cost(event_ids: tuple[str, ...]) -> int:
        calls.append(event_ids)
        return 9 if len(event_ids) == 2 else len(event_ids)

    with pytest.raises(BudgetExceeded) as error:
        ConversationMemory(
            "s", RuntimeConfig(history_budget_bytes=8, workspace_budget_bytes=8)
        ).prepare(store, nonadditive_cost, set(), "d1")

    assert error.value.required_event_ids == ("s:m0", "s:m1")
    assert error.value.required_cost == 9
    assert calls == [("s:m0", "s:m1")]


def test_persistent_lease_continues_then_expires_without_replay() -> None:
    config = RuntimeConfig(
        mode="persistent",
        history_budget_bytes=4,
        workspace_budget_bytes=4,
        lease_decisions=3,
        max_retrieved_events=1,
    )
    memory = ConversationMemory("s", config)
    messages = _retrieval_trace()
    first = memory.prepare(_store("s", messages), _count_cost, set(), "d1")
    assert first.retrieved_event_ids == ("s:m1",)

    messages += [
        {"role": "assistant", "content": "I have the receipt."},
        {"role": "user", "content": "Continue."},
    ]
    second = memory.prepare(
        _store("s", messages), _count_cost, {"s:m6", "s:m7", "s:m8"}, "d2"
    )
    assert second.retrieved_event_ids == ()
    assert second.retained_event_ids == ("s:m1",)

    messages += [{"role": "assistant", "content": "Still working."}]
    third = memory.prepare(
        _store("s", messages), _count_cost, {"s:m8", "s:m9"}, "d3"
    )
    assert third.retained_event_ids == ("s:m1",)

    messages += [{"role": "assistant", "content": "Next decision."}]
    fourth = memory.prepare(
        _store("s", messages), _count_cost, {"s:m8", "s:m10"}, "d4"
    )
    assert "s:m1" not in fourth.selected_event_ids
    assert fourth.metadata["expired_lease_event_ids"] == ("s:m1",)


def test_explicit_user_revision_cancels_an_active_lease() -> None:
    memory = ConversationMemory(
        "s",
        RuntimeConfig(
            mode="persistent",
            history_budget_bytes=4,
            workspace_budget_bytes=4,
            lease_decisions=5,
            max_retrieved_events=1,
        ),
    )
    messages = _retrieval_trace()
    memory.prepare(_store("s", messages), _count_cost, set(), "d1")
    messages += [{"role": "assistant", "content": "Okay."}]
    messages += [{"role": "user", "content": "Actually, change to a new invoice."}]

    revised = memory.prepare(_store("s", messages), _count_cost, set(), "d2")

    assert "s:m1" not in revised.retained_event_ids
    assert revised.metadata["revision_cancelled_event_ids"] == ("s:m1",)


def test_recover_once_does_not_retain_evidence() -> None:
    memory = ConversationMemory(
        "s",
        RuntimeConfig(
            mode="recover_once",
            history_budget_bytes=4,
            workspace_budget_bytes=4,
            max_retrieved_events=1,
        ),
    )
    messages = _retrieval_trace()
    first = memory.prepare(_store("s", messages), _count_cost, set(), "d1")
    assert first.retrieved_event_ids == ("s:m1",)
    messages += [
        {"role": "assistant", "content": "I used the receipt."},
        {"role": "user", "content": "Continue."},
    ]

    second = memory.prepare(
        _store("s", messages), _count_cost, {"s:m6", "s:m7", "s:m8"}, "d2"
    )

    assert second.retained_event_ids == ()
    assert "s:m1" not in second.selected_event_ids


def test_same_opening_sessions_are_strictly_isolated() -> None:
    messages = [{"role": "user", "content": "same opening"}]
    memory = ConversationMemory("task-a")
    a = memory.prepare(_store("task-a", messages), _count_cost, set(), "d1")
    assert a.selected_event_ids == ("task-a:m0",)

    with pytest.raises(PolicyInputError, match="Session mismatch"):
        memory.prepare(_store("task-b", messages), _count_cost, set(), "d2")


def test_repeated_decision_is_idempotent_and_does_not_age_lease() -> None:
    memory = ConversationMemory(
        "s",
        RuntimeConfig(
            mode="persistent",
            history_budget_bytes=4,
            workspace_budget_bytes=4,
            lease_decisions=2,
            max_retrieved_events=1,
        ),
    )
    store = _store("s", _retrieval_trace())
    first = memory.prepare(store, _count_cost, set(), "same")
    again = memory.prepare(store, lambda _: 999, set(), "same")
    assert again is first
    assert again.metadata["decision_index"] == 1

    messages = _retrieval_trace() + [{"role": "assistant", "content": "next"}]
    next_selection = memory.prepare(
        _store("s", messages), _count_cost, {"s:m6", "s:m7"}, "next"
    )
    assert next_selection.retained_event_ids == ("s:m1",)


def test_new_direct_reference_refreshes_a_selected_lease() -> None:
    memory = ConversationMemory(
        "s",
        RuntimeConfig(
            mode="persistent",
            history_budget_bytes=4,
            workspace_budget_bytes=4,
            lease_decisions=2,
            max_retrieved_events=1,
        ),
    )
    messages = _retrieval_trace()
    memory.prepare(_store("s", messages), _count_cost, set(), "d1")
    messages += [
        {"role": "assistant", "content": "Okay."},
        {"role": "user", "content": "Use INV-2026-0042 once more."},
    ]

    refreshed = memory.prepare(
        _store("s", messages), _count_cost, {"s:m6", "s:m7", "s:m8"}, "d2"
    )

    assert refreshed.retained_event_ids == ("s:m1",)
    assert "s:m1" in refreshed.metadata["direct_source_candidate_ids"]
    messages += [
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Continue."},
    ]
    third = memory.prepare(
        _store("s", messages), _count_cost, {"s:m8", "s:m9", "s:m10"}, "d3"
    )
    assert third.retained_event_ids == ("s:m1",)


def test_reused_decision_key_with_changed_input_is_rejected() -> None:
    memory = ConversationMemory("s")
    messages = [{"role": "user", "content": "one"}]
    memory.prepare(_store("s", messages), _count_cost, set(), "d1")
    messages += [{"role": "assistant", "content": "two"}]

    with pytest.raises(PolicyInputError, match="reused"):
        memory.prepare(_store("s", messages), _count_cost, set(), "d1")


def test_history_rewrite_and_shrink_are_rejected() -> None:
    memory = ConversationMemory("s")
    messages = [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
    ]
    memory.prepare(_store("s", messages), _count_cost, set(), "d1")

    with pytest.raises(PolicyInputError, match="truncated or rewritten"):
        memory.prepare(_store("s", messages[:1]), _count_cost, set(), "d2")
    rewritten = [dict(message) for message in messages]
    rewritten[0]["content"] = "changed"
    with pytest.raises(PolicyInputError, match="truncated or rewritten"):
        memory.prepare(_store("s", rewritten), _count_cost, set(), "d3")


def test_no_gist_uses_history_budget_but_never_adds_budgets() -> None:
    store = _store(
        "s",
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "middle"},
            {"role": "user", "content": "current"},
        ],
    )
    selection = ConversationMemory(
        "s",
        RuntimeConfig(
            mode="no_gist", history_budget_bytes=2, workspace_budget_bytes=10
        ),
    ).prepare(store, _count_cost, set(), "d1")

    assert selection.metadata["budget_bytes"] == 2
    assert selection.selected_event_ids == ("s:m1", "s:m2")
    assert any(
        item["event_id"] == "s:m0" and item["reason"] == "no_gist_history"
        for item in selection.metadata["skipped_for_budget"]
    )


def test_no_gist_reuses_retrieval_and_persistent_lease_controller() -> None:
    memory = ConversationMemory(
        "s",
        RuntimeConfig(
            mode="no_gist",
            history_budget_bytes=3,
            workspace_budget_bytes=99,
            lease_decisions=3,
            max_retrieved_events=1,
        ),
    )
    messages = _retrieval_trace()
    first = memory.prepare(_store("s", messages), _count_cost, set(), "d1")
    assert first.retrieved_event_ids == ("s:m1",)
    messages += [
        {"role": "assistant", "content": "Receipt loaded."},
        {"role": "user", "content": "Continue."},
    ]

    second = memory.prepare(
        _store("s", messages), _count_cost, {"s:m6", "s:m7", "s:m8"}, "d2"
    )

    assert second.retained_event_ids == ("s:m1",)
    assert second.metadata["budget_bytes"] == 3


def test_workspace_modes_also_respect_the_total_history_budget() -> None:
    store = _store("s", [{"role": "user", "content": "current"}])
    with pytest.raises(BudgetExceeded) as error:
        ConversationMemory(
            "s",
            RuntimeConfig(
                mode="protect", history_budget_bytes=0, workspace_budget_bytes=1
            ),
        ).prepare(store, _count_cost, set(), "d1")

    assert error.value.budget == 0


def test_full_shared_delegates_history_and_only_protects_workspace() -> None:
    store = _store(
        "s",
        [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "middle"},
            {"role": "user", "content": "current"},
        ],
    )
    selection = ConversationMemory(
        "s",
        RuntimeConfig(
            mode="full_shared", history_budget_bytes=20, workspace_budget_bytes=1
        ),
    ).prepare(store, _count_cost, {"s:m0", "s:m1"}, "d1")

    assert selection.selected_event_ids == ("s:m2",)
    assert selection.metadata["budget_bytes"] == 1
    assert selection.metadata["full_shared_history_delegated"] is True


def test_gold_and_target_inputs_are_rejected_and_metadata_is_ignored() -> None:
    messages = _retrieval_trace("Continue")
    messages[-1]["gold_event_id"] = "s:m1"
    store = _store("s", messages)
    memory = ConversationMemory(
        "s",
        RuntimeConfig(
            mode="recover_once",
            history_budget_bytes=3,
            workspace_budget_bytes=3,
        ),
    )
    selection = memory.prepare(store, _count_cost, set(), "d1")
    assert "s:m1" not in selection.retrieved_event_ids

    other = ConversationMemory("s")
    with pytest.raises(PolicyInputError, match="forbidden"):
        other.prepare(
            store,
            _count_cost,
            set(),
            "d1",
            gold_event_ids=("s:m1",),
            target_action="submit",
        )
