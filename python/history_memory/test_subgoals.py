"""CPU-only contracts for observable subgoal lifecycle and source grouping."""

from __future__ import annotations

import pytest

from history_memory.events import EventStore
from history_memory.subgoals import (
    ACTIVE,
    CLOSED_BY_NEXT_DECLARATION,
    CLOSED_BY_USER_TURN,
    UNASSIGNED,
    build_subgoal_ledger,
    group_subgoal_history,
    group_subgoal_sources,
)


def call(call_id: str, name: str = "lookup") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


def result(call_id: str, content: str = '{"success":true}') -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_next_declaration_closes_previous_without_claiming_semantic_success():
    first_prefix = [
        {"role": "user", "content": "Complete the request."},
        {
            "role": "assistant",
            "content": "Subgoal: inspect account",
            "tool_calls": [call("c1")],
        },
        result("c1"),
    ]
    first_store = EventStore.from_messages("task-a", first_prefix)
    first = build_subgoal_ledger(first_store).records[0]
    assert first.lifecycle_status == ACTIVE
    assert first.archivable is False
    assert first.event_ids == ("task-a:m1",)
    assert first.source_indices == (1, 2)

    second_prefix = first_prefix + [
        {
            "role": "assistant",
            "content": "Subgoal: submit result",
            "tool_calls": [call("c2", "submit")],
        },
        result("c2"),
    ]
    second_store = EventStore.from_messages("task-a", second_prefix)
    ledger = build_subgoal_ledger(second_store)
    closed, active = ledger.records
    assert closed.subgoal_id == first.subgoal_id == "task-a:sg1"
    assert closed.event_ids == first.event_ids
    assert closed.source_indices == first.source_indices
    assert closed.lifecycle_status == CLOSED_BY_NEXT_DECLARATION
    assert closed.controller_completion_claimed is True
    assert closed.metadata()["semantic_completion"] == "unknown"
    assert closed.archivable is True
    assert active.lifecycle_status == ACTIVE and active.archivable is False

    extended = EventStore.from_messages(
        "task-a",
        second_prefix + [{"role": "assistant", "content": "waiting"}],
    )
    assert build_subgoal_ledger(extended).records[0] == closed


def test_user_turn_is_a_nonsemantic_boundary_and_not_supersession():
    messages = [
        {"role": "user", "content": "Inspect the account."},
        {
            "role": "assistant",
            "content": "Subgoal: inspect account",
            "tool_calls": [call("c1")],
        },
        result("c1"),
        {"role": "user", "content": "Also check the balance."},
        {
            "role": "assistant",
            "content": "Subgoal: check balance",
            "tool_calls": [call("c2")],
        },
        result("c2"),
    ]
    ledger = build_subgoal_ledger(EventStore.from_messages("task-b", messages))
    prior, current = ledger.records
    assert prior.lifecycle_status == CLOSED_BY_USER_TURN
    assert prior.boundary_event_id == "task-b:m3"
    assert prior.controller_completion_claimed is False
    assert prior.metadata()["semantic_completion"] == "unknown"
    assert "supersed" not in prior.lifecycle_status
    assert current.goal_event_id == "task-b:m3"
    assert current.lifecycle_status == ACTIVE


def test_pending_tool_event_is_never_archivable_or_packable():
    messages = [
        {"role": "user", "content": "Do both steps."},
        {
            "role": "assistant",
            "content": "Subgoal: first step",
            "tool_calls": [call("pending")],
        },
        {
            "role": "assistant",
            "content": "Subgoal: second step",
            "tool_calls": [call("c2")],
        },
        result("c2"),
    ]
    store = EventStore.from_messages("task-c", messages)
    first, second = build_subgoal_ledger(store).records
    assert first.lifecycle_status == CLOSED_BY_NEXT_DECLARATION
    assert first.pending_event_ids == ("task-c:m1",)
    assert first.archivable is False
    assert second.lifecycle_status == ACTIVE
    with pytest.raises(ValueError, match="Pending tool events cannot be packed"):
        group_subgoal_sources(store, [1])


def test_active_completed_prefix_can_be_grouped_for_always_compress():
    messages = [
        {"role": "system", "content": "Use tools."},
        {"role": "user", "content": "Inspect two records."},
        {
            "role": "assistant",
            "content": "Subgoal: inspect records",
            "tool_calls": [call("c1")],
        },
        result("c1", '{"record":"alpha"}'),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [call("c2")],
        },
        result("c2", '{"record":"beta"}'),
    ]
    store = EventStore.from_messages("task-d", messages)
    groups = group_subgoal_sources(store, [2, 3, 4, 5])
    assert len(groups) == 1
    group = groups[0]
    assert group.lifecycle_status == ACTIVE
    assert group.archivable is False
    assert group.source_indices == (2, 3, 4, 5)
    assert group.event_ids == ("task-d:m2", "task-d:m4")
    assert [message["content"] for message in group.source_messages(store)] == [
        "Subgoal: inspect records",
        '{"record":"alpha"}',
        None,
        '{"record":"beta"}',
    ]
    assert group.metadata()["source_identity"] == "original_event_store_indices"

    history = group_subgoal_history(store, 6, [2, 3, 4, 5])
    assert history == groups
    with pytest.raises(ValueError, match="strictly before source_cutoff"):
        group_subgoal_history(store, 5, [2, 3, 4, 5])


@pytest.mark.parametrize(
    "content,tool_calls,reason",
    [
        ("Subgoal: text only", None, "subgoal_declaration_requires_native_tool_call"),
        ("Subgoal: ", [call("c1")], "subgoal_label_is_empty"),
        (
            "Subgoal: first line\nextra prose",
            [call("c1")],
            "subgoal_content_must_be_one_line",
        ),
    ],
)
def test_invalid_subgoal_candidates_do_not_create_boundaries(content, tool_calls, reason):
    message = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    messages = [{"role": "user", "content": "Do it."}, message]
    if tool_calls:
        messages.append(result(tool_calls[0]["id"]))
    store = EventStore.from_messages("task-e", messages)
    ledger = build_subgoal_ledger(store)
    assert ledger.records == ()
    assert [violation.reason for violation in ledger.violations] == [reason]


def test_grouping_preserves_exact_source_order_and_unassigned_events():
    messages = [
        {"role": "system", "content": "System."},
        {"role": "user", "content": "First request."},
        {"role": "assistant", "content": "ordinary preamble"},
        {
            "role": "assistant",
            "content": "Subgoal: inspect",
            "tool_calls": [call("c1")],
        },
        result("c1"),
        {
            "role": "assistant",
            "content": "Subgoal: finish",
            "tool_calls": [call("c2")],
        },
        result("c2"),
    ]
    store = EventStore.from_messages("task-f", messages)
    selected = [1, 2, 3, 4, 5, 6]
    groups = group_subgoal_sources(store, selected)
    assert tuple(index for group in groups for index in group.source_indices) == tuple(selected)
    assert [group.lifecycle_status for group in groups] == [
        UNASSIGNED,
        UNASSIGNED,
        CLOSED_BY_NEXT_DECLARATION,
        ACTIVE,
    ]
    assert groups[2].source_indices == (3, 4)
    assert groups[3].source_indices == (5, 6)
    with pytest.raises(ValueError, match="strictly increasing"):
        group_subgoal_sources(store, [2, 1])
    with pytest.raises(ValueError, match="strictly increasing"):
        group_subgoal_sources(store, [1, 1])


def test_parallel_results_retain_event_and_source_identity():
    messages = [
        {"role": "user", "content": "Look up both."},
        {
            "role": "assistant",
            "content": "Subgoal: parallel lookup",
            "tool_calls": [call("a"), call("b")],
        },
        result("b", "beta"),
        result("a", "alpha"),
    ]
    store = EventStore.from_messages("task-g", messages)
    record = build_subgoal_ledger(store).records[0]
    assert record.event_ids == ("task-g:m1",)
    assert record.source_indices == (1, 2, 3)
    group = group_subgoal_sources(store, [1, 2, 3])[0]
    assert group.source_indices == (1, 2, 3)
    assert [message.get("tool_call_id") for message in group.source_messages(store)] == [
        None,
        "b",
        "a",
    ]
