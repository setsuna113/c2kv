"""Contracts for AppWorld code-action history in the CUDA delivery runtime."""

from types import SimpleNamespace

from history_memory.events import EventStore
from memory_runtime.recovery.set_protocol import context_from_prepared


def _messages():
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Find the contact."},
        {"role": "assistant", "content": "contacts = app.contacts.list()"},
        {"role": "user", "content": "Execution result: contact_id=old"},
        {"role": "assistant", "content": "print(contact_id)"},
        {"role": "user", "content": "Execution result: contact_id=new"},
    ]


def _prepared(store):
    return SimpleNamespace(
        _store=store,
        memory=SimpleNamespace(
            raw_source_indices=(),
            view=SimpleNamespace(raw_event_ids=()),
        ),
        metadata={"decision_key": "turn-0/step-2"},
    )


def test_appworld_pairs_preserve_sources_without_native_tool_calls():
    messages = _messages()
    store = EventStore.from_messages(
        "acon/task/attempt-0", messages, benchmark="acon_appworld"
    )

    assert [event.kind for event in store.events] == [
        "instruction", "user", "tool_event", "tool_event"
    ]
    assert [event.source_indices for event in store.events] == [
        (0,), (1,), (2, 3), (4, 5)
    ]
    assert all(not event.tool_call_ids for event in store.events[2:])


def test_appworld_context_uses_task_goal_observation_and_active_code_draft():
    store = EventStore.from_messages(
        "acon/task/attempt-0", _messages(), benchmark="acon_appworld"
    )
    context = context_from_prepared(_prepared(store), [], "next_action()")

    assert context["goal"] == "Find the contact."
    assert [message["content"] for message in context["last_action_observation"]] == [
        "print(contact_id)", "Execution result: contact_id=new"
    ]
    assert context["parse_ok"] is True
    assert context["is_stop"] is False


def test_native_history_keeps_existing_user_and_stop_semantics():
    store = EventStore.from_messages("native/session", _messages())
    assert [event.source_indices for event in store.events] == [
        (0,), (1,), (2,), (3,), (4,), (5,)
    ]
    context = context_from_prepared(_prepared(store), [], "next_action()")
    assert context["goal"] == "Execution result: contact_id=new"
    assert context["last_action_observation"] == []
    assert context["is_stop"] is True
