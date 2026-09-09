"""Regression tests for incremental decision-prefix construction."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Iterator, Mapping, Sequence

import pytest

import history_memory.dataset as dataset_module
from history_memory.dataset import Decision, iter_decision_metadata, iter_decisions
from history_memory.events import EventRecord, EventStore, Message


def _call(call_id: str, name: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": '{"value":1}'},
    }


def _row(messages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "session_id": "incremental-session",
        "source": "fixture-source",
        "split": "train",
        "task_id": "incremental-task",
        "template_id": "incremental-template",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "messages": list(messages),
    }


def _old_build_events(
    session_id: str, messages: Sequence[Message]
) -> tuple[EventRecord, ...]:
    """Literal copy of the pre-incremental event algorithm."""
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("An explicit nonempty session_id is required")
    drafts: list[dict[str, Any]] = []
    pending: dict[str, int] = {}
    for index, raw in enumerate(messages):
        message = raw.to_dict()
        role = message["role"]
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise ValueError(
                    f"Unmatched or duplicate tool result at source index {index}"
                )
            draft = drafts[pending.pop(call_id)]
            draft["source_indices"].append(index)
            draft["missing"].remove(call_id)
            continue

        calls = message.get("tool_calls")
        call_ids: list[str] = []
        if calls:
            if role != "assistant" or not isinstance(calls, list):
                raise ValueError(f"Invalid tool_calls at source index {index}")
            for call in calls:
                call_id = call.get("id") if isinstance(call, dict) else None
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError(f"Missing tool call ID at source index {index}")
                if call_id in call_ids or call_id in pending:
                    raise ValueError(
                        f"Ambiguous tool call ID at source index {index}: {call_id}"
                    )
                if not isinstance(function, dict) or not isinstance(
                    function.get("name"), str
                ):
                    raise ValueError(f"Missing function name at source index {index}")
                call_ids.append(call_id)
        if message.get("function_call"):
            raise ValueError("Legacy function_call requires an explicit source adapter")
        kind = (
            "tool_event"
            if call_ids
            else ("instruction" if role in {"system", "developer"} else role)
        )
        drafts.append(
            {
                "event_id": f"{session_id}:m{index}",
                "kind": kind,
                "source_indices": [index],
                "tool_call_ids": call_ids,
                "missing": set(call_ids),
            }
        )
        for call_id in call_ids:
            pending[call_id] = len(drafts) - 1

    return tuple(
        EventRecord(
            event_id=draft["event_id"],
            kind=draft["kind"],
            source_indices=tuple(draft["source_indices"]),
            complete=not draft["missing"],
            tool_call_ids=tuple(draft["tool_call_ids"]),
            missing_tool_call_ids=tuple(
                call_id
                for call_id in draft["tool_call_ids"]
                if call_id in draft["missing"]
            ),
        )
        for draft in drafts
    )


def _old_store(
    session_id: str, messages: Sequence[Mapping[str, Any]]
) -> EventStore:
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("An explicit nonempty session_id is required")
    snapshots = tuple(Message.from_dict(message) for message in messages)
    return EventStore(session_id, snapshots, _old_build_events(session_id, snapshots))


def _old_iter_decisions(row: Mapping[str, Any]) -> Iterator[Decision]:
    """Literal copy of the pre-incremental decision algorithm."""
    snapshot = dataset_module._json_snapshot(dict(row))
    dataset_module._validate_row_shape(snapshot)
    session_id = snapshot["session_id"]
    messages = snapshot["messages"]
    tools_json = json.dumps(
        snapshot.get("tools") or [], ensure_ascii=False, allow_nan=False
    )
    decision_index = 0
    for source_message_index, message in enumerate(messages):
        if not dataset_module._is_decision_message(message):
            continue
        visible_prefix = tuple(
            dataset_module._model_visible_message(prefix_message)
            for prefix_message in messages[:source_message_index]
        )
        visible_target = dataset_module._model_visible_message(message)
        store = _old_store(
            dataset_module.event_store_session_id(snapshot["source"], session_id),
            visible_prefix,
        )
        yield Decision(
            decision_id=dataset_module._decision_id(
                source=snapshot["source"],
                session_id=session_id,
                task_id=snapshot["task_id"],
                template_id=snapshot["template_id"],
                source_message_index=source_message_index,
            ),
            session_id=session_id,
            source=snapshot["source"],
            split=snapshot["split"],
            task_id=snapshot["task_id"],
            template_id=snapshot["template_id"],
            decision_index=decision_index,
            source_message_index=source_message_index,
            store=store,
            target=Message.from_dict(visible_target),
            tools_json=tools_json,
        )
        decision_index += 1


def _decision_signature(decision: Decision) -> dict[str, Any]:
    return {
        "decision_id": decision.decision_id,
        "session_id": decision.session_id,
        "source": decision.source,
        "split": decision.split,
        "task_id": decision.task_id,
        "template_id": decision.template_id,
        "decision_index": decision.decision_index,
        "source_message_index": decision.source_message_index,
        "store_session_id": decision.store.session_id,
        "messages": tuple(message.json_text for message in decision.store.messages),
        "events": tuple(asdict(event) for event in decision.store.events),
        "target": decision.target.json_text,
        "tools_json": decision.tools_json,
    }


def _metadata_signature(value: Any) -> dict[str, Any]:
    return {
        key: getattr(value, key)
        for key in (
            "decision_id",
            "session_id",
            "source",
            "split",
            "task_id",
            "template_id",
            "decision_index",
            "source_message_index",
        )
    }


def _capture(iterator: Iterator[Any], signature) -> tuple[list[Any], tuple[type, str] | None]:
    values: list[Any] = []
    try:
        for value in iterator:
            values.append(signature(value))
    except Exception as exc:  # noqa: BLE001 - exact failure compatibility is under test
        return values, (type(exc), str(exc))
    return values, None


def _parallel_revision_row() -> dict[str, Any]:
    return _row(
        [
            {"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Read both values."},
            {"role": "assistant", "content": None, "tool_calls": []},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("a", "read"), _call("b", "read")],
            },
            {"role": "tool", "tool_call_id": "b", "content": "ERROR: timeout"},
            {"role": "assistant", "content": "One result has returned."},
            {"role": "tool", "tool_call_id": "a", "content": "value a"},
            {"role": "user", "content": "Revise after retrying the failed read."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("a", "read")],
            },
            {"role": "tool", "tool_call_id": "a", "content": "value b"},
            {"role": "assistant", "content": "", "tool_calls": []},
            {"role": "assistant", "content": "Final answer."},
        ]
    )


def test_incremental_decisions_are_exactly_equal_to_literal_old_algorithm():
    row = _parallel_revision_row()

    expected = tuple(_old_iter_decisions(row))
    actual = tuple(iter_decisions(row))

    assert [_decision_signature(value) for value in actual] == [
        _decision_signature(value) for value in expected
    ]
    assert [value.source_message_index for value in actual] == [3, 5, 8, 11]
    partial = actual[1].store.event(
        f"{actual[1].store.session_id}:m3"
    )
    completed = actual[2].store.event(
        f"{actual[2].store.session_id}:m3"
    )
    assert partial.complete is False
    assert partial.source_indices == (3, 4)
    assert partial.missing_tool_call_ids == ("a",)
    assert completed.complete is True
    assert completed.source_indices == (3, 4, 6)
    assert all(
        max(event.source_indices) < decision.source_message_index
        for decision in actual
        for event in decision.store.events
    )


def test_metadata_matches_decisions_without_allocating_store_snapshots(monkeypatch):
    row = _parallel_revision_row()
    expected = tuple(_old_iter_decisions(row))

    def fail_snapshot(self):
        raise AssertionError("metadata iteration must not allocate EventStore snapshots")

    monkeypatch.setattr(dataset_module.EventStoreBuilder, "snapshot", fail_snapshot)
    metadata = tuple(iter_decision_metadata(row))

    assert [_metadata_signature(value) for value in metadata] == [
        _metadata_signature(value) for value in expected
    ]
    assert all(value.session_key == expected[0].store.session_id for value in metadata)


@pytest.mark.parametrize(
    "messages",
    [
        [
            {"role": "tool", "tool_call_id": "missing", "content": "bad"},
            {"role": "assistant", "content": "target"},
        ],
        [
            {"role": "assistant", "tool_calls": [_call("x", "read")]},
            {"role": "assistant", "tool_calls": [_call("x", "read")]},
            {"role": "assistant", "content": "later"},
        ],
        [
            {"role": "assistant", "tool_calls": [_call("x", "read")]},
            {"role": "tool", "tool_call_id": "x", "content": "ok"},
            {"role": "tool", "tool_call_id": "x", "content": "duplicate"},
            {"role": "assistant", "content": "later"},
        ],
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "x",
                        "type": "function",
                        "function": {"name": None, "arguments": "{}"},
                    }
                ],
            },
            {"role": "assistant", "content": "later"},
        ],
    ],
)
def test_incremental_and_metadata_iterators_preserve_old_error_behavior(messages):
    row = _row(messages)
    expected_values, expected_error = _capture(
        _old_iter_decisions(row), _decision_signature
    )
    expected_metadata = [
        {
            key: value[key]
            for key in (
                "decision_id",
                "session_id",
                "source",
                "split",
                "task_id",
                "template_id",
                "decision_index",
                "source_message_index",
            )
        }
        for value in expected_values
    ]

    actual_values, actual_error = _capture(iter_decisions(row), _decision_signature)
    metadata_values, metadata_error = _capture(
        iter_decision_metadata(row), _metadata_signature
    )

    assert actual_values == expected_values
    assert actual_error == expected_error
    assert metadata_values == expected_metadata
    assert metadata_error == expected_error


def test_each_visible_message_is_normalized_only_once(monkeypatch):
    messages: list[dict[str, Any]] = [{"role": "system", "content": "system"}]
    for index in range(40):
        messages.extend(
            [
                {"role": "user", "content": f"question {index}"},
                {"role": "assistant", "content": f"answer {index}"},
            ]
        )
    calls = 0
    original = dataset_module._model_visible_message

    def counted(message):
        nonlocal calls
        calls += 1
        return original(message)

    monkeypatch.setattr(dataset_module, "_model_visible_message", counted)

    decisions = tuple(iter_decisions(_row(messages)))

    assert len(decisions) == 40
    assert calls == len(messages)


def test_event_lookup_cache_preserves_first_match_and_missing_error():
    first = EventRecord("duplicate", "user", (0,), True)
    second = EventRecord("duplicate", "assistant", (1,), True)
    store = EventStore(
        "manual",
        (Message.from_dict({"role": "user", "content": "value"}),),
        (first, second),
    )

    assert store.event("duplicate") is first
    assert store.event("duplicate") is first
    with pytest.raises(KeyError, match="Event is not in the visible prefix: missing"):
        store.event("missing")


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda row: row.pop("split"), "split.*before expansion"),
        (
            lambda row: row["messages"].append(
                {"role": "assistant", "content": {"not": "text"}}
            ),
            "Non-text content",
        ),
        (
            lambda row: row["messages"].append(
                {
                    "role": "assistant",
                    "function_call": {"name": "legacy", "arguments": "{}"},
                }
            ),
            "Legacy function_call",
        ),
        (lambda row: row.update(tools=[{"value": float("nan")}]), "JSON compliant"),
    ],
)
def test_metadata_preserves_row_and_tools_validation(mutate, expected):
    row = _parallel_revision_row()
    mutate(row)

    with pytest.raises((TypeError, ValueError), match=expected):
        tuple(iter_decision_metadata(row))
