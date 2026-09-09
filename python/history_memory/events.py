"""Immutable events built exclusively from an observable OpenAI message prefix.

An assistant tool call and all its results form one event, even when parallel
results arrive out of order. Incomplete events remain addressable but may not
be compressed. Source messages are never rewritten or relabelled as users.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class Message:
    """An owned JSON snapshot; callers only receive fresh mutable copies."""

    json_text: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Message:
        if value.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported message role: {value.get('role')!r}")
        return cls(json.dumps(dict(value), ensure_ascii=False, allow_nan=False))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.json_text)

    @property
    def role(self) -> str:
        return self.to_dict()["role"]


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    kind: str
    source_indices: tuple[int, ...]
    complete: bool
    tool_call_ids: tuple[str, ...] = ()
    missing_tool_call_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EventStore:
    session_id: str
    messages: tuple[Message, ...]
    events: tuple[EventRecord, ...]

    @classmethod
    def from_messages(
        cls, session_id: str, messages: Sequence[Mapping[str, Any]]
    ) -> EventStore:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("An explicit nonempty session_id is required")
        snapshots = tuple(Message.from_dict(message) for message in messages)
        return cls(session_id, snapshots, build_events(session_id, snapshots))

    def event(self, event_id: str) -> EventRecord:
        index = self.__dict__.get("_event_index")
        if index is None:
            index = {}
            for event in self.events:
                # Preserve the first-match behavior for manually constructed
                # stores that contain duplicate event IDs.
                index.setdefault(event.event_id, event)
            object.__setattr__(self, "_event_index", index)
        try:
            return index[event_id]
        except KeyError:
            raise KeyError(f"Event is not in the visible prefix: {event_id}") from None

    def event_messages(self, event_id: str) -> tuple[Message, ...]:
        return tuple(self.messages[i] for i in self.event(event_id).source_indices)


class EventStoreBuilder:
    """Incrementally build immutable prefix snapshots without reparsing history."""

    def __init__(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("An explicit nonempty session_id is required")
        self.session_id = session_id
        self._messages: list[Message] = []
        self._events: list[EventRecord] = []
        self._pending: dict[str, int] = {}

    def append(self, raw: Message | Mapping[str, Any]) -> Message:
        """Append one observable message and update only its affected event."""
        message = raw if isinstance(raw, Message) else Message.from_dict(raw)
        value = message.to_dict()
        index = len(self._messages)
        role = value["role"]

        if role == "tool":
            call_id = value.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in self._pending:
                raise ValueError(
                    f"Unmatched or duplicate tool result at source index {index}"
                )
            event_index = self._pending[call_id]
            event = self._events[event_index]
            missing = tuple(
                pending_id
                for pending_id in event.missing_tool_call_ids
                if pending_id != call_id
            )
            self._messages.append(message)
            self._events[event_index] = EventRecord(
                event_id=event.event_id,
                kind=event.kind,
                source_indices=event.source_indices + (index,),
                complete=not missing,
                tool_call_ids=event.tool_call_ids,
                missing_tool_call_ids=missing,
            )
            del self._pending[call_id]
            return message

        calls = value.get("tool_calls")
        call_ids: list[str] = []
        if calls:
            if role != "assistant" or not isinstance(calls, list):
                raise ValueError(f"Invalid tool_calls at source index {index}")
            for call in calls:
                call_id = call.get("id") if isinstance(call, dict) else None
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(call_id, str) or not call_id:
                    raise ValueError(f"Missing tool call ID at source index {index}")
                if call_id in call_ids or call_id in self._pending:
                    raise ValueError(
                        f"Ambiguous tool call ID at source index {index}: {call_id}"
                    )
                if not isinstance(function, dict) or not isinstance(
                    function.get("name"), str
                ):
                    raise ValueError(f"Missing function name at source index {index}")
                call_ids.append(call_id)
        if value.get("function_call"):
            raise ValueError("Legacy function_call requires an explicit source adapter")

        kind = (
            "tool_event"
            if call_ids
            else ("instruction" if role in {"system", "developer"} else role)
        )
        event = EventRecord(
            event_id=f"{self.session_id}:m{index}",
            kind=kind,
            source_indices=(index,),
            complete=not call_ids,
            tool_call_ids=tuple(call_ids),
            missing_tool_call_ids=tuple(call_ids),
        )
        self._messages.append(message)
        self._events.append(event)
        event_index = len(self._events) - 1
        for call_id in call_ids:
            self._pending[call_id] = event_index
        return message

    def snapshot(self) -> EventStore:
        """Return an immutable store for the current observable prefix."""
        return EventStore(
            session_id=self.session_id,
            messages=tuple(self._messages),
            events=tuple(self._events),
        )


def build_events(
    session_id: str, messages: Sequence[Message | Mapping[str, Any]]
) -> tuple[EventRecord, ...]:
    """Build stable source-index IDs, refusing ambiguous call/result bindings.

    This function accepts only already observable messages. It neither reads
    a target action nor infers completion from text such as 'success'. A tool
    event is complete when every declared call has exactly one result; that
    result can contain an error. Old completed events retain their IDs when
    more messages arrive. Repeated IDs after completion are allowed; an ID
    cannot be reused while a result is outstanding.
    """
    builder = EventStoreBuilder(session_id)
    for raw in messages:
        builder.append(raw)
    return builder.snapshot().events
