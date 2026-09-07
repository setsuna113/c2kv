"""CPU-only construction of observable history-memory training decisions.

Input rows must already be normalized OpenAI conversations and carry their
split.  This module snapshots and validates every row before expanding any
assistant decisions, so a group split conflict cannot leak through expansion.
The lifecycle planner sees only an :class:`EventStore` built from messages
strictly before the target and the target's zero-based decision ordinal.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence, TypeAlias

from .events import EventStore, Message
from .packing import MemoryView, select_view


SCHEMA_VERSION = "history-memory-decision-v1"
_REQUIRED_ROW_FIELDS = ("session_id", "source", "split", "task_id", "template_id")
_GROUP_FIELDS = ("session_id", "task_id", "template_id")


def _json_snapshot(value: Any) -> Any:
    """Own a JSON-shaped value without parsing or normalizing string payloads."""
    return copy.deepcopy(value)


def _nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (Mapping, Sequence)) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return len(value) > 0
    return True


def _is_decision_message(message: Mapping[str, Any]) -> bool:
    """Return whether an assistant message contains an observable action."""
    if message.get("role") != "assistant":
        return False
    return any(_nonempty(message.get(field)) for field in ("content", "tool_calls"))


def _model_visible_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """Whitelist OpenAI text/tool fields while retaining argument strings.

    ``packing.visible_message`` parses valid argument JSON for the native chat
    template.  Dataset snapshots deliberately precede that boundary so the
    original argument string remains byte-for-byte equal as a Python string.
    """
    if "function_call" in message and message["function_call"] is not None:
        raise ValueError("Legacy function_call requires an explicit source adapter")
    role = message.get("role")
    result = {
        key: _json_snapshot(message[key])
        for key in ("role", "content", "name", "tool_call_id")
        if key in message
    }
    if role not in {"system", "developer", "user", "assistant", "tool"}:
        # Keep the authoritative role error in EventStore/Message consistent.
        result["role"] = role
    if message.get("tool_calls"):
        calls = []
        for call in message["tool_calls"]:
            if not isinstance(call, Mapping):
                raise ValueError("Every tool call must be a mapping")
            function = call.get("function")
            if not isinstance(function, Mapping):
                raise ValueError("Every tool call requires a function mapping")
            calls.append(
                {
                    "id": _json_snapshot(call.get("id")),
                    "type": _json_snapshot(call.get("type", "function")),
                    "function": {
                        "name": _json_snapshot(function.get("name")),
                        "arguments": _json_snapshot(function.get("arguments", {})),
                    },
                }
            )
        result["tool_calls"] = calls
    return result


def _require_nonempty_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Normalized row requires a nonempty {field!r} before expansion")
    return value


def _validate_row_shape(row: Mapping[str, Any]) -> None:
    if not isinstance(row, Mapping):
        raise TypeError(f"Expected a normalized row mapping, got {type(row).__name__}")
    for field in _REQUIRED_ROW_FIELDS:
        _require_nonempty_string(row, field)
    messages = row.get("messages")
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes, bytearray)):
        raise ValueError("Normalized row requires a messages sequence before expansion")
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be a mapping")
        if "function_call" in message and message["function_call"] is not None:
            raise ValueError("Legacy function_call requires an explicit source adapter")
        if message.get("role") not in {
            "system",
            "developer",
            "user",
            "assistant",
            "tool",
        }:
            raise ValueError(
                f"Unsupported message role at messages[{index}]: {message.get('role')!r}"
            )
        if message.get("content") is not None and not isinstance(
            message.get("content"), str
        ):
            raise ValueError(
                f"Non-text content at messages[{index}] requires an explicit source adapter"
            )
    tools = row.get("tools", ())
    if tools is None:
        tools = ()
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes, bytearray)):
        raise ValueError("tools must be a sequence when supplied")
    for index, tool in enumerate(tools):
        if not isinstance(tool, Mapping):
            raise ValueError(f"tools[{index}] must be a mapping")


def snapshot_and_validate_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Snapshot rows and reject group split leakage before decision expansion.

    Session, task, and template identifiers are each treated as grouping keys
    within a source.  Reusing any such identifier across splits is an error.
    """
    snapshots = tuple(_json_snapshot(dict(row)) for row in rows)
    for row in snapshots:
        _validate_row_shape(row)

    assigned_splits: dict[tuple[str, str, str], str] = {}
    for row in snapshots:
        source = row["source"]
        split = row["split"]
        for field in _GROUP_FIELDS:
            key = (source, field, row[field])
            previous = assigned_splits.setdefault(key, split)
            if previous != split:
                raise ValueError(
                    "Group split leakage: "
                    f"source={source!r} {field}={row[field]!r} occurs in "
                    f"both {previous!r} and {split!r}"
                )
    return snapshots


def _decision_id(
    *,
    source: str,
    session_id: str,
    task_id: str,
    template_id: str,
    source_message_index: int,
) -> str:
    identity = json.dumps(
        [source, session_id, task_id, template_id, source_message_index],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "decision-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _store_session_id(source: str, session_id: str) -> str:
    """Namespace event identity when source families reuse session IDs."""
    return json.dumps([source, session_id], ensure_ascii=False, separators=(",", ":"))


@dataclass(frozen=True)
class Decision:
    """One immutable next-assistant-action target and its observable prefix."""

    decision_id: str
    session_id: str
    source: str
    split: str
    task_id: str
    template_id: str
    decision_index: int
    source_message_index: int
    store: EventStore
    target: Message
    tools_json: str

    @property
    def prefix(self) -> tuple[Message, ...]:
        return self.store.messages

    @property
    def tools(self) -> tuple[dict[str, Any], ...]:
        # Fresh values preserve the immutable snapshot held by this record.
        return tuple(json.loads(self.tools_json))

    def target_dict(self) -> dict[str, Any]:
        return self.target.to_dict()


def iter_decisions(row: Mapping[str, Any]) -> Iterator[Decision]:
    """Yield every nonempty assistant message as a prefix-only decision.

    A direct call still requires an explicit split.  Cross-row split
    consistency is enforced by :func:`snapshot_and_validate_rows` and all
    multi-row builders below.
    """
    snapshot = _json_snapshot(dict(row))
    _validate_row_shape(snapshot)
    session_id = snapshot["session_id"]
    messages = snapshot["messages"]
    tools_json = json.dumps(
        snapshot.get("tools") or [], ensure_ascii=False, allow_nan=False
    )
    decision_index = 0
    for source_message_index, message in enumerate(messages):
        if not _is_decision_message(message):
            continue
        visible_prefix = tuple(
            _model_visible_message(prefix_message)
            for prefix_message in messages[:source_message_index]
        )
        visible_target = _model_visible_message(message)
        store = EventStore.from_messages(
            _store_session_id(snapshot["source"], session_id), visible_prefix
        )
        yield Decision(
            decision_id=_decision_id(
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


@dataclass(frozen=True)
class LifecycleSelection:
    """Prefix-local event IDs selected by a stateful lifecycle planner."""

    restored_event_ids: tuple[str, ...] = ()
    pinned_event_ids: tuple[str, ...] = ()


LifecyclePlan: TypeAlias = LifecycleSelection | MemoryView


class LifecyclePlanner(Protocol):
    """State may persist across calls; inputs contain no target or future.

    Stateful planners must key conversation state by ``store.session_id``,
    which is the JSON identity ``[source, original_session_id]``.
    """

    def __call__(self, store: EventStore, decision_index: int) -> LifecyclePlan:
        ...


@dataclass(frozen=True)
class MemoryDecisionRecord:
    """A decision exposed through one static (C) or lifecycle (B) view."""

    decision: Decision
    arm: str
    view: MemoryView
    repetition_index: int = 0
    weight: float = 1.0

    @property
    def decision_id(self) -> str:
        return self.decision.decision_id

    @property
    def target(self) -> Message:
        return self.decision.target

    def to_dict(self) -> dict[str, Any]:
        store = self.decision.store
        return {
            "schema_version": SCHEMA_VERSION,
            "decision_id": self.decision.decision_id,
            "arm": self.arm,
            "repetition_index": self.repetition_index,
            "weight": self.weight,
            "session_id": self.decision.session_id,
            "event_store_session_id": store.session_id,
            "source": self.decision.source,
            "split": self.decision.split,
            "task_id": self.decision.task_id,
            "template_id": self.decision.template_id,
            "decision_index": self.decision.decision_index,
            "source_message_index": self.decision.source_message_index,
            "tools": list(self.decision.tools),
            "prefix_messages": [message.to_dict() for message in store.messages],
            "events": [
                {
                    "event_id": event.event_id,
                    "kind": event.kind,
                    "source_indices": list(event.source_indices),
                    "complete": event.complete,
                    "tool_call_ids": list(event.tool_call_ids),
                    "missing_tool_call_ids": list(event.missing_tool_call_ids),
                }
                for event in store.events
            ],
            "memory_view": {
                "gist_event_ids": list(self.view.gist_event_ids),
                "raw_event_ids": list(self.view.raw_event_ids),
            },
            "target": self.decision.target.to_dict(),
        }


def _validate_repetitions(repetitions: int) -> None:
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise ValueError("repetitions must be an integer >= 1")


def _static_records_from_snapshots(
    rows: Sequence[Mapping[str, Any]],
    *,
    recent_tool_events: int,
    repetitions: int,
) -> tuple[MemoryDecisionRecord, ...]:
    records: list[MemoryDecisionRecord] = []
    for row in rows:
        for decision in iter_decisions(row):
            view = select_view(decision.store, recent_tool_events=recent_tool_events)
            records.extend(
                MemoryDecisionRecord(
                    decision=decision,
                    arm="C",
                    view=view,
                    repetition_index=repetition_index,
                )
                for repetition_index in range(repetitions)
            )
    return tuple(records)


def build_static_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    recent_tool_events: int = 1,
    repetitions: int = 1,
) -> tuple[MemoryDecisionRecord, ...]:
    """Build static C-view records after validating all group splits."""
    _validate_repetitions(repetitions)
    snapshots = snapshot_and_validate_rows(rows)
    return _static_records_from_snapshots(
        snapshots,
        recent_tool_events=recent_tool_events,
        repetitions=repetitions,
    )


def build_paired_records(
    rows: Iterable[Mapping[str, Any]],
    planner: LifecyclePlanner,
    *,
    recent_tool_events: int = 1,
    repetitions: int = 1,
) -> tuple[MemoryDecisionRecord, ...]:
    """Build matched C/B records with one planner call per real decision.

    The same decision target, unit weight, and repetition index are emitted for
    both arms.  A planner can retain state across successive calls.  Returning
    ``LifecycleSelection`` delegates view construction to ``select_view``;
    returning ``MemoryView`` supplies an already constructed prefix-local view.
    """
    _validate_repetitions(repetitions)
    if not callable(planner):
        raise TypeError("planner must be callable")
    snapshots = snapshot_and_validate_rows(rows)
    records: list[MemoryDecisionRecord] = []
    for row in snapshots:
        for decision in iter_decisions(row):
            static_view = select_view(
                decision.store, recent_tool_events=recent_tool_events
            )
            lifecycle_plan = planner(decision.store, decision.decision_index)
            if isinstance(lifecycle_plan, LifecycleSelection):
                lifecycle_view = select_view(
                    decision.store,
                    recent_tool_events=recent_tool_events,
                    restored_event_ids=lifecycle_plan.restored_event_ids,
                    pinned_event_ids=lifecycle_plan.pinned_event_ids,
                )
            elif isinstance(lifecycle_plan, MemoryView):
                lifecycle_view = lifecycle_plan
            else:
                raise TypeError(
                    "planner must return LifecycleSelection or MemoryView, got "
                    f"{type(lifecycle_plan).__name__}"
                )
            lifecycle_view.validate(decision.store)
            for repetition_index in range(repetitions):
                records.append(
                    MemoryDecisionRecord(
                        decision=decision,
                        arm="C",
                        view=static_view,
                        repetition_index=repetition_index,
                    )
                )
                records.append(
                    MemoryDecisionRecord(
                        decision=decision,
                        arm="B",
                        view=lifecycle_view,
                        repetition_index=repetition_index,
                    )
                )
    return tuple(records)


def read_jsonl_rows(paths: Sequence[str | Path]) -> tuple[dict[str, Any], ...]:
    """Read normalized JSONL rows, attaching no inferred split or labels."""
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                rows.append(row)
    return tuple(rows)


def write_jsonl_records(
    path: str | Path, records: Iterable[MemoryDecisionRecord]
) -> int:
    """Write records without rewriting nested tool argument strings."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record.to_dict(), ensure_ascii=False, allow_nan=False) + "\n"
            )
            count += 1
    return count


__all__ = [
    "Decision",
    "LifecyclePlan",
    "LifecyclePlanner",
    "LifecycleSelection",
    "MemoryDecisionRecord",
    "build_paired_records",
    "build_static_records",
    "iter_decisions",
    "read_jsonl_rows",
    "snapshot_and_validate_rows",
    "write_jsonl_records",
]
