"""Complete, source-linked tool observations for opt-in repair policies.

This reader does not alter the source event store. In particular, ACEBench
text actions are exposed only when the receipt-backed adapter marked their
event complete and the aggregate result still has one item per action.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from history_memory.events import EventRecord, EventStore

from ..acebench_source import parse_acebench_draft


@dataclass(frozen=True)
class ObservedOperation:
    event_id: str
    event_source_indices: tuple[int, ...]
    call_source_index: int
    result_source_index: int
    tool_call_id: str
    tool: str
    arguments: Any
    observed_result: Any
    failure_reported: bool
    order: int
    call_signature_id: str
    observation_version: int
    previous_source: dict[str, Any] | None


def parse_json_or_text(value: Any) -> Any:
    """Decode a whole JSON source field, preserving non-JSON executor text."""
    if isinstance(value, str):
        def unique_pairs(pairs):
            result = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = item
            return result

        def reject_constant(constant):
            raise ValueError("non-JSON constant: " + constant)

        try:
            return json.loads(value, object_pairs_hook=unique_pairs,
                              parse_constant=reject_constant)
        except (TypeError, ValueError):
            pass
    return value


def failure_reported(value: Any) -> bool:
    """Recognize explicit executor failure reports, not error-like substrings."""
    if isinstance(value, str):
        return value.startswith("Error during execution:")
    if isinstance(value, dict):
        return (value.get("success") is False
                or value.get("error") not in (None, "", False, [], {}))
    return False


def _signature(tool: str, arguments: Any) -> str:
    encoded = json.dumps(
        {"tool": tool, "arguments": arguments}, sort_keys=True,
        ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def call_signature(tool: str, arguments: Any) -> str:
    """Typed identity for one exact tool and argument object."""
    return _signature(tool, parse_json_or_text(arguments))


def _native_pairs(store: EventStore, event: EventRecord):
    indexed = [(index, store.messages[index].to_dict()) for index in event.source_indices]
    results = {}
    for index, message in indexed:
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id in results:
                return
            results[call_id] = (index, parse_json_or_text(message.get("content")))
    for call_index, message in indexed:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or ():
            if not isinstance(call, dict):
                return
            function = call.get("function")
            if not isinstance(function, dict) or call.get("id") not in results:
                return
            if not isinstance(function.get("name"), str):
                return
            result_index, result = results[call["id"]]
            yield (call_index, result_index, call["id"], function["name"],
                   parse_json_or_text(function.get("arguments")), result)


def _ace_pairs(store: EventStore, event: EventRecord):
    # A generic EventStore.from_messages may group ACE-like rows without an
    # official receipt. The ACE adapter alone supplies validated call IDs.
    if not event.tool_call_ids or len(event.source_indices) != 2:
        return
    call_index, result_index = event.source_indices
    assistant = store.messages[call_index].to_dict()
    execution = store.messages[result_index].to_dict()
    if assistant.get("role") != "assistant" or execution.get("role") != "tool":
        return
    draft = parse_acebench_draft(
        assistant.get("content"), call_id_prefix=event.event_id,
    ) if isinstance(assistant.get("content"), str) else None
    if draft is None or draft.status != "tool_calls":
        return
    if tuple(call["id"] for call in draft.tool_calls) != event.tool_call_ids:
        return
    try:
        aggregate = json.loads(execution["content"])
    except (KeyError, TypeError, ValueError):
        return
    if not isinstance(aggregate, list) or len(aggregate) != len(draft.tool_calls):
        return
    for call, result in zip(draft.tool_calls, aggregate):
        function = call["function"]
        yield (call_index, result_index, call["id"], function["name"],
               parse_json_or_text(function["arguments"]),
               parse_json_or_text(result))


def operation_records(
    store: EventStore, *, current_request_only: bool = False,
) -> tuple[ObservedOperation, ...]:
    """Return complete operations in source call order, with stable origins."""
    latest_user = next((event for event in reversed(store.events)
                        if event.kind == "user"), None)
    boundary = max(latest_user.source_indices) if latest_user else -1
    records: list[ObservedOperation] = []
    versions: dict[str, int] = {}
    previous: dict[str, dict[str, Any]] = {}
    for event in store.events:
        if event.kind != "tool_event" or not event.complete:
            continue
        first = store.messages[event.source_indices[0]].to_dict()
        pairs = (_native_pairs(store, event) if first.get("tool_calls")
                 else _ace_pairs(store, event))
        for call_index, result_index, call_id, tool, arguments, result in pairs:
            signature = _signature(tool, arguments)
            version = versions.get(signature, 0) + 1
            versions[signature] = version
            row = ObservedOperation(
                event_id=event.event_id,
                event_source_indices=event.source_indices,
                call_source_index=call_index,
                result_source_index=result_index,
                tool_call_id=call_id,
                tool=tool,
                arguments=arguments,
                observed_result=result,
                failure_reported=failure_reported(result),
                order=len(records),
                call_signature_id=signature,
                observation_version=version,
                previous_source=previous.get(signature),
            )
            records.append(row)
            previous[signature] = {
                "event_id": event.event_id,
                "result_source_index": result_index,
                "observation_version": version,
            }
    if current_request_only:
        return tuple(row for row in records if row.call_source_index > boundary)
    return tuple(records)


def current_request(
    store: EventStore,
) -> tuple[EventRecord | None, tuple[ObservedOperation, ...]]:
    """Latest user source and operations observed after it."""
    user = next((event for event in reversed(store.events)
                 if event.kind == "user"), None)
    return user, operation_records(store, current_request_only=True)


__all__ = [
    "ObservedOperation", "call_signature", "current_request", "failure_reported",
    "operation_records", "parse_json_or_text",
]
