"""Source-message packing for a unified raw/gist budget allocation.

Each gist unit contains one complete observable message. Event boundaries are
retained for provenance and tool-result binding, but do not force all of an
event's source messages into the same representation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .events import EventRecord, EventStore
from .packing import (
    EncoderChunk,
    PackedMemory,
    PackingBudgetError,
    native_ids,
    visible_message,
)


@dataclass(frozen=True)
class SourceMemoryView:
    raw_source_indices: tuple[int, ...]
    gist_source_indices: tuple[int, ...]
    omitted_source_indices: tuple[int, ...]
    mandatory_raw_source_indices: tuple[int, ...]
    raw_event_ids: tuple[str, ...]
    gist_event_ids: tuple[str, ...]
    omitted_event_ids: tuple[str, ...]
    mandatory_raw_event_ids: tuple[str, ...]
    partial_event_ids: tuple[str, ...]
    evidence_event_ids: tuple[str, ...] = ()

    def validate(self, store: EventStore) -> None:
        expected = make_source_view(
            store,
            self.raw_source_indices,
            self.gist_source_indices,
            mandatory_raw_source_indices=self.mandatory_raw_source_indices,
        )
        if self != expected:
            raise ValueError("Source view fields do not match their source partition")


def _ordered_indices(indices: Sequence[int], name: str, count: int) -> tuple[int, ...]:
    result = tuple(indices)
    if any(isinstance(index, bool) or not isinstance(index, int) for index in result):
        raise TypeError(f"{name} must contain integer source indices")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be strictly increasing and unique")
    if any(index < 0 or index >= count for index in result):
        raise ValueError(f"{name} must belong to the visible source prefix")
    return result


def make_source_view(
    store: EventStore,
    raw_source_indices: Sequence[int],
    gist_source_indices: Sequence[int],
    mandatory_raw_source_indices: Sequence[int] = (),
) -> SourceMemoryView:
    """Partition every visible source into raw, gist or omitted exactly once."""
    count = len(store.messages)
    raw = _ordered_indices(raw_source_indices, "raw_source_indices", count)
    gist = _ordered_indices(gist_source_indices, "gist_source_indices", count)
    mandatory = _ordered_indices(
        mandatory_raw_source_indices, "mandatory_raw_source_indices", count
    )
    raw_set, gist_set, mandatory_set = set(raw), set(gist), set(mandatory)
    if raw_set & gist_set:
        raise ValueError("Raw and gist sources must be disjoint")
    if not mandatory_set <= raw_set:
        raise ValueError("Mandatory raw sources must remain raw")
    omitted_set = set(range(count)) - raw_set - gist_set
    for event in store.events:
        if ((not event.complete or event.kind == "instruction")
                and not set(event.source_indices) <= raw_set):
            raise ValueError(
                f"Incomplete events and instructions must remain raw: {event.event_id}"
            )
    raw_events = []
    gist_events = []
    omitted_events = []
    mandatory_events = []
    partial_events = []
    for event in store.events:
        sources = set(event.source_indices)
        if sources <= raw_set:
            raw_events.append(event.event_id)
        if sources & gist_set:
            gist_events.append(event.event_id)
        if sources <= omitted_set:
            omitted_events.append(event.event_id)
        if sources <= mandatory_set:
            mandatory_events.append(event.event_id)
        if sum(bool(sources & group) for group in (raw_set, gist_set, omitted_set)) > 1:
            partial_events.append(event.event_id)
    return SourceMemoryView(
        raw_source_indices=raw,
        gist_source_indices=gist,
        omitted_source_indices=tuple(sorted(omitted_set)),
        mandatory_raw_source_indices=mandatory,
        raw_event_ids=tuple(raw_events),
        gist_event_ids=tuple(gist_events),
        omitted_event_ids=tuple(omitted_events),
        mandatory_raw_event_ids=tuple(mandatory_events),
        partial_event_ids=tuple(partial_events),
    )


def _source_event(store: EventStore, source_index: int) -> EventRecord:
    matches = [event for event in store.events if source_index in event.source_indices]
    if len(matches) != 1:
        raise ValueError(f"Source index {source_index} must belong to exactly one event")
    return matches[0]


def _producer_binding(
    store: EventStore, event: EventRecord, source_index: int,
) -> dict[str, Any] | None:
    message = store.messages[source_index].to_dict()
    if message["role"] != "tool":
        return None
    call_id = message.get("tool_call_id")
    for producer_index in event.source_indices:
        if producer_index >= source_index:
            continue
        producer = store.messages[producer_index].to_dict()
        for call in producer.get("tool_calls") or ():
            if call["id"] == call_id:
                return {
                    "source_index": producer_index,
                    "tool_call_id": call_id,
                    "tool_name": call["function"]["name"],
                }
    # Some source adapters pair an assistant action with a synthetic tool
    # observation without a native tool_calls field (e.g. ACEBench).
    if len(event.source_indices) == 2:
        producer_index = event.source_indices[0]
        if producer_index < source_index and store.messages[producer_index].role == "assistant":
            return {
                "source_index": producer_index,
                "tool_call_id": call_id,
                "tool_name": message.get("name"),
            }
    raise ValueError(f"Tool result at source {source_index} has no producer in its event")


def source_encoder_messages(
    store: EventStore, source_index: int,
) -> tuple[dict[str, str], ...]:
    """Use a typed envelope with one complete visible source message."""
    if isinstance(source_index, bool) or not isinstance(source_index, int):
        raise TypeError("source_index must be an integer")
    if source_index < 0 or source_index >= len(store.messages):
        raise ValueError("source_index must belong to the visible source prefix")
    event = _source_event(store, source_index)
    if not event.complete or event.kind == "instruction":
        raise ValueError(f"Only completed non-instruction sources can be encoded: {source_index}")
    envelope: dict[str, Any] = {
        "type": "history_source",
        "kind": event.kind,
        "event_id": event.event_id,
        "source_index": source_index,
        "message": visible_message(store.messages[source_index]),
    }
    binding = _producer_binding(store, event, source_index)
    if binding is not None:
        envelope["producer"] = binding
    return ({
        "role": "user",
        "content": json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
    },)


def encode_source_chunks(
    store: EventStore,
    source_indices: Sequence[int],
    tokenizer: Any,
    *,
    max_chunk_tokens: int = 768,
    chunk_overlap: int = 64,
) -> tuple[EncoderChunk, ...]:
    if max_chunk_tokens <= 0 or not 0 <= chunk_overlap < max_chunk_tokens:
        raise ValueError("Require max_chunk_tokens > chunk_overlap >= 0")
    selected = _ordered_indices(source_indices, "source_indices", len(store.messages))
    chunks = []
    for source_index in selected:
        event = _source_event(store, source_index)
        ids = native_ids(tokenizer, source_encoder_messages(store, source_index))
        unit_id = f"{event.event_id}:source:{source_index}"
        start = 0
        part_index = 0
        while start < len(ids):
            end = min(start + max_chunk_tokens, len(ids))
            chunks.append(EncoderChunk(
                unit_id, part_index, (source_index,), start, end, ids[start:end]
            ))
            if end == len(ids):
                break
            start = end - chunk_overlap
            part_index += 1
    return tuple(chunks)


def pack_source_memory(
    store: EventStore,
    view: SourceMemoryView,
    tokenizer: Any,
    *,
    tools=None,
    max_chunk_tokens: int = 768,
    chunk_overlap: int = 64,
    max_chunks: int | None = None,
    max_raw_tokens: int | None = None,
    derived_workspace_prefix_messages: Sequence[Mapping[str, Any]] = (),
) -> PackedMemory:
    """Pack the complete selected source partition with the native prefix."""
    view.validate(store)
    chunks = encode_source_chunks(
        store, view.gist_source_indices, tokenizer,
        max_chunk_tokens=max_chunk_tokens, chunk_overlap=chunk_overlap,
    )
    if max_chunks is not None and len(chunks) > max_chunks:
        raise PackingBudgetError(f"Complete sources need {len(chunks)} chunks; budget is {max_chunks}")
    raw_messages = [visible_message(store.messages[index]) for index in view.raw_source_indices]
    if not raw_messages:
        raise ValueError("A decision view requires an observable raw message")
    derived = []
    for message in derived_workspace_prefix_messages:
        if (set(message) != {"role", "content"} or message["role"] != "user"
                or not isinstance(message["content"], str) or not message["content"]):
            raise ValueError("Derived workspace observations require a nonempty user text message")
        derived.append(dict(message))
    prefix_length = 0
    while prefix_length < len(raw_messages) and raw_messages[prefix_length]["role"] == "system":
        prefix_length += 1
    raw_messages[prefix_length:prefix_length] = derived
    full_ids = native_ids(tokenizer, raw_messages, tools=tools, generation=True)
    prefix_messages = []
    for message in raw_messages:
        if message["role"] != "system":
            break
        prefix_messages.append(message)
    dummy = {"role": "user", "content": ""}
    dummy_ids = native_ids(tokenizer, [dummy])
    prefix_and_dummy = native_ids(tokenizer, prefix_messages + [dummy], tools=tools)
    if not dummy_ids or prefix_and_dummy[-len(dummy_ids):] != dummy_ids:
        raise ValueError("Native template does not support a separable system/tools prefix")
    prefix_ids = prefix_and_dummy[:-len(dummy_ids)]
    if full_ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Native system/tools prefix changed with workspace content")
    if max_raw_tokens is not None and len(full_ids) > max_raw_tokens:
        raise PackingBudgetError(
            f"System/tools plus complete raw workspace need {len(full_ids)} tokens; "
            f"budget is {max_raw_tokens}"
        )
    return PackedMemory(
        view, prefix_ids, full_ids[len(prefix_ids):], view.raw_source_indices, chunks
    )
