"""Plan speculative encoder work from an already observable source prefix."""

from __future__ import annotations

from typing import Any

from .encoding_scope import plan_encoding_scope
from .events import EventStore
from .packing import EncoderChunk, EncodingScopeCapacityError, encode_scope_chunks


def plan_cross_turn_chunks(
    store: EventStore,
    tokenizer: Any,
    *,
    encoding_scope: str,
    max_chunk_tokens: int,
    chunk_overlap: int,
    atomic_unit_token_limit: int,
    source_cutoff: int,
    benchmark: str | None = None,
) -> tuple[EncoderChunk, ...]:
    """Use the foreground packer on complete events that are raw this turn.

    A new assistant/tool or user message can move the source cutoff on the next
    request. Only existing source messages are encoded; an assistant echo or a
    pending tool result is never predicted.
    """
    if encoding_scope not in {"current", "event"}:
        return ()
    task_packet_id = None
    if benchmark == "acon_appworld":
        users = [event for event in store.events if event.kind == "user"]
        if users:
            task_packet_id = users[0].event_id
    candidates = tuple(
        event.event_id for event in store.events
        if event.complete
        and event.kind != "instruction"
        and event.event_id != task_packet_id
        and any(index >= source_cutoff for index in event.source_indices)
    )
    if not candidates:
        return ()
    scope = plan_encoding_scope(store, candidates, encoding_scope)
    chunks = []
    for group in scope.event_groups:
        try:
            chunks.extend(encode_scope_chunks(
                store, group, tokenizer,
                encoding_scope=encoding_scope,
                max_chunk_tokens=max_chunk_tokens,
                chunk_overlap=chunk_overlap,
                atomic_unit_token_limit=atomic_unit_token_limit,
                event_groups=(group,),
            ))
        except EncodingScopeCapacityError:
            # A future foreground call may also reject this atomic unit. Do not
            # turn an otherwise committed decision into a speculative failure.
            continue
    return tuple(chunks)
