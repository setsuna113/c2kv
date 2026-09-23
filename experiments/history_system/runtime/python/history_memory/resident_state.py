"""Committed source residency across append-only inference decisions.

The transcript remains an audit source. It is not an admission source for old
messages or encoder fragments that the previous final memory discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .events import EventStore
from .packing import EncoderChunk, PackedMemory, encode_event_chunks


FragmentKey = tuple[str, tuple[int, ...], int, int, tuple[int, ...]]


def fragment_key(chunk: EncoderChunk) -> FragmentKey:
    """Identify the exact encoded input, including its source and token span."""
    return (
        chunk.event_id,
        chunk.source_indices,
        chunk.source_token_start,
        chunk.source_token_end,
        chunk.token_ids,
    )


@dataclass(frozen=True)
class ResidentSourceState:
    """Only content in a committed final view may supply a later old source."""

    source_message_count: int
    raw_source_indices: frozenset[int]
    fragment_keys: frozenset[FragmentKey]

    @classmethod
    def from_memory(cls, store: EventStore, memory: PackedMemory) -> "ResidentSourceState":
        if not isinstance(store, EventStore) or not isinstance(memory, PackedMemory):
            raise TypeError("A resident state requires an EventStore and PackedMemory")
        count = len(store.messages)
        raw = frozenset(memory.raw_source_indices)
        if any(index < 0 or index >= count for index in raw):
            raise ValueError("Final raw memory references an unobserved source message")
        chunks = tuple(memory.chunks)
        if any(
            index < 0 or index >= count
            for chunk in chunks
            for index in chunk.source_indices
        ):
            raise ValueError("Final gist memory references an unobserved source message")
        return cls(count, raw, frozenset(map(fragment_key, chunks)))

    def admitted_fragments(
        self, chunks: Iterable[EncoderChunk]
    ) -> tuple[EncoderChunk, ...]:
        """Keep prior exact fragments and fragments wholly sourced from new input."""
        return tuple(
            chunk
            for chunk in chunks
            if (
                all(
                    index >= self.source_message_count or index in self.raw_source_indices
                    for index in chunk.source_indices
                )
                or fragment_key(chunk) in self.fragment_keys
            )
        )

    def admissible_event_ids(
        self,
        store: EventStore,
        event_ids: Iterable[str],
        tokenizer: Any,
        *,
        max_chunk_tokens: int,
        chunk_overlap: int,
    ) -> tuple[str, ...]:
        """Whole-event planners may use only events with every chunk resident.

        A partially retained event remains addressable through
        ``admitted_fragments``. A whole-event packer cannot repack its missing
        chunks from the archived original event.
        """
        admitted = []
        for event_id in event_ids:
            event = store.event(event_id)
            if not event.complete:
                continue
            chunks = encode_event_chunks(
                store,
                event_id,
                tokenizer,
                max_chunk_tokens=max_chunk_tokens,
                chunk_overlap=chunk_overlap,
            )
            if len(self.admitted_fragments(chunks)) == len(chunks):
                admitted.append(event_id)
        return tuple(admitted)

    def racer_admission_receipt(self, memory: PackedMemory) -> dict[str, Any]:
        """Report old content newly admitted by an explicit recovery policy."""
        old_raw = sorted(
            index
            for index in memory.raw_source_indices
            if index < self.source_message_count
            and index not in self.raw_source_indices
        )
        old_fragments = [
            chunk
            for chunk in memory.chunks
            if any(index < self.source_message_count for index in chunk.source_indices)
            and fragment_key(chunk) not in self.fragment_keys
            and not all(index in self.raw_source_indices for index in chunk.source_indices)
        ]
        return {
            "newly_admitted_old_raw_source_indices": old_raw,
            "newly_admitted_old_fragments": [
                {
                    "event_id": chunk.event_id,
                    "source_indices": list(chunk.source_indices),
                    "part_index": chunk.part_index,
                    "source_token_start": chunk.source_token_start,
                    "source_token_end": chunk.source_token_end,
                }
                for chunk in old_fragments
            ],
        }


__all__ = ["ResidentSourceState", "fragment_key"]
