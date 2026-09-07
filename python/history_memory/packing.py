"""Pure CPU packing shared by training and serving adapters.

Event extraction uses a stable, typed envelope because native Qwen templates
omit call IDs. The raw workspace uses the native chat template. This module
plans token and RoPE positions; it does not extract, rotate or cache tensors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .events import EventStore, Message

PACKING_VERSION = "history-event-v1"


class PackingBudgetError(ValueError):
    """The complete representation cannot fit; no content was truncated."""


@dataclass(frozen=True)
class MemoryView:
    gist_event_ids: tuple[str, ...]
    raw_event_ids: tuple[str, ...]

    def validate(self, store: EventStore) -> None:
        known = {event.event_id for event in store.events}
        for ids in (self.gist_event_ids, self.raw_event_ids):
            if len(ids) != len(set(ids)):
                raise ValueError("Duplicate event within a memory view component")
        selected = set(self.gist_event_ids) | set(self.raw_event_ids)
        if selected != known:
            raise ValueError(f"View must cover exactly the visible events; missing={known - selected}, unknown={selected - known}")
        for event_id in self.gist_event_ids:
            event = store.event(event_id)
            if not event.complete or event.kind == "instruction":
                raise ValueError(f"Incomplete events and instructions must remain raw: {event_id}")


def select_view(
    store: EventStore, *, recent_tool_events: int = 1,
    restored_event_ids: Iterable[str] = (), pinned_event_ids: Iterable[str] = (),
) -> MemoryView:
    """A fixed recent workspace plus caller-selected observable evidence.

    The current user request, latest message's event, instructions, pending
    calls and the most recently COMPLETED tool events are protected. A owns
    the rules for additional live bindings and persistent evidence leases.
    """
    if recent_tool_events < 0:
        raise ValueError("recent_tool_events must be nonnegative")
    raw = set(restored_event_ids) | set(pinned_event_ids)
    for event_id in raw:
        store.event(event_id)
    raw.update(event.event_id for event in store.events if not event.complete or event.kind == "instruction")
    users = [event for event in store.events if event.kind == "user"]
    if users:
        raw.add(users[-1].event_id)
    if store.events:
        raw.add(max(store.events, key=lambda event: max(event.source_indices)).event_id)
    completed = sorted(
        (event for event in store.events if event.kind == "tool_event" and event.complete),
        key=lambda event: max(event.source_indices),
    )
    if recent_tool_events:
        raw.update(event.event_id for event in completed[-recent_tool_events:])
    view = MemoryView(
        gist_event_ids=tuple(event.event_id for event in store.events if event.event_id not in raw),
        raw_event_ids=tuple(event.event_id for event in store.events if event.event_id in raw),
    )
    view.validate(store)
    return view


def visible_message(message: Message | Mapping[str, Any]) -> dict[str, Any]:
    """Keep model-visible fields; parse JSON arguments exactly once.

    Source snapshots retain their original argument strings. A valid JSON
    object becomes an object for templating, avoiding double encoding. Invalid
    historical arguments remain verbatim so error-recovery traces are usable.
    Non-text content requires a source adapter instead of silently vanishing
    in a text-only template. Metadata such as reward is not model input.
    """
    source = message.to_dict() if isinstance(message, Message) else dict(message)
    content = source.get("content")
    if content is not None and not isinstance(content, str):
        raise ValueError("Non-text content requires an explicit source adapter")
    if source["role"] == "developer":
        raise ValueError("The Qwen text packing profile requires developer-role adaptation")
    result = {key: source[key] for key in ("role", "content", "name", "tool_call_id") if key in source}
    if source.get("tool_calls"):
        calls = []
        for call in source["tool_calls"]:
            function = call["function"]
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    decoded = json.loads(arguments)
                    if isinstance(decoded, dict):
                        arguments = decoded
                except json.JSONDecodeError:
                    pass
            calls.append({
                "id": call["id"], "type": call.get("type", "function"),
                "function": {"name": function["name"], "arguments": arguments},
            })
        result["tool_calls"] = calls
    return result


def native_ids(tokenizer: Any, messages: Sequence[Mapping[str, Any]], *, tools=None, generation=False) -> tuple[int, ...]:
    if not messages:
        raise ValueError("The native text template needs at least one message")
    ids = tokenizer.apply_chat_template(
        list(messages), tools=tools, tokenize=True,
        add_generation_prompt=generation, enable_thinking=False,
        truncation=False,
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return tuple(int(token) for token in ids)


def event_encoder_messages(store: EventStore, event_id: str) -> tuple[dict[str, Any], ...]:
    """Content depends only on this completed event, never its current view."""
    event = store.event(event_id)
    if not event.complete or event.kind == "instruction":
        raise ValueError("Only complete history events can be independently encoded")
    envelope = {
        "type": "history_event", "kind": event.kind,
        "messages": [visible_message(message) for message in store.event_messages(event_id)],
    }
    return ({"role": "user", "content": json.dumps(envelope, ensure_ascii=False, separators=(",", ":"), allow_nan=False)},)


@dataclass(frozen=True)
class EncoderChunk:
    event_id: str
    part_index: int
    source_indices: tuple[int, ...]
    source_token_start: int
    source_token_end: int
    token_ids: tuple[int, ...]

    def encoding_key(self, *, parameter_version: str | int, ratio: int) -> tuple[Any, ...]:
        """An exact-input key; the caller advances version after optimizer.step.

        Runtime RoPE offsets are deliberately excluded: cached gist keys must
        be pre-RoPE, then rotated for the current layout by the model adapter.
        """
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        return (PACKING_VERSION, parameter_version, ratio, self.token_ids)


def encode_event_chunks(
    store: EventStore, event_id: str, tokenizer: Any, *, max_chunk_tokens: int = 768,
    chunk_overlap: int = 64,
) -> tuple[EncoderChunk, ...]:
    if max_chunk_tokens <= 0 or not 0 <= chunk_overlap < max_chunk_tokens:
        raise ValueError("Require max_chunk_tokens > chunk_overlap >= 0")
    ids = native_ids(tokenizer, event_encoder_messages(store, event_id))
    chunks = []
    start = 0
    while start < len(ids):
        end = min(start + max_chunk_tokens, len(ids))
        chunks.append(EncoderChunk(event_id, len(chunks), store.event(event_id).source_indices, start, end, ids[start:end]))
        if end == len(ids):
            break
        start = end - chunk_overlap
    return tuple(chunks)


@dataclass(frozen=True)
class GistPlacement:
    chunk: EncoderChunk
    source_position_start: int
    position_ids: tuple[int, ...]


@dataclass(frozen=True)
class PackedMemory:
    view: MemoryView
    system_input_ids: tuple[int, ...]
    workspace_input_ids: tuple[int, ...]
    raw_source_indices: tuple[int, ...]
    chunks: tuple[EncoderChunk, ...]

    def gist_layout(self, ratio: int) -> tuple[GistPlacement, ...]:
        """Match dynamic-interleave's source-span RoPE, not compressed offsets."""
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        offset = len(self.system_input_ids)
        placements = []
        for chunk in self.chunks:
            length = len(chunk.token_ids)
            positions = tuple(offset + min(start + ratio, length) - 1 for start in range(0, length, ratio))
            placements.append(GistPlacement(chunk, offset, positions))
            offset += length
        return tuple(placements)

    @property
    def workspace_position_start(self) -> int:
        return len(self.system_input_ids) + sum(len(chunk.token_ids) for chunk in self.chunks)

    def costs(self, ratio: int) -> dict[str, int]:
        gist_tokens = sum(len(placement.position_ids) for placement in self.gist_layout(ratio))
        presented = sum(len(chunk.token_ids) for chunk in self.chunks)
        unique = sum(
            max(chunk.source_token_end for chunk in self.chunks if chunk.event_id == event_id)
            for event_id in {chunk.event_id for chunk in self.chunks}
        )
        return {
            "system_tokens": len(self.system_input_ids), "raw_tokens": len(self.workspace_input_ids),
            "gist_tokens": gist_tokens, "presented_encoder_tokens": presented,
            "encoder_overlap_tokens": presented - unique,
            "resident_kv_tokens": len(self.system_input_ids) + len(self.workspace_input_ids) + gist_tokens,
            "raw_gist_overlap_events": len(set(self.view.raw_event_ids) & set(self.view.gist_event_ids)),
        }

    def causal_mask(self, ratio: int, *, target_tokens: int = 0) -> tuple[tuple[bool, ...], ...]:
        """Small CPU reference: ordinary queries see all memory and causal raw.

        Tensor adapters should materialize an equivalent efficient mask. Event
        extraction remains separate per chunk; no event sees another event.
        """
        if target_tokens < 0:
            raise ValueError("target_tokens must be nonnegative")
        past = len(self.system_input_ids) + sum(len(p.position_ids) for p in self.gist_layout(ratio))
        ordinary = len(self.workspace_input_ids) + target_tokens
        return tuple(tuple(key < past + query + 1 for key in range(past + ordinary)) for query in range(ordinary))


def pack_memory(
    store: EventStore, view: MemoryView, tokenizer: Any, *, tools=None,
    max_chunk_tokens: int = 768, chunk_overlap: int = 64, max_chunks: int | None = None,
    max_raw_tokens: int | None = None,
) -> PackedMemory:
    """Pack all selected content or raise; no first-plus-tail selection occurs.

    Chunks follow event source order regardless of the caller's ID ordering.
    Raw events are restored in original message order. The system/tools prefix
    is separated at a verified native-template boundary; the model adapter
    inserts gist KV between this prefix and the exact workspace.
    """
    view.validate(store)
    chunks = tuple(
        chunk for event in store.events if event.event_id in view.gist_event_ids
        for chunk in encode_event_chunks(store, event.event_id, tokenizer, max_chunk_tokens=max_chunk_tokens, chunk_overlap=chunk_overlap)
    )
    if max_chunks is not None and len(chunks) > max_chunks:
        raise PackingBudgetError(f"Complete events need {len(chunks)} chunks; budget is {max_chunks}")
    raw_indices = tuple(sorted({index for event in store.events if event.event_id in view.raw_event_ids for index in event.source_indices}))
    raw_messages = [visible_message(store.messages[index]) for index in raw_indices]
    if not raw_messages:
        raise ValueError("A decision view requires an observable raw message")
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
    workspace_ids = full_ids[len(prefix_ids):]
    if max_raw_tokens is not None and len(full_ids) > max_raw_tokens:
        raise PackingBudgetError(f"System/tools plus complete raw workspace need {len(full_ids)} tokens; budget is {max_raw_tokens}")
    return PackedMemory(view, prefix_ids, workspace_ids, raw_indices, chunks)


def pack_target(
    tokenizer: Any, target: Message | Mapping[str, Any], *, max_target_tokens: int | None = None,
) -> tuple[int, ...]:
    """Native assistant continuation including its terminator, never sliced."""
    message = visible_message(target)
    if message["role"] != "assistant":
        raise ValueError("Only assistant messages are supervised")
    dummy = {"role": "user", "content": ""}
    prompt = native_ids(tokenizer, [dummy], generation=True)
    completion = native_ids(tokenizer, [dummy, message])
    if completion[:len(prompt)] != prompt:
        raise ValueError("Native assistant target is not a continuation of the generation prompt")
    target_ids = completion[len(prompt):]
    if max_target_tokens is not None and len(target_ids) > max_target_tokens:
        raise PackingBudgetError(f"Complete target needs {len(target_ids)} tokens; budget is {max_target_tokens}")
    return target_ids


def training_sequence(memory: PackedMemory, target_ids: Sequence[int]) -> dict[str, tuple[int, ...]]:
    """Unpadded ordinary inputs; CE is normalized per decision by the trainer."""
    if not target_ids:
        raise ValueError("An assistant target must contain tokens")
    ids = memory.workspace_input_ids + tuple(target_ids)
    return {
        "input_ids": ids,
        "labels": (-100,) * len(memory.workspace_input_ids) + tuple(target_ids),
        "attention_mask": (1,) * len(ids),
        "position_ids": tuple(range(memory.workspace_position_start, memory.workspace_position_start + len(ids))),
    }
