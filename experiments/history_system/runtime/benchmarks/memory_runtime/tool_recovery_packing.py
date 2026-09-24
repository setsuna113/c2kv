"""Replace only the native system prefix after a structured tool-plan change."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from history_memory.events import EventStore, RenderedMessages
from history_memory.packing import PackedMemory, native_ids


class ToolRecoveryContextExceeded(ValueError):
    """The rebuilt native prompt cannot fit the physical model context."""


def _system_prefix(tokenizer: Any, messages: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    system = []
    for message in messages:
        if message["role"] != "system":
            break
        system.append(message)
    if not system:
        raise ValueError("Tool re-rendering requires a leading system message")
    dummy = {"role": "user", "content": ""}
    dummy_ids = native_ids(tokenizer, [dummy])
    prefix_and_dummy = native_ids(tokenizer, [*system, dummy])
    if not dummy_ids or prefix_and_dummy[-len(dummy_ids):] != dummy_ids:
        raise ValueError("Native template has no separable system prefix")
    return prefix_and_dummy[:-len(dummy_ids)]


def rebuild_tool_history_memory(
    tokenizer: Any,
    selected_memory: PackedMemory,
    selected_metadata: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    old_rendered_messages: Sequence[Mapping[str, Any]],
    new_rendered_messages: Sequence[Mapping[str, Any]],
    *,
    ratio: int,
    max_new_tokens: int,
    model_context: int,
    benchmark: str | None = None,
) -> PackedMemory:
    """Keep the selected history/workspace byte-for-byte and replace its tool prefix.

    The inner controller has already selected and packed history. Its workspace
    may contain capacity projections, derived observations, or source carries;
    none is reconstructed here. T0/schema places structured tool definitions in
    the leading system protocol and passes no native ``tools`` to the template.
    The caller attaches the replacement plan's tool chunks after this function.
    """
    if not isinstance(selected_memory, PackedMemory) or not isinstance(selected_metadata, Mapping):
        raise TypeError("Expected selected history memory and metadata")
    if (selected_memory.raw_tool_segments or selected_memory.tool_gist_segments
            or any(chunk.projection_set == "tool" for chunk in selected_memory.chunks)):
        raise ValueError("Tool recovery requires history-only selected memory")
    if (isinstance(ratio, bool) or not isinstance(ratio, int) or ratio <= 0
            or isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int)
            or max_new_tokens < 0 or isinstance(model_context, bool)
            or not isinstance(model_context, int) or model_context <= 0):
        raise ValueError("Invalid ratio, completion length, or model context")
    session_id = source_payload["session_id"]
    source_messages = source_payload["messages"]
    if not source_messages:
        raise ValueError("Tool re-rendering requires source messages")
    # T0/schema writes its protocol into the leading system message and inserts
    # one when the source has none (a BFCL task starts with the user request).
    # Both plans then share that rendered frame, one index after the source.
    inserted = 0 if source_messages[0].get("role") == "system" else 1
    if (len(old_rendered_messages) != len(source_messages) + inserted
            or len(new_rendered_messages) != len(old_rendered_messages)):
        raise ValueError("Tool re-rendering cannot change source message indices")
    if inserted and any(messages[0].get("role") != "system"
                        for messages in (old_rendered_messages, new_rendered_messages)):
        raise ValueError("Tool re-rendering requires a leading protocol system message")
    resolved_benchmark = benchmark if benchmark is not None else source_payload.get("benchmark")
    source_store = EventStore.from_messages(session_id, source_messages, benchmark=resolved_benchmark)
    old_store = EventStore.from_messages(
        session_id, RenderedMessages(old_rendered_messages, source=source_messages),
        benchmark=resolved_benchmark,
    )
    new_store = EventStore.from_messages(
        session_id, RenderedMessages(new_rendered_messages, source=source_messages),
        benchmark=resolved_benchmark,
    )
    # With an inserted protocol message the event indices of the rendered frame
    # are shifted; the message-by-message check below binds them to the source.
    if ((not inserted and source_store.events != old_store.events)
            or old_store.events != new_store.events):
        raise ValueError("Tool re-rendering changed event or source identity")
    aligned = [(None, old_store.messages[0], new_store.messages[0])] if inserted else []
    aligned += zip(source_store.messages, old_store.messages[inserted:],
                   new_store.messages[inserted:], strict=True)
    for source, old, new in aligned:
        source_message, old_message, new_message = (
            None if source is None else source.to_dict(), old.to_dict(), new.to_dict(),
        )
        if old_message["role"] != "system":
            if source_message != old_message or old_message != new_message:
                raise ValueError("Tool re-rendering changed a non-system source message")
        elif {key: value for key, value in old_message.items() if key != "content"} != {
            key: value for key, value in new_message.items() if key != "content"
        }:
            raise ValueError("Tool re-rendering changed system message fields")
    selected_memory.view.validate(old_store)
    old_prefix = _system_prefix(tokenizer, old_rendered_messages)
    if selected_memory.system_input_ids != old_prefix:
        raise ValueError("Selected memory does not match the original tool prefix")
    new_prefix = _system_prefix(tokenizer, new_rendered_messages)
    rebuilt = replace(selected_memory, system_input_ids=new_prefix)
    logical_end = rebuilt.workspace_position_start + len(rebuilt.workspace_input_ids)
    resident_end = rebuilt.costs(ratio)["resident_kv_tokens"]
    if max(logical_end, resident_end) + max_new_tokens > model_context:
        raise ToolRecoveryContextExceeded("Tool re-rendering exceeds the model context")
    return rebuilt
