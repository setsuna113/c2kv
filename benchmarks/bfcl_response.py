"""The BFCL assistant echo contract shared by its handler and exact-KV proxy."""
from __future__ import annotations


def normalize_native_message(message, response_id):
    """Parse complete native tool blocks without changing their actions.

    The handler echoes this structured representation. Exact-KV history must
    validate that echo, then restore the receipt's original generated text.
    Malformed drafts and already structured messages remain unchanged.
    """
    if message.get("tool_calls") or not isinstance(message.get("content"), str):
        return message
    from experiments.history_system.runtime.benchmarks.memory_runtime.event_native_draft import parse_native_draft

    draft = parse_native_draft(message["content"], call_id_prefix=f"bfcl_native_{response_id}")
    if draft.status != "tool_calls":
        return message
    return dict(message, tool_calls=list(draft.tool_calls), content=draft.content or None)
