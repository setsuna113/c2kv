"""Render exact field corrections while retaining actual generation provenance."""
from __future__ import annotations

import json
from dataclasses import replace

from ..event_native_draft import parse_native_draft


def corrected_draft(draft, calls, *, benchmark):
    """Only field changes are allowed; action count, names, and IDs stay fixed."""
    if len(calls) != len(draft.tool_calls):
        raise ValueError("A Goal field patch cannot change action count")
    for old, new in zip(draft.tool_calls, calls):
        if (old.get("id") != new.get("id") or old.get("type") != new.get("type")
                or old["function"]["name"] != new["function"]["name"]):
            raise ValueError("A Goal field patch cannot change call identity")
    actions = [{"name": call["function"]["name"],
                "arguments": json.loads(call["function"]["arguments"])} for call in calls]
    if benchmark == "acebench":
        from ..acebench_source import parse_acebench_draft
        text = "[" + ", ".join(action["name"] + "(" + ", ".join(
            key + "=" + repr(value) for key, value in action["arguments"].items()) + ")"
            for action in actions) + "]"
        parsed = parse_acebench_draft(text, call_id_prefix="verify")
        if parsed.status != "tool_calls":
            raise ValueError("Corrected ACE action must preserve the official grammar")
        return replace(draft, text=text, content=text, tool_calls=tuple(calls),
                       reason="source_corrected_native_calls")
    blocks = "\n".join("<tool_call>" + json.dumps(action, ensure_ascii=False,
                        separators=(",", ":"), allow_nan=False) + "</tool_call>" for action in actions)
    text = (draft.content + "\n" if draft.content else "") + blocks
    if draft.reasoning_content is not None:
        text = "<think>" + draft.reasoning_content + "</think>\n" + text
    parsed = parse_native_draft(text, call_id_prefix="verify")
    if parsed.status != "tool_calls":
        raise ValueError("Corrected action must remain valid native tool calls")
    return replace(draft, text=text, tool_calls=tuple(calls), reason="source_corrected_native_calls")
