"""The training surface for history messages, shared by proxy and hf_server.

`train_data_multiturn._normal_agent_message` is what the checkpoint was
trained on: an assistant turn's tool calls live *inside* `content` as

    <existing content>\n\nAction:\n<tool_call>{minified json}</tool_call>

not in a separate OpenAI `tool_calls` field.  Both halves of the bench stack
have to agree on this, because they sit on opposite sides of the compression
boundary: hf_server renders raw history messages, while the proxy has to
render them *before* hashing them into a gist.  When only one side rendered,
every assistant tool-call turn that got compressed was handed to
/v1/c2kv/extract as the literal two-character string `""` -- the action was
deleted, not compressed.  Keeping the single implementation here is what stops
that from drifting apart again.
"""
from __future__ import annotations

import json
from typing import Any, Dict


def render_tool_calls(tool_calls: Any) -> str:
    """`Action:\\n<tool_call>...</tool_call>` for a list of OpenAI tool calls."""
    blocks = []
    for call in tool_calls or []:
        function = call.get("function") or {}
        raw_arguments = function.get("arguments")
        try:
            arguments = json.loads(raw_arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            arguments = raw_arguments or {}
        blocks.append(
            "<tool_call>\n"
            + json.dumps(
                {"name": function.get("name"), "arguments": arguments},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n</tool_call>"
        )
    return "Action:\n" + "\n".join(blocks) if blocks else ""


def message_text(message: Dict[str, Any]) -> str:
    """The exact string this message contributes to the model's context.

    Mirrors hf_server's in-place normalisation.  Non-string content is JSON
    encoded (tool results arrive as objects); `None` content with tool calls
    yields the action block alone rather than `json.dumps(None or "")`.
    """
    content = message.get("content")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False) if content else ""
    if message.get("role") == "assistant" and message.get("tool_calls"):
        action = render_tool_calls(message["tool_calls"])
        if action:
            return content + "\n\n" + action if content else action
    return content
