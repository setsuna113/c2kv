"""Training-dialect rendering for assistant tool calls (shared by proxy + hf_server).

The training surface (train_data_multiturn._normal_agent_message /
_render_agent_output_messages) renders an assistant turn as

    content + "\\n\\n" + "Action:\\n" + "\\n".join(<tool_call> blocks)

with minified JSON inside each block.  Both the RAW path (hf_server chat) and
the COMPRESSED path (proxy before /v1/c2kv/extract) must produce this exact
text — historically the proxy sent the literal `""` for content=None
tool-call turns, silently deleting every historical assistant action from
compressed history.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List


def render_action_dialect(content: str, tool_calls: List[Dict[str, Any]]) -> str:
    """content + Action block in the training dialect (minified JSON)."""
    blocks = []
    for call in tool_calls or []:
        function = call.get("function") or {}
        raw_args = function.get("arguments", call.get("arguments"))
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                raw_args = {"_raw": raw_args}
        blocks.append(
            "<tool_call>\n"
            + json.dumps(
                {"name": function.get("name") or call.get("name"),
                 "arguments": raw_args or {}},
                ensure_ascii=False, separators=(",", ":"),
            )
            + "\n</tool_call>"
        )
    if not blocks:
        return content
    action = "Action:\n" + "\n".join(blocks)
    return content + "\n\n" + action if content else action


def normalize_message_content(message: Dict[str, Any]) -> str:
    """Full content of a message in the training dialect.

    - assistant turns with tool_calls render content + Action block
    - non-string content is JSON-dropped as before
    - tool_calls are popped from a COPY; the caller's dict is not mutated
    """
    content = message.get("content")
    content = content if isinstance(content, str) else (
        json.dumps(content, ensure_ascii=False) if content else ""
    )
    tool_calls = message.get("tool_calls")
    if message.get("role") == "assistant" and tool_calls:
        content = render_action_dialect(content, tool_calls)
    return content
