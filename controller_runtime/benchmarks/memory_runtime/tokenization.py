"""The observable SGLang function-tool serialization used by the 1088 profile."""
from __future__ import annotations

import copy

TOOL_SCHEMA_PROFILE = "sglang-function-full-v1"

# Tool-prologue schema modes for the native actor prompt.  ``sglang-full`` is
# the release default: the native prologue matches the Full SGLang actor
# (protocol.Function.model_dump(), c2kv_tools_dump=full).  ``raw`` passes the
# client's tool JSON through unchanged, i.e. the pre-2026-09-21 native
# prologue that kept benchmark ``response`` schemas and carried no ``strict``.
TOOL_SCHEMA_MODES = ("sglang-full", "raw")
DEFAULT_TOOL_SCHEMA = "sglang-full"


def serving_tools(tools):
    """Match SGLang protocol.Tool/Function.model_dump(), c2kv_tools_dump=full.

    The server drops unknown function fields (including benchmark response
    schemas) and emits the four declared fields in model-definition order.
    Passing the original client JSON directly to a tokenizer changes the
    common tool prologue even when the chat messages are identical.
    """
    if not tools:
        return None
    output = []
    for tool in tools:
        if tool.get("type", "function") != "function":
            raise ValueError("The 1088 tool profile accepts function tools only")
        function = tool["function"]
        output.append({"type": "function", "function": {
            "description": function.get("description"),
            "name": function["name"],
            "parameters": function.get("parameters"),
            "strict": function.get("strict", False),
        }})
    return output


def model_tools(tools, schema=DEFAULT_TOOL_SCHEMA):
    """Render the client tool list for the actor prompt under one schema mode.

    The result is always a fresh list; the caller's snapshot is never shared.
    Unknown modes are rejected so a misspelt flag cannot silently serve the
    default prologue.
    """
    if schema not in TOOL_SCHEMA_MODES:
        raise ValueError(
            f"Unknown tool schema mode {schema!r}; expected one of {TOOL_SCHEMA_MODES}")
    if schema == "raw":
        return [copy.deepcopy(dict(tool)) for tool in tools or []]
    return serving_tools(tools) or []
