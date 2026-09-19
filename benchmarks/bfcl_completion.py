"""Distinguish terminal benchmark failures from incomplete BFCL execution.

The original result and traceback must reach the official scorer unchanged.
A transport status alone never establishes a terminal model/method failure.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping


LEGACY_FC_DECODE_ERROR = "'str' object has no attribute 'items'"
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-JSON constant: {value}")


def _json_object(value) -> bool:
    if isinstance(value, Mapping):
        return True
    if not isinstance(value, str):
        return False
    try:
        return isinstance(json.loads(value, object_pairs_hook=_unique_object,
                                     parse_constant=_reject_constant), dict)
    except (TypeError, ValueError, OverflowError):
        return False


def _structured_tool_call(value) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("type") == "function" and isinstance(value.get("function"), Mapping):
        value = value["function"]
    elif set(value) != {"name", "arguments"}:
        if len(value) != 1:
            return False
        name, arguments = next(iter(value.items()))
        return isinstance(name, str) and bool(name) and _json_object(arguments)
    return (set(value) == {"name", "arguments"}
            and isinstance(value["name"], str) and bool(value["name"])
            and _json_object(value["arguments"]))


def _complete_native_tool_calls(text: str) -> bool:
    """Match the native draft parser's complete explicit tool-block contract."""
    body = text
    if body.lstrip().startswith("<think>"):
        start = body.index("<think>") + len("<think>")
        end = body.find("</think>", start)
        if end < 0:
            return False
        body = body[end + len("</think>"):]
    cursor = 0
    found = False
    while True:
        start = body.find(_TOOL_OPEN, cursor)
        closing = body.find(_TOOL_CLOSE, cursor)
        if start < 0:
            return found and closing < 0
        if 0 <= closing < start:
            return False
        end = body.find(_TOOL_CLOSE, start + len(_TOOL_OPEN))
        if end < 0:
            return False
        payload_text = body[start + len(_TOOL_OPEN):end]
        if _TOOL_OPEN in payload_text:
            return False
        try:
            payload = json.loads(payload_text, object_pairs_hook=_unique_object,
                                 parse_constant=_reject_constant)
        except (TypeError, ValueError, OverflowError):
            return False
        if not _structured_tool_call(payload) or set(payload) != {"name", "arguments"}:
            return False
        found = True
        cursor = end + len(_TOOL_CLOSE)


def _has_proven_tool_action(value) -> bool:
    if isinstance(value, str):
        return _complete_native_tool_calls(value)
    if isinstance(value, Mapping):
        return _structured_tool_call(value)
    if isinstance(value, list):
        return any(_structured_tool_call(item) for item in value)
    return False


def has_legacy_fc_decode_error(row: Mapping) -> bool:
    """Identify a tool action lost to the old string/FC-dict handler contract."""
    inference_log = row.get("inference_log")
    if not isinstance(inference_log, list):
        return False
    for turn in inference_log:
        if not isinstance(turn, Mapping):
            continue
        for key, entries in turn.items():
            if not isinstance(key, str) or not key.startswith("step_") or not isinstance(entries, list):
                continue
            if any(isinstance(entry, Mapping) and entry.get("role") == "tool"
                   for entry in entries):
                continue
            for index, entry in enumerate(entries):
                if (not isinstance(entry, Mapping)
                        or entry.get("role") != "handler_log"
                        or entry.get("error") != LEGACY_FC_DECODE_ERROR):
                    continue
                for previous in reversed(entries[:index]):
                    if not isinstance(previous, Mapping) or previous.get("role") != "assistant":
                        continue
                    if (_has_proven_tool_action(previous.get("model_response"))
                            or _has_proven_tool_action(previous.get("content"))):
                        return True
                    break
    return False


def completion_kind(row: Mapping, *, fc_model: bool = False) -> str:
    if fc_model and has_legacy_fc_decode_error(row):
        return "incomplete"
    if "result" not in row:
        return "incomplete"
    failure = row.get("traceback")
    if failure is None:
        return "model_output"
    text = failure if isinstance(failure, str) else json.dumps(failure, ensure_ascii=False)
    # Only the server's explicit typed method failure is terminal. Generic
    # runner failures and transport errors remain incomplete.
    if re.search(r"[\"']code[\"']\s*:\s*[\"']c2kv_capacity_infeasible[\"']", text):
        return "capacity_infeasible"
    if re.search(r"[\"']code[\"']\s*:\s*[\"']acon_history_budget_exceeded[\"']", text):
        return "acon_history_budget_exceeded"
    # SGLang's explicit context admission error, also recognized by prefix replay.
    # BFCL embeds the OpenAI exception repr, which can escape the apostrophe.
    if re.search(r"The input \(\d+ tokens\) is longer than the model\\*'s context length \(\d+ tokens\)", text):
        return "context_overflow"
    # These are actor-selected invalid retrievals, not backend availability errors.
    if any(marker in text for marker in (
        "HiAgent requested nonexistent completed subgoals",
        "HiAgent requested an already revealed trajectory without advancing",
    )):
        return "hiagent_invalid_retrieval"
    return "incomplete"


def bfcl_row_is_terminal(row: Mapping, *, fc_model: bool = False) -> bool:
    return completion_kind(row, fc_model=fc_model) != "incomplete"


def terminal_failure_kind(row: Mapping, *, fc_model: bool = False) -> str | None:
    kind = completion_kind(row, fc_model=fc_model)
    return kind if kind not in {"incomplete", "model_output"} else None
