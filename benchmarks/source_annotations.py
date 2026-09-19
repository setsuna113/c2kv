"""Exact source intervals for tool definitions already visible to benchmarks.

These helpers only identify text from a known producer. They never add a
catalog, parse the task's gold answer, or infer tools from arbitrary prose.
"""
from __future__ import annotations

import ast
import json
from typing import Any, Mapping, Sequence


def _ace_function_text(functions: Any) -> str:
    if functions is None or functions == [] or functions == {}:
        return ""
    return functions if isinstance(functions, str) else json.dumps(functions, ensure_ascii=False)


def _json_array_items(text: str) -> list[tuple[int, int]]:
    """Return exact top-level element slices, or one opaque catalog slice."""
    if not text.startswith("[") or not text.endswith("]"):
        return [(0, len(text))] if text else []
    decoder = json.JSONDecoder()
    index = 1
    items = []
    while index < len(text) - 1:
        while index < len(text) - 1 and text[index] in " \t\r\n,":
            index += 1
        if index >= len(text) - 1:
            break
        start = index
        try:
            _, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            return [(0, len(text))]
        items.append((start, index))
    return items or ([(0, len(text))] if text else [])


def acebench_function_spans(messages: Sequence[Mapping[str, Any]], functions: Any) -> list[dict[str, Any]]:
    """Locate ACEBench's actual ``functions`` insertion in its system text."""
    if not messages or not isinstance(messages[0].get("content"), str):
        return []
    text = _ace_function_text(functions)
    if not text:
        return []
    content = messages[0]["content"]
    start = content.rfind(text)
    if start < 0:
        raise ValueError("ACEBench functions are absent from the rendered system prompt")
    return [{"message_index": 0, "start": start + lo, "end": start + hi,
             "source": "acebench.functions"} for lo, hi in _json_array_items(text)]


_DOC_CALLS = {"show_app_descriptions", "show_api_descriptions", "show_api_doc"}


def pure_appworld_doc_action(code: str) -> str | None:
    """Recognize code whose *entire* output is from official api_docs calls."""
    try:
        body = ast.parse(code).body
    except SyntaxError:
        return None
    if not body:
        return None
    names = []
    for statement in body:
        if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
            return None
        printer = statement.value
        if not isinstance(printer.func, ast.Name) or printer.func.id != "print" or len(printer.args) != 1:
            return None
        call = printer.args[0]
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            return None
        docs = call.func.value
        if not (isinstance(docs, ast.Attribute) and docs.attr == "api_docs"
                and isinstance(docs.value, ast.Name) and docs.value.id == "apis"
                and call.func.attr in _DOC_CALLS):
            return None
        names.append(call.func.attr)
    return "appworld.api_docs." + "+".join(names)


def appworld_doc_spans(messages: Sequence[Mapping[str, Any]],
                       observed_outputs: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    """Bind executed api_docs outputs to subsequent user observation turns."""
    spans = []
    # ACON renders AppWorld's execution result as a user turn following the
    # assistant action. Restrict matches to that producer/observation pair:
    # the same bytes in a system instruction or quoted assistant code are not
    # a newly supplied tool definition.
    observations = [index for index in range(1, len(messages))
                    if messages[index].get("role") == "user"
                    and messages[index - 1].get("role") == "assistant"
                    and isinstance(messages[index].get("content"), str)]
    occupied: set[int] = set()
    for source, output in observed_outputs:
        if not output:
            continue
        for message_index in observations:
            if message_index in occupied:
                continue
            found = messages[message_index]["content"].find(output)
            if found < 0:
                continue
            spans.append({"message_index": message_index, "start": found,
                          "end": found + len(output), "source": source})
            occupied.add(message_index)
            break
    return sorted(spans, key=lambda item: (item["message_index"], item["start"]))
