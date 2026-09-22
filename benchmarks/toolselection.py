"""Versioned lexical policies for selecting native tool definitions.

This module intentionally has no dependency on ``toolmemory`` so the latter
can load it both as a regular module and through the history runtime's
path-based import.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_SELECTOR_POLICY = "last_user_topk_v1"
SELECTOR_POLICIES = (
    DEFAULT_SELECTOR_POLICY,
    "latest_event_topk_v1",
    "last_user_adaptive_v1",
)
SELECTOR_VERSION = "tool-selection-v1"
ADAPTIVE_RELATIVE_THRESHOLD = 0.5
MAX_QUERY_COMPONENT_CHARS = 16_384


def selector_version(policy: str) -> str:
    if policy not in SELECTOR_POLICIES:
        raise ValueError(f"unknown tool selector policy {policy!r}")
    return SELECTOR_VERSION


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), Mapping) else {}
    return str(function.get("name") or tool.get("name") or tool.get("tool_name")
               or tool.get("function_name") or "")


def _tool_search_text(tool: Mapping[str, Any]) -> str:
    function = tool.get("function") if isinstance(tool.get("function"), Mapping) else {}
    fields = [
        _tool_name(tool),
        function.get("description", ""),
        tool.get("description", ""),
        function.get("parameters", ""),
        tool.get("parameters", ""),
        tool.get("input_schema", ""),
        tool.get("schema", ""),
        tool.get("text", ""),
    ]
    return " ".join(item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
                    for item in fields if item)


def message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def last_user_query(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message_text(message)
    return message_text(messages[-1]) if messages else ""


def lexical_scores(tools: Sequence[Mapping[str, Any]], query: str) -> tuple[float, ...]:
    """The frozen name4/text1 score, aligned to catalog order."""
    query_tokens = set(_tokens(query))
    if not query_tokens:
        return tuple(0.0 for _ in tools)
    scores = []
    for tool in tools:
        name_overlap = len(query_tokens & set(_tokens(_tool_name(tool))))
        text_overlap = len(query_tokens & set(_tokens(_tool_search_text(tool))))
        scores.append(4.0 * name_overlap + float(text_overlap))
    return tuple(scores)


def rank_scores(scores: Sequence[float]) -> tuple[int, ...]:
    return tuple(sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index)))


def lexical_rank(tools: Sequence[Mapping[str, Any]], query: str) -> tuple[int, ...]:
    return rank_scores(lexical_scores(tools, query))


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def _bounded(text: str) -> str:
    return text[:MAX_QUERY_COMPONENT_CHARS]


def _latest_user(messages: Sequence[Mapping[str, Any]]) -> tuple[int | None, str]:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index, message_text(messages[index])
    return None, message_text(messages[-1]) if messages else ""


def _completed_execution_group(
    messages: Sequence[Mapping[str, Any]], user_index: int | None,
) -> str | None:
    """Return the latest completed tool-call batch after the current user.

    A batch with call ids is complete only when every id has a matching tool
    result. Producers without ids must provide at least one contiguous tool
    result per call. Assistant drafts and observations before the current user
    boundary are never included.
    """
    if user_index is None:
        return None
    latest: tuple[Sequence[Any], list[Mapping[str, Any]]] | None = None
    for index in range(user_index + 1, len(messages)):
        message = messages[index]
        calls = message.get("tool_calls")
        if message.get("role") != "assistant" or not isinstance(calls, list) or not calls:
            continue
        observations: list[Mapping[str, Any]] = []
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].get("role") == "tool":
            observations.append(messages[cursor])
            cursor += 1
        if not observations:
            continue
        call_ids = [call.get("id") if isinstance(call, Mapping) else None for call in calls]
        if all(isinstance(call_id, str) and call_id for call_id in call_ids):
            by_id = {observation.get("tool_call_id"): observation for observation in observations
                     if isinstance(observation.get("tool_call_id"), str)}
            if any(call_id not in by_id for call_id in call_ids):
                continue
            selected = [by_id[call_id] for call_id in call_ids]
        elif any(call_id is not None for call_id in call_ids):
            continue
        elif len(observations) >= len(calls):
            selected = observations[:len(calls)]
        else:
            continue
        latest = calls, selected
    if latest is None:
        return None
    calls, observations = latest
    action = _bounded(_stable_json(calls))
    results = _bounded(_stable_json([
        {"tool_call_id": observation.get("tool_call_id"),
         "content": observation.get("content")}
        for observation in observations
    ]))
    return "[executed_action]\n" + action + "\n[observation]\n" + results


def _normalized(scores: Sequence[float]) -> tuple[float, ...]:
    maximum = max(scores, default=0.0)
    if maximum <= 0.0:
        return tuple(0.0 for _ in scores)
    return tuple(float(score) / maximum for score in scores)


def _query_hash(*parts: str) -> str:
    return hashlib.sha256(_stable_json(list(parts)).encode("utf-8")).hexdigest()


def tool_selection(
    tools: Sequence[Mapping[str, Any]], spec: Any,
    messages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select catalog indices under the versioned policy carried by ``spec``."""
    policy = getattr(spec, "selector_policy", DEFAULT_SELECTOR_POLICY)
    if policy not in SELECTOR_POLICIES:
        raise ValueError(f"unknown tool selector policy {policy!r}")
    user_index, user_query = _latest_user(messages)
    scoring_user_query = user_query
    scores = lexical_scores(tools, scoring_user_query)
    rank = rank_scores(scores)
    event = None
    if policy == "latest_event_topk_v1":
        event = _completed_execution_group(messages, user_index)
        if event is not None:
            scoring_user_query = _bounded(user_query)
            scores = lexical_scores(tools, scoring_user_query)
            user_scores = _normalized(scores)
            event_scores = _normalized(lexical_scores(tools, event))
            scores = tuple(0.5 * (left + right)
                           for left, right in zip(user_scores, event_scores))
            rank = rank_scores(scores)
    if getattr(spec, "layout", "uniform") == "uniform":
        native = ()
    elif policy == "last_user_adaptive_v1":
        maximum = max(scores, default=0.0)
        native = tuple(sorted(index for index, score in enumerate(scores)
                              if score > 0.0
                              and score >= ADAPTIVE_RELATIVE_THRESHOLD * maximum))
    else:
        native = tuple(sorted(rank[:int(getattr(spec, "top_k", 0))]))
    hash_parts = ((scoring_user_query,) if event is None
                  else (scoring_user_query, event))
    return {
        "native_indices": native,
        "scores": tuple(float(score) for score in scores),
        "rank": rank,
        "policy": policy,
        "selector_version": selector_version(policy),
        "query_sha256": _query_hash(*hash_parts),
        "latest_io_present": event is not None,
    }
