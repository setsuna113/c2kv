"""Incremental multi-turn history baselines.

The methods in this module deliberately share one append-only protocol.  The
benchmark sends a growing transcript, but a method only folds newly closed
turns into its state.  The raw transcript is retained as an external archive;
it is never silently re-read as model input on the next request.

These are text-level compatibility baselines for the CUDA experiment.  They
are useful for comparing lifecycle and recovery policies through the existing
OpenAI proxy, but they do not claim to reproduce a paper's private KV kernel.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


METHODS = ("agentfold", "commitkv", "agentkv")
MODEL_FAMILY = "qwen3-4b"
MAX_RETROSPECTIVE_RECOVERIES = 1


class TextarmCompressorError(RuntimeError):
    """The compatibility compressor did not produce a usable memory unit."""

    kind = "history_method_compressor_error"


def normalize_model_family(value: str | None) -> str:
    value = str(value or "").strip().lower().replace("_", "-")
    value = re.sub(r"\s+", "-", value)
    if value in {"qwen3-4b", "qwen-3-4b", "qwen3-4b-instruct-2507",
                 "qwen-3-4b-instruct-2507"}:
        return MODEL_FAMILY
    return value


def require_model_family(value: str | None) -> None:
    family = normalize_model_family(value)
    if family != MODEL_FAMILY:
        raise ValueError(
            "multi-turn history methods require model family qwen3-4b; "
            f"got {value!r}")


def _canonical_message(message: Mapping[str, Any]) -> str:
    return json.dumps(message, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), default=str)


def _turn_id(turn: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(_canonical_message(message) for message in turn)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _content(message: Mapping[str, Any]) -> str:
    value = message.get("content")
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _has_tool_call(message: Mapping[str, Any]) -> bool:
    return bool(message.get("role") == "assistant" and message.get("tool_calls"))


def _is_commit_turn(turn: Sequence[Mapping[str, Any]]) -> bool:
    return any(_has_tool_call(message) for message in turn) and any(
        message.get("role") == "tool" for message in turn)


def _render_turn(turn: Sequence[Mapping[str, Any]], action_dialect) -> str:
    lines: List[str] = []
    for message in turn:
        role = message.get("role") or "user"
        if role == "assistant" and message.get("tool_calls"):
            lines.append("Action: " + action_dialect(dict(message)))
        elif role == "tool":
            lines.append("Observation: " + _content(message))
        else:
            lines.append(f"{role.title()}: " + _content(message))
    return "\n".join(lines)


def _words(text: str) -> set[str]:
    return {word for word in re.findall(r"[A-Za-z0-9_:-]{3,}", text.lower())}


@dataclass
class HistoryMethodState:
    """Per-conversation state owned by one proxy process."""

    system_messages: List[Dict[str, Any]] = field(default_factory=list)
    source_messages: List[Dict[str, Any]] = field(default_factory=list)
    raw_archive: List[List[Dict[str, Any]]] = field(default_factory=list)
    compressed: List[Dict[str, Any]] = field(default_factory=list)
    turn_ids: set[str] = field(default_factory=set)
    recovered_turn_ids: set[str] = field(default_factory=set)
    recovery_count: int = 0
    version: int = 0


Compressor = Callable[[Dict[str, Any]], str]


def _split_turns(messages: Sequence[Mapping[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split non-system messages at user boundaries without changing payloads."""
    starts = [index for index, message in enumerate(messages)
              if message.get("role") == "user"]
    if not starts:
        return []
    turns: List[List[Dict[str, Any]]] = []
    for offset, start in enumerate(starts):
        end = starts[offset + 1] if offset + 1 < len(starts) else len(messages)
        turns.append([copy.deepcopy(dict(message)) for message in messages[start:end]])
    return turns


def _summary_payload(method: str, level: str, rendered: str, model: str) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "You maintain an append-only agent memory. Preserve exact tool "
                "names, identifiers, arguments, constraints and observations. "
                "Do not invent facts.")},
            {"role": "user", "content": (
                f"Method={method}; fold_level={level}. Summarize this newly "
                "closed action-observation turn in one compact memory unit.\n\n"
                + rendered)},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": 256 if level == "granular" else 384,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _fold_level(method: str, turn: Sequence[Mapping[str, Any]]) -> str:
    if method != "agentfold":
        return "commit" if method == "commitkv" else "memory_unit"
    text_len = sum(len(_content(message)) for message in turn)
    # AgentFold's learned choice is represented by the same two protocol
    # levels, with a deterministic threshold so the baseline is reproducible.
    return "granular" if text_len <= 1200 else "deep"


def transform(
    messages: Sequence[Mapping[str, Any]],
    state: HistoryMethodState,
    method: str,
    compressor: Compressor,
    action_dialect,
    *,
    model: str = "qwen3-4b",
    model_family: str = MODEL_FAMILY,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply one incremental history method to a growing transcript.

    ``messages`` may contain the complete benchmark transcript.  Only turns
    that became closed since the previous call are sent to ``compressor``.
    A non-append edit is rejected because silently rebuilding state would
    violate the persistent-history contract.
    """
    method = str(method).strip().lower()
    if method not in METHODS:
        raise ValueError(f"unknown multi-turn history method {method!r}")
    require_model_family(model_family)

    system = [copy.deepcopy(dict(message)) for message in messages
              if message.get("role") == "system"]
    body = [copy.deepcopy(dict(message)) for message in messages
            if message.get("role") != "system"]
    if state.system_messages and system != state.system_messages:
        raise ValueError(
            f"{method} requires an unchanged system prefix across history requests")
    if state.source_messages and body[:len(state.source_messages)] != state.source_messages:
        raise ValueError(
            f"{method} requires append-only history; the request rewrote a prior event")
    state.system_messages = copy.deepcopy(system)
    state.source_messages = copy.deepcopy(body)
    state.version += 1

    turns = _split_turns(body)
    last_user = max((index for index, message in enumerate(body)
                     if message.get("role") == "user"), default=None)
    current_start = last_user if last_user is not None else len(body)
    stats: Dict[str, Any] = {
        "method": method,
        "protocol": "append_only_event_state_v1",
        "implementation": "text_surrogate",
        "model_family": MODEL_FAMILY,
        "state_version": state.version,
        "source_messages_seen": len(body),
        "processed_new_turns": 0,
        "compressed_turns": 0,
        "recovery": {"attempted": False, "count": state.recovery_count,
                      "max": MAX_RETROSPECTIVE_RECOVERIES},
    }

    # A turn is eligible only after a later user message exists.  This keeps
    # the just-returned tool observation raw for the next model decision.
    eligible: List[List[Dict[str, Any]]] = []
    cursor = 0
    for turn in turns:
        turn_end = cursor + len(turn)
        if turn and turn_end <= current_start:
            eligible.append(turn)
        cursor = turn_end
    for turn in eligible:
        tid = _turn_id(turn)
        if tid in state.turn_ids:
            continue
        state.raw_archive.append(copy.deepcopy(turn))
        state.turn_ids.add(tid)
        stats["processed_new_turns"] += 1
        if method == "commitkv" and not _is_commit_turn(turn):
            # CommitKV does not retire an uncommitted turn.  Keep it in the
            # external archive and expose it as raw state below.
            state.compressed.append({"turn_id": tid, "raw": copy.deepcopy(turn),
                                     "committed": False})
            continue
        rendered = _render_turn(turn, action_dialect)
        level = _fold_level(method, turn)
        summary = compressor(_summary_payload(method, level, rendered, model)).strip()
        if not summary:
            raise ValueError(f"{method} compressor returned an empty memory unit")
        state.compressed.append({
            "turn_id": tid,
            "summary": summary,
            "level": level,
            "source_message_count": len(turn),
            "committed": True,
        })
        stats["compressed_turns"] += 1

    rendered_history: List[Dict[str, Any]] = []
    for item in state.compressed:
        if not item.get("committed"):
            rendered_history.extend(copy.deepcopy(item["raw"]))
            continue
        label = f"[{method} {item['level']} memory unit] "
        rendered_history.append({"role": "user", "content": label + item["summary"]})

    # AgentKV's extra path is an explicit, bounded dereference of one archived
    # event.  It is query-driven and never replays more than one unit.
    if method == "agentkv" and state.recovery_count < MAX_RETROSPECTIVE_RECOVERIES:
        query = _content(body[last_user]) if last_user is not None else ""
        query_words = _words(query)
        candidates: List[Tuple[int, int, str, List[Dict[str, Any]]]] = []
        for index, turn in enumerate(state.raw_archive):
            tid = _turn_id(turn)
            if tid in state.recovered_turn_ids:
                continue
            score = len(query_words & _words(_render_turn(turn, action_dialect)))
            if score:
                candidates.append((score, -index, tid, turn))
        if candidates:
            _, _, tid, turn = max(candidates)
            rendered_history.append({
                "role": "user",
                "content": "[agentkv retrospective recovery]\n" +
                           _render_turn(turn, action_dialect),
            })
            state.recovered_turn_ids.add(tid)
            state.recovery_count += 1
            stats["recovery"] = {"attempted": True, "count": state.recovery_count,
                                  "max": MAX_RETROSPECTIVE_RECOVERIES,
                                  "turn_id": tid, "candidate_count": len(candidates)}

    # The active suffix is the only new model-visible raw input.  It includes
    # the latest user and any action/observation feedback after that user.
    rendered_history.extend(copy.deepcopy(body[current_start:]))
    output = system + rendered_history
    stats["output_messages"] = len(output)
    stats["archive_turns"] = len(state.raw_archive)
    stats["memory_units"] = sum(1 for item in state.compressed if item.get("committed"))
    return output, stats
