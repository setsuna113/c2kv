"""Source-preserving adapters from the G corpora to canonical sessions.

The old G training loaders render tool calls as text and relabel tool results
as users.  These adapters read the same raw rows but emit visible OpenAI
messages, preserving message roles and call IDs. Explicit OTel tool-response
parts become tool messages, with the container role retained in provenance.
A deterministic ID is created only when a legacy source has no ID. Missing result IDs require a
unique binding; only Toucan opts into its source-defined list order. Every
inferred binding is counted in the audit.
"""

from __future__ import annotations

import ast
import copy
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence


DEFAULT_QA_TARGET_FRACTION = 0.15
_HELDOUT_BENCHMARK_RE = re.compile(
    r"(?:^|[:/_-])(?:test|eval|evaluation|dev|validation)(?:$|[:/_-])",
    re.IGNORECASE,
)
_LONGMAGPIE_SEGMENT_RE = re.compile(r"[^.!?]*\?[\"'”’)\]]*\s*")
_ALLOWED_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})


class SourceRowError(ValueError):
    """A raw row cannot be converted without ambiguous or lost provenance."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class LoadedSources:
    rows: tuple[dict[str, Any], ...]
    audit: Mapping[str, int]


def _loads(value: Any, *, field: str) -> Any:
    if not isinstance(value, str):
        return copy.deepcopy(value)
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise SourceRowError("invalid_json", f"{field}: {exc}") from exc


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text_content(
    value: Any, *, audit: MutableMapping[str, int] | None = None
) -> str | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, list):
        text: list[str] = []
        for part in value:
            if isinstance(part, Mapping) and part.get("type") in {
                "tool_call",
                "function_call",
            }:
                if audit is not None:
                    audit["content_tool_parts_promoted"] = (
                        audit.get("content_tool_parts_promoted", 0) + 1
                    )
                continue
            if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                text.append(part["text"])
            elif isinstance(part, Mapping) and isinstance(part.get("content"), str):
                text.append(part["content"])
            elif isinstance(part, str):
                text.append(part)
            else:
                raise SourceRowError("non_text_content", type(part).__name__)
        return "\n".join(text)
    raise SourceRowError("non_text_content", type(value).__name__)


def _message_parts(message: Mapping[str, Any]) -> list[Any]:
    parts = _loads(message.get("parts") or [], field="message.parts")
    if isinstance(parts, Mapping):
        parts = [parts]
    if not isinstance(parts, list):
        raise SourceRowError("invalid_parts", type(parts).__name__)
    return parts


def _raw_tool_calls(message: Mapping[str, Any]) -> list[Any]:
    explicit = message.get("tool_calls")
    if explicit is None:
        explicit = message.get("toolCalls")
    if explicit is None and message.get("function_call") is not None:
        explicit = [message["function_call"]]
    explicit = _loads(explicit or [], field="message.tool_calls")
    if isinstance(explicit, Mapping):
        explicit = [explicit]
    if not isinstance(explicit, list):
        raise SourceRowError("invalid_tool_calls", type(explicit).__name__)
    part_calls = [
        part
        for part in _message_parts(message)
        if isinstance(part, Mapping)
        and part.get("type") in {"tool_call", "function_call"}
    ]
    result: list[Any] = []
    identities: set[str] = set()
    for raw in [*explicit, *part_calls]:
        parsed = _loads(raw, field="tool_call")
        if not isinstance(parsed, Mapping):
            raise SourceRowError("invalid_tool_call", type(parsed).__name__)
        function = (
            parsed.get("function")
            if isinstance(parsed.get("function"), Mapping)
            else parsed
        )
        identity = _json_key(
            [
                parsed.get("id") or parsed.get("tool_call_id"),
                function.get("name")
                or parsed.get("tool_name")
                or parsed.get("function_name"),
                function.get("arguments", parsed.get("arguments", parsed.get("args", parsed.get("input", {})))),
            ]
        )
        if identity not in identities:
            result.append(parsed)
            identities.add(identity)
    return result


def _normalize_call(
    raw: Any,
    *,
    namespace: str,
    source_index: int,
    call_index: int,
    audit: MutableMapping[str, int],
) -> dict[str, Any]:
    raw = _loads(raw, field="tool_call")
    if not isinstance(raw, Mapping):
        raise SourceRowError("invalid_tool_call", type(raw).__name__)
    function = raw.get("function") if isinstance(raw.get("function"), Mapping) else raw
    name = function.get("name") or raw.get("tool_name") or raw.get("function_name")
    if not isinstance(name, str) or not name:
        raise SourceRowError("missing_tool_name")
    arguments = function.get("arguments")
    if arguments is None:
        arguments = raw.get("arguments", raw.get("args", raw.get("input", {})))
    call_id = raw.get("id") or raw.get("tool_call_id")
    if not isinstance(call_id, str) or not call_id:
        call_id = f"{namespace}:m{source_index}:c{call_index}"
        audit["generated_tool_call_ids"] = audit.get("generated_tool_call_ids", 0) + 1
    raw_type = raw.get("type")
    if raw_type not in (None, "function", "tool_call", "function_call"):
        audit["tool_call_types_normalized_to_function"] = (
            audit.get("tool_call_types_normalized_to_function", 0) + 1
        )
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": copy.deepcopy(arguments)},
    }


def normalize_openai_messages(
    raw_messages: Sequence[Any],
    *,
    namespace: str,
    audit: MutableMapping[str, int] | None = None,
    source_indices: Sequence[Any] | None = None,
    normalized_index_map: list[Any] | None = None,
    allow_ordered_tool_result_binding: bool = False,
) -> list[dict[str, Any]]:
    """Normalize visible text/tool aliases without changing role semantics.

    Missing result IDs are inferred only when the binding is unique.  A
    source adapter may opt into ordered binding when that source explicitly
    defines tool results to correspond to pending calls in list order.
    ``normalized_index_map`` records one source coordinate for every emitted
    message so adapter transformations remain auditable.
    """
    counts: MutableMapping[str, int] = audit if audit is not None else Counter()
    if source_indices is not None and len(source_indices) != len(raw_messages):
        raise ValueError("source_indices must align one-to-one with raw_messages")
    messages: list[dict[str, Any]] = []
    pending: list[tuple[str, str]] = []
    for source_index, raw in enumerate(raw_messages):
        if not isinstance(raw, Mapping):
            raise SourceRowError("invalid_message", f"index={source_index}")
        role = raw.get("role") or raw.get("type")
        if role not in _ALLOWED_ROLES:
            raise SourceRowError("unsupported_role", f"index={source_index} role={role!r}")
        content_value = raw.get("content")
        if content_value is None and raw.get("parts") is not None:
            content_value = _message_parts(raw)
        content = _text_content(content_value, audit=counts)
        message: dict[str, Any] = {"role": role, "content": content}
        if isinstance(raw.get("name"), str):
            message["name"] = raw["name"]

        if role == "assistant":
            calls = [
                _normalize_call(
                    call,
                    namespace=namespace,
                    source_index=source_index,
                    call_index=call_index,
                    audit=counts,
                )
                for call_index, call in enumerate(_raw_tool_calls(raw))
            ]
            if calls:
                message["tool_calls"] = calls
                for call in calls:
                    call_id = call["id"]
                    if any(pending_id == call_id for pending_id, _ in pending):
                        raise SourceRowError("duplicate_pending_call_id", call_id)
                    pending.append((call_id, call["function"]["name"]))
            if not calls and content in (None, ""):
                counts["empty_assistant_messages_skipped"] = (
                    counts.get("empty_assistant_messages_skipped", 0) + 1
                )
                continue
        elif role == "tool":
            call_id = raw.get("tool_call_id") or raw.get("toolCallId")
            if not isinstance(call_id, str) or not call_id:
                if not pending:
                    raise SourceRowError("unmatched_tool_result", f"index={source_index}")
                result_name = raw.get("name")
                named = [
                    pending_id
                    for pending_id, function_name in pending
                    if isinstance(result_name, str) and function_name == result_name
                ]
                if len(named) == 1:
                    call_id = named[0]
                    counts["tool_result_ids_bound_by_unique_name"] = (
                        counts.get("tool_result_ids_bound_by_unique_name", 0) + 1
                    )
                elif len(pending) == 1:
                    call_id = pending[0][0]
                    counts["tool_result_ids_bound_by_single_pending"] = (
                        counts.get("tool_result_ids_bound_by_single_pending", 0) + 1
                    )
                elif allow_ordered_tool_result_binding:
                    call_id = pending[0][0]
                    counts["tool_result_ids_bound_by_source_order"] = (
                        counts.get("tool_result_ids_bound_by_source_order", 0) + 1
                    )
                else:
                    raise SourceRowError(
                        "ambiguous_tool_result_binding", f"index={source_index}"
                    )
                counts["generated_tool_result_bindings"] = (
                    counts.get("generated_tool_result_bindings", 0) + 1
                )
            pending_ids = [pending_id for pending_id, _ in pending]
            if call_id not in pending_ids:
                raise SourceRowError("unmatched_tool_result", call_id)
            pending.pop(pending_ids.index(call_id))
            message["tool_call_id"] = call_id
        messages.append(message)
        if normalized_index_map is not None:
            normalized_index_map.append(
                copy.deepcopy(
                    source_indices[source_index]
                    if source_indices is not None
                    else source_index
                )
            )
    return messages


def _parse_tools(value: Any) -> list[dict[str, Any]]:
    tools = _loads(value or [], field="tools")
    if isinstance(tools, Mapping):
        tools = [tools]
    if not isinstance(tools, list):
        raise SourceRowError("invalid_tools", type(tools).__name__)
    result: list[dict[str, Any]] = []
    for item in tools:
        tool = _loads(item, field="tools[]")
        if not isinstance(tool, Mapping):
            raise SourceRowError("invalid_tool_definition", type(tool).__name__)
        result.append(copy.deepcopy(dict(tool)))
    return result


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, Mapping) and isinstance(function.get("name"), str):
        return function["name"]
    if isinstance(tool.get("name"), str):
        return tool["name"]
    raise SourceRowError("missing_tool_definition_name")


def _merge_tools(
    current: list[dict[str, Any]], incoming: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_name = {_tool_name(tool): _json_key(tool) for tool in current}
    for tool in incoming:
        name = _tool_name(tool)
        encoded = _json_key(tool)
        if name in by_name and by_name[name] != encoded:
            raise SourceRowError("conflicting_tool_definition", name)
        if name not in by_name:
            current.append(tool)
            by_name[name] = encoded
    return current


def _span_attributes(span: Any) -> Mapping[str, Any]:
    span = _loads(span, field="span")
    if not isinstance(span, Mapping):
        raise SourceRowError("invalid_span")
    attributes = _loads(span.get("attributes", span), field="span.attributes")
    if not isinstance(attributes, Mapping):
        raise SourceRowError("invalid_span_attributes")
    return attributes


def _sort_spans(value: Any) -> list[Mapping[str, Any]]:
    spans = _loads(value or [], field="spans")
    if not isinstance(spans, list):
        raise SourceRowError("invalid_spans")
    parsed = [span for span in spans if isinstance(span, Mapping)]
    return sorted(
        parsed,
        key=lambda span: (str(span.get("start_time") or ""), str(span.get("span_id") or "")),
    )


def _message_sequence(value: Any, *, field: str, default_role: str | None = None) -> list[Any]:
    messages = _loads(value, field=field)
    if isinstance(messages, Mapping):
        messages = [messages]
    if not isinstance(messages, list):
        raise SourceRowError("invalid_messages", field)
    if default_role is None:
        return messages
    result = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise SourceRowError("invalid_message", field)
        item = dict(message)
        item.setdefault("role", default_role)
        result.append(item)
    return result


def _benchmark_reason(benchmark: str) -> str | None:
    if "airline" in benchmark.casefold():
        return "excluded_airline"
    if _HELDOUT_BENCHMARK_RE.search(benchmark):
        return "excluded_heldout_benchmark"
    return None


def _trace_messages(
    value: Any, *, field: str, span_id: str, audit: MutableMapping[str, int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decode explicit OTel tool responses, including user-role containers."""
    raw_messages = _message_sequence(
        value, field=field,
        default_role="assistant" if field == "gen_ai.output.messages" else None,
    )
    messages: list[dict[str, Any]] = []
    coordinates: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_messages):
        coordinate = {"span_id": span_id, "field": field, "index": index}
        parts = _message_parts(raw)
        if any(isinstance(part, Mapping) and part.get("type") == "tool_call_response"
               for part in parts):
            if raw.get("role") not in {"user", "tool"}:
                raise SourceRowError("invalid_tool_response_container")
            if raw.get("content") not in (None, ""):
                raise SourceRowError("ambiguous_tool_response_content")
            for part_index, part in enumerate(parts):
                if isinstance(part, Mapping) and part.get("type") == "tool_call_response":
                    if "result" not in part:
                        raise SourceRowError("missing_tool_response_result")
                    result = part["result"]
                    messages.append({
                        "role": "tool", "tool_call_id": part.get("id"),
                        "content": result if isinstance(result, str) else _json_key(result),
                    })
                    audit["trace_tool_response_parts_promoted"] = audit.get(
                        "trace_tool_response_parts_promoted", 0
                    ) + 1
                else:
                    messages.append({"role": raw.get("role"), "parts": [part]})
                coordinates.append({**coordinate, "part_index": part_index,
                                    "source_role": raw.get("role")})
            continue
        message = dict(raw)
        # Some harnesses record blank assistant call text as this exact sentinel
        # when replaying an output in the following span's input.
        if raw.get("role") == "assistant" and _raw_tool_calls(raw):
            content = _text_content(raw.get("content") if raw.get("content") is not None
                                    else parts)
            if content is None or content.strip() in {"", "(no content)"}:
                message["content"] = ""
                audit["trace_empty_call_text_normalized"] = audit.get(
                    "trace_empty_call_text_normalized", 0
                ) + 1
        messages.append(message)
        coordinates.append(coordinate)
    return messages, coordinates


def adapt_agent_trace_row(
    row: Mapping[str, Any],
    *,
    split: str = "train",
    audit: MutableMapping[str, int] | None = None,
) -> dict[str, Any]:
    """Merge chronological trace spans into one prefix-consistent session."""
    if split != "train":
        raise SourceRowError("non_train_split", split)
    counts: MutableMapping[str, int] = audit if audit is not None else Counter()
    session_id = str(row.get("session_id") or row.get("trace_id") or row.get("id") or "").strip()
    if not session_id:
        raise SourceRowError("missing_session_id")
    benchmark = str(
        row.get("benchmark")
        or row.get("subset")
        or row.get("dataset")
        or row.get("task")
        or "unknown"
    )
    if reason := _benchmark_reason(benchmark):
        raise SourceRowError(reason, benchmark)

    namespace = f"agent-llm-traces:{session_id}"
    conversation: list[dict[str, Any]] = []
    conversation_sources: list[Any] = []
    tools: list[dict[str, Any]] = []
    usable_spans = 0
    for span_index, span in enumerate(_sort_spans(row.get("spans"))):
        attributes = _span_attributes(span)
        span_identity = str(span.get("span_id") or span_index)
        if attributes.get("gen_ai.tool.definitions") not in (None, "", []):
            _merge_tools(tools, _parse_tools(attributes["gen_ai.tool.definitions"]))
        raw_input = attributes.get("gen_ai.input.messages")
        raw_output = attributes.get("gen_ai.output.messages")
        if raw_input is None or raw_output is None:
            counts["spans_missing_messages"] = counts.get("spans_missing_messages", 0) + 1
            continue
        input_messages, input_sources = _trace_messages(
            raw_input, field="gen_ai.input.messages", span_id=span_identity, audit=counts
        )
        output_messages, output_sources = _trace_messages(
            raw_output, field="gen_ai.output.messages", span_id=span_identity, audit=counts
        )
        normalized_input = normalize_openai_messages(input_messages, namespace=namespace)
        complete_sources = [*input_sources, *output_sources]
        normalized_complete_sources: list[Any] = []
        normalized_complete = normalize_openai_messages(
            [*input_messages, *output_messages],
            namespace=namespace,
            audit=counts,
            source_indices=complete_sources,
            normalized_index_map=normalized_complete_sources,
        )
        normalized_output = normalized_complete[len(normalized_input) :]
        normalized_output_sources = normalized_complete_sources[len(normalized_input) :]
        if not normalized_output:
            counts["spans_empty_output"] = counts.get("spans_empty_output", 0) + 1
            continue
        if conversation:
            if normalized_input[: len(conversation)] != conversation:
                raise SourceRowError("non_monotone_trace_prefix", session_id)
            conversation = list(normalized_input)
            conversation_sources = normalized_complete_sources[: len(normalized_input)]
        else:
            conversation = list(normalized_input)
            conversation_sources = normalized_complete_sources[: len(normalized_input)]
        conversation.extend(normalized_output)
        conversation_sources.extend(normalized_output_sources)
        usable_spans += 1
    if not usable_spans or not conversation:
        raise SourceRowError("no_usable_spans")
    if not tools:
        raise SourceRowError("missing_tools")
    return {
        "session_id": session_id,
        "source": "agent-llm-traces",
        "split": split,
        "task_id": session_id,
        "template_id": benchmark,
        "messages": conversation,
        "tools": tools,
        "source_metadata": {
            "benchmark": benchmark,
            "usable_spans": usable_spans,
            "normalized_message_sources": conversation_sources,
        },
    }


def _toucan_call(content: Any) -> dict[str, Any]:
    parsed = content
    if isinstance(content, str):
        try:
            parsed = ast.literal_eval(content)
        except (SyntaxError, ValueError) as exc:
            raise SourceRowError("invalid_toucan_tool_call") from exc
    if not isinstance(parsed, Mapping):
        raise SourceRowError("invalid_toucan_tool_call")
    name = parsed.get("name") or parsed.get("tool_name") or parsed.get("function_name")
    if not isinstance(name, str) or not name:
        raise SourceRowError("missing_tool_name")
    result = {
        "type": "function",
        "function": {
            "name": name,
            "arguments": copy.deepcopy(parsed.get("arguments", {})),
        },
    }
    call_id = parsed.get("id") or parsed.get("tool_call_id")
    if isinstance(call_id, str) and call_id:
        result["id"] = call_id
    return result


def adapt_toucan_row(
    row: Mapping[str, Any],
    *,
    split: str = "train",
    audit: MutableMapping[str, int] | None = None,
) -> dict[str, Any]:
    if split != "train":
        raise SourceRowError("non_train_split", split)
    if row.get("subset_name") != "multi-turn":
        raise SourceRowError("non_multiturn_subset")
    uuid = str(row.get("uuid") or "").strip()
    if not uuid:
        raise SourceRowError("missing_session_id")
    raw = _message_sequence(row.get("messages"), field="messages")
    grouped: list[dict[str, Any]] = []
    grouped_sources: list[Any] = []
    cursor = 0
    while cursor < len(raw):
        item = raw[cursor]
        if not isinstance(item, Mapping):
            raise SourceRowError("invalid_message", f"index={cursor}")
        role = item.get("role")
        if role == "assistant":
            combined = {"role": "assistant", "content": item.get("content")}
            calls = list(_raw_tool_calls(item))
            lookahead = cursor + 1
            while lookahead < len(raw):
                following = raw[lookahead]
                if not isinstance(following, Mapping) or following.get("role") != "tool_call":
                    break
                calls.append(_toucan_call(following.get("content")))
                lookahead += 1
            if calls:
                combined["tool_calls"] = calls
            grouped.append(combined)
            grouped_sources.append({"field": "messages", "indices": list(range(cursor, lookahead))})
            cursor = lookahead
            continue
        if role == "tool_call":
            grouped.append(
                {"role": "assistant", "content": None, "tool_calls": [_toucan_call(item.get("content"))]}
            )
        elif role == "tool_response":
            tool_result = {"role": "tool", "content": item.get("content")}
            call_id = item.get("tool_call_id") or item.get("toolCallId")
            if isinstance(call_id, str) and call_id:
                tool_result["tool_call_id"] = call_id
            if isinstance(item.get("name"), str):
                tool_result["name"] = item["name"]
            grouped.append(tool_result)
        elif role in {"system", "developer", "user", "tool"}:
            grouped.append(copy.deepcopy(dict(item)))
        else:
            raise SourceRowError("unsupported_role", str(role))
        grouped_sources.append({"field": "messages", "indices": [cursor]})
        cursor += 1
    counts: MutableMapping[str, int] = audit if audit is not None else Counter()
    normalized_sources: list[Any] = []
    messages = normalize_openai_messages(
        grouped,
        namespace=f"toucan:{uuid}",
        audit=counts,
        source_indices=grouped_sources,
        normalized_index_map=normalized_sources,
        allow_ordered_tool_result_binding=True,
    )
    if not any(message["role"] == "assistant" for message in messages):
        raise SourceRowError("no_assistant_decisions")
    return {
        "session_id": uuid,
        "source": "toucan",
        "split": split,
        "task_id": uuid,
        "template_id": "multi-turn",
        "messages": messages,
        "tools": _parse_tools(row.get("tools")),
        "source_metadata": {
            "subset_name": "multi-turn",
            "missing_result_id_contract": "source_order",
            "normalized_message_sources": normalized_sources,
        },
    }


def adapt_openswe_row(
    row: Mapping[str, Any],
    *,
    config_name: str,
    split: str = "train",
    fallback_tools: Sequence[Mapping[str, Any]] | None = None,
    audit: MutableMapping[str, int] | None = None,
) -> dict[str, Any]:
    if split != "train":
        raise SourceRowError("non_train_split", split)
    if row.get("resolved") != 1:
        raise SourceRowError("unresolved")
    session_id = str(row.get("trajectory_id") or row.get("instance_id") or "").strip()
    if not session_id:
        raise SourceRowError("missing_session_id")
    raw = row.get("trajectory") if row.get("trajectory") is not None else row.get("messages")
    raw_messages = _message_sequence(raw, field="trajectory")
    counts: MutableMapping[str, int] = audit if audit is not None else Counter()
    normalized_sources: list[Any] = []
    messages = normalize_openai_messages(
        raw_messages,
        namespace=f"openswe:{session_id}",
        audit=counts,
        source_indices=[{"field": "trajectory", "index": index} for index in range(len(raw_messages))],
        normalized_index_map=normalized_sources,
    )
    tools_value = row.get("tools")
    tools = _parse_tools(tools_value) if tools_value is not None else [copy.deepcopy(dict(t)) for t in fallback_tools or ()]
    if not tools:
        raise SourceRowError("missing_tools")
    task_id = str(row.get("instance_id") or session_id)
    return {
        "session_id": session_id,
        "source": f"openswe:{config_name}",
        "split": split,
        "task_id": task_id,
        "template_id": config_name,
        "messages": messages,
        "tools": tools,
        "source_metadata": {
            "config_name": config_name,
            "instance_id": task_id,
            "repo": str(row.get("repo") or ""),
            "resolved": 1,
            "normalized_message_sources": normalized_sources,
        },
    }


def _qa_session(
    *,
    family: str,
    row_id: str,
    documents: Sequence[str],
    question: Any,
    answer: Any,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    question_text = str(question or "").strip()
    answer_text = str(answer or "").strip()
    document_texts = [str(document) for document in documents if str(document).strip()]
    if not document_texts:
        raise SourceRowError("missing_documents")
    if not question_text:
        raise SourceRowError("missing_question")
    if not answer_text:
        raise SourceRowError("missing_answer")
    source = f"qa:{family}"
    return {
        "session_id": row_id,
        "source": source,
        "split": "train",
        "task_id": row_id,
        "template_id": f"{family}-document-qa-v1",
        "messages": [
            *({"role": "user", "content": document} for document in document_texts),
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": answer_text},
        ],
        "tools": [],
        "source_metadata": {
            "family": family,
            "normalized_message_sources": [
                *(
                    {"field": "documents", "index": index}
                    for index in range(len(document_texts))
                ),
                {"field": "question"},
                {"field": "answer"},
            ],
            **dict(source_metadata or {}),
        },
    }


def adapt_hotpotqa_row(
    row: Mapping[str, Any],
    *,
    row_index: int = 0,
    audit: MutableMapping[str, int] | None = None,
) -> dict[str, Any]:
    documents = _loads(row.get("documents"), field="documents")
    source_metadata: dict[str, Any] = {}
    if documents is None:
        context = _loads(row.get("context"), field="context")
        documents = []
        if isinstance(context, Mapping):
            titles = context.get("title")
            sentence_groups = context.get("sentences")
            if isinstance(titles, list) and isinstance(sentence_groups, list):
                if len(titles) != len(sentence_groups):
                    raise SourceRowError(
                        "invalid_hotpotqa_context",
                        f"title_count={len(titles)} sentence_group_count={len(sentence_groups)}",
                    )
                document_sources = []
                for index, (title, sentences) in enumerate(zip(titles, sentence_groups)):
                    body = (
                        " ".join(str(sentence) for sentence in sentences)
                        if isinstance(sentences, list)
                        else str(sentences or "")
                    )
                    document = f"{title or ''}\n{body}".strip()
                    if document:
                        documents.append(document)
                        document_sources.append(
                            {
                                "field": "context",
                                "title_index": index,
                                "sentences_index": index,
                            }
                        )
                source_metadata["normalized_message_sources"] = [
                    *document_sources,
                    {"field": "question"},
                    {"field": "answer"},
                ]
    if not isinstance(documents, list):
        documents = [documents]
    row_id = str(row.get("_id") or row.get("id") or row_index)
    return _qa_session(
        family="hotpotqa",
        row_id=row_id,
        documents=[str(document) for document in documents],
        question=row.get("question"),
        answer=row.get("answer"),
        source_metadata=source_metadata,
    )


def adapt_wiki2_row(
    row: Mapping[str, Any],
    *,
    row_index: int = 0,
    audit: MutableMapping[str, int] | None = None,
) -> dict[str, Any]:
    context = _loads(row.get("context"), field="context")
    documents: list[str] = []
    if isinstance(context, list):
        for entry in context:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                title, sentences = entry
                sentences = _loads(sentences, field="context.sentences")
                body = " ".join(str(sentence) for sentence in sentences) if isinstance(sentences, list) else str(sentences)
                documents.append(f"{title}\n{body}")
            elif entry is not None:
                documents.append(str(entry))
    elif isinstance(context, str) and context.strip():
        documents.append(context)
    row_id = str(row.get("_id") or row.get("id") or row_index)
    return _qa_session(
        family="2wiki",
        row_id=row_id,
        documents=documents,
        question=row.get("question"),
        answer=row.get("answer"),
    )


def split_longmagpie_question(text: str) -> tuple[str, str] | None:
    body = (text or "").strip()
    end = len(body)
    start = end
    for match in reversed(list(_LONGMAGPIE_SEGMENT_RE.finditer(body))):
        if match.end() != start:
            break
        start = match.start()
    if start == end:
        return None
    context = body[:start].rstrip()
    question = body[start:end].strip()
    return (context, question) if context and question else None


def adapt_longmagpie_row(
    row: Mapping[str, Any],
    *,
    row_index: int = 0,
    shard: str = "unknown",
    audit: MutableMapping[str, int] | None = None,
) -> dict[str, Any]:
    raw = _message_sequence(row.get("messages"), field="messages")
    user = next((message for message in raw if message.get("role") == "user"), None)
    assistant = next((message for message in raw if message.get("role") == "assistant"), None)
    if user is None or assistant is None:
        raise SourceRowError("missing_qa_messages")
    split = split_longmagpie_question(str(user.get("content") or ""))
    if split is None:
        raise SourceRowError("longmagpie_no_question_suffix")
    context, question = split
    row_id = f"{shard}:{row_index}"
    return _qa_session(
        family="longmagpie",
        row_id=row_id,
        documents=[context],
        question=question,
        answer=assistant.get("content"),
        source_metadata={"shard": shard, "row_in_shard": row_index},
    )


def _find_files(path: str | Path, *, suffix: str, subdirs: Sequence[str] = ("data",)) -> list[Path]:
    root = Path(path)
    if root.is_file() and root.suffix == suffix:
        return [root]
    for candidate in [*(root / subdir for subdir in subdirs), root]:
        if candidate.is_dir():
            files = sorted(candidate.glob(f"*{suffix}"))
            if not files:
                files = sorted(candidate.rglob(f"*{suffix}"))
            if files:
                return files
    return []


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SourceRowError("invalid_json", f"{path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise SourceRowError("invalid_row", f"{path}:{line_number}")
            yield row


def _iter_parquet(path: Path, columns: Sequence[str] | None = None) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema_arrow.names)
    selected = [column for column in columns or () if column in available] or None
    try:
        table = pq.read_table(path, columns=selected)
    except Exception:
        table = pq.ParquetFile(path).read(columns=selected)
    for row in table.to_pylist():
        if isinstance(row, dict):
            yield row


def _iter_rows(
    path: str | Path,
    *,
    subdirs: Sequence[str] = ("data",),
    columns: Sequence[str] | None = None,
) -> Iterator[tuple[Path, int, dict[str, Any]]]:
    files = _find_files(path, suffix=".parquet", subdirs=subdirs)
    if not files:
        files = _find_files(path, suffix=".jsonl", subdirs=subdirs)
    if not files:
        raise FileNotFoundError(f"No parquet/jsonl files found under {path}")
    for file in files:
        iterator = _iter_parquet(file, columns) if file.suffix == ".parquet" else _iter_jsonl(file)
        for row_index, row in enumerate(iterator):
            yield file, row_index, row


def load_train_session_ids(path: str | Path, *, split_name: str) -> frozenset[str]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = manifest if "train_session_ids" in manifest else manifest.get(split_name)
    if not isinstance(selected, Mapping):
        raise ValueError(f"Split manifest has no mapping {split_name!r}")
    train_ids = {str(value) for value in selected.get("train_session_ids", [])}
    eval_ids = {str(value) for value in selected.get("eval_session_ids", [])}
    overlap = train_ids & eval_ids
    if overlap:
        raise ValueError(f"Split manifest train/eval overlap: {sorted(overlap)[:5]!r}")
    return frozenset(train_ids)


def _fallback_tool_tables(path: Path) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    for prefix, filename in (
        ("openhands", "openhands_tools.json"),
        ("sweagent", "sweagent_tools.json"),
        ("minisweagent", "mini_sweagent_tools.json"),
    ):
        for candidate in (path / filename, path.parent / filename):
            if candidate.is_file():
                tables[prefix] = _parse_tools(candidate.read_text(encoding="utf-8"))
                break
    return tables


def _fallback_for_config(
    config_name: str, tables: Mapping[str, list[dict[str, Any]]]
) -> list[dict[str, Any]] | None:
    lowered = config_name.casefold()
    for prefix in ("minisweagent", "openhands", "sweagent"):
        if prefix in lowered and prefix in tables:
            return tables[prefix]
    return None


def load_g_sources(
    *,
    traces_path: str | Path | None = None,
    traces_split_manifest: str | Path | None = None,
    traces_split_name: str = "taskproxy_disjoint",
    toucan_path: str | Path | None = None,
    openswe_path: str | Path | None = None,
    hotpotqa_path: str | Path | None = None,
    wiki2_path: str | Path | None = None,
    longmagpie_path: str | Path | None = None,
    max_rows_per_source: int | None = None,
    file_order_seed: int = 42,
) -> LoadedSources:
    """Load enabled G families in deterministic order with audit counters."""
    if traces_path is not None and traces_split_manifest is None:
        raise ValueError("traces_split_manifest is required for train-only traces loading")
    if max_rows_per_source is not None and max_rows_per_source < 1:
        raise ValueError("max_rows_per_source must be >= 1")
    audit: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []

    def accept(family: str, raw: Mapping[str, Any], converter, **kwargs: Any) -> None:
        audit[f"{family}.rows_seen"] += 1
        row_audit: Counter[str] = Counter()
        try:
            converted = converter(raw, audit=row_audit, **kwargs)
        except SourceRowError as exc:
            for key, value in row_audit.items():
                audit[f"{family}.{key}"] += value
            audit[f"{family}.skipped.{exc.reason}"] += 1
            return
        for key, value in row_audit.items():
            audit[f"{family}.{key}"] += value
        rows.append(converted)
        audit[f"{family}.sessions_emitted"] += 1

    if traces_path is not None:
        train_ids = load_train_session_ids(
            traces_split_manifest, split_name=traces_split_name
        )
        seen = 0
        columns = (
            "benchmark",
            "subset",
            "dataset",
            "task",
            "session_id",
            "trace_id",
            "id",
            "spans",
        )
        for _, _, raw in _iter_rows(traces_path, columns=columns):
            if max_rows_per_source is not None and seen >= max_rows_per_source:
                audit["agent-llm-traces.truncated_at_max_rows"] += 1
                break
            seen += 1
            session_id = str(raw.get("session_id") or raw.get("trace_id") or raw.get("id") or "")
            if session_id not in train_ids:
                audit["agent-llm-traces.rows_seen"] += 1
                audit["agent-llm-traces.skipped.not_train_manifest"] += 1
                continue
            accept("agent-llm-traces", raw, adapt_agent_trace_row)

    if toucan_path is not None:
        eligible_seen = 0
        for _, _, raw in _iter_rows(
            toucan_path,
            subdirs=("SFT",),
            columns=("uuid", "subset_name", "tools", "messages"),
        ):
            audit["toucan.rows_scanned"] += 1
            if raw.get("subset_name") != "multi-turn":
                accept("toucan", raw, adapt_toucan_row)
                continue
            if max_rows_per_source is not None and eligible_seen >= max_rows_per_source:
                audit["toucan.truncated_at_max_rows"] += 1
                break
            eligible_seen += 1
            accept("toucan", raw, adapt_toucan_row)

    if openswe_path is not None:
        root = Path(openswe_path)
        files = _find_files(root, suffix=".parquet", subdirs=("data",))
        if not files:
            files = _find_files(root, suffix=".jsonl", subdirs=("data",))
        if not files:
            raise FileNotFoundError(f"No parquet/jsonl files found under {openswe_path}")
        random.Random(file_order_seed).shuffle(files)
        fallback_tables = _fallback_tool_tables(root)
        seen = 0
        for file in files:
            config_name = file.parent.name if file.parent.name != "data" else file.stem
            iterator = _iter_parquet(file) if file.suffix == ".parquet" else _iter_jsonl(file)
            for raw in iterator:
                if max_rows_per_source is not None and seen >= max_rows_per_source:
                    audit["openswe.truncated_at_max_rows"] += 1
                    break
                seen += 1
                accept(
                    "openswe",
                    raw,
                    adapt_openswe_row,
                    config_name=config_name,
                    fallback_tools=_fallback_for_config(config_name, fallback_tables),
                )
            if max_rows_per_source is not None and seen >= max_rows_per_source:
                break

    for family, path, converter, subdirs in (
        ("qa:hotpotqa", hotpotqa_path, adapt_hotpotqa_row, ("data",)),
        ("qa:2wiki", wiki2_path, adapt_wiki2_row, ()),
    ):
        if path is None:
            continue
        seen = 0
        for _, row_index, raw in _iter_rows(path, subdirs=subdirs):
            if max_rows_per_source is not None and seen >= max_rows_per_source:
                audit[f"{family}.truncated_at_max_rows"] += 1
                break
            seen += 1
            accept(family, raw, converter, row_index=row_index)

    if longmagpie_path is not None:
        seen = 0
        for file, row_index, raw in _iter_rows(longmagpie_path):
            if max_rows_per_source is not None and seen >= max_rows_per_source:
                audit["qa:longmagpie.truncated_at_max_rows"] += 1
                break
            seen += 1
            accept(
                "qa:longmagpie",
                raw,
                adapt_longmagpie_row,
                row_index=row_index,
                shard=file.stem,
            )
    audit["sessions_total"] = len(rows)
    return LoadedSources(tuple(rows), dict(sorted(audit.items())))


__all__ = [
    "DEFAULT_QA_TARGET_FRACTION",
    "LoadedSources",
    "SourceRowError",
    "adapt_agent_trace_row",
    "adapt_hotpotqa_row",
    "adapt_longmagpie_row",
    "adapt_openswe_row",
    "adapt_toucan_row",
    "adapt_wiki2_row",
    "load_g_sources",
    "load_train_session_ids",
    "normalize_openai_messages",
    "split_longmagpie_question",
]
