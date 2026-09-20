"""Source-linked pending-obligation supplement to the original Goal STOP review."""

from __future__ import annotations

import json
import re
import copy
from typing import Any

from .observations import ObservedOperation, current_request, parse_json_or_text
from .repair_protocol import RepairContext, RepairProposal


VERSION = "goal-pending-v1"
_EXECUTION_VERBS = frozenset({
    "add", "book", "buy", "cancel", "create", "delete", "edit", "fund",
    "move", "notify", "order", "pay", "place", "post", "purchase",
    "register", "remove", "resolve", "sell", "send", "set", "submit",
    "transfer", "update", "write", "lock", "fill", "start", "press",
    "cp", "mkdir", "mv", "rm", "touch", "rmdir", "echo",
})
_LOOKUP_VERBS = frozenset({
    "cat", "check", "find", "get", "grep", "inspect", "list", "lookup",
    "ls", "read", "retrieve", "search", "view",
})
_INSTRUCTION = (
    "Apply the original Goal review above clause by clause. Indexed statuses "
    "describe receipts only: lookup returns and generic non-error results do "
    "not certify requested actions. A deferred consumer needs its actual "
    "producer receipt. Decide in this generation whether to continue, finish, "
    "or clarify."
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _kind(tool: str) -> str:
    first = re.split(r"[_\-]|(?=[A-Z])", tool)[0].casefold()
    if first in _EXECUTION_VERBS:
        return "execution"
    if first in _LOOKUP_VERBS:
        return "lookup"
    return "unknown"


def _status(tool: str, result: Any, failure: bool) -> str:
    kind = _kind(tool)
    if failure:
        return ("execution_reported_failure" if kind == "execution" else
                "lookup_reported_failure" if kind == "lookup" else
                "tool_reported_failure")
    if kind == "lookup":
        return "lookup_returned"
    if kind == "execution":
        if isinstance(result, dict) and result.get("success") is True:
            return "execution_reported_success"
        return "execution_completion_unknown"
    return "tool_completion_unknown"


def _row(record: ObservedOperation) -> dict[str, Any]:
    return {
        "result_source_index": record.result_source_index,
        "status": _status(record.tool, record.observed_result,
                          record.failure_reported),
    }


def _original_admitted(original_messages) -> set[int]:
    indices = set()
    for message in original_messages:
        content = message.get("content")
        if not isinstance(content, str) or "\n" not in content:
            continue
        try:
            payload = json.loads(content.split("\n", 1)[1])
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        for row in payload.get("observed_operations", ()):
            if isinstance(row, dict) and isinstance(row.get("result_source_index"), int):
                indices.add(row["result_source_index"])
    return indices


def _consumer(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    function = value.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return {"tool": function["name"],
                "arguments": parse_json_or_text(function.get("arguments"))}
    if isinstance(value.get("tool"), str):
        return {"tool": value["tool"],
                "arguments": parse_json_or_text(value.get("arguments"))}
    return None


def _deferred(value: dict[str, Any], admitted: set[int]) -> dict[str, Any]:
    # Transport IDs stay in the controller receipt, outside the model packet.
    item = {"original_request_event_id": value.get("request_event_id"),
            "original_consumer": _consumer(value.get("call", value.get("consumer"))),
            "unbound_paths": value.get("unbound_paths", ()),
            "status": "requires_model_review"}
    producer = value.get("producer")
    if isinstance(producer, dict):
        tool = producer.get("tool")
        result = parse_json_or_text(producer.get("observed_result"))
        source = {key: producer[key] for key in (
            "event_id", "call_source_index", "result_source_index")
                  if key in producer}
        source["status"] = (_status(tool, result, bool(producer.get("failure_reported")))
                            if isinstance(tool, str) else "tool_completion_unknown")
        if source.get("result_source_index") not in admitted:
            # The original Goal packet cannot provide this prior receipt.
            source["observed_result"] = result
        item["producer_source"] = source
    return item


def review_messages(
    context: RepairContext,
    original_messages: tuple[dict[str, Any], ...],
    *,
    deferred_consumers=(),
) -> RepairProposal | None:
    """Append a bounded status packet without rewriting Goal's admitted review.

    The caller invokes this only for the original Goal STOP review. The local
    guard also excludes malformed drafts and ordinary call reviews.
    """
    if context.parse_error is not None or context.draft_tool_calls or not original_messages:
        return None
    store = context.prepared._store
    user, records = current_request(store)
    deferred_inputs = tuple(deferred_consumers)
    if user is None or (not records and not deferred_inputs):
        return None
    original = tuple(original_messages)
    if context.token_counter(original) >= context.token_budget:
        return None
    admitted = _original_admitted(original)
    deferred = [_deferred(row, admitted) for row in deferred_inputs]
    status_rows = [_row(row) for row in records
                   if row.result_source_index in admitted]

    def render(selected):
        supplement = {"role": "user", "content": _INSTRUCTION + "\n" + _canonical({
            "version": VERSION,
            "request_event_id": user.event_id,
            "obligations": [{"request_event_id": user.event_id,
                             "status": "requires_model_review"}],
            "operation_statuses": selected,
            "omitted_status_row_count": len(status_rows) - len(selected),
            "deferred_consumers": deferred,
            "goal_completion": "unknown",
        })}
        return (*original, supplement)

    selected: list[dict[str, Any]] = []
    for row in reversed(status_rows):
        trial = [row, *selected]
        if context.token_counter(render(trial)) <= context.token_budget:
            selected = trial
    messages = render(selected)
    if (not selected and not deferred) or context.token_counter(messages) > context.token_budget:
        return None
    return RepairProposal(
        reason="goal_pending_completion_review",
        messages=messages,
        receipt={
            "version": VERSION,
            "status": "prepared",
            "original_goal_review_preserved": True,
            "current_request_event_id": user.event_id,
            "observed_sources": [row["result_source_index"] for row in selected],
            "observed_operation_count": len(records),
            "omitted_status_row_count": len(status_rows) - len(selected),
            "deferred_consumer_count": len(deferred),
            "deferred_consumers": copy.deepcopy(list(deferred_inputs)),
            "goal_completion": "unknown",
            "semantic_verification": "same_generation_model_review",
            "token_budget": context.token_budget,
            "packet_tokens": context.token_counter(messages),
        },
    )


__all__ = ["review_messages"]
