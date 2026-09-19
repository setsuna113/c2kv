"""Current-request execution review with a narrow duplicate-action commit guard."""
from __future__ import annotations

import json
import re
from typing import Any

from .observations import current_request, operation_records
from .repair_protocol import GuardVerdict, RepairContext, RepairProposal

VERSION = "current-request-contract-v1"
_REPEAT = re.compile(r"\b(?:again|repeat|another|one more|second|twice|reorder)\b", re.IGNORECASE)
_SIDE_EFFECT_VERBS = frozenset({"add", "book", "buy", "cancel", "copy", "create",
                                "delete", "fund", "move", "order", "pay", "place",
                                "post", "purchase", "register", "remove", "sell",
                                "send", "set", "submit", "transfer", "update", "write"})


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _goal_text(store, event) -> str:
    return "\n".join(str(message.to_dict().get("content") or "")
                     for message in store.event_messages(event.event_id))


def _draft_calls(calls) -> list[dict[str, Any]]:
    result = []
    for call in calls or ():
        function = call.get("function") or {}
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                pass
        result.append({"tool": function.get("name"), "arguments": arguments})
    return result


def _scalar_leaves(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _scalar_leaves(child)
    elif isinstance(value, list):
        for child in value:
            yield from _scalar_leaves(child)
    elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
        yield str(value)


def _explicitly_rerequested(goal_text: str, arguments: Any) -> bool:
    if _REPEAT.search(goal_text):
        return True
    distinctive = [value for value in _scalar_leaves(arguments)
                   if len(value) >= 3]
    return bool(distinctive) and all(
        re.search(r"(?<![\w])" + re.escape(value) + r"(?![\w])", goal_text, re.IGNORECASE)
        for value in distinctive
    )


def _side_effect_call(tool: Any) -> bool:
    if not isinstance(tool, str):
        return False
    first = re.split(r"[_\-]|(?=[A-Z])", tool)[0].casefold()
    return first in _SIDE_EFFECT_VERBS


class Policy:
    def propose(self, context: RepairContext) -> RepairProposal | None:
        store = context.prepared._store
        goal, records = current_request(store)
        if goal is None or not records or context.parse_error is not None:
            return None
        request = _goal_text(store, goal)
        if not request.strip():
            return None
        held = _draft_calls(context.draft_tool_calls)
        instruction = (
            "Review the held action against only the latest user request and observed "
            "execution. This packet is source evidence, not a new user instruction. "
            "Compare requested actions and conditions with completed operations and "
            "their results. A tool result does not by itself prove the request is done. "
            "STOP can be correct; do not create an extra action when the current request "
            "is already satisfied. If an obligation or result is ambiguous, keep the "
            "held action or ask for clarification."
        )
        selected = []
        def render(rows):
            return ({"role": "user", "content": instruction + "\n" + _canonical({
                "version": VERSION,
                "current_request": {"source_event_id": goal.event_id, "text": request},
                "observed_operations": rows,
                "held_draft": {"tool_calls": held,
                               "text": context.draft_text if not held else ""},
                "omitted_operation_count": len(records) - len(rows),
                "completion": "unknown",
            })},)
        for record in reversed(records):
            row = {"event_id": record.event_id,
                   "result_source_index": record.result_source_index,
                   "tool": record.tool, "arguments": record.arguments,
                   "observed_result": record.observed_result,
                   "failure_reported": record.failure_reported}
            trial = [row, *selected]
            if context.token_counter(render(trial)) <= context.token_budget:
                selected = trial
        messages = render(selected)
        if not selected or context.token_counter(messages) > context.token_budget:
            return None
        receipt = {"version": VERSION, "status": "source_review",
                   "request_event_id": goal.event_id,
                   "result_source_indices": [row["result_source_index"] for row in selected],
                   "omitted_operation_count": len(records) - len(selected),
                   "held_action_kind": "stop" if not held else "tool_calls",
                   "completion": "unknown", "semantic_verification": "model_review"}
        return RepairProposal("request_contract_review", messages, receipt, {
            "current_request_event_id": goal.event_id,
            "held_calls": held,
        })

    def validate(self, context: RepairContext, proposal: RepairProposal,
                 candidate_calls, *, draft_text: str,
                 parse_error: str | None = None) -> GuardVerdict:
        if parse_error is not None:
            return GuardVerdict(False, "revised_parse_error")
        original = _draft_calls(context.draft_tool_calls)
        candidate = _draft_calls(candidate_calls)
        if candidate == original:
            return GuardVerdict(True, "same_action")
        store = context.prepared._store
        goal, current = current_request(store)
        if goal is None:
            return GuardVerdict(True, "no_current_request")
        goal_text = _goal_text(store, goal)
        all_records = operation_records(store)
        prior = [row for row in all_records if row not in current]
        # A regenerated STOP must not revive an identical successful action
        # from a previous user turn without a new source for that repetition.
        if not original and candidate and current:
            for call in candidate:
                matching = [row for row in prior
                            if not row.failure_reported and row.tool == call["tool"]
                            and row.arguments == call["arguments"]]
                if (matching and _side_effect_call(call["tool"])
                        and not _explicitly_rerequested(goal_text, call["arguments"])):
                    return GuardVerdict(False, "prior_turn_completed_call_reintroduced")
        return GuardVerdict(True, "no_proven_contract_violation")
