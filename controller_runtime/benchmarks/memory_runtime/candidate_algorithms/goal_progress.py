"""Bounded Goal review for an observable same-epoch pure-read loop."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .observations import ObservedOperation, call_signature, current_request, parse_json_or_text
from .progress import review_messages as original_goal_review_messages
from .repair_protocol import GuardVerdict, RepairContext, RepairProposal


VERSION = "goal-progress-v1"

# Explicitly inspected read-only or pure-computation tools from the offline
# trigger screen. Unknown tools are possible mutations and reset the epoch.
PURE_READ_TOOLS = frozenset({
    "ls", "cat", "find", "pwd", "grep", "tail", "wc", "du", "diff",
    "get_flight_cost", "get_stock_info", "get_order_details", "get_booking_history",
    "get_watchlist", "get_zipcode_based_on_city", "get_nearest_airport_by_city",
    "retrieve_invoice", "check_tire_pressure", "estimate_distance",
    "get_user_tickets", "get_symbol_by_name", "get_account_info",
    "get_order_history", "get_user_id", "view_messages_sent",
    "get_transaction_history", "posting_get_login_status",
    "get_available_stocks", "get_all_credit_cards", "compute_exchange_rate",
    "estimate_drive_feasibility_by_mileage", "find_nearest_tire_shop",
    "ticket_get_login_status", "travel_get_login_status", "list_all_airports",
    "get_ticket", "message_get_login_status", "get_credit_card_balance",
    "get_budget_fiscal_year", "search_tweets", "list_users", "logarithm",
    "trading_get_login_status", "gallon_to_liter",
    "liter_to_gallon", "get_user_stats", "get_current_fuel_level",
    "get_current_speed", "get_user_tweets", "mean",
    "add", "divide", "standard_deviation", "displayCarStatus",
})
_RECHECK = re.compile(
    r"\b(?:recheck|re-check|poll|refresh|monitor|check again|"
    r"check repeatedly|keep checking)\b", re.IGNORECASE)
_ACTION_WORDS = frozenset({
    "add", "book", "buy", "cancel", "copy", "create", "delete", "edit",
    "fill", "lock", "move", "notify", "pay", "place", "post",
    "press", "purchase", "register", "remove", "resolve", "send", "set",
    "start", "submit", "transfer", "update", "write",
})
_GENERIC_WORDS = frozenset({
    "a", "an", "and", "by", "details", "for", "from", "get", "info",
    "information", "of", "on", "the", "to", "tool", "with",
})

_INSTRUCTION = (
    "The held action is one exact pure read already observed in this current "
    "user turn. Since its latest occurrence, the visible record contains no "
    "new call-and-result observation or possible state mutation. Review the "
    "full current request and all admitted complete operation records above. "
    "Do not repeat a read merely to obtain the same evidence. Choose a "
    "different supported action or ask for clarification when appropriate; "
    "the task's completion remains unknown."
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _call_key(call: Any) -> tuple[str, str] | None:
    if not isinstance(call, dict):
        return None
    function = call.get("function")
    if not isinstance(function, dict):
        return None
    name = function.get("name")
    arguments = parse_json_or_text(function.get("arguments"))
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    try:
        return name, call_signature(name, arguments)
    except (TypeError, ValueError, OverflowError):
        return None


def _words(value: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return {word.casefold() for word in re.findall(r"[A-Za-z][A-Za-z0-9]*", spaced)}


def _current_request_text(store, user) -> str:
    return "\n".join(str(store.messages[index].to_dict().get("content") or "")
                     for index in user.source_indices)


def _tool_description(context: RepairContext, name: str) -> tuple[bool, str]:
    for declaration in getattr(context.prepared, "_tools", ()) or ():
        if not isinstance(declaration, dict):
            continue
        function = declaration.get("function")
        if isinstance(function, dict) and function.get("name") == name:
            return True, str(function.get("description") or "")
    return False, ""


def _supported_action(context: RepairContext, name: str) -> bool:
    user, _ = current_request(context.prepared._store)
    if user is None:
        return False
    request_words = _words(_current_request_text(context.prepared._store, user))
    declared, description = _tool_description(context, name)
    declared_tools = getattr(context.prepared, "_tools", ()) or ()
    if declared_tools and not declared:
        return False
    role_words = _words(name) | _words(description)
    subject_words = role_words - _ACTION_WORDS - _GENERIC_WORDS
    if not (subject_words & request_words):
        return False
    if name in PURE_READ_TOOLS:
        return True
    # A possible mutation needs both a declared tool role and an action word
    # requested by the user. Different syntax alone is not progress evidence.
    return declared and bool(role_words & request_words & _ACTION_WORDS)


def _source(row: ObservedOperation) -> dict[str, Any]:
    return {"event_id": row.event_id,
            "call_source_index": row.call_source_index,
            "result_source_index": row.result_source_index}


@dataclass(frozen=True)
class _Epoch:
    index: int
    prior: ObservedOperation
    read_signatures: frozenset[str]
    state_key: str


def _progress_epoch(store, user, records, held_signature: str) -> _Epoch | None:
    # An incomplete event can still contain an unreturned mutation. No epoch
    # conclusion is safe until its result is observable.
    boundary = max(user.source_indices)
    if any(event.kind == "tool_event" and not event.complete
           and max(event.source_indices) > boundary for event in store.events):
        return None
    event_counts = Counter(row.event_id for row in records)
    seen: set[tuple[str, str]] = set()
    epoch = 0
    latest_read: dict[str, tuple[int, ObservedOperation]] = {}
    for row in records:
        try:
            observation = (row.call_signature_id, _canonical(row.observed_result))
        except (TypeError, ValueError, OverflowError):
            return None
        if observation not in seen:
            seen.add(observation)
            epoch += 1
        if event_counts[row.event_id] != 1 or row.tool not in PURE_READ_TOOLS:
            # A parallel event is conservatively treated as a possible state
            # change even if it contains a read with a familiar result.
            epoch += 1
            continue
        latest_read[row.call_signature_id] = (epoch, row)
    prior = latest_read.get(held_signature)
    if prior is None or prior[0] != epoch:
        return None
    state_key = hashlib.sha256(_canonical({
        "request_event_id": user.event_id,
        "epoch": epoch,
        "observations": sorted(seen),
    }).encode("utf-8")).hexdigest()
    read_signatures = frozenset(signature for signature, (at, _) in latest_read.items()
                                if at == epoch)
    return _Epoch(epoch, prior[1], read_signatures, state_key)


def _original_row(row: ObservedOperation) -> dict[str, Any]:
    return {"event_id": row.event_id,
            "result_source_index": row.result_source_index,
            "tool": row.tool, "arguments": row.arguments,
            "observed_result": row.observed_result,
            "failure_reported": row.failure_reported}


class Policy:
    def propose(self, context: RepairContext) -> RepairProposal | None:
        # The composition controller invokes this only after original Goal
        # abstains. This policy never turns an original STOP into a call review.
        if context.parse_error is not None or len(context.draft_tool_calls) != 1:
            return None
        held = _call_key(context.draft_tool_calls[0])
        if held is None or held[0] not in PURE_READ_TOOLS:
            return None
        store = context.prepared._store
        user, records = current_request(store)
        if user is None or not records:
            return None
        if _RECHECK.search(_current_request_text(store, user)):
            return None
        progress = _progress_epoch(store, user, records, held[1])
        if progress is None:
            return None
        evidence = {"version": VERSION,
                    "held_read": {"tool": held[0],
                                  "arguments": parse_json_or_text(
                                      context.draft_tool_calls[0]["function"]["arguments"])},
                    "prior_read_source": _source(progress.prior),
                    "goal_completion": "unknown"}
        supplement = {"role": "user", "content": _INSTRUCTION + "\n" + _canonical(evidence)}
        remaining_budget = context.token_budget - context.token_counter((supplement,))
        if remaining_budget <= 0:
            return None
        original_records = [_original_row(row) for row in records]
        messages, review_receipt = original_goal_review_messages(
            store, user, original_records, token_counter=context.token_counter,
            token_budget=remaining_budget)
        if review_receipt.get("status") != "prepared":
            return None
        packet = (*messages, supplement)
        if context.token_counter(packet) > context.token_budget:
            return None
        return RepairProposal(
            reason="goal_progress_read_review",
            messages=packet,
            receipt={"version": VERSION, "status": "prepared",
                     "current_request_event_id": user.event_id,
                     "prior_read_source": _source(progress.prior),
                     "observed_operation_count": len(records),
                     "admitted_operation_count": (
                         len(records) - review_receipt["omitted_operation_count"]),
                     "omitted_operation_count": review_receipt["omitted_operation_count"],
                     "admitted_source_event_ids": review_receipt["source_event_ids"],
                     "state_key": progress.state_key,
                     "goal_completion": "unknown",
                     "token_budget": context.token_budget,
                     "packet_tokens": context.token_counter(packet)},
            guard={"state_key": progress.state_key,
                   "read_signatures": sorted(progress.read_signatures)},
        )

    def validate(self, context: RepairContext, proposal: RepairProposal,
                 candidate_calls, *, draft_text: str,
                 parse_error: str | None = None) -> GuardVerdict:
        if parse_error is not None:
            return GuardVerdict(False, "candidate_parse_error", "original")
        if not candidate_calls:
            return GuardVerdict(False, "candidate_stop", "original")
        repeat_signatures = set(proposal.guard.get("read_signatures", ()))
        for call in candidate_calls:
            parsed = _call_key(call)
            if parsed is None:
                return GuardVerdict(False, "candidate_malformed_call", "original")
            if parsed[0] in PURE_READ_TOOLS and parsed[1] in repeat_signatures:
                return GuardVerdict(False, "candidate_repeats_read_without_evidence",
                                    "original")
            if not _supported_action(context, parsed[0]):
                return GuardVerdict(False, "candidate_unsupported_action", "original")
        return GuardVerdict(True, "candidate_changes_action")


__all__ = ["Policy", "PURE_READ_TOOLS"]
