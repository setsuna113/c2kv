"""Deterministic, source-backed ledger for a deliberately small action grammar.

The ledger never guesses a tool's semantics from its name.  Version one only
recognizes the rules declared below, requires a matching JSON schema, and uses
complete current-request observations as execution evidence.  Unsupported or
ambiguous language remains ``unknown``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .observations import ObservedOperation, current_request, operation_records
from .repair_protocol import GuardVerdict, RepairContext, RepairProposal


ACTION_LEDGER_VERSION = "static-action-ledger-v1"
ACTION_RULES_VERSION = "action-ledger-rules-v1"
ACTION_STATES = ("ready_unexecuted", "completed", "blocked", "unknown")


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    )


def _unique_pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key")
        output[key] = value
    return output


def _decode_arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value, object_pairs_hook=_unique_pairs)
        except (TypeError, ValueError):
            return None
    if not isinstance(value, dict):
        return None
    try:
        _json(value)
    except (TypeError, ValueError):
        return None
    return value


def _semantic_call(call: Any) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
        return None
    function = call["function"]
    name = function.get("name")
    arguments = _decode_arguments(function.get("arguments", {}))
    if not isinstance(name, str) or not name or arguments is None:
        return None
    return name, arguments


@dataclass(frozen=True)
class UserSourceSpan:
    """One complete literal span in an observable user source message."""

    field: str
    source_index: int
    start: int
    end: int
    text: str

    def to_receipt(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "source_index": self.source_index,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass(frozen=True)
class ActionObligation:
    """Immutable semantic action plus its source and execution classification."""

    obligation_id: str
    request_event_id: str
    request_source_indices: tuple[int, ...]
    tool: str
    arguments_json: str
    state: str
    reason: str
    source_spans: tuple[UserSourceSpan, ...]
    evidence_json: str

    @property
    def arguments(self) -> dict[str, Any]:
        return json.loads(self.arguments_json)

    @property
    def evidence(self) -> tuple[dict[str, Any], ...]:
        return tuple(json.loads(self.evidence_json))

    def semantic_call(self) -> dict[str, Any]:
        return {"tool": self.tool, "arguments": self.arguments}

    def to_receipt(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "request_event_id": self.request_event_id,
            "request_source_indices": list(self.request_source_indices),
            "tool": self.tool,
            "arguments": self.arguments,
            "state": self.state,
            "reason": self.reason,
            "source_spans": [span.to_receipt() for span in self.source_spans],
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class LedgerAssessment:
    """A frozen snapshot; mapping views are decoded afresh on every access."""

    obligations: tuple[ActionObligation, ...]
    request_event_id: str | None
    prefix_message_count: int
    _receipt_json: str

    @property
    def ready_obligations(self) -> tuple[ActionObligation, ...]:
        return tuple(row for row in self.obligations
                     if row.state == "ready_unexecuted")

    @property
    def ready(self) -> tuple[ActionObligation, ...]:
        return self.ready_obligations

    @property
    def receipt(self) -> dict[str, Any]:
        return json.loads(self._receipt_json)


@dataclass(frozen=True)
class _RequestSource:
    event_id: str
    source_indices: tuple[int, ...]
    text: str
    # global start, global end, source index, local content
    segments: tuple[tuple[int, int, int, str], ...]

    def span(self, field: str, start: int, end: int) -> UserSourceSpan | None:
        for global_start, global_end, source_index, content in self.segments:
            if global_start <= start <= end <= global_end:
                local_start, local_end = start - global_start, end - global_start
                return UserSourceSpan(
                    field, source_index, local_start, local_end,
                    content[local_start:local_end],
                )
        return None


@dataclass(frozen=True)
class _DeclaredSchema:
    tool: str
    required: tuple[str, ...]
    properties: Mapping[str, Mapping[str, Any]]

    def evidence(self) -> dict[str, Any]:
        return {
            "kind": "declared_tool_schema",
            "tool": self.tool,
            "required": list(self.required),
            "property_types": {
                name: declaration.get("type")
                for name, declaration in sorted(self.properties.items())
                if isinstance(declaration, dict)
            },
        }


@dataclass(frozen=True)
class _Attempt:
    tool: str
    arguments_json: str | None
    status: str
    event_id: str
    call_source_index: int
    result_source_index: int | None = None
    result: Any = None
    failure_reported: bool = False

    @property
    def arguments(self) -> dict[str, Any] | None:
        return None if self.arguments_json is None else json.loads(self.arguments_json)


# This registry is the complete v1 semantics surface.  Field layouts are
# explicit; a tool-name prefix never grants authorization.
TOOL_SEMANTICS: Mapping[str, Mapping[str, Any]] = {
    "book_flight": {
        "layouts": (
            ("travel_from", "travel_to"),
            ("origin", "destination"),
        ),
        "success_field": "booking_status",
        "success_statuses": ("booked", "confirmed", "completed", "success"),
        "invalidated_by": ("cancel_booking", "cancel_flight"),
    },
    "send_message": {
        "layouts": (
            ("receiver_id", "message"),
            ("recipient", "message"),
            ("recipient", "text"),
            ("phone_number", "content"),
        ),
        "success_field": "sent_status",
        "success_statuses": ("sent", "delivered", "completed", "success"),
        "invalidated_by": (),
    },
    "fund_account": {
        "layouts": (("amount",), ("amount", "currency")),
        "fixed_currency": "USD",
        "success_field": None,
        "success_statuses": ("funded", "completed", "success"),
        "invalidated_by": (),
    },
}

_QUOTED = re.compile(
    r"(?<!\w)(?:'[^'\\]*'(?!\w)|\"[^\"\\]*\"(?!\w))"
)
_ROUTE = re.compile(
    r"\bfrom\s+(?P<origin>'[^'\\]*'|\"[^\"\\]*\"|[A-Za-z0-9][\w.-]*)"
    r"\s+to\s+(?P<destination>'[^'\\]*'|\"[^\"\\]*\"|[A-Za-z0-9][\w.-]*)\b",
    re.IGNORECASE,
)
_BOOK_ACTION = re.compile(
    r"\b(?:book|reserve)\s+(?:(?:a|the|another)\s+)?flight\b", re.I
)
_FARE_QUERY = re.compile(
    r"\b(?:check|find|get|look\s+up)\s+(?:the\s+)?(?:fare|flight\s+cost)\b",
    re.I,
)
_CONDITION = re.compile(
    r"\bif\s+(?:the\s+)?(?:fare|flight\s+cost|it)\s+(?:is\s+)?"
    r"(?P<operator>under|below|less\s+than|at\s+most|no\s+more\s+than)\s+"
    r"(?P<amount>\d[\d,]*(?:\.\d+)?)\s+(?P<currency>[A-Z]{3})\b",
    re.I,
)
_SEND_PATTERNS = (
    re.compile(
        r"\bsend\s+(?:a\s+)?message\s+to\s+"
        r"(?P<label>recipient|receiver)(?:\s+(?:id|named))?\s+"
        r"(?P<target>'[^'\\]*'|\"[^\"\\]*\"|[A-Za-z0-9_@.+-]+)\s+"
        r"(?:saying|with\s+(?:the\s+)?(?:message|content))\s+"
        r"(?P<body>'[^'\\]*'|\"[^\"\\]*\")",
        re.I,
    ),
    re.compile(
        r"\bsend\s+(?P<label>recipient|receiver)(?:\s+(?:id|named))?\s+"
        r"(?P<target>'[^'\\]*'|\"[^\"\\]*\"|[A-Za-z0-9_@.+-]+)\s+"
        r"(?:the\s+)?message\s+(?P<body>'[^'\\]*'|\"[^\"\\]*\")",
        re.I,
    ),
    re.compile(
        r"\bsend\s+(?:a\s+)?message\s+to\s+phone\s+number\s+"
        r"(?P<target>'[^'\\]*'|\"[^\"\\]*\"|\+?[0-9][0-9 .()-]*)\s+"
        r"(?:saying|with\s+(?:the\s+)?content)\s+"
        r"(?P<body>'[^'\\]*'|\"[^\"\\]*\")",
        re.I,
    ),
)
_FUND_ACTION = re.compile(
    r"\b(?:fund|top\s+up)\s+(?:my\s+)?(?:trading\s+)?account\b", re.I
)
_FUND_VALUE = re.compile(
    r"\b(?:fund|top\s+up)\s+(?:my\s+)?(?:trading\s+)?account\s+(?:with|by)\s+"
    r"(?:(?P<currency_first>[A-Z]{3})\s+(?P<amount_first>\d[\d,]*(?:\.\d+)?)|"
    r"(?P<amount_last>\d[\d,]*(?:\.\d+)?)\s+(?P<currency_last>[A-Z]{3}))\b",
    re.I,
)
_MULTIPLICITY = re.compile(
    r"\b(?:again|another|additional|twice|two|second|both|each|every|multiple|"
    r"repeat|one\s+more|until|hourly|daily|weekly|monthly)\b",
    re.I,
)
_CONDITIONAL_WORD = re.compile(r"\b(?:if|unless|provided|when)\b", re.I)
_NEGATION = re.compile(
    r"\b(?:do\s+not|don't|never|not|avoid|without|refrain|stop)\b", re.I
)
_DEFERRED_CONDITION = re.compile(
    r"\b(?:only\s+after|after\s+approval|before\s+approval|once|unless|provided|when)\b",
    re.I,
)
_NON_AUTHORIZING_QUESTION = re.compile(
    r"^\s*(?:should|could|can|would|did|have|do)\s+(?:i|we)\b", re.I
)
_BOOK_AUTHORIZATION = re.compile(
    r"\b(?:book(?:\s+(?:a|the|another))?\s+flight|reserve(?:\s+(?:a|the))?\s+flight|"
    r"make\s+(?:the|a)\s+booking|secure\s+(?:a|the)\s+seat)\b",
    re.I,
)
_SEND_AUTHORIZATION = re.compile(r"\bsend\s+(?:a\s+)?message\b", re.I)

_VALUE_TOKEN = r"(?:'[^'\\]*'|\"[^\"\\]*\"|[A-Za-z0-9][\w.-]*)"
_ROUTE_TEXT = rf"from\s+{_VALUE_TOKEN}\s+to\s+{_VALUE_TOKEN}"
_COURTESY = r"(?:(?:please\s+)|(?:(?:could|would|can|will)\s+you(?:\s+please)?\s+))?"
_BOOK_DIRECT_FULL = re.compile(
    rf"^\s*{_COURTESY}(?:book|reserve)\s+(?:a\s+|the\s+)?flight\s+"
    rf"{_ROUTE_TEXT}\s*[.!?]?\s*$",
    re.I,
)
_BOOK_QUERY_FULL = re.compile(
    rf"^\s*(?:check|find|get|look\s+up)\s+(?:the\s+)?(?:fare|flight\s+cost)\s+"
    rf"{_ROUTE_TEXT}\s+(?:and\s+then|then|and)\s+"
    rf"{_COURTESY}(?:book|reserve)\s+(?:a\s+|the\s+)?flight\s*[.!?]?\s*$",
    re.I,
)
_BOOK_CONDITION_FULL = re.compile(
    rf"^\s*(?:check|find|get|look\s+up)\s+(?:the\s+)?(?:fare|flight\s+cost)\s+"
    rf"{_ROUTE_TEXT}\s+(?:and\s*,?\s*)?if\s+(?:the\s+)?"
    rf"(?:fare|flight\s+cost|it)\s+(?:is\s+)?"
    rf"(?:under|below|less\s+than|at\s+most|no\s+more\s+than)\s+"
    rf"\d[\d,]*(?:\.\d+)?\s+[A-Z]{{3}}\s*,?\s*"
    rf"{_COURTESY}(?:book|reserve)\s+(?:a\s+|the\s+)?flight\s*[.!?]?\s*$",
    re.I,
)
_SEND_FULL_PATTERNS = tuple(re.compile(r"^\s*" + _COURTESY + pattern.pattern
                                       + r"\s*[.!?]?\s*$", re.I)
                            for pattern in _SEND_PATTERNS)
_FUND_FULL = re.compile(
    r"^\s*" + _COURTESY
    + r"(?:fund|top\s+up)\s+(?:my\s+)?(?:trading\s+)?account\s+(?:with|by)\s+"
    + r"(?:[A-Z]{3}\s+\d[\d,]*(?:\.\d+)?|"
      r"\d[\d,]*(?:\.\d+)?\s+[A-Z]{3})\s*[.!?]?\s*$",
    re.I,
)


def _masked(text: str) -> str:
    """Keep offsets stable while preventing quoted data from becoming intent."""
    characters = list(text)
    for match in _QUOTED.finditer(text):
        characters[match.start():match.end()] = " " * (match.end() - match.start())
    return "".join(characters)


def _literal(match: re.Match, group: str) -> tuple[str, int, int] | None:
    value = match.group(group)
    if value is None:
        return None
    start, end = match.span(group)
    if value[:1] in {"'", '"'}:
        if (len(value) < 2 or value[-1] != value[0]
                or end < len(match.string) and match.string[end].isalnum()):
            return None
        return value[1:-1], start + 1, end - 1
    return value, start, end


def _request_source(context: RepairContext) -> _RequestSource | None:
    store = context.prepared._store
    user, _ = current_request(store)
    if user is None:
        return None
    parts: list[str] = []
    segments = []
    offset = 0
    for source_index in user.source_indices:
        value = store.messages[source_index].to_dict().get("content")
        if not isinstance(value, str):
            return None
        if parts:
            parts.append("\n")
            offset += 1
        parts.append(value)
        segments.append((offset, offset + len(value), source_index, value))
        offset += len(value)
    return _RequestSource(user.event_id, user.source_indices, "".join(parts),
                          tuple(segments))


def _schema(context: RepairContext, tool: str) -> _DeclaredSchema | None:
    matches = []
    for declaration in getattr(context.prepared, "_tools", ()) or ():
        if not isinstance(declaration, dict):
            continue
        function = declaration.get("function")
        if not isinstance(function, dict) or function.get("name") != tool:
            continue
        parameters = function.get("parameters")
        if (not isinstance(parameters, dict)
                or parameters.get("type") not in {"object", "dict"}):
            continue
        properties, required = parameters.get("properties"), parameters.get("required")
        if (not isinstance(properties, dict) or not isinstance(required, list)
                or not required or len(required) != len(set(required))
                or not all(isinstance(name, str) and name in properties
                           for name in required)):
            continue
        if not all(isinstance(item, dict) for item in properties.values()):
            continue
        matches.append(_DeclaredSchema(tool, tuple(required), properties))
    return matches[0] if len(matches) == 1 else None


def _schema_accepts(schema: _DeclaredSchema, arguments: Mapping[str, Any]) -> bool:
    if set(arguments) != set(schema.required):
        return False
    for name, value in arguments.items():
        declaration = schema.properties[name]
        kind = declaration.get("type")
        if kind == "string":
            accepted = isinstance(value, str)
        elif kind == "integer":
            accepted = isinstance(value, int) and not isinstance(value, bool)
        elif kind in {"number", "float"}:
            accepted = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            return False
        enum = declaration.get("enum")
        if not accepted or isinstance(enum, list) and value not in enum:
            return False
    return True


def _layout(schema: _DeclaredSchema, tool: str) -> tuple[str, ...] | None:
    required = set(schema.required)
    for layout in TOOL_SEMANTICS[tool]["layouts"]:
        if required == set(layout):
            return tuple(layout)
    return None


def _current_attempts(context: RepairContext, source: _RequestSource) -> tuple[_Attempt, ...]:
    store = context.prepared._store
    rows = operation_records(store, current_request_only=True)
    attempts = [
        _Attempt(
            row.tool,
            _json(row.arguments) if isinstance(row.arguments, dict) else None,
            "returned",
            row.event_id,
            row.call_source_index,
            row.result_source_index,
            copy.deepcopy(row.observed_result),
            row.failure_reported,
        )
        for row in rows
    ]
    observed = {(row.event_id, row.tool_call_id) for row in rows}
    boundary = max(source.source_indices)
    for event in store.events:
        if event.kind != "tool_event" or min(event.source_indices) <= boundary:
            continue
        call_source_index = min(event.source_indices)
        message = store.messages[call_source_index].to_dict()
        for call in message.get("tool_calls") or ():
            if not isinstance(call, dict) or (event.event_id, call.get("id")) in observed:
                continue
            function = call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                continue
            arguments = _decode_arguments(function.get("arguments", {}))
            attempts.append(_Attempt(
                function["name"], None if arguments is None else _json(arguments),
                "pending" if not event.complete else "ambiguous",
                event.event_id, call_source_index,
            ))
    # A native receipt is required.  A parseable-looking text action without
    # one is an ambiguous attempt, not evidence that no attempt happened.
    for event in store.events:
        if event.kind != "assistant" or min(event.source_indices) <= boundary:
            continue
        message = store.messages[event.source_indices[0]].to_dict()
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for tool in TOOL_SEMANTICS:
            if re.match(r"^\s*\[?\s*" + re.escape(tool) + r"\s*\(", content, re.I):
                attempts.append(_Attempt(
                    tool, None, "ambiguous", event.event_id,
                    event.source_indices[0],
                ))
    return tuple(attempts)


def _success(tool: str, result: Any) -> bool:
    statuses = TOOL_SEMANTICS[tool]["success_statuses"]
    if isinstance(result, dict):
        success_field = TOOL_SEMANTICS[tool].get("success_field")
        if success_field is not None and success_field in result:
            return (result[success_field] is True
                    and result.get("success") is not False
                    and result.get("error") in (None, "", False, [], {}))
        if result.get("success") is True and result.get("error") in (None, "", False, [], {}):
            return True
        status = result.get("status")
        if tool == "fund_account":
            return (status == "Account funded successfully"
                    and isinstance(result.get("new_balance"), (int, float))
                    and not isinstance(result.get("new_balance"), bool))
        return isinstance(status, str) and status.casefold() in statuses
    return isinstance(result, str) and result.strip().casefold() in statuses


def _action_failure(tool: str, result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    success_field = TOOL_SEMANTICS[tool].get("success_field")
    return success_field is not None and result.get(success_field) is False


def _attempt_evidence(attempt: _Attempt, *, kind: str) -> dict[str, Any]:
    output = {
        "kind": kind,
        "event_id": attempt.event_id,
        "call_source_index": attempt.call_source_index,
        "status": attempt.status,
    }
    if attempt.result_source_index is not None:
        output["result_source_index"] = attempt.result_source_index
    return output


def _classify_action(tool: str, arguments: Mapping[str, Any],
                     attempts: tuple[_Attempt, ...], *, repeat: bool,
                     base_state: str, base_reason: str,
                     evidence: list[dict[str, Any]]) -> tuple[str, str, list[dict[str, Any]]]:
    if repeat:
        return "unknown", "ambiguous_multiplicity", evidence
    relevant = [row for row in attempts if row.tool == tool]
    exact = [row for row in relevant
             if row.arguments_json == _json(arguments)]
    conflicting = [row for row in relevant if row not in exact]
    if conflicting:
        evidence.extend(_attempt_evidence(row, kind="conflicting_action_attempt")
                        for row in conflicting)
        if any(row.arguments is None for row in conflicting):
            return "unknown", "ambiguous_action_attempt", evidence
        return "unknown", "conflicting_action_attempt", evidence
    if len(exact) > 1:
        evidence.extend(_attempt_evidence(row, kind="repeated_action_attempt")
                        for row in exact)
        return "unknown", "ambiguous_attempt_multiplicity", evidence
    if exact:
        attempt = exact[0]
        evidence.append(_attempt_evidence(attempt, kind="action_attempt"))
        if attempt.status == "pending":
            return "blocked", "pending_action_attempt", evidence
        if attempt.status == "ambiguous" or attempt.arguments is None:
            return "unknown", "ambiguous_action_attempt", evidence
        if attempt.failure_reported or _action_failure(tool, attempt.result):
            return "blocked", "failed_action_attempt", evidence
        if _success(tool, attempt.result):
            invalidators = TOOL_SEMANTICS[tool].get("invalidated_by", ())
            later = [row for row in attempts
                     if row.tool in invalidators
                     and row.call_source_index > (attempt.result_source_index
                                                   or attempt.call_source_index)]
            if later:
                evidence.extend(_attempt_evidence(
                    row, kind="intervening_invalidator") for row in later)
                return "unknown", "completed_action_later_invalidated", evidence
            return "completed", "unambiguous_success_receipt", evidence
        return "unknown", "ambiguous_action_receipt", evidence
    return base_state, base_reason, evidence


def _number(value: str) -> int | float | None:
    try:
        parsed = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    if not parsed.is_finite():
        return None
    if parsed == parsed.to_integral_value():
        return int(parsed)
    return float(parsed)


def _obligation(source: _RequestSource, tool: str,
                arguments: Mapping[str, Any], state: str, reason: str,
                spans: tuple[UserSourceSpan, ...],
                evidence: tuple[dict[str, Any], ...]) -> ActionObligation:
    identity = {
        "rules_version": ACTION_RULES_VERSION,
        "request_event_id": source.event_id,
        "tool": tool,
        "arguments": arguments,
        "source_spans": [span.to_receipt() for span in spans],
    }
    obligation_id = "sha256:" + hashlib.sha256(
        _json(identity).encode("utf-8")
    ).hexdigest()
    return ActionObligation(
        obligation_id, source.event_id, source.source_indices, tool,
        _json(arguments), state, reason, spans, _json(evidence),
    )


def _unknown(source: _RequestSource, tool: str, reason: str,
             *, evidence: tuple[dict[str, Any], ...] = ()) -> ActionObligation:
    return _obligation(source, tool, {}, "unknown", reason, (), evidence)


def _authorization_span(source: _RequestSource, tool: str) -> UserSourceSpan | None:
    """Prove one singular positive action without deriving its arguments."""
    masked = _masked(source.text)
    pattern = {
        "book_flight": _BOOK_AUTHORIZATION,
        "send_message": _SEND_AUTHORIZATION,
        "fund_account": _FUND_ACTION,
    }[tool]
    matches = tuple(pattern.finditer(masked))
    if len(matches) != 1 or _MULTIPLICITY.search(masked):
        return None
    match = matches[0]
    clause_start = max(masked.rfind(mark, 0, match.start()) for mark in ".!?") + 1
    clause_ends = [masked.find(mark, match.end()) for mark in ".!?" ]
    clause_end = min((end for end in clause_ends if end >= 0), default=len(masked))
    clause = masked[clause_start:clause_end]
    if (_NEGATION.search(clause) or _DEFERRED_CONDITION.search(clause)
            or _NON_AUTHORIZING_QUESTION.search(clause)):
        return None
    return source.span("action_authorization", match.start(), match.end())


def _completed_from_observation(
    context: RepairContext, source: _RequestSource, tool: str,
    attempts: tuple[_Attempt, ...],
) -> ActionObligation | None:
    """Classify a proved duplicate without reconstructing six action fields.

    This path is intentionally completion-only: the complete observed call
    supplies the effective arguments, while the user source supplies singular
    positive authorization.  It can remove an exact repeat but can never
    synthesize a missing call.
    """
    authorization = _authorization_span(source, tool)
    schema = _schema(context, tool)
    relevant = [row for row in attempts if row.tool == tool]
    successful = [
        row for row in relevant
        if row.status == "returned" and row.arguments is not None
        and not row.failure_reported and _success(tool, row.result)
    ]
    if (authorization is None or schema is None or len(relevant) != 1
            or len(successful) != 1
            or not _schema_accepts(schema, successful[0].arguments)):
        return None
    attempt = successful[0]
    invalidators = TOOL_SEMANTICS[tool].get("invalidated_by", ())
    if any(row.tool in invalidators
           and row.call_source_index > (attempt.result_source_index
                                         or attempt.call_source_index)
           for row in attempts):
        return None
    evidence = (
        schema.evidence(),
        _attempt_evidence(attempt, kind="action_attempt"),
    )
    return _obligation(
        source, tool, attempt.arguments, "completed",
        "unambiguous_success_receipt", (authorization,), evidence,
    )


def _fare_rows(records: tuple[ObservedOperation, ...], origin: str,
               destination: str) -> list[ObservedOperation]:
    output = []
    for row in records:
        if row.tool != "get_flight_cost" or not isinstance(row.arguments, dict):
            continue
        valid = (
            row.arguments == {"travel_from": origin, "travel_to": destination}
            or row.arguments == {"origin": origin, "destination": destination}
        )
        if valid:
            output.append(row)
    return output


def _fare_value(result: Any) -> tuple[int | float, str | None] | None:
    if not isinstance(result, dict):
        return None
    numeric = [(name, value) for name, value in result.items()
               if name in {"fare", "cost", "price"}
               and isinstance(value, (int, float)) and not isinstance(value, bool)]
    if len(numeric) != 1:
        return None
    currency = result.get("currency")
    if currency is not None and (not isinstance(currency, str)
                                 or not re.fullmatch(r"[A-Z]{3}", currency)):
        return None
    return numeric[0][1], currency


def _parse_book(context: RepairContext, source: _RequestSource,
                attempts: tuple[_Attempt, ...]) -> ActionObligation:
    text, masked = source.text, _masked(source.text)
    schema = _schema(context, "book_flight")
    route_matches = tuple(_ROUTE.finditer(text))
    action_matches = tuple(_BOOK_ACTION.finditer(masked))
    if schema is None:
        return _unknown(source, "book_flight", "missing_or_ambiguous_declared_schema")
    layout = _layout(schema, "book_flight")
    if _MULTIPLICITY.search(masked):
        return _unknown(source, "book_flight", "ambiguous_multiplicity",
                        evidence=(schema.evidence(),))
    complete_grammar = any(pattern.fullmatch(text) for pattern in (
        _BOOK_DIRECT_FULL, _BOOK_QUERY_FULL, _BOOK_CONDITION_FULL,
    ))
    if (not complete_grammar or _NEGATION.search(masked)
            or len(action_matches) != 1 or len(route_matches) != 1
            or layout is None):
        return _unknown(source, "book_flight", "unsupported_or_ambiguous_booking_grammar",
                        evidence=(schema.evidence(),))
    route = route_matches[0]
    origin = _literal(route, "origin")
    destination = _literal(route, "destination")
    if origin is None or destination is None:
        return _unknown(source, "book_flight", "incomplete_route_source_span")
    origin_value, origin_start, origin_end = origin
    destination_value, destination_start, destination_end = destination
    arguments = {layout[0]: origin_value, layout[1]: destination_value}
    spans = (
        source.span(layout[0], origin_start, origin_end),
        source.span(layout[1], destination_start, destination_end),
    )
    if any(span is None for span in spans) or not _schema_accepts(schema, arguments):
        return _unknown(source, "book_flight", "booking_schema_or_source_mismatch",
                        evidence=(schema.evidence(),))
    evidence = [schema.evidence()]
    records = operation_records(context.prepared._store, current_request_only=True)
    query_requested = bool(_FARE_QUERY.search(masked))
    condition_matches = tuple(_CONDITION.finditer(masked))
    conditional_words = tuple(_CONDITIONAL_WORD.finditer(masked))
    base_state, base_reason = "ready_unexecuted", "explicit_action_ready"
    if conditional_words and len(condition_matches) != 1:
        base_state, base_reason = "unknown", "unsupported_or_ambiguous_condition"
    elif len(condition_matches) > 1:
        base_state, base_reason = "unknown", "ambiguous_condition"
    elif query_requested or condition_matches:
        rows = _fare_rows(records, origin_value, destination_value)
        pending_query = [row for row in attempts
                         if row.tool == "get_flight_cost"
                         and row.arguments in (
                             {"travel_from": origin_value, "travel_to": destination_value},
                             {"origin": origin_value, "destination": destination_value},
                         ) and row.status != "returned"]
        if pending_query:
            base_state, base_reason = "blocked", "fare_query_attempt_pending"
            evidence.extend(_attempt_evidence(row, kind="fare_query_attempt")
                            for row in pending_query)
        elif len(rows) != 1:
            base_state = "blocked" if not rows else "unknown"
            base_reason = "fare_query_not_completed" if not rows else "ambiguous_fare_query"
        else:
            row = rows[0]
            evidence.append({
                "kind": "fare_query_result",
                "event_id": row.event_id,
                "call_source_index": row.call_source_index,
                "result_source_index": row.result_source_index,
                "call_signature_id": row.call_signature_id,
                "observation_version": row.observation_version,
            })
            fare = _fare_value(row.observed_result)
            if row.failure_reported:
                base_state, base_reason = "blocked", "fare_query_failed"
            elif fare is None:
                base_state, base_reason = "unknown", "ambiguous_fare_query_result"
            elif condition_matches:
                condition = condition_matches[0]
                threshold = _number(condition.group("amount"))
                currency = condition.group("currency").upper()
                actual, actual_currency = fare
                evidence[-1].update(
                    observed_fare=actual, observed_currency=actual_currency,
                    operator=condition.group("operator").casefold(),
                    threshold=threshold, threshold_currency=currency,
                )
                if threshold is None or actual_currency != currency:
                    base_state, base_reason = "unknown", "currency_or_threshold_unknown"
                else:
                    operator = condition.group("operator").casefold()
                    holds = actual <= threshold if operator in {
                        "at most", "no more than",
                    } else actual < threshold
                    if not holds:
                        base_state, base_reason = "blocked", "condition_false"
    state, reason, evidence = _classify_action(
        "book_flight", arguments, attempts,
        repeat=bool(_MULTIPLICITY.search(masked)),
        base_state=base_state, base_reason=base_reason, evidence=evidence,
    )
    return _obligation(source, "book_flight", arguments, state, reason,
                       tuple(span for span in spans if span is not None),
                       tuple(evidence))


def _parse_send(context: RepairContext, source: _RequestSource,
                attempts: tuple[_Attempt, ...]) -> ActionObligation:
    schema = _schema(context, "send_message")
    if schema is None:
        return _unknown(source, "send_message", "missing_or_ambiguous_declared_schema")
    layout = _layout(schema, "send_message")
    if _MULTIPLICITY.search(_masked(source.text)):
        return _unknown(source, "send_message", "ambiguous_multiplicity",
                        evidence=(schema.evidence(),))
    matches = [match for pattern in _SEND_FULL_PATTERNS
               for match in (pattern.fullmatch(source.text),) if match is not None]
    masked = _masked(source.text)
    if (len(matches) != 1 or layout is None or _NEGATION.search(masked)
            or _CONDITIONAL_WORD.search(masked)
            or _NON_AUTHORIZING_QUESTION.search(masked)):
        return _unknown(source, "send_message", "unsupported_or_ambiguous_send_grammar",
                        evidence=(schema.evidence(),))
    match = matches[0]
    target, body = _literal(match, "target"), _literal(match, "body")
    if target is None or body is None:
        return _unknown(source, "send_message", "incomplete_send_source_span")
    target_value, target_start, target_end = target
    body_value, body_start, body_end = body
    target_prefix = source.text[match.start():match.start("target")].casefold()
    if (layout[0] == "receiver_id"
            and re.search(r"\b(?:receiver|recipient)\s+id\s+$",
                          target_prefix) is None):
        return _unknown(source, "send_message", "recipient_role_schema_mismatch",
                        evidence=(schema.evidence(),))
    if layout[0] == "phone_number" and "phone number" not in match.group(0).casefold():
        return _unknown(source, "send_message", "recipient_role_schema_mismatch",
                        evidence=(schema.evidence(),))
    if layout[0] != "phone_number" and "phone number" in match.group(0).casefold():
        return _unknown(source, "send_message", "recipient_role_schema_mismatch",
                        evidence=(schema.evidence(),))
    arguments = {layout[0]: target_value, layout[1]: body_value}
    spans = (
        source.span(layout[0], target_start, target_end),
        source.span(layout[1], body_start, body_end),
    )
    if any(span is None for span in spans) or not _schema_accepts(schema, arguments):
        return _unknown(source, "send_message", "send_schema_or_source_mismatch",
                        evidence=(schema.evidence(),))
    state, reason, evidence = _classify_action(
        "send_message", arguments, attempts,
        repeat=bool(_MULTIPLICITY.search(_masked(source.text))),
        base_state="ready_unexecuted", base_reason="explicit_action_ready",
        evidence=[schema.evidence()],
    )
    return _obligation(source, "send_message", arguments, state, reason,
                       tuple(span for span in spans if span is not None),
                       tuple(evidence))


def _parse_fund(context: RepairContext, source: _RequestSource,
                attempts: tuple[_Attempt, ...]) -> ActionObligation:
    schema = _schema(context, "fund_account")
    if schema is None:
        return _unknown(source, "fund_account", "missing_or_ambiguous_declared_schema")
    layout = _layout(schema, "fund_account")
    if _MULTIPLICITY.search(_masked(source.text)):
        return _unknown(source, "fund_account", "ambiguous_multiplicity",
                        evidence=(schema.evidence(),))
    matches = tuple(_FUND_VALUE.finditer(_masked(source.text)))
    masked = _masked(source.text)
    if (not _FUND_FULL.fullmatch(source.text) or len(matches) != 1
            or layout is None or _NEGATION.search(masked)
            or _CONDITIONAL_WORD.search(masked)
            or _NON_AUTHORIZING_QUESTION.search(masked)):
        return _unknown(source, "fund_account", "unsupported_or_ambiguous_funding_grammar",
                        evidence=(schema.evidence(),))
    match = matches[0]
    currency_group = "currency_first" if match.group("currency_first") else "currency_last"
    amount_group = "amount_first" if match.group("amount_first") else "amount_last"
    currency = match.group(currency_group).upper()
    amount = _number(match.group(amount_group))
    fixed_currency = TOOL_SEMANTICS["fund_account"]["fixed_currency"]
    if amount is None or len(layout) == 1 and currency != fixed_currency:
        return _unknown(source, "fund_account", "currency_or_amount_unknown",
                        evidence=(schema.evidence(),))
    arguments = {"amount": amount}
    if "currency" in layout:
        arguments["currency"] = currency
    amount_span = source.span("amount", *match.span(amount_group))
    currency_span = source.span(
        "currency" if "currency" in layout else "fixed_currency_semantics",
        *match.span(currency_group),
    )
    spans = (amount_span, currency_span)
    if any(span is None for span in spans) or not _schema_accepts(schema, arguments):
        return _unknown(source, "fund_account", "funding_schema_or_source_mismatch",
                        evidence=(schema.evidence(),))
    evidence = [schema.evidence()]
    if len(layout) == 1:
        evidence.append({
            "kind": "registered_fixed_currency_semantics",
            "tool": "fund_account", "currency": fixed_currency,
        })
    state, reason, evidence = _classify_action(
        "fund_account", arguments, attempts,
        repeat=bool(_MULTIPLICITY.search(_masked(source.text))),
        base_state="ready_unexecuted", base_reason="explicit_action_ready",
        evidence=evidence,
    )
    return _obligation(source, "fund_account", arguments, state, reason,
                       tuple(span for span in spans if span is not None),
                       tuple(evidence))


class Policy:
    """Pure CPU policy for inspecting, proposing, guarding and filtering actions."""

    def inspect(self, context: RepairContext) -> LedgerAssessment:
        source = _request_source(context)
        if source is None:
            receipt = {
                "version": ACTION_LEDGER_VERSION,
                "rules_version": ACTION_RULES_VERSION,
                "status": "no_current_request",
                "request_event_id": None,
                "prefix_message_count": len(context.prepared._store.messages),
                "obligations": [], "ready_obligation_ids": [],
            }
            return LedgerAssessment((), None, len(context.prepared._store.messages),
                                    _json(receipt))
        attempts = _current_attempts(context, source)
        masked = _masked(source.text)
        recognized = []
        if _BOOK_AUTHORIZATION.search(masked):
            recognized.append("book_flight")
        if re.search(r"\bsend\b.{0,80}\bmessage\b", masked, re.I):
            recognized.append("send_message")
        if _FUND_ACTION.search(masked):
            recognized.append("fund_account")
        if len(recognized) > 1:
            obligations = tuple(_unknown(source, tool, "multiple_supported_actions")
                                for tool in recognized)
        elif recognized:
            tool = recognized[0]
            completed = _completed_from_observation(
                context, source, tool, attempts
            )
            if completed is not None:
                obligations = (completed,)
            elif tool == "book_flight":
                obligations = (_parse_book(context, source, attempts),)
            elif tool == "send_message":
                obligations = (_parse_send(context, source, attempts),)
            else:
                obligations = (_parse_fund(context, source, attempts),)
        else:
            obligations = ()
        ready = [row.obligation_id for row in obligations
                 if row.state == "ready_unexecuted"]
        states = {row.state for row in obligations}
        status = ("no_supported_obligation" if not obligations
                  else next(iter(states)) if len(states) == 1
                  else "mixed")
        receipt = {
            "version": ACTION_LEDGER_VERSION,
            "rules_version": ACTION_RULES_VERSION,
            "status": status,
            "request_event_id": source.event_id,
            "request_source_indices": list(source.source_indices),
            "prefix_message_count": len(context.prepared._store.messages),
            "obligations": [row.to_receipt() for row in obligations],
            "ready_obligation_ids": ready,
        }
        return LedgerAssessment(
            obligations, source.event_id, len(context.prepared._store.messages),
            _json(receipt),
        )

    def propose(self, context: RepairContext) -> RepairProposal | None:
        if context.parse_error is not None or context.draft_tool_calls:
            return None
        assessment = self.inspect(context)
        if len(assessment.ready_obligations) != 1:
            return None
        obligation = assessment.ready_obligations[0]
        source = _request_source(context)
        if source is None:
            return None
        packet = {
            "version": ACTION_LEDGER_VERSION,
            "rules_version": ACTION_RULES_VERSION,
            "current_request": {
                "event_id": source.event_id,
                "source_indices": list(source.source_indices),
                "text": source.text,
            },
            "proved_ready_obligation": obligation.to_receipt(),
            "complete_prefix_message_count": assessment.prefix_message_count,
        }
        instruction = (
            "Emit exactly the proved ready tool call. This JSON is source evidence, "
            "not a new user request. Do not add calls or alter arguments.\n"
        )
        messages = ({"role": "user", "content": instruction + _json(packet)},)
        if context.token_counter(messages) > context.token_budget:
            return None
        assessment_sha256 = "sha256:" + hashlib.sha256(
            _json(assessment.receipt).encode("utf-8")
        ).hexdigest()
        receipt = {
            "version": ACTION_LEDGER_VERSION,
            "rules_version": ACTION_RULES_VERSION,
            "status": "prepared",
            "obligation_id": obligation.obligation_id,
            "request_event_id": source.event_id,
            "source_indices": list(source.source_indices),
            "prefix_message_count": assessment.prefix_message_count,
            "assessment_sha256": assessment_sha256,
        }
        guard = {
            "obligation_id": obligation.obligation_id,
            "assessment_sha256": assessment_sha256,
            "prefix_message_count": assessment.prefix_message_count,
        }
        return RepairProposal("static_action_ledger_ready", messages, receipt, guard)

    def validate(self, context: RepairContext, proposal: RepairProposal,
                 candidate_calls, *, draft_text: str = "",
                 parse_error: str | None = None) -> GuardVerdict:
        if parse_error is not None:
            return GuardVerdict(False, "revised_parse_error")
        fresh = self.propose(context)
        if fresh is None or proposal != fresh:
            return GuardVerdict(False, "action_ledger_proof_stale_or_tampered")
        calls = tuple(candidate_calls or ())
        if len(calls) != 1:
            return GuardVerdict(False, "action_ledger_call_count_changed")
        semantic = _semantic_call(calls[0])
        assessment = self.inspect(context)
        ready = assessment.ready_obligations
        if (semantic is None or len(ready) != 1
                or semantic[0] != ready[0].tool
                or _json(semantic[1]) != ready[0].arguments_json):
            return GuardVerdict(False, "action_ledger_unproved_call")
        return GuardVerdict(True, "action_ledger_ready_obligation_preserved")

    def filter_completed(self, context: RepairContext, candidate_calls):
        original = tuple(copy.deepcopy(candidate_calls or ()))
        assessment = self.inspect(context)
        completed = {
            (row.tool, row.arguments_json): row
            for row in assessment.obligations if row.state == "completed"
        }
        kept, removed = [], []
        for index, call in enumerate(original):
            semantic = _semantic_call(call)
            obligation = None if semantic is None else completed.get(
                (semantic[0], _json(semantic[1]))
            )
            if obligation is None:
                kept.append(call)
                continue
            success_sources = [item for item in obligation.evidence
                               if item.get("kind") == "action_attempt"]
            item = {
                "candidate_index": index,
                "obligation_id": obligation.obligation_id,
                "tool": obligation.tool,
                "success_sources": success_sources,
            }
            if isinstance(call, dict) and isinstance(call.get("id"), str):
                item["candidate_call_id"] = call["id"]
            removed.append(item)
        receipt = {
            "version": ACTION_LEDGER_VERSION,
            "rules_version": ACTION_RULES_VERSION,
            "status": "filtered_completed_duplicates" if removed
                      else "no_proven_completed_duplicate",
            "request_event_id": assessment.request_event_id,
            "prefix_message_count": assessment.prefix_message_count,
            "removed": removed,
            "preserved_candidate_count": len(kept),
            "assessment": assessment.receipt,
        }
        return tuple(kept), receipt


__all__ = [
    "ACTION_LEDGER_VERSION", "ACTION_RULES_VERSION", "ACTION_STATES",
    "ActionObligation", "LedgerAssessment", "Policy", "TOOL_SEMANTICS",
    "UserSourceSpan",
]
