"""Deterministic field repairs backed by a complete current-request proof."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .observations import ObservedOperation, current_request, operation_records
from .repair_protocol import GuardVerdict, RepairContext


VERSION = "verified-binding-v1"
PROOF_REGISTRY_VERSION = "verified-binding-rules-v1"
PROOF_KINDS = ("credential_literal", "quoted_text_literal", "ordinal_list_result")

_CREDENTIAL = re.compile(
    r"\b(?:secure|access|authorization)\s+(?:token|code)\b\s*"
    r"(?:(?:is|of)\s+|[:=]\s*)?"
    r"(?:'(?P<single>[^']+)'|\"(?P<double>[^\"]+)\"|"
    r"(?P<plain>[A-Za-z0-9][A-Za-z0-9_@.+-]*))",
    re.IGNORECASE,
)
_CARD_ID = re.compile(
    r"\bcard\s+(?:with\s+)?id\b\s*(?:[:=]\s*)?"
    r"(?:'(?P<single>[^']+)'|\"(?P<double>[^\"]+)\"|"
    r"(?P<plain>[A-Za-z0-9][A-Za-z0-9_@.+-]*))",
    re.IGNORECASE,
)
_QUOTED = re.compile(
    r"\b(?P<label>message|note|description|saying|say)\b\s*"
    r"(?:(?:is|of|like)\s+|[:=]\s*)?"
    r"(?:'(?P<single>[^']+)'|\"(?P<double>[^\"]+)\")",
    re.IGNORECASE,
)
_TEMPLATE = re.compile(r"\$\{[^{}]+\}|\{[^{}]+\}")
_QUOTED_SPAN = re.compile(r"(?<!\w)'[^']*'|(?<!\w)\"[^\"]*\"")
_NEGATED_ORDINAL = re.compile(
    r"\b(?:not|instead\s+of|rather\s+than)\s+(?:the\s+)?(?:first|last)\b",
    re.IGNORECASE,
)
_MUTATION = re.compile(
    r"^(?:add|book|buy|cancel|create|delete|edit|fill|move|place|"
    r"post|purchase|register|remove|set|submit|transfer|update|write)(?:_|[A-Z]|$)",
    re.IGNORECASE,
)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def _unique_pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key")
        output[key] = value
    return output


def _calls(calls) -> tuple[dict[str, Any], ...] | None:
    output = []
    for call in calls or ():
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            return None
        function = call["function"]
        name, arguments = function.get("name"), function.get("arguments")
        if not isinstance(name, str) or not name:
            return None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments, object_pairs_hook=_unique_pairs)
            except ValueError:
                return None
        if not isinstance(arguments, dict):
            return None
        try:
            _json(arguments)
        except (TypeError, ValueError):
            return None
        output.append({"tool": name, "arguments": arguments})
    return tuple(output)


def _signature(calls: tuple[dict[str, Any], ...]) -> str:
    return "sha256:" + hashlib.sha256(_json(calls).encode("utf-8")).hexdigest()


def _transport_shape(calls):
    output = []
    for call in calls or ():
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            return None
        item = copy.deepcopy(call)
        arguments = item["function"].get("arguments")
        item["function"]["arguments"] = "<arguments>" if isinstance(arguments, str) else {}
        output.append(item)
    return tuple(output)


def _schema(context: RepairContext, tool: str) -> dict[str, Any] | None:
    matches = []
    for declaration in getattr(context.prepared, "_tools", ()) or ():
        if not isinstance(declaration, dict):
            continue
        function = declaration.get("function")
        if isinstance(function, dict) and function.get("name") == tool:
            parameters = function.get("parameters") or {}
            properties = parameters.get("properties") if isinstance(parameters, dict) else None
            if isinstance(properties, dict):
                matches.append(properties)
    return matches[0] if len(matches) == 1 else None


def _string_slot(context: RepairContext, call: dict[str, Any], field: str,
                 after: str | None = None) -> bool:
    properties = _schema(context, call["tool"])
    declaration = properties.get(field) if properties is not None else None
    return (isinstance(declaration, dict) and declaration.get("type") == "string"
            and isinstance(call["arguments"].get(field), str)
            and (after is None or not isinstance(declaration.get("enum"), list)
                 or after in declaration["enum"]))


def _request(context: RepairContext):
    store = context.prepared._store
    user, _ = current_request(store)
    if user is None:
        return None
    text = "\n".join(str(store.messages[index].to_dict().get("content") or "")
                     for index in user.source_indices)
    return user, text


def _literal_matches(pattern: re.Pattern, text: str):
    found = []
    quoted_spans = tuple(match.span() for match in _QUOTED_SPAN.finditer(text))
    for match in pattern.finditer(text):
        if pattern in (_CREDENTIAL, _CARD_ID) and any(start <= match.start() < end
                                          for start, end in quoted_spans):
            continue
        group = next((name for name in ("single", "double", "plain")
                      if match.group(name) is not None), None)
        if group is None:
            continue
        value = match.group(group)
        start, end = match.span(group)
        # This registry supports unescaped, fully delimited literals. A quote
        # inside a word or an escaped delimiter is not a complete source span.
        if group != "plain" and (value.endswith("\\") or
                (match.end() < len(text) and text[match.end()].isalnum())):
            continue
        if group == "plain":
            value = value.rstrip(".")
            end = start + len(value)
            if not value or not any(character.isdigit() for character in value):
                continue
        found.append((value, (start, end), match))
    return tuple(found)


def _sentence_at(text: str, offset: int) -> str:
    boundaries = [0, *(match.end() for match in re.finditer(r"[.!?](?:\s|$)", text)),
                  len(text)]
    start = max(boundary for boundary in boundaries if boundary <= offset)
    end = min(boundary for boundary in boundaries if boundary > offset)
    return text[start:end]


def _list_result(row: ObservedOperation, field: str):
    value = row.observed_result
    path: tuple[str | int, ...] = ()
    if isinstance(value, dict):
        value = value.get(field)
        path = (field,)
    elif field != "airports":
        return None
    if isinstance(value, str) and len(value) <= 20_000 and value.startswith("["):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError, TypeError, MemoryError):
            return None
    if (not isinstance(value, list) or not value
            or not all(isinstance(item, str) and item for item in value)
            or len(set(value)) != len(value)):
        return None
    return path, tuple(value)


@dataclass(frozen=True)
class BindingProof:
    kind: str
    request_event_id: str
    held_signature: str
    call_index: int
    tool: str
    field_path: tuple[str, ...]
    before: str
    after: str
    literal_span: tuple[int, int] | None = None
    producer_event_id: str | None = None
    producer_call_signature_id: str | None = None
    producer_result_source_index: int | None = None
    producer_observation_version: int | None = None
    producer_result_path: tuple[str | int, ...] | None = None


@dataclass(frozen=True)
class BindingProposal:
    request_event_id: str
    held_signature: str
    prefix_source_index: int
    proofs: tuple[BindingProof, ...]
    version: str = VERSION
    proof_registry_version: str = PROOF_REGISTRY_VERSION

    def to_receipt(self) -> dict[str, Any]:
        return asdict(self)


class BindingRule(Protocol):
    kind: str

    def derive(self, context: RepairContext, request: str,
               held: tuple[dict[str, Any], ...], records: tuple[ObservedOperation, ...],
               request_event_id: str, held_signature: str) -> tuple[BindingProof, ...]: ...


class CredentialRule:
    kind = "credential_literal"
    action_scopes = {
        "set_budget_limit": re.compile(r"\b(?:set|establish)\b.{0,50}\bbudget\b", re.I),
        "book_flight": re.compile(r"\b(?:book|booking|reservation|reserve)\b", re.I),
    }

    def derive(self, context, request, held, records, request_event_id, held_signature):
        matches = _literal_matches(_CREDENTIAL, request)
        if len(matches) != 1:
            return ()
        value, span, match = matches[0]
        clause = _sentence_at(request, match.start())
        output = []
        for index, call in enumerate(held):
            action = self.action_scopes.get(call["tool"])
            if action is None or not action.search(clause):
                continue
            if call["tool"] == "book_flight":
                card_matches = _literal_matches(_CARD_ID, request)
                if (len(card_matches) != 1
                        or call["arguments"].get("card_id") != card_matches[0][0]):
                    continue
            if not _string_slot(context, call, "access_token", value):
                continue
            properties = _schema(context, call["tool"])
            if any(name in properties for name in ("refresh_token", "client_secret", "password")):
                continue
            before = call["arguments"]["access_token"]
            if before != value:
                output.append(BindingProof(self.kind, request_event_id, held_signature,
                                           index, call["tool"], ("access_token",),
                                           before, value, literal_span=span))
        return tuple(output) if len(output) == 1 else ()


class QuotedTextRule:
    kind = "quoted_text_literal"

    def derive(self, context, request, held, records, request_event_id, held_signature):
        matches = _literal_matches(_QUOTED, request)
        if len(matches) != 1:
            return ()
        value, span, match = matches[0]
        if _TEMPLATE.search(value):
            return ()
        label = match.group("label").casefold()
        targets = (("send_message", "message") if label != "description"
                   else ("create_ticket", "description"))
        output = []
        for index, call in enumerate(held):
            tool, field = targets
            if call["tool"] != tool or not _string_slot(context, call, field, value):
                continue
            before = call["arguments"][field]
            if before != value:
                output.append(BindingProof(self.kind, request_event_id, held_signature,
                                           index, tool, (field,), before, value,
                                           literal_span=span))
        return tuple(output) if len(output) == 1 else ()


class OrdinalListRule:
    kind = "ordinal_list_result"

    def derive(self, context, request, held, records, request_event_id, held_signature):
        if _NEGATED_ORDINAL.search(request):
            return ()
        targets = []
        if re.search(r"\blast\s+stock\b", request, re.I) or re.search(
                r"\bstocks?\s+listed\b.{0,40}\blast\s+one\b", request, re.I):
            targets.append(("get_watchlist", "watchlist", "get_stock_info", "symbol", -1))
        if re.search(r"\bfirst\s+airport\b", request, re.I):
            targets.append(("list_all_airports", "airports", "get_flight_cost", "travel_from", 0))
        if re.search(r"\blast\s+airport\b", request, re.I):
            targets.append(("list_all_airports", "airports", "get_flight_cost", "travel_to", -1))
        output = []
        for producer_tool, result_field, consumer_tool, argument_field, position in targets:
            candidate_calls = [(index, call) for index, call in enumerate(held)
                               if call["tool"] == consumer_tool
                               and _string_slot(context, call, argument_field)]
            if len(candidate_calls) != 1:
                continue
            producer_rows = [(row, parsed) for row in records
                             if row.tool == producer_tool and not row.failure_reported
                             and row.result_source_index < len(context.prepared._store.messages)
                             if (parsed := _list_result(row, result_field)) is not None]
            if len(producer_rows) != 1:
                continue
            row, (path, items) = producer_rows[0]
            entity = "stock" if producer_tool == "get_watchlist" else "airport"
            if any(other is not row and not other.failure_reported
                   and entity in (other.tool + " " + " ".join(
                       map(str, other.observed_result.keys())
                       if isinstance(other.observed_result, dict) else ())).casefold()
                   and (isinstance(other.observed_result, list)
                        or isinstance(other.observed_result, dict) and any(
                            isinstance(value, list) for value in other.observed_result.values()))
                   for other in records):
                continue
            if any(later.call_source_index > row.result_source_index
                   and _MUTATION.match(later.tool) for later in records):
                continue
            index, call = candidate_calls[0]
            value = items[position]
            if not _string_slot(context, call, argument_field, value):
                continue
            before = call["arguments"][argument_field]
            if before != value:
                output.append(BindingProof(
                    self.kind, request_event_id, held_signature, index,
                    call["tool"], (argument_field,), before, value,
                    producer_event_id=row.event_id,
                    producer_call_signature_id=row.call_signature_id,
                    producer_result_source_index=row.result_source_index,
                    producer_observation_version=row.observation_version,
                    producer_result_path=path + ((len(items) - 1) if position == -1 else position,),
                ))
        return tuple(output)


RULES: tuple[BindingRule, ...] = (CredentialRule(), QuotedTextRule(), OrdinalListRule())


class Policy:
    def propose(self, context: RepairContext) -> BindingProposal | None:
        if context.parse_error is not None:
            return None
        held = _calls(context.draft_tool_calls)
        request = _request(context)
        if not held or request is None:
            return None
        user, text = request
        signature = _signature(held)
        records = operation_records(context.prepared._store)
        proofs = tuple(proof for rule in RULES for proof in rule.derive(
            context, text, held, records, user.event_id, signature))
        fields = [(proof.call_index, proof.field_path) for proof in proofs]
        if not proofs or len(set(fields)) != len(fields):
            return None
        return BindingProposal(user.event_id, signature,
                               len(context.prepared._store.messages) - 1, proofs)

    def _fresh(self, context: RepairContext, proposal: BindingProposal) -> bool:
        return (isinstance(proposal, BindingProposal)
                and proposal.version == VERSION
                and proposal.proof_registry_version == PROOF_REGISTRY_VERSION
                and self.propose(context) == proposal)

    def validate(self, context: RepairContext, proposal: BindingProposal,
                 candidate_calls) -> GuardVerdict:
        if not self._fresh(context, proposal):
            return GuardVerdict(False, "binding_proof_stale_or_tampered")
        held = _calls(context.draft_tool_calls)
        candidate = _calls(candidate_calls)
        if held is None or candidate is None or len(held) != len(candidate):
            return GuardVerdict(False, "binding_call_structure_changed")
        if _transport_shape(candidate_calls) != _transport_shape(context.draft_tool_calls):
            return GuardVerdict(False, "binding_transport_changed")
        expected = copy.deepcopy(held)
        for proof in proposal.proofs:
            if proof.call_index >= len(expected) or len(proof.field_path) != 1:
                return GuardVerdict(False, "binding_proof_invalid_path")
            call = expected[proof.call_index]
            field = proof.field_path[0]
            if (call["tool"] != proof.tool or call["arguments"].get(field) != proof.before):
                return GuardVerdict(False, "binding_proof_source_changed")
            call["arguments"][field] = proof.after
        if candidate != tuple(expected):
            return GuardVerdict(False, "binding_unproved_field_change")
        return GuardVerdict(True, "verified_bindings_preserved")

    def apply(self, context: RepairContext, proposal: BindingProposal,
              candidate_calls) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
        original = tuple(copy.deepcopy(candidate_calls or ()))
        receipt = {"version": VERSION, "proof_registry_version": PROOF_REGISTRY_VERSION,
                   "status": "refused", "proofs": proposal.to_receipt().get("proofs", [])
                   if isinstance(proposal, BindingProposal) else []}
        if not self._fresh(context, proposal):
            return original, {**receipt, "reason": "binding_proof_stale_or_tampered"}
        held = _calls(context.draft_tool_calls)
        if (_calls(original) != held or held is None
                or _transport_shape(original) != _transport_shape(context.draft_tool_calls)):
            return original, {**receipt, "reason": "binding_held_call_changed"}
        corrected = copy.deepcopy(original)
        for index, call in enumerate(corrected):
            changes = [proof for proof in proposal.proofs if proof.call_index == index]
            if not changes:
                continue
            function = call["function"]
            arguments = copy.deepcopy(held[index]["arguments"])
            for proof in changes:
                arguments[proof.field_path[0]] = proof.after
            function["arguments"] = _json(arguments) if isinstance(function["arguments"], str) else arguments
        verdict = self.validate(context, proposal, corrected)
        if not verdict.accepted:
            return original, {**receipt, "reason": verdict.reason}
        return tuple(corrected), {**receipt, "status": "applied", "reason": verdict.reason,
                                  "request_event_id": proposal.request_event_id,
                                  "held_signature": proposal.held_signature}


__all__ = ["Policy", "BindingProposal", "BindingProof", "BindingRule", "RULES",
           "VERSION", "PROOF_REGISTRY_VERSION", "PROOF_KINDS"]
