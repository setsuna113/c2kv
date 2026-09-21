"""Deterministic repairs for three source-proven cross-event relations."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .observations import ObservedOperation, current_request, operation_records
from .repair_protocol import GuardVerdict, RepairContext


VERSION = "relational-binding-v1"
PROOF_REGISTRY_VERSION = "verified-binding-relations-v2"
PROOF_KINDS = (
    "mean_from_wc_results",
    "delete_receiver_from_message_receipt",
    "card_key_from_card_number",
)

_NEGATION = re.compile(
    r"\b(?:do\s+not|don't|never|without|instead\s+of|rather\s+than)\b",
    re.IGNORECASE,
)
_FILE_MUTATION_TOOLS = frozenset({
    "append", "append_file", "cp", "echo", "edit_file", "move_file", "mv",
    "remove_file", "replace", "rm", "sed", "touch", "truncate", "write",
    "write_file",
})
_MESSAGE_MUTATION_TOOLS = frozenset({
    "delete_message", "edit_message", "update_message",
})
_CARD_MUTATION = re.compile(
    r"^(?:(?:add|delete|edit|register|remove|set|update)_credit_card|"
    r"credit_card_(?:add|delete|edit|register|remove|set|update))$",
    re.IGNORECASE,
)
_WC_TYPES = {"l": "lines", "w": "words", "c": "characters"}
_CARD_ROLES = frozenset({"default", "main", "primary"})


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


def _arguments(value: Any) -> dict[str, Any] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value, object_pairs_hook=_unique_pairs)
        except ValueError:
            return None
    if not isinstance(value, dict):
        return None
    try:
        _json(value)
    except (TypeError, ValueError):
        return None
    return value


def _calls(calls) -> tuple[dict[str, Any], ...] | None:
    output = []
    for call in calls or ():
        if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
            return None
        function = call["function"]
        name = function.get("name")
        arguments = _arguments(function.get("arguments"))
        if not isinstance(name, str) or not name or arguments is None:
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
        value = item["function"].get("arguments")
        item["function"]["arguments"] = "<arguments>" if isinstance(value, str) else {}
        output.append(item)
    return tuple(output)


def _proof_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_proof_value(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _proof_value(item)) for key, item in sorted(value.items()))
    return value


def _runtime_value(value: Any) -> Any:
    if isinstance(value, tuple):
        if all(isinstance(item, tuple) and len(item) == 2
               and isinstance(item[0], str) for item in value):
            return {key: _runtime_value(item) for key, item in value}
        return [_runtime_value(item) for item in value]
    return value


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
                 after: str) -> bool:
    properties = _schema(context, call["tool"])
    declaration = properties.get(field) if properties is not None else None
    return (isinstance(declaration, dict) and declaration.get("type") == "string"
            and isinstance(call["arguments"].get(field), str)
            and (not isinstance(declaration.get("enum"), list)
                 or after in declaration["enum"]))


def _number_array_slot(context: RepairContext, call: dict[str, Any],
                       field: str) -> bool:
    properties = _schema(context, call["tool"])
    declaration = properties.get(field) if properties is not None else None
    items = declaration.get("items") if isinstance(declaration, dict) else None
    value = call["arguments"].get(field)
    return (isinstance(declaration, dict) and declaration.get("type") == "array"
            and isinstance(items, dict) and items.get("type") in {"integer", "number"}
            and isinstance(value, list) and len(value) == 3
            and all(type(item) in (int, float) and math.isfinite(item) for item in value))


def _event_text(store, event) -> str:
    return "\n".join(str(store.messages[index].to_dict().get("content") or "")
                     for index in event.source_indices)


def _request(context: RepairContext):
    user, _ = current_request(context.prepared._store)
    if user is None:
        return None
    return user, _event_text(context.prepared._store, user)


def _previous_user(store, current):
    users = [event for event in store.events if event.kind == "user"
             and max(event.source_indices) < min(current.source_indices)]
    return users[-1] if users else None


def _named_files(text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(1) for match in re.finditer(
        r"['\"]([^'\"\r\n]+\.[A-Za-z0-9]{1,12})['\"]", text,
    )))


def _referenced_file(store, request_event) -> str | None:
    text = _event_text(store, request_event)
    named = _named_files(text)
    if named:
        return named[0] if len(named) == 1 else None
    if not re.search(r"\b(?:previous|same|that)\s+file\b", text, re.IGNORECASE):
        return None
    earlier = [event for event in store.events if event.kind == "user"
               and max(event.source_indices) < min(request_event.source_indices)]
    for event in reversed(earlier):
        named = _named_files(_event_text(store, event))
        if named:
            return named[0] if len(named) == 1 else None
    return None


def _pending_calls(store):
    for event in store.events:
        if event.kind != "tool_event" or event.complete:
            continue
        for source_index in event.source_indices:
            message = store.messages[source_index].to_dict()
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or ():
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                arguments = _arguments(function.get("arguments"))
                if isinstance(name, str) and arguments is not None:
                    yield source_index, name, arguments


def _same_file(arguments: Any, file_name: str) -> bool:
    return (isinstance(arguments, dict)
            and any(isinstance(value, str) and value == file_name
                    for value in arguments.values()))


def _file_changed_after(context: RepairContext, records, file_name: str,
                        source_index: int) -> bool:
    if any(row.call_source_index > source_index and not row.failure_reported
           and row.tool.casefold() in _FILE_MUTATION_TOOLS
           and _same_file(row.arguments, file_name) for row in records):
        return True
    return any(index > source_index and name.casefold() in _FILE_MUTATION_TOOLS
               and _same_file(arguments, file_name)
               for index, name, arguments in _pending_calls(context.prepared._store))


def _producer(row: ObservedOperation, path: tuple[str | int, ...],
              value: Any) -> "ProducerProof":
    return ProducerProof(
        producer_event_id=row.event_id,
        producer_call_signature_id=row.call_signature_id,
        producer_result_source_index=row.result_source_index,
        producer_observation_version=row.observation_version,
        producer_result_path=path,
        observed_value=_proof_value(value),
    )


@dataclass(frozen=True)
class ProducerProof:
    producer_event_id: str
    producer_call_signature_id: str
    producer_result_source_index: int
    producer_observation_version: int
    producer_result_path: tuple[str | int, ...]
    observed_value: Any


@dataclass(frozen=True)
class RelationProof:
    kind: str
    request_event_id: str
    held_signature: str
    call_index: int
    tool: str
    field_path: tuple[str, ...]
    before: Any
    after: Any
    producers: tuple[ProducerProof, ...]
    related_user_event_id: str | None = None


@dataclass(frozen=True)
class RelationProposal:
    request_event_id: str
    held_signature: str
    prefix_source_index: int
    proofs: tuple[RelationProof, ...]
    version: str = VERSION
    proof_registry_version: str = PROOF_REGISTRY_VERSION

    def to_receipt(self) -> dict[str, Any]:
        return asdict(self)


class RelationRule(Protocol):
    kind: str

    def derive(self, context: RepairContext, request_event, request: str,
               held: tuple[dict[str, Any], ...], records: tuple[ObservedOperation, ...],
               held_signature: str) -> tuple[RelationProof, ...]: ...


class MeanFromWcRule:
    kind = "mean_from_wc_results"

    def derive(self, context, request_event, request, held, records, held_signature):
        if (_NEGATION.search(request)
                or not re.search(r"\b(?:average|mean)\b", request, re.IGNORECASE)
                or not re.search(r"\b(?:three|3)\b", request, re.IGNORECASE)
                or not re.search(r"\b(?:numeric(?:al)?|numbers?|values?|counts?)\b",
                                 request, re.IGNORECASE)
                or not re.search(r"\b(?:obtained|observed|returned|previous(?:ly)?|above)\b",
                                 request, re.IGNORECASE)):
            return ()
        previous = _previous_user(context.prepared._store, request_event)
        if previous is None:
            return ()
        previous_text = _event_text(context.prepared._store, previous)
        referenced_file = _referenced_file(context.prepared._store, previous)
        if (_NEGATION.search(previous_text)
                or not all(re.search(rf"\b{term}\b", previous_text, re.IGNORECASE)
                           for term in ("lines", "words", "characters"))
                or referenced_file is None):
            return ()
        candidates = [(index, call) for index, call in enumerate(held)
                      if call["tool"] == "mean"
                      and _number_array_slot(context, call, "numbers")]
        if len(candidates) != 1:
            return ()
        lower = max(previous.source_indices)
        upper = min(request_event.source_indices)
        grouped: dict[str, dict[str, list[tuple[ObservedOperation, Any]]]] = {}
        for row in records:
            if (row.tool != "wc" or row.failure_reported
                    or not (lower < row.call_source_index < upper)
                    or not isinstance(row.arguments, dict)):
                continue
            file_name = row.arguments.get("file_name")
            mode = row.arguments.get("mode")
            result = row.observed_result
            if (not isinstance(file_name, str) or not file_name or mode not in _WC_TYPES
                    or not isinstance(result, dict)
                    or result.get("type") != _WC_TYPES[mode]):
                continue
            count = result.get("count")
            if type(count) not in (int, float) or not math.isfinite(count):
                continue
            grouped.setdefault(file_name, {}).setdefault(mode, []).append((row, count))
        if set(grouped) != {referenced_file}:
            return ()
        file_name = referenced_file
        modes = grouped[file_name]
        if (set(modes) != set(_WC_TYPES)
                or any(len({_json(count) for _, count in modes[mode]}) != 1
                       for mode in _WC_TYPES)):
            return ()
        producer_rows = [item for mode in _WC_TYPES for item in modes[mode]]
        if _file_changed_after(context, records, file_name,
                               min(row.result_source_index for row, _ in producer_rows)):
            return ()
        if any(call["tool"].casefold() in _FILE_MUTATION_TOOLS
               and _same_file(call["arguments"], file_name) for call in held):
            return ()
        values = tuple(modes[mode][0][1] for mode in _WC_TYPES)
        index, call = candidates[0]
        before = call["arguments"]["numbers"]
        if sorted(before) == sorted(values):
            return ()
        producers = tuple(
            _producer(row, ("count",), count)
            for mode in _WC_TYPES for row, count in modes[mode]
        )
        return (RelationProof(
            self.kind, request_event.event_id, held_signature, index, "mean",
            ("numbers",), _proof_value(before), _proof_value(values), producers,
            related_user_event_id=previous.event_id,
        ),)


def _requested_message_id(request: str) -> int | None:
    if _NEGATION.search(request):
        return None
    matches = re.findall(
        r"\b(?:delete|remove)\b.{0,60}?\bmessage\s+(?:with\s+)?id\s*"
        r"(?:is\s+|[:=#]\s*)?(\d+)\b",
        request,
        re.IGNORECASE,
    )
    values = {int(value) for value in matches}
    return next(iter(values)) if len(values) == 1 else None


def _successful_send(row: ObservedOperation, message_id: int | None = None) -> bool:
    result = row.observed_result
    if not isinstance(result, dict) or not isinstance(result.get("message_id"), dict):
        return False
    found = result["message_id"].get("new_id")
    return (row.tool == "send_message" and not row.failure_reported
            and isinstance(row.arguments, dict)
            and isinstance(row.arguments.get("receiver_id"), str)
            and result.get("sent_status") is True
            and type(found) is int
            and (message_id is None or found == message_id))


def _later_message_change(row: ObservedOperation, receiver: str) -> bool:
    if (row.tool.casefold() not in ({"send_message"} | _MESSAGE_MUTATION_TOOLS)
            or not isinstance(row.arguments, dict)
            or row.arguments.get("receiver_id") != receiver
            or row.failure_reported):
        return False
    if isinstance(row.observed_result, dict) and any(
            row.observed_result.get(field) is False
            for field in ("sent_status", "deleted_status", "edited_status", "updated_status")
            if field in row.observed_result):
        return False
    return True


class MessageReceiptRule:
    kind = "delete_receiver_from_message_receipt"

    def derive(self, context, request_event, request, held, records, held_signature):
        message_id = _requested_message_id(request)
        if message_id is None:
            return ()
        candidates = [(index, call) for index, call in enumerate(held)
                      if call["tool"] == "delete_message"]
        if len(candidates) != 1:
            return ()
        index, call = candidates[0]
        properties = _schema(context, "delete_message")
        if (properties is None or "message_id" in properties
                or "message_id" in call["arguments"]):
            return ()
        sources = [row for row in records if _successful_send(row, message_id)]
        if len(sources) != 1:
            return ()
        source = sources[0]
        receiver = source.arguments["receiver_id"]
        if not _string_slot(context, call, "receiver_id", receiver):
            return ()
        before = call["arguments"]["receiver_id"]
        if before == receiver:
            return ()
        if any(other_index != index
               and other["tool"].casefold() in ({"send_message"} | _MESSAGE_MUTATION_TOOLS)
               and other["arguments"].get("receiver_id") == receiver
               for other_index, other in enumerate(held)):
            return ()
        if any(row.result_source_index > source.result_source_index
               and _later_message_change(row, receiver) for row in records):
            return ()
        if any(call_index > source.result_source_index
               and name.casefold() in ({"send_message"} | _MESSAGE_MUTATION_TOOLS)
               and arguments.get("receiver_id") == receiver
               for call_index, name, arguments in _pending_calls(context.prepared._store)):
            return ()
        return (RelationProof(
            self.kind, request_event.event_id, held_signature, index, "delete_message",
            ("receiver_id",), before, receiver,
            (_producer(source, ("message_id", "new_id"), message_id),),
        ),)


def _card_list(row: ObservedOperation) -> dict[str, dict[str, Any]] | None:
    result = row.observed_result
    cards = result.get("credit_card_list") if isinstance(result, dict) else None
    if (row.tool != "get_all_credit_cards" or row.failure_reported
            or not isinstance(cards, dict) or not cards):
        return None
    if not all(isinstance(key, str) and key and isinstance(value, dict)
               and isinstance(value.get("card_number"), str)
               for key, value in cards.items()):
        return None
    return cards


def _authorized_card_role(request: str, key: str) -> bool:
    if _NEGATION.search(request) or not re.search(
            r"\b(?:book|reserve|ticket)\w*\b", request, re.IGNORECASE):
        return False
    match = re.fullmatch(r"([a-z]+)_card", key.casefold())
    if match is None or match.group(1) not in _CARD_ROLES:
        return False
    role = re.escape(match.group(1))
    return bool(re.search(rf"\b(?:my\s+)?{role}\s+(?:credit\s+)?card\b",
                          request, re.IGNORECASE))


class CardKeyRule:
    kind = "card_key_from_card_number"

    def derive(self, context, request_event, request, held, records, held_signature):
        candidates = [(index, call) for index, call in enumerate(held)
                      if call["tool"] == "book_flight"
                      and isinstance(call["arguments"].get("card_id"), str)]
        if len(candidates) != 1:
            return ()
        index, call = candidates[0]
        snapshots = [(row, cards) for row in records
                     if (cards := _card_list(row)) is not None]
        if not snapshots:
            return ()
        source, cards = snapshots[-1]
        before = call["arguments"]["card_id"]
        if before in cards:
            return ()
        matches = [key for key, details in cards.items()
                   if details["card_number"] == before]
        if len(matches) != 1:
            return ()
        after = matches[0]
        if (not _authorized_card_role(request, after)
                or not _string_slot(context, call, "card_id", after)):
            return ()
        if any(other_index != index and _CARD_MUTATION.match(other["tool"])
               for other_index, other in enumerate(held)):
            return ()
        if any(row.call_source_index > source.result_source_index
               and not row.failure_reported and _CARD_MUTATION.match(row.tool)
               for row in records):
            return ()
        if any(call_index > source.result_source_index and _CARD_MUTATION.match(name)
               for call_index, name, _ in _pending_calls(context.prepared._store)):
            return ()
        return (RelationProof(
            self.kind, request_event.event_id, held_signature, index, "book_flight",
            ("card_id",), before, after,
            (_producer(source, ("credit_card_list", after, "card_number"), before),),
        ),)


RULES: tuple[RelationRule, ...] = (
    MeanFromWcRule(), MessageReceiptRule(), CardKeyRule(),
)


class Policy:
    def propose(self, context: RepairContext) -> RelationProposal | None:
        if context.parse_error is not None:
            return None
        held = _calls(context.draft_tool_calls)
        request = _request(context)
        if not held or request is None:
            return None
        event, text = request
        signature = _signature(held)
        records = operation_records(context.prepared._store)
        proofs = tuple(proof for rule in RULES for proof in rule.derive(
            context, event, text, held, records, signature))
        fields = [(proof.call_index, proof.field_path) for proof in proofs]
        if not proofs or len(set(fields)) != len(fields):
            return None
        return RelationProposal(
            event.event_id, signature, len(context.prepared._store.messages) - 1, proofs,
        )

    def _fresh(self, context: RepairContext, proposal: RelationProposal) -> bool:
        return (isinstance(proposal, RelationProposal)
                and proposal.version == VERSION
                and proposal.proof_registry_version == PROOF_REGISTRY_VERSION
                and self.propose(context) == proposal)

    def validate(self, context: RepairContext, proposal: RelationProposal,
                 candidate_calls) -> GuardVerdict:
        if not self._fresh(context, proposal):
            return GuardVerdict(False, "relation_proof_stale_or_tampered")
        held = _calls(context.draft_tool_calls)
        candidate = _calls(candidate_calls)
        if held is None or candidate is None or len(held) != len(candidate):
            return GuardVerdict(False, "relation_call_structure_changed")
        if _transport_shape(candidate_calls) != _transport_shape(context.draft_tool_calls):
            return GuardVerdict(False, "relation_transport_changed")
        expected = copy.deepcopy(held)
        for proof in proposal.proofs:
            if proof.call_index >= len(expected) or len(proof.field_path) != 1:
                return GuardVerdict(False, "relation_proof_invalid_path")
            call = expected[proof.call_index]
            field = proof.field_path[0]
            if (call["tool"] != proof.tool
                    or _proof_value(call["arguments"].get(field)) != proof.before):
                return GuardVerdict(False, "relation_proof_source_changed")
            call["arguments"][field] = _runtime_value(proof.after)
        if candidate != tuple(expected):
            return GuardVerdict(False, "relation_unproved_field_change")
        return GuardVerdict(True, "verified_relations_preserved")

    def apply(self, context: RepairContext, proposal: RelationProposal,
              candidate_calls) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
        original = tuple(copy.deepcopy(candidate_calls or ()))
        receipt = {
            "version": VERSION,
            "proof_registry_version": PROOF_REGISTRY_VERSION,
            "status": "refused",
            "proofs": proposal.to_receipt().get("proofs", [])
            if isinstance(proposal, RelationProposal) else [],
        }
        if not self._fresh(context, proposal):
            return original, {**receipt, "reason": "relation_proof_stale_or_tampered"}
        held = _calls(context.draft_tool_calls)
        if (held is None or _calls(original) != held
                or _transport_shape(original) != _transport_shape(context.draft_tool_calls)):
            return original, {**receipt, "reason": "relation_held_call_changed"}
        corrected = copy.deepcopy(original)
        for index, call in enumerate(corrected):
            changes = [proof for proof in proposal.proofs if proof.call_index == index]
            if not changes:
                continue
            function = call["function"]
            arguments = copy.deepcopy(held[index]["arguments"])
            for proof in changes:
                arguments[proof.field_path[0]] = _runtime_value(proof.after)
            function["arguments"] = (_json(arguments)
                                     if isinstance(function["arguments"], str)
                                     else arguments)
        verdict = self.validate(context, proposal, corrected)
        if not verdict.accepted:
            return original, {**receipt, "reason": verdict.reason}
        return tuple(corrected), {
            **receipt,
            "status": "applied",
            "reason": verdict.reason,
            "request_event_id": proposal.request_event_id,
            "prefix_source_index": proposal.prefix_source_index,
            "held_signature": proposal.held_signature,
        }


__all__ = [
    "Policy", "ProducerProof", "RelationProof", "RelationProposal", "RelationRule",
    "RULES", "VERSION", "PROOF_REGISTRY_VERSION", "PROOF_KINDS",
]
