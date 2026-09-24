"""Scoped, source-faithful native KV protection requests.

The allocator proposes independent source units. Only the serving engine can
admit their resident KV, and only the selected action commits a lease.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import replace

from ..policy import _anchors, _event_text, _is_explicit_revision, _tokens
from .allocator import PersistentHistoryAllocator


_SENTENCE_END = re.compile(r"(?<=[.!?。！？])\s+|\r?\n")
_UNBOUND_START = re.compile(r"^(?:it|its|this|that|they|their|these|those|he|she|the status)\b",
                            re.IGNORECASE)
_RECORD_KEYS = frozenset({"id", "name", "key", "path", "url", "uri", "sku", "code",
                          "account_id", "project_id", "record_id", "item_id"})


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _object_spans(content):
    """Return original JSON object slices that retain their own record context."""
    stack = []
    quote = escape = False
    candidates = []
    for index, char in enumerate(content):
        if quote:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                quote = False
            continue
        if char == '"':
            quote = True
        elif char in "[{":
            stack.append((char, index))
        elif char in "]}" and stack:
            opener, start = stack.pop()
            if (opener, char) != ("[", "]") and (opener, char) != ("{", "}"):
                return ()
            if opener == "{" and (not stack or stack[-1][0] == "[") and len(stack) <= 2:
                candidate = content[start:index + 1]
                try:
                    parsed = json.loads(candidate)
                except ValueError:
                    continue
                if (isinstance(parsed, dict) and len(parsed) >= 2
                        and _RECORD_KEYS.intersection(str(key).casefold() for key in parsed)):
                    candidates.append(candidate)
    if stack or quote:
        return ()
    # A whole root object is useful only when there are no smaller records.
    smaller = [value for value in candidates if len(value) < len(content.strip())]
    return tuple(dict.fromkeys(smaller or candidates))


def _matching_fragments(content, query_tokens, anchors):
    if not isinstance(content, str) or not content.strip():
        return (), None
    objects = _object_spans(content)
    if objects:
        scores = [_match_score(value, query_tokens, anchors) for value in objects]
        best = max(scores, default=0)
        matches = [value for value, score in zip(objects, scores) if score == best and best]
        if matches:
            return tuple(matches), "json_object"
    pieces = [piece for piece in _SENTENCE_END.split(content) if piece.strip()]
    if len(pieces) > 1:
        scores = [_match_score(piece, query_tokens, anchors)
                  if len(_tokens(piece)) >= 2 and not _UNBOUND_START.match(piece.strip())
                  else 0 for piece in pieces]
        best = max(scores, default=0)
        matches = [piece for piece, score in zip(pieces, scores) if score == best and best]
        if matches:
            return tuple(dict.fromkeys(matches)), "sentence"
    return (), None


def _match_score(text, query_tokens, anchors):
    lower = text.casefold()
    return 10 * sum(anchor in lower for anchor in anchors) + len(_tokens(text) & query_tokens)


def _unit(event_id, source_indices, fragments, kind, complete_event):
    canonical = json.dumps([event_id, list(source_indices), fragments, kind],
                           ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"unit_id": "pu:" + _digest(canonical), "event_id": event_id,
            "source_indices": list(source_indices), "fragments": fragments,
            "kind": kind, "complete_event": complete_event}


class NativeProtectionV2Allocator(PersistentHistoryAllocator):
    """Offer source units and retain only engine-admitted units in one user scope."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._protection_leases = {}
        self._active_plans = {}

    def _try_measure(self, store, tools, raw_ids, mandatory_ids, gist_ids,
                     eligible_event_ids, common_tokens, max_new_tokens, *, derived_messages=()):
        measure = super()._try_measure(
            store, tools, raw_ids, mandatory_ids, gist_ids, eligible_event_ids,
            common_tokens, max_new_tokens, derived_messages=derived_messages)
        plan = self._active_plans.get(store.session_id)
        if measure is None or plan is None:
            return measure
        # The first preparation has no recovered event. Repacking a held draft
        # must carry its protection request into the alternative generation.
        recovered = set(raw_ids) - set(mandatory_ids)
        if not recovered and not derived_messages:
            return measure
        from dataclasses import replace as dataclass_replace
        return dataclass_replace(measure, memory=dataclass_replace(measure.memory,
            protection_units=copy.deepcopy(plan.protection_units),
            protection_scope_id=plan.protection_scope_id,
            protection_source_events=copy.deepcopy(plan.protection_source_events),
            protection_event_ids=plan.protection_event_ids,
            protection_source_indices=plan.protection_source_indices))

    def _prepare_view(self, store, tools, *, ratio, max_new_tokens, decision_key, decision_index):
        prepared = super()._prepare_view(
            store, tools, ratio=ratio, max_new_tokens=max_new_tokens,
            decision_key=decision_key, decision_index=decision_index)
        memory = prepared.memory
        users = [event for event in store.events if event.kind == "user"]
        current_user = users[-1] if users else None
        current_text = _event_text(store, current_user) if current_user else ""
        scope_id = ("ps:" + _digest(json.dumps([current_user.event_id, current_text],
                    ensure_ascii=False, separators=(",", ":"))) if current_user else "")
        historical = set(range(memory.history_start_message_count, memory.history_message_count))
        complete_tools = [event for event in store.events
                          if event.kind == "tool_event" and event.complete]
        requested = {event.event_id for event in store.events
                     if not event.complete or event.kind == "instruction"}
        if current_user:
            requested.add(current_user.event_id)
        if complete_tools:
            requested.add(complete_tools[-1].event_id)
        if self.benchmark == "acon_appworld" and users:
            requested.add(users[0].event_id)
        lexical, _, lexical_receipt = self._lexical_request(
            store, tools, requested, recent_tool_event_visible=True)
        requested.update(lexical.source_ids)

        old_scope, leased = self._protection_leases.get(store.session_id, (None, ()))
        if old_scope == scope_id:
            requested.update(unit["event_id"] for unit in leased)
        else:
            leased = ()
            self._protection_leases.pop(store.session_id, None)
        cancelled = set(prepared.metadata.get("revision_cancelled_event_ids") or ())
        if current_user and _is_explicit_revision(store, current_user):
            # A new request/revision starts a new scope. Do not carry old units.
            leased = ()
            self._protection_leases.pop(store.session_id, None)
        if cancelled and leased:
            leased = tuple(unit for unit in leased if unit["event_id"] not in cancelled)
            self._protection_leases[store.session_id] = (scope_id, leased)
        requested.difference_update(cancelled)
        query_tokens, anchors = _tokens(current_text), _anchors(current_text)
        units = []
        leased_by_id = {unit["unit_id"]: unit for unit in leased}
        priority = [*(unit["event_id"] for unit in leased),
                    *([current_user.event_id] if current_user else []),
                    *([complete_tools[-1].event_id] if complete_tools else []),
                    *lexical.source_ids,
                    *(event.event_id for event in reversed(store.events)
                      if event.event_id in requested)]
        for event_id in dict.fromkeys(priority):
            if event_id not in requested:
                continue
            event = store.event(event_id)
            source_indices = [index for index in event.source_indices
                              if index in historical and store.messages[index].role != "system"]
            if not source_indices or len(source_indices) != len(event.source_indices):
                continue
            old = [unit for unit in leased_by_id.values() if unit["event_id"] == event.event_id
                   and set(unit["source_indices"]) <= set(source_indices)]
            if old:
                units.extend(copy.deepcopy(old))
                continue
            fragments = []
            fragment_kinds = set()
            for index in source_indices:
                content = store.messages[index].to_dict().get("content")
                matches, kind = _matching_fragments(content, query_tokens, anchors)
                if matches:
                    fragment_kinds.add(kind)
                    fragments.extend({"source_index": index, "text": text} for text in matches)
            if fragments and fragment_kinds == {"json_object"}:
                for fragment in fragments:
                    units.append(_unit(event.event_id, [fragment["source_index"]],
                                       [fragment], "json_object", False))
            elif fragments and fragment_kinds == {"sentence"}:
                for fragment in fragments:
                    units.append(_unit(event.event_id, [fragment["source_index"]],
                                       [fragment], "sentence", False))
            else:
                units.append(_unit(event.event_id, source_indices, [], "event", True))
        indices = tuple(sorted({index for unit in units for index in unit["source_indices"]}))
        event_ids = tuple(dict.fromkeys(unit["event_id"] for unit in units))
        source_events = tuple({"event_id": event_id,
                               "source_indices": list(store.event(event_id).source_indices)}
                              for event_id in event_ids)
        prepared.memory = replace(memory, protection_units=tuple(copy.deepcopy(units)),
                                  protection_scope_id=scope_id,
                                  protection_source_events=source_events,
                                  protection_event_ids=event_ids,
                                  protection_source_indices=indices)
        self._active_plans[store.session_id] = prepared.memory
        prepared.metadata["native_protection_request"] = {
            "schema": "racer-native-protection-v2", "enabled": True,
            "scope_id": scope_id, "unit_ids": [unit["unit_id"] for unit in units],
            "event_ids": list(event_ids), "source_indices": list(indices),
            "lexical": lexical_receipt, "status": "requires_engine_admission",
            "input_rewritten": False,
            "original_history_budget_tokens": memory.history_budget_tokens,
        }
        return prepared

    def observe_native_protection(self, prepared, *, memory, stats):
        """Expose only proved full-event visibility to source recovery."""
        report = stats.get("kv_memory_report") if isinstance(stats, dict) else None
        receipt = report.get("racer_native_protection") if isinstance(report, dict) else None
        if not isinstance(receipt, dict) or receipt.get("schema") != "racer-native-protection-v2":
            raise ValueError("Native protection v2 requires an engine receipt")
        if receipt.get("scope_id") != memory.protection_scope_id:
            raise ValueError("Native protection receipt scope differs from request")
        if set(receipt.get("unit_ids") or ()) != {unit["unit_id"] for unit in memory.protection_units}:
            raise ValueError("Native protection receipt unit IDs differ from request")
        if set(receipt.get("event_ids") or ()) != set(memory.protection_event_ids):
            raise ValueError("Native protection receipt events differ from request")
        if receipt.get("decision_id") != prepared.metadata.get("decision_key"):
            raise ValueError("Native protection receipt decision differs from request")
        full = set()
        for entry in receipt.get("event_coverage", ()):
            if not isinstance(entry, dict) or entry.get("status") != "full":
                continue
            rows = entry.get("total_rows")
            if (type(rows) is not int or rows <= 0
                    or type(entry.get("full_rows")) is not int
                    or entry["full_rows"] != rows):
                raise ValueError("Native protection full event coverage requires every model row")
            full.add(entry.get("event_id"))
        full.intersection_update(memory.protection_event_ids)
        prepared.metadata["native_protection_full_event_ids"] = sorted(full)
        prepared.metadata["native_protection_receipt"] = copy.deepcopy(receipt)

    def commit_native_protection(self, prepared, *, memory, stats):
        """Save only admitted units from the selected, successfully resolved view."""
        if not memory.protection_scope_id:
            self._protection_leases.pop(prepared.metadata["session_id"], None)
            return
        self.observe_native_protection(prepared, memory=memory, stats=stats)
        receipt = prepared.metadata["native_protection_receipt"]
        admitted = {entry.get("unit_id") for entry in receipt.get("units", ())
                    if isinstance(entry, dict) and entry.get("total_rows", 0) > 0
                    and ((entry.get("status") == "admitted"
                          and entry.get("admitted_rows") == entry.get("total_rows"))
                         or (entry.get("status") == "retained"
                             and entry.get("coverage_status") == "full"
                             and entry.get("retained_rows") == entry.get("total_rows")))}
        units = tuple(copy.deepcopy(unit) for unit in memory.protection_units
                      if unit["unit_id"] in admitted)
        self._protection_leases[prepared.metadata["session_id"]] = (
            memory.protection_scope_id, units)

    def clear_native_protection(self):
        self._protection_leases.clear()
        self._active_plans.clear()
