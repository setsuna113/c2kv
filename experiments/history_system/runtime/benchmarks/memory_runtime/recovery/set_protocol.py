"""Observable context and finite evidence actions for recovery protocol v2."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import re
from dataclasses import replace

SET_PROTOCOL = "evidence_sets_v1"
PROPOSAL_PROTOCOL = "evidence_set_proposals_h0_v1"


def canonical_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def context_from_prepared(prepared, draft_tool_calls, draft_text, parse_error=None):
    """Read only the observed archive, current exact view, and held draft."""
    from history_memory.packing import visible_message

    store = prepared._store
    users = [event for event in store.events if event.kind == "user"]
    user_messages = [visible_message(message) for message in store.event_messages(users[-1].event_id)] if users else []
    goal = "\n".join(str(message.get("content", "")) for message in user_messages)
    completed = [event for event in store.events if event.kind == "tool_event" and event.complete]
    latest = [visible_message(message) for message in store.event_messages(completed[-1].event_id)] if completed else []
    raw = [visible_message(store.messages[index]) for index in prepared.memory.raw_source_indices]
    # Previously appended spans are exact visible evidence, unlike gist.
    visible_units = list(getattr(prepared, "_gp_visible", ()))
    raw.extend({"role": "user", "content": unit.text, "source_id": unit.unit_id} for unit in visible_units)
    shadow = getattr(prepared, "_shadow_features", None) or {}
    prefill = shadow.get("prefill") or {}
    captured = prefill.get("status") == "captured"
    logprobs = list(getattr(prepared, "_set_draft_logprobs", ()))
    return {
        "schema": "recovery-selection-context-v1", "session_id": store.session_id,
        "decision_key": prepared.metadata["decision_key"], "goal": goal,
        "last_action_observation": latest, "raw_visible": raw,
        "raw_source_ids": list(prepared.memory.view.raw_event_ids) + [unit.unit_id for unit in visible_units],
        "prefill_hidden": copy.deepcopy(prefill.get("hidden")) if captured else None,
        "prefill_contract": {"layer": prefill.get("layer"), "readout": prefill.get("readout"),
            "position_kind": (prefill.get("position") or {}).get("kind"),
            "bindings": copy.deepcopy(shadow.get("bindings") or {})},
        "draft_logprobs": logprobs, "draft_text": draft_text,
        "draft_tool_calls": copy.deepcopy(list(draft_tool_calls)),
        "parse_ok": parse_error is None, "is_stop": parse_error is None and not draft_tool_calls,
    }


def typed_parameters(calls):
    """Extract distinct string/number leaves; bool and null are not parameters."""
    values = set()
    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            values.add(("string", value))
        elif type(value) in (int, float) and math.isfinite(value):
            values.add(("number", value))
    for call in calls:
        arguments = (call.get("function") or {}).get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (ValueError, TypeError):
                continue
        visit(arguments)
    return values


def supported_parameters(parameters, text):
    """Match typed JSON leaves, with bounded literal matches in ordinary text."""
    found = set()
    def collect(item):
        if isinstance(item, dict):
            for key, child in item.items():
                if key in {"content", "arguments"} and isinstance(child, str):
                    found.update(supported_parameters(parameters, child))
                else:
                    collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)
        else:
            kind = "string" if isinstance(item, str) else "number" if type(item) in (int, float) else None
            if kind and (kind, item) in parameters:
                found.add((kind, item))
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        pass
    else:
        collect(value)
        return found
    # Evidence can contain ordinary text followed by serialized messages.
    # Parse those containers first so JSON keys/quoted numbers do not become
    # untyped matches; search the remaining ordinary text independently.
    decoder, cursor, plain_start, plain = json.JSONDecoder(), 0, 0, []
    while cursor < len(text):
        if text[cursor] not in "{[":
            cursor += 1
            continue
        try:
            value, end = decoder.raw_decode(text, cursor)
        except ValueError:
            cursor += 1
            continue
        plain.append(text[plain_start:cursor])
        collect(value)
        cursor = plain_start = end
    plain.append(text[plain_start:])
    for part in plain:
        for kind, value in parameters:
            if kind == "string":
                if value and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", part):
                    found.add((kind, value))
            else:
                pattern = r'(?<![\w.\"\'])' + re.escape(str(value)) + r'(?![\w.\"\'])'
                if re.search(pattern, part):
                    found.add((kind, value))
    return found


def spans_overlap(left, right):
    return any(a.event_id == b.event_id and a.source_index == b.source_index
        and a.container_path == b.container_path
        and max(a.char_start, b.char_start) < min(a.char_end, b.char_end)
        for a in left.provenance for b in right.provenance)


def uncovered_units(unit, store, tokenizer, raw_source_indices, visible_units):
    """Subtract only exact source coverage, never inferred gist contents."""
    from .evidence_units import _make_unit, _span_container_text, _sha256

    raw = set(raw_source_indices)
    covering = [span for other in visible_units for span in other.provenance]
    changed = False
    remaining = []
    for span in unit.provenance:
        if span.source_index in raw:
            changed = True
            continue
        intervals = [(span.char_start, span.char_end)]
        for covered in covering:
            if (span.event_id, span.source_index, span.container_path) != (
                    covered.event_id, covered.source_index, covered.container_path):
                continue
            updated = []
            for start, end in intervals:
                if covered.char_end <= start or covered.char_start >= end:
                    updated.append((start, end))
                else:
                    changed = True
                    if start < covered.char_start:
                        updated.append((start, covered.char_start))
                    if covered.char_end < end:
                        updated.append((covered.char_end, end))
            intervals = updated
        source = _span_container_text(store, span)
        for start, end in intervals:
            text = source[start:end]
            remaining.append((replace(span, char_start=start, char_end=end, sha256=_sha256(text)), text))
    if not changed:
        return [unit]
    return [_make_unit(source_type=unit.source_type, event_id=unit.event_id, text=text,
        tokenizer=tokenizer, provenance=(span,), association_id=unit.association_id,
        metadata={**unit.metadata, "exact_uncovered_parent_id": unit.unit_id}) for span, text in remaining]


def build_legal_sets(candidates, admissible, *, maximum=4):
    """Enumerate IDs only: empty, singles, pairs, and ranked top three/four."""
    units = [row["unit"] for row in candidates]
    proposed = [()]
    proposed.extend((index,) for index in range(len(units)))
    if maximum >= 2:
        proposed.extend(itertools.combinations(range(len(units)), 2))
    for size in (3, 4):
        if maximum >= size and len(units) >= size:
            proposed.append(tuple(range(size)))
    actions, rejected = [()], []
    for indices in proposed[1:]:
        chosen = [units[index] for index in indices]
        ids = tuple(unit.unit_id for unit in chosen)
        if any(spans_overlap(a, b) for a, b in itertools.combinations(chosen, 2)):
            rejected.append({"selected_ids": list(ids), "reason": "overlapping_exact_source_spans"})
            continue
        fits, receipt = admissible(chosen)
        if fits:
            actions.append(ids)
        else:
            rejected.append({"selected_ids": list(ids), "reason": "set_not_admitted", "admission": receipt})
    return actions, rejected


def choose_parameter_source(context, candidates, actions):
    parameters = typed_parameters(context["draft_tool_calls"]) if context["parse_ok"] else set()
    visible = context["goal"] + "\n" + canonical_text(context["raw_visible"])
    unsupported = parameters - supported_parameters(parameters, visible)
    by_id = {row["unit_id"]: row for row in candidates}
    ranked = []
    for action in actions:
        if not action:
            continue
        rows = [by_id[identifier] for identifier in action]
        coverage = set().union(*(supported_parameters(unsupported, row["text"]) for row in rows))
        ranked.append((-len(coverage), len({row["event_id"] for row in rows}),
            sum(row["token_count"] for row in rows), len(action), tuple(action)))
    best = min(ranked) if ranked else None
    return (best[-1], -best[0]) if best and best[0] < 0 else ((), 0)


def build_proposal_catalog(context, candidates, actions):
    """Build the frozen H0 proposal catalog from the full legal action catalog.

    ``Slex`` is exactly C0's first feasible ranked singleton. ``Ssrc`` is
    exactly the existing parameter-source choice over the *full* legal action
    catalog.  Origin aliases are retained even when both rules name the same
    ordered evidence set, while the action passed to a selector is deduplicated.
    """
    candidate_ids = {
        row.get("unit_id") for row in candidates
        if isinstance(row, dict) and isinstance(row.get("unit_id"), str)
    }
    legal = []
    legal_by_set = {}
    discarded_actions = []
    for value in actions:
        if not isinstance(value, (list, tuple)):
            discarded_actions.append({"selected_ids": [], "reason": "action_not_a_sequence"})
            continue
        action = tuple(value)
        if any(
            not isinstance(identifier, str) or identifier not in candidate_ids
            for identifier in action
        ):
            discarded_actions.append({
                "selected_ids": list(action), "reason": "unknown_candidate_id"
            })
            continue
        if len(set(action)) != len(action):
            discarded_actions.append({
                "selected_ids": list(action), "reason": "repeated_candidate_id"
            })
            continue
        action_set = frozenset(action)
        if action_set in legal_by_set:
            discarded_actions.append({
                "selected_ids": list(action), "reason": "duplicate_candidate_set",
                "canonical_selected_ids": list(legal_by_set[action_set]),
            })
            continue
        legal_by_set[action_set] = action
        legal.append(action)
    if frozenset() not in legal_by_set:
        legal.insert(0, ())
        legal_by_set[frozenset()] = ()

    slex = next((action for action in legal if len(action) == 1), ())
    ssrc, source_coverage = choose_parameter_source(context, candidates, legal)
    if frozenset(ssrc) not in legal_by_set:
        ssrc = ()
        source_coverage = 0

    requested = (("empty", ()), ("Slex", slex), ("Ssrc", tuple(ssrc)))
    proposal_actions = []
    aliases = []
    by_action_set = {}
    proposals = {}
    for origin, action in requested:
        available = origin == "empty" or bool(action)
        proposals[origin] = {
            "selected_ids": list(action),
            "available": available,
            "legal": frozenset(action) in legal_by_set,
        }
        if origin == "Ssrc":
            proposals[origin]["unsupported_parameter_coverage_count"] = source_coverage
        if not available or frozenset(action) not in legal_by_set:
            continue
        action_set = frozenset(action)
        if action_set not in by_action_set:
            by_action_set[action_set] = len(proposal_actions)
            canonical_action = legal_by_set[action_set]
            proposal_actions.append(canonical_action)
            aliases.append({
                "selected_ids": list(canonical_action),
                "origins": [origin],
                "origin_selected_ids": {origin: list(action)},
            })
        else:
            alias = aliases[by_action_set[action_set]]
            alias["origins"].append(origin)
            alias["origin_selected_ids"][origin] = list(action)
        proposals[origin]["canonical_selected_ids"] = list(
            proposal_actions[by_action_set[action_set]]
        )

    distinct = bool(slex and ssrc and slex != ssrc)
    fallback = {
        "applied": not bool(ssrc) and bool(slex),
        "from": "Ssrc",
        "to": "Slex" if not ssrc and slex else None,
        "reason": "source_proposal_empty" if not ssrc and slex else None,
    }
    receipt = {
        "schema": "recovery-proposal-catalog-v1",
        "protocol": PROPOSAL_PROTOCOL,
        "proposals": proposals,
        "available": {origin: row["available"] for origin, row in proposals.items()},
        "distinct": distinct,
        "fallback": fallback,
        "deduplicated_actions": [list(action) for action in proposal_actions],
        "origin_aliases": aliases,
        "discarded_actions": discarded_actions,
    }
    return tuple(proposal_actions), receipt


def context_digest(context):
    return hashlib.sha256(canonical_text(context).encode("utf-8")).hexdigest()
