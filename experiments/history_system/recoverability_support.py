"""Source-bound known-support packing for the recoverability diagnostic.

The annotation says which already observed bytes a researcher judged useful.
It is never rendered into the actor input and is not treated as an automatic
proof that those bytes semantically support a correct action.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence

from benchmarks.memory_runtime.recovery.evidence_units import (
    EvidenceUnit,
    SpanProvenance,
    _sha256,
    _span_container_text,
    _validate_unit,
    build_catalog,
    unit_is_covered,
)
from benchmarks.memory_runtime.recovery.set_protocol import uncovered_units


SCHEMA = "recoverability-support-plan-v1"
SEARCH_COMBINATION_LIMIT = 200_000


class SupportAnnotationError(ValueError):
    """A claimed source span is not exactly present in the observed prefix."""


class SupportSearchIncomplete(RuntimeError):
    """The bounded set search ended before representability was established."""


def _canonical(value) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _digest(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _path(value, *, label):
    if not isinstance(value, (list, tuple)) or any(
        isinstance(part, bool) or not isinstance(part, (str, int)) for part in value
    ):
        raise SupportAnnotationError(f"{label} must be a string/integer path")
    return tuple(value)


def _validate_annotations(store, annotations):
    if not isinstance(annotations, Sequence) or isinstance(annotations, (str, bytes)) or not annotations:
        raise SupportAnnotationError("support annotations must be a nonempty list")
    validated = []
    for index, raw in enumerate(annotations):
        if not isinstance(raw, Mapping):
            raise SupportAnnotationError(f"annotation[{index}] must be an object")
        if raw.get("reviewed") is not True:
            raise SupportAnnotationError(f"annotation[{index}] must be explicitly reviewed")
        reason = raw.get("support_reason")
        if not isinstance(reason, str) or not reason.strip():
            raise SupportAnnotationError(f"annotation[{index}] support_reason must be nonempty")
        quote = raw.get("quote")
        if not isinstance(quote, str) or not quote:
            raise SupportAnnotationError(f"annotation[{index}] quote must be nonempty")
        event_id, source_index = raw.get("event_id"), raw.get("source_index")
        if not isinstance(event_id, str) or not event_id:
            raise SupportAnnotationError(f"annotation[{index}] event_id must be nonempty")
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise SupportAnnotationError(f"annotation[{index}] source_index must be an integer")
        try:
            event = store.event(event_id)
        except KeyError as error:
            raise SupportAnnotationError(
                f"annotation[{index}] source event is outside the observed prefix"
            ) from error
        if not event.complete:
            raise SupportAnnotationError(f"annotation[{index}] source event is incomplete")
        if source_index not in event.source_indices or not 0 <= source_index < len(store.messages):
            raise SupportAnnotationError(f"annotation[{index}] source index is not in its event")
        char_range = raw.get("char_range")
        if (
            not isinstance(char_range, (list, tuple))
            or len(char_range) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in char_range)
        ):
            raise SupportAnnotationError(f"annotation[{index}] char_range must contain two integers")
        start, end = char_range
        span = SpanProvenance(
            event_id=event_id,
            source_index=source_index,
            container_path=_path(raw.get("container_path"), label=f"annotation[{index}] container_path"),
            field_path=_path(raw.get("field_path"), label=f"annotation[{index}] field_path"),
            char_start=start,
            char_end=end,
            sha256=raw.get("sha256"),
        )
        try:
            container = _span_container_text(store, span)
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise SupportAnnotationError(
                f"annotation[{index}] container path does not resolve"
            ) from error
        if not 0 <= start < end <= len(container):
            raise SupportAnnotationError(f"annotation[{index}] character range is invalid")
        source_quote = container[start:end]
        if quote != source_quote:
            raise SupportAnnotationError(f"annotation[{index}] quote differs from observed source")
        if not isinstance(span.sha256, str) or span.sha256 != _sha256(source_quote):
            raise SupportAnnotationError(f"annotation[{index}] hash differs from observed source")
        public = {
            **span.to_dict(),
            "quote_sha256": _sha256(quote),
            "support_reason_sha256": _sha256(reason),
            "reviewed": True,
        }
        validated.append((span, {**public, "annotation_sha256": _digest(dict(raw))}))
    return validated


def _matching(left, right):
    return (
        left.event_id,
        left.source_index,
        left.container_path,
    ) == (right.event_id, right.source_index, right.container_path)


def _subtract(span, covering, raw_indices):
    if span.source_index in raw_indices:
        return []
    intervals = [(span.char_start, span.char_end)]
    for other in covering:
        if not _matching(span, other):
            continue
        next_intervals = []
        for start, end in intervals:
            if other.char_end <= start or other.char_start >= end:
                next_intervals.append((start, end))
            else:
                if start < other.char_start:
                    next_intervals.append((start, other.char_start))
                if other.char_end < end:
                    next_intervals.append((other.char_end, end))
        intervals = next_intervals
    return intervals


def _candidate_pieces(controller, prepared, source_type):
    store = prepared._store
    eligible = set(prepared.metadata["eligible_extraction"]["eligible_event_ids"])
    cancelled = set(prepared.metadata.get("revision_cancelled_event_ids") or ())
    visible = list(prepared._gp_visible)
    by_id = {}
    catalog = build_catalog(store, controller.tokenizer, source_type)
    for unit in catalog:
        if unit.event_id not in eligible or unit.event_id in cancelled:
            continue
        for piece in uncovered_units(
            unit, store, controller.tokenizer, prepared.memory.raw_source_indices, visible
        ):
            prior = by_id.setdefault(piece.unit_id, piece)
            if prior != piece:
                raise ValueError("static archive produced conflicting unit identities")
    return catalog, list(by_id.values())


def _masks(candidates, remaining):
    atoms = []
    for annotation_index, span, start, end in remaining:
        boundaries = {start, end}
        for unit in candidates:
            for provenance in unit.provenance:
                if _matching(span, provenance):
                    left, right = max(start, provenance.char_start), min(end, provenance.char_end)
                    if left < right:
                        boundaries.update((left, right))
        ordered = sorted(boundaries)
        atoms.extend(
            (annotation_index, span, left, right)
            for left, right in zip(ordered, ordered[1:])
            if left < right
        )
    masks, relevant = [], []
    for unit in candidates:
        mask = 0
        for bit, (_, span, start, end) in enumerate(atoms):
            if any(
                _matching(span, provenance)
                and provenance.char_start <= start
                and end <= provenance.char_end
                for provenance in unit.provenance
            ):
                mask |= 1 << bit
        if mask:
            relevant.append(unit)
            masks.append(mask)
    return relevant, masks, (1 << len(atoms)) - 1


def _search(controller, prepared, candidates, remaining, maximum, *, probe_limit):
    candidates, masks, full = _masks(candidates, remaining)
    order = sorted(range(len(candidates)), key=lambda index: candidates[index].unit_id)
    candidates = [candidates[index] for index in order]
    masks = [masks[index] for index in order]
    checked = 0
    best = None
    best_admission = None
    active = list(prepared._gp_visible)
    for size in range(1, maximum + 1):
        covers = []
        for indices in itertools.combinations(range(len(candidates)), size):
            checked += 1
            if checked > probe_limit:
                raise SupportSearchIncomplete(
                    f"support pack search exceeded {probe_limit} combinations"
                )
            mask = 0
            for index in indices:
                mask |= masks[index]
            if mask == full:
                pack = tuple(candidates[index] for index in indices)
                covers.append(pack)
        covers.sort(key=lambda pack: (sum(unit.token_count for unit in pack), tuple(unit.unit_id for unit in pack)))
        if covers and best is None:
            best = covers[0]
        for pack in covers:
            measure, admission, _ = controller._append_measure(prepared, [*active, *pack])
            if best_admission is None:
                best_admission = admission
            if measure is not None:
                return list(pack), admission, checked, True, candidates
    return list(best or ()), best_admission, checked, False, candidates


def _held_candidate_ids(prepared):
    frozen = getattr(prepared, "_t02_selection", None)
    if not isinstance(frozen, tuple) or not frozen:
        return None
    rows = frozen[0]
    if not isinstance(rows, Sequence):
        return None
    ids = []
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("unit_id"), str):
            ids.append(row["unit_id"])
    return tuple(ids)


def _binding(controller, prepared, validated):
    store = prepared._store
    source_ids = list(dict.fromkeys(span.event_id for span, _ in validated))
    archive = {
        "session_id": store.session_id,
        "message_sha256": [_sha256(message.json_text) for message in store.messages],
        "events": [
            {"event_id": event.event_id, "kind": event.kind,
             "source_indices": list(event.source_indices), "complete": event.complete}
            for event in store.events
        ],
    }
    state = {
        "decision_key": prepared.metadata["decision_key"],
        "raw_source_indices": list(prepared.memory.raw_source_indices),
        "raw_event_ids": list(prepared.memory.view.raw_event_ids),
        "gist_event_ids": list(prepared.memory.view.gist_event_ids),
        "eligible_event_ids": list(prepared.metadata["eligible_extraction"]["eligible_event_ids"]),
        "cancelled_event_ids": list(prepared.metadata.get("revision_cancelled_event_ids") or ()),
        "visible_exact_unit_ids": [unit.unit_id for unit in prepared._gp_visible],
        "actual_history_bytes": prepared.metadata.get("actual_history_bytes"),
        "history_budget_bytes": controller._packer.policy_config.history_budget_bytes,
        "workspace_budget_bytes": controller._packer.policy_config.workspace_budget_bytes,
        "gp": {key: controller.gp.get(key) for key in
               ("U", "B", "P", "order", "K", "selector_max_units", "fallback_unit")},
    }
    sources = []
    for event_id in source_ids:
        event = store.event(event_id)
        sources.append({"event_id": event_id, "kind": event.kind,
            "source_indices": list(event.source_indices), "complete": event.complete,
            "source_message_sha256": [_sha256(store.messages[index].json_text)
                                      for index in event.source_indices]})
    return {"archive_sha256": _digest(archive), "state_sha256": _digest(state),
            "session_id": store.session_id, "decision_key": state["decision_key"],
            "source_events": sources, "state": state}


def plan_support(controller, prepared, annotations):
    """Return the smallest admitted static source pack and a JSON-safe receipt."""

    if controller.gp.get("B") != "source" or controller.gp.get("P") != "quoted":
        raise ValueError("known-support diagnostic requires the frozen B=source, P=quoted contract")
    key = (prepared._store.session_id, prepared.metadata["decision_key"])
    if controller._prepared.get(key) is not prepared:
        raise ValueError("prepared decision does not belong to this controller")
    validated = _validate_annotations(prepared._store, annotations)
    raw = set(prepared.memory.raw_source_indices)
    visible_spans = [span for unit in prepared._gp_visible for span in unit.provenance]
    remaining, coverage = [], []
    for annotation_index, (span, public) in enumerate(validated):
        intervals = _subtract(span, visible_spans, raw)
        state = "already_exact_visible" if not intervals else "requires_static_unit"
        coverage.append({"annotation_sha256": public["annotation_sha256"], "state": state,
                         "remaining_ranges": [list(interval) for interval in intervals]})
        remaining.extend((annotation_index, span, start, end) for start, end in intervals)
    maximum = min(controller.gp["K"], controller.gp.get("selector_max_units", controller.gp["K"]))
    receipt = {"schema": SCHEMA, "status": None, "diagnostic_only": True,
        "semantic_support": "reviewed_source_annotation_not_automatic_proof",
        "annotation_text_injected": False, "selector_or_model_called": False,
        "input_binding": _binding(controller, prepared, validated),
        "annotations": [public for _, public in validated], "coverage": coverage,
        "catalog": {"unit_type": controller.gp["U"], "fallback_unit": controller.gp.get("fallback_unit"),
                    "max_units": maximum, "search_combination_limit": SEARCH_COMBINATION_LIMIT},
        "candidate_ids": [], "selected_unit_ids": [], "admission": None}
    held = _held_candidate_ids(prepared)
    receipt["held_candidates"] = {"present": held is not None,
        "candidate_ids": list(held or ()), "selected_id_membership": []}
    if not remaining:
        receipt["status"] = "already_exact_visible"
        return [], receipt
    eligible = set(prepared.metadata["eligible_extraction"]["eligible_event_ids"])
    cancelled = set(prepared.metadata.get("revision_cancelled_event_ids") or ())
    ineligible = sorted({span.event_id for _, span, _, _ in remaining
                         if span.event_id not in eligible or span.event_id in cancelled
                         or prepared._store.event(span.event_id).kind == "instruction"})
    if ineligible:
        receipt["status"] = "source_not_eligible"
        receipt["ineligible_source_event_ids"] = ineligible
        return [], receipt

    base_catalog, base = _candidate_pieces(controller, prepared, controller.gp["U"])
    selected, admission, checked, admitted, relevant = _search(
        controller, prepared, base, remaining, maximum, probe_limit=SEARCH_COMBINATION_LIMIT
    )
    receipt["catalog"].update(archive_unit_count=len(base_catalog),
        relevant_base_candidate_ids=[unit.unit_id for unit in relevant],
        combinations_checked=checked, search_complete=True)

    if not admitted and controller.gp.get("fallback_unit"):
        failed_parents = []
        for parent in base_catalog:
            if parent.event_id not in eligible or parent.event_id in cancelled:
                continue
            pieces = uncovered_units(parent, prepared._store, controller.tokenizer,
                prepared.memory.raw_source_indices, list(prepared._gp_visible))
            if not pieces:
                continue
            if all(controller._append_measure(prepared, [*prepared._gp_visible, piece])[0] is None
                   for piece in pieces):
                failed_parents.append(parent)
        fallback_catalog, fallback_pieces = _candidate_pieces(
            controller, prepared, controller.gp["fallback_unit"]
        )
        allowed_fallback = [child for child in fallback_pieces
            if any(child.event_id == parent.event_id and unit_is_covered(child, (parent,))
                   for parent in failed_parents)]
        individually_admitted_base = [unit for unit in base
            if controller._append_measure(prepared, [*prepared._gp_visible, unit])[0] is not None]
        fallback_selected, fallback_admission, extra_checked, fallback_admitted, fallback_relevant = _search(
            controller, prepared, [*individually_admitted_base, *allowed_fallback], remaining,
            maximum, probe_limit=SEARCH_COMBINATION_LIMIT - checked
        )
        checked += extra_checked
        receipt["catalog"].update(fallback_archive_unit_count=len(fallback_catalog),
            fallback_parent_ids=[unit.unit_id for unit in failed_parents],
            relevant_fallback_candidate_ids=[unit.unit_id for unit in fallback_relevant],
            combinations_checked=checked)
        if fallback_admitted or (not selected and fallback_selected):
            selected, admission, admitted = fallback_selected, fallback_admission, fallback_admitted

    receipt["candidate_ids"] = list(dict.fromkeys(
        receipt["catalog"].get("relevant_base_candidate_ids", [])
        + receipt["catalog"].get("relevant_fallback_candidate_ids", [])))
    receipt["selected_unit_ids"] = [unit.unit_id for unit in selected]
    receipt["admission"] = admission
    held_set = set(held or ())
    receipt["held_candidates"]["selected_id_membership"] = [
        {"unit_id": unit.unit_id, "present": unit.unit_id in held_set} for unit in selected
    ]
    receipt["held_candidates"]["all_selected_present"] = (
        all(unit.unit_id in held_set for unit in selected) if held is not None else None
    )
    if admitted:
        receipt["status"] = "admitted"
    elif selected:
        receipt["status"] = "selected_support_set_not_admitted_under_b0"
    else:
        receipt["status"] = "support_not_representable_in_static_units"
    selected_ids = {unit.unit_id for unit in selected}
    for row, (span, _) in zip(coverage, validated, strict=True):
        if row["state"] == "already_exact_visible":
            row["covering_selected_unit_ids"] = []
        else:
            row["covering_selected_unit_ids"] = [unit.unit_id for unit in selected
                if unit.unit_id in selected_ids and any(_matching(span, item) and
                    item.char_start < span.char_end and span.char_start < item.char_end
                    for item in unit.provenance)]
    return selected, receipt


def commit_support(controller, prepared, selected):
    """Commit a planned pack without invoking retrieval, a selector, or a model."""

    if not isinstance(selected, Sequence) or any(not isinstance(unit, EvidenceUnit) for unit in selected):
        raise TypeError("selected must be a sequence of EvidenceUnit values")
    key = (prepared._store.session_id, prepared.metadata["decision_key"])
    if controller._prepared.get(key) is not prepared:
        raise ValueError("prepared decision does not belong to this controller")
    if prepared._checked_result is not None:
        raise ValueError("prepared support decision has already been committed")
    maximum = min(controller.gp["K"], controller.gp.get("selector_max_units", controller.gp["K"]))
    if not selected or len(selected) > maximum or len({unit.unit_id for unit in selected}) != len(selected):
        raise ValueError("selected support pack must respect the frozen nonempty size limit")
    eligible = set(prepared.metadata["eligible_extraction"]["eligible_event_ids"])
    cancelled = set(prepared.metadata.get("revision_cancelled_event_ids") or ())
    base_catalog, base = _candidate_pieces(controller, prepared, controller.gp["U"])
    canonical = {unit.unit_id: unit for unit in base}
    fallback_type = controller.gp.get("fallback_unit")
    if fallback_type:
        _, fallback = _candidate_pieces(controller, prepared, fallback_type)
        failed_parents = []
        for parent in base_catalog:
            if parent.event_id not in eligible or parent.event_id in cancelled:
                continue
            pieces = uncovered_units(parent, prepared._store, controller.tokenizer,
                prepared.memory.raw_source_indices, list(prepared._gp_visible))
            if pieces and all(controller._append_measure(
                    prepared, [*prepared._gp_visible, piece])[0] is None for piece in pieces):
                failed_parents.append(parent)
        canonical.update({child.unit_id: child for child in fallback
            if any(child.event_id == parent.event_id and unit_is_covered(child, (parent,))
                   for parent in failed_parents)})
    for unit in selected:
        _validate_unit(prepared._store, unit)
        if unit.event_id not in eligible or unit.event_id in cancelled:
            raise ValueError("selected support source is no longer eligible")
        if canonical.get(unit.unit_id) != unit:
            raise ValueError("selected support unit is not a current canonical static source unit")
    candidates = [{"unit_id": unit.unit_id, "unit": unit} for unit in selected]
    chosen = tuple(unit.unit_id for unit in selected)
    decision = {"version": SCHEMA, "status": "abstain", "reason": None,
        "gate_type": "known_support_diagnostic", "diagnostic_only": True,
        "uses_reviewed_source_support": True, "judges_action_correctness": False,
        "regeneration_allowed": False, "selected_unit_ids": list(chosen),
        "selected_unit_count": len(chosen), "appended_unit_count": 0,
        "candidate_supply": {"selected_ids": list(chosen), "actually_appended_ids": [],
            "rejection_reason": None},
        "selection": {"reason": "reviewed_known_support", "selected_ids": list(chosen),
            "selector_or_model_called": False},
        "gate": {"type": "known_support_diagnostic", "triggered": True,
            "reason": "reviewed_known_support"}}
    return controller._commit_evidence_set(prepared, candidates, chosen, decision)


__all__ = ["SupportAnnotationError", "SupportSearchIncomplete", "commit_support", "plan_support"]
