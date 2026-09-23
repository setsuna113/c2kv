"""Gist-backed argument views after all incumbent capacity fallbacks fail.

Only completed historical native calls are projected. The actor keeps call
identities and all observations; original arguments stay in the immutable
source archive and the complete event's encoder input. No execution payload
or common-input budget boundary is rewritten.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import json

from history_memory.encoding_scope import plan_encoding_scope, validate_encoding_scope
from history_memory.events import EventStore
from history_memory.packing import visible_message

from .capacity_fallback import (
    CapacityFallbackAllocator, _MAX_PROJECTION_CANDIDATES,
    _MAX_PROJECTION_ATTEMPTS, _bounded_projection_sets,
)
from ..adapter import raw_source_cutoff
from ..always_compress import coverage_accounting


POLICY_VERSION = "c2kv-terminal-tool-arguments-gist-v1"
ARGUMENTS_IN_GIST = {"__c2kv_gist__": "arguments"}


def _project_arguments(source, call_ids=None):
    """Replace nonempty JSON objects, never malformed historical arguments."""
    projected = copy.deepcopy(source)
    changed = []
    for call in projected.get("tool_calls") or ():
        if call_ids is not None and call["id"] not in call_ids:
            continue
        function = call["function"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                return None, ()
        if not isinstance(arguments, dict):
            return None, ()
        if arguments:
            function["arguments"] = dict(ARGUMENTS_IN_GIST)
            changed.append(call["id"])
    return projected, tuple(changed)


def rescue_raw_overrides(store, metadata):
    """Reconstruct only the new rescue view; old successful paths stay exact."""
    receipt = metadata.get("capacity_fallback") or {}
    if not receipt.get("argument_projections"):
        return {}
    overrides = {}
    for row in receipt["argument_projections"]:
        index = row["source_index"]
        projected, _ = _project_arguments(
            store.messages[index].to_dict(), row["projected_call_ids"],
        )
        if projected is None:
            raise ValueError("Argument rescue source is no longer a JSON call")
        overrides[index] = projected
    for row in receipt.get("narrative_projections", ()):
        index = row["source_index"]
        projected = overrides.setdefault(index, store.messages[index].to_dict())
        projected["content"] = None
    return overrides


def refresh_rescue_metadata(metadata, memory):
    """Separate partial raw presentation from complete source coverage."""
    receipt = metadata.get("capacity_fallback") or {}
    if not receipt.get("argument_projections"):
        return
    raw = set(memory.raw_source_indices)
    partial = {
        row["source_index"]
        for key in ("argument_projections", "narrative_projections")
        for row in receipt.get(key, ())
        if row["source_index"] in raw
    }
    exact = raw - partial
    partial_events = {
        row["event_id"]
        for key in ("argument_projections", "narrative_projections")
        for row in receipt.get(key, ()) if row["source_index"] in partial
    }
    metadata["partial_raw_source_indices"] = sorted(partial)
    metadata["exact_raw_source_indices"] = sorted(exact)
    metadata["partial_raw_event_ids"] = sorted(partial_events)
    coverage = metadata["source_coverage"]
    coverage.update(coverage_accounting(
        eligible_sources=frozenset(coverage["eligible_source_indices"]),
        raw_sources=exact,
        retained_blocks=metadata["retained_history_packing_fragments"],
        packing_fragments=metadata["history_packing_fragments"],
    ))
    coverage["partial_raw_source_indices"] = sorted(partial)
    coverage["partial_raw_event_ids"] = sorted(partial_events)
    coverage["raw_event_ids"] = [event_id for event_id in memory.view.raw_event_ids
                                 if event_id not in partial_events
                                 and event_id in coverage["eligible_event_ids"]]
    coverage["raw_gist_overlap_event_ids"] = [event_id for event_id in coverage["raw_event_ids"]
                                             if event_id in memory.view.gist_event_ids]
    metadata["full_source_coverage"] = coverage["complete_history_coverage"]
    receipt.update(
        actual_history_bytes=metadata["actual_history_bytes"],
        rendered_raw_prompt_tokens=metadata["raw_prompt_tokens"],
    )


class ToolEventRescueAllocator(CapacityFallbackAllocator):
    """Append a bounded terminal rescue to the unchanged C1 v2 allocator."""

    def _terminal_failure(self, payload, *, ratio, max_new_tokens,
                          initial_error, attempts, measured_raw,
                          narrative_candidates):
        candidates = self._argument_candidates(payload, measured_raw)
        # Prefer arguments alone; only then combine with the already-authorized
        # narrative projections. Each phase has at most 2*N-1 measurements.
        for narratives in ((), narrative_candidates) if narrative_candidates else ((),):
            for arguments in _bounded_projection_sets(candidates):
                combined = {row["source_index"]: row for row in narratives}
                for row in arguments:
                    value = copy.deepcopy(row)
                    if row["source_index"] in combined:
                        value["projected_message"]["content"] = None
                    combined[row["source_index"]] = value
                prepared = self._attempt(
                    payload, ratio=ratio, max_new_tokens=max_new_tokens,
                    remove_duplicate_gist=True,
                    projections=tuple(combined.values()), attempts=attempts,
                    stage="compress_complete_tool_arguments",
                )
                if prepared is None:
                    continue
                raw = set(prepared.memory.raw_source_indices)
                active_arguments = tuple(row for row in arguments if row["source_index"] in raw)
                active_narratives = tuple(row for row in narratives if row["source_index"] in raw)
                self._annotate(
                    prepared, initial_error, attempts,
                    stage="compress_complete_tool_arguments",
                    remove_duplicate_gist=True, projections=active_narratives,
                )
                receipt = prepared.metadata["capacity_fallback"]
                receipt.pop("field_or_token_truncation_used", None)
                receipt.update(
                    terminal_rescue_version=POLICY_VERSION,
                    trigger="incumbent_capacity_fallback_exhausted",
                    argument_projections=[{
                        key: copy.deepcopy(row[key]) for key in (
                            "event_id", "source_index", "result_source_indices",
                            "call_ids", "projected_call_ids", "source_sha256",
                            "standalone_rendered_token_delta",
                        )
                    } for row in active_arguments],
                    argument_representation="explicit_marker_with_complete_source_gist",
                    argument_projection_candidate_limit=_MAX_PROJECTION_CANDIDATES,
                    terminal_projection_attempt_limit=_MAX_PROJECTION_ATTEMPTS * (
                        2 if narrative_candidates else 1
                    ),
                    tool_calls_arguments_results_ids_order_unchanged=False,
                    tool_names_call_ids_types_results_order_unchanged=True,
                    raw_argument_objects_projected=True,
                    source_or_encoder_truncation_used=False,
                    encoder_scope=self.encoding_scope,
                )
                refresh_rescue_metadata(prepared.metadata, prepared.memory)
                return prepared
        raise initial_error

    def _argument_candidates(self, payload, measured_raw):
        scope = validate_encoding_scope(self.encoding_scope)
        # Record encodings may retain result spans without the producer's
        # arguments. Only whole-event encoder layouts back this projection.
        if scope not in {"current", "event"}:
            return ()
        store = EventStore.from_messages(
            payload["session_id"], payload["messages"], benchmark=self.benchmark,
        )
        cutoff = raw_source_cutoff([message.to_dict() for message in store.messages])
        # Packing can fail before the old controller records a measured view.
        # An event touching the common suffix is nevertheless mandatory in S0.
        measured_raw = set(measured_raw) | {
            event.source_indices[0] for event in store.events
            if any(index >= cutoff for index in event.source_indices)
        }
        users = [event for event in store.events if event.kind == "user"]
        task_packet = users[0].event_id if users and self.benchmark == "acon_appworld" else None
        eligible = set(plan_encoding_scope(
            store,
            tuple(event.event_id for event in store.events
                  if event.complete and event.kind != "instruction"
                  and event.event_id != task_packet
                  and any(index < cutoff for index in event.source_indices)),
            scope,
        ).compressible_event_ids)
        rows = []
        for event in store.events:
            index = event.source_indices[0]
            if (event.kind != "tool_event" or not event.complete
                    or not event.tool_call_ids or event.event_id not in eligible
                    or index >= cutoff or index not in measured_raw):
                continue
            source = store.messages[index].to_dict()
            if source.get("role") != "assistant" or not source.get("tool_calls"):
                continue
            results = tuple(event.source_indices[1:])
            result_ids = tuple(store.messages[i].to_dict().get("tool_call_id") for i in results)
            if len(results) != len(event.tool_call_ids) or set(result_ids) != set(event.tool_call_ids):
                continue
            selected_calls = []
            for call in source["tool_calls"]:
                single = {"role": "assistant", "content": None, "tool_calls": [call]}
                compact, changed = _project_arguments(single)
                if compact is not None and changed and (
                    self._base._count([visible_message(single)], ())
                    > self._base._count([visible_message(compact)], ())
                ):
                    selected_calls.append(call["id"])
            projected, changed = _project_arguments(source, selected_calls)
            if projected is None or not changed:
                continue
            delta = self._base._count([visible_message(source)], ()) - self._base._count(
                [visible_message(projected)], (),
            )
            if delta <= 0:
                continue
            rows.append({
                "event_id": event.event_id, "source_index": index,
                "result_source_indices": list(results), "call_ids": list(event.tool_call_ids),
                "projected_call_ids": list(changed), "projected_message": projected,
                "source_sha256": hashlib.sha256(store.messages[index].json_text.encode("utf-8")).hexdigest(),
                "standalone_rendered_token_delta": delta,
            })
        rows.sort(key=lambda row: (-row["standalone_rendered_token_delta"], row["source_index"]))
        return tuple(rows[:_MAX_PROJECTION_CANDIDATES])

    @contextlib.contextmanager
    def measurement_context(self, prepared):
        receipt = prepared.metadata.get("capacity_fallback") or {}
        if not receipt.get("argument_projections"):
            with super().measurement_context(prepared):
                yield
            return
        overrides = rescue_raw_overrides(prepared._store, prepared.metadata)
        value = self._context_value(
            ratio=int(receipt["requested_ratio"]),
            remove_duplicate_gist=bool(receipt["remove_duplicate_gist"]),
            projections=tuple({"source_index": index, "projected_message": message}
                              for index, message in overrides.items()),
            collector={},
        )
        token = self._measurement_context.set(value)
        try:
            yield
        finally:
            self._measurement_context.reset(token)
