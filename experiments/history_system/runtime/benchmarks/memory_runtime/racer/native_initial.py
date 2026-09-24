"""Run the configured initial strategy against native persistent KV costs.

The existing controller owns selection, priorities, cues and recovery. This
representation supplies exact-source evidence and a residual native KV pool;
compact-source labels are planner candidates, never per-source coverage claims.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace

from history_memory.packing import native_ids, visible_message
from history_memory.source_packing import make_source_view

from ..adapter import raw_source_cutoff
from ..event_native_s0_policy import EventNativeS0Controller, _Measurement
from ..event_native_raw import RuntimeMemoryView
from ..backend_capacity import current_constraints
from .allocator import PersistentMemory


@dataclass(frozen=True)
class NativeEventView(RuntimeMemoryView):
    raw_source_indices: tuple = ()
    gist_source_indices: tuple = ()
    omitted_source_indices: tuple = ()
    mandatory_raw_source_indices: tuple = ()
    partial_event_ids: tuple = ()


def _source_evidence(store, indices):
    selected = set(indices)
    rows = []
    for event in store.events:
        sources = [index for index in event.source_indices if index in selected]
        if sources:
            rows.append({"event_id": event.event_id, "source_indices": sources,
                         "complete_event": set(event.source_indices) <= selected,
                         "messages": [visible_message(store.messages[index]) for index in sources]})
    if not rows:
        return ()
    return ({"role": "user", "content": "Historical source evidence (not new tool execution):\n"
             + json.dumps(rows, ensure_ascii=False, separators=(",", ":"))},)


def _native_input_layout(tokenizer, source, tools):
    cutoff = raw_source_cutoff(list(source))
    start = min(next((i for i, row in enumerate(source) if row["role"] != "system"), len(source)), cutoff)
    original = native_ids(tokenizer, source, tools=list(tools), generation=True)
    prefix = native_ids(tokenizer, source[:cutoff], tools=list(tools)) if cutoff else ()
    protected = native_ids(tokenizer, source[:start], tools=list(tools)) if start else ()
    full_history = max(0, len(prefix) - len(protected))
    return cutoff, start, original, full_history, len(original) - full_history


class NativeInitialRepresentation:
    """Only packing and its cost/coverage contract differ from the policy base."""

    # Ratio changes and projected raw messages require complete source gists.
    # A residual native pool cannot supply those representations.
    supports_gist_capacity_fallback = False

    def _boundary(self, store, tools):
        boundary = super()._boundary(store, tools)
        # A native pool has no complete-source compact representation. Required
        # producer/current inputs must therefore pass exact native admission.
        if not getattr(self, "allow_native_history_demotion", False):
            boundary["mandatory"] = set(boundary["required"])
        return boundary

    def _prepare_view(self, *args, **kwargs):
        self._planning_initial_native = True
        try:
            prepared = super()._prepare_view(*args, **kwargs)
        finally:
            self._planning_initial_native = False
        memory = prepared.memory
        prepared.memory = replace(memory,
            initial_s0_messages=memory.recovery_messages,
            initial_s0_source_indices=memory.native_evidence_source_indices,
            initial_s0_event_ids=memory.native_evidence_event_ids,
            recovered_source_indices=())
        prepared.eligible_chunks = ()
        self._native_metadata(prepared.metadata, prepared.memory, stage="initial_s0")
        return prepared

    def _native_metadata(self, metadata, memory, *, stage):
        metadata["native_initial_allocation"] = {
            "schema": "racer-native-initial-allocation-v2", "stage": stage,
            "selection_policy_class": self._native_policy_class,
            "selection_and_protection_policy_reused": True,
            "representation": "exact_native_sources_plus_backend_residual_pool",
            "compact_sources_are_complete_representations": False,
            "required_sources_use_exact_native_admission": not getattr(
                self, "allow_native_history_demotion", False),
            "source_indices": list(memory.native_evidence_source_indices),
            "event_ids": list(memory.native_evidence_event_ids),
            "initial_s0_source_indices": list(memory.initial_s0_source_indices),
            "initial_s0_event_ids": list(memory.initial_s0_event_ids),
            "native_evidence_tokens": memory.native_evidence_tokens,
            "retained_history_cap": memory.retained_history_cap,
            "retained_history_min_tokens": memory.retained_history_min_tokens,
            "history_budget_tokens": memory.history_budget_tokens,
        }
        metadata["accounting_status"] = "admission_upper_bound_until_engine_receipt"
        metadata["representation_unit"] = "native_exact_source_or_backend_pool"
        metadata["atomic_packing_unit"] = "native_exact_source"
        metadata["min_gist_reservation_required"] = False
        metadata["min_gist_reservation_met"] = None
        allocation = metadata.get("source_allocation")
        if allocation is not None:
            allocation["representation_unit"] = "exact_source_or_unprotected_native_history"
            allocation["unprotected_history_source_indices"] = allocation.pop(
                "gist_source_indices", allocation.get("unprotected_history_source_indices", []))
            allocation["required_source_indices"] = allocation.pop(
                "required_represented_source_indices", allocation.get("required_source_indices", []))
            allocation["complete_source_coverage_guaranteed"] = False
            metadata["unprotected_history_source_indices"] = allocation["unprotected_history_source_indices"]
            metadata.pop("gist_source_indices", None)
        constraints = current_constraints(metadata["session_id"], metadata["decision_key"])
        if constraints is not None:
            metadata["backend_capacity_constraints"] = {
                "stage": constraints.stage, "provenance": constraints.provenance,
                "mandatory_history_tokens": constraints.mandatory_history_tokens,
                "mandatory_source_indices": list(constraints.mandatory_source_indices),
                "admitted_minimum_history_tokens": memory.retained_history_min_tokens,
                "release": constraints.release,
            }
            if constraints.new_source_start is not None:
                metadata["backend_capacity_constraints"].update(
                    new_source_start=constraints.new_source_start,
                    tool_event_mandatory_history_tokens=constraints.tool_event_mandatory_history_tokens,
                    tool_event_mandatory_source_indices=list(constraints.tool_event_mandatory_source_indices))
        metadata["common_raw_prompt_tokens"] = memory.common_tokens
        metadata["actual_raw_history_tokens"] = memory.native_evidence_tokens
        metadata["actual_gist_tokens"] = 0
        row = metadata["per_ratio"][str(metadata["requested_ratio"])]
        for field in ("history_bytes", "managed_history_tokens", "managed_history_bytes",
                      "total_resident_kv_tokens", "total_resident_kv_bytes"):
            metadata["actual_" + field] = row[field]
        needs = metadata.get("source_needs")
        if isinstance(needs, dict):
            needs["history_representation"] = "backend-native-persistent"
            needs.setdefault("source_policy_admission_rule", needs.get("admission_rule"))
            needs["admission_rule"] = "configured_source_policy_with_native_costs"
        reservation = metadata.get("gist_reservation")
        if isinstance(reservation, dict):
            reservation.update(backend_representation="native_residual_pool",
                reserved_native_kv_tokens=memory.retained_history_min_tokens,
                compact_sources_are_complete_representations=False)
        task_accounting = metadata.get("task_packet_accounting")
        if isinstance(task_accounting, dict):
            task_evidence = bool(set(metadata.get("task_packet_source_indices", ()))
                                 & set(memory.native_evidence_source_indices))
            task_accounting.update(charged_to_history_budget=task_evidence,
                                   charged_to_workspace_budget=task_evidence)
        extraction = metadata.get("eligible_extraction")
        if isinstance(extraction, dict):
            extraction.update(status="native_pool_candidates_no_gist_encoding",
                backend_execution_required=False, eligible_encoder_unit_ids=[],
                eligible_chunk_count=0, eligible_presented_encoder_tokens=0,
                eligible_unique_encoder_tokens=0, retained_encoder_unit_ids=[],
                retained_chunk_count=0, retained_presented_encoder_tokens=0,
                encoding_performed=False)

    def _measure_sources(self, store, tools, raw, gist, mandatory, common_tokens,
                         ratio, max_new_tokens, *, derived_messages=()):
        view = make_source_view(store, sorted(raw), sorted(gist), sorted(mandatory))
        return self._native_measure(store, tools, view, raw, mandatory,
                                    max_new_tokens, derived_messages)

    def _try_measure(self, store, tools, raw_ids, mandatory_ids, gist_ids,
                     eligible_event_ids, common_tokens, max_new_tokens, *, derived_messages=()):
        raw, gist, mandatory = set(raw_ids), set(gist_ids), set(mandatory_ids)
        ordered = lambda ids: tuple(event.event_id for event in store.events if event.event_id in ids)
        indices = lambda ids: tuple(sorted({index for event_id in ids
                                            for index in store.event(event_id).source_indices}))
        raw_sources, gist_sources = indices(raw), indices(gist)
        view = NativeEventView(raw_event_ids=ordered(raw), gist_event_ids=ordered(gist),
            mandatory_raw_event_ids=ordered(mandatory), evidence_event_ids=ordered(raw - mandatory),
            omitted_event_ids=ordered({event.event_id for event in store.events} - raw - gist),
            raw_control_layout="racer-shared-s0-native-v2", raw_source_indices=raw_sources,
            gist_source_indices=gist_sources, mandatory_raw_source_indices=indices(mandatory),
            omitted_source_indices=tuple(sorted(set(range(len(store.messages)))
                                                - set(raw_sources) - set(gist_sources))))
        return self._native_measure(store, tools, view, raw_sources, indices(mandatory),
                                    max_new_tokens, derived_messages)

    def _native_measure(self, store, tools, view, raw_sources, mandatory_sources,
                        max_new_tokens, derived_messages):
        source = tuple(visible_message(message) for message in store.messages)
        cutoff, start, original, full_history, common = _native_input_layout(self.tokenizer, source, tools)
        admitted = tuple(sorted(index for index in raw_sources
                                if index < cutoff and source[index]["role"] != "system"))
        event_ids = tuple(event.event_id for event in store.events
                          if set(event.source_indices) & set(admitted))
        evidence = _source_evidence(store, admitted) + tuple(self._merge_protected_derived(derived_messages))
        rendered = (*source, {"role": "assistant", "content": ""}, *evidence) if evidence else source
        ids = native_ids(self.tokenizer, rendered, tools=list(tools), generation=True)
        extra = len(ids) - len(original)
        budget = self.backend_config.history_budget_tokens
        minimum = 1 if full_history else 0
        phases = ["others"] * len(source)
        for event in store.events:
            if event.kind == "tool_event":
                for index in event.source_indices:
                    phases[index] = "act" if source[index]["role"] == "assistant" else "tool"
        constraints = current_constraints(store.session_id)
        if constraints is not None:
            if constraints.history_budget_tokens != budget:
                raise ValueError("Backend constraint budget differs from the configured history budget")
            # The engine opens a lifecycle window on a tool role or tool phase.
            event_phases = ["tool" if row["role"] == "tool" else phase for row, phase in zip(source, phases)]
            minimum = max(minimum, constraints.minimum_history_tokens(admitted, source_phases=event_phases))
        remaining = budget - extra
        retained = min(full_history, max(0, remaining))
        reasons = []
        if extra < 0 or remaining < minimum:
            reasons.append("native_evidence_exceeds_history_budget")
        if self.model_context is not None and len(ids) + max_new_tokens > self.model_context:
            reasons.append("model_logical_context")
        if common + retained + extra + max_new_tokens > self.packing.max_sequence_tokens:
            reasons.append("physical_sequence_budget")
        initial = None
        state = self._sessions.get(store.session_id)
        if not getattr(self, "_planning_initial_native", False) and state is not None:
            initial = state.decisions[state.active_decision_key][1].memory
        initial_sources = tuple(getattr(initial, "initial_s0_source_indices", ()))
        memory = PersistentMemory(view, (), ids, tuple(sorted(raw_sources)), (),
            source_messages=source, source_tools=tuple(tools), recovery_messages=evidence,
            history_message_count=cutoff, history_start_message_count=start,
            history_budget_tokens=budget, retained_history_cap=retained,
            recovery_tokens=extra, common_tokens=common, full_history_tokens=full_history,
            backend_identity=self.backend_config.backend, source_event_phases=tuple(phases),
            recovered_source_indices=tuple(index for index in admitted if index not in initial_sources),
            initial_s0_messages=tuple(getattr(initial, "initial_s0_messages", ())),
            initial_s0_source_indices=initial_sources,
            initial_s0_event_ids=tuple(getattr(initial, "initial_s0_event_ids", ())),
            native_evidence_source_indices=admitted, native_evidence_event_ids=event_ids,
            native_evidence_tokens=extra, retained_history_min_tokens=minimum)
        active = retained + extra
        row = {**memory.costs(8), "history_gist_tokens": 0, "history_raw_tokens": extra,
               "history_total_tokens": active, "history_bytes": active * self.kv_bytes_per_token,
               "managed_history_tokens": active, "managed_history_bytes": active * self.kv_bytes_per_token,
               "managed_workspace_tokens": active, "managed_workspace_bytes": active * self.kv_bytes_per_token,
               "total_resident_kv_tokens": common + active,
               "total_resident_kv_bytes": (common + active) * self.kv_bytes_per_token,
               "history_budget_bytes": budget * self.kv_bytes_per_token,
               "workspace_budget_bytes": budget * self.kv_bytes_per_token,
               "native_evidence_tokens": extra, "native_retained_history_cap": retained,
               "max_new_tokens": max_new_tokens, "sequence_tokens": common + active + max_new_tokens,
               "logical_sequence_tokens": len(ids) + max_new_tokens}
        return _Measurement(memory, common + extra, common, extra,
                            {str(ratio): dict(row) for ratio in self.packing.ratios},
                            len(ids) + max_new_tokens, tuple(reasons))

    def _coverage(self, store, eligible, sources, chunks, memory):
        return ({"complete_history_coverage": False, "status": "requires_engine_resident_receipt",
                 "native_exact_source_indices": list(memory.native_evidence_source_indices),
                 "native_admitted_event_ids": list(memory.native_evidence_event_ids),
                 "compact_candidates_do_not_imply_source_coverage": True}, [], [])

    def _full_reference(self, store, tools, common_tokens, max_new_tokens):
        source = tuple(visible_message(message) for message in store.messages)
        _, _, original, history, common = _native_input_layout(self.tokenizer, source, tools)
        return {"render_only": True, "generation_performed": False,
                "full_prompt_tokens": len(original), "full_history_tokens": history,
                "full_history_bytes": history * self.kv_bytes_per_token,
                "common_live_tokens": common, "common_live_bytes": common * self.kv_bytes_per_token,
                "max_new_tokens": max_new_tokens}

    def _compression_ratio(self, full, measure, coverage, ratio):
        return {"status": "requires_engine_resident_receipt", "nominal_ratio_is_not_savings": True}


class NativeInitialFactory:
    """Instantiate the factory-selected policy with its native representation.

    Selection remains in the ordinary policy factory. The mixin type is chosen
    before construction; no existing instance or wrapper graph is modified.
    """

    def __init__(self, backend):
        self.backend = backend
        self.initial = None

    def __call__(self, policy_type, tokenizer, **kwargs):
        if self.initial is not None or not issubclass(policy_type, EventNativeS0Controller):
            raise ValueError("RACER requires exactly one configured initial allocation controller")
        native_type = type("Native" + policy_type.__name__, (NativeInitialRepresentation, policy_type), {})
        initial = native_type(tokenizer, **kwargs)
        configured = min(initial.policy_config.history_budget_bytes, initial.policy_config.workspace_budget_bytes)
        if configured != self.backend.history_budget_tokens * initial.kv_bytes_per_token:
            raise ValueError("RACER planner and runtime policy budgets must agree")
        initial._native_policy_class = policy_type.__name__
        initial.backend_config = self.backend
        self.initial = initial
        return initial

    def compose(self, composer, tokenizer, **kwargs):
        if self.initial is not None:
            raise ValueError("RACER requires exactly one configured initial allocation controller")

        def branch_factory(policy_type, branch_tokenizer, **branch_kwargs):
            return NativeInitialFactory(self.backend)(policy_type, branch_tokenizer, **branch_kwargs)

        root = composer(tokenizer, initial_allocator_factory=branch_factory, **kwargs)
        self.initial = root
        return root
