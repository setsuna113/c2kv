"""Source admission for persistent token-selecting backends.

The engine selects retained history KV. This planner only admits complete native
evidence under the same cap; it never substitutes gist estimates for KV receipts.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass

from history_memory.packing import PackedMemory, native_ids, visible_message
from ..adapter import raw_source_cutoff
from ..always_compress import CapacityInfeasible
from ..event_native_raw import RuntimeMemoryView
from ..event_native_s0_policy import EventNativeS0Controller, PreparedEventNativeS0, _Measurement


@dataclass(frozen=True)
class PersistentMemory(PackedMemory):
    source_messages: tuple = ()
    source_tools: tuple = ()
    recovery_messages: tuple = ()
    history_message_count: int = 0
    history_start_message_count: int = 0
    history_budget_tokens: int = 0
    retained_history_cap: int = 0
    recovery_tokens: int = 0
    common_tokens: int = 0
    full_history_tokens: int = 0
    backend_identity: str = ""
    source_event_phases: tuple = ()
    recovered_source_indices: tuple = ()
    tool_plan: object = None
    initial_s0_messages: tuple = ()
    initial_s0_source_indices: tuple = ()
    initial_s0_event_ids: tuple = ()
    native_evidence_source_indices: tuple = ()
    native_evidence_event_ids: tuple = ()
    native_evidence_tokens: int = 0
    retained_history_min_tokens: int = 0
    protection_source_indices: tuple = ()
    protection_event_ids: tuple = ()

    def costs(self, ratio):
        result = super().costs(ratio)
        # This is a conservative admission cap, not a served-KV measurement.
        result.update(resident_kv_tokens=self.common_tokens + self.retained_history_cap
                      + self.recovery_tokens, gist_tokens=0, presented_encoder_tokens=0,
                      encoder_overlap_tokens=0, raw_gist_overlap_events=0)
        return result


def _evidence_message(store, event_ids):
    if not event_ids:
        return ()
    rows = [{"event_id": event_id, "source_indices": list(store.event(event_id).source_indices),
             "messages": [visible_message(row) for row in store.event_messages(event_id)]}
            for event_id in event_ids]
    return ({"role": "user", "content": "Historical source evidence (not new tool execution):\n"
             + json.dumps(rows, ensure_ascii=False, separators=(",", ":"))},)


class PersistentHistoryAllocator(EventNativeS0Controller):
    """Existing recovery policies use this measured source-admission interface."""

    def __init__(self, *args, backend_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.backend_config = backend_config
        configured = min(self.policy_config.history_budget_bytes,
                         self.policy_config.workspace_budget_bytes)
        if configured != backend_config.history_budget_tokens * self.kv_bytes_per_token:
            raise ValueError("RACER planner and runtime policy budgets must agree")

    def _prepare_view(self, store, tools, *, ratio, max_new_tokens, decision_key, decision_index):
        cutoff = raw_source_cutoff([row.to_dict() for row in store.messages])
        mandatory = {event.event_id for event in store.events if event.kind == "instruction"
                     or not event.complete or any(index >= cutoff for index in event.source_indices)}
        eligible = tuple(event.event_id for event in store.events if event.event_id not in mandatory)
        measure = self._try_measure(store, tools, mandatory, mandatory, eligible, eligible,
                                    0, max_new_tokens)
        if measure is None or measure.reasons:
            raise CapacityInfeasible("Persistent history input exceeds its declared context capacity")
        memory = measure.memory
        costs = measure.per_ratio[str(ratio)]
        sources = [index for event_id in eligible for index in store.event(event_id).source_indices]
        metadata = {
            "session_id": store.session_id, "decision_key": decision_key,
            "decision_index": decision_index, "requested_ratio": ratio,
            "max_new_tokens": max_new_tokens,
            "route": {"baseline_identity": self.backend_config.receipt()["identity"],
                      "recovery_enabled": False, "max_generations_per_decision": 1},
            "racer_backend": self.backend_config.receipt(),
            "allocation": "backend_native_persistent",
            "common_raw_prompt_tokens": measure.common_raw_prompt_tokens,
            "raw_prompt_tokens": measure.raw_prompt_tokens,
            "actual_history_bytes": costs["history_bytes"],
            "actual_raw_history_tokens": 0, "actual_gist_tokens": 0,
            "actual_managed_history_tokens": costs["managed_history_tokens"],
            "actual_managed_history_bytes": costs["managed_history_bytes"],
            "actual_total_resident_kv_tokens": costs["total_resident_kv_tokens"],
            "actual_total_resident_kv_bytes": costs["total_resident_kv_bytes"],
            "raw_event_ids": list(memory.view.raw_event_ids), "gist_event_ids": list(eligible),
            "mandatory_raw_event_ids": list(memory.view.mandatory_raw_event_ids),
            "protected_event_ids": list(mandatory), "derived_workspace_prefix_messages": [],
            "eligible_extraction": {"eligible_event_ids": list(eligible),
                "eligible_source_indices": sources, "retained_event_ids": list(eligible),
                "retained_encoder_unit_ids": [], "encoding_performed": False},
            "gist_refill_priority_event_ids": list(eligible),
            "gist_reservation": {"reserved_event_id": None}, "min_gist_reservation_met": None,
            "same_prefix_full_reference": {"history_tokens": memory.full_history_tokens},
            "per_ratio": copy.deepcopy(measure.per_ratio),
            "source_coverage": {"complete_history_coverage": False,
                                "status": "requires_engine_resident_receipt"},
            "accounting_status": "admission_upper_bound_until_engine_receipt",
        }
        return PreparedEventNativeS0(memory, metadata, (), self._owner, store.session_id, decision_key)

    def _try_measure(self, store, tools, raw_ids, mandatory_ids, gist_ids, eligible_event_ids,
                     common_tokens, max_new_tokens, *, derived_messages=()):
        mandatory, raw = set(mandatory_ids), set(raw_ids)
        eligible = set(eligible_event_ids)
        restored = tuple(event.event_id for event in store.events
                         if event.event_id in raw - mandatory)
        if any(event_id not in eligible or not store.event(event_id).complete for event_id in restored):
            raise ValueError("RACER admits only complete archived source events")
        source = tuple(visible_message(message) for message in store.messages)
        cutoff = raw_source_cutoff(list(source))
        start = next((i for i, row in enumerate(source) if row["role"] != "system"), len(source))
        start = min(start, cutoff)
        original = native_ids(self.tokenizer, source, tools=list(tools), generation=True)
        prefix = native_ids(self.tokenizer, source[:cutoff], tools=list(tools)) if cutoff else ()
        protected = native_ids(self.tokenizer, source[:start], tools=list(tools)) if start else ()
        full_history = max(0, len(prefix) - len(protected))
        common = len(original) - full_history
        evidence = _evidence_message(store, restored) + tuple(copy.deepcopy(derived_messages))
        # Account for the empty assistant framing required to discard a held
        # generation, as well as all source/review text and the new prompt.
        rendered = (*source, {"role": "assistant", "content": ""}, *evidence) if evidence else source
        ids = native_ids(self.tokenizer, rendered, tools=list(tools), generation=True)
        extra = len(ids) - len(original)
        budget = self.backend_config.history_budget_tokens
        remaining = budget - extra
        reasons = []
        if extra < 0 or remaining < (1 if full_history else 0):
            reasons.append("native_evidence_exceeds_history_budget")
        retained = min(full_history, max(0, remaining))
        if self.model_context is not None and len(ids) + max_new_tokens > self.model_context:
            reasons.append("model_logical_context")
        if common + retained + extra + max_new_tokens > self.packing.max_sequence_tokens:
            reasons.append("physical_sequence_budget")
        ordered = lambda selected: tuple(event.event_id for event in store.events if event.event_id in selected)
        view = RuntimeMemoryView(gist_event_ids=ordered(eligible - raw), raw_event_ids=ordered(raw),
                evidence_event_ids=restored, omitted_event_ids=(), mandatory_raw_event_ids=ordered(mandatory),
                raw_control_layout="persistent-history-source-admission-v1")
        phases = ["others"] * len(source)
        for event in store.events:
            if event.kind == "tool_event":
                for index in event.source_indices:
                    phases[index] = "act" if source[index]["role"] == "assistant" else "tool"
        memory = PersistentMemory(view, (), ids, tuple(sorted({i for event_id in raw
                                  for i in store.event(event_id).source_indices})), (),
                source_messages=source, source_tools=tuple(tools), recovery_messages=evidence,
                history_message_count=cutoff, history_start_message_count=start,
                history_budget_tokens=budget, retained_history_cap=retained, recovery_tokens=extra,
                common_tokens=common, full_history_tokens=full_history,
                backend_identity=self.backend_config.backend, source_event_phases=tuple(phases),
                recovered_source_indices=tuple(sorted({index for event_id in restored
                    for index in store.event(event_id).source_indices})))
        active = retained + extra
        row = {**memory.costs(8), "history_gist_tokens": 0, "history_raw_tokens": extra,
               "history_total_tokens": active, "history_bytes": active * self.kv_bytes_per_token,
               "managed_history_tokens": active, "managed_history_bytes": active * self.kv_bytes_per_token,
               "total_resident_kv_tokens": common + active,
               "total_resident_kv_bytes": (common + active) * self.kv_bytes_per_token,
               "history_budget_bytes": budget * self.kv_bytes_per_token,
               "workspace_budget_bytes": budget * self.kv_bytes_per_token}
        return _Measurement(memory, common + extra, common, extra,
                            {str(ratio): dict(row) for ratio in self.packing.ratios},
                            len(ids) + max_new_tokens, tuple(reasons))

    def _coverage(self, store, eligible, sources, chunks, memory):
        return ({"complete_history_coverage": False, "status": "requires_engine_resident_receipt",
                 "native_recovered_event_ids": list(memory.view.evidence_event_ids)}, [], [])

    def _compression_ratio(self, full, measure, coverage, ratio):
        return {"status": "requires_engine_resident_receipt", "nominal_ratio_is_not_savings": True}
