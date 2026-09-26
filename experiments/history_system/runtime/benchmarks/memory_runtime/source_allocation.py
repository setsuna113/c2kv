"""Initial budget allocation over exact raw or compressed source messages.

Events retain execution/provenance identity and remain the recovery unit. The
actor representation is selected per complete message, before generation;
there is no incumbent-failure retry or argument rewriting path.
"""
from __future__ import annotations

import copy

from history_memory.packing import PackingBudgetError, visible_message
from history_memory.source_packing import (
    SourceMemoryView, encode_source_chunks, make_source_view, pack_source_memory,
)

from .adapter import raw_source_cutoff
from .always_compress import ALWAYS_COMPRESSION_POLICY, CapacityInfeasible
from .event_native_always import NATIVE_ALWAYS_IMPLEMENTATION_PROFILE, NATIVE_S0_MODE
from .event_native_policy import (
    CURRENT_INPUT_BASELINE, EVENT_NATIVE_POLICY_VERSION, HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT, WORKSPACE_BUDGET_DEFINITION,
)
from .event_native_s0_policy import PreparedEventNativeS0, _Measurement
from .failed_operation import PROMPT_CAP, VERSION as FAILURE_VERSION, cue_for, records
from .policy import PolicyInputError
from .observed_entity_slot import ObservedEntitySlot, slot_message
from .same_event_bridge_only import (
    SAME_EVENT_BRIDGE_ONLY_POLICY, SameEventBridgeOnlyS0Controller,
)


SOURCE_ALLOCATION_VERSION = "c2kv-source-budget-allocation-v1"
MAX_SOURCE_ALLOCATION_STATES = 32


class SourceAllocatedS0Controller(SameEventBridgeOnlyS0Controller):
    """Use S0's lifecycle and lexical query, with one source allocation rule."""

    def __init__(self, tokenizer, *, packing, policy, model_context=None,
                 s0_config=None, benchmark="bfcl"):
        config = dict(s0_config or {})
        bridge = config.pop("observed_entity_slot_policy", SAME_EVENT_BRIDGE_ONLY_POLICY)
        super().__init__(tokenizer, packing=packing, policy=policy,
                         model_context=model_context, s0_config=config or None,
                         benchmark=benchmark, same_event_bridge_only_policy=bridge)
        self.preserve_candidate_derived_messages = True

    def _boundary(self, store, tools):
        cutoff = raw_source_cutoff([message.to_dict() for message in store.messages])
        common = set(range(cutoff, len(store.messages)))
        common.update(index for event in store.events if event.kind == "instruction"
                      for index in event.source_indices)
        users = [event for event in store.events if event.kind == "user"]
        task = users[0] if users and self.benchmark == "acon_appworld" else None
        task_sources = set(task.source_indices) if task else set()
        common.update(task_sources)
        mandatory = set(common)
        if users:
            mandatory.update(users[-1].source_indices)
        mandatory.update(index for event in store.events if not event.complete
                         for index in event.source_indices)
        common_tokens = self._count(
            [visible_message(store.messages[index]) for index in sorted(common)], tools)
        eligible = tuple(event.event_id for event in store.events
                         if event.complete and event.kind != "instruction"
                         and event is not task
                         and any(index < cutoff for index in event.source_indices))
        sources = frozenset(index for event_id in eligible
                            for index in store.event(event_id).source_indices if index < cutoff)
        # Current results pin their own messages, not their historical producer.
        # The producer must still be represented, in either raw or gist form.
        required = set(mandatory)
        required.update(index for event in store.events
                        if mandatory.intersection(event.source_indices)
                        for index in event.source_indices)
        return {"cutoff": cutoff, "common": common, "common_tokens": common_tokens,
                "mandatory": mandatory, "required": required, "eligible": eligible,
                "eligible_sources": sources, "task": task, "task_sources": task_sources}

    def _measure_sources(self, store, tools, raw, gist, mandatory, common_tokens,
                         ratio, max_new_tokens, *, derived_messages=()):
        view = make_source_view(store, sorted(raw), sorted(gist), sorted(mandatory))
        try:
            memory = pack_source_memory(
                store, view, self.tokenizer, tools=tools,
                max_chunk_tokens=self.packing.max_chunk_tokens,
                chunk_overlap=self.packing.chunk_overlap, max_chunks=self.packing.max_chunks,
                derived_workspace_prefix_messages=self._merge_protected_derived(derived_messages))
        except PackingBudgetError:
            return None
        raw_tokens = len(memory.system_input_ids) + len(memory.workspace_input_ids)
        managed_raw = raw_tokens - common_tokens
        if managed_raw < 0:
            raise PolicyInputError("Source allocation changed the common-input boundary")
        cost = memory.costs(ratio)
        history = managed_raw + cost["gist_tokens"]
        history_bytes = history * self.kv_bytes_per_token
        sequence = cost["resident_kv_tokens"] + max_new_tokens
        logical = memory.workspace_position_start + len(memory.workspace_input_ids) + max_new_tokens
        reasons = []
        for exceeded, reason in (
            (len(memory.system_input_ids) > self.packing.max_system_tokens, "system_budget"),
            (len(memory.workspace_input_ids) > self.packing.max_workspace_tokens, "workspace_token_budget"),
            (sum(len(chunk.token_ids) for chunk in memory.chunks) > self.packing.max_encoder_tokens, "encoder_budget"),
            (history_bytes > self.policy_config.history_budget_bytes, f"history_byte_budget:{ratio}"),
            (history_bytes > self.policy_config.workspace_budget_bytes, f"workspace_byte_budget:{ratio}"),
            (sequence > self.packing.max_sequence_tokens, f"physical_sequence_budget:{ratio}"),
            (self.model_context is not None and sequence > self.model_context, f"model_physical_context:{ratio}"),
            (self.model_context is not None and logical > self.model_context, "model_logical_context"),
        ):
            if exceeded:
                reasons.append(reason)
        row = {**cost, "max_new_tokens": max_new_tokens, "sequence_tokens": sequence,
               "logical_sequence_tokens": logical, "history_gist_tokens": cost["gist_tokens"],
               "history_raw_tokens": managed_raw, "history_total_tokens": history,
               "history_bytes": history_bytes, "managed_history_tokens": history,
               "managed_history_bytes": history_bytes, "managed_workspace_tokens": history,
               "managed_workspace_bytes": history_bytes,
               "total_resident_kv_tokens": cost["resident_kv_tokens"],
               "total_resident_kv_bytes": cost["resident_kv_tokens"] * self.kv_bytes_per_token,
               "history_budget_bytes": self.policy_config.history_budget_bytes,
               "workspace_budget_bytes": self.policy_config.workspace_budget_bytes}
        return _Measurement(memory, raw_tokens, common_tokens, managed_raw,
                            {str(ratio): row}, logical, tuple(reasons))

    def _allocate(self, store, tools, *, ratio, max_new_tokens, boundary,
                  derived_messages=(), force_raw=(), preferred_events=None):
        mandatory = set(boundary["mandatory"]) | set(force_raw)
        required = set(boundary["required"]) | mandatory
        trials = 0
        pruned_states = 0
        measurements = {}

        def measure(raw, gist):
            nonlocal trials
            signature = (tuple(sorted(raw)), tuple(sorted(gist)))
            if signature in measurements:
                return measurements[signature]
            trials += 1
            measurements[signature] = self._measure_sources(store, tools, raw, gist, mandatory,
                                         boundary["common_tokens"], ratio, max_new_tokens,
                                         derived_messages=derived_messages)
            return measurements[signature]

        def key(item):
            if item is None:
                return (float("inf"), float("inf"), float("inf"))
            structural = sum(not reason.startswith(("history_byte_budget:", "workspace_byte_budget:"))
                             for reason in item.reasons)
            return (structural, item.per_ratio[str(ratio)]["history_bytes"],
                    len(item.memory.view.gist_source_indices))

        def minimum(raw, gist, indices):
            nonlocal pruned_states
            raw, gist = set(raw), set(gist)
            states = [(raw, gist, measure(raw, gist))]
            for index in sorted(set(indices) - raw - gist):
                expanded = []
                for current_raw, current_gist, _ in states:
                    for next_raw, next_gist in (
                        (current_raw | {index}, current_gist),
                        (current_raw, current_gist | {index}),
                    ):
                        measured = measure(next_raw, next_gist)
                        if measured is not None:
                            expanded.append((next_raw, next_gist, measured))
                if not expanded:
                    return raw, gist, None
                # Keep joint alternatives before any feasibility decision: a
                # low-byte gist can consume more logical context than raw.
                # Ordinary tool events exhaustively compare all combinations;
                # unusually large parallel events retain bounded cost/context
                # alternatives, with pruning disclosed in the receipt.
                expanded.sort(key=lambda state: key(state[2]))
                if len(expanded) > MAX_SOURCE_ALLOCATION_STATES:
                    half = MAX_SOURCE_ALLOCATION_STATES // 2
                    retained = expanded[:half]
                    for state in sorted(expanded, key=lambda state: (state[2].logical_sequence_tokens, key(state[2]))):
                        if not any(state[0] == old[0] and state[1] == old[1] for old in retained):
                            retained.append(state)
                        if len(retained) == MAX_SOURCE_ALLOCATION_STATES:
                            break
                    pruned_states += len(expanded) - len(retained)
                    expanded = retained
                states = expanded
            return min(states, key=lambda state: key(state[2]))

        raw, gist, selected = minimum(mandatory, set(), required)
        if selected is None or selected.reasons:
            return None, {"version": SOURCE_ALLOCATION_VERSION,
                          "phase": "initial_budget_allocation", "status": "infeasible",
                          "legacy_fallback_invoked": False, "measurement_count": trials,
                          "pruned_allocation_states": pruned_states,
                          "mandatory_raw_source_indices": sorted(mandatory),
                          "required_represented_source_indices": sorted(required),
                          "history_tokens": None if selected is None else selected.per_ratio[str(ratio)]["history_total_tokens"],
                          "reasons": ["packing_budget_exceeded"] if selected is None else list(selected.reasons)}

        # Prefer exact current history when affordable, without reserving a
        # duplicate gist. This is the same choice used for older messages.
        for index in sorted(gist, reverse=True):
            exact = measure(raw | {index}, gist - {index})
            if exact is not None and not exact.reasons:
                raw.add(index)
                gist.remove(index)
                selected = exact

        full_raw_events = selected.memory.view.raw_event_ids
        requested, _, needs = self._lexical_request(
            store, tools, full_raw_events,
            recent_tool_event_visible=any(event.kind == "tool_event" and event.complete
                                         and event.event_id in full_raw_events for event in store.events[-1:]))
        eligible = list(boundary["eligible"])
        recent_tools = [event.event_id for event in reversed(store.events)
                        if event.kind == "tool_event" and event.complete and event.event_id in eligible]
        priority = list(dict.fromkeys([
            *(preferred_events or ()), *requested.source_ids,
            *recent_tools[:1], *eligible[:1], *reversed(eligible)]))
        admitted, omitted = [], []
        prefer_raw = set(preferred_events or ()) | set(requested.source_ids) | set(recent_tools[:1])
        for event_id in priority:
            indices = set(store.event(event_id).source_indices)
            if indices <= raw | gist:
                continue
            exact = measure(raw | indices, gist - indices) if event_id in prefer_raw else None
            if exact is not None and not exact.reasons:
                raw.update(indices)
                gist.difference_update(indices)
                selected = exact
                admitted.append(event_id)
                continue
            proposed_raw, proposed_gist, compact = minimum(raw, gist, indices)
            if compact is not None and not compact.reasons:
                raw, gist, selected = proposed_raw, proposed_gist, compact
                admitted.append(event_id)
            else:
                omitted.append(event_id)
        receipt = {"version": SOURCE_ALLOCATION_VERSION, "phase": "initial_budget_allocation",
                   "status": "allocated", "legacy_fallback_invoked": False,
                   "representation_unit": "complete_source_message",
                   "selection_rule": "minimum_required_cost_then_priority_exact_or_gist",
                   "measurement_count": trials, "measurement_ratios": [ratio],
                   "max_allocation_states": MAX_SOURCE_ALLOCATION_STATES,
                   "pruned_allocation_states": pruned_states,
                   "mandatory_raw_source_indices": sorted(mandatory),
                   "required_represented_source_indices": sorted(required),
                   "raw_source_indices": sorted(raw), "gist_source_indices": sorted(gist),
                   "priority_event_ids": priority, "admitted_optional_event_ids": admitted,
                   "omitted_optional_event_ids": omitted,
                   "lexical_requested_event_ids": list(requested.source_ids),
                   "lexical_request_status": requested.status, "source_needs_input": needs}
        return selected, receipt

    def _prepare_view(self, store, tools, *, ratio, max_new_tokens, decision_key, decision_index):
        if self.encoding_scope not in {"current", "event"}:
            raise PolicyInputError("Source allocation requires the complete-source encoding contract")
        boundary = self._boundary(store, tools)
        measure, allocation = self._allocate(store, tools, ratio=ratio,
                                             max_new_tokens=max_new_tokens, boundary=boundary)
        if measure is None:
            raise CapacityInfeasible(f"No admitted source allocation for required inputs: {allocation!r}")
        derived = ()
        diagnostic = records(store)
        failed = diagnostic["failures"][0] if diagnostic["failures"] else None
        failure = {"version": FAILURE_VERSION, "status": "no_current_goal_failure",
                   "selected_record": copy.deepcopy(failed), "failure_record_count": len(diagnostic["failures"]),
                   "incremental_raw_tokens": 0, "extra_bytes": 0, "prompt_cap": PROMPT_CAP,
                   "goal_completion": "unknown", "input_source": "preceding observed prefix only",
                   "derived_workspace_message_count": 0, "source_indices_are_not_raw_copies": True}
        if failed is not None:
            cue = cue_for(failed)
            standalone = self._count([cue], ())
            failure["standalone_tokens"] = standalone
            failure["status"] = "cue_prompt_cap"
            if standalone <= PROMPT_CAP:
                trial = self._measure_sources(store, tools, measure.memory.view.raw_source_indices,
                    measure.memory.view.gist_source_indices, boundary["mandatory"],
                    boundary["common_tokens"], ratio, max_new_tokens, derived_messages=(cue,))
                failure["status"] = "workspace_cap"
                if trial is not None and not trial.reasons:
                    delta = trial.raw_prompt_tokens - measure.raw_prompt_tokens
                    failure.update(status="admitted", incremental_raw_tokens=delta,
                                   extra_bytes=delta * self.kv_bytes_per_token,
                                   derived_workspace_message_count=1)
                    measure, derived = trial, (cue,)
        chunks = encode_source_chunks(store, sorted(boundary["eligible_sources"]), self.tokenizer,
                                      max_chunk_tokens=self.packing.max_chunk_tokens,
                                      chunk_overlap=self.packing.chunk_overlap)
        metadata = self._metadata(store, tools, measure, allocation, boundary, chunks,
                                  ratio, max_new_tokens, decision_key, decision_index, derived)
        metadata["failed_operation_cue"] = failure
        prepared = PreparedEventNativeS0(measure.memory, metadata, chunks, self._owner,
                                        store.session_id, decision_key)
        return self._apply_same_event_bridge(prepared, store, tools, ratio=ratio,
                                             max_new_tokens=max_new_tokens)

    def _coverage(self, store, eligible_event_ids, eligible_source_indices, eligible_chunks, memory):
        coverage, fragments, retained = super()._coverage(
            store, eligible_event_ids, eligible_source_indices, eligible_chunks, memory)
        coverage["source_message_encoded_indices"] = coverage.pop("whole_event_encoded_source_indices")
        coverage["source_message_extra_indices"] = coverage.pop("whole_event_extra_source_indices")
        coverage["partial_event_ids"] = list(memory.view.partial_event_ids)
        coverage["representation_unit"] = "complete_source_message"
        return coverage, fragments, retained

    def _metadata(self, store, tools, measure, allocation, boundary, chunks, ratio,
                  max_new_tokens, decision_key, decision_index, derived):
        view = measure.memory.view
        eligible = boundary["eligible"]
        coverage, fragments, retained = self._coverage(
            store, eligible, boundary["eligible_sources"], chunks, measure.memory)
        full = self._full_reference(store, tools, boundary["common_tokens"], max_new_tokens)
        row = measure.per_ratio[str(ratio)]
        task = boundary["task"]
        without_task = boundary["common"] - boundary["task_sources"]
        task_tokens = boundary["common_tokens"] - self._count(
            [visible_message(store.messages[index]) for index in sorted(without_task)], tools) if task else 0
        return {
            "event_native_s0_version": SOURCE_ALLOCATION_VERSION,
            "event_native_policy_version": EVENT_NATIVE_POLICY_VERSION,
            "policy_source_commit": POLICY_SOURCE_COMMIT, "session_id": store.session_id,
            "benchmark": self.benchmark, "decision_key": decision_key, "decision_index": decision_index,
            "view_mode": NATIVE_S0_MODE, "mode": NATIVE_S0_MODE, "route_mode": NATIVE_S0_MODE,
            "route": {"view_mode": NATIVE_S0_MODE, "baseline_identity": SOURCE_ALLOCATION_VERSION,
                      "recovery_enabled": False, "max_generations_per_decision": 1,
                      "legacy_1088_equivalent": False},
            "compression_policy": ALWAYS_COMPRESSION_POLICY,
            "implementation_profile": NATIVE_ALWAYS_IMPLEMENTATION_PROFILE,
            "history_view_protocol": "fixed-budget-main", "requested_ratio": ratio,
            "max_new_tokens": max_new_tokens, "model_context": self.model_context,
            "configured_max_sequence_tokens": self.packing.max_sequence_tokens,
            "chunk_geometry": {key: getattr(self.packing, key) for key in
                               ("max_chunk_tokens", "chunk_overlap", "max_chunks")},
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "history_budget_bytes": self.policy_config.history_budget_bytes,
            "workspace_budget_bytes": self.policy_config.workspace_budget_bytes,
            "shared_allocation_budget_bytes": min(self.policy_config.history_budget_bytes,
                                                   self.policy_config.workspace_budget_bytes),
            "history_budget_definition": HISTORY_BUDGET_DEFINITION,
            "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
            "current_input_baseline": CURRENT_INPUT_BASELINE,
            "s0_common_input_definition": "source instructions plus raw_source_cutoff suffix plus the AppWorld first-user task packet",
            "raw_source_cutoff": boundary["cutoff"], "common_input_source_indices": sorted(boundary["common"]),
            "common_raw_prompt_tokens": boundary["common_tokens"],
            "task_packet_protection": "first_non_system_user_raw" if task else "none",
            "task_packet_event_id": task.event_id if task else None,
            "task_packet_source_indices": sorted(boundary["task_sources"]),
            "task_packet_raw_tokens": task_tokens, "task_packet_raw_bytes": task_tokens * self.kv_bytes_per_token,
            "task_packet_accounting": {"charged_to_history_budget": False,
                                       "charged_to_workspace_budget": False, "included_in_total_resident_kv": bool(task)},
            "source_allocation": copy.deepcopy(allocation),
            "raw_source_indices": list(measure.memory.raw_source_indices),
            "gist_source_indices": list(view.gist_source_indices),
            "omitted_source_indices": list(view.omitted_source_indices),
            "raw_event_ids": list(view.raw_event_ids), "gist_event_ids": list(view.gist_event_ids),
            "omitted_event_ids": list(view.omitted_event_ids), "partial_raw_event_ids": list(view.partial_event_ids),
            "mandatory_raw_event_ids": list(view.mandatory_raw_event_ids),
            "protected_event_ids": list(view.mandatory_raw_event_ids), "evidence_event_ids": [],
            "raw_evidence_event_ids": list(view.raw_event_ids), "selected_event_ids": list(view.raw_event_ids),
            "selected_source_indices": list(measure.memory.raw_source_indices),
            "retrieved_event_ids": [item for item in allocation["lexical_requested_event_ids"] if item in view.raw_event_ids],
            "retained_event_ids": [], "recency_selected_event_ids": [], "lease_decisions": 0,
            "max_retrieved_events": 2, "pre_draft_retrieval": True,
            "recovery_stage": "pre_generation_source_allocation", "post_draft_exact_recovery_applied": False,
            "derived_workspace_prefix_messages": copy.deepcopy(list(self._merge_protected_derived(derived))),
            "derived_workspace_prefix_message_count": len(self._merge_protected_derived(derived)),
            "derived_workspace_source_indices": [],
            "source_needs": {"strategy": "lexical", "predictor_calls": 0,
                             "requested_event_ids": allocation["lexical_requested_event_ids"],
                             "input_receipt": allocation["source_needs_input"],
                             "admitted_event_ids": [item for item in allocation["lexical_requested_event_ids"] if item in view.raw_event_ids],
                             "admission_rule": "one initial raw/gist source allocation"},
            "gist_reservation": {"required": False, "reserved_event_id": None, "status": "source_cost_allocation"},
            "min_gist_reservation_required": False, "min_gist_reservation_met": None,
            "gist_refill_priority_event_ids": allocation["priority_event_ids"],
            "gist_refilled_event_ids": list(view.gist_event_ids),
            "eligible_extraction": {"status": "planned_for_pre_generation_extraction",
                "source": "preceding observable EventStore prefix", "eligible_event_ids": list(eligible),
                "encoding_scope": self.encoding_scope,
                "representation_unit": "complete_source_message",
                "encoding_event_groups": [[item] for item in eligible],
                "pending_raw_event_ids": [event.event_id for event in store.events if not event.complete],
                "eligible_encoder_unit_ids": list(dict.fromkeys(chunk.event_id for chunk in chunks)),
                "eligible_source_indices": sorted(boundary["eligible_sources"]),
                "source_message_encoded_indices": sorted(boundary["eligible_sources"]),
                "eligible_chunk_count": len(chunks), "eligible_presented_encoder_tokens": sum(len(chunk.token_ids) for chunk in chunks),
                "eligible_unique_encoder_tokens": self._unique_encoder_tokens(chunks),
                "retained_event_ids": list(view.gist_event_ids),
                "retained_encoder_unit_ids": list(dict.fromkeys(chunk.event_id for chunk in measure.memory.chunks)),
                "retained_chunk_count": len(measure.memory.chunks),
                "retained_presented_encoder_tokens": sum(len(chunk.token_ids) for chunk in measure.memory.chunks),
                "omitted_active_gist_event_ids": [item for item in eligible if item not in view.gist_event_ids],
                "raw_gist_overlap_event_ids": [], "backend_execution_required": bool(chunks),
                "cache_reuse_known_after_generation": True, "charged_separately_from_active_history_bytes": True},
            "history_packing_fragments": fragments, "retained_history_packing_fragments": retained,
            "source_coverage": coverage, "full_source_coverage": coverage["complete_history_coverage"],
            "per_ratio": copy.deepcopy(measure.per_ratio), "raw_prompt_tokens": measure.raw_prompt_tokens,
            "actual_raw_history_tokens": measure.raw_history_tokens, "actual_gist_tokens": row["history_gist_tokens"],
            "actual_history_bytes": row["history_bytes"], "actual_managed_history_tokens": row["managed_history_tokens"],
            "actual_managed_history_bytes": row["managed_history_bytes"],
            "actual_total_resident_kv_tokens": row["total_resident_kv_tokens"],
            "actual_total_resident_kv_bytes": row["total_resident_kv_bytes"],
            "logical_sequence_tokens": measure.logical_sequence_tokens,
            "same_prefix_full_reference": full, "compression_ratio": self._compression_ratio(full, measure, coverage, ratio),
            "no_eligible_history": not eligible, "encoding_scope": self.encoding_scope,
            "representation_unit": "complete_source_message",
            "source_message_basis": "controller_visible_messages",
            "upstream_tool_rendering": store.source_prefix is not None,
            "source_identity_basis": "source_prefix" if store.source_prefix is not None else "messages",
            "pending_encoding_event_ids": [], "atomic_packing_unit": "whole_source_message_all_encoder_chunks",
            "legacy_1088_block_parity": False,
        }

    def _apply_same_event_bridge(self, prepared, store, tools, *, ratio, max_new_tokens):
        result = super()._apply_same_event_bridge(
            prepared, store, tools, ratio=ratio, max_new_tokens=max_new_tokens)
        metadata = result.metadata
        metadata["same_event_reference"]["no_bridge_input_semantics"] = "unchanged-initial-source-allocation"
        row = metadata["per_ratio"][str(ratio)]
        metadata.update(
            actual_managed_history_tokens=row["managed_history_tokens"],
            actual_managed_history_bytes=row["managed_history_bytes"],
            actual_total_resident_kv_tokens=row["total_resident_kv_tokens"],
            actual_total_resident_kv_bytes=row["total_resident_kv_bytes"],
        )
        return result

    def measure_source_view(self, prepared, store, tools, *, ratio, max_new_tokens, derived_messages=()):
        view = prepared.memory.view
        return self._measure_sources(store, tools, view.raw_source_indices, view.gist_source_indices,
                                     view.mandatory_raw_source_indices, prepared.metadata["common_raw_prompt_tokens"],
                                     ratio, max_new_tokens, derived_messages=derived_messages)

    def repack_sources(self, prepared, *, candidate=None, derived_messages=None, goal_view=False):
        store, tools = prepared._store, prepared._tools
        ratio, max_new = prepared.metadata["requested_ratio"], prepared.metadata["max_new_tokens"]
        derived = list(prepared.metadata.get("derived_workspace_prefix_messages") or ())
        for message in derived_messages or ():
            if message not in derived:
                derived.append(message)
        boundary = self._boundary(store, tools)
        force_raw = set()
        bridge = copy.deepcopy(prepared.metadata.get("same_event_reference", {}))
        if candidate is not None:
            event = store.event(candidate)
            if not event.complete or candidate not in boundary["eligible"]:
                raise ValueError("Recovery requires one eligible complete source event")
            force_raw.update(event.source_indices)
            if bridge.get("status") == "admitted" and bridge.get("source_event_id") == candidate:
                reference = self._select_bridge_reference(
                    store, tools, raw_source_indices=prepared.memory.raw_source_indices).reference
                if reference is None or reference.event_id != candidate:
                    raise PolicyInputError("Recorded SAME bridge no longer matches its source")
                message = slot_message([ObservedEntitySlot(reference.field, reference.value)])
                derived = [item for item in derived if item != message]
                bridge.update(status="replaced_by_complete_event_raw",
                              source_result_present_in_raw_workspace=True,
                              incremental_raw_tokens=0, extra_bytes=0)
        if (candidate is None and not derived_messages and not goal_view
                and isinstance(prepared.memory.view, SourceMemoryView)):
            measure = self.measure_source_view(prepared, store, tools, ratio=ratio,
                                               max_new_tokens=max_new, derived_messages=derived)
            if measure is None or measure.reasons:
                raise PolicyInputError("Unchanged source view no longer fits its recorded budget")
            return measure, copy.deepcopy(prepared.metadata), {
                "policy": SOURCE_ALLOCATION_VERSION, "status": "admitted", "candidate_event_id": None,
                "b0_rechecked_after_all_changes": True, "source_partition_reused": True}
        measure, allocation = self._allocate(store, tools, ratio=ratio, max_new_tokens=max_new,
            boundary=boundary, derived_messages=derived, force_raw=force_raw,
            preferred_events=prepared.metadata.get("retrieved_event_ids", ()))
        receipt = {"policy": SOURCE_ALLOCATION_VERSION,
                   "status": "admitted" if measure is not None else "abstained",
                   "candidate_event_id": candidate, "b0_rechecked_after_all_changes": True,
                   "complete_event_raw_required": candidate is not None, "allocation": allocation}
        if measure is None:
            return None, None, receipt
        chunks = prepared.eligible_chunks
        if not isinstance(prepared.memory.view, type(measure.memory.view)):
            # Event-based incumbent chunks are not source-message encoder units.
            chunks = encode_source_chunks(store, sorted(boundary["eligible_sources"]), self.tokenizer,
                max_chunk_tokens=self.packing.max_chunk_tokens,
                chunk_overlap=self.packing.chunk_overlap)
        metadata = self._metadata(store, tools, measure, allocation, boundary, chunks,
            ratio, max_new, prepared.metadata["decision_key"], prepared.metadata["decision_index"], derived)
        for key in ("candidate_algorithm", "route", "same_event_reference", "failed_operation_cue",
                    "post_draft_recovery_config"):
            if key in prepared.metadata:
                metadata[key] = copy.deepcopy(prepared.metadata[key])
        if bridge:
            bridge["source_result_present_in_raw_workspace"] = bridge.get("result_source_index") in measure.memory.raw_source_indices
            metadata["same_event_reference"] = bridge
        metadata.update(post_draft_exact_recovery_applied=True, recovery_stage="post_draft_pre_tool_e1",
                        post_draft_recovery_allocation=copy.deepcopy(receipt))
        return measure, metadata, receipt
