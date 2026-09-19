"""First-draft event allocation for the four candidate algorithms.

The non-S0 variants reuse the native packer and B0 measurement, but never
prepare an S0 view as a temporary input.  Every represented source event is
chosen from the observable prefix before the draft is generated.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from history_memory.encoding_scope import plan_encoding_scope, validate_encoding_scope
from history_memory.packing import encode_scope_chunks, select_view, visible_message

from ..adapter import raw_source_cutoff
from ..always_compress import ALWAYS_COMPRESSION_POLICY, CapacityInfeasible
from ..event_native_always import NATIVE_ALWAYS_IMPLEMENTATION_PROFILE
from ..event_native_policy import (
    CURRENT_INPUT_BASELINE,
    EVENT_NATIVE_POLICY_VERSION,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from ..event_native_s0_policy import (
    _FIXED_GEOMETRY,
    EventNativeS0Controller,
    PreparedEventNativeS0,
)
from ..failed_operation import PROMPT_CAP, VERSION as FAILED_OPERATION_VERSION
from ..failed_operation import cue_for, records as failed_operation_records


CANDIDATE_ALLOCATION_VERSION = "candidate-first-allocation-v1"
VARIANTS = frozenset({
    "static_t02", "turn_c1", "goal_rescue", "dependency_first",
})


class CandidateAllocator(EventNativeS0Controller):
    """Keep the native lifecycle while replacing non-S0 initial allocation."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        packing: Mapping[str, Any],
        policy: Mapping[str, Any],
        variant: str,
        model_context: int | None = None,
        s0_config: Mapping[str, Any] | None = None,
        benchmark: str = "bfcl",
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown candidate allocator variant: {variant!r}")
        super().__init__(
            tokenizer,
            packing=packing,
            policy=policy,
            model_context=model_context,
            s0_config=s0_config,
            benchmark=benchmark,
        )
        self.variant = variant

    def _prepare_view(
        self,
        store,
        tools,
        *,
        ratio: int,
        max_new_tokens: int,
        decision_key: str,
        decision_index: int,
    ) -> PreparedEventNativeS0:
        if self.variant == "goal_rescue":
            prepared = super()._prepare_view(
                store,
                tools,
                ratio=ratio,
                max_new_tokens=max_new_tokens,
                decision_key=decision_key,
                decision_index=decision_index,
            )
            prepared.metadata["candidate_algorithm"] = self.variant
            prepared.metadata["candidate_allocation_version"] = (
                CANDIDATE_ALLOCATION_VERSION
            )
            return prepared

        cutoff = raw_source_cutoff([message.to_dict() for message in store.messages])
        users = [event for event in store.events if event.kind == "user"]
        task_packet = min(
            users, key=lambda event: min(event.source_indices)
        ) if users and self.benchmark == "acon_appworld" else None
        task_sources = set(task_packet.source_indices) if task_packet else set()
        common_sources = {
            index
            for event in store.events
            if event.kind == "instruction"
            for index in event.source_indices
        } | set(range(cutoff, len(store.messages))) | task_sources
        common_messages = [
            visible_message(store.messages[index]) for index in sorted(common_sources)
        ]
        common_tokens = self._count(common_messages, tools)
        without_task = [
            visible_message(store.messages[index])
            for index in sorted(common_sources - task_sources)
        ]
        task_raw_tokens = common_tokens - self._count(without_task, tools)
        if task_raw_tokens < 0:
            raise ValueError("Task-packet marginal token count must be nonnegative")

        mandatory = {
            event.event_id for event in store.events
            if not event.complete
            or event.kind == "instruction"
            or any(index >= cutoff for index in event.source_indices)
        }
        if users:
            mandatory.add(users[-1].event_id)
        if task_packet:
            mandatory.add(task_packet.event_id)

        historical = tuple(
            event.event_id for event in store.events
            if event.complete
            and event.kind != "instruction"
            and event.event_id != (task_packet.event_id if task_packet else None)
            and any(index < cutoff for index in event.source_indices)
        )
        scope = plan_encoding_scope(
            store, historical, validate_encoding_scope(self.encoding_scope)
        )
        eligible = tuple(scope.compressible_event_ids)
        mandatory.update(scope.pending_event_ids)
        eligible_sources = frozenset(
            index
            for event_id in eligible
            for index in store.event(event_id).source_indices
            if index < cutoff
        )
        chunks = encode_scope_chunks(
            store,
            eligible,
            self.tokenizer,
            encoding_scope=scope.scope,
            max_chunk_tokens=self.packing.max_chunk_tokens,
            chunk_overlap=self.packing.chunk_overlap,
            atomic_unit_token_limit=self._atomic_unit_token_limit(),
            event_groups=scope.event_groups,
        )

        raw = set(mandatory)
        if self.variant in {"static_t02", "dependency_first"}:
            static = select_view(
                store, recent_tool_events=self.packing.recent_tool_events
            )
            raw.update(static.raw_event_ids)
        measure = self._fit(
            store, tools, raw, mandatory, (), eligible, common_tokens,
            max_new_tokens,
        )
        current_turn_raw: list[str] = []
        skipped_current: list[dict[str, Any]] = []
        if self.variant == "turn_c1" and users:
            current_user_index = min(users[-1].source_indices)
            current_events = [
                event for event in store.events
                if event.complete
                and event.kind == "tool_event"
                and min(event.source_indices) > current_user_index
            ]
            for event in reversed(current_events):
                if event.event_id in raw:
                    continue
                trial = self._try_measure(
                    store, tools, raw | {event.event_id}, mandatory, (),
                    eligible, common_tokens, max_new_tokens,
                )
                if trial is None or trial.reasons:
                    skipped_current.append({
                        "event_id": event.event_id,
                        "reasons": ["packing_budget_exceeded"]
                        if trial is None else list(trial.reasons),
                    })
                    continue
                raw.add(event.event_id)
                measure = trial
            current_turn_raw = [
                event.event_id for event in current_events if event.event_id in raw
            ]

        # The static view defines a raw/gist partition.  Admit gist units from
        # newest to oldest only while the same B0 and geometry remain feasible.
        # Unlike S0, no oldest-gist reservation or lexical/raw reserve is used.
        gist: list[str] = []
        skipped_gist: list[dict[str, Any]] = []
        for group in reversed(tuple(scope.event_groups)):
            if set(group) & raw:
                continue
            trial = self._try_measure(
                store, tools, raw, mandatory, (*gist, *group), eligible,
                common_tokens, max_new_tokens,
            )
            if trial is None or trial.reasons:
                skipped_gist.append({
                    "event_ids": list(group),
                    "reasons": ["packing_budget_exceeded"]
                    if trial is None else list(trial.reasons),
                })
                continue
            gist.extend(group)
            measure = trial

        derived: tuple[dict[str, Any], ...] = ()
        failed_receipt = {
            "version": FAILED_OPERATION_VERSION,
            "status": "disabled_for_variant",
            "selected_record": None,
            "failure_record_count": 0,
            "incremental_raw_tokens": 0,
            "extra_bytes": 0,
            "prompt_cap": PROMPT_CAP,
            "goal_completion": "unknown",
        }
        if self.variant == "turn_c1":
            derived, measure, failed_receipt = self._failed_cue(
                store, tools, raw, mandatory, gist, eligible, common_tokens,
                max_new_tokens, measure,
            )

        dependency_receipt: dict[str, Any] | None = None
        if self.variant == "dependency_first":
            derived, measure, dependency_receipt = self._dependency_packets(
                store, tools, raw, mandatory, gist, eligible, common_tokens,
                max_new_tokens, measure,
            )

        return self._build_prepared(
            store=store,
            tools=tools,
            ratio=ratio,
            max_new_tokens=max_new_tokens,
            decision_key=decision_key,
            decision_index=decision_index,
            cutoff=cutoff,
            task_packet=task_packet,
            task_sources=task_sources,
            task_raw_tokens=task_raw_tokens,
            common_sources=common_sources,
            common_tokens=common_tokens,
            scope=scope,
            eligible=eligible,
            eligible_sources=eligible_sources,
            chunks=chunks,
            mandatory=mandatory,
            measure=measure,
            derived=derived,
            current_turn_raw=current_turn_raw,
            skipped_current=skipped_current,
            skipped_gist=skipped_gist,
            failed_receipt=failed_receipt,
            dependency_receipt=dependency_receipt,
        )

    def _fit(
        self, store, tools, raw, mandatory, gist, eligible, common_tokens,
        max_new_tokens, *, derived=(),
    ):
        measure = self._try_measure(
            store, tools, raw, mandatory, gist, eligible, common_tokens,
            max_new_tokens, derived_messages=derived,
        )
        if measure is None or measure.reasons:
            reasons = ["packing_budget_exceeded"] if measure is None else list(
                measure.reasons
            )
            raise CapacityInfeasible(
                f"Candidate mandatory native input cannot fit declared limits: {reasons!r}"
            )
        return measure

    def _failed_cue(
        self, store, tools, raw, mandatory, gist, eligible, common_tokens,
        max_new_tokens, measure,
    ):
        diagnostic = failed_operation_records(store)
        selected = diagnostic["failures"][0] if diagnostic["failures"] else None
        receipt = {
            "version": FAILED_OPERATION_VERSION,
            "status": "no_current_goal_failure",
            "selected_record": copy.deepcopy(selected),
            "failure_record_count": len(diagnostic["failures"]),
            "incremental_raw_tokens": 0,
            "extra_bytes": 0,
            "prompt_cap": PROMPT_CAP,
            "goal_completion": "unknown",
            "input_source": "preceding observed prefix only",
            "source_indices_are_not_raw_copies": True,
        }
        if selected is None:
            return (), measure, receipt
        cue = cue_for(selected)
        standalone = self._count([cue], ())
        receipt["standalone_tokens"] = standalone
        if standalone > PROMPT_CAP:
            receipt["status"] = "cue_prompt_cap"
            return (), measure, receipt
        trial = self._try_measure(
            store, tools, raw, mandatory, gist, eligible, common_tokens,
            max_new_tokens, derived_messages=(cue,),
        )
        if trial is None or trial.reasons:
            receipt["status"] = "workspace_cap"
            receipt["admission_failures"] = ["packing_budget_exceeded"] if (
                trial is None
            ) else list(trial.reasons)
            return (), measure, receipt
        delta = trial.raw_prompt_tokens - measure.raw_prompt_tokens
        receipt.update(
            status="admitted",
            incremental_raw_tokens=delta,
            extra_bytes=delta * self.kv_bytes_per_token,
            derived_workspace_message_count=1,
            source_already_fully_raw_visible=set(selected["source_indices"])
            <= set(measure.memory.raw_source_indices),
            source_already_fully_gist_backed=selected["event_id"]
            in set(measure.memory.view.gist_event_ids),
        )
        return (cue,), trial, receipt

    def _dependency_packets(
        self, store, tools, raw, mandatory, gist, eligible, common_tokens,
        max_new_tokens, measure,
    ):
        from .dependencies import build_dependency_workspace

        remaining_history = max(0, (
            min(
                self.policy_config.history_budget_bytes,
                self.policy_config.workspace_budget_bytes,
            ) - self._max_history_bytes(measure)
        ) // self.kv_bytes_per_token)
        remaining_workspace = max(
            0, self.packing.max_workspace_tokens - len(measure.memory.workspace_input_ids)
        )
        packet_budget = min(remaining_history, remaining_workspace)
        packets, receipt = build_dependency_workspace(
            store, tools,
            token_counter=lambda messages: self._count(messages, ()),
            token_budget=packet_budget,
        )
        accepted: list[dict[str, Any]] = []
        skipped: list[int] = []
        for index, packet in enumerate(packets):
            trial = self._try_measure(
                store, tools, raw, mandatory, gist, eligible, common_tokens,
                max_new_tokens, derived_messages=(*accepted, packet),
            )
            if trial is None or trial.reasons:
                skipped.append(index)
                continue
            accepted.append(packet)
            measure = trial
        receipt = copy.deepcopy(receipt)
        receipt["b0_admitted_packet_count"] = len(accepted)
        receipt["b0_skipped_packet_indices"] = skipped
        receipt["b0_rechecked"] = True
        receipt["b0_admitted_source_indices"] = (
            list(receipt.get("source_indices") or ()) if accepted else []
        )
        receipt["b0_admitted_group_ids"] = (
            [row["group_id"] for row in receipt.get("selected_groups") or ()]
            if accepted else []
        )
        if skipped and not accepted:
            receipt["status"] = "b0_rejected"
        elif skipped:
            receipt["status"] = "partially_admitted"
        return tuple(accepted), measure, receipt

    def _build_prepared(
        self, *, store, tools, ratio, max_new_tokens, decision_key,
        decision_index, cutoff, task_packet, task_sources, task_raw_tokens,
        common_sources, common_tokens, scope, eligible, eligible_sources,
        chunks, mandatory, measure, derived, current_turn_raw,
        skipped_current, skipped_gist, failed_receipt, dependency_receipt,
    ) -> PreparedEventNativeS0:
        view = measure.memory.view
        raw = set(view.raw_event_ids)
        gist = set(view.gist_event_ids)
        if raw & gist:
            raise AssertionError("Initial candidate raw and gist events overlap")
        coverage, packing_fragments, retained_fragments = self._coverage(
            store, eligible, eligible_sources, chunks, measure.memory
        )
        full = self._full_reference(
            store, tools, common_tokens, max_new_tokens
        )
        ratio_cost = measure.per_ratio[str(ratio)]
        omitted_eligible = [
            event_id for event_id in eligible if event_id not in raw | gist
        ]
        eligible_units = list(dict.fromkeys(chunk.event_id for chunk in chunks))
        retained_units = list(dict.fromkeys(
            chunk.event_id for chunk in measure.memory.chunks
        ))
        route_mode = f"candidate_{self.variant}_native_v1"
        effective_derived = self._merge_protected_derived(derived)
        metadata = {
            "candidate_algorithm": self.variant,
            "candidate_allocation_version": CANDIDATE_ALLOCATION_VERSION,
            "event_native_policy_version": EVENT_NATIVE_POLICY_VERSION,
            "policy_source_commit": POLICY_SOURCE_COMMIT,
            "session_id": store.session_id,
            "benchmark": self.benchmark,
            "decision_key": decision_key,
            "decision_index": decision_index,
            "view_mode": route_mode,
            "mode": route_mode,
            "route_mode": route_mode,
            "route": {
                "view_mode": route_mode,
                "baseline_identity": f"candidate-{self.variant}-native-allocation",
                "recovery_enabled": False,
                "max_generations_per_decision": 1,
                "legacy_1088_equivalent": False,
            },
            "compression_policy": ALWAYS_COMPRESSION_POLICY,
            "implementation_profile": NATIVE_ALWAYS_IMPLEMENTATION_PROFILE,
            "history_view_protocol": "fixed-budget-main",
            "requested_ratio": ratio,
            "max_new_tokens": max_new_tokens,
            "model_context": self.model_context,
            "configured_max_sequence_tokens": self.packing.max_sequence_tokens,
            "chunk_geometry": copy.deepcopy(_FIXED_GEOMETRY),
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "history_budget_bytes": self.policy_config.history_budget_bytes,
            "workspace_budget_bytes": self.policy_config.workspace_budget_bytes,
            "shared_allocation_budget_bytes": min(
                self.policy_config.history_budget_bytes,
                self.policy_config.workspace_budget_bytes,
            ),
            "history_budget_definition": HISTORY_BUDGET_DEFINITION,
            "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
            "current_input_baseline": CURRENT_INPUT_BASELINE,
            "common_input_definition": (
                "source instructions plus raw_source_cutoff suffix plus the "
                "AppWorld first-user task packet"
            ),
            "raw_source_cutoff": cutoff,
            "common_input_source_indices": sorted(common_sources),
            "common_raw_prompt_tokens": common_tokens,
            "raw_source_indices": list(measure.memory.raw_source_indices),
            "derived_workspace_prefix_messages": copy.deepcopy(
                list(effective_derived)
            ),
            "derived_workspace_source_indices": [],
            "raw_event_ids": list(view.raw_event_ids),
            "gist_event_ids": list(view.gist_event_ids),
            "omitted_event_ids": list(view.omitted_event_ids),
            "mandatory_raw_event_ids": list(view.mandatory_raw_event_ids),
            "evidence_event_ids": [],
            "raw_evidence_event_ids": [],
            "selected_event_ids": [
                event.event_id for event in store.events if event.event_id in raw
            ],
            "selected_source_indices": list(measure.memory.raw_source_indices),
            "protected_event_ids": [
                event.event_id for event in store.events
                if event.event_id in mandatory
            ],
            "retrieved_event_ids": [],
            "retained_event_ids": [],
            "recency_selected_event_ids": list(current_turn_raw),
            "lease_decisions": 0,
            "max_retrieved_events": 2,
            "pre_draft_retrieval": False,
            "recovery_stage": "pre_generation_candidate_allocation",
            "post_draft_exact_recovery_applied": False,
            "source_needs": {
                "strategy": "none",
                "history_representation": "c2kv-native-event",
                "prediction_status": "not_run",
                "candidate_event_ids": list(eligible),
                "requested_event_ids": [],
                "typed_needs": [],
                "admitted_event_ids": [],
                "skipped_for_budget": [],
                "predictor_calls": 0,
                "admission_rule": "candidate_variant_native_first_view",
            },
            "latest_complete_tool_protection": {
                "status": "static_select_view"
                if self.variant != "turn_c1" else "current_turn_priority",
            },
            "gist_reservation": {
                "reserved_event_id": None,
                "reserved_event_ids": [],
                "satisfied": True,
            },
            "gist_refill_priority_event_ids": list(view.gist_event_ids),
            "gist_refilled_event_ids": list(view.gist_event_ids),
            "skipped_gist_refill_events": skipped_gist,
            "raw_reserve": {"status": "disabled_for_variant"},
            "failed_operation_cue": failed_receipt,
            "current_turn_raw_event_ids": list(current_turn_raw),
            "skipped_current_turn_raw_events": skipped_current,
            "source_coverage": coverage,
            "eligible_extraction": {
                "status": "planned_for_pre_generation_extraction",
                "source": "preceding observable EventStore prefix",
                "eligible_event_ids": list(eligible),
                "encoding_scope": scope.scope,
                "encoding_event_groups": [list(group) for group in scope.event_groups],
                "pending_raw_event_ids": list(scope.pending_event_ids),
                "eligible_encoder_unit_ids": eligible_units,
                "eligible_source_indices": sorted(eligible_sources),
                "whole_event_encoded_source_indices": sorted({
                    index for chunk in chunks for index in chunk.source_indices
                }),
                "eligible_chunk_count": len(chunks),
                "eligible_presented_encoder_tokens": sum(
                    len(chunk.token_ids) for chunk in chunks
                ),
                "eligible_unique_encoder_tokens": self._unique_encoder_tokens(chunks),
                "retained_event_ids": list(view.gist_event_ids),
                "retained_encoder_unit_ids": retained_units,
                "retained_chunk_count": len(measure.memory.chunks),
                "retained_presented_encoder_tokens": sum(
                    len(chunk.token_ids) for chunk in measure.memory.chunks
                ),
                "omitted_active_gist_event_ids": omitted_eligible,
                "raw_gist_overlap_event_ids": [],
                "backend_execution_required": bool(chunks),
                "cache_reuse_known_after_generation": True,
                "charged_separately_from_active_history_bytes": True,
            },
            "history_packing_fragments": packing_fragments,
            "retained_history_packing_fragments": retained_fragments,
            "per_ratio": copy.deepcopy(measure.per_ratio),
            "raw_prompt_tokens": measure.raw_prompt_tokens,
            "actual_raw_history_tokens": measure.raw_history_tokens,
            "actual_gist_tokens": ratio_cost["history_gist_tokens"],
            "actual_history_bytes": ratio_cost["history_bytes"],
            "actual_managed_history_tokens": ratio_cost["managed_history_tokens"],
            "actual_managed_history_bytes": ratio_cost["managed_history_bytes"],
            "actual_total_resident_kv_tokens": ratio_cost["total_resident_kv_tokens"],
            "actual_total_resident_kv_bytes": ratio_cost["total_resident_kv_bytes"],
            "logical_sequence_tokens": measure.logical_sequence_tokens,
            "same_prefix_full_reference": full,
            "compression_ratio": self._compression_ratio(
                full, measure, coverage, ratio
            ),
            "no_eligible_history": not eligible,
            "full_source_coverage": coverage["complete_history_coverage"],
            "encoding_scope": scope.scope,
            "pending_encoding_event_ids": list(scope.pending_event_ids),
            "atomic_packing_unit": (
                "whole_event_all_encoder_chunks"
                if scope.scope == "current" else "complete_event_group"
            ),
            "min_gist_reservation_required": False,
            "min_gist_reservation_met": None,
            "legacy_1088_block_parity": False,
            "task_packet_protection": (
                "first_non_system_user_raw" if task_packet else "none"
            ),
            "task_packet_event_id": task_packet.event_id if task_packet else None,
            "task_packet_source_indices": sorted(task_sources),
            "task_packet_raw_tokens": task_raw_tokens,
            "task_packet_raw_bytes": task_raw_tokens * self.kv_bytes_per_token,
            "task_packet_accounting": {
                "token_definition": (
                    "marginal tokens contributed by the AppWorld first-user task "
                    "packet within the rendered common input"
                ),
                "charged_to_history_budget": False,
                "charged_to_workspace_budget": False,
                "included_in_total_resident_kv": bool(task_sources),
            },
        }
        if dependency_receipt is not None:
            metadata["dependency_packet"] = dependency_receipt
        return PreparedEventNativeS0(
            memory=measure.memory,
            metadata=metadata,
            eligible_chunks=chunks,
            _owner=self._owner,
            _session_id=store.session_id,
            _decision_key=decision_key,
        )
