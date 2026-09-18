"""Candidate-first D3 recovery on the native S0 raw-event allocator."""

from __future__ import annotations

import copy
import time
from typing import Any

from ..policy import PolicyInputError
from .config import public_recovery_config
from .gate import canonical
from .orchestrator import (
    EventNativeRecoveryController,
    PreparedEventNativeRecovery,
    _copy_result,
    _decision,
)
from .source import select_source_event, source_event_receipt, valid_tool_calls


D3_HYBRID_RECOVERY_VERSION = "a-d3-hybrid-post-draft-event-recovery-v1"


class D3HybridRecoveryController(EventNativeRecoveryController):
    """Use C1 ordering/query semantics with D3 complete-event B0 repacking.

    Candidate ranking and all B0 trials happen before the frozen Prefill gate,
    but remain side-effect free.  Only a feasible candidate that passes the
    original detector and shared task-generation limit is committed.  The
    legacy cumulative one-fifth recovery quota is deliberately not applied.
    """

    def prepare(self, payload, *, ratio, max_new_tokens):
        prepared = super().prepare(
            payload, ratio=ratio, max_new_tokens=max_new_tokens
        )
        if hasattr(prepared, "_d3_hybrid_signature"):
            return prepared

        self._apply_hybrid_identity(
            prepared.metadata, prepared._base_prepared.metadata
        )
        prepared.metadata["d3_hybrid_recovery"] = self._public_hybrid_config()
        prepared._d3_hybrid_signature = None
        prepared._d3_hybrid_base_result = None
        return prepared

    def reconsider(
        self,
        prepared: PreparedEventNativeRecovery,
        draft_tool_calls: Any,
        *,
        draft_text: str,
        parse_error: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(prepared, PreparedEventNativeRecovery):
            raise PolicyInputError(
                "Prepared decision belongs to another D3 hybrid controller"
            )
        key = (prepared._store.session_id, prepared.metadata["decision_key"])
        if self._prepared.get(key) is not prepared:
            raise PolicyInputError(
                "Prepared decision belongs to another D3 hybrid controller"
            )
        if not isinstance(draft_text, str):
            raise TypeError("draft_text must be a string")

        signature = canonical([draft_tool_calls, draft_text, parse_error])
        if prepared._checked_result is not None:
            if signature != prepared._d3_hybrid_signature:
                raise PolicyInputError(
                    "A recovery decision cannot inspect two different drafts"
                )
            return _copy_result(prepared._checked_result)

        prepared._d3_hybrid_signature = signature
        if prepared._d3_hybrid_base_result is None:
            prepared._d3_hybrid_base_result = self.base.reconsider(
                prepared._base_prepared,
                draft_tool_calls,
                draft_text=draft_text,
                parse_error=parse_error,
            )
        base_result = prepared._d3_hybrid_base_result

        decision_index = prepared.metadata["decision_index"]
        decision = _decision(decision_index, self.config["gate"])
        decision["version"] = D3_HYBRID_RECOVERY_VERSION
        decision["controller_mode"] = "d3_hybrid_candidate_first"
        decision["limits"] = {
            "one_regeneration_per_decision": True,
            "online_cumulative_quota_applied": False,
            "legacy_e1_quota_applied": False,
            "task_generation_limit": self.config["task_generation_limit"],
        }
        phases: list[dict[str, Any]] = []

        def measured(name, operation):
            started = time.perf_counter_ns()
            try:
                return operation()
            finally:
                phases.append(
                    {
                        "phase": name,
                        "duration_ns": time.perf_counter_ns() - started,
                    }
                )

        if self.config["gate"] == "disabled":
            decision.update(status="no_op", reason="recovery_disabled")
            decision["measurement_phases"] = phases
            return self._finish(prepared, base_result, decision)

        _, source = measured(
            "d3_hybrid_complete_event_ranking",
            lambda: self._select_ranked_events(
                prepared, draft_tool_calls, draft_text=draft_text
            ),
        )
        ranked = tuple(source.get("ranked_candidate_event_ids") or ())
        decision["source"] = source

        trials = []
        candidate = None
        admitted = None

        def try_candidates():
            nonlocal candidate, admitted
            for rank, event_id in enumerate(ranked, 1):
                trial = self._admit(prepared, event_id)
                trials.append(
                    {
                        "rank": rank,
                        "event_id": event_id,
                        "admitted": trial["measure"] is not None,
                        "allocation": copy.deepcopy(trial["receipt"]),
                    }
                )
                if trial["measure"] is not None:
                    candidate = event_id
                    admitted = trial
                    break

        measured("d3_hybrid_b0_feasibility", try_candidates)
        decision["candidate_feasibility"] = {
            "policy": "ranked-complete-events-first-feasible-d3-b0-v1",
            "ranked_candidate_count": len(ranked),
            "tried_candidate_count": len(trials),
            "first_feasible_event_id": candidate,
            "trials": trials,
            "state_mutated_before_gate": False,
        }
        decision["candidate_event_id"] = candidate
        if candidate is not None:
            source["selected_event"] = source_event_receipt(prepared, candidate)
            source["reason"] = "first_ranked_event_feasible_under_d3_b0"
        else:
            source["selected_event"] = None
            if ranked:
                source["reason"] = "ranked_events_not_admitted_under_d3_b0"

        calls = valid_tool_calls(draft_tool_calls)
        if not draft_text.strip() and not calls:
            decision.update(
                status="abstain", reason="no_held_draft_text_or_valid_tool_call"
            )
            decision["measurement_phases"] = phases
            return self._finish(prepared, base_result, decision)
        if not ranked:
            decision.update(status="abstain", reason=source["reason"])
            decision["measurement_phases"] = phases
            return self._finish(prepared, base_result, decision)
        if candidate is None or admitted is None:
            decision.update(
                status="abstain", reason="no_ranked_candidate_admitted_under_b0"
            )
            decision["measurement_phases"] = phases
            return self._finish(prepared, base_result, decision)

        gate = measured("d3_hybrid_prefill_detector", lambda: self._gate(prepared))
        decision["gate"] = gate
        if not gate["triggered"]:
            decision.update(status="abstain", reason=gate["reason"])
            decision["measurement_phases"] = phases
            return self._finish(prepared, base_result, decision)

        recovered = self._recovery_counts.get(prepared._store.session_id, 0)
        decision["limits"].update(
            recovery_count_before=recovered,
            projected_generation_calls_with_regeneration=(
                decision_index + recovered + 1
            ),
        )
        if (
            decision_index + recovered + 1
            > self.config["task_generation_limit"]
        ):
            decision.update(
                status="abstain", reason="shared_task_generation_limit"
            )
            decision["measurement_phases"] = phases
            return self._finish(prepared, base_result, decision)

        measure = admitted["measure"]
        allocation = admitted["receipt"]
        metadata = measured(
            "d3_hybrid_commit_metadata",
            lambda: self._metadata_after_admission(
                prepared, measure, candidate, allocation
            ),
        )
        restored_event = self._restored_event_receipt(
            prepared, candidate, measure, allocation
        )
        self._recovery_counts[prepared._store.session_id] = recovered + 1
        decision.update(
            status="recover",
            reason="prefill_gate_triggered_first_feasible_event_admitted",
            allocation=copy.deepcopy(allocation),
            upgrade_count=1,
            upgraded_event_id=candidate,
            regeneration_allowed=True,
            task_recovery_count=recovered + 1,
            restored_event=restored_event,
        )
        decision["measurement_phases"] = phases
        metadata["exact_recovery"] = copy.deepcopy(decision)
        metadata["d3_hybrid_recovery"] = self._public_hybrid_config()
        result = {
            "regenerate": True,
            "memory": measure.memory,
            "metadata": metadata,
            "decision": copy.deepcopy(decision),
        }
        prepared._checked_result = _copy_result(result)
        return result

    def _finish(self, prepared, base_result, decision):
        result = super()._finish(prepared, base_result, decision)
        self._apply_hybrid_identity(
            result["metadata"], prepared._base_prepared.metadata
        )
        result["metadata"]["d3_hybrid_recovery"] = (
            self._public_hybrid_config()
        )
        prepared._checked_result = _copy_result(result)
        return result

    @staticmethod
    def _select_ranked_events(prepared, draft_tool_calls, *, draft_text):
        return select_source_event(
            prepared,
            draft_tool_calls,
            draft_text=draft_text,
            include_latest_complete_observation=True,
            explicit_revision_abstain=False,
            allow_empty_draft_query=True,
        )

    def _public_hybrid_config(self) -> dict[str, Any]:
        detector = public_recovery_config(self.config)
        return {
            "schema": D3_HYBRID_RECOVERY_VERSION,
            "enabled": self.config["gate"] != "disabled",
            "candidate_order": "before_prefill_detector",
            "source_policy": (
                "goal-plus-held-draft-plus-latest-complete-observation-lexical"
            ),
            "explicit_revision_global_abstain": False,
            "revision_cancelled_events_excluded": True,
            "empty_text_and_no_valid_call_abstain": True,
            "candidate_unit": "complete_event",
            "admission_policy": "first-ranked-feasible-d3-b0-repack",
            "presentation": "native_raw_event",
            "online_cumulative_quota_applied": False,
            "legacy_e1_quota_applied": False,
            "one_regeneration_per_decision": True,
            "task_generation_limit": self.config["task_generation_limit"],
            "detector": detector,
        }

    def _apply_hybrid_identity(self, metadata, base_metadata) -> None:
        base_identity = base_metadata["route"]["baseline_identity"]
        metadata["route"] = {
            **copy.deepcopy(metadata["route"]),
            "baseline_identity": (
                base_identity + "+d3-hybrid-post-draft-event-recovery"
            ),
            "recovery_enabled": self.config["gate"] != "disabled",
            "max_generations_per_decision": (
                1 if self.config["gate"] == "disabled" else 2
            ),
        }

    def _restored_event_receipt(
        self, prepared, candidate, measure, allocation
    ) -> dict[str, Any]:
        """Measure the raw event's exact marginal cost in the committed view."""

        store, tools = prepared._store, prepared._tools
        raw = [
            event_id
            for event_id in prepared.memory.view.raw_event_ids
            if event_id not in set(allocation["demoted_raw_event_ids"])
        ]
        gist = [
            event_id
            for event_id in prepared.memory.view.gist_event_ids
            if event_id not in set(allocation["released_gist_event_ids"])
        ]
        mandatory = set(prepared.memory.view.mandatory_raw_event_ids)
        eligible = prepared.metadata["eligible_extraction"]["eligible_event_ids"]
        derived = tuple(
            prepared.metadata.get("derived_workspace_prefix_messages") or ()
        )
        without = self.base._try_measure(
            store,
            tools,
            raw,
            mandatory,
            gist,
            eligible,
            prepared.metadata["common_raw_prompt_tokens"],
            prepared.metadata["max_new_tokens"],
            derived_messages=derived,
        )
        if without is None or without.reasons:
            raise PolicyInputError(
                "D3 hybrid could not measure the admitted event marginal cost"
            )
        ratio = str(prepared.metadata["requested_ratio"])
        source = source_event_receipt(prepared, candidate)
        source.update(
            representation="native_raw_event",
            marginal_raw_prompt_tokens=(
                measure.raw_prompt_tokens - without.raw_prompt_tokens
            ),
            marginal_raw_history_tokens=(
                measure.raw_history_tokens - without.raw_history_tokens
            ),
            marginal_active_history_bytes=(
                measure.per_ratio[ratio]["history_bytes"]
                - without.per_ratio[ratio]["history_bytes"]
            ),
            marginal_measurement=(
                "same-final-demotions-and-gist-releases-without-restored-event"
            ),
        )
        return source


def wrap_with_d3_hybrid_recovery(base, config, *, benchmark=None):
    return D3HybridRecoveryController(base, config, benchmark=benchmark)


__all__ = [
    "D3_HYBRID_RECOVERY_VERSION",
    "D3HybridRecoveryController",
    "wrap_with_d3_hybrid_recovery",
]
