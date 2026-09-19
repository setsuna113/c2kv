"""Opt-in T02-gated candidate controllers with one bounded regeneration."""
from __future__ import annotations

import copy
import hashlib
import math

from . import VARIANTS, VERSION
from .progress import review_request, review_messages
from .repacking import repack
from ..policy import PolicyInputError
from ..recovery.config import E1_RECOVERY_VERSION
from ..recovery.orchestrator import EventNativeRecoveryController
from ..recovery.gate import canonical
from ..recovery.set_models import C1RiskArtifact
from ..recovery.set_protocol import context_from_prepared
from ..recovery.source import select_source_event, source_event_receipt


def memory_signature(memory):
    return hashlib.sha256(canonical({
        "system": list(memory.system_input_ids), "workspace": list(memory.workspace_input_ids),
        "chunks": [(chunk.event_id, list(chunk.token_ids)) for chunk in memory.chunks],
        "raw": list(memory.raw_source_indices),
    }).encode()).hexdigest()


class CandidateRecoveryController(EventNativeRecoveryController):
    stable_call_ids = True
    max_recovery_rounds = 1

    def __init__(self, base, config, *, risk_model=None):
        variant = config.get("variant")
        if variant not in VARIANTS:
            raise ValueError("Unknown candidate algorithm")
        threshold = config.get("risk_threshold", 0.5)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("risk_threshold must be a finite probability")
        # Reuse only the legacy lifecycle, not its gate or cumulative quota.
        super().__init__(base, {"schema": E1_RECOVERY_VERSION, "gate": "disabled"},
                         benchmark=base.benchmark)
        self.variant = variant
        # Keep legacy recovery behavior unchanged while carrying the actual
        # C1 bridge into candidate-only replacement views.
        self.base.preserve_candidate_derived_messages = True
        self.threshold = float(threshold)
        self.risk_model = risk_model if risk_model is not None else C1RiskArtifact(config["risk_artifact"])
        self._reviewed_states = set()

    def prepare(self, payload, *, ratio, max_new_tokens):
        if ratio != 8:
            raise ValueError("Paper candidates require c2kv ratio8")
        prepared = super().prepare(payload, ratio=ratio, max_new_tokens=max_new_tokens)
        if not hasattr(prepared, "_candidate_signature"):
            prepared._candidate_signature = None
            prepared._set_draft_logprobs = ()
            prepared.metadata.pop("post_draft_recovery_config", None)
            prepared.metadata["candidate_algorithm"] = {
                "schema": VERSION, "variant": self.variant, "risk_threshold": self.threshold,
                "gate_order": "before_retrieval", "one_regeneration_per_decision": True,
                "cumulative_quota": False, "stable_call_ids": True,
            }
            prepared.metadata["route"].update(
                baseline_identity=VERSION + ":" + self.variant,
                recovery_enabled=True, max_generations_per_decision=2)
        return prepared

    def observe_selection_draft(self, *, session_id, decision_key, token_logprobs):
        prepared = self._prepared[(session_id, decision_key)]
        if prepared._checked_result is None:
            prepared._set_draft_logprobs = tuple(token_logprobs)

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error=None):
        key = (prepared._store.session_id, prepared.metadata["decision_key"])
        if self._prepared.get(key) is not prepared:
            raise PolicyInputError("Candidate decision belongs to another controller")
        signature = canonical([draft_tool_calls, draft_text, parse_error])
        if prepared._checked_result is not None:
            if signature != prepared._candidate_signature:
                raise PolicyInputError("Candidate decision cannot inspect another held draft")
            return copy.deepcopy(prepared._checked_result)
        prepared._candidate_signature = signature
        base_result = self.base.reconsider(prepared._base_prepared, draft_tool_calls,
                                          draft_text=draft_text, parse_error=parse_error)
        decision = {"version": VERSION, "variant": self.variant, "status": "abstain",
                    "reason": None, "regeneration_allowed": False, "upgrade_count": 0,
                    "decision_index": prepared.metadata["decision_index"],
                    "limits": {"one_regeneration_per_decision": True,
                               "task_generation_limit": self.required_task_generation_limit,
                               "online_cumulative_quota_applied": False}}

        def finish(reason, measure=None, metadata=None):
            decision["reason"] = reason
            if measure is None:
                result = copy.deepcopy(base_result)
                result["regenerate"] = False
                result["metadata"]["candidate_algorithm"] = copy.deepcopy(prepared.metadata["candidate_algorithm"])
                result["metadata"]["route"] = copy.deepcopy(prepared.metadata["route"])
            else:
                decision.update(status="recover", regeneration_allowed=True, upgrade_count=1)
                self._recovery_counts[key[0]] = self._recovery_counts.get(key[0], 0) + 1
                result = {"regenerate": True, "memory": measure.memory, "metadata": metadata}
            result["decision"] = copy.deepcopy(decision)
            result["metadata"]["exact_recovery"] = copy.deepcopy(decision)
            prepared._checked_result = copy.deepcopy(result)
            return result

        if decision["decision_index"] + self._recovery_counts.get(key[0], 0) + 1 > self.required_task_generation_limit:
            return finish("shared_task_generation_limit")
        context = context_from_prepared(prepared, draft_tool_calls, draft_text, parse_error)
        prediction = self.risk_model.predict_risk(context)
        decision["selection"] = {
            "selector": "risk", "score_semantics": "current_turn_failure_risk",
            "available": prediction.available, "score": prediction.score,
            "reason": prediction.reason, "selected_ids": [],
        }
        if not prediction.available or prediction.score is None:
            raise PolicyInputError("Candidate T02 risk unavailable: " + str(prediction.reason))
        triggered = prediction.score > self.threshold
        decision["gate"] = {"type": "risk", "score": prediction.score,
                            "threshold": self.threshold, "triggered": triggered,
                            "reason": "risk_triggered" if triggered else "risk_not_above_threshold"}
        review_reason, review_key, goal, records = (None, None, None, None)
        if self.variant == "goal_rescue":
            review_reason, review_key, goal, records = review_request(
                prepared, draft_tool_calls, parse_error=parse_error)
            state_key = (key[0], review_key)
            if state_key in self._reviewed_states:
                review_reason = None
                decision["goal_review"] = {"status": "observation_state_already_reviewed"}
        if review_reason:
            budget = min(self.policy_config.history_budget_bytes,
                         self.policy_config.workspace_budget_bytes) // self.kv_bytes_per_token
            messages, receipt = review_messages(
                prepared._store, goal, records,
                token_counter=lambda rows: self.base._count(rows, ()), token_budget=budget)
            decision["goal_review"] = {**receipt, "trigger": review_reason,
                                       "state_sha256": review_key, "stop_is_error_label": False}
            if messages:
                measure, metadata, allocation = repack(self.base, prepared,
                    derived_messages=messages, goal_view=True)
                decision["allocation"] = allocation
                if measure is not None and memory_signature(measure.memory) != memory_signature(prepared.memory):
                    self._reviewed_states.add(state_key)
                    decision["goal_review"]["status"] = "admitted"
                    return finish(review_reason, measure, metadata)
        if not triggered:
            return finish("risk_not_above_threshold")
        _, source = select_source_event(prepared, draft_tool_calls, draft_text=draft_text,
            include_latest_complete_observation=True, explicit_revision_abstain=False,
            allow_empty_draft_query=True)
        decision["source"] = source
        trials = []
        for candidate in source["ranked_candidate_event_ids"]:
            measure, metadata, allocation = repack(self.base, prepared, candidate=candidate)
            trials.append(allocation)
            if measure is None:
                continue
            if memory_signature(measure.memory) == memory_signature(prepared.memory):
                continue
            decision.update(candidate_event_id=candidate, upgraded_event_id=candidate,
                            allocation=allocation, candidate_trials=trials)
            restored = source_event_receipt(prepared, candidate)
            restored["representation"] = "native_raw_event"
            decision["restored_event"] = restored
            decision["selection"]["selected_ids"] = [candidate]
            return finish("risk_triggered_complete_event_replaced", measure, metadata)
        decision["candidate_trials"] = trials
        return finish("no_feasible_new_complete_event")


def wrap_with_candidate_recovery(base, config):
    from . import REPAIR_VARIANTS
    if config.get("variant") in REPAIR_VARIANTS:
        from .repair_controller import RepairController
        return RepairController(base, config)
    return CandidateRecoveryController(base, config)
