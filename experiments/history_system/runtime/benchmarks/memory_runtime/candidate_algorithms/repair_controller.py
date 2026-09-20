"""One source-backed proposal, measured admission, and explicit commit review."""
from __future__ import annotations

import copy
import hashlib

from . import REPAIR_VARIANTS
from .controller import memory_signature
from .observations import current_request, operation_records
from .repair_protocol import REPAIR_VERSION, RepairContext, policy_for
from .repacking import repack
from ..policy import PolicyInputError
from ..recovery.config import E1_RECOVERY_VERSION
from ..recovery.gate import canonical
from ..recovery.orchestrator import EventNativeRecoveryController


class RepairController(EventNativeRecoveryController):
    stable_call_ids = True
    max_recovery_rounds = 1

    def __init__(self, base, config, *, policy=None):
        variant = config.get("variant")
        if variant not in REPAIR_VARIANTS:
            raise ValueError("Unknown source repair variant")
        super().__init__(base, {"schema": E1_RECOVERY_VERSION, "gate": "disabled"},
                         benchmark=base.benchmark)
        self.variant = variant
        self.policy = policy if policy is not None else policy_for(variant)
        self.base.preserve_candidate_derived_messages = True
        self._reviewed = set()

    def prepare(self, payload, *, ratio, max_new_tokens):
        if ratio != 8:
            raise ValueError("Source repair candidates require ratio8")
        prepared = super().prepare(payload, ratio=ratio, max_new_tokens=max_new_tokens)
        if not hasattr(prepared, "_repair_signature"):
            if self.benchmark == "acebench":
                from ..acebench_source import build_ace_event_store
                prepared._store = build_ace_event_store(
                    payload["session_id"], payload["messages"], payload["c2kv_ace_source"])
            prepared._repair_signature = None
            prepared._repair_context = None
            prepared._repair_proposal = None
            prepared.metadata.pop("post_draft_recovery_config", None)
            prepared.metadata["candidate_algorithm"] = {
                "schema": REPAIR_VERSION, "variant": self.variant,
                "gate_order": "source_evidence_then_admission_then_commit",
                "detector": "policy_evidence", "risk_threshold": None,
                "one_regeneration_per_decision": True, "cumulative_quota": False,
                "stable_call_ids": True, "commit_validation": True,
            }
            prepared.metadata["route"].update(
                baseline_identity=REPAIR_VERSION + ":" + self.variant,
                recovery_enabled=True, max_generations_per_decision=2)
        return prepared

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error=None):
        key = (prepared._store.session_id, prepared.metadata["decision_key"])
        if self._prepared.get(key) is not prepared:
            raise PolicyInputError("Repair decision belongs to another controller")
        signature = canonical([draft_tool_calls, draft_text, parse_error])
        if prepared._checked_result is not None:
            if signature != prepared._repair_signature:
                raise PolicyInputError("Repair decision cannot inspect another held draft")
            return copy.deepcopy(prepared._checked_result)
        prepared._repair_signature = signature
        base_result = self.base.reconsider(prepared._base_prepared, draft_tool_calls,
                                          draft_text=draft_text, parse_error=parse_error)
        decision = {
            "version": REPAIR_VERSION, "variant": self.variant, "status": "abstain",
            "reason": None, "regeneration_allowed": False, "upgrade_count": 0,
            "decision_index": prepared.metadata["decision_index"],
            "selection": {"selector": "policy_evidence", "selected_ids": []},
            "limits": {"one_regeneration_per_decision": True,
                       "task_generation_limit": self.required_task_generation_limit,
                       "online_cumulative_quota_applied": False},
        }

        def finish(reason, measure=None, metadata=None):
            decision["reason"] = reason
            if measure is None:
                result = copy.deepcopy(base_result)
                result["regenerate"] = False
            else:
                decision.update(status="recover", regeneration_allowed=True, upgrade_count=1)
                self._recovery_counts[key[0]] = self._recovery_counts.get(key[0], 0) + 1
                result = {"regenerate": True, "memory": measure.memory, "metadata": metadata}
            result["metadata"]["candidate_algorithm"] = copy.deepcopy(prepared.metadata["candidate_algorithm"])
            result["metadata"]["route"] = copy.deepcopy(prepared.metadata["route"])
            result["decision"] = copy.deepcopy(decision)
            result["metadata"]["exact_recovery"] = copy.deepcopy(decision)
            prepared._checked_result = copy.deepcopy(result)
            return result

        if decision["decision_index"] + self._recovery_counts.get(key[0], 0) + 1 > self.required_task_generation_limit:
            return finish("shared_task_generation_limit")
        budget = min(self.policy_config.history_budget_bytes,
                     self.policy_config.workspace_budget_bytes) // self.kv_bytes_per_token
        context = RepairContext(prepared, tuple(copy.deepcopy(draft_tool_calls)), draft_text,
                                parse_error, lambda rows: self.base._count(rows, ()), budget)
        proposal = self.policy.propose(context)
        if proposal is None:
            return finish("no_source_supported_repair")
        prepared._repair_context = context
        prepared._repair_proposal = proposal
        decision["proposal"] = copy.deepcopy(proposal.receipt)
        decision["guard"] = copy.deepcopy(proposal.guard)
        if not proposal.messages or context.token_counter(proposal.messages) > budget:
            return finish("repair_packet_exceeds_budget")
        # Ignore prose-only draft variation; calls and the proposal's exact
        # source packet define the observable action/observation state.
        semantic_calls = [call.get("function", {}) for call in draft_tool_calls]
        goal, _ = current_request(prepared._store)
        source_state = {
            "request": ([message.to_dict() for message in prepared._store.event_messages(goal.event_id)]
                        if goal is not None else []),
            "request_event_id": goal.event_id if goal is not None else None,
            "observations": [(row.event_id, row.result_source_index,
                              row.call_signature_id, row.observed_result)
                             for row in operation_records(prepared._store)],
            "tools": prepared._tools,
        }
        state = hashlib.sha256(canonical([source_state, semantic_calls, parse_error]).encode()).hexdigest()
        decision["proposal_state_sha256"] = state
        if (key[0], state) in self._reviewed:
            return finish("source_action_state_already_reviewed")
        measure, metadata, allocation = repack(
            self.base, prepared, derived_messages=proposal.messages,
            goal_view=self.variant == "request_contract")
        decision["allocation"] = allocation
        if measure is None or memory_signature(measure.memory) == memory_signature(prepared.memory):
            return finish("no_feasible_new_repair_view")
        self._reviewed.add((key[0], state))
        return finish(proposal.reason, measure, metadata)

    def validate_commit(self, prepared, candidate_calls, *, draft_text, parse_error=None):
        if prepared._repair_proposal is None:
            return {"schema": REPAIR_VERSION, "variant": self.variant,
                    "accepted": True, "reason": "no_source_supported_guard",
                    "fallback": "original"}
        verdict = self.policy.validate(
            prepared._repair_context, prepared._repair_proposal, candidate_calls,
            draft_text=draft_text, parse_error=parse_error)
        if verdict.fallback not in {"original", "stop"}:
            raise PolicyInputError("Unknown repair commit fallback")
        return {"schema": REPAIR_VERSION, "variant": self.variant,
                "accepted": verdict.accepted, "reason": verdict.reason,
                "fallback": verdict.fallback}
