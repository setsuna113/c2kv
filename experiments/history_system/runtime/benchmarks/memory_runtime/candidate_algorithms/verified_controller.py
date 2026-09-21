"""Opt-in, proof-checked field commits after the original Goal abstains."""
from __future__ import annotations

import copy

from . import VERIFIED_VARIANTS, VERIFIED_VERSION
from .goal_controller import GoalCompositionController
from .verified_binding import Policy, PROOF_REGISTRY_VERSION
from ..policy import PolicyInputError
from ..recovery.gate import canonical


class VerifiedBindingController(GoalCompositionController):
    """Reuse Goal/Pending generation verbatim; verify a separate field proposal.

    This controller never imports the historical Source or Progress policies.
    Proof discovery and verification use only the observed prefix, and consume
    no generation slot or additional model workspace.
    """

    def __init__(self, base, config, *, risk_model=None, binding_policy=None):
        if config.get("variant") not in VERIFIED_VARIANTS:
            raise ValueError("Unknown verified-binding variant")
        if config.get("proof_registry_version", PROOF_REGISTRY_VERSION) != PROOF_REGISTRY_VERSION:
            raise ValueError("Verified binding proof registry version mismatch")
        super().__init__(base, config, risk_model=risk_model, policies=())
        self.pending_enabled = self.goal_variant == "pending_verified"
        self.binding_policy = Policy() if binding_policy is None else binding_policy

    def _identity(self, metadata):
        metadata["candidate_algorithm"].update(
            schema=VERIFIED_VERSION, variant=self.goal_variant,
            backbone="goal_pending" if self.pending_enabled else "goal_rescue",
            gate_order="goal_then_verified_binding",
            proof_registry_version=PROOF_REGISTRY_VERSION,
            binding_uses_model_generation=False,
            binding_requires_goal_abstention=True,
        )
        metadata["route"]["baseline_identity"] = VERIFIED_VERSION + ":" + self.goal_variant

    def prepare(self, payload, *, ratio, max_new_tokens):
        prepared = super().prepare(payload, ratio=ratio, max_new_tokens=max_new_tokens)
        if not hasattr(prepared, "_binding_result"):
            prepared._binding_result = None
            prepared._binding_proposal = None
            prepared._binding_commit_accepted = False
            prepared._binding_commit_signature = None
            prepared._binding_finalized = None
            self._identity(prepared.metadata)
        return prepared

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error=None):
        # The parent checks decision ownership and rejects another held draft.
        original = super().reconsider(
            prepared, draft_tool_calls, draft_text=draft_text, parse_error=parse_error)
        if prepared._binding_result is not None:
            return copy.deepcopy(prepared._binding_result)
        result = copy.deepcopy(original)
        decision = result["decision"]
        decision.update(version=VERIFIED_VERSION, variant=self.goal_variant)
        binding = {"version": PROOF_REGISTRY_VERSION, "status": "not_applicable",
                   "additional_generations": 0, "additional_model_workspace_tokens": 0}
        if result["regenerate"] or decision["backbone"]["status"] != "abstain":
            binding["status"] = "original_goal_recovery_preserved"
        elif decision["reason"] == "shared_task_generation_limit":
            binding["status"] = "task_generation_limit_preserved"
        elif parse_error is not None:
            binding["status"] = "malformed_draft_preserved"
        elif draft_tool_calls:
            proposal = self.binding_policy.propose(prepared._goal_context)
            prepared._binding_proposal = proposal
            if proposal is not None:
                binding.update(status="proposed", proposal=proposal.to_receipt())
            else:
                binding["status"] = "no_verified_binding"
        decision["verified_binding"] = binding
        self._identity(result["metadata"])
        result["metadata"]["exact_recovery"] = copy.deepcopy(decision)
        prepared._binding_result = copy.deepcopy(result)
        # Preserve the original parent result for its held-draft checks. The
        # separate cache prevents proposing twice on an idempotent retry.
        return result

    def validate_commit(self, prepared, candidate_calls, *, draft_text, parse_error=None):
        verdict = super().validate_commit(
            prepared, candidate_calls, draft_text=draft_text, parse_error=parse_error)
        verdict.update(schema=VERIFIED_VERSION, variant=self.goal_variant)
        prepared._binding_commit_accepted = bool(verdict["accepted"] and parse_error is None)
        prepared._binding_commit_signature = canonical(candidate_calls)
        return verdict

    def finalize_commit(self, prepared, candidate_calls):
        original = tuple(copy.deepcopy(candidate_calls))
        signature = canonical(original)
        if prepared._binding_finalized is not None:
            prior_signature, output = prepared._binding_finalized
            if signature != prior_signature:
                raise PolicyInputError("Verified binding cannot finalize another selected draft")
            return copy.deepcopy(output)
        receipt = {"schema": VERIFIED_VERSION, "variant": self.goal_variant,
                   "proof_registry_version": PROOF_REGISTRY_VERSION,
                   "changed": False, "status": "no_verified_proposal",
                   "additional_generations": 0, "additional_model_workspace_tokens": 0}
        result = prepared._binding_result
        proposal = prepared._binding_proposal
        output = original
        if result is None:
            receipt["status"] = "decision_not_reviewed"
        elif result["regenerate"] or result["decision"]["backbone"]["status"] != "abstain":
            receipt["status"] = "original_goal_recovery_preserved"
        elif not prepared._binding_commit_accepted:
            receipt["status"] = "selected_commit_not_accepted"
        elif signature != prepared._binding_commit_signature:
            receipt["status"] = "selected_commit_changed"
        elif proposal is not None:
            corrected, proof_receipt = self.binding_policy.apply(
                prepared._goal_context, proposal, original)
            verdict = self.binding_policy.validate(prepared._goal_context, proposal, corrected)
            receipt.update(proposal=proposal.to_receipt(), verification=copy.deepcopy(proof_receipt),
                           accepted=verdict.accepted, reason=verdict.reason)
            if verdict.accepted:
                output = tuple(copy.deepcopy(corrected))
                receipt.update(status="verified_binding_committed",
                               changed=canonical(output) != signature)
            else:
                receipt["status"] = "binding_verification_rejected"
        finalized = (output, receipt)
        prepared._binding_finalized = (signature, copy.deepcopy(finalized))
        return finalized
