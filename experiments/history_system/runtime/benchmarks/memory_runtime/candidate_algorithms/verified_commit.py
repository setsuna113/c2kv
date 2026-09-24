"""Proof-based commit extension independent of completion-review controllers."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .verified_binding import Policy, PROOF_REGISTRY_VERSION
from ..policy import PolicyInputError
from ..recovery.gate import canonical


@dataclass
class VerifiedCommitState:
    context: Any
    eligible: bool
    proposal: Any
    receipt: dict
    accepted: bool = False
    selected_signature: str | None = None
    finalized: Any = None


class VerifiedCommitPolicy:
    """Keep the frozen Verified proof rules and recovery-abstention contract."""

    def __init__(self, policy=None):
        self.policy = Policy() if policy is None else policy

    def inspect(self, context, recovery):
        receipt = {"version": PROOF_REGISTRY_VERSION, "status": "not_applicable",
                   "additional_generations": 0, "additional_model_workspace_tokens": 0}
        decision = recovery["decision"]
        proposal = None
        eligible = False
        if recovery["regenerate"] or decision["status"] != "abstain":
            receipt["status"] = "event_recovery_preserved"
        elif decision["reason"] == "shared_task_generation_limit":
            receipt["status"] = "task_generation_limit_preserved"
        elif context.parse_error is not None:
            receipt["status"] = "malformed_draft_preserved"
        elif context.draft_tool_calls:
            eligible = True
            proposal = self.policy.propose(context)
            receipt["status"] = "proposed" if proposal is not None else "no_verified_binding"
            if proposal is not None:
                receipt["proposal"] = proposal.to_receipt()
        return VerifiedCommitState(context, eligible, proposal, receipt)

    def validate(self, state, calls, *, parse_error=None):
        if state.finalized is not None and canonical(calls) != state.selected_signature:
            raise PolicyInputError("Verified commit cannot validate another finalized selection")
        state.accepted = parse_error is None
        state.selected_signature = canonical(calls)
        return {"accepted": state.accepted, "fallback": "original",
                "reason": "original_commit" if state.accepted else "malformed_selected_draft"}

    def finalize(self, state, calls):
        original = tuple(copy.deepcopy(calls))
        signature = canonical(original)
        if state.finalized is not None:
            previous, finalized = state.finalized
            if previous != signature:
                raise PolicyInputError("Verified commit cannot finalize another selected draft")
            return copy.deepcopy(finalized)
        output = original
        receipt = {"proof_registry_version": PROOF_REGISTRY_VERSION,
                   "changed": False, "status": "no_verified_proposal",
                   "additional_generations": 0, "additional_model_workspace_tokens": 0}
        if not state.eligible:
            receipt["status"] = state.receipt["status"]
        elif not state.accepted:
            receipt["status"] = "selected_commit_not_accepted"
        elif signature != state.selected_signature:
            receipt["status"] = "selected_commit_changed"
        elif state.proposal is not None:
            corrected, proof = self.policy.apply(state.context, state.proposal, original)
            verdict = self.policy.validate(state.context, state.proposal, corrected)
            receipt.update(proposal=state.proposal.to_receipt(), verification=copy.deepcopy(proof),
                           accepted=verdict.accepted, reason=verdict.reason)
            if verdict.accepted:
                output = tuple(copy.deepcopy(corrected))
                receipt.update(status="verified_binding_committed", changed=canonical(output) != signature)
            else:
                receipt["status"] = "binding_verification_rejected"
        finalized = (output, receipt)
        state.finalized = (signature, copy.deepcopy(finalized))
        return finalized


class CoreCommitPolicy(VerifiedCommitPolicy):
    """Keep selected-generation validation; never propose an argument correction."""

    def inspect(self, context, recovery):
        receipt = {"version": PROOF_REGISTRY_VERSION, "status": "argument_correction_disabled_core",
                   "additional_generations": 0, "additional_model_workspace_tokens": 0}
        return VerifiedCommitState(context, False, None, receipt)
