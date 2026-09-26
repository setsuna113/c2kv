"""Shared incumbent-first protection, capacity rescue, T02 and Verified."""
from __future__ import annotations

import copy
import os

from . import C1_V2_VARIANTS, C1_V2_VERSION
from .controller import CandidateRecoveryController
from .repair_protocol import RepairContext
from .verified_binding import PROOF_REGISTRY_VERSION
from .verified_commit import CoreCommitPolicy, VerifiedCommitPolicy
from ..policy import PolicyInputError
from .capacity_source_gate import POLICY_VERSION, CapacityGatedSourceAllocator

# Event-only ablation: "off" keeps detector, recovery and selected-generation validation but
# never applies source-verified argument correction. Every decision's verified_binding and
# commit_transform receipts then carry CoreCommitPolicy.STATUS.
ARGUMENT_CORRECTION_ENV = "C2KV_RACER_ARGUMENT_CORRECTION"


def argument_correction_enabled():
    value = os.environ.get(ARGUMENT_CORRECTION_ENV, "on")
    if value not in {"on", "off"}:
        raise ValueError(f"{ARGUMENT_CORRECTION_ENV} must be 'on' or 'off'")
    return value == "on"


def c1_v2_fields(variant):
    if variant not in C1_V2_VARIANTS:
        raise ValueError("Unknown C1 v2 variant")
    return {
        "initial_view": {"policy": "s0_capacity_fallback", "version": "c2kv-s0-capacity-fallback-v1",
                         "terminal_rescue": "c2kv-terminal-tool-arguments-gist-v1",
                         "source_rescue": POLICY_VERSION,
                         "source_rescue_trigger": "incumbent_c1_capacity_infeasible"},
        "recovery_backbone": "t02_complete_event", "completion_review": False,
        "proof_registry_version": PROOF_REGISTRY_VERSION,
    }


def validate_c1_v2_config(candidate):
    fields = c1_v2_fields(candidate.get("variant"))
    if any(candidate.get(key) != value for key, value in fields.items()):
        raise ValueError("C1 v2 policy contract mismatch")
    if candidate.get("completion_review") is not False:
        raise ValueError("C1 v2 requires completion_review=False")
    threshold = candidate.get("risk_threshold")
    if isinstance(threshold, bool) or threshold != 0.5 or "risk_artifact" not in candidate:
        raise ValueError("C1 v2 requires the frozen T02 artifact and threshold 0.5")
    return fields


def build_c1_v2(tokenizer, *, candidate, packing, policy, model_context, s0_config, benchmark,
                initial_allocator_factory=None):
    from ..initial_factory import instantiate_composed_initial
    validate_c1_v2_config(candidate)
    base = instantiate_composed_initial(CapacityGatedSourceAllocator,
        tokenizer, packing=packing, policy=policy, model_context=model_context,
        s0_config=s0_config, benchmark=benchmark,
        initial_allocator_factory=initial_allocator_factory)
    return C1V2VerifiedController(base, candidate)


class C1V2VerifiedController(CandidateRecoveryController):
    """Explicit composition; budget never switches completion review on."""

    def __init__(self, base, config, *, risk_model=None, binding_policy=None):
        self.c1_variant = config["variant"]
        self.c1_contract = validate_c1_v2_config(config)
        self.commit_policy = (VerifiedCommitPolicy(binding_policy) if argument_correction_enabled()
                              else CoreCommitPolicy())
        super().__init__(base, {**config, "variant": "goal_rescue"},
                         risk_model=risk_model, completion_review_enabled=False)

    def _identity(self, metadata):
        metadata["candidate_algorithm"].update(
            schema=C1_V2_VERSION, variant=self.c1_variant,
            **copy.deepcopy(self.c1_contract), gate_order="t02_event_then_verified_commit")
        metadata["route"]["baseline_identity"] = C1_V2_VERSION + ":" + self.c1_variant

    def prepare(self, payload, *, ratio, max_new_tokens):
        prepared = super().prepare(payload, ratio=ratio, max_new_tokens=max_new_tokens)
        if not hasattr(prepared, "_c1_v2_result"):
            prepared._c1_v2_result = None
            prepared._c1_v2_commit = None
            self._identity(prepared.metadata)
        return prepared

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error=None):
        # Let the original lifecycle validate ownership and held-draft identity.
        original = super().reconsider(prepared, draft_tool_calls,
                                     draft_text=draft_text, parse_error=parse_error)
        if prepared._c1_v2_result is not None:
            return copy.deepcopy(prepared._c1_v2_result)
        budget = min(self.policy_config.history_budget_bytes,
                     self.policy_config.workspace_budget_bytes) // self.kv_bytes_per_token
        context = RepairContext(prepared, tuple(copy.deepcopy(draft_tool_calls)),
                                draft_text, parse_error, lambda rows: self.base._count(rows, ()), budget)
        state = self.commit_policy.inspect(context, original)
        prepared._c1_v2_commit = state
        result = copy.deepcopy(original)
        result["decision"].update(
            version=C1_V2_VERSION, variant=self.c1_variant,
            **copy.deepcopy(self.c1_contract), verified_binding=copy.deepcopy(state.receipt))
        self._identity(result["metadata"])
        result["metadata"]["exact_recovery"] = copy.deepcopy(result["decision"])
        prepared._c1_v2_result = copy.deepcopy(result)
        return result

    def _commit_state(self, prepared):
        key = (prepared._store.session_id, prepared.metadata["decision_key"])
        if self._prepared.get(key) is not prepared or prepared._c1_v2_commit is None:
            raise PolicyInputError("C1 v2 commit requires this controller's reviewed decision")
        return prepared._c1_v2_commit

    def validate_commit(self, prepared, candidate_calls, *, draft_text, parse_error=None):
        verdict = self.commit_policy.validate(self._commit_state(prepared), candidate_calls,
                                              parse_error=parse_error)
        return {"schema": C1_V2_VERSION, "variant": self.c1_variant, **verdict}

    def finalize_commit(self, prepared, candidate_calls):
        output, receipt = self.commit_policy.finalize(self._commit_state(prepared), candidate_calls)
        return output, {"schema": C1_V2_VERSION, "variant": self.c1_variant, **receipt}
