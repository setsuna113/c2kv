"""Shared incumbent-first protection, capacity rescue, T02 and Verified."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json

from . import C1_V2_VARIANTS, C1_V2_VERSION
from .controller import CandidateRecoveryController
from .repair_protocol import RepairContext
from .verified_binding import PROOF_REGISTRY_VERSION
from .verified_commit import CoreCommitPolicy, VerifiedCommitPolicy
from ..policy import PolicyInputError
from ..recovery.source import select_source_event
from .capacity_source_gate import POLICY_VERSION, CapacityGatedSourceAllocator

SELF_REVISION_PROMPT = (
    "Review the draft below using only the context already provided. Revise it if needed, "
    "then return the final response or tool calls in the required format.")
_ABLATION_FIELDS = {
    "c1_v2_core": {"argument_correction": False},
    "c1_v2_selfrev": {"argument_correction": False,
                      "post_draft_action": "evidence_free_self_revision",
                      "self_revision_prompt": SELF_REVISION_PROMPT},
    "c1_v2_nodraftq": {"argument_correction": False,
                       "retrieval_query": "current_request_plus_latest_complete_observation"},
    "c1_v2_probe": {"argument_correction": False,
                    "probe_gate": "forced_at_target_closed_elsewhere"},
}


def c1_v2_fields(variant):
    if variant not in C1_V2_VARIANTS:
        raise ValueError("Unknown C1 v2 variant")
    fields = {
        "initial_view": {"policy": "s0_capacity_fallback", "version": "c2kv-s0-capacity-fallback-v1",
                         "terminal_rescue": "c2kv-terminal-tool-arguments-gist-v1",
                         "source_rescue": POLICY_VERSION,
                         "source_rescue_trigger": "incumbent_c1_capacity_infeasible"},
        "recovery_backbone": "t02_complete_event", "completion_review": False,
        "proof_registry_version": PROOF_REGISTRY_VERSION,
    }
    if variant in _ABLATION_FIELDS:
        fields["ablation"] = {"variant": variant, **_ABLATION_FIELDS[variant]}
    return fields


def probe_targets(candidate):
    """Frozen task -> decision_key map of the frozen-state probe."""
    targets = candidate.get("probe_targets")
    if (not isinstance(targets, dict) or not targets or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in targets.items())):
        raise ValueError("C1 v2 probe requires a task -> decision_key target map")
    digest = hashlib.sha256(json.dumps(
        dict(sorted(targets.items())), sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if candidate.get("probe_targets_sha256") != digest:
        raise ValueError("C1 v2 probe targets differ from their recorded digest")
    return dict(targets)


def validate_c1_v2_config(candidate):
    fields = c1_v2_fields(candidate.get("variant"))
    if any(candidate.get(key) != value for key, value in fields.items()):
        raise ValueError("C1 v2 policy contract mismatch")
    if candidate.get("completion_review") is not False:
        raise ValueError("C1 v2 requires completion_review=False")
    threshold = candidate.get("risk_threshold")
    if isinstance(threshold, bool) or threshold != 0.5 or "risk_artifact" not in candidate:
        raise ValueError("C1 v2 requires the frozen T02 artifact and threshold 0.5")
    if candidate.get("variant") == "c1_v2_probe":
        probe_targets(candidate)
    elif "probe_targets" in candidate:
        raise ValueError("Only the C1 v2 probe accepts probe targets")
    return fields


def build_c1_v2(tokenizer, *, candidate, packing, policy, model_context, s0_config, benchmark,
                initial_allocator_factory=None):
    from ..initial_factory import instantiate_composed_initial
    validate_c1_v2_config(candidate)
    base = instantiate_composed_initial(CapacityGatedSourceAllocator,
        tokenizer, packing=packing, policy=policy, model_context=model_context,
        s0_config=s0_config, benchmark=benchmark,
        initial_allocator_factory=initial_allocator_factory)
    return c1_v2_controller_class(candidate["variant"])(base, candidate)


def c1_v2_controller_class(variant):
    if variant not in C1_V2_VARIANTS:
        raise ValueError("Unknown C1 v2 variant")
    return {"c1_v2_verified": C1V2VerifiedController, "c1_v2_core": C1V2CoreController,
            "c1_v2_selfrev": C1V2SelfRevisionController,
            "c1_v2_nodraftq": C1V2NoDraftQueryController,
            "c1_v2_probe": C1V2ProbeController}[variant]


class C1V2VerifiedController(CandidateRecoveryController):
    """Explicit composition; budget never switches completion review on."""

    def __init__(self, base, config, *, risk_model=None, binding_policy=None):
        self.c1_variant = config["variant"]
        self.c1_contract = validate_c1_v2_config(config)
        self.commit_policy = VerifiedCommitPolicy(binding_policy)
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
        original = self._post_draft_action(prepared, original, draft_text=draft_text)
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

    def _post_draft_action(self, prepared, original, *, draft_text):
        """The full system regenerates from the recovered evidence view unchanged."""
        return original

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


class C1V2CoreController(C1V2VerifiedController):
    """RACER-core: identical recovery, no source-verified argument correction.

    The commit policy still validates the selected generation, so a malformed
    regeneration falls back to the held draft exactly as in the full system.
    """

    def __init__(self, base, config, *, risk_model=None, binding_policy=None):
        super().__init__(base, config, risk_model=risk_model, binding_policy=binding_policy)
        self.commit_policy = CoreCommitPolicy()


class C1V2NoDraftQueryController(C1V2CoreController):
    """RACER-core whose retrieval query omits every held-draft signal.

    The detector input, timing, candidate universe and admission are unchanged;
    relevance filtering and ranking are recomputed from the current request and
    the latest complete observation only.
    """

    def _select_source(self, prepared, draft_tool_calls, draft_text):
        selected, source = select_source_event(
            prepared, [], draft_text="", include_latest_complete_observation=True,
            explicit_revision_abstain=False, allow_empty_draft_query=True)
        source["draft_in_retrieval_query"] = False
        return selected, source


class C1V2SelfRevisionController(C1V2CoreController):
    """Same trigger and eligibility as RACER-core; regenerate without new evidence.

    When RACER-core would regenerate from a recovered view, the actor instead
    keeps the original initial context and receives one appended user message:
    the frozen review prompt followed by the held draft text.
    """

    def _review_ids(self, draft_text):
        from history_memory.packing import native_ids
        dummy = {"role": "user", "content": ""}
        plain = native_ids(self.tokenizer, [dummy])
        prompt = native_ids(self.tokenizer, [dummy], generation=True)
        review = native_ids(self.tokenizer, [dummy, {
            "role": "user", "content": SELF_REVISION_PROMPT + "\n\n" + draft_text}])
        if prompt[:len(plain)] != plain or review[:len(plain)] != plain:
            raise PolicyInputError("Native template does not render separable user turns")
        return review[len(plain):], prompt[len(plain):]

    def _post_draft_action(self, prepared, original, *, draft_text):
        if not original["regenerate"]:
            return original
        session_id = prepared._store.session_id
        decision = copy.deepcopy(original["decision"])
        eligibility = {key: decision.pop(key) for key in (
            "candidate_event_id", "upgraded_event_id", "restored_event", "allocation",
            "candidate_trials") if key in decision}
        eligibility["selected_ids"] = decision["selection"].get("selected_ids", [])
        decision["selection"]["selected_ids"] = []
        review_ids, generation_ids = self._review_ids(draft_text)
        memory = prepared.memory
        workspace = tuple(memory.workspace_input_ids)
        if not generation_ids or workspace[-len(generation_ids):] != tuple(generation_ids):
            raise PolicyInputError("Self-revision requires a workspace ending in the generation prompt")
        revised = dataclasses.replace(memory, workspace_input_ids=(
            workspace[:-len(generation_ids)] + tuple(review_ids) + tuple(generation_ids)))
        added = len(revised.workspace_input_ids) - len(workspace)
        max_new_tokens = prepared.metadata["max_new_tokens"]
        resident = revised.costs(8)["resident_kv_tokens"]
        reasons = []
        if len(revised.workspace_input_ids) > self.packing.max_workspace_tokens:
            reasons.append("workspace_token_budget")
        if self.model_context is not None and (
                revised.workspace_position_start + len(revised.workspace_input_ids)
                + max_new_tokens > self.model_context):
            reasons.append("model_logical_context")
        if resident + max_new_tokens > self.packing.max_sequence_tokens:
            reasons.append("physical_sequence_budget")
        if self.model_context is not None and resident + max_new_tokens > self.model_context:
            reasons.append("model_physical_context")
        receipt = {"schema": "c2kv-evidence-free-self-revision-v1",
                   "prompt_sha256": hashlib.sha256(SELF_REVISION_PROMPT.encode()).hexdigest(),
                   "draft_text_sha256": hashlib.sha256(draft_text.encode()).hexdigest(),
                   "added_live_tokens": added, "eligibility": eligibility,
                   "evidence_provided_to_actor": False,
                   "context_check": {"status": "rejected" if reasons else "passed",
                                     "reasons": reasons}}
        decision["self_revision"] = receipt
        if reasons:
            # Keep the held draft; the admitted recovery was never generated.
            self._recovery_counts[session_id] -= 1
            decision.update(status="abstain", reason="self_revision_context_limit",
                            regeneration_allowed=False, upgrade_count=0)
            result = {"regenerate": False, "memory": prepared.memory,
                      "metadata": copy.deepcopy(prepared.metadata), "decision": decision}
        else:
            decision["reason"] = "risk_triggered_evidence_free_self_revision"
            metadata = copy.deepcopy(prepared.metadata)
            metadata["common_raw_prompt_tokens"] += added
            if "history_only_resident_kv_tokens" in metadata:
                metadata["history_only_resident_kv_tokens"] += added
            metadata["self_revision"] = copy.deepcopy(receipt)
            result = {"regenerate": True, "memory": revised, "metadata": metadata,
                      "decision": decision}
        result["metadata"]["exact_recovery"] = copy.deepcopy(decision)
        prepared._checked_result = copy.deepcopy(result)
        return result


class C1V2ProbeController(C1V2CoreController):
    """Frozen-state branch: RACER-core forced once at a frozen target decision.

    Every other decision skips the detector and commits the held draft, which
    matches the recovery-off policy; the target records its real risk score
    and then bypasses only the threshold.
    """

    def __init__(self, base, config, *, risk_model=None, binding_policy=None):
        super().__init__(base, config, risk_model=risk_model, binding_policy=binding_policy)
        self.probe_targets = probe_targets(config)
        self.probe_targets_sha256 = config["probe_targets_sha256"]

    def _probe_target(self, prepared):
        session_id = prepared._store.session_id
        parts = session_id.split("/")
        task_id = parts[1] if len(parts) == 3 else session_id
        return self.probe_targets.get(task_id) == prepared.metadata["decision_key"]

    def _risk_gate(self, prepared, draft_tool_calls, draft_text, parse_error, decision):
        if not self._probe_target(prepared):
            decision["selection"] = {"selector": "probe_closed", "available": False,
                                     "score": None, "reason": "probe_gate_closed",
                                     "selected_ids": []}
            decision["gate"] = {"type": "probe", "score": None, "threshold": self.threshold,
                                "triggered": False, "reason": "probe_gate_closed"}
            decision["probe"] = {"target": False, "targets_sha256": self.probe_targets_sha256}
            return decision["gate"]
        gate = super()._risk_gate(prepared, draft_tool_calls, draft_text, parse_error, decision)
        decision["probe"] = {"target": True, "risk_triggered": gate["triggered"],
                             "targets_sha256": self.probe_targets_sha256}
        decision["gate"] = {**gate, "triggered": True, "risk_triggered": gate["triggered"],
                            "reason": "probe_forced_at_target"}
        return decision["gate"]
