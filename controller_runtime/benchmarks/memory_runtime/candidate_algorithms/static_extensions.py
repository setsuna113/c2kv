"""Independent commit extensions to the frozen Static-T02 recovery lifecycle."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace

from . import STATIC_EXTENSION_VARIANTS, STATIC_EXTENSION_VERSION
from .controller import CandidateRecoveryController, memory_signature
from .initial_view import STATIC_INITIAL_VIEW, build_initial_view_allocator
from .repair_protocol import RepairContext
from .verified_binding import PROOF_REGISTRY_VERSION
from ..policy import PolicyInputError
from ..recovery.admission import metadata_after_admission
from ..recovery.gate import canonical


def extension_fields(variant):
    if variant not in STATIC_EXTENSION_VARIANTS:
        raise ValueError("Unknown Static extension")
    if variant == "static_verified_v2":
        from .relational_binding import PROOF_REGISTRY_VERSION as RELATIONAL_VERSION
        return {
            "recovery_backbone": "static_t02",
            "initial_view": copy.deepcopy(STATIC_INITIAL_VIEW),
            "commit_policy": "verified_binding_v2",
            "proof_registry_version": RELATIONAL_VERSION,
            "base_proof_registry_version": PROOF_REGISTRY_VERSION,
        }
    return {
        "recovery_backbone": "static_t02",
        "initial_view": copy.deepcopy(STATIC_INITIAL_VIEW),
        "commit_policy": "verified_binding" if variant == "static_verified" else "action_ledger",
        **({"proof_registry_version": PROOF_REGISTRY_VERSION}
           if variant == "static_verified" else {
               "action_ledger_version": "static-action-ledger-v1",
               "action_rules_version": "action-ledger-rules-v1"}),
    }


def validate_extension_config(candidate):
    fields = extension_fields(candidate.get("variant"))
    if any(candidate.get(key) != value for key, value in fields.items()):
        raise ValueError("Static extension policy contract mismatch")
    if "risk_artifact" not in candidate or candidate.get("risk_threshold") != 0.5:
        raise ValueError("Static extensions require the frozen T02 artifact and threshold 0.5")
    return fields


def build_static_extension(tokenizer, *, candidate, packing, policy,
                           model_context, s0_config, benchmark):
    validate_extension_config(candidate)
    base = build_initial_view_allocator(
        tokenizer, initial_view=STATIC_INITIAL_VIEW, packing=packing, policy=policy,
        model_context=model_context, s0_config=s0_config, benchmark=benchmark)
    cls = {
        "static_verified": StaticVerifiedController,
        "static_action_ledger": StaticActionLedgerController,
        "static_verified_v2": StaticVerifiedV2Controller,
    }[candidate["variant"]]
    return cls(base, candidate)


class StaticExtensionController(CandidateRecoveryController):
    """Own extension identity and commit state while preserving Static's cache."""

    def __init__(self, base, config, *, risk_model=None):
        self.extension_variant = config["variant"]
        self.extension_contract = validate_extension_config(config)
        super().__init__(base, {**config, "variant": "static_t02"}, risk_model=risk_model)

    def _identity(self, metadata):
        metadata["candidate_algorithm"].update(
            schema=STATIC_EXTENSION_VERSION, variant=self.extension_variant,
            **copy.deepcopy(self.extension_contract),
            gate_order="static_event_recovery_then_commit_extension",
            original_static_initial_view_preserved=True)
        metadata["route"]["baseline_identity"] = STATIC_EXTENSION_VERSION + ":" + self.extension_variant

    def prepare(self, payload, *, ratio, max_new_tokens):
        prepared = super().prepare(payload, ratio=ratio, max_new_tokens=max_new_tokens)
        if not hasattr(prepared, "_static_extension_result"):
            prepared._static_extension_result = None
            prepared._static_extension_context = None
            prepared._static_extension_proposal = None
            prepared._static_commit_accepted = False
            prepared._static_commit_signature = None
            prepared._static_finalized = None
            self._identity(prepared.metadata)
        return prepared

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error=None):
        original = super().reconsider(prepared, draft_tool_calls,
                                     draft_text=draft_text, parse_error=parse_error)
        if prepared._static_extension_result is not None:
            return copy.deepcopy(prepared._static_extension_result)
        budget = min(self.policy_config.history_budget_bytes,
                     self.policy_config.workspace_budget_bytes) // self.kv_bytes_per_token
        prepared._static_extension_context = RepairContext(
            prepared, tuple(copy.deepcopy(draft_tool_calls)), draft_text, parse_error,
            lambda rows: self.base._count(rows, ()), budget)
        result = copy.deepcopy(original)
        result["decision"]["backbone"] = {
            "variant": "static_t02", "status": original["decision"]["status"],
            "reason": original["decision"]["reason"]}
        self._extend(prepared, result)
        result["decision"].update(version=STATIC_EXTENSION_VERSION,
                                  variant=self.extension_variant,
                                  **copy.deepcopy(self.extension_contract))
        self._identity(result["metadata"])
        result["metadata"]["exact_recovery"] = copy.deepcopy(result["decision"])
        prepared._static_extension_result = copy.deepcopy(result)
        return result

    def _extend(self, prepared, result):
        raise NotImplementedError

    def _check_prepared(self, prepared):
        key = (prepared._store.session_id, prepared.metadata["decision_key"])
        if self._prepared.get(key) is not prepared or prepared._static_extension_result is None:
            raise PolicyInputError("Static commit requires this controller's reviewed decision")

    def validate_commit(self, prepared, candidate_calls, *, draft_text, parse_error=None):
        self._check_prepared(prepared)
        verdict = self._validate_selected(prepared, candidate_calls, draft_text, parse_error)
        prepared._static_commit_accepted = verdict["accepted"] and parse_error is None
        prepared._static_commit_signature = canonical(candidate_calls)
        return {"schema": STATIC_EXTENSION_VERSION, "variant": self.extension_variant,
                **verdict}

    def _validate_selected(self, prepared, calls, draft_text, parse_error):
        return {"accepted": parse_error is None, "fallback": "original",
                "reason": "malformed_selected_draft" if parse_error else "original_static_commit_preserved"}

    def finalize_commit(self, prepared, candidate_calls):
        self._check_prepared(prepared)
        signature = canonical(candidate_calls)
        if prepared._static_finalized is not None:
            prior, finalized = prepared._static_finalized
            if prior != signature:
                raise PolicyInputError("Static extension cannot finalize another selected draft")
            return copy.deepcopy(finalized)
        calls = tuple(copy.deepcopy(candidate_calls))
        receipt = {"schema": STATIC_EXTENSION_VERSION, "variant": self.extension_variant,
                   **copy.deepcopy(self.extension_contract), "changed": False,
                   "status": "selected_commit_not_accepted", "additional_generations": 0,
                   "additional_model_workspace_tokens": 0}
        if (prepared._static_commit_accepted
                and prepared._static_commit_signature == signature):
            calls, details = self._finalize_selected(prepared, calls)
            receipt.update(details, changed=canonical(calls) != signature)
        finalized = (tuple(calls), receipt)
        prepared._static_finalized = (signature, copy.deepcopy(finalized))
        return finalized


class StaticVerifiedController(StaticExtensionController):
    def __init__(self, base, config, *, risk_model=None, binding_policy=None):
        from .verified_binding import Policy
        self.binding_policy = Policy() if binding_policy is None else binding_policy
        super().__init__(base, config, risk_model=risk_model)

    def _extend(self, prepared, result):
        context = prepared._static_extension_context
        receipt = {"version": PROOF_REGISTRY_VERSION, "additional_generations": 0,
                   "additional_model_workspace_tokens": 0}
        if result["regenerate"]:
            receipt["status"] = "original_static_recovery_preserved"
        elif result["decision"]["reason"] == "shared_task_generation_limit":
            receipt["status"] = "task_generation_limit_preserved"
        elif context.parse_error is not None or not context.draft_tool_calls:
            receipt["status"] = "no_eligible_tool_draft"
        else:
            proposal = self.binding_policy.propose(context)
            prepared._static_extension_proposal = proposal
            receipt["status"] = "proposed" if proposal is not None else "no_verified_binding"
            if proposal is not None:
                receipt["proposal"] = proposal.to_receipt()
        result["decision"]["verified_binding"] = receipt

    def _finalize_selected(self, prepared, calls):
        proposal = prepared._static_extension_proposal
        context = prepared._static_extension_context
        if proposal is None or canonical(calls) != canonical(context.draft_tool_calls):
            return calls, {"status": "original_static_commit_preserved"}
        corrected, receipt = self.binding_policy.apply(context, proposal, calls)
        verdict = self.binding_policy.validate(context, proposal, corrected)
        return (tuple(corrected) if verdict.accepted else calls), {
            "status": "verified_binding_committed" if verdict.accepted else "binding_verification_rejected",
            "proposal": proposal.to_receipt(), "verification": receipt,
            "accepted": verdict.accepted, "reason": verdict.reason}


class StaticVerifiedV2Controller(StaticVerifiedController):
    """Keep legacy eligibility; prove new relations against the final selected calls."""

    def __init__(self, base, config, *, risk_model=None, binding_policy=None,
                 relation_policy=None):
        from .relational_binding import Policy
        self.relation_policy = Policy() if relation_policy is None else relation_policy
        super().__init__(base, config, risk_model=risk_model, binding_policy=binding_policy)

    def _finalize_selected(self, prepared, calls):
        base_calls, base_receipt = super()._finalize_selected(prepared, calls)
        # A recovered draft can differ from the held draft used by legacy rules.
        # Rebuild the proof context from the actual selected, legacy-checked calls.
        context = replace(prepared._static_extension_context,
                          draft_tool_calls=tuple(copy.deepcopy(base_calls)),
                          draft_text="", parse_error=None)
        proposal = self.relation_policy.propose(context)
        details = {"status": base_receipt["status"], "base_verification": base_receipt,
                   "relational_verification": {"status": "no_verified_relation"}}
        if proposal is None:
            return base_calls, details
        corrected, receipt = self.relation_policy.apply(context, proposal, base_calls)
        verdict = self.relation_policy.validate(context, proposal, corrected)
        details.update(
            status="relational_binding_committed" if verdict.accepted else "relational_binding_rejected",
            relational_verification={"proposal": proposal.to_receipt(), "verification": receipt,
                                     "accepted": verdict.accepted, "reason": verdict.reason})
        return (tuple(corrected) if verdict.accepted else base_calls), details


class StaticActionLedgerController(StaticExtensionController):
    def __init__(self, base, config, *, risk_model=None, ledger_policy=None):
        from .action_ledger import Policy
        self.ledger_policy = Policy() if ledger_policy is None else ledger_policy
        self._ledger_reviewed = set()
        super().__init__(base, config, risk_model=risk_model)

    def _extend(self, prepared, result):
        context = prepared._static_extension_context
        receipt = {"status": "original_static_recovery_preserved" if result["regenerate"] else "no_ready_witness"}
        result["decision"]["action_ledger"] = receipt
        if (result["regenerate"] or context.parse_error is not None or context.draft_tool_calls
                or result["decision"]["reason"] == "shared_task_generation_limit"):
            return
        receipt["assessment"] = self.ledger_policy.inspect(context).receipt
        proposal = self.ledger_policy.propose(context)
        if proposal is None:
            return
        state_key = (prepared._store.session_id, proposal.receipt["obligation_id"])
        if state_key in self._ledger_reviewed:
            receipt["status"] = "witness_already_reviewed"
            return
        measure, metadata = self._admit_packet(prepared, proposal.messages)
        if measure is None:
            receipt.update(status="packet_not_admitted_without_eviction", proposal=proposal.receipt)
            return
        prepared._static_extension_proposal = proposal
        self._ledger_reviewed.add(state_key)
        session = prepared._store.session_id
        self._recovery_counts[session] = self._recovery_counts.get(session, 0) + 1
        receipt.update(status="ready_witness_review", proposal=proposal.receipt,
                       original_raw_gist_partition_preserved=True)
        result.update(regenerate=True, memory=measure.memory, metadata=metadata)
        result["decision"].update(status="recover", reason="static_ready_action_review",
                                  regeneration_allowed=True, upgrade_count=1)

    def _admit_packet(self, prepared, messages):
        view = prepared.memory.view
        derived = list(prepared.metadata.get("derived_workspace_prefix_messages") or ())
        for message in messages:
            if message not in derived:
                derived.append(message)
        derived = tuple(self.base._merge_protected_derived(tuple(derived)))
        eligible = prepared.metadata["eligible_extraction"]["eligible_event_ids"]
        measure = self.base._try_measure(
            prepared._store, prepared._tools, view.raw_event_ids,
            view.mandatory_raw_event_ids, view.gist_event_ids, eligible,
            prepared.metadata["common_raw_prompt_tokens"],
            prepared.metadata["max_new_tokens"], derived_messages=derived)
        if measure is None or measure.reasons or memory_signature(measure.memory) == memory_signature(prepared.memory):
            return None, None
        if (measure.memory.view.raw_event_ids != view.raw_event_ids
                or measure.memory.view.gist_event_ids != view.gist_event_ids):
            raise PolicyInputError("Ledger review must preserve the Static raw/gist partition")
        receipt = {"policy": "static-ledger-no-eviction-v1", "status": "admitted",
                   "demoted_raw_event_ids": [], "released_gist_event_ids": [],
                   "b0_rechecked_after_all_changes": True}
        metadata = metadata_after_admission(self.base, prepared, measure, None, receipt)
        metadata["derived_workspace_prefix_messages"] = list(derived)
        metadata["eligible_extraction"]["retained_encoder_unit_ids"] = list(
            dict.fromkeys(chunk.event_id for chunk in measure.memory.chunks))
        return measure, metadata

    def _validate_selected(self, prepared, calls, draft_text, parse_error):
        proposal = prepared._static_extension_proposal
        if proposal is None:
            return super()._validate_selected(prepared, calls, draft_text, parse_error)
        verdict = self.ledger_policy.validate(
            prepared._static_extension_context, proposal, calls,
            draft_text=draft_text, parse_error=parse_error)
        return asdict(verdict)

    def _finalize_selected(self, prepared, calls):
        return self.ledger_policy.filter_completed(prepared._static_extension_context, calls)

    def render_commit(self, draft, calls, *, receipt):
        from .static_commit import render_filtered_commit
        return render_filtered_commit(draft, calls, benchmark=self.benchmark)
