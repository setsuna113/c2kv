"""Compose bounded policies around the original Goal recovery lifecycle."""
from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict

from . import GOAL_VERSION
from .controller import CandidateRecoveryController, memory_signature
from .goal_protocol import modules_for
from .observations import current_request, operation_records, parse_json_or_text
from .progress import review_messages
from .repair_protocol import RepairContext
from .repacking import repack
from ..policy import PolicyInputError
from ..recovery.gate import canonical


def call_value(call):
    function = call.get("function", {})
    return function.get("name"), parse_json_or_text(function.get("arguments", {}))


def same_call(left, right):
    return canonical(call_value(left)) == canonical(call_value(right))


def matches_operation(row, call, unbound_paths=()):
    name, arguments = call_value(call)
    ignored = {tuple(path) for path in unbound_paths}
    if row.tool != name or not isinstance(arguments, dict) or not isinstance(row.arguments, dict):
        return False
    def compatible(expected, actual, path=()):
        if path in ignored:
            return True
        if isinstance(expected, dict):
            return isinstance(actual, dict) and all(
                key in actual and compatible(value, actual[key], (*path, key))
                for key, value in expected.items())
        return expected == actual
    return compatible(arguments, row.arguments)


class GoalCompositionController(CandidateRecoveryController):
    """Keep Goal first; additional policies compete for its unused single slot."""

    def __init__(self, base, config, *, risk_model=None, policies=None):
        self.goal_variant = config["variant"]
        self.policies = modules_for(self.goal_variant) if policies is None else tuple(policies)
        self.pending_enabled = self.goal_variant in {"goal_pending", "goal_joint"}
        self._additional_reviewed = set()
        self._deferred = {}
        super().__init__(base, {**config, "variant": "goal_rescue"}, risk_model=risk_model)

    def prepare(self, payload, *, ratio, max_new_tokens):
        prepared = super().prepare(payload, ratio=ratio, max_new_tokens=max_new_tokens)
        if not hasattr(prepared, "_goal_composition_result"):
            if self.benchmark == "acebench":
                from ..acebench_source import build_ace_event_store
                prepared._store = build_ace_event_store(
                    payload["session_id"], payload["messages"], payload["c2kv_ace_source"])
            prepared._goal_composition_result = None
            prepared._goal_context = None
            prepared._goal_proposal = None
            prepared._goal_policy = None
            prepared.metadata["candidate_algorithm"].update(
                schema=GOAL_VERSION, variant=self.goal_variant,
                backbone="goal_rescue", gate_order="goal_then_source_then_progress",
                commit_validation=True, current_user_obligations=self.pending_enabled)
            prepared.metadata["route"]["baseline_identity"] = GOAL_VERSION + ":" + self.goal_variant
        return prepared

    def _goal_review_request(self, prepared, draft_tool_calls, *, parse_error):
        if self.benchmark != "acebench":
            return super()._goal_review_request(prepared, draft_tool_calls, parse_error=parse_error)
        goal, _ = current_request(prepared._store)
        rows = [asdict(row) for row in operation_records(prepared._store, current_request_only=True)]
        signatures = [canonical([row["tool"], row["arguments"], row["observed_result"]]) for row in rows]
        repeated = len(rows) >= 2 and signatures[-1] == signatures[-2] and rows[-1]["failure_reported"]
        reason = ("stop_completion_review" if parse_error is None and not draft_tool_calls
                  else "repeated_observed_failure" if repeated else None)
        state = hashlib.sha256(canonical({"goal": goal.event_id if goal else None,
                                          "observations": signatures}).encode()).hexdigest()
        return reason if goal and rows else None, state, goal, rows

    def _active_deferred(self, prepared):
        session = prepared._store.session_id
        goal, _ = current_request(prepared._store)
        request_id = goal.event_id if goal else None
        records = operation_records(prepared._store, current_request_only=True)
        pending, active = [], []
        for entry in self._deferred.get(session, ()):
            if entry.get("request_event_id") != request_id:
                continue
            later = [row for row in records if row.result_source_index > entry["after_source_index"]]
            consumer = entry.get("call", {})
            # An observed attempt discharges the deferred draft, not the user goal.
            # Its success or failure remains in the ordinary Goal execution record.
            if any(matches_operation(row, consumer, entry.get("unbound_paths", ())) for row in later):
                continue
            producers = entry.get("producer_calls", ())
            paths = entry.get("producer_unbound_paths", [[] for _ in producers])
            observed = [asdict(row) for row in later if any(
                matches_operation(row, call, ignored) for call, ignored in zip(producers, paths))]
            pending.append(entry)
            if observed:
                active.append({**entry, "producer_observations": observed, "producer": observed[-1],
                               "consumer_status": "not_yet_observed"})
        self._deferred[session] = pending
        return tuple(active)

    def _goal_review_messages(self, prepared, goal, records, budget):
        original, receipt = review_messages(prepared._store, goal, records,
            token_counter=lambda rows: self.base._count(rows, ()), token_budget=budget)
        deferred = self._active_deferred(prepared)
        if not original or (not self.pending_enabled and not deferred):
            return original, receipt
        from .goal_pending import review_messages as augment
        proposal = augment(prepared._goal_context, original, deferred_consumers=deferred)
        if proposal is None:
            return original, {**receipt, "composition": "original_packet_retained"}
        # Probe the complete packet before substituting it for original Goal.
        # Failure falls back before generation, preserving the existing review.
        measure, _, allocation = repack(self.base, prepared,
                                       derived_messages=proposal.messages, goal_view=True)
        if measure is None:
            return original, {**receipt, "composition": "extension_not_admitted",
                              "extension_allocation": allocation}
        return proposal.messages, {**receipt, "composition": "pending_receipts",
                                   "extension": proposal.receipt}

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error=None):
        signature = canonical([draft_tool_calls, draft_text, parse_error])
        if prepared._goal_composition_result is not None:
            if signature != prepared._candidate_signature:
                raise PolicyInputError("Goal decision cannot inspect another held draft")
            return copy.deepcopy(prepared._goal_composition_result)
        budget = min(self.policy_config.history_budget_bytes,
                     self.policy_config.workspace_budget_bytes) // self.kv_bytes_per_token
        context = RepairContext(prepared, tuple(copy.deepcopy(draft_tool_calls)), draft_text,
                                parse_error, lambda rows: self.base._count(rows, ()), budget)
        prepared._goal_context = context
        result = super().reconsider(prepared, draft_tool_calls, draft_text=draft_text, parse_error=parse_error)
        original = copy.deepcopy(result["decision"])
        decision = copy.deepcopy(original)
        decision.update(version=GOAL_VERSION, variant=self.goal_variant,
                        backbone={"variant": "goal_rescue", "status": original["status"],
                                  "reason": original["reason"]}, additional_policy=None)
        key = prepared._store.session_id
        self._active_deferred(prepared)
        if not result["regenerate"] and original["reason"] != "shared_task_generation_limit":
            attempts = []
            for name, policy in self.policies:
                proposal = policy.propose(context)
                if proposal is None:
                    continue
                state = proposal.receipt.get("state_key") or hashlib.sha256(canonical([
                    proposal.messages, [call.get("function") for call in draft_tool_calls]]).encode()).hexdigest()
                review_key = (key, name, state)
                attempt = {"policy": name, "proposal": copy.deepcopy(proposal.receipt)}
                attempts.append(attempt)
                if review_key in self._additional_reviewed:
                    attempt["status"] = "observation_state_already_reviewed"
                    continue
                if not proposal.messages or context.token_counter(proposal.messages) > budget:
                    attempt["status"] = "packet_exceeds_budget"
                    continue
                measure, metadata, allocation = repack(self.base, prepared, derived_messages=proposal.messages)
                attempt["allocation"] = allocation
                if measure is None or memory_signature(measure.memory) == memory_signature(prepared.memory):
                    attempt["status"] = "no_feasible_new_repair_view"
                    continue
                prepared._goal_proposal = proposal
                prepared._goal_policy = policy
                self._additional_reviewed.add(review_key)
                self._recovery_counts[key] = self._recovery_counts.get(key, 0) + 1
                attempt["status"] = "admitted"
                decision.update(status="recover", reason=proposal.reason, regeneration_allowed=True,
                                upgrade_count=1, additional_policy=name,
                                proposal=copy.deepcopy(proposal.receipt), allocation=allocation)
                result = {"regenerate": True, "memory": measure.memory, "metadata": metadata}
                break
            decision["additional_attempts"] = attempts
        result["decision"] = decision
        result["metadata"]["exact_recovery"] = copy.deepcopy(decision)
        result["metadata"]["candidate_algorithm"] = copy.deepcopy(prepared.metadata["candidate_algorithm"])
        result["metadata"]["route"] = copy.deepcopy(prepared.metadata["route"])
        prepared._goal_composition_result = copy.deepcopy(result)
        prepared._checked_result = copy.deepcopy(result)
        return result

    def validate_commit(self, prepared, candidate_calls, *, draft_text, parse_error=None):
        verdict = {"schema": GOAL_VERSION, "variant": self.goal_variant, "accepted": True,
                   "reason": "original_goal_commit", "fallback": "original"}
        if prepared._goal_proposal is not None:
            checked = prepared._goal_policy.validate(prepared._goal_context, prepared._goal_proposal,
                candidate_calls, draft_text=draft_text, parse_error=parse_error)
            verdict.update(accepted=checked.accepted, reason=checked.reason)
        return verdict

    def finalize_commit(self, prepared, candidate_calls):
        calls = tuple(copy.deepcopy(candidate_calls))
        receipt = {"schema": GOAL_VERSION, "variant": self.goal_variant, "changed": False}
        for name, policy in self.policies:
            if name == "source":
                calls, correction = policy.patch_calls(prepared._goal_context, calls)
                receipt["source_patch"] = correction
                receipt["changed"] = canonical(calls) != canonical(candidate_calls)
        proposal = prepared._goal_proposal
        deferred = [] if proposal is None else proposal.guard.get("deferred_consumers", [])
        registered = []
        for entry in deferred:
            producers = entry.get("producer_calls", ())
            providers = [next((row for row in registered if same_call(row["call"], producer)), None)
                         for producer in producers]
            if not producers or not all(provider is not None or any(
                    same_call(committed, producer) for committed in calls)
                    for producer, provider in zip(producers, providers)):
                continue
            if any(same_call(committed, entry.get("call", {})) for committed in calls):
                continue
            saved = {**copy.deepcopy(entry), "after_source_index": len(prepared._store.messages) - 1,
                     "producer_unbound_paths": [row.get("unbound_paths", []) if row else []
                                                for row in providers]}
            existing = self._deferred.setdefault(prepared._store.session_id, [])
            if saved not in existing:
                existing.append(saved)
                registered.append(saved)
        receipt["deferred_consumers_registered"] = registered
        return calls, receipt
