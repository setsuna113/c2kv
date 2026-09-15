"""Finite development controls for trigger timing and exact-source retention.

This optional legacy adapter extension is separate from the frozen event-native
baseline. It consumes only the current observable prefix and native drafts.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace

from .adapter import PreparedExact, RuntimeAdapter
from .exact_gap import _json, _string_leaves, detect_exact_source_gap, message_has_literal
from .policy import BudgetExceeded, PolicyInputError, _checked_cost, _ordered_ids


VERSION = "a-phase4-dev-policy-v1"


def validate_phase4_policy(value):
    fields = {"schema", "trigger", "retention", "random_probability", "random_seed"}
    if not isinstance(value, dict) or set(value) != fields or value["schema"] != VERSION:
        raise ValueError("Invalid Phase4 development policy")
    if value["trigger"] not in {"off", "conservative", "random"}:
        raise ValueError("Unknown Phase4 trigger")
    if value["retention"] not in {"finite", "final_reference"}:
        raise ValueError("Unknown Phase4 retention")
    if value["retention"] == "final_reference" and value["trigger"] != "conservative":
        raise ValueError("Final-reference retention requires conservative acquisition")
    rate = value["random_probability"]
    if (not isinstance(rate, dict) or set(rate) != {"numerator", "denominator"}
            or any(type(rate[key]) is not int for key in rate)
            or not 0 <= rate["numerator"] <= rate["denominator"]
            or rate["denominator"] <= 0 or type(value["random_seed"]) is not int):
        raise ValueError("Random probability must be an exact rational in [0, 1]")
    return json.loads(json.dumps(value))


def _final_strings(calls):
    if calls is None or calls == []:
        return set(), "no_native_tool_calls"
    if not isinstance(calls, list):
        return set(), "malformed_arguments"
    values = set()
    try:
        for call in calls:
            arguments = call["function"]["arguments"]
            parsed = _json(arguments) if isinstance(arguments, str) else arguments
            if not isinstance(parsed, dict):
                raise ValueError("Native arguments must be an object")
            values.update(value for _, value in _string_leaves(parsed))
    except (KeyError, TypeError, ValueError):
        return set(), "malformed_arguments"
    return values, "parsed"


class Phase4RuntimeAdapter(RuntimeAdapter):
    """Keep the original budget and one-upgrade controller for all controls."""

    def __init__(self, config, token_counter):
        self.phase4_policy = validate_phase4_policy(config.get("phase4_policy"))
        if config.get("mode") != "capacity_exact_persistent":
            raise ValueError("Phase4 development requires capacity_exact_persistent")
        super().__init__(config, token_counter)
        if self.config.max_retrieved_events != 1 or self.config.lease_decisions <= 0:
            raise ValueError("Phase4 requires one retrieved event and a finite positive lease")
        self._witnesses = {}
        self._final_commits = {}

    def _validate_prepared(self, prepared):
        if not isinstance(prepared, PreparedExact) or prepared.owner is not self:
            raise PolicyInputError("Prepared decision belongs to another runtime")
        if prepared.memory._active_handle is not prepared.decision:
            raise PolicyInputError("Prepared decision is stale")

    def proposed_source(self, prepared, draft_tool_calls):
        """Propose one deterministic source before the random timing gate.

        A conservative gap uses its original source. For an already visible
        binding, a unique additional hidden source supplies a false-trigger
        opportunity without substituting a random retrieval algorithm.
        """
        self._validate_prepared(prepared)
        if prepared.render is None:
            return None
        handle = prepared.decision
        chosen = set(handle.selection.selected_event_ids)
        gap = detect_exact_source_gap(handle.store,
            visible_source_indices=prepared.visible_source_indices,
            source_cutoff=prepared.source_cutoff, draft_tool_calls=draft_tool_calls)
        event_id = gap.event_id if gap.status == "gap" else None
        if gap.reason == "all_bindings_visible":
            native = [message.to_dict() for message in handle.store.messages]
            for binding in sorted(gap.bindings, key=lambda item: item.argument_path):
                sources = [event.event_id for event in handle.store.events
                    if event.complete and event.event_id not in chosen
                    and max(event.source_indices) < prepared.source_cutoff
                    and not set(event.source_indices) & prepared.visible_source_indices
                    and any(message_has_literal(native[index], binding.value)
                            for index in event.source_indices)]
                if len(sources) == 1:
                    event_id = sources[0]
                    break
        if event_id is None:
            return None
        event = handle.store.event(event_id)
        if (event_id in chosen or set(event.source_indices) & prepared.visible_source_indices):
            return None
        ids = _ordered_ids(handle.store, chosen | {event_id})
        return event_id if _checked_cost(handle._cost_fn, ids) <= prepared.memory._selection_budget else None

    def _draw(self, prepared, purpose):
        # Independent, reproducible streams; run/arm names cannot change a draw.
        key = [VERSION, self.phase4_policy["random_seed"],
               prepared.decision.store.session_id, prepared.decision.decision_key, purpose]
        encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()
        return int.from_bytes(hashlib.sha256(encoded).digest(), "big")

    def reconsider(self, prepared, draft_tool_calls):
        with self._lock:
            phase4_started = time.perf_counter()
            self._validate_prepared(prepared)
            if prepared.checked_result is not None:
                signature = json.dumps(draft_tool_calls, sort_keys=True, ensure_ascii=False)
                if signature != prepared.checked_signature:
                    raise PolicyInputError("A prepared decision cannot inspect a second different draft")
                return prepared.checked_result
            trigger = self.phase4_policy["trigger"]
            if trigger == "conservative":
                gap = detect_exact_source_gap(prepared.decision.store,
                    visible_source_indices=prepared.visible_source_indices,
                    source_cutoff=prepared.source_cutoff, draft_tool_calls=draft_tool_calls)
                result = super().reconsider(prepared, draft_tool_calls)
                if result["regenerate"]:
                    witness = {binding.value for binding in gap.bindings
                               if binding.source_event_ids == (gap.event_id,)}
                    self._witnesses[(id(prepared.memory), gap.event_id)] = witness
            else:
                started = time.perf_counter()
                candidate = self.proposed_source(prepared, draft_tool_calls) if trigger == "random" else None
                numerator, denominator = (self.phase4_policy["random_probability"][key]
                                          for key in ("numerator", "denominator"))
                coin = self._draw(prepared, "trigger")
                fired = trigger == "random" and candidate is not None and coin * denominator < numerator * (1 << 256)
                event_id = candidate if fired else None
                trace = {"version": VERSION, "status": "no_op",
                    "reason": "recovery_disabled" if trigger == "off" else "random_not_triggered",
                    "candidate_event_id": candidate, "random_gate_passed": fired,
                    "decision_index": prepared.decision.decision_index,
                    "upgrade_count": 0, "regeneration_allowed": False,
                    "judges_action_correctness": False}
                regenerate = False
                if event_id is not None:
                    try:
                        selection = prepared.memory.upgrade_decision(prepared.decision, event_id)
                    except BudgetExceeded as error:
                        trace.update(status="abstain", reason="budget_exhausted",
                                     required_cost_bytes=error.required_cost, budget_bytes=error.budget)
                    else:
                        old_raw = prepared.counts["memory_runtime"].get("raw_history_event_ids")
                        prepared.messages, prepared.counts = prepared.render(selection, started)
                        if old_raw is not None:
                            new_raw = set(prepared.counts["memory_runtime"]["raw_history_event_ids"])
                            prepared.counts["memory_runtime"]["raw_body_evicted_on_upgrade"] = [
                                event for event in old_raw if event not in new_raw]
                        trace.update(status="gap", reason="random_valid_source_gate",
                                     upgrade_count=1, regeneration_allowed=True, upgraded_event_id=event_id)
                        regenerate = True
                trace["controller_wall_sec"] = time.perf_counter() - started
                prepared.counts["memory_runtime"]["exact_recovery"] = trace
                result = {"regenerate": regenerate, "messages": prepared.messages,
                          "counts": prepared.counts, "decision": trace}
                prepared.checked_signature = json.dumps(draft_tool_calls, sort_keys=True, ensure_ascii=False)
                prepared.checked_result = result
            result["counts"]["memory_runtime"]["phase4_policy"] = self.phase4_policy
            result["counts"]["memory_runtime"]["phase4_controller_wall_sec"] = time.perf_counter() - phase4_started
            return result

    def commit_final(self, prepared, final_tool_calls):
        """Renew existing visible leases from final native argument references only."""
        with self._lock:
            started = time.perf_counter()
            self._validate_prepared(prepared)
            if prepared.checked_result is None:
                raise PolicyInputError("Final commit requires completed reconsideration")
            memory, handle = prepared.memory, prepared.decision
            key = (id(memory), handle.decision_key)
            signature = json.dumps(final_tool_calls, sort_keys=True, ensure_ascii=False)
            previous = self._final_commits.get(key)
            if previous is not None:
                if previous[0] != signature:
                    raise PolicyInputError("A decision cannot commit a different final draft")
                return previous[1]
            values, parse_status = _final_strings(final_tool_calls)
            visible = set(handle.visible_event_ids)
            visible.update(prepared.counts["memory_runtime"].get("selected_event_ids", ()))
            renewals = []
            if self.phase4_policy["retention"] == "final_reference":
                for event_id, lease in list(memory._leases.items()):
                    witnesses = self._witnesses.get((id(memory), event_id), set())
                    if event_id not in visible or not values.intersection(witnesses):
                        continue
                    expiry = handle.decision_index + self.config.lease_decisions
                    if expiry > lease.expires_at_decision:
                        memory._leases[event_id] = replace(lease, expires_at_decision=expiry)
                        renewals.append({"event_id": event_id,
                            "previous_expiry": lease.expires_at_decision, "new_expiry": expiry})
            result = {"version": VERSION, "retention": self.phase4_policy["retention"],
                      "decision_index": handle.decision_index, "parse_status": parse_status,
                      "renewals": renewals, "new_leases": 0, "extra_generation": 0,
                      "active_lease_expiry": {event: lease.expires_at_decision
                                              for event, lease in memory._leases.items()},
                      "wall_sec": time.perf_counter() - started}
            self._final_commits[key] = (signature, result)
            return result
