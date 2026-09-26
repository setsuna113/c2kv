"""Diagnostic-only interventions on an already restored, held T02 actor.

The BFCL adapter remains responsible for environment restoration, official
scoring, remaining execution limits, and retaining the live snapshot. These
helpers make no model, checkpoint, seed, or tool-environment substitutions.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

from recoverability_support import plan_support, commit_support


def _held(actor):
    if actor._held is None or actor._scope is None:
        raise ValueError("Diagnostic intervention requires a restored held T02 decision")
    return actor._held


def review_packet(actor, planned_state):
    """Export observed source material for review before releasing the snapshot."""
    import t02
    from benchmarks.memory_runtime.recovery.evidence_units import build_catalog

    held = _held(actor)
    state = held["selection"]["decision"]["selection_state"]
    if (planned_state["state_id"] != state["state_id"]
            or planned_state["decision_key"] != held["payload"]["decision_key"]):
        raise ValueError("Review packet must refer to the currently held state")
    prepared, controller = held["prepared"], actor.runner.controller
    units = build_catalog(prepared._store, controller.tokenizer, controller.gp["U"])
    return {"schema": "recoverability-review-packet-v1", "diagnostic_only": True,
        "state_id": planned_state["state_id"], "state_sha256": t02._digest(planned_state),
        "observed_archive_messages": [message.to_dict() for message in prepared._store.messages],
        "static_units": [{**unit.to_receipt(), "text": unit.text} for unit in units],
        "draft": copy.deepcopy(state["draft"]), "q": copy.deepcopy(state["q"]),
        "source_scope": "held_decision_observed_prefix_only",
        "support_annotations": [], "prefix_validity": None}


def prepare_support(actor, annotations):
    selected, receipt = plan_support(actor.runner.controller, _held(actor)["prepared"], annotations)
    return {"candidate_ids": [unit.unit_id for unit in selected], "receipt": receipt}


def submit_support(actor, annotations, *, expected_candidate_ids):
    """Admit verified archive units directly, then regenerate exactly once."""
    held, runner = _held(actor), actor.runner
    controller, prepared = runner.controller, held["prepared"]
    selected, receipt = plan_support(controller, prepared, annotations)
    ids = [unit.unit_id for unit in selected]
    if receipt["status"] != "admitted" or ids != expected_candidate_ids:
        raise ValueError("Known support changed or no longer fits the frozen budget")
    applied = commit_support(controller, prepared, selected)
    if not applied["regenerate"]:
        raise ValueError("Diagnostic known support was not appended")
    return _regenerate(actor, applied["memory"], applied["metadata"],
        {"branch": "known_support", "source_receipt": receipt,
         "same_B0": True, "diagnostic_only": True})


def full_history_controller(compressed_controller):
    """Use the existing Full-original packer with an explicit reference budget."""
    from benchmarks.memory_runtime.event_native_controls import EventNativeOnePassController
    from benchmarks.memory_runtime.event_native_s0_policy import _validate_append_only_tools

    base = compressed_controller._packer
    packing = asdict(base.packing)
    context = base.model_context or packing["max_sequence_tokens"]
    sequence_cap = min(context, packing["max_sequence_tokens"])

    class FullReference(EventNativeOnePassController):
        def prepare(self, payload, *, ratio, max_new_tokens):
            session_id, _, _, tools, tools_json, _ = self._validate_request(payload, ratio, max_new_tokens)
            previous = self._sessions.get(session_id)
            if previous is not None and previous.tools_json != tools_json:
                _validate_append_only_tools(previous.tools_json, tools)
                # BFCL can reveal new tools at a later turn. Keep the same
                # history and decision clock, with the same append-only rule
                # as the frozen compressed actor.
                self._sessions[session_id] = replace(previous, tools_json=tools_json)
            try:
                prepared = super().prepare(payload, ratio=ratio, max_new_tokens=max_new_tokens)
            except Exception:
                if previous is not None:
                    self._sessions[session_id] = previous
                raise
            prepared.metadata.update(
                common_raw_prompt_tokens=prepared.metadata["fixed_baseline_tokens"],
                actual_history_bytes=prepared.metadata["history_bytes"],
                diagnostic_only=True, same_B0=False,
                original_B0_bytes=min(compressed_controller.policy_config.history_budget_bytes,
                                      compressed_controller.policy_config.workspace_budget_bytes),
                reference_history_cap_bytes=sequence_cap * self.kv_bytes_per_token)
            return prepared

    reference = FullReference(base.tokenizer, packing=packing, policy=base.policy,
                             view_mode="full_original", model_context=context)
    reference.kv_bytes_per_token = compressed_controller.kv_bytes_per_token
    # The generation guard still checks measured bytes and context limits. This
    # cap is intentionally a full-history reference, not the deployed B0 budget.
    reference.policy_config = SimpleNamespace(
        history_budget_bytes=sequence_cap * reference.kv_bytes_per_token,
        workspace_budget_bytes=sequence_cap * reference.kv_bytes_per_token)
    return reference


def submit_full_history(actor):
    """Regenerate from all observed history and keep Full-original thereafter."""
    held, runner = _held(actor), actor.runner
    reference = full_history_controller(runner.controller)
    prepared = reference.prepare(held["payload"], ratio=runner.ratio,
                                 max_new_tokens=runner.max_new_tokens)
    decision_index = held["prepared"].metadata["decision_index"]
    prepared.metadata["decision_index"] = decision_index
    reference._sessions[held["payload"]["session_id"]].decision_index = decision_index
    # Subsequent actor.generate calls now use this same full-history route.
    # The next T02 restore reinstates the original compressed controller.
    runner.controller = reference
    return _regenerate(actor, prepared.memory, prepared.metadata,
        {"branch": "full_history", "diagnostic_only": True, "same_B0": False,
         "continuation_memory_policy": "full_observed_history_every_decision"})


def _regenerate(actor, memory, metadata, receipt):
    held, runner = _held(actor), actor.runner
    record = held["record"]
    try:
        record["generation_trace"][-1]["discarded"] = True
        result, draft = actor._generate(memory, metadata, record, "regeneration")
        record.update(status="ok", diagnostic=receipt, response=actor._response(result, draft))
        actor._end_scope()
        record["session_cache_after"] = runner.generator.session_cache_info()
        runner._totals(record)
        actor._save_record(record)
        return copy.deepcopy(record["response"])
    except Exception as error:
        record.update(status="failed", diagnostic=receipt, response=None,
                      error={"type": type(error).__name__, "message": str(error)})
        actor._save_record(record)
        raise
    finally:
        actor._held = None
        actor._end_scope()
