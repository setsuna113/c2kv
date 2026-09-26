"""Small, diagnostic-only continuations of verified T02 failure states.

The T02 owner retains the snapshots and owns device execution. This module
reuses its A0/A1/A2 results and schedules only known-support and full-history
branches. Support annotations are offline research data, never actor prose.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import t02

PLAN_SCHEMA = "recoverability-plan-v1"
RESULT_SCHEMA = "recoverability-results-v1"
BRANCH_SCHEMA = "recoverability-branch-result-v1"
MAX_STATES = 12
MAX_EXTRA_BRANCHES = 24
FULL_POLICY = "full_observed_history_every_decision"


def _verify_file(receipt):
    path = Path(receipt["artifact"])
    if not path.is_file() or t02._file_digest(path) != receipt["artifact_sha256"]:
        raise ValueError(f"Missing or changed evidence artifact: {path}")


def build_plan(parent, results, reviews, *, max_states=MAX_STATES):
    """Select one reviewed failure per task, without examining rescue outcomes.

The first batch uses turn-start states, avoiding an additional claim that
arbitrary partial tool execution is reversible. Earlier turns must have passed
the official checker. Fewer than twelve eligible states is an honest batch.
"""
    if type(max_states) is not int or not 1 <= max_states <= MAX_STATES:
        raise ValueError("max_states must be between 1 and 12")
    t02.check_plan(parent)
    indexed = t02._index_results(results, parent)
    if results.get("schema") != t02.RESULT_SET_SCHEMA:
        raise ValueError("Use the complete T02 result artifact, not detached labels")
    snapshots = {row["state_id"]: row for row in results.get("snapshots", [])}
    if len(snapshots) != len(results.get("snapshots", [])):
        raise ValueError("Duplicate T02 snapshots")
    forbidden, _ = t02.load_forbidden_manifests({
        row["name"]: row["path"] for row in parent["forbidden_manifests"]})
    reviews_by_id = {row["state_id"]: row for row in reviews}
    if len(reviews_by_id) != len(reviews):
        raise ValueError("Duplicate support reviews")
    states, skipped, groups = [], [], set()
    policy_sha = parent["frozen_subsequent_policy_sha256"]
    ordered = sorted(parent["states"], key=lambda row: (row["task_group_id"], row["decision_key"], row["state_id"]))
    for state in ordered:
        sid, group = state["state_id"], state["task_group_id"]
        reason = None
        review = reviews_by_id.get(sid)
        a0 = indexed.get((sid, "A0"))
        if state["task_id"] in forbidden or group in forbidden:
            raise ValueError("Diagnostic state overlaps D128/F128")
        if state.get("split") not in {"train", "calibration"}:
            reason = "not_train_or_calibration"
        elif a0 is None or a0["official_outcome"]["turn_success"] is not False:
            reason = "no_verified_A0_current_turn_failure"
        elif review is None:
            reason = "support_review_missing"
        elif group in groups:
            reason = "one_state_per_task_group"
        elif len(states) >= max_states:
            reason = "batch_cap"
        if reason:
            skipped.append({"state_id": sid, "reason": reason})
            continue
        if review.get("state_sha256") != t02._digest(state):
            raise ValueError("Support review is bound to another state")
        snapshot = t02._check_snapshot(snapshots[sid], state_id=sid, policy_sha256=policy_sha)
        prefix = review["prefix_validity"]
        if (prefix.get("source") != "official" or prefix.get("all_previous_turns_success") is not True
                or prefix.get("current_turn_step") != 0 or type(prefix.get("current_turn_step")) is not int
                or prefix.get("decision_key") != state["decision_key"]):
            skipped.append({"state_id": sid, "reason": "valid_turn_start_prefix_not_established"})
            continue
        # BFCL decision keys use a global step counter; the environment receipt
        # establishes a turn start, including starts after nonzero global steps.
        if (prefix.get("snapshot_id") != snapshot["snapshot_id"]
                or prefix.get("component_digests") != snapshot["component_digests"]):
            raise ValueError("Prefix validity is bound to another starting snapshot")
        _verify_file(prefix)
        prefix_artifact = json.loads(Path(prefix["artifact"]).read_text(encoding="utf-8"))
        if any(prefix_artifact.get(key) != prefix.get(key) for key in (
                "source", "all_previous_turns_success", "current_turn_step", "decision_key",
                "snapshot_id", "component_digests")):
            raise ValueError("Prefix validity receipt differs from its official artifact")
        annotations = review.get("support_annotations")
        if not isinstance(annotations, list) or not annotations:
            raise ValueError("Reviewed support annotations are required")
        for annotation in annotations:
            if (annotation.get("reviewed") is not True or not annotation.get("quote")
                    or not annotation.get("support_reason")):
                raise ValueError("Support needs a reviewed source quote and rationale")
        reused = []
        expected_restore = {"schema": t02.RESTORE_SCHEMA, **{key: snapshot[key] for key in
            ("state_id", "snapshot_id", "frozen_policy_sha256", "component_digests")}, "restored": True}
        for bid in t02.BRANCH_IDS:
            row = indexed.get((sid, bid))
            if row is None:
                continue
            if (row["snapshot_id"] != snapshot["snapshot_id"]
                    or row["restore_receipt_sha256"] != t02._digest(expected_restore)):
                raise ValueError("Reused T02 branch has a different starting snapshot")
            _verify_file(row["official_outcome"])
            reused.append(copy.deepcopy(row))
        states.append({"state": copy.deepcopy(state), "snapshot": snapshot,
            "review": copy.deepcopy(review), "reused_results": reused})
        groups.add(group)
    return {"schema": PLAN_SCHEMA, "diagnostic_only": True, "training_eligible": False,
        "parent_plan_sha256": t02._digest(parent), "parent_results_sha256": t02._digest(results),
        "frozen_subsequent_policy": copy.deepcopy(parent["frozen_subsequent_policy"]),
        "frozen_subsequent_policy_sha256": policy_sha,
        "forbidden_manifests": copy.deepcopy(parent["forbidden_manifests"]),
        "max_states": max_states, "max_extra_branch_executions": 2 * max_states,
        "selection_rule": "reviewed_A0_turn_failure_at_valid_turn_start_one_per_task",
        "full_history_policy": FULL_POLICY,
        "full_history_budget_comparable": False, "states": states, "skipped": skipped,
        "status": "planned" if states else "waiting_for_eligible_reviewed_T02_states"}


def _branch_receipt(branch):
    return {"submitted_original_draft": False,
        "initial_regeneration_count": 1,
        "observations_replayed_from_other_branch": False,
        "actor_sampling_tools_and_remaining_limits_unchanged": True,
        "continuation_memory_policy": branch["continuation_memory_policy"],
        "support_annotations_in_actor_input": False,
        "recovery_candidate_ids": branch["candidate_ids"]}


def execute_plan(plan, adapter, *, output=None):
    """Execute through the T02 owner's live snapshot adapter.

Adapter extensions: prepare_support(state, annotations) returns candidate_ids
and a source/B0 receipt; run_diagnostic accepts the same keyword arguments as
T02 run_branch. No implicit fresh task run can replace a released snapshot.
The optional output is checkpointed before each attempt, so a failed attempt
cannot silently disappear or be auto-rerun by this entry point.
"""
    if plan.get("schema") != PLAN_SCHEMA or plan.get("training_eligible") is not False:
        raise ValueError("Expected a diagnostic-only plan")
    if not (1 <= plan["max_states"] <= MAX_STATES and len(plan["states"]) <= plan["max_states"]
            and plan["max_extra_branch_executions"] == 2 * plan["max_states"]):
        raise ValueError("Diagnostic budget changed")
    if output and Path(output).exists():
        raise ValueError("Diagnostic output already exists; inspect it before any rerun")
    caps = t02._check_capabilities(adapter.capabilities())
    if caps.get("recoverability_diagnostic") != {"known_support": True, "full_history": True}:
        raise t02.T02CapabilityError("Adapter lacks known-support/full-history continuation")
    policy = plan["frozen_subsequent_policy"]
    if t02._digest(policy) != plan["frozen_subsequent_policy_sha256"]:
        raise ValueError("Frozen policy changed")
    result = {"schema": RESULT_SCHEMA, "plan_sha256": t02._digest(plan),
        "diagnostic_only": True, "training_eligible": False, "status": "running",
        "attempted_extra_branches": 0, "completed_extra_branches": 0,
        "quality_label": "preliminary, n=1", "states": []}
    def save():
        if output:
            t02._write_json(Path(output), result)
    for item in plan["states"]:
        state, snapshot = item["state"], item["snapshot"]
        t02._check_snapshot(snapshot, state_id=state["state_id"],
                            policy_sha256=plan["frozen_subsequent_policy_sha256"])
        row = {"state_id": state["state_id"], "task_id": state["task_id"],
               "reused_results": item["reused_results"], "branches": []}
        result["states"].append(row)
        save()
        active = adapter.capture_state(state, frozen_policy=policy)
        if active != snapshot:
            raise ValueError("Live diagnostic snapshot differs from the reused T02 snapshot")
        t02._check_restore(adapter.restore_state(snapshot, frozen_policy=policy), snapshot=snapshot)
        support = adapter.prepare_support(state, item["review"]["support_annotations"])
        row["support"] = copy.deepcopy(support)
        branches = []
        if support["receipt"]["status"] == "admitted":
            if not support["candidate_ids"]:
                raise ValueError("Admitted support needs at least one source unit")
            branches.append({"branch_id": "known_support", "candidate_ids": support["candidate_ids"],
                "support_receipt": support["receipt"], "continuation_memory_policy": "frozen_T02_policy"})
        branches.append({"branch_id": "full_history", "candidate_ids": [],
                         "continuation_memory_policy": FULL_POLICY})
        for branch in branches:
            if result["attempted_extra_branches"] >= plan["max_extra_branch_executions"]:
                raise ValueError("Diagnostic branch budget exhausted")
            restore = t02._check_restore(adapter.restore_state(snapshot, frozen_policy=policy), snapshot=snapshot)
            entry = {"branch_id": branch["branch_id"], "execution_status": "attempted"}
            row["branches"].append(entry)
            result["attempted_extra_branches"] += 1
            save()
            try:
                raw = adapter.run_diagnostic(state, branch, frozen_policy=policy, restore_receipt=restore)
                expected = {"schema": BRANCH_SCHEMA, "state_id": state["state_id"],
                    "branch_id": branch["branch_id"], "snapshot_id": snapshot["snapshot_id"],
                    "restore_receipt_sha256": t02._digest(restore),
                    "frozen_policy_sha256": plan["frozen_subsequent_policy_sha256"],
                    "execution_status": "complete", "execution_receipt": _branch_receipt(branch)}
                if any(raw.get(key) != value for key, value in expected.items()):
                    raise ValueError("Diagnostic branch does not match its frozen intervention")
                outcome = t02._check_official_outcome(raw.get("official_outcome"))
                _verify_file(outcome)
                if not isinstance(raw.get("cost"), dict) or not raw["cost"]:
                    raise ValueError("Both diagnostic branches need separate measured cost receipts")
                entry.update(t02._json_copy(raw))
                result["completed_extra_branches"] += 1
            except Exception as error:
                entry.update(execution_status="failed", error_type=type(error).__name__, error=str(error))
                result["status"] = "incomplete"
                save()
                raise
            save()
        row["diagnosis"] = classify(row)
        save()
    result["status"] = "complete" if plan["states"] else "no_eligible_states"
    result["diagnosis_counts"] = dict(Counter(item["diagnosis"] for item in result["states"]))
    save()
    return result


def classify(row):
    """Decision clues at current-turn scope, never a claimed ability upper bound."""
    results = {item["branch_id"]: item for item in row["branches"]}
    def success(branch):
        item = results.get(branch, {})
        return item.get("official_outcome", {}).get("turn_success") if item.get("execution_status") == "complete" else None
    known, full = success("known_support"), success("full_history")
    status = row["support"]["receipt"]["status"]
    ordinary = {item["branch_id"]: item["official_outcome"]["turn_success"] for item in row["reused_results"]}
    if known is True:
        if all(ordinary.get(bid) is False for bid in ("A1", "A2")):
            return "inspect_candidate_recall_ranking_or_selection"
        if any(ordinary.get(bid) is True for bid in ("A1", "A2")):
            return "ordinary_recovery_already_rescues_current_turn"
        return "known_support_rescues_ordinary_comparison_incomplete"
    if full is False or known is False:
        return "inspect_execution_task_understanding_or_remaining_errors"
    if status == "already_exact_visible":
        return "support_already_visible_inspect_evidence_use"
    if status in {"support_not_representable_in_static_units", "selected_support_set_not_admitted_under_b0"}:
        return "inspect_packaging_or_initial_allocation"
    return "incomplete_or_other_source_constraint"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t02-plan", type=Path, required=True)
    parser.add_argument("--t02-results", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-states", type=int, default=MAX_STATES)
    args = parser.parse_args(argv)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    plan = build_plan(read(args.t02_plan), read(args.t02_results), read(args.reviews), max_states=args.max_states)
    if args.output.exists():
        raise ValueError("Plan output already exists; choose a new path")
    t02._write_json(args.output, plan)
    print(json.dumps({"status": plan["status"], "states": len(plan["states"]),
                      "max_extra_branch_executions": plan["max_extra_branch_executions"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
