"""Diagnostic provenance, reuse, and branch accounting using CPU fixtures."""
import copy
import json
from pathlib import Path

import pytest

from test_t02 import _plan, ExactFakeAdapter
import t02
import recoverability as diagnostic


def artifact(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return {"artifact": str(path), "artifact_sha256": t02._file_digest(path)}


def inputs(tmp_path, *, decision_key="turn-1/step-0"):
    parent = _plan(tmp_path)
    for state in parent["states"]:
        state["decision_key"] = decision_key
    results = t02.execute_plan(parent, ExactFakeAdapter())
    for index, row in enumerate(results["results"]):
        row["official_outcome"].update(artifact(tmp_path, f"outcome-{index}.json", row["official_outcome"]))
    reviews = []
    snapshots = {row["state_id"]: row for row in results["snapshots"]}
    for index, state in enumerate(parent["states"]):
        snapshot = snapshots[state["state_id"]]
        prefix = {"source": "official", "all_previous_turns_success": True,
            "current_turn_step": 0, "decision_key": state["decision_key"],
            "snapshot_id": snapshot["snapshot_id"], "component_digests": snapshot["component_digests"]}
        prefix.update(artifact(tmp_path, f"prefix-{index}.json", prefix))
        reviews.append({"state_id": state["state_id"], "state_sha256": t02._digest(state),
            "prefix_validity": prefix, "support_annotations": [{"reviewed": True,
                "quote": "observed evidence", "support_reason": "test-only source relation"}]})
    return parent, results, reviews


class DiagnosticFake(ExactFakeAdapter):
    def __init__(self, tmp_path, *, support_status="admitted", fail=False):
        super().__init__()
        self.tmp_path, self.support_status, self.fail = tmp_path, support_status, fail
        self.executed = []
        self.active = None

    def capture_state(self, state, *, frozen_policy):
        self.active = super().capture_state(state, frozen_policy=frozen_policy)
        return self.active

    def restore_state(self, snapshot, *, frozen_policy):
        assert snapshot == self.active, "Activate the exact live snapshot before restoring"
        return super().restore_state(snapshot, frozen_policy=frozen_policy)

    def capabilities(self):
        return {**super().capabilities(), "recoverability_diagnostic": {
            "known_support": True, "full_history": True}}

    def prepare_support(self, state, annotations):
        return {"candidate_ids": ["observed-support"] if self.support_status == "admitted" else [],
                "receipt": {"status": self.support_status}}

    def run_diagnostic(self, state, branch, *, frozen_policy, restore_receipt):
        self.executed.append(branch["branch_id"])
        if self.fail:
            raise RuntimeError("test interrupted branch")
        outcome = {"source": "official", "scorer": "cpu-fixture-only",
                   "turn_success": True, "task_success": False}
        outcome.update(artifact(self.tmp_path, f"diagnostic-{state['state_id']}-{branch['branch_id']}.json", outcome))
        return {"schema": diagnostic.BRANCH_SCHEMA, "state_id": state["state_id"],
            "branch_id": branch["branch_id"], "snapshot_id": restore_receipt["snapshot_id"],
            "restore_receipt_sha256": t02._digest(restore_receipt),
            "frozen_policy_sha256": t02._digest(frozen_policy), "execution_status": "complete",
            "execution_receipt": diagnostic._branch_receipt(branch), "official_outcome": outcome,
            "cost": {"measurement_scope": "cpu-fixture-only", "generation_calls": 1}}


def test_batch_uses_valid_failure_and_one_state_per_task(tmp_path):
    parent, results, reviews = inputs(tmp_path)
    plan = diagnostic.build_plan(parent, results, reviews)
    assert len(plan["states"]) == 3
    assert len({item["state"]["task_group_id"] for item in plan["states"]}) == 3
    assert plan["max_extra_branch_executions"] == 24
    assert plan["training_eligible"] is False
    assert all(len(item["reused_results"]) == 3 for item in plan["states"])
    assert all(item["reason"] == "one_state_per_task_group" for item in plan["skipped"])


def test_unknown_prefix_and_missing_annotations_are_not_eligible(tmp_path):
    parent, results, reviews = inputs(tmp_path)
    for review in reviews:
        review["prefix_validity"]["all_previous_turns_success"] = None
    plan = diagnostic.build_plan(parent, results, reviews)
    assert plan["states"] == []
    assert plan["status"] == "waiting_for_eligible_reviewed_T02_states"
    assert diagnostic.build_plan(parent, results, [])["states"] == []


def test_later_turn_start_uses_environment_step_and_exact_prefix_snapshot(tmp_path):
    parent, results, reviews = inputs(tmp_path, decision_key="turn-2/step-6")
    assert len(diagnostic.build_plan(parent, results, reviews)["states"]) == 3
    reviews[0]["prefix_validity"]["snapshot_id"] = "other-snapshot"
    with pytest.raises(ValueError, match="Prefix validity.*another"):
        diagnostic.build_plan(parent, results, reviews)


def test_replacement_live_snapshot_cannot_reuse_old_t02_labels(tmp_path):
    plan = diagnostic.build_plan(*inputs(tmp_path), max_states=1)
    adapter = DiagnosticFake(tmp_path)
    original_capture = adapter.capture_state

    def replaced(state, **kwargs):
        return {**original_capture(state, **kwargs), "snapshot_id": "new-task-attempt"}

    adapter.capture_state = replaced
    with pytest.raises(ValueError, match="Live diagnostic snapshot differs"):
        diagnostic.execute_plan(plan, adapter)
    assert adapter.executed == [] and adapter.restores == []


def test_prefix_receipt_cannot_override_official_artifact(tmp_path):
    parent, results, reviews = inputs(tmp_path)
    prefix = reviews[0]["prefix_validity"]
    data = json.loads(Path(prefix["artifact"]).read_text())
    data["all_previous_turns_success"] = False
    prefix.update(artifact(tmp_path, "failed-prefix.json", data))
    with pytest.raises(ValueError, match="differs from its official artifact"):
        diagnostic.build_plan(parent, results, reviews)


def test_cannot_reuse_old_snapshot_or_changed_official_artifact(tmp_path):
    parent, results, reviews = inputs(tmp_path)
    changed = copy.deepcopy(results)
    changed["results"][0]["snapshot_id"] = "legacy-different-task-run"
    with pytest.raises(ValueError, match="different starting snapshot"):
        diagnostic.build_plan(parent, changed, reviews)
    Path(results["results"][0]["official_outcome"]["artifact"]).write_text("tampered")
    with pytest.raises(ValueError, match="changed evidence"):
        diagnostic.build_plan(parent, results, reviews)


def test_executes_only_extra_branches_restoring_each(tmp_path):
    plan = diagnostic.build_plan(*inputs(tmp_path), max_states=2)
    adapter = DiagnosticFake(tmp_path)
    output = tmp_path / "diagnostic.json"
    result = diagnostic.execute_plan(plan, adapter, output=output)
    assert adapter.executed == ["known_support", "full_history"] * 2
    assert len(adapter.restores) == 6
    assert result["attempted_extra_branches"] == result["completed_extra_branches"] == 4
    assert result["status"] == "complete" and result["training_eligible"] is False
    assert all(row["diagnosis"] == "ordinary_recovery_already_rescues_current_turn" for row in result["states"])
    with pytest.raises(ValueError, match="already exists"):
        diagnostic.execute_plan(plan, adapter, output=output)


def test_infeasible_support_is_recorded_and_not_executed(tmp_path):
    plan = diagnostic.build_plan(*inputs(tmp_path), max_states=1)
    adapter = DiagnosticFake(tmp_path, support_status="selected_support_set_not_admitted_under_b0")
    result = diagnostic.execute_plan(plan, adapter)
    assert adapter.executed == ["full_history"]
    assert result["states"][0]["diagnosis"] == "inspect_packaging_or_initial_allocation"


def test_failed_branch_is_counted_and_cannot_be_silently_retried(tmp_path):
    plan = diagnostic.build_plan(*inputs(tmp_path), max_states=1)
    output = tmp_path / "failed.json"
    with pytest.raises(RuntimeError, match="interrupted"):
        diagnostic.execute_plan(plan, DiagnosticFake(tmp_path, fail=True), output=output)
    result = json.loads(output.read_text())
    assert result["attempted_extra_branches"] == 1 and result["completed_extra_branches"] == 0
    assert result["states"][0]["branches"][0]["execution_status"] == "failed"
    assert result["status"] == "incomplete"


def test_no_diagnostic_adapter_cannot_start(tmp_path):
    plan = diagnostic.build_plan(*inputs(tmp_path))
    with pytest.raises(t02.T02CapabilityError, match="known-support/full-history"):
        diagnostic.execute_plan(plan, ExactFakeAdapter())


def test_rescue_classification_does_not_count_unknown_ordinary_as_failure():
    row = {"support": {"receipt": {"status": "admitted"}}, "reused_results": [],
           "branches": [{"branch_id": "known_support", "execution_status": "complete",
                         "official_outcome": {"turn_success": True, "task_success": False}}]}
    assert diagnostic.classify(row) == "known_support_rescues_ordinary_comparison_incomplete"
    row["reused_results"] = [{"branch_id": name, "official_outcome": {"turn_success": False}}
                              for name in ("A1", "A2")]
    assert diagnostic.classify(row) == "inspect_candidate_recall_ranking_or_selection"
