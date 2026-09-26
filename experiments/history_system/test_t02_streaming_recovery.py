"""Focused CPU tests for per-source T02 streaming recovery."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "runtime" / "python"))
sys.path.insert(0, str(HERE / "runtime"))

import t02
import t02_streaming_launch as launch
import t02_streaming_recovery as streaming
from benchmarks.memory_runtime.recovery.set_training import _read_dataset
from test_t02_recovery_plan import _fresh_candidate, _source


class FakeAdapter:
    def __init__(self, states):
        self.live = {state["state_id"] for state in states}
        self.active = None
        self.counts = {}

    def capabilities(self):
        return {
            "schema": t02.CAPABILITY_SCHEMA,
            "exact_same_state_restore": True,
            "components": list(t02.REQUIRED_COMPONENTS),
            "frozen_subsequent_policy": True,
            "official_turn_outcome": True,
            "official_task_outcome": True,
        }

    def capture_state(self, state, *, frozen_policy):
        assert state["state_id"] in self.live
        self.active = state["state_id"]
        return {
            "schema": t02.SNAPSHOT_SCHEMA,
            "state_id": state["state_id"],
            "snapshot_id": f"snap-{state['state_id']}",
            "frozen_policy_sha256": t02._digest(frozen_policy),
            "component_digests": {
                name: f"{index + 1:064x}"
                for index, name in enumerate(t02.REQUIRED_COMPONENTS)
            },
        }

    def restore_state(self, snapshot, *, frozen_policy):
        assert snapshot["state_id"] == self.active
        return {
            "schema": t02.RESTORE_SCHEMA,
            "state_id": snapshot["state_id"],
            "snapshot_id": snapshot["snapshot_id"],
            "frozen_policy_sha256": snapshot["frozen_policy_sha256"],
            "component_digests": copy.deepcopy(snapshot["component_digests"]),
            "restored": True,
        }

    def run_branch(self, state, branch, *, frozen_policy, restore_receipt):
        state_id = state["state_id"]
        count = self.counts.get(state_id, 0) + 1
        self.counts[state_id] = count
        if count == 3:
            self.live.remove(state_id)
            self.active = None
        return {
            "schema": t02.RESULT_SCHEMA,
            "state_id": state_id,
            "branch_id": branch["branch_id"],
            "candidate_ids": copy.deepcopy(branch["candidate_ids"]),
            "snapshot_id": restore_receipt["snapshot_id"],
            "restore_receipt_sha256": t02._digest(restore_receipt),
            "frozen_policy_sha256": t02._digest(frozen_policy),
            "execution_status": "complete",
            "execution_receipt": {
                "submitted_original_draft": branch["submit_original_draft"],
                "recovery_candidate_ids": copy.deepcopy(branch["candidate_ids"]),
                "regeneration_count": 1 if branch["regenerate"] else 0,
                "continued_with_frozen_policy": True,
                "observations_replayed_from_other_branch": False,
            },
            "official_outcome": {
                "source": "official",
                "scorer": "fake-official",
                "artifact": f"official/{state_id}/{branch['branch_id']}.json",
                "artifact_sha256": "a" * 64,
                "turn_success": branch["branch_id"] != "A0",
                "task_success": branch["branch_id"] == "A2",
            },
        }


def _task_candidates(source_plan, source):
    by_slot = {
        (state["task_id"], state["draft_kind"]): state
        for state in source_plan["states"]
    }
    return [
        _fresh_candidate(by_slot[(slot["task_id"], slot["draft_kind"])], index)
        for index, slot in enumerate(source["expected_slots"])
    ]


def _write_package(tmp_path, source_plan, recovery):
    (tmp_path / "recovery").mkdir()
    (tmp_path / "recovery" / "source_plan.json").write_text(
        json.dumps(source_plan), encoding="utf-8"
    )
    (tmp_path / "recovery" / "recovery_contract.json").write_text(
        json.dumps(recovery), encoding="utf-8"
    )
    task_ids = [row["task_id"] for row in recovery["source_tasks"]]
    parallel = {
        "schema": "t02-parallel-contract-v1",
        "dynamic_source_queue": True,
        "recovery_contract_path": "recovery/recovery_contract.json",
        "prior_complete_branches": recovery["budget"]["complete_branches_before_recovery"],
        "remaining_branch_budget": recovery["budget"]["new_complete_branch_executions"],
        "prior_source_starts": recovery["source_budget"]["prior_source_starts"],
        "cumulative_source_start_cap": (
            recovery["source_budget"]["prior_source_starts"] + len(task_ids)
        ),
        "max_source_tasks_per_worker": len(task_ids),
        "snapshot_cap_per_worker": 2,
        "workers": [{
            "worker_id": "npu1", "physical_device": 1, "port": 31001,
            "task_ids": task_ids,
        }],
    }
    (tmp_path / "contract.json").write_text(json.dumps(parallel), encoding="utf-8")
    return parallel


def test_one_source_executes_and_persists_each_branch_before_release(tmp_path):
    source_plan, recovery = _source(tmp_path)
    source = recovery["source_tasks"][0]
    candidates = _task_candidates(source_plan, source)
    plan = streaming.build_task_plan(
        recovery, source["task_id"], candidates,
        source_plan=source_plan,
        frozen_subsequent_policy={"name": "streaming", "version": 1},
    )
    adapter = FakeAdapter(plan["states"])
    progress = []
    results = streaming.execute_task_plan(
        plan, adapter,
        on_progress=lambda artifact: progress.append(copy.deepcopy(artifact)),
    )
    assert [row["complete_branch_executions"] for row in progress] == list(
        range(1, 3 * len(plan["states"]) + 1)
    )
    assert results["complete_branch_executions"] == 3 * len(plan["states"])
    assert adapter.live == set()
    assert results["plan_sha256"] == streaming._digest(plan)


def test_multi_plan_labels_keep_every_real_plan_and_load_in_trainer(tmp_path):
    source_plan, recovery = _source(tmp_path)
    task_labels = []
    for source_index, source in enumerate(recovery["source_tasks"]):
        candidates = _task_candidates(source_plan, source)
        for state_index, candidate in enumerate(candidates):
            candidate["state_id"] = f"fresh-{source_index}-{state_index}"
        plan = streaming.build_task_plan(
            recovery, source["task_id"], candidates,
            source_plan=source_plan,
            frozen_subsequent_policy={"name": "streaming", "version": 1},
        )
        results = streaming.execute_task_plan(plan, FakeAdapter(plan["states"]))
        labels = streaming.label_task_results(plan, results)
        labels.update(
            plan_file_sha256="b" * 64,
            label_file_sha256="c" * 64,
            label_payload_sha256=streaming._digest(labels),
        )
        task_labels.append(labels)
    preserved = recovery["preserved_states"][0]
    preserved_labels = {
        "plan_sha256": "old-plan",
        "rows": [{
            "state_id": preserved["state_id"],
            "task_id": preserved["task_id"],
            "draft_kind": preserved["draft_kind"],
            "task_group_id": preserved["task_group_id"],
            "split": preserved["split"],
        }],
    }
    combined = streaming.merge_label_partitions(recovery, preserved_labels, task_labels)
    assert combined["state_count"] == recovery["target"]["combined_exact_state_count"]
    assert combined["synthetic_global_plan_sha256"] is None
    assert len(combined["partitions"]) == 1 + len(recovery["source_tasks"])
    for partition in combined["partitions"][1:]:
        rows = [row for row in combined["rows"] if row["task_id"] == partition["task_id"]]
        assert {row["result_plan_sha256"] for row in rows} == {partition["plan_sha256"]}
        assert partition["state_ids"] == [row["state_id"] for row in rows]
    path = tmp_path / "combined_labels.json"
    path.write_text(json.dumps(combined), encoding="utf-8")
    loaded, source = _read_dataset(path)
    assert len(loaded) == combined["state_count"]
    assert source["dataset_schema"] == streaming.COMBINED_LABEL_SCHEMA


def test_failed_source_is_terminal_and_next_source_is_claimable(tmp_path):
    source_plan, recovery = _source(tmp_path)
    _write_package(tmp_path, source_plan, recovery)
    streaming.initialize_ledger(tmp_path)
    first = streaming.claim_task(tmp_path, "npu1")
    expected = first["expected_slot_count"] * 3
    streaming.fail_task(
        tmp_path, "npu1", first["task_id"],
        error=RuntimeError("label write failed after durable branches"),
        complete_branch_executions=expected,
    )
    second = streaming.claim_task(tmp_path, "npu1")
    assert second is not None and second["task_id"] != first["task_id"]
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["tasks"][0]["status"] == "failed_no_retry"
    assert ledger["tasks"][0]["complete_branch_executions"] == expected


def test_orphan_reconciliation_recovers_durable_partial_count(tmp_path):
    source_plan, recovery = _source(tmp_path)
    _write_package(tmp_path, source_plan, recovery)
    streaming.initialize_ledger(tmp_path)
    claim = streaming.claim_task(tmp_path, "npu1")
    task_root = streaming.task_artifact_dir(tmp_path / "workers" / "npu1", claim)
    task_root.mkdir(parents=True)
    (task_root / "partial_results.json").write_text(
        json.dumps({"complete_branch_executions": 2}), encoding="utf-8"
    )
    streaming.mark_orphaned_and_unstarted(tmp_path, ["npu1"])
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["tasks"][0]["status"] == "failed_no_retry"
    assert ledger["tasks"][0]["complete_branch_executions"] == 2
    assert all(row["status"] in streaming.TERMINAL_TASK_STATUSES for row in ledger["tasks"])


def test_unexpected_pilot_owner_exit_releases_waiters(tmp_path):
    source_plan, recovery = _source(tmp_path)
    _write_package(tmp_path, source_plan, recovery)
    streaming.initialize_ledger(tmp_path)
    pilot = streaming.claim_task(tmp_path, "npu1")
    assert pilot["task_id"] == recovery["source_tasks"][0]["task_id"]
    assert launch._reconcile_pilot_owner_failure(
        tmp_path, "npu1", RuntimeError("engine died")
    )
    assert (tmp_path / "pilot_failed.json").is_file()
    with pytest.raises(streaming.PilotGateFailed, match="engine died"):
        streaming.wait_for_pilot(tmp_path, poll_seconds=0)
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["tasks"][0]["status"] == "failed_no_retry"
    assert all(
        row["status"] == "not_started_pilot_failed"
        for row in ledger["tasks"][1:]
    )
    assert all(row["status"] in streaming.TERMINAL_TASK_STATUSES for row in ledger["tasks"])
