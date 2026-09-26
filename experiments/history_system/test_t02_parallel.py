import copy
from concurrent.futures import ThreadPoolExecutor

import pytest

import t02
import t02_parallel as parallel
from test_t02 import ExactFakeAdapter, _plan


def test_disjoint_workers_restore_only_owned_states_and_match_serial_labels(tmp_path):
    plan = _plan(tmp_path)
    owners = {row["state_id"]: f"worker{i % 2}" for i, row in enumerate(plan["states"])}
    partials = {}
    for worker_id in sorted(set(owners.values())):
        adapter = ExactFakeAdapter()
        ids = [state_id for state_id, owner in owners.items() if owner == worker_id]
        progress = []
        partials[worker_id] = t02.execute_plan(plan, adapter, state_ids=ids,
                                             on_progress=progress.append)
        assert len(adapter.restores) == 3 * len(ids)
        assert [row["complete_branch_executions"] for row in progress] == list(range(1, 3 * len(ids) + 1))
        assert all(set(row["executed_state_ids"]) == {snapshot["state_id"] for snapshot in row["snapshots"]}
                   for row in progress)
        assert {row["state_id"] for row in partials[worker_id]["results"]} == set(ids)
    merged = parallel.merge_results(plan, partials, owners)
    serial = t02.execute_plan(plan, ExactFakeAdapter())
    assert t02.label_results(plan, merged) == t02.label_results(plan, serial)
    broken = copy.deepcopy(partials)
    broken["worker0"]["results"].append(broken["worker1"]["results"][0])
    with pytest.raises(ValueError, match="another worker"):
        parallel.merge_results(plan, broken, owners)


def test_worker_cannot_restore_unknown_or_duplicate_state(tmp_path):
    plan = _plan(tmp_path)
    adapter = ExactFakeAdapter()
    for state_ids in (["absent"], [plan["states"][0]["state_id"]] * 2):
        with pytest.raises(t02.T02CapabilityError):
            t02.execute_plan(plan, adapter, state_ids=state_ids)
    assert adapter.restores == []


def test_ledger_counts_source_start_before_execution_and_never_reassigns(tmp_path):
    parallel.save(tmp_path / "contract.json", {
        "workers": [{"worker_id": "a", "task_ids": ["task-a"]},
                    {"worker_id": "b", "task_ids": ["task-b"]}],
        "excluded_started_task_ids": ["old-task"], "prior_source_starts": 102,
        "target_states": 119, "prior_complete_branches": 1})
    parallel.initialize_ledger(tmp_path)
    with ThreadPoolExecutor(2) as pool:
        leases = list(pool.map(lambda worker: parallel.claim_task(tmp_path, worker), ["a", "b"]))
    assert {row["task_id"] for row in leases} == {"task-a", "task-b"}
    with pytest.raises(RuntimeError, match="unfinished"):
        parallel.claim_task(tmp_path, "a")
    with pytest.raises(RuntimeError, match="exclusive"):
        parallel.complete_task(tmp_path, "b", "task-a", [], {})
    terminal = {
        "schema": "t02-bfcl-capacity-termination-v1",
        "reason": "capacity_infeasible",
        "terminal_status": "budget_terminated",
        "response_fabricated": False,
    }
    parallel.complete_task(
        tmp_path, "a", "task-a", [], {}, source_terminal=terminal
    )
    assert parallel.claim_task(tmp_path, "a") is None
    ledger = parallel.read(tmp_path / "ledger.json")
    assert next(row for row in ledger["tasks"] if row["task_id"] == "task-a")[
        "source_terminal"
    ] == terminal
    assert ledger["prior_source_starts"] + sum(row["status"] != "pending" for row in ledger["tasks"]) == 104


def test_ledger_rejects_past_tasks_and_cumulative_overbudget(tmp_path):
    contract = {"workers": [{"worker_id": "a", "task_ids": ["old-task"]}],
        "excluded_started_task_ids": ["old-task"], "prior_source_starts": 10,
        "target_states": 119, "prior_complete_branches": 1}
    parallel.save(tmp_path / "contract.json", contract)
    with pytest.raises(ValueError, match="previously started"):
        parallel.initialize_ledger(tmp_path)
    contract["excluded_started_task_ids"] = []
    contract["prior_source_starts"] = 104
    parallel.save(tmp_path / "contract.json", contract)
    with pytest.raises(ValueError, match="Source budget"):
        parallel.initialize_ledger(tmp_path)


def test_explicit_source_cap_charges_old_attempts_before_new_queue(tmp_path):
    contract = {"workers": [{"worker_id": "a", "task_ids": ["new-task"]}],
        "excluded_started_task_ids": [], "prior_source_starts": 104,
        "cumulative_source_start_cap": 105,
        "target_states": 119, "prior_complete_branches": 1}
    parallel.save(tmp_path / "contract.json", contract)
    parallel.initialize_ledger(tmp_path)
    assert parallel.claim_task(tmp_path, "a")["task_id"] == "new-task"
    parallel.complete_task(tmp_path, "a", "new-task", [], {})
    # A later queue append cannot reset the cumulative accounting.
    with parallel.locked_ledger(tmp_path) as ledger:
        ledger["tasks"].append({"task_id": "extra", "task_index": 1,
                                "worker_id": "a", "status": "pending"})
    with pytest.raises(RuntimeError, match="cap reached"):
        parallel.claim_task(tmp_path, "a")


def test_dynamic_queue_can_steal_static_tasks_once_and_enforces_worker_caps(tmp_path):
    task_ids = [f"task-{index:03d}" for index in range(53)]
    parallel.save(tmp_path / "contract.json", {
        "dynamic_source_queue": True,
        "max_source_tasks_per_worker": 26,
        "snapshot_cap_per_worker": 52,
        "snapshot_host_byte_cap_per_worker": 206158430208,
        "workers": [
            {"worker_id": "early", "task_ids": task_ids[:1]},
            {"worker_id": "late", "task_ids": task_ids[1:]},
        ],
        "excluded_started_task_ids": [],
        "prior_source_starts": 54,
        "cumulative_source_start_cap": 107,
        "target_states": 1,
        "prior_complete_branches": 0,
    })
    parallel.initialize_ledger(tmp_path)
    claimed = []
    for index in range(26):
        claim = parallel.claim_task(tmp_path, "early")
        claimed.append(claim["task_id"])
        candidates = [
            {"task_id": claim["task_id"], "state_id": f"state-{index}-{slot}"}
            for slot in range(2)
        ]
        parallel.complete_task(tmp_path, "early", claim["task_id"], candidates, {})
    assert claimed[1] in task_ids[1:]
    assert len(set(claimed)) == 26
    assert parallel.claim_task(tmp_path, "early") is None
    late = parallel.claim_task(tmp_path, "late")
    assert late["task_id"] not in claimed
    ledger = parallel.read(tmp_path / "ledger.json")
    assert ledger["max_source_tasks_per_worker"] == 26
    assert ledger["snapshot_cap_per_worker"] == 52
    assert ledger["snapshot_host_byte_cap_per_worker"] == 206158430208
    assert sum(row["worker_id"] == "early" for row in ledger["candidates"]) == 52


def test_dynamic_late_slot_exits_before_device_or_engine_when_queue_is_done(monkeypatch, tmp_path):
    import t02_parallel_launch as launcher

    parallel.save(tmp_path / "contract.json", {
        "launch_authorized": True,
        "dynamic_source_queue": True,
        "max_source_tasks_per_worker": 26,
        "snapshot_cap_per_worker": 52,
        "snapshot_host_byte_cap_per_worker": 206158430208,
        "workers": [
            {"worker_id": "early", "physical_device": 4, "port": 36540,
             "task_ids": ["task-a", "task-b"]},
            {"worker_id": "late", "physical_device": 6, "port": 36560,
             "task_ids": []},
        ],
        "excluded_started_task_ids": [],
        "prior_source_starts": 156,
        "cumulative_source_start_cap": 158,
        "target_states": 1,
        "prior_complete_branches": 0,
    })
    parallel.initialize_ledger(tmp_path)
    with parallel.locked_ledger(tmp_path) as ledger:
        for row in ledger["tasks"]:
            row.update(worker_id="early", status="completed")
    assert parallel.source_participants(tmp_path) == ["early"]
    monkeypatch.setattr(
        launcher, "available", lambda *args: pytest.fail("late no-work slot checked device")
    )
    monkeypatch.setattr(
        launcher, "launch", lambda *args: pytest.fail("late no-work slot launched engine")
    )
    assert launcher.slot(tmp_path, "late") == 0
    assert parallel.read(tmp_path / "workers/late/slot.json")["phase"] == "completed_no_work"


def test_coordinator_marks_partial_failed_after_one_branch_worker_fails(
        monkeypatch, tmp_path):
    plan = _plan(tmp_path)
    failed_ids = [state["state_id"] for state in plan["states"][:2]]
    completed_ids = [state["state_id"] for state in plan["states"][2:]]
    owners = {state_id: "failed" for state_id in failed_ids}
    owners.update({state_id: "completed" for state_id in completed_ids})
    parallel.save(tmp_path / "contract.json", {
        "workers": [
            {"worker_id": "failed", "task_ids": ["task-failed"]},
            {"worker_id": "completed", "task_ids": ["task-completed"]},
        ],
    })
    parallel.save(tmp_path / "state_owners.json", owners)
    for worker in ("failed", "completed"):
        parallel.save(tmp_path / "workers" / worker / "ready.json", {
            "worker_id": worker,
        })
    completed = t02.execute_plan(
        plan, ExactFakeAdapter(), state_ids=completed_ids
    )
    failed_partial = t02.execute_plan(
        plan, ExactFakeAdapter(), state_ids=failed_ids[:1]
    )
    parallel.save(
        tmp_path / "workers/completed/results.json", completed
    )
    parallel.save(
        tmp_path / "workers/failed/partial_results.json", failed_partial
    )
    parallel.save(tmp_path / "workers/failed/status.json", {
        "schema": "t02-parallel-worker-status-v1",
        "phase": "failed",
        "worker_id": "failed",
        "failure_scope": "branch_execution",
        "global_abort_required": False,
        "error_type": "RuntimeError",
        "error": "synthetic unknown branch failure",
    })
    monkeypatch.setattr(parallel, "make_global_plan", lambda root: plan)

    assert parallel.coordinate(tmp_path) == 1
    assert not (tmp_path / "abort.json").exists()
    assert (tmp_path / "workers/completed/results.json").exists()
    assert (tmp_path / "workers/failed/partial_results.json").exists()
    assert not (tmp_path / "run/results.json").exists()
    assert not (tmp_path / "run/labels.json").exists()
    summary = parallel.read(tmp_path / "run/summary.json")
    assert summary["status"] == "partial_failed"
    assert summary["completed_worker_ids"] == ["completed"]
    assert summary["failed_worker_ids"] == ["failed"]
    assert summary["complete_results_emitted"] is False
    assert summary["labels_emitted"] is False
    assert summary["training_allowed"] is False
    assert parallel.read(tmp_path / "status.json")["phase"] == "partial_failed"


def test_slot_does_not_broadcast_isolated_branch_failure(monkeypatch, tmp_path):
    import t02_parallel_launch as launcher

    parallel.save(tmp_path / "contract.json", {
        "launch_authorized": True,
        "workers": [{
            "worker_id": "npu4", "physical_device": 4, "port": 36540,
            "task_ids": ["task-a"],
        }],
    })
    parallel.save(tmp_path / "workers/npu4/status.json", {
        "schema": "t02-parallel-worker-status-v1",
        "phase": "failed",
        "worker_id": "npu4",
        "failure_scope": "branch_execution",
        "global_abort_required": False,
        "error_type": "RuntimeError",
        "error": "synthetic unknown branch failure",
    })

    class Process:
        def __init__(self, pid, returncode):
            self.pid = pid
            self.returncode = returncode

        def poll(self):
            return self.returncode

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Opener:
        def open(self, *args, **kwargs):
            return Response()

    launched = iter([Process(11, None), Process(12, 1)])
    monkeypatch.setattr(launcher, "available", lambda root, worker: {
        "physical_device": 4, "port": 36540,
    })
    monkeypatch.setattr(launcher, "launch", lambda *args: next(launched))
    monkeypatch.setattr(launcher, "owned_stop", lambda process: None)
    monkeypatch.setattr(launcher.urllib.request, "build_opener", lambda *args: Opener())

    assert launcher.slot(tmp_path, "npu4") == 1
    assert not (tmp_path / "abort.json").exists()
    status = parallel.read(tmp_path / "workers/npu4/slot.json")
    assert status["phase"] == "failed_no_retry"
    assert status["failure_scope"] == "branch_execution"
    assert status["global_abort_required"] is False


def test_supervisor_preserves_partial_failed_and_does_not_launch_training(
        monkeypatch, tmp_path):
    import t02_parallel_launch as launcher

    parallel.save(tmp_path / "contract.json", {
        "launch_authorized": True,
        "workers": [
            {"worker_id": "npu4", "physical_device": 4},
            {"worker_id": "npu6", "physical_device": 6},
        ],
    })
    launched = []

    class Process:
        def __init__(self, pid, code, *, coordinator=False):
            self.pid = pid
            self.returncode = code
            self.coordinator = coordinator

        def wait(self):
            if self.coordinator:
                parallel.save(tmp_path / "status.json", {
                    "phase": "partial_failed", "training_allowed": False,
                })
            return self.returncode

        def poll(self):
            return self.returncode

    def fake_launch(args, root, log_name):
        launched.append(list(args))
        return Process(len(launched), 1 if len(launched) == 1 else 0,
                       coordinator=len(launched) == 1)

    monkeypatch.setattr(launcher, "verify", lambda root: 3)
    monkeypatch.setattr(launcher, "initialize_ledger", lambda root: None)
    monkeypatch.setattr(launcher, "launch", fake_launch)
    monkeypatch.setattr(launcher, "owned_stop", lambda process: None)

    assert launcher.supervise(tmp_path) == 1
    assert len(launched) == 3
    assert all("train.sh" not in " ".join(args) for args in launched)
    assert parallel.read(tmp_path / "status.json") == {
        "phase": "partial_failed", "training_allowed": False,
    }


def test_prepared_but_unauthorized_contract_cannot_start_or_initialize(tmp_path):
    import t02_parallel_launch as launcher
    parallel.save(tmp_path / "source_files.json", {"files": {}})
    parallel.save(tmp_path / "contract.json", {"launch_authorized": False})
    with pytest.raises(PermissionError, match="not been authorized"):
        launcher.supervise(tmp_path)
    with pytest.raises(PermissionError, match="not been authorized"):
        launcher.slot(tmp_path, "npu0")
    assert not (tmp_path / "ledger.json").exists()
    assert not (tmp_path / "started.json").exists()


def test_repair_authorization_binds_the_exact_frozen_contract(tmp_path):
    import hashlib
    import t02_parallel_launch as launcher
    contract = {"launch_authorized": False, "cumulative_source_start_cap": 129}
    parallel.save(tmp_path / "contract.json", contract)
    digest = hashlib.sha256((tmp_path / "contract.json").read_bytes()).hexdigest()
    parallel.save(tmp_path / "launch_authorization.json", {
        "approved": True, "contract_sha256": digest})
    launcher.require_launch_authorization(tmp_path, contract)
    contract["cumulative_source_start_cap"] = 130
    parallel.save(tmp_path / "contract.json", contract)
    with pytest.raises(PermissionError, match="not been authorized"):
        launcher.require_launch_authorization(tmp_path, contract)
