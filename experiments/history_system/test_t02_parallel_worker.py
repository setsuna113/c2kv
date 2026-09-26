"""Focused orchestration contracts for same-engine T02 parallel workers."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import t02
import t02_bfcl
import t02_parallel
import t02_parallel_worker


def test_dynamic_recovery_queue_binds_explicit_manifest_subset(tmp_path):
    recovery_path = tmp_path / "recovery" / "recovery_contract.json"
    _write(recovery_path, {
        "schema": "t02-exact-recovery-contract-v1",
        "source_tasks": [{"task_id": "task-a"}, {"task_id": "task-c"}],
    })
    contract = {"recovery_contract_path": "recovery/recovery_contract.json"}
    t02_parallel_worker._validate_dynamic_task_universe(
        tmp_path.resolve(), contract, {"task-a", "task-c"},
        ["task-a", "task-b", "task-c"],
    )
    with pytest.raises(ValueError, match="exact recovery task subset"):
        t02_parallel_worker._validate_dynamic_task_universe(
            tmp_path.resolve(), contract, {"task-a", "task-b"},
            ["task-a", "task-b", "task-c"],
        )


def test_dynamic_nonrecovery_queue_still_requires_full_manifest(tmp_path):
    with pytest.raises(ValueError, match="cover the frozen task manifest"):
        t02_parallel_worker._validate_dynamic_task_universe(
            tmp_path.resolve(), {}, {"task-a"}, ["task-a", "task-b"]
        )


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _contract(tmp_path: Path):
    bfcl_root = tmp_path / "bfcl"
    bfcl_root.mkdir()
    design = tmp_path / "design.json"
    design.write_text("{}", encoding="utf-8")
    policy = {
        "name": "parallel-frozen-C0",
        "design_sha256": hashlib.sha256(design.read_bytes()).hexdigest(),
        "sampling": {"mode": "greedy", "temperature": 0, "seed": 0},
        "family_bindings": {"schema": "families"},
        "official_source_files": {"schema": "official"},
    }
    common = {
        "bfcl_root": str(bfcl_root),
        "bfcl_dependency_path": None,
        "task_manifest": str(tmp_path / "tasks.json"),
        "family_audit": str(tmp_path / "audit.json"),
        "d128_manifest": str(tmp_path / "D128.json"),
        "f128_manifest": str(tmp_path / "F128.json"),
        "design": str(design),
        "checkpoint": str(tmp_path / "checkpoint"),
        "seed": 0,
        "max_states_per_task": 2,
    }
    contract = {
        "schema": t02_parallel_worker.CONTRACT_SCHEMA,
        "common_args": common,
        "frozen_policy": policy,
        "workers": [
            {"worker_id": "npu4", "physical_device": 4, "port": 36540,
             "task_ids": ["multi_turn_base_3"]},
            {"worker_id": "npu6", "physical_device": 6, "port": 36560,
             "task_ids": ["multi_turn_long_context_13"]},
        ],
        "target_states": 2,
        "train_states": 1,
        "min_task_groups": 1,
        "max_states_per_group": 8,
        "remaining_branch_budget": 6,
    }
    _write(tmp_path / "contract.json", contract)
    return contract


class FakeBindings:
    def load_task(self, task_id):
        return {"id": task_id}, [["ok"]]


class FakeEnvironment:
    def __init__(self, task, ground_truth, *, bindings):
        self.task_id = task["id"]


class FakeAdapter:
    instances = []
    source_terminals = {}

    def __init__(self, *, frozen_policy, artifact_dir, family_bindings):
        self.policy = frozen_policy
        self.states = {}
        self.pruned_to = None
        self.closed = False
        self.tasks = 0
        type(self).instances.append(self)

    def discover_task(self, environment, actor, *, seed, max_states):
        self.tasks += 1
        state = {"state_id": f"state-{environment.task_id}", "task_id": environment.task_id}
        self.states[state["state_id"]] = state
        return [state]

    def cost_summary(self):
        return {"source_collection_tasks": self.tasks,
                "completed_branch_executions": 0}

    def source_task_terminal(self, task_id):
        return self.source_terminals.get(task_id)

    def prune(self, selected_state_ids):
        self.pruned_to = list(selected_state_ids)
        self.states = {key: value for key, value in self.states.items()
                       if key in selected_state_ids}

    def close(self):
        self.closed = True


def _patch_runtime(monkeypatch, contract, claims, completions):
    monkeypatch.setattr(
        t02_bfcl, "load_family_bindings",
        lambda manifest, audit: (
            ["multi_turn_base_3", "multi_turn_long_context_13"],
            {"multi_turn_base_3": "bfcl_pair_3",
             "multi_turn_long_context_13": "bfcl_pair_13"},
            {"schema": "families"},
        ),
    )
    monkeypatch.setattr(
        t02_bfcl, "verify_official_source_files",
        lambda root, manifest: {"schema": "official"},
    )
    monkeypatch.setattr(t02_bfcl, "OfficialBFCLBindings", FakeBindings)
    monkeypatch.setattr(t02_bfcl, "BFCLTaskEnvironment", FakeEnvironment)
    monkeypatch.setattr(t02_bfcl, "ExactBFCLBranchAdapter", FakeAdapter)

    actor = SimpleNamespace(runner=SimpleNamespace(
        controller=SimpleNamespace(backends=object(), base=None)))
    monkeypatch.setitem(sys.modules, "t02_runtime", SimpleNamespace(
        build_actor=lambda **kwargs: actor))

    queue = list(claims)
    monkeypatch.setattr(t02_parallel, "claim_task",
                        lambda root, worker_id: queue.pop(0))
    monkeypatch.setattr(
        t02_parallel, "complete_task",
        lambda root, worker_id, task_id, candidates, costs, source_terminal=None:
            completions.append((worker_id, task_id, candidates, costs, source_terminal)),
    )
    monkeypatch.setattr(t02, "check_plan", lambda plan: None)


def test_worker_executes_only_states_owned_by_its_live_adapter(monkeypatch, tmp_path):
    contract = _contract(tmp_path)
    own_state = "state-multi_turn_base_3"
    plan = {
        "states": [{"state_id": own_state}, {"state_id": "state-foreign"}],
        "frozen_subsequent_policy": contract["frozen_policy"],
        "max_complete_branch_executions": 6,
        "authorized_complete_branch_cap": 359,
    }
    _write(tmp_path / "global_plan.json", plan)
    completions = []
    _patch_runtime(
        monkeypatch, contract,
        claims=[{"task_id": "multi_turn_base_3", "task_index": 9}, None],
        completions=completions,
    )
    executed = {}

    def execute_plan(global_plan, adapter, *, state_ids, on_progress):
        executed.setdefault("calls", []).append(list(state_ids))
        executed["adapter_state_ids"] = sorted(adapter.states)
        snapshots = [{"state_id": state_id, "snapshot_id": f"snap-{state_id}"}
                     for state_id in state_ids]
        results = [{"state_id": state_id, "branch_id": branch}
                   for state_id in state_ids for branch in t02.BRANCH_IDS]
        partial = {
            "schema": t02.RESULT_SET_SCHEMA,
            "plan_sha256": t02._digest(global_plan),
            "adapter_capabilities": {"schema": "fake"},
            "executed_state_ids": list(state_ids),
            "partial_worker_result": True,
            "complete_branch_executions": len(results),
            "branch_cap": 6,
            "authorized_complete_branch_cap": 359,
            "snapshots": snapshots,
            "results": results,
        }
        for count in range(1, len(results) + 1):
            on_progress({**partial, "results": results[:count],
                         "complete_branch_executions": count})
        return partial

    monkeypatch.setattr(t02, "execute_plan", execute_plan)
    terminal = {
        "schema": "t02-bfcl-capacity-termination-v1",
        "reason": "capacity_infeasible",
        "terminal_status": "budget_terminated",
        "response_fabricated": False,
    }
    monkeypatch.setattr(FakeAdapter, "source_terminals", {
        "multi_turn_base_3": terminal,
    })
    FakeAdapter.instances.clear()

    assert t02_parallel_worker.run_worker(tmp_path, "npu4", "http://127.0.0.1:36540") == 0
    assert executed == {"calls": [[own_state], []], "adapter_state_ids": [own_state]}
    assert completions[0][0:2] == ("npu4", "multi_turn_base_3")
    assert completions[0][2][0]["state_id"] == own_state
    assert completions[0][4] == terminal
    assert _read(tmp_path / "workers/npu4/ready.json")["candidate_state_ids"] == [own_state]
    assert _read(tmp_path / "workers/npu4/ready.json")["source_task_terminals"] == {
        "multi_turn_base_3": terminal,
    }
    assert _read(tmp_path / "workers/npu4/partial_results.json")["executed_state_ids"] == [own_state]
    assert _read(tmp_path / "workers/npu4/results.json")["complete_branch_executions"] == 3
    assert _read(tmp_path / "pilot_passed.json")["pilot_state_id"] == own_state
    assert FakeAdapter.instances[-1].closed is True


def test_post_pilot_branch_failure_is_local_and_keeps_partial_results(monkeypatch, tmp_path):
    contract = _contract(tmp_path)
    first = "state-multi_turn_base_3-a"
    second = "state-multi_turn_base_3-b"
    plan = {
        "states": [{"state_id": first}, {"state_id": second}],
        "frozen_subsequent_policy": contract["frozen_policy"],
        "max_complete_branch_executions": 6,
        "authorized_complete_branch_cap": 359,
    }
    _write(tmp_path / "global_plan.json", plan)
    _patch_runtime(
        monkeypatch, contract,
        claims=[{"task_id": "multi_turn_base_3", "task_index": 9}, None],
        completions=[],
    )

    def discover_task(adapter, environment, actor, *, seed, max_states):
        adapter.tasks += 1
        states = [
            {"state_id": first, "task_id": environment.task_id},
            {"state_id": second, "task_id": environment.task_id},
        ]
        adapter.states.update({state["state_id"]: state for state in states})
        return states

    monkeypatch.setattr(FakeAdapter, "discover_task", discover_task)
    calls = []

    def execute_plan(global_plan, adapter, *, state_ids, on_progress):
        calls.append(list(state_ids))
        results = [
            {"state_id": state_id, "branch_id": branch}
            for state_id in state_ids for branch in t02.BRANCH_IDS
        ]
        if state_ids == [second]:
            results = results[:1]
        partial = {
            "schema": t02.RESULT_SET_SCHEMA,
            "plan_sha256": t02._digest(global_plan),
            "adapter_capabilities": {"schema": "fake"},
            "executed_state_ids": list(state_ids),
            "partial_worker_result": True,
            "complete_branch_executions": len(results),
            "branch_cap": 6,
            "authorized_complete_branch_cap": 359,
            "snapshots": [{"state_id": state_id} for state_id in state_ids],
            "results": results,
        }
        on_progress(partial)
        if state_ids == [second]:
            raise RuntimeError("synthetic unknown branch failure")
        return partial

    monkeypatch.setattr(t02, "execute_plan", execute_plan)
    monkeypatch.setattr(FakeAdapter, "source_terminals", {})
    FakeAdapter.instances.clear()

    with pytest.raises(RuntimeError, match="synthetic unknown branch failure"):
        t02_parallel_worker.run_worker(
            tmp_path, "npu4", "http://127.0.0.1:36540"
        )

    assert calls == [[first], [second]]
    failure = _read(tmp_path / "workers/npu4/status.json")
    assert failure["failure_scope"] == "branch_execution"
    assert failure["global_abort_required"] is False
    assert failure["error_type"] == "RuntimeError"
    assert failure["error"] == "synthetic unknown branch failure"
    assert not (tmp_path / "abort.json").exists()
    partial = _read(tmp_path / "workers/npu4/partial_results.json")
    assert partial["executed_state_ids"] == [first, second]
    assert partial["complete_branch_executions"] == 4
    assert not (tmp_path / "workers/npu4/results.json").exists()
    assert FakeAdapter.instances[-1].closed is True


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _enable_dynamic(contract, tmp_path):
    contract.update({
        "dynamic_source_queue": True,
        "max_source_tasks_per_worker": 26,
        "snapshot_cap_per_worker": 52,
        "snapshot_host_byte_cap_per_worker": 206158430208,
        "prior_source_starts": 156,
        "cumulative_source_start_cap": 158,
    })
    _write(tmp_path / "contract.json", contract)


def test_dynamic_worker_can_execute_task_from_another_static_shard(monkeypatch, tmp_path):
    contract = _contract(tmp_path)
    _enable_dynamic(contract, tmp_path)
    stolen_task = "multi_turn_long_context_13"
    state_id = f"state-{stolen_task}"
    plan = {
        "states": [{"state_id": state_id}],
        "frozen_subsequent_policy": contract["frozen_policy"],
        "max_complete_branch_executions": 3,
        "authorized_complete_branch_cap": 359,
    }
    _write(tmp_path / "global_plan.json", plan)
    completions = []
    _patch_runtime(
        monkeypatch, contract,
        claims=[{"task_id": stolen_task, "task_index": 1}, None],
        completions=completions,
    )

    def execute_plan(global_plan, adapter, *, state_ids, on_progress):
        snapshots = [{"state_id": value, "snapshot_id": f"snap-{value}"}
                     for value in state_ids]
        results = [{"state_id": value, "branch_id": branch}
                   for value in state_ids for branch in t02.BRANCH_IDS]
        partial = {
            "schema": t02.RESULT_SET_SCHEMA,
            "plan_sha256": t02._digest(global_plan),
            "adapter_capabilities": {"schema": "fake"},
            "executed_state_ids": list(state_ids),
            "partial_worker_result": True,
            "complete_branch_executions": len(results),
            "branch_cap": 3,
            "authorized_complete_branch_cap": 359,
            "snapshots": snapshots,
            "results": results,
        }
        if results:
            on_progress(partial)
        return partial

    monkeypatch.setattr(t02, "execute_plan", execute_plan)
    FakeAdapter.instances.clear()
    assert t02_parallel_worker.run_worker(
        tmp_path, "npu4", "http://127.0.0.1:36540"
    ) == 0
    ready = _read(tmp_path / "workers/npu4/ready.json")
    assert ready["claimed_task_ids"] == [stolen_task]
    assert ready["candidate_state_ids"] == [state_id]
    assert completions[0][0:2] == ("npu4", stolen_task)
    assert _read(tmp_path / "workers/npu4/results.json")["executed_state_ids"] == [state_id]


def test_dynamic_worker_with_no_lease_completes_without_plan_execution(monkeypatch, tmp_path):
    contract = _contract(tmp_path)
    _enable_dynamic(contract, tmp_path)
    _patch_runtime(monkeypatch, contract, claims=[None], completions=[])
    monkeypatch.setattr(
        t02, "execute_plan", lambda *args, **kwargs: pytest.fail("no-work worker executed plan")
    )
    assert t02_parallel_worker.run_worker(
        tmp_path, "npu6", "http://127.0.0.1:36560"
    ) == 0
    assert _read(tmp_path / "workers/npu6/ready.json")["claimed_task_ids"] == []
    assert _read(tmp_path / "workers/npu6/status.json")["phase"] == "completed_no_work"
    assert not (tmp_path / "workers/npu6/results.json").exists()


def test_worker_rejects_global_plan_with_different_frozen_policy(monkeypatch, tmp_path):
    contract = _contract(tmp_path)
    _write(tmp_path / "global_plan.json", {
        "states": [{"state_id": "state-multi_turn_base_3"}],
        "frozen_subsequent_policy": {"name": "different"},
        "max_complete_branch_executions": 3,
        "authorized_complete_branch_cap": 359,
    })
    _patch_runtime(
        monkeypatch, contract,
        claims=[{"task_id": "multi_turn_base_3", "task_index": 9}, None],
        completions=[],
    )
    monkeypatch.setattr(
        t02, "execute_plan",
        lambda *args, **kwargs: pytest.fail("mismatched policy reached branch execution"),
    )

    with pytest.raises(ValueError, match="frozen continuation policy"):
        t02_parallel_worker.run_worker(tmp_path, "npu4", "http://127.0.0.1:36540")
    failure = _read(tmp_path / "workers/npu4/status.json")
    assert failure["phase"] == "failed"
    assert failure["failure_scope"] == "global_contract"
    assert failure["global_abort_required"] is True
    assert _read(tmp_path / "abort.json")["phase"] == "worker"
    assert FakeAdapter.instances[-1].closed is True


def test_worker_rejects_overlapping_static_task_shards(monkeypatch, tmp_path):
    contract = _contract(tmp_path)
    contract["workers"][1]["task_ids"] = ["multi_turn_base_3"]
    _write(tmp_path / "contract.json", contract)
    _patch_runtime(monkeypatch, contract, claims=[], completions=[])
    with pytest.raises(ValueError, match="task shards overlap"):
        t02_parallel_worker.run_worker(tmp_path, "npu4", "http://127.0.0.1:36540")


def test_real_execute_plan_progress_merges_only_started_snapshots(tmp_path):
    from test_t02 import ExactFakeAdapter, _plan

    plan = _plan(tmp_path)
    assigned = [state["state_id"] for state in plan["states"][:2]]
    progress = []

    def capture(partial):
        progress.append(t02_parallel_worker._merge_partial_results(plan, [partial]))

    final = t02.execute_plan(
        plan, ExactFakeAdapter(), state_ids=assigned, on_progress=capture
    )
    merged = t02_parallel_worker._merge_partial_results(
        plan, [final], require_complete_states=True
    )

    assert len(progress) == 6
    assert progress[0]["executed_state_ids"] == assigned[:1]
    assert len(progress[0]["snapshots"]) == 1
    assert progress[3]["executed_state_ids"] == assigned
    assert len(progress[3]["snapshots"]) == 2
    assert merged["executed_state_ids"] == assigned
    assert merged["complete_branch_executions"] == 6
