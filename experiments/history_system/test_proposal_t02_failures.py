"""Failure-path contracts for H0 proposal-matched T02."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runtime" / "python"))
sys.path.insert(0, str(HERE / "runtime"))
sys.path.insert(0, str(HERE))

import proposal_t02 as proposal
import proposal_t02_bfcl
import t02
import t02_bfcl
from benchmarks.memory_runtime.always_compress import CapacityInfeasible
from test_proposal_t02 import FakeAdapter, _manifests, _policy, _state


def _collect_kwargs(tmp_path: Path) -> dict:
    return {
        "task_ids": ["multi_turn_base_0"],
        "actor_factory": lambda task_id, task_index: object(),
        "bindings": object(),
        "forbidden_manifest_paths": _manifests(tmp_path),
        "frozen_policy": _policy(),
        "artifact_dir": tmp_path / "outcomes",
        "seed": 0,
        "target_states": 1,
        "train_states": 1,
        "candidate_snapshot_cap": 1,
        "max_states_per_group": 8,
        "max_states_per_task": 2,
        "max_source_task_starts": 1,
        "family_bindings": {"multi_turn_base_0": "bfcl_pair_0"},
        "split_bindings": {},
    }


def test_live_collection_rejects_more_than_60_candidate_snapshots(tmp_path):
    kwargs = _collect_kwargs(tmp_path)
    kwargs["candidate_snapshot_cap"] = proposal.MAX_NEW_STATES + 1
    with pytest.raises(ValueError, match=r"\[1, 60\]"):
        proposal_t02_bfcl.collect_live_proposal_plan(**kwargs)


class _SourceFailureAdapter:
    def __init__(self) -> None:
        self.source_collection_tasks = 0
        self.source_collection_decisions = 0
        self.source_collection_generation_calls = 0
        self.closed = False

    def discover_task(self, environment, actor, *, seed, max_states):
        del environment, actor, seed, max_states
        self.source_collection_tasks += 1
        self.source_collection_decisions += 1
        self.source_collection_generation_calls += 2
        if self.source_collection_tasks == 1:
            return [_state(0)]
        raise RuntimeError("source acquisition failed")

    def source_task_terminal(self, task_id):
        if task_id != "multi_turn_base_0":
            return None
        return {"terminal_status": "budget_terminated", "response_fabricated": False}

    def cost_summary(self):
        return {
            "source_collection_tasks": self.source_collection_tasks,
            "source_collection_decisions": self.source_collection_decisions,
            "source_collection_generation_calls": self.source_collection_generation_calls,
            "branch_generation_calls": 0,
            "total_generation_calls": self.source_collection_generation_calls,
            "completed_branch_executions": 0,
        }

    def close(self):
        self.closed = True


class _Bindings:
    def load_task(self, task_id):
        return {"id": task_id}, []


def test_source_failure_checkpoint_preserves_candidates_terminals_and_cost(
    tmp_path, monkeypatch
):
    adapter = _SourceFailureAdapter()
    checkpoints = []
    task_ids = ["multi_turn_base_0", "multi_turn_long_context_0"]
    monkeypatch.setattr(
        proposal_t02_bfcl.t02_bfcl,
        "BFCLTaskEnvironment",
        lambda task, ground_truth, bindings: object(),
    )
    kwargs = _collect_kwargs(tmp_path)
    kwargs.update(
        task_ids=task_ids,
        bindings=_Bindings(),
        target_states=2,
        train_states=1,
        candidate_snapshot_cap=2,
        max_source_task_starts=2,
        family_bindings={task_id: "bfcl_pair_0" for task_id in task_ids},
        adapter=adapter,
        on_checkpoint=lambda value: checkpoints.append(copy.deepcopy(value)),
    )

    with pytest.raises(RuntimeError, match="source acquisition failed"):
        proposal_t02_bfcl.collect_live_proposal_plan(**kwargs)

    checkpoint = checkpoints[-1]
    assert checkpoint["schema"] == "proposal-t02-source-checkpoint-h0-v1"
    assert checkpoint["status"] == "failed"
    assert checkpoint["candidate_state_count"] == 1
    assert [row["state_id"] for row in checkpoint["candidates"]] == ["state-0"]
    assert checkpoint["source_task_terminals"] == {
        "multi_turn_base_0": {
            "terminal_status": "budget_terminated",
            "response_fabricated": False,
        }
    }
    assert checkpoint["source_task_budget"] == {"authorized": 2, "actual": 2}
    assert checkpoint["generation_calls"]["total_generation_calls"] == 4
    assert checkpoint["last_task_id"] == "multi_turn_long_context_0"
    assert adapter.closed is True


class _BranchFailureAdapter(FakeAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.branch_calls = 0

    def run_branch(self, state, branch, *, frozen_policy, restore_receipt):
        self.branch_calls += 1
        if self.branch_calls == 2:
            raise RuntimeError("continuation failed")
        return super().run_branch(
            state, branch,
            frozen_policy=frozen_policy,
            restore_receipt=restore_receipt,
        )


def test_continuation_failure_checkpoint_retains_completed_results_and_is_not_trainable(
    tmp_path,
):
    plan = proposal.build_plan(
        [_state(1, source_mode="distinct")],
        forbidden_manifest_paths=_manifests(tmp_path),
        frozen_subsequent_policy=_policy(),
        target_states=1,
        train_states=1,
    )
    checkpoints = []
    with pytest.raises(RuntimeError, match="continuation failed"):
        proposal.execute_plan(
            plan,
            _BranchFailureAdapter(),
            on_checkpoint=lambda value: checkpoints.append(copy.deepcopy(value)),
        )

    checkpoint = checkpoints[-1]
    assert checkpoint["schema"] == proposal.RESULT_SET_SCHEMA
    assert checkpoint["status"] == "in_progress"
    assert checkpoint["plan_sha256"] == proposal._digest(plan)
    assert checkpoint["expected_continuation_executions"] == 3
    assert checkpoint["complete_continuation_executions"] == 1
    assert len(checkpoint["snapshots"]) == 1
    assert [row["branch_id"] for row in checkpoint["results"]] == ["A0"]
    assert checkpoint["executed_state_ids"] == []
    with pytest.raises(proposal.ProposalT02Error, match="partial or contain extra"):
        proposal.label_results(plan, checkpoint)


class _CapacityActor:
    def __init__(self) -> None:
        self.total_generation_calls = 0

    def submit_held(self, candidate_ids):
        del candidate_ids
        self.total_generation_calls += 1
        raise CapacityInfeasible("intervention does not fit")


class _CapacityEnvironment:
    turn_index = 0
    finished = False
    all_model_response = []

    def __init__(self) -> None:
        self.commit_called = False
        self.capacity_receipt = None

    def commit_response(self, response):
        del response
        self.commit_called = True

    def terminate_capacity_infeasible(self, error, *, stage):
        self.finished = True
        self.capacity_receipt = {
            "reason": "capacity_infeasible",
            "terminal_status": "budget_terminated",
            "stage": stage,
            "error": str(error),
            "response_fabricated": False,
        }
        return copy.deepcopy(self.capacity_receipt)

    def official_outcomes(self, intervention_turn):
        assert intervention_turn == 0
        return {"turn_success": False, "task_success": False}


def test_initial_branch_capacity_failure_becomes_a_complete_terminal_result(tmp_path):
    state = _state(1)
    state_id = state["state_id"]
    policy = _policy()
    environment = _CapacityEnvironment()
    actor = _CapacityActor()
    adapter = t02_bfcl.ExactBFCLBranchAdapter(
        frozen_policy=policy,
        artifact_dir=tmp_path / "outcomes",
    )
    adapter._active = {
        "state": state,
        "environment": environment,
        "actor": actor,
        "public": {"snapshot_id": "snapshot-1"},
    }
    branch = {
        "branch_id": "A1lex",
        "candidate_ids": ["u1-0"],
        "submit_original_draft": False,
        "regenerate": True,
    }

    result = adapter.run_branch(
        state,
        branch,
        frozen_policy=policy,
        restore_receipt={"schema": t02.RESTORE_SCHEMA},
    )

    assert result["execution_status"] == "complete"
    assert result["official_outcome"]["capacity_termination"]["stage"] == "branch_intervention"
    assert result["official_outcome"]["capacity_termination"]["response_fabricated"] is False
    assert adapter.cost_summary()["branch_generation_calls"] == 1
    assert environment.commit_called is False
    artifact = json.loads(Path(result["official_outcome"]["artifact"]).read_text(encoding="utf-8"))
    assert artifact["capacity_termination"] == environment.capacity_receipt
    assert artifact["state_id"] == state_id
