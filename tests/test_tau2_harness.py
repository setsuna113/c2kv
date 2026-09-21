"""Single-task tau2 integration at the official artifact seam."""
import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from generality import tau2_harness as harness


def _cell(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    return {
        "python_tau2": "/envs/bench312/bin/python",
        "python_bench": "/envs/bench/bin/python",
        "benchmark_dir": str(tmp_path / "tau2"),
        "paper_root": str(paper),
        "model_name": "test-model",
        "tau2_task_set": "airline",
        "tau2_split": "base",
    }


def _write_official(cell, request, *, task_id="7", termination="agent_stop",
                    reward=0.0, evaluated=True):
    source = (Path(cell["benchmark_dir"]) / "data" / "simulations"
              / request["run_name"])
    source.mkdir(parents=True)
    simulation = {"task_id": task_id, "termination_reason": termination,
                  "messages": []}
    (source / "results.json").write_text(
        json.dumps({"simulations": [simulation]}), encoding="utf-8")
    if evaluated:
        scored = {**simulation, "reward_info": {"reward": reward}}
        (source / "updated_results.json").write_text(
            json.dumps({"simulations": [scored]}), encoding="utf-8")


def _write_summary(command, *, task_id="7", termination="agent_stop", reward=0.0,
                   failure_code=None, official_reward=None):
    summary = {
        "task_ids": [task_id],
        "task_rows": [{"task_id": task_id, "semantic_score": reward,
                       "termination": termination}],
    }
    if failure_code is not None:
        summary["task_failures"] = {task_id: failure_code}
        summary["task_rows"][0]["task_failure_kind"] = failure_code
        summary["task_rows"][0]["official_reward"] = official_reward
    Path(command[5]).write_text(json.dumps(summary), encoding="utf-8")


def _write_typed_final(batch: Path, code: str, *, pending=0, failed=0):
    server = batch / "server"
    server.mkdir(parents=True, exist_ok=True)
    (server / "ready.json").write_text(json.dumps({
        "schema": "a-event-native-server-v1", "status": "ready",
        "benchmark": "tau2", "allowed_task_ids": ["7"], "run_id": "test-run",
        "max_generation_calls": 2, "max_decisions": 2,
    }), encoding="utf-8")
    used_key, cap_key = (("decisions_reserved", "max_decisions")
                         if code == "decision_cap_reached" else
                         ("generation_calls_reserved", "max_generation_calls"))
    (server / "budget_rejections.jsonl").write_text(json.dumps({
        "schema": "a-event-native-budget-rejection-v1", "run_id": "test-run",
        "task_id": "7", "session_id": "tau2/7/attempt-0", "decision_key": "d96",
        "status_code": 429, "code": code, used_key: 2, cap_key: 2,
    }) + "\n", encoding="utf-8")
    (server / "final.json").write_text(json.dumps({
        "status": "stopped",
        "cost_summary": {"recorded": True},
        "journal_summary": {
            "schema": "a-runtime-attempt-journal-v1",
            "started": 2, "completed": 2 - pending - failed,
            "pending": pending, "failed": failed,
        },
        "api_health": {
            "allowed_task_ids": ["7"], "terminal_reason": code,
            "generation_calls_reserved": 2, "max_generation_calls": 2,
            "decisions_reserved": 2, "max_decisions": 2,
        },
    }), encoding="utf-8")


def test_one_task_uses_shared_adapter_with_split_endpoints_and_official_reward(tmp_path):
    cell = _cell(tmp_path)
    out = tmp_path / "tasks" / "7"
    calls = []

    def fake_worker(command, **kwargs):
        assert command[0] == cell["python_tau2"]
        assert "from benchmarks.adapters.tau2_adapter import run_tau2" in command[2]
        assert kwargs["cwd"] == cell["paper_root"]
        request = json.loads(Path(command[4]).read_text(encoding="utf-8"))
        calls.append(request)
        _write_official(cell, request)
        _write_summary(command)
        return 0

    with patch.object(harness, "run_owned_worker", side_effect=fake_worker):
        receipt = harness.run_tau2_task(
            cell, "7", "http://127.0.0.1:37001/v1",
            "http://127.0.0.1:35020", out)
        again = harness.run_tau2_task(
            cell, "7", "http://127.0.0.1:37001/v1",
            "http://127.0.0.1:35020", out)

    assert len(calls) == 1
    assert calls[0]["base_url"] == "http://127.0.0.1:37001/v1"
    assert calls[0]["user_base_url"] == "http://127.0.0.1:35020"
    assert calls[0]["task_ids"] == ["7"]
    assert calls[0]["task_set"] == "airline"
    assert calls[0]["task_split"] == "base"
    assert calls[0]["num_workers"] == calls[0]["num_trials"] == 1
    assert calls[0]["native"] is False
    assert calls[0]["user_model"] == "gen-c1000"
    assert calls[0]["agent_max_tokens"] == 2048
    assert calls[0]["python"] == cell["python_tau2"]
    assert "native_server_dir" not in calls[0]
    assert receipt["status"] == again["status"] == "completed"
    assert receipt["semantic_score"] == 0.0
    assert receipt["termination"] == "agent_stop"
    assert harness.completed_tau2_task(out, "7")
    assert not harness.completed_tau2_task(out, "8")
    assert (out / receipt["official_results"]).is_file()


def test_native_controller_uses_native_tau2_transport_and_frozen_cap(tmp_path):
    cell = _cell(tmp_path)
    cell.update(backend="c2kv", condition="tracer_history",
                caps={"max_completion_tokens": 1536},
                upstream_model_name="raw-model")
    seen = []
    server_dir = tmp_path / "batch" / "server"
    server_dir.mkdir(parents=True)

    def fake_worker(command, **_kwargs):
        request = json.loads(Path(command[4]).read_text(encoding="utf-8"))
        seen.append(request)
        _write_official(cell, request)
        _write_summary(command)
        return 0

    with patch.object(harness, "run_owned_worker", side_effect=fake_worker):
        harness.run_tau2_task(cell, "7", "http://127.0.0.1:1",
                              "http://127.0.0.1:2", tmp_path / "task",
                              native_server_dir=server_dir)
    assert seen[0]["native"] is True
    assert seen[0]["agent_max_tokens"] == 1536
    assert seen[0]["user_model"] == "raw-model"
    assert seen[0]["native_server_dir"] == str(server_dir.resolve())
    assert 'request["native_server_dir"] = Path(request["native_server_dir"])' in (
        harness._WORKER)


def test_native_server_dir_must_be_explicit_existing_directory(tmp_path):
    cell = _cell(tmp_path)
    with pytest.raises(ValueError, match="existing controller server directory"):
        harness.run_tau2_task(
            cell, "7", "http://127.0.0.1:1", "http://127.0.0.1:2",
            tmp_path / "task", native_server_dir=tmp_path / "missing-server")


def test_receipt_alone_does_not_complete_unscored_task(tmp_path):
    out = tmp_path / "tasks" / "7"
    out.mkdir(parents=True)
    (out / "done.json").write_text(json.dumps({"task_id": "7", "status": "completed"}),
                                   encoding="utf-8")
    assert not harness.completed_tau2_task(out, "7")


def test_invalid_official_output_preserves_attempt_and_retries(tmp_path):
    cell = _cell(tmp_path)
    out = tmp_path / "tasks" / "7"
    calls = []

    def fake_worker(command, **_kwargs):
        request = json.loads(Path(command[4]).read_text(encoding="utf-8"))
        calls.append(request)
        if len(calls) == 1:
            _write_official(cell, request, termination="infrastructure_error")
        else:
            _write_official(cell, request, reward=1.0)
        if len(calls) == 2:
            _write_summary(command, reward=1.0)
        return 0

    with patch.object(harness, "run_owned_worker", side_effect=fake_worker):
        first = harness.run_tau2_task(cell, "7", "http://127.0.0.1:1",
                                      "http://127.0.0.1:2", out)
        assert first["status"] == "infra_error"
        assert not harness.completed_tau2_task(out, "7")
        second = harness.run_tau2_task(cell, "7", "http://127.0.0.1:1",
                                       "http://127.0.0.1:2", out)

    assert second["status"] == "completed"
    assert second["semantic_score"] == 1.0
    assert calls[0]["run_name"] != calls[1]["run_name"]
    assert len(list((out / "attempts").glob("*/official/updated_results.json"))) == 2


def test_official_reward_and_terminal_are_both_required(tmp_path):
    cell = _cell(tmp_path)
    out = tmp_path / "tasks" / "7"
    attempts = [
        {"evaluated": False},
        {"reward": None},
        {"termination": ""},
        {"task_id": "8"},
    ]

    def fake_worker(command, **_kwargs):
        request = json.loads(Path(command[4]).read_text(encoding="utf-8"))
        _write_official(cell, request, **attempts.pop(0))
        _write_summary(command)
        return 0

    with patch.object(harness, "run_owned_worker", side_effect=fake_worker):
        for _ in range(4):
            receipt = harness.run_tau2_task(cell, "7", "http://127.0.0.1:1",
                                            "http://127.0.0.1:2", out)
            assert receipt["status"] == "infra_error"
            assert not harness.completed_tau2_task(out, "7")


def test_adapter_summary_must_match_official_score(tmp_path):
    cell = _cell(tmp_path)
    out = tmp_path / "tasks" / "7"

    def fake_worker(command, **_kwargs):
        request = json.loads(Path(command[4]).read_text(encoding="utf-8"))
        _write_official(cell, request, reward=1.0)
        _write_summary(command, reward=0.0)
        return 0

    with patch.object(harness, "run_owned_worker", side_effect=fake_worker):
        receipt = harness.run_tau2_task(cell, "7", "http://127.0.0.1:1",
                                        "http://127.0.0.1:2", out)
    assert receipt["status"] == "infra_error"
    assert not harness.completed_tau2_task(out, "7")


@pytest.mark.parametrize("code", ("decision_cap_reached", "generation_cap_reached"))
@pytest.mark.parametrize("official_reward", (None, 0.0, 1.0))
def test_typed_budget_failure_completes_once_and_keeps_zero_provenance(
        tmp_path, code, official_reward):
    cell = _cell(tmp_path)
    cell_dir = tmp_path / "cell"
    out = cell_dir / "batches" / "one" / "tau2_worker" / "7"
    calls = []

    def fake_worker(command, **_kwargs):
        request = json.loads(Path(command[4]).read_text(encoding="utf-8"))
        calls.append(request)
        _write_official(cell, request, termination="infrastructure_error",
                        reward=official_reward)
        _write_summary(command, termination="infrastructure_error", reward=0.0,
                       failure_code=code, official_reward=official_reward)
        return 0

    with patch.object(harness, "run_owned_worker", side_effect=fake_worker):
        result = harness.run_tau2_task(
            cell, "7", "http://127.0.0.1:1", "http://127.0.0.1:2", out)
        resumed = harness.run_tau2_task(
            cell, "7", "http://127.0.0.1:1", "http://127.0.0.1:2", out)

    assert len(calls) == 1
    assert result == resumed
    assert result["status"] == "completed"
    assert result["semantic_score"] == 0.0
    assert result["termination"] == "infrastructure_error"
    assert result["task_failure_kind"] == code
    assert result["score_source"] == "typed_harness_budget_failure"
    assert result["official_reward"] == official_reward
    assert harness.completed_tau2_task(out, "7")

    _write_typed_final(cell_dir / "batches" / "one", code)

    sys.modules.setdefault("current", types.SimpleNamespace())
    sys.modules.setdefault("evidence_sets", types.SimpleNamespace())
    sys.modules.setdefault("c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell

    summary = c2kv_cell.tau2_score_summary({
        "cell_dir": str(cell_dir), "cell_id": "test", "task_ids": ["7"]})
    assert summary["n_official_scored"] == 0
    assert summary["n_budget_failures"] == summary["n_completed"] == 1
    assert summary["budget_failure_task_ids"] == ["7"]
    assert summary["pending_task_ids"] == []
    assert summary["task_rows"] == [{
        "task_id": "7", "semantic_score": 0.0,
        "termination": "infrastructure_error",
        "score_source": "typed_harness_budget_failure",
        "task_failure_kind": code,
        "official_reward": official_reward,
    }]


def test_old_untyped_harness_cap_remains_retryable(tmp_path):
    cell = _cell(tmp_path)
    out = tmp_path / "tasks" / "7"

    def fake_worker(command, **_kwargs):
        request = json.loads(Path(command[4]).read_text(encoding="utf-8"))
        _write_official(cell, request, termination="infrastructure_error", reward=None)
        _write_summary(command, termination="infrastructure_error", reward=0.0,
                       failure_code="harnesscap0")
        return 0

    with patch.object(harness, "run_owned_worker", side_effect=fake_worker):
        result = harness.run_tau2_task(
            cell, "7", "http://127.0.0.1:1", "http://127.0.0.1:2", out)
    assert result["status"] == "infra_error"
    assert not harness.completed_tau2_task(out, "7")


@pytest.mark.parametrize("change", ("wrong_map", "wrong_row_kind", "wrong_reward"))
def test_typed_budget_failure_requires_matching_adapter_evidence(tmp_path, change):
    cell = _cell(tmp_path)
    out = tmp_path / "tasks" / "7"

    def fake_worker(command, **_kwargs):
        request = json.loads(Path(command[4]).read_text(encoding="utf-8"))
        _write_official(cell, request, termination="infrastructure_error", reward=1.0)
        _write_summary(command, termination="infrastructure_error", reward=0.0,
                       failure_code="generation_cap_reached", official_reward=1.0)
        summary_path = Path(command[5])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if change == "wrong_map":
            summary["task_failures"] = {"8": "generation_cap_reached"}
        elif change == "wrong_row_kind":
            summary["task_rows"][0]["task_failure_kind"] = "decision_cap_reached"
        else:
            summary["task_rows"][0]["official_reward"] = 0.0
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return 0

    with patch.object(harness, "run_owned_worker", side_effect=fake_worker):
        result = harness.run_tau2_task(
            cell, "7", "http://127.0.0.1:1", "http://127.0.0.1:2", out)
    assert result["status"] == "infra_error"
    assert not harness.completed_tau2_task(out, "7")


@pytest.mark.parametrize("problem", (
    "missing", "pending", "failed", "server_failed",
    "rejection_missing", "rejection_mismatch"))
def test_typed_budget_resume_requires_clean_final_journal(tmp_path, problem):
    cell = _cell(tmp_path)
    cell_dir = tmp_path / "cell"
    batch = cell_dir / "batches" / "one"
    out = batch / "tau2_worker" / "7"

    def fake_worker(command, **_kwargs):
        request = json.loads(Path(command[4]).read_text(encoding="utf-8"))
        _write_official(cell, request, termination="infrastructure_error", reward=None)
        _write_summary(command, termination="infrastructure_error", reward=0.0,
                       failure_code="generation_cap_reached")
        return 0

    with patch.object(harness, "run_owned_worker", side_effect=fake_worker):
        assert harness.run_tau2_task(
            cell, "7", "http://127.0.0.1:1", "http://127.0.0.1:2", out)["status"] == "completed"

    if problem != "missing":
        _write_typed_final(batch, "generation_cap_reached",
                           pending=int(problem == "pending"), failed=int(problem == "failed"))
        if problem == "server_failed":
            path = batch / "server" / "final.json"
            final = json.loads(path.read_text(encoding="utf-8"))
            final["status"] = "failed"
            path.write_text(json.dumps(final), encoding="utf-8")
        elif problem == "rejection_missing":
            (batch / "server" / "budget_rejections.jsonl").unlink()
        elif problem == "rejection_mismatch":
            path = batch / "server" / "budget_rejections.jsonl"
            rejection = json.loads(path.read_text(encoding="utf-8"))
            rejection["task_id"] = "8"
            path.write_text(json.dumps(rejection) + "\n", encoding="utf-8")

    sys.modules.setdefault("current", types.SimpleNamespace())
    sys.modules.setdefault("evidence_sets", types.SimpleNamespace())
    sys.modules.setdefault("c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell

    assert not c2kv_cell.tau2_task_completed(cell_dir, "7")
    summary = c2kv_cell.tau2_score_summary({
        "cell_dir": str(cell_dir), "cell_id": "test", "task_ids": ["7"]})
    assert summary["n_completed"] == 0
    assert summary["n_budget_failures"] == 0
    assert summary["pending_task_ids"] == ["7"]
