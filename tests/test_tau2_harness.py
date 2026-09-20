"""Single-task tau2 integration at the official artifact seam."""
import json
from pathlib import Path
from unittest.mock import patch

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


def _write_summary(command, *, task_id="7", termination="agent_stop", reward=0.0):
    Path(command[5]).write_text(json.dumps({
        "task_ids": [task_id],
        "task_rows": [{"task_id": task_id, "semantic_score": reward,
                       "termination": termination}],
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

    def fake_worker(command, **_kwargs):
        request = json.loads(Path(command[4]).read_text(encoding="utf-8"))
        seen.append(request)
        _write_official(cell, request)
        _write_summary(command)
        return 0

    with patch.object(harness, "run_owned_worker", side_effect=fake_worker):
        harness.run_tau2_task(cell, "7", "http://127.0.0.1:1",
                              "http://127.0.0.1:2", tmp_path / "task")
    assert seen[0]["native"] is True
    assert seen[0]["agent_max_tokens"] == 1536
    assert seen[0]["user_model"] == "raw-model"


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
