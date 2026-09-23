"""A typed native capacity rejection ends one AppWorld task, not the cell.

CPU only. The fixture is reduced from the box8 appworld b512 task fd1f8fa_2
that stopped the 6432ae7 cell with 'has a generation error, not an official
score' (see its provenance block).
"""
import copy
import json
from pathlib import Path

import pytest

from benchmarks.adapters import acon_adapter as acon
from benchmarks.paper import c1 as paper_c1, c1_appworld

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "benchmarks" / "fixtures" / "appworld_c1v2_capacity_b512_20260923.json"
TASK = "fd1f8fa_2"


def recorded():
    return copy.deepcopy(json.loads(FIXTURE.read_text(encoding="utf-8")))


def write_evidence(task_out, run_dir, data, task=TASK):
    server = Path(task_out) / "server"
    server.mkdir(parents=True, exist_ok=True)
    (server / "ready.json").write_text(json.dumps(data["ready"]), encoding="utf-8")
    (server / "steps.jsonl").write_text(
        json.dumps({"schema": "a-event-native-exact-step-v1", "status": "completed"}) + "\n"
        + json.dumps(data["last_step"]) + "\n", encoding="utf-8")
    (server / "final.json").write_text(json.dumps(data["final"]), encoding="utf-8")
    results = acon.appworld_task_dir(run_dir, task) / "results.json"
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(data["results"]), encoding="utf-8")


def test_recorded_rejection_is_the_methods_capacity_failure(tmp_path):
    task_out, run_dir = tmp_path / "task_shards" / TASK, tmp_path / "run"
    write_evidence(task_out, run_dir, recorded())
    message = c1_appworld.capacity_rejection(task_out, run_dir, TASK)
    assert message is not None and "CapacityInfeasible" in message


@pytest.mark.parametrize("change", [
    "other_generation_error", "not_generation_error", "other_task_step", "untyped_step",
    "runner_failure_step", "last_step_completed", "ready_other_task", "no_results",
])
def test_other_generation_errors_are_not_capacity_failures(tmp_path, change):
    data = recorded()
    if change == "other_generation_error":
        data["results"]["error"] = "Model generation failed: Error code: 500 - {'error': {'code': 'runner_failed'}}"
    elif change == "not_generation_error":
        data["results"]["termination_reason"] = "error"
    elif change == "other_task_step":
        data["last_step"]["session_id"] = "acon_appworld/fd1f8fa_3/attempt-0"
    elif change == "untyped_step":
        del data["last_step"]["failure_code"]
    elif change == "runner_failure_step":
        data["last_step"]["failure_kind"] = "runner_failed"
    elif change == "last_step_completed":
        data["last_step"]["status"] = "completed"
    elif change == "ready_other_task":
        data["ready"]["allowed_task_ids"] = ["fd1f8fa_3"]
    task_out, run_dir = tmp_path / "task_shards" / TASK, tmp_path / "run"
    write_evidence(task_out, run_dir, data)
    if change == "no_results":
        (acon.appworld_task_dir(run_dir, TASK) / "results.json").unlink()
    assert c1_appworld.capacity_rejection(task_out, run_dir, TASK) is None


class Proceeded(Exception):
    pass


@pytest.mark.parametrize("capacity", [True, False])
def test_harness_stops_before_scoring_only_for_the_capacity_rejection(tmp_path, monkeypatch, capacity):
    data = recorded()
    if not capacity:
        data["results"]["error"] = "Model generation failed: Error code: 500 - runner_failed"
    task_out, run_dir = tmp_path / "task_shards" / TASK, tmp_path / "run"
    write_evidence(task_out, run_dir, data)
    commands = []
    monkeypatch.setattr(c1_appworld.acon, "validate_appworld_runner_patches", lambda _root: None)
    monkeypatch.setattr(c1_appworld, "_prepare_appworld_run", lambda *args: tmp_path / "run_root")
    monkeypatch.setattr(c1_appworld.acon, "appworld_run_dir", lambda *args: run_dir)
    monkeypatch.setattr(c1_appworld.acon, "appworld_runner_env", lambda *args: {})
    monkeypatch.setattr(c1_appworld, "run_owned", lambda command, **_kwargs: commands.append(command))

    def telemetry(*_args):
        raise Proceeded()   # the unchanged path goes on to validation and official scoring

    monkeypatch.setattr(c1_appworld.acon, "validate_appworld_telemetry", telemetry)
    config = {"acon_dir": str(tmp_path / "acon"), "appworld_python": "/venv/python"}
    with pytest.raises(RuntimeError if capacity else Proceeded):
        c1_appworld._run_official_harness(config, TASK, task_out, "http://127.0.0.1:39123/v1", "m")
    assert len(commands) == 1   # the runner ran; appworld evaluate did not


def test_closed_loop_scores_the_rejected_task_zero_and_runs_the_next(tmp_path, monkeypatch):
    cell = tmp_path / "cell"
    native = cell / "native"
    native.mkdir(parents=True)
    tasks = [TASK, "fd1f8fa_3"]
    completed_metrics = {"task_id": tasks[1], "official_score": 1.0,
                         "normal_termination": True, "protocol_legal": None}
    completed = {"task_id": tasks[1], "status": "completed", "unified_metrics": completed_metrics,
                 "official_summary": {"task_id": tasks[1],
                                      "official_task_outcome": {"success": True, "difficulty": 1}}}
    calls = []

    def run_task(_config, task_id, directory, *_args):
        calls.append(task_id)
        if task_id == tasks[1]:
            return completed, completed_metrics
        task_out = Path(directory) / "task_shards" / task_id
        write_evidence(task_out, tmp_path / "run", recorded())
        raise RuntimeError(f"AppWorld task {task_id} ended by the native capacity rejection")

    monkeypatch.setattr(paper_c1, "load_delivery", lambda: object())
    monkeypatch.setattr(paper_c1, "selected_tasks", lambda *_args, **_kwargs: tasks)
    monkeypatch.setattr(paper_c1, "prepare_native",
                        lambda *_args, **_kwargs: (native, object(), native / "controller.json"))
    monkeypatch.setattr(c1_appworld, "run_task", run_task)
    assert paper_c1.run_closed_loop({}, "appworld", cell) == native
    assert calls == tasks
    receipt = json.loads((native / "task_shards" / TASK / "paper_task_result.json").read_text())
    assert receipt["status"] == "method_failure"
    assert receipt["failure"]["kind"] == "capacity_infeasible"
    assert receipt["unified_metrics"]["official_score"] == 0.0
    summary = json.loads((cell / f"summary_{paper_c1.ARM}.json").read_text())
    assert summary["n"] == 2 and summary["semantic_score"] == 0.5
    assert summary["method_failure_task_ids"] == [TASK]
