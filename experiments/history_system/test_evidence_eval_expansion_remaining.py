from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evidence_eval_expansion_remaining as remaining


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _source(root: Path, lane_name: str, *, remaining_count: int = 3,
            device: int = 0, lane_tasks: bool = True) -> Path:
    root.mkdir(parents=True)
    lane_root = root / "lanes" / lane_name
    runtime = lane_root / "runtime"
    (root / "history_system").mkdir()
    shutil.copyfile(HERE / "evidence_eval.py", root / "evidence_eval.py")
    shutil.copyfile(HERE / "runner.py", root / "history_system/runner.py")
    (root / "sglang").mkdir()
    (root / "sglang/version.py").write_text("VERSION='fixture'\n", encoding="utf-8")
    _write(root / "sglang_files.json", {
        "schema": "fixture-sglang-v1", "file_count": 1,
        "files": {"sglang/version.py": remaining._sha(root / "sglang/version.py")}})
    _write(runtime / "configs/controller.json", {
        "gp_experiments": {"G": "current", "R": 1, "set_selector": "candidate_rule"}})
    _write(runtime / "configs/eval_policy.json", {"frozen": True})
    task_ids = [f"multi_turn_base_{lane_name}_{index}"
                for index in range(2 + remaining_count)]
    tasks = {"schema": "a-history-system-task-manifest-v1",
             "manifest_id": lane_name, "task_ids": task_ids}
    _write(root / "tasks.json", tasks)
    if lane_tasks:
        _write(lane_root / "tasks.json", tasks)
    design = {
        "schema": "a-history-system-candidate-design-v1", "status": "frozen",
        "candidate_id": f"algo_{lane_name}", "run_id_template": lane_name,
        "launch_authorized": True, "automatic_reruns": 0,
        "task_ids": task_ids, "task_manifest_sha256": remaining._sha(root / "tasks.json"),
        "limits": {"tasks": len(task_ids), "history_tokens": 768},
        "sampling": {"mode": "greedy", "temperature": 0, "seed": 0,
                     "max_completion_tokens": 4096},
        "checkpoint_selection": {"path": "/checkpoint/frozen", "step": 1000},
        "runtime": {"sglang_backend_url": "http://127.0.0.1:19000",
                    "device": "npu", "dtype": "bfloat16"},
        "resolved_configs": {"controller": remaining._read(
            runtime / "configs/controller.json")},
        "search_contract": {"fixed_denominator": lane_name, "history": "H0",
                            "recovery_attempts": 1},
        "source_files": remaining.local_base._tree_hashes(runtime),
    }
    _write(lane_root / "design.json", design)
    _write(lane_root / "lane.json", {
        "name": lane_name, "physical_device": device,
        "engine_port": 19000 + device * 10,
        "task_port_base": 20000 + device * 100,
        "task_ports": list(range(20000 + device * 100,
                                 20000 + device * 100 + len(task_ids))),
        "model_smokes": ["embedding"]})
    _write(root / "provenance.json", {
        "controller": lane_name.split("_", 1)[0],
        "required_env": {"C2KV_STRICT_NONFINITE_SAMPLING": "1"}})
    _write(root / "launch_contract.json", {
        "launch_authorized": True,
        "required_environment": {"C2KV_STRICT_NONFINITE_SAMPLING": "1"},
        "automatic_retries": 0, "automatic_reruns": 0})
    _write(root / "static_files.json", remaining.local_base._static_manifest(root))
    outcomes = [
        {"task_id": task_ids[0], "outcome": "official_completed",
         "runtime_completed": True, "worker_returncode": 0, "server_returncode": 0,
         "official_summary": "completed.json"},
        {"task_id": task_ids[1], "outcome": "runtime_failure_in_denominator",
         "runtime_completed": False, "worker_returncode": 0, "server_returncode": 1,
         "official_summary": "failed.json"},
        *[{"task_id": task_id, "outcome": "not_started",
           "in_fixed_denominator": True, "official_summary": None}
          for task_id in task_ids[2:]],
    ]
    for task_id in task_ids[:2]:
        (lane_root / "results/task_shards" / task_id).mkdir(parents=True)
    _write(lane_root / "results/stage_manifest.json", {
        "schema": "a-history-system-run-v1", "status": "stopped_on_actor_runtime_failure",
        "state": "failed", "task_ids": task_ids, "task_cells_started": 2,
        "completed_task_cells": 2, "task_outcomes": outcomes,
        "automatic_retries": 0, "automatic_reruns": 0})
    _write(lane_root / "run/status.json", {
        "schema": "evidence-sets-eval-lane-status-v1", "lane": lane_name,
        "state": "failed_no_rerun", "completed_task_cells": 2,
        "runner": {"returncode": 6}, "automatic_retries": 0,
        "automatic_reruns": 0})
    return root


def _predecessor(root: Path, lane_name: str, state: str = "running_fixed_d20",
                 device: int = 0) -> Path:
    lane_root = root / "lanes" / lane_name
    _write(lane_root / "design.json", {"candidate_id": lane_name})
    _write(lane_root / "lane.json", {"name": lane_name,
                                     "physical_device": device})
    _write(root / "static_files.json", {"schema": "fixture", "file_count": 2,
                                        "files": {}})
    status = lane_root / "run/status.json"
    _write(status, {"schema": "evidence-sets-eval-lane-status-v1",
                    "lane": lane_name, "state": state,
                    "automatic_retries": 0, "automatic_reruns": 0})
    return status


def _authorization(package: Path) -> dict:
    value = remaining.authorization_requirements(package)
    value.update(status="authorized", launch_authorized=True, authorized_by="root")
    return value


def test_prepare_clones_only_original_tail_and_binds_runner_operation(
        tmp_path: Path) -> None:
    source0 = _source(tmp_path / "source0", "C0_part0")
    source1 = _source(tmp_path / "source1", "C5_part0", device=1,
                      lane_tasks=False)
    pred0 = _predecessor(tmp_path / "h1_0", "h1_C0_part0")
    pred1 = _predecessor(tmp_path / "h1_1", "h1_C5_part0", device=1)
    original_runner = remaining._sha(source0 / "history_system/runner.py")
    package = tmp_path / "remaining"
    result = remaining.prepare(package, [source0, source1], [pred0, pred1])
    assert result["verification"]["total_task_execution_budget"] == 6
    contract = remaining._read(package / "continuation_contract.json")
    assert [row["shard_id"] for row in contract["shards"]] == [
        "C0_part0.remaining", "C5_part0.remaining"]
    assert [row["device"] for row in contract["shards"]] == [0, 1]
    assert not ({5, 7} & {row["device"] for row in contract["shards"]})
    assert contract["runner_patch"]["source_sha256"] == original_runner
    assert contract["runner_patch"]["operation"] == remaining.RUNNER_CONTINUE_MARKER
    assert remaining.RUNNER_CONTINUE_MARKER in (
        package / "history_system/runner.py").read_text(encoding="utf-8")
    assert remaining.RUNNER_CONTINUE_MARKER not in (
        source0 / "history_system/runner.py").read_text(encoding="utf-8")
    for row in contract["shards"]:
        origin = remaining._origin(Path(row["source_shard"]))
        lane_root = package / "lanes" / row["shard_id"]
        assert remaining._read(lane_root / "tasks.json")["task_ids"] == origin[
            "remaining_ids"]
        assert remaining._lane_payload_hashes(lane_root) == remaining._lane_payload_hashes(
            origin["lane_root"])
        design = remaining._read(lane_root / "design.json")
        assert remaining._without_operational(design) == remaining._without_operational(
            origin["design"])
        assert row["predecessor"]["status"] in {str(pred0.resolve()), str(pred1.resolve())}
        binding = remaining._read(lane_root / "remainder_source_binding.json")
        assert binding["source_lane_tasks_sha256"] == origin["hashes"].get(
            "lane_tasks")
    completed = subprocess.run(
        [sys.executable, "-I", str(package / "evidence_eval_expansion_remaining.py"),
         "verify-package", "--package", str(package)],
        cwd=tmp_path, text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout)["status"] == "passed"


def test_rejects_nonterminal_duplicate_or_nonpristine_pending(tmp_path: Path) -> None:
    source = _source(tmp_path / "source", "C0_part0")
    status_path = source / "lanes/C0_part0/run/status.json"
    status = remaining._read(status_path)
    status["state"] = "running"
    _write(status_path, status)
    with pytest.raises(ValueError, match="terminal failed tail"):
        remaining.prepare(tmp_path / "bad_state", [source])

    source = _source(tmp_path / "native/source", "C0_part0")
    task_id = remaining._origin(source)["remaining_ids"][0]
    (source / "lanes/C0_part0/results/task_shards" / task_id).mkdir(parents=True)
    with pytest.raises(ValueError, match="not pristine never-started"):
        remaining.prepare(tmp_path / "bad_native", [source])

    source = _source(tmp_path / "duplicate/source", "C0_part0")
    with pytest.raises(ValueError, match="repeated"):
        remaining.prepare(tmp_path / "bad_duplicate", [source, source])


def test_predecessor_waits_for_exact_terminal_receipt(tmp_path: Path) -> None:
    status_path = _predecessor(tmp_path / "h1", "h1_C0_part0")
    binding = remaining._predecessor_binding(status_path)
    assert remaining._predecessor_terminal(binding) is False
    status = remaining._read(status_path)
    status["state"] = "starting_actor_engine"
    _write(status_path, status)
    assert remaining._predecessor_terminal(binding) is False
    status["state"] = "starting_unknown_future_state"
    _write(status_path, status)
    with pytest.raises(RuntimeError, match="unknown state"):
        remaining._predecessor_terminal(binding)
    status["state"] = "completed"
    _write(status_path, status)
    assert remaining._predecessor_terminal(binding) is True
    status["automatic_reruns"] = 1
    _write(status_path, status)
    with pytest.raises(RuntimeError, match="malformed"):
        remaining._predecessor_terminal(binding)


def test_run_requires_exact_authorization_and_pristine_lane(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _source(tmp_path / "source", "C0_part0")
    package = tmp_path / "remaining"
    remaining.prepare(package, [source])
    bad = tmp_path / "bad.json"
    receipt = _authorization(package)
    receipt["total_task_execution_budget"] += 1
    _write(bad, receipt)
    with pytest.raises(ValueError, match="exact remainder"):
        remaining.verify_authorization(package, bad)
    good = tmp_path / "good.json"
    _write(good, _authorization(package))
    calls = []
    lane = remaining._read(package / "lanes/C0_part0.remaining/lane.json")
    fake = SimpleNamespace(LANES={}, DEFAULT_REMOTE_ROOT=None,
                           _assert_lane_free=lambda value: calls.append(("free", value["name"])),
                           run_lane=lambda _package, name: calls.append(("run", name)) or 0)
    monkeypatch.setattr(remaining, "_base_lane", lambda *_: (fake, copy.deepcopy(lane)))
    monkeypatch.setattr(remaining.trained, "ascend_environment",
                        lambda: calls.append(("ascend", None)))
    monkeypatch.setattr(remaining.trained, "enable_strict_sampling",
                        lambda: calls.append(("strict", None)))
    assert remaining.run_shard(package, "C0_part0.remaining", good) == 0
    assert calls == [("free", "C0_part0.remaining"), ("ascend", None),
                     ("strict", None), ("run", "C0_part0.remaining")]
    (package / "lanes/C0_part0.remaining/run").mkdir()
    with pytest.raises(FileExistsError, match="rerun"):
        remaining.run_shard(package, "C0_part0.remaining", good)


def test_overlay_only_maps_original_pending_and_exposes_quality_fields(
        tmp_path: Path) -> None:
    source = _source(tmp_path / "source", "C0_part0")
    package = tmp_path / "remaining"
    remaining.prepare(package, [source])
    lane_root = package / "lanes/C0_part0.remaining"
    tasks = remaining._read(lane_root / "tasks.json")["task_ids"]
    outcomes = [
        {"task_id": tasks[0], "outcome": "official_completed",
         "runtime_completed": True, "worker_returncode": 0, "server_returncode": 0},
        {"task_id": tasks[1], "outcome": "runtime_failure_in_denominator",
         "runtime_completed": False, "worker_returncode": 0, "server_returncode": 1},
        {"task_id": tasks[2], "outcome": "not_started", "official_summary": None},
    ]
    _write(lane_root / "results/stage_manifest.json", {
        "task_ids": tasks, "task_outcomes": outcomes})
    _write(lane_root / "run/status.json", {"state": "failed_no_rerun"})
    official = {"scored": True, "n_total": 1, "n_scored": 1,
                "correct_count": 1, "semantic_score": 1.0,
                "total_gold_checker_seconds": 0, "total_handler_http_calls": 0}
    _write(lane_root / f"results/task_shards/{tasks[0]}/bfcl/official_summary.json",
           official)
    _write(lane_root / f"results/task_shards/{tasks[0]}/server/final.json",
           {"wall_seconds": 4.5, "cost": {"generation_attempts": 2,
                                           "prompt_tokens": 10,
                                           "completion_tokens": 2,
                                           "total_tokens": 12}})
    value = remaining.overlay(package)
    assert value["schema"] == remaining.OVERLAY_SCHEMA
    assert [row["status"] for row in value["rows"]] == [
        "completed", "runtime_failed", "pending"]
    completed = value["rows"][0]
    assert completed["source_outcome"]["outcome"] == "not_started"
    assert completed["official"] == official
    assert completed["measured_cost"]["total_tokens"] == 12
    assert completed["stage_outcome"]["runtime_completed"] is True
    assert completed["evidence"]
    assert all(row["source_outcome"] == {
        "task_id": row["task_id"], "outcome": "not_started",
        "in_fixed_denominator": True, "official_summary": None}
        for row in value["rows"])


def test_dispatch_continues_later_wave_after_lane_failure(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sources = [_source(tmp_path / f"source{index}", f"C{index}_part0",
                       remaining_count=1,
                       device=remaining.DEVICES[index % len(remaining.DEVICES)])
               for index in range(7)]
    package = tmp_path / "remaining"
    remaining.prepare(package, sources)
    authorization = tmp_path / "authorization.json"
    _write(authorization, _authorization(package))
    commands = []

    class Process:
        def __init__(self, command, **_kwargs):
            self.command = command
            self.pid = 1000 + len(commands)
            commands.append(command)

        def poll(self):
            return 2 if "C0_part0.remaining" in self.command else 0

    fake_base = SimpleNamespace(_assert_lane_free=lambda _lane: None)
    monkeypatch.setattr(remaining, "_base_lane", lambda _package, row: (
        fake_base, {"name": row["shard_id"]}))
    monkeypatch.setattr(remaining, "write_overlay", lambda *_: {})
    assert remaining.dispatch(package, authorization, poll_seconds=0,
                              popen=Process) == 2
    launched = [command[command.index("--shard") + 1] for command in commands]
    assert "C0_part0.remaining" in launched
    assert "C6_part0.remaining" in launched
    assert launched.index("C6_part0.remaining") > launched.index("C0_part0.remaining")
    state = remaining._read(package / "dispatch.json")
    assert state["status"] == "partial_failed_no_retry"
    assert next(row for row in state["shards"]
                if row["shard_id"] == "C6_part0.remaining")["status"] == "completed"
