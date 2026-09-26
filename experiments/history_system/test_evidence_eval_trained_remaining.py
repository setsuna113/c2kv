from __future__ import annotations

import json
from pathlib import Path

import pytest

import evidence_eval_trained_remaining as remaining
from test_evidence_eval_trained_parallel import _fake_canonical


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    canonical = _fake_canonical(tmp_path / "canonical", remaining.LANE)
    first = [f"multi_turn_base_{index}" for index in range(10)]
    shard1_tasks = [
        "multi_turn_base_100", "multi_turn_long_context_100",
        "multi_turn_base_120", "multi_turn_long_context_120", *remaining.TASKS]
    tasks = {"schema": "a-history-system-task-manifest-v1",
             "task_ids": first + shard1_tasks}
    _write(canonical / "tasks.d20.json", tasks)
    _write(canonical / f"lanes/{remaining.LANE}/tasks.json", tasks)
    design_path = canonical / f"lanes/{remaining.LANE}/design.json"
    design = remaining._read(design_path)
    design["task_ids"] = tasks["task_ids"]
    design["task_manifest_sha256"] = remaining.base._sha(canonical / "tasks.d20.json")
    _write(design_path, design)
    _write(canonical / "static_files.json", remaining.base._static_manifest(canonical))
    monkeypatch.setattr(remaining.base, "_preview_lane",
                        lambda _package, lane: {"task_count": 1,
                                                "engine_port": lane["engine_port"]})
    failed = tmp_path / "shard1"
    remaining.parallel._clone_shard(canonical, failed, remaining.LANE, 1)
    lane_root = failed / f"lanes/{remaining.LANE}"
    outcomes = []
    for index, task_id in enumerate(shard1_tasks):
        if index < 3:
            outcome = {"task_id": task_id, "outcome": "official_completed",
                       "runtime_completed": True, "worker_returncode": 0,
                       "server_returncode": 0}
        elif index == 3:
            outcome = {"task_id": task_id,
                       "outcome": "runtime_failure_in_denominator",
                       "runtime_completed": False, "worker_returncode": None,
                       "server_returncode": 2}
        else:
            outcome = {"task_id": task_id, "outcome": "not_started",
                       "in_fixed_denominator": True, "official_summary": None}
        outcomes.append(outcome)
        if index < 4:
            (lane_root / "results/task_shards" / task_id).mkdir(parents=True)
    _write(lane_root / "results/stage_manifest.json", {
        "schema": "a-history-system-run-v1",
        "status": "stopped_on_actor_runtime_failure", "state": "failed",
        "task_ids": shard1_tasks, "completed_task_cells": 4,
        "task_outcomes": outcomes, "automatic_retries": 0,
        "automatic_reruns": 0})
    _write(lane_root / "run/status.json", {
        "schema": "evidence-sets-eval-lane-status-v1",
        "state": "failed_no_rerun", "runner": {"returncode": 2},
        "completed_task_cells": 4, "automatic_retries": 0,
        "automatic_reruns": 0})
    return canonical, failed


def test_prepare_freezes_only_six_never_started_tasks_and_schedule(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    canonical, failed = _inputs(tmp_path, monkeypatch)
    root = tmp_path / "remaining"
    receipt = remaining.prepare(root, canonical, failed)
    assert (root / "continuation.json").is_file()
    assert receipt["schema"] == remaining.SCHEMA
    assert [row["task_id"] for row in receipt["rows"]] == list(remaining.TASKS)
    assert [row["physical_device"] for row in receipt["rows"]] == [0, 1, 2, 6, 0, 1]
    assert [row["wave"] for row in receipt["rows"]] == [0, 0, 0, 0, 1, 1]
    assert receipt["total_task_execution_budget"] == 6
    source_artifact = canonical / f"lanes/{remaining.LANE}/trained_artifact.json"
    for index, task_id in enumerate(remaining.TASKS):
        package = root / f"{remaining.LANE}.remaining{index}"
        verified = remaining.verify(package, pristine=True)
        assert verified["task_id"] == task_id
        assert remaining.base._sha(
            package / f"lanes/{remaining.LANE}/trained_artifact.json") == (
                remaining.base._sha(source_artifact))
        assert remaining._read(package / "launch_contract.json")[
            "required_environment"] == {"C2KV_STRICT_NONFINITE_SAMPLING": "1"}


def test_prepare_rejects_non_not_started_or_native_task_folder(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    canonical, failed = _inputs(tmp_path, monkeypatch)
    manifest_path = failed / f"lanes/{remaining.LANE}/results/stage_manifest.json"
    manifest = remaining._read(manifest_path)
    manifest["task_outcomes"][4]["outcome"] = "official_completed"
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="not pristine never-started"):
        remaining.prepare(tmp_path / "bad", canonical, failed)

    canonical, failed = _inputs(tmp_path / "native", monkeypatch)
    native = failed / f"lanes/{remaining.LANE}/results/task_shards/{remaining.TASKS[0]}"
    native.mkdir(parents=True)
    with pytest.raises(ValueError, match="not pristine never-started"):
        remaining.prepare(tmp_path / "native_bad", canonical, failed)


def test_origin_rejects_duplicate_task_outcomes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    canonical, failed = _inputs(tmp_path, monkeypatch)
    manifest_path = failed / f"lanes/{remaining.LANE}/results/stage_manifest.json"
    manifest = remaining._read(manifest_path)
    manifest["task_outcomes"][-1]["task_id"] = remaining.TASKS[0]
    _write(manifest_path, manifest)
    with pytest.raises(ValueError, match="terminal failure|duplicate"):
        remaining._origin(canonical, failed)


def test_verify_rejects_scientific_mutation_even_if_refrozen(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    canonical, failed = _inputs(tmp_path, monkeypatch)
    root = tmp_path / "remaining"
    remaining.prepare(root, canonical, failed)
    package = root / f"{remaining.LANE}.remaining0"
    artifact = package / f"lanes/{remaining.LANE}/trained_artifact.json"
    _write(artifact, {"changed": True})
    _write(package / "static_files.json", remaining.base._static_manifest(package))
    with pytest.raises(ValueError, match="changed controller, artifact"):
        remaining.verify(package)


def test_run_checks_pristine_device_environment_and_existing_runner(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lane = remaining.shard_spec(0)
    monkeypatch.setattr(remaining, "verify", lambda _package, pristine: {
        "lane_spec": lane, "lane": remaining.LANE})
    calls = []
    monkeypatch.setattr(remaining.base, "_assert_lane_free",
                        lambda value: calls.append(("free", value["physical_device"])))
    monkeypatch.setattr(remaining.trained, "ascend_environment",
                        lambda: calls.append(("ascend", None)))
    monkeypatch.setattr(remaining.trained, "enable_strict_sampling",
                        lambda: calls.append(("strict", None)))
    monkeypatch.setattr(remaining.base, "run_lane",
                        lambda package, lane_name: calls.append(("run", lane_name)) or 0)
    assert remaining.run(tmp_path / "package") == 0
    assert calls == [("free", 0), ("ascend", None), ("strict", None),
                     ("run", remaining.LANE)]


def test_wait_run_obeys_predecessor_and_only_waits_for_occupancy(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / f"{remaining.LANE}.remaining4"
    predecessor = tmp_path / f"{remaining.LANE}.remaining0/lanes/{remaining.LANE}/run/status.json"
    _write(predecessor, {"state": "failed_no_rerun"})
    lane = remaining.shard_spec(4)
    monkeypatch.setattr(remaining, "verify", lambda _package, pristine: {
        "lane_spec": lane})
    calls = []

    def free(_lane):
        calls.append("free")
        if len(calls) == 1:
            raise RuntimeError("Physical NPU 0 acquired by PIDs [9]")

    monkeypatch.setattr(remaining.base, "_assert_lane_free", free)
    monkeypatch.setattr(remaining.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(remaining, "run", lambda _package: calls.append("run") or 0)
    assert remaining.wait_run(package, poll_seconds=0) == 0
    assert calls == ["free", "free", "run"]
