from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import evidence_eval_trained_parallel as parallel


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fake_canonical(package: Path, lane_name: str = "C1") -> Path:
    tasks = {"schema": "a-history-system-task-manifest-v1",
             "task_ids": [f"multi_turn_base_{index}" for index in range(20)]}
    lane = next(row for row in parallel.trained.lane_specs() if row["name"] == lane_name)
    lane_root = package / "lanes" / lane_name
    _write(package / "tasks.d20.json", tasks)
    _write(package / "tasks.t02.expanded104.json", {"task_ids": ["multi_turn_base_100"]})
    _write(package / "remote_lineage.json", {"schema": "fixture"})
    _write(lane_root / "runtime/configs/controller.json",
           {"gp_experiments": {"set_selector": lane["selector"], "fixture": "science"}})
    (lane_root / "runtime/python/frozen.py").parent.mkdir(parents=True, exist_ok=True)
    (lane_root / "runtime/python/frozen.py").write_text("FROZEN = True\n", encoding="utf-8")
    _write(lane_root / "gp.json", {"set_selector": lane["selector"], "fixture": "science"})
    _write(lane_root / "trained_artifact.json", {"model_kind": "fixture", "weights": [1, 2]})
    _write(lane_root / "tasks.json", tasks)
    _write(lane_root / "lane.json", lane)
    _write(lane_root / "preview.json", {"task_count": 20})
    _write(lane_root / "design.json", {
        "schema": "a-history-system-candidate-design-v1",
        "status": "frozen",
        "task_ids": tasks["task_ids"],
        "task_manifest_sha256": parallel.base._sha(package / "tasks.d20.json"),
        "limits": {"tasks": 20, "steps": 40},
        "runtime": {"sglang_backend_url": f"http://127.0.0.1:{lane['engine_port']}",
                    "controller": "configs/controller.json"},
        "resolved_configs": {"controller": {"fixture": "science"}},
        "search_contract": {"fixed_denominator": "r001_mixed20",
                            "trained_artifact_sha256": "a" * 64},
        "automatic_reruns": 0,
    })
    sglang_file = package / "sglang/python/sglang/sampling.py"
    sglang_file.parent.mkdir(parents=True, exist_ok=True)
    sglang_file.write_text("STRICT_NONFINITE = True\n", encoding="utf-8")
    _write(package / "sglang_files.json", {
        "schema": "evidence-sets-eval-sglang-files-v1", "file_count": 1,
        "files": {"sglang/python/sglang/sampling.py": parallel.base._sha(sglang_file)}})
    (package / "evidence_eval.py").write_text("# fixture\n", encoding="utf-8")
    (package / "evidence_eval_trained.py").write_text("# fixture\n", encoding="utf-8")
    (package / "evidence_eval_trained_parallel.py").write_text("# fixture\n", encoding="utf-8")
    (package / "history_system/runner.py").parent.mkdir(parents=True, exist_ok=True)
    (package / "history_system/runner.py").write_text("# fixture\n", encoding="utf-8")
    _write(package / "launch_contract.json", {
        "schema": "evidence-sets-trained-eval-v1", "launch_authorized": False,
        "purpose": "parallel_shard_source_only", "total_task_execution_budget": 20,
        "automatic_retries": 0, "automatic_reruns": 0, "lanes": [lane],
        "required_environment": {"C2KV_STRICT_NONFINITE_SAMPLING": "1"},
    })
    _write(package / "static_files.json", parallel.base._static_manifest(package))
    return package


def _make_shards(tmp_path: Path, monkeypatch, lane_name: str = "C1"):
    canonical = _fake_canonical(tmp_path / "canonical", lane_name)
    monkeypatch.setattr(parallel.base, "_preview_lane",
                        lambda package, lane: {"task_count": 10,
                                               "engine_port": lane["engine_port"]})
    shard0 = tmp_path / "shard0"
    shard1 = tmp_path / "shard1"
    parallel._clone_shard(canonical, shard0, lane_name, 0)
    parallel._clone_shard(canonical, shard1, lane_name, 1)
    return canonical, shard0, shard1


def test_fixed_mapping_uses_all_six_authorized_devices_and_disjoint_ports():
    specs = [row for lane in parallel.LANE_SHARDS for row in parallel.shard_specs(lane)]
    assert [row["physical_device"] for row in specs] == [0, 4, 1, 6, 2, 3]
    assert not {5, 7}.intersection(row["physical_device"] for row in specs)
    ports = [port for row in specs for port in [row["engine_port"], *row["task_ports"]]]
    assert len(ports) == len(set(ports))


def test_clone_partitions_ordered_d20_and_preserves_scientific_files(tmp_path, monkeypatch):
    canonical, shard0, shard1 = _make_shards(tmp_path, monkeypatch)
    zero = parallel.verify_shard_package(shard0, require_pristine=True)
    one = parallel.verify_shard_package(shard1, require_pristine=True)
    original = json.loads((canonical / "tasks.d20.json").read_text())["task_ids"]
    assert zero["task_ids"] + one["task_ids"] == original
    assert set(zero["task_ids"]).isdisjoint(one["task_ids"])
    for relative in ("runtime/configs/controller.json", "gp.json", "trained_artifact.json"):
        source = canonical / "lanes/C1" / relative
        assert parallel.base._sha(shard0 / "lanes/C1" / relative) == parallel.base._sha(source)
        assert parallel.base._sha(shard1 / "lanes/C1" / relative) == parallel.base._sha(source)
    assert json.loads((shard0 / "launch_contract.json").read_text())[
        "required_environment"] == {"C2KV_STRICT_NONFINITE_SAMPLING": "1"}


def test_shard_verifier_rejects_scientific_design_change_even_if_refrozen(tmp_path, monkeypatch):
    _, shard0, _ = _make_shards(tmp_path, monkeypatch)
    design_path = shard0 / "lanes/C1/design.json"
    design = json.loads(design_path.read_text())
    design["resolved_configs"]["controller"]["fixture"] = "changed"
    _write(design_path, design)
    _write(shard0 / "static_files.json", parallel.base._static_manifest(shard0))
    with pytest.raises(ValueError, match="scientific design"):
        parallel.verify_shard_package(shard0)


def test_prepare_calls_canonical_builder_once_and_freezes_both_shards(tmp_path, monkeypatch):
    root = tmp_path / "parallel"
    training = tmp_path / "prepared_v8"
    binding = tmp_path / "binding.json"
    _write(binding, {"fixture": True})
    calls = []
    monkeypatch.setattr(parallel.trained, "verify_training_binding",
                        lambda *args: {"status": "passed"})

    def fake_prepare(package, source, training_path, lane_name, binding_path):
        calls.append((package, lane_name))
        _fake_canonical(package, lane_name)
        return {"status": "passed"}

    monkeypatch.setattr(parallel.trained, "prepare", fake_prepare)
    monkeypatch.setattr(parallel.base, "_preview_lane",
                        lambda package, lane: {"task_count": 10})
    receipt = parallel.prepare_parallel(root, tmp_path / "source", training, "C1", binding)
    assert len(calls) == 1
    assert receipt["status"] == "prepared"
    assert len(receipt["shards"]) == 2
    assert receipt["shards"][0]["task_ids"] + receipt["shards"][1]["task_ids"] == (
        receipt["original_d20_task_ids"])
    with pytest.raises(FileExistsError, match="overwrite"):
        parallel.prepare_parallel(root, tmp_path / "source", training, "C1", binding)


def _native_results(package: Path, lane_name: str, task_ids: list[str]) -> None:
    lane_root = package / "lanes" / lane_name
    outcomes = []
    for task_id in task_ids:
        shard = lane_root / "results/task_shards" / task_id
        official = {"benchmark": "BFCL", "categories": ["multi_turn"], "mode": "official",
                    "n_total": 1, "n_scored": 1, "correct_count": 1,
                    "semantic_score": 1.0, "scored": True,
                    "total_gold_checker_seconds": 0.1, "total_handler_http_calls": 1}
        final = {"status": "completed", "stop_reason": "finished",
                 "wall_seconds": 1.0, "wall_seconds_final": True,
                 "cost_summary": {"generation_attempts": 1, "costs": {}}}
        _write(shard / "bfcl/official_summary.json", official)
        _write(shard / "server/final.json", final)
        outcomes.append({"task_id": task_id, "outcome": "official_completed",
                         "in_fixed_denominator": True, "worker_returncode": 0,
                         "server_returncode": 0, "runtime_completed": True})
    _write(lane_root / "results/stage_manifest.json", {
        "schema": "a-history-system-run-v1", "status": "completed_fixed_manifest",
        "state": "completed", "task_ids": task_ids, "completed_task_cells": 10,
        "task_outcomes": outcomes, "automatic_retries": 0, "automatic_reruns": 0})
    _write(lane_root / "run/status.json", {
        "schema": "evidence-sets-eval-lane-status-v1", "state": "completed",
        "runner": {"returncode": 0}, "completed_task_cells": 10,
        "automatic_retries": 0, "automatic_reruns": 0})


def test_aggregate_reconstructs_actual_native_twenty_without_copying_results(tmp_path, monkeypatch):
    root = tmp_path / "parallel"
    canonical, shard0, shard1 = _make_shards(root, monkeypatch)
    source = parallel._canonical_source(canonical, "C1")
    manifest = {"schema": parallel.SCHEMA, "status": "prepared", "lane": "C1",
                "canonical_package": str(canonical), "canonical_sha256": source["sha256"],
                "original_d20_task_ids": source["task_ids"]}
    _write(root / "parallel_manifest.json", manifest)
    zero = parallel.verify_shard_package(shard0)["task_ids"]
    one = parallel.verify_shard_package(shard1)["task_ids"]
    _native_results(shard0, "C1", zero)
    _native_results(shard1, "C1", one)
    result = parallel.aggregate_results(root)
    assert result["status"] == "completed" and result["task_count"] == 20
    assert result["declared_tasks"] == zero + one
    assert list(result["task_observations"]) == zero + one
    first = result["task_observations"][zero[0]]
    assert first["official_summary"]["scored"] is True
    assert first["native_artifacts"]["official_summary"]["sha256"]


def test_aggregate_rejects_partial_native_shard(tmp_path, monkeypatch):
    root = tmp_path / "parallel"
    canonical, shard0, shard1 = _make_shards(root, monkeypatch)
    source = parallel._canonical_source(canonical, "C1")
    _write(root / "parallel_manifest.json", {
        "schema": parallel.SCHEMA, "status": "prepared", "lane": "C1",
        "canonical_package": str(canonical), "canonical_sha256": source["sha256"],
        "original_d20_task_ids": source["task_ids"]})
    zero = parallel.verify_shard_package(shard0)["task_ids"]
    _native_results(shard0, "C1", zero)
    with pytest.raises(RuntimeError, match="no terminal native result"):
        parallel.aggregate_results(root)


def test_run_shard_checks_device_then_uses_existing_once_only_runner(tmp_path, monkeypatch):
    lane = parallel.shard_specs("C1")[0]
    monkeypatch.setattr(parallel, "verify_shard_package",
                        lambda package, require_pristine: {
                            "lane": "C1", "lane_spec": lane})
    calls = []
    monkeypatch.setattr(parallel.base, "_assert_lane_free",
                        lambda value: calls.append(("free", value["physical_device"])))
    monkeypatch.setattr(parallel.trained, "ascend_environment",
                        lambda: calls.append(("ascend", None)))
    monkeypatch.setattr(parallel.trained, "enable_strict_sampling",
                        lambda: calls.append(("strict", None)))
    monkeypatch.setattr(parallel.base, "run_lane",
                        lambda package, lane_name: calls.append(("run", lane_name)) or 0)
    assert parallel.run_shard(tmp_path / "shard") == 0
    assert calls == [("free", 0), ("ascend", None), ("strict", None), ("run", "C1")]


def test_waiter_rejects_terminal_training_failure_without_polling(tmp_path, monkeypatch):
    training = tmp_path / "prepared_v8"
    _write(training / "status.json", {"phase": "partial_failed"})
    monkeypatch.setattr(parallel.trained, "verify_training_binding",
                        lambda *args: pytest.fail("must not verify a failed run"))
    with pytest.raises(RuntimeError, match="forbidden"):
        parallel.wait_for_training(training, tmp_path / "binding", "C1",
                                   poll_seconds=0, timeout_seconds=0)
