from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

import evidence_eval as base
import evidence_eval_expansion as expansion
from evidence_expansion import LANES


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source(tmp_path: Path, controller: str, *, trained: bool) -> tuple[Path, dict]:
    source = tmp_path / f"source-{controller}"
    source.mkdir(parents=True)
    lane_root = source / "lanes" / controller
    runtime = lane_root / "runtime"
    shutil.copyfile(Path(base.__file__), source / "evidence_eval.py")
    runner = source / "history_system/runner.py"
    runner.parent.mkdir(parents=True)
    runner.write_text("# frozen runner fixture\n", encoding="utf-8")
    sglang_file = source / "sglang/python/sglang/frozen.py"
    sglang_file.parent.mkdir(parents=True)
    sglang_file.write_text("SOURCE = 'frozen'\n", encoding="utf-8")
    controller_config = {"route": controller, "semantic_budget": 4}
    artifact_rows = []
    trained_sha = None
    if trained:
        artifact = lane_root / "trained_artifact.json"
        _write(artifact, {"model_kind": "fixture", "fit": {"coef": [1.0]}})
        trained_sha = expansion._sha(artifact)
        controller_config["gp_experiments"] = {
            "selector_artifact": json.loads(artifact.read_text())}
        artifact_rows.append({"path": f"lanes/{controller}/trained_artifact.json",
                              "sha256": trained_sha})
    _write(runtime / "configs/controller.json", controller_config)
    _write(runtime / "configs/eval_policy.json", {"policy": "fixed"})
    tasks = [f"multi_turn_base_{index}" for index in range(20)]
    design = {
        "schema": "a-history-system-candidate-design-v1",
        "status": "frozen",
        "candidate_id": f"algorithm-{controller}",
        "run_id_template": f"run-{controller}",
        "task_ids": tasks,
        "task_manifest_sha256": "0" * 64,
        "limits": {"tasks": 20, "max_generation_calls_per_task": 96},
        "runtime": {"controller": "configs/controller.json",
                    "sglang_backend_url": "http://127.0.0.1:37200"},
        "resolved_configs": {"controller": controller_config},
        "search_contract": {"fixed_denominator": "D20", "history": "H0",
                            **({"trained_artifact_sha256": trained_sha} if trained else {})},
        "automatic_reruns": 0,
    }
    _write(lane_root / "design.json", design)
    _write(lane_root / "lane.json", {"name": controller, "selector": "fixture",
                                      "physical_device": 0, "engine_port": 1,
                                      "task_port_base": 2, "model_smokes": ["embedding"],
                                      "task_ports": list(range(2, 22))})
    _write(source / "launch_contract.json", {
        "schema": "fixture-source-launch-v1",
        "launch_authorized": True,
        "required_environment": (
            {"C2KV_STRICT_NONFINITE_SAMPLING": "1"} if trained else {}),
    })
    _write(source / "provenance.json", {"source": f"frozen-{controller}"})
    _write(lane_root / "gp.json", {"controller": controller, "selector": "fixture"})
    sglang_manifest = expansion._manifest(
        source, "evidence-sets-eval-sglang-files-v1",
        excluded=tuple(name for name in expansion._tree_hashes(source)
                       if not name.startswith("sglang/")),
    )
    _write(source / "sglang_files.json", sglang_manifest)
    static_manifest = expansion._manifest(
        source, "evidence-sets-eval-static-files-v1", skip_sglang=True,
        excluded=("static_files.json",),
    )
    _write(source / "static_files.json", static_manifest)
    binding = {
        "controller": controller,
        "source_package": str(source.resolve()),
        "source_lane": controller,
        "source_algorithm_id": design["candidate_id"],
        "source_static_files_sha256": expansion._sha(source / "static_files.json"),
        "source_sglang_files_sha256": expansion._sha(source / "sglang_files.json"),
        "source_design_sha256": expansion._sha(lane_root / "design.json"),
        "source_controller_sha256": expansion._sha(runtime / "configs/controller.json"),
        "model_artifacts": artifact_rows,
        "required_env": ({"C2KV_STRICT_NONFINITE_SAMPLING": "1"} if trained else {}),
    }
    return source, binding


def _readiness(selected=("C0", "C1")) -> dict:
    remaining = [f"multi_turn_base_{index}" for index in range(20, 128)]
    reused = [f"multi_turn_base_{index}" for index in range(20)]
    manifests = {
        controller: {
            "schema": "a-history-system-task-manifest-v1",
            "manifest_id": f"exp3_{controller}_d128_remaining108",
            "stage": "development_search",
            "task_ids": remaining,
            "reused_task_ids": reused,
            "full_task_ids": reused + remaining,
            "reuse_requires_unchanged_algorithm_and_verified_d20_provenance": True,
        }
        for controller in selected
    }
    devices = ((0, 1, 2), (3, 4, 6))
    shards = []
    for rank, controller in enumerate(selected):
        for part, device in enumerate(devices[rank]):
            shards.append({
                "shard_id": f"{controller}_part{part}",
                "controller": controller,
                "preferred_physical_device": device,
                "wait_for_device_free_and_verify_owner": True,
                "task_ids": remaining[part::3],
                "task_budget": 36,
                "automatic_retries": 0,
                "launch_authorized": False,
            })
    return {
        "schema": expansion.READINESS_SCHEMA,
        "quality_label": "preliminary, n=1",
        "launch_authorized": False,
        "phase": "promotion_ready",
        "promotion": {"phase": "promotion_ready", "selected": list(selected)},
        "lanes": {
            controller: {
                "all_cells_terminal": True,
                "clean_d20": True,
                "quality_cells": [
                    {"task_id": task, "quality_source_id": f"owner-{controller}-{index}",
                     "status": "completed"}
                    for index, task in enumerate(reused)
                ],
            }
            for controller in LANES
        },
        "next_manifests": manifests,
        "new_execution_count": 216,
        "proposed_shards": shards,
        "automatic_retries": 0,
        "reuse_validation": {
            "status": "audited_historical_lanes_bound_to_target",
            "audited_lanes": ["C0", "C3", "C5"],
        },
        "inputs": {"reuse_audit": {"path": "fixture-audit.json", "sha256": "a" * 64}},
    }


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    readiness_path = tmp_path / "readiness.json"
    bindings_path = tmp_path / "bindings.json"
    c0_source, c0 = _source(tmp_path, "C0", trained=False)
    _, c1 = _source(tmp_path, "C1", trained=True)
    c0["source_id"] = "failed_repair_v1"
    c1["source_id"] = "eval_trained_v4_C1"
    runtime_relative = "configs/controller.json"
    runtime_hash = expansion._sha(c0_source / "lanes/C0/runtime" / runtime_relative)
    audit = {
        "schema": "evidence-sets-d20-runtime-compatibility-v1",
        "target": {"source_id": "failed_repair_v1",
                   "provenance_sha256": expansion._sha(c0_source / "provenance.json")},
        "conclusion": {
            "status": "semantic_nonactivation_compatible",
            "result_reuse_supported": True,
            "lanes": {"C0": "semantic_nonactivation_compatible_20_of_20"},
        },
        "lanes": {"C0": {
            "classification": "semantic_nonactivation_compatible_20_of_20",
            "target_design_sha256": c0["source_design_sha256"],
            "target_gp_sha256": expansion._sha(c0_source / "lanes/C0/gp.json"),
            "partition_analysis": {"fixture": {"audited_file_hashes": {
                runtime_relative: {"source": runtime_hash, "target": runtime_hash}
            }}},
        }},
    }
    audit_path = tmp_path / "audit.json"
    _write(audit_path, audit)
    readiness = _readiness()
    readiness["inputs"]["reuse_audit"] = {
        "path": str(audit_path), "sha256": expansion._sha(audit_path)}
    _write(readiness_path, readiness)
    _write(bindings_path, {
        "schema": expansion.SOURCE_BINDINGS_SCHEMA,
        "reuse_audit": {"path": str(audit_path), "sha256": expansion._sha(audit_path)},
        "bindings": {"C0": c0, "C1": c1},
    })
    return readiness_path, bindings_path


def _authorization(package: Path) -> dict:
    receipt = expansion.authorization_requirements(package)
    receipt["status"] = "authorized"
    receipt["launch_authorized"] = True
    receipt["authorized_by"] = "root"
    return receipt


def test_prepared_six_shards_have_exact_budget_no_overlap_and_reserved_ports(tmp_path: Path):
    readiness, bindings = _inputs(tmp_path)
    package = tmp_path / "expansion"
    expansion.prepare(package, readiness, bindings)
    receipt = expansion.verify_package(package)
    contract = json.loads((package / "expansion_contract.json").read_text())
    assert receipt["total_task_execution_budget"] == 216
    assert [row["device"] for row in contract["shards"]] == [0, 1, 2, 3, 4, 6]
    assert [row["engine_port"] for row in contract["shards"]] == [
        19000, 19010, 19020, 19030, 19040, 19060]
    assert [row["task_port_base"] for row in contract["shards"]] == [
        20000, 20200, 20400, 20600, 20800, 21200]
    pairs = []
    for row in contract["shards"]:
        tasks = json.loads((package / "shards" / row["shard_id"] / "tasks.json").read_text())["task_ids"]
        assert len(tasks) == len(set(tasks)) == 36
        pairs.extend((row["controller"], task) for task in tasks)
    assert len(pairs) == len(set(pairs)) == 216


def test_clone_rebinds_real_source_endpoint_and_preserves_real_artifact(tmp_path: Path):
    readiness, bindings = _inputs(tmp_path)
    package = tmp_path / "expansion"
    expansion.prepare(package, readiness, bindings)
    shard = package / "shards/C1_part0"
    source_design = json.loads((shard / "source_design.json").read_text())
    design = json.loads((shard / "lanes/C1_part0/design.json").read_text())
    expansion.assert_clone_only_operational_changes(source_design, design)
    lane = json.loads((shard / "lanes/C1_part0/lane.json").read_text())
    assert source_design["runtime"]["sglang_backend_url"] == "http://127.0.0.1:37200"
    assert design["runtime"]["sglang_backend_url"] == "http://127.0.0.1:19030"
    assert design["runtime"]["sglang_backend_url"] == (
        f"http://127.0.0.1:{lane['engine_port']}")
    assert design["candidate_id"] == source_design["candidate_id"]
    assert design["resolved_configs"] == source_design["resolved_configs"]
    assert design["limits"]["max_generation_calls_per_task"] == 96
    artifact = shard / "lanes/C1_part0/model_artifacts/trained_artifact.json"
    assert artifact.is_file() and json.loads(artifact.read_text())["fit"] == {"coef": [1.0]}
    provenance = json.loads((shard / "provenance.json").read_text())
    assert provenance["required_env"] == {"C2KV_STRICT_NONFINITE_SAMPLING": "1"}
    assert provenance["historical_d20_and_new108_single_exact_config_claim_allowed"] is False


def test_shard_verifier_rejects_source_endpoint_instead_of_lane_port(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    readiness, bindings = _inputs(tmp_path)
    package = tmp_path / "expansion"
    expansion.prepare(package, readiness, bindings)
    contract = json.loads((package / "expansion_contract.json").read_text())
    row = contract["shards"][0]
    shard = package / "shards" / row["shard_id"]
    design_path = shard / "lanes" / row["shard_id"] / "design.json"
    design = json.loads(design_path.read_text())
    design["runtime"]["sglang_backend_url"] = "http://127.0.0.1:37200"
    _write(design_path, design)
    monkeypatch.setattr(
        expansion, "_load_base_module",
        lambda _path: SimpleNamespace(verify_package=lambda _root: {"status": "passed"}),
    )
    with pytest.raises(ValueError, match="Frozen shard contract differs"):
        expansion._verify_shard(shard, row)


def test_empty_trained_or_synthetic_untrained_artifacts_are_rejected(tmp_path: Path):
    readiness_path, bindings_path = _inputs(tmp_path)
    bindings = json.loads(bindings_path.read_text())
    trained_source = Path(bindings["bindings"]["C1"]["source_package"])
    artifact = trained_source / "lanes/C1/trained_artifact.json"
    _write(artifact, {})
    bindings["bindings"]["C1"]["model_artifacts"][0]["sha256"] = expansion._sha(artifact)
    design = json.loads((trained_source / "lanes/C1/design.json").read_text())
    design["search_contract"]["trained_artifact_sha256"] = expansion._sha(artifact)
    _write(trained_source / "lanes/C1/design.json", design)
    bindings["bindings"]["C1"]["source_design_sha256"] = expansion._sha(
        trained_source / "lanes/C1/design.json")
    _write(trained_source / "static_files.json", expansion._manifest(
        trained_source, "evidence-sets-eval-static-files-v1", skip_sglang=True,
        excluded=("static_files.json",)))
    bindings["bindings"]["C1"]["source_static_files_sha256"] = expansion._sha(
        trained_source / "static_files.json")
    _write(bindings_path, bindings)
    with pytest.raises(ValueError, match="empty or untrained"):
        expansion.prepare(tmp_path / "bad", readiness_path, bindings_path)

    _, c0 = _source(tmp_path / "second", "C0", trained=True)
    c0["required_env"] = {"C2KV_STRICT_NONFINITE_SAMPLING": "1"}
    source = Path(c0["source_package"])
    design = json.loads((source / "lanes/C0/design.json").read_text())
    design["search_contract"].pop("trained_artifact_sha256")
    _write(source / "lanes/C0/design.json", design)
    c0["source_design_sha256"] = expansion._sha(source / "lanes/C0/design.json")
    _write(source / "static_files.json", expansion._manifest(
        source, "evidence-sets-eval-static-files-v1", skip_sglang=True,
        excluded=("static_files.json",)))
    c0["source_static_files_sha256"] = expansion._sha(source / "static_files.json")
    with pytest.raises(ValueError, match="synthetic training artifact"):
        expansion._verify_artifacts(source, design, c0)


def test_occupied_device_blocks_before_runner_and_does_not_kill_or_write(monkeypatch, tmp_path: Path):
    readiness, bindings = _inputs(tmp_path)
    package = tmp_path / "expansion"
    expansion.prepare(package, readiness, bindings)
    authorization = tmp_path / "authorization.json"
    _write(authorization, _authorization(package))
    called = {"run": 0}

    def occupied(_lane):
        raise RuntimeError("Physical NPU 0 acquired by PIDs [999]")

    fake = SimpleNamespace(
        LANES={}, DEFAULT_REMOTE_ROOT=None,
        _assert_lane_free=occupied,
        run_lane=lambda *_: called.__setitem__("run", called["run"] + 1),
    )
    monkeypatch.setattr(expansion, "verify_package", lambda _package: {"status": "passed"})
    monkeypatch.setattr(expansion, "_load_base_module", lambda _path: fake)
    with pytest.raises(RuntimeError, match="acquired by PIDs"):
        expansion.run_shard(package, "C0_part0", authorization)
    assert called["run"] == 0
    assert not (package / "shards/C0_part0/lanes/C0_part0/run").exists()
    assert not (package / "shards/C0_part0/lanes/C0_part0/results").exists()


def test_no_overwrite_and_frozen_mutation_is_detected(tmp_path: Path):
    readiness, bindings = _inputs(tmp_path)
    package = tmp_path / "expansion"
    expansion.prepare(package, readiness, bindings)
    with pytest.raises(FileExistsError, match="overwrite"):
        expansion.prepare(package, readiness, bindings)
    controller = package / "shards/C0_part0/lanes/C0_part0/runtime/configs/controller.json"
    original = json.loads(controller.read_text())
    original["semantic_budget"] = 999
    _write(controller, original)
    with pytest.raises(RuntimeError, match="changed"):
        expansion.verify_package(package)


def test_run_requires_exact_external_authorization_and_refuses_existing_output(tmp_path: Path):
    readiness, bindings = _inputs(tmp_path)
    package = tmp_path / "expansion"
    expansion.prepare(package, readiness, bindings)
    bad = tmp_path / "bad-auth.json"
    receipt = _authorization(package)
    receipt["total_task_execution_budget"] = 217
    _write(bad, receipt)
    with pytest.raises(ValueError, match="exact frozen package"):
        expansion.verify_authorization(package, bad)
    lane = package / "shards/C0_part0/lanes/C0_part0"
    (lane / "results").mkdir()
    good = tmp_path / "good-auth.json"
    _write(good, _authorization(package))
    with pytest.raises(FileExistsError, match="rerun"):
        expansion.run_shard(package, "C0_part0", good)


def test_frozen_package_verifies_in_isolated_subprocess_without_worktree_imports(tmp_path: Path):
    readiness, bindings = _inputs(tmp_path)
    package = tmp_path / "expansion"
    expansion.prepare(package, readiness, bindings)
    completed = subprocess.run(
        [sys.executable, "-I", str(package / "evidence_eval_expansion.py"),
         "verify-package", "--package", str(package)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["total_task_execution_budget"] == 216


def test_source_binding_must_match_readiness_audit_and_exact_target_runtime(tmp_path: Path):
    readiness_path, bindings_path = _inputs(tmp_path)
    readiness = json.loads(readiness_path.read_text())
    bindings = json.loads(bindings_path.read_text())
    bindings["bindings"]["C0"]["source_id"] = "never_started_v3"
    with pytest.raises(ValueError, match="audited D20 target"):
        expansion.verify_source_bindings(readiness, bindings)

    bindings = json.loads(bindings_path.read_text())
    bindings["reuse_audit"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs from promotion readiness"):
        expansion.verify_source_bindings(readiness, bindings)


def test_authorized_trained_shard_applies_frozen_env_and_binding_before_run(monkeypatch, tmp_path: Path):
    readiness, bindings = _inputs(tmp_path)
    package = tmp_path / "expansion"
    expansion.prepare(package, readiness, bindings)
    authorization = tmp_path / "authorization.json"
    _write(authorization, _authorization(package))
    calls = []

    def free(lane):
        calls.append(("free", lane["name"], len(lane["task_ports"])))
        return {"ports_bindable": True}

    def run(shard_package, shard_id):
        calls.append(("run", shard_id, str(shard_package)))
        assert expansion.os.environ["C2KV_STRICT_NONFINITE_SAMPLING"] == "1"
        return 0

    fake = SimpleNamespace(
        LANES={}, DEFAULT_REMOTE_ROOT=None, _assert_lane_free=free, run_lane=run)
    monkeypatch.setattr(expansion, "verify_package", lambda _package: {"status": "passed"})
    monkeypatch.setattr(expansion, "_load_base_module", lambda _path: fake)
    assert expansion.run_shard(package, "C1_part0", authorization) == 0
    assert calls[0] == ("free", "C1_part0", 36)
    assert calls[1][0:2] == ("run", "C1_part0")
    assert list(fake.LANES) == ["C1_part0"]
