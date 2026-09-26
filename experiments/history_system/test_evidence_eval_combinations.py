from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evidence_c1_h1_calibration as c1_calibration
import evidence_eval_combinations as combinations
import evidence_eval_expansion as expansion
import evidence_terminal_results as terminal_results
import runner as real_runner


def _save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path, controller: str, *, history: str = "current") -> dict:
    source = tmp_path / f"source_{controller}_{history}"
    source.mkdir()
    lane_name = "frozen_lane"
    runtime = source / "lanes" / lane_name / "runtime"
    shutil.copyfile(HERE / "evidence_eval.py", source / "evidence_eval.py")
    (source / "history_system").mkdir()
    (source / "history_system" / "runner.py").write_text("# frozen runner\n", encoding="utf-8")
    (source / "sglang").mkdir()
    (source / "sglang" / "version.py").write_text("VERSION = 'frozen'\n", encoding="utf-8")
    for relative in combinations.G04_SOURCE_HASHES:
        source_path = HERE / "runtime" / relative
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, target)
    gp = {
        "G": history,
        "P": 8,
        "E": 4,
        "F": False,
        "R": 1,
        "reserve_tokens": 0,
        "history_token_budget": 768,
        "set_selector": {
            "C0": "candidate_rule", "C2": "reranker", "C3": "local_llm",
            "C5": "parameter_source",
        }.get(controller, "candidate_rule"),
        "local_models": {"selector": {"model_name_or_path": "/model/frozen",
                                        "revision": "checkpoint-frozen"}},
    }
    model_artifacts = []
    if controller in {"C1", "C4_turn"}:
        artifact = {
            "model_kind": "c1_risk_logistic" if controller == "C1" else "c4_gain_turn",
            "fit": {"weights": [0.25], "bias": -0.1},
        }
        artifact_path = source / "model_artifacts" / "selector.json"
        _save(artifact_path, artifact)
        gp.update({
            "set_selector": "risk" if controller == "C1" else "gain_turn",
            "selector_artifact": artifact,
            "selector_threshold": 0.5,
        })
        if controller == "C4_turn":
            gp["gain_delta"] = 0.0
        model_artifacts.append({"path": artifact_path.relative_to(source).as_posix(),
                                "sha256": _sha(artifact_path)})
    controller_config = {"schema": "controller-v1", "unrelated": {"keep": True},
                         "gp_experiments": gp}
    _save(runtime / "configs" / "controller.json", controller_config)
    _save(runtime / "configs" / "eval_policy.json", {"policy": "frozen"})
    source_files = expansion._tree_hashes(runtime)
    search_contract = {
        "history": "H0" if history == "current" else "H1",
        "recovery_attempts": 1,
        "fixed_denominator": "D128",
        "history_token_budget": 768,
    }
    if model_artifacts:
        search_contract["trained_artifact_sha256"] = model_artifacts[0]["sha256"]
    algorithm_id = f"algo_{controller}_{history}"
    design = {
        "schema": "a-history-system-candidate-design-v1",
        "status": "frozen",
        "candidate_id": algorithm_id,
        "run_id_template": "source",
        "launch_authorized": True,
        "automatic_reruns": 0,
        "task_ids": [f"multi_turn_base_{index}" for index in range(128)],
        "task_manifest_sha256": "a" * 64,
        "limits": {"tasks": 128, "history_tokens": 768, "max_turns": 16},
        "sampling": dict(combinations.STRICT_SAMPLING),
        "checkpoint_selection": {"path": "/checkpoint/frozen", "step": 1000},
        "runtime": {"sglang_backend_url": "http://127.0.0.1:39000",
                    "dtype": "bfloat16", "device": "npu"},
        "resolved_configs": {"controller": controller_config,
                             "eval_policy": {"policy": "frozen"}},
        "search_contract": search_contract,
        "source_files": source_files,
    }
    _save(source / "lanes" / lane_name / "design.json", design)
    _save(source / "lanes" / lane_name / "lane.json",
          {"name": lane_name, "physical_device": 0, "engine_port": 39000,
           "task_port_base": 40000, "task_ports": list(range(40000, 40128))})
    required_env = {"C2KV_STRICT_NUMERIC": "1"} if model_artifacts else {}
    _save(source / "launch_contract.json", {"required_environment": required_env})
    _save(source / "sglang_files.json",
          expansion._manifest(source, "source-sglang-v1", excluded=(), skip_sglang=False))
    # Bind only the dedicated SGLang tree in the SGLang manifest.
    sglang_manifest = {
        "schema": "source-sglang-v1", "file_count": 1,
        "files": {"sglang/version.py": _sha(source / "sglang/version.py")},
    }
    _save(source / "sglang_files.json", sglang_manifest)
    _save(source / "static_files.json", expansion._manifest(
        source, "source-static-v1", skip_sglang=True,
        excluded=("static_files.json",)))

    config_source = tmp_path / "config_source"
    (config_source / "history_system" / "runtime").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(HERE / "evidence_sets.py",
                    config_source / "history_system" / "evidence_sets.py")
    for relative in combinations.G04_SOURCE_HASHES:
        target = config_source / "history_system" / "runtime" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / "runtime" / relative, target)

    manifest_path = tmp_path / "tasks.d128.json"
    if not manifest_path.exists():
        _save(manifest_path, {"schema": "task-manifest-v1",
                              "task_ids": [f"multi_turn_base_{index}"
                                           for index in range(128)]})
    receipt = tmp_path / f"complete_{controller}_{history}.json"
    tasks = [f"multi_turn_base_{index}" for index in range(128)]
    _save(receipt, {
        "schema": combinations.COMPLETE_RESULT_SCHEMA,
        "status": "completed",
        "package_kind": "fixture_full128",
        "stage": "H0_R1",
        "controller": controller,
        "source_algorithm_id": algorithm_id,
        "task_manifest_sha256": _sha(manifest_path),
        "completed_task_cells": 128,
        "runtime_failed_task_cells": 0,
        "pending_task_cells": 0,
        "configuration_binding": {"status": "exact_frozen_config"},
        "evidence": [{"path": "results.json", "sha256": "b" * 64}],
        "quality_cells": [
            {
                "task_id": task_id,
                "cohort": "full128",
                "quality_source_id": algorithm_id,
                "shard_id": None,
                "status": "completed",
                "status_reason": "official_completed",
                "runtime_completed": True,
                "worker_returncode": 0,
                "server_returncode": 0,
                "official": {"scored": True, "n_total": 1, "n_scored": 1,
                             "correct_count": index % 2,
                             "semantic_score": float(index % 2)},
                "measured_cost": {},
                "evidence": [{"path": f"task/{task_id}.json",
                              "sha256": "d" * 64}],
            }
            for index, task_id in enumerate(tasks)
        ],
    })
    return {
        "schema": combinations.SOURCE_SCHEMA,
        "controller": controller,
        "source_id": "failed_repair_v1",
        "source_package": str(source),
        "source_lane": lane_name,
        "source_algorithm_id": algorithm_id,
        "source_static_files_sha256": _sha(source / "static_files.json"),
        "source_sglang_files_sha256": _sha(source / "sglang_files.json"),
        "source_design_sha256": _sha(source / "lanes" / lane_name / "design.json"),
        "source_controller_sha256": _sha(runtime / "configs" / "controller.json"),
        "required_env": required_env,
        "model_artifacts": model_artifacts,
        "config_source_package": str(config_source),
        "source_evidence_sets_sha256": _sha(
            config_source / "history_system" / "evidence_sets.py"),
        "complete_d128_receipt": {"path": str(receipt), "sha256": _sha(receipt)},
    }


def _spec(tmp_path: Path, sources: list[dict], *, r3: dict | None = None,
          calibrations: dict | None = None,
          d20_promotion: dict | None = None) -> Path:
    manifest = tmp_path / "tasks.d128.json"
    value = {
        "schema": combinations.SPEC_SCHEMA,
        "d128_manifest": {"path": str(manifest), "sha256": _sha(manifest)},
        "h1_sources": sources,
        "c1_h1_calibrations": calibrations or {},
    }
    if r3 is not None:
        value["r3"] = r3
    if d20_promotion is not None:
        value["h1_d20_promotion_receipt"] = d20_promotion
    path = tmp_path / "spec.json"
    _save(path, value)
    return path


def _d20_promotion(tmp_path: Path, sources: list[dict]) -> dict:
    manifest = expansion._read(tmp_path / "tasks.d128.json")
    tasks = manifest["task_ids"]
    reused, new = tasks[:20], tasks[20:]
    lanes = {}
    next_manifests = {}
    for source in sources:
        controller = source["controller"]
        lanes[controller] = {
            "completed": 20,
            "clean_d20": True,
            "all_cells_terminal": True,
            "operational_selection_eligible": True,
            "operational_denominator": 20,
            "quality_cells": [
                {
                    "task_id": task_id,
                    "status": "completed",
                    "official": {"scored": True, "n_total": 1, "n_scored": 1,
                                 "correct_count": int(index % 3 == 0)},
                }
                for index, task_id in enumerate(reused)
            ],
        }
        next_manifests[controller] = {
            "schema": "a-history-system-task-manifest-v1",
            "task_ids": new,
            "reused_task_ids": reused,
            "full_task_ids": tasks,
        }
    receipt = tmp_path / "d20_promotion.json"
    _save(receipt, {
        "schema": combinations.D20_PROMOTION_SCHEMA,
        "phase": "promotion_ready",
        "launch_authorized": False,
        "promotion": {"phase": "promotion_ready",
                      "selected": list(combinations.D20_PROMOTED_CONTROLLERS)},
        "lanes": lanes,
        "next_manifests": next_manifests,
        "reuse_validation": {
            "status": "audited_historical_lanes_bound_to_target",
            "audited_lanes": list(combinations.D20_PROMOTED_CONTROLLERS),
            "target": {"source_id": "failed_repair_v1"},
        },
    })
    return {"path": str(receipt), "sha256": _sha(receipt)}


def _calibration(tmp_path: Path) -> dict:
    input_path = tmp_path / "c1_h1_calibration_input.json"
    _save(input_path, {"schema": c1_calibration.DATASET_SCHEMA,
                       "status": "completed", "evidence_mode": "production"})
    receipt_path = tmp_path / "c1_h1_calibration_receipt.json"
    _save(receipt_path, {
        "schema": c1_calibration.RECEIPT_SCHEMA,
        "status": "completed",
        "production_eligible": True,
        "selected_threshold": 0.625,
    })
    return {
        "receipt": {"path": str(receipt_path), "sha256": _sha(receipt_path)},
        "input": {"path": str(input_path), "sha256": _sha(input_path)},
    }


def _authorization(package: Path) -> dict:
    receipt = combinations.authorization_requirements(package)
    receipt["status"] = "authorized"
    receipt["launch_authorized"] = True
    receipt["authorized_by"] = "root"
    return receipt


def test_g04_hashes_match_frozen_evidence_sets_and_sources(tmp_path: Path) -> None:
    expected = {
        relative: _sha(HERE / "runtime" / relative)
        for relative in combinations.G04_SOURCE_HASHES
    }
    assert combinations.G04_SOURCE_HASHES == expected
    source = _source(tmp_path, "C0")
    packing = (Path(source["source_package"]) / "lanes" / source["source_lane"]
               / "runtime/python/history_memory/packing.py")
    packing.write_text("# drift\n", encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError), match="(source static_files|G04 source)"):
        combinations.prepare(tmp_path / "package", _spec(tmp_path, [source, _source(tmp_path, "C3")]))


def test_prepare_h1_two_sources_preserves_semantics_and_fixed_budgets(tmp_path: Path) -> None:
    sources = [_source(tmp_path, "C0"), _source(tmp_path, "C4_turn")]
    source_hashes = {
        source["controller"]: source["source_static_files_sha256"] for source in sources}
    package = tmp_path / "package"
    result = combinations.prepare(package, _spec(tmp_path, sources))
    assert result["status"] == "passed"
    assert result["h1_task_execution_budget"] == 256
    assert result["r3_task_execution_budget"] == 0
    contract = expansion._read(package / "combination_contract.json")
    assert len(contract["shards"]) == 6
    assert contract["launch_authorized"] is False
    assert {row["preferred_device"] for row in contract["shards"]} == {0, 1, 2, 3, 4, 6}
    assert {row["wave"] for row in contract["shards"]} == {0}
    assert [row["engine_port"] for row in contract["shards"]] == [
        23000, 23010, 23020, 23030, 23040, 23060]
    assert [row["task_port_base"] for row in contract["shards"]] == [
        24000, 24100, 24200, 24300, 24400, 24600]
    d128_tasks = expansion._read(package / "tasks.d128.json")["task_ids"]
    expected_devices = {"C0": [0, 1, 2], "C4_turn": [3, 4, 6]}
    for controller, devices in expected_devices.items():
        controller_rows = [row for row in contract["shards"]
                           if row["controller"] == controller]
        assert [row["preferred_device"] for row in controller_rows] == devices
        assert [row["task_budget"] for row in controller_rows] == [43, 43, 42]
        owned = []
        for row in controller_rows:
            owned.extend(expansion._read(
                package / "shards" / row["shard_id"] / "tasks.json")["task_ids"])
        assert len(owned) == 128
        assert len(set(owned)) == 128
        assert set(owned) == set(d128_tasks)
    for row in contract["shards"]:
        shard = package / "shards" / row["shard_id"]
        design = expansion._read(shard / "lanes" / row["shard_id"] / "design.json")
        launch = expansion._read(shard / "launch_contract.json")
        source_design = expansion._read(shard / "source_design.json")
        gp = design["resolved_configs"]["controller"]["gp_experiments"]
        source_gp = source_design["resolved_configs"]["controller"]["gp_experiments"]
        assert gp["G"] == "record_bound" and gp["R"] == 1
        assert design["sampling"] == combinations.STRICT_SAMPLING
        assert design["launch_authorized"] is True
        assert launch["launch_authorized"] is True
        assert design["runtime"]["sglang_backend_url"] == (
            f"http://127.0.0.1:{row['engine_port']}")
        assert design["checkpoint_selection"] == source_design["checkpoint_selection"]
        assert gp["history_token_budget"] == source_gp["history_token_budget"]
        if row["controller"] == "C4_turn":
            assert gp["selector_artifact"] == source_gp["selector_artifact"]
            semantics = expansion._read(shard / "provenance.json")["model_semantics"]
            assert semantics == {"artifact_training_history": "H0",
                                 "h1_trained": False,
                                 "artifact_migration": "fixed_H0_artifact_delta0"}
    for source in sources:
        assert _sha(Path(source["source_package"]) / "static_files.json") == source_hashes[
            source["controller"]]


def test_prepare_h1_directly_from_exact_d20_promotion_without_d128_results(
        tmp_path: Path) -> None:
    sources = [_source(tmp_path, "C0"), _source(tmp_path, "C5")]
    for source in sources:
        del source["complete_d128_receipt"]
    promotion = _d20_promotion(tmp_path, sources)
    package = tmp_path / "h1_from_d20"
    result = combinations.prepare(
        package, _spec(tmp_path, sources, d20_promotion=promotion))
    assert result == {
        "schema": "experiment3-combination-verification-v1",
        "status": "passed",
        "package": str(package.resolve()),
        "h1_task_execution_budget": 256,
        "r3_task_execution_budget": 0,
        "shard_count": 6,
    }
    contract = expansion._read(package / "combination_contract.json")
    assert contract["h1_controllers"] == ["C0", "C5"]
    assert contract["h1_source_gate"] == "d20_promotion"
    assert contract["h1_d20_promotion_receipt_sha256"] == promotion["sha256"]
    assert _sha(package / "h1_d20_promotion_receipt.json") == promotion["sha256"]
    assert [row["task_budget"] for row in contract["shards"]] == [43, 43, 42] * 2
    for row in contract["shards"]:
        provenance = expansion._read(
            package / "shards" / row["shard_id"] / "provenance.json")
        assert provenance["source_selection_provenance"] == {
            "mode": "d20_promotion",
            "receipt": "h1_d20_promotion_receipt.json",
            "receipt_sha256": promotion["sha256"],
            "selected_controllers": ["C0", "C5"],
            "source_id": "failed_repair_v1",
        }
        assert "complete_d128_receipt_sha256" not in provenance


def test_d20_promotion_gate_rejects_r3_and_false_or_mismatched_provenance(
        tmp_path: Path) -> None:
    sources = [_source(tmp_path, "C0"), _source(tmp_path, "C5")]
    complete_c0 = copy.deepcopy(sources[0])
    promotion = _d20_promotion(tmp_path, sources)
    for source in sources:
        del source["complete_d128_receipt"]

    with pytest.raises(ValueError, match="cannot authorize R3"):
        combinations.prepare(tmp_path / "r3_rejected", _spec(
            tmp_path, sources, r3={"source": complete_c0},
            d20_promotion=promotion))

    receipt_path = Path(promotion["path"])
    receipt = expansion._read(receipt_path)
    receipt["next_manifests"]["C5"]["full_task_ids"] = list(reversed(
        receipt["next_manifests"]["C5"]["full_task_ids"]))
    _save(receipt_path, receipt)
    bad_order = {"path": str(receipt_path), "sha256": _sha(receipt_path)}
    with pytest.raises(ValueError, match="task provenance differs"):
        combinations.prepare(tmp_path / "manifest_rejected", _spec(
            tmp_path, sources, d20_promotion=bad_order))

    receipt["next_manifests"]["C5"]["full_task_ids"] = list(
        expansion._read(tmp_path / "tasks.d128.json")["task_ids"])
    receipt["reuse_validation"]["target"]["source_id"] = "other_runtime"
    _save(receipt_path, receipt)
    wrong_runtime = {"path": str(receipt_path), "sha256": _sha(receipt_path)}
    with pytest.raises(ValueError, match="target differs"):
        combinations.prepare(tmp_path / "runtime_rejected", _spec(
            tmp_path, sources, d20_promotion=wrong_runtime))


def test_d20_promotion_gate_refuses_synthetic_complete_d128_claim(
        tmp_path: Path) -> None:
    sources = [_source(tmp_path, "C0"), _source(tmp_path, "C5")]
    promotion = _d20_promotion(tmp_path, sources)
    with pytest.raises(ValueError, match="must not claim complete D128"):
        combinations.prepare(tmp_path / "rejected", _spec(
            tmp_path, sources, d20_promotion=promotion))


def test_c1_h1_requires_production_verifier_and_keeps_h0_artifact(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c1 = _source(tmp_path, "C1")
    c0 = _source(tmp_path, "C0")
    with pytest.raises(ValueError, match="C1 H1 requires a calibration binding"):
        combinations.prepare(tmp_path / "rejected", _spec(tmp_path, [c1, c0]))
    assert not (tmp_path / "rejected").exists()
    calibration = _calibration(tmp_path)
    calls = []

    def verified(receipt_path, *, artifact_path, input_path, require_production):
        calls.append((Path(receipt_path), Path(artifact_path), Path(input_path),
                      require_production))
        return {"selected_threshold": 0.625, "production_eligible": True}

    monkeypatch.setattr(c1_calibration, "verify_calibration_receipt", verified)
    package = tmp_path / "accepted"
    combinations.prepare(package, _spec(tmp_path, [c1, c0],
                                         calibrations={"C1": calibration}))
    shard = package / "shards" / "h1_C1_part0"
    design = expansion._read(shard / "lanes/h1_C1_part0/design.json")
    provenance = expansion._read(shard / "provenance.json")
    gp = design["resolved_configs"]["controller"]["gp_experiments"]
    source_design = expansion._read(shard / "source_design.json")
    source_gp = source_design["resolved_configs"]["controller"]["gp_experiments"]
    assert gp["selector_artifact"] == source_gp["selector_artifact"]
    assert gp["selector_threshold"] == 0.625
    assert (design["search_contract"]["trained_artifact_sha256"]
            == source_design["search_contract"]["trained_artifact_sha256"])
    assert provenance["model_semantics"]["h1_trained"] is False
    assert provenance["model_semantics"]["h1_threshold_calibrated"] is True
    assert calls and all(call[3] is True for call in calls)
    assert all(_sha(call[1]) == c1["model_artifacts"][0]["sha256"] for call in calls)


def test_rejects_d128_receipt_with_noncompleted_quality_cell(tmp_path: Path) -> None:
    c0 = _source(tmp_path, "C0")
    receipt_binding = c0["complete_d128_receipt"]
    receipt_path = Path(receipt_binding["path"])
    receipt = expansion._read(receipt_path)
    receipt["quality_cells"][5]["runtime_completed"] = False
    _save(receipt_path, receipt)
    receipt_binding["sha256"] = _sha(receipt_path)
    with pytest.raises(ValueError, match="D128 quality cell is not completed"):
        combinations.prepare(tmp_path / "package", _spec(
            tmp_path, [c0, _source(tmp_path, "C3")]))


def test_r3_requires_explicit_leading_complete_receipt_and_changes_only_r(tmp_path: Path) -> None:
    leading = _source(tmp_path, "C3", history="record_bound")
    receipt_path = tmp_path / "leading.json"
    _save(receipt_path, {
        "schema": combinations.LEADING_SCHEMA,
        "status": "selected_after_complete_d128",
        "controller": "C3",
        "source_algorithm_id": leading["source_algorithm_id"],
        "complete_d128_receipt_sha256": leading["complete_d128_receipt"]["sha256"],
    })
    r3 = {"source": leading, "leading_receipt": {
        "path": str(receipt_path), "sha256": _sha(receipt_path)}}
    package = tmp_path / "r3_package"
    result = combinations.prepare(package, _spec(tmp_path, [], r3=r3))
    assert result["r3_task_execution_budget"] == 128
    assert result["shard_count"] == 6
    contract = expansion._read(package / "combination_contract.json")
    assert [row["task_budget"] for row in contract["shards"]] == [22, 22, 21, 21, 21, 21]
    assert [row["preferred_device"] for row in contract["shards"]] == [0, 1, 2, 3, 4, 6]
    assert {row["wave"] for row in contract["shards"]} == {0}
    assert [row["engine_port"] for row in contract["shards"]] == [
        27000, 27010, 27020, 27030, 27040, 27060]
    assert [row["task_port_base"] for row in contract["shards"]] == [
        28000, 28100, 28200, 28300, 28400, 28600]
    owned = []
    for part in range(6):
        shard_id = f"r3_leading_part{part}"
        shard = package / "shards" / shard_id
        owned.extend(expansion._read(shard / "tasks.json")["task_ids"])
        source_design = expansion._read(shard / "source_design.json")
        design = expansion._read(shard / "lanes" / shard_id / "design.json")
        source_gp = source_design["resolved_configs"]["controller"]["gp_experiments"]
        gp = design["resolved_configs"]["controller"]["gp_experiments"]
        assert gp == {**source_gp, "R": 3}
    assert len(owned) == len(set(owned)) == 128


def test_r3_accepts_only_exact_terminal_receipt_and_freezes_provenance(
        tmp_path: Path) -> None:
    leading = _source(tmp_path, "C0", history="record_bound")
    source_result = leading.pop("complete_d128_receipt")
    terminal_spec = tmp_path / "terminal_spec.json"
    _save(terminal_spec, {
        "schema": terminal_results.BUILD_SCHEMA,
        "source_result_receipt": source_result,
        "capacity_failure_audit": None,
        "continuation_overlays": [],
    })
    terminal_path = tmp_path / "C0.terminal_d128.json"
    _save(terminal_path, terminal_results.build_terminal_receipt(terminal_spec))
    leading["terminal_d128_receipt"] = {
        "path": str(terminal_path), "sha256": _sha(terminal_path)}
    receipt_path = tmp_path / "leading_terminal.json"
    _save(receipt_path, {
        "schema": combinations.TERMINAL_LEADING_SCHEMA,
        "status": "selected_after_terminal_d128",
        "controller": "C0",
        "source_algorithm_id": leading["source_algorithm_id"],
        "terminal_d128_receipt_sha256": _sha(terminal_path),
    })
    r3 = {"source": leading, "leading_receipt": {
        "path": str(receipt_path), "sha256": _sha(receipt_path)}}
    package = tmp_path / "r3_terminal_package"
    result = combinations.prepare(package, _spec(tmp_path, [], r3=r3))
    assert result["r3_task_execution_budget"] == 128
    contract = expansion._read(package / "combination_contract.json")
    assert contract["r3_source_gate"] == "terminal_d128"
    assert contract["r3_terminal_d128_receipt_sha256"] == _sha(terminal_path)
    assert _sha(package / "r3_terminal_d128_receipt.json") == _sha(terminal_path)
    assert _sha(package / "r3_leading_receipt.json") == _sha(receipt_path)
    for row in contract["shards"]:
        provenance = expansion._read(
            package / "shards" / row["shard_id"] / "provenance.json")
        assert provenance["source_selection_provenance"] == {
            "mode": "terminal_d128",
            "receipt": "r3_terminal_d128_receipt.json",
            "terminal_d128_receipt_sha256": _sha(terminal_path),
            "leading_receipt_sha256": _sha(receipt_path),
        }

    mixed = copy.deepcopy(leading)
    mixed["complete_d128_receipt"] = source_result
    with pytest.raises(ValueError, match="exactly one complete or terminal"):
        combinations.prepare(tmp_path / "mixed", _spec(
            tmp_path, [], r3={"source": mixed, "leading_receipt": r3[
                "leading_receipt"]}))


def test_shard_plan_has_no_reserved_devices_or_concurrent_overlap() -> None:
    rows = combinations.shard_plan(["C0", "C4_turn"], True)
    assert len(rows) == 12
    for stage, wave in {(row["stage"], row["wave"]) for row in rows}:
        active = [row for row in rows if row["stage"] == stage and row["wave"] == wave]
        assert len({row["preferred_device"] for row in active}) == len(active)
        assert not ({5, 7} & {row["preferred_device"] for row in active})
    assert sum(row["task_budget"] for row in rows if row["stage"] == "H1_R1") == 256
    assert sum(row["task_budget"] for row in rows if row["stage"] == "R3") == 128
    assert {row["wave"] for row in rows if row["stage"] == "H1_R1"} == {0}
    assert {row["wave"] for row in rows if row["stage"] == "R3"} == {1}
    assert combinations._ports("R3", 1, 6) == (27160, 29600)


def test_package_is_no_launch_standalone_frozen_and_no_overwrite(tmp_path: Path) -> None:
    package = tmp_path / "package"
    spec = _spec(tmp_path, [_source(tmp_path, "C0"), _source(tmp_path, "C4_turn")])
    combinations.prepare(package, spec)
    assert hasattr(combinations, "run_shard")
    with pytest.raises(FileExistsError):
        combinations.prepare(package, spec)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package)
    completed = subprocess.run(
        [sys.executable, str(package / "evidence_eval_combinations.py"),
         "verify-package", "--package", str(package)],
        cwd=tmp_path, env=env, text=True, capture_output=True, check=True)
    assert json.loads(completed.stdout)["status"] == "passed"
    authorization = subprocess.run(
        [sys.executable, "-I", str(package / "evidence_eval_combinations.py"),
         "authorization-requirements", "--package", str(package)],
        cwd=tmp_path, text=True, capture_output=True, check=True)
    requirements = json.loads(authorization.stdout)
    assert requirements["launch_authorized"] is False
    assert requirements["total_task_execution_budget"] == 256
    controller = package / "shards/h1_C0_part0/lanes/h1_C0_part0/runtime/configs/controller.json"
    controller.write_text(controller.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen combination package changed"):
        combinations.verify_package(package)


def test_declared_design_guard_rejects_model_or_budget_changes(tmp_path: Path) -> None:
    source_binding = _source(tmp_path, "C0")
    source_design = expansion._read(
        Path(source_binding["source_package"]) / "lanes/frozen_lane/design.json")
    target = copy.deepcopy(source_design)
    target["checkpoint_selection"]["step"] = 999
    with pytest.raises(ValueError, match="model/checkpoint"):
        combinations.assert_declared_design_changes(source_design, target, "H1_R1", "C0")
    target = copy.deepcopy(source_design)
    target["limits"]["history_tokens"] = 256
    with pytest.raises(ValueError, match="non-task limit"):
        combinations.assert_declared_design_changes(source_design, target, "H1_R1", "C0")


def test_run_requires_exact_authorization_and_refuses_existing_output(
        tmp_path: Path) -> None:
    package = tmp_path / "package"
    combinations.prepare(package, _spec(
        tmp_path, [_source(tmp_path, "C0"), _source(tmp_path, "C4_turn")]))
    bad = tmp_path / "bad-authorization.json"
    receipt = _authorization(package)
    receipt["total_task_execution_budget"] += 1
    _save(bad, receipt)
    with pytest.raises(ValueError, match="exact H1/R3 package"):
        combinations.verify_authorization(package, bad)
    lane = package / "shards/h1_C0_part0/lanes/h1_C0_part0"
    (lane / "results").mkdir()
    good = tmp_path / "authorization.json"
    _save(good, _authorization(package))
    with pytest.raises(FileExistsError, match="rerun"):
        combinations.run_shard(package, "h1_C0_part0", good)


def test_inner_design_passes_real_runner_authorization_seam(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "package"
    combinations.prepare(package, _spec(
        tmp_path, [_source(tmp_path, "C0"), _source(tmp_path, "C4_turn")]))
    design = expansion._read(
        package / "shards/h1_C0_part0/lanes/h1_C0_part0/design.json")
    assert combinations.verify_package(package)["status"] == "passed"
    assert combinations.authorization_requirements(package)["launch_authorized"] is False

    class ReachedModelPreflight(Exception):
        pass

    monkeypatch.setattr(real_runner, "validate", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(real_runner, "checkpoint_binding", lambda _design: {})

    def reached_preflight(_design, _checkpoint):
        raise ReachedModelPreflight

    monkeypatch.setattr(real_runner, "preflight_checkpoint", reached_preflight)
    with pytest.raises(ReachedModelPreflight):
        real_runner.run(design, SimpleNamespace(checkpoint="/checkpoint/frozen"))


def test_occupied_device_blocks_before_runner_without_writes_or_cleanup(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "package"
    combinations.prepare(package, _spec(
        tmp_path, [_source(tmp_path, "C0"), _source(tmp_path, "C4_turn")]))
    authorization = tmp_path / "authorization.json"
    _save(authorization, _authorization(package))
    calls = {"run": 0}

    def occupied(_lane):
        raise RuntimeError("Physical NPU 0 acquired by PIDs [999]")

    fake = SimpleNamespace(
        LANES={}, DEFAULT_REMOTE_ROOT=None, _assert_lane_free=occupied,
        run_lane=lambda *_: calls.__setitem__("run", calls["run"] + 1))
    monkeypatch.setattr(combinations, "verify_package", lambda _package: {
        "status": "passed"})
    monkeypatch.setattr(expansion, "_load_base_module", lambda _path: fake)
    with pytest.raises(RuntimeError, match="acquired by PIDs"):
        combinations.run_shard(package, "h1_C0_part0", authorization)
    assert calls["run"] == 0
    lane = package / "shards/h1_C0_part0/lanes/h1_C0_part0"
    assert not (lane / "run").exists()
    assert not (lane / "results").exists()


def test_authorized_shard_applies_env_and_frozen_lane_before_run(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "package"
    combinations.prepare(package, _spec(
        tmp_path, [_source(tmp_path, "C0"), _source(tmp_path, "C4_turn")]))
    authorization = tmp_path / "authorization.json"
    _save(authorization, _authorization(package))
    calls = []
    monkeypatch.setenv("C2KV_STRICT_NUMERIC", "fixture-before-run")

    def free(lane):
        calls.append(("free", lane["name"], len(lane["task_ports"])))
        return {"ports_bindable": True}

    def run(shard_package, shard_id):
        calls.append(("run", shard_id, str(shard_package)))
        assert os.environ["C2KV_STRICT_NUMERIC"] == "1"
        return 0

    fake = SimpleNamespace(
        LANES={}, DEFAULT_REMOTE_ROOT=None, _assert_lane_free=free, run_lane=run)
    monkeypatch.setattr(combinations, "verify_package", lambda _package: {
        "status": "passed"})
    monkeypatch.setattr(expansion, "_load_base_module", lambda _path: fake)
    assert combinations.run_shard(
        package, "h1_C4_turn_part0", authorization) == 0
    assert calls[0] == ("free", "h1_C4_turn_part0", 43)
    assert calls[1][0:2] == ("run", "h1_C4_turn_part0")
    assert list(fake.LANES) == ["h1_C4_turn_part0"]
    assert fake.DEFAULT_REMOTE_ROOT == package / "shards/h1_C4_turn_part0"
