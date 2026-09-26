"""Freeze CPU-only H1 and R3 Experiment 3 evaluation packages.

Preparation does not select controllers, calibrate C1, launch evaluators, or
run model smokes. It consumes explicit D128 result gates and calibration
receipts, then creates immutable round-robin shards. Running one shard
additionally requires an exact package-bound root authorization. H1 changes
only G=current to G=record_bound (plus an independently verified C1 H1
threshold); R3 changes only R=1 to R=3 on one explicitly selected strict-
complete or fixed-denominator terminal configuration.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import evidence_eval as local_base
import evidence_eval_expansion as expansion
import evidence_terminal_results as terminal_results


SPEC_SCHEMA = "experiment3-combination-build-v1"
PACKAGE_SCHEMA = "experiment3-h1-r3-package-v1"
SOURCE_SCHEMA = "experiment3-combination-source-v1"
COMPLETE_RESULT_SCHEMA = "experiment3-complete-d128-result-v1"
D20_PROMOTION_SCHEMA = "experiment3-expansion-readiness-v1"
LEADING_SCHEMA = "experiment3-leading-complete-config-v1"
TERMINAL_LEADING_SCHEMA = "experiment3-leading-terminal-config-v1"
AUTHORIZATION_SCHEMA = "experiment3-combination-launch-authorization-v1"
D20_PROMOTED_CONTROLLERS = ("C0", "C5")
DEVICES = (0, 1, 2, 3, 4, 6)
H1_SHARDS_PER_CONTROLLER = 3
R3_SHARD_COUNT = 6
H1_EXECUTION_BUDGET = 256
R3_EXECUTION_BUDGET = 128
STRICT_SAMPLING = {
    "mode": "greedy", "temperature": 0, "seed": 0,
    "max_completion_tokens": 4096,
}
CONTROLLER_SELECTORS = {
    "C0": {"candidate_rule"},
    "C1": {"risk"},
    "C2": {"reranker"},
    "C3": {"local_llm"},
    "C4": {"gain_turn", "gain_task"},
    "C4_turn": {"gain_turn"},
    "C4_task": {"gain_task"},
    "C5": {"parameter_source"},
}
G04_SOURCE_HASHES = {
    "python/history_memory/encoding_scope.py": (
        "81036aba03bbdc054e7c9189922ff3cd4c42fda6cff16469a302f81a6b3a63ce"
    ),
    "python/history_memory/packing.py": (
        "064e0757d1adfea5ccbb819e41873cd8e41ca5f4d3ab9cdcbb037f473b1faa2f"
    ),
}


def _read(path: Path) -> dict[str, Any]:
    return expansion._read(path)


def _save(path: Path, value: Mapping[str, Any]) -> None:
    expansion._save(path, value)


def _sha(path: Path) -> str:
    return expansion._sha(path)


def _bound_document(binding: Mapping[str, Any], key: str, schema: str) -> tuple[Path, dict[str, Any]]:
    value = binding.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Missing {key} binding")
    path_value = value.get("path")
    expected = value.get("sha256")
    if not isinstance(path_value, str) or not path_value or not expansion._valid_sha256(expected):
        raise ValueError(f"Invalid {key} binding")
    path = Path(path_value).resolve()
    if not path.is_file() or _sha(path) != expected:
        raise ValueError(f"{key} differs from its explicit hash")
    document = _read(path)
    if document.get("schema") != schema:
        raise ValueError(f"Unsupported {key} schema")
    return path, document


def validate_d128_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not expansion._valid_sha256(expected_sha256) or not path.is_file() or _sha(path) != expected_sha256:
        raise ValueError("D128 manifest differs from its explicit binding")
    value = _read(path)
    tasks = value.get("task_ids")
    if not isinstance(tasks, list) or len(tasks) != 128 or len(set(tasks)) != 128:
        raise ValueError("D128 manifest must contain 128 unique tasks")
    expansion.task_groups(tasks)
    return value


def _verify_complete_result(binding: Mapping[str, Any], controller: str,
                            algorithm_id: str, manifest: Mapping[str, Any],
                            manifest_sha256: str) -> dict[str, Any]:
    path, value = _bound_document(binding, "complete_d128_receipt", COMPLETE_RESULT_SCHEMA)
    config_binding = value.get("configuration_binding")
    evidence = value.get("evidence")
    cells = value.get("quality_cells")
    if (
        value.get("status") != "completed"
        or value.get("controller") != controller
        or value.get("source_algorithm_id") != algorithm_id
        or value.get("task_manifest_sha256") != manifest_sha256
        or value.get("completed_task_cells") != 128
        or not isinstance(config_binding, dict)
        or config_binding.get("status") not in {
            "exact_frozen_config", "audited_semantic_compatibility"}
        or not isinstance(evidence, list)
        or not evidence
        or any(not isinstance(row, dict) or not isinstance(row.get("path"), str)
               or not row["path"] or not expansion._valid_sha256(row.get("sha256"))
               for row in evidence)
        or not isinstance(cells, list)
        or len(cells) != 128
        or [cell.get("task_id") for cell in cells] != manifest["task_ids"]
    ):
        raise ValueError(f"{controller}: complete D128 receipt is not usable")
    for cell in cells:
        official = cell.get("official")
        cell_evidence = cell.get("evidence")
        if (
            cell.get("status") != "completed"
            or cell.get("runtime_completed") is not True
            or cell.get("worker_returncode") != 0
            or cell.get("server_returncode") != 0
            or not isinstance(cell.get("quality_source_id"), str)
            or not cell["quality_source_id"]
            or cell.get("cohort") not in {"historical20", "new108", "full128"}
            or not isinstance(official, dict)
            or official.get("scored") is not True
            or official.get("n_total") != 1
            or official.get("n_scored") != 1
            or type(official.get("correct_count")) is not int
            or official["correct_count"] not in {0, 1}
            or isinstance(official.get("semantic_score"), bool)
            or not isinstance(official.get("semantic_score"), (int, float))
            or not math.isfinite(float(official["semantic_score"]))
            or not isinstance(cell_evidence, list)
            or not cell_evidence
            or any(not isinstance(row, dict) or not isinstance(row.get("path"), str)
                   or not row["path"]
                   or not expansion._valid_sha256(row.get("sha256"))
                   for row in cell_evidence)
        ):
            raise ValueError(f"{controller}: D128 quality cell is not completed")
    return {"path": path, "document": value, "sha256": _sha(path)}


def _verify_terminal_result(binding: Mapping[str, Any], controller: str,
                            algorithm_id: str, manifest: Mapping[str, Any],
                            manifest_sha256: str) -> dict[str, Any]:
    raw = binding.get("terminal_d128_receipt")
    if not isinstance(raw, dict):
        raise ValueError(f"{controller}: missing terminal D128 receipt binding")
    path_value = raw.get("path")
    expected_sha256 = raw.get("sha256")
    if (not isinstance(path_value, str) or not path_value
            or not expansion._valid_sha256(expected_sha256)):
        raise ValueError(f"{controller}: invalid terminal D128 receipt binding")
    result = terminal_results.verify_terminal_receipt(
        Path(path_value), expected_sha256=expected_sha256,
        expected_controller=controller, expected_algorithm_id=algorithm_id,
        expected_manifest_sha256=manifest_sha256,
        expected_task_ids=manifest["task_ids"], verify_inputs=True,
    )
    value = result["document"]
    if value.get("status") != "terminal" or value.get("selection_eligible") is not True:
        raise ValueError(f"{controller}: terminal D128 receipt is not selection eligible")
    return result


def _verify_config_source(binding: Mapping[str, Any], runtime: Path) -> dict[str, Any]:
    root_value = binding.get("config_source_package")
    expected = binding.get("source_evidence_sets_sha256")
    if not isinstance(root_value, str) or not root_value or not expansion._valid_sha256(expected):
        raise ValueError("Source binding lacks frozen evidence_sets.py identity")
    root = Path(root_value).resolve()
    evidence_sets = root / "history_system/evidence_sets.py"
    if not evidence_sets.is_file() or _sha(evidence_sets) != expected:
        raise ValueError("Frozen evidence_sets.py differs from source binding")
    observed = {}
    for relative, expected_hash in G04_SOURCE_HASHES.items():
        runtime_path = runtime / relative
        config_path = root / "history_system/runtime" / relative
        if (not runtime_path.is_file() or not config_path.is_file()
                or _sha(runtime_path) != expected_hash
                or _sha(config_path) != expected_hash):
            raise ValueError(f"Frozen G04 source hash differs: {relative}")
        observed[relative] = expected_hash
    return {"root": root, "evidence_sets": evidence_sets,
            "evidence_sets_sha256": expected, "g04_source_hashes": observed}


def _verify_source(binding: Mapping[str, Any], manifest: Mapping[str, Any],
                   manifest_sha256: str, *, result_gate: str = "complete_d128") -> dict[str, Any]:
    if binding.get("schema") != SOURCE_SCHEMA:
        raise ValueError("Unknown combination source schema")
    controller = binding.get("controller")
    source_lane = binding.get("source_lane")
    source_value = binding.get("source_package")
    if not all(isinstance(value, str) and value for value in
               (controller, source_lane, source_value, binding.get("source_algorithm_id"))):
        raise ValueError("Combination source identity is incomplete")
    source = Path(source_value).resolve()
    expansion._source_manifest_hash(
        source, "static_files.json", binding.get("source_static_files_sha256"))
    sglang = expansion._source_manifest_hash(
        source, "sglang_files.json", binding.get("source_sglang_files_sha256"))
    required = (source / "evidence_eval.py", source / "history_system/runner.py",
                source / "launch_contract.json", source / "sglang",
                source / "lanes" / source_lane / "runtime",
                source / "lanes" / source_lane / "design.json",
                source / "lanes" / source_lane / "lane.json")
    if any(not path.exists() for path in required):
        raise FileNotFoundError(f"{controller}: incomplete frozen source package")
    design_path = source / "lanes" / source_lane / "design.json"
    runtime = source / "lanes" / source_lane / "runtime"
    controller_path = runtime / "configs/controller.json"
    design = _read(design_path)
    resolved = design.get("resolved_configs", {}).get("controller")
    controller_config = _read(controller_path)
    gp = controller_config.get("gp_experiments")
    if (
        controller not in CONTROLLER_SELECTORS
        or design.get("candidate_id") != binding.get("source_algorithm_id")
        or binding.get("source_design_sha256") != _sha(design_path)
        or binding.get("source_controller_sha256") != _sha(controller_path)
        or resolved != controller_config
        or design.get("source_files") != expansion._tree_hashes(runtime)
        or not isinstance(gp, dict)
        or gp.get("set_selector") not in CONTROLLER_SELECTORS.get(controller, set())
        or gp.get("R") != 1
        or gp.get("G") not in {"current", "record_bound"}
        or design.get("sampling") != STRICT_SAMPLING
        or design.get("automatic_reruns") != 0
    ):
        raise ValueError(f"{controller}: source design/controller binding differs")
    required_env = binding.get("required_env", {})
    if (not isinstance(required_env, dict)
            or required_env != _read(source / "launch_contract.json").get(
                "required_environment", {})):
        raise ValueError(f"{controller}: required environment differs from source")
    artifacts = expansion._verify_artifacts(source, design, binding)
    has_complete = "complete_d128_receipt" in binding
    has_terminal = "terminal_d128_receipt" in binding
    if result_gate == "complete_d128":
        if not has_complete or has_terminal:
            raise ValueError(f"{controller}: source requires only a complete D128 result")
        complete = _verify_complete_result(
            binding, controller, design["candidate_id"], manifest, manifest_sha256)
        terminal = None
    elif result_gate == "terminal_d128":
        if not has_terminal or has_complete:
            raise ValueError(f"{controller}: source requires only a terminal D128 result")
        complete = None
        terminal = _verify_terminal_result(
            binding, controller, design["candidate_id"], manifest, manifest_sha256)
    elif result_gate == "none":
        if has_complete or has_terminal:
            raise ValueError(f"{controller}: ungated source must not claim a D128 result")
        complete = None
        terminal = None
    else:
        raise ValueError(f"Unknown source result gate: {result_gate}")
    config_source = _verify_config_source(binding, runtime)
    return {"binding": dict(binding), "controller": controller, "source": source,
            "source_lane": source_lane, "design": design, "runtime": runtime,
            "controller_config": controller_config, "gp": gp, "artifacts": artifacts,
            "sglang_manifest": sglang, "complete_result": complete,
            "terminal_result": terminal, "result_gate": result_gate,
            "config_source": config_source}


def _verify_d20_promotion(
        spec: Mapping[str, Any], manifest: Mapping[str, Any],
        source_bindings: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    binding = spec.get("h1_d20_promotion_receipt")
    if binding is None:
        return None
    if not isinstance(binding, dict):
        raise ValueError("h1_d20_promotion_receipt must be an exact document binding")
    if spec.get("r3") is not None:
        raise ValueError("D20 promotion provenance is H1-only and cannot authorize R3")
    if len(source_bindings) != 2:
        raise ValueError("D20 promotion provenance requires exactly two H1 sources")
    if any("complete_d128_receipt" in source for source in source_bindings):
        raise ValueError("D20 promotion H1 sources must not claim complete D128 results")
    path, receipt = _bound_document(
        spec, "h1_d20_promotion_receipt", D20_PROMOTION_SCHEMA)
    controllers = [source.get("controller") for source in source_bindings]
    promotion = receipt.get("promotion")
    if (controllers != list(D20_PROMOTED_CONTROLLERS)
            or receipt.get("phase") != "promotion_ready"
            or receipt.get("launch_authorized") is not False
            or not isinstance(promotion, dict)
            or promotion.get("phase") != "promotion_ready"
            or promotion.get("selected") != controllers):
        raise ValueError("D20 promotion receipt does not select exact C0/C5 H1 sources")
    lanes = receipt.get("lanes")
    next_manifests = receipt.get("next_manifests")
    if not isinstance(lanes, dict) or not isinstance(next_manifests, dict):
        raise ValueError("D20 promotion receipt lacks lane or D128 manifest evidence")
    d128_tasks = manifest["task_ids"]
    for controller in controllers:
        lane = lanes.get(controller)
        next_manifest = next_manifests.get(controller)
        if not isinstance(lane, dict) or not isinstance(next_manifest, dict):
            raise ValueError(f"{controller}: D20 promotion provenance is incomplete")
        cells = lane.get("quality_cells")
        if (lane.get("completed") != 20
                or lane.get("clean_d20") is not True
                or lane.get("all_cells_terminal") is not True
                or lane.get("operational_selection_eligible") is not True
                or lane.get("operational_denominator") != 20
                or not isinstance(cells, list) or len(cells) != 20
                or len({cell.get("task_id") for cell in cells}) != 20
                or any(not isinstance(cell, dict) or cell.get("status") != "completed"
                       or not isinstance(cell.get("official"), dict)
                       or cell["official"].get("scored") is not True
                       or cell["official"].get("n_total") != 1
                       or cell["official"].get("n_scored") != 1
                       or cell["official"].get("correct_count") not in {0, 1}
                       for cell in cells)):
            raise ValueError(f"{controller}: selected D20 lane is not clean terminal D20")
        new_tasks = next_manifest.get("task_ids")
        reused_tasks = next_manifest.get("reused_task_ids")
        full_tasks = next_manifest.get("full_task_ids")
        if (not isinstance(new_tasks, list) or len(new_tasks) != 108
                or not isinstance(reused_tasks, list) or len(reused_tasks) != 20
                or len(set(new_tasks)) != 108 or len(set(reused_tasks)) != 20
                or set(new_tasks) & set(reused_tasks)
                or full_tasks != d128_tasks
                or set(new_tasks) | set(reused_tasks) != set(d128_tasks)):
            raise ValueError(f"{controller}: promoted D20/D128 task provenance differs")
    reuse = receipt.get("reuse_validation")
    target = reuse.get("target") if isinstance(reuse, dict) else None
    target_source_id = target.get("source_id") if isinstance(target, dict) else None
    audited = reuse.get("audited_lanes") if isinstance(reuse, dict) else None
    if (not isinstance(reuse, dict)
            or reuse.get("status") != "audited_historical_lanes_bound_to_target"
            or not isinstance(audited, list)
            or any(controller not in audited for controller in controllers)
            or not isinstance(target_source_id, str) or not target_source_id
            or any(source.get("source_id") != target_source_id
                   for source in source_bindings)):
        raise ValueError("D20 promotion target differs from frozen H1 source bindings")
    return {"path": path, "sha256": _sha(path), "controllers": controllers,
            "source_id": target_source_id}


def _verify_c1_calibration(binding: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    module = importlib.import_module("evidence_c1_h1_calibration")
    if Path(module.__file__).resolve() != (
            PACKAGE_DIR / "evidence_c1_h1_calibration.py").resolve():
        raise RuntimeError("Loaded C1 H1 calibration verifier from an unexpected path")
    receipt_path, _ = _bound_document(binding, "receipt", module.RECEIPT_SCHEMA)
    input_path, _ = _bound_document(binding, "input", module.DATASET_SCHEMA)
    configured = source["gp"].get("selector_artifact")
    matches = [row for row in source["artifacts"]
               if _read(row["source_path"]) == configured]
    if len(matches) != 1:
        raise ValueError("C1 source must bind its one exact frozen H0 artifact")
    artifact_row = matches[0]
    observed = module.verify_calibration_receipt(
        receipt_path, artifact_path=artifact_row["source_path"],
        input_path=input_path, require_production=True)
    threshold = observed.get("selected_threshold")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))):
        raise ValueError("Verified C1 H1 calibration has no finite selected_threshold")
    return {"path": receipt_path, "receipt": observed,
            "input_path": input_path, "input_sha256": _sha(input_path),
            "artifact_path": artifact_row["source_path"],
            "artifact": configured, "artifact_sha256": artifact_row["sha256"],
            "threshold": float(threshold)}


def _same_except(source: Mapping[str, Any], target: Mapping[str, Any], keys: set[str]) -> bool:
    return {key: value for key, value in source.items() if key not in keys} == {
        key: value for key, value in target.items() if key not in keys}


def assert_declared_design_changes(source: Mapping[str, Any], target: Mapping[str, Any],
                                   stage: str, controller: str) -> None:
    """Reject changes outside task plumbing and the declared G or R change."""

    allowed_top = {
        "candidate_id", "run_id_template", "launch_authorized", "task_ids",
        "task_manifest_sha256", "limits", "runtime", "resolved_configs",
        "search_contract", "source_files",
    }
    if not _same_except(source, target, allowed_top):
        raise ValueError("Combination design changed frozen model/checkpoint semantics")
    if not _same_except(source["limits"], target["limits"], {"tasks"}):
        raise ValueError("Combination design changed a non-task limit")
    if not _same_except(
            source["runtime"], target["runtime"], {"sglang_backend_url"}):
        raise ValueError("Combination design changed frozen runtime semantics")
    if not _same_except(source["resolved_configs"], target["resolved_configs"],
                        {"controller"}):
        raise ValueError("Combination design changed a non-controller config")
    search_allowed = {"fixed_denominator", "history"} if stage == "H1_R1" else {
        "fixed_denominator", "recovery_attempts"}
    if not _same_except(source["search_contract"], target["search_contract"],
                        search_allowed):
        raise ValueError("Combination design changed undeclared search semantics")
    if not _same_except(source["source_files"], target["source_files"],
                        {"configs/controller.json"}):
        raise ValueError("Combination runtime changed outside controller.json")


def build_controller(source: Mapping[str, Any], *, stage: str,
                     c1_calibration: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    controller = copy.deepcopy(source["controller_config"])
    original_gp = source["gp"]
    gp = copy.deepcopy(original_gp)
    metadata: dict[str, Any] = {}
    if stage == "H1_R1":
        if original_gp.get("G") != "current" or original_gp.get("R") != 1:
            raise ValueError("H1 source must be an H0/R1 configuration")
        gp["G"] = "record_bound"
        allowed = {"G"}
        if source["controller"] == "C1":
            if c1_calibration is None:
                raise ValueError("C1 H1 requires independent calibration")
            gp["selector_threshold"] = c1_calibration["threshold"]
            allowed.add("selector_threshold")
            metadata.update({"artifact_training_history": "H0",
                             "h1_trained": False,
                             "h1_threshold_calibrated": True,
                             "h1_calibration_receipt_sha256": _sha(c1_calibration["path"]),
                             "h1_calibration_input_sha256": c1_calibration["input_sha256"]})
        elif source["controller"] in {"C4_turn", "C4_task", "C4"}:
            if gp.get("set_selector") not in {"gain_turn", "gain_task"} or gp.get("gain_delta") != 0.0:
                raise ValueError("C4 H1 migration requires the frozen gain artifact and delta=0")
            metadata.update({"artifact_training_history": "H0", "h1_trained": False,
                             "artifact_migration": "fixed_H0_artifact_delta0"})
        if not _same_except(original_gp, gp, allowed):
            raise ValueError("H1 changed fields outside its declared contract")
    elif stage == "R3":
        if original_gp.get("R") != 1:
            raise ValueError("R3 source must be a complete R1 configuration")
        gp["R"] = 3
        if not _same_except(original_gp, gp, {"R"}):
            raise ValueError("R3 changed fields outside recovery rounds")
    else:
        raise ValueError(f"Unknown combination stage: {stage}")
    controller["gp_experiments"] = gp
    return controller, metadata


def shard_plan(h1_controllers: Sequence[str], include_r3: bool, *,
               task_count: int = R3_EXECUTION_BUDGET) -> list[dict[str, Any]]:
    rows = []
    device_groups = (DEVICES[:3], DEVICES[3:])
    for rank, controller in enumerate(h1_controllers):
        for part, device in enumerate(device_groups[rank]):
            rows.append({"stage": "H1_R1", "controller": controller,
                         "shard_id": f"h1_{controller}_part{part}", "part": part,
                         "wave": 0, "preferred_device": device,
                         "allowed_devices": list(DEVICES),
                         "task_budget": len(range(part, task_count,
                                                  H1_SHARDS_PER_CONTROLLER))})
    if include_r3:
        r3_wave = 1 if h1_controllers else 0
        for part, device in enumerate(DEVICES):
            rows.append({"stage": "R3", "controller": None,
                         "shard_id": f"r3_leading_part{part}", "part": part,
                         "wave": r3_wave, "preferred_device": device,
                         "allowed_devices": list(DEVICES),
                         "task_budget": len(range(part, task_count,
                                                  R3_SHARD_COUNT))})
    return rows


def _ports(stage: str, wave: int, device: int) -> tuple[int, int]:
    offset = 0 if stage == "H1_R1" else 4000
    return 23000 + offset + wave * 100 + device * 10, 24000 + offset + wave * 1000 + device * 100


def _build_design(source: Mapping[str, Any], controller: Mapping[str, Any], *,
                  stage: str, shard_id: str, tasks: Sequence[str], task_sha256: str,
                  engine_port: int) -> dict[str, Any]:
    design = copy.deepcopy(source["design"])
    design["candidate_id"] = source["design"]["candidate_id"] + (
        "__h1_r1" if stage == "H1_R1" else "__r3")
    design["run_id_template"] = shard_id
    design["launch_authorized"] = True
    design["task_ids"] = list(tasks)
    design["task_manifest_sha256"] = task_sha256
    design["limits"] = {**design["limits"], "tasks": len(tasks)}
    design["runtime"] = {**design["runtime"],
                         "sglang_backend_url": f"http://127.0.0.1:{engine_port}"}
    design["resolved_configs"] = {**design["resolved_configs"], "controller": controller}
    search = copy.deepcopy(design["search_contract"])
    search["fixed_denominator"] = shard_id
    if stage == "H1_R1":
        search["history"] = "H1"
    else:
        search["recovery_attempts"] = 3
    design["search_contract"] = search
    return design


def _write_shard(root: Path, plan: Mapping[str, Any], source: Mapping[str, Any],
                 d128: Mapping[str, Any], controller: Mapping[str, Any],
                 stage_metadata: Mapping[str, Any], calibration: Mapping[str, Any] | None) -> dict[str, Any]:
    shard_id = plan["shard_id"]
    controller_name = source["controller"]
    shard_count = (H1_SHARDS_PER_CONTROLLER
                   if plan["stage"] == "H1_R1" else R3_SHARD_COUNT)
    tasks = d128["task_ids"][plan["part"]::shard_count]
    task_budget = len(tasks)
    if task_budget != plan["task_budget"]:
        raise ValueError("Combination shard task budget differs from D128 partition")
    lane_root = root / "lanes" / shard_id
    lane_root.mkdir(parents=True)
    (root / "history_system").mkdir()
    shutil.copyfile(source["source"] / "evidence_eval.py", root / "evidence_eval.py")
    shutil.copyfile(source["source"] / "history_system/runner.py", root / "history_system/runner.py")
    shutil.copyfile(source["config_source"]["evidence_sets"], root / "history_system/evidence_sets.py")
    expansion._copy_tree(source["source"] / "sglang", root / "sglang")
    expansion._copy_tree(source["runtime"], lane_root / "runtime")
    _save(lane_root / "runtime/configs/controller.json", controller)
    manifest = {"schema": "a-history-system-task-manifest-v1",
                "manifest_id": shard_id, "stage": "development_search",
                "task_ids": tasks, "parent_task_count": 128}
    _save(root / "tasks.json", manifest)
    engine_port, task_port_base = _ports(plan["stage"], plan["wave"], plan["preferred_device"])
    design = _build_design(source, controller, stage=plan["stage"], shard_id=shard_id,
                           tasks=tasks, task_sha256=_sha(root / "tasks.json"),
                           engine_port=engine_port)
    design["source_files"] = expansion._tree_hashes(lane_root / "runtime")
    _save(lane_root / "design.json", design)
    source_lane = _read(source["source"] / "lanes" / source["source_lane"] / "lane.json")
    lane = copy.deepcopy(source_lane)
    lane.update({"name": shard_id, "physical_device": plan["preferred_device"],
                 "engine_port": engine_port, "task_port_base": task_port_base,
                 "task_ports": list(range(task_port_base, task_port_base + task_budget))})
    _save(lane_root / "lane.json", lane)
    copied_artifacts = []
    artifact_rows = source["artifacts"]
    for row in artifact_rows:
        target = lane_root / "model_artifacts" / row["relative"].name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["source_path"], target)
        copied_artifacts.append({"path": target.relative_to(root).as_posix(),
                                 "sha256": row["sha256"]})
    calibration_evidence = None
    if calibration is not None:
        calibration_root = root / "calibration"
        calibration_root.mkdir()
        shutil.copyfile(calibration["path"], calibration_root / "receipt.json")
        shutil.copyfile(calibration["input_path"], calibration_root / "input.json")
        calibration_evidence = {
            "receipt": {"path": "calibration/receipt.json",
                        "sha256": _sha(calibration_root / "receipt.json")},
            "input": {"path": "calibration/input.json",
                      "sha256": _sha(calibration_root / "input.json")},
        }
    provenance = {"schema": "experiment3-combination-shard-provenance-v1",
                  "stage": plan["stage"], "controller": controller_name,
                  "source_algorithm_id": source["design"]["candidate_id"],
                  "output_algorithm_id": design["candidate_id"],
                  "source_selection_provenance": copy.deepcopy(
                      source["selection_provenance"]),
                  "source_design_sha256": source["binding"]["source_design_sha256"],
                  "source_controller_sha256": source["binding"]["source_controller_sha256"],
                  "source_evidence_sets_sha256": source["config_source"]["evidence_sets_sha256"],
                  "g04_source_hashes": source["config_source"]["g04_source_hashes"],
                  "required_env": source["binding"].get("required_env", {}),
                  "model_semantics": dict(stage_metadata),
                  "copied_model_artifacts": copied_artifacts,
                  "calibration_evidence": calibration_evidence,
                  "launch_authorized": False}
    _save(root / "provenance.json", provenance)
    shutil.copyfile(
        source["source"] / "lanes" / source["source_lane"] / "design.json",
        root / "source_design.json")
    _save(root / "sglang_files.json", source["sglang_manifest"])
    _save(root / "launch_contract.json", {
        "schema": "experiment3-combination-shard-launch-v1",
        "launch_authorized": True,
        "external_combination_authorization_required": True,
        "task_budget": task_budget,
        "automatic_retries": 0, "automatic_reruns": 0,
        "required_environment": source["binding"].get("required_env", {}),
        "no_model_smoke_during_preparation": True})
    _save(root / "static_files.json", expansion._manifest(
        root, "experiment3-combination-shard-static-v1", skip_sglang=True,
        excluded=("static_files.json",)))
    return {"stage": plan["stage"], "controller": controller_name,
            "shard_id": shard_id, "relative_package": root.name,
            "wave": plan["wave"], "preferred_device": plan["preferred_device"],
            "allowed_devices": list(DEVICES), "engine_port": engine_port,
            "task_port_base": task_port_base, "task_budget": task_budget,
            "static_files_sha256": _sha(root / "static_files.json")}


def _leading_source(spec: Mapping[str, Any], manifest: Mapping[str, Any],
                    manifest_sha256: str) -> dict[str, Any] | None:
    r3 = spec.get("r3")
    if r3 is None:
        return None
    if not isinstance(r3, dict) or not isinstance(r3.get("source"), dict):
        raise ValueError("R3 requires an explicit source")
    source_binding = r3["source"]
    has_complete = "complete_d128_receipt" in source_binding
    has_terminal = "terminal_d128_receipt" in source_binding
    if has_complete == has_terminal:
        raise ValueError("R3 source requires exactly one complete or terminal D128 result")
    if has_terminal:
        source = _verify_source(
            source_binding, manifest, manifest_sha256, result_gate="terminal_d128")
        path, receipt = _bound_document(
            r3, "leading_receipt", TERMINAL_LEADING_SCHEMA)
        if (receipt.get("status") != "selected_after_terminal_d128"
                or receipt.get("controller") != source["controller"]
                or receipt.get("source_algorithm_id")
                != source["design"]["candidate_id"]
                or receipt.get("terminal_d128_receipt_sha256")
                != source["terminal_result"]["sha256"]):
            raise ValueError("R3 leading receipt does not select this terminal configuration")
    else:
        source = _verify_source(
            source_binding, manifest, manifest_sha256, result_gate="complete_d128")
        path, receipt = _bound_document(r3, "leading_receipt", LEADING_SCHEMA)
        if (receipt.get("status") != "selected_after_complete_d128"
                or receipt.get("controller") != source["controller"]
                or receipt.get("source_algorithm_id")
                != source["design"]["candidate_id"]
                or receipt.get("complete_d128_receipt_sha256")
                != source["complete_result"]["sha256"]):
            raise ValueError("R3 leading receipt does not select this complete configuration")
    source["leading_receipt"] = {"path": path, "sha256": _sha(path)}
    return source


def prepare(package: Path, spec_path: Path) -> dict[str, Any]:
    package = package.resolve()
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite combination package: {package}")
    spec = _read(spec_path)
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError("Unknown H1/R3 build spec")
    d128_binding = spec.get("d128_manifest")
    if not isinstance(d128_binding, dict):
        raise ValueError("Build spec requires d128_manifest")
    d128_path = Path(d128_binding.get("path", "")).resolve()
    d128_sha256 = d128_binding.get("sha256")
    d128 = validate_d128_manifest(d128_path, d128_sha256)
    h1_values = spec.get("h1_sources", [])
    if not isinstance(h1_values, list) or len(h1_values) not in {0, 2}:
        raise ValueError("H1 requires exactly two explicit promoted sources")
    d20_promotion = _verify_d20_promotion(spec, d128, h1_values)
    h1_sources = [
        _verify_source(value, d128, d128_sha256,
                       result_gate=("complete_d128" if d20_promotion is None else "none"))
        for value in h1_values
    ]
    if len({row["controller"] for row in h1_sources}) != len(h1_sources):
        raise ValueError("H1 promoted sources must be distinct controllers")
    for source in h1_sources:
        if d20_promotion is None:
            source["selection_provenance"] = {
                "mode": "complete_d128",
                "complete_d128_receipt_sha256": source["complete_result"]["sha256"],
            }
        else:
            source["selection_provenance"] = {
                "mode": "d20_promotion",
                "receipt": "h1_d20_promotion_receipt.json",
                "receipt_sha256": d20_promotion["sha256"],
                "selected_controllers": list(d20_promotion["controllers"]),
                "source_id": d20_promotion["source_id"],
            }
    leading = _leading_source(spec, d128, d128_sha256)
    if leading is not None:
        if leading["result_gate"] == "terminal_d128":
            leading["selection_provenance"] = {
                "mode": "terminal_d128",
                "receipt": "r3_terminal_d128_receipt.json",
                "terminal_d128_receipt_sha256": leading["terminal_result"]["sha256"],
                "leading_receipt_sha256": leading["leading_receipt"]["sha256"],
            }
        else:
            leading["selection_provenance"] = {
                "mode": "complete_d128",
                "complete_d128_receipt_sha256": leading["complete_result"]["sha256"],
                "leading_receipt_sha256": leading["leading_receipt"]["sha256"],
            }
    if not h1_sources and leading is None:
        raise ValueError("Build spec contains neither H1 nor R3 work")
    plan = shard_plan([row["controller"] for row in h1_sources], leading is not None,
                      task_count=len(d128["task_ids"]))
    package.mkdir(parents=True)
    try:
        shutil.copyfile(Path(__file__), package / Path(__file__).name)
        shutil.copyfile(Path(expansion.__file__), package / Path(expansion.__file__).name)
        shutil.copyfile(
            Path(terminal_results.__file__), package / Path(terminal_results.__file__).name)
        shutil.copyfile(Path(local_base.__file__), package / "evidence_eval.py")
        shutil.copyfile(spec_path, package / "build_spec.json")
        shutil.copyfile(d128_path, package / "tasks.d128.json")
        if d20_promotion is not None:
            shutil.copyfile(
                d20_promotion["path"], package / "h1_d20_promotion_receipt.json")
        if leading is not None and leading["result_gate"] == "terminal_d128":
            shutil.copyfile(
                leading["terminal_result"]["path"],
                package / "r3_terminal_d128_receipt.json")
            shutil.copyfile(
                leading["leading_receipt"]["path"],
                package / "r3_leading_receipt.json")
        rows = []
        source_by_controller = {row["controller"]: row for row in h1_sources}
        h1_calibrations = spec.get("c1_h1_calibrations", {})
        if not isinstance(h1_calibrations, dict):
            raise ValueError("c1_h1_calibrations must be a mapping")
        for shard in plan:
            source = (leading if shard["stage"] == "R3"
                      else source_by_controller[shard["controller"]])
            calibration = None
            if shard["stage"] == "H1_R1" and source["controller"] == "C1":
                binding = h1_calibrations.get("C1")
                if not isinstance(binding, dict):
                    raise ValueError("C1 H1 requires a calibration binding")
                calibration = _verify_c1_calibration(binding, source)
            controller, metadata = build_controller(
                source, stage=shard["stage"], c1_calibration=calibration)
            rows.append(_write_shard(package / "shards" / shard["shard_id"], shard,
                                     source, d128, controller, metadata, calibration))
        contract = {"schema": PACKAGE_SCHEMA,
                    "status": "prepared_not_launch_authorized",
                    "launch_authorized": False, "d128_manifest_sha256": d128_sha256,
                    "h1_controllers": [row["controller"] for row in h1_sources],
                    "h1_source_gate": (
                        "d20_promotion" if d20_promotion is not None
                        else "complete_d128" if h1_sources else "none"),
                    "h1_d20_promotion_receipt_sha256": (
                        d20_promotion["sha256"] if d20_promotion is not None
                        else None),
                    "h1_task_execution_budget": (
                        H1_EXECUTION_BUDGET if h1_sources else 0),
                     "r3_task_execution_budget": (
                         R3_EXECUTION_BUDGET if leading is not None else 0),
                     "r3_source_gate": (
                         leading["result_gate"] if leading is not None else "none"),
                     "r3_terminal_d128_receipt_sha256": (
                         leading["terminal_result"]["sha256"]
                         if leading is not None
                         and leading["result_gate"] == "terminal_d128" else None),
                     "r3_leading_receipt_sha256": (
                         leading["leading_receipt"]["sha256"]
                         if leading is not None
                         and leading["result_gate"] == "terminal_d128" else None),
                     "automatic_retries": 0, "automatic_reruns": 0,
                    "shards": rows}
        _save(package / "combination_contract.json", contract)
        _save(package / "static_files.json", expansion._manifest(
            package, "experiment3-combination-static-v1",
            excluded=("static_files.json",)))
        return verify_package(package)
    except BaseException:
        if package.exists():
            shutil.rmtree(package)
        raise


def _verify_shard(root: Path, row: Mapping[str, Any], d128: Mapping[str, Any]) -> None:
    base = expansion._load_base_module(root / "evidence_eval.py")
    base.verify_package(root)
    shard_id = row["shard_id"]
    lane_root = root / "lanes" / shard_id
    lane = _read(lane_root / "lane.json")
    design = _read(lane_root / "design.json")
    source_design = _read(root / "source_design.json")
    controller = _read(lane_root / "runtime/configs/controller.json")
    source_controller = source_design["resolved_configs"]["controller"]
    gp = controller["gp_experiments"]
    source_gp = source_controller["gp_experiments"]
    tasks = _read(root / "tasks.json")["task_ids"]
    shard_count = (H1_SHARDS_PER_CONTROLLER
                   if row["stage"] == "H1_R1" else R3_SHARD_COUNT)
    expected_tasks = d128["task_ids"][int(shard_id.rsplit("part", 1)[1])::shard_count]
    task_budget = len(tasks)
    if (tasks != expected_tasks
            or task_budget != row["task_budget"]
            or design["task_ids"] != tasks
            or design["task_manifest_sha256"] != _sha(root / "tasks.json")
            or design["limits"]["tasks"] != task_budget
            or design["resolved_configs"]["controller"] != controller
            or lane["physical_device"] != row["preferred_device"]
            or lane["engine_port"] != row["engine_port"]
            or design.get("runtime", {}).get("sglang_backend_url")
            != f"http://127.0.0.1:{lane.get('engine_port')}"
            or lane["task_port_base"] != row["task_port_base"]
            or set(row["allowed_devices"]) != set(DEVICES)
            or 5 in row["allowed_devices"] or 7 in row["allowed_devices"]
            or lane["task_ports"] != list(range(row["task_port_base"],
                                                row["task_port_base"] + task_budget))
            or design.get("launch_authorized") is not True):
        raise ValueError(f"Invalid combination shard: {shard_id}")
    provenance = _read(root / "provenance.json")
    launch = _read(root / "launch_contract.json")
    if (_sha(root / "source_design.json")
            != provenance.get("source_design_sha256")
            or source_design.get("sampling") != STRICT_SAMPLING):
        raise ValueError("Combination shard source design binding differs")
    selection = provenance.get("source_selection_provenance")
    if not isinstance(selection, dict):
        raise ValueError("Combination shard lacks source selection provenance")
    if selection.get("mode") == "complete_d128":
        if not expansion._valid_sha256(
                selection.get("complete_d128_receipt_sha256")):
            raise ValueError("Combination shard complete-D128 provenance differs")
    elif selection.get("mode") == "terminal_d128":
        if (row["stage"] != "R3"
                or selection.get("receipt") != "r3_terminal_d128_receipt.json"
                or not expansion._valid_sha256(
                    selection.get("terminal_d128_receipt_sha256"))
                or not expansion._valid_sha256(
                    selection.get("leading_receipt_sha256"))):
            raise ValueError("Combination shard terminal-D128 provenance differs")
    elif selection.get("mode") == "d20_promotion":
        if (row["stage"] != "H1_R1"
                or selection.get("receipt") != "h1_d20_promotion_receipt.json"
                or not expansion._valid_sha256(selection.get("receipt_sha256"))
                or selection.get("selected_controllers")
                != list(D20_PROMOTED_CONTROLLERS)
                or row["controller"] not in selection["selected_controllers"]
                or not isinstance(selection.get("source_id"), str)
                or not selection["source_id"]):
            raise ValueError("Combination shard D20 promotion provenance differs")
    else:
        raise ValueError("Combination shard source selection mode is unknown")
    assert_declared_design_changes(source_design, design, row["stage"],
                                   row["controller"])
    if (launch.get("launch_authorized") is not True
            or launch.get("external_combination_authorization_required") is not True
            or launch.get("automatic_retries") != 0
            or launch.get("automatic_reruns") != 0
            or launch.get("required_environment") != provenance.get("required_env")):
        raise ValueError("Combination shard changed required environment guards")
    for artifact in provenance.get("copied_model_artifacts", []):
        path = expansion._within(root, artifact.get("path"), "model artifact")
        if (not path.is_file() or path.stat().st_size == 0
                or _sha(path) != artifact.get("sha256")):
            raise ValueError("Combination shard fitted artifact changed")
    if row["stage"] == "H1_R1":
        allowed = {"G"}
        if row["controller"] == "C1":
            allowed.add("selector_threshold")
        if gp.get("G") != "record_bound" or gp.get("R") != 1 or not _same_except(source_gp, gp, allowed):
            raise ValueError("H1 shard changes fields outside G/calibration")
        for relative, expected in G04_SOURCE_HASHES.items():
            if _sha(lane_root / "runtime" / relative) != expected:
                raise ValueError("H1 shard changed frozen G04 source")
        if row["controller"] in {"C4", "C4_turn", "C4_task"}:
            semantics = provenance.get("model_semantics", {})
            if semantics.get("artifact_training_history") != "H0" or semantics.get("h1_trained") is not False:
                raise ValueError("C4 H1 artifact provenance was relabeled")
        if row["controller"] == "C1":
            semantics = provenance.get("model_semantics", {})
            if (semantics.get("artifact_training_history") != "H0"
                    or semantics.get("h1_trained") is not False
                    or semantics.get("h1_threshold_calibrated") is not True
                    or design["search_contract"].get("trained_artifact_sha256")
                    != source_design["search_contract"].get(
                        "trained_artifact_sha256")):
                raise ValueError("C1 H1 calibration provenance differs")
            calibration_evidence = provenance.get("calibration_evidence")
            if not isinstance(calibration_evidence, dict):
                raise ValueError("C1 H1 calibration evidence is absent")
            for binding in calibration_evidence.values():
                path = expansion._within(root, binding.get("path"),
                                         "calibration evidence")
                if not path.is_file() or _sha(path) != binding.get("sha256"):
                    raise ValueError("C1 H1 calibration evidence changed")
    else:
        if gp.get("R") != 3 or not _same_except(source_gp, gp, {"R"}):
            raise ValueError("R3 shard changes fields outside recovery rounds")


def verify_package(package: Path) -> dict[str, Any]:
    package = package.resolve()
    expansion._verify_manifest(package, _read(package / "static_files.json"),
                               "combination package")
    contract = _read(package / "combination_contract.json")
    d128 = validate_d128_manifest(
        package / "tasks.d128.json", contract.get("d128_manifest_sha256"))
    rows = contract.get("shards")
    h1_controllers = contract.get("h1_controllers")
    h1_source_gate = contract.get("h1_source_gate")
    promotion_sha256 = contract.get("h1_d20_promotion_receipt_sha256")
    r3_budget = contract.get("r3_task_execution_budget")
    r3_source_gate = contract.get("r3_source_gate")
    if r3_source_gate is None:
        # Packages frozen before the terminal-D128 route used only strict complete receipts.
        r3_source_gate = "complete_d128" if r3_budget else "none"
    if (contract.get("schema") != PACKAGE_SCHEMA
            or contract.get("launch_authorized") is not False
            or contract.get("automatic_retries") != 0
            or contract.get("automatic_reruns") != 0
            or not isinstance(rows, list)
            or not isinstance(h1_controllers, list)
            or len(h1_controllers) not in {0, 2}
            or len(set(h1_controllers)) != len(h1_controllers)
            or contract.get("h1_task_execution_budget")
            != len(h1_controllers) * R3_EXECUTION_BUDGET
            or r3_budget not in {
                0, R3_EXECUTION_BUDGET}
            or h1_source_gate not in {"none", "complete_d128", "d20_promotion"}
            or r3_source_gate not in {"none", "complete_d128", "terminal_d128"}
            or (r3_budget == 0) != (r3_source_gate == "none")):
        raise ValueError("Invalid H1/R3 package contract")
    promotion_path = package / "h1_d20_promotion_receipt.json"
    if h1_source_gate == "d20_promotion":
        promotion = _read(promotion_path) if promotion_path.is_file() else {}
        if (h1_controllers != list(D20_PROMOTED_CONTROLLERS)
                or contract.get("r3_task_execution_budget") != 0
                or not expansion._valid_sha256(promotion_sha256)
                or not promotion_path.is_file()
                or _sha(promotion_path) != promotion_sha256
                or promotion.get("schema") != D20_PROMOTION_SCHEMA
                or promotion.get("phase") != "promotion_ready"
                or promotion.get("promotion", {}).get("selected")
                != h1_controllers):
            raise ValueError("Invalid packaged D20 promotion H1 gate")
    elif (promotion_sha256 is not None or promotion_path.exists()
          or (h1_source_gate == "complete_d128" and not h1_controllers)
          or (h1_source_gate == "none" and h1_controllers)):
        raise ValueError("Invalid H1 source gate provenance")
    terminal_path = package / "r3_terminal_d128_receipt.json"
    leading_path = package / "r3_leading_receipt.json"
    terminal_sha256 = contract.get("r3_terminal_d128_receipt_sha256")
    leading_sha256 = contract.get("r3_leading_receipt_sha256")
    if r3_source_gate == "terminal_d128":
        if (not expansion._valid_sha256(terminal_sha256)
                or not terminal_path.is_file()
                or _sha(terminal_path) != terminal_sha256
                or not expansion._valid_sha256(leading_sha256)
                or not leading_path.is_file()
                or _sha(leading_path) != leading_sha256):
            raise ValueError("Invalid packaged terminal-D128 R3 gate")
        terminal = terminal_results.verify_terminal_receipt(
            terminal_path, expected_sha256=terminal_sha256,
            expected_manifest_sha256=contract["d128_manifest_sha256"],
            expected_task_ids=d128["task_ids"], verify_inputs=False,
        )["document"]
        leading_receipt = _read(leading_path)
        if (terminal.get("status") != "terminal"
                or terminal.get("selection_eligible") is not True
                or leading_receipt.get("schema") != TERMINAL_LEADING_SCHEMA
                or leading_receipt.get("status") != "selected_after_terminal_d128"
                or leading_receipt.get("terminal_d128_receipt_sha256")
                != terminal_sha256):
            raise ValueError("Packaged terminal-D128 R3 selection differs")
    elif (terminal_sha256 is not None or terminal_path.exists()
          or leading_path.exists()):
        raise ValueError("Unexpected packaged terminal-D128 R3 receipt")
    if len({row.get("shard_id") for row in rows}) != len(rows):
        raise ValueError("Combination package repeats a shard ID")
    by_stage: dict[str, list[dict[str, Any]]] = {"H1_R1": [], "R3": []}
    for row in rows:
        by_stage[row["stage"]].append(row)
        _verify_shard(package / "shards" / row["shard_id"], row, d128)
    if (len(by_stage["H1_R1"])
            != len(contract["h1_controllers"]) * H1_SHARDS_PER_CONTROLLER
            or {row["controller"] for row in by_stage["H1_R1"]}
            != set(h1_controllers)
            or any(sum(row["controller"] == controller
                       for row in by_stage["H1_R1"])
                   != H1_SHARDS_PER_CONTROLLER
                   for controller in h1_controllers)
            or len(by_stage["R3"])
            != (R3_SHARD_COUNT if contract["r3_task_execution_budget"] else 0)):
        raise ValueError("Combination shard count differs from fixed budgets")
    for controller in h1_controllers:
        controller_rows = sorted(
            (row for row in by_stage["H1_R1"] if row["controller"] == controller),
            key=lambda row: row["shard_id"],
        )
        expected = [len(d128["task_ids"][part::H1_SHARDS_PER_CONTROLLER])
                    for part in range(H1_SHARDS_PER_CONTROLLER)]
        if [row["task_budget"] for row in controller_rows] != expected:
            raise ValueError("H1 shard task budgets differ from 3-way D128 split")
    if by_stage["R3"]:
        r3_rows = sorted(by_stage["R3"], key=lambda row: row["shard_id"])
        expected = [len(d128["task_ids"][part::R3_SHARD_COUNT])
                    for part in range(R3_SHARD_COUNT)]
        if [row["task_budget"] for row in r3_rows] != expected:
            raise ValueError("R3 shard task budgets differ from 6-way D128 split")
        selections = [
            _read(package / "shards" / row["shard_id"] / "provenance.json")[
                "source_selection_provenance"]
            for row in r3_rows
        ]
        if r3_source_gate == "terminal_d128":
            if any(
                selection.get("mode") != "terminal_d128"
                or selection.get("terminal_d128_receipt_sha256") != terminal_sha256
                or selection.get("leading_receipt_sha256") != leading_sha256
                for selection in selections
            ):
                raise ValueError("R3 shards differ from terminal-D128 selection gate")
        elif any(selection.get("mode") != "complete_d128"
                 for selection in selections):
            raise ValueError("R3 shards differ from complete-D128 selection gate")
    for wave in {row["wave"] for row in rows}:
        active = [row for row in rows if row["wave"] == wave]
        if len({row["preferred_device"] for row in active}) != len(active):
            raise ValueError(f"Wave {wave} assigns one device twice")
        ports = [port for row in active for port in
                 [row["engine_port"], *range(row["task_port_base"],
                                              row["task_port_base"]
                                              + row["task_budget"])]]
        if len(ports) != len(set(ports)):
            raise ValueError(f"Wave {wave} ports overlap")
    return {"schema": "experiment3-combination-verification-v1", "status": "passed",
            "package": str(package), "h1_task_execution_budget":
            contract["h1_task_execution_budget"], "r3_task_execution_budget":
            contract["r3_task_execution_budget"], "shard_count": len(rows)}


def authorization_requirements(package: Path) -> dict[str, Any]:
    package = package.resolve()
    contract = _read(package / "combination_contract.json")
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "explicit_root_authorization_required",
        "launch_authorized": False,
        "required_authorizer": "root",
        "authorization_scope": "experiment3_h1_r3_d128",
        "package_static_files_sha256": _sha(package / "static_files.json"),
        "combination_contract_sha256": _sha(
            package / "combination_contract.json"),
        "build_spec_sha256": _sha(package / "build_spec.json"),
        "d128_manifest_sha256": _sha(package / "tasks.d128.json"),
        "authorized_shard_ids": [row["shard_id"] for row in contract["shards"]],
        "h1_task_execution_budget": contract["h1_task_execution_budget"],
        "r3_task_execution_budget": contract["r3_task_execution_budget"],
        "total_task_execution_budget": (
            contract["h1_task_execution_budget"]
            + contract["r3_task_execution_budget"]),
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }


def verify_authorization(package: Path, receipt_path: Path) -> dict[str, Any]:
    requirements = authorization_requirements(package)
    actual = _read(receipt_path)
    bound_fields = {
        key: value for key, value in requirements.items()
        if key not in {"status", "launch_authorized", "required_authorizer"}
    }
    if (
        actual.get("schema") != AUTHORIZATION_SCHEMA
        or actual.get("status") != "authorized"
        or actual.get("launch_authorized") is not True
        or actual.get("authorized_by") != "root"
        or any(actual.get(key) != value for key, value in bound_fields.items())
    ):
        raise ValueError(
            "Launch authorization does not bind this exact H1/R3 package")
    return actual


def run_shard(package: Path, shard_id: str, authorization: Path) -> int:
    """Run one externally authorized shard through the frozen evaluator."""

    package = package.resolve()
    verify_package(package)
    receipt = verify_authorization(package, authorization)
    contract = _read(package / "combination_contract.json")
    rows = {row["shard_id"]: row for row in contract["shards"]}
    if shard_id not in rows or shard_id not in receipt["authorized_shard_ids"]:
        raise ValueError(f"Unknown or unauthorized shard: {shard_id}")
    shard_root = package / "shards" / shard_id
    lane_root = shard_root / "lanes" / shard_id
    if (lane_root / "run").exists() or (lane_root / "results").exists():
        raise FileExistsError(f"Refusing rerun of H1/R3 shard {shard_id}")
    base = expansion._load_base_module(shard_root / "evidence_eval.py")
    lane = _read(lane_root / "lane.json")
    lane_raw = {key: value for key, value in lane.items()
                if key not in {"name", "task_ports"}}
    base.LANES = {shard_id: lane_raw}
    base.lane_specs = lambda: [copy.deepcopy(lane)]
    base.DEFAULT_REMOTE_ROOT = shard_root
    provenance = _read(shard_root / "provenance.json")
    for key, value in provenance.get("required_env", {}).items():
        os.environ[key] = value
    # The frozen evaluator repeats this immediately before model work and only
    # cleans up the engine process group that it starts.
    base._assert_lane_free(lane)
    return base.run_lane(shard_root, shard_id)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--package", type=Path, required=True)
    prepare_parser.add_argument("--spec", type=Path, required=True)
    verify_parser = sub.add_parser("verify-package")
    verify_parser.add_argument("--package", type=Path, required=True)
    auth_parser = sub.add_parser("authorization-requirements")
    auth_parser.add_argument("--package", type=Path, required=True)
    run_parser = sub.add_parser("run-shard")
    run_parser.add_argument("--package", type=Path, required=True)
    run_parser.add_argument("--shard", required=True)
    run_parser.add_argument("--authorization", type=Path, required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--controller", action="append", default=[])
    plan_parser.add_argument("--include-r3", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.package, args.spec)
    elif args.command == "verify-package":
        result = verify_package(args.package)
    elif args.command == "authorization-requirements":
        result = authorization_requirements(args.package)
    elif args.command == "run-shard":
        return run_shard(args.package, args.shard, args.authorization)
    else:
        if len(args.controller) not in {0, 2}:
            raise ValueError("plan requires zero or two H1 controllers")
        result = {"schema": "experiment3-combination-shard-plan-v1",
                  "launch_authorized": False,
                  "shards": shard_plan(args.controller, args.include_r3)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
