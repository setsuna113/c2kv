"""Freeze and run the bounded Experiment 3 D128 remaining-108 expansion.

Preparation is local and never launches model work.  Each promoted controller
is bound to exactly one already-frozen evaluation package/lane.  Six 36-task
packages clone that source lane's evaluator, SGLang tree, runtime, controller,
and (when applicable) fitted artifact.  Running a shard additionally requires
an external, package-bound launch authorization receipt.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Mapping, Sequence

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import evidence_eval as local_base


READINESS_SCHEMA = "experiment3-expansion-readiness-v1"
SOURCE_BINDINGS_SCHEMA = "experiment3-expansion-source-bindings-v1"
PACKAGE_SCHEMA = "experiment3-d128-expansion-package-v1"
AUTHORIZATION_SCHEMA = "experiment3-expansion-launch-authorization-v1"
CONTROLLERS = ("C0", "C1", "C2", "C3", "C4_turn", "C4_task", "C5")
DEVICES = (0, 1, 2, 3, 4, 6)
ENGINE_PORT_BASE = 19000
TASK_PORT_BASE = 20000
TASKS_PER_SHARD = 36
TOTAL_TASK_BUDGET = 216
TASK_PATTERN = re.compile(r"multi_turn_[a-z0-9_]+_([0-9]+)$")


def task_groups(task_ids: Sequence[str]) -> set[int]:
    groups = set()
    for task_id in task_ids:
        match = TASK_PATTERN.fullmatch(task_id)
        if match is None:
            raise ValueError(f"Unsupported BFCL task ID: {task_id}")
        groups.add(int(match.group(1)))
    return groups


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _save(path: Path, value: Mapping[str, Any]) -> None:
    local_base._save(path, value)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _within(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {label} path")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} path escapes its frozen source package") from error
    return path


def _copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
    )


def _tree_hashes(root: Path, *, skip_sglang: bool = False) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix != ".pyc"
        and (not skip_sglang or path.relative_to(root).parts[0] != "sglang")
    }


def _manifest(root: Path, schema: str, *, skip_sglang: bool = False,
              excluded: Sequence[str] = ()) -> dict[str, Any]:
    excluded_set = set(excluded)
    files = {
        name: digest
        for name, digest in _tree_hashes(root, skip_sglang=skip_sglang).items()
        if name not in excluded_set
    }
    if not files:
        raise ValueError(f"Refusing an empty {schema} manifest")
    return {"schema": schema, "file_count": len(files), "files": files}


def _verify_manifest(root: Path, manifest: Mapping[str, Any], label: str) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{label} requires a nonempty files mapping")
    if manifest.get("file_count") != len(files):
        raise ValueError(f"{label} file_count differs from files")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not _valid_sha256(expected):
            raise ValueError(f"{label} contains an invalid file binding")
        path = _within(root, relative, label)
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"Frozen {label} changed: {relative}")


def _validate_manifest_document(controller: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{controller}: missing remaining-108 manifest")
    tasks = value.get("task_ids")
    reused = value.get("reused_task_ids")
    full = value.get("full_task_ids")
    if (
        not isinstance(tasks, list)
        or len(tasks) != 108
        or len(set(tasks)) != 108
        or not isinstance(reused, list)
        or len(reused) != 20
        or len(set(reused)) != 20
        or not isinstance(full, list)
        or len(full) != 128
        or len(set(full)) != 128
        or set(tasks) & set(reused)
        or set(tasks) | set(reused) != set(full)
    ):
        raise ValueError(f"{controller}: invalid 20+108 D128 manifest")
    task_groups(tasks)
    task_groups(reused)
    if value.get("reuse_requires_unchanged_algorithm_and_verified_d20_provenance") is not True:
        raise ValueError(f"{controller}: D20 provenance guard is absent")
    manifest_id = value.get("manifest_id")
    if not isinstance(manifest_id, str) or controller not in manifest_id:
        raise ValueError(f"{controller}: manifest_id does not bind the controller")
    return value


def validate_readiness(readiness: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete promotion decision and exact six-shard proposal."""
    promotion = readiness.get("promotion")
    selected = promotion.get("selected") if isinstance(promotion, dict) else None
    lanes = readiness.get("lanes")
    manifests = readiness.get("next_manifests")
    shards = readiness.get("proposed_shards")
    reuse_validation = readiness.get("reuse_validation")
    reuse_input = readiness.get("inputs", {}).get("reuse_audit")
    if (
        readiness.get("schema") != READINESS_SCHEMA
        or readiness.get("phase") != "promotion_ready"
        or not isinstance(promotion, dict)
        or promotion.get("phase") != "promotion_ready"
        or not isinstance(selected, list)
        or len(selected) != 2
        or len(set(selected)) != 2
        or any(name not in CONTROLLERS for name in selected)
        or not isinstance(lanes, dict)
        or set(lanes) != set(CONTROLLERS)
        or any(not isinstance(lanes[name], dict) or not lanes[name].get("all_cells_terminal")
               for name in CONTROLLERS)
        or not isinstance(manifests, dict)
        or list(manifests) != selected
        or readiness.get("new_execution_count") != TOTAL_TASK_BUDGET
        or readiness.get("automatic_retries") != 0
        or readiness.get("launch_authorized") is not False
        or not isinstance(reuse_validation, dict)
        or reuse_validation.get("status")
        != "audited_historical_lanes_bound_to_target"
        or reuse_validation.get("audited_lanes") != ["C0", "C3", "C5"]
        or not isinstance(reuse_input, dict)
        or not _valid_sha256(reuse_input.get("sha256"))
    ):
        raise ValueError("Readiness is not a complete frozen promotion_ready decision")
    checked = {name: _validate_manifest_document(name, manifests[name]) for name in selected}
    reused_sets = [checked[name]["reused_task_ids"] for name in selected]
    if reused_sets[0] != reused_sets[1]:
        raise ValueError("Promoted controllers do not share the same ordered D20")
    canonical_d20 = reused_sets[0]
    for name in CONTROLLERS:
        cells = lanes[name].get("quality_cells")
        if (not isinstance(cells, list) or len(cells) != 20
                or [cell.get("task_id") for cell in cells] != canonical_d20):
            raise ValueError(f"{name}: readiness does not preserve the exact ordered D20")
    if any(lanes[name].get("clean_d20") is not True for name in selected):
        raise ValueError("A promoted controller lacks a clean D20 outcome")
    if not isinstance(shards, list) or len(shards) != 6:
        raise ValueError("Readiness must propose exactly six shards")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in shards:
        if not isinstance(row, dict) or not isinstance(row.get("shard_id"), str):
            raise ValueError("Malformed proposed shard")
        if row["shard_id"] in by_id:
            raise ValueError("Duplicate proposed shard ID")
        by_id[row["shard_id"]] = row
    expected_ids = []
    for rank, controller in enumerate(selected):
        tasks = checked[controller]["task_ids"]
        for part in range(3):
            shard_id = f"{controller}_part{part}"
            expected_ids.append(shard_id)
            row = by_id.get(shard_id)
            device = DEVICES[rank * 3 + part]
            if (
                row is None
                or row.get("controller") != controller
                or row.get("preferred_physical_device") != device
                or row.get("task_ids") != tasks[part::3]
                or row.get("task_budget") != TASKS_PER_SHARD
                or row.get("automatic_retries") != 0
                or row.get("launch_authorized") is not False
            ):
                raise ValueError(f"Readiness shard differs from fixed plan: {shard_id}")
    if set(by_id) != set(expected_ids):
        raise ValueError("Readiness contains unexpected shards")
    return {"selected": selected, "manifests": checked,
            "shards": [dict(by_id[name]) for name in expected_ids]}


def _source_manifest_hash(source: Path, name: str, expected: object) -> dict[str, Any]:
    path = source / name
    if not _valid_sha256(expected) or not path.is_file() or _sha(path) != expected:
        raise ValueError(f"Frozen source {name} differs from its explicit binding")
    value = _read(path)
    _verify_manifest(source, value, f"source {name}")
    return value


def _verify_artifacts(source: Path, design: Mapping[str, Any], binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = binding.get("model_artifacts")
    if not isinstance(rows, list):
        raise ValueError("Source binding requires an explicit model_artifacts list")
    trained_sha = design.get("search_contract", {}).get("trained_artifact_sha256")
    verified = []
    payloads = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("Malformed model artifact binding")
        path = _within(source, row.get("path"), f"model artifact {index}")
        expected = row.get("sha256")
        if not _valid_sha256(expected) or not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Missing or empty fitted model artifact")
        if _sha(path) != expected:
            raise ValueError("Fitted model artifact differs from its source binding")
        try:
            payload = _read(path)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
            raise ValueError("Fitted model artifact must be a nonempty JSON object") from error
        if not payload:
            raise ValueError("Fitted model artifact must not be empty or untrained")
        payloads.append(payload)
        verified.append({"source_path": path, "relative": path.relative_to(source),
                         "sha256": expected})
    if trained_sha is not None:
        if not _valid_sha256(trained_sha) or trained_sha not in {row["sha256"] for row in verified}:
            raise ValueError("Trained controller lacks its exact fitted artifact")
        configured = (design.get("resolved_configs", {}).get("controller", {})
                      .get("gp_experiments", {}).get("selector_artifact"))
        if configured not in payloads:
            raise ValueError("Fitted artifact is not the artifact referenced by the controller")
    elif verified:
        raise ValueError("Untrained controller must not acquire a synthetic training artifact")
    return verified


def verify_source_bindings(readiness: Mapping[str, Any], document: Mapping[str, Any]) -> dict[str, Any]:
    state = validate_readiness(readiness)
    bindings = document.get("bindings")
    if document.get("schema") != SOURCE_BINDINGS_SCHEMA or not isinstance(bindings, dict):
        raise ValueError("Unknown Experiment 3 source binding schema")
    if set(bindings) != set(state["selected"]):
        raise ValueError("Source bindings must cover exactly the two promoted controllers")
    readiness_audit = readiness["inputs"]["reuse_audit"]
    document_audit = document.get("reuse_audit")
    if (not isinstance(document_audit, dict)
            or document_audit.get("sha256") != readiness_audit.get("sha256")):
        raise ValueError("Source binding audit SHA differs from promotion readiness")
    audited_controllers = set(state["selected"]) & {"C0", "C3", "C5"}
    audit = None
    audit_sha256 = None
    if audited_controllers:
        audit_binding = document.get("reuse_audit")
        if not isinstance(audit_binding, dict):
            raise ValueError("Selected controller requires the frozen D20 compatibility audit")
        audit_path_value = audit_binding.get("path")
        if not isinstance(audit_path_value, str) or not audit_path_value:
            raise ValueError("D20 compatibility audit path is missing")
        audit_path = Path(audit_path_value).resolve()
        audit_sha256 = audit_binding.get("sha256")
        if (not _valid_sha256(audit_sha256) or not audit_path.is_file()
                or _sha(audit_path) != audit_sha256):
            raise ValueError("D20 compatibility audit differs from its explicit binding")
        audit = _read(audit_path)
        conclusion = audit.get("conclusion", {})
        if (audit.get("schema") != "evidence-sets-d20-runtime-compatibility-v1"
                or conclusion.get("status") != "semantic_nonactivation_compatible"
                or conclusion.get("result_reuse_supported") is not True):
            raise ValueError("D20 compatibility audit does not support result reuse")
    result = {}
    for controller in state["selected"]:
        binding = bindings[controller]
        if (not isinstance(binding, dict) or binding.get("controller") != controller
                or not isinstance(binding.get("source_id"), str)
                or not binding.get("source_id")):
            raise ValueError(f"{controller}: source binding controller mismatch")
        source_value = binding.get("source_package")
        if not isinstance(source_value, str) or not source_value:
            raise ValueError(f"{controller}: missing source_package")
        source = Path(source_value).resolve()
        source_lane = binding.get("source_lane")
        if not isinstance(source_lane, str) or not source_lane:
            raise ValueError(f"{controller}: missing source_lane")
        static = _source_manifest_hash(source, "static_files.json",
                                       binding.get("source_static_files_sha256"))
        sglang = _source_manifest_hash(source, "sglang_files.json",
                                       binding.get("source_sglang_files_sha256"))
        required = (source / "evidence_eval.py", source / "history_system/runner.py",
                    source / "launch_contract.json",
                    source / "sglang", source / "lanes" / source_lane / "runtime",
                    source / "lanes" / source_lane / "design.json",
                    source / "lanes" / source_lane / "lane.json")
        if any(not path.exists() for path in required):
            raise FileNotFoundError(f"{controller}: frozen source package/lane is incomplete")
        design_path = source / "lanes" / source_lane / "design.json"
        controller_path = source / "lanes" / source_lane / "runtime/configs/controller.json"
        design = _read(design_path)
        if (
            binding.get("source_algorithm_id") != design.get("candidate_id")
            or not isinstance(design.get("candidate_id"), str)
            or design.get("status") != "frozen"
            or binding.get("source_design_sha256") != _sha(design_path)
            or binding.get("source_controller_sha256") != _sha(controller_path)
            or design.get("resolved_configs", {}).get("controller") != _read(controller_path)
            or design.get("automatic_reruns") != 0
        ):
            raise ValueError(f"{controller}: source algorithm/design binding differs")
        required_env = binding.get("required_env", {})
        source_required_env = _read(source / "launch_contract.json").get(
            "required_environment", {})
        if (not isinstance(required_env, dict)
                or any(not isinstance(key, str) or not isinstance(value, str)
                       for key, value in required_env.items())
                or required_env != source_required_env):
            raise ValueError(
                f"{controller}: required_env differs from the frozen source launch contract")
        artifacts = _verify_artifacts(source, design, binding)
        if design.get("search_contract", {}).get("trained_artifact_sha256") is not None:
            if required_env.get("C2KV_STRICT_NONFINITE_SAMPLING") != "1":
                raise ValueError(f"{controller}: trained source lost strict numeric guard")
        if controller in audited_controllers:
            assert audit is not None
            audit_lane = audit.get("lanes", {}).get(controller, {})
            target = audit.get("target", {})
            gp_path = source / "lanes" / source_lane / "gp.json"
            provenance_path = source / "provenance.json"
            target_runtime_hashes: dict[str, str] = {}
            partitions = audit_lane.get("partition_analysis")
            if not isinstance(partitions, dict) or not partitions:
                raise ValueError(f"{controller}: audit has no target runtime hashes")
            for partition in partitions.values():
                files = partition.get("audited_file_hashes") if isinstance(partition, dict) else None
                if not isinstance(files, dict) or not files:
                    raise ValueError(f"{controller}: audit partition lacks runtime hashes")
                for relative, hashes in files.items():
                    target_hash = hashes.get("target") if isinstance(hashes, dict) else None
                    if not _valid_sha256(target_hash):
                        raise ValueError(f"{controller}: invalid audited target runtime hash")
                    previous = target_runtime_hashes.setdefault(relative, target_hash)
                    if previous != target_hash:
                        raise ValueError(f"{controller}: inconsistent audited target runtime hash")
            runtime = source / "lanes" / source_lane / "runtime"
            runtime_matches = all(
                (runtime / relative).is_file() and _sha(runtime / relative) == expected
                for relative, expected in target_runtime_hashes.items())
            if (
                binding.get("source_id") != target.get("source_id")
                or audit.get("conclusion", {}).get("lanes", {}).get(controller)
                != "semantic_nonactivation_compatible_20_of_20"
                or audit_lane.get("classification")
                != "semantic_nonactivation_compatible_20_of_20"
                or audit_lane.get("target_design_sha256")
                != binding.get("source_design_sha256")
                or not gp_path.is_file()
                or audit_lane.get("target_gp_sha256") != _sha(gp_path)
                or not provenance_path.is_file()
                or target.get("provenance_sha256") != _sha(provenance_path)
                or not runtime_matches
            ):
                raise ValueError(
                    f"{controller}: source package/lane differs from the audited D20 target")
        result[controller] = {
            "binding": dict(binding), "source": source, "source_lane": source_lane,
            "design": design, "static_manifest": static, "sglang_manifest": sglang,
            "artifacts": artifacts,
            "reuse_audit_sha256": audit_sha256 if controller in audited_controllers else None,
        }
    return result


def _set_path(value: dict[str, Any], path: Sequence[str], replacement: Any) -> None:
    current = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


OPERATIONAL_DESIGN_PATHS = (
    ("task_ids",),
    ("task_manifest_sha256",),
    ("limits", "tasks"),
    ("runtime", "sglang_backend_url"),
    ("search_contract", "fixed_denominator"),
)


def _without_operational_fields(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    for path in OPERATIONAL_DESIGN_PATHS:
        current: Any = result
        for key in path[:-1]:
            if not isinstance(current, dict) or key not in current:
                current = None
                break
            current = current[key]
        if isinstance(current, dict):
            current.pop(path[-1], None)
    return result


def assert_clone_only_operational_changes(source: Mapping[str, Any], clone: Mapping[str, Any]) -> None:
    if _without_operational_fields(source) != _without_operational_fields(clone):
        raise ValueError("Shard design changes model semantics outside operational fields")


def _clone_design(source: Mapping[str, Any], tasks: Sequence[str], manifest_id: str,
                  manifest_sha256: str, engine_port: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(source))
    _set_path(result, ("task_ids",), list(tasks))
    _set_path(result, ("task_manifest_sha256",), manifest_sha256)
    _set_path(result, ("limits", "tasks"), TASKS_PER_SHARD)
    _set_path(result, ("runtime", "sglang_backend_url"),
              f"http://127.0.0.1:{engine_port}")
    _set_path(result, ("search_contract", "fixed_denominator"), manifest_id)
    assert_clone_only_operational_changes(source, result)
    return result


def _lane_spec(source_lane: Mapping[str, Any], shard_id: str, device: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(source_lane))
    result.update({
        "name": shard_id,
        "physical_device": device,
        "engine_port": ENGINE_PORT_BASE + device * 10,
        "task_port_base": TASK_PORT_BASE + device * 200,
    })
    result["task_ports"] = list(range(result["task_port_base"],
                                      result["task_port_base"] + TASKS_PER_SHARD))
    return result


def _write_shard(shard_root: Path, proposal: Mapping[str, Any], manifest: Mapping[str, Any],
                 source_row: Mapping[str, Any]) -> dict[str, Any]:
    source = source_row["source"]
    source_lane = source_row["source_lane"]
    shard_id = proposal["shard_id"]
    lane_root = shard_root / "lanes" / shard_id
    lane_root.mkdir(parents=True)
    (shard_root / "history_system").mkdir()
    shutil.copyfile(source / "evidence_eval.py", shard_root / "evidence_eval.py")
    shutil.copyfile(source / "history_system/runner.py", shard_root / "history_system/runner.py")
    _copy_tree(source / "sglang", shard_root / "sglang")
    _copy_tree(source / "lanes" / source_lane / "runtime", lane_root / "runtime")
    for optional in ("gp.json",):
        path = source / "lanes" / source_lane / optional
        if path.is_file():
            shutil.copyfile(path, lane_root / optional)

    task_document = {
        "schema": "a-history-system-task-manifest-v1",
        "manifest_id": f"{manifest['manifest_id']}_{shard_id}",
        "stage": manifest.get("stage", "development_search"),
        "task_ids": proposal["task_ids"],
        "parent_manifest_id": manifest["manifest_id"],
    }
    task_path = shard_root / "tasks.json"
    _save(task_path, task_document)
    source_lane_value = _read(source / "lanes" / source_lane / "lane.json")
    lane = _lane_spec(source_lane_value, shard_id, proposal["preferred_physical_device"])
    design = _clone_design(source_row["design"], proposal["task_ids"],
                           task_document["manifest_id"], _sha(task_path),
                           lane["engine_port"])
    _save(lane_root / "design.json", design)
    _save(lane_root / "tasks.json", task_document)
    _save(lane_root / "lane.json", lane)

    artifact_rows = []
    for row in source_row["artifacts"]:
        target = lane_root / "model_artifacts" / row["relative"].name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["source_path"], target)
        artifact_rows.append({"path": target.relative_to(shard_root).as_posix(),
                              "sha256": row["sha256"]})
    _save(shard_root / "source_design.json", source_row["design"])
    provenance = {
        "schema": "experiment3-expansion-shard-provenance-v1",
        "controller": proposal["controller"],
        "shard_id": shard_id,
        "source_package": str(source),
        "source_lane": source_lane,
        "source_algorithm_id": source_row["binding"]["source_algorithm_id"],
        "source_static_files_sha256": source_row["binding"]["source_static_files_sha256"],
        "source_sglang_files_sha256": source_row["binding"]["source_sglang_files_sha256"],
        "source_design_sha256": source_row["binding"]["source_design_sha256"],
        "source_controller_sha256": source_row["binding"]["source_controller_sha256"],
        "required_env": source_row["binding"].get("required_env", {}),
        "copied_model_artifacts": artifact_rows,
        "historical_d20_and_new108_single_exact_config_claim_allowed": False,
    }
    _save(shard_root / "provenance.json", provenance)
    _save(shard_root / "sglang_files.json", source_row["sglang_manifest"])
    launch = {
        "schema": "experiment3-expansion-shard-launch-v1",
        "launch_authorized": True,
        "external_expansion_authorization_required": True,
        "per_lane_task_budget": TASKS_PER_SHARD,
        "total_task_execution_budget": TASKS_PER_SHARD,
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "lanes": [{"lane": shard_id, "controller": proposal["controller"],
                   "physical_device": lane["physical_device"],
                   "engine_port": lane["engine_port"],
                   "task_port_base": lane["task_port_base"]}],
    }
    _save(shard_root / "launch_contract.json", launch)
    _save(shard_root / "static_files.json", _manifest(
        shard_root, "experiment3-expansion-shard-static-files-v1",
        skip_sglang=True, excluded=("static_files.json",)))
    return {"shard_id": shard_id, "controller": proposal["controller"],
            "relative_package": shard_root.name, "task_count": TASKS_PER_SHARD,
            "device": lane["physical_device"], "engine_port": lane["engine_port"],
            "task_port_base": lane["task_port_base"],
            "static_files_sha256": _sha(shard_root / "static_files.json")}


def prepare(package: Path, readiness_path: Path, bindings_path: Path) -> dict[str, Any]:
    package = package.resolve()
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite expansion package: {package}")
    readiness = _read(readiness_path)
    bindings_document = _read(bindings_path)
    state = validate_readiness(readiness)
    bindings = verify_source_bindings(readiness, bindings_document)
    package.mkdir(parents=True)
    try:
        shutil.copyfile(Path(__file__), package / Path(__file__).name)
        shutil.copyfile(Path(local_base.__file__), package / "evidence_eval.py")
        shutil.copyfile(readiness_path, package / "readiness.json")
        shutil.copyfile(bindings_path, package / "source_bindings.json")
        audit_path = Path(bindings_document["reuse_audit"]["path"]).resolve()
        shutil.copyfile(audit_path, package / "d20_compatibility_audit.json")
        shard_rows = []
        for proposal in state["shards"]:
            controller = proposal["controller"]
            shard_rows.append(_write_shard(package / "shards" / proposal["shard_id"],
                                           proposal, state["manifests"][controller],
                                           bindings[controller]))
        all_ports = []
        for row in shard_rows:
            all_ports.extend([row["engine_port"],
                              *range(row["task_port_base"],
                                     row["task_port_base"] + TASKS_PER_SHARD)])
        if len(all_ports) != len(set(all_ports)):
            raise ValueError("Expansion ports overlap")
        d20_provenance = {
            name: [{key: cell.get(key) for key in ("task_id", "quality_source_id", "status")}
                   for cell in readiness["lanes"][name].get("quality_cells", [])]
            for name in state["selected"]
        }
        contract = {
            "schema": PACKAGE_SCHEMA,
            "status": "prepared_waiting_for_external_authorization",
            "launch_authorized": False,
            "authorization_schema": AUTHORIZATION_SCHEMA,
            "selected_controllers": state["selected"],
            "shards": shard_rows,
            "devices": list(DEVICES),
            "per_shard_task_budget": TASKS_PER_SHARD,
            "total_task_execution_budget": TOTAL_TASK_BUDGET,
            "automatic_retries": 0,
            "automatic_reruns": 0,
            "d20_compatibility_audit_sha256": _sha(
                package / "d20_compatibility_audit.json"),
            "historical_d20_task_provenance": d20_provenance,
            "combined_d128_reporting": {
                "single_exact_config_claim_allowed": False,
                "pending": "root per-task D20 compatibility audit",
                "required_views": ["historical_D20", "new108", "combined_with_provenance"],
            },
        }
        _save(package / "expansion_contract.json", contract)
        _save(package / "static_files.json", _manifest(
            package, "experiment3-expansion-static-files-v1",
            excluded=("static_files.json",)))
        receipt = verify_package(package)
        return {"schema": "experiment3-expansion-prepare-v1", "status": "prepared",
                "package": str(package), "verification": receipt,
                "authorization_requirements": authorization_requirements(package)}
    except BaseException:
        if package.exists():
            shutil.rmtree(package)
        raise


def _load_base_module(path: Path):
    name = "_experiment3_source_evidence_eval_" + hashlib.sha256(
        str(path).encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _verify_shard(shard_root: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    base = _load_base_module(shard_root / "evidence_eval.py")
    base.verify_package(shard_root)
    shard_id = row["shard_id"]
    lane = _read(shard_root / "lanes" / shard_id / "lane.json")
    tasks = _read(shard_root / "tasks.json")
    design = _read(shard_root / "lanes" / shard_id / "design.json")
    source_design = _read(shard_root / "source_design.json")
    provenance = _read(shard_root / "provenance.json")
    if (
        lane.get("name") != shard_id
        or lane.get("physical_device") != row["device"]
        or lane.get("engine_port") != row["engine_port"]
        or lane.get("task_port_base") != row["task_port_base"]
        or lane.get("task_ports") != list(range(row["task_port_base"],
                                                row["task_port_base"] + TASKS_PER_SHARD))
        or len(tasks.get("task_ids", [])) != TASKS_PER_SHARD
        or design.get("task_ids") != tasks["task_ids"]
        or design.get("task_manifest_sha256") != _sha(shard_root / "tasks.json")
        or design.get("limits", {}).get("tasks") != TASKS_PER_SHARD
        or design.get("runtime", {}).get("sglang_backend_url")
        != f"http://127.0.0.1:{lane.get('engine_port')}"
        or design.get("candidate_id") != provenance.get("source_algorithm_id")
        or provenance.get("historical_d20_and_new108_single_exact_config_claim_allowed") is not False
    ):
        raise ValueError(f"Frozen shard contract differs: {shard_id}")
    assert_clone_only_operational_changes(source_design, design)
    for artifact in provenance.get("copied_model_artifacts", []):
        path = _within(shard_root, artifact.get("path"), "copied model artifact")
        if (not path.is_file() or path.stat().st_size == 0
                or _sha(path) != artifact.get("sha256")):
            raise RuntimeError(f"Frozen fitted artifact changed: {path}")
    return {"shard_id": shard_id, "task_count": TASKS_PER_SHARD,
            "physical_device": lane["physical_device"]}


def verify_package(package: Path) -> dict[str, Any]:
    package = package.resolve()
    _verify_manifest(package, _read(package / "static_files.json"), "expansion package")
    contract = _read(package / "expansion_contract.json")
    rows = contract.get("shards")
    readiness = _read(package / "readiness.json")
    bindings = _read(package / "source_bindings.json")
    audit_sha256 = _sha(package / "d20_compatibility_audit.json")
    if (
        contract.get("schema") != PACKAGE_SCHEMA
        or contract.get("launch_authorized") is not False
        or contract.get("total_task_execution_budget") != TOTAL_TASK_BUDGET
        or contract.get("automatic_retries") != 0
        or contract.get("automatic_reruns") != 0
        or contract.get("d20_compatibility_audit_sha256") != audit_sha256
        or readiness.get("inputs", {}).get("reuse_audit", {}).get("sha256") != audit_sha256
        or bindings.get("reuse_audit", {}).get("sha256") != audit_sha256
        or not isinstance(rows, list)
        or len(rows) != 6
        or [row.get("device") for row in rows] != list(DEVICES)
    ):
        raise ValueError("Invalid frozen expansion contract")
    verified = []
    task_pairs = []
    ports = []
    for row in rows:
        shard_root = package / "shards" / row["shard_id"]
        verified.append(_verify_shard(shard_root, row))
        tasks = _read(shard_root / "tasks.json")["task_ids"]
        task_pairs.extend((row["controller"], task) for task in tasks)
        ports.extend([row["engine_port"],
                      *range(row["task_port_base"], row["task_port_base"] + TASKS_PER_SHARD)])
    if len(task_pairs) != TOTAL_TASK_BUDGET or len(set(task_pairs)) != TOTAL_TASK_BUDGET:
        raise ValueError("Expansion controller/task executions overlap or exceed budget")
    if len(ports) != len(set(ports)):
        raise ValueError("Expansion ports overlap")
    return {"schema": "experiment3-expansion-package-verification-v1",
            "status": "passed", "package": str(package),
            "shards": verified, "total_task_execution_budget": len(task_pairs)}


def authorization_requirements(package: Path) -> dict[str, Any]:
    package = package.resolve()
    contract = _read(package / "expansion_contract.json")
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "explicit_root_authorization_required",
        "launch_authorized": False,
        "required_authorizer": "root",
        "authorization_scope": "experiment3_d128_remaining108",
        "package_static_files_sha256": _sha(package / "static_files.json"),
        "expansion_contract_sha256": _sha(package / "expansion_contract.json"),
        "readiness_sha256": _sha(package / "readiness.json"),
        "source_bindings_sha256": _sha(package / "source_bindings.json"),
        "authorized_shard_ids": [row["shard_id"] for row in contract["shards"]],
        "total_task_execution_budget": TOTAL_TASK_BUDGET,
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
        raise ValueError("Launch authorization does not bind this exact frozen package")
    return actual


def run_shard(package: Path, shard_id: str, authorization: Path) -> int:
    package = package.resolve()
    verify_package(package)
    receipt = verify_authorization(package, authorization)
    contract = _read(package / "expansion_contract.json")
    rows = {row["shard_id"]: row for row in contract["shards"]}
    if shard_id not in rows or shard_id not in receipt["authorized_shard_ids"]:
        raise ValueError(f"Unknown or unauthorized shard: {shard_id}")
    row = rows[shard_id]
    shard_root = package / "shards" / shard_id
    lane_root = shard_root / "lanes" / shard_id
    if (lane_root / "run").exists() or (lane_root / "results").exists():
        raise FileExistsError(f"Refusing rerun of expansion shard {shard_id}")
    base = _load_base_module(shard_root / "evidence_eval.py")
    lane = _read(lane_root / "lane.json")
    lane_raw = {key: value for key, value in lane.items()
                if key not in {"name", "task_ports"}}
    base.LANES = {shard_id: lane_raw}
    base.lane_specs = lambda: [copy.deepcopy(lane)]
    base.DEFAULT_REMOTE_ROOT = shard_root
    provenance = _read(shard_root / "provenance.json")
    for key, value in provenance.get("required_env", {}).items():
        os.environ[key] = value
    # This is the final acquisition check before base.run_lane creates any run
    # state.  base.run_lane repeats it and only cleans up the engine it starts.
    base._assert_lane_free(lane)
    return base.run_lane(shard_root, shard_id)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--package", type=Path, required=True)
    prepare_parser.add_argument("--readiness", type=Path, required=True)
    prepare_parser.add_argument("--source-bindings", type=Path, required=True)
    verify_parser = sub.add_parser("verify-package")
    verify_parser.add_argument("--package", type=Path, required=True)
    auth_parser = sub.add_parser("authorization-requirements")
    auth_parser.add_argument("--package", type=Path, required=True)
    run_parser = sub.add_parser("run-shard")
    run_parser.add_argument("--package", type=Path, required=True)
    run_parser.add_argument("--shard", required=True)
    run_parser.add_argument("--authorization", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.package, args.readiness, args.source_bindings)
    elif args.command == "verify-package":
        result = verify_package(args.package)
    elif args.command == "authorization-requirements":
        result = authorization_requirements(args.package)
    else:
        return run_shard(args.package, args.shard, args.authorization)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
