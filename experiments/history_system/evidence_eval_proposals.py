"""Freeze, run, audit, and summarize the H0 proposal-selector D128 campaign.

The module deliberately does not schedule work.  ``build-design`` derives a
hash-bound design from frozen source packages; ``prepare`` creates immutable
per-shard packages; and ``run-shard`` still requires an exact external launch
authorization receipt.  A failed task is recorded once and the patched runner
continues with later pristine tasks.  It never retries the failed task.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evidence_eval as local_base


DESIGN_SCHEMA = "proposal-h0-d128-design-v1"
SOURCE_CATALOG_SCHEMA = "proposal-h0-d128-source-catalog-v1"
PACKAGE_SCHEMA = "proposal-h0-d128-package-v1"
AUTHORIZATION_SCHEMA = "proposal-h0-d128-launch-authorization-v1"
SUMMARY_SCHEMA = "proposal-h0-d128-summary-v1"
AUDIT_SCHEMA = "experiment3-d128-runtime-failure-audit-v1"
PROPOSAL_PROTOCOL = "evidence_set_proposals_h0_v1"
EXPECTED_D128_SHA256 = "e6063f178788c0ace9cc20c3933e551baf0a2afa242b5a252af09b23bef3afc9"
METHOD_ORDER = ("E07", "E09", "E11")
DEVICES = (0, 1, 2, 3, 4, 6)
SHARD_SIZES = (22, 22, 21, 21, 21, 21)
CAMPAIGN_TASK_EXECUTION_CEILING = 384
STRICT_ENVIRONMENT = {"C2KV_STRICT_NONFINITE_SAMPLING": "1"}
HEX64 = re.compile(r"[0-9a-f]{64}")
TASK_PATTERN = re.compile(r"multi_turn_[a-z0-9_]+_[0-9]+")

METHOD_SPECS: dict[str, dict[str, Any]] = {
    "E07": {
        "selector": "risk_source_proposal",
        "source_lane": "C1",
        "artifact_kind": "c1_risk_logistic",
        "requires_proposal_protocol": False,
        "model_smokes": ["embedding"],
    },
    "E09": {
        "selector": "gain_turn_proposals",
        "source_lane": "C4_turn",
        "artifact_kind": "c4_gain_turn",
        "requires_proposal_protocol": True,
        "model_smokes": ["embedding", "reranker"],
    },
    "E11": {
        "selector": "gain_task_proposals",
        "source_lane": "C4_task",
        "artifact_kind": "c4_gain_task",
        "requires_proposal_protocol": True,
        "model_smokes": ["embedding", "reranker"],
    },
}

RUNNER_CONTINUE_MARKER = "continue_after_actor_runtime_failure_for_pristine_remainder_v1"
RUNNER_STOP_BLOCK = '''            if runtime_failed:
                outcome["dispatch_stop_reason"] = "actor_runtime_failure_even_if_officially_scored"
                terminalize(record, status="stopped_on_actor_runtime_failure", started=started)
                save(manifest, record)
                return 6
'''
RUNNER_CONTINUE_BLOCK = f'''            if runtime_failed:
                outcome["dispatch_continue_reason"] = "{RUNNER_CONTINUE_MARKER}"
                save(manifest, record)
'''


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", "*.pyc"))


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
              excluded: Iterable[str] = ()) -> dict[str, Any]:
    excluded_set = set(excluded)
    files = {name: digest for name, digest in
             _tree_hashes(root, skip_sglang=skip_sglang).items()
             if name not in excluded_set}
    if not files:
        raise ValueError(f"Refusing empty manifest: {schema}")
    return {"schema": schema, "file_count": len(files), "files": files}


def _within(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {label} path")
    path = Path(value)
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} path escapes root: {value}") from error
    return path


def _verify_manifest(root: Path, value: Mapping[str, Any], label: str) -> None:
    files = value.get("files")
    if (not isinstance(files, dict) or not files
            or value.get("file_count") != len(files)):
        raise ValueError(f"Malformed {label} manifest")
    for relative, expected in files.items():
        if not isinstance(relative, str) or not _valid_sha(expected):
            raise ValueError(f"Malformed {label} file binding")
        path = _within(root, relative, label)
        if not path.is_file() or _sha(path) != expected:
            raise RuntimeError(f"Frozen {label} changed: {relative}")


def _load_module(path: Path, prefix: str = "_proposal_h0_"):
    name = prefix + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _normal_methods(methods: Sequence[str]) -> list[str]:
    result = list(methods)
    if (not result or len(set(result)) != len(result)
            or any(item not in METHOD_ORDER for item in result)
            or result != [item for item in METHOD_ORDER if item in result]):
        raise ValueError("Methods must be a nonempty ordered subset of E07,E09,E11")
    return result


def _canonical_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or _sha(path) != EXPECTED_D128_SHA256:
        raise ValueError("Canonical D128 manifest differs from its frozen SHA256")
    value = _read(path)
    tasks = value.get("task_ids")
    if (value.get("schema") != "a-history-system-task-manifest-v1"
            or value.get("manifest_id") != "R2D128"
            or value.get("fixed_denominator") != 128
            or value.get("automatic_reruns") != 0
            or not isinstance(tasks, list) or len(tasks) != 128
            or len(set(tasks)) != 128
            or any(not isinstance(task, str) or not TASK_PATTERN.fullmatch(task)
                   for task in tasks)):
        raise ValueError("Canonical D128 manifest is malformed")
    return value


def _patched_runner_bytes(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8")
    if text.count(RUNNER_STOP_BLOCK) != 1 or RUNNER_CONTINUE_MARKER in text:
        raise ValueError("Frozen runner lacks the exact first-failure stop seam")
    return text.replace(RUNNER_STOP_BLOCK, RUNNER_CONTINUE_BLOCK).encode("utf-8")


def _source_paths(package: Path, lane: str) -> dict[str, Path]:
    lane_root = package / "lanes" / lane
    return {
        "evaluator": package / "evidence_eval.py",
        "runner": package / "history_system/runner.py",
        "static_manifest": package / "static_files.json",
        "sglang_manifest": package / "sglang_files.json",
        "sglang": package / "sglang",
        "lane": lane_root / "lane.json",
        "design": lane_root / "design.json",
        "controller": lane_root / "runtime/configs/controller.json",
        "runtime": lane_root / "runtime",
        "launch_contract": package / "launch_contract.json",
    }


def _artifact(path: Path, method: str) -> dict[str, Any]:
    value = _read(path)
    spec = METHOD_SPECS[method]
    if value.get("model_kind") != spec["artifact_kind"]:
        raise ValueError(f"{method}: artifact model_kind does not match its frozen model")
    if (spec["requires_proposal_protocol"]
            and value.get("proposal_protocol") != PROPOSAL_PROTOCOL):
        raise ValueError(f"{method}: artifact is not trained for the proposal protocol")
    return value


def _runtime_bundle_validation(runtime: Path, artifact_path: Path,
                               controller_path: Path, method: str) -> dict[str, Any]:
    """Validate the full artifact/controller contract in an isolated import."""
    script = r'''
import json, sys
from pathlib import Path
runtime, artifact_path, controller_path, selector, needs_protocol = sys.argv[1:]
sys.path[:0] = [str(Path(runtime) / "python"), runtime]
from benchmarks.memory_runtime.recovery.experiment_config import parse_gp_config
from benchmarks.memory_runtime.recovery.experiment import GPRecoveryController
from benchmarks.memory_runtime.recovery.local_selection_models import LocalSelectionModels
from benchmarks.memory_runtime.recovery.set_models import (
    base_selector_kind, load_set_selector, validate_proposal_artifact,
    validate_selector_score_models,
)
artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
controller = json.loads(Path(controller_path).read_text(encoding="utf-8"))
gp = parse_gp_config(controller["gp_experiments"])
loaded = load_set_selector(artifact)
# Importing the controller is part of the overlay check: proposal selectors are
# unusable if their dispatch module cannot import the shared protocol/model code.
assert GPRecoveryController is not None
expected = base_selector_kind(selector)
if gp["set_selector"] != selector or loaded.kind != expected:
    raise ValueError("selector/artifact/controller identity mismatch")
if needs_protocol == "1":
    validate_proposal_artifact(loaded.artifact)
if expected != "risk":
    validate_selector_score_models(
        loaded, LocalSelectionModels(gp["local_models"]),
        semantic_query_overflow_policy=gp["semantic_query_overflow_policy"])
print(json.dumps({"status":"passed","selector":selector,"artifact_kind":artifact["model_kind"]}))
'''
    selector = METHOD_SPECS[method]["selector"]
    completed = subprocess.run(
        [sys.executable, "-c", script, str(runtime), str(artifact_path),
         str(controller_path), selector,
         "1" if METHOD_SPECS[method]["requires_proposal_protocol"] else "0"],
        check=True, capture_output=True, text=True)
    receipt = json.loads(completed.stdout)
    if not isinstance(receipt, dict) or receipt.get("status") != "passed":
        raise RuntimeError(f"{method}: runtime artifact validation did not pass")
    return receipt


def _validate_schedule(method: str, value: object) -> list[dict[str, int]]:
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"{method}: six explicit shard schedules are required")
    result: list[dict[str, int]] = []
    for part, (row, device, size) in enumerate(zip(value, DEVICES, SHARD_SIZES)):
        if not isinstance(row, dict):
            raise ValueError(f"{method}: malformed shard schedule")
        expected_id = f"{method}_part{part}"
        clean = {key: row.get(key) for key in
                 ("shard_id", "device", "engine_port", "task_port_base")}
        if (clean["shard_id"] != expected_id or clean["device"] != device
                or type(clean["engine_port"]) is not int
                or type(clean["task_port_base"]) is not int
                or not (1024 <= clean["engine_port"] <= 65535)
                or not (1024 <= clean["task_port_base"] <= 65535 - size + 1)):
            raise ValueError(f"{method}: invalid explicit schedule for {expected_id}")
        result.append(clean)  # type: ignore[arg-type]
    return result


def _source_identity(package: Path, lane: str) -> dict[str, Any]:
    package = package.resolve()
    paths = _source_paths(package, lane)
    if any(not path.exists() for path in paths.values()):
        raise FileNotFoundError(f"Incomplete frozen source package: {package}/{lane}")
    base = _load_module(paths["evaluator"], "_proposal_source_")
    base.verify_package(package)
    static = _read(paths["static_manifest"])
    sglang = _read(paths["sglang_manifest"])
    _verify_manifest(package, static, "source static")
    _verify_manifest(package, sglang, "source SGLang")
    design = _read(paths["design"])
    controller = _read(paths["controller"])
    launch = _read(paths["launch_contract"])
    if (design.get("automatic_reruns") != 0
            or launch.get("automatic_reruns") != 0
            or launch.get("required_environment") != STRICT_ENVIRONMENT
            or design.get("resolved_configs", {}).get("controller") != controller):
        raise ValueError(f"{lane}: frozen source contract is incompatible")
    patched = _patched_runner_bytes(paths["runner"])
    return {
        "package": str(package),
        "lane": lane,
        "static_files_sha256": _sha(paths["static_manifest"]),
        "sglang_files_sha256": _sha(paths["sglang_manifest"]),
        "evaluator_sha256": _sha(paths["evaluator"]),
        "runner_sha256": _sha(paths["runner"]),
        "patched_runner_sha256": _bytes_sha(patched),
        "runner_operation": RUNNER_CONTINUE_MARKER,
        "design_sha256": _sha(paths["design"]),
        "controller_sha256": _sha(paths["controller"]),
        "lane_sha256": _sha(paths["lane"]),
    }


def build_design_document(*, source_catalog_path: Path,
                          canonical_manifest_path: Path,
                          runtime_source_root: Path,
                          overlay_files: Sequence[str],
                          methods: Sequence[str]) -> dict[str, Any]:
    """Derive every digest; callers never manually transcribe source hashes."""
    selected = _normal_methods(methods)
    catalog_path = source_catalog_path.resolve()
    catalog = _read(catalog_path)
    rows = catalog.get("methods")
    if catalog.get("schema") != SOURCE_CATALOG_SCHEMA or not isinstance(rows, dict):
        raise ValueError("Unknown proposal source catalog schema")
    manifest = _canonical_manifest(canonical_manifest_path)
    runtime_source = runtime_source_root.resolve()
    overlay_names = list(overlay_files)
    if (not overlay_names or len(set(overlay_names)) != len(overlay_names)
            or "benchmarks/memory_runtime/recovery/experiment.py" not in overlay_names):
        raise ValueError("Runtime overlay must be explicit and include recovery/experiment.py")
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in overlay_names):
        raise ValueError("Runtime overlay paths must be safe runtime-relative paths")

    method_rows = []
    all_ports: list[int] = []
    old_hashes: dict[str, str] = {}
    for method in selected:
        catalog_row = rows.get(method)
        if not isinstance(catalog_row, dict):
            raise ValueError(f"Source catalog lacks {method}")
        spec = METHOD_SPECS[method]
        package = Path(str(catalog_row.get("source_package", ""))).resolve()
        lane = catalog_row.get("source_lane", spec["source_lane"])
        if lane != spec["source_lane"]:
            raise ValueError(f"{method}: wrong frozen source lane")
        identity = _source_identity(package, lane)
        artifact_path = Path(str(catalog_row.get("artifact_path", ""))).resolve()
        artifact = _artifact(artifact_path, method)
        schedule = _validate_schedule(method, catalog_row.get("shards"))
        source_runtime = package / "lanes" / lane / "runtime"
        overlays = []
        for relative in overlay_names:
            old = source_runtime / relative
            new = runtime_source / relative
            if not old.is_file() or not new.is_file():
                raise FileNotFoundError(f"{method}: missing runtime overlay file {relative}")
            old_sha = _sha(old)
            previous = old_hashes.setdefault(relative, old_sha)
            if previous != old_sha:
                raise ValueError(f"Frozen source runtimes differ for overlay file {relative}")
            overlays.append({
                "relative_path": relative,
                "source_path": str(new),
                "source_sha256": old_sha,
                "target_sha256": _sha(new),
            })
        for row, size in zip(schedule, SHARD_SIZES):
            all_ports.extend([row["engine_port"],
                              *range(row["task_port_base"],
                                     row["task_port_base"] + size)])
        method_rows.append({
            "method": method,
            "selector": spec["selector"],
            "proposal_protocol": PROPOSAL_PROTOCOL,
            "source": identity,
            "artifact": {
                "path": str(artifact_path),
                "sha256": _sha(artifact_path),
                "model_kind": artifact["model_kind"],
                "proposal_protocol": artifact.get("proposal_protocol"),
            },
            "runtime_overlay": overlays,
            "shards": schedule,
        })
    if len(all_ports) != len(set(all_ports)):
        raise ValueError("Explicit campaign engine/task ports overlap")
    total = 128 * len(selected)
    return {
        "schema": DESIGN_SCHEMA,
        "status": "designed_not_launched",
        "methods": method_rows,
        "canonical_manifest": {
            "path": str(canonical_manifest_path.resolve()),
            "sha256": EXPECTED_D128_SHA256,
            "manifest_id": manifest["manifest_id"],
            "task_ids": manifest["task_ids"],
        },
        "runtime_source_root": str(runtime_source),
        "invariants": {"history": "H0", "R": 1, "G": "current", "B": "B0"},
        "strict_environment": STRICT_ENVIRONMENT,
        "new_task_executions": total,
        "campaign_task_execution_ceiling": CAMPAIGN_TASK_EXECUTION_CEILING,
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "launch_authorized": False,
        "source_catalog": {"path": str(catalog_path), "sha256": _sha(catalog_path)},
    }


def build_design(output: Path, **kwargs: Any) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite design: {output}")
    value = build_design_document(**kwargs)
    _save(output, value)
    return {"status": "designed", "output": str(output), "sha256": _sha(output),
            "methods": [row["method"] for row in value["methods"]]}


def validate_design(value: Mapping[str, Any], *, methods: Sequence[str] | None = None
                    ) -> tuple[list[str], dict[str, Mapping[str, Any]], list[str]]:
    rows = value.get("methods")
    configured = ([row.get("method") for row in rows]
                  if isinstance(rows, list) and all(isinstance(row, dict) for row in rows)
                  else [])
    configured = _normal_methods(configured)
    selected = _normal_methods(methods) if methods else configured
    if any(method not in configured for method in selected):
        raise ValueError("Requested method is absent from the frozen design")
    by_method = {row["method"]: row for row in rows}  # type: ignore[index]
    canonical = value.get("canonical_manifest")
    if (value.get("schema") != DESIGN_SCHEMA
            or value.get("status") != "designed_not_launched"
            or not isinstance(canonical, dict)
            or canonical.get("sha256") != EXPECTED_D128_SHA256
            or canonical.get("manifest_id") != "R2D128"
            or not isinstance(canonical.get("task_ids"), list)
            or len(canonical["task_ids"]) != 128
            or len(set(canonical["task_ids"])) != 128
            or value.get("invariants") != {"history": "H0", "R": 1,
                                            "G": "current", "B": "B0"}
            or value.get("strict_environment") != STRICT_ENVIRONMENT
            or value.get("new_task_executions") != 128 * len(configured)
            or value.get("campaign_task_execution_ceiling") != 384
            or value.get("automatic_retries") != 0
            or value.get("automatic_reruns") != 0
            or value.get("launch_authorized") is not False):
        raise ValueError("Malformed proposal H0 D128 design")
    tasks = list(canonical["task_ids"])
    all_ports: list[int] = []
    for method in selected:
        row = by_method[method]
        spec = METHOD_SPECS[method]
        if (row.get("selector") != spec["selector"]
                or row.get("proposal_protocol") != PROPOSAL_PROTOCOL
                or row.get("source", {}).get("lane") != spec["source_lane"]
                or row.get("source", {}).get("runner_operation") != RUNNER_CONTINUE_MARKER
                or row.get("artifact", {}).get("model_kind") != spec["artifact_kind"]):
            raise ValueError(f"{method}: design identity differs")
        if (spec["requires_proposal_protocol"]
                and row.get("artifact", {}).get("proposal_protocol") != PROPOSAL_PROTOCOL):
            raise ValueError(f"{method}: artifact protocol differs")
        schedule = _validate_schedule(method, row.get("shards"))
        overlays = row.get("runtime_overlay")
        if (not isinstance(overlays, list) or not overlays
                or any(not isinstance(item, dict)
                       or not isinstance(item.get("relative_path"), str)
                       or not isinstance(item.get("source_path"), str)
                       or not _valid_sha(item.get("source_sha256"))
                       or not _valid_sha(item.get("target_sha256"))
                       for item in overlays)):
            raise ValueError(f"{method}: malformed runtime overlay")
        for schedule_row, size in zip(schedule, SHARD_SIZES):
            all_ports.extend([schedule_row["engine_port"],
                              *range(schedule_row["task_port_base"],
                                     schedule_row["task_port_base"] + size)])
    if len(all_ports) != len(set(all_ports)):
        raise ValueError("Selected method ports overlap")
    return selected, by_method, tasks


def _verify_bound_source(row: Mapping[str, Any]) -> dict[str, Any]:
    source = row["source"]
    package = Path(source["package"]).resolve()
    lane = source["lane"]
    actual = _source_identity(package, lane)
    if actual != source:
        raise RuntimeError(f"Frozen source package changed: {package}/{lane}")
    artifact_path = Path(row["artifact"]["path"]).resolve()
    artifact = _artifact(artifact_path, row["method"])
    if _sha(artifact_path) != row["artifact"]["sha256"]:
        raise RuntimeError(f"Frozen artifact changed: {artifact_path}")
    for overlay in row["runtime_overlay"]:
        old = package / "lanes" / lane / "runtime" / overlay["relative_path"]
        new = Path(overlay["source_path"]).resolve()
        if (_sha(old) != overlay["source_sha256"]
                or not new.is_file() or _sha(new) != overlay["target_sha256"]):
            raise RuntimeError(f"Runtime overlay changed: {overlay['relative_path']}")
    return {"package": package, "lane": lane, "artifact_path": artifact_path,
            "artifact": artifact}


def _controller(source: Mapping[str, Any], method: str,
                artifact: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    controller = copy.deepcopy(dict(source))
    gp = controller.get("gp_experiments")
    if not isinstance(gp, dict):
        raise ValueError("Frozen controller lacks gp_experiments")
    if (gp.get("semantic_query_overflow_policy")
            != "task_head_tail_preserve_draft_v1"
            or gp.get("export_selection_state") is not False
            or gp.get("local_models", {}).get("reranker", {}).get("batch_size") != 1
            or gp.get("selection_protocol") != "evidence_sets_v1"
            or gp.get("G") != "current"
            or gp.get("R") != 1
            or gp.get("candidate_limit") != 8
            or gp.get("K") != 4
            or gp.get("retrieval_limit") != 24
            or gp.get("retrieval_route_limit") != 24
            or gp.get("fallback_unit") is not None
            or gp.get("recovery_reserve_tokens") != 0):
        raise ValueError(f"{method}: frozen source model/payload policy differs")
    if method == "E07":
        if gp.get("selector_threshold") != 0.5:
            raise ValueError("E07: frozen source risk threshold differs")
    elif gp.get("gain_delta") != 0.0:
        raise ValueError(f"{method}: frozen source gain delta differs")
    gp.update({
        "set_selector": METHOD_SPECS[method]["selector"],
        "selector_artifact": copy.deepcopy(dict(artifact)),
        "proposal_protocol": PROPOSAL_PROTOCOL,
    })
    return controller, copy.deepcopy(gp)


def _design_for_shard(source: Mapping[str, Any], *, method: str,
                      shard_id: str, tasks: Sequence[str], task_sha256: str,
                      engine_port: int, controller: Mapping[str, Any],
                      artifact_sha256: str, runtime: Path) -> dict[str, Any]:
    value = copy.deepcopy(dict(source))
    value.update({
        "candidate_id": f"proposal_h0_d128_{method.lower()}",
        "run_id_template": f"proposal_h0_d128_{method.lower()}_{shard_id.lower()}",
        "evaluation_stage": "development_search",
        "launch_authorized": True,
        "task_ids": list(tasks),
        "task_manifest_sha256": task_sha256,
        "automatic_reruns": 0,
    })
    value.setdefault("limits", {})["tasks"] = len(tasks)
    value.setdefault("runtime", {})["sglang_backend_url"] = (
        f"http://127.0.0.1:{engine_port}")
    value.setdefault("resolved_configs", {})["controller"] = copy.deepcopy(dict(controller))
    value["source_files"] = _tree_hashes(runtime)
    contract = value.setdefault("search_contract", {})
    contract.update({
        "fixed_denominator": "R2D128",
        "history": "H0",
        "selection_protocol": PROPOSAL_PROTOCOL,
        "proposal_protocol": PROPOSAL_PROTOCOL,
        "recovery_attempts": 1,
        "fallback_unit": None,
        "recovery_reserve_tokens": 0,
        "trained_artifact_sha256": artifact_sha256,
        "new_controller_requires_full_d128": True,
        "original_d20_reuse_allowed": False,
        "G": "current",
        "B": "B0",
    })
    return value


def _write_shard(package: Path, method_row: Mapping[str, Any], tasks: list[str],
                 part: int, source_state: Mapping[str, Any]) -> dict[str, Any]:
    method = method_row["method"]
    schedule = method_row["shards"][part]
    shard_id = schedule["shard_id"]
    shard_root = package / "shards" / shard_id
    lane_root = shard_root / "lanes" / shard_id
    runtime = lane_root / "runtime"
    source_package: Path = source_state["package"]
    source_lane = source_state["lane"]
    source_lane_root = source_package / "lanes" / source_lane
    lane_root.mkdir(parents=True)
    (shard_root / "history_system").mkdir()
    shutil.copyfile(source_package / "evidence_eval.py", shard_root / "evidence_eval.py")
    patched = _patched_runner_bytes(source_package / "history_system/runner.py")
    (shard_root / "history_system/runner.py").write_bytes(patched)
    _copy_tree(source_package / "sglang", shard_root / "sglang")
    shutil.copyfile(source_package / "sglang_files.json", shard_root / "sglang_files.json")
    _copy_tree(source_lane_root / "runtime", runtime)
    for overlay in method_row["runtime_overlay"]:
        target = runtime / overlay["relative_path"]
        shutil.copyfile(Path(overlay["source_path"]), target)

    artifact = source_state["artifact"]
    controller, gp = _controller(_read(runtime / "configs/controller.json"),
                                 method, artifact)
    _save(runtime / "configs/controller.json", controller)
    task_document = {
        "schema": "a-history-system-task-manifest-v1",
        "manifest_id": f"R2D128_{shard_id}",
        "stage": "development_search",
        "task_ids": tasks,
        "parent_manifest_id": "R2D128",
        "fixed_denominator": 128,
        "automatic_reruns": 0,
    }
    _save(lane_root / "tasks.json", task_document)
    source_lane_value = _read(source_lane_root / "lane.json")
    lane = copy.deepcopy(source_lane_value)
    lane.update({
        "name": shard_id,
        "selector": METHOD_SPECS[method]["selector"],
        "physical_device": schedule["device"],
        "engine_port": schedule["engine_port"],
        "task_port_base": schedule["task_port_base"],
        "task_ports": list(range(schedule["task_port_base"],
                                 schedule["task_port_base"] + len(tasks))),
        "model_smokes": METHOD_SPECS[method]["model_smokes"],
    })
    _save(lane_root / "lane.json", lane)
    source_design = _read(source_lane_root / "design.json")
    design = _design_for_shard(
        source_design, method=method, shard_id=shard_id, tasks=tasks,
        task_sha256=_sha(lane_root / "tasks.json"),
        engine_port=schedule["engine_port"], controller=controller,
        artifact_sha256=method_row["artifact"]["sha256"], runtime=runtime)
    _save(lane_root / "design.json", design)
    _save(lane_root / "gp.json", gp)
    _save(lane_root / "trained_artifact.json", artifact)
    validation = _runtime_bundle_validation(
        runtime, lane_root / "trained_artifact.json",
        runtime / "configs/controller.json", method)
    _save(lane_root / "runtime_validation.json", validation)
    _save(shard_root / "source_design.json", source_design)
    provenance = {
        "schema": "proposal-h0-d128-shard-provenance-v1",
        "method": method,
        "controller": method,
        "shard_id": shard_id,
        "source_package": str(source_package),
        "source_lane": source_lane,
        "source_static_files_sha256": method_row["source"]["static_files_sha256"],
        "source_sglang_files_sha256": method_row["source"]["sglang_files_sha256"],
        "source_design_sha256": method_row["source"]["design_sha256"],
        "source_controller_sha256": method_row["source"]["controller_sha256"],
        "source_runner_sha256": method_row["source"]["runner_sha256"],
        "patched_runner_sha256": method_row["source"]["patched_runner_sha256"],
        "runner_operation": RUNNER_CONTINUE_MARKER,
        "runtime_overlay": copy.deepcopy(method_row["runtime_overlay"]),
        "artifact": copy.deepcopy(method_row["artifact"]),
        "proposal_protocol": PROPOSAL_PROTOCOL,
        "output_algorithm_id": design["candidate_id"],
        "original_d20_reused": False,
        "required_env": STRICT_ENVIRONMENT,
    }
    _save(shard_root / "provenance.json", provenance)
    _save(shard_root / "launch_contract.json", {
        "schema": "proposal-h0-d128-shard-launch-v1",
        "launch_authorized": True,
        "external_package_authorization_required": True,
        "total_task_execution_budget": len(tasks),
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "required_environment": STRICT_ENVIRONMENT,
        "runner_operation": RUNNER_CONTINUE_MARKER,
        "lanes": [lane],
    })
    _save(shard_root / "static_files.json", _manifest(
        shard_root, "proposal-h0-d128-shard-static-files-v1",
        skip_sglang=True, excluded=("static_files.json",)))
    return {
        "shard_id": shard_id,
        "method": method,
        "controller": method,
        "relative_package": f"shards/{shard_id}",
        "task_ids": tasks,
        "task_budget": len(tasks),
        "device": schedule["device"],
        "engine_port": schedule["engine_port"],
        "task_port_base": schedule["task_port_base"],
        "static_files_sha256": _sha(shard_root / "static_files.json"),
    }


def prepare(package: Path, design_path: Path, *, methods: Sequence[str] | None = None
            ) -> dict[str, Any]:
    package = package.resolve()
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite proposal package: {package}")
    design_path = design_path.resolve()
    design = _read(design_path)
    selected, by_method, tasks = validate_design(design, methods=methods)
    catalog_binding = design.get("source_catalog")
    if (not isinstance(catalog_binding, dict)
            or not isinstance(catalog_binding.get("path"), str)
            or not _valid_sha(catalog_binding.get("sha256"))):
        raise ValueError("Design lacks its source-catalog binding")
    catalog_path = Path(catalog_binding["path"]).resolve()
    if not catalog_path.is_file() or _sha(catalog_path) != catalog_binding["sha256"]:
        raise RuntimeError("Frozen source catalog changed after build-design")
    canonical_path = Path(design["canonical_manifest"]["path"]).resolve()
    canonical = _canonical_manifest(canonical_path)
    if canonical["task_ids"] != tasks:
        raise ValueError("Design task order differs from canonical D128")
    sources = {method: _verify_bound_source(by_method[method]) for method in selected}
    package.mkdir(parents=True)
    try:
        shutil.copyfile(Path(__file__), package / Path(__file__).name)
        shutil.copyfile(design_path, package / "design.json")
        shutil.copyfile(canonical_path, package / "tasks.d128.json")
        shard_rows = []
        for method in selected:
            for part in range(6):
                shard_tasks = tasks[part::6]
                if len(shard_tasks) != SHARD_SIZES[part]:
                    raise AssertionError("D128 partition size drift")
                shard_rows.append(_write_shard(
                    package, by_method[method], shard_tasks, part, sources[method]))
        contract = {
            "schema": PACKAGE_SCHEMA,
            "status": "prepared_waiting_for_external_authorization",
            "launch_authorized": False,
            "authorization_schema": AUTHORIZATION_SCHEMA,
            "methods": selected,
            "stage": "H0_R1",
            "invariants": design["invariants"],
            "proposal_protocol": PROPOSAL_PROTOCOL,
            "d128_manifest_sha256": _sha(package / "tasks.d128.json"),
            "design_sha256": _sha(package / "design.json"),
            "shards": shard_rows,
            "new_task_executions": 128 * len(selected),
            "campaign_task_execution_ceiling": 384,
            "automatic_retries": 0,
            "automatic_reruns": 0,
            "failed_task_policy": RUNNER_CONTINUE_MARKER,
            "failed_task_retry_count": 0,
            "required_environment": STRICT_ENVIRONMENT,
            "original_d20_reused": False,
        }
        _save(package / "proposal_contract.json", contract)
        _save(package / "static_files.json", _manifest(
            package, "proposal-h0-d128-static-files-v1",
            skip_sglang=True, excluded=("static_files.json",)))
        verification = verify_package(package)
        return {"schema": "proposal-h0-d128-prepare-v1", "status": "prepared",
                "package": str(package), "verification": verification,
                "authorization_requirements": authorization_requirements(package)}
    except BaseException:
        if package.exists():
            shutil.rmtree(package)
        raise


def _verify_shard(package: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    shard_id = row["shard_id"]
    shard_root = package / "shards" / shard_id
    _verify_manifest(shard_root, _read(shard_root / "static_files.json"),
                     f"{shard_id} static")
    _verify_manifest(shard_root, _read(shard_root / "sglang_files.json"),
                     f"{shard_id} SGLang")
    base = _load_module(shard_root / "evidence_eval.py", "_proposal_shard_")
    base.verify_package(shard_root)
    lane_root = shard_root / "lanes" / shard_id
    lane = _read(lane_root / "lane.json")
    tasks = _read(lane_root / "tasks.json")
    design = _read(lane_root / "design.json")
    gp = _read(lane_root / "gp.json")
    artifact = _read(lane_root / "trained_artifact.json")
    validation = _read(lane_root / "runtime_validation.json")
    provenance = _read(shard_root / "provenance.json")
    method = row["method"]
    spec = METHOD_SPECS[method]
    if (lane.get("name") != shard_id
            or lane.get("selector") != spec["selector"]
            or lane.get("physical_device") != row["device"]
            or lane.get("engine_port") != row["engine_port"]
            or lane.get("task_port_base") != row["task_port_base"]
            or lane.get("task_ports") != list(range(
                row["task_port_base"], row["task_port_base"] + row["task_budget"]))
            or tasks.get("task_ids") != row["task_ids"]
            or design.get("task_ids") != row["task_ids"]
            or design.get("task_manifest_sha256") != _sha(lane_root / "tasks.json")
            or design.get("limits", {}).get("tasks") != row["task_budget"]
            or design.get("automatic_reruns") != 0
            or design.get("runtime", {}).get("sglang_backend_url")
            != f"http://127.0.0.1:{row['engine_port']}"
            or design.get("source_files") != _tree_hashes(lane_root / "runtime")
            or design.get("resolved_configs", {}).get("controller")
            != _read(lane_root / "runtime/configs/controller.json")
            or gp.get("set_selector") != spec["selector"]
            or gp.get("selector_artifact") != artifact
            or gp.get("proposal_protocol") != PROPOSAL_PROTOCOL
            or provenance.get("proposal_protocol") != PROPOSAL_PROTOCOL
            or provenance.get("original_d20_reused") is not False
            or provenance.get("required_env") != STRICT_ENVIRONMENT
            or _sha(shard_root / "history_system/runner.py")
            != provenance.get("patched_runner_sha256")
            or provenance.get("runner_operation") != RUNNER_CONTINUE_MARKER):
        raise ValueError(f"Frozen shard contract differs: {shard_id}")
    _artifact(lane_root / "trained_artifact.json", method)
    observed_validation = _runtime_bundle_validation(
        lane_root / "runtime", lane_root / "trained_artifact.json",
        lane_root / "runtime/configs/controller.json", method)
    if validation != observed_validation:
        raise ValueError(f"Frozen runtime validation differs: {shard_id}")
    for overlay in provenance.get("runtime_overlay", []):
        target = lane_root / "runtime" / overlay["relative_path"]
        if not target.is_file() or _sha(target) != overlay["target_sha256"]:
            raise RuntimeError(f"Frozen runtime overlay changed: {shard_id}")
    return {"shard_id": shard_id, "method": method,
            "task_count": len(row["task_ids"]), "device": row["device"]}


def verify_package(package: Path) -> dict[str, Any]:
    package = package.resolve()
    _verify_manifest(package, _read(package / "static_files.json"), "proposal package")
    contract = _read(package / "proposal_contract.json")
    design = _read(package / "design.json")
    methods, _, canonical_tasks = validate_design(design, methods=contract.get("methods"))
    rows = contract.get("shards")
    if (contract.get("schema") != PACKAGE_SCHEMA
            or contract.get("status") != "prepared_waiting_for_external_authorization"
            or contract.get("launch_authorized") is not False
            or contract.get("stage") != "H0_R1"
            or contract.get("invariants") != {"history": "H0", "R": 1,
                                               "G": "current", "B": "B0"}
            or contract.get("proposal_protocol") != PROPOSAL_PROTOCOL
            or contract.get("d128_manifest_sha256") != EXPECTED_D128_SHA256
            or contract.get("design_sha256") != _sha(package / "design.json")
            or contract.get("new_task_executions") != 128 * len(methods)
            or contract.get("campaign_task_execution_ceiling") != 384
            or contract.get("automatic_retries") != 0
            or contract.get("automatic_reruns") != 0
            or contract.get("failed_task_policy") != RUNNER_CONTINUE_MARKER
            or contract.get("failed_task_retry_count") != 0
            or contract.get("required_environment") != STRICT_ENVIRONMENT
            or contract.get("original_d20_reused") is not False
            or not isinstance(rows, list) or len(rows) != 6 * len(methods)):
        raise ValueError("Malformed frozen proposal package contract")
    if _canonical_manifest(package / "tasks.d128.json")["task_ids"] != canonical_tasks:
        raise ValueError("Packaged D128 task order differs")
    verified = []
    ports: list[int] = []
    for method in methods:
        method_rows = [row for row in rows if row.get("method") == method]
        if [row.get("shard_id") for row in method_rows] != [
                f"{method}_part{part}" for part in range(6)]:
            raise ValueError(f"{method}: shards are missing, duplicated, or reordered")
        owned: list[str] = []
        for part, row in enumerate(method_rows):
            expected = canonical_tasks[part::6]
            if row.get("task_ids") != expected or row.get("task_budget") != len(expected):
                raise ValueError(f"{method}: shard partition differs from canonical D128")
            owned.extend(expected)
            ports.extend([row["engine_port"], *range(
                row["task_port_base"], row["task_port_base"] + len(expected))])
            verified.append(_verify_shard(package, row))
        if owned != [task for part in range(6) for task in canonical_tasks[part::6]]:
            raise AssertionError("Unexpected partition flattening")
        if set(owned) != set(canonical_tasks) or len(owned) != len(set(owned)):
            raise ValueError(f"{method}: tasks are not owned exactly once")
    if len(ports) != len(set(ports)):
        raise ValueError("Frozen shard ports overlap")
    return {"schema": "proposal-h0-d128-package-verification-v1",
            "status": "passed", "package": str(package), "methods": methods,
            "shards": verified, "new_task_executions": 128 * len(methods)}


def authorization_requirements(package: Path) -> dict[str, Any]:
    package = package.resolve()
    verify_package(package)
    contract = _read(package / "proposal_contract.json")
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "explicit_root_authorization_required",
        "launch_authorized": False,
        "required_authorizer": "root",
        "authorization_scope": "proposal_h0_d128_new_controllers",
        "package_static_files_sha256": _sha(package / "static_files.json"),
        "proposal_contract_sha256": _sha(package / "proposal_contract.json"),
        "design_sha256": _sha(package / "design.json"),
        "d128_manifest_sha256": _sha(package / "tasks.d128.json"),
        "authorized_shard_ids": [row["shard_id"] for row in contract["shards"]],
        "new_task_executions": contract["new_task_executions"],
        "campaign_task_execution_ceiling": 384,
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }


def verify_authorization(package: Path, receipt_path: Path) -> dict[str, Any]:
    expected = authorization_requirements(package)
    actual = _read(receipt_path)
    ignored = {"status", "launch_authorized", "required_authorizer"}
    if (actual.get("schema") != AUTHORIZATION_SCHEMA
            or actual.get("status") != "authorized"
            or actual.get("launch_authorized") is not True
            or actual.get("authorized_by") != "root"
            or any(actual.get(key) != value for key, value in expected.items()
                   if key not in ignored)):
        raise ValueError("Launch authorization does not bind this exact package")
    return actual


def run_shard(package: Path, shard_id: str, authorization: Path) -> int:
    package = package.resolve()
    verify_package(package)
    receipt = verify_authorization(package, authorization)
    rows = {row["shard_id"]: row for row in
            _read(package / "proposal_contract.json")["shards"]}
    if shard_id not in rows or shard_id not in receipt["authorized_shard_ids"]:
        raise ValueError(f"Unknown or unauthorized shard: {shard_id}")
    row = rows[shard_id]
    shard_root = package / "shards" / shard_id
    lane_root = shard_root / "lanes" / shard_id
    if (lane_root / "run").exists() or (lane_root / "results").exists():
        raise FileExistsError(f"Refusing rerun of proposal shard {shard_id}")
    base = _load_module(shard_root / "evidence_eval.py", "_proposal_run_")
    lane = _read(lane_root / "lane.json")
    lane_raw = {key: value for key, value in lane.items()
                if key not in {"name", "task_ports"}}
    base.LANES = {shard_id: lane_raw}
    base.lane_specs = lambda: [copy.deepcopy(lane)]
    base.DEFAULT_REMOTE_ROOT = shard_root
    os.environ.update(STRICT_ENVIRONMENT)
    if any(os.environ.get(key) != value for key, value in STRICT_ENVIRONMENT.items()):
        raise RuntimeError("Strict nonfinite environment was not applied")
    base._assert_lane_free(lane)
    return base.run_lane(shard_root, shard_id)


def _binding(path: Path, role: str | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value: dict[str, Any] = {"path": str(path), "sha256": _sha(path),
                             "size_bytes": path.stat().st_size}
    if role is not None:
        value["role"] = role
    return value


def collect_failure_audit(package: Path,
                          *, audit_helper: Path | None = None) -> dict[str, Any]:
    """Use the proven exact CapacityInfeasible collector on proposal shards."""
    package = package.resolve()
    verify_package(package)
    if audit_helper is None:
        try:
            import evidence_d128_failure_audit as audit
        except ImportError as error:
            raise RuntimeError("Capacity failure audit module is unavailable") from error
        helper_path = Path(audit.__file__).resolve()
    else:
        helper_path = audit_helper.resolve()
        if not helper_path.is_file():
            raise FileNotFoundError(helper_path)
        audit = _load_module(helper_path, "_proposal_failure_audit_")
    for name in ("SUPPORTED_CONTROLLERS", "_failure"):
        if not hasattr(audit, name):
            raise ValueError(f"Audit helper lacks required interface: {name}")
    contract_path = package / "proposal_contract.json"
    contract = _read(contract_path)
    package_static = _binding(package / "static_files.json")
    package_contract = _binding(contract_path)
    failures = []
    snapshot = []
    previous = set(audit.SUPPORTED_CONTROLLERS)
    audit.SUPPORTED_CONTROLLERS = set(METHOD_ORDER)
    try:
        for row in contract["shards"]:
            shard_id = row["shard_id"]
            shard = package / "shards" / shard_id
            lane = shard / "lanes" / shard_id
            stage_path = lane / "results/stage_manifest.json"
            if not stage_path.is_file():
                snapshot.append({"shard": shard_id, "stage_manifest": None})
                continue
            stage = _read(stage_path)
            outcomes = stage.get("task_outcomes")
            if not isinstance(outcomes, list):
                raise ValueError(f"Malformed task outcomes: {stage_path}")
            terminal = [item for item in outcomes if isinstance(item, dict)
                        and item.get("outcome") == "runtime_failure_in_denominator"]
            snapshot.append({"shard": shard_id, "status": stage.get("status"),
                             "runtime_failure_task_ids": [item.get("task_id")
                                                           for item in terminal],
                             "stage_manifest": _binding(stage_path)})
            for outcome in terminal:
                failures.append(audit._failure(
                    package, shard, lane, stage_path, stage, outcome,
                    stage_name="H0_R1", controller=row["method"],
                    source_provenance_path=shard / "provenance.json",
                    source_provenance_kind="shard_provenance",
                    source_design_path=shard / "source_design.json",
                    package_static=package_static,
                    package_contract=package_contract))
    finally:
        audit.SUPPORTED_CONTROLLERS = previous
    unknown = [row for row in failures
               if row.get("classification", {}).get("status") != "audited_in_contract"]
    return {
        "schema": AUDIT_SCHEMA,
        "status": "audited" if not unknown else "contains_unknown_failure",
        "verified_at_utc": _now(),
        "verification_mode": "read-only native artifacts; no model calls or reruns",
        "audit_helper": _binding(helper_path),
        "package": {"path": str(package), "stage": "H0_R1",
                    "static_files": package_static, "contract": package_contract},
        "scope": {"fixed_d128_runtime_outcomes": True,
                  "automatic_retry_or_rerun": False,
                  "official_scores_preserved": True,
                  "observed_failure_count": len(failures),
                  "audited_capacity_failure_count": len(failures) - len(unknown),
                  "unknown_failure_count": len(unknown)},
        "failures": failures,
        "all_shards_snapshot": snapshot,
    }


def write_failure_audit(package: Path, output: Path,
                        *, audit_helper: Path | None = None) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite failure audit: {output}")
    value = collect_failure_audit(package, audit_helper=audit_helper)
    _save(output, value)
    return {"status": value["status"], "output": str(output),
            "sha256": _sha(output),
            "observed_failure_count": len(value["failures"])}


def _official_valid(value: object) -> bool:
    return bool(isinstance(value, dict)
                and value.get("scored") is True
                and value.get("n_total") == 1
                and value.get("n_scored") == 1
                and type(value.get("correct_count")) is int
                and value.get("correct_count") in {0, 1}
                and isinstance(value.get("semantic_score"), (int, float))
                and not isinstance(value.get("semantic_score"), bool)
                and math.isfinite(float(value["semantic_score"])))


def _raw_cells(package: Path, row: Mapping[str, Any]) -> list[dict[str, Any]]:
    shard_id = row["shard_id"]
    lane = package / "shards" / shard_id / "lanes" / shard_id
    stage_path = lane / "results/stage_manifest.json"
    if not stage_path.is_file():
        return [{"method": row["method"], "shard_id": shard_id,
                 "task_id": task, "status": "pending",
                 "status_reason": "stage_manifest_missing", "official": None,
                 "evidence": []} for task in row["task_ids"]]
    stage = _read(stage_path)
    outcomes = stage.get("task_outcomes")
    if not isinstance(outcomes, list):
        raise ValueError(f"Malformed task outcomes: {stage_path}")
    by_task: dict[str, Mapping[str, Any]] = {}
    for outcome in outcomes:
        if not isinstance(outcome, dict) or not isinstance(outcome.get("task_id"), str):
            raise ValueError(f"Malformed task outcome: {stage_path}")
        if outcome["task_id"] in by_task:
            raise ValueError(f"Duplicate task outcome: {outcome['task_id']}")
        by_task[outcome["task_id"]] = outcome
    stage_binding = _binding(stage_path, "stage_manifest")
    result = []
    for task in row["task_ids"]:
        outcome = by_task.get(task)
        if outcome is None or outcome.get("outcome") == "not_started":
            result.append({"method": row["method"], "shard_id": shard_id,
                           "task_id": task, "status": "pending",
                           "status_reason": "not_started_or_not_recorded",
                           "official": None, "evidence": [stage_binding]})
            continue
        task_root = lane / "results/task_shards" / task
        final_path = task_root / "server/final.json"
        official_path = task_root / "bfcl/official_summary.json"
        official = _read(official_path) if official_path.is_file() else None
        normal = (outcome.get("outcome") == "official_completed"
                  and outcome.get("runtime_completed") is True
                  and outcome.get("worker_returncode") == 0
                  and outcome.get("server_returncode") == 0
                  and final_path.is_file() and _official_valid(official))
        evidence = [stage_binding]
        if final_path.is_file():
            evidence.append(_binding(final_path, "server_final"))
        if official_path.is_file():
            evidence.append(_binding(official_path, "official_summary"))
        result.append({
            "method": row["method"], "shard_id": shard_id, "task_id": task,
            "status": "normal" if normal else "unknown",
            "status_reason": "official_completed" if normal else str(
                outcome.get("outcome") or "unclassified_runtime_outcome"),
            "runtime_completed": outcome.get("runtime_completed"),
            "worker_returncode": outcome.get("worker_returncode"),
            "server_returncode": outcome.get("server_returncode"),
            "official": official if normal else None,
            "raw_official": official if not normal else None,
            "stage_outcome": copy.deepcopy(dict(outcome)),
            "evidence": evidence,
        })
    extras = set(by_task) - set(row["task_ids"])
    if extras:
        raise ValueError(f"Shard stage contains undeclared tasks: {sorted(extras)}")
    return result


def _evidence_map(value: object) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, dict):
        rows = value.values()
    elif isinstance(value, list):
        rows = value
    else:
        return {}
    result = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("role"), str):
            result[row["role"]] = row
        elif isinstance(row, dict) and isinstance(row.get("path"), str):
            # Collector evidence is a role->binding object; recover its key below.
            continue
    if isinstance(value, dict):
        result.update({role: row for role, row in value.items()
                       if isinstance(role, str) and isinstance(row, dict)})
    return result


def _audit_index(package: Path, audit_path: Path | None,
                 cells: Mapping[tuple[str, str], Mapping[str, Any]],
                 *, audit_helper: Path | None = None,
                 ) -> dict[tuple[str, str], dict[str, Any]]:
    if audit_path is None:
        return {}
    audit_path = audit_path.resolve()
    audit = _read(audit_path)
    helper_binding = audit.get("audit_helper")
    if not isinstance(helper_binding, dict) or not isinstance(
            helper_binding.get("path"), str):
        raise ValueError("Failure audit lacks its helper binding")
    bound_helper = Path(helper_binding["path"]).resolve()
    selected_helper = audit_helper.resolve() if audit_helper is not None else bound_helper
    if (selected_helper != bound_helper or not bound_helper.is_file()
            or _sha(bound_helper) != helper_binding.get("sha256")):
        raise RuntimeError("Failure audit helper changed or differs from the requested helper")
    # Do not trust a document merely because it asserts the right
    # classification.  Re-run the read-only collector over the exact package
    # and require byte-derived failure rows to match.
    recomputed = collect_failure_audit(package, audit_helper=selected_helper)
    if audit.get("failures") != recomputed.get("failures"):
        raise ValueError("Failure audit rows differ from current raw runtime evidence")
    package_binding = audit.get("package")
    if (audit.get("schema") != AUDIT_SCHEMA or not isinstance(package_binding, dict)
            or Path(str(package_binding.get("path", ""))).resolve() != package.resolve()
            or package_binding.get("stage") != "H0_R1"
            or package_binding.get("static_files", {}).get("sha256")
            != _sha(package / "static_files.json")
            or package_binding.get("contract", {}).get("sha256")
            != _sha(package / "proposal_contract.json")):
        raise ValueError("Failure audit is not bound to this exact proposal package")
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    required_roles = {
        "stage_manifest", "steps", "attempts", "server_final",
        "official_summary", "frozen_capacity_exception_source",
        "frozen_capacity_policy_source", "evaluated_design", "source_design",
        "frozen_eval_capacity", "server_startup", "engine_log", "engine_status",
        "package_contract", "package_static_files", "source_provenance",
    }
    for row in audit.get("failures", []):
        if not isinstance(row, dict):
            raise ValueError("Malformed failure audit row")
        method, task, shard = row.get("controller"), row.get("task_id"), row.get("shard")
        key = (method, task)
        source = cells.get(key)
        classification = row.get("classification")
        checks = classification.get("checks") if isinstance(classification, dict) else None
        stage = row.get("stage_outcome")
        binding = row.get("source_binding")
        evidence = _evidence_map(row.get("evidence"))
        if (source is None or source.get("status") != "unknown"
                or source.get("shard_id") != shard
                or row.get("stage") != "H0_R1"
                or row.get("quality_status") != "runtime_failed"
                or not isinstance(stage, dict) or stage != source.get("stage_outcome")
                or stage.get("outcome") != "runtime_failure_in_denominator"
                or not isinstance(classification, dict)
                or classification.get("status") != "audited_in_contract"
                or classification.get("selection_eligible") is not True
                or classification.get("budget_contract_verified") is not True
                or classification.get("family") != "pre_generation_capacity_infeasible"
                or classification.get("exception_type") != "CapacityInfeasible"
                or classification.get("source_exception") != "CapacityInfeasible"
                or not isinstance(checks, dict) or not checks
                or not isinstance(binding, dict)
                or binding.get("controller") != method
                or binding.get("task_id") != task
                or binding.get("shard") != shard
                or binding.get("package_root") != str(package.resolve())
                or binding.get("source_package_static_sha256")
                != _sha(package / "static_files.json")
                or binding.get("source_package_contract_sha256")
                != _sha(package / "proposal_contract.json")
                or not required_roles <= set(evidence)):
            continue
        for role in required_roles:
            item = evidence[role]
            path = Path(str(item.get("path", ""))).resolve()
            if not path.is_file() or _sha(path) != item.get("sha256"):
                raise RuntimeError(f"Audited failure evidence changed: {method}/{task}/{role}")
        stage_sha = next((item.get("sha256") for item in source.get("evidence", [])
                          if item.get("role") == "stage_manifest"), None)
        if evidence["stage_manifest"].get("sha256") != stage_sha:
            raise ValueError(f"Audit stage differs from raw source: {method}/{task}")
        if key in selected:
            raise ValueError(f"Duplicate audited failure: {method}/{task}")
        selected[key] = {"audit_path": str(audit_path),
                         "audit_sha256": _sha(audit_path), "row": row}
    return selected


def summarize(package: Path, *, failure_audit: Path | None = None,
              audit_helper: Path | None = None) -> dict[str, Any]:
    package = package.resolve()
    verification = verify_package(package)
    contract = _read(package / "proposal_contract.json")
    rows = contract["shards"]
    raw = [cell for row in rows for cell in _raw_cells(package, row)]
    by_key = {(cell["method"], cell["task_id"]): cell for cell in raw}
    if len(by_key) != len(raw):
        raise ValueError("Summary contains duplicate method/task ownership")
    audited = _audit_index(package, failure_audit, by_key,
                           audit_helper=audit_helper)
    for key, binding in audited.items():
        cell = by_key[key]
        cell["status"] = "audited_failed"
        cell["status_reason"] = "audited_pre_generation_capacity_infeasible"
        cell["official"] = None
        cell["audit_binding"] = {"path": binding["audit_path"],
                                 "sha256": binding["audit_sha256"]}
    method_summaries = []
    for method in contract["methods"]:
        cells = [by_key[(method, task)] for task in
                 _read(package / "tasks.d128.json")["task_ids"]]
        counts = {status: sum(cell["status"] == status for cell in cells)
                  for status in ("normal", "audited_failed", "unknown", "pending")}
        terminal = counts["pending"] == 0
        fully_audited = terminal and counts["unknown"] == 0
        method_summaries.append({
            "method": method,
            "quality_label": "preliminary, n=1",
            "expected": 128,
            **counts,
            "terminal": terminal,
            "fully_audited": fully_audited,
            "status": "complete" if fully_audited else "incomplete",
            "operational_success_count": sum(
                cell["official"]["correct_count"] for cell in cells
                if cell["status"] == "normal"),
            "quality_cells": cells,
        })
    return {
        "schema": SUMMARY_SCHEMA,
        "generated_at": _now(),
        "package": str(package),
        "package_static_files_sha256": _sha(package / "static_files.json"),
        "proposal_contract_sha256": _sha(package / "proposal_contract.json"),
        "d128_manifest_sha256": _sha(package / "tasks.d128.json"),
        "verification": verification,
        "methods": method_summaries,
        "interpretation": (
            "Only normal cells carry official quality. Audited failures are fixed-denominator "
            "operational non-successes; unknown failures are never reinterpreted as valid."),
    }


def write_summary(package: Path, output: Path,
                  *, failure_audit: Path | None = None,
                  audit_helper: Path | None = None) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {output}")
    value = summarize(package, failure_audit=failure_audit,
                      audit_helper=audit_helper)
    _save(output, value)
    return {"status": "written", "output": str(output), "sha256": _sha(output),
            "methods": [{key: row[key] for key in
                         ("method", "normal", "audited_failed", "unknown", "pending")}
                        for row in value["methods"]]}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("build-design")
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--source-catalog", type=Path, required=True)
    item.add_argument("--canonical-manifest", type=Path, required=True)
    item.add_argument("--runtime-source-root", type=Path, required=True)
    item.add_argument("--overlay-file", action="append", required=True)
    item.add_argument("--methods", nargs="+", choices=METHOD_ORDER, required=True)
    item = sub.add_parser("prepare")
    item.add_argument("--package", type=Path, required=True)
    item.add_argument("--design", type=Path, required=True)
    item.add_argument("--methods", nargs="+", choices=METHOD_ORDER)
    item = sub.add_parser("verify-package")
    item.add_argument("--package", type=Path, required=True)
    item = sub.add_parser("authorization-requirements")
    item.add_argument("--package", type=Path, required=True)
    item = sub.add_parser("run-shard")
    item.add_argument("--package", type=Path, required=True)
    item.add_argument("--shard", required=True)
    item.add_argument("--authorization", type=Path, required=True)
    item = sub.add_parser("audit-capacity-failures")
    item.add_argument("--package", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    item.add_argument("--audit-helper", type=Path)
    item = sub.add_parser("summarize")
    item.add_argument("--package", type=Path, required=True)
    item.add_argument("--failure-audit", type=Path)
    item.add_argument("--audit-helper", type=Path)
    item.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build-design":
        result = build_design(
            args.output, source_catalog_path=args.source_catalog,
            canonical_manifest_path=args.canonical_manifest,
            runtime_source_root=args.runtime_source_root,
            overlay_files=args.overlay_file, methods=args.methods)
    elif args.command == "prepare":
        result = prepare(args.package, args.design, methods=args.methods)
    elif args.command == "verify-package":
        result = verify_package(args.package)
    elif args.command == "authorization-requirements":
        result = authorization_requirements(args.package)
    elif args.command == "run-shard":
        return run_shard(args.package, args.shard, args.authorization)
    elif args.command == "audit-capacity-failures":
        result = write_failure_audit(args.package, args.output,
                                     audit_helper=args.audit_helper)
    elif args.output:
        result = write_summary(args.package, args.output,
                               failure_audit=args.failure_audit,
                               audit_helper=args.audit_helper)
    else:
        result = summarize(args.package, failure_audit=args.failure_audit,
                           audit_helper=args.audit_helper)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True,
                     allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
