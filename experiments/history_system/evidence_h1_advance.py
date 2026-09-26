"""Advance one exact Experiment 3 H0 D128 package to its H1 successor.

The module never chooses controllers or experiment semantics.  ``prepare``
freezes a conditional contract for the already selected C0/C5 H0v2 inputs.
``run`` requires an exact root authorization for that contract, waits only
while the bound H0 dispatcher is demonstrably alive, produces native D128
receipts, freezes the H1-only package, derives its package authorization from
the frozen builder, and starts the dispatcher once.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evidence_eval_combinations as combinations
import evidence_expansion_summary as expansion_summary


CONTRACT_SCHEMA = "experiment3-h0-h1-conditional-v1"
AUTHORIZATION_SCHEMA = "experiment3-h0-h1-conditional-authorization-v1"
RECEIPT_SCHEMA = "experiment3-h0-h1-advance-receipt-v1"
STARTED_SCHEMA = "experiment3-h1-started-v1"
H0_STARTED_SCHEMA = "experiment3-d128-started-v1"
H0_DISPATCH_SCHEMA = "experiment3-expansion-dispatch-v1"
TOOLING_FREEZE_SCHEMA = "experiment3-expansion-tooling-freeze-v1"
SOURCE_CATALOG_SCHEMA = "experiment3-source-catalog-v1"
H0_SOURCE_BINDINGS_SCHEMA = "experiment3-expansion-source-bindings-v1"
CONTROLLERS = ("C0", "C5")
H1_TASK_EXECUTION_BUDGET = 256
GATE_COMPLETE_H0 = "complete_h0_d128"
GATE_D20_PROMOTION = "d20_promotion_device_available"
HEX64 = re.compile(r"[0-9a-f]{64}")
REQUIRED_TOOL_FILES = (
    "evidence_h1_advance.py",
    "evidence_eval_combinations.py",
    "evidence_expansion_summary.py",
    "evidence_expansion_dispatch.py",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _save(path: Path, value: Mapping[str, Any], *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite temporary file: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _paths(target: Path, h0_package: Path) -> dict[str, Path]:
    return {
        "h0_dispatch": h0_package / "dispatch.json",
        "h0_started": h0_package.with_name(h0_package.name + ".started.json"),
        "summary": h0_package.with_name(h0_package.name + ".summary"),
        "spec": target.with_name(target.name + ".build_spec.json"),
        "h1_authorization": target.with_name(target.name + ".authorization.json"),
        "started": target.with_name(target.name + ".started.json"),
        "log": target.with_name(target.name + ".dispatch.log"),
        "receipt": target.with_name(target.name + ".advance.json"),
    }


def _verify_tooling(root: Path, expected_freeze_sha256: str) -> dict[str, Any]:
    freeze_path = root / "freeze.json"
    if not _valid_sha(expected_freeze_sha256):
        raise ValueError("Tooling freeze SHA256 is invalid")
    if not freeze_path.is_file() or _sha(freeze_path) != expected_freeze_sha256:
        raise ValueError("Tooling freeze differs from its explicit binding")
    freeze = _read(freeze_path)
    files = freeze.get("files")
    if (freeze.get("schema") != TOOLING_FREEZE_SCHEMA
            or freeze.get("status") != "cpu_ready"
            or freeze.get("model_calls") != 0
            or freeze.get("evaluation_started") is not False
            or not isinstance(files, dict)):
        raise ValueError("Tooling freeze is not a CPU-only ready freeze")
    for relative in REQUIRED_TOOL_FILES:
        expected = files.get(relative)
        path = root / relative
        if not _valid_sha(expected) or not path.is_file() or _sha(path) != expected:
            raise ValueError(f"Frozen tooling file differs: {relative}")
    loaded = {
        "evidence_h1_advance.py": Path(__file__).resolve(),
        "evidence_eval_combinations.py": Path(combinations.__file__).resolve(),
        "evidence_expansion_summary.py": Path(expansion_summary.__file__).resolve(),
    }
    for relative, path in loaded.items():
        if _sha(path) != files[relative]:
            raise RuntimeError(f"Loaded helper differs from tooling freeze: {relative}")
    return freeze


def _catalog_sources(catalog_path: Path, base: Path) -> dict[str, dict[str, Any]]:
    catalog = _read(catalog_path)
    bindings = catalog.get("bindings")
    if (catalog.get("schema") != SOURCE_CATALOG_SCHEMA
            or catalog.get("role") != "available_sources_not_promotion"
            or catalog.get("selected") != []
            or catalog.get("requires_completed_d20_promotion_before_use") is not True
            or not isinstance(bindings, dict)):
        raise ValueError("Source catalog is not the frozen availability catalog")
    expected_source = (base / "eval_failed_repair_v1").resolve()
    result = {}
    for controller in CONTROLLERS:
        binding = bindings.get(controller)
        if (not isinstance(binding, dict)
                or binding.get("controller") != controller
                or binding.get("source_id") != "failed_repair_v1"
                or Path(binding.get("source_package", "")).resolve() != expected_source
                or binding.get("source_lane") != controller
                or not isinstance(binding.get("source_algorithm_id"), str)
                or not binding["source_algorithm_id"]):
            raise ValueError(
                f"{controller}: source catalog does not bind eval_failed_repair_v1")
        for key in ("source_static_files_sha256", "source_sglang_files_sha256",
                    "source_design_sha256", "source_controller_sha256"):
            if not _valid_sha(binding.get(key)):
                raise ValueError(f"{controller}: invalid source catalog hash {key}")
        result[controller] = dict(binding)
    return result


def _verify_h0_static(package: Path, sources: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    required = ("static_files.json", "expansion_contract.json", "readiness.json",
                "source_bindings.json", "evidence_eval_expansion.py")
    if any(not (package / name).is_file() for name in required):
        raise FileNotFoundError("H0 package is not a frozen expansion package")
    contract = _read(package / "expansion_contract.json")
    readiness = _read(package / "readiness.json")
    source_bindings = _read(package / "source_bindings.json")
    if (contract.get("schema") != expansion_summary.EXPANSION_PACKAGE_SCHEMA
            or contract.get("selected_controllers") != list(CONTROLLERS)
            or contract.get("total_task_execution_budget") != 216
            or contract.get("automatic_retries") != 0
            or contract.get("automatic_reruns") != 0
            or readiness.get("promotion", {}).get("selected") != list(CONTROLLERS)
            or source_bindings.get("schema") != H0_SOURCE_BINDINGS_SCHEMA
            or set(source_bindings.get("bindings", {})) != set(CONTROLLERS)):
        raise ValueError("H0 package is not the exact C0/C5 remaining108 package")
    for controller in CONTROLLERS:
        observed = source_bindings["bindings"][controller]
        expected = sources[controller]
        if any(observed.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{controller}: H0 source binding differs from source catalog")
    shard_ids = [row.get("shard_id") for row in contract.get("shards", [])]
    expected_shards = [f"{controller}_part{part}"
                       for controller in CONTROLLERS for part in range(3)]
    if shard_ids != expected_shards:
        raise ValueError("H0 shard schedule differs from the six fixed C0/C5 shards")
    return contract


def _verify_d20_promotion(path: Path, h0_package: Path) -> dict[str, Any]:
    promotion = _read(path)
    if (promotion.get("schema") != expansion_summary.EXPANSION_READINESS_SCHEMA
            or promotion.get("phase") != "promotion_ready"
            or promotion.get("launch_authorized") is not False
            or promotion.get("promotion", {}).get("phase") != "promotion_ready"
            or promotion.get("promotion", {}).get("selected") != list(CONTROLLERS)):
        raise ValueError("D20 promotion receipt does not select exact C0/C5 in order")
    copied = h0_package / "readiness.json"
    if not copied.is_file() or _sha(copied) != _sha(path):
        raise ValueError("H0 package does not bind the exact D20 promotion receipt")
    return promotion


def _verify_started(path: Path, h0_package: Path) -> dict[str, Any]:
    started = _read(path)
    command = started.get("command")
    package_index = (command.index("--package")
                     if isinstance(command, list) and "--package" in command
                     else -1)
    authorization_index = (command.index("--authorization")
                           if isinstance(command, list) and "--authorization" in command
                           else -1)
    if (started.get("schema") != H0_STARTED_SCHEMA
            or started.get("status") != "dispatcher_started"
            or type(started.get("pid")) is not int
            or started["pid"] <= 0
            or not isinstance(command, list)
            or not all(isinstance(value, str) and value for value in command)
            or package_index < 0 or package_index + 1 >= len(command)
            or command[package_index + 1] != str(h0_package)
            or authorization_index < 0 or authorization_index + 1 >= len(command)
            or started.get("selected") != list(CONTROLLERS)
            or started.get("task_execution_budget") != 216
            or started.get("automatic_retries") != 0
            or started.get("automatic_reruns") != 0):
        raise ValueError("H0 started receipt does not bind the exact H0v2 dispatcher")
    return started


def prepare_contract(*, contract_path: Path, h0_package: Path,
                     source_catalog: Path, d128_manifest: Path,
                     config_source_package: Path, tooling_root: Path,
                     tooling_freeze_sha256: str, target: Path,
                     gate: str = GATE_COMPLETE_H0,
                     d20_promotion_receipt: Path | None = None) -> dict[str, Any]:
    paths = [path.resolve() for path in (
        contract_path, h0_package, source_catalog, d128_manifest,
        config_source_package, tooling_root, target)]
    (contract_path, h0_package, source_catalog, d128_manifest,
     config_source_package, tooling_root, target) = paths
    base = config_source_package.parent
    expected = {
        "contract": base / "d128_h1_v1.conditional.json",
        "h0": base / "d128_h0_v2",
        "catalog": base / "d128_h0_inputs_v2/source_catalog.json",
        "config": base / "prepared_v8",
        "d128": base / "prepared_v8/configs/D128.json",
        "target": base / "d128_h1_v1",
    }
    observed = {"contract": contract_path, "h0": h0_package,
                "catalog": source_catalog, "config": config_source_package,
                "d128": d128_manifest, "target": target}
    for key, value in expected.items():
        if observed[key] != value.resolve():
            raise ValueError(f"Fixed Experiment 3 path differs: {key}")
    if tooling_root.parent != base or not tooling_root.name.startswith("expansion_tools_v"):
        raise ValueError("Tooling root must be an explicitly versioned sibling package")
    if contract_path.exists():
        raise FileExistsError(f"Conditional contract already exists: {contract_path}")
    generated = _paths(target, h0_package)
    successor_paths = [path for key, path in generated.items()
                       if not key.startswith("h0_")]
    duplicates = [str(path) for path in (target, *successor_paths) if path.exists()]
    if duplicates:
        raise FileExistsError(f"Successor state already exists: {duplicates}")
    if not d128_manifest.is_file():
        raise FileNotFoundError(d128_manifest)
    d128 = _read(d128_manifest)
    task_ids = d128.get("task_ids")
    if not isinstance(task_ids, list) or len(task_ids) != 128 or len(set(task_ids)) != 128:
        raise ValueError("Fixed D128 manifest must contain 128 unique task IDs")
    evidence_sets = config_source_package / "history_system/evidence_sets.py"
    if not evidence_sets.is_file():
        raise FileNotFoundError(evidence_sets)
    _verify_tooling(tooling_root, tooling_freeze_sha256)
    sources = _catalog_sources(source_catalog, base)
    h0_contract = _verify_h0_static(h0_package, sources)
    started = _verify_started(generated["h0_started"], h0_package)
    if gate not in {GATE_COMPLETE_H0, GATE_D20_PROMOTION}:
        raise ValueError(f"Unknown H1 advance gate: {gate}")
    promotion_binding = None
    if gate == GATE_D20_PROMOTION:
        if d20_promotion_receipt is None:
            raise ValueError("D20 promotion H1 gate requires its exact receipt")
        promotion_path = d20_promotion_receipt.resolve()
        expected_promotion = (base / "d128_h0_inputs_v2/readiness.json").resolve()
        if promotion_path != expected_promotion:
            raise ValueError("Fixed D20 promotion receipt path differs")
        _verify_d20_promotion(promotion_path, h0_package)
        promotion_binding = {"path": str(promotion_path),
                             "sha256": _sha(promotion_path)}
    elif d20_promotion_receipt is not None:
        raise ValueError("Complete-H0 gate must not add a D20 promotion receipt")
    value = {
        "schema": CONTRACT_SCHEMA,
        "status": "prepared_not_authorized",
        "launch_authorized": False,
        "prepared_at": _now(),
        "stage": "H0_to_H1_only",
        "gate": gate,
        "h1_controllers": list(CONTROLLERS),
        "h1_task_execution_budget": H1_TASK_EXECUTION_BUDGET,
        "r3_task_execution_budget": 0,
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "h0": {
            "package": str(h0_package),
            "package_static_files_sha256": _sha(h0_package / "static_files.json"),
            "expansion_contract_sha256": _sha(h0_package / "expansion_contract.json"),
            "readiness_sha256": _sha(h0_package / "readiness.json"),
            "source_bindings_sha256": _sha(h0_package / "source_bindings.json"),
            "dispatch": str(generated["h0_dispatch"]),
            "started_receipt": str(generated["h0_started"]),
            "started_receipt_sha256": _sha(generated["h0_started"]),
            "authorization": started["command"][
                started["command"].index("--authorization") + 1],
            "authorization_sha256": _sha(Path(started["command"][
                started["command"].index("--authorization") + 1])),
            "dispatcher_pid": started["pid"],
            "dispatcher_command": started["command"],
            "shard_ids": [row["shard_id"] for row in h0_contract["shards"]],
        },
        "source_catalog": {"path": str(source_catalog),
                           "sha256": _sha(source_catalog)},
        "d20_promotion_receipt": promotion_binding,
        "d128_manifest": {"path": str(d128_manifest),
                          "sha256": _sha(d128_manifest)},
        "config_source_package": str(config_source_package),
        "source_evidence_sets_sha256": _sha(evidence_sets),
        "tooling": {"root": str(tooling_root),
                    "freeze": str(tooling_root / "freeze.json"),
                    "freeze_sha256": tooling_freeze_sha256},
        "target": str(target),
        "generated_paths": {key: str(path) for key, path in generated.items()},
        "delegated_action": (
            "prepare_authorize_and_dispatch_exact_h1_once_under_bound_gate"),
    }
    _save(contract_path, value, exclusive=True)
    return {"contract": value,
            "authorization_requirements": authorization_requirements(contract_path)}


def authorization_requirements(contract_path: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve()
    contract = _read(contract_path)
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError("Unknown H0 to H1 conditional contract schema")
    result = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "explicit_root_authorization_required",
        "launch_authorized": False,
        "required_authorizer": "root",
        "authorization_scope": "experiment3_exact_h0_v2_to_h1_v1_once",
        "conditional_contract_sha256": _sha(contract_path),
        "h0_package": contract["h0"]["package"],
        "h0_package_static_files_sha256": contract["h0"][
            "package_static_files_sha256"],
        "h0_started_receipt_sha256": contract["h0"]["started_receipt_sha256"],
        "h0_authorization_sha256": contract["h0"]["authorization_sha256"],
        "gate": contract["gate"],
        "source_catalog_sha256": contract["source_catalog"]["sha256"],
        "d128_manifest_sha256": contract["d128_manifest"]["sha256"],
        "source_evidence_sets_sha256": contract["source_evidence_sets_sha256"],
        "tooling_freeze_sha256": contract["tooling"]["freeze_sha256"],
        "target": contract["target"],
        "h1_controllers": contract["h1_controllers"],
        "h1_task_execution_budget": contract["h1_task_execution_budget"],
        "r3_task_execution_budget": 0,
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "delegated_action": contract["delegated_action"],
    }
    if contract.get("d20_promotion_receipt") is not None:
        result["d20_promotion_receipt_sha256"] = contract[
            "d20_promotion_receipt"]["sha256"]
    return result


def verify_authorization(contract_path: Path, authorization_path: Path) -> dict[str, Any]:
    requirements = authorization_requirements(contract_path)
    actual = _read(authorization_path)
    bound = {key: value for key, value in requirements.items()
             if key not in {"status", "launch_authorized", "required_authorizer"}}
    if (actual.get("schema") != AUTHORIZATION_SCHEMA
            or actual.get("status") != "authorized"
            or actual.get("launch_authorized") is not True
            or actual.get("authorized_by") != "root"
            or any(actual.get(key) != value for key, value in bound.items())):
        raise ValueError("Root authorization does not bind this exact H0 to H1 contract")
    return actual


def _verify_contract(contract_path: Path) -> dict[str, Any]:
    contract = _read(contract_path)
    if (contract.get("schema") != CONTRACT_SCHEMA
            or contract.get("status") != "prepared_not_authorized"
            or contract.get("launch_authorized") is not False
            or contract.get("stage") != "H0_to_H1_only"
            or contract.get("h1_controllers") != list(CONTROLLERS)
            or contract.get("h1_task_execution_budget") != H1_TASK_EXECUTION_BUDGET
            or contract.get("r3_task_execution_budget") != 0
            or contract.get("automatic_retries") != 0
            or contract.get("automatic_reruns") != 0):
        raise ValueError("Conditional contract changes the fixed H1-only semantics")
    gate = contract.get("gate")
    if gate not in {GATE_COMPLETE_H0, GATE_D20_PROMOTION}:
        raise ValueError("Conditional contract has an unknown H1 gate")
    h0 = Path(contract["h0"]["package"]).resolve()
    target = Path(contract["target"]).resolve()
    base = Path(contract["config_source_package"]).resolve().parent
    if h0 != (base / "d128_h0_v2").resolve() or target != (base / "d128_h1_v1").resolve():
        raise ValueError("Conditional contract no longer binds H0v2 to H1v1")
    immutable = {
        h0 / "static_files.json": contract["h0"]["package_static_files_sha256"],
        h0 / "expansion_contract.json": contract["h0"]["expansion_contract_sha256"],
        h0 / "readiness.json": contract["h0"]["readiness_sha256"],
        h0 / "source_bindings.json": contract["h0"]["source_bindings_sha256"],
        Path(contract["h0"]["started_receipt"]): contract["h0"][
            "started_receipt_sha256"],
        Path(contract["h0"]["authorization"]): contract["h0"][
            "authorization_sha256"],
        Path(contract["source_catalog"]["path"]): contract["source_catalog"]["sha256"],
        Path(contract["d128_manifest"]["path"]): contract["d128_manifest"]["sha256"],
        Path(contract["config_source_package"]) / "history_system/evidence_sets.py":
            contract["source_evidence_sets_sha256"],
        Path(contract["tooling"]["freeze"]): contract["tooling"]["freeze_sha256"],
    }
    promotion_binding = contract.get("d20_promotion_receipt")
    if gate == GATE_D20_PROMOTION:
        if not isinstance(promotion_binding, dict):
            raise ValueError("D20 promotion gate lost its receipt binding")
        promotion_path = Path(promotion_binding.get("path", ""))
        immutable[promotion_path] = promotion_binding.get("sha256")
    elif promotion_binding is not None:
        raise ValueError("Complete-H0 gate acquired an unexpected promotion receipt")
    for path, expected in immutable.items():
        if not path.is_file() or not _valid_sha(expected) or _sha(path) != expected:
            raise ValueError(f"Conditional input differs from its frozen hash: {path}")
    tooling_root = Path(contract["tooling"]["root"]).resolve()
    _verify_tooling(tooling_root, contract["tooling"]["freeze_sha256"])
    sources = _catalog_sources(Path(contract["source_catalog"]["path"]), base)
    _verify_h0_static(h0, sources)
    _verify_started(Path(contract["h0"]["started_receipt"]), h0)
    if gate == GATE_D20_PROMOTION:
        _verify_d20_promotion(Path(promotion_binding["path"]), h0)
    return contract


def _process_matches(pid: int, command: Sequence[str], proc_root: Path) -> bool:
    process = proc_root / str(pid)
    try:
        state = (process / "stat").read_text(encoding="utf-8").split()[2]
        observed = [part.decode("utf-8") for part in
                    (process / "cmdline").read_bytes().split(b"\0") if part]
    except (FileNotFoundError, PermissionError, UnicodeDecodeError, IndexError):
        return False
    return state != "Z" and observed == list(command)


def _read_h0_dispatch(contract: Mapping[str, Any]) -> dict[str, Any]:
    h0 = contract["h0"]
    dispatch_path = Path(h0["dispatch"])
    if not dispatch_path.is_file():
        raise RuntimeError("Bound H0 dispatcher has no dispatch state")
    dispatch = _read(dispatch_path)
    if (dispatch.get("schema") != H0_DISPATCH_SCHEMA
            or Path(dispatch.get("package", "")).resolve()
            != Path(h0["package"]).resolve()
            or Path(dispatch.get("authorization", "")).resolve()
            != Path(h0["authorization"]).resolve()
            or dispatch.get("automatic_retries") != 0
            or dispatch.get("automatic_reruns") != 0
            or [row.get("shard_id") for row in dispatch.get("shards", [])]
            != h0["shard_ids"]):
        raise RuntimeError("Bound H0 dispatch state is unknown or changed")
    return dispatch


def _observe_h0_without_quality_gate(contract: Mapping[str, Any], *,
                                     proc_root: Path) -> dict[str, Any]:
    dispatch = _read_h0_dispatch(contract)
    status = dispatch.get("status")
    if status == "running":
        h0 = contract["h0"]
        if not _process_matches(h0["dispatcher_pid"], h0["dispatcher_command"],
                                proc_root):
            raise RuntimeError("Bound H0 dispatcher is not alive with its recorded command")
    elif status not in {"completed", "partial_failed_no_retry"}:
        raise RuntimeError(f"Bound H0 dispatch has an unknown state: {status}")
    return dispatch


def _wait_for_h0(contract: Mapping[str, Any], *, poll_seconds: float,
                 proc_root: Path) -> dict[str, Any]:
    h0 = contract["h0"]
    while True:
        dispatch = _read_h0_dispatch(contract)
        status = dispatch.get("status")
        if status == "running":
            if not _process_matches(h0["dispatcher_pid"], h0["dispatcher_command"],
                                    proc_root):
                raise RuntimeError("Bound H0 dispatcher is not alive with its recorded command")
            time.sleep(poll_seconds)
            continue
        if status != "completed":
            raise RuntimeError(f"Bound H0 dispatch stopped without completion: {status}")
        rows = dispatch["shards"]
        if any(row.get("status") != "completed" or row.get("returncode") != 0
               for row in rows):
            raise RuntimeError("Bound H0 dispatch contains a failed or unknown shard")
        return dispatch


def _validate_summary(index: Mapping[str, Any], summary_root: Path,
                      contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = index.get("receipts")
    if (index.get("schema") != expansion_summary.INDEX_SCHEMA
            or Path(index.get("package", "")).resolve()
            != Path(contract["h0"]["package"]).resolve()
            or not isinstance(rows, list)
            or [row.get("receipt_id") for row in rows] != list(CONTROLLERS)):
        raise RuntimeError("Native H0 summary index is incomplete or unexpected")
    catalog = _read(Path(contract["source_catalog"]["path"]))["bindings"]
    receipts = {}
    for row in rows:
        controller = row["receipt_id"]
        path = Path(row.get("path", "")).resolve()
        try:
            path.relative_to(summary_root.resolve())
        except ValueError as error:
            raise RuntimeError("Native H0 receipt escaped the summary directory") from error
        if (not path.is_file() or not _valid_sha(row.get("sha256"))
                or _sha(path) != row["sha256"]):
            raise RuntimeError(f"{controller}: native H0 receipt hash differs")
        receipt = _read(path)
        evidence = receipt.get("evidence")
        cells = receipt.get("quality_cells")
        if (receipt.get("schema") != combinations.COMPLETE_RESULT_SCHEMA
                or receipt.get("stage") != "H0_R1"
                or receipt.get("status") != "completed"
                or receipt.get("controller") != controller
                or receipt.get("source_algorithm_id")
                != catalog[controller]["source_algorithm_id"]
                or receipt.get("task_manifest_sha256")
                != contract["d128_manifest"]["sha256"]
                or receipt.get("expected_task_cells") != 128
                or receipt.get("completed_task_cells") != 128
                or receipt.get("runtime_failed_task_cells") != 0
                or receipt.get("pending_task_cells") != 0
                or not isinstance(evidence, list) or not evidence
                or any(not isinstance(item, dict) or not item.get("path")
                       or not _valid_sha(item.get("sha256")) for item in evidence)
                or not isinstance(cells, list) or len(cells) != 128):
            raise RuntimeError(f"{controller}: native H0 summary is not completed128")
        for cell in cells:
            cell_evidence = cell.get("evidence")
            if (cell.get("status") != "completed"
                    or cell.get("runtime_completed") is not True
                    or cell.get("worker_returncode") != 0
                    or cell.get("server_returncode") != 0
                    or not isinstance(cell_evidence, list) or not cell_evidence
                    or any(not isinstance(item, dict) or not item.get("path")
                           or not _valid_sha(item.get("sha256"))
                           for item in cell_evidence)):
                raise RuntimeError(f"{controller}: native H0 summary has an unknown cell")
        receipts[controller] = {"path": str(path), "sha256": row["sha256"],
                                "document": receipt}
    return receipts


def _build_spec(contract: Mapping[str, Any],
                receipts: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    catalog = _read(Path(contract["source_catalog"]["path"]))["bindings"]
    config_source = Path(contract["config_source_package"]).resolve()
    source_package = (config_source.parent / "eval_failed_repair_v1").resolve()
    sources = []
    for controller in CONTROLLERS:
        binding = dict(catalog[controller])
        if Path(binding["source_package"]).resolve() != source_package:
            raise RuntimeError(f"{controller}: H0 result/source catalog identity differs")
        binding.update({
            "schema": combinations.SOURCE_SCHEMA,
            "config_source_package": str(config_source),
            "source_evidence_sets_sha256": contract["source_evidence_sets_sha256"],
        })
        if contract["gate"] == GATE_COMPLETE_H0:
            if (receipts is None
                    or binding["source_algorithm_id"]
                    != receipts[controller]["document"]["source_algorithm_id"]):
                raise RuntimeError(
                    f"{controller}: H0 result/source catalog identity differs")
            binding["complete_d128_receipt"] = {
                "path": receipts[controller]["path"],
                "sha256": receipts[controller]["sha256"],
            }
        sources.append(binding)
    spec = {
        "schema": combinations.SPEC_SCHEMA,
        "d128_manifest": dict(contract["d128_manifest"]),
        "h1_sources": sources,
        "c1_h1_calibrations": {},
    }
    if contract["gate"] == GATE_D20_PROMOTION:
        spec["h1_d20_promotion_receipt"] = dict(
            contract["d20_promotion_receipt"])
    return spec


def _stop(receipt_path: Path, receipt: dict[str, Any], phase: str,
          error: Exception) -> int:
    receipt.update(status=f"stopped_on_{phase}", finished_at=_now(),
                   error_type=type(error).__name__, error=str(error))
    _save(receipt_path, receipt)
    return 2


def advance(contract_path: Path, authorization_path: Path, *,
            poll_seconds: float = 20.0, proc_root: Path = Path("/proc"),
            popen: Any = subprocess.Popen) -> int:
    if not 0 <= poll_seconds <= 30:
        raise ValueError("poll_seconds must be between 0 and 30")
    contract_path = contract_path.resolve()
    authorization_path = authorization_path.resolve()
    contract = _verify_contract(contract_path)
    root_authorization = verify_authorization(contract_path, authorization_path)
    paths = {key: Path(value).resolve()
             for key, value in contract["generated_paths"].items()}
    receipt_path = paths["receipt"]
    if receipt_path.exists():
        raise FileExistsError(f"Advance receipt already exists: {receipt_path}")
    receipt = {
        "schema": RECEIPT_SCHEMA, "status": "checking_bound_h0",
        "created_at": _now(), "conditional_contract": str(contract_path),
        "conditional_contract_sha256": _sha(contract_path),
        "root_authorization": str(authorization_path),
        "root_authorization_sha256": _sha(authorization_path),
        "target": contract["target"], "automatic_retries": 0,
        "automatic_reruns": 0, "r3_started": False,
    }
    _save(receipt_path, receipt, exclusive=True)
    phase = "duplicate_successor_state"
    try:
        duplicates = [str(path) for key, path in paths.items()
                      if key not in {"receipt", "h0_dispatch", "h0_started"}
                      and path.exists()]
        target = Path(contract["target"]).resolve()
        if target.exists() or duplicates:
            raise FileExistsError(
                f"Refusing duplicate successor state: {[str(target)] + duplicates}")
        phase = "h0_dispatch"
        completed = None
        if contract["gate"] == GATE_D20_PROMOTION:
            dispatch = _observe_h0_without_quality_gate(
                contract, proc_root=proc_root)
            receipt.update(
                status="preparing_exact_h1",
                gate=GATE_D20_PROMOTION,
                h0_dispatch_observed_status=dispatch.get("status"),
                h0_quality_gate_used=False,
                d20_promotion_receipt_sha256=contract[
                    "d20_promotion_receipt"]["sha256"],
            )
        else:
            dispatch = _wait_for_h0(contract, poll_seconds=poll_seconds,
                                    proc_root=proc_root)
            receipt.update(status="summarizing_bound_h0",
                           gate=GATE_COMPLETE_H0,
                           h0_dispatch_finished_at=dispatch.get("finished_at"))
            _save(receipt_path, receipt)
            phase = "h0_native_summary"
            index = expansion_summary.write_receipts(
                Path(contract["h0"]["package"]), paths["summary"])
            completed = _validate_summary(index, paths["summary"], contract)
            receipt.update(
                status="preparing_exact_h1",
                h0_quality_gate_used=True,
                h0_summary_index_sha256=_sha(paths["summary"] / "index.json"),
                h0_completed_task_cells={name: 128 for name in CONTROLLERS},
                h0_unknown_task_cells={name: 0 for name in CONTROLLERS})
        _save(receipt_path, receipt)
        phase = "h1_preparation"
        spec = _build_spec(contract, completed)
        _save(paths["spec"], spec, exclusive=True)
        verification = combinations.prepare(target, paths["spec"])
        if (verification.get("status") != "passed"
                or verification.get("h1_task_execution_budget")
                != H1_TASK_EXECUTION_BUDGET
                or verification.get("r3_task_execution_budget") != 0
                or verification.get("shard_count") != 6):
            raise RuntimeError("Prepared H1 package differs from the fixed H1-only budget")
        phase = "h1_authorization"
        package_authorization = combinations.authorization_requirements(target)
        if (package_authorization.get("h1_task_execution_budget")
                != H1_TASK_EXECUTION_BUDGET
                or package_authorization.get("r3_task_execution_budget") != 0
                or package_authorization.get("total_task_execution_budget")
                != H1_TASK_EXECUTION_BUDGET
                or package_authorization.get("automatic_retries") != 0
                or package_authorization.get("automatic_reruns") != 0):
            raise RuntimeError("H1 package authorization requirements changed the contract")
        package_authorization.update({
            "status": "authorized", "launch_authorized": True,
            "authorized_by": "root", "authorized_at": _now(),
            "delegated_by_schema": AUTHORIZATION_SCHEMA,
            "delegated_by_authorization_sha256": _sha(authorization_path),
            "delegated_action": root_authorization["delegated_action"],
        })
        _save(paths["h1_authorization"], package_authorization, exclusive=True)
        combinations.verify_authorization(target, paths["h1_authorization"])
        phase = "h1_dispatch_launch"
        command = [
            sys.executable,
            str(Path(contract["tooling"]["root"]) / "evidence_expansion_dispatch.py"),
            "--package", str(target),
            "--authorization", str(paths["h1_authorization"]),
            "--poll-seconds", str(poll_seconds),
        ]
        started = {
            "schema": STARTED_SCHEMA, "status": "dispatch_starting",
            "created_at": _now(), "command": command, "package": str(target),
            "package_static_files_sha256": package_authorization[
                "package_static_files_sha256"],
            "authorization": str(paths["h1_authorization"]),
            "authorization_sha256": _sha(paths["h1_authorization"]),
            "conditional_authorization_sha256": _sha(authorization_path),
            "h1_controllers": list(CONTROLLERS),
            "task_execution_budget": H1_TASK_EXECUTION_BUDGET,
            "automatic_retries": 0, "automatic_reruns": 0,
        }
        _save(paths["started"], started, exclusive=True)
        log = paths["log"].open("x", encoding="utf-8", newline="\n")
        try:
            process = popen(command, cwd=Path(contract["tooling"]["root"]),
                            env=os.environ.copy(), stdin=subprocess.DEVNULL,
                            stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
        except Exception:
            log.close()
            raise
        immediate = process.poll()
        if immediate is not None:
            log.close()
            raise RuntimeError(f"H1 dispatcher exited during launch: {immediate}")
        log.close()
        started.update(status="dispatcher_started", started_at=_now(),
                       pid=process.pid, log=str(paths["log"]))
        _save(paths["started"], started)
        receipt.update(
            status="h1_dispatcher_started", finished_at=_now(),
            h1_package_static_files_sha256=package_authorization[
                "package_static_files_sha256"],
            h1_authorization_sha256=_sha(paths["h1_authorization"]),
            h1_started_receipt_sha256=_sha(paths["started"]),
            h1_dispatcher_pid=process.pid,
        )
        _save(receipt_path, receipt)
        return 0
    except Exception as error:
        return _stop(receipt_path, receipt, phase, error)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--contract", type=Path, required=True)
    prepare.add_argument("--h0-package", type=Path, required=True)
    prepare.add_argument("--source-catalog", type=Path, required=True)
    prepare.add_argument("--d128-manifest", type=Path, required=True)
    prepare.add_argument("--config-source-package", type=Path, required=True)
    prepare.add_argument("--tooling-root", type=Path, required=True)
    prepare.add_argument("--tooling-freeze-sha256", required=True)
    prepare.add_argument("--target", type=Path, required=True)
    prepare.add_argument("--launch-gate", choices=(GATE_COMPLETE_H0,
                                                    GATE_D20_PROMOTION),
                         default=GATE_COMPLETE_H0)
    prepare.add_argument("--d20-promotion-receipt", type=Path)
    requirements = sub.add_parser("authorization-requirements")
    requirements.add_argument("--contract", type=Path, required=True)
    verify = sub.add_parser("verify-authorization")
    verify.add_argument("--contract", type=Path, required=True)
    verify.add_argument("--authorization", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--contract", type=Path, required=True)
    run.add_argument("--authorization", type=Path, required=True)
    run.add_argument("--poll-seconds", type=float, default=20.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_contract(
            contract_path=args.contract, h0_package=args.h0_package,
            source_catalog=args.source_catalog, d128_manifest=args.d128_manifest,
            config_source_package=args.config_source_package,
            tooling_root=args.tooling_root,
            tooling_freeze_sha256=args.tooling_freeze_sha256,
            target=args.target, gate=args.launch_gate,
            d20_promotion_receipt=args.d20_promotion_receipt)
    elif args.command == "authorization-requirements":
        result = authorization_requirements(args.contract)
    elif args.command == "verify-authorization":
        result = verify_authorization(args.contract, args.authorization)
    else:
        return advance(args.contract, args.authorization,
                       poll_seconds=args.poll_seconds)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
