"""Continue only pristine never-started tasks from terminal H0/H1 shards."""
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
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import evidence_eval as local_base
import evidence_eval_trained as trained


CONTRACT_SCHEMA = "experiment3-expansion-remainder-v1"
BINDING_SCHEMA = "experiment3-expansion-remainder-source-v1"
AUTHORIZATION_SCHEMA = "experiment3-expansion-remainder-launch-authorization-v1"
OVERLAY_SCHEMA = "experiment3-expansion-remainder-overlay-v1"
DISPATCH_SCHEMA = "experiment3-expansion-remainder-dispatch-v1"
DEVICES = (0, 1, 2, 3, 4, 6)
PREDECESSOR_WAITING_STATES = (
    None,
    "queued",
    "preflight",
    "starting",
    "starting_actor_engine",
    "running",
    "running_fixed_d20",
)
ENGINE_PORT_BASE = 15000
TASK_PORT_BASE = 16000
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
OPERATIONAL_DESIGN_PATHS = (
    ("run_id_template",),
    ("task_ids",),
    ("task_manifest_sha256",),
    ("limits", "tasks"),
    ("runtime", "sglang_backend_url"),
    ("search_contract", "fixed_denominator"),
)


def _read(path: Path) -> dict[str, Any]:
    return local_base._read(path)


def _sha(path: Path) -> str:
    return local_base._sha(path)


def _save(path: Path, value: Mapping[str, Any]) -> None:
    local_base._save(path, value)


def _save_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _save(path, value)


def _save_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_base(path: Path):
    name = "_experiment3_remainder_base_" + hashlib.sha256(
        str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(source, target, ignore=shutil.ignore_patterns(
        "__pycache__", ".pytest_cache", "*.pyc"))


def _without_operational(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    for path in OPERATIONAL_DESIGN_PATHS:
        current: Any = result
        for key in path[:-1]:
            current = current.get(key) if isinstance(current, dict) else None
        if isinstance(current, dict):
            current.pop(path[-1], None)
    return result


def _patched_runner(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    if text.count(RUNNER_STOP_BLOCK) != 1 or RUNNER_CONTINUE_MARKER in text:
        raise ValueError("Frozen runner lacks the exact stop-on-runtime-failure seam")
    return text.replace(RUNNER_STOP_BLOCK, RUNNER_CONTINUE_BLOCK)


def _write_patched_runner(source: Path, target: Path) -> dict[str, str]:
    text = _patched_runner(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8", newline="\n")
    return {"source_sha256": _sha(source), "patched_sha256": _sha(target),
            "operation": RUNNER_CONTINUE_MARKER}


def _origin(source: Path) -> dict[str, Any]:
    source = source.resolve()
    required = (source / "evidence_eval.py", source / "history_system/runner.py",
                source / "static_files.json", source / "sglang_files.json",
                source / "sglang", source / "lanes")
    if any(not path.exists() for path in required):
        raise FileNotFoundError(f"Incomplete terminal source shard: {source}")
    base = _load_base(source / "evidence_eval.py")
    base.verify_package(source)
    lanes = [path for path in (source / "lanes").iterdir()
             if path.is_dir() and (path / "lane.json").is_file()]
    if len(lanes) != 1:
        raise ValueError("Terminal source shard must contain exactly one lane")
    lane_root = lanes[0]
    lane_name = lane_root.name
    paths = {
        "tasks": source / "tasks.json",
        "design": lane_root / "design.json",
        "lane": lane_root / "lane.json",
        "stage": lane_root / "results/stage_manifest.json",
        "status": lane_root / "run/status.json",
        "provenance": source / "provenance.json",
        "launch": source / "launch_contract.json",
    }
    lane_tasks_path = lane_root / "tasks.json"
    if lane_tasks_path.is_file():
        paths["lane_tasks"] = lane_tasks_path
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("Terminal source receipts or frozen inputs are missing")
    tasks = _read(paths["tasks"])
    lane_tasks = _read(lane_tasks_path) if lane_tasks_path.is_file() else None
    design = _read(paths["design"])
    lane = _read(paths["lane"])
    stage = _read(paths["stage"])
    status = _read(paths["status"])
    provenance = _read(paths["provenance"])
    launch = _read(paths["launch"])
    task_ids = tasks.get("task_ids")
    outcomes = stage.get("task_outcomes")
    if (not isinstance(task_ids, list) or not task_ids
            or len(set(task_ids)) != len(task_ids)
            or (lane_tasks is not None and lane_tasks.get("task_ids") != task_ids)
            or design.get("task_ids") != task_ids
            or stage.get("task_ids") != task_ids
            or not isinstance(outcomes, list)
            or [row.get("task_id") for row in outcomes] != task_ids
            or len({row.get("task_id") for row in outcomes}) != len(outcomes)):
        raise ValueError("Terminal source task order or outcomes differ")
    started = [row for row in outcomes if row.get("outcome") != "not_started"]
    remaining = [row for row in outcomes if row.get("outcome") == "not_started"]
    remaining_ids = [row["task_id"] for row in remaining]
    if (not remaining_ids
            or task_ids[-len(remaining_ids):] != remaining_ids
            or stage.get("schema") != "a-history-system-run-v1"
            or stage.get("state") != "failed"
            or not str(stage.get("status", "")).startswith("stopped_")
            or stage.get("task_cells_started") != len(started)
            or stage.get("completed_task_cells") != len(started)
            or stage.get("automatic_retries") != 0
            or stage.get("automatic_reruns") != 0
            or status.get("state") != "failed_no_rerun"
            or status.get("completed_task_cells") != len(started)
            or status.get("automatic_retries") != 0
            or status.get("automatic_reruns") != 0
            or status.get("runner", {}).get("returncode") in {None, 0}):
        raise ValueError("Source shard is not an exact terminal failed tail")
    results = lane_root / "results"
    for row in remaining:
        task_id = row["task_id"]
        native = results / "task_shards" / task_id
        named_paths = [path for path in results.rglob("*") if task_id in path.parts]
        if (row.get("official_summary") is not None
                or row.get("runtime_completed") is not None
                or row.get("worker_returncode") is not None
                or row.get("server_returncode") is not None
                or native.exists() or named_paths):
            raise ValueError(f"Task is not pristine never-started work: {task_id}")
    source_required_env = provenance.get("required_env", {})
    launch_required_env = launch.get("required_environment", source_required_env)
    if (not isinstance(source_required_env, dict)
            or launch_required_env != source_required_env):
        raise ValueError("Source required environment binding differs")
    return {
        "source": source, "base": base, "lane_root": lane_root,
        "lane_name": lane_name, "tasks": tasks, "design": design, "lane": lane,
        "stage": stage, "status": status, "provenance": provenance,
        "remaining": remaining, "remaining_ids": remaining_ids,
        "required_env": source_required_env,
        "hashes": {key: _sha(path) for key, path in paths.items()},
        "static_files_sha256": _sha(source / "static_files.json"),
        "sglang_files_sha256": _sha(source / "sglang_files.json"),
        "evidence_eval_sha256": _sha(source / "evidence_eval.py"),
        "runner_sha256": _sha(source / "history_system/runner.py"),
    }


def _lane_payload_hashes(lane_root: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for name in ("runtime", "model_artifacts"):
        path = lane_root / name
        if path.is_dir():
            rows.update({f"{name}/{key}": value
                         for key, value in local_base._tree_hashes(path).items()})
    for name in ("gp.json", "trained_artifact.json"):
        path = lane_root / name
        if path.is_file():
            rows[name] = _sha(path)
    return rows


def _clone_design(source: Mapping[str, Any], tasks: Sequence[str], manifest_id: str,
                  manifest_sha256: str, engine_port: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(source))
    result["run_id_template"] = manifest_id
    result["task_ids"] = list(tasks)
    result["task_manifest_sha256"] = manifest_sha256
    result["limits"] = {**result["limits"], "tasks": len(tasks)}
    result["runtime"] = {**result["runtime"],
                         "sglang_backend_url": f"http://127.0.0.1:{engine_port}"}
    result["search_contract"] = {**result["search_contract"],
                                 "fixed_denominator": manifest_id}
    if _without_operational(source) != _without_operational(result):
        raise ValueError("Remainder design changed scientific configuration")
    return result


def _schedule(device: int, wave: int, task_count: int) -> dict[str, Any]:
    if not 0 < task_count < 100:
        raise ValueError("Remainder lane task count must fit its fixed port block")
    if device not in DEVICES:
        raise ValueError("Source shard uses a reserved or unknown physical device")
    return {"physical_device": device, "wave": wave,
            "engine_port": ENGINE_PORT_BASE + device * 10,
            "task_port_base": TASK_PORT_BASE + device * 100}


def _predecessor_binding(status_path: Path | None) -> dict[str, Any] | None:
    if status_path is None:
        return None
    status_path = status_path.resolve()
    if (status_path.name != "status.json" or status_path.parent.name != "run"
            or status_path.parents[2].name != "lanes"):
        raise ValueError("Predecessor must be an exact lane run/status.json path")
    lane_root = status_path.parents[1]
    package = status_path.parents[3]
    static = package / "static_files.json"
    design = lane_root / "design.json"
    lane = lane_root / "lane.json"
    if not static.is_file() or not design.is_file() or not lane.is_file():
        raise FileNotFoundError("Predecessor frozen package identity is incomplete")
    lane_value = _read(lane)
    device = lane_value.get("physical_device")
    if device not in DEVICES:
        raise ValueError("Predecessor uses a reserved or unknown physical device")
    return {"package": str(package),
            "package_static_files_sha256": _sha(static),
            "lane": lane_root.name, "status": str(status_path),
            "design_sha256": _sha(design), "lane_sha256": _sha(lane),
            "physical_device": device,
            "terminal_states": ["completed", "failed_no_rerun"],
            "automatic_retries": 0, "automatic_reruns": 0}


def _predecessor_terminal(binding: Mapping[str, Any] | None) -> bool:
    if binding is None:
        return True
    package = Path(binding["package"]).resolve()
    lane_root = package / "lanes" / binding["lane"]
    if (_sha(package / "static_files.json")
            != binding.get("package_static_files_sha256")
            or _sha(lane_root / "design.json") != binding.get("design_sha256")
            or _sha(lane_root / "lane.json") != binding.get("lane_sha256")):
        raise RuntimeError("Predecessor frozen package identity changed")
    status_path = Path(binding["status"]).resolve()
    if not status_path.is_file():
        return False
    status = _read(status_path)
    if (status.get("schema") != "evidence-sets-eval-lane-status-v1"
            or status.get("lane") != binding["lane"]
            or status.get("automatic_retries") != 0
            or status.get("automatic_reruns") != 0):
        raise RuntimeError("Predecessor status receipt is malformed")
    state = status.get("state")
    if state in binding["terminal_states"]:
        return True
    if state in PREDECESSOR_WAITING_STATES:
        return False
    raise RuntimeError(f"Predecessor stopped in an unknown state: {state}")


def _write_lane(package: Path, origin: Mapping[str, Any], schedule: Mapping[str, Any],
                predecessor: Mapping[str, Any] | None) -> dict[str, Any]:
    source_lane = origin["lane_name"]
    lane_id = f"{source_lane}.remaining"
    task_ids = origin["remaining_ids"]
    lane_root = package / "lanes" / lane_id
    lane_root.mkdir(parents=True)
    _copy_tree(origin["lane_root"] / "runtime", lane_root / "runtime")
    for name in ("model_artifacts",):
        path = origin["lane_root"] / name
        if path.is_dir():
            _copy_tree(path, lane_root / name)
    for name in ("gp.json", "trained_artifact.json"):
        path = origin["lane_root"] / name
        if path.is_file():
            shutil.copyfile(path, lane_root / name)
    tasks = copy.deepcopy(origin["tasks"])
    tasks.update(manifest_id=lane_id, task_ids=list(task_ids),
                 parent_source_shard=str(origin["source"]),
                 parent_source_lane=source_lane)
    _save(lane_root / "tasks.json", tasks)
    lane = copy.deepcopy(origin["lane"])
    lane.update(name=lane_id, **schedule,
                task_ports=list(range(schedule["task_port_base"],
                                      schedule["task_port_base"] + len(task_ids))))
    _save(lane_root / "lane.json", lane)
    design = _clone_design(origin["design"], task_ids, lane_id,
                           _sha(lane_root / "tasks.json"), lane["engine_port"])
    _save(lane_root / "design.json", design)
    _save(lane_root / "source_design.json", origin["design"])
    binding = {
        "schema": BINDING_SCHEMA, "continuation_lane": lane_id,
        "source_shard": str(origin["source"]), "source_lane": source_lane,
        "source_static_files_sha256": origin["static_files_sha256"],
        "source_sglang_files_sha256": origin["sglang_files_sha256"],
        "source_design_sha256": origin["hashes"]["design"],
        "source_lane_sha256": origin["hashes"]["lane"],
        "source_tasks_sha256": origin["hashes"]["tasks"],
        "source_lane_tasks_sha256": origin["hashes"].get("lane_tasks"),
        "source_stage_manifest_sha256": origin["hashes"]["stage"],
        "source_status_sha256": origin["hashes"]["status"],
        "source_runner_sha256": origin["runner_sha256"],
        "source_task_ids": origin["tasks"]["task_ids"],
        "source_outcomes": copy.deepcopy(origin["remaining"]),
        "remaining_task_ids": list(task_ids),
        "required_env": copy.deepcopy(origin["required_env"]),
        "lane_payload_hashes": _lane_payload_hashes(origin["lane_root"]),
        "operational_runner_change": RUNNER_CONTINUE_MARKER,
        "automatic_retries": 0, "automatic_reruns": 0,
    }
    _save(lane_root / "remainder_source_binding.json", binding)
    return {"shard_id": lane_id, "source_shard": str(origin["source"]),
            "source_lane": source_lane, "controller": origin["provenance"].get("controller"),
            "task_ids": list(task_ids), "task_count": len(task_ids),
            "device": lane["physical_device"], "wave": schedule["wave"],
            "engine_port": lane["engine_port"],
            "task_port_base": lane["task_port_base"],
            "required_env": copy.deepcopy(origin["required_env"]),
            "predecessor": copy.deepcopy(predecessor),
            "source_stage_manifest_sha256": origin["hashes"]["stage"],
            "source_status_sha256": origin["hashes"]["status"]}


def prepare(package: Path, source_shards: Sequence[Path],
            predecessor_statuses: Sequence[Path] | None = None) -> dict[str, Any]:
    package = package.resolve()
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite remainder package: {package}")
    if not source_shards:
        raise ValueError("At least one terminal source shard is required")
    predecessors = list(predecessor_statuses or [])
    if predecessors and len(predecessors) != len(source_shards):
        raise ValueError("Each source shard requires one corresponding predecessor status")
    if not predecessors:
        predecessors = [None] * len(source_shards)
    origins = [_origin(path) for path in source_shards]
    if len({row["source"] for row in origins}) != len(origins):
        raise ValueError("Terminal source shard is repeated")
    common = ("evidence_eval_sha256", "runner_sha256", "sglang_files_sha256")
    if any(origin[key] != origins[0][key] for origin in origins for key in common):
        raise ValueError("Mixed source runtime versions require separate remainder packages")
    lane_ids = [f"{origin['lane_name']}.remaining" for origin in origins]
    if len(set(lane_ids)) != len(lane_ids):
        raise ValueError("Continuation lane IDs collide")
    predecessor_bindings = [_predecessor_binding(path) if path is not None else None
                            for path in predecessors]
    waves: dict[int, int] = {}
    schedules = []
    for origin, predecessor in zip(origins, predecessor_bindings):
        device = origin["lane"].get("physical_device")
        if predecessor is not None and predecessor["physical_device"] != device:
            raise ValueError("Source and predecessor must bind the same physical device")
        wave = waves.get(device, 0)
        schedules.append(_schedule(device, wave, len(origin["remaining_ids"])))
        waves[device] = wave + 1
    package.mkdir(parents=True)
    try:
        shutil.copyfile(origins[0]["source"] / "evidence_eval.py",
                        package / "evidence_eval.py")
        shutil.copyfile(Path(__file__), package / Path(__file__).name)
        shutil.copyfile(Path(trained.__file__), package / "evidence_eval_trained.py")
        _copy_tree(origins[0]["source"] / "sglang", package / "sglang")
        shutil.copyfile(origins[0]["source"] / "sglang_files.json",
                        package / "sglang_files.json")
        runner_patch = _write_patched_runner(
            origins[0]["source"] / "history_system/runner.py",
            package / "history_system/runner.py")
        rows = [_write_lane(package, origin, schedules[index],
                            predecessor_bindings[index])
                for index, origin in enumerate(origins)]
        contract = {
            "schema": CONTRACT_SCHEMA,
            "status": "prepared_waiting_for_exact_root_authorization",
            "launch_authorized": False,
            "authorization_schema": AUTHORIZATION_SCHEMA,
            "devices": list(DEVICES), "shards": rows,
            "source_shard_count": len(origins),
            "total_task_execution_budget": sum(row["task_count"] for row in rows),
            "budget_semantics": "only_original_not_started_cells_once",
            "runner_patch": runner_patch,
            "automatic_retries": 0, "automatic_reruns": 0,
        }
        _save(package / "continuation_contract.json", contract)
        _save(package / "launch_contract.json", {
            "schema": "experiment3-expansion-remainder-launch-v1",
            "launch_authorized": True,
            "external_exact_root_authorization_required": True,
            "lanes": rows,
            "total_task_execution_budget": contract["total_task_execution_budget"],
            "runner_patch": runner_patch,
            "automatic_retries": 0, "automatic_reruns": 0,
        })
        _save(package / "static_files.json", local_base._static_manifest(package))
        verification = verify_package(package)
        return {"schema": "experiment3-expansion-remainder-prepare-v1",
                "status": "prepared", "package": str(package),
                "verification": verification,
                "authorization_requirements": authorization_requirements(package)}
    except BaseException:
        if package.exists():
            shutil.rmtree(package)
        raise


def _verify_lane(package: Path, row: Mapping[str, Any]) -> None:
    lane_id = row["shard_id"]
    lane_root = package / "lanes" / lane_id
    binding = _read(lane_root / "remainder_source_binding.json")
    origin = _origin(Path(binding.get("source_shard", "")))
    tasks = _read(lane_root / "tasks.json")
    lane = _read(lane_root / "lane.json")
    design = _read(lane_root / "design.json")
    source_design = _read(lane_root / "source_design.json")
    expected = _clone_design(source_design, origin["remaining_ids"], lane_id,
                             _sha(lane_root / "tasks.json"), row["engine_port"])
    if (binding.get("schema") != BINDING_SCHEMA
            or binding.get("continuation_lane") != lane_id
            or binding.get("source_lane") != origin["lane_name"]
            or binding.get("source_static_files_sha256") != origin["static_files_sha256"]
            or binding.get("source_sglang_files_sha256") != origin["sglang_files_sha256"]
            or binding.get("source_design_sha256") != origin["hashes"]["design"]
            or binding.get("source_lane_sha256") != origin["hashes"]["lane"]
            or binding.get("source_tasks_sha256") != origin["hashes"]["tasks"]
            or binding.get("source_lane_tasks_sha256")
            != origin["hashes"].get("lane_tasks")
            or binding.get("source_stage_manifest_sha256") != origin["hashes"]["stage"]
            or binding.get("source_status_sha256") != origin["hashes"]["status"]
            or binding.get("source_runner_sha256") != origin["runner_sha256"]
            or binding.get("source_outcomes") != origin["remaining"]
            or binding.get("remaining_task_ids") != origin["remaining_ids"]
            or binding.get("required_env") != origin["required_env"]
            or binding.get("operational_runner_change") != RUNNER_CONTINUE_MARKER
            or binding.get("automatic_retries") != 0
            or binding.get("automatic_reruns") != 0
            or binding.get("lane_payload_hashes") != _lane_payload_hashes(origin["lane_root"])
            or _lane_payload_hashes(lane_root) != binding["lane_payload_hashes"]
            or tasks.get("task_ids") != origin["remaining_ids"]
            or source_design != origin["design"]
            or design != expected
            or lane.get("name") != lane_id
            or lane.get("physical_device") != row["device"]
            or lane.get("engine_port") != row["engine_port"]
            or lane.get("task_port_base") != row["task_port_base"]
            or lane.get("task_ports") != list(range(
                row["task_port_base"], row["task_port_base"] + row["task_count"]))
            or row.get("task_ids") != origin["remaining_ids"]
            or row.get("task_count") != len(origin["remaining_ids"])
            or row.get("required_env") != origin["required_env"]
            or row.get("source_stage_manifest_sha256") != origin["hashes"]["stage"]
            or row.get("source_status_sha256") != origin["hashes"]["status"]
            or (row.get("predecessor") is not None
                and row.get("predecessor") != _predecessor_binding(
                    Path(row["predecessor"]["status"])))):
        raise ValueError(f"Remainder lane differs from frozen source: {lane_id}")


def verify_package(package: Path) -> dict[str, Any]:
    package = package.resolve()
    local_base.verify_package(package)
    contract = _read(package / "continuation_contract.json")
    launch = _read(package / "launch_contract.json")
    rows = contract.get("shards")
    patch = contract.get("runner_patch")
    if (contract.get("schema") != CONTRACT_SCHEMA
            or contract.get("launch_authorized") is not False
            or contract.get("automatic_retries") != 0
            or contract.get("automatic_reruns") != 0
            or contract.get("devices") != list(DEVICES)
            or not isinstance(rows, list) or not rows
            or len({row.get("shard_id") for row in rows}) != len(rows)
            or not isinstance(patch, dict)
            or patch.get("operation") != RUNNER_CONTINUE_MARKER
            or _sha(package / "history_system/runner.py") != patch.get("patched_sha256")
            or launch.get("external_exact_root_authorization_required") is not True
            or launch.get("runner_patch") != patch
            or launch.get("lanes") != rows
            or launch.get("automatic_retries") != 0
            or launch.get("automatic_reruns") != 0):
        raise ValueError("Invalid expansion remainder contract")
    total = 0
    ports_by_wave: dict[int, list[int]] = {}
    for row in rows:
        if (row.get("device") not in DEVICES or row.get("device") in {5, 7}
                or type(row.get("wave")) is not int or row["wave"] < 0):
            raise ValueError("Invalid remainder device schedule")
        _verify_lane(package, row)
        total += row["task_count"]
        ports_by_wave.setdefault(row["wave"], []).extend([
            row["engine_port"], *range(row["task_port_base"],
                                       row["task_port_base"] + row["task_count"])])
    if (total != contract.get("total_task_execution_budget")
            or total != launch.get("total_task_execution_budget")):
        raise ValueError("Remainder execution budget differs from original pending tasks")
    for wave, ports in ports_by_wave.items():
        active = [row for row in rows if row["wave"] == wave]
        if (len({row["device"] for row in active}) != len(active)
                or len(ports) != len(set(ports))):
            raise ValueError(f"Remainder wave {wave} overlaps devices or ports")
    return {"schema": "experiment3-expansion-remainder-verification-v1",
            "status": "passed", "package": str(package),
            "source_shard_count": len(rows),
            "total_task_execution_budget": total}


def authorization_requirements(package: Path) -> dict[str, Any]:
    package = package.resolve()
    contract = _read(package / "continuation_contract.json")
    return {"schema": AUTHORIZATION_SCHEMA,
            "status": "explicit_root_authorization_required",
            "launch_authorized": False, "required_authorizer": "root",
            "authorization_scope": "experiment3_expansion_original_not_started_only",
            "package_static_files_sha256": _sha(package / "static_files.json"),
            "continuation_contract_sha256": _sha(
                package / "continuation_contract.json"),
            "authorized_shard_ids": [row["shard_id"] for row in contract["shards"]],
            "total_task_execution_budget": contract["total_task_execution_budget"],
            "automatic_retries": 0, "automatic_reruns": 0}


def verify_authorization(package: Path, receipt_path: Path) -> dict[str, Any]:
    expected = authorization_requirements(package)
    actual = _read(receipt_path)
    bound = {key: value for key, value in expected.items()
             if key not in {"status", "launch_authorized", "required_authorizer"}}
    if (actual.get("schema") != AUTHORIZATION_SCHEMA
            or actual.get("status") != "authorized"
            or actual.get("launch_authorized") is not True
            or actual.get("authorized_by") != "root"
            or any(actual.get(key) != value for key, value in bound.items())):
        raise ValueError("Authorization does not bind this exact remainder package")
    return actual


def _base_lane(package: Path, row: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    base = _load_base(package / "evidence_eval.py")
    lane = _read(package / "lanes" / row["shard_id"] / "lane.json")
    return base, lane


def run_shard(package: Path, shard_id: str, authorization: Path) -> int:
    package = package.resolve()
    verify_package(package)
    receipt = verify_authorization(package, authorization)
    rows = {row["shard_id"]: row for row in
            _read(package / "continuation_contract.json")["shards"]}
    if shard_id not in rows or shard_id not in receipt["authorized_shard_ids"]:
        raise ValueError(f"Unknown or unauthorized remainder shard: {shard_id}")
    row = rows[shard_id]
    lane_root = package / "lanes" / shard_id
    if (lane_root / "run").exists() or (lane_root / "results").exists():
        raise FileExistsError(f"Refusing rerun of remainder shard {shard_id}")
    base, lane = _base_lane(package, row)
    lane_raw = {key: value for key, value in lane.items()
                if key not in {"name", "task_ports"}}
    base.LANES = {shard_id: lane_raw}
    base.lane_specs = lambda: [copy.deepcopy(lane)]
    base.DEFAULT_REMOTE_ROOT = package
    for key, value in row.get("required_env", {}).items():
        os.environ[key] = value
    base._assert_lane_free(lane)
    trained.ascend_environment()
    trained.enable_strict_sampling()
    return base.run_lane(package, shard_id)


def _evidence(path: Path) -> dict[str, str] | None:
    return {"path": str(path), "sha256": _sha(path)} if path.is_file() else None


def _official_valid(value: Any) -> bool:
    score = value.get("semantic_score") if isinstance(value, dict) else None
    return bool(isinstance(value, dict) and value.get("scored") is True
                and value.get("n_total") == 1 and value.get("n_scored") == 1
                and type(value.get("correct_count")) is int
                and value.get("correct_count") in {0, 1}
                and isinstance(score, (int, float)) and not isinstance(score, bool)
                and math.isfinite(float(score)))


def _strict_total(value: Any, key: str) -> int | float | None:
    item = value.get(key) if isinstance(value, dict) else None
    total = item.get("strict_total") if isinstance(item, dict) else None
    return total if isinstance(total, (int, float)) and not isinstance(total, bool) else None


def _measured_cost(final: Mapping[str, Any], official: Mapping[str, Any]) -> dict[str, Any]:
    summary = final.get("cost_summary")
    summary = summary if isinstance(summary, dict) else {}
    costs = summary.get("costs")
    costs = costs if isinstance(costs, dict) else {}
    resident = costs.get("openai_resident_usage")
    resident = resident if isinstance(resident, dict) else {}
    work = costs.get("actual_model_work")
    work = work if isinstance(work, dict) else {}
    flat = final.get("cost")
    flat = flat if isinstance(flat, dict) else {}

    def number(value: Any) -> int | float | None:
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    attempts = summary.get("generation_attempts", flat.get("generation_attempts"))
    return {"wall_seconds": number(final.get("wall_seconds")),
            "gold_checker_seconds": number(official.get("total_gold_checker_seconds")),
            "handler_http_calls": official.get("total_handler_http_calls")
            if type(official.get("total_handler_http_calls")) is int else None,
            "generation_attempts": attempts if type(attempts) is int else None,
            "prompt_tokens": _strict_total(resident, "prompt_tokens")
            if resident else number(flat.get("prompt_tokens")),
            "completion_tokens": _strict_total(resident, "completion_tokens")
            if resident else number(flat.get("completion_tokens")),
            "total_tokens": _strict_total(resident, "total_tokens")
            if resident else number(flat.get("total_tokens")),
            "materialized_encoder_tokens": _strict_total(
                work, "materialized_encoder_tokens")
            if work else number(flat.get("materialized_encoder_tokens"))}


def overlay(package: Path) -> dict[str, Any]:
    package = package.resolve()
    verify_package(package)
    contract = _read(package / "continuation_contract.json")
    rows_out = []
    for row in contract["shards"]:
        lane_id = row["shard_id"]
        lane_root = package / "lanes" / lane_id
        binding = _read(lane_root / "remainder_source_binding.json")
        stage_path = lane_root / "results/stage_manifest.json"
        status_path = lane_root / "run/status.json"
        stage = _read(stage_path) if stage_path.is_file() else {}
        outcomes = stage.get("task_outcomes", [])
        if outcomes and [item.get("task_id") for item in outcomes] != row["task_ids"]:
            raise ValueError(f"Continuation outcomes changed task order: {lane_id}")
        by_task = {item.get("task_id"): item for item in outcomes}
        if len(by_task) != len(outcomes):
            raise ValueError(f"Continuation outcomes repeat tasks: {lane_id}")
        source_outcomes = {item["task_id"]: item for item in binding["source_outcomes"]}
        for task_id in row["task_ids"]:
            outcome = by_task.get(task_id)
            task_root = lane_root / "results/task_shards" / task_id
            official_path = task_root / "bfcl/official_summary.json"
            final_path = task_root / "server/final.json"
            official = _read(official_path) if official_path.is_file() else None
            final = _read(final_path) if final_path.is_file() else {}
            if outcome is None or outcome.get("outcome") == "not_started":
                status, reason = "pending", "continuation_not_started"
            elif (outcome.get("outcome") == "official_completed"
                  and outcome.get("runtime_completed") is True
                  and outcome.get("worker_returncode") == 0
                  and outcome.get("server_returncode") == 0
                  and final_path.is_file() and _official_valid(official)):
                status, reason = "completed", "official_completed"
            else:
                status, reason = "runtime_failed", str(outcome.get("outcome"))
            artifacts = [item for item in (
                _evidence(official_path), _evidence(final_path),
                _evidence(stage_path), _evidence(status_path)) if item is not None]
            rows_out.append({
                "source_shard": binding["source_shard"],
                "source_lane": binding["source_lane"], "task_id": task_id,
                "source_stage_manifest_sha256": binding[
                    "source_stage_manifest_sha256"],
                "source_status_sha256": binding["source_status_sha256"],
                "source_outcome": source_outcomes[task_id],
                "continuation_lane": lane_id,
                "status": status, "status_reason": reason,
                "stage_outcome": copy.deepcopy(outcome),
                "official": copy.deepcopy(official),
                "measured_cost": _measured_cost(final, official or {}),
                "evidence": artifacts,
            })
    pending = sum(row["status"] == "pending" for row in rows_out)
    return {"schema": OVERLAY_SCHEMA,
            "status": "completed" if pending == 0 else "partial_pending",
            "package": str(package),
            "continuation_contract_sha256": _sha(
                package / "continuation_contract.json"),
            "total_original_not_started_cells": len(rows_out),
            "pending_cells": pending, "rows": rows_out,
            "automatic_retries": 0, "automatic_reruns": 0}


def write_overlay(package: Path, output: Path) -> dict[str, Any]:
    value = overlay(package)
    _save_new(output.resolve(), value)
    return value


def _occupied(error: RuntimeError) -> bool:
    message = str(error)
    return "acquired by PIDs" in message or "ports are occupied" in message


def dispatch(package: Path, authorization: Path, *, poll_seconds: float = 20.0,
             popen: Any = subprocess.Popen) -> int:
    package, authorization = package.resolve(), authorization.resolve()
    verify_package(package)
    verify_authorization(package, authorization)
    state_path, log_root = package / "dispatch.json", package / "dispatch_logs"
    overlay_path = package / "continuation.json"
    if state_path.exists() or log_root.exists() or overlay_path.exists():
        raise FileExistsError("Refusing to overwrite remainder dispatch state")
    rows = _read(package / "continuation_contract.json")["shards"]
    log_root.mkdir()
    state = {"schema": DISPATCH_SCHEMA, "status": "running",
             "package": str(package), "authorization": str(authorization),
             "created_at": _now(), "automatic_retries": 0, "automatic_reruns": 0,
             "shards": [{**row, "status": "queued", "pid": None,
                          "returncode": None} for row in rows]}
    _save_atomic(state_path, state)
    pending = {row["shard_id"] for row in rows}
    running: dict[str, tuple[Any, Any]] = {}
    while pending or running:
        for row in state["shards"]:
            shard_id = row["shard_id"]
            if shard_id not in pending:
                continue
            earlier = [other for other in state["shards"]
                       if other["device"] == row["device"]
                       and other["wave"] < row["wave"]]
            if any(other["status"] in {"queued", "running"} for other in earlier):
                continue
            try:
                if not _predecessor_terminal(row.get("predecessor")):
                    continue
                base, lane = _base_lane(package, row)
                base._assert_lane_free(lane)
            except RuntimeError as error:
                if _occupied(error):
                    continue
                row.update(status="dispatch_failed_no_retry", error=str(error),
                           finished_at=_now())
                pending.remove(shard_id)
                _save_atomic(state_path, state)
                continue
            command = [sys.executable, str(package / Path(__file__).name), "run-shard",
                       "--package", str(package), "--shard", shard_id,
                       "--authorization", str(authorization)]
            log = (log_root / f"{shard_id}.log").open(
                "x", encoding="utf-8", newline="\n")
            try:
                process = popen(command, cwd=package, env=os.environ.copy(),
                                stdin=subprocess.DEVNULL, stdout=log,
                                stderr=subprocess.STDOUT)
            except Exception as error:
                log.close()
                row.update(status="dispatch_failed_no_retry", error=str(error),
                           finished_at=_now(), log=str(log.name))
                pending.remove(shard_id)
                _save_atomic(state_path, state)
                continue
            row.update(status="running", pid=process.pid, command=command,
                       started_at=_now(), log=str(log.name))
            pending.remove(shard_id)
            running[shard_id] = (process, log)
            _save_atomic(state_path, state)
        for shard_id, (process, log) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            row = next(item for item in state["shards"] if item["shard_id"] == shard_id)
            row.update(status="completed" if code == 0 else "failed_no_retry",
                       returncode=code, finished_at=_now())
            del running[shard_id]
            _save_atomic(state_path, state)
        if pending or running:
            time.sleep(poll_seconds)
    failed = [row for row in state["shards"] if row["status"] != "completed"]
    state.update(status="completed" if not failed else "partial_failed_no_retry",
                 finished_at=_now(), failed_shards=[row["shard_id"] for row in failed])
    _save_atomic(state_path, state)
    write_overlay(package, overlay_path)
    return 0 if not failed else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("prepare")
    item.add_argument("--package", type=Path, required=True)
    item.add_argument("--source-shard", type=Path, action="append", required=True)
    item.add_argument("--predecessor-status", type=Path, action="append")
    item = sub.add_parser("verify-package")
    item.add_argument("--package", type=Path, required=True)
    item = sub.add_parser("authorization-requirements")
    item.add_argument("--package", type=Path, required=True)
    item = sub.add_parser("run-shard")
    item.add_argument("--package", type=Path, required=True)
    item.add_argument("--shard", required=True)
    item.add_argument("--authorization", type=Path, required=True)
    item = sub.add_parser("dispatch")
    item.add_argument("--package", type=Path, required=True)
    item.add_argument("--authorization", type=Path, required=True)
    item.add_argument("--poll-seconds", type=float, default=20.0)
    item = sub.add_parser("write-overlay")
    item.add_argument("--package", type=Path, required=True)
    item.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.package, args.source_shard, args.predecessor_status)
    elif args.command == "verify-package":
        result = verify_package(args.package)
    elif args.command == "authorization-requirements":
        result = authorization_requirements(args.package)
    elif args.command == "run-shard":
        return run_shard(args.package, args.shard, args.authorization)
    elif args.command == "dispatch":
        return dispatch(args.package, args.authorization,
                        poll_seconds=args.poll_seconds)
    else:
        result = write_overlay(args.package, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
