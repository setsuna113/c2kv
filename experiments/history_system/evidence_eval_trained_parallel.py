"""Prepare and run two exact 10-task shards for each trained D20 lane.

The canonical 20-task package is built once by ``evidence_eval_trained``.  Its
two shard packages preserve the frozen runtime, controller, fitted artifact,
and SGLang tree.  Only the task subset, physical device, and ports differ.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence

import evidence_eval as base
import evidence_eval_trained as trained


SCHEMA = "evidence-sets-trained-d20-parallel-v1"
SOURCE_BINDING_SCHEMA = "evidence-sets-trained-d20-shard-binding-v1"
AGGREGATE_SCHEMA = "evidence-sets-trained-d20-parallel-result-v1"
LANE_SHARDS = {
    "C1": (
        {"physical_device": 0, "engine_port": 37600, "task_port_base": 37610},
        {"physical_device": 4, "engine_port": 37630, "task_port_base": 37640},
    ),
    "C4_turn": (
        {"physical_device": 1, "engine_port": 37700, "task_port_base": 37710},
        {"physical_device": 6, "engine_port": 37730, "task_port_base": 37740},
    ),
    "C4_task": (
        {"physical_device": 2, "engine_port": 37800, "task_port_base": 37810},
        {"physical_device": 3, "engine_port": 37830, "task_port_base": 37840},
    ),
}
TASKS_PER_SHARD = 10


def _read(path: Path) -> dict[str, Any]:
    return base._read(path)


def _save_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def shard_specs(lane_name: str) -> list[dict[str, Any]]:
    """Return the fixed two-shard device and port mapping for one lane."""

    if lane_name not in LANE_SHARDS:
        raise ValueError(f"Unknown trained lane: {lane_name}")
    source = next(row for row in trained.lane_specs() if row["name"] == lane_name)
    result = []
    for index, placement in enumerate(LANE_SHARDS[lane_name]):
        row = copy.deepcopy(source)
        row.update(placement)
        row["shard_index"] = index
        row["shard_id"] = f"{lane_name}.part{index}"
        row["task_ports"] = list(range(row["task_port_base"],
                                       row["task_port_base"] + TASKS_PER_SHARD))
        result.append(row)
    return result


@contextlib.contextmanager
def _bound_base(package: Path, lane: Mapping[str, Any]) -> Iterator[None]:
    previous = (base.LANES, base.lane_specs, base.DEFAULT_REMOTE_ROOT)
    raw = {key: copy.deepcopy(value) for key, value in lane.items()
           if key not in {"name", "shard_index", "shard_id", "task_ports"}}
    base.LANES = {lane["name"]: raw}
    base.lane_specs = lambda: [copy.deepcopy(dict(lane))]
    base.DEFAULT_REMOTE_ROOT = package
    try:
        yield
    finally:
        base.LANES, base.lane_specs, base.DEFAULT_REMOTE_ROOT = previous


def wait_for_training(
    training: Path,
    training_binding: Path,
    lane_name: str,
    *,
    poll_seconds: float = 20.0,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Wait for this lane's successful v8 receipt; never accepts partial training."""

    started = time.monotonic()
    while True:
        status_path = training / "status.json"
        status = _read(status_path) if status_path.is_file() else {}
        if status.get("phase") in {"failed_no_retry", "partial_failed"}:
            raise RuntimeError("T02 failed; trained D20 evaluation is forbidden")
        try:
            return trained.verify_training_binding(training, training_binding, lane_name)
        except trained.TrainingPending:
            if status.get("phase") == "training_partial_or_failed":
                raise RuntimeError(f"{lane_name} has no successful training receipt")
        if timeout_seconds is not None and time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"Timed out waiting for {lane_name} training")
        time.sleep(poll_seconds)


def _canonical_source(canonical: Path, lane_name: str) -> dict[str, Any]:
    verification = base.verify_package(canonical)
    tasks = _read(canonical / "tasks.d20.json")
    task_ids = tasks.get("task_ids")
    if not isinstance(task_ids, list) or len(task_ids) != 20 or len(set(task_ids)) != 20:
        raise ValueError("Canonical source is not the exact unique D20")
    lane_root = canonical / "lanes" / lane_name
    contract = _read(canonical / "launch_contract.json")
    if (contract.get("launch_authorized") is not False
            or contract.get("purpose") != "parallel_shard_source_only"):
        raise ValueError("Canonical full20 package is not frozen as source-only")
    paths = {
        "static_files": canonical / "static_files.json",
        "sglang_files": canonical / "sglang_files.json",
        "launch_contract": canonical / "launch_contract.json",
        "tasks_d20": canonical / "tasks.d20.json",
        "design": lane_root / "design.json",
        "lane": lane_root / "lane.json",
        "controller": lane_root / "runtime/configs/controller.json",
        "gp": lane_root / "gp.json",
        "trained_artifact": lane_root / "trained_artifact.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("Canonical trained package is incomplete")
    return {
        "verification": verification,
        "tasks": tasks,
        "task_ids": task_ids,
        "paths": paths,
        "sha256": {name: base._sha(path) for name, path in paths.items()},
    }


def _expected_design(
    canonical_design: Mapping[str, Any],
    lane: Mapping[str, Any],
    task_ids: Sequence[str],
    task_manifest_sha256: str,
) -> dict[str, Any]:
    result = copy.deepcopy(canonical_design)
    result["task_ids"] = list(task_ids)
    result["task_manifest_sha256"] = task_manifest_sha256
    result["limits"]["tasks"] = TASKS_PER_SHARD
    result["runtime"]["sglang_backend_url"] = (
        f"http://127.0.0.1:{lane['engine_port']}"
    )
    result["parallel_execution"] = {
        "schema": SCHEMA,
        "shard_id": lane["shard_id"],
        "shard_index": lane["shard_index"],
        "task_count": TASKS_PER_SHARD,
        "canonical_denominator": 20,
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }
    return result


def _finalize_canonical(canonical: Path, lane_name: str) -> None:
    shutil.copyfile(Path(__file__), canonical / Path(__file__).name)
    contract_path = canonical / "launch_contract.json"
    contract = _read(contract_path)
    contract.update(
        launch_authorized=False,
        purpose="parallel_shard_source_only",
        original_d20_manifest_sha256=base._sha(canonical / "tasks.d20.json"),
    )
    base._save(contract_path, contract)
    base._save(canonical / "static_files.json", base._static_manifest(canonical))
    _canonical_source(canonical, lane_name)


def _clone_shard(canonical: Path, package: Path, lane_name: str, index: int) -> dict[str, Any]:
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite shard package: {package}")
    source = _canonical_source(canonical, lane_name)
    lane = shard_specs(lane_name)[index]
    task_ids = source["task_ids"][index * TASKS_PER_SHARD:(index + 1) * TASKS_PER_SHARD]
    if len(task_ids) != TASKS_PER_SHARD:
        raise ValueError("Canonical D20 cannot be partitioned into exact 10-task shards")
    shutil.copytree(canonical, package,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
    static_path = package / "static_files.json"
    static_path.unlink()
    lane_root = package / "lanes" / lane_name
    tasks = copy.deepcopy(source["tasks"])
    tasks["task_ids"] = list(task_ids)
    shard_tasks_path = package / "tasks.shard.json"
    base._save(shard_tasks_path, tasks)
    base._save(lane_root / "tasks.json", tasks)
    canonical_design = _read(canonical / "lanes" / lane_name / "design.json")
    design = _expected_design(
        canonical_design, lane, task_ids, base._sha(shard_tasks_path)
    )
    base._save(lane_root / "design.json", design)
    base._save(lane_root / "lane.json", lane)

    binding = {
        "schema": SOURCE_BINDING_SCHEMA,
        "lane": lane_name,
        "shard_id": lane["shard_id"],
        "shard_index": index,
        "canonical_package": str(canonical.resolve()),
        "canonical_sha256": source["sha256"],
        "original_d20_task_ids_sha256": _digest(source["task_ids"]),
        "original_d20_manifest_sha256": source["sha256"]["tasks_d20"],
        "shard_task_ids": list(task_ids),
        "shard_task_ids_sha256": _digest(task_ids),
        "preserved_files": ["controller", "gp", "trained_artifact", "sglang_files"],
        "allowed_changes": ["task subset", "physical device", "engine/task ports"],
    }
    binding_path = package / "parallel_source_binding.json"
    base._save(binding_path, binding)
    contract_path = package / "launch_contract.json"
    contract = _read(contract_path)
    contract.update(
        launch_authorized=True,
        purpose="parallel_10_task_shard",
        total_task_execution_budget=TASKS_PER_SHARD,
        lanes=[lane],
        parallel_source_binding="parallel_source_binding.json",
        parallel_source_binding_sha256=base._sha(binding_path),
    )
    base._save(contract_path, contract)
    with _bound_base(package, lane):
        base._save(lane_root / "preview.json", base._preview_lane(package, lane))
    base._save(static_path, base._static_manifest(package))
    return verify_shard_package(package, require_pristine=True)


def prepare_parallel(
    root: Path,
    source: Path,
    training: Path,
    lane_name: str,
    training_binding: Path,
) -> dict[str, Any]:
    """Build one canonical full20 package and its two frozen task shards."""

    root = root.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite parallel evaluation root: {root}")
    trained.verify_training_binding(training, training_binding, lane_name)
    root.mkdir(parents=True)
    status_path = root / "prepare_status.json"
    base._save(status_path, {"phase": "preparing", "lane": lane_name,
                             "automatic_retries": 0, "automatic_reruns": 0})
    try:
        canonical = root / "canonical_full20"
        canonical_lane = next(row for row in trained.lane_specs()
                              if row["name"] == lane_name)
        with _bound_base(canonical, canonical_lane):
            trained.prepare(canonical, source, training, lane_name, training_binding)
        _finalize_canonical(canonical, lane_name)
        shards = []
        for index in range(2):
            package = root / f"shard{index}"
            verification = _clone_shard(canonical, package, lane_name, index)
            shards.append({
                "shard_id": shard_specs(lane_name)[index]["shard_id"],
                "package": str(package),
                "static_files_sha256": base._sha(package / "static_files.json"),
                "sglang_files_sha256": base._sha(package / "sglang_files.json"),
                "source_binding_sha256": base._sha(package / "parallel_source_binding.json"),
                "task_ids": verification["task_ids"],
            })
        canonical_source = _canonical_source(canonical, lane_name)
        result = {
            "schema": SCHEMA,
            "status": "prepared",
            "lane": lane_name,
            "canonical_package": str(canonical),
            "canonical_sha256": canonical_source["sha256"],
            "original_d20_task_ids": canonical_source["task_ids"],
            "original_d20_task_ids_sha256": _digest(canonical_source["task_ids"]),
            "training_package": str(training.resolve()),
            "training_binding": str(training_binding.resolve()),
            "training_binding_sha256": base._sha(training_binding),
            "shards": shards,
            "total_task_execution_budget": 20,
            "automatic_retries": 0,
            "automatic_reruns": 0,
        }
        _save_new(root / "parallel_manifest.json", result)
        base._save(status_path, {"phase": "prepared", "lane": lane_name,
                                 "manifest_sha256": base._sha(root / "parallel_manifest.json"),
                                 "automatic_retries": 0, "automatic_reruns": 0})
        return result
    except BaseException as error:
        base._save(status_path, {"phase": "failed_no_retry", "lane": lane_name,
                                 "error": f"{type(error).__name__}: {error}",
                                 "automatic_retries": 0, "automatic_reruns": 0})
        raise


def verify_shard_package(package: Path, *, require_pristine: bool = False) -> dict[str, Any]:
    """Verify a shard against the immutable canonical full20 source."""

    package = package.resolve()
    base.verify_package(package)
    binding = _read(package / "parallel_source_binding.json")
    if binding.get("schema") != SOURCE_BINDING_SCHEMA:
        raise ValueError("Unknown trained D20 shard binding schema")
    lane_name = binding.get("lane")
    index = binding.get("shard_index")
    if lane_name not in LANE_SHARDS or index not in (0, 1):
        raise ValueError("Shard binding has an invalid lane or index")
    canonical = Path(binding.get("canonical_package", "")).resolve()
    source = _canonical_source(canonical, lane_name)
    if (binding.get("canonical_sha256") != source["sha256"]
            or binding.get("original_d20_manifest_sha256")
            != source["sha256"]["tasks_d20"]
            or binding.get("original_d20_task_ids_sha256") != _digest(source["task_ids"])):
        raise ValueError("Shard differs from its canonical full20 source binding")
    expected_tasks = source["task_ids"][index * TASKS_PER_SHARD:(index + 1) * TASKS_PER_SHARD]
    if (binding.get("shard_task_ids") != expected_tasks
            or binding.get("shard_task_ids_sha256") != _digest(expected_tasks)):
        raise ValueError("Shard task partition differs from ordered canonical D20")
    lane = shard_specs(lane_name)[index]
    lane_root = package / "lanes" / lane_name
    tasks = _read(package / "tasks.shard.json")
    if (_read(lane_root / "tasks.json") != tasks
            or tasks.get("task_ids") != expected_tasks
            or _read(lane_root / "lane.json") != lane):
        raise ValueError("Shard task or device/port mapping differs from the fixed plan")
    expected_design = _expected_design(
        _read(canonical / "lanes" / lane_name / "design.json"),
        lane,
        expected_tasks,
        base._sha(package / "tasks.shard.json"),
    )
    if _read(lane_root / "design.json") != expected_design:
        raise ValueError("Shard changed the canonical scientific design")
    preserved = {
        "controller": lane_root / "runtime/configs/controller.json",
        "gp": lane_root / "gp.json",
        "trained_artifact": lane_root / "trained_artifact.json",
        "sglang_files": package / "sglang_files.json",
    }
    if any(base._sha(path) != source["sha256"][name]
           for name, path in preserved.items()):
        raise ValueError("Shard changed a frozen controller, artifact, or SGLang binding")
    contract = _read(package / "launch_contract.json")
    if (contract.get("launch_authorized") is not True
            or contract.get("purpose") != "parallel_10_task_shard"
            or contract.get("total_task_execution_budget") != TASKS_PER_SHARD
            or contract.get("automatic_reruns") != 0
            or contract.get("parallel_source_binding_sha256")
            != base._sha(package / "parallel_source_binding.json")
            or contract.get("lanes") != [lane]):
        raise ValueError("Shard launch contract differs from the fixed once-only plan")
    if require_pristine and (lane_root / "run").exists():
        raise FileExistsError("Shard already has a run status; automatic rerun is forbidden")
    if require_pristine and (lane_root / "results").exists():
        raise FileExistsError("Shard already has native results; automatic rerun is forbidden")
    return {"status": "passed", "lane": lane_name, "shard_index": index,
            "lane_spec": lane, "task_ids": expected_tasks,
            "canonical_package": str(canonical)}


def wait_for_device(package: Path, *, poll_seconds: float = 20.0) -> dict[str, Any]:
    """Wait until this shard's exact device and ports are free."""

    verified = verify_shard_package(package, require_pristine=True)
    while True:
        try:
            return base._assert_lane_free(verified["lane_spec"])
        except RuntimeError:
            time.sleep(poll_seconds)


def run_shard(package: Path) -> int:
    """Run one verified shard once; device occupancy is an immediate refusal."""

    verified = verify_shard_package(package, require_pristine=True)
    base._assert_lane_free(verified["lane_spec"])
    trained.ascend_environment()
    trained.enable_strict_sampling()
    with _bound_base(package.resolve(), verified["lane_spec"]):
        return base.run_lane(package, verified["lane"])


def wait_run_shard(package: Path, *, poll_seconds: float = 20.0) -> int:
    wait_for_device(package, poll_seconds=poll_seconds)
    return run_shard(package)


def _slim_official(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {key: value.get(key) for key in (
        "benchmark", "categories", "mode", "n_total", "n_scored",
        "correct_count", "semantic_score", "scored",
        "total_gold_checker_seconds", "total_handler_http_calls",
    )}


def _strict_total(value: Any, key: str) -> int | float | None:
    item = value.get(key, {}) if isinstance(value, dict) else {}
    total = item.get("strict_total")
    return total if isinstance(total, (int, float)) and not isinstance(total, bool) else None


def _slim_final(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    summary = value.get("cost_summary", {})
    costs = summary.get("costs", {}) if isinstance(summary, dict) else {}
    usage = costs.get("openai_resident_usage", {}) if isinstance(costs, dict) else {}
    work = costs.get("actual_model_work", {}) if isinstance(costs, dict) else {}
    return {"status": value.get("status"), "stop_reason": value.get("stop_reason"),
            "wall_seconds": value.get("wall_seconds"),
            "wall_seconds_final": value.get("wall_seconds_final"),
            "cost": {"generation_attempts": summary.get("generation_attempts"),
                     "prompt_tokens": _strict_total(usage, "prompt_tokens"),
                     "completion_tokens": _strict_total(usage, "completion_tokens"),
                     "total_tokens": _strict_total(usage, "total_tokens"),
                     "materialized_encoder_tokens":
                         _strict_total(work, "materialized_encoder_tokens")}}


def aggregate_results(root: Path) -> dict[str, Any]:
    """Expose two successful native shards as one ordered 20-task D20 lane."""

    root = root.resolve()
    manifest = _read(root / "parallel_manifest.json")
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "prepared":
        raise ValueError("Parallel D20 manifest is missing or invalid")
    lane_name = manifest.get("lane")
    canonical = _canonical_source(Path(manifest["canonical_package"]), lane_name)
    if (manifest.get("canonical_sha256") != canonical["sha256"]
            or manifest.get("original_d20_task_ids") != canonical["task_ids"]):
        raise ValueError("Parallel manifest differs from canonical D20")
    observations: dict[str, Any] = {}
    shard_rows = []
    for index in range(2):
        package = root / f"shard{index}"
        verified = verify_shard_package(package)
        lane_root = package / "lanes" / lane_name
        status_path = lane_root / "run/status.json"
        stage_path = lane_root / "results/stage_manifest.json"
        if not status_path.is_file() or not stage_path.is_file():
            raise RuntimeError(f"{verified['lane_spec']['shard_id']} has no terminal native result")
        status = _read(status_path)
        stage = _read(stage_path)
        outcomes = stage.get("task_outcomes")
        if (status.get("state") != "completed"
                or status.get("runner", {}).get("returncode") != 0
                or status.get("completed_task_cells") != TASKS_PER_SHARD
                or stage.get("status") != "completed_fixed_manifest"
                or stage.get("task_ids") != verified["task_ids"]
                or stage.get("completed_task_cells") != TASKS_PER_SHARD
                or not isinstance(outcomes, list)
                or [row.get("task_id") for row in outcomes] != verified["task_ids"]):
            raise RuntimeError(f"{verified['lane_spec']['shard_id']} is partial or failed")
        for outcome in outcomes:
            task_id = outcome["task_id"]
            shard = lane_root / "results/task_shards" / task_id
            official_path = shard / "bfcl/official_summary.json"
            final_path = shard / "server/final.json"
            if (outcome.get("runtime_completed") is not True
                    or not official_path.is_file() or not final_path.is_file()):
                raise RuntimeError(f"{task_id} lacks a complete native result")
            observations[task_id] = {
                "stage_outcome": outcome,
                "server_final": _slim_final(_read(final_path)),
                "official_summary": _slim_official(_read(official_path)),
                "native_artifacts": {
                    "shard_id": verified["lane_spec"]["shard_id"],
                    "official_summary": {"path": str(official_path),
                                         "sha256": base._sha(official_path)},
                    "server_final": {"path": str(final_path),
                                     "sha256": base._sha(final_path)},
                },
            }
        shard_rows.append({"shard_id": verified["lane_spec"]["shard_id"],
                           "package": str(package), "status_sha256": base._sha(status_path),
                           "stage_manifest_sha256": base._sha(stage_path),
                           "task_ids": verified["task_ids"]})
    if list(observations) != canonical["task_ids"]:
        raise RuntimeError("Native shards do not reconstruct the ordered canonical D20")
    return {
        "schema": AGGREGATE_SCHEMA,
        "status": "completed",
        "lane": lane_name,
        "declared_tasks": canonical["task_ids"],
        "task_count": 20,
        "task_observations": observations,
        "canonical_package": str(Path(manifest["canonical_package"]).resolve()),
        "canonical_sha256": canonical["sha256"],
        "shards": shard_rows,
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }


def watch_lane(args: argparse.Namespace) -> int:
    """Once-only watcher: wait, prepare, then supervise two independent shards."""

    waiting = args.root.parent / f"{args.root.name}.waiting.json"
    _save_new(waiting, {"phase": "waiting_for_training", "lane": args.lane,
                        "pid": os.getpid(), "automatic_retries": 0,
                        "automatic_reruns": 0})
    try:
        wait_for_training(args.training, args.training_binding, args.lane,
                          poll_seconds=args.poll_seconds)
        prepare_parallel(args.root, args.source, args.training, args.lane,
                         args.training_binding)
        children = []
        logs = []
        for index in range(2):
            package = args.root / f"shard{index}"
            script = package / Path(__file__).name
            log_path = args.root / f"shard{index}.supervisor.log"
            log = log_path.open("x", encoding="utf-8", newline="\n")
            logs.append(log)
            command = [sys.executable, str(script), "wait-run-shard",
                       "--package", str(package), "--poll-seconds", str(args.poll_seconds)]
            child = subprocess.Popen(command, cwd=package, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            children.append((child, command))
        base._save(waiting, {"phase": "running_shards", "lane": args.lane,
                             "pid": os.getpid(),
                             "children": [{"pid": child.pid, "command": command}
                                          for child, command in children],
                             "automatic_retries": 0, "automatic_reruns": 0})
        codes = [child.wait() for child, _ in children]
        for log in logs:
            log.close()
        if codes != [0, 0]:
            base._save(waiting, {"phase": "failed_no_retry", "lane": args.lane,
                                 "returncodes": codes, "automatic_retries": 0,
                                 "automatic_reruns": 0})
            return 2
        aggregate = aggregate_results(args.root)
        _save_new(args.root / "aggregate.json", aggregate)
        base._save(waiting, {"phase": "completed", "lane": args.lane,
                             "returncodes": codes,
                             "aggregate_sha256": base._sha(args.root / "aggregate.json"),
                             "automatic_retries": 0, "automatic_reruns": 0})
        return 0
    except BaseException as error:
        base._save(waiting, {"phase": "failed_no_retry", "lane": args.lane,
                             "error": f"{type(error).__name__}: {error}",
                             "automatic_retries": 0, "automatic_reruns": 0})
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    wait = commands.add_parser("wait")
    prepare = commands.add_parser("prepare")
    watch = commands.add_parser("watch")
    for command in (wait, prepare, watch):
        command.add_argument("--training", type=Path, required=True)
        command.add_argument("--training-binding", type=Path, required=True)
        command.add_argument("--lane", choices=tuple(LANE_SHARDS), required=True)
    for command in (prepare, watch):
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--source", type=Path, required=True)
    wait.add_argument("--poll-seconds", type=float, default=20.0)
    watch.add_argument("--poll-seconds", type=float, default=20.0)
    run = commands.add_parser("run-shard")
    run.add_argument("--package", type=Path, required=True)
    wait_run = commands.add_parser("wait-run-shard")
    wait_run.add_argument("--package", type=Path, required=True)
    wait_run.add_argument("--poll-seconds", type=float, default=20.0)
    verify = commands.add_parser("verify-shard")
    verify.add_argument("--package", type=Path, required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--root", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "wait":
        wait_for_training(args.training, args.training_binding, args.lane,
                          poll_seconds=args.poll_seconds)
        return 0
    if args.command == "prepare":
        prepare_parallel(args.root, args.source, args.training, args.lane,
                         args.training_binding)
        return 0
    if args.command == "watch":
        return watch_lane(args)
    if args.command == "run-shard":
        return run_shard(args.package)
    if args.command == "wait-run-shard":
        return wait_run_shard(args.package, poll_seconds=args.poll_seconds)
    if args.command == "verify-shard":
        print(json.dumps(verify_shard_package(args.package), sort_keys=True))
        return 0
    result = aggregate_results(args.root)
    _save_new(args.output, result)
    print(json.dumps({"status": result["status"], "lane": result["lane"],
                      "task_count": result["task_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_SCHEMA",
    "LANE_SHARDS",
    "SCHEMA",
    "aggregate_results",
    "prepare_parallel",
    "run_shard",
    "shard_specs",
    "verify_shard_package",
    "wait_for_device",
    "wait_for_training",
    "wait_run_shard",
]
