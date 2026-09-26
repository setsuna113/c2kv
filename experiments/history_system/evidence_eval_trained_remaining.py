"""Freeze and run only the six never-started C4_turn D20 cells."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

import evidence_eval as base
import evidence_eval_trained as trained
import evidence_eval_trained_parallel as parallel


SCHEMA = "evidence-sets-trained-d20-remaining-v1"
BINDING_SCHEMA = "evidence-sets-trained-d20-remaining-binding-v1"
LANE = "C4_turn"
TASKS = (
    "multi_turn_base_130", "multi_turn_long_context_130",
    "multi_turn_base_170", "multi_turn_long_context_170",
    "multi_turn_base_190", "multi_turn_long_context_190",
)
FAILED_TASK = "multi_turn_long_context_120"
SHARD1_TASKS = (
    "multi_turn_base_100", "multi_turn_long_context_100",
    "multi_turn_base_120", FAILED_TASK, *TASKS,
)
PLACEMENTS = (
    {"physical_device": 0, "engine_port": 38300, "task_port_base": 38400, "wave": 0},
    {"physical_device": 1, "engine_port": 38310, "task_port_base": 38410, "wave": 0},
    {"physical_device": 2, "engine_port": 38320, "task_port_base": 38420, "wave": 0},
    {"physical_device": 6, "engine_port": 38360, "task_port_base": 38460, "wave": 0},
    {"physical_device": 0, "engine_port": 38500, "task_port_base": 38600, "wave": 1,
     "predecessor": "C4_turn.remaining0"},
    {"physical_device": 1, "engine_port": 38510, "task_port_base": 38610, "wave": 1,
     "predecessor": "C4_turn.remaining1"},
)
DEFAULT_CANONICAL = Path(
    "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_trained_v5/"
    "C4_turn/canonical_full20")
DEFAULT_FAILED = DEFAULT_CANONICAL.parent / "shard1"
DEFAULT_ROOT = Path(
    "/home/liuyancheng/c2kv-evidence-sets-20260916/"
    "eval_trained_remaining_v1/C4_turn")


def _read(path: Path) -> dict[str, Any]:
    return base._read(path)


def _save_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8")


def shard_spec(index: int) -> dict[str, Any]:
    source = next(row for row in trained.lane_specs() if row["name"] == LANE)
    row = copy.deepcopy(source)
    row.update(PLACEMENTS[index])
    row.update(shard_index=index, shard_id=f"{LANE}.remaining{index}",
               task_ports=[PLACEMENTS[index]["task_port_base"]])
    return row


def _origin(canonical: Path, failed: Path) -> dict[str, Any]:
    source = parallel._canonical_source(canonical.resolve(), LANE)
    failed = failed.resolve()
    verified = parallel.verify_shard_package(failed)
    if verified.get("lane") != LANE or verified.get("shard_index") != 1:
        raise ValueError("Original failure is not exact C4_turn shard1")
    lane_root = failed / "lanes" / LANE
    manifest_path = lane_root / "results/stage_manifest.json"
    status_path = lane_root / "run/status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("Original C4_turn shard1 terminal receipts are missing")
    manifest, status = _read(manifest_path), _read(status_path)
    outcomes = manifest.get("task_outcomes")
    if (manifest.get("status") != "stopped_on_actor_runtime_failure"
            or manifest.get("state") != "failed"
            or manifest.get("completed_task_cells") != 4
            or manifest.get("automatic_retries") != 0
            or manifest.get("automatic_reruns") != 0
            or status.get("state") != "failed_no_rerun"
            or status.get("completed_task_cells") != 4
            or status.get("automatic_retries") != 0
            or status.get("automatic_reruns") != 0
            or status.get("runner", {}).get("returncode") in {None, 0}
            or verified["task_ids"] != list(SHARD1_TASKS)
            or not isinstance(outcomes, list)
            or [row.get("task_id") for row in outcomes] != verified["task_ids"]
            or len(outcomes) != 10):
        raise ValueError("Original C4_turn shard1 is not the exact terminal failure")
    by_task = {row["task_id"]: row for row in outcomes}
    if len(by_task) != len(outcomes) or set(TASKS) - set(by_task):
        raise ValueError("Original shard outcomes are duplicate or incomplete")
    for task_id in SHARD1_TASKS[:3]:
        row = by_task[task_id]
        if (row.get("outcome") != "official_completed"
                or row.get("runtime_completed") is not True
                or row.get("worker_returncode") != 0
                or row.get("server_returncode") != 0):
            raise ValueError("Original shard lacks its exact three completed cells")
    failure = by_task[FAILED_TASK]
    if (failure.get("outcome") != "runtime_failure_in_denominator"
            or failure.get("runtime_completed") is not False):
        raise ValueError("Original shard lacks the exact fourth runtime failure")
    for task_id in TASKS:
        row = by_task[task_id]
        native = lane_root / "results/task_shards" / task_id
        if (row.get("outcome") != "not_started"
                or row.get("official_summary") is not None or native.exists()):
            raise ValueError(f"Task is not pristine never-started work: {task_id}")
    return {"source": source, "verified": verified, "manifest": manifest,
            "manifest_path": manifest_path, "status_path": status_path,
            "manifest_sha256": base._sha(manifest_path),
            "status_sha256": base._sha(status_path)}


def _design(canonical: Path, lane: Mapping[str, Any], task_id: str,
            manifest_sha256: str) -> dict[str, Any]:
    value = copy.deepcopy(_read(canonical / f"lanes/{LANE}/design.json"))
    value["task_ids"] = [task_id]
    value["task_manifest_sha256"] = manifest_sha256
    value["limits"] = {**value["limits"], "tasks": 1}
    value["runtime"] = {**value["runtime"],
                        "sglang_backend_url":
                        f"http://127.0.0.1:{lane['engine_port']}"}
    value["run_id_template"] = lane["shard_id"]
    value["remaining_execution"] = {
        "schema": SCHEMA, "shard_id": lane["shard_id"],
        "task_id": task_id, "fixed_plan_task_count": 6,
        "automatic_retries": 0, "automatic_reruns": 0,
    }
    return value


def _clone(canonical: Path, failed: Path, package: Path, index: int) -> dict[str, Any]:
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite remaining shard: {package}")
    origin = _origin(canonical, failed)
    lane, task_id = shard_spec(index), TASKS[index]
    shutil.copytree(canonical, package,
                    ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
    (package / "static_files.json").unlink()
    shutil.copyfile(Path(__file__), package / Path(__file__).name)
    lane_root = package / "lanes" / LANE
    tasks = copy.deepcopy(origin["source"]["tasks"])
    tasks["task_ids"] = [task_id]
    base._save(package / "tasks.remaining.json", tasks)
    base._save(lane_root / "tasks.json", tasks)
    base._save(lane_root / "lane.json", lane)
    base._save(lane_root / "design.json", _design(
        canonical, lane, task_id, base._sha(package / "tasks.remaining.json")))
    binding = {
        "schema": BINDING_SCHEMA, "shard_id": lane["shard_id"],
        "shard_index": index, "lane": LANE, "task_id": task_id,
        "canonical_package": str(canonical.resolve()),
        "canonical_sha256": origin["source"]["sha256"],
        "failed_shard": str(failed.resolve()),
        "failed_manifest_sha256": origin["manifest_sha256"],
        "failed_status_sha256": origin["status_sha256"],
        "source_outcome": copy.deepcopy(next(
            row for row in origin["manifest"]["task_outcomes"]
            if row["task_id"] == task_id)),
        "fixed_remaining_task_ids": list(TASKS),
        "total_task_execution_budget": 6,
    }
    base._save(package / "remaining_source_binding.json", binding)
    contract = _read(package / "launch_contract.json")
    contract.update(
        launch_authorized=True, purpose="C4_turn_never_started_single_task",
        total_task_execution_budget=1, automatic_retries=0, automatic_reruns=0,
        lanes=[lane], remaining_source_binding="remaining_source_binding.json",
        remaining_source_binding_sha256=base._sha(
            package / "remaining_source_binding.json"))
    base._save(package / "launch_contract.json", contract)
    with parallel._bound_base(package, lane):
        base._save(lane_root / "preview.json", base._preview_lane(package, lane))
    base._save(package / "static_files.json", base._static_manifest(package))
    return verify(package, pristine=True)


def prepare(root: Path, canonical: Path = DEFAULT_CANONICAL,
            failed: Path = DEFAULT_FAILED) -> dict[str, Any]:
    root = root.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite remaining root: {root}")
    _origin(canonical, failed)
    root.mkdir(parents=True)
    rows = []
    try:
        for index, task_id in enumerate(TASKS):
            package = root / f"{LANE}.remaining{index}"
            result = _clone(canonical, failed, package, index)
            lane = result["lane_spec"]
            rows.append({"id": lane["shard_id"], "root": str(package),
                         "lane": LANE, "task_id": task_id,
                         "physical_device": lane["physical_device"],
                         "engine_port": lane["engine_port"],
                         "task_port_base": lane["task_port_base"],
                         "wave": lane["wave"],
                         "predecessor": lane.get("predecessor")})
        receipt = {"schema": SCHEMA, "status": "prepared", "lane": LANE,
                   "canonical_package": str(canonical.resolve()),
                   "failed_shard": str(failed.resolve()), "rows": rows,
                   "total_task_execution_budget": 6,
                   "automatic_retries": 0, "automatic_reruns": 0}
        _save_new(root / "continuation.json", receipt)
        return receipt
    except BaseException:
        shutil.rmtree(root)
        raise


def verify(package: Path, *, pristine: bool = False) -> dict[str, Any]:
    package = package.resolve()
    base.verify_package(package)
    binding = _read(package / "remaining_source_binding.json")
    if binding.get("schema") != BINDING_SCHEMA or binding.get("lane") != LANE:
        raise ValueError("Unknown remaining shard binding")
    index = binding.get("shard_index")
    if type(index) is not int or index not in range(6):
        raise ValueError("Remaining shard index is invalid")
    canonical = Path(binding["canonical_package"]).resolve()
    failed = Path(binding["failed_shard"]).resolve()
    origin = _origin(canonical, failed)
    lane, task_id = shard_spec(index), TASKS[index]
    lane_root = package / "lanes" / LANE
    if (binding.get("shard_id") != lane["shard_id"]
            or binding.get("task_id") != task_id
            or binding.get("fixed_remaining_task_ids") != list(TASKS)
            or binding.get("canonical_sha256") != origin["source"]["sha256"]
            or binding.get("failed_manifest_sha256") != origin["manifest_sha256"]
            or binding.get("failed_status_sha256") != origin["status_sha256"]
            or binding.get("source_outcome") != next(
                row for row in origin["manifest"]["task_outcomes"]
                if row["task_id"] == task_id)
            or _read(package / "tasks.remaining.json").get("task_ids") != [task_id]
            or _read(lane_root / "tasks.json").get("task_ids") != [task_id]
            or _read(lane_root / "lane.json") != lane
            or _read(lane_root / "design.json") != _design(
                canonical, lane, task_id, base._sha(package / "tasks.remaining.json"))):
        raise ValueError("Remaining shard differs from its exact frozen plan")
    source = origin["source"]
    preserved = {"controller": lane_root / "runtime/configs/controller.json",
                 "gp": lane_root / "gp.json",
                 "trained_artifact": lane_root / "trained_artifact.json",
                 "sglang_files": package / "sglang_files.json"}
    if any(base._sha(path) != source["sha256"][name]
           for name, path in preserved.items()):
        raise ValueError("Remaining shard changed controller, artifact, or SGLang")
    contract = _read(package / "launch_contract.json")
    if (contract.get("launch_authorized") is not True
            or contract.get("purpose") != "C4_turn_never_started_single_task"
            or contract.get("total_task_execution_budget") != 1
            or contract.get("automatic_retries") != 0
            or contract.get("automatic_reruns") != 0
            or contract.get("lanes") != [lane]
            or contract.get("required_environment", {}).get(
                "C2KV_STRICT_NONFINITE_SAMPLING") != "1"):
        raise ValueError("Remaining shard launch contract differs")
    if pristine and ((lane_root / "run").exists() or (lane_root / "results").exists()):
        raise FileExistsError("Remaining shard already has run/results; retry forbidden")
    return {"status": "passed", "shard_id": lane["shard_id"],
            "lane": LANE, "task_id": task_id, "lane_spec": lane}


def run(package: Path) -> int:
    checked = verify(package, pristine=True)
    base._assert_lane_free(checked["lane_spec"])
    trained.ascend_environment()
    trained.enable_strict_sampling()
    with parallel._bound_base(package.resolve(), checked["lane_spec"]):
        return base.run_lane(package, LANE)


def wait_run(package: Path, *, poll_seconds: float = 20.0) -> int:
    checked = verify(package, pristine=True)
    predecessor = checked["lane_spec"].get("predecessor")
    if predecessor:
        predecessor_status = package.parent / predecessor / f"lanes/{LANE}/run/status.json"
        while True:
            status = _read(predecessor_status) if predecessor_status.is_file() else {}
            if status.get("state") in {"completed", "failed_no_rerun"}:
                break
            time.sleep(poll_seconds)
    while True:
        try:
            base._assert_lane_free(checked["lane_spec"])
            break
        except RuntimeError as error:
            if "acquired by PIDs" not in str(error) and "ports are occupied" not in str(error):
                raise
            time.sleep(poll_seconds)
    return run(package)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    prepare_parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    prepare_parser.add_argument("--failed-shard", type=Path, default=DEFAULT_FAILED)
    for command in ("verify", "run", "wait-run"):
        item = sub.add_parser(command)
        item.add_argument("--package", type=Path, required=True)
        if command == "wait-run":
            item.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args.root, args.canonical, args.failed_shard), indent=2))
        return 0
    if args.command == "verify":
        print(json.dumps(verify(args.package), indent=2))
        return 0
    if args.command == "run":
        return run(args.package)
    return wait_run(args.package, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
