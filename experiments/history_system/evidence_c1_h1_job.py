"""Freeze and run the independent H1 calibration job for a selected C1 head."""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import evidence_eval as base
import evidence_eval_trained as trained

SCHEMA = "c1-h1-calibration-job-v1"
DEVICE_IDS = {0, 1, 2, 3, 4, 6}


def make_source_design(source: dict, controller: dict, tasks: list[str]) -> dict:
    design = copy.deepcopy(source)
    design.update(candidate_id=source["candidate_id"] + "__h1_calibration_c0",
                  task_ids=tasks, launch_authorized=False, status="frozen")
    design["resolved_configs"]["controller"] = controller
    design["limits"]["tasks"] = len(tasks)
    design["search_contract"].update(history="H1", selector="candidate_rule",
        calibration_only=True, evaluation_tasks_used_for_tuning=False)
    return design


def prepare(package: Path, *, training: Path, training_binding: Path,
            c1_package: Path, c1_lane: str, device: int) -> dict:
    if device not in DEVICE_IDS:
        raise ValueError("Calibration device must be an authorized free NPU")
    package, training, c1_package = map(Path.resolve, (package, training, c1_package))
    if package.exists():
        raise FileExistsError("Refusing to overwrite H1 calibration package")
    trained.verify_training_binding(training, training_binding, "C1")
    base.verify_package(c1_package)
    source_lane = c1_package / "lanes" / c1_lane
    design = base._read(source_lane / "design.json")
    original = base._read(source_lane / "runtime/configs/controller.json")
    artifact = base._read(training / "training/c1_risk.json")
    if (design["resolved_configs"]["controller"] != original
            or original["gp_experiments"].get("G") != "current"
            or original["gp_experiments"].get("R") != 1
            or original["gp_experiments"].get("set_selector") != "risk"
            or original["gp_experiments"].get("selector_artifact") != artifact):
        raise ValueError("Source must be the exact newly trained H0 C1 controller")
    runtime = source_lane / "runtime"
    sys.path[:0] = [str(runtime / "python"), str(runtime)]
    from benchmarks.memory_runtime.recovery.experiment_config import parse_gp_config
    from evidence_c1_h1_calibration import H1_SOURCE_HASHES, json_file_binding
    for name, sha in H1_SOURCE_HASHES.items():
        if base._sha(runtime / name) != sha:
            raise ValueError("Source runtime lacks the frozen H1 implementation")
    controller = copy.deepcopy(original)
    gp = controller["gp_experiments"]
    gp.update(G="record_bound", set_selector="candidate_rule", selector_artifact=None,
              export_selection_state=False)
    controller["gp_experiments"] = gp = parse_gp_config(gp)
    labels = base._read(training / "run/labels.json")
    cal_groups = {row["task_group_id"] for row in labels["rows"] if row["split"] == "calibration"}
    sys.path.insert(0, str(training / "history_system"))
    import t02
    import t02_bfcl
    common = base._read(training / "launch_contract.json")["common_args"]
    task_manifest = training / "configs/tasks.json"
    if not task_manifest.is_file():
        raise FileNotFoundError("The frozen original 104-task manifest is required")
    tasks = [task for task in base._read(task_manifest)["task_ids"]
             if t02.canonical_task_group_id(task) in cal_groups]
    tasks = t02_bfcl.round_robin_family_tasks(tasks, {
        task: t02.canonical_task_group_id(task) for task in tasks})
    package.mkdir(parents=True)
    history = package / "history_system"
    history.mkdir()
    for name in ("t02.py", "t02_bfcl.py", "t02_runtime.py"):
        shutil.copyfile(training / "history_system" / name, history / name)
    for name in ("evidence_c1_h1_collect.py", "evidence_c1_h1_calibration.py"):
        shutil.copyfile(Path(__file__).parent / name, history / name)
    for name in ("evidence_eval.py", "evidence_eval_trained.py", Path(__file__).name):
        shutil.copyfile(Path(__file__).parent / name, package / name)
    base._copy_tree(runtime, history / "runtime")
    base._save(history / "runtime/configs/controller.json", controller)
    lane_root = package / "lanes/calibration"
    lane_root.mkdir(parents=True)
    base._copy_tree(history / "runtime", lane_root / "runtime")
    base._copy_tree(c1_package / "sglang", package / "sglang")
    shutil.copyfile(c1_package / "sglang_files.json", package / "sglang_files.json")
    source_design = make_source_design(design, controller, tasks)
    source_design["source_files"] = base._tree_hashes(history / "runtime")
    base._save(history / "design.json", source_design)
    base._save(history / "gp.json", gp)
    shutil.copyfile(training / "training/c1_risk.json", package / "c1_risk.json")
    lane = {"name": "calibration", "physical_device": device,
            "engine_port": 41000 + device * 10, "task_port_base": 41100 + device * 40,
            "task_ports": [], "model_smokes": ["embedding"], "selector": "candidate_rule"}
    base._save(lane_root / "lane.json", lane)
    spec = {"schema": "c1-h1-calibration-collection-spec-v1",
            "source_design": json_file_binding(history / "design.json"),
            "t02_labels": json_file_binding(training / "run/labels.json"),
            "t02_summary": json_file_binding(training / "run/summary.json"),
            "task_manifest": json_file_binding(task_manifest),
            "d128_manifest": json_file_binding(training / "configs/D128.json"),
            "f128_manifest": json_file_binding(training / "configs/F128.json"),
            "task_ids": tasks, "max_states": 38, "automatic_retries": 0,
            "physical_device": device, "backend_url": f"http://127.0.0.1:{lane['engine_port']}",
            "checkpoint": str(base.CHECKPOINT), "bfcl_root": str(base.BENCHMARK_DIR),
            "bfcl_dependency_path": common["bfcl_dependency_path"],
            "history_binding": {"source_history": "H1", "G": "record_bound",
                "gp_path": str(history / "gp.json"),
                "gp_file_sha256": base._sha(history / "gp.json"),
                "gp_payload_sha256": trained._canonical_digest(gp),
                "h1_source_hashes": H1_SOURCE_HASHES},
            "frozen_policy": {"design_sha256": base._sha(history / "design.json"),
                              "gp_file_sha256": base._sha(history / "gp.json")}}
    base._save(package / "collection_spec.json", spec)
    base._save(package / "launch_contract.json", {"schema": SCHEMA, "launch_authorized": True,
        "source_training_package": str(training), "source_c1_package": str(c1_package),
        "source_c1_artifact_sha256": base._sha(package / "c1_risk.json"),
        "required_environment": {"C2KV_STRICT_NONFINITE_SAMPLING": "1"},
        "state_cap": 38, "complete_task_T02_branches": 0, "automatic_retries": 0})
    base._save(package / "static_files.json", base._static_manifest(package))
    base.verify_package(package)
    result = subprocess.run([sys.executable, str(history / "evidence_c1_h1_collect.py"),
        "--spec", str(package / "collection_spec.json"), "--output", str(package / "collected"),
        "--validate-only"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Frozen collector input validation failed: {result.stderr}")
    return {"status": "prepared", "package": str(package), "model_calls": 0,
            "static_files_sha256": base._sha(package / "static_files.json")}


def run(package: Path) -> int:
    package = package.resolve()
    if os.name == "nt":
        raise RuntimeError("H1 production calibration must run on the remote NPU")
    base.verify_package(package)
    contract = base._read(package / "launch_contract.json")
    lane = base._read(package / "lanes/calibration/lane.json")
    if contract.get("schema") != SCHEMA or contract.get("launch_authorized") is not True:
        raise ValueError("Missing frozen calibration authorization")
    run_root = package / "run"
    run_root.mkdir(exist_ok=False)
    status_path = run_root / "status.json"
    status = {"phase": "preflight", "pid": os.getpid(), "device": lane["physical_device"]}
    engine = None
    try:
        trained.ascend_environment()
        trained.enable_strict_sampling()
        base._assert_lane_free(lane)
        env = base._lane_env(package, lane)
        base._apply_supervisor_network_env(env)
        with (run_root / "engine.log").open("x", encoding="utf-8") as log:
            engine = subprocess.Popen(base._engine_command(package, lane), cwd=package, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status.update(phase="starting_engine", engine_pid=engine.pid)
        base._save(status_path, status)
        base._save(run_root / "model_info.json", base._wait_engine(engine, lane))
        status["phase"] = "collecting_h1_a0_turns"
        base._save(status_path, status)
        with (run_root / "collector.log").open("x", encoding="utf-8") as log:
            result = subprocess.run([sys.executable,
                str(package / "history_system/evidence_c1_h1_collect.py"),
                "--spec", str(package / "collection_spec.json"), "--output", str(package / "collected")],
                cwd=package, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"H1 collection failed with return code {result.returncode}")
        sys.path.insert(0, str(package / "history_system"))
        calibration = importlib.import_module("evidence_c1_h1_calibration")
        receipt = calibration.build_calibration_receipt(package / "c1_risk.json", package / "collected/dataset.json")
        base._save(package / "calibration_receipt.json", receipt)
        calibration.verify_calibration_receipt(package / "calibration_receipt.json",
            artifact_path=package / "c1_risk.json", input_path=package / "collected/dataset.json")
        status.update(phase="completed", selected_threshold=receipt["selected_threshold"])
        return 0
    except Exception as error:
        status.update(phase="failed_no_retry", error=f"{type(error).__name__}: {error}")
        return 2
    finally:
        if engine is not None:
            status["engine_cleanup_returncode"] = base._stop_owned_group(engine)
        base._save(status_path, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--package", type=Path, required=True)
    prep.add_argument("--training", type=Path, required=True)
    prep.add_argument("--training-binding", type=Path, required=True)
    prep.add_argument("--c1-package", type=Path, required=True)
    prep.add_argument("--c1-lane", default="C1")
    prep.add_argument("--device", type=int, required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--package", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        return run(args.package)
    print(json.dumps(prepare(args.package, training=args.training, training_binding=args.training_binding,
        c1_package=args.c1_package, c1_lane=args.c1_lane, device=args.device)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
