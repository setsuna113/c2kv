"""Freeze and dispatch the original-budget C0/C5 never-started D20 cells.

This is a deliberately small wrapper around ``evidence_eval_repair``.  It
derives the six task IDs for each selected lane from the frozen runtime audit,
uses the repaired runtime overlay, and refuses any task that was previously
started.  The package has no extra-execution allowance.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import evidence_eval as base
import evidence_eval_repair as repair


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVAL_ROOT = REPO / "outputs/history_system_search/evidence_sets_v1/eval"
DEFAULT_SOURCE_PACKAGE = EVAL_ROOT / "prepared_v1_final"
DEFAULT_OUTPUT = EVAL_ROOT / "never_started/prepared_v3"
DEFAULT_REMOTE_ROOT = PurePosixPath(
    "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_never_started_v3"
)
AUDIT_PATH = EVAL_ROOT / "runtime_failure_audit.json"
ACTIVE_AUDIT_PATH = AUDIT_PATH
SELECTED_LANES = {
    "C0": {
        "selector": "candidate_rule",
        "physical_device": 0,
        "engine_port": 36800,
        "task_port_base": 36900,
        "model_smokes": ("embedding",),
    },
    "C5": {
        "selector": "parameter_source",
        "physical_device": 3,
        "engine_port": 36830,
        "task_port_base": 36920,
        "model_smokes": ("embedding",),
    },
}
EXPECTED_TASKS = (
    "multi_turn_base_130",
    "multi_turn_long_context_130",
    "multi_turn_base_170",
    "multi_turn_long_context_170",
    "multi_turn_base_190",
    "multi_turn_long_context_190",
)
ASCEND_SET_ENVS = (
    Path("/usr/local/Ascend/ascend-toolkit/set_env.sh"),
    Path("/usr/local/Ascend/nnal/atb/set_env.sh"),
)


def _ensure_ascend_environment() -> None:
    """Load the server's CANN runtime for supervisors launched by plain ssh."""

    if os.name == "nt":
        return
    missing = [path for path in ASCEND_SET_ENVS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Ascend environment scripts are missing: {missing}")
    source_commands = "; ".join(f"source {path} >/dev/null 2>&1" for path in ASCEND_SET_ENVS)
    command = f"{source_commands}; env -0"
    completed = subprocess.run(
        ["/bin/bash", "-c", command],
        check=True,
        capture_output=True,
    )
    for entry in completed.stdout.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        name, value = entry.split(b"=", 1)
        os.environ[name.decode()] = value.decode()


def _read_audit(path: Path) -> dict[str, Any]:
    audit = base._read(path)
    if audit.get("schema") != "evidence-eval-runtime-failure-audit-v1":
        raise RuntimeError("Unexpected runtime failure audit schema")
    return audit


def _audit_tasks(path: Path = AUDIT_PATH) -> tuple[str, ...]:
    audit = _read_audit(path)
    by_lane: dict[str, tuple[str, ...]] = {}
    for lane_name in SELECTED_LANES:
        rows = audit.get("lanes", {}).get(lane_name, {}).get("tasks", [])
        tasks = tuple(
            row["task_id"]
            for row in rows
            if row.get("classification") == "not_started"
        )
        if tasks != EXPECTED_TASKS:
            raise RuntimeError(
                f"Audit-derived not_started tasks changed for {lane_name}: {tasks}"
            )
        if audit["lanes"][lane_name].get("counts", {}).get("not_started") != len(
            tasks
        ):
            raise RuntimeError(f"Audit count disagrees for {lane_name}")
        by_lane[lane_name] = tasks
    if len(set(by_lane.values())) != 1:
        raise RuntimeError("Selected lanes do not have the same never-started tasks")
    return next(iter(by_lane.values()))


def _configure(audit_path: Path = AUDIT_PATH) -> tuple[str, ...]:
    global ACTIVE_AUDIT_PATH
    ACTIVE_AUDIT_PATH = audit_path
    tasks = _audit_tasks(audit_path)
    repair.DEFAULT_REMOTE_ROOT = DEFAULT_REMOTE_ROOT
    repair.APPROVED_EXTRA_EXECUTIONS = 0
    repair.REPAIR_TASKS = tasks
    repair.CLEAN_REUSE_TASKS = ()
    repair.LANES = SELECTED_LANES
    repair.lane_specs = lane_specs
    repair._configure_base()
    return tasks


def lane_specs() -> list[dict[str, Any]]:
    tasks = _audit_tasks(ACTIVE_AUDIT_PATH)
    rows = []
    for name, raw in SELECTED_LANES.items():
        row = {"name": name, **raw}
        row["model_smokes"] = list(row["model_smokes"])
        row["task_ports"] = list(
            range(row["task_port_base"], row["task_port_base"] + len(tasks))
        )
        rows.append(row)
    ports = [port for row in rows for port in [row["engine_port"], *row["task_ports"]]]
    if len(ports) != len(set(ports)):
        raise RuntimeError("Never-started lane ports overlap")
    return rows


def _rewrite_package(package: Path, source_package: Path, tasks: tuple[str, ...]) -> None:
    lanes_root = package / "lanes"
    for child in tuple(lanes_root.iterdir()):
        if child.name not in SELECTED_LANES:
            shutil.rmtree(child)

    old_manifest = package / "tasks.repair15.json"
    manifest = package / "tasks.never_started6.json"
    old_manifest.replace(manifest)
    (package / "tasks.clean_reuse5.json").unlink(missing_ok=True)
    (package / "clean_reuse.remote.json").unlink(missing_ok=True)
    manifest_value = base._read(manifest)
    manifest_value.update(
        {
            "manifest_id": "D20-C0-C5-never-started6",
            "stage": "development_search_original_budget_continuation",
            "task_ids": list(tasks),
            "fixed_denominator": len(tasks),
            "automatic_reruns": 0,
        }
    )
    base._save(manifest, manifest_value)

    lane_receipts = []
    for lane in lane_specs():
        lane_root = lanes_root / lane["name"]
        design_path = lane_root / "design.json"
        design = base._read(design_path)
        design["candidate_id"] = (
            f"evidence_sets_v1_h0_{lane['name'].lower()}_never_started_r1"
        )
        design["run_id_template"] = (
            "a_history_evidence_sets_v1_never_started_" + lane["name"].lower()
        )
        design["task_ids"] = list(tasks)
        design["task_manifest_sha256"] = base._sha(manifest)
        design["limits"] = {**design["limits"], "tasks": len(tasks)}
        contract = design["search_contract"]
        contract["fixed_denominator"] = "audit_not_started6_only"
        contract["never_started_task_count"] = len(tasks)
        contract["runtime_failure_audit"] = str(package / "runtime_failure_audit.json")
        contract["extra_task_executions"] = 0
        for key in (
            "repair_task_count",
            "clean_reuse_task_count",
            "clean_reuse_receipt",
        ):
            contract.pop(key, None)
        base._save(design_path, design)
        base._save(lane_root / "tasks.json", manifest_value)
        base._save(lane_root / "lane.json", lane)
        base._save(lane_root / "preview.json", base._preview_lane(package, lane))
        lane_receipts.append(
            {
                "lane": lane["name"],
                "selector": lane["selector"],
                "physical_device": lane["physical_device"],
                "engine_port": lane["engine_port"],
                "task_port_base": lane["task_port_base"],
                "task_count": len(tasks),
                "task_ids": list(tasks),
                "status_path": str(
                    DEFAULT_REMOTE_ROOT
                    / "lanes"
                    / lane["name"]
                    / "run/status.json"
                ),
                "design_sha256": base._sha(design_path),
            }
        )

    shutil.copyfile(Path(__file__), package / "evidence_eval_never_started.py")
    contract = base._read(package / "launch_contract.json")
    contract.update(
        {
            "schema": "evidence-sets-d20-never-started-eval-v1",
            "status": "frozen_original_budget_continuation",
            "launch_authorized": False,
            "launch_gate": {
                "flag": "--approved-extra-task-executions",
                "required_value": 0,
            },
            "remote_root": str(DEFAULT_REMOTE_ROOT),
            "fixed_task_manifest": manifest.name,
            "per_lane_task_budget": len(tasks),
            "total_task_execution_budget": len(tasks) * len(SELECTED_LANES),
            "original_task_execution_budget": 80,
            "observed_original_task_starts": 54,
            "cumulative_task_starts_after_continuation": 66,
            "extra_task_executions_requiring_approval": 0,
            "previously_started_cells_excluded": 54,
            "other_never_started_cells_deferred": 14,
            "automatic_retries": 0,
            "automatic_reruns": 0,
            "lanes": lane_receipts,
        }
    )
    for key in (
        "clean_reuse_manifest",
        "clean_reuse_receipt",
        "cumulative_task_starts_after_repair",
        "clean_cells_reused",
        "failed_or_incomplete_cells_not_reused",
    ):
        contract.pop(key, None)
    base._save(package / "launch_contract.json", contract)

    provenance = base._read(package / "provenance.json")
    provenance.update(
        {
            "schema": "evidence-sets-eval-never-started-provenance-v1",
            "source_package": str(source_package),
            "never_started_manifest_sha256": base._sha(manifest),
            "runtime_failure_audit_sha256": base._sha(
                package / "runtime_failure_audit.json"
            ),
            "wrapper_sha256": base._sha(package / "evidence_eval_never_started.py"),
        }
    )
    for key in ("repair_manifest_sha256", "clean_manifest_sha256", "clean_reuse_receipt_sha256"):
        provenance.pop(key, None)
    base._save(package / "provenance.json", provenance)

    stage = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={DEFAULT_REMOTE_ROOT}
SOURCE={base.REMOTE_SOURCE}
[[ -d \"$ROOT\" ]] || {{ echo \"missing uploaded eval_never_started_v1\" >&2; exit 66; }}
[[ ! -e \"$ROOT/sglang\" ]] || {{ echo \"refusing to replace eval_never_started_v1/sglang\" >&2; exit 73; }}
cp -a \"$SOURCE/sglang\" \"$ROOT/sglang\"
exec {base.SGL_PYTHON} \"$ROOT/evidence_eval_never_started.py\" verify-package --package \"$ROOT\"
"""
    (package / "stage_remote.sh").write_text(stage, encoding="utf-8", newline="\n")
    base._save(package / "static_files.json", base._static_manifest(package))


def prepare(package: Path, *, source_package: Path = DEFAULT_SOURCE_PACKAGE) -> dict[str, Any]:
    tasks = _configure()
    package = package.resolve()
    source_package = source_package.resolve()
    original_verify = repair.verify_package
    repair.verify_package = lambda candidate, require_sglang=True: base.verify_package(
        Path(candidate), require_sglang=require_sglang
    )
    try:
        repair.prepare(package, source_package=source_package)
    finally:
        repair.verify_package = original_verify
    _rewrite_package(package, source_package, tasks)
    verification = verify_package(package, require_sglang=False)
    result = {
        "schema": "evidence-sets-eval-never-started-prepare-v1",
        "status": "prepared_original_budget_continuation",
        "package": str(package),
        "remote_root": str(DEFAULT_REMOTE_ROOT),
        "task_ids": list(tasks),
        "lanes": [row["name"] for row in lane_specs()],
        "total_task_execution_budget": len(tasks) * len(SELECTED_LANES),
        "extra_task_executions": 0,
        "verification": verification,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def verify_package(package: Path, *, require_sglang: bool = True) -> dict[str, Any]:
    tasks = _configure(package / "runtime_failure_audit.json")
    package = package.resolve()
    result = base.verify_package(package, require_sglang=require_sglang)
    contract = base._read(package / "launch_contract.json")
    expected_total = len(tasks) * len(SELECTED_LANES)
    if (
        contract.get("schema") != "evidence-sets-d20-never-started-eval-v1"
        or contract.get("per_lane_task_budget") != len(tasks)
        or contract.get("total_task_execution_budget") != expected_total
        or contract.get("extra_task_executions_requiring_approval") != 0
        or contract.get("cumulative_task_starts_after_continuation") > 80
        or contract.get("launch_authorized") is not False
    ):
        raise RuntimeError("Never-started launch/budget contract changed")
    audit = _read_audit(package / "runtime_failure_audit.json")
    for lane in lane_specs():
        lane_root = package / "lanes" / lane["name"]
        if base._read(lane_root / "lane.json") != lane:
            raise RuntimeError(f"Lane binding changed: {lane['name']}")
        design = base._read(lane_root / "design.json")
        if design.get("task_ids") != list(tasks) or design["limits"].get("tasks") != len(tasks):
            raise RuntimeError(f"Never-started task budget changed: {lane['name']}")
        classifications = {
            row["task_id"]: row.get("classification")
            for row in audit["lanes"][lane["name"]]["tasks"]
        }
        if any(classifications.get(task) != "not_started" for task in tasks):
            raise RuntimeError(f"Previously started task admitted: {lane['name']}")
        gp = design["resolved_configs"]["controller"]["gp_experiments"]
        if gp.get("semantic_query_overflow_policy") != "task_head_tail_preserve_draft_v1":
            raise RuntimeError(f"Query overflow policy changed: {lane['name']}")
    unexpected = sorted(path.name for path in (package / "lanes").iterdir() if path.name not in SELECTED_LANES)
    if unexpected:
        raise RuntimeError(f"Unexpected lanes in package: {unexpected}")
    result.update(
        {
            "lanes": list(SELECTED_LANES),
            "never_started_task_count_per_lane": len(tasks),
            "total_task_execution_budget": expected_total,
            "extra_task_executions": 0,
            "launch_authorized": False,
        }
    )
    return result


def run_lane(package: Path, lane_name: str) -> int:
    _ensure_ascend_environment()
    _configure(package / "runtime_failure_audit.json")
    return repair.run_lane(package, lane_name)


def launch(package: Path, *, approved_extra_task_executions: int) -> dict[str, Any]:
    _ensure_ascend_environment()
    _configure(package / "runtime_failure_audit.json")
    package = package.resolve()
    if approved_extra_task_executions != 0:
        raise RuntimeError(
            "Never-started launch requires --approved-extra-task-executions 0"
        )
    verification = verify_package(package)
    lane_rows = []
    for lane in lane_specs():
        lane_root = package / "lanes" / lane["name"]
        if (lane_root / "run").exists() or (lane_root / "results").exists():
            raise FileExistsError(f"Refusing automatic rerun of lane {lane['name']}")
        lane_rows.append(base._assert_lane_free(lane))
    launch_root = package / "run"
    if launch_root.exists():
        raise FileExistsError("Launch receipt exists; refusing duplicate launch")
    launch_root.mkdir()
    receipt: dict[str, Any] = {
        "schema": "evidence-sets-eval-never-started-launch-v1",
        "status": "dispatching",
        "launched_at": base._now(),
        "approved_extra_task_executions": 0,
        "verification": verification,
        "resource_preflight": lane_rows,
        "supervisors": [],
        "total_task_execution_budget": 12,
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }
    base._save(launch_root / "launch.json", receipt)
    for lane in lane_specs():
        log_path = launch_root / f"supervisor.{lane['name']}.log"
        log = log_path.open("x", encoding="utf-8", newline="\n")
        command = [
            base.SGL_PYTHON,
            str(package / "evidence_eval_never_started.py"),
            "run-lane",
            "--package",
            str(package),
            "--lane",
            lane["name"],
        ]
        process = subprocess.Popen(
            command,
            cwd=package,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        receipt["supervisors"].append(
            {
                "lane": lane["name"],
                "pid": process.pid,
                "command": command,
                "log": str(log_path),
            }
        )
    receipt["status"] = "dispatched"
    base._save(launch_root / "launch.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def status(package: Path) -> dict[str, Any]:
    _configure(package / "runtime_failure_audit.json")
    result = base.status(package)
    result["never_started_task_execution_budget"] = 12
    result["extra_task_executions"] = 0
    result.pop("fixed_budget", None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--package", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--source-package", type=Path, default=DEFAULT_SOURCE_PACKAGE)
    verify = sub.add_parser("verify-package")
    verify.add_argument("--package", type=Path, required=True)
    verify.add_argument("--allow-missing-sglang", action="store_true")
    run = sub.add_parser("run-lane")
    run.add_argument("--package", type=Path, required=True)
    run.add_argument("--lane", choices=tuple(SELECTED_LANES), required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--package", type=Path, required=True)
    launch_parser.add_argument("--approved-extra-task-executions", type=int, required=True)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--package", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args.package, source_package=args.source_package)
        return 0
    if args.command == "verify-package":
        result = verify_package(args.package, require_sglang=not args.allow_missing_sglang)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-lane":
        return run_lane(args.package, args.lane)
    if args.command == "launch":
        launch(args.package, approved_extra_task_executions=args.approved_extra_task_executions)
        return 0
    if args.command == "status":
        status(args.package)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
