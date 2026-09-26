"""Freeze and relay the one-time 41-cell repair before T02 uses NPU0-3.

The frozen runtime audit supplies the original 34 failed/interrupted cells.
The terminated C2 continuation supplies one observed runtime failure and six
never-started cells, bringing C2 to 15 and the total to 41.  No clean or
successfully completed cell can enter this package.  A single waiting
supervisor per lane starts only after its predecessor (when present) is
terminal and the physical device and dedicated ports are free.  There is no
automatic retry or rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import evidence_eval as base
import evidence_eval_repair as repair


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVAL_ROOT = REPO / "outputs/history_system_search/evidence_sets_v1/eval"
DEFAULT_SOURCE_PACKAGE = EVAL_ROOT / "prepared_v1_final"
DEFAULT_OUTPUT = EVAL_ROOT / "failed_repair/prepared_v1"
DEFAULT_REMOTE_ROOT = PurePosixPath(
    "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_failed_repair_v1"
)
C3_PREDECESSOR_ROOT = PurePosixPath(
    "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_never_started_c2c3_v2"
)
AUDIT_PATH = EVAL_ROOT / "runtime_failure_audit.json"
RERANKER_SMOKE_PATH = EVAL_ROOT / "reranker_batch1_long_payload_smoke.json"
C2_FOLLOWUP_ROOT = EVAL_ROOT / "remote_receipts/c2_v2_failed"
C2_FOLLOWUP_STAGE_PATH = C2_FOLLOWUP_ROOT / "stage_manifest.json"
C2_FOLLOWUP_STATUS_PATH = C2_FOLLOWUP_ROOT / "status.json"
ACTIVE_AUDIT_PATH = AUDIT_PATH
ACTIVE_C2_FOLLOWUP_STAGE_PATH = C2_FOLLOWUP_STAGE_PATH
SELECTED_LANES = {
    "C0": {
        "selector": "candidate_rule",
        "physical_device": 0,
        "engine_port": 37200,
        "task_port_base": 37300,
        "model_smokes": ("embedding",),
        "after_status_path": None,
    },
    "C2": {
        "selector": "reranker",
        "physical_device": 2,
        "engine_port": 37210,
        "task_port_base": 37320,
        "model_smokes": ("embedding", "reranker"),
        "after_status_path": None,
    },
    "C3": {
        "selector": "local_llm",
        "physical_device": 1,
        "engine_port": 37220,
        "task_port_base": 37340,
        "model_smokes": ("embedding", "selector"),
        "after_status_path": str(C3_PREDECESSOR_ROOT / "lanes/C3/run/status.json"),
    },
    "C5": {
        "selector": "parameter_source",
        "physical_device": 3,
        "engine_port": 37230,
        "task_port_base": 37360,
        "model_smokes": ("embedding",),
        "after_status_path": None,
    },
}
EXPECTED_COUNTS = {"C0": 9, "C2": 15, "C3": 8, "C5": 9}
FAILED_CLASSIFICATIONS = frozenset({"runtime_failure", "interrupted"})
APPROVED_REPAIR_EXECUTIONS = 41
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


def _audit_tasks(
    path: Path = AUDIT_PATH,
    c2_followup_path: Path | None = None,
) -> dict[str, tuple[str, ...]]:
    audit = _read_audit(path)
    followup_path = ACTIVE_C2_FOLLOWUP_STAGE_PATH if c2_followup_path is None else c2_followup_path
    followup = base._read(followup_path)
    followup_rows = followup.get("task_outcomes", [])
    expected_followup = (
        ("multi_turn_long_context_120", "runtime_failure_in_denominator", False),
        ("multi_turn_base_130", "not_started", None),
        ("multi_turn_long_context_130", "not_started", None),
        ("multi_turn_base_170", "not_started", None),
        ("multi_turn_long_context_170", "not_started", None),
        ("multi_turn_base_190", "not_started", None),
        ("multi_turn_long_context_190", "not_started", None),
    )
    observed_followup = tuple(
        (row.get("task_id"), row.get("outcome"), row.get("runtime_completed"))
        for row in followup_rows
    )
    if observed_followup != expected_followup:
        raise RuntimeError(f"C2 continuation audit changed: {observed_followup!r}")
    by_lane: dict[str, tuple[str, ...]] = {}
    for lane_name in SELECTED_LANES:
        rows = audit.get("lanes", {}).get(lane_name, {}).get("tasks", [])
        failed_tasks = tuple(
            row["task_id"]
            for row in rows
            if row.get("classification") in FAILED_CLASSIFICATIONS
        )
        tasks = failed_tasks
        if lane_name == "C2":
            tasks = (*tasks, *(row[0] for row in expected_followup))
        if len(tasks) != EXPECTED_COUNTS[lane_name] or len(tasks) != len(set(tasks)):
            raise RuntimeError(
                f"Audit-derived failed task count changed for {lane_name}: {tasks}"
            )
        classifications = {
            row["task_id"]: row.get("classification") for row in rows
        }
        if any(
            classifications[task] not in FAILED_CLASSIFICATIONS
            for task in failed_tasks
        ):
            raise RuntimeError(f"Nonfailed cell admitted for {lane_name}")
        by_lane[lane_name] = tasks
    if sum(map(len, by_lane.values())) != APPROVED_REPAIR_EXECUTIONS:
        raise RuntimeError("Failed repair/continuation total changed from 41")
    return by_lane


def _configure(
    audit_path: Path = AUDIT_PATH,
    c2_followup_path: Path | None = None,
) -> dict[str, tuple[str, ...]]:
    global ACTIVE_AUDIT_PATH, ACTIVE_C2_FOLLOWUP_STAGE_PATH
    ACTIVE_AUDIT_PATH = audit_path
    candidate_followup = (
        audit_path.parent / "c2_v2_followup.stage_manifest.json"
        if c2_followup_path is None and (audit_path.parent / "c2_v2_followup.stage_manifest.json").is_file()
        else c2_followup_path or C2_FOLLOWUP_STAGE_PATH
    )
    ACTIVE_C2_FOLLOWUP_STAGE_PATH = candidate_followup
    tasks = _audit_tasks(audit_path, candidate_followup)
    repair.DEFAULT_REMOTE_ROOT = DEFAULT_REMOTE_ROOT
    repair.APPROVED_EXTRA_EXECUTIONS = APPROVED_REPAIR_EXECUTIONS
    # The generic repair freezer needs only a temporary common manifest; this
    # wrapper replaces it with the exact per-lane manifests before verification.
    repair.REPAIR_TASKS = tasks["C0"]
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
            range(row["task_port_base"], row["task_port_base"] + len(tasks[name]))
        )
        rows.append(row)
    ports = [port for row in rows for port in [row["engine_port"], *row["task_ports"]]]
    if len(ports) != len(set(ports)):
        raise RuntimeError("Failed-repair lane ports overlap")
    return rows


def _rewrite_package(
    package: Path,
    source_package: Path,
    tasks_by_lane: Mapping[str, tuple[str, ...]],
) -> None:
    lanes_root = package / "lanes"
    (package / "tasks.repair15.json").unlink(missing_ok=True)
    (package / "tasks.clean_reuse5.json").unlink(missing_ok=True)
    (package / "clean_reuse.remote.json").unlink(missing_ok=True)
    shutil.copyfile(RERANKER_SMOKE_PATH, package / RERANKER_SMOKE_PATH.name)
    shutil.copyfile(
        C2_FOLLOWUP_STAGE_PATH, package / "c2_v2_followup.stage_manifest.json"
    )
    shutil.copyfile(C2_FOLLOWUP_STATUS_PATH, package / "c2_v2_followup.status.json")

    lane_receipts = []
    manifest_hashes: dict[str, str] = {}
    for lane in lane_specs():
        lane_name = lane["name"]
        tasks = tasks_by_lane[lane_name]
        manifest = package / f"tasks.failed_repair.{lane_name}.json"
        manifest_value = repair._manifest(
            f"D20-{lane_name}-failed-repair{len(tasks)}",
            "development_search_one_time_runtime_repair",
            tasks,
        )
        base._save(manifest, manifest_value)
        manifest_hashes[lane_name] = base._sha(manifest)
        lane_root = lanes_root / lane["name"]
        design_path = lane_root / "design.json"
        design = base._read(design_path)
        design["candidate_id"] = (
            f"evidence_sets_v1_h0_{lane_name.lower()}_failed_repair_r1"
        )
        design["run_id_template"] = (
            "a_history_evidence_sets_v1_failed_repair_" + lane_name.lower()
        )
        design["task_ids"] = list(tasks)
        design["task_manifest_sha256"] = base._sha(manifest)
        design["limits"] = {**design["limits"], "tasks": len(tasks)}
        gp = design["resolved_configs"]["controller"]["gp_experiments"]
        gp["local_models"]["reranker"]["batch_size"] = 1
        controller_path = lane_root / "runtime" / design["runtime"]["controller"]
        controller = base._read(controller_path)
        controller["gp_experiments"] = gp
        base._save(controller_path, controller)
        design["source_files"][design["runtime"]["controller"]] = base._sha(
            controller_path
        )
        contract = design["search_contract"]
        contract["fixed_denominator"] = f"audit_failed_or_interrupted{len(tasks)}_only"
        contract["repair_task_count"] = len(tasks)
        contract["runtime_failure_audit"] = str(package / "runtime_failure_audit.json")
        contract["one_time_repair_execution_count"] = len(tasks)
        contract["original_failed_attempts_retained"] = True
        contract["not_started_cells_included"] = 6 if lane_name == "C2" else 0
        for key in (
            "clean_reuse_task_count",
            "clean_reuse_receipt",
        ):
            contract.pop(key, None)
        base._save(design_path, design)
        base._save(lane_root / "gp.json", gp)
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
                "after_status_path": lane["after_status_path"],
                "status_path": str(
                    DEFAULT_REMOTE_ROOT
                    / "lanes"
                    / lane["name"]
                    / "run/status.json"
                ),
                "design_sha256": base._sha(design_path),
            }
        )

    shutil.copyfile(Path(__file__), package / "evidence_eval_failed_repair.py")
    contract = base._read(package / "launch_contract.json")
    contract.update(
        {
            "schema": "evidence-sets-d20-failed-repair-eval-v1",
            "status": "frozen_one_time_repair_waiting_for_predecessor_or_free_device",
            "launch_authorized": False,
            "launch_gate": {
                "flag": "--approved-extra-task-executions",
                "required_value": APPROVED_REPAIR_EXECUTIONS,
            },
            "remote_root": str(DEFAULT_REMOTE_ROOT),
            "c3_predecessor_root": str(C3_PREDECESSOR_ROOT),
            "fixed_task_manifests": {
                lane: f"tasks.failed_repair.{lane}.json" for lane in SELECTED_LANES
            },
            "per_lane_task_budget": {
                lane: len(tasks) for lane, tasks in tasks_by_lane.items()
            },
            "total_task_execution_budget": APPROVED_REPAIR_EXECUTIONS,
            "original_task_execution_budget": 80,
            "observed_original_task_starts": 54,
            "c2_continuation_task_starts": 1,
            "c2_continuation_not_started_cells": 6,
            "one_time_repair_task_executions": APPROVED_REPAIR_EXECUTIONS,
            "failed_or_interrupted_repair_task_executions": 35,
            "never_started_continuation_task_executions": 6,
            "authorization_basis": "complete_authorized_D20_after_runtime_interface_repair",
            "original_failed_attempts_retained": True,
            "not_started_cells_included": 6,
            "automatic_retries": 0,
            "automatic_reruns": 0,
            "relay_poll_seconds": 20,
            "relay_terminal_predecessor_states": ["completed", "failed_no_rerun"],
            "lanes": lane_receipts,
        }
    )
    for key in (
        "clean_reuse_manifest",
        "clean_reuse_receipt",
        "clean_cells_reused",
        "failed_or_incomplete_cells_not_reused",
        "fixed_task_manifest",
    ):
        contract.pop(key, None)
    base._save(package / "launch_contract.json", contract)

    provenance = base._read(package / "provenance.json")
    provenance.update(
        {
            "schema": "evidence-sets-eval-failed-repair-provenance-v1",
            "source_package": str(source_package),
            "failed_repair_manifest_sha256": manifest_hashes,
            "runtime_failure_audit_sha256": base._sha(
                package / "runtime_failure_audit.json"
            ),
            "c2_v2_followup_stage_manifest_sha256": base._sha(
                package / "c2_v2_followup.stage_manifest.json"
            ),
            "c2_v2_followup_status_sha256": base._sha(
                package / "c2_v2_followup.status.json"
            ),
            "reranker_batch1_long_payload_smoke_sha256": base._sha(
                package / RERANKER_SMOKE_PATH.name
            ),
            "wrapper_sha256": base._sha(package / "evidence_eval_failed_repair.py"),
        }
    )
    for key in ("repair_manifest_sha256", "clean_manifest_sha256", "clean_reuse_receipt_sha256"):
        provenance.pop(key, None)
    base._save(package / "provenance.json", provenance)

    stage = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={DEFAULT_REMOTE_ROOT}
SOURCE={base.REMOTE_SOURCE}
[[ -d \"$ROOT\" ]] || {{ echo \"missing uploaded eval_failed_repair_v1\" >&2; exit 66; }}
[[ ! -e \"$ROOT/sglang\" ]] || {{ echo \"refusing to replace eval_failed_repair_v1/sglang\" >&2; exit 73; }}
cp -a \"$SOURCE/sglang\" \"$ROOT/sglang\"
exec {base.SGL_PYTHON} \"$ROOT/evidence_eval_failed_repair.py\" verify-package --package \"$ROOT\"
"""
    (package / "stage_remote.sh").write_text(stage, encoding="utf-8", newline="\n")
    base._save(package / "static_files.json", base._static_manifest(package))


def prepare(package: Path, *, source_package: Path = DEFAULT_SOURCE_PACKAGE) -> dict[str, Any]:
    tasks_by_lane = _configure()
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
    _rewrite_package(package, source_package, tasks_by_lane)
    verification = verify_package(package, require_sglang=False)
    result = {
        "schema": "evidence-sets-eval-failed-repair-prepare-v1",
        "status": "prepared_one_time_repair_relay",
        "package": str(package),
        "remote_root": str(DEFAULT_REMOTE_ROOT),
        "task_ids_by_lane": {
            lane: list(tasks) for lane, tasks in tasks_by_lane.items()
        },
        "lanes": [row["name"] for row in lane_specs()],
        "total_task_execution_budget": APPROVED_REPAIR_EXECUTIONS,
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "verification": verification,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def verify_package(package: Path, *, require_sglang: bool = True) -> dict[str, Any]:
    tasks_by_lane = _configure(package / "runtime_failure_audit.json")
    package = package.resolve()
    result = base.verify_package(package, require_sglang=require_sglang)
    contract = base._read(package / "launch_contract.json")
    if (
        contract.get("schema") != "evidence-sets-d20-failed-repair-eval-v1"
        or contract.get("per_lane_task_budget")
        != {lane: len(tasks) for lane, tasks in tasks_by_lane.items()}
        or contract.get("total_task_execution_budget") != APPROVED_REPAIR_EXECUTIONS
        or contract.get("one_time_repair_task_executions")
        != APPROVED_REPAIR_EXECUTIONS
        or contract.get("not_started_cells_included") != 6
        or contract.get("launch_authorized") is not False
    ):
        raise RuntimeError("Failed-repair launch/budget contract changed")
    smoke = base._read(package / "selector_long_payload_smoke.json")
    if (
        smoke.get("status") != "passed"
        or smoke.get("prompt_tokens") != 72338
        or smoke.get("selector_max_input_tokens") != 131056
        or smoke.get("legal_action") is not True
    ):
        raise RuntimeError("C3 long-payload selector smoke receipt changed")
    reranker_smoke = base._read(package / RERANKER_SMOKE_PATH.name)
    if (
        reranker_smoke.get("status") != "passed"
        or reranker_smoke.get("batch_size") != 1
        or reranker_smoke.get("candidate_count") != 1
        or reranker_smoke.get("retained_pair_tokens") != [32768]
        or reranker_smoke.get("scores_finite") is not True
        or reranker_smoke.get("actor_or_task_execution") is not False
    ):
        raise RuntimeError("C2 batch-one long-payload reranker smoke receipt changed")
    audit = _read_audit(package / "runtime_failure_audit.json")
    c2_followup_ids = {
        row["task_id"]
        for row in base._read(package / "c2_v2_followup.stage_manifest.json").get(
            "task_outcomes", []
        )
    }
    for lane in lane_specs():
        lane_root = package / "lanes" / lane["name"]
        if base._read(lane_root / "lane.json") != lane:
            raise RuntimeError(f"Lane binding changed: {lane['name']}")
        design = base._read(lane_root / "design.json")
        tasks = tasks_by_lane[lane["name"]]
        if design.get("task_ids") != list(tasks) or design["limits"].get("tasks") != len(tasks):
            raise RuntimeError(f"Failed-repair task budget changed: {lane['name']}")
        classifications = {
            row["task_id"]: row.get("classification")
            for row in audit["lanes"][lane["name"]]["tasks"]
        }
        if any(
            not (lane["name"] == "C2" and task in c2_followup_ids)
            and classifications.get(task) not in FAILED_CLASSIFICATIONS
            for task in tasks
        ):
            raise RuntimeError(f"Nonfailed task admitted: {lane['name']}")
        manifest = package / f"tasks.failed_repair.{lane['name']}.json"
        if base._read(manifest).get("task_ids") != list(tasks):
            raise RuntimeError(f"Failed-repair manifest changed: {lane['name']}")
        gp = design["resolved_configs"]["controller"]["gp_experiments"]
        if gp.get("semantic_query_overflow_policy") != "task_head_tail_preserve_draft_v1":
            raise RuntimeError(f"Query overflow policy changed: {lane['name']}")
        if gp.get("local_models", {}).get("reranker", {}).get("batch_size") != 1:
            raise RuntimeError(f"Reranker batch size changed: {lane['name']}")
    unexpected = sorted(path.name for path in (package / "lanes").iterdir() if path.name not in SELECTED_LANES)
    if unexpected:
        raise RuntimeError(f"Unexpected lanes in package: {unexpected}")
    result.update(
        {
            "lanes": list(SELECTED_LANES),
            "failed_task_count_per_lane": {
                lane: len(tasks) for lane, tasks in tasks_by_lane.items()
            },
            "total_task_execution_budget": APPROVED_REPAIR_EXECUTIONS,
            "launch_authorized": False,
        }
    )
    return result


def run_lane(package: Path, lane_name: str) -> int:
    _ensure_ascend_environment()
    _configure(package / "runtime_failure_audit.json")
    package = package.resolve()
    approval = base._read(package / "run/launch.json")
    if approval.get("approved_extra_task_executions") != APPROVED_REPAIR_EXECUTIONS:
        raise RuntimeError("Failed-repair launch receipt is missing the approved 41 executions")
    original_read = base._read

    def approved_read(path: Path) -> dict[str, Any]:
        value = original_read(path)
        if Path(path).resolve() == (package / "launch_contract.json").resolve():
            value = {**value, "launch_authorized": True}
        return value

    base._read = approved_read
    try:
        return base.run_lane(package, lane_name)
    finally:
        base._read = original_read


def _terminal_predecessor_receipt(lane: Mapping[str, Any]) -> dict[str, Any] | None:
    configured = lane.get("after_status_path")
    if configured is None:
        return {"kind": "no_predecessor", "state": "terminal"}
    path = Path(configured)
    if not path.is_file():
        return None
    receipt = base._read(path)
    if receipt.get("state") not in {"completed", "failed_no_rerun"}:
        return None
    if receipt.get("physical_device") != lane["physical_device"]:
        raise RuntimeError("Predecessor terminal receipt device differs from repair lane")
    return {
        "kind": "lane_status",
        "path": str(path),
        "sha256": base._sha(path),
        "state": receipt["state"],
        "finished_at": receipt.get("finished_at"),
    }


def wait_lane(package: Path, lane_name: str) -> int:
    _ensure_ascend_environment()
    _configure(package / "runtime_failure_audit.json")
    package = package.resolve()
    lane = next(row for row in lane_specs() if row["name"] == lane_name)
    relay_root = package / "relay"
    relay_root.mkdir(exist_ok=True)
    relay_path = relay_root / f"{lane_name}.json"
    if relay_path.exists():
        raise FileExistsError(f"Relay receipt exists for {lane_name}; no automatic restart")
    receipt: dict[str, Any] = {
        "schema": "evidence-sets-failed-repair-relay-v1",
        "lane": lane_name,
        "state": "waiting_for_predecessor_terminal",
        "supervisor_pid": os.getpid(),
        "physical_device": lane["physical_device"],
        "after_status_path": lane["after_status_path"],
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }
    base._save(relay_path, receipt)
    while True:
        dependency = _terminal_predecessor_receipt(lane)
        if dependency is not None:
            break
        time.sleep(20)
    receipt.update(state="waiting_for_free_device", predecessor_receipt=dependency)
    base._save(relay_path, receipt)
    while True:
        try:
            resource = base._assert_lane_free(lane)
            break
        except RuntimeError as error:
            receipt["last_resource_wait_error"] = str(error)
            base._save(relay_path, receipt)
            time.sleep(20)
    receipt.update(state="starting_failed_repair", resource_preflight=resource)
    base._save(relay_path, receipt)
    code = run_lane(package, lane_name)
    receipt.update(
        state="completed" if code == 0 else "failed_no_rerun",
        lane_returncode=code,
        finished_at=base._now(),
    )
    base._save(relay_path, receipt)
    return code


def launch(package: Path, *, approved_extra_task_executions: int) -> dict[str, Any]:
    _ensure_ascend_environment()
    _configure(package / "runtime_failure_audit.json")
    package = package.resolve()
    if approved_extra_task_executions != APPROVED_REPAIR_EXECUTIONS:
        raise RuntimeError(
            "Failed repair launch requires --approved-extra-task-executions 41"
        )
    verification = verify_package(package)
    for lane in lane_specs():
        lane_root = package / "lanes" / lane["name"]
        if (
            (lane_root / "run").exists()
            or (lane_root / "results").exists()
            or (package / "relay" / f"{lane['name']}.json").exists()
        ):
            raise FileExistsError(f"Refusing automatic rerun of lane {lane['name']}")
    launch_root = package / "run"
    if launch_root.exists():
        raise FileExistsError("Launch receipt exists; refusing duplicate launch")
    launch_root.mkdir()
    receipt: dict[str, Any] = {
        "schema": "evidence-sets-eval-failed-repair-launch-v1",
        "status": "dispatching_waiting_relays",
        "launched_at": base._now(),
        "approved_extra_task_executions": APPROVED_REPAIR_EXECUTIONS,
        "verification": verification,
        "supervisors": [],
        "total_task_execution_budget": APPROVED_REPAIR_EXECUTIONS,
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }
    base._save(launch_root / "launch.json", receipt)
    for lane in lane_specs():
        log_path = launch_root / f"supervisor.{lane['name']}.log"
        log = log_path.open("x", encoding="utf-8", newline="\n")
        command = [
            base.SGL_PYTHON,
            str(package / "evidence_eval_failed_repair.py"),
            "wait-lane",
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
    receipt["status"] = "waiting_relays_dispatched"
    base._save(launch_root / "launch.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def status(package: Path) -> dict[str, Any]:
    _configure(package / "runtime_failure_audit.json")
    result = base.status(package)
    relays = []
    for lane_name in SELECTED_LANES:
        relay_path = package / "relay" / f"{lane_name}.json"
        relays.append(
            base._read(relay_path)
            if relay_path.is_file()
            else {
                "lane": lane_name,
                "state": "not_dispatched",
                "automatic_retries": 0,
                "automatic_reruns": 0,
            }
        )
    result["repair_task_execution_budget"] = APPROVED_REPAIR_EXECUTIONS
    result["failed_or_interrupted_repair_task_executions"] = 35
    result["never_started_continuation_task_executions"] = 6
    result["clean_cells_reused"] = 0
    result["relay"] = relays
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
    wait_lane_parser = sub.add_parser("wait-lane")
    wait_lane_parser.add_argument("--package", type=Path, required=True)
    wait_lane_parser.add_argument("--lane", choices=tuple(SELECTED_LANES), required=True)
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
    if args.command == "wait-lane":
        return wait_lane(args.package, args.lane)
    if args.command == "status":
        status(args.package)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
