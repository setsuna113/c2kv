"""Release the frozen C2 remaining queue after a terminal T02 v6 failure.

This wrapper does not alter the frozen evaluation package.  It records the
actual failed T02 terminal receipt, requires the global abort receipt, checks
that the old waiting relay started no task, and then delegates to the frozen
``run_lane`` only after NPU 0 and every dedicated port are free.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


OLD_WAIT_RELAY_PID = 849483
EXPECTED_TASK_EXECUTIONS = 8
T02_ROOT = Path(
    "/home/liuyancheng/c2kv-evidence-sets-20260916/prepared_v6"
)
T02_SLOT = T02_ROOT / "workers/npu0/slot.json"
T02_STATUS = T02_ROOT / "status.json"
T02_ABORT = T02_ROOT / "abort.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def classify_release(
    slot: Mapping[str, Any],
    global_status: Mapping[str, Any] | None,
    abort: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a truthful release receipt for an accepted terminal state."""

    if slot.get("physical_device") != 0 or slot.get("worker_id") != "npu0":
        raise RuntimeError("T02 release receipt is not the NPU0 worker slot")
    phase = slot.get("phase")
    finished = slot.get("finished_at_epoch")
    if phase in {"completed", "completed_no_work"}:
        if not isinstance(finished, (int, float)):
            raise RuntimeError("Completed T02 slot lacks finished_at_epoch")
        return {
            "release_kind": "successful_terminal",
            "slot_phase": phase,
            "finished_at_epoch": finished,
        }
    if phase != "failed_no_retry":
        return None
    if not isinstance(finished, (int, float)):
        raise RuntimeError("Failed T02 slot lacks finished_at_epoch")
    if not isinstance(global_status, Mapping) or global_status.get("phase") != "failed_no_retry":
        raise RuntimeError("Failed T02 slot lacks matching global failed_no_retry status")
    if not isinstance(abort, Mapping) or not isinstance(abort.get("error"), str):
        raise RuntimeError("Failed T02 slot lacks a structured global abort receipt")
    if global_status.get("abort_receipt") != "abort.json":
        raise RuntimeError("T02 global failure does not bind abort.json")
    if slot.get("error") != "Global run aborted":
        raise RuntimeError("NPU0 failure was not the recorded global-abort release")
    return {
        "release_kind": "terminal_global_failure",
        "slot_phase": phase,
        "finished_at_epoch": finished,
        "slot_error": slot["error"],
        "global_phase": global_status["phase"],
        "coordinator_exit_code": global_status.get("coordinator_exit_code"),
        "abort_phase": abort.get("phase"),
        "abort_error_type": abort.get("error_type"),
        "abort_error": abort["error"],
        "abort_updated_at_epoch": abort.get("updated_at_epoch"),
    }


def verify_zero_started(package: Path) -> dict[str, Any]:
    lane = package / "lanes/C2"
    run = lane / "run"
    results = lane / "results"
    if run.exists() or results.exists():
        raise RuntimeError("Frozen C2 queue already has run/results evidence; refusing release")
    launch = _read(package / "run/launch.json")
    relay = _read(package / "relay/C2.json")
    if (
        launch.get("approved_task_executions") != EXPECTED_TASK_EXECUTIONS
        or launch.get("automatic_retries") != 0
        or launch.get("automatic_reruns") != 0
    ):
        raise RuntimeError("Original C2 launch receipt changed")
    if (
        relay.get("state") != "waiting_for_t02_npu0_terminal"
        or relay.get("supervisor_pid") != OLD_WAIT_RELAY_PID
    ):
        raise RuntimeError("Original C2 waiting relay receipt changed")
    return {
        "task_starts": 0,
        "lane_run_absent": True,
        "lane_results_absent": True,
        "old_relay_state": relay["state"],
        "old_relay_pid": relay["supervisor_pid"],
    }


def _dependency_receipt() -> dict[str, Any] | None:
    if not T02_SLOT.is_file():
        return None
    slot = _read(T02_SLOT)
    status = _read(T02_STATUS) if T02_STATUS.is_file() else None
    abort = _read(T02_ABORT) if T02_ABORT.is_file() else None
    return classify_release(slot, status, abort)


def wait_run(package: Path) -> int:
    package = package.resolve()
    sys.path.insert(0, str(package))
    try:
        import evidence_eval_c2_remaining as evaluation
    finally:
        sys.path.pop(0)
    if Path(evaluation.__file__).resolve() != (package / "evidence_eval_c2_remaining.py"):
        raise RuntimeError("Loaded C2 evaluator from an unexpected path")
    evaluation._ensure_ascend_environment()
    evaluation._configure()
    release_root = package / "release_v2"
    status_path = release_root / "status.json"
    if status_path.exists():
        raise FileExistsError("Terminal-failure release already has a status receipt")
    status: dict[str, Any] = {
        "schema": "evidence-sets-c2-terminal-failure-release-v2",
        "state": "waiting_for_terminal_t02_release",
        "supervisor_pid": os.getpid(),
        "physical_device": 0,
        "task_execution_cap": EXPECTED_TASK_EXECUTIONS,
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "old_wait_relay_pid": OLD_WAIT_RELAY_PID,
    }
    evaluation.base._save(status_path, status)
    try:
        status["package_verification"] = evaluation.verify_package(package)
        status["zero_started_before_release"] = verify_zero_started(package)
        while True:
            dependency = _dependency_receipt()
            if dependency is not None:
                break
            time.sleep(20)
        status.update(
            state="waiting_for_fresh_npu0_and_ports",
            predecessor_receipt=dependency,
            predecessor_hashes={
                "slot": evaluation.base._sha(T02_SLOT),
                "status": evaluation.base._sha(T02_STATUS),
                "abort": evaluation.base._sha(T02_ABORT),
            },
        )
        evaluation.base._save(status_path, status)
        lane = evaluation.lane_specs()[0]
        while True:
            try:
                resource = evaluation.base._assert_lane_free(lane)
                break
            except RuntimeError as error:
                status["last_resource_wait_error"] = str(error)
                evaluation.base._save(status_path, status)
                time.sleep(20)
        status["zero_started_at_handoff"] = verify_zero_started(package)
        status.update(state="starting_frozen_c2_remaining", resource_preflight=resource)
        evaluation.base._save(status_path, status)
        code = evaluation.run_lane(package)
        status.update(
            state="completed" if code == 0 else "failed_no_rerun",
            lane_returncode=code,
            finished_at=evaluation.base._now(),
        )
        evaluation.base._save(status_path, status)
        return code
    except BaseException as error:
        status.update(
            state="failed_no_rerun",
            error=f"{type(error).__name__}: {error}",
        )
        evaluation.base._save(status_path, status)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return 2


def dispatch(package: Path) -> dict[str, Any]:
    package = package.resolve()
    zero_started = verify_zero_started(package)
    release_root = package / "release_v2"
    if release_root.exists():
        raise FileExistsError("Terminal-failure release already exists")
    release_root.mkdir()
    log_path = release_root / "supervisor.log"
    command = [
        "/home/liuyancheng/envs/sgl/bin/python",
        str(Path(__file__).resolve()),
        "wait-run",
        "--package",
        str(package),
    ]
    receipt: dict[str, Any] = {
        "schema": "evidence-sets-c2-terminal-failure-dispatch-v2",
        "state": "dispatching",
        "old_wait_relay_pid": OLD_WAIT_RELAY_PID,
        "old_wait_relay_stopped_before_dispatch": True,
        "zero_started": zero_started,
        "command": command,
        "supervisor_pid": None,
        "log": str(log_path),
        "task_execution_cap": EXPECTED_TASK_EXECUTIONS,
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }
    launch_path = release_root / "launch.json"
    with log_path.open("x", encoding="utf-8", newline="\n") as log:
        process = subprocess.Popen(
            command,
            cwd=package,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt.update(state="dispatched", supervisor_pid=process.pid)
    temporary = launch_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(launch_path)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dispatch", "wait-run"):
        child = sub.add_parser(name)
        child.add_argument("--package", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "dispatch":
        dispatch(args.package)
        return 0
    if args.command == "wait-run":
        return wait_run(args.package)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
