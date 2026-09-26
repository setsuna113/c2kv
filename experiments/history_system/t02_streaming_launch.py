"""Launch bounded T02 streaming recovery without cross-worker failure aborts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from t02_parallel import read, save
from t02_parallel_launch import available, launch, owned_stop, require_launch_authorization, verify
from t02_streaming_recovery import (
    TERMINAL_TASK_STATUSES,
    initialize_ledger,
    mark_pilot_failed,
)


def _queue_done(root: Path) -> bool:
    path = root / "ledger.json"
    if not path.exists():
        return False
    ledger = read(path)
    return all(row["status"] in TERMINAL_TASK_STATUSES for row in ledger["tasks"])


def _reconcile_pilot_owner_failure(root: Path, worker_id: str, error: BaseException) -> bool:
    """Release pilot waiters when its owner dies before writing either marker."""

    if (root / "pilot_passed.json").exists() or (root / "pilot_failed.json").exists():
        return False
    ledger_path = root / "ledger.json"
    if not ledger_path.exists():
        return False
    ledger = read(ledger_path)
    pilot_id = ledger.get("pilot_task_id")
    rows = [row for row in ledger.get("tasks", []) if row.get("task_id") == pilot_id]
    if (len(rows) != 1 or rows[0].get("worker_id") != worker_id
            or rows[0].get("status") != "started"):
        return False
    mark_pilot_failed(root, worker_id, pilot_id, error)
    return True


def slot(root: str | Path, worker_id: str) -> int:
    root = Path(root).resolve()
    contract = read(root / "contract.json")
    require_launch_authorization(root, contract)
    matches = [row for row in contract["workers"] if row["worker_id"] == worker_id]
    if len(matches) != 1:
        raise ValueError(f"contract must define worker exactly once: {worker_id}")
    worker = matches[0]
    directory = root / "workers" / worker_id
    status_path = directory / "slot.json"
    if status_path.exists():
        raise FileExistsError("Slot already has a start receipt; no automatic restart")
    engine = collector = None
    exit_code = 0
    status = {
        "schema": "t02-streaming-slot-v1",
        "phase": "waiting_for_released_device",
        "worker_id": worker_id,
        "supervisor_pid": os.getpid(),
        "physical_device": worker["physical_device"],
    }
    save(status_path, status)
    try:
        while True:
            if _queue_done(root):
                status.update(
                    phase="completed_no_work",
                    reason="streaming_source_queue_already_terminal",
                )
                return 0
            preflight = available(root, worker)
            if preflight is not None:
                break
            time.sleep(20)
        status.update(phase="starting_engine", resource_preflight=preflight)
        engine = launch([
            "bash", str(root / "scripts" / "engine.sh"),
            "--resource-coordination-approved",
            "--physical-device", str(worker["physical_device"]),
            "--port", str(worker["port"]),
        ], root, f"workers/{worker_id}/engine.log")
        status["engine_pid"] = engine.pid
        save(status_path, status)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if engine.poll() is not None:
                raise RuntimeError("owned engine exited during startup")
            try:
                with opener.open(
                    f"http://127.0.0.1:{worker['port']}/health", timeout=3
                ) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(5)
        else:
            raise TimeoutError("Engine did not become ready within 900 seconds")
        collector = launch([
            "bash", str(root / "scripts" / "parallel_worker.sh"),
            worker_id, str(worker["physical_device"]), str(worker["port"]),
        ], root, f"workers/{worker_id}/worker.log")
        status.update(phase="worker_running", worker_pid=collector.pid)
        save(status_path, status)
        while collector.poll() is None:
            if engine.poll() is not None:
                owned_stop(collector)
                raise RuntimeError("owned engine exited while streaming worker was active")
            time.sleep(5)
        exit_code = collector.returncode
        status.update(
            phase="completed" if exit_code == 0 else "failed_no_retry",
            worker_exit_code=exit_code,
            global_abort_required=False,
        )
        return exit_code
    except Exception as error:
        exit_code = 1
        status.update(
            phase="failed_no_retry",
            error_type=type(error).__name__,
            error=str(error),
            global_abort_required=False,
        )
        return 1
    finally:
        if exit_code:
            _reconcile_pilot_owner_failure(
                root,
                worker_id,
                RuntimeError("pilot owner slot exited before recording a pilot result"),
            )
        owned_stop(collector)
        owned_stop(engine)
        status["finished_at_epoch"] = time.time()
        save(status_path, status)


def supervise(root: str | Path) -> int:
    root = Path(root).resolve()
    verified_files = verify(root)
    if (root / "started.json").exists():
        raise FileExistsError("Streaming run already started; automatic restart is disabled")
    contract = read(root / "contract.json")
    require_launch_authorization(root, contract)
    if contract.get("max_source_tasks_per_worker") != len(
        read(root / "recovery" / "recovery_contract.json")["source_tasks"]
    ):
        raise ValueError("streaming workers must be able to drain the full queue after peer loss")
    if contract.get("snapshot_cap_per_worker") != 2:
        raise ValueError("streaming recovery snapshot cap must be exactly two live states")
    initialize_ledger(root)
    processes: list[subprocess.Popen] = []
    coordinator = None
    status = {
        "schema": "t02-streaming-launch-v1",
        "phase": "starting_workers",
        "verified_files": verified_files,
        "supervisor_pid": os.getpid(),
        "started_at_epoch": time.time(),
        "slots": [],
    }
    save(root / "started.json", status)
    save(root / "status.json", {
        "phase": "streaming_recovery",
        "training_allowed": False,
        "updated_at_epoch": time.time(),
    })
    try:
        coordinator = launch([
            sys.executable,
            str(root / "history_system" / "t02_streaming_parallel.py"),
            "coordinate", "--root", str(root),
        ], root, "coordinator.log")
        status["coordinator_pid"] = coordinator.pid
        for worker in contract["workers"]:
            process = launch([
                sys.executable,
                str(root / "history_system" / "t02_streaming_launch.py"),
                "--root", str(root),
                "--slot", worker["worker_id"],
            ], root, f"workers/{worker['worker_id']}/supervisor.log")
            processes.append(process)
            status["slots"].append({
                "worker_id": worker["worker_id"],
                "pid": process.pid,
                "physical_device": worker["physical_device"],
            })
            save(root / "started.json", status)
        coordinator_code = coordinator.wait()
        # Training may only start after every worker supervisor has released its
        # engine, including locally failed workers.
        slot_codes = [process.wait() for process in processes]
        status["slot_exit_codes"] = slot_codes
        status["coordinator_exit_code"] = coordinator_code
        save(root / "started.json", status)
        if coordinator_code:
            return coordinator_code
        run_status = read(root / "status.json")
        if run_status.get("phase") != "labels_completed" or run_status.get("training_allowed") is not True:
            raise RuntimeError("coordinator returned success without complete streaming labels")
        training_worker = next(
            row for row in contract["workers"] if row["physical_device"] == 6
        )
        while available(root, training_worker) is None:
            time.sleep(20)
        trainer = launch([
            "bash", str(root / "scripts" / "train.sh"),
            "--resource-coordination-approved", "--physical-device", "6",
        ], root, "train.log")
        save(root / "status.json", {
            "phase": "training",
            "training_allowed": True,
            "trainer_pid": trainer.pid,
            "updated_at_epoch": time.time(),
        })
        code = trainer.wait()
        save(root / "status.json", {
            "phase": "completed" if code == 0 else "training_partial_or_failed",
            "training_allowed": True,
            "trainer_exit_code": code,
            "updated_at_epoch": time.time(),
        })
        return code
    finally:
        owned_stop(coordinator)
        for process in processes:
            owned_stop(process)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--slot")
    args = parser.parse_args()
    return slot(args.root, args.slot) if args.slot else supervise(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
