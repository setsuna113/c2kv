"""Start each frozen T02 slot only after its physical device is released."""
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

from t02_parallel import (dynamic_worker_no_work, initialize_ledger,
                          isolated_branch_failure, read, save)


def verify(root):
    manifest = read(root / "source_files.json")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source differs: {name}")
    return len(manifest["files"])


def require_launch_authorization(root, contract):
    if contract.get("launch_authorized") is True:
        return
    marker = root / "launch_authorization.json"
    if marker.exists():
        receipt = read(marker)
        digest = hashlib.sha256((root / "contract.json").read_bytes()).hexdigest()
        if receipt.get("approved") is True and receipt.get("contract_sha256") == digest:
            return
    raise PermissionError("This prepared contract has not been authorized for launch")


def owned_stop(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def launch(args, root, log_name):
    path = root / log_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as log:
        return subprocess.Popen(args, cwd=root, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def available(root, worker):
    device, port = worker["physical_device"], worker["port"]
    if device not in (0, 1, 2, 3, 4, 6):
        raise ValueError("Physical device is outside the frozen released-device allocation")
    dependency = worker.get("after_eval_lane")
    if dependency:
        marker = Path(dependency) / "run/status.json"
        if not marker.exists() or read(marker).get("state") not in ("completed", "failed_no_rerun"):
            return None
    raw = subprocess.run(["npu-smi", "info"], check=True, capture_output=True, text=True).stdout
    devices = {int(x) for x in re.findall(r"^\|\s+(\d+)\s+910\w+\s*\|", raw, re.M)}
    if devices != set(range(8)) or "Process id" not in raw:
        raise RuntimeError("Unrecognized npu-smi layout")
    if not re.search(rf"^\|\s+{device}\s+910\w+\s*\|\s+OK\s", raw, re.M):
        return None
    for line in raw.splitlines():
        match = re.match(r"^\|\s+(\d+)\s+(\d+)\s*\|\s+(\d+)\s*\|", line)
        if match and int(match.group(1)) == device:
            return None
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return None
    return {"checked_at_epoch": time.time(), "physical_device": device,
            "port": port, "npu_smi_sha256": hashlib.sha256(raw.encode()).hexdigest()}


def slot(root, worker_id):
    contract = read(root / "contract.json")
    require_launch_authorization(root, contract)
    worker = next(row for row in contract["workers"] if row["worker_id"] == worker_id)
    directory = root / "workers" / worker_id
    status_path = directory / "slot.json"
    if status_path.exists():
        raise FileExistsError("Slot already has a start receipt; no automatic restart")
    engine = collector = None
    status = {"phase": "waiting_for_released_device", "worker_id": worker_id,
              "supervisor_pid": os.getpid(), "physical_device": worker["physical_device"]}
    save(status_path, status)
    try:
        while True:
            if (root / "abort.json").exists():
                raise RuntimeError("Global collection aborted before slot start")
            if dynamic_worker_no_work(root, worker_id):
                status.update(
                    phase="completed_no_work",
                    reason="dynamic_source_queue_completed_without_lease",
                )
                return 0
            preflight = available(root, worker)
            if preflight is not None:
                break
            time.sleep(20)
        status.update(phase="starting_engine", resource_preflight=preflight)
        engine = launch(["bash", str(root / "scripts/engine.sh"),
            "--resource-coordination-approved", "--physical-device", str(worker["physical_device"]),
            "--port", str(worker["port"])], root, f"workers/{worker_id}/engine.log")
        status["engine_pid"] = engine.pid
        save(status_path, status)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if (root / "abort.json").exists() or engine.poll() is not None:
                raise RuntimeError("Engine exited or global run aborted during startup")
            try:
                with opener.open(f"http://127.0.0.1:{worker['port']}/health", timeout=3) as response:
                    if response.status == 200:
                        break
            except OSError:
                pass
            time.sleep(5)
        else:
            raise TimeoutError("Engine did not become ready within 900 seconds")
        collector = launch(["bash", str(root / "scripts/parallel_worker.sh"), worker_id,
                            str(worker["physical_device"]), str(worker["port"])], root,
                            f"workers/{worker_id}/worker.log")
        status.update(phase="worker_running", worker_pid=collector.pid)
        save(status_path, status)
        while collector.poll() is None:
            if (root / "abort.json").exists():
                raise RuntimeError("Global run aborted")
            time.sleep(5)
        if collector.returncode:
            raise RuntimeError(f"Worker exited with code {collector.returncode}")
        status.update(phase="completed", worker_exit_code=0)
        return 0
    except Exception as error:
        local_failure = isolated_branch_failure(root, worker_id)
        status.update(phase="failed_no_retry", error_type=type(error).__name__,
                      error=str(error),
                      failure_scope=("branch_execution" if local_failure else "global"),
                      global_abort_required=local_failure is None,
                      worker_failure=local_failure)
        if local_failure is None and not (root / "abort.json").exists():
            save(root / "abort.json", {"worker_id": worker_id, "phase": "slot",
                                       "error": str(error), "updated_at_epoch": time.time()})
        return 1
    finally:
        owned_stop(collector)
        owned_stop(engine)
        status["finished_at_epoch"] = time.time()
        save(status_path, status)


def supervise(root):
    verified_files = verify(root)
    if (root / "started.json").exists():
        raise FileExistsError("Parallel run already started; automatic restart is disabled")
    contract = read(root / "contract.json")
    require_launch_authorization(root, contract)
    initialize_ledger(root)
    processes = []
    coordinator = None
    status = {"schema": "t02-parallel-launch-v1", "phase": "starting_workers",
              "verified_files": verified_files, "supervisor_pid": os.getpid(),
              "started_at_epoch": time.time(), "slots": []}
    save(root / "started.json", status)
    save(root / "status.json", {"phase": "source_collection", "updated_at_epoch": time.time()})
    try:
        coordinator = launch([sys.executable, str(root / "history_system/t02_parallel.py"),
                              "coordinate", "--root", str(root)], root, "coordinator.log")
        status["coordinator_pid"] = coordinator.pid
        for worker in contract["workers"]:
            process = launch([sys.executable, str(root / "history_system/t02_parallel_launch.py"),
                "--root", str(root), "--slot", worker["worker_id"]], root,
                f"workers/{worker['worker_id']}/supervisor.log")
            processes.append(process)
            status["slots"].append({"worker_id": worker["worker_id"], "pid": process.pid,
                                     "physical_device": worker["physical_device"]})
            save(root / "started.json", status)
        code = coordinator.wait()
        for process in processes:
            process.wait()
        if code:
            coordinator_status = read(root / "status.json")
            if coordinator_status.get("phase") != "partial_failed":
                save(root / "status.json", {"phase": "failed_no_retry",
                    "coordinator_exit_code": code, "abort_receipt": "abort.json",
                    "updated_at_epoch": time.time()})
            return code
        training_worker = next(row for row in contract["workers"] if row["physical_device"] == 6)
        while available(root, training_worker) is None:
            time.sleep(20)
        trainer = launch(["bash", str(root / "scripts/train.sh"),
                          "--resource-coordination-approved", "--physical-device", "6"],
                          root, "train.log")
        save(root / "status.json", {"phase": "training", "trainer_pid": trainer.pid,
                                    "updated_at_epoch": time.time()})
        code = trainer.wait()
        save(root / "status.json", {"phase": "completed" if code == 0 else "training_partial_or_failed",
                                    "trainer_exit_code": code, "updated_at_epoch": time.time()})
        return code
    finally:
        owned_stop(coordinator)
        for process in processes:
            owned_stop(process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--slot")
    args = parser.parse_args()
    raise SystemExit(slot(args.root, args.slot) if args.slot else supervise(args.root))
