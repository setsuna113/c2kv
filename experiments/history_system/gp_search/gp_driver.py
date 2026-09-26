"""Sequential per-card driver: run frozen GP candidates on mixed20 (server-side).

One process per card; each candidate runs runner.py over the full 20-task
manifest against the card-local engine. Progress is appended atomically to
<run_root>/progress/card<N>.json. A stop file (<run_root>/progress/stop_card<N>)
makes the driver exit after the current candidate.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

# The login shell exports http_proxy; 127.0.0.1 is not in no_proxy, so any
# localhost HTTP from this driver or its children would be hijacked by Squid.
for _key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
             "ALL_PROXY", "all_proxy"):
    os.environ.pop(_key, None)
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"
# The CPU controller must not auto-load torch_npu (no Ascend LD_LIBRARY_PATH here).
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def choose_port_base(span: int = 20) -> int:
    """Pick a run of consecutive free loopback ports.

    Other users on this server grab 36xxx ports dynamically, so any fixed
    port_base collides eventually (observed on 36319 and 367xx).
    """
    import random
    import socket
    candidates = list(range(36200, 60000 - span, 7))
    random.shuffle(candidates)
    for base in candidates:
        held = []
        for offset in range(span):
            probe = socket.socket()
            try:
                probe.bind(("127.0.0.1", base + offset))
                held.append(probe)
            except OSError:
                break
        for probe in held:
            probe.close()
        if len(held) == span:
            return base
    raise SystemExit("no free 20-port range found")


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def engine_ready(port: int, timeout: float = 10.0) -> bool:
    try:
        with _OPENER.open(f"http://127.0.0.1:{port}/model_info", timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def run_status(manifest_path: Path) -> str | None:
    """One of: completed_fixed_manifest, stopped_* (terminal), running, None."""
    if not manifest_path.exists():
        return None
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return "unreadable"
    status = record.get("status")
    state = record.get("state")
    if status:
        return status
    return state if state != "running" else "running"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--card", type=int, required=True)
    parser.add_argument("--slot", default="",
                        help="engine-slot suffix, e.g. s1 for the second engine")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    run_root = Path(plan["run_root"])
    engine_port = int(plan["engine_port"])
    port_base = int(plan["port_base"]) if plan.get("port_base") else choose_port_base()
    plan["port_base"] = port_base
    progress_path = run_root / "progress" / f"card{args.card}{args.slot}.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    stop_path = run_root / "progress" / f"stop_card{args.card}{args.slot}"
    progress = {
        "schema": "a-history-gp-driver-progress-v1",
        "card": args.card,
        "slot": args.slot,
        "engine_port": engine_port,
        "port_base": port_base,
        "driver_state": "running",
        "entries": [],
    }
    save(progress_path, progress)
    for entry in plan["candidates"]:
        if stop_path.exists():
            progress["driver_state"] = "stopped_by_file"
            save(progress_path, progress)
            return 0
        candidate = entry["candidate_id"]
        directory = Path(entry["directory"])
        run_name = directory.name
        output = run_root / "runs" / run_name
        record = {
            "candidate_id": candidate,
            "run_name": run_name,
            "state": "pending",
            "started_at_epoch": time.time(),
        }
        if not engine_ready(engine_port):
            waited = 0.0
            while waited < 1800 and not engine_ready(engine_port):
                time.sleep(30)
                waited += 30
            if not engine_ready(engine_port):
                record.update({"state": "engine_down", "finished_at_epoch": time.time()})
                progress["entries"].append(record)
                progress["driver_state"] = "engine_down"
                save(progress_path, progress)
                return 3
        manifest = output / "stage_manifest.json"
        status = run_status(manifest)
        if status is not None and (
                status == "completed_fixed_manifest" or not plan.get("retry_failed")):
            record.update({
                "state": f"already_{status}",
                "skipped": True,
                "finished_at_epoch": time.time(),
            })
            progress["entries"].append(record)
            save(progress_path, progress)
            continue
        if output.exists():
            stamp = time.strftime("%Y%m%dT%H%M%S")
            output.rename(run_root / "runs" / f"{run_name}.failed_{stamp}")
        started = time.monotonic()
        command = [
            plan["python"], str(run_root / "history_system" / "runner.py"),
            "run",
            "--design", str(directory / "design.json"),
            "--checkpoint", plan["checkpoint"],
            "--output", str(output),
            "--benchmark-dir", plan["benchmark_dir"],
            "--python", plan["python"],
            "--bfcl-python", plan["bfcl_python"],
            "--port-base", str(plan["port_base"]),
            "--runtime-root", str(directory / "runtime"),
        ]
        log_path = run_root / "runs" / f"{run_name}.driver.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        record.update({
            "state": "completed" if code == 0 else f"runner_exit_{code}",
            "returncode": code,
            "wall_seconds": time.monotonic() - started,
            "finished_at_epoch": time.time(),
            "driver_log": str(log_path),
        })
        progress["entries"].append(record)
        save(progress_path, progress)
    progress["driver_state"] = "done"
    save(progress_path, progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
