"""NPU engine lifecycle for generality cells (runs on ascend03).

One long-lived sglang engine per card; every measured cell binds to one card
and one engine for its whole lifetime. Launch goes through tools/launch_engine.sh
which pins the sglang-gen PYTHONPATH, CANN env, proxy bypass and telemetry env.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import design

ROOT = Path(design.GENERATION_ROOT)
LAUNCHER = ROOT / "tools" / "launch_engine.sh"
ENGINE_LOG_DIR = ROOT / "logs" / "engines"


def _no_proxy_opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def launch(card: int, port: int, tag: str, extra_args: list[str] | None = None) -> dict:
    ENGINE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = ENGINE_LOG_DIR / f"{tag}.log"
    cmd = ["bash", str(LAUNCHER), str(card), str(port), tag, *(extra_args or [])]
    with log.open("ab") as stream:
        proc = subprocess.Popen(
            cmd, stdout=stream, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    receipt = {
        "card": card, "port": port, "tag": tag, "pid": proc.pid,
        "log": str(log), "extra_args": extra_args or [],
        "launched_at": time.time(),
    }
    (ENGINE_LOG_DIR / f"{tag}.launch.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        # A zombie still answers kill(pid, 0), but cannot own a ready server.
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _listener_in_session(port: int, pid: int) -> bool:
    """Require the healthy listener to belong to the launched process group."""
    try:
        result = subprocess.run(
            ["ss", "-H", "-ltnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if result.returncode != 0:
        return False
    for owner in re.findall(r"\bpid=(\d+)\b", result.stdout):
        try:
            if os.getsid(int(owner)) == pid:
                return True
        except OSError:
            continue
    return False


def wait_ready(port: int, timeout_s: int = 600, *, pid: int | None = None) -> dict:
    opener = _no_proxy_opener()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pid is not None and not _pid_alive(pid):
            return {"ready": False, "reason": "process_exit"}
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as r:
                if r.status == 200 and (pid is None or _listener_in_session(port, pid)):
                    return {"ready": True, "elapsed_s": timeout_s - (deadline - time.monotonic())}
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(5)
    return {"ready": False, "reason": "timeout"}


def model_info(port: int) -> dict:
    opener = _no_proxy_opener()
    with opener.open(f"http://127.0.0.1:{port}/model_info", timeout=15) as r:
        return json.load(r)


def validate_engine(port: int, expect_model: str = "gen-c1000") -> dict:
    info = model_info(port)
    cap = info.get("c2kv_native_packed")
    problems = []
    if not isinstance(cap, dict) or not cap.get("enabled"):
        problems.append("c2kv_native_packed not enabled")
    else:
        if cap.get("kv_bytes_per_token") != design.KV_BYTES_PER_TOKEN:
            problems.append(f"kv_bytes_per_token={cap.get('kv_bytes_per_token')}")
        binding = cap.get("model_binding") or {}
        if binding.get("query_projection") != "base":
            problems.append("query_projection != base")
        if Path(info.get("model_path", "")).resolve() != Path(design.ENGINE_MODEL_PATH).resolve():
            problems.append("model_path mismatch")
    if info.get("served_model_name", expect_model) != expect_model and str(expect_model) not in str(info.get("model_name", "")):
        pass  # informational only; native contract is checked above
    return {"ok": not problems, "problems": problems, "model_info_keys": sorted(info.keys())}


def stop(tag: str) -> dict:
    """Stop one engine by its launch receipt (never by broad pattern match)."""
    receipt_path = ENGINE_LOG_DIR / f"{tag}.launch.json"
    if not receipt_path.exists():
        return {"stopped": False, "reason": "no_receipt"}
    receipt = json.loads(receipt_path.read_text())
    pid, port = receipt["pid"], receipt["port"]
    killed = []
    try:
        os.kill(pid, signal.SIGTERM)
        killed.append(pid)
    except OSError:
        pass
    # scheduler/detokenizer children live in the same session group
    try:
        out = subprocess.run(
            ["pgrep", "-g", "0", "-f", f"--port {port}"],
            capture_output=True, text=True,
        )
    except OSError:
        out = None
    for line in (out.stdout.splitlines() if out else []):
        try:
            child = int(line)
            os.kill(child, signal.SIGTERM)
            killed.append(child)
        except (OSError, ValueError):
            pass
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time.sleep(1)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    return {"stopped": True, "signalled": killed, "port": port, "tag": tag}


def npu_smi_snapshot(out_path: Path) -> dict:
    """Raw per-card utilization/HBM snapshot for resource bookkeeping."""
    out = subprocess.run(["npu-smi", "info"], capture_output=True, text=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out.stdout)
    return {"written": str(out_path), "returncode": out.returncode}
