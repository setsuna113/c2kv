"""Launch an isolated NPU server and run the proxy integration gates.

Run after CPU/tensor contracts pass. Existing services are never restarted.
Only the process groups created by this command are stopped; outputs persist.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib import request

HERE = Path(__file__).resolve().parent


def stop_owned_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def validate_mode(args, mode):
    mode_dir = args.out / mode
    mode_dir.mkdir()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, SGLANG_DIR=str(args.sglang_dir), MODEL_PATH=str(args.model),
               PYTHON_BIN=str(args.server_python), DEVICE=str(args.device),
               PORT=str(port), QUERY_PROJECTION=mode)
    opener = request.build_opener(request.ProxyHandler({}))
    upstream = f"http://127.0.0.1:{port}"
    print(f"Starting {mode} server on device {args.device}, port {port}", flush=True)
    result = {"mode": mode, "device": args.device, "port": port, "passed": False}
    with (mode_dir / "server.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(["bash", str(HERE / "launch_sgl1088.sh")],
                                env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            deadline = time.monotonic() + args.startup_timeout
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited with {proc.returncode}")
                try:
                    with opener.open(upstream + "/health", timeout=5):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError("server startup timeout")
                    time.sleep(1)
            print(f"Running {mode} proxy gates", flush=True)
            command = [sys.executable, str(HERE / "server_smoke.py"),
                       "--upstream", upstream, "--out", str(mode_dir / "smoke"),
                       "--query-projection", mode]
            if args.arms:
                command += ["--arms", *args.arms]
            tests = subprocess.Popen(command, start_new_session=True)
            try:
                result["returncode"] = tests.wait(timeout=args.test_timeout)
                result["passed"] = result["returncode"] == 0
            finally:
                stop_owned_group(tests)
            if result["passed"]:
                semantic_command = [sys.executable, str(args.sglang_dir / "scripts" /
                                    "c2kv" / "smoke_c2kv_semantics.py"),
                                    "--base-url", upstream]
                with (mode_dir / "semantics.log").open("w", encoding="utf-8") as semantic_log:
                    semantic_tests = subprocess.Popen(
                        semantic_command, stdout=semantic_log, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    try:
                        result["semantics_returncode"] = semantic_tests.wait(timeout=args.test_timeout)
                        result["passed"] = result["semantics_returncode"] == 0
                    finally:
                        stop_owned_group(semantic_tests)
        except Exception as error:
            result["error"] = f"{type(error).__name__}: {error}"
        finally:
            stop_owned_group(proc)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--server-python", type=Path,
                        default=Path("/home/liuyancheng/envs/sgl/bin/python"))
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("base", "gist"), default=["base", "gist"])
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--test-timeout", type=int, default=1800)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("this launcher requires the Linux NPU host")
    args.out = args.out.resolve()
    args.sglang_dir = args.sglang_dir.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    results = []
    for mode in args.modes:
        results.append(validate_mode(args, mode))
        (args.out / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        if not results[-1]["passed"]:
            break
    raise SystemExit(0 if results and all(r["passed"] for r in results) else 1)


if __name__ == "__main__":
    main()
