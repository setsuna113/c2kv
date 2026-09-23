"""Bounded synthetic CUDA integration check; never runs the paper matrix."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.error
import urllib.request

from .runner import (DEFAULT_CONFIG, ROOT, history_kv_budget_args,
                     resolve_history_kv_budgets, server_command, wait_server)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--history-kv-budget-tokens", type=int, metavar="B",
                        help="shared absolute history-token cap for the configured KV arms")
    parser.add_argument("--sglang-source", type=Path, default=ROOT.parent.parent / "sglang-paper")
    parser.add_argument("--cpu-offload-gb", type=float, default=0,
                        help="optional laptop-only weight offload for integration checks")
    parser.add_argument("--arms", default="", help="comma-separated proxy arms; empty checks all configured proxy arms")
    parser.add_argument("--replay-prefixes", type=Path,
                        help="also validate a previously recorded Full prefix corpus")
    parser.add_argument("--validate-replay", action="store_true",
                        help="record a fresh Full source and replay it through every checked arm")
    args = parser.parse_args(argv)
    if "c2kv_c1_t02_r8" in args.arms.split(","):
        parser.error("Use benchmarks.paper.c1 --task-ids for the native C1 controller smoke")
    config = json.loads(args.config.read_text())
    if args.history_kv_budget_tokens is not None:
        config["history_kv_budget_tokens"] = args.history_kv_budget_tokens
    config = resolve_history_kv_budgets(config)
    config.update(max_total_tokens=8192, context_length=8192,
                  mem_fraction_static=0.85, c2kv_pool_fraction=0.01,
                  chunked_prefill_size=256, server_port=34300, proxy_port=34301)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(args.sglang_source.resolve() / "python"), str(ROOT.parent)])
    env["C2KV_PAPER_TELEMETRY"] = "1"
    env["SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION"] = "0"
    env["C2KV_PAPER_TELEMETRY_LOG"] = str(output / "server_telemetry.jsonl")
    env.setdefault("CUDA_HOME", "/opt/cuda")
    env["PATH"] = str(Path(config["server_python"]).parent) + os.pathsep + env["PATH"]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    import socket
    for port in (config["server_port"], config["proxy_port"]):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Smoke port {port} is already occupied")
    results = []
    replay_source = args.replay_prefixes
    if args.validate_replay and args.arms and "full" not in args.arms.split(",") and not replay_source:
        raise ValueError("Replay validation needs the full arm or --replay-prefixes")
    with (output / "server.log").open("w") as server_log:
        # Laptop-only validation: CPU weight offload keeps the 16 GB display GPU
        # out of WDDM paging. The formal runner never adds this option.
        smoke_command = server_command(config, args.sglang_source)
        if args.cpu_offload_gb:
            smoke_command += ["--cpu-offload-gb", str(args.cpu_offload_gb)]
        server = subprocess.Popen(smoke_command, env=env,
                                  stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_server(server, config["server_port"])
            print("CUDA server ready", flush=True)
            for method in config["methods"]:
                arm = method["arm"]
                if arm == "c2kv_c1_t02_r8":
                    continue
                if args.arms and arm not in args.arms.split(","):
                    continue
                directory = output / arm
                directory.mkdir()
                cmd = [config["bench_python"], str(ROOT / "proxy.py"), "--upstream", "http://127.0.0.1:34300",
                       "--port", "34301", "--arm", arm, "--backend", "sglang", "--doc-packing", "turn",
                       "--max-doc-length", "1000", "--max-doc-num", "1000", "--query-projection", "base",
                       "--request-log", str(directory / "requests.jsonl"),
                       "--telemetry-log", str(directory / "proxy_telemetry.jsonl")]
                cmd += history_kv_budget_args(method)
                if arm == "full":
                    cmd += ["--record-prefixes", str(directory / "full_prefixes.jsonl")]
                with (directory / "proxy.log").open("w") as proxy_log:
                    proxy = subprocess.Popen(cmd, env=env, stdout=proxy_log, stderr=subprocess.STDOUT)
                    try:
                        wait_server(proxy, config["proxy_port"])
                        messages = [{"role": "system", "content": "You are a helpful assistant. Answer briefly using the recorded facts."},
                                    {"role": "user", "content": "Remember the archive access code."}]
                        for n in range(3):
                            messages += [{"role": "assistant", "content": f"Subgoal: inspect archive {n}\nI will record the code."},
                                         {"role": "user", "content": ("Archive note: the access code is BLUE-17.\n" + "x" * 6000
                                            if method["method"] == "ACON" else "Archive note: the access code is BLUE-17.\n" * 8)}]
                        messages += [{"role": "assistant", "content": "Subgoal: answer the user's question"},
                                     {"role": "user", "content": "What is the access code? Reply with the code only."}]
                        for turn in range(2 if method["method"] in ("H2O", "SnapKV") else 1):
                            payload = {"model": config["model"], "messages": messages, "temperature": 0,
                                       "max_tokens": 64, "stream": False,
                                       "c2kv_measurement_session_id": "synthetic-" + arm}
                            start = time.monotonic()
                            request = urllib.request.Request("http://127.0.0.1:34301/v1/chat/completions",
                                                             data=json.dumps(payload).encode(),
                                                             headers={"Content-Type": "application/json"})
                            try:
                                with opener.open(request, timeout=600) as response:
                                    body = json.load(response)
                            except urllib.error.HTTPError as error:
                                raise RuntimeError(error.read().decode()) from error
                            (directory / f"response-{turn}.json").write_text(json.dumps(body, indent=2))
                            if not body.get("choices"):
                                raise RuntimeError(f"No completion: {body}")
                            measured = body.get("c2kv_proxy", {}).get("server_measurement", {})
                            if not measured.get("request_peak_resident_kv_bytes"):
                                raise RuntimeError("Missing measured resident KV peak")
                            if method["method"] in ("H2O", "SnapKV"):
                                if not measured.get("history_kv_physical_eviction_success"):
                                    raise RuntimeError("Physical history eviction did not succeed")
                                if measured.get("full_history_reprefill"):
                                    raise RuntimeError("Persistent history unexpectedly re-prefilled")
                            results.append({"arm": arm, "turn": turn, "elapsed_sec": time.monotonic() - start,
                                            "finish_reason": body["choices"][0].get("finish_reason")})
                            print(json.dumps(results[-1]), flush=True)
                            messages += [body["choices"][0]["message"], {"role": "user", "content": "Repeat the code once more."}]
                        if args.validate_replay and arm == "full" and replay_source is None:
                            import shutil
                            replay_source = output / "shared_full_prefixes.jsonl"
                            shutil.copyfile(directory / "full_prefixes.jsonl", replay_source)
                        if replay_source:
                            from benchmarks.measurement.replay import replay_prefixes
                            replay_summary = replay_prefixes(
                                replay_source, "http://127.0.0.1:34301",
                                directory / "prefix_replay.jsonl",
                                source_run_id="synthetic-full", target_run_id=arm)
                            if replay_summary["failed"]:
                                raise RuntimeError(f"Common-prefix replay failed: {replay_summary}")
                            print(json.dumps({"arm": arm, "replay": replay_summary}), flush=True)
                        # Close every session opened by this proxy before changing arms.
                        session_ids = set()
                        for path in directory.glob("response-*.json"):
                            response = json.loads(path.read_text())
                            lifecycle = response.get("c2kv_proxy", {}).get("server_measurement", {}).get("history_kv_lifecycle") or {}
                            if lifecycle.get("session_id"):
                                session_ids.add(lifecycle["session_id"])
                        replay_path = directory / "prefix_replay.jsonl"
                        if replay_path.exists():
                            for line in replay_path.read_text().splitlines():
                                response = json.loads(line).get("response") or {}
                                lifecycle = response.get("c2kv_proxy", {}).get("server_measurement", {}).get("history_kv_lifecycle") or {}
                                if lifecycle.get("session_id"):
                                    session_ids.add(lifecycle["session_id"])
                        for session_id in session_ids:
                            request = urllib.request.Request("http://127.0.0.1:34300/close_session",
                                data=json.dumps({"session_id": session_id}).encode(),
                                headers={"Content-Type": "application/json"})
                            with opener.open(request, timeout=15) as response:
                                response.read()
                        request = urllib.request.Request("http://127.0.0.1:34300/flush_cache?timeout=10", data=b"{}",
                                                         headers={"Content-Type": "application/json"})
                        with opener.open(request, timeout=15) as response:
                            if not response.read().decode().startswith("Cache flushed."):
                                raise RuntimeError("Smoke cache flush failed")
                    finally:
                        proxy.terminate()
                        try:
                            proxy.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            proxy.kill()
                            proxy.wait()
            (output / "result.json").write_text(json.dumps({"completed": True, "requests": results}, indent=2))
        finally:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait()


if __name__ == "__main__":
    main()
