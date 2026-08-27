"""Unified runner: one benchmark x one arm -> unified rows + summary.

Thin dispatcher around the per-benchmark adapters; owns the proxy lifecycle
(spawn benchmarks/proxy.py in the requested arm, tear it down after).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def start_proxy(upstream: str, arm: str, port: int, log_dir: Path):
    log_path = log_dir / f"proxy_{arm}_{port}.jsonl"
    proc = subprocess.Popen(
        [
            sys.executable, str(HERE / "proxy.py"),
            "--upstream", upstream, "--arm", arm,
            "--port", str(port), "--request-log", str(log_path),
        ],
        stdout=log_dir / f"proxy_{arm}_{port}.out",
        stderr=subprocess.STDOUT,
    )
    import urllib.request

    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return proc, log_path
        except OSError:
            time.sleep(0.2)
    proc.terminate()
    raise SystemExit(f"proxy did not come up on port {port}")


def run_benchmark(name: str, base_url: str, user_base_url: str, out_dir: Path, **kwargs) -> Dict[str, Any]:
    if name == "tau2":
        from adapters import tau2_adapter

        return tau2_adapter.run(Path(kwargs.get("benchmark_dir", "")), base_url, user_base_url, out_dir, task_set=kwargs.get("task_set", "airline"), num_workers=kwargs.get("num_workers", 4), max_tasks=kwargs.get("max_tasks"))
    if name == "bfcl":
        from adapters import bfcl_adapter

        return bfcl_adapter.run(base_url, kwargs.get("categories", ["multi_turn"]).split(",") if isinstance(kwargs.get("categories"), str) else kwargs.get("categories", ["multi_turn"]), out_dir)
    if name == "toolsandbox":
        from adapters import toolsandbox_adapter

        return toolsandbox_adapter.run(base_url, user_base_url, out_dir, max_scenarios=kwargs.get("max_scenarios", 0))
    raise SystemExit(f"unknown benchmark {name!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=["tau2", "bfcl", "toolsandbox"])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:34000",
                        help="SGLang base URL (shared by all arms)")
    parser.add_argument("--user-upstream", default="",
                        help="base URL for user-simulator/judge traffic (defaults to --upstream; only the agent arm proxy compresses)")
    parser.add_argument("--proxy-port", type=int, default=34100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--categories", default="multi_turn")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-scenarios", type=int, default=0)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    log_dir = args.out / "logs"
    log_dir.mkdir(exist_ok=True)
    proxy_proc, request_log = start_proxy(args.upstream, args.arm, args.proxy_port, log_dir)
    try:
        base_url = f"http://127.0.0.1:{args.proxy_port}"
        user_base_url = args.user_upstream or args.upstream
        summary = run_benchmark(
            args.benchmark, base_url, user_base_url, args.out,
            task_set=args.task_set, categories=args.categories,
            num_workers=args.num_workers, max_tasks=args.max_tasks,
            max_scenarios=args.max_scenarios,
        )
    finally:
        proxy_proc.terminate()
    summary["arm"] = args.arm
    summary["benchmark"] = args.benchmark
    summary["request_log"] = str(request_log)
    (args.out / f"summary_{args.arm}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
