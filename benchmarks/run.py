"""Unified runner: one benchmark x one arm -> unified rows + summary.

Thin dispatcher around the per-benchmark adapters; owns the proxy lifecycle
(spawn benchmarks/proxy.py in the requested arm, tear it down after).
The compression ratio comes from the arm registry (arms.py), NOT from a
--ratio flag.
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
    out_handle = open(log_dir / f"proxy_{arm}_{port}.out", "w")
    proc = subprocess.Popen(
        [
            sys.executable, str(HERE / "proxy.py"),
            "--upstream", upstream, "--arm", arm,
            "--port", str(port), "--request-log", str(log_path),
        ],
        stdout=out_handle,
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


def _task_ids(value: str) -> str:
    """`--task-ids` as a comma-separated string; `@path` reads one id per line."""
    if not value:
        return ""
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.exists():
            raise SystemExit(f"FATAL: --task-ids file not found: {path}")
        ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.startswith("#")]
        if not ids:
            raise SystemExit(f"FATAL: --task-ids file {path} is empty")
        return ",".join(ids)
    return value


def run_benchmark(name: str, base_url: str, user_base_url: str, out_dir: Path, **kwargs) -> Dict[str, Any]:
    task_ids = kwargs.get("task_ids") or ""
    if task_ids and name in ("tau2", "toolsandbox"):
        # Only BFCL exposes an id filter (--run-ids) today.  Accepting the flag
        # and quietly running the whole set would report full-set numbers as if
        # they were the oracle subset's.
        raise SystemExit(
            f"FATAL: --task-ids is not plumbed for {name!r} yet (only bfcl has "
            "--run-ids). Wire the id filter into its adapter before using the "
            "oracle-subset re-run on this benchmark."
        )
    if name == "tau2":
        from adapters import tau2_adapter

        return tau2_adapter.run(
            Path(kwargs.get("benchmark_dir", "")), base_url, user_base_url,
            out_dir, task_set=kwargs.get("task_set", "airline"),
            num_workers=kwargs.get("num_workers", 4),
            max_tasks=kwargs.get("max_tasks"),
            run_name=kwargs.get("run_name", "c2kv_run"),
        )
    if name == "bfcl":
        from adapters import bfcl_adapter

        return bfcl_adapter.run(
            base_url,
            categories=kwargs.get("categories", "multi_turn_base"),
            run_ids=kwargs.get("task_ids", "") or "",
        )
    if name == "toolsandbox":
        from adapters import toolsandbox_adapter

        return toolsandbox_adapter.run(
            base_url, out_dir, test_mode=not kwargs.get("full", False)
        )
    raise SystemExit(f"unknown benchmark {name!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=["tau2", "bfcl", "toolsandbox"])
    parser.add_argument("--arm", required=True)
    parser.add_argument("--upstream", default="http://127.0.0.1:34000",
                        help="SGLang/hf_server base URL (shared by all arms)")
    parser.add_argument("--user-upstream", default="",
                        help="base URL for user-simulator/judge traffic (defaults to --upstream; only the agent arm proxy compresses)")
    parser.add_argument("--proxy-port", type=int, default=34100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--run-name", default="c2kv_run")
    parser.add_argument("--task-ids", default="",
                        help="comma-separated task ids, or @file with one per "
                             "line. Used to re-run only the oracle subset "
                             "(tasks the full arm passed and this arm failed).")
    parser.add_argument("--full", action="store_true",
                        help="toolsandbox: full suite instead of test mode")
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
            run_name=args.run_name, full=args.full,
            task_ids=_task_ids(args.task_ids),
        )
    finally:
        proxy_proc.terminate()
    summary["arm"] = args.arm
    summary["benchmark"] = args.benchmark
    summary["request_log"] = str(request_log)
    # Per-task rows keyed by task_id: this is what oracle_subset.py joins
    # across arms.  Without them an arm's summary is a single number and the
    # task-level oracle cannot be computed at all.
    rows = summary.pop("rows", None)
    if rows:
        rows_path = args.out / f"rows_{args.arm}.jsonl"
        with rows_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["rows_file"] = str(rows_path)
        summary["n_rows"] = len(rows)
    (args.out / f"summary_{args.arm}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
