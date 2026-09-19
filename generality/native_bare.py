"""Run the shared native paper client against an existing NPU engine.

This entry point does not change the generality experiment's S0 ablations,
start an engine, or mutate a scheduler queue.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def command(args):
    root = args.paper_root.resolve()
    if not (root / "experiments/history_system/native_bare.py").is_file():
        raise RuntimeError("Install the shared native-bare paper source before launching")
    result = [args.python, "-m", "benchmarks.paper.c1", "--config", str(args.config.resolve()),
              "--arm", getattr(args, "arm", "c2kv_native_r4"), "--benchmark", args.benchmark,
              "--upstream", args.upstream, "--proxy-port", str(args.proxy_port),
              "--out", str(args.out.resolve()), "--stage", args.stage]
    if args.task_ids:
        result += ["--task-ids", args.task_ids]
    if args.prefixes:
        result += ["--prefixes", str(args.prefixes.resolve())]
    if getattr(args, "tool_memory", None):
        result += ["--tool-memory", args.tool_memory]
        if getattr(args, "tool_checkpoint", None) is not None:
            result += ["--tool-checkpoint", str(args.tool_checkpoint.resolve())]
        if getattr(args, "tool_budget_tokens", None) is not None:
            result += ["--tool-budget-tokens", str(args.tool_budget_tokens)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--arm", default="c2kv_native_r4")
    parser.add_argument("--benchmark", choices=("bfcl_base", "bfcl_long_context", "appworld", "acebench_agent", "toolsandbox"), required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-ids")
    parser.add_argument("--stage", choices=("closed_loop", "common_prefix"), default="closed_loop")
    parser.add_argument("--prefixes", type=Path)
    parser.add_argument("--tool-memory")
    parser.add_argument("--tool-checkpoint", type=Path)
    parser.add_argument("--tool-budget-tokens", type=int)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the shared-paper command without contacting the engine")
    args = parser.parse_args()
    if args.stage == "common_prefix" and args.prefixes is None:
        parser.error("common_prefix requires --prefixes")
    if not args.tool_memory and (args.tool_checkpoint is not None or args.tool_budget_tokens is not None):
        parser.error("Tool options require --tool-memory")
    if args.tool_memory and args.tool_memory.startswith("t0:") and args.tool_checkpoint is None:
        parser.error("T0 tool memory requires --tool-checkpoint")
    if args.tool_budget_tokens is not None and args.tool_budget_tokens <= 0:
        parser.error("--tool-budget-tokens must be positive")
    argv = command(args)
    if args.dry_run:
        print(json.dumps({"command": argv, "engine_preflight": "skipped (dry-run)"}, indent=2))
        return
    env = dict(os.environ)
    # The shared paper package must precede any old controller_runtime package.
    env["PYTHONPATH"] = str(args.paper_root.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(argv, cwd=args.paper_root, env=env, check=True)


if __name__ == "__main__":
    main()
