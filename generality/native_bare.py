"""Run the shared native bare paper client against an existing NPU engine.

This entry point does not change the generality experiment's S0 ablations,
start an engine, or mutate a scheduler queue.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def command(args):
    root = args.paper_root.resolve()
    if not (root / "experiments/history_system/native_bare.py").is_file():
        raise RuntimeError("Install the shared native-bare paper source before launching")
    result = [args.python, "-m", "benchmarks.paper.c1", "--config", str(args.config.resolve()),
              "--arm", "c2kv_native_r4", "--benchmark", args.benchmark,
              "--upstream", args.upstream, "--proxy-port", str(args.proxy_port),
              "--out", str(args.out.resolve()), "--stage", args.stage]
    if args.task_ids:
        result += ["--task-ids", args.task_ids]
    if args.prefixes:
        result += ["--prefixes", str(args.prefixes.resolve())]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--benchmark", choices=("bfcl_base", "bfcl_long_context", "appworld", "acebench_agent", "toolsandbox"), required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--proxy-port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-ids")
    parser.add_argument("--stage", choices=("closed_loop", "common_prefix"), default="closed_loop")
    parser.add_argument("--prefixes", type=Path)
    args = parser.parse_args()
    if args.stage == "common_prefix" and args.prefixes is None:
        parser.error("common_prefix requires --prefixes")
    argv = command(args)
    env = dict(os.environ)
    # The shared paper package must precede any old controller_runtime package.
    env["PYTHONPATH"] = str(args.paper_root.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(argv, cwd=args.paper_root, env=env, check=True)


if __name__ == "__main__":
    main()
