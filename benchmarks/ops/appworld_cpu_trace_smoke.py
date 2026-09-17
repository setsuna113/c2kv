"""Run a three-action AppWorld CPU smoke and invoke the official evaluator.

This diagnostic uses harmless ``print`` actions.  It validates environment
execution, final-budget handling, state persistence, telemetry, and scorer
plumbing; its semantic score is expected to be zero.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


FINAL_MARKER = "C2KV_FINAL_BUDGETED_ACTION_EXECUTED"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness-root", type=Path, required=True,
                        help="private AppWorld root containing data/")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--split", default="test_normal")
    parser.add_argument("--experiment", default="c2kv_appworld_cpu_trace_smoke")
    parser.add_argument("--max-interactions", type=int, default=3)
    args = parser.parse_args(argv)
    if args.max_interactions < 1:
        raise SystemExit("--max-interactions must be positive")

    root = args.harness_root.resolve()
    # sitecustomize imports ACON before this script starts, and AppWorld binds
    # its root during that import.  The process must therefore start here.
    if Path.cwd().resolve() != root:
        raise SystemExit(f"launch the smoke with cwd={root}")
    from productive_agents.env.appworld.config import AppWorldEnvConfig
    from productive_agents.env.appworld.env import AppWorldEnv

    config = AppWorldEnvConfig(
        experiment_name=args.experiment,
        dataset_split=args.split,
        max_interactions=args.max_interactions,
        verbose=False,
    )
    env = AppWorldEnv(config)
    task_dir = (root / "outputs" / args.experiment / args.split
                / f"task_{args.task_id}")
    env.reset(seed=42, task_id=args.task_id)
    try:
        final = None
        for index in range(1, args.max_interactions + 1):
            marker = FINAL_MARKER if index == args.max_interactions else f"smoke_action_{index}"
            final = env.step(f"print({marker!r})")
        if len(env.trajectory) != args.max_interactions:
            raise RuntimeError(
                f"expected {args.max_interactions} executed actions, "
                f"found {len(env.trajectory)}")
        if FINAL_MARKER not in str(final[0]):
            raise RuntimeError("final budgeted action did not reach AppWorld.execute")
        task_dir.mkdir(parents=True, exist_ok=True)
        env.dump_history(str(task_dir))
        (task_dir / "results.json").write_text(json.dumps({
            "task_id": args.task_id,
            "success": False,
            "iterations": args.max_interactions,
            "done": final[2],
            "info": final[3],
            "termination_reason": final[3].get("reason"),
        }, indent=2) + "\n", encoding="utf-8")
    finally:
        env.close()

    scorer_env = dict(os.environ)
    scorer_env.pop("C2KV_APPWORLD_TELEMETRY_PATH", None)
    scorer_env.pop("C2KV_APPWORLD_RUN_DIR", None)
    appworld = shutil.which("appworld")
    if not appworld:
        raise SystemExit("appworld CLI is not on PATH")
    subprocess.run(
        [appworld, "evaluate", args.experiment, args.split],
        cwd=root, env=scorer_env, check=True)
    evaluation = (root / "experiments" / "outputs" / args.experiment
                  / "evaluations" / f"{args.split}.json")
    print(json.dumps({
        "task_id": args.task_id,
        "executed_actions": len(env.trajectory),
        "final_marker": FINAL_MARKER,
        "evaluation": str(evaluation),
        "task_dir": str(task_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
