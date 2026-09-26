"""Run isolated native task servers sequentially in one owned lane process.

Only module imports and the immutable tokenizer survive a task boundary.
Each task still executes the ordinary server lifecycle with its own output,
controller, generator, API, budgets, journals and finalization.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from . import event_native_server


def run_plan(path: Path, *, dynamic: bool = False) -> None:
    commands = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(commands, list) or not commands:
        raise ValueError("Persistent lane plan must be a nonempty list")
    frozen = None
    outputs = set()
    parsed = []
    for command in commands:
        if (not isinstance(command, list) or len(command) < 4
                or command[1:3] != ["-m", "benchmarks.memory_runtime.event_native_server"]
                or any(not isinstance(part, str) for part in command)):
            raise ValueError("Persistent lane plan contains an invalid server command")
        args = event_native_server.parser().parse_args(["--serve-child", *command[3:]])
        if len(args.task_ids.split(",")) != 1:
            raise ValueError("Persistent lane requires one task per server lifecycle")
        identity = (args.checkpoint.resolve(), args.benchmark, args.port,
                    args.sglang_backend_url, args.view_mode, args.tool_memory)
        if frozen is None:
            frozen = identity
        elif identity != frozen:
            raise ValueError("Persistent lane cannot change model, route, tool mode, or port")
        output = args.out.resolve()
        if output in outputs or output.exists():
            raise FileExistsError(f"Persistent task output already exists: {output}")
        outputs.add(output)
        parsed.append(args)

    def serve_task(args):
        args._persistent_lane = True
        start_file = args.out.parent / "persistent_task_start.requested"
        while not start_file.exists():
            time.sleep(0.05)
        event_native_server._serve(args)
        final = json.loads((args.out / "final.json").read_text(encoding="utf-8"))
        if final.get("status") != "stopped" or final.get("stop_reason") == "signal":
            raise RuntimeError(f"Persistent task stopped unexpectedly: {args.out}")
        # This marker is written only after final.json has been closed. The
        # parent cannot interpret an incomplete final write as completion.
        event_native_server.save_json(args.out / "persistent_task_complete.json", {
            "schema": "c1-persistent-task-complete-v1",
            "task_id": args.task_ids,
            "stop_reason": final.get("stop_reason"),
            "worker_pid": os.getpid(),
            "dynamic_dispatch": dynamic,
        })
        stop_file = args.out / "persistent_task_stop.requested"
        while not stop_file.exists():
            time.sleep(0.05)

    if not dynamic:
        for args in parsed:
            serve_task(args)
        return

    completed = set()
    lane_stop = path.parent / "persistent_lane_stop.requested"
    while True:
        ready = [args for args in parsed
                 if args.task_ids not in completed
                 and (args.out.parent / "persistent_task_start.requested").exists()]
        if len(ready) > 1:
            raise RuntimeError("Persistent lane received overlapping task starts")
        if ready:
            args = ready[0]
            serve_task(args)
            completed.add(args.task_ids)
        elif lane_stop.exists():
            return
        else:
            time.sleep(0.05)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args(argv)
    run_plan(args.plan, dynamic=args.dynamic)


if __name__ == "__main__":
    main()
