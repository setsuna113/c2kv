"""Preview or serve the selected D3 algorithm from the active source tree."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"


def load_config():
    return json.loads((HERE / "configs/current_algorithm.json").read_text(encoding="utf-8"))


def server_command(args):
    # Share the benchmark runner's argument construction and budget contract.
    import importlib.util

    spec = importlib.util.spec_from_file_location("history_runner", HERE / "runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    config = load_config()
    if args.device is not None:
        config["runtime"]["device"] = args.device
        config["runtime"]["npu_allocator_metrics"] = args.device.startswith("npu")
    return runner.server_command(
        config, task_id=args.task_id, checkpoint=str(args.checkpoint.resolve()),
        output=str(args.out.resolve()), port=args.port, python=sys.executable,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "serve"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-id", required=True, help="One BFCL task/session ID.")
    parser.add_argument("--port", type=int, default=28800)
    parser.add_argument("--device", help="Defaults to the selected NPU runtime.")
    args = parser.parse_args(argv)
    command = server_command(args)
    if args.action == "preview":
        print(json.dumps({"algorithm": load_config()["name"], "command": command,
                          "model_calls": 0}, indent=2))
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(RUNTIME / "python"), str(RUNTIME)))
    return subprocess.call(command, cwd=RUNTIME, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
