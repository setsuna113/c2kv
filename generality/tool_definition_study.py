"""Run the paper's recorded tool-definition study on NPU from shared source.

Preparation, selection, budgets and scoring live in the paper checkout. This
launcher supplies only the import path; evaluation targets an existing SGLang
endpoint on NPU. It starts no model or serving engine.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def command(paper_root, python, action, forwarded):
    root = Path(paper_root).resolve()
    if not (root / "benchmarks/tool_definition/cli.py").is_file():
        raise ValueError("Install the paper tool-definition evaluator in --paper-root")
    if action not in {"prepare", "evaluate"}:
        raise ValueError("Expected prepare or evaluate")
    result = [str(python), "-m", "benchmarks.tool_definition.cli", action, *forwarded]
    if action == "evaluate":
        if any(item == "--device" or item.startswith("--device=") for item in forwarded):
            raise ValueError("SGLang owns the device; pass --upstream for the running NPU server")
        if not any(item == "--upstream" or item.startswith("--upstream=") for item in forwarded):
            raise ValueError("Evaluation requires --upstream for the running NPU SGLang server")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "evaluate"))
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    cmd = command(args.paper_root, args.python, args.action, forwarded)
    if args.dry_run:
        print(json.dumps({"command": cmd, "execution": "not started"}, indent=2))
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.paper_root.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=args.paper_root, env=env, check=True)


if __name__ == "__main__":
    main()
