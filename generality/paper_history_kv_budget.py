"""Forward one paper history-KV budget cell to the shared device-neutral client.

The upstream NPU server is owned by the caller; this entry point never starts it.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def command(paper_root: Path, python: str, forwarded: list[str]) -> list[str]:
    root = paper_root.resolve()
    required = ("benchmarks/paper/history_kv_client.py",
                "benchmarks/paper/runner.py", "benchmarks/history_budget.py")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError("--paper-root lacks shared history-KV client sources: " + ", ".join(missing))
    return [python, "-m", "benchmarks.paper.history_kv_client", *forwarded]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable,
                        help="shared client interpreter (default: this interpreter)")
    args, forwarded = parser.parse_known_args(argv)
    if not forwarded or forwarded == ["--help"]:
        if forwarded == ["--help"]:
            forwarded = ["--help"]
        else:
            parser.error("pass shared client options: --config, --benchmark, --history-kv-budget, --upstream, --out")
    try:
        argv = command(args.paper_root, args.python, forwarded)
    except ValueError as error:
        parser.error(str(error))
    root = args.paper_root.resolve()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    if os.name == "posix":
        # Let termination reach the shared client's owned-process cleanup.
        os.chdir(root)
        os.execvpe(argv[0], argv, env)
        return
    subprocess.run(argv, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
