"""Isolated BFCL generation process.

The official BFCL Typer generate command terminates its Python process after
completion. The adapter launches this wrapper as a child so generation cannot
prevent the parent from running the official evaluator and writing summary files.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--categories", default="multi_turn_base")
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--served-model-name", required=True)
    parser.add_argument("--run-ids", action="store_true")
    args = parser.parse_args()

    from adapters.bfcl_adapter import _benchmark_dir, install_handler, run_cli

    checkout = _benchmark_dir(args.benchmark_dir)
    project_root = args.out_dir.resolve()
    os.environ["BFCL_PROJECT_ROOT"] = str(project_root)
    safe_arm = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in args.arm
    )
    registry_name = f"c2kv-bfcl-{safe_arm}"
    install_handler(args.base_url, registry_name, args.served_model_name)
    command = [
        "generate",
        "--model", registry_name,
        "--test-category", args.categories,
        "--result-dir", "result",
    ]
    if args.run_ids:
        command.append("--run-ids")
    print(f"BFCL checkout: {checkout}", flush=True)
    run_cli(command)


if __name__ == "__main__":
    main()
