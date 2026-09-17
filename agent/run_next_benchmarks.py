#!/usr/bin/env python3
"""Freeze and run full official benchmark manifests for next-compression."""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from next_compression.benchmarks import (  # noqa: E402
    BENCHMARK_ORDER,
    build_plan,
    execute_plan,
    freeze_manifest,
    parse_benchmarks,
)


def _python(value: str | None) -> str:
    if not value:
        raise ValueError("a benchmark interpreter must be explicit")
    return value


def _print_plan(plan: dict) -> None:
    print(json.dumps({key: value for key, value in plan.items() if key != "entries"},
                     indent=2, ensure_ascii=False))
    for entry in plan["entries"]:
        print(shlex.join(entry["command"]))


def _prepare(args: argparse.Namespace) -> int:
    selected = parse_benchmarks(args.benchmarks)
    values = {
        "bfcl": (args.bfcl_root, args.bfcl_python),
        "tau2": (args.tau2_root, args.tau2_python),
        "toolsandbox": (args.toolsandbox_root, args.toolsandbox_python),
        "acebench": (args.acebench_root, args.acebench_python),
        "appworld": (args.acon_root, args.appworld_python),
    }
    missing = [name for name in selected if values[name][0] is None or values[name][1] is None]
    if missing:
        raise ValueError(f"selected benchmarks need explicit root and interpreter: {missing}")
    if "appworld" in selected and args.appworld_root is None:
        raise ValueError("AppWorld needs both --acon-root and --appworld-root")
    manifest = freeze_manifest(
        selected=selected,
        roots={name: Path(values[name][0]) for name in selected},
        pythons={name: _python(values[name][1]) for name in selected},
        user_endpoint=args.user_endpoint,
        user_model_alias=args.user_model_alias,
        output_dir=args.output_dir,
        appworld_root=args.appworld_root,
    )
    path = Path(args.output_dir).resolve() / "benchmark_manifest.json"
    print(json.dumps({"status": manifest["status"], "manifest": str(path),
                      "fixed_denominator": manifest["fixed_denominator"],
                      "benchmark_denominators": {
                          name: manifest["benchmarks"][name]["denominator"]
                          for name in manifest["benchmark_order"]}},
                     indent=2, ensure_ascii=False))
    return 0


def _execute(args: argparse.Namespace) -> int:
    selected = parse_benchmarks(args.benchmarks) if args.benchmarks else None
    plan = build_plan(args.manifest, args.endpoint, args.model_alias,
                      args.output_dir, benchmarks=selected,
                      smoke_tasks=args.smoke_tasks, expected_backend=args.expected_backend)
    _print_plan(plan)
    return execute_plan(plan, run=args.run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    prepare = sub.add_parser(
        "prepare",
        help="enumerate installed official sources and freeze exact full task IDs",
    )
    prepare.add_argument("--output-dir", required=True, type=Path)
    prepare.add_argument("--benchmarks", default=",".join(BENCHMARK_ORDER),
                         help="comma-separated subset; default is all five")
    prepare.add_argument("--user-endpoint", required=True,
                         help="independent uncompressed OpenAI /v1 endpoint")
    prepare.add_argument("--user-model-alias", required=True,
                         help="wire model accepted by the uncompressed endpoint")
    for name in ("bfcl", "tau2", "toolsandbox", "acebench"):
        prepare.add_argument(f"--{name}-root", type=Path)
        prepare.add_argument(f"--{name}-python")
    prepare.add_argument("--acon-root", type=Path,
                         help="pinned microsoft/acon checkout")
    prepare.add_argument("--appworld-root", type=Path,
                         help="installed AppWorld root containing data/datasets/test_normal.txt")
    prepare.add_argument("--appworld-python",
                         help="interpreter with ACON and appworld==0.1.3.post1")
    prepare.set_defaults(func=_prepare)

    execute = sub.add_parser(
        "execute",
        help="print an official execution plan; add --run only on the H100 host",
    )
    execute.add_argument("--manifest", required=True, type=Path)
    execute.add_argument("--output-dir", required=True, type=Path)
    execute.add_argument("--endpoint", required=True,
                         help="evaluated candidate OpenAI /v1 endpoint")
    execute.add_argument("--model-alias", required=True,
                         help="candidate wire model alias advertised by /health")
    execute.add_argument("--expected-backend", choices=("sglang", "native"), default="sglang",
                         help="require this candidate backend; native is an explicit reference run")
    execute.add_argument("--benchmarks", default="",
                         help="comma-separated subset; omitted uses every frozen benchmark")
    execute.add_argument("--smoke-tasks", type=int,
                         help="explicit artifact-scope smoke prefix per benchmark; omitted is full")
    execute.add_argument("--run", action="store_true",
                         help="actually invoke official harnesses (default only writes/prints plan)")
    execute.set_defaults(func=_execute)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
