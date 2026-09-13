#!/usr/bin/env python3
"""Run bounded greedy evaluation for one next-compression checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from next_compression.inference import evaluate_checkpoint, save_evaluation


def _ratios(value: str) -> tuple[int, ...]:
    try:
        ratios = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("ratios must be comma-separated integers") from error
    if ratios != (8, 12):
        raise argparse.ArgumentTypeError("ratios must be exactly 8,12")
    return ratios


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratios", type=_ratios, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument(
        "--dtype", choices=("float32", "bfloat16", "float16"), required=True
    )
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-decisions", type=int, required=True)
    parser.add_argument(
        "--no-uniform-ce",
        action="store_true",
        help="Skip the optional uniform target CE pass.",
    )
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.max_decisions <= 0:
        parser.error("--max-decisions must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> dict:
    args = arguments(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Evaluation output already exists: {output}")
    result = evaluate_checkpoint(
        args.checkpoint,
        args.data_root,
        ratios=args.ratios,
        max_new_tokens=args.max_new_tokens,
        max_decisions_per_ratio=args.max_decisions,
        device=args.device,
        dtype=args.dtype,
        compute_uniform_ce=not args.no_uniform_ce,
    )
    save_evaluation(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
