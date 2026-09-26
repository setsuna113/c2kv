#!/usr/bin/env python3
"""Evaluate every Exp 1 tool-allocation layout of one frozen manifest.

The T0 checkpoint serves the gist layouts (full, uniform, hybrid, random,
retrieval) and lends its frozen base weights to the eviction layouts
(snapkv, snapkv_hybrid, h2o, h2o_hybrid).  The T1 checkpoint serves the
field-level layout.  The output JSON carries every generated continuation,
per-cell metrics, paired exact tests, and a contract binding the manifest,
checkpoints, code, and protocol.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "python"))

from next_compression.exp1_eval import EvictionSettings, evaluate_exp1, save_evaluation
from next_compression.exp1_tools import EVICTION_LAYOUTS, GIST_LAYOUTS

ALL_LAYOUTS = GIST_LAYOUTS + EVICTION_LAYOUTS


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--checkpoint-t0", type=Path, required=True)
    parser.add_argument("--checkpoint-t1", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts", nargs="+", default=list(ALL_LAYOUTS), choices=ALL_LAYOUTS)
    parser.add_argument("--hybrid-k", type=int, default=3,
                        help="k of the hybrid records that protect and budget the *_hybrid eviction layouts")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--obs-window", type=int, default=16)
    parser.add_argument("--kernel", type=int, default=7)
    parser.add_argument("--h2o-recent-fraction", type=float, default=0.5)
    parser.add_argument("--eviction-chunk-size", type=int, default=2048)
    parser.add_argument("--no-uniform-ce", action="store_true")
    parser.add_argument("--hash-weights", action="store_true",
                        help="Also record the SHA256 of every safetensors shard (slow)")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0 or args.hybrid_k <= 0:
        parser.error("--max-new-tokens and --hybrid-k must be positive")
    if "t1" in args.layouts and args.checkpoint_t1 is None:
        parser.error("--checkpoint-t1 is required for the t1 layout")
    if args.obs_window <= 0 or args.kernel <= 0 or args.kernel % 2 == 0:
        parser.error("--obs-window must be positive and --kernel a positive odd integer")
    if not 0.0 <= args.h2o_recent_fraction <= 1.0:
        parser.error("--h2o-recent-fraction must lie in [0, 1]")
    if args.eviction_chunk_size <= 0:
        parser.error("--eviction-chunk-size must be positive")
    if args.output.exists():
        parser.error(f"--output already exists: {args.output}")
    return args


def _table(result: dict) -> str:
    lines = [
        f"{'cell':<28}{'n':>5}{'call':>8}{'name':>8}{'FC':>8}{'KV':>10}",
    ]
    for cell, metric in result["metrics"].items():
        call = metric["strict_ordered_call_accuracy"]
        name = metric["tool_name_accuracy"]
        fc = metric["false_tool_call_rate"]
        fmt = lambda value: "  --  " if value is None else f"{100 * value:6.1f}"
        lines.append(
            f"{cell:<28}{metric['records']:>5}{fmt(call):>8}{fmt(name):>8}{fmt(fc):>8}"
            f"{metric['resident_kv_tokens_mean']:>10.1f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> dict:
    args = arguments(argv)
    result = evaluate_exp1(
        manifest_root=args.manifest_root,
        checkpoint_t0=args.checkpoint_t0,
        checkpoint_t1=args.checkpoint_t1,
        layouts=tuple(args.layouts),
        hybrid_k=args.hybrid_k,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        eviction=EvictionSettings(
            obs_window=args.obs_window,
            kernel=args.kernel,
            recent_fraction=args.h2o_recent_fraction,
            chunk_size=args.eviction_chunk_size,
        ),
        compute_uniform_ce=not args.no_uniform_ce,
        hash_weights=args.hash_weights,
    )
    save_evaluation(args.output, result)
    print(_table(result))
    print(json.dumps({"event": "exp1_evaluation_written", "path": str(args.output.resolve()),
                      "records": result["records_count"], "cells": result["cells"]}), flush=True)
    return result


if __name__ == "__main__":
    main()
