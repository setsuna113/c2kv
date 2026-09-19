"""Build the seven-cell tool allocation x history recovery study.

This creates a configuration only. The existing paper runner owns execution,
official scoring and receipts. CUDA and NPU launchers consume the same source.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def joint_config(base, *, tool_checkpoint, benchmarks=("bfcl_base",),
                 tool_ratio=8, top_k=3, tool_budget_tokens=None, output_root=None):
    from benchmarks.toolmemory import parse_tool_memory_spec

    uniform = f"t0:r{tool_ratio}"
    hybrid = f"{uniform}:hybrid{top_k}"
    parse_tool_memory_spec(uniform)
    parse_tool_memory_spec(hybrid)
    if not str(tool_checkpoint):
        raise ValueError("A T0 tool checkpoint is required")
    available = {row["name"] for row in base["benchmarks"]}
    if not benchmarks or len(set(benchmarks)) != len(benchmarks) or not set(benchmarks) <= available:
        raise ValueError("Study benchmarks must be unique entries in the base configuration")
    if tool_budget_tokens is not None and (type(tool_budget_tokens) is not int or tool_budget_tokens <= 0):
        raise ValueError("The tool budget must be a positive integer")
    result = copy.deepcopy(base)
    result["output_root"] = str(output_root or (str(base["output_root"]) + "-tool-history"))
    if result["output_root"] == str(base["output_root"]):
        raise ValueError("The tool/history study needs a separate output root")
    result["benchmarks"] = [row for row in result["benchmarks"] if row["name"] in benchmarks]
    result.setdefault("c1", {}).update(detector="t02_risk", selector_threshold=0.5,
                                       history_variant="H0", recovery_rounds=1)
    result["tool_contexts"] = [
        {"name": "uniform", "spec": uniform, "checkpoint": str(tool_checkpoint)},
        {"name": "hybrid", "spec": hybrid, "checkpoint": str(tool_checkpoint)},
    ]
    if tool_budget_tokens is not None:
        for context in result["tool_contexts"]:
            context["budget_tokens"] = tool_budget_tokens
    result["methods"] = [
        {"method": "Full", "arm": "full", "group": "tool_history_anchor",
         "tool_contexts": ["raw", "uniform"]},
        {"method": "C1 initial allocation (recovery off)", "arm": "c2kv_c1_off_r8",
         "group": "tool_history_factorial", "ratio": 8,
         "tool_contexts": ["raw", "uniform", "hybrid"]},
        {"method": "C1 T02 recovery", "arm": "c2kv_c1_t02_r8",
         "group": "tool_history_factorial", "ratio": 8,
         "tool_contexts": ["uniform", "hybrid"]},
    ]
    result["tool_history_study"] = {
        "schema": "paper-tool-history-factorial-v1",
        "tool_factor": ["uniform", "hybrid"],
        "history_factor": ["c2kv_c1_off_r8", "c2kv_c1_t02_r8"],
        "fixed_initial_history_policy": "S0/H0/ratio8",
        "history_checkpoint": result["checkpoint"],
        "detector": "t02_risk", "threshold": 0.5,
        "budget_comparison": "fixed caps; report achieved tool/history/whole KV separately",
        "tool_allocation_comparison": "hybrid retains additional native schemas; equal ratio is not equal resident KV",
        "claim_scope": "paired component effects and interaction on the same official task cohort",
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--tool-checkpoint", required=True)
    parser.add_argument("--benchmarks", default="bfcl_base")
    parser.add_argument("--tool-ratio", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--tool-budget-tokens", type=int)
    parser.add_argument("--output-root", required=True,
                        help="Separate results directory for the seven-cell study")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = joint_config(json.loads(args.base_config.read_text(encoding="utf-8")),
                          tool_checkpoint=args.tool_checkpoint,
                          benchmarks=tuple(args.benchmarks.split(",")),
                          tool_ratio=args.tool_ratio, top_k=args.top_k,
                          tool_budget_tokens=args.tool_budget_tokens,
                          output_root=args.output_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(str(args.out.resolve()))


if __name__ == "__main__":
    main()
