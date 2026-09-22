"""Compare static and event-updated tool selection with fixed history recovery."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from benchmarks.paper.candidate_matrix import with_candidate_methods


def joint_selector_config(base, *, tool_checkpoint, benchmarks=("bfcl_base",),
                          output_root):
    from benchmarks.toolmemory import parse_tool_memory_spec

    if not str(tool_checkpoint):
        raise ValueError("A T0 tool checkpoint is required")
    if not str(output_root) or str(output_root) == str(base["output_root"]):
        raise ValueError("The selector study needs a separate output root")
    result = copy.deepcopy(base)
    result.pop("tool_history_study", None)
    result["methods"] = []
    result = with_candidate_methods(result, ("pending_verified",), benchmarks)
    result["benchmarks"] = [row for row in result["benchmarks"] if row["name"] in benchmarks]
    result["output_root"] = str(output_root)
    result.setdefault("c1", {}).update(detector="t02_risk", selector_threshold=0.5,
                                       history_variant="H0", recovery_rounds=1)
    policies = ("last_user_topk_v1", "latest_event_topk_v1")
    result["tool_contexts"] = []
    for policy in policies:
        spec = "t0:r8:hybrid3:schema"
        if policy != "last_user_topk_v1":
            spec += ":selector=" + policy
        parse_tool_memory_spec(spec)
        result["tool_contexts"].append({
            "name": policy, "spec": spec, "checkpoint": str(tool_checkpoint),
        })
    result["methods"][0]["tool_contexts"] = list(policies)
    result["tool_selector_study"] = {
        "schema": "paper-tool-selector-comparison-v1",
        "selector_policies": list(policies),
        "fixed_history_arm": "c2kv_pending_verified_r8",
        "fixed_tool_encoder": "t0",
        "fixed_tool_ratio": 8,
        "fixed_top_k": 3,
        "interface_policy": "schema",
        "tool_budget_tokens": None,
        "query_update": "after completed execution; fixed during draft and recovery",
    }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--tool-checkpoint", required=True)
    parser.add_argument("--benchmarks", default="bfcl_base")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = joint_selector_config(
        json.loads(args.base_config.read_text(encoding="utf-8")),
        tool_checkpoint=args.tool_checkpoint, benchmarks=tuple(args.benchmarks.split(",")),
        output_root=args.output_root,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(str(args.out.resolve()))


if __name__ == "__main__":
    main()
