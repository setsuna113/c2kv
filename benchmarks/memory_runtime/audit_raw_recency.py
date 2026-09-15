"""Replay recorded Full requests on CPU to verify raw-recency budget behavior."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import proxy
from arms import get_arm
from memory_runtime.adapter import RuntimeAdapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prototype = RuntimeAdapter.from_config(str(HERE / "configs/full_shared.json"), args.tokenizer)
    configuration = json.loads((HERE / "configs/full_shared.json").read_text())
    budgets = (1536, 768)
    rows = []
    source_count = 0
    for path in sorted((args.root / "full/logs").glob("proxy_*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            request = json.loads(line)
            native = request["request_view"]
            context = request["eval_context"]
            render = lambda source: proxy._assemble(source, get_arm("full"))
            full, full_counts = render(native["messages"])
            wire = request["forwarded_request_views"]
            if len(wire) != 1 or full != wire[0]["messages"]:
                raise ValueError("Full renderer does not reproduce the recorded forwarded view")
            tools = native.get("tools")
            full_tokens = prototype._token_counter(full, tools)
            if full_tokens != request["usage"]["prompt_tokens"]:
                raise ValueError("Recorded Full raw prompt token parity failed")
            if request["bytes_per_kv_token"] != configuration["bytes_per_kv_token"]:
                raise ValueError("Recorded KV geometry differs from the configuration")
            reference, counts = prototype.apply(native["messages"], full, full_counts, context, tools)
            if reference != full:
                raise ValueError("Full raw-history accounting altered the reference")
            full_history_tokens = counts["memory_runtime"]["raw_history_tokens"]
            for token_budget in budgets:
                config = dict(configuration, mode="raw_recency",
                              run_id=context.get("run_id", configuration["run_id"]),
                              history_budget_bytes=token_budget * configuration["bytes_per_kv_token"])
                runtime = RuntimeAdapter(config, prototype._token_counter)
                selected, updated = runtime.apply(native["messages"], full, full_counts, context,
                                                   tools, render_full=render)
                meta = updated["memory_runtime"]
                identity = selected == full
                original_fits = full_history_tokens <= token_budget
                if identity != original_fits:
                    raise ValueError("Raw recency changed an in-budget Full view or kept an over-budget view")
                if (meta["active_history_bytes"] > config["history_budget_bytes"]
                        or meta["total_raw_prompt_tokens"] != prototype._token_counter(selected, tools)
                        or meta["gist_tokens"] != 0 or meta["evidence_bytes"] != 0):
                    raise ValueError("Raw recency failed its measured budget contract")
                rows.append({
                    "source_path": str(path.relative_to(args.root)), "source_line": number,
                    "context": context, "budget_token_equivalent": token_budget,
                    "full_raw_token_parity": True, "full_wire_identity": True,
                    "full_raw_history_tokens": full_history_tokens,
                    "full_history_within_budget": original_fits,
                    "raw_recency_matches_full": identity,
                    "memory_runtime": meta,
                })
            source_count += 1
    if not rows:
        raise ValueError("No recorded Full request views")
    pressured = [row for row in rows if row["budget_token_equivalent"] == 768
                 and not row["full_history_within_budget"]]
    result = {
        "schema": "a-runtime-raw-recency-cpu-audit-v1", "status": "passed",
        "additional_chat_requests": 0, "additional_extraction_requests": 0,
        "scope": "CPU selection and token accounting on recorded Full prefixes; no generation or task benefit measured",
        "budget_boundary": "system/tools/current input suffix excluded; earlier current user query included",
        "original_full_request_count": source_count,
        "b1536_full_identity_count": sum(row["raw_recency_matches_full"] for row in rows
                                         if row["budget_token_equivalent"] == 1536),
        "b768_over_budget_full_context_count": len(pressured),
        "b768_over_budget_full_task_ids": sorted({row["context"]["task_id"] for row in pressured}),
        "rows": rows,
    }
    source_bundle = HERE.parents[1] / "tmp/a_memory_runtime_20260907/source_bundle.json"
    if source_bundle.exists():
        result["source_bundle"] = json.loads(source_bundle.read_text())
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"rows", "source_bundle"}}))


if __name__ == "__main__":
    main()
