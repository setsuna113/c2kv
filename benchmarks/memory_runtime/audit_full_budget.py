"""CPU-only budget calibration of the recorded Full reference trajectories."""
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
    rows = []
    for path in (args.root / "full/logs").glob("proxy_*.jsonl"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            request = json.loads(line)
            view = request["request_view"]
            base, counts = proxy._assemble(view["messages"], get_arm("full"))
            assembled, counts = prototype.apply(view["messages"], base, counts,
                                                request["eval_context"], view.get("tools"))
            meta = counts["memory_runtime"]
            parity = meta["total_raw_prompt_tokens"] == request["usage"]["prompt_tokens"]
            geometry = meta["bytes_per_kv_token"] == request["bytes_per_kv_token"]
            if not parity or not geometry or assembled != base:
                raise ValueError("Full calibration differs from the executed reference view")
            rows.append({
                "source_path": str(path.relative_to(args.root)), "source_line": number,
                "context": request["eval_context"], "raw_token_parity": parity,
                "raw_history_tokens": meta["raw_history_tokens"],
                "active_history_bytes": meta["active_history_bytes"],
                "history_budget_bytes": meta["history_budget_bytes"],
                "within_history_budget": meta["active_history_bytes"] <= meta["history_budget_bytes"],
            })
    if not rows:
        raise ValueError("No captured Full requests")
    result = {
        "schema": "a-runtime-full-budget-calibration-v1", "status": "passed",
        "additional_chat_requests": 0, "additional_extraction_requests": 0,
        "scope": "measured raw Full reference trajectories only; no new raw-recency arm was executed",
        "budget_boundary": "same 1088 compatibility boundary: system/tools/current input suffix excluded; earlier user query included",
        "all_full_reference_views_within_budget": all(row["within_history_budget"] for row in rows),
        "raw_history_tokens_max": max(row["raw_history_tokens"] for row in rows),
        "active_history_bytes_max": max(row["active_history_bytes"] for row in rows),
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "all_full_reference_views_within_budget", "raw_history_tokens_max", "active_history_bytes_max")}))


if __name__ == "__main__":
    main()
