"""Verify actual raw-recency source selection against forwarded rollout views."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import proxy
from arms import get_arm
from memory_runtime.adapter import EventStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted((args.root / "raw_recency/logs").glob("proxy_*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            request = json.loads(line)
            metadata = request["memory_runtime"]
            context = request["eval_context"]
            source = request["request_view"]["messages"]
            indices = metadata["selected_source_indices"]
            selected = set(indices)
            store = EventStore.from_messages(context["task_id"], source)
            expected, _ = proxy._assemble([source[index] for index in indices], get_arm("full"))
            full, _ = proxy._assemble(source, get_arm("full"))
            forwarded = request["forwarded_request_views"]
            wire_identity = len(forwarded) == 1 and expected == forwarded[0]["messages"]
            events_complete = all(
                not selected.intersection(event.source_indices)
                or set(event.source_indices) <= selected
                for event in store.events
            )
            actual_ids = [event.event_id for event in store.events
                          if set(event.source_indices) <= selected]
            checks = {
                "selected_view_matches_wire": wire_identity,
                "selected_events_are_whole": events_complete,
                "selected_event_ids_match_sources": actual_ids == metadata["selected_event_ids"],
                "all_fit_flag_matches_full_identity": (expected == full) == metadata["all_history_fits"],
                "indices_unique_ordered": indices == sorted(selected),
            }
            if not all(checks.values()):
                raise ValueError(f"Raw rollout selection mismatch at {path}:{number}: {checks}")
            rows.append({
                "source_path": str(path.relative_to(args.root)), "source_line": number,
                "context": context, "checks": checks,
                "all_history_fits": metadata["all_history_fits"],
                "raw_history_tokens": metadata["raw_history_tokens"],
                "active_history_bytes": metadata["active_history_bytes"],
                "evicted_event_ids": metadata["evicted_event_ids"],
                "native_tool_names": request["native_tool_names"],
                "finish_reason": request["finish_reason"],
            })
    if not rows:
        raise ValueError("No raw-recency requests")
    evicted = [row for row in rows if row["evicted_event_ids"]]
    result = {
        "schema": "a-runtime-raw-rollout-inspection-v1", "status": "passed",
        "additional_chat_requests": 0, "additional_extraction_requests": 0,
        "scope": "actual event selection and wire identity; task success comes from the official scorer",
        "request_count": len(rows), "requests_with_eviction": len(evicted),
        "task_ids_with_eviction": sorted({row["context"]["task_id"] for row in evicted}),
        "raw_history_tokens_max": max(row["raw_history_tokens"] for row in rows),
        "rows": rows,
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
