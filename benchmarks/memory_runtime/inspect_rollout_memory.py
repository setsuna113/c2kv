"""Inspect first-input controls and evidence actually carried in a valid pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect(root):
    collection = json.loads((root / "collection.json").read_text())
    if collection["status"] != "valid":
        raise ValueError("Inspect only a fully validated official pilot")
    rows = {
        variant: [(path, number, json.loads(line))
                  for path in (root / variant / "logs").glob("proxy_*.jsonl")
                  for number, line in enumerate(path.read_text().splitlines(), 1) if line.strip()]
        for variant in collection["manifest"]["variants"]
    }
    firsts = {(row["eval_context"]["task_id"]): row for _, _, row in rows["full"]
              if row["eval_context"]["user_turn"] == row["eval_context"]["step"] == 0}
    controls, carried = [], []
    for variant, requests in rows.items():
        for path, number, row in requests:
            context = row["eval_context"]
            source = {"path": str(path.relative_to(root)), "line": number}
            if context["user_turn"] == context["step"] == 0:
                controls.append({
                    "variant": variant, "task_id": context["task_id"], "source": source,
                    "forwarded_input_identical_to_full": row["forwarded_request_views"] == firsts[context["task_id"]]["forwarded_request_views"],
                    "native_tool_names": row["native_tool_names"],
                })
            metadata = row.get("memory_runtime") or {}
            retained = metadata.get("retained_event_ids") or []
            if not retained:
                continue
            packet_index = metadata["evidence_out_index"]
            message = row["forwarded_request_views"][0]["messages"][packet_index]
            envelope = json.loads(message["content"].split("\n", 1)[1])
            event_ids = [event["event_id"] for event in envelope["events"]]
            carried.append({
                "variant": variant, "context": context, "source": source,
                "retained_event_ids": retained, "packet_event_ids": event_ids,
                "all_retained_events_in_forwarded_packet": set(retained).issubset(event_ids),
            })
    return {
        "schema": "a-runtime-rollout-memory-inspection-v1", "source_collection": "collection.json",
        "scope": "input and memory carriage checks; no additional generation or task-success metric",
        "first_input_controls": controls,
        "all_first_inputs_identical_to_full": bool(controls) and all(row["forwarded_input_identical_to_full"] for row in controls),
        "retained_packets": carried,
        "all_retained_events_carried": bool(carried) and all(row["all_retained_events_in_forwarded_packet"] for row in carried),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = inspect(args.root)
    (args.root / "memory_inspection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"first_inputs": len(result["first_input_controls"]), "first_inputs_identical": result["all_first_inputs_identical_to_full"],
                      "retained_packets": len(result["retained_packets"]), "retained_events_carried": result["all_retained_events_carried"]}))
