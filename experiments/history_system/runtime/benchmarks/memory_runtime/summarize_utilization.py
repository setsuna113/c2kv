"""Collect recorded probe responses without assigning task-success labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def collect(root):
    receipt = json.loads((root / "receipt.json").read_text())
    rows = []
    wires = []
    for path in sorted([*root.glob("layout_*.json"), *root.glob("lease_*.json")]):
        item = json.loads(path.read_text())
        if not isinstance(item, dict) or "cell_id" not in item:
            continue
        meta = item["counts"]["memory_runtime"]
        answer = item.get("normalized", {})
        wires.append((item["cell_id"], item.get("forwarded"), answer))
        rows.append({
            "cell_id": item["cell_id"], "label": item["label"], "context": item["eval_context"],
            "status": item["status"], "source_file": path.name,
            "selected_event_ids": meta["selected_event_ids"],
            "retained_event_ids": meta.get("retained_event_ids", []),
            "gist_tokens": meta["gist_tokens"], "raw_history_tokens": meta["raw_history_tokens"],
            "evidence_bytes": meta["evidence_bytes"], "active_history_bytes": meta["active_history_bytes"],
            "raw_token_parity": meta.get("raw_prompt_tokens_verified_by_backend", False),
            "byte_geometry_verified": meta.get("byte_geometry_verified_by_backend", False),
            "content": answer.get("content"), "tool_calls": answer.get("tool_calls"),
            "finish_reason": answer.get("finish_reason"), "usage": answer.get("usage"),
            "chat_wall_seconds": item.get("chat_wall_seconds"),
        })
    transport = [json.loads(line) for line in (root / "transport.jsonl").read_text().splitlines() if line]
    extracts = [row for row in transport if row["path"] != "/v1/chat/completions"]
    equal_inputs = []
    for index, (left_id, left_wire, left) in enumerate(wires):
        for right_id, right_wire, right in wires[index + 1:]:
            if left_wire is not None and left_wire == right_wire:
                equal_inputs.append({
                    "cells": [left_id, right_id], "forwarded_inputs_identical": True,
                    "content_identical": left.get("content") == right.get("content"),
                    "tool_functions_identical": [call.get("function") for call in left.get("tool_calls") or []] == [call.get("function") for call in right.get("tool_calls") or []],
                    "finish_reason_identical": left.get("finish_reason") == right.get("finish_reason"),
                })
    return {
        "schema": "a-runtime-utilization-collection-v1",
        "scope": "fixed-prefix response inspection; preliminary, n=1; no official task score",
        "source_commit": receipt.get("source_commit"), "receipt_status": receipt["status"],
        "chat_attempts": receipt["chat_attempts"], "chat_completed": receipt["chat_completed"],
        "row_count": len(rows), "all_response_checks_passed": bool(rows) and all(
            row["status"] == "completed" and row["raw_token_parity"] and row["byte_geometry_verified"] for row in rows),
        "extraction_attempts": len(extracts),
        "extraction_wall_seconds": sum(row["wall_seconds"] for row in extracts),
        "lease_differences": receipt.get("lease_differences"), "rows": rows,
        "wire_identical_pairs": equal_inputs,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = collect(args.root)
    (args.root / "collection.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("receipt_status", "row_count", "chat_completed", "all_response_checks_passed")}))
