"""Posthoc-only inspection of a frozen capacity-protect rollout.

This file is intentionally outside the model source bundle freeze. It performs
no generation, extraction, or tokenization; all byte values come from the
recorded proxy rows.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "python"))
import proxy
from arms import get_arm
from history_memory.events import EventStore
from history_memory.evidence import evidence_message


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _rows(root: Path):
    paths = sorted((root / "capacity_protect" / "logs").glob("proxy_*.jsonl"))
    if not paths:
        raise ValueError("No capacity_protect proxy logs")
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: request row must be an object")
            yield path, number, value


def _inspect_row(root: Path, path: Path, number: int, request: dict) -> dict:
    source = request["request_view"]["messages"]
    forwarded = request["forwarded_request_views"]
    metadata = request["memory_runtime"]
    gate = metadata["capacity_gate"]
    context = request["eval_context"]
    if not isinstance(source, list) or not all(isinstance(item, dict) for item in source):
        raise ValueError("request_view.messages is not a message list")
    if len(forwarded) != 1 or not isinstance(forwarded[0].get("messages"), list):
        raise ValueError("expected exactly one captured forwarded message view")
    wire = forwarded[0]["messages"]

    activated = gate.get("compression_activated")
    if type(activated) is not bool:
        raise ValueError("capacity activation is not boolean")
    full_bytes = _nonnegative_int(gate.get("full_history_bytes"), "full_history_bytes")
    budget = _nonnegative_int(gate.get("history_budget_bytes"), "history_budget_bytes")

    full_wire_exact = None
    evidence_packet_exact = None
    if not activated:
        expected, _ = proxy._assemble(source, get_arm("full"))
        full_wire_exact = wire == expected
        if not full_wire_exact:
            raise ValueError("below-B forwarded messages differ from current Full assembly")
    else:
        task_id = context.get("task_id")
        selected = metadata.get("selected_event_ids")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("activated request lacks task_id")
        if (not isinstance(selected, list)
                or any(not isinstance(event_id, str) for event_id in selected)):
            raise ValueError("activated request has invalid selected_event_ids")
        expected_packet = evidence_message(
            EventStore.from_messages(task_id, source), selected)
        packet_index = metadata.get("evidence_out_index")
        if expected_packet is None:
            evidence_packet_exact = packet_index is None
        else:
            evidence_packet_exact = (
                type(packet_index) is int
                and 0 <= packet_index < len(wire)
                and wire[packet_index] == expected_packet
                and [i for i, message in enumerate(wire) if message == expected_packet]
                == [packet_index]
            )
        if not evidence_packet_exact:
            raise ValueError("activated evidence packet differs from selected source events")

    bytes_per_token = _nonnegative_int(
        metadata.get("bytes_per_kv_token"), "bytes_per_kv_token")
    if bytes_per_token == 0:
        raise ValueError("bytes_per_kv_token must be positive")
    raw_tokens = _nonnegative_int(metadata.get("raw_history_tokens"), "raw_history_tokens")
    gist_tokens = _nonnegative_int(metadata.get("gist_tokens"), "gist_tokens")
    evidence_bytes = _nonnegative_int(metadata.get("evidence_bytes"), "evidence_bytes")
    active_bytes = _nonnegative_int(
        metadata.get("active_history_bytes"), "active_history_bytes")
    packed_candidates = request.get("history_packed_candidate_doc_count")
    packed_candidates = packed_candidates if type(packed_candidates) is int else None
    gist_messages = request.get("n_gist_messages")
    gist_messages = gist_messages if type(gist_messages) is int else None
    source_ref = {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "line": number,
    }
    return {
        "source": source_ref,
        "context": context,
        "native_tool_names": request.get("native_tool_names"),
        "compression_activated": activated,
        "full_history_vs_B": {
            "full_history_bytes": full_bytes,
            "history_budget_bytes": budget,
            "relation": "above" if full_bytes > budget else "at_or_below",
            "over_budget_bytes": max(0, full_bytes - budget),
        },
        "checks": {
            "below_B_full_wire_exact": full_wire_exact,
            "activated_evidence_packet_exact": evidence_packet_exact,
        },
        "bytes": {
            "raw": raw_tokens * bytes_per_token,
            "gist": gist_tokens * bytes_per_token,
            "evidence": evidence_bytes,
            "active": active_bytes,
        },
        "counts": {
            "retained_gist_messages": gist_messages,
            "packed_candidate_docs": packed_candidates,
            "exact_extraction_requests": None,
        },
    }


def inspect(root: Path) -> dict:
    root = root.resolve()

    def forbidden(*args, **kwargs):
        raise AssertionError("posthoc inspection cannot call generation or extraction")

    original_post, original_extract = proxy._post_json, proxy._extract
    proxy._post_json = forbidden
    proxy._extract = forbidden
    rows = []
    try:
        for path, number, request in _rows(root):
            try:
                rows.append(_inspect_row(root, path, number, request))
            except (AssertionError, KeyError, TypeError, ValueError) as error:
                relative = path.resolve().relative_to(root).as_posix()
                raise ValueError(f"{relative}:{number}: {error}") from error
    finally:
        proxy._post_json, proxy._extract = original_post, original_extract
    if not rows:
        raise ValueError("Capacity-protect proxy logs contain no request rows")
    activated = [row for row in rows if row["compression_activated"]]
    gist_counts = [row["counts"]["retained_gist_messages"] for row in rows
                   if row["counts"]["retained_gist_messages"] is not None]
    candidate_counts = [row["counts"]["packed_candidate_docs"] for row in rows
                        if row["counts"]["packed_candidate_docs"] is not None]
    return {
        "schema": "a-runtime-capacity-rollout-inspection-v1",
        "status": "passed",
        "analysis": "posthoc_only_after_model_source_bundle_freeze",
        "additional_chat_requests": 0,
        "additional_extraction_requests": 0,
        "exact_extraction_request_count_available": False,
        "packed_candidate_doc_count_note": (
            "recorded final packed documents, not an exact extraction-request count"
        ),
        "request_count": len(rows),
        "below_B_full_identity_count": len(rows) - len(activated),
        "activation_count": len(activated),
        "all_below_B_full_wire_exact": True,
        "all_activated_evidence_packets_exact": True,
        "retained_gist_message_count_recorded": sum(gist_counts) if gist_counts else None,
        "packed_candidate_doc_count_recorded": (
            sum(candidate_counts) if candidate_counts else None),
        "activation_contexts": [
            {"source": row["source"], "context": row["context"],
             "native_tool_names": row["native_tool_names"],
             "full_history_vs_B": row["full_history_vs_B"],
             "counts": row["counts"], "bytes": row["bytes"]}
            for row in activated
        ],
        "byte_totals": {
            key: sum(row["bytes"][key] for row in rows)
            for key in ("raw", "gist", "evidence", "active")
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"rows", "activation_contexts"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
