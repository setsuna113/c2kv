"""Read-only classification of the finite frozen-view continuation diagnosis."""
from __future__ import annotations

import json
import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _invalid_constant(value):
    raise ValueError("Non-finite JSON number")


def classify_response(response: Mapping[str, Any], tools: list[dict]) -> dict[str, Any]:
    """Recognize native call structure without executing or judging the action."""
    declared = {tool.get("function", {}).get("name") for tool in tools
                if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)}
    result = {"native_continuation": None, "native_present": False,
              "category": "malformed_response", "signature": None, "finish_reason": None}
    choices = response.get("choices") if isinstance(response, Mapping) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        return result
    choice = choices[0]
    result["finish_reason"] = choice.get("finish_reason")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        return result
    calls = message.get("tool_calls")
    result["native_present"] = isinstance(calls, list) and bool(calls)
    signatures = []
    malformed = False
    if calls is not None and not isinstance(calls, list):
        malformed = True
    for call in calls if isinstance(calls, list) else []:
        if (not isinstance(call, Mapping) or call.get("type") != "function"
                or not isinstance(call.get("function"), Mapping)):
            malformed = True
            break
        function = call["function"]
        name = function.get("name")
        if not isinstance(name, str) or name not in declared:
            malformed = True
            break
        arguments = function.get("arguments")
        try:
            if not isinstance(arguments, str):
                raise ValueError("Native arguments must be a JSON string")
            decoded = json.loads(arguments, parse_constant=_invalid_constant)
            if not isinstance(decoded, dict):
                raise ValueError("Native arguments must decode to an object")
        except (ValueError, TypeError):
            malformed = True
            break
        signatures.append({"name": name, "arguments": decoded})
    if not malformed:
        result["signature"] = json.dumps(signatures, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if choice.get("finish_reason") == "length":
        result["category"] = "truncated_generation"
    elif malformed:
        result.update(category="malformed_native_call", native_continuation=False)
    elif signatures:
        result.update(category="native_call", native_continuation=True)
    else:
        result.update(category="no_native_call", native_continuation=False)
    return result


def summarize_responses(rows: list[dict]) -> dict[str, Any]:
    """Aggregate technical repeats without treating censored samples as stops."""
    classifications = [row["classification"] for row in rows]
    observed = [row["native_continuation"] for row in classifications
                if row["native_continuation"] is not None]
    signatures = Counter(row["signature"] for row in classifications if row["signature"] is not None)
    return {
        "completed_cells": len(rows),
        "scorable_cells": len(observed),
        "native_continuation_count": sum(observed),
        "censored_or_malformed_response_cells": len(rows) - len(observed),
        "categories": dict(Counter(row["category"] for row in classifications)),
        "primary_sequence": [row["native_continuation"] for row in classifications],
        "primary_all_continue": len(observed) == len(rows) and bool(rows) and all(observed),
        "primary_all_stop": len(observed) == len(rows) and bool(rows) and not any(observed),
        "action_signatures": [{"signature": signature, "count": count}
                              for signature, count in sorted(signatures.items())],
        "replication_scope": "fixed-seed technical repeats in one task, not independent seeds",
    }


def next_action(groups: Mapping[str, Mapping[str, Any]], complete: bool) -> dict[str, str]:
    """Apply only the finite protocol's predeclared local follow-up rules."""
    scope = "local diagnosis only; no task-success, recovery-effect, or significance claim"
    if not complete or any(groups.get(key, {}).get("completed_cells") != 4 for key in (
            "negative:A", "negative:C", "main:A", "main:B", "main:C", "main:D")):
        return {"decision": "incomplete_no_refill", "scope": scope}
    if any(group["scorable_cells"] != group["completed_cells"] for group in groups.values()):
        return {"decision": "censored_or_invalid_response_no_expansion", "scope": scope}

    negative_a, negative_c = groups["negative:A"], groups["negative:C"]
    same_input_stable = ((negative_a["primary_all_continue"] and negative_c["primary_all_continue"])
                         or (negative_a["primary_all_stop"] and negative_c["primary_all_stop"]))
    negative_split = ((negative_a["primary_all_continue"] and negative_c["primary_all_stop"])
                      or (negative_a["primary_all_stop"] and negative_c["primary_all_continue"]))
    a, b, c, d = (groups["main:" + view] for view in "ABCD")
    if negative_split:
        decision = "inspect_resolved_request_and_service_order"
    elif a["primary_all_continue"] and all(group["primary_all_stop"] for group in (b, c, d)):
        decision = "inspect_common_evidence_renderer"
    elif (same_input_stable and all(group["primary_all_continue"] for group in (a, b, d))
          and c["primary_all_stop"]):
        decision = "inspect_c2kv_layout_positions_and_serving"
    elif all(group["primary_all_continue"] for group in (a, b, c, d)):
        decision = "no_local_stop_reproduced_end_diagnosis"
    elif a["primary_all_stop"]:
        decision = "full_reference_did_not_continue_end_diagnosis"
    else:
        decision = "mixed_pattern_end_without_expansion"
    return {"decision": decision, "scope": scope}


def analyze_directory(root: Path, prepared_path: Path) -> dict[str, Any]:
    """Validate recorded cells against the frozen schedule without model calls."""
    from .attempt_journal import read_attempt_journal, summarize_attempt_journal
    from .frozen_view_probe import _validate_prepared, _file_digest

    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    _validate_prepared(prepared)
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    errors = []
    if (receipt.get("freeze_digest") != prepared["freeze_digest"]
            or receipt.get("prepared_file_sha256") != _file_digest(prepared_path)):
        errors.append("Receipt does not identify the supplied frozen input")
    cells_path = root / "cells.jsonl"
    rows = [json.loads(line) for line in cells_path.read_text(encoding="utf-8").splitlines()
            if line.strip()] if cells_path.exists() else []
    groups = {key: [] for key in prepared["views"]}
    journal_path = root / "attempts.jsonl"
    accounting = summarize_attempt_journal(journal_path) if journal_path.exists() else None
    starts = [row for row in read_attempt_journal(journal_path)["records"]
              if row["event"] == "started" and row["kind"] == "generation"] if accounting else []
    if len(rows) > len(prepared["schedule"]):
        errors.append("More cells than the frozen schedule")
    classified = []
    for index, row in enumerate(rows):
        if index >= len(prepared["schedule"]):
            break
        cell = prepared["schedule"][index]
        if any(row.get(key) != cell[key] for key in ("cell_id", "block", "phase", "view")):
            errors.append(f"Cell {index + 1} differs from the frozen order")
            continue
        key = f"{cell['phase']}:{cell['view']}"
        if row.get("wire_digest") != prepared["views"][key]["wire_digest"]:
            errors.append(f"Cell {cell['cell_id']} has a different input")
            continue
        unsubmitted_terminal = (row.get("generation_attempt_index") is None
                                and row.get("status") != "completed"
                                and index == len(starts) and index == len(rows) - 1)
        if not unsubmitted_terminal and (index >= len(starts) or any(
                row.get(key) != starts[index].get(journal_key)
                for key, journal_key in (("request_id", "request_id"),
                                         ("generation_attempt_index", "attempt_index")))):
            errors.append(f"Cell {cell['cell_id']} does not match its durable attempt")
        entry = {key: row[key] for key in ("cell_id", "block", "phase", "view", "status")}
        if row["status"] == "completed":
            entry["classification"] = classify_response(
                row.get("response"), prepared["views"][key]["payload"].get("tools") or [])
            groups[key].append(entry)
        else:
            entry["classification"] = {"category": "transport_or_backend_failure",
                                       "native_continuation": None}
        classified.append(entry)
    if receipt.get("status") == "completed":
        if len(rows) != len(prepared["schedule"]) or any(row["status"] != "completed" for row in rows):
            errors.append("Completed receipt lacks the complete successful cell schedule")
        if (accounting is None or accounting["pending"] or accounting["failed"]
                or accounting["truncated_tail"]
                or accounting["by_kind"]["generation"]["started"] != len(prepared["schedule"])
                or accounting["by_kind"]["extraction"]["started"] != prepared["extraction_manifest"]["count"]):
            errors.append("Completed receipt disagrees with the durable attempt journal")
    summarized = {key: summarize_responses(value) for key, value in groups.items()}
    complete = receipt.get("status") == "completed" and not errors
    return {
        "schema": "a-frozen-view-analysis-v1", "status": receipt.get("status"),
        "validation_errors": errors, "complete_valid": complete,
        "scope": prepared["design"]["scope"], "sample_label": "preliminary, n=1",
        "task_id": prepared["design"]["task_id"],
        "repeat_scope": prepared["design"]["repeat_scope"],
        "freeze_digest": prepared["freeze_digest"], "groups": summarized,
        "cells": classified, "next_action": next_action(summarized, complete),
        "attempt_journal": accounting, "wall_seconds": receipt.get("wall_seconds"),
        "tool_execution": False, "scorer_access": False, "regeneration": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_directory(args.root, args.prepared)
    target = args.root / "analysis.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"analysis_path": str(target), "status": result["status"],
                      "complete_valid": result["complete_valid"],
                      "next_action": result["next_action"],
                      "validation_errors": result["validation_errors"]}))
    return 0 if not result["validation_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
