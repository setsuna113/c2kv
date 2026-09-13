"""Bounded development-set action scoring and per-ratio candidate selection."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any


def canonical_calls(calls):
    if not isinstance(calls, list):
        raise ValueError("Tool calls must be a list")
    result = []
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("Each tool call must be an object")
        function = call.get("function", call)
        if not isinstance(function, dict):
            raise ValueError("A tool-call function must be an object")
        name, arguments = function.get("name"), function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            raise ValueError("Tool calls require a function name and object arguments")
        result.append({"name": name, "arguments": arguments})
    return result


def parse_calls(text: str):
    """Parse complete Qwen tool-call JSON, including closing tags inside strings."""
    calls, cursor = [], 0
    decoder = json.JSONDecoder()
    while True:
        start = text.find("<tool_call>", cursor)
        if start < 0:
            if "</tool_call>" in text[cursor:]:
                raise ValueError("Unmatched tool-call closing tag")
            return canonical_calls(calls)
        payload = text[start + len("<tool_call>"):].lstrip()
        value, end = decoder.raw_decode(payload)
        remainder = payload[end:].lstrip()
        if not remainder.startswith("</tool_call>") or not isinstance(value, dict):
            raise ValueError("Incomplete tool call")
        calls.append(value)
        cursor = len(text) - len(remainder) + len("</tool_call>")


def score_records(records):
    """Strict ordered action match is a dev proxy, not official task success."""
    if not records:
        raise ValueError("An empty evaluation cannot select a checkpoint")
    tool_total = tool_correct = non_tool_total = false_calls = invalid = 0
    losses = []
    seen = set()
    for row in records:
        identity = (row["decision_id"], row["ratio"])
        if identity in seen:
            raise ValueError("Duplicate evaluation decision/ratio")
        seen.add(identity)
        metadata = row.get("metadata") or {}
        supplied_gold = row.get("gold_tool_calls", metadata.get("gold_tool_calls"))
        gold = canonical_calls(supplied_gold) if supplied_gold is not None else parse_calls(row["target_text"])
        try:
            predicted = parse_calls(row["generated_text"])
            malformed = False
        except (ValueError, TypeError, KeyError):
            predicted, malformed = [], True
        invalid += int(malformed)
        if gold:
            tool_total += 1
            tool_correct += int(not malformed and predicted == gold)
        else:
            non_tool_total += 1
            false_calls += int(malformed or bool(predicted))
        loss = row.get("uniform_ce")
        if not isinstance(loss, (int, float)) or isinstance(loss, bool) or not math.isfinite(loss):
            raise ValueError("Every decision must report finite uniform CE")
        losses.append(float(loss))
    if not tool_total or not non_tool_total:
        raise ValueError("Selection dev must contain tool and non-tool decisions")
    return {
        "records": len(records), "tool_decisions": tool_total,
        "strict_ordered_tool_call_correct": tool_correct,
        "strict_ordered_tool_call_accuracy": tool_correct / tool_total,
        "non_tool_decisions": non_tool_total, "false_tool_calls": false_calls,
        "false_tool_call_rate": false_calls / non_tool_total,
        "malformed_outputs": invalid, "uniform_ce": sum(losses) / len(losses),
        "scope": "Development prefix action proxy; no tool execution or official task success.",
    }


def rank_key(item):
    metric = item["metrics"]
    return (-metric["strict_ordered_tool_call_accuracy"], metric["false_tool_call_rate"],
            metric["uniform_ce"], item["step"])


def choose_candidates(evaluations):
    """Compare fixed protocols within a variant; retain both ratio winners and final."""
    groups = defaultdict(list)
    for entry in evaluations:
        groups[entry["variant"]].append(entry)
    result = {}
    for variant, entries in sorted(groups.items()):
        contracts = {json.dumps(item["contract"], sort_keys=True) for item in entries}
        if len(contracts) != 1:
            raise ValueError(f"Evaluation protocols differ for {variant}")
        if len({item["checkpoint"] for item in entries}) != len(entries):
            raise ValueError(f"Duplicate checkpoint for {variant}")
        rankings = {}
        reasons = defaultdict(list)
        for ratio in (8, 12):
            candidates = []
            reference_ids = None
            for entry in entries:
                rows = [row for row in entry["records"] if row["ratio"] == ratio]
                ids = {row["decision_id"] for row in rows}
                if reference_ids is None:
                    reference_ids = ids
                elif ids != reference_ids:
                    raise ValueError(f"Evaluation IDs differ for {variant} ratio {ratio}")
                candidates.append({"checkpoint": entry["checkpoint"], "step": entry["step"],
                                   "metrics": score_records(rows)})
            candidates.sort(key=rank_key)
            rankings[str(ratio)] = candidates
            reasons[candidates[0]["checkpoint"]].append(f"best_dev_ratio_{ratio}")
        final = max(entries, key=lambda item: item["step"])
        reasons[final["checkpoint"]].append("final_saved_checkpoint")
        result[variant] = {
            "rankings": rankings,
            "return_candidates": [{"checkpoint": path, "reasons": tags} for path, tags in reasons.items()],
            "selection_rule": "Strict ordered call accuracy descending, false-call rate ascending, uniform CE ascending, earlier step.",
            "scope": "Provisional dev selection within this variant; final checkpoint retained for downstream A evaluation.",
        }
    return result
