"""Per-decision intervention receipts and per-task cost for every ablation cell.

Intervention stages are kept apart: eligibility (gate reached), trigger, repack
feasible, regeneration completed, regenerated action committed, and correction
proposed / verified / changed. History KV is the assembled history-plus-evidence
KV charged to B immediately before each completed generation (draft and
regeneration), the quantity the main table averages.

usage: python -m racer_ablation.extract_decisions OUT_JSON LABEL ROOT [ROOT ...]
"""
from __future__ import annotations

import sys

from .common import CELLS, steps, task_rows, turn_of, write_json


def decision_row(record):
    recovery = record.get("exact_recovery") or {}
    gate = recovery.get("gate") or {}
    traces = record.get("generation_trace") or []
    checks = record.get("pre_generation_budget_checks") or []
    validation = record.get("commit_validation") or {}
    transform = record.get("commit_transform") or {}
    binding = recovery.get("verified_binding") or {}
    regenerations = [trace for trace in traces if trace.get("phase") == "regeneration"]
    selected = validation.get("selected_generation_index")
    history_tokens = []
    for index, trace in enumerate(traces):
        if trace.get("status") == "completed" and index < len(checks):
            check = checks[index]
            history_tokens.append(check["active_history_bytes"] // check["kv_bytes_per_token"])
    usage = record.get("generation_usage_total") or {}
    self_revision = recovery.get("self_revision") or {}
    return {
        "decision_key": record["decision_key"], "turn": turn_of(record["decision_key"]),
        "status": record.get("status"), "failure_code": record.get("failure_code"),
        "reason": recovery.get("reason"), "recovery_status": recovery.get("status"),
        "risk_score": gate.get("score"), "gate_type": gate.get("type"),
        "triggered": gate.get("triggered") is True,
        "risk_triggered": gate.get("risk_triggered", gate.get("triggered")) is True,
        "generation_limit": recovery.get("reason") == "shared_task_generation_limit",
        "repack_feasible": recovery.get("status") == "recover",
        "no_feasible_candidate": recovery.get("reason") == "no_feasible_new_complete_event",
        "self_revision_context_limit": recovery.get("reason") == "self_revision_context_limit",
        "regeneration_attempted": bool(regenerations),
        "regeneration_completed": any(trace.get("status") == "completed" for trace in regenerations),
        "regenerated_action_committed": (validation.get("accepted") is True
                                         and isinstance(selected, int) and selected > 0),
        "parse_fallback": (validation.get("accepted") is False
                           and validation.get("fallback") == "original" and bool(regenerations)),
        "correction_proposed": binding.get("status") == "proposed",
        "correction_verified": transform.get("status") == "verified_binding_committed",
        "correction_changed": transform.get("changed") is True,
        "self_revision_added_tokens": self_revision.get("added_live_tokens"),
        "probe_target": (recovery.get("probe") or {}).get("target") is True,
        "draft_in_retrieval_query": (recovery.get("source") or {}).get("draft_in_retrieval_query"),
        "history_kv_tokens": history_tokens,
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
        "generations_completed": record.get("generation_completed"),
    }


def extract(roots, cell):
    rows, failures = task_rows(roots, cell)
    tasks = {}
    for task, row in sorted(rows.items()):
        metrics = row.get("unified_metrics") or {}
        decisions = [decision_row(record) for record in steps(row)]
        history = [value for decision in decisions for value in decision["history_kv_tokens"]]
        tasks[task] = {
            "status": row.get("status"), "official_score": metrics.get("official_score"),
            "correct": (metrics.get("official_score") or 0) >= 1.0,
            "method_failure": task in failures["method"], "harness_failure": task in failures["harness"],
            "decisions": decisions, "decision_count": len(decisions),
            "generation_calls": metrics.get("generation_calls"),
            "generations_completed": sum(d["generations_completed"] or 0 for d in decisions),
            "history_kv_tokens_sum": sum(history), "history_kv_generations": len(history),
            "unified_active_history_kv": metrics.get("active_history_kv"),
            "prompt_tokens": sum(d["prompt_tokens"] or 0 for d in decisions),
            "completion_tokens": sum(d["completion_tokens"] or 0 for d in decisions),
            "generation_prefill_tokens": metrics.get("generation_prefill_tokens"),
            "recovery_prefill_tokens": metrics.get("recovery_prefill_tokens"),
            "wall_time": metrics.get("wall_time"),
        }
    return {"cell": cell, "roots": list(roots), "tasks": tasks,
            "failures": {key: sorted(value) for key, value in failures.items()}}


def main(argv):
    out, label, roots = argv[0], argv[1], argv[2:]
    write_json(out, extract(roots, CELLS[label]))


if __name__ == "__main__":
    main(sys.argv[1:])
