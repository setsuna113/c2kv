"""Validate a native server's durable, task-bound budget rejection."""
from __future__ import annotations

import json
from pathlib import Path


NATIVE_BUDGET_CODES = frozenset({"decision_cap_reached", "generation_cap_reached"})


def native_budget_failure(server_dir: Path, task_id: str) -> str | None:
    """Recover a stripped transport code without inferring failure from counters alone."""
    try:
        ready = json.loads((server_dir / "ready.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in
                (server_dir / "budget_rejections.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (OSError, ValueError):
        return None
    if (not isinstance(ready, dict) or not rows or not isinstance(rows[-1], dict)
            or ready.get("schema") != "a-event-native-server-v1"
            or ready.get("status") != "ready" or ready.get("benchmark") != "tau2"
            or ready.get("allowed_task_ids") != [task_id]
            or not isinstance(ready.get("run_id"), str) or not ready["run_id"]):
        return None
    row = rows[-1]
    code = row.get("code")
    if (not isinstance(code, str) or code not in NATIVE_BUDGET_CODES
            or row.get("schema") != "a-event-native-budget-rejection-v1"
            or row.get("run_id") != ready["run_id"]
            or row.get("task_id") != task_id
            or row.get("session_id") != f"tau2/{task_id}/attempt-0"
            or row.get("status_code") != 429):
        return None
    cap_key, used_key = (("max_decisions", "decisions_reserved")
                         if code == "decision_cap_reached" else
                         ("max_generation_calls", "generation_calls_reserved"))
    cap, used = ready.get(cap_key), row.get(used_key)
    if (type(cap) is not int or cap <= 0 or type(used) is not int
            or used != cap or row.get(cap_key) != cap):
        return None
    return code
