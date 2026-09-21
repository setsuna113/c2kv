"""Validate a native server's durable, task-bound budget rejection."""
from __future__ import annotations

import json
from pathlib import Path


NATIVE_BUDGET_CODES = frozenset({"decision_cap_reached", "generation_cap_reached"})
NATIVE_CAPACITY_CODE = "c2kv_capacity_infeasible"


def native_server_task(server_dir: Path, task_id: str, benchmark: str) -> bool:
    """Check the immutable single-task server identity without reading request data."""
    try:
        ready = json.loads((server_dir / "ready.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(
        isinstance(ready, dict)
        and ready.get("schema") == "a-event-native-server-v1"
        and ready.get("status") == "ready"
        and ready.get("benchmark") == benchmark
        and ready.get("allowed_task_ids") == [task_id]
        and isinstance(ready.get("run_id"), str)
        and ready["run_id"]
    )


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


def native_capacity_failure(server_dir: Path, task_id: str,
                            benchmark: str) -> str | None:
    """Validate the controller's last task-bound capacity failure step."""
    try:
        ready = json.loads((server_dir / "ready.json").read_text(encoding="utf-8"))
        rows = [json.loads(line) for line in
                (server_dir / "steps.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (OSError, ValueError):
        return None
    if (not native_server_task(server_dir, task_id, benchmark)
            or not isinstance(ready, dict) or not rows or not isinstance(rows[-1], dict)):
        return None
    row = rows[-1]
    error = row.get("error")
    if (row.get("schema") != "a-event-native-exact-step-v1"
            or row.get("status") != "failed"
            or row.get("session_id") != f"{benchmark}/{task_id}/attempt-0"
            or not isinstance(row.get("decision_key"), str) or not row["decision_key"]
            or not isinstance(row.get("outer_request_id"), str)
            or not row["outer_request_id"]
            or row.get("failure_kind") != "method_failure"
            or row.get("failure_code") != NATIVE_CAPACITY_CODE
            or row.get("response") is not None
            or not isinstance(error, dict)
            or error.get("type") != "CapacityInfeasible"):
        return None
    return NATIVE_CAPACITY_CODE
