"""Bounded per-source T02 recovery with truthful multi-plan provenance.

The v7 recovery retained every live snapshot until all source collection had
finished.  This module makes one source task the persistence boundary: select
its frozen slots, execute A0/A1/A2 immediately, persist after each branch, and
only then claim another source.  A source task plan is deliberately distinct
from the old global sampling plan and keeps its own digest through results,
labels, and the final training dataset.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import t02
from t02_parallel import locked_ledger, read, save
from t02_recovery_plan import (
    COMBINED_LABEL_SCHEMA,
    _validate_source_plan,
    validate_recovery_contract,
)


LEDGER_SCHEMA = "t02-streaming-recovery-ledger-v1"
TASK_PLAN_SCHEMA = "t02-streaming-task-plan-v1"
TASK_STATUS_SCHEMA = "t02-streaming-task-status-v1"
PILOT_PASSED_SCHEMA = "t02-streaming-pilot-passed-v1"
PILOT_FAILED_SCHEMA = "t02-streaming-pilot-failed-v1"
SUMMARY_SCHEMA = "t02-streaming-recovery-summary-v1"
TERMINAL_TASK_STATUSES = {
    "completed",
    "failed_no_retry",
    "not_started_pilot_failed",
    "not_started_workers_exhausted",
}


class PilotGateFailed(RuntimeError):
    """The deterministic pilot did not complete its first three branches."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return hashlib.sha256(payload).hexdigest()


def _task_output_name(task_index: int, task_id: str) -> str:
    safe = "".join(character if character.isalnum() or character in "-_" else "_"
                   for character in task_id)
    return f"{task_index:03d}-{safe}"


def _recovery_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parallel = read(root / "contract.json")
    recovery = read(root / "recovery" / "recovery_contract.json")
    source_plan = read(root / "recovery" / "source_plan.json")
    validate_recovery_contract(recovery)
    _validate_source_plan(source_plan)
    if recovery["source_plan"]["sha256"] != t02._digest(source_plan):
        raise ValueError("recovery contract source plan digest changed")
    if parallel.get("recovery_contract_path") != "recovery/recovery_contract.json":
        raise ValueError("parallel contract does not bind the recovery contract")
    if parallel.get("dynamic_source_queue") is not True:
        raise ValueError("streaming recovery requires the frozen dynamic source queue")
    if parallel.get("prior_complete_branches") != recovery["budget"]["complete_branches_before_recovery"]:
        raise ValueError("parallel and recovery prior branch accounting differ")
    if parallel.get("remaining_branch_budget") != recovery["budget"]["new_complete_branch_executions"]:
        raise ValueError("parallel and recovery new branch budgets differ")
    return parallel, recovery, source_plan


def initialize_ledger(root: str | Path) -> dict[str, Any]:
    """Create the exact-once 83-source queue without starting any source."""

    root = Path(root).resolve()
    if (root / "ledger.json").exists():
        raise FileExistsError("Refusing to reset the streaming recovery ledger")
    parallel, recovery, _ = _recovery_inputs(root)
    source_tasks = recovery["source_tasks"]
    frozen_task_ids = [row["task_id"] for row in source_tasks]
    if len(frozen_task_ids) != len(set(frozen_task_ids)):
        raise ValueError("recovery source task order repeats a task")
    worker_tasks = [
        task_id
        for worker in parallel.get("workers", [])
        for task_id in worker.get("task_ids", [])
    ]
    if set(worker_tasks) != set(frozen_task_ids) or len(worker_tasks) != len(frozen_task_ids):
        raise ValueError("parallel worker shards differ from the frozen recovery source queue")
    if any(not 1 <= len(row["expected_slots"]) <= 2 for row in source_tasks):
        raise ValueError("every source task must contain one or two frozen slots")
    expected_states = sum(len(row["expected_slots"]) for row in source_tasks)
    budget = recovery["budget"]
    if expected_states != recovery["target"]["new_exact_state_count"]:
        raise ValueError("source task slots differ from the recovery target")
    if expected_states * len(t02.BRANCH_IDS) != budget["new_complete_branch_executions"]:
        raise ValueError("source task slots differ from the frozen branch budget")
    if budget["cumulative_complete_branches"] > budget["complete_branch_cap"]:
        raise ValueError("streaming recovery exceeds the cumulative branch cap")
    prior_starts = parallel.get("prior_source_starts")
    if prior_starts + len(source_tasks) != parallel.get("cumulative_source_start_cap"):
        raise ValueError("streaming queue differs from the cumulative source cap")
    ledger = {
        "schema": LEDGER_SCHEMA,
        "pilot_task_id": frozen_task_ids[0],
        "prior_source_starts": prior_starts,
        "cumulative_source_start_cap": parallel["cumulative_source_start_cap"],
        "complete_branches_before_recovery": budget["complete_branches_before_recovery"],
        "new_complete_branch_cap": budget["new_complete_branch_executions"],
        "cumulative_complete_branch_cap": budget["complete_branch_cap"],
        "tasks": [
            {
                "task_id": row["task_id"],
                "task_index": index,
                "expected_slots": copy.deepcopy(row["expected_slots"]),
                "expected_slot_count": len(row["expected_slots"]),
                "status": "pending",
                "worker_id": None,
                "complete_branch_executions": 0,
                "plan_path": None,
                "plan_sha256": None,
                "plan_file_sha256": None,
                "result_path": None,
                "result_file_sha256": None,
                "label_path": None,
                "label_file_sha256": None,
                "failure": None,
            }
            for index, row in enumerate(source_tasks)
        ],
        "worker_source_starts": {},
    }
    save(root / "ledger.json", ledger)
    return ledger


def claim_task(root: str | Path, worker_id: str) -> dict[str, Any] | None:
    """Lease the next source exactly once; failed leases are never reassigned."""

    root = Path(root)
    parallel = read(root / "contract.json")
    maximum = parallel.get("max_source_tasks_per_worker", 26)
    with locked_ledger(root) as ledger:
        if ledger.get("schema") != LEDGER_SCHEMA:
            raise ValueError("streaming recovery ledger schema changed")
        if any(row["worker_id"] == worker_id and row["status"] == "started"
               for row in ledger["tasks"]):
            raise RuntimeError("worker already owns an unfinished streaming source")
        starts = ledger["worker_source_starts"].get(worker_id, 0)
        if starts >= maximum:
            return None
        for row in ledger["tasks"]:
            if row["status"] != "pending":
                continue
            total_started = sum(item["status"] != "pending" for item in ledger["tasks"])
            if ledger["prior_source_starts"] + total_started >= ledger["cumulative_source_start_cap"]:
                raise RuntimeError("cumulative source start cap reached")
            row.update(
                status="started",
                worker_id=worker_id,
                started_at_epoch=time.time(),
            )
            ledger["worker_source_starts"][worker_id] = starts + 1
            return copy.deepcopy(row)
    return None


def _task_row(ledger: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    matches = [row for row in ledger["tasks"] if row["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"streaming ledger must contain task exactly once: {task_id}")
    return matches[0]


def complete_task(root: str | Path, worker_id: str, task_id: str, *,
                  plan_path: Path, plan_sha256: str, plan_file_sha256: str,
                  result_path: Path, result_file_sha256: str,
                  label_path: Path, label_file_sha256: str,
                  complete_branch_executions: int) -> None:
    root = Path(root)
    with locked_ledger(root) as ledger:
        row = _task_row(ledger, task_id)
        if row["status"] != "started" or row["worker_id"] != worker_id:
            raise RuntimeError("streaming task completion does not match its exclusive lease")
        expected = row["expected_slot_count"] * len(t02.BRANCH_IDS)
        if complete_branch_executions != expected:
            raise ValueError("streaming task did not complete three branches per frozen slot")
        completed_elsewhere = sum(
            item["complete_branch_executions"] for item in ledger["tasks"]
            if item is not row
        )
        if completed_elsewhere + complete_branch_executions > ledger["new_complete_branch_cap"]:
            raise ValueError("streaming task would exceed the frozen new-branch cap")
        row.update(
            status="completed",
            plan_path=str(plan_path),
            plan_sha256=plan_sha256,
            plan_file_sha256=plan_file_sha256,
            result_path=str(result_path),
            result_file_sha256=result_file_sha256,
            label_path=str(label_path),
            label_file_sha256=label_file_sha256,
            complete_branch_executions=complete_branch_executions,
            finished_at_epoch=time.time(),
        )


def fail_task(root: str | Path, worker_id: str, task_id: str, *,
              error: BaseException, complete_branch_executions: int,
              plan_sha256: str | None = None,
              partial_result_path: Path | None = None) -> None:
    """Make a source failure terminal while retaining its completed branches."""

    root = Path(root)
    with locked_ledger(root) as ledger:
        row = _task_row(ledger, task_id)
        if row["status"] != "started" or row["worker_id"] != worker_id:
            raise RuntimeError("streaming task failure does not match its exclusive lease")
        expected = row["expected_slot_count"] * len(t02.BRANCH_IDS)
        if type(complete_branch_executions) is not int or not 0 <= complete_branch_executions <= expected:
            raise ValueError("failed task complete branch count is invalid")
        row.update(
            status="failed_no_retry",
            plan_sha256=plan_sha256,
            result_path=str(partial_result_path) if partial_result_path else None,
            complete_branch_executions=complete_branch_executions,
            failure={"error_type": type(error).__name__, "error": str(error)},
            finished_at_epoch=time.time(),
        )


def mark_pilot_failed(root: str | Path, worker_id: str, task_id: str,
                      error: BaseException) -> None:
    """Record the failed gate and stop unstarted sources without charging starts."""

    root = Path(root)
    marker = {
        "schema": PILOT_FAILED_SCHEMA,
        "pilot_task_id": task_id,
        "worker_id": worker_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "passed": False,
        "updated_at_epoch": time.time(),
    }
    if not (root / "pilot_failed.json").exists():
        _atomic_write(root / "pilot_failed.json", marker)
    with locked_ledger(root) as ledger:
        if task_id != ledger["pilot_task_id"]:
            raise ValueError("pilot failure refers to a non-pilot source")
        for row in ledger["tasks"]:
            if (row["task_id"] == task_id and row["status"] == "started"
                    and row.get("worker_id") == worker_id):
                task_root = (
                    root / "workers" / worker_id / "tasks"
                    / _task_output_name(row["task_index"], row["task_id"])
                )
                partial_path = task_root / "partial_results.json"
                observed = 0
                if partial_path.exists():
                    try:
                        durable = read(partial_path).get("complete_branch_executions")
                        maximum = row["expected_slot_count"] * len(t02.BRANCH_IDS)
                        if type(durable) is int and 0 <= durable <= maximum:
                            observed = durable
                    except (OSError, json.JSONDecodeError):
                        pass
                row.update(
                    status="failed_no_retry",
                    complete_branch_executions=observed,
                    result_path=(str(partial_path) if partial_path.exists() else None),
                    result_file_sha256=(
                        hashlib.sha256(partial_path.read_bytes()).hexdigest()
                        if partial_path.exists() else None
                    ),
                    failure={"error_type": type(error).__name__, "error": str(error)},
                    finished_at_epoch=time.time(),
                )
            elif row["status"] == "pending":
                row.update(
                    status="not_started_pilot_failed",
                    failure={
                        "error_type": "PilotGateFailed",
                        "error": f"pilot source failed before this source started: {task_id}",
                    },
                    finished_at_epoch=time.time(),
                )


def mark_orphaned_and_unstarted(root: str | Path, worker_ids: Sequence[str]) -> None:
    """Close the queue after every worker slot is terminal, without reassigning work."""

    root = Path(root)
    worker_set = set(worker_ids)
    with locked_ledger(root) as ledger:
        for row in ledger["tasks"]:
            if row["status"] == "started" and row.get("worker_id") in worker_set:
                task_root = (
                    root / "workers" / row["worker_id"] / "tasks"
                    / _task_output_name(row["task_index"], row["task_id"])
                )
                result_path = task_root / "results.json"
                partial_path = task_root / "partial_results.json"
                artifact_path = result_path if result_path.exists() else partial_path
                observed = row["complete_branch_executions"]
                if artifact_path.exists():
                    try:
                        artifact = read(artifact_path)
                        durable = artifact.get("complete_branch_executions")
                        maximum = row["expected_slot_count"] * len(t02.BRANCH_IDS)
                        if type(durable) is not int or not 0 <= durable <= maximum:
                            raise ValueError("orphaned task branch count is invalid")
                        observed = max(observed, durable)
                    except (OSError, json.JSONDecodeError, ValueError):
                        pass
                plan_path = task_root / "plan.json"
                row.update(
                    status="failed_no_retry",
                    complete_branch_executions=observed,
                    plan_path=str(plan_path) if plan_path.exists() else row.get("plan_path"),
                    plan_sha256=(
                        _digest(read(plan_path)) if plan_path.exists() else row.get("plan_sha256")
                    ),
                    plan_file_sha256=(
                        hashlib.sha256(plan_path.read_bytes()).hexdigest()
                        if plan_path.exists() else row.get("plan_file_sha256")
                    ),
                    result_path=(str(artifact_path) if artifact_path.exists()
                                 else row.get("result_path")),
                    result_file_sha256=(
                        hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                        if artifact_path.exists() else row.get("result_file_sha256")
                    ),
                    failure={
                        "error_type": "WorkerExited",
                        "error": "owning worker exited before recording a task terminal",
                    },
                    finished_at_epoch=time.time(),
                )
            elif row["status"] == "pending":
                row.update(
                    status="not_started_workers_exhausted",
                    failure={
                        "error_type": "WorkersExhausted",
                        "error": "all frozen workers exited before this source was claimed",
                    },
                    finished_at_epoch=time.time(),
                )


def expected_slots(recovery: Mapping[str, Any], task_id: str) -> list[dict[str, Any]]:
    matches = [row for row in recovery["source_tasks"] if row["task_id"] == task_id]
    if len(matches) != 1:
        raise ValueError(f"recovery contract must contain task exactly once: {task_id}")
    return copy.deepcopy(matches[0]["expected_slots"])


def build_task_plan(
    recovery: Mapping[str, Any],
    task_id: str,
    candidate_states: Sequence[Mapping[str, Any]],
    *,
    source_plan: Mapping[str, Any],
    frozen_subsequent_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one live source's states to its frozen ``(task_id, draft_kind)`` slots."""

    validate_recovery_contract(recovery)
    _validate_source_plan(source_plan)
    if recovery["source_plan"]["sha256"] != t02._digest(source_plan):
        raise ValueError("source plan digest differs from the recovery contract")
    slots = expected_slots(recovery, task_id)
    by_kind: dict[str, list[Mapping[str, Any]]] = {"call": [], "stop": []}
    for candidate in candidate_states:
        if candidate.get("task_id") != task_id:
            raise ValueError("candidate belongs to another source task")
        kind = candidate.get("draft_kind")
        if kind in by_kind:
            by_kind[kind].append(candidate)
    planned = []
    for slot in slots:
        matches = by_kind[slot["draft_kind"]]
        if len(matches) != 1:
            raise ValueError(
                f"missing or ambiguous recovery slot: {(task_id, slot['draft_kind'])}"
            )
        state = t02._normalize_candidate_state(matches[0])
        if state["task_group_id"] != slot["task_group_id"]:
            raise ValueError("recollected task slot changed its task group")
        row = dict(state)
        row.update(
            schema=t02.PLANNED_STATE_SCHEMA,
            split=slot["split"],
            branches=list(t02._branch_actions(state, seed=source_plan["seed"])),
            branch_selection_uses_future_outcome=False,
        )
        planned.append(row)
    policy = json.loads(_canonical(frozen_subsequent_policy).decode("utf-8"))
    plan = {
        "schema": TASK_PLAN_SCHEMA,
        "task_id": task_id,
        "seed": source_plan["seed"],
        "target_states": len(planned),
        "branches_per_state": len(t02.BRANCH_IDS),
        "max_complete_branch_executions": len(planned) * len(t02.BRANCH_IDS),
        "authorized_complete_branch_cap": len(planned) * len(t02.BRANCH_IDS),
        "frozen_subsequent_policy": policy,
        "frozen_subsequent_policy_sha256": t02._digest(policy),
        "required_snapshot_components": list(t02.REQUIRED_COMPONENTS),
        "prefix_replay_is_exact_snapshot": False,
        "source_plan_sha256": recovery["source_plan"]["sha256"],
        "recovery_contract_sha256": _digest(recovery),
        "slot_identity": ["task_id", "draft_kind"],
        "expected_slots": slots,
        "states": planned,
    }
    validate_task_plan(plan, recovery=recovery)
    return plan


def validate_task_plan(plan: Mapping[str, Any], *,
                       recovery: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if plan.get("schema") != TASK_PLAN_SCHEMA:
        raise ValueError(f"streaming task plan schema must be {TASK_PLAN_SCHEMA}")
    states = plan.get("states")
    if not isinstance(states, list) or not 1 <= len(states) <= 2:
        raise ValueError("streaming task plan must contain one or two states")
    if plan.get("target_states") != len(states):
        raise ValueError("streaming task plan target state count differs")
    if plan.get("max_complete_branch_executions") != 3 * len(states):
        raise ValueError("streaming task plan must cap exactly three branches per state")
    if plan.get("authorized_complete_branch_cap") != 3 * len(states):
        raise ValueError("streaming task plan authorized cap must equal its exact task cap")
    if tuple(plan.get("required_snapshot_components", ())) != t02.REQUIRED_COMPONENTS:
        raise ValueError("streaming task plan snapshot component contract changed")
    if plan.get("prefix_replay_is_exact_snapshot") is not False:
        raise ValueError("prefix replay cannot be an exact streaming snapshot")
    if t02._digest(plan.get("frozen_subsequent_policy")) != plan.get(
        "frozen_subsequent_policy_sha256"
    ):
        raise ValueError("streaming task frozen policy digest differs")
    task_id = plan.get("task_id")
    expected = plan.get("expected_slots")
    if not isinstance(expected, list) or len(expected) != len(states):
        raise ValueError("streaming task expected slots differ from its states")
    if recovery is not None:
        if plan.get("recovery_contract_sha256") != _digest(recovery):
            raise ValueError("streaming task recovery contract digest differs")
        if expected != expected_slots(recovery, task_id):
            raise ValueError("streaming task slots differ from the frozen recovery order")
    seen_ids: set[str] = set()
    for state, slot in zip(states, expected):
        if state.get("schema") != t02.PLANNED_STATE_SCHEMA:
            raise ValueError("streaming task plan contains a non-planned state")
        if state.get("task_id") != task_id or state.get("draft_kind") != slot.get("draft_kind"):
            raise ValueError("streaming task state differs from its frozen slot")
        if state.get("task_group_id") != slot.get("task_group_id") or state.get("split") != slot.get("split"):
            raise ValueError("streaming task state changed group or split")
        if state["state_id"] in seen_ids:
            raise ValueError("streaming task plan repeats a state")
        seen_ids.add(state["state_id"])
        if [branch.get("branch_id") for branch in state.get("branches", [])] != list(t02.BRANCH_IDS):
            raise ValueError("streaming task state must contain ordered A0/A1/A2")
        candidate = dict(state)
        for name in ("branches", "split", "branch_selection_uses_future_outcome"):
            candidate.pop(name, None)
        candidate["schema"] = t02.CANDIDATE_SCHEMA
        normalized = t02._normalize_candidate_state(candidate)
        if state["branches"] != list(t02._branch_actions(normalized, seed=plan["seed"])):
            raise ValueError("streaming task branch actions differ from the frozen selector")
    return {
        "schema": "t02-streaming-task-plan-check-v1",
        "valid": True,
        "task_id": task_id,
        "state_count": len(states),
        "complete_branch_cap": 3 * len(states),
    }


def execute_task_plan(
    plan: Mapping[str, Any],
    adapter: Any,
    *,
    state_ids: Sequence[str] | None = None,
    on_progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute a task plan with the same exact restore checks as ``t02.execute_plan``."""

    validate_task_plan(plan)
    capabilities = t02._check_capabilities(adapter.capabilities())
    policy = plan["frozen_subsequent_policy"]
    policy_sha256 = plan["frozen_subsequent_policy_sha256"]
    selected = None if state_ids is None else set(state_ids)
    planned_ids = {state["state_id"] for state in plan["states"]}
    if selected is not None and (len(selected) != len(state_ids) or not selected <= planned_ids):
        raise ValueError("streaming execution state IDs repeat or are outside the task plan")
    selected_states = [
        state for state in plan["states"]
        if selected is None or state["state_id"] in selected
    ]
    snapshots: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    def artifact() -> dict[str, Any]:
        return {
            "schema": t02.RESULT_SET_SCHEMA,
            "plan_schema": TASK_PLAN_SCHEMA,
            "plan_sha256": _digest(plan),
            "adapter_capabilities": capabilities,
            "complete_branch_executions": len(results),
            "branch_cap": plan["max_complete_branch_executions"],
            "authorized_complete_branch_cap": plan["authorized_complete_branch_cap"],
            "snapshots": copy.deepcopy(snapshots),
            "results": copy.deepcopy(results),
            "executed_state_ids": [row["state_id"] for row in snapshots],
            "assigned_state_ids": [row["state_id"] for row in selected_states],
            "partial_worker_result": state_ids is not None,
        }

    for state in selected_states:
        snapshot = t02._check_snapshot(
            adapter.capture_state(state, frozen_policy=policy),
            state_id=state["state_id"],
            policy_sha256=policy_sha256,
        )
        snapshots.append(snapshot)
        for branch in state["branches"]:
            if len(results) >= 3 * len(selected_states):
                raise ValueError("per-task branch cap would be exceeded")
            restore = t02._check_restore(
                adapter.restore_state(snapshot, frozen_policy=policy),
                snapshot=snapshot,
            )
            raw = adapter.run_branch(
                state,
                branch,
                frozen_policy=policy,
                restore_receipt=restore,
            )
            result = t02._json_copy(raw, label="streaming branch result")
            expected = {
                "schema": t02.RESULT_SCHEMA,
                "state_id": state["state_id"],
                "branch_id": branch["branch_id"],
                "candidate_ids": branch["candidate_ids"],
                "snapshot_id": snapshot["snapshot_id"],
                "restore_receipt_sha256": t02._digest(restore),
                "frozen_policy_sha256": policy_sha256,
                "execution_status": "complete",
            }
            for name, value in expected.items():
                if result.get(name) != value:
                    raise ValueError(f"streaming branch result {name} is not bound to its task plan")
            result["official_outcome"] = t02._check_official_outcome(
                result.get("official_outcome")
            )
            expected_execution = {
                "submitted_original_draft": branch["submit_original_draft"],
                "recovery_candidate_ids": branch["candidate_ids"],
                "regeneration_count": 1 if branch["regenerate"] else 0,
                "continued_with_frozen_policy": True,
                "observations_replayed_from_other_branch": False,
            }
            if result.get("execution_receipt") != expected_execution:
                raise ValueError("streaming branch execution receipt changed")
            results.append(result)
            if on_progress is not None:
                on_progress(artifact())
    return artifact()


def merge_task_results(plan: Mapping[str, Any], parts: Sequence[Mapping[str, Any]], *,
                       require_complete: bool) -> dict[str, Any]:
    validate_task_plan(plan)
    plan_sha256 = _digest(plan)
    snapshots: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    seen_results: set[tuple[str, str]] = set()
    capabilities = None
    for part in parts:
        if part.get("plan_sha256") != plan_sha256:
            raise ValueError("streaming result part refers to another task plan")
        current = part.get("adapter_capabilities")
        if capabilities is None:
            capabilities = current
        elif capabilities != current:
            raise ValueError("streaming adapter capabilities changed within one source")
        part_states = part.get("executed_state_ids")
        if not isinstance(part_states, list) or seen_states.intersection(part_states):
            raise ValueError("streaming result parts repeat a state")
        seen_states.update(part_states)
        for snapshot in part.get("snapshots", []):
            if snapshot.get("state_id") not in part_states:
                raise ValueError("streaming snapshot is outside its result part")
            snapshots.append(copy.deepcopy(snapshot))
        for result in part.get("results", []):
            key = (result.get("state_id"), result.get("branch_id"))
            if key in seen_results or key[0] not in part_states:
                raise ValueError("streaming result part repeats or misowns a branch")
            seen_results.add(key)
            results.append(copy.deepcopy(result))
    planned = {state["state_id"] for state in plan["states"]}
    if not seen_states <= planned:
        raise ValueError("streaming result contains a state outside its task plan")
    required = {(state_id, branch) for state_id in planned for branch in t02.BRANCH_IDS}
    if require_complete and (seen_states != planned or seen_results != required):
        raise ValueError("streaming task result omits a frozen state or branch")
    return {
        "schema": t02.RESULT_SET_SCHEMA,
        "plan_schema": TASK_PLAN_SCHEMA,
        "plan_sha256": plan_sha256,
        "adapter_capabilities": capabilities,
        "complete_branch_executions": len(results),
        "branch_cap": plan["max_complete_branch_executions"],
        "authorized_complete_branch_cap": plan["authorized_complete_branch_cap"],
        "snapshots": snapshots,
        "results": results,
        "executed_state_ids": [
            state["state_id"] for state in plan["states"] if state["state_id"] in seen_states
        ],
        "partial_worker_result": True,
    }


def label_task_results(plan: Mapping[str, Any], results: Mapping[str, Any]) -> dict[str, Any]:
    """Label only this task's states while preserving its actual plan digest."""

    validate_task_plan(plan)
    indexed = t02._index_results(results, plan)
    required = {
        (state["state_id"], branch)
        for state in plan["states"] for branch in t02.BRANCH_IDS
    }
    if set(indexed) != required:
        raise ValueError("complete streaming labels require all three branches per state")
    rows = []
    for state in plan["states"]:
        outcomes = {
            branch: t02._outcome_or_unknown(indexed, state["state_id"], branch)
            for branch in t02.BRANCH_IDS
        }
        baseline = outcomes["A0"]
        labels: dict[str, Any] = {}
        for branch_id in ("A1", "A2"):
            turn, turn_status = t02._delta(
                outcomes[branch_id]["turn_success"], baseline["turn_success"]
            )
            task, task_status = t02._delta(
                outcomes[branch_id]["task_success"], baseline["task_success"]
            )
            branch = next(row for row in state["branches"] if row["branch_id"] == branch_id)
            labels[branch_id] = {
                "candidate_ids": copy.deepcopy(branch["candidate_ids"]),
                "delta_turn": turn,
                "delta_task": task,
                "turn_label_status": turn_status,
                "task_label_status": task_status,
            }
        tested = {branch["action_id"] for branch in state["branches"]}
        untested = [
            {
                "action_id": action["action_id"],
                "candidate_ids": copy.deepcopy(action["candidate_ids"]),
                "delta_turn": None,
                "delta_task": None,
                "turn_label_status": "unknown",
                "task_label_status": "unknown",
            }
            for action in state["allowed_actions"] if action["action_id"] not in tested
        ]
        a0_turn = baseline["turn_success"]
        rows.append({
            "schema": t02.LABELED_STATE_SCHEMA,
            "state_id": state["state_id"],
            "benchmark": state["benchmark"],
            "task_id": state["task_id"],
            "task_group_id": state["task_group_id"],
            "decision_key": state["decision_key"],
            "split": state["split"],
            "q": copy.deepcopy(state["q"]),
            "draft": copy.deepcopy(state["draft"]),
            "draft_kind": state["draft_kind"],
            "candidates": copy.deepcopy(state["candidates"]),
            "allowed_actions": copy.deepcopy(state["allowed_actions"]),
            "branches": copy.deepcopy(state["branches"]),
            "outcomes": outcomes,
            "labels": labels,
            "untested_actions": untested,
            "c1_risk_label": None if a0_turn is None else 1 - int(a0_turn),
            "c1_label_status": "unknown" if a0_turn is None else "known",
            "c1_label_source": "A0.official_outcome.turn_success",
            "statistical_unit": "state",
            "branches_are_independent_states": False,
            "result_plan_sha256": _digest(plan),
        })
    artifact = {
        "schema": t02.LABELED_SET_SCHEMA,
        "plan_schema": TASK_PLAN_SCHEMA,
        "plan_sha256": _digest(plan),
        "state_count": len(rows),
        "observed_branch_count": len(indexed),
        "rows": rows,
    }
    t02.check_labeled(artifact)
    return artifact


def merge_label_partitions(
    recovery: Mapping[str, Any],
    preserved_labels: Mapping[str, Any],
    task_labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge old triplets and every task plan without inventing a global plan hash."""

    validate_recovery_contract(recovery)
    old_rows = preserved_labels.get("rows")
    if not isinstance(old_rows, list):
        raise ValueError("preserved label partition must contain rows")
    old_by_id = {row.get("state_id"): row for row in old_rows if isinstance(row, Mapping)}
    old_plan_sha256 = preserved_labels.get("plan_sha256")
    selected_old = []
    for item in recovery["preserved_states"]:
        if item["state_id"] not in old_by_id:
            raise ValueError("preserved labels omit an exact completed state")
        row = copy.deepcopy(old_by_id[item["state_id"]])
        row["result_plan_sha256"] = old_plan_sha256
        selected_old.append(row)

    expected_by_task = {
        source["task_id"]: source["expected_slots"]
        for source in recovery["source_tasks"]
    }
    labels_by_task: dict[str, Mapping[str, Any]] = {}
    for artifact in task_labels:
        rows = artifact.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ValueError("streaming task labels must contain rows")
        task_ids = {row.get("task_id") for row in rows if isinstance(row, Mapping)}
        if len(task_ids) != 1:
            raise ValueError("one streaming label partition must contain one source task")
        task_id = next(iter(task_ids))
        if task_id in labels_by_task or task_id not in expected_by_task:
            raise ValueError("streaming label task is repeated or unexpected")
        slots = [(row.get("task_id"), row.get("draft_kind")) for row in rows]
        expected = [(row["task_id"], row["draft_kind"]) for row in expected_by_task[task_id]]
        if slots != expected:
            raise ValueError("streaming task labels differ from frozen slot order")
        plan_sha256 = artifact.get("plan_sha256")
        if any(row.get("result_plan_sha256") != plan_sha256 for row in rows):
            raise ValueError("streaming label row lost its actual task plan hash")
        labels_by_task[task_id] = artifact
    if set(labels_by_task) != set(expected_by_task):
        raise ValueError("streaming labels do not cover all frozen source tasks")

    new_rows = [
        copy.deepcopy(row)
        for source in recovery["source_tasks"]
        for row in labels_by_task[source["task_id"]]["rows"]
    ]
    combined = [*selected_old, *new_rows]
    if len({row["state_id"] for row in combined}) != len(combined):
        raise ValueError("multi-plan labels repeat a state ID")
    bindings = recovery["group_split_bindings"]
    for row in combined:
        if bindings.get(row.get("task_group_id")) != row.get("split"):
            raise ValueError("multi-plan label changed the frozen group split")
    target = recovery["target"]
    if len(combined) != target["combined_exact_state_count"]:
        raise ValueError("multi-plan labels differ from the exact recovery target")
    split_counts = Counter(row["split"] for row in combined)
    if dict(split_counts) != target["split_counts"]:
        raise ValueError("multi-plan labels changed the frozen train/calibration counts")
    if len({row["task_group_id"] for row in combined}) != target["task_group_count"]:
        raise ValueError("multi-plan labels changed the frozen task groups")

    partitions = [{
        "role": "preserved_v6_completed_triplets",
        "plan_sha256": old_plan_sha256,
        "source_artifact_sha256": preserved_labels.get(
            "label_payload_sha256", _digest(preserved_labels)
        ),
        "source_artifact_file_sha256": preserved_labels.get("label_file_sha256"),
        "selected_state_count": len(selected_old),
        "state_ids": [row["state_id"] for row in selected_old],
    }]
    for source in recovery["source_tasks"]:
        artifact = labels_by_task[source["task_id"]]
        partitions.append({
            "role": "streaming_exact_recollection",
            "task_id": source["task_id"],
            "plan_sha256": artifact["plan_sha256"],
            "source_artifact_sha256": artifact.get(
                "label_payload_sha256", _digest(artifact)
            ),
            "plan_file_sha256": artifact.get("plan_file_sha256"),
            "source_artifact_file_sha256": artifact.get("label_file_sha256"),
            "selected_state_count": len(artifact["rows"]),
            "state_ids": [row["state_id"] for row in artifact["rows"]],
        })
    return {
        "schema": COMBINED_LABEL_SCHEMA,
        "state_count": len(combined),
        "rows": combined,
        "partitions": partitions,
        "plan_sha256s": [row["plan_sha256"] for row in partitions],
        "recovery_contract_sha256": _digest(recovery),
        "rows_sha256": _digest(combined),
        "old_result_plan_sha256_rewritten": False,
        "synthetic_global_plan_sha256": None,
    }


def write_pilot_passed(root: str | Path, worker_id: str, plan: Mapping[str, Any],
                       result: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(root)
    first_state = plan["states"][0]
    keys = {(row["state_id"], row["branch_id"]) for row in result["results"]}
    required = {(first_state["state_id"], branch) for branch in t02.BRANCH_IDS}
    if keys != required or result.get("complete_branch_executions") != 3:
        raise ValueError("pilot marker requires exactly the first state's three branches")
    marker = {
        "schema": PILOT_PASSED_SCHEMA,
        "pilot_task_id": plan["task_id"],
        "pilot_state_id": first_state["state_id"],
        "worker_id": worker_id,
        "plan_sha256": _digest(plan),
        "complete_branch_executions": 3,
        "result_sha256": _digest(result),
        "passed": True,
        "updated_at_epoch": time.time(),
    }
    _atomic_write(root / "pilot_passed.json", marker)
    return marker


def wait_for_pilot(root: str | Path, *, poll_seconds: float = 1.0) -> dict[str, Any]:
    root = Path(root)
    while True:
        failed = root / "pilot_failed.json"
        if failed.exists():
            value = read(failed)
            raise PilotGateFailed(value.get("error") or "deterministic pilot failed")
        passed = root / "pilot_passed.json"
        if passed.exists():
            value = read(passed)
            if value.get("schema") != PILOT_PASSED_SCHEMA or value.get("passed") is not True:
                raise ValueError("pilot passed marker is invalid")
            return value
        time.sleep(poll_seconds)


def task_artifact_dir(worker_root: Path, claim: Mapping[str, Any]) -> Path:
    return worker_root / "tasks" / _task_output_name(claim["task_index"], claim["task_id"])


def terminal_summary(root: str | Path) -> dict[str, Any]:
    ledger = read(Path(root) / "ledger.json")
    counts = Counter(row["status"] for row in ledger["tasks"])
    completed_branches = sum(row["complete_branch_executions"] for row in ledger["tasks"])
    return {
        "task_status_counts": dict(counts),
        "new_complete_branch_executions": completed_branches,
        "cumulative_complete_branch_executions": (
            ledger["complete_branches_before_recovery"] + completed_branches
        ),
        "all_tasks_terminal": all(row["status"] in TERMINAL_TASK_STATUSES for row in ledger["tasks"]),
        "all_tasks_completed": all(row["status"] == "completed" for row in ledger["tasks"]),
    }
