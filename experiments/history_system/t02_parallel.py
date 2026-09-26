"""One global T02 sampling plan with independent, same-engine live workers."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import t02


DYNAMIC_SOURCE_STARTS_PER_WORKER = 26
DYNAMIC_LIVE_SNAPSHOTS_PER_WORKER = 52
DYNAMIC_SNAPSHOT_HOST_BYTE_CAP_PER_WORKER = 192 * 1024 ** 3


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def dynamic_source_queue(contract: Mapping[str, Any]) -> bool:
    value = contract.get("dynamic_source_queue", False)
    if type(value) is not bool:
        raise ValueError("dynamic_source_queue must be a boolean")
    return value


def dynamic_source_limits(contract: Mapping[str, Any]) -> dict[str, int]:
    """Validate and return the frozen per-worker limits for dynamic collection."""

    if not dynamic_source_queue(contract):
        return {}
    defaults = {
        "max_source_tasks_per_worker": DYNAMIC_SOURCE_STARTS_PER_WORKER,
        "snapshot_cap_per_worker": DYNAMIC_LIVE_SNAPSHOTS_PER_WORKER,
        "snapshot_host_byte_cap_per_worker": DYNAMIC_SNAPSHOT_HOST_BYTE_CAP_PER_WORKER,
    }
    limits = {name: contract.get(name, default) for name, default in defaults.items()}
    for name, value in limits.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        if value > defaults[name]:
            raise ValueError(f"{name} exceeds the supported dynamic worker cap")
    if limits["snapshot_cap_per_worker"] < 2 * limits["max_source_tasks_per_worker"]:
        raise ValueError("snapshot_cap_per_worker cannot retain two states per source task")
    return limits


@contextmanager
def locked_ledger(root: Path):
    root = Path(root)
    with (root / "ledger.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.seek(0, 2)
            if lock.tell() == 0:
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            ledger = read(root / "ledger.json")
            yield ledger
            save(root / "ledger.json", ledger)
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def initialize_ledger(root: Path):
    root = Path(root)
    if (root / "ledger.json").exists():
        raise FileExistsError("Refusing to reset the source execution ledger")
    contract = read(root / "contract.json")
    dynamic = dynamic_source_queue(contract)
    dynamic_limits = dynamic_source_limits(contract)
    source_cap = contract.get("cumulative_source_start_cap", 104)
    if type(source_cap) is not int or source_cap <= 0:
        raise ValueError("Source start cap must be a positive integer")
    seen = set()
    tasks = []
    for worker in contract["workers"]:
        for task_id in worker["task_ids"]:
            if task_id in seen or task_id in contract["excluded_started_task_ids"]:
                raise ValueError("Duplicate or previously started source task")
            seen.add(task_id)
            tasks.append({
                "task_id": task_id,
                "task_index": len(tasks),
                "worker_id": None if dynamic else worker["worker_id"],
                "static_worker_id": worker["worker_id"],
                "status": "pending",
            })
    if len(tasks) + contract["prior_source_starts"] > source_cap:
        raise ValueError("Source budget exceeds the authorized cumulative cap")
    if dynamic and len(tasks) + contract["prior_source_starts"] != source_cap:
        raise ValueError("Dynamic source queue must bind every remaining authorized source start")
    if contract["target_states"] * 3 + contract["prior_complete_branches"] > 360:
        raise ValueError("Branch budget exceeds the authorized cumulative cap")
    save(root / "ledger.json", {"schema": "t02-parallel-source-ledger-v1",
        "dynamic_source_queue": dynamic,
        **dynamic_limits,
        "prior_source_starts": contract["prior_source_starts"],
        "cumulative_source_start_cap": source_cap, "tasks": tasks,
        "candidates": [], "worker_costs": {}})


def claim_task(root: Path, worker_id: str):
    root = Path(root)
    if (root / "abort.json").exists():
        raise RuntimeError("Global collection has been aborted")
    with locked_ledger(root) as ledger:
        if any(row["worker_id"] == worker_id and row["status"] == "started"
               for row in ledger["tasks"]):
            raise RuntimeError("Worker already owns an unfinished source task")
        dynamic = ledger.get("dynamic_source_queue", False)
        worker_starts = sum(
            row["worker_id"] == worker_id and row["status"] != "pending"
            for row in ledger["tasks"]
        )
        worker_snapshots = sum(
            row["worker_id"] == worker_id for row in ledger["candidates"]
        )
        if dynamic and (
            worker_starts >= ledger["max_source_tasks_per_worker"]
            or worker_snapshots + 2 > ledger["snapshot_cap_per_worker"]
        ):
            return None
        for row in ledger["tasks"]:
            eligible = (
                row["status"] == "pending"
                and (dynamic or row["worker_id"] == worker_id)
            )
            if eligible:
                started = sum(item["status"] != "pending" for item in ledger["tasks"])
                if ledger["prior_source_starts"] + started >= ledger.get("cumulative_source_start_cap", 104):
                    raise RuntimeError("Cumulative source start cap reached")
                row.update(
                    status="started",
                    worker_id=worker_id,
                    started_at_epoch=time.time(),
                )
                return {"task_id": row["task_id"], "task_index": row["task_index"]}
    return None


def dynamic_worker_no_work(root: Path, worker_id: str) -> bool:
    """Return true only after a dynamic queue drained without leasing this worker."""

    root = Path(root)
    contract = read(root / "contract.json")
    if not dynamic_source_queue(contract) or not (root / "ledger.json").exists():
        return False
    with locked_ledger(root) as ledger:
        return (
            all(row["status"] == "completed" for row in ledger["tasks"])
            and not any(row["worker_id"] == worker_id for row in ledger["tasks"])
        )


def source_participants(root: Path) -> list[str]:
    """Return frozen workers that actually own at least one source lease."""

    root = Path(root)
    contract = read(root / "contract.json")
    frozen = [row["worker_id"] for row in contract["workers"]]
    if not dynamic_source_queue(contract):
        return frozen
    ledger = read(root / "ledger.json")
    if any(row["status"] != "completed" for row in ledger["tasks"]):
        raise RuntimeError("Dynamic source queue has not completed")
    owners = {row["worker_id"] for row in ledger["tasks"]}
    return [worker_id for worker_id in frozen if worker_id in owners]


def isolated_branch_failure(root: Path, worker_id: str) -> dict[str, Any] | None:
    """Return an auditable post-pilot worker failure that must stay local."""

    path = Path(root) / "workers" / worker_id / "status.json"
    if not path.exists():
        return None
    value = read(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"worker status must be an object: {worker_id}")
    if (value.get("schema") == "t02-parallel-worker-status-v1"
            and value.get("phase") == "failed"
            and value.get("failure_scope") == "branch_execution"
            and value.get("global_abort_required") is False):
        return dict(value)
    return None


def complete_task(root: Path, worker_id: str, task_id: str, candidates,
                  cost_summary: Mapping[str, Any], *,
                  source_terminal: Mapping[str, Any] | None = None):
    with locked_ledger(Path(root)) as ledger:
        rows = [row for row in ledger["tasks"] if row["task_id"] == task_id]
        if len(rows) != 1 or rows[0]["worker_id"] != worker_id or rows[0]["status"] != "started":
            raise RuntimeError("Task completion does not match its exclusive source lease")
        if len(candidates) > 2:
            raise ValueError("A source variant exceeds its two-state cap")
        if ledger.get("dynamic_source_queue", False):
            retained = sum(
                row["worker_id"] == worker_id for row in ledger["candidates"]
            )
            if retained + len(candidates) > ledger["snapshot_cap_per_worker"]:
                raise RuntimeError("Worker snapshot cap would be exceeded")
        ids = {row["state"]["state_id"] for row in ledger["candidates"]}
        for candidate in candidates:
            if candidate["task_id"] != task_id or candidate["state_id"] in ids:
                raise ValueError("Candidate task identity differs or state ID repeats")
            ids.add(candidate["state_id"])
            ledger["candidates"].append({"worker_id": worker_id, "state": candidate})
        if source_terminal is not None:
            if (source_terminal.get("schema") != "t02-bfcl-capacity-termination-v1"
                    or source_terminal.get("reason") != "capacity_infeasible"
                    or source_terminal.get("terminal_status") != "budget_terminated"
                    or source_terminal.get("response_fabricated") is not False):
                raise ValueError("Source terminal is not an auditable capacity failure")
        rows[0].update(status="completed", candidate_count=len(candidates),
                       finished_at_epoch=time.time(),
                       source_terminal=(dict(source_terminal)
                                        if source_terminal is not None else None))
        ledger["worker_costs"][worker_id] = dict(cost_summary)


def make_global_plan(root: Path):
    from t02_bfcl import validate_training_feature_contract
    root = Path(root)
    contract = read(root / "contract.json")
    ledger = read(root / "ledger.json")
    if any(row["status"] != "completed" for row in ledger["tasks"]):
        raise RuntimeError("Sources must complete before global state selection")
    args = contract["common_args"]
    candidates = [row["state"] for row in ledger["candidates"]]
    plan = t02.build_plan(candidates,
        forbidden_manifest_paths={"D128": args["d128_manifest"], "F128": args["f128_manifest"]},
        frozen_subsequent_policy=contract["frozen_policy"], seed=args["seed"],
        target_states=contract["target_states"], train_states=contract["train_states"],
        min_task_groups=contract["min_task_groups"],
        max_states_per_task_group=contract["max_states_per_group"],
        max_states_per_task=args["max_states_per_task"],
        max_complete_branch_executions=contract["remaining_branch_budget"])
    plan["training_feature_contract"] = validate_training_feature_contract(plan["states"])
    owners = {row["state"]["state_id"]: row["worker_id"] for row in ledger["candidates"]}
    save(root / "state_owners.json", {state["state_id"]: owners[state["state_id"]]
                                      for state in plan["states"]})
    save(root / "global_plan.json", plan)
    return plan


def merge_results(plan, partials, owners):
    """Verify a complete partition before labels can see any partial result."""
    t02.check_plan(plan)
    expected = {state["state_id"] for state in plan["states"]}
    if set(owners) != expected:
        raise ValueError("State ownership does not cover the global plan")
    results, snapshots, seen, seen_snapshots = [], [], set(), set()
    capabilities = None
    for worker_id, partial in partials.items():
        if partial.get("plan_sha256") != t02._digest(plan):
            raise ValueError("Worker result refers to a different global plan")
        bound_ids = {state_id for state_id, owner in owners.items() if owner == worker_id}
        if set(partial.get("executed_state_ids", [])) != bound_ids:
            raise ValueError("Worker result state ownership differs")
        if partial["complete_branch_executions"] != 3 * len(bound_ids):
            raise ValueError("Worker did not complete all three branches for every owned state")
        current = t02._check_capabilities(partial["adapter_capabilities"])
        if capabilities is not None and current != capabilities:
            raise ValueError("Worker capability contracts differ")
        capabilities = current
        for row in partial["snapshots"]:
            if row["state_id"] not in bound_ids or row["state_id"] in seen_snapshots:
                raise ValueError("Snapshot ownership differs or repeats")
            seen_snapshots.add(row["state_id"])
            snapshots.append(row)
        for row in partial["results"]:
            key = (row["state_id"], row["branch_id"])
            if row["state_id"] not in bound_ids or key in seen:
                raise ValueError("Branch belongs to another worker or repeats")
            seen.add(key)
            results.append(row)
    required = {(state_id, branch) for state_id in expected for branch in t02.BRANCH_IDS}
    if seen != required or seen_snapshots != expected:
        raise ValueError("Global results have missing states or branches")
    merged = {"schema": t02.RESULT_SET_SCHEMA, "plan_sha256": t02._digest(plan),
        "adapter_capabilities": capabilities, "complete_branch_executions": len(results),
        "branch_cap": plan["max_complete_branch_executions"],
        "authorized_complete_branch_cap": plan["authorized_complete_branch_cap"],
        "snapshots": snapshots, "results": results}
    t02._index_results(merged, plan)
    return merged


def validate_worker_artifact(plan: Mapping[str, Any], artifact: Mapping[str, Any],
                             owned_state_ids: set[str], *,
                             require_complete: bool) -> int:
    """Validate a worker artifact without treating an incomplete run as complete."""

    if (artifact.get("plan_sha256") != t02._digest(plan)
            or artifact.get("partial_worker_result") is not True):
        raise ValueError("Worker artifact differs from the global plan contract")
    state_ids = artifact.get("executed_state_ids")
    if (not isinstance(state_ids, list) or len(state_ids) != len(set(state_ids))
            or not set(state_ids).issubset(owned_state_ids)):
        raise ValueError("Worker artifact state ownership differs")
    if require_complete and set(state_ids) != owned_state_ids:
        raise ValueError("Completed worker artifact omits an owned state")
    snapshots = artifact.get("snapshots")
    results = artifact.get("results")
    if not isinstance(snapshots, list) or not isinstance(results, list):
        raise ValueError("Worker artifact snapshots and results must be lists")
    snapshot_ids = [row.get("state_id") for row in snapshots if isinstance(row, Mapping)]
    if (len(snapshot_ids) != len(snapshots)
            or len(snapshot_ids) != len(set(snapshot_ids))
            or set(snapshot_ids) != set(state_ids)):
        raise ValueError("Worker artifact snapshots differ from its started states")
    result_keys = [
        (row.get("state_id"), row.get("branch_id"))
        for row in results if isinstance(row, Mapping)
    ]
    if (len(result_keys) != len(results) or len(result_keys) != len(set(result_keys))
            or any(state_id not in set(state_ids) or branch not in t02.BRANCH_IDS
                   for state_id, branch in result_keys)
            or artifact.get("complete_branch_executions") != len(results)):
        raise ValueError("Worker artifact branch results are invalid")
    required = {(state_id, branch) for state_id in state_ids for branch in t02.BRANCH_IDS}
    if require_complete and set(result_keys) != required:
        raise ValueError("Completed worker artifact omits an official branch")
    return len(results)


def coordinate(root: Path):
    root = Path(root)
    contract = read(root / "contract.json")
    worker_ids = [row["worker_id"] for row in contract["workers"]]
    try:
        if dynamic_source_queue(contract):
            while True:
                if (root / "abort.json").exists():
                    raise RuntimeError("Worker failed during collection")
                ledger = read(root / "ledger.json")
                if all(row["status"] == "completed" for row in ledger["tasks"]):
                    break
                time.sleep(5)
            worker_ids = source_participants(root)
        while not all((root / "workers" / worker / "ready.json").exists()
                      for worker in worker_ids):
            if (root / "abort.json").exists():
                raise RuntimeError("Worker failed during collection")
            time.sleep(5)
        plan = make_global_plan(root)
        save(root / "status.json", {"phase": "branching", "state_count": len(plan["states"]),
                                    "updated_at_epoch": time.time()})
        isolated_failures: dict[str, dict[str, Any]] = {}
        while True:
            if (root / "abort.json").exists():
                raise RuntimeError("Worker failed during branch execution")
            isolated_failures = {
                worker: failure
                for worker in worker_ids
                if not (root / "workers" / worker / "results.json").exists()
                and (failure := isolated_branch_failure(root, worker)) is not None
            }
            terminal_workers = {
                worker for worker in worker_ids
                if (root / "workers" / worker / "results.json").exists()
                or worker in isolated_failures
            }
            if terminal_workers == set(worker_ids):
                break
            time.sleep(5)
        if isolated_failures:
            owners = read(root / "state_owners.json")
            expected_state_ids = {state["state_id"] for state in plan["states"]}
            if (not isinstance(owners, Mapping) or set(owners) != expected_state_ids
                    or not set(owners.values()).issubset(set(worker_ids))):
                raise ValueError("State ownership does not cover the partial global plan")
            completed_workers = [
                worker for worker in worker_ids
                if (root / "workers" / worker / "results.json").exists()
            ]
            worker_artifacts = {}
            observed_branch_executions = 0
            for worker in worker_ids:
                worker_root = root / "workers" / worker
                result_path = worker_root / "results.json"
                partial_path = worker_root / "partial_results.json"
                owned_ids = {
                    state_id for state_id, owner in owners.items() if owner == worker
                }
                artifact_path = result_path if result_path.exists() else partial_path
                branch_executions = 0
                if artifact_path.exists():
                    branch_executions = validate_worker_artifact(
                        plan, read(artifact_path), owned_ids,
                        require_complete=result_path.exists(),
                    )
                    observed_branch_executions += branch_executions
                worker_artifacts[worker] = {
                    "status": "failed" if worker in isolated_failures else "completed",
                    "results": str(result_path) if result_path.exists() else None,
                    "partial_results": str(partial_path) if partial_path.exists() else None,
                    "observed_branch_executions": branch_executions,
                    "failure": isolated_failures.get(worker),
                }
            save(root / "run/plan.json", plan)
            save(root / "run/summary.json", {
                "status": "partial_failed",
                "state_count": len(plan["states"]),
                "worker_count": len(worker_ids),
                "completed_worker_ids": completed_workers,
                "failed_worker_ids": sorted(isolated_failures),
                "worker_artifacts": worker_artifacts,
                "observed_branch_executions": observed_branch_executions,
                "plan_sha256": t02._digest(plan),
                "complete_results_emitted": False,
                "labels_emitted": False,
                "training_allowed": False,
            })
            save(root / "status.json", {
                "phase": "partial_failed",
                "failed_worker_ids": sorted(isolated_failures),
                "completed_worker_ids": completed_workers,
                "training_allowed": False,
                "updated_at_epoch": time.time(),
            })
            return 1
        partials = {worker: read(root / "workers" / worker / "results.json") for worker in worker_ids}
        merged = merge_results(plan, partials, read(root / "state_owners.json"))
        labels = t02.label_results(plan, merged)
        save(root / "run/results.json", merged)
        save(root / "run/labels.json", labels)
        save(root / "run/plan.json", plan)
        from collections import Counter
        save(root / "run/summary.json", {"status": "completed", "state_count": len(plan["states"]),
            "complete_branch_executions": merged["complete_branch_executions"],
            "cumulative_complete_branch_executions": contract["prior_complete_branches"] + len(merged["results"]),
            "actual_split_counts": dict(Counter(row["split"] for row in plan["states"])),
            "worker_count": len(worker_ids), "plan_sha256": t02._digest(plan)})
        save(root / "status.json", {"phase": "labels_completed", "updated_at_epoch": time.time()})
        return 0
    except Exception as error:
        save(root / "abort.json", {"phase": "coordinator", "error_type": type(error).__name__,
                                   "error": str(error), "updated_at_epoch": time.time()})
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["initialize", "coordinate"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "initialize":
        initialize_ledger(args.root)
    else:
        raise SystemExit(coordinate(args.root))
