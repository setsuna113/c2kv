"""Coordinate the exact-state T02 recollection and two-plan label merge."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import time

import t02
from t02_bfcl import validate_training_feature_contract
from t02_parallel import (
    dynamic_source_queue,
    isolated_branch_failure,
    merge_results,
    read,
    save,
    source_participants,
    validate_worker_artifact,
)
from t02_recovery_plan import build_recovery_plan, merge_label_partitions


def _recovery_paths(root: Path) -> dict[str, Path]:
    return {
        "contract": root / "recovery" / "recovery_contract.json",
        "source_plan": root / "recovery" / "source_plan.json",
        "source_partials": root / "recovery" / "source_partials",
    }


def make_global_plan(root: Path):
    root = Path(root)
    contract = read(root / "contract.json")
    ledger = read(root / "ledger.json")
    if any(row["status"] != "completed" for row in ledger["tasks"]):
        raise RuntimeError("Sources must complete before exact-slot selection")
    paths = _recovery_paths(root)
    recovery_contract = read(paths["contract"])
    source_plan = read(paths["source_plan"])
    candidates = [row["state"] for row in ledger["candidates"]]
    plan = build_recovery_plan(
        recovery_contract,
        candidates,
        source_plan=source_plan,
        frozen_subsequent_policy=contract["frozen_policy"],
    )
    plan["training_feature_contract"] = validate_training_feature_contract(plan["states"])
    owners = {row["state"]["state_id"]: row["worker_id"] for row in ledger["candidates"]}
    selected_owners = {state["state_id"]: owners[state["state_id"]] for state in plan["states"]}
    save(root / "state_owners.json", selected_owners)
    save(root / "global_plan.json", plan)
    return plan


def _preserved_labels(root: Path, recovery_contract, source_plan):
    rows = []
    for path in sorted(_recovery_paths(root)["source_partials"].glob("*.json")):
        artifact = read(path)
        if artifact.get("plan_sha256") != t02._digest(source_plan):
            raise ValueError("preserved partial result has different source plan provenance")
        rows.extend(artifact["results"])
    result_artifact = {
        "schema": t02.RESULT_SET_SCHEMA,
        "plan_sha256": t02._digest(source_plan),
        "complete_branch_executions": len(rows),
        "branch_cap": source_plan["max_complete_branch_executions"],
        "authorized_complete_branch_cap": source_plan["authorized_complete_branch_cap"],
        "results": rows,
    }
    labels = t02.label_results(source_plan, result_artifact)
    selected_ids = {row["state_id"] for row in recovery_contract["preserved_states"]}
    selected = [row for row in labels["rows"] if row["state_id"] in selected_ids]
    if len(selected) != len(selected_ids):
        raise ValueError("preserved label reconstruction omitted an exact completed state")
    return labels, {
        "schema": "t02-preserved-label-partition-v1",
        "plan_sha256": labels["plan_sha256"],
        "state_count": len(selected),
        "observed_branch_count": 3 * len(selected),
        "rows": selected,
        "source_partial_complete_branch_count": len(rows),
    }


def _write_partial_failure(root, plan, worker_ids, failures):
    owners = read(root / "state_owners.json")
    completed_workers = [
        worker for worker in worker_ids
        if (root / "workers" / worker / "results.json").exists()
    ]
    worker_artifacts = {}
    observed = 0
    for worker in worker_ids:
        worker_root = root / "workers" / worker
        result_path = worker_root / "results.json"
        partial_path = worker_root / "partial_results.json"
        owned_ids = {state_id for state_id, owner in owners.items() if owner == worker}
        artifact_path = result_path if result_path.exists() else partial_path
        count = 0
        if artifact_path.exists():
            count = validate_worker_artifact(
                plan, read(artifact_path), owned_ids, require_complete=result_path.exists()
            )
            observed += count
        worker_artifacts[worker] = {
            "status": "failed" if worker in failures else "completed",
            "results": str(result_path) if result_path.exists() else None,
            "partial_results": str(partial_path) if partial_path.exists() else None,
            "observed_branch_executions": count,
            "failure": failures.get(worker),
        }
    save(root / "run" / "recovery_plan.json", plan)
    save(root / "run" / "summary.json", {
        "schema": "t02-recovery-run-summary-v1",
        "status": "partial_failed",
        "state_count": len(plan["states"]),
        "completed_worker_ids": completed_workers,
        "failed_worker_ids": sorted(failures),
        "worker_artifacts": worker_artifacts,
        "observed_branch_executions": observed,
        "plan_sha256": t02._digest(plan),
        "complete_results_emitted": False,
        "labels_emitted": False,
        "training_allowed": False,
    })
    save(root / "status.json", {
        "phase": "partial_failed",
        "failed_worker_ids": sorted(failures),
        "completed_worker_ids": completed_workers,
        "training_allowed": False,
        "updated_at_epoch": time.time(),
    })


def coordinate(root: Path) -> int:
    root = Path(root)
    contract = read(root / "contract.json")
    worker_ids = [row["worker_id"] for row in contract["workers"]]
    try:
        if dynamic_source_queue(contract):
            while True:
                if (root / "abort.json").exists():
                    raise RuntimeError("Worker failed during recovery source collection")
                ledger = read(root / "ledger.json")
                if all(row["status"] == "completed" for row in ledger["tasks"]):
                    break
                time.sleep(5)
            worker_ids = source_participants(root)
        while not all((root / "workers" / worker / "ready.json").exists() for worker in worker_ids):
            if (root / "abort.json").exists():
                raise RuntimeError("Worker failed during recovery source collection")
            time.sleep(5)
        plan = make_global_plan(root)
        save(root / "status.json", {
            "phase": "branching", "state_count": len(plan["states"]),
            "updated_at_epoch": time.time(),
        })
        while True:
            if (root / "abort.json").exists():
                raise RuntimeError("Worker failed outside isolated branch execution")
            failures = {
                worker: failure
                for worker in worker_ids
                if not (root / "workers" / worker / "results.json").exists()
                and (failure := isolated_branch_failure(root, worker)) is not None
            }
            terminal = {
                worker for worker in worker_ids
                if (root / "workers" / worker / "results.json").exists() or worker in failures
            }
            if terminal == set(worker_ids):
                break
            time.sleep(5)
        if failures:
            _write_partial_failure(root, plan, worker_ids, failures)
            return 1
        partials = {worker: read(root / "workers" / worker / "results.json") for worker in worker_ids}
        results = merge_results(plan, partials, read(root / "state_owners.json"))
        recovery_labels = t02.label_results(plan, results)
        recovery_contract = read(_recovery_paths(root)["contract"])
        source_plan = read(_recovery_paths(root)["source_plan"])
        old_all_labels, old_selected_labels = _preserved_labels(
            root, recovery_contract, source_plan
        )
        combined = merge_label_partitions(recovery_contract, old_all_labels, recovery_labels)
        save(root / "run" / "recovery_plan.json", plan)
        save(root / "run" / "recovery_results.json", results)
        save(root / "run" / "recovery_labels.json", recovery_labels)
        save(root / "run" / "preserved_labels.json", old_selected_labels)
        save(root / "run" / "labels.json", combined)
        summary = {
            "schema": "t02-recovery-run-summary-v1",
            "status": "labels_completed",
            "combined_exact_state_count": combined["state_count"],
            "preserved_exact_state_count": len(old_selected_labels["rows"]),
            "new_exact_state_count": recovery_labels["state_count"],
            "new_complete_branch_executions": results["complete_branch_executions"],
            "cumulative_complete_branch_executions": recovery_contract["budget"]["cumulative_complete_branches"],
            "actual_split_counts": dict(Counter(row["split"] for row in combined["rows"])),
            "task_group_count": len({row["task_group_id"] for row in combined["rows"]}),
            "source_plan_sha256": source_plan and t02._digest(source_plan),
            "recovery_plan_sha256": t02._digest(plan),
            "combined_labels_sha256": t02._digest(combined),
            "training_allowed": True,
        }
        save(root / "run" / "summary.json", summary)
        save(root / "status.json", {
            "phase": "labels_completed", "training_allowed": True,
            "combined_exact_state_count": combined["state_count"],
            "updated_at_epoch": time.time(),
        })
        return 0
    except Exception as error:
        save(root / "abort.json", {
            "phase": "recovery_coordinator", "error_type": type(error).__name__,
            "error": str(error), "updated_at_epoch": time.time(),
        })
        return 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["coordinate"], default="coordinate")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(coordinate(args.root))
