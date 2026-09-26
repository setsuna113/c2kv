"""Finalize bounded T02 streaming recovery after every source is terminal."""

from __future__ import annotations

import argparse
import hashlib
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import t02
import t02_streaming_recovery as streaming
from t02_parallel import read, save
from t02_recovery_parallel import _preserved_labels


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _all_slots_terminal(root: Path, worker_ids: Sequence[str]) -> bool:
    terminal = {"completed", "completed_no_work", "failed_no_retry"}
    for worker_id in worker_ids:
        path = root / "workers" / worker_id / "slot.json"
        if not path.exists():
            return False
        try:
            phase = read(path).get("phase")
        except Exception:
            return False
        if phase not in terminal:
            return False
    return True


def _write_partial_summary(root: Path, ledger: Mapping[str, Any]) -> int:
    status_counts = Counter(row["status"] for row in ledger["tasks"])
    observed = sum(row["complete_branch_executions"] for row in ledger["tasks"])
    failed = [
        {
            "task_id": row["task_id"],
            "task_index": row["task_index"],
            "status": row["status"],
            "worker_id": row.get("worker_id"),
            "plan_sha256": row.get("plan_sha256"),
            "complete_branch_executions": row["complete_branch_executions"],
            "result_path": row.get("result_path"),
            "failure": row.get("failure"),
        }
        for row in ledger["tasks"] if row["status"] != "completed"
    ]
    summary = {
        "schema": streaming.SUMMARY_SCHEMA,
        "phase": "partial_failed",
        "status": "partial_failed",
        "source_task_count": len(ledger["tasks"]),
        "task_status_counts": dict(status_counts),
        "failed_or_unstarted_tasks": failed,
        "observed_new_complete_branch_executions": observed,
        "cumulative_observed_complete_branch_executions": (
            ledger["complete_branches_before_recovery"] + observed
        ),
        "complete_results_emitted": False,
        "labels_emitted": False,
        "training_allowed": False,
        "automatic_failed_task_retries": 0,
    }
    save(root / "run" / "summary.json", summary)
    save(root / "status.json", {
        "phase": "partial_failed",
        "training_allowed": False,
        "failed_or_unstarted_task_count": len(failed),
        "updated_at_epoch": time.time(),
    })
    return 1


def finalize(root: str | Path) -> int:
    """Validate all per-task artifacts and emit labels only for exact completion."""

    root = Path(root).resolve()
    parallel, recovery, source_plan = streaming._recovery_inputs(root)
    ledger = read(root / "ledger.json")
    if ledger.get("schema") != streaming.LEDGER_SCHEMA:
        raise ValueError("streaming recovery ledger schema changed")
    if not all(row["status"] in streaming.TERMINAL_TASK_STATUSES for row in ledger["tasks"]):
        raise RuntimeError("cannot finalize while a streaming source is nonterminal")
    if any(row["status"] != "completed" for row in ledger["tasks"]):
        return _write_partial_summary(root, ledger)

    expected_new_branches = recovery["budget"]["new_complete_branch_executions"]
    observed_new_branches = sum(row["complete_branch_executions"] for row in ledger["tasks"])
    if observed_new_branches != expected_new_branches:
        raise ValueError("completed streaming ledger differs from the frozen branch budget")

    task_labels: list[dict[str, Any]] = []
    task_receipts = []
    for source, row in zip(recovery["source_tasks"], ledger["tasks"]):
        if row["task_id"] != source["task_id"]:
            raise ValueError("streaming ledger changed frozen source order")
        plan_path = Path(row["plan_path"])
        result_path = Path(row["result_path"])
        label_path = Path(row["label_path"])
        if not all(path.is_file() for path in (plan_path, result_path, label_path)):
            raise FileNotFoundError("completed streaming task artifact is missing")
        if _file_sha256(plan_path) != row["plan_file_sha256"]:
            raise ValueError("streaming task plan file digest changed")
        if _file_sha256(result_path) != row["result_file_sha256"]:
            raise ValueError("streaming task result file digest changed")
        if _file_sha256(label_path) != row["label_file_sha256"]:
            raise ValueError("streaming task label file digest changed")
        plan = read(plan_path)
        results = read(result_path)
        labels = read(label_path)
        streaming.validate_task_plan(plan, recovery=recovery)
        if streaming._digest(plan) != row["plan_sha256"]:
            raise ValueError("streaming task plan object digest changed")
        merged_results = streaming.merge_task_results(
            plan, [results], require_complete=True
        )
        if streaming._digest(merged_results) != streaming._digest(results):
            raise ValueError("streaming task result does not canonicalize exactly")
        recomputed_labels = streaming.label_task_results(plan, results)
        if streaming._digest(recomputed_labels) != streaming._digest(labels):
            raise ValueError("streaming task labels differ from their result artifact")
        labels["plan_file_sha256"] = row["plan_file_sha256"]
        labels["label_file_sha256"] = row["label_file_sha256"]
        labels["label_payload_sha256"] = streaming._digest(recomputed_labels)
        task_labels.append(labels)
        task_receipts.append({
            "task_id": row["task_id"],
            "task_index": row["task_index"],
            "worker_id": row["worker_id"],
            "plan_path": str(plan_path),
            "plan_sha256": row["plan_sha256"],
            "plan_file_sha256": row["plan_file_sha256"],
            "result_path": str(result_path),
            "result_file_sha256": row["result_file_sha256"],
            "label_path": str(label_path),
            "label_file_sha256": row["label_file_sha256"],
            "state_ids": [item["state_id"] for item in labels["rows"]],
            "complete_branch_executions": row["complete_branch_executions"],
        })

    old_all_labels, old_selected_labels = _preserved_labels(
        root, recovery, source_plan
    )
    preserved_path = root / "run" / "preserved_labels.json"
    save(preserved_path, old_selected_labels)
    preserved_for_merge = dict(old_selected_labels)
    preserved_for_merge["label_payload_sha256"] = streaming._digest(old_selected_labels)
    preserved_for_merge["label_file_sha256"] = _file_sha256(preserved_path)
    combined = streaming.merge_label_partitions(
        recovery, preserved_for_merge, task_labels
    )
    save(root / "run" / "task_partitions.json", {
        "schema": "t02-streaming-task-partitions-v1",
        "task_count": len(task_receipts),
        "state_count": sum(len(row["state_ids"]) for row in task_receipts),
        "complete_branch_executions": observed_new_branches,
        "tasks": task_receipts,
    })
    save(root / "run" / "labels.json", combined)
    split_counts = Counter(row["split"] for row in combined["rows"])
    summary = {
        "schema": streaming.SUMMARY_SCHEMA,
        "phase": "labels_completed",
        "status": "labels_completed",
        "source_task_count": len(recovery["source_tasks"]),
        "combined_exact_state_count": combined["state_count"],
        "preserved_exact_state_count": len(old_selected_labels["rows"]),
        "new_exact_state_count": sum(len(value["rows"]) for value in task_labels),
        "streaming_task_plan_count": len(task_labels),
        "new_complete_branch_executions": observed_new_branches,
        "cumulative_complete_branch_executions": (
            recovery["budget"]["complete_branches_before_recovery"]
            + observed_new_branches
        ),
        "actual_split_counts": dict(split_counts),
        "task_group_count": len({row["task_group_id"] for row in combined["rows"]}),
        "source_plan_sha256": t02._digest(source_plan),
        "actual_task_plan_sha256s": [
            partition["plan_sha256"] for partition in combined["partitions"][1:]
        ],
        "synthetic_global_plan_sha256": None,
        "combined_labels_sha256": streaming._digest(combined),
        "training_allowed": True,
    }
    save(root / "run" / "summary.json", summary)
    save(root / "status.json", {
        "phase": "labels_completed",
        "training_allowed": True,
        "combined_exact_state_count": combined["state_count"],
        "streaming_task_plan_count": len(task_labels),
        "updated_at_epoch": time.time(),
    })
    return 0


def coordinate(root: str | Path, *, poll_seconds: float = 5.0) -> int:
    root = Path(root).resolve()
    contract = read(root / "contract.json")
    worker_ids = [row["worker_id"] for row in contract["workers"]]
    while True:
        ledger = read(root / "ledger.json")
        if all(row["status"] in streaming.TERMINAL_TASK_STATUSES for row in ledger["tasks"]):
            break
        if _all_slots_terminal(root, worker_ids):
            streaming.mark_orphaned_and_unstarted(root, worker_ids)
            break
        time.sleep(poll_seconds)
    try:
        return finalize(root)
    except Exception as error:
        save(root / "run" / "summary.json", {
            "schema": streaming.SUMMARY_SCHEMA,
            "status": "finalization_failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "labels_emitted": False,
            "training_allowed": False,
        })
        save(root / "status.json", {
            "phase": "finalization_failed",
            "training_allowed": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "updated_at_epoch": time.time(),
        })
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["coordinate", "finalize"],
                        default="coordinate")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    return finalize(args.root) if args.command == "finalize" else coordinate(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
