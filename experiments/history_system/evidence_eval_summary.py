"""Read-only aggregation of the four frozen Experiment 3 D20 evaluation lanes.

The report deliberately separates the quality owner of each D20 cell from every
attempt made for that cell.  An officially written zero from an execution that
failed at runtime is therefore retained in attempt accounting but never reused
as a quality result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LANES = ("C0", "C2", "C3", "C5")
SOURCE_SPECS = (
    {
        "id": "original_eval_v1",
        "remote_root": "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_v1",
    },
    {
        "id": "never_started_v3",
        "remote_root": "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_never_started_v3",
    },
    {
        "id": "never_started_c2c3_v2",
        "remote_root": "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_never_started_c2c3_v2",
    },
    {
        "id": "failed_repair_v1",
        "remote_root": "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_failed_repair_v1",
    },
    {
        "id": "c2_remaining_v1",
        "remote_root": "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_c2_remaining_v1",
        "task_layout": "isolated_attempts",
    },
)

REMOTE_COLLECTOR = r'''
import hashlib
import json
from pathlib import Path

LANES = ("C0", "C2", "C3", "C5")

def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None

def sha(path):
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None

def scalar_total(value, key):
    item = value.get(key, {}) if isinstance(value, dict) else {}
    total = item.get("strict_total")
    return total if isinstance(total, (int, float)) and not isinstance(total, bool) else None

def slim_final(value):
    if not isinstance(value, dict):
        return None
    summary = value.get("cost_summary", {})
    costs = summary.get("costs", {}) if isinstance(summary, dict) else {}
    usage = costs.get("openai_resident_usage", {}) if isinstance(costs, dict) else {}
    work = costs.get("actual_model_work", {}) if isinstance(costs, dict) else {}
    return {
        "status": value.get("status"),
        "stop_reason": value.get("stop_reason"),
        "wall_seconds": value.get("wall_seconds"),
        "wall_seconds_final": value.get("wall_seconds_final"),
        "cost": {
            "generation_attempts": summary.get("generation_attempts"),
            "prompt_tokens": scalar_total(usage, "prompt_tokens"),
            "completion_tokens": scalar_total(usage, "completion_tokens"),
            "total_tokens": scalar_total(usage, "total_tokens"),
            "materialized_encoder_tokens": scalar_total(work, "materialized_encoder_tokens"),
        },
    }

def slim_official(value):
    if not isinstance(value, dict):
        return None
    return {key: value.get(key) for key in (
        "benchmark", "categories", "mode", "n_total", "n_scored",
        "correct_count", "semantic_score", "scored",
        "total_gold_checker_seconds", "total_handler_http_calls",
    )}

def runtime_contract(lane_root, design):
    gp = (((design or {}).get("resolved_configs") or {}).get("controller") or {}).get("gp_experiments") or {}
    models = gp.get("local_models") or {}
    reranker = models.get("reranker") or {}
    selector = models.get("selector") or {}
    implementation = lane_root / "runtime/benchmarks/memory_runtime/recovery/local_selection_models.py"
    text = implementation.read_text(encoding="utf-8") if implementation.is_file() else ""
    configured = reranker.get("batch_size")
    default_eight_present = '"batch_size": 8' in text
    effective = configured if isinstance(configured, int) and not isinstance(configured, bool) else (8 if default_eight_present else None)
    return {
        "set_selector": gp.get("set_selector"),
        "semantic_query_overflow_policy": gp.get("semantic_query_overflow_policy"),
        "selector_max_input_tokens": selector.get("max_input_tokens"),
        "reranker_batch_size_configured": configured,
        "reranker_batch_size_effective": effective,
        "reranker_batch_size_basis": "explicit_config" if configured is not None else "frozen_runtime_default",
        "local_selection_models_sha256": sha(implementation),
    }

def collect_source(spec):
    root = Path(spec["remote_root"])
    provenance = read(root / "provenance.json") or {}
    source = {
        "id": spec["id"],
        "remote_root": str(root),
        "provenance": {
            key: value for key, value in provenance.items()
            if key == "schema" or key == "prepared_at" or "sha256" in key or key in ("source_snapshot", "source_package")
        },
        "launch_contract": read(root / "launch_contract.json"),
        "lanes": {},
    }
    for lane in LANES:
        lane_root = root / "lanes" / lane
        if not lane_root.is_dir():
            continue
        tasks_doc = read(lane_root / "tasks.json") or {}
        task_ids = tasks_doc.get("task_ids") or []
        stage = read(lane_root / "results/stage_manifest.json") or {}
        status = read(lane_root / "run/status.json") or {}
        outcomes = {
            item.get("task_id"): item
            for item in stage.get("task_outcomes", [])
            if isinstance(item, dict) and isinstance(item.get("task_id"), str)
        }
        design = read(lane_root / "design.json") or {}
        observations = {}
        for task_id in task_ids:
            shard = lane_root / "results/task_shards" / task_id
            if spec.get("task_layout") == "isolated_attempts":
                shard = lane_root / "results/task_attempts" / task_id / "results/task_shards" / task_id
            observations[task_id] = {
                "stage_outcome": outcomes.get(task_id),
                "server_final": slim_final(read(shard / "server/final.json")),
                "official_summary": slim_official(read(shard / "bfcl/official_summary.json")),
            }
        source["lanes"][lane] = {
            "declared_tasks": task_ids,
            "tasks_manifest_sha256": sha(lane_root / "tasks.json"),
            "status": {
                key: status.get(key) for key in (
                    "schema", "state", "started_at", "finished_at",
                    "completed_task_cells", "stage_status", "automatic_retries", "automatic_reruns",
                )
            },
            "stage": {
                key: stage.get(key) for key in (
                    "schema", "state", "status", "task_cells_started", "completed_task_cells",
                    "denominator_observed", "wall_seconds", "wall_seconds_final", "terminal_error",
                    "automatic_retries", "automatic_reruns",
                )
            },
            "runtime_contract": runtime_contract(lane_root, design),
            "task_observations": observations,
        }
    return source

print(json.dumps({"sources": {spec["id"]: collect_source(spec) for spec in SOURCE_SPECS}}, sort_keys=True))
'''


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _collect_remote(host: str) -> dict[str, Any]:
    prefix = "SOURCE_SPECS = " + repr(list(SOURCE_SPECS)) + "\n"
    completed = subprocess.run(
        ["ssh", host, "python3", "-"],
        input=prefix + REMOTE_COLLECTOR,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Remote read-only collector failed with code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Remote collector did not return one JSON document") from error
    if not isinstance(value, dict) or not isinstance(value.get("sources"), dict):
        raise RuntimeError("Remote collector returned an invalid snapshot")
    value["snapshot_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return value


def _audit_by_lane(audit: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for lane in LANES:
        rows = audit.get("lanes", {}).get(lane, {}).get("tasks", [])
        result[lane] = {row["task_id"]: row for row in rows}
    return result


def _source_lane(snapshot: dict[str, Any], source_id: str, lane: str) -> dict[str, Any]:
    try:
        return snapshot["sources"][source_id]["lanes"][lane]
    except KeyError as error:
        raise ValueError(f"Missing source lane {source_id}/{lane}") from error


def _quality_partition(
    snapshot: dict[str, Any], audit: dict[str, Any]
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    audited = _audit_by_lane(audit)
    owners: dict[str, dict[str, str]] = {}
    validation: dict[str, Any] = {"status": "passed", "lanes": {}}
    canonical_reference: list[str] | None = None
    for lane in LANES:
        canonical = list(_source_lane(snapshot, "original_eval_v1", lane)["declared_tasks"])
        if len(canonical) != 20 or len(set(canonical)) != 20:
            raise ValueError(f"{lane}: original D20 is not exactly 20 unique tasks")
        if canonical_reference is None:
            canonical_reference = canonical
        elif canonical != canonical_reference:
            raise ValueError(f"{lane}: canonical D20 ordering differs across lanes")
        if set(audited[lane]) != set(canonical):
            raise ValueError(f"{lane}: runtime audit does not cover canonical D20 exactly")

        parts: list[tuple[str, set[str]]] = [
            (
                "original_eval_v1",
                {task for task, row in audited[lane].items() if row.get("classification") == "completed_without_runtime_failure"},
            )
        ]
        if lane in ("C0", "C5"):
            parts.append(("never_started_v3", set(_source_lane(snapshot, "never_started_v3", lane)["declared_tasks"])))
        if lane == "C3":
            parts.append(("never_started_c2c3_v2", set(_source_lane(snapshot, "never_started_c2c3_v2", lane)["declared_tasks"])))
        repair_tasks = set(_source_lane(snapshot, "failed_repair_v1", lane)["declared_tasks"])
        if lane == "C2" and "c2_remaining_v1" in snapshot["sources"]:
            continued = set(_source_lane(snapshot, "c2_remaining_v1", lane)["declared_tasks"])
            repair_lane = _source_lane(snapshot, "failed_repair_v1", lane)
            unstarted = {
                task for task in repair_tasks
                if (repair_lane["task_observations"][task].get("stage_outcome") or {}).get("outcome") == "not_started"
            }
            if len(continued) != 8 or continued != unstarted:
                raise ValueError("C2: remaining8 must cover only the repair's eight never-started tasks")
            parts.append(("c2_remaining_v1", continued))
            repair_tasks -= continued
        parts.append(("failed_repair_v1", repair_tasks))

        seen: dict[str, str] = {}
        overlaps: dict[str, list[str]] = {}
        for source_id, tasks in parts:
            for task in tasks:
                if task in seen:
                    overlaps.setdefault(task, [seen[task]]).append(source_id)
                seen[task] = source_id
        missing = sorted(set(canonical) - set(seen))
        unexpected = sorted(set(seen) - set(canonical))
        if overlaps or missing or unexpected or len(seen) != 20:
            raise ValueError(
                f"{lane}: invalid quality partition overlaps={overlaps} "
                f"missing={missing} unexpected={unexpected}"
            )

        original_not_started = {
            task for task, row in audited[lane].items() if row.get("classification") == "not_started"
        }
        original_failed = {
            task for task, row in audited[lane].items()
            if row.get("classification") in ("runtime_failure", "interrupted")
        }
        if lane in ("C0", "C5"):
            continuation = set(_source_lane(snapshot, "never_started_v3", lane)["declared_tasks"])
            if continuation != original_not_started:
                raise ValueError(f"{lane}: never-started continuation differs from original audit")
            repair_expected = original_failed
        elif lane == "C3":
            continuation = set(_source_lane(snapshot, "never_started_c2c3_v2", lane)["declared_tasks"])
            if continuation != original_not_started:
                raise ValueError(f"{lane}: C3 continuation differs from original audit")
            repair_expected = original_failed
        else:
            followup = set(_source_lane(snapshot, "never_started_c2c3_v2", lane)["declared_tasks"])
            if followup != original_not_started:
                raise ValueError("C2: batch8 follow-up differs from original not-started cells")
            repair_expected = original_failed | followup
        repair_observed = set(_source_lane(snapshot, "failed_repair_v1", lane)["declared_tasks"])
        if repair_observed != repair_expected:
            raise ValueError(f"{lane}: repair task set does not match failed/unstarted contract")

        owners[lane] = seen
        validation["lanes"][lane] = {
            "canonical_task_count": 20,
            "unique_owned_task_count": len(seen),
            "missing": [],
            "overlap": [],
            "owner_counts": {
                source_id: sum(owner == source_id for owner in seen.values())
                for source_id, _ in parts
            },
        }
    return owners, validation


def _official_valid(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("scored") is True
        and value.get("n_total") == 1
        and value.get("n_scored") == 1
        and isinstance(value.get("correct_count"), int)
        and value.get("correct_count") in (0, 1)
        and isinstance(value.get("semantic_score"), (int, float))
        and math.isfinite(float(value["semantic_score"]))
    )


def _task_state(
    source_id: str,
    observation: dict[str, Any],
    audit_row: dict[str, Any] | None = None,
) -> tuple[str, str | None]:
    stage = observation.get("stage_outcome")
    official = observation.get("official_summary")
    if source_id == "original_eval_v1":
        if audit_row is None:
            return "runtime_failed", "missing_runtime_audit"
        classification = audit_row.get("classification")
        if classification == "not_started":
            return "pending", "not_started"
        if classification in ("runtime_failure", "interrupted"):
            return "runtime_failed", classification
        runtime_completed = classification == "completed_without_runtime_failure"
    elif not isinstance(stage, dict) or stage.get("outcome") == "not_started":
        return "pending", "not_started_or_not_yet_recorded"
    else:
        if stage.get("runtime_completed") is False:
            return "runtime_failed", str(stage.get("outcome") or "runtime_completed_false")
        if stage.get("outcome") in (
            "runtime_failure_in_denominator", "failed_in_denominator",
            "interrupted_in_denominator", "server_start_failure",
        ):
            return "runtime_failed", str(stage.get("outcome"))
        runtime_completed = stage.get("runtime_completed") is True
        if not runtime_completed:
            return "pending", "started_without_terminal_runtime_outcome"
    if not _official_valid(official):
        return "runtime_failed", "official_summary_missing_or_invalid_after_runtime_completion"
    return "completed", None


def _cost_fields(observation: dict[str, Any]) -> dict[str, float | int | None]:
    final = observation.get("server_final") or {}
    cost = final.get("cost") or {}
    official = observation.get("official_summary") or {}
    return {
        "wall_seconds": final.get("wall_seconds") if isinstance(final.get("wall_seconds"), (int, float)) else None,
        "gold_checker_seconds": official.get("total_gold_checker_seconds") if isinstance(official.get("total_gold_checker_seconds"), (int, float)) else None,
        "handler_http_calls": official.get("total_handler_http_calls") if isinstance(official.get("total_handler_http_calls"), int) else None,
        "generation_attempts": cost.get("generation_attempts") if isinstance(cost.get("generation_attempts"), int) else None,
        "prompt_tokens": cost.get("prompt_tokens") if isinstance(cost.get("prompt_tokens"), (int, float)) else None,
        "completion_tokens": cost.get("completion_tokens") if isinstance(cost.get("completion_tokens"), (int, float)) else None,
        "total_tokens": cost.get("total_tokens") if isinstance(cost.get("total_tokens"), (int, float)) else None,
        "materialized_encoder_tokens": cost.get("materialized_encoder_tokens") if isinstance(cost.get("materialized_encoder_tokens"), (int, float)) else None,
    }


def _sum_observed(rows: Iterable[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row["measured_cost"][key] for row in rows if row["measured_cost"].get(key) is not None]
    return {"observed_attempts": len(values), "sum": sum(values) if values else None}


def _attempt_rows(
    snapshot: dict[str, Any], audit: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    audited = _audit_by_lane(audit)
    rows: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANES}
    for source_id, source in snapshot["sources"].items():
        for lane, source_lane in source["lanes"].items():
            for task_id in source_lane["declared_tasks"]:
                observation = source_lane["task_observations"][task_id]
                stage = observation.get("stage_outcome")
                audit_row = audited[lane].get(task_id) if source_id == "original_eval_v1" else None
                if source_id == "original_eval_v1":
                    started = audit_row is not None and audit_row.get("classification") != "not_started"
                else:
                    started = isinstance(stage, dict) and stage.get("outcome") != "not_started"
                if not started:
                    continue
                state, reason = _task_state(source_id, observation, audit_row)
                rows[lane].append({
                    "source_id": source_id,
                    "task_id": task_id,
                    "attempt_status": state,
                    "failure_reason": reason if state == "runtime_failed" else None,
                    "stage_outcome": stage.get("outcome") if isinstance(stage, dict) else None,
                    "worker_returncode": stage.get("worker_returncode") if isinstance(stage, dict) else (audit_row or {}).get("worker_returncode"),
                    "server_returncode": stage.get("server_returncode") if isinstance(stage, dict) else (audit_row or {}).get("server_returncode"),
                    "official_summary_written": observation.get("official_summary") is not None,
                    "official_result_accepted_for_quality": state == "completed",
                    "measured_cost": _cost_fields(observation),
                })
    return rows


def build_summary(
    snapshot: dict[str, Any], audit: dict[str, Any], observed_at: str
) -> dict[str, Any]:
    owners, partition_validation = _quality_partition(snapshot, audit)
    audited = _audit_by_lane(audit)
    attempts = _attempt_rows(snapshot, audit)
    result: dict[str, Any] = {
        "schema": "evidence-sets-d20-summary-v1",
        "observed_at": observed_at,
        "quality_label": "preliminary, n=1",
        "scope": "Four controller lanes over the fixed 20-task D20 manifest per lane.",
        "completion_rule": "A cell is completed only when runtime completion and a valid one-task official summary both exist. Runtime-failed official zeros are excluded from quality results and retained as attempt cost.",
        "remote_snapshot_sha256": snapshot.get("snapshot_sha256"),
        "runtime_failure_audit_canonical_sha256": hashlib.sha256(
            json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "partition_validation": partition_validation,
        "sources": {},
        "lanes": {},
    }
    for source_id, source in snapshot["sources"].items():
        result["sources"][source_id] = {
            "remote_root": source["remote_root"],
            "provenance": source.get("provenance"),
            "launch_contract_schema": (source.get("launch_contract") or {}).get("schema"),
            "lane_runtime_contracts": {
                lane: data["runtime_contract"] for lane, data in source["lanes"].items()
            },
            "lane_states": {
                lane: {
                    "run_state": data["status"].get("state"),
                    "stage_state": data["stage"].get("state"),
                    "stage_status": data["stage"].get("status"),
                    "declared_task_count": len(data["declared_tasks"]),
                }
                for lane, data in source["lanes"].items()
            },
        }

    all_final = True
    for lane in LANES:
        canonical = _source_lane(snapshot, "original_eval_v1", lane)["declared_tasks"]
        cells = []
        for task_id in canonical:
            source_id = owners[lane][task_id]
            source_lane = _source_lane(snapshot, source_id, lane)
            observation = source_lane["task_observations"][task_id]
            audit_row = audited[lane].get(task_id) if source_id == "original_eval_v1" else None
            state, reason = _task_state(source_id, observation, audit_row)
            official = observation.get("official_summary") if state == "completed" else None
            cells.append({
                "task_id": task_id,
                "quality_source_id": source_id,
                "status": state,
                "status_reason": reason,
                "official": official,
                "measured_cost": _cost_fields(observation),
            })
        completed = [cell for cell in cells if cell["status"] == "completed"]
        pending = [cell for cell in cells if cell["status"] == "pending"]
        failed = [cell for cell in cells if cell["status"] == "runtime_failed"]
        n_scored = sum(cell["official"]["n_scored"] for cell in completed)
        correct = sum(cell["official"]["correct_count"] for cell in completed)
        final_ready = len(completed) == 20 and not pending and not failed and n_scored == 20
        all_final = all_final and final_ready
        lane_attempts = attempts[lane]
        failed_attempts = [row for row in lane_attempts if row["attempt_status"] == "runtime_failed"]
        result["lanes"][lane] = {
            "expected_unique_tasks": 20,
            "counts": {
                "completed": len(completed),
                "pending": len(pending),
                "runtime_failed": len(failed),
                "official_scored_cells": len(completed),
                "official_correct_count": correct,
                "official_n_scored": n_scored,
            },
            "partial_completed_official": {
                "is_final": False,
                "completed_cells": len(completed),
                "n_scored": n_scored,
                "correct_count": correct,
                "semantic_score": correct / n_scored if n_scored else None,
            },
            "final_d20_official": (
                {"n_scored": n_scored, "correct_count": correct, "semantic_score": correct / n_scored}
                if final_ready else None
            ),
            "quality_cells": cells,
            "quality_latency_and_cost": {
                key: _sum_observed(completed, key)
                for key in (
                    "wall_seconds", "gold_checker_seconds", "handler_http_calls", "generation_attempts",
                    "prompt_tokens", "completion_tokens", "total_tokens", "materialized_encoder_tokens",
                )
            },
            "attempt_accounting": {
                "attempts_started": len(lane_attempts),
                "runtime_failed_attempts": len(failed_attempts),
                "failed_attempts": failed_attempts,
                "all_started_attempts": lane_attempts,
                "measured_latency_and_cost": {
                    key: _sum_observed(lane_attempts, key)
                    for key in (
                        "wall_seconds", "gold_checker_seconds", "handler_http_calls", "generation_attempts",
                        "prompt_tokens", "completion_tokens", "total_tokens", "materialized_encoder_tokens",
                    )
                },
            },
        }
    result["final_results_ready"] = all_final
    result["final_score_policy"] = (
        "All four final_d20_official objects are available."
        if all_final else "At least one lane is incomplete; partial_completed_official is descriptive only."
    )
    result["table_rows"] = [
        {
            "lane": lane,
            **result["lanes"][lane]["counts"],
            "partial_semantic_score": result["lanes"][lane]["partial_completed_official"]["semantic_score"],
            "final_semantic_score": (
                result["lanes"][lane]["final_d20_official"] or {}
            ).get("semantic_score"),
        }
        for lane in LANES
    ]
    result["totals"] = {
        key: sum(result["lanes"][lane]["counts"][key] for lane in LANES)
        for key in (
            "completed", "pending", "runtime_failed", "official_scored_cells",
            "official_correct_count", "official_n_scored",
        )
    }
    result["totals"]["attempts_started_including_superseded_failures"] = sum(
        result["lanes"][lane]["attempt_accounting"]["attempts_started"] for lane in LANES
    )
    result["totals"]["runtime_failed_attempts_including_superseded_failures"] = sum(
        result["lanes"][lane]["attempt_accounting"]["runtime_failed_attempts"] for lane in LANES
    )
    return result


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_eval = repo_root / "outputs/history_system_search/evidence_sets_v1/eval"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default="npu")
    parser.add_argument("--audit", type=Path, default=default_eval / "runtime_failure_audit.json")
    parser.add_argument("--output", type=Path, default=default_eval / "summary.latest.json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    snapshot = _collect_remote(args.ssh_host)
    audit = _read_json(args.audit)
    observed_at = datetime.now(timezone.utc).isoformat()
    summary = build_summary(snapshot, audit, observed_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "final_results_ready": summary["final_results_ready"],
        "lanes": {lane: summary["lanes"][lane]["counts"] for lane in LANES},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
