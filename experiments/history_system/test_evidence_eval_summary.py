from __future__ import annotations

import copy

import pytest

from experiments.history_system.evidence_eval_summary import build_summary


LANES = ("C0", "C2", "C3", "C5")
TASKS = [f"task_{index:02d}" for index in range(20)]


def _official(correct: int = 0) -> dict:
    return {
        "benchmark": "bfcl",
        "categories": "fixture",
        "mode": "both",
        "n_total": 1,
        "n_scored": 1,
        "correct_count": correct,
        "semantic_score": float(correct),
        "scored": True,
        "total_gold_checker_seconds": 0.1,
        "total_handler_http_calls": 0,
    }


def _observation(outcome: str | None, runtime_completed: bool | None, *, official: dict | None) -> dict:
    stage = None
    if outcome is not None:
        stage = {
            "task_id": "filled_by_caller",
            "outcome": outcome,
            "runtime_completed": runtime_completed,
            "worker_returncode": 0,
            "server_returncode": 0 if runtime_completed else 1,
        }
    return {
        "stage_outcome": stage,
        "server_final": {
            "wall_seconds": 10.0,
            "cost": {
                "generation_attempts": 2,
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "total_tokens": 110,
                "materialized_encoder_tokens": 5,
            },
        } if outcome not in (None, "not_started") else None,
        "official_summary": official,
    }


def _lane(tasks: list[str], observations: dict[str, dict]) -> dict:
    for task_id, observation in observations.items():
        if isinstance(observation.get("stage_outcome"), dict):
            observation["stage_outcome"]["task_id"] = task_id
    return {
        "declared_tasks": tasks,
        "tasks_manifest_sha256": "fixture",
        "status": {"state": "fixture"},
        "stage": {"state": "fixture", "status": "fixture"},
        "runtime_contract": {"fixture": True},
        "task_observations": observations,
    }


def _source(root: str) -> dict:
    return {
        "remote_root": root,
        "provenance": {"schema": "fixture"},
        "launch_contract": {"schema": "fixture"},
        "lanes": {},
    }


def _fixture(*, repair_complete: bool = False) -> tuple[dict, dict]:
    sources = {
        "original_eval_v1": _source("/original"),
        "never_started_v3": _source("/never-v3"),
        "never_started_c2c3_v2": _source("/c2c3-v2"),
        "failed_repair_v1": _source("/repair"),
    }
    audit = {"lanes": {}}
    for lane in LANES:
        clean = set(TASKS[:5])
        not_started = set(TASKS[14:] if lane in ("C0", "C5") else TASKS[13:])
        failed = set(TASKS) - clean - not_started
        audit_rows = []
        original_observations = {}
        for index, task_id in enumerate(TASKS):
            if task_id in clean:
                classification = "completed_without_runtime_failure"
                original_observations[task_id] = _observation(
                    "official_completed", None, official=_official(index % 2)
                )
            elif task_id in not_started:
                classification = "not_started"
                original_observations[task_id] = _observation(None, None, official=None)
            else:
                classification = "runtime_failure" if task_id != TASKS[13] else "interrupted"
                # This official zero must never be accepted as a quality result.
                original_observations[task_id] = _observation(
                    "official_completed", None, official=_official(0)
                )
                original_observations[task_id]["stage_outcome"]["server_returncode"] = 1
            audit_rows.append({"task_id": task_id, "classification": classification})
        audit["lanes"][lane] = {"tasks": audit_rows}
        sources["original_eval_v1"]["lanes"][lane] = _lane(TASKS.copy(), original_observations)

        if lane in ("C0", "C5"):
            ids = sorted(not_started)
            observations = {
                task: _observation("official_completed", True, official=_official(1)) for task in ids
            }
            sources["never_started_v3"]["lanes"][lane] = _lane(ids, observations)
        else:
            ids = sorted(not_started)
            observations = {}
            for pos, task in enumerate(ids):
                if lane == "C2" and pos == 0:
                    observations[task] = _observation(
                        "runtime_failure_in_denominator", False, official=_official(0)
                    )
                else:
                    observations[task] = _observation(
                        "official_completed" if lane == "C3" else "not_started",
                        True if lane == "C3" else None,
                        official=_official(1) if lane == "C3" else None,
                    )
            sources["never_started_c2c3_v2"]["lanes"][lane] = _lane(ids, observations)

        repair_ids = sorted(failed | (not_started if lane == "C2" else set()))
        repair_observations = {
            task: (
                _observation("official_completed", True, official=_official(1))
                if repair_complete else _observation(None, None, official=None)
            )
            for task in repair_ids
        }
        sources["failed_repair_v1"]["lanes"][lane] = _lane(repair_ids, repair_observations)
    return {"sources": sources, "snapshot_sha256": "fixture"}, audit


def test_runtime_failed_official_zero_is_attempt_cost_not_quality() -> None:
    snapshot, audit = _fixture(repair_complete=False)
    summary = build_summary(snapshot, audit, "2026-09-17T00:00:00+00:00")

    assert summary["partition_validation"]["status"] == "passed"
    assert summary["final_results_ready"] is False
    assert summary["lanes"]["C2"]["counts"] == {
        "completed": 5,
        "pending": 15,
        "runtime_failed": 0,
        "official_scored_cells": 5,
        "official_correct_count": 2,
        "official_n_scored": 5,
    }
    assert summary["lanes"]["C2"]["final_d20_official"] is None
    failed_attempts = summary["lanes"]["C2"]["attempt_accounting"]["failed_attempts"]
    assert len(failed_attempts) == 9  # eight original failures plus one batch8 follow-up OOM
    assert all(not row["official_result_accepted_for_quality"] for row in failed_attempts)


def test_complete_partition_emits_final_only_after_all_twenty_cells() -> None:
    snapshot, audit = _fixture(repair_complete=True)
    summary = build_summary(snapshot, audit, "2026-09-17T00:00:00+00:00")

    assert summary["final_results_ready"] is True
    for lane in LANES:
        assert summary["lanes"][lane]["counts"]["completed"] == 20
        assert summary["lanes"][lane]["counts"]["pending"] == 0
        assert summary["lanes"][lane]["counts"]["runtime_failed"] == 0
        assert summary["lanes"][lane]["final_d20_official"]["n_scored"] == 20


def test_partition_rejects_overlap_even_when_all_sources_exist() -> None:
    snapshot, audit = _fixture(repair_complete=False)
    broken = copy.deepcopy(snapshot)
    repair = broken["sources"]["failed_repair_v1"]["lanes"]["C0"]
    repair["declared_tasks"].append(TASKS[0])
    repair["task_observations"][TASKS[0]] = _observation(None, None, official=None)

    with pytest.raises(ValueError, match="invalid quality partition"):
        build_summary(broken, audit, "2026-09-17T00:00:00+00:00")


def _add_c2_remaining(snapshot):
    repair = snapshot["sources"]["failed_repair_v1"]["lanes"]["C2"]
    tasks = repair["declared_tasks"][-8:]
    for task in tasks:
        repair["task_observations"][task] = _observation("not_started", False, official=None)
    source = _source("/remaining8")
    source["lanes"]["C2"] = _lane(tasks, {
        task: _observation("official_completed", True, official=_official(1)) for task in tasks
    })
    snapshot["sources"]["c2_remaining_v1"] = source
    return tasks


def test_remaining8_transfers_only_unstarted_cells_and_retains_failure():
    snapshot, audit = _fixture(repair_complete=True)
    continued = _add_c2_remaining(snapshot)
    repair = snapshot["sources"]["failed_repair_v1"]["lanes"]["C2"]
    failed_task = next(task for task in repair["declared_tasks"] if task not in continued)
    repair["task_observations"][failed_task] = _observation("runtime_failure_in_denominator", False, official=_official(0))
    summary = build_summary(snapshot, audit, "2026-09-17T00:00:00+00:00")
    lane = summary["lanes"]["C2"]
    assert lane["counts"]["completed"] == 19
    assert lane["counts"]["runtime_failed"] == 1
    assert lane["counts"]["pending"] == 0
    assert lane["final_d20_official"] is None
    assert sum(cell["quality_source_id"] == "c2_remaining_v1" for cell in lane["quality_cells"]) == 8


def test_remaining8_cannot_replace_an_already_executed_cell():
    snapshot, audit = _fixture(repair_complete=True)
    continued = _add_c2_remaining(snapshot)
    repair = snapshot["sources"]["failed_repair_v1"]["lanes"]["C2"]
    repair["task_observations"][continued[0]] = _observation("official_completed", True, official=_official(0))
    with pytest.raises(ValueError, match="eight never-started"):
        build_summary(snapshot, audit, "2026-09-17T00:00:00+00:00")
