from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from benchmarks.memory_runtime import analyze_s1_long_stage as single
from benchmarks.memory_runtime import analyze_s1_spare_raw_pair as analysis


TASKS = [f"multi_turn_long_context_{index}" for index in range(0, 200, 10)]


def _spare_receipt(*, missing: int = 0) -> dict:
    return {
        "known_receipts": 1,
        "missing_receipts": missing,
        "requests_with_explicit_policy": 1,
        "requests_with_input_change": 1,
        "event_attempts_added": 1,
        "event_attempts_already_raw": 0,
        "event_attempts_over_budget": 1,
        "added_event_receipt_appearances": 1,
        "added_source_occurrence_appearances": 2,
        "requests_admitted_over_budget": 0,
        "added_event_attempts_over_budget": 0,
        "incremental_raw_tokens_sum": 3,
        "incremental_bytes_sum": 6,
        "maximum_incremental_raw_tokens": 3,
        "maximum_incremental_bytes": 6,
        "maximum_active_history_bytes_before": 10,
        "maximum_active_history_bytes_after": 16,
        "policies": [analysis.EXPECTED_POLICY],
        "versions": ["dependency-packet-spare-raw-v1"],
    }


def _round(candidate: str, run_id: str, *, complete: bool, scored: int = 20) -> dict:
    rows = []
    correct = 0
    missing = []
    for ordinal, task in enumerate(TASKS):
        known = ordinal < scored
        if not known:
            missing.append(task)
        is_correct = known and ordinal == 0
        correct += int(is_correct)
        row = {
            "task_id": task,
            "quality": {
                "official_score_known": known,
                "correct_count": int(is_correct) if known else None,
            },
            "cost": {
                "request_count": 1 if known else None,
                "task_wall_seconds": 2.0 if known else None,
                "resources": {
                    "active_history_bytes_max": 16 if known else None,
                    "extraction_producer_calls": 1 if known else None,
                },
            },
            "source_coverage": {
                "known_receipts": 1 if known else 0,
                "missing_receipts": 0 if known else 1,
                "eligible_source_occurrence_appearances": 2 if known else 0,
                "fully_represented_source_occurrence_appearances": 2 if known else 0,
                "unrepresented_source_occurrence_appearances": 0,
            },
            "dependency_packet_exposure": {
                "known_receipts": 1 if known else 0,
                "missing_receipts": 0 if known else 1,
                "requests_with_nonempty_packet": 1 if known else 0,
                "requests_without_nonempty_packet": 0,
                "retained_fact_appearances": 1 if known else 0,
                "eligible_fact_appearances": 1 if known else 0,
                "field_limit_omission_appearances": 0,
                "unrepresented_requested_source_appearances": 0,
                "workspace_drop_appearances": 0,
            },
        }
        if candidate == single.SPARE_RAW_CANDIDATE:
            row["dependency_packet_spare_raw_exposure"] = (
                _spare_receipt() if known else {
                    **_spare_receipt(), "known_receipts": 0,
                    "missing_receipts": 0,
                    "requests_with_explicit_policy": 0,
                    "policies": [], "versions": [],
                }
            )
        rows.append(row)
    return {
        "schema": single.SCHEMA,
        "status": "complete" if complete else "partial",
        "run_identity": {
            "candidate_id": candidate,
            "run_id": run_id,
            "task_ids": TASKS,
        },
        "quality": {
            "fixed_task_denominator": 20,
            "official_scored_tasks": scored,
            "official_missing_tasks": 20 - scored,
            "official_missing_task_ids": missing,
            "official_correct_tasks": correct,
        },
        "recovery": {
            "missing_task_count": 0 if complete else 20 - scored,
            "completion_errors": [],
        },
        "per_task": rows,
        "paired_with_s0_bounded_latest": {"paired_task_count": scored},
    }


def _design(path: Path, candidate: str, run_id: str) -> str:
    path.write_text(json.dumps({
        "schema": analysis.FROZEN_DESIGN_SCHEMA,
        "status": "frozen",
        "candidate_id": candidate,
        "run_id": run_id,
        "task_ids": TASKS,
    }), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_spare_raw_receipt_aggregation_validates_bytes_and_statuses() -> None:
    rows = [{
        "eval_context": {"user_turn": 1, "step": 2},
        "memory_runtime": {"bytes_per_kv_token": 3, "dependency_packet_spare_raw": {
            "version": "dependency-packet-spare-raw-v1",
            "policy": analysis.EXPECTED_POLICY,
            "uses_gold_future_s0_or_hidden_state": False,
            "existing_gist_packet_state_raw_and_failed_cue_unchanged": True,
            "incremental_raw_tokens": 2,
            "incremental_bytes": 6,
            "active_history_bytes_before": 10,
            "active_history_bytes_after": 16,
            "budget_bytes": 20,
            "added_event_ids": ["redacted-fixture-id"],
            "added_source_indices": [1, 2],
            "items": [
                {"status": "added", "candidate_active_history_bytes": 16},
                {"status": "over_budget", "candidate_active_history_bytes": 21},
                {"status": "already_raw", "candidate_active_history_bytes": 16},
            ],
        }},
    }]
    receipt = single._spare_raw(rows)

    assert receipt["policies"] == [analysis.EXPECTED_POLICY]
    assert receipt["requests_with_input_change"] == 1
    assert receipt["event_attempts_added"] == 1
    assert receipt["event_attempts_over_budget"] == 1
    assert receipt["maximum_active_history_bytes_after"] == 16
    assert receipt["requests_admitted_over_budget"] == 0


def test_partial_keeps_fixed_denominator_and_unknown_costs_null() -> None:
    old = _round(single.CANDIDATE, "old", complete=True)
    new = _round(single.SPARE_RAW_CANDIDATE, "new", complete=False, scored=1)

    result = analysis.compare(old, new, mode="partial")

    assert result["status"] == "partial"
    assert result["quality"]["new"]["official_missing_tasks"] == 19
    assert result["quality"]["unpaired_official_task_count"] == 19
    assert result["cost"]["request_count"]["strict_fixed_20_new"] is None
    assert result["spare_raw"]["new_policy_exposure"]["status"] == (
        "expected_policy_fully_observed")


def test_final_requires_frozen_bindings_and_complete_policy(tmp_path: Path) -> None:
    old = _round(single.CANDIDATE, "old", complete=True)
    new = _round(single.SPARE_RAW_CANDIDATE, "new", complete=True)
    with pytest.raises(
            analysis.SpareRawPairedAnalysisError,
            match="both old and new frozen-design bindings are required"):
        analysis.compare(old, new, mode="final")

    old_design = tmp_path / "old.json"
    new_design = tmp_path / "new.json"
    old["run_identity"]["design_sha256"] = _design(
        old_design, single.CANDIDATE, "old")
    new["run_identity"]["design_sha256"] = _design(
        new_design, single.SPARE_RAW_CANDIDATE, "new")
    result = analysis.compare(
        old, new, mode="final",
        old_frozen_design=old_design, new_frozen_design=new_design)

    assert result["status"] == "complete"
    assert result["quality"]["paired_official_task_count"] == 20
    assert result["spare_raw"]["metrics"]["event_attempts_added"][
        "strict_fixed_20_new"] == 20


def test_final_rejects_missing_spare_raw_receipt(tmp_path: Path) -> None:
    old = _round(single.CANDIDATE, "old", complete=True)
    new = _round(single.SPARE_RAW_CANDIDATE, "new", complete=True)
    new["per_task"][0]["dependency_packet_spare_raw_exposure"]["missing_receipts"] = 1
    old_design = tmp_path / "old.json"
    new_design = tmp_path / "new.json"
    old["run_identity"]["design_sha256"] = _design(
        old_design, single.CANDIDATE, "old")
    new["run_identity"]["design_sha256"] = _design(
        new_design, single.SPARE_RAW_CANDIDATE, "new")

    with pytest.raises(
            analysis.SpareRawPairedAnalysisError,
            match="do not fully expose valid spare-raw policy"):
        analysis.compare(
            old, new, mode="final",
            old_frozen_design=old_design, new_frozen_design=new_design)
