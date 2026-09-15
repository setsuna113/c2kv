import copy
import json
from pathlib import Path

import pytest

from benchmarks.memory_runtime import analyze_s1_field_priority_pair as paired
from benchmarks.memory_runtime import analyze_s1_long_stage as single


TASKS = [f"multi_turn_long_context_{index * 10}" for index in range(20)]


def _analysis(
    run_id: str,
    *,
    scored_tasks: int = 20,
    policy: str | None = paired.EXPECTED_NEW_POLICY,
) -> dict:
    rows = []
    for index, task_id in enumerate(TASKS):
        known = index < scored_tasks
        correct = int(known and index in {0, 3, 7})
        explicit = int(policy is not None)
        implicit = int(policy is None)
        rows.append({
            "task_id": task_id,
            "quality": {
                "official_score_known": known,
                "correct_count": correct if known else None,
            },
            "cost": {
                "request_count": index + 1 if known else None,
                "ok_request_count": index + 1 if known else None,
                "error_request_count": 0 if known else None,
                "task_wall_seconds": float(index + 2) if known else None,
                "proxy_request_wall_seconds": float(index + 1) if known else None,
                "generation": {
                    "generation_attempts": index + 1 if known else None,
                    "known_prompt_tokens": (index + 1) * 10 if known else None,
                    "known_completion_tokens": index + 1 if known else None,
                },
                "resources": {
                    "extraction_lookups": index + 1 if known else None,
                    "extraction_client_cache_hits": index if known else None,
                    "extraction_producer_calls": 1 if known else None,
                    "extraction_producer_successes": 1 if known else None,
                    "extraction_producer_failures": 0 if known else None,
                    "extraction_lookup_wall_seconds": 0.1 if known else None,
                    "extraction_producer_wall_seconds": 0.1 if known else None,
                    "controller_wall_seconds": 0.2 if known else None,
                    "compressed_assembly_wall_seconds": 0.3 if known else None,
                    "gist_tokens_sum": index + 4 if known else None,
                    "active_history_bytes_max": 100 + index if known else None,
                    "evidence_bytes_max": 50 + index if known else None,
                },
            },
            "source_coverage": {
                "known_receipts": 1 if known else 0,
                "missing_receipts": 0,
                "eligible_source_occurrence_appearances": 2 if known else 0,
                "fully_represented_source_occurrence_appearances": 1 if known else 0,
                "unrepresented_source_occurrence_appearances": 1 if known else 0,
                "gist_fully_represented_source_occurrence_appearances": 1 if known else 0,
                "raw_source_occurrence_appearances": 0,
                "fitted_fragment_appearances": 1 if known else 0,
                "retained_fragment_appearances": 1 if known else 0,
                "fitted_encoder_input_tokens": 8 if known else 0,
                "retained_encoder_input_tokens": 8 if known else 0,
            },
            "dependency_packet_exposure": {
                "known_receipts": 1 if known else 0,
                "missing_receipts": 0,
                "requests_with_nonempty_packet": 1 if known else 0,
                "requests_without_nonempty_packet": 0,
                "retained_fact_appearances": 1 if known else 0,
                "eligible_fact_appearances": 2 if known else 0,
                "field_limit_omission_appearances": 1 if known else 0,
                "unrepresented_requested_source_appearances": 0,
                "workspace_drop_appearances": 0,
                "requests_with_explicit_field_priority_policy": explicit if known else 0,
                "requests_with_legacy_implicit_field_priority": implicit if known else 0,
                "field_priority_policies": [policy] if policy is not None and known else [],
            },
        })
    correct_count = sum(row["quality"]["correct_count"] or 0 for row in rows)
    missing_ids = TASKS[scored_tasks:]
    return {
        "schema": single.SCHEMA,
        "status": "complete" if scored_tasks == 20 else "partial",
        "mode": "final" if scored_tasks == 20 else "partial",
        "run_identity": {
            "run_id": run_id,
            "candidate_id": single.CANDIDATE,
            "task_ids": TASKS,
        },
        "quality": {
            "fixed_task_denominator": 20,
            "official_scored_tasks": scored_tasks,
            "official_missing_tasks": 20 - scored_tasks,
            "official_missing_task_ids": missing_ids,
            "official_correct_tasks": correct_count,
        },
        "recovery": {
            "missing_task_count": 0 if scored_tasks == 20 else 20 - scored_tasks,
            "completion_errors": [] if scored_tasks == 20 else [
                {"task_id": task_id, "errors": ["missing_shard"]}
                for task_id in missing_ids
            ],
        },
        "paired_with_s0_bounded_latest": {
            "paired_task_count": scored_tasks,
            "outcome_counts_over_fixed_20": {},
        },
        "per_task": rows,
    }


def test_self_comparison_keeps_fixed_denominator_and_has_zero_deltas() -> None:
    source = _analysis("same-fixture")
    result = paired.compare(
        source,
        copy.deepcopy(source),
        mode="final",
        allow_self_comparison_fixture=True,
    )

    assert result["status"] == "complete"
    assert result["scope"]["fixed_task_denominator"] == 20
    assert result["quality"]["paired_official_task_count"] == 20
    assert result["quality"]["outcome_counts_over_fixed_20"] == {
        "both_correct": 3,
        "both_incorrect": 17,
    }
    assert result["quality"]["new_only_correct_task_ids"] == []
    assert result["quality"]["old_only_correct_task_ids"] == []
    assert all(
        receipt["new_minus_old_on_paired"] == 0
        for receipt in result["cost"].values()
    )
    assert all(
        receipt["new_minus_old_on_paired"] == 0
        for receipt in result["source_coverage"]["metrics"].values()
    )
    assert result["dependency_packet"]["new_policy_exposure"]["status"] == (
        "expected_policy_observed")


def test_partial_preserves_missing_cells_and_final_rejects_them() -> None:
    old = _analysis("old", policy=None)
    new = _analysis("new", scored_tasks=1)

    result = paired.compare(old, new, mode="partial")

    assert result["status"] == "partial"
    assert result["quality"]["new"]["official_missing_tasks"] == 19
    assert result["quality"]["paired_official_task_count"] == 1
    assert result["quality"]["outcome_counts_over_fixed_20"] == {
        "both_correct": 1,
        "unpaired_missing_new": 19,
    }
    assert result["cost"]["request_count"]["new_known_task_count"] == 1
    assert result["cost"]["request_count"]["strict_fixed_20_new"] is None
    assert result["dependency_packet"]["new_policy_exposure"]["status"] == (
        "expected_policy_observed")

    with pytest.raises(paired.S1PairedAnalysisError, match="final mode requires"):
        paired.compare(old, new, mode="final")


def test_final_requires_actual_policy_receipt_exposure() -> None:
    old = _analysis("old", policy=None)
    new = _analysis("new")
    for row in new["per_task"]:
        packet = row["dependency_packet_exposure"]
        packet.pop("field_priority_policies")
        packet.pop("requests_with_explicit_field_priority_policy")
        packet.pop("requests_with_legacy_implicit_field_priority")

    partial = paired.compare(old, new, mode="partial")
    assert partial["dependency_packet"]["new_policy_exposure"]["status"] == (
        "not_reported_by_source_analysis")
    with pytest.raises(paired.S1PairedAnalysisError, match="policy exposure"):
        paired.compare(old, new, mode="final")


def test_same_run_requires_explicit_fixture_opt_in() -> None:
    source = _analysis("same")
    with pytest.raises(paired.S1PairedAnalysisError, match="self-comparison fixture"):
        paired.compare(source, source, mode="partial")


def test_frozen_design_binds_run_and_ordered_fixed_tasks(tmp_path: Path) -> None:
    source = _analysis("bound-run")
    design = tmp_path / "design.frozen.json"
    design.write_text(json.dumps({
        "schema": paired.FROZEN_DESIGN_SCHEMA,
        "status": "frozen",
        "candidate_id": single.CANDIDATE,
        "run_id": "bound-run",
        "task_ids": TASKS,
    }) + "\n", encoding="utf-8")

    result = paired.compare(
        source,
        copy.deepcopy(source),
        mode="final",
        allow_self_comparison_fixture=True,
        old_frozen_design=design,
        new_frozen_design=design,
    )
    assert result["provenance"]["old_frozen_design"]["sha256"]
    assert result["provenance"]["new_frozen_design"]["sha256"]

    bad = json.loads(design.read_text(encoding="utf-8"))
    bad["run_id"] = "other-run"
    design.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(paired.S1PairedAnalysisError, match="does not bind"):
        paired.compare(
            source,
            copy.deepcopy(source),
            mode="final",
            allow_self_comparison_fixture=True,
            old_frozen_design=design,
        )
