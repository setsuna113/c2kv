"""Outcome-semantics tests for the pre-B paired official analysis."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks"))
from memory_runtime.analyze_pre_b_official import AnalysisInputError, analyze


TASKS = ["t0", "t1", "t2", "t3"]


def _metric(passed, generations, extractions):
    return {
        "official_pass": passed,
        "request_count": generations,
        "generation_attempts": generations,
        "extraction_attempts": extractions,
        "prompt_tokens": generations * 10,
        "completion_tokens": generations * 2,
        "proxy_wall_seconds": generations / 2,
        "task_wall_seconds": generations,
        "generation_resources": {
            "controller_wall_seconds": 0.25,
            "extraction_producer_calls": extractions,
        },
    }


def _official(passed, generations, extractions):
    return {
        "status": "official_terminal",
        "valid": True,
        "official_pass": passed,
        "official_score_known": True,
        "generation_attempts": generations,
        "extraction_attempts": extractions,
    }


def _report():
    full = {
        "t0": _official(True, 2, 0),
        "t1": _official(False, 3, 0),
        "t2": _official(True, 4, 0),
        "t3": _official(False, 5, 0),
    }
    method = {
        "t0": _official(False, 2, 1),
        "t1": _official(True, 3, 2),
        "t2": {
            "status": "capacity_infeasible",
            "valid": True,
            "official_pass": None,
            "official_score_known": False,
            "generation_attempts": 0,
            "extraction_attempts": 3,
        },
        "t3": {
            "status": "missing",
            "valid": False,
            "official_pass": None,
            "official_score_known": False,
            "path": "task_shards/t3/method",
        },
    }
    full_metrics = {
        task_id: _metric(cell["official_pass"], cell["generation_attempts"], 0)
        for task_id, cell in full.items()
    }
    method_metrics = {
        "t0": _metric(False, 2, 1),
        "t1": _metric(True, 3, 2),
        "t2": {
            "official_pass": None,
            "official_score_known": False,
            "operational_status": "capacity_infeasible",
            "generation_attempts": 0,
            "extraction_attempts": 3,
        },
    }
    return {
        "schema": "a-runtime-bfcl-official-collection-v1",
        "status": "partial",
        "valid_for_method_comparison": False,
        "valid_for_pre_b_delivery": False,
        "errors": [],
        "manifest": {
            "design": "pre-b-p3",
            "task_ids": TASKS,
            "expected_variants": ["full", "method"],
            "execution_inputs": {
                "checkpoint": {"profile_fingerprint": "checkpoint"},
                "data": {"sha256": "data"},
                "scorer": {"sha256": "scorer"},
            },
        },
        "task_matrix": [
            {"task_id": task_id, "arms": {"full": full[task_id], "method": method[task_id]}}
            for task_id in TASKS
        ],
        "performance": {
            "full": {"task_metrics": full_metrics},
            "method": {"task_metrics": method_metrics},
        },
    }


def test_official_failure_method_failure_and_missing_remain_distinct():
    result = analyze(_report())

    method = result["outcome_coverage"]["method"]
    assert method["counts"] == {
        "official_failure": 1,
        "official_pass": 1,
        "method_failure": 1,
        "missing": 1,
    }
    assert method["official_scored_coverage"] == {
        "numerator": 2,
        "denominator": 4,
        "rate": 0.5,
    }
    rows = {row["task_id"]: row for row in result["task_rows"]}
    assert rows["t0"]["arms"]["method"]["outcome_kind"] == "official_failure"
    assert rows["t2"]["arms"]["method"]["outcome_kind"] == "method_failure"
    assert rows["t2"]["arms"]["method"]["official_pass"] is None
    assert rows["t3"]["arms"]["method"]["outcome_kind"] == "missing"
    assert rows["t3"]["arms"]["method"]["cost"]["generation_attempts"] is None

    pair = result["paired_with_full"]["method"]
    assert pair["paired_official_table"] == {
        "full_pass_method_pass": 0,
        "full_pass_method_official_failure": 1,
        "full_official_failure_method_pass": 1,
        "full_official_failure_method_official_failure": 0,
    }
    assert pair["unpaired"] == {
        "method_terminal_failure_pair": 1,
        "missing_pair": 1,
    }
    assert pair["full_success_retention"]["complete_rate"] is None
    assert pair["full_success_retention"]["bounds_over_all_eligible_full_tasks"] == [0.0, 0.5]
    assert pair["reverse_rescue_on_full_failures"]["count"] == 1
    assert pair["reverse_rescue_on_full_failures"]["complete_rate"] is None
    assert pair["reverse_rescue_on_full_failures"]["bounds_over_all_eligible_full_tasks"] == [0.5, 1.0]


def test_p4a_requires_matching_explicit_p3_full_reference():
    reference = _report()
    p4a = copy.deepcopy(reference)
    p4a["manifest"]["design"] = "pre-b-p4a"
    p4a["manifest"]["expected_variants"] = ["method"]
    for row in p4a["task_matrix"]:
        row["arms"] = {"method": row["arms"]["method"]}
    p4a["performance"] = {"method": p4a["performance"]["method"]}

    result = analyze(p4a, full_reference=reference)

    assert result["full_baseline_source"]["kind"] == "external_p3_reference"
    assert result["paired_with_full"]["method"]["paired_official_table"][
        "full_official_failure_method_pass"
    ] == 1


@pytest.mark.parametrize("task_id", ["t2", "t3"])
def test_unscored_cells_cannot_be_rewritten_as_official_failures(task_id):
    report = _report()
    row = next(item for item in report["task_matrix"] if item["task_id"] == task_id)
    row["arms"]["method"]["official_pass"] = False

    with pytest.raises(AnalysisInputError, match="official outcome|terminal outcome"):
        analyze(report)
