"""Comparison-table handling for auditable partial benchmark results."""

import json

from .report import write_comparison


def _cell():
    return {
        "cell_id": "appworld__agentkv",
        "benchmark": "appworld",
        "method": "AgentKV",
        "arm": "agentkv",
        "group": "reference",
        "ratio": 8,
    }


def _write_cell(output, summary):
    directory = output / "closed_loop" / "appworld__agentkv"
    directory.mkdir(parents=True)
    (directory / "complete.json").write_text("{}", encoding="utf-8")
    (directory / "measurement_summary.json").write_text(json.dumps({
        "counts": {"requests": 7, "tool_actions": 3},
    }), encoding="utf-8")
    (directory / "summary_agentkv.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )


def test_comparison_marks_infrastructure_partial_as_needs_review(tmp_path):
    _write_cell(tmp_path, {
        "n": 2,
        "n_official_scored": 1,
        "semantic_score": None,
        "score_valid": False,
        "result_status": "completed_with_infrastructure_failures",
        "n_infrastructure_failures": 1,
        "infrastructure_failure_task_ids": ["timeout"],
        "official_aggregate": None,
    })
    row = write_comparison(tmp_path, [_cell()])[0]
    assert row["result_status"] == "completed_with_infrastructure_failures"
    assert row["score_valid"] is False
    assert row["semantic_score"] is None
    assert row["scored_tasks"] == 1
    assert row["n_infrastructure_failures"] == 1
    assert row["infrastructure_failure_task_ids"] == ["timeout"]
    assert row["appworld_task_goal_completion"] is None


def test_comparison_keeps_normal_result_status_and_count(tmp_path):
    _write_cell(tmp_path, {
        "n": 2,
        "semantic_score": 0.5,
        "official_aggregate": {"task_goal_completion": 0.5},
    })
    row = write_comparison(tmp_path, [_cell()])[0]
    assert row["result_status"] == "preliminary, n=1"
    assert row["score_valid"] is None
    assert row["scored_tasks"] == 2
    assert row["semantic_score"] == 0.5
