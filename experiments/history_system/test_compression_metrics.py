import json
from pathlib import Path

import pytest

from experiments.history_system.compression_metrics import (
    analyze_run,
    analyze_task_records,
    linear_quantile,
    weighted_linear_quantile,
)


def _coverage(*, eligible, raw=(), touched=(), fully=(), dropped=()):
    return {
        "eligible_source_indices": list(eligible),
        "raw_source_indices": list(raw),
        "gist_touched_source_indices": list(touched),
        "gist_fully_represented_source_indices": list(fully),
        "unrepresented_source_indices": list(dropped),
        "complete_history_coverage": not dropped,
        "fitted_encoder_input_tokens": len(eligible) * 40,
        "retained_encoder_input_tokens": len(fully) * 40,
    }


def _step(
    h,
    a,
    *,
    s=100,
    decision="d1",
    coverage=None,
    configured_ratio=4,
    guard=None,
):
    if coverage is None:
        coverage = _coverage(eligible=(), raw=(), touched=(), fully=(), dropped=())
    ratio = {
        "schema": "a-same-prefix-compression-ratio-v1",
        "full_history_bytes": h,
        "common_live_bytes": s,
        "active_history_bytes": a,
        "active_gist_bytes": a,
        "active_raw_history_bytes": 0,
        "n_history": h / a if h and a else None,
        "n_total": (s + h) / (s + a) if s + a else None,
        "configured_gist_ratio": configured_ratio,
        "actual_source_to_gist_ratio": 3.5 if a else None,
        "includes_coverage_loss": not coverage["complete_history_coverage"],
    }
    result = {
        "decision_key": decision,
        "generation_trace": [
            {
                "phase": "draft",
                "status": "completed",
                "discarded": False,
                "controller": {
                    "compression_ratio": ratio,
                    "source_coverage": coverage,
                    "history_budget_bytes": 100,
                    "workspace_budget_bytes": 100,
                },
            }
        ],
    }
    if guard is not None:
        result["pre_generation_budget_checks"] = guard
    return result


def test_linear_quantile_is_bounded_for_small_samples():
    assert linear_quantile([], 0.1) is None
    assert linear_quantile([7], 0.1) == 7
    assert linear_quantile([2, 8], 0.1) == pytest.approx(2.6)
    with pytest.raises(ValueError):
        linear_quantile([1], -0.1)
    assert weighted_linear_quantile([(2, 1), (8, 1)], 0.5) == 5


def test_task_metrics_exclude_warmup_and_keep_coverage_loss_in_primary():
    complete = _coverage(eligible=(0, 1), raw=(0,), touched=(1,), fully=(1,))
    lossy = _coverage(
        eligible=(0, 1, 2),
        raw=(0,),
        touched=(1, 2),
        fully=(1,),
        dropped=(2,),
    )
    row = analyze_task_records(
        "task-a",
        [
            _step(0, 0, decision="warmup"),
            _step(40, 20, decision="full", coverage=complete),
            _step(
                80,
                10,
                decision="lossy",
                coverage=lossy,
                guard={"status": "passed"},
            ),
        ],
        b0_cap_bytes=15,
    )

    assert row["activation"]["no_history_warmup_receipts"] == 1
    assert row["activation"]["activated_receipts"] == 2
    assert row["system_active_history_reduction"]["median"] == 5
    assert row["system_active_history_reduction"]["low_quantile"] == pytest.approx(2.6)
    assert row["system_active_history_reduction"]["coverage_loss_receipts"] == 1
    strict = row["strict_complete_coverage_subset"]
    assert strict["activated_receipts"] == 1
    assert strict["system_active_history_reduction"]["median"] == 2
    assert row["source_coverage"]["eligible_source_occurrences"] == 5
    assert row["source_coverage"]["fully_represented_source_occurrences"] == 4
    assert row["source_coverage"]["unrepresented_source_occurrences"] == 1
    assert row["source_coverage"][
        "gist_touched_but_unrepresented_source_occurrences"
    ] == 1
    assert row["active_history"]["observed_peak_bytes"] == 20
    assert row["active_history"]["observed_b0_violation_count"] == 1
    assert row["configured_gist_ratios_observed"] == [4.0]
    assert row["pre_generation_budget_guard"]["known_passes"] == 1


def test_run_uses_task_equal_summaries_and_preserves_missing_task(tmp_path: Path):
    root = tmp_path / "returned"
    short = root / "task_shards" / "short" / "server" / "steps.jsonl"
    long = root / "task_shards" / "long" / "server" / "steps.jsonl"
    short.parent.mkdir(parents=True)
    long.parent.mkdir(parents=True)
    short.write_text(json.dumps(_step(20, 10)) + "\n", encoding="utf-8")
    long.write_text(
        "\n".join(json.dumps(_step(100, 10, decision=f"d{i}")) for i in range(5))
        + "\n",
        encoding="utf-8",
    )

    complete = analyze_run(root, ["short", "long"], b0_cap_bytes=100)
    metric = complete["task_equal"]["system_active_history_reduction"]
    assert metric["all_selected_tasks_represented"] is True
    assert metric["all_selected_task_mean_of_per_task_medians"] == 6
    assert metric["all_selected_task_mean_of_per_task_low_quantiles"] == 6
    weighted = metric["task_equal_weighted_activated_receipts"]
    assert weighted["all_selected_task_weighted_median"] == 6
    assert weighted["all_selected_task_weighted_low_quantile"] == 2
    assert complete["source_coverage"]["activated_receipts"] == 6

    missing = analyze_run(root, ["short", "long", "missing"], b0_cap_bytes=100)
    metric = missing["task_equal"]["system_active_history_reduction"]
    assert missing["fixed_task_denominator"] == 3
    assert missing["receipt_completeness"]["tasks_with_source_file"] == 2
    assert missing["receipt_completeness"]["tasks_with_no_activated_measurement"] == [
        "missing"
    ]
    assert metric["tasks_contributing"] == 2
    assert metric["observed_known_task_mean_of_per_task_medians"] == 6
    assert metric["all_selected_task_mean_of_per_task_medians"] is None


def test_anomalies_malformed_rows_and_controllerless_steps_are_visible(tmp_path: Path):
    root = tmp_path / "returned"
    path = root / "task_shards" / "task-a" / "server" / "steps.jsonl"
    path.parent.mkdir(parents=True)
    anomalous = _step(10, 0)
    path.write_text(
        json.dumps(anomalous)
        + "\n"
        + "not json\n"
        + json.dumps({"status": "failed", "generation_trace": []})
        + "\n",
        encoding="utf-8",
    )

    result = analyze_run(root, ["task-a"], b0_cap_bytes=100)
    task = result["tasks"][0]
    assert task["source"]["malformed_records"] == 1
    assert task["source"]["malformed_line_numbers"] == [2]
    assert task["source"]["controllerless_step_rows"] == 1
    assert task["activation"][
        "full_history_without_active_footprint_anomalies"
    ] == 1
    assert result["receipt_completeness"][
        "full_history_without_active_footprint_anomalies"
    ] == 1
    assert result["task_equal"]["system_active_history_reduction"][
        "all_selected_task_mean_of_per_task_medians"
    ] is None
