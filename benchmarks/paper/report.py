"""Offline comparison table, keeping closed-loop and shared-prefix costs separate."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def write_comparison(output: Path, plan):
    rows = []
    for stage in ("closed_loop", "common_prefix"):
        for cell in plan:
            directory = output / stage / cell["cell_id"]
            path = directory / "measurement_summary.json"
            if not path.exists():
                continue
            measured = json.loads(path.read_text())
            ratios = "common_prefix_token_ratios" if stage == "common_prefix" else "token_ratios"
            score_path = directory / ("summary_" + cell["arm"] + ".json")
            scores = json.loads(score_path.read_text()) if score_path.exists() else {}
            row = {key: cell.get(key) for key in ("cell_id", "benchmark", "method", "arm", "group", "ratio", "retention")}
            row.update(stage=stage, result_status="preliminary, n=1",
                       semantic_score=scores.get("semantic_score"),
                       appworld_task_goal_completion=nested(scores, "official_aggregate", "task_goal_completion"),
                       appworld_scenario_goal_completion=nested(scores, "official_aggregate", "scenario_goal_completion"),
                       scored_tasks=scores.get("n_scored", scores.get("n")),
                       requests=nested(measured, "counts", "requests"),
                       committed_actions=nested(measured, "counts", "tool_actions"),
                       resident_kv_peak_bytes=nested(measured, "memory", "request_peak_resident_kv_bytes", "max"),
                       generation_active_kv_peak_bytes=nested(measured, "memory", "generation_active_kv_bytes", "max"),
                       torch_allocated_peak_bytes=nested(measured, "memory", "torch_peak_allocated_bytes", "max"),
                       torch_reserved_peak_bytes=nested(measured, "memory", "torch_peak_reserved_bytes", "max"),
                       nvml_process_sampled_peak_bytes=nested(measured, "memory", "nvml_process_peak_used_bytes", "max"),
                       model_ms_per_committed_action=nested(measured, "latency_ms", "complete_model_side_per_committed_action", "mean_ms"),
                       model_request_mean_ms=nested(measured, "latency_ms", "request_algorithm", "mean"),
                       model_request_p95_ms=nested(measured, "latency_ms", "request_algorithm", "p95"),
                       model_request_p99_ms=nested(measured, "latency_ms", "request_algorithm", "p99"),
                       observed_request_mean_ms=nested(measured, "latency_ms", "request", "mean"),
                       external_tool_mean_ms=nested(measured, "latency_ms", "external_tool", "mean"),
                       episode_mean_ms=nested(measured, "latency_ms", "episode_wall", "mean"),
                       whole_retained_fraction=nested(measured, ratios, "whole", "ratio_of_sums"),
                       history_retained_fraction=nested(measured, ratios, "history", "ratio_of_sums"),
                       measurement_file=str(path), official_score_file=str(score_path) if score_path.exists() else None)
            rows.append(row)
    full = {(r["stage"], r["benchmark"]): r for r in rows if r["arm"] == "full"}
    for row in rows:
        baseline = full.get((row["stage"], row["benchmark"]), {})
        for metric in ("resident_kv_peak_bytes", "torch_allocated_peak_bytes", "nvml_process_sampled_peak_bytes"):
            actual, reference = row[metric], baseline.get(metric)
            row[metric + "_saving_vs_full_pct"] = (
                100 * (1 - actual / reference)
                if actual is not None and reference is not None and reference > 0 else None)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    if rows:
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows
