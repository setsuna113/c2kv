"""Offline comparison table, keeping closed-loop and shared-prefix costs separate."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .runner import is_native_arm
from .task_subsets import is_subset


EXACT_OUTPUT_ARMS = {"agentkv", "commitkv"}


def uses_c2kv_pool(arm):
    return arm == "c2kv4" or is_native_arm(arm)


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
            if (not path.exists() or not (directory / "complete.json").is_file()
                    or is_subset(cell, directory)
                    or (directory / "AUDIT_EXCLUSION.json").exists()
                    or (stage == "common_prefix" and cell["arm"] in EXACT_OUTPUT_ARMS)):
                continue
            measured = json.loads(path.read_text())
            conversion_path = directory / "c1_conversion.json"
            conversion = (json.loads(conversion_path.read_text())
                          if conversion_path.is_file() else {})
            ratios = "common_prefix_token_ratios" if stage == "common_prefix" else "token_ratios"
            score_path = directory / ("summary_" + cell["arm"] + ".json")
            scores = json.loads(score_path.read_text()) if score_path.exists() else {}
            infrastructure_failures = int(scores.get("n_infrastructure_failures") or 0)
            score_valid = scores.get("score_valid")
            result_status = "preliminary, n=1"
            if score_valid is False or infrastructure_failures:
                result_status = (
                    scores.get("result_status")
                    or "needs_review_infrastructure_failure"
                )
            row = {key: cell.get(key) for key in ("cell_id", "benchmark", "method", "arm", "group", "ratio", "retention", "tool_context", "history_budget_tokens")}
            row.update(stage=stage, result_status=result_status,
                       comparison_basis=("own_output_closed_loop" if stage == "closed_loop"
                                         else "Full_teacher_forced_prefix_target_policy"),
                       sampling_contract=("target_greedy_overrides_recorded_Full_temperature"
                                          if stage == "common_prefix" and is_native_arm(cell["arm"])
                                          and cell["benchmark"] in {"bfcl_base", "bfcl_long_context"}
                                          else "target_arm_recorded_contract"),
                       unattributed_native_server_requests=nested(
                           conversion, "unattributed_native_engine_work", "server_requests"),
                       unattributed_native_server_request_duration_coverage=nested(
                           conversion, "unattributed_native_engine_work",
                           "server_request_duration_coverage"),
                       unattributed_native_server_request_duration_ms=(
                           nested(conversion, "unattributed_native_engine_work",
                                  "server_request_duration_ns_sum") / 1e6
                           if nested(conversion, "unattributed_native_engine_work",
                                     "server_request_duration_ns_sum") is not None else None),
                       semantic_score=scores.get("semantic_score"),
                       score_valid=score_valid,
                       n_infrastructure_failures=infrastructure_failures,
                       infrastructure_failure_task_ids=scores.get(
                           "infrastructure_failure_task_ids"),
                       appworld_task_goal_completion=nested(scores, "official_aggregate", "task_goal_completion"),
                       appworld_scenario_goal_completion=nested(scores, "official_aggregate", "scenario_goal_completion"),
                       scored_tasks=scores.get(
                           "n_official_scored", scores.get("n_scored", scores.get("n"))),
                       requests=nested(measured, "counts", "requests"),
                       replay_attempted_prefixes=nested(measured, "counts", "prefix_replays_attempted"),
                       replay_successful_prefixes=nested(measured, "counts", "prefix_replays_successful"),
                       replay_failed_attempted_prefixes=nested(measured, "counts", "prefix_replays_failed_attempted"),
                       replay_unattempted_prefixes=nested(measured, "counts", "prefix_replays_unattempted"),
                       committed_actions=nested(measured, "counts", "tool_actions"),
                       resident_kv_peak_bytes=nested(measured, "memory", "request_peak_resident_kv_bytes", "max"),
                       resident_kv_peak_c2kv_cache_accounting_available=nested(
                           measured, "memory", "resident_peak_chain",
                           "request_peak_c2kv_cache_accounting_available"),
                       # Line items of the resident total, reported alongside it:
                       cached_evictable_kv_at_resident_peak_bytes=nested(measured, "memory", "resident_peak_chain", "request_peak_cached_evictable_kv_bytes"),
                       c2kv_cached_evictable_kv_at_resident_peak_bytes=nested(
                           measured, "memory", "resident_peak_chain",
                           "request_peak_c2kv_cached_evictable_kv_bytes"),
                       cached_evictable_kv_peak_bytes=nested(measured, "memory", "cached_evictable_kv_peak_bytes", "max"),
                       generation_active_kv_peak_bytes=nested(measured, "memory", "generation_active_kv_bytes", "max"),
                       torch_allocated_peak_bytes=nested(measured, "memory", "torch_peak_allocated_bytes", "max"),
                       torch_reserved_peak_bytes=nested(measured, "memory", "torch_peak_reserved_bytes", "max"),
                       nvml_process_sampled_peak_bytes=nested(measured, "memory", "nvml_process_peak_used_bytes", "max"),
                       model_ms_per_committed_action=nested(measured, "latency_ms", "complete_model_side_per_committed_action_excluding_gist", "mean_ms"),
                       model_ms_per_committed_action_including_gist=nested(measured, "latency_ms", "complete_model_side_per_committed_action", "mean_ms"),
                       gist_generation_total_ms=(nested(measured, "latency_ms", "gist_generation", "total_ns") / 1e6
                                                 if nested(measured, "latency_ms", "gist_generation", "total_ns") is not None else None),
                       model_request_excluding_gist_mean_ms=nested(measured, "latency_ms", "request_algorithm_excluding_gist", "mean"),
                       model_request_excluding_gist_p95_ms=nested(measured, "latency_ms", "request_algorithm_excluding_gist", "p95"),
                       model_request_excluding_gist_p99_ms=nested(measured, "latency_ms", "request_algorithm_excluding_gist", "p99"),
                       model_request_mean_ms=nested(measured, "latency_ms", "request_algorithm", "mean"),
                       model_request_p95_ms=nested(measured, "latency_ms", "request_algorithm", "p95"),
                       model_request_p99_ms=nested(measured, "latency_ms", "request_algorithm", "p99"),
                       observed_request_mean_ms=nested(measured, "latency_ms", "request", "mean"),
                       external_tool_mean_ms=nested(measured, "latency_ms", "external_tool", "mean"),
                       episode_mean_ms=nested(measured, "latency_ms", "episode_wall", "mean"),
                       whole_retained_fraction=nested(measured, ratios, "whole", "ratio_of_sums"),
                       history_retained_fraction=nested(measured, ratios, "history", "ratio_of_sums"),
                       measurement_file=str(path), official_score_file=str(score_path) if score_path.exists() else None)
            if (uses_c2kv_pool(cell["arm"])
                    and row["resident_kv_peak_c2kv_cache_accounting_available"] is not True):
                row["cached_evictable_kv_at_resident_peak_bytes"] = None
                row["cached_evictable_kv_peak_bytes"] = None
            rows.append(row)
    full = {(r["stage"], r["benchmark"]): r for r in rows if r["arm"] == "full"}
    for row in rows:
        baseline = full.get((row["stage"], row["benchmark"]), {})
        for metric in ("resident_kv_peak_bytes", "torch_allocated_peak_bytes", "nvml_process_sampled_peak_bytes"):
            actual, reference = row[metric], baseline.get(metric)
            if (metric == "resident_kv_peak_bytes" and uses_c2kv_pool(row["arm"])
                    and (row["resident_kv_peak_c2kv_cache_accounting_available"] is not True
                         or baseline.get("resident_kv_peak_c2kv_cache_accounting_available") is not True)):
                row[metric + "_saving_vs_full_pct"] = None
                continue
            row[metric + "_saving_vs_full_pct"] = (
                100 * (1 - actual / reference)
                if actual is not None and reference is not None and reference > 0 else None)
    (output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    if not rows:
        # Do not leave a stale CSV after an audit excludes the last row.
        (output / "comparison.csv").write_text("", encoding="utf-8")
    if rows:
        with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows
