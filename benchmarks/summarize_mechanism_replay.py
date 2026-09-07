#!/usr/bin/env python3
"""Summarize a mechanism replay JSONL without changing its measurements."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SUMMARY_SCHEMA_VERSION = "mechanism_replay_summary_v1"
EXPECTED_REPLAY_SCHEMA_VERSION = "mechanism_replay_v1"
PRELIMINARY_LABEL = "preliminary, n=1"
OUTPUT_NAMES = (
    "case_arm.csv",
    "arm_summary.csv",
    "arm_summary.json",
)


class SummaryError(RuntimeError):
    """The replay cannot be summarized unambiguously."""


def _number(value: Any) -> Optional[float | int]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(float(value)):
        return None
    return value


def _integer(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, dict) else {}


def _deep_get(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _sum_available(values: Iterable[Any]) -> Optional[float | int]:
    available = [value for value in values if _number(value) is not None]
    return sum(available) if available else None


def _median_available(values: Iterable[Any]) -> Optional[float]:
    available = [float(value) for value in values if _number(value) is not None]
    return statistics.median(available) if available else None


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return numerator / denominator if denominator else None


def _ratio(numerator: Any, denominator: Any) -> Optional[float]:
    top = _number(numerator)
    bottom = _number(denominator)
    if top is None or bottom is None or bottom <= 0:
        return None
    return float(top) / float(bottom)


def _same_number(left: Any, right: Any) -> Optional[bool]:
    left_number = _number(left)
    right_number = _number(right)
    if left_number is None or right_number is None:
        return None
    return math.isclose(
        float(left_number), float(right_number), rel_tol=0.0, abs_tol=1e-9)


def _read_replay(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    raise SummaryError(f"blank replay line {line_number}")
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise SummaryError(
                        f"invalid JSON at replay line {line_number}: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise SummaryError(
                        f"replay line {line_number} must be a JSON object")
                rows.append({"line": line_number, "value": value})
    except OSError as error:
        raise SummaryError(f"cannot read replay {path}: {error}") from error
    if not rows:
        raise SummaryError("replay is empty")
    return rows


def _unique_row(
    rows: Sequence[Mapping[str, Any]], row_type: str,
) -> Mapping[str, Any]:
    matches = [row for row in rows if row["value"].get("row_type") == row_type]
    if len(matches) != 1:
        raise SummaryError(
            f"expected exactly one {row_type} row, found {len(matches)}")
    return matches[0]


def _required_unique_strings(value: Any, label: str) -> List[str]:
    if (not isinstance(value, list) or not value
            or any(not isinstance(item, str) or not item for item in value)):
        raise SummaryError(f"{label} must be a non-empty list of strings")
    if len(value) != len(set(value)):
        raise SummaryError(f"{label} contains duplicates")
    return list(value)


def _timing_scope(shared_device: str) -> str:
    scopes = {
        "true": "shared_device_observed_wall_time",
        "false": "non_shared_device_declared_observed_wall_time",
        "unknown": (
            "device_sharing_unknown_observed_wall_time_no_exclusive_claim"),
    }
    return scopes[shared_device]


def _http_events(
    trial_record: Mapping[str, Any], errors: List[str],
) -> List[Dict[str, Any]]:
    trial = trial_record["value"]
    raw_events = trial.get("http")
    if not isinstance(raw_events, list):
        errors.append(
            f"line {trial_record['line']}: trial http must be a list")
        raw_events = []
    output: List[Dict[str, Any]] = []
    for index, raw_event in enumerate(raw_events, 1):
        if not isinstance(raw_event, dict):
            errors.append(
                f"line {trial_record['line']}: HTTP event {index} must be an object")
            event: Mapping[str, Any] = {}
        else:
            event = raw_event
        kind = event.get("kind")
        status = event.get("status")
        path = event.get("path")
        if kind not in ("policy_generation", "auxiliary"):
            errors.append(
                f"line {trial_record['line']}: HTTP event {index} has invalid kind")
        if status not in ("ok", "failed", "started"):
            errors.append(
                f"line {trial_record['line']}: HTTP event {index} has invalid status")
        if not isinstance(path, str) or not path:
            errors.append(
                f"line {trial_record['line']}: HTTP event {index} lacks path")
        wall_sec = event.get("wall_sec")
        if wall_sec is not None and _number(wall_sec) is None:
            errors.append(
                f"line {trial_record['line']}: HTTP event {index} has invalid wall_sec")
        output.append({
            "label": PRELIMINARY_LABEL,
            "replay_line": trial_record["line"],
            "run_id": trial.get("run_id"),
            "trial_id": trial.get("trial_id"),
            "case_id": trial.get("case_id"),
            "arm": _deep_get(trial, "intervention", "name"),
            "event_index_in_trial": index,
            "ordinal": event.get("ordinal"),
            "path": path,
            "kind": kind,
            "status": status,
            "http_status": event.get("http_status"),
            "wall_sec": wall_sec,
            "attempt": event.get("attempt"),
            "automatic_retries": event.get("automatic_retries"),
        })
        ordinal = _integer(event.get("ordinal"))
        if ordinal is None or ordinal < 1:
            errors.append(
                f"line {trial_record['line']}: HTTP event {index} has invalid ordinal")
        if _integer(event.get("attempt")) != 1:
            errors.append(
                f"line {trial_record['line']}: HTTP event {index} attempt is not 1")
        if _integer(event.get("automatic_retries")) != 0:
            errors.append(
                f"line {trial_record['line']}: HTTP event {index} retries are not 0")
        if (kind == "policy_generation"
                and isinstance(path, str)
                and not path.rstrip("/").endswith("/v1/chat/completions")):
            errors.append(
                f"line {trial_record['line']}: policy HTTP event {index} has wrong path")
        if (kind == "auxiliary"
                and isinstance(path, str)
                and path.rstrip("/").endswith("/v1/chat/completions")):
            errors.append(
                f"line {trial_record['line']}: auxiliary HTTP event {index} has policy path")
        if status == "started":
            errors.append(
                f"line {trial_record['line']}: HTTP event {index} is unfinished")
    recorded_http_count = _integer(trial.get("http_request_count"))
    if recorded_http_count != len(raw_events):
        errors.append(
            f"line {trial_record['line']}: recorded http_request_count "
            f"{recorded_http_count!r} != observed {len(raw_events)}")
    policy_count = sum(
        event["kind"] == "policy_generation" for event in output)
    recorded_policy_count = _integer(trial.get("policy_generation_count"))
    if recorded_policy_count != policy_count:
        errors.append(
            f"line {trial_record['line']}: recorded policy_generation_count "
            f"{recorded_policy_count!r} != observed {policy_count}")
    return output


def _uncompressed_status(trial: Mapping[str, Any]) -> Tuple[Optional[bool], str]:
    arm = _deep_get(trial, "intervention", "name")
    if arm == "full":
        return True, "full_reference"
    if _deep_get(trial, "protection", "no_compression") is True:
        return True, "protection_identity"
    before = _deep_get(
        trial, "measurement", "tensor_bytes",
        "full_equivalent_selected_history_bytes")
    after = _deep_get(
        trial, "measurement", "tensor_bytes", "active_history_bytes")
    if _number(before) is not None and _number(after) is not None:
        if _same_number(before, after):
            return True, "selected_history_before_equals_after"
        return False, "selected_history_before_differs_from_after"
    return None, "measurement_unavailable"


def _case_row(
    trial_record: Mapping[str, Any], shared_device: str,
    timing_scope: str, errors: List[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    trial = trial_record["value"]
    intervention = _mapping(trial.get("intervention"))
    identity = _mapping(trial.get("identity"))
    action = _mapping(trial.get("action"))
    paired = _mapping(trial.get("paired_full_agreement"))
    legacy = _mapping(trial.get("legacy_full_descriptive_agreement"))
    timing = _mapping(trial.get("timing"))
    measurement = _mapping(trial.get("measurement"))
    injection = _mapping(measurement.get("injection"))
    tensor = _mapping(measurement.get("tensor_bytes"))
    http_events = _http_events(trial_record, errors)

    arm = intervention.get("name")
    comparison_scope = (
        "full_self_reference" if arm == "full"
        else "method_vs_paired_full_reference")
    prompt_tokens_raw = measurement.get("prompt_tokens_raw")
    gist_tokens_actual = injection.get("gist_tokens_actual")
    repair_tokens_actual = injection.get("repair_tokens_actual")
    logical_components = (
        prompt_tokens_raw, gist_tokens_actual, repair_tokens_actual)
    logical_recomputed: Optional[float | int]
    if all(_number(value) is not None for value in logical_components):
        logical_recomputed = sum(logical_components)  # type: ignore[arg-type]
    else:
        logical_recomputed = None
    logical_recorded = tensor.get("logical_active_prompt_tokens")
    bytes_per_token = tensor.get("bytes_per_kv_token")
    logical_bytes_recorded = tensor.get("logical_active_prompt_bytes")
    logical_bytes_recomputed: Optional[float | int] = None
    if (_number(logical_recomputed) is not None
            and _number(bytes_per_token) is not None):
        logical_bytes_recomputed = logical_recomputed * bytes_per_token
    token_match = _same_number(logical_recomputed, logical_recorded)
    byte_match = _same_number(logical_bytes_recomputed, logical_bytes_recorded)
    if trial.get("status") == "ok" and token_match is False:
        errors.append(
            f"line {trial_record['line']}: logical prompt tokens do not equal "
            "prompt_tokens_raw + gist_tokens_actual + repair_tokens_actual")
    if trial.get("status") == "ok" and byte_match is False:
        errors.append(
            f"line {trial_record['line']}: logical prompt bytes do not equal "
            "logical prompt tokens * bytes_per_kv_token")

    before_bytes = tensor.get("full_equivalent_selected_history_bytes")
    after_bytes = tensor.get("active_history_bytes")
    uncompressed, uncompressed_basis = _uncompressed_status(trial)
    tool_calls = action.get("tool_calls")
    tool_call_count = len(tool_calls) if isinstance(tool_calls, list) else None
    policy_http = sum(
        event["kind"] == "policy_generation" for event in http_events)
    auxiliary_http = sum(event["kind"] == "auxiliary" for event in http_events)
    extract_http = sum(
        isinstance(event["path"], str)
        and event["path"].rstrip("/").endswith("/extract")
        for event in http_events)
    repair_extract_http = sum(
        isinstance(event["path"], str)
        and event["path"].rstrip("/").endswith("/repair_extract")
        for event in http_events)
    failed_http = sum(event["status"] == "failed" for event in http_events)
    non_ok_http = sum(event["status"] != "ok" for event in http_events)

    row = {
        "label": PRELIMINARY_LABEL,
        "replay_line": trial_record["line"],
        "run_id": trial.get("run_id"),
        "trial_id": trial.get("trial_id"),
        "case_id": trial.get("case_id"),
        "task_id": identity.get("task_id"),
        "task_numeric_id": identity.get("task_numeric_id"),
        "request_ordinal": identity.get("request_ordinal"),
        "turn_index": identity.get("turn_index"),
        "step_index": identity.get("step_index"),
        "arm": arm,
        "arm_role": intervention.get("role"),
        "base_arm": intervention.get("base_arm"),
        "trial_status": trial.get("status"),
        "error_kind": trial.get("error_kind"),
        "error": trial.get("error"),
        "shared_device": shared_device,
        "timing_scope": timing_scope,
        "action_kind": action.get("kind"),
        "tool_call_count": tool_call_count,
        "paired_full_comparison_scope": comparison_scope,
        "paired_full_method_quality_eligible": arm != "full",
        "paired_full_agreement_status": paired.get("status"),
        "paired_full_action_match": paired.get("action_match"),
        "paired_full_action_raw_match": paired.get("action_raw_match"),
        "legacy_full_descriptive_scope": "descriptive_reference_not_repeatability",
        "legacy_full_descriptive_is_repeatability": False,
        "legacy_full_descriptive_agreement_status": legacy.get("status"),
        "legacy_full_descriptive_action_match": legacy.get("action_match"),
        "legacy_full_descriptive_action_raw_match": legacy.get("action_raw_match"),
        "prompt_tokens_raw": prompt_tokens_raw,
        "gist_tokens_actual": gist_tokens_actual,
        "repair_tokens_actual": repair_tokens_actual,
        "logical_active_prompt_tokens": logical_recomputed,
        "logical_active_prompt_tokens_recorded": logical_recorded,
        "logical_active_prompt_tokens_match_recorded": token_match,
        "bytes_per_kv_token": bytes_per_token,
        "logical_active_prompt_bytes": logical_bytes_recorded,
        "logical_active_prompt_bytes_recomputed": logical_bytes_recomputed,
        "logical_active_prompt_bytes_match_recomputed": byte_match,
        "selected_history_before_bytes": before_bytes,
        "selected_history_after_bytes": after_bytes,
        "selected_history_per_request_ratio_after_over_before": _ratio(
            after_bytes, before_bytes),
        "uncompressed": uncompressed,
        "uncompressed_basis": uncompressed_basis,
        "assemble_sec": timing.get("assemble_sec"),
        "trial_wall_sec": timing.get("trial_wall_sec"),
        "run_elapsed_sec": timing.get("run_elapsed_sec"),
        "http_event_count_observed": len(http_events),
        "http_request_count_recorded": trial.get("http_request_count"),
        "policy_http_event_count_observed": policy_http,
        "policy_generation_count_recorded": trial.get(
            "policy_generation_count"),
        "auxiliary_http_event_count_observed": auxiliary_http,
        "extract_http_event_count_observed": extract_http,
        "repair_extract_http_event_count_observed": repair_extract_http,
        "auxiliary_extract_total_http_event_count_observed": (
            extract_http + repair_extract_http),
        "failed_http_event_count_observed": failed_http,
        "non_ok_http_event_count_observed": non_ok_http,
        "http_wall_sec_sum_over_available_events": _sum_available(
            event["wall_sec"] for event in http_events),
    }
    if trial.get("label") != PRELIMINARY_LABEL:
        errors.append(
            f"line {trial_record['line']}: trial label is not "
            f"{PRELIMINARY_LABEL!r}")
    return row, http_events


def _count_matches(
    rows: Sequence[Mapping[str, Any]], field: str, value: Any,
) -> int:
    return sum(row.get(field) == value for row in rows)


def _arm_summary(
    arm: str, rows: Sequence[Mapping[str, Any]],
    shared_device: str, timing_scope: str,
) -> Dict[str, Any]:
    ratio_rows = [
        row for row in rows
        if _ratio(row.get("selected_history_after_bytes"),
                  row.get("selected_history_before_bytes")) is not None
    ]
    ratio_before_sum = _sum_available(
        row.get("selected_history_before_bytes") for row in ratio_rows)
    ratio_after_sum = _sum_available(
        row.get("selected_history_after_bytes") for row in ratio_rows)
    ratio_sum = _ratio(ratio_after_sum, ratio_before_sum)
    ratio_median = _median_available(
        row.get("selected_history_per_request_ratio_after_over_before")
        for row in ratio_rows)

    full_self_rows = [
        row for row in rows
        if row.get("paired_full_comparison_scope") == "full_self_reference"
        and row.get("paired_full_agreement_status") == "compared"
    ]
    method_rows = [
        row for row in rows
        if row.get("paired_full_comparison_scope")
        == "method_vs_paired_full_reference"
        and row.get("paired_full_agreement_status") == "compared"
    ]
    legacy_rows = [
        row for row in rows
        if row.get("legacy_full_descriptive_agreement_status") == "compared"
    ]
    http_count = sum(
        int(row["http_event_count_observed"]) for row in rows)
    policy_http_count = sum(
        int(row["policy_http_event_count_observed"]) for row in rows)
    auxiliary_http_count = sum(
        int(row["auxiliary_http_event_count_observed"]) for row in rows)
    extract_http_count = sum(
        int(row["extract_http_event_count_observed"]) for row in rows)
    repair_extract_http_count = sum(
        int(row["repair_extract_http_event_count_observed"]) for row in rows)
    failed_http_count = sum(
        int(row["failed_http_event_count_observed"]) for row in rows)
    non_ok_http_count = sum(
        int(row["non_ok_http_event_count_observed"]) for row in rows)

    summary = {
        "label": PRELIMINARY_LABEL,
        "arm": arm,
        "arm_role": next((row.get("arm_role") for row in rows
                          if row.get("arm_role") is not None), None),
        "shared_device": shared_device,
        "timing_scope": timing_scope,
        "trial_count": len(rows),
        "ok_trial_count": _count_matches(rows, "trial_status", "ok"),
        "failed_trial_count": _count_matches(rows, "trial_status", "failed"),
        "action_tool_rows": _count_matches(rows, "action_kind", "tool_calls"),
        "action_text_rows": _count_matches(rows, "action_kind", "text"),
        "action_unavailable_rows": sum(
            row.get("action_kind") not in ("tool_calls", "text") for row in rows),
        "tool_call_count_sum_over_available_rows": _sum_available(
            row.get("tool_call_count") for row in rows),
        "tool_call_count_available_row_count": sum(
            _number(row.get("tool_call_count")) is not None for row in rows),
        "full_self_reference_compared_count": len(full_self_rows),
        "full_self_reference_action_match_count": _count_matches(
            full_self_rows, "paired_full_action_match", True),
        "full_self_reference_is_method_quality": False,
        "paired_full_method_compared_count": len(method_rows),
        "paired_full_method_action_match_count": _count_matches(
            method_rows, "paired_full_action_match", True),
        "paired_full_method_action_match_rate": _rate(
            _count_matches(method_rows, "paired_full_action_match", True),
            len(method_rows)),
        "legacy_full_descriptive_compared_count": len(legacy_rows),
        "legacy_full_descriptive_action_match_count": _count_matches(
            legacy_rows, "legacy_full_descriptive_action_match", True),
        "legacy_full_descriptive_action_match_rate": _rate(
            _count_matches(legacy_rows, "legacy_full_descriptive_action_match", True),
            len(legacy_rows)),
        "legacy_full_descriptive_is_repeatability": False,
        "prompt_tokens_raw_sum_over_available_trials": _sum_available(
            row.get("prompt_tokens_raw") for row in rows),
        "prompt_tokens_raw_available_trial_count": sum(
            _number(row.get("prompt_tokens_raw")) is not None for row in rows),
        "gist_tokens_actual_sum_over_available_trials": _sum_available(
            row.get("gist_tokens_actual") for row in rows),
        "gist_tokens_actual_available_trial_count": sum(
            _number(row.get("gist_tokens_actual")) is not None for row in rows),
        "repair_tokens_actual_sum_over_available_trials": _sum_available(
            row.get("repair_tokens_actual") for row in rows),
        "repair_tokens_actual_available_trial_count": sum(
            _number(row.get("repair_tokens_actual")) is not None for row in rows),
        "logical_active_prompt_tokens_sum_over_available_trials": _sum_available(
            row.get("logical_active_prompt_tokens") for row in rows),
        "logical_active_prompt_tokens_available_trial_count": sum(
            _number(row.get("logical_active_prompt_tokens")) is not None
            for row in rows),
        "logical_active_prompt_bytes_sum_over_available_trials": _sum_available(
            row.get("logical_active_prompt_bytes") for row in rows),
        "logical_active_prompt_bytes_available_trial_count": sum(
            _number(row.get("logical_active_prompt_bytes")) is not None
            for row in rows),
        "selected_history_before_bytes_sum_over_available_trials": _sum_available(
            row.get("selected_history_before_bytes") for row in rows),
        "selected_history_before_bytes_available_trial_count": sum(
            _number(row.get("selected_history_before_bytes")) is not None
            for row in rows),
        "selected_history_after_bytes_sum_over_available_trials": _sum_available(
            row.get("selected_history_after_bytes") for row in rows),
        "selected_history_after_bytes_available_trial_count": sum(
            _number(row.get("selected_history_after_bytes")) is not None
            for row in rows),
        "selected_history_sum_ratio_after_over_before": ratio_sum,
        "selected_history_sum_ratio_numerator_after_bytes_sum": ratio_after_sum,
        "selected_history_sum_ratio_denominator_before_bytes_sum": ratio_before_sum,
        "selected_history_sum_ratio_request_count": len(ratio_rows),
        "selected_history_median_per_request_ratio_after_over_before": ratio_median,
        "selected_history_median_per_request_ratio_request_count": len(ratio_rows),
        "uncompressed_trial_count": _count_matches(rows, "uncompressed", True),
        "compressed_trial_count": _count_matches(rows, "uncompressed", False),
        "compression_status_unavailable_trial_count": _count_matches(
            rows, "uncompressed", None),
        "uncompressed_full_reference_trial_count": _count_matches(
            rows, "uncompressed_basis", "full_reference"),
        "uncompressed_protection_identity_trial_count": _count_matches(
            rows, "uncompressed_basis", "protection_identity"),
        "uncompressed_selected_history_before_equals_after_trial_count": _count_matches(
            rows, "uncompressed_basis", "selected_history_before_equals_after"),
        "logical_prompt_token_mismatch_count": _count_matches(
            rows, "logical_active_prompt_tokens_match_recorded", False),
        "logical_prompt_byte_mismatch_count": _count_matches(
            rows, "logical_active_prompt_bytes_match_recomputed", False),
        "http_event_count": http_count,
        "policy_http_event_count": policy_http_count,
        "auxiliary_http_event_count": auxiliary_http_count,
        "extract_http_event_count": extract_http_count,
        "repair_extract_http_event_count": repair_extract_http_count,
        "auxiliary_extract_total_http_event_count": (
            extract_http_count + repair_extract_http_count),
        "failed_http_event_count": failed_http_count,
        "non_ok_http_event_count": non_ok_http_count,
        "trial_wall_sec_sum_over_available_trials": _sum_available(
            row.get("trial_wall_sec") for row in rows),
        "trial_wall_sec_available_trial_count": sum(
            _number(row.get("trial_wall_sec")) is not None for row in rows),
        "trial_wall_sec_median_over_available_trials": _median_available(
            row.get("trial_wall_sec") for row in rows),
        "http_wall_sec_sum_over_available_events": _sum_available(
            row.get("http_wall_sec_sum_over_available_events") for row in rows),
        "trial_provenance_json": json.dumps([
            {
                "replay_line": row.get("replay_line"),
                "run_id": row.get("run_id"),
                "trial_id": row.get("trial_id"),
                "case_id": row.get("case_id"),
            }
            for row in rows
        ], ensure_ascii=False, separators=(",", ":")),
    }
    return summary


def _failed_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for record in rows:
        row = record["value"]
        if row.get("status") != "failed":
            continue
        output.append({
            "label": PRELIMINARY_LABEL,
            "replay_line": record["line"],
            "row_type": row.get("row_type"),
            "run_id": row.get("run_id"),
            "trial_id": row.get("trial_id"),
            "case_id": row.get("case_id"),
            "arm": _deep_get(row, "intervention", "name"),
            "error_kind": row.get("error_kind", row.get("failure_kind")),
            "error": row.get("error"),
        })
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise SummaryError(f"refusing to write empty CSV {path.name}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def summarize(replay: Path, out_dir: Path, shared_device: str) -> int:
    records = _read_replay(replay)
    manifest_record = _unique_row(records, "run_manifest")
    final_record = _unique_row(records, "run_final")
    manifest = manifest_record["value"]
    final = final_record["value"]
    selected_cases = _required_unique_strings(
        manifest.get("selected_case_ids"), "run_manifest.selected_case_ids")
    arms = _required_unique_strings(
        manifest.get("interventions"), "run_manifest.interventions")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise SummaryError("run_manifest.run_id must be a non-empty string")
    if final.get("run_id") != run_id:
        raise SummaryError("run_final.run_id differs from run_manifest.run_id")

    errors: List[str] = []
    recognized_types = {"run_manifest", "trial", "run_final"}
    for record in records:
        row = record["value"]
        if row.get("row_type") not in recognized_types:
            errors.append(
                f"line {record['line']}: unknown row_type {row.get('row_type')!r}")
        if row.get("schema_version") != EXPECTED_REPLAY_SCHEMA_VERSION:
            errors.append(
                f"line {record['line']}: unexpected schema_version "
                f"{row.get('schema_version')!r}")
        if row.get("run_id") != run_id:
            errors.append(
                f"line {record['line']}: run_id differs from manifest")

    trial_records = [
        record for record in records
        if record["value"].get("row_type") == "trial"
    ]
    timing_scope = _timing_scope(shared_device)
    case_rows: List[Dict[str, Any]] = []
    all_http_events: List[Dict[str, Any]] = []
    for trial_record in trial_records:
        case_row, http_events = _case_row(
            trial_record, shared_device, timing_scope, errors)
        case_rows.append(case_row)
        all_http_events.extend(http_events)

    expected_pairs = {(case_id, arm) for case_id in selected_cases for arm in arms}
    pair_counts = Counter((row.get("case_id"), row.get("arm")) for row in case_rows)
    observed_pairs = set(pair_counts)
    missing_pairs = [
        {"case_id": case_id, "arm": arm}
        for case_id in selected_cases for arm in arms
        if (case_id, arm) not in observed_pairs
    ]
    duplicate_pairs = [
        {"case_id": case_id, "arm": arm, "count": count}
        for (case_id, arm), count in pair_counts.items() if count > 1
    ]
    unexpected_pairs = [
        {"case_id": case_id, "arm": arm, "count": pair_counts[(case_id, arm)]}
        for case_id, arm in observed_pairs - expected_pairs
    ]
    if duplicate_pairs:
        errors.append(f"matrix contains {len(duplicate_pairs)} duplicate pairs")
    if unexpected_pairs:
        errors.append(f"matrix contains {len(unexpected_pairs)} unexpected pairs")
    if missing_pairs and final.get("status") == "complete":
        errors.append(
            f"complete run is missing {len(missing_pairs)} expected matrix pairs")

    expected_trial_count = len(expected_pairs)
    recorded_manifest_planned = _integer(manifest.get("planned_policy_generations"))
    if recorded_manifest_planned != expected_trial_count:
        errors.append(
            "run_manifest.planned_policy_generations does not equal "
            "selected cases times arms")
    recorded_final_completed = _integer(final.get("completed_trial_count"))
    if recorded_final_completed != len(trial_records):
        errors.append(
            "run_final.completed_trial_count does not equal observed trial rows")
    recorded_final_planned = _integer(final.get("planned_trial_count"))
    if recorded_final_planned != expected_trial_count:
        errors.append(
            "run_final.planned_trial_count does not equal selected cases times arms")

    observed_http_count = len(all_http_events)
    observed_policy_http_count = sum(
        event["kind"] == "policy_generation" for event in all_http_events)
    observed_auxiliary_http_count = sum(
        event["kind"] == "auxiliary" for event in all_http_events)
    observed_extract_http_count = sum(
        isinstance(event["path"], str)
        and event["path"].rstrip("/").endswith("/extract")
        for event in all_http_events)
    observed_repair_extract_http_count = sum(
        isinstance(event["path"], str)
        and event["path"].rstrip("/").endswith("/repair_extract")
        for event in all_http_events)
    observed_failed_http_count = sum(
        event["status"] == "failed" for event in all_http_events)
    observed_non_ok_http_count = sum(
        event["status"] != "ok" for event in all_http_events)
    if _integer(final.get("http_request_count")) != observed_http_count:
        errors.append(
            "run_final.http_request_count does not equal observed HTTP events")
    if _integer(final.get("policy_generation_count")) != observed_policy_http_count:
        errors.append(
            "run_final.policy_generation_count does not equal observed policy HTTP events")
    ordinal_counts = Counter(event.get("ordinal") for event in all_http_events)
    expected_ordinals = set(range(1, observed_http_count + 1))
    observed_ordinals = {
        ordinal for ordinal in ordinal_counts if _integer(ordinal) is not None
    }
    if observed_ordinals != expected_ordinals:
        errors.append("HTTP ordinals do not cover 1..observed HTTP event count")
    if any(count != 1 for count in ordinal_counts.values()):
        errors.append("HTTP ordinals contain duplicates")

    declared_arm_set = set(arms)
    output_arm_order = arms + sorted({
        str(row.get("arm")) for row in case_rows
        if row.get("arm") not in declared_arm_set
    })
    rows_by_arm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        rows_by_arm[str(row.get("arm"))].append(row)
    arm_rows = [
        _arm_summary(arm, rows_by_arm.get(arm, []), shared_device, timing_scope)
        for arm in output_arm_order
    ]

    failed_rows = _failed_rows(records)
    failed_http_events = [
        event for event in all_http_events if event["status"] == "failed"
    ]
    non_ok_http_events = [
        event for event in all_http_events if event["status"] != "ok"
    ]
    validation = {
        "integrity_ok": not errors,
        "integrity_errors": errors,
        "run_manifest_count": 1,
        "run_final_count": 1,
        "selected_case_count": len(selected_cases),
        "arm_count": len(arms),
        "expected_matrix_trial_count": expected_trial_count,
        "observed_trial_count": len(trial_records),
        "matrix_complete": not missing_pairs and not duplicate_pairs
        and not unexpected_pairs,
        "missing_pairs": missing_pairs,
        "duplicate_pairs": duplicate_pairs,
        "unexpected_pairs": unexpected_pairs,
        "failed_row_count": len(failed_rows),
        "failed_rows": failed_rows,
        "failed_http_event_count": len(failed_http_events),
        "failed_http_events": failed_http_events,
        "non_ok_http_event_count": len(non_ok_http_events),
        "non_ok_http_events": non_ok_http_events,
        "final_recorded_http_request_count": final.get("http_request_count"),
        "final_http_request_count_matches_observed": (
            _integer(final.get("http_request_count")) == observed_http_count),
        "final_recorded_policy_generation_count": final.get(
            "policy_generation_count"),
        "final_policy_generation_count_matches_observed": (
            _integer(final.get("policy_generation_count"))
            == observed_policy_http_count),
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "label": PRELIMINARY_LABEL,
        "source": {
            "replay_path": str(replay.resolve()),
            "replay_line_count": len(records),
            "run_id": run_id,
            "run_manifest_line": manifest_record["line"],
            "run_final_line": final_record["line"],
            "run_status": final.get("status"),
            "selection": manifest.get("selection"),
        },
        "timing": {
            "shared_device": shared_device,
            "scope": timing_scope,
            "run_wall_sec_recorded": final.get("wall_sec"),
            "trial_wall_sec_sum_over_available_trials": _sum_available(
                row.get("trial_wall_sec") for row in case_rows),
            "trial_wall_sec_available_trial_count": sum(
                _number(row.get("trial_wall_sec")) is not None
                for row in case_rows),
            "http_wall_sec_sum_over_available_events": _sum_available(
                event.get("wall_sec") for event in all_http_events),
            "http_wall_sec_available_event_count": sum(
                _number(event.get("wall_sec")) is not None
                for event in all_http_events),
        },
        "counts": {
            "replay_line_count": len(records),
            "trial_count": len(case_rows),
            "ok_trial_count": _count_matches(case_rows, "trial_status", "ok"),
            "failed_trial_count": _count_matches(
                case_rows, "trial_status", "failed"),
            "http_event_count": observed_http_count,
            "policy_http_event_count": observed_policy_http_count,
            "auxiliary_http_event_count": observed_auxiliary_http_count,
            "extract_http_event_count": observed_extract_http_count,
            "repair_extract_http_event_count": observed_repair_extract_http_count,
            "auxiliary_extract_total_http_event_count": (
                observed_extract_http_count + observed_repair_extract_http_count),
            "failed_http_event_count": observed_failed_http_count,
            "non_ok_http_event_count": observed_non_ok_http_count,
        },
        "metric_contract": {
            "prompt_tokens_raw": (
                "measurement.prompt_tokens_raw; no gist or repair subtraction"),
            "logical_active_prompt_tokens": (
                "prompt_tokens_raw + gist_tokens_actual + repair_tokens_actual"),
            "logical_active_prompt_bytes": (
                "logical_active_prompt_tokens * bytes_per_kv_token"),
            "selected_history_before_bytes": (
                "measurement.tensor_bytes.full_equivalent_selected_history_bytes"),
            "selected_history_after_bytes": (
                "measurement.tensor_bytes.active_history_bytes"),
            "selected_history_sum_ratio_after_over_before": (
                "sum(after bytes for ratio-eligible requests) / "
                "sum(before bytes for the same requests)"),
            "selected_history_median_per_request_ratio_after_over_before": (
                "median(after bytes / before bytes per ratio-eligible request)"),
            "ratio_eligible_request": (
                "before and after bytes are numeric and before bytes > 0"),
            "paired_full_agreement": "action agreement with paired live full action",
            "full_self_reference": (
                "full action compared with itself; reference only; "
                "not included in paired_full_method fields"),
            "legacy_full_descriptive_agreement": (
                "descriptive comparison with saved full action; not repeatability"),
        },
        "validation": validation,
        "http_events": all_http_events,
        "arms": arm_rows,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / OUTPUT_NAMES[0], case_rows)
    _write_csv(out_dir / OUTPUT_NAMES[1], arm_rows)
    _write_json(out_dir / OUTPUT_NAMES[2], summary)
    return 0 if validation["integrity_ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path,
                        help="mechanism_replay_v1 JSONL")
    parser.add_argument("--out", required=True, type=Path,
                        help="directory for case_arm.csv and arm summaries")
    parser.add_argument(
        "--shared-device", nargs="?", const="true", default="unknown",
        choices=("true", "false", "unknown"),
        help="device-sharing scope; bare --shared-device means true",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return summarize(args.replay, args.out, args.shared_device)
    except SummaryError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
