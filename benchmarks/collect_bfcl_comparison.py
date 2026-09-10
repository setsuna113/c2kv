#!/usr/bin/env python3
"""Collect auditable BFCL arm metrics from one benchmark-matrix root.

The collector is deliberately artifact-driven.  Official score headers provide
the aggregate score, while per-task correctness is reconstructed only when the
matching prediction ID set and the score detail rows make that reconstruction
unambiguous.  Missing artifacts remain ``null``.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = "bfcl_comparison_v1"
SCORE_RE = re.compile(r"^BFCL_v4_(.+)_score\.json$")
HBM_SNAPSHOT_SCOPE = (
    "status=ok request snapshots of currently occupied KV in the shared process; "
    "not per-session HBM and excludes reserved allocator/model memory; lifetime peak fields ignored"
)
BYTE_SUM_SCOPE = (
    "sum of per-request byte values (byte-requests); not a resident-memory budget"
)
PROMPT_KV_SCOPE = (
    "derived from successful SGLang OpenAI chat requests: "
    "(usage.prompt_tokens + injected gist_len + injected repair_len) * bytes_per_kv_token; "
    "includes raw system/tools/current context, excludes decode, page rounding, "
    "shared radix/sidepool copies, indexes and auxiliary extraction prefills; "
    "not allocator HBM or total-system resident bytes; physical_eviction unsupported"
)
KV_HISTORY_RECONSTRUCTION_SCOPE = (
    "active history payload reconstructed per status=ok request from measured "
    "physical-eviction kept tokens, injected c2kv_layout blocks, or history "
    "selection metadata, in that order. kv_history_full_equivalent describes "
    "the original span of selected docs and excludes prior doc-window eviction; "
    "unverifiable rows remain missing"
)
KV_HISTORY_REPORTED_SCOPE = (
    "request_log history_kv reported counters preserved verbatim from the "
    "server report; legacy repair_extract active counters may contain client-"
    "hint plus scheduler-injection double counting, while legacy no-hint full-"
    "equivalent counters may contain only the first injected original; neither "
    "is used as actual payload without independent provenance"
)
HISTORY_PACKED_SCOPE = (
    "history_packed_original_tokens is the sum of original_seq_len over every "
    "fitted packed-history candidate before turn tail selection; original_tokens "
    "remains the selected-history ledger for compatibility. Native/uncompressed "
    "paths are N/A. Legacy rows with dropped_docs but no packed fields have an "
    "unknown whole-history denominator."
)


CSV_FIELDS = [
    "cell_id", "arm", "arm_role", "cell_status", "preliminary_n", "summary_status",
    "scorer_status", "scorer_numerator", "scorer_denominator", "scorer_score",
    "task_correctness_status", "request_log_status", "request_count", "ok_count",
    "task_id_coverage_status", "task_id_coverage_expected_count",
    "task_id_coverage_audit_count", "task_id_coverage_proxy_count",
    "error_count", "proxy_request_wall_sec_sum", "proxy_request_wall_sec_p50",
    "proxy_request_wall_sec_p90", "proxy_request_wall_scope",
    "runner_adapter_wall_sec", "runner_wall_scope",
    "hbm_snapshot_scope", "total_gpu_kv_bytes_snapshot_count",
    "total_gpu_kv_bytes_snapshot_p50", "total_gpu_kv_bytes_snapshot_max",
    "physical_main_kv_bytes_snapshot_count", "physical_main_kv_bytes_snapshot_p50",
    "physical_main_kv_bytes_snapshot_max", "physical_c2kv_pool_bytes_snapshot_count",
    "physical_c2kv_pool_bytes_snapshot_p50", "physical_c2kv_pool_bytes_snapshot_max",
    "logical_prompt_kv_bytes_count", "logical_prompt_kv_bytes_p50",
    "logical_prompt_kv_bytes_max", "logical_prompt_kv_scope",
    "task_wall_status", "task_wall_count", "task_wall_sec_sum", "task_wall_sec_p50",
    "task_wall_sec_p90", "task_wall_scope", "original_history_tokens_sum",
    "history_packed_accounting_status",
    "history_packed_accounting_coverage_ok_request_count",
    "history_packed_accounting_request_count",
    "history_packed_whole_history_unknown_request_count",
    "history_packed_original_tokens_sum", "history_dropped_original_tokens_sum",
    "history_packed_candidate_doc_count_sum", "history_packed_retained_fraction",
    "compressed_history_tokens_before_recovery_sum",
    "compressed_history_tokens_after_recovery_sum",
    "history_tensor_full_equivalent_bytes_sum",
    "history_tensor_full_equivalent_bytes_count",
    "history_tensor_full_equivalent_bytes_p50",
    "history_tensor_full_equivalent_bytes_max",
    "history_tensor_before_recovery_bytes_sum",
    "history_tensor_before_recovery_bytes_count",
    "history_tensor_before_recovery_bytes_p50",
    "history_tensor_before_recovery_bytes_max",
    "history_tensor_after_recovery_bytes_sum",
    "history_tensor_after_recovery_bytes_count",
    "history_tensor_after_recovery_bytes_p50",
    "history_tensor_after_recovery_bytes_max",
    "history_tensor_accounting_request_count", "history_tensor_coverage_ok_request_count",
    "kv_history_full_equivalent_tokens_sum",
    "kv_history_full_equivalent_tokens_source",
    "kv_history_reported_full_equivalent_tokens_sum",
    "kv_history_active_tokens_sum", "kv_history_active_tokens_source",
    "kv_history_active_tokens_missing_count", "kv_history_reconstruction_scope",
    "kv_history_reported_active_tokens_sum", "kv_history_reported_scope",
    "kv_history_tensor_full_equivalent_bytes_sum",
    "kv_history_tensor_full_equivalent_bytes_count",
    "kv_history_tensor_full_equivalent_bytes_p50",
    "kv_history_tensor_full_equivalent_bytes_max",
    "kv_history_reported_tensor_full_equivalent_bytes_sum",
    "kv_history_reported_tensor_full_equivalent_bytes_count",
    "kv_history_reported_tensor_full_equivalent_bytes_p50",
    "kv_history_reported_tensor_full_equivalent_bytes_max",
    "kv_history_tensor_active_bytes_sum", "kv_history_tensor_active_bytes_count",
    "kv_history_tensor_active_bytes_p50", "kv_history_tensor_active_bytes_max",
    "kv_history_reported_tensor_active_bytes_sum",
    "kv_history_reported_tensor_active_bytes_count",
    "kv_history_reported_tensor_active_bytes_p50",
    "kv_history_reported_tensor_active_bytes_max",
    "kv_history_tensor_accounting_request_count",
    "kv_history_reported_accounting_request_count",
    "kv_history_tensor_coverage_ok_request_count", "byte_sum_scope",
    "dropped_docs_total", "dropped_requests",
    "gold_metrics_status", "gold_trigger_count", "gold_intervene_count",
    "gold_recovered_count", "gold_no_witness_count", "recovery_append_request_count",
    "recovery_distinct_event_count", "raw_recompute_event_count",
    "raw_recompute_wall_sec_sum",
    "compressor_metrics_status", "compressor_calls", "compressor_prompt_tokens",
    "compressor_completion_tokens", "compressor_wall_sec",
    "retrieval_calls", "retrieval_prompt_tokens", "retrieval_completion_tokens",
    "native_tool_call_presence_status", "native_tool_call_presence_numerator",
    "native_tool_call_presence_denominator", "native_tool_call_presence_rate",
    "summary_path", "request_log_path", "task_audit_path",
]


@dataclass(frozen=True)
class Cell:
    cell_id: str
    arm: str
    run_dir: Path
    summary_path: Path


def _is_number(value: Any) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _strict_sum(values: Iterable[Any]) -> Optional[float]:
    picked = list(values)
    if not picked or any(not _is_number(value) for value in picked):
        return None
    return sum(float(value) for value in picked)


def _number_sum(rows: Sequence[Mapping[str, Any]], key: str) -> Optional[float]:
    return _strict_sum(row.get(key) for row in rows)


def _observed_distribution(values: Iterable[Any]) -> Dict[str, Any]:
    observed = [float(value) for value in values if _is_number(value)]
    return {
        "count": len(observed),
        "p50": _percentile(observed, 0.50),
        "max": max(observed) if observed else None,
    }


def _complete_distribution(values: Iterable[Any]) -> Dict[str, Any]:
    materialized = list(values)
    observed = _observed_distribution(materialized)
    if observed["count"] != len(materialized):
        observed["p50"] = None
        observed["max"] = None
    return observed


def _put_distribution(out: Dict[str, Any], prefix: str,
                      stats: Mapping[str, Any]) -> None:
    for suffix in ("count", "p50", "max"):
        out[f"{prefix}_{suffix}"] = stats[suffix]


def _logical_prompt_kv_bytes(row: Mapping[str, Any]) -> Optional[int]:
    # OpenAI serving_chat removes annotated carriers and creates zero-width
    # segments.  origin_input_ids (reported as prompt_tokens) excludes each
    # injected block; schedule_batch tracks the enlarged virtual sequence.
    # The physical-eviction path has a different usage contract and no ledger.
    if (row.get("backend") != "sglang"
            or row.get("history_kv_backend") == "physical_eviction"
            or row.get("c2kv_injection_error")
            or row.get("finish_reason") in ("abort", "error")):
        return None
    usage = row.get("usage")
    layout = row.get("c2kv_layout")
    unit = row.get("bytes_per_kv_token")
    if (not isinstance(usage, dict) or not isinstance(layout, list)
            or not _is_int(unit) or unit <= 0):
        return None
    tokens = usage.get("prompt_tokens")
    if not _is_int(tokens) or tokens < 0:
        return None
    for item in layout:
        if not isinstance(item, dict):
            return None
        key = {"gist": "gist_len", "repair": "repair_len"}.get(item.get("kind"))
        if key is None or not _is_int(item.get(key)) or item[key] < 0:
            return None
        tokens += item[key]
    return tokens * unit


def _history_kv_active_tokens(
    row: Mapping[str, Any],
) -> Tuple[Optional[int], Optional[str]]:
    """Return auditable active history tokens without trusting legacy totals."""
    if row.get("history_kv_backend") == "physical_eviction":
        kept = row.get("history_kv_kept_tokens")
        if row.get("history_kv_eviction_ok") is True and _is_int(kept) and kept >= 0:
            return kept, "physical_eviction.kept_history_tokens"
        return None, None

    layout = row.get("c2kv_layout")
    if isinstance(layout, list) and layout:
        total = 0
        for item in layout:
            if not isinstance(item, dict):
                break
            key = {"gist": "gist_len", "repair": "repair_len"}.get(item.get("kind"))
            if key is None or not _is_int(item.get(key)) or item[key] < 0:
                break
            total += item[key]
        else:
            return total, "c2kv_layout.injected_tokens"

    selection = row.get("history_kv_selection")
    if isinstance(selection, dict):
        selected = selection.get("selected_token_count")
        if _is_int(selected) and selected >= 0:
            return selected, "history_kv_selection.selected_token_count"

    selected = row.get("history_kv_selected_tokens")
    if _is_int(selected) and selected >= 0:
        return selected, "history_kv_selected_tokens"

    source = row.get("history_kv_active_tokens_source")
    reported = row.get("history_kv_active_tokens")
    if source in {"scheduler_runtime", "physical_eviction_measured"}:
        if _is_int(reported) and reported >= 0:
            return reported, f"server_report.{source}"
    return None, None


def _history_kv_full_equivalent_tokens(
    row: Mapping[str, Any],
) -> Tuple[Optional[int], Optional[str]]:
    """Return the original span of selected history docs with provenance."""
    if row.get("history_kv_backend") == "physical_eviction":
        history = row.get("history_kv_history_tokens")
        if row.get("history_kv_eviction_ok") is True and _is_int(history) and history >= 0:
            return history, "physical_eviction.history_tokens"
        return None, None

    selection = row.get("history_kv_selection")
    if isinstance(selection, dict):
        requested = selection.get("requested_span_tokens")
        if _is_int(requested) and requested >= 0:
            return requested, "history_kv_selection.requested_span_tokens"

    span = row.get("history_kv_span_tokens")
    if _is_int(span) and span >= 0:
        return span, "history_kv_span_tokens"

    original = row.get("original_tokens")
    gist = row.get("gist_tokens")
    if (_is_int(original) and original > 0
            and _is_int(gist) and gist > 0):
        return original, "request_log.original_tokens"

    layout = row.get("c2kv_layout")
    if isinstance(layout, list) and layout:
        gist_originals: List[int] = []
        for item in layout:
            if not isinstance(item, dict):
                return None, None
            if item.get("kind") == "repair":
                continue
            if item.get("kind") != "gist":
                return None, None
            value = item.get("original_seq_len")
            if not _is_int(value) or value < 0:
                return None, None
            gist_originals.append(value)
        if gist_originals:
            return sum(gist_originals), "c2kv_layout.gist_original_seq_len"

    reported = row.get("history_kv_full_equivalent_tokens")
    source = row.get("history_kv_full_equivalent_tokens_source")
    if source == "request_hint" and _is_int(reported) and reported >= 0:
        return reported, "server_report.request_hint"
    return None, None


def _read_object(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        return None, f"cannot read {path}: {error}"
    except json.JSONDecodeError as error:
        return None, f"invalid JSON {path}: {error}"
    if not isinstance(value, dict):
        return None, f"JSON root is not an object: {path}"
    return value, None


def _read_records(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        return [], [f"cannot read {path}: {error}"]
    if not text.strip():
        return [], []
    records: List[Dict[str, Any]] = []
    errors: List[str] = []
    if text.lstrip().startswith("["):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            return [], [f"invalid JSON array {path}: {error}"]
        if not isinstance(value, list):
            return [], [f"JSON root is not an array: {path}"]
        for index, row in enumerate(value, 1):
            if isinstance(row, dict):
                records.append(row)
            else:
                errors.append(f"{path}: array item {index} is not an object")
        return records, errors
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"{path}:{lineno}: invalid JSON: {error.msg}")
            continue
        if isinstance(row, dict):
            records.append(row)
        else:
            errors.append(f"{path}:{lineno}: row is not an object")
    return records, errors


def _local_plan_path(root: Path, value: Any, fallback: Path) -> Path:
    """Use copied matrix layout instead of stale absolute paths in a plan."""
    if fallback.exists():
        return fallback
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.exists():
            return candidate
    return fallback


def _is_smoke_run_dir(run_dir: Path) -> bool:
    return run_dir.name.lower().startswith("smoke")


def _argv_value(argv: Sequence[str], flag: str) -> Optional[str]:
    for index, value in enumerate(argv):
        if value == flag and index + 1 < len(argv):
            return argv[index + 1]
        prefix = flag + "="
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def discover_cells(root: Path, *, include_smoke: bool = False) -> Tuple[List[Cell], List[str]]:
    root = root.resolve()
    warnings: List[str] = []
    found: Dict[Tuple[str, str], Cell] = {}
    plan_path = root / "matrix_plan.json"
    if plan_path.is_file():
        plan, error = _read_object(plan_path)
        if error:
            warnings.append(error)
        else:
            for item in plan.get("cells", []):
                if not isinstance(item, dict) or item.get("benchmark") != "bfcl":
                    continue
                arm = item.get("arm")
                cell_id = item.get("id")
                if not isinstance(arm, str) or not isinstance(cell_id, str):
                    warnings.append(f"invalid BFCL cell in {plan_path}: {item!r}")
                    continue
                local_run = root / "cells" / cell_id / "run"
                run_dir = _local_plan_path(root, item.get("cell_dir"), local_run)
                if run_dir.name != "run":
                    run_dir = run_dir / "run"
                fallback_summary = run_dir / f"summary_{arm}.json"
                summary_path = _local_plan_path(
                    root, item.get("summary_path"), fallback_summary)
                if include_smoke or not _is_smoke_run_dir(run_dir):
                    found[(cell_id, arm)] = Cell(cell_id, arm, run_dir, summary_path)

    cells_root = root / "cells"
    if cells_root.is_dir():
        for cell_dir in sorted(path for path in cells_root.iterdir() if path.is_dir()):
            if not cell_dir.name.startswith("bfcl__"):
                continue
            arm = cell_dir.name.split("__", 1)[1]
            key = (cell_dir.name, arm)
            run_dir = cell_dir / "run"
            if include_smoke or not _is_smoke_run_dir(run_dir):
                found.setdefault(key, Cell(
                    cell_dir.name, arm, run_dir,
                    run_dir / f"summary_{arm}.json"))

    # Queue-backed campaigns persist the exact argv before starting a cell.
    # A failed process can therefore have command.json but no summary.  Keep
    # that cell as partial so infrastructure failures cannot disappear from
    # the comparison table.
    for command_path in sorted(root.rglob("command.json")):
        run_dir = command_path.parent
        if not include_smoke and _is_smoke_run_dir(run_dir):
            continue
        try:
            argv = json.loads(command_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            warnings.append(f"invalid command artifact {command_path}: {error}")
            continue
        if (not isinstance(argv, list)
                or any(not isinstance(value, str) for value in argv)):
            warnings.append(f"command artifact is not a JSON argv list: {command_path}")
            continue
        if _argv_value(argv, "--benchmark") != "bfcl":
            continue
        arm = _argv_value(argv, "--arm")
        if not arm:
            warnings.append(f"BFCL command has no --arm: {command_path}")
            continue
        try:
            cell_id = run_dir.relative_to(root).as_posix()
        except ValueError:
            cell_id = run_dir.name
        found.setdefault((cell_id, arm), Cell(
            cell_id, arm, run_dir, run_dir / f"summary_{arm}.json"))

    for summary_path in sorted(root.rglob("summary_*.json")):
        if summary_path.name in ("comparison.json", "matrix_summary.json"):
            continue
        summary, _ = _read_object(summary_path)
        if (summary and summary.get("benchmark") is not None
                and summary.get("benchmark") != "bfcl"):
            continue
        arm = summary.get("arm") if summary else None
        if not isinstance(arm, str):
            arm = summary_path.stem.removeprefix("summary_")
        if not arm:
            continue
        parent = summary_path.parent
        if not include_smoke and _is_smoke_run_dir(parent):
            continue
        if parent.name == "run" and parent.parent.name.startswith("bfcl__"):
            cell_id = parent.parent.name
        else:
            try:
                cell_id = parent.relative_to(root).as_posix()
            except ValueError:
                cell_id = parent.name
        found.setdefault((cell_id, arm), Cell(cell_id, arm, parent, summary_path))

    return sorted(found.values(), key=lambda cell: (cell.arm, cell.cell_id)), warnings


def _resolve_named_artifact(
    root: Path,
    run_dir: Path,
    named: Any,
    patterns: Sequence[str],
) -> Tuple[Optional[Path], str]:
    candidates: List[Path] = []
    if isinstance(named, str) and named.strip():
        supplied = Path(named)
        for candidate in (supplied, run_dir / supplied.name,
                          run_dir / "logs" / supplied.name):
            try:
                inside_root = candidate.resolve().is_relative_to(root.resolve())
            except (OSError, ValueError):
                inside_root = False
            if candidate.is_file() and inside_root:
                candidates.append(candidate.resolve())
    if not candidates:
        for pattern in patterns:
            candidates.extend(path.resolve() for path in run_dir.glob(pattern)
                              if path.is_file())
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0], "available"
    if not unique:
        return None, "unavailable"
    return None, "ambiguous"


def _valid_score_header(header: Mapping[str, Any]) -> Optional[str]:
    total = header.get("total_count")
    correct = header.get("correct_count")
    accuracy = header.get("accuracy")
    if (not _is_int(total) or total <= 0 or not _is_int(correct)
            or correct < 0 or correct > total or not _is_number(accuracy)):
        return f"invalid score header: {dict(header)!r}"
    if not math.isclose(float(accuracy), correct / total,
                        rel_tol=0.0, abs_tol=1e-12):
        return ("score header accuracy mismatch: "
                f"accuracy={accuracy} correct_count={correct} total_count={total}")
    return None


def _prediction_ids(path: Path, expected: int) -> Tuple[Optional[set[str]], List[str]]:
    rows, errors = _read_records(path)
    if errors:
        return None, errors
    ids: List[str] = []
    for index, row in enumerate(rows, 1):
        if row.get("id") is None:
            errors.append(f"{path}: prediction row {index} has no id")
        else:
            ids.append(str(row["id"]))
    if errors:
        return None, errors
    unique = set(ids)
    if len(unique) != len(ids):
        return None, [f"{path}: duplicate prediction ids"]
    if len(unique) != expected:
        return None, [f"{path}: prediction id count {len(unique)} != header total {expected}"]
    return unique, []


def _category_artifacts(score_path: Path, prediction_path: Optional[Path],
                        category: str) -> Dict[str, Any]:
    score_rows, errors = _read_records(score_path)
    result: Dict[str, Any] = {
        "category": category,
        "score_path": str(score_path),
        "prediction_path": str(prediction_path) if prediction_path else None,
        "header_valid": False,
        "correctness_status": "unavailable",
        "errors": errors,
        "correctness": {},
    }
    if errors or not score_rows:
        if not score_rows and not errors:
            result["errors"].append(f"empty score artifact: {score_path}")
        return result
    header = score_rows[0]
    header_error = _valid_score_header(header)
    if header_error:
        result["errors"].append(f"{score_path}: {header_error}")
        return result
    result.update({
        "header_valid": True,
        "total_count": int(header["total_count"]),
        "correct_count": int(header["correct_count"]),
        "accuracy": float(header["accuracy"]),
    })
    if prediction_path is None:
        result["errors"].append(f"missing prediction artifact for {score_path}")
        return result
    prediction_ids, prediction_errors = _prediction_ids(
        prediction_path, result["total_count"])
    if prediction_errors or prediction_ids is None:
        result["errors"].extend(prediction_errors)
        return result

    known: Dict[str, bool] = {}
    detail_errors: List[str] = []
    # Pinned BFCL 6ea5797 writes only failed entry_result objects after the
    # header.  eval_runner returns their verdict as top-level ``valid`` and
    # eval_runner_helper.save_eval_results inserts the aggregate header.
    for index, row in enumerate(score_rows[1:], 2):
        row_id = row.get("id")
        valid = row.get("valid")
        if row_id is None:
            detail_errors.append(f"{score_path}: score row {index} has no id")
            continue
        key = str(row_id)
        if key not in prediction_ids:
            detail_errors.append(f"{score_path}: score id {key!r} has no prediction")
            continue
        if key in known:
            detail_errors.append(f"{score_path}: duplicate score id {key!r}")
            continue
        if not isinstance(valid, bool):
            detail_errors.append(
                f"{score_path}: score row {key!r} has no boolean valid field")
            continue
        known[key] = valid
    if detail_errors:
        result["errors"].extend(detail_errors)
        return result

    known_correct = sum(known.values())
    known_incorrect = len(known) - known_correct
    required_correct = result["correct_count"] - known_correct
    required_incorrect = (result["total_count"] - result["correct_count"]
                          - known_incorrect)
    unknown = prediction_ids - set(known)
    if required_correct < 0 or required_incorrect < 0:
        result["errors"].append(
            f"{score_path}: detail correctness counts contradict header")
        return result
    if required_correct + required_incorrect != len(unknown):
        result["errors"].append(
            f"{score_path}: score details and header do not cover prediction ids")
        return result
    if unknown and required_incorrect == 0:
        known.update((key, True) for key in unknown)
        derivation = "remaining prediction ids inferred correct after failure-count validation"
    elif unknown and required_correct == 0:
        known.update((key, False) for key in unknown)
        derivation = "remaining prediction ids inferred incorrect after count validation"
    elif unknown:
        result["errors"].append(
            f"{score_path}: header leaves {len(unknown)} prediction ids ambiguous")
        return result
    else:
        derivation = "explicit score detail rows"
    if sum(known.values()) != result["correct_count"] or len(known) != result["total_count"]:
        result["errors"].append(f"{score_path}: reconstructed correctness contradicts header")
        return result
    result["correctness"] = known
    result["correctness_status"] = "verified"
    result["correctness_derivation"] = derivation
    return result


def _expected_categories(summary: Optional[Mapping[str, Any]]) -> List[str]:
    if not summary:
        return []
    headers = summary.get("official_score_headers")
    if not isinstance(headers, list):
        return []
    return sorted({str(item["category"]) for item in headers
                   if isinstance(item, dict) and item.get("category")})


def _official_metrics(run_dir: Path, arm: str,
                      summary: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    empty = {
        "scorer_status": "unavailable", "scorer_numerator": None,
        "scorer_denominator": None, "scorer_score": None,
        "task_correctness_status": "unavailable",
        "official_categories": [], "task_correctness": {}, "errors": [],
    }
    if summary and summary.get("scored") is False:
        empty["scorer_status"] = "not_scored"
        return empty
    handler = f"c2kv-{arm.replace('_', '-')}"
    score_root = run_dir / "score" / handler
    result_root = run_dir / "result" / handler
    expected = _expected_categories(summary)
    score_files: List[Path] = []
    if expected:
        for category in expected:
            hits = sorted(score_root.rglob(f"BFCL_v4_{category}_score.json")) \
                if score_root.is_dir() else []
            if len(hits) != 1:
                empty["errors"].append(
                    f"expected one score artifact for {category}, found {len(hits)}")
            else:
                score_files.append(hits[0])
    elif score_root.is_dir():
        score_files = sorted(score_root.rglob("BFCL_v4_*_score.json"))
    if not score_files:
        return empty

    categories: List[Dict[str, Any]] = []
    seen_categories: set[str] = set()
    duplicate_category = False
    for score_path in score_files:
        match = SCORE_RE.match(score_path.name)
        if not match:
            continue
        category = match.group(1)
        if category in seen_categories:
            empty["errors"].append(f"duplicate score category {category}")
            duplicate_category = True
            continue
        seen_categories.add(category)
        relative_parent = score_path.relative_to(score_root).parent
        direct_prediction = (result_root / relative_parent
                             / f"BFCL_v4_{category}_result.json")
        if direct_prediction.is_file():
            prediction_path: Optional[Path] = direct_prediction
        else:
            hits = sorted(result_root.rglob(f"BFCL_v4_{category}_result.json")) \
                if result_root.is_dir() else []
            prediction_path = hits[0] if len(hits) == 1 else None
            if len(hits) > 1:
                empty["errors"].append(
                    f"ambiguous prediction artifacts for {category}: {len(hits)}")
        categories.append(_category_artifacts(score_path, prediction_path, category))

    if duplicate_category:
        empty["official_categories"] = categories
        empty["scorer_status"] = "invalid_or_partial_headers"
        return empty
    if expected and set(expected) != seen_categories:
        empty["errors"].append(
            f"score category coverage mismatch: expected={expected} found={sorted(seen_categories)}")
        empty["official_categories"] = categories
        empty["scorer_status"] = "invalid_or_partial_headers"
        return empty
    if not categories or any(not item["header_valid"] for item in categories):
        empty["official_categories"] = categories
        empty["scorer_status"] = "invalid_or_partial_headers"
        return empty

    denominator = sum(item["total_count"] for item in categories)
    numerator = sum(item["correct_count"] for item in categories)
    score = numerator / denominator
    mismatches: List[str] = []
    if summary:
        for key, expected_value in (("n_scored", denominator),
                                    ("correct_count", numerator),
                                    ("n_total", denominator)):
            value = summary.get(key)
            if value is not None and value != expected_value:
                mismatches.append(
                    f"summary {key}={value!r} != official headers {expected_value!r}")
        value = summary.get("semantic_score")
        if value is not None and (not _is_number(value) or not math.isclose(
                float(value), score, rel_tol=0.0, abs_tol=1e-12)):
            mismatches.append(
                f"summary semantic_score={value!r} != official headers {score!r}")
    empty["official_categories"] = categories
    empty["errors"].extend(error for item in categories for error in item["errors"])
    if mismatches:
        empty["errors"].extend(mismatches)
        empty["scorer_status"] = "summary_header_mismatch"
        return empty

    correctness: Dict[Tuple[str, str], bool] = {}
    for item in categories:
        if item["correctness_status"] == "verified":
            correctness.update({(item["category"], task_id): value
                                for task_id, value in item["correctness"].items()})
    if correctness and all(item["correctness_status"] == "verified"
                           for item in categories):
        correctness_status = "verified"
    elif correctness:
        correctness_status = "partial"
    else:
        correctness_status = "unavailable"
    empty.update({
        "scorer_status": "verified_official_headers",
        "scorer_numerator": numerator,
        "scorer_denominator": denominator,
        "scorer_score": score,
        "task_correctness_status": correctness_status,
        "task_correctness": correctness,
    })
    return empty


def _strict_derived_after_tokens(ok_rows: Sequence[Mapping[str, Any]]) -> Optional[float]:
    if not ok_rows:
        return None
    values: List[float] = []
    for row in ok_rows:
        gist = row.get("gist_tokens")
        if not _is_number(gist):
            return None
        recovery = row.get("gold_recovery")
        extra = 0
        if isinstance(recovery, dict) and recovery.get("status") == "appended":
            if not _is_number(recovery.get("recovery_block_tokens")):
                return None
            extra = recovery["recovery_block_tokens"]
        values.append(float(gist) + float(extra))
    return sum(values)


def _packed_history_accounting(
    row: Mapping[str, Any],
) -> Optional[Tuple[float, float, int, Optional[float]]]:
    """Validate one proxy row's pre-selection packed-history denominator.

    ``None`` is intentional for a native/full request and for legacy rows;
    callers keep its coverage separate from the selected-history ledger.
    """
    original = row.get("history_packed_original_tokens")
    dropped = row.get("history_dropped_original_tokens")
    candidates = row.get("history_packed_candidate_doc_count")
    retained = row.get("history_retained_fraction")
    if (not _is_number(original) or float(original) < 0
            or not _is_number(dropped) or float(dropped) < 0
            or not _is_int(candidates) or candidates < 0):
        return None
    original_f, dropped_f = float(original), float(dropped)
    if dropped_f > original_f:
        return None
    expected = None if original_f == 0 else (original_f - dropped_f) / original_f
    if expected is None:
        if retained is not None:
            return None
    elif (not _is_number(retained)
          or not math.isclose(float(retained), expected, rel_tol=0.0, abs_tol=1e-9)):
        return None
    return original_f, dropped_f, candidates, expected


def _request_metrics(path: Optional[Path], resolution: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "request_log_status": resolution, "request_log_path": str(path) if path else None,
        "request_count": None, "ok_count": None, "error_count": None,
        "proxy_request_wall_sec_sum": None, "proxy_request_wall_sec_p50": None,
        "proxy_request_wall_sec_p90": None,
        "proxy_request_wall_scope": None,
        "original_history_tokens_sum": None,
        "history_packed_accounting_status": "unavailable",
        "history_packed_accounting_coverage_ok_request_count": None,
        "history_packed_accounting_request_count": None,
        "history_packed_whole_history_unknown_request_count": None,
        "history_packed_original_tokens_sum": None,
        "history_dropped_original_tokens_sum": None,
        "history_packed_candidate_doc_count_sum": None,
        "history_packed_retained_fraction": None,
        "compressed_history_tokens_before_recovery_sum": None,
        "compressed_history_tokens_after_recovery_sum": None,
        "history_tensor_full_equivalent_bytes_sum": None,
        "history_tensor_before_recovery_bytes_sum": None,
        "history_tensor_after_recovery_bytes_sum": None,
        "kv_history_full_equivalent_tokens_sum": None,
        "kv_history_full_equivalent_tokens_source": None,
        "kv_history_full_equivalent_tokens_source_counts": None,
        "kv_history_reported_full_equivalent_tokens_sum": None,
        "kv_history_active_tokens_sum": None,
        "kv_history_active_tokens_source": None,
        "kv_history_active_tokens_source_counts": None,
        "kv_history_active_tokens_missing_count": None,
        "kv_history_reported_active_tokens_sum": None,
        "kv_history_tensor_full_equivalent_bytes_sum": None,
        "kv_history_reported_tensor_full_equivalent_bytes_sum": None,
        "kv_history_tensor_active_bytes_sum": None,
        "kv_history_reported_tensor_active_bytes_sum": None,
        "dropped_docs_total": None, "dropped_requests": None,
        "recovery_append_request_count": None,
        "recovery_distinct_event_count": None,
        "raw_recompute_event_count": None, "raw_recompute_wall_sec_sum": None,
        "compressor_metrics_status": "unavailable", "compressor_calls": None,
        "compressor_prompt_tokens": None, "compressor_completion_tokens": None,
        "compressor_wall_sec": None, "retrieval_calls": None,
        "retrieval_prompt_tokens": None, "retrieval_completion_tokens": None,
        "native_tool_call_presence_status": "unavailable",
        "native_tool_call_presence_numerator": None,
        "native_tool_call_presence_denominator": None,
        "native_tool_call_presence_rate": None,
        "request_log_errors": [],
    }
    out["hbm_snapshot_scope"] = HBM_SNAPSHOT_SCOPE
    out["byte_sum_scope"] = BYTE_SUM_SCOPE
    out["logical_prompt_kv_scope"] = PROMPT_KV_SCOPE
    out["kv_history_reconstruction_scope"] = KV_HISTORY_RECONSTRUCTION_SCOPE
    out["kv_history_reported_scope"] = KV_HISTORY_REPORTED_SCOPE
    out["history_tensor_accounting_request_count"] = None
    out["history_tensor_coverage_ok_request_count"] = None
    out["kv_history_tensor_accounting_request_count"] = None
    out["kv_history_reported_accounting_request_count"] = None
    out["kv_history_tensor_coverage_ok_request_count"] = None
    missing_distribution = {"count": None, "p50": None, "max": None}
    for prefix in (
        "total_gpu_kv_bytes_snapshot",
        "physical_main_kv_bytes_snapshot",
        "physical_c2kv_pool_bytes_snapshot",
        "logical_prompt_kv_bytes",
        "history_tensor_full_equivalent_bytes",
        "history_tensor_before_recovery_bytes",
        "history_tensor_after_recovery_bytes",
        "kv_history_tensor_full_equivalent_bytes",
        "kv_history_reported_tensor_full_equivalent_bytes",
        "kv_history_tensor_active_bytes",
        "kv_history_reported_tensor_active_bytes",
    ):
        _put_distribution(out, prefix, missing_distribution)
    if path is None:
        return out
    rows, errors = _read_records(path)
    if errors:
        out["request_log_status"] = "invalid_or_partial"
        out["request_log_errors"] = errors
        return out
    if any("status" not in row for row in rows):
        out["request_log_status"] = "invalid_schema"
        out["request_log_errors"] = [f"{path}: request row lacks status"]
        return out
    out["request_log_status"] = "available"
    ok = [row for row in rows if row.get("status") == "ok"]
    out.update({
        "request_count": len(rows), "ok_count": len(ok),
        "error_count": len(rows) - len(ok),
    })
    for field in ("total_gpu_kv_bytes", "physical_main_kv_bytes",
                  "physical_c2kv_pool_bytes"):
        _put_distribution(
            out, f"{field}_snapshot",
            _observed_distribution(row.get(field) for row in ok))
    _put_distribution(
        out, "logical_prompt_kv_bytes",
        _observed_distribution(_logical_prompt_kv_bytes(row) for row in ok))
    walls = [row.get("wall_sec") for row in ok]
    if walls and all(_is_number(value) for value in walls):
        wall_values = [float(value) for value in walls]
        out.update({
            "proxy_request_wall_sec_sum": sum(wall_values),
            "proxy_request_wall_sec_p50": _percentile(wall_values, 0.50),
            "proxy_request_wall_sec_p90": _percentile(wall_values, 0.90),
            "proxy_request_wall_scope": (
                "sum and percentiles over status=ok proxy requests; distinct from "
                "BFCL task wall and runner adapter wall"),
        })
    out["original_history_tokens_sum"] = _number_sum(ok, "original_tokens")
    out["compressed_history_tokens_before_recovery_sum"] = _number_sum(ok, "gist_tokens")
    out["compressed_history_tokens_after_recovery_sum"] = _strict_derived_after_tokens(ok)
    dropped = [row.get("dropped_docs") for row in ok]
    if dropped and all(_is_number(value) for value in dropped):
        out["dropped_docs_total"] = sum(float(value) for value in dropped)
        out["dropped_requests"] = sum(float(value) > 0 for value in dropped)

    packed = [_packed_history_accounting(row) for row in ok]
    complete_packed = [item for item in packed if item is not None]
    unknown_whole_history = [
        row for row, item in zip(ok, packed)
        if (_is_number(row.get("dropped_docs"))
            and float(row["dropped_docs"]) > 0 and item is None)
    ]
    out["history_packed_accounting_coverage_ok_request_count"] = len(ok)
    out["history_packed_accounting_request_count"] = len(complete_packed)
    out["history_packed_whole_history_unknown_request_count"] = len(unknown_whole_history)
    if ok and len(complete_packed) == len(ok):
        original_total = sum(item[0] for item in complete_packed)
        dropped_total = sum(item[1] for item in complete_packed)
        out.update({
            "history_packed_accounting_status": "complete",
            "history_packed_original_tokens_sum": original_total,
            "history_dropped_original_tokens_sum": dropped_total,
            "history_packed_candidate_doc_count_sum": sum(item[2] for item in complete_packed),
            "history_packed_retained_fraction": (
                (original_total - dropped_total) / original_total
                if original_total else None),
        })
    elif unknown_whole_history:
        out["history_packed_accounting_status"] = "legacy_dropped_whole_history_unknown"
    elif ok:
        out["history_packed_accounting_status"] = "partial_or_not_applicable"

    tensor = [row["history_tensor_accounting"] for row in ok
              if isinstance(row.get("history_tensor_accounting"), dict)]
    out["history_tensor_coverage_ok_request_count"] = len(ok)
    out["history_tensor_accounting_request_count"] = len(tensor)
    if tensor:
        out["history_tensor_full_equivalent_bytes_sum"] = _number_sum(
            tensor, "full_equivalent_selected_history_bytes")
        out["history_tensor_before_recovery_bytes_sum"] = _number_sum(
            tensor, "before_recovery_bytes")
        out["history_tensor_after_recovery_bytes_sum"] = _number_sum(
            tensor, "after_recovery_bytes")
    for field, prefix in (
        ("full_equivalent_selected_history_bytes",
         "history_tensor_full_equivalent_bytes"),
        ("before_recovery_bytes", "history_tensor_before_recovery_bytes"),
        ("after_recovery_bytes", "history_tensor_after_recovery_bytes"),
    ):
        _put_distribution(
            out, prefix, _complete_distribution(row.get(field) for row in tensor))

    kv_rows = [row for row in ok if "history_kv_full_equivalent_tokens" in row
               or "history_kv_active_tokens" in row
               or "history_kv_selected_tokens" in row
               or isinstance(row.get("history_kv_selection"), dict)
               or (_is_number(row.get("original_tokens"))
                   and float(row["original_tokens"]) > 0
                   and _is_number(row.get("gist_tokens"))
                   and float(row["gist_tokens"]) > 0
                   and isinstance(row.get("c2kv_layout"), list))]
    out["kv_history_tensor_coverage_ok_request_count"] = len(ok)
    reconstructed_active = [_history_kv_active_tokens(row) for row in kv_rows]
    active_values = [value for value, _ in reconstructed_active]
    active_sources = [source for value, source in reconstructed_active
                      if value is not None and source is not None]
    source_counts = Counter(active_sources)
    out["kv_history_tensor_accounting_request_count"] = sum(
        value is not None for value in active_values)
    out["kv_history_active_tokens_missing_count"] = sum(
        value is None for value in active_values)
    out["kv_history_active_tokens_source_counts"] = dict(sorted(source_counts.items()))
    if len(source_counts) == 1:
        out["kv_history_active_tokens_source"] = next(iter(source_counts))
    elif source_counts:
        out["kv_history_active_tokens_source"] = "mixed"

    reconstructed_full = [
        _history_kv_full_equivalent_tokens(row) for row in kv_rows
    ]
    full_values = [value for value, _ in reconstructed_full]
    full_sources = [source for value, source in reconstructed_full
                    if value is not None and source is not None]
    full_source_counts = Counter(full_sources)
    out["kv_history_full_equivalent_tokens_source_counts"] = dict(
        sorted(full_source_counts.items()))
    if len(full_source_counts) == 1:
        out["kv_history_full_equivalent_tokens_source"] = next(
            iter(full_source_counts))
    elif full_source_counts:
        out["kv_history_full_equivalent_tokens_source"] = "mixed"

    reported_full_values = [
        row.get("history_kv_full_equivalent_tokens") for row in kv_rows
    ]
    reported_values = [row.get("history_kv_active_tokens") for row in kv_rows]
    out["kv_history_reported_accounting_request_count"] = sum(
        _is_number(value) for value in reported_values)
    if kv_rows and all(_is_number(value) for value in reported_values):
        out["kv_history_reported_active_tokens_sum"] = sum(
            float(value) for value in reported_values)

    if kv_rows and all(_is_number(value) for value in reported_full_values):
        out["kv_history_reported_full_equivalent_tokens_sum"] = sum(
            float(value) for value in reported_full_values)
    if kv_rows and all(value is not None for value in full_values):
        out["kv_history_full_equivalent_tokens_sum"] = sum(
            float(value) for value in full_values if value is not None)
    if kv_rows and all(value is not None for value in active_values):
        out["kv_history_active_tokens_sum"] = sum(
            float(value) for value in active_values if value is not None)
    if kv_rows and all(value is not None for value in full_values) and all(
            _is_number(row.get("bytes_per_kv_token")) for row in kv_rows):
        out["kv_history_tensor_full_equivalent_bytes_sum"] = sum(
            float(value) * float(row["bytes_per_kv_token"])
            for row, value in zip(kv_rows, full_values) if value is not None)
    if kv_rows and all(_is_number(value) for value in reported_full_values) and all(
            _is_number(row.get("bytes_per_kv_token")) for row in kv_rows):
        out["kv_history_reported_tensor_full_equivalent_bytes_sum"] = sum(
            float(value) * float(row["bytes_per_kv_token"])
            for row, value in zip(kv_rows, reported_full_values))
    if kv_rows and all(value is not None for value in active_values) and all(
            _is_number(row.get("bytes_per_kv_token")) for row in kv_rows):
        out["kv_history_tensor_active_bytes_sum"] = sum(
            float(value) * float(row["bytes_per_kv_token"])
            for row, value in zip(kv_rows, active_values) if value is not None)
    if kv_rows and all(_is_number(value) for value in reported_values) and all(
            _is_number(row.get("bytes_per_kv_token")) for row in kv_rows):
        out["kv_history_reported_tensor_active_bytes_sum"] = sum(
            float(value) * float(row["bytes_per_kv_token"])
            for row, value in zip(kv_rows, reported_values))
    kv_full_bytes = [
        (float(value) * float(row["bytes_per_kv_token"]))
        if (value is not None
            and _is_number(row.get("bytes_per_kv_token"))) else None
        for row, value in zip(kv_rows, full_values)
    ]
    kv_reported_full_bytes = [
        (float(value) * float(row["bytes_per_kv_token"]))
        if (_is_number(value) and _is_number(row.get("bytes_per_kv_token"))) else None
        for row, value in zip(kv_rows, reported_full_values)
    ]
    kv_active_bytes = [
        (float(value)
         * float(row["bytes_per_kv_token"]))
        if (value is not None
            and _is_number(row.get("bytes_per_kv_token"))) else None
        for row, value in zip(kv_rows, active_values)
    ]
    kv_reported_active_bytes = [
        (float(value) * float(row["bytes_per_kv_token"]))
        if (_is_number(value) and _is_number(row.get("bytes_per_kv_token"))) else None
        for row, value in zip(kv_rows, reported_values)
    ]
    _put_distribution(
        out, "kv_history_tensor_full_equivalent_bytes",
        _complete_distribution(kv_full_bytes))
    _put_distribution(
        out, "kv_history_reported_tensor_full_equivalent_bytes",
        _complete_distribution(kv_reported_full_bytes))
    _put_distribution(
        out, "kv_history_tensor_active_bytes",
        _complete_distribution(kv_active_bytes))
    _put_distribution(
        out, "kv_history_reported_tensor_active_bytes",
        _complete_distribution(kv_reported_active_bytes))

    recovery = [row["gold_recovery"] for row in ok
                if isinstance(row.get("gold_recovery"), dict)]
    if recovery and all(isinstance(item.get("status"), str) for item in recovery):
        appended = [item for item in recovery if item.get("status") == "appended"]
        out["recovery_append_request_count"] = len(appended)
        if all(item.get("event_id") is not None for item in appended):
            try:
                out["recovery_distinct_event_count"] = len({
                    json.dumps(item["event_id"], sort_keys=True, separators=(",", ":"))
                    for item in appended
                })
            except (TypeError, ValueError):
                pass
        if all(isinstance(item.get("raw_kv_cache_hit"), bool) for item in appended):
            recomputed = [item for item in appended
                          if item.get("raw_kv_cache_hit") is False]
            out["raw_recompute_event_count"] = len(recomputed)
        else:
            recomputed = None
        if (recomputed is not None
                and all(_is_number(item.get("recovery_extract_sec"))
                        for item in recomputed)):
            out["raw_recompute_wall_sec_sum"] = sum(
                float(item["recovery_extract_sec"]) for item in recomputed)

    text_rows = [row["textarm"] for row in ok if isinstance(row.get("textarm"), dict)]
    if text_rows:
        compressor = [item.get("compressor_usage") for item in text_rows]
        retrieval = [item.get("retrieval_usage") for item in text_rows]
        if (all(_is_int(item.get("n_compressor_calls")) for item in text_rows)
                and all(isinstance(item, dict) for item in compressor)
                and all(_is_number(item.get(key)) for item in compressor
                        for key in ("calls", "prompt_tokens", "completion_tokens", "wall_sec"))
                and all(item["n_compressor_calls"] == usage["calls"]
                        for item, usage in zip(text_rows, compressor))):
            out.update({
                "compressor_metrics_status": "available",
                "compressor_calls": sum(item["calls"] for item in compressor),
                "compressor_prompt_tokens": sum(item["prompt_tokens"] for item in compressor),
                "compressor_completion_tokens": sum(
                    item["completion_tokens"] for item in compressor),
                "compressor_wall_sec": sum(float(item["wall_sec"]) for item in compressor),
            })
        if all(isinstance(item, dict) for item in retrieval) and all(
                _is_number(item.get(key)) for item in retrieval
                for key in ("calls", "prompt_tokens", "completion_tokens")):
            out.update({
                "retrieval_calls": sum(item["calls"] for item in retrieval),
                "retrieval_prompt_tokens": sum(item["prompt_tokens"] for item in retrieval),
                "retrieval_completion_tokens": sum(
                    item["completion_tokens"] for item in retrieval),
            })

    if ok and all(_is_int(row.get("n_tools")) for row in ok):
        eligible = [row for row in ok if row["n_tools"] > 0]
        out["native_tool_call_presence_denominator"] = len(eligible)
        if not eligible:
            out["native_tool_call_presence_status"] = "no_eligible_requests"
        elif all(_is_int(row.get("n_native_tool_calls")) for row in eligible):
            numerator = sum(row["n_native_tool_calls"] > 0 for row in eligible)
            out.update({
                "native_tool_call_presence_status": "available",
                "native_tool_call_presence_numerator": numerator,
                "native_tool_call_presence_rate": numerator / len(eligible),
            })
        else:
            out["native_tool_call_presence_status"] = "unavailable_missing_field"
    elif ok:
        out["native_tool_call_presence_status"] = "unavailable_missing_field"
    return out


def _task_audit_metrics(path: Optional[Path], resolution: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "task_audit_path": str(path) if path else None,
        "task_wall_status": resolution, "task_wall_count": None,
        "task_wall_sec_sum": None, "task_wall_sec_p50": None,
        "task_wall_sec_p90": None, "task_wall_scope": None,
        "gold_metrics_status": "unavailable",
        "gold_trigger_count": None, "gold_intervene_count": None,
        "gold_recovered_count": None, "gold_no_witness_count": None,
        "task_audit_errors": [],
    }
    if path is None:
        return out
    rows, errors = _read_records(path)
    if errors:
        out["task_wall_status"] = "invalid_or_partial"
        out["task_audit_errors"] = errors
        return out
    task_ids = [row.get("task_id") for row in rows]
    task_ids_valid = (all(task_id is not None for task_id in task_ids)
                      and len({str(task_id) for task_id in task_ids}) == len(task_ids))
    walls = [row.get("total_wall_seconds") for row in rows]
    if rows and task_ids_valid and all(_is_number(value) for value in walls):
        values = [float(value) for value in walls]
        out.update({
            "task_wall_status": "available", "task_wall_count": len(values),
            "task_wall_sec_sum": sum(values),
            "task_wall_sec_p50": _percentile(values, 0.50),
            "task_wall_sec_p90": _percentile(values, 0.90),
            "task_wall_scope": (
                "sum and percentiles over BFCL task_audit.total_wall_seconds; "
                "includes per-task generation, tool execution and gold checks"),
        })
    elif rows:
        out["task_wall_status"] = (
            "invalid_task_ids" if not task_ids_valid else "unavailable_missing_field")
    else:
        out["task_wall_status"] = "available_empty"
        out["task_wall_count"] = 0

    oracle_rows = [row for row in rows if row.get("selector") is not None]
    required = ("trigger_turn", "recovered", "did_intervene", "intervention_statuses")
    if (rows and all("selector" in row for row in rows)
            and all(all(key in row for key in required) for row in oracle_rows)):
        out.update({
            "gold_metrics_status": "available",
            "gold_trigger_count": sum(row["trigger_turn"] is not None for row in oracle_rows),
            "gold_intervene_count": sum(row["did_intervene"] is True for row in oracle_rows),
            "gold_recovered_count": sum(row["recovered"] is True for row in oracle_rows),
            "gold_no_witness_count": sum(
                "no_literal_witness" in (row["intervention_statuses"] or [])
                for row in oracle_rows),
        })
    return out


def _task_id_coverage(expected: Mapping[Tuple[str, str], bool],
                      request_path: Optional[Path], audit_path: Optional[Path]) -> Dict[str, Any]:
    out = {
        "task_id_coverage_status": "unavailable",
        "task_id_coverage_expected_count": len(expected) if expected else None,
        "task_id_coverage_audit_count": None,
        "task_id_coverage_proxy_count": None,
        "task_id_coverage_missing_audit": None,
        "task_id_coverage_missing_proxy": None,
        "task_id_coverage_extra_audit": None,
        "task_id_coverage_extra_proxy": None,
    }
    if not expected or request_path is None or audit_path is None:
        return out
    requests, request_errors = _read_records(request_path)
    audits, audit_errors = _read_records(audit_path)
    if request_errors or audit_errors:
        out["task_id_coverage_status"] = "invalid_artifacts"
        return out
    wanted = {task_id for _, task_id in expected}
    audit_ids = [str(row["task_id"]) for row in audits if row.get("task_id") is not None]
    proxy_ids = [str(row["eval_context"]["task_id"]) for row in requests
                 if isinstance(row.get("eval_context"), dict)
                 and row["eval_context"].get("task_id") is not None]
    actual_audit, actual_proxy = set(audit_ids), set(proxy_ids)
    out.update({
        "task_id_coverage_audit_count": len(actual_audit),
        "task_id_coverage_proxy_count": len(actual_proxy),
        "task_id_coverage_missing_audit": sorted(wanted - actual_audit),
        "task_id_coverage_missing_proxy": sorted(wanted - actual_proxy),
        "task_id_coverage_extra_audit": sorted(actual_audit - wanted),
        "task_id_coverage_extra_proxy": sorted(actual_proxy - wanted),
        "task_id_coverage_status": "verified" if (
            actual_audit == wanted == actual_proxy
            and len(audit_ids) == len(audits) == len(actual_audit)
            and len(proxy_ids) == len(requests)) else "mismatch",
    })
    return out


def collect_cell(root: Path, cell: Cell) -> Tuple[Dict[str, Any], Dict[Tuple[str, str], bool]]:
    summary: Optional[Dict[str, Any]]
    if cell.summary_path.is_file():
        summary, summary_error = _read_object(cell.summary_path)
        summary_status = "available" if summary is not None else "invalid"
    else:
        summary, summary_error = None, None
        summary_status = "missing"
    errors = [summary_error] if summary_error else []
    if summary and summary.get("arm") not in (None, cell.arm):
        errors.append(
            f"summary arm {summary.get('arm')!r} does not match cell arm {cell.arm!r}")

    request_path, request_resolution = _resolve_named_artifact(
        root, cell.run_dir, summary.get("request_log") if summary else None,
        (f"logs/proxy_{cell.arm}_*.jsonl", "logs/request_log.jsonl",
         f"proxy_{cell.arm}_*.jsonl", "request_log.jsonl"))
    audit_path, audit_resolution = _resolve_named_artifact(
        root, cell.run_dir, summary.get("task_telemetry_path") if summary else None,
        (f"task_audit/c2kv-{cell.arm.replace('_', '-')}.jsonl", "task_audit/*.jsonl"))
    official = _official_metrics(cell.run_dir, cell.arm, summary)
    request = _request_metrics(request_path, request_resolution)
    audit = _task_audit_metrics(audit_path, audit_resolution)
    coverage = _task_id_coverage(official["task_correctness"], request_path, audit_path)

    runner_wall = summary.get("runner_adapter_wall_sec") if summary else None
    if not _is_number(runner_wall):
        runner_wall = None
    runner_scope = summary.get("runner_wall_scope") if summary else None
    row: Dict[str, Any] = {
        "cell_id": cell.cell_id, "arm": cell.arm,
        "arm_role": ("native_full" if cell.arm == "full_native" else
                     "dialect_control" if cell.arm == "full" else "comparison_arm"),
        "cell_status": "complete" if (summary_status == "available"
                                         and official["scorer_status"] == "verified_official_headers"
                                         and official["task_correctness_status"] == "verified"
                                         and request["request_log_status"] == "available"
                                         and coverage["task_id_coverage_status"] == "verified"
                                         and audit["task_wall_status"] == "available")
        else "partial",
        "preliminary_n": 1,
        "summary_status": summary_status,
        "summary_path": str(cell.summary_path),
        "runner_adapter_wall_sec": float(runner_wall) if runner_wall is not None else None,
        "runner_wall_scope": runner_scope,
        **{key: value for key, value in official.items() if key != "task_correctness"},
        **request,
        **audit,
        **coverage,
    }
    row["errors"] = [str(value) for value in (
        errors + official["errors"] + request["request_log_errors"]
        + audit["task_audit_errors"]) if value]
    return row, official["task_correctness"]


def _reference(rows: Sequence[Mapping[str, Any]]) -> Tuple[Optional[int], str, List[str]]:
    warnings: List[str] = []
    native = [index for index, row in enumerate(rows) if row["arm"] == "full_native"]
    if native:
        if len(native) == 1:
            return native[0], "native_full_reference", warnings
        warnings.append("multiple full_native cells; paired reference is ambiguous")
        return None, "ambiguous", warnings
    dialect = [index for index, row in enumerate(rows) if row["arm"] == "full"]
    if len(dialect) == 1:
        warnings.append("full_native is absent; full is retained as an explicit dialect control")
        return dialect[0], "dialect_control", warnings
    if len(dialect) > 1:
        warnings.append("multiple full dialect-control cells; paired reference is ambiguous")
        return None, "ambiguous", warnings
    warnings.append("no full_native or full reference cell")
    return None, "unavailable", warnings


def _paired_rows(
    rows: Sequence[Mapping[str, Any]],
    correctness: Sequence[Mapping[Tuple[str, str], bool]],
    reference_index: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if reference_index is None:
        return [], []
    reference = rows[reference_index]
    reference_map = correctness[reference_index]
    pairs: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    labels = {(True, True): "both_correct", (True, False): "reference_only",
              (False, True): "arm_only", (False, False): "both_incorrect"}
    for index, row in enumerate(rows):
        if index == reference_index:
            continue
        arm_map = correctness[index]
        counts: Counter[str] = Counter()
        keys = sorted(set(reference_map) | set(arm_map))
        for category, task_id in keys:
            reference_correct = reference_map.get((category, task_id))
            arm_correct = arm_map.get((category, task_id))
            if reference_correct is None:
                status, quadrant = "missing_reference_correctness", None
            elif arm_correct is None:
                status, quadrant = "missing_arm_correctness", None
            else:
                status = "paired"
                quadrant = labels[(reference_correct, arm_correct)]
                counts[quadrant] += 1
            pairs.append({
                "category": category, "official_id": task_id,
                "reference_cell_id": reference["cell_id"],
                "reference_arm": reference["arm"],
                "arm_cell_id": row["cell_id"], "arm": row["arm"],
                "reference_correct": reference_correct,
                "arm_correct": arm_correct, "pair_status": status,
                "quadrant": quadrant,
            })
        summaries.append({
            "reference_cell_id": reference["cell_id"], "reference_arm": reference["arm"],
            "arm_cell_id": row["cell_id"], "arm": row["arm"],
            "joined_id_count": sum(counts.values()),
            "union_id_count": len(keys),
            "unavailable_id_count": len(keys) - sum(counts.values()),
            **{label: counts[label] for label in labels.values()},
        })
    return pairs, summaries


def collect(root: Path, *, include_smoke: bool = False
            ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    root = root.resolve()
    cells, warnings = discover_cells(root, include_smoke=include_smoke)
    rows: List[Dict[str, Any]] = []
    correctness: List[Mapping[Tuple[str, str], bool]] = []
    for cell in cells:
        row, task_map = collect_cell(root, cell)
        rows.append(row)
        correctness.append(task_map)
    reference_index, reference_role, reference_warnings = _reference(rows)
    warnings.extend(reference_warnings)
    pairs, paired_counts = _paired_rows(rows, correctness, reference_index)
    reference = None if reference_index is None else {
        "cell_id": rows[reference_index]["cell_id"],
        "arm": rows[reference_index]["arm"], "role": reference_role,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "include_smoke": include_smoke,
        "preliminary": "preliminary, n=1 per arm",
        "metric_scopes": {
            "process_hbm_snapshots": HBM_SNAPSHOT_SCOPE,
            "byte_request_sums": BYTE_SUM_SCOPE,
            "logical_prompt_kv": PROMPT_KV_SCOPE,
            "kv_history_reconstruction": KV_HISTORY_RECONSTRUCTION_SCOPE,
            "kv_history_reported": KV_HISTORY_REPORTED_SCOPE,
            "packed_history_denominator": HISTORY_PACKED_SCOPE,
        },
        "reference": reference,
        "reference_role": reference_role,
        "rows": rows,
        "paired_counts": paired_counts,
        "warnings": warnings,
    }
    return report, pairs


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_outputs(root: Path, report: Mapping[str, Any],
                  paired: Sequence[Mapping[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8")
    with (root / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in report["rows"]:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})
    with (root / "paired_tasks.jsonl").open("w", encoding="utf-8") as handle:
        for row in paired:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="benchmark matrix root containing cells/ and matrix_plan.json")
    parser.add_argument("--include-smoke", action="store_true",
                        help="include run directories whose name starts with smoke")
    args = parser.parse_args(argv)
    report, paired = collect(args.root, include_smoke=args.include_smoke)
    write_outputs(args.root, report, paired)
    print(f"wrote {len(report['rows'])} arm rows and {len(paired)} paired task rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
