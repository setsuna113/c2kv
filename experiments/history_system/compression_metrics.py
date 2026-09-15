"""Task-equal aggregation for C2KV mixed-history compression telemetry.

The primary per-receipt metric is ``system_active_history_reduction = H / A``:

* ``H`` is ``compression_ratio.full_history_bytes`` from the Full renderer on
  the same observable prefix.
* ``A`` is the system's actual active history footprint,
  ``compression_ratio.active_history_bytes``.  It includes gist KV and any
  retained raw history.
* ``S`` is ``compression_ratio.common_live_bytes``.  The separate full-input
  metric is ``(S + H) / (S + A)``.

A receipt is activated when ``H > 0`` and ``A > 0``.  ``H == A == 0`` is a
no-history warmup and is excluded.  ``H > 0`` with zero or missing ``A`` is an
anomaly, not infinite compression.  Coverage-losing receipts remain in the
primary mixed-system metric because selection and eviction are part of the
system; their loss flags and source counts are always reported alongside it.
A strict complete-coverage subset is reported separately.

Aggregation first computes a median and bounded linear-interpolation low
quantile within each selected task.  It then gives every task one equal weight
by averaging those task summaries.  Step rows are never pooled across tasks,
so a long or failing trajectory cannot dominate the aggregate.  Missing tasks
remain in the fixed denominator.  Known-subset values are labelled as such,
and all-selected-task values are null unless every selected task contributes.

This module only reads JSONL telemetry and uses the Python standard library.
It performs no model, scorer, network, or repository writes unless its CLI is
given an explicit ``--output`` path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import json
import math
from pathlib import Path
import statistics
from typing import Any


METRIC_SCHEMA = "c2kv-history-system-compression-metrics-v1"
RATIO_SCHEMA = "a-same-prefix-compression-ratio-v1"
DEFAULT_LOW_QUANTILE = 0.10


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def linear_quantile(values: Iterable[float], probability: float) -> float | None:
    """Return the bounded type-7 empirical quantile used by this analysis."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("quantile values must be finite")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def weighted_linear_quantile(
    value_weights: Iterable[tuple[float, float]], probability: float
) -> float | None:
    """Interpolate a weighted empirical CDF at bin midpoints.

    Equal task weighting is obtained by assigning every activated receipt in a
    task weight ``1 / activated_receipts_in_that_task``.  Equal-valued bins are
    merged before interpolation, so duplicate ordering cannot change a result.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    combined: dict[float, float] = {}
    for value, weight in value_weights:
        value = float(value)
        weight = float(weight)
        if not math.isfinite(value) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("weighted quantile values and weights must be finite and positive")
        combined[value] = combined.get(value, 0.0) + weight
    if not combined:
        return None
    ordered = sorted(combined.items())
    total = sum(weight for _, weight in ordered)
    positions: list[float] = []
    cumulative = 0.0
    for _, weight in ordered:
        positions.append((cumulative + 0.5 * weight) / total)
        cumulative += weight
    if probability <= positions[0]:
        return ordered[0][0]
    if probability >= positions[-1]:
        return ordered[-1][0]
    for index in range(1, len(ordered)):
        if probability <= positions[index]:
            lower_position = positions[index - 1]
            upper_position = positions[index]
            fraction = (probability - lower_position) / (
                upper_position - lower_position
            )
            return ordered[index - 1][0] * (1.0 - fraction) + ordered[index][
                0
            ] * fraction
    raise AssertionError("weighted quantile interpolation did not terminate")


def _distribution(values: Iterable[float], low_quantile: float) -> dict[str, Any]:
    finite = [float(value) for value in values]
    if not finite:
        return {
            "count": 0,
            "minimum": None,
            "low_quantile": None,
            "median": None,
            "mean": None,
            "maximum": None,
        }
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("distribution values must be finite")
    return {
        "count": len(finite),
        "minimum": min(finite),
        "low_quantile": linear_quantile(finite, low_quantile),
        "median": statistics.median(finite),
        "mean": statistics.fmean(finite),
        "maximum": max(finite),
    }


def _index_set(
    coverage: Mapping[str, Any], field: str, errors: list[str]
) -> set[int] | None:
    value = coverage.get(field)
    if not isinstance(value, list) or any(
        not isinstance(index, int) or isinstance(index, bool) for index in value
    ):
        errors.append(f"invalid_source_coverage_field:{field}")
        return None
    if len(set(value)) != len(value):
        errors.append(f"duplicate_source_coverage_indices:{field}")
        return None
    return set(value)


def _coverage_receipt(
    controller: Mapping[str, Any], ratio: Mapping[str, Any], errors: list[str]
) -> dict[str, Any] | None:
    coverage = controller.get("source_coverage")
    if not isinstance(coverage, Mapping):
        errors.append("missing_or_invalid_source_coverage")
        return None

    eligible = _index_set(coverage, "eligible_source_indices", errors)
    raw = _index_set(coverage, "raw_source_indices", errors)
    gist_touched = _index_set(coverage, "gist_touched_source_indices", errors)
    gist_fully = _index_set(
        coverage, "gist_fully_represented_source_indices", errors
    )
    unrepresented = _index_set(
        coverage, "unrepresented_source_indices", errors
    )
    if None in (eligible, raw, gist_touched, gist_fully, unrepresented):
        return None

    assert eligible is not None
    assert raw is not None
    assert gist_touched is not None
    assert gist_fully is not None
    assert unrepresented is not None
    if not raw <= eligible:
        errors.append("raw_sources_outside_eligible_sources")
    if not gist_fully <= gist_touched:
        errors.append("gist_fully_represented_sources_not_touched")
    if not gist_touched <= eligible:
        errors.append("gist_touched_sources_outside_eligible_sources")
    if not unrepresented <= eligible:
        errors.append("unrepresented_sources_outside_eligible_sources")

    represented = eligible & (raw | gist_fully)
    expected_unrepresented = eligible - represented
    if unrepresented != expected_unrepresented:
        errors.append("source_coverage_set_algebra_mismatch")

    complete = coverage.get("complete_history_coverage")
    if not isinstance(complete, bool):
        errors.append("invalid_complete_history_coverage")
        complete = None
    elif complete != (not unrepresented):
        errors.append("complete_history_coverage_flag_mismatch")

    includes_loss = ratio.get("includes_coverage_loss")
    if not isinstance(includes_loss, bool):
        errors.append("invalid_includes_coverage_loss")
        includes_loss = None
    elif complete is not None and includes_loss == complete:
        errors.append("coverage_loss_ratio_flag_mismatch")

    coverage_loss: bool | None
    if includes_loss is True or complete is False:
        coverage_loss = True
    elif includes_loss is False and complete is True:
        coverage_loss = False
    else:
        coverage_loss = None

    fitted = coverage.get("fitted_encoder_input_tokens")
    retained = coverage.get("retained_encoder_input_tokens")
    if not _is_nonnegative_int(fitted):
        fitted = None
    if not _is_nonnegative_int(retained):
        retained = None
    if fitted is not None and retained is not None and retained > fitted:
        errors.append("retained_encoder_tokens_exceed_fitted_tokens")

    return {
        "eligible_source_occurrences": len(eligible),
        "raw_source_occurrences": len(raw),
        "gist_touched_source_occurrences": len(gist_touched),
        "gist_fully_represented_source_occurrences": len(gist_fully),
        "fully_represented_source_occurrences": len(represented),
        "unrepresented_source_occurrences": len(unrepresented),
        "gist_touched_but_unrepresented_source_occurrences": len(
            gist_touched - gist_fully - raw
        ),
        "fitted_encoder_input_tokens": fitted,
        "retained_encoder_input_tokens": retained,
        "complete_history_coverage": complete,
        "includes_coverage_loss": includes_loss,
        "coverage_loss": coverage_loss,
        "valid_set_algebra": not any(
            error
            in {
                "raw_sources_outside_eligible_sources",
                "gist_fully_represented_sources_not_touched",
                "gist_touched_sources_outside_eligible_sources",
                "unrepresented_sources_outside_eligible_sources",
                "source_coverage_set_algebra_mismatch",
                "complete_history_coverage_flag_mismatch",
                "coverage_loss_ratio_flag_mismatch",
            }
            for error in errors
        ),
    }


def _controller_observation(
    *,
    task_id: str,
    line_number: int,
    trace_index: int,
    step: Mapping[str, Any],
    trace: Mapping[str, Any],
    b0_cap_bytes: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    controller = trace.get("controller")
    if not isinstance(controller, Mapping):
        return None, ["missing_or_invalid_controller"]
    ratio = controller.get("compression_ratio")
    if not isinstance(ratio, Mapping):
        return None, ["missing_or_invalid_compression_ratio"]
    if ratio.get("schema") != RATIO_SCHEMA:
        errors.append("unexpected_compression_ratio_schema")

    h = ratio.get("full_history_bytes")
    s = ratio.get("common_live_bytes")
    a = ratio.get("active_history_bytes")
    if not _is_nonnegative_int(h):
        errors.append("invalid_full_history_bytes")
        h = None
    if not _is_nonnegative_int(s):
        errors.append("invalid_common_live_bytes")
        s = None
    if not _is_nonnegative_int(a):
        errors.append("invalid_active_history_bytes")
        a = None

    activation: str
    history_reduction: float | None = None
    full_input_reduction: float | None = None
    if h == 0 and a == 0:
        activation = "no_history_warmup"
    elif h is not None and h > 0 and a is not None and a > 0:
        activation = "activated"
        history_reduction = h / a
        if s is not None:
            full_input_reduction = (s + h) / (s + a)
    elif h is not None and h > 0 and (a is None or a == 0):
        activation = "anomaly_full_history_without_active_footprint"
    elif h == 0 and a is not None and a > 0:
        activation = "anomaly_active_footprint_without_full_history"
    else:
        activation = "anomaly_unknown_history_or_active_footprint"

    if history_reduction is not None:
        logged = ratio.get("n_history")
        if not _is_finite_number(logged) or not math.isclose(
            float(logged), history_reduction, rel_tol=1e-12, abs_tol=1e-12
        ):
            errors.append("logged_history_reduction_mismatch")
    if full_input_reduction is not None:
        logged = ratio.get("n_total")
        if not _is_finite_number(logged) or not math.isclose(
            float(logged), full_input_reduction, rel_tol=1e-12, abs_tol=1e-12
        ):
            errors.append("logged_full_input_reduction_mismatch")

    active_gist = ratio.get("active_gist_bytes")
    active_raw = ratio.get("active_raw_history_bytes")
    if _is_nonnegative_int(active_gist) and _is_nonnegative_int(active_raw):
        if a is not None and active_gist + active_raw != a:
            errors.append("active_history_component_sum_mismatch")
    else:
        active_gist = None
        active_raw = None

    coverage = _coverage_receipt(controller, ratio, errors)
    declared_caps = {
        field: controller.get(field)
        for field in (
            "history_budget_bytes",
            "workspace_budget_bytes",
            "shared_allocation_budget_bytes",
        )
        if _is_nonnegative_int(controller.get(field))
    }
    configured_ratio = ratio.get("configured_gist_ratio")
    if not _is_finite_number(configured_ratio) or float(configured_ratio) <= 0:
        configured_ratio = None

    return {
        "task_id": task_id,
        "line_number": line_number,
        "trace_index": trace_index,
        "decision_key": step.get("decision_key"),
        "trace_phase": trace.get("phase"),
        "trace_status": trace.get("status"),
        "trace_discarded": trace.get("discarded"),
        "activation": activation,
        "full_history_bytes": h,
        "common_live_bytes": s,
        "active_history_bytes": a,
        "full_input_bytes": s + h if s is not None and h is not None else None,
        "active_input_bytes": s + a if s is not None and a is not None else None,
        "active_gist_bytes": active_gist,
        "active_raw_history_bytes": active_raw,
        "system_active_history_reduction": history_reduction,
        "system_full_input_reduction": full_input_reduction,
        "coverage": coverage,
        "configured_gist_ratio": configured_ratio,
        "actual_source_to_gist_ratio": ratio.get("actual_source_to_gist_ratio"),
        "declared_caps": declared_caps,
        "b0_cap_bytes": b0_cap_bytes,
        "observed_b0_violation": a is not None and a > b0_cap_bytes,
        "errors": errors,
    }, errors


def _guard_receipts(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    value = step.get("pre_generation_budget_checks")
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _guard_failed(receipt: Mapping[str, Any]) -> bool | None:
    for field in ("within_cap", "passed", "ok"):
        value = receipt.get(field)
        if isinstance(value, bool):
            return not value
    status = receipt.get("status")
    if isinstance(status, str):
        normalized = status.lower()
        if normalized in {"passed", "ok", "within_cap", "within_budget"}:
            return False
        if normalized in {
            "failed",
            "violation",
            "over_cap",
            "over_budget",
            "mismatch",
        }:
            return True
    return None


def analyze_task_records(
    task_id: str,
    step_records: Sequence[Mapping[str, Any]],
    *,
    b0_cap_bytes: int,
    low_quantile: float = DEFAULT_LOW_QUANTILE,
    malformed_records: int = 0,
    malformed_line_numbers: Sequence[int] = (),
    source_line_numbers: Sequence[int] | None = None,
    source_path: str | None = None,
    source_exists: bool = True,
) -> dict[str, Any]:
    """Aggregate already decoded step records for one selected task."""
    if not task_id:
        raise ValueError("task_id must be nonempty")
    if not _is_nonnegative_int(b0_cap_bytes) or b0_cap_bytes == 0:
        raise ValueError("b0_cap_bytes must be a positive integer")
    if not 0.0 <= low_quantile <= 0.5:
        raise ValueError("low_quantile must be between zero and 0.5")
    if not _is_nonnegative_int(malformed_records):
        raise ValueError("malformed_records must be a nonnegative integer")
    if len(malformed_line_numbers) != malformed_records:
        raise ValueError("malformed line numbers must match malformed_records")
    if source_line_numbers is not None and len(source_line_numbers) != len(step_records):
        raise ValueError("source_line_numbers must match step_records")

    observations: list[dict[str, Any]] = []
    receipt_errors: list[dict[str, Any]] = []
    controllerless_rows = 0
    empty_trace_rows = 0
    invalid_trace_rows = 0
    guard_receipts: list[Mapping[str, Any]] = []
    for ordinal, step in enumerate(step_records, start=1):
        line_number = (
            source_line_numbers[ordinal - 1]
            if source_line_numbers is not None
            else ordinal
        )
        if not isinstance(step, Mapping):
            receipt_errors.append(
                {"line_number": line_number, "errors": ["step_record_not_object"]}
            )
            controllerless_rows += 1
            continue
        guard_receipts.extend(_guard_receipts(step))
        traces = step.get("generation_trace")
        if not isinstance(traces, list):
            invalid_trace_rows += 1
            controllerless_rows += 1
            receipt_errors.append(
                {"line_number": line_number, "errors": ["invalid_generation_trace"]}
            )
            continue
        if not traces:
            empty_trace_rows += 1
            controllerless_rows += 1
            continue
        found_controller = False
        for trace_index, trace in enumerate(traces):
            if not isinstance(trace, Mapping):
                receipt_errors.append(
                    {
                        "line_number": line_number,
                        "trace_index": trace_index,
                        "errors": ["generation_trace_not_object"],
                    }
                )
                continue
            guard_receipts.extend(_guard_receipts(trace))
            observation, errors = _controller_observation(
                task_id=task_id,
                line_number=line_number,
                trace_index=trace_index,
                step=step,
                trace=trace,
                b0_cap_bytes=b0_cap_bytes,
            )
            if observation is None:
                receipt_errors.append(
                    {
                        "line_number": line_number,
                        "trace_index": trace_index,
                        "errors": errors,
                    }
                )
                continue
            found_controller = True
            observations.append(observation)
            if errors:
                receipt_errors.append(
                    {
                        "line_number": line_number,
                        "trace_index": trace_index,
                        "errors": errors,
                    }
                )
        if not found_controller:
            controllerless_rows += 1

    activated = [
        observation
        for observation in observations
        if observation["activation"] == "activated"
    ]
    strict = [
        observation
        for observation in activated
        if observation["coverage"] is not None
        and observation["coverage"]["coverage_loss"] is False
        and observation["coverage"]["valid_set_algebra"] is True
    ]
    history_values = [
        observation["system_active_history_reduction"]
        for observation in activated
        if observation["system_active_history_reduction"] is not None
    ]
    full_input_values = [
        observation["system_full_input_reduction"]
        for observation in activated
        if observation["system_full_input_reduction"] is not None
    ]
    strict_history_values = [
        observation["system_active_history_reduction"]
        for observation in strict
        if observation["system_active_history_reduction"] is not None
    ]
    strict_full_input_values = [
        observation["system_full_input_reduction"]
        for observation in strict
        if observation["system_full_input_reduction"] is not None
    ]

    component_fields = (
        "full_history_bytes",
        "common_live_bytes",
        "active_history_bytes",
        "full_input_bytes",
        "active_input_bytes",
    )
    component_distributions = {
        field: _distribution(
            (
                observation[field]
                for observation in activated
                if observation[field] is not None
            ),
            low_quantile,
        )
        for field in component_fields
    }

    coverage_receipts = [
        observation["coverage"]
        for observation in activated
        if observation["coverage"] is not None
    ]
    coverage_fields = (
        "eligible_source_occurrences",
        "raw_source_occurrences",
        "gist_touched_source_occurrences",
        "gist_fully_represented_source_occurrences",
        "fully_represented_source_occurrences",
        "unrepresented_source_occurrences",
        "gist_touched_but_unrepresented_source_occurrences",
    )
    coverage_totals = {
        field: sum(receipt[field] for receipt in coverage_receipts)
        for field in coverage_fields
    }
    eligible = coverage_totals["eligible_source_occurrences"]
    represented = coverage_totals["fully_represented_source_occurrences"]
    fitted_values = [
        receipt["fitted_encoder_input_tokens"]
        for receipt in coverage_receipts
        if receipt["fitted_encoder_input_tokens"] is not None
    ]
    retained_values = [
        receipt["retained_encoder_input_tokens"]
        for receipt in coverage_receipts
        if receipt["retained_encoder_input_tokens"] is not None
    ]
    fitted_complete = len(fitted_values) == len(coverage_receipts)
    retained_complete = len(retained_values) == len(coverage_receipts)
    fitted_total = sum(fitted_values) if fitted_complete else None
    retained_total = sum(retained_values) if retained_complete else None

    active_values = [
        observation["active_history_bytes"]
        for observation in observations
        if observation["active_history_bytes"] is not None
    ]
    b0_violations = [
        observation
        for observation in observations
        if observation["observed_b0_violation"]
    ]
    activation_counts = Counter(
        observation["activation"] for observation in observations
    )
    configured_ratios = sorted(
        {
            observation["configured_gist_ratio"]
            for observation in observations
            if observation["configured_gist_ratio"] is not None
        }
    )
    guard_statuses = [_guard_failed(receipt) for receipt in guard_receipts]

    return {
        "task_id": task_id,
        "in_fixed_denominator": True,
        "source": {
            "path": source_path,
            "exists": source_exists,
            "decoded_step_records": len(step_records),
            "malformed_records": malformed_records,
            "malformed_line_numbers": list(malformed_line_numbers),
            "controller_receipts": len(observations),
            "controllerless_step_rows": controllerless_rows,
            "empty_generation_trace_rows": empty_trace_rows,
            "invalid_generation_trace_rows": invalid_trace_rows,
            "receipt_error_count": len(receipt_errors),
            "receipt_errors": receipt_errors,
            "complete_jsonl_parse": source_exists and malformed_records == 0,
        },
        "activation": {
            "activated_receipts": activation_counts["activated"],
            "no_history_warmup_receipts": activation_counts["no_history_warmup"],
            "full_history_without_active_footprint_anomalies": activation_counts[
                "anomaly_full_history_without_active_footprint"
            ],
            "active_footprint_without_full_history_anomalies": activation_counts[
                "anomaly_active_footprint_without_full_history"
            ],
            "unknown_history_or_active_footprint_anomalies": activation_counts[
                "anomaly_unknown_history_or_active_footprint"
            ],
        },
        "system_active_history_reduction": {
            **_distribution(history_values, low_quantile),
            "coverage_loss_receipts": sum(
                observation["coverage"] is not None
                and observation["coverage"]["coverage_loss"] is True
                for observation in activated
            ),
            "coverage_unknown_receipts": sum(
                observation["coverage"] is None
                or observation["coverage"]["coverage_loss"] is None
                for observation in activated
            ),
        },
        "system_full_input_reduction": _distribution(
            full_input_values, low_quantile
        ),
        "strict_complete_coverage_subset": {
            "activated_receipts": len(strict),
            "system_active_history_reduction": _distribution(
                strict_history_values, low_quantile
            ),
            "system_full_input_reduction": _distribution(
                strict_full_input_values, low_quantile
            ),
        },
        "component_bytes": component_distributions,
        "active_history": {
            "observed_receipts": len(active_values),
            "observed_peak_bytes": max(active_values, default=None),
            "b0_cap_bytes": b0_cap_bytes,
            "observed_b0_violation_count": len(b0_violations),
            "observed_b0_violation_receipts": [
                {
                    "line_number": observation["line_number"],
                    "trace_index": observation["trace_index"],
                    "decision_key": observation["decision_key"],
                    "active_history_bytes": observation["active_history_bytes"],
                }
                for observation in b0_violations
            ],
        },
        "source_coverage": {
            "observed_activated_receipts": len(coverage_receipts),
            "unknown_activated_receipts": len(activated) - len(coverage_receipts),
            "complete_coverage_receipts": sum(
                receipt["coverage_loss"] is False for receipt in coverage_receipts
            ),
            "coverage_loss_receipts": sum(
                receipt["coverage_loss"] is True for receipt in coverage_receipts
            ),
            "coverage_status_unknown_receipts": sum(
                receipt["coverage_loss"] is None for receipt in coverage_receipts
            ),
            **coverage_totals,
            "fully_represented_source_occurrence_fraction": (
                represented / eligible if eligible else None
            ),
            "fitted_encoder_input_tokens": fitted_total,
            "retained_encoder_input_tokens": retained_total,
            "fitted_encoder_token_retained_fraction": (
                retained_total / fitted_total
                if fitted_total is not None
                and retained_total is not None
                and fitted_total > 0
                else None
            ),
            "token_totals_complete": fitted_complete and retained_complete,
        },
        "configured_gist_ratios_observed": configured_ratios,
        "configured_gist_ratio_is_not_realized_reduction": True,
        "pre_generation_budget_guard": {
            "optional_for_legacy_sources": True,
            "receipts": len(guard_receipts),
            "known_passes": sum(value is False for value in guard_statuses),
            "known_failures": sum(value is True for value in guard_statuses),
            "unknown_status": sum(value is None for value in guard_statuses),
        },
        "observations": observations,
    }


def load_task_steps(
    path: Path,
) -> tuple[list[Mapping[str, Any]], list[int], list[int]]:
    """Load object-valued JSONL rows and preserve physical line numbers."""
    records: list[Mapping[str, Any]] = []
    source_lines: list[int] = []
    malformed_lines: list[int] = []
    if not path.exists():
        return records, source_lines, malformed_lines
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                malformed_lines.append(line_number)
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines.append(line_number)
                continue
            if not isinstance(value, Mapping):
                malformed_lines.append(line_number)
                continue
            records.append(value)
            source_lines.append(line_number)
    return records, source_lines, malformed_lines


def _task_equal_summary(
    task_rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    low_quantile: float,
) -> dict[str, Any]:
    medians = [
        row[field]["median"]
        for row in task_rows
        if row[field]["median"] is not None
    ]
    low_values = [
        row[field]["low_quantile"]
        for row in task_rows
        if row[field]["low_quantile"] is not None
    ]
    selected = len(task_rows)
    contributors = len(medians)
    all_selected = selected > 0 and contributors == selected and len(low_values) == selected
    known_median_mean = statistics.fmean(medians) if medians else None
    known_low_mean = statistics.fmean(low_values) if low_values else None
    return {
        "selected_task_denominator": selected,
        "tasks_contributing": contributors,
        "all_selected_tasks_represented": all_selected,
        "observed_known_task_mean_of_per_task_medians": known_median_mean,
        "observed_known_task_mean_of_per_task_low_quantiles": known_low_mean,
        "all_selected_task_mean_of_per_task_medians": (
            known_median_mean if all_selected else None
        ),
        "all_selected_task_mean_of_per_task_low_quantiles": (
            known_low_mean if all_selected else None
        ),
        "distribution_of_per_task_medians": _distribution(medians, low_quantile),
        "distribution_of_per_task_low_quantiles": _distribution(
            low_values, low_quantile
        ),
    }


def _task_equal_weighted_receipts(
    task_rows: Sequence[Mapping[str, Any]],
    observation_field: str,
    *,
    low_quantile: float,
    complete_coverage_only: bool = False,
) -> dict[str, Any]:
    weighted: list[tuple[float, float]] = []
    contributing_tasks = 0
    receipt_count = 0
    for row in task_rows:
        values = []
        for observation in row["observations"]:
            if observation["activation"] != "activated":
                continue
            if complete_coverage_only:
                coverage = observation["coverage"]
                if (
                    coverage is None
                    or coverage["coverage_loss"] is not False
                    or coverage["valid_set_algebra"] is not True
                ):
                    continue
            value = observation.get(observation_field)
            if _is_finite_number(value):
                values.append(float(value))
        if not values:
            continue
        contributing_tasks += 1
        receipt_count += len(values)
        per_receipt_weight = 1.0 / len(values)
        weighted.extend((value, per_receipt_weight) for value in values)
    all_selected = bool(task_rows) and contributing_tasks == len(task_rows)
    observed_low = weighted_linear_quantile(weighted, low_quantile)
    observed_median = weighted_linear_quantile(weighted, 0.5)
    return {
        "selected_task_denominator": len(task_rows),
        "tasks_contributing": contributing_tasks,
        "activated_receipts_contributing": receipt_count,
        "all_selected_tasks_represented": all_selected,
        "receipt_weight_within_each_task": (
            "1 / contributing activated receipts in that task"
        ),
        "weighted_quantile_interpolation": (
            "linear interpolation on merged weighted-CDF bin midpoints"
        ),
        "observed_known_task_weighted_low_quantile": observed_low,
        "observed_known_task_weighted_median": observed_median,
        "all_selected_task_weighted_low_quantile": (
            observed_low if all_selected else None
        ),
        "all_selected_task_weighted_median": (
            observed_median if all_selected else None
        ),
    }


def _task_equal_scalar(
    task_rows: Sequence[Mapping[str, Any]], field_path: tuple[str, ...]
) -> dict[str, Any]:
    values: list[float] = []
    for row in task_rows:
        value: Any = row
        for field in field_path:
            value = value[field]
        if _is_finite_number(value):
            values.append(float(value))
    all_selected = bool(task_rows) and len(values) == len(task_rows)
    known_mean = statistics.fmean(values) if values else None
    return {
        "selected_task_denominator": len(task_rows),
        "tasks_contributing": len(values),
        "all_selected_tasks_represented": all_selected,
        "observed_known_task_equal_mean": known_mean,
        "all_selected_task_equal_mean": known_mean if all_selected else None,
    }


def _task_equal_component_summary(
    task_rows: Sequence[Mapping[str, Any]],
    component: str,
    *,
    low_quantile: float,
) -> dict[str, Any]:
    projected = [
        {**row, "component_metric": row["component_bytes"][component]}
        for row in task_rows
    ]
    return _task_equal_summary(
        projected, "component_metric", low_quantile=low_quantile
    )


def aggregate_task_rows(
    task_rows: Sequence[Mapping[str, Any]],
    *,
    b0_cap_bytes: int,
    low_quantile: float = DEFAULT_LOW_QUANTILE,
) -> dict[str, Any]:
    """Combine per-task rows without pooling their step-level receipts."""
    if len({row["task_id"] for row in task_rows}) != len(task_rows):
        raise ValueError("task_ids must be unique")
    selected = len(task_rows)
    active_peaks = [
        row["active_history"]["observed_peak_bytes"]
        for row in task_rows
        if row["active_history"]["observed_peak_bytes"] is not None
    ]
    source_fields = (
        "eligible_source_occurrences",
        "raw_source_occurrences",
        "gist_touched_source_occurrences",
        "gist_fully_represented_source_occurrences",
        "fully_represented_source_occurrences",
        "unrepresented_source_occurrences",
        "gist_touched_but_unrepresented_source_occurrences",
    )
    source_totals = {
        field: sum(row["source_coverage"][field] for row in task_rows)
        for field in source_fields
    }
    eligible = source_totals["eligible_source_occurrences"]
    represented = source_totals["fully_represented_source_occurrences"]
    coverage_loss_receipts = sum(
        row["source_coverage"]["coverage_loss_receipts"] for row in task_rows
    )
    coverage_unknown_receipts = sum(
        row["source_coverage"]["unknown_activated_receipts"]
        + row["source_coverage"]["coverage_status_unknown_receipts"]
        for row in task_rows
    )
    history_summary = _task_equal_summary(
        task_rows,
        "system_active_history_reduction",
        low_quantile=low_quantile,
    )
    history_summary["task_equal_weighted_activated_receipts"] = (
        _task_equal_weighted_receipts(
            task_rows,
            "system_active_history_reduction",
            low_quantile=low_quantile,
        )
    )
    full_input_summary = _task_equal_summary(
        task_rows,
        "system_full_input_reduction",
        low_quantile=low_quantile,
    )
    full_input_summary["task_equal_weighted_activated_receipts"] = (
        _task_equal_weighted_receipts(
            task_rows,
            "system_full_input_reduction",
            low_quantile=low_quantile,
        )
    )
    strict_rows = [
        {
            **row,
            "strict_metric": row["strict_complete_coverage_subset"][
                "system_active_history_reduction"
            ],
        }
        for row in task_rows
    ]
    strict_summary = _task_equal_summary(
        strict_rows,
        "strict_metric",
        low_quantile=low_quantile,
    )
    strict_summary["task_equal_weighted_activated_receipts"] = (
        _task_equal_weighted_receipts(
            task_rows,
            "system_active_history_reduction",
            low_quantile=low_quantile,
            complete_coverage_only=True,
        )
    )
    return {
        "schema": METRIC_SCHEMA,
        "fixed_task_denominator": selected,
        "low_quantile_probability": low_quantile,
        "metric_definition": {
            "H": "same-prefix Full-rendered history bytes",
            "A": "actual active gist plus retained raw-history bytes",
            "S": "same-prefix common system plus current live-input bytes",
            "system_active_history_reduction": "H / A for H > 0 and A > 0",
            "system_full_input_reduction": "(S + H) / (S + A)",
            "activation": "H > 0 and A > 0",
            "warmup": "H == 0 and A == 0; excluded from reduction statistics",
            "low_quantile": (
                "bounded linear interpolation at (n - 1) * probability"
            ),
            "task_equal_weighted_receipt_quantile": (
                "each task has total weight one; receipts split that task's "
                "weight equally, and quantiles interpolate weighted-CDF bin midpoints"
            ),
            "task_weighting": (
                "median and low quantile are computed within each task, then "
                "task summaries are averaged with one equal weight per task; "
                "step receipts are never pooled across tasks"
            ),
            "coverage": (
                "primary H/A includes mixed-system selection and eviction; loss "
                "counts are adjacent, and complete-coverage results are separate"
            ),
            "missing_data": (
                "selected tasks remain in the fixed denominator; known-subset "
                "values stay labelled, and all-selected-task values are null "
                "unless every task contributes"
            ),
        },
        "task_equal": {
            "system_active_history_reduction": history_summary,
            "system_full_input_reduction": full_input_summary,
            "strict_complete_coverage_system_active_history_reduction": (
                strict_summary
            ),
            "fully_represented_source_occurrence_fraction": _task_equal_scalar(
                task_rows,
                ("source_coverage", "fully_represented_source_occurrence_fraction"),
            ),
            "component_bytes": {
                component: _task_equal_component_summary(
                    task_rows, component, low_quantile=low_quantile
                )
                for component in (
                    "full_history_bytes",
                    "common_live_bytes",
                    "active_history_bytes",
                    "full_input_bytes",
                    "active_input_bytes",
                )
            },
        },
        "active_history": {
            "b0_cap_bytes": b0_cap_bytes,
            "observed_peak_bytes": max(active_peaks, default=None),
            "tasks_with_observed_peak": len(active_peaks),
            "observed_receipts": sum(
                row["active_history"]["observed_receipts"] for row in task_rows
            ),
            "observed_b0_violation_count": sum(
                row["active_history"]["observed_b0_violation_count"]
                for row in task_rows
            ),
            "tasks_with_observed_b0_violation": sum(
                row["active_history"]["observed_b0_violation_count"] > 0
                for row in task_rows
            ),
            "violations_by_task": {
                row["task_id"]: row["active_history"][
                    "observed_b0_violation_receipts"
                ]
                for row in task_rows
                if row["active_history"]["observed_b0_violation_count"]
            },
        },
        "source_coverage": {
            "activated_receipts": sum(
                row["activation"]["activated_receipts"] for row in task_rows
            ),
            "observed_activated_receipts": sum(
                row["source_coverage"]["observed_activated_receipts"]
                for row in task_rows
            ),
            "coverage_loss_receipts": coverage_loss_receipts,
            "coverage_unknown_receipts": coverage_unknown_receipts,
            "tasks_with_coverage_loss": sum(
                row["source_coverage"]["coverage_loss_receipts"] > 0
                for row in task_rows
            ),
            **source_totals,
            "fully_represented_source_occurrence_fraction": (
                represented / eligible if eligible else None
            ),
            "gist_touched_is_not_counted_as_represented": True,
        },
        "receipt_completeness": {
            "selected_tasks": selected,
            "tasks_with_source_file": sum(row["source"]["exists"] for row in task_rows),
            "tasks_with_activated_measurement": sum(
                row["system_active_history_reduction"]["count"] > 0
                for row in task_rows
            ),
            "tasks_with_no_activated_measurement": [
                row["task_id"]
                for row in task_rows
                if row["system_active_history_reduction"]["count"] == 0
            ],
            "malformed_records": sum(
                row["source"]["malformed_records"] for row in task_rows
            ),
            "controllerless_step_rows": sum(
                row["source"]["controllerless_step_rows"] for row in task_rows
            ),
            "receipt_error_count": sum(
                row["source"]["receipt_error_count"] for row in task_rows
            ),
            "full_history_without_active_footprint_anomalies": sum(
                row["activation"][
                    "full_history_without_active_footprint_anomalies"
                ]
                for row in task_rows
            ),
            "pre_generation_budget_guard_receipts": sum(
                row["pre_generation_budget_guard"]["receipts"] for row in task_rows
            ),
            "pre_generation_budget_guard_known_failures": sum(
                row["pre_generation_budget_guard"]["known_failures"]
                for row in task_rows
            ),
            "legacy_sources_without_guard_are_allowed": True,
        },
        "configured_gist_ratios_observed": sorted(
            {
                ratio
                for row in task_rows
                for ratio in row["configured_gist_ratios_observed"]
            }
        ),
        "configured_gist_ratio_is_not_realized_reduction": True,
        "tasks": list(task_rows),
    }


def analyze_run(
    returned_root: Path,
    task_ids: Sequence[str],
    *,
    b0_cap_bytes: int,
    low_quantile: float = DEFAULT_LOW_QUANTILE,
) -> dict[str, Any]:
    """Analyze selected tasks under ``returned_root/task_shards``."""
    if not task_ids:
        raise ValueError("task_ids must be nonempty to preserve a fixed denominator")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be unique")
    root = Path(returned_root)
    task_rows = []
    for task_id in task_ids:
        path = root / "task_shards" / task_id / "server" / "steps.jsonl"
        records, source_lines, malformed_lines = load_task_steps(path)
        task_rows.append(
            analyze_task_records(
                task_id,
                records,
                b0_cap_bytes=b0_cap_bytes,
                low_quantile=low_quantile,
                malformed_records=len(malformed_lines),
                malformed_line_numbers=malformed_lines,
                source_line_numbers=source_lines,
                source_path=str(path.resolve()),
                source_exists=path.exists(),
            )
        )
    return aggregate_task_rows(
        task_rows, b0_cap_bytes=b0_cap_bytes, low_quantile=low_quantile
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returned-root", type=Path, required=True)
    parser.add_argument(
        "--task-id", action="append", required=True, dest="task_ids"
    )
    parser.add_argument("--b0-cap-bytes", type=int, required=True)
    parser.add_argument(
        "--low-quantile", type=float, default=DEFAULT_LOW_QUANTILE
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = analyze_run(
        args.returned_root,
        args.task_ids,
        b0_cap_bytes=args.b0_cap_bytes,
        low_quantile=args.low_quantile,
    )
    if args.output is not None:
        _write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
