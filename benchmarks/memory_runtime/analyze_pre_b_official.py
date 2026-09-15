"""Summarize validated pre-B task-shard collections without rescoring them.

The input is the JSON emitted by ``collect_official.py``.  This module never
opens shard artifacts, executes benchmark tools, or fills absent official
outcomes.  In particular, an official failure, a validated method-terminal
failure, and an unobserved cell remain three different outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


COLLECTION_SCHEMA = "a-runtime-bfcl-official-collection-v1"
ANALYSIS_SCHEMA = "a-pre-b-paired-official-analysis-v1"
PRE_B_DESIGNS = frozenset({"pre-b-p3", "pre-b-p4a", "pre-b-p4b"})
OFFICIAL_KINDS = frozenset({"official_pass", "official_failure"})
ALLOWED_CELL_STATUSES = frozenset(
    {"official_terminal", "capacity_infeasible", "missing"}
)
COST_FIELDS = (
    "request_count",
    "generation_attempts",
    "extraction_attempts",
    "prompt_tokens",
    "completion_tokens",
    "proxy_wall_seconds",
    "task_wall_seconds",
)
RESOURCE_COST_FIELDS = (
    "resource_scope",
    "controller_wall_seconds",
    "compressed_assembly_wall_seconds",
    "extraction_lookups",
    "extraction_client_cache_hits",
    "extraction_producer_calls",
    "extraction_producer_successes",
    "extraction_producer_failures",
    "extraction_lookup_wall_seconds",
    "extraction_producer_wall_seconds",
    "extraction_scope",
)


class AnalysisInputError(ValueError):
    """The collector JSON does not have a safe, interpretable task matrix."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisInputError(message)


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _need(isinstance(value, dict), f"{path} does not contain a JSON object")
    return value


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _variants(manifest: Mapping[str, Any]) -> list[str]:
    values = manifest.get("expected_variants", manifest.get("variants"))
    _need(
        isinstance(values, list)
        and values
        and all(isinstance(value, str) and value for value in values)
        and len(values) == len(set(values)),
        "collector manifest lacks unique expected_variants",
    )
    return list(values)


def _cost_summary(
    cell: Mapping[str, Any], metric: Mapping[str, Any] | None
) -> dict[str, Any]:
    if metric is None:
        return {field: None for field in (*COST_FIELDS, "total_tokens", "resources")}

    result: dict[str, Any] = {}
    for field in COST_FIELDS:
        value = metric.get(field, cell.get(field))
        _need(
            value is None
            or (isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0),
            f"task metric has invalid {field}",
        )
        result[field] = value
    prompt = result["prompt_tokens"]
    completion = result["completion_tokens"]
    result["total_tokens"] = (
        prompt + completion if prompt is not None and completion is not None else None
    )
    resources = metric.get("generation_resources")
    if resources is None:
        result["resources"] = None
    else:
        _need(isinstance(resources, Mapping), "generation_resources is not an object")
        result["resources"] = {
            field: resources.get(field) for field in RESOURCE_COST_FIELDS
        }
    return result


def _normalize_cell(
    cell: Mapping[str, Any],
    metric: Mapping[str, Any] | None,
    *,
    task_id: str,
    variant: str,
) -> dict[str, Any]:
    status = cell.get("status")
    _need(
        status in ALLOWED_CELL_STATUSES,
        f"{task_id}/{variant} has unsupported collector status {status!r}",
    )
    official_pass = cell.get("official_pass")
    score_known = cell.get("official_score_known")
    valid = cell.get("valid")

    if status == "official_terminal":
        _need(
            valid is True and score_known is True and isinstance(official_pass, bool),
            f"{task_id}/{variant} official terminal is not a validated official outcome",
        )
        _need(isinstance(metric, Mapping), f"{task_id}/{variant} lacks task metrics")
        _need(
            metric.get("official_pass") is official_pass,
            f"{task_id}/{variant} task metric disagrees with its official cell",
        )
        outcome_kind = "official_pass" if official_pass else "official_failure"
    elif status == "capacity_infeasible":
        _need(
            valid is True and score_known is False and official_pass is None,
            f"{task_id}/{variant} method failure fabricates an official outcome",
        )
        _need(
            isinstance(metric, Mapping)
            and metric.get("official_pass") is None
            and metric.get("official_score_known") is False
            and metric.get("operational_status") == "capacity_infeasible",
            f"{task_id}/{variant} lacks validated method-terminal metrics",
        )
        outcome_kind = "method_failure"
    else:
        _need(
            valid is False and score_known is False and official_pass is None,
            f"{task_id}/{variant} missing cell has an invented terminal outcome",
        )
        _need(metric is None, f"{task_id}/{variant} missing cell has task metrics")
        outcome_kind = "missing"

    cost = _cost_summary(cell, metric)
    if outcome_kind != "missing":
        for field in ("generation_attempts", "extraction_attempts"):
            _need(
                isinstance(cost[field], int) and not isinstance(cost[field], bool),
                f"{task_id}/{variant} validated terminal lacks integer {field}",
            )
    return {
        "collector_status": status,
        "outcome_kind": outcome_kind,
        "official_score_known": score_known,
        "official_pass": official_pass,
        "cost": cost,
    }


def _normalize_report(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    _need(report.get("schema") == COLLECTION_SCHEMA, f"{label} has wrong schema")
    _need(report.get("status") in {"valid", "partial"}, f"{label} is not validated")
    _need(report.get("errors") == [], f"{label} contains collector errors")
    manifest = report.get("manifest")
    _need(isinstance(manifest, Mapping), f"{label} lacks collector manifest")
    design = manifest.get("design")
    _need(design in PRE_B_DESIGNS, f"{label} is not a pre-B P3/P4 collection")
    task_ids = manifest.get("task_ids")
    _need(
        isinstance(task_ids, list)
        and task_ids
        and all(isinstance(value, str) and value for value in task_ids)
        and len(task_ids) == len(set(task_ids)),
        f"{label} lacks unique planned task IDs",
    )
    variants = _variants(manifest)
    matrix = report.get("task_matrix")
    _need(isinstance(matrix, list), f"{label} lacks task_matrix")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in matrix:
        _need(isinstance(row, Mapping), f"{label} task_matrix row is not an object")
        task_id = row.get("task_id")
        _need(
            isinstance(task_id, str) and task_id not in rows,
            f"{label} repeats or omits a task ID",
        )
        rows[task_id] = row
    _need(set(rows) == set(task_ids), f"{label} task_matrix differs from planned tasks")

    performance = report.get("performance")
    _need(isinstance(performance, Mapping), f"{label} lacks pre-B performance map")
    normalized: dict[str, dict[str, dict[str, Any]]] = {
        variant: {} for variant in variants
    }
    for task_id in task_ids:
        arms = rows[task_id].get("arms")
        _need(isinstance(arms, Mapping), f"{label} {task_id} lacks arm cells")
        _need(
            set(arms) == set(variants),
            f"{label} {task_id} arm cells differ from expected variants",
        )
        for variant in variants:
            cell = arms[variant]
            _need(isinstance(cell, Mapping), f"{label} {task_id}/{variant} is not an object")
            arm_performance = performance.get(variant)
            _need(
                isinstance(arm_performance, Mapping),
                f"{label} lacks performance entry for {variant}",
            )
            metrics = arm_performance.get("task_metrics")
            _need(isinstance(metrics, Mapping), f"{label} {variant} lacks task_metrics")
            metric = metrics.get(task_id)
            _need(
                metric is None or isinstance(metric, Mapping),
                f"{label} {task_id}/{variant} task metric is invalid",
            )
            normalized[variant][task_id] = _normalize_cell(
                cell, metric, task_id=task_id, variant=variant
            )

    return {
        "design": design,
        "collector_status": report.get("status"),
        "valid_for_method_comparison": report.get("valid_for_method_comparison"),
        "valid_for_pre_b_delivery": report.get("valid_for_pre_b_delivery"),
        "task_ids": list(task_ids),
        "variants": variants,
        "cells": normalized,
        "execution_inputs": manifest.get("execution_inputs"),
    }


def _coverage(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    counts = {
        kind: sum(cell["outcome_kind"] == kind for cell in cells.values())
        for kind in (*sorted(OFFICIAL_KINDS), "method_failure", "missing")
    }
    planned = len(cells)
    scored = counts["official_pass"] + counts["official_failure"]
    unscored = counts["method_failure"] + counts["missing"]
    return {
        "sample_label": "preliminary, n=1",
        "planned_tasks": planned,
        "counts": counts,
        "official_scored_coverage": {
            "numerator": scored,
            "denominator": planned,
            "rate": _ratio(scored, planned),
        },
        "validated_terminal_coverage": {
            "numerator": scored + counts["method_failure"],
            "denominator": planned,
            "rate": _ratio(scored + counts["method_failure"], planned),
        },
        "official_success": {
            "count": counts["official_pass"],
            "completed_subset_denominator": scored,
            "completed_subset_rate": _ratio(counts["official_pass"], scored),
            "all_planned_rate": (
                _ratio(counts["official_pass"], planned) if unscored == 0 else None
            ),
            "all_planned_bounds": (
                [
                    _ratio(counts["official_pass"], planned),
                    _ratio(counts["official_pass"] + unscored, planned),
                ]
                if planned
                else None
            ),
        },
    }


def _breakdown(cells: list[Mapping[str, Any]]) -> dict[str, int]:
    return {
        kind: sum(cell["outcome_kind"] == kind for cell in cells)
        for kind in (*sorted(OFFICIAL_KINDS), "method_failure", "missing")
    }


def _paired_summary(
    task_ids: list[str],
    full_cells: Mapping[str, Mapping[str, Any]],
    method_cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    table = {
        "full_pass_method_pass": 0,
        "full_pass_method_official_failure": 0,
        "full_official_failure_method_pass": 0,
        "full_official_failure_method_official_failure": 0,
    }
    task_outcomes: dict[str, Any] = {}
    paired_official = 0
    pair_missing = 0
    pair_method_failure = 0
    for task_id in task_ids:
        full = full_cells[task_id]
        method = method_cells[task_id]
        full_kind = full["outcome_kind"]
        method_kind = method["outcome_kind"]
        label = None
        if full_kind in OFFICIAL_KINDS and method_kind in OFFICIAL_KINDS:
            paired_official += 1
            full_label = (
                "full_pass"
                if full_kind == "official_pass"
                else "full_official_failure"
            )
            method_label = (
                "method_pass"
                if method_kind == "official_pass"
                else "method_official_failure"
            )
            label = full_label + "_" + method_label
            table[label] += 1
        elif "missing" in {full_kind, method_kind}:
            pair_missing += 1
        else:
            pair_method_failure += 1
        task_outcomes[task_id] = {
            "full": full_kind,
            "method": method_kind,
            "paired_official_outcome": label,
        }

    full_success_ids = [
        task_id for task_id in task_ids
        if full_cells[task_id]["outcome_kind"] == "official_pass"
    ]
    full_failure_ids = [
        task_id for task_id in task_ids
        if full_cells[task_id]["outcome_kind"] == "official_failure"
    ]

    def conditional(ids: list[str], positive: str) -> dict[str, Any]:
        cells = [method_cells[task_id] for task_id in ids]
        breakdown = _breakdown(cells)
        scored = breakdown["official_pass"] + breakdown["official_failure"]
        positive_count = breakdown[positive]
        unknown = breakdown["method_failure"] + breakdown["missing"]
        denominator = len(ids)
        return {
            "eligible_full_tasks": denominator,
            "method_official_scored_tasks": scored,
            "method_scored_coverage": _ratio(scored, denominator),
            "count": positive_count,
            "observed_rate_on_scored_pairs": _ratio(positive_count, scored),
            "complete_rate": (
                _ratio(positive_count, denominator) if unknown == 0 else None
            ),
            "bounds_over_all_eligible_full_tasks": (
                [
                    _ratio(positive_count, denominator),
                    _ratio(positive_count + unknown, denominator),
                ]
                if denominator
                else None
            ),
            "method_outcomes": breakdown,
        }

    return {
        "sample_label": "preliminary, n=1",
        "planned_tasks": len(task_ids),
        "paired_official_coverage": {
            "numerator": paired_official,
            "denominator": len(task_ids),
            "rate": _ratio(paired_official, len(task_ids)),
        },
        "unpaired": {
            "method_terminal_failure_pair": pair_method_failure,
            "missing_pair": pair_missing,
        },
        "paired_official_table": table,
        "full_success_retention": conditional(full_success_ids, "official_pass"),
        "reverse_rescue_on_full_failures": conditional(full_failure_ids, "official_pass"),
        "task_outcomes": task_outcomes,
    }


def _identity_fields(view: Mapping[str, Any]) -> dict[str, Any]:
    execution = view.get("execution_inputs")
    _need(isinstance(execution, Mapping), "collector manifest lacks execution_inputs")
    checkpoint = execution.get("checkpoint")
    data = execution.get("data")
    scorer = execution.get("scorer")
    _need(
        all(isinstance(value, Mapping) for value in (checkpoint, data, scorer)),
        "collector execution_inputs lacks checkpoint/data/scorer identity",
    )
    identity = {
        "checkpoint_profile_fingerprint": checkpoint.get("profile_fingerprint"),
        "data_sha256": data.get("sha256"),
        "scorer_sha256": scorer.get("sha256"),
    }
    _need(
        all(isinstance(value, str) and value for value in identity.values()),
        "collector execution identity fields are incomplete",
    )
    return identity


def analyze(
    collection: Mapping[str, Any],
    *,
    full_reference: Mapping[str, Any] | None = None,
    sources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return paired official outcomes and observed per-task cost accounting."""
    target = _normalize_report(collection, "collection")
    task_ids = target["task_ids"]
    if "full" in target["variants"]:
        _need(full_reference is None, "collection already contains Full; reference is redundant")
        full_cells = target["cells"]["full"]
        full_source = {
            "kind": "same_collection",
            "design": target["design"],
        }
    else:
        _need(
            target["design"] == "pre-b-p4a" and full_reference is not None,
            "a collection without Full requires an explicit P3 Full reference",
        )
        reference = _normalize_report(full_reference, "full reference")
        _need(reference["design"] == "pre-b-p3", "Full reference must be P3")
        _need("full" in reference["variants"], "P3 reference lacks Full")
        _need(reference["task_ids"] == task_ids, "P3 Full reference task IDs differ")
        _need(
            _identity_fields(reference) == _identity_fields(target),
            "P3 Full reference checkpoint, data, or scorer identity differs",
        )
        full_cells = reference["cells"]["full"]
        full_source = {
            "kind": "external_p3_reference",
            "design": reference["design"],
            "collector_status": reference["collector_status"],
        }

    coverage = {
        variant: _coverage(target["cells"][variant])
        for variant in target["variants"]
    }
    methods = [variant for variant in target["variants"] if variant != "full"]
    paired = {
        variant: _paired_summary(task_ids, full_cells, target["cells"][variant])
        for variant in methods
    }
    task_rows = []
    for task_id in task_ids:
        arms = {
            variant: target["cells"][variant][task_id]
            for variant in target["variants"]
        }
        task_rows.append(
            {
                "task_id": task_id,
                "full_baseline": full_cells[task_id],
                "arms": arms,
            }
        )

    all_target_scored = all(
        cell["outcome_kind"] in OFFICIAL_KINDS
        for variant in target["variants"]
        for cell in target["cells"][variant].values()
    ) and all(cell["outcome_kind"] in OFFICIAL_KINDS for cell in full_cells.values())
    return {
        "schema": ANALYSIS_SCHEMA,
        "status": (
            "complete_official_matrix" if all_target_scored else "partial_official_matrix"
        ),
        "sample_label": "preliminary, n=1",
        "design": target["design"],
        "selection_decision": False,
        "official_endpoint": "frozen official whole-task scorer",
        "collector": {
            "status": target["collector_status"],
            "valid_for_method_comparison": target["valid_for_method_comparison"],
            "valid_for_pre_b_delivery": target["valid_for_pre_b_delivery"],
        },
        "task_ids": task_ids,
        "variants": target["variants"],
        "full_baseline_source": full_source,
        "outcome_coverage": coverage,
        "full_baseline_coverage": _coverage(full_cells),
        "paired_with_full": paired,
        "task_rows": task_rows,
        "semantics": {
            "official_failure": "official_terminal with official_pass=false",
            "method_failure": "validated capacity_infeasible; official outcome remains null",
            "missing": "unobserved/incomplete cell; neither official nor method failure",
            "cost": "observed collector task metrics only; missing values remain null",
        },
        **({"inputs": dict(sources)} if sources is not None else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--full-reference", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    collection = _read(args.collection)
    reference = _read(args.full_reference) if args.full_reference else None
    sources: dict[str, Any] = {
        "collection": {
            "path": str(args.collection),
            "sha256": _sha256(args.collection),
        }
    }
    if args.full_reference:
        sources["full_reference"] = {
            "path": str(args.full_reference),
            "sha256": _sha256(args.full_reference),
        }
    result = analyze(collection, full_reference=reference, sources=sources)
    if args.out:
        _atomic_write(args.out, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "design": result["design"],
                    "out": str(args.out),
                }
            )
        )
    else:
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
