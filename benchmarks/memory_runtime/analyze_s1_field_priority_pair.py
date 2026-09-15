"""Pair two recovered S1 Long20 rounds without running models or scorers.

Each side may be an existing analyze_s1_long_stage.py artifact or a returned
stage root containing stage_manifest.json and task_shards/.  Missing official
cells remain in the fixed Long20 denominator in partial mode.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.memory_runtime import analyze_s1_long_stage as single


SCHEMA = "a-s1-field-priority-paired-analysis-v1"
EXPECTED_TASKS = 20
EXPECTED_NEW_POLICY = "top-level-schema-slot-v1"
FROZEN_DESIGN_SCHEMA = "a-structure-long-candidate-frozen-design-v1"


class S1PairedAnalysisError(RuntimeError):
    """Raised when two inputs cannot support the requested comparison."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S1PairedAnalysisError(
            f"Cannot read JSON {path}: {type(error).__name__}: {error}") from error
    if not isinstance(value, dict):
        raise S1PairedAnalysisError(f"JSON is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _path(row: Mapping[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _validate_analysis(value: Mapping[str, Any], label: str) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    if value.get("schema") != single.SCHEMA:
        raise S1PairedAnalysisError(f"{label} input is not an S1 Long20 analysis")
    identity = value.get("run_identity") or {}
    if identity.get("candidate_id") != single.CANDIDATE:
        raise S1PairedAnalysisError(f"{label} candidate_id is not {single.CANDIDATE}")
    tasks = identity.get("task_ids")
    if (
        not isinstance(tasks, list)
        or len(tasks) != EXPECTED_TASKS
        or len(set(tasks)) != EXPECTED_TASKS
        or not all(isinstance(task, str) for task in tasks)
    ):
        raise S1PairedAnalysisError(f"{label} analysis lacks the fixed 20 unique tasks")
    rows = value.get("per_task")
    if not isinstance(rows, list) or len(rows) != EXPECTED_TASKS:
        raise S1PairedAnalysisError(f"{label} analysis lacks 20 per-task rows")
    by_task: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        task_id = row.get("task_id") if isinstance(row, Mapping) else None
        if not isinstance(task_id, str) or task_id in by_task:
            raise S1PairedAnalysisError(f"{label} has malformed or duplicate per-task rows")
        by_task[task_id] = row
    if set(by_task) != set(tasks):
        raise S1PairedAnalysisError(f"{label} task identities disagree within the analysis")
    scored = [row for row in rows if _path(row, "quality", "official_score_known") is True]
    correct = sum(int(_path(row, "quality", "correct_count")) for row in scored)
    missing = [task for task in tasks if _path(by_task[task], "quality", "official_score_known") is not True]
    quality = value.get("quality") or {}
    if (
        quality.get("fixed_task_denominator") != EXPECTED_TASKS
        or quality.get("official_scored_tasks") != len(scored)
        or quality.get("official_correct_tasks") != correct
        or quality.get("official_missing_tasks") != len(missing)
        or quality.get("official_missing_task_ids") != missing
    ):
        raise S1PairedAnalysisError(f"{label} quality summary disagrees with per-task rows")
    return list(tasks), by_task


def _load_side(
    label: str,
    *,
    analysis_path: Path | None,
    returned_root: Path | None,
    s0_root: Path | None,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (analysis_path is None) == (returned_root is None):
        raise S1PairedAnalysisError(
            f"{label} requires exactly one of analysis_path or returned_root")
    if analysis_path is not None:
        analysis = _read_json(analysis_path)
        return analysis, {
            "kind": "existing_analysis",
            "path": str(analysis_path),
            "sha256": _sha256(analysis_path),
        }
    if s0_root is None:
        raise S1PairedAnalysisError("s0_root is required with returned roots")
    assert returned_root is not None
    manifest = returned_root / "stage_manifest.json"
    shards = returned_root / "task_shards"
    analysis = single.analyze(
        manifest, shards, s0_root, mode=mode, sidecar_root=returned_root)
    return analysis, {
        "kind": "returned_root_analyzed_in_memory",
        "path": str(returned_root),
        "stage_manifest_sha256": _sha256(manifest),
        "analysis_canonical_sha256": _canonical_hash(analysis),
    }


def _validate_frozen_design(
    path: Path | None,
    analysis: Mapping[str, Any],
    task_ids: list[str],
    label: str,
) -> dict[str, Any] | None:
    if path is None:
        return None
    design = _read_json(path)
    if (
        design.get("schema") != FROZEN_DESIGN_SCHEMA
        or design.get("status") != "frozen"
        or design.get("candidate_id") != single.CANDIDATE
        or design.get("task_ids") != task_ids
        or design.get("run_id") != _path(analysis, "run_identity", "run_id")
    ):
        raise S1PairedAnalysisError(
            f"{label} frozen design does not bind the analyzed run and fixed task list")
    return {"path": str(path), "sha256": _sha256(path)}


def _official_pair(old: Mapping[str, Any], new: Mapping[str, Any]) -> str:
    old_known = _path(old, "quality", "official_score_known") is True
    new_known = _path(new, "quality", "official_score_known") is True
    if not old_known and not new_known:
        return "unpaired_missing_both"
    if not old_known:
        return "unpaired_missing_old"
    if not new_known:
        return "unpaired_missing_new"
    old_correct = _path(old, "quality", "correct_count") == 1
    new_correct = _path(new, "quality", "correct_count") == 1
    if old_correct and new_correct:
        return "both_correct"
    if new_correct:
        return "new_only_correct"
    if old_correct:
        return "old_only_correct"
    return "both_incorrect"


def _numeric_pair(
    tasks: list[str],
    old_rows: Mapping[str, Mapping[str, Any]],
    new_rows: Mapping[str, Mapping[str, Any]],
    getter: Callable[[Mapping[str, Any]], Any],
    *,
    reduction: str = "sum",
) -> dict[str, Any]:
    if reduction not in {"sum", "maximum"}:
        raise ValueError("unsupported reduction")
    old_values = {task: getter(old_rows[task]) for task in tasks}
    new_values = {task: getter(new_rows[task]) for task in tasks}
    old_known = {task: value for task, value in old_values.items() if _finite(value)}
    new_known = {task: value for task, value in new_values.items() if _finite(value)}
    paired_tasks = [task for task in tasks if task in old_known and task in new_known]

    def reduce(values: list[int | float]) -> int | float | None:
        if not values:
            return None
        return sum(values) if reduction == "sum" else max(values)

    old_all = reduce(list(old_known.values()))
    new_all = reduce(list(new_known.values()))
    old_paired = reduce([old_known[task] for task in paired_tasks])
    new_paired = reduce([new_known[task] for task in paired_tasks])
    return {
        "reduction": reduction,
        "old_known_task_count": len(old_known),
        "new_known_task_count": len(new_known),
        "paired_task_count": len(paired_tasks),
        "old_known_value": old_all,
        "new_known_value": new_all,
        "old_on_paired": old_paired,
        "new_on_paired": new_paired,
        "new_minus_old_on_paired": (
            new_paired - old_paired
            if _finite(old_paired) and _finite(new_paired) else None
        ),
        "strict_fixed_20_old": old_all if len(old_known) == EXPECTED_TASKS else None,
        "strict_fixed_20_new": new_all if len(new_known) == EXPECTED_TASKS else None,
        "scope": (
            "Paired value uses only tasks with numeric receipts on both independently "
            "generated trajectories; strict fixed-20 values are null unless all tasks are known."
        ),
    }


COST_METRICS: dict[str, tuple[tuple[str, ...], str]] = {
    "request_count": (("cost", "request_count"), "sum"),
    "ok_request_count": (("cost", "ok_request_count"), "sum"),
    "error_request_count": (("cost", "error_request_count"), "sum"),
    "task_wall_seconds": (("cost", "task_wall_seconds"), "sum"),
    "proxy_request_wall_seconds": (("cost", "proxy_request_wall_seconds"), "sum"),
    "generation_attempts": (("cost", "generation", "generation_attempts"), "sum"),
    "known_generation_prompt_tokens": (("cost", "generation", "known_prompt_tokens"), "sum"),
    "known_generation_completion_tokens": (("cost", "generation", "known_completion_tokens"), "sum"),
    "extraction_lookups": (("cost", "resources", "extraction_lookups"), "sum"),
    "extraction_client_cache_hits": (("cost", "resources", "extraction_client_cache_hits"), "sum"),
    "extraction_producer_calls": (("cost", "resources", "extraction_producer_calls"), "sum"),
    "extraction_producer_successes": (("cost", "resources", "extraction_producer_successes"), "sum"),
    "extraction_producer_failures": (("cost", "resources", "extraction_producer_failures"), "sum"),
    "extraction_lookup_wall_seconds": (("cost", "resources", "extraction_lookup_wall_seconds"), "sum"),
    "extraction_producer_wall_seconds": (("cost", "resources", "extraction_producer_wall_seconds"), "sum"),
    "controller_wall_seconds": (("cost", "resources", "controller_wall_seconds"), "sum"),
    "compressed_assembly_wall_seconds": (("cost", "resources", "compressed_assembly_wall_seconds"), "sum"),
    "gist_tokens_sum": (("cost", "resources", "gist_tokens_sum"), "sum"),
    "maximum_active_history_bytes": (("cost", "resources", "active_history_bytes_max"), "maximum"),
    "maximum_evidence_bytes": (("cost", "resources", "evidence_bytes_max"), "maximum"),
}


COVERAGE_METRICS = (
    "known_receipts",
    "missing_receipts",
    "eligible_source_occurrence_appearances",
    "fully_represented_source_occurrence_appearances",
    "unrepresented_source_occurrence_appearances",
    "gist_fully_represented_source_occurrence_appearances",
    "raw_source_occurrence_appearances",
    "fitted_fragment_appearances",
    "retained_fragment_appearances",
    "fitted_encoder_input_tokens",
    "retained_encoder_input_tokens",
)


PACKET_METRICS = (
    "known_receipts",
    "missing_receipts",
    "requests_with_nonempty_packet",
    "requests_without_nonempty_packet",
    "retained_fact_appearances",
    "eligible_fact_appearances",
    "field_limit_omission_appearances",
    "unrepresented_requested_source_appearances",
    "workspace_drop_appearances",
    "requests_with_explicit_field_priority_policy",
    "requests_with_legacy_implicit_field_priority",
)


def _policy_exposure(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    policies = set()
    unreported = []
    unbalanced = []
    for task_id, row in rows.items():
        packet = row.get("dependency_packet_exposure") or {}
        if not all(key in packet for key in (
            "field_priority_policies",
            "requests_with_explicit_field_priority_policy",
            "requests_with_legacy_implicit_field_priority",
        )):
            unreported.append(task_id)
            continue
        known = packet.get("known_receipts")
        missing = packet.get("missing_receipts")
        explicit = packet.get("requests_with_explicit_field_priority_policy")
        implicit = packet.get("requests_with_legacy_implicit_field_priority")
        if not all(type(value) is int and value >= 0 for value in (known, missing, explicit, implicit)):
            raise S1PairedAnalysisError("field-priority exposure counters are malformed")
        if explicit + implicit != known:
            unbalanced.append(task_id)
        totals.update({
            "known_packet_receipts": known,
            "missing_packet_receipts": missing,
            "explicit_policy_receipts": explicit,
            "legacy_implicit_policy_receipts": implicit,
        })
        raw_policies = packet.get("field_priority_policies")
        if not isinstance(raw_policies, list) or not all(isinstance(value, str) for value in raw_policies):
            raise S1PairedAnalysisError("field-priority policy list is malformed")
        policies.update(raw_policies)
    return {
        **dict(totals),
        "observed_explicit_policies": sorted(policies),
        "analysis_field_unreported_task_ids": unreported,
        "receipt_balance_error_task_ids": unbalanced,
        "scope": (
            "Policy names and counts come from actual dependency_packet receipts in recorded "
            "requests. Missing fields in an older analysis artifact remain unverified."
        ),
    }


def _policy_status(exposure: Mapping[str, Any], expected: str | None) -> str:
    if exposure.get("analysis_field_unreported_task_ids"):
        return "not_reported_by_source_analysis"
    if exposure.get("receipt_balance_error_task_ids"):
        return "malformed_receipt_balance"
    known = exposure.get("known_packet_receipts", 0)
    if not known:
        return "no_returned_packet_receipts"
    if expected is None:
        return (
            "legacy_implicit_observed"
            if exposure.get("legacy_implicit_policy_receipts") == known
            and not exposure.get("observed_explicit_policies")
            else "explicit_or_mixed_policy_observed"
        )
    if (
        exposure.get("missing_packet_receipts") == 0
        and exposure.get("explicit_policy_receipts") == known
        and exposure.get("legacy_implicit_policy_receipts") == 0
        and exposure.get("observed_explicit_policies") == [expected]
    ):
        return "expected_policy_observed"
    return "mixed_missing_or_unexpected_policy"


def _quality_summary(
    tasks: list[str], rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    scored = [task for task in tasks if _path(rows[task], "quality", "official_score_known") is True]
    correct = [task for task in scored if _path(rows[task], "quality", "correct_count") == 1]
    missing = [task for task in tasks if task not in scored]
    return {
        "official_scored_tasks": len(scored),
        "official_missing_tasks": len(missing),
        "official_missing_task_ids": missing,
        "official_correct_tasks": len(correct),
        "successes_over_fixed_20": {
            "numerator": len(correct),
            "denominator": EXPECTED_TASKS,
            "rate": len(correct) / EXPECTED_TASKS,
            "interpretation": (
                "final accuracy" if len(scored) == EXPECTED_TASKS
                else "partial lower bound; unfinished cells are not estimated"
            ),
        },
        "accuracy_over_scored_only": len(correct) / len(scored) if scored else None,
    }


def _final_ready(value: Mapping[str, Any]) -> bool:
    return (
        value.get("status") == "complete"
        and _path(value, "quality", "official_scored_tasks") == EXPECTED_TASKS
        and _path(value, "quality", "official_missing_tasks") == 0
        and _path(value, "recovery", "missing_task_count") == 0
        and not (_path(value, "recovery", "completion_errors") or [])
    )


def compare(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    mode: str = "partial",
    expected_new_policy: str | None = EXPECTED_NEW_POLICY,
    allow_self_comparison_fixture: bool = False,
    old_source: Mapping[str, Any] | None = None,
    new_source: Mapping[str, Any] | None = None,
    old_frozen_design: Path | None = None,
    new_frozen_design: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"partial", "final"}:
        raise S1PairedAnalysisError("mode must be partial or final")
    old_tasks, old_rows = _validate_analysis(old, "old")
    new_tasks, new_rows = _validate_analysis(new, "new")
    if old_tasks != new_tasks:
        raise S1PairedAnalysisError("old and new analyses do not use the same ordered fixed Long20 tasks")
    tasks = old_tasks
    old_run = _path(old, "run_identity", "run_id")
    new_run = _path(new, "run_identity", "run_id")
    same_run = old_run == new_run
    if same_run and not allow_self_comparison_fixture:
        raise S1PairedAnalysisError(
            "old and new run_id are equal; use the explicit self-comparison fixture option")
    old_design_binding = _validate_frozen_design(old_frozen_design, old, tasks, "old")
    new_design_binding = _validate_frozen_design(new_frozen_design, new, tasks, "new")

    pair_by_task = {task: _official_pair(old_rows[task], new_rows[task]) for task in tasks}
    pair_counts = Counter(pair_by_task.values())
    paired_tasks = [task for task in tasks if not pair_by_task[task].startswith("unpaired_")]
    old_correct_paired = sum(
        _path(old_rows[task], "quality", "correct_count") == 1 for task in paired_tasks)
    new_correct_paired = sum(
        _path(new_rows[task], "quality", "correct_count") == 1 for task in paired_tasks)

    cost = {
        name: _numeric_pair(
            tasks, old_rows, new_rows,
            lambda row, keys=keys: _path(row, *keys), reduction=reduction)
        for name, (keys, reduction) in COST_METRICS.items()
    }
    coverage = {
        name: _numeric_pair(
            tasks, old_rows, new_rows,
            lambda row, key=name: _path(row, "source_coverage", key))
        for name in COVERAGE_METRICS
    }
    packet = {
        name: _numeric_pair(
            tasks, old_rows, new_rows,
            lambda row, key=name: _path(row, "dependency_packet_exposure", key))
        for name in PACKET_METRICS
    }
    old_policy = _policy_exposure(old_rows)
    new_policy = _policy_exposure(new_rows)
    old_policy["status"] = _policy_status(old_policy, None)
    new_policy["status"] = _policy_status(new_policy, expected_new_policy)

    if mode == "final":
        reasons = []
        if not _final_ready(old):
            reasons.append("old analysis is not a complete valid 20-header recovery")
        if not _final_ready(new):
            reasons.append("new analysis is not a complete valid 20-header recovery")
        if expected_new_policy is not None and new_policy["status"] != "expected_policy_observed":
            reasons.append("new request receipts do not fully expose the expected field-priority policy")
        if reasons:
            raise S1PairedAnalysisError(
                "final mode requires complete official headers and policy exposure: "
                + "; ".join(reasons))

    per_task = []
    per_task_metrics = {
        "request_count": ("cost", "request_count"),
        "extraction_producer_calls": ("cost", "resources", "extraction_producer_calls"),
        "maximum_active_history_bytes": ("cost", "resources", "active_history_bytes_max"),
        "eligible_source_occurrences": ("source_coverage", "eligible_source_occurrence_appearances"),
        "fully_represented_source_occurrences": ("source_coverage", "fully_represented_source_occurrence_appearances"),
        "unrepresented_source_occurrences": ("source_coverage", "unrepresented_source_occurrence_appearances"),
        "retained_fact_appearances": ("dependency_packet_exposure", "retained_fact_appearances"),
        "workspace_drop_appearances": ("dependency_packet_exposure", "workspace_drop_appearances"),
    }
    for task in tasks:
        metrics = {}
        for name, keys in per_task_metrics.items():
            old_value = _path(old_rows[task], *keys)
            new_value = _path(new_rows[task], *keys)
            metrics[name] = {
                "old": old_value,
                "new": new_value,
                "new_minus_old": (
                    new_value - old_value
                    if _finite(old_value) and _finite(new_value) else None),
            }
        per_task.append({
            "task_id": task,
            "official_pair": pair_by_task[task],
            "old_official_known": _path(old_rows[task], "quality", "official_score_known") is True,
            "new_official_known": _path(new_rows[task], "quality", "official_score_known") is True,
            "metrics": metrics,
        })

    result = {
        "schema": SCHEMA,
        "status": "complete" if mode == "final" else "partial",
        "mode": mode,
        "sample_label": (
            "preliminary, n=1; paired Long20 development rounds with independently generated trajectories"
        ),
        "run_identity": {
            "candidate_id": single.CANDIDATE,
            "old_run_id": old_run,
            "new_run_id": new_run,
            "comparison_kind": "self_comparison_fixture" if same_run else "independent_rounds",
            "task_ids": tasks,
            "fixed_task_denominator": EXPECTED_TASKS,
            "old_analysis_canonical_sha256": _canonical_hash(old),
            "new_analysis_canonical_sha256": _canonical_hash(new),
        },
        "quality": {
            "old": _quality_summary(tasks, old_rows),
            "new": _quality_summary(tasks, new_rows),
            "paired_official_task_count": len(paired_tasks),
            "unpaired_official_task_count": EXPECTED_TASKS - len(paired_tasks),
            "outcome_counts_over_fixed_20": dict(sorted(pair_counts.items())),
            "new_only_correct_task_ids": [task for task in tasks if pair_by_task[task] == "new_only_correct"],
            "old_only_correct_task_ids": [task for task in tasks if pair_by_task[task] == "old_only_correct"],
            "both_correct_task_ids": [task for task in tasks if pair_by_task[task] == "both_correct"],
            "both_incorrect_task_ids": [task for task in tasks if pair_by_task[task] == "both_incorrect"],
            "old_correct_on_paired": old_correct_paired,
            "new_correct_on_paired": new_correct_paired,
            "observed_accuracy_difference_new_minus_old_on_paired": (
                (new_correct_paired - old_correct_paired) / len(paired_tasks)
                if paired_tasks else None),
        },
        "cost": cost,
        "source_coverage": {
            "metrics": coverage,
            "old_pooled_fully_represented_fraction": (
                _path(old, "per_task") and _ratio_from_rollup(coverage, "old_on_paired")
            ),
            "new_pooled_fully_represented_fraction": (
                _path(new, "per_task") and _ratio_from_rollup(coverage, "new_on_paired")
            ),
            "scope": (
                "Coverage occurrences are request-view appearances on each independently generated "
                "trajectory; differing request counts and prefixes remain explicit."
            ),
        },
        "dependency_packet": {
            "metrics": packet,
            "old_policy_exposure": old_policy,
            "new_policy_exposure": new_policy,
            "expected_new_policy": expected_new_policy,
        },
        "paired_with_s0_bounded_latest": {
            "old_round": old.get("paired_with_s0_bounded_latest"),
            "new_round": new.get("paired_with_s0_bounded_latest"),
            "scope": (
                "The existing S0 comparisons are preserved per round. They share task IDs but "
                "also come from independently generated trajectories."
            ),
        },
        "per_task": per_task,
        "provenance": {
            "old_source": dict(old_source or {}),
            "new_source": dict(new_source or {}),
            "old_frozen_design": old_design_binding,
            "new_frozen_design": new_design_binding,
            "parser_code": {
                "paired_analyzer": {"path": str(Path(__file__)), "sha256": _sha256(Path(__file__))},
                "single_stage_analyzer": {
                    "path": str(Path(single.__file__)), "sha256": _sha256(Path(single.__file__))},
            },
        },
        "scope": {
            "cpu_only": True,
            "model_requests": 0,
            "remote_requests": 0,
            "scorer_calls": 0,
            "tool_executions": 0,
            "fixed_task_denominator": EXPECTED_TASKS,
            "causal_interpretation": (
                "Descriptive task-level pairing only. Same task IDs do not isolate the field-priority "
                "intervention: each round generates its own actions, observations, request count, and "
                "prefixes, and a prefix can change even where packet field ordering has no effect."
            ),
        },
        "limits": [
            "Partial mode retains all 20 tasks and does not impute missing official headers or receipts.",
            "Final mode requires both complete recovered analyses, all 20 new official headers, and full expected-policy exposure in recorded packet receipts.",
            "New-only and old-only tasks are descriptive round outcomes, not field-priority causal effects.",
            "No raw messages, packet values, tool arguments, or scorer detail are copied into this analysis.",
        ],
    }
    return result


def _ratio_from_rollup(coverage: Mapping[str, Any], field: str) -> float | None:
    represented = _path(coverage, "fully_represented_source_occurrence_appearances", field)
    eligible = _path(coverage, "eligible_source_occurrence_appearances", field)
    return represented / eligible if _finite(represented) and _finite(eligible) and eligible else None


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    old = parser.add_mutually_exclusive_group(required=True)
    old.add_argument("--old-analysis", type=Path)
    old.add_argument("--old-returned-root", type=Path)
    new = parser.add_mutually_exclusive_group(required=True)
    new.add_argument("--new-analysis", type=Path)
    new.add_argument("--new-returned-root", type=Path)
    parser.add_argument("--s0-root", type=Path)
    parser.add_argument("--old-frozen-design", type=Path)
    parser.add_argument("--new-frozen-design", type=Path)
    parser.add_argument("--expected-new-policy", default=EXPECTED_NEW_POLICY)
    parser.add_argument("--skip-expected-new-policy", action="store_true")
    parser.add_argument("--allow-self-comparison-fixture", action="store_true")
    parser.add_argument("--mode", choices=("partial", "final"), default="partial")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.skip_expected_new_policy and not args.allow_self_comparison_fixture:
        parser.error("--skip-expected-new-policy is limited to explicit self-comparison fixtures")
    try:
        old_analysis, old_source = _load_side(
            "old", analysis_path=args.old_analysis,
            returned_root=args.old_returned_root, s0_root=args.s0_root,
            mode=args.mode)
        new_analysis, new_source = _load_side(
            "new", analysis_path=args.new_analysis,
            returned_root=args.new_returned_root, s0_root=args.s0_root,
            mode=args.mode)
        result = compare(
            old_analysis, new_analysis,
            mode=args.mode,
            expected_new_policy=(
                None if args.skip_expected_new_policy else args.expected_new_policy),
            allow_self_comparison_fixture=args.allow_self_comparison_fixture,
            old_source=old_source, new_source=new_source,
            old_frozen_design=args.old_frozen_design,
            new_frozen_design=args.new_frozen_design,
        )
        _write_new(args.out, result)
        print(json.dumps({
            "status": result["status"],
            "mode": result["mode"],
            "out": str(args.out),
            "paired_official_tasks": result["quality"]["paired_official_task_count"],
            "new_official_missing_tasks": result["quality"]["new"]["official_missing_tasks"],
            "new_policy_exposure": result["dependency_packet"]["new_policy_exposure"]["status"],
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, S1PairedAnalysisError, single.S1LongAnalysisError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
