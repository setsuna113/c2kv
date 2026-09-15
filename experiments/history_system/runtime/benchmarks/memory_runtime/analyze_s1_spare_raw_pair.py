"""Pair the spare-raw S1 Long20 round with original S1 and preserved S0.

This is an offline reader. It does not run a model, scorer, tool environment,
or remote command. Partial mode keeps all fixed Long20 cells missing as missing;
final mode requires complete official headers, frozen-design bindings, and
complete actual spare-raw runtime receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.memory_runtime import analyze_s1_field_priority_pair as paired
from benchmarks.memory_runtime import analyze_s1_long_stage as single


SCHEMA = "a-s1-spare-raw-paired-analysis-v1"
EXPECTED_TASKS = 20
EXPECTED_POLICY = "requested-full-events-v1"
FROZEN_DESIGN_SCHEMA = "a-structure-long-candidate-frozen-design-v1"


class SpareRawPairedAnalysisError(RuntimeError):
    """Raised when recovered artifacts cannot support this comparison."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SpareRawPairedAnalysisError(
            f"Cannot read JSON {path}: {type(error).__name__}: {error}") from error
    if not isinstance(value, dict):
        raise SpareRawPairedAnalysisError(f"JSON is not an object: {path}")
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


def _validate_analysis(
    value: Mapping[str, Any], label: str, candidate_id: str
) -> tuple[list[str], dict[str, Mapping[str, Any]]]:
    if value.get("schema") != single.SCHEMA:
        raise SpareRawPairedAnalysisError(f"{label} input is not an S1 Long20 analysis")
    identity = value.get("run_identity") or {}
    if identity.get("candidate_id") != candidate_id:
        raise SpareRawPairedAnalysisError(
            f"{label} candidate_id is not {candidate_id}")
    tasks = identity.get("task_ids")
    if (not isinstance(tasks, list) or len(tasks) != EXPECTED_TASKS
            or len(set(tasks)) != EXPECTED_TASKS
            or not all(isinstance(task, str) for task in tasks)):
        raise SpareRawPairedAnalysisError(
            f"{label} analysis lacks the fixed 20 unique tasks")
    raw_rows = value.get("per_task")
    if not isinstance(raw_rows, list) or len(raw_rows) != EXPECTED_TASKS:
        raise SpareRawPairedAnalysisError(f"{label} analysis lacks 20 per-task rows")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        task_id = row.get("task_id") if isinstance(row, Mapping) else None
        if not isinstance(task_id, str) or task_id in rows:
            raise SpareRawPairedAnalysisError(
                f"{label} has malformed or duplicate per-task rows")
        rows[task_id] = row
    if set(rows) != set(tasks):
        raise SpareRawPairedAnalysisError(
            f"{label} task identities disagree within the analysis")
    scored = [task for task in tasks
              if paired._path(rows[task], "quality", "official_score_known") is True]
    correct = sum(
        paired._path(rows[task], "quality", "correct_count") == 1 for task in scored)
    missing = [task for task in tasks if task not in scored]
    quality = value.get("quality") or {}
    if (quality.get("fixed_task_denominator") != EXPECTED_TASKS
            or quality.get("official_scored_tasks") != len(scored)
            or quality.get("official_correct_tasks") != correct
            or quality.get("official_missing_tasks") != len(missing)
            or quality.get("official_missing_task_ids") != missing):
        raise SpareRawPairedAnalysisError(
            f"{label} quality summary disagrees with per-task rows")
    return list(tasks), rows


def _load_side(
    label: str,
    *,
    candidate_id: str,
    analysis_path: Path | None,
    returned_root: Path | None,
    s0_root: Path | None,
    mode: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (analysis_path is None) == (returned_root is None):
        raise SpareRawPairedAnalysisError(
            f"{label} requires exactly one of analysis_path or returned_root")
    if analysis_path is not None:
        analysis = _read_json(analysis_path)
        return analysis, {
            "kind": "existing_analysis",
            "path": str(analysis_path),
            "sha256": _sha256(analysis_path),
        }
    if s0_root is None:
        raise SpareRawPairedAnalysisError("s0_root is required with returned roots")
    assert returned_root is not None
    manifest = returned_root / "stage_manifest.json"
    analysis = single.analyze(
        manifest,
        returned_root / "task_shards",
        s0_root,
        mode=mode,
        sidecar_root=returned_root,
        candidate_id=candidate_id,
    )
    return analysis, {
        "kind": "returned_root_analyzed_in_memory",
        "path": str(returned_root),
        "stage_manifest_sha256": _sha256(manifest),
        "analysis_canonical_sha256": _canonical_hash(analysis),
    }


def _validate_frozen_design(
    path: Path | None,
    analysis: Mapping[str, Any],
    tasks: list[str],
    candidate_id: str,
    label: str,
) -> dict[str, Any] | None:
    if path is None:
        return None
    design = _read_json(path)
    design_sha256 = _sha256(path)
    if (design.get("schema") != FROZEN_DESIGN_SCHEMA
            or design.get("status") != "frozen"
            or design.get("candidate_id") != candidate_id
            or design.get("task_ids") != tasks
            or design.get("run_id")
            != paired._path(analysis, "run_identity", "run_id")
            or paired._path(analysis, "run_identity", "design_sha256")
            != design_sha256):
        raise SpareRawPairedAnalysisError(
            f"{label} frozen design does not bind its analyzed run and fixed task list")
    return {"path": str(path), "sha256": design_sha256}


SPARE_METRICS: dict[str, str] = {
    "known_receipts": "sum",
    "missing_receipts": "sum",
    "requests_with_explicit_policy": "sum",
    "requests_with_input_change": "sum",
    "event_attempts_added": "sum",
    "event_attempts_already_raw": "sum",
    "event_attempts_over_budget": "sum",
    "added_event_receipt_appearances": "sum",
    "added_source_occurrence_appearances": "sum",
    "requests_admitted_over_budget": "sum",
    "added_event_attempts_over_budget": "sum",
    "incremental_raw_tokens_sum": "sum",
    "incremental_bytes_sum": "sum",
    "maximum_incremental_raw_tokens": "maximum",
    "maximum_incremental_bytes": "maximum",
    "maximum_active_history_bytes_before": "maximum",
    "maximum_active_history_bytes_after": "maximum",
}


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
)


def _spare_policy_exposure(
    tasks: list[str], rows: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    totals = Counter()
    policies, versions = set(), set()
    unreported = []
    malformed = []
    for task in tasks:
        receipt = rows[task].get("dependency_packet_spare_raw_exposure")
        if not isinstance(receipt, Mapping):
            unreported.append(task)
            continue
        fields = (
            "known_receipts", "missing_receipts", "requests_with_explicit_policy",
            "requests_admitted_over_budget", "added_event_attempts_over_budget",
        )
        values = [receipt.get(field) for field in fields]
        if not all(type(value) is int and value >= 0 for value in values):
            malformed.append(task)
            continue
        request_count = paired._path(rows[task], "cost", "request_count")
        if (paired._finite(request_count)
                and values[0] + values[1] != request_count):
            malformed.append(task)
            continue
        totals.update(dict(zip(fields, values)))
        raw_policies = receipt.get("policies")
        raw_versions = receipt.get("versions")
        if (not isinstance(raw_policies, list)
                or not all(isinstance(value, str) for value in raw_policies)
                or not isinstance(raw_versions, list)
                or not all(isinstance(value, str) for value in raw_versions)):
            malformed.append(task)
            continue
        policies.update(raw_policies)
        versions.update(raw_versions)
    known = totals["known_receipts"]
    if unreported:
        status = "not_reported_by_source_analysis"
    elif malformed:
        status = "malformed_receipt_summary"
    elif not known:
        status = "no_returned_spare_raw_receipts"
    elif (totals["missing_receipts"] == 0
          and totals["requests_with_explicit_policy"] == known
          and policies == {EXPECTED_POLICY}
          and totals["requests_admitted_over_budget"] == 0
          and totals["added_event_attempts_over_budget"] == 0):
        status = "expected_policy_fully_observed"
    else:
        status = "missing_mixed_or_invalid_policy_receipts"
    return {
        **dict(totals),
        "policies": sorted(policies),
        "versions": sorted(versions),
        "analysis_field_unreported_task_ids": unreported,
        "malformed_task_ids": malformed,
        "status": status,
        "scope": (
            "Policy and admission counts come only from actual per-request "
            "dependency_packet_spare_raw receipts."
        ),
    }


def compare(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    mode: str = "partial",
    old_source: Mapping[str, Any] | None = None,
    new_source: Mapping[str, Any] | None = None,
    old_frozen_design: Path | None = None,
    new_frozen_design: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"partial", "final"}:
        raise SpareRawPairedAnalysisError("mode must be partial or final")
    old_tasks, old_rows = _validate_analysis(old, "old", single.CANDIDATE)
    new_tasks, new_rows = _validate_analysis(new, "new", single.SPARE_RAW_CANDIDATE)
    if old_tasks != new_tasks:
        raise SpareRawPairedAnalysisError(
            "old and new analyses do not use the same ordered fixed Long20 tasks")
    tasks = old_tasks
    old_design = _validate_frozen_design(
        old_frozen_design, old, tasks, single.CANDIDATE, "old")
    new_design = _validate_frozen_design(
        new_frozen_design, new, tasks, single.SPARE_RAW_CANDIDATE, "new")

    pairs = {task: paired._official_pair(old_rows[task], new_rows[task])
             for task in tasks}
    pair_counts = Counter(pairs.values())
    paired_tasks = [task for task in tasks if not pairs[task].startswith("unpaired_")]
    old_correct = sum(
        paired._path(old_rows[task], "quality", "correct_count") == 1
        for task in paired_tasks)
    new_correct = sum(
        paired._path(new_rows[task], "quality", "correct_count") == 1
        for task in paired_tasks)

    costs = {
        name: paired._numeric_pair(
            tasks, old_rows, new_rows,
            lambda row, keys=keys: paired._path(row, *keys), reduction=reduction)
        for name, (keys, reduction) in paired.COST_METRICS.items()
    }
    coverage = {
        name: paired._numeric_pair(
            tasks, old_rows, new_rows,
            lambda row, key=name: paired._path(row, "source_coverage", key))
        for name in paired.COVERAGE_METRICS
    }
    packets = {
        name: paired._numeric_pair(
            tasks, old_rows, new_rows,
            lambda row, key=name: paired._path(
                row, "dependency_packet_exposure", key))
        for name in PACKET_METRICS
    }
    spare = {
        name: paired._numeric_pair(
            tasks, old_rows, new_rows,
            lambda row, key=name: paired._path(
                row, "dependency_packet_spare_raw_exposure", key),
            reduction=reduction,
        )
        for name, reduction in SPARE_METRICS.items()
    }
    new_policy = _spare_policy_exposure(tasks, new_rows)

    if mode == "final":
        reasons = []
        if not paired._final_ready(old):
            reasons.append("old analysis is not a complete valid 20-header recovery")
        if not paired._final_ready(new):
            reasons.append("new analysis is not a complete valid 20-header recovery")
        if old_design is None or new_design is None:
            reasons.append("both old and new frozen-design bindings are required")
        if new_policy["status"] != "expected_policy_fully_observed":
            reasons.append("new request receipts do not fully expose valid spare-raw policy")
        if reasons:
            raise SpareRawPairedAnalysisError(
                "final mode requires complete official headers, bindings, and policy exposure: "
                + "; ".join(reasons))

    per_task = []
    selected_metrics = {
        "request_count": ("cost", "request_count"),
        "extraction_producer_calls": (
            "cost", "resources", "extraction_producer_calls"),
        "maximum_active_history_bytes": (
            "cost", "resources", "active_history_bytes_max"),
        "fully_represented_source_occurrences": (
            "source_coverage", "fully_represented_source_occurrence_appearances"),
        "unrepresented_source_occurrences": (
            "source_coverage", "unrepresented_source_occurrence_appearances"),
    }
    for task in tasks:
        metrics = {}
        for name, path in selected_metrics.items():
            old_value = paired._path(old_rows[task], *path)
            new_value = paired._path(new_rows[task], *path)
            metrics[name] = {
                "old": old_value,
                "new": new_value,
                "new_minus_old": (
                    new_value - old_value
                    if paired._finite(old_value) and paired._finite(new_value) else None),
            }
        per_task.append({
            "task_id": task,
            "official_pair": pairs[task],
            "metrics": metrics,
            "new_spare_raw_exposure": new_rows[task].get(
                "dependency_packet_spare_raw_exposure"),
        })

    return {
        "schema": SCHEMA,
        "status": "complete" if mode == "final" else "partial",
        "mode": mode,
        "sample_label": (
            "preliminary, n=1; paired Long20 development rounds with independently "
            "generated trajectories"
        ),
        "run_identity": {
            "old_candidate_id": single.CANDIDATE,
            "new_candidate_id": single.SPARE_RAW_CANDIDATE,
            "old_run_id": paired._path(old, "run_identity", "run_id"),
            "new_run_id": paired._path(new, "run_identity", "run_id"),
            "task_ids": tasks,
            "fixed_task_denominator": EXPECTED_TASKS,
            "old_analysis_canonical_sha256": _canonical_hash(old),
            "new_analysis_canonical_sha256": _canonical_hash(new),
        },
        "quality": {
            "old": paired._quality_summary(tasks, old_rows),
            "new": paired._quality_summary(tasks, new_rows),
            "paired_official_task_count": len(paired_tasks),
            "unpaired_official_task_count": EXPECTED_TASKS - len(paired_tasks),
            "outcome_counts_over_fixed_20": dict(sorted(pair_counts.items())),
            "new_only_correct_task_ids": [
                task for task in tasks if pairs[task] == "new_only_correct"],
            "old_only_correct_task_ids": [
                task for task in tasks if pairs[task] == "old_only_correct"],
            "old_correct_on_paired": old_correct,
            "new_correct_on_paired": new_correct,
            "observed_accuracy_difference_new_minus_old_on_paired": (
                (new_correct - old_correct) / len(paired_tasks)
                if paired_tasks else None),
        },
        "cost": costs,
        "source_coverage": {
            "metrics": coverage,
            "old_pooled_fully_represented_fraction": paired._ratio_from_rollup(
                coverage, "old_on_paired"),
            "new_pooled_fully_represented_fraction": paired._ratio_from_rollup(
                coverage, "new_on_paired"),
            "scope": (
                "Coverage occurrences use each round's actual request views and therefore "
                "retain differing request-count denominators."
            ),
        },
        "dependency_packet": {"metrics": packets},
        "spare_raw": {
            "expected_policy": EXPECTED_POLICY,
            "new_policy_exposure": new_policy,
            "metrics": spare,
        },
        "paired_with_s0_bounded_latest": {
            "old_round": old.get("paired_with_s0_bounded_latest"),
            "new_round": new.get("paired_with_s0_bounded_latest"),
            "scope": "Preserved per-round S0 comparisons; each trajectory is independently generated.",
        },
        "per_task": per_task,
        "provenance": {
            "old_source": dict(old_source or {}),
            "new_source": dict(new_source or {}),
            "old_frozen_design": old_design,
            "new_frozen_design": new_design,
            "parser_code": {
                "spare_raw_pair": {"path": str(Path(__file__)), "sha256": _sha256(Path(__file__))},
                "field_pair_helpers": {"path": str(Path(paired.__file__)), "sha256": _sha256(Path(paired.__file__))},
                "single_stage": {"path": str(Path(single.__file__)), "sha256": _sha256(Path(single.__file__))},
            },
        },
        "scope": {
            "cpu_only": True,
            "model_requests": 0,
            "remote_requests": 0,
            "scorer_calls": 0,
            "tool_executions": 0,
            "causal_interpretation": (
                "Descriptive task-level pairing only. Same task IDs do not isolate spare-raw: "
                "the two rounds generate independent actions, observations, request counts, and prefixes."
            ),
        },
        "limits": [
            "Partial mode retains missing cells in the fixed denominator and never imputes costs.",
            "Final mode requires 20 official headers on both sides, both frozen designs, and complete actual spare-raw policy receipts.",
            "Added and over-budget counts are request-view appearances, not unique events across a task or round.",
            "No user text, tool values, arguments, or scorer detail is copied.",
        ],
    }


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
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
    parser.add_argument("--mode", choices=("partial", "final"), default="partial")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        old_analysis, old_source = _load_side(
            "old", candidate_id=single.CANDIDATE,
            analysis_path=args.old_analysis, returned_root=args.old_returned_root,
            s0_root=args.s0_root, mode=args.mode)
        new_analysis, new_source = _load_side(
            "new", candidate_id=single.SPARE_RAW_CANDIDATE,
            analysis_path=args.new_analysis, returned_root=args.new_returned_root,
            s0_root=args.s0_root, mode=args.mode)
        result = compare(
            old_analysis, new_analysis, mode=args.mode,
            old_source=old_source, new_source=new_source,
            old_frozen_design=args.old_frozen_design,
            new_frozen_design=args.new_frozen_design)
        _write_new(args.out, result)
        print(json.dumps({
            "status": result["status"],
            "mode": result["mode"],
            "out": str(args.out),
            "paired_official_tasks": result["quality"]["paired_official_task_count"],
            "new_official_missing_tasks": result["quality"]["new"]["official_missing_tasks"],
            "new_spare_raw_policy_exposure": result["spare_raw"]["new_policy_exposure"]["status"],
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, SpareRawPairedAnalysisError,
            single.S1LongAnalysisError, paired.S1PairedAnalysisError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
