"""Normalize frozen native and multi-benchmark results for delivery reports.

The collector is deliberately fail closed.  It publishes an official score only
after the complete fixed denominator is terminal and all local evidence needed
for that score has been hash checked.  Running, missing, and infrastructure
failed cells retain their observed counts but never receive a synthetic zero or
an incomplete-subset score.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping, Sequence


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SAMPLE_LABEL = "preliminary, n=1"
INDEX_SCHEMA = "a-delivery-report-results-v1"
NATIVE_STAGE_SCHEMA = "a-history-system-run-v1"
NATIVE_ANALYSIS_SCHEMA = "c2kv-history-system-result-collection-v1"
SUITE_STAGE_SCHEMA = "a-history-multibench-stage-v1"
SUITE_SCHEMA = "a-history-multibench-frozen-suite-v1"
SUITE_TASK_MANIFEST_SCHEMA = "a-history-multibench-task-manifest-v1"
SUITE_TASK_RESULT_SCHEMA = "history-system-official-result-v1"
R001_TASK_MANIFEST_SCHEMA = "a-history-system-task-manifest-v1"
PEER_SOURCE_SCHEMA = "a-history-system-r001-peer-sources-v1"
BENCHMARKS = ("bfcl", "tau2", "toolsandbox", "acebench", "acon_appworld")
SUITE_BENCHMARKS = BENCHMARKS[1:]


def _read_object(path: Path, what: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {what} at {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{what} at {path} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(path: Path, kind: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"required {kind} artifact is missing: {path}")
    return {
        "kind": kind,
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _resolve_bound_path(value: Any, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("bound artifact path must be a nonempty string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _verified_bound_artifact(
    value: Any, *, repo_root: Path, kind: str, fallback: Path | None = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{kind} binding must be an object")
    path = _resolve_bound_path(value.get("path"), repo_root)
    if not path.is_file() and fallback is not None:
        path = fallback.resolve()
    actual = _artifact(path, kind)
    expected_hash, expected_bytes = value.get("sha256"), value.get("bytes")
    if expected_hash != actual["sha256"]:
        raise ValueError(f"{kind} SHA-256 mismatch: {path}")
    if expected_bytes != actual["bytes"]:
        raise ValueError(f"{kind} byte-count mismatch: {path}")
    return actual


def _finite_fraction(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise ValueError(f"{field} must be a finite fraction in [0, 1]")
    return float(value)


def _nonnegative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _deduplicate_artifacts(values: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_identity: dict[tuple[str, str], dict[str, Any]] = {}
    for value in values:
        key = (str(value.get("path")), str(value.get("sha256")))
        by_identity[key] = dict(value)
    return sorted(by_identity.values(), key=lambda row: (row["kind"], row["path"]))


def _native_paths(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.name == "returned":
        return root / "stage_manifest.json", root.parent / "analysis.json"
    returned = root / "returned" / "stage_manifest.json"
    return (returned if returned.is_file() else root / "stage_manifest.json", root / "analysis.json")


def _pending_native_cell(root: Path) -> dict[str, Any]:
    sources: list[dict[str, Any]] = []
    candidate, planned, cohort, checkpoint = root.name, 0, "frozen native output unavailable", None
    freeze_path = root / "freeze.json"
    if freeze_path.is_file():
        freeze = _read_object(freeze_path, "native freeze receipt")
        if isinstance(freeze.get("candidate_id"), str) and freeze["candidate_id"]:
            candidate = freeze["candidate_id"]
        sources.append(_artifact(freeze_path, "native_freeze"))
    tasks_path = root / "submitted" / "tasks.json"
    if tasks_path.is_file():
        tasks = _read_object(tasks_path, "native task manifest")
        task_ids = tasks.get("task_ids")
        if (
            tasks.get("schema") != R001_TASK_MANIFEST_SCHEMA
            or not isinstance(task_ids, list)
            or tasks.get("fixed_denominator") != len(task_ids)
        ):
            raise ValueError("pending native output has an invalid frozen task manifest")
        planned = len(task_ids)
        cohort = f"{tasks.get('stage', 'frozen')} / {planned} fixed BFCL tasks"
        sources.append(_artifact(tasks_path, "native_task_manifest"))
    design_path = root / "submitted" / "design.json"
    if design_path.is_file():
        design = _read_object(design_path, "native frozen design")
        checkpoint = design.get("checkpoint_selection")
        if isinstance(checkpoint, Mapping):
            checkpoint = dict(checkpoint)
            if checkpoint.get("ratio") is None and design.get("ratio") is not None:
                checkpoint["ratio"] = design["ratio"]
        sources.append(_artifact(design_path, "native_design"))
    return {
        "benchmark": "bfcl",
        "method": candidate,
        "cohort": cohort,
        "status": "pending",
        "source_status": "returned_stage_missing",
        "official_score": None,
        "n_scored": 0,
        "n_planned": planned,
        "sample_label": SAMPLE_LABEL,
        "checkpoint": checkpoint,
        "sources": _deduplicate_artifacts(sources),
    }


def _source_status(value: Any) -> str:
    return value if isinstance(value, str) and value else "missing"


def _native_checkpoint(stage: Mapping[str, Any]) -> Any:
    binding = stage.get("checkpoint_selection")
    if not isinstance(binding, Mapping):
        return binding
    result = dict(binding)
    if result.get("ratio") is None and stage.get("ratio") is not None:
        result["ratio"] = stage["ratio"]
    return result


def _status_from_native(stage: Mapping[str, Any] | None) -> str:
    if stage is None:
        return "pending"
    status = _source_status(stage.get("status"))
    state = _source_status(stage.get("state"))
    if status == "completed_fixed_manifest" and state == "completed":
        return "completed"
    lowered = f"{status} {state}".lower()
    if any(word in lowered for word in ("infra", "failed", "exhausted", "interrupted")):
        return "infra_failed"
    return "pending"


def _native_compression(
    analysis: Mapping[str, Any], analysis_source: Mapping[str, Any], root: Path, planned: int
) -> dict[str, Any] | None:
    compression = analysis.get("compression")
    if not isinstance(compression, Mapping) or compression.get("status") != "available":
        return None
    metrics = compression.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("fixed_task_denominator") != planned:
        raise ValueError("native compression denominator differs from the completed task denominator")
    try:
        weighted = metrics["task_equal"]["system_active_history_reduction"][
            "task_equal_weighted_activated_receipts"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("native compression is missing the task-equal weighted receipt summary") from error
    ratio = weighted.get("all_selected_task_weighted_median")
    if (
        ratio is not None
        and (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not math.isfinite(float(ratio))
            or ratio <= 0
        )
    ):
        raise ValueError("native compression ratio must be finite and positive")
    if ratio is not None and weighted.get("all_selected_tasks_represented") is not True:
        raise ValueError("native compression claims a full value without all selected tasks")

    trace_sources: list[dict[str, Any]] = []
    candidate_root = root.parent if root.name == "returned" else root
    sources = analysis.get("sources")
    for row in sources.get("steps", []) if isinstance(sources, Mapping) else []:
        if not isinstance(row, Mapping) or row.get("status") != "present":
            continue
        task_id = row.get("task_id")
        fallback = (
            candidate_root / "returned" / "task_shards" / str(task_id) / "server" / "steps.jsonl"
            if isinstance(task_id, str)
            else None
        )
        trace_sources.append(
            _verified_bound_artifact(
                row.get("source"), repo_root=REPO, kind="native_steps", fallback=fallback
            )
        )
    if ratio is not None and len(trace_sources) != planned:
        raise ValueError("native compression does not bind one local steps trace per planned task")
    strict = metrics.get("task_equal", {}).get(
        "strict_complete_coverage_system_active_history_reduction", {}).get(
        "task_equal_weighted_activated_receipts", {})
    strict_ratio = strict.get("all_selected_task_weighted_median")
    if strict_ratio is not None and strict.get("all_selected_tasks_represented") is not True:
        raise ValueError("Complete-coverage summary is missing selected tasks")
    return {
        "source_coverage": metrics.get("source_coverage"),
        "complete_coverage_history_reduction": strict_ratio,
        "status": "available" if ratio is not None else "unavailable",
        "metric": "system_active_history_reduction",
        "aggregation": "task_equal_weighted_activated_receipt_median",
        "full_bytes_over_resident_bytes": float(ratio) if ratio is not None else None,
        "selected_task_denominator": planned,
        "tasks_contributing": weighted.get("tasks_contributing"),
        "sources": _deduplicate_artifacts([analysis_source, *trace_sources]),
    }


def _validate_completed_native(
    stage: Mapping[str, Any], analysis: Mapping[str, Any]
) -> tuple[int, float]:
    if stage.get("schema") != NATIVE_STAGE_SCHEMA:
        raise ValueError("unexpected native stage schema")
    planned = _nonnegative_int(stage.get("whole_task_denominator"), "native denominator")
    task_ids = stage.get("task_ids")
    outcomes = stage.get("task_outcomes")
    if not isinstance(task_ids, list) or len(task_ids) != planned or len(set(task_ids)) != planned:
        raise ValueError("native stage task_ids do not preserve the fixed denominator")
    if not isinstance(outcomes, list) or len(outcomes) != planned:
        raise ValueError("native stage task outcomes do not preserve the fixed denominator")
    if stage.get("completed_task_cells") != planned or stage.get("denominator_observed") != planned:
        raise ValueError("native completed stage did not observe its full fixed denominator")
    for row in outcomes:
        if not isinstance(row, Mapping) or not (
            row.get("outcome") == "official_completed"
            and row.get("in_fixed_denominator") is True
            and row.get("worker_returncode") == 0
            and row.get("server_returncode") == 0
        ):
            raise ValueError("native completed stage contains a non-official task outcome")

    if analysis.get("schema") != NATIVE_ANALYSIS_SCHEMA:
        raise ValueError("unexpected native analysis schema")
    if analysis.get("candidate_id") != stage.get("candidate_id"):
        raise ValueError("native analysis candidate differs from its stage")
    if analysis.get("sample_label") != SAMPLE_LABEL:
        raise ValueError("native analysis must retain preliminary, n=1")
    identity = analysis.get("manifest_identity")
    if not isinstance(identity, Mapping) or identity.get("task_ids") != task_ids:
        raise ValueError("native analysis task identity differs from its stage")
    if identity.get("fixed_denominator") != planned:
        raise ValueError("native analysis denominator differs from its stage")
    quality = analysis.get("quality")
    overall = quality.get("overall") if isinstance(quality, Mapping) else None
    if not isinstance(overall, Mapping) or overall.get("sample_label") != SAMPLE_LABEL:
        raise ValueError("native analysis is missing the preliminary overall result")
    successes = _nonnegative_int(overall.get("algorithm_successes"), "native successes")
    failures = _nonnegative_int(overall.get("algorithm_failures"), "native failures")
    unknown = _nonnegative_int(overall.get("unknown"), "native unknown count")
    scored = _nonnegative_int(overall.get("scored_tasks"), "native scored count")
    if (successes + failures + unknown, scored, unknown) != (planned, planned, 0):
        raise ValueError("native completed analysis is not fully official-scored")
    score = _finite_fraction(
        overall.get("official_accuracy_over_fixed_denominator"), "native official score"
    )
    if not math.isclose(score, successes / planned, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("native official score differs from successes/fixed denominator")
    return planned, score


def _native_long10_cell(
    stage: Mapping[str, Any],
    analysis: Mapping[str, Any],
    sources: Sequence[Mapping[str, Any]],
    root: Path,
) -> dict[str, Any] | None:
    identity = analysis.get("manifest_identity")
    task_ids = identity.get("task_ids") if isinstance(identity, Mapping) else None
    if identity.get("manifest_id") != "r001_mixed20" or not isinstance(task_ids, list):
        return None
    long_ids = [task_id for task_id in task_ids if str(task_id).startswith("multi_turn_long_context_")]
    base_ids = [task_id for task_id in task_ids if str(task_id).startswith("multi_turn_base_")]
    if len(long_ids) != 10 or len(base_ids) != 10 or len(task_ids) != 20:
        return None
    candidate_root = root.parent if root.name == "returned" else root
    manifest_source = _verified_bound_artifact(
        identity.get("manifest_source"),
        repo_root=REPO,
        kind="r001_task_manifest",
        fallback=candidate_root / "submitted" / "tasks.json",
    )
    manifest = _read_object(Path(manifest_source["path"]), "r001 task manifest")
    if (
        manifest.get("schema") != R001_TASK_MANIFEST_SCHEMA
        or manifest.get("manifest_id") != "r001_mixed20"
        or manifest.get("task_ids") != task_ids
        or manifest.get("fixed_denominator") != 20
    ):
        raise ValueError("native long10 slice is not bound to the exact r001 mixed20 manifest")
    quality = analysis.get("quality")
    long_quality = quality.get("long") if isinstance(quality, Mapping) else None
    if not isinstance(long_quality, Mapping) or long_quality.get("sample_label") != SAMPLE_LABEL:
        raise ValueError("r001 native analysis is missing its long10 quality slice")
    successes = _nonnegative_int(long_quality.get("algorithm_successes"), "long10 successes")
    failures = _nonnegative_int(long_quality.get("algorithm_failures"), "long10 failures")
    unknown = _nonnegative_int(long_quality.get("unknown"), "long10 unknown")
    scored = _nonnegative_int(long_quality.get("scored_tasks"), "long10 scored")
    if long_quality.get("fixed_denominator") != 10 or (successes + failures, unknown, scored) != (10, 0, 10):
        raise ValueError("r001 long10 slice is not completely official-scored")
    score = _finite_fraction(
        long_quality.get("official_accuracy_over_fixed_denominator"), "long10 official score"
    )
    if not math.isclose(score, successes / 10, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("r001 long10 score differs from successes/fixed denominator")
    return {
        "benchmark": "bfcl",
        "method": str(stage.get("candidate_id")),
        "cohort": "r001 long10 (variant=long)",
        "status": "completed",
        "official_score": score,
        "n_scored": 10,
        "n_planned": 10,
        "sample_label": SAMPLE_LABEL,
        "comparison_role": "active_native_long10_slice",
        "checkpoint": _native_checkpoint(stage),
        "sources": _deduplicate_artifacts([*sources, manifest_source]),
    }


def collect_native_output(root: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    root = root.resolve()
    stage_path, analysis_path = _native_paths(root)
    if not stage_path.is_file():
        return _pending_native_cell(root.parent if root.name == "returned" else root), None

    stage = _read_object(stage_path, "native stage")
    if stage.get("schema") != NATIVE_STAGE_SCHEMA:
        raise ValueError(f"unexpected native stage schema at {stage_path}")
    stage_source = _artifact(stage_path, "native_stage")
    planned = _nonnegative_int(stage.get("whole_task_denominator"), "native denominator")
    outcomes = stage.get("task_outcomes")
    scored = sum(
        isinstance(row, Mapping) and row.get("outcome") == "official_completed"
        for row in outcomes if isinstance(outcomes, list)
    )
    status = _status_from_native(stage)
    cell = {
        "benchmark": "bfcl",
        "method": str(stage.get("candidate_id") or root.name),
        "cohort": f"{stage.get('evaluation_stage', 'frozen')} / {planned} fixed BFCL tasks",
        "status": status,
        "source_status": _source_status(stage.get("status")),
        "official_score": None,
        "n_scored": scored,
        "n_planned": planned,
        "sample_label": SAMPLE_LABEL,
        "checkpoint": _native_checkpoint(stage),
        "sources": [stage_source],
    }
    comparison = None
    if status == "completed":
        if not analysis_path.is_file():
            raise ValueError("completed native stage is missing analysis.json")
        analysis = _read_object(analysis_path, "native analysis")
        verified_planned, score = _validate_completed_native(stage, analysis)
        analysis_source = _artifact(analysis_path, "native_analysis")
        cell.update(
            official_score=score,
            n_scored=verified_planned,
            sources=_deduplicate_artifacts([stage_source, analysis_source]),
            compression=_native_compression(analysis, analysis_source, root, verified_planned),
        )
        comparison = _native_long10_cell(stage, analysis, cell["sources"], root)
    return cell, comparison


def _suite_paths(root: Path) -> tuple[Path | None, Path | None, Path | None]:
    root = root.resolve()
    stage_candidates = (root / "returned" / "stage.json", root / "results" / "stage.json", root / "stage.json")
    suite_candidates = (root / "submitted" / "suite.json", root / "suite.json", root / "package" / "suite.json")
    task_candidates = (root / "submitted" / "tasks.json", root / "tasks.json", root / "package" / "tasks.json")
    return (
        next((path for path in stage_candidates if path.is_file()), None),
        next((path for path in suite_candidates if path.is_file()), None),
        next((path for path in task_candidates if path.is_file()), None),
    )


def _suite_manifest_rows(path: Path) -> list[dict[str, Any]]:
    manifest = _read_object(path, "suite task manifest")
    tasks = manifest.get("tasks")
    if manifest.get("schema") != SUITE_TASK_MANIFEST_SCHEMA or not isinstance(tasks, list):
        raise ValueError("unexpected suite task manifest schema")
    if manifest.get("fixed_denominator") != len(tasks):
        raise ValueError("suite task manifest denominator differs from its rows")
    result = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping) or task.get("benchmark") not in SUITE_BENCHMARKS:
            raise ValueError(f"suite task manifest row {index} has an unsupported benchmark")
        if not isinstance(task.get("task_id"), str) or not task["task_id"]:
            raise ValueError(f"suite task manifest row {index} has no task_id")
        result.append(dict(task))
    return result


def _pending_suite_pieces(root: Path, suite_path: Path | None, tasks_path: Path | None) -> list[dict[str, Any]]:
    candidate = root.name
    sources: list[dict[str, Any]] = []
    if suite_path is not None:
        suite = _read_object(suite_path, "frozen suite")
        if suite.get("schema") != SUITE_SCHEMA:
            raise ValueError("unexpected frozen suite schema")
        candidate = str(suite.get("candidate_id") or candidate)
        sources.append(_artifact(suite_path, "frozen_suite"))
        if tasks_path is None:
            relative = suite.get("task_manifest")
            if isinstance(relative, str):
                candidate_path = (suite_path.parent / relative).resolve()
                tasks_path = candidate_path if candidate_path.is_file() else None
    rows = _suite_manifest_rows(tasks_path) if tasks_path is not None else []
    if tasks_path is not None:
        sources.append(_artifact(tasks_path, "suite_task_manifest"))
    benchmarks = sorted({row["benchmark"] for row in rows}, key=BENCHMARKS.index) or list(SUITE_BENCHMARKS)
    return [
        {
            "candidate_id": candidate,
            "benchmark": benchmark,
            "status": "pending",
            "source_status": "missing",
            "terminal": False,
            "scores": [],
            "task_ids": [row["task_id"] for row in rows if row["benchmark"] == benchmark],
            "n_scored": 0,
            "n_planned": sum(row["benchmark"] == benchmark for row in rows),
            "checkpoint": None,
            "sources": list(sources),
            "compression_tasks": [],
        }
        for benchmark in benchmarks
    ]


def _task_equal_weighted_median(task_values: Sequence[Sequence[float]]) -> float | None:
    weighted: list[tuple[float, float]] = []
    for values in task_values:
        if not values:
            return None
        weight = 1.0 / len(values)
        weighted.extend((float(value), weight) for value in values)
    if not weighted:
        return None
    combined: dict[float, float] = defaultdict(float)
    for value, weight in weighted:
        combined[value] += weight
    ordered = sorted(combined.items())
    total = sum(weight for _, weight in ordered)
    positions: list[float] = []
    cumulative = 0.0
    for _, weight in ordered:
        positions.append((cumulative + 0.5 * weight) / total)
        cumulative += weight
    probability = 0.5
    if probability <= positions[0]:
        return ordered[0][0]
    if probability >= positions[-1]:
        return ordered[-1][0]
    for index in range(1, len(ordered)):
        if probability <= positions[index]:
            fraction = (probability - positions[index - 1]) / (
                positions[index] - positions[index - 1]
            )
            return ordered[index - 1][0] * (1.0 - fraction) + ordered[index][0] * fraction
    raise AssertionError("weighted median interpolation did not terminate")


def _ace_recovered_score(root: Path, result_root: Path, row: Mapping[str, Any]):
    proof = root / "returned_scoring" / row["task_key"] / "result.json"
    if row.get("benchmark") != "acebench" or not proof.exists():
        return None
    recovered = _read_object(proof, "ACE scorer recovery")
    if not (recovered.get("task_id") == row["task_id"]
            and recovered.get("task_key") == row["task_key"]
            and recovered.get("status") == "official_scored_from_existing_generations"
            and recovered.get("model_calls") == 0
            and recovered.get("generation_trace_unchanged") is True):
        raise ValueError("ACE scorer recovery contract mismatch")
    sources = [_artifact(proof, "ace_scorer_recovery")]
    score_rows = result_rows = None
    has_trace = has_original = False
    for binding in recovered["artifacts"]:
        remote = binding["path"]
        if "/results/" in remote:
            base = result_root
            relative = remote.split("/results/", 1)[1]
        elif "/scoring_recovery/" in remote:
            base = root / "returned_scoring"
            relative = remote.split("/scoring_recovery/", 1)[1]
        else:
            raise ValueError("ACE recovery artifact outside known result roots")
        local = (base / relative).resolve()
        if base.resolve() not in local.parents:
            raise ValueError("ACE recovery artifact escaped its result root")
        source = _artifact(local, "ace_recovered_official_artifact")
        if source["sha256"] != binding["sha256"]:
            raise ValueError("ACE recovery artifact hash mismatch")
        sources.append(source)
        if local.name.endswith("_score.json"):
            score_rows = [json.loads(line) for line in local.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif local.name.endswith("_result.json"):
            result_rows = [json.loads(line) for line in local.read_text(encoding="utf-8").splitlines() if line.strip()]
        has_trace |= local.name == "steps.jsonl"
        has_original |= local.name == "result.json" and local.parent.name == "official"
    if not score_rows or not result_rows or len(result_rows) != 1 or not has_trace or not has_original:
        raise ValueError("ACE recovery requires the original generation and one official scored task")
    if str(result_rows[0]["id"]) != row["task_id"]:
        raise ValueError("ACE recovery task ID differs from official generation")
    failed = any(x.get("id") == 0 or str(x.get("id")) == row["task_id"] for x in score_rows[1:])
    score = 0.0 if failed else 1.0
    if _finite_fraction(recovered.get("official_score"), "ACE recovered score") != score:
        raise ValueError("ACE recovered score differs from official failure rows")
    return score, sources


def _suite_piece_rows(root: Path, stage_path: Path) -> list[dict[str, Any]]:
    stage = _read_object(stage_path, "multi-benchmark stage")
    if stage.get("schema") != SUITE_STAGE_SCHEMA:
        raise ValueError(f"unexpected multi-benchmark stage schema at {stage_path}")
    stage_source = _artifact(stage_path, "suite_stage")
    outcomes = stage.get("task_outcomes")
    planned = _nonnegative_int(stage.get("fixed_denominator"), "suite denominator")
    if not isinstance(outcomes, list) or len(outcomes) != planned:
        raise ValueError("suite stage outcomes do not preserve the fixed denominator")
    task_order = stage.get("task_order")
    if not isinstance(task_order, list) or len(task_order) != planned:
        raise ValueError("suite stage task order does not preserve the fixed denominator")
    terminal = stage.get("status") in {
        "completed_fixed_manifest", "completed_fixed_manifest_with_infra_failures",
        "stage_wall_exhausted", "interrupted_without_rerun",
    }
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, raw in enumerate(outcomes):
        if not isinstance(raw, Mapping):
            raise ValueError(f"suite stage outcome {index} is not an object")
        row = dict(raw)
        if row.get("benchmark") not in SUITE_BENCHMARKS:
            raise ValueError(f"suite stage outcome {index} has an unsupported benchmark")
        if row.get("task_key") != task_order[index] or not isinstance(row.get("task_id"), str):
            raise ValueError(f"suite stage outcome {index} differs from the frozen order")
        by_benchmark[row["benchmark"]].append(row)

    parent_ids = None
    selection_path = root / "delivery_task_selection.json"
    if selection_path.exists():
        selection = _read_object(selection_path, "suite continuation selection")
        parent_ids = selection["parent_task_ids"]
        if parent_ids != [row["task_id"] for row in outcomes]:
            raise ValueError("continuation parent differs from original fixed denominator")
        selected = set(selection["selected_task_ids"])
        companion = _verified_bound_artifact(selection["continuation_task_manifest"], repo_root=REPO, kind="continuation_task_manifest")
        remaining = _read_object(Path(companion["path"]), "continuation tasks")["tasks"]
        if selected & {x["task_id"] for x in remaining} or selected | {x["task_id"] for x in remaining} != set(parent_ids):
            raise ValueError("continuation must partition the original denominator without overlap")
        for row in outcomes:
            if row["task_id"] not in selected and row["outcome"] not in {"not_started_in_denominator", "not_started_stage_wall_in_denominator"}:
                raise ValueError("continuation cannot omit a previously started task")
        by_benchmark = {key: [row for row in rows if row["task_id"] in selected] for key, rows in by_benchmark.items()}
    pieces = []
    result_root = stage_path.parent
    for benchmark, rows in by_benchmark.items():
        scores: list[float] = []
        task_ids: list[str] = []
        sources: list[dict[str, Any]] = [stage_source]
        if selection_path.exists():
            sources.extend([_artifact(selection_path, "continuation_selection"), companion])
        compression_tasks: list[dict[str, Any]] = []
        failures = 0
        for row in rows:
            task_ids.append(row["task_id"])
            recovered = _ace_recovered_score(root, result_root, row)
            if recovered is not None:
                score, recovered_sources = recovered
                scores.append(score)
                sources.extend(recovered_sources)
            elif row.get("outcome") == "official_scored":
                result_path = result_root / "task_shards" / row["task_key"] / "official" / "result.json"
                result = _read_object(result_path, "official task result")
                result_source = _artifact(result_path, "official_task_result")
                score = _finite_fraction(result.get("official_score"), "suite official task score")
                if not (
                    result.get("schema") == SUITE_TASK_RESULT_SCHEMA
                    and result.get("status") == "completed"
                    and result.get("scored") is True
                    and result.get("benchmark") == benchmark
                    and result.get("task_id") == row["task_id"]
                    and isinstance(result.get("benchmark_summary"), Mapping)
                    and isinstance(result.get("source_binding"), list)
                    and result.get("error") is None
                    and row.get("scored") is True
                    and isinstance(row.get("official_result"), Mapping)
                ):
                    raise ValueError(f"official result contract mismatch for {row['task_key']}")
                row_score = _finite_fraction(row.get("official_score"), "suite stage task score")
                if not math.isclose(score, row_score, rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"official result score mismatch for {row['task_key']}")
                bound = row["official_result"].get("artifact")
                if not isinstance(bound, Mapping) or bound.get("sha256") != result_source["sha256"]:
                    raise ValueError(f"official result hash mismatch for {row['task_key']}")
                if bound.get("bytes") != result_source["bytes"]:
                    raise ValueError(f"official result byte-count mismatch for {row['task_key']}")
                scores.append(score)
                sources.append(result_source)
            elif row.get("outcome") == "infra_failed_in_denominator":
                failures += 1
            elif terminal and row.get("outcome") not in {
                "not_started_stage_wall_in_denominator", "not_started_in_denominator"
            }:
                failures += 1

            evidence = row.get("server_evidence")
            values: list[float] = []
            extra_generations = None
            if isinstance(evidence, Mapping):
                receipts = evidence.get("compression_receipts")
                if isinstance(receipts, list) and evidence.get("compression_receipt_errors") == 0:
                    for receipt in receipts:
                        value = receipt.get("system_active_history_reduction") if isinstance(receipt, Mapping) else None
                        if value is None:
                            continue
                        if (
                            isinstance(value, bool)
                            or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))
                            or value <= 0
                        ):
                            raise ValueError(f"invalid compression receipt for {row['task_key']}")
                        if receipt.get("errors") not in ([], None):
                            raise ValueError(f"errored compression receipt for {row['task_key']}")
                        values.append(float(value))
                generation_count, step_count = evidence.get("generation_count"), evidence.get("step_count")
                if (
                    isinstance(generation_count, int) and not isinstance(generation_count, bool)
                    and isinstance(step_count, int) and not isinstance(step_count, bool)
                    and generation_count >= step_count >= 0
                ):
                    extra_generations = generation_count - step_count
                steps_binding = evidence.get("steps")
                if isinstance(steps_binding, Mapping):
                    steps_path = result_root / "task_shards" / row["task_key"] / "server" / "steps.jsonl"
                    steps_source = _verified_bound_artifact(
                        steps_binding, repo_root=REPO, kind="suite_steps", fallback=steps_path
                    )
                    sources.append(steps_source)
            compression_tasks.append(
                {"task_id": row["task_id"], "values": values, "extra_generations": extra_generations}
            )

        status = "pending"
        if terminal:
            status = "infra_failed" if failures or len(scores) != len(rows) else "completed"
        pieces.append({
            "continuation_parent_task_ids": parent_ids,
            "candidate_id": str(stage.get("candidate_id") or root.name),
            "benchmark": benchmark,
            "status": status,
            "source_status": _source_status(stage.get("status")),
            "terminal": terminal,
            "scores": scores,
            "task_ids": task_ids,
            "n_scored": len(scores),
            "n_infra_failed": failures,
            "n_planned": len(rows),
            "checkpoint": stage.get("checkpoint"),
            "sources": _deduplicate_artifacts(sources),
            "compression_tasks": compression_tasks,
        })
    return pieces


def collect_suite_output(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    if not root.exists():
        return _pending_suite_pieces(root, None, None)
    stage_path, suite_path, tasks_path = _suite_paths(root)
    if stage_path is None:
        return _pending_suite_pieces(root, suite_path, tasks_path)
    return _suite_piece_rows(root, stage_path)


def _merge_suite_pieces(pieces: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for piece in pieces:
        grouped[(str(piece["candidate_id"]), str(piece["benchmark"]))].append(piece)
    cells = []
    for (candidate, benchmark), rows in grouped.items():
        task_ids = [task_id for row in rows for task_id in row["task_ids"]]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(f"duplicate {benchmark} task IDs across explicit suite outputs")
        for row in rows:
            parent = row.get("continuation_parent_task_ids")
            if parent is not None and not set(parent).issubset(task_ids):
                raise ValueError("all continuation pieces are required to preserve the parent denominator")
        checkpoints = {
            json.dumps(row.get("checkpoint"), sort_keys=True, separators=(",", ":"))
            for row in rows if row.get("checkpoint") is not None
        }
        if len(checkpoints) > 1:
            raise ValueError(f"checkpoint identity differs across {candidate}/{benchmark} outputs")
        n_planned = sum(int(row["n_planned"]) for row in rows)
        n_scored = sum(int(row["n_scored"]) for row in rows)
        completed = all(row["status"] == "completed" for row in rows) and n_scored == n_planned
        failed = any(row["status"] == "infra_failed" for row in rows)
        scores = [float(score) for row in rows for score in row["scores"]]
        official_score = statistics.fmean(scores) if completed and scores else None
        compression_tasks = [task for row in rows for task in row["compression_tasks"]]
        ratio = (
            _task_equal_weighted_median([task["values"] for task in compression_tasks])
            if completed and len(compression_tasks) == n_planned
            else None
        )
        extra_values = [task["extra_generations"] for task in compression_tasks]
        extra_generations = (
            sum(extra_values)
            if completed and len(extra_values) == n_planned and all(value is not None for value in extra_values)
            else None
        )
        sources = _deduplicate_artifacts(source for row in rows for source in row["sources"])
        cell = {
            "benchmark": benchmark,
            "method": candidate,
            "cohort": f"fixed {n_planned}-task denominator",
            "status": "completed" if completed else ("infra_failed" if failed else "pending"),
            "source_statuses": sorted({_source_status(row.get("source_status")) for row in rows}),
            "official_score": official_score,
            "score_aggregation": "mean_over_fixed_task_denominator" if official_score is not None else None,
            "n_scored": n_scored,
            "n_infra_failed": sum(int(row.get("n_infra_failed", 0)) for row in rows),
            "n_unscored": n_planned - n_scored,
            "all_pieces_terminal": all(bool(row.get("terminal")) for row in rows),
            "n_planned": n_planned,
            "sample_label": SAMPLE_LABEL,
            "checkpoint": rows[0].get("checkpoint"),
            "sources": sources,
        }
        if any(task["values"] for task in compression_tasks):
            cell["compression"] = {
                "status": "available" if ratio is not None else "partial_not_reported",
                "metric": "system_active_history_reduction",
                "aggregation": "task_equal_weighted_activated_receipt_median",
                "full_bytes_over_resident_bytes": ratio,
                "selected_task_denominator": n_planned,
                "tasks_contributing": sum(bool(task["values"]) for task in compression_tasks),
                "sources": [source for source in sources if source["kind"] in {"suite_stage", "suite_steps"}],
            }
            if extra_generations is not None:
                cell["compression"]["extra_generations"] = extra_generations
        cells.append(cell)
    return sorted(cells, key=lambda row: (BENCHMARKS.index(row["benchmark"]), row["method"]))


def collect_peer_long10(path: Path, *, repo_root: Path = REPO) -> list[dict[str, Any]]:
    path = path.resolve()
    peers = _read_object(path, "peer source index")
    if peers.get("schema") != PEER_SOURCE_SCHEMA or peers.get("sample_label") != SAMPLE_LABEL:
        raise ValueError("unexpected peer source index contract")
    peer_source = _artifact(path, "peer_source_index")
    manifest_source = _verified_bound_artifact(
        peers.get("r001_manifest"), repo_root=repo_root, kind="r001_task_manifest"
    )
    manifest = _read_object(Path(manifest_source["path"]), "r001 task manifest")
    task_ids = manifest.get("task_ids")
    if (
        manifest.get("schema") != R001_TASK_MANIFEST_SCHEMA
        or manifest.get("manifest_id") != "r001_mixed20"
        or not isinstance(task_ids, list)
        or manifest.get("fixed_denominator") != 20
    ):
        raise ValueError("peer source index does not bind the exact r001 mixed20 manifest")
    long_ids = [task_id for task_id in task_ids if str(task_id).startswith("multi_turn_long_context_")]
    if len(long_ids) != 10:
        raise ValueError("r001 manifest does not contain the expected long10 slice")
    required_identity = peers.get("required_comparison_identity")
    checkpoint = required_identity.get("checkpoint") if isinstance(required_identity, Mapping) else None
    if not isinstance(checkpoint, Mapping):
        raise ValueError("peer source index is missing its checkpoint identity")
    methods = peers.get("reuse_audit", {}).get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("peer source index is missing audited methods")
    labels = {"raw": "Raw", "text": "Text", "full": "Full", "hiagent": "HiAgent"}
    cells = []
    for method in labels:
        record = methods.get(method)
        reuse = record.get("long_reuse") if isinstance(record, Mapping) else None
        rows = reuse.get("cells") if isinstance(reuse, Mapping) else None
        if not (
            isinstance(rows, list)
            and reuse.get("status") == "complete_for_selected_r001_long10"
            and reuse.get("sample_label") == SAMPLE_LABEL
            and reuse.get("fixed_denominator") == 10
            and reuse.get("validated_official_cells") == 10
            and reuse.get("unknown") == 0
        ):
            raise ValueError(f"peer method {method} is not a completed audited long10 result")
        if [row.get("task_id") for row in rows if isinstance(row, Mapping)] != long_ids:
            raise ValueError(f"peer method {method} task IDs differ from r001 long10")
        successes = 0
        sources: list[dict[str, Any]] = [peer_source, manifest_source]
        for row in rows:
            if not isinstance(row, Mapping) or row.get("sample_label") != SAMPLE_LABEL:
                raise ValueError(f"peer method {method} has an invalid row")
            correct = row.get("correct")
            expected_outcome = "algorithm_success" if correct is True else "algorithm_failure"
            if not isinstance(correct, bool) or row.get("outcome") != expected_outcome:
                raise ValueError(f"peer method {method} has inconsistent outcome evidence")
            if row.get("reuse_status") != "reused_validated_official_result":
                raise ValueError(f"peer method {method} row was not validated for reuse")
            successes += int(correct)
            sources.append(
                _verified_bound_artifact(
                    row.get("official_summary"), repo_root=repo_root, kind="peer_official_summary"
                )
            )
            headers = row.get("official_score_headers")
            if not isinstance(headers, list) or not headers:
                raise ValueError(f"peer method {method} has no official score header")
            sources.extend(
                _verified_bound_artifact(header, repo_root=repo_root, kind="peer_official_score")
                for header in headers
            )
        if reuse.get("algorithm_successes") != successes or reuse.get("algorithm_failures") != 10 - successes:
            raise ValueError(f"peer method {method} aggregate differs from its official rows")
        cells.append({
            "benchmark": "bfcl",
            "method": f"{labels[method]} (B500)",
            "cohort": "r001 long10 (variant=long)",
            "status": "completed",
            "official_score": successes / 10,
            "n_scored": 10,
            "n_planned": 10,
            "sample_label": SAMPLE_LABEL,
            "comparison_role": "audited_peer_long10",
            "checkpoint": {
                "selected_arm": checkpoint.get("training_arm"),
                "selected_step": checkpoint.get("parameter_version"),
                "config_sha256": checkpoint.get("config_sha256"),
                "trainer_state_sha256": checkpoint.get("trainer_state_sha256"),
                "ratio": None,
            },
            "sources": _deduplicate_artifacts(sources),
        })
    return cells


def collect_delivery(
    *,
    native_outputs: Sequence[Path] = (),
    suite_outputs: Sequence[Path] = (),
    algorithm_name: str,
    execution_note: str,
    peer_sources: Path | None = None,
    repo_root: Path = REPO,
) -> dict[str, Any]:
    if not algorithm_name.strip() or not execution_note.strip():
        raise ValueError("algorithm name and execution note must be nonempty")
    native_cells: list[dict[str, Any]] = []
    comparison_cells: list[dict[str, Any]] = []
    for root in native_outputs:
        cell, comparison = collect_native_output(root)
        native_cells.append(cell)
        if comparison is not None:
            comparison_cells.append(comparison)
    suite_pieces = [piece for root in suite_outputs for piece in collect_suite_output(root)]
    if peer_sources is not None:
        comparison_cells.extend(collect_peer_long10(peer_sources, repo_root=repo_root))
    cells = sorted(
        [*native_cells, *_merge_suite_pieces(suite_pieces)],
        key=lambda row: (BENCHMARKS.index(row["benchmark"]), row["method"]),
    )
    return {
        "schema": INDEX_SCHEMA,
        "sample_label": SAMPLE_LABEL,
        "algorithm": {"name": algorithm_name, "execution_note": execution_note},
        "cells": cells,
        "comparison_cells": sorted(comparison_cells, key=lambda row: row["method"]),
        "collection_contract": {
            "native_authority": "returned/stage_manifest.json plus analysis.json",
            "suite_authority": "returned stage.json plus local official/result.json artifacts",
            "terminal_only_official_scores": True,
            "missing_and_infra_failed_scores": None,
            "partial_scored_counts_are_not_promoted_to_official_score": True,
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-output", action="append", default=[], type=Path)
    parser.add_argument("--suite-output", action="append", default=[], type=Path)
    parser.add_argument("--peer-sources", type=Path)
    parser.add_argument("--algorithm-name", required=True)
    parser.add_argument("--execution-note", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    index = collect_delivery(
        native_outputs=args.native_output,
        suite_outputs=args.suite_output,
        algorithm_name=args.algorithm_name,
        execution_note=args.execution_note,
        peer_sources=args.peer_sources,
    )
    _write_json(args.out, index)
    print(json.dumps({
        "output": str(args.out.resolve()),
        "cells": len(index["cells"]),
        "comparison_cells": len(index["comparison_cells"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
