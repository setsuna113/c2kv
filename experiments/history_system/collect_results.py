"""Collect fixed-manifest quality and compression results for one candidate.

The collector is read-only except for explicitly requested JSON and Markdown
outputs.  A task is an algorithm success or failure only when its one-task
``official_summary.json`` is internally valid and every declared official
score pointer that can identify a score file is resolved and agrees with that
file's first JSON header.  Missing, ambiguous, malformed, or inconsistent
evidence remains ``unknown`` in the original manifest denominator.

Quality comparisons are paired only when the exact ordered task manifests are
identical.  Compression uses ``compression_metrics`` and therefore preserves
its per-task weighting, warmup rules, coverage-loss accounting, and strict
complete-coverage subset.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

if __package__:
    from experiments.history_system.compression_metrics import (
        DEFAULT_LOW_QUANTILE,
        aggregate_task_rows,
        analyze_run,
    )
else:  # Support ``python experiments/history_system/collect_results.py``.
    from compression_metrics import (
        DEFAULT_LOW_QUANTILE,
        aggregate_task_rows,
        analyze_run,
    )


COLLECTION_SCHEMA = "c2kv-history-system-result-collection-v1"
COMPARISON_SCHEMA = "c2kv-history-system-result-comparison-v1"
SAMPLE_LABEL = "preliminary, n=1"
TASK_ID_RE = re.compile(r"^multi_turn_(base|long_context)_(\d+)$")


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _task_variant(task_id: str) -> tuple[str, str | None]:
    match = TASK_ID_RE.fullmatch(task_id)
    if match is None:
        return "other", None
    variant = "long" if match.group(1) == "long_context" else "base"
    return variant, match.group(2)


def _resolve_results_roots(
    candidate_root: Path, explicit: Path | Sequence[Path] | None
) -> list[Path]:
    if explicit is not None:
        values = [explicit] if isinstance(explicit, Path) else list(explicit)
        if not values:
            raise ValueError("at least one results root is required")
        roots = [Path(value).resolve() for value in values]
        if len(set(roots)) != len(roots):
            raise ValueError("results roots must be unique")
        return roots
    returned = candidate_root / "returned"
    if returned.is_dir():
        return [returned.resolve()]
    results = candidate_root / "results"
    if results.is_dir():
        return [results.resolve()]
    return [candidate_root.resolve()]


def _resolve_design(candidate_root: Path, explicit: Path | None) -> Path:
    path = explicit if explicit is not None else candidate_root / "submitted" / "design.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"candidate design not found: {path}; use --design for a legacy tree"
        )
    return path.resolve()


def _load_manifest(
    candidate_root: Path,
    design: Mapping[str, Any],
    explicit: Path | None,
) -> tuple[list[str], Mapping[str, Any] | None, Path | None]:
    manifest_path = explicit
    if manifest_path is None:
        default = candidate_root / "submitted" / "tasks.json"
        if default.is_file():
            manifest_path = default
    manifest: Mapping[str, Any] | None = None
    if manifest_path is not None:
        manifest_path = manifest_path.resolve()
        loaded = _read_json(manifest_path)
        if not isinstance(loaded, Mapping):
            raise ValueError("task manifest must be a JSON object")
        manifest = loaded
        task_ids = manifest.get("task_ids")
    else:
        task_ids = design.get("task_ids")

    if not isinstance(task_ids, list) or not task_ids or any(
        not isinstance(task_id, str) or not task_id for task_id in task_ids
    ):
        raise ValueError("task manifest must contain a nonempty task_ids string list")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task manifest task_ids must be unique")

    design_task_ids = design.get("task_ids")
    if design_task_ids != task_ids:
        raise ValueError("design task_ids do not exactly match the task manifest")
    if manifest is not None:
        denominator = manifest.get("fixed_denominator")
        if denominator is not None and denominator != len(task_ids):
            raise ValueError("task manifest fixed_denominator does not match task_ids")
        expected_sha = design.get("task_manifest_sha256")
        if expected_sha is not None and expected_sha != _sha256(manifest_path):
            raise ValueError("design task_manifest_sha256 does not match the manifest file")
    return task_ids, manifest, manifest_path


def _resolve_summary(task_root: Path) -> tuple[Path | None, str | None]:
    direct = task_root / "bfcl" / "official_summary.json"
    if direct.is_file():
        return direct.resolve(), None
    matches = (
        sorted(task_root.rglob("official_summary.json"))
        if task_root.is_dir()
        else []
    )
    if len(matches) == 1:
        return matches[0].resolve(), None
    if len(matches) > 1:
        return None, "ambiguous_official_summary"
    legacy = sorted((task_root / "bfcl" / "official").glob("summary_*.json"))
    if len(legacy) == 1:
        return legacy[0].resolve(), None
    if len(legacy) > 1:
        return None, "ambiguous_legacy_official_summary"
    return None, "missing_official_summary"


def _task_result_root(
    results_roots: Sequence[Path], task_id: str
) -> tuple[Path, str | None]:
    matches = [
        root for root in results_roots if (root / "task_shards" / task_id).is_dir()
    ]
    if len(matches) == 1:
        return matches[0], None
    if len(matches) > 1:
        with_summary = []
        for root in matches:
            summary_path, summary_error = _resolve_summary(
                root / "task_shards" / task_id
            )
            if summary_error is None and summary_path is not None:
                with_summary.append(root)
        if len(with_summary) == 1:
            return with_summary[0], None
        return matches[0], "ambiguous_task_result_source"
    return results_roots[0], None


def _read_score_header(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("official score header is not a JSON object")
                return value
    raise ValueError("official score file has no JSON header")


def _resolve_score_pointer(summary_root: Path, pointer: str) -> tuple[Path | None, str | None]:
    name = Path(pointer.replace("\\", "/")).name
    if not name:
        return None, "invalid_official_score_pointer"
    matches = sorted(summary_root.rglob(name))
    if len(matches) == 1:
        return matches[0].resolve(), None
    if not matches:
        return None, "missing_official_score_file"
    return None, "ambiguous_official_score_file"


def _numbers_agree(left: object, right: object) -> bool:
    return _finite_number(left) and _finite_number(right) and math.isclose(
        float(left), float(right), rel_tol=0.0, abs_tol=1e-12
    )


def _collect_official_task(
    results_roots: Sequence[Path], task_id: str
) -> dict[str, Any]:
    variant, group_id = _task_variant(task_id)
    row: dict[str, Any] = {
        "task_id": task_id,
        "variant": variant,
        "group_id": group_id,
        "outcome": "unknown",
        "correct": None,
        "unknown_reason": None,
        "official_summary": None,
        "official_score_headers": [],
    }
    results_root, root_error = _task_result_root(results_roots, task_id)
    row["result_source_root"] = str(results_root)
    if root_error is not None:
        row["unknown_reason"] = root_error
        return row
    task_root = results_root / "task_shards" / task_id
    summary_path, error = _resolve_summary(task_root)
    if error is not None:
        row["unknown_reason"] = error
        return row
    assert summary_path is not None
    row["official_summary"] = _source(summary_path)
    try:
        summary = _read_json(summary_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        row["unknown_reason"] = f"invalid_official_summary:{type(exc).__name__}"
        return row
    if not isinstance(summary, Mapping):
        row["unknown_reason"] = "invalid_official_summary:not_object"
        return row

    correct = summary.get("correct_count")
    scored = summary.get("n_scored")
    if (
        summary.get("scored") is not True
        or not _nonnegative_int(correct)
        or not _nonnegative_int(scored)
        or scored != 1
        or correct not in (0, 1)
        or ("n_total" in summary and summary.get("n_total") != 1)
        or ("n" in summary and summary.get("n") != 1)
    ):
        row["unknown_reason"] = "official_summary_not_one_scored_task"
        return row

    declared_headers = summary.get("official_score_headers", [])
    if not isinstance(declared_headers, list):
        row["unknown_reason"] = "invalid_official_score_headers"
        return row
    header_correct = 0
    header_total = 0
    for index, declared in enumerate(declared_headers):
        if not isinstance(declared, Mapping):
            row["unknown_reason"] = "invalid_official_score_header_entry"
            return row
        pointer = declared.get("path")
        if not isinstance(pointer, str) or not pointer:
            row["unknown_reason"] = "invalid_official_score_pointer"
            return row
        score_path, pointer_error = _resolve_score_pointer(summary_path.parent, pointer)
        header_row: dict[str, Any] = {
            "index": index,
            "declared_path": pointer,
            "resolved_source": None,
            "status": "unknown",
        }
        row["official_score_headers"].append(header_row)
        if pointer_error is not None:
            header_row["reason"] = pointer_error
            row["unknown_reason"] = pointer_error
            return row
        assert score_path is not None
        header_row["resolved_source"] = _source(score_path)
        try:
            header = _read_score_header(score_path)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            header_row["reason"] = f"invalid_official_score_header:{type(exc).__name__}"
            row["unknown_reason"] = header_row["reason"]
            return row
        header_correct_value = header.get("correct_count")
        header_total_value = header.get("total_count")
        header_accuracy = header.get("accuracy")
        if (
            not _nonnegative_int(header_correct_value)
            or not _nonnegative_int(header_total_value)
            or header_correct_value > header_total_value
            or header_total_value == 0
            or not _finite_number(header_accuracy)
            or not _numbers_agree(
                header_accuracy, header_correct_value / header_total_value
            )
        ):
            header_row["reason"] = "invalid_official_score_header_counts"
            row["unknown_reason"] = header_row["reason"]
            return row
        for key in ("correct_count", "total_count", "accuracy"):
            if key in declared and not _numbers_agree(declared[key], header[key]):
                header_row["reason"] = f"official_score_pointer_header_mismatch:{key}"
                row["unknown_reason"] = header_row["reason"]
                return row
        header_correct += header_correct_value
        header_total += header_total_value
        header_row["status"] = "validated"
        header_row["correct_count"] = header_correct_value
        header_row["total_count"] = header_total_value
        header_row["accuracy"] = float(header_accuracy)

    if declared_headers and (header_correct != correct or header_total != scored):
        row["unknown_reason"] = "official_summary_score_header_aggregate_mismatch"
        return row

    row["correct"] = bool(correct)
    row["outcome"] = "algorithm_success" if correct == 1 else "algorithm_failure"
    row["unknown_reason"] = None
    row["summary_counts"] = {
        "correct_count": correct,
        "n_scored": scored,
        "semantic_score": summary.get("semantic_score"),
    }
    row["score_header_status"] = (
        "validated" if declared_headers else "not_declared"
    )
    return row


def _quality_scope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    denominator = len(rows)
    success = sum(row["outcome"] == "algorithm_success" for row in rows)
    failure = sum(row["outcome"] == "algorithm_failure" for row in rows)
    unknown = denominator - success - failure
    scored = success + failure
    return {
        "sample_label": SAMPLE_LABEL,
        "fixed_denominator": denominator,
        "algorithm_successes": success,
        "algorithm_failures": failure,
        "unknown": unknown,
        "scored_tasks": scored,
        "accuracy_over_fixed_denominator_lower_bound": (
            success / denominator if denominator else None
        ),
        "official_accuracy_over_fixed_denominator": (
            success / denominator if denominator and unknown == 0 else None
        ),
        "accuracy_on_scored_known_subset": success / scored if scored else None,
    }


def _group_rows(
    task_rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any] | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_group: dict[str, dict[str, Mapping[str, Any]]] = {}
    derived_order: list[str] = []
    for row in task_rows:
        group_id = row.get("group_id")
        if group_id is None:
            continue
        group_id = str(group_id)
        if group_id not in by_group:
            by_group[group_id] = {}
            derived_order.append(group_id)
        by_group[group_id][str(row["variant"])] = row

    declared = manifest.get("group_ordinals") if manifest is not None else None
    if isinstance(declared, list) and all(
        isinstance(value, (str, int)) and not isinstance(value, bool)
        for value in declared
    ):
        order = [str(value) for value in declared]
        for value in derived_order:
            if value not in order:
                order.append(value)
    else:
        order = derived_order

    groups: list[dict[str, Any]] = []
    for group_id in order:
        members = by_group.get(group_id, {})
        base = members.get("base")
        long = members.get("long")
        pair_complete = base is not None and long is not None
        outcomes = [member["outcome"] for member in (base, long) if member is not None]
        pair_known = pair_complete and all(outcome != "unknown" for outcome in outcomes)
        groups.append(
            {
                "group_id": group_id,
                "base_task_id": base["task_id"] if base is not None else None,
                "base_outcome": base["outcome"] if base is not None else None,
                "long_task_id": long["task_id"] if long is not None else None,
                "long_outcome": long["outcome"] if long is not None else None,
                "pair_complete_in_manifest": pair_complete,
                "pair_result_known": pair_known,
                "known_pair_success_count": (
                    sum(outcome == "algorithm_success" for outcome in outcomes)
                    if pair_known
                    else None
                ),
            }
        )
    complete = [group for group in groups if group["pair_complete_in_manifest"]]
    known = [group for group in complete if group["pair_result_known"]]
    histogram = Counter(group["known_pair_success_count"] for group in known)
    summary = {
        "sample_label": SAMPLE_LABEL,
        "declared_group_denominator": len(groups),
        "complete_base_long_pairs_in_manifest": len(complete),
        "unpaired_manifest_groups": len(groups) - len(complete),
        "known_complete_pair_results": len(known),
        "unknown_complete_pair_results": len(complete) - len(known),
        "known_pair_success_count_histogram": {
            "zero": histogram[0],
            "one": histogram[1],
            "two": histogram[2],
        },
    }
    return groups, summary


def _derive_b0_cap(design: Mapping[str, Any]) -> tuple[int | None, str | None]:
    candidates = [
        (design.get("search_contract"), "B0_history_bytes"),
        (design.get("b0_contract"), "history_budget_bytes"),
    ]
    resolved = design.get("resolved_configs")
    if isinstance(resolved, Mapping):
        policy = resolved.get("eval_policy")
        if isinstance(policy, Mapping):
            policy = policy.get("policy")
            candidates.append((policy, "history_budget_bytes"))
    for container, key in candidates:
        if isinstance(container, Mapping):
            value = container.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value, f"design.{key}"
    return None, None


def _collect_compression(
    results_roots: Sequence[Path],
    task_ids: Sequence[str],
    design: Mapping[str, Any],
    explicit_b0_cap_bytes: int | None,
    low_quantile: float,
) -> dict[str, Any]:
    derived_cap, cap_source = _derive_b0_cap(design)
    cap = explicit_b0_cap_bytes if explicit_b0_cap_bytes is not None else derived_cap
    if cap is None:
        return {
            "status": "unknown",
            "unknown_reasons": ["missing_b0_cap_bytes"],
            "b0_cap_bytes": None,
            "b0_cap_source": None,
            "metrics": None,
        }
    if cap <= 0:
        raise ValueError("b0_cap_bytes must be positive")
    task_metrics = []
    ambiguous_tasks = []
    for task_id in task_ids:
        results_root, root_error = _task_result_root(results_roots, task_id)
        if root_error is not None:
            ambiguous_tasks.append(task_id)
            results_root = results_roots[0] / "__ambiguous_task_result_source__"
        task_metrics.append(
            analyze_run(
                results_root,
                [task_id],
                b0_cap_bytes=cap,
                low_quantile=low_quantile,
            )["tasks"][0]
        )
    metrics = aggregate_task_rows(
        task_metrics, b0_cap_bytes=cap, low_quantile=low_quantile
    )
    completeness = metrics["receipt_completeness"]
    reasons: list[str] = []
    if completeness["tasks_with_source_file"] < len(task_ids):
        reasons.append("missing_step_sources")
    if completeness["tasks_with_activated_measurement"] < len(task_ids):
        reasons.append("tasks_without_activated_compression_measurements")
    if completeness["malformed_records"]:
        reasons.append("malformed_step_records")
    if completeness["controllerless_step_rows"]:
        reasons.append("controllerless_step_rows")
    if completeness["receipt_error_count"]:
        reasons.append("invalid_compression_receipts")
    if completeness["full_history_without_active_footprint_anomalies"]:
        reasons.append("full_history_without_active_footprint")
    if metrics["source_coverage"]["coverage_unknown_receipts"]:
        reasons.append("unknown_source_coverage_receipts")
    if ambiguous_tasks:
        reasons.append("ambiguous_task_result_sources")
    return {
        "status": "available" if not reasons else "partial_or_unknown",
        "unknown_reasons": reasons,
        "b0_cap_bytes": cap,
        "b0_cap_source": "cli" if explicit_b0_cap_bytes is not None else cap_source,
        "metrics": metrics,
    }


def _steps_sources(
    results_roots: Sequence[Path], task_ids: Sequence[str]
) -> list[dict[str, Any]]:
    sources = []
    for task_id in task_ids:
        results_root, root_error = _task_result_root(results_roots, task_id)
        path = results_root / "task_shards" / task_id / "server" / "steps.jsonl"
        sources.append(
            {
                "task_id": task_id,
                "result_source_root": str(results_root),
                "source": _source(path) if path.is_file() else None,
                "status": (
                    "ambiguous"
                    if root_error is not None
                    else "present"
                    if path.is_file()
                    else "missing"
                ),
            }
        )
    return sources


def collect_candidate(
    candidate_root: Path,
    *,
    results_root: Path | Sequence[Path] | None = None,
    design_path: Path | None = None,
    task_manifest_path: Path | None = None,
    candidate_label: str | None = None,
    b0_cap_bytes: int | None = None,
    low_quantile: float = DEFAULT_LOW_QUANTILE,
) -> dict[str, Any]:
    """Collect one candidate without mutating its submitted or result tree."""
    candidate_root = Path(candidate_root).resolve()
    resolved_design = _resolve_design(candidate_root, design_path)
    design = _read_json(resolved_design)
    if not isinstance(design, Mapping):
        raise ValueError("candidate design must be a JSON object")
    task_ids, manifest, manifest_path = _load_manifest(
        candidate_root, design, task_manifest_path
    )
    resolved_results = _resolve_results_roots(candidate_root, results_root)
    task_rows = [
        _collect_official_task(resolved_results, task_id) for task_id in task_ids
    ]
    groups, group_summary = _group_rows(task_rows, manifest)
    manifest_identity = {
        "ordered_task_ids_sha256": _canonical_sha256(task_ids),
        "task_ids": task_ids,
        "fixed_denominator": len(task_ids),
        "manifest_id": manifest.get("manifest_id") if manifest is not None else None,
        "manifest_source": _source(manifest_path) if manifest_path is not None else None,
        "identity_rule": "exact ordered task_ids; names and cohort sizes alone do not match",
    }
    label = candidate_label or design.get("candidate_id") or candidate_root.name
    compression = _collect_compression(
        resolved_results,
        task_ids,
        design,
        b0_cap_bytes,
        low_quantile,
    )
    return {
        "schema": COLLECTION_SCHEMA,
        "candidate_label": label,
        "candidate_id": design.get("candidate_id"),
        "sample_label": SAMPLE_LABEL,
        "manifest_identity": manifest_identity,
        "quality": {
            "overall": _quality_scope(task_rows),
            "base": _quality_scope(
                [row for row in task_rows if row["variant"] == "base"]
            ),
            "long": _quality_scope(
                [row for row in task_rows if row["variant"] == "long"]
            ),
            "other": _quality_scope(
                [row for row in task_rows if row["variant"] == "other"]
            ),
            "missing_and_invalid_evidence_is_unknown_not_failure": True,
        },
        "group_pairs": {"summary": group_summary, "groups": groups},
        "compression": compression,
        "unknown_costs": compression["unknown_reasons"],
        "sources": {
            "candidate_root": str(candidate_root),
            "results_roots": [str(path) for path in resolved_results],
            "design": _source(resolved_design),
            "task_manifest": (
                _source(manifest_path) if manifest_path is not None else None
            ),
            "steps": _steps_sources(resolved_results, task_ids),
        },
        "scorer_source_bindings": (
            design.get("task_and_scorer_lineage", {}).get("source_bindings")
            if isinstance(design.get("task_and_scorer_lineage"), Mapping)
            else None
        ),
        "tasks": task_rows,
    }


def compare_collections(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> dict[str, Any]:
    """Return descriptive paired quality deltas for identical ordered manifests."""
    left_manifest = left["manifest_identity"]
    right_manifest = right["manifest_identity"]
    same = left_manifest["task_ids"] == right_manifest["task_ids"]
    result: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "left": left.get("candidate_label"),
        "right": right.get("candidate_label"),
        "sample_label": SAMPLE_LABEL,
        "quality_comparable": same,
        "reason": None if same else "task_manifest_mismatch",
        "comparison_scope": {
            "kind": "same-task descriptive whole-system comparison",
            "ordered_task_manifest_identity_verified": same,
            "checkpoint_scorer_sampling_identity": (
                "must be verified from source bindings outside this manifest-only gate"
            ),
            "checkpoint_scorer_sampling_identity_verified_by_collector": False,
            "strict_causal_claim": False,
        },
        "quality_delta": None,
        "paired_tasks": None,
    }
    if not same:
        return result
    left_rows = {row["task_id"]: row for row in left["tasks"]}
    right_rows = {row["task_id"]: row for row in right["tasks"]}
    transitions: Counter[str] = Counter()
    known_pairs = 0
    for task_id in left_manifest["task_ids"]:
        left_outcome = left_rows[task_id]["outcome"]
        right_outcome = right_rows[task_id]["outcome"]
        if "unknown" in (left_outcome, right_outcome):
            transitions["unknown_pair"] += 1
            continue
        known_pairs += 1
        transitions[f"{left_outcome}_to_{right_outcome}"] += 1
    left_overall = left["quality"]["overall"]
    right_overall = right["quality"]["overall"]
    result["paired_tasks"] = {
        "fixed_denominator": len(left_manifest["task_ids"]),
        "known_pairs": known_pairs,
        "unknown_pairs": len(left_manifest["task_ids"]) - known_pairs,
        "transitions": dict(sorted(transitions.items())),
    }
    if left_overall["unknown"] == 0 and right_overall["unknown"] == 0:
        result["quality_delta"] = {
            "algorithm_success_count_right_minus_left": (
                right_overall["algorithm_successes"]
                - left_overall["algorithm_successes"]
            ),
            "official_accuracy_right_minus_left": (
                right_overall["official_accuracy_over_fixed_denominator"]
                - left_overall["official_accuracy_over_fixed_denominator"]
            ),
        }
    return result


def _fmt_quality(value: object, *, ratio: bool = False) -> str:
    if value is None:
        return "unknown"
    if ratio:
        return f"{float(value):.4f} ({SAMPLE_LABEL})"
    return f"{value} ({SAMPLE_LABEL})"


def _fmt_number(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(
    collection: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Render a compact audit table; every numeric quality cell is labelled."""
    lines = [
        f"# {collection['candidate_label']} result index",
        "",
        f"Sample: `{SAMPLE_LABEL}`. Missing or invalid official evidence stays `unknown` in the fixed denominator.",
        "",
        "## Quality",
        "",
        "| Scope | Success | Failure | Unknown | Scored | Denominator | Official accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scope in ("overall", "base", "long", "other"):
        quality = collection["quality"][scope]
        if quality["fixed_denominator"] == 0:
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    scope,
                    _fmt_quality(quality["algorithm_successes"]),
                    _fmt_quality(quality["algorithm_failures"]),
                    _fmt_quality(quality["unknown"]),
                    _fmt_quality(quality["scored_tasks"]),
                    _fmt_quality(quality["fixed_denominator"]),
                    _fmt_quality(
                        quality["official_accuracy_over_fixed_denominator"], ratio=True
                    ),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Base/long groups",
            "",
            "| Group | Base | Long | Manifest pair | Result known |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for group in collection["group_pairs"]["groups"]:
        lines.append(
            f"| {group['group_id']} | {group['base_outcome'] or 'absent'} | "
            f"{group['long_outcome'] or 'absent'} | "
            f"{'yes' if group['pair_complete_in_manifest'] else 'no'} | "
            f"{'yes' if group['pair_result_known'] else 'no'} |"
        )

    compression = collection["compression"]
    lines.extend(["", "## Compression and cost", ""])
    if compression["metrics"] is None:
        lines.append(
            "Compression/cost: `unknown` ("
            + ", ".join(compression["unknown_reasons"])
            + ")."
        )
    else:
        metrics = compression["metrics"]
        active = metrics["task_equal"]["system_active_history_reduction"]
        full = metrics["task_equal"]["system_full_input_reduction"]
        strict = metrics["task_equal"][
            "strict_complete_coverage_system_active_history_reduction"
        ]
        coverage = metrics["source_coverage"]
        lines.extend(
            [
                "| Metric | Value |",
                "| --- | ---: |",
                f"| H/A task-weighted median | {_fmt_number(active['task_equal_weighted_activated_receipts']['all_selected_task_weighted_median'])} |",
                f"| H/A task-weighted q{int(metrics['low_quantile_probability'] * 100)} | {_fmt_number(active['task_equal_weighted_activated_receipts']['all_selected_task_weighted_low_quantile'])} |",
                f"| Full-input task-weighted median | {_fmt_number(full['task_equal_weighted_activated_receipts']['all_selected_task_weighted_median'])} |",
                f"| Strict complete-coverage H/A median | {_fmt_number(strict['task_equal_weighted_activated_receipts']['all_selected_task_weighted_median'])} |",
                f"| Active peak bytes | {_fmt_number(metrics['active_history']['observed_peak_bytes'])} |",
                f"| B0 violations | {_fmt_number(metrics['active_history']['observed_b0_violation_count'])} |",
                f"| Coverage-loss receipts | {_fmt_number(coverage['coverage_loss_receipts'])}/{_fmt_number(coverage['activated_receipts'])} |",
                f"| Fully represented source-occurrence fraction | {_fmt_number(coverage['fully_represented_source_occurrence_fraction'])} |",
            ]
        )
        if compression["unknown_reasons"]:
            lines.extend(
                [
                    "",
                    "Unknown cost coverage: `"
                    + ", ".join(compression["unknown_reasons"])
                    + "`.",
                ]
            )

    lines.extend(
        [
            "",
            "## Sources",
            "",
            f"- Design SHA-256: `{collection['sources']['design']['sha256']}`",
            f"- Ordered task manifest SHA-256: `{collection['manifest_identity']['ordered_task_ids_sha256']}`",
        ]
    )
    manifest_source = collection["sources"]["task_manifest"]
    if manifest_source is not None:
        lines.append(f"- Task manifest file SHA-256: `{manifest_source['sha256']}`")
    summary_hashes = [
        {"task_id": row["task_id"], "sha256": row["official_summary"]["sha256"]}
        for row in collection["tasks"]
        if row["official_summary"] is not None
    ]
    score_hashes = [
        {
            "task_id": row["task_id"],
            "index": header["index"],
            "sha256": header["resolved_source"]["sha256"],
        }
        for row in collection["tasks"]
        for header in row["official_score_headers"]
        if header["resolved_source"] is not None
    ]
    lines.append(
        f"- Official summary files: `{len(summary_hashes)}`; source-set SHA-256: "
        + (f"`{_canonical_sha256(summary_hashes)}`" if summary_hashes else "`none`")
    )
    lines.append(
        f"- Resolved official score headers: `{len(score_hashes)}`; source-set SHA-256: "
        + (f"`{_canonical_sha256(score_hashes)}`" if score_hashes else "`none`")
    )

    if comparisons:
        lines.extend(["", "## Quality comparisons", ""])
        for comparison in comparisons:
            if not comparison["quality_comparable"]:
                lines.append(
                    f"- `{comparison['left']}` vs `{comparison['right']}`: noncomparable (`{comparison['reason']}`)."
                )
            elif comparison["quality_delta"] is None:
                lines.append(
                    f"- `{comparison['left']}` vs `{comparison['right']}`: identical manifest, but the quality delta is `unknown` because at least one task outcome is unknown."
                )
            else:
                delta = comparison["quality_delta"]
                lines.append(
                    f"- `{comparison['left']}` vs `{comparison['right']}`: success delta "
                    f"{_fmt_quality(delta['algorithm_success_count_right_minus_left'])}; "
                    f"accuracy delta {_fmt_quality(delta['official_accuracy_right_minus_left'], ratio=True)}."
                )
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--results-root",
        type=Path,
        action="append",
        help="result tree containing task_shards; repeat for legacy continuations",
    )
    parser.add_argument("--design", type=Path)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--candidate-label")
    parser.add_argument("--b0-cap-bytes", type=int)
    parser.add_argument("--low-quantile", type=float, default=DEFAULT_LOW_QUANTILE)
    parser.add_argument("--compare-json", type=Path, action="append", default=[])
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args(argv)

    collection = collect_candidate(
        args.candidate_root,
        results_root=args.results_root,
        design_path=args.design,
        task_manifest_path=args.task_manifest,
        candidate_label=args.candidate_label,
        b0_cap_bytes=args.b0_cap_bytes,
        low_quantile=args.low_quantile,
    )
    comparisons = []
    for path in args.compare_json:
        other = _read_json(path)
        if not isinstance(other, Mapping) or other.get("schema") != COLLECTION_SCHEMA:
            raise ValueError(f"comparison input is not a collection: {path}")
        comparisons.append(compare_collections(other, collection))
    output = {**collection, "comparisons": comparisons}
    _write_json(args.json_out, output)
    _write_text(args.markdown_out, render_markdown(collection, comparisons))
    print(
        json.dumps(
            {
                "candidate_label": collection["candidate_label"],
                "json": str(args.json_out.resolve()),
                "markdown": str(args.markdown_out.resolve()),
                "quality": collection["quality"]["overall"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
