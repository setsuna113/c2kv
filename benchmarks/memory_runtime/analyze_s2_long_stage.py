"""Analyze paired S2 Long20 stages without running models or scorers."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


BENCHMARKS_ROOT = Path(__file__).resolve().parents[1]
if str(BENCHMARKS_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_ROOT))

import reqlog  # noqa: E402
from memory_runtime import analyze_s1_long_stage as common  # noqa: E402


SCHEMA = "a-s2-long20-paired-analysis-v1"
STAGE_SCHEMA = common.STAGE_SCHEMA
EXPECTED_TASKS = common.EXPECTED_TASKS
FINAL_STAGE_STATUS = common.FINAL_STAGE_STATUS
S2 = "s2_subgoal"
CONTROL = "s2_note_turn_control"
CANDIDATES = {
    S2: {"history_organization": "subgoal-v1"},
    CONTROL: {"history_organization": "turn"},
}


class S2LongAnalysisError(RuntimeError):
    """Raised when artifacts cannot support the requested analysis mode."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return common._read_json(path)
    except common.S1LongAnalysisError as error:
        raise S2LongAnalysisError(str(error)) from error


def _single(paths: Iterable[Path], label: str) -> tuple[Path | None, str | None]:
    return common._single(paths, label)


def _sha256(path: Path) -> str:
    return common._sha256(path)


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _outcomes(
    manifest: Mapping[str, Any], sidecar_root: Path, observation_path: Path | None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    try:
        records, sources = common._outcome_records(manifest, sidecar_root)
    except common.S1LongAnalysisError as error:
        raise S2LongAnalysisError(str(error)) from error
    if observation_path is None:
        return records, sources
    observation = _read_json(observation_path)
    completed = observation.get("completed") or []
    if not isinstance(completed, list):
        raise S2LongAnalysisError("observation.completed is not a list")
    used = False
    for row in completed:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
            raise S2LongAnalysisError("observation contains a malformed completed row")
        existing = records.get(row["task_id"])
        if existing is not None:
            for key in ("official_score_known", "official_score_header", "operational_outcome"):
                if key in existing and key in row and existing[key] != row[key]:
                    raise S2LongAnalysisError(
                        f"observation disagrees with manifest outcome for {row['task_id']}")
            continue
        records[row["task_id"]] = row
        used = True
    if used:
        sources.append(str(observation_path))
    return records, sources


def _subgoal_exposure(rows: list[dict[str, Any]], candidate_id: str) -> dict[str, Any]:
    expected_organization = CANDIDATES[candidate_id]["history_organization"]
    organization_counts = Counter()
    prompt_protocol_counts = Counter()
    lifecycle_receipts = 0
    views_with_valid_subgoals = 0
    views_with_closed_by_next = 0
    views_with_multi_fragment_group = 0
    record_appearances = 0
    group_fragment_appearances = 0
    subgoal_group_fragment_appearances = 0
    unassigned_group_fragment_appearances = 0
    multi_source_group_fragment_appearances = 0
    multi_fragment_group_view_appearances = 0
    group_status_appearances = Counter()
    violation_reason_appearances = Counter()
    distinct_groups: set[str] = set()
    distinct_violations: set[tuple[str, str]] = set()
    latest_records: dict[str, Mapping[str, Any]] = {}
    maxima = Counter()

    for row in rows:
        runtime = row.get("memory_runtime")
        if not isinstance(runtime, Mapping):
            continue
        organization = runtime.get("history_organization")
        protocol = runtime.get("actor_prompt_protocol")
        organization_counts[str(organization)] += 1
        prompt_protocol_counts[str(protocol)] += 1
        if organization != expected_organization:
            raise S2LongAnalysisError(
                f"{candidate_id} request changed history_organization")
        if protocol != "native-subgoal-note-v1":
            raise S2LongAnalysisError(
                f"{candidate_id} request changed actor_prompt_protocol")

        lifecycle = runtime.get("subgoal_organization")
        groups = runtime.get("retained_subgoal_groups")
        if candidate_id == CONTROL:
            if lifecycle is not None or groups is not None:
                raise S2LongAnalysisError(
                    "Matched-note turn control unexpectedly carries subgoal grouping")
            continue
        if lifecycle is None:
            if groups not in (None, []):
                raise S2LongAnalysisError("Subgoal groups lack their lifecycle receipt")
            continue
        if not isinstance(lifecycle, Mapping):
            raise S2LongAnalysisError("subgoal_organization is not an object")
        lifecycle_receipts += 1
        if lifecycle.get("version") != "observable-subgoal-lifecycle-v1":
            raise S2LongAnalysisError("Subgoal lifecycle version changed")
        for key in (
            "n_started", "n_active", "n_archivable",
            "n_closed_by_next_declaration", "n_closed_by_user_turn",
        ):
            value = lifecycle.get(key)
            if type(value) is not int or value < 0:
                raise S2LongAnalysisError(f"Malformed subgoal lifecycle count: {key}")
            maxima[f"maximum_{key}"] = max(maxima[f"maximum_{key}"], value)
        views_with_valid_subgoals += lifecycle["n_started"] > 0
        views_with_closed_by_next += lifecycle["n_closed_by_next_declaration"] > 0
        records = lifecycle.get("records") or []
        violations = lifecycle.get("violations") or []
        if not isinstance(records, list) or not isinstance(violations, list):
            raise S2LongAnalysisError("Subgoal record or violation ledger is malformed")
        record_appearances += len(records)
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get("subgoal_id"), str):
                raise S2LongAnalysisError("Subgoal record lacks a stable subgoal_id")
            latest_records[record["subgoal_id"]] = record
        for violation in violations:
            if not isinstance(violation, Mapping):
                raise S2LongAnalysisError("Subgoal violation is not an object")
            reason = str(violation.get("reason"))
            event_id = str(violation.get("event_id"))
            violation_reason_appearances[reason] += 1
            distinct_violations.add((event_id, reason))

        if not isinstance(groups, list):
            raise S2LongAnalysisError("retained_subgoal_groups is not a list")
        per_view_groups = Counter()
        for group in groups:
            if not isinstance(group, Mapping) or not isinstance(group.get("group_id"), str):
                raise S2LongAnalysisError("Subgoal group lacks a stable group_id")
            sources = group.get("source_indices")
            if (not isinstance(sources, list)
                    or not all(type(index) is int and index >= 0 for index in sources)):
                raise S2LongAnalysisError("Subgoal group source provenance is malformed")
            group_fragment_appearances += 1
            distinct_groups.add(group["group_id"])
            per_view_groups[group["group_id"]] += 1
            status = str(group.get("lifecycle_status"))
            group_status_appearances[status] += 1
            subgoal_group_fragment_appearances += group.get("subgoal_id") is not None
            unassigned_group_fragment_appearances += group.get("subgoal_id") is None
            multi_source_group_fragment_appearances += len(sources) > 1
        fragmented = sum(count > 1 for count in per_view_groups.values())
        views_with_multi_fragment_group += fragmented > 0
        multi_fragment_group_view_appearances += fragmented

    latest_statuses = Counter(str(record.get("lifecycle_status"))
                              for record in latest_records.values())
    lifecycle_not_applicable = len(rows) if candidate_id == CONTROL else 0
    return {
        "request_count": len(rows),
        "history_organization_counts": dict(sorted(organization_counts.items())),
        "actor_prompt_protocol_counts": dict(sorted(prompt_protocol_counts.items())),
        "known_lifecycle_receipts": lifecycle_receipts,
        "missing_lifecycle_receipts": (
            0 if candidate_id == CONTROL else len(rows) - lifecycle_receipts),
        "lifecycle_receipts_not_applicable": lifecycle_not_applicable,
        "views_with_valid_subgoals": views_with_valid_subgoals,
        "views_with_closed_by_next_declaration": views_with_closed_by_next,
        "valid_subgoal_record_appearances": record_appearances,
        "distinct_valid_subgoals": len(latest_records),
        "latest_lifecycle_status_counts": dict(sorted(latest_statuses.items())),
        "distinct_closed_by_next_declaration": latest_statuses["closed_by_next_declaration"],
        "distinct_closed_by_user_turn": latest_statuses["closed_by_user_turn"],
        "distinct_active": latest_statuses["active"],
        **dict(maxima),
        "declaration_violation_appearances": sum(violation_reason_appearances.values()),
        "distinct_declaration_violations": len(distinct_violations),
        "declaration_violation_reason_appearance_counts": dict(
            sorted(violation_reason_appearances.items())),
        "retained_group_fragment_appearances": group_fragment_appearances,
        "distinct_retained_groups": len(distinct_groups),
        "retained_subgoal_group_fragment_appearances": subgoal_group_fragment_appearances,
        "retained_unassigned_group_fragment_appearances": unassigned_group_fragment_appearances,
        "multi_source_group_fragment_appearances": multi_source_group_fragment_appearances,
        "group_lifecycle_status_appearance_counts": dict(sorted(group_status_appearances.items())),
        "views_with_multi_fragment_group": views_with_multi_fragment_group,
        "multi_fragment_group_view_appearances": multi_fragment_group_view_appearances,
        "scope": (
            "Lifecycle records are prefix snapshots. Distinct counts use stable IDs and "
            "latest observed status; group appearances count retained packing fragments. "
            "closed_by_next_declaration is an observable boundary, not task success."
        ),
    }


def _reencoding_and_coverage_cost(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    seen_producer_sources: set[tuple[str, int]] = set()
    distinct_producer_keys: set[str] = set()
    known_prefill_values: list[int | float] = []
    telemetry_receipts = 0
    for row in rows:
        runtime = row.get("memory_runtime")
        coverage = runtime.get("source_coverage") if isinstance(runtime, Mapping) else None
        if isinstance(coverage, Mapping):
            for source, target in (
                ("fitted_fragment_count", "fitted_fragment_appearances"),
                ("retained_fragment_count", "retained_fragment_appearances"),
                ("fitted_encoder_input_tokens", "fitted_encoder_input_token_appearances"),
                ("retained_encoder_input_tokens", "retained_encoder_input_token_appearances"),
            ):
                value = coverage.get(source)
                if type(value) is not int or value < 0:
                    raise S2LongAnalysisError(f"Malformed source coverage cost: {source}")
                totals[target] += value
        telemetry = row.get("extraction_telemetry")
        if not isinstance(telemetry, Mapping):
            continue
        telemetry_receipts += 1
        summary = telemetry.get("summary") or {}
        if not isinstance(summary, Mapping):
            raise S2LongAnalysisError("Extraction telemetry summary is malformed")
        for key in (
            "lookups", "client_cache_hits", "producer_calls", "producer_successes",
            "producer_failures", "producer_response_original_seq_len_sum",
        ):
            value = summary.get(key)
            if type(value) is not int or value < 0:
                raise S2LongAnalysisError(f"Malformed extraction summary count: {key}")
            totals[key] += value
        prefill = summary.get("actual_extraction_prefill_tokens")
        if _finite(prefill) and prefill >= 0:
            known_prefill_values.append(prefill)
        events = telemetry.get("events") or []
        if not isinstance(events, list):
            raise S2LongAnalysisError("Extraction telemetry events are malformed")
        task_id = str((row.get("eval_context") or {}).get("task_id"))
        for event in events:
            if not isinstance(event, Mapping):
                raise S2LongAnalysisError("Extraction event is not an object")
            sources = event.get("source_indices") or []
            if not isinstance(sources, list) or not all(type(index) is int for index in sources):
                raise S2LongAnalysisError("Extraction event source provenance is malformed")
            totals["source_occurrence_lookup_appearances"] += len(sources)
            if event.get("producer_called") is True:
                identities = {(task_id, index) for index in sources}
                totals["producer_source_occurrence_appearances"] += len(identities)
                repeated = identities & seen_producer_sources
                totals["reencoded_source_occurrence_appearances"] += len(repeated)
                totals["producer_events_reencoding_prior_sources"] += bool(repeated)
                seen_producer_sources.update(identities)
                key_hash = event.get("key_hash")
                if isinstance(key_hash, str):
                    distinct_producer_keys.add(key_hash)
    return {
        "known_extraction_telemetry_receipts": telemetry_receipts,
        "missing_extraction_telemetry_receipts": len(rows) - telemetry_receipts,
        **dict(totals),
        "dropped_fragment_appearances": (
            totals["fitted_fragment_appearances"]
            - totals["retained_fragment_appearances"]),
        "distinct_source_occurrences_encoded_by_producer": len(seen_producer_sources),
        "distinct_producer_key_hashes": len(distinct_producer_keys),
        "actual_extraction_prefill_tokens_known_receipts": len(known_prefill_values),
        "actual_extraction_prefill_tokens_known_total": sum(known_prefill_values),
        "actual_extraction_prefill_tokens_strict_total": (
            sum(known_prefill_values)
            if telemetry_receipts and len(known_prefill_values) == telemetry_receipts
            else None
        ),
        "scope": (
            "Fitted/retained encoder tokens are per-request input-token appearances. "
            "Producer calls are actual client producer invocations. Reencoding counts "
            "source occurrence appearances repeated across producer-called fragments, "
            "not exact per-source token attribution."
        ),
    }


def _pair(candidate: Mapping[str, Any], other: Mapping[str, Any], names: tuple[str, str]) -> str:
    left = candidate["quality"]
    right = other["quality"]
    if not left["official_score_known"] or not right["official_score_known"]:
        return "unpaired_missing_official"
    if left["correct_count"] == 1 and right["correct_count"] == 1:
        return "both_correct"
    if left["correct_count"] == 1:
        return f"{names[0]}_only_correct"
    if right["correct_count"] == 1:
        return f"{names[1]}_only_correct"
    return "both_incorrect"


def _candidate(
    candidate_id: str,
    manifest_path: Path,
    shards_root: Path,
    sidecar_root: Path,
    observation_path: Path | None,
    task_ids: list[str],
    s0_by_task: Mapping[str, Mapping[str, Any]],
    mode: str,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if (manifest.get("schema") != STAGE_SCHEMA
            or manifest.get("candidate_id") != candidate_id
            or manifest.get("task_ids") != task_ids):
        raise S2LongAnalysisError(f"Stage manifest does not match {candidate_id} Long20")
    outcomes, outcome_sources = _outcomes(manifest, sidecar_root, observation_path)
    per_task: list[dict[str, Any]] = []
    completion_errors: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        shard = Path(shards_root) / task_id / candidate_id
        exists = shard.is_dir()
        score_path, score_path_error = _single(
            (shard / "score").rglob(common.SCORE_NAME), "official_score") if exists else (None, "missing_shard")
        request_path, request_path_error = _single(
            (path for path in (shard / "logs").glob("proxy_*.jsonl")
             if not path.name.startswith("attempts_")), "request_log") if exists else (None, "missing_shard")
        attempt_path, attempt_path_error = _single(
            (shard / "logs").glob("attempts_proxy_*.jsonl"), "attempt_journal") if exists else (None, "missing_shard")
        quality, checker_error_type, score_error = common._score(score_path)
        request_rows, request_error = common._request_rows(request_path)
        attempts, attempt_error = common._attempt_summary(attempt_path)
        all_rows.extend(request_rows)
        cost = None
        cost_error = request_error
        if request_error is None:
            try:
                cost = common._selected_generation_cost(reqlog.summarize(request_rows))
            except (KeyError, TypeError, ValueError) as error:
                cost_error = f"{type(error).__name__}: {error}"
        outcome = outcomes.get(task_id)
        operational = {
            "outcome_record_known": outcome is not None,
            "outcome": outcome.get("operational_outcome") if outcome else None,
            "runner_returncode": outcome.get("runner_returncode") if outcome else None,
            "request_failure_count": len(outcome.get("request_failures") or []) if outcome else None,
            "infrastructure_reasons": outcome.get("infrastructure_reasons") or [] if outcome else [],
        }
        s0 = common._s0_task(s0_by_task[task_id])
        artifact_errors = [error for error in (
            score_path_error, request_path_error, attempt_path_error, score_error,
            request_error, attempt_error, cost_error,
        ) if error]
        if outcome is None:
            artifact_errors.append("missing_operational_outcome_record")
        artifact_errors = list(dict.fromkeys(artifact_errors))
        if artifact_errors:
            completion_errors.append({"task_id": task_id, "errors": artifact_errors})
        quality_record = {
            "official_score_known": quality is not None,
            **(quality or {"status": "missing", "correct_count": None,
                           "total_count": None, "accuracy": None}),
            "checker_error_type": checker_error_type,
            "score_parse_error": score_error,
        }
        per_task.append({
            "task_id": task_id,
            "cell_recovered": exists,
            "quality": quality_record,
            "operational": operational,
            "cost": {"status": "known" if cost is not None else "missing_or_invalid",
                     "parse_error": cost_error,
                     "task_wall_seconds": outcome.get("task_wall_seconds") if outcome else None,
                     **(cost or {})},
            "attempts": {**attempts, "parse_error": attempt_error},
            "source_coverage": common._coverage(request_rows),
            "subgoal_exposure": _subgoal_exposure(request_rows, candidate_id),
            "reencoding_and_coverage_cost": _reencoding_and_coverage_cost(request_rows),
            "s0_bounded_latest": s0,
            "paired_official_outcome_vs_s0": _pair(
                {"quality": quality_record}, {"quality": s0["quality"]},
                (candidate_id, "s0")),
            "artifact_errors": artifact_errors,
            "artifact_hashes": {
                "request_log": _sha256(request_path) if request_path else None,
                "attempt_journal": _sha256(attempt_path) if attempt_path else None,
                "official_score": _sha256(score_path) if score_path else None,
            },
        })

    recovered = [row for row in per_task if row["cell_recovered"]]
    scored = [row for row in per_task if row["quality"]["official_score_known"]]
    correct = sum(row["quality"]["correct_count"] for row in scored)
    operational_counts = Counter(row["operational"]["outcome"] or "missing"
                                 for row in per_task)
    s0_pair_counts = Counter(row["paired_official_outcome_vs_s0"] for row in per_task)
    paired_s0 = [row for row in per_task
                 if row["paired_official_outcome_vs_s0"] != "unpaired_missing_official"]
    candidate_complete = (
        manifest.get("status") == FINAL_STAGE_STATUS
        and len(manifest.get("task_outcomes") or []) == EXPECTED_TASKS
        and len(recovered) == EXPECTED_TASKS
        and len(scored) == EXPECTED_TASKS
        and not completion_errors
    )
    if mode == "final":
        reasons = []
        if manifest.get("status") != FINAL_STAGE_STATUS:
            reasons.append(f"stage status is {manifest.get('status')!r}")
        if len(manifest.get("task_outcomes") or []) != EXPECTED_TASKS:
            reasons.append("stage manifest does not contain 20 task outcomes")
        if len(recovered) != EXPECTED_TASKS:
            reasons.append(f"only {len(recovered)}/20 task shards are recovered")
        if len(scored) != EXPECTED_TASKS:
            reasons.append(f"only {len(scored)}/20 official score headers are known")
        if completion_errors:
            reasons.append("one or more task artifacts are missing or invalid")
        if reasons:
            raise S2LongAnalysisError(
                f"final mode requires complete {candidate_id} Long20: " + "; ".join(reasons))
    aggregate_cost = None
    aggregate_cost_error = None
    try:
        aggregate_cost = common._selected_generation_cost(reqlog.summarize(all_rows))
    except (KeyError, TypeError, ValueError) as error:
        aggregate_cost_error = f"{type(error).__name__}: {error}"
    return {
        "candidate_analysis_status": "complete" if candidate_complete else "partial",
        "manifest": manifest,
        "outcome_sources": outcome_sources,
        "per_task": per_task,
        "quality": {
            "fixed_task_denominator": EXPECTED_TASKS,
            "official_scored_tasks": len(scored),
            "official_missing_tasks": EXPECTED_TASKS - len(scored),
            "official_missing_task_ids": [row["task_id"] for row in per_task
                                          if not row["quality"]["official_score_known"]],
            "official_correct_tasks": correct,
            "official_incorrect_scored_tasks": len(scored) - correct,
            "official_successes_over_fixed_20": {
                "numerator": correct, "denominator": EXPECTED_TASKS,
                "rate": correct / EXPECTED_TASKS,
                "interpretation": ("final accuracy" if candidate_complete else
                                   "partial lower bound; not final accuracy"),
            },
            "official_accuracy_over_scored_only": _ratio(correct, len(scored)),
        },
        "operational": {
            "outcome_counts_over_fixed_20": dict(sorted(operational_counts.items())),
            "missing_outcome_records": sum(
                not row["operational"]["outcome_record_known"] for row in per_task),
        },
        "cost": {"status": "known" if aggregate_cost is not None else "missing_or_invalid",
                 "parse_error": aggregate_cost_error, **(aggregate_cost or {})},
        "source_coverage": common._coverage(all_rows),
        "subgoal_exposure": _subgoal_exposure(all_rows, candidate_id),
        "reencoding_and_coverage_cost": _reencoding_and_coverage_cost(all_rows),
        "paired_with_s0_bounded_latest": {
            "paired_task_count": len(paired_s0),
            "unpaired_task_count": EXPECTED_TASKS - len(paired_s0),
            "outcome_counts_over_fixed_20": dict(sorted(s0_pair_counts.items())),
            "candidate_correct_on_paired": sum(
                row["quality"]["correct_count"] for row in paired_s0),
            "s0_correct_on_paired": sum(
                row["s0_bounded_latest"]["quality"]["correct_count"]
                for row in paired_s0),
            "observed_accuracy_difference_candidate_minus_s0_on_paired": _ratio(
                sum(row["quality"]["correct_count"] for row in paired_s0)
                - sum(row["s0_bounded_latest"]["quality"]["correct_count"]
                      for row in paired_s0),
                len(paired_s0)),
            "scope": (
                "Same task IDs with independently generated trajectories; descriptive, "
                "not a controlled structure effect."
            ),
        },
        "recovery": {
            "recovered_task_count": len(recovered),
            "missing_task_count": EXPECTED_TASKS - len(recovered),
            "completion_errors": completion_errors,
        },
    }


def _difference(left: object, right: object) -> int | float | None:
    return left - right if _finite(left) and _finite(right) else None


def analyze(
    s2_manifest: Path,
    s2_shards_root: Path,
    control_manifest: Path,
    control_shards_root: Path,
    s0_root: Path,
    *,
    mode: str = "partial",
    s2_sidecar_root: Path | None = None,
    control_sidecar_root: Path | None = None,
    s2_observation: Path | None = None,
    control_observation: Path | None = None,
) -> dict[str, Any]:
    if mode not in {"partial", "final"}:
        raise S2LongAnalysisError("mode must be partial or final")
    s2_manifest_value = _read_json(s2_manifest)
    control_manifest_value = _read_json(control_manifest)
    task_ids = s2_manifest_value.get("task_ids")
    if (not isinstance(task_ids, list) or len(task_ids) != EXPECTED_TASKS
            or len(set(task_ids)) != EXPECTED_TASKS
            or not all(isinstance(task_id, str) for task_id in task_ids)
            or control_manifest_value.get("task_ids") != task_ids):
        raise S2LongAnalysisError("S2 manifests must contain the same 20 unique task IDs")
    try:
        _, _, _, s0_rows = common._s0_inputs(s0_root, task_ids)
    except common.S1LongAnalysisError as error:
        raise S2LongAnalysisError(str(error)) from error
    s0_by_task = {row["task_id"]: row for row in s0_rows}
    candidates = {
        S2: _candidate(
            S2, s2_manifest, s2_shards_root,
            s2_sidecar_root or Path(s2_shards_root).parent,
            s2_observation, task_ids, s0_by_task, mode),
        CONTROL: _candidate(
            CONTROL, control_manifest, control_shards_root,
            control_sidecar_root or Path(control_shards_root).parent,
            control_observation, task_ids, s0_by_task, mode),
    }
    s2_by_task = {row["task_id"]: row for row in candidates[S2]["per_task"]}
    control_by_task = {row["task_id"]: row for row in candidates[CONTROL]["per_task"]}
    paired_rows = []
    pair_counts = Counter()
    for task_id in task_ids:
        pair = _pair(s2_by_task[task_id], control_by_task[task_id], ("s2", "control"))
        pair_counts[pair] += 1
        paired_rows.append({
            "task_id": task_id,
            S2: s2_by_task[task_id],
            CONTROL: control_by_task[task_id],
            "paired_official_outcome_s2_vs_control": pair,
        })
    paired = [row for row in paired_rows
              if row["paired_official_outcome_s2_vs_control"]
              != "unpaired_missing_official"]
    s2_correct = sum(row[S2]["quality"]["correct_count"] for row in paired)
    control_correct = sum(row[CONTROL]["quality"]["correct_count"] for row in paired)
    cost_keys = (
        "lookups", "client_cache_hits", "producer_calls",
        "producer_source_occurrence_appearances",
        "reencoded_source_occurrence_appearances",
        "fitted_fragment_appearances", "retained_fragment_appearances",
        "fitted_encoder_input_token_appearances",
        "retained_encoder_input_token_appearances",
    )
    coverage_keys = (
        "eligible_source_occurrence_appearances",
        "fully_represented_source_occurrence_appearances",
        "unrepresented_source_occurrence_appearances",
    )
    paired_recode: dict[str, dict[str, int | float]] = {}
    paired_coverage: dict[str, dict[str, int | float | None]] = {}
    for candidate_id in (S2, CONTROL):
        paired_recode[candidate_id] = {
            key: sum(
                row[candidate_id]["reencoding_and_coverage_cost"].get(key, 0)
                for row in paired)
            for key in cost_keys
        }
        counts = {
            key: sum(row[candidate_id]["source_coverage"].get(key, 0)
                     for row in paired)
            for key in coverage_keys
        }
        counts["pooled_fully_represented_source_occurrence_fraction"] = _ratio(
            counts["fully_represented_source_occurrence_appearances"],
            counts["eligible_source_occurrence_appearances"],
        )
        paired_coverage[candidate_id] = counts
    result = {
        "schema": SCHEMA,
        "status": "complete" if mode == "final" else "partial",
        "mode": mode,
        "sample_label": (
            "preliminary, n=1; unfiltered historically exposed Long20 development tasks"
            if mode == "final" else
            "preliminary, n=1; partial in-flight Long20 development result"),
        "scope": {
            "cpu_only": True,
            "model_requests": 0,
            "remote_requests_during_analysis": 0,
            "scorer_calls": 0,
            "tool_executions": 0,
            "fixed_task_denominator": EXPECTED_TASKS,
            "official_and_operational_outcomes_separate": True,
            "official_quality_source": "existing per-task official score headers",
            "cost_parser": "benchmarks.reqlog.summarize with generation_costs",
        },
        "run_identity": {
            candidate_id: {
                "run_id": candidates[candidate_id]["manifest"].get("run_id"),
                "candidate_id": candidate_id,
                "stage_status": candidates[candidate_id]["manifest"].get("status"),
                "stage_manifest_sha256": _sha256(
                    s2_manifest if candidate_id == S2 else control_manifest),
                "design_sha256": candidates[candidate_id]["manifest"].get("design_sha256"),
                "task_ids": task_ids,
                "planned_tasks": EXPECTED_TASKS,
                "outcome_record_sources": candidates[candidate_id]["outcome_sources"],
            }
            for candidate_id in CANDIDATES
        },
        "candidates": {
            candidate_id: {
                key: value for key, value in candidates[candidate_id].items()
                if key not in {"manifest", "outcome_sources", "per_task"}
            }
            for candidate_id in CANDIDATES
        },
        "paired_s2_vs_matched_note_turn_control": {
            "paired_task_count": len(paired),
            "unpaired_task_count": EXPECTED_TASKS - len(paired),
            "outcome_counts_over_fixed_20": dict(sorted(pair_counts.items())),
            "s2_correct_on_paired": s2_correct,
            "control_correct_on_paired": control_correct,
            "observed_accuracy_difference_s2_minus_control_on_paired": _ratio(
                s2_correct - control_correct, len(paired)),
            "aggregate_reencoding_and_coverage_cost_difference_s2_minus_control": {
                key: _difference(
                    paired_recode[S2].get(key), paired_recode[CONTROL].get(key))
                for key in cost_keys
            },
            "aggregate_source_coverage_difference_s2_minus_control": {
                key: _difference(
                    paired_coverage[S2].get(key), paired_coverage[CONTROL].get(key))
                for key in (*coverage_keys,
                            "pooled_fully_represented_source_occurrence_fraction")
            },
            "scope": (
                "Matched prompt protocol and task IDs, with independently generated "
                "trajectories. This compares routes differing in history organization; "
                "it does not isolate a causal packing effect. Cost and coverage differences include paired "
                "officially scored tasks only; per-request denominators can diverge."
            ),
        },
        "per_task": paired_rows,
        "provenance": {
            "inputs": {
                "s2_stage_manifest": {"path": str(s2_manifest), "sha256": _sha256(s2_manifest)},
                "control_stage_manifest": {"path": str(control_manifest), "sha256": _sha256(control_manifest)},
                "s0_analysis": {"path": str(s0_root / "analysis.final.json"),
                                "sha256": _sha256(s0_root / "analysis.final.json")},
                "s0_recorded_cost": {"path": str(s0_root / "recorded_cost.final.json"),
                                     "sha256": _sha256(s0_root / "recorded_cost.final.json")},
                "s0_long_ratio_audit": {"path": str(s0_root / "long_ratio_audit.final.json"),
                                        "sha256": _sha256(s0_root / "long_ratio_audit.final.json")},
                **({"s2_observation": {"path": str(s2_observation),
                                       "sha256": _sha256(s2_observation)}}
                   if s2_observation else {}),
                **({"control_observation": {"path": str(control_observation),
                                            "sha256": _sha256(control_observation)}}
                   if control_observation else {}),
            },
            "parser_code": {
                "analyzer": {"path": str(Path(__file__)), "sha256": _sha256(Path(__file__))},
                "reused_s1_analyzer": {"path": str(Path(common.__file__)),
                                       "sha256": _sha256(Path(common.__file__))},
                "reqlog": {"path": str(BENCHMARKS_ROOT / "reqlog.py"),
                           "sha256": _sha256(BENCHMARKS_ROOT / "reqlog.py")},
                "generation_costs": {
                    "path": str(Path(__file__).with_name("generation_costs.py")),
                    "sha256": _sha256(Path(__file__).with_name("generation_costs.py")),
                },
            },
        },
        "limits": [
            "Partial mode retains the fixed 20-task denominator and does not estimate missing cells.",
            "Operational outcomes never replace available official score headers.",
            "Subgoal boundaries are observable declarations and do not prove semantic task completion.",
            "Reencoding source-occurrence counts use fragment provenance and are not exact token attribution.",
        ],
    }
    return result


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    try:
        common._write_new(path, value)
    except common.S1LongAnalysisError as error:
        raise S2LongAnalysisError(str(error)) from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s2-stage-manifest", required=True, type=Path)
    parser.add_argument("--s2-task-shards-root", required=True, type=Path)
    parser.add_argument("--s2-sidecar-root", type=Path)
    parser.add_argument("--s2-observation", type=Path)
    parser.add_argument("--control-stage-manifest", required=True, type=Path)
    parser.add_argument("--control-task-shards-root", required=True, type=Path)
    parser.add_argument("--control-sidecar-root", type=Path)
    parser.add_argument("--control-observation", type=Path)
    parser.add_argument("--s0-root", required=True, type=Path)
    parser.add_argument("--mode", choices=("partial", "final"), default="partial")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze(
            args.s2_stage_manifest, args.s2_task_shards_root,
            args.control_stage_manifest, args.control_task_shards_root,
            args.s0_root, mode=args.mode,
            s2_sidecar_root=args.s2_sidecar_root,
            control_sidecar_root=args.control_sidecar_root,
            s2_observation=args.s2_observation,
            control_observation=args.control_observation,
        )
        _write_new(args.out, result)
        print(json.dumps({
            "status": result["status"], "mode": result["mode"],
            "out": str(args.out),
            "s2_official_scored_tasks": result["candidates"][S2]["quality"]["official_scored_tasks"],
            "control_official_scored_tasks": result["candidates"][CONTROL]["quality"]["official_scored_tasks"],
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, S2LongAnalysisError,
            common.S1LongAnalysisError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
