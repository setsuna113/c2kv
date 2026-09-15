"""Analyze one recovered S1 Long20 stage without running models or scorers."""
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


SCHEMA = "a-s1-long20-recovered-analysis-v1"
VALIDATION_SCHEMA = "a-s1-long20-recovered-analysis-validation-v1"
STAGE_SCHEMA = "a-structure-long-candidate-stage-run-v1"
CANDIDATE = "s1_dependency_packet"
SPARE_RAW_CANDIDATE = "s1_dependency_packet_spare_raw"
SUPPORTED_CANDIDATES = (CANDIDATE, SPARE_RAW_CANDIDATE)
S0_ROUTE = "ac_native_needs_lexical_raw_reserve_failed_operation"
SCORE_NAME = "BFCL_v4_multi_turn_long_context_score.json"
EXPECTED_TASKS = 20
FINAL_STAGE_STATUS = "completed_fixed_structure_long_candidate"


class S1LongAnalysisError(RuntimeError):
    """Raised when recovered artifacts cannot support the requested mode."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise S1LongAnalysisError(f"Cannot read JSON {path}: {type(error).__name__}: {error}") from error
    if not isinstance(value, dict):
        raise S1LongAnalysisError(f"JSON is not an object: {path}")
    return value


def _json_objects(path: Path) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise S1LongAnalysisError(f"Cannot read JSONL {path}: {type(error).__name__}: {error}") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise S1LongAnalysisError(f"Malformed JSONL {path}:{number}: {error}") from error
        if not isinstance(value, dict):
            raise S1LongAnalysisError(f"JSONL row is not an object: {path}:{number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _single(paths: Iterable[Path], label: str) -> tuple[Path | None, str | None]:
    found = sorted(paths)
    if len(found) == 1:
        return found[0], None
    return None, f"expected_one_{label}_found_{len(found)}"


def _score(path: Path | None) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if path is None:
        return None, None, "missing_official_score"
    try:
        objects = _json_objects(path)
    except S1LongAnalysisError as error:
        return None, None, str(error)
    if not objects:
        return None, None, "official_score_has_no_json_objects"
    header = objects[0]
    if set(header) != {"accuracy", "correct_count", "total_count"}:
        return None, None, "official_score_header_schema_mismatch"
    correct, total, accuracy = header["correct_count"], header["total_count"], header["accuracy"]
    if type(correct) is not int or correct not in {0, 1} or total != 1 or not _finite_number(accuracy):
        return None, None, "official_score_header_value_mismatch"
    if not math.isclose(float(accuracy), correct / total, rel_tol=0, abs_tol=1e-12):
        return None, None, "official_score_accuracy_mismatch"
    error_type = None
    if len(objects) > 1 and isinstance(objects[1].get("error"), Mapping):
        raw = objects[1]["error"].get("error_type")
        error_type = str(raw) if raw is not None else None
    return {
        "status": "success" if correct == 1 else "failure",
        "correct_count": correct,
        "total_count": total,
        "accuracy": float(accuracy),
    }, error_type, None


def _request_rows(path: Path | None) -> tuple[list[dict[str, Any]], str | None]:
    if path is None:
        return [], "missing_request_log"
    try:
        strict = _json_objects(path)
    except S1LongAnalysisError as error:
        return [], str(error)
    parsed = reqlog.read_rows(path)
    if len(strict) != len(parsed):
        return [], "reqlog_parser_dropped_rows"
    return parsed, None


def _attempt_summary(path: Path | None) -> tuple[dict[str, Any], str | None]:
    if path is None:
        return {"event_count": 0, "counts": {}}, "missing_attempt_journal"
    try:
        rows = _json_objects(path)
    except S1LongAnalysisError as error:
        return {"event_count": 0, "counts": {}}, str(error)
    counts = Counter(f"{row.get('kind')}:{row.get('event')}:{row.get('status')}" for row in rows)
    return {"event_count": len(rows), "counts": dict(sorted(counts.items()))}, None


def _selected_generation_cost(summary: Mapping[str, Any]) -> dict[str, Any]:
    generation = summary.get("generation_costs") or {}
    resources = summary.get("generation_resources") or {}
    resource_fields = (
        "resource_scope", "server_allocator_scope", "extraction_scope",
        "active_history_bytes_max", "active_history_bytes_max_known_lower_bound",
        "evidence_bytes_max", "evidence_bytes_max_known_lower_bound",
        "gist_tokens_sum", "gist_tokens_sum_known_lower_bound",
        "gist_tokens_max", "gist_tokens_max_known_lower_bound",
        "controller_wall_seconds", "controller_wall_seconds_known_lower_bound",
        "compressed_assembly_wall_seconds", "compressed_assembly_wall_seconds_known_lower_bound",
        "extraction_lookups", "extraction_lookups_known_lower_bound",
        "extraction_client_cache_hits", "extraction_client_cache_hits_known_lower_bound",
        "extraction_producer_calls", "extraction_producer_calls_known_lower_bound",
        "extraction_producer_successes", "extraction_producer_successes_known_lower_bound",
        "extraction_producer_failures", "extraction_producer_failures_known_lower_bound",
        "extraction_lookup_wall_seconds", "extraction_lookup_wall_seconds_known_lower_bound",
        "extraction_producer_wall_seconds", "extraction_producer_wall_seconds_known_lower_bound",
        "kv_resident_tokens_max", "kv_resident_tokens_max_known_lower_bound",
        "kv_peak_resident_tokens_max", "kv_peak_resident_tokens_max_known_lower_bound",
        "total_gpu_kv_bytes_max", "total_gpu_kv_bytes_max_known_lower_bound",
        "peak_total_gpu_kv_bytes_max", "peak_total_gpu_kv_bytes_max_known_lower_bound",
    )
    return {
        "request_count": summary.get("n_requests"),
        "ok_request_count": summary.get("n_ok"),
        "error_request_count": summary.get("n_error"),
        "request_error_kinds": summary.get("error_kinds") or {},
        "proxy_request_wall_seconds": summary.get("wall_sec_total"),
        "generation": dict(generation),
        "resources": {key: resources.get(key) for key in resource_fields},
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    fractions: list[float] = []
    known = 0
    for row in rows:
        runtime = row.get("memory_runtime")
        receipt = runtime.get("source_coverage") if isinstance(runtime, Mapping) else None
        if not isinstance(receipt, Mapping):
            continue
        known += 1
        eligible = receipt.get("eligible_source_indices") or []
        unrepresented = receipt.get("unrepresented_source_indices") or []
        if not isinstance(eligible, list) or not isinstance(unrepresented, list):
            raise S1LongAnalysisError("source_coverage source indices must be lists")
        totals.update({
            "eligible_source_occurrence_appearances": len(eligible),
            "fully_represented_source_occurrence_appearances": len(eligible) - len(unrepresented),
            "unrepresented_source_occurrence_appearances": len(unrepresented),
            "gist_fully_represented_source_occurrence_appearances": len(receipt.get("gist_fully_represented_source_indices") or []),
            "raw_source_occurrence_appearances": len(receipt.get("raw_source_indices") or []),
            "fitted_fragment_appearances": receipt.get("fitted_fragment_count") or 0,
            "retained_fragment_appearances": receipt.get("retained_fragment_count") or 0,
            "fitted_encoder_input_tokens": receipt.get("fitted_encoder_input_tokens") or 0,
            "retained_encoder_input_tokens": receipt.get("retained_encoder_input_tokens") or 0,
        })
        if not eligible:
            totals["no_history_views"] += 1
        elif receipt.get("complete_history_coverage") is True:
            totals["full_coverage_history_views"] += 1
        else:
            totals["incomplete_coverage_history_views"] += 1
        fraction = receipt.get("fully_represented_source_fraction")
        if _finite_number(fraction):
            fractions.append(float(fraction))
    return {
        "known_receipts": known,
        "missing_receipts": len(rows) - known,
        **dict(totals),
        "pooled_fully_represented_source_occurrence_fraction": _ratio(
            totals["fully_represented_source_occurrence_appearances"],
            totals["eligible_source_occurrence_appearances"],
        ),
        "minimum_per_view_fully_represented_source_fraction": min(fractions) if fractions else None,
        "scope": "Source occurrence appearances across recorded actor requests; packet facts are credited only if the runtime coverage receipt credits them.",
    }


def _packet(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    roles = Counter()
    kinds = Counter()
    distinct_facts: set[str] = set()
    incremental_tokens: list[int] = []
    fit_tokens: list[int] = []
    packet_bytes: list[int] = []
    first = None
    versions, selectors, extractors, wires = set(), set(), set(), set()
    field_priority_policies = set()
    known = 0
    for ordinal, row in enumerate(rows):
        runtime = row.get("memory_runtime")
        receipt = runtime.get("dependency_packet") if isinstance(runtime, Mapping) else None
        if not isinstance(receipt, Mapping):
            continue
        known += 1
        if receipt.get("uses_gold_future_or_hidden_state") is not False:
            raise S1LongAnalysisError("dependency packet lacks a false gold/future/hidden-state receipt")
        fit = receipt.get("fit_receipt") or {}
        inc = receipt.get("incremental_raw_tokens")
        prompt = fit.get("prompt_tokens")
        byte_count = runtime.get("dependency_packet_bytes")
        if type(inc) is not int or inc < 0 or type(prompt) is not int or prompt < 0 or type(byte_count) is not int or byte_count < 0:
            raise S1LongAnalysisError("dependency packet cost receipt is malformed")
        incremental_tokens.append(inc)
        fit_tokens.append(prompt)
        packet_bytes.append(byte_count)
        if inc > 0:
            totals["requests_with_nonempty_packet"] += 1
            if first is None:
                context = row.get("eval_context") or {}
                first = {"request_ordinal": ordinal, "user_turn": context.get("user_turn"), "step": context.get("step")}
        facts = receipt.get("fact_provenance") or []
        omitted = receipt.get("omitted") or {}
        if not isinstance(facts, list) or not isinstance(omitted, Mapping):
            raise S1LongAnalysisError("dependency packet fact/omission receipt is malformed")
        totals.update({
            "retained_fact_appearances": len(facts),
            "eligible_fact_appearances": omitted.get("eligible_fact_count") or 0,
            "field_limit_omission_appearances": omitted.get("field_limit_omissions") or 0,
            "unrepresented_requested_source_appearances": len(omitted.get("unrepresented_requested_source_ids") or []),
            "workspace_drop_appearances": len(receipt.get("dropped_for_workspace") or []),
            "packet_counts_as_complete_event_coverage_true_requests": receipt.get("counts_as_complete_event_coverage") is True,
        })
        for fact in facts:
            if not isinstance(fact, Mapping):
                raise S1LongAnalysisError("dependency packet fact provenance is not an object")
            distinct_facts.add(hashlib.sha256(json.dumps(fact, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
            roles[str((fact.get("source") or {}).get("role"))] += 1
            kinds[str(fact.get("kind"))] += 1
        if receipt.get("version") is not None:
            versions.add(str(receipt["version"]))
        if receipt.get("selector") is not None:
            selectors.add(str(receipt["selector"]))
        if receipt.get("extractor_source") is not None:
            extractors.add(str(receipt["extractor_source"]))
        if fit.get("wire_form") is not None:
            wires.add(str(fit["wire_form"]))
        if "dependency_packet_field_priority_policy" in receipt:
            policy = receipt["dependency_packet_field_priority_policy"]
            if not isinstance(policy, str) or not policy:
                raise S1LongAnalysisError(
                    "dependency packet field priority policy receipt is malformed")
            field_priority_policies.add(policy)
            totals["requests_with_explicit_field_priority_policy"] += 1
        else:
            totals["requests_with_legacy_implicit_field_priority"] += 1
    return {
        "known_receipts": known,
        "missing_receipts": len(rows) - known,
        **dict(totals),
        "requests_with_explicit_field_priority_policy": totals[
            "requests_with_explicit_field_priority_policy"],
        "requests_with_legacy_implicit_field_priority": totals[
            "requests_with_legacy_implicit_field_priority"],
        "requests_without_nonempty_packet": known - totals["requests_with_nonempty_packet"],
        "distinct_full_fact_records": len(distinct_facts),
        "fact_source_role_appearance_counts": dict(sorted(roles.items())),
        "fact_kind_appearance_counts": dict(sorted(kinds.items())),
        "user_sourced_fact_appearances": roles["user"],
        "maximum_incremental_raw_tokens": max(incremental_tokens) if incremental_tokens else None,
        "maximum_fit_prompt_tokens": max(fit_tokens) if fit_tokens else None,
        "maximum_packet_bytes": max(packet_bytes) if packet_bytes else None,
        "first_nonempty_packet": first,
        "versions": sorted(versions),
        "selectors": sorted(selectors),
        "extractor_sources": sorted(extractors),
        "wire_forms": sorted(wires),
        "field_priority_policies": sorted(field_priority_policies),
        "field_priority_policy_receipt_scope": (
            "Actual per-request dependency-packet runtime receipts; absence denotes the "
            "legacy implicit ordering, not an inferred opt-in policy."
        ),
        "scope": "Counts are appearances across requests, not unique historical facts unless explicitly named distinct.",
    }


def _spare_raw(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    policies, versions = set(), set()
    incremental_tokens: list[int] = []
    incremental_bytes: list[int] = []
    active_before: list[int] = []
    active_after: list[int] = []
    first_change = None
    known = 0
    for ordinal, row in enumerate(rows):
        runtime = row.get("memory_runtime")
        receipt = (
            runtime.get("dependency_packet_spare_raw")
            if isinstance(runtime, Mapping) else None
        )
        if not isinstance(receipt, Mapping):
            continue
        known += 1
        if receipt.get("uses_gold_future_s0_or_hidden_state") is not False:
            raise S1LongAnalysisError(
                "dependency-packet spare-raw lacks a false gold/future/hidden-state receipt")
        if receipt.get("existing_gist_packet_state_raw_and_failed_cue_unchanged") is not True:
            raise S1LongAnalysisError(
                "dependency-packet spare-raw lacks the unchanged-existing-evidence receipt")
        policy = receipt.get("policy")
        if not isinstance(policy, str) or not policy:
            raise S1LongAnalysisError("dependency-packet spare-raw policy is malformed")
        policies.add(policy)
        totals["requests_with_explicit_policy"] += 1
        if receipt.get("version") is not None:
            versions.add(str(receipt["version"]))
        numeric = {
            "incremental_raw_tokens": receipt.get("incremental_raw_tokens"),
            "incremental_bytes": receipt.get("incremental_bytes"),
            "active_history_bytes_before": receipt.get("active_history_bytes_before"),
            "active_history_bytes_after": receipt.get("active_history_bytes_after"),
            "budget_bytes": receipt.get("budget_bytes"),
        }
        if not all(type(value) is int and value >= 0 for value in numeric.values()):
            raise S1LongAnalysisError("dependency-packet spare-raw byte receipt is malformed")
        if (numeric["active_history_bytes_after"]
                - numeric["active_history_bytes_before"]
                != numeric["incremental_bytes"]):
            raise S1LongAnalysisError("dependency-packet spare-raw active-byte delta is inconsistent")
        unit = runtime.get("bytes_per_kv_token")
        if (type(unit) is not int or unit <= 0
                or numeric["incremental_bytes"]
                != numeric["incremental_raw_tokens"] * unit):
            raise S1LongAnalysisError("dependency-packet spare-raw token/byte geometry is inconsistent")
        if numeric["active_history_bytes_after"] > numeric["budget_bytes"]:
            totals["requests_admitted_over_budget"] += 1
        incremental_tokens.append(numeric["incremental_raw_tokens"])
        incremental_bytes.append(numeric["incremental_bytes"])
        active_before.append(numeric["active_history_bytes_before"])
        active_after.append(numeric["active_history_bytes_after"])
        if numeric["incremental_bytes"] > 0:
            totals["requests_with_input_change"] += 1
            if first_change is None:
                context = row.get("eval_context") or {}
                first_change = {
                    "request_ordinal": ordinal,
                    "user_turn": context.get("user_turn"),
                    "step": context.get("step"),
                }
        items = receipt.get("items")
        if not isinstance(items, list):
            raise S1LongAnalysisError("dependency-packet spare-raw items are malformed")
        added_ids = receipt.get("added_event_ids")
        added_sources = receipt.get("added_source_indices")
        if not isinstance(added_ids, list) or not isinstance(added_sources, list):
            raise S1LongAnalysisError("dependency-packet spare-raw added-event ledger is malformed")
        totals["added_event_receipt_appearances"] += len(added_ids)
        totals["added_source_occurrence_appearances"] += len(added_sources)
        item_statuses = Counter()
        for item in items:
            if not isinstance(item, Mapping):
                raise S1LongAnalysisError("dependency-packet spare-raw item is not an object")
            status = item.get("status")
            if status not in {"added", "already_raw", "over_budget"}:
                raise S1LongAnalysisError("dependency-packet spare-raw item status is unknown")
            item_statuses[status] += 1
            totals[f"event_attempts_{status}"] += 1
            candidate_bytes = item.get("candidate_active_history_bytes")
            if type(candidate_bytes) is not int or candidate_bytes < 0:
                raise S1LongAnalysisError(
                    "dependency-packet spare-raw candidate bytes are malformed")
            if status == "added" and candidate_bytes > numeric["budget_bytes"]:
                totals["added_event_attempts_over_budget"] += 1
            if status == "over_budget" and candidate_bytes <= numeric["budget_bytes"]:
                raise S1LongAnalysisError(
                    "dependency-packet spare-raw over-budget item fits its receipt budget")
        if item_statuses["added"] != len(added_ids):
            raise S1LongAnalysisError(
                "dependency-packet spare-raw added-event ledger disagrees with added items")
    return {
        "known_receipts": known,
        "missing_receipts": len(rows) - known,
        **dict(totals),
        "requests_with_explicit_policy": totals["requests_with_explicit_policy"],
        "requests_with_input_change": totals["requests_with_input_change"],
        "event_attempts_added": totals["event_attempts_added"],
        "event_attempts_already_raw": totals["event_attempts_already_raw"],
        "event_attempts_over_budget": totals["event_attempts_over_budget"],
        "requests_admitted_over_budget": totals["requests_admitted_over_budget"],
        "added_event_attempts_over_budget": totals["added_event_attempts_over_budget"],
        "incremental_raw_tokens_sum": sum(incremental_tokens),
        "incremental_bytes_sum": sum(incremental_bytes),
        "maximum_incremental_raw_tokens": max(incremental_tokens) if incremental_tokens else None,
        "maximum_incremental_bytes": max(incremental_bytes) if incremental_bytes else None,
        "maximum_active_history_bytes_before": max(active_before) if active_before else None,
        "maximum_active_history_bytes_after": max(active_after) if active_after else None,
        "policies": sorted(policies),
        "versions": sorted(versions),
        "first_input_change": first_change,
        "scope": (
            "Actual per-request dependency_packet_spare_raw receipts. Event and source "
            "counts are appearances across independently generated request views."
        ),
    }


def _outcome_records(manifest: Mapping[str, Any], sidecar_root: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    sources: list[str] = []
    raw = manifest.get("task_outcomes") or []
    if not isinstance(raw, list):
        raise S1LongAnalysisError("stage_manifest.task_outcomes is not a list")
    for row in raw:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
            raise S1LongAnalysisError("stage_manifest has a malformed task outcome")
        if row["task_id"] in records:
            raise S1LongAnalysisError(f"duplicate manifest outcome for {row['task_id']}")
        records[row["task_id"]] = row
    if raw:
        sources.append("stage_manifest.task_outcomes")
    for path in sorted(Path(sidecar_root).glob("task*_outcome.json")):
        row = _read_json(path)
        task_id = row.get("task_id")
        if not isinstance(task_id, str):
            raise S1LongAnalysisError(f"sidecar outcome lacks task_id: {path}")
        if task_id not in records:
            records[task_id] = row
            sources.append(path.as_posix())
    return records, sources


def _s0_inputs(root: Path, task_ids: list[str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    analysis = _read_json(root / "analysis.final.json")
    costs = _read_json(root / "recorded_cost.final.json")
    ratios = _read_json(root / "long_ratio_audit.final.json")
    cells = [row for row in analysis.get("cell_classifications") or [] if row.get("variant") == S0_ROUTE]
    by_task = {row.get("task_id"): row for row in cells}
    if len(by_task) != EXPECTED_TASKS or set(by_task) != set(task_ids):
        raise S1LongAnalysisError("S0 task set does not match the recovered S1 Long20 manifest")
    cost_by_task = {row.get("task_id"): row for row in costs.get("per_cell") or [] if row.get("variant") == S0_ROUTE}
    ratio_tasks = ratios.get("tasks") or {}
    if set(cost_by_task) != set(task_ids) or not set(task_ids) <= set(ratio_tasks):
        raise S1LongAnalysisError("S0 cost or coverage rows do not cover the Long20 task set")
    return analysis, costs, ratios, [
        {
            "task_id": task_id,
            "cell": by_task[task_id],
            "cost": cost_by_task[task_id],
            "coverage": ratio_tasks[task_id][S0_ROUTE]["source_coverage"],
        }
        for task_id in task_ids
    ]


def _s0_task(row: Mapping[str, Any]) -> dict[str, Any]:
    cell, cost, coverage = row["cell"], row["cost"], row["coverage"]
    header = cell.get("official_score_header") if cell.get("official_score_known") else None
    generation = (cost.get("generation_and_proxy_request") or {}).get("known_usage_tokens") or {}
    extraction = cost.get("extraction") or {}
    active = cost.get("active_history") or {}
    return {
        "quality": {
            "official_score_known": header is not None,
            "status": ("success" if header and header.get("correct_count") == 1 else "failure") if header else "missing",
            "correct_count": header.get("correct_count") if header else None,
            "total_count": header.get("total_count") if header else None,
        },
        "operational": {"outcome": cell.get("operational_outcome"), "runner_returncode": cell.get("runner_returncode")},
        "cost": {
            "request_count": cell.get("request_count"),
            "task_wall_seconds": cell.get("task_wall_seconds"),
            "generation_prompt_tokens": generation.get("prompt_tokens"),
            "generation_completion_tokens": generation.get("completion_tokens"),
            "generation_attempts": cell.get("generation_attempts_started"),
            "extraction_producer_calls": (extraction.get("recorded_totals") or {}).get("producer_calls"),
            "maximum_active_history_bytes": active.get("maximum_recorded_active_history_bytes"),
        },
        "coverage": {
            "history_views": coverage.get("views_with_history"),
            "eligible_source_occurrences": coverage.get("eligible_source_occurrences"),
            "fully_represented_source_occurrences": coverage.get("fully_represented_source_occurrences"),
            "fully_represented_source_occurrence_fraction": coverage.get("fully_represented_source_occurrence_fraction"),
            "unrepresented_source_occurrences": coverage.get("unrepresented_source_occurrences"),
        },
    }


def analyze(stage_manifest: Path, task_shards_root: Path, s0_root: Path, *, mode: str = "partial", sidecar_root: Path | None = None, candidate_id: str = CANDIDATE) -> dict[str, Any]:
    if mode not in {"partial", "final"}:
        raise S1LongAnalysisError("mode must be partial or final")
    if candidate_id not in SUPPORTED_CANDIDATES:
        raise S1LongAnalysisError(f"unsupported S1 candidate_id: {candidate_id}")
    manifest = _read_json(stage_manifest)
    if manifest.get("schema") != STAGE_SCHEMA or manifest.get("candidate_id") != candidate_id:
        raise S1LongAnalysisError("stage manifest is not the frozen S1 Long20 candidate")
    task_ids = manifest.get("task_ids")
    if not isinstance(task_ids, list) or len(task_ids) != EXPECTED_TASKS or len(set(task_ids)) != EXPECTED_TASKS or not all(isinstance(x, str) for x in task_ids):
        raise S1LongAnalysisError("stage manifest must contain 20 unique task IDs")
    _, _, _, s0_rows = _s0_inputs(s0_root, task_ids)
    s0_by_task = {row["task_id"]: row for row in s0_rows}
    outcomes, outcome_sources = _outcome_records(manifest, sidecar_root or Path(task_shards_root).parent)

    per_task: list[dict[str, Any]] = []
    completion_errors: list[dict[str, Any]] = []
    for task_id in task_ids:
        shard = Path(task_shards_root) / task_id / candidate_id
        shard_exists = shard.is_dir()
        score_path, score_path_error = _single((shard / "score").rglob(SCORE_NAME), "official_score") if shard_exists else (None, "missing_shard")
        request_path, request_path_error = _single((p for p in (shard / "logs").glob("proxy_*.jsonl") if not p.name.startswith("attempts_")), "request_log") if shard_exists else (None, "missing_shard")
        attempt_path, attempt_path_error = _single((shard / "logs").glob("attempts_proxy_*.jsonl"), "attempt_journal") if shard_exists else (None, "missing_shard")
        quality, checker_error_type, score_error = _score(score_path)
        request_rows, request_error = _request_rows(request_path)
        attempts, attempt_error = _attempt_summary(attempt_path)
        cost, cost_error = None, None
        if request_error is None:
            try:
                cost = _selected_generation_cost(reqlog.summarize(request_rows))
            except (KeyError, TypeError, ValueError) as error:
                cost_error = f"{type(error).__name__}: {error}"
        else:
            cost_error = request_error
        outcome = outcomes.get(task_id)
        operational = {
            "outcome_record_known": outcome is not None,
            "outcome": outcome.get("operational_outcome") if outcome else None,
            "runner_returncode": outcome.get("runner_returncode") if outcome else None,
            "request_failure_count": len(outcome.get("request_failures") or []) if outcome else None,
            "infrastructure_reasons": outcome.get("infrastructure_reasons") or [] if outcome else [],
        }
        s0 = _s0_task(s0_by_task[task_id])
        if quality is None or not s0["quality"]["official_score_known"]:
            pair = "unpaired_missing_official"
        elif quality["correct_count"] == 1 and s0["quality"]["correct_count"] == 1:
            pair = "both_correct"
        elif quality["correct_count"] == 1:
            pair = "s1_only_correct"
        elif s0["quality"]["correct_count"] == 1:
            pair = "s0_only_correct"
        else:
            pair = "both_incorrect"
        artifact_errors = [x for x in (score_path_error, request_path_error, attempt_path_error, score_error, request_error, attempt_error, cost_error) if x]
        if outcome is None:
            artifact_errors.append("missing_operational_outcome_record")
        if artifact_errors:
            completion_errors.append({
                "task_id": task_id,
                "errors": list(dict.fromkeys(artifact_errors)),
            })
        per_task.append({
            "task_id": task_id,
            "cell_recovered": shard_exists,
            "quality": {"official_score_known": quality is not None, **(quality or {"status": "missing", "correct_count": None, "total_count": None, "accuracy": None}), "checker_error_type": checker_error_type, "score_parse_error": score_error},
            "operational": operational,
            "cost": {"status": "known" if cost is not None else "missing_or_invalid", "parse_error": cost_error, "task_wall_seconds": outcome.get("task_wall_seconds") if outcome else None, **(cost or {})},
            "attempts": {**attempts, "parse_error": attempt_error},
            "source_coverage": _coverage(request_rows),
            "dependency_packet_exposure": _packet(request_rows),
            "dependency_packet_spare_raw_exposure": _spare_raw(request_rows),
            "s0_bounded_latest": s0,
            "paired_official_outcome": pair,
            "artifact_errors": list(dict.fromkeys(artifact_errors)),
            "artifact_hashes": {
                "request_log": _sha256(request_path) if request_path else None,
                "attempt_journal": _sha256(attempt_path) if attempt_path else None,
                "official_score": _sha256(score_path) if score_path else None,
            },
        })

    recovered = [row for row in per_task if row["cell_recovered"]]
    scored = [row for row in per_task if row["quality"]["official_score_known"]]
    correct = sum(row["quality"]["correct_count"] for row in scored)
    missing = [row["task_id"] for row in per_task if not row["quality"]["official_score_known"]]
    operational_counts = Counter(row["operational"]["outcome"] or "missing" for row in per_task)
    pair_counts = Counter(row["paired_official_outcome"] for row in per_task)
    paired = [row for row in per_task if row["paired_official_outcome"] != "unpaired_missing_official"]
    s1_correct_on_paired = sum(row["quality"]["correct_count"] for row in paired)
    s0_correct = sum(row["s0_bounded_latest"]["quality"]["correct_count"] for row in paired)
    if mode == "final":
        final_reasons = []
        if manifest.get("status") != FINAL_STAGE_STATUS:
            final_reasons.append(f"stage status is {manifest.get('status')!r}, not {FINAL_STAGE_STATUS!r}")
        if len(manifest.get("task_outcomes") or []) != EXPECTED_TASKS:
            final_reasons.append("stage manifest does not contain 20 task outcomes")
        if len(recovered) != EXPECTED_TASKS:
            final_reasons.append(f"only {len(recovered)}/20 task shards are recovered")
        if len(scored) != EXPECTED_TASKS:
            final_reasons.append(f"only {len(scored)}/20 official score headers are known")
        if completion_errors:
            final_reasons.append("one or more task artifacts are missing or invalid")
        if final_reasons:
            raise S1LongAnalysisError("final mode requires a complete 20-task recovery: " + "; ".join(final_reasons))

    fixed_rate = correct / EXPECTED_TASKS
    result = {
        "schema": SCHEMA,
        "status": "complete" if mode == "final" else "partial",
        "mode": mode,
        "sample_label": "preliminary, n=1; unfiltered historically exposed Long20 development tasks" if mode == "final" else "preliminary, n=1; partial in-flight Long20 development result",
        "scope": {
            "cpu_only": True, "model_requests": 0, "remote_requests": 0, "scorer_calls": 0, "tool_executions": 0,
            "task_selection_changed": False, "configuration_changed": False,
            "official_quality_source": "existing per-task official score headers; scorer detail is used only for its error type",
            "cost_parser": "benchmarks.reqlog.summarize using benchmarks.memory_runtime.generation_costs",
        },
        "run_identity": {
            "run_id": manifest.get("run_id"), "candidate_id": candidate_id, "stage_status": manifest.get("status"),
            "stage_manifest_sha256": _sha256(stage_manifest), "design_sha256": manifest.get("design_sha256"),
            "task_ids": task_ids, "planned_tasks": EXPECTED_TASKS, "outcome_record_sources": outcome_sources,
        },
        "quality": {
            "fixed_task_denominator": EXPECTED_TASKS,
            "official_scored_tasks": len(scored), "official_missing_tasks": len(missing), "official_missing_task_ids": missing,
            "official_correct_tasks": correct, "official_incorrect_scored_tasks": len(scored) - correct,
            "official_successes_over_fixed_20": {"numerator": correct, "denominator": EXPECTED_TASKS, "rate": fixed_rate,
                "interpretation": "final accuracy" if mode == "final" else "partial lower bound; not a final accuracy while official cells are missing"},
            "official_accuracy_over_scored_only": _ratio(correct, len(scored)),
        },
        "operational": {"outcome_counts_over_fixed_20": dict(sorted(operational_counts.items())), "missing_outcome_records": sum(not row["operational"]["outcome_record_known"] for row in per_task)},
        "paired_with_s0_bounded_latest": {
            "paired_task_count": len(paired), "unpaired_task_count": EXPECTED_TASKS - len(paired),
            "outcome_counts_over_fixed_20": dict(sorted(pair_counts.items())),
            "s1_correct_on_paired": s1_correct_on_paired, "s0_correct_on_paired": s0_correct,
            "observed_accuracy_difference_s1_minus_s0_on_paired": _ratio(s1_correct_on_paired - s0_correct, len(paired)),
            "scope": "Same task IDs, but independently generated trajectories. Cost and coverage denominators vary with trajectory; this is descriptive, not a controlled packet effect.",
        },
        "recovery": {"recovered_task_count": len(recovered), "missing_task_count": EXPECTED_TASKS - len(recovered), "completion_errors": completion_errors},
        "per_task": per_task,
        "provenance": {
            "inputs": {
                "stage_manifest": {"path": str(stage_manifest), "sha256": _sha256(stage_manifest)},
                "s0_analysis": {"path": str(s0_root / 'analysis.final.json'), "sha256": _sha256(s0_root / 'analysis.final.json')},
                "s0_recorded_cost": {"path": str(s0_root / 'recorded_cost.final.json'), "sha256": _sha256(s0_root / 'recorded_cost.final.json')},
                "s0_long_ratio_audit": {"path": str(s0_root / 'long_ratio_audit.final.json'), "sha256": _sha256(s0_root / 'long_ratio_audit.final.json')},
            },
            "parser_code": {
                "analyzer": {"path": str(Path(__file__)), "sha256": _sha256(Path(__file__))},
                "reqlog": {"path": str(BENCHMARKS_ROOT / 'reqlog.py'), "sha256": _sha256(BENCHMARKS_ROOT / 'reqlog.py')},
                "generation_costs": {"path": str(Path(__file__).with_name('generation_costs.py')), "sha256": _sha256(Path(__file__).with_name('generation_costs.py'))},
            },
        },
        "limits": [
            "Partial mode preserves all 20 planned task rows and marks missing cells; it does not estimate the unfinished cells.",
            "Operational outcomes never replace an available official score header. Capacity and postprocess failures remain separately visible.",
            "No exact packet values, user source text, tool arguments, or checker ground-truth state are copied into this analysis.",
        ],
    }
    return result


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-manifest", required=True, type=Path)
    parser.add_argument("--task-shards-root", required=True, type=Path)
    parser.add_argument("--s0-root", required=True, type=Path)
    parser.add_argument("--sidecar-root", type=Path)
    parser.add_argument("--candidate-id", choices=SUPPORTED_CANDIDATES, default=CANDIDATE)
    parser.add_argument("--mode", choices=("partial", "final"), default="partial")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = analyze(args.stage_manifest, args.task_shards_root, args.s0_root, mode=args.mode, sidecar_root=args.sidecar_root, candidate_id=args.candidate_id)
        _write_new(args.out, result)
        print(json.dumps({"status": result["status"], "mode": result["mode"], "out": str(args.out), "official_scored_tasks": result["quality"]["official_scored_tasks"], "official_missing_tasks": result["quality"]["official_missing_tasks"]}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, S1LongAnalysisError) as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
