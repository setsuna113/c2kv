"""Aggregate one shared engine's paper telemetry ledger after a serving cell.

The engine writes one JSONL row per scheduler request. Its request durations
can overlap, and gist generation is inside an extraction request. None of the
sums below is GPU wall time. Pool peaks are maxima of recorded samples, not a
continuously measured high-water mark.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_POOL_FIELDS = (
    "main_live_kv_tokens",
    "c2kv_live_kv_tokens",
    "resident_kv_tokens",
    "resident_kv_bytes",
    "cached_evictable_kv_tokens",
    "cached_protected_kv_tokens",
)
_SNAPSHOTS = ("baseline", "peak", "pooled_peak", "final")
_LEDGER_SCOPE = "Completed engine request records; does not prove every submitted request was recorded."
_NOTES = (
    "Request durations may overlap; their sum is not engine or GPU wall time.",
    "Gist generation is nested inside extraction duration; do not add the two.",
    "Pool occupancy is process-shared during overlap; maxima cover recorded snapshots and peak fields, not a continuous true peak.",
)


def _nonnegative_int(value: Any) -> int | None:
    if type(value) is int and value >= 0:
        return value
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _unavailable(path: Path, reason: str, parsed_records: int | None) -> dict[str, Any]:
    return {
        "source": str(path),
        "status": "incomplete" if parsed_records is not None else "unavailable",
        "reason": reason,
        "ledger_scope": _LEDGER_SCOPE,
        "parsed_records": parsed_records,
        "request_count": None,
        "generation_request_count": None,
        "extraction_request_count": None,
        "repair_extraction_request_count": None,
        "failed_request_count": None,
        "superseded_request_count": None,
        "request_attribution_status": None,
        "recorded_memory_scopes": None,
        "raw_prefix_cache": None,
        "extraction_cache_hit_count": None,
        "extraction_cache_miss_count": None,
        "request_duration_ns_sum": None,
        "request_duration_ns_sum_by_kind": None,
        "extraction_request_duration_ns_sum": None,
        "gist_generation_duration_ns_sum": None,
        "timeline": None,
        "pool_occupancy": None,
        "unavailable_fields": {"all_aggregates": reason},
        "measurement_notes": list(_NOTES),
    }


def _timeline(rows: list[dict[str, Any]]) -> tuple[dict[str, int] | None, str | None]:
    intervals = []
    for row in rows:
        start = _nonnegative_int(_mapping(row.get("baseline")).get("monotonic_ns"))
        end = _nonnegative_int(_mapping(row.get("final")).get("monotonic_ns"))
        if start is None or end is None or end < start:
            return None, "missing_or_invalid_monotonic_request_interval"
        intervals.append((start, end))
    # End points precede starts at the same instant. Zero-length intervals do
    # not occupy time and cannot contribute to simultaneous active requests.
    events = []
    for start, end in intervals:
        if end > start:
            events.extend(((start, 1), (end, -1)))
    active = maximum = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        maximum = max(maximum, active)
    return {
        "first_start_monotonic_ns": min(start for start, _ in intervals),
        "last_end_monotonic_ns": max(end for _, end in intervals),
        "observed_span_ns": max(end for _, end in intervals) - min(start for start, _ in intervals),
        "max_overlapping_requests": maximum,
        "interval_count": len(intervals),
    }, None


def _pool_occupancy(rows: list[dict[str, Any]]) -> dict[str, dict[str, int | None]]:
    last = rows[-1]
    result = {}
    for field in _POOL_FIELDS:
        samples = [
            value
            for row in rows
            for snapshot in _SNAPSHOTS
            if (value := _nonnegative_int(
                _mapping(_mapping(row.get(snapshot)).get("kv")).get(field)
            )) is not None
        ]
        dedicated_peaks = []
        if field == "cached_evictable_kv_tokens":
            dedicated_peaks = [
                value for row in rows
                if (value := _nonnegative_int(_mapping(row.get("metrics")).get(
                    "cached_evictable_kv_peak_tokens"
                ))) is not None
            ]
        final = _nonnegative_int(_mapping(_mapping(last.get("final")).get("kv")).get(field))
        result[field] = {
            "sampled_max": max([*samples, *dedicated_peaks]) if samples or dedicated_peaks else None,
            "last_observed_final": final,
            "summary_snapshot_count": len(samples),
            "dedicated_peak_record_count": len(dedicated_peaks),
        }
    return result


def _raw_prefix_summary(rows):
    generations = [row for row in rows if row.get("kind") == "generation"]
    records = [item for row in generations
               if isinstance(item := _mapping(row.get("metrics")).get("c2kv_raw_prefix_cache"), dict)]
    totals = {}
    for field in ("hit_tokens", "inserted_tokens"):
        values = [_nonnegative_int(item.get(field)) for item in records]
        totals[f"recorded_{field}_sum"] = (
            sum(values) if values and all(value is not None for value in values) else None)
    statuses = {}
    for item in records:
        status = str(item.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    return {"record_count": len(records), "generation_record_count": len(generations),
            "status_counts": statuses, **totals,
            "scope": "Only records carrying native raw-prefix telemetry; not gist-cache hits or exclusive GPU work."}


def aggregate_engine_ledger(path: str | Path) -> dict[str, Any]:
    """Summarize one engine JSONL ledger without assuming single-flight execution.

    Missing, empty, malformed, or unterminated ledgers produce null aggregates.
    The number of successfully parsed prefix rows remains visible as evidence,
    but is never presented as the complete request count.
    """
    source = Path(path)
    if not source.is_file():
        return _unavailable(source, "ledger_missing", None)
    rows: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.endswith("\n"):
                    return _unavailable(source, f"unterminated_jsonl_line:{line_number}", len(rows))
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    return _unavailable(source, f"malformed_jsonl_line:{line_number}", len(rows))
                if not isinstance(row, dict):
                    return _unavailable(source, f"non_object_jsonl_line:{line_number}", len(rows))
                rows.append(row)
    except (OSError, UnicodeError) as error:
        return _unavailable(source, f"ledger_read_error:{type(error).__name__}", len(rows))
    if not rows:
        return _unavailable(source, "ledger_empty", None)

    missing: dict[str, str] = {}
    kinds = [row.get("kind") for row in rows]
    kinds_available = all(isinstance(kind, str) and bool(kind) for kind in kinds)
    if not kinds_available:
        for field in ("generation_request_count", "extraction_request_count",
                      "repair_extraction_request_count"):
            missing[field] = "missing_or_invalid_kind"
    extraction = [row for row in rows if row.get("kind") == "c2kv_extract"]
    durations = [_nonnegative_int(row.get("duration_ns")) for row in rows]
    if any(duration is None for duration in durations):
        for field in ("request_duration_ns_sum", "request_duration_ns_sum_by_kind",
                      "extraction_request_duration_ns_sum"):
            missing[field] = "missing_or_invalid_duration_ns"
    if not kinds_available:
        for field in ("request_duration_ns_sum_by_kind",
                      "extraction_request_duration_ns_sum"):
            missing[field] = "missing_or_invalid_kind"
    hits = [_mapping(row.get("metrics")).get("cache_hit") for row in extraction]
    if not kinds_available or any(type(hit) is not bool for hit in hits):
        for field in ("extraction_cache_hit_count", "extraction_cache_miss_count"):
            missing[field] = (
                "missing_or_invalid_kind" if not kinds_available else "missing_or_invalid_cache_hit"
            )
    gist = [
        _nonnegative_int(_mapping(row.get("metrics")).get("gist_generation_duration_ns"))
        for row in extraction
    ]
    if not kinds_available or any(duration is None for duration in gist):
        missing["gist_generation_duration_ns_sum"] = (
            "missing_or_invalid_kind" if not kinds_available
            else "missing_or_invalid_gist_generation_duration_ns"
        )
    timeline, timeline_error = _timeline(rows)
    if timeline_error:
        missing["timeline"] = timeline_error
    pool = _pool_occupancy(rows)
    for field, values in pool.items():
        if values["sampled_max"] is None:
            missing[f"pool_occupancy.{field}.sampled_max"] = "no_recorded_sample"
        if values["last_observed_final"] is None:
            missing[f"pool_occupancy.{field}.last_observed_final"] = "missing_last_final_sample"

    by_kind = None
    if kinds_available and all(duration is not None for duration in durations):
        by_kind = {}
        for kind, duration in zip(kinds, durations):
            by_kind[kind] = by_kind.get(kind, 0) + duration
    failures = [row.get("success") for row in rows]
    if any(type(success) is not bool for success in failures):
        missing["failed_request_count"] = "missing_or_invalid_success"
    if any("error" not in row or row["error"] is not None
           and not isinstance(row["error"], str) for row in rows):
        missing["superseded_request_count"] = "missing_or_invalid_error"
        missing["request_attribution_status"] = "missing_or_invalid_error"
    superseded = (
        sum(row["error"] == "superseded_by_next_request" for row in rows)
        if "superseded_request_count" not in missing else None
    )
    memory_scopes = [
        _mapping(row.get("metrics")).get("memory_scope") for row in rows
    ]
    observed_scopes = sorted({scope for scope in memory_scopes
                              if isinstance(scope, str) and scope})
    if not observed_scopes:
        missing["recorded_memory_scopes"] = "memory_scope_not_recorded"
    notes = list(_NOTES)
    if superseded:
        notes.append("Superseded requests have unusable per-request attribution; inspect their raw ledger rows.")

    return {
        "source": str(source),
        "status": "complete",
        "reason": None,
        "ledger_scope": _LEDGER_SCOPE,
        "parsed_records": len(rows),
        "request_count": len(rows),
        "raw_prefix_cache": _raw_prefix_summary(rows),
        "generation_request_count": kinds.count("generation") if kinds_available else None,
        "extraction_request_count": len(extraction) if kinds_available else None,
        "repair_extraction_request_count": kinds.count("c2kv_repair_extract") if kinds_available else None,
        "failed_request_count": failures.count(False) if "failed_request_count" not in missing else None,
        "superseded_request_count": superseded,
        "request_attribution_status": (
            "unusable_superseded_requests" if superseded else "no_superseded_requests_observed"
        ) if superseded is not None else None,
        "recorded_memory_scopes": (
            {"values": observed_scopes,
             "records_with_value": sum(isinstance(scope, str) and bool(scope)
                                       for scope in memory_scopes),
             "record_count": len(rows)}
            if observed_scopes else None
        ),
        "extraction_cache_hit_count": hits.count(True) if "extraction_cache_hit_count" not in missing else None,
        "extraction_cache_miss_count": hits.count(False) if "extraction_cache_miss_count" not in missing else None,
        "request_duration_ns_sum": sum(durations) if "request_duration_ns_sum" not in missing else None,
        "request_duration_ns_sum_by_kind": by_kind,
        "extraction_request_duration_ns_sum": (
            by_kind.get("c2kv_extract", 0) if by_kind is not None else None
        ),
        "gist_generation_duration_ns_sum": (
            sum(gist) if "gist_generation_duration_ns_sum" not in missing else None
        ),
        "timeline": timeline,
        "pool_occupancy": pool,
        "unavailable_fields": missing,
        "measurement_notes": notes,
    }
