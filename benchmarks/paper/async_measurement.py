"""Observed wall intervals for asynchronous history preparation in serving."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


def _map(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _duration(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _interval(start: Any, end: Any) -> tuple[int, int] | None:
    if type(start) is int and type(end) is int and 0 <= start <= end:
        return start, end
    return None


def _union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(end, merged[-1][1])
        else:
            merged.append((start, end))
    return merged


def _length(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _union(intervals))


def _intersection(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    left, right = _union(a), _union(b)
    i = j = total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _files(directory: Path, filename: str) -> list[Path]:
    return sorted({path.resolve() for path in directory.rglob(filename)
                   if path.parent.name == "server" and path.is_file()})


def _phase(values: list[Any]) -> dict[str, int | None]:
    observed = [_duration(value) for value in values]
    known = [value for value in observed if value is not None]
    return {"observed_count": len(known), "missing_count": len(values) - len(known),
            "sum_ns": sum(known) if known else None}


def _execution(trace: dict) -> dict:
    generation = _map(trace.get("generation"))
    stats = _map(generation.get("stats") or trace.get("stats"))
    return _map(_map(stats.get("native_telemetry")).get("execution_timing"))


def _lookahead_results(rows: list[dict]) -> list[dict]:
    seen = set()
    results = []
    for index, row in enumerate(rows):
        history = _map(row.get("history_lookahead"))
        for slot in ("poll", "during_generation"):
            result = _map(history.get(slot))
            if not result:
                continue
            submitted = _map(result.get("timing")).get("submitted_perf_ns")
            digest = result.get("source_prefix_sha256")
            key = ((digest, submitted) if isinstance(digest, str)
                   and type(submitted) is int else (index, slot))
            if key not in seen:
                seen.add(key)
                results.append(result)
    return results


def _receipts(record: dict) -> list[dict]:
    cross = _map(record.get("cross_turn_prewarm"))
    candidates = [cross.get("prior_receipt")]
    for field in ("session_cache_after", "session_cache_after_close"):
        cache = _map(record.get(field))
        for name in ("cross_turn_prewarm", "async_compression"):
            candidates.append(_map(cache.get(name)).get("last_receipt"))
    for trace in record.get("generation_trace", ()):
        if not isinstance(trace, dict):
            continue
        generation = _map(trace.get("generation"))
        stats = _map(generation.get("stats") or trace.get("stats"))
        candidates.append(_map(stats.get("async_compression")).get("last_receipt"))
    return [receipt for receipt in candidates
            if isinstance(receipt, dict)
            and receipt.get("schema") == "c2kv-native-prewarm-response-v1"
            and isinstance(receipt.get("job_id"), str)]


def _receipt_rank(receipt: dict) -> tuple[int, int, int]:
    return (len(receipt.get("results")) if isinstance(receipt.get("results"), list) else 0,
            _duration(receipt.get("completed_chunks")) or 0,
            int(receipt.get("status") in {"completed", "cancelled", "failed"}))


def summarize_async_compression(directory: Path) -> dict[str, Any]:
    """Aggregate measured intervals; absent timing remains unknown, never zero."""
    directory = Path(directory)
    step_files = _files(directory, "steps.jsonl")
    final_files = _files(directory, "final.json")
    issues = []

    def read_records(path: Path, *, lines: bool) -> list[dict]:
        try:
            contents = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            issues.append({"path": str(path), "error": type(error).__name__})
            return []
        records = []
        for index, text in enumerate(contents.splitlines() if lines else [contents], 1):
            if not text.strip():
                continue
            try:
                record = json.loads(text)
                if not isinstance(record, dict):
                    raise ValueError("record must be an object")
            except ValueError as error:
                issues.append({"path": str(path), "line": index,
                               "error": type(error).__name__})
            else:
                records.append(record)
        return records

    rows_by_server: dict[Path, list[dict]] = {}
    for path in step_files:
        rows_by_server[path.parent] = read_records(path, lines=True)
    steps = [row for rows in rows_by_server.values() for row in rows]
    finals = [row for path in final_files for row in read_records(path, lines=False)]
    results_by_server = {server: _lookahead_results(rows)
                         for server, rows in rows_by_server.items()}
    traces = [_map(trace) for row in steps for trace in row.get("generation_trace", ())
              if isinstance(trace, dict)]
    executions = [_execution(trace) for trace in traces]
    polls = [result for results in results_by_server.values() for result in results]
    phases = {
        "controller_prepare": _phase([_map(row.get("controller_timing")).get(
            "prepare_duration_ns") for row in steps]),
        "foreground_reconcile": _phase([_map(row.get("cross_turn_prewarm")).get(
            "foreground_reconcile_duration_ns") for row in steps]),
        "foreground_lookahead_hook": _phase([_map(row.get("history_lookahead")).get(
            "foreground_hook_duration_ns") for row in steps]),
        "cpu_lookahead_compute": _phase([_map(poll.get("timing")).get(
            "compute_duration_ns") for poll in polls if poll]),
        "engine_foreground_prewarm_wait": _phase([entry.get(
            "foreground_prewarm_wait_ns") for entry in executions]),
        "engine_selected_extraction_wall": _phase([entry.get(
            "selected_extraction_wall_ns") for entry in executions]),
        "engine_generation_wall": _phase([entry.get(
            "generation_wall_ns") for entry in executions]),
        "engine_extras_overlap_with_generation": _phase([entry.get(
            "extras_overlap_with_generation_ns") for entry in executions]),
        "engine_extras_tail_wait": _phase([entry.get(
            "extras_tail_wait_ns") for entry in executions]),
    }

    cpu_statuses: Counter[str] = Counter()
    cpu_compute_wall_ns_union = 0
    cpu_timed = 0
    cpu_source_identity_missing = 0
    cpu_overlap_ns = 0
    cpu_overlap_known = cpu_overlap_unknown = 0
    cpu_timing_missing = 0
    cpu_generation_known = cpu_generation_missing = 0
    cpu_submitted = 0
    for server, rows in rows_by_server.items():
        generation_intervals = []
        local_cpu_intervals = []
        for row in rows:
            for trace in row.get("generation_trace", ()):
                if not isinstance(trace, dict):
                    continue
                interval = _interval(trace.get("start_perf_ns"), trace.get("end_perf_ns"))
                if interval is None:
                    cpu_generation_missing += 1
                else:
                    cpu_generation_known += 1
                    generation_intervals.append(interval)
        cpu_submitted += sum(
            _map(_map(row.get("history_lookahead")).get("submission")).get("status") == "queued"
            for row in rows)
        for poll in results_by_server[server]:
            cpu_statuses[str(poll.get("status", "unknown"))] += 1
            if not isinstance(poll.get("source_prefix_sha256"), str):
                cpu_source_identity_missing += 1
            timing = _map(poll.get("timing"))
            interval = _interval(timing.get("started_perf_ns"),
                                 timing.get("finished_perf_ns"))
            if interval is None:
                cpu_timing_missing += 1
                continue
            local_cpu_intervals.append(interval)
            cpu_timed += 1
        cpu_compute_wall_ns_union += _length(local_cpu_intervals)
        if generation_intervals:
            cpu_overlap_ns += _intersection(local_cpu_intervals, generation_intervals)
            cpu_overlap_known += len(local_cpu_intervals)
        else:
            cpu_overlap_unknown += len(local_cpu_intervals)

    engine_generation: list[tuple[int, int]] = []
    engine_generation_missing = 0
    for execution in executions:
        interval = _interval(execution.get("generation_admitted_monotonic_ns"),
                             execution.get("generation_finished_monotonic_ns"))
        if interval is None:
            engine_generation_missing += 1
        else:
            engine_generation.append(interval)

    submissions = set()
    outstanding = set()
    receipts: dict[str, dict] = {}
    for record in [*steps, *finals]:
        cross = _map(record.get("cross_turn_prewarm"))
        lookahead = _map(record.get("history_lookahead"))
        for candidate in (cross.get("submission"), lookahead.get("offer")):
            job_id = _map(candidate).get("job_id")
            if isinstance(job_id, str):
                submissions.add(job_id)
        background_stats = []
        for field in ("session_cache_after", "session_cache_after_close"):
            background_stats.append(_map(_map(record.get(field)).get("async_compression")))
        for trace in record.get("generation_trace", ()):
            generation = _map(_map(trace).get("generation"))
            stats = _map(generation.get("stats") or _map(trace).get("stats"))
            background_stats.append(_map(stats.get("async_compression")))
        for stats in background_stats:
            job_id = stats.get("outstanding_job_id")
            if isinstance(job_id, str):
                outstanding.add(job_id)
            offer_id = _map(stats.get("provider_offer")).get("job_id")
            if isinstance(offer_id, str):
                submissions.add(offer_id)
        for receipt in _receipts(record):
            job_id = receipt["job_id"]
            if job_id not in receipts or _receipt_rank(receipt) > _receipt_rank(receipts[job_id]):
                receipts[job_id] = receipt
    jobs = submissions | outstanding | receipts.keys()
    job_statuses: Counter[str] = Counter()
    extraction_intervals: list[tuple[int, int]] = []
    extraction_missing = extraction_results = cache_hits = 0
    for job_id in jobs:
        receipt = receipts.get(job_id, {})
        status = receipt.get("status")
        job_statuses[status if status in {"completed", "cancelled", "failed"} else "pending"] += 1
        for result in receipt.get("results", ()) if isinstance(receipt.get("results"), list) else ():
            if not isinstance(result, dict):
                continue
            if result.get("cache_hit") is True:
                cache_hits += 1
                continue
            extraction_results += 1
            interval = _interval(result.get("extraction_started_monotonic_ns"),
                                 result.get("extraction_finished_monotonic_ns"))
            if interval is None:
                extraction_missing += 1
            else:
                extraction_intervals.append(interval)

    return {
        "schema": "paper-async-compression-measurement-v1",
        "source": str(directory),
        "coverage": {"step_file_count": len(step_files), "final_file_count": len(final_files),
                     "step_count": len(steps), "generation_trace_count": len(traces),
                     "read_issues": issues},
        "phase_duration_ns": phases,
        "cpu_lookahead": {
            "submitted_count": cpu_submitted, "polled_status_counts": dict(cpu_statuses),
            "timed_count": cpu_timed, "missing_timing_count": cpu_timing_missing,
            "missing_source_identity_count": cpu_source_identity_missing,
            "compute_wall_ns_union": (cpu_compute_wall_ns_union if cpu_timed else None),
            "generation_interval_count": cpu_generation_known,
            "generation_interval_missing_count": cpu_generation_missing,
            "overlap_known_count": cpu_overlap_known,
            "overlap_unknown_count": cpu_overlap_unknown,
            "overlap_with_same_process_generation_ns": (
                cpu_overlap_ns if cpu_overlap_known else None),
        },
        "engine_background": {
            "job_count": len(jobs), "submission_observed_count": len(submissions),
            "job_status_counts": dict(job_statuses),
            "deduplicated_receipt_count": len(receipts),
            "result_extraction_count": extraction_results,
            "cache_hit_result_count": cache_hits,
            "extraction_interval_count": len(extraction_intervals),
            "extraction_interval_missing_count": extraction_missing,
            "extraction_wall_ns_union": (
                _length(extraction_intervals) if extraction_intervals else None),
            "generation_interval_count": len(engine_generation),
            "generation_interval_missing_count": engine_generation_missing,
            "overlap_with_all_recorded_engine_generation_ns": (
                _intersection(extraction_intervals, engine_generation)
                if extraction_intervals and engine_generation else None),
        },
        "measurement_notes": [
            "Intervals measure wall overlap, not saved latency, GPU utilization, or GPU time.",
            "CPU perf-counter intervals are compared only within one server process.",
            "CPU overlap uses the complete native client call, including selected extraction and HTTP; engine generation intervals begin at admission.",
            "Engine monotonic intervals are compared across this cohort's recorded requests on the shared engine.",
            "Background extraction intervals include cache misses only; cache-hit results are counted separately.",
            "Missing timing and unrecorded engine requests limit coverage; observed sums may be partial.",
        ],
    }
