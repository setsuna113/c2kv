"""CPU-only contract checks for shared engine ledger aggregation."""

from __future__ import annotations

import json

import pytest

from benchmarks.paper.serving_measurement import aggregate_engine_ledger


def row(request_id, kind, start, end, *, pool_peak, pool_final,
        cache_hit=None, gist_ns=None, success=True, error=None,
        memory_scope=None):
    def snapshot(stamp, pool):
        return {
            "monotonic_ns": stamp,
            "kv": {
                "main_live_kv_tokens": pool,
                "c2kv_live_kv_tokens": 2,
                "resident_kv_tokens": pool + 2,
                "resident_kv_bytes": (pool + 2) * 4,
                "cached_evictable_kv_tokens": 1,
                "cached_protected_kv_tokens": 1,
            },
        }

    metrics = {}
    if kind == "c2kv_extract":
        metrics = {"cache_hit": cache_hit,
                   "gist_generation_duration_ns": gist_ns,
                   "extraction_duration_ns": end - start}
    if memory_scope is not None:
        metrics["memory_scope"] = memory_scope
    return {
        "schema_version": 1,
        "outer_request_id": "outer",
        "server_request_id": request_id,
        "kind": kind,
        "success": success,
        "error": error,
        "duration_ns": end - start,
        "metrics": metrics,
        "baseline": snapshot(start, 0),
        "peak": snapshot(start, pool_peak),
        "pooled_peak": snapshot(start, pool_peak),
        "final": snapshot(end, pool_final),
    }


def write_ledger(path, rows):
    path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")


def test_shared_ledger_counts_overlap_and_uses_observed_pool_maximum(tmp_path):
    path = tmp_path / "server_telemetry.jsonl"
    write_ledger(path, [
        row("gen-1", "generation", 10, 50, pool_peak=8, pool_final=4,
            memory_scope="process_shared_during_overlap"),
        row("gen-2", "generation", 20, 60, pool_peak=12, pool_final=7,
            memory_scope="process_shared_during_overlap"),
        row("gen-3", "generation", 60, 70, pool_peak=9, pool_final=5,
            memory_scope="request_single_flight"),
    ])
    result = aggregate_engine_ledger(path)
    assert result["status"] == "complete"
    assert result["request_count"] == result["generation_request_count"] == 3
    assert result["extraction_request_count"] == 0
    assert result["request_duration_ns_sum"] == 90  # overlapping request time
    assert result["timeline"]["observed_span_ns"] == 60
    assert result["timeline"]["max_overlapping_requests"] == 2
    assert result["ledger_scope"].startswith("Completed engine request records")
    assert result["recorded_memory_scopes"] == {
        "values": ["process_shared_during_overlap", "request_single_flight"],
        "records_with_value": 3, "record_count": 3,
    }
    assert result["pool_occupancy"]["resident_kv_tokens"] == {
        "sampled_max": 14, "last_observed_final": 7, "summary_snapshot_count": 12,
        "dedicated_peak_record_count": 0,
    }
    assert "not engine or GPU wall time" in result["measurement_notes"][0]


def test_cached_extraction_is_counted_but_its_nested_gist_duration_is_zero(tmp_path):
    path = tmp_path / "native_engine_telemetry.jsonl"
    write_ledger(path, [
        row("miss", "c2kv_extract", 100, 120, pool_peak=6, pool_final=3,
            cache_hit=False, gist_ns=12),
        row("hit", "c2kv_extract", 120, 125, pool_peak=5, pool_final=4,
            cache_hit=True, gist_ns=0),
        row("gen", "generation", 125, 150, pool_peak=7, pool_final=2),
    ])
    result = aggregate_engine_ledger(path)
    assert result["extraction_request_count"] == 2
    assert result["extraction_cache_hit_count"] == 1
    assert result["extraction_cache_miss_count"] == 1
    assert result["extraction_request_duration_ns_sum"] == 25
    assert result["gist_generation_duration_ns_sum"] == 12
    assert result["request_duration_ns_sum_by_kind"] == {
        "c2kv_extract": 25, "generation": 25,
    }


def test_cache_occupancy_uses_dedicated_sampled_peak_when_available(tmp_path):
    path = tmp_path / "server_telemetry.jsonl"
    event = row("gen", "generation", 1, 10, pool_peak=5, pool_final=2)
    event["metrics"]["cached_evictable_kv_peak_tokens"] = 9
    write_ledger(path, [event])
    result = aggregate_engine_ledger(path)
    assert result["pool_occupancy"]["cached_evictable_kv_tokens"] == {
        "sampled_max": 9, "last_observed_final": 1,
        "summary_snapshot_count": 4, "dedicated_peak_record_count": 1,
    }


def test_raw_prefix_counters_are_separate_from_gist_cache_and_show_coverage(tmp_path):
    path = tmp_path / "native_engine_telemetry.jsonl"
    first = row("first", "generation", 1, 5, pool_peak=8, pool_final=4)
    second = row("second", "generation", 6, 10, pool_peak=8, pool_final=4)
    third = row("unreported", "generation", 11, 15, pool_peak=8, pool_final=4)
    first["metrics"]["c2kv_raw_prefix_cache"] = {
        "status": "inserted", "hit_tokens": 0, "inserted_tokens": 4}
    second["metrics"]["c2kv_raw_prefix_cache"] = {
        "status": "hit", "hit_tokens": 4, "inserted_tokens": 0}
    write_ledger(path, [first, second, third])
    measured = aggregate_engine_ledger(path)["raw_prefix_cache"]
    assert measured["record_count"] == 2 and measured["generation_record_count"] == 3
    assert measured["recorded_hit_tokens_sum"] == 4
    assert measured["recorded_inserted_tokens_sum"] == 4
    assert measured["status_counts"] == {"inserted": 1, "hit": 1}
    del second["metrics"]["c2kv_raw_prefix_cache"]["hit_tokens"]
    write_ledger(path, [second])
    assert aggregate_engine_ledger(path)["raw_prefix_cache"]["recorded_hit_tokens_sum"] is None


@pytest.mark.parametrize("tail,reason", [
    ("{bad\n", "malformed_jsonl_line:2"),
    ('{"kind":"generation"', "unterminated_jsonl_line:2"),
    ("[]\n", "non_object_jsonl_line:2"),
])
def test_malformed_or_truncated_ledger_keeps_only_a_prefix_count(tmp_path, tail, reason):
    path = tmp_path / "server_telemetry.jsonl"
    write_ledger(path, [row("first", "generation", 1, 10,
                            pool_peak=3, pool_final=2)])
    with path.open("a", encoding="utf-8") as stream:
        stream.write(tail)
    result = aggregate_engine_ledger(path)
    assert result["status"] == "incomplete"
    assert result["reason"] == reason
    assert result["parsed_records"] == 1
    assert result["request_count"] is None
    assert result["request_duration_ns_sum"] is None
    assert result["pool_occupancy"] is None


def test_missing_and_empty_ledgers_have_no_invented_counts(tmp_path):
    path = tmp_path / "server_telemetry.jsonl"
    missing = aggregate_engine_ledger(path)
    assert missing["status"] == "unavailable"
    assert missing["reason"] == "ledger_missing"
    assert missing["request_count"] is None
    path.touch()
    empty = aggregate_engine_ledger(path)
    assert empty["status"] == "unavailable"
    assert empty["reason"] == "ledger_empty"
    assert empty["request_count"] is None


def test_missing_fields_are_null_with_specific_reasons(tmp_path):
    path = tmp_path / "server_telemetry.jsonl"
    incomplete = row("extract", "c2kv_extract", 1, 10,
                     pool_peak=4, pool_final=2, cache_hit=None, gist_ns=None)
    del incomplete["duration_ns"]
    del incomplete["final"]["monotonic_ns"]
    write_ledger(path, [incomplete])
    result = aggregate_engine_ledger(path)
    assert result["status"] == "complete"  # file integrity, not field coverage
    assert result["request_count"] == 1
    assert result["extraction_cache_hit_count"] is None
    assert result["unavailable_fields"]["extraction_cache_hit_count"] == "missing_or_invalid_cache_hit"
    assert result["request_duration_ns_sum"] is None
    assert result["unavailable_fields"]["request_duration_ns_sum"] == "missing_or_invalid_duration_ns"
    assert result["gist_generation_duration_ns_sum"] is None
    assert result["timeline"] is None
    assert result["unavailable_fields"]["timeline"] == "missing_or_invalid_monotonic_request_interval"


def test_superseded_request_is_visible_as_failed_engine_measurement(tmp_path):
    path = tmp_path / "server_telemetry.jsonl"
    write_ledger(path, [row("superseded", "generation", 1, 3,
                            pool_peak=2, pool_final=1, success=False,
                            error="superseded_by_next_request")])
    result = aggregate_engine_ledger(path)
    assert result["failed_request_count"] == 1
    assert result["superseded_request_count"] == 1
    assert result["request_attribution_status"] == "unusable_superseded_requests"
    assert any("unusable per-request attribution" in note
               for note in result["measurement_notes"])
