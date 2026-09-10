from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))

from memory_runtime.extraction_telemetry import (
    ExtractionBudget,
    ExtractionBudgetExceeded,
    capture_extractions,
    current_extraction_trace,
    extraction_sources,
    validate_extraction_budget_rows,
)


KEY = ("user", "content-sha", 4, "tools-sha")
RESULT = {"key_hash": "gist-a", "original_seq_len": 20, "gist_len": 5,
          "ignored_text": "must not be captured"}


def record(trace, key=KEY, result=RESULT, **overrides):
    values = dict(cache_hit=False, force=False, lookup_wall_sec=0.3,
                  producer_wall_sec=0.2, result=result, error_type=None)
    values.update(overrides)
    trace.record_lookup(key, **values)


def test_nested_capture_and_source_scopes_restore_context():
    assert current_extraction_trace() is None
    with capture_extractions() as outer:
        with extraction_sources([2, 3]):
            record(outer)
            with capture_extractions() as inner:
                assert current_extraction_trace() is inner
                with extraction_sources([9]):
                    record(inner)
            assert current_extraction_trace() is outer
        with capture_extractions(enabled=False) as disabled:
            assert disabled is None and current_extraction_trace() is None
    assert current_extraction_trace() is None
    assert outer.snapshot(block_refs=[], forwarded_requests=[], request_status="ok")[
        "events"][0]["source_indices"] == [2, 3]
    assert inner.snapshot(block_refs=[], forwarded_requests=[], request_status="ok")[
        "events"][0]["source_indices"] == [9]


def test_contextvar_trace_is_isolated_from_new_thread():
    def worker():
        assert current_extraction_trace() is None
        with capture_extractions() as child:
            record(child)
            return child.snapshot(block_refs=[], forwarded_requests=[], request_status="ok")

    with capture_extractions() as parent:
        with ThreadPoolExecutor(max_workers=1) as pool:
            child = pool.submit(worker).result()
        record(parent, cache_hit=True, producer_wall_sec=0)
    assert child["summary"]["producer_calls"] == 1
    assert parent.snapshot(block_refs=[], forwarded_requests=[], request_status="ok")[
        "summary"]["client_cache_hits"] == 1


def test_error_event_records_failure_without_error_text_or_result():
    with capture_extractions() as trace:
        with extraction_sources([4]):
            record(trace, result=None, error_type="RuntimeError")
    snapshot = trace.snapshot(block_refs=[], forwarded_requests=[], request_status="extract_error")
    event = snapshot["events"][0]
    assert event["producer_called"] is True
    assert event["producer_failed"] is True
    assert event["error_type"] == "RuntimeError"
    assert event["key_hash"] is None and event["source_indices"] == [4]
    assert snapshot["summary"]["producer_failures"] == 1
    assert "ignored_text" not in event


def test_empty_trace_and_disabled_capture_have_no_events():
    with capture_extractions(enabled=False) as trace:
        assert trace is None
        with extraction_sources([-1]):
            assert current_extraction_trace() is None
    with capture_extractions() as enabled:
        snapshot = enabled.snapshot(
            block_refs=[], forwarded_requests=[], request_status="ok")
    assert snapshot["events"] == []
    assert snapshot["summary"]["lookups"] == 0
    assert snapshot["summary"]["server_cache_hit"] is None
    assert snapshot["summary"]["actual_extraction_prefill_tokens"] is None


def test_snapshot_separates_retained_and_forwarded_and_is_detached():
    blocks = [{"key_hash": "gist-a", "source_indices": [1, 2], "gist_tokens": 5}]
    forwarded = [["gist-b", "gist-b"], []]
    with capture_extractions() as trace:
        record(trace)
        record(trace, key=("user", "other-sha", 4, "tools-sha"),
               result={"key_hash": "gist-b", "original_seq_len": 12, "gist_len": 3})
        snapshot = trace.snapshot(
            block_refs=blocks, forwarded_requests=forwarded, request_status="ok")
        record(trace, cache_hit=True, producer_wall_sec=0)
    blocks[0]["source_indices"].append(99)
    forwarded[0].append("gist-a")
    links = {item["key_hash"]: item for item in snapshot["key_links"]}
    assert links["gist-a"]["retained_block_indices"] == [0]
    assert links["gist-a"]["forwarded_occurrences"] == []
    assert links["gist-b"]["retained_block_indices"] == []
    assert len(links["gist-b"]["forwarded_occurrences"]) == 2
    assert len(snapshot["events"]) == 2
    assert snapshot["block_refs"][0]["source_indices"] == [1, 2]
    assert snapshot["forwarded_requests"] == [["gist-b", "gist-b"], []]
    assert snapshot["summary"]["producer_response_original_seq_len_sum"] == 32


def test_extraction_budget_reservations_are_atomic_and_fail_before_slot_33():
    budget = ExtractionBudget(32)

    def reserve_once():
        try:
            return budget.reserve()
        except ExtractionBudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: reserve_once(), range(64)))
    assert sorted(value for value in results if value is not None) == list(range(1, 33))
    assert results.count(None) == 32
    assert budget.consumed == 32


def _budget_row(before, indices, events):
    return {
        "status": "ok",
        "extraction_budget": {
            "limit": 3, "consumed_before": before,
            "consumed_after": before + len(indices), "attempt_indices": indices,
        },
        "extraction_telemetry": {
            "events": events,
            "summary": {"producer_calls": sum(
                event.get("producer_called") is True for event in events)},
        },
    }


def test_extraction_ledger_matches_producers_and_ignores_cache_hits():
    miss = {"producer_called": True, "client_cache_hit": False,
            "budget_attempt_index": 1}
    hit = {"producer_called": False, "client_cache_hit": True,
           "budget_attempt_index": None}
    failed = {"producer_called": True, "client_cache_hit": False,
              "producer_failed": True, "budget_attempt_index": 2}
    rows = [_budget_row(0, [1], [miss, hit]), _budget_row(1, [2], [failed])]
    assert validate_extraction_budget_rows(rows, 3) == 2

    wrong = _budget_row(0, [1], [{**hit, "budget_attempt_index": 1}])
    with pytest.raises(ValueError, match="telemetry disagrees|cache hits"):
        validate_extraction_budget_rows([wrong], 3)
