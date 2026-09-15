"""Synthetic and captured-trace checks for cache allocation lifetime profiling."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "python"))

from history_memory.cache_trace import CACHE_MEMORY_RANGE_PREFIX
from memory_runtime.cache_memory_profile import summarize_cache_memory_profile


def _operation(op_id: str, *, annotated: bool = True) -> dict:
    row = {
        "attempt_uid": "attempt-a",
        "session_id": "session-a",
        "decision_key": "decision-a",
        "phase": "draft",
        "op_id": op_id,
        "kind": "extract",
        "status": "completed",
    }
    if annotated:
        row["memory_profile_annotation"] = CACHE_MEMORY_RANGE_PREFIX + op_id
    return row


def _range(op_id: str, start: float, duration: float, *, pid: int = 1, tid: int = 2) -> dict:
    return {
        "ph": "X",
        "name": CACHE_MEMORY_RANGE_PREFIX + op_id,
        "ts": start,
        "dur": duration,
        "pid": pid,
        "tid": tid,
    }


def _memory(
    event_index: int,
    timestamp: float,
    byte_delta: int,
    address: int,
    *,
    pid: int = 1,
    tid: int = 2,
    device_type: int = 0,
    device_id: int = -1,
) -> dict:
    return {
        "ph": "i",
        "name": "[memory]",
        "ts": timestamp,
        "pid": pid,
        "tid": tid,
        "args": {
            "Ev Idx": event_index,
            "Addr": address,
            "Bytes": byte_delta,
            "Device Type": device_type,
            "Device Id": device_id,
            "Total Allocated": 0,
            "Total Reserved": 0,
        },
    }


def _trace(*events: dict) -> dict:
    return {"traceEvents": list(events)}


def _device(profile: dict, operation_index: int = 0, device_index: int = 0) -> dict:
    return profile["operations"][operation_index]["devices"][device_index]


def test_alloc_free_net_zero_still_reports_lifetime_peak() -> None:
    profile = summarize_cache_memory_profile(
        _trace(
            _range("extract-1", 0, 10),
            _memory(1, 1, 64, 100),
            _memory(2, 3, -64, 100),
        ),
        operations=[_operation("extract-1")],
    )

    device = _device(profile)
    assert device["allocated_bytes_total"] == 64
    assert device["new_allocation_live_peak_bytes"] == 64
    assert device["new_allocations_live_at_range_exit_bytes"] == 0
    assert device["temporary_subset_live_peak_bytes"] == 64
    assert profile["global_memory_events"]["allocated_bytes_total"] == 64
    assert profile["global_memory_events"]["freed_bytes_total"] == 64


def test_nested_ranges_assign_once_to_the_unique_innermost_operation() -> None:
    profile = summarize_cache_memory_profile(
        _trace(
            _range("parent", 0, 20),
            _range("child", 2, 6),
            _memory(1, 3, 48, 100),
            _memory(2, 4, -48, 100),
        ),
        operations=[_operation("parent"), _operation("child")],
    )

    assert profile["operations"][0]["devices"] == []
    child = _device(profile, 1)
    assert child["allocated_bytes_total"] == 48
    assert child["new_allocation_live_peak_bytes"] == 48
    assert profile["global_memory_events"]["unassigned_allocation_events"] == 0


def test_cross_thread_free_matches_the_lifetime_but_is_not_temporary() -> None:
    profile = summarize_cache_memory_profile(
        _trace(
            _range("extract-1", 0, 10, tid=2),
            _memory(1, 2, 80, 100, tid=2),
            _memory(2, 12, -80, 100, tid=9),
        ),
        operations=[_operation("extract-1")],
    )

    device = _device(profile)
    assert device["matched_free_events"] == 1
    assert device["new_allocation_live_peak_bytes"] == 80
    assert device["new_allocations_live_at_range_exit_bytes"] == 80
    assert device["temporary_subset_live_peak_bytes"] == 0
    assert device["right_censored"] is False


def test_cross_thread_free_inside_the_range_is_a_temporary_lifetime() -> None:
    profile = summarize_cache_memory_profile(
        _trace(
            _range("extract-1", 0, 10, tid=2),
            _memory(1, 2, 80, 100, tid=2),
            _memory(2, 3, -80, 100, tid=9),
        ),
        operations=[_operation("extract-1")],
    )

    device = _device(profile)
    assert device["matched_free_events"] == 1
    assert device["new_allocations_live_at_range_exit_bytes"] == 0
    assert device["temporary_allocation_events"] == 1
    assert device["temporary_subset_live_peak_bytes"] == 80


def test_address_reuse_before_a_free_is_an_inconsistency_not_a_precise_peak() -> None:
    profile = summarize_cache_memory_profile(
        _trace(
            _range("extract-1", 0, 10),
            _memory(1, 1, 64, 100),
            _memory(2, 2, 64, 100),
            _memory(3, 3, -64, 100),
        ),
        operations=[_operation("extract-1")],
    )

    device = _device(profile)
    assert device["allocated_bytes_total"] == 128
    assert device["new_allocation_live_peak_bytes"] is None
    assert "allocation_reused_before_matching_free" in device["inconsistencies"]


def test_mismatched_free_size_does_not_claim_a_precise_peak() -> None:
    profile = summarize_cache_memory_profile(
        _trace(
            _range("extract-1", 0, 10),
            _memory(1, 1, 64, 100),
            _memory(2, 2, -32, 100),
        ),
        operations=[_operation("extract-1")],
    )

    device = _device(profile)
    assert device["new_allocation_live_peak_bytes"] is None
    assert "matching_free_size_differs_from_allocation" in device["inconsistencies"]


def test_address_reuse_after_its_free_starts_a_new_lifetime() -> None:
    profile = summarize_cache_memory_profile(
        _trace(
            _range("extract-1", 0, 10),
            _memory(1, 1, 64, 100),
            _memory(2, 2, -64, 100),
            _memory(3, 3, 32, 100),
            _memory(4, 4, -32, 100),
        ),
        operations=[_operation("extract-1")],
    )

    device = _device(profile)
    assert device["allocation_events"] == 2
    assert device["allocated_bytes_total"] == 96
    assert device["new_allocation_live_peak_bytes"] == 64
    assert device["temporary_subset_live_peak_bytes"] == 64


def test_unmatched_free_is_kept_as_preexisting_or_untracked() -> None:
    profile = summarize_cache_memory_profile(
        _trace(
            _range("extract-1", 0, 10),
            _memory(1, 1, -32, 700),
            _memory(2, 2, 16, 100),
            _memory(3, 3, -16, 100),
        ),
        operations=[_operation("extract-1")],
    )

    global_events = profile["global_memory_events"]
    assert global_events["preexisting_or_untracked_free_events"] == 1
    assert global_events["preexisting_or_untracked_free_bytes"] == 32
    assert _device(profile)["new_allocation_live_peak_bytes"] == 16


def test_unannotated_and_missing_annotation_ranges_have_no_zero_metrics() -> None:
    profile = summarize_cache_memory_profile(
        _trace(_range("present", 0, 10), _memory(1, 1, 20, 100)),
        operations=[_operation("present"), _operation("missing"), _operation("old", annotated=False)],
    )

    present, missing, old = profile["operations"]
    assert present["coverage"] == "complete"
    assert missing["coverage"] == "annotation_range_missing"
    assert missing["devices"] == []
    assert old["coverage"] == "not_profiled"
    assert old["devices"] == []


def test_partial_capture_nulls_strict_peaks_but_keeps_observed_exit_liveness() -> None:
    profile = summarize_cache_memory_profile(
        _trace(_range("extract-1", 0, 10), _memory(1, 1, 64, 100)),
        operations=[_operation("extract-1")],
        capture_complete=False,
    )

    device = _device(profile)
    assert profile["operations"][0]["coverage"] == "partial_capture"
    assert device["allocated_bytes_total"] == 64
    assert device["new_allocation_live_peak_bytes"] is None
    assert device["temporary_subset_live_peak_bytes"] is None
    assert device["new_allocations_live_at_range_exit_bytes"] is None
    assert device["right_censored"] is True


def test_open_lifetime_in_a_complete_capture_keeps_its_observed_peak() -> None:
    profile = summarize_cache_memory_profile(
        _trace(_range("extract-1", 0, 10), _memory(1, 1, 64, 100)),
        operations=[_operation("extract-1")],
    )

    device = _device(profile)
    assert device["new_allocation_live_peak_bytes"] == 64
    assert device["new_allocations_live_at_range_exit_bytes"] == 64
    assert device["right_censored"] is True


def test_non_cpu_devices_remain_backend_unverified() -> None:
    profile = summarize_cache_memory_profile(
        _trace(
            _range("extract-1", 0, 10),
            _memory(1, 1, 64, 100, device_type=2, device_id=0),
            _memory(2, 2, -64, 100, device_type=2, device_id=0),
        ),
        operations=[_operation("extract-1")],
    )

    device = _device(profile)
    assert device["backend_validation"] == "unverified"
    assert device["new_allocation_live_peak_bytes"] is None
    assert device["new_allocations_live_at_range_exit_bytes"] is None
    assert device["temporary_subset_live_peak_bytes"] is None


def test_boundary_and_same_timestamp_changes_do_not_claim_precise_peaks() -> None:
    boundary = summarize_cache_memory_profile(
        _trace(_range("extract-1", 0, 10), _memory(1, 0, 64, 100)),
        operations=[_operation("extract-1")],
    )
    boundary_device = _device(boundary)
    assert boundary_device["boundary_event_count"] == 1
    assert boundary_device["new_allocation_live_peak_bytes"] is None
    assert boundary["global_memory_events"]["unassigned_allocated_bytes"] == 64

    same_timestamp = summarize_cache_memory_profile(
        _trace(
            _range("extract-1", 0, 10),
            _memory(1, 2, 64, 100),
            _memory(2, 2, -64, 100),
        ),
        operations=[_operation("extract-1")],
    )
    same_time_device = _device(same_timestamp)
    assert same_time_device["new_allocation_live_peak_bytes"] is None
    assert "same_timestamp_positive_and_negative_memory_changes" in same_time_device["ambiguities"]


def test_ev_idx_exact_duplicates_are_deduplicated_but_conflicts_are_rejected() -> None:
    allocation = _memory(1, 1, 64, 100)
    profile = summarize_cache_memory_profile(
        _trace(_range("extract-1", 0, 10), allocation, copy.deepcopy(allocation)),
        operations=[_operation("extract-1")],
    )
    assert profile["global_memory_events"]["memory_allocation_events"] == 1
    assert profile["global_memory_events"]["duplicate_memory_events"] == 1

    with pytest.raises(ValueError, match="Ev Idx 1 is repeated with different content"):
        summarize_cache_memory_profile(
            _trace(
                _range("extract-1", 0, 10),
                _memory(1, 1, 64, 100),
                _memory(1, 2, 64, 200),
            ),
            operations=[_operation("extract-1")],
        )


def test_actual_cpu_probe_trace_reports_the_observed_524288_byte_peak() -> None:
    probe_path = (
        ROOT
        / "tmp"
        / "a_memory_runtime_20260907"
        / "memory_profiler_probe"
        / "chrome_trace.json"
    )
    trace = json.loads(probe_path.read_text(encoding="utf-8"))
    memory_events = [event for event in trace["traceEvents"] if event.get("name") == "[memory]"]
    first, last = memory_events[0], memory_events[-1]
    start = first["ts"] - 1.0
    trace["traceEvents"].append(
        _range(
            "probe-op",
            start,
            (last["ts"] + 1.0) - start,
            pid=first["pid"],
            tid=first["tid"],
        )
    )

    device = _device(
        summarize_cache_memory_profile(trace, operations=[_operation("probe-op")])
    )
    assert device["allocated_bytes_total"] == 524288
    assert device["new_allocation_live_peak_bytes"] == 524288
    assert device["new_allocations_live_at_range_exit_bytes"] == 0
    assert device["temporary_subset_live_peak_bytes"] == 524288
