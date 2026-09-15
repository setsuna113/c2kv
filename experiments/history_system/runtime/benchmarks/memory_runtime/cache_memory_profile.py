"""Attribute observed torch Chrome ``[memory]`` lifetimes to cache operations.

The profiler's memory events, rather than inclusive record-function deltas or
tensor ``numel()``, are the source of every byte reported here.  The module is
intentionally independent of torch so an already-exported Chrome trace can be
audited on a CPU-only machine.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

# This module is also used as a standalone post-capture reader, where the
# repository's ``python`` package root is not necessarily on ``sys.path``.
PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from history_memory.cache_trace import (
    CACHE_MEMORY_OPERATION_KINDS,
    CACHE_MEMORY_RANGE_PREFIX,
)


SCHEMA = "a-cache-memory-profile-v1"
_IDENTITY_FIELDS = (
    "attempt_uid",
    "session_id",
    "decision_key",
    "phase",
    "op_id",
    "kind",
    "status",
)


def _identifier(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _integer(value: Any, *, label: str, allow_negative: bool = True) -> int:
    if type(value) is not int or (not allow_negative and value < 0):
        sign = "an integer" if allow_negative else "a nonnegative integer"
        raise ValueError(f"{label} must be {sign}")
    return value


def _trace_identifier(value: Any, *, label: str) -> int | str:
    if type(value) not in (int, str):
        raise ValueError(f"{label} must be an integer or string")
    return value


def _decimal(value: Any, *, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError(f"{label} must be a finite number")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not parsed.is_finite():
        raise ValueError(f"{label} must be a finite number")
    return parsed


def _canonical_json(value: Any, *, label: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain finite JSON data") from error


def _events(chrome_trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if not isinstance(chrome_trace, Mapping):
        raise ValueError("chrome_trace must be an object")
    events = chrome_trace.get("traceEvents")
    if not isinstance(events, list):
        raise ValueError("chrome_trace.traceEvents must be a list")
    result = []
    for index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"chrome_trace.traceEvents[{index}] must be an object")
        result.append(event)
    return result


def _range_event(event: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    ts = _decimal(event.get("ts"), label=f"{label}.ts")
    duration = _decimal(event.get("dur"), label=f"{label}.dur")
    if duration < 0:
        raise ValueError(f"{label}.dur must not be negative")
    return {
        "start": ts,
        "end": ts + duration,
        "duration": duration,
        "pid": _trace_identifier(event.get("pid"), label=f"{label}.pid"),
        "tid": _trace_identifier(event.get("tid"), label=f"{label}.tid"),
    }


def _memory_event(event: Mapping[str, Any], *, label: str, order: int) -> dict[str, Any]:
    args = event.get("args")
    if not isinstance(args, Mapping):
        raise ValueError(f"{label}.args must be an object")
    bytes_delta = _integer(args.get("Bytes"), label=f"{label}.args.Bytes")
    if bytes_delta == 0:
        raise ValueError(f"{label}.args.Bytes must not be zero")
    return {
        "ts": _decimal(event.get("ts"), label=f"{label}.ts"),
        "pid": _trace_identifier(event.get("pid"), label=f"{label}.pid"),
        "tid": _trace_identifier(event.get("tid"), label=f"{label}.tid"),
        "event_index": _integer(args.get("Ev Idx"), label=f"{label}.args.Ev Idx", allow_negative=False),
        "addr": _integer(args.get("Addr"), label=f"{label}.args.Addr", allow_negative=False),
        "bytes": bytes_delta,
        "device_type": _integer(args.get("Device Type"), label=f"{label}.args.Device Type"),
        "device_id": _integer(args.get("Device Id"), label=f"{label}.args.Device Id"),
        "order": order,
        "raw": event,
    }


def _operation_states(operations: list[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    states: list[dict[str, Any]] = []
    annotation_to_index: dict[str, int] = {}
    for index, operation in enumerate(operations):
        if not isinstance(operation, Mapping):
            raise ValueError(f"operations[{index}] must be an object")
        identity = {
            name: _identifier(operation.get(name), label=f"operations[{index}].{name}")
            for name in _IDENTITY_FIELDS
        }
        annotation = operation.get("memory_profile_annotation")
        if annotation is not None:
            expected = CACHE_MEMORY_RANGE_PREFIX + identity["op_id"]
            if annotation != expected:
                raise ValueError(
                    f"operations[{index}].memory_profile_annotation must equal {expected!r}"
                )
            if annotation in annotation_to_index:
                raise ValueError(f"operations repeat memory profile annotation {annotation!r}")
            annotation_to_index[annotation] = index
        states.append(
            {
                "identity": identity,
                "annotation": annotation,
                "range": None,
                "range_count": 0,
                "range_issue": None,
                "lives_by_device": defaultdict(list),
                "device_flags": defaultdict(lambda: {"ambiguities": set(), "inconsistencies": set(), "boundary_events": 0}),
            }
        )
    return states, annotation_to_index


def _add_flag(state: dict[str, Any], device: tuple[int, int], category: str, value: str) -> None:
    state["device_flags"][device][category].add(value)


def _record_boundary_hits(states: list[dict[str, Any]], memory: Mapping[str, Any]) -> None:
    device = (memory["device_type"], memory["device_id"])
    for state in states:
        marker = state["range"]
        if marker is None:
            continue
        if memory["pid"] == marker["pid"] and memory["tid"] == marker["tid"] and memory["ts"] in (marker["start"], marker["end"]):
            state["device_flags"][device]["boundary_events"] += 1
            _add_flag(state, device, "ambiguities", "memory_event_on_annotation_boundary")


def _assign_allocation(
    states: list[dict[str, Any]],
    memory: Mapping[str, Any],
) -> tuple[int | None, bool, bool]:
    """Return owner index, whether it is unassigned, and whether it is off-thread."""

    candidates: list[tuple[int, dict[str, Any]]] = []
    for index, state in enumerate(states):
        marker = state["range"]
        if marker is None:
            continue
        if memory["pid"] != marker["pid"] or memory["tid"] != marker["tid"]:
            continue
        if marker["start"] < memory["ts"] < marker["end"]:
            candidates.append((index, marker))
    if candidates:
        shortest = min(marker["duration"] for _, marker in candidates)
        winners = [index for index, marker in candidates if marker["duration"] == shortest]
        if len(winners) == 1:
            return winners[0], False, False
        device = (memory["device_type"], memory["device_id"])
        for index in winners:
            _add_flag(states[index], device, "ambiguities", "allocation_has_no_unique_innermost_annotation")
        return None, True, False

    off_thread = any(
        state["range"] is not None
        and memory["pid"] == state["range"]["pid"]
        and memory["tid"] != state["range"]["tid"]
        and state["range"]["start"] < memory["ts"] < state["range"]["end"]
        for state in states
    )
    return None, True, off_thread


def _mark_life_issue(life: Mapping[str, Any], states: list[dict[str, Any]], issue: str) -> None:
    owner = life["owner"]
    if owner is None:
        return
    _add_flag(states[owner], life["device"], "inconsistencies", issue)


def _same_process_inside(memory: Mapping[str, Any], marker: Mapping[str, Any]) -> bool:
    return (
        memory["pid"] == marker["pid"]
        and marker["start"] < memory["ts"] < marker["end"]
    )


def _sweep_peak(lives: list[Mapping[str, Any]]) -> int:
    changes: dict[Decimal, int] = defaultdict(int)
    for life in lives:
        changes[life["allocation"]["ts"]] += life["bytes"]
        if life["free"] is not None:
            changes[life["free"]["ts"]] -= life["bytes"]
    live = 0
    peak = 0
    for timestamp in sorted(changes):
        live += changes[timestamp]
        peak = max(peak, live)
    return peak


def _device_metrics(
    state: dict[str, Any],
    device: tuple[int, int],
    *,
    capture_complete: bool,
) -> dict[str, Any]:
    marker = state["range"]
    assert marker is not None
    lives = state["lives_by_device"][device]
    flags = state["device_flags"][device]
    ambiguities = set(flags["ambiguities"])
    inconsistencies = set(flags["inconsistencies"])
    for life in lives:
        free = life["free"]
        if free is not None and free["ts"] in (marker["start"], marker["end"]):
            ambiguities.add("matching_free_on_annotation_boundary")
        if life["same_timestamp_mixed_change"]:
            ambiguities.add("same_timestamp_positive_and_negative_memory_changes")
        if life["forced_end"]:
            inconsistencies.add("allocation_reused_before_matching_free")
        if life["size_mismatch"]:
            inconsistencies.add("matching_free_size_differs_from_allocation")

    temporary = [
        life
        for life in lives
        if life["free"] is not None and _same_process_inside(life["free"], marker)
    ]
    exit_live = sum(
        life["bytes"]
        for life in lives
        if life["allocation"]["ts"] < marker["end"]
        and (life["free"] is None or life["free"]["ts"] > marker["end"])
    )
    backend_validation = "validated_cpu_memory_events" if device[0] == 0 else "unverified"
    strict_peaks_available = (
        capture_complete
        and state["range_issue"] is None
        and device[0] == 0
        and not ambiguities
        and not inconsistencies
    )
    return {
        "device_type": device[0],
        "device_id": device[1],
        "backend_validation": backend_validation,
        "allocation_events": len(lives),
        "matched_free_events": sum(life["free"] is not None for life in lives),
        "open_allocation_events": sum(life["free"] is None for life in lives),
        "temporary_allocation_events": len(temporary),
        "allocated_bytes_total": sum(life["bytes"] for life in lives),
        "new_allocation_live_peak_bytes": _sweep_peak(lives) if strict_peaks_available else None,
        "new_allocations_live_at_range_exit_bytes": (
            exit_live if strict_peaks_available else None
        ),
        "temporary_subset_live_peak_bytes": (
            _sweep_peak(temporary) if strict_peaks_available else None
        ),
        "right_censored": any(life["free"] is None for life in lives),
        "strict_peaks_available": strict_peaks_available,
        "boundary_event_count": flags["boundary_events"],
        "ambiguities": sorted(ambiguities),
        "inconsistencies": sorted(inconsistencies),
    }


def summarize_cache_memory_profile(
    chrome_trace: Mapping[str, Any],
    *,
    operations: list[Mapping[str, Any]],
    capture_complete: bool = True,
) -> dict[str, Any]:
    """Summarize observed allocation lifetimes for annotated cache operations.

    ``capture_complete`` means the Chrome capture covers the requested
    profiling interval.  With an incomplete capture, observed allocation byte
    counts remain useful lower-bound observations but strict peak values are
    deliberately null.
    """

    if type(capture_complete) is not bool:
        raise ValueError("capture_complete must be a bool")
    if not isinstance(operations, list):
        raise ValueError("operations must be a list")
    raw_events = _events(chrome_trace)
    states, expected_annotations = _operation_states(operations)

    ranges_by_annotation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unexpected_annotation_ranges = 0
    raw_memory_events: list[dict[str, Any]] = []
    for index, event in enumerate(raw_events):
        name = event.get("name")
        if event.get("ph") == "X" and type(name) is str and name.startswith(CACHE_MEMORY_RANGE_PREFIX):
            if name in expected_annotations:
                ranges_by_annotation[name].append(
                    _range_event(event, label=f"chrome_trace.traceEvents[{index}]")
                )
            else:
                unexpected_annotation_ranges += 1
        if event.get("ph") == "i" and event.get("name") == "[memory]":
            raw_memory_events.append(
                _memory_event(event, label=f"chrome_trace.traceEvents[{index}]", order=index)
            )

    annotation_ranges_found = 0
    missing_annotation_ranges = 0
    for annotation, state_index in expected_annotations.items():
        ranges = ranges_by_annotation[annotation]
        state = states[state_index]
        state["range_count"] = len(ranges)
        if len(ranges) == 1:
            state["range"] = ranges[0]
            annotation_ranges_found += 1
        elif not ranges:
            state["range_issue"] = "annotation_range_missing"
            missing_annotation_ranges += 1
        else:
            state["range_issue"] = "annotation_range_repeated"
            missing_annotation_ranges += 1

    unique_memory_events: list[dict[str, Any]] = []
    event_index_contents: dict[int, str] = {}
    duplicate_memory_events = 0
    for memory in raw_memory_events:
        event_index = memory["event_index"]
        content = _canonical_json(memory["raw"], label=f"memory Ev Idx {event_index}")
        previous = event_index_contents.get(event_index)
        if previous is not None:
            if previous != content:
                raise ValueError(f"memory Ev Idx {event_index} is repeated with different content")
            duplicate_memory_events += 1
            continue
        event_index_contents[event_index] = content
        unique_memory_events.append(memory)

    by_timestamp: dict[Decimal, list[dict[str, Any]]] = defaultdict(list)
    for memory in unique_memory_events:
        by_timestamp[memory["ts"]].append(memory)
    for same_time in by_timestamp.values():
        mixed = any(event["bytes"] > 0 for event in same_time) and any(
            event["bytes"] < 0 for event in same_time
        )
        for event in same_time:
            event["same_timestamp_mixed_change"] = mixed

    active_lives: dict[tuple[int | str, int, int, int], dict[str, Any]] = {}
    unmatched_free_events = 0
    unmatched_free_bytes = 0
    unassigned_allocation_events = 0
    unassigned_allocated_bytes = 0
    off_thread_allocation_events = 0
    off_thread_allocated_bytes = 0
    allocation_events = 0
    free_events = 0
    allocated_bytes_total = 0
    freed_bytes_total = 0

    # The input position gives a deterministic bookkeeping order only.  It is
    # never used to claim causality between opposite-sign events at one ts.
    for memory in sorted(unique_memory_events, key=lambda event: (event["ts"], event["order"])):
        _record_boundary_hits(states, memory)
        device = (memory["device_type"], memory["device_id"])
        physical_key = (memory["pid"], memory["device_type"], memory["device_id"], memory["addr"])
        if memory["bytes"] > 0:
            allocation_events += 1
            allocated_bytes_total += memory["bytes"]
            owner, unassigned, off_thread = _assign_allocation(states, memory)
            if unassigned:
                unassigned_allocation_events += 1
                unassigned_allocated_bytes += memory["bytes"]
                if off_thread:
                    off_thread_allocation_events += 1
                    off_thread_allocated_bytes += memory["bytes"]
            prior = active_lives.get(physical_key)
            if prior is not None:
                prior["forced_end"] = True
                prior["free"] = memory
                _mark_life_issue(prior, states, "allocation_reused_before_matching_free")
            life = {
                "allocation": memory,
                "free": None,
                "bytes": memory["bytes"],
                "device": device,
                "owner": owner,
                "forced_end": False,
                "size_mismatch": False,
                "same_timestamp_mixed_change": memory["same_timestamp_mixed_change"],
            }
            active_lives[physical_key] = life
            if prior is not None:
                _mark_life_issue(life, states, "allocation_reused_before_matching_free")
            if owner is not None:
                states[owner]["lives_by_device"][device].append(life)
        else:
            free_events += 1
            freed_bytes_total += -memory["bytes"]
            life = active_lives.pop(physical_key, None)
            if life is None:
                unmatched_free_events += 1
                unmatched_free_bytes += -memory["bytes"]
                continue
            life["free"] = memory
            life["same_timestamp_mixed_change"] = (
                life["same_timestamp_mixed_change"] or memory["same_timestamp_mixed_change"]
            )
            if -memory["bytes"] != life["bytes"]:
                life["size_mismatch"] = True
                _mark_life_issue(life, states, "matching_free_size_differs_from_allocation")

    rows = []
    for state in states:
        identity = state["identity"]
        row = {
            **identity,
            "memory_profile_annotation": state["annotation"],
            "annotation_range_count": state["range_count"],
            "coverage": "not_profiled",
            "devices": [],
        }
        if state["annotation"] is None:
            rows.append(row)
            continue
        if state["range_issue"] is not None:
            row["coverage"] = state["range_issue"]
            rows.append(row)
            continue
        row["coverage"] = "complete" if capture_complete else "partial_capture"
        devices = set(state["lives_by_device"]) | set(state["device_flags"])
        row["devices"] = [
            _device_metrics(state, device, capture_complete=capture_complete)
            for device in sorted(devices)
        ]
        rows.append(row)

    return {
        "schema": SCHEMA,
        "capture_complete": capture_complete,
        "backend_validation": "CPU Device Type 0 observed allocation lifetimes only",
        "definitions": {
            "source": "torch Chrome [memory] events are authoritative; profiler record-function inclusive deltas are not summed",
            "scope": "CPU Device Type 0 is lifetime-validated. This is not OS RSS, whole-process HBM, allocator reservation, or GPU elapsed-time evidence.",
            "allocated_bytes_total": "sum of positive [memory] Bytes events assigned to the unique innermost same-thread annotation range",
            "new_allocation_live_peak_bytes": "peak live bytes of allocations first observed in one operation range; never a sum of parent and child profiler deltas",
            "new_allocations_live_at_range_exit_bytes": "observed live bytes from that operation's new allocations at annotation-range exit",
            "temporary_subset_live_peak_bytes": "peak only for allocations whose matching free is in the same process and strictly inside the annotation time range",
            "strict_null_policy": "peak and range-exit values are null when capture is incomplete, the device backend is unverified, or the relevant lifetime is ambiguous or inconsistent",
            "ignored_profiler_counters": "Total Allocated and Total Reserved are not used as absolute baselines; raw [memory] allocation/free events define the observed lifetimes",
            "annotatable_operation_kinds": sorted(CACHE_MEMORY_OPERATION_KINDS),
        },
        "annotation_coverage": {
            "annotated_operations": len(expected_annotations),
            "annotation_ranges_found": annotation_ranges_found,
            "missing_or_repeated_annotation_ranges": missing_annotation_ranges,
            "unexpected_annotation_ranges": unexpected_annotation_ranges,
            "unannotated_operations": len(states) - len(expected_annotations),
        },
        "global_memory_events": {
            "memory_allocation_events": allocation_events,
            "memory_free_events": free_events,
            "allocated_bytes_total": allocated_bytes_total,
            "freed_bytes_total": freed_bytes_total,
            "duplicate_memory_events": duplicate_memory_events,
            "unassigned_allocation_events": unassigned_allocation_events,
            "unassigned_allocated_bytes": unassigned_allocated_bytes,
            "off_thread_allocation_events": off_thread_allocation_events,
            "off_thread_allocated_bytes": off_thread_allocated_bytes,
            "preexisting_or_untracked_free_events": unmatched_free_events,
            "preexisting_or_untracked_free_bytes": unmatched_free_bytes,
        },
        "operations": rows,
    }
