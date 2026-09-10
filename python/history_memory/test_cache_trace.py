"""Contract tests for the event-native cache provenance trace."""

from __future__ import annotations

import json
import math

import pytest

from history_memory.cache_trace import CacheTrace


def test_trace_records_json_safe_operations_entries_and_placements() -> None:
    trace = CacheTrace(
        {
            "attempt_uid": "attempt-7",
            "session_id": "session-1",
            "decision_key": "turn-1/step-2",
            "phase": "draft",
        }
    )
    extract = trace.start_op("extract", input_tokens_requested=12)
    trace.finish_op(extract, input_tokens_completed=12, transfer_bytes=0)
    entry = trace.new_entry(
        "encoded_device",
        extract,
        origin_extraction_op_id=extract["op_id"],
        prefix_bindings=[{"event_id": "s:m1"}],
    )
    placement = trace.add_placement(
        access_op_id=extract["op_id"],
        accessed_entry_id=entry["entry_id"],
        source_indices=[1, 2],
    )

    assert trace.data["context_complete"] is True
    assert extract["op_id"] == "attempt-7:op:1"
    assert extract["status"] == "completed"
    assert extract["logical_bytes"] is None
    assert entry == {
        "entry_id": "attempt-7:entry:1",
        "kind": "encoded_device",
        "created_by_op_id": "attempt-7:op:1",
        "producer_attempt_uid": "attempt-7",
        "parent_entry_id": None,
        "origin_extraction_op_id": "attempt-7:op:1",
        "prefix_bindings": [{"event_id": "s:m1"}],
    }
    assert placement["placement_id"] == "attempt-7:placement:1"
    assert json.loads(json.dumps(trace.data, allow_nan=False)) == trace.data
    assert trace.data["workspace_source_group"] == []
    assert trace.data["commit_status"] == "not_requested"


def test_child_entry_is_flat_and_does_not_mutate_parent_provenance() -> None:
    parent_trace = CacheTrace({"attempt_uid": "first"})
    extraction = parent_trace.start_op("extract")
    parent = parent_trace.new_entry(
        "encoded_device",
        extraction,
        origin_extraction_op_id=extraction["op_id"],
    )
    parent_before = json.loads(json.dumps(parent))

    child_trace = CacheTrace({"attempt_uid": "second"})
    hydrate = child_trace.start_op("cpu_memo_hydrate")
    child = child_trace.new_entry("encoded_device", hydrate, parent=parent)

    assert parent == parent_before
    assert child["parent_entry_id"] == parent["entry_id"]
    assert child["origin_extraction_op_id"] == extraction["op_id"]
    assert child["created_by_op_id"] == hydrate["op_id"]
    assert child["producer_attempt_uid"] == "second"
    assert "producer" not in child
    assert "lineage" not in child


def test_missing_context_is_incomplete_and_rejects_non_json_values() -> None:
    trace = CacheTrace()
    assert trace.data["context_complete"] is False
    assert trace.data["session_id"] is None
    assert trace.data["attempt_uid"] == trace.attempt_uid
    assert len(trace.attempt_uid.split("-")) == 5

    with pytest.raises(TypeError, match="JSON-native"):
        trace.start_op("extract", payload=(1, 2))
    with pytest.raises(TypeError, match="JSON-native"):
        trace.add_placement(tensor_like=object())
    with pytest.raises(ValueError, match="non-finite"):
        trace.start_op("extract", elapsed=math.nan)

    failed = trace.start_op("extract", input_tokens_requested=4)
    trace.fail_op(failed, RuntimeError("do not expose exception text"))
    assert failed["status"] == "failed"
    assert failed["input_tokens_requested"] == 4
    assert failed["input_tokens_completed"] is None
    assert failed["transfer_bytes"] is None
    assert failed["logical_bytes"] is None
    assert failed["error_type"] == "RuntimeError"
    assert "do not expose exception text" not in json.dumps(failed)

    op = trace.start_op("extract")
    with pytest.raises(ValueError, match="unsupported"):
        CacheTrace({"attempt_uid": "a", "extra": "no"})
    with pytest.raises(ValueError, match="op is not started"):
        trace.finish_op(op)
        trace.finish_op(op)
