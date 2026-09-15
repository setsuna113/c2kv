"""Failure and lifetime provenance checks for the tiny CPU cache runtime."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest

torch = pytest.importorskip("torch")

from history_memory.inference import EventNativeGenerator
from history_memory.test_incremental_inference import _runtime
from history_memory.test_inference import _chunk, _memory, _memory_with_chunks


def _generate(
    generator: EventNativeGenerator,
    memory,
    attempt_uid: str,
    *,
    max_new_tokens: int = 2,
):
    return generator.generate(
        memory,
        ratio=4,
        max_new_tokens=max_new_tokens,
        trace_context={
            "attempt_uid": attempt_uid,
            "session_id": "trace-session",
            "decision_key": attempt_uid,
            "phase": "draft",
        },
    )


def _ops(trace: dict, kind: str) -> list[dict]:
    return [op for op in trace["ops"] if op["kind"] == kind]


def test_partial_cpu_commit_copy_failure_preserves_known_work_and_releases_it(monkeypatch):
    generator = EventNativeGenerator(_runtime(81, "eager", torch.float32))
    memory = _memory_with_chunks(
        _chunk("first", (13, 14, 15, 16, 17), 1),
        _chunk("second", (31, 32, 33, 34), 2),
    )
    copy_encoded = generator._copy_encoded_to_cpu
    copies = 0

    def fail_second_copy(encoded):
        nonlocal copies
        copies += 1
        if copies == 2:
            raise RuntimeError("second CPU memo copy failed")
        return copy_encoded(encoded)

    monkeypatch.setattr(generator, "_copy_encoded_to_cpu", fail_second_copy)
    with pytest.raises(RuntimeError, match="second CPU memo copy failed"):
        with generator.decision_scope(session_id="session-a"):
            _generate(generator, memory, "commit-copy-failure")

    trace = generator.last_generation_trace
    completed, failed = _ops(trace, "cpu_memo_copy")
    assert completed["status"] == "completed"
    assert completed["logical_bytes"] > 0
    assert completed["transfer_bytes"] == 0
    assert failed["status"] == "failed"
    assert failed["input_tokens_completed"] is None
    assert failed["transfer_bytes"] is None
    assert failed["logical_bytes"] is None
    assert failed["error_type"] == "RuntimeError"

    cpu_entries = [entry for entry in trace["entries"] if entry["kind"] == "encoded_cpu"]
    assert len(cpu_entries) == 1
    releases = [
        op for op in _ops(trace, "release")
        if op.get("reason") == "unpublished_commit_failure"
    ]
    assert [op["source_entry_id"] for op in releases] == [cpu_entries[0]["entry_id"]]
    assert trace["commit_status"] == "cleared_on_failure"
    info = generator.session_cache_info()
    assert info["session_id"] is None
    assert not info["cpu_memo_present"]
    assert not info["device_raw_snapshot_present"]
    json.dumps(trace, allow_nan=False)


def test_final_generation_trace_is_not_mutated_by_later_cache_lifecycle_events():
    generator = EventNativeGenerator(_runtime(82, "eager", torch.float32))
    with generator.decision_scope(session_id="first-session"):
        first = _generate(generator, _memory(), "first-attempt")
    first_trace = first.stats["cache_trace"]
    first_snapshot = deepcopy(first_trace)

    generator.close_session()
    assert first_trace == first_snapshot
    closed_lifecycle = generator.session_cache_info()["last_lifecycle_trace"]
    assert closed_lifecycle is not first_trace
    assert closed_lifecycle["schema"] == "event-native-cache-lifecycle-v1"
    assert closed_lifecycle["associated_attempt_uid"] == "first-attempt"
    assert _ops(closed_lifecycle, "release")

    with generator.decision_scope(session_id="old-session"):
        old = _generate(generator, _memory(), "switch-attempt")
    old_trace = old.stats["cache_trace"]
    old_snapshot = deepcopy(old_trace)
    with generator.decision_scope(session_id="new-session"):
        pass

    assert old_trace == old_snapshot
    switched_lifecycle = generator.session_cache_info()["last_lifecycle_trace"]
    assert switched_lifecycle is not old_trace
    assert switched_lifecycle["schema"] == "event-native-cache-lifecycle-v1"
    assert switched_lifecycle["associated_attempt_uid"] == "switch-attempt"
    assert _ops(switched_lifecycle, "release")
    assert json.loads(json.dumps([first_trace, old_trace, switched_lifecycle]))[0] == first_trace


def test_full_recompute_trace_counts_each_actual_target_forward_once():
    generator = EventNativeGenerator(
        _runtime(83, "eager", torch.float32),
        decode_strategy="full_recompute",
    )
    memory = _memory(chunk_tokens=None)
    result = _generate(generator, memory, "full-recompute", max_new_tokens=3)
    trace = result.stats["cache_trace"]
    forwards = [
        op for op in trace["ops"]
        if op["kind"] in {"raw_prefill", "full_recompute"}
    ]

    assert [op["kind"] for op in forwards] == [
        "raw_prefill", "full_recompute", "full_recompute",
    ]
    assert [op["input_tokens_requested"] for op in forwards] == [3, 4, 5]
    assert all(
        op["input_tokens_completed"] == op["input_tokens_requested"]
        for op in forwards
    )
    assert sum(op["input_tokens_completed"] for op in forwards) == result.stats[
        "target_input_tokens"
    ]
    assert len(forwards) == result.stats["target_forward_calls"]


def test_caller_failure_before_next_generate_keeps_original_error_and_releases_snapshot():
    generator = EventNativeGenerator(_runtime(84, "eager", torch.float32))
    with generator.decision_scope(session_id="s"):
        first = _generate(generator, _memory(), "previous-attempt")
    frozen = deepcopy(first.stats["cache_trace"])
    with pytest.raises(ValueError, match="caller failed before generate"):
        with generator.decision_scope(session_id="s"):
            raise ValueError("caller failed before generate")
    assert generator._session_cache is None
    assert first.stats["cache_trace"] == frozen
    lifecycle = generator.session_cache_info()["last_lifecycle_trace"]
    assert lifecycle["reason"] == "scope_failure_before_generation"
    assert len(_ops(lifecycle, "release")) == 3
