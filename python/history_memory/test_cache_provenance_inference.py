"""Actual cache branch and partial-cost provenance on tiny CPU Qwen."""
from dataclasses import replace
import json

import pytest

torch = pytest.importorskip("torch")

from history_memory.inference import EventNativeGenerator
from history_memory.test_incremental_inference import _runtime
from history_memory.test_inference import _chunk, _memory, _memory_with_chunks


def _call(generator, memory, attempt, *, phase="draft"):
    return generator.generate(memory, ratio=4, max_new_tokens=2, trace_context={
        "attempt_uid": attempt, "session_id": "s", "decision_key": attempt,
        "phase": phase,
    })


def _ops(trace, kind):
    return [op for op in trace["ops"] if op["kind"] == kind]


def _entries(trace, kind):
    return [entry for entry in trace["entries"] if entry["kind"] == kind]


def test_duplicate_consumers_and_scope_reuse_keep_actual_producer():
    generator = EventNativeGenerator(_runtime(72, "eager", torch.float32))
    memory = _memory(duplicate_chunk=True)
    with generator.decision_scope():
        first = _call(generator, memory, "a1")
        second = _call(generator, memory, "a2", phase="regeneration")
    left, right = first.stats["cache_trace"], second.stats["cache_trace"]
    assert len(_ops(left, "extract")) == len(_ops(left, "call_memo_reuse")) == 1
    extraction = _ops(left, "extract")[0]
    assert extraction["input_tokens_completed"] == len(memory.chunks[0].token_ids)
    assert len(left["placements"]) == 2
    assert len({row["event_id"] for row in left["placements"]}) == 2
    assert len({row["accessed_entry_id"] for row in left["placements"]}) == 1
    assert all(row["origin_extraction_op_id"] == extraction["op_id"] for row in right["placements"])
    assert len(_ops(right, "scope_memo_reuse")) == len(_ops(right, "call_memo_reuse")) == 1
    assert not _ops(right, "extract") and not _entries(right, "encoded_device")
    assert first.token_ids == second.token_ids
    assert json.loads(json.dumps([left, right], allow_nan=False))[0] == left


def test_snapshot_consumers_preserve_origin_and_cpu_hydration_lineage():
    generator = EventNativeGenerator(_runtime(73, "eager", torch.float32))
    initial = _memory()
    with generator.decision_scope(session_id="s"):
        first = _call(generator, initial, "a1")
    left = first.stats["cache_trace"]
    origin = _ops(left, "extract")[0]["op_id"]
    cpu = _entries(left, "encoded_cpu")[0]
    old_snapshot = _entries(left, "raw_snapshot")[0]
    # Content and layout are identical, but the current consumer source differs.
    renamed = replace(initial, chunks=(replace(initial.chunks[0], event_id="new-event", source_indices=(9,)),),
                      raw_source_indices=(8,))
    with generator.decision_scope(session_id="s"):
        second = _call(generator, renamed, "a2")
    middle = second.stats["cache_trace"]
    assert not _ops(middle, "extract") and not _ops(middle, "cpu_memo_hydrate")
    row = middle["placements"][0]
    assert row["event_id"] == "new-event" and row["source_indices"] == [9]
    assert row["accessed_entry_id"] == old_snapshot["entry_id"]
    assert row["origin_extraction_op_id"] == origin
    assert old_snapshot["raw_source_group"] == [0]
    assert middle["workspace_source_group"] == [8]
    assert _ops(middle, "cpu_memo_retain")[0]["result_entry_id"] == cpu["entry_id"]

    changed = _memory_with_chunks(renamed.chunks[0], _chunk("extra", (31, 32, 33, 34), 10))
    with generator.decision_scope(session_id="s"):
        third = _call(generator, changed, "a3")
    right = third.stats["cache_trace"]
    hydrate = _ops(right, "cpu_memo_hydrate")[0]
    hydrated = next(entry for entry in right["entries"] if entry["entry_id"] == hydrate["result_entry_id"])
    assert hydrated["entry_id"] != cpu["entry_id"]
    assert hydrated["parent_entry_id"] == cpu["entry_id"]
    assert hydrated["origin_extraction_op_id"] == origin
    assert hydrate["transfer_bytes"] == 0
    assert len(_ops(right, "extract")) == 1
    assert _ops(left, "cpu_memo_copy")[0]["logical_bytes"] > 0
    assert _ops(left, "cpu_memo_copy")[0]["transfer_bytes"] == 0
    assert all(trace["commit_status"] == "committed" for trace in (left, middle, right))
    generator.close_session()
    assert not generator.session_cache_info()["cpu_memo_present"]
    json.dumps([left, middle, right], allow_nan=False)


@pytest.mark.parametrize("kind, initial, appended", [
    ("snapshot_take_whole", _memory(), True),
    ("snapshot_clone_prefix", _memory(), False),
    ("snapshot_empty_prefix", _memory(chunk_tokens=None, system=(), workspace=(21,)), False),
])
def test_snapshot_operation_matches_the_actual_ownership_branch(kind, initial, appended):
    generator = EventNativeGenerator(_runtime(74, "eager", torch.float32))
    with generator.decision_scope(session_id="s"):
        first = _call(generator, initial, "a1")
    next_memory = replace(initial, workspace_input_ids=initial.workspace_input_ids + first.token_ids + (24,)) if appended else initial
    with generator.decision_scope(session_id="s"):
        second = _call(generator, next_memory, "a2")
    assert len(_ops(second.stats["cache_trace"], kind)) == 1
    snapshot = _entries(second.stats["cache_trace"], "raw_snapshot")[0]
    assert snapshot["parent_entry_id"] == _entries(first.stats["cache_trace"], "raw_snapshot")[0]["entry_id"]
    generator.close_session()


def test_failed_extraction_preserves_completed_work_and_clears_tensors(monkeypatch):
    generator = EventNativeGenerator(_runtime(75, "eager", torch.float32))
    memory = _memory_with_chunks(_chunk("ok", (13, 14, 15, 16, 17), 1),
                                 _chunk("fail", (31, 32, 33, 34), 2))
    encode = generator.runtime._encode_chunk

    def failing(chunk, ratio):
        if chunk.event_id == "fail":
            raise RuntimeError("synthetic extraction failure")
        return encode(chunk, ratio)

    monkeypatch.setattr(generator.runtime, "_encode_chunk", failing)
    with pytest.raises(RuntimeError, match="synthetic extraction failure"):
        with generator.decision_scope(session_id="s"):
            _call(generator, memory, "a1")
    trace = generator.last_generation_trace
    first, second = _ops(trace, "extract")
    assert first["status"] == "completed" and first["input_tokens_completed"] == 5
    assert second["status"] == "failed" and second["input_tokens_requested"] == 4
    assert second["input_tokens_completed"] is second["transfer_bytes"] is None
    assert len(trace["placements"]) == 1 and trace["status"] == "failed"
    assert generator._session_cache is None and generator._active_decision_scope is None
    json.dumps(trace, allow_nan=False)
    # A rejected subsequent call must not expose a stale successful/failed trace.
    with pytest.raises(ValueError):
        generator.generate(memory, ratio=0, max_new_tokens=2)
    assert generator.last_generation_trace["attempt_uid"] != "a1"
    assert generator.last_generation_trace["ops"] == []


def test_regeneration_keeps_discarded_cost_and_commits_only_final_view():
    generator = EventNativeGenerator(_runtime(76, "eager", torch.float32))
    old, shared, new = _chunk("old", (7, 8, 9, 10), 1), _chunk("shared", (13, 14, 15, 16, 17), 2), _chunk("new", (31, 32, 33, 34), 3)
    with generator.decision_scope(session_id="s"):
        draft = _call(generator, _memory_with_chunks(old, shared), "a1")
        final = _call(generator, _memory_with_chunks(shared, new), "a2", phase="regeneration")
    left, right = draft.stats["cache_trace"], final.stats["cache_trace"]
    assert left["commit_status"] == "discarded_by_regeneration"
    assert sum(op["input_tokens_completed"] for op in _ops(left, "extract")) == 9
    assert any(op.get("reason") == "discarded_by_regeneration" for op in _ops(left, "release"))
    assert right["commit_status"] == "committed"
    assert len(_ops(right, "scope_memo_reuse")) == 1 and len(_ops(right, "extract")) == 1
    assert len(generator._session_cache.cpu_encoded_by_key) == 2
    generator.close_session()
