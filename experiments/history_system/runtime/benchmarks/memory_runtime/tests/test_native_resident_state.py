import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.event_native_exact_policy import EventNativeExactController
from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE, HISTORY_BUDGET_DEFINITION, POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from benchmarks.memory_runtime.event_native_raw import RuntimeMemoryView
from benchmarks.memory_runtime.policy import PolicyInputError
from history_memory.events import EventStore
from history_memory.packing import encode_event_chunks, pack_memory, select_view
from history_memory.resident_state import ResidentSourceState


class Tokenizer:
    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False, **kwargs):
        text = json.dumps(messages) + ("<assistant>" if add_generation_prompt else "")
        return [ord(char) for char in text]


def controller(*, max_sequence_tokens=20_000):
    return EventNativeExactController(
        Tokenizer(),
        packing={
            "ratios": [4], "recent_tool_events": 1, "max_chunk_tokens": 128,
            "chunk_overlap": 8, "max_chunks": 100, "max_encoder_tokens": 100_000,
            "max_system_tokens": 20_000, "max_workspace_tokens": 50_000,
            "max_target_tokens": 32, "max_sequence_tokens": max_sequence_tokens,
        },
        policy={
            "mode": "persistent", "history_budget_bytes": 1_000_000,
            "workspace_budget_bytes": 1_000_000, "lease_decisions": 0,
            "max_retrieved_events": 2, "kv_bytes_per_token": 1,
            "source_commit": POLICY_SOURCE_COMMIT,
            "history_budget_definition": HISTORY_BUDGET_DEFINITION,
            "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
            "current_input_baseline": CURRENT_INPUT_BASELINE,
        },
        mode="ac_gist_static",
        compression_policy=ALWAYS_COMPRESSION_POLICY,
        benchmark="bfcl",
    )


def messages():
    result = [{"role": "user", "content": "Find the answer"}]
    for index, size in enumerate((100, 100, 1200, 5)):
        result.extend((
            {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": f"call-{index}", "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }],
            },
            {
                "role": "tool", "tool_call_id": f"call-{index}",
                "content": chr(65 + index) * size,
            },
        ))
    return result


def payload(count, key):
    return {
        "session_id": "resident-task", "decision_key": key,
        "messages": messages()[:count], "tools": [],
    }


def _candidate_tokens(count, gist_indices):
    store = EventStore.from_messages("resident-task", messages()[:count], benchmark="bfcl")
    static = select_view(store, recent_tool_events=1)
    gist = tuple(store.events[index].event_id for index in gist_indices)
    raw = static.raw_event_ids
    omitted = tuple(event_id for event_id in static.gist_event_ids if event_id not in gist)
    view = RuntimeMemoryView(
        gist_event_ids=gist, raw_event_ids=raw,
        omitted_event_ids=omitted, mandatory_raw_event_ids=raw,
        raw_control_layout="event-native-always-compress-gist-v1",
    )
    memory = pack_memory(
        store, view, Tokenizer(), max_chunk_tokens=128,
        chunk_overlap=8, max_chunks=100,
    )
    return memory.costs(4)["resident_kv_tokens"] + 8


def test_bare_committed_final_memory_blocks_old_event_resurrection():
    one_at_step_three = _candidate_tokens(7, (1,))
    all_at_step_three = _candidate_tokens(7, (1, 2))
    all_at_step_four = _candidate_tokens(9, (1, 2, 3))
    assert one_at_step_three < all_at_step_three
    assert all_at_step_four <= one_at_step_three
    engine = controller(max_sequence_tokens=one_at_step_three)

    for count, key in ((3, "0"), (5, "1"), (7, "2")):
        prepared = engine.prepare(payload(count, key), ratio=4, max_new_tokens=8)
        assert engine.prepare(payload(count, key), ratio=4, max_new_tokens=8) is prepared
        if key == "2":
            dropped = prepared._store.events[2].event_id
            assert dropped in prepared.memory.view.omitted_event_ids
        engine.commit_memory(prepared, prepared.memory)
        engine.commit_memory(prepared, prepared.memory)

    prepared = engine.prepare(payload(9, "3"), ratio=4, max_new_tokens=8)
    assert dropped in prepared.memory.view.omitted_event_ids
    assert dropped not in prepared.memory.view.gist_event_ids
    assert dropped not in prepared.memory.view.raw_event_ids
    assert prepared.metadata["capacity_gate"]["resident_blocked_event_ids"] == [dropped]
    coverage = prepared.metadata["source_coverage"]
    assert coverage["resident_evicted_event_ids"] == [dropped]
    assert dropped not in coverage["budget_omitted_event_ids"]
    assert coverage["complete_history_coverage"] is False
    assert all_at_step_four <= one_at_step_three

    engine.commit_memory(prepared, prepared.memory)
    # A later recency rule cannot silently turn an evicted source into raw input.
    engine.packing = replace(engine.packing, recent_tool_events=4)
    with pytest.raises(PolicyInputError, match="restore a prior compressed") as error:
        engine.prepare(payload(9, "4"), ratio=4, max_new_tokens=8)
    assert dropped in str(error.value)


def test_uncommitted_or_foreign_final_view_cannot_advance_bare_state():
    engine = controller()
    first = engine.prepare(payload(3, "0"), ratio=4, max_new_tokens=8)
    with pytest.raises(PolicyInputError, match="no committed final memory"):
        engine.prepare(payload(5, "1"), ratio=4, max_new_tokens=8)
    assert engine.prepare(payload(3, "0"), ratio=4, max_new_tokens=8) is first
    second_engine = controller()
    foreign = second_engine.prepare(payload(3, "0"), ratio=4, max_new_tokens=8)
    with pytest.raises(PolicyInputError, match="not a prepared decision view"):
        engine.commit_memory(first, foreign.memory)
    assert engine._sessions["resident-task"].resident is None
    engine.commit_memory(first, first.memory)
    second = engine.prepare(payload(5, "1"), ratio=4, max_new_tokens=8)
    assert second is not first


def test_partial_gist_does_not_repack_missing_chunks_from_archive():
    store = EventStore.from_messages("partial", messages()[:5], benchmark="bfcl")
    event_id = store.events[1].event_id
    chunks = encode_event_chunks(
        store, event_id, Tokenizer(), max_chunk_tokens=128, chunk_overlap=8,
    )
    assert len(chunks) > 1
    view = select_view(store, recent_tool_events=1)
    memory = pack_memory(store, view, Tokenizer(), max_chunk_tokens=128, chunk_overlap=8)
    partial_memory = replace(memory, chunks=(chunks[0],))
    resident = ResidentSourceState.from_memory(store, partial_memory)
    assert resident.admitted_fragments(chunks) == (chunks[0],)
    assert resident.admissible_event_ids(
        store, (event_id,), Tokenizer(), max_chunk_tokens=128,
        chunk_overlap=8,
    ) == ()
    receipt = resident.racer_admission_receipt(replace(memory, chunks=chunks))
    assert len(receipt["newly_admitted_old_fragments"]) == len(chunks) - 1
    assert receipt["newly_admitted_old_raw_source_indices"] == []
