"""CPU contracts for G encoder boundaries and provenance."""

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from history_memory.encoding_scope import extract_record_spans, plan_encoding_scope
from history_memory.events import EventStore
from history_memory.packing import (
    EncodingScopeCapacityError,
    MemoryView,
    encode_scope_chunks,
    pack_memory,
)


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tools=None, add_generation_prompt=False, **kwargs
    ):
        text = "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>" if tools else ""
        for message in messages:
            text += (
                "<"
                + message["role"]
                + ">"
                + json.dumps(message, sort_keys=True, ensure_ascii=False)
                + "</end>"
            )
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]


def _packing():
    return {
        "ratios": [8],
        "recent_tool_events": 1,
        "max_chunk_tokens": 768,
        "chunk_overlap": 64,
        "max_chunks": 48,
        "max_encoder_tokens": 100_000,
        "max_system_tokens": 20_000,
        "max_workspace_tokens": 50_000,
        "max_target_tokens": 32,
        "max_sequence_tokens": 100_000,
    }


def _policy():
    return {
        "mode": "persistent",
        "history_budget_bytes": 1_000_000,
        "workspace_budget_bytes": 1_000_000,
        "lease_decisions": 0,
        "max_retrieved_events": 2,
        "kv_bytes_per_token": 1,
        "source_commit": POLICY_SOURCE_COMMIT,
        "history_budget_definition": HISTORY_BUDGET_DEFINITION,
        "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
        "current_input_baseline": CURRENT_INPUT_BASELINE,
    }


def _payload(messages, key):
    return {
        "session_id": "g-scope/session",
        "decision_key": key,
        "messages": copy.deepcopy(messages),
        "tools": [],
    }


def test_record_span_extraction_preserves_exact_json_and_paragraph_ranges():
    source = ' {"records": [ {"name":"α"}, {"name":"β"} ], "next":"p2"} '
    spans = extract_record_spans(source)
    assert [span.text for span in spans] == ['{"name":"α"}', '{"name":"β"}']
    assert [span.field_path for span in spans] == [("records", 0), ("records", 1)]
    assert all(source[span.char_start : span.char_end] == span.text for span in spans)

    prose = " First paragraph. \n\nSecond paragraph with emoji 🧪. "
    paragraphs = extract_record_spans(prose)
    assert [span.text for span in paragraphs] == [
        "First paragraph.",
        "Second paragraph with emoji 🧪.",
    ]
    assert all(prose[span.char_start : span.char_end] == span.text for span in paragraphs)
    assert extract_record_spans('{"ok":true} trailing') == ()


def test_event_scope_is_one_atomic_encoder_call_beyond_legacy_chunk_size():
    messages = [{"role": "assistant", "content": "x" * 900}]
    store = EventStore.from_messages("atomic-event", messages)
    event_id = store.events[0].event_id
    current = encode_scope_chunks(
        store,
        (event_id,),
        Tokenizer(),
        encoding_scope="current",
        max_chunk_tokens=200,
        chunk_overlap=20,
    )
    atomic = encode_scope_chunks(
        store,
        (event_id,),
        Tokenizer(),
        encoding_scope="event",
        max_chunk_tokens=200,
        chunk_overlap=20,
        atomic_unit_token_limit=5_000,
    )

    assert len(current) > 1
    assert len(atomic) == 1
    assert len(atomic[0].token_ids) > 200
    assert atomic[0].source_indices == (0,)
    assert len(atomic[0].token_ids) == atomic[0].source_token_end
    with pytest.raises(EncodingScopeCapacityError, match="Atomic event encoder unit"):
        encode_scope_chunks(
            store,
            (event_id,),
            Tokenizer(),
            encoding_scope="event",
            atomic_unit_token_limit=200,
        )


def test_record_scope_emits_atomic_unique_units_with_parent_context():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "list_items",
                    "arguments": '{"account":"A"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"records":[{"id":1},{"id":2}],"next_page":"p2","count":2}',
        },
    ]
    store = EventStore.from_messages("record-event", messages)
    event_id = store.events[0].event_id
    chunks = encode_scope_chunks(
        store,
        (event_id,),
        Tokenizer(),
        encoding_scope="record",
        atomic_unit_token_limit=10_000,
    )
    rendered = ["".join(chr(token) for token in chunk.token_ids) for chunk in chunks]

    assert len(chunks) == 3
    assert len({chunk.event_id for chunk in chunks}) == len(chunks)
    assert all(chunk.part_index == 0 for chunk in chunks)
    assert all(chunk.source_token_start == 0 for chunk in chunks)
    assert any("call-1" in text and "list_items" in text for text in rendered)
    result_records = [text for text in rendered if "next_page" in text]
    assert len(result_records) == 2
    assert all("p2" in text and "count" in text for text in result_records)
    assert {index for chunk in chunks for index in chunk.source_indices} == {0, 1}


def test_record_scope_keeps_whole_event_raw_when_any_container_is_unsplittable():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-broken",
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": '{"account":"A"}',
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": "call-broken",
            "content": '{"records":[{"id":1}]',
        },
        {"role": "user", "content": "Continue."},
    ]
    store = EventStore.from_messages("g-scope/session", messages)
    broken_event_id = store.events[0].event_id
    controller = EventNativeS0Controller(
        Tokenizer(), packing=_packing(), policy=_policy(), model_context=100_000
    )
    controller.encoding_scope = "record"
    prepared = controller.prepare(_payload(messages, "broken"), ratio=8, max_new_tokens=8)

    assert prepared.eligible_chunks == ()
    assert prepared.metadata["pending_encoding_event_ids"] == [broken_event_id]
    assert prepared.metadata["eligible_extraction"]["pending_raw_event_ids"] == [
        broken_event_id
    ]
    assert broken_event_id in prepared.memory.view.raw_event_ids
    assert set(store.event(broken_event_id).source_indices) <= set(
        prepared.memory.raw_source_indices
    )


def test_adjacent_pair_pending_is_raw_and_prior_encoder_keys_stay_stable():
    first_prefix = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "assistant", "content": "pending"},
        {"role": "user", "content": "current"},
    ]
    controller = EventNativeS0Controller(
        Tokenizer(), packing=_packing(), policy=_policy(), model_context=100_000
    )
    controller.encoding_scope = "adjacent_pair"
    first = controller.prepare(_payload(first_prefix, "d1"), ratio=8, max_new_tokens=8)
    first_store = EventStore.from_messages("g-scope/session", first_prefix)
    first_plan = plan_encoding_scope(
        first_store,
        first.metadata["eligible_extraction"]["eligible_event_ids"]
        + first.metadata["eligible_extraction"]["pending_raw_event_ids"],
        "adjacent_pair",
    )
    pending_id = first_plan.pending_event_ids[0]

    assert len(first.eligible_chunks) == 2
    assert pending_id in first.memory.view.raw_event_ids
    assert pending_id not in first.memory.view.gist_event_ids
    assert set(first.memory.raw_source_indices) >= set(
        first_store.event(pending_id).source_indices
    )
    assert {
        chunk.encoding_key(parameter_version=0, ratio=8)
        for chunk in first.memory.chunks
    } <= {
        chunk.encoding_key(parameter_version=0, ratio=8)
        for chunk in first.eligible_chunks
    }

    next_prefix = [
        *first_prefix,
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "next"},
    ]
    second = controller.prepare(_payload(next_prefix, "d2"), ratio=8, max_new_tokens=8)
    first_keys = [
        chunk.encoding_key(parameter_version=0, ratio=8)
        for chunk in first.eligible_chunks
    ]
    second_keys = [
        chunk.encoding_key(parameter_version=0, ratio=8)
        for chunk in second.eligible_chunks
    ]
    assert second_keys[:2] == first_keys
    assert len(second_keys) == 3
    assert second.metadata["pending_encoding_event_ids"]


def test_current_scope_preserves_default_packing_behavior():
    messages = [
        {"role": "assistant", "content": "old " + "x" * 900},
        {"role": "user", "content": "now"},
    ]
    store = EventStore.from_messages("current", messages)
    view = MemoryView(
        gist_event_ids=(store.events[0].event_id,),
        raw_event_ids=(store.events[1].event_id,),
    )
    default = pack_memory(
        store, view, Tokenizer(), max_chunk_tokens=200, chunk_overlap=20
    )
    explicit = pack_memory(
        store,
        view,
        Tokenizer(),
        max_chunk_tokens=200,
        chunk_overlap=20,
        encoding_scope="current",
    )
    assert explicit == default


def test_protected_recovery_messages_are_reserved_once_and_reported():
    messages = [
        {"role": "assistant", "content": "historical value"},
        {"role": "user", "content": "current request"},
    ]
    lease_a = {"role": "user", "content": "protected lease A"}
    lease_b = {"role": "user", "content": "protected lease B"}
    bridge = {"role": "user", "content": "native bridge"}
    controller = EventNativeS0Controller(
        Tokenizer(), packing=_packing(), policy=_policy(), model_context=100_000
    )
    controller.protected_recovery_messages = (lease_a, lease_b)

    assert controller._merge_protected_derived((bridge, lease_a)) == (
        lease_a,
        lease_b,
        bridge,
    )

    prepared = controller.prepare(_payload(messages, "protected"), ratio=8, max_new_tokens=8)
    assert prepared.metadata["derived_workspace_prefix_messages"] == [lease_a, lease_b]
    rendered = "".join(chr(token) for token in prepared.memory.workspace_input_ids)
    assert rendered.count("protected lease A") == 1
    assert rendered.count("protected lease B") == 1
