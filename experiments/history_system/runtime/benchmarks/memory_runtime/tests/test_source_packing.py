import json
import sys
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.event_native_tool import ToolRegionController
from history_memory.events import EventStore
from history_memory.packing import PackingBudgetError
from history_memory.source_packing import (
    encode_source_chunks,
    make_source_view,
    pack_source_memory,
    source_encoder_messages,
)


class Tokenizer:
    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False,
                            tokenize=True, **kwargs):
        prefix = "<tools>" + json.dumps(tools) if tools else ""
        text = prefix + "".join(
            f"<{message['role']}>" + json.dumps(message, sort_keys=True)
            for message in messages
        )
        if add_generation_prompt:
            text += "<assistant>"
        return tuple(map(ord, text)) if tokenize else text

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def __call__(self, text, **kwargs):
        return {"input_ids": tuple(map(ord, text)),
                "offset_mapping": tuple((index, index + 1) for index in range(len(text)))}


def _call(call_id, name, arguments):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": arguments,
    }}


def _store():
    return EventStore.from_messages("source", [
        {"role": "system", "content": "Follow the tool protocol."},
        {"role": "user", "content": "Old question"},
        {"role": "assistant", "content": None, "tool_calls": [
            _call("reused", "search", '{"query":"' + "long " * 100 + '"}'),
            _call("other", "lookup", '{"key":"x"}'),
        ]},
        {"role": "tool", "tool_call_id": "other", "name": "lookup", "content": "other result"},
        {"role": "tool", "tool_call_id": "reused", "name": "search", "content": "answer " * 60},
        {"role": "assistant", "content": None, "tool_calls": [
            _call("reused", "final_lookup", '{"key":"second"}')
        ]},
        {"role": "tool", "tool_call_id": "reused", "name": "final_lookup", "content": "second answer"},
        {"role": "user", "content": "Current question"},
    ])


def test_partition_and_event_provenance_do_not_claim_partial_raw_events():
    store = _store()
    view = make_source_view(store, (0, 3, 4, 6, 7), (2, 5), (0, 7))
    assert view.omitted_source_indices == (1,)
    assert view.raw_event_ids == (
        store.events[0].event_id, store.events[-1].event_id,
    )
    assert view.gist_event_ids == (store.events[2].event_id, store.events[3].event_id)
    assert view.omitted_event_ids == (store.events[1].event_id,)
    assert view.partial_event_ids == (store.events[2].event_id, store.events[3].event_id)
    assert view.mandatory_raw_event_ids == (
        store.events[0].event_id, store.events[-1].event_id,
    )
    view.validate(store)
    with pytest.raises(ValueError, match="partition"):
        replace(view, raw_event_ids=view.raw_event_ids + (store.events[2].event_id,)).validate(store)
    with pytest.raises(ValueError, match="disjoint"):
        make_source_view(store, (0, 2, 7), (2,))
    with pytest.raises(ValueError, match="Mandatory"):
        make_source_view(store, (0, 7), (2,), (2,))
    with pytest.raises(ValueError, match="strictly increasing"):
        make_source_view(store, (7, 0), (2,))
    with pytest.raises(ValueError, match="visible source prefix"):
        make_source_view(store, (0, 7, 8), (2,))


def test_incomplete_and_instruction_sources_must_be_raw():
    pending = EventStore.from_messages("pending", [
        {"role": "system", "content": "Rules"},
        {"role": "assistant", "content": None, "tool_calls": [
            _call("call", "search", '{}')
        ]},
    ])
    with pytest.raises(ValueError, match="remain raw"):
        make_source_view(pending, (0,), (1,))
    with pytest.raises(ValueError, match="remain raw"):
        make_source_view(pending, (0,), ())
    with pytest.raises(ValueError, match="remain raw"):
        make_source_view(pending, (1,), ())
    view = make_source_view(pending, (0, 1), ())
    assert view.raw_event_ids == tuple(event.event_id for event in pending.events)


def test_source_envelope_keeps_complete_message_and_correct_producer_binding():
    store = _store()
    producer = json.loads(source_encoder_messages(store, 4)[0]["content"])
    assert producer["type"] == "history_source"
    assert producer["source_index"] == 4
    assert producer["message"]["content"] == "answer " * 60
    assert producer["producer"] == {
        "source_index": 2, "tool_call_id": "reused", "tool_name": "search",
    }
    assert "arguments" not in producer["producer"]
    other = json.loads(source_encoder_messages(store, 3)[0]["content"])
    assert other["producer"] == {
        "source_index": 2, "tool_call_id": "other", "tool_name": "lookup",
    }
    reused = json.loads(source_encoder_messages(store, 6)[0]["content"])
    assert reused["producer"] == {
        "source_index": 5, "tool_call_id": "reused", "tool_name": "final_lookup",
    }
    assert json.loads(source_encoder_messages(store, 2)[0]["content"])["message"]["tool_calls"][0]["function"]["arguments"]["query"] == "long " * 100


def test_result_only_gist_retains_its_raw_producer_binding():
    store = _store()
    tokenizer = Tokenizer()
    view = make_source_view(store, (0, 2, 5, 7), (4, 6), (0, 7))
    memory = pack_source_memory(store, view, tokenizer, max_chunk_tokens=100, chunk_overlap=9)
    workspace = "".join(map(chr, memory.workspace_input_ids))
    assert "second answer" not in workspace
    assert "final_lookup" in workspace
    gist_sources = {index for chunk in memory.chunks for index in chunk.source_indices}
    assert gist_sources == {4, 6}
    result = json.loads(source_encoder_messages(store, 6)[0]["content"])
    assert result["producer"]["source_index"] == 5
    assert result["producer"]["tool_call_id"] == "reused"
    assert result["producer"]["tool_name"] == "final_lookup"


def test_chunks_cover_every_selected_source_and_repack_is_deterministic():
    store = _store()
    tokenizer = Tokenizer()
    chunks = encode_source_chunks(store, (2, 4, 5), tokenizer, max_chunk_tokens=100, chunk_overlap=9)
    assert {chunk.source_indices for chunk in chunks} == {(2,), (4,), (5,)}
    assert [chunk.source_indices[0] for chunk in chunks] == sorted(
        chunk.source_indices[0] for chunk in chunks
    )
    for source_index in (2, 4, 5):
        parts = [chunk for chunk in chunks if chunk.source_indices == (source_index,)]
        expected = tokenizer.apply_chat_template(source_encoder_messages(store, source_index))
        assert all(chunk.event_id == f"{next(event.event_id for event in store.events if source_index in event.source_indices)}:source:{source_index}" for chunk in parts)
        assert [chunk.part_index for chunk in parts] == list(range(len(parts)))
        assert parts[0].source_token_start == 0
        assert parts[-1].source_token_end == len(expected)
        assert all(part.token_ids == expected[part.source_token_start:part.source_token_end] for part in parts)
        assert all(right.source_token_start == left.source_token_end - 9 for left, right in pairwise(parts))
    assert chunks == encode_source_chunks(store, (2, 4, 5), tokenizer, max_chunk_tokens=100, chunk_overlap=9)
    with pytest.raises(ValueError, match="Only completed non-instruction"):
        encode_source_chunks(EventStore.from_messages("p", [
            {"role": "assistant", "content": None, "tool_calls": [_call("x", "f", '{}')]}
        ]), (0,), tokenizer)


def test_packing_preserves_prefix_positions_and_full_chunk_budget():
    store = _store()
    tokenizer = Tokenizer()
    view = make_source_view(store, (0, 3, 4, 6, 7), (2, 5), (0, 7))
    tools = [{"type": "function", "function": {"name": "search"}}]
    kwargs = {
        "tools": tools, "max_chunk_tokens": 100, "chunk_overlap": 9,
        "derived_workspace_prefix_messages": ({"role": "user", "content": "Derived observation"},),
    }
    memory = pack_source_memory(store, view, tokenizer, **kwargs)
    assert memory.view == view
    assert memory.raw_source_indices == view.raw_source_indices
    assert memory.chunks == encode_source_chunks(store, view.gist_source_indices, tokenizer, max_chunk_tokens=100, chunk_overlap=9)
    full = "".join(map(chr, memory.system_input_ids + memory.workspace_input_ids))
    assert full.startswith("<tools>")
    assert full.index("Follow the tool protocol.") < full.index("Derived observation")
    assert full.index("Derived observation") < full.index("other result")
    assert "Current question" in full
    assert "Old question" not in full
    assert "long long" not in full
    assert memory.workspace_position_start == len(memory.system_input_ids) + sum(len(chunk.token_ids) for chunk in memory.chunks)
    assert memory.gist_layout(4) == pack_source_memory(store, view, tokenizer, **kwargs).gist_layout(4)
    assert memory.costs(4)["encoder_overlap_tokens"] > 0
    with pytest.raises(PackingBudgetError, match="Complete sources"):
        pack_source_memory(store, view, tokenizer, **kwargs, max_chunks=len(memory.chunks) - 1)
    with pytest.raises(PackingBudgetError, match="complete raw workspace"):
        pack_source_memory(store, view, tokenizer, **kwargs, max_raw_tokens=len(memory.system_input_ids + memory.workspace_input_ids) - 1)


def test_raw_tool_wrapper_aligns_partial_raw_result_and_derived_prefix():
    messages = [
        {"role": "system", "content": "Rules"},
        {"role": "assistant", "content": None, "tool_calls": [_call("a", "search", '{}')]},
        {"role": "tool", "tool_call_id": "a", "content": "DOC: result"},
        {"role": "user", "content": "Current question"},
    ]
    payload = {"session_id": "tool-source", "messages": messages, "tools": []}
    store = EventStore.from_messages(payload["session_id"], messages)
    view = make_source_view(store, (0, 2, 3), (1,), (0, 3))
    assert store.events[1].event_id in view.partial_event_ids
    derived = ({"role": "user", "content": "Derived observation"},)
    tokenizer = Tokenizer()
    memory = pack_source_memory(
        store, view, tokenizer, max_chunk_tokens=100, chunk_overlap=9,
        derived_workspace_prefix_messages=derived,
    )
    recorded = []

    def repair(ids, *, span_start, span_end, method, target_tokens):
        recorded.append(("".join(map(chr, ids[span_start:span_end])), method))
        return {"key_hash": "repair", "token_len": target_tokens}

    wrapper = ToolRegionController.__new__(ToolRegionController)
    wrapper.tokenizer = tokenizer
    wrapper.spec = SimpleNamespace(encoder="h2o", ratio=2)
    wrapper.generator = SimpleNamespace(repair_tool_span=repair)
    wrapper.tool_budget_tokens = None
    plan = SimpleNamespace(
        source_spans=(SimpleNamespace(message_index=2, start=0, end=3,
                                      text="DOC", source="source-document"),),
        compressed_source_indices=(0,),
    )
    augmented = wrapper._raw_tool_memory(
        memory, plan, payload, derived_messages=derived)
    assert recorded == [("DOC", "h2o")]
    assert len(augmented.raw_tool_segments) == 1
    assert augmented.raw_tool_segments[0]["token_len"] == 2

    repeated = ({"role": "user", "content": "Current question"},)
    repeated_memory = pack_source_memory(
        store, view, tokenizer, max_chunk_tokens=100, chunk_overlap=9,
        derived_workspace_prefix_messages=repeated,
    )
    user_plan = SimpleNamespace(
        source_spans=(SimpleNamespace(message_index=3, start=0, end=7,
                                      text="Current", source="current-user"),),
        compressed_source_indices=(0,),
    )
    repeated_tool = wrapper._raw_tool_memory(
        repeated_memory, user_plan, payload, derived_messages=repeated)
    logical = tokenizer.decode(
        repeated_memory.system_input_ids
        + tuple(token for chunk in repeated_memory.chunks for token in chunk.token_ids)
        + repeated_memory.workspace_input_ids
    )
    assert repeated_tool.raw_tool_segments[0]["token_start"] == logical.rfind("Current")
