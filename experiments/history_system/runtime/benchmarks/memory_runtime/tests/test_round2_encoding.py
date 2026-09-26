"""CPU contracts for G04/G05 source-bound record encoding."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.encoding_scope import (
    plan_encoding_scope,
    structural_tool_result_record_spans,
)
from history_memory.events import EventStore
from history_memory.packing import encode_event_chunks, encode_scope_chunks
from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller


class CapturingTokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(
        self, messages, *, tools=None, add_generation_prompt=False, **kwargs
    ):
        snapshot = json.loads(json.dumps(messages, ensure_ascii=False))
        self.calls.append(snapshot)
        text = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]


def _tool_event(call_id, name, arguments, result):
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


def _captured_payloads(tokenizer):
    return [json.loads(call[0]["content"]) for call in tokenizer.calls]


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


def test_record_bound_carries_exact_producer_path_and_common_source_text():
    arguments = ' { "account" : "ORIGINAL-ACCOUNT" } '
    result = (
        '{"request_id" : "REQ-7", "records" : '
        '[ {"id":"α","value":"ONE"},\n {"id":"β","value":"TWO"} ], '
        '"next_page" : "p2"}'
    )
    messages = _tool_event("call-7", "list_items", arguments, result)
    store = EventStore.from_messages("round2-bound", messages)
    event_id = store.events[0].event_id
    tokenizer = CapturingTokenizer()

    chunks = encode_scope_chunks(
        store,
        (event_id,),
        tokenizer,
        encoding_scope="record_bound",
        atomic_unit_token_limit=100_000,
    )
    payloads = _captured_payloads(tokenizer)
    result_payloads = [
        payload
        for payload in payloads
        if payload["binding"]["path"][:1] == ["content"]
    ]

    assert len(chunks) == 3  # one call-argument record plus two result records
    assert len(result_payloads) == 2
    assert [payload["record_text"] for payload in result_payloads] == [
        '{"id":"α","value":"ONE"}',
        '{"id":"β","value":"TWO"}',
    ]
    for index, payload in enumerate(result_payloads):
        binding = payload["binding"]
        assert binding["path"] == ["content", "records", index]
        assert binding["producer_call"] == messages[0]["tool_calls"][0]
        assert binding["producer_call"]["function"]["arguments"] == arguments
        assert binding["message_header"] == {
            "role": "tool",
            "tool_call_id": "call-7",
        }
        assert binding["common_header"] == [
            {"path": ["content", "request_id"], "text": '"REQ-7"'},
            {"path": ["content", "next_page"], "text": '"p2"'},
        ]
        assert payload["record_text"] in result
    assert all(chunk.source_indices == (0, 1) for chunk in chunks[1:])
    assert "TWO" not in json.dumps(result_payloads[0], ensure_ascii=False)
    assert "ONE" not in json.dumps(result_payloads[1], ensure_ascii=False)


def test_record_bound_structural_uses_records_only_for_multi_record_tool_results():
    messages = [
        {"role": "assistant", "content": "ordinary historical text"},
        *_tool_event(
            "single",
            "get_item",
            '{"id":"ONLY"}',
            '{"records":[{"id":"ONLY"}],"request_id":"S"}',
        ),
        *_tool_event(
            "multi",
            "list_items",
            '{"owner":"ORIGINAL"}',
            '{"records":[{"id":"A"},{"id":"B"}],"request_id":"M"}',
        ),
    ]
    store = EventStore.from_messages("round2-structural", messages)
    event_ids = tuple(event.event_id for event in store.events)
    plan = plan_encoding_scope(store, event_ids, "record_bound_structural")

    assert plan.event_groups == tuple((event_id,) for event_id in event_ids)
    assert plan.pending_event_ids == ()
    assert structural_tool_result_record_spans(store, event_ids[1]) == ()
    assert len(structural_tool_result_record_spans(store, event_ids[2])) == 2

    tokenizer = CapturingTokenizer()
    chunks = encode_scope_chunks(
        store,
        event_ids,
        tokenizer,
        encoding_scope="record_bound_structural",
        max_chunk_tokens=10_000,
        chunk_overlap=0,
        atomic_unit_token_limit=100_000,
        event_groups=plan.event_groups,
    )
    fallback = [chunk for chunk in chunks if chunk.event_id in event_ids[:2]]
    bound = [chunk for chunk in chunks if ":record_bound_structural:" in chunk.event_id]

    assert fallback == [
        *encode_event_chunks(
            store, event_ids[0], CapturingTokenizer(), max_chunk_tokens=10_000,
            chunk_overlap=0,
        ),
        *encode_event_chunks(
            store, event_ids[1], CapturingTokenizer(), max_chunk_tokens=10_000,
            chunk_overlap=0,
        ),
    ]
    assert len(bound) == 2
    bound_payloads = [
        payload
        for payload in _captured_payloads(tokenizer)
        if payload.get("type") == "history_record_bound"
    ]
    assert [payload["record_text"] for payload in bound_payloads] == [
        '{"id":"A"}',
        '{"id":"B"}',
    ]
    assert all(
        payload["binding"]["producer_call"] == messages[3]["tool_calls"][0]
        for payload in bound_payloads
    )
    assert all(chunk.source_indices == (3, 4) for chunk in bound)


def test_structural_scope_falls_back_for_mixed_parallel_results():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "many",
                    "type": "function",
                    "function": {"name": "list_items", "arguments": "{}"},
                },
                {
                    "id": "one",
                    "type": "function",
                    "function": {"name": "get_item", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "many",
            "content": '[{"id":1},{"id":2}]',
        },
        {"role": "tool", "tool_call_id": "one", "content": '{"id":3}'},
    ]
    store = EventStore.from_messages("round2-mixed", messages)
    event_id = store.events[0].event_id
    assert structural_tool_result_record_spans(store, event_id) == ()

    actual = encode_scope_chunks(
        store,
        (event_id,),
        CapturingTokenizer(),
        encoding_scope="record_bound_structural",
        max_chunk_tokens=10_000,
        chunk_overlap=0,
    )
    expected = encode_event_chunks(
        store,
        event_id,
        CapturingTokenizer(),
        max_chunk_tokens=10_000,
        chunk_overlap=0,
    )
    assert actual == expected


def test_record_bound_falls_back_to_current_without_making_source_raw():
    messages = [{"role": "assistant", "content": '{"records":[{"id":1}]'}]
    store = EventStore.from_messages("round2-bound-fallback", messages)
    event_id = store.events[0].event_id
    plan = plan_encoding_scope(store, (event_id,), "record_bound")

    assert plan.event_groups == ((event_id,),)
    assert plan.pending_event_ids == ()
    actual = encode_scope_chunks(
        store,
        (event_id,),
        CapturingTokenizer(),
        encoding_scope="record_bound",
        max_chunk_tokens=10_000,
        chunk_overlap=0,
    )
    expected = encode_event_chunks(
        store,
        event_id,
        CapturingTokenizer(),
        max_chunk_tokens=10_000,
        chunk_overlap=0,
    )
    assert actual == expected


def test_record_bound_uses_only_event_snapshot_and_ignores_future_reused_call_id():
    old_messages = _tool_event(
        "reused",
        "list_items",
        '{"owner":"OLD"}',
        '{"records":[{"id":"A"},{"id":"B"}],"request_id":"OLD"}',
    )
    prefix = EventStore.from_messages("round2-future", old_messages)
    before_json = tuple(message.json_text for message in prefix.messages)
    prefix_tokenizer = CapturingTokenizer()
    prefix_chunks = encode_scope_chunks(
        prefix,
        (prefix.events[0].event_id,),
        prefix_tokenizer,
        encoding_scope="record_bound",
        atomic_unit_token_limit=100_000,
    )

    future_messages = [
        *old_messages,
        *_tool_event(
            "reused",
            "list_items",
            '{"owner":"FUTURE"}',
            '{"records":[{"id":"X"},{"id":"Y"}],"request_id":"FUTURE"}',
        ),
    ]
    extended = EventStore.from_messages("round2-future", future_messages)
    extended_tokenizer = CapturingTokenizer()
    extended_chunks = encode_scope_chunks(
        extended,
        (extended.events[0].event_id,),
        extended_tokenizer,
        encoding_scope="record_bound",
        atomic_unit_token_limit=100_000,
    )

    assert extended_chunks == prefix_chunks
    assert tuple(message.json_text for message in prefix.messages) == before_json
    old_result_payloads = [
        payload
        for payload in _captured_payloads(extended_tokenizer)
        if payload["binding"]["path"][:1] == ["content"]
    ]
    assert old_result_payloads
    assert all(
        payload["binding"]["producer_call"]["function"]["arguments"]
        == '{"owner":"OLD"}'
        for payload in old_result_payloads
    )
    assert "FUTURE" not in json.dumps(old_result_payloads, ensure_ascii=False)


def test_new_scopes_report_their_actual_atomic_packing_mode():
    messages = [
        {"role": "assistant", "content": "old paragraph"},
        {"role": "user", "content": "current request"},
    ]
    labels = {
        "record_bound": "source_bound_record_or_current_event",
        "record_bound_structural": (
            "source_bound_multi_record_tool_result_or_current_event"
        ),
    }
    for scope, expected_label in labels.items():
        controller = EventNativeS0Controller(
            CapturingTokenizer(),
            packing=_packing(),
            policy=_policy(),
            model_context=100_000,
        )
        controller.encoding_scope = scope
        prepared = controller.prepare(
            {
                "session_id": f"round2-label-{scope}",
                "decision_key": "d0",
                "messages": messages,
                "tools": [],
            },
            ratio=8,
            max_new_tokens=8,
        )
        assert prepared.metadata["atomic_packing_unit"] == expected_label
        assert prepared.metadata["pending_encoding_event_ids"] == []
