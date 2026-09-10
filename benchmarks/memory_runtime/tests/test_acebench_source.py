"""CPU contracts for ACEBench textual actions and execution receipts."""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.memory_runtime.acebench_source import (
    ACE_OPAQUE_EVENT_KIND,
    build_ace_event_store,
    parse_ace_draft,
)


def _receipt(
    *,
    execution_index: int = 3,
    agent_history_index: int = 1,
    decoded_calls=None,
    count: int | None = 1,
    decode_status: str = "ok",
    executor_status: str = "returned",
    shape: str | None = "list",
):
    return {
        "execution_message_index": execution_index,
        "version": "acebench-execution-receipt-v1",
        "agent_history_index": agent_history_index,
        "decode_status": decode_status,
        "decoded_calls": (
            ["Reserve(city='Paris', nights=2)"]
            if decoded_calls is None and decode_status == "ok"
            else decoded_calls
        ),
        "executor_status": executor_status,
        "executor_return_shape": shape,
        "executor_return_count": count,
    }


def _messages(action="[Reserve(city='Paris', nights=2)]", result=None):
    return [
        {"role": "system", "content": "Use the supplied APIs."},
        {"role": "user", "content": "Book Paris."},
        {"role": "assistant", "content": action},
        {
            "role": "tool",
            "content": json.dumps(
                result if result is not None else [{"reservation_id": "R-7"}],
                ensure_ascii=False,
            ),
        },
        {"role": "user", "content": "Cancel reservation R-7."},
    ]


def _source(receipt=None):
    return {
        "version": "acebench-text-actions-v1",
        "receipts": [receipt or _receipt()],
    }


def test_parser_preserves_original_text_and_supported_value_order() -> None:
    text = "[First(s='x', ok=True, missing=None, n=-3, f=-1.25, xs=['a', 2]),Second()]"
    draft = parse_ace_draft(text, call_id_prefix="d4_r0")

    assert draft.text == draft.content == text
    assert draft.reasoning_content is None
    assert draft.status == "tool_calls"
    assert [call["id"] for call in draft.tool_calls] == ["d4_r0_0", "d4_r0_1"]
    assert [call["function"]["name"] for call in draft.tool_calls] == [
        "First",
        "Second",
    ]
    assert list(json.loads(draft.tool_calls[0]["function"]["arguments"])) == [
        "s",
        "ok",
        "missing",
        "n",
        "f",
        "xs",
    ]


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("[]", "empty_call_list"),
        ("[Call(1)]", "positional_argument"),
        ("[Call(*xs)]", "starred_argument"),
        ("[Call(**xs)]", "keyword_unpack"),
        ("[Call(x=1, x=2)]", "duplicate_keyword"),
        ("[obj.Call()]", "callee_not_name"),
        ("[Call(x={'a': 1})]", "dict_value"),
        ("[Call(x=(1, 2))]", "tuple_value"),
        ("[Call(x=name)]", "name_value"),
        ("[Call(x=Other())]", "nested_call"),
        ("[Call(x=1 + 2)]", "binary_operator"),
        ("[Call(x=lambda: 1)]", "lambda_value"),
        ("[Call(x=xs[0])]", "subscript_value"),
        ("[Call(x={1, 2})]", "set_value"),
        ("[Call(x=[v for v in xs])]", "comprehension_value"),
        ("[Call(x=b'bytes')]", "bytes_value"),
        ("[Call(x=...)]", "ellipsis_value"),
        ("[Call(x=+1)]", "unary_operator"),
        ("[Call(x=1e309)]", "non_finite_number"),
    ],
)
def test_valid_ast_outside_supported_grammar_is_atomic_malformed(text, code) -> None:
    draft = parse_ace_draft(text, call_id_prefix="draft")
    assert draft.status == "malformed"
    assert draft.reason == f"unsupported_ace_grammar:{code}"
    assert draft.tool_calls == ()
    assert draft.text == draft.content == text


def test_routing_subset_does_not_promote_prose_or_leading_space_to_action() -> None:
    for text in ("", "finish conversation", "Need another detail.", " [Call(x=1)]"):
        draft = parse_ace_draft(text, call_id_prefix="draft")
        assert draft.status == "text"
        assert draft.tool_calls == ()
        assert draft.text == draft.content == text
    for text in ("[Call(x=1)", "[Call(\n x=1)]", "[Call()] trailing"):
        draft = parse_ace_draft(text, call_id_prefix="draft")
        assert draft.status == "malformed"
        assert draft.tool_calls == ()


def test_receipt_backed_batch_is_one_complete_tool_event_without_rewriting() -> None:
    messages = _messages()
    store = build_ace_event_store("ace/session", messages, _source())

    assert [message.to_dict() for message in store.messages] == messages
    event = store.event("ace/session:m2")
    assert event.kind == "tool_event"
    assert event.complete is True
    assert event.source_indices == (2, 3)
    assert event.tool_call_ids == ("ace/session:m2_0",)
    assert [message.to_dict() for message in store.event_messages(event.event_id)] == [
        messages[2],
        messages[3],
    ]


@pytest.mark.parametrize(
    "receipt",
    [
        _receipt(decode_status="error", decoded_calls=None,
                 executor_status="not_called", shape=None, count=None),
        _receipt(decoded_calls=["Reserve(city='Rome', nights=2)"]),
        _receipt(count=2),
        _receipt(shape="non_list", count=None),
    ],
)
def test_valid_but_unverified_execution_receipts_stay_opaque(receipt) -> None:
    store = build_ace_event_store("ace/session", _messages(), _source(receipt))
    event = store.event("ace/session:m2")
    assert event.kind == ACE_OPAQUE_EVENT_KIND
    assert event.complete is False
    assert event.source_indices == (2, 3)
    assert event.tool_call_ids == ()
    assert event.missing_tool_call_ids == ()


def test_action_without_observation_is_opaque_then_expands_at_same_event_id() -> None:
    prefix = _messages()[:3]
    initial = build_ace_event_store(
        "ace/session",
        prefix,
        {"version": "acebench-text-actions-v1", "receipts": []},
    )
    completed = build_ace_event_store("ace/session", _messages(), _source())

    assert initial.event("ace/session:m2").kind == ACE_OPAQUE_EVENT_KIND
    assert initial.event("ace/session:m2").complete is False
    assert completed.event("ace/session:m2").kind == "tool_event"
    assert completed.event("ace/session:m2").complete is True


def test_action_without_receipt_stays_opaque_when_later_user_arrives() -> None:
    messages = _messages()[:3] + [{"role": "user", "content": "Try something else."}]
    store = build_ace_event_store(
        "ace/session",
        messages,
        {"version": "acebench-text-actions-v1", "receipts": []},
    )
    assert store.event("ace/session:m2").kind == ACE_OPAQUE_EVENT_KIND
    assert store.event("ace/session:m2").source_indices == (2,)
    assert store.event("ace/session:m3").kind == "user"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda source: source.update(extra=True),
        lambda source: source.update(version="wrong"),
        lambda source: source["receipts"][0].update(extra=True),
        lambda source: source["receipts"][0].update(execution_message_index=4),
        lambda source: source["receipts"][0].update(agent_history_index=0),
        lambda source: source["receipts"][0].update(decoded_calls=[1]),
        lambda source: source["receipts"][0].update(executor_return_count=True),
    ],
)
def test_source_and_receipt_schema_are_strict(mutate) -> None:
    source = _source()
    mutate(source)
    with pytest.raises(ValueError):
        build_ace_event_store("ace/session", _messages(), source)


def test_tool_observation_without_receipt_is_rejected() -> None:
    with pytest.raises(ValueError, match="require execution receipts"):
        build_ace_event_store(
            "ace/session",
            _messages(),
            {"version": "acebench-text-actions-v1", "receipts": []},
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda messages: messages[2].update(tool_calls=[]),
        lambda messages: messages[2].update(function_call={"name": "Reserve"}),
        lambda messages: messages[1].update(metadata="privileged"),
        lambda messages: messages[1].update(role="developer"),
        lambda messages: messages[2].update(content=None),
        lambda messages: messages[1].update(tool_call_id="mixed-protocol"),
    ],
)
def test_mixed_native_or_non_ace_message_protocol_is_rejected(mutation) -> None:
    messages = _messages()
    mutation(messages)
    with pytest.raises(ValueError):
        build_ace_event_store("ace/session", messages, _source())


def test_original_tool_call_id_is_preserved_but_does_not_bind_completion() -> None:
    messages = _messages()
    messages[3]["tool_call_id"] = "upstream-metadata"
    store = build_ace_event_store("ace/session", messages, _source())
    assert store.messages[3].to_dict()["tool_call_id"] == "upstream-metadata"
    assert store.event("ace/session:m2").complete is True


def test_receipt_never_invents_completion_from_only_matching_count() -> None:
    source = _source()
    source["receipts"][0]["decoded_calls"] = [
        "Reserve(nights=2, city='Paris')"
    ]
    store = build_ace_event_store("ace/session", _messages(), source)
    assert store.event("ace/session:m2").kind == ACE_OPAQUE_EVENT_KIND
    assert store.event("ace/session:m2").complete is False


@pytest.mark.parametrize("decoded", ["Flag(value=1)", "Flag(value=1.0)"])
def test_receipt_boolean_and_number_arguments_are_not_equal(decoded) -> None:
    messages = _messages(
        action="[Flag(value=True)]", result=[{"accepted": True}]
    )
    receipt = _receipt(decoded_calls=[decoded])
    store = build_ace_event_store("ace/session", messages, _source(receipt))
    assert store.event("ace/session:m2").kind == ACE_OPAQUE_EVENT_KIND
    assert store.event("ace/session:m2").complete is False


def test_inputs_are_snapshotted() -> None:
    messages = _messages()
    source = _source()
    store = build_ace_event_store("ace/session", messages, source)
    messages[2]["content"] = "changed"
    source["receipts"][0]["decoded_calls"][0] = "Changed()"
    assert store.messages[2].to_dict()["content"] == "[Reserve(city='Paris', nights=2)]"
    assert store.event("ace/session:m2").complete is True
