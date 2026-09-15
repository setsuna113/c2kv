"""Focused CPU contracts for the default-off same-event reference bridge."""

from __future__ import annotations

import copy
import json

from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.same_event_reference import (
    SAME_EVENT_REFERENCE_POLICY,
    SameEventReferenceS0Controller,
    select_same_event_reference,
)
from history_memory.events import EventStore


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tools=None, add_generation_prompt=False, **kwargs
    ):
        text = "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>" if tools else ""
        for message in messages:
            text += "<" + message["role"] + ">" + json.dumps(
                message, sort_keys=True
            ) + "</end>"
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token) for token in token_ids)


def _packing():
    return {
        "ratios": [4],
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
    from benchmarks.memory_runtime.event_native_policy import (
        CURRENT_INPUT_BASELINE,
        HISTORY_BUDGET_DEFINITION,
        POLICY_SOURCE_COMMIT,
        WORKSPACE_BUDGET_DEFINITION,
    )

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


def _s0_config():
    return {
        "source_index_max_events": 12,
        "predictor_prompt_token_cap": 2048,
        "predictor_completion_token_cap": 256,
        "latest_complete_tool_protection": "budgeted",
    }


def _tool(name, fields, *, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {field: {"type": "string"} for field in fields},
                "required": list(required),
            },
        },
    }


TOOLS = [
    _tool("get_details", ["order_id"], required=["order_id"]),
    _tool("cancel", ["order_id"], required=["order_id"]),
    _tool("fund", ["amount"], required=["amount"]),
    _tool("ping", []),
]


def _tool_event(call_id, name, arguments, result):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(arguments),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)},
    ]


def _messages(arguments, result, *, current_user="Continue."):
    return [
        {"role": "user", "content": "Read the order."},
        *_tool_event("details", "get_details", arguments, result),
        *_tool_event("ping-1", "ping", {}, {"ok": "one"}),
        *_tool_event("ping-2", "ping", {}, {"ok": "two"}),
        {"role": "user", "content": current_user},
    ]


def _payload(messages, key="d1"):
    return {
        "session_id": "same-event/session",
        "decision_key": key,
        "messages": copy.deepcopy(messages),
        "tools": copy.deepcopy(TOOLS),
    }


def _base():
    return EventNativeS0Controller(Tokenizer(), packing=_packing(), policy=_policy())


def _candidate():
    return SameEventReferenceS0Controller(
        Tokenizer(),
        packing=_packing(),
        policy=_policy(),
        same_event_reference_policy=SAME_EVENT_REFERENCE_POLICY,
    )


def _baseline_raw(messages):
    return _base().prepare(_payload(messages), ratio=4, max_new_tokens=8)


def test_unique_same_event_pair_uses_result_value_and_records_provenance():
    secret = "opaque-order-reference"
    messages = _messages(
        {"order_id": secret},
        {"id": secret, "amount": "10"},
    )
    baseline = _baseline_raw(messages)
    candidate = _candidate().prepare(_payload(messages), ratio=4, max_new_tokens=8)
    receipt = candidate.metadata["same_event_reference"]
    workspace = Tokenizer().decode(candidate.memory.workspace_input_ids)

    assert receipt["status"] == "admitted"
    assert receipt["call_argument_field"] == "order_id"
    assert receipt["result_field"] == "id"
    assert receipt["result_path"] == ["id"]
    assert receipt["event_call_count"] == receipt["event_result_count"] == 1
    assert receipt["call_argument_is_value_source"] is False
    assert receipt["result_scalar_is_value_source"] is True
    assert receipt["source_result_present_in_raw_workspace"] is False
    assert "amount=10" in workspace
    assert "order_id=" + secret in workspace
    assert secret not in json.dumps(receipt, sort_keys=True)
    assert candidate.metadata["raw_prompt_tokens"] > baseline.metadata["raw_prompt_tokens"]


def test_duplicate_equal_result_scalars_abstain_as_ambiguous():
    messages = _messages(
        {"order_id": "same"},
        {"id": "same", "shadow": "same", "amount": "10"},
    )
    baseline = _baseline_raw(messages)
    selected = select_same_event_reference(
        EventStore.from_messages("same-event/session", messages),
        TOOLS,
        raw_source_indices=baseline.memory.raw_source_indices,
    )
    assert selected.reference is None
    assert selected.status == "ambiguous_scalar_pairs"
    assert selected.candidate_pair_count == 2


def test_failed_result_event_is_not_a_bridge_source():
    messages = _messages(
        {"order_id": "failed-value"},
        {"error": "request failed", "id": "failed-value", "amount": "10"},
    )
    baseline = _baseline_raw(messages)
    selected = select_same_event_reference(
        EventStore.from_messages("same-event/session", messages),
        TOOLS,
        raw_source_indices=baseline.memory.raw_source_indices,
    )
    assert selected.reference is None
    assert selected.status == "base_row_not_selected"
    assert selected.base.slots == ()


def test_bool_and_integer_equality_does_not_bridge():
    messages = _messages(
        {"order_id": True},
        {"id": 1, "amount": "10"},
    )
    baseline = _baseline_raw(messages)
    selected = select_same_event_reference(
        EventStore.from_messages("same-event/session", messages),
        TOOLS,
        raw_source_indices=baseline.memory.raw_source_indices,
    )
    assert selected.reference is None
    assert selected.status == "no_unique_same_type_scalar_pair"
    assert selected.candidate_pair_count == 0


def test_current_user_exact_binding_prevents_bridge():
    messages = _messages(
        {"order_id": "old-value"},
        {"id": "old-value", "amount": "10"},
        current_user="order_id=user-value",
    )
    baseline = _baseline_raw(messages)
    selected = select_same_event_reference(
        EventStore.from_messages("same-event/session", messages),
        TOOLS,
        raw_source_indices=baseline.memory.raw_source_indices,
    )
    assert selected.reference is None
    assert selected.status == "current_user_binding_precedence"
    assert selected.current_user_override_fields == ("order_id",)


def test_factory_requires_explicit_policy_and_leaves_default_s0_unchanged():
    default = build_event_native_controller(
        Tokenizer(),
        packing=_packing(),
        policy=_policy(),
        view_mode=NATIVE_S0_MODE,
        compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config=_s0_config(),
    )
    opted_in = build_event_native_controller(
        Tokenizer(),
        packing=_packing(),
        policy=_policy(),
        view_mode=NATIVE_S0_MODE,
        compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config={
            **_s0_config(),
            "observed_entity_slot_policy": SAME_EVENT_REFERENCE_POLICY,
        },
    )
    assert type(default) is EventNativeS0Controller
    assert type(opted_in) is SameEventReferenceS0Controller
