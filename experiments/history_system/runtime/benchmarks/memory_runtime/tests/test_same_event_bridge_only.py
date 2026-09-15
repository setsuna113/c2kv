"""Focused CPU checks for the default-off bridge-only SAME controller."""

from __future__ import annotations

import copy
import json

from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.event_native import memory_to_dict
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.same_event_bridge_only import (
    SAME_EVENT_BRIDGE_ONLY_POLICY,
    SameEventBridgeOnlyS0Controller,
)


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


def packing():
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


def policy():
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


def s0_config():
    return {
        "source_index_max_events": 12,
        "predictor_prompt_token_cap": 2048,
        "predictor_completion_token_cap": 256,
        "latest_complete_tool_protection": "budgeted",
    }


def tool(name, fields, *, required=()):
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
    tool("get_details", ["order_id"], required=["order_id"]),
    tool("cancel", ["order_id"], required=["order_id"]),
    tool("fund", ["amount"], required=["amount"]),
    tool("ping", []),
]


def tool_event(call_id, name, arguments, result):
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


def messages(arguments, result):
    return [
        {"role": "user", "content": "Read the order."},
        *tool_event("details", "get_details", arguments, result),
        *tool_event("ping-1", "ping", {}, {"ok": "one"}),
        *tool_event("ping-2", "ping", {}, {"ok": "two"}),
        {"role": "user", "content": "Continue."},
    ]


def owner(candidate=False):
    config = s0_config()
    if candidate:
        config["observed_entity_slot_policy"] = SAME_EVENT_BRIDGE_ONLY_POLICY
    return build_event_native_controller(
        Tokenizer(),
        packing=packing(),
        policy=policy(),
        view_mode=NATIVE_S0_MODE,
        compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config=config,
    )


def prepare(controller, value, result):
    return controller.prepare(
        {
            "session_id": "bridge-only/session",
            "decision_key": "d1",
            "messages": copy.deepcopy(messages({"order_id": value}, result)),
            "tools": copy.deepcopy(TOOLS),
        },
        ratio=4,
        max_new_tokens=8,
    )


def test_factory_is_opt_in_and_preserves_original_s0_path():
    assert type(owner()) is EventNativeS0Controller
    assert type(owner(candidate=True)) is SameEventBridgeOnlyS0Controller


def test_no_unique_bridge_keeps_exact_original_s0_memory():
    value = "opaque-reference"
    baseline = prepare(owner(), value, {"id": "different", "amount": "10"})
    candidate = prepare(
        owner(candidate=True), value, {"id": "different", "amount": "10"}
    )
    assert memory_to_dict(candidate.memory) == memory_to_dict(baseline.memory)
    receipt = candidate.metadata["same_event_reference"]
    assert receipt["status"] == "no_unique_same_type_scalar_pair"
    assert receipt["incremental_raw_tokens"] == 0
    assert receipt["inherited_missing_required_fields_rendered"] is False


def test_unique_bridge_renders_only_reference_field_and_keeps_source_provenance():
    value = "opaque-reference"
    baseline = prepare(owner(), value, {"id": value, "amount": "10"})
    candidate = prepare(owner(candidate=True), value, {"id": value, "amount": "10"})
    workspace = Tokenizer().decode(candidate.memory.workspace_input_ids)
    receipt = candidate.metadata["same_event_reference"]

    assert receipt["status"] == "admitted"
    assert receipt["call_argument_field"] == "order_id"
    assert receipt["result_field"] == "id"
    assert receipt["call_source_message_sha256"]
    assert receipt["result_source_message_sha256"]
    assert receipt["inherited_missing_required_fields_rendered"] is False
    assert "order_id=" + value in workspace
    assert "amount=10" not in workspace
    assert candidate.metadata["raw_prompt_tokens"] > baseline.metadata["raw_prompt_tokens"]
    assert value not in json.dumps(receipt, sort_keys=True)
