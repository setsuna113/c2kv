"""Focused CPU checks for the direct result-key bridge fallback."""

from __future__ import annotations

import copy
import json

from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.event_native import memory_to_dict
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.result_key_bridge import (
    RESULT_KEY_BRIDGE_POLICY,
    ResultKeyBridgeS0Controller,
)
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


def policy(*, budget=1_000_000):
    from benchmarks.memory_runtime.event_native_policy import (
        CURRENT_INPUT_BASELINE,
        HISTORY_BUDGET_DEFINITION,
        POLICY_SOURCE_COMMIT,
        WORKSPACE_BUDGET_DEFINITION,
    )

    return {
        "mode": "persistent",
        "history_budget_bytes": budget,
        "workspace_budget_bytes": budget,
        "lease_decisions": 0,
        "max_retrieved_events": 2,
        "kv_bytes_per_token": 1,
        "source_commit": POLICY_SOURCE_COMMIT,
        "history_budget_definition": HISTORY_BUDGET_DEFINITION,
        "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
        "current_input_baseline": CURRENT_INPUT_BASELINE,
    }


def s0_config(candidate_policy):
    config = {
        "source_index_max_events": 12,
        "predictor_prompt_token_cap": 2048,
        "predictor_completion_token_cap": 256,
        "latest_complete_tool_protection": "budgeted",
    }
    if candidate_policy is not None:
        config["observed_entity_slot_policy"] = candidate_policy
    return config


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
    tool("book_flight", ["route"], required=["route"]),
    tool("purchase_insurance", ["booking_id"], required=["booking_id"]),
    tool("retrieve_invoice", ["invoice_id"], required=["invoice_id"]),
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


def source_messages(result, *, current_user="Continue.", include_pings=True):
    messages = [
        {"role": "user", "content": "Arrange the trip."},
        *tool_event("book", "book_flight", {"route": "A-B"}, result),
    ]
    if include_pings:
        messages += [
            *tool_event("ping-1", "ping", {}, {"ok": "one"}),
            *tool_event("ping-2", "ping", {}, {"ok": "two"}),
        ]
    messages.append({"role": "user", "content": current_user})
    return messages


def owner(candidate_policy=RESULT_KEY_BRIDGE_POLICY, *, budget=1_000_000):
    return build_event_native_controller(
        Tokenizer(),
        packing=packing(),
        policy=policy(budget=budget),
        view_mode=NATIVE_S0_MODE,
        compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config=s0_config(candidate_policy),
    )


def prepare(
    controller,
    result,
    *,
    messages=None,
    tools=TOOLS,
    decision_key="d1",
):
    return controller.prepare(
        {
            "session_id": "result-key-bridge/session",
            "decision_key": decision_key,
            "messages": copy.deepcopy(messages or source_messages(result)),
            "tools": copy.deepcopy(tools),
        },
        ratio=4,
        max_new_tokens=8,
    )


def test_factory_is_independently_opt_in():
    assert type(owner()) is ResultKeyBridgeS0Controller
    assert type(owner(SAME_EVENT_BRIDGE_ONLY_POLICY)) is SameEventBridgeOnlyS0Controller


def test_generated_result_key_is_admitted_with_truthful_receipt():
    value = "opaque-generated-reference"
    baseline = prepare(owner(None), {"booking_id": value})
    candidate = prepare(owner(), {"booking_id": value})
    receipt = candidate.metadata["same_event_reference"]
    workspace = Tokenizer().decode(candidate.memory.workspace_input_ids)

    expected_fields = {
        "version",
        "policy",
        "selection_status",
        "status",
        "selection_anchor_policy",
        "base_source_event_id",
        "base_result_source_index",
        "candidate_pair_count",
        "event_call_count",
        "event_result_count",
        "current_user_override_fields",
        "relation",
        "single_complete_event_only",
        "same_type_scalar_equality_required",
        "direct_result_key_required_consumer_schema",
        "call_argument_field_relation_used",
        "global_alias_table_used",
        "cross_event_join_used",
        "call_argument_is_value_source",
        "result_scalar_is_value_source",
        "direct_result_key_candidate_count",
        "inherited_missing_required_fields_rendered",
        "no_bridge_input_semantics",
        "values_serialized_in_receipt",
        "whole_field_only",
        "truncation_allowed",
        "uses_gold_future_or_hidden_state",
        "incremental_raw_tokens",
        "extra_bytes",
        "source_event_id",
        "source_event_source_indices",
        "call_source_index",
        "result_source_index",
        "call_source_message_sha256",
        "result_source_message_sha256",
        "producer_tool",
        "consumer_argument_field",
        "result_field",
        "result_path",
        "value_sha256",
        "value_type",
        "required_consumer_tools",
        "source_result_present_in_raw_workspace",
        "admitted_field_name",
        "candidate_active_history_bytes",
        "budget_bytes",
        "admission_failures",
    }
    assert expected_fields - {"admission_failures"} <= set(receipt)
    assert "admission_failures" not in receipt
    assert receipt["status"] == "admitted"
    assert receipt["selection_status"] == "selected_direct_result_key_required_consumer"
    assert receipt["relation"] == "direct_result_key_required_consumer"
    assert receipt["same_type_scalar_equality_required"] is False
    assert receipt["direct_result_key_required_consumer_schema"] is True
    assert receipt["call_argument_field_relation_used"] is False
    assert receipt["consumer_argument_field"] == "booking_id"
    assert receipt["result_field"] == "booking_id"
    assert "call_argument_field" not in receipt
    assert receipt["required_consumer_tools"] == ["purchase_insurance"]
    assert "booking_id=" + value in workspace
    assert candidate.metadata["raw_prompt_tokens"] > baseline.metadata["raw_prompt_tokens"]
    assert candidate.metadata["actual_history_bytes"] <= receipt["budget_bytes"]
    assert value not in json.dumps(receipt, sort_keys=True)


def test_existing_equality_relation_and_workspace_are_preserved():
    tools = [
        tool("get_details", ["order_id"], required=["order_id"]),
        tool("cancel", ["order_id"], required=["order_id"]),
        tool("fund", ["amount"], required=["amount"]),
        tool("ping", []),
    ]
    value = "existing-reference"
    messages = [
        {"role": "user", "content": "Read the order."},
        *tool_event(
            "details",
            "get_details",
            {"order_id": value},
            {"id": value, "amount": "10"},
        ),
        *tool_event("ping-1", "ping", {}, {"ok": "one"}),
        *tool_event("ping-2", "ping", {}, {"ok": "two"}),
        {"role": "user", "content": "Continue."},
    ]
    frozen = prepare(
        owner(SAME_EVENT_BRIDGE_ONLY_POLICY),
        {},
        messages=messages,
        tools=tools,
    )
    candidate = prepare(owner(), {}, messages=messages, tools=tools)
    receipt = candidate.metadata["same_event_reference"]

    assert memory_to_dict(candidate.memory) == memory_to_dict(frozen.memory)
    assert candidate.metadata["actual_history_bytes"] == frozen.metadata[
        "actual_history_bytes"
    ]
    assert receipt["status"] == "admitted"
    assert receipt["relation"] == "same_type_scalar_equality"
    assert receipt["same_type_scalar_equality_required"] is True
    assert receipt["direct_result_key_required_consumer_schema"] is False
    assert receipt["call_argument_field_relation_used"] is True
    assert receipt["call_argument_field"] == "order_id"
    assert receipt["result_field"] == "id"


def test_ambiguous_direct_keys_abstain_without_changing_original_s0_memory():
    result = {"booking_id": "booking", "invoice_id": "invoice"}
    baseline = prepare(owner(None), result)
    candidate = prepare(owner(), result)
    receipt = candidate.metadata["same_event_reference"]

    assert receipt["status"] == "ambiguous_direct_result_key_required_consumers"
    assert receipt["incremental_raw_tokens"] == 0
    assert receipt["relation"] is None
    assert receipt["direct_result_key_candidate_count"] == 2
    assert memory_to_dict(candidate.memory) == memory_to_dict(baseline.memory)


def test_user_override_and_raw_visible_source_abstain():
    value = "opaque-generated-reference"
    overridden = prepare(
        owner(),
        {"booking_id": value},
        messages=source_messages(
            {"booking_id": value}, current_user="booking_id: newer-user-value"
        ),
    )
    assert overridden.metadata["same_event_reference"]["status"] == (
        "base_row_not_selected"
    )
    assert overridden.metadata["same_event_reference"][
        "current_user_override_fields"
    ] == ["booking_id"]
    assert overridden.metadata["same_event_reference"]["selection_anchor_policy"] == (
        "current_user_binding_precedence"
    )

    visible = prepare(
        owner(),
        {"booking_id": value},
        messages=source_messages({"booking_id": value}, include_pings=False),
    )
    assert visible.metadata["same_event_reference"]["status"] == (
        "base_row_not_selected"
    )
    assert visible.metadata["same_event_reference"]["selection_anchor_policy"] == (
        "latest_matching_result_already_raw_visible"
    )


def test_failed_and_multi_call_sources_abstain():
    failed = prepare(owner(), {"booking_id": "bad", "success": False})
    assert failed.metadata["same_event_reference"]["status"] == "base_row_not_selected"

    multi = [
        {"role": "user", "content": "Arrange the trip."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "book",
                    "type": "function",
                    "function": {
                        "name": "book_flight",
                        "arguments": json.dumps({"route": "A-B"}),
                    },
                },
                {
                    "id": "ping-in-event",
                    "type": "function",
                    "function": {"name": "ping", "arguments": "{}"},
                },
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "book",
            "content": json.dumps({"booking_id": "generated"}),
        },
        {
            "role": "tool",
            "tool_call_id": "ping-in-event",
            "content": json.dumps({"ok": True}),
        },
        *tool_event("ping-1", "ping", {}, {"ok": "one"}),
        *tool_event("ping-2", "ping", {}, {"ok": "two"}),
        {"role": "user", "content": "Continue."},
    ]
    multiple = prepare(owner(), {}, messages=multi)
    receipt = multiple.metadata["same_event_reference"]
    assert receipt["status"] == "not_single_call_single_result"
    assert receipt["event_call_count"] == 2
    assert receipt["event_result_count"] == 2


def test_exact_budget_boundary_admits_and_one_byte_less_abstains():
    result = {"booking_id": "opaque-generated-reference"}
    probe = prepare(owner(), result)
    boundary = probe.metadata["actual_history_bytes"]

    exact = prepare(owner(budget=boundary), result)
    assert exact.metadata["same_event_reference"]["status"] == "admitted"
    assert exact.metadata["actual_history_bytes"] == boundary
    assert exact.metadata["actual_history_bytes"] <= exact.metadata[
        "same_event_reference"
    ]["budget_bytes"]

    one_less = prepare(owner(budget=boundary - 1), result)
    receipt = one_less.metadata["same_event_reference"]
    assert receipt["status"] == "whole_field_over_budget"
    assert receipt["admission_failures"]
    assert all("budget" in reason for reason in receipt["admission_failures"])
    assert one_less.metadata["actual_history_bytes"] <= boundary - 1
    assert "booking_id=opaque-generated-reference" not in Tokenizer().decode(
        one_less.memory.workspace_input_ids
    )
