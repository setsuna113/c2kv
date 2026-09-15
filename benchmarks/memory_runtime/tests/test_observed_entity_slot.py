"""Focused contracts for the opt-in native observed entity slot candidate."""

from __future__ import annotations

import copy
import json

import pytest

from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.event_native_always import NATIVE_RAW_S0_MODE, NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.observed_entity_slot import (
    OBSERVED_ENTITY_SLOT_POLICY,
    ObservedEntitySlotS0Controller,
    select_observed_entity_slots,
)
from history_memory.events import EventStore


class Tokenizer:
    def apply_chat_template(
        self, messages, *, tools=None, add_generation_prompt=False, **kwargs
    ):
        text = "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>" if tools else ""
        for message in messages:
            text += "<" + message["role"] + ">" + json.dumps(message, sort_keys=True) + "</end>"
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]


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


def _policy(budget=1_000_000):
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


def _tool(name, fields):
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {field: {"type": "string"} for field in fields},
            },
        },
    }


def _tool_event(call_id, name, arguments, result):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)},
    ]


def _payload(messages, tools, key="d1"):
    return {
        "session_id": "observed-slot/session",
        "decision_key": key,
        "messages": copy.deepcopy(messages),
        "tools": copy.deepcopy(tools),
    }


def _base(*, budget=1_000_000):
    return EventNativeS0Controller(Tokenizer(), packing=_packing(), policy=_policy(budget))


def _candidate(*, budget=1_000_000, enabled=True, history_representation="c2kv"):
    return ObservedEntitySlotS0Controller(
        Tokenizer(),
        packing=_packing(),
        policy=_policy(budget),
        history_representation=history_representation,
        observed_entity_slot_policy=(OBSERVED_ENTITY_SLOT_POLICY if enabled else None),
    )


def _messages(result_value="server-value", current="Use the entity now."):
    return [
        {"role": "user", "content": "Create an entity."},
        *_tool_event("create", "create_entity", {}, {"entity_id": result_value}),
        {"role": "assistant", "content": "Entity created."},
        {"role": "user", "content": current},
    ]


TOOLS = [_tool("consume_entity", ["entity_id"])]


def test_default_none_is_exact_native_s0_and_opt_in_materializes_real_value():
    payload = _payload(_messages(), TOOLS)
    baseline = _base().prepare(payload, ratio=4, max_new_tokens=8)
    default = _candidate(enabled=False).prepare(payload, ratio=4, max_new_tokens=8)
    selected = _candidate().prepare(payload, ratio=4, max_new_tokens=8)

    assert default.memory == baseline.memory
    assert default.metadata == baseline.metadata
    assert selected.memory.system_input_ids == baseline.memory.system_input_ids
    assert selected.memory.chunks == baseline.memory.chunks
    assert selected.memory.raw_source_indices == baseline.memory.raw_source_indices
    assert selected.memory.view == baseline.memory.view
    assert selected.memory.workspace_input_ids != baseline.memory.workspace_input_ids
    workspace = "".join(chr(token) for token in selected.memory.workspace_input_ids)
    assert "entity_id=server-value" in workspace
    assert "entity_id=server-value" not in json.dumps(selected.metadata, sort_keys=True)
    receipt = selected.metadata["observed_entity_slot"]
    assert receipt["status"] == "admitted"
    assert receipt["admitted_field_names"] == ["entity_id"]
    assert receipt["source_message_role"] == "tool"
    assert "complete visible tool-result" in receipt["source_semantics"]
    assert "latest visible user" in receipt["user_override_semantics"]
    assert receipt["incremental_raw_tokens"] > 0
    assert receipt["candidate_active_history_bytes"] <= receipt["budget_bytes"]
    assert receipt["values_serialized_in_receipt"] is False


def test_no_match_preserves_the_complete_legacy_memory_view():
    messages = _messages()
    tools = [_tool("consume_entity", ["other_id"])]
    payload = _payload(messages, tools)
    baseline = _base().prepare(payload, ratio=4, max_new_tokens=8)
    selected = _candidate().prepare(payload, ratio=4, max_new_tokens=8)

    assert selected.memory == baseline.memory
    assert selected.metadata["observed_entity_slot"]["status"] == (
        "no_matching_successful_observation"
    )


def test_current_explicit_user_binding_has_precedence():
    messages = _messages(current="entity_id=user-value")
    payload = _payload(messages, TOOLS)
    baseline = _base().prepare(payload, ratio=4, max_new_tokens=8)
    selected = _candidate().prepare(payload, ratio=4, max_new_tokens=8)

    assert selected.memory == baseline.memory
    receipt = selected.metadata["observed_entity_slot"]
    assert receipt["status"] == "current_user_binding_precedence"
    assert receipt["admitted_field_names"] == []
    assert receipt["skipped_fields"] == [
        {"field": "entity_id", "reason": "current_user_binding_precedence"}
    ]


def test_failed_result_does_not_replace_prior_successful_value():
    messages = [
        {"role": "user", "content": "Create an entity."},
        *_tool_event("good", "create_entity", {}, {"entity_id": "good-value"}),
        *_tool_event(
            "failed",
            "refresh_entity",
            {},
            {"entity_id": "bad-value", "error": "refresh failed"},
        ),
        {"role": "user", "content": "Use the entity."},
    ]
    selection = select_observed_entity_slots(
        EventStore.from_messages("observed-slot/session", messages), TOOLS
    )

    assert selection.status == "selected"
    assert [(slot.field, slot.value) for slot in selection.slots] == [
        ("entity_id", "good-value")
    ]


def test_over_budget_skips_the_complete_field_without_truncation():
    value = "v" * 500
    messages = _messages(result_value=value)
    payload = _payload(messages, TOOLS)
    roomy = _base().prepare(payload, ratio=4, max_new_tokens=8)
    boundary = roomy.metadata["actual_history_bytes"]
    baseline = _base(budget=boundary).prepare(payload, ratio=4, max_new_tokens=8)
    selected = _candidate(budget=boundary).prepare(payload, ratio=4, max_new_tokens=8)

    assert selected.memory == baseline.memory
    receipt = selected.metadata["observed_entity_slot"]
    assert receipt["status"] == "over_budget"
    assert receipt["admitted_field_names"] == []
    assert receipt["skipped_fields"][0]["reason"] == "whole_field_over_budget"
    assert receipt["truncation_allowed"] is False
    assert value not in json.dumps(receipt, sort_keys=True)


def test_policy_identity_and_representation_are_explicit():
    with pytest.raises(ValueError, match="Unknown observed entity slot policy"):
        ObservedEntitySlotS0Controller(
            Tokenizer(),
            packing=_packing(),
            policy=_policy(),
            observed_entity_slot_policy="unknown",
        )
    with pytest.raises(ValueError, match="require the C2KV"):
        _candidate(history_representation="raw")


def test_runtime_builder_is_explicit_opt_in_and_keeps_default_controller():
    base_config = {
        "source_index_max_events": 12,
        "predictor_prompt_token_cap": 2048,
        "predictor_completion_token_cap": 256,
        "latest_complete_tool_protection": "budgeted",
    }
    baseline = build_event_native_controller(
        Tokenizer(),
        packing=_packing(),
        policy=_policy(),
        view_mode=NATIVE_S0_MODE,
        compression_policy="always-compress-v1",
        s0_config=base_config,
    )
    candidate = build_event_native_controller(
        Tokenizer(),
        packing=_packing(),
        policy=_policy(),
        view_mode=NATIVE_S0_MODE,
        compression_policy="always-compress-v1",
        s0_config={
            **base_config,
            "observed_entity_slot_policy": OBSERVED_ENTITY_SLOT_POLICY,
        },
    )

    assert type(baseline) is EventNativeS0Controller
    assert isinstance(candidate, ObservedEntitySlotS0Controller)
    assert candidate.s0_config == base_config
    assert candidate.observed_entity_slot_policy == OBSERVED_ENTITY_SLOT_POLICY


def test_runtime_builder_rejects_candidate_on_raw_or_unknown_policy():
    config = {
        "source_index_max_events": 12,
        "predictor_prompt_token_cap": 2048,
        "predictor_completion_token_cap": 256,
        "latest_complete_tool_protection": "budgeted",
        "observed_entity_slot_policy": OBSERVED_ENTITY_SLOT_POLICY,
    }
    with pytest.raises(ValueError, match="require the native S0 route"):
        build_event_native_controller(
            Tokenizer(),
            packing=_packing(),
            policy=_policy(),
            view_mode=NATIVE_RAW_S0_MODE,
            s0_config=config,
        )
    config["observed_entity_slot_policy"] = "unknown"
    with pytest.raises(ValueError, match="Unknown observed entity slot policy"):
        build_event_native_controller(
            Tokenizer(),
            packing=_packing(),
            policy=_policy(),
            view_mode=NATIVE_S0_MODE,
            compression_policy="always-compress-v1",
            s0_config=config,
        )
