"""Regression contract for the AppWorld first-user task packet."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)


class Tokenizer:
    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False, **kwargs):
        text = "<tools>" + json.dumps(tools, sort_keys=True) + "</tools>" if tools else ""
        for message in messages:
            text += "<" + message["role"] + ">" + json.dumps(message, sort_keys=True) + "</end>"
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]


def _controller(*, history_budget_bytes=1_000_000, workspace_budget_bytes=1_000_000):
    return EventNativeS0Controller(
        Tokenizer(),
        packing={
            "ratios": [4], "recent_tool_events": 1,
            "max_chunk_tokens": 768, "chunk_overlap": 64,
            "max_chunks": 48, "max_encoder_tokens": 100_000,
            "max_system_tokens": 20_000, "max_workspace_tokens": 50_000,
            "max_target_tokens": 32, "max_sequence_tokens": 100_000,
        },
        policy={
            "mode": "persistent", "history_budget_bytes": history_budget_bytes,
            "workspace_budget_bytes": workspace_budget_bytes, "lease_decisions": 0,
            "max_retrieved_events": 2, "kv_bytes_per_token": 1,
            "source_commit": POLICY_SOURCE_COMMIT,
            "history_budget_definition": HISTORY_BUDGET_DEFINITION,
            "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
            "current_input_baseline": CURRENT_INPUT_BASELINE,
        },
        benchmark="acon_appworld",
    )


def test_first_appworld_user_is_mandatory_raw_event():
    messages = [
        {"role": "system", "content": "API schema"},
        {"role": "user", "content": "APPWORLD TASK PACKET: create a calendar event"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "calendar.create", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call-1", "content": "created"},
        {"role": "user", "content": "What happened?"},
    ]
    payload = {
        "session_id": "appworld/task",
        "decision_key": "d1",
        "messages": copy.deepcopy(messages),
        "tools": [],
    }
    prepared = _controller().prepare(payload, ratio=4, max_new_tokens=8)
    assert prepared.metadata["benchmark"] == "acon_appworld"
    assert prepared.metadata["task_packet_protection"] == "first_non_system_user_raw"
    assert prepared.metadata["task_packet_event_id"] == "appworld/task:m1"
    assert "appworld/task:m1" in prepared.metadata["mandatory_raw_event_ids"]


def test_long_task_packet_is_fixed_common_input_outside_managed_budgets():
    messages = [
        {"role": "system", "content": "API schema"},
        {"role": "user", "content": "APPWORLD TASK PACKET: " + "x" * 2_000},
        {"role": "assistant", "content": "Ready."},
        {"role": "user", "content": "Take the first action."},
    ]
    prepared = _controller(
        history_budget_bytes=128,
        workspace_budget_bytes=128,
    ).prepare(
        {
            "session_id": "appworld/long-task-packet",
            "decision_key": "d1",
            "messages": copy.deepcopy(messages),
            "tools": [],
        },
        ratio=4,
        max_new_tokens=8,
    )

    metadata = prepared.metadata
    row = metadata["per_ratio"]["4"]
    assert metadata["task_packet_source_indices"] == [1]
    assert metadata["task_packet_raw_tokens"] > 128
    assert metadata["task_packet_raw_bytes"] == metadata["task_packet_raw_tokens"]
    assert metadata["task_packet_accounting"] == {
        "token_definition": (
            "marginal tokens contributed by the AppWorld first-user task packet "
            "within the rendered common input"
        ),
        "charged_to_history_budget": False,
        "charged_to_workspace_budget": False,
        "included_in_total_resident_kv": True,
    }
    assert 1 in metadata["common_input_source_indices"]
    assert row["managed_history_bytes"] <= 128
    assert row["managed_workspace_bytes"] <= 128
    assert row["total_resident_kv_bytes"] > 128
    assert metadata["actual_managed_history_bytes"] == row["managed_history_bytes"]
    assert metadata["actual_total_resident_kv_bytes"] == row[
        "total_resident_kv_bytes"
    ]


def test_static_native_compression_keeps_appworld_task_packet_raw_and_common():
    packing = {
        "ratios": [4], "recent_tool_events": 1,
        "max_chunk_tokens": 768, "chunk_overlap": 64,
        "max_chunks": 48, "max_encoder_tokens": 100_000,
        "max_system_tokens": 20_000, "max_workspace_tokens": 50_000,
        "max_target_tokens": 32, "max_sequence_tokens": 100_000,
    }
    policy = {
        "mode": "persistent", "history_budget_bytes": 512,
        "workspace_budget_bytes": 512, "lease_decisions": 0,
        "max_retrieved_events": 2, "kv_bytes_per_token": 1,
        "source_commit": POLICY_SOURCE_COMMIT,
        "history_budget_definition": HISTORY_BUDGET_DEFINITION,
        "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
        "current_input_baseline": CURRENT_INPUT_BASELINE,
    }
    controller = build_event_native_controller(
        Tokenizer(), packing=packing, policy=policy,
        view_mode="ac_gist_static", compression_policy=ALWAYS_COMPRESSION_POLICY,
        benchmark="acon_appworld",
    )
    messages = [
        {"role": "system", "content": "API schema"},
        {"role": "user", "content": "APPWORLD TASK PACKET: " + "x" * 2_000},
        {"role": "assistant", "content": "Ready."},
        {"role": "user", "content": "Take the first action."},
    ]
    prepared = controller.prepare(
        {"session_id": "appworld/static", "decision_key": "d1",
         "messages": messages, "tools": []},
        ratio=4, max_new_tokens=8,
    )
    packet_id = "appworld/static:m1"
    assert packet_id in prepared.memory.view.raw_event_ids
    assert packet_id not in prepared.memory.view.gist_event_ids
    assert packet_id not in prepared.memory.view.evidence_event_ids
    assert 1 in prepared.memory.raw_source_indices
    assert packet_id not in prepared.metadata["source_coverage"]["eligible_event_ids"]
    assert prepared.metadata["task_packet_event_id"] == packet_id
    assert 1 in prepared.metadata["common_input_source_indices"]
    assert prepared.metadata["task_packet_raw_tokens"] > 512
    assert prepared.metadata["task_packet_raw_bytes"] == prepared.metadata["task_packet_raw_tokens"]
    assert prepared.metadata["task_packet_common_extra_tokens"] > 512
    assert prepared.metadata["task_packet_accounting"] == {
        "charged_to_history_budget": False,
        "charged_to_workspace_budget": False,
        "included_in_total_resident_kv": True,
    }
    assert prepared.metadata["actual_history_bytes"] <= 512
    assert prepared.metadata["same_prefix_full_reference"]["full_history_bytes"] <= 512
    assert prepared.memory.costs(4)["resident_kv_tokens"] > 512
