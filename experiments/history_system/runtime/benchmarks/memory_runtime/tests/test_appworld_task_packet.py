"""Regression contract for the AppWorld first-user task packet."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
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


def _controller():
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
            "mode": "persistent", "history_budget_bytes": 1_000_000,
            "workspace_budget_bytes": 1_000_000, "lease_decisions": 0,
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
