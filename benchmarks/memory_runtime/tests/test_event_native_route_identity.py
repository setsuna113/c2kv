"""Cross-route source-identity parity without loading model weights."""

from __future__ import annotations

import copy
import time

from benchmarks.memory_runtime.event_native import memory_to_dict
from benchmarks.memory_runtime.event_native_api import EventNativeAPI
from benchmarks.memory_runtime.event_native_controls import (
    build_event_native_controller,
)
from benchmarks.memory_runtime.tests.test_event_native_exact import (
    MAX_NEW_TOKENS,
    RATIO,
    _activated_settings,
    _messages,
)
from benchmarks.memory_runtime.tests.test_event_native_policy import Tokenizer


class _PreparingRunner:
    max_generation_calls = 1
    generation_calls = 0

    def __init__(self, controller):
        self.controller = controller
        self.payloads = []
        self.prepared = []

    def run(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        self.prepared.append(
            self.controller.prepare(
                payload, ratio=RATIO, max_new_tokens=MAX_NEW_TOKENS
            )
        )
        return {
            "schema": "a-event-native-exact-step-v1",
            "status": "ok",
            "generation_trace": [],
            "response": {
                "role": "assistant",
                "content": "prepared without generation",
                "tool_calls": [],
                "reasoning_content": None,
                "finish_reason": "stop",
            },
            "generation_attempts": 0,
            "generation_usage_total": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }


def _api(tmp_path, *, run_id, view_mode, controller):
    return EventNativeAPI(
        _PreparingRunner(controller),
        run_id=run_id,
        model_name="tokenizer-only",
        view_mode=view_mode,
        max_new_tokens=MAX_NEW_TOKENS,
        allowed_task_ids={"multi_turn_base_7"},
        max_decisions=1,
        deadline_monotonic=time.monotonic() + 30,
        steps_path=tmp_path / run_id / "steps.jsonl",
    )


def _request():
    return {
        "messages": copy.deepcopy(_messages()),
        "tools": [],
        "model": "tokenizer-only",
        "temperature": 0,
        "store": False,
        "max_completion_tokens": MAX_NEW_TOKENS,
        "seed": 0,
        "c2kv_eval_context": {
            "benchmark": "bfcl",
            "task_id": "multi_turn_base_7",
            "user_turn": 0,
            "step": 0,
            "attempt": 0,
        },
    }


def test_stable_api_session_makes_protect_and_once_first_drafts_identical(
    tmp_path,
) -> None:
    packing, policy = _activated_settings(lease_decisions=3)
    protect = _api(
        tmp_path,
        run_id="protect-run",
        view_mode="capacity_protect",
        controller=build_event_native_controller(
            Tokenizer(),
            packing=packing,
            policy=policy,
            view_mode="capacity_protect",
        ),
    )
    once = _api(
        tmp_path,
        run_id="once-run",
        view_mode="capacity_exact_once",
        controller=build_event_native_controller(
            Tokenizer(),
            packing=packing,
            policy=policy,
            view_mode="capacity_exact_once",
        ),
    )

    protect.handle_chat(_request())
    once.handle_chat(_request())
    protect_runner, once_runner = protect.runner, once.runner
    assert protect_runner.payloads[0]["session_id"] == once_runner.payloads[0][
        "session_id"
    ] == "bfcl/multi_turn_base_7/attempt-0"

    protect_input = memory_to_dict(protect_runner.prepared[0].memory)
    once_input = memory_to_dict(once_runner.prepared[0].memory)
    assert protect_input == once_input
    assert protect_input["system_input_ids"] == once_input["system_input_ids"]
    assert protect_input["workspace_input_ids"] == once_input["workspace_input_ids"]
    assert [row["token_ids"] for row in protect_input["chunks"]] == [
        row["token_ids"] for row in once_input["chunks"]
    ]
    evidence = protect_input["view"]["evidence_event_ids"]
    assert evidence == once_input["view"]["evidence_event_ids"]
    assert evidence
