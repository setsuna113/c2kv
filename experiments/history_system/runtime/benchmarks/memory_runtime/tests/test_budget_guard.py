"""Pre-generation budget dispatch follows the controller's accounting contract."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.acebench_controls import AceEventNativeExactController
from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_exact_policy import EventNativeExactController
from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE,
    HISTORY_BUDGET_DEFINITION,
    POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner


def _controller(cls, *, history=50, workspace=50):
    controller = object.__new__(cls)
    controller.policy_config = SimpleNamespace(
        history_budget_bytes=history, workspace_budget_bytes=workspace
    )
    controller.kv_bytes_per_token = 1
    return controller


@pytest.fixture
def memory():
    return SimpleNamespace(
        costs=lambda ratio: {"resident_kv_tokens": 120, "gist_tokens": 10}
    )


@pytest.mark.parametrize("cls", [EventNativeExactController, AceEventNativeExactController])
@pytest.mark.parametrize(
    "metadata",
    [
        {"actual_history_bytes": 20},
        {"common_raw_prompt_tokens": 100, "actual_history_bytes": 20},
    ],
)
def test_exact_controllers_use_their_own_capacity_accounting(memory, cls, metadata):
    controller = _controller(cls)
    receipt = history_budget_receipt(
        memory, metadata, controller, ratio=4, phase="draft"
    )
    assert receipt["status"] == "not_applicable"
    assert receipt["errors"] == []
    assert receipt["controller"] == cls.__name__

    runner = object.__new__(EventNativeDecisionRunner)
    runner.controller = controller
    runner.ratio = 4
    runner.generation_calls = runner.max_generation_calls = 0
    record = {}
    with pytest.raises(RuntimeError, match="Finite generation-call cap exhausted"):
        runner._generate(memory, metadata, record, "draft")
    assert record["pre_generation_budget_checks"][0] == receipt


def test_bfcl_bare_prepared_view_has_no_s0_boundary():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False, **kwargs):
            text = ""
            for message in messages:
                text += "<" + message["role"] + ">" + json.dumps(message, sort_keys=True) + "</end>"
            if add_generation_prompt:
                text += "<assistant>"
            return [ord(char) for char in text]

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
        benchmark="bfcl",
    )
    prepared = controller.prepare(
        {
            "session_id": "bfcl/bare", "decision_key": "d1", "tools": [],
            "messages": [
                {"role": "system", "content": "Rules"},
                {"role": "user", "content": "Previous request"},
                {"role": "assistant", "content": "Previous answer"},
                {"role": "user", "content": "Current request"},
            ],
        },
        ratio=4, max_new_tokens=8,
    )
    assert "common_raw_prompt_tokens" not in prepared.metadata
    assert prepared.metadata["actual_history_bytes"] <= 512
    receipt = history_budget_receipt(
        prepared.memory, prepared.metadata, controller, ratio=4, phase="draft"
    )
    assert receipt["status"] == "not_applicable"


@pytest.mark.parametrize("common", [None, "100", 100.0, True])
def test_s0_malformed_common_boundary_is_not_skipped(memory, common):
    controller = _controller(EventNativeS0Controller)
    with pytest.raises(ValueError, match="explicit native history budget contract"):
        history_budget_receipt(
            memory,
            {"common_raw_prompt_tokens": common, "actual_history_bytes": 20},
            controller, ratio=4, phase="draft",
        )


def test_s0_missing_common_boundary_is_not_skipped(memory):
    controller = _controller(EventNativeS0Controller)
    with pytest.raises(ValueError, match="explicit native history budget contract"):
        history_budget_receipt(
            memory, {"actual_history_bytes": 20}, controller,
            ratio=4, phase="draft",
        )


def test_s0_budget_and_accounting_are_enforced(memory):
    controller = _controller(EventNativeS0Controller)
    metadata = {"common_raw_prompt_tokens": 100, "actual_history_bytes": 20}
    assert history_budget_receipt(
        memory, metadata, controller, ratio=4, phase="draft"
    )["status"] == "passed"
    wrong = dict(metadata, actual_history_bytes=19)
    assert "assembled_history_differs_from_controller_accounting" in history_budget_receipt(
        memory, wrong, controller, ratio=4, phase="draft"
    )["errors"]
    controller.policy_config.history_budget_bytes = 10
    assert "active_history_exceeds_budget" in history_budget_receipt(
        memory, metadata, controller, ratio=4, phase="draft"
    )["errors"]


def test_unrecognized_controller_cannot_skip_missing_common_boundary(memory):
    controller = _controller(type("OtherController", (), {}))
    with pytest.raises(ValueError, match="explicit native history budget contract"):
        history_budget_receipt(
            memory, {"actual_history_bytes": 20}, controller,
            ratio=4, phase="draft",
        )
