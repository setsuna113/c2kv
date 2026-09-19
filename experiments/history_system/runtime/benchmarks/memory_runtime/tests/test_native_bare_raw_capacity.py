import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY, CapacityInfeasible
from benchmarks.memory_runtime.event_native_exact_policy import EventNativeExactController
from benchmarks.memory_runtime.event_native_policy import (
    CURRENT_INPUT_BASELINE, HISTORY_BUDGET_DEFINITION, POLICY_SOURCE_COMMIT,
    WORKSPACE_BUDGET_DEFINITION,
)
from history_memory.packing import PackingBudgetError


class Tokenizer:
    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt=False, **kwargs):
        text = json.dumps(messages) + ("<assistant>" if add_generation_prompt else "")
        return [ord(char) for char in text]


def controller(route="ac_gist_static"):
    return EventNativeExactController(
        Tokenizer(), packing={
            "ratios": [4], "recent_tool_events": 1, "max_chunk_tokens": 128,
            "chunk_overlap": 8, "max_chunks": 48, "max_encoder_tokens": 100_000,
            "max_system_tokens": 20_000, "max_workspace_tokens": 50_000,
            "max_target_tokens": 32, "max_sequence_tokens": 512,
        }, policy={
            "mode": "persistent", "history_budget_bytes": 1_000_000,
            "workspace_budget_bytes": 1_000_000, "lease_decisions": 0,
            "max_retrieved_events": 2, "kv_bytes_per_token": 1,
            "source_commit": POLICY_SOURCE_COMMIT,
            "history_budget_definition": HISTORY_BUDGET_DEFINITION,
            "workspace_budget_definition": WORKSPACE_BUDGET_DEFINITION,
            "current_input_baseline": CURRENT_INPUT_BASELINE,
        }, mode=route,
        compression_policy=ALWAYS_COMPRESSION_POLICY if route == "ac_gist_static" else None,
        benchmark="bfcl",
    )


def payload(task, text):
    return {"session_id": task, "decision_key": "0",
            "messages": [{"role": "user", "content": text}], "tools": []}


def test_bare_mandatory_raw_overflow_is_task_capacity_and_next_task_can_prepare():
    engine = controller()
    with pytest.raises(CapacityInfeasible, match="no eligible gist") as failed:
        engine.prepare(payload("oversize", "x" * 600), ratio=4, max_new_tokens=8)
    assert isinstance(failed.value.__cause__, PackingBudgetError)
    assert "sequence needs" in str(failed.value)
    assert not engine._sessions
    prepared = engine.prepare(payload("next", "small input"), ratio=4, max_new_tokens=8)
    assert prepared.metadata["capacity_gate"]["natural_zero_eligible_raw"] is True


def test_invalid_generation_contract_is_not_reclassified_as_capacity():
    with pytest.raises(PackingBudgetError, match="Generation needs"):
        controller().prepare(payload("task", "input"), ratio=4, max_new_tokens=33)
