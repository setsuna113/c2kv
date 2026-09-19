"""NPU-bundled ACE receipt routing for the opt-in source repair policies."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.acebench_controls import build_acebench_controller
from benchmarks.memory_runtime.acebench_runtime import (
    AceEventNativeDecisionRunner,
    describe_ace_source_contract,
)
from benchmarks.memory_runtime.acebench_source import parse_acebench_draft
from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.candidate_algorithms import REPAIR_VARIANTS
from benchmarks.memory_runtime.candidate_algorithms.observations import (
    operation_records,
)
from benchmarks.memory_runtime.candidate_algorithms.repair_controller import (
    RepairController,
)
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_s0_policy import (
    S0_CONFIG_DEFAULTS,
    EventNativeS0Controller,
)
from benchmarks.memory_runtime.event_native_server import _sampling_params_for_benchmark
from benchmarks.memory_runtime.tests.test_candidate_allocation import (
    Tokenizer,
    packing,
    policy,
)


def _controller(variant):
    geometry = packing()
    geometry["ratios"] = [8]
    return build_acebench_controller(
        Tokenizer(), packing=geometry, policy=policy(),
        view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config={**S0_CONFIG_DEFAULTS, "candidate_algorithm": {"variant": variant}},
    )


def _payload():
    return {
        "session_id": "acebench/agent_1", "decision_key": "turn-1/step-0",
        "messages": [
            {"role": "system", "content": "Use the visible APIs."},
            {"role": "user", "content": "Look up key 1."},
            {"role": "assistant", "content": "[Lookup(key=1)]"},
            {"role": "tool", "tool_call_id": "acebench-execution-2",
             "content": '["Error during execution: missing key"]'},
        ],
        "tools": [],
        "c2kv_ace_source": {"version": "acebench-text-actions-v1", "receipts": [{
            "version": "acebench-execution-receipt-v1",
            "execution_message_index": 3,
            "agent_history_index": 1,
            "decode_status": "ok",
            "decoded_calls": ["Lookup(key=1)"],
            "executor_status": "returned",
            "executor_return_shape": "list",
            "executor_return_count": 1,
        }]},
    }


@pytest.mark.parametrize("variant", REPAIR_VARIANTS)
def test_real_ace_repair_controller_routes_without_risk_artifact(variant):
    controller = _controller(variant)
    assert isinstance(controller, RepairController)
    assert isinstance(controller.base, EventNativeS0Controller)
    prepared = controller.prepare(_payload(), ratio=8, max_new_tokens=8)
    assert prepared.metadata["route"]["baseline_identity"] == (
        "c2kv-source-repair-v1:" + variant)
    records = operation_records(prepared._store)
    assert len(records) == 1
    assert records[0].tool == "Lookup"
    assert records[0].failure_reported is True


def test_real_ace_no_progress_uses_receipt_to_trigger_source_repair():
    controller = _controller("no_progress")
    prepared = controller.prepare(_payload(), ratio=8, max_new_tokens=8)
    draft = parse_acebench_draft("[Lookup(key=1)]", call_id_prefix="held")
    decision = controller.reconsider(
        prepared, list(draft.tool_calls), draft_text=draft.text)["decision"]
    assert decision["variant"] == "no_progress"
    assert decision["proposal"]["reason"] == "draft_repeats_failed_operation"
    assert decision["status"] == "recover"


def test_ace_runner_forwards_repair_chunks_and_keeps_stable_call_id(tmp_path):
    class DraftTokenizer(Tokenizer):
        def decode(self, ids, **_kwargs):
            return "".join(map(chr, ids))

    class Generator:
        def generate(self, memory, **kwargs):
            self.kwargs = kwargs
            action = "[Lookup(key=1)]"
            return SimpleNamespace(
                token_ids=tuple(map(ord, action)), finish_reason="stop",
                token_logprobs=(0.0,) * len(action), stats={"eos_token_ids": ()},
            )

    geometry = packing()
    geometry["ratios"] = [8]
    controller = build_acebench_controller(
        DraftTokenizer(), packing=geometry, policy=policy(),
        view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config={**S0_CONFIG_DEFAULTS,
                   "candidate_algorithm": {"variant": "no_progress"}},
    )
    prepared = controller.prepare(_payload(), ratio=8, max_new_tokens=8)
    generator = Generator()
    runner = AceEventNativeDecisionRunner(
        controller, generator, DraftTokenizer(), ratio=8, max_new_tokens=8,
        max_generation_calls=1, journal=AttemptJournal(tmp_path / "attempts.jsonl"),
    )
    record = {"session_id": "acebench/agent_1",
              "decision_key": "turn-1/step-0", "generation_trace": []}
    chunks = tuple(prepared.eligible_chunks)
    _, draft = runner._generate(
        prepared.memory, prepared.metadata, record, "regeneration",
        compression_chunks=chunks,
    )
    assert generator.kwargs["compression_chunks"] == chunks
    assert draft.tool_calls[0]["id"].endswith("_r0_0")


def test_ace_repair_route_keeps_the_source_sampling_contract():
    assert NATIVE_S0_MODE in describe_ace_source_contract()["supported_views"]
    assert _sampling_params_for_benchmark("acebench", NATIVE_S0_MODE) == {
        "temperature": 0.001, "top_p": 1.0,
    }
