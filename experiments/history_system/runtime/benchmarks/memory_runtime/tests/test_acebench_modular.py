"""CPU contracts for ACEBench's native recovery and textual-action route."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
HISTORY_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(RUNTIME_ROOT), str(RUNTIME_ROOT / "python")]

from benchmarks.memory_runtime.acebench_controls import build_acebench_controller
from benchmarks.memory_runtime.acebench_runtime import (
    AceEventNativeDecisionRunner,
    describe_ace_source_contract,
)
from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.candidate_algorithms import VARIANTS
from benchmarks.memory_runtime.candidate_algorithms.controller import CandidateRecoveryController
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller, S0_CONFIG_DEFAULTS
from benchmarks.memory_runtime.event_native_server import _route_kwargs
from benchmarks.memory_runtime.recovery.hybrid import D3HybridRecoveryController
from benchmarks.memory_runtime.recovery.set_models import C1RiskArtifact
from benchmarks.memory_runtime.tests.test_d3_hybrid_recovery import (
    Tokenizer as BaseTokenizer,
    detector_config,
    packing_config,
    policy_config,
)


class Tokenizer(BaseTokenizer):
    def decode(self, ids, **_kwargs):
        return "".join(map(chr, ids))


def _controller(config):
    packing = packing_config()
    packing["ratios"] = [4, 8]
    return build_acebench_controller(
        Tokenizer(), packing=packing, policy=policy_config(),
        view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config=config,
    )


def _payload():
    return {
        "session_id": "acebench/agent_1", "decision_key": "turn-0/step-0",
        "messages": [
            {"role": "system", "content": "Use the visible APIs."},
            {"role": "user", "content": "Look up an item."},
        ],
        "tools": [],
        "c2kv_ace_source": {"version": "acebench-text-actions-v1", "receipts": []},
    }


@pytest.mark.parametrize("ratio", [4, 8])
def test_real_ace_d3_controller_prepares_receipt_backed_request(ratio):
    controller = _controller({
        **S0_CONFIG_DEFAULTS,
        "d3_hybrid_recovery": True,
        "post_draft_recovery": detector_config(),
    })
    assert isinstance(controller, D3HybridRecoveryController)
    assert isinstance(controller.base, EventNativeS0Controller)
    assert controller.base.benchmark == "acebench"
    prepared = controller.prepare(_payload(), ratio=ratio, max_new_tokens=8)
    assert prepared._store.session_id == "acebench/agent_1"
    assert prepared._store.events
    assert prepared.metadata["route"]["recovery_enabled"] is True
    reconsidered = controller.reconsider(prepared, [], draft_text="Finish conversation")
    assert reconsidered["decision"]["controller_mode"] == "d3_hybrid_candidate_first"
    assert reconsidered["decision"]["candidate_feasibility"]["ranked_candidate_count"] == 0


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_real_ace_candidate_controller_preserves_exact_variant(variant):
    artifact = HISTORY_ROOT / "artifacts" / "c1_risk.t02_v1.json"
    controller = _controller({
        **S0_CONFIG_DEFAULTS,
        "candidate_algorithm": {
            "variant": variant, "risk_artifact": str(artifact),
            "risk_threshold": 0.5,
        },
    })
    assert isinstance(controller, CandidateRecoveryController)
    assert isinstance(controller.base, EventNativeS0Controller)
    assert controller.base.benchmark == "acebench"
    assert controller.variant == variant
    assert isinstance(controller.risk_model, C1RiskArtifact)
    prepared = controller.prepare(_payload(), ratio=8, max_new_tokens=8)
    assert prepared._store.session_id == "acebench/agent_1"
    assert prepared._store.events
    assert prepared.metadata["candidate_algorithm"]["variant"] == variant
    assert prepared.metadata["route"]["recovery_enabled"] is True
    controller.risk_model = SimpleNamespace(predict_risk=lambda _context:
        SimpleNamespace(available=True, score=0.2, reason=None))
    reconsidered = controller.reconsider(prepared, [], draft_text="Finish conversation")
    assert reconsidered["decision"]["variant"] == variant
    assert reconsidered["decision"]["selection"]["score"] == 0.2


def test_ace_generate_forwards_compression_chunks_and_parses_action(tmp_path):
    class Generator:
        def generate(self, memory, **kwargs):
            self.kwargs = kwargs
            action = "[Lookup(id=1)]"
            return SimpleNamespace(
                token_ids=tuple(map(ord, action)), finish_reason="stop",
                token_logprobs=(0.0,) * len(action), stats={"eos_token_ids": ()},
            )

    controller = _controller(S0_CONFIG_DEFAULTS)
    prepared = controller.prepare(_payload(), ratio=8, max_new_tokens=8)
    generator = Generator()
    runner = AceEventNativeDecisionRunner(
        controller, generator, Tokenizer(), ratio=8, max_new_tokens=8,
        max_generation_calls=1, journal=AttemptJournal(tmp_path / "attempts.jsonl"),
    )
    record = {
        "session_id": "acebench/agent_1", "decision_key": "turn-0/step-0",
        "outer_request_id": "cpu-ace-1", "generation_trace": [],
    }
    chunks = tuple(prepared.eligible_chunks)
    _, draft = runner._generate(
        prepared.memory, prepared.metadata, record, "draft",
        compression_chunks=chunks,
    )
    assert generator.kwargs["compression_chunks"] == chunks
    assert draft.status == "tool_calls"
    assert draft.tool_calls[0]["function"]["name"] == "Lookup"
    assert record["generation_trace"][0]["draft_protocol"] == "acebench-text-actions-v1"


def test_ace_source_contract_contains_the_bundled_official_patch():
    contract = describe_ace_source_contract()
    name = "benchmarks/acebench_patches/0001-endpoint-env-and-model-registry.patch"
    patch = RUNTIME_ROOT / name
    assert patch.is_file()
    assert contract["source_sha256"][name] == hashlib.sha256(patch.read_bytes()).hexdigest()


def test_ace_s0_server_passes_its_always_compress_route_to_controller_and_api():
    kwargs = _route_kwargs(
        "acebench-text-actions-v1", NATIVE_S0_MODE,
        ALWAYS_COMPRESSION_POLICY, "fixed-budget-main")
    assert kwargs == {"compression_policy": ALWAYS_COMPRESSION_POLICY,
                      "history_view_protocol": "fixed-budget-main"}
    assert _route_kwargs("acebench-text-actions-v1", "full_original", None,
                         "fixed-budget-main") == {}
