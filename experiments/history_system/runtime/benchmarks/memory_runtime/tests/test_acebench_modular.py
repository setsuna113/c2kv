"""CPU contracts for ACEBench's native recovery and textual-action route."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
HISTORY_ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(RUNTIME_ROOT), str(RUNTIME_ROOT / "python")]

from benchmarks.memory_runtime.acebench_controls import build_acebench_controller
from benchmarks.memory_runtime.acebench_runtime import (
    AceEventNativeAPI,
    AceEventNativeDecisionRunner,
    describe_ace_source_contract,
)
from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.candidate_algorithms import VARIANTS
from benchmarks.memory_runtime.candidate_algorithms.controller import CandidateRecoveryController
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_api import make_server
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller, S0_CONFIG_DEFAULTS
from benchmarks.memory_runtime.event_native_tool import ToolRegionController, parse_native_tool_spec
from benchmarks.memory_runtime.event_native_server import _route_kwargs
from benchmarks.memory_runtime.recovery.hybrid import D3HybridRecoveryController
from benchmarks.memory_runtime.recovery.set_models import C1RiskArtifact
from benchmarks.memory_runtime.tests.test_d3_hybrid_recovery import (
    Tokenizer as BaseTokenizer,
    detector_config,
    packing_config,
    policy_config,
)
from benchmarks.memory_runtime.tests.test_event_native_tool import CharacterTokenizer


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


def test_ace_http_tool_spans_reach_the_real_tool_controller(tmp_path):
    tokenizer = CharacterTokenizer()
    packing = packing_config()
    packing["ratios"] = [4, 8]
    inner = build_acebench_controller(
        tokenizer, packing=packing, policy=policy_config(),
        view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config=S0_CONFIG_DEFAULTS,
    )
    spec = parse_native_tool_spec("t0:r8")
    controller = ToolRegionController(
        inner, tokenizer, spec, model_context=100_000, generator=object())

    class Runner:
        def run(self, payload):
            self.payload = payload
            visible = {key: value for key, value in payload.items()
                       if key != "outer_request_id"}
            self.prepared = controller.prepare(visible, ratio=4, max_new_tokens=8)
            prompt_tokens = self.prepared.memory.costs(4)["resident_kv_tokens"]
            return {
                "status": "ok",
                "outer_request_id": payload["outer_request_id"],
                "response": {"role": "assistant", "content": "Finish conversation", "tool_calls": [],
                             "finish_reason": "stop"},
                "generation_usage_total": {"prompt_tokens": prompt_tokens,
                                           "completion_tokens": 1,
                                           "total_tokens": prompt_tokens + 1},
            }

    runner = Runner()
    api = AceEventNativeAPI(
        runner, run_id="ace-http-cpu", model_name="c1_d3_hybrid",
        benchmark="acebench", view_mode=NATIVE_S0_MODE, max_new_tokens=8,
        allowed_task_ids=["agent_multi_step_0"], max_decisions=2,
        deadline_monotonic=time.monotonic() + 30,
        steps_path=tmp_path / "steps.jsonl",
        compression_policy=ALWAYS_COMPRESSION_POLICY,
        tool_memory_contract={"spec": spec.as_dict()},
    )
    server = make_server(api)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    definition = "Wifi(ssid='office')"
    prefix = "Available API: "
    request_payload = {
        "model": "c1_d3_hybrid", "temperature": 0.001, "top_p": 1,
        "max_tokens": 8, "store": False,
        "messages": [{"role": "system", "content": prefix + definition},
                     {"role": "user", "content": "Connect to office wifi."}],
        "c2kv_ace_source": {"version": "acebench-text-actions-v1", "receipts": []},
        "c2kv_eval_context": {"benchmark": "acebench", "task_id": "agent_multi_step_0",
                              "user_turn": 0, "step": 0, "attempt": 0},
        "c2kv_tool_spans_v1": [{"message_index": 0, "start": len(prefix),
                                "end": len(prefix) + len(definition),
                                "source": "acebench_function_list"}],
    }
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"

    def post(payload):
        wire = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        return opener.open(wire, timeout=5)

    try:
        with post(request_payload) as response:
            assert response.status == 200
            assert json.load(response)["choices"][0]["message"]["content"] == "Finish conversation"
        assert runner.payload["c2kv_tool_spans_v1"] == request_payload["c2kv_tool_spans_v1"]
        assert runner.prepared.plan is not None
        memory = runner.prepared.memory
        assert memory.tool_gist_segments or any(
            chunk.projection_set == "tool" for chunk in memory.chunks)
        invalid = dict(request_payload)
        invalid["c2kv_eval_context"] = dict(request_payload["c2kv_eval_context"], step=1)
        invalid["c2kv_tool_spans_v1"] = [dict(request_payload["c2kv_tool_spans_v1"][0],
                                              end=10_000)]
        with pytest.raises(urllib.error.HTTPError) as error:
            post(invalid)
        assert error.value.code == 400
        assert json.load(error.value)["error"]["code"] == "invalid_tool_spans"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
