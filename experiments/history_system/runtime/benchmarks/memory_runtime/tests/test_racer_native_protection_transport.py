"""Independent protection must not rewrite draft input or recovery transactions."""
from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.racer.config import BackendConfig
from benchmarks.memory_runtime.racer.generator import PersistentRacerGenerator
from benchmarks.memory_runtime.racer.native_protection import NativeProtectionAllocator
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, packing, policy
from benchmarks.memory_runtime.tests.test_racer_composition import allocator, prepare, config
from benchmarks.memory_runtime.tests.test_racer_transport import Native, Decoder, context
from history_memory.sglang_generator import SGLangEventNativeError


BACKENDS = ("commitkv", "h2o", "snapkv", "streamingllm", "pyramidkv", "agentkv")


def independent_config(backend, protection="on"):
    return BackendConfig.parse({"schema": "racer-backend-v3", "backend": backend,
        "policy": "off", "history_budget_tokens": 1024, "extra_protection": protection,
        "allocation": "backend_native_persistent", "detector_calibration": "not_used"})


class NativeProtection(Native):
    def _read_json(self, request, **kwargs):
        result, status = super()._read_json(request, **kwargs)
        if request.full_url.endswith("/v1/chat/completions"):
            session = json.loads(request.data)["c2kv_kv_memory_hint"]["persistent_history_session"]
            plan = session.get("extra_protection")
            if plan is not None:
                result["metadata"]["kv_memory_report"]["racer_native_protection"] = {
                    "schema": plan["schema"], "event_ids": plan["event_ids"],
                    "decision_id": session["transaction"]["decision_id"],
                    "applied": False, "status": "full_history_fits",
                }
        return result, status


def send(native, cfg, memory):
    generator = PersistentRacerGenerator(native, Decoder(), cfg)
    with generator.decision_scope(session_id="s"):
        generator.generate(memory, ratio=8, max_new_tokens=32, trace_context=context())
    request = copy.deepcopy(native.requests[-1][1])
    generator.close_session()
    return request


@pytest.mark.parametrize("backend", BACKENDS)
def test_off_sends_exact_legacy_wire_request(backend):
    memory = prepare(allocator(backend)).memory
    old = send(Native(), config(backend), memory)
    new = send(Native(), independent_config(backend, "off"), memory)
    assert old == new
    assert "extra_protection" not in new["c2kv_kv_memory_hint"]["persistent_history_session"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_on_requests_existing_positions_without_appending_evidence(backend):
    cfg = independent_config(backend)
    control = NativeProtectionAllocator(Tokenizer(), backend_config=cfg,
        packing={**packing(), "ratios": [8]}, policy=policy(1024))
    protected = prepare(control)
    ordinary = prepare(allocator(backend))
    memory = protected.memory
    assert memory.protection_source_indices
    assert memory.source_messages == ordinary.memory.source_messages
    assert memory.workspace_input_ids == ordinary.memory.workspace_input_ids
    assert memory.recovery_messages == memory.initial_s0_messages == ()
    assert memory.recovery_tokens == memory.native_evidence_tokens == 0
    assert memory.retained_history_cap == ordinary.memory.retained_history_cap
    request = send(NativeProtection(), cfg, memory)
    ordinary_request = send(Native(), config(backend), ordinary.memory)
    session = request["c2kv_kv_memory_hint"]["persistent_history_session"]
    plan = session.pop("extra_protection")
    assert plan["source_message_indices"] == list(memory.protection_source_indices)
    assert plan["event_ids"] == list(memory.protection_event_ids)
    assert request == ordinary_request


def test_on_requires_a_bound_engine_receipt_even_for_no_op():
    memory = prepare(allocator("h2o")).memory
    with pytest.raises(SGLangEventNativeError, match="native protection receipt"):
        send(Native(), independent_config("h2o"), memory)


def test_recovery_does_not_inherit_draft_protection():
    cfg = independent_config("h2o")
    memory = replace(prepare(allocator("h2o")).memory,
                     protection_source_indices=(1,), protection_event_ids=("event",))
    native = NativeProtection()
    generator = PersistentRacerGenerator(native, Decoder(), cfg)
    with generator.decision_scope(session_id="s"):
        generator.generate(memory, ratio=8, max_new_tokens=32, trace_context=context())
        recovered = replace(memory, recovery_messages=({"role": "user", "content": "Recovered source"},),
                            recovery_tokens=16, retained_history_cap=1008, recovered_source_indices=(1,))
        generator.generate(recovered, ratio=8, max_new_tokens=32,
                           trace_context=context(phase="regeneration"))
    session = native.requests[-1][1]["c2kv_kv_memory_hint"]["persistent_history_session"]
    assert "extra_protection" not in session
    assert session["recovery_append"]["source_message_indices"] == [1]
    assert session["transaction"]["phase"] == "regenerate"
    generator.close_session()


def test_legacy_configs_cannot_enable_new_protection_implicitly():
    value = config("h2o").receipt()
    value.pop("identity")
    value.pop("quality_validated")
    value["extra_protection"] = "on"
    with pytest.raises(ValueError, match="requires racer-backend-v3"):
        BackendConfig.parse(value)
