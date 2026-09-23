"""Exercise explicit initial protection on every persistent backend transport."""
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
from benchmarks.memory_runtime.tests.test_racer_composition import allocator, prepare
from benchmarks.memory_runtime.tests.test_racer_transport import Native, Decoder, context
from history_memory.sglang_generator import SGLangEventNativeError


class ProtectionNative(Native):
    def _read_json(self, request, **kwargs):
        result, status = super()._read_json(request, **kwargs)
        if not request.full_url.endswith("/v1/chat/completions"):
            return result, status
        hint = json.loads(request.data)["c2kv_kv_memory_hint"]["persistent_history_session"]
        plan = hint.get("racer_initial_allocation")
        if plan is not None:
            result["metadata"]["kv_memory_report"]["racer_initial_allocation"] = {
                **copy.deepcopy(plan), "applied": True, "evidence_tokens": hint["native_evidence_tokens"],
                "backend_native_selection_preserved": True,
            }
        return result, status


def protected_config(backend):
    return BackendConfig.parse({"schema": "racer-backend-v2", "backend": backend,
        "policy": "off", "mode": "protected_off", "history_budget_tokens": 1024,
        "allocation": "racer_s0", "detector_calibration": "not_used"})


def protected_memory(backend):
    base = prepare(allocator(backend)).memory
    evidence = ({"role": "user", "content": "Explicit S0 historical evidence"},)
    return replace(base, initial_s0_messages=evidence, recovery_messages=evidence,
        initial_s0_source_indices=(1,), initial_s0_event_ids=("s0-source",),
        native_evidence_source_indices=(1,), native_evidence_event_ids=("s0-source",),
        native_evidence_tokens=16, recovery_tokens=16, retained_history_cap=1008)


@pytest.mark.parametrize("backend", ["h2o", "snapkv", "streamingllm", "pyramidkv", "commitkv", "agentkv"])
def test_initial_and_recovery_share_explicit_protected_admission(backend):
    native = ProtectionNative()
    generator = PersistentRacerGenerator(native, Decoder(), protected_config(backend))
    memory = protected_memory(backend)
    with generator.decision_scope(session_id="s"):
        generator.generate(memory, ratio=8, max_new_tokens=32, trace_context=context())
        request = native.requests[-1][1]
        initial = request["c2kv_kv_memory_hint"]["persistent_history_session"]["initial_s0_append"]
        assert initial["source_message_indices"] == [1]
        appended = initial["evidence_message_indices"]
        assert appended == [len(request["messages"]) - 2, len(request["messages"]) - 1]
        assert request["messages"][appended[0]] == {"role": "assistant", "content": ""}
        assert request["messages"][appended[1]] == memory.initial_s0_messages[0]
        recovered = replace(memory,
            recovery_messages=({"role": "user", "content": "Combined S0 and recovered source"},),
            native_evidence_source_indices=(1, 2), native_evidence_tokens=32,
            recovery_tokens=32, retained_history_cap=992, recovered_source_indices=(2,))
        generator.generate(recovered, ratio=8, max_new_tokens=32, trace_context=context(phase="regeneration"))
        recovery = native.requests[-1][1]["c2kv_kv_memory_hint"]["persistent_history_session"]
        assert recovery["recovery_append"]["source_message_indices"] == [1, 2]
        assert recovery["native_evidence_tokens"] == 32
        assert "racer_initial_allocation" not in recovery
    generator.close_session()


def test_full_racer_rejects_engine_without_initial_protection_receipt():
    native = Native()
    generator = PersistentRacerGenerator(native, Decoder(), protected_config("h2o"))
    with pytest.raises(SGLangEventNativeError, match="initial protection receipt"):
        with generator.decision_scope(session_id="s"):
            generator.generate(protected_memory("h2o"), ratio=8, max_new_tokens=32, trace_context=context())
    assert native.closed


@pytest.mark.parametrize("backend", ["h2o", "snapkv", "streamingllm", "pyramidkv", "commitkv", "agentkv"])
def test_next_decision_selects_new_source_after_internal_evidence(backend):
    native = ProtectionNative()
    generator = PersistentRacerGenerator(native, Decoder(), protected_config(backend))
    memory = protected_memory(backend)
    with generator.decision_scope(session_id="s"):
        result = generator.generate(memory, ratio=8, max_new_tokens=32, trace_context=context())
        generator.resolve_decision({"role": "assistant", "content": "Done"}, result=result,
            record={"generation_trace": [{"discarded": False, "status": "completed"}]})
    first_wire = copy.deepcopy(native.requests[-1][1]["messages"])
    source_count = len(memory.source_messages)
    new_source = (*memory.source_messages, {"role": "assistant", "content": "Done"},
                  {"role": "user", "content": "Continue with the next action"})
    evidence = ({"role": "user", "content": "This decision explicitly protects the last action"},)
    next_memory = replace(memory, source_messages=new_source, source_event_phases=(),
        history_message_count=source_count + 1,
        initial_s0_messages=evidence, recovery_messages=evidence,
        initial_s0_source_indices=(source_count,), initial_s0_event_ids=("s0-next",),
        native_evidence_source_indices=(source_count,), native_evidence_event_ids=("s0-next",))
    with generator.decision_scope(session_id="s"):
        generator.generate(next_memory, ratio=8, max_new_tokens=32, trace_context=context("d2"))
    second = native.requests[-1][1]
    session = second["c2kv_kv_memory_hint"]["persistent_history_session"]
    assert session["transaction"] == {"decision_id": "d2", "phase": "draft", "resolution": "commit"}
    assert second["messages"][:len(first_wire)] == first_wire
    selected = session["initial_s0_append"]["source_message_indices"]
    assert selected == [len(first_wire)]
    assert second["messages"][selected[0]] == {"role": "assistant", "content": "Done"}
    assert session["racer_initial_allocation"]["event_ids"] == ["s0-next"]
    assert generator._source == list(new_source)
    assert evidence[0] not in generator._source
    generator.close_session()
