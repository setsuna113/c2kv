"""V4 source-instance transport remains token neutral and binds all receipts."""
import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.racer.protection_transport import (
    EVIDENCE_PREFIX, protection_request, validate_protection_receipt)
from benchmarks.memory_runtime.racer.config import BackendConfig
from benchmarks.memory_runtime.tests.test_racer_native_protection_transport import send
from benchmarks.memory_runtime.tests.test_racer_composition import allocator, prepare, config
from benchmarks.memory_runtime.tests.test_racer_transport import Native
from benchmarks.memory_runtime.tests.test_racer_transport import Decoder, context
from benchmarks.memory_runtime.racer.generator import PersistentRacerGenerator
from benchmarks.memory_runtime.racer.native_protection_v2 import NativeProtectionV2Allocator
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, packing, policy


def memory():
    sources = ({"role": "user", "content": "Find Alpha"},
               {"role": "tool", "content": '{"name":"Alpha","id":7}'})
    unit = {"unit_id": "u", "event_id": "e", "complete_event": False,
            "source_indices": [1], "fragments": [{"source_index": 1, "text": sources[1]["content"]}]}
    return SimpleNamespace(source_messages=sources, protection_scope_id="s",
        protection_units=(unit,), protection_event_ids=("e",), protection_source_indices=(1,),
        protection_source_events=({"event_id": "e", "source_indices": [1]},))


def evidence(mem):
    row = {"event_id": "e", "source_indices": [1], "messages": [mem.source_messages[1]]}
    return {"role": "user", "content": EVIDENCE_PREFIX + json.dumps([row], ensure_ascii=False, separators=(",", ":"))}


def test_recovery_copy_bound_to_original_source_and_carrier_map():
    mem = memory()
    messages = [*mem.source_messages, {"role": "assistant", "content": ""}, evidence(mem)]
    before = copy.deepcopy(messages)
    plan = protection_request(mem, messages, [0, 1], {0: 0, 1: 2, 2: 3, 3: 4})
    original, recovered = plan["units"][0]["instances"]
    assert original["source_message_indices"] == [2]
    assert recovered["source_message_indices"] == [4]
    assert recovered["fragments"][0]["text"] == r'{\"name\":\"Alpha\",\"id\":7}'
    assert plan["events"][0]["instances"][1]["complete_event"] is True
    assert messages == before


def test_user_marker_and_wrong_source_never_become_recovery_aliases():
    mem = memory()
    wrong = evidence(mem)
    wrong["content"] = wrong["content"].replace("Alpha", "Beta")
    source_marker = evidence(mem)
    mem.source_messages += (source_marker,)
    plan = protection_request(mem, [*mem.source_messages, wrong], [0, 1, 2], {i: i for i in range(4)})
    assert len(plan["units"][0]["instances"]) == 1


def test_receipt_rejects_missing_coverage_even_when_applied_is_false():
    mem = memory()
    receipt = {"schema": "racer-native-protection-v2", "decision_id": "d", "scope_id": "s",
               "unit_ids": ["u"], "event_ids": ["e"], "applied": False, "status": "full_history_fits",
               "units": [{"unit_id": "u"}], "event_coverage": [{"event_id": "e", "status": "partial",
                "full_rows": 0, "partial_rows": 1, "total_rows": 1}]}
    validate_protection_receipt(receipt, mem, "d")
    receipt["event_coverage"] = []
    with pytest.raises(ValueError, match="coverage"):
        validate_protection_receipt(receipt, mem, "d")


def test_full_coverage_requires_every_nonempty_row():
    mem = memory()
    receipt = {"schema": "racer-native-protection-v2", "decision_id": "d", "scope_id": "s",
        "unit_ids": ["u"], "event_ids": ["e"], "applied": False, "status": "already_retained",
        "units": [{"unit_id": "u"}], "event_coverage": [{"event_id": "e", "status": "full",
        "full_rows": 1, "partial_rows": 1, "total_rows": 2}]}
    with pytest.raises(ValueError, match="coverage"):
        validate_protection_receipt(receipt, mem, "d")


@pytest.mark.parametrize("backend", ["commitkv", "snapkv", "h2o", "pyramidkv", "agentkv", "streamingllm"])
def test_v4_off_is_exact_legacy_request(backend):
    mem = prepare(allocator(backend)).memory
    value = {"schema": "racer-backend-v4", "backend": backend, "policy": "off",
             "history_budget_tokens": 1024, "extra_protection": "off",
             "allocation": "backend_native_persistent", "detector_calibration": "not_used"}
    assert send(Native(), BackendConfig.parse(value), mem) == send(Native(), config(backend), mem)


class NativeV2(Native):
    def _read_json(self, request, **kwargs):
        result, status = super()._read_json(request, **kwargs)
        if request.full_url.endswith("/v1/chat/completions"):
            session = json.loads(request.data)["c2kv_kv_memory_hint"]["persistent_history_session"]
            plan = session["extra_protection"]
            result["metadata"]["kv_memory_report"]["racer_native_protection"] = {
                "schema": plan["schema"], "scope_id": plan["scope_id"],
                "event_ids": plan["event_ids"], "unit_ids": plan["unit_ids"],
                "decision_id": session["transaction"]["decision_id"],
                "applied": False, "status": "coverage_only",
                "units": [{"unit_id": item} for item in plan["unit_ids"]],
                "event_coverage": [{"event_id": item, "status": "partial", "full_rows": 0,
                    "partial_rows": 1, "total_rows": 1} for item in plan["event_ids"]]}
        return result, status


def test_v4_draft_and_regeneration_share_plan_without_initial_evidence():
    cfg = BackendConfig.parse({"schema": "racer-backend-v4", "backend": "snapkv", "policy": "off",
        "history_budget_tokens": 1024, "extra_protection": "on", "allocation": "backend_native_persistent",
        "detector_calibration": "not_used"})
    control = NativeProtectionV2Allocator(Tokenizer(), backend_config=cfg,
        packing={**packing(), "ratios": [8]}, policy=policy(1024))
    prepared = prepare(control)
    mem = prepared.memory
    native = NativeV2()
    generator = PersistentRacerGenerator(native, Decoder(), cfg)
    with generator.decision_scope(session_id="s"):
        generator.generate(mem, ratio=8, max_new_tokens=32, trace_context=context())
        draft = copy.deepcopy(native.requests[-1][1])
        recovered = replace(mem, recovery_messages=({"role": "user", "content": "Recovered source"},),
                            recovery_tokens=16, retained_history_cap=1008, recovered_source_indices=(1,))
        result = generator.generate(recovered, ratio=8, max_new_tokens=32,
                                    trace_context=context(phase="regeneration"))
    draft_session = draft["c2kv_kv_memory_hint"]["persistent_history_session"]
    regen_session = native.requests[-1][1]["c2kv_kv_memory_hint"]["persistent_history_session"]
    assert "initial_s0_append" not in draft_session
    assert draft["messages"] == list(mem.source_messages)
    assert draft_session["extra_protection"]["unit_ids"] == regen_session["extra_protection"]["unit_ids"]
    assert regen_session["transaction"]["phase"] == "regenerate"
    assert result.stats["kv_memory_report"]["racer_native_protection"]["schema"] == "racer-native-protection-v2"
    generator.close_session()
