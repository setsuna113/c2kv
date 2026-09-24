"""CPU parity for independent RACER v3 protection and existing recovery."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.candidate_algorithms.c1_v2 import c1_v2_fields
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_s0_policy import S0_CONFIG_DEFAULTS
from benchmarks.memory_runtime.racer.allocator import PersistentHistoryAllocator
from benchmarks.memory_runtime.racer.c2kv_state import C2KVResidentPolicy
from benchmarks.memory_runtime.racer.config import BACKENDS
from benchmarks.memory_runtime.racer.native_protection import NativeProtectionAllocator
from benchmarks.memory_runtime.tests.test_candidate_allocation import (
    Tokenizer, messages, packing, policy,
)
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.tests.test_static_extensions import proof_payload
from benchmarks.memory_runtime.tests.test_verified_binding import _call


BUDGET = 4096
NATIVE_BACKENDS = tuple(backend for backend in BACKENDS if backend != "c2kv")


def _backend(backend, candidate_policy, *, schema="racer-backend-v1", extra=None):
    record = {
        "schema": schema,
        "backend": backend,
        "policy": candidate_policy,
        "history_budget_tokens": BUDGET,
        "detector_calibration": ("not_used" if candidate_policy == "off" else
                                 "reference" if backend == "c2kv" else
                                 "frozen_c2kv_unvalidated_transfer"),
        "allocation": "c2kv_s0" if backend == "c2kv" else "backend_native_persistent",
    }
    if schema == "racer-backend-v3":
        record["extra_protection"] = extra
    return record


def _controller(backend, candidate_policy="off", *, schema="racer-backend-v1", extra=None):
    configuration = {**S0_CONFIG_DEFAULTS,
        "racer_backend": _backend(backend, candidate_policy, schema=schema, extra=extra)}
    if candidate_policy == "static_t02":
        configuration["candidate_algorithm"] = {
            "variant": "static_t02", "risk_threshold": 0.5, "risk_artifact": "fixture"}
    elif candidate_policy == "c1_v2_verified":
        configuration["candidate_algorithm"] = {
            "variant": "c1_v2_verified", "risk_threshold": 0.5,
            "risk_artifact": "fixture", **c1_v2_fields("c1_v2_verified")}
    return build_event_native_controller(
        Tokenizer(), packing={**packing(), "ratios": [8]}, policy=policy(BUDGET),
        view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config=configuration, model_context=100_000, benchmark="bfcl")


def _payload():
    return {"session_id": "protection-parity", "decision_key": "turn-0/step-0",
            "messages": messages(), "tools": []}


def _prepare(controller, payload=None):
    return controller.prepare(payload or _payload(), ratio=8, max_new_tokens=32)


def _unprotected(memory):
    return replace(memory, protection_source_indices=(), protection_event_ids=())


def _intrinsic_state(metadata):
    return {key: value for key, value in metadata["c2kv_resident_state"].items()
            if key not in {"extra_protection", "extra_protection_status"}}


@pytest.mark.parametrize("backend", BACKENDS)
def test_v3_off_preserves_v1_initial_allocation_and_costs(backend):
    old = _controller(backend)
    off = _controller(backend, schema="racer-backend-v3", extra="off")
    old_prepared, off_prepared = _prepare(old), _prepare(off)
    assert off_prepared.memory == old_prepared.memory
    assert off_prepared.memory.view == old_prepared.memory.view
    assert off_prepared.memory.costs(8) == old_prepared.memory.costs(8)
    assert off_prepared.eligible_chunks == old_prepared.eligible_chunks
    for key in ("raw_event_ids", "gist_event_ids", "per_ratio",
                "actual_history_bytes", "actual_total_resident_kv_tokens"):
        assert off_prepared.metadata.get(key) == old_prepared.metadata.get(key)
    assert "native_protection_request" not in off_prepared.metadata
    assert "native_initial_allocation" not in off_prepared.metadata
    if backend == "c2kv":
        assert isinstance(off, C2KVResidentPolicy)
        assert _intrinsic_state(off_prepared.metadata) == _intrinsic_state(old_prepared.metadata)
        assert off_prepared.metadata["c2kv_resident_state"]["extra_protection"] == "off"
    else:
        assert off_prepared.metadata["racer_backend"]["extra_protection"] == "off"
        assert type(off.inner) is PersistentHistoryAllocator
        assert not off_prepared.memory.recovery_messages
        assert (history_budget_receipt(off_prepared.memory, off_prepared.metadata,
                                       off, ratio=8, phase="draft")["status"] == "passed")


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_v3_on_only_adds_native_source_protection_request(backend):
    off = _controller(backend, schema="racer-backend-v3", extra="off")
    on = _controller(backend, schema="racer-backend-v3", extra="on")
    off_prepared, on_prepared = _prepare(off), _prepare(on)
    assert type(on.inner) is NativeProtectionAllocator
    assert _unprotected(on_prepared.memory) == off_prepared.memory
    assert on_prepared.memory.view == off_prepared.memory.view
    assert on_prepared.memory.costs(8) == off_prepared.memory.costs(8)
    assert on_prepared.memory.recovery_messages == off_prepared.memory.recovery_messages == ()
    assert on_prepared.memory.protection_source_indices
    assert set(on_prepared.memory.protection_source_indices) <= set(
        range(on_prepared.memory.history_start_message_count,
              on_prepared.memory.history_message_count))
    request = on_prepared.metadata["native_protection_request"]
    assert request["source_indices"] == list(on_prepared.memory.protection_source_indices)
    assert request["input_rewritten"] is False
    assert request["status"] == "requires_engine_admission"
    assert "native_protection_request" not in off_prepared.metadata


def test_c2kv_v3_protection_switch_keeps_intrinsic_s0_allocation():
    off = _controller("c2kv", schema="racer-backend-v3", extra="off")
    on = _controller("c2kv", schema="racer-backend-v3", extra="on")
    off_prepared, on_prepared = _prepare(off), _prepare(on)
    assert on_prepared.memory == off_prepared.memory
    assert on_prepared.eligible_chunks == off_prepared.eligible_chunks
    assert _intrinsic_state(on_prepared.metadata) == _intrinsic_state(off_prepared.metadata)
    assert on_prepared.metadata["c2kv_resident_state"]["extra_protection_status"] == "intrinsic_c2kv_s0_preserved"
    assert "native_protection_request" not in on_prepared.metadata
    assert on.commit_memory(on_prepared, on_prepared.memory)["status"] == "committed"


@pytest.mark.parametrize("backend", ("snapkv", "commitkv"))
def test_v3_off_preserves_v1_recovery_decision_and_source_repack(monkeypatch, backend):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
        lambda artifact: Risk(0.9))
    old = _controller(backend, "static_t02")
    off = _controller(backend, "static_t02", schema="racer-backend-v3", extra="off")
    on = _controller(backend, "static_t02", schema="racer-backend-v3", extra="on")
    old_prepared, off_prepared, on_prepared = _prepare(old), _prepare(off), _prepare(on)
    candidate = old_prepared.memory.view.gist_event_ids[0]
    monkeypatch.setattr(
        "benchmarks.memory_runtime.candidate_algorithms.controller.select_source_event",
        lambda *args, **kwargs: (candidate, {"ranked_candidate_event_ids": [candidate]}))
    old_result = old.reconsider(old_prepared, [], draft_text="Need the earlier value")
    off_result = off.reconsider(off_prepared, [], draft_text="Need the earlier value")
    on_result = on.reconsider(on_prepared, [], draft_text="Need the earlier value")
    assert old_result["regenerate"] and off_result["regenerate"] and on_result["regenerate"]
    assert off_result["memory"] == old_result["memory"]
    assert on_result["memory"] == old_result["memory"]
    assert off_result["decision"]["reason"] == old_result["decision"]["reason"]
    assert on_result["decision"]["reason"] == old_result["decision"]["reason"]
    assert off_result["decision"]["selection"] == old_result["decision"]["selection"]
    assert on_result["decision"]["selection"] == old_result["decision"]["selection"]
    assert off_result["memory"].recovered_source_indices == old_result["memory"].recovered_source_indices
    assert (history_budget_receipt(off_result["memory"], off_result["metadata"],
                                   off, ratio=8, phase="regeneration")["status"] == "passed")


@pytest.mark.parametrize("backend", ("snapkv", "commitkv"))
def test_v3_protection_does_not_change_verified_binding_commit(monkeypatch, backend):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
        lambda artifact: Risk(0.1))
    controls = (
        _controller(backend, "c1_v2_verified"),
        _controller(backend, "c1_v2_verified", schema="racer-backend-v3", extra="off"),
        _controller(backend, "c1_v2_verified", schema="racer-backend-v3", extra="on"),
    )
    prepared = tuple(_prepare(control, proof_payload()) for control in controls)
    assert prepared[0].memory == prepared[1].memory
    assert _unprotected(prepared[2].memory) == prepared[1].memory
    held = [_call("set_budget_limit", {"access_token": "wrong", "budget_limit": 1500})]
    results = tuple(control.reconsider(view, held, draft_text="call")
                    for control, view in zip(controls, prepared))
    assert all(not result["regenerate"] for result in results)
    assert all(result["decision"]["verified_binding"] ==
               results[0]["decision"]["verified_binding"] for result in results)
    for control, view in zip(controls, prepared):
        control.validate_commit(view, held, draft_text="call")
    finalized = tuple(control.finalize_commit(view, held)
                      for control, view in zip(controls, prepared))
    assert finalized[0] == finalized[1] == finalized[2]
    changed, receipt = finalized[0]
    assert receipt["changed"] and receipt["additional_generations"] == 0
    assert json.loads(changed[0]["function"]["arguments"])["access_token"] != "wrong"
    assert changed[0]["id"] == held[0]["id"]


def test_c2kv_v3_switch_preserves_c1_verified_binding(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
        lambda artifact: Risk(0.1))
    controls = (
        _controller("c2kv", "c1_v2_verified"),
        _controller("c2kv", "c1_v2_verified", schema="racer-backend-v3", extra="off"),
        _controller("c2kv", "c1_v2_verified", schema="racer-backend-v3", extra="on"),
    )
    prepared = tuple(_prepare(control, proof_payload()) for control in controls)
    assert prepared[0].memory == prepared[1].memory == prepared[2].memory
    held = [_call("set_budget_limit", {"access_token": "wrong", "budget_limit": 1500})]
    results = tuple(control.reconsider(view, held, draft_text="call")
                    for control, view in zip(controls, prepared))
    assert all(not result["regenerate"] for result in results)
    assert all(result["decision"]["verified_binding"] ==
               results[0]["decision"]["verified_binding"] for result in results)
    for control, view in zip(controls, prepared):
        control.validate_commit(view, held, draft_text="call")
    finalized = tuple(control.finalize_commit(view, held)
                      for control, view in zip(controls, prepared))
    assert finalized[0] == finalized[1] == finalized[2]
    assert finalized[0][1]["changed"]
