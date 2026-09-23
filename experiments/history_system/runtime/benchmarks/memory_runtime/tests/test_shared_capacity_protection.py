"""CPU integration contracts for the shared native protection capacity gate."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.source_packing import SourceMemoryView

from benchmarks.memory_runtime.always_compress import CapacityInfeasible
from benchmarks.memory_runtime.backend_capacity import BackendCapacityConstraints, capacity_scope
from benchmarks.memory_runtime.candidate_algorithms.capacity_fallback import (
    build_capacity_fallback_allocator,
)
from benchmarks.memory_runtime.candidate_algorithms.capacity_source_gate import (
    CapacityGatedSourceAllocator,
)
from benchmarks.memory_runtime.candidate_algorithms.repacking import repack
from benchmarks.memory_runtime.event_native_s0_policy import S0_CONFIG_DEFAULTS
from benchmarks.memory_runtime.initial_factory import instantiate_composed_initial
from benchmarks.memory_runtime.racer.config import BackendConfig
from benchmarks.memory_runtime.racer.native_initial import NativeInitialFactory
from benchmarks.memory_runtime.recovery.config import E1_RECOVERY_VERSION
from benchmarks.memory_runtime.recovery.orchestrator import EventNativeRecoveryController
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, packing, policy
from benchmarks.memory_runtime.tests.test_source_allocation import request


NATIVE_BACKENDS = ("h2o", "snapkv", "streamingllm", "pyramidkv", "commitkv", "agentkv")
BUDGET = 384
FULL_PROTECTION_TOKENS = 339
SMALL_PROTECTION_TOKENS = 20
PENDING_FLOOR = 64


def _arguments(budget):
    return dict(packing={**packing(), "ratios": [8]}, policy=policy(budget),
                s0_config=S0_CONFIG_DEFAULTS, benchmark="bfcl", model_context=None)


def _backend(name, budget):
    return BackendConfig(backend=name, policy="off", history_budget_tokens=budget,
                         allocation="racer_s0")


def _gate(name="h2o", budget=BUDGET):
    return instantiate_composed_initial(
        CapacityGatedSourceAllocator, Tokenizer(),
        initial_allocator_factory=NativeInitialFactory(_backend(name, budget)),
        **_arguments(budget))


def _incumbent(name, budget):
    return build_capacity_fallback_allocator(
        Tokenizer(), **_arguments(budget), terminal_tool_rescue=True,
        initial_allocator_factory=NativeInitialFactory(_backend(name, budget)))


def _constraint(payload, budget, floor=PENDING_FLOOR, *, stage="draft", sources=(1,),
                release="never"):
    return BackendCapacityConstraints(
        session_id=payload["session_id"], decision_key=payload["decision_key"],
        stage=stage, history_budget_tokens=budget, mandatory_history_tokens=floor,
        mandatory_source_indices=sources, release=release,
        provenance="scripted_backend_receipt")


def _wrapped(gate):
    return EventNativeRecoveryController(
        gate, {"schema": E1_RECOVERY_VERSION, "gate": "disabled"}, benchmark="bfcl")


@pytest.mark.parametrize("backend", NATIVE_BACKENDS)
def test_six_native_backends_share_one_gate_and_preserve_successful_incumbent(backend):
    payload = request()
    gate = _gate(backend, BUDGET)
    baseline = _incumbent(backend, BUDGET)
    prepared = gate.prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    expected = baseline.prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)

    assert gate.incumbent._sessions is gate.source_allocator._sessions is gate._sessions
    assert gate.incumbent._owner is gate.source_allocator._owner is gate._owner
    assert gate._sessions[payload["session_id"]].decisions[payload["decision_key"]][1] is prepared
    assert gate.prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8) is prepared
    assert prepared.memory == expected.memory
    assert prepared.metadata == expected.metadata
    assert prepared.eligible_chunks == expected.eligible_chunks == ()
    assert prepared.memory.native_evidence_tokens == FULL_PROTECTION_TOKENS
    assert "capacity_source_gate" not in prepared.metadata


@pytest.mark.parametrize("backend", ["h2o", "commitkv", "c2kv"])
def test_long_producer_uses_small_exact_protection_and_native_pool(backend):
    payload = request()
    with pytest.raises(CapacityInfeasible):
        _incumbent(backend, 64).prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)

    prepared = _gate(backend, 64).prepare(payload, ratio=8, max_new_tokens=8)
    memory = prepared.memory
    assert isinstance(memory.view, SourceMemoryView)
    assert memory.native_evidence_tokens == SMALL_PROTECTION_TOKENS
    assert memory.native_evidence_source_indices == (0,)
    assert memory.retained_history_min_tokens == 1
    assert memory.retained_history_cap + memory.native_evidence_tokens <= 64
    assert memory.chunks == prepared.eligible_chunks == ()
    assert prepared.metadata["actual_gist_tokens"] == 0
    assert prepared.metadata["source_allocation"]["unprotected_history_source_indices"] == [1]
    assert prepared.metadata["source_allocation"]["complete_source_coverage_guaranteed"] is False
    assert prepared.metadata["capacity_source_gate"]["phase"] == "initial"


@pytest.mark.parametrize("backend", ["h2o", "c2kv"])
def test_native_residual_pool_skips_inapplicable_gist_retries(monkeypatch, backend):
    gate = _gate(backend, 64)
    assert gate.incumbent.base.supports_gist_capacity_fallback is False
    monkeypatch.setattr(gate.incumbent, "_attempt", lambda *args, **kwargs: pytest.fail(
        "native fallback cannot measure a complete-source gist"))
    monkeypatch.setattr(gate.incumbent, "_terminal_failure", lambda *args, **kwargs: pytest.fail(
        "native fallback cannot project a source into gist"))

    prepared = gate.prepare(request(), ratio=8, max_new_tokens=8)
    assert prepared.memory.native_evidence_tokens == SMALL_PROTECTION_TOKENS
    assert prepared.metadata["capacity_source_gate"]["incumbent_terminal_rescue_exhausted"] is False
    assert prepared.metadata["capacity_source_gate"]["incumbent_gist_fallback"] == {
        "status": "inapplicable",
        "reason": "native_residual_pool_has_no_complete_source_gist",
    }
    assert prepared.metadata["native_initial_allocation"][
        "required_sources_use_exact_native_admission"] is False


@pytest.mark.parametrize("backend", ["h2o", "commitkv", "c2kv"])
def test_current_pending_floor_selects_small_protection_only_when_needed(backend):
    payload = request()
    ordinary = _gate(backend).prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    assert ordinary.memory.native_evidence_tokens == FULL_PROTECTION_TOKENS
    assert FULL_PROTECTION_TOKENS <= BUDGET
    assert FULL_PROTECTION_TOKENS + PENDING_FLOOR > BUDGET
    assert SMALL_PROTECTION_TOKENS + PENDING_FLOOR <= BUDGET

    gate = _gate(backend)
    with capacity_scope(_constraint(payload, BUDGET)):
        protected = gate.prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    assert isinstance(protected.memory.view, SourceMemoryView)
    assert protected.memory.native_evidence_tokens == SMALL_PROTECTION_TOKENS
    assert protected.memory.retained_history_min_tokens == PENDING_FLOOR
    assert protected.memory.retained_history_cap + protected.memory.native_evidence_tokens <= BUDGET
    assert protected.metadata["capacity_source_gate"]["phase"] == "initial"


def test_draft_pending_does_not_release_when_its_source_is_selected_exact():
    payload = request()
    with capacity_scope(_constraint(payload, BUDGET, stage="draft", sources=(0,),
                                    release="replaced_source_message")):
        prepared = _gate("commitkv").prepare(payload, ratio=8, max_new_tokens=8)
    assert prepared.memory.native_evidence_source_indices == (0,)
    assert prepared.memory.native_evidence_tokens == SMALL_PROTECTION_TOKENS
    assert prepared.memory.retained_history_min_tokens == PENDING_FLOOR
    assert prepared.metadata["capacity_source_gate"]["phase"] == "initial"


def test_repack_rejects_a_draft_constraint_bound_to_the_recovery_stage():
    payload = request()
    gate = _gate("commitkv")
    prepared = _wrapped(gate).prepare(payload, ratio=8, max_new_tokens=8)
    target = prepared.metadata["eligible_extraction"]["eligible_event_ids"][0]
    with capacity_scope(_constraint(payload, BUDGET, stage="draft")):
        with pytest.raises(ValueError, match="another stage"):
            repack(gate, prepared, candidate=target)


@pytest.mark.parametrize("backend", ["h2o", "commitkv"])
def test_no_candidate_repack_converts_incumbent_event_view_to_disjoint_source_view(backend):
    payload = request()
    gate = _gate(backend)
    prepared = _wrapped(gate).prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    original_memory = prepared.memory
    assert original_memory.native_evidence_tokens == FULL_PROTECTION_TOKENS

    with capacity_scope(_constraint(payload, BUDGET, stage="regeneration", sources=(2,))):
        measured, metadata, receipt = repack(gate, prepared)
    assert receipt["status"] == "admitted"
    assert measured is not None
    assert isinstance(measured.memory.view, SourceMemoryView)
    assert measured.memory.native_evidence_tokens == SMALL_PROTECTION_TOKENS
    assert measured.memory.retained_history_min_tokens == PENDING_FLOOR
    assert set(measured.memory.view.raw_source_indices).isdisjoint(
        measured.memory.view.gist_source_indices)
    assert measured.memory.chunks == ()
    assert metadata["actual_gist_tokens"] == 0
    assert metadata["source_coverage"]["complete_history_coverage"] is False
    assert metadata["source_allocation"]["complete_source_coverage_guaranteed"] is False
    assert metadata["capacity_source_gate"]["phase"] == "recovery"
    assert prepared.memory is original_memory
    assert prepared.memory.native_evidence_tokens == FULL_PROTECTION_TOKENS


def test_b256_complete_protection_232_plus_pending_28_uses_small_exact_source():
    payload = request()
    payload["messages"][1]["tool_calls"][0]["function"]["arguments"] = json.dumps({
        "file_name": "report.txt", "content": " ".join(["long"] * 213),
    })
    gate = _gate("commitkv", 256)
    prepared = _wrapped(gate).prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    assert prepared.memory.native_evidence_tokens == 232
    assert 232 + 28 > 256
    target = prepared.metadata["eligible_extraction"]["eligible_event_ids"][0]
    assert set(prepared._store.event(target).source_indices) == {0}

    with capacity_scope(_constraint(payload, 256, floor=28, stage="regeneration",
                                    sources=(2,), release="replaced_source_message")):
        measured, metadata, receipt = repack(gate, prepared, candidate=target)
    assert receipt["status"] == "admitted"
    assert measured.memory.native_evidence_tokens == SMALL_PROTECTION_TOKENS
    assert measured.memory.retained_history_min_tokens == 28
    assert measured.memory.native_evidence_source_indices == (0,)
    assert measured.memory.native_evidence_tokens + 28 <= 256
    assert target in measured.memory.view.raw_event_ids
    assert metadata["capacity_source_gate"]["phase"] == "recovery"
    assert metadata["actual_gist_tokens"] == 0
    assert measured.memory.chunks == ()


def test_recovery_drops_surrounding_protection_but_keeps_complete_target_exact():
    payload = request()
    gate = _gate("commitkv")
    prepared = _wrapped(gate).prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    assert prepared.memory.native_evidence_tokens == FULL_PROTECTION_TOKENS
    target = prepared.metadata["eligible_extraction"]["eligible_event_ids"][0]
    target_sources = set(prepared._store.event(target).source_indices)
    assert target_sources == {0}

    with capacity_scope(_constraint(payload, BUDGET, stage="regeneration", sources=(2,),
                                    release="replaced_source_message")):
        measured, metadata, receipt = repack(gate, prepared, candidate=target)
    assert receipt["status"] == "admitted"
    assert measured is not None
    assert metadata["capacity_source_gate"]["phase"] == "recovery"
    assert target_sources <= set(measured.memory.native_evidence_source_indices)
    assert target in measured.memory.view.raw_event_ids
    assert measured.memory.native_evidence_source_indices == (0,)
    assert measured.memory.native_evidence_tokens == SMALL_PROTECTION_TOKENS
    assert measured.memory.retained_history_min_tokens == PENDING_FLOOR
    assert measured.memory.retained_history_cap + measured.memory.native_evidence_tokens <= BUDGET
    assert measured.memory.chunks == ()
    assert metadata["actual_gist_tokens"] == 0


def test_recovery_abstains_if_complete_target_and_pending_floor_cannot_both_fit():
    payload = request()
    gate = _gate("commitkv", 64)
    prepared = _wrapped(gate).prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    target = prepared.metadata["eligible_extraction"]["eligible_event_ids"][1]
    assert set(prepared._store.event(target).source_indices) == {1, 2}
    assert prepared.memory.native_evidence_tokens == SMALL_PROTECTION_TOKENS

    with capacity_scope(_constraint(payload, 64, floor=16, stage="regeneration", sources=(2,),
                                    release="replaced_source_message")):
        measured, metadata, receipt = repack(gate, prepared, candidate=target)
    assert measured is metadata is None
    assert receipt["status"] == "abstained"
    assert receipt["candidate_event_id"] == target
    assert receipt["complete_event_raw_required"] is True
    assert prepared.memory.native_evidence_tokens == SMALL_PROTECTION_TOKENS
    assert prepared.metadata["capacity_source_gate"]["phase"] == "initial"
