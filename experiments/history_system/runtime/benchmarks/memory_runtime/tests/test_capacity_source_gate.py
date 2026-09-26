"""CPU contracts for the capacity-only C1 source allocation gate."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.source_packing import SourceMemoryView

from benchmarks.memory_runtime.always_compress import CapacityInfeasible
from benchmarks.memory_runtime.candidate_algorithms.capacity_fallback import (
    build_capacity_fallback_allocator,
)
from benchmarks.memory_runtime.candidate_algorithms.capacity_source_gate import (
    POLICY_VERSION, CapacityGatedSourceAllocator,
)
from benchmarks.memory_runtime.candidate_algorithms.repacking import repack
from benchmarks.memory_runtime.event_native_s0_policy import S0_CONFIG_DEFAULTS
from benchmarks.memory_runtime.policy import PolicyInputError
from benchmarks.memory_runtime.recovery.config import E1_RECOVERY_VERSION
from benchmarks.memory_runtime.recovery.orchestrator import EventNativeRecoveryController
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, packing, policy
from benchmarks.memory_runtime.tests.test_source_allocation import request


def arguments(budget):
    return dict(packing={**packing(), "ratios": [8]}, policy=policy(budget),
                s0_config=S0_CONFIG_DEFAULTS, benchmark="bfcl", model_context=None)


def incumbent(budget):
    return build_capacity_fallback_allocator(
        Tokenizer(), **arguments(budget), terminal_tool_rescue=True)


def gated(budget):
    return CapacityGatedSourceAllocator(Tokenizer(), **arguments(budget))


def with_recovery(base):
    return EventNativeRecoveryController(
        base, {"schema": E1_RECOVERY_VERSION, "gate": "disabled"}, benchmark="bfcl")


@pytest.mark.parametrize("budget,stage", [(512, None), (96, "compress_complete_tool_arguments")])
def test_incumbent_success_preserves_prepared_identity_and_behavior(monkeypatch, budget, stage):
    payload = request()
    gate = gated(budget)
    baseline = incumbent(budget)
    monkeypatch.setattr(gate.source_allocator, "prepare",
                        lambda *args, **kwargs: pytest.fail("source rescue ran after incumbent success"))

    prepared = gate.prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    expected = baseline.prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    assert prepared is gate.incumbent._sessions[payload["session_id"]].decisions["d1"][1]
    assert gate.prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8) is prepared
    assert prepared.memory == expected.memory
    assert prepared.metadata == expected.metadata
    assert prepared.eligible_chunks == expected.eligible_chunks
    assert prepared.metadata.get("capacity_fallback", {}).get("stage") == stage
    assert "capacity_source_gate" not in prepared.metadata
    assert gate.reconsider(prepared, [], draft_text="done") == baseline.reconsider(
        expected, [], draft_text="done")


def test_source_rescue_requires_terminal_incumbent_failure_and_shares_lifecycle(monkeypatch):
    payload = request()
    with pytest.raises(CapacityInfeasible):
        incumbent(64).prepare(payload, ratio=8, max_new_tokens=8)

    gate = gated(64)
    prepared = gate.prepare(payload, ratio=8, max_new_tokens=8)
    assert isinstance(prepared.memory.view, SourceMemoryView)
    assert prepared.metadata["capacity_source_gate"] == {
        "version": POLICY_VERSION,
        "trigger": "incumbent_c1_capacity_infeasible",
        "incumbent_terminal_rescue_exhausted": True,
        "source_allocator_version": "c2kv-source-budget-allocation-v1",
    }
    assert prepared.metadata["source_allocation"]["phase"] == "after_incumbent_capacity_failure"
    assert prepared.metadata["source_allocation"]["legacy_fallback_invoked"] is True
    assert gate.source_allocator._sessions is gate.incumbent._sessions
    assert gate.source_allocator._owner is gate.incumbent._owner
    assert gate.prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8) is prepared
    assert gate.incumbent._sessions[payload["session_id"]].decisions["d1"][1] is prepared

    next_payload = {**payload, "decision_key": "d2",
                    "messages": [*payload["messages"], {"role": "user", "content": "Continue."}]}
    monkeypatch.setattr(gate.source_allocator, "prepare", lambda *args, **kwargs: pytest.fail(
        "source rescue persisted after the previous decision"))
    following = gate.prepare(next_payload, ratio=8, max_new_tokens=8)
    assert following.metadata["decision_index"] == 2
    assert not isinstance(following.memory.view, SourceMemoryView)
    assert "capacity_source_gate" not in following.metadata
    assert gate.incumbent._sessions[payload["session_id"]].decisions["d2"][1] is following


def test_both_allocators_failing_remains_typed_capacity_infeasible():
    with pytest.raises(CapacityInfeasible):
        incumbent(1).prepare(request(), ratio=8, max_new_tokens=8)
    with pytest.raises(CapacityInfeasible):
        gated(1).prepare(request(), ratio=8, max_new_tokens=8)


def test_source_rescue_t02_candidate_repack_abstains_under_small_budget():
    gate = gated(64)
    prepared = with_recovery(gate).prepare(request(), ratio=8, max_new_tokens=8)
    assert isinstance(prepared.memory.view, SourceMemoryView)
    candidate = prepared.memory.view.gist_event_ids[0]
    measure, metadata, receipt = repack(gate, prepared, candidate=candidate)
    assert measure is metadata is None
    assert receipt["status"] == "abstained"
    assert receipt["candidate_event_id"] == candidate
    assert prepared.metadata["source_allocation"]["legacy_fallback_invoked"] is True


def test_non_capacity_error_does_not_dispatch_source(monkeypatch):
    gate = gated(64)
    original = gate.prepare(request(), ratio=8, max_new_tokens=8)
    assert isinstance(original.memory.view, SourceMemoryView)
    changed = request()
    changed["messages"][0]["content"] = "Different request"
    monkeypatch.setattr(gate.source_allocator, "prepare",
                        lambda *args, **kwargs: pytest.fail("source rescue masked input error"))
    with pytest.raises(PolicyInputError, match="history was truncated or rewritten"):
        gate.prepare(changed, ratio=8, max_new_tokens=8)


@pytest.mark.parametrize("budget,source_selected", [(64, True), (96, False)])
def test_reconsider_and_repack_follow_the_selected_allocator(monkeypatch, budget, source_selected):
    gate = gated(budget)
    wrapped = with_recovery(gate)
    prepared = wrapped.prepare(request(), ratio=8, max_new_tokens=8)
    assert isinstance(prepared.memory.view, SourceMemoryView) is source_selected

    selected = gate.source_allocator if source_selected else gate.incumbent.base
    unselected = gate.incumbent.base if source_selected else gate.source_allocator
    selected_reconsider = selected.reconsider
    calls = []

    def observed_reconsider(*args, **kwargs):
        calls.append("selected")
        return selected_reconsider(*args, **kwargs)

    monkeypatch.setattr(selected, "reconsider", observed_reconsider)
    monkeypatch.setattr(unselected, "reconsider",
                        lambda *args, **kwargs: pytest.fail("wrong reconsider allocator"))
    result = wrapped.reconsider(prepared, [], draft_text="done")
    assert not result["regenerate"]
    assert calls == ["selected"]

    if source_selected:
        original_repack = gate.source_allocator.repack_sources

        def observed_repack(*args, **kwargs):
            calls.append("source_repack")
            return original_repack(*args, **kwargs)

        monkeypatch.setattr(gate.source_allocator, "repack_sources", observed_repack)
    else:
        monkeypatch.setattr(gate.source_allocator, "repack_sources",
                            lambda *args, **kwargs: pytest.fail("source repack ran for incumbent view"))
    measure, metadata, receipt = repack(gate, prepared)
    assert measure is not None and receipt["status"] == "admitted"
    if source_selected:
        assert calls[-1] == "source_repack"
        assert measure.memory == prepared.memory
        assert metadata["capacity_source_gate"] == prepared.metadata["capacity_source_gate"]
    else:
        assert "source_repack" not in calls
        assert metadata["capacity_fallback"]["argument_projections"]
