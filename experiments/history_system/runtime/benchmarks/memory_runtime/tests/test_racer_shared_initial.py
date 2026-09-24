"""Configured S0 selection with native costs, plus recovery-off parity."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.always_compress import CapacityInfeasible
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.candidate_algorithms.repacking import repack
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.racer.config import BackendConfig
from benchmarks.memory_runtime.racer.native_initial import NativeInitialFactory
from benchmarks.memory_runtime.racer.policies import InitialOnlyPolicy
from benchmarks.memory_runtime.source_allocation import SourceAllocatedS0Controller
from benchmarks.memory_runtime.source_needs import SourceRequest
from benchmarks.memory_runtime.tests.test_candidate_allocation import (
    Tokenizer, messages, packing, policy, tool_pair,
)
from benchmarks.memory_runtime.tests.test_source_allocation import request as producer_request
from benchmarks.memory_runtime.candidate_algorithms import (
    ALL_VARIANTS, REPAIR_VARIANTS, INITIAL_VIEW_BACKBONES, STATIC_EXTENSION_VARIANTS, C1_V2_VARIANTS,
)


def native(cls=EventNativeS0Controller, budget=1024):
    backend = BackendConfig(backend="h2o", policy="off", history_budget_tokens=budget,
                            allocation="racer_s0")
    return NativeInitialFactory(backend)(cls, Tokenizer(),
        packing={**packing(), "ratios": [4, 8]}, policy=policy(budget))


def prepare(base, rows=None, session="shared"):
    return base.prepare({"session_id": session, "decision_key": "d1",
                         "messages": rows or messages(), "tools": []},
                        ratio=8, max_new_tokens=8)


@pytest.mark.parametrize("cls", [EventNativeS0Controller, SourceAllocatedS0Controller])
def test_shared_policy_runs_its_real_selection_with_native_accounting(cls):
    base = native(cls)
    prepared = prepare(base)
    memory = prepared.memory
    assert isinstance(base, cls)
    assert memory.initial_s0_messages == memory.recovery_messages
    assert memory.initial_s0_source_indices == memory.native_evidence_source_indices
    assert memory.recovered_source_indices == ()
    assert memory.native_evidence_tokens == memory.recovery_tokens > 0
    assert memory.retained_history_cap + memory.native_evidence_tokens <= 1024
    assert memory.retained_history_min_tokens == 1
    assert memory.chunks == prepared.eligible_chunks == ()
    assert "failed_operation_cue" in prepared.metadata
    assert prepared.metadata["native_initial_allocation"]["selection_policy_class"] == cls.__name__
    assert prepared.metadata["source_coverage"]["complete_history_coverage"] is False
    assert history_budget_receipt(memory, prepared.metadata, base,
                                  ratio=8, phase="draft")["status"] == "passed"
    assert all(row["history_gist_tokens"] == 0 for row in prepared.metadata["per_ratio"].values())
    assert prepared.metadata["per_ratio"]["4"]["history_bytes"] == prepared.metadata["per_ratio"]["8"]["history_bytes"]


def test_shared_lexical_selector_is_used_without_approximate_replacement(monkeypatch):
    seen = []
    original = EventNativeS0Controller._lexical_request

    def tracked(self, store, tools, raw, **kwargs):
        seen.append((store.session_id, tuple(raw)))
        return original(self, store, tools, raw, **kwargs)

    monkeypatch.setattr(EventNativeS0Controller, "_lexical_request", tracked)
    prepared = prepare(native())
    assert seen
    assert "latest_complete_tool_protection" in prepared.metadata
    assert prepared.metadata["latest_complete_tool_protection"]["initially_protected"]
    assert prepared.metadata["source_needs"]["strategy"] == "lexical"


def test_native_required_producer_cannot_be_faked_as_a_cheap_gist():
    payload = producer_request()
    c2kv = SourceAllocatedS0Controller(Tokenizer(), packing={**packing(), "ratios": [8]},
                                      policy=policy(128))
    packed = c2kv.prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    assert packed.memory.view.gist_source_indices == (1,)
    with pytest.raises(CapacityInfeasible):
        native(SourceAllocatedS0Controller, budget=128).prepare(payload, ratio=8, max_new_tokens=8)


def test_native_optional_source_must_fit_exact_evidence_and_minimum_pool(monkeypatch):
    base = native(SourceAllocatedS0Controller, budget=128)
    monkeypatch.setattr(base, "_lexical_request", lambda *args, **kwargs: (
        SourceRequest((), (), "no_lexical_match"), None, {"status": "no_lexical_match"}))
    rows = [{"role": "user", "content": "Old"},
            *tool_pair(1, result={"value": " ".join(["old"] * 500)}),
            {"role": "user", "content": "Current"},
            *tool_pair(2, result={"value": "ok"})]
    prepared = prepare(base, rows)
    assert 2 not in prepared.memory.native_evidence_source_indices
    assert prepared.memory.retained_history_cap >= 1
    assert prepared.metadata["source_coverage"]["compact_candidates_do_not_imply_source_coverage"]


@pytest.mark.parametrize("cls", [EventNativeS0Controller, SourceAllocatedS0Controller])
def test_recovery_off_preserves_initial_selection_and_protection(cls):
    ordinary = prepare(native(cls), session="ordinary")
    off = InitialOnlyPolicy(native(cls), {"variant": "c1_v2"})
    prepared = prepare(off, session="off")
    assert ordinary.memory.source_messages == prepared.memory.source_messages
    assert ordinary.memory.native_evidence_tokens == prepared.memory.native_evidence_tokens
    assert ordinary.memory.initial_s0_source_indices == prepared.memory.initial_s0_source_indices
    assert prepared.metadata["recovery_disabled_ablation"]["initial_policy_preserved"]
    assert off.reconsider(prepared, [], draft_text="Done")["regenerate"] is False


def test_source_recovery_keeps_initial_admission_fields(monkeypatch):
    base = native(SourceAllocatedS0Controller, budget=256)
    monkeypatch.setattr(base, "_lexical_request", lambda *args, **kwargs: (
        SourceRequest((), (), "no_lexical_match"), None, {"status": "no_lexical_match"}))
    rows = [{"role": "user", "content": "Old"},
            *tool_pair(1, result={"value": " ".join(["old"] * 300)}),
            {"role": "user", "content": "Current"},
            *tool_pair(2, result={"value": "ok"})]
    from benchmarks.memory_runtime.tests.test_source_allocation import wrapped
    prepared = prepare(wrapped(base), rows)
    candidate = prepared.memory.view.gist_event_ids[0]
    measure, _, receipt = repack(base, prepared, candidate=candidate)
    assert measure is None and receipt["status"] == "abstained"
    unchanged, _, _ = repack(base, prepared)
    assert unchanged.memory.initial_s0_messages == prepared.memory.initial_s0_messages
    assert unchanged.memory.native_evidence_tokens == prepared.memory.native_evidence_tokens


@pytest.mark.parametrize("variant", ALL_VARIANTS)
@pytest.mark.parametrize("backend_name", ["h2o", "c2kv"])
def test_public_factory_keeps_each_configured_initial_policy_when_recovery_is_off(monkeypatch, variant, backend_name):
    from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
    from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
    from benchmarks.memory_runtime.event_native_s0_policy import S0_CONFIG_DEFAULTS
    from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
    from benchmarks.memory_runtime.candidate_algorithms.initial_view import STATIC_INITIAL_VIEW
    from benchmarks.memory_runtime.candidate_algorithms.static_extensions import extension_fields
    from benchmarks.memory_runtime.candidate_algorithms.c1_v2 import c1_v2_fields
    from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact", lambda value: Risk(0.1))
    candidate = {"variant": variant, "risk_threshold": 0.5, "risk_artifact": "fixture"}
    if variant in INITIAL_VIEW_BACKBONES:
        candidate.update(initial_view=STATIC_INITIAL_VIEW, recovery_backbone=INITIAL_VIEW_BACKBONES[variant],
                         proof_registry_version="verified-binding-rules-v1")
    elif variant in STATIC_EXTENSION_VARIANTS:
        candidate.update(extension_fields(variant))
    elif variant in C1_V2_VARIANTS:
        candidate.update(c1_v2_fields(variant))
        if variant == "c1_v2_probe":
            from benchmarks.memory_runtime.tests.test_racer_ablation_variants import targets_fields
            candidate.update(targets_fields({"task": "turn-0/step-0"}))
    results = []
    for mode in ("on", "protected_off"):
        backend = {"schema": "racer-backend-v2", "backend": backend_name, "policy": variant,
                   "mode": mode, "history_budget_tokens": 1024, "allocation": "racer_s0",
                   "detector_calibration": "not_used" if mode == "protected_off" or variant in REPAIR_VARIANTS
                       else "reference" if backend_name == "c2kv" else "frozen_c2kv_unvalidated_transfer"}
        control = build_event_native_controller(Tokenizer(),
            s0_config={**S0_CONFIG_DEFAULTS, "racer_backend": backend, "candidate_algorithm": candidate},
            packing={**packing(), "ratios": [8]}, policy=policy(1024), model_context=100000,
            benchmark="bfcl", view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY)
        prepared = prepare(control)
        if backend_name != "c2kv":
            assert prepared.metadata["native_initial_allocation"]["selection_and_protection_policy_reused"]
        if mode == "protected_off":
            def forbidden(*args, **kwargs):
                raise AssertionError("Recovery-off called a recovery/commit hook")
            wrapper = control.inner
            for hook in InitialOnlyPolicy._disabled_hooks:
                monkeypatch.setattr(wrapper.inner, hook, forbidden, raising=False)
                assert getattr(control, hook, None) is None
            monkeypatch.setattr(wrapper.inner, "reconsider", forbidden)
            assert not control.reconsider(prepared, [], draft_text="Done")["regenerate"]
        results.append(prepared)
    assert results[0].memory == results[1].memory
    identity_fields = ("event_native_s0_version", "candidate_allocation_version")
    assert any(field in results[0].metadata for field in identity_fields)
    assert tuple(results[0].metadata.get(field) for field in identity_fields) == tuple(
        results[1].metadata.get(field) for field in identity_fields)
