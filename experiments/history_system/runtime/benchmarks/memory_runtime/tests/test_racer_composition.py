"""CPU behavior at the shared policy/persistent adapter seam."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, messages, packing, policy
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.racer.allocator import PersistentHistoryAllocator, PersistentMemory
from benchmarks.memory_runtime.racer.config import BackendConfig
from benchmarks.memory_runtime.candidate_algorithms.controller import CandidateRecoveryController
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.candidate_algorithms import (
    ALL_VARIANTS, REPAIR_VARIANTS, INITIAL_VIEW_BACKBONES, STATIC_EXTENSION_VARIANTS, C1_V2_VARIANTS)


def config(backend="commitkv", mode="off", budget=1024):
    return BackendConfig.parse({"schema": "racer-backend-v1", "backend": backend,
        "policy": mode, "history_budget_tokens": budget,
        "detector_calibration": "not_used" if mode in ("off", *REPAIR_VARIANTS) else "frozen_c2kv_unvalidated_transfer",
        "allocation": "backend_native_persistent"})


def allocator(backend="commitkv", budget=1024):
    geometry = packing()
    geometry["ratios"] = [8]
    return PersistentHistoryAllocator(Tokenizer(), packing=geometry, policy=policy(budget),
                                      backend_config=config(backend, budget=budget))


def prepare(control, rows=None):
    return control.prepare({"session_id": "s", "decision_key": "d1", "messages": rows or messages(),
                            "tools": []}, ratio=8, max_new_tokens=32)


@pytest.mark.parametrize("backend", ["commitkv", "h2o", "snapkv", "streamingllm"])
def test_off_uses_named_backend_and_never_encodes_history(backend):
    base = allocator(backend)
    prepared = prepare(base)
    assert isinstance(prepared.memory, PersistentMemory)
    assert prepared.memory.source_messages
    assert not prepared.memory.chunks and not prepared.eligible_chunks
    assert not base.reconsider(prepared, [], draft_text="Done")["regenerate"]
    assert base.backend_config.history_spec()["method"] == ("snapkv_persistent" if backend == "snapkv" else backend)
    assert history_budget_receipt(prepared.memory, prepared.metadata, base, ratio=8, phase="draft")["status"] == "passed"


@pytest.mark.parametrize("backend", ["commitkv", "h2o", "snapkv", "streamingllm"])
def test_existing_policy_recovers_complete_source_under_same_budget(monkeypatch, backend):
    base = allocator(backend)
    control = CandidateRecoveryController(base, {"variant": "static_t02"}, risk_model=Risk(0.9))
    prepared = prepare(control)
    candidate = prepared.memory.view.gist_event_ids[0]
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.select_source_event",
        lambda *args, **kwargs: (candidate, {"ranked_candidate_event_ids": [candidate]}))
    result = control.reconsider(prepared, [], draft_text="Need the earlier value")
    assert result["regenerate"]
    memory = result["memory"]
    assert memory.source_messages == prepared.memory.source_messages
    assert candidate in memory.view.raw_event_ids
    assert memory.recovery_messages and not memory.chunks
    assert memory.retained_history_cap + memory.recovery_tokens <= memory.history_budget_tokens
    assert history_budget_receipt(memory, result["metadata"], control, ratio=8, phase="regeneration")["status"] == "passed"


def test_too_large_evidence_abstains_without_changing_budget(monkeypatch):
    base = allocator(budget=8)
    control = CandidateRecoveryController(base, {"variant": "static_t02"}, risk_model=Risk(0.9))
    prepared = prepare(control)
    candidate = prepared.memory.view.gist_event_ids[0]
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.select_source_event",
        lambda *args, **kwargs: (candidate, {"ranked_candidate_event_ids": [candidate]}))
    result = control.reconsider(prepared, [], draft_text="Need the earlier value")
    assert not result["regenerate"]
    assert result["memory"] == prepared.memory


def test_transfer_cannot_claim_reference_detector_calibration():
    with pytest.raises(ValueError, match="Detector calibration"):
        BackendConfig.parse({"schema": "racer-backend-v1", "backend": "h2o", "policy": "t02",
                             "history_budget_tokens": 128, "detector_calibration": "reference"})


@pytest.mark.parametrize("variant", ALL_VARIANTS)
@pytest.mark.parametrize("backend", ["commitkv", "h2o", "snapkv", "streamingllm"])
def test_every_existing_policy_constructs_and_keeps_its_identity(monkeypatch, variant, backend):
    from dataclasses import asdict
    from benchmarks.memory_runtime.racer.policies import build_controller
    from benchmarks.memory_runtime.candidate_algorithms.initial_view import STATIC_INITIAL_VIEW
    from benchmarks.memory_runtime.candidate_algorithms.static_extensions import extension_fields
    from benchmarks.memory_runtime.candidate_algorithms.c1_v2 import c1_v2_fields
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact", lambda value: Risk(0.1))
    candidate = {"variant": variant, "risk_threshold": 0.5, "risk_artifact": "fixture"}
    if variant in INITIAL_VIEW_BACKBONES:
        candidate.update(initial_view=STATIC_INITIAL_VIEW, recovery_backbone=INITIAL_VIEW_BACKBONES[variant],
                         proof_registry_version="verified-binding-rules-v1")
    elif variant in STATIC_EXTENSION_VARIANTS:
        candidate.update(extension_fields(variant))
    elif variant in C1_V2_VARIANTS:
        candidate.update(c1_v2_fields(variant))
    cfg = {"schema": "racer-backend-v1", **asdict(config(backend, variant))}
    geometry = packing()
    geometry["ratios"] = [8]
    control = build_controller(Tokenizer(), config={"racer_backend": cfg, "candidate_algorithm": candidate},
                               packing=geometry, policy=policy(1024), model_context=100000, benchmark="bfcl")
    prepared = prepare(control)
    call = {"id": "draft", "type": "function", "function": {"name": "lookup", "arguments": '{"key":3}'}}
    result = control.reconsider(prepared, [call], draft_text="Need the key")
    assert result["metadata"]["candidate_algorithm"]["variant"] == variant
    assert result["metadata"]["racer_backend"]["backend"] == backend
    assert result["metadata"]["candidate_algorithm"]["c2kv_initial_allocation_applied"] is False
