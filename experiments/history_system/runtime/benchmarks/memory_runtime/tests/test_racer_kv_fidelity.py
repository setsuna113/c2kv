"""Compare RACER's persistent KV wire with the registered proxy methods."""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[3]
PAPER = RUNTIME.parents[2]
ENGINE_TRANSACTION = (PAPER.parent / "sglang-paper" / "python" / "sglang" /
                      "srt" / "mem_cache" / "racer_transaction.py")
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "python"), str(PAPER)]

from benchmarks.memory_runtime.candidate_algorithms.controller import CandidateRecoveryController
from benchmarks.memory_runtime.candidate_algorithms.repacking import repack
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.racer.config import BackendConfig
from benchmarks.memory_runtime.racer.generator import PersistentRacerGenerator
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.tests.test_racer_composition import allocator, config, prepare
from benchmarks.memory_runtime.tests.test_racer_transport import Native, Decoder, context


PROXY_ARMS = {
    "h2o": "history_kv_h2o_r25_persistent",
    "snapkv": "history_kv_snapkv_r25_persistent",
    "pyramidkv": "history_kv_pyramidkv_r25_persistent",
    "commitkv": "commitkv",
    "agentkv": "agentkv",
}


def _proxy_arms():
    # The runtime snapshot also has a benchmarks.arms module. Load the paper
    # registry by its source path so this comparison cannot select that copy.
    name = "_paper_proxy_arms_for_racer_test"
    spec = importlib.util.spec_from_file_location(name, PAPER / "benchmarks" / "arms.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("backend,arm_name", PROXY_ARMS.items())
def test_absolute_racer_spec_preserves_registered_proxy_method(backend, arm_name):
    arms = _proxy_arms()
    proxy = arms.history_kv_spec(arms.ARMS[arm_name])
    racer = config(backend, budget=256).history_spec()
    for key in ("method", "backend", "persistent_session", "recent_window",
                "kernel_size", "pooling", "h2o_recent_fraction"):
        assert racer[key] == proxy[key]
    assert racer["retention_ratio"] is None
    assert racer["target_tokens"] == 256
    assert config(backend, budget=256).history_spec(200)["target_tokens"] == 200


@pytest.mark.parametrize("override", [
    {"method": "snapkv_persistent"},
    {"backend": "reference_attention"},
    {"persistent_session": False},
    {"retention_ratio": 0.25},
    {"target_tokens": 128},
    {"recent_window": 32},
    {"kernel_size": 3},
    {"pooling": "maxpool"},
    {"h2o_recent_fraction": 0.8},
])
def test_explicit_h2o_settings_cannot_silently_change_proxy_method(override):
    with pytest.raises(ValueError, match="differs from the registered proxy arm"):
        BackendConfig.parse({"schema": "racer-backend-v1", "backend": "h2o",
            "policy": "off", "history_budget_tokens": 256,
            "backend_config": override})


def test_agentkv_first_turn_uses_reference_runtime_and_transaction():
    native = Native()
    generator = PersistentRacerGenerator(native, Decoder(), config("agentkv"))
    memory = prepare(allocator("agentkv"), rows=[
        {"role": "system", "content": "Use APIs."},
        {"role": "user", "content": "Start"},
    ]).memory
    assert memory.history_message_count == memory.history_start_message_count
    with generator.decision_scope(session_id="s"):
        generator.generate(memory, ratio=8, max_new_tokens=32, trace_context=context())
    request = native.requests[-1][1]
    hint = request["c2kv_kv_memory_hint"]
    assert hint["history_kv_method"] == "agentkv"
    assert hint["history_kv_backend"] == "reference_attention"
    assert hint["history_kv_reference_config"]["target_tokens"] == 1024
    assert "history_kv_eviction" not in hint
    assert hint["persistent_history_session"]["transaction"] == {
        "decision_id": "d1", "phase": "draft"}
    assert request["c2kv_use_gist_projection"] is False


def test_off_allocator_does_not_enter_s0_selection(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("S0 selection entered")

    monkeypatch.setattr(EventNativeS0Controller, "_prepare_view", forbidden)
    monkeypatch.setattr(EventNativeS0Controller, "_try_measure", forbidden)
    backend = allocator("agentkv")
    prepared = prepare(backend)
    result = backend.reconsider(prepared, [], draft_text="Done")
    assert result["regenerate"] is False
    assert prepared.memory.backend_identity == "agentkv"


def _engine_budget_function():
    """Exercise the engine's real budget function without importing its GPU stack."""
    tree = ast.parse(ENGINE_TRANSACTION.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and node.name in {"enforce_request_budget", "transaction_config",
                                     "RacerCapacityInfeasible"}]
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]),
                 str(ENGINE_TRANSACTION), "exec"), namespace)
    return namespace["enforce_request_budget"]


def test_commitkv_recovery_keeps_total_budget_and_clamps_effective_target():
    control = CandidateRecoveryController(allocator("commitkv"),
                                          {"variant": "static_t02"}, risk_model=Risk(0.9))
    prepared = prepare(control)
    candidate = prepared.memory.view.gist_event_ids[0]
    recovered, _, _ = repack(control.base, prepared, candidate=candidate)
    native = Native()
    generator = PersistentRacerGenerator(native, Decoder(), config("commitkv"))
    with generator.decision_scope(session_id="s"):
        generator.generate(prepared.memory, ratio=8, max_new_tokens=32,
                           trace_context=context())
        generator.generate(recovered.memory, ratio=8, max_new_tokens=32,
                           trace_context=context(phase="regeneration"))
    requests = [body for url, body in native.requests
                if url.endswith("/v1/chat/completions")]
    draft, regeneration = requests
    assert draft["c2kv_kv_memory_hint"]["history_kv_reference_config"]["target_tokens"] == 1024
    hint = regeneration["c2kv_kv_memory_hint"]
    evidence = recovered.memory.recovery_tokens
    assert 0 < evidence < 1024
    assert hint["history_kv_reference_config"]["target_tokens"] == 1024 - evidence
    assert hint["history_kv_eviction"]["target_tokens"] == 1024 - evidence
    # The engine resolves actual evidence spans and retains CommitKV's total
    # session budget separately from this request's effective history cap.
    hint["racer_active_ephemeral_source_spans"] = [[0, evidence]]
    _engine_budget_function()(hint)
    assert hint["history_kv_reference_config"]["target_tokens"] == 1024
    assert hint["history_kv_reference_config"]["racer_effective_target_tokens"] == 1024 - evidence
    assert hint["history_kv_eviction"]["target_tokens"] == 1024 - evidence
    assert hint["racer_budget"]["native_evidence_tokens"] == evidence


def test_reference_history_candidates_cannot_resurrect_evicted_kv():
    import torch

    path = ENGINE_TRANSACTION.with_name("history_kv_reference.py")
    name = "_paper_reference_history_for_racer_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    # Position 20 was evicted on the preceding turn. The next selector sees
    # only the retained old state (10, 30) and the newly prefilled delta (40).
    old = module.ReferenceLayerKV(
        key=torch.tensor([[[10.0], [30.0]]]),
        value=torch.tensor([[[10.0], [30.0]]]),
        positions=torch.tensor([[10, 30]], dtype=torch.long),
    )
    new = torch.tensor([[[40.0]]])
    _, _, candidates = module.merge_reference_candidates(old, new, new, [40])
    assert candidates.tolist() == [[10, 30, 40]]
    selected = module.gather_reference_candidates(
        old, new, new, [40], torch.tensor([[1, 2]]))
    assert selected.positions.tolist() == [[30, 40]]
    assert selected.key.squeeze(-1).tolist() == [[30.0, 40.0]]
