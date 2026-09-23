"""A rejected regeneration must not commit its chunk handles to the session."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.packing import MemoryView, PackedMemory
from history_memory.sglang_generator import (
    SGLangEventNativeError, SGLangEventNativeGenerationResult,
    SGLangEventNativeGenerator,
)


def _generator(tmp_path):
    generator = SGLangEventNativeGenerator(
        "http://127.0.0.1:30000", expected_model_path=tmp_path,
        model_context=100, max_new_tokens=4, max_generation_calls=4,
        max_extraction_calls=4, timeout_seconds=1, eos_token_ids=(0,),
        eos_source="cpu-test",
    )
    generator._model_binding = {"model_path": str(tmp_path)}
    generator._kv_bytes_per_token = 1
    generator._prepare_chunks = lambda memory, ratio, chunks: (
        [{"handle": f"chunk-{memory.system_input_ids[0]}", "token_ids": [7]}], [], [])
    generator._post_native_generate = lambda payload, *args, **kwargs: ({}, 200)
    generator._charge_extractions = lambda response: None

    def result(response, *, payload, scope, **kwargs):
        return SGLangEventNativeGenerationResult(
            (42,), "stop", (-0.1,), {
                "decision_scope_generation_index": scope.generate_calls,
                "sglang_transport": {"generation_id": payload["generation_id"]},
                "session_cache_commit_status": "pending",
            })

    generator._result_from_response = result
    return generator


def _memory(marker):
    return PackedMemory(MemoryView((), ()), (marker,), (8,), (0,), ())


def _generate(generator, marker, phase):
    return generator.generate(
        _memory(marker), ratio=8, max_new_tokens=2,
        trace_context={"attempt_uid": f"d1:{phase}", "phase": phase},
    )


def test_commit_validation_original_fallback_retains_original_handles(tmp_path):
    generator = _generator(tmp_path)
    with generator.decision_scope(session_id="task"):
        original = _generate(generator, 11, "draft")
        rejected = _generate(generator, 22, "regeneration")
        receipt = generator.resolve_decision(
            {"role": "assistant", "content": "original"}, result=original,
            record={"commit_validation": {"selected_generation_index": 0}},
        )
        assert receipt["retained_chunk_handles"] == ["chunk-11"]
    assert generator._session_cache.handles == {"chunk-11"}
    assert original.stats["session_cache_commit_status"] == "committed"
    assert rejected.stats["session_cache_commit_status"] == "discarded_by_resolution"


def test_exact_restore_recovers_generation_handle_registry(tmp_path):
    generator = _generator(tmp_path)
    digest = {name: "a" * 64 for name in
              ("actor_kv", "actor_positions", "backend_stats", "rng")}

    def exact(operation, snapshot_id=None):
        response = {"schema": "c2kv-exact-backend-state-v1",
                    "operation": operation, "snapshot_id": "snapshot-1",
                    "component_digests": digest, "exact": True}
        if operation == "restore":
            response["verified_from_live_state"] = True
        return response

    generator._exact_request = exact
    with generator.decision_scope(session_id="task"):
        original = _generate(generator, 11, "draft")
        snapshot = generator.capture_exact_state()
        _generate(generator, 22, "regeneration")
        generator.restore_exact_state(snapshot)
        assert generator._active_decision_scope.generate_calls == 1
        generator.resolve_decision(
            {"role": "assistant", "content": "original"}, result=original,
            record={"commit_validation": {"selected_generation_index": 0}},
        )
    assert generator._session_cache.handles == {"chunk-11"}


def test_commit_rejects_result_outside_current_scope(tmp_path):
    generator = _generator(tmp_path)
    with generator.decision_scope(session_id="task"):
        result = _generate(generator, 11, "draft")
    with generator.decision_scope(session_id="task"):
        _generate(generator, 22, "draft")
        with pytest.raises(SGLangEventNativeError, match="not in this decision scope"):
            generator.resolve_decision({}, result=result, record={})
