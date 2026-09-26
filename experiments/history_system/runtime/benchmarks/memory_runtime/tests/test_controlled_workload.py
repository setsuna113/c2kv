"""CPU contracts for the opt-in controlled native serving workload."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.event_native import memory_to_dict
from benchmarks.memory_runtime.event_native_api import EventNativeAPI, EventNativeAPIError
from history_memory.controlled_workload import ControlledWorkload, ControlledWorkloadError
from history_memory.packing import EncoderChunk, MemoryView, PackedMemory
from history_memory.sglang_generator import SGLangEventNativeGenerator


def _memory():
    chunk = EncoderChunk("event-1", 0, (1,), 0, 2, (101, 102))
    return PackedMemory(MemoryView((), ()), (11, 12), (13, 14), (0,), (chunk,))


def _generator(monkeypatch):
    monkeypatch.delenv("C2KV_NATIVE_ASYNC_COMPRESSION", raising=False)
    monkeypatch.delenv("C2KV_NATIVE_CROSS_TURN_PREWARM", raising=False)
    generator = SGLangEventNativeGenerator(
        "http://127.0.0.1:30000", expected_model_path="/checkpoint",
        model_context=128, max_new_tokens=4, max_generation_calls=8,
        max_extraction_calls=8, timeout_seconds=1, eos_token_ids=(0,),
        eos_source="cpu-test",
        shadow_feature_config=SimpleNamespace(
            enabled=True, prefill_layer=1, memgen_layer=None),
    )
    generator._model_binding = {"model_path": "/checkpoint"}
    generator._kv_bytes_per_token = 1
    return generator


def _response(payload, memory, *, output=(91, 92), hidden=0.5):
    selected = payload["encoder_chunks"]
    costs = memory.costs(8)
    return {
        "schema": "c2kv-native-packed-generation-response-v1",
        "rid": payload["rid"], "session_id": payload["session_id"],
        "generation_id": payload["generation_id"], "text": "actual",
        "output_ids": list(output), "token_logprobs": [-0.2] * len(output),
        "finish_reason": "length",
        "encoder_chunks": [{
            "chunk_id": row["chunk_id"], "handle": row["handle"],
            "cache_key": "actual-cache-key", "cache_hit": False,
            "original_seq_len": len(row["token_ids"]), "gist_len": 1,
        } for row in selected],
        "compression_chunks": [],
        "costs": {
            **{name: costs[name] for name in (
                "system_tokens", "raw_tokens", "presented_encoder_tokens",
                "gist_tokens", "resident_kv_tokens")},
            "system_prefix_kv_logical_bytes": costs["system_tokens"],
            "gist_prefix_kv_logical_bytes": costs["gist_tokens"],
            "raw_workspace_kv_logical_bytes": costs["raw_tokens"],
            "resident_kv_logical_bytes": costs["resident_kv_tokens"],
        },
        "extraction": {
            "requested_chunks": 1, "unique_chunks": 1,
            "cache_hits": 0, "cache_misses": 1, "model_calls": 1,
            "max_extraction_calls": payload["max_extraction_calls"],
        },
        "shadow_features": {
            "schema": "event-native-shadow-features-v1",
            "prefill": {"status": "captured", "layer": 1,
                        "position": {"kind": "prompt_last", "logical_position": 5},
                        "readout": "decoder_layer_output", "hidden": [hidden]},
        },
    }


def _fixture(directory, decisions):
    task = {"schema": "c2kv.controlled_task_fixture.v1", "task_id": "task-1",
            "session_id": "bfcl/task-1/attempt-0", "decisions": decisions,
            "source": "ignored provenance"}
    data = json.dumps(task, ensure_ascii=False).encode("utf-8")
    (directory / "task-1.json").write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (directory / "manifest.json").write_text(json.dumps({
        "schema": "c2kv.controlled_workload_fixture.v1",
        "tasks": {"task-1": {"file": "task-1.json", "sha256": digest}},
    }), encoding="utf-8")
    return digest


def _runner_payload(step):
    return {"session_id": "bfcl/task-1/attempt-0",
            "decision_key": f"turn-0/step-{step}",
            "outer_request_id": f"source-outer-{step}",
            "messages": [{"role": "user", "content": f"question {step}"}],
            "tools": []}


def _generation(request, memory, *, phase="draft", source_output=(51, 52)):
    response = _response(request, memory, output=source_output, hidden=0.7)
    response["text"] = "recorded"
    response["finish_reason"] = "stop"
    return {"request": request, "response": response, "phase": phase,
            "attempt_uid": request["rid"],
            "prepared_input": memory_to_dict(memory)}


def _source_request(monkeypatch, memory, step, ordinal):
    monkeypatch.delenv("C2KV_NATIVE_CONTROLLED_WORKLOAD_DIR", raising=False)
    generator = _generator(monkeypatch)
    captured = []
    generator._post_native_generate = lambda payload, *_args, **_kwargs: (
        captured.append(copy.deepcopy(payload)) or _response(payload, memory), 200)
    with generator.decision_scope(session_id="bfcl/task-1/attempt-0"):
        for index in range(ordinal):
            generator.generate(memory, ratio=8, max_new_tokens=4,
                               trace_context={
                                   "attempt_uid": f"source-{step}-{index}",
                                   "session_id": "bfcl/task-1/attempt-0",
                                   "decision_key": f"turn-0/step-{step}",
                                   "outer_request_id": f"source-outer-{step}",
                                   "phase": "draft" if index == 0 else "regeneration",
                               })
    return captured


def _controlled_generate(generator, memory, *, step, ordinal):
    return generator.generate(memory, ratio=8, max_new_tokens=4,
                              trace_context={
                                  "attempt_uid": f"actual-{step}-{ordinal}",
                                  "session_id": "bfcl/task-1/attempt-0",
                                  "decision_key": f"turn-0/step-{step}",
                                  "outer_request_id": f"actual-outer-{step}",
                                  "phase": "draft" if ordinal == 0 else "regeneration",
                              })


def test_default_off_preserves_native_sampling(monkeypatch):
    memory = _memory()
    source = _source_request(monkeypatch, memory, 0, 1)
    assert source[0]["sampling_params"] == {
        "temperature": 0.0, "sampling_seed": 0,
        "max_new_tokens": 4, "stop_token_ids": [0]}
    assert _generator(monkeypatch).controlled_workload is None


def test_controlled_two_generations_and_next_decision_keep_real_stats(tmp_path, monkeypatch):
    memory = _memory()
    first = _source_request(monkeypatch, memory, 0, 2)
    second = _source_request(monkeypatch, memory, 1, 1)
    decisions = [
        {"runner_payload": _runner_payload(0), "response": {"content": "first"},
         "generations": [_generation(first[0], memory),
                         _generation(first[1], memory, phase="regeneration")]},
        {"runner_payload": _runner_payload(1), "response": {"content": "second"},
         "generations": [_generation(second[0], memory)]},
    ]
    digest = _fixture(tmp_path, decisions)
    monkeypatch.setenv("C2KV_NATIVE_CONTROLLED_WORKLOAD_DIR", str(tmp_path))
    generator = _generator(monkeypatch)
    posted = []
    generator._post_native_generate = lambda payload, *_args, **_kwargs: (
        posted.append(copy.deepcopy(payload)) or _response(payload, memory), 200)
    for step, count in ((0, 2), (1, 1)):
        actual = _runner_payload(step)
        actual["outer_request_id"] = f"actual-outer-{step}"
        generator.controlled_workload.begin_decision("task-1", actual)
        with generator.decision_scope(session_id="bfcl/task-1/attempt-0"):
            for ordinal in range(count):
                result = _controlled_generate(generator, memory, step=step, ordinal=ordinal)
                assert result.token_ids == (51, 52)
                assert result.token_logprobs == (-0.2, -0.2)
                assert result.finish_reason == "stop"
                assert result.stats["shadow_features"]["prefill"]["hidden"] == [0.7]
                assert result.stats["sglang_extraction"]["model_calls"] == 1
                assert result.stats["sglang_chunk_handles"]["selected"]
                assert result.stats["controlled_workload"]["fixture_sha256"] == digest
                assert result.stats["controlled_workload"]["actual_output_tokens"] == 2
        generator.controlled_workload.complete_decision({
            "generation_trace": [{}] * count, "response": decisions[step]["response"]})
    assert len(posted) == 3
    assert all(row["sampling_params"]["max_new_tokens"] == 2
               and row["sampling_params"]["min_new_tokens"] == 2
               and row["sampling_params"]["ignore_eos"] is True
               and row["sampling_params"]["stop_token_ids"] == []
               for row in posted)
    assert generator.extraction_calls_reserved == 3


def test_semantic_mismatch_rejects_before_native_post(tmp_path, monkeypatch):
    memory = _memory()
    source = _source_request(monkeypatch, memory, 0, 1)[0]
    _fixture(tmp_path, [{"runner_payload": _runner_payload(0),
                         "response": {"content": "first"},
                         "generations": [_generation(source, memory)]}])
    monkeypatch.setenv("C2KV_NATIVE_CONTROLLED_WORKLOAD_DIR", str(tmp_path))
    generator = _generator(monkeypatch)
    posted = []
    generator._post_native_generate = lambda payload, *_args, **_kwargs: posted.append(payload)
    actual = _runner_payload(0)
    actual["outer_request_id"] = "actual-outer-0"
    generator.controlled_workload.begin_decision("task-1", actual)
    different = PackedMemory(memory.view, (17, 18), memory.workspace_input_ids,
                             memory.raw_source_indices, memory.chunks)
    with pytest.raises(ControlledWorkloadError, match="prepared input mismatch"):
        with generator.decision_scope(session_id="bfcl/task-1/attempt-0"):
            _controlled_generate(generator, different, step=0, ordinal=0)
    assert posted == []


def test_actual_decode_length_mismatch_is_terminal(tmp_path, monkeypatch):
    memory = _memory()
    source = _source_request(monkeypatch, memory, 0, 1)[0]
    _fixture(tmp_path, [{"runner_payload": _runner_payload(0),
                         "response": {"content": "first"},
                         "generations": [_generation(source, memory)]}])
    monkeypatch.setenv("C2KV_NATIVE_CONTROLLED_WORKLOAD_DIR", str(tmp_path))
    generator = _generator(monkeypatch)
    generator._post_native_generate = lambda payload, *_args, **_kwargs: (
        _response(payload, memory, output=(91,)), 200)
    actual = _runner_payload(0)
    actual["outer_request_id"] = "actual-outer-0"
    generator.controlled_workload.begin_decision("task-1", actual)
    with pytest.raises(RuntimeError, match="actual decode length mismatch"):
        with generator.decision_scope(session_id="bfcl/task-1/attempt-0"):
            _controlled_generate(generator, memory, step=0, ordinal=0)


def _api_request(message="question"):
    return {
        "model": "actor", "temperature": 0, "store": False,
        "max_completion_tokens": 4,
        "c2kv_eval_context": {"benchmark": "bfcl", "task_id": "task-1",
                              "user_turn": 0, "step": 0, "attempt": 0},
        "messages": [{"role": "user", "content": message}], "tools": [],
    }


def _api(tmp_path, runner, *, run_id="actual-run"):
    return EventNativeAPI(
        runner, run_id=run_id, model_name="actor", view_mode="static",
        max_new_tokens=4, allowed_task_ids=["task-1"], max_decisions=4,
        deadline_monotonic=time.monotonic() + 60,
        steps_path=tmp_path / "steps.jsonl")


@pytest.mark.parametrize("mismatch", ["input", "generation_count"])
def test_api_controlled_mismatch_is_terminal(tmp_path, monkeypatch, mismatch):
    monkeypatch.delenv("C2KV_NATIVE_CONTROLLED_WORKLOAD_DIR", raising=False)
    source_payload = _api(tmp_path, SimpleNamespace(run=lambda _: None),
                          run_id="source-run")._validate_request(_api_request())[0]
    generation = {
        "request": {"rid": "source", "sampling_params": {"max_new_tokens": 4}},
        "response": {"output_ids": [42], "token_logprobs": [-0.1],
                     "finish_reason": "stop", "shadow_features": None},
        "phase": "draft", "attempt_uid": "source", "prepared_input": {},
    }
    _fixture(tmp_path, [{"runner_payload": source_payload,
                         "response": {"content": "fixture"},
                         "generations": [generation] if mismatch == "generation_count" else []}])
    monkeypatch.setenv("C2KV_NATIVE_CONTROLLED_WORKLOAD_DIR", str(tmp_path))
    controlled = ControlledWorkload(tmp_path)
    calls = []
    runner = SimpleNamespace(
        generator=SimpleNamespace(controlled_workload=controlled),
        run=lambda payload: (calls.append(payload) or {
            "response": {"content": "fixture"}, "generation_trace": []}))
    api = _api(tmp_path, runner)
    request = _api_request("different" if mismatch == "input" else "question")
    with pytest.raises(EventNativeAPIError, match="generation failed terminally"):
        api.handle_chat(request)
    assert len(calls) == (0 if mismatch == "input" else 1)
    assert api.health()["terminal"] is True
