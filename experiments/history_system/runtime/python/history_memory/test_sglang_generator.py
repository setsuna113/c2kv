"""CPU-only contracts for the SGLang event-native generator adapter."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from benchmarks.memory_runtime.event_native_costs import summarize_event_native_steps
from history_memory.packing import EncoderChunk, MemoryView, PackedMemory
from history_memory.sglang_generator import (
    HANDLE_SCHEMA,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    EXTRACTION_BUDGET_ERROR_CODE,
    EXTRACTION_FAILURE_SCHEMA,
    SGLangEventNativeError,
    SGLangExtractionBudgetExhausted,
    SGLangEventNativeGenerator,
)
MODEL_PATH = "/models/checkpoint-1000"
MODEL_BINDING = {
    "model_path": MODEL_PATH,
    "weight_version": "v1",
    "dtype": "bfloat16",
    "gist": {"type": "dynamic-interleave", "param": "qkv"},
}
KV_BYTES = 16


def _load_engine_contract():
    contract_path = (
        Path(__file__).resolve().parents[6]
        / "sglang-c2kv"
        / "python"
        / "sglang"
        / "srt"
        / "mem_cache"
        / "c2kv_native_packed.py"
    )
    module_name = "_c2kv_native_packed_contract_for_adapter_test"
    spec = importlib.util.spec_from_file_location(module_name, contract_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _Response:
    status = 200

    def __init__(self, payload):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self, _limit=None):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Opener:
    def __init__(self, *, hit_calls=()):
        self.requests = []
        self.hit_calls = set(hit_calls)
        self.generate_calls = 0

    def open(self, request, *, timeout):
        method = request.get_method()
        payload = json.loads(request.data.decode("utf-8")) if request.data else None
        self.requests.append(
            {"url": request.full_url, "method": method, "timeout": timeout, "payload": payload}
        )
        if method == "GET":
            return _Response(
                {
                    "model_path": MODEL_PATH,
                    "c2kv_native_packed": {
                        "model_binding": MODEL_BINDING,
                        "kv_bytes_per_token": KV_BYTES,
                    }
                }
            )
        self.generate_calls += 1
        return _Response(
            _native_response(payload, cache_hit=self.generate_calls in self.hit_calls)
        )


class _FailingOpener(_Opener):
    def open(self, request, *, timeout):
        if request.get_method() == "GET":
            return super().open(request, timeout=timeout)
        self.requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "timeout": timeout,
                "payload": json.loads(request.data.decode("utf-8")),
            }
        )
        raise OSError("scripted transport failure")


class _BudgetFailureOpener(_Opener):
    def __init__(self, *, mutate=None):
        super().__init__()
        self.mutate = mutate

    def open(self, request, *, timeout):
        if request.get_method() == "GET":
            return super().open(request, timeout=timeout)
        payload = json.loads(request.data.decode("utf-8"))
        self.requests.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "timeout": timeout,
                "payload": payload,
            }
        )
        rows = payload["encoder_chunks"] + payload["compression_chunks"]
        receipt = {
            "error": {
                "code": EXTRACTION_BUDGET_ERROR_CODE,
                "message": EXTRACTION_BUDGET_ERROR_CODE + ": denied miss",
            },
            "extraction": {
                "schema": EXTRACTION_FAILURE_SCHEMA,
                "requested_chunks": len(rows),
                "unique_chunks": len({row["handle"] for row in rows}),
                "processed_chunks": 2,
                "cache_hits": 0,
                "cache_misses": 2,
                "model_calls": 2,
                "max_extraction_calls": payload["max_extraction_calls"],
                "failed_chunk_index": 2,
            },
        }
        if self.mutate is not None:
            self.mutate(receipt)
        body = io.BytesIO(json.dumps(receipt).encode("utf-8"))
        raise HTTPError(request.full_url, 400, "Bad Request", {}, body)


class _ExactOpener(_Opener):
    def __init__(self):
        super().__init__()
        self.exact_calls = []
        self.saved = {}

    def open(self, request, *, timeout):
        if not request.full_url.endswith("/exact_state"):
            return super().open(request, timeout=timeout)
        payload = json.loads(request.data)
        operation = payload["operation"]
        self.exact_calls.append(operation)
        snapshot_id = payload.get("snapshot_id")
        if operation == "capture":
            snapshot_id = f"s{len(self.saved)}"
            self.saved[snapshot_id] = {name: str(index) * 64 for index, name in enumerate((
                "actor_kv", "actor_positions", "backend_stats", "rng"))}
        if operation == "release":
            del self.saved[snapshot_id]
            return _Response({"schema": "c2kv-exact-backend-state-v1", "operation": operation,
                "snapshot_id": snapshot_id, "released": True})
        return _Response({"schema": "c2kv-exact-backend-state-v1", "operation": operation,
            "snapshot_id": snapshot_id, "exact": True, "verified_from_live_state": operation == "restore",
            "component_digests": self.saved[snapshot_id]})


def test_exact_restore_preserves_new_active_context_and_does_not_replay():
    opener = _ExactOpener()
    generator = _generator(opener)
    with generator.decision_scope(session_id="task-a"):
        generator.generate(_memory(), ratio=2, max_new_tokens=8)
        snapshot = generator.capture_exact_state()
        baseline = generator._exact_local_state()
        original_scope = generator._active_decision_scope
    with generator.decision_scope(session_id="task-b"):
        generator.generate(_memory(), ratio=2, max_new_tokens=8)
    with generator.decision_scope(session_id="task-a"):
        new_scope = generator._active_decision_scope
        assert new_scope is not original_scope
        calls_before = opener.generate_calls
        restored = generator.restore_exact_state(snapshot)
        assert generator._active_decision_scope is new_scope
        assert generator._exact_local_state() == baseline
        assert opener.generate_calls == calls_before
        assert restored["verified_from_live_state"]
        assert restored["component_digests"] == snapshot["component_digests"]
        generator.release_exact_state(snapshot)
    assert opener.exact_calls == ["capture", "restore", "release"]


def test_exact_restore_requires_matching_scope_and_real_backend_verification():
    opener = _ExactOpener()
    generator = _generator(opener)
    with generator.decision_scope(session_id="task-a"):
        generator.generate(_memory(), ratio=2, max_new_tokens=8)
        snapshot = generator.capture_exact_state()
    with pytest.raises(SGLangEventNativeError, match="decision-scope boundary"):
        generator.restore_exact_state(snapshot)
    with generator.decision_scope(session_id="task-a"):
        opener.saved[snapshot["snapshot_id"]]["actor_kv"] = "f" * 64
        with pytest.raises(SGLangEventNativeError, match="live exact-state verification"):
            generator.restore_exact_state(snapshot)


def _memory():
    chunks = (
        EncoderChunk("s:m0", 0, (0,), 0, 3, (10, 11, 12)),
        EncoderChunk("s:m1", 0, (1,), 0, 2, (20, 21)),
    )
    return PackedMemory(
        view=MemoryView(("s:m0", "s:m1"), ("s:m2",)),
        system_input_ids=(1, 2),
        workspace_input_ids=(30, 31),
        raw_source_indices=(2,),
        chunks=chunks,
    )


def _extra():
    return EncoderChunk("s:m-extra", 0, (3,), 0, 4, (40, 41, 42, 43))


def _native_response(request, *, cache_hit):
    encoder = request["encoder_chunks"]
    compression = request["compression_chunks"]

    def row(item):
        original = len(item["token_ids"])
        return {
            "chunk_id": item["chunk_id"],
            "handle": item["handle"],
            "cache_key": "server:" + item["handle"],
            "cache_hit": cache_hit,
            "gist_len": (original + request["compression_ratio"] - 1)
            // request["compression_ratio"],
            "original_seq_len": original,
        }

    unique = {item["handle"] for item in encoder + compression}
    presented = sum(len(item["token_ids"]) for item in encoder)
    gist = sum(
        (len(item["token_ids"]) + request["compression_ratio"] - 1)
        // request["compression_ratio"]
        for item in encoder
    )
    system = len(request["system_input_ids"])
    raw = len(request["workspace_input_ids"])
    misses = 0 if cache_hit else len(unique)
    logical_last = system + presented + raw - 1
    return {
        "schema": RESPONSE_SCHEMA,
        "rid": request["rid"],
        "session_id": request["session_id"],
        "generation_id": request["generation_id"],
        "output_ids": [71, 72],
        "text": "tool output",
        "token_logprobs": [-0.25, -0.5],
        "finish_reason": "stop",
        "encoder_chunks": [row(item) for item in encoder],
        "compression_chunks": [row(item) for item in compression],
        "extraction": {
            "requested_chunks": len(encoder) + len(compression),
            "unique_chunks": len(unique),
            "cache_hits": len(unique) - misses,
            "cache_misses": misses,
            "model_calls": misses,
            "max_extraction_calls": request["max_extraction_calls"],
        },
        "costs": {
            "system_tokens": system,
            "raw_tokens": raw,
            "presented_encoder_tokens": presented,
            "gist_tokens": gist,
            "resident_kv_tokens": system + raw + gist,
            "system_prefix_kv_logical_bytes": system * KV_BYTES,
            "gist_prefix_kv_logical_bytes": gist * KV_BYTES,
            "raw_workspace_kv_logical_bytes": raw * KV_BYTES,
            "resident_kv_logical_bytes": (system + raw + gist) * KV_BYTES,
        },
        "shadow_features": {
            "schema": "event-native-shadow-features-v1",
            "status": "not_present",
            "bindings": {"source": "sglang_native_packed"},
            "signals": {},
            "tool_name": {"status": "not_present", "reason": "no_tool_call"},
            "prefill": {
                "status": "captured",
                "reason": None,
                "layer": 34,
                "position": {"kind": "prompt_last", "logical_position": logical_last},
                "readout": "decoder_layer_output",
                "stored_dtype": "float16",
                "hidden": [1.0, -2.0],
            },
            "memgen": {"status": "disabled", "layer": None, "hidden": None},
            "capture_errors": [],
        },
        "allocator": {"active_c2kv_gist_tokens": gist},
    }


def _generator(opener, **overrides):
    arguments = dict(
        upstream="http://127.0.0.1:36000",
        expected_model_path=MODEL_PATH,
        model_context=128,
        max_new_tokens=8,
        max_generation_calls=4,
        max_extraction_calls=20,
        timeout_seconds=11,
        eos_token_ids=(151645,),
        eos_source="checkpoint_tokenizer.eos_token_id",
        encoding_scope="record",
        sampling_params={"temperature": 0.0, "seed": 0},
        shadow_feature_config=SimpleNamespace(
            enabled=True,
            decode_token_ids=lambda ids: str(list(ids)),
            prefill_layer=-2,
            memgen_layer=None,
        ),
        opener=opener,
    )
    arguments.update(overrides)
    return SGLangEventNativeGenerator(**arguments)


def test_posts_exact_packed_layout_and_content_addressed_handles():
    memory = _memory()
    extra = _extra()
    opener = _Opener()
    generator = _generator(opener)
    assert generator.decode_strategy == "incremental"

    with generator.decision_scope(session_id="task-1"):
        result = generator.generate(
            memory,
            ratio=2,
            max_new_tokens=8,
            compression_chunks=(*memory.chunks, extra),
            trace_context={
                "attempt_uid": "attempt-1",
                "session_id": "task-1",
                "decision_key": "d1",
                "phase": "draft",
            },
        )

    assert [request["method"] for request in opener.requests] == ["GET", "POST"]
    assert opener.requests[0]["url"].endswith("/model_info")
    posted = opener.requests[1]["payload"]
    assert opener.requests[1]["url"].endswith("/v1/c2kv/native_generate")
    assert posted["schema"] == REQUEST_SCHEMA
    assert posted["sampling_params"]["sampling_seed"] == 0
    assert "seed" not in posted["sampling_params"]
    assert posted["system_input_ids"] == [1, 2]
    assert posted["workspace_input_ids"] == [30, 31]
    assert posted["packing_version"] == "history-event-v1"
    assert posted["raw_layout_profile"] == "event-native-evidence-v1"
    assert posted["encoding_scope"] == "record"
    assert posted["compression_ratio"] == 2
    assert posted["shadow_features"] == {"prefill_layer": -2}
    assert len(posted["encoder_chunks"]) == 2
    assert len(posted["compression_chunks"]) == 1
    assert [item["source_position_start"] for item in posted["encoder_chunks"]] == [2, 5]
    assert [item["gist_position_ids"] for item in posted["encoder_chunks"]] == [[3, 4], [6]]

    first = posted["encoder_chunks"][0]
    engine_contract = _load_engine_contract()
    assert HANDLE_SCHEMA == engine_contract.NATIVE_CHUNK_HANDLE_SCHEMA
    assert first["handle"] == engine_contract.canonical_chunk_handle(
        first,
        model_binding=MODEL_BINDING,
        packing_version=posted["packing_version"],
        encoding_scope=posted["encoding_scope"],
        compression_ratio=posted["compression_ratio"],
    )
    plan = engine_contract.plan_native_packed_request(
        system_input_ids=posted["system_input_ids"],
        workspace_input_ids=posted["workspace_input_ids"],
        encoder_chunks=posted["encoder_chunks"],
        compression_chunks=posted["compression_chunks"],
        model_binding=MODEL_BINDING,
        packing_version=posted["packing_version"],
        raw_layout_profile=posted["raw_layout_profile"],
        encoding_scope=posted["encoding_scope"],
        compression_ratio=posted["compression_ratio"],
    )
    assert plan.costs["raw_workspace_kv_tokens"] == len(posted["workspace_input_ids"])
    assert plan.costs["resident_kv_tokens"] == 7
    assert result.token_ids == (71, 72)
    assert result.token_logprobs == (-0.25, -0.5)
    assert result.stats["materialized_encoder_tokens"] == 9
    assert result.stats["resident_prefix_kv_bytes"] == 5 * KV_BYTES
    assert result.stats["server_kv_memory_report"] == {
        "active_c2kv_gist_tokens": 3
    }
    assert result.stats["torch_allocator_peak_allocated_bytes"] is None
    assert result.stats["target_input_tokens"] is None
    assert result.stats["shadow_features"]["prefill"]["position"] == {
        "kind": "prompt_last",
        "logical_position": 8,
    }
    assert generator.session_cache_info()["chunk_handle_count"] == 2


def test_draft_regeneration_and_next_decision_reuse_stable_handles():
    memory = _memory()
    opener = _Opener(hit_calls=(2, 3))
    generator = _generator(opener)

    with generator.decision_scope(session_id="task-1"):
        draft = generator.generate(memory, ratio=2, max_new_tokens=8)
        regenerated = generator.generate(memory, ratio=2, max_new_tokens=8)
        assert draft.stats["session_cache_commit_status"] == "discarded_by_regeneration"
        assert regenerated.stats["scope_reused_chunks"] == 2
        assert regenerated.stats["scope_reused_encoder_tokens"] == 5
    assert regenerated.stats["session_cache_commit_status"] == "committed"

    with generator.decision_scope(session_id="task-1"):
        following = generator.generate(memory, ratio=2, max_new_tokens=8)
        assert following.stats["session_reused_gist_chunks"] == 2
        assert following.stats["session_reused_encoder_tokens"] == 5
    assert generator.session_cache_info()["generation"] == 2
    generator.close_session()
    assert generator.session_cache_info()["session_id"] is None
    assert len([item for item in opener.requests if item["method"] == "GET"]) == 1


def test_response_contract_failure_is_not_retried_or_fallback_generated():
    opener = _FailingOpener()
    generator = _generator(opener, max_generation_calls=1)
    with pytest.raises(SGLangEventNativeError, match="without retry"):
        generator.generate(_memory(), ratio=2, max_new_tokens=8)
    assert [item["method"] for item in opener.requests] == ["GET", "POST"]
    with pytest.raises(RuntimeError, match="cap exhausted"):
        generator.generate(_memory(), ratio=2, max_new_tokens=8)
    assert [item["method"] for item in opener.requests] == ["GET", "POST"]


def test_verified_extraction_budget_failure_is_charged_once_and_typed():
    opener = _BudgetFailureOpener()
    generator = _generator(opener, max_extraction_calls=2)

    with pytest.raises(SGLangExtractionBudgetExhausted) as captured:
        generator.generate(
            _memory(),
            ratio=2,
            max_new_tokens=8,
            compression_chunks=(*_memory().chunks, _extra()),
        )

    assert generator.extraction_calls_reserved == 2
    assert captured.value.receipt["model_calls"] == 2
    assert captured.value.receipt["processed_chunks"] == 2
    assert [item["method"] for item in opener.requests] == ["GET", "POST"]


def test_malformed_extraction_budget_failure_is_not_charged_or_typed():
    opener = _BudgetFailureOpener(
        mutate=lambda receipt: receipt["extraction"].__setitem__(
            "model_calls", 1
        )
    )
    generator = _generator(opener, max_extraction_calls=2)

    with pytest.raises(SGLangEventNativeError) as captured:
        generator.generate(
            _memory(),
            ratio=2,
            max_new_tokens=8,
            compression_chunks=(*_memory().chunks, _extra()),
        )

    assert type(captured.value) is SGLangEventNativeError
    assert generator.extraction_calls_reserved == 0
    assert [item["method"] for item in opener.requests] == ["GET", "POST"]


def test_rejects_server_layout_or_prompt_last_drift():
    class BadOpener(_Opener):
        def open(self, request, *, timeout):
            response = super().open(request, timeout=timeout)
            if request.get_method() == "POST":
                body = json.loads(response._body.decode("utf-8"))
                body["costs"]["gist_tokens"] += 1
                body["shadow_features"]["prefill"]["position"]["logical_position"] += 1
                return _Response(body)
            return response

    with pytest.raises(SGLangEventNativeError, match="costs.gist_tokens mismatch"):
        _generator(BadOpener()).generate(_memory(), ratio=2, max_new_tokens=8)

    class BadFeatureOpener(_Opener):
        def open(self, request, *, timeout):
            response = super().open(request, timeout=timeout)
            if request.get_method() == "POST":
                body = json.loads(response._body.decode("utf-8"))
                body["shadow_features"]["prefill"]["position"]["logical_position"] += 1
                return _Response(body)
            return response

    with pytest.raises(SGLangEventNativeError, match="prompt_last position mismatch"):
        _generator(BadFeatureOpener()).generate(_memory(), ratio=2, max_new_tokens=8)


def test_requires_eligible_compression_set_to_cover_selected_chunks_before_post():
    opener = _Opener()
    generator = _generator(opener)
    with pytest.raises(ValueError, match="include every retained gist chunk"):
        generator.generate(
            _memory(), ratio=2, max_new_tokens=8, compression_chunks=(_extra(),)
        )
    assert [item["method"] for item in opener.requests] == ["GET"]


def test_stats_are_accepted_by_current_cost_summarizer_with_unknown_remote_work():
    opener = _Opener()
    generator = _generator(opener)
    with generator.decision_scope(session_id="task-1"):
        result = generator.generate(_memory(), ratio=2, max_new_tokens=8)
    record = {
        "schema": "a-event-native-exact-step-v1",
        "status": "ok",
        "session_id": "task-1",
        "decision_key": "d1",
        "generation_trace": [
            {
                "phase": "draft",
                "status": "completed",
                "discarded": False,
                "attempt_uid": "attempt-1",
                "attempt_index": 1,
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "total_tokens": 9,
                },
                "generation": {"stats": result.stats},
            }
        ],
        "session_cache_after": generator.session_cache_info(),
    }
    summary = summarize_event_native_steps([record])
    work = summary["sessions"][0]["costs"]["actual_model_work"]
    assert work["materialized_encoder_tokens"]["strict_total"] == 5
    assert work["target_input_tokens"]["strict_total"] is None
    assert summary["cache_provenance"]["trace_coverage"]["unknown_attempts"] == 1


def test_model_binding_and_encoding_scope_change_handles():
    memory = _memory()
    opener_a = _Opener()
    opener_b = _Opener()
    first = _generator(opener_a, encoding_scope="event")
    second = _generator(opener_b, encoding_scope="record")
    first.generate(memory, ratio=2, max_new_tokens=8)
    second.generate(memory, ratio=2, max_new_tokens=8)
    handle_a = opener_a.requests[1]["payload"]["encoder_chunks"][0]["handle"]
    handle_b = opener_b.requests[1]["payload"]["encoder_chunks"][0]["handle"]
    assert handle_a != handle_b


def test_rejects_non_packed_or_mutated_server_counts():
    with pytest.raises(ValueError, match="max_extraction_calls must be a positive integer"):
        _generator(_Opener(), max_extraction_calls=None)

    generator = _generator(_Opener())
    with pytest.raises(TypeError, match="PackedMemory"):
        generator.generate(object(), ratio=2, max_new_tokens=8)

    class CountOpener(_Opener):
        def open(self, request, *, timeout):
            response = super().open(request, timeout=timeout)
            if request.get_method() == "POST":
                body = json.loads(response._body.decode("utf-8"))
                body["encoder_chunks"] = body["encoder_chunks"][:-1]
                return _Response(body)
            return response

    with pytest.raises(SGLangEventNativeError, match="encoder_chunks count mismatch"):
        _generator(CountOpener()).generate(_memory(), ratio=2, max_new_tokens=8)
