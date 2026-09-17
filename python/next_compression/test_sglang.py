"""CPU/mock-HTTP tests for next-compression's shared SGLang transport."""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from history_memory.packing import EncoderChunk, MemoryView, PackedMemory
from next_compression.common import TRAINING_PROFILE, sha256_file
from next_compression.sglang import (
    SGLangCheckpointBindingError,
    SGLangNextCompressionGenerator,
    _validate_server_info,
    load_sglang_checkpoint,
)
from next_compression.vendor.sglang_transport.sglang_generator import (
    RESPONSE_SCHEMA,
    SGLangEventNativeError,
)


KV_BYTES = 16


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir(parents=True)
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "vocab_size": 200000,
        "max_position_embeddings": 4096,
        "eos_token_id": 151645,
        "history_memory_training_profile": TRAINING_PROFILE,
        "history_memory_variant": "H0",
        "history_memory_compression_domain": "history",
        "history_memory_supported_ratios": [8, 12],
        "history_memory_normal_query": "base",
        "history_memory_corpus_identity": "a" * 64,
        "history_memory_render_profile": "event-native-evidence-v1",
        "history_memory_loss_profile": "decision-mean-complete-ce-v1",
        "history_memory_initialization_id": "history-fresh-v1",
        "history_memory_initialization_group": "history",
        "history_memory_initialization_kind": "fresh-base",
        "gist_type": "dynamic-interleave",
        "gist_param": "qkv",
        "gist_extra_embed_num": 1,
        "gist_residual_type": "embed-mean",
        "gist_overlap": 0,
    }
    (checkpoint / "config.json").write_text(
        json.dumps(config, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checkpoint


def _model_info(checkpoint: Path, dtype: str = "bfloat16") -> dict:
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    weight_version = "next-compression:" + sha256_file(checkpoint / "config.json")
    binding = {
        "model_path": str(checkpoint.resolve()),
        "tokenizer_path": str(checkpoint.resolve()),
        "weight_version": weight_version,
        "dtype": dtype,
        "kv_cache_dtype": dtype,
        "gist_parameter_dtype": "float32",
        "gist_compute_dtype": dtype,
        "gist_type": "dynamic-interleave",
        "gist_param": "qkv",
        "gist_extra_embed_num": 1,
        "gist_residual_type": "embed-mean",
        "gist_overlap": 0,
        "pic_enabled": False,
        "pic_param": "qkv",
        "query_projection": "base",
    }
    return {
        "model_path": str(checkpoint.resolve()),
        "weight_version": weight_version,
        "model_type": config["model_type"],
        "architectures": config["architectures"],
        "c2kv_native_packed": {
            "schema": "c2kv-native-packed-capability-v1",
            "enabled": True,
            "endpoint": "/v1/c2kv/native_generate",
            "packing_version": "history-event-v1",
            "raw_layout_profile": "event-native-evidence-v1",
            "model_binding": binding,
            "parameter_version": weight_version,
            "kv_bytes_per_token": KV_BYTES,
            "gist_parameter_dtype": "float32",
            "gist_compute_dtype": dtype,
            "base_query_enforced": True,
        },
    }


def _server_info(device: str = "npu", context_length: int = 2048) -> dict:
    return {
        "device": device,
        "context_length": context_length,
    }


@pytest.mark.parametrize(
    ("device", "backend", "disable_graph", "page_size"),
    (("cuda", "flashinfer", False, 1), ("cuda", "torch_native", True, 1),
     ("npu", "ascend", True, 128)),
)
def test_engine_receipt_records_actual_execution_profile(
    device, backend, disable_graph, page_size
):
    execution = {
        "attention_backend": backend,
        "disable_cuda_graph": disable_graph,
        "disable_piecewise_cuda_graph": True,
        "disable_overlap_schedule": device == "cuda",
        "disable_radix_cache": True,
        "page_size": page_size,
        "max_running_requests": 1,
    }
    receipt = _validate_server_info(
        {**_server_info(device), **execution}, device=device,
        checkpoint_context_length=4096,
    )
    assert receipt["execution"] == execution


def _install_fake_tokenizer(monkeypatch, eos_token_id=151645) -> None:
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, *, local_files_only):
            assert Path(path).is_dir()
            assert local_files_only is True
            return SimpleNamespace(eos_token_id=eos_token_id)

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoTokenizer=AutoTokenizer)
    )


def test_loader_binds_exact_model_config_and_multiple_checkpoint_eos(
    tmp_path, monkeypatch
):
    checkpoint = _checkpoint(tmp_path)
    (checkpoint / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [151645, 151643]}), encoding="utf-8"
    )
    info = _model_info(checkpoint)
    monkeypatch.setattr(
        "next_compression.sglang._read_model_info", lambda _upstream: info
    )
    monkeypatch.setattr(
        "next_compression.sglang._read_server_info",
        lambda _upstream: _server_info(),
    )
    _install_fake_tokenizer(monkeypatch, eos_token_id=199999)

    generator, tokenizer, profile = load_sglang_checkpoint(
        checkpoint,
        upstream="http://127.0.0.1:36000",
        device="npu",
        dtype="bfloat16",
        max_new_tokens=64,
        max_requests=7,
    )

    expected_weight_version = "next-compression:" + sha256_file(
        checkpoint / "config.json"
    )
    assert tokenizer.eos_token_id == 199999
    assert generator.eos_token_ids == (151643, 151645)
    assert generator.eos_source == "checkpoint_generation_config.eos_token_id"
    assert generator.model_context == 2048
    assert profile["checkpoint"] == str(checkpoint.resolve())
    assert profile["dtype"] == "bfloat16"
    assert profile["device"] == "npu"
    assert profile["gist_parameter_dtype"] == "float32"
    engine = profile["serving_engine"]
    assert engine["weight_version"] == expected_weight_version
    assert engine["model_binding"] == info["c2kv_native_packed"]["model_binding"]
    assert engine["base_query_enforced"] is True
    assert engine["eos_token_ids"] == [151645, 151643]
    assert engine["request_budget_owner"] == "LiveNextCompressionService"
    assert engine["frontend_max_requests"] == 7
    assert engine["requested_device"] == "npu"
    assert engine["device"] == "npu"
    assert engine["device_verification"] == "server_info.device"
    assert engine["checkpoint_context_length"] == 4096
    assert engine["engine_context_length"] == 2048
    assert engine["effective_model_context"] == 2048
    assert engine["transport_source"]["schema"].endswith("vendor-v1")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda info: info.__setitem__("weight_version", "wrong"),
            "config identity",
        ),
        (
            lambda info: info["c2kv_native_packed"].__setitem__(
                "base_query_enforced", False
            ),
            "base_query_enforced",
        ),
        (
            lambda info: info["c2kv_native_packed"]["model_binding"].__setitem__(
                "gist_parameter_dtype", "bfloat16"
            ),
            "gist_parameter_dtype",
        ),
        (
            lambda info: info["c2kv_native_packed"]["model_binding"].__setitem__(
                "dtype", "float16"
            ),
            "dtype",
        ),
        (
            lambda info: info["c2kv_native_packed"]["model_binding"].__setitem__(
                "tokenizer_path", "override-tokenizer"
            ),
            "tokenizer_path",
        ),
        (
            lambda info: info["c2kv_native_packed"]["model_binding"].__setitem__(
                "kv_cache_dtype", "fp8_e4m3"
            ),
            "kv_cache_dtype",
        ),
    ],
)
def test_loader_rejects_unverified_engine_binding(
    tmp_path, monkeypatch, mutation, message
):
    checkpoint = _checkpoint(tmp_path)
    info = _model_info(checkpoint)
    mutation(info)
    monkeypatch.setattr(
        "next_compression.sglang._read_model_info", lambda _upstream: info
    )
    monkeypatch.setattr(
        "next_compression.sglang._read_server_info",
        lambda _upstream: _server_info(),
    )
    _install_fake_tokenizer(monkeypatch)
    with pytest.raises(SGLangCheckpointBindingError, match=message):
        load_sglang_checkpoint(
            checkpoint,
            upstream="http://127.0.0.1:36000",
            device="npu",
            dtype="bfloat16",
            max_new_tokens=64,
            max_requests=1,
        )


@pytest.mark.parametrize(
    ("server_info", "message"),
    [
        (_server_info(device="cuda"), "device"),
        ({"device": "npu", "context_length": None}, "context_length"),
    ],
)
def test_loader_rejects_unverified_server_device_or_context(
    tmp_path, monkeypatch, server_info, message
):
    checkpoint = _checkpoint(tmp_path)
    monkeypatch.setattr(
        "next_compression.sglang._read_model_info",
        lambda _upstream: _model_info(checkpoint),
    )
    monkeypatch.setattr(
        "next_compression.sglang._read_server_info",
        lambda _upstream: server_info,
    )
    _install_fake_tokenizer(monkeypatch)
    with pytest.raises(SGLangCheckpointBindingError, match=message):
        load_sglang_checkpoint(
            checkpoint,
            upstream="http://127.0.0.1:36000",
            device="npu",
            dtype="bfloat16",
            max_new_tokens=64,
            max_requests=1,
        )


def _memory() -> PackedMemory:
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


def _native_response(request: dict, *, finish_reason: str = "length") -> dict:
    rows = request["encoder_chunks"]

    def row(item):
        original = len(item["token_ids"])
        return {
            "chunk_id": item["chunk_id"],
            "handle": item["handle"],
            "cache_key": "server:" + item["handle"],
            "cache_hit": False,
            "gist_len": (
                original + request["compression_ratio"] - 1
            )
            // request["compression_ratio"],
            "original_seq_len": original,
        }

    system = len(request["system_input_ids"])
    raw = len(request["workspace_input_ids"])
    presented = sum(len(item["token_ids"]) for item in rows)
    gist = sum(
        (len(item["token_ids"]) + request["compression_ratio"] - 1)
        // request["compression_ratio"]
        for item in rows
    )
    return {
        "schema": RESPONSE_SCHEMA,
        "rid": request["rid"],
        "session_id": request["session_id"],
        "generation_id": request["generation_id"],
        "output_ids": [71, 151645],
        "text": "<tool_call>payload</tool_call>",
        "token_logprobs": [-0.25, -0.5],
        "finish_reason": finish_reason,
        "encoder_chunks": [row(item) for item in rows],
        "compression_chunks": [],
        "extraction": {
            "requested_chunks": len(rows),
            "unique_chunks": len(rows),
            "cache_hits": 0,
            "cache_misses": len(rows),
            "model_calls": len(rows),
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
        "shadow_features": None,
        "allocator": {"active_c2kv_gist_tokens": gist},
    }


class _Response:
    status = 200

    def __init__(self, payload):
        self.body = json.dumps(payload).encode("utf-8")

    def read(self, _limit=None):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Opener:
    def __init__(self, requests, barrier=None, finish_reason="length"):
        self.requests = requests
        self.barrier = barrier
        self.finish_reason = finish_reason

    def open(self, request, *, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        self.requests.append((payload, timeout))
        return _Response(
            _native_response(payload, finish_reason=self.finish_reason)
        )


def _generator(tmp_path, opener_factory) -> SGLangNextCompressionGenerator:
    checkpoint = (tmp_path / "checkpoint").resolve()
    checkpoint.mkdir(exist_ok=True)
    binding = _model_info(_checkpoint(tmp_path / "binding"))[
        "c2kv_native_packed"
    ]["model_binding"]
    binding["model_path"] = str(checkpoint)
    binding["tokenizer_path"] = str(checkpoint)
    return SGLangNextCompressionGenerator(
        checkpoint_path=checkpoint,
        upstream="http://127.0.0.1:36000",
        model_context=128,
        max_new_tokens=8,
        eos_token_ids=(151645, 151643),
        eos_source="checkpoint_generation_config.eos_token_id",
        model_binding=binding,
        kv_bytes_per_token=KV_BYTES,
        journal_path=None,
        opener_factory=opener_factory,
    )


def test_generate_transmits_stops_and_preserves_tokens_eos_and_length(tmp_path):
    requests = []
    generator = _generator(tmp_path, lambda: _Opener(requests))
    with generator.decision_scope(session_id=None):
        result = generator.generate(
            _memory(),
            ratio=2,
            max_new_tokens=8,
            stop_strings=("</tool_call>", "DONE"),
            trace_context={"attempt_uid": "request-1"},
        )
    assert len(requests) == 1
    payload, timeout = requests[0]
    assert timeout == 600.0
    assert payload["sampling_params"]["stop"] == ["</tool_call>", "DONE"]
    assert payload["sampling_params"]["no_stop_trim"] is True
    assert payload["sampling_params"]["stop_token_ids"] == [151643, 151645]
    assert payload["max_extraction_calls"] == len(_memory().chunks)
    assert result.token_ids == (71, 151645)
    assert result.finish_reason == "length"
    assert result.stats["eos_token_ids"] == [151643, 151645]
    assert result.stats["max_extraction_calls"] == 2


def test_raw_only_request_uses_zero_extraction_budget(tmp_path):
    requests = []
    generator = _generator(tmp_path, lambda: _Opener(requests))
    memory = PackedMemory(
        view=MemoryView((), ()),
        system_input_ids=(1, 2),
        workspace_input_ids=(30, 31),
        raw_source_indices=(),
        chunks=(),
    )
    with generator.decision_scope(session_id=None):
        result = generator.generate(memory, ratio=8, max_new_tokens=8)
    assert requests[0][0]["max_extraction_calls"] == 0
    assert result.stats["max_extraction_calls"] == 0
    assert result.stats["extracted_chunks"] == 0


@pytest.mark.parametrize("finish_reason", ["abort", "content_filter", "unknown"])
def test_generate_rejects_abnormal_finish_reason(tmp_path, finish_reason):
    requests = []
    generator = _generator(
        tmp_path,
        lambda: _Opener(requests, finish_reason=finish_reason),
    )
    with pytest.raises(SGLangEventNativeError, match="must be stop or length"):
        generator.generate(_memory(), ratio=2, max_new_tokens=8)


def test_concurrent_decision_scopes_use_independent_native_adapters(tmp_path):
    requests = []
    barrier = threading.Barrier(2)
    generator = _generator(tmp_path, lambda: _Opener(requests, barrier))

    def run(index):
        with generator.decision_scope(session_id=None):
            result = generator.generate(
                _memory(),
                ratio=2,
                max_new_tokens=8,
                stop_strings=(f"stop-{index}",),
                trace_context={"attempt_uid": f"request-{index}"},
            )
            generator.close_session()
            return result.token_ids

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, (1, 2)))
    assert results == [(71, 151645), (71, 151645)]
    assert {item[0]["rid"] for item in requests} == {"request-1", "request-2"}
    assert {
        tuple(item[0]["sampling_params"]["stop"]) for item in requests
    } == {("stop-1",), ("stop-2",)}
    assert all(item[0]["max_extraction_calls"] == 2 for item in requests)
