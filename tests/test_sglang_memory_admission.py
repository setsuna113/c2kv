from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest

from benchmarks.memory_runtime.sglang_memory_admission import (
    assess_sglang_memory,
    inspect_checkpoint_weights,
    summarize_checkpoint_headers,
)


def qwen_config(**overrides):
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "gist_param": "qkv",
        "dtype": "bfloat16",
        "tie_word_embeddings": True,
        "attention_bias": False,
        "use_sliding_window": False,
        "sliding_window": None,
        "vocab_size": 10,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "layer_types": ["full_attention"],
    }
    config.update(overrides)
    return config


def observed_header(config, *, extra=None):
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    q_width = config["num_attention_heads"] * config["head_dim"]
    kv_width = config["num_key_value_heads"] * config["head_dim"]
    shapes = {
        "model.embed_tokens.weight": (config["vocab_size"], hidden),
        "model.gist_embed_tokens.weight": (1, hidden),
        "model.norm.weight": (hidden,),
        "model.layers.0.input_layernorm.weight": (hidden,),
        "model.layers.0.post_attention_layernorm.weight": (hidden,),
        "model.layers.0.mlp.down_proj.weight": (hidden, intermediate),
        "model.layers.0.mlp.gate_proj.weight": (intermediate, hidden),
        "model.layers.0.mlp.up_proj.weight": (intermediate, hidden),
        "model.layers.0.self_attn.q_proj.weight": (q_width, hidden),
        "model.layers.0.self_attn.k_proj.weight": (kv_width, hidden),
        "model.layers.0.self_attn.v_proj.weight": (kv_width, hidden),
        "model.layers.0.self_attn.o_proj.weight": (hidden, q_width),
        "model.layers.0.self_attn.gist_q_proj.weight": (q_width, hidden),
        "model.layers.0.self_attn.gist_k_proj.weight": (kv_width, hidden),
        "model.layers.0.self_attn.gist_v_proj.weight": (kv_width, hidden),
        "model.layers.0.self_attn.q_norm.weight": (config["head_dim"],),
        "model.layers.0.self_attn.k_norm.weight": (config["head_dim"],),
    }
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, shape in shapes.items():
        size = 2
        for dimension in shape:
            size *= dimension
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    if extra is not None:
        header[extra[0]] = extra[1]
        offset = extra[1]["data_offsets"][1]
    return header, offset


def saved_headers(config, **kwargs):
    header, payload_bytes = observed_header(config, **kwargs)
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return {
        "model.safetensors": {
            "file_bytes": 8 + len(raw_header) + payload_bytes,
            "header_length": len(raw_header),
            "header_sha256": hashlib.sha256(raw_header).hexdigest(),
            "header": header,
        }
    }


def admission(**overrides):
    values = {
        "free_memory_bytes": 10_000,
        "total_memory_bytes": 10_000,
        "weight_bytes_lower_bound": 100,
        "kv_bytes_per_token": 10,
        "mem_fraction_static": 0.2,
        "c2kv_pool_fraction": 0.01,
        "raw_token_cap": 16_384,
        "required_raw_tokens": 100,
        "required_gist_tokens": 2,
        "page_size": 128,
    }
    values.update(overrides)
    return assess_sglang_memory(**values)


def test_summarize_qwen3_qkv_bf16_headers_counts_only_expected_storage():
    config = qwen_config()
    summary = summarize_checkpoint_headers(config, saved_headers(config))
    assert summary["weight_bytes_lower_bound"] == 1_648
    assert summary["kv_bytes_per_token"] == 16
    assert summary["supported_tensor_parallel_size"] == 1
    assert summary["ignored_tied_tensor_names"] == []


def test_summarize_accepts_qwen3_query_width_distinct_from_hidden_size():
    config = qwen_config(hidden_size=6)
    summary = summarize_checkpoint_headers(config, saved_headers(config))
    assert summary["kv_bytes_per_token"] == 16


def test_summarize_rejects_unsupported_config_and_unknown_tensor():
    config = qwen_config(dtype="float16")
    with pytest.raises(ValueError, match="dtype"):
        summarize_checkpoint_headers(config, saved_headers(config))

    config = qwen_config()
    header, payload = observed_header(config)
    header["model.unexpected.weight"] = {
        "dtype": "BF16", "shape": [1], "data_offsets": [payload, payload + 2]
    }
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    headers = {"model.safetensors": {
        "file_bytes": 8 + len(raw_header) + payload + 2,
        "header_length": len(raw_header),
        "header_sha256": hashlib.sha256(raw_header).hexdigest(),
        "header": header,
    }}
    with pytest.raises(ValueError, match="tensor set"):
        summarize_checkpoint_headers(config, headers)


def test_tied_lm_head_is_ignored_only_after_its_shape_is_verified():
    config = qwen_config()
    _, payload = observed_header(config)
    summary = summarize_checkpoint_headers(config, saved_headers(config, extra=(
        "lm_head.weight",
        {"dtype": "BF16", "shape": [10, 8], "data_offsets": [payload, payload + 160]},
    )))
    assert summary["weight_bytes_lower_bound"] == 1_648
    assert summary["ignored_tied_tensor_names"] == ["lm_head.weight"]

    with pytest.raises(ValueError, match="lm_head.weight shape"):
        summarize_checkpoint_headers(config, saved_headers(config, extra=(
            "lm_head.weight",
            {"dtype": "BF16", "shape": [9, 8], "data_offsets": [payload, payload + 144]},
        )))


def test_inspect_reads_config_and_header_but_no_tensor_payload(tmp_path):
    config = qwen_config()
    headers = saved_headers(config)
    header = headers["model.safetensors"]["header"]
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_bytes = headers["model.safetensors"]["file_bytes"] - 8 - len(raw_header)
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(raw_header)) + raw_header + b"x" * payload_bytes
    )

    profile = inspect_checkpoint_weights(tmp_path)
    assert profile["weight_bytes_lower_bound"] == 1_648
    assert profile["tensor_payload_bytes_read"] == 0
    assert profile["bytes_read"] == len((tmp_path / "config.json").read_bytes()) + 8 + len(raw_header)
    assert profile["header_identity"] == {
        "model.safetensors": {
            "file_bytes": 8 + len(raw_header) + payload_bytes,
            "header_length": len(raw_header),
            "header_sha256": hashlib.sha256(raw_header).hexdigest(),
        }
    }


def test_frozen_1088_preload_snapshot_rejects_raw_capacity_before_weights():
    mib = 1024 * 1024
    result = admission(
        free_memory_bytes=44_280 * mib,
        total_memory_bytes=65_536 * mib,
        weight_bytes_lower_bound=9_177_403_392,
        kv_bytes_per_token=147_456,
        mem_fraction_static=0.2,
        c2kv_pool_fraction=0.06,
        required_raw_tokens=3_899,
        required_gist_tokens=224,
    )
    assert result["status"] == "rejected"
    assert result["reasons"] == ["raw_pool_capacity_below_paged_request"]
    assert result["calculated_bounds"]["minimum_coexisting_bytes_lower_bound"] < 44_280 * mib


def test_rejects_fixed_gist_pool_even_when_raw_capacity_is_sufficient():
    result = admission(
        free_memory_bytes=1_000,
        total_memory_bytes=10_000,
        weight_bytes_lower_bound=100,
        kv_bytes_per_token=10,
        mem_fraction_static=0.9,
        c2kv_pool_fraction=0.5,
        raw_token_cap=100,
        required_raw_tokens=1,
        required_gist_tokens=1,
        page_size=1,
    )
    assert result["reasons"] == ["minimum_coexisting_allocation_exceeds_preload_free_memory"]


def test_gist_capacity_uses_total_device_memory_not_free_memory():
    result = admission(
        free_memory_bytes=500,
        total_memory_bytes=10_000,
        weight_bytes_lower_bound=100,
        kv_bytes_per_token=10,
        mem_fraction_static=0.9,
        c2kv_pool_fraction=0.1,
        raw_token_cap=10,
        required_raw_tokens=1,
        required_gist_tokens=30,
        page_size=1,
    )
    assert result["calculated_bounds"]["gist_pool_token_capacity"] == 37
    assert "gist_pool_capacity_below_request" not in result["reasons"]
    assert "minimum_coexisting_allocation_exceeds_preload_free_memory" in result["reasons"]


def test_paged_raw_minimum_rejects_an_unaligned_cap():
    result = admission(
        raw_token_cap=15,
        required_raw_tokens=9,
        page_size=8,
        c2kv_pool_fraction=0.001,
        required_gist_tokens=0,
    )
    assert result["calculated_bounds"]["required_raw_paged_token_capacity"] == 16
    assert result["calculated_bounds"]["raw_resolved_token_capacity_upper_bound"] == 8
    assert "raw_pool_capacity_below_paged_request" in result["reasons"]


def test_sufficient_profile_below_user_cap_allows_an_attempt_without_adjustment():
    result = admission()
    assert result["status"] == "necessary_conditions_passed"
    assert result["preflight_allows_attempt"] is True
    assert result["inputs"]["raw_token_cap"] == 16_384
    assert result["calculated_bounds"]["raw_resolved_token_capacity_upper_bound"] == 128
    assert result["calculated_bounds"]["raw_resolved_token_capacity_upper_bound"] < result["inputs"]["raw_token_cap"]


@pytest.mark.parametrize("field,value", [
    ("raw_token_cap", 0),
    ("raw_token_cap", True),
    ("mem_fraction_static", float("nan")),
])
def test_assessment_rejects_zero_bool_and_nonfinite_inputs(field, value):
    with pytest.raises(ValueError):
        admission(**{field: value})
