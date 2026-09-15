"""CPU-only pre-weight-load admission checks for the audited SGLang setup."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
from typing import Any


MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
_BF16_BYTES = 2
_INT64_BYTES = 8


def _require_int(value: Any, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        qualifier = "positive" if minimum == 1 else f">= {minimum}"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _require_fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite fraction in (0, 1)")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0 < fraction < 1:
        raise ValueError(f"{name} must be a finite fraction in (0, 1)")
    return fraction


def _positive_config_int(config: Mapping[str, Any], name: str) -> int:
    return _require_int(config.get(name), f"config.{name}")


def _qwen3_qkv_geometry(config: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    if config.get("model_type") != "qwen3":
        raise ValueError("only model_type='qwen3' is supported")
    if config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise ValueError("only architectures=['Qwen3ForCausalLM'] is supported")
    if config.get("gist_param") != "qkv":
        raise ValueError("only gist_param='qkv' is supported")
    if config.get("dtype") != "bfloat16":
        raise ValueError("only dtype='bfloat16' is supported")
    if config.get("tie_word_embeddings") is not True:
        raise ValueError("only tied word embeddings are supported")
    if config.get("attention_bias") is not False:
        raise ValueError("only attention_bias=false is supported")
    if config.get("use_sliding_window") is not False or config.get("sliding_window") is not None:
        raise ValueError("only full-attention Qwen3 is supported")
    if config.get("quantization_config") is not None:
        raise ValueError("quantized checkpoints are unsupported")
    if config.get("pretraining_tp", 1) != 1:
        raise ValueError("only tensor-parallel size 1 checkpoint layout is supported")

    geometry = {
        name: _positive_config_int(config, name)
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        )
    }
    if geometry["num_attention_heads"] % geometry["num_key_value_heads"]:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list) or layer_types != ["full_attention"] * geometry["num_hidden_layers"]:
        raise ValueError("only all-full_attention Qwen3 layer_types are supported")
    return geometry


def _expected_tensor_shapes(geometry: Mapping[str, int]) -> dict[str, tuple[int, ...]]:
    hidden = geometry["hidden_size"]
    intermediate = geometry["intermediate_size"]
    head_dim = geometry["head_dim"]
    q_width = geometry["num_attention_heads"] * head_dim
    kv_width = geometry["num_key_value_heads"] * head_dim
    expected = {
        "model.embed_tokens.weight": (geometry["vocab_size"], hidden),
        "model.gist_embed_tokens.weight": (1, hidden),
        "model.norm.weight": (hidden,),
    }
    for layer in range(geometry["num_hidden_layers"]):
        prefix = f"model.layers.{layer}"
        expected.update(
            {
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.self_attn.q_proj.weight": (q_width, hidden),
                f"{prefix}.self_attn.k_proj.weight": (kv_width, hidden),
                f"{prefix}.self_attn.v_proj.weight": (kv_width, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, q_width),
                f"{prefix}.self_attn.gist_q_proj.weight": (q_width, hidden),
                f"{prefix}.self_attn.gist_k_proj.weight": (kv_width, hidden),
                f"{prefix}.self_attn.gist_v_proj.weight": (kv_width, hidden),
                f"{prefix}.self_attn.q_norm.weight": (head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (head_dim,),
            }
        )
    return expected


def _validate_header_identity(name: str, entry: Mapping[str, Any]) -> tuple[int, int, str, Mapping[str, Any]]:
    if not isinstance(entry, Mapping):
        raise ValueError(f"header identity for {name} must be a mapping")
    file_bytes = _require_int(entry.get("file_bytes"), f"headers.{name}.file_bytes")
    header_length = _require_int(entry.get("header_length"), f"headers.{name}.header_length", minimum=0)
    if header_length > MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"headers.{name}.header_length exceeds the 16 MiB limit")
    header_sha256 = entry.get("header_sha256")
    if not isinstance(header_sha256, str) or len(header_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in header_sha256.lower()
    ):
        raise ValueError(f"headers.{name}.header_sha256 must be a SHA-256 hex digest")
    header = entry.get("header")
    if not isinstance(header, Mapping):
        raise ValueError(f"headers.{name}.header must be a mapping")
    if file_bytes < 8 + header_length:
        raise ValueError(f"headers.{name} is shorter than its declared safetensors header")
    return file_bytes, header_length, header_sha256, header


def _validate_tensor(
    name: str,
    tensor: Any,
    expected_shape: tuple[int, ...],
    payload_bytes: int,
) -> tuple[int, int]:
    if not isinstance(tensor, Mapping):
        raise ValueError(f"tensor {name} must have a mapping header")
    if tensor.get("dtype") != "BF16":
        raise ValueError(f"tensor {name} must use unquantized BF16 storage")
    shape = tensor.get("shape")
    if not isinstance(shape, (list, tuple)) or any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        raise ValueError(f"tensor {name} has an invalid shape")
    if tuple(shape) != expected_shape:
        raise ValueError(f"tensor {name} shape {tuple(shape)!r} does not match {expected_shape!r}")
    offsets = tensor.get("data_offsets")
    if not isinstance(offsets, (list, tuple)) or len(offsets) != 2 or any(type(offset) is not int for offset in offsets):
        raise ValueError(f"tensor {name} has invalid data_offsets")
    start, end = offsets
    if not 0 <= start <= end <= payload_bytes:
        raise ValueError(f"tensor {name} data_offsets fall outside the safetensors payload")
    expected_bytes = math.prod(expected_shape) * _BF16_BYTES
    if end - start != expected_bytes:
        raise ValueError(f"tensor {name} byte range does not match its BF16 shape")
    return start, end


def summarize_checkpoint_headers(
    config: Mapping[str, Any], headers: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Validate saved safetensors headers without reading any tensor payload."""
    geometry = _qwen3_qkv_geometry(config)
    if not isinstance(headers, Mapping) or set(headers) != {"model.safetensors"}:
        raise ValueError("only one unsharded model.safetensors header is supported")
    file_bytes, header_length, header_sha256, header = _validate_header_identity(
        "model.safetensors", headers["model.safetensors"]
    )
    metadata = header.get("__metadata__")
    if not isinstance(metadata, Mapping) or metadata.get("format") != "pt":
        raise ValueError("model.safetensors must declare __metadata__.format='pt'")

    expected = _expected_tensor_shapes(geometry)
    tensor_names = set(header) - {"__metadata__"}
    allowed = set(expected) | {"lm_head.weight"}
    if not tensor_names.issubset(allowed) or not set(expected).issubset(tensor_names):
        raise ValueError("safetensors tensor set is unsupported for Qwen3 qkv BF16")
    if "lm_head.weight" in tensor_names and config.get("tie_word_embeddings") is not True:
        raise ValueError("lm_head.weight can only be ignored for tied word embeddings")

    payload_bytes = file_bytes - 8 - header_length
    ranges: list[tuple[int, int, str]] = []
    weight_bytes_lower_bound = 0
    for name in sorted(tensor_names):
        expected_shape = expected.get(name, expected["model.embed_tokens.weight"])
        start, end = _validate_tensor(name, header[name], expected_shape, payload_bytes)
        ranges.append((start, end, name))
        if name != "lm_head.weight":
            weight_bytes_lower_bound += end - start
    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise ValueError(f"safetensors payload is non-contiguous before tensor {name}")
        cursor = end
    if cursor != payload_bytes:
        raise ValueError("safetensors payload does not match its tensor offsets")

    kv_bytes_per_token = (
        geometry["num_hidden_layers"]
        * geometry["num_key_value_heads"]
        * (geometry["head_dim"] + geometry["head_dim"])
        * _BF16_BYTES
    )
    return {
        "schema": "a-sglang-checkpoint-weight-profile-v1",
        "config": dict(config),
        "weight_bytes_lower_bound": weight_bytes_lower_bound,
        "kv_bytes_per_token": kv_bytes_per_token,
        "supported_tensor_parallel_size": 1,
        "header_identity": {
            "model.safetensors": {
                "file_bytes": file_bytes,
                "header_length": header_length,
                "header_sha256": header_sha256,
            }
        },
        "ignored_tied_tensor_names": ["lm_head.weight"] if "lm_head.weight" in tensor_names else [],
    }


def inspect_checkpoint_weights(checkpoint: Path) -> dict[str, Any]:
    """Inspect config plus a bounded safetensors header; never load model weights."""
    checkpoint = Path(checkpoint).resolve()
    config_path = checkpoint / "config.json"
    weights_path = checkpoint / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise ValueError("checkpoint must contain config.json and model.safetensors")
    shards = sorted(path.name for path in checkpoint.glob("*.safetensors"))
    if shards != ["model.safetensors"]:
        raise ValueError("only one unsharded model.safetensors checkpoint is supported")

    config_bytes = config_path.read_bytes()
    try:
        config = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("config.json is not valid UTF-8 JSON") from exc
    if not isinstance(config, Mapping):
        raise ValueError("config.json must contain an object")

    file_bytes = weights_path.stat().st_size
    with weights_path.open("rb") as handle:
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError("model.safetensors is missing its 8-byte header length")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError("model.safetensors header exceeds the 16 MiB limit")
        if file_bytes < 8 + header_length:
            raise ValueError("model.safetensors is shorter than its declared header")
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError("model.safetensors header is truncated")
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("model.safetensors header is not valid UTF-8 JSON") from exc
    if not isinstance(header, Mapping):
        raise ValueError("model.safetensors header must contain an object")

    profile = summarize_checkpoint_headers(
        config,
        {
            "model.safetensors": {
                "file_bytes": file_bytes,
                "header_length": header_length,
                "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
                "header": header,
            }
        },
    )
    return {
        **profile,
        "checkpoint": str(checkpoint),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "bytes_read": len(config_bytes) + 8 + header_length,
        "tensor_payload_bytes_read": 0,
    }


def assess_sglang_memory(
    *,
    free_memory_bytes: int,
    total_memory_bytes: int,
    weight_bytes_lower_bound: int,
    kv_bytes_per_token: int,
    mem_fraction_static: float,
    c2kv_pool_fraction: float,
    raw_token_cap: int,
    required_raw_tokens: int,
    required_gist_tokens: int,
    page_size: int = 128,
) -> dict[str, Any]:
    """Reject only pre-load configurations that fail a necessary condition."""
    free_memory_bytes = _require_int(free_memory_bytes, "free_memory_bytes")
    total_memory_bytes = _require_int(total_memory_bytes, "total_memory_bytes")
    if free_memory_bytes > total_memory_bytes:
        raise ValueError("free_memory_bytes cannot exceed total_memory_bytes")
    weight_bytes_lower_bound = _require_int(
        weight_bytes_lower_bound, "weight_bytes_lower_bound", minimum=0
    )
    kv_bytes_per_token = _require_int(kv_bytes_per_token, "kv_bytes_per_token")
    raw_token_cap = _require_int(raw_token_cap, "raw_token_cap")
    required_raw_tokens = _require_int(required_raw_tokens, "required_raw_tokens")
    required_gist_tokens = _require_int(
        required_gist_tokens, "required_gist_tokens", minimum=0
    )
    page_size = _require_int(page_size, "page_size")
    mem_fraction_static = _require_fraction(mem_fraction_static, "mem_fraction_static")
    c2kv_pool_fraction = _require_fraction(c2kv_pool_fraction, "c2kv_pool_fraction")

    static_budget_upper_bound = math.ceil(free_memory_bytes * mem_fraction_static)
    raw_rest_upper_bound = static_budget_upper_bound - weight_bytes_lower_bound
    raw_profiled_upper_bound = raw_rest_upper_bound // kv_bytes_per_token
    raw_capped_upper_bound = min(raw_profiled_upper_bound, raw_token_cap)
    raw_resolved_upper_bound = (raw_capped_upper_bound // page_size) * page_size
    required_raw_paged = math.ceil(required_raw_tokens / page_size) * page_size

    gist_budget_bytes = int(total_memory_bytes * c2kv_pool_fraction)
    gist_bytes_per_token = kv_bytes_per_token + 2 * _INT64_BYTES
    gist_reserved_slot_bytes = kv_bytes_per_token + _INT64_BYTES
    gist_capacity = max(gist_budget_bytes - gist_reserved_slot_bytes, 0) // gist_bytes_per_token
    gist_pool_allocation_lower_bound = (
        (gist_capacity + 1) * kv_bytes_per_token
        + (gist_capacity + 1) * _INT64_BYTES
        + gist_capacity * _INT64_BYTES
    )
    required_raw_kv_lower_bound = (required_raw_paged + 1) * kv_bytes_per_token
    minimum_coexisting_bytes_lower_bound = (
        weight_bytes_lower_bound
        + required_raw_kv_lower_bound
        + gist_pool_allocation_lower_bound
    )

    reasons: list[str] = []
    if raw_resolved_upper_bound < required_raw_paged:
        reasons.append("raw_pool_capacity_below_paged_request")
    if gist_capacity < required_gist_tokens:
        reasons.append("gist_pool_capacity_below_request")
    if minimum_coexisting_bytes_lower_bound > free_memory_bytes:
        reasons.append("minimum_coexisting_allocation_exceeds_preload_free_memory")

    inputs = {
        "free_memory_bytes": free_memory_bytes,
        "total_memory_bytes": total_memory_bytes,
        "weight_bytes_lower_bound": weight_bytes_lower_bound,
        "kv_bytes_per_token": kv_bytes_per_token,
        "mem_fraction_static": mem_fraction_static,
        "c2kv_pool_fraction": c2kv_pool_fraction,
        "raw_token_cap": raw_token_cap,
        "required_raw_tokens": required_raw_tokens,
        "required_gist_tokens": required_gist_tokens,
        "page_size": page_size,
    }
    return {
        "schema": "a-sglang-memory-admission-v1",
        "status": "rejected" if reasons else "necessary_conditions_passed",
        "preflight_allows_attempt": not reasons,
        "reasons": reasons,
        "inputs": inputs,
        "calculated_bounds": {
            "static_fraction_budget_upper_bound_bytes": static_budget_upper_bound,
            "raw_rest_memory_upper_bound_bytes": raw_rest_upper_bound,
            "raw_profiled_token_capacity_upper_bound": raw_profiled_upper_bound,
            "raw_capped_token_capacity_upper_bound": raw_capped_upper_bound,
            "raw_resolved_token_capacity_upper_bound": raw_resolved_upper_bound,
            "required_raw_paged_token_capacity": required_raw_paged,
            "required_raw_kv_lower_bound_bytes": required_raw_kv_lower_bound,
            "gist_pool_budget_bytes": gist_budget_bytes,
            "gist_bytes_per_token": gist_bytes_per_token,
            "gist_reserved_slot_bytes": gist_reserved_slot_bytes,
            "gist_pool_token_capacity": gist_capacity,
            "gist_pool_allocation_lower_bound_bytes": gist_pool_allocation_lower_bound,
            "minimum_coexisting_bytes_lower_bound": minimum_coexisting_bytes_lower_bound,
        },
        "scope": (
            "This pre-weight-load screen rejects only failed necessary conditions. "
            "The ceil(static_fraction * free_memory) bound is intentionally optimistic "
            "by at most one byte; a pass does not guarantee fit. The after-load SGLang "
            "profiler remains authoritative, and a rejection does not claim a physical OOM."
        ),
    }
