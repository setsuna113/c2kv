from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import blend_gist_key_values, get_model_class  # noqa: E402


NPU_FUSION_ATTENTION_IMPL = "npu_fusion_attention"


def _load_safe_attn_impl(attn_impl: str) -> str:
    return "eager" if attn_impl == NPU_FUSION_ATTENTION_IMPL else attn_impl


def _set_model_attn_impl(model: Any, attn_impl: str) -> None:
    if attn_impl != NPU_FUSION_ATTENTION_IMPL:
        return
    model.config._attn_implementation = attn_impl
    inner_model = getattr(model, "model", None)
    if inner_model is not None and hasattr(inner_model, "config"):
        inner_model.config._attn_implementation = attn_impl


def _gist_compatible_config(config_class: Any, model_path: str, tokenizer: Any) -> Any:
    return config_class.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        gist_type="dynamic-interleave",
        gist_param="qkv",
        gist_residual_type="embed-mean",
        gist_overlap=64,
        gist_token_id=tokenizer.eos_token_id,
        pad_token_id=None,
    )


def _setup_device(device_type: str) -> str:
    if device_type == "auto":
        if hasattr(torch, "npu") and torch.npu.is_available():
            device_type = "npu"
        elif torch.cuda.is_available():
            device_type = "cuda"
        else:
            device_type = "cpu"
    if device_type == "npu":
        import torch_npu  # noqa: F401

        torch.npu.set_device(0)
    elif device_type == "cuda":
        torch.cuda.set_device(0)
    return device_type


def _sync_device(device: Any) -> None:
    device_type = getattr(device, "type", str(device))
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device_type == "npu" and hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()


def chat_template_ids(
    tokenizer: Any,
    messages: Sequence[Dict[str, Any]],
    *,
    tools: Optional[List[Dict[str, Any]]] = None,
    add_generation_prompt: bool = False,
    keep_bos: bool = False,
    max_length: Optional[int] = None,
) -> List[int]:
    encoded = tokenizer.apply_chat_template(
        list(messages),
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
        max_length=max_length + 1 if max_length is not None and not keep_bos else max_length,
        truncation=max_length is not None,
    )
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
    if not keep_bos and ids and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    return ids


@torch.inference_mode()
def _prefill_system(model: Any, input_ids: torch.Tensor, attn_impl: str) -> tuple[Any, int, float]:
    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    attention_mask = torch.ones_like(input_ids)
    _sync_device(input_ids.device)
    start = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        logits_to_keep=1,
    )
    _sync_device(input_ids.device)
    elapsed = time.perf_counter() - start
    model.model.config._attn_implementation = original_attn_impl
    return outputs.past_key_values, int(attention_mask.sum().item()), elapsed


@torch.inference_mode()
def _prefill_tokens_with_cache(
    model: Any,
    input_ids: torch.Tensor,
    past_key_values: Any,
    past_length: int,
    attn_impl: str,
) -> tuple[Any, int, float]:
    if input_ids.shape[1] == 0:
        return past_key_values, 0, 0.0
    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    input_length = input_ids.shape[1]
    attention_mask = torch.ones(
        (input_ids.shape[0], past_length + input_length),
        dtype=torch.long,
        device=input_ids.device,
    )
    position_ids = torch.arange(
        past_length,
        past_length + input_length,
        dtype=torch.long,
        device=input_ids.device,
    ).unsqueeze(0)
    _sync_device(input_ids.device)
    start = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=True,
        logits_to_keep=1,
    )
    _sync_device(input_ids.device)
    elapsed = time.perf_counter() - start
    model.model.config._attn_implementation = original_attn_impl
    return outputs.past_key_values, input_length, elapsed


@torch.inference_mode()
def _build_tool_cache(
    model: Any,
    context_input_ids: torch.Tensor,
    system_cache: Any,
    system_length: int,
    attn_impl: str,
    override_ratio: int,
) -> tuple[Any, int, int, float, float, float]:
    device = model.device
    context_input_ids = context_input_ids.to(device)
    valid_mask = context_input_ids != -100
    tool_length = int(valid_mask.sum().item())
    input_ids = context_input_ids.clone()
    input_ids[~valid_mask] = model.model.gist_token_id

    original_attn_impl = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    gist_kwargs = {}
    if getattr(model.config, "gist_type", None) == "dynamic-interleave":
        gist_kwargs["ratio"] = override_ratio
    _sync_device(input_ids.device)
    compress_start = time.perf_counter()
    outputs, gist_mask, pos_ids = model.model.generate_gist(
        input_ids=input_ids,
        attention_mask=valid_mask,
        **gist_kwargs,
    )
    _sync_device(input_ids.device)
    tool_compress_sec = time.perf_counter() - compress_start
    model.model.config._attn_implementation = original_attn_impl

    _sync_device(input_ids.device)
    blend_start = time.perf_counter()
    tool_cache, _ = blend_gist_key_values(
        model.config,
        [outputs.past_key_values],
        [gist_mask],
        [pos_ids],
        model.model.rotary_emb,
        system_length,
    )
    for system_layer, tool_layer in zip(system_cache.layers, tool_cache.layers):
        tool_layer.keys = torch.cat([system_layer.keys, tool_layer.keys], dim=-2)
        tool_layer.values = torch.cat([system_layer.values, tool_layer.values], dim=-2)
    _sync_device(input_ids.device)
    blend_sec = time.perf_counter() - blend_start
    gist_tokens = max(0, tool_cache.get_seq_length() - system_length)
    actual_ratio = float(tool_length / gist_tokens) if gist_tokens else 0.0
    return tool_cache, tool_length, gist_tokens, actual_ratio, tool_compress_sec, blend_sec


@torch.inference_mode()
def _generate_from_input_ids(
    model: Any,
    tokenizer: Any,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    attn_impl: str,
    use_gist: bool = False,
    position_ids: Optional[torch.Tensor] = None,
    past_key_values: Any = None,
) -> tuple[str, float, int, float]:
    original_attn_impl = model.model.config._attn_implementation if hasattr(model, "model") else None
    if original_attn_impl is not None:
        model.model.config._attn_implementation = attn_impl
    attention_mask = torch.ones_like(input_ids)
    generate_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if position_ids is not None:
        generate_kwargs["position_ids"] = position_ids
    if past_key_values is not None:
        generate_kwargs["past_key_values"] = past_key_values
    if use_gist:
        generate_kwargs["use_gist"] = True
    _sync_device(input_ids.device)
    start = time.perf_counter()
    generated = model.generate(**generate_kwargs)
    _sync_device(input_ids.device)
    latency = time.perf_counter() - start
    if original_attn_impl is not None:
        model.model.config._attn_implementation = original_attn_impl
    prediction = tokenizer.decode(
        generated[0, input_ids.shape[1] :],
        skip_special_tokens=True,
    ).strip()
    generated_tokens = int(generated.shape[1] - input_ids.shape[1])
    tbt_sec = latency / generated_tokens if generated_tokens > 0 else 0.0
    return prediction, latency, generated_tokens, tbt_sec


def _load_model(args: argparse.Namespace, tokenizer: Any, device: str) -> Any:
    model_path = args.base_model if getattr(args, "untrained_c2kv", False) else args.model
    dtype = (
        torch.bfloat16
        if args.dtype == "bf16"
        else torch.float16
        if args.dtype == "fp16"
        else torch.float32
    )
    mode = getattr(args, "mode", "c2kv")
    baseline_model_class = getattr(args, "baseline_model_class", "gist")
    if mode in ("full", "truncate") and baseline_model_class == "auto":
        if args.generate_attn_impl == NPU_FUSION_ATTENTION_IMPL:
            config_class, model_class = get_model_class(model_path, "qkv")
            config = _gist_compatible_config(config_class, model_path, tokenizer)
            model = model_class.from_pretrained(
                model_path,
                config=config,
                trust_remote_code=True,
                local_files_only=True,
                device_map={"": device} if device != "cpu" else None,
                dtype=dtype,
                attn_implementation=_load_safe_attn_impl(args.generate_attn_impl),
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
                device_map={"": device} if device != "cpu" else None,
                dtype=dtype,
                attn_implementation=args.generate_attn_impl,
            )
        _set_model_attn_impl(model, args.generate_attn_impl)
        model.eval()
        return model

    config_class, model_class = get_model_class(model_path, "qkv")
    if getattr(args, "untrained_c2kv", False):
        if not args.base_model:
            raise ValueError("--base_model is required for untrained C2KV.")
        config = _gist_compatible_config(config_class, model_path, tokenizer)
        model = model_class.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": device} if device != "cpu" else None,
            dtype=dtype,
            attn_implementation=_load_safe_attn_impl(args.generate_attn_impl),
        )
    else:
        model = model_class.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            device_map={"": device} if device != "cpu" else None,
            dtype=dtype,
            attn_implementation=_load_safe_attn_impl(args.generate_attn_impl),
        )
    _set_model_attn_impl(model, args.generate_attn_impl)
    model.eval()
    return model
