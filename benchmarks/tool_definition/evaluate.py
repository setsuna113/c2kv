"""Five-method, recorded next-action tool-definition evaluation on one T0 base."""
from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Mapping

from benchmarks import toolmemory
from .core import (canonical_calls, deserialize_memory, parse_calls,
                   runtime_modules, sha256_file)
from .eviction import evictable_mask, generate_with_eviction
from .prepare import LAYOUTS, SCHEMA

METHODS = ("c2kv", "streamingllm", "h2o", "snapkv", "pyramidkv")
RAW_METHODS = METHODS[1:]


def _validate_t0_generation_contract(config, ratio: int) -> None:
    """The bundled A-line generator predates the frozen T0 checkpoint profile."""
    expected = {
        "gist_type": "dynamic-interleave",
        "gist_param": "qkv",
        "gist_residual_type": "embed-mean",
        "history_memory_normal_query": "base",
        "history_memory_training_profile": "next-compression-base-query-v1",
        "history_memory_variant": "T0",
        "history_memory_compression_domain": "tool",
        "history_memory_render_profile": toolmemory.RENDER_PROFILE,
    }
    for name, value in expected.items():
        if getattr(config, name, None) != value:
            raise ValueError(f"T0 generation config {name} differs from {value!r}")
    supported = getattr(config, "history_memory_supported_ratios", None)
    if supported != [8, 12] or ratio not in supported:
        raise ValueError("T0 generation requires ratio 8 or 12 on an exact [8, 12] checkpoint")


def _gist_parameters(model, checkpoint: Path) -> None:
    """Restore saved FP32 gist tensors after a BF16/FP16 model load."""
    import torch
    from safetensors import safe_open

    def is_gist(name: str) -> bool:
        return name.startswith("model.gist_embed_tokens.") or any(
            marker in name for marker in (".gist_q_proj.", ".gist_k_proj.", ".gist_v_proj."))

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
        files = {name: checkpoint / filename for name, filename in weight_map.items() if is_gist(name)}
    else:
        single = checkpoint / "model.safetensors"
        with safe_open(single, framework="pt", device="cpu") as handle:
            files = {name: single for name in handle.keys() if is_gist(name)}
    saved = {}
    for path in set(files.values()):
        with safe_open(path, framework="pt", device="cpu") as handle:
            saved.update({name: handle.get_tensor(name) for name, location in files.items()
                          if location == path})
    live = {name: parameter for name, parameter in model.named_parameters() if is_gist(name)}
    if not saved or set(saved) != set(live):
        raise ValueError("T0 checkpoint gist parameters are incomplete")
    with torch.no_grad():
        for name, parameter in live.items():
            if parameter.shape != saved[name].shape:
                raise ValueError(f"T0 gist shape differs for {name}")
            parameter.data = saved[name].to(device=parameter.device, dtype=torch.float32)


def load_t0(checkpoint: Path, *, device: str, dtype: str):
    import torch
    from transformers import AutoTokenizer

    if device.startswith("npu"):
        import torch_npu  # noqa: F401 - registers the NPU device and kernels

    runtime_modules()
    from history_memory.inference import EventNativeGenerator
    from history_memory.runtime import HistoryMemoryModel
    from models.qwen3.configuration_qwen3 import Qwen3Config
    from models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    class T0EventNativeGenerator(EventNativeGenerator):
        def _validate_model_contract(self, ratio: int) -> None:
            _validate_t0_generation_contract(self.runtime.base_model.config, ratio)

    checkpoint = checkpoint.resolve()
    toolmemory.load_tool_checkpoint_contract(checkpoint, toolmemory.ToolMemorySpec(ratio=8))
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("dtype must be bfloat16, float16, or float32")
    compute_dtype = getattr(torch, dtype)
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if selected.type == "npu" and not torch.npu.is_available():
        raise RuntimeError("NPU is unavailable")
    if selected.type not in {"cuda", "npu", "cpu"}:
        raise ValueError("recorded tool evaluator supports CUDA, NPU, or CPU")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    config = Qwen3Config.from_pretrained(checkpoint, local_files_only=True)
    model = Qwen3ForCausalLM.from_pretrained(
        checkpoint, config=config, local_files_only=True,
        dtype=compute_dtype, attn_implementation="eager", low_cpu_mem_usage=True,
    ).to(selected)
    _gist_parameters(model, checkpoint)
    model.eval()
    generator = T0EventNativeGenerator(HistoryMemoryModel(model))
    return model, tokenizer, generator


def read_manifest(path: Path, checkpoint: Path) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, dict]]]:
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("purpose") != "recorded_next_action_tool_definition":
        raise ValueError("not a recorded tool-definition manifest")
    if manifest["checkpoint"]["config_sha256"] != sha256_file(checkpoint / "config.json"):
        raise ValueError("T0 checkpoint config differs from preparation")
    model_files = manifest["checkpoint"].get("model_files_sha256")
    observed_model_files = {item.name for item in checkpoint.glob("model*.safetensors")}
    if not isinstance(model_files, dict) or not model_files or set(model_files) != observed_model_files:
        raise ValueError("T0 checkpoint weight file set differs from preparation")
    if any(sha256_file(checkpoint / name) != digest for name, digest in model_files.items()):
        raise ValueError("T0 checkpoint weights differ from preparation")
    for ratio in manifest["ratios"]:
        toolmemory.load_tool_checkpoint_contract(checkpoint, toolmemory.ToolMemorySpec(ratio=ratio))
    tokenizer_files = manifest.get("tokenizer_files_sha256")
    if not isinstance(tokenizer_files, dict) or not tokenizer_files.get("tokenizer.json"):
        raise ValueError("prepared manifest lacks tokenizer identity")
    if any(not (checkpoint / name).is_file() or sha256_file(checkpoint / name) != digest
           for name, digest in tokenizer_files.items()):
        raise ValueError("checkpoint tokenizer differs from preparation")
    observed_tokenizer_files = {
        name for name in ("tokenizer.json", "tokenizer_config.json",
                          "special_tokens_map.json", "added_tokens.json",
                          "chat_template.jinja")
        if (checkpoint / name).is_file()
    }
    if set(tokenizer_files) != observed_tokenizer_files:
        raise ValueError("checkpoint tokenizer file set differs from preparation")
    info = manifest["records"]
    records_path = (path.parent / info["path"]).resolve()
    if records_path.parent != path.parent or not records_path.is_file():
        raise ValueError("records must sit next to manifest")
    if records_path.stat().st_size != info["bytes"] or sha256_file(records_path) != info["sha256"]:
        raise ValueError("recorded decisions differ from manifest hash/size")
    groups: dict[tuple[str, int], dict[str, dict]] = {}
    count = 0
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("schema") != SCHEMA or record.get("layout") not in LAYOUTS:
                raise ValueError("invalid tool-definition record")
            key = (record["decision_id"], record["ratio"])
            layouts = groups.setdefault(key, {})
            if record["layout"] in layouts:
                raise ValueError(f"duplicate layout for {key}")
            layouts[record["layout"]] = record
            count += 1
    if count != info["count"] or len(groups) != manifest["decisions"] * len(manifest["ratios"]):
        raise ValueError("prepared decision count differs from manifest")
    for key, layouts in groups.items():
        if set(layouts) != set(LAYOUTS):
            raise ValueError(f"incomplete paired layouts for {key}")
        if len({record["prompt_sha256"] for record in layouts.values()}) != 1:
            raise ValueError(f"paired layouts use different visible source for {key}")
        if len({json.dumps(record["gold_tool_calls"], sort_keys=True) for record in layouts.values()}) != 1:
            raise ValueError(f"paired layouts use different gold action for {key}")
        if len({record["base_prompt_tokens_without_tool_protocol"] for record in layouts.values()}) != 1:
            raise ValueError(f"paired layouts use different base prompt length for {key}")
    return manifest, groups


def _eos_ids(tokenizer) -> tuple[int, ...]:
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(int(tokenizer.eos_token_id))
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        ids.append(im_end)
    return tuple(dict.fromkeys(ids))


def _outcome(gold: list[dict], text: str) -> dict[str, Any]:
    try:
        predicted = parse_calls(text)
        malformed = False
    except (ValueError, TypeError, KeyError):
        predicted, malformed = [], True
    return {
        "tool_decision": bool(gold),
        "strict_ordered_call_correct": (not malformed and predicted == gold) if gold else None,
        "first_tool_name_correct": (bool(predicted) and predicted[0]["name"] == gold[0]["name"]) if gold else None,
        "false_tool_call": (malformed or bool(predicted)) if not gold else None,
        "malformed": malformed,
        "predicted_tool_calls": predicted,
    }


def _record_result(record: Mapping[str, Any], *, method: str, layout: str,
                   tokens: tuple[int, ...], finish_reason: str, tokenizer,
                   per_layer_kv: tuple[int, ...], allowance: int | None,
                   replayed: bool, full_resident_tokens: int,
                   kv_logical_bytes: int | None = None,
                   measurement: str = "cache_shape",
                   budget_status: str | None = None) -> dict[str, Any]:
    text = tokenizer.decode(tokens, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    gold = canonical_calls(record["gold_tool_calls"])
    if not per_layer_kv:
        raise ValueError("resident KV needs at least one layer")
    base = int(record["base_prompt_tokens_without_tool_protocol"])
    denominator = (full_resident_tokens - base) * len(per_layer_kv)
    if denominator <= 0:
        raise ValueError("Full native tool protocol has no positive marginal KV denominator")
    resident_sum = sum(per_layer_kv)
    numerator = resident_sum - base * len(per_layer_kv)
    if budget_status is None:
        budget_status = ("not_matched" if allowance is None else
                         "within_allowance" if resident_sum <= allowance * len(per_layer_kv)
                         else "exceeds_allowance")
    return {
        "schema": "c2kv-paper-tool-definition-result-v1",
        "decision_id": record["decision_id"], "source": record["source"],
        "ratio": record["ratio"], "method": method, "layout": layout,
        "k": record["k"], "seed": record["seed"],
        "prompt_sha256": record["prompt_sha256"],
        "gold_tool_calls": gold,
        "generated_token_ids": list(tokens), "generated_text": text,
        "finish_reason": finish_reason,
        "outcome": _outcome(gold, text),
        "tool_token_spans": record.get("tool_token_spans"),
        "native_tool_indices": record["native_indices"],
        "resident_kv_tokens_by_layer": list(per_layer_kv),
        "resident_kv_tokens_layer_sum": resident_sum,
        "resident_kv_tokens_full_model_equivalent": (
            (resident_sum + len(per_layer_kv) - 1) // len(per_layer_kv)),
        "resident_kv_measurement": measurement,
        "resident_kv_logical_bytes_after_prefill": kv_logical_bytes,
        "resident_kv_bytes_per_layer_token": (
            kv_logical_bytes / resident_sum if kv_logical_bytes is not None else None),
        "base_prompt_tokens_without_tool_protocol": base,
        "full_native_resident_kv_tokens_per_layer": full_resident_tokens,
        "marginal_tool_kv_tokens_layer_sum": numerator,
        "marginal_full_native_tool_kv_tokens_layer_sum": denominator,
        "R_tool_numerator_tokens": denominator,
        "R_tool_denominator_tokens": numerator,
        "R_tool": denominator / numerator if numerator > 0 else None,
        "tool_kv_retention_fraction": numerator / denominator,
        "R_tool_accounting": "marginal resident KV against original messages without tool protocol; includes native schemas, gist, and protocol",
        "matched_allowance_tokens_per_layer": allowance,
        "budget_status": budget_status,
        "first_token_replayed_after_selection": replayed,
        "scope": "recorded next-action proxy; no tool execution or official task success",
    }


def _accumulate(groups: dict[tuple[str, str, int], dict[str, Any]], row: Mapping[str, Any]) -> None:
    key = (row["method"], row["layout"], row["ratio"])
    summary = groups.setdefault(key, {
        "method": key[0], "layout": key[1], "ratio": key[2],
        "rows": 0, "gold_tool_rows": 0, "strict_ordered_call_correct": 0,
        "first_tool_name_correct": 0, "gold_no_call_rows": 0,
        "false_tool_calls": 0, "resident_kv_tokens_layer_sum": 0,
        "marginal_tool_kv_tokens_layer_sum": 0,
        "marginal_full_native_tool_kv_tokens_layer_sum": 0,
        "budget_exceeded_rows": 0,
    })
    summary["rows"] += 1
    outcome = row["outcome"]
    if outcome["tool_decision"]:
        summary["gold_tool_rows"] += 1
        summary["strict_ordered_call_correct"] += int(outcome["strict_ordered_call_correct"])
        summary["first_tool_name_correct"] += int(outcome["first_tool_name_correct"])
    else:
        summary["gold_no_call_rows"] += 1
        summary["false_tool_calls"] += int(outcome["false_tool_call"])
    summary["resident_kv_tokens_layer_sum"] += row["resident_kv_tokens_layer_sum"]
    summary["marginal_tool_kv_tokens_layer_sum"] += row["marginal_tool_kv_tokens_layer_sum"]
    summary["marginal_full_native_tool_kv_tokens_layer_sum"] += row["marginal_full_native_tool_kv_tokens_layer_sum"]
    summary["budget_exceeded_rows"] += int(row["budget_status"] == "exceeds_allowance")


def _finalize_summaries(groups: dict[tuple[str, str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for key in sorted(groups):
        item = dict(groups[key])
        tool = item["gold_tool_rows"]
        no_call = item["gold_no_call_rows"]
        item["strict_ordered_call_accuracy"] = item["strict_ordered_call_correct"] / tool if tool else None
        item["first_tool_name_accuracy"] = item["first_tool_name_correct"] / tool if tool else None
        item["false_tool_call_rate"] = item["false_tool_calls"] / no_call if no_call else None
        retained = item["marginal_tool_kv_tokens_layer_sum"]
        native = item["marginal_full_native_tool_kv_tokens_layer_sum"]
        item["R_tool_numerator_tokens"] = native
        item["R_tool_denominator_tokens"] = retained
        item["R_tool"] = native / retained if retained > 0 else None
        item["tool_kv_retention_fraction"] = retained / native
        summaries.append(item)
    return summaries


def evaluate(manifest_path: Path, checkpoint: Path, output: Path, *, device: str,
             dtype: str, max_new_tokens: int, methods: tuple[str, ...] = METHODS,
             layouts: tuple[str, ...] = LAYOUTS, limit: int | None = None) -> dict[str, Any]:
    import torch

    if max_new_tokens < 1 or not methods or any(item not in METHODS for item in methods):
        raise ValueError("invalid methods or max_new_tokens")
    if not layouts or any(item not in LAYOUTS for item in layouts):
        raise ValueError("unknown or empty layout set")
    manifest, groups = read_manifest(manifest_path, checkpoint.resolve())
    if output.exists():
        raise FileExistsError(output)
    model, tokenizer, generator = load_t0(checkpoint, device=device, dtype=dtype)
    output.mkdir(parents=True)
    rows_path = output / "results.jsonl"
    count = 0
    summaries: dict[tuple[str, str, int], dict[str, Any]] = {}
    try:
        with rows_path.open("w", encoding="utf-8") as sink:
            for key in sorted(groups)[:limit]:
                pair = groups[key]
                full = pair["full"]
                full_memory = deserialize_memory(full["memory"])
                prompt = tuple(full_memory.system_input_ids) + tuple(full_memory.workspace_input_ids)
                spans = [tuple(span) for span in full["tool_token_spans"]]
                region = (spans[0][0], spans[-1][1])
                for method in methods:
                    for layout in layouts:
                        if method != "c2kv" and layout in {"full", "retrieval"}:
                            continue
                        record = pair[layout]
                        if method == "c2kv":
                            memory = deserialize_memory(record["memory"])
                            with torch.inference_mode():
                                generation = generator.generate(
                                    memory, ratio=record["ratio"],
                                    max_new_tokens=max_new_tokens,
                                    eos_token_id=_eos_ids(tokenizer),
                                )
                            actual_kv = generation.stats.get("resident_kv_tokens_after_raw_prefill")
                            actual_bytes = generation.stats.get("resident_kv_logical_bytes_after_raw_prefill")
                            planned_kv = int(record["resident_kv_tokens"])
                            if actual_kv is None or int(actual_kv) != planned_kv:
                                raise RuntimeError(f"T0 resident KV disagrees with plan for {key}/{layout}: "
                                                   f"actual={actual_kv}, planned={planned_kv}")
                            if actual_bytes is None or int(actual_bytes) <= 0:
                                raise RuntimeError("T0 generation lacks physical KV byte measurement")
                            result = _record_result(
                                record, method=method, layout=layout,
                                tokens=generation.token_ids,
                                finish_reason=generation.finish_reason,
                                tokenizer=tokenizer,
                                per_layer_kv=(int(actual_kv),)
                                * int(model.config.num_hidden_layers),
                                allowance=(pair["hybrid"]["resident_kv_tokens"]
                                           if layout in {"random", "retrieval"} else None),
                                replayed=False,
                                full_resident_tokens=int(full["resident_kv_tokens"]),
                                kv_logical_bytes=int(actual_bytes),
                                measurement="event_native_actual_after_raw_prefill",
                            )
                        else:
                            protected = record["native_indices"] if layout != "uniform" else []
                            mask = evictable_mask(region, spans, protected)
                            n_evictable = int(mask.sum())
                            allowance = (pair["uniform"]["resident_kv_tokens"]
                                         if layout == "uniform" else pair["hybrid"]["resident_kv_tokens"])
                            mandatory = len(prompt) - n_evictable
                            keep = min(n_evictable, max(0, allowance - mandatory))
                            with torch.inference_mode():
                                generation = generate_with_eviction(
                                    model, prompt, region=region, evictable=mask,
                                    keep=keep, method=method,
                                    max_new_tokens=max_new_tokens,
                                    eos_ids=_eos_ids(tokenizer),
                                    obs_window=64 if method == "pyramidkv" else 16,
                                    kernel=5 if method == "pyramidkv" else 7,
                                )
                            per_layer = generation.per_layer_kept_tokens
                            if mandatory <= allowance:
                                if max(per_layer) > allowance and method != "pyramidkv":
                                    raise RuntimeError("raw tool KV exceeded matched allowance")
                                if method == "pyramidkv" and sum(per_layer) > allowance * len(per_layer):
                                    raise RuntimeError("PyramidKV layer-sum exceeded matched allowance")
                            result = _record_result(
                                record, method=method, layout=layout,
                                tokens=generation.token_ids,
                                finish_reason=generation.finish_reason,
                                tokenizer=tokenizer, per_layer_kv=per_layer,
                                allowance=allowance,
                                replayed=generation.first_token_replayed_after_eviction,
                                full_resident_tokens=int(full["resident_kv_tokens"]),
                                kv_logical_bytes=generation.resident_kv_logical_bytes_after_replay,
                                measurement="raw_cache_tensor_shapes_after_replay",
                                budget_status=("exceeds_allowance" if sum(per_layer) > allowance * len(per_layer)
                                               else "layer_mean_within_allowance" if max(per_layer) > allowance
                                               else "within_allowance"),
                            )
                            result["eviction"] = {
                                "evictable_tool_tokens": generation.evictable_tokens,
                                "retained_evictable_tool_tokens_nominal": generation.keep,
                                "prefill_chunks": generation.prefill_chunks,
                                "selection_profile": ("pyramidkv_official_schedule_headwise_reference_v1"
                                                      if method == "pyramidkv" else method),
                            }
                        sink.write(json.dumps(result, ensure_ascii=False,
                                              separators=(",", ":"), allow_nan=False) + "\n")
                        count += 1
                        _accumulate(summaries, result)
        report = {
            "schema": "c2kv-paper-tool-definition-evaluation-v1",
            "status": "completed",
            "manifest_sha256": sha256_file(manifest_path.resolve()),
            "checkpoint_config_sha256": manifest["checkpoint"]["config_sha256"],
            "device": device, "dtype": dtype,
            "max_new_tokens": max_new_tokens,
            "methods": list(methods), "layouts": list(layouts),
            "limit": limit, "result_rows": count,
            "results_sha256": sha256_file(rows_path),
            "result_scope": "recorded next-action proxy; no tool execution or official task success",
            "raw_kv_adaptation": "full prefill; select tool KV; replay final prompt token against selected KV",
            "initial_prefill_saved": False,
            "grouped_metrics": _finalize_summaries(summaries),
        }
        (output / "evaluation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    except BaseException as exc:
        failure = {
            "schema": "c2kv-paper-tool-definition-evaluation-v1",
            "status": "failed", "manifest_sha256": sha256_file(manifest_path.resolve()),
            "checkpoint_config_sha256": manifest["checkpoint"]["config_sha256"],
            "device": device, "dtype": dtype, "methods": list(methods),
            "layouts": list(layouts), "limit": limit, "result_rows": count,
            "results_sha256": sha256_file(rows_path) if rows_path.exists() else None,
            "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            "result_scope": "incomplete; no grouped metrics are valid",
        }
        (output / "evaluation.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
