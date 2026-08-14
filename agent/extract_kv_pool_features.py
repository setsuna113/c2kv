#!/usr/bin/env python
"""Extract pooled KV features for the gist readability probe (R2 forensics).

For every frozen S4 sample (configs/s4_frozen_qids.json) this builds the prefix
KV cache exactly like the S4 history eval does — gist arm = ``c2kv`` mode via
``_build_c2kv_prefix``, full arm = ``full`` mode via
``_build_full_or_truncate_prefix`` (agent/eval_agent_history_c2kv.py) — and
stores the pre-registered pooling spectrum of ``prefix["cache"]``:

  for every layer and every KV head:
    - K and V at the LAST prefix position (the position generation starts from)
    - mean of K and V over ALL positions of that head ("layer-mean")

  flat npz keys: layer{i}_head{h}_{k_last,v_last,k_mean,v_mean}

Raw multi-position KV is never persisted; only the pooled float16 arrays above.
Per-sample metadata goes to features_index.jsonl (one row per qid x arm),
flushed after every sample so the job is checkpointable; --resume skips
(qid, arm) pairs already present in the index.

Per-sample RuntimeError handling mirrors the eval: OOM -> skipped index row
(_is_oom_error), anything else re-raises. Cache is freed between samples with
the eval's _clear_device_cache helper.

Runs on the NPU box (torch + torch_npu required); sklearn is NOT needed here.

Example:
  python agent/extract_kv_pool_features.py \
    --qids_file configs/s4_frozen_qids.json \
    --out_dir ./outputs/gist_probe_features \
    --arms gist,full --device npu:0 --ratio 4
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_agent_tool_definition_c2kv import (  # noqa: E402
    _extract_tool_name,
    _load_model,
    _setup_device,
)
from eval_agent_history_c2kv import (  # noqa: E402
    _build_c2kv_prefix,
    _build_full_or_truncate_prefix,
    _build_history_chunks,
    _clear_device_cache,
    _is_oom_error,
    _load_tokenizer,
    _resolve_model_checkpoint,
)
from train.train_data_multiturn import (  # noqa: E402
    AgentLLMTracesCompressHistorySource,
    CompressHistoryExample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

POOL_ARRAYS = ("k_last", "v_last", "k_mean", "v_mean")
FEATURE_BYTES_TARGET = 1_000_000  # pre-registered <= ~1MB/sample budget


def _mirror_s4_eval_args(cli: argparse.Namespace) -> argparse.Namespace:
    """Minimal args Namespace mirroring the S4-eval defaults.

    Sources:
      - agent/eval_agent_history_s4_npu.sh (S4 invocation): split=eval,
        split_manifest_name=subset_disjoint, split_seed=42, eval_ratio=0.1,
        max_samples_per_session=4, include_tools=True, ratio=4, eager attn.
      - agent/eval_agent_history_c2kv.py parse_args() defaults for the rest.

    Fields required by the reused code paths (from reading the eval script):
      - _build_c2kv_prefix / _build_history_chunks / _history_messages:
        min_doc_num, max_doc_length, max_doc_num, max_history_tokens,
        history_selection, split_oversized_history_docs, max_system_length,
        system_attn_impl, gist_attn_impl, override_ratio
      - _build_full_or_truncate_prefix (mode="full"): additionally
        truncate_selection, max_prompt_tokens, max_baseline_input_tokens,
        generate_attn_impl
      - _load_model (tooldef script): model, base_model, untrained_c2kv, mode,
        baseline_model_class, generate_attn_impl, dtype
      - _load_tokenizer: tokenizer, base_model, model
    """
    return argparse.Namespace(
        # dataset/source (S4 script values)
        dataset_path=cli.dataset_path,
        split="eval",
        eval_ratio=0.1,
        split_seed=42,
        split_manifest_file=None,
        split_manifest_name="subset_disjoint",
        max_samples_per_session=4,
        max_source_examples=None,
        require_tool_call=False,
        max_input_chars=None,
        max_answer_chars=None,
        include_tools=True,
        prefix_history_doc_num=None,
        prefix_history_exact=False,
        selection_filter="c2kv",
        # history chunking / prefix builders (eval parse_args defaults)
        max_doc_length=768,
        min_doc_num=1,
        max_doc_num=16,
        max_history_tokens=12288,
        max_length=1536,
        max_system_length=4096,
        max_prompt_tokens=1536,
        max_baseline_input_tokens=16000,
        history_selection="tail",
        truncate_selection="tail",
        split_oversized_history_docs=True,
        # model / generation
        model=cli.model_path,
        base_model=cli.base_model_path or None,
        tokenizer=cli.tokenizer_path or None,
        override_ratio=cli.ratio,
        system_attn_impl="eager",
        gist_attn_impl="eager",
        generate_attn_impl="eager",
        dtype="bf16",
        baseline_model_class="auto",
        untrained_c2kv=False,
        mode="c2kv",  # overridden per arm
    )


def _load_frozen_qids(path: str) -> List[str]:
    """Tolerant loader for configs/s4_frozen_qids.json (schema provided later).

    Accepts: a JSON list of qid strings (or dicts with a "qid" field), or a
    JSON dict with a "qids"/"frozen_qids"/"s4_frozen_qids" list, or a dict
    keyed by qid.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    qids: List[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and "qid" in item:
                qids.append(str(item["qid"]))
            else:
                qids.append(str(item))
    elif isinstance(payload, dict):
        for key in ("qids", "frozen_qids", "s4_frozen_qids"):
            if isinstance(payload.get(key), list):
                qids = [str(item) for item in payload[key]]
                break
        else:
            qids = [str(key) for key in payload]
    else:
        raise ValueError(f"Unsupported qids file schema in {path}: {type(payload).__name__}")
    if not qids:
        raise ValueError(f"No qids found in {path}")
    return qids


def _setup_requested_device(device: str) -> str:
    device_type, _, index = device.partition(":")
    resolved = _setup_device(device_type)
    if index:
        if resolved == "npu" and hasattr(torch, "npu"):
            torch.npu.set_device(int(index))
        elif resolved == "cuda":
            torch.cuda.set_device(int(index))
    return resolved


def _estimate_feature_bytes(config: Any) -> Tuple[int, int, int, int]:
    """(n_layers, n_kv_heads, head_dim, bytes) for the pooling spectrum."""
    n_layers = int(getattr(config, "num_hidden_layers", 0) or 0)
    n_kv_heads = int(getattr(config, "num_key_value_heads", 0) or 0)
    head_dim = getattr(config, "head_dim", None)
    if not head_dim:
        hidden = int(getattr(config, "hidden_size", 0) or 0)
        n_heads = int(getattr(config, "num_attention_heads", 0) or 0)
        head_dim = hidden // n_heads if n_heads else 0
    total = n_layers * n_kv_heads * int(head_dim or 0) * len(POOL_ARRAYS) * 2  # float16
    return n_layers, n_kv_heads, int(head_dim or 0), total


def _warn_if_feature_bytes_exceeds(config: Any, arm: str) -> None:
    n_layers, n_kv_heads, head_dim, total = _estimate_feature_bytes(config)
    logger.info(
        "Pooling spectrum size (%s arm, from model config): %d layers x %d kv heads x "
        "%d head_dim x %d arrays x 2 bytes = %d bytes (%.2f MB)",
        arm, n_layers, n_kv_heads, head_dim, len(POOL_ARRAYS), total, total / 1e6,
    )
    if total > FEATURE_BYTES_TARGET:
        logger.warning(
            "Pooling spectrum %.2f MB exceeds the pre-registered <= %.2f MB/sample target!",
            total / 1e6, FEATURE_BYTES_TARGET / 1e6,
        )


def _iter_cache_layer_kv(cache: Any):
    """Yield (keys, values) per layer from an HF DynamicCache.

    keys/values have shape (batch=1, n_kv_heads, seq_len, head_dim); the eval
    builds and concatenates them along dim=-2 (see _build_tool_cache /
    _append_independent_cache in the eval scripts).
    """
    layers = getattr(cache, "layers", None)
    if layers is not None:
        for layer in layers:
            yield layer.keys, layer.values
        return
    if hasattr(cache, "key_cache"):  # legacy transformers cache layout
        for keys, values in zip(cache.key_cache, cache.value_cache):
            yield keys, values
        return
    for layer_idx in range(len(cache)):
        yield cache[layer_idx]


def _pool_prefix_cache(cache: Any) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Pre-registered pooling spectrum of a prefix cache.

    For every layer and KV head: K/V at the last prefix position plus the mean
    of K/V over all positions of that head. Means are computed in float32 (the
    cache is bf16) and everything is stored as float16. Raw multi-position KV
    is dropped as soon as the pooled arrays exist.
    """
    features: Dict[str, np.ndarray] = {}
    n_layers = 0
    n_kv_heads = 0
    head_dim = 0
    seq_len = int(cache.get_seq_length()) if hasattr(cache, "get_seq_length") else 0
    for layer_idx, (keys, values) in enumerate(_iter_cache_layer_kv(cache)):
        k = keys[0].detach().to(torch.float32).cpu()  # (n_kv_heads, seq_len, head_dim)
        v = values[0].detach().to(torch.float32).cpu()
        if not seq_len:
            seq_len = int(k.shape[1])
        pooled = {
            "k_last": k[:, -1, :],
            "v_last": v[:, -1, :],
            "k_mean": k.mean(dim=1),
            "v_mean": v.mean(dim=1),
        }
        del k, v, keys, values
        n_layers += 1
        n_kv_heads = int(pooled["k_last"].shape[0])
        head_dim = int(pooled["k_last"].shape[1])
        for head in range(n_kv_heads):
            for name in POOL_ARRAYS:
                features[f"layer{layer_idx}_head{head}_{name}"] = (
                    pooled[name][head].numpy().astype(np.float16)
                )
        del pooled
    meta = {
        "n_layers": n_layers,
        "n_kv_heads": n_kv_heads,
        "head_dim": head_dim,
        "cache_seq_len": seq_len,
    }
    return features, meta


def _session_tool_names(example: CompressHistoryExample) -> List[str]:
    """Tool names of the session, in definition order (deduped)."""
    names: List[str] = []
    for tool in example.tools or []:
        if not isinstance(tool, dict):
            continue
        name = tool.get("name")
        if not name and isinstance(tool.get("function"), dict):
            name = tool["function"].get("name")
        if name and str(name) not in names:
            names.append(str(name))
    return names


def _sanitize_qid(qid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", qid)


def _index_row(
    example: CompressHistoryExample,
    arm: str,
    *,
    skipped: bool,
    skip_reason: Optional[str],
    meta: Optional[Dict[str, int]] = None,
    features_file: Optional[str] = None,
    prefix: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = meta or {}
    prefix = prefix or {}
    return {
        "qid": example.qid,
        "session_id": example.qid.rsplit(":", 1)[0] if ":" in example.qid else None,
        "arm": arm,
        "n_layers": meta.get("n_layers"),
        "n_kv_heads": meta.get("n_kv_heads"),
        "head_dim": meta.get("head_dim"),
        "cache_seq_len": meta.get("cache_seq_len"),
        "target_tool_name": _extract_tool_name(example.answer),
        "session_tool_names": _session_tool_names(example),
        "skipped": skipped,
        "skip_reason": skip_reason,
        # extras beyond the pre-registered schema (harmless, useful for audits)
        "features_file": features_file,
        "gist_tokens": prefix.get("gist_tokens"),
        "doc_tokens": prefix.get("doc_tokens"),
        "system_length": prefix.get("system_length"),
        "history_length": prefix.get("history_length"),
        "actual_compression_ratio": prefix.get("actual_compression_ratio"),
    }


def _load_done_pairs(index_path: Path, out_dir: Path) -> set:
    done = set()
    if not index_path.exists():
        return done
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Ignoring unparseable trailing index line (crash remnant)")
                continue
            if row.get("skipped"):
                done.add((row.get("qid"), row.get("arm")))
                continue
            features_file = row.get("features_file")
            if features_file and (out_dir / features_file).exists():
                done.add((row.get("qid"), row.get("arm")))
    return done


def _load_model_for_arm(model_args: argparse.Namespace, tokenizer: Any, device: str, arm: str) -> Any:
    arm_args = copy.copy(model_args)
    if arm == "gist":
        arm_args.mode = "c2kv"
        arm_args.model = _resolve_model_checkpoint(arm_args.model)
    else:
        # S4 mirror: the full arm runs the BASE model (plain AutoModelForCausalLM),
        # falling back to the c2kv checkpoint path if no base model is given.
        arm_args.mode = "full"
        arm_args.baseline_model_class = "auto"
        if arm_args.base_model:
            arm_args.model = arm_args.base_model
    logger.info("Loading model for arm=%s model=%s", arm, arm_args.model)
    model = _load_model(arm_args, tokenizer, device)
    _warn_if_feature_bytes_exceeds(model.config, arm)
    return model


def extract(cli: argparse.Namespace) -> Dict[str, Any]:
    device = _setup_requested_device(cli.device)
    args = _mirror_s4_eval_args(cli)
    args.model = _resolve_model_checkpoint(args.model)
    tokenizer = _load_tokenizer(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    frozen_qids = _load_frozen_qids(cli.qids_file)
    frozen_set = set(frozen_qids)
    logger.info("Loaded %d frozen qids from %s", len(frozen_qids), cli.qids_file)

    source = AgentLLMTracesCompressHistorySource(
        args.dataset_path,
        split=args.split,
        eval_ratio=args.eval_ratio,
        split_seed=args.split_seed,
        split_manifest_file=args.split_manifest_file,
        split_manifest_name=args.split_manifest_name,
        max_samples_per_session=args.max_samples_per_session,
        max_records=args.max_source_examples,
        require_tool_call=args.require_tool_call,
        max_input_chars=args.max_input_chars,
        max_answer_chars=args.max_answer_chars,
        include_tools=args.include_tools,
        prefix_history_doc_num=args.prefix_history_doc_num,
        prefix_history_exact=args.prefix_history_exact,
    )
    selection_skips: Counter[str] = Counter()
    examples: List[CompressHistoryExample] = []
    seen_qids = set()
    for example in source:
        if example.qid not in frozen_set or example.qid in seen_qids:
            continue
        # Mirror the eval's default selection_filter="c2kv" so a frozen qid that
        # no longer passes chunk selection is recorded as skipped, not run.
        _, _, _, _, select_skip = _build_history_chunks(tokenizer, example, args)
        if select_skip is not None:
            selection_skips[select_skip] += 1
        seen_qids.add(example.qid)
        examples.append((example, select_skip))
        if cli.max_examples and len(examples) >= cli.max_examples:
            break
    missing = frozen_set - seen_qids
    if missing:
        logger.warning("%d frozen qids not found in the eval split, e.g. %s", len(missing), sorted(missing)[:3])
    logger.info("Selected %d examples; selection_skips=%s", len(examples), dict(selection_skips))

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "features_index.jsonl"
    if cli.resume:
        done = _load_done_pairs(index_path, out_dir)
        logger.info("Resume: %d (qid, arm) pairs already done", len(done))
        index_handle = index_path.open("a", encoding="utf-8")
    else:
        if index_path.exists():
            raise RuntimeError(
                f"{index_path} already exists; pass --resume to continue it or choose a fresh --out_dir"
            )
        done = set()
        index_handle = index_path.open("w", encoding="utf-8")

    arms = [item.strip() for item in cli.arms.split(",") if item.strip()]
    stats: Dict[str, Counter[str]] = {arm: Counter() for arm in arms}
    try:
        for arm in arms:
            model = _load_model_for_arm(args, tokenizer, device, arm)
            try:
                for example, select_skip in tqdm(examples, desc=f"extract[{arm}]"):
                    if (example.qid, arm) in done:
                        stats[arm]["resume_skipped"] += 1
                        continue
                    if select_skip is not None:
                        row = _index_row(example, arm, skipped=True, skip_reason=select_skip)
                        index_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        index_handle.flush()
                        stats[arm]["selection_skipped"] += 1
                        continue
                    prefix: Optional[Dict[str, Any]] = None
                    try:
                        if arm == "gist":
                            prefix, skip_reason = _build_c2kv_prefix(model, tokenizer, example, args)
                        else:
                            prefix, skip_reason = _build_full_or_truncate_prefix(
                                model, tokenizer, example, args, "full"
                            )
                        if skip_reason is not None or prefix is None:
                            row = _index_row(example, arm, skipped=True, skip_reason=skip_reason)
                            stats[arm]["build_skipped"] += 1
                        else:
                            features, meta = _pool_prefix_cache(prefix["cache"])
                            filename = f"{_sanitize_qid(example.qid)}_{arm}.npz"
                            np.savez(out_dir / filename, **features)
                            del features
                            row = _index_row(
                                example,
                                arm,
                                skipped=False,
                                skip_reason=None,
                                meta=meta,
                                features_file=filename,
                                prefix=prefix,
                            )
                            stats[arm]["ok"] += 1
                    except RuntimeError as error:
                        if not _is_oom_error(error):
                            raise
                        logger.warning("Skipping sample after OOM: arm=%s qid=%s", arm, example.qid)
                        row = _index_row(example, arm, skipped=True, skip_reason="oom")
                        stats[arm]["oom"] += 1
                    index_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    index_handle.flush()
                    del prefix
                    _clear_device_cache(device)
            finally:
                del model
                _clear_device_cache(device)
    finally:
        index_handle.close()

    summary = {
        "qids_file": cli.qids_file,
        "out_dir": str(out_dir),
        "arms": arms,
        "ratio": args.override_ratio,
        "num_frozen_qids": len(frozen_qids),
        "num_examples": len(examples),
        "num_missing_qids": len(missing),
        "selection_skips": dict(selection_skips),
        "stats": {arm: dict(counter) for arm, counter in stats.items()},
    }
    logger.info("Summary: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract pooled KV features for the gist readability probe.")
    parser.add_argument("--qids_file", default="configs/s4_frozen_qids.json")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--arms", default="gist,full", help="Comma-separated subset of gist,full.")
    parser.add_argument("--model_path", default="./checkpoints/qwen3-4b-agent-history-c2kv-npu")
    parser.add_argument(
        "--base_model_path",
        default="./models/Qwen3-4B-Instruct-2507",
        help="Base model for the full arm (S4 mirror). Empty string reuses --model_path.",
    )
    parser.add_argument("--tokenizer_path", default=None, help="Defaults to base model then model path.")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--ratio", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=0, help="<=0 means all frozen examples.")
    parser.add_argument("--resume", action="store_true", help="Skip (qid, arm) pairs already in the index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown_arms = set(args.arms.split(",")) - {"gist", "full"}
    if unknown_arms:
        raise SystemExit(f"Unknown arms: {sorted(unknown_arms)} (allowed: gist,full)")
    print(json.dumps(extract(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
