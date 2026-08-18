"""Generation eval for true-joint C2KV checkpoints on agent-llm-traces.

Mirrors ``agent/eval_unified_next_action_c2kv.py`` (row/summary shape, modes
c2kv/c2kv_untrained/truncate/full, tool arg-name/value F1) and
``agent/eval_agent_history_c2kv.py`` (``_target_metrics`` metric bundle,
``--compare_modes``/``--ratios`` loop, split-manifest args,
``_resolve_model_checkpoint``, OOM-skip policy), but builds examples with
``AgentLLMTracesJointSource`` (``python/train/train_data_joint.py``) so the
context grid matches true-joint training: tool-schema chunks FIRST, then
history-turn chunks in chronological order, bare system prefix, current turn
as the ordinary prompt, next assistant action as the target.

Conditions
----------
``--condition {joint, tool_only, history_only}`` selects which document
subset is C2KV-compressed (the same doc-side construction as
``JointDataset.preprocess_example``):

- ``joint``: tool chunks (up to ``max_tool_chunks``) + history chunks
  (remaining ``max_doc_num`` slots, tail-biased);
- ``tool_only``: tool chunks get all ``max_doc_num`` slots; history turns are
  NOT shown at all (absent, mirroring the tooldef-path setting);
- ``history_only``: history chunks get all slots; tool schemas are absent.

Modes (per condition) mirror the existing evals: ``c2kv`` (checkpoint gist
params), ``c2kv_untrained`` (base-init gist params), ``full`` (all documents
in the plain prompt, no compression), ``truncate`` (plain prompt documents
head-truncated to ``ceil(doc_tokens / ratio)``, the unified eval's truncate
semantics).  Baseline modes load ``--base_model`` when given (history-eval
convention); with the frozen-base training recipe
(``train_agent_tool_definition_c2kv.py`` ``param.requires_grad_("gist" in
name)``) the checkpoint's base weights equal the base model's, and full /
truncate generate with ``use_gist=False`` so gist params are never touched —
the two conventions are output-equivalent.

J-separate arm (``--separate``)
--------------------------------
Two checkpoints, ``--checkpoint_tool`` and ``--checkpoint_history``: tool docs
are compressed with the TOOL checkpoint's gist params, history docs with the
HISTORY checkpoint's gist params, and both gist-KV sets are concatenated into
one prefix after the system cache.  Per-doc RoPE repositioning is EXACTLY the
single-model one: each side is blended with
``models.blend_gist_key_values`` (gist_utils.py:781), whose
``_concat_gist_key_values`` (gist_utils.py:751-779) accumulates each doc's
gist positions by its ORIGINAL token length starting from a caller-given
prefix length — tool side starts at ``system_length``, history side at
``system_length + tool_doc_tokens``, so history gist positions are identical
to what one model compressing the joint grid would produce (the same
accumulation as ``process_context_input_ids``, gist_utils.py:594-601).

Exactness rationale, verified against the modeling code:

- Gist KVs depend ONLY on the compressing checkpoint's gist params
  (``gist_q/k/v_proj`` + ``gist_embed_tokens``): ``generate_gist``
  (python/models/qwen3/modeling_qwen3.py:533-599) routes through
  ``forward_with_gist`` (:287-350), where doc tokens use base projections and
  gist tokens use gist projections.  So A-tool-KV and B-history-KV are exactly
  what each checkpoint would produce on its own docs.
- Base weights are architecturally identical across the two checkpoints:
  joint/tooldef/history training freezes everything but gist params
  (``param.requires_grad_("gist" in name)``), so embed_tokens / q,k,v,o_proj /
  MLP / norms / lm_head / rotary ``inv_freq`` are bit-identical; using the
  generator checkpoint's ``rotary_emb`` for the blend is exact.

CAVEAT — the prompt/answer forward is NOT gist-free: with ``use_gist=True``
the ordinary attention forward substitutes ``gist_q/k/v_proj`` for the
prompt/answer tokens too (modeling_qwen3.py:242-246).  Training sets
``use_gist=True`` for the prompt forward whenever context docs are present
(modeling_qwen3.py:660), and ALL existing c2kv evals generate with
``use_gist=True`` when the prefix was gist-compressed
(``_generate_from_input_ids(..., use_gist=True)``).  This eval keeps that
convention for every c2kv arm, including ``--separate``: generation therefore
exercises the GENERATOR checkpoint's gist projections on the prompt/answer
path, so "generate under either model" is exact w.r.t. base weights but NOT
w.r.t. the answer-side gist projections — A-generate and B-generate differ by
exactly that.  Pick the generator with ``--separate_generator {tool,history}``
(default tool).  Generating with ``use_gist=False`` would be
generator-independent (identical frozen base weights) but would diverge from
both training conditions and the other arms' generation semantics, so it is
not done here.

Ratio control: ``--ratios`` (default 8) maps to the same override mechanism
``python/inference/expr_c2kv.py`` uses — ``generate_gist(ratio=...)`` for
``gist_type == "dynamic-interleave"`` (see ``_build_tool_cache`` in
eval_agent_tool_definition_c2kv.py:289-337).

Outputs: per-row jsonl + ``.summary.json`` aggregates in the unified eval's
shape (grouped by condition × mode × ratio), so ``agent/merge_*.py``-style
tooling keeps working; ``--merge_only`` merges condition-aware shards written
by ``agent/eval_joint_next_action_c2kv_npu.sh``.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import blend_gist_key_values  # noqa: E402
from eval_agent_tool_definition_c2kv import (  # noqa: E402
    _build_tool_cache,
    _generate_from_input_ids,
    _load_model,
    _prefill_system,
    _prefill_tokens_with_cache,
    _setup_device,
    _sync_device,
)
from eval_agent_history_c2kv import (  # noqa: E402
    _clear_device_cache,
    _is_oom_error,
    _load_tokenizer,
    _oom_row,
    _resolve_model_checkpoint,
    _target_metrics,
)
from eval_toolathlon_first_tool_c2kv import _arg_f1s, _parse_pred_call  # noqa: E402
from train.train_data_joint import (  # noqa: E402
    AgentLLMTracesJointSource,
    JointExample,
    TOOL_DOC_PREFIX,
    _default_max_tool_chunks,
)
from train.train_data_multiturn import (  # noqa: E402
    _chat_template_ids,
    _fit_reused_history,
    _pad,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

CONDITIONS = ("joint", "tool_only", "history_only")
C2KV_MODES = {"c2kv", "c2kv_untrained"}
BASELINE_MODES = {"full", "truncate"}
SEPARATE_MODE = "c2kv_separate"


def _jsonl_write(path: str, rows: List[Dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Condition -> document-subset construction (mirrors
# JointDataset.preprocess_example doc-side logic in train_data_joint.py).
# ---------------------------------------------------------------------------


def _condition_doc_chunks(
    tokenizer: Any,
    example: JointExample,
    condition: str,
    *,
    max_doc_length: int,
    max_doc_num: int,
    max_tool_chunks: Optional[int],
    max_tool_definition_tokens: int,
    history_selection: str,
    split_oversized_history_docs: bool,
) -> Tuple[Optional[List[List[int]]], Optional[List[List[int]]], Optional[str]]:
    """Build (tool_chunks, history_chunks) id lists for one condition.

    Each chunk is a chat-template-wrapped document of at most
    ``max_doc_length`` tokens (NOT padded).  Returns ``(None, None, reason)``
    when the tool side exceeds ``max_tool_definition_tokens`` (the training
    skip).  Budget allocation is exactly ``JointDataset.preprocess_example``:
    joint caps tools at ``min(max_tool_chunks, max_doc_num)`` and gives
    history the remaining slots; tool_only/history_only give their side all
    ``max_doc_num`` slots and drop the other side entirely.
    """

    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition!r}")
    if max_tool_chunks is None:
        max_tool_chunks = _default_max_tool_chunks(max_doc_num)

    tool_chunks: List[List[int]] = []
    if condition != "history_only":
        tool_cap = max_doc_num if condition == "tool_only" else min(max_tool_chunks, max_doc_num)
        doc_id_groups = [
            _chat_template_ids(tokenizer, [{"role": "user", "content": TOOL_DOC_PREFIX + document}])
            for document in example.tool_documents
            if document.strip()
        ]
        doc_tokens = sum(len(doc_ids) for doc_ids in doc_id_groups)
        if doc_tokens > max_tool_definition_tokens:
            return None, None, f"tool_definition_tokens>{max_tool_definition_tokens}"
        for doc_ids in doc_id_groups:
            tool_chunks.extend(
                doc_ids[start : start + max_doc_length]
                for start in range(0, len(doc_ids), max_doc_length)
            )
        tool_chunks = tool_chunks[:tool_cap]

    history_chunks: List[List[int]] = []
    if condition != "tool_only":
        history_budget = max_doc_num if condition == "history_only" else max_doc_num - len(tool_chunks)
        raw_history = [
            {"role": "user", "content": text}
            for text in example.history_documents
            if text and text.strip()
        ]
        if history_budget > 0 and raw_history:
            fitted = _fit_reused_history(
                tokenizer,
                raw_history,
                max_doc_length=max_doc_length,
                max_doc_num=history_budget,
                policy=history_selection,
                split_oversized_history_docs=split_oversized_history_docs,
            )
            history_chunks = [
                _chat_template_ids(tokenizer, [message], max_length=max_doc_length)
                for message in fitted
            ]
    return tool_chunks, history_chunks, None


def _doc_grid(chunks: Sequence[List[int]], max_doc_length: int) -> torch.Tensor:
    rows = [_pad(chunk, max_doc_length, -100) for chunk in chunks]
    return torch.tensor(rows, dtype=torch.long)


def _flat_doc_ids(tool_chunks: Sequence[List[int]], history_chunks: Sequence[List[int]]) -> List[int]:
    return [token for chunk in [*tool_chunks, *history_chunks] for token in chunk]


# ---------------------------------------------------------------------------
# Metrics: history-eval bundle + unified-eval argument F1.
# ---------------------------------------------------------------------------


def _target_payload(target: str) -> Optional[Dict[str, Any]]:
    parsed = _parse_pred_call(target)
    if not parsed:
        return None
    return {"name": parsed.get("name"), "arguments": parsed.get("arguments", {})}


def _prediction_metrics(tokenizer: Any, target: str, prediction: str) -> Dict[str, Any]:
    metrics = _target_metrics(tokenizer, target, prediction)
    target_payload = _target_payload(target) or {"name": None, "arguments": {}}
    pred_payload = _parse_pred_call(prediction)
    arg_name_f1, arg_value_f1 = _arg_f1s(target_payload, pred_payload)
    metrics.update({
        "argument_name_f1": round(arg_name_f1, 4),
        "argument_value_f1": round(arg_value_f1, 4),
    })
    return metrics


# ---------------------------------------------------------------------------
# Prefix construction.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _build_c2kv_prefix(
    model: Any,
    tokenizer: Any,
    example: JointExample,
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    tool_chunks, history_chunks, skip_reason = _condition_doc_chunks(
        tokenizer,
        example,
        args.condition,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_chunks=args.max_tool_chunks,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        history_selection=args.history_selection,
        split_oversized_history_docs=args.split_oversized_history_docs,
    )
    if skip_reason is not None:
        return None, skip_reason
    chunks = [*tool_chunks, *history_chunks]
    if len(chunks) < args.min_doc_num:
        return None, f"doc_num<{args.min_doc_num}"

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )

    context_input_ids = _doc_grid(chunks, args.max_doc_length)
    cache, doc_tokens, gist_tokens, actual_ratio, compress_sec, blend_sec = _build_tool_cache(
        model,
        context_input_ids,
        system_cache,
        system_length,
        args.gist_attn_impl,
        args.override_ratio,
    )
    return {
        "cache": cache,
        "system_length": system_length,
        "doc_length": doc_tokens,
        "cache_length": cache.get_seq_length(),
        "use_gist": True,
        "doc_tokens": doc_tokens,
        "doc_chunks": len(chunks),
        "tool_doc_chunks": len(tool_chunks),
        "history_doc_chunks": len(history_chunks),
        "gist_tokens": gist_tokens,
        "compressed_tokens": gist_tokens,
        "actual_compression_ratio": actual_ratio,
        "system_prefill_sec": system_prefill_sec,
        "tool_compress_sec": compress_sec,
        "full_prefill_sec": 0.0,
        "blend_sec": blend_sec,
    }, None


@torch.inference_mode()
def _compress_docs_to_cache(
    model: Any,
    context_input_ids: torch.Tensor,
    prefix_length: int,
    attn_impl: str,
    override_ratio: int,
) -> Tuple[Any, int, int, float, float]:
    """Compress one doc grid with ``model``'s gist params (NO system concat).

    Returns (cache, doc_tokens, gist_tokens, compress_sec, blend_sec).  This
    is the ``_build_tool_cache`` (eval_agent_tool_definition_c2kv.py:289)
    flow minus the system-cache concatenation, so the caller can place the
    resulting gist KVs at an arbitrary original-token prefix length; the
    per-doc RoPE repositioning inside ``blend_gist_key_values`` accumulates
    from ``prefix_length`` exactly as the single-model blend does from
    ``system_length``.
    """

    device = model.device
    context_input_ids = context_input_ids.to(device)
    valid_mask = context_input_ids != -100
    doc_tokens = int(valid_mask.sum().item())
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
    compress_sec = time.perf_counter() - compress_start
    model.model.config._attn_implementation = original_attn_impl

    _sync_device(input_ids.device)
    blend_start = time.perf_counter()
    cache, _ = blend_gist_key_values(
        model.config,
        [outputs.past_key_values],
        [gist_mask],
        [pos_ids],
        model.model.rotary_emb,
        prefix_length,
    )
    _sync_device(input_ids.device)
    blend_sec = time.perf_counter() - blend_start
    gist_tokens = cache.get_seq_length()
    return cache, doc_tokens, gist_tokens, compress_sec, blend_sec


@torch.inference_mode()
def _build_separate_prefix(
    tool_model: Any,
    history_model: Any,
    generator_model: Any,
    tokenizer: Any,
    example: JointExample,
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """J-separate prefix: A-tool-KV + B-history-KV, one generator model.

    Tool docs are compressed with ``tool_model``'s gist params at prefix
    length ``system_length``; history docs with ``history_model``'s gist
    params at prefix length ``system_length + tool_doc_tokens``.  The
    per-side blends reproduce the single-model joint-grid positions exactly
    (see module docstring).  The system prefix is prefilled with
    ``generator_model`` — exact because base weights are frozen across
    checkpoints.  See the module docstring for the use_gist caveat on the
    answer-side forward.
    """

    tool_chunks, history_chunks, skip_reason = _condition_doc_chunks(
        tokenizer,
        example,
        "joint",
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_chunks=args.max_tool_chunks,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        history_selection=args.history_selection,
        split_oversized_history_docs=args.split_oversized_history_docs,
    )
    if skip_reason is not None:
        return None, skip_reason
    chunks = [*tool_chunks, *history_chunks]
    if len(chunks) < args.min_doc_num:
        return None, f"doc_num<{args.min_doc_num}"

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=generator_model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        generator_model, system_input_ids, args.system_attn_impl
    )

    side_caches: List[Any] = []
    prefix_length = system_length
    doc_tokens = 0
    gist_tokens = 0
    compress_sec = 0.0
    blend_sec = 0.0
    for side_model, side_chunks in (
        (tool_model, tool_chunks),
        (history_model, history_chunks),
    ):
        if not side_chunks:
            continue
        side_grid = _doc_grid(side_chunks, args.max_doc_length)
        side_cache, side_tokens, side_gist_tokens, side_compress_sec, side_blend_sec = (
            _compress_docs_to_cache(
                side_model,
                side_grid,
                prefix_length,
                args.gist_attn_impl,
                args.override_ratio,
            )
        )
        side_caches.append(side_cache)
        prefix_length += side_tokens
        doc_tokens += side_tokens
        gist_tokens += side_gist_tokens
        compress_sec += side_compress_sec
        blend_sec += side_blend_sec

    # Concatenate system + per-side gist KVs into one prefix cache (same
    # layer-concat idiom as _build_tool_cache).
    prefix_cache = side_caches[0]
    for layer_index, prefix_layer in enumerate(prefix_cache.layers):
        prefix_layer.keys = torch.cat(
            [system_cache.layers[layer_index].keys, prefix_layer.keys]
            + [side_cache.layers[layer_index].keys for side_cache in side_caches[1:]],
            dim=-2,
        )
        prefix_layer.values = torch.cat(
            [system_cache.layers[layer_index].values, prefix_layer.values]
            + [side_cache.layers[layer_index].values for side_cache in side_caches[1:]],
            dim=-2,
        )
    return {
        "cache": prefix_cache,
        "system_length": system_length,
        "doc_length": doc_tokens,
        "cache_length": prefix_cache.get_seq_length(),
        "use_gist": True,
        "doc_tokens": doc_tokens,
        "doc_chunks": len(chunks),
        "tool_doc_chunks": len(tool_chunks),
        "history_doc_chunks": len(history_chunks),
        "gist_tokens": gist_tokens,
        "compressed_tokens": gist_tokens,
        "actual_compression_ratio": doc_tokens / gist_tokens if gist_tokens else 0.0,
        "system_prefill_sec": system_prefill_sec,
        "tool_compress_sec": compress_sec,
        "full_prefill_sec": 0.0,
        "blend_sec": blend_sec,
    }, None


@torch.inference_mode()
def _build_baseline_prefix(
    model: Any,
    tokenizer: Any,
    example: JointExample,
    args: argparse.Namespace,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """full/truncate baselines: condition docs in the plain prompt.

    ``truncate`` keeps the first ``ceil(doc_tokens / ratio)`` tokens of the
    concatenated (tool-first) document stream — the unified eval's head
    truncation, NOT the history eval's tail-biased one.
    """

    tool_chunks, history_chunks, skip_reason = _condition_doc_chunks(
        tokenizer,
        example,
        args.condition,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_chunks=args.max_tool_chunks,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        history_selection=args.history_selection,
        split_oversized_history_docs=args.split_oversized_history_docs,
    )
    if skip_reason is not None:
        return None, skip_reason
    chunks = [*tool_chunks, *history_chunks]
    if len(chunks) < args.min_doc_num:
        return None, f"doc_num<{args.min_doc_num}"

    system_ids = _chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        keep_bos=True,
        max_length=args.max_system_length,
    )
    doc_ids = _flat_doc_ids(tool_chunks, history_chunks)
    doc_tokens = len(doc_ids)
    if args.mode == "truncate":
        kept_tokens = max(1, (doc_tokens + args.override_ratio - 1) // args.override_ratio)
        doc_ids = doc_ids[:kept_tokens]
    else:
        kept_tokens = doc_tokens

    prompt_ids = _chat_template_ids(
        tokenizer, example.current_messages, add_generation_prompt=True
    )
    if args.max_prompt_tokens and len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    total_len = len(system_ids) + len(doc_ids) + len(prompt_ids)
    if args.max_baseline_input_tokens and total_len > args.max_baseline_input_tokens:
        return None, f"baseline_input_tokens>{args.max_baseline_input_tokens}"

    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = _prefill_system(
        model, system_input_ids, args.system_attn_impl
    )
    doc_cache = system_cache
    doc_length = 0
    full_prefill_sec = 0.0
    if doc_ids:
        doc_input_ids = torch.tensor([doc_ids], dtype=torch.long, device=model.device)
        doc_cache, doc_length, full_prefill_sec = _prefill_tokens_with_cache(
            model,
            doc_input_ids,
            past_key_values=system_cache,
            past_length=system_length,
            attn_impl=args.generate_attn_impl,
        )
    return {
        "cache": doc_cache,
        "system_length": system_length,
        "doc_length": doc_length,
        "cache_length": doc_cache.get_seq_length(),
        "use_gist": False,
        "doc_tokens": doc_tokens,
        "doc_chunks": len(chunks),
        "tool_doc_chunks": len(tool_chunks),
        "history_doc_chunks": len(history_chunks),
        "gist_tokens": 0,
        "compressed_tokens": kept_tokens,
        "actual_compression_ratio": doc_tokens / kept_tokens if kept_tokens else 0.0,
        "system_prefill_sec": system_prefill_sec,
        "tool_compress_sec": 0.0,
        "full_prefill_sec": full_prefill_sec,
        "blend_sec": 0.0,
    }, None


# ---------------------------------------------------------------------------
# Generation + rows.
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _generate_with_prefix(
    model: Any,
    tokenizer: Any,
    example: JointExample,
    prefix: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    prompt_ids = _chat_template_ids(
        tokenizer, example.current_messages, add_generation_prompt=True
    )
    if args.max_prompt_tokens and len(prompt_ids) > args.max_prompt_tokens:
        prompt_ids = prompt_ids[-args.max_prompt_tokens :]
    prompt_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=model.device)
    mock_cache_ids = prompt_input_ids.new_zeros((1, prefix["cache_length"]))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    original_prefix_length = prefix["system_length"] + prefix["doc_length"]
    position_ids = torch.arange(
        original_prefix_length,
        original_prefix_length + prompt_input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    prediction, generate_sec, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=prefix["use_gist"],
        position_ids=position_ids,
        past_key_values=prefix["cache"],
    )
    metrics = _prediction_metrics(tokenizer, example.answer, prediction)
    metrics["generated_tokens"] = generated_tokens
    ttft = (
        prefix["system_prefill_sec"]
        + prefix["tool_compress_sec"]
        + prefix["full_prefill_sec"]
        + prefix["blend_sec"]
    )
    metrics.update({
        "doc_tokens": prefix["doc_tokens"],
        "doc_chunks": prefix["doc_chunks"],
        "tool_doc_chunks": prefix["tool_doc_chunks"],
        "history_doc_chunks": prefix["history_doc_chunks"],
        "gist_tokens": prefix["gist_tokens"],
        "compressed_tokens": prefix["compressed_tokens"],
        "prompt_tokens": len(prompt_ids),
        "actual_compression_ratio": round(prefix["actual_compression_ratio"], 4),
        "system_prefill_sec": round(prefix["system_prefill_sec"], 4),
        "tool_compress_sec": round(prefix["tool_compress_sec"], 4),
        "full_prefill_sec": round(prefix["full_prefill_sec"], 4),
        "blend_sec": round(prefix["blend_sec"], 4),
        "ttft_sec": round(ttft, 4),
        "latency_sec": round(generate_sec, 4),
        "generate_sec": round(generate_sec, 4),
        "tbt_sec": round(tbt_sec, 6),
        "total_sec": round(ttft + generate_sec, 4),
    })
    return metrics


def _row_base(example: JointExample, condition: str, mode: str, ratio: int) -> Dict[str, Any]:
    return {
        "qid": example.qid,
        "session_id": example.session_id,
        "subset": example.subset,
        "condition": condition,
        "mode": mode,
        "ratio": ratio,
    }


@torch.inference_mode()
def _generate_one(
    model: Any,
    tokenizer: Any,
    example: JointExample,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if args.mode in BASELINE_MODES:
        prefix, skip_reason = _build_baseline_prefix(model, tokenizer, example, args)
    else:
        prefix, skip_reason = _build_c2kv_prefix(model, tokenizer, example, args)
    row = _row_base(example, args.condition, args.row_mode, args.override_ratio)
    if prefix is None:
        row.update({"skipped": True, "skip_reason": skip_reason})
        return row
    row.update({"skipped": False})
    row.update(_generate_with_prefix(model, tokenizer, example, prefix, args))
    return row


@torch.inference_mode()
def _generate_one_separate(
    models: Dict[str, Any],
    tokenizer: Any,
    example: JointExample,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    prefix, skip_reason = _build_separate_prefix(
        models["tool"], models["history"], models["generator"], tokenizer, example, args
    )
    row = _row_base(example, "joint", SEPARATE_MODE, args.override_ratio)
    if prefix is None:
        row.update({"skipped": True, "skip_reason": skip_reason})
        return row
    row.update({"skipped": False, "separate_generator": args.separate_generator})
    row.update(_generate_with_prefix(models["generator"], tokenizer, example, prefix, args))
    return row


# ---------------------------------------------------------------------------
# Summary (unified-eval aggregate names, grouped by condition x mode x ratio).
# ---------------------------------------------------------------------------


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    keys = sorted({(row.get("condition"), row.get("mode"), row.get("ratio")) for row in rows})
    for condition, mode, ratio in keys:
        group = [
            row
            for row in rows
            if row.get("condition") == condition and row.get("mode") == mode and row.get("ratio") == ratio
        ]
        valid = [row for row in group if not row.get("skipped")]
        skips = Counter(row.get("skip_reason", "unknown") for row in group if row.get("skipped"))
        generated_total = sum(row.get("generated_tokens", 0) for row in valid)
        doc_token_total = sum(row.get("doc_tokens", 0) for row in valid)
        compressed_total = sum(row.get("compressed_tokens", 0) for row in valid)

        def _avg(field: str) -> float:
            return sum(row.get(field, 0.0) for row in valid) / len(valid) if valid else 0.0

        summaries.append({
            "condition": condition,
            "mode": mode,
            "ratio": ratio,
            "num_examples": len(group),
            "num_valid": len(valid),
            "num_skipped": len(group) - len(valid),
            "skip_reasons": dict(skips),
            "exact_match": sum(1 for row in valid if row["exact_match"]) / len(valid) if valid else 0.0,
            "tool_name_accuracy": (
                sum(1 for row in valid if row["tool_name_match"]) / len(valid) if valid else 0.0
            ),
            "tool_call_rate": sum(1 for row in valid if row["has_tool_call"]) / len(valid) if valid else 0.0,
            "response_type_accuracy": (
                sum(1 for row in valid if row["response_type_match"]) / len(valid) if valid else 0.0
            ),
            "argument_name_f1": _avg("argument_name_f1"),
            "argument_value_f1": _avg("argument_value_f1"),
            "avg_text_token_f1": _avg("text_token_f1"),
            "avg_rouge_l_f1": _avg("rouge_l_f1"),
            "avg_doc_tokens": _avg("doc_tokens"),
            "avg_doc_chunks": _avg("doc_chunks"),
            "avg_gist_tokens": _avg("gist_tokens"),
            "avg_compressed_tokens": _avg("compressed_tokens"),
            "avg_prompt_tokens": _avg("prompt_tokens"),
            "avg_generated_tokens": generated_total / len(valid) if valid else 0.0,
            "avg_actual_compression_ratio": _avg("actual_compression_ratio"),
            "token_weighted_actual_compression_ratio": (
                doc_token_total / compressed_total if compressed_total else 0.0
            ),
            "avg_system_prefill_sec": _avg("system_prefill_sec"),
            "avg_tool_compress_sec": _avg("tool_compress_sec"),
            "avg_full_prefill_sec": _avg("full_prefill_sec"),
            "avg_blend_sec": _avg("blend_sec"),
            "avg_ttft_sec": _avg("ttft_sec"),
            "avg_generate_sec": _avg("generate_sec"),
            "avg_tbt_sec": _avg("tbt_sec"),
            "token_weighted_tbt_sec": (
                sum(row.get("generate_sec", 0.0) for row in valid) / generated_total
                if generated_total else 0.0
            ),
            "avg_total_sec": _avg("total_sec"),
        })
    return summaries


# ---------------------------------------------------------------------------
# Example loading + eval loop.
# ---------------------------------------------------------------------------


def _load_examples(args: argparse.Namespace) -> List[JointExample]:
    source = AgentLLMTracesJointSource(
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
        prefix_history_doc_num=args.prefix_history_doc_num,
        prefix_history_exact=args.prefix_history_exact,
        canonical_format_prob=args.canonical_format_prob,
        minified_json_prob=args.minified_json_prob,
        shuffle_tools=args.shuffle_tools,
        truncate_description_chars=args.truncate_description_chars,
        max_tools_per_sample=args.max_tools_per_sample,
        same_namespace_negative_tools=args.same_namespace_negative_tools,
        random_negative_tools=args.random_negative_tools,
    )
    examples = list(source)
    if args.max_examples is not None and args.max_examples > 0:
        examples = examples[: args.max_examples]
    return examples


def _parse_csv(value: Optional[str], default: str) -> List[str]:
    return [item.strip() for item in (value or default).split(",") if item.strip()]


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = _setup_device(args.device_type)
    if args.model:
        args.model = _resolve_model_checkpoint(args.model)
    tokenizer = _load_tokenizer(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    examples = _load_examples(args)
    logger.info("Loaded %d joint %s examples", len(examples), args.split)
    ratios = [int(item) for item in _parse_csv(args.ratios, str(args.override_ratio))]
    rows: List[Dict[str, Any]] = []

    if args.separate:
        if not args.checkpoint_tool or not args.checkpoint_history:
            raise ValueError("--separate requires --checkpoint_tool and --checkpoint_history")
        args.checkpoint_tool = _resolve_model_checkpoint(args.checkpoint_tool)
        args.checkpoint_history = _resolve_model_checkpoint(args.checkpoint_history)
        models = {}
        for name, checkpoint in (("tool", args.checkpoint_tool), ("history", args.checkpoint_history)):
            model_args = copy.copy(args)
            model_args.model = checkpoint
            model_args.mode = "c2kv"
            model_args.untrained_c2kv = False
            logger.info("Loading %s checkpoint %s", name, checkpoint)
            models[name] = _load_model(model_args, tokenizer, device)
        models["generator"] = models[args.separate_generator]
        for ratio in ratios:
            run_args = copy.copy(args)
            run_args.override_ratio = ratio
            for example in tqdm(examples, desc=f"{SEPARATE_MODE}@{ratio}x"):
                try:
                    row = _generate_one_separate(models, tokenizer, example, run_args)
                except RuntimeError as error:
                    if not _is_oom_error(error):
                        raise
                    logger.warning("Skipping sample after OOM: mode=%s ratio=%s qid=%s", SEPARATE_MODE, ratio, example.qid)
                    row = _oom_row(example, SEPARATE_MODE, ratio)
                    row["condition"] = "joint"
                    _clear_device_cache(device)
                rows.append(row)
                _clear_device_cache(device)
        conditions = ["joint"]
        modes = [SEPARATE_MODE]
        del models
        _clear_device_cache(device)
    else:
        conditions = _parse_csv(args.conditions, args.condition)
        for condition in conditions:
            if condition not in CONDITIONS:
                raise ValueError(f"Unsupported --conditions entry {condition!r}; choose from {CONDITIONS}")
        modes = _parse_csv(args.compare_modes, args.mode)
        for mode in modes:
            if mode not in C2KV_MODES | BASELINE_MODES:
                raise ValueError(f"Unsupported mode {mode!r}; choose from {sorted(C2KV_MODES | BASELINE_MODES)}")
            run_ratios = [1] if mode == "full" else ratios
            model_args = copy.copy(args)
            model_args.untrained_c2kv = mode == "c2kv_untrained"
            model_args.mode = "c2kv" if mode == "c2kv_untrained" else mode
            model_args.row_mode = mode
            if mode in BASELINE_MODES and args.base_model:
                # History-eval convention: baselines run on the base model.
                # Output-equivalent to the joint checkpoint's (frozen) base
                # weights, and full/truncate never touch gist params.
                model_args.model = args.base_model
            logger.info("Loading model for mode=%s model=%s", mode, model_args.model)
            model = _load_model(model_args, tokenizer, device)
            for condition in conditions:
                for ratio in run_ratios:
                    run_args = copy.copy(model_args)
                    run_args.override_ratio = ratio
                    run_args.condition = condition
                    desc = f"{condition}/{mode}@{ratio}x" if mode != "full" else f"{condition}/full"
                    for example in tqdm(examples, desc=desc):
                        try:
                            row = _generate_one(model, tokenizer, example, run_args)
                        except RuntimeError as error:
                            if not _is_oom_error(error):
                                raise
                            logger.warning(
                                "Skipping sample after OOM: condition=%s mode=%s ratio=%s qid=%s",
                                condition, mode, ratio, example.qid,
                            )
                            row = _oom_row(example, mode, ratio)
                            row["condition"] = condition
                            _clear_device_cache(device)
                        rows.append(row)
                        _clear_device_cache(device)
            del model
            _clear_device_cache(device)

    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "separate": args.separate,
        "checkpoint_tool": args.checkpoint_tool if args.separate else None,
        "checkpoint_history": args.checkpoint_history if args.separate else None,
        "separate_generator": args.separate_generator if args.separate else None,
        "conditions": conditions,
        "modes": modes,
        "ratios": ratios,
        "history_selection": args.history_selection,
        "max_doc_length": args.max_doc_length,
        "max_doc_num": args.max_doc_num,
        "max_tool_chunks": args.max_tool_chunks,
        "min_doc_num": args.min_doc_num,
        "max_tool_definition_tokens": args.max_tool_definition_tokens,
        "max_system_length": args.max_system_length,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_baseline_input_tokens": args.max_baseline_input_tokens,
        "num_rows": len(rows),
        "results": _summarize(rows),
    }
    if args.output_file:
        _jsonl_write(args.output_file, rows)
        summary_path = str(Path(args.output_file).with_suffix(".summary.json"))
        Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wrote predictions to %s", args.output_file)
        logger.info("Wrote summary to %s", summary_path)
    return summary


# ---------------------------------------------------------------------------
# --merge_only: condition-aware shard merge (merge_unified_next_action_eval.py
# analog; that tool groups by (mode, ratio) only and would mix conditions).
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _common_valid_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = sorted({(row.get("condition"), row.get("mode"), row.get("ratio")) for row in rows})
    if not keys:
        return []
    valid_ids = []
    for condition, mode, ratio in keys:
        valid_ids.append({
            row.get("qid")
            for row in rows
            if row.get("condition") == condition
            and row.get("mode") == mode
            and row.get("ratio") == ratio
            and not row.get("skipped")
        })
    common = set.intersection(*valid_ids) if valid_ids else set()
    return [row for row in rows if row.get("qid") in common]


def merge_shards(args: argparse.Namespace) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for input_file in args.input_files:
        rows.extend(_read_jsonl(Path(input_file)))
    if args.output_file:
        _jsonl_write(args.output_file, rows)
    common_rows = _common_valid_rows(rows)
    summary = {
        "model": args.model,
        "base_model": args.base_model,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "separate": args.separate,
        "checkpoint_tool": args.checkpoint_tool if args.separate else None,
        "checkpoint_history": args.checkpoint_history if args.separate else None,
        "separate_generator": args.separate_generator if args.separate else None,
        "num_rows": len(rows),
        "results": _summarize(rows),
        "common_num_qids": len({row.get("qid") for row in common_rows}),
        "common_results": _summarize(common_rows),
    }
    if args.output_file:
        summary_path = str(Path(args.output_file).with_suffix(".summary.json"))
        Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Wrote merged predictions to %s", args.output_file)
        logger.info("Wrote merged summary to %s", summary_path)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate true-joint C2KV next-action checkpoints.")
    parser.add_argument("--model", help="Joint C2KV checkpoint path (dir or checkpoint-* parent).")
    parser.add_argument("--base_model", help="Base model path for full/truncate/c2kv_untrained baselines.")
    parser.add_argument("--tokenizer", help="Tokenizer path. Defaults to --base_model/--model.")
    parser.add_argument("--dataset_path", default="./datasets/agent-llm-traces")
    parser.add_argument("--output_file", default="./outputs/joint_next_action_eval.jsonl")
    parser.add_argument("--split", choices=["train", "eval"], default="eval")
    parser.add_argument("--condition", choices=list(CONDITIONS), default="joint", help=argparse.SUPPRESS)
    parser.add_argument("--conditions", default=None, help="Comma-separated doc-subset conditions.")
    parser.add_argument("--mode", choices=["c2kv", "c2kv_untrained", "truncate", "full"], default="c2kv")
    parser.add_argument("--compare_modes", default="c2kv,full")
    parser.add_argument("--ratios", default="8")
    parser.add_argument("--override_ratio", type=int, default=8)
    parser.add_argument(
        "--separate",
        action="store_true",
        help="J-separate arm: compress tool docs with --checkpoint_tool gist params and "
        "history docs with --checkpoint_history gist params, then generate with one model.",
    )
    parser.add_argument("--checkpoint_tool", help="Tool-side C2KV checkpoint for --separate.")
    parser.add_argument("--checkpoint_history", help="History-side C2KV checkpoint for --separate.")
    parser.add_argument("--separate_generator", choices=["tool", "history"], default="tool")
    parser.add_argument("--merge_only", action="store_true", help="Only merge --input_files shards.")
    parser.add_argument("--input_files", nargs="*", default=None)
    parser.add_argument("--max_examples", type=int, default=100, help="Maximum examples; <=0 means all.")
    parser.add_argument("--max_source_examples", type=int)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--split_manifest_file")
    parser.add_argument("--split_manifest_name", default="subset_disjoint")
    parser.add_argument("--max_samples_per_session", type=int, default=4)
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--max_input_chars", type=int)
    parser.add_argument("--max_answer_chars", type=int)
    parser.add_argument("--prefix_history_doc_num", type=int)
    parser.add_argument("--prefix_history_exact", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--canonical_format_prob", type=float, default=0.7)
    parser.add_argument("--minified_json_prob", type=float, default=0.2)
    parser.add_argument("--shuffle_tools", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--truncate_description_chars", type=int, default=600)
    parser.add_argument("--max_tools_per_sample", type=int, default=32)
    parser.add_argument("--same_namespace_negative_tools", type=int, default=8)
    parser.add_argument("--random_negative_tools", type=int, default=24)
    parser.add_argument("--max_doc_length", type=int, default=1024)
    parser.add_argument("--max_doc_num", type=int, default=24)
    parser.add_argument("--max_tool_chunks", type=int, default=None)
    parser.add_argument("--min_doc_num", type=int, default=1)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=32000)
    parser.add_argument("--max_system_length", type=int, default=512)
    parser.add_argument("--max_prompt_tokens", type=int, default=1920)
    parser.add_argument("--max_baseline_input_tokens", type=int, default=16000)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--history_selection", choices=["head", "tail"], default="tail")
    parser.add_argument("--split_oversized_history_docs", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--device_type", choices=["auto", "cuda", "npu", "cpu"], default="auto")
    parser.add_argument("--system_attn_impl", default="eager")
    parser.add_argument("--gist_attn_impl", default="eager")
    parser.add_argument("--generate_attn_impl", default="eager")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--baseline_model_class", choices=["gist", "auto"], default="gist")
    parser.add_argument("--untrained_c2kv", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.merge_only:
        if not args.input_files:
            parser.error("--merge_only requires --input_files")
        return args
    if not args.separate and not args.model:
        parser.error("--model is required unless --separate or --merge_only")
    return args


def main() -> None:
    args = parse_args()
    summary = merge_shards(args) if args.merge_only else evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
