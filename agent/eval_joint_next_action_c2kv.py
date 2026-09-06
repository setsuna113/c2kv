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
  (``max_doc_num - max_tool_chunks`` slots, tail-biased);
- ``tool_only``: tool chunks with the SAME ``max_tool_chunks`` cap as joint;
  history turns are NOT shown at all (absent);
- ``history_only``: history chunks with the same per-side budget as joint;
  tool schemas are absent.

Per-side budgets are constant across conditions (the G-Q3 fairness
constraint) and budget truncation always keeps the target tool's schema in
the grid; ``--legacy_mode_caps`` reproduces the pre-fix behavior (single-side
conditions got all ``max_doc_num`` slots, plain head-truncation could drop
the target schema) for diffing old runs.

Modes (per condition) mirror the existing evals: ``c2kv`` (checkpoint gist
params), ``c2kv_untrained`` (base-init gist params), ``full`` (all documents
in the plain prompt, no compression), ``truncate`` (plain prompt documents
head-truncated to ``ceil(doc_tokens / ratio)``, the unified eval's truncate
semantics).  Baseline modes load ``--base_model`` when given (history-eval
convention) with the gist config injected by ``_load_baseline_model`` — the
base model's config.json has no gist fields, so the custom model class
asserts without injection (modeling_qwen3.py:461); this mirrors the
``untrained_c2kv`` branch of the shared ``_load_model``, the only prior-eval
path that loads the plain base model.  With the frozen-base training recipe
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

History chunking (experiment B)
-------------------------------
``--chunk_policy {fixed-256,fixed-512,fixed-1024,agent-turn,structural}``
selects where the history side is cut, and ``--delay_recent_turns k`` holds
the last k turns out of the compressed grid and prepends them RAW to the
prompt.  All policies re-cut the SAME frozen content stream (the incumbent
``_fit_reused_history`` output plus turn provenance), so the arms are
content-matched by construction and ``agent-turn``/``delay_recent_turns=0``
short-circuits to today's exact call — the default pipeline is unchanged.
Both flags reach the doc builder through
``train_data_joint.build_history_chunks``, the same function the trainer
runs.  Per-row accounting for the pilot's budget checks lands in
``raw_recent_tokens`` / ``history_wrapped_tokens`` /
``history_content_tokens``; ``avg_gist_tokens`` deliberately keeps its old
meaning (compressed grid only).  ``--qid_manifest`` restricts the run to a
frozen qid list, in manifest order, and counts anything the source failed to
reproduce.  ``--do_sample/--temperature/--top_p/--gen_seed`` enable sampled
decoding with a per-row seed; the default stays greedy.

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
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import blend_gist_key_values, get_model_class  # noqa: E402
from eval_agent_tool_definition_c2kv import (  # noqa: E402
    _build_tool_cache,
    _generate_from_input_ids,
    _gist_compatible_config,
    _load_model,
    _load_safe_attn_impl,
    _prefill_system,
    _prefill_tokens_with_cache,
    _set_model_attn_impl,
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
from train.chunk_policy import CHUNK_POLICIES  # noqa: E402
from train.train_data_joint import (  # noqa: E402
    AgentLLMTracesJointSource,
    JointExample,
    TOOL_DOC_PREFIX,
    _default_max_tool_chunks,
    build_history_chunks,
    build_tool_chunks,
    cap_regime_name,
    regime_from_record,
)
from train.train_data_multiturn import (  # noqa: E402
    _chat_template_ids,
    _normal_chat_message,
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


def _wrap_history_message_ids(
    tokenizer: Any, message: Dict[str, Any], max_doc_length: int
) -> Tuple[List[int], bool]:
    """Chat-template-wrap one kept/delayed history message; flag a ceiling hit.

    The chunk policies size their windows so the wrap fits under
    ``max_doc_length`` (fixed-N keeps an 8-token re-encode margin), but that
    margin is a heuristic: BPE decode->re-encode drift beyond it would make
    ``_chat_template_ids`` truncate SILENTLY, breaking the frozen-content
    identity the B arms rest on.  A candidate hit (wrapped length at the
    ceiling) is confirmed against an unbounded second encode — normally there
    are zero candidates, so the extra encode costs nothing.
    """

    ids = _chat_template_ids(tokenizer, [message], max_length=max_doc_length)
    truncated = len(ids) >= max_doc_length and len(
        _chat_template_ids(tokenizer, [message])
    ) > len(ids)
    return ids, truncated


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
    per_side_caps: bool = True,
    chunk_policy: str = "agent-turn",
    delay_recent_turns: int = 0,
) -> Tuple[Optional[List[List[int]]], Optional[List[List[int]]], Optional[str], Dict[str, Any]]:
    """Build (tool_chunks, history_chunks) id lists for one condition.

    Each chunk is a chat-template-wrapped document of at most
    ``max_doc_length`` tokens (NOT padded).  Returns ``(None, None, reason,
    meta)`` when the tool side exceeds ``max_tool_definition_tokens`` (the
    training skip).  The tool side delegates to
    ``train_data_joint.build_tool_chunks`` and the history side to
    ``train_data_joint.build_history_chunks`` — the SAME code the trainer
    runs — so train and eval cannot drift.  ``meta`` carries the builder's
    ``target_known``/``target_in_grid`` flags for per-row logging plus the
    B-line chunking accounting (``raw_history_ids`` for the delayed docs,
    ``raw_recent_tokens`` / ``history_content_tokens`` /
    ``history_wrapped_tokens`` and the structural counters).

    The returned tuple arity is unchanged: delayed docs are NOT part of
    ``history_chunks`` (they never enter the compressed grid); the caller
    prepends ``meta["raw_history_ids"]`` to the plain prompt.
    """

    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition!r}")
    if max_tool_chunks is None:
        max_tool_chunks = _default_max_tool_chunks(max_doc_num)

    tool_chunks, skip_reason, tool_meta = build_tool_chunks(
        tokenizer,
        example,
        condition,
        max_doc_length=max_doc_length,
        max_doc_num=max_doc_num,
        max_tool_chunks=max_tool_chunks,
        max_tool_definition_tokens=max_tool_definition_tokens,
        per_side_caps=per_side_caps,
    )
    meta: Dict[str, Any] = dict(tool_meta)
    meta.update({
        "chunk_policy": chunk_policy,
        "delay_recent_turns": delay_recent_turns,
        "raw_history_ids": [],
        "raw_recent_tokens": 0,
        "history_content_tokens": 0,
        "history_wrapped_tokens": 0,
        "structural_fallback_docs": 0,
        "structural_partial_docs": 0,
        "delayed_docs": 0,
        "wrap_truncated_docs": 0,
    })
    if skip_reason is not None:
        return None, None, skip_reason, meta

    kept, delayed, history_meta = build_history_chunks(
        tokenizer,
        example,
        condition,
        max_doc_length=max_doc_length,
        max_doc_num=max_doc_num,
        max_tool_chunks=max_tool_chunks,
        num_tool_chunks=len(tool_chunks),
        per_side_caps=per_side_caps,
        history_selection=history_selection,
        split_oversized_history_docs=split_oversized_history_docs,
        chunk_policy=chunk_policy,
        delay_recent_turns=delay_recent_turns,
        # Eval-only: the presented-token check (analyze_b_pilot) and the gist
        # declaration both need the frozen-content token count.  The trainer
        # leaves this off — it is a second full encode of the history text.
        need_content_tokens=True,
    )
    wrap_truncated_docs = 0
    history_chunks: List[List[int]] = []
    for message in kept:
        ids, truncated = _wrap_history_message_ids(tokenizer, message, max_doc_length)
        wrap_truncated_docs += int(truncated)
        history_chunks.append(ids)
    raw_history_ids: List[int] = []
    for message in delayed:
        ids, truncated = _wrap_history_message_ids(tokenizer, message, max_doc_length)
        wrap_truncated_docs += int(truncated)
        raw_history_ids.extend(ids)
    meta.update(history_meta)
    meta.update({
        "raw_history_ids": raw_history_ids,
        "raw_recent_tokens": len(raw_history_ids),
        "history_content_tokens": history_meta.get("content_tokens", 0),
        "history_wrapped_tokens": sum(len(chunk) for chunk in history_chunks)
        + len(raw_history_ids),
        "wrap_truncated_docs": wrap_truncated_docs,
    })
    return tool_chunks, history_chunks, None, meta


def _chunk_meta_fields(meta: Dict[str, Any]) -> Dict[str, Any]:
    """B-line chunking accounting copied from the doc-chunk meta into a prefix.

    ``raw_history_ids`` are the delayed (uncompressed) history docs; every
    other field is per-row bookkeeping for the gist-declaration and
    presented-token checks in ``agent/analyze_b_pilot.py``.
    """

    return {
        "raw_history_ids": meta.get("raw_history_ids") or [],
        "raw_recent_tokens": meta.get("raw_recent_tokens", 0),
        "history_content_tokens": meta.get("history_content_tokens", 0),
        "history_wrapped_tokens": meta.get("history_wrapped_tokens", 0),
        "chunk_policy": meta.get("chunk_policy", "agent-turn"),
        "delay_recent_turns": meta.get("delay_recent_turns", 0),
        "structural_fallback_docs": meta.get("structural_fallback_docs", 0),
        "structural_partial_docs": meta.get("structural_partial_docs", 0),
        # Re-encode drift visibility: history docs whose chat-template wrap hit
        # the max_doc_length ceiling (0 normally — any non-zero value means the
        # policy's window margin was insufficient and content was cut).
        "wrap_truncated_docs": meta.get("wrap_truncated_docs", 0),
    }


def _doc_grid(chunks: Sequence[List[int]], max_doc_length: int) -> torch.Tensor:
    rows = [_pad(chunk, max_doc_length, -100) for chunk in chunks]
    return torch.tensor(rows, dtype=torch.long)


def _flat_doc_ids(tool_chunks: Sequence[List[int]], history_chunks: Sequence[List[int]]) -> List[int]:
    return [token for chunk in [*tool_chunks, *history_chunks] for token in chunk]


def _current_prompt_ids(
    tokenizer: Any,
    example: JointExample,
    max_prompt_tokens: Optional[int] = None,
) -> List[int]:
    """Current-turn prompt ids, normalized exactly as JointDataset does
    (train_data_joint.py: messages with empty content dropped unless
    assistant, ``tool`` role mapped to ``user``, non-string content JSON
    dumped), then tail-truncated to ``max_prompt_tokens``.
    """

    current = [
        _normal_chat_message(message)
        for message in example.current_messages
        if message.get("content") or message.get("role") == "assistant"
    ]
    prompt_ids = _chat_template_ids(tokenizer, current, add_generation_prompt=True)
    if max_prompt_tokens and len(prompt_ids) > max_prompt_tokens:
        prompt_ids = prompt_ids[-max_prompt_tokens:]
    return prompt_ids


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
# Baseline model loading (full / truncate / c2kv_untrained).
# ---------------------------------------------------------------------------


def _load_baseline_model(args: argparse.Namespace, tokenizer: Any, device: str) -> Any:
    """Load the BASE model for full/truncate/c2kv_untrained, gist config injected.

    The plain base model's config.json carries no gist fields, so the repo's
    custom Qwen3 class asserts ``config.gist_token_id is not None``
    (modeling_qwen3.py:461) when instantiated without config injection.
    Training injects the gist fields in
    ``model_utils.get_model_and_tokenizer`` (model_utils.py:151-201: all
    ``gist*`` ModelArgs + ``gist_token_id = tokenizer.eos_token_id``); the
    eval-side equivalent is ``_gist_compatible_config`` — the
    ``untrained_c2kv`` branch of ``_load_model`` is the ONLY prior-eval path
    that loads the plain base model.  Prior evals never hit this for
    full/truncate: the history eval uses ``baseline_model_class="auto"``
    (plain ``AutoModelForCausalLM``) and the tooldef/unified evals load the
    checkpoint (whose config.json has gist fields).  This driver switches
    baselines to ``--base_model`` (history-eval convention), so it must
    inject — this function mirrors the proven ``untrained_c2kv`` branch for
    all baseline modes.  Values follow the training recipe
    (dynamic-interleave, qkv, embed-mean, overlap 64, gist_token_id=eos);
    for full/truncate the gist params are inert anyway (generation uses
    ``use_gist=False``).
    """

    model_path = args.base_model or args.model
    config_class, model_class = get_model_class(model_path, "qkv")
    config = _gist_compatible_config(config_class, model_path, tokenizer)
    model = model_class.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
        device_map={"": device} if device != "cpu" else None,
        dtype=(
            torch.bfloat16 if args.dtype == "bf16"
            else torch.float16 if args.dtype == "fp16"
            else torch.float32
        ),
        attn_implementation=_load_safe_attn_impl(args.generate_attn_impl),
    )
    _set_model_attn_impl(model, args.generate_attn_impl)
    model.eval()
    return model


def _gist_params_at_init_fraction(model: Any) -> float:
    """Fraction of layers whose gist projections EXACTLY equal the base ones.

    ``init_gist_proj``/``init_gist_embed`` (gist_utils.py:819-868) copy base
    q/k/v (and the eos embedding) into the gist params for every key MISSING
    from the loaded checkpoint.  A loaded model whose gist params all exactly
    equal the base projections is therefore running with UNTRAINED gist
    params — either the checkpoint never trained them or (silently) they did
    not load.  The c2kv arm then degenerates to c2kv_untrained: scores
    collapse to base-model fallback text while full/truncate (which never
    touch gist params) stay fine.  A trained checkpoint has drifted away from
    the copies, so the fraction is < 1 (typically 0).
    """

    equal_layers = 0
    total_layers = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        total_layers += 1
        if (
            torch.equal(attn.gist_q_proj.weight, attn.q_proj.weight)
            and torch.equal(attn.gist_k_proj.weight, attn.k_proj.weight)
            and torch.equal(attn.gist_v_proj.weight, attn.v_proj.weight)
        ):
            equal_layers += 1
    return equal_layers / total_layers if total_layers else 0.0


def _log_gist_init_check(model: Any, model_path: str, mode: str) -> float:
    fraction = _gist_params_at_init_fraction(model)
    if fraction > 0:
        logger.warning(
            "GIST PARAMS AT INIT (== base projections) in %.0f%% of layers for model=%s mode=%s. "
            "If this checkpoint was trained, its gist weights did NOT load (or it is an early/at-init "
            "checkpoint): c2kv degenerates to untrained-compressor behavior. Check transformers "
            "'missing keys' load warnings and the checkpoint's safetensors for gist_* tensors.",
            fraction * 100,
            model_path,
            mode,
        )
    else:
        logger.info("gist params differ from base projections (trained) for model=%s mode=%s", model_path, mode)
    return fraction


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
    tool_chunks, history_chunks, skip_reason, tool_meta = _condition_doc_chunks(
        tokenizer,
        example,
        args.condition,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_chunks=args.max_tool_chunks,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        history_selection=args.history_selection,
        split_oversized_history_docs=args.split_oversized_history_docs,
        per_side_caps=not args.legacy_mode_caps,
        chunk_policy=getattr(args, "chunk_policy", "agent-turn"),
        delay_recent_turns=getattr(args, "delay_recent_turns", 0),
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
        "target_known": tool_meta.get("target_known"),
        "target_in_grid": tool_meta.get("target_in_grid"),
        "target_truncated_to_cap": tool_meta.get("target_truncated_to_cap"),
        **_chunk_meta_fields(tool_meta),
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

    tool_chunks, history_chunks, skip_reason, tool_meta = _condition_doc_chunks(
        tokenizer,
        example,
        "joint",
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_chunks=args.max_tool_chunks,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        history_selection=args.history_selection,
        split_oversized_history_docs=args.split_oversized_history_docs,
        per_side_caps=not args.legacy_mode_caps,
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
        "target_known": tool_meta.get("target_known"),
        "target_in_grid": tool_meta.get("target_in_grid"),
        "target_truncated_to_cap": tool_meta.get("target_truncated_to_cap"),
        **_chunk_meta_fields(tool_meta),
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

    ``delay_recent_turns`` is forced to 0 here: full/truncate present every
    document in the plain prompt already, so holding turns back would only
    reorder the same raw tokens.
    """

    tool_chunks, history_chunks, skip_reason, tool_meta = _condition_doc_chunks(
        tokenizer,
        example,
        args.condition,
        max_doc_length=args.max_doc_length,
        max_doc_num=args.max_doc_num,
        max_tool_chunks=args.max_tool_chunks,
        max_tool_definition_tokens=args.max_tool_definition_tokens,
        history_selection=args.history_selection,
        split_oversized_history_docs=args.split_oversized_history_docs,
        per_side_caps=not args.legacy_mode_caps,
        chunk_policy=getattr(args, "chunk_policy", "agent-turn"),
        delay_recent_turns=0,
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

    prompt_ids = _current_prompt_ids(tokenizer, example, args.max_prompt_tokens)
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
        "target_known": tool_meta.get("target_known"),
        "target_in_grid": tool_meta.get("target_in_grid"),
        "target_truncated_to_cap": tool_meta.get("target_truncated_to_cap"),
        **_chunk_meta_fields(tool_meta),
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


def _seed_row(args: argparse.Namespace, example: JointExample, mode: str) -> None:
    """Per-row generation seed (shared formula with eval_agent_history_c2kv)."""
    seed = (int(getattr(args, "gen_seed", 0)) * 1_000_003) ^ zlib.crc32(
        f"{example.qid}:{mode}:{args.override_ratio}".encode()
    )
    torch.manual_seed(seed)


@torch.inference_mode()
def _generate_with_prefix(
    model: Any,
    tokenizer: Any,
    example: JointExample,
    prefix: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    # Delayed history docs stay uncompressed: they ride in front of the
    # current turn in the plain prompt.  position_ids already start at
    # system_length + doc_length, and doc_length counts ONLY the original
    # tokens of the compressed grid, so the raw turn naturally occupies the
    # positions right after it — no extra bookkeeping needed.
    raw_ids = prefix.get("raw_history_ids") or []
    prompt_ids = _current_prompt_ids(tokenizer, example, args.max_prompt_tokens)
    prompt_input_ids = torch.tensor(
        [list(raw_ids) + prompt_ids], dtype=torch.long, device=model.device
    )
    mock_cache_ids = prompt_input_ids.new_zeros((1, prefix["cache_length"]))
    input_ids = torch.cat([mock_cache_ids, prompt_input_ids], dim=1)
    original_prefix_length = prefix["system_length"] + prefix["doc_length"]
    position_ids = torch.arange(
        original_prefix_length,
        original_prefix_length + prompt_input_ids.shape[1],
        dtype=torch.long,
        device=model.device,
    ).unsqueeze(0)
    do_sample = bool(getattr(args, "do_sample", False))
    if do_sample:
        _seed_row(args, example, getattr(args, "row_mode", args.mode))
    prediction, generate_sec, generated_tokens, tbt_sec = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        attn_impl=args.generate_attn_impl,
        use_gist=prefix["use_gist"],
        position_ids=position_ids,
        past_key_values=prefix["cache"],
        do_sample=do_sample,
        temperature=getattr(args, "temperature", None),
        top_p=getattr(args, "top_p", None),
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
        "target_known": prefix.get("target_known"),
        "target_in_grid": prefix.get("target_in_grid"),
        "target_truncated_to_cap": prefix.get("target_truncated_to_cap"),
        "gist_tokens": prefix["gist_tokens"],
        "compressed_tokens": prefix["compressed_tokens"],
        "prompt_tokens": len(prompt_ids),
        "raw_recent_tokens": prefix.get("raw_recent_tokens", 0),
        "history_wrapped_tokens": prefix.get("history_wrapped_tokens", 0),
        "history_content_tokens": prefix.get("history_content_tokens", 0),
        "chunk_policy": prefix.get("chunk_policy", "agent-turn"),
        "delay_recent_turns": prefix.get("delay_recent_turns", 0),
        "structural_fallback_docs": prefix.get("structural_fallback_docs", 0),
        "structural_partial_docs": prefix.get("structural_partial_docs", 0),
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

        # Rows merged from heterogeneous shards (different code versions, or a
        # partially-written shard) can be missing a metric key outright.  A
        # missing key is folded in as falsy/0.0 so `merge_shards` cannot die at
        # the last step of a long run, but it is NOT the same thing as a
        # measured zero: every occurrence is counted per field and surfaced on
        # the summary as `missing_metric_fields`, so a deflated rate can never
        # be read as a real one.
        missing: "Counter[str]" = Counter()

        def _count_missing(field: str) -> None:
            absent = sum(1 for row in valid if field not in row)
            if absent:
                missing[field] += absent

        def _rate(field: str) -> float:
            _count_missing(field)
            return sum(1 for row in valid if row.get(field)) / len(valid) if valid else 0.0

        def _avg(field: str) -> float:
            _count_missing(field)
            return (
                sum(row.get(field) or 0.0 for row in valid) / len(valid) if valid else 0.0
            )

        entry = {
            "condition": condition,
            "mode": mode,
            "ratio": ratio,
            "num_examples": len(group),
            "num_valid": len(valid),
            "num_skipped": len(group) - len(valid),
            "skip_reasons": dict(skips),
            "exact_match": _rate("exact_match"),
            "tool_name_accuracy": _rate("tool_name_match"),
            "tool_call_rate": _rate("has_tool_call"),
            "response_type_accuracy": _rate("response_type_match"),
            "argument_name_f1": _avg("argument_name_f1"),
            "argument_value_f1": _avg("argument_value_f1"),
            "avg_text_token_f1": _avg("text_token_f1"),
            "avg_rouge_l_f1": _avg("rouge_l_f1"),
            "avg_doc_tokens": _avg("doc_tokens"),
            "avg_doc_chunks": _avg("doc_chunks"),
            # avg_gist_tokens keeps its original meaning (compressed grid
            # only): the delayed raw turn is reported separately as
            # avg_raw_recent_tokens and NEVER folded in, or the 5% gist
            # declaration check (判据1) would read a delay arm as inflated.
            "avg_gist_tokens": _avg("gist_tokens"),
            "avg_compressed_tokens": _avg("compressed_tokens"),
            "avg_prompt_tokens": _avg("prompt_tokens"),
            "avg_raw_recent_tokens": _avg("raw_recent_tokens"),
            "avg_history_wrapped_tokens": _avg("history_wrapped_tokens"),
            "avg_history_content_tokens": _avg("history_content_tokens"),
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
        }
        # Never silent: a rate computed over rows that did not carry the field
        # is deflated, and the reader has to be able to see that.
        entry["missing_metric_fields"] = dict(sorted(missing.items()))
        if missing:
            logger.warning(
                "condition=%s mode=%s ratio=%s: %d/%d valid rows are missing metric keys %s — "
                "those rows were folded in as 0 and the affected rates are DEFLATED; "
                "this usually means shards from different code versions were merged",
                condition,
                mode,
                ratio,
                max(missing.values()),
                len(valid),
                dict(sorted(missing.items())),
            )
        summaries.append(entry)
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


def _load_qid_manifest(path: str) -> List[str]:
    """Frozen qid list: a JSON list, a JSON object with ``qids``, or one per line."""
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("[") or stripped.startswith("{"):
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("qids")
        if not isinstance(payload, list):
            raise ValueError(f"{path}: expected a JSON list of qids (or a 'qids' key)")
        return [str(item) for item in payload]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _filter_by_manifest(
    examples: List[JointExample], manifest_path: str
) -> Tuple[List[JointExample], List[str]]:
    """Keep only manifest qids, IN MANIFEST ORDER; report the missing ones.

    A missing qid is never silent: the caller logs a warning and the count
    lands in the run summary, because a shrunken frozen set breaks the paired
    contrast the whole B analysis rests on.
    """

    qids = _load_qid_manifest(manifest_path)
    by_qid: Dict[str, JointExample] = {}
    for example in examples:
        by_qid.setdefault(example.qid, example)
    ordered = [by_qid[qid] for qid in qids if qid in by_qid]
    missing = [qid for qid in qids if qid not in by_qid]
    return ordered, missing


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    device = _setup_device(args.device_type)
    if args.model:
        args.model = _resolve_model_checkpoint(args.model)
    tokenizer = _load_tokenizer(args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    examples = _load_examples(args)
    logger.info("Loaded %d joint %s examples", len(examples), args.split)
    manifest_missing: List[str] = []
    if getattr(args, "qid_manifest", None):
        examples, manifest_missing = _filter_by_manifest(examples, args.qid_manifest)
        if manifest_missing:
            logger.warning(
                "qid manifest %s: %d/%d qids not reproduced by the source (first 5: %s) — "
                "the frozen paired set is INCOMPLETE",
                args.qid_manifest,
                len(manifest_missing),
                len(manifest_missing) + len(examples),
                manifest_missing[:5],
            )
        logger.info("qid manifest %s: %d examples kept", args.qid_manifest, len(examples))
    ratios = [int(item) for item in _parse_csv(args.ratios, str(args.override_ratio))]
    rows: List[Dict[str, Any]] = []
    gist_init_fractions: Dict[str, float] = {}

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
            gist_init_fractions[f"separate_{name}"] = _log_gist_init_check(models[name], checkpoint, SEPARATE_MODE)
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
            model_args.mode = "c2kv" if mode == "c2kv_untrained" else mode
            model_args.row_mode = mode
            if mode == "c2kv":
                # Trained joint checkpoint: its config.json carries the gist
                # fields, so the shared _load_model path works as-is.
                logger.info("Loading model for mode=%s model=%s", mode, model_args.model)
                model = _load_model(model_args, tokenizer, device)
                gist_init_fractions[mode] = _log_gist_init_check(model, model_args.model, mode)
            else:
                # Baseline modes run on the base model (history-eval
                # convention); its config.json has NO gist fields, so the
                # custom class needs the gist config injected — see
                # _load_baseline_model.  Output-equivalent to the joint
                # checkpoint's (frozen) base weights for full/truncate.
                if mode == "c2kv_untrained" and not args.base_model:
                    raise ValueError("--base_model is required for c2kv_untrained baseline")
                logger.info("Loading base model for mode=%s model=%s", mode, model_args.base_model or model_args.model)
                model = _load_baseline_model(model_args, tokenizer, device)
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
        "gist_init_fractions": gist_init_fractions,
        "conditions": conditions,
        "modes": modes,
        "ratios": ratios,
        "history_selection": args.history_selection,
        "max_doc_length": args.max_doc_length,
        "max_doc_num": args.max_doc_num,
        "max_tool_chunks": args.max_tool_chunks,
        "legacy_mode_caps": args.legacy_mode_caps,
        "cap_regime": cap_regime_name(args.legacy_mode_caps),
        "min_doc_num": args.min_doc_num,
        "max_tool_definition_tokens": args.max_tool_definition_tokens,
        "max_system_length": args.max_system_length,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_baseline_input_tokens": args.max_baseline_input_tokens,
        "chunk_policy": args.chunk_policy,
        "delay_recent_turns": args.delay_recent_turns,
        "qid_manifest": args.qid_manifest,
        "qid_manifest_missing": len(manifest_missing),
        "num_examples": len(examples),
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gen_seed": args.gen_seed,
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
    gist_init_fractions: Dict[str, float] = {}
    shard_cap_modes: List[Any] = []
    shard_chunk_policies: List[Any] = []
    shard_qid_manifests: List[str] = []
    qid_manifest_missing_total = 0
    shard_cap_regimes: List[str] = []
    for input_file in args.input_files:
        rows.extend(_read_jsonl(Path(input_file)))
        # Aggregate the shard summaries' gist-init diagnostics (worst case per
        # key) so the gate-1 pick guard sees them on the MERGED summary too —
        # without this the guard was dead code on the parallel-shard path.
        shard_summary_path = Path(input_file).with_suffix(".summary.json")
        if shard_summary_path.exists():
            try:
                shard_summary = json.loads(shard_summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Unreadable shard summary skipped: %s", shard_summary_path)
                shard_summary = {}
            for key, value in (shard_summary.get("gist_init_fractions") or {}).items():
                if isinstance(value, (int, float)):
                    previous = gist_init_fractions.get(key)
                    gist_init_fractions[key] = (
                        float(value) if previous is None else max(previous, float(value))
                    )
            if "legacy_mode_caps" in shard_summary:
                shard_cap_modes.append(bool(shard_summary["legacy_mode_caps"]))
            if "chunk_policy" in shard_summary:
                shard_chunk_policies.append(str(shard_summary["chunk_policy"]))
            if shard_summary.get("qid_manifest") is not None:
                shard_qid_manifests.append(str(shard_summary["qid_manifest"]))
            missing = shard_summary.get("qid_manifest_missing")
            if isinstance(missing, (int, float)) and not isinstance(missing, bool):
                qid_manifest_missing_total += int(missing)
            if "legacy_mode_caps" in shard_summary or "cap_regime" in shard_summary:
                shard_cap_modes.append(bool(shard_summary.get("legacy_mode_caps")))
                # Normalized regime string: pre-string summaries map to
                # legacy/per_side_caps(v1) via the boolean, so a v1 shard and
                # a v2 (empty-tool-reclaim) shard are told apart here even
                # though both carry legacy_mode_caps=False.
                shard_cap_regimes.append(
                    regime_from_record(
                        shard_summary.get("legacy_mode_caps"), shard_summary.get("cap_regime")
                    )
                )
    # The doc-budget regime must be visible on the merged summary: legacy and
    # fixed caps produce non-comparable numbers, and mixing shards from both
    # regimes in one merge is almost certainly an ops mistake.
    cap_modes = sorted(set(shard_cap_modes))
    if len(cap_modes) > 1:
        logger.warning(
            "MERGING SHARDS FROM DIFFERENT DOC-BUDGET REGIMES (legacy_mode_caps=%s) — "
            "the merged numbers are not internally comparable",
            cap_modes,
        )
    cap_regimes = sorted(set(shard_cap_regimes))
    if len(cap_regimes) > 1:
        logger.warning(
            "MERGING SHARDS FROM DIFFERENT DOC-BUDGET REGIMES (cap_regime=%s) — "
            "the merged numbers are not internally comparable",
            cap_regimes,
        )
    merged_legacy_mode_caps: Any = cap_modes[0] if len(cap_modes) == 1 else (cap_modes or None)
    # Same treatment for the chunking policy: merging arms into one file
    # destroys the per-arm contrast the B analysis needs, so it is loud.
    chunk_policies = sorted(set(shard_chunk_policies))
    if len(chunk_policies) > 1:
        logger.warning(
            "MERGING SHARDS FROM DIFFERENT CHUNK POLICIES (%s) — these are separate B arms "
            "and must not be pooled into one summary",
            chunk_policies,
        )
    merged_chunk_policy: Any = (
        chunk_policies[0] if len(chunk_policies) == 1 else (chunk_policies or None)
    )
    # The frozen-set bookkeeping must survive the merge: b_prereg.md §2 gates
    # every paired table on qid_manifest_missing == 0, and the merged summary
    # is the artefact the B run script prints and the operator reads — losing
    # the count here made the gate invisible on the parallel-shard path.
    distinct_manifests = sorted(set(shard_qid_manifests))
    if len(distinct_manifests) > 1:
        logger.warning(
            "MERGING SHARDS RUN AGAINST DIFFERENT QID MANIFESTS (%s) — "
            "the frozen paired set is not well-defined for this merge",
            distinct_manifests,
        )
    merged_qid_manifest = shard_qid_manifests[0] if shard_qid_manifests else None
    if qid_manifest_missing_total > 0:
        logger.warning(
            "qid_manifest_missing=%d across the merged shards — the frozen paired "
            "set is INCOMPLETE and this round must not enter any paired table "
            "(b_prereg.md §2)",
            qid_manifest_missing_total,
        )
    merged_cap_regime: Any = cap_regimes[0] if len(cap_regimes) == 1 else (cap_regimes or None)
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
        "legacy_mode_caps": merged_legacy_mode_caps,
        "chunk_policy": merged_chunk_policy,
        "qid_manifest": merged_qid_manifest,
        "qid_manifest_missing": qid_manifest_missing_total,
        "cap_regime": merged_cap_regime,
        "gist_init_fractions": gist_init_fractions,
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
    parser.add_argument("--require_tool_call", type=lambda x: str(x).lower() == "true", default=True)
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
    parser.add_argument(
        "--legacy_mode_caps",
        action="store_true",
        help="Reproduce the pre-fix doc budgets (single-side conditions get all max_doc_num "
        "slots; plain head-truncation may drop the target tool schema). For diffing old runs only.",
    )
    parser.add_argument("--min_doc_num", type=int, default=2)
    parser.add_argument("--max_tool_definition_tokens", type=int, default=32000)
    parser.add_argument("--max_system_length", type=int, default=512)
    parser.add_argument("--max_prompt_tokens", type=int, default=1920)
    parser.add_argument("--max_baseline_input_tokens", type=int, default=16000)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--history_selection", choices=["head", "tail"], default="tail")
    parser.add_argument("--split_oversized_history_docs", type=lambda x: str(x).lower() == "true", default=True)
    # ---- experiment B: chunking policy / delayed compression / sampling ----
    parser.add_argument(
        "--chunk_policy",
        choices=list(CHUNK_POLICIES),
        default="agent-turn",
        help="History-side chunk boundaries. agent-turn is the incumbent (in-distribution "
        "reference); fixed-* ignore turn boundaries; structural cuts at atomic "
        "action+observation blocks. All arms share the same frozen content stream.",
    )
    parser.add_argument(
        "--delay_recent_turns",
        type=int,
        default=0,
        help="Hold the last k turns out of the compressed grid and prepend them raw to the "
        "prompt (turn granularity). Not defined for the fixed-* policies.",
    )
    parser.add_argument(
        "--qid_manifest",
        help="Frozen qid list (JSON list, {'qids': [...]} or one qid per line). Examples are "
        "filtered to it and kept in manifest order; missing qids are counted and warned.",
    )
    parser.add_argument("--do_sample", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. MANDATORY when --do_sample true: leaving it unset would "
        "silently inherit the checkpoint's generation_config while the run summary records "
        "null, so the decode configuration could not be recovered from the artefacts.",
    )
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument(
        "--gen_seed",
        type=int,
        default=0,
        help="Per-row generation seed base; only used when --do_sample true.",
    )
    parser.add_argument("--device_type", choices=["auto", "cuda", "npu", "cpu"], default="auto")
    parser.add_argument("--system_attn_impl", default="eager")
    parser.add_argument("--gist_attn_impl", default="eager")
    parser.add_argument("--generate_attn_impl", default="eager")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--untrained_c2kv", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.merge_only:
        if not args.input_files:
            parser.error("--merge_only requires --input_files")
        return args
    if not args.separate and not args.model:
        parser.error("--model is required unless --separate or --merge_only")
    if args.delay_recent_turns < 0:
        parser.error("--delay_recent_turns must be non-negative")
    if args.chunk_policy.startswith("fixed-") and args.delay_recent_turns > 0:
        parser.error(
            f"--chunk_policy {args.chunk_policy} destroys turn boundaries; "
            "--delay_recent_turns > 0 needs agent-turn or structural"
        )
    # Traceability, not taste: with do_sample=True and no explicit temperature,
    # transformers silently falls back to the checkpoint's generation_config,
    # while the run summary records "temperature": null.  The decode
    # configuration of the run would then be unrecoverable from its artefacts.
    if args.do_sample and args.temperature is None:
        parser.error(
            "--do_sample true requires an explicit --temperature: otherwise the sampling "
            "temperature comes from the checkpoint's generation_config and the run summary "
            "records temperature=null, making the decode configuration untraceable. "
            "Pass --temperature (and --top_p) explicitly, or run greedy with --do_sample false."
        )
    return args


def main() -> None:
    args = parse_args()
    summary = merge_shards(args) if args.merge_only else evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
