"""D0 anchor arms + D1 raw sidecar oracle repair arms.

Contract (D0, 2026-08-30):
- P_k captured during normal compression via SidecarStore (zero extra forward)
- repair: oracle(k*) -> load/decode(P_k*) -> edit -> query/decode
- NO forward of any already-seen history token
- strict metric: tool_name + exact arguments
- timing: T_capture / T_load / T_edit / T_query + warm repair latency

D0 anchors (with c2kv baseline):
  wrongblock_sidecar_sham  inject wrong block's sidecar payload (same bytes/layout)
  oracle_target_only       store only P_{k*} (operator headroom)
  allblock_sidecar         store all P_k, oracle selects (full cold-storage bytes)

D1 raw sidecar oracle repair (all use sidecar R_k^local, no scratch prefill):
  raw_keepG@k      keep all gists + append R_k (splice_keep layout, sidecar source)
  raw_replaceG@k   remove G_k, place R_k at original logical position (splice_rep layout)
  raw_erratum_tail keep G_k, RoPE-reanchor R_k at repair tail
  wrongblock_raw_sham  same bytes/layout as raw_keepG but wrong block's KV
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import torch

import eval_agent_history_c2kv as HH
from d0_sidecar import SidecarStore

logger = logging.getLogger(__name__)

ARM_MODES = {
    # D0 anchors
    "wrongblock_sidecar_sham": "d_wrongblock_sidecar",
    "oracle_target_only": "d_oracle_target_only",
    "allblock_sidecar": "d_allblock_sidecar",
    # D1 raw sidecar oracle repair
    "raw_keepG": "d_raw_keepG",
    "raw_replaceG": "d_raw_replaceG",
    "raw_erratum_tail": "d_raw_erratum_tail",
    "wrongblock_raw_sham": "d_wrongblock_raw",
}


@torch.inference_mode()
def _sidecar_raw_span(
    store: SidecarStore,
    qid: str,
    doc_index: int,
    logical_start: int,
    rope_theta: float,
    rope_type: Optional[str],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Return per-layer (keys, values) for one doc, K rotated to logical_start.

    The sidecar stores pre-RoPE K at local positions 0..L; this function
    applies the RoPE rotation to place K at absolute position
    ``logical_start`` (the doc's original logical offset in the conversation).
    V needs no rotation.
    """
    from inference.rope_reposition import rotate_k_cache_rope

    keys = store.get(qid, doc_index, "k")   # List[layer] of (kv_heads, L, D)
    values = store.get(qid, doc_index, "v")
    span = []
    for k, v in zip(keys, values):
        # rotate_k_cache_rope expects (heads, seq, dim); stored keys are
        # (kv_heads, L, D) — exact match
        rotated = rotate_k_cache_rope(k, logical_start, rope_theta, rope_type)
        span.append((rotated.unsqueeze(0), v.unsqueeze(0)))  # add batch dim (1, H, L, D)
    return span


@torch.inference_mode()
def _cat_span_to_cache(
    cache: Any,
    span_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    insert_at: Optional[int] = None,
) -> Any:
    """Append per-layer (K,V) span to cache. If insert_at is given, insert
    at that physical cache position (for in-place layouts)."""
    if insert_at is None:
        for layer, (keys, values) in zip(cache.layers, span_kv):
            layer.keys = torch.cat([layer.keys, keys], dim=-2)
            layer.values = torch.cat([layer.values, values], dim=-2)
    else:
        for layer, (keys, values) in zip(cache.layers, span_kv):
            layer.keys = torch.cat(
                [layer.keys[..., :insert_at, :], keys, layer.keys[..., insert_at:, :]], dim=-2
            )
            layer.values = torch.cat(
                [layer.values[..., :insert_at, :], values, layer.values[..., insert_at:, :]], dim=-2
            )
    return cache


@torch.inference_mode()
def build_d_contract_prefix(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    mode: str,
    store: SidecarStore,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build a D-contract prefix using sidecar payload (no history forward)."""

    context_input_ids, doc_tokens, doc_chunks, history, skip_reason = HH._build_history_chunks(
        tokenizer, example, args
    )
    if context_input_ids is None:
        return None, skip_reason
    doc_ids = [
        HH._chat_template_ids(tokenizer, [m], max_length=args.max_doc_length)
        for m in history
    ]
    n_docs = len(doc_ids)
    if n_docs == 0:
        return None, "d_no_history_docs"
    k_star = (n_docs - 1) // 2

    system_ids = HH._chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long, device=model.device)
    system_cache, system_length, system_prefill_sec = HH._prefill_system(
        model, system_input_ids, args.system_attn_impl
    )
    offsets: List[int] = []
    offset = system_length
    for ids in doc_ids:
        offsets.append(offset)
        offset += len(ids)
    doc_logical_start = offsets[k_star]
    assert doc_logical_start > 0

    # --- compression with sidecar capture (the ONLY forward) ---
    grid = context_input_ids
    valid_mask = grid != -100
    doc_lengths = [int(v.sum().item()) for v in valid_mask]
    capture_store = store

    def compress_call():
        ids = grid.clone().to(model.device)
        ids[~valid_mask] = model.model.gist_token_id
        gist_kwargs = {}
        if getattr(model.config, "gist_type", None) == "dynamic-interleave":
            gist_kwargs["ratio"] = args.override_ratio
        return model.model.generate_gist(
            input_ids=ids, attention_mask=valid_mask.to(model.device), **gist_kwargs
        )

    t_capture_start = time.perf_counter()
    outputs = capture_store.capture(example.qid, compress_call, doc_lengths)
    t_capture = capture_store.last_capture_sec

    # blend gists onto system cache (standard path)
    from models.gist_utils import blend_gist_key_values
    gist_mask = outputs[1]
    pos_ids = outputs[2]
    gist_len = int(gist_mask.shape[-1])
    pos_ids = pos_ids[:, -gist_len:]
    prefix_cache, _ = blend_gist_key_values(
        model.config, [outputs[0].past_key_values], [gist_mask],
        [pos_ids], model.model.rotary_emb, system_length,
    )
    # cat system layers
    for sys_layer, gist_layer in zip(system_cache.layers, prefix_cache.layers):
        gist_layer.keys = torch.cat([sys_layer.keys, gist_layer.keys], dim=-2)
        gist_layer.values = torch.cat([sys_layer.values, gist_layer.values], dim=-2)
    gist_tokens = prefix_cache.get_seq_length() - system_length

    # --- arm-specific repair (NO history forward) ---
    rope_theta = getattr(model.config, "rope_theta", 1000000.0)
    rope_type = getattr(getattr(model.config, "rope_scaling", None), "rope_type", None)

    t_load_start = time.perf_counter()
    sidecar_bytes_all = capture_store.bytes_of(example.qid)
    sidecar_bytes_target = capture_store.bytes_of(example.qid, [k_star])

    d_mode_info = {"sidecar_bytes_all": sidecar_bytes_all,
                   "sidecar_bytes_target": sidecar_bytes_target,
                   "t_capture_sec": round(t_capture, 4)}

    if mode == "d_wrongblock_sidecar":
        # wrong block: same bytes, same layout, wrong block's KV
        wrong_k = (k_star + n_docs // 2) % n_docs
        span = _sidecar_raw_span(capture_store, example.qid, wrong_k,
                                 offsets[wrong_k], rope_theta, rope_type)
        prefix_cache = _cat_span_to_cache(prefix_cache, span)
        d_mode_info["wrong_doc_index"] = wrong_k
        d_mode_info["t_load_sec"] = round(time.perf_counter() - t_load_start, 4)
        d_mode_info["sidecar_bytes_used"] = capture_store.bytes_of(example.qid, [wrong_k])

    elif mode == "d_oracle_target_only":
        # store only k*: operator headroom (payload available, just measure)
        capture_store.drop_docs(example.qid, [k_star])
        d_mode_info["t_load_sec"] = round(time.perf_counter() - t_load_start, 4)
        d_mode_info["sidecar_bytes_used"] = sidecar_bytes_target
        d_mode_info["note"] = "no injection; measures compression+storage only"

    elif mode == "d_allblock_sidecar":
        # all P_k stored, oracle selects k*: same injection as raw_keepG but
        # report full cold-storage bytes
        span = _sidecar_raw_span(capture_store, example.qid, k_star,
                                 doc_logical_start, rope_theta, rope_type)
        prefix_cache = _cat_span_to_cache(prefix_cache, span)
        d_mode_info["t_load_sec"] = round(time.perf_counter() - t_load_start, 4)
        d_mode_info["sidecar_bytes_used"] = sidecar_bytes_all

    elif mode == "d_raw_keepG":
        # keep all gists + append R_k at end (splice_keep layout)
        span = _sidecar_raw_span(capture_store, example.qid, k_star,
                                 doc_logical_start, rope_theta, rope_type)
        prefix_cache = _cat_span_to_cache(prefix_cache, span)
        d_mode_info["t_load_sec"] = round(time.perf_counter() - t_load_start, 4)
        d_mode_info["sidecar_bytes_used"] = sidecar_bytes_target

    elif mode == "d_raw_replaceG":
        # remove G_k, place R_k at original logical position
        # physical position of G_k in cache: system_length + gist tokens of docs 0..k*-1
        # (right-padded grid gists are concatenated in doc order by blend)
        # Approximate: rebuild without doc k* in grid, then insert R_k
        # More correct: compute gist prefix length for docs 0..k*-1
        # For now: build with left docs only, insert raw, then right gists
        # (matches d_splice_rep layout from B1)
        left_docs = doc_ids[:k_star]
        right_docs = doc_ids[k_star + 1:]
        # rebuild: left gists -> R_k -> right gists
        left_grid = HH._grid_from_doc_ids(left_docs, args.max_doc_length, args.max_doc_num)
        left_valid = left_grid != -100
        left_lengths = [int(v.sum().item()) for v in left_valid]

        def left_compress():
            ids = left_grid.clone().to(model.device)
            ids[~left_valid] = model.model.gist_token_id
            gkw = {}
            if getattr(model.config, "gist_type", None) == "dynamic-interleave":
                gkw["ratio"] = args.override_ratio
            return model.model.generate_gist(
                input_ids=ids, attention_mask=left_valid.to(model.device), **gkw
            )

        left_store = SidecarStore(model)
        left_out = left_store.capture(example.qid + "_L", left_compress, left_lengths)
        left_gist_mask = left_out[1]
        left_pos = left_out[2]
        left_glen = int(left_gist_mask.shape[-1])
        left_pos = left_pos[:, -left_glen:]
        left_cache, _ = blend_gist_key_values(
            model.config, [left_out[0].past_key_values], [left_gist_mask],
            [left_pos], model.model.rotary_emb, system_length,
        )
        for sys_l, gl in zip(system_cache.layers, left_cache.layers):
            gl.keys = torch.cat([sys_l.keys, gl.keys], dim=-2)
            gl.values = torch.cat([sys_l.values, gl.values], dim=-2)
        prefix_cache = left_cache

        # insert R_k at its original logical position (after left gists)
        span = _sidecar_raw_span(capture_store, example.qid, k_star,
                                 doc_logical_start, rope_theta, rope_type)
        prefix_cache = _cat_span_to_cache(prefix_cache, span)

        # right gists at offsets[k*+1]
        if right_docs:
            right_grid = HH._grid_from_doc_ids(right_docs, args.max_doc_length, args.max_doc_num)
            right_valid = right_grid != -100
            right_lengths = [int(v.sum().item()) for v in right_valid]

            def right_compress():
                ids = right_grid.clone().to(model.device)
                ids[~right_valid] = model.model.gist_token_id
                gkw = {}
                if getattr(model.config, "gist_type", None) == "dynamic-interleave":
                    gkw["ratio"] = args.override_ratio
                return model.model.generate_gist(
                    input_ids=ids, attention_mask=right_valid.to(model.device), **gkw
                )

            right_store = SidecarStore(model)
            right_out = right_store.capture(example.qid + "_R", right_compress, right_lengths)
            right_gist_mask = right_out[1]
            right_pos = right_out[2]
            right_glen = int(right_gist_mask.shape[-1])
            right_pos = right_pos[:, -right_glen:]
            right_cache, _ = blend_gist_key_values(
                model.config, [right_out[0].past_key_values], [right_gist_mask],
                [right_pos], model.model.rotary_emb, offsets[k_star + 1],
            )
            for pl, rl in zip(prefix_cache.layers, right_cache.layers):
                pl.keys = torch.cat([pl.keys, rl.keys], dim=-2)
                pl.values = torch.cat([pl.values, rl.values], dim=-2)

        d_mode_info["t_load_sec"] = round(time.perf_counter() - t_load_start, 4)
        d_mode_info["sidecar_bytes_used"] = sidecar_bytes_target
        d_mode_info["extra_compression_sec"] = round(
            left_store.last_capture_sec + right_store.last_capture_sec, 4
        )

    elif mode == "d_raw_erratum_tail":
        # keep G_k, RoPE-reanchor R_k at repair tail
        # (already at correct logical positions from sidecar; append at end)
        span = _sidecar_raw_span(capture_store, example.qid, k_star,
                                 doc_logical_start, rope_theta, rope_type)
        prefix_cache = _cat_span_to_cache(prefix_cache, span)
        d_mode_info["t_load_sec"] = round(time.perf_counter() - t_load_start, 4)
        d_mode_info["sidecar_bytes_used"] = sidecar_bytes_target

    elif mode == "d_wrongblock_raw":
        # identical layout to d_raw_keepG but wrong block's raw KV
        wrong_k = (k_star + n_docs // 2) % n_docs
        span = _sidecar_raw_span(capture_store, example.qid, wrong_k,
                                 offsets[wrong_k], rope_theta, rope_type)
        prefix_cache = _cat_span_to_cache(prefix_cache, span)
        d_mode_info["wrong_doc_index"] = wrong_k
        d_mode_info["t_load_sec"] = round(time.perf_counter() - t_load_start, 4)
        d_mode_info["sidecar_bytes_used"] = capture_store.bytes_of(example.qid, [wrong_k])
    else:
        return None, f"d_contract_unknown_mode:{mode}"

    # cleanup sidecar entries for this qid
    capture_store.release(example.qid)
    for suffix in ("_L", "_R"):
        capture_store.release(example.qid + suffix)

    cache_length = prefix_cache.get_seq_length()
    return {
        "cache": prefix_cache,
        "system_length": system_length,
        "history_length": doc_tokens,
        "cache_length": cache_length,
        "doc_tokens": doc_tokens,
        "doc_chunks": doc_chunks,
        "kept_history_tokens": doc_tokens,
        "gist_tokens": gist_tokens,
        "actual_compression_ratio": float(doc_tokens / gist_tokens) if gist_tokens else 0.0,
        "system_prefill_sec": system_prefill_sec,
        "full_prefill_sec": 0.0,
        "tool_compress_sec": t_capture,
        "blend_sec": 0.0,
        "use_gist": True,
        "d_corr_doc_index": k_star,
        "d_corr_span_tokens": len(doc_ids[k_star]),
        "d_sham_tokens": 0,
        "d_recompute_tokens": 0,
        "d_recompute_docs": 0,
        "d_dropped_gist_tokens": 0,
        "d_corr_slice_prefill_sec": d_mode_info.get("t_load_sec", 0.0),
        "d_recompute_prefill_sec": 0.0,
        "d_contract_info": d_mode_info,
    }, None
