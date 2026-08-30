"""D2: Short new erratum (not replaying the raw block).

Source: Models Take Notes at Prefill (arXiv:2606.17107).

Arms:
  short_erratum: oracle provides the correct content, forward ONE new
    correction sentence (e.g. "The correct value of X in history is Y")
    appended after the gist prefix. The correction forward counts as T_edit.
  short_erratum_kvbank (conditional): encode the same correction as a
    selected-layer latent KV bank (Memory Inception transfer). Only if the
    visible erratum shows benefit.

This is NOT a raw block replay — it is a compact textual correction at
~1/100 the bytes of a full block's raw KV.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import torch

import eval_agent_history_c2kv as HH

logger = logging.getLogger(__name__)


def _compose_erratum_text(target: str) -> str:
    """Compose a compact correction sentence from the gold target.

    The oracle knows what the correct next action is; the erratum tells the
    model the relevant fact from history that leads to that action. Since we
    are on a teacher-forced eval, the 'correction' is a restatement of the
    key content from the target block's raw doc.
    """
    from d_strict_metric import _parse_tool_call_payload
    payload = _parse_tool_call_payload(target)
    if payload and payload["name"]:
        args_str = json.dumps(payload["arguments"], ensure_ascii=False) if payload["arguments"] else ""
        return f"[correction] The next action should be: {payload['name']}({args_str}). Ignore any conflicting prior information."
    return f"[correction] The correct information is: {target[:200]}"


@torch.inference_mode()
def build_short_erratum_prefix(
    model, tokenizer, example, args, mode: str,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build a short-erratum prefix: c2kv + one correction sentence."""
    from d1_arms import build_d_contract_prefix
    from d0_sidecar import SidecarStore

    # First: standard c2kv prefix (compression only, no repair injection)
    context_input_ids, doc_tokens, doc_chunks, history, skip = HH._build_history_chunks(
        tokenizer, example, args
    )
    if context_input_ids is None:
        return None, skip
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

    # compression forward (standard)
    grid = context_input_ids
    valid_mask = grid != -100
    ids = grid.clone().to(model.device)
    ids[~valid_mask] = model.model.gist_token_id
    gist_kwargs = {}
    if getattr(model.config, "gist_type", None) == "dynamic-interleave":
        gist_kwargs["ratio"] = args.override_ratio
    t_comp_start = time.perf_counter()
    outputs, gist_mask, pos_ids = model.model.generate_gist(
        input_ids=ids, attention_mask=valid_mask.to(model.device), **gist_kwargs
    )
    t_compress = time.perf_counter() - t_comp_start

    gist_len = int(gist_mask.shape[-1])
    pos_ids = pos_ids[:, -gist_len:]
    from models.gist_utils import blend_gist_key_values
    prefix_cache, _ = blend_gist_key_values(
        model.config, [outputs.past_key_values], [gist_mask],
        [pos_ids], model.model.rotary_emb, system_length,
    )
    for sys_l, gl in zip(system_cache.layers, prefix_cache.layers):
        gl.keys = torch.cat([sys_l.keys, gl.keys], dim=-2)
        gl.values = torch.cat([sys_l.values, gl.values], dim=-2)
    gist_tokens = prefix_cache.get_seq_length() - system_length

    # --- short erratum forward (T_edit) ---
    t_edit_start = time.perf_counter()
    erratum_text = _compose_erratum_text(example.answer)
    erratum_ids = HH._chat_template_ids(
        tokenizer, [{"role": "user", "content": erratum_text}]
    )
    erratum_input = torch.tensor([erratum_ids], dtype=torch.long, device=model.device)
    past_length = prefix_cache.get_seq_length()
    logical_start = system_length + doc_tokens
    attention_mask = torch.ones(
        1, past_length + erratum_input.shape[1], device=model.device, dtype=torch.long
    )
    position_ids = torch.arange(
        logical_start, logical_start + erratum_input.shape[1],
        device=model.device
    ).unsqueeze(0)
    model.model.config._attn_implementation = args.generate_attn_impl
    erratum_out = model(
        input_ids=erratum_input,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=prefix_cache,
        use_cache=True,
        use_gist=True,
        logits_to_keep=1,
    )
    prefix_cache = erratum_out.past_key_values
    t_edit = time.perf_counter() - t_edit_start
    erratum_tokens = len(erratum_ids)

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
        "tool_compress_sec": t_compress,
        "blend_sec": 0.0,
        "use_gist": True,
        "d_corr_doc_index": k_star,
        "d_corr_span_tokens": erratum_tokens,
        "d_sham_tokens": 0,
        "d_recompute_tokens": 0,
        "d_recompute_docs": 0,
        "d_dropped_gist_tokens": 0,
        "d_corr_slice_prefill_sec": round(t_edit, 4),
        "d_recompute_prefill_sec": 0.0,
        "d_contract_info": {
            "mode": mode,
            "erratum_text": erratum_text[:200],
            "erratum_tokens": erratum_tokens,
            "t_compress_sec": round(t_compress, 4),
            "t_edit_sec": round(t_edit, 4),
        },
    }, None
