"""D0 anchor arms + D1 raw sidecar oracle repair arms (v2 rewrite, 2026-08-30).

Contract (D0, prereg v2):
- P_k captured during normal compression via SidecarStore (zero extra forward)
- repair: oracle(k*) -> load/decode(P_k*) -> edit -> query/decode
- NO forward of any already-seen history token (raw_replaceG edits the
  already-blended gist cache in place — it never re-compresses)
- k* = witness-IDF selection (prereg v2.2) injected via HH.D_CONTRACT_K;
  absent table -> median fallback (legacy column); k*=None -> explicit
  no-injection row (injected=false), never an exception

D0 anchors (with c2kv baseline):
  oracle_target_only       store only P_{k*} (operator headroom; injected=false)
  allblock_sidecar         same cache as raw_keepG; reports FULL cold-storage
                           bytes (the difference is the bytes ledger only)

D1 raw sidecar oracle repair (all use sidecar R_k^local, no scratch prefill):
  raw_keepG@k      keep all gists + append R_k anchored at the doc's ORIGINAL
                   logical offset (double coverage, splice_keep semantics)
  raw_replaceG@k   slice G_k out of the blended cache, insert R_k at G_k's
                   physical slot anchored at the same original offset
  raw_erratum_tail keep G_k, anchor R_k at the REPAIR TAIL (positions
                   system+doc_tokens..) and advance the position ledger by L

Wrongblock/sham arms were CANCELLED by prereg v2.3 (degenerate construction);
the k-sweep's non-witness ks are the wrong-block distribution.

Layout invariants (acceptance #5 regression): the three D1 arms differ as
(cache_length, k_anchor, history_length) triples —
  keepG          (S+G+Lk, offsets[k*],      doc_tokens)
  replaceG       (S+G-|Gk|+Lk, offsets[k*], doc_tokens)
  erratum_tail   (S+G+Lk, S+doc_tokens,     doc_tokens+Lk)

The k-sweep driver shares ONE compression forward per qid:
`prepare_d_contract_state` (system prefill + capture + blend) runs once,
`ksweep_prefix_for_k` rebuilds only the splice per k.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "inference"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import torch

import eval_agent_history_c2kv as HH
from d0_sidecar import SidecarStore

logger = logging.getLogger(__name__)

ARM_MODES = {
    # D0 anchors
    "oracle_target_only": "d_oracle_target_only",
    "allblock_sidecar": "d_allblock_sidecar",
    # D1 raw sidecar oracle repair
    "raw_keepG": "d_raw_keepG",
    "raw_replaceG": "d_raw_replaceG",
    "raw_erratum_tail": "d_raw_erratum_tail",
    # D2 (built by d2_short_erratum.py; dispatched through D_CONTRACT_MODES)
    "short_erratum": "d_short_erratum",
    # D1 fourth arm (S1.1): SSA's DEFAULT config SG+SR — keep ONLY k*'s
    # gist, drop every other doc's gist, append R_k* (raw_keepG is the
    # paper's AG+SR, measured and rejected; raw_replaceG ~ SR-only)
    "raw_SGSR": "d_raw_SGSR",
    # D4/D5/D6/D7 runtime arms (d37_arms.py; run-gated on the |R| verdict)
    "reskv_capsule": "d_reskv_capsule",
    "keepkv_capsule": "d_keepkv_capsule",
    "less_fold": "d_less_fold",
    "grkv_v_edit": "d_grkv_v_edit",
    "selkv_bias": "d_selkv_bias",
    "selkv_count": "d_selkv_count",
}
D_CONTRACT_MODES = set(ARM_MODES.values())

@torch.inference_mode()
def _keep_only_gist_span(cache: Any, keep_start: int, keep_end: int, system_length: int) -> Any:
    """Cut every gist slot OUTSIDE [keep_start, keep_end) from the cache
    (SG+SR surgery: the system block is preserved)."""
    for layer in cache.layers:
        layer.keys = torch.cat(
            [layer.keys[..., :system_length, :],
             layer.keys[..., keep_start:keep_end, :]], dim=-2)
        layer.values = torch.cat(
            [layer.values[..., :system_length, :],
             layer.values[..., keep_start:keep_end, :]], dim=-2)
    return cache


class _KMissing:
    """Sentinel: qid absent from the injected witness table."""


_K_MISSING = _KMissing()


def gist_doc_spans(gist_mask: torch.Tensor) -> List[Tuple[int, int]]:
    """EXACT per-doc gist spans [start, end) in the blended gist region.

    Derived from the compression mask itself: row i contributes
    ``mask[i].sum()`` gist tokens, concatenated in row order by
    ``blend_gist_key_values`` — cumulative counts are exact.  This replaces
    ``HH._gist_spans_from_doc_lengths`` for splicing (that helper allocates
    gists proportionally to token counts and overlaps one gist at every
    fractional doc boundary; it stays untouched for its attribution caller).
    """
    counts = [int(c) for c in gist_mask.sum(dim=1).tolist()]
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for c in counts:
        if c > 0:
            spans.append((cursor, cursor + c))
            cursor += c
    return spans


def resolve_k_star(example_qid: str, n_docs: int, k_override: Optional[int]) -> Tuple[Optional[int], str]:
    """k* per prereg v2.2: sweep override > witness table > median fallback."""
    if k_override is not None:
        return int(k_override), "sweep"
    k_witness = HH.D_CONTRACT_K.get(example_qid, _K_MISSING)
    if k_witness is _K_MISSING:
        return (n_docs - 1) // 2, "median_fallback"
    if k_witness is None:
        return None, "witness_none"
    return int(k_witness), "witness"


@torch.inference_mode()
def prepare_d_contract_state(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    store: SidecarStore,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Per-qid shared state: tokenize, system prefill, ONE compression with
    sidecar capture, gist blend.  The returned gist cache is PRE-system-cat
    so every consumer rebuilds a fresh final cache from immutable tensors.

    Returns (state, None) or (None, skip_reason).
    """
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

    # --- compression with sidecar capture (the ONLY forward over history) ---
    grid = context_input_ids  # (max_doc_num, max_doc_length), filler rows all -100
    valid_mask = grid != -100
    doc_lengths = [int(v.sum().item()) for v in valid_mask]  # trailing zeros = fillers

    def compress_call():
        ids = grid.clone().to(model.device)
        ids[~valid_mask] = model.model.gist_token_id
        gist_kwargs = {}
        if getattr(model.config, "gist_type", None) == "dynamic-interleave":
            gist_kwargs["ratio"] = args.override_ratio
        return model.model.generate_gist(
            input_ids=ids, attention_mask=valid_mask.to(model.device), **gist_kwargs
        )

    outputs = store.capture(example.qid, compress_call, doc_lengths)
    t_capture = store.last_compress_with_capture_sec

    # offline distortion bench feed: persist the first N captured qids'
    # full sidecar (pre-RoPE K/V, plus Q when the store has want_q) before
    # any arm logic touches or releases it (driver sets HH.D_CONTRACT_DUMP)
    dump_cfg = getattr(HH, "D_CONTRACT_DUMP", None)
    if dump_cfg and dump_cfg.get("remaining", 0) > 0:
        entry = store.entries[example.qid]
        Path(dump_cfg["path"]).mkdir(parents=True, exist_ok=True)
        torch.save({
            "qid": example.qid,
            "doc_lengths": [len(ids) for ids in doc_ids],
            "n_layers": len(entry),
            "k": [layer["k"] for layer in entry],
            "v": [layer["v"] for layer in entry],
            "q": [layer["q"] for layer in entry],
        }, str(Path(dump_cfg["path"]) / f"{example.qid.replace(':', '_')}.pt"))
        dump_cfg["remaining"] -= 1

    from models.gist_utils import blend_gist_key_values
    gist_mask = outputs[1]
    pos_ids = outputs[2]
    gist_len = int(gist_mask.shape[-1])
    pos_ids = pos_ids[:, -gist_len:]
    gist_cache, _ = blend_gist_key_values(
        model.config, [outputs[0].past_key_values], [gist_mask],
        [pos_ids], model.model.rotary_emb, system_length,
    )
    total_gist_tokens = int(sum(c for c in gist_mask.sum(dim=1).tolist()))
    assert gist_cache.get_seq_length() == total_gist_tokens, (
        f"gist cache {gist_cache.get_seq_length()} != gists {total_gist_tokens}"
    )
    spans = gist_doc_spans(gist_mask)
    assert len(spans) == n_docs, f"gist spans {len(spans)} != real docs {n_docs}"

    return {
        "qid": example.qid,
        "system_cache": system_cache,
        "gist_cache": gist_cache,      # PRE-system-cat: immutable per-qid tensors
        "system_length": system_length,
        "system_prefill_sec": system_prefill_sec,
        "doc_ids": doc_ids,
        "offsets": offsets,
        "doc_tokens": doc_tokens,
        "doc_chunks": doc_chunks,
        "spans": spans,
        "total_gist_tokens": total_gist_tokens,
        "t_capture": t_capture,
        "sidecar_bytes_all": store.bytes_of(example.qid),
    }, None


def _merge_system_gist(state: Dict[str, Any], model_config: Any) -> Any:
    """Fresh final cache = system ++ gist (new tensors; pristine parts never
    mutated — safe to call repeatedly across a k-sweep)."""
    from transformers.cache_utils import DynamicCache

    per_layer = []
    for sys_layer, gist_layer in zip(state["system_cache"].layers, state["gist_cache"].layers):
        per_layer.append((
            torch.cat([sys_layer.keys, gist_layer.keys], dim=-2),
            torch.cat([sys_layer.values, gist_layer.values], dim=-2),
        ))
    return DynamicCache(per_layer, config=model_config)


@torch.inference_mode()
def _sidecar_raw_span(
    store: SidecarStore,
    qid: str,
    doc_index: int,
    abs_start: int,
    rotary_emb: Any,
    device: Any,
    dtype: torch.dtype,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer (keys, values) for one doc; K RoPE'd onto ABSOLUTE positions
    abs_start..abs_start+L-1 via the model's own rotary (apply_abs_rope —
    full per-token RoPE, legal for the sidecar's PRE-RoPE K)."""
    from inference.abs_rope import apply_abs_rope

    keys = store.get(qid, doc_index, "k", device=device, dtype=dtype)
    values = store.get(qid, doc_index, "v", device=device, dtype=dtype)
    span = []
    for k, v in zip(keys, values):
        rotated = apply_abs_rope(k, abs_start, rotary_emb)  # (kv_heads, L, D)
        span.append((rotated.unsqueeze(0), v.unsqueeze(0)))  # (1, H, L, D)
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
def _replace_span_in_cache(
    cache: Any,
    span_kv: List[Tuple[torch.Tensor, torch.Tensor]],
    phys_start: int,
    phys_end: int,
) -> Any:
    """Cut cache[..., phys_start:phys_end, :] out and splice span_kv in."""
    for layer, (keys, values) in zip(cache.layers, span_kv):
        layer.keys = torch.cat(
            [layer.keys[..., :phys_start, :], keys, layer.keys[..., phys_end:, :]], dim=-2
        )
        layer.values = torch.cat(
            [layer.values[..., :phys_start, :], values, layer.values[..., phys_end:, :]], dim=-2
        )
    return cache


def _finish_prefix(
    state: Dict[str, Any],
    cache: Any,
    *,
    history_length: int,
    gist_tokens_final: int,
    span_tokens: int,
    dropped_gist_tokens: int,
    d_mode_info: Dict[str, Any],
    t_load_sec: float,
) -> Dict[str, Any]:
    cache_length = cache.get_seq_length()
    d_mode_info["t_load_sec"] = round(t_load_sec, 4)
    compressed_footprint = max(1, cache_length - state["system_length"])
    return {
        "cache": cache,
        "system_length": state["system_length"],
        "history_length": history_length,
        "cache_length": cache_length,
        "doc_tokens": state["doc_tokens"],
        "doc_chunks": state["doc_chunks"],
        "kept_history_tokens": state["doc_tokens"],
        "gist_tokens": gist_tokens_final,
        "actual_compression_ratio": float(state["doc_tokens"] / compressed_footprint),
        "system_prefill_sec": state["system_prefill_sec"],
        "full_prefill_sec": 0.0,
        "tool_compress_sec": state["t_capture"],
        "blend_sec": 0.0,
        "use_gist": True,
        "d_corr_doc_index": d_mode_info.get("k_star"),
        "d_corr_span_tokens": span_tokens,
        "d_sham_tokens": 0,
        "d_recompute_tokens": 0,
        "d_recompute_docs": 0,
        "d_dropped_gist_tokens": dropped_gist_tokens,
        "d_corr_slice_prefill_sec": round(t_load_sec, 4),
        "d_recompute_prefill_sec": 0.0,
        "d_contract_info": d_mode_info,
    }


@torch.inference_mode()
def build_d_contract_prefix(
    model: Any,
    tokenizer: Any,
    example: Any,
    args: argparse.Namespace,
    mode: str,
    store: SidecarStore,
    k_override: Optional[int] = None,
) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Build a D-contract prefix using sidecar payload (no history forward)."""
    if mode not in D_CONTRACT_MODES:
        return None, f"d_contract_unknown_mode:{mode}"

    state, skip_reason = prepare_d_contract_state(model, tokenizer, example, args, store)
    if state is None:
        return None, skip_reason
    doc_ids: List[List[int]] = state["doc_ids"]
    offsets: List[int] = state["offsets"]
    spans = state["spans"]
    n_docs = len(doc_ids)
    system_length = state["system_length"]
    doc_tokens = state["doc_tokens"]
    k_star, k_policy = resolve_k_star(example.qid, n_docs, k_override)

    prefix_cache = _merge_system_gist(state, model.config)
    _, cache_device, cache_dtype = (
        prefix_cache.get_seq_length(),
        prefix_cache.layers[0].keys.device,
        prefix_cache.layers[0].keys.dtype,
    )
    rotary_emb = model.model.rotary_emb

    HH._sync_device(model.device)
    t_load_start = time.perf_counter()
    sidecar_bytes_all = state["sidecar_bytes_all"]
    sidecar_bytes_target = store.bytes_of(example.qid, [k_star]) if k_star is not None else 0

    d_mode_info: Dict[str, Any] = {
        "k_policy": k_policy,
        "k_star": k_star,
        "sidecar_bytes_all": sidecar_bytes_all,
        "sidecar_bytes_target": sidecar_bytes_target,
        "t_capture_sec": round(state["t_capture"], 4),
        "gist_span_of_target": None,
        "k_anchor": None,
    }
    injected = False
    dropped_gist_tokens = 0
    span_tokens = 0
    history_length = doc_tokens

    if k_star is None:
        # prereg v2.2: no literal witness in history — a RESULT, not an error.
        # No block is selectable, so nothing is injected or stored.
        store.release(example.qid)
        d_mode_info.update(
            injected=False,
            sidecar_bytes_used=0,
            note="k_star=None (witness): synthesized-argument qid, no repair channel",
        )
        HH._sync_device(model.device)
        t_load = time.perf_counter() - t_load_start
        return _finish_prefix(
            state, prefix_cache,
            history_length=history_length,
            gist_tokens_final=state["total_gist_tokens"],
            span_tokens=0, dropped_gist_tokens=0,
            d_mode_info=d_mode_info, t_load_sec=t_load,
        ), None

    if mode == "d_oracle_target_only":
        # operator headroom: storage-only arm. Injects NOTHING; the bytes
        # ledger reports the stored payload only, and span tokens are 0 so
        # byte-Pareto never credits storage this arm did not inject (B13).
        store.drop_docs(example.qid, [k_star])
        d_mode_info.update(
            injected=False,
            sidecar_bytes_used=sidecar_bytes_target,
            note="no injection; measures compression+storage only",
        )
        store.release(example.qid)
        HH._sync_device(model.device)
        t_load = time.perf_counter() - t_load_start
        return _finish_prefix(
            state, prefix_cache,
            history_length=history_length,
            gist_tokens_final=state["total_gist_tokens"],
            span_tokens=0, dropped_gist_tokens=0,
            d_mode_info=d_mode_info, t_load_sec=t_load,
        ), None

    gs, ge = spans[k_star]
    phys_start = system_length + gs
    phys_end = system_length + ge
    if mode == "d_raw_keepG":
        anchor = offsets[k_star]
        span = _sidecar_raw_span(
            store, example.qid, k_star, anchor, rotary_emb, cache_device, cache_dtype
        )
        prefix_cache = _cat_span_to_cache(prefix_cache, span)
        injected, span_tokens = True, len(doc_ids[k_star])
        d_mode_info.update(injected=True, sidecar_bytes_used=sidecar_bytes_target, k_anchor=anchor)
    elif mode == "d_allblock_sidecar":
        # SAME cache state as d_raw_keepG (verified by the layout regression
        # test); the ONLY difference is the bytes ledger: this anchor reports
        # full cold-storage bytes for ALL P_k.
        anchor = offsets[k_star]
        span = _sidecar_raw_span(
            store, example.qid, k_star, anchor, rotary_emb, cache_device, cache_dtype
        )
        prefix_cache = _cat_span_to_cache(prefix_cache, span)
        injected, span_tokens = True, len(doc_ids[k_star])
        d_mode_info.update(
            injected=True, sidecar_bytes_used=sidecar_bytes_all, k_anchor=anchor,
            note="cache identical to d_raw_keepG; bytes ledger = all docs",
        )
    elif mode == "d_raw_replaceG":
        # Slice G_k out of the ALREADY-BLENDED cache and insert R_k in its
        # physical slot — no second compression forward (the old code
        # re-ran generate_gist on left/right doc groups: replay-forbidden
        # and non-deterministic vs the full grid).
        anchor = offsets[k_star]
        span = _sidecar_raw_span(
            store, example.qid, k_star, anchor, rotary_emb, cache_device, cache_dtype
        )
        prefix_cache = _replace_span_in_cache(prefix_cache, span, phys_start, phys_end)
        dropped_gist_tokens = ge - gs
        injected, span_tokens = True, len(doc_ids[k_star])
        d_mode_info.update(
            injected=True, sidecar_bytes_used=sidecar_bytes_target, k_anchor=anchor,
            gist_span_of_target=[gs, ge],
        )
    elif mode == "d_raw_SGSR":
        # S1.1 — SSA's DEFAULT configuration (SG+SR): keep ONLY k*'s gist,
        # drop every other doc's gist, then append R_k* at its original
        # offset.  (raw_keepG = the paper's AG+SR, which §4.5 measured and
        # rejected; raw_replaceG ~ SR-only, its worst row.)
        prefix_cache = _keep_only_gist_span(prefix_cache, phys_start, phys_end, system_length)
        anchor = offsets[k_star]
        span = _sidecar_raw_span(
            store, example.qid, k_star, anchor, rotary_emb, cache_device, cache_dtype
        )
        prefix_cache = _cat_span_to_cache(prefix_cache, span)
        dropped_gist_tokens = state["total_gist_tokens"] - (ge - gs)
        injected, span_tokens = True, len(doc_ids[k_star])
        d_mode_info.update(injected=True, sidecar_bytes_used=sidecar_bytes_all,
                           k_anchor=anchor, note="SG+SR: other docs' gists dropped")
    elif mode == "d_raw_erratum_tail":
        # keep G_k; anchor R_k at the REPAIR TAIL — positions continuing
        # after the logical history — and advance the position ledger by L
        # so decode never collides with the tail block (B9: the old code
        # was byte-identical to raw_keepG with no reanchor).
        anchor = system_length + doc_tokens
        span = _sidecar_raw_span(
            store, example.qid, k_star, anchor, rotary_emb, cache_device, cache_dtype
        )
        prefix_cache = _cat_span_to_cache(prefix_cache, span)
        injected, span_tokens = True, len(doc_ids[k_star])
        history_length = doc_tokens + span_tokens
        d_mode_info.update(injected=True, sidecar_bytes_used=sidecar_bytes_target, k_anchor=anchor)
    else:  # pragma: no cover - guarded at function entry
        return None, f"d_contract_unknown_mode:{mode}"

    store.release(example.qid)
    HH._sync_device(model.device)
    t_load = time.perf_counter() - t_load_start
    assert injected
    return _finish_prefix(
        state, prefix_cache,
        history_length=history_length,
        gist_tokens_final=state["total_gist_tokens"] - dropped_gist_tokens,
        span_tokens=span_tokens,
        dropped_gist_tokens=dropped_gist_tokens,
        d_mode_info=d_mode_info, t_load_sec=t_load,
    ), None


@torch.inference_mode()
def ksweep_prefix_for_k(
    model: Any,
    state: Dict[str, Any],
    store: SidecarStore,
    k: int,
) -> Dict[str, Any]:
    """Fresh raw_keepG-layout prefix for doc k from the SHARED per-qid state
    (one compression forward for the whole sweep).  The caller releases the
    store entry after the last k."""
    prefix_cache = _merge_system_gist(state, model.config)
    device, dtype = prefix_cache.layers[0].keys.device, prefix_cache.layers[0].keys.dtype
    HH._sync_device(model.device)
    t_load_start = time.perf_counter()
    anchor = state["offsets"][k]
    span = _sidecar_raw_span(
        store, state["qid"], k, anchor, model.model.rotary_emb, device, dtype
    )
    prefix_cache = _cat_span_to_cache(prefix_cache, span)
    HH._sync_device(model.device)
    t_load = time.perf_counter() - t_load_start
    bytes_target = store.bytes_of(state["qid"], [k])
    d_mode_info = {
        "k_policy": "sweep",
        "k_star": k,
        "sidecar_bytes_all": state["sidecar_bytes_all"],
        "sidecar_bytes_target": bytes_target,
        "sidecar_bytes_used": bytes_target,
        "t_capture_sec": round(state["t_capture"], 4),
        "gist_span_of_target": list(state["spans"][k]),
        "k_anchor": anchor,
        "injected": True,
    }
    return _finish_prefix(
        state, prefix_cache,
        history_length=state["doc_tokens"],
        gist_tokens_final=state["total_gist_tokens"],
        span_tokens=len(state["doc_ids"][k]),
        dropped_gist_tokens=0,
        d_mode_info=d_mode_info, t_load_sec=t_load,
    )
