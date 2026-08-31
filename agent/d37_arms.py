"""D4/D5/D6/D7 runtime arms (line A wiring; smoke-ready, run-gated on |R|).

All five builders reuse `d1_arms.prepare_d_contract_state` (ONE compression
forward + sidecar capture) and k* from the frozen witness table (fallback
doc 0 so smokes always run).  They differ in what they inject:

  d_reskv_capsule   k-means r capsule spliced INTO G_k's physical slot
                    (cache shrinks); log(count) enters as a KEY-LOGIT BIAS
                    through the eager registry (never a V scale)
  d_keepkv_capsule  vote-merged r capsule, same splice; log(votes) bias
  d_less_fold       no cache change; per-layer elu+1 ledgers folded into
                    the live attention numerator/denominator
  d_grkv_v_edit     closed-form ΔV written back into G_k's values
                    (needs want_q=True); no registry
  d_selkv_bias      no cache change; α·log R (log-space geometric mean)
                    key-logit bias over G_k's gist span
  d_selkv_count     control arm: α·log(token_count) uniform bias

The registry (python/inference/attn_bias.py) stays ACTIVE during decode;
the driver clears it after every row.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import torch

import eval_agent_history_c2kv as HH
from d1_arms import (
    _finish_prefix,
    _merge_system_gist,
    _replace_span_in_cache,
    prepare_d_contract_state,
    resolve_k_star,
)
from inference import attn_bias
from inference.abs_rope import apply_abs_rope

N_REP = 4  # Qwen3-4B: 32 q heads / 8 kv heads

D37_ARM_MODES = {
    "d_reskv_capsule": "reskv",
    "d_keepkv_capsule": "keepkv",
    "d_less_fold": None,
    "d_grkv_v_edit": None,
    "d_selkv_bias": None,
    "d_selkv_count": None,
}


@torch.inference_mode()
def build_d37_prefix(model, tokenizer, example, args, mode, store):
    from d4_capsules_v2 import decode_reskv, encode_keepkv, encode_reskv, equalize_r
    from d5_v2 import build_layer_entries, encode_less
    from d67_v2 import grkv_edit_cache, selkv_count_bias, selkv_mass_ratio

    splice = D37_ARM_MODES[mode]
    state, skip = prepare_d_contract_state(model, tokenizer, example, args, store)
    if state is None:
        return None, skip
    n_docs = len(state["doc_ids"])
    k_witness, _ = resolve_k_star(example.qid, n_docs, None)
    k_star = 0 if k_witness is None else k_witness  # smoke fallback
    offsets, spans = state["offsets"], state["spans"]
    gs, ge = spans[k_star]
    phys_start = state["system_length"] + gs
    phys_end = state["system_length"] + ge
    qid = example.qid

    prefix_cache = _merge_system_gist(state, model.config)
    device = prefix_cache.layers[0].keys.device
    dtype = prefix_cache.layers[0].keys.dtype
    rotary = model.model.rotary_emb
    want_q = bool(store.want_q)
    # GQA group size from the model config (was hardcoded 4)
    n_rep = int(model.config.num_attention_heads) // int(model.config.num_key_value_heads)
    n_kv_heads = int(model.config.num_key_value_heads)

    info: Dict[str, Any] = {"k_star": k_star, "k_witness": k_witness,
                            "gist_span": [gs, ge], "mode": mode}
    entries = {}
    dropped = 0

    HH._sync_device(model.device)
    t0 = time.perf_counter()

    if splice is not None:
        k_raw = store.get(qid, k_star, "k", device=device, dtype=torch.float32)
        v_raw = store.get(qid, k_star, "v", device=device, dtype=torch.float32)
        q_raw = (store.get(qid, k_star, "q", device=device, dtype=torch.float32)
                 if want_q else None)
        r = 16
        if want_q:
            # k_raw[0] is layer 0's (H_kv, L, D) — equalize_r expects exactly
            # that; the extra unsqueeze made every capsule einsum 4-D (smoke)
            eq = equalize_r(k_raw[0], v_raw[0], q_raw[0])
            r = eq.get("reskv_r" if splice == "reskv" else "keepkv_r") or 16
            info["capsule_bytes"] = eq
        span_kv = []
        for li in range(len(prefix_cache.layers)):
            if splice == "reskv":
                cap = encode_reskv(k_raw[li], v_raw[li], r)
                ck, cv, bias = decode_reskv(cap)          # (H, r, D) x2, (H, r)
            else:
                q_l = q_raw[li] if q_raw is not None else torch.zeros_like(k_raw[li])
                cap = encode_keepkv(k_raw[li], v_raw[li], q_l, r)
                # stack ALL heads (caps[0] would splice head 0's capsule
                # into every head — the 5/6 smoke failure)
                ck = torch.stack([c["k"] for c in cap["caps"]])       # (H, r, D)
                cv = torch.stack([c["v"] for c in cap["caps"]])       # (H, r, D)
                bias = torch.stack([torch.log(c["votes"].clamp_min(1e-9))
                                    for c in cap["caps"]])            # (H, r)
            ck = apply_abs_rope(ck.float(), offsets[k_star], rotary, dtype=dtype, device=device)
            span_kv.append((ck.unsqueeze(0), cv.to(dtype).unsqueeze(0)))
            # key_bias is PER KV HEAD: (H_kv, P) — the registry GQA-expands it
            bias_full = torch.zeros(n_kv_heads, prefix_cache.get_seq_length(),
                                    dtype=torch.float32, device=device)
            bias_full[:, phys_start:phys_start + ck.shape[-2]] = bias.float().to(device)
            entries[li] = attn_bias.LayerBiasEntry(key_bias=bias_full)
        prefix_cache = _replace_span_in_cache(prefix_cache, span_kv, phys_start, phys_end)
        dropped = ge - gs
        info.update(capsule_kind=splice, r=r)

    elif mode == "d_less_fold":
        k_raw = store.get(qid, k_star, "k", device=device, dtype=torch.float32)
        v_raw = store.get(qid, k_star, "v", device=device, dtype=torch.float32)
        ledgers = {li: encode_less(k_raw[li], v_raw[li])
                   for li in range(len(prefix_cache.layers))}
        entries = build_layer_entries(ledgers)
        info.update(ledger="elu+1", layers=len(ledgers))

    elif mode == "d_grkv_v_edit":
        if not want_q:
            store.release(qid)
            return None, "d6_needs_want_q"
        residuals = grkv_edit_cache(prefix_cache, store, qid, k_star,
                                    (phys_start, phys_end), offsets[k_star], rotary,
                                    n_rep=n_rep)
        info.update(residual_first=round(residuals[0], 3),
                    residual_last=round(residuals[-1], 3))

    else:  # d_selkv_bias / d_selkv_count
        if not want_q:
            store.release(qid)
            return None, "selkv_needs_want_q"
        alpha = float(getattr(args, "selkv_alpha", 0.5))
        L_k = len(state["doc_ids"][k_star])  # state has no doc_lengths key
        k_raw = store.get(qid, k_star, "k", device=device, dtype=torch.float32)
        q_raw = store.get(qid, k_star, "q", device=device, dtype=torch.float32)
        for li, layer in enumerate(prefix_cache.layers):
            k_gist = layer.keys[0, :, phys_start:phys_end, :]
            q_rot = apply_abs_rope(q_raw[li], offsets[k_star], rotary)
            if mode == "d_selkv_bias":
                bias_slots = alpha * selkv_mass_ratio(k_gist.float(), k_raw[li], q_rot, n_rep=n_rep)
            else:
                bias_slots = selkv_count_bias(L_k, phys_end - phys_start, alpha)
            # per-kv-head bias over physical cache positions: (H_kv, P)
            bias_full = torch.zeros(n_kv_heads, prefix_cache.get_seq_length(),
                                    dtype=torch.float32, device=device)
            bias_full[:, phys_start:phys_end] = bias_slots.float().to(device)
            entries[li] = attn_bias.LayerBiasEntry(key_bias=bias_full)
        info.update(alpha=alpha)

    HH._sync_device(model.device)
    t_load = time.perf_counter() - t0
    if entries:
        attn_bias.set_entries(entries)
        info["registry_active"] = True
    store.release(qid)

    return _finish_prefix(
        state, prefix_cache,
        history_length=state["doc_tokens"],
        gist_tokens_final=state["total_gist_tokens"] - dropped,
        span_tokens=len(state["doc_ids"][k_star]) if splice else 0,
        dropped_gist_tokens=dropped,
        d_mode_info=info,
        t_load_sec=t_load,
    ), None
