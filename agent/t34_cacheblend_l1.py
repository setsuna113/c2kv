# -*- coding: utf-8 -*-
"""t34 U1b (c) -- CacheBlend layer-1 deviation (arXiv 2405.16444), digest 4.5.

CacheBlend's transferable idea is the self-referential surrogate: it never
knows the full-prefill reference, so it measures a candidate's deviation by
*cheaply recomputing that candidate* (Sec. 6.3, Insight 1 + the explicit
"this intuitive scheme is not feasible as it needs to know full-prefilled KV
cache").  Ported here: for a decision point with ``n_docs`` compressed blocks,
run a PARTIAL prefill of candidate block ``k``'s raw token ids through
``embed_tokens`` + ``layers[0..probe_layer]`` at that block's ledger positions,
keep the probe layer's K,V of the raw span, and compare the query-projected
attention OUTPUT of the gist span against that of the raw span:

    dev(k) = || Attn_1(q, gist-span-of-k) - Attn_1(q, raw-span-of-k) ||_F

with the softmax taken WITHIN each span.  This is NEITHER of the paper's two
deviations verbatim: KVD differences K,V per token (impossible here --
``gist_len != original_seq_len``, so it would need a padding/pooling whose
product the number would then be, digest 4.5 "shape mismatch is the whole
problem"), and CAD is the L-2 norm of the difference of the attention MATRIX
(also impossible -- a [1, gist_len] and a [1, raw_len] probability row have no
common shape).  What survives the shape change is the attention OUTPUT, which
lives in R^{H x d} whatever the span length; dev(k) is its norm.  Same
motivation as CAD (query-side, post-softmax), different space -- see
DEVIATIONS, "deviation statistic".

Two estimators, registered SEPARATELY (digest 4.5):
  * locator: ``argmax_k dev(k)``  -> t34_common.locate_table vs the 25.0 % floor
  * trigger: ``max_k dev(k)``     -> a feature in the 161-row trigger frame

RUNBOOK

  1. NPU   python agent/t34_cacheblend_l1.py pregate \\
             --sidecar results/t34/sidecar_c2kv.jsonl --model <ckpt> --n-rows 24 \\
             --probe-layers 8 --out results/t34/cacheblend/pregate.json
           Layer-wise Spearman rank correlation of per-block deviation on a
           pilot (the paper's Insight 2).  STOP if it is not high: the cascade
           premise does not hold on a trained compressor and the port is
           falsified before it is scored (digest 4.5 pre-gate).
  2. NPU   python agent/t34_cacheblend_l1.py probe \\
             --sidecar results/t34/sidecar_c2kv.jsonl --model <ckpt> --arm c2kv \\
             --probe-layer 1 --out results/t34/cacheblend/dev_c2kv.jsonl
     NPU   ... --arm full --s0-swap --out results/t34/cacheblend/dev_full_s0.jsonl
           (the S0 twin: the same deviation computed on the FULL prefix
            against itself with one block swapped)
  3. here  PYTHONIOENCODING=utf-8 python agent/t34_cacheblend_l1.py score \\
             --root . --dev results/t34/cacheblend/dev_c2kv.jsonl \\
             --dev-s0 results/t34/cacheblend/dev_full_s0.jsonl \\
             --features results/t34/features_cacheblend.jsonl \\
             --report results/t34/cacheblend_report.json

WIRING (steps 1-2): the partial prefill mirrors the D-line slice-prefill
position convention -- block ``k``'s ledger start is
``offsets[k] = system_length + sum(len(doc_ids[:k]))``
(``eval_agent_history_c2kv.py:2016-2024``), and the query forward continues at
``system_length + history_length`` (the LOGICAL, uncompressed offset; the cache
is physically shorter -- ``d_kv_intervene.py`` asserts that gap is constant, and
``t33_svip_gamma._forward_stats`` documents the same router convention).  The
compressed-side layer-1 query state and gist K,V come from the ordinary
``forward`` path; ``forward_with_gist`` bypasses nn.Module forward hooks
(``modeling_qwen3.py:303``), so the gist K,V must be read from the prefix cache,
not from a hook on the compression pass.

Everything except :func:`partial_prefill_kv` / :func:`probe_row` is pure numpy
and is unit-tested here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import t34_common as C  # noqa: E402

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "deviation statistic",
        "paper": "2405.16444",
        "what": "dev(k) is the norm of the difference of the query-projected "
                "attention OUTPUT, block-level.  It is neither of the paper's "
                "two deviations: KVD (Sec. 6.1) differences K,V per token, and "
                "CAD (Sec. 6.1) is the L-2 norm of the difference of the "
                "attention MATRIX.  Both are shape-locked to a common token "
                "axis, which a gist span and a raw span do not have.",
        "why": "gist_len != original_seq_len, so there is no token-to-gist "
               "correspondence to difference and no common probability-row "
               "shape; padding or pooling one side would make the number a "
               "product of the pooling (card, 'What is missing', item 2; "
               "digest 4.5).  The attention output is the nearest quantity "
               "that is defined on both sides, and it keeps CAD's motivation "
               "(query-side, post-softmax) -- but no number of the paper's is "
               "comparable with ours, and none is imported.",
    },
    {
        "method": "softmax normalisation",
        "paper": "2405.16444",
        "what": "The softmax is taken WITHIN the compared span, so the two "
                "sides are compared as span-conditional attention outputs of "
                "the same query.",
        "why": "The gist span and the raw span have different lengths and sit "
               "in different prefixes; a global softmax would make dev(k) a "
               "function of the other blocks' lengths.  Pre-registered here, "
               "before any evaluation row is read.",
    },
    {
        "method": "partial prefill context",
        "paper": "2405.16444",
        "what": "The candidate block is prefilled standalone at its LEDGER "
                "positions (system prefix excluded), i.e. as if it were a "
                "prefix -- which is also CacheBlend's own premise for its "
                "precomputed chunks (Sec. 6.1).",
        "why": "A layer-1 probe that first re-prefilled blocks 0..k-1 would "
               "cost a fraction of a prefill per BLOCK, not per decision, and "
               "would collapse back into the S15/S2 territory the digest "
               "already killed.",
    },
    {
        "method": "gradual filtering cascade (Sec. 6.3)",
        "paper": "2405.16444",
        "what": "Not ported.  We compute the layer-1 statistic once and stop; "
                "only its PRE-GATE (Insight 2's layer-to-layer Spearman rank "
                "correlation) is reproduced, on a pilot.",
        "why": "The cascade selects tokens to recompute inside one prefill; "
               "our action edits one block, so there is no r1 > r2 > ... "
               "schedule to run.  The pre-gate is what decides whether the "
               "layer-1 statistic is informative at all.",
    },
    {
        "method": "Insight-2 pre-gate estimand",
        "paper": "2405.16444",
        "what": "The adjacent-layer Spearman rank correlation is computed over "
                "the n_docs BLOCKS of one decision point (5-16 items), on the "
                "attention-output deviation above; the paper computes it over "
                "the per-TOKEN KV deviation of a whole prefill (thousands of "
                "items, Fig. rankcorr).",
        "why": "Our recompute unit is a block, not a token, so a token-level "
               "rank correlation has no operational meaning here.  The cost is "
               "a rank statistic on <=16 items: it is noisy per row, which is "
               "why the gate is read on the DISTRIBUTION over pilot rows "
               "(pregate_summary reports mean / median / p10, not one rho) and "
               "why the paper's own figure values are not imported as a "
               "threshold.",
    },
    {
        "method": "S0 twin arm",
        "paper": "2405.16444",
        "what": "The S0 twin is computed with the query state and the compared "
                "prefix of the FULL (uncompressed) arm, whose history blocks "
                "are replayed from the same token ids as the compressed grid "
                "rows; only the swap partner differs.  arm='full' therefore "
                "does not build the gist cache at all.",
        "why": "Digest 4.0: S0 is the same feature recomputed on the full arm.  "
               "A twin whose query state came from the compressed forward "
               "would be a function of the very thing it controls for.",
    },
    {
        "method": "locator chooser",
        "paper": "2405.16444",
        "what": "argmax_k dev(k) is a pure argmax; the frozen chooser used by "
                "the specificity control is reached through "
                "t34_qrhead.chooser_domain so its forward arm reproduces that "
                "argmax exactly.",
        "why": "t34_common.chooser_argmax carries select_k_star's abstain-when-"
               "nothing-is-positive rule, written for the witness-IDF mass; "
               "without the shift the control would score a different chooser "
               "than the table reports.",
    },
    {
        "method": "operating point r*",
        "paper": "2405.16444",
        "what": "The cost-equality rule and r* = 15% are NOT imported as a "
                "threshold; our trigger threshold is chosen by "
                "t34_common.fixed_rate_threshold inside inner CV folds.",
        "why": "r* was read off a quality-vs-ratio figure with no held-out "
               "split (card, 'Decision rule'), and the paper has no failure "
               "label and no detector metric of any kind.",
    },
]

DEFAULT_PROBE_LAYER = 1

#: Pre-registered risk orientations (+1 = higher is riskier for C->W), fixed
#: before any evaluation row is read.  Machine-readable copy:
#: ``configs/t34/orientations_qrhead.json`` (this unit shares one file).
ORIENTATION_RATIONALE: Dict[str, Tuple[int, str]] = {
    "cacheblend_dev_max": (
        1, "the whole premise: a block whose gist-side attention output is far "
           "from its own cheap raw re-prefill is a block the compressor "
           "damaged (2405.16444 Sec. 6.3, Insight 1)."),
    "cacheblend_dev_mean": (
        1, "the length-insensitive companion of dev_max; declared so that a "
           "'many blocks slightly off' regime is distinguishable from 'one "
           "block badly off'."),
    "cacheblend_dev_margin": (
        1, "dev_max - dev_second.  2405.16444 Sec. 6.3 reports ~10-15% of "
           "tokens carrying most of the deviation, and our best-k scan has "
           "42/81 flipping at a single k: concentrated damage is the regime "
           "the repair channel is built for."),
}


# --------------------------------------------------------------------------
# span-restricted attention output  (arXiv 2405.16444 CAD form, Sec. 6.1)
# --------------------------------------------------------------------------

def _repeat_kv(x: np.ndarray, n_heads: int) -> np.ndarray:
    """GQA expansion, mirroring modeling_qwen3.repeat_kv (:150)."""
    x = np.asarray(x, dtype=np.float64)
    n_kv = x.shape[0]
    if n_kv == n_heads:
        return x
    if n_heads % n_kv != 0:
        raise ValueError(f"n_heads {n_heads} not divisible by n_kv_heads {n_kv}")
    return np.repeat(x, n_heads // n_kv, axis=0)


def span_attention_output(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    span: Optional[Tuple[int, int]] = None,
    scale: Optional[float] = None,
) -> np.ndarray:
    """``Attn_l(q, span)``: softmax(q k^T * scale) over the span keys, times v.

    The output-space stand-in for arXiv 2405.16444 Sec. 6.1's attention
    deviation, restricted to one query row and one key span (DEVIATIONS,
    "deviation statistic").

    ``q`` is ``[H, d]`` (one query position -- the current query's last token),
    ``k`` / ``v`` are ``[H_kv, S, d]`` (GQA-expanded here).  Returns ``[H, d]``.

    The softmax is taken WITHIN the span; see DEVIATIONS ("softmax
    normalisation") for why, and why that choice is pre-registered.
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"q must be [H,d], got {q.shape}")
    if k.ndim != 3 or v.ndim != 3:
        raise ValueError(f"k,v must be [H_kv,S,d], got {k.shape} / {v.shape}")
    n_heads, head_dim = q.shape
    if span is not None:
        lo, hi = int(span[0]), int(span[1])
        if hi <= lo:
            raise ValueError(f"empty span {span}")
        k = k[:, lo:hi, :]
        v = v[:, lo:hi, :]
    if k.shape[1] == 0:
        raise ValueError("span selects zero keys")
    kk = _repeat_kv(k, n_heads)
    vv = _repeat_kv(v, n_heads)
    s = float(scale) if scale is not None else head_dim ** -0.5
    logits = np.einsum("hd,hsd->hs", q, kk) * s
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p = p / p.sum(axis=1, keepdims=True)
    return np.einsum("hs,hsd->hd", p, vv)


def deviation(out_a: np.ndarray, out_b: np.ndarray) -> float:
    """``|| Attn(q, A) - Attn(q, B) ||`` -- the Frobenius norm over (head, dim).

    Motivated by arXiv 2405.16444 Sec. 6.1's attention deviation (CAD, "the L-2
    norm of its difference with A^full"), but taken in OUTPUT space rather than
    on the attention matrix: the two compared spans have different lengths, so
    their probability rows have no common shape.  See DEVIATIONS, "deviation
    statistic" -- this is a declared departure, not the paper's CAD."""
    a = np.asarray(out_a, dtype=np.float64)
    b = np.asarray(out_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.linalg.norm(a - b))


def block_deviations(
    q: np.ndarray,
    gist_k: np.ndarray,
    gist_v: np.ndarray,
    gist_spans: Sequence[Tuple[int, int]],
    raw_kv: Sequence[Optional[Tuple[np.ndarray, np.ndarray]]],
    *,
    scale: Optional[float] = None,
) -> List[Optional[float]]:
    """``dev(k)`` for every candidate block (arXiv 2405.16444, ported to output
    space -- DEVIATIONS, "deviation statistic").

    ``raw_kv[k]`` is the probe layer's (K, V) of block ``k``'s partial prefill,
    each ``[H_kv, S_k, d]``; None (a block whose probe failed or that was
    dropped by the tail window) yields None -- never a sentinel.
    """
    out: List[Optional[float]] = []
    for i, span in enumerate(gist_spans):
        kv = raw_kv[i] if i < len(raw_kv) else None
        if kv is None or span is None or span[1] <= span[0]:
            out.append(None)
            continue
        try:
            a = span_attention_output(q, gist_k, gist_v, span=span, scale=scale)
            b = span_attention_output(q, kv[0], kv[1], span=None, scale=scale)
        except ValueError:
            out.append(None)
            continue
        out.append(deviation(a, b))
    return out


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------

def dev_argmax(dev: Sequence[Optional[float]]) -> Optional[int]:
    """Locator estimator ``argmax_k dev(k)`` (digest 4.5: the two uses of the
    CacheBlend deviation are pre-registered as SEPARATE estimands; arXiv
    2405.16444 Sec. 6.3 ranks candidates by deviation).  None when nothing is
    defined; ties resolve to the lowest index.

    PURE argmax, like the paper's ranking; the frozen
    ``t34_common.chooser_argmax`` used by the specificity control is reached
    through ``t34_qrhead.chooser_domain`` so that its forward arm reproduces
    THIS function exactly (see that helper's docstring)."""
    v = np.asarray([np.nan if d is None else float(d) for d in dev], dtype=np.float64)
    if v.size == 0 or not np.isfinite(v).any():
        return None
    return int(np.argmax(np.where(np.isfinite(v), v, -np.inf)))


TRIGGER_FEATURES = ("cacheblend_dev_max", "cacheblend_dev_mean",
                    "cacheblend_dev_margin")


def dev_trigger_scalars(dev: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """Trigger estimator ``max_k dev(k)`` plus two companions (digest 4.5, the
    second pre-registered estimand of the arXiv 2405.16444 deviation).  None =
    undefined (never 0.0, which would absorb a failed probe into "no damage")."""
    vals = np.asarray([d for d in dev if d is not None and np.isfinite(d)],
                      dtype=np.float64)
    if vals.size == 0:
        return {k: None for k in TRIGGER_FEATURES}
    srt = np.sort(vals)[::-1]
    return {
        "cacheblend_dev_max": float(srt[0]),
        "cacheblend_dev_mean": float(vals.mean()),
        "cacheblend_dev_margin": float(srt[0] - srt[1]) if srt.size >= 2 else None,
    }


def s0_swap_control(raw_kv: Sequence[Optional[Tuple[np.ndarray, np.ndarray]]],
                    q: np.ndarray, *, swap: int = 1,
                    scale: Optional[float] = None) -> List[Optional[float]]:
    """S0 twin (digest 4.5): the same deviation computed on the FULL prefix
    against ITSELF with one block swapped.

    Block ``k``'s reference side becomes block ``(k + swap) % n``'s raw span, so
    the statistic reads no compressed information at all.  Its orientation is
    fixed here, before any evaluation row is read: a candidate feature must
    beat this on the paired delta, exactly as the S2 kill required.
    """
    n = len(raw_kv)
    out: List[Optional[float]] = []
    for i in range(n):
        a = raw_kv[i]
        b = raw_kv[(i + int(swap)) % n] if n else None
        if a is None or b is None:
            out.append(None)
            continue
        try:
            oa = span_attention_output(q, a[0], a[1], scale=scale)
            ob = span_attention_output(q, b[0], b[1], scale=scale)
        except ValueError:
            out.append(None)
            continue
        out.append(deviation(oa, ob))
    return out


# --------------------------------------------------------------------------
# pre-gate: Insight 2, layer-to-layer rank correlation (Sec. 6.3)
# --------------------------------------------------------------------------

def layerwise_spearman(dev_by_layer: Sequence[Sequence[Optional[float]]]) -> Dict[str, Any]:
    """Spearman rank correlation of per-block deviation between ADJACENT layers
    (arXiv 2405.16444 Sec. 6.3, Insight 2: "Tokens with the highest KV
    deviations on one layer are likely to have the highest KV deviations on the
    next layer"; the paper's own numbers live in a figure and are not imported).

    ``dev_by_layer[l][k]`` = dev of block k measured at probe layer l.
    Returns the per-adjacent-pair r and the mean; blocks undefined on either
    layer are dropped from that pair and the surviving n is reported.
    """
    from scipy.stats import spearmanr  # scipy is installed on both boxes

    pairs: List[Dict[str, Any]] = []
    for l in range(len(dev_by_layer) - 1):
        a_raw, b_raw = dev_by_layer[l], dev_by_layer[l + 1]
        keep = [i for i in range(min(len(a_raw), len(b_raw)))
                if a_raw[i] is not None and b_raw[i] is not None
                and np.isfinite(a_raw[i]) and np.isfinite(b_raw[i])]
        if len(keep) < 3:
            pairs.append({"layers": [l, l + 1], "n": len(keep), "rho": None})
            continue
        a = np.array([a_raw[i] for i in keep], dtype=np.float64)
        b = np.array([b_raw[i] for i in keep], dtype=np.float64)
        rho = spearmanr(a, b).statistic
        pairs.append({"layers": [l, l + 1], "n": len(keep),
                      "rho": None if not np.isfinite(rho) else float(rho)})
    vals = [p["rho"] for p in pairs if p["rho"] is not None]
    return {
        "pairs": pairs,
        "mean_rho": float(np.mean(vals)) if vals else None,
        "n_pairs_defined": len(vals),
        "note": "pre-gate: if this is not high the cascade premise (Insight 2) "
                "does not hold on a trained compressor and the port stops here "
                "(digest 4.5, '跑不出来就早停').",
    }


def pregate_summary(per_row: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate the arXiv 2405.16444 Sec. 6.3 Insight-2 pre-gate over the
    pilot rows: the mean of each row's mean adjacent-layer rho."""
    means = [r["mean_rho"] for r in per_row
             if r.get("mean_rho") is not None and np.isfinite(r["mean_rho"])]
    return {
        "n_rows": len(per_row),
        "n_rows_defined": len(means),
        "mean_rho": float(np.mean(means)) if means else None,
        "median_rho": float(np.median(means)) if means else None,
        "p10_rho": float(np.percentile(means, 10)) if means else None,
    }


# --------------------------------------------------------------------------
# NPU side: partial prefill through embed_tokens + layers[0..probe_layer]
# --------------------------------------------------------------------------

def _import_rope() -> Any:  # pragma: no cover - torch-side helper
    """``modeling_qwen3.apply_rotary_pos_emb``; the harness puts <root>/python
    on sys.path, so both spellings are tried (same helper as unit U1a)."""
    last: Optional[BaseException] = None
    for mod in ("models.qwen3.modeling_qwen3", "python.models.qwen3.modeling_qwen3"):
        try:
            return __import__(mod, fromlist=["apply_rotary_pos_emb"]).apply_rotary_pos_emb
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise RuntimeError(f"cannot import apply_rotary_pos_emb: {last}")


def partial_prefill_kv(model: Any, token_ids: Sequence[int],
                       position_ids: Sequence[int],
                       probe_layer: int = DEFAULT_PROBE_LAYER
                       ) -> Tuple[np.ndarray, np.ndarray, float]:  # pragma: no cover
    """Partial prefill of arXiv 2405.16444 Sec. 6.2 steps (1)-(2), stopped at
    the probe layer: ``embed_tokens`` + ``layers[0..probe_layer-1]`` on one block's RAW
    token ids at its ledger positions and return ``(K, V, seconds)`` of layer
    ``probe_layer``, each ``[H_kv, S, d]``.

    Position convention: ``position_ids`` are the block's LEDGER positions
    ``offsets[k] .. offsets[k] + len(ids)`` (``eval_agent_history_c2kv``:2016),
    which is what the D-line slice-prefill uses; the block is prefilled
    standalone (see DEVIATIONS, "partial prefill context").
    """
    import torch

    t0 = time.perf_counter()
    device = model.device
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    pos = torch.tensor([list(position_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        hidden = model.model.embed_tokens(ids)
        pos_emb = model.model.rotary_emb(hidden, pos)
        for idx in range(int(probe_layer)):
            layer = model.model.layers[idx]
            hidden = layer(hidden, position_embeddings=pos_emb,
                           attention_mask=None, past_key_values=None)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        attn = model.model.layers[int(probe_layer)].self_attn
        normed = model.model.layers[int(probe_layer)].input_layernorm(hidden)
        shape = (*normed.shape[:-1], -1, attn.head_dim)
        k = attn.k_norm(attn.k_proj(normed).view(shape)).transpose(1, 2)
        v = attn.v_proj(normed).view(shape).transpose(1, 2)
        apply_rotary_pos_emb = _import_rope()
        cos, sin = pos_emb
        _, k = apply_rotary_pos_emb(k, k, cos, sin)
        k_np = k[0].float().cpu().numpy()
        v_np = v[0].float().cpu().numpy()
    return k_np, v_np, time.perf_counter() - t0


def probe_row(model: Any, ctx: Dict[str, Any], *,
              probe_layer: int = DEFAULT_PROBE_LAYER,
              s0_swap: bool = False) -> Dict[str, Any]:  # pragma: no cover
    """One decision point: gist-vs-raw layer-``probe_layer`` deviation per block
    -- the ported deviation statistic of arXiv 2405.16444 Sec. 6.1/6.3
    (output space; DEVIATIONS, "deviation statistic").

    ``ctx`` (built by the NPU driver from ``eval_agent_history_c2kv`` helpers)
    supplies ``q`` (the current query's last-token probe-layer query state from
    the COMPRESSED forward), ``gist_k`` / ``gist_v`` (the probe layer's K,V of
    the compressed prefix), ``gist_spans``, ``doc_ids``, ``offsets``, ``scale``.
    """
    raw_kv: List[Optional[Tuple[np.ndarray, np.ndarray]]] = []
    timers: List[Optional[float]] = []
    for i, ids in enumerate(ctx["doc_ids"]):
        if not ids:
            raw_kv.append(None)
            timers.append(None)
            continue
        start = int(ctx["offsets"][i])
        k, v, sec = partial_prefill_kv(model, ids, range(start, start + len(ids)),
                                       probe_layer)
        raw_kv.append((k, v))
        timers.append(round(sec, 6))
    q = np.asarray(ctx["q"], dtype=np.float64)
    dev = block_deviations(q, np.asarray(ctx["gist_k"]), np.asarray(ctx["gist_v"]),
                           ctx["gist_spans"], raw_kv, scale=ctx.get("scale"))
    row: Dict[str, Any] = {
        "qid": ctx["qid"],
        "session_id": C.session_of(ctx["qid"]),
        "arm": ctx.get("arm", "c2kv"),
        "probe_layer": int(probe_layer),
        "dev": dev,
        "layer1_probe_sec": timers,
        "layer1_probe_sec_total": round(sum(t for t in timers if t), 6),
        "n_docs": len(ctx["doc_ids"]),
    }
    if s0_swap:
        row["dev_s0_swap"] = s0_swap_control(raw_kv, q, scale=ctx.get("scale"))
    return row


def load_model_for_probe(model_path: str, device_type: str = "npu"):  # pragma: no cover - NPU only
    """Load a checkpoint with EAGER attention.  The probe reads K,V straight out
    of the prefix cache and recomputes one query state by hand, so no hook is
    needed -- which matters because ``forward_with_gist`` bypasses nn.Module
    forward hooks (``modeling_qwen3.py``:303)."""
    from t34_model_loader import load_model_and_tokenizer

    # harness loader: the repo's gist-aware class on the NPU (the compression
    # pass and the gist K/V do not exist on a plain AutoModelForCausalLM)
    model, tok, _mode = load_model_and_tokenizer(
        model_path, device_type=device_type, attn_impl="eager", mode="c2kv")
    return model, tok


def probe_ctx_from_sidecar(model: Any, tokenizer: Any, rec: Dict[str, Any], *,
                           probe_layer: int = DEFAULT_PROBE_LAYER,
                           arm: str = "c2kv",
                           max_doc_length: int = 768, max_doc_num: int = 16,
                           override_ratio: int = 8
                           ) -> Optional[Dict[str, Any]]:  # pragma: no cover
    """Assemble the ``ctx`` :func:`probe_row` consumes for one decision point.

    Ledger convention (``eval_agent_history_c2kv.py``:2016-2024):
    ``offsets[k] = system_length + sum(len(doc_ids[:k]))`` -- the LOGICAL raw
    positions, which is where the D-line slice-prefill puts a block's raw span.
    The query state is recomputed by hand from ``output_hidden_states`` rather
    than hooked, and the compared-side K,V are read out of the prefix cache.

    ``arm`` selects WHICH prefix the query state and the compared span come
    from.  ``"c2kv"`` builds the compressed prefix (``_build_tool_cache``,
    ``use_gist=True``) -- the candidate feature.  ``"full"`` builds the
    UNCOMPRESSED history prefix (``_build_full_history_cache_with_spans``, no
    ``use_gist``) -- the S0 twin, which must not read a single compressed
    quantity (digest 4.0: S0 is "the same feature recomputed on the full arm").
    Passing ``arm="full"`` and silently keeping the gist forward would make the
    control a function of the thing it is controlling for.
    """
    import torch
    import eval_agent_history_c2kv as HH
    import t34_qrhead as QR

    if arm not in ("c2kv", "full"):
        raise ValueError(f"unknown arm {arm!r}")
    plan = QR.sidecar_prefix_plan(rec, tokenizer, harness=HH,
                                  max_doc_length=max_doc_length,
                                  max_doc_num=max_doc_num)
    sys_t = torch.tensor([plan["system_ids"]], dtype=torch.long, device=model.device)
    system_cache, system_length, _ = HH._prefill_system(model, sys_t, "eager")
    use_gist = arm == "c2kv"
    if use_gist:
        rows = [QR._pad_row(ids, max_doc_length) for ids in plan["doc_ids"]]
        rows += [[-100] * max_doc_length for _ in range(max_doc_num - len(rows))]
        grid = torch.tensor(rows, dtype=torch.long)
        cache, _, gist_tokens, _, _, _ = HH._build_tool_cache(
            model, grid, system_cache, system_length, "eager", override_ratio)
        if gist_tokens <= 0:
            return None
        rel = HH._gist_spans_from_doc_lengths(plan["doc_lengths"], gist_tokens)
        gist_spans = [(system_length + a, system_length + b) for a, b in rel]
        logical = system_length + plan["history_logical_len"]
    else:
        # The full arm replays the SAME token ids as the compressed grid rows
        # (plan["doc_ids"], already chat-templated in the sidecar's decoded
        # text), block by block.  Going through
        # _build_full_history_cache_with_spans instead would re-apply the chat
        # template to text that already carries it, so the S0 prefix would
        # differ from the candidate prefix by more than compression.
        cache, hist_len, rel = system_cache, 0, []
        for ids in plan["doc_ids"]:
            ids_t = torch.tensor([ids], dtype=torch.long, device=model.device)
            cache, length, _ = HH._prefill_tokens_with_cache(
                model, ids_t, past_key_values=cache,
                past_length=system_length + hist_len, attn_impl="eager")
            rel.append((hist_len, hist_len + length))
            hist_len += length
        gist_spans = [(system_length + a, system_length + b) for a, b in rel]
        logical = system_length + hist_len
    q_ids = torch.tensor([plan["query_ids"]], dtype=torch.long, device=model.device)
    attention_mask = torch.ones((1, cache.get_seq_length() + q_ids.shape[1]),
                                dtype=torch.long, device=model.device)
    position_ids = torch.arange(logical, logical + q_ids.shape[1],
                                dtype=torch.long, device=model.device).unsqueeze(0)
    kwargs: Dict[str, Any] = {
        "input_ids": q_ids, "attention_mask": attention_mask,
        "position_ids": position_ids, "past_key_values": cache,
        "use_cache": False, "output_hidden_states": True, "logits_to_keep": 1,
    }
    if use_gist:
        kwargs["use_gist"] = True
    with torch.inference_mode():
        out = model(**kwargs)
        # hidden_states[l] is the INPUT of layer l; take the query's LAST token.
        hidden = out.hidden_states[int(probe_layer)][:, -1:, :]
        layer = model.model.layers[int(probe_layer)]
        attn = layer.self_attn
        normed = layer.input_layernorm(hidden)
        shape = (*normed.shape[:-1], -1, attn.head_dim)
        q = attn.q_norm(attn.q_proj(normed).view(shape)).transpose(1, 2)
        last_pos = position_ids[:, -1:]
        cos, sin = model.model.rotary_emb(hidden, last_pos)
        q, _ = _import_rope()(q, q, cos, sin)
        q_np = q[0, :, 0, :].float().cpu().numpy()
        gk = _cache_layer(cache, int(probe_layer), "keys")
        gv = _cache_layer(cache, int(probe_layer), "values")
    offsets, cur = [], system_length
    for ids in plan["doc_ids"]:
        offsets.append(cur)
        cur += len(ids)
    return {
        "qid": rec["qid"], "arm": arm, "scale": float(attn.scaling),
        "doc_ids": plan["doc_ids"], "offsets": offsets,
        "gist_spans": gist_spans, "gist_k": gk, "gist_v": gv, "q": q_np,
    }


def _cache_layer(cache: Any, layer_idx: int, field: str) -> np.ndarray:  # pragma: no cover
    """``[H_kv, S, d]`` of the prefix cache at one layer."""
    layer = cache.layers[layer_idx]
    return getattr(layer, field)[0].float().cpu().numpy()


# --------------------------------------------------------------------------
# scoring (here; pure numpy)
# --------------------------------------------------------------------------

def load_sidecar(path: Path) -> Dict[str, Dict[str, Any]]:
    """``results/t34/sidecar_<arm>.jsonl`` (unit U2's dump), keyed by qid."""
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["qid"]] = rec
    return out


def load_dev_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read the NPU probe's per-decision deviation rows (arXiv 2405.16444
    port), keyed by qid."""
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["qid"]] = rec
    return out


def locator_report(dev_rows: Dict[str, Dict[str, Any]],
                   frame: "C.FrozenFrame", *, key: str = "dev") -> Dict[str, Any]:
    """``argmax_k dev(k)`` scored with t34_common.locate_table against the
    25.0 % wrong-block floor, plus the inverted-score control."""
    import t34_qrhead as QR  # same unit; shared pre-registered controls

    truth: Dict[str, Optional[int]] = {}
    vectors: Dict[str, Sequence[float]] = {}
    hits: Dict[str, Optional[bool]] = {}
    n_scored = 0
    for qid in frame.cw_qids():
        entry = frame.witness_entry(qid)
        ref = None if not entry else entry.get("k_witness")
        truth[qid] = None if ref is None else int(ref)
        rec = dev_rows.get(qid)
        dev = None if rec is None else rec.get(key)
        if dev is None or ref is None:
            hits[qid] = None
            # An unprobed row still enters the control with an EMPTY vector so
            # the control's denominator is the same 93 as the headline table.
            vectors[qid] = []
            continue
        vec = [np.nan if d is None else float(d) for d in dev]
        vectors[qid] = vec
        n_scored += 1
        k = dev_argmax(dev)
        hits[qid] = None if k is None else bool(k == int(ref))

    out = {
        "table": C.locate_table(hits, label=f"cacheblend_{key}_argmax"),
        "inverted_control": C.inverted_score_control(
            {q: QR.chooser_domain(v) for q, v in vectors.items()}, truth)
        if n_scored else None,
        "random_matched_rate": QR.random_locator_control(
            {q: len(vectors[q]) for q in vectors}, truth),
        "legacy_k_first": C.locate_table(
            {q: (None if truth.get(q) is None else bool(truth[q] == 0))
             for q in frame.cw_qids()}, label="k_first"),
        "n_with_dev": n_scored,
    }
    return out


def feature_rows(dev_rows: Dict[str, Dict[str, Any]], frame: "C.FrozenFrame",
                 *, arm: str, key: str = "dev") -> List[Dict[str, Any]]:
    """Trigger features (``max_k dev``, arXiv 2405.16444) over the 161-row
    trigger frame; undefined -> None, never a sentinel (t34 rule 3)."""
    rows: List[Dict[str, Any]] = []
    for rec in frame.trigger_subset():
        qid = rec["qid"]
        src = dev_rows.get(qid)
        row: Dict[str, Any] = {"qid": qid, "session_id": C.session_of(qid), "arm": arm}
        dev = None if src is None else src.get(key)
        row.update(dev_trigger_scalars(dev or []))
        row["layer1_probe_sec_total"] = None if src is None else src.get("layer1_probe_sec_total")
        row["probe_layer"] = None if src is None else src.get("probe_layer")
        rows.append(row)
    return rows


def evaluate(rows: Sequence[Dict[str, Any]], frame: "C.FrozenFrame",
             orientations: Dict[str, int],
             *, s0_rows: Optional[Sequence[Dict[str, Any]]] = None,
             reps: int = 2000) -> Dict[str, Any]:
    """Default scoring contract of digest 4.0 applied to the arXiv 2405.16444
    trigger estimand: AP / AUROC vs the evaluation frame's own prevalence, a
    session-clustered bootstrap, the parse-failure baseline, the S0 swap twin
    as a paired delta, and the cap / length / nested-CV controls."""
    from t33_labels import parse_fail_baseline
    import t34_qrhead as QR  # same unit; shared pre-registered controls

    label = frame.label_by_qid
    c2kv = frame.c2kv_by_qid
    cap = frame.cap_tokens()
    by_qid = {r["qid"]: r for r in rows}
    s0_by_qid = {r["qid"]: r for r in (s0_rows or [])}

    base_qids, base_scores, base_y = [], [], []
    for rec in frame.trigger_subset():
        qid = rec["qid"]
        r = c2kv.get(qid) or {}
        base_qids.append(qid)
        base_scores.append(1.0 if parse_fail_baseline(
            str(r.get("prediction") or ""), bool(r.get("target_has_tool_call"))) else 0.0)
        base_y.append(int(label[qid]))
    s_b = np.array(base_scores, dtype=float)
    y_b = np.array(base_y, dtype=int)
    cl_b = C.session_clusters([C.session_of(q) for q in base_qids])

    out: Dict[str, Any] = {
        "baseline_parse_fail": {
            "n": int(y_b.size), "n_pos": int(y_b.sum()),
            "prevalence": C.prevalence(y_b),
            "ap": C.average_precision(s_b, y_b), "auroc": C.auroc(s_b, y_b),
            "ap_ci": C.clustered_bootstrap(C.average_precision, s_b, y_b, cl_b, reps=reps),
        },
        "features": {},
        "cap_tokens": int(cap),
    }
    for name in TRIGGER_FEATURES:
        orient = int(orientations.get(name, 1))
        qids, vals, ys = [], [], []
        for rec in frame.trigger_subset():
            qid = rec["qid"]
            v = (by_qid.get(qid) or {}).get(name)
            if v is None:
                continue
            qids.append(qid)
            vals.append(orient * float(v))
            ys.append(int(label[qid]))
        if not qids:
            out["features"][name] = {"n": 0, "n_pos": 0, "note": "no defined rows"}
            continue
        s = np.array(vals, dtype=float)
        y = np.array(ys, dtype=int)
        cl = C.session_clusters([C.session_of(q) for q in qids])
        entry: Dict[str, Any] = {
            "orientation": orient,
            "n": int(y.size), "n_pos": int(y.sum()),
            "prevalence_chance_ap": C.prevalence(y),
            "ap": C.average_precision(s, y), "auroc": C.auroc(s, y),
            "ap_ci": C.clustered_bootstrap(C.average_precision, s, y, cl, reps=reps),
            "auroc_ci": C.clustered_bootstrap(C.auroc, s, y, cl, reps=reps),
            "operating_point_at_baseline_fires": C.operating_point(s, y, int(s_b.sum())),
            "nested_cv_operating_point": QR.nested_cv_operating_point(s, y, cl),
            "by_cap_stratum": QR.stratified_metrics(
                s, y, [QR.cap_stratum(c2kv.get(q) or {}, cap) for q in qids]),
            "length_controls": QR.length_control_report(frame, qids, y),
        }
        s0 = np.array([
            (np.nan if (s0_by_qid.get(q) or {}).get(name) is None
             else orient * float((s0_by_qid[q])[name])) for q in qids], dtype=float)
        if np.isfinite(s0).all():
            entry["s0_full_arm_swap"] = {
                "ap": C.average_precision(s0, y), "auroc": C.auroc(s0, y),
                "delta_ap_ci": C.paired_delta_bootstrap(
                    C.average_precision, s, s0, y, cl, reps=reps),
            }
        else:
            entry["s0_full_arm_swap"] = {
                "n_defined": int(np.isfinite(s0).sum()),
                "note": "S0 twin undefined on some rows; NOT imputed",
            }
        out["features"][name] = entry
    return out


def cost_summary(dev_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """``layer1_probe_sec`` accounting -- the new GPU-sec column digest 4.5
    requires.  Reports the per-block and per-decision totals, no estimates."""
    per_block: List[float] = []
    per_row: List[float] = []
    for rec in dev_rows.values():
        ts = [t for t in (rec.get("layer1_probe_sec") or []) if t]
        per_block.extend(float(t) for t in ts)
        if rec.get("layer1_probe_sec_total") is not None:
            per_row.append(float(rec["layer1_probe_sec_total"]))
    return {
        "n_rows": len(dev_rows),
        "n_block_probes": len(per_block),
        "layer1_probe_sec_per_block_mean": float(np.mean(per_block)) if per_block else None,
        "layer1_probe_sec_per_block_median": float(np.median(per_block)) if per_block else None,
        "layer1_probe_sec_per_decision_mean": float(np.mean(per_row)) if per_row else None,
        "layer1_probe_sec_per_decision_total": float(np.sum(per_row)) if per_row else None,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI for the arXiv 2405.16444 port; see the RUNBOOK at the top of this
    module for the execution order and which stage runs where."""
    ap = argparse.ArgumentParser(
        description="CacheBlend layer-1 deviation port (arXiv 2405.16444)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pregate", help="[NPU] Insight-2 layer-wise Spearman pilot")
    p.add_argument("--root", default=".", help="worktree root (frozen battery + manifest) used to exclude evaluation rows from the pilot")
    p.add_argument("--sidecar", required=True)
    p.add_argument("--model", required=True, help="checkpoint path or hub id")
    p.add_argument("--arm", choices=("c2kv", "full"), default="c2kv")
    p.add_argument("--n-rows", type=int, default=24)
    p.add_argument("--probe-layers", type=int, default=8, help="layers 1..L_probe")
    p.add_argument("--device_type", default="npu", help="npu (default) | cpu")
    p.add_argument("--out", required=True)

    p = sub.add_parser("probe", help="[NPU] per-block layer-1 deviation -> jsonl")
    p.add_argument("--sidecar", required=True)
    p.add_argument("--model", required=True, help="checkpoint path or hub id")
    p.add_argument("--arm", choices=("c2kv", "full"), default="c2kv")
    p.add_argument("--probe-layer", type=int, default=DEFAULT_PROBE_LAYER)
    p.add_argument("--device_type", default="npu", help="npu (default) | cpu")
    p.add_argument("--s0-swap", action="store_true",
                   help="also emit the full-arm self-deviation control")
    p.add_argument("--out", required=True)

    p = sub.add_parser("score", help="locator S@k + trigger features")
    p.add_argument("--root", default=".")
    p.add_argument("--dev", required=True)
    p.add_argument("--dev-s0", default=None)
    p.add_argument("--orientations", default=None)
    p.add_argument("--features", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--reps", type=int, default=2000)

    args = ap.parse_args(argv)

    if args.cmd in ("pregate", "probe"):  # pragma: no cover - needs torch
        sidecar = load_sidecar(Path(args.sidecar))
        model, tok = load_model_for_probe(args.model, getattr(args, "device_type", "npu"))
        qids = sorted(sidecar)
        if args.cmd == "pregate":
            # The go/no-go gate is label-free (a between-layer rank correlation),
            # but a decision read off evaluation rows would still be a decision
            # taken on the evaluation frame: pilot rows are drawn from OUTSIDE
            # the 161-row trigger subset (t34_common.FrozenAssets), sorted for
            # determinism, and the count actually available is reported.
            import t34_common as _C
            _frame = _C.FrozenAssets(Path(args.root)).load()
            _eval = {r["qid"] for r in _frame.trigger_subset()}
            _pool = [q for q in qids if q not in _eval]
            print(json.dumps({"pregate_pilot_pool_outside_eval_frame": len(_pool),
                              "n_excluded_eval_rows": len(qids) - len(_pool),
                              "n_rows_requested": int(args.n_rows)}))
            qids = _pool[: int(args.n_rows)]
            per_row = []
            for qid in qids:
                dev_by_layer = []
                for layer in range(1, int(args.probe_layers) + 1):
                    ctx = probe_ctx_from_sidecar(model, tok, sidecar[qid],
                                                 probe_layer=layer, arm=args.arm)
                    if ctx is None:
                        break
                    dev_by_layer.append(probe_row(model, ctx, probe_layer=layer)["dev"])
                if len(dev_by_layer) >= 2:
                    row = layerwise_spearman(dev_by_layer)
                    row["qid"] = qid
                    per_row.append(row)
            report = {"per_row": per_row, "summary": pregate_summary(per_row),
                      "probe_layers": int(args.probe_layers),
                      "deviations": DEVIATIONS}
            sha = C.freeze_json(Path(args.out), report)
            print(f"pregate rows={len(per_row)} mean_rho={report['summary']['mean_rho']} "
                  f"sha256={sha}")
            return 0
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for qid in qids:
                ctx = probe_ctx_from_sidecar(model, tok, sidecar[qid],
                                             probe_layer=args.probe_layer,
                                             arm=args.arm)
                if ctx is None:
                    continue
                row = probe_row(model, ctx, probe_layer=args.probe_layer,
                                s0_swap=args.s0_swap)
                fh.write(json.dumps(row) + "\n")
                n += 1
        print(f"probe rows={n} -> {out_path}")
        return 0

    if args.cmd == "score":
        frame = C.FrozenAssets(Path(args.root)).load()
        dev_rows = load_dev_rows(Path(args.dev))
        orient = C.load_orientations(
            Path(args.orientations) if args.orientations
            else _HERE.parent / "configs/t34/orientations_qrhead.json")
        rows = feature_rows(dev_rows, frame, arm="c2kv")
        C.write_features_jsonl(Path(args.features), rows, context="t34_cacheblend_l1")
        s0_rows = None
        if args.dev_s0:
            s0_src = load_dev_rows(Path(args.dev_s0))
            s0_rows = feature_rows(s0_src, frame, arm="full", key="dev_s0_swap")
        report = {
            "locator": locator_report(dev_rows, frame),
            "trigger": evaluate(rows, frame, orient, s0_rows=s0_rows, reps=args.reps),
            "cost": cost_summary(dev_rows),
            "dev_sha256": C.sha256_file(Path(args.dev)),
            "deviations": DEVIATIONS,
        }
        sha = C.freeze_json(Path(args.report), report)
        tbl = report["locator"]["table"]
        print(f"locator S@k={tbl['s_at_k']} hits={tbl['hits']}/{tbl['n']} "
              f"p_vs_floor={tbl['p_vs_floor']:.3g} report_sha256={sha}")
        return 0

    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
