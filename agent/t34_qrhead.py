# -*- coding: utf-8 -*-
"""t34 U1b (a) -- QRHead / QRRetriever port (arXiv 2506.09944), digest 4.5.

RUNBOOK (execution order; "NPU" = needs torch + the checkpoint, "here" = this
Windows box, numpy only)

  1. NPU   python -m agent.t34_dump_sidecar ...                      (unit U2)
           -> results/t34/sidecar_c2kv.jsonl, sidecar_full.jsonl
  2. here  PYTHONIOENCODING=utf-8 python agent/t34_qrhead.py detection-set \\
             --root . --sidecar results/t34/sidecar_c2kv.jsonl \\
             --size 256 --out configs/t34/qrhead_detection_set.json
           Freezes the (leakage-proof) detection roster + its gold blocks D*.
  3. NPU   python agent/t34_qrhead.py capture \\
             --rows configs/t34/qrhead_detection_set.json --arm c2kv \\
             --model <ckpt> --sidecar results/t34/sidecar_c2kv.jsonl \\
             --query-proj gist --out results/t34/qrhead/attn_detect
           One prefill per row; writes doc_mass/class_mass npz per qid.
  4. here  PYTHONIOENCODING=utf-8 python agent/t34_qrhead.py detect-heads \\
             --capture-dir results/t34/qrhead/attn_detect \\
             --detection-set configs/t34/qrhead_detection_set.json \\
             --model-config <ckpt>/config.json --frac 0.01 \\
             --out configs/t34/qrhead_heads_fixed_joint.json
           FREEZES the head table (sha256) BEFORE any evaluation row is read.
  5. NPU   python agent/t34_qrhead.py capture --rows <eval roster> --arm c2kv \\
             --model <ckpt> --sidecar results/t34/sidecar_c2kv.jsonl \\
             --query-proj gist --calibrate --out results/t34/qrhead/attn_eval
           (--calibrate adds the q_null="N/A" prefill and times it)
     NPU   ... --arm full --out results/t34/qrhead/attn_eval_full   (S0 twin)
  6. here  PYTHONIOENCODING=utf-8 python agent/t34_qrhead.py score \\
             --root . --heads configs/t34/qrhead_heads_fixed_joint.json \\
             --capture-dir results/t34/qrhead/attn_eval \\
             --capture-dir-full results/t34/qrhead/attn_eval_full \\
             --features results/t34/features_qrhead.jsonl \\
             --report results/t34/qrhead_report.json
  7. here  PYTHONIOENCODING=utf-8 python agent/t34_qrhead.py diagnose \\
             --capture-dir results/t34/qrhead/attn_eval --root . \\
             --heads configs/t34/qrhead_heads_fixed_joint.json
           The cheap pre-diagnostic the digest demands: query-focused mass over
           {gist, raw tail, sink} on C->C vs C->W.  Run it before believing 6.

WIRING (steps 3/5): the forward-side capture uses unit U1a's
``agent/t34_attention.py`` -- ``KeyClassMap`` + ``AttentionRowCapture`` -- and
unit U2's sidecar.  It is imported LAZILY inside :func:`capture_rows` so this
module stays importable (and testable) on a torch-free box.  The prefix build
mirrors ``eval_agent_history_c2kv._rank_history_by_attention``
(system prefill -> ``_build_tool_cache`` -> query forward, router position
convention: ``position_ids`` continue at ``system_length + history_length``,
the *logical* (uncompressed) offset, while the cache is physically shorter).

Everything in stages 2/4/6/7 is torch-free numpy and is unit-tested here.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import t34_common as C  # noqa: E402
from d_witness_core import select_k_star, target_values  # noqa: E402

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "QRHead head detection (Eq. score_agg)",
        "paper": "2506.09944",
        "what": "Documents d_i are GIST spans of the compressed history, not "
                "natural-language passages; the inner sum runs over ~1/8 as "
                "many key positions.",
        "why": "That is the object our locator has to choose among.  QRscore "
               "is a span-mass statistic with no token-identity test, so a "
               "gist span is a legal document span (card, 'Signals used').",
    },
    {
        "method": "QRHead head detection set",
        "paper": "2506.09944",
        "what": "Detection set is 128-256 rows drawn from the frozen 900-row "
                "battery under three hard constraints (session-disjoint from "
                "the 72 C->W evaluation sessions, containing none of the 93 "
                "C->W qids, toolset-disjoint where feasible), and the head "
                "table is frozen with a sha256 before any evaluation row is "
                "touched.  The paper only holds out its detection examples.",
        "why": "digest 4.5 / card 'Expected pitfalls' 2: our D* comes from a "
               "gold-scored witness table, so the leakage guard has to be "
               "mechanical, not by inspection.",
    },
    {
        "method": "number of heads m",
        "paper": "2506.09944",
        "what": "m = round(frac * num_hidden_layers * num_attention_heads) "
                "with frac in {0.01, 0.02} read from the checkpoint's "
                "config.json, both declared as variants.  The paper fixes "
                "m = 16 for <10B models (which it calls ~1-2% of heads).",
        "why": "Qwen3-4B has a different L*H than the paper's models; the "
               "transferable statement in the card is the 1-2% fraction, not "
               "the literal 16.",
    },
    {
        "method": "QRRetriever ranking chooser",
        "paper": "2506.09944",
        "what": "khat is the paper's PURE argmax over R (no abstention).  The "
                "frozen chooser t34_common.chooser_argmax, which the specificity "
                "control uses, abstains when nothing scores above zero; that "
                "rule belongs to the non-negative witness-IDF mass and would "
                "abstain on nearly every row of the CALIBRATED arm, whose score "
                "R - R(q_null) is a signed difference.  The control is therefore "
                "fed chooser_domain(v) = v - min(v) + 1, a rank- and "
                "tie-preserving shift under which chooser_argmax reproduces "
                "khat exactly.",
        "why": "Otherwise the inverted-score control's FORWARD arm would not be "
               "the locator that is reported, and its McNemar p would compare "
               "two different estimators (digest 4.0 locator rule).",
    },
    {
        "method": "detection-set disjointness",
        "paper": "2506.09944",
        "what": "Digest 4.5 requires the roster to be session-disjoint from the "
                "72 C->W evaluation sessions and to contain none of the 93 C->W "
                "qids.  We strengthen it to the WHOLE 161-row evaluation frame "
                "(93 C->W + 68 C->C) and its sessions.",
        "why": "The trigger arm is scored on all 161 rows, so a C->C row in the "
               "roster would select the head table on a row that is later "
               "scored -- selection on evaluation rows.  Strengthening a "
               "constraint costs roster size (reported as `dropped` and `n`), "
               "never validity.",
    },
    {
        "method": "QRRetriever ranking",
        "paper": "2506.09944",
        "what": "We report S@1 (argmax_k) against the frozen D-line locator "
                "harness instead of top-k nDCG/Recall.",
        "why": "The repair action edits ONE block; there is no k>1 endpoint on "
               "our side (digest 4.5, 'this is a locator').",
    },
    {
        "method": "null-query calibration",
        "paper": "2506.09944",
        "what": "q_null = \"N/A\" is prefilled against the SAME compressed "
                "prefix (a ~2-token forward) and its cost is recorded as "
                "qnull_prefill_sec; both arms are reported.",
        "why": "The paper never prices calibration; our budget accounting "
               "requires the extra forward to be charged (card, 'Cost').",
    },
    {
        "method": "query span reconstruction",
        "paper": "2506.09944",
        "what": "The sidecar stores `query` as the templated current turn "
                "WITHOUT the generation prompt; the plan re-appends the "
                "generation-prompt suffix (derived from the tokenizer's own "
                "template) and truncates the rebuilt system prefill at the "
                "frozen recipe's max_system_length = 4096.",
        "why": "Eq. score_perdoc normalises by |q|, and the CacheBlend probe "
               "reads the query's LAST token; the harness's router builds the "
               "query span with add_generation_prompt=True "
               "(eval_agent_history_c2kv.py:2777).  system_length shifts every "
               "ledger offset, so its cap cannot be left to drift either.",
    },
    {
        "method": "null query structure",
        "paper": "2506.09944",
        "what": "q_null is the SAME chat turn with content \"N/A\" (plus the "
                "generation prompt), not the bare two-token string.",
        "why": "The paper swaps only the query; with a chat-templated prompt, "
               "tokenising the bare string would change the prompt structure "
               "too and the subtracted term would stop being the attention "
               "bias of the same shape.",
    },
    {
        "method": "null-query calibration wiring",
        "paper": "2506.09944",
        "what": "The prefix cache is cropped back to its frozen length before "
                "EVERY query forward (reset_cache_to).  Qwen3Attention.forward "
                "calls past_key_values.update() even with use_cache=False, so "
                "without the crop the q_null forward would run against "
                "prefix + the previous query.",
        "why": "Sec. 4.1 swaps ONLY the query; a q_null term computed against a "
               "different context is not the attention bias the subtraction is "
               "supposed to remove, and nothing about it would raise.",
    },
    {
        "method": "detection-set sensitivity",
        "paper": "2506.09944",
        "what": "Head-set stability is measured as top-m overlap across "
                "disjoint detection halves (the paper's Sec. 6.4 statistic) "
                "but with our n, not theirs; no number is imported.",
        "why": "No seeds/error bars anywhere in the paper (card, Open "
               "question 6): its gaps are directional priors only.",
    },
]

#: Key classes, mirroring unit U1a's ``t34_attention.CLASSES``.  Kept as a
#: literal so this module can be imported (and tested) without U1a.
CLASSES: Tuple[str, ...] = (
    "system_raw",
    "history_gist",
    "history_raw_tail",
    "current_query",
    "generated_so_far",
)

#: The paper's own head-count rule, as a fraction of L*H (card, "Decision rule
#: & thresholds": "approximately 1-2% of the total attention heads").
HEAD_FRACTIONS: Tuple[float, float] = (0.01, 0.02)
#: Pre-declared PRIMARY head-count fraction (the paper's <10B setting: 16 heads
#: of Qwen-2.5-7B / Llama-3.1-8B ~ 1 %); 0.02 is the declared SECONDARY variant.
#: The winner rule reads one primary; the secondary is reported beside it.
PRIMARY_HEAD_FRACTION: float = 0.01

NULL_QUERY_TEXT = "N/A"

#: Pre-registered risk orientation of every feature this module emits
#: (+1 = higher is riskier for C->W), with the reason, fixed BEFORE any
#: evaluation row is scored.  The machine-readable copy is
#: ``configs/t34/orientations_qrhead.json``; a test asserts they agree.
ORIENTATION_RATIONALE: Dict[str, Tuple[int, str]] = {
    "qrhead_r_max": (
        -1, "max_k R = does the query attend to ANY history block at all "
            "(card, 'Secondary port'); no mass anywhere is the risky state."),
    "qrhead_margin_top1_top2": (
        -1, "a flat top-2 means the query-focused heads cannot separate the "
            "blocks -- ambiguity is the risky state."),
    "qrhead_calib_delta": (
        -1, "max_k R_cal - max_k R_unc.  A large negative delta means the "
            "uncalibrated mass was attention bias that q_null reproduces, "
            "i.e. the score carried no query-specific information."),
    "qrhead_mass_gist": (
        -1, "2404.15574 Sec. 4.1: under hallucination the retrieval heads "
            "leave the content; mass ON the compressed history is the healthy "
            "state."),
    "qrhead_mass_raw_tail": (
        1, "mass migrating to the uncompressed tail = the gist blocks are not "
           "being used.  Declared as a guess, not a finding."),
    "qrhead_mass_sink": (
        1, "2404.15574 Sec. 4.1's attention-sink observation, stated there "
           "from figure inspection only -- we are putting the first number on "
           "it, so the direction is pre-registered, not read off."),
}


# --------------------------------------------------------------------------
# Eq. score_perdoc / score_agg  (arXiv 2506.09944 Sec. 3)
# --------------------------------------------------------------------------

def qr_score_perdoc(doc_mass: np.ndarray) -> np.ndarray:
    """Eq. ``score_perdoc`` of arXiv 2506.09944 Sec. 3.1.

    ``QRscore_h(q, d_i) = (1/|q|) * sum_{t_q in q} sum_{t_d in d_i} A_h^{t_q->t_d}``

    ``doc_mass`` is U1a's ``doc_mass_tensor()`` of shape ``[L, H, Q, D]``: the
    inner ``sum_{t_d in d_i}`` is already done per query row, so the equation
    reduces to the mean over the ``Q`` query rows.  Returns ``[L, H, D]``.
    """
    arr = np.asarray(doc_mass, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError(f"doc_mass must be [L,H,Q,D], got shape {arr.shape}")
    if arr.shape[2] == 0:
        raise ValueError("doc_mass has zero query rows: |q| = 0 is undefined")
    return arr.mean(axis=2)


def qr_score_perdoc_dense(
    attn: np.ndarray,
    query_rows: Sequence[int],
    doc_spans: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Reference (dense) implementation of Eq. ``score_perdoc``, 2506.09944.

    Sums the post-softmax weights ``attn[l, h, t_q, t_d]`` over each document
    span and averages over the listed query rows.  Used to check the reduced
    form in :func:`qr_score_perdoc` against the equation as written.
    """
    a = np.asarray(attn, dtype=np.float64)
    if a.ndim != 4:
        raise ValueError(f"attn must be [L,H,Q,K], got shape {a.shape}")
    rows = list(query_rows)
    if not rows:
        raise ValueError("query_rows is empty: |q| = 0 is undefined")
    out = np.zeros((a.shape[0], a.shape[1], len(doc_spans)), dtype=np.float64)
    for i, (start, end) in enumerate(doc_spans):
        if end <= start:
            continue
        out[:, :, i] = a[:, :, rows, start:end].sum(axis=(2, 3)) / len(rows)
    return out


def qr_score_agg(perdoc: np.ndarray, gold_blocks: Sequence[int]) -> np.ndarray:
    """Eq. ``score_agg`` of arXiv 2506.09944 Sec. 3.1.

    ``QRscore_h(q) = (1/|q|) sum_{d_i in D*} sum_{t_q} sum_{t_d} A_h^{t_q->t_d}``
    -- i.e. the per-document scores summed over the gold document set ``D*``.
    Returns ``[L, H]``.
    """
    p = np.asarray(perdoc, dtype=np.float64)
    gold = [int(g) for g in gold_blocks]
    if not gold:
        raise ValueError("empty gold block set D*")
    bad = [g for g in gold if g < 0 or g >= p.shape[2]]
    if bad:
        raise ValueError(f"gold blocks out of range for D={p.shape[2]}: {bad}")
    return p[:, :, gold].sum(axis=2)


def mean_head_table(agg_scores: Sequence[np.ndarray]) -> np.ndarray:
    """Average Eq. ``score_agg`` over the detection set T (2506.09944 Sec. 3.1:
    "then averaged over a detection set")."""
    stack = np.stack([np.asarray(a, dtype=np.float64) for a in agg_scores], axis=0)
    return stack.mean(axis=0)


def heads_from_fraction(num_hidden_layers: int, num_attention_heads: int,
                        frac: float) -> int:
    """``m = round(frac * L * H)``; the paper's rule is "approximately 1-2% of
    the total attention heads" (2506.09944, head-count paragraph).  L and H are
    read from the checkpoint's config.json by :func:`read_head_grid`."""
    total = int(num_hidden_layers) * int(num_attention_heads)
    m = int(round(float(frac) * total))
    return max(1, min(total, m))


def read_head_grid(config_path: Path) -> Tuple[int, int]:
    """(num_hidden_layers, num_attention_heads) from a checkpoint config.json.

    arXiv 2506.09944 fixes the head count as a FRACTION of the total head grid
    ("approximately 1-2% of the total attention heads"), so L and H must come
    from the checkpoint, never from memory (card, Migration recipe step 2)."""
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    try:
        return int(cfg["num_hidden_layers"]), int(cfg["num_attention_heads"])
    except KeyError as exc:  # pragma: no cover - config shape is fixed
        raise KeyError(f"{config_path}: missing {exc}") from exc


def select_heads(head_table: np.ndarray, m: int) -> List[Tuple[int, int]]:
    """Top-``m`` ``(layer, head)`` pairs by mean Eq. ``score_agg``.

    Ties break on (layer, head) ascending so the frozen list is deterministic.
    """
    t = np.asarray(head_table, dtype=np.float64)
    if t.ndim != 2:
        raise ValueError(f"head_table must be [L,H], got {t.shape}")
    pairs = [(int(l), int(h), float(t[l, h]))
             for l in range(t.shape[0]) for h in range(t.shape[1])]
    pairs.sort(key=lambda x: (-x[2], x[0], x[1]))
    return [(l, h) for l, h, _ in pairs[: int(m)]]


def head_overlap(a: Sequence[Tuple[int, int]], b: Sequence[Tuple[int, int]]) -> int:
    """Top-m overlap count -- the paper's Sec. 6.4 stability statistic."""
    return len(set(map(tuple, a)) & set(map(tuple, b)))


def retriever_scores(perdoc: np.ndarray,
                     heads: Sequence[Tuple[int, int]]) -> np.ndarray:
    """``R(q, d_i) = (1/|H_select|) sum_{h in H_select} QRscore_h(q, d_i)``
    (2506.09944 Sec. 4.1).  Returns ``[D]``."""
    p = np.asarray(perdoc, dtype=np.float64)
    if not heads:
        raise ValueError("empty head set H_select")
    idx_l = np.array([h[0] for h in heads], dtype=int)
    idx_h = np.array([h[1] for h in heads], dtype=int)
    return p[idx_l, idx_h, :].mean(axis=0)


def calibrate(r_query: np.ndarray, r_null: np.ndarray) -> np.ndarray:
    """``R(q, d_i) - R(q_null, d_i)`` with ``q_null = "N/A"`` (2506.09944
    Sec. 4.1, "Calibration"; attributed there to chen2025icr)."""
    return np.asarray(r_query, dtype=np.float64) - np.asarray(r_null, dtype=np.float64)


def khat(scores: Sequence[float]) -> Optional[int]:
    """Locator estimand of arXiv 2506.09944 Sec. 4.1 ("passages are ranked by
    R"), reduced to the top-1 our repair action can use.

    PURE argmax, as the paper writes it: there is no abstention rule in
    QRRetriever.  None only when the score vector is empty or all non-finite;
    ties resolve to the lowest index.

    This is deliberately NOT ``t34_common.chooser_argmax``: that chooser carries
    ``d_witness_core.select_k_star``'s "abstain when nothing scores above zero"
    rule, which is written for the non-negative witness-IDF mass and would
    abstain on almost every row of the CALIBRATED arm, whose score
    ``R - R(q_null)`` is a signed difference.  :func:`chooser_domain` is the
    bridge that lets the frozen chooser reproduce this function exactly.
    """
    v = np.asarray(scores, dtype=np.float64)
    if v.size == 0 or not np.isfinite(v).any():
        return None
    v = np.where(np.isfinite(v), v, -np.inf)
    return int(np.argmax(v))


def chooser_domain(scores: Sequence[float]) -> np.ndarray:
    """Rank-preserving map of a score vector into the FROZEN chooser's domain.

    ``t34_common.chooser_argmax`` / ``chooser_argmin`` (and therefore
    ``inverted_score_control``) abstain on a vector whose maximum is <= 0.  That
    rule belongs to the non-negative witness-IDF mass, not to a signed score
    such as the calibrated ``R - R(q_null)``; applied verbatim it would make the
    specificity control's FORWARD arm disagree with the S@k table this module
    actually reports.  Shifting by the minimum and adding 1 preserves every
    ranking and every tie, so

        chooser_argmax(chooser_domain(v)) == khat(v)

    holds for every vector, while ``chooser_argmin`` keeps its own documented
    abstention on a constant vector (an inverted constant score has no ranking
    to invert).  Non-finite entries stay non-finite.
    """
    v = np.asarray(scores, dtype=np.float64)
    fin = np.isfinite(v)
    if v.size == 0 or not fin.any():
        return v.astype(np.float64)
    return np.where(fin, v - float(v[fin].min()) + 1.0, np.nan)


def top1_top2_margin(scores: Sequence[float]) -> Optional[float]:
    """Top1 - top2 of ``R`` -- the ambiguity scalar of the card's "Secondary
    port (B-adapt)" list (arXiv 2506.09944).  None (never a sentinel) when
    D < 2."""
    v = np.asarray(scores, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return None
    part = np.sort(v)[::-1]
    return float(part[0] - part[1])


def trigger_scalars(r_unc: Sequence[float],
                    r_cal: Optional[Sequence[float]] = None) -> Dict[str, Optional[float]]:
    """The three secondary-port trigger features of the card ("Secondary port
    (B-adapt)"): ``max_k R``, the top1-top2 margin, and the
    calibrated-minus-uncalibrated gap.  None = undefined, never a sentinel."""
    u = np.asarray(r_unc, dtype=np.float64)
    out: Dict[str, Optional[float]] = {
        "qrhead_r_max": float(np.max(u)) if u.size and np.isfinite(u).any() else None,
        "qrhead_margin_top1_top2": top1_top2_margin(u),
        "qrhead_calib_delta": None,
    }
    if r_cal is not None:
        c = np.asarray(r_cal, dtype=np.float64)
        if c.size and u.size and np.isfinite(c).any() and np.isfinite(u).any():
            out["qrhead_calib_delta"] = float(np.max(c) - np.max(u))
    return out


# --------------------------------------------------------------------------
# the cheap pre-diagnostic the digest demands (digest 4.5, QRHead risks)
# --------------------------------------------------------------------------

def class_mass_profile(class_mass: np.ndarray,
                       heads: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    """Query-focused attention mass split over ``CLASSES``, averaged over the
    query rows and over ``H_select``, renormalised to sum to 1.

    This is the diagnostic digest 4.5 calls a hard prerequisite ("mass in
    {gist / raw tail / sink} on C->C vs C->W"); ``system_raw`` is our sink
    class (2404.15574 Sec. 4.1's attention-sink observation).

    DENOMINATOR: the renormalisation is over the CLASSIFIED mass, not over the
    whole attention row.  ``KeyClassMap`` can leave key positions unclassified
    (padding, boundary adjustments), so a profile alone cannot distinguish "the
    query attends to the gist blocks" from "almost nothing was classified and
    the gist blocks won the leftovers".  :func:`class_mass_coverage` returns
    that denominator and ``score_rows`` carries it per row -- read the two
    together."""
    cm = np.asarray(class_mass, dtype=np.float64)
    if cm.ndim != 4 or cm.shape[3] != len(CLASSES):
        raise ValueError(f"class_mass must be [L,H,Q,{len(CLASSES)}], got {cm.shape}")
    idx_l = np.array([h[0] for h in heads], dtype=int)
    idx_h = np.array([h[1] for h in heads], dtype=int)
    per = cm[idx_l, idx_h, :, :].mean(axis=(0, 1))  # -> [5]
    total = float(per.sum())
    if total <= 0:
        return {c: float("nan") for c in CLASSES}
    return {c: float(per[i] / total) for i, c in enumerate(CLASSES)}


def class_mass_coverage(class_mass: np.ndarray,
                        heads: Sequence[Tuple[int, int]]) -> Optional[float]:
    """Fraction of the query rows' attention mass that ``CLASSES`` accounts for.

    Each post-softmax attention row sums to 1 over ALL keys, so the sum of the
    five class masses is the share :func:`class_mass_profile` renormalises by.
    Reported alongside every profile so a near-empty denominator is visible
    rather than hidden by the renormalisation (digest 4.5 pre-diagnostic)."""
    cm = np.asarray(class_mass, dtype=np.float64)
    if cm.ndim != 4 or cm.shape[3] != len(CLASSES) or not heads:
        return None
    idx_l = np.array([h[0] for h in heads], dtype=int)
    idx_h = np.array([h[1] for h in heads], dtype=int)
    per = cm[idx_l, idx_h, :, :].mean(axis=(0, 1))
    total = float(per.sum())
    return total if np.isfinite(total) else None


def mass_split_diagnostic(profiles: Dict[str, Dict[str, float]],
                          labels: Dict[str, Optional[int]]) -> Dict[str, Any]:
    """C->C vs C->W distribution of the class mass split.  Reports n per side
    and the per-class mean/median; no test is run -- this is a look, not a
    claim (MDE at n=93 is 17-25 pp, ``t34_common.MDE_PP``)."""
    out: Dict[str, Any] = {}
    for name, want in (("cw", 1), ("cc", 0)):
        vals = [profiles[q] for q in profiles
                if labels.get(q) == want and q in profiles]
        out[name] = {"n": len(vals)}
        for cls in CLASSES:
            col = np.array([v[cls] for v in vals if np.isfinite(v.get(cls, np.nan))],
                           dtype=np.float64)
            out[name][cls] = {
                "mean": float(col.mean()) if col.size else None,
                "median": float(np.median(col)) if col.size else None,
                "n": int(col.size),
            }
    return out


# --------------------------------------------------------------------------
# detection set (leakage-proof by construction)
# --------------------------------------------------------------------------

def witness_gold_block_label(docs: Sequence[str],
                             tool_name: Optional[str],
                             target_args: Any) -> Optional[int]:
    """LABEL-side gold block ``D*`` for a detection row.

    Runs the frozen witness-IDF construction (``d_witness_core``,
    ``sum 1/df(v)`` over ``[gold tool name] + gold argument JSON leaves``) on
    the decoded doc grid.  This reads the reference action, so it is a
    ``*_label_*`` quantity: legal for offline head detection (2506.09944 is
    explicit that head detection needs gold *document* annotation and no gold
    answers) and forbidden inside any feature.
    """
    values = target_values(tool_name, target_args)
    if not values:
        return None
    return select_k_star(list(docs), values)


def _tool_names(sidecar_row: Dict[str, Any]) -> frozenset:
    names = set()
    for t in sidecar_row.get("tools") or []:
        if isinstance(t, dict):
            fn = t.get("function") if isinstance(t.get("function"), dict) else None
            name = (fn or t).get("name")
            if name:
                names.add(str(name))
    return frozenset(names)


def build_detection_set(
    frame: "C.FrozenFrame",
    sidecar: Dict[str, Dict[str, Any]],
    *,
    size: int = 256,
    seed: int = 20260905,
    require_toolset_disjoint: bool = True,
) -> Dict[str, Any]:
    """Detection roster for head selection, with the three hard constraints of
    digest 4.5 / card step 1.

    (a) sessions disjoint from the EVALUATION FRAME's sessions;
    (b) none of the 161 evaluation qids (93 C->W and 68 C->C);
    (c) toolset-disjoint from every evaluation session where feasible -- if
        constraint (c) cannot be met for a row it is DROPPED when
        ``require_toolset_disjoint`` and the shortfall is reported, never
        silently relaxed.

    Digest 4.5 words (a)/(b) as "the 72 C->W sessions" and "the 93 C->W qids".
    That is not sufficient here: the trigger arm is scored on the 161-row frame,
    whose 68 C->C rows would otherwise be allowed into the roster that SELECTS
    the head table -- selection on evaluation rows (t34 rule 6).  The constraint
    is therefore strengthened to the whole evaluation frame; see DEVIATIONS.

    Rows without a witness gold block (``k* is None``) are excluded: Eq.
    ``score_agg`` needs a non-empty ``D*``.
    """
    cw = set(frame.cw_qids())
    eval_qids = {r["qid"] for r in frame.trigger_subset()}
    eval_sessions = {C.session_of(q) for q in eval_qids}
    eval_tools: set = set()
    for q in eval_qids:
        row = sidecar.get(q)
        if row:
            eval_tools |= set(_tool_names(row))

    full_by_qid = frame.full_by_qid
    cand: List[Dict[str, Any]] = []
    dropped = {"in_cw": 0, "in_eval_frame_cc": 0, "eval_session": 0,
               "no_sidecar": 0, "no_gold_block": 0, "toolset_overlap": 0}
    for rec in frame.labels:
        qid = rec["qid"]
        if qid in cw:
            dropped["in_cw"] += 1
            continue
        if qid in eval_qids:
            dropped["in_eval_frame_cc"] += 1
            continue
        if C.session_of(qid) in eval_sessions:
            dropped["eval_session"] += 1
            continue
        row = sidecar.get(qid)
        if not row:
            dropped["no_sidecar"] += 1
            continue
        tools = _tool_names(row)
        overlaps = bool(tools & eval_tools)
        if require_toolset_disjoint and overlaps:
            dropped["toolset_overlap"] += 1
            continue
        fr = full_by_qid.get(qid) or {}
        gold = witness_gold_block_label(
            row.get("docs") or [],
            fr.get("target_tool_name"),
            _target_args_of(fr),
        )
        if gold is None:
            dropped["no_gold_block"] += 1
            continue
        cand.append({
            "qid": qid,
            "session_id": C.session_of(qid),
            "gold_blocks": [int(gold)],
            "n_docs": len(row.get("docs") or []),
            "toolset_disjoint": not overlaps,
        })

    # Session-level sampling so the roster never splits a session across the
    # detection/eval boundary (it cannot -- eval sessions are excluded -- but
    # it also keeps the roster's session count honest).
    rng = np.random.default_rng(seed)
    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for r in cand:
        by_session.setdefault(r["session_id"], []).append(r)
    sessions = sorted(by_session)
    order = rng.permutation(len(sessions))
    picked: List[Dict[str, Any]] = []
    for i in order:
        if len(picked) >= size:
            break
        picked.extend(sorted(by_session[sessions[i]], key=lambda r: r["qid"]))
    picked = picked[:size]
    picked.sort(key=lambda r: r["qid"])
    return {
        "rows": picked,
        "n": len(picked),
        "n_sessions": len({r["session_id"] for r in picked}),
        "n_candidates": len(cand),
        "dropped": dropped,
        "eval_sessions_excluded": sorted(eval_sessions),
        "require_toolset_disjoint": bool(require_toolset_disjoint),
        "seed": int(seed),
        "n_eval_qids_excluded": len(eval_qids),
        "constraints": [
            "session-disjoint from every session of the 161-row evaluation frame",
            "contains none of the 161 evaluation qids (93 C->W + 68 C->C)",
            "toolset-disjoint from every evaluation session"
            if require_toolset_disjoint else "toolset overlap ALLOWED (reported)",
        ],
    }


def _target_args_of(full_row: Dict[str, Any]) -> Any:
    """Gold argument object of a battery row, parsed out of ``target``.

    Label side only (``target`` is a guarded column)."""
    from d_strict_metric import parse_tool_call  # local import: label-side only

    parsed = parse_tool_call(str(full_row.get("target") or ""))
    if not parsed:
        return None
    return parsed.get("arguments")


def assert_detection_set_disjoint(det: Dict[str, Any], frame: "C.FrozenFrame") -> None:
    """Mechanical leakage guard (card, pitfall 2: 'the assert must be
    mechanical, not by inspection')."""
    eval_qids = {r["qid"] for r in frame.trigger_subset()}
    eval_sessions = {C.session_of(q) for q in eval_qids}
    qids = [r["qid"] for r in det["rows"]]
    bad_q = sorted(set(qids) & eval_qids)
    bad_s = sorted({C.session_of(q) for q in qids} & eval_sessions)
    if bad_q or bad_s:
        raise AssertionError(
            f"detection set leaks evaluation rows: qids={bad_q} sessions={bad_s}")


# --------------------------------------------------------------------------
# npz capture store (written on the NPU, read here)
# --------------------------------------------------------------------------

def capture_path(capture_dir: Path, qid: str) -> Path:
    """Path of one row's attention capture (one prefill of {D, q}, arXiv
    2506.09944 Sec. 3.1); ":" is not a legal Windows filename character."""
    return Path(capture_dir) / (qid.replace(":", "__") + ".npz")


def write_capture(capture_dir: Path, qid: str, *, doc_mass: np.ndarray,
                  class_mass: np.ndarray,
                  doc_mass_null: Optional[np.ndarray] = None,
                  meta: Optional[Dict[str, Any]] = None) -> Path:
    """Freeze one row's ``[L,H,Q,D]`` / ``[L,H,Q,5]`` masses (the inputs to Eq.
    score_perdoc of arXiv 2506.09944) plus the run meta (arm, query_proj)."""
    path = capture_path(capture_dir, qid)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "doc_mass": np.asarray(doc_mass, dtype=np.float32),
        "class_mass": np.asarray(class_mass, dtype=np.float32),
        "meta": np.frombuffer(
            json.dumps(meta or {}, sort_keys=True).encode("utf-8"), dtype=np.uint8),
    }
    if doc_mass_null is not None:
        arrays["doc_mass_null"] = np.asarray(doc_mass_null, dtype=np.float32)
    np.savez_compressed(path, **arrays)
    return path


def read_capture(capture_dir: Path, qid: str) -> Optional[Dict[str, Any]]:
    """Inverse of :func:`write_capture`; None when the row was never captured
    (missingness stays visible -- arXiv 2506.09944 needs no imputation)."""
    path = capture_path(capture_dir, qid)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        out: Dict[str, Any] = {
            "doc_mass": z["doc_mass"],
            "class_mass": z["class_mass"],
            "doc_mass_null": z["doc_mass_null"] if "doc_mass_null" in z.files else None,
            "meta": json.loads(bytes(z["meta"]).decode("utf-8")) if "meta" in z.files else {},
        }
    return out


# --------------------------------------------------------------------------
# stage 4: head detection
# --------------------------------------------------------------------------

def detect_heads(capture_dir: Path, detection_set: Dict[str, Any], m: int) -> Dict[str, Any]:
    """Eq. ``score_agg`` averaged over the detection set, ranked, top-m frozen.

    Also reports the paper's Sec. 6.4 stability statistic on two disjoint
    halves of OUR detection set (its own numbers are not imported)."""
    aggs: List[np.ndarray] = []
    used: List[str] = []
    missing: List[str] = []
    query_projs: set = set()
    for row in detection_set["rows"]:
        cap = read_capture(capture_dir, row["qid"])
        if cap is None:
            missing.append(row["qid"])
            continue
        perdoc = qr_score_perdoc(cap["doc_mass"])
        gold = [g for g in row["gold_blocks"] if 0 <= int(g) < perdoc.shape[2]]
        if not gold:
            missing.append(row["qid"])
            continue
        aggs.append(qr_score_agg(perdoc, gold))
        used.append(row["qid"])
        qp = (cap.get("meta") or {}).get("query_proj")
        if qp:
            query_projs.add(str(qp))
    if not aggs:
        raise RuntimeError("no detection captures found under " + str(capture_dir))
    if len(query_projs) > 1:
        raise RuntimeError(f"detection captures mix query_proj modes: {sorted(query_projs)}")
    table = mean_head_table(aggs)
    heads = select_heads(table, m)
    # arXiv 2506.09944 Sec. 6.4 measures top-m overlap across DISJOINT detection
    # subsets.  Splitting our roster by row index would put rows of the same
    # session on both sides and make the overlap optimistic, so the halves are
    # session-disjoint; with fewer than two sessions the statistic is reported
    # as undefined rather than computed on a split that is not disjoint.
    sess = sorted({C.session_of(q) for q in used})
    side = {s_: i % 2 for i, s_ in enumerate(sess)}
    idx_a = [i for i, q in enumerate(used) if side[C.session_of(q)] == 0]
    idx_b = [i for i, q in enumerate(used) if side[C.session_of(q)] == 1]
    if idx_a and idx_b:
        overlap: Optional[int] = head_overlap(
            select_heads(mean_head_table([aggs[i] for i in idx_a]), m),
            select_heads(mean_head_table([aggs[i] for i in idx_b]), m))
        split_note = (f"session-disjoint halves ({len(idx_a)} vs {len(idx_b)} rows, "
                      f"{len(sess)} sessions)")
    else:
        overlap = None
        split_note = ("undefined: fewer than 2 detection sessions, so no "
                      "disjoint split exists")
    return {
        "heads": [[int(l), int(h)] for l, h in heads],
        "m": int(m),
        "n_layers": int(table.shape[0]),
        "n_heads": int(table.shape[1]),
        "n_detection_rows": len(used),
        "n_detection_sessions": len({C.session_of(q) for q in used}),
        "missing_captures": missing,
        "query_proj": sorted(query_projs)[0] if query_projs else None,
        "head_table": [[float(v) for v in row] for row in table],
        "stability_top_m_overlap_halves": overlap,
        "stability_split": split_note,
        "detection_rows": [r["qid"] for r in detection_set["rows"]],
        "paper": "2506.09944 Eq. score_agg (Sec. 3.1)",
    }


# --------------------------------------------------------------------------
# stage 6: scoring (locator + trigger)
# --------------------------------------------------------------------------

def score_rows(capture_dir: Path, heads: Sequence[Tuple[int, int]],
               qids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Per-qid ``R(q, d_i)`` vectors (arXiv 2506.09944 Sec. 4.1), uncalibrated
    and -- when the ``q_null`` capture exists -- calibrated, plus the
    class-mass profile used by the digest 4.5 pre-diagnostic."""
    out: Dict[str, Dict[str, Any]] = {}
    for qid in qids:
        cap = read_capture(capture_dir, qid)
        if cap is None:
            continue
        perdoc = qr_score_perdoc(cap["doc_mass"])
        r_unc = retriever_scores(perdoc, heads)
        r_cal = None
        if cap["doc_mass_null"] is not None:
            perdoc_null = qr_score_perdoc(cap["doc_mass_null"])
            r_cal = calibrate(r_unc, retriever_scores(perdoc_null, heads))
        out[qid] = {
            "r_unc": r_unc,
            "r_cal": r_cal,
            "profile": class_mass_profile(cap["class_mass"], heads),
            "class_mass_coverage": class_mass_coverage(cap["class_mass"], heads),
            "meta": cap.get("meta") or {},
        }
    return out


def locator_tables(scored: Dict[str, Dict[str, Any]],
                   frame: "C.FrozenFrame") -> Dict[str, Any]:
    """S@k against the frozen witness k* on the 93 C->W qids, with the
    inverted-score control (t34_common.inverted_score_control)."""
    truth: Dict[str, Optional[int]] = {}
    vectors_unc: Dict[str, Sequence[float]] = {}
    vectors_cal: Dict[str, Sequence[float]] = {}
    for qid in frame.cw_qids():
        entry = frame.witness_entry(qid)
        ref = None if not entry else entry.get("k_witness")
        truth[qid] = None if ref is None else int(ref)
        rec = scored.get(qid)
        # Missing rows enter the control with an EMPTY vector so that the
        # inverted control's denominator is the same 93 as the headline table
        # (t34_common.inverted_score_control scores an empty vector as an
        # abstention, which locate_table counts in n and reports separately).
        vectors_unc[qid] = [] if rec is None else rec["r_unc"]
        # Every C->W qid gets an entry on BOTH arms, empty where the score is
        # undefined (row never captured, or captured without the q_null
        # forward).  An empty vector scores as an abstention, which
        # locate_table counts in n and reports separately, so the inverted
        # control's denominator stays the same 93 as the headline table on the
        # calibrated arm too.
        if rec is not None and rec["r_cal"] is not None:
            vectors_cal[qid] = rec["r_cal"]
        else:
            vectors_cal[qid] = []

    def _hits(vectors: Dict[str, Sequence[float]]) -> Dict[str, Optional[bool]]:
        hits: Dict[str, Optional[bool]] = {}
        for qid in frame.cw_qids():
            ref = truth.get(qid)
            vec = vectors.get(qid)
            if ref is None or vec is None:
                hits[qid] = None
                continue
            k = khat(vec)
            hits[qid] = None if k is None else bool(k == ref)
        return hits

    inv_unc = {q: chooser_domain(v) for q, v in vectors_unc.items()}
    out: Dict[str, Any] = {
        "uncalibrated": C.locate_table(_hits(vectors_unc), label="qrhead_uncal"),
        "inverted_control_uncalibrated": C.inverted_score_control(inv_unc, truth)
        if vectors_unc else None,
        "legacy_k_first": C.locate_table(
            {q: (None if truth.get(q) is None else bool(truth[q] == 0))
             for q in frame.cw_qids()}, label="k_first"),
        "random_matched_rate": random_locator_control(
            {q: len(vectors_unc[q]) for q in vectors_unc}, truth),
    }
    if any(len(np.asarray(v, dtype=float)) for v in vectors_cal.values()):
        inv_cal = {q: chooser_domain(v) for q, v in vectors_cal.items()}
        out["calibrated"] = C.locate_table(_hits(vectors_cal), label="qrhead_cal")
        out["inverted_control_calibrated"] = C.inverted_score_control(inv_cal, truth)
        # paired McNemar, calibrated vs uncalibrated, on the rows both scored
        hu, hc = _hits(vectors_unc), _hits(vectors_cal)
        b = sum(1 for q in hu if hu[q] is True and hc.get(q) is not True)
        c = sum(1 for q in hu if hc.get(q) is True and hu[q] is not True)
        out["mcnemar_cal_vs_uncal"] = {"b": b, "c": c, "p": C.mcnemar_exact(b, c)}
    return out


TRIGGER_FEATURES = ("qrhead_r_max", "qrhead_margin_top1_top2", "qrhead_calib_delta",
                    "qrhead_mass_gist", "qrhead_mass_raw_tail", "qrhead_mass_sink")


def feature_rows(scored: Dict[str, Dict[str, Any]],
                 frame: "C.FrozenFrame", *, arm: str) -> List[Dict[str, Any]]:
    """Trigger features of the card's "Secondary port (B-adapt)" (arXiv
    2506.09944), one row per qid of the 161-row trigger frame.  Undefined ->
    None, never a sentinel (t34 rule 3)."""
    rows: List[Dict[str, Any]] = []
    for rec in frame.trigger_subset():
        qid = rec["qid"]
        s = scored.get(qid)
        row: Dict[str, Any] = {"qid": qid, "session_id": C.session_of(qid), "arm": arm}
        if s is None:
            row.update({k: None for k in TRIGGER_FEATURES})
            row["query_proj"] = None
            rows.append(row)
            continue
        row.update(trigger_scalars(s["r_unc"], s["r_cal"]))
        prof = s["profile"]
        row["qrhead_mass_gist"] = _finite(prof.get("history_gist"))
        row["qrhead_mass_raw_tail"] = _finite(prof.get("history_raw_tail"))
        row["qrhead_mass_sink"] = _finite(prof.get("system_raw"))
        row["query_proj"] = (s.get("meta") or {}).get("query_proj")
        rows.append(row)
    return rows


def _finite(v: Optional[float]) -> Optional[float]:
    if v is None:
        return None
    v = float(v)
    return v if np.isfinite(v) else None


def evaluate_features(rows: Sequence[Dict[str, Any]],
                      frame: "C.FrozenFrame",
                      orientations: Dict[str, int],
                      *, s0_rows: Optional[Sequence[Dict[str, Any]]] = None,
                      reps: int = 2000) -> Dict[str, Any]:
    """Default scoring contract of digest 4.0 applied to the card's trigger
    features (arXiv 2506.09944, "Secondary port").

    AP / AUROC against the EVALUATION FRAME's own prevalence (never the
    900-frame 0.1033) with a session-clustered bootstrap, the parse-failure
    baseline, the S0 twin (the same feature recomputed on the full arm) as a
    paired delta, plus the pre-registered cap / length / nested-CV controls."""
    label = frame.label_by_qid
    c2kv = frame.c2kv_by_qid
    cap = frame.cap_tokens()
    by_qid = {r["qid"]: r for r in rows}
    s0_by_qid = {r["qid"]: r for r in (s0_rows or [])}

    out: Dict[str, Any] = {"features": {}, "cap_tokens": int(cap)}
    # parse-failure baseline (t33_labels.parse_fail_baseline), same frame
    from t33_labels import parse_fail_baseline
    base_qids, base_scores, base_y = [], [], []
    for rec in frame.trigger_subset():
        qid = rec["qid"]
        row = c2kv.get(qid) or {}
        base_qids.append(qid)
        base_scores.append(1.0 if parse_fail_baseline(
            str(row.get("prediction") or ""), bool(row.get("target_has_tool_call"))) else 0.0)
        base_y.append(int(label[qid]))
    y_b = np.array(base_y, dtype=int)
    s_b = np.array(base_scores, dtype=float)
    cl_b = C.session_clusters([C.session_of(q) for q in base_qids])
    out["baseline_parse_fail"] = {
        "n": int(y_b.size), "n_pos": int(y_b.sum()),
        "prevalence": C.prevalence(y_b),
        "ap": C.average_precision(s_b, y_b),
        "auroc": C.auroc(s_b, y_b),
        "ap_ci": C.clustered_bootstrap(C.average_precision, s_b, y_b, cl_b, reps=reps),
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
            "ap": C.average_precision(s, y),
            "auroc": C.auroc(s, y),
            "ap_ci": C.clustered_bootstrap(C.average_precision, s, y, cl, reps=reps),
            "auroc_ci": C.clustered_bootstrap(C.auroc, s, y, cl, reps=reps),
            "operating_point_at_baseline_fires": C.operating_point(
                s, y, int(s_b.sum())),
            "nested_cv_operating_point": nested_cv_operating_point(s, y, cl),
            "by_cap_stratum": stratified_metrics(
                s, y, [cap_stratum(c2kv.get(q) or {}, cap) for q in qids]),
            "length_controls": length_control_report(frame, qids, y),
        }
        # S0 twin: same feature on the FULL arm, same row subset, paired.
        s0_vals = []
        for qid in qids:
            v0 = (s0_by_qid.get(qid) or {}).get(name)
            s0_vals.append(np.nan if v0 is None else orient * float(v0))
        s0 = np.array(s0_vals, dtype=float)
        if np.isfinite(s0).all():
            entry["s0_full_arm"] = {
                "ap": C.average_precision(s0, y),
                "auroc": C.auroc(s0, y),
                "delta_ap_ci": C.paired_delta_bootstrap(
                    C.average_precision, s, s0, y, cl, reps=reps),
            }
        else:
            entry["s0_full_arm"] = {
                "n_defined": int(np.isfinite(s0).sum()),
                "note": "S0 twin undefined on some rows; NOT imputed",
            }
        out["features"][name] = entry
    return out


# --------------------------------------------------------------------------
# pre-registered controls shared by this unit's three ports
# (digest 4.0 default scoring contract + digest 4.5 per-entry controls)
# --------------------------------------------------------------------------

#: Objective used to pick the fire rate in the INNER folds.  Harm-free on
#: purpose: the closed-loop net coverage needs the 188-row harm manifest,
#: which has zero result rows (digest 1.3), so no harm-weighted objective can
#: be fitted today.  Declared before any evaluation row is scored.
INNER_FOLD_OBJECTIVE = "youden_J = TPR - FPR on the inner-fold rows"
FIRE_RATE_GRID: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)


def cap_stratum(c2kv_row: Dict[str, Any], cap_tokens: int) -> str:
    """``censored_at_cap`` stratum of a battery row.

    The digest's default contract says every table is reported in two rows,
    stratified by whether the compressed arm ran into the ``max_new_tokens``
    cap -- 128 in the frozen r2 recipe, where "protocol legal" degenerates
    into "finished within 128 tokens" (digest 1.4, S11)."""
    gen = c2kv_row.get("generated_tokens")
    if gen is None:
        return "unknown"
    return "censored" if int(gen) >= int(cap_tokens) else "uncensored"


def length_control_scores(frame: "C.FrozenFrame", qids: Sequence[str]) -> Dict[str, np.ndarray]:
    """Free S8 length/cap covariates on the SAME rows, as the 'a simple
    baseline dominates' control (digest 3.4).

    ``n_docs`` and ``kept_history_tokens`` cost nothing and are already logged;
    an attention feature has to earn its increment over them."""
    c2kv = frame.c2kv_by_qid
    out: Dict[str, List[float]] = {"n_docs": [], "kept_history_tokens": [],
                                   "generated_tokens": []}
    for qid in qids:
        row = c2kv.get(qid) or {}
        out["n_docs"].append(float(row.get("doc_chunks") or np.nan))
        out["kept_history_tokens"].append(float(row.get("kept_history_tokens") or np.nan))
        out["generated_tokens"].append(float(row.get("generated_tokens") or np.nan))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def length_control_report(frame: "C.FrozenFrame", qids: Sequence[str],
                          y: np.ndarray) -> Dict[str, Any]:
    """The free-covariate control, reported in BOTH orientations.

    These covariates have no pre-registered risk direction (unlike the features
    in ``ORIENTATION_RATIONALE``), and the stopping criterion they serve is
    "dominated by a simple baseline" (digest 3.4).  Scoring them in one
    arbitrary direction hides a covariate that separates the classes the other
    way round -- so both signs are printed and NEITHER is picked as the winner:
    a post-hoc sign flip is diagnostic-only (digest 2.1, the S0 `a_made_call`
    specimen)."""
    y = np.asarray(y, dtype=int)
    out: Dict[str, Any] = {
        "note": "both orientations reported; no winner is picked "
                "(post-hoc sign flips are diagnostic-only, digest 2.1)",
    }
    for nm, v in length_control_scores(frame, qids).items():
        ok = np.isfinite(v)
        if ok.sum() == 0 or y[ok].sum() in (0, int(ok.sum())):
            out[nm] = {"n": int(ok.sum()), "note": "undefined or one-class"}
            continue
        out[nm] = {
            "n": int(ok.sum()),
            "ap_pos": C.average_precision(v[ok], y[ok]),
            "auroc_pos": C.auroc(v[ok], y[ok]),
            "ap_neg": C.average_precision(-v[ok], y[ok]),
            "auroc_neg": C.auroc(-v[ok], y[ok]),
        }
    return out


def random_locator_control(n_blocks_by_qid: Dict[str, int],
                           truth: Dict[str, Optional[int]],
                           *, seed: int = 20260905, reps: int = 2000) -> Dict[str, Any]:
    """random@matched-rate: choose a block uniformly at random on the SAME rows.

    This is the empirical companion of the frozen 25.0 % wrong-block floor
    (``t34_common.LOCATE_FLOOR_WRONG_BLOCK``): the floor is a fixed reference
    value, this control is measured on our own block counts, so a locator that
    only reflects ``n_docs`` is visible."""
    rng = np.random.default_rng(seed)
    qids = [q for q in truth if truth[q] is not None and n_blocks_by_qid.get(q, 0) > 0]
    if not qids:
        return {"n": 0, "mean_s_at_k": None}
    hits = []
    for _ in range(reps):
        k = 0
        for q in qids:
            k += int(rng.integers(0, n_blocks_by_qid[q]) == truth[q])
        hits.append(k / len(qids))
    arr = np.asarray(hits)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {"n": len(qids), "reps": reps, "mean_s_at_k": float(arr.mean()),
            "ci95": [float(lo), float(hi)],
            "frozen_floor": C.LOCATE_FLOOR_WRONG_BLOCK}


def nested_cv_operating_point(scores: np.ndarray, y: np.ndarray, groups: np.ndarray,
                              *, rates: Sequence[float] = FIRE_RATE_GRID,
                              outer_folds: int = 5, inner_folds: int = 3,
                              seed: int = 20260905) -> Dict[str, Any]:
    """Fire-rate threshold chosen in INNER folds of a session-grouped nested CV.

    The fire rate is picked on inner folds only (objective:
    ``INNER_FOLD_OBJECTIVE``), the threshold is then set with
    ``t34_common.fixed_rate_threshold`` on the outer-training rows, and the
    fire decision is taken on held-out rows.  Nothing is chosen on the rows it
    is scored on (t34 rule 6)."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    oof = np.zeros(len(s), dtype=bool)
    chosen: List[float] = []
    for test_mask in C.grouped_folds(groups, outer_folds, seed):
        train = ~test_mask
        if train.sum() == 0 or test_mask.sum() == 0:
            continue
        best_rate, best_obj = rates[0], -np.inf
        inner_masks = C.grouped_folds(groups[train], inner_folds, seed + 1)
        for q in rates:
            objs = []
            for inner_test in inner_masks:
                inner_train = ~inner_test
                st = s[train][inner_train]
                sv, yv = s[train][inner_test], y[train][inner_test]
                if inner_train.sum() == 0 or inner_test.sum() == 0 or yv.sum() == 0:
                    continue
                thr = C.fixed_rate_threshold(st, q)
                fire = sv >= thr
                tpr = float((fire & (yv == 1)).sum()) / max(1, int((yv == 1).sum()))
                fpr = (float((fire & (yv == 0)).sum()) / int((yv == 0).sum())
                       if (yv == 0).sum() else 0.0)
                objs.append(tpr - fpr)
            if objs and float(np.mean(objs)) > best_obj:
                best_obj, best_rate = float(np.mean(objs)), q
        chosen.append(best_rate)
        thr = C.fixed_rate_threshold(s[train], best_rate)
        oof[test_mask] = s[test_mask] >= thr
    n_fire = int(oof.sum())
    cov = int((oof & (y == 1)).sum())
    fr = int((oof & (y == 0)).sum())
    return {
        "objective": INNER_FOLD_OBJECTIVE,
        "rates_grid": list(rates),
        "chosen_rates": chosen,
        "fires": n_fire,
        "coverage": cov, "n_pos": int((y == 1).sum()),
        "false_resets": fr, "n_neg": int((y == 0).sum()),
        "precision": (cov / n_fire) if n_fire else None,
        "false_reset_rate": (fr / int((y == 0).sum())) if (y == 0).sum() else None,
        "per_step_false_fire_rate": C.per_step_false_fire_rate(oof, y),
    }


def stratified_metrics(scores: np.ndarray, y: np.ndarray,
                       strata: Sequence[str]) -> Dict[str, Any]:
    """AP / AUROC per ``censored_at_cap`` stratum, with n and n_pos each --
    the two-row report the digest 4.0 default contract requires."""
    out: Dict[str, Any] = {}
    strata = list(strata)
    for name in sorted(set(strata)):
        m = np.array([s == name for s in strata], dtype=bool)
        if m.sum() == 0:
            continue
        out[name] = {
            "n": int(m.sum()), "n_pos": int(y[m].sum()),
            "prevalence_chance_ap": C.prevalence(y[m]),
            "ap": C.average_precision(scores[m], y[m]),
            "auroc": C.auroc(scores[m], y[m]),
        }
    return out


# --------------------------------------------------------------------------
# stage 3/5: forward capture (NPU; torch + U1a imported lazily)
# --------------------------------------------------------------------------

def _load_u1a():
    """Lazy import of unit U1a's attention capture (torch-dependent)."""
    import t34_attention  # noqa: WPS433 -- deliberately lazy
    return t34_attention


def _remove_capture(capture: Any, model: Any) -> None:
    """U1a's ``AttentionRowCapture.remove()`` takes no argument (it restores the
    modules it patched); tolerate a ``remove(model)`` signature too so this
    module does not break if that interface moves."""
    try:
        capture.remove()
    except TypeError:
        capture.remove(model)


def capture_rows(
    rows: Sequence[Dict[str, Any]],
    sidecar: Dict[str, Dict[str, Any]],
    *,
    build_prefix,
    out_dir: Path,
    arm: str,
    query_proj: str,
    calibrate_null: bool = False,
    attention_module: Any = None,
    layers: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Run one query prefill per row and dump the per-class / per-doc mass.

    ``build_prefix(qid)`` is the harness callable that returns a dict with
    ``model, tokenizer, query_ids, null_query_ids, system_len, doc_spans,
    raw_tail_span, query_span, generated_start, forward(input_ids)`` -- it is
    supplied by the CLI (which imports ``eval_agent_history_c2kv``) so this
    reduction is testable with a fake.  ``attention_module`` defaults to unit
    U1a's ``t34_attention``.

    The query rows are the WHOLE current-query span: ``query_mode`` is
    ``prefill_last_n`` with ``last_n = len(query_ids)`` and we assert the span
    fits, because Eq. ``score_perdoc`` averages over ``t_q in q`` and a
    truncated ``q`` silently changes the estimand.
    """
    A = attention_module if attention_module is not None else _load_u1a()
    stats = {"n": 0, "skipped": [], "qnull_prefill_sec": 0.0, "capture_sec": 0.0}
    for row in rows:
        qid = row["qid"]
        ctx = build_prefix(qid)
        if ctx is None:
            stats["skipped"].append(qid)
            continue
        q_ids = ctx["query_ids"]
        key_map = A.KeyClassMap(
            system_len=ctx["system_len"],
            doc_spans=ctx["doc_spans"],
            raw_tail_span=ctx.get("raw_tail_span"),
            query_span=ctx["query_span"],
            generated_start=ctx.get("generated_start"),
        )
        last_n = len(q_ids)
        q0, q1 = ctx["query_span"]
        assert last_n >= (q1 - q0), (
            f"{qid}: prefill_last_n={last_n} does not cover the query span "
            f"[{q0},{q1}) -- Eq. score_perdoc averages over the WHOLE query")
        cap = A.AttentionRowCapture(key_map, query_mode="prefill_last_n",
                                    last_n=last_n, layers=layers)
        cap.install(ctx["model"])
        t0 = time.perf_counter()
        try:
            ctx["forward"](q_ids)
            doc_mass = np.asarray(cap.doc_mass_tensor(), dtype=np.float32)
            class_mass = np.asarray(cap.class_mass_tensor(), dtype=np.float32)
        finally:
            _remove_capture(cap, ctx["model"])
        stats["capture_sec"] += time.perf_counter() - t0

        doc_mass_null = None
        qnull_sec = 0.0
        if calibrate_null:
            null_ids = ctx["null_query_ids"]
            cap2 = A.AttentionRowCapture(key_map, query_mode="prefill_last_n",
                                         last_n=len(null_ids), layers=layers)
            cap2.install(ctx["model"])
            t1 = time.perf_counter()
            try:
                ctx["forward"](null_ids)
                doc_mass_null = np.asarray(cap2.doc_mass_tensor(), dtype=np.float32)
            finally:
                _remove_capture(cap2, ctx["model"])
            qnull_sec = time.perf_counter() - t1
            stats["qnull_prefill_sec"] += qnull_sec

        write_capture(
            Path(out_dir), qid,
            doc_mass=doc_mass, class_mass=class_mass, doc_mass_null=doc_mass_null,
            meta={"qid": qid, "arm": arm, "query_proj": query_proj,
                  "n_docs": doc_mass.shape[3], "q_len": int(last_n),
                  "qnull_prefill_sec": round(qnull_sec, 6),
                  "null_query_text": NULL_QUERY_TEXT if calibrate_null else None},
        )
        stats["n"] += 1
    return stats


#: ``--max_system_length`` of the frozen r2 recipe, read off the harness argv
#: in ``agent/t34_dump_sidecar.py`` (``_harness_args``), not from memory.
FROZEN_MAX_SYSTEM_LENGTH = 4096


def generation_prompt_ids(tokenizer: Any, harness: Any) -> List[int]:
    """Token ids the chat template appends when ``add_generation_prompt=True``.

    The sidecar stores ``query`` as the decoded
    ``_chat_template_ids(tokenizer, current_messages)`` -- WITHOUT the
    generation prompt (``agent/t34_dump_sidecar.py``) -- while the harness's
    attention router builds its query span WITH it
    (``eval_agent_history_c2kv.py``:2777).  Eq. ``score_perdoc`` averages over
    ``t_q in q`` and the CacheBlend probe reads the query's LAST token, so the
    missing suffix would change both estimands.  It is derived here from the
    tokenizer's own template as the difference on a probe message, never
    hard-coded.
    """
    probe = [{"role": "user", "content": "x"}]
    without = list(harness._chat_template_ids(tokenizer, probe))
    with_gen = list(harness._chat_template_ids(tokenizer, probe,
                                               add_generation_prompt=True))
    if len(with_gen) <= len(without) or with_gen[: len(without)] != without:
        raise AssertionError(
            "chat template does not append a generation prompt as a suffix "
            f"(len {len(without)} -> {len(with_gen)}); the query span cannot be "
            "reconstructed from the sidecar text")
    return with_gen[len(without):]


def null_query_ids(tokenizer: Any, harness: Any) -> List[int]:
    """``q_null`` token ids for the calibration term of arXiv 2506.09944 Sec. 4.1.

    The paper swaps ONLY the query text for the context-free ``"N/A"``.  Our
    query is a whole chat turn (plus the generation prompt), so "only the query
    changes" means the same turn with ``"N/A"`` as its content -- tokenising the
    bare string would change the prompt STRUCTURE as well as the query and the
    subtracted term would no longer be the attention bias of the same shape.
    The paper prices calibration as a ~2-token prefill; ours is a few tokens
    more and the measured cost is recorded as ``qnull_prefill_sec`` either way.
    """
    return list(harness._chat_template_ids(
        tokenizer, [{"role": "user", "content": NULL_QUERY_TEXT}],
        add_generation_prompt=True))


def sidecar_prefix_plan(rec: Dict[str, Any], tokenizer: Any, *, harness: Any,
                        max_doc_length: int = 768,
                        max_doc_num: int = 16,
                        max_system_length: int = FROZEN_MAX_SYSTEM_LENGTH,
                        add_generation_prompt: bool = True) -> Dict[str, Any]:
    """Torch-free half of the prefix build: ids, ledger lengths and asserts.

    The sidecar's ``docs`` are the DECODED grid rows (post chat template, post
    ``max_doc_length`` truncation) -- the same texts the frozen witness table
    scored (``d_witness_core`` docstring, note 2).  Re-tokenising them is a
    round trip, so the plan asserts that each row's length matches the
    sidecar's ``doc_lengths``: a tokenizer round-trip that changes a length
    would shift every gist span silently.

    Two things the sidecar text does NOT carry are restored here rather than
    left to drift: the system prefill is truncated at the frozen recipe's
    ``max_system_length`` (otherwise ``system_length``, and with it every
    ledger offset, moves), and the query span gets the generation-prompt
    suffix the harness appends (:func:`generation_prompt_ids`).

    Returns ``system_ids``, ``doc_ids`` (padded grid rows are built later),
    ``query_ids``, ``doc_lengths`` and ``history_logical_len``
    (= ``sum(doc_lengths)``, the LOGICAL uncompressed history length that the
    router position convention counts with).
    """
    system_ids = harness._chat_template_ids(
        tokenizer, [{"role": "system", "content": rec.get("system_prompt") or ""}],
        tools=(rec.get("tools") or None), keep_bos=True,
        max_length=max_system_length)
    doc_ids: List[List[int]] = []
    for i, text in enumerate(rec.get("docs") or []):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) > max_doc_length:
            ids = ids[:max_doc_length]
        doc_ids.append(list(ids))
    declared = list(rec.get("doc_lengths") or [])
    if declared:
        got = [len(x) for x in doc_ids]
        if got != declared:
            raise AssertionError(
                f"{rec.get('qid')}: sidecar doc_lengths {declared} != re-tokenised "
                f"{got}; every gist span would be shifted (see docstring)")
    if len(doc_ids) > max_doc_num:
        raise AssertionError(
            f"{rec.get('qid')}: {len(doc_ids)} docs exceeds max_doc_num {max_doc_num}")
    query_ids = list(tokenizer(rec.get("query") or "",
                               add_special_tokens=False)["input_ids"])
    if not query_ids:
        raise AssertionError(f"{rec.get('qid')}: empty query -- |q| = 0 in Eq. "
                             "score_perdoc")
    gen_ids: List[int] = []
    if add_generation_prompt:
        gen_ids = generation_prompt_ids(tokenizer, harness)
        query_ids = query_ids + gen_ids
    return {
        "qid": rec.get("qid"),
        "system_ids": list(system_ids),
        "doc_ids": doc_ids,
        "doc_lengths": [len(x) for x in doc_ids],
        "history_logical_len": sum(len(x) for x in doc_ids),
        "query_ids": list(query_ids),
        "n_generation_prompt_tokens": len(gen_ids),
        "max_system_length": int(max_system_length),
        "dropped_docs": list(rec.get("dropped_docs") or []),
    }


def reset_cache_to(cache: Any, prefix_len: int) -> int:
    """Truncate a prefix cache back to ``prefix_len`` keys and return its length.

    ``Qwen3Attention.forward`` calls ``past_key_values.update(...)``
    unconditionally whenever a cache is passed (``modeling_qwen3.py``:277) --
    ``use_cache=False`` does NOT stop it -- so every query forward APPENDS its
    own K,V to the prefix cache.  The calibrated arm of arXiv 2506.09944
    Sec. 4.1 runs a SECOND forward with ``q_null`` against the same context;
    without this reset that forward would see the first query still in the
    prefix, so the subtracted term would not be ``R(q_null, d_i)`` on the same
    ``D`` and the calibrated score would be silently wrong (no error, no shape
    mismatch).  The same reset keeps the doc spans aligned, because the query
    keys land immediately after them.

    Raises rather than returning a wrong length: a cache that cannot be cropped
    must not be reused for a second forward.
    """
    n = int(cache.get_seq_length())
    prefix_len = int(prefix_len)
    if n == prefix_len:
        return n
    if n < prefix_len:
        raise RuntimeError(
            f"prefix cache shrank below its frozen length ({n} < {prefix_len})")
    crop = getattr(cache, "crop", None)
    if crop is None:
        raise RuntimeError(
            "prefix cache grew to {} keys (frozen prefix is {}) and exposes no "
            "crop(); the q_null calibration forward of arXiv 2506.09944 "
            "Sec. 4.1 cannot be run against the same context".format(n, prefix_len))
    crop(prefix_len)
    got = int(cache.get_seq_length())
    if got != prefix_len:
        raise RuntimeError(
            f"crop({prefix_len}) left the cache at {got} keys")
    return got


def _build_prefix_factory(args: argparse.Namespace,
                          sidecar: Dict[str, Dict[str, Any]]):  # pragma: no cover
    """Harness-side prefix builder mirroring ``_rank_history_by_attention``
    (``eval_agent_history_c2kv.py``:2703-2860).

    WIRING: system prefill -> ``_build_tool_cache`` (compressed arm) or
    ``_build_full_history_cache_with_spans`` (full arm / S0 twin) -> one query
    forward with ``position_ids`` continuing at
    ``system_length + history_logical_len`` (LOGICAL, uncompressed; the cache is
    physically shorter under gist) and ``use_gist=True`` on the compressed arm.
    ``forward_with_gist`` bypasses nn.Module forward hooks
    (``modeling_qwen3.py``:303), which is why U1a's capture monkeypatches the
    attention modules instead of registering hooks.
    """
    import torch
    import eval_agent_history_c2kv as HH
    from t34_model_loader import load_model_and_tokenizer

    # harness loader: the repo's gist-aware class on the NPU (a plain
    # AutoModelForCausalLM drops gist_q_proj, which --query-proj gist needs)
    model, tokenizer, _mode = load_model_and_tokenizer(
        args.model, device_type=getattr(args, "device_type", "npu"), attn_impl="eager",
        mode="c2kv")

    def build_prefix(qid: str) -> Optional[Dict[str, Any]]:
        rec = sidecar.get(qid)
        if not rec:
            return None
        plan = sidecar_prefix_plan(rec, tokenizer, harness=HH,
                                   max_doc_length=args.max_doc_length,
                                   max_doc_num=args.max_doc_num)
        sys_t = torch.tensor([plan["system_ids"]], dtype=torch.long, device=model.device)
        system_cache, system_length, _ = HH._prefill_system(model, sys_t, "eager")
        if args.arm == "full":
            # Replay the SAME token ids as the compressed grid rows.  The
            # sidecar's decoded docs already carry the chat template, so
            # _build_full_history_cache_with_spans would wrap them a second
            # time and the S0 prefix would differ from the candidate prefix by
            # more than compression.
            cache, hist_len, spans = system_cache, 0, []
            for ids in plan["doc_ids"]:
                ids_t = torch.tensor([ids], dtype=torch.long, device=model.device)
                cache, length, _ = HH._prefill_tokens_with_cache(
                    model, ids_t, past_key_values=cache,
                    past_length=system_length + hist_len, attn_impl="eager")
                spans.append((hist_len, hist_len + length))
                hist_len += length
            use_gist = False
            key_spans = [(system_length + a, system_length + b) for a, b in spans]
            logical = system_length + hist_len
        else:
            rows = [_pad_row(ids, args.max_doc_length) for ids in plan["doc_ids"]]
            rows += [[-100] * args.max_doc_length
                     for _ in range(args.max_doc_num - len(rows))]
            grid = torch.tensor(rows, dtype=torch.long)
            cache, _, gist_tokens, _, _, _ = HH._build_tool_cache(
                model, grid, system_cache, system_length, "eager", args.override_ratio)
            if gist_tokens <= 0:
                return None
            spans = HH._gist_spans_from_doc_lengths(plan["doc_lengths"], gist_tokens)
            key_spans = [(system_length + a, system_length + b) for a, b in spans]
            use_gist = True
            logical = system_length + plan["history_logical_len"]

        prefix_len = cache.get_seq_length()

        def forward(ids: Sequence[int]) -> None:
            # The previous forward appended its own K,V to this cache (see
            # reset_cache_to); the q_null arm must start from the SAME prefix.
            reset_cache_to(cache, prefix_len)
            ids_t = torch.tensor([list(ids)], dtype=torch.long, device=model.device)
            attention_mask = torch.ones(
                (1, prefix_len + len(ids)), dtype=torch.long,
                device=model.device)
            position_ids = torch.arange(
                logical, logical + len(ids), dtype=torch.long,
                device=model.device).unsqueeze(0)
            kwargs = {"input_ids": ids_t, "attention_mask": attention_mask,
                      "position_ids": position_ids, "past_key_values": cache,
                      "use_cache": False, "logits_to_keep": 1}
            if use_gist and args.query_proj == "gist":
                kwargs["use_gist"] = True
            with torch.inference_mode():
                model(**kwargs)

        q0 = prefix_len
        return {
            "model": model, "query_ids": plan["query_ids"],
            "null_query_ids": null_query_ids(tokenizer, HH),
            "system_len": system_length, "doc_spans": key_spans,
            "raw_tail_span": None,
            "query_span": (q0, q0 + len(plan["query_ids"])),
            "generated_start": q0 + len(plan["query_ids"]),
            "forward": forward,
        }

    return build_prefix


def _pad_row(ids: Sequence[int], width: int, pad: int = -100) -> List[int]:
    """Grid row padding, mirroring ``eval_agent_history_c2kv._pad``."""
    row = list(ids)[:width]
    return row + [pad] * (width - len(row))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _load_sidecar(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["qid"]] = rec
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI for the arXiv 2506.09944 port; see the RUNBOOK at the top of this
    module for the execution order and which stage runs where."""
    ap = argparse.ArgumentParser(
        description="QRHead / QRRetriever locator + trigger port (arXiv 2506.09944)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("detection-set", help="freeze the leakage-proof detection roster")
    p.add_argument("--root", default=".", help="worktree root (frozen assets)")
    p.add_argument("--sidecar", required=True, help="results/t34/sidecar_c2kv.jsonl")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--allow-toolset-overlap", action="store_true",
                   help="relax constraint (c); the shortfall is reported either way")
    p.add_argument("--out", required=True)

    p = sub.add_parser("capture", help="[NPU] one query prefill per row -> npz")
    p.add_argument("--rows", required=True, help="detection-set json or a qid list json")
    p.add_argument("--sidecar", required=True)
    p.add_argument("--model", required=True, help="checkpoint path or hub id")
    p.add_argument("--arm", choices=("c2kv", "full"), default="c2kv")
    p.add_argument("--query-proj", choices=("gist", "base"), required=True)
    p.add_argument("--calibrate", action="store_true", help="add the q_null='N/A' prefill")
    p.add_argument("--layers", default=None, help="comma-separated layer ids (default all)")
    p.add_argument("--max-doc-length", type=int, default=768)
    p.add_argument("--max-doc-num", type=int, default=16)
    p.add_argument("--override-ratio", type=int, default=8)
    p.add_argument("--device_type", default="npu", help="npu (default) | cpu")
    p.add_argument("--out", required=True)

    p = sub.add_parser("detect-heads", help="Eq. score_agg -> frozen head table")
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--detection-set", required=True)
    p.add_argument("--model-config", required=True, help="checkpoint config.json")
    p.add_argument("--frac", type=float, default=HEAD_FRACTIONS[0],
                   choices=list(HEAD_FRACTIONS))
    p.add_argument("--out", required=True)

    p = sub.add_parser("score", help="locator S@k + trigger features")
    p.add_argument("--root", default=".")
    p.add_argument("--heads", required=True)
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--capture-dir-full", default=None, help="S0 twin captures")
    p.add_argument("--orientations", default=None)
    p.add_argument("--features", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--reps", type=int, default=2000)

    p = sub.add_parser("diagnose", help="{gist, raw tail, sink} mass on C->C vs C->W")
    p.add_argument("--root", default=".")
    p.add_argument("--heads", required=True)
    p.add_argument("--capture-dir", required=True)
    p.add_argument("--out", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "detection-set":
        frame = C.FrozenAssets(Path(args.root)).load()
        sidecar = _load_sidecar(Path(args.sidecar))
        det = build_detection_set(
            frame, sidecar, size=args.size, seed=args.seed,
            require_toolset_disjoint=not args.allow_toolset_overlap)
        assert_detection_set_disjoint(det, frame)
        sha = C.freeze_json(Path(args.out), det)
        print(f"detection-set n={det['n']} sessions={det['n_sessions']} "
              f"dropped={det['dropped']} sha256={sha}")
        return 0

    if args.cmd == "capture":  # pragma: no cover - NPU only
        rows_obj = json.loads(Path(args.rows).read_text(encoding="utf-8"))
        rows = rows_obj["rows"] if isinstance(rows_obj, dict) else rows_obj
        sidecar = _load_sidecar(Path(args.sidecar))
        layers = None if not args.layers else [int(x) for x in args.layers.split(",")]
        stats = capture_rows(rows, sidecar,
                             build_prefix=_build_prefix_factory(args, sidecar),
                             out_dir=Path(args.out), arm=args.arm,
                             query_proj=args.query_proj,
                             calibrate_null=args.calibrate, layers=layers)
        print(json.dumps(stats, sort_keys=True))
        return 0

    if args.cmd == "detect-heads":
        det = json.loads(Path(args.detection_set).read_text(encoding="utf-8"))
        n_layers, n_heads = read_head_grid(Path(args.model_config))
        m = heads_from_fraction(n_layers, n_heads, args.frac)
        table = detect_heads(Path(args.capture_dir), det, m)
        table["frac"] = float(args.frac)
        table["frac_role"] = "primary" if float(args.frac) == PRIMARY_HEAD_FRACTION else "secondary"
        table["config_num_hidden_layers"] = n_layers
        table["config_num_attention_heads"] = n_heads
        table["detection_set_sha256"] = C.sha256_file(Path(args.detection_set))
        sha = C.freeze_json(Path(args.out), table)
        print(f"heads m={m} (frac={args.frac}, L*H={n_layers * n_heads}) "
              f"rows={table['n_detection_rows']} "
              f"half-overlap={table['stability_top_m_overlap_halves']}/{m} sha256={sha}")
        return 0

    if args.cmd in ("score", "diagnose"):
        frame = C.FrozenAssets(Path(args.root)).load()
        head_obj = json.loads(Path(args.heads).read_text(encoding="utf-8"))
        heads = [(int(l), int(h)) for l, h in head_obj["heads"]]
        qids = [r["qid"] for r in frame.trigger_subset()]
        scored = score_rows(Path(args.capture_dir), heads, qids)

        if args.cmd == "diagnose":
            diag = mass_split_diagnostic(
                {q: s["profile"] for q, s in scored.items()}, frame.label_by_qid)
            cov = [s["class_mass_coverage"] for s in scored.values()
                   if s.get("class_mass_coverage") is not None]
            diag["class_mass_coverage"] = {
                "n": len(cov),
                "mean": float(np.mean(cov)) if cov else None,
                "median": float(np.median(cov)) if cov else None,
                "min": float(np.min(cov)) if cov else None,
                "note": "denominator of the profile above: share of the query "
                        "rows' attention that CLASSES accounts for.  A small "
                        "value means the profile is a split of leftovers.",
            }
            diag["heads_sha256"] = C.sha256_file(Path(args.heads))
            diag["n_scored"] = len(scored)
            text = json.dumps(diag, indent=1, sort_keys=True)
            if args.out:
                C.freeze_json(Path(args.out), diag)
            print(text)
            return 0

        orient = C.load_orientations(
            Path(args.orientations) if args.orientations
            else _HERE.parent / "configs/t34/orientations_qrhead.json")
        rows = feature_rows(scored, frame, arm="c2kv")
        C.write_features_jsonl(Path(args.features), rows, context="t34_qrhead")
        s0_rows = None
        if args.capture_dir_full:
            s0_scored = score_rows(Path(args.capture_dir_full), heads, qids)
            s0_rows = feature_rows(s0_scored, frame, arm="full")
        report = {
            "locator": locator_tables(scored, frame),
            "trigger": evaluate_features(rows, frame, orient, s0_rows=s0_rows,
                                         reps=args.reps),
            "heads_sha256": C.sha256_file(Path(args.heads)),
            "n_scored": len(scored),
            "query_proj": sorted({(s.get("meta") or {}).get("query_proj")
                                  for s in scored.values()} - {None}),
            "deviations": DEVIATIONS,
        }
        sha = C.freeze_json(Path(args.report), report)
        loc = report["locator"]["uncalibrated"]
        print(f"locator S@k={loc['s_at_k']} hits={loc['hits']}/{loc['n']} "
              f"p_vs_floor={loc['p_vs_floor']:.3g} report_sha256={sha}")
        return 0

    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
