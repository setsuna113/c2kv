# -*- coding: utf-8 -*-
"""t34 U6 -- Verify-when-Uncertain cascade (t1, t*, p, t2) + GCN detection ceiling.

Migrates arXiv 2502.15845 ("Verify when Uncertain: Beyond Self-Consistency in
Black Box Hallucination Detection") per digest 4.8 (lines 1183-1191) and 4.12
(lines 1833-1841).  Two objects are transferred, neither of them a signal:

  1. **Algorithm 1** (paper Sec. 5, Fig. 1): a two-stage cascade parameterised by
     (t1, t*, t2) where t* is NOT chosen directly -- given t1 it is set so that
     exactly a fraction ``p`` of the calibration cheap-scores falls inside the
     uncertainty band [t1, t*].  ``p`` is the budget dial.
     s_cheap < t1            -> no fire (confident negative)
     s_cheap > t*            -> fire   (confident positive)
     t1 <= s_cheap <= t*     -> pay the expensive stage, fire iff s_exp >= t2.
  2. **The GCN ceiling** (paper Sec. 4.1 / 4.2.1, read directly in the local
     LaTeX at lines 831 and 857-861): a two-layer GCN trained with BCE on the
     RAW pairwise representation, with train/test drawn independently, read as
     "the upper bound of any scalar function of that representation".  Our
     detection side has never had a ceiling (digest Sec. 4.7 / card).

Our instantiation (digest "how to migrate"):
  * stage 1 = deterministic and free.  Rung 0 is the parse-failure-only
    indicator -- the T1 kill line's baseline that every candidate must beat --
    then a PRE-DECLARED ladder of free prefix scalars (S8/S9/S10).
  * stage 2 = a **sub-full action**: one single-block slice-prefill + regenerate,
    scored by whether the emitted action CHANGED.  Never a full-KV prefix
    (that would re-enter Tracy's cost-disqualified KV_VERIFIERS and VeriCache's
    C-K2 territory).
  * cost axis = the frozen GPU-sec sum (SPEC 5.6), never the paper's FLOPs ratio
    whose zero point already contains 10 generations + 100 NLI calls.

WIRING (bench face; this module never imports or edits anything under
benchmarks/ -- every bench-face function here is pure over dicts):
  * stage 1, pre-model  -> ``benchmarks/proxy.py::plan_repair`` (:827): the free
    prefix scalars (S8/S9/S10) are already in ``counts`` / the extract response;
    :func:`prefix_scalars_from_counts` takes that dict.
  * stage 1, post-answer -> ``benchmarks/proxy.py::RecoverState.check`` (:494):
    :func:`parse_fail_indicator` over the response message, with
    ``expects_call`` from :func:`expects_call_from_tools` on the REQUEST's tools.
  * stage 2 -> re-issue with a single-block slice prefill and compare
    :func:`canonical_action` of the two response messages (same canonicalisation
    as ``proxy.action_canonical`` :466 / ``_canon_calls`` :406).
  * BLOCKER (do not fix here): ``benchmarks/arms.py::Arm.validate`` refuses
    ``repair`` and ``recover`` together -- read at tmp/bench-recover
    benchmarks/arms.py:189 (the transfer card cites arms.py:74-75 of an older
    revision).  A cascade both chooses a block and decides whether to
    regenerate, so that line has to be relaxed deliberately.

PUBLIC API (this module is the single implementation of Algorithm 1 in t34;
``agent/t34_judge.py``'s cascade helpers are thin wrappers over these -- do not
fork the decision rule again):
  * :func:`band_upper_threshold` (cal_scores, t1, p) -> (t_star, p_realized, k)
  * :func:`cascade_decide` (s_cheap, :class:`CascadeThresholds`, s_expensive)
  * :func:`apply_cascade` (vectorised) / :func:`cascade` (LAZY stage 2, the
    deployable form: the expensive callback fires only inside the band)
  * :func:`thresholds_from_levels` (q1, q2, p -> t1, t*, t2)
  * :func:`fit_thresholds` (inner-fold level search, reports ``n_combos``) and
    :func:`fit_thresholds_direct` (ORACLE regime only)
  * :func:`cascade_nested_cv` (practical cross-session regime) /
    :func:`cascade_oracle_regime` (same-row regime)
  * :func:`p_sweep` (the whole deliverable table) and :data:`P_GRID`
  * :func:`trigger_metrics`, :func:`cascade_cost`, :func:`winner_rule`
  * :func:`expensive_block_choice_audit` (was the escalated block chosen
    gold-free?) and :func:`expensive_coverage_audit` (is stage-2 availability
    the label?) -- both must be clean before any p>0 number is reportable.
``CascadeThresholds`` accepts (t1, t_star, t2) alone -- the budget fields
default to NaN -- so a caller holding three absolute thresholds can reuse
:func:`cascade_decide` without inventing a budget it never measured.
Two semantics a wrapper must preserve rather than re-derive: at p = 0
:func:`band_upper_threshold` puts t* strictly BELOW t1 so the band is empty
(a wrapper returning t* = t1 would escalate every exact tie), and a band row
with no expensive score takes :data:`BAND_FALLBACK_FIRE` (False) and is
counted in ``n_band_unavailable``, never imputed.

RUNBOOK (execution order)
  # 1. HERE (Windows, CPU).  Single-stage (p=0) rows + the ladder audit + the
  #    parse-failure baseline + the length control.  Runs today on the frozen
  #    battery; p>0 rows are REFUSED unless a stage-2 table covering both
  #    classes exists (see step 2).
  PYTHONIOENCODING=utf-8 python agent/t34_cascade.py sweep \
      --root . --rungs 6 --out results/t34/cascade_sweep.json \
      --features_out results/t34/features_cascade_c2kv.jsonl
  #    (--rungs 6 declares the whole ladder; the run reports which rungs are
  #     actually usable -- on this face S9 and S10 are not, and saying so is
  #     part of the deliverable)
  # 2. NPU (server, GPU).  The stage-2 table.  d_corr.jsonl covers only the 93
  #    C->W rows, so its availability IS the label: the 68 C->C rows must be run
  #    through the SAME single-block slice-prefill + regenerate arm before any
  #    p>0 number is reportable.  Emit {qid, prediction, d_corr_slice_prefill_sec,
  #    generate_sec} for all 161 rows of the trigger frame.
  #    (until then:  --expensive results/bdf_pilot/d_r2/d_corr.jsonl
  #                  --allow_partial_expensive   -> family="diagnostic_partial")
  # 3. HERE.  Full p-sweep, both transfer regimes, cost column first:
  PYTHONIOENCODING=utf-8 python agent/t34_cascade.py sweep \
      --root . --rungs 6 --expensive results/t34/stage2_161.jsonl \
      --out results/t34/cascade_sweep.json
  # 4. HERE.  Detection ceiling.  Needs the per-doc sidecar (unit U2) because
  #    the frozen witness table exists for the 93 positives ONLY:
  PYTHONIOENCODING=utf-8 python agent/t34_cascade.py ceiling \
      --root . --sidecar results/t34/sidecar_c2kv.jsonl \
      --node_features query_lexical --out results/t34/cascade_ceiling.json
  # 5. HERE.  Tests:
  PYTHONIOENCODING=utf-8 python -m pytest agent/test_t34_cascade.py -q

ESTIMANDS.  The evaluation frame is the 161-row trigger subset (93 C->W = 1,
68 C->C = 0, 100 sessions); chance AP is its prevalence 0.578, NEVER the
900-frame base rate 0.1033.  coverage / precision / false-reset use the frame's
own denominators (93 / fires / 68).  Every threshold is chosen inside INNER
folds of session-grouped nested CV and the number of threshold combinations
tried is reported.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from t34_common import (  # noqa: E402
    FrozenAssets,
    check_docs_against_witness,
    FrozenFrame,
    auroc,
    average_precision,
    clustered_bootstrap,
    freeze_json,
    grouped_folds,
    load_decoded_docs,
    operating_point,
    paired_delta_bootstrap,
    permutation_band,
    prevalence,
    session_clusters,
    session_of,
    step_index,
    write_features_jsonl,
)
from t33_labels import cw_label, load_jsonl  # noqa: E402
from t33_spanmap import parse_tool_call  # noqa: E402
from d_witness_core import target_values, witness_scores  # noqa: E402

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "Algorithm 1 stage 1 (MPD(M_self))",
        "paper": "2502.15845 Sec. 4 / Sec. 5",
        "what": "s_cheap is NOT the mean-pairwise-disentailment of m=10 resamples; "
                "rung 0 is a deterministic parse-failure indicator and the higher "
                "rungs are free pre-generation prefix scalars (S8/S9/S10).",
        "why": "m=10 tau'=1.0 samples break the determinism gate (SPEC 5.5.9) and "
               "cost ~111 GPU-s/step against a 0.79-3.80 s recovery action; the "
               "digest (line 1189) fixes parse-failure-only as the starting point "
               "because it is the T1 kill line's baseline.",
    },
    {
        "method": "Algorithm 1 stage 2 (MPD(M_cross) from a verifier LLM)",
        "paper": "2502.15845 Sec. 4.2 / Sec. 5",
        "what": "s_expensive is a single-block slice-prefill + regenerate whose "
                "score is 1.0 iff the canonical action CHANGED, 0.0 otherwise "
                "(binary, so the t2 grid degenerates to one interior cut).",
        "why": "A second (larger) model verifying against the full KV history is "
               "Tracy's cost-disqualified KV_VERIFIERS and VeriCache's occupied "
               "cell (C-K2).  A sub-full slice prefill is outside both; the "
               "escalation is the action itself, not a second judgement.",
    },
    {
        "method": "Threshold calibration (400 validation examples, 5 seeds)",
        "paper": "2502.15845 Sec. 6",
        "what": "(t1, t2) are chosen as QUANTILE LEVELS in inner folds of a "
                "session-grouped nested CV over 100 session clusters, and the "
                "number of combinations tried is reported.",
        "why": "n=161 rows / 100 sessions, not 400 i.i.d. questions; three "
               "thresholds on this many clusters is the binding statistical risk "
               "the digest names (line 1841).  Quantile levels (not absolute "
               "values) are the knob so inner folds are comparable.",
    },
    {
        "method": "Two threshold-transfer regimes",
        "paper": "2502.15845 Sec. 6 (same-question resample vs different questions)",
        "what": "The oracle regime is calibration ON THE EVALUATION ROWS "
                "THEMSELVES; there is no independent redraw of the same qid.",
        "why": "Decoding is greedy and frozen -- an independent resample of the "
               "same qid does not exist on this face.  Reported as an optimistic "
               "bound, which is what the paper's oracle regime measures too.  "
               "CONFOUND to carry: the two regimes here differ in TWO ways, not "
               "one -- the practical regime also refits the stage-1 ladder's "
               "ECDF per outer-train fold while the oracle regime scores the "
               "whole-frame ECDF, so at n_rungs>1 the reported gap mixes "
               "threshold transfer with score normalisation (at n_rungs=1 "
               "s_cheap is the bare parse-failure indicator and the two "
               "normalisations coincide, so the gap is clean there).",
    },
    {
        "method": "Cost metric (relative additional FLOPs, p * N_v / N_t)",
        "paper": "2502.15845 Sec. 5",
        "what": "Cost is the measured frozen GPU-sec sum per SPEC 5.6 "
                "(system_prefill + full_prefill + tool_compress + blend + "
                "generate, plus slice_prefill + generate for escalated rows).",
        "why": "Their zero point is 'self-consistency only', which already "
               "contains 10 generations + 100 NLI calls; our zero point is ONE "
               "compressed generation, so their 'relative cost 0' is already "
               "~10x our entire per-step budget (card, Cost per decision).",
    },
    {
        "method": "GCN ceiling on M_self",
        "paper": "2502.15845 Sec. 4.1 (LaTeX line 831) and Sec. 4.2.1 (857-861)",
        "what": "Node set is the history blocks of one decision step (not m "
                "sampled answers); node features are a per-doc score vector and "
                "the adjacency is an RBF kernel over those features, not an "
                "entailment matrix.  Implemented in numpy with explicit "
                "backward (finite-difference tested) so it runs without torch.",
        "why": "We have no entailment estimator and no resampling; the raw "
               "representation whose ceiling we want is the per-doc score "
               "vector.  The construction (flexible model, BCE, independent "
               "train/test) is what transfers.",
    },
    {
        "method": "Ceiling label discipline",
        "paper": "2502.15845 Sec. 4.1",
        "what": "Gold-scored per-doc witness vectors are exposed only through "
                "``gold_witness_label_node_features`` (family='oracle'); the "
                "detector-side family is the gold-free query-lexical vector.",
        "why": "The frozen witness table is gold-scored AND exists for the 93 "
               "positives only -- its availability is the label.  A ceiling "
               "built on it is an oracle envelope, not a detection ceiling.",
    },
    {
        "method": "Threshold selection objective",
        "paper": "2502.15845 Sec. 6 (AUROC / AURAC)",
        "what": "(q1, q2) are chosen by Youden's J on the three-metric "
                "denominators (coverage rate - false-reset rate), not by AUROC "
                "or AURAC; ``--select_by`` also offers precision / coverage and "
                "the choice travels with every reported row.",
        "why": "Algorithm 1 emits a BINARY decision, so an AUROC over the "
               "cascade's own output is degenerate; the house contract scores "
               "triggers on coverage / precision / false-reset (digest 1.3).  "
               "The transfer gap between regimes is therefore reported on this "
               "objective as well as on coverage.",
    },
    {
        "method": "Stage-2 block choice",
        "paper": "2502.15845 Sec. 5 (the verifier is a second model, so the "
                 "paper has no block-choice analogue)",
        "what": "s_expensive comes from a repair dump whose prefilled block was "
                "chosen by a POSITION heuristic (``k_median`` on all 93 rows of "
                "the frozen d_corr.jsonl), never by the gold witness selector; "
                "``expensive_block_choice_audit`` checks this and reports None "
                "rather than assuming it when the witness table is absent.",
        "why": "A block index taken from the gold-scored witness table would "
               "make the whole escalation branch gold-conditioned, which is the "
               "other half of the card's label/feature-separation pitfall.",
    },
    {
        "method": "Stage-1 undefined-rung handling",
        "paper": "n/a (2502.15845's stage 1 is a single resampling scalar)",
        "what": "A rung that is undefined for SOME rows takes the neutral ECDF "
                "rank 0.5 inside s_cheap; the count is reported per rung in "
                "``Stage1Ladder.audit()['neutral_rank_imputations']``.  The "
                "EMITTED feature for such a row stays None -- the 0.5 exists "
                "only inside the composite score, never in the feature frame.",
        "why": "A composite score has to be defined on every row it ranks; "
               "dropping the row instead would change the denominators.  "
               "Rungs undefined on EVERY row are declared unusable and excluded "
               "outright rather than imputed (S9 and S10 on this face).",
    },
    {
        "method": "Feature-dump normalisation",
        "paper": "n/a (house contract)",
        "what": "``cascade_s_cheap`` in the emitted jsonl uses an ECDF fitted on "
                "the WHOLE frame (label-free but transductive); the scored path "
                "inside p_sweep refits the ladder on each outer-train fold.",
        "why": "The dump is one row per qid with no fold structure; any consumer "
               "that scores that column inherits the transductive normalisation "
               "and must say so.",
    },
    {
        "method": "Evaluation frame / chance level",
        "paper": "n/a (house contract; digest 4.0 default writes AUPRC against "
                 "the 900-frame base rate 0.1033)",
        "what": "Scoring runs on the 161-row trigger subset (93 C->W = 1, "
                "68 C->C = 0, 100 session clusters) and chance AP is that "
                "frame's own prevalence 0.578.",
        "why": "The 900-frame's other 739 rows are W->C / W->W steps where the "
               "full arm was already wrong: a fire there is neither a rescue nor "
               "a false reset, so they belong to no denominator in the "
               "three-metric contract.  ``trigger_subset`` is the frozen t34 "
               "convention; the 0.1033 base rate is NEVER used as chance here.",
    },
    {
        "method": "parse-failure indicator",
        "paper": "n/a (our T1 baseline, digest line 1189)",
        "what": "``parse_fail_indicator(prediction, expects_call)`` takes the "
                "expected-call flag as an ARGUMENT; t33_labels.parse_fail_baseline "
                "reads ``target_has_tool_call`` off the row.",
        "why": "Feature code may not read target fields (rule 4).  On the 161-row "
               "frame the two agree exactly: all 161 rows have "
               "target_has_tool_call=True, so EXPECTS_CALL_ALWAYS is equivalent "
               "there; in deployment the flag comes from the request's tools "
               "(``expects_call_from_tools``).",
    },
]

# ---------------------------------------------------------------------------
# stage 1: the pre-declared ladder (order is frozen; never reordered per run)
# ---------------------------------------------------------------------------

#: Every rung: (name, signal id, orientation, doc).  Orientation +1 = higher is
#: riskier.  Rung 0 is the baseline every candidate must beat.
STAGE1_LADDER: Tuple[Tuple[str, str, int, str], ...] = (
    ("cascade_parse_fail", "S11", +1,
     "compressed arm's own emission is unparseable while a call is expected"),
    ("s8_compression_ratio", "S8", +1,
     "actual_compression_ratio: how lossy this step's history became"),
    ("s8_doc_chunks", "S8", +1,
     "number of history blocks packed into the gist prefix"),
    ("s8_gist_per_doc", "S8", -1,
     "gist_tokens / doc_chunks: gist budget spent per history block"),
    ("s9_ledger_gap", "S9", +1,
     "position-ledger gap / frame delta -- single-frame on the battery face, "
     "free on the serving face only"),
    ("s10_hybrid_tail_frac", "S10", +1,
     "hybrid raw tail size / raw-compressed mix -- hybrid_top_k"),
)

#: Length controls the default scoring contract requires (digest 4.0 winner
#: rule, "relative length control").  ``input_tokens`` is the whole prefix,
#: ``kept_history_tokens`` the PRE-compression history, ``prompt_tokens`` the
#: current query segment alone -- a candidate must beat all three, not just the
#: shortest one.
LENGTH_CONTROLS: Tuple[Tuple[str, str, int], ...] = (
    ("ctl_input_tokens", "input_tokens", +1),
    ("ctl_history_tokens", "kept_history_tokens", +1),
    ("ctl_prompt_tokens", "prompt_tokens", +1),
)

#: The paper's budget sweep, as fixed by the digest (line 1189).
P_GRID: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)

#: Quantile levels the thresholds are searched over (levels, not absolute
#: values, so inner folds are comparable).  |Q1| * |Q2| combinations per fold.
Q1_GRID: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
Q2_GRID: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)

#: Declared fallback when a band row has no expensive score available.
#: Conservative: do NOT fire (the escalation could not be paid for).
BAND_FALLBACK_FIRE = False

#: The compressed arm's frozen GPU-sec sum (SPEC 5.6), base half.
BASE_COST_FIELDS = ("system_prefill_sec", "full_prefill_sec", "tool_compress_sec",
                    "blend_sec", "generate_sec")
#: The escalation's marginal half.
EXPENSIVE_COST_FIELDS = ("d_corr_slice_prefill_sec", "d_recompute_prefill_sec",
                         "generate_sec")


def expects_call_from_tools(tools: Optional[Sequence[Dict[str, Any]]]) -> bool:
    """Deployment form of the expected-call flag: a tool call is expected when
    the REQUEST offered tools.  Reads the request only -- never the target.

    2502.15845 has no analogue (its stage 1 is a resampling statistic); this is
    the digest's substitution (line 1189, "parse failure alone is the baseline").
    """
    return bool(tools)


def parse_fail_indicator(prediction: Optional[str], expects_call: bool) -> bool:
    """Rung 0 of s_cheap: the L1 parse-failure baseline (digest 1.3 / T1 line).

    Same parser as :func:`t33_labels.parse_fail_baseline` (``t33_spanmap``), but
    the expected-call flag is an ARGUMENT so no target field is read.  See
    DEVIATIONS["parse-failure indicator"].
    """
    if not expects_call:
        return False
    return not parse_tool_call(prediction or "")["parse_ok"]


def prefix_scalars(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Free pre-generation prefix scalars (S8/S9/S10) from the compressed arm's
    OWN battery row.  ``None`` where the face does not carry the signal -- never
    a sentinel (pitfall 9).

    Not from the paper: 2502.15845's stage-1 features are resampling statistics.
    These are the digest's declared free substitutes (line 1189).
    """
    ratio = row.get("actual_compression_ratio")
    chunks = row.get("doc_chunks")
    gist = row.get("gist_tokens")
    out: Dict[str, Optional[float]] = {
        "s8_compression_ratio": float(ratio) if ratio is not None else None,
        "s8_doc_chunks": float(chunks) if chunks is not None else None,
        "s8_gist_per_doc": (float(gist) / float(chunks))
        if (gist is not None and chunks) else None,
        # S9 is single-frame on the battery face (digest 1.4 table) -> undefined.
        "s9_ledger_gap": None,
        # S10 is structurally absent when hybrid is off (hybrid_top_k is None on
        # all 900 rows); a 0.0 here would be a structurally-zero anchor.
        "s10_hybrid_tail_frac": (float(row["hybrid_top_k"])
                                 if row.get("hybrid_top_k") else None),
    }
    return out


def prefix_scalars_from_counts(counts: Dict[str, Any],
                               extract: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[float]]:
    """Bench-face twin of :func:`prefix_scalars` (pure over the proxy's dicts).

    ``counts`` is the proxy's raw/compressed count block
    (``system_raw``/``history_raw``/``current_raw``/``compressed``, plus
    ``n_docs``/``dropped_docs``/``doc_packing``); ``extract`` is the
    ``/v1/c2kv/extract`` response (``gist_len``, ``original_seq_len``).  Wired at
    HOOK 1 (``proxy.py::plan_repair``).
    """
    extract = extract or {}
    n_docs = counts.get("n_docs")
    gist = extract.get("gist_len")
    orig = extract.get("original_seq_len")
    dropped = counts.get("dropped_docs")
    return {
        "s8_compression_ratio": (float(orig) / float(gist)) if (gist and orig) else None,
        "s8_doc_chunks": float(n_docs) if n_docs is not None else None,
        "s8_gist_per_doc": (float(gist) / float(n_docs)) if (gist and n_docs) else None,
        "s8_dropped_docs": float(dropped) if dropped is not None else None,
        "s9_ledger_gap": (float(counts["repair_frame_delta"])
                          if counts.get("repair_frame_delta") is not None else None),
        "s10_hybrid_tail_frac": (
            float(counts.get("history_raw") or 0)
            / float((counts.get("history_raw") or 0) + (counts.get("compressed") or 0))
            if ((counts.get("history_raw") or 0) + (counts.get("compressed") or 0)) else None),
    }


class Stage1Ladder:
    """s_cheap = parse_fail + 0.999 * mean(oriented ECDF rank of the added rungs).

    The ladder order is the module constant :data:`STAGE1_LADDER` and is
    PRE-DECLARED; ``n_rungs=1`` is the parse-failure-only baseline.  Ranks are
    the empirical CDF of the CALIBRATION rows only (``fit``), so scoring a test
    row never looks at the test distribution.  No weights are fitted, so the
    combiner adds no selection layer of its own.

    The additive shape keeps every parse-failure row strictly above every
    non-failure row: the ladder can never re-rank the baseline away.
    """

    def __init__(self, n_rungs: int = 1) -> None:
        if not 1 <= n_rungs <= len(STAGE1_LADDER):
            raise ValueError(f"n_rungs must be in 1..{len(STAGE1_LADDER)}")
        self.n_rungs = int(n_rungs)
        self.rungs = STAGE1_LADDER[1:self.n_rungs]  # continuous rungs only
        self._ecdf: Dict[str, np.ndarray] = {}
        self.usable: List[str] = []
        self.unusable: Dict[str, str] = {}
        #: rows whose value for a USABLE rung was undefined and therefore took
        #: the neutral rank 0.5 in the last :meth:`score` call.  This is the one
        #: in-score fallback in the module and it is counted, never hidden; the
        #: EMITTED feature for such a row stays ``None`` (see DEVIATIONS).
        self.neutral_counts: Dict[str, int] = {}
        self.n_scored: int = 0

    # -- fit / score ------------------------------------------------------
    def fit(self, rows: Sequence[Dict[str, Any]]) -> "Stage1Ladder":
        self._ecdf = {}
        self.usable = []
        self.unusable = {}
        for name, _sig, _ori, _doc in self.rungs:
            vals = np.array([_num(prefix_scalars(r).get(name)) for r in rows], dtype=float)
            ok = np.isfinite(vals)
            if ok.sum() == 0:
                self.unusable[name] = "undefined on this face (all None)"
                continue
            uniq = np.unique(vals[ok])
            if uniq.size < 2:
                self.unusable[name] = f"structurally constant (value={uniq[0]!r})"
                continue
            self._ecdf[name] = np.sort(vals[ok])
            self.usable.append(name)
        return self

    def score(self, rows: Sequence[Dict[str, Any]],
              expects_call: Sequence[bool]) -> np.ndarray:
        if len(rows) != len(expects_call):
            raise ValueError("rows and expects_call must align")
        base = np.array([float(parse_fail_indicator(r.get("prediction"), bool(e)))
                         for r, e in zip(rows, expects_call)], dtype=float)
        self.neutral_counts = {}
        self.n_scored = len(rows)
        if not self.usable:
            return base
        ranks = np.zeros((len(rows), len(self.usable)), dtype=float)
        orient = {name: ori for name, _s, ori, _d in STAGE1_LADDER}
        for j, name in enumerate(self.usable):
            ref = self._ecdf[name]
            vals = np.array([_num(prefix_scalars(r).get(name)) for r in rows], dtype=float)
            r = np.searchsorted(ref, vals, side="right") / float(len(ref))
            undef = ~np.isfinite(vals)
            r[undef] = 0.5                        # undefined -> neutral, COUNTED below
            if undef.any():
                self.neutral_counts[name] = int(undef.sum())
            if orient[name] < 0:
                r = 1.0 - r
            ranks[:, j] = r
        return base + 0.999 * ranks.mean(axis=1)

    def audit(self) -> Dict[str, Any]:
        return {"n_rungs": self.n_rungs,
                "declared": [n for n, _s, _o, _d in STAGE1_LADDER[:self.n_rungs]],
                "usable": list(self.usable),
                "unusable": dict(self.unusable),
                "orientations": {n: o for n, _s, o, _d in STAGE1_LADDER[:self.n_rungs]},
                "n_scored": int(self.n_scored),
                "neutral_rank_imputations": dict(self.neutral_counts)}


def _num(v: Any) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else float("nan")


def length_control_scores(rows: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Pre-registered length controls: each length alone as s_cheap
    (digest 4.0 winner rule, "relative length control")."""
    return {name: np.array([_num(r.get(field)) for r in rows], dtype=float)
            for name, field, _ori in LENGTH_CONTROLS}


# ---------------------------------------------------------------------------
# stage 2: the sub-full action (pure functions over emitted text / messages)
# ---------------------------------------------------------------------------

def _sorted_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sorted_keys(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_sorted_keys(v) for v in value]
    return value


def canonical_action(text: Optional[str]) -> Dict[str, Any]:
    """Canonical form of ONE emission, mirroring ``proxy.action_canonical`` (:466)
    and ``_canon_calls`` (:406): tool name + arguments with recursively sorted
    keys, plus the stripped text.  Battery-face input is raw prediction text, so
    the call is recovered with ``t33_spanmap.parse_tool_call``.
    """
    parsed = parse_tool_call(text or "")
    calls: List[Dict[str, Any]] = []
    if parsed["has_tool_call"]:
        calls.append({"name": parsed.get("name"),
                      "arguments": _sorted_keys(parsed.get("arguments") or {})})
    return {"tool_calls": calls, "text": (text or "").strip()}


def stage2_action_changed(pred_compressed: Optional[str],
                          pred_repaired: Optional[str]) -> Optional[float]:
    """s_expensive: 1.0 iff the single-block slice-prefill + regenerate changed
    the canonical ACTION, 0.0 if not, None if the repaired emission is missing.

    2502.15845's stage 2 is MPD(M_cross) from a verifier model; ours is the
    action itself (digest line 1839, "stage 2 is not a second judge but the
    action").  Both arguments are the COMPRESSED arm's own emissions -- no
    target, no gold, no full-arm field is read.
    """
    if pred_repaired is None:
        return None
    a = canonical_action(pred_compressed)["tool_calls"]
    b = canonical_action(pred_repaired)["tool_calls"]
    return 1.0 if a != b else 0.0


def load_expensive_table(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read the stage-2 table: ``{qid, prediction, d_corr_slice_prefill_sec,
    d_recompute_prefill_sec, generate_sec}``.  ONLY those keys are taken -- the
    D-line dump also carries scoring columns and they must not travel.

    DISCIPLINE (not enforceable from this file alone; use
    :func:`expensive_block_choice_audit`): the block the escalation prefilled
    must have been chosen WITHOUT gold.  The frozen ``d_r2/d_corr.jsonl`` picks
    ``k_median`` (a pure position heuristic) on all 93 rows, which is fine; a
    dump whose block index is the witness ``k_star``/``k_witness`` would make
    ``s_expensive`` gold-conditioned and every p>0 number an oracle number."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in load_jsonl(str(path)):
        rec: Dict[str, Any] = {"prediction": r.get("prediction")}
        for f in EXPENSIVE_COST_FIELDS:
            rec[f] = r.get(f)
        out[r["qid"]] = rec
    return out


def expensive_block_choice_audit(path: Path, frame: Optional[FrozenFrame] = None
                                 ) -> Dict[str, Any]:
    """Gold-leak guard for stage 2: which block did the escalation prefill?

    Reads ``d_corr_doc_index`` from the stage-2 dump and, when the frozen
    witness table is available, checks which selector explains it on EVERY
    checked row:

      * every index == ``k_median`` (a pure position heuristic) -> gold_free True;
      * every index == ``k_witness`` (the gold-scored selector) -> gold_free False;
      * neither explains all rows -> gold_free None, provenance unknown.

    Coincidental agreement with ``k_witness`` on some rows is NOT evidence of
    gold-conditioning (on the frozen ``d_corr.jsonl`` 19/93 rows have
    ``k_median == k_witness``), so the count is reported but never decides.
    The card's pitfall 4 (label/feature separation) is about the emitted action;
    this is the other half -- the block choice.
    """
    n = 0
    n_index = 0
    eq_witness = 0
    eq_median = 0
    n_checked = 0
    for r in load_jsonl(str(path)):
        n += 1
        idx = r.get("d_corr_doc_index")
        if idx is None:
            continue
        n_index += 1
        ent = frame.witness_entry(r["qid"]) if frame is not None else None
        if not ent:
            continue
        n_checked += 1
        if ent.get("k_witness") == idx:
            eq_witness += 1
        if ent.get("k_median") == idx:
            eq_median += 1
    if n_checked == 0:
        gold_free, explained_by = None, None
    elif eq_median == n_checked:
        gold_free, explained_by = True, "k_median (position heuristic, gold-free)"
    elif eq_witness == n_checked:
        gold_free, explained_by = False, "k_witness (gold-scored witness selector)"
    else:
        gold_free, explained_by = None, None
    return {
        "n_rows": n, "n_with_block_index": n_index, "n_checked_against_witness": n_checked,
        "n_equal_k_witness_gold": eq_witness, "n_equal_k_median_gold_free": eq_median,
        "gold_free": gold_free, "explained_by": explained_by,
        "note": "gold_free is True only when the GOLD-FREE selector (k_median) "
                "explains the block index on every checked row, and False only "
                "when the gold witness selector does.  Partial agreement with "
                "k_witness is coincidence (19/93 on the frozen d_corr.jsonl) and "
                "decides nothing; None = provenance unknown, never assume "
                "gold-free.",
    }


def expensive_coverage_audit(qids: Sequence[str], y: np.ndarray,
                             available: Sequence[bool]) -> Dict[str, Any]:
    """Refuse silently-labelled escalation.

    The frozen ``d_corr.jsonl`` exists for the 93 C->W rows ONLY: if stage-2
    scores are available for one class and not the other, availability IS the
    label and every p>0 number is contaminated.  Returns the per-class coverage
    and ``ok`` = both classes covered at the same rate.
    """
    y = np.asarray(y, dtype=int)
    av = np.asarray(available, dtype=bool)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    cov_pos = float(av[y == 1].mean()) if n_pos else float("nan")
    cov_neg = float(av[y == 0].mean()) if n_neg else float("nan")
    balanced = bool(n_pos and n_neg and abs(cov_pos - cov_neg) < 1e-9)
    ok = bool(balanced and av.sum() > 0)
    if ok:
        reason = None
    elif av.sum() == 0:
        reason = "no stage-2 scores at all: the escalation branch is untestable"
    else:
        reason = "availability differs by class: availability is the label"
    return {"n": len(qids), "n_available": int(av.sum()),
            "coverage_pos": cov_pos, "coverage_neg": cov_neg,
            "balanced": balanced, "ok": ok, "reason": reason}


# ---------------------------------------------------------------------------
# Algorithm 1
# ---------------------------------------------------------------------------

@dataclass
class CascadeThresholds:
    """(t1, t*, t2) plus the budget bookkeeping.

    ``p_target``/``p_realized`` default to NaN so a caller that already holds
    three absolute thresholds (e.g. the thin wrapper in ``t34_judge``) can build
    one without inventing a budget it did not measure.
    """

    t1: float
    t_star: float
    t2: float
    p_target: float = float("nan")
    p_realized: float = float("nan")
    q1: Optional[float] = None
    q2: Optional[float] = None


def band_upper_threshold(cal_scores: Sequence[float], t1: float, p: float
                         ) -> Tuple[float, float, int]:
    """``t*`` is NOT chosen directly (2502.15845 Sec. 5 / Fig. 1).

    Given ``t1``, ``t*`` is set so that exactly a fraction ``p`` of the
    calibration cheap-scores falls inside the band ``[t1, t*]``.  Returns
    ``(t_star, realized_p, k_band)``.

    ``k = round(p * n)`` is taken from the scores at or above ``t1`` (scores
    below ``t1`` are already decided); with ties the exact fraction can be
    unattainable, so the REALIZED p is returned and reported, never assumed.
    ``k = 0`` degenerates to a single-stage threshold at ``t1``: ``t*`` is set
    just below ``t1`` so every score >= t1 fires without escalation.
    """
    s = np.asarray([v for v in cal_scores if np.isfinite(v)], dtype=float)
    n = s.size
    if n == 0:
        return float(t1), 0.0, 0
    ge = np.sort(s[s >= t1])
    k = int(round(float(p) * n))
    k = max(0, min(k, ge.size))
    if k == 0:
        t_star = float(np.nextafter(t1, -np.inf))
        return t_star, 0.0, 0
    t_star = float(max(ge[k - 1], t1))
    realized = float(((s >= t1) & (s <= t_star)).sum()) / float(n)
    return t_star, realized, k


def cascade_decide(s_cheap: float, thr: CascadeThresholds,
                   s_expensive: Optional[float]) -> Dict[str, Any]:
    """One decision under Algorithm 1 (2502.15845 Sec. 5, Fig. 1).

    s < t1        -> stage 1 negative;
    s > t*        -> stage 1 positive;
    t1 <= s <= t* -> escalate, fire iff s_expensive >= t2.
    A band row whose expensive score is unavailable takes the declared
    :data:`BAND_FALLBACK_FIRE` and is counted separately.
    """
    if not np.isfinite(s_cheap):
        return {"fire": False, "escalated": False, "stage": "undefined",
                "fallback": False}
    if s_cheap < thr.t1:
        return {"fire": False, "escalated": False, "stage": "cheap_negative",
                "fallback": False}
    if s_cheap > thr.t_star:
        return {"fire": True, "escalated": False, "stage": "cheap_positive",
                "fallback": False}
    if s_expensive is None or not np.isfinite(s_expensive):
        return {"fire": bool(BAND_FALLBACK_FIRE), "escalated": True,
                "stage": "band_unavailable", "fallback": True}
    return {"fire": bool(s_expensive >= thr.t2), "escalated": True,
            "stage": "band", "fallback": False}


def apply_cascade(s_cheap: Sequence[float], thr: CascadeThresholds,
                  s_expensive: Sequence[Optional[float]]) -> Dict[str, np.ndarray]:
    """Vectorised :func:`cascade_decide`; ``s_expensive`` may be a callable-backed
    list with ``None`` holes."""
    if len(s_cheap) != len(s_expensive):
        raise ValueError("s_cheap and s_expensive must align")
    dec = [cascade_decide(float(s), thr, e) for s, e in zip(s_cheap, s_expensive)]
    return {
        "fire": np.array([d["fire"] for d in dec], dtype=bool),
        "escalated": np.array([d["escalated"] for d in dec], dtype=bool),
        "fallback": np.array([d["fallback"] for d in dec], dtype=bool),
        "stage": np.array([d["stage"] for d in dec], dtype=object),
    }


def cascade(s_cheap: Sequence[float], s_expensive_fn: Callable[[Any], Optional[float]],
            thr: CascadeThresholds, *, keys: Optional[Sequence[Any]] = None
            ) -> Dict[str, np.ndarray]:
    """Algorithm 1 with a LAZY stage 2 -- the deployable form.

    ``s_expensive_fn(key)`` is invoked ONLY for rows inside the band, which is
    the entire point of the budget dial ``p`` (2502.15845 Sec. 5): the expensive
    stage is paid for a fraction p of decisions, not for all of them.  ``keys``
    defaults to the row indices; on the bench face pass the request key so the
    callback can issue the single-block slice-prefill + regenerate.
    """
    keys = list(keys) if keys is not None else list(range(len(s_cheap)))
    n_calls = 0
    fire, esc, fb, exp = [], [], [], []
    for s, k in zip(s_cheap, keys):
        s = float(s)
        if not np.isfinite(s) or s < thr.t1 or s > thr.t_star:
            d = cascade_decide(s, thr, None)
            exp.append(None)
        else:
            n_calls += 1
            v = s_expensive_fn(k)
            exp.append(v)
            d = cascade_decide(s, thr, v)
        fire.append(d["fire"])
        esc.append(d["escalated"])
        fb.append(d["fallback"])
    return {"fire": np.array(fire, dtype=bool), "escalated": np.array(esc, dtype=bool),
            "fallback": np.array(fb, dtype=bool),
            "s_expensive": np.array([np.nan if v is None else v for v in exp], dtype=float),
            "n_expensive_calls": n_calls}



def declared_orientations() -> Dict[str, int]:
    """Orientations this module acts on: the STAGE1_LADDER rungs and the length
    controls.  ``configs/t34/orientations_cascade.json`` must agree with them
    (single source of truth is the code constant; the json is the declaration
    the scorer reads) - :func:`assert_orientations_consistent` enforces it."""
    out = {name: int(ori) for name, _s, ori, _d in STAGE1_LADDER}
    out.update({name: int(ori) for name, _f, ori in LENGTH_CONTROLS})
    return out


def assert_orientations_consistent(config_path: Optional[Path] = None) -> Dict[str, int]:
    """Raise if the declared json disagrees with the code constants on any key
    both carry; keys only in the json (e.g. cascade_s_cheap) are allowed."""
    import json as _json
    path = Path(config_path) if config_path else Path(__file__).resolve().parent.parent / "configs/t34/orientations_cascade.json"
    declared = {k: int(v) for k, v in _json.loads(path.read_text(encoding="utf-8")).items()
                if not k.startswith("_") and isinstance(v, int) and not isinstance(v, bool)}
    code = declared_orientations()
    bad = {k: (code[k], declared[k]) for k in code if k in declared and declared[k] != code[k]}
    if bad:
        raise ValueError(f"orientations_cascade.json disagrees with the code constants: {bad}")
    return {**code, **declared}


def net_coverage(tpr: Optional[float], l1: Optional[float], l2: Optional[float],
                 fpr: Optional[float], harm: Optional[float]) -> Dict[str, Any]:
    """Closed-loop objective  TPR*L1*L2 - FPR*harm  (digest S1.3 / S4.0).

    ``harm`` is the measured damage of a false reset (the K2 harm arm).  It has
    never been measured on this branch, so when it is None the function
    REFUSES (value None with a reason) instead of assuming 0 or 1 - a default
    here would silently decide the closed-loop pick."""
    missing = [n for n, v in (("tpr", tpr), ("l1", l1), ("l2", l2), ("fpr", fpr), ("harm", harm)) if v is None]
    if missing:
        return {"net_coverage": None, "reason": f"unmeasured inputs: {missing}"}
    return {"net_coverage": float(tpr * l1 * l2 - fpr * harm), "reason": None}


def trigger_metrics(fire: Sequence[bool], y: Sequence[int]) -> Dict[str, Any]:
    """The three metrics with the frame's own three denominators
    (digest 1.3): coverage /n_pos, precision /fires, false-reset /n_neg."""
    f = np.asarray(fire, dtype=bool)
    yy = np.asarray(y, dtype=int)
    n_pos, n_neg = int((yy == 1).sum()), int((yy == 0).sum())
    tp = int((f & (yy == 1)).sum())
    fp = int((f & (yy == 0)).sum())
    return {
        "n": int(len(yy)), "n_pos": n_pos, "n_neg": n_neg,
        "fires": int(f.sum()),
        "coverage": (tp / n_pos) if n_pos else None,
        "coverage_hits": tp,
        "precision": (tp / int(f.sum())) if f.sum() else None,
        "false_reset": (fp / n_neg) if n_neg else None,
        "false_resets": fp,
        "prevalence": prevalence(yy),
    }


def _youden(fire: Sequence[bool], y: Sequence[int]) -> float:
    m = trigger_metrics(fire, y)
    cov = m["coverage"] if m["coverage"] is not None else 0.0
    fr = m["false_reset"] if m["false_reset"] is not None else 0.0
    return float(cov - fr)


SELECTORS: Dict[str, Callable[[Sequence[bool], Sequence[int]], float]] = {
    "youden": _youden,
    "precision": lambda f, y: float(trigger_metrics(f, y)["precision"] or 0.0),
    "coverage": lambda f, y: float(trigger_metrics(f, y)["coverage"] or 0.0),
}


# ---------------------------------------------------------------------------
# threshold fitting -- quantile levels, chosen in inner folds only
# ---------------------------------------------------------------------------

def thresholds_from_levels(cal_scores: Sequence[float],
                           cal_expensive: Sequence[Optional[float]],
                           q1: float, q2: float, p: float) -> CascadeThresholds:
    """Materialise (t1, t*, t2) from quantile LEVELS on a calibration set."""
    s = np.asarray([v for v in cal_scores if np.isfinite(v)], dtype=float)
    t1 = float(np.quantile(s, q1)) if s.size else 0.0
    t_star, realized, _k = band_upper_threshold(s, t1, p)
    band_exp = [e for sc, e in zip(cal_scores, cal_expensive)
                if e is not None and np.isfinite(sc) and t1 <= sc <= t_star]
    if band_exp:
        t2 = float(np.quantile(np.asarray(band_exp, dtype=float), q2))
        # a strictly-interior cut for a binary score: q2 selects <=0 or >0
        if t2 <= 0.0:
            t2 = float(np.nextafter(0.0, np.inf)) if q2 > 0.0 else 0.0
    else:
        t2 = 0.5
    return CascadeThresholds(t1=t1, t_star=t_star, t2=t2, p_target=float(p),
                             p_realized=realized, q1=q1, q2=q2)


def fit_thresholds(cal_scores: Sequence[float], cal_y: Sequence[int],
                   cal_expensive: Sequence[Optional[float]], p: float,
                   *, groups: Optional[Sequence[Any]] = None,
                   inner_folds: int = 3, seed: int = 20260905,
                   select_by: str = "youden",
                   q1_grid: Sequence[float] = Q1_GRID,
                   q2_grid: Sequence[float] = Q2_GRID) -> Tuple[CascadeThresholds, Dict[str, Any]]:
    """Choose (q1, q2) on INNER session-grouped folds of the calibration set.

    Reproduces 2502.15845's calibration step (Sec. 6) under our CV discipline:
    the levels are scored on inner-validation splits only, then materialised on
    the whole calibration set.  Returns the thresholds and a report containing
    ``n_combos`` (the digest requires the number of threshold combinations tried
    to be printed).
    """
    scores = np.asarray(cal_scores, dtype=float)
    y = np.asarray(cal_y, dtype=int)
    exp = list(cal_expensive)
    sel = SELECTORS[select_by]
    groups = np.asarray(groups if groups is not None else np.arange(len(y)))
    combos = [(q1, q2) for q1 in q1_grid for q2 in q2_grid]
    inner_masks = grouped_folds(groups, inner_folds, seed)
    table: List[Tuple[float, float, float]] = []
    for q1, q2 in combos:
        vals: List[float] = []
        for m in inner_masks:
            itr, ite = ~m, m
            if ite.sum() == 0 or itr.sum() == 0:
                continue
            thr = thresholds_from_levels(scores[itr], [exp[i] for i in np.where(itr)[0]],
                                         q1, q2, p)
            out = apply_cascade(scores[ite], thr, [exp[i] for i in np.where(ite)[0]])
            vals.append(sel(out["fire"], y[ite]))
        table.append((q1, q2, float(np.mean(vals)) if vals else float("-inf")))
    best = max(table, key=lambda t: (t[2], -t[0]))
    thr = thresholds_from_levels(scores, exp, best[0], best[1], p)
    report = {"n_combos": len(combos), "inner_folds": int(inner_folds),
              "select_by": select_by, "inner_objective": best[2],
              "q1": best[0], "q2": best[1],
              "n_evaluations": len(combos) * len(inner_masks)}
    return thr, report


def cascade_nested_cv(scores: Sequence[float], y: Sequence[int],
                      expensive: Sequence[Optional[float]], groups: Sequence[Any],
                      p: float, *, outer_folds: int = 5, inner_folds: int = 3,
                      seed: int = 20260905, select_by: str = "youden",
                      rows: Optional[Sequence[Dict[str, Any]]] = None,
                      n_rungs: Optional[int] = None,
                      expects_call: Optional[Sequence[bool]] = None,
                      fit_fn: Optional[Callable[..., Tuple[CascadeThresholds, Dict[str, Any]]]] = None
                      ) -> Dict[str, Any]:
    """PRACTICAL regime: thresholds fitted on other SESSIONS (2502.15845 Sec. 6's
    'different questions' transfer), evaluated on held-out sessions.

    When ``rows``/``n_rungs`` are given the stage-1 ladder's ECDF is REFIT on
    each outer-train fold too, so no part of s_cheap is normalised with the test
    fold's own distribution; ``scores`` is then only the fallback for folds the
    refit cannot serve.  ``fit_fn`` is injectable so a test can spy on exactly
    which rows the fitter was allowed to see.
    """
    fit_fn = fit_fn or fit_thresholds
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    exp = list(expensive)
    groups = np.asarray(groups)
    fire = np.zeros(len(y), dtype=bool)
    escalated = np.zeros(len(y), dtype=bool)
    fallback = np.zeros(len(y), dtype=bool)
    scored = np.zeros(len(y), dtype=bool)
    chosen: List[Dict[str, Any]] = []
    refit = rows is not None and n_rungs is not None
    expects = list(expects_call) if expects_call is not None else [True] * len(y)
    for mask in grouped_folds(groups, outer_folds, seed):
        tr, te = ~mask, mask
        if te.sum() == 0 or tr.sum() == 0 or y[tr].sum() in (0, int(tr.sum())):
            continue
        idx_tr = np.where(tr)[0]
        idx_te = np.where(te)[0]
        if refit:
            lad = Stage1Ladder(int(n_rungs)).fit([rows[i] for i in idx_tr])
            s_tr = lad.score([rows[i] for i in idx_tr], [expects[i] for i in idx_tr])
            s_te = lad.score([rows[i] for i in idx_te], [expects[i] for i in idx_te])
        else:
            s_tr, s_te = scores[tr], scores[te]
        thr, rep = fit_fn(s_tr, y[tr], [exp[i] for i in idx_tr], p,
                          groups=groups[tr], inner_folds=inner_folds, seed=seed,
                          select_by=select_by)
        out = apply_cascade(s_te, thr, [exp[i] for i in idx_te])
        fire[te] = out["fire"]
        escalated[te] = out["escalated"]
        fallback[te] = out["fallback"]
        scored[te] = True
        chosen.append({"n_test": int(te.sum()), **asdict(thr), **rep})
    return {"fire": fire, "escalated": escalated, "fallback": fallback,
            "scored": scored, "folds": chosen, "regime": "practical_cross_session",
            "ladder_refit_per_fold": bool(refit),
            "n_threshold_combos": chosen[0]["n_combos"] if chosen else 0}


def fit_thresholds_direct(cal_scores: Sequence[float], cal_y: Sequence[int],
                          cal_expensive: Sequence[Optional[float]], p: float,
                          *, select_by: str = "youden",
                          q1_grid: Sequence[float] = Q1_GRID,
                          q2_grid: Sequence[float] = Q2_GRID
                          ) -> Tuple[CascadeThresholds, Dict[str, Any]]:
    """Pick (q1, q2) by maximising the objective ON the given rows themselves.

    Only the ORACLE transfer regime may use this: it is the extreme form of
    2502.15845 Sec. 6's same-question calibration, where the thresholds see the
    ground truth of the very questions they are scored on.
    """
    scores = np.asarray(cal_scores, dtype=float)
    y = np.asarray(cal_y, dtype=int)
    exp = list(cal_expensive)
    sel = SELECTORS[select_by]
    combos = [(q1, q2) for q1 in q1_grid for q2 in q2_grid]
    best = None
    for q1, q2 in combos:
        thr = thresholds_from_levels(scores, exp, q1, q2, p)
        out = apply_cascade(scores, thr, exp)
        v = sel(out["fire"], y)
        if best is None or v > best[0]:
            best = (v, thr, q1, q2)
    assert best is not None
    return best[1], {"n_combos": len(combos), "inner_folds": 0,
                     "select_by": select_by, "inner_objective": best[0],
                     "q1": best[2], "q2": best[3], "n_evaluations": len(combos)}


def cascade_oracle_regime(scores: Sequence[float], y: Sequence[int],
                          expensive: Sequence[Optional[float]], groups: Sequence[Any],
                          p: float, *, inner_folds: int = 3, seed: int = 20260905,
                          select_by: str = "youden") -> Dict[str, Any]:
    """ORACLE regime: thresholds calibrated on THE SAME rows they are scored on.

    2502.15845 Sec. 6 calibrates on independently resampled answers for the SAME
    questions; greedy decoding gives us no independent redraw, so the honest
    analogue is same-row calibration with the labels visible -- an optimistic
    bound over the threshold grid, reported next to the practical regime so the
    gap between them is the honest statement of what calibration is worth.
    Note the gap is not sign-guaranteed at n=161: the practical regime fits a
    DIFFERENT threshold pair per outer fold, and a per-fold mixture can beat any
    single global pair.  It also refits the ladder ECDF per fold while this
    function scores the whole-frame ECDF, so at ``n_rungs>1`` the gap mixes
    threshold transfer with score normalisation (see DEVIATIONS).  ``groups``/``inner_folds``/``seed`` are accepted for
    signature symmetry and deliberately unused.
    """
    thr, rep = fit_thresholds_direct(scores, y, expensive, p, select_by=select_by)
    out = apply_cascade(scores, thr, expensive)
    return {"fire": out["fire"], "escalated": out["escalated"],
            "fallback": out["fallback"],
            "scored": np.ones(len(y), dtype=bool),
            "folds": [{"n_test": len(y), **asdict(thr), **rep}],
            "regime": "oracle_same_qid", "n_threshold_combos": rep["n_combos"]}


# ---------------------------------------------------------------------------
# cost (frozen GPU-sec, SPEC 5.6) and the ceiling / null rows
# ---------------------------------------------------------------------------

def gpu_sec_columns(rows: Sequence[Dict[str, Any]],
                    expensive: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-qid measured GPU-sec: ``base_sec`` (the compressed arm's frozen sum)
    and ``expensive_sec`` (slice prefill + regeneration), plus which fields were
    missing.  The cost column is built BEFORE the AUROC column (digest 1191)."""
    base: Dict[str, float] = {}
    exp: Dict[str, Optional[float]] = {}
    missing_base: Dict[str, int] = {}
    for r in rows:
        tot = 0.0
        for f in BASE_COST_FIELDS:
            v = r.get(f)
            if isinstance(v, (int, float)):
                tot += float(v)
            else:
                missing_base[f] = missing_base.get(f, 0) + 1
        base[r["qid"]] = tot
        e = expensive.get(r["qid"])
        if e is None:
            exp[r["qid"]] = None
        else:
            s = 0.0
            seen = False
            for f in EXPENSIVE_COST_FIELDS:
                v = e.get(f)
                if isinstance(v, (int, float)):
                    s += float(v)
                    seen = True
            exp[r["qid"]] = s if seen else None
    return {"base_sec": base, "expensive_sec": exp, "missing_base_fields": missing_base}


def cascade_cost(qids: Sequence[str], escalated: Sequence[bool],
                 cost: Dict[str, Any]) -> Dict[str, Any]:
    """Total measured GPU-sec of a decision vector.

    NOT the paper's ``p * N_v / N_t`` (see DEVIATIONS): base cost is charged for
    every step because our zero point is one compressed generation.
    """
    base = float(sum(cost["base_sec"].get(q, 0.0) for q in qids))
    esc = 0.0
    n_esc = 0
    n_unpriced = 0
    for q, e in zip(qids, escalated):
        if not e:
            continue
        n_esc += 1
        v = cost["expensive_sec"].get(q)
        if v is None:
            n_unpriced += 1
        else:
            esc += float(v)
    n = max(1, len(qids))
    return {"base_gpu_sec": base, "escalation_gpu_sec": esc,
            "total_gpu_sec": base + esc, "gpu_sec_per_step": (base + esc) / n,
            "n_escalated": n_esc, "n_escalated_unpriced": n_unpriced,
            "relative_overhead": (esc / base) if base > 0 else None}


def oracle_trigger_ceiling(y: Sequence[int], qids: Sequence[str],
                           cost: Dict[str, Any]) -> Dict[str, Any]:
    """The horizontal ceiling line of the p-sweep plot: fire exactly on the
    positives (digest line 1189, "oracle trigger drawn as the horizontal
    ceiling").  Label-side by construction -- registered family='oracle'."""
    yy = np.asarray(y, dtype=int)
    fire = yy == 1
    m = trigger_metrics(fire, yy)
    m["family"] = "oracle"
    m["cost"] = cascade_cost(qids, fire, cost)
    return m


def random_at_matched_rate(y: Sequence[int], n_fires: int, *, reps: int = 2000,
                           seed: int = 20260905) -> Dict[str, Any]:
    """Null control: fire uniformly at random at the SAME fire count.  Its mean
    precision is the frame's prevalence -- the number a cascade must beat."""
    rng = np.random.default_rng(seed)
    yy = np.asarray(y, dtype=int)
    n = len(yy)
    cov, prec, fr = [], [], []
    for _ in range(reps):
        idx = rng.choice(n, size=min(n_fires, n), replace=False)
        f = np.zeros(n, dtype=bool)
        f[idx] = True
        m = trigger_metrics(f, yy)
        cov.append(m["coverage"] or 0.0)
        prec.append(m["precision"] or 0.0)
        fr.append(m["false_reset"] or 0.0)
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]  # noqa: E731
    return {"n_fires": int(n_fires), "reps": int(reps),
            "coverage_mean": float(np.mean(cov)), "coverage_ci": q(cov),
            "precision_mean": float(np.mean(prec)), "precision_ci": q(prec),
            "false_reset_mean": float(np.mean(fr)), "false_reset_ci": q(fr)}


def _delta_block(cand: np.ndarray, ref: np.ndarray, y: np.ndarray,
                 clusters: np.ndarray, *, reps: int) -> Dict[str, Any]:
    """Clustered PAIRED delta of AP (candidate - comparator) on the same
    resamples.  The digest's winner rule is stated on the delta's CI lower
    bound, not on two point estimates side by side."""
    ok = np.isfinite(cand) & np.isfinite(ref)
    if ok.sum() == 0 or y[ok].sum() in (0, int(ok.sum())):
        return {"n_scored": int(ok.sum()), "delta_auprc": None,
                "ci95": None, "ci_lower_gt_0": None}
    d, lo, hi = paired_delta_bootstrap(average_precision, cand[ok], ref[ok],
                                       y[ok], clusters[ok], reps=reps)
    return {"n_scored": int(ok.sum()), "delta_auprc": d,
            "ci95": None if lo is None else [lo, hi],
            "ci_lower_gt_0": None if lo is None else bool(lo > 0.0)}


def _three_metrics_at_matched_rate(scores: np.ndarray, y: np.ndarray,
                                   n_fires: int) -> Dict[str, Any]:
    """coverage / precision / false-reset of a SCORE read at the comparator's
    own fire count, so the three-metric clause compares like with like."""
    v = np.where(np.isfinite(scores), scores, -np.inf)
    op = operating_point(v, y, n_fires)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    # operating_point breaks ties by stable row order; a score with many ties at
    # the cut therefore gets an arbitrary (possibly flattering) fire set, so the
    # tie count travels with the number instead of being hidden.
    ties = (int((v == op["threshold"]).sum()) if op["threshold"] is not None else 0)
    return {"fires": op["fires"],
            "coverage": (op["coverage"] / n_pos) if n_pos else None,
            "precision": op["precision"],
            "false_reset": (op["false_resets"] / n_neg) if n_neg else None,
            "ties_at_threshold": ties}


def winner_rule(s_cheap: np.ndarray, y: np.ndarray, clusters: np.ndarray, *,
                baseline: np.ndarray, s0_twin: np.ndarray,
                controls: Dict[str, np.ndarray], baseline_fires: int,
                censored: Optional[Sequence[bool]] = None,
                reps: int = 2000) -> Dict[str, Any]:
    """The digest's 4.0 winner rule, evaluated rather than left to the reader.

    A trigger signal is ALIVE only if all of:
      (a) it beats the parse-failure baseline on all three metrics read at the
          baseline's own fire count;
      (b) its AUPRC and the lower end of its session-clustered CI both exceed
          chance -- which on THIS frame is the evaluation-frame prevalence
          93/161 = 0.578, never the 900-frame 0.1033;
      (c) the clustered CI lower bound of Delta-AUPRC against its own S0 twin
          (the same score computed on the FULL arm) is > 0;
      (d) the same against every pre-registered length control;
      (e) the direction is unchanged on the UNCENSORED stratum -- the digest's
          4.0 clause "``censored_at_cap`` 分层 ... 在未截断子集上方向不变".
          Checked as AUPRC > that stratum's OWN prevalence.  When ``censored``
          is not supplied, or the uncensored stratum is single-class, the
          clause is reported as unchecked (None) and listed in
          ``unchecked_criteria`` -- never silently passed.
    Not from 2502.15845 -- their winner rule is "closest to the GCN ceiling";
    this is the house contract the migration is scored under.
    """
    y = np.asarray(y, dtype=int)
    chance = prevalence(y)
    ok = np.isfinite(s_cheap)
    ap = average_precision(s_cheap[ok], y[ok]) if ok.sum() else None
    lo = hi = ncl = None
    if ok.sum():
        lo, hi, ncl = clustered_bootstrap(average_precision, s_cheap[ok], y[ok],
                                          clusters[ok], reps=reps)
    base_m = _three_metrics_at_matched_rate(baseline, y, baseline_fires)
    cand_m = _three_metrics_at_matched_rate(s_cheap, y, baseline_fires)
    beats_three = all(v is not None for v in
                      (base_m["coverage"], cand_m["coverage"],
                       base_m["precision"], cand_m["precision"],
                       base_m["false_reset"], cand_m["false_reset"])) and (
        cand_m["coverage"] > base_m["coverage"]
        and cand_m["precision"] > base_m["precision"]
        and cand_m["false_reset"] < base_m["false_reset"])
    vs_s0 = _delta_block(s_cheap, s0_twin, y, clusters, reps=reps)
    vs_ctl = {name: _delta_block(s_cheap, vec, y, clusters, reps=reps)
              for name, vec in controls.items()}
    vs_base = _delta_block(s_cheap, baseline, y, clusters, reps=reps)

    # (e) direction unchanged on the uncensored stratum (digest 4.0)
    unc: Dict[str, Any] = {"checked": False, "reason": "censored mask not supplied",
                           "n": None, "prevalence": None, "auprc": None,
                           "auroc": None, "direction_holds": None}
    if censored is not None:
        m_unc = ~np.asarray(censored, dtype=bool)
        keep = m_unc & ok
        if keep.sum() and y[keep].sum() not in (0, int(keep.sum())):
            ap_u = average_precision(s_cheap[keep], y[keep])
            ch_u = prevalence(y[keep])
            unc = {"checked": True, "reason": None, "n": int(keep.sum()),
                   "prevalence": ch_u, "auprc": ap_u,
                   "auroc": auroc(s_cheap[keep], y[keep]),
                   "direction_holds": bool(ap_u is not None and ap_u > ch_u)}
        else:
            unc["reason"] = "uncensored stratum is empty or single-class"

    failed: List[str] = []
    unchecked: List[str] = []
    if not beats_three:
        failed.append("three_metrics_vs_parse_fail_baseline")
    if not (ap is not None and lo is not None and ap > chance and lo > chance):
        failed.append("auprc_ci_lower_above_chance")
    if vs_s0["ci_lower_gt_0"] is not True:
        failed.append("delta_auprc_vs_s0_twin")
    for name, blk in vs_ctl.items():
        if blk["ci_lower_gt_0"] is not True:
            failed.append(f"delta_auprc_vs_{name}")
    if unc["checked"]:
        if unc["direction_holds"] is not True:
            failed.append("direction_holds_on_uncensored_stratum")
    else:
        unchecked.append("direction_holds_on_uncensored_stratum")
    return {
        "chance_ap": chance,
        "chance_note": "evaluation-frame prevalence (161-row trigger subset), "
                       "NOT the 900-frame base rate 0.1033",
        "candidate": {"auprc": ap, "auprc_ci95": None if lo is None else [lo, hi],
                      "n_clusters": ncl,
                      "auroc": auroc(s_cheap[ok], y[ok]) if ok.sum() else None},
        "three_metrics_at_baseline_fire_count": {"baseline": base_m, "candidate": cand_m,
                                                 "passes": bool(beats_three)},
        "delta_auprc_vs_baseline": vs_base,
        "delta_auprc_vs_s0_twin": vs_s0,
        "delta_auprc_vs_length_controls": vs_ctl,
        "uncensored_stratum_direction": unc,
        "failed_criteria": failed,
        "unchecked_criteria": unchecked,
        "verdict": "alive" if not failed else "dead",
        "verdict_note": ("all clauses evaluated" if not unchecked else
                         "verdict conditional: " + ", ".join(unchecked) +
                         " could not be evaluated on this frame"),
    }


# ---------------------------------------------------------------------------
# label-side functions (never called from the feature path)
# ---------------------------------------------------------------------------

def cw_label_for_row(full_row: Dict[str, Any], c2kv_row: Dict[str, Any]) -> Optional[int]:
    """THE label, delegated to :func:`t33_labels.cw_label` so it exists in one
    place only.  Kept here as a named seam for the mechanical assertion test:
    the feature path must produce identical output when every field this
    function reads is deleted from the row."""
    return cw_label(full_row, c2kv_row)


def s_cheap_label_s0_twin(full_rows: Sequence[Dict[str, Any]],
                          ladder: Stage1Ladder,
                          expects_call: Sequence[bool]) -> np.ndarray:
    """S0 control: the SAME s_cheap computed on the FULL arm's rows.

    Label-side by construction (it touches a full-arm field), so it is named
    ``*_label_*`` and never enters the feature frame.  The digest's winner rule
    requires the candidate to beat its own S0 twin."""
    return ladder.score(full_rows, expects_call)


# ---------------------------------------------------------------------------
# GCN detection ceiling (numpy; explicit forward/backward)
# ---------------------------------------------------------------------------

def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """``D^-1/2 (A + I) D^-1/2`` -- the standard GCN propagation matrix
    (2502.15845 Sec. 4.1 uses a two-layer GCN over the pairwise matrix)."""
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    Ah = A + np.eye(n)
    d = Ah.sum(axis=1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    return (Ah * dinv[:, None]) * dinv[None, :]


def similarity_graph(X: np.ndarray, *, gamma: Optional[float] = None) -> np.ndarray:
    """RBF similarity over node features, zero diagonal.

    DEVIATION: the paper's adjacency IS the entailment matrix; we have no
    entailment estimator, so the pairwise structure is rebuilt from the per-doc
    score vector."""
    X = np.asarray(X, dtype=float)
    n, f = X.shape
    if gamma is None:
        gamma = 1.0 / max(1, f)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    A = np.exp(-gamma * d2)
    np.fill_diagonal(A, 0.0)
    return A


@dataclass
class GCNParams:
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    def flat(self) -> np.ndarray:
        return np.concatenate([self.W1.ravel(), self.b1.ravel(),
                               self.W2.ravel(), self.b2.ravel()])

    def like(self, flat: np.ndarray) -> "GCNParams":
        i = 0
        out = []
        for a in (self.W1, self.b1, self.W2, self.b2):
            k = a.size
            out.append(flat[i:i + k].reshape(a.shape))
            i += k
        return GCNParams(*out)


def gcn_init(n_features: int, hidden: int, seed: int = 20260905) -> GCNParams:
    rng = np.random.default_rng(seed)
    s1 = math.sqrt(1.0 / max(1, n_features))
    s2 = math.sqrt(1.0 / max(1, hidden))
    return GCNParams(W1=rng.normal(0, s1, size=(n_features, hidden)),
                     b1=np.zeros(hidden),
                     W2=rng.normal(0, s2, size=(hidden, 1)),
                     b2=np.zeros(1))


def gcn_forward(params: GCNParams, graphs: Sequence[Dict[str, np.ndarray]]) -> np.ndarray:
    """Graph logits: mean-pooled two-layer GCN.

    ``H = relu(A_hat X W1 + b1)``; ``Z = A_hat H W2 + b2``; ``logit = mean(Z)``.
    """
    out = np.zeros(len(graphs), dtype=float)
    for i, g in enumerate(graphs):
        Ah, X = g["A_hat"], g["X"]
        pre1 = Ah @ X @ params.W1 + params.b1
        H = np.maximum(pre1, 0.0)
        Z = Ah @ H @ params.W2 + params.b2
        out[i] = float(Z.mean())
    return out


def gcn_loss_and_grads(params: GCNParams, graphs: Sequence[Dict[str, np.ndarray]],
                       y: np.ndarray, l2: float = 0.0
                       ) -> Tuple[float, GCNParams]:
    """Mean BCE-with-logits + L2 on the weights, with the explicit backward pass
    (finite-difference tested in ``test_t34_cascade.py``)."""
    y = np.asarray(y, dtype=float)
    N = max(1, len(graphs))
    gW1 = np.zeros_like(params.W1)
    gb1 = np.zeros_like(params.b1)
    gW2 = np.zeros_like(params.W2)
    gb2 = np.zeros_like(params.b2)
    loss = 0.0
    for i, g in enumerate(graphs):
        Ah, X = g["A_hat"], g["X"]
        AX = Ah @ X
        pre1 = AX @ params.W1 + params.b1
        H = np.maximum(pre1, 0.0)
        AH = Ah @ H
        Z = AH @ params.W2 + params.b2
        logit = float(Z.mean())
        # stable BCE-with-logits
        loss += (max(logit, 0.0) - logit * y[i] + math.log1p(math.exp(-abs(logit)))) / N
        dlogit = (1.0 / (1.0 + math.exp(-logit)) - y[i]) / N
        n_nodes = Z.shape[0]
        dZ = np.full((n_nodes, 1), dlogit / n_nodes)
        gW2 += AH.T @ dZ
        gb2 += dZ.sum(axis=0)
        dH = Ah.T @ dZ @ params.W2.T
        dpre1 = dH * (pre1 > 0)
        gW1 += AX.T @ dpre1
        gb1 += dpre1.sum(axis=0)
    if l2:
        loss += 0.5 * l2 * (float((params.W1 ** 2).sum()) + float((params.W2 ** 2).sum()))
        gW1 += l2 * params.W1
        gW2 += l2 * params.W2
    return float(loss), GCNParams(gW1, gb1, gW2, gb2)


def gcn_fit(graphs: Sequence[Dict[str, np.ndarray]], y: np.ndarray, *,
            hidden: int = 8, l2: float = 1e-2, epochs: int = 300,
            lr: float = 0.05, seed: int = 20260905) -> GCNParams:
    """Full-batch Adam on the BCE objective (2502.15845 Sec. 4.1: two-layer GCN
    trained with BCE against the label)."""
    n_features = graphs[0]["X"].shape[1]
    p = gcn_init(n_features, hidden, seed)
    m = np.zeros_like(p.flat())
    v = np.zeros_like(p.flat())
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    for t in range(1, int(epochs) + 1):
        _loss, grad = gcn_loss_and_grads(p, graphs, y, l2=l2)
        gflat = grad.flat()
        m = b1_ * m + (1 - b1_) * gflat
        v = b2_ * v + (1 - b2_) * gflat ** 2
        mhat = m / (1 - b1_ ** t)
        vhat = v / (1 - b2_ ** t)
        p = p.like(p.flat() - lr * mhat / (np.sqrt(vhat) + eps))
    return p


def build_graphs(score_vectors: Dict[str, Sequence[Sequence[float]]],
                 qids: Sequence[str]) -> List[Dict[str, np.ndarray]]:
    """One graph per qid: nodes = history blocks, features = the per-doc score
    vector rows, adjacency = :func:`similarity_graph`."""
    graphs = []
    for q in qids:
        X = np.asarray(score_vectors[q], dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        graphs.append({"X": X, "A_hat": normalize_adjacency(similarity_graph(X))})
    return graphs


def gcn_detection_ceiling(score_vectors: Dict[str, Sequence[Sequence[float]]],
                          qids: Sequence[str], y: Sequence[int],
                          groups: Sequence[Any], *, hidden_grid: Sequence[int] = (4, 8),
                          l2_grid: Sequence[float] = (1e-2, 1e-1),
                          outer_folds: int = 5, inner_folds: int = 3,
                          epochs: int = 200, seed: int = 20260905,
                          n_perm: int = 0, family: str = "detector") -> Dict[str, Any]:
    """The detection-side ceiling (2502.15845 Sec. 4.1's construction).

    Session-grouped nested CV: (hidden, l2) chosen on INNER folds, the model
    refit on the outer-train fold, out-of-fold logits pooled.  Read as "the
    upper bound of any scalar function of this per-doc representation", NOT as a
    deployable detector.  ``n_perm`` adds a label-permutation null so a ceiling
    on 161 rows is not read as a real effect.
    """
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    graphs_all = build_graphs(score_vectors, qids)
    oof = np.full(len(y), np.nan)
    chosen: List[Dict[str, Any]] = []

    def _standardise(train_idx, test_idx):
        allX = np.concatenate([graphs_all[i]["X"] for i in train_idx], axis=0)
        mu, sd = allX.mean(axis=0), allX.std(axis=0) + 1e-9
        def view(idx):
            out = []
            for i in idx:
                X = (graphs_all[i]["X"] - mu) / sd
                out.append({"X": X, "A_hat": normalize_adjacency(similarity_graph(X))})
            return out
        return view(train_idx), view(test_idx)

    for mask in grouped_folds(groups, outer_folds, seed):
        tr, te = np.where(~mask)[0], np.where(mask)[0]
        if te.size == 0 or tr.size == 0 or y[tr].sum() in (0, tr.size):
            continue
        best = None
        for hidden in hidden_grid:
            for l2 in l2_grid:
                inner_scores = np.full(tr.size, np.nan)
                gtr = groups[tr]
                for imask in grouped_folds(gtr, inner_folds, seed + 1):
                    itr, ite = tr[~imask], tr[imask]
                    if ite.size == 0 or itr.size == 0 or y[itr].sum() in (0, itr.size):
                        continue
                    Gtr, Gte = _standardise(itr, ite)
                    p = gcn_fit(Gtr, y[itr], hidden=hidden, l2=l2, epochs=epochs, seed=seed)
                    inner_scores[np.isin(tr, ite)] = gcn_forward(p, Gte)
                ok = np.isfinite(inner_scores)
                m = (auroc(inner_scores[ok], y[tr][ok])
                     if ok.sum() and y[tr][ok].sum() not in (0, int(ok.sum())) else None)
                if m is not None and (best is None or m > best[0]):
                    best = (m, hidden, l2)
        if best is None:
            continue
        _, hidden, l2 = best
        Gtr, Gte = _standardise(tr, te)
        p = gcn_fit(Gtr, y[tr], hidden=hidden, l2=l2, epochs=epochs, seed=seed)
        oof[te] = gcn_forward(p, Gte)
        chosen.append({"hidden": int(hidden), "l2": float(l2),
                       "inner_auroc": float(best[0]), "n_test": int(te.size)})

    ok = np.isfinite(oof)
    out: Dict[str, Any] = {
        "family": family,
        "n_scored": int(ok.sum()), "n_pos": int(y[ok].sum()),
        "prevalence": prevalence(y[ok]) if ok.sum() else None,
        "auroc": auroc(oof[ok], y[ok]) if ok.sum() else None,
        "auprc": average_precision(oof[ok], y[ok]) if ok.sum() else None,
        "chosen": chosen,
        "note": "ceiling of any scalar function of this per-doc representation; "
                "not a deployable detector (2502.15845 Sec. 4.1)",
    }
    if ok.sum():
        clusters = session_clusters([str(g) for g in groups[ok]])
        lo, hi, ncl = clustered_bootstrap(average_precision, oof[ok], y[ok], clusters)
        out["auprc_ci95"] = [lo, hi]
        out["n_clusters"] = ncl
    if n_perm:
        def _refit(yp: np.ndarray) -> float:
            r = gcn_detection_ceiling(score_vectors, qids, yp, groups,
                                      hidden_grid=hidden_grid[:1], l2_grid=l2_grid[:1],
                                      outer_folds=outer_folds, inner_folds=inner_folds,
                                      epochs=epochs, seed=seed, n_perm=0)
            return float("nan") if r["auroc"] is None else float(r["auroc"])
        # the ENTIRE pipeline (including the inner (hidden, l2) search) is rerun
        # on permuted labels -- that is what makes the band honest
        out["permutation_auroc"] = permutation_band(_refit, y, groups,
                                                    n_perm=int(n_perm), seed=seed)
    return out


# ---------------------------------------------------------------------------
# node-feature providers
# ---------------------------------------------------------------------------

_TOKEN_MIN_CHARS = 3


def query_lexical_node_features(docs: Sequence[str], query: str) -> List[List[float]]:
    """GOLD-FREE per-doc score vector: witness-IDF (``d_witness_core``) run with
    the CURRENT QUERY's own tokens as the value set, plus the doc's length and
    position.  Reads the request only (query + docs), never the target."""
    values = []
    seen = set()
    for tok in "".join(ch if (ch.isalnum() or ch in "_-.@") else " " for ch in (query or "")).split():
        if len(tok) >= _TOKEN_MIN_CHARS and tok not in seen:
            seen.add(tok)
            values.append(tok)
    _df, scores = witness_scores(list(docs), values) if values else ({}, [0.0] * len(docs))
    n = max(1, len(docs) - 1)
    return [[float(scores[i]), float(len(docs[i])), i / n] for i in range(len(docs))]


def gold_witness_label_node_features(frame: FrozenFrame, qid: str) -> Optional[List[List[float]]]:
    """ORACLE (family='oracle') per-doc vector from the frozen, GOLD-SCORED
    witness table.  Label-side: it uses the target's tool name and argument
    leaves, and it exists for the 93 C->W rows only, so its availability is the
    label.  Never feed this to a detection ceiling that is reported as
    achievable (see DEVIATIONS)."""
    ent = frame.witness_entry(qid)
    if not ent:
        return None
    score = ent.get("score") or []
    lens = ent.get("doc_lengths") or [0] * len(score)
    n = max(1, len(score) - 1)
    return [[float(score[i]), float(lens[i] if i < len(lens) else 0), i / n]
            for i in range(len(score))]


def label_witness_values(target_tool_name: Optional[str], target_args: Any) -> List[str]:
    """Label-side helper mirroring ``d_witness_core.target_values`` -- kept here
    only so the oracle path has a named ``*_label_*`` seam."""
    return target_values(target_tool_name, target_args)


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def p_sweep(frame: FrozenFrame, *, expensive: Optional[Dict[str, Dict[str, Any]]] = None,
            n_rungs: int = 1, p_grid: Sequence[float] = P_GRID,
            outer_folds: int = 5, inner_folds: int = 3, seed: int = 20260905,
            select_by: str = "youden", allow_partial_expensive: bool = False,
            reps: int = 2000) -> Dict[str, Any]:
    """The deliverable table: for each p, both transfer regimes, three metrics,
    measured GPU-sec, plus the baseline / control / ceiling rows.

    Cost column first, AUROC column second (digest 1191).
    """
    sub = frame.trigger_subset()
    qids = [r["qid"] for r in sub]
    y = np.array([r["label_cw"] for r in sub], dtype=int)
    groups = np.array([session_of(q) for q in qids])
    steps = np.array([step_index(q) for q in qids])
    c2kv = frame.c2kv_by_qid
    full = frame.full_by_qid
    rows = [c2kv[q] for q in qids]
    full_rows = [full[q] for q in qids]
    expects = [True] * len(qids)          # EXPECTS_CALL_ALWAYS on this frame

    ladder = Stage1Ladder(n_rungs).fit(rows)
    s_cheap = ladder.score(rows, expects)
    s0_twin = s_cheap_label_s0_twin(full_rows, ladder, expects)
    ctl_len = length_control_scores(rows)

    expensive = expensive or {}
    s_exp: List[Optional[float]] = []
    for q in qids:
        e = expensive.get(q)
        s_exp.append(None if e is None
                     else stage2_action_changed(c2kv[q].get("prediction"), e.get("prediction")))
    available = [v is not None for v in s_exp]
    audit = expensive_coverage_audit(qids, y, available)
    cost = gpu_sec_columns(rows, expensive)

    clusters = session_clusters([str(g) for g in groups])
    out: Dict[str, Any] = {
        "frame": {"n": len(qids), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                  "n_sessions": len(set(groups.tolist())),
                  "prevalence_is_chance_ap": prevalence(y),
                  "cap_tokens": frame.cap_tokens(),
                  "n_with_predecessor": int(sum(1 for i, q in enumerate(qids)
                                                if f"{session_of(q)}:{steps[i]-1}" in set(qids)))},
        "ladder": ladder.audit(),
        "expensive_coverage": audit,
        "cost_inputs": {"missing_base_fields": cost["missing_base_fields"],
                        "n_priced_expensive": int(sum(1 for q in qids
                                                      if cost["expensive_sec"].get(q) is not None))},
        "baseline_parse_fail": None,
        "length_control": None,
        "s0_twin": None,
        "oracle_trigger": oracle_trigger_ceiling(y, qids, cost),
        "p_rows": [],
        "refusals": [],
    }

    # -- baseline (rung 0 alone), length control, S0 twin: ranking view -----
    base_only = Stage1Ladder(1).fit(rows).score(rows, expects)
    b_fire = base_only >= 1.0
    out["baseline_parse_fail"] = {
        **trigger_metrics(b_fire, y),
        "auprc": average_precision(base_only, y), "auroc": auroc(base_only, y),
        "cost": cascade_cost(qids, np.zeros(len(qids), dtype=bool), cost),
        "random_at_matched_rate": random_at_matched_rate(y, int(b_fire.sum()), reps=reps),
    }
    lo, hi, ncl = clustered_bootstrap(average_precision, base_only, y, clusters, reps=reps)
    out["baseline_parse_fail"]["auprc_ci95"] = [lo, hi]
    out["baseline_parse_fail"]["n_clusters"] = ncl

    out["length_control"] = {"note": "each length alone as s_cheap "
                                     "(pre-registered length control)"}
    for name, vec in ctl_len.items():
        ok_len = np.isfinite(vec)
        entry: Dict[str, Any] = {
            "n_scored": int(ok_len.sum()),
            "auprc": average_precision(vec[ok_len], y[ok_len]) if ok_len.sum() else None,
            "auroc": auroc(vec[ok_len], y[ok_len]) if ok_len.sum() else None,
        }
        if ok_len.sum():
            cl = session_clusters([str(g) for g in groups[ok_len]])
            lo, hi, ncl = clustered_bootstrap(average_precision, vec[ok_len],
                                              y[ok_len], cl, reps=reps)
            entry["auprc_ci95"] = [lo, hi]
            entry["n_clusters"] = ncl
        out["length_control"][name] = entry
    out["s0_twin"] = {
        "family": "control",
        "auprc": average_precision(s0_twin, y), "auroc": auroc(s0_twin, y),
        "note": "the SAME s_cheap on the FULL arm's rows (label-side control)",
    }
    out["s_cheap"] = {
        "auprc": average_precision(s_cheap, y), "auroc": auroc(s_cheap, y),
        "n_distinct": int(np.unique(s_cheap).size),
    }
    lo, hi, ncl = clustered_bootstrap(average_precision, s_cheap, y, clusters, reps=reps)
    out["s_cheap"]["auprc_ci95"] = [lo, hi]
    out["s_cheap"]["n_clusters"] = ncl

    # -- censored_at_cap stratification (required by the default contract) --
    cens = np.array([bool(r["censored_at_cap"]) for r in sub])

    # -- the digest 4.0 winner rule, evaluated (not left to the reader) ----
    out["winner_rule"] = winner_rule(
        s_cheap, y, clusters, baseline=base_only, s0_twin=s0_twin,
        controls={k: v for k, v in ctl_len.items()},
        baseline_fires=int(b_fire.sum()), censored=cens, reps=reps)

    out["strata"] = {}
    for name, m in (("censored_at_cap", cens), ("uncensored", ~cens)):
        if m.sum() and y[m].sum() not in (0, int(m.sum())):
            out["strata"][name] = {"n": int(m.sum()), "n_pos": int(y[m].sum()),
                                   "prevalence": prevalence(y[m]),
                                   "auprc": average_precision(s_cheap[m], y[m]),
                                   "auroc": auroc(s_cheap[m], y[m])}
        else:
            out["strata"][name] = {"n": int(m.sum()), "n_pos": int(y[m].sum()),
                                   "auprc": None, "auroc": None}

    # -- the p sweep -------------------------------------------------------
    for p in p_grid:
        if p > 0 and not audit["ok"] and not allow_partial_expensive:
            out["refusals"].append({
                "p": float(p), "reason": audit["reason"],
                "fix": "run the single-block slice-prefill + regenerate arm on the "
                       "C->C rows too (RUNBOOK step 2)"})
            continue
        row: Dict[str, Any] = {"p": float(p),
                               "family": "candidate" if (p == 0 or audit["ok"])
                               else "diagnostic_partial"}
        for regime_fn, key in ((cascade_nested_cv, "practical"),
                               (cascade_oracle_regime, "oracle")):
            if key == "practical":
                res = regime_fn(s_cheap, y, s_exp, groups, p, outer_folds=outer_folds,
                                inner_folds=inner_folds, seed=seed, select_by=select_by,
                                rows=rows, n_rungs=n_rungs, expects_call=expects)
            else:
                res = regime_fn(s_cheap, y, s_exp, groups, p, inner_folds=inner_folds,
                                seed=seed, select_by=select_by)
            sc = res["scored"]
            m = trigger_metrics(res["fire"][sc], y[sc])
            m["regime"] = res["regime"]
            m["n_threshold_combos"] = res["n_threshold_combos"]
            m["n_escalated"] = int(res["escalated"][sc].sum())
            m["n_band_unavailable"] = int(res["fallback"][sc].sum())
            m["ladder_refit_per_fold"] = bool(res.get("ladder_refit_per_fold", False))
            m["p_realized_mean"] = float(np.mean([f["p_realized"] for f in res["folds"]])) \
                if res["folds"] else None
            m["cost"] = cascade_cost([q for q, s in zip(qids, sc) if s],
                                     res["escalated"][sc], cost)
            m["random_at_matched_rate"] = random_at_matched_rate(
                y[sc], int(res["fire"][sc].sum()), reps=reps)
            m["select_by"] = select_by
            # the value of the objective the thresholds were actually chosen for
            m["selection_objective"] = float(SELECTORS[select_by](res["fire"][sc], y[sc]))
            row[key] = m
        if row.get("practical") and row.get("oracle"):
            # The transfer gap must be read on the objective the thresholds were
            # SELECTED for.  coverage alone can go the "wrong" way (the oracle
            # regime buys its higher objective with fewer, cleaner fires), which
            # would misread as transfer failure -- so both are reported and the
            # objective one is named as the primary.
            row["transfer_gap_primary"] = f"transfer_gap_{select_by}"
            row[f"transfer_gap_{select_by}"] = (
                row["oracle"]["selection_objective"] - row["practical"]["selection_objective"])
            row["transfer_gap_coverage"] = (
                (row["oracle"]["coverage"] or 0.0) - (row["practical"]["coverage"] or 0.0))
            row["transfer_gap_note"] = (
                "oracle regime = same-row calibration with labels visible (an "
                "optimistic bound over the threshold grid); the practical regime "
                "fits a DIFFERENT threshold pair per outer fold, so neither gap "
                "is sign-guaranteed at n=161")
        out["p_rows"].append(row)

    return out


def emit_features(frame: FrozenFrame, path: Path, *, n_rungs: int,
                  expensive: Optional[Dict[str, Dict[str, Any]]] = None) -> int:
    """Per-qid scalar features into the shared feature frame (rule 3).

    ``None`` where undefined -- never a sentinel.  Column names go through the
    leakage guard inside ``write_features_jsonl``.  NOTE: the ladder's ECDF is
    fitted on the whole frame for this DUMP (label-free, but transductive), so a
    consumer that scores ``cascade_s_cheap`` inherits that; the practical regime
    inside :func:`p_sweep` refits the ladder on each outer-train fold instead."""
    sub = frame.trigger_subset()
    c2kv = frame.c2kv_by_qid
    rows = [c2kv[r["qid"]] for r in sub]
    ladder = Stage1Ladder(n_rungs).fit(rows)
    s = ladder.score(rows, [True] * len(rows))
    expensive = expensive or {}
    out = []
    for i, r in enumerate(sub):
        q = r["qid"]
        row = c2kv[q]
        e = expensive.get(q)
        feats = prefix_scalars(row)
        out.append({
            "qid": q, "session_id": r["session_id"], "arm": "c2kv",
            "cascade_parse_fail": float(parse_fail_indicator(row.get("prediction"), True)),
            "cascade_s_cheap": float(s[i]),
            "cascade_stage2_action_changed":
                (None if e is None
                 else stage2_action_changed(row.get("prediction"), e.get("prediction"))),
            **{name: (float(row[field]) if row.get(field) is not None else None)
               for name, field, _ori in LENGTH_CONTROLS},
            **{k: v for k, v in feats.items()},
        })
    return write_features_jsonl(Path(path), out, context="t34_cascade features")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cmd_sweep(args: argparse.Namespace) -> int:
    frame = FrozenAssets(Path(args.root)).load()
    expensive = load_expensive_table(Path(args.expensive)) if args.expensive else None
    block_audit = (expensive_block_choice_audit(Path(args.expensive), frame)
                   if args.expensive else None)
    res = p_sweep(frame, expensive=expensive, n_rungs=args.rungs,
                  p_grid=tuple(args.p) if args.p else P_GRID,
                  outer_folds=args.outer_folds, inner_folds=args.inner_folds,
                  seed=args.seed, select_by=args.select_by,
                  allow_partial_expensive=args.allow_partial_expensive,
                  reps=args.reps)
    res["deviations"] = DEVIATIONS
    res["expensive_block_choice"] = block_audit
    if args.features_out:
        res["features_written"] = emit_features(frame, Path(args.features_out),
                                                n_rungs=args.rungs, expensive=expensive)
        res["features_path"] = str(args.features_out)
    sha = freeze_json(Path(args.out), res)
    print(json.dumps({"out": str(args.out), "sha256": sha,
                      "n_p_rows": len(res["p_rows"]),
                      "n_refusals": len(res["refusals"])}, indent=1))
    return 0


def _cmd_ceiling(args: argparse.Namespace) -> int:
    frame = FrozenAssets(Path(args.root)).load()
    sub = frame.trigger_subset()
    y_all = {r["qid"]: r["label_cw"] for r in sub}
    vectors: Dict[str, List[List[float]]] = {}
    stale: Dict[str, Any] = {}
    if args.node_features == "query_lexical":
        if not args.sidecar:
            print(json.dumps({"refused": "query_lexical node features need --sidecar "
                                         "(results/t34/sidecar_<arm>.jsonl, unit U2)"}, indent=1))
            return 2
        docs_map = load_decoded_docs(Path(args.sidecar))
        # stale-dump guard: the sidecar's doc texts must hash to the frozen
        # witness table's per-doc sha256 wherever the table has an entry
        stale = check_docs_against_witness(docs_map, frame)
        if stale["mismatched_qids"] and not args.allow_partial:
            print(json.dumps({"refused": "sidecar disagrees with the frozen witness "
                                         "doc hashes", "stale": stale}, indent=1))
            return 2
        queries: Dict[str, str] = {}
        for r in load_jsonl(str(args.sidecar)):
            queries[r["qid"]] = r.get("query") or ""
        for q, docs in docs_map.items():
            if q in y_all and docs:
                vectors[q] = query_lexical_node_features(docs, queries.get(q, ""))
        family = "detector"
    else:
        for q in y_all:
            v = gold_witness_label_node_features(frame, q)
            if v:
                vectors[q] = v
        family = "oracle"
    qids = sorted(vectors)
    y = np.array([y_all[q] for q in qids], dtype=int)
    groups = np.array([session_of(q) for q in qids])
    avail_audit = expensive_coverage_audit(
        [r["qid"] for r in sub], np.array([r["label_cw"] for r in sub]),
        [r["qid"] in vectors for r in sub])
    if not avail_audit["ok"] and not args.allow_partial:
        print(json.dumps({"refused": avail_audit["reason"], "audit": avail_audit}, indent=1))
        return 2
    res = gcn_detection_ceiling(vectors, qids, y, groups, epochs=args.epochs,
                               outer_folds=args.outer_folds, inner_folds=args.inner_folds,
                               seed=args.seed, n_perm=args.n_perm, family=family)
    res["node_feature_availability"] = avail_audit
    res["sidecar_witness_check"] = stale or None
    res["deviations"] = DEVIATIONS
    sha = freeze_json(Path(args.out), res)
    print(json.dumps({"out": str(args.out), "sha256": sha, "auroc": res["auroc"],
                      "auprc": res["auprc"], "family": family}, indent=1))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    assert_orientations_consistent()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("sweep", help="Algorithm 1 p-sweep on the 161-row frame")
    sw.add_argument("--root", default=".", help="worktree root holding results/ and configs/")
    sw.add_argument("--expensive", default=None,
                    help="stage-2 table jsonl {qid, prediction, *_sec}")
    sw.add_argument("--rungs", type=int, default=1,
                    help="depth of the PRE-DECLARED stage-1 ladder (1 = parse-fail only)")
    sw.add_argument("--p", type=float, nargs="*", default=None, help="budget grid")
    sw.add_argument("--outer_folds", type=int, default=5)
    sw.add_argument("--inner_folds", type=int, default=3)
    sw.add_argument("--seed", type=int, default=20260905)
    sw.add_argument("--select_by", default="youden", choices=sorted(SELECTORS))
    sw.add_argument("--reps", type=int, default=2000, help="bootstrap / null reps")
    sw.add_argument("--allow_partial_expensive", action="store_true",
                    help="emit p>0 rows as family=diagnostic_partial when stage-2 "
                         "coverage differs by class (NOT reportable)")
    sw.add_argument("--features_out", default=None)
    sw.add_argument("--out", required=True)
    sw.set_defaults(func=_cmd_sweep)

    ce = sub.add_parser("ceiling", help="GCN detection ceiling (2502.15845 Sec. 4.1)")
    ce.add_argument("--root", default=".")
    ce.add_argument("--sidecar", default=None, help="results/t34/sidecar_c2kv.jsonl")
    ce.add_argument("--node_features", default="query_lexical",
                    choices=("query_lexical", "gold_witness_label"))
    ce.add_argument("--epochs", type=int, default=200)
    ce.add_argument("--outer_folds", type=int, default=5)
    ce.add_argument("--inner_folds", type=int, default=3)
    ce.add_argument("--seed", type=int, default=20260905)
    ce.add_argument("--n_perm", type=int, default=0)
    ce.add_argument("--allow_partial", action="store_true")
    ce.add_argument("--out", required=True)
    ce.set_defaults(func=_cmd_ceiling)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
