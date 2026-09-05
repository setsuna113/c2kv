# -*- coding: utf-8 -*-
"""t34 unit U10b -- digest section 4.11 controls (lines 1743-1797).

Two papers, both zero-GPU re-analysis of frozen assets:

* **2606.21399** "Calibration Is Not Control: Why LLM-Agent Oversight Needs
  Intervention" -- the *control* target ``tau_adv`` next to our *prediction*
  target (the C->W label), the RF+LCB witness controller, the abstraction loss
  ``Gap(g) = V* - V_g``, a control-regret table and the exploitability
  diagnostic.
* **2608.10441** "Detecting an Effect Is Not Learning to Act on It: A Reward-SNR
  Floor for LLM Acquisition Agents" -- the three guards: ``random@matched-rate``
  as a mandatory comparator row, the matched-moment noise **placebo** on the
  best-k locator envelope, and the **positive control** at a dialled signal
  strength through the SAME nested-CV + metric code.

Every quantity that consumes an arm's realised correctness is on the LABEL side
and is named ``*_label_*`` so it cannot drift into a feature frame (HARD RULE 4).
The only feature frame this module writes is the free prefix frame
(:data:`PREFIX_FEATURE_COLUMNS` + ``step_idx``), through
``t34_common.write_features_jsonl``; its pre-declared risk orientations are in
``configs/t34/orientations_controls.json``.  The *synthetic* positive-control
signal (``pc_synthetic_signal``, orientation +1) is declared there too but is
deliberately NEVER written to disk -- it is built from ``y`` and would be a leak
by construction, so it exists only inside the positive-control pipeline.

RUNBOOK (every command runs HERE on the Windows box; zero GPU, zero server)
--------------------------------------------------------------------------
Run from the worktree root ``C:/Users/yl998/Documents/programming/c2kv/tmp/t34-migration``
with ``PYTHONIOENCODING=utf-8``.

1. tau_adv / population split + cost x penalty sweep (2606.21399 step 1-2)::

     python agent/t34_controls.py tau-adv --root . \
         --arms results/bdf_pilot/d_r2/d_corr.jsonl \
         --arms results/bdf_pilot/d_r2/d_corr_re.jsonl \
         --arms results/bdf_pilot/d_r2/d_sham.jsonl \
         --harm-weight 1.0 --out results/t34/controls_tau_adv.json

2. witness controller (RF+LCB) regret table, abstraction Gap, exploitability
   (2606.21399 step 2-3); ``--write-features`` dumps the free prefix frame
   through the leakage guard::

     python agent/t34_controls.py witness-regret --root . \
         --arms results/bdf_pilot/d_r2/d_corr.jsonl \
         --arms results/bdf_pilot/d_r2/d_corr_re.jsonl \
         --arms results/bdf_pilot/d_r2/d_sham.jsonl \
         --write-features results/t34/features_controls_prefix.jsonl \
         --out results/t34/controls_regret.json

   Add ``--include-quit`` for the paper's three-way action set (U_quit = 0,
   section 9.2).  Regret LEVELS are comparable to Table tab:fork only in that
   run; without it the oracle is strictly smaller and only the within-table
   ordering transfers.  The flag moves the controller, the abstraction Gap and
   the exploitability oracle action together.

3. random@matched-rate comparator band beside parse-failure-only
   (2608.10441 guard 1)::

     python agent/t34_controls.py random-rate --root . --n-fires 20 \
         --out results/t34/controls_random_rate.json

   ``--n-fires`` defaults to the parse-failure baseline's own fire count, so the
   two rows are matched by construction.  Add ``--ladder`` for the optional
   granularity diagnostic (per-instance -> KMeans K=4..64 -> declared regimes).

4. matched-moment placebo on the best-k envelope (2608.10441 guard 2); needs the
   frozen (qid, k) flip table copied back from the server
   (``~/bench_results/d_v2/`` -> ``results/t34/flip_table.jsonl``, one
   ``{qid, k, correct}`` object per line)::

     python agent/t34_controls.py placebo-bestk --root . \
         --flips results/t34/flip_table.jsonl \
         --out results/t34/controls_placebo_bestk.json

   A missing or empty flip file ABORTS with exit code 2 rather than producing a
   placebo over nothing; the ``flip_table`` block in the report states how many
   (qid, k) trials were actually loaded against the digest's 823, so a smoke
   table can never be read as the guard-2 result.

5. positive control: detection strength s* of the pre-registered bar, and where
   a real feature matrix sits (2608.10441 guard 3)::

     python agent/t34_controls.py positive-control --root . \
         [--features results/t34/features_<unit>.jsonl] \
         --out results/t34/controls_positive_control.json

6. reward-SNR floor arithmetic, with the mandatory Delta redefinition
   (2608.10441 Prop. 1)::

     python agent/t34_controls.py snr-floor --root . \
         --arms results/bdf_pilot/d_r2/d_corr.jsonl --arm corr \
         --delta-definition "Delta_i = 1[arm corr correct] - 1[c2kv correct]" \
         --out results/t34/controls_snr_floor.json

   ``--arm`` takes the file's own ``d_arm`` value (``corr`` / ``corr_re`` /
   ``sham``), NOT the ``d_``-prefixed file stem.  A name that matches nothing
   now raises instead of reporting ``N = 0`` / ``above_floor: false``.

No step in this runbook touches the NPU server.  Step 4 is the only one that
needs an artefact that is not already on this box.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import t34_common as C  # noqa: E402
from t33_labels import load_jsonl  # noqa: E402


PAPER_CONTROL = "2606.21399"      # Calibration Is Not Control
PAPER_SNR = "2608.10441"          # Reward-SNR floor / placebo / positive control


DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "tau_adv / utility model",
        "paper": PAPER_CONTROL,
        "what": "The paper fixes a single flat intervention cost per benchmark "
                "(ALFWorld c_a=0.05, s=0.01, w=1.0, section 9.2).  We normalise c_a "
                "PER ARM from the measured KV bytes and GPU-seconds of that arm, "
                "rescaled onto [0, cost_scale] with cost_scale defaulting to the "
                "paper's 0.05.",
        "why": "Our K repair arms differ in cost by an order of magnitude "
               "(slice-prefill vs full recompute); a flat c_a would erase exactly "
               "the cost asymmetry the utility model exists to express.  The "
               "paper-faithful flat-cost variant is kept behind cost_mode='flat'.",
    },
    {
        "method": "tau_adv / branch length m(h,a)",
        "paper": PAPER_CONTROL,
        "what": "m is the number of remaining agent steps in the paper; we use the "
                "arm's generated-token count divided by the battery cap "
                "(manifest kv_recipe.max_new_tokens, 128) so m stays in [0,1].  "
                "The CONTINUE branch is charged m = 0 (and c = 0), so it pays no "
                "step cost at all while every arm pays s*m <= s.",
        "why": "The battery is single-step teacher-forced: there is no multi-step "
               "branch to measure.  Normalising keeps s comparable to the paper's "
               "per-step 0.01 instead of silently rescaling the harm term by ~128.  "
               "The m=0 continue branch makes the utility model conservative in "
               "the direction that matters (it never manufactures intervention "
               "advantage); at s=0.01 the asymmetry is bounded by 0.01 against a "
               "utility range of 1+w.",
    },
    {
        "method": "action set / quit is optional and OFF by default",
        "paper": PAPER_CONTROL,
        "what": "The paper's control set is three-way -- {continue, intervention, "
                "quit} -- with U_quit = 0 by definition (section 9.2, 'Quit always "
                "yields utility zero').  Our default action set is {continue} + the "
                "K repair arms; quit is available as an OPTIONAL action "
                "(``include_quit=True`` / ``--include-quit``) and is then a "
                "deterministic U = 0 column -- no forest, because its utility is "
                "not a random variable.  ``tau_adv`` keeps the paper's definition "
                "(EU_i - EU_c, intervention vs continue) and is NOT quit-augmented.",
        "why": "The battery has no abstain branch executed on disk: U_quit = 0 is an "
               "assumption, not a measured branch, which is why it is not the "
               "default.  It is load-bearing for comparability: with w = 1 quit "
               "dominates (U = 0 > -1) on exactly the rows where continue is wrong "
               "and no arm rescues, so WITHOUT it the tau_adv <= 0 subpopulation "
               "means 'no arm beats continuing' rather than 'no action beats "
               "quitting' and regret is measured against a strictly smaller oracle "
               "than Table tab:fork.  Only the ``include_quit=True`` table is on "
               "the paper's action set; ``regret_comparability`` in every report "
               "says which of the two was run.  ``--include-quit`` switches the "
               "controller, the abstraction Gap and the exploitability oracle "
               "action TOGETHER, so the three always share one action set "
               "(``action_set`` / ``gap_action_set`` in the report).",
    },
    {
        "method": "tau_adv / EU_continue",
        "paper": PAPER_CONTROL,
        "what": "EU_continue uses the compressed arm's own realised correctness "
                "rather than a fitted P(U_c <= 0 | h).",
        "why": "Single-step teacher forcing makes the continue branch a realised "
               "outcome, not a random variable to be predicted; the paper's own "
               "prefix-branching protocol executes the continue branch too.",
    },
    {
        "method": "witness controller",
        "paper": PAPER_CONTROL,
        "what": "RandomForest(200 trees, depth 6, min_samples_leaf=2) as in "
                "section 9.5; beta_a defaults to 1.0 because the paper's beta value is "
                "not stated in the sections the transfer card read.",
        "why": "Card explicitly records beta_a as unspecified; we expose it as an "
               "argument and report the value used instead of inventing the "
               "paper's number.",
    },
    {
        "method": "witness controller / scalar baseline",
        "paper": PAPER_CONTROL,
        "what": "The scalar p_fail is a probe score fitted on the 161-row trigger "
                "subset and read off on the arm-covered rows; the paper thresholds "
                "a p_fail estimate on the same rows it evaluates.  Because the two "
                "frames carry DIFFERENT session splits, every fitted use of the "
                "scalar -- the threshold baseline, the RF+LCB scalar ablation, the "
                "abstraction-Gap binning variable and the exploitability "
                "scalar-only predictor -- goes through ``make_pfail_scalar_fn``, "
                "which refits the probe inside each outer fold with that fold's "
                "sessions held out.  A fold whose refit cannot score every row is "
                "SKIPPED and counted "
                "(``n_folds_skipped_scalar_unavailable``), never filled in.",
        "why": "The D-line arm sweep only ran on the 93 C->W rows, where the C->W "
               "label is constant; a scalar fitted there would be degenerate, so "
               "the probe is trained on the frame where the label varies.  Reusing "
               "one out-of-fold vector across a different split would hand the "
               "paper's own baseline the evaluation fold's labels: it inflates "
               "acc_scalar_only (deflating the exploitability increment) and moves "
               "the Gap's bin edges.  ``scalar_split_alignment`` records which "
               "regime produced every number.",
    },
    {
        "method": "abstraction loss Gap(g)",
        "paper": PAPER_CONTROL,
        "what": "V* and V_g are estimated with REALISED per-row utilities and "
                "quantile bins of g, on session-grouped held-out folds, not with "
                "true conditional expectations.",
        "why": "We have realised branch outcomes (that is what the D-line sweep "
               "is) but no access to E[U_a | h]; the plug-in estimator keeps "
               "Gap >= 0 by max-of-means <= mean-of-maxes.",
    },
    {
        "method": "5x5 cost x penalty sweep / cell statistic",
        "paper": PAPER_CONTROL,
        "what": "The paper's sweep (section 5.3) counts how many of the 25 cells keep "
                "the SIGN OF THE CONTROLLER GAIN (witness regret below scalar "
                "regret; 25/25 on ALFWorld, 23/25 on ScienceWorld).  Our cell "
                "statistic is instead the fraction of rows with tau_adv > 0, and a "
                "cell 'keeps sign' when that fraction exceeds 0.5.",
        "why": "The sweep here is attached to the tau_adv population split, which is "
               "a pure re-analysis of realised arm outcomes and costs no fit; "
               "re-fitting the RF+LCB controller in all 25 cells would produce 25 "
               "regret differences that are all far below the n=93 MDE and would "
               "read as an arm ranking, which digest 4.0 forbids.  The gain-sign "
               "count is therefore NOT reported, and our count must not be compared "
               "against their 25/25.",
    },
    {
        "method": "regret / arm ranking",
        "paper": PAPER_CONTROL,
        "what": "We report a population split and a regret table but NEVER an arm "
                "ranking; MDE_PP (17-25 pp at n=93) is printed with every table.",
        "why": "Digest 4.0 forbids rankings finer than the MDE; the paper's "
               "ALFWorld n is far larger than 93.",
    },
    {
        "method": "random@matched-rate",
        "paper": PAPER_SNR,
        "what": "The paper draws random acquisitions i.i.d. at a budget b.  We draw "
                "EXACTLY b fires confined to a randomly chosen union of whole "
                "sessions (see random_matched_rate_masks for the scheme).",
        "why": "Our rows are clustered in 100 sessions on the 161-row frame; an "
               "i.i.d. row null would understate the variance of a candidate whose "
               "fires clump inside sessions.",
    },
    {
        "method": "matched-moment placebo",
        "paper": PAPER_SNR,
        "what": "The paper matches the moments of a CONTINUOUS Delta.  Our flip "
                "vector is binary, so matched moments means a per-entry "
                "Bernoulli(p) draw: level='per_qid' matches each qid's own flip "
                "rate (and hence its variance p(1-p)); level='pooled' matches the "
                "global non-witness flip rate and reproduces the analytic envelope "
                "1-(1-p)^n_docs of agent/d_ksweep_analysis.py.  BOTH levels now "
                "use the same denominator -- k != k_witness -- so p_q and p_bar "
                "are the per-qid and pooled versions of one quantity; a qid with "
                "no k_witness (or with only the witness k logged) falls back to "
                "all of its k and is counted in "
                "``n_qids_rate_over_all_k``.",
        "why": "For a binary vector the mean determines the variance; there is no "
               "second free moment to match.  Both levels are reported so the "
               "reader can see how much of the observed gap is per-qid "
               "over-dispersion rather than content-specific localisation.",
    },
    {
        "method": "positive control",
        "paper": PAPER_SNR,
        "what": "The paper injects a synthetic CLUSTER signal at a dialled "
                "cluster-SNR into a recommendation pipeline.  We inject a synthetic "
                "per-row scalar at correlation strength s (part row-level, part "
                "session-level) into t34_common.nested_cv_logistic, and define the "
                "bar as 'AUPRC clustered-bootstrap lower bound above the evaluation "
                "frame's prevalence'.",
        "why": "Our estimand is a binary trigger label, not an NDCG delta; the "
               "cluster-SNR scale does not transfer.  The bar is the computable "
               "subset of the digest 4.0 winner rule (the S0-twin and "
               "three-metric clauses need a real feature, not a synthetic one).",
    },
    {
        "method": "granularity ladder",
        "paper": PAPER_SNR,
        "what": "The paper's ladder has FOUR rungs -- per-instance, KMeans "
                "K=4..64, hand-defined cross-product regimes, and honest uplift "
                "trees (section 6).  Ours has three: the honest-uplift-tree rung "
                "is NOT implemented.  Empirical-Bayes shrinkage uses a fixed "
                "pseudo-count of 1 toward the training prior rather than a fitted "
                "prior variance, and KMeans runs with n_init=4.",
        "why": "No uplift-tree dependency is installed on this box.  The claim "
               "'no granularity rescues the policy' therefore covers three of "
               "their four rungs and must be written that way; a flat ladder here "
               "is weaker evidence than theirs.",
    },
    {
        "method": "reward-SNR floor rho*(N) = 2.8/sqrt(N)",
        "paper": PAPER_SNR,
        "what": "Reported as an MDE-in-disguise with a mandatory caller-supplied "
                "Delta definition string; never used as a post-hoc significance "
                "test, and never applied to the binary C->W label directly.",
        "why": "The paper's floor is a one-sample MEAN-detection bound on a "
               "continuous Delta.  Our label is binary with prevalence 0.1033 on "
               "the 900-frame; the repo already forbids using an MDE as a post-hoc "
               "test.",
    },
]


# ===========================================================================
# label-side loaders (arm outcomes are realised correctness -> LABEL side)
# ===========================================================================

#: canonical arm-outcome record produced by the loaders below
_ARM_FIELDS = ("qid", "arm", "correct", "gpu_sec", "kv_bytes", "gen_tokens")


def load_arm_outcomes_label(path: Path) -> List[Dict[str, Any]]:
    """Canonical arm-outcome loader (2606.21399 section 3, prefix branching).

    Reads a jsonl whose lines are ``{qid, arm, correct, gpu_sec, kv_bytes}``
    (``gen_tokens`` optional).  This is the D-line sweep's per-row outcome for
    each of the K repair arms and it is LABEL-side data: ``correct`` is the
    realised ``tool_name_match`` of that arm.  Never join it into a feature
    frame.
    """
    rows: List[Dict[str, Any]] = []
    for r in load_jsonl(str(path)):
        if r.get("skipped"):
            continue
        rows.append({
            "qid": r["qid"],
            "arm": str(r["arm"]),
            "correct": bool(r["correct"]),
            "gpu_sec": _opt_float(r.get("gpu_sec")),
            "kv_bytes": _opt_float(r.get("kv_bytes")),
            "gen_tokens": _opt_float(r.get("gen_tokens")),
        })
    return rows


def arm_outcomes_label_from_dline(paths: Sequence[Path],
                                  *,
                                  bytes_per_kv_token: Optional[float] = None,
                                  arm_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Adapter: frozen D-line arm jsonl -> the canonical arm-outcome schema.

    Field map (all fields are the arm file's own columns):

    ==================  =========================================
    canonical           D-line column
    ==================  =========================================
    ``arm``             ``d_arm`` (or ``arm_name`` override)
    ``correct``         ``tool_name_match``            [LABEL side]
    ``gpu_sec``         ``d_corr_slice_prefill_sec`` + ``d_recompute_prefill_sec``
                        + ``generate_sec`` when present, else ``latency_sec``
    ``kv_bytes``        ``d_corr_span_tokens`` * ``bytes_per_kv_token``
                        (None unless the caller supplies the constant)
    ``gen_tokens``      ``generated_tokens``
    ==================  =========================================

    ``bytes_per_kv_token`` is NOT guessed: without it ``kv_bytes`` stays None and
    the cost model falls back to GPU-seconds only (and says so in its report).
    """
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for r in load_jsonl(str(path)):
            if r.get("skipped"):
                continue
            gpu = _sum_opt(r.get("d_corr_slice_prefill_sec"),
                           r.get("d_recompute_prefill_sec"),
                           r.get("generate_sec"))
            if gpu is None:
                gpu = _opt_float(r.get("latency_sec"))
            span = _opt_float(r.get("d_corr_span_tokens"))
            kv_bytes = (span * bytes_per_kv_token) if (span is not None and
                                                       bytes_per_kv_token) else None
            rows.append({
                "qid": r["qid"],
                "arm": str(arm_name or r.get("d_arm") or Path(path).stem),
                "correct": bool(r.get("tool_name_match")),
                "gpu_sec": gpu,
                "kv_bytes": kv_bytes,
                "gen_tokens": _opt_float(r.get("generated_tokens")),
            })
    return rows


def continue_outcomes_label(frame: "C.FrozenFrame",
                            qids: Iterable[str]) -> Dict[str, bool]:
    """Realised outcome of the ``continue`` (do-nothing) branch per qid.

    2606.21399 section 2.1: ``EU_c(h)``.  On the frozen battery this is the
    compressed arm's own ``tool_name_match`` -- LABEL side, never a feature.
    """
    by_qid = frame.c2kv_by_qid
    return {q: bool(by_qid[q].get("tool_name_match")) for q in qids if q in by_qid}


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _sum_opt(*vals: Any) -> Optional[float]:
    got = [_opt_float(v) for v in vals]
    got = [g for g in got if g is not None]
    return float(sum(got)) if got else None


# ===========================================================================
# (A) 2606.21399 -- utility model, tau_adv, witness controller, Gap, regret
# ===========================================================================

def arm_cost_table(arm_rows: Sequence[Dict[str, Any]],
                   *,
                   cost_scale: float = 0.05,
                   byte_weight: float = 0.5,
                   time_weight: float = 0.5,
                   cost_mode: str = "normalised") -> Dict[str, Dict[str, Any]]:
    """Per-arm intervention cost ``c_a`` (2606.21399 section 9.2, Table of costs).

    The paper fixes ``c_a = 0.05`` (ALFWorld/ScienceWorld) or ``0.10``
    (GSM8K/HotpotQA) flat.  ``cost_mode='flat'`` reproduces that.  The default
    ``cost_mode='normalised'`` spends the same budget but splits it across arms
    by their measured resources::

        c_a = cost_scale * ( byte_weight * bytes_a / max_a bytes_a
                           + time_weight * sec_a   / max_a sec_a )

    with the missing component's weight redistributed when an arm file carries
    no ``kv_bytes`` (see DEVIATIONS).  ``bytes_a`` / ``sec_a`` are the arm's mean
    over the rows it covers.
    """
    if cost_mode not in ("normalised", "flat"):
        raise ValueError("cost_mode must be 'normalised' or 'flat'")
    arms = sorted({r["arm"] for r in arm_rows})
    mean_bytes: Dict[str, Optional[float]] = {}
    mean_sec: Dict[str, Optional[float]] = {}
    mean_m: Dict[str, float] = {}
    for a in arms:
        rows = [r for r in arm_rows if r["arm"] == a]
        b = [r["kv_bytes"] for r in rows if r["kv_bytes"] is not None]
        s = [r["gpu_sec"] for r in rows if r["gpu_sec"] is not None]
        g = [r["gen_tokens"] for r in rows if r["gen_tokens"] is not None]
        mean_bytes[a] = float(np.mean(b)) if b else None
        mean_sec[a] = float(np.mean(s)) if s else None
        mean_m[a] = float(np.mean(g)) if g else 0.0
    have_bytes = any(v is not None for v in mean_bytes.values())
    have_sec = any(v is not None for v in mean_sec.values())
    wb = byte_weight if have_bytes else 0.0
    wt = time_weight if have_sec else 0.0
    tot = wb + wt
    if tot <= 0:
        wb, wt, tot = 0.0, 1.0, 1.0  # degenerate: flat cost
    max_b = max([v for v in mean_bytes.values() if v is not None] or [1.0]) or 1.0
    max_s = max([v for v in mean_sec.values() if v is not None] or [1.0]) or 1.0
    out: Dict[str, Dict[str, Any]] = {}
    for a in arms:
        if cost_mode == "flat":
            c_a = cost_scale
        else:
            frac = 0.0
            if wb:
                frac += wb * ((mean_bytes[a] or 0.0) / max_b)
            if wt:
                frac += wt * ((mean_sec[a] or 0.0) / max_s)
            c_a = cost_scale * frac / tot
        out[a] = {
            "c_a": float(c_a),
            "mean_kv_bytes": mean_bytes[a],
            "mean_gpu_sec": mean_sec[a],
            "mean_gen_tokens": mean_m[a],
            "n_rows": sum(1 for r in arm_rows if r["arm"] == a),
        }
    return out


#: the digest's ten D-line repair arms (section 4.11).  Only a subset is on this
#: box; the missing ones can only RAISE ``max_a EU_a``, so a short arm table makes
#: the ``tau_adv <= 0`` split an upper bound (see :func:`arm_coverage_status`).
DIGEST_ARM_SWEEP_N = 10


def cost_inputs_status(arm_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Which cost axes actually carry data -- reported beside every ``c_a``.

    ``kv_bytes`` is None on every local arm file (the D-line jsonl logs
    ``d_corr_span_tokens``, not bytes), so unless the caller supplies
    ``--bytes-per-kv-token`` the per-arm cost rests on GPU-seconds alone.  That
    is a MISSING INPUT, not a modelling choice, and it is named here rather than
    guessed: the byte-vs-time asymmetry that motivates the per-arm normalisation
    deviation is unexercised while ``kv_bytes_available`` is false.
    """
    arms = sorted({r["arm"] for r in arm_rows})
    miss_b = [a for a in arms
              if not any(r["kv_bytes"] is not None for r in arm_rows if r["arm"] == a)]
    miss_s = [a for a in arms
              if not any(r["gpu_sec"] is not None for r in arm_rows if r["arm"] == a)]
    axes = [ax for ax, ok in (("kv_bytes", len(miss_b) < len(arms)),
                              ("gpu_sec", len(miss_s) < len(arms))) if ok]
    return {
        "kv_bytes_available": bool(len(miss_b) < len(arms)),
        "gpu_sec_available": bool(len(miss_s) < len(arms)),
        "arms_missing_kv_bytes": miss_b,
        "arms_missing_gpu_sec": miss_s,
        "cost_axes_used": axes,
        "note": _cost_inputs_note(len(arms), len(miss_b), len(miss_s)),
    }


def _cost_inputs_note(n_arms: int, n_miss_bytes: int, n_miss_sec: int) -> str:
    if n_arms and n_miss_bytes == n_arms:
        return ("MISSING INPUT: kv_bytes is unknown for every arm, so c_a rests on "
                "GPU-seconds alone and the byte-vs-time asymmetry that motivates "
                "the per-arm normalisation is UNEXERCISED.  Pass "
                "--bytes-per-kv-token (bytes per appended KV token at this recipe) "
                "to switch the byte axis on; it is deliberately not guessed.")
    if n_miss_bytes or n_miss_sec:
        return ("Some arms are missing a cost axis; their contribution on that axis "
                "is 0 of the budget, which under-charges them.")
    return "Both cost axes carry data on every arm."


def arm_coverage_status(arm_rows: Sequence[Dict[str, Any]],
                        *, expected_n: int = DIGEST_ARM_SWEEP_N) -> Dict[str, Any]:
    """How much of the digest's ten-arm sweep is actually loaded.

    ``tau_adv = max_a EU_a - EU_none`` is monotone in the arm set: adding an arm
    can only raise ``max_a EU_a``.  So with fewer arms than the sweep, the
    ``tau_adv <= 0`` population is an UPPER bound on the no-advantage subgroup
    and the ``tau_adv > 0`` population a lower bound.  Reported, never silently
    absorbed.
    """
    arms = sorted({r["arm"] for r in arm_rows})
    complete = len(arms) >= int(expected_n)
    return {
        "arms_loaded": arms,
        "n_arms_loaded": len(arms),
        "n_arms_digest_sweep": int(expected_n),
        "arm_table_complete": bool(complete),
        "note": ("Complete arm table." if complete else
                 f"Only {len(arms)} of the digest's {expected_n} repair arms are "
                 "loaded.  tau_adv is monotone in the arm set, so the "
                 "tau_adv <= 0 count here is an UPPER bound on the no-advantage "
                 "subgroup (and n_positive a lower bound); rerun once the full "
                 "arm table is copied back from the server."),
    }


def utility(correct: bool, *, c_a: float, w: float, s: float, m: float) -> float:
    """``U_a(h) = u(h,a) - c_a - s*m(h,a)`` (2606.21399 section 9.2).

    ``u = 1`` for a correct outcome and ``-w`` for an incorrect one; ``c_a`` is
    the intervention cost, ``s`` the per-step cost and ``m`` the branch length
    (here: generated tokens / cap, see DEVIATIONS).  ``continue`` takes
    ``c_a = 0``.
    """
    u = 1.0 if correct else -float(w)
    return float(u - c_a - s * m)


def tau_adv_label_table(arm_rows: Sequence[Dict[str, Any]],
                        continue_correct: Dict[str, bool],
                        labels: Dict[str, Optional[int]],
                        *,
                        w: float = 1.0,
                        s: float = 0.01,
                        cap_tokens: int = 128,
                        costs: Optional[Dict[str, Dict[str, Any]]] = None,
                        cost_scale: float = 0.05,
                        cost_mode: str = "normalised") -> Dict[str, Any]:
    """``tau_adv(row) = max_a EU_a - EU_none`` and its sign agreement with C->W.

    2606.21399 section 2.1 (prediction target ``p_fail`` vs control target
    ``tau_adv``) and Prop. ``prop:sufficiency`` (a scalar supports lossless
    routing iff it recovers ``sign(tau_adv)``).

    Single-step teacher forcing makes every branch a realised outcome, so
    ``EU_a`` is the realised ``U_a`` (see DEVIATIONS).  Returns per-row
    ``tau_adv``, the sign x label contingency, and the ``tau_adv <= 0``
    subpopulation -- the rows where firing is pure cost.

    LABEL side: consumes realised arm correctness.  Not a feature.
    """
    costs = costs or arm_cost_table(arm_rows, cost_scale=cost_scale,
                                    cost_mode=cost_mode)
    cap = max(1, int(cap_tokens))
    by_qid: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in arm_rows:
        by_qid.setdefault(r["qid"], {})[r["arm"]] = r

    per_row: Dict[str, Dict[str, Any]] = {}
    for qid, arms in sorted(by_qid.items()):
        if qid not in continue_correct:
            continue
        eu_none = utility(continue_correct[qid], c_a=0.0, w=w, s=s, m=0.0)
        eus: Dict[str, float] = {}
        for a, r in arms.items():
            m = (r["gen_tokens"] or 0.0) / cap
            eus[a] = utility(r["correct"], c_a=costs[a]["c_a"], w=w, s=s, m=m)
        best_arm = max(eus, key=lambda a: (eus[a], a))
        tau = eus[best_arm] - eu_none
        per_row[qid] = {
            "tau_adv": float(tau),
            "eu_none": float(eu_none),
            "best_arm": best_arm,
            "eu_by_arm": {a: float(v) for a, v in sorted(eus.items())},
            "label_cw": labels.get(qid),
            "any_arm_correct": bool(any(r["correct"] for r in arms.values())),
        }

    tau = np.array([v["tau_adv"] for v in per_row.values()], dtype=float)
    y = np.array([1 if per_row[q]["label_cw"] == 1 else 0 for q in per_row], dtype=int)
    has_label = np.array([per_row[q]["label_cw"] in (0, 1) for q in per_row], dtype=bool)
    sign = np.sign(tau)

    cont: Dict[str, int] = {}
    for sg in (-1, 0, 1):
        for lb in (0, 1):
            cont[f"sign{sg:+d}_label{lb}"] = int(((sign == sg) & (y == lb) & has_label).sum())
    n = int(has_label.sum())
    agree = int((((tau > 0) & (y == 1)) | ((tau <= 0) & (y == 0)))[has_label].sum())

    return {
        "paper": PAPER_CONTROL,
        "n_rows": len(per_row),
        "n_rows_with_label": n,
        "n_arms": len(costs),
        "arms": costs,
        "cost_inputs": cost_inputs_status(arm_rows),
        "arm_coverage": arm_coverage_status(arm_rows),
        "action_set": ["continue"] + sorted(costs),
        "regret_comparability": (
            "tau_adv is the paper's EU_i - EU_c (intervention vs continue) and is "
            "NOT quit-augmented; the quit action lives on the controller "
            "(witness_controller_regret_label(include_quit=True)).  See DEVIATIONS "
            "'action set / quit is optional and OFF by default'."),
        "params": {"w": w, "s": s, "cost_scale": cost_scale,
                   "cost_mode": cost_mode, "cap_tokens": cap},
        "tau_adv": {
            "mean": float(tau.mean()) if tau.size else None,
            "median": float(np.median(tau)) if tau.size else None,
            "n_positive": int((tau > 0).sum()),
            "n_zero": int((tau == 0).sum()),
            "n_non_positive": int((tau <= 0).sum()),
            "frac_non_positive": float((tau <= 0).mean()) if tau.size else None,
        },
        "sign_label_contingency": cont,
        "sign_agreement_rate": (agree / n) if n else None,
        "label_support": {
            "n_label1": int(((y == 1) & has_label).sum()),
            "n_label0": int(((y == 0) & has_label).sum()),
            "both_classes_present": bool(((y == 1) & has_label).any()
                                         and ((y == 0) & has_label).any()),
            "note": ("Prop. prop:sufficiency asks whether the label recovers "
                     "sign(tau_adv); with only ONE label class on this frame the "
                     "agreement rate degenerates to the fraction of rows with "
                     "positive advantage and says nothing about sufficiency.  "
                     "The D-line arm sweep covers only the 93 C->W rows, so the "
                     "C->C column needs the never-run harm arms."),
        },
        "per_row": per_row,
        "mde_pp": list(C.MDE_PP),
        "scope": ("population partition only -- at n=93 the MDE is "
                  f"{C.MDE_PP[0]}-{C.MDE_PP[1]} pp, so this cannot rank arms "
                  "(digest 4.0)."),
    }


def cost_penalty_sweep_label(arm_rows: Sequence[Dict[str, Any]],
                             continue_correct: Dict[str, bool],
                             labels: Dict[str, Optional[int]],
                             *,
                             cost_grid: Sequence[float] = (0.01, 0.025, 0.05, 0.10, 0.20),
                             w_grid: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
                             **kwargs: Any) -> Dict[str, Any]:
    """5x5 intervention-cost x wrong-answer-penalty sweep (2606.21399 section 5.3).

    The paper reports how many cells keep the sign of the effect (25/25 on
    ALFWorld, 23/25 on ScienceWorld).  Our cell statistic is the fraction of
    rows with ``tau_adv > 0``; a cell "keeps sign" when that fraction is > 0.5.
    """
    cells = []
    for c in cost_grid:
        for w in w_grid:
            t = tau_adv_label_table(arm_rows, continue_correct, labels,
                                    w=w, cost_scale=c, **kwargs)
            n_rows = max(1, t["n_rows"])
            cells.append({"cost_scale": float(c), "w": float(w),
                          "frac_tau_positive": t["tau_adv"]["n_positive"] / n_rows,
                          "frac_tau_non_positive": t["tau_adv"]["frac_non_positive"]})
    kept = sum(1 for cell in cells if (cell["frac_tau_positive"] or 0.0) > 0.5)
    return {"paper": PAPER_CONTROL, "n_cells": len(cells),
            "cells_sign_kept": kept, "cells": cells,
            "cell_statistic": "fraction of rows with tau_adv > 0; a cell 'keeps "
                              "sign' when that fraction exceeds 0.5",
            "scope": ("NOT the paper's statistic: section 5.3 counts cells where "
                      "the CONTROLLER GAIN keeps its sign (25/25 ALFWorld). This "
                      "count must never be compared against theirs -- see "
                      "DEVIATIONS '5x5 cost x penalty sweep / cell statistic'.")}


# ---------------------------------------------------------------------------
# witness controller: RF + pessimistic LCB
# ---------------------------------------------------------------------------

def _rf(seed: int):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=200, max_depth=6,
                                  min_samples_leaf=2, random_state=seed,
                                  n_jobs=1)


def _rf_fit_predict_lcb(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
                        *, beta: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 200-tree / depth-6 forest and return ``(p_hat, sigma_hat)``.

    2606.21399 section 4.1: "a random forest because it provides per-tree variance
    estimates for the confidence bound".  ``sigma_hat`` is the spread of the
    per-tree class-1 probabilities (the tree-vote spread); the pessimistic
    estimate used by the controller is ``p_hat - beta * sigma_hat``.
    """
    y_tr = np.asarray(y_tr, dtype=int)
    if len(np.unique(y_tr)) < 2:
        p = np.full(len(X_te), float(y_tr[0]) if len(y_tr) else 0.0)
        return p, np.zeros(len(X_te))
    rf = _rf(seed).fit(X_tr, y_tr)
    pos = int(np.where(rf.classes_ == 1)[0][0])
    per_tree = np.stack([est.predict_proba(X_te)[:, pos] for est in rf.estimators_])
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def witness_controller_regret_label(
    arm_rows: Sequence[Dict[str, Any]],
    continue_correct: Dict[str, bool],
    features: Dict[str, Dict[str, float]],
    scalar: Dict[str, float],
    *,
    w: float = 1.0,
    s: float = 0.01,
    cap_tokens: int = 128,
    beta: float = 1.0,
    cost_scale: float = 0.05,
    cost_mode: str = "normalised",
    outer_folds: int = 5,
    seed: int = 20260905,
    scalar_fit_fn: Optional[Callable[[set], Dict[str, float]]] = None,
    include_quit: bool = False,
) -> Dict[str, Any]:
    """RF+LCB witness controller vs scalar routing, scored by control regret.

    2606.21399 sections 4.1 / 4.3 and Table ``tab:fork``.  For each action ``a``
    a forest predicts ``p_hat_a = P(success | h, a)`` from the *prefix* features;
    ``EU_a = p_hat_a r_a^+ + (1 - p_hat_a) r_a^-`` with
    ``r_a^+ = 1 - c_a - s m_a`` and ``r_a^- = -w - c_a - s m_a``; the controller
    takes the pessimistic ``p_hat_a - beta sigma_hat_a`` before the argmax.
    ``Regret(pi) = U(pi_oracle) - U(pi)`` with ``pi_oracle`` the best REALISED
    branch per row.

    ``continue`` is action index 0 and gets its own forest, exactly as in the
    paper ("The witness evaluates each candidate action at the current prefix,
    **not just the continuation branch**", section 4.1) with ``c_continue = 0``
    and ``m_continue = 0``.  The controller therefore never reads the realised
    continue outcome at decision time -- that outcome IS the label component,
    and comparing a predicted ``EU_a`` against the realised ``U_continue``
    would hand the policy the answer it is being scored on.

    Three comparators, all on the same session-grouped held-out folds:

    * ``continue_always`` -- never intervene;
    * ``scalar_threshold`` -- the paper's Def. ``def:ft`` failure-trigger: fire the
      single best training arm when the risk scalar exceeds a threshold chosen on
      INNER folds (digest 4.0 item 6);
    * ``scalar_rf_lcb`` -- the paper's load-bearing ablation (section 5.1): the SAME
      RF+LCB family fed only the one-dimensional scalar;
    * ``witness`` -- RF+LCB on the full prefix feature vector.

    LABEL side (arm correctness is the training target of ``p_hat_a``); the
    FEATURES are the compressed arm's own free prefix scalars.

    ``include_quit=True`` appends the paper's third action: a deterministic
    ``U_quit = 0`` column ("Quit always yields utility zero", section 9.2).  It
    gets no forest -- its utility is not a random variable -- so the controller
    simply compares every pessimistic ``EU_a`` against 0.  Only with quit is the
    action set the paper's, and only then are regret LEVELS comparable to Table
    ``tab:fork``; ``regret_comparability`` in the report says which was run.
    The threshold baseline (Def. ``def:ft``) stays a fire / do-not-fire rule: its
    do-not-fire branch becomes the better of {continue, quit} on the TRAINING
    rows when quit is available.

    ``scalar_fit_fn(held_out_sessions) -> {qid: p_fail score}`` refits the risk
    scalar inside EVERY outer fold, excluding that fold's sessions.  Without it
    the ``scalar`` dict is reused as given, and because that dict is produced by
    a probe cross-validated on the 161-row frame (a DIFFERENT session split),
    the scalar values on the controller's TRAINING rows were produced by models
    that had seen the controller's test-fold labels -- an advantage handed to
    the paper's own baseline.  ``scalar_split_alignment`` in the report records
    which of the two was used; use the callable for anything reportable.  An
    outer fold whose refit cannot score every row is left UNSCORED and counted
    in ``n_folds_skipped_scalar_unavailable`` -- never back-filled with the
    reused scalar, and never silently absorbed into the remaining folds.

    The report also carries ``cost_inputs`` and ``arm_coverage`` beside the
    regret numbers: ``c_a`` rests on whatever cost axes the arm files actually
    logged, and with fewer than the digest's ten arms every regret level in the
    table is measured against a smaller oracle than the full sweep would give.
    """
    costs = arm_cost_table(arm_rows, cost_scale=cost_scale, cost_mode=cost_mode)
    cap = max(1, int(cap_tokens))
    arms = sorted(costs)
    by_qid: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in arm_rows:
        by_qid.setdefault(r["qid"], {})[r["arm"]] = r

    qids = [q for q in sorted(by_qid)
            if q in continue_correct and q in features and q in scalar
            and set(by_qid[q]) == set(arms)]
    if not qids:
        return {"paper": PAPER_CONTROL, "n_rows": 0,
                "note": "no rows with all arms + features + scalar"}

    qids, X, feat_names, dropped = feature_matrix(features, qids)
    if not qids:
        return {"paper": PAPER_CONTROL, "n_rows": 0,
                "note": "no rows with usable prefix features"}
    g = np.array([float(scalar[q]) for q in qids]).reshape(-1, 1)
    groups = np.array([C.session_of(q) for q in qids])

    # realised utilities per (row, action); column 0 = continue, then the arms,
    # then (optionally) quit as the LAST column with U = 0 by definition.
    n_pred = len(arms) + 1                            # actions that need a forest
    quit_idx = n_pred if include_quit else None
    n_act = n_pred + (1 if include_quit else 0)
    U = np.zeros((len(qids), n_act))
    Y = np.zeros((len(qids), n_act), dtype=int)       # per-ACTION success labels
    for i, q in enumerate(qids):
        U[i, 0] = utility(continue_correct[q], c_a=0.0, w=w, s=s, m=0.0)
        Y[i, 0] = int(bool(continue_correct[q]))
        for j, a in enumerate(arms):
            r = by_qid[q][a]
            m = (r["gen_tokens"] or 0.0) / cap
            U[i, 1 + j] = utility(r["correct"], c_a=costs[a]["c_a"], w=w, s=s, m=m)
            Y[i, 1 + j] = int(r["correct"])
        if quit_idx is not None:
            U[i, quit_idx] = 0.0                      # "Quit always yields zero"

    # r_a^+ / r_a^- per PREDICTED ACTION (index 0 = continue: no cost, m = 0)
    r_plus = np.array([1.0] + [1.0 - costs[a]["c_a"] - s * (costs[a]["mean_gen_tokens"] / cap)
                               for a in arms])
    r_minus = np.array([-w] + [-w - costs[a]["c_a"] - s * (costs[a]["mean_gen_tokens"] / cap)
                               for a in arms])

    chosen = {name: np.full(len(qids), -1, dtype=int)
              for name in ("continue_always", "scalar_threshold", "scalar_rf_lcb", "witness")}
    chosen["continue_always"][:] = 0
    thresholds: List[float] = []
    n_folds_skipped = 0

    for fold, te_mask in enumerate(C.grouped_folds(groups, outer_folds, seed)):
        tr = ~te_mask
        te = te_mask
        if te.sum() == 0 or tr.sum() == 0:
            continue
        g_fold = g
        if scalar_fit_fn is not None:
            # refit the risk scalar with THIS fold's sessions held out, so no
            # value of g -- on train or test rows -- was produced by a model
            # that saw a test-fold label (digest 4.0 item 6).
            sc_fold = scalar_fit_fn(set(groups[te].tolist()))
            g_fold = np.array([[float(sc_fold.get(q, np.nan))] for q in qids])
            if not np.isfinite(g_fold[tr]).all() or not np.isfinite(g_fold[te]).all():
                # MISSING INPUT, not an imputable hole: the fold stays unscored
                # and is counted, never filled with the reused scalar.
                n_folds_skipped += 1
                continue
        # --- witness + scalar RF+LCB: one forest per ACTION, continue included
        #     ("The witness evaluates each candidate action at the current
        #      prefix, not just the continuation branch", section 4.1).  The
        #     realised continue outcome is NEVER read at decision time.
        eu_w = np.zeros((int(te.sum()), n_act))
        eu_s = np.zeros((int(te.sum()), n_act))
        for j in range(n_pred):
            p, sd = _rf_fit_predict_lcb(X[tr], Y[tr, j], X[te], beta=beta, seed=seed + j)
            p_lcb = np.clip(p - beta * sd, 0.0, 1.0)
            eu_w[:, j] = p_lcb * r_plus[j] + (1 - p_lcb) * r_minus[j]
            p2, sd2 = _rf_fit_predict_lcb(g_fold[tr], Y[tr, j], g_fold[te],
                                          beta=beta, seed=seed + j)
            p2_lcb = np.clip(p2 - beta * sd2, 0.0, 1.0)
            eu_s[:, j] = p2_lcb * r_plus[j] + (1 - p2_lcb) * r_minus[j]
        if quit_idx is not None:
            # U_quit = 0 deterministically: no forest, nothing to predict.
            eu_w[:, quit_idx] = 0.0
            eu_s[:, quit_idx] = 0.0
        for name, eu in (("witness", eu_w), ("scalar_rf_lcb", eu_s)):
            chosen[name][te] = eu.argmax(axis=1)      # 0 = continue
        # --- scalar threshold (Def. def:ft): best training arm above a threshold
        #     picked on INNER folds of the training rows.  The do-not-fire branch
        #     is continue, or the better of {continue, quit} when quit exists.
        j_star = 1 + int(np.argmax(U[tr][:, 1:n_pred].mean(axis=0)))
        j_low = 0
        if quit_idx is not None:
            j_low = int(0 if U[tr][:, 0].mean() >= U[tr][:, quit_idx].mean()
                        else quit_idx)
        theta = _select_threshold_inner(g_fold[tr, 0], U[tr], j_star, groups[tr], seed,
                                        j_low=j_low)
        thresholds.append(float(theta))
        chosen["scalar_threshold"][te] = np.where(g_fold[te, 0] > theta, j_star, j_low)

    scored = chosen["witness"] >= 0
    oracle = U.max(axis=1)
    out_pol: Dict[str, Any] = {}
    for name, pick in chosen.items():
        ok = scored & (pick >= 0)
        realised = U[np.arange(len(qids)), np.maximum(pick, 0)]
        is_arm = (pick >= 1) & (pick < n_pred)        # quit is NOT an intervention
        out_pol[name] = {
            "utility": float(realised[ok].mean()) if ok.any() else None,
            "regret": float((oracle[ok] - realised[ok]).mean()) if ok.any() else None,
            "intervene_rate": float(is_arm[ok].mean()) if ok.any() else None,
            "continue_rate": float((pick[ok] == 0).mean()) if ok.any() else None,
            "quit_rate": (float((pick[ok] == quit_idx).mean())
                          if (ok.any() and quit_idx is not None) else None),
            "n_scored": int(ok.sum()),
        }
    out_pol["oracle"] = {"utility": float(oracle[scored].mean()) if scored.any() else None,
                         "regret": 0.0, "n_scored": int(scored.sum())}

    return {
        "paper": PAPER_CONTROL,
        "n_rows": len(qids),
        "n_dropped_nan_features": dropped,
        "n_sessions": int(len(set(groups))),
        "arms": arms,
        "action_set": ["continue"] + arms + (["quit"] if include_quit else []),
        "include_quit": bool(include_quit),
        "regret_comparability": (
            "action set = {continue, K arms, quit} with U_quit = 0: the paper's "
            "three-way set (section 9.2), so regret LEVELS are on the same "
            "oracle as Table tab:fork."
            if include_quit else
            "action set = {continue, K arms} -- NO quit.  U_quit = 0 would "
            "dominate on every row where continue is wrong and no arm rescues "
            "(w = 1 => U_continue = -1 < 0), so the oracle here is strictly "
            "smaller than Table tab:fork's and the regret LEVELS are not "
            "comparable to theirs; only the within-table ordering is.  Rerun "
            "with include_quit=True / --include-quit for the paper's set."),
        "cost_inputs": cost_inputs_status(arm_rows),
        "arm_coverage": arm_coverage_status(arm_rows),
        "feature_names": feat_names,
        "params": {"w": w, "s": s, "beta": beta, "cost_scale": cost_scale,
                   "cost_mode": cost_mode, "outer_folds": outer_folds,
                   "rf": "200 trees / depth 6 / min_samples_leaf 2 (section 9.5)"},
        "frame_degeneracy": {
            "n_continue_correct": int(Y[:, 0].sum()),
            "continue_column_constant": bool(Y[:, 0].sum() in (0, len(qids))),
            "note": ("The D-line arm sweep covers only the C->W rows, so the "
                     "continue branch is wrong on every row and p_fail is "
                     "constant on this frame.  The paper's scalar-vs-"
                     "action-conditioned contrast (Table tab:fork) is then NOT "
                     "identified: 'scalar routing' degenerates to a constant "
                     "policy and its regret measures only which arm is fired, "
                     "not whether the scalar recovers sign(tau_adv).  Read the "
                     "regret rows as arm-choice diagnostics until the harm arms "
                     "supply C->C rows."),
        },
        "scalar_thresholds_per_fold": thresholds,
        "scalar_split_alignment": ("refit-per-outer-fold" if scalar_fit_fn is not None
                                   else "reused-oof-from-a-different-split"),
        "n_folds_skipped_scalar_unavailable": int(n_folds_skipped),
        "policies": out_pol,
        "mde_pp": list(C.MDE_PP),
        "scope": ("regret differences below the MDE band "
                  f"({C.MDE_PP[0]}-{C.MDE_PP[1]} pp at n=93) are 'indistinguishable'; "
                  "this table never ranks arms."),
    }


def _select_threshold_inner(g_tr: np.ndarray, U_tr: np.ndarray, j_star: int,
                            groups_tr: np.ndarray, seed: int,
                            inner_folds: int = 3, j_low: int = 0) -> float:
    """Pick the failure-trigger threshold on INNER folds only (digest 4.0 item 6).

    ``j_star`` and ``j_low`` are ACTION indices into ``U_tr`` (the fire and the
    do-not-fire branch respectively), not arm offsets.
    """
    grid = np.unique(np.quantile(g_tr, np.linspace(0.0, 1.0, 21)))
    best, best_u = float(grid[-1]), -np.inf
    for theta in grid:
        vals = []
        for inner in C.grouped_folds(groups_tr, inner_folds, seed + 7):
            itr, ite = ~inner, inner
            if ite.sum() == 0 or itr.sum() == 0:
                continue
            pick = np.where(g_tr[ite] > theta, j_star, j_low)
            vals.append(U_tr[ite][np.arange(int(ite.sum())), pick].mean())
        if vals and float(np.mean(vals)) > best_u:
            best_u, best = float(np.mean(vals)), float(theta)
    return best


def abstraction_gap_label(U: np.ndarray, g: np.ndarray, groups: np.ndarray,
                          *, n_bins: int = 4, folds: int = 5,
                          seed: int = 20260905,
                          scalar_fit_fn: Optional[Callable[[set], Dict[str, float]]] = None,
                          qids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """``Gap(g) = V* - V_g`` (2606.21399 section 2.3).

    ``V* = E[max_a EU_a(h)]`` chooses before averaging; ``V_g = E[max_a
    E[EU_a(h) | g(h)]]`` averages every state sharing a scalar value and only
    then chooses.  Plug-in estimator on session-grouped held-out folds: bin
    ``g`` by TRAINING quantiles, take each bin's per-action training mean, apply
    the resulting per-bin action to the held-out rows.  ``Gap >= 0`` holds by
    max-of-means <= mean-of-maxes.

    ``U`` is the realised (n_rows, n_actions) utility matrix -- LABEL side.

    ``g`` is only a BINNING variable here, but it is still a fitted object: pass
    ``scalar_fit_fn`` (with ``qids`` naming the rows of ``U`` in order) and the
    scalar is refitted inside every fold with that fold's sessions held out, so
    no bin edge and no per-bin action was chosen with the help of a held-out
    row's own label.  ``scalar_split_alignment`` in the report says which of the
    two was used; folds whose refit cannot produce a finite scalar for every row
    are SKIPPED and counted, never silently filled.
    """
    U = np.asarray(U, dtype=float)
    g = np.asarray(g, dtype=float).ravel()
    groups = np.asarray(groups)
    fold_honest = scalar_fit_fn is not None and qids is not None
    if fold_honest and len(qids) != U.shape[0]:
        raise ValueError("qids must name the rows of U in order "
                         f"({len(qids)} qids vs {U.shape[0]} rows)")
    v_star_terms: List[float] = []
    v_g_terms: List[float] = []
    n_skipped = 0
    for te in C.grouped_folds(groups, folds, seed):
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        g_fold = g
        if fold_honest:
            sc = scalar_fit_fn(set(groups[te].tolist()))
            g_fold = np.array([float(sc.get(q, np.nan)) for q in qids], dtype=float)
            if not np.isfinite(g_fold).all():
                n_skipped += 1
                continue
        edges = np.unique(np.quantile(g_fold[tr], np.linspace(0, 1, n_bins + 1)[1:-1]))
        b_tr = np.digitize(g_fold[tr], edges)
        b_te = np.digitize(g_fold[te], edges)
        U_tr, U_te = U[tr], U[te]
        for i, b in enumerate(b_te):
            sel = b_tr == b
            a = int(U_tr[sel].mean(axis=0).argmax()) if sel.any() else int(U_tr.mean(axis=0).argmax())
            v_g_terms.append(float(U_te[i, a]))
            v_star_terms.append(float(U_te[i].max()))
    v_star = float(np.mean(v_star_terms)) if v_star_terms else None
    v_g = float(np.mean(v_g_terms)) if v_g_terms else None
    gap = (v_star - v_g) if (v_star is not None and v_g is not None) else None
    return {"paper": PAPER_CONTROL, "V_star": v_star, "V_g": v_g,
            "gap": (max(0.0, gap) if gap is not None else None),
            "gap_raw": gap, "n_bins": n_bins, "n_scored": len(v_g_terms),
            "scalar_split_alignment": ("refit-per-outer-fold" if fold_honest
                                       else "reused-oof-from-a-different-split"),
            "n_folds_skipped_scalar_unavailable": int(n_skipped)}


def exploitability_label(features: Dict[str, Dict[str, float]],
                         scalar: Dict[str, float],
                         oracle_action: Dict[str, int],
                         *, folds: int = 5, seed: int = 20260905,
                         scalar_fit_fn: Optional[Callable[[set], Dict[str, float]]] = None
                         ) -> Dict[str, Any]:
    """Exploitability-beyond-scalar diagnostic (2606.21399 section 5.4).

    "held-out improvement in oracle-action prediction obtained by adding prefix
    features to a scalar-only predictor".  Both predictors are the same forest
    family; folds are session-grouped.  The paper reports r = 0.716 between this
    diagnostic and deployable gain across 84 regimes -- that correlation is
    THEIRS, we only compute our own increment.

    ``scalar_fit_fn(held_out_sessions) -> {qid: p_fail}`` refits the scalar-only
    predictor's single column inside EVERY fold with that fold's sessions held
    out.  Without it the ``scalar`` dict is reused as given, and a scalar that
    was cross-validated on a DIFFERENT session split carries the test fold's
    labels into the baseline -- which here inflates ``acc_scalar_only`` and so
    DEFLATES the increment (conservative, but not fold-honest).  Both arms are
    scored inside one fold loop on identical folds, so the increment is always a
    paired difference; folds whose refit cannot score every row are skipped and
    counted.
    """
    fold_honest = scalar_fit_fn is not None
    want = [q for q in sorted(oracle_action)
            if q in features and (fold_honest or q in scalar)]
    qids, X, feat_names, _dropped = feature_matrix(features, want)
    if len(qids) < 5:
        return {"paper": PAPER_CONTROL, "n_rows": len(qids), "increment": None,
                "acc_scalar_only": None, "acc_scalar_plus_prefix": None,
                "note": "too few complete rows"}
    g = np.array([[float(scalar[q]) if q in scalar else np.nan] for q in qids])
    y = np.array([int(oracle_action[q]) for q in qids])
    groups = np.array([C.session_of(q) for q in qids])
    if len(np.unique(y)) < 2:
        return {"paper": PAPER_CONTROL, "n_rows": len(qids), "increment": None,
                "acc_scalar_only": None, "acc_scalar_plus_prefix": None,
                "note": "oracle action is constant on this frame"}

    hits_s = hits_f = n_scored = n_skipped = 0
    for te in C.grouped_folds(groups, folds, seed):
        tr = ~te
        if te.sum() == 0 or tr.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        g_fold = g
        if fold_honest:
            sc = scalar_fit_fn(set(groups[te].tolist()))
            g_fold = np.array([[float(sc.get(q, np.nan))] for q in qids])
        if not np.isfinite(g_fold).all():
            n_skipped += 1
            continue
        Z_full = np.hstack([g_fold, X])
        pred_s = _rf(seed).fit(g_fold[tr], y[tr]).predict(g_fold[te])
        pred_f = _rf(seed).fit(Z_full[tr], y[tr]).predict(Z_full[te])
        hits_s += int((pred_s == y[te]).sum())
        hits_f += int((pred_f == y[te]).sum())
        n_scored += int(te.sum())
    if n_scored == 0:
        return {"paper": PAPER_CONTROL, "n_rows": len(qids), "increment": None,
                "acc_scalar_only": None, "acc_scalar_plus_prefix": None,
                "n_folds_skipped_scalar_unavailable": int(n_skipped),
                "note": "no fold could be scored (scalar unavailable or a "
                        "single-class training fold)"}
    acc_scalar = hits_s / n_scored
    acc_full = hits_f / n_scored
    return {"paper": PAPER_CONTROL, "n_rows": len(qids),
            "n_classes": int(len(np.unique(y))),
            "n_scored": int(n_scored),
            "n_folds_skipped_scalar_unavailable": int(n_skipped),
            "scalar_split_alignment": ("refit-per-outer-fold" if fold_honest
                                       else "reused-oof-from-a-different-split"),
            "acc_scalar_only": acc_scalar, "acc_scalar_plus_prefix": acc_full,
            "increment": float(acc_full - acc_scalar),
            "feature_names": feat_names}


# ===========================================================================
# (B) 2608.10441 -- random@matched-rate, placebo, positive control, floor
# ===========================================================================

def random_matched_rate_masks(clusters: Sequence[Any], n_fires: int, *,
                              reps: int = 2000,
                              seed: int = 20260905) -> np.ndarray:
    """B session-clustered random fire masks with EXACTLY ``n_fires`` fires.

    2608.10441 section 2 / section 6: every acquisition policy is compared to a
    *random acquisition at the same budget*.  The paper's null is i.i.d. over
    examples; our rows are clustered in sessions, so the scheme is (documented
    exactly, per the digest):

    1. shuffle the session ids;
    2. walk the shuffled sessions accumulating ALL of their rows into an
       eligible pool until the pool holds at least ``n_fires`` rows;
    3. draw exactly ``n_fires`` rows uniformly without replacement from that
       pool.

    So the null fires ROWS (not whole sessions) but the rows it may fire are a
    union of whole sessions -- a candidate whose fires clump inside a few
    sessions is compared against a null with the same clumping opportunity.
    Returns a boolean array of shape (reps, n_rows); every row sums to
    ``n_fires``.
    """
    clusters = np.asarray(clusters)
    n = len(clusters)
    n_fires = int(n_fires)
    if not 0 <= n_fires <= n:
        raise ValueError(f"n_fires={n_fires} outside [0, {n}]")
    uniq = np.unique(clusters)
    members = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    out = np.zeros((reps, n), dtype=bool)
    for b in range(reps):
        order = rng.permutation(uniq)
        pool: List[int] = []
        for c in order:
            pool.extend(members[c].tolist())
            if len(pool) >= n_fires:
                break
        pick = rng.choice(np.asarray(pool, dtype=int), size=n_fires, replace=False)
        out[b, pick] = True
    return out


def mask_metrics(mask: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """coverage(/n_pos), precision(/fires), false-reset(/n_neg) for a fire mask."""
    m = np.asarray(mask, dtype=bool)
    yy = np.asarray(y, dtype=int)
    fires = int(m.sum())
    cov = int((m & (yy == 1)).sum())
    fr = int((m & (yy == 0)).sum())
    n_pos, n_neg = int((yy == 1).sum()), int((yy == 0).sum())
    return {"fires": fires, "coverage": cov, "n_pos": n_pos,
            "coverage_rate": (cov / n_pos) if n_pos else None,
            "precision": (cov / fires) if fires else None,
            "false_resets": fr, "n_neg": n_neg,
            "false_reset_rate": (fr / n_neg) if n_neg else None}


def random_matched_rate_table(y: Sequence[int], clusters: Sequence[Any],
                              n_fires: int, *, reps: int = 2000,
                              seed: int = 20260905,
                              alpha: float = 0.05) -> Dict[str, Any]:
    """The mandatory ``random@matched-rate`` row (2608.10441 guard 1).

    Reports percentile bands of coverage / precision / false-reset over ``reps``
    cluster-respecting random fire masks at the candidate's own measured fire
    count.  Put this row beside ``parse-failure-only`` in the three-metric table.
    """
    y = np.asarray(y, dtype=int)
    masks = random_matched_rate_masks(clusters, n_fires, reps=reps, seed=seed)
    cov = np.array([(m & (y == 1)).sum() for m in masks], dtype=float)
    fr = np.array([(m & (y == 0)).sum() for m in masks], dtype=float)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    prec = cov / max(1, n_fires)
    q = [100 * alpha / 2, 50.0, 100 * (1 - alpha / 2)]

    def band(v: np.ndarray, denom: int) -> Dict[str, Any]:
        lo, mid, hi = np.percentile(v, q)
        return {"lo": float(lo), "median": float(mid), "hi": float(hi),
                "mean": float(v.mean()), "denominator": denom}

    return {
        "paper": PAPER_SNR, "row": "random@matched-rate",
        "n_fires": int(n_fires), "reps": int(reps),
        "n_rows": int(len(y)), "n_pos": n_pos, "n_neg": n_neg,
        "n_clusters": int(len(np.unique(np.asarray(clusters)))),
        "coverage": band(cov / max(1, n_pos), n_pos),
        "precision": band(prec, int(n_fires)),
        "false_reset": band(fr / max(1, n_neg), n_neg),
        "scheme": random_matched_rate_masks.__doc__.strip().splitlines()[0],
    }


def baseline_vs_random_band(candidate: Dict[str, Any],
                            random_table: Dict[str, Any]) -> Dict[str, Any]:
    """Does a candidate sit INSIDE its own matched-rate random band?

    2608.10441 guard 1: ``random@matched-rate`` is a mandatory comparator ROW,
    which only means something if the comparison is actually made.  This puts the
    verdict in the JSON instead of leaving it to the reader: a candidate whose
    coverage and precision both fall inside the band of random firing at its own
    measured rate is not distinguishable from random at that rate.

    The band is a comparator, NOT a significance test: "inside the band" is not
    a p-value and "outside the band" is not a win (digest 4.0 keeps the winner
    rule).  The comparison is void unless the fire counts match, which
    ``rate_matched`` records.
    """
    pairs = (("coverage", "coverage_rate"), ("precision", "precision"),
             ("false_reset", "false_reset_rate"))
    out: Dict[str, Any] = {}
    for metric, key in pairs:
        band = random_table.get(metric) or {}
        v = candidate.get(key)
        lo, hi, med = band.get("lo"), band.get("hi"), band.get("median")
        out[metric] = {
            "candidate": v,
            "random_lo": lo, "random_median": med, "random_hi": hi,
            "inside_random_band": (None if (v is None or lo is None or hi is None)
                                   else bool(lo <= v <= hi)),
            "above_random_median": (None if (v is None or med is None)
                                    else bool(v > med)),
        }
    inside = [out[m]["inside_random_band"] for m, _k in pairs[:2]]
    matched = (candidate.get("fires") == random_table.get("n_fires"))
    return {
        "rate_matched": bool(matched),
        "metrics": out,
        "verdict": (
            "COMPARISON VOID: the candidate fire count "
            f"({candidate.get('fires')}) is not the band budget "
            f"({random_table.get('n_fires')}), so the rows are not matched."
            if not matched else
            "Coverage AND precision both sit INSIDE the matched-rate random "
            "band: at this fire rate the candidate is not distinguishable from "
            "random firing, so 'beats this baseline' cannot mean 'beats random "
            "at the same rate'."
            if inside == [True, True] else
            "At least one of coverage / precision falls outside the "
            "matched-rate random band; the band is a comparator, not a "
            "significance test, and the winner rule still applies."),
    }


# ---------------------------------------------------------------------------
# placebo best-k scan
# ---------------------------------------------------------------------------

def pooled_nonwitness_flip_rate(flips: Dict[str, Dict[int, bool]],
                                witness: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-trial flip rate over NON-witness k, exactly as
    ``agent/d_ksweep_analysis.py`` computes it (``p_flip``, the
    ``wrong_block_distribution`` rate that replaced the cancelled sham arms).

    The source loop, reproduced: for every qid with ``k_witness`` not None,
    every k other than ``k_witness`` contributes one trial and ``correct[k]``
    contributes to the numerator.
    """
    trials = correct = 0
    for qid, by_k in flips.items():
        k_w = (witness.get(qid) or {}).get("k_witness")
        if k_w is None:
            continue
        for k, c in by_k.items():
            if k == k_w:
                continue
            trials += 1
            correct += int(bool(c))
    return {"p_nonwitness_flip": (correct / trials) if trials else 0.0,
            "nonwitness_trials": trials, "nonwitness_correct": correct}


def expected_random_bestk(flips: Dict[str, Dict[int, bool]],
                          witness: Dict[str, Dict[str, Any]],
                          qids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Random best-k envelope ``E[max] = 1 - (1-p)^n_docs``.

    Reused verbatim from ``agent/d_ksweep_analysis.py`` (``best_k_envelope``):
    ``p`` is :func:`pooled_nonwitness_flip_rate` and ``n_docs`` comes from the
    frozen witness table (``max(1, n_docs)``).  best-k is an upper envelope,
    never a point estimate; it must beat this envelope to mean anything.
    """
    qids = list(qids if qids is not None else sorted(flips))
    p = pooled_nonwitness_flip_rate(flips, witness)["p_nonwitness_flip"]
    total = 0.0
    for qid in qids:
        n_docs = max(1, int((witness.get(qid) or {}).get("n_docs",
                                                         len(flips.get(qid, {})) or 1)))
        total += 1.0 - (1.0 - p) ** n_docs
    n = len(qids)
    return {"expected_random_sum": float(total),
            "expected_random_rate": float(total / n) if n else None,
            "p_nonwitness_flip": float(p), "n": n}


def observed_bestk(flips: Dict[str, Dict[int, bool]],
                   qids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """best-k = "any k flips this row" over the frozen (qid, k) flip table."""
    qids = list(qids if qids is not None else sorted(flips))
    hits = sum(1 for q in qids if any(flips.get(q, {}).values()))
    return {"hits": hits, "n": len(qids),
            "rate": (hits / len(qids)) if qids else None}


def placebo_bestk_scan(flips: Dict[str, Dict[int, bool]],
                       witness: Dict[str, Dict[str, Any]],
                       *, level: str = "per_qid", reps: int = 2000,
                       seed: int = 20260905,
                       qids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Matched-moment noise placebo on the best-k scan (2608.10441 guard 2).

    The paper replaces the realised ``Delta_i`` with i.i.d. noise matched to its
    moments and reruns the SAME in-sample top-b oracle; if the placebo
    reproduces the oracle's apparent gain, the headroom is order statistics of
    noise (MIND: +0.0518 = +0.0518, >= 100 %).

    Our flip vector is binary, so a matched-moment draw is Bernoulli
    (see DEVIATIONS):

    * ``level='per_qid'`` -- each entry of qid q ~ Bernoulli(p_q) with p_q the
      qid's own NON-WITNESS flip rate (k != k_witness, the same denominator as
      the pooled p_bar): matches that qid's mean AND its variance p_q(1-p_q).
      A qid with no ``k_witness`` in the frozen witness table, or with only the
      witness k logged, falls back to all of its k and is counted in
      ``n_qids_rate_over_all_k``;
    * ``level='pooled'`` -- every entry ~ Bernoulli(p_bar) with p_bar the pooled
      non-witness rate: matches the global moments and must reproduce the
      analytic envelope of :func:`expected_random_bestk` (a self-check on the
      code path).

    Reported: the fraction of the observed ``best-k - random envelope`` gap the
    placebo reproduces.
    """
    if level not in ("per_qid", "pooled"):
        raise ValueError("level must be 'per_qid' or 'pooled'")
    qids = list(qids if qids is not None else sorted(flips))
    rng = np.random.default_rng(seed)
    env = expected_random_bestk(flips, witness, qids)
    obs = observed_bestk(flips, qids)
    p_bar = env["p_nonwitness_flip"]

    sizes: List[int] = []
    probs: List[float] = []
    n_all_k = 0                     # qids whose p_q had to fall back to all k
    for q in qids:
        by_k = flips.get(q, {})
        w_ent = witness.get(q) or {}
        n_k = max(1, int(w_ent.get("n_docs", len(by_k) or 1)))
        sizes.append(n_k)
        if level == "pooled" or not by_k:
            probs.append(p_bar)
            continue
        k_w = w_ent.get("k_witness")
        vals = ([int(bool(v)) for k, v in by_k.items() if k != k_w]
                if k_w is not None else [])
        if not vals:                # no k_witness, or only the witness k logged
            vals = [int(bool(v)) for v in by_k.values()]
            n_all_k += 1
        probs.append(float(np.mean(vals)))

    rates = np.empty(reps, dtype=float)
    ent_mean = np.zeros(len(qids))
    ent_var = np.zeros(len(qids))
    for b in range(reps):
        hits = 0
        for i, (n_k, p) in enumerate(zip(sizes, probs)):
            draw = rng.random(n_k) < p
            hits += int(draw.any())
            ent_mean[i] += draw.mean()
            ent_var[i] += draw.var()
        rates[b] = hits / len(qids)
    ent_mean /= reps
    ent_var /= reps

    obs_rate = obs["rate"] or 0.0
    env_rate = env["expected_random_rate"] or 0.0
    gap = obs_rate - env_rate
    plac = float(rates.mean())
    return {
        "paper": PAPER_SNR, "level": level, "reps": int(reps),
        "n_qids": len(qids),
        "n_qids_rate_over_all_k": int(n_all_k),
        "rate_denominator": ("k != k_witness on both levels; the "
                             "n_qids_rate_over_all_k qids above had no usable "
                             "k_witness and fell back to all of their k"),
        "observed_bestk_rate": obs_rate, "observed_bestk_hits": obs["hits"],
        "random_envelope_rate": env_rate,
        "p_nonwitness_flip": p_bar,
        "placebo_bestk_rate_mean": plac,
        "placebo_bestk_rate_ci95": [float(np.percentile(rates, 2.5)),
                                    float(np.percentile(rates, 97.5))],
        "observed_minus_envelope_pp": float(100.0 * gap),
        "placebo_minus_envelope_pp": float(100.0 * (plac - env_rate)),
        "gap_reproduced_fraction": (float((plac - env_rate) / gap)
                                    if abs(gap) > 1e-12 else None),
        "guard_status": (
            "level='pooled' is the PAPER-FAITHFUL matched-moment draw (i.i.d. "
            "noise at the global mean; for a binary vector the mean fixes the "
            "variance).  At that level P(any k flips) is exactly the analytic "
            "envelope 1-(1-p)^n_docs, so the pooled placebo can only ever "
            "reproduce the envelope and gap_reproduced_fraction is ~0 by "
            "construction: it is a self-check on this code path, NOT an "
            "independent guard.  The informative level is 'per_qid', but its "
            "p_q are the qid's OWN realised flip rate, so it is an in-sample "
            "over-dispersion diagnostic and must not be quoted as the paper's "
            "'placebo reproduces >=100% of the oracle gain' result."
            if level == "pooled" else
            "level='per_qid' draws Bernoulli(p_q) with p_q the qid's OWN "
            "realised NON-witness flip rate (k != k_witness, the same "
            "denominator as the pooled p_bar), so it is fitted in sample: it "
            "measures how "
            "much of the observed best-k minus envelope gap is per-qid "
            "over-dispersion rather than content-specific localisation.  It is "
            "NOT the paper's global matched-moment placebo (that is "
            "level='pooled', which for a binary vector collapses onto the "
            "analytic envelope) and must not be quoted as their >=100% result."),
        "matched_moments": {"target_mean": [float(p) for p in probs[:5]],
                            "empirical_mean_head": [float(v) for v in ent_mean[:5]],
                            "empirical_var_head": [float(v) for v in ent_var[:5]],
                            "note": "binary vector: the mean fixes the variance p(1-p)"},
    }


def flip_table_status(flips: Dict[str, Dict[int, bool]],
                      witness: Dict[str, Dict[str, Any]],
                      *, source: Optional[str] = None) -> Dict[str, Any]:
    """What the loaded (qid, k) flip table actually contains -- guard 2's input.

    The frozen 823-trial table lives on the NPU server at ``~/bench_results/d_v2/``
    and is NOT on this box; the placebo is only as real as the file it was given,
    so its shape is reported beside every placebo number instead of being assumed.
    ``k_witness`` coverage is load-bearing: it is the denominator of BOTH the
    pooled ``p_bar`` and the per-qid ``p_q``.
    """
    n_trials = int(sum(len(v) for v in flips.values()))
    with_w = [q for q in flips if (witness.get(q) or {}).get("k_witness") is not None]
    with_n = [q for q in flips if (witness.get(q) or {}).get("n_docs") is not None]
    return {
        "source": source,
        "n_qids": len(flips),
        "n_trials": n_trials,
        "n_qids_with_k_witness": len(with_w),
        "n_qids_with_n_docs": len(with_n),
        "witness_table_covers_all_qids": bool(len(with_w) == len(flips) and flips),
        "note": ("The digest's frozen k-sweep is 823 (qid, k) trials "
                 "(~/bench_results/d_v2/ on the NPU server).  This run loaded "
                 f"{n_trials} trials over {len(flips)} qids; if that is not the "
                 "823-trial table, the placebo is a smoke test of the code path "
                 "and NOT the guard-2 result."),
    }


# ---------------------------------------------------------------------------
# positive control
# ---------------------------------------------------------------------------

def synthetic_signal(y: Sequence[int], groups: Sequence[Any], s: float,
                     *, session_share: float = 0.5,
                     seed: int = 20260905) -> np.ndarray:
    """A label-correlated synthetic feature at dialled strength ``s in [0, 1]``.

    2608.10441 App. "Positive Control": hold the folds / base / features fixed
    and replace the lift with a synthetic signal at a controllable strength.
    Ours mixes a row-level and a session-level component so the injected column
    has the same clustering structure as a real per-session statistic::

        core = (1 - session_share) * z(y_row) + session_share * z(session mean of y)
        signal = s * z(core) + sqrt(1 - s^2) * noise

    ``s = 0`` is pure noise; ``s = 1`` is the (clustered) label itself.  This
    column is built FROM the label and must never be written to a feature frame
    -- it exists only inside the positive-control pipeline.
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    s = float(np.clip(s, 0.0, 1.0))

    def _z(v: np.ndarray) -> np.ndarray:
        sd = v.std()
        return (v - v.mean()) / (sd if sd > 0 else 1.0)

    sess_mean = np.array([y[groups == g].mean() for g in groups])
    core = _z((1.0 - session_share) * _z(y) + session_share * _z(sess_mean))
    return s * core + math.sqrt(max(0.0, 1.0 - s * s)) * rng.standard_normal(len(y))


def _auprc_with_lb(scores: np.ndarray, y: np.ndarray, clusters: np.ndarray,
                   reps: int) -> Tuple[Optional[float], Optional[float], float]:
    ap = C.average_precision(scores, y)
    lo, _hi, _n = C.clustered_bootstrap(C.average_precision, scores, y, clusters,
                                        reps=reps)
    return ap, lo, C.prevalence(y)


def positive_control_curve(y: Sequence[int], groups: Sequence[Any],
                           *, s_grid: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4,
                                                         0.5, 0.6, 0.8),
                           X_base: Optional[np.ndarray] = None,
                           reps: int = 500, seed: int = 20260905,
                           c_grid: Sequence[float] = (1e-2, 1e-1, 1.0),
                           outer_folds: int = 5, inner_folds: int = 3
                           ) -> Dict[str, Any]:
    """Run the SAME nested-CV + metric pipeline on a synthetic signal at each ``s``.

    2608.10441 guard 3.  For every strength the injected column goes through
    :func:`t34_common.nested_cv_logistic` (session-grouped, knobs picked on
    inner folds) and is scored with the frame's own prevalence as chance AP --
    never 0.1033 (digest 4.0 / HARD RULE 3).

    The pre-registered bar implemented here is the computable subset of the
    digest 4.0 winner rule: **AUPRC above the frame prevalence AND the
    session-clustered bootstrap lower bound also above it**.  ``s_star`` is the
    smallest grid strength that clears it.
    """
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    prev = C.prevalence(y)
    points: List[Dict[str, Any]] = []
    for i, s in enumerate(s_grid):
        z = synthetic_signal(y, groups, s, seed=seed + 101 * i).reshape(-1, 1)
        X = z if X_base is None else np.hstack([np.asarray(X_base, dtype=float), z])
        res = C.nested_cv_logistic(X, y, groups, c_grid=c_grid,
                                   outer_folds=outer_folds, inner_folds=inner_folds,
                                   seed=seed, select_by="ap")
        ok = res["scored_mask"]
        # clusters must come from the RETURNED groups: nested_cv_logistic drops
        # rows with non-finite features, so the caller's own group vector can be
        # longer than the scored frame whenever X_base carries missing values.
        clusters = C.session_clusters([str(g) for g in res["groups"]])
        ap, lo, _ = _auprc_with_lb(res["oof_scores"][ok], res["labels"][ok],
                                   clusters[ok], reps)
        points.append({"s": float(s), "auprc": ap, "auprc_lb95": lo,
                       "auroc": C.auroc(res["oof_scores"][ok], res["labels"][ok]),
                       "n_scored": res["n_scored"], "n_pos": res["n_pos_scored"],
                       "clears_bar": bool(ap is not None and lo is not None
                                          and ap > prev and lo > prev)})
    s_star = next((p["s"] for p in points if p["clears_bar"]), None)
    return {"paper": PAPER_SNR, "prevalence_chance_ap": prev,
            "n_rows": int(len(y)), "n_pos": int(y.sum()),
            "n_clusters": int(len(np.unique(groups))),
            "bar": "AUPRC > prevalence AND clustered bootstrap LB > prevalence",
            "curve": points, "s_star": s_star,
            "with_real_features": X_base is not None}


def locate_real_on_curve(curve: Dict[str, Any],
                         real_auprc: Optional[float]) -> Optional[float]:
    """Where a real feature's AUPRC sits on the positive-control curve (``s'``).

    Linear interpolation between the two bracketing grid points; None when the
    real AUPRC is off the curve's range (report it as ``< s_min`` / ``> s_max``
    rather than extrapolating).
    """
    if real_auprc is None:
        return None
    pts = [(p["s"], p["auprc"]) for p in curve["curve"] if p["auprc"] is not None]
    pts.sort()
    for (s0, a0), (s1, a1) in zip(pts, pts[1:]):
        lo, hi = min(a0, a1), max(a0, a1)
        if lo <= real_auprc <= hi and a1 != a0:
            return float(s0 + (s1 - s0) * (real_auprc - a0) / (a1 - a0))
    return None


def granularity_ladder(y: Sequence[int], groups: Sequence[Any],
                       X: np.ndarray, n_fires: int,
                       *, ks: Sequence[int] = (4, 8, 16, 32, 64),
                       regimes: Optional[Sequence[Any]] = None,
                       folds: int = 5, seed: int = 20260905,
                       reps: int = 500) -> Dict[str, Any]:
    """Optional diagnostic: per-instance -> KMeans K=4..64 -> declared regimes.

    2608.10441 section 6 ("Cluster / regime / tree").  Each granularity gets a
    label-free out-of-fold assignment, an empirical-Bayes shrunk per-group mean
    of the label on the training rows, top-``n_fires`` selection on the held-out
    rows, and is compared to :func:`random_matched_rate_table` at the same
    budget.  A flat, near-random ladder is the evidence that no granularity
    rescues the policy.
    """
    from sklearn.cluster import KMeans

    y = np.asarray(y, dtype=int)
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    clusters = C.session_clusters([str(g) for g in groups])
    rand = random_matched_rate_table(y, clusters, n_fires, reps=reps, seed=seed)

    def _oof_scores(assign_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        out = np.full(len(y), np.nan)
        prior = y.mean()
        for te in C.grouped_folds(groups, folds, seed):
            tr = ~te
            if te.sum() == 0 or tr.sum() == 0:
                continue
            a_tr, a_te = assign_fn(tr, te)
            for gid in np.unique(a_te):
                sel = a_tr == gid
                n_g = int(sel.sum())
                mean_g = float(y[tr][sel].mean()) if n_g else prior
                # empirical-Bayes shrinkage toward the training prior
                shrunk = (n_g * mean_g + 1.0 * prior) / (n_g + 1.0)
                idx = np.where(te)[0][a_te == gid]
                out[idx] = shrunk
        return out

    rows: List[Dict[str, Any]] = []

    # per-instance: an out-of-fold logistic probe on the same features
    res = C.nested_cv_logistic(X, y, groups, outer_folds=folds, inner_folds=3,
                               seed=seed, select_by="ap")
    ok = res["scored_mask"]
    mask = np.zeros(len(y), dtype=bool)
    order = np.argsort(-np.where(ok, res["oof_scores"], -np.inf), kind="mergesort")
    mask[order[:n_fires]] = True
    rows.append({"granularity": "per_instance", **mask_metrics(mask, y)})

    for k in ks:
        if k >= len(y):
            continue

        def _assign(tr: np.ndarray, te: np.ndarray, k: int = k) -> Tuple[np.ndarray, np.ndarray]:
            km = KMeans(n_clusters=k, n_init=4, random_state=seed).fit(X[tr])
            return km.labels_, km.predict(X[te])

        sc = _oof_scores(_assign)
        m = np.zeros(len(y), dtype=bool)
        m[np.argsort(-np.nan_to_num(sc, nan=-np.inf), kind="mergesort")[:n_fires]] = True
        rows.append({"granularity": f"kmeans_k{k}", **mask_metrics(m, y)})

    if regimes is not None:
        reg = np.asarray(regimes)
        codes = {r: i for i, r in enumerate(sorted(set(reg.tolist())))}
        code = np.array([codes[r] for r in reg])

        def _assign_reg(tr: np.ndarray, te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return code[tr], code[te]

        sc = _oof_scores(_assign_reg)
        m = np.zeros(len(y), dtype=bool)
        m[np.argsort(-np.nan_to_num(sc, nan=-np.inf), kind="mergesort")[:n_fires]] = True
        rows.append({"granularity": "declared_regimes", **mask_metrics(m, y)})

    return {"paper": PAPER_SNR, "n_fires": int(n_fires),
            "random_at_matched_rate": rand, "ladder": rows}


def reward_snr_floor(deltas: Sequence[float], *, delta_definition: str,
                     alpha: float = 0.05, power: float = 0.8) -> Dict[str, Any]:
    """``rho = mu/sigma`` against ``rho*(N) = (z_{1-a/2}+z_{1-b})/sqrt(N)``
    (2608.10441 Prop. 1, section 8).

    The paper states three times that this is a NECESSARY mean-detectability
    condition -- not a sufficient policy-learning bound, not an impossibility
    theorem.  Our label is binary, so ``Delta`` must be redefined before the
    formula means anything: ``delta_definition`` is mandatory and non-empty, and
    the returned ``caveat`` travels with every printed number.  ``rho*(N)`` is an
    MDE in different clothing and must never be used as a post-hoc significance
    test.
    """
    if not (delta_definition or "").strip():
        raise ValueError(
            "delta_definition is mandatory: the floor is a mean-detection bound "
            "on a CONTINUOUS Delta; state how Delta is defined for the binary "
            "C->W setting before computing rho (2608.10441 section 8 / card pitfall 1)."
        )
    from scipy.stats import norm
    d = np.asarray([x for x in deltas if x is not None], dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    if n < 2:
        raise ValueError(
            f"reward_snr_floor needs at least two finite Delta values, got {n}. "
            "An empty Delta vector would return rho=None / above_floor=False, "
            "which reads like a real 'below the floor' verdict.")
    mu = float(d.mean()) if n else float("nan")
    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    rho = (mu / sd) if (n > 1 and sd > 0) else None
    z = float(norm.ppf(1 - alpha / 2) + norm.ppf(power))
    rho_star = z / math.sqrt(n) if n else None
    n_min = (z / abs(rho)) ** 2 if rho else None
    return {
        "paper": PAPER_SNR, "delta_definition": delta_definition,
        "N": n, "mean_delta": mu, "sd_delta": sd, "rho": rho,
        "z_sum": z, "rho_star": rho_star, "N_min": n_min,
        "above_floor": (bool(rho is not None and rho_star is not None
                             and abs(rho) >= rho_star)),
        "caveat": ("rho*(N) = 2.8/sqrt(N) is a NECESSARY mean-detectability "
                   "condition on a continuous Delta, not a sufficient "
                   "policy-learning or regret bound and not an impossibility "
                   "theorem; it is an MDE in disguise and must never be used as "
                   "a post-hoc significance test.  Our C->W label is binary "
                   "(prevalence 0.1033 on the 900-row frame), so the Delta "
                   "definition above is load-bearing."),
        "mde_pp": list(C.MDE_PP),
    }


def reward_deltas_label(arm_rows: Sequence[Dict[str, Any]],
                        continue_correct: Dict[str, bool],
                        arm: str) -> Tuple[List[str], List[float]]:
    """``Delta_i = 1[arm correct] - 1[continue correct]`` per row (LABEL side).

    Raises when ``arm`` names nothing in ``arm_rows``: a typo would otherwise
    hand :func:`reward_snr_floor` an empty vector and produce a report with
    ``N = 0`` and ``rho = None`` that reads like a legitimate 'below the floor'
    result.  The D-line ``d_arm`` values are bare (``corr``, ``corr_re``,
    ``sham``), NOT the ``d_``-prefixed file stems.
    """
    present = sorted({r["arm"] for r in arm_rows})
    if arm not in present:
        raise ValueError(
            f"arm {arm!r} is not in the loaded arm rows; available arms: {present}. "
            "The D-line files carry the bare d_arm value, not the file stem.")
    qids, out = [], []
    for r in sorted(arm_rows, key=lambda x: (x["qid"], x["arm"])):
        if r["arm"] != arm or r["qid"] not in continue_correct:
            continue
        qids.append(r["qid"])
        out.append(float(int(r["correct"]) - int(continue_correct[r["qid"]])))
    if not out:
        raise ValueError(
            f"arm {arm!r} has no rows that join the continue frame; nothing to "
            "compute rho on.")
    return qids, out


# ===========================================================================
# free prefix features (compressed arm's own row; no label, no full arm)
# ===========================================================================

#: S8/S9/S10 free prefix scalars, all from the compressed arm's own battery row.
#: ``decision_step`` is deliberately absent -- session order is ``step_index(qid)``
#: (HARD RULE 5); ``full_prefill_sec`` is absent because the leakage guard
#: refuses ``full_``-prefixed names.
PREFIX_FEATURE_COLUMNS = (
    "gist_tokens", "actual_compression_ratio", "doc_chunks", "doc_tokens",
    "kept_history_tokens", "compressed_history_tokens", "hybrid_top_k",
    "history_turns", "prompt_tokens", "input_tokens", "cache_tokens",
)


def prefix_features(frame: "C.FrozenFrame",
                    qids: Optional[Iterable[str]] = None
                    ) -> Dict[str, Dict[str, Optional[float]]]:
    """Free prefix scalars per qid from the COMPRESSED arm's own row.

    These are the paper's "prefix-available" features (2606.21399 section 9.5:
    length statistics, progress indicators, state shape) mapped onto the fields
    the battery already logs.  ``step_idx`` is ``t34_common.step_index(qid)``.
    Missing values stay ``None`` (never a sentinel, never imputed here); the
    matrix builder decides what to do with them and reports it.
    """
    rows = frame.c2kv_by_qid
    qids = list(qids if qids is not None else rows)
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for q in qids:
        r = rows.get(q)
        if r is None:
            continue
        feats: Dict[str, Optional[float]] = {c: _opt_float(r.get(c))
                                             for c in PREFIX_FEATURE_COLUMNS}
        feats["step_idx"] = float(C.step_index(q))
        out[q] = feats
    return out


def feature_matrix(features: Dict[str, Dict[str, Optional[float]]],
                   qids: Sequence[str]) -> Tuple[List[str], np.ndarray, List[str], int]:
    """``(qids_kept, X, feature_names, n_rows_dropped)`` from a feature dict.

    Columns that are missing on EVERY row are dropped first and returned in the
    report (an all-missing column would otherwise delete the whole frame); rows
    still carrying a missing value are then dropped and counted.  Nothing is
    imputed -- HARD RULE 9 forbids median-imputing inside the evaluation set.
    """
    qids = [q for q in qids if q in features]
    if not qids:
        return [], np.zeros((0, 0)), [], 0
    names = sorted(set().union(*[set(features[q]) for q in qids]))
    raw = np.array([[np.nan if features[q].get(n) is None else float(features[q][n])
                     for n in names] for q in qids], dtype=float)
    usable = np.isfinite(raw).any(axis=0)
    names = [n for n, u in zip(names, usable) if u]
    raw = raw[:, usable]
    keep = np.isfinite(raw).all(axis=1) if raw.size else np.zeros(len(qids), dtype=bool)
    return ([q for q, k in zip(qids, keep) if k], raw[keep], names,
            int((~keep).sum()))


def make_pfail_scalar_fn(frame: "C.FrozenFrame",
                         features: Dict[str, Dict[str, float]],
                         *, seed: int = 20260905,
                         c_grid: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0),
                         inner_folds: int = 3
                         ) -> Callable[[set], Dict[str, float]]:
    """Build ``fn(held_out_sessions) -> {qid: p_fail score}`` for the controller.

    2606.21399 Def. ``def:ft``: the scalar baseline ESTIMATES the continuation
    failure probability.  The estimator is refitted for every held-out session
    set, so a controller fold never sees a scalar that was produced with its own
    test labels; ``C`` is picked on INNER session folds of the surviving rows
    (digest 4.0 item 6) and the fitted model then scores every feature row.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    sub = frame.trigger_subset()
    label = {r["qid"]: int(r["label_cw"]) for r in sub}
    sess = {r["qid"]: str(r["session_id"]) for r in sub}
    fit_qids, X_fit, names, _drop = feature_matrix(features, [r["qid"] for r in sub])
    all_qids, X_all, names_all, _d2 = feature_matrix(features, sorted(features))
    if not fit_qids or names_all != names:
        # column sets must agree or the fitted model cannot be applied
        keep = [n for n in names if n in names_all]
        idx_fit = [names.index(n) for n in keep]
        idx_all = [names_all.index(n) for n in keep]
        X_fit, X_all = X_fit[:, idx_fit], X_all[:, idx_all]

    y_fit = np.array([label[q] for q in fit_qids], dtype=int)
    g_fit = np.array([sess[q] for q in fit_qids])

    def fn(held_out: set) -> Dict[str, float]:
        keep = np.array([s not in held_out for s in g_fit], dtype=bool)
        if keep.sum() < 5 or y_fit[keep].sum() in (0, int(keep.sum())):
            return {}
        Xk, yk, gk = X_fit[keep], y_fit[keep], g_fit[keep]
        best_c, best_m = c_grid[0], -np.inf
        for c in c_grid:
            oof = np.full(len(yk), np.nan)
            for inner in C.grouped_folds(gk, inner_folds, seed + 1):
                itr, ite = ~inner, inner
                if ite.sum() == 0 or yk[itr].sum() in (0, int(itr.sum())):
                    continue
                sc = StandardScaler().fit(Xk[itr])
                clf = LogisticRegression(C=c, solver="liblinear", max_iter=2000)
                clf.fit(sc.transform(Xk[itr]), yk[itr])
                oof[ite] = clf.decision_function(sc.transform(Xk[ite]))
            ok = np.isfinite(oof)
            m = (C.average_precision(oof[ok], yk[ok])
                 if ok.sum() and yk[ok].sum() not in (0, int(ok.sum())) else None)
            if m is not None and m > best_m:
                best_m, best_c = m, c
        sc = StandardScaler().fit(Xk)
        clf = LogisticRegression(C=best_c, solver="liblinear", max_iter=2000)
        clf.fit(sc.transform(Xk), yk)
        scores = clf.decision_function(sc.transform(X_all))
        return {q: float(v) for q, v in zip(all_qids, scores)}

    return fn


def scalar_risk_scores(frame: "C.FrozenFrame",
                       features: Dict[str, Dict[str, float]],
                       *, seed: int = 20260905,
                       outer_folds: int = 5, inner_folds: int = 3
                       ) -> Dict[str, float]:
    """Out-of-fold ``p_fail``-like risk scalar over the 161-row trigger subset.

    2606.21399 Def. ``def:ft``: the scalar baseline is an ESTIMATE of the
    continuation-failure probability, not the label.  Fitted with
    :func:`t34_common.nested_cv_logistic` on the frame where the C->W label
    actually varies, then read off for whatever rows the arm sweep covers
    (see DEVIATIONS).
    """
    sub = frame.trigger_subset()
    label = {r["qid"]: int(r["label_cw"]) for r in sub}
    qids, X, _names, _dropped = feature_matrix(features, [r["qid"] for r in sub])
    if not qids:
        return {}
    y = np.array([label[q] for q in qids])
    groups = np.array([C.session_of(q) for q in qids])
    res = C.nested_cv_logistic(X, y, groups, outer_folds=outer_folds,
                               inner_folds=inner_folds, seed=seed, select_by="ap")
    return {q: float(v) for q, v in zip(qids, res["oof_scores"])
            if np.isfinite(v)}


# ===========================================================================
# CLI
# ===========================================================================

def _load_frame(root: str) -> "C.FrozenFrame":
    return C.FrozenAssets(Path(root)).load()


def _dump(obj: Any, out: Optional[str]) -> None:
    printable = json.dumps(_strip_big(obj), indent=1, ensure_ascii=True, default=str)
    print(printable)
    if out:
        C.freeze_json(Path(out), obj)
        print(f"[written] {out}")


def _strip_big(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: ("<%d rows>" % len(v) if k in ("per_row", "cells", "curve",
                                                  "ladder") and isinstance(v, (dict, list))
                    and len(v) > 12 else _strip_big(v))
                for k, v in obj.items()}
    return obj


def _arm_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if getattr(args, "canonical_arms", None):
        rows: List[Dict[str, Any]] = []
        for p in args.canonical_arms:
            rows.extend(load_arm_outcomes_label(Path(p)))
        return rows
    return arm_outcomes_label_from_dline([Path(p) for p in args.arms],
                                         bytes_per_kv_token=args.bytes_per_kv_token)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="t34_controls",
        description="t34 U10b -- section 4.11 controls: Calibration-Is-Not-Control "
                    "(2606.21399) tau_adv / RF+LCB witness / regret, and Reward-SNR "
                    "(2608.10441) random@matched-rate / placebo / positive control.")
    p.add_argument("--seed", type=int, default=20260905)
    sub = p.add_subparsers(dest="cmd", required=True)

    def _arms(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--arms", action="append", default=[],
                        help="D-line arm jsonl (repeatable)")
        sp.add_argument("--canonical-arms", action="append", default=[],
                        help="jsonl already in {qid,arm,correct,gpu_sec,kv_bytes} form")
        sp.add_argument("--bytes-per-kv-token", type=float, default=None,
                        help="bytes per appended KV token; without it kv_bytes stays "
                             "unknown and c_a uses GPU-seconds only")
        sp.add_argument("--harm-weight", type=float, default=1.0,
                        help="w, the wrong-outcome penalty (paper ALFWorld: 1.0)")
        sp.add_argument("--step-cost", type=float, default=0.01, help="s (paper: 0.01)")
        sp.add_argument("--cost-scale", type=float, default=0.05,
                        help="c_a budget (paper ALFWorld: 0.05)")
        sp.add_argument("--cost-mode", choices=("normalised", "flat"),
                        default="normalised")

    sp = sub.add_parser("tau-adv", help="tau_adv population split + 5x5 cost sweep")
    sp.add_argument("--root", default=".")
    sp.add_argument("--out", default=None)
    sp.add_argument("--no-sweep", action="store_true")
    _arms(sp)

    sp = sub.add_parser("witness-regret", help="RF+LCB controller, Gap, exploitability")
    sp.add_argument("--root", default=".")
    sp.add_argument("--out", default=None)
    sp.add_argument("--beta", type=float, default=1.0)
    sp.add_argument("--write-features", default=None,
                    help="write the free prefix feature frame through the guard")
    sp.add_argument("--include-quit", action="store_true",
                    help="add the paper third action (U_quit = 0, section 9.2); "
                         "only then are regret LEVELS comparable to Table tab:fork")
    _arms(sp)

    sp = sub.add_parser("random-rate", help="random@matched-rate comparator band")
    sp.add_argument("--root", default=".")
    sp.add_argument("--n-fires", type=int, default=None)
    sp.add_argument("--reps", type=int, default=2000)
    sp.add_argument("--ladder", action="store_true",
                    help="also run the optional granularity ladder "
                         "(per-instance -> KMeans K=4..64 -> declared regimes) "
                         "on the free prefix features")
    sp.add_argument("--out", default=None)

    sp = sub.add_parser("placebo-bestk", help="matched-moment placebo on best-k")
    sp.add_argument("--root", default=".")
    sp.add_argument("--flips", required=True, help="flip table jsonl {qid,k,correct}")
    sp.add_argument("--reps", type=int, default=2000)
    sp.add_argument("--out", default=None)

    sp = sub.add_parser("positive-control", help="detection strength s* of the bar")
    sp.add_argument("--root", default=".")
    sp.add_argument("--features", default=None,
                    help="optional real feature jsonl to place on the curve (s')")
    sp.add_argument("--reps", type=int, default=500)
    sp.add_argument("--out", default=None)

    sp = sub.add_parser("snr-floor", help="rho vs rho*(N) with a mandatory Delta")
    sp.add_argument("--root", default=".")
    sp.add_argument("--arm", required=True)
    sp.add_argument("--delta-definition", required=True)
    sp.add_argument("--out", default=None)
    _arms(sp)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    # fail fast on a missing INPUT before loading the frozen frame, so a missing
    # server artefact reads as a missing artefact and not as a frame error.
    if args.cmd == "placebo-bestk" and not Path(args.flips).exists():
        print("[ABORT] flip table not found: %s\n"
              "        guard 2 needs the frozen (qid, k) flip table, one "
              "{qid, k, correct} object per line.  The 823-trial run lives on "
              "the NPU server at ~/bench_results/d_v2/ and is NOT on this box; "
              "copy it back before quoting any placebo number." % args.flips)
        return 2
    frame = _load_frame(args.root)
    labels = frame.label_by_qid

    if args.cmd == "tau-adv":
        rows = _arm_rows(args)
        cont = continue_outcomes_label(frame, {r["qid"] for r in rows})
        rep = tau_adv_label_table(rows, cont, labels, w=args.harm_weight,
                                  s=args.step_cost, cap_tokens=frame.cap_tokens(),
                                  cost_scale=args.cost_scale, cost_mode=args.cost_mode)
        if not args.no_sweep:
            rep["cost_penalty_sweep"] = cost_penalty_sweep_label(
                rows, cont, labels, s=args.step_cost,
                cap_tokens=frame.cap_tokens(), cost_mode=args.cost_mode)
        rep["deviations"] = DEVIATIONS
        _dump(rep, args.out)
        return 0

    if args.cmd == "witness-regret":
        rows = _arm_rows(args)
        cont = continue_outcomes_label(frame, {r["qid"] for r in rows})
        feats = prefix_features(frame)
        if args.write_features:
            C.write_features_jsonl(
                Path(args.write_features),
                [{"qid": q, "session_id": C.session_of(q), **feats[q]}
                 for q in sorted(feats)],
                context="t34_controls prefix frame")
        # the reused-OOF scalar is kept ONLY to pick the row population and as a
        # fallback; every fitted use of g below goes through pfail_fn, which is
        # refitted inside each fold with that fold sessions held out.
        scalar = scalar_risk_scores(frame, feats, seed=args.seed)
        pfail_fn = make_pfail_scalar_fn(frame, feats, seed=args.seed)
        rep = witness_controller_regret_label(
            rows, cont, feats, scalar, w=args.harm_weight, s=args.step_cost,
            cap_tokens=frame.cap_tokens(), beta=args.beta,
            cost_scale=args.cost_scale, cost_mode=args.cost_mode, seed=args.seed,
            scalar_fit_fn=pfail_fn, include_quit=args.include_quit)
        # Gap + exploitability on the same rows and the same action set
        costs = arm_cost_table(rows, cost_scale=args.cost_scale, cost_mode=args.cost_mode)
        arms = sorted(costs)
        by = {}
        for r in rows:
            by.setdefault(r["qid"], {})[r["arm"]] = r
        want = [q for q in sorted(by)
                if q in cont and q in scalar and set(by[q]) == set(arms)]
        qids, _X, _n, _d = feature_matrix(feats, want)
        if not qids:
            rep["abstraction_gap"] = {"paper": PAPER_CONTROL, "n_scored": 0,
                                      "note": "no usable rows"}
            rep["exploitability"] = {"paper": PAPER_CONTROL, "n_rows": 0,
                                     "increment": None, "note": "no usable rows"}
            rep["deviations"] = DEVIATIONS
            _dump(rep, args.out)
            return 0
        U = np.array([[utility(cont[q], c_a=0.0, w=args.harm_weight, s=args.step_cost, m=0.0)]
                      + [utility(by[q][a]["correct"], c_a=costs[a]["c_a"],
                                 w=args.harm_weight, s=args.step_cost,
                                 m=(by[q][a]["gen_tokens"] or 0.0) / max(1, frame.cap_tokens()))
                         if a in by[q] else -np.inf for a in arms]
                      + ([0.0] if args.include_quit else [])
                      for q in qids], dtype=float)
        g = np.array([scalar[q] for q in qids])
        groups = np.array([C.session_of(q) for q in qids])
        rep["abstraction_gap"] = abstraction_gap_label(
            U, g, groups, seed=args.seed, scalar_fit_fn=pfail_fn, qids=qids)
        rep["exploitability"] = exploitability_label(
            feats, scalar, {q: int(U[i].argmax()) for i, q in enumerate(qids)},
            seed=args.seed, scalar_fit_fn=pfail_fn)
        rep["gap_action_set"] = (["continue"] + arms
                                 + (["quit"] if args.include_quit else []))
        rep["deviations"] = DEVIATIONS
        _dump(rep, args.out)
        return 0

    if args.cmd == "random-rate":
        sub = frame.trigger_subset()
        y = np.array([int(r["label_cw"]) for r in sub])
        clusters = C.session_clusters([r["session_id"] for r in sub])
        pf = np.array([bool(r["parse_fail_fire"]) for r in sub])
        n_fires = args.n_fires if args.n_fires is not None else int(pf.sum())
        pf_metrics = mask_metrics(pf, y)
        rnd = random_matched_rate_table(y, clusters, n_fires, reps=args.reps,
                                        seed=args.seed)
        rep = {
            "paper": PAPER_SNR,
            "parse_failure_only": pf_metrics,
            "random_at_matched_rate": rnd,
            "parse_failure_vs_random_band": baseline_vs_random_band(pf_metrics, rnd),
            "prevalence_chance_ap": C.prevalence(y),
            "deviations": DEVIATIONS,
        }
        if args.ladder:
            feats = prefix_features(frame)
            label = {r["qid"]: int(r["label_cw"]) for r in sub}
            qids, X, names, dropped = feature_matrix(feats, [r["qid"] for r in sub])
            yy = np.array([label[q] for q in qids])
            gg = np.array([C.session_of(q) for q in qids])
            rep["granularity_ladder"] = granularity_ladder(
                yy, gg, X, min(n_fires, len(qids)), seed=args.seed,
                reps=min(args.reps, 500))
            rep["granularity_ladder"]["feature_names"] = names
            rep["granularity_ladder"]["n_rows_dropped_missing"] = dropped
        _dump(rep, args.out)
        return 0

    if args.cmd == "placebo-bestk":
        flip_path = Path(args.flips)
        flips = C.load_flip_table(flip_path)
        if not flips:
            print("[ABORT] flip table %s parsed to zero (qid, k) rows; nothing "
                  "to run guard 2 on." % flip_path)
            return 2
        witness = (frame.witness or {}).get("entries", {})
        rep = {"paper": PAPER_SNR,
               "flip_table": flip_table_status(flips, witness,
                                               source=str(flip_path)),
               "per_qid": placebo_bestk_scan(flips, witness, level="per_qid",
                                             reps=args.reps, seed=args.seed),
               "pooled": placebo_bestk_scan(flips, witness, level="pooled",
                                            reps=args.reps, seed=args.seed),
               "deviations": DEVIATIONS}
        _dump(rep, args.out)
        return 0

    if args.cmd == "positive-control":
        sub = frame.trigger_subset()
        y = np.array([int(r["label_cw"]) for r in sub])
        groups = np.array([r["session_id"] for r in sub])
        rep = positive_control_curve(y, groups, reps=args.reps, seed=args.seed)
        if args.features:
            raw = {r["qid"]: {k: v for k, v in r.items() if k not in C.META_COLS}
                   for r in load_jsonl(args.features)}
            label = {r["qid"]: int(r["label_cw"]) for r in sub}
            qids, X, names, dropped = feature_matrix(raw, [r["qid"] for r in sub])
            yy = np.array([label[q] for q in qids])
            gg = np.array([C.session_of(q) for q in qids])
            res = C.nested_cv_logistic(X, yy, gg, seed=args.seed, select_by="ap")
            rep["real_features"] = {"auprc": res["auprc"], "n_scored": res["n_scored"],
                                    "n_rows_dropped_missing": dropped,
                                    "columns": names}
            rep["s_prime"] = locate_real_on_curve(rep, res["auprc"])
        rep["deviations"] = DEVIATIONS
        _dump(rep, args.out)
        return 0

    if args.cmd == "snr-floor":
        rows = _arm_rows(args)
        cont = continue_outcomes_label(frame, {r["qid"] for r in rows})
        _q, deltas = reward_deltas_label(rows, cont, args.arm)
        rep = reward_snr_floor(deltas, delta_definition=args.delta_definition)
        rep["arm"] = args.arm
        rep["deviations"] = DEVIATIONS
        _dump(rep, args.out)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
