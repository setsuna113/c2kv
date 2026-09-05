# -*- coding: utf-8 -*-
"""t34 U10a / digest 4.11 - thresholds and guarantees: turning a trigger score
into fire / do-not-fire with a distribution-free statement attached.

Five machines, each faithful to one paper, none of them a signal:

  (A) CALM / LTT fixed-sequence testing            2207.07061 4 + 2110.01052 2.2/2.3.1
  (B) CRC feasibility floor + three nested UCBs    2606.29054 2.1/2.2
  (C) Clopper-Pearson recall gate                  2607.06503 3.3/3.4/3.4.1
  (D) weighted (non-exchangeable) CRC              2310.01262 Thm 1
  (E) stratified / group-conditional adaptive CRC  2406.17819 2 example 2
  base primitive: conformal risk control           2208.02814 eq (4) / Thm 1

Everything here is torch-free numpy/scipy and runs ON THIS BOX; no GPU, no
model.  Nothing in this module computes a feature: every entry point takes a
score vector (risk-oriented, +1 = higher is riskier, declared in
``configs/t34/orientations_calib.json``) plus a LABEL-SIDE loss vector that the
caller supplies, and returns a threshold with the statement that goes with it.

ORIENTATION CONTRACT.  Our features are risk-oriented (+1 = higher is riskier).
2606.29054's ``s`` is a RELIABILITY score (emit iff ``s >= lambda``) and
2607.06503's ``f`` is a FAILURE score (abort iff ``f > tau``).  Every routine
here takes ``orientation`` explicitly and converts once, at the top; nothing
downstream re-guesses a sign.  (Digest 4.0 pitfall: "residualising an
un-oriented score".)

LABEL DISCIPLINE.  The loss inputs of (A)-(E) are label-side by construction
(they read ``tool_name_match``).  They are produced here only by functions
whose names contain ``_label_`` and they never enter a feature frame.  The only
score this module computes itself is the L1 parse-failure baseline, which reads
the compressed arm's own emitted text and nothing else.

DENOMINATORS.  The 161-row trigger subset (93 C->W / 68 C->C, 100 session
clusters) is the evaluation frame; its prevalence 0.578 is chance AP there.
The 900-row paired frame (227 sessions) is the frame for Reading A's mu =
93/900 = 0.1033.  The 188-row harm manifest is 100 session clusters.  All of
these are COMPUTED from the frozen assets by the CLI, never typed in.

RUNBOOK (all steps run HERE, on this box; zero GPU-sec)
------------------------------------------------------
  0. cd C:/Users/yl998/Documents/programming/c2kv/tmp/t34-migration
     set PYTHONIOENCODING=utf-8

  1. Feasibility gate FIRST (2606.29054 migration step 1-3): decide whether a
     coverage-guaranteed trigger is worth building at all, under both readings
     of the loss, before any threshold is fitted.

       python agent/t34_calibrate.py feasibility --root . \
           --alphas 0.10 0.05 0.02 0.01 --out results/t34/calib_feasibility.json

  2. Sample-complexity table (2607.06503 3.4.1): which false-reset promise our
     n can support at all.  Row unit AND session-cluster unit.

       python agent/t34_calibrate.py sample-complexity --root . \
           --rho-star 0.90 0.95 0.98 0.99 --out results/t34/calib_npos.json

  3. Clopper-Pearson recall gate on the battery, single gate (R_g = 1; the
     cascade is IDENTICALLY a single gate on a single-step face and no cascade
     claim may be made there).

       python agent/t34_calibrate.py recall-gate --root . \
           --rho-star 0.95 --margin 0.02 --out results/t34/calib_recall_gate.json

  4. CALM/LTT fixed sequence.  Needs a repair-outcome input: either the D-line
     per-row repaired correctness (--repair-outcomes jsonl, {qid, correct}) or
     an EXPLICIT assumption (--assume-repair-success P), which is echoed in the
     artifact as an assumption and is not a measurement.

       python agent/t34_calibrate.py calm --root . --delta 0.20 --eps 0.05 \
           --assume-repair-success 0.806 --out results/t34/calib_calm.json

  5. Weighted (non-exchangeable) CRC, battery -> bench threshold transport.
     Requires BOTH faces' loss rows; prints the source/target violation pair.

       python agent/t34_calibrate.py weighted-crc --root . \
           --target-losses results/t34/bench_face_losses.jsonl \
           --rho 0.9 --alpha 0.10 --out results/t34/calib_weighted.json

  6. Stratified CRC on a label-free prefix scalar; the stratum count is derived
     from the n_pos rule in step 2 and is never chosen post hoc.

       python agent/t34_calibrate.py stratified-crc --root . \
           --stratifier actual_compression_ratio --alpha 0.10 \
           --out results/t34/calib_stratified.json

  Tests (here):
       python -m pytest agent/test_t34_calibrate.py -q

DEVIATIONS from the papers are enumerated in ``DEVIATIONS`` below; the prereg
copies that list verbatim.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from t33_labels import guard_columns  # noqa: E402
from t34_common import (  # noqa: E402
    BASE_RATE_900,
    FrozenAssets,
    FrozenFrame,
    clustered_bootstrap,
    freeze_json,
    grouped_folds,
    operating_point,
    prevalence,
    session_clusters,
    session_of,
)

#: Label-free prefix scalars admissible as stratifiers for (E).  Digest 4.11
#: names gist_tokens / actual_compression_ratio / n_docs / dropped_docs /
#: kept_history_tokens / hybrid_top_k; on the frozen battery row ``n_docs`` is
#: called ``doc_chunks`` and ``dropped_docs`` exists only in the U2 sidecar, so
#: it is deliberately absent here rather than silently substituted.  Anything
#: outside this map is refused by the CLI, and every entry additionally passes
#: ``t33_labels.guard_columns``.  ``decision_step`` is NOT admissible: it is
#: ``doc_chunks + 1``, not the session's step order (digest 4.0 / rule 5).
PREFIX_SCALARS: Dict[str, str] = {
    "gist_tokens": "gist_tokens",
    "actual_compression_ratio": "actual_compression_ratio",
    "n_docs": "doc_chunks",
    "doc_chunks": "doc_chunks",
    "kept_history_tokens": "kept_history_tokens",
    "hybrid_top_k": "hybrid_top_k",
    "compressed_history_tokens": "compressed_history_tokens",
}

__all__ = [
    "DEVIATIONS", "PREFIX_SCALARS",
    "hoeffding_pvalue", "hoeffding_bentkus_pvalue", "h1_bernoulli_kl",
    "hoeffding_slack", "aggregate_to_units", "fixed_sequence_test",
    "bonferroni_reject", "certify_fire_rate",
    "ucb_hoeffding", "ucb_empirical_bernstein", "ecrc_wealth", "ecrc_certified",
    "crc_select_lambda", "bound_ordering_report", "recommend_bound",
    "abstention_lower_bound", "finite_sample_term", "feasibility_table",
    "cp_lower_one_sided", "cp_recall_threshold", "sample_complexity_n_pos",
    "certifiable_recall_ceiling", "assert_no_cascade_claim",
    "budget_feasibility_search", "frozen_certificate", "cascade_thresholds",
    "crc_lambda_hat", "weighted_crc_lambda_hat", "similarity_weights",
    "face_violation_report", "selective_loss_matrix",
    "derive_n_strata", "stratified_crc",
    "battery_label_divergence_c2kv", "battery_label_cw_loss", "json_safe",
    "parse_fail_score",
]

# ---------------------------------------------------------------------------
# Pre-registered departures from the source algorithms.  The prereg copies this.
# ---------------------------------------------------------------------------

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "CALM / LTT fixed sequence",
        "paper": "2207.07061 4 (p-value + FST); 2110.01052 2.2 Prop. 'Hoeffding-Bentkus "
                 "inequality p-values' eq. (hb-p-value), 2.3.1 Alg. 'Fixed sequence testing'",
        "what": "The calibration unit is a SESSION (mean divergence over the session's rows), "
                "not a prompt.  n in the concentration inequality is the number of session "
                "clusters present in the frame, not the number of rows.",
        "why": "CALM's p-values assume i.i.d. calibration draws; our rows are clustered by "
               "session (transfer card 2207.07061, pitfall 2; digest 4.0 item 1).  Row-level n "
               "is still computed and printed, flagged iid_assumption_violated=True, so the "
               "4.1pp-vs-14.4pp cost of the correction is visible rather than assumed away.",
    },
    {
        "method": "CALM / LTT fixed sequence",
        "paper": "2207.07061 4 ('lambda := min(Lambda_valid U {1})')",
        "what": "The grid direction is reversed and the always-valid fallback leg is REMOVED.  "
                "Our grid walks from 'always fire' (safe, expensive, inexact) toward 'never "
                "fire', and when nothing is rejected we return lambda_chosen=None with "
                "certified=False instead of falling back.",
        "why": "CALM's lambda=1 leg is valid because LLM_early(P,1) = LLM_full(P) EXACTLY.  Our "
               "maximal-effort arm is not exact (raw_erratum_tail 75/93 = 80.6%, card "
               "2207.07061 'Structural mismatches' (b) and pitfall 3), so the feasible region "
               "is bounded away from delta=0 and 'certification failed' is a legal outcome "
               "(digest 4.11, the delta=0.1/textual/state row returning 8.00/8).",
    },
    {
        "method": "CALM / LTT fixed sequence",
        "paper": "2207.07061 4 (Lambda = a fixed grid of exit thresholds lambda_j; "
                 "L_i(lambda_j) is a function of row i alone)",
        "what": "Our grid indexes a FIRE RATE q, and the threshold at grid point q is the "
                "empirical (1-q)-quantile of the calibration scores.  L_i(q) is therefore a "
                "function of the WHOLE calibration sample, not of row i alone.",
        "why": "The cost side of our decision is a fire rate (bytes and GPU-sec), not a score "
               "value, and a score-value grid is not comparable across the score families this "
               "unit has to serve.  The statistical price is real and is flagged in the "
               "artifact as grid_is_empirical_quantile: the Hoeffding / Hoeffding-Bentkus "
               "p-values assume the L_i(lambda_j) are independent for a FIXED lambda_j, which a "
               "sample quantile breaks (on top of the session clustering of DEVIATIONS #1).  "
               "The p-values are reported as APPROXIMATE for that reason; a caller who needs "
               "the exact LTT statement must pass a pre-registered grid of score VALUES.",
    },
    {
        "method": "CALM / LTT fixed sequence",
        "paper": "2110.01052 2.3.1 (FST) vs 2.3 (any FWER-controlling procedure)",
        "what": "Bonferroni is provided as a swappable FWER component alongside FST.",
        "why": "Digest 4.11 (Learn-then-Test entry): CALM leaves the FWER procedure as a "
               "replaceable part; the validity of any substitute must be checked against the LTT "
               "original, not inferred from CALM.  2110.01052 2.3.1 names Bonferroni as the "
               "comparator and reports FST is more powerful; we implement both and make no claim "
               "beyond what that section states.",
    },
    {
        "method": "CRC feasibility / e-CRC",
        "paper": "2606.29054 2.1 + Appendix Alg. 'e-CRC' (W_j = W_{j-1}(1 + kappa_j(alpha - "
                 "r_j)); kappa_j = clip((alpha - mu_hat_j)/(alpha(1-alpha)), 0, 0.5) with "
                 "mu_hat_j the running mean of r_1..r_{j-1}; certify when W_m >= 1/delta by Ville)",
        "what": "The DEFAULT bet (kelly='paper') is their pseudocode's closed form verbatim, "
                "with kappa_1 = 0 because mu_hat_1 is undefined (their algorithm defines "
                "mu_hat_j only for j >= 2 and prints no first bet).  Two alternative "
                "PREDICTABLE bets are offered and are NOT theirs: kelly='approx' "
                "(second-order Kelly clip(E[alpha-r]/E[(alpha-r)^2], 0, 0.5) from running "
                "statistics) and kelly='grid' (exact 51-point log-growth search over the past).",
        "why": "Ville's inequality needs a non-negative supermartingale, i.e. kappa_j "
               "predictable w.r.t. the past; the transfer card says only 'Kelly-optimal', but "
               "the arXiv source's Appendix algorithm gives both the closed form and the "
               "predictability, so the card's wording is superseded by the source.  All three "
               "bets are predictable and therefore valid; only the default reproduces their "
               "published wealth path, and the other two are exposed so the power cost of a "
               "different bet is measurable rather than hidden.  Also: their e-CRC loop uses "
               "|E_lambda| >= 5 while their protocol's non-degeneracy rule is |E_lambda| >= 10; "
               "we use 10 everywhere (the stricter of the two, and the one the digest names).",
    },
    {
        "method": "CRC feasibility / bound ordering",
        "paper": "2606.29054 2.1 Prop. 'Bound Ordering' (Lambda*_Hoeff subset Lambda*_Bern "
                 "subset Lambda*_e-CRC)",
        "what": "The nesting is REPORTED and checked on the data at hand, never asserted.",
        "why": "Empirical Bernstein is tighter than Hoeffding only when sigma^2 is small (the "
               "card states 'tighter whenever sigma^2 < 1/4'); asserting the inclusion would "
               "hide the case where it fails.  bound_ordering_report returns the three sets and "
               "a nesting_holds flag.",
    },
    {
        "method": "CRC feasibility - loss reading",
        "paper": "2606.29054 3.2 Table 'tab:base_risk' (mu -> floor (mu-alpha)/(1-alpha))",
        "what": "Two readings are computed side by side and both are printed: Reading A "
                "(loss = C->W indicator) and Reading B (loss = 1 - tool_name_match of the "
                "compressed arm).  Neither is hard-coded; both mu are read off the frozen "
                "assets.",
        "why": "The card's open question 1: the feasibility verdict flips entirely on this "
               "choice (0.1033 feasible vs ~0.79 hopeless).  Printing only one would be a "
               "silent pre-registration.",
    },
    {
        "method": "Conformal risk control primitive",
        "paper": "2208.02814 eq. (4) lambda_hat = inf{lambda : n/(n+1) Rhat_n(lambda) + "
                 "B/(n+1) <= alpha}, Thm 1",
        "what": "The digest lists 2208.02814 as cite-only ('bu qian yi dai ma').  We implement "
                "its eq. (4) exactly anyway, as the base calibrator that (D) and (E) extend.",
        "why": "(D) 2310.01262 Thm 1 and (E) 2406.17819 2 example 2 are both defined as "
               "modifications of this estimator; implementing them without their base would "
               "make the equal-weight / single-group reductions untestable.  No number from "
               "2208.02814 is quoted.",
    },
    {
        "method": "CRC for selective prediction (loss matrix construction)",
        "paper": "2208.02814 Thm 1 (monotone loss) vs 2606.29054 2 (conditional target "
                 "E[R | s >= lambda*] <= alpha)",
        "what": "selective_loss_matrix builds L_i(lambda) = r_i * 1{emit_i(lambda)}, which is "
                "non-increasing in lambda and gives MARGINAL control of E[R * 1{emit}] <= alpha "
                "-- NOT the conditional target of 2606.29054.",
        "why": "The two papers control different quantities and the difference matters at low "
               "emit rates (marginal control is trivially satisfied by abstaining).  Whenever "
               "the conditional statement is what is wanted, use crc_select_lambda (B), not the "
               "CRC primitive.  Both are provided and the docstrings say which is which.",
    },
    {
        "method": "CRC feasibility - finite-sample term",
        "paper": "2606.29054 2.2 Prop. 'Minimum Abstention Lower Bound' "
                 "(abstention >= (mu-alpha)/(1-alpha) - O(sqrt(log(1/delta)/n)))",
        "what": "finite_sample_term returns the BARE rate sqrt(log(1/delta)/n) with no "
                "constant, and the feasibility table's floor_exceeds_correction flag is "
                "therefore a comparison against that bare rate, not a separation.  Every "
                "row carries separation_claimable=None.",
        "why": "The constant inside their O() is in the unread proof appendix (card open "
               "question 2).  At n = 900, delta = 0.1 the bare rate is 0.0506 while the "
               "alpha = 0.05 floor is 0.0561: any constant above ~1.11 erases the gap, so "
               "quoting it as a separation would be a claim the source does not support.  "
               "The card's conservative read (only alpha <= 0.02 survives) is printed "
               "beside it.",
    },
    {
        "method": "Automatically adaptive CRC - stratum count denominator",
        "paper": "2607.06503 3.4.1 ('let n_pos be the number of SUCCESSFUL episodes in the "
                 "independent certification sample')",
        "what": "derive_n_strata and the thin-stratum fallback divide the SUCCESS "
                "(must-not-fire) count by n_pos_required, not the total row count; the CLI "
                "passes success_mask = (label_cw == 0).  Callers may omit the mask, in "
                "which case the raw unit count is used and derivation_basis says so.",
        "why": "Their n_pos counts successes, and on the 161-row trigger frame 93 of those "
               "rows are positives.  Dividing 161 by 59 would license two strata and print "
               "conditional_guarantee=True for both, while each stratum would hold only "
               "~34 C->C rows -- exactly the failure digest 4.11 warns about ('68 C->C rows "
               "cut into four strata leaves each stratum's Clopper-Pearson bound "
               "meaningless').  With the mask the frozen frame yields one stratum, i.e. the "
               "digest's own conclusion that the calibration set cannot be cut.",
    },
    {
        "method": "Clopper-Pearson recall gate",
        "paper": "2607.06503 3.1/3.3 (per-round gates, R_g = 6 rounds, task grouping)",
        "what": "On the battery face R_g is forced to 1 (assert_no_cascade_claim) and the "
                "grouping unit is the SESSION, not the task.",
        "why": "The battery is single-step teacher-forced, so the cascade's entire 1.5x-8.8x "
               "advantage over a single gate is identically zero there and claiming it would be "
               "an overclaim (card 2607.06503 pitfall 2; digest 4.11).  The multi-gate code path "
               "exists but is parameterised for the bench proxy face only.",
    },
    {
        "method": "Clopper-Pearson recall gate",
        "paper": "2607.06503 3.4 ('the selected cascade is frozen and evaluated once on an "
                 "independent certification sample; because the cascade is fixed before these "
                 "data are seen, the bound is exact and distribution-free')",
        "what": "The gate reported as the operating point is calibrated on ALL C->C rows, so "
                "the Clopper-Pearson bound recomputed on those same rows is IN-SAMPLE and is "
                "returned under 'certificate_in_sample' with valid_post_selection=False.  The "
                "defensible number is 'certificate_split': tau is refitted on a "
                "session-grouped calibration half and certified on the disjoint half.",
        "why": "Their certificate is exact only because the cascade is fixed before the "
               "certification data are observed.  Recomputing the bound on the calibration "
               "rows that chose tau is precisely the post-selection error the construction "
               "exists to avoid, and at n_r = 68 the difference is not cosmetic -- the split "
               "halves n_pos and therefore lowers the certifiable recall ceiling "
               "alpha_m^(1/n_pos).  Both numbers are printed so the cost of honesty is visible.",
    },
    {
        "method": "Clopper-Pearson recall gate",
        "paper": "2607.06503 5 (the paper reports no precision anywhere)",
        "what": "Precision on the fire set is computed and returned next to recall.",
        "why": "Card 2607.06503, 'Mapping': precision is our metric 2 and is absent from their "
               "protocol; on a 0.1033 base rate a recall-optimised gate can certify while being "
               "useless.  We must supply it ourselves.",
    },
    {
        "method": "Non-exchangeable CRC",
        "paper": "2310.01262 Thm 1 eq. (lambda_hat), weights section 'How to choose weights'",
        "what": "Implemented exactly (N_w = sum w_i, Rhat_n = (1/N_w) sum w_i L_i, lambda_hat = "
                "inf{lambda : N_w/(N_w+1) Rhat + B/(N_w+1) <= alpha}).  The weight rule is "
                "supplied by the caller and declared; similarity_weights implements one "
                "documented rule (exponential decay in a declared covariate distance, the "
                "maxent choice their weights section justifies).",
        "why": "The transfer card for 2310.01262 does not exist (catalogue critic C.3, ID and "
               "abstract only).  The formula above was read from the arXiv LaTeX source IN THIS "
               "SESSION, not from the card; the card-level statement in the digest ('the "
               "degradation term is not given in the card') is superseded by the source, which "
               "gives it as (B-A) * sum_i wtilde_i * d_TV(Z, Z^i).  That term is NOT estimable "
               "from our data, so NO numeric finite-sample guarantee is claimed anywhere in "
               "this module; only the source-face / target-face violation pair is reported.",
    },
    {
        "method": "Automatically adaptive CRC",
        "paper": "2406.17819 2 example 2 (Lambda = {Phi(x)^T theta}, Phi = group indicators "
                 "=> E[loss | Phi(X)=j] <= alpha for all j); Thm 'main-validity'",
        "what": "Only the DISJOINT-group special case is implemented: strata are a partition "
                "induced by quantile bins of one label-free prefix scalar, and each stratum runs "
                "the CRC primitive on its own rows.  The general vector-space / regulariser "
                "optimisation of their main theorem is NOT implemented.  Strata too small for "
                "the n_pos rule fall back to the GLOBAL threshold and are flagged "
                "conditional_guarantee=False.",
        "why": "Digest 4.11 prescribes exactly the disjoint stratified version with a global "
               "fallback, over label-free prefix covariates.  The fallback is our addition, not "
               "theirs, and it voids the per-stratum conditional statement for that stratum -- "
               "which is why every fallback stratum is flagged rather than silently pooled.  "
               "The stratum COUNT is derived from 2607.06503's n_pos >= ln alpha_m / ln rho* "
               "rule before the data are cut, never tuned afterwards.",
    },
]


# ---------------------------------------------------------------------------
# (A) CALM / LTT: p-values, fixed sequence testing, direction reversal
# ---------------------------------------------------------------------------

def hoeffding_pvalue(emp_risk: float, delta: float, n: int) -> float:
    """p_j = exp(-2 n max(0, delta - Ehat(lambda_j))^2).

    Implements 2207.07061 4 eq. (p^Hoeffding), the inverted Hoeffding bound
    for H_j: "lambda_j is not consistent".  Small p <=> the calibration mean
    sits far enough BELOW the tolerance delta to reject.
    """
    if n <= 0:
        return 1.0
    slack = max(0.0, float(delta) - float(emp_risk))
    return float(math.exp(-2.0 * n * slack * slack))


def h1_bernoulli_kl(a: float, b: float) -> float:
    """h_1(a, b) = a log(a/b) + (1-a) log((1-a)/(1-b)).

    The binary KL used by the Hoeffding-Bentkus p-value of 2110.01052 2.2,
    Prop. "Hoeffding-Bentkus inequality p-values" (eq. hb-p-value).
    """
    a = float(a)
    b = float(b)
    if not (0.0 < b < 1.0):
        return float("inf")
    t1 = 0.0 if a <= 0.0 else a * math.log(a / b)
    t2 = 0.0 if a >= 1.0 else (1.0 - a) * math.log((1.0 - a) / (1.0 - b))
    return float(t1 + t2)


def hoeffding_bentkus_pvalue(emp_risk: float, delta: float, n: int) -> float:
    """p^HB = min( exp{-n h_1(Rhat ^ delta, delta)},  e * P(Bin(n, delta) <= ceil(n Rhat)) ).

    Implements 2110.01052 2.2, Prop. "Hoeffding-Bentkus inequality p-values",
    eq. (hb-p-value), with their alpha playing the role of CALM's tolerance
    delta.  2207.07061 4 footnote says this is the bound CALM actually uses;
    CALM itself prints only the Hoeffding form, so the HB expression is taken
    from the LTT original (read from its LaTeX source in this session), not
    from the CALM transfer card.
    """
    from scipy.stats import binom
    if n <= 0:
        return 1.0
    rhat = float(emp_risk)
    d = float(delta)
    a = min(rhat, d)
    first = math.exp(-n * h1_bernoulli_kl(a, d)) if 0.0 < d < 1.0 else 1.0
    second = math.e * float(binom.cdf(math.ceil(n * rhat), n, d))
    return float(min(1.0, first, second))


def hoeffding_slack(n: int, eps: float = 0.05) -> float:
    """Slack the Hoeffding p-value needs to reject at level eps: sqrt(ln(1/eps)/(2n)).

    Card 2207.07061, "n available and the number that decides feasibility":
    rejecting at eps = 0.05 needs ``delta - Ehat >= sqrt(ln(20)/(2n))``.  The
    numbers this returns at our two candidate units (rows vs session clusters)
    are COMPUTED here, never transcribed.
    """
    if n <= 0 or not (0.0 < eps < 1.0):
        return float("nan")
    return float(math.sqrt(math.log(1.0 / eps) / (2.0 * n)))


def aggregate_to_units(unit_ids: Sequence[str], values: Sequence[float]) -> Tuple[List[str], np.ndarray]:
    """Mean of ``values`` within each unit (a session), units sorted by id.

    Digest 4.11 / card pitfall 2: L_i must be the SESSION's mean divergence
    before it enters a concentration inequality, otherwise the independence
    assumption behind the p-value is false.
    """
    order: Dict[str, List[float]] = {}
    for u, v in zip(unit_ids, values):
        order.setdefault(str(u), []).append(float(v))
    keys = sorted(order)
    return keys, np.array([float(np.mean(order[k])) for k in keys], dtype=float)


@dataclass
class FSTResult:
    """Outcome of a fixed-sequence walk; ``certified`` False is a legal result."""
    rows: List[Dict[str, Any]]
    lambda_valid: List[float]
    lambda_chosen: Optional[float]
    certified: bool
    stopped_at: Optional[int]
    n_units: int
    unit: str
    eps: float
    delta: float
    pvalue_kind: str
    n_starts: int
    slack_required: float
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["extras"] = dict(self.extras)
        return d


def fixed_sequence_test(
    grid: Sequence[float],
    emp_risks: Sequence[float],
    *,
    n_units: int,
    delta: float,
    eps: float = 0.05,
    pvalue: str = "hb",
    n_starts: int = 1,
    unit: str = "session",
) -> FSTResult:
    """Fixed sequence testing over an ORDERED grid; stop at the first acceptance.

    2110.01052 2.3.1, Algorithm "Fixed sequence testing" and Prop. "Fixed
    sequence testing controls FWER"; 2207.07061 4 uses the |J| = 1 instance,
    which is what this implements: ONE start, at the walking level
    ``eps / n_starts``.  ``n_starts > 1`` therefore does NOT run their
    multi-start variant -- it only tightens the level of the single walk, which
    is strictly more conservative than Algorithm "Fixed sequence testing" with
    |J| = n_starts and so never invalid, merely less powerful.

    ``grid`` must be ordered from the SAFEST end toward the cheapest end; the
    caller owns that ordering (LTT 2.3.1: "selecting the ordering without
    looking at the calibration data").  ``emp_risks[j]`` is Ehat(grid[j]) on
    the ``n_units`` independent calibration units.

    Returns the rejection prefix and ``lambda_chosen`` = the LAST rejected grid
    point (the cheapest certifiable setting).  There is deliberately no
    always-valid fallback leg: when nothing is rejected the result is
    ``certified=False, lambda_chosen=None`` (see DEVIATIONS).
    """
    if pvalue not in ("hoeffding", "hb"):
        raise ValueError("pvalue must be 'hoeffding' or 'hb'")
    if len(grid) != len(emp_risks):
        raise ValueError("grid and emp_risks must have the same length")
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")
    level = float(eps) / float(n_starts)
    fn = hoeffding_pvalue if pvalue == "hoeffding" else hoeffding_bentkus_pvalue

    rows: List[Dict[str, Any]] = []
    valid: List[float] = []
    stopped_at: Optional[int] = None
    for j, (lam, r) in enumerate(zip(grid, emp_risks)):
        p = fn(float(r), float(delta), int(n_units))
        rejected = bool(stopped_at is None and p <= level)
        rows.append({"index": j, "lambda": float(lam), "emp_risk": float(r),
                     "p": float(p), "rejected": rejected,
                     "tested": bool(stopped_at is None)})
        if stopped_at is None:
            if rejected:
                valid.append(float(lam))
            else:
                stopped_at = j
    return FSTResult(
        rows=rows,
        lambda_valid=valid,
        lambda_chosen=(valid[-1] if valid else None),
        certified=bool(valid),
        stopped_at=stopped_at,
        n_units=int(n_units),
        unit=unit,
        eps=float(eps),
        delta=float(delta),
        pvalue_kind=pvalue,
        n_starts=int(n_starts),
        slack_required=hoeffding_slack(int(n_units), level),
    )


def bonferroni_reject(pvalues: Sequence[float], eps: float = 0.05) -> List[bool]:
    """Bonferroni FWER control: reject H_j iff p_j <= eps / N.

    The swappable-component variant.  2110.01052 2.3.1 names Bonferroni as the
    comparator to fixed sequence testing and states FST "offers large power
    improvements over Bonferroni"; the validity of ANY substitute FWER
    procedure must be checked against 2110.01052 2.3, never inferred from
    2207.07061's use of FST (digest 4.11, Learn-then-Test entry).
    """
    n = len(pvalues)
    if n == 0:
        return []
    lvl = float(eps) / n
    return [bool(float(p) <= lvl) for p in pvalues]


def certify_fire_rate(
    scores: Sequence[float],
    loss_nofire: Sequence[float],
    loss_fire: Sequence[float],
    session_ids: Sequence[str],
    *,
    delta: float,
    eps: float = 0.05,
    step: float = 0.05,
    pvalue: str = "hb",
    orientation: int = 1,
    unit: str = "session",
    n_starts: int = 1,
) -> FSTResult:
    """CALM's calibration, run on OUR reversed grid: cheapest certifiable fire rate.

    2207.07061 4, Def. "textual consistency" ``E[D(Y_trig(lambda), Y_full)] <=
    delta`` with D = 1 - tool_name_match in {0,1} (card 2207.07061, "Definitions
    to adopt verbatim"), solved by LTT fixed sequence testing (2110.01052
    2.3.1) at eps = 0.05 with grid step 0.05.

    Direction: CALM walks lambda DOWN from lambda = 1, whose fallback is exact.
    Our safe end is "always fire", which is neither exact nor free, so the grid
    is a FIRE RATE walking 1.00 -> 0.00 and the returned value is the smallest
    (cheapest) fire rate still in the rejection prefix.

    ``scores`` are risk-oriented when ``orientation = +1``: the top ``q``
    fraction fires.  ``loss_nofire[i]`` is the per-row divergence if row i is
    not repaired (label side, from ``battery_label_divergence_c2kv``);
    ``loss_fire[i]`` is the divergence if it IS repaired -- a measured D-line
    repair outcome, or an explicitly declared assumption.  Both must lie in
    [0, 1] (LTT requires a bounded loss).

    L_i(q) is aggregated to SESSION means before the p-value (DEVIATIONS #1).
    """
    s = np.asarray(scores, dtype=float)
    ln = np.asarray(loss_nofire, dtype=float)
    lf = np.asarray(loss_fire, dtype=float)
    if not (len(s) == len(ln) == len(lf) == len(session_ids)):
        raise ValueError("scores / losses / session_ids length mismatch")
    for name, arr in (("loss_nofire", ln), ("loss_fire", lf)):
        if arr.size and (np.nanmin(arr) < 0.0 or np.nanmax(arr) > 1.0):
            raise ValueError(f"{name} must be bounded in [0, 1] (LTT requirement)")
    risk = s if orientation >= 0 else -s

    n_steps = int(round(1.0 / step))
    grid = [round(1.0 - k * step, 10) for k in range(n_steps + 1)]  # 1.00 -> 0.00

    rows_by_unit = list(map(str, session_ids))
    emp: List[float] = []
    fire_counts: List[int] = []
    for q in grid:
        n_fire = int(round(q * len(risk)))
        order = np.argsort(-risk, kind="mergesort")
        fire = np.zeros(len(risk), dtype=bool)
        fire[order[:n_fire]] = True
        li = np.where(fire, lf, ln)
        _, per_unit = aggregate_to_units(rows_by_unit, li)
        emp.append(float(per_unit.mean()) if unit == "session" else float(li.mean()))
        fire_counts.append(int(fire.sum()))

    n_units = len(set(rows_by_unit)) if unit == "session" else len(risk)
    res = fixed_sequence_test(grid, emp, n_units=n_units, delta=delta, eps=eps,
                              pvalue=pvalue, n_starts=n_starts, unit=unit)
    for row, fc in zip(res.rows, fire_counts):
        row["n_fire"] = fc
    n_distinct = int(np.unique(risk[np.isfinite(risk)]).size)
    res.extras = {
        "n_rows": int(len(risk)),
        "n_sessions": int(len(set(rows_by_unit))),
        "slack_rows_pp": 100.0 * hoeffding_slack(len(risk), eps),
        "slack_sessions_pp": 100.0 * hoeffding_slack(len(set(rows_by_unit)), eps),
        "iid_assumption_violated": bool(unit != "session"),
        "grid_is_empirical_quantile": True,
        "grid_note": ("the threshold at each grid point is the empirical (1-q)-quantile of "
                      "the calibration scores, so L_i(q) depends on the whole sample and the "
                      "p-values are APPROXIMATE (DEVIATIONS: CALM quantile grid)"),
        "score_is_binary": bool(n_distinct <= 2),
        "n_distinct_scores": n_distinct,
        "tie_note": ("a score with <= 2 distinct values cannot realise most fire rates: the "
                     "grid points inside a tie block differ only by an arbitrary index order"
                     if n_distinct <= 2 else None),
        "grid_step": float(step),
        "slack_note": ("slack_required on this result is the HOEFFDING requirement at the "
                       "walking level; with pvalue='hb' the Hoeffding-Bentkus bound is "
                       "tighter, so a rejection at a smaller observed slack is HB doing its "
                       "job, not an inconsistency"),
        "safe_end": "always_fire (fire_rate = 1.0)",
        "certification_buys_nothing": bool(res.certified and res.lambda_chosen is not None
                                           and float(res.lambda_chosen) >= 1.0 - 1e-12),
        "certification_buys_nothing_note": ("certified=True with lambda_chosen at the safe "
                                            "end means only 'always fire' clears delta: the "
                                            "certificate is real and the saving is zero"),
        "note": "no always-valid fallback leg: the maximal-effort arm is not exact",
    }
    return res


# ---------------------------------------------------------------------------
# (B) CRC feasibility: three nested UCBs, the abstention floor
# ---------------------------------------------------------------------------

def ucb_hoeffding(rhat: float, n: int, delta: float = 0.1) -> float:
    """U_H = Rhat + sqrt(log(2/delta) / (2 |E_lambda|)).  2606.29054 2.1."""
    if n <= 0:
        return float("inf")
    return float(rhat + math.sqrt(math.log(2.0 / delta) / (2.0 * n)))


def ucb_empirical_bernstein(rhat: float, var: float, n: int, delta: float = 0.1) -> float:
    """U_B = Rhat + sqrt(2 sigmahat^2 log(2/delta) / n) + 7 log(2/delta) / (3 (n-1)).

    2606.29054 2.1 (empirical Bernstein UCB).  Requires n >= 2.
    """
    if n <= 1:
        return float("inf")
    lg = math.log(2.0 / delta)
    return float(rhat + math.sqrt(2.0 * max(0.0, var) * lg / n) + 7.0 * lg / (3.0 * (n - 1)))


def _kelly_kappa_grid(past: np.ndarray, alpha: float, clip: Tuple[float, float]) -> float:
    """Exact Kelly bet by grid search over the PAST losses only (O(|past|) per call)."""
    lo, hi = clip
    if past.size == 0:
        return float(lo)
    grid = np.linspace(lo, hi, 51)
    terms = 1.0 + grid[:, None] * (alpha - past)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = np.where(np.all(terms > 0.0, axis=1),
                        np.sum(np.log(np.maximum(terms, 1e-300)), axis=1), -np.inf)
    return float(grid[int(np.argmax(vals))])


def ecrc_wealth(risks: Sequence[float], alpha: float,
                kappa_clip: Tuple[float, float] = (0.0, 0.5),
                kelly: str = "paper") -> np.ndarray:
    """Testing-by-betting wealth W_j = W_{j-1} (1 + kappa_j (alpha - r_j)), W_0 = 1.

    2606.29054 2.1 and Appendix Alg. "e-CRC: Betting-Based Risk Control".  The
    process runs in the given row order, which the caller must fix (a
    session-grouped calibration order) -- reordering changes the wealth path
    though not its validity.

    ``kelly='paper'`` (DEFAULT) is the bet their pseudocode actually prints:

        mu_hat_j = (1/(j-1)) sum_{k<j} r_k       (predictable, defined for j >= 2)
        kappa_j  = clip( (alpha - mu_hat_j) / (alpha (1 - alpha)), 0, 0.5 )

    with ``kappa_1 = 0`` because mu_hat_1 is undefined there (their algorithm
    says "predictable, for j >= 2" and prints no j = 1 bet; betting nothing on
    the first row is the conservative reading, W_1 = W_0).

    ``kelly='approx'`` uses the second-order Kelly approximation
    ``kappa = clip(E[alpha - r] / E[(alpha - r)^2], 0, 0.5)`` and
    ``kelly='grid'`` the exact 51-point log-growth search over the past.  All
    three are PREDICTABLE, which is what Ville's inequality needs; only
    ``'paper'`` reproduces their published bet (DEVIATIONS #4).
    """
    if kelly not in ("paper", "approx", "grid"):
        raise ValueError("kelly must be 'paper' | 'approx' | 'grid'")
    r = np.asarray(risks, dtype=float)
    a = float(alpha)
    lo, hi = kappa_clip
    denom = a * (1.0 - a)
    w = 1.0
    out = np.empty(r.size, dtype=float)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    for j in range(r.size):
        if kelly == "grid":
            k = _kelly_kappa_grid(r[:j], a, kappa_clip)
        elif j == 0:
            k = float(lo)
        elif kelly == "paper":
            mu_hat = s0 / j                       # mean of r_1..r_{j-1}
            k = (float(lo) if denom <= 0.0
                 else float(min(max((a - mu_hat) / denom, lo), hi)))
        else:
            m1, m2 = s1 / j, s2 / j
            k = float(lo) if m2 <= 0.0 else float(min(max(m1 / m2, lo), hi))
        w = max(w * (1.0 + k * (a - r[j])), 0.0)
        out[j] = w
        d = a - r[j]
        s0 += float(r[j])
        s1 += d
        s2 += d * d
    return out


def ecrc_certified(risks: Sequence[float], alpha: float, delta: float = 0.1,
                   kappa_clip: Tuple[float, float] = (0.0, 0.5),
                   kelly: str = "paper") -> Tuple[bool, float]:
    """Certify H_0: E[R] >= alpha is rejected, i.e. W_m >= 1/delta (Ville).

    2606.29054 2.1.  Returns (certified, final wealth).
    """
    w = ecrc_wealth(risks, alpha, kappa_clip, kelly=kelly)
    final = float(w[-1]) if w.size else 1.0
    return bool(final >= 1.0 / float(delta)), final


def crc_select_lambda(
    scores: Sequence[float],
    risks: Sequence[float],
    *,
    alpha: float,
    delta: float = 0.1,
    bound: str = "bernstein",
    orientation: int = 1,
    min_emit: int = 10,
    kappa_clip: Tuple[float, float] = (0.0, 0.5),
    kelly: str = "paper",
) -> Dict[str, Any]:
    """Selective-prediction calibration: emit iff s >= lambda; lambda* = min{lambda : U(lambda) <= alpha}.

    2606.29054 2 (target ``E[R | s >= lambda*] <= alpha`` w.p. >= 1 - delta) and
    2.1 (the three UCBs).  Non-degeneracy ``|E_lambda| >= min_emit`` (their
    rule: >= 10, "none excluded post hoc").  delta defaults to their 0.1.

    ``orientation = +1`` means ``scores`` are RISK-oriented (our convention);
    they are converted once to their reliability convention here.  ``risks``
    are the per-row bounded losses in [0, 1] (label side).
    """
    if bound not in ("hoeffding", "bernstein", "ecrc"):
        raise ValueError("bound must be hoeffding | bernstein | ecrc")
    s = np.asarray(scores, dtype=float)
    r = np.asarray(risks, dtype=float)
    if s.size != r.size:
        raise ValueError("scores and risks length mismatch")
    rel = -s if orientation >= 0 else s.copy()

    cands = np.unique(rel[np.isfinite(rel)])
    rows: List[Dict[str, Any]] = []
    chosen: Optional[float] = None
    for lam in cands:                       # ascending: emit set shrinks
        mask = rel >= lam
        n_l = int(mask.sum())
        if n_l < min_emit:
            rows.append({"lambda": float(lam), "n_emit": n_l, "degenerate": True,
                         "ucb": None, "rhat": None})
            continue
        rr = r[mask]
        rhat = float(rr.mean())
        if bound == "hoeffding":
            u = ucb_hoeffding(rhat, n_l, delta)
            ok = u <= alpha
        elif bound == "bernstein":
            u = ucb_empirical_bernstein(rhat, float(rr.var(ddof=0)), n_l, delta)
            ok = u <= alpha
        else:
            ok, wealth = ecrc_certified(rr, alpha, delta, kappa_clip, kelly=kelly)
            u = float(wealth)
        # e-CRC certifies by wealth, not by a risk upper bound: the two must not
        # share a key name in the artifact or a reader compares W_m against alpha.
        rows.append({"lambda": float(lam), "n_emit": n_l, "degenerate": False,
                     "ucb": (None if bound == "ecrc" else float(u)),
                     "wealth": (float(u) if bound == "ecrc" else None),
                     "rhat": rhat, "certified": bool(ok)})
        if ok and chosen is None:
            chosen = float(lam)
    n = int(s.size)
    n_emit = int((rel >= chosen).sum()) if chosen is not None else 0
    emit_mask = (rel >= chosen) if chosen is not None else np.zeros(n, dtype=bool)
    return {
        "bound": bound,
        "alpha": float(alpha),
        "delta": float(delta),
        "min_emit": int(min_emit),
        "orientation": int(orientation),
        "lambda_star_reliability": chosen,
        "threshold_in_score_units": (None if chosen is None
                                     else (-chosen if orientation >= 0 else chosen)),
        "certified": chosen is not None,
        "n": n,
        "n_emit": n_emit,
        "abstention_rate": (1.0 - n_emit / n) if n else None,
        "risk_on_emitted": (float(r[emit_mask].mean()) if n_emit else None),
        "grid": rows,
    }


def bound_ordering_report(scores: Sequence[float], risks: Sequence[float], *,
                          alpha: float, delta: float = 0.1, orientation: int = 1,
                          min_emit: int = 10) -> Dict[str, Any]:
    """Report (never assert) 2606.29054 2.1 Prop. "Bound Ordering".

    Their proposition states ``Lambda*_Hoeff subset Lambda*_Bern subset
    Lambda*_e-CRC``.  Empirical Bernstein beats Hoeffding only in the
    low-variance regime the card names (sigma^2 < 1/4), so we compute the three
    certified sets on the data at hand and return whether the nesting holds
    here.
    """
    sets: Dict[str, List[float]] = {}
    for b in ("hoeffding", "bernstein", "ecrc"):
        res = crc_select_lambda(scores, risks, alpha=alpha, delta=delta, bound=b,
                                orientation=orientation, min_emit=min_emit)
        sets[b] = sorted({row["lambda"] for row in res["grid"]
                          if row.get("certified")})
    sh, sb, se = set(sets["hoeffding"]), set(sets["bernstein"]), set(sets["ecrc"])
    vacuous = not (sh or sb or se)
    return {
        "sizes": {k: len(v) for k, v in sets.items()},
        "hoeffding_subset_bernstein": sh.issubset(sb),
        "bernstein_subset_ecrc": sb.issubset(se),
        "nesting_holds": sh.issubset(sb) and sb.issubset(se),
        "vacuous": vacuous,
        "vacuous_note": ("all three certified sets are empty: the inclusion is "
                         "trivially true and says nothing" if vacuous else None),
        "lambda_sets": sets,
    }


def recommend_bound(n_calibration: int, variance: float) -> Dict[str, Any]:
    """Bound-selection rule, copied from 2606.29054's own Step 2 as the digest states it.

    Digest 4.11 / card migration step 4: never Hoeffding; empirical Bernstein
    by default when the loss variance is in their low-variance regime
    (sigma^2 < 0.1, where they certify 96% vs Hoeffding's 71%); e-CRC when the
    calibration split falls below ~200 rows (their data-efficiency result: at
    calibration ratio 0.2, e-CRC 10.0% vs Hoeffding 0.0%).
    """
    low_var = bool(variance < 0.1)
    small = bool(n_calibration < 200)
    choice = "ecrc" if small else "bernstein"
    return {
        "n_calibration": int(n_calibration),
        "variance": float(variance),
        "low_variance_regime": low_var,
        "small_calibration": small,
        "recommended": choice,
        "rule": "never Hoeffding; empirical Bernstein by default (sigma^2 < 0.1); "
                "e-CRC when calibration < ~200 rows",
        "note": (None if low_var else
                 "sigma^2 >= 0.1: outside the regime the card's rule is written for.  "
                 "The card gives no substitute rule for the high-variance case (their "
                 "high-variance subset certifies 1.0 / 1.5 / 1.7 % of configs), so "
                 "Bernstein stays the default and the choice is not evidence-backed here."),
    }


def abstention_lower_bound(mu: float, alpha: float) -> float:
    """Minimum abstention (for us: minimum FIRE rate) = max(0, (mu - alpha)/(1 - alpha)).

    2606.29054 2.2, "Minimum Abstention Lower Bound".  Holds for ANY
    distribution-free selective predictor "regardless of bound tightness, score
    quality, or calibration size".
    """
    mu = float(mu)
    alpha = float(alpha)
    if alpha >= 1.0:
        return 0.0
    return float(max(0.0, (mu - alpha) / (1.0 - alpha)))


def finite_sample_term(n: int, delta: float = 0.1) -> float:
    """The O(sqrt(log(1/delta)/n)) correction printed BESIDE the floor.

    2606.29054 2.2.  The constant inside the O() is in their proof appendix
    (unread, card open question 2), so this is the bare rate, not a bound: at
    our n it is larger than the floor itself for loose alpha, which is the point
    the card insists on making.
    """
    if n <= 0:
        return float("nan")
    return float(math.sqrt(math.log(1.0 / delta) / n))


def feasibility_table(mu: float, alphas: Sequence[float], n: int,
                      delta: float = 0.1) -> List[Dict[str, Any]]:
    """The floor table: one row per alpha with the floor and the finite-sample term.

    ``floor_exceeds_correction`` compares the floor against the BARE rate
    ``sqrt(log(1/delta)/n)``.  2606.29054 2.2 writes the correction as
    ``O(sqrt(log(1/delta)/n))`` and the constant lives in an unread proof
    appendix, so that boolean is NOT a separation claim; ``separation_claimable``
    is therefore None everywhere until the constant is read.
    """
    fs = finite_sample_term(n, delta)
    return [{"alpha": float(a),
             "min_fire_rate": abstention_lower_bound(mu, a),
             "finite_sample_term": fs,
             "finite_sample_term_is_bare_rate": True,
             "floor_exceeds_correction": bool(abstention_lower_bound(mu, a) > fs),
             "separation_claimable": None,
             "separation_note": ("the O() constant in 2606.29054 2.2 is unread, so a floor "
                                 "above the bare rate is not yet a separation; the card's "
                                 "conservative read is that only alpha <= 0.02 survives")}
            for a in alphas]


# ---------------------------------------------------------------------------
# (C) Clopper-Pearson recall gate
# ---------------------------------------------------------------------------

def cp_lower_one_sided(k: int, n: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson one-sided lower bound Beta^{-1}(alpha; k, n - k + 1).

    2607.06503 3.3.  Equals ``t34_common.clopper_pearson(k, n, 2*alpha)[0]``;
    the one-sided form is written out here because the paper's gate is
    one-sided and doubling alpha to reuse a two-sided helper would be an easy
    place to lose a factor of two.
    """
    from scipy.stats import beta
    if n <= 0 or k < 0 or k > n:
        return float("nan")
    if k == 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))


def cp_recall_threshold(success_scores: Sequence[float], t_r: float,
                        alpha: float = 0.05) -> Dict[str, Any]:
    """tau_r = min tau with Beta^{-1}(alpha; k(tau), n_r - k(tau) + 1) >= t_r.

    2607.06503 3.3.  ``success_scores`` = S_r, the FAILURE scores of the
    surviving SUCCESS episodes (for us: the C->C rows, whose score must stay
    below the gate); ``k(tau) = |{s in S_r : s <= tau}|`` counts the successes
    that survive a gate at tau.  Abort iff score > tau, so the score is
    risk-oriented (+1) exactly as our features are.

    ``t_r = 1`` closes the gate: the CP lower bound is ``alpha^{1/n} < 1`` for
    every finite n, so no finite tau qualifies and the returned threshold is
    +inf (fire on nothing), matching the paper's "t_r = 1 disables the gate".
    """
    s = np.sort(np.asarray(success_scores, dtype=float))
    n_r = int(s.size)
    if n_r == 0:
        return {"tau": float("inf"), "gate_open": False, "n_r": 0,
                "k": 0, "cp_lower": None, "t_r": float(t_r),
                "reason": "no surviving successes: gate abstains"}
    if float(t_r) >= 1.0:
        return {"tau": float("inf"), "gate_open": False, "n_r": n_r, "k": n_r,
                "cp_lower": cp_lower_one_sided(n_r, n_r, alpha), "t_r": float(t_r),
                "reason": "t_r = 1 closes the gate (CP lower bound < 1 for finite n)"}
    for tau in s:
        k = int((s <= tau).sum())
        if cp_lower_one_sided(k, n_r, alpha) >= float(t_r):
            return {"tau": float(tau), "gate_open": True, "n_r": n_r, "k": k,
                    "cp_lower": cp_lower_one_sided(k, n_r, alpha), "t_r": float(t_r),
                    "reason": "certified"}
    return {"tau": float("inf"), "gate_open": False, "n_r": n_r, "k": n_r,
            "cp_lower": cp_lower_one_sided(n_r, n_r, alpha), "t_r": float(t_r),
            "reason": "budget not supportable at this n_r: gate abstains"}


def sample_complexity_n_pos(rho_star: float, alpha_m: float = 0.05) -> int:
    """n_pos >= ln(alpha_m) / ln(rho*)  ->  the ceiling of that, as a count.

    2607.06503 3.4.1: even a cascade that aborts nothing is bounded below by
    ``Beta^{-1}(alpha_m; n_pos, 1) = alpha_m^{1/n_pos}`` (the rule of three),
    so no certification of a recall target above that is possible, "independent
    of scorer quality".
    """
    if not (0.0 < rho_star < 1.0) or not (0.0 < alpha_m < 1.0):
        return -1
    return int(math.ceil(math.log(alpha_m) / math.log(rho_star)))


def certifiable_recall_ceiling(n_pos: int, alpha_m: float = 0.05) -> float:
    """alpha_m^{1/n_pos}: the highest recall an n_pos-sized certification set can support.

    2607.06503 3.4.1 (the same rule read the other way round).
    """
    if n_pos <= 0:
        return float("nan")
    return float(alpha_m ** (1.0 / n_pos))


def assert_no_cascade_claim(n_gates: int, face: str) -> None:
    """Guard: the battery face is single-step, so R_g must be 1 there.

    2607.06503 5.2 reports the cascade beating a single gate by 1.5x-8.8x by
    DISTRIBUTING budget across rounds; on a single-step teacher-forced face
    that advantage is identically zero (card pitfall 2).  Any multi-gate run
    must be on the bench proxy face.
    """
    if face == "battery" and int(n_gates) != 1:
        raise ValueError(
            "battery face is single-step teacher-forced: R_g must be 1 and no "
            "cascade-vs-single-gate claim may be made there "
            "(2607.06503 card, pitfall 2)"
        )


def cascade_thresholds(gate_success_scores: Sequence[Sequence[float]],
                       budget_vector: Sequence[float],
                       alpha: float = 0.05,
                       face: str = "bench") -> List[Dict[str, Any]]:
    """Per-round CP thresholds for a budget vector t = (t_1 ... t_Rg).

    2607.06503 3.3 + 3.4.  Parameterised for the BENCH PROXY face, where rounds
    exist; ``face='battery'`` is rejected for R_g > 1 by
    :func:`assert_no_cascade_claim`.  Per-round guarantees do not compose
    (their 3.1) -- the global recall of the composed cascade must be measured
    on a disjoint validation split, which is what
    :func:`budget_feasibility_search` does.
    """
    assert_no_cascade_claim(len(gate_success_scores), face)
    if len(gate_success_scores) != len(budget_vector):
        raise ValueError("one budget per gate required")
    return [cp_recall_threshold(s, t, alpha)
            for s, t in zip(gate_success_scores, budget_vector)]


def budget_feasibility_search(
    candidates: Sequence[Sequence[float]],
    cal_success_scores: Sequence[Sequence[float]],
    val_apply: Callable[[Sequence[float]], Dict[str, Any]],
    *,
    rho_star: float,
    margin: float = 0.02,
    alpha: float = 0.05,
    face: str = "bench",
) -> Dict[str, Any]:
    """Margin-guarded budget search: deploy only if rhohat_val >= rho* + delta.

    2607.06503 3.4 (delta = 0.02 fixed a priori) with the search performed on a
    VALIDATION split disjoint from calibration.  ``val_apply(taus)`` must
    return at least ``{"recall": float, "savings": float}`` measured on that
    validation split; the deployed vector maximises ``savings`` among the
    feasible ones.  An empty feasible set is a legal outcome: the policy
    ABSTAINS, which on our data loses to the parse-failure baseline and trips
    kill line T1 -- pre-registered, per the digest.
    """
    feasible: List[Dict[str, Any]] = []
    for t in candidates:
        taus_info = cascade_thresholds(cal_success_scores, t, alpha=alpha, face=face)
        taus = [d["tau"] for d in taus_info]
        m = val_apply(taus)
        rec = float(m.get("recall", float("nan")))
        ok = bool(np.isfinite(rec) and rec >= float(rho_star) + float(margin))
        row = {"budget": list(map(float, t)), "taus": taus,
               "val_recall": rec, "val_savings": float(m.get("savings", 0.0)),
               "feasible": ok, "metrics": m}
        if ok:
            feasible.append(row)
    if not feasible:
        return {"abstain": True, "reason": "no margin-feasible budget vector",
                "rho_star": float(rho_star), "margin": float(margin),
                "n_candidates": len(candidates), "deployed": None}
    best = max(feasible, key=lambda r: r["val_savings"])
    return {"abstain": False, "deployed": best, "n_feasible": len(feasible),
            "rho_star": float(rho_star), "margin": float(margin),
            "n_candidates": len(candidates)}


def frozen_certificate(cert_success_scores: Sequence[float], taus: Sequence[float],
                       alpha_m: float = 0.05) -> Dict[str, Any]:
    """Post-selection certificate on an INDEPENDENT sample.

    2607.06503 3.4: "because the cascade is fixed before these data are
    observed, the bound is exact and distribution-free regardless of the
    preceding search size".  ``cert_success_scores`` are the must-not-fire rows
    of the certification split; a row survives iff its score is <= every tau.
    """
    s = np.asarray(cert_success_scores, dtype=float)
    n_pos = int(s.size)
    if n_pos == 0:
        return {"n_pos": 0, "survivors": 0, "recall_hat": None,
                "cp_lower": None, "ceiling": None}
    survive = np.ones(n_pos, dtype=bool)
    for tau in taus:
        survive &= (s <= float(tau))
    k = int(survive.sum())
    return {
        "n_pos": n_pos,
        "survivors": k,
        "recall_hat": k / n_pos,
        "cp_lower": cp_lower_one_sided(k, n_pos, alpha_m),
        "ceiling": certifiable_recall_ceiling(n_pos, alpha_m),
        "alpha_m": float(alpha_m),
    }


# ---------------------------------------------------------------------------
# (D) CRC primitive + non-exchangeable weighted CRC
# ---------------------------------------------------------------------------

def _crc_hat(lambdas: Sequence[float], losses: np.ndarray, weights: np.ndarray,
             alpha: float, B: float) -> Dict[str, Any]:
    lam = np.asarray(lambdas, dtype=float)
    if losses.ndim != 2 or losses.shape[1] != lam.size:
        raise ValueError("losses must be (n_units, n_lambdas)")
    if np.any(np.diff(lam) <= 0):
        raise ValueError("lambdas must be strictly ascending")
    dec = np.diff(losses, axis=1)
    monotone = bool(np.all(dec <= 1e-12))
    nw = float(weights.sum())
    if nw <= 0:
        raise ValueError("weights must sum to > 0")
    rhat = (weights[:, None] * losses).sum(axis=0) / nw
    lhs = (nw / (nw + 1.0)) * rhat + float(B) / (nw + 1.0)
    ok = np.where(lhs <= float(alpha))[0]
    idx = int(ok[0]) if ok.size else int(lam.size - 1)
    return {
        "lambda_hat": float(lam[idx]),
        "lambda_hat_index": idx,
        "set_empty": bool(ok.size == 0),
        "N_w": nw,
        "rhat": [float(x) for x in rhat],
        "lhs": [float(x) for x in lhs],
        "alpha": float(alpha),
        "B": float(B),
        "loss_monotone_nonincreasing": monotone,
    }


def crc_lambda_hat(lambdas: Sequence[float], losses: np.ndarray,
                   alpha: float, B: float = 1.0) -> Dict[str, Any]:
    """lambda_hat = inf{lambda : n/(n+1) Rhat_n(lambda) + B/(n+1) <= alpha}.

    2208.02814 eq. (4), Thm 1 (``E[L_{n+1}(lambda_hat)] <= alpha`` for
    exchangeable, non-increasing, bounded losses with
    ``L_i(lambda_max) <= alpha``).  When the set is empty the estimator returns
    ``lambda_max``, exactly as the paper defines it.

    Exchangeability across our steps is VIOLATED by construction (same-session
    steps are dependent), which is why (D)'s weighted variant exists and why no
    guarantee statement is attached to this estimator's output here.
    """
    L = np.asarray(losses, dtype=float)
    w = np.ones(L.shape[0], dtype=float)
    out = _crc_hat(lambdas, L, w, alpha, B)
    out["weighted"] = False
    return out


def weighted_crc_lambda_hat(lambdas: Sequence[float], losses: np.ndarray,
                            weights: Sequence[float], alpha: float,
                            B: float = 1.0) -> Dict[str, Any]:
    """Non-exchangeable CRC: N_w = sum w_i, Rhat_n = (1/N_w) sum w_i L_i(lambda),
    lambda_hat = inf{lambda : N_w/(N_w+1) Rhat_n + B/(N_w+1) <= alpha}.

    2310.01262 Thm 1 ("Non-exchangeable conformal risk control"), eq.
    (lambda_hat), read from the arXiv LaTeX source in this session (no transfer
    card exists for this id -- catalogue critic C.3).  With ``w_i = 1`` it
    reduces exactly to 2208.02814 eq. (4).

    NO finite-sample guarantee is claimed here.  Their bound is
    ``E[L(lambda_hat)] <= alpha + (B - A) * sum_i wtilde_i * d_TV(Z, Z^i)``
    with ``wtilde_i = w_i / (N_w + 1)``; the total-variation term between our
    battery cell and our bench cell is not estimable from anything on disk, so
    the only empirical statement this module makes about transport is the
    source-face / target-face violation PAIR from
    :func:`face_violation_report`.

    ``weights`` must be data-independent and in [0, 1] (their theorem statement
    says data-independent in bold); the rule that produced them is the caller's
    and must be written into the prereg.
    """
    L = np.asarray(losses, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.size != L.shape[0]:
        raise ValueError("one weight per calibration unit required")
    if np.any(w < 0.0) or np.any(w > 1.0):
        raise ValueError("weights must lie in [0, 1] (2310.01262 Thm 1)")
    out = _crc_hat(lambdas, L, w, alpha, B)
    out["weighted"] = True
    out["guarantee"] = ("alpha + (B-A) * sum_i wtilde_i * d_TV(Z, Z^i); the d_TV term is "
                        "NOT estimable here, so no numeric guarantee is claimed")
    return out


def similarity_weights(cal_covariates: Sequence[Any], target_covariate: Any,
                       *, rho: float = 0.9, distance: Optional[Callable[[Any, Any], float]] = None
                       ) -> np.ndarray:
    """One DOCUMENTED weight rule: w_i = rho ** d(covariate_i, target).

    2310.01262 "How to choose weights": weights should be large for calibration
    points whose swapped sequence stays close in distribution to the test
    sequence, and their maximum-entropy argument yields exponentially decaying
    weights ``wtilde_i ~ rho^{n+1-i}`` (they cite barber2022conformal for the
    same choice and use ``w_i = 0.99^{n+1-i}`` in their experiments).

    Our declared covariate is the CELL identity (face / checkpoint / caliber);
    the default distance is 0 for an exact cell match and 1 otherwise, so
    same-cell rows keep weight 1 and other-cell rows are down-weighted to rho.
    The rule -- and the covariate that defines similarity -- must be frozen in
    the prereg before the weights are computed.
    """
    if not (0.0 < rho <= 1.0):
        raise ValueError("rho must lie in (0, 1]")
    d = distance or (lambda a, b: 0.0 if a == b else 1.0)
    return np.array([float(rho) ** float(d(c, target_covariate)) for c in cal_covariates],
                    dtype=float)


def face_violation_report(threshold: float, source_losses: Sequence[float],
                          target_losses: Sequence[float], alpha: float) -> Dict[str, Any]:
    """The required pair: the SAME threshold's achieved violation rate on both faces.

    Digest 4.11 (Non-Exchangeable CRC entry): "the same threshold's achieved
    violation rate on the source face and on the target face -- without this
    pair the weighting has not been validated".  A "violation" is a face whose
    mean loss at this threshold exceeds alpha.
    """
    src = np.asarray(source_losses, dtype=float)
    tgt = np.asarray(target_losses, dtype=float)
    out = {"threshold": float(threshold), "alpha": float(alpha)}
    for name, arr in (("source", src), ("target", tgt)):
        m = float(arr.mean()) if arr.size else None
        out[name] = {"n": int(arr.size),
                     "mean_loss": m,                      # None, never NaN: JSON-safe
                     "violated": (None if m is None else bool(m > float(alpha))),
                     "per_row_violation_rate": (float((arr > float(alpha)).mean())
                                                if arr.size else None)}
    if out["target"]["n"] == 0:
        out["pair_incomplete"] = ("no target-face rows supplied: the weighting is NOT "
                                  "validated until both violation rates are reported")
    return out


def selective_loss_matrix(scores: Sequence[float], risks: Sequence[float],
                          lambdas: Sequence[float], orientation: int = 1) -> np.ndarray:
    """L_i(lambda) = r_i * 1{emit_i at lambda}, non-increasing in lambda.

    The bridge from a selective-prediction score to the monotone-loss shape the
    CRC primitive (2208.02814 Thm 1) requires.  NOTE the estimand: this
    controls the MARGINAL ``E[R * 1{emit}] <= alpha``, not the CONDITIONAL
    ``E[R | emit] <= alpha`` of 2606.29054 (DEVIATIONS #8) -- marginal control
    is trivially met by abstaining, so read the emit rate alongside it.

    ``lambdas`` are RELIABILITY thresholds (ascending); with ``orientation=+1``
    the input scores are risk-oriented and converted here.
    """
    s = np.asarray(scores, dtype=float)
    r = np.asarray(risks, dtype=float)
    rel = -s if orientation >= 0 else s
    lam = np.asarray(lambdas, dtype=float)
    emit = rel[:, None] >= lam[None, :]
    return r[:, None] * emit


# ---------------------------------------------------------------------------
# (E) stratified / group-conditional adaptive CRC
# ---------------------------------------------------------------------------

def derive_n_strata(n_units: int, rho_star: float, alpha_m: float = 0.05,
                    max_strata: int = 8, n_pos_units: Optional[int] = None) -> Dict[str, Any]:
    """Stratum count DERIVED from 2607.06503 3.4.1, never chosen post hoc.

    Digest 4.11 (Automatically Adaptive CRC entry): "the number of strata must
    be back-derived from ``n_pos >= ln alpha_m / ln rho*`` first, not tuned
    afterwards" -- 68 C->C rows cut into four strata leaves each stratum's
    Clopper-Pearson lower bound meaningless.

    ``n_pos`` in 2607.06503 3.4.1 counts SUCCESSFUL (must-not-fire) episodes,
    not all rows.  Pass ``n_pos_units`` with that count whenever it is known;
    the divisor is then the success count, which is the only number the rule is
    about.  Leaving it None divides the raw unit count instead, which OVERSTATES
    how finely the set may be cut whenever positives are present, so the CLI
    always supplies it.
    """
    need = sample_complexity_n_pos(rho_star, alpha_m)
    if need <= 0:
        return {"n_strata": 1, "n_pos_required": need, "reason": "invalid rho*/alpha_m"}
    basis = int(n_units if n_pos_units is None else n_pos_units)
    k = int(max(1, min(int(max_strata), basis // need)))
    return {"n_strata": k, "n_pos_required": need, "n_units": int(n_units),
            "n_pos_units": (None if n_pos_units is None else int(n_pos_units)),
            "derivation_basis": ("all units (n_pos count not supplied)" if n_pos_units is None
                                 else "success / must-not-fire units"),
            "rho_star": float(rho_star), "alpha_m": float(alpha_m),
            "reason": f"floor({basis} / {need}) capped at {max_strata}"}


def stratified_crc(
    strat_values: Sequence[float],
    lambdas: Sequence[float],
    losses: np.ndarray,
    *,
    alpha: float,
    rho_star: float = 0.95,
    alpha_m: float = 0.05,
    n_strata: Optional[int] = None,
    B: float = 1.0,
    max_strata: int = 8,
    stratifier_name: str = "prefix_scalar",
    success_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Group-conditional CRC over disjoint strata of one LABEL-FREE prefix scalar.

    2406.17819 2, example 2: with ``Phi`` a vector of group indicators the
    guarantee becomes ``E[loss | Phi(X) = j] <= alpha`` for every group j; for a
    PARTITION that specialises to running the CRC primitive inside each group,
    which is what this implements (the general vector-space / regulariser
    optimisation of their main theorem is not implemented -- DEVIATIONS #12).

    Strata are equal-count quantile bins of ``strat_values``; the count comes
    from :func:`derive_n_strata`.  A stratum with fewer than ``n_pos_required``
    units falls back to the GLOBAL threshold and is returned with
    ``conditional_guarantee=False`` -- that fallback is ours, not the paper's,
    and it voids the conditional statement for that stratum.

    ``success_mask`` marks the SUCCESS (must-not-fire) units.  2607.06503
    3.4.1's ``n_pos`` counts successes, so when the mask is given both the
    stratum count and the thin-stratum test count successes only; without it
    they count every row, which overstates how finely the set may be cut.

    Returns per-stratum rows AND pooled metrics.
    """
    v = np.asarray(strat_values, dtype=float)
    L = np.asarray(losses, dtype=float)
    if v.size != L.shape[0]:
        raise ValueError("strat_values and losses must agree on n_units")
    succ = None if success_mask is None else np.asarray(success_mask, dtype=bool)
    if succ is not None and succ.size != v.size:
        raise ValueError("success_mask must agree with strat_values on n_units")
    derived = derive_n_strata(v.size, rho_star, alpha_m, max_strata,
                              n_pos_units=(None if succ is None else int(succ.sum())))
    k = int(n_strata) if n_strata else int(derived["n_strata"])
    need = int(derived["n_pos_required"])

    glob = crc_lambda_hat(lambdas, L, alpha, B)
    if k <= 1:
        assign = np.zeros(v.size, dtype=int)
        edges: List[float] = []
    else:
        qs = np.quantile(v[np.isfinite(v)], np.linspace(0, 1, k + 1)[1:-1])
        edges = [float(x) for x in qs]
        assign = np.searchsorted(qs, v, side="right")

    rows: List[Dict[str, Any]] = []
    per_row_lambda = np.full(v.size, float(glob["lambda_hat"]), dtype=float)
    for j in range(max(1, k)):
        idx = np.where(assign == j)[0]
        if idx.size == 0:
            rows.append({"stratum": j, "n": 0, "n_success": (None if succ is None else 0),
                         "fallback": True,
                         "conditional_guarantee": False,
                         "lambda_hat": float(glob["lambda_hat"]),
                         "risk_at_lambda": None})
            continue
        n_basis = int(idx.size if succ is None else int(succ[idx].sum()))
        if n_basis < need:
            lam_hat = float(glob["lambda_hat"])
            fallback = True
            sub = None
        else:
            sub = crc_lambda_hat(lambdas, L[idx], alpha, B)
            lam_hat = float(sub["lambda_hat"])
            fallback = False
        per_row_lambda[idx] = lam_hat
        col = int(np.searchsorted(np.asarray(lambdas, dtype=float), lam_hat))
        col = min(col, L.shape[1] - 1)
        rows.append({
            "stratum": j, "n": int(idx.size),
            "n_success": (None if succ is None else int(succ[idx].sum())),
            "n_pos_basis": n_basis, "n_pos_required": need,
            "fallback": fallback,
            "conditional_guarantee": bool(not fallback),
            "lambda_hat": lam_hat,
            "risk_at_lambda": float(L[idx, col].mean()),
            "set_empty": (None if sub is None else bool(sub["set_empty"])),
            "range": [float(v[idx].min()), float(v[idx].max())],
        })

    cols = np.clip(np.searchsorted(np.asarray(lambdas, dtype=float), per_row_lambda),
                   0, L.shape[1] - 1)
    pooled_risk = float(L[np.arange(L.shape[0]), cols].mean())
    return {
        "stratifier": stratifier_name,
        "n_strata": max(1, k),
        "derived": derived,
        "edges": edges,
        "global_lambda_hat": float(glob["lambda_hat"]),
        "global_set_empty": bool(glob["set_empty"]),
        "strata": rows,
        "pooled_risk": pooled_risk,
        "risk_is_in_sample": True,
        "risk_note": ("risk_at_lambda and pooled_risk are CALIBRATION risks at the fitted "
                      "lambda_hat, evaluated on the same rows that fitted it; 2406.17819's "
                      "statement is about a fresh point, so these are diagnostics, not "
                      "validation.  A held-out check needs a disjoint split."),
        "pooled_alpha": float(alpha),
        "pooled_violated": bool(pooled_risk > float(alpha)),
        "n_fallback_strata": int(sum(1 for r in rows if r["fallback"])),
        "assignment": [int(a) for a in assign],
        "per_row_lambda": [float(x) for x in per_row_lambda],
    }


# ---------------------------------------------------------------------------
# label-side inputs (never features) and the one label-free score we compute
# ---------------------------------------------------------------------------

def battery_label_divergence_c2kv(frame: FrozenFrame, qids: Sequence[str]) -> np.ndarray:
    """LABEL SIDE.  D_i = 1 - tool_name_match(compressed arm) in {0, 1}.

    Reading B's loss in 2606.29054's feasibility algebra, and the bounded
    dissimilarity D of 2207.07061 Def. "textual consistency" evaluated at the
    never-fire end of our grid.  Reads ``tool_name_match``; it is a LABEL, it
    lives here behind a ``_label_`` name, and it never enters a feature frame.
    """
    by_qid = frame.c2kv_by_qid
    return np.array([0.0 if by_qid[q].get("tool_name_match") else 1.0 for q in qids],
                    dtype=float)


def battery_label_cw_loss(frame: FrozenFrame, qids: Sequence[str]) -> np.ndarray:
    """LABEL SIDE.  Reading A's loss: the C->W indicator (1 iff full-correct and
    compressed-wrong).  ``None`` labels (W->C / W->W) contribute 0.

    2606.29054 migration step 1, reading (i) -- the reading the trigger contract
    implies, pre-registered before mu is computed.
    """
    lab = frame.label_by_qid
    return np.array([1.0 if lab.get(q) == 1 else 0.0 for q in qids], dtype=float)


def parse_fail_score(frame: FrozenFrame, qids: Sequence[str]) -> np.ndarray:
    """The L1 parse-failure baseline as a 0/1 RISK score (+1 orientation).

    Digest 4.0 / kill line T1: the comparator every threshold machine here is
    measured against.  Reads only the compressed arm's own emitted text via
    ``t33_labels.parse_fail_baseline``; no target, no full arm.
    """
    by_qid = {r["qid"]: r for r in frame.labels}
    return np.array([1.0 if by_qid[q]["parse_fail_fire"] else 0.0 for q in qids],
                    dtype=float)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _frame_and_subset(root: str) -> Tuple[FrozenFrame, List[Dict[str, Any]]]:
    frame = FrozenAssets(Path(root)).load()
    return frame, frame.trigger_subset()


def _mu_ci(losses: np.ndarray, sessions: Sequence[str], reps: int = 2000) -> Dict[str, Any]:
    clusters = session_clusters(list(sessions))
    dummy = np.zeros(losses.size, dtype=int)
    lo, hi, n_cl = clustered_bootstrap(
        lambda s, y: float(np.mean(s)) if s.size else None,
        losses, dummy, clusters, reps=reps)
    return {"mu": float(losses.mean()), "ci95": [lo, hi], "n_clusters": n_cl,
            "n": int(losses.size), "var": float(losses.var(ddof=0))}


def cmd_feasibility(args: argparse.Namespace) -> Dict[str, Any]:
    frame, _ = _frame_and_subset(args.root)
    qids = [r["qid"] for r in frame.labels]
    sess = [r["session_id"] for r in frame.labels]
    alphas = list(args.alphas)

    reading_a = battery_label_cw_loss(frame, qids)
    reading_b = battery_label_divergence_c2kv(frame, qids)
    out: Dict[str, Any] = {
        "paper": "2606.29054 (feasibility floor); 2208.02814 (CRC primitive)",
        "n_rows": len(qids),
        "n_sessions": len(set(sess)),
        "base_rate_900_reference": BASE_RATE_900,
        "delta": args.delta,
        "readings": {},
    }
    for name, loss in (("A_cw_indicator", reading_a), ("B_one_minus_tool_name_match", reading_b)):
        ci = _mu_ci(loss, sess, reps=args.reps)
        out["readings"][name] = {
            **ci,
            "floor_table": feasibility_table(ci["mu"], alphas, len(qids), args.delta),
            "finite_sample_term_row_unit": finite_sample_term(len(qids), args.delta),
            "finite_sample_term_session_unit": finite_sample_term(ci["n_clusters"], args.delta),
            "finite_sample_unit_note": (
                "floor_table's finite_sample_term uses the ROW count, which is the number "
                "the digest names (n = 900 -> 0.0506).  Rows inside a session are dependent, "
                "so the honest unit is the session cluster; that term is printed beside it "
                "and is roughly twice as large, which only widens the region where the "
                "impossibility bound says nothing"),
            "bound_choice": recommend_bound(int(0.6 * len(qids)), ci["var"]),
        }

    # 60/40 session-grouped calibration/test split, scored with the L1 baseline.
    sub = frame.trigger_subset()
    sub_q = [r["qid"] for r in sub]
    sub_s = [r["session_id"] for r in sub]
    y = np.array([r["label_cw"] for r in sub], dtype=int)
    score_name = "parse_fail_fire (L1 baseline, orientation +1)"
    if getattr(args, "score_column", None):
        # a real-valued candidate score from another unit's features jsonl:
        # "<path>:<column>[:<orientation>]"; rows missing the value are dropped
        # and counted, never imputed.
        spec = str(args.score_column)
        ori = 1
        for suffix, val in ((":-1", -1), (":+1", 1), (":1", 1)):
            if spec.endswith(suffix):
                spec, ori = spec[: -len(suffix)], val
                break
        path, col = spec.rsplit(":", 1)      # rsplit: Windows drive letters contain ':'

        import json as _json
        by_qid = {}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = _json.loads(line)
                    by_qid[r["qid"]] = r.get(col)
        keep = [i for i, q in enumerate(sub_q) if by_qid.get(q) is not None]
        n_dropped = len(sub_q) - len(keep)
        sub_q = [sub_q[i] for i in keep]
        sub_s = [sub_s[i] for i in keep]
        y = y[keep]
        score = np.array([ori * float(by_qid[q]) for q in sub_q], dtype=float)
        score_name = f"{col} from {path} (orientation {ori:+d}; {n_dropped} rows without a value dropped)"
    else:
        score = parse_fail_score(frame, sub_q)
    risk = battery_label_cw_loss(frame, sub_q)
    groups = np.array(sub_s)
    folds = grouped_folds(groups, 5, seed=20260905)
    cal_mask = folds[0] | folds[1]                     # ~40 % held out as test
    test_mask = ~cal_mask
    # 60/40 as in 2606.29054 3: calibration is the LARGER part here
    cal_mask, test_mask = test_mask, cal_mask
    split_rows = []
    for bound in ("hoeffding", "bernstein", "ecrc"):
        sel = crc_select_lambda(score[cal_mask], risk[cal_mask], alpha=args.alphas[0],
                                delta=args.delta, bound=bound, orientation=1,
                                min_emit=args.min_emit)
        thr = sel["threshold_in_score_units"]
        if thr is None:
            split_rows.append({"bound": bound, "certified": False,
                               "held_out_violations": None, "n_emit_test": 0})
            continue
        emit = score[test_mask] <= thr
        r_test = risk[test_mask][emit]
        split_rows.append({
            "bound": bound, "certified": True, "threshold": thr,
            "n_emit_test": int(emit.sum()),
            "risk_on_emitted_test": (float(r_test.mean()) if r_test.size else None),
            "held_out_violations": (int(r_test.sum()) if r_test.size else 0),
            "violated_at_alpha": (bool(r_test.mean() > args.alphas[0]) if r_test.size else None),
            "abstention_rate_test": float(1.0 - emit.mean()) if emit.size else None,
        })
    out["calibration_split"] = {
        "score": score_name,
        "frame": "161-row trigger subset",
        "n_cal": int(cal_mask.sum()), "n_test": int(test_mask.sum()),
        "n_sessions_cal": int(len(set(groups[cal_mask]))),
        "n_sessions_test": int(len(set(groups[test_mask]))),
        "prevalence_subset": prevalence(y),
        "chance_ap_on_this_frame": prevalence(y),
        "mu_on_this_frame": float(risk.mean()),
        "alpha": args.alphas[0],
        "min_alpha_reachable_note": (
            "the trigger subset's own base risk is mu above; the abstention floor "
            "(mu - alpha)/(1 - alpha) applies HERE too, so an alpha far below mu "
            "cannot certify at any threshold.  Chance AP on this frame is the "
            "prevalence printed above, never the 900-frame 0.1033."),
        "rows": split_rows,
    }
    out["bound_ordering"] = bound_ordering_report(score[cal_mask], risk[cal_mask],
                                                  alpha=args.alphas[0], delta=args.delta,
                                                  orientation=1, min_emit=args.min_emit)
    return out


def cmd_sample_complexity(args: argparse.Namespace) -> Dict[str, Any]:
    frame, sub = _frame_and_subset(args.root)
    cc = frame.cc_qids()
    harm_path = Path(args.root) / "configs/bdf_pilot/d_harm_manifest_r2.json"
    harm_q: List[str] = []
    if harm_path.exists():
        harm_q = list(json.loads(harm_path.read_text(encoding="utf-8")).get("cw_qids") or [])
    units = {
        "cc_rows": len(cc),
        "cc_sessions": len({session_of(q) for q in cc}),
        "harm_rows": len(harm_q),
        "harm_sessions": len({session_of(q) for q in harm_q}),
    }
    table = []
    for rho in args.rho_star:
        need = sample_complexity_n_pos(rho, args.alpha_m)
        table.append({"rho_star": float(rho), "n_pos_required": need,
                      "supported_by": {k: bool(v >= need) for k, v in units.items()}})
    ceilings = {k: (certifiable_recall_ceiling(v, args.alpha_m) if v > 0 else None)
                for k, v in units.items()}
    return {
        "paper": "2607.06503 3.4.1 (n_pos >= ln alpha_m / ln rho*)",
        "alpha_m": args.alpha_m,
        "units": units,
        "certifiable_recall_ceiling": ceilings,
        "requirement_table": table,
        "note": "session-cluster units are the defensible ones; row units are printed "
                "for comparison only",
    }


def cmd_recall_gate(args: argparse.Namespace) -> Dict[str, Any]:
    frame, sub = _frame_and_subset(args.root)
    assert_no_cascade_claim(1, "battery")
    qids = [r["qid"] for r in sub]
    y = np.array([r["label_cw"] for r in sub], dtype=int)
    score = parse_fail_score(frame, qids)
    success_scores = score[y == 0]                       # C->C: must not fire
    gate = cp_recall_threshold(success_scores, args.rho_star, alpha=args.alpha)
    tau = gate["tau"]
    fire = score > tau if np.isfinite(tau) else np.zeros(score.size, dtype=bool)
    op = operating_point(np.where(fire, 1.0, 0.0), y, int(fire.sum()))

    # 2607.06503 3.4 requires the certificate to be computed on data the gate has
    # NOT seen.  Recomputing it on the calibration rows is post-selection and is
    # returned only as a flagged diagnostic; the defensible number comes from a
    # session-grouped split (sessions, never rows, so no session straddles it).
    cert_in_sample = frozen_certificate(success_scores, [tau], alpha_m=args.alpha_m)
    cert_in_sample.update({
        "in_sample": True,
        "valid_post_selection": False,
        "warning": "tau was chosen on these same rows: this bound is NOT the paper's "
                   "exact post-selection certificate (2607.06503 3.4)",
    })
    groups = np.array([r["session_id"] for r in sub])
    half = grouped_folds(groups, 2, seed=20260905)[0]
    cal_succ = score[(y == 0) & half]
    cert_succ = score[(y == 0) & ~half]
    split_gate = cp_recall_threshold(cal_succ, args.rho_star, alpha=args.alpha)
    cert_split = frozen_certificate(cert_succ, [split_gate["tau"]], alpha_m=args.alpha_m)
    cert_split.update({
        "in_sample": False,
        "valid_post_selection": True,
        "split": "session-grouped halves of the trigger subset (seed 20260905)",
        "n_sessions_cal": int(len(set(groups[half]))),
        "n_sessions_cert": int(len(set(groups[~half]))),
        "gate": split_gate,
        "rho_star": float(args.rho_star),
        "target_above_ceiling": (None if cert_split["ceiling"] is None
                                 else bool(float(args.rho_star) > cert_split["ceiling"])),
        "note": "halving n_pos lowers the certifiable ceiling alpha_m^(1/n_pos); that cost "
                "is the price of the paper's independence requirement, not a defect.  When "
                "target_above_ceiling is true the target is unattainable on this "
                "certification split regardless of scorer quality (2607.06503 3.4.1).",
    })
    return {
        "paper": "2607.06503 3.3/3.4/3.4.1",
        "face": "battery", "R_g": 1,
        "cascade_claim_forbidden": True,
        "score": "parse_fail_fire (L1 baseline, orientation +1)",
        "frame": {"n": len(qids), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
                  "n_sessions": len({r["session_id"] for r in sub})},
        "score_summary": {
            "n_success_scored_positive": int((success_scores > 0).sum()),
            "n_trigger_scored_positive": int((score[y == 1] > 0).sum()),
            "note": "a binary score can only fire on its whole positive block; when a "
                    "large share of the must-not-fire rows sit in that block the gate "
                    "abstains by construction, which is the pre-registered outcome "
                    "(digest 4.11: abstention loses to the parse-failure baseline and "
                    "trips kill line T1)",
        },
        "gate": gate,
        "operating_point": op,
        "precision": op["precision"],
        "precision_note": "2607.06503 reports no precision anywhere; this column is ours",
        "certificate_in_sample": cert_in_sample,
        "certificate_split": cert_split,
        "certificate_to_quote": "certificate_split",
        "margin_delta": args.margin,
    }


def cmd_calm(args: argparse.Namespace) -> Dict[str, Any]:
    frame, sub = _frame_and_subset(args.root)
    qids = [r["qid"] for r in sub]
    sess = [r["session_id"] for r in sub]
    score = parse_fail_score(frame, qids)
    nofire = battery_label_divergence_c2kv(frame, qids)

    assumption: Optional[float] = None
    if args.repair_outcomes:
        table = {}
        with open(args.repair_outcomes, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    table[rec["qid"]] = 1.0 - (1.0 if rec.get("correct") else 0.0)
        missing = [q for q in qids if q not in table]
        if missing:
            raise SystemExit(f"repair outcomes missing for {len(missing)} qids "
                             f"(first: {missing[:3]})")
        fire = np.array([table[q] for q in qids], dtype=float)
    elif args.assume_repair_success is not None:
        assumption = float(args.assume_repair_success)
        fire = np.full(len(qids), 1.0 - assumption, dtype=float)
    else:
        raise SystemExit("pass --repair-outcomes or --assume-repair-success")

    res = certify_fire_rate(score, nofire, fire, sess, delta=args.delta, eps=args.eps,
                            step=args.step, pvalue=args.pvalue, orientation=1,
                            unit=args.unit)
    d = res.to_dict()
    d["paper"] = ("2207.07061 4 (CALM calibration) + 2110.01052 2.2/2.3.1 "
                  "(LTT p-values + fixed sequence testing)")
    d["score"] = "parse_fail_fire (L1 baseline, orientation +1)"
    d["frame"] = {
        "name": "161-row trigger subset (C->W + C->C)",
        "n_rows": len(qids), "n_sessions": len(set(sess)),
        "n_pos": int(sum(1 for r in sub if r["label_cw"] == 1)),
        "estimand_note": ("the certified delta is E[D | the full arm was correct]: the "
                          "frame is selected on the FULL arm's label, so it is not the "
                          "900-row deployment population"),
    }
    d["assumed_repair_success"] = assumption
    d["assumption_warning"] = (None if assumption is None else
                               "ASSUMPTION, NOT A MEASUREMENT: loss_fire = 1 - "
                               f"{assumption} applied uniformly to every row")
    d["bonferroni_variant"] = bonferroni_reject([r["p"] for r in res.rows], args.eps)
    return d


def cmd_weighted_crc(args: argparse.Namespace) -> Dict[str, Any]:
    frame, sub = _frame_and_subset(args.root)
    qids = [r["qid"] for r in sub]
    score = parse_fail_score(frame, qids)
    risk = battery_label_cw_loss(frame, qids)
    lam = np.linspace(-1.0, 1.0, 41)
    L = selective_loss_matrix(score, risk, lam, orientation=1)

    tgt_scores: List[float] = []
    tgt_risks: List[float] = []
    if args.target_losses and Path(args.target_losses).exists():
        with open(args.target_losses, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    tgt_scores.append(float(rec["score"]))
                    tgt_risks.append(float(rec["risk"]))
    cells = ["battery"] * len(qids)
    w = similarity_weights(cells, args.target_cell, rho=args.rho)
    plain = crc_lambda_hat(lam, L, args.alpha)
    weighted = weighted_crc_lambda_hat(lam, L, w, args.alpha)

    col = int(min(np.searchsorted(lam, weighted["lambda_hat"]), L.shape[1] - 1))
    src_losses = L[:, col]
    if tgt_scores:
        Lt = selective_loss_matrix(tgt_scores, tgt_risks, lam, orientation=1)
        tgt_losses = Lt[:, col]
    else:
        tgt_losses = np.array([], dtype=float)
    return {
        "paper": "2310.01262 Thm 1 (weighted CRC); base 2208.02814 eq. (4)",
        "weight_rule": f"w_i = rho**d(cell_i, '{args.target_cell}'), rho = {args.rho}, "
                       "d = 0 on exact cell match else 1 (declared in the prereg)",
        "target_cell": args.target_cell,
        "plain": plain, "weighted": weighted,
        "face_violation_pair": face_violation_report(weighted["lambda_hat"], src_losses,
                                                     tgt_losses, args.alpha),
        "target_rows_available": len(tgt_scores),
        "weight_vector_degenerate": bool(len(set(np.asarray(w).tolist())) <= 1),
        "weight_note": ("only one cell is present on this frame, so every weight is equal: "
                        "the run exercises the B/(N_w+1) inflation of a uniform down-weight, "
                        "NOT differential transport.  No battery->bench transport claim may "
                        "be made until target-face rows exist"),
        "guarantee_language": "none claimed; the d_TV degradation term is not estimable here",
    }


def cmd_stratified_crc(args: argparse.Namespace) -> Dict[str, Any]:
    frame, sub = _frame_and_subset(args.root)
    qids = [r["qid"] for r in sub]
    c2kv = frame.c2kv_by_qid
    field_name = PREFIX_SCALARS.get(args.stratifier)
    if field_name is None:
        raise SystemExit(
            f"stratifier {args.stratifier!r} is not on the label-free prefix-scalar "
            f"allowlist; choose one of: {sorted(PREFIX_SCALARS)}  "
            "(dropped_docs lives only in the U2 sidecar, not on the battery row)")
    guard_columns([field_name], context="stratified_crc stratifier")
    if field_name not in c2kv[qids[0]]:
        raise SystemExit(f"stratifier {args.stratifier!r} (-> {field_name!r}) is not on "
                         "the frozen battery row")
    strat = np.array([float(c2kv[q][field_name]) for q in qids], dtype=float)
    score = parse_fail_score(frame, qids)
    risk = battery_label_cw_loss(frame, qids)
    y = np.array([r["label_cw"] for r in sub], dtype=int)
    lam = np.linspace(-1.0, 1.0, 41)
    L = selective_loss_matrix(score, risk, lam, orientation=1)
    # 2607.06503 3.4.1's n_pos counts the SUCCESS (must-not-fire) rows, which on
    # this frame are the 68 C->C rows, not all 161: passing the mask is what makes
    # the digest's "the calibration set cannot be cut" warning bind.
    res = stratified_crc(strat, lam, L, alpha=args.alpha, rho_star=args.rho_star,
                         alpha_m=args.alpha_m, stratifier_name=args.stratifier,
                         success_mask=(y == 0))
    res["paper"] = "2406.17819 2 example 2 (group-conditional CRC, disjoint-group case)"
    res["stratifier_field"] = field_name
    res["label_free"] = True

    # Digest 4.11, adaptive-CRC entry: "the three metrics reported PER STRATUM and
    # then pooled -- that is the only difference from the default scoring."  emit
    # iff -score >= lambda, so the FIRE set at a stratum's threshold is score > -lambda.
    assign = np.asarray(res["assignment"], dtype=int)
    prl = np.asarray(res["per_row_lambda"], dtype=float)
    fire = score > -prl
    per_stratum = []
    for j in range(int(res["n_strata"])):
        idx = np.where(assign == j)[0]
        if idx.size == 0:
            per_stratum.append({"stratum": j, "n": 0, "operating_point": None,
                                "prevalence": None})
            continue
        f = fire[idx]
        per_stratum.append({
            "stratum": j, "n": int(idx.size),
            "prevalence": prevalence(y[idx]),
            "chance_precision": prevalence(y[idx]),
            "operating_point": operating_point(np.where(f, 1.0, 0.0), y[idx], int(f.sum())),
        })
    res["per_stratum_metrics"] = per_stratum
    res["pooled_metrics"] = operating_point(np.where(fire, 1.0, 0.0), y, int(fire.sum()))
    res["pooled_prevalence"] = prevalence(y)
    res["metrics_are_in_sample"] = True
    res["metrics_note"] = ("coverage / precision / false_resets are measured on the same "
                           "rows that fitted lambda_hat; chance precision is the stratum's "
                           "own prevalence, never the 900-frame 0.1033")
    return res


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="t34_calibrate",
        description="4.11 thresholds and guarantees: CALM/LTT fixed sequence, CRC "
                    "feasibility + UCBs, Clopper-Pearson recall gate, weighted CRC, "
                    "stratified CRC.  All commands run locally; zero GPU.")
    p.add_argument("--out", default=None, help="write the result json here (frozen, sha printed)")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("feasibility", help="2606.29054 floor table + UCB certification")
    f.add_argument("--root", default=".", help="worktree root holding results/ and configs/")
    f.add_argument("--alphas", type=float, nargs="+", default=[0.10, 0.05, 0.02, 0.01])
    f.add_argument("--delta", type=float, default=0.1)
    f.add_argument("--reps", type=int, default=2000)
    f.add_argument("--min-emit", type=int, default=10, help="2606.29054 non-degeneracy |E_l| >= 10")
    f.add_argument("--score-column", default=None,
                   help="score a real-valued candidate instead of the binary parse-failure baseline: "
                        "<features.jsonl>:<column>[:<orientation +1|-1>]")
    f.set_defaults(fn=cmd_feasibility)

    s = sub.add_parser("sample-complexity", help="2607.06503 n_pos >= ln alpha_m / ln rho*")
    s.add_argument("--root", default=".")
    s.add_argument("--rho-star", type=float, nargs="+", default=[0.90, 0.95, 0.98, 0.99])
    s.add_argument("--alpha-m", type=float, default=0.05)
    s.set_defaults(fn=cmd_sample_complexity)

    g = sub.add_parser("recall-gate", help="2607.06503 single Clopper-Pearson gate (battery)")
    g.add_argument("--root", default=".")
    g.add_argument("--rho-star", type=float, default=0.95)
    g.add_argument("--alpha", type=float, default=0.05)
    g.add_argument("--alpha-m", type=float, default=0.05)
    g.add_argument("--margin", type=float, default=0.02)
    g.set_defaults(fn=cmd_recall_gate)

    c = sub.add_parser("calm", help="2207.07061 + 2110.01052 fixed-sequence certification")
    c.add_argument("--root", default=".")
    c.add_argument("--delta", type=float, default=0.20, help="tolerance in the consistency constraint")
    c.add_argument("--eps", type=float, default=0.05)
    c.add_argument("--step", type=float, default=0.05)
    c.add_argument("--pvalue", choices=["hoeffding", "hb"], default="hb")
    c.add_argument("--unit", choices=["session", "row"], default="session")
    c.add_argument("--repair-outcomes", default=None,
                   help="jsonl {qid, correct} from the D-line repair sweep")
    c.add_argument("--assume-repair-success", type=float, default=None,
                   help="EXPLICIT assumption echoed into the artifact; not a measurement")
    c.set_defaults(fn=cmd_calm)

    w = sub.add_parser("weighted-crc", help="2310.01262 weighted CRC + face violation pair")
    w.add_argument("--root", default=".")
    w.add_argument("--alpha", type=float, default=0.10)
    w.add_argument("--rho", type=float, default=0.9)
    w.add_argument("--target-cell", default="bench")
    w.add_argument("--target-losses", default=None,
                   help="jsonl {score, risk} rows of the TARGET face")
    w.set_defaults(fn=cmd_weighted_crc)

    t = sub.add_parser("stratified-crc", help="2406.17819 group-conditional CRC (disjoint groups)")
    t.add_argument("--root", default=".")
    t.add_argument("--stratifier", default="actual_compression_ratio")
    t.add_argument("--alpha", type=float, default=0.10)
    t.add_argument("--rho-star", type=float, default=0.95)
    t.add_argument("--alpha-m", type=float, default=0.05)
    t.set_defaults(fn=cmd_stratified_crc)
    return p


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with None so every artifact is STRICT
    json.  NaN/Infinity are not JSON and a downstream reader that accepts them
    silently is exactly how an undefined quantity turns into a number."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    return obj


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    result = json_safe(args.fn(args))
    result["deviations"] = DEVIATIONS
    text = json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True,
                      allow_nan=False)
    print(text)
    if args.out:
        sha = freeze_json(Path(args.out), result)
        print(f"# frozen: {args.out} sha256={sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
