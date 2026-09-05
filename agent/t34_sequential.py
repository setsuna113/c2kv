# -*- coding: utf-8 -*-
"""t34 unit U7b - survey 4.9 sequential family (bench face + battery session face).

Four migrations, all torch-free and zero-GPU:

  (A) Quickest Detection of Hallucination Onset (2606.12476) - the q/p persistence
      diagnostic that GATES the whole sequential family, the per-step false-fire
      companion column, and the learned CUSUM with ARL-aligned thresholds + delay.
  (B) Calibrated e-CUSUM Decoding (2607.11317) - the bounded alarm score
      a_t = min(1, 0.7 r_t + 0.3 u_t), the betting process E_n and its CUSUM-floored
      log form, mu_0 calibrated as the 90th percentile of per-token a_t pooled over
      healthy (C->C) traces, nominal delta vs ACHIEVED false-alarm rate.
  (C) CURA (2608.27808) - the Learn-then-Test certified threshold (calibration pool =
      C->C only, no failure labels), the exact-binomial alpha feasibility grid at our n,
      the success-only per-tool physiology Gaussian, and the prefix-only gate with
      a-priori fixed signs; length-only baseline at matched FPR + fold-internal floor.
  (D) ESN + per-channel-max CUSUM telemetry (2608.02464) - the memoryless
      Delta-Mahalanobis baseline FIRST, then a fixed-seed never-trained sparse
      reservoir per channel with a ridge readout fitted on HEALTHY episodes only,
      per-channel-max fusion, threshold from healthy VALIDATION episodes only.

WIRING (bench face).  Nothing here imports or edits ``benchmarks/*``.  Part D consumes
proxy request-log rows as plain dicts (``t34_common.load_proxy_log``), i.e. the rows
``proxy.py:1285 _log_request`` already writes: ``arm / conv_id / turn / fp / status /
error_kind / finish_reason / usage / wall_sec / gist_tokens / original_tokens / n_docs /
dropped_docs / doc_packing / repair_frame_delta`` plus the ``RecoverState.check``
(``proxy.py:494``) flags ``match / diverged_now / re_diverged / tracking_lost``.  To run
the monitor online the entry point is HOOK 2 - ``RecoverState.check`` (proxy.py:494) -
which would call :func:`EsnMonitor.step` with the same row dict just before
``_log_request``; the ``healthy`` episode set must come from bench TASK SUCCESS (never
from "compressed matched full", which is the oracle being removed).  Part B's serving
``u`` channel would be filled at the same hook from SGLang ``top_logprobs`` via
:func:`truncated_entropy` - it is a TRUNCATED entropy and is named as such everywhere.

RUNBOOK (execution order; every command runs HERE, zero GPU)
-------------------------------------------------------------
0. (server, owned by another unit) the capture rerun that produces
   ``<capture_dir>/c2kv/p0.steps.jsonl`` - only needed for the ``u`` channel of (B).
   Everything below runs without it and reports the u-channel columns as null.

1. GATE FIRST.  Nothing else in this module may be reported unless this licenses it::

     PYTHONIOENCODING=utf-8 python agent/t34_sequential.py qp \
        --root . --out results/t34/seq_qp_diagnostic.json

2. e-CUSUM calibration + evaluation (mu_0 from C->C, nominal vs achieved FA)::

     PYTHONIOENCODING=utf-8 python agent/t34_sequential.py ecusum \
        --root . [--capture_dir results/t33/capture] \
        --out results/t34/seq_ecusum.json

3. LTT feasibility grid (ten lines; it decides the shape of the whole CURA port)::

     PYTHONIOENCODING=utf-8 python agent/t34_sequential.py ltt-grid --n 68 \
        --out results/t34/seq_ltt_grid.json

4. CURA: LTT threshold + physiology + prefix gate + length-only matched-FPR control::

     PYTHONIOENCODING=utf-8 python agent/t34_sequential.py cura \
        --root . [--sidecar results/t34/sidecar_c2kv.jsonl] \
        --out results/t34/seq_cura.json

5. Per-qid feature frame for the shared winner table (once per arm; the ``full`` arm
   frame is the S0 twin control and is written to its own file).  The cura_* columns
   are OUT OF FOLD by default - the sessions are cut into three grouped folds and
   each row's physiology / prefix constants are fitted on the other two thirds, with
   ``cura_constants_out_of_fold`` stamped per row.  ``--in_sample_dump`` restores the
   old descriptive behaviour and must never feed a winner table::

     PYTHONIOENCODING=utf-8 python agent/t34_sequential.py features --root . --arm c2kv \
        --out results/t34/features_seq_c2kv.jsonl
     PYTHONIOENCODING=utf-8 python agent/t34_sequential.py features --root . --arm full \
        --out results/t34/features_seq_full.jsonl

6. ONLY IF step 1 licensed the family - learned CUSUM, ARL-aligned, with the
   shuffled-order control, the no-accumulation arm, and the delay column for the
   parse-failure baseline.  k and the ARL null pool come from a session-grouped
   CALIBRATION half; the arms are scored on the held-out sessions::

     PYTHONIOENCODING=utf-8 python agent/t34_sequential.py arl --root . \
        --gate results/t34/seq_qp_diagnostic.json \
        --out results/t34/seq_arl.json

7. Bench face (needs a proxy request log with >= 15 healthy episodes; not satisfiable on
   today's three bench faces - the command says so and aborts)::

     PYTHONIOENCODING=utf-8 python agent/t34_sequential.py esn \
        --proxy_log <path>.jsonl --healthy <conv_ids.json> \
        --out results/t34/seq_esn.json

Orientations: configs/t34/orientations_benchb.json (declared BEFORE any scoring).
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

from t34_common import (  # noqa: E402
    FrozenAssets,
    FrozenFrame,
    auroc,
    average_precision,
    clustered_bootstrap,
    freeze_json,
    grouped_folds,
    load_proxy_log,
    nested_cv_logistic,
    operating_point,
    per_step_false_fire_rate,
    prevalence,
    session_clusters,
    session_of,
    step_index,
    write_features_jsonl,
)

# --------------------------------------------------------------------------
# DEVIATIONS from the four papers (the prereg copies this list verbatim)
# --------------------------------------------------------------------------

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "qp_persistence_diagnostic",
        "paper": "2606.12476 sec.3.1 (tab:markov)",
        "what": "The unit of the label process is a DECISION STEP inside a battery "
                "session, not a generated token; ordering is step_index(qid) (the qid "
                "suffix), never the logged decision_step (= doc_chunks + 1).",
        "why": "Our frozen face has no per-token faithfulness labels; the pre-registered "
               "label C->W is defined once per decision step (t33_labels.cw_label).",
    },
    {
        "method": "qp_persistence_diagnostic",
        "paper": "2606.12476 sec.3.1",
        "what": "Steps whose label is undefined (W->C / W->W) BREAK the chain instead of "
                "being coerced to y=0; the number of adjacent pairs dropped that way is "
                "reported as its own denominator.",
        "why": "Coercing them to 0 would silently import the full arm's correctness into "
               "the transition table (label-side leakage into a diagnostic).",
    },
    {
        "method": "qp_persistence_diagnostic",
        "paper": "2606.12476 sec.3.1 (gate rule)",
        "what": "The pre-registered rule (CI of q_hat - p_hat contains 0 -> not licensed) "
                "is kept unchanged, but licensing additionally requires "
                "n_adjacent_pairs >= MIN_ADJACENT_PAIRS = 30 (declared before looking). "
                "The frozen battery SUBSAMPLES decision steps inside a session (qid "
                "suffixes such as 8/13/16/19), leaving only 15 genuinely adjacent "
                "labelled pairs, so the 2x2 table is not estimable on this face.",
        "why": "A gate that licenses the whole sequential family off 8 and 7 transitions "
               "would be a verdict-changing artefact; the extra clause is strictly "
               "CONSERVATIVE (it can only refuse, never license).",
    },
    {
        "method": "markov_order_ladder",
        "paper": "2606.12476 sec.3.1 (tab:markov, orders 1-4)",
        "what": "Orders 2-4 are fitted but their context counts are printed alongside; on "
                "2-4 step sessions orders 3 and 4 have almost no data and the ladder is "
                "reported as uninformative rather than as a model-selection result.",
        "why": "Their L ~= 120 tokens per generation; our sessions are 2-4 steps.",
    },
    {
        "method": "learned_cusum",
        "paper": "2606.12476 sec.3.3",
        "what": "logit(p_hat_t) is the LogisticRegression decision_function of a "
                "session-grouped nested-CV probe (out-of-fold), not a trained causal GRU; "
                "k = (mu_0+mu_1)/2 is the textbook value computed on the CALIBRATION pool, "
                "never fitted.",
        "why": "93 positives cannot train a recurrent labeler; the paper's own "
               "decomposition says the accumulator is worth -4.5 tokens and the score "
               "-12.9, so the cheap score is the honest arm.",
    },
    {
        "method": "threshold_for_arl",
        "paper": "2606.12476 sec.4.1 (ARL in {50,100,200})",
        "what": "ARL is estimated by i.i.d. resampling of NEGATIVE-pool increments, and "
                "the report carries a flag saying the requested ARL exceeds our maximum "
                "episode length (max 4 decision steps per session).",
        "why": "ln(gamma)/D does not exist at our horizon; the pre-registered pitfall is "
               "an output rather than something silently produced.",
    },
    {
        "method": "ecusum_alarm_score",
        "paper": "2607.11317 sec.3.1",
        "what": "w_rep=0.7 / w_ent=0.3 are kept exactly; when the u channel is absent the "
                "score is NOT back-filled with u=0 under the same name - a separately "
                "named a_rep_only = min(1, w_rep*r) column is emitted and the fused column "
                "is null.",
        "why": "A silent u=0 is a sentinel fallback under a fused name (t34 rule 3 / the "
               "parse-failure absorption pitfall).",
    },
    {
        "method": "ecusum_cross_step",
        "paper": "2607.11317 sec.3.2",
        "what": "n indexes DECISION STEPS in a session (the paper indexes tokens); r_n is "
                "the max over the causal per-token r_t stream of that step's own emitted "
                "text; the reset that breaks the supermartingale property is declared, as "
                "the paper declares it.",
        "why": "The digest's cross-step migration; the intra-step token-window version is "
               "the runner's t33 arm and is not duplicated here.",
    },
    {
        "method": "entropy_spike_relative",
        "paper": "2607.11317 sec.3.1 (u_t relative to recent H_t history)",
        "what": "The rolling history is the mean entropy of the PREVIOUS steps of the same "
                "session (window ECUSUM_ROLL_W = 3); the first step of a session has no "
                "history and gets None, never 0.  Absolute variants live in separate "
                "columns.",
        "why": "Battery rows have no t-1 inside a step; the session is the only rolling "
               "unit that exists on this face.",
    },
    {
        "method": "ecusum_calibrate_and_evaluate",
        "paper": "2607.11317 sec.3.3 + sec.7",
        "what": "The calibration/evaluation split IS strict here: C->C sessions are cut "
                "in half BY SESSION, mu_0 comes from the calibration half only, and the "
                "achieved false-alarm rate is measured on the held-out half.  'Healthy "
                "session' is the strict reading (>= 1 C->C row and NO C->W row).",
        "why": "The paper admits (sec.7) its own split was not strict; our false-reset "
               "metric is pre-registered and cannot be reported in-sample.",
    },
    {
        "method": "truncated_entropy",
        "paper": "2607.11317 sec.3.1",
        "what": "On the serving face only a top-k renormalised entropy is obtainable "
                "(--enable-custom-logit-processor is not in the launcher); the column is "
                "named truncated_entropy everywhere and is never called 'entropy'. The "
                "local dump shape [logprob, token_id] is VERIFIED against its writer "
                "(t33_capture.py:234); the serving shape is not, so an unrecognised row "
                "raises with the row printed instead of reading element 0 of whatever "
                "arrived. A --capture_dir without the dump raises CaptureUnavailable "
                "rather than degrading to a silent null u channel, and every command "
                "that can use the channel emits a u_channel_status block.",
        "why": "Card 2607.11317: 'do not quietly call it entropy'; and a u channel "
               "that is absent must be visibly absent, not a column of nulls whose "
               "cause the reader has to guess.",
    },
    {
        "method": "matched_fpr_comparison (length-only baseline)",
        "paper": "2608.27808 sec.5 C",
        "what": "Each arm's operating point is the quantile of ITS OWN negatives on a "
                "session-grouped CALIBRATION half and every reported rate is measured "
                "on the disjoint held-out half; the paper calibrates both alarms by "
                "LTT on a separate healthy pool. The LTT-calibrated length arm is NOT "
                "implemented: it needs the 4096-caliber rows, and at the 128 cap the "
                "length score is ~48% tied, which the result reports as matched=false "
                "with the tie mass beside it.",
        "why": "A threshold read from the very negatives the arm is then scored "
               "against is selection on the evaluation frame; an unmatched budget is "
               "reported as unmatched rather than silently compared.",
    },
    {
        "method": "ltt_select_threshold",
        "paper": "2608.27808 sec.4 (Learn-then-Test, fixed sequence, delta=0.05)",
        "what": "Implemented exactly (descending candidates, exact binomial H_j: "
                "FPR(theta_j) > alpha, stop at first non-rejection, return the last "
                "rejected).  The calibration pool is the C->C rows (n=68), not successful "
                "GUI trajectories, and the achievable-alpha grid is computed FIRST.",
        "why": "Our only failure-label-free healthy pool on the battery face is C->C.",
    },
    {
        "method": "physiology_fit",
        "paper": "2608.27808 sec.4 (per-action-type diagonal Gaussian on successes)",
        "what": "Keyed by the tool name the COMPRESSED arm itself emitted (parsed from its "
                "own text with t33_spanmap.parse_tool_call, never read from the "
                "prediction_tool_name scoring column); tools with fewer than "
                "PHYS_MIN_TOOL_ROWS calibration rows fall back to the POOLED Gaussian and "
                "are flagged by a tool_seen companion column; with no pooled model at all "
                "the value is None.",
        "why": "68 C->C rows spread over many tool names; a per-tool sigma from n=1 is "
               "undefined.  The fallback is a declared pooled estimator, not a sentinel.",
    },
    {
        "method": "cura_prefix_gate",
        "paper": "2608.27808 sec.4 (S_t = sum_j sign_j (s_j-mu_j)/sigma_j, W_t = max(0, W-0.5))",
        "what": "k = 0.5 and the a-priori sign vector are kept; the signal list is our "
                "prefix-only family (CURA_PREFIX_SIGNS).  hybrid_top_k / dropped_docs / "
                "n_docs / repair_frame_delta are null on the battery face and are dropped "
                "from S_t per row, with the participating-signal count reported.  The "
                "sidecar's `docs` / `doc_lengths` are the KEPT blocks only and "
                "`dropped_docs` indexes the post-split history list, so the n_docs signal "
                "is kept + dropped and is None when the dropped side is missing - never "
                "the visible count under the total's name.",
        "why": "Those fields exist only in the sidecar / proxy row; a missing signal must "
               "not enter the sum as a zero.",
    },
    {
        "method": "esn_observation_vector",
        "paper": "2608.02464 sec.3-4 (x_t = [e;u;m], d in {43,51,60})",
        "what": "No u channel at all (declared, per the paper's own +0.000 ablation) and no "
                "e channel; the channels are size / timing / protocol built from proxy row "
                "fields.  Per-channel-max fusion is the headline; mean fusion is computed "
                "only as the contrast the paper's ablation demands.",
        "why": "We have never emitted a logprob on the serving face, and the paper measures "
               "the surprisal channel at +0.000 AUROC.",
    },
    {
        "method": "esn_monitor",
        "paper": "2608.02464 sec.3 (theta from healthy validation episodes)",
        "what": "healthy = bench TASK SUCCESS only; the monitor aborts with a message when "
                "fewer than ESN_MIN_HEALTHY_EPISODES=15 healthy episodes are supplied, and "
                "rows whose error_kind is in ESN_EXCLUDED_ERROR_KINDS (C2KV_CACHE_MISS) "
                "form an explicit exclusion class counted in the output.",
        "why": "'compressed matched full' is the oracle we are removing; the gist-pool LRU "
               "eviction would otherwise turn the monitor into a cache-miss detector.",
    },
    {
        "method": "learned_cusum / cusum_reference_k (CLI arm `arl`)",
        "paper": "2606.12476 sec.3.3 + sec.4.1",
        "what": "k = (mu_0+mu_1)/2 and the ARL null pool are computed on a "
                "session-grouped CALIBRATION half and every arm is scored on the "
                "disjoint held-out sessions; the halving leaves ~34 calibration "
                "negatives, so the output carries n_calibration_negatives and the "
                "declared null_pool_underpowered flag (ARL_MIN_NULL_POOL = 20) "
                "instead of presenting the alignment as a matched operating point. "
                "A one-class calibration half aborts rather than borrowing the "
                "evaluation rows. The all-rows k is reported beside it under a name "
                "that says it is NOT used.",
        "why": "Reading k and the null quantile off the rows the arms are then "
               "scored on is threshold selection on the evaluation frame; the "
               "coarse-resolution consequence of splitting is a reported flag, not "
               "a reason to keep the in-sample number.",
    },
    {
        "method": "ecusum_calibrate_and_evaluate (mu_0 pooling unit)",
        "paper": "2607.11317 sec.3.3",
        "what": "mu_0 actually used is the 90th percentile of the per-STEP a_n over "
                "the calibration-half C->C sessions, i.e. the unit of the statistic "
                "that is accumulated. The paper's per-TOKEN pooling is reported as a "
                "companion (calibration_paper_unit_companion) and is NOT used.",
        "why": "Our n indexes decision steps and a_n is the max of the step's causal "
               "per-token r_t stream; a per-token 90th percentile is not an upper "
               "bound on E[a_n | F_{n-1}] and would mis-scale the betting increment.",
    },
    {
        "method": "ltt_select_threshold (candidate sequence)",
        "paper": "2608.27808 sec.4",
        "what": "The descending fixed sequence gets one rung just above the observed "
                "calibration maximum so that k = 0 false alarms is expressible; the "
                "result flags theta_above_all_calibration_scores when that rung wins.",
        "why": "At n = 68 alpha = 0.05 admits at most 0 false alarms, so a grid taken "
               "from the observed scores alone could never certify it for an "
               "arithmetic reason rather than a data reason.",
    },
    {
        "method": "esn_channel_fit (sigma_err, ensemble, leak)",
        "paper": "2608.02464 sec.3 (eq.2 reservoir, eq.4 surprise)",
        "what": "sigma_err (eq.4) and the (q_mean, q_std) the CUSUM z-scores with are "
                "measured on a HELD-OUT healthy split (sigma_frac = 0.2 of the healthy "
                "episodes, disjoint from both the readout fit split and the threshold "
                "validation split), as the paper does; if that split ends up empty the "
                "fit residuals are used and the channel reports sigma_in_sample = True "
                "/ sigma_err_source.held_out = False. The reservoir ensemble is K = 1 "
                "(the paper averages q over K reservoirs); the leak rate of eq.2 is "
                "declared at alpha = 1.0 (the paper writes the leaky form but fixes no "
                "value).",
        "why": "Residuals measured on the rows the readout was solved on are "
               "optimistically small and inflate every q_t on unseen episodes; K and "
               "alpha are reported rather than tuned, and both are exposed as "
               "constructor arguments.",
    },
    {
        "method": "esn_monitor (false-alarm denominator)",
        "paper": "2608.02464 sec.3 (theta from healthy validation, FPR on unseen)",
        "what": "False positives are reported as raw counts on each of the three "
                "healthy splits - FIT (in-sample for the readout), VALIDATION (the "
                "threshold's own source) and SIGMA (unseen by both, but the pool "
                "sigma_err and the q normalisation were measured on) - each labelled "
                "with what it is in-sample for. None of the three is a clean held-out "
                "false-alarm rate.",
        "why": "Reporting one pooled count over fit+val under the paper's name would "
                "read as their 0/63 held-out figure when it is not.",
    },
    {
        "method": "cura_fold_internal_floor / _cmd_cura",
        "paper": "2608.27808 sec.4 + sec.5 (three-way split)",
        "what": "The physiology Gaussian and the prefix (mu_j, sigma_j) are fitted on "
                "a DISJOINT fit half of the C->C sessions before LTT calibration, and "
                "refitted inside every fold of the floor and inside every replicate of "
                "the three-way split (ltt_three_way(score_fn=...)). The `features` "
                "subcommand now emits OUT-OF-FOLD columns too: the sessions are cut "
                "into FEATURES_OOF_FOLDS = 3 grouped folds and each row's cura_* "
                "columns come from constants fitted on the other folds' C->C rows, "
                "stamped per row by cura_constants_out_of_fold. The in-sample dump "
                "survives only behind --in_sample_dump, which stamps 0.0 on every "
                "C->C row and prints the warning. The matched-FPR contrast runs on "
                "the rows OUTSIDE the fit half only.",
        "why": "Fitting the constants on every C->C row and then calibrating or "
                "scoring on those same rows is the leakage the fold-internal floor "
                "exists to exclude.",
    },
]

# --------------------------------------------------------------------------
# constants the papers fix, and the ones we fix a priori
# --------------------------------------------------------------------------

#: 2607.11317 sec.3.1 - hand-set, not fitted.
ECUSUM_W_REP = 0.7
ECUSUM_W_ENT = 0.3
#: 2607.11317 sec.3.3 - mu_0 is the 90th percentile of the pooled healthy a_t.
ECUSUM_MU0_QUANTILE = 0.90
#: declared betting fraction (the paper leaves lambda a hyper-parameter).
ECUSUM_LAMBDA = 0.5
#: declared nominal Ville level; the ACHIEVED rate is always reported beside it.
ECUSUM_DELTA_NOMINAL = 0.05
#: rolling window (in previous decision steps of the same session) for u_t.
ECUSUM_ROLL_W = 3
#: n-gram order and sliding window (in tokens) for r_t.
ECUSUM_NGRAM = 3
ECUSUM_WINDOW = 64
#: absolute-entropy variant scale (nats); used only by the *_abs columns.
ENTROPY_ABS_SCALE_NATS = 2.0

#: 2608.27808 sec.4 - CUSUM reference value, fixed by the paper.
CURA_K = 0.5
CURA_DELTA = 0.05
PHYS_VITALS = ("generate_sec", "ttft_sec", "tbt_sec", "generated_tokens", "prompt_tokens")
PHYS_MIN_TOOL_ROWS = 3

#: Signs fixed A PRIORI (2608.27808 sec.4: "all sign_j were fixed a priori").
#: +1 = larger value is riskier for the compressed arm.
CURA_PREFIX_SIGNS: Dict[str, int] = {
    "gist_tokens": -1,               # more gist budget = more retained history
    "actual_compression_ratio": +1,  # harder compression = riskier
    "n_docs": +1,                    # more history blocks that can be damaged
    "doc_chunks": +1,
    "kept_history_tokens": +1,       # more raw history squeezed into gists
    "hybrid_top_k": -1,              # more raw blocks kept uncompressed
    "max_doc_num_saturated": +1,     # the tail window dropped blocks
    "dropped_docs": +1,
    "repair_frame_delta": +1,        # position-ledger gap
}
#: the tail-window caliber the D line ran at (saturation-flag reference).
MAX_DOC_NUM_CALIBER = 16

#: 2608.02464 - data requirement and exclusion class.
ESN_MIN_HEALTHY_EPISODES = 15
ESN_EXCLUDED_ERROR_KINDS = ("C2KV_CACHE_MISS",)
ESN_CPU_COST_NOTE = (
    "2608.02464 reports ~200 us/step monitor + 674 us/step telemetry on a CPU host. "
    "These are NOT to be entered into the NPU cost table (borrowed-host rule)."
)

CAP_CONFOUND_NOTE = (
    "The r2 trigger set was generated at the 128-token cap "
    "(kv_recipe.max_new_tokens=128), so any length / termination / repetition column is "
    "partly reading 'hit the cap'. Every length-bearing row below must also be run at the "
    "4096 caliber before it is reported as a behavioural signal."
)

ARL_TARGETS = (50, 100, 200)

#: Estimability precondition on the q/p gate (OUR addition, see DEVIATIONS).
#: The frozen battery samples a handful of decision steps per session, so the number
#: of genuinely adjacent labelled pairs is tiny; a transition table built from single
#: digits of transitions cannot license a family.  Declared before looking.
MIN_ADJACENT_PAIRS = 30

#: Estimability precondition on the ARL alignment (OUR addition, see DEVIATIONS).
#: k = (mu_0+mu_1)/2 and the ARL null pool are read from a session-grouped
#: CALIBRATION half; below this many calibration negatives the bisected threshold is
#: coarser than the ARL targets it is supposed to hit, and the flag says so.
ARL_MIN_NULL_POOL = 20


# ==========================================================================
# (A) Quickest Detection of Hallucination Onset - 2606.12476
# ==========================================================================

def label_sequences(frame: FrozenFrame) -> Dict[str, List[Tuple[int, Optional[int]]]]:
    """Session -> [(step_index, label_cw)] ordered by ``step_index(qid)``.

    2606.12476 sec.3.1 groups by generation and orders by token index; our unit is a
    decision step and the ordering key is the qid suffix (NOT ``decision_step``,
    which is ``doc_chunks + 1``).
    """
    seqs: Dict[str, List[Tuple[int, Optional[int]]]] = {}
    for rec in frame.labels:
        seqs.setdefault(rec["session_id"], []).append(
            (step_index(rec["qid"]), rec["label_cw"]))
    for sid in seqs:
        seqs[sid].sort(key=lambda t: t[0])
    return seqs


def adjacent_pairs(seqs: Dict[str, List[Tuple[int, Optional[int]]]],
                   *, max_gap: Optional[int] = 1
                   ) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[str, Any]]:
    """Adjacent (y_{t-1}, y_t) pairs per session (2606.12476 sec.3.1).

    A pair is counted only when both steps carry a defined label AND their step
    indices differ by at most ``max_gap`` (1 = the paper's estimand).  Pairs
    dropped because a step is undefined (W->C / W->W) or because the session
    skips steps are counted separately - see the DEVIATIONS entry: coercing
    undefined steps to y=0 would import the full arm's correctness into the
    diagnostic.  ``max_gap=None`` accepts any gap and is the NAIVE variant, kept
    only so the frozen battery's within-session subsampling is visible.
    """
    pairs: Dict[str, List[Tuple[int, int]]] = {}
    stats: Dict[str, Any] = {"n_pairs": 0, "dropped_undefined": 0,
                             "dropped_nonadjacent": 0, "max_gap": max_gap,
                             "gap_hist": {}}
    for sid, items in seqs.items():
        out: List[Tuple[int, int]] = []
        for (i0, y0), (i1, y1) in zip(items, items[1:]):
            gap = i1 - i0
            stats["gap_hist"][str(gap)] = stats["gap_hist"].get(str(gap), 0) + 1
            if max_gap is not None and gap > max_gap:
                stats["dropped_nonadjacent"] += 1
                continue
            if y0 is None or y1 is None:
                stats["dropped_undefined"] += 1
                continue
            out.append((int(y0), int(y1)))
        pairs[sid] = out
        stats["n_pairs"] += len(out)
    return pairs, stats


def transition_pq(pairs: Dict[str, List[Tuple[int, int]]]) -> Dict[str, Any]:
    """Row-normalised 2x2 transition table (2606.12476 sec.3.1, Assumption 1).

    ``p = P(y_t=1 | y_{t-1}=0)``, ``q = P(y_t=1 | y_{t-1}=1)``.
    """
    n = np.zeros((2, 2), dtype=float)
    for lst in pairs.values():
        for y0, y1 in lst:
            n[y0, y1] += 1.0
    row0, row1 = n[0].sum(), n[1].sum()
    p = float(n[0, 1] / row0) if row0 else float("nan")
    q = float(n[1, 1] / row1) if row1 else float("nan")
    return {
        "counts": {"n00": n[0, 0], "n01": n[0, 1], "n10": n[1, 0], "n11": n[1, 1]},
        "p_hat": p,
        "q_hat": q,
        "q_minus_p": (q - p) if (row0 and row1) else float("nan"),
        "q_over_p": (q / p) if (row0 and row1 and p > 0) else None,
        "n_from_0": int(row0),
        "n_from_1": int(row1),
    }


def markov_order_ladder(seqs: Dict[str, List[Tuple[int, Optional[int]]]],
                        max_order: int = 4) -> List[Dict[str, Any]]:
    """Log-likelihood ladder for Markov orders 1..K (2606.12476 sec.3.1, tab:markov).

    Maximum-likelihood conditional tables from counts, evaluated on the same data
    (as the paper does).  ``n_contexts_seen`` is printed beside every rung: on 2-4
    step sessions orders 3-4 see almost no context and the rung is uninformative.
    """
    chains: List[List[int]] = []
    for items in seqs.values():
        run: List[int] = []
        prev_idx: Optional[int] = None
        for idx, y in items:
            if y is None or (prev_idx is not None and idx - prev_idx != 1):
                if len(run) > 1:
                    chains.append(run)
                run = [] if y is None else [int(y)]
            else:
                run.append(int(y))
            prev_idx = idx
        if len(run) > 1:
            chains.append(run)

    ladder: List[Dict[str, Any]] = []
    prev_ll: Optional[float] = None
    for k in range(1, max_order + 1):
        counts: Dict[Tuple[int, ...], List[float]] = {}
        for run in chains:
            for t in range(k, len(run)):
                ctx = tuple(run[t - k:t])
                counts.setdefault(ctx, [0.0, 0.0])[run[t]] += 1.0
        ll = 0.0
        n_terms = 0
        for ctx, (c0, c1) in counts.items():
            tot = c0 + c1
            for c in (c0, c1):
                if c > 0:
                    ll += c * math.log(c / tot)
            n_terms += int(tot)
        rung = {
            "order": k,
            "loglik": ll,
            "params": 2 ** k,
            "n_terms": n_terms,
            "n_contexts_seen": len(counts),
            "delta_pct_vs_prev": (abs(ll - prev_ll) / abs(prev_ll) * 100.0
                                  if prev_ll not in (None, 0.0) else None),
        }
        ladder.append(rung)
        prev_ll = ll
    return ladder


def _bootstrap_sessions(pairs: Dict[str, List[Tuple[int, int]]],
                        stat: Callable[[Dict[str, List[Tuple[int, int]]]], Optional[float]],
                        *, reps: int = 2000, seed: int = 20260905,
                        alpha: float = 0.05) -> Tuple[Optional[float], Optional[float], int]:
    """Session-clustered percentile bootstrap of a transition-table statistic."""
    rng = np.random.default_rng(seed)
    sids = sorted(pairs)
    vals: List[float] = []
    for _ in range(reps):
        pick = rng.choice(len(sids), size=len(sids), replace=True)
        boot = {f"b{i}": pairs[sids[j]] for i, j in enumerate(pick)}
        v = stat(boot)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return None, None, len(sids)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), len(sids)


def steps_per_session(seqs: Dict[str, List[Tuple[int, Optional[int]]]]) -> Dict[str, Any]:
    v = np.array([len(x) for x in seqs.values()], dtype=float)
    if v.size == 0:
        return {"n_sessions": 0}
    return {
        "n_sessions": int(v.size),
        "mean": float(v.mean()),
        "min": int(v.min()),
        "max": int(v.max()),
        "hist": {str(int(k)): int(c) for k, c in
                 zip(*np.unique(v.astype(int), return_counts=True))},
    }


def qp_persistence_diagnostic(frame: FrozenFrame, *, reps: int = 2000,
                              seed: int = 20260905) -> Dict[str, Any]:
    """The step-0 gate of 2606.12476 (sec.3.1 + the card's Migration step 0).

    Pre-registered verdict: if the session-clustered bootstrap CI of ``q_hat - p_hat``
    contains 0, the returned ``sequential_licensed`` is False and the caller MUST NOT
    run the rest of the family (see :func:`require_gate`).
    """
    seqs = label_sequences(frame)
    pairs, pair_stats = adjacent_pairs(seqs)
    table = transition_pq(pairs)

    def _p(pp): return transition_pq(pp)["p_hat"]

    def _q(pp): return transition_pq(pp)["q_hat"]

    def _d(pp):
        t = transition_pq(pp)
        return t["q_minus_p"]

    p_lo, p_hi, n_clusters = _bootstrap_sessions(pairs, _p, reps=reps, seed=seed)
    q_lo, q_hi, _ = _bootstrap_sessions(pairs, _q, reps=reps, seed=seed + 1)
    d_lo, d_hi, _ = _bootstrap_sessions(pairs, _d, reps=reps, seed=seed + 2)

    ci_excludes_zero = bool(d_lo is not None and d_hi is not None and d_lo > 0.0)
    estimable = bool(pair_stats["n_pairs"] >= MIN_ADJACENT_PAIRS)
    licensed = bool(ci_excludes_zero and estimable)
    if not estimable:
        verdict = (
            f"NOT ESTIMABLE: only {pair_stats['n_pairs']} genuinely adjacent "
            f"(gap = 1) labelled pairs exist (< MIN_ADJACENT_PAIRS="
            f"{MIN_ADJACENT_PAIRS}); the frozen battery SUBSAMPLES decision steps "
            "inside a session, so the adjacent-pair transition table has no usable "
            "denominator on this face. Sequential family not opened."
        )
    elif ci_excludes_zero:
        verdict = ("q_hat - p_hat CI excludes 0: a persistent latent damaged state is "
                   "detectable at this n; the sequential family is licensed.")
    else:
        verdict = ("no persistent latent damaged state detectable at this n; "
                   "sequential family not opened (2606.12476 migration step 0, "
                   "pre-registered).")

    # NAIVE variant, reported only so the subsampling above is visible: it treats
    # consecutive BATTERY ROWS as a transition regardless of the step gap, which is
    # NOT the paper's estimand.
    naive_pairs, naive_stats = adjacent_pairs(seqs, max_gap=None)
    naive_table = transition_pq(naive_pairs)
    n_lo, n_hi, _ = _bootstrap_sessions(naive_pairs, lambda pp: transition_pq(pp)["q_minus_p"],
                                        reps=reps, seed=seed + 3)

    return {
        "rule_version": "d_cw_v1",
        "ordering_key": "step_index(qid)",
        "transition": table,
        "pair_stats": pair_stats,
        "ci95": {"p_hat": [p_lo, p_hi], "q_hat": [q_lo, q_hi], "q_minus_p": [d_lo, d_hi]},
        "n_session_clusters": n_clusters,
        "bootstrap_reps": reps,
        "markov_ladder": markov_order_ladder(seqs),
        "steps_per_session": steps_per_session(seqs),
        "ci_excludes_zero": ci_excludes_zero,
        "estimable": estimable,
        "min_adjacent_pairs": MIN_ADJACENT_PAIRS,
        "sequential_licensed": licensed,
        "verdict": verdict,
        "variant_consecutive_in_frame_NOT_THE_ESTIMAND": {
            "transition": naive_table,
            "pair_stats": naive_stats,
            "ci95_q_minus_p": [n_lo, n_hi],
            "warning": ("consecutive battery rows in a session are typically 3-10 "
                        "decision steps apart; this row is diagnostic of the "
                        "subsampling, not an estimate of P(y_t | y_{t-1})."),
        },
        "notes": [
            "teacher-forced single-step battery: the compressed arm's own errors cannot "
            "propagate, so this face measures a FLOOR on persistence.",
            "on fixed_joint W->C (120) > C->W (93): prima facie evidence against a "
            "one-sided change-point model on this face.",
            "the frozen battery samples a few decision steps per session (qid suffixes "
            "such as 8/13/16/19), so adjacency in the SESSION is not adjacency in the "
            "FRAME; the sequential family needs new rows, not a re-read of these.",
            CAP_CONFOUND_NOTE,
        ],
    }


class SequentialGateError(RuntimeError):
    """Raised when a sequential arm is requested but the q/p gate did not license it."""


def require_gate(diagnostic: Dict[str, Any], *, force: bool = False) -> None:
    """Honour the pre-registered gate of 2606.12476 (migration step 0).

    ``force`` exists only so a unit test can exercise the downstream code; using it
    in a report is a pre-registration violation and the caller must say so.
    """
    if diagnostic.get("sequential_licensed"):
        return
    if force:
        return
    raise SequentialGateError(diagnostic.get("verdict", "sequential family not licensed"))


# ---- learned CUSUM (2606.12476 sec.3.3) ----------------------------------

def cusum_reference_k(increments: Sequence[float], y: Sequence[int]) -> float:
    """k = (mu_0 + mu_1)/2, the textbook CUSUM reference value (2606.12476 sec.3.3).

    ``mu_0`` / ``mu_1`` are the mean log-odds under negatives / positives of the
    CALIBRATION pool.  Explicitly not fitted and independent of calibration of the
    posterior.
    """
    s = np.asarray(increments, dtype=float)
    yy = np.asarray(y, dtype=int)
    m0 = float(s[yy == 0].mean()) if (yy == 0).any() else float("nan")
    m1 = float(s[yy == 1].mean()) if (yy == 1).any() else float("nan")
    return 0.5 * (m0 + m1)


def learned_cusum_path(increments: Sequence[float], k: float) -> np.ndarray:
    """S_t = max(0, S_{t-1} + logit(p_hat_t) - k) (2606.12476 sec.3.3)."""
    s = 0.0
    out = np.empty(len(increments), dtype=float)
    for i, x in enumerate(increments):
        s = max(0.0, s + float(x) - k)
        out[i] = s
    return out


def first_alarm(path: Sequence[float], h: float) -> Optional[int]:
    """tau = min{t : S_t >= h}; None when the run never crosses."""
    for i, v in enumerate(path):
        if v >= h:
            return i
    return None


def simulate_arl(null_increments: Sequence[float], *, k: float, h: float,
                 n_runs: int = 2000, max_steps: int = 4000,
                 seed: int = 20260905) -> Dict[str, Any]:
    """ARL(h) = E_inf[tau] estimated by i.i.d. resampling of the negative pool.

    2606.12476 sec.4.1 aligns detectors at a common ARL by sweeping thresholds on the
    faithful stream; our faithful stream is built by resampling the C->C step
    increments because our episodes are at most 4 steps long.
    """
    rng = np.random.default_rng(seed)
    pool = np.asarray(null_increments, dtype=float)
    if pool.size == 0:
        return {"arl": float("nan"), "censored_frac": 1.0, "h": h}
    lengths = np.empty(n_runs, dtype=float)
    censored = 0
    for r in range(n_runs):
        s = 0.0
        t = 0
        draws = pool[rng.integers(0, pool.size, size=max_steps)]
        for t in range(1, max_steps + 1):
            s = max(0.0, s + draws[t - 1] - k)
            if s >= h:
                break
        else:
            censored += 1
        lengths[r] = t
    return {"arl": float(lengths.mean()), "censored_frac": censored / n_runs,
            "h": float(h), "max_steps": max_steps}


def threshold_for_arl(null_increments: Sequence[float], *, k: float, target_arl: float,
                      n_runs: int = 1000, max_steps: int = 4000, seed: int = 20260905,
                      lo: float = 0.0, hi: float = 200.0,
                      iters: int = 30) -> Dict[str, Any]:
    """Bisect h so that the simulated ARL matches ``target_arl`` (2606.12476 sec.4.1)."""
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        est = simulate_arl(null_increments, k=k, h=mid, n_runs=n_runs,
                           max_steps=max_steps, seed=seed)
        best = est
        if est["arl"] < target_arl:
            lo = mid
        else:
            hi = mid
    out = dict(best or {})
    out["target_arl"] = target_arl
    return out


def threshold_for_arl_memoryless(null_increments: Sequence[float], *,
                                 target_arl: float) -> Dict[str, Any]:
    """Threshold of the NO-ACCUMULATION arm at the same ARL (2606.12476 sec.4.1).

    The paper matches all five detectors "at a common ARL by sweeping THEIR
    thresholds" (sec.4.1), so the per-token/per-step threshold arm may not borrow
    the CUSUM's h: a memoryless rule fires geometrically under the null, so
    ``ARL(h') = 1 / P_0(score >= h')`` and the aligned threshold is the
    ``1 - 1/gamma`` quantile of the null score pool.
    """
    pool = np.asarray(null_increments, dtype=float)
    if pool.size == 0:
        return {"h": float("nan"), "target_arl": target_arl, "n_null": 0}
    q = 1.0 - 1.0 / float(target_arl)
    h = float(np.quantile(pool, q))
    realised = float((pool >= h).mean())
    return {
        "h": h,
        "target_arl": target_arl,
        "n_null": int(pool.size),
        "null_fire_rate": realised,
        "realised_arl": (1.0 / realised) if realised > 0 else float("inf"),
        "quantile": q,
        "note": ("memoryless arm: ARL = 1/P_0(score >= h). With n_null = "
                 f"{pool.size} the finest realisable false-alarm rate is "
                 f"{1.0 / pool.size:.4g}, i.e. the largest exactly-realisable ARL "
                 f"is {pool.size}."),
    }


def align_thresholds_at_arl(increment_pools: Dict[str, Sequence[float]],
                            ks: Dict[str, float], *,
                            targets: Sequence[int] = ARL_TARGETS,
                            max_episode_len: Optional[int] = None,
                            **kw: Any) -> Dict[str, Any]:
    """Thresholds for every detector at a COMMON ARL (2606.12476 sec.4.1).

    The output carries ``arl_exceeds_episode_length`` because on our face the
    shortest requested budget (50 steps) is already longer than any episode.
    """
    out: Dict[str, Any] = {"targets": list(targets), "detectors": {}}
    for name, pool in increment_pools.items():
        rows = {}
        for g in targets:
            rows[str(g)] = threshold_for_arl(pool, k=ks[name], target_arl=float(g), **kw)
        out["detectors"][name] = rows
    if max_episode_len is not None:
        out["max_episode_len"] = int(max_episode_len)
        out["arl_exceeds_episode_length"] = bool(min(targets) > max_episode_len)
        out["note"] = (
            "2606.12476 pitfall 1: their L ~= 120 tokens/generation; our episodes are "
            f"at most {max_episode_len} decision steps, so an ARL of {min(targets)} steps "
            "is longer than any episode and the asymptotic regime does not exist here."
        )
    return out


def detection_delays(fire_step: Dict[str, Optional[int]],
                     onset_step: Dict[str, Optional[int]],
                     last_step: Dict[str, int]) -> Dict[str, Any]:
    """Delay among detected + censored EDD + recall (2606.12476 sec.4.1).

    All three keyed by session.  ``delay = (tau - theta)+`` for sessions the detector
    catches at or after onset; the censored EDD charges a miss the maximum possible
    delay (steps remaining after onset) so it cannot be inflated by low recall.
    """
    delays: List[int] = []
    censored: List[int] = []
    n_onsets = 0
    n_detected = 0
    for sid, theta in onset_step.items():
        if theta is None:
            continue
        n_onsets += 1
        tau = fire_step.get(sid)
        if tau is not None and tau >= theta:
            n_detected += 1
            d = tau - theta
            delays.append(d)
            censored.append(d)
        else:
            censored.append(max(0, last_step[sid] - theta))
    return {
        "n_onsets": n_onsets,
        "n_detected": n_detected,
        "recall": (n_detected / n_onsets) if n_onsets else None,
        "delay_among_detected": float(np.mean(delays)) if delays else None,
        "censored_edd": float(np.mean(censored)) if censored else None,
    }


def shuffle_within_sessions(values_by_session: Dict[str, List[float]],
                            *, seed: int = 20260905) -> Dict[str, List[float]]:
    """Shuffled-ORDER control (2606.12476 sec.4.1 shuffled-sequence arm).

    Same scores, same cost, session order destroyed - the only correct floor for a
    temporal claim.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, List[float]] = {}
    for sid, vals in values_by_session.items():
        arr = np.array(vals, dtype=float)
        rng.shuffle(arr)
        out[sid] = arr.tolist()
    return out


def per_step_threshold_arm(values_by_session: Dict[str, List[float]],
                           h: float) -> Dict[str, Optional[int]]:
    """No-accumulation arm: fire at the first step whose SCORE exceeds h.

    2606.12476 sec.4.2 isolates accumulation from score exactly this way
    (HistGBM-threshold vs HistGBM-CUSUM).
    """
    out: Dict[str, Optional[int]] = {}
    for sid, vals in values_by_session.items():
        out[sid] = first_alarm(vals, h)
    return out


def cusum_arm(values_by_session: Dict[str, List[float]], *, k: float,
              h: float) -> Dict[str, Optional[int]]:
    """Accumulating arm: first crossing of the learned-CUSUM path."""
    return {sid: first_alarm(learned_cusum_path(vals, k), h)
            for sid, vals in values_by_session.items()}


# ==========================================================================
# (B) Calibrated e-CUSUM Decoding - 2607.11317
# ==========================================================================

def tokenize_words(text: str) -> List[str]:
    """Default token stream for r_t: whitespace/punctuation words, lowercased.

    2607.11317 sec.3.1 computes r_t on emitted TEXT; the paper does not fix a
    tokenizer.  A model tokenizer can be injected instead (``tokens=`` argument).
    """
    import re
    return re.findall(r"\w+|[^\w\s]", (text or "").lower())


def causal_repeat_coverage(tokens: Sequence[str], *, n: int = ECUSUM_NGRAM,
                           window: int = ECUSUM_WINDOW) -> np.ndarray:
    """r_t = sliding-window coverage by RECURRING n-grams (2607.11317 sec.3.1).

    Causal by construction: r_t depends only on ``tokens[:t+1]``.  A position is
    covered when it lies inside an occurrence of an n-gram that appears at least
    twice inside the window ending at t.
    """
    toks = list(tokens)
    out = np.zeros(len(toks), dtype=float)
    for t in range(len(toks)):
        lo = max(0, t - window + 1)
        win = toks[lo:t + 1]
        m = len(win)
        if m < 2 * n:
            out[t] = 0.0
            continue
        grams: Dict[Tuple[str, ...], List[int]] = {}
        for i in range(m - n + 1):
            grams.setdefault(tuple(win[i:i + n]), []).append(i)
        covered = np.zeros(m, dtype=bool)
        for _, pos in grams.items():
            if len(pos) >= 2:
                for i in pos:
                    covered[i:i + n] = True
        out[t] = float(covered.sum() / m)
    return out


#: the row shapes :func:`truncated_entropy` accepts, in the order it tries them.
TOP_LOGPROB_ROW_SHAPES = (
    "[logprob, token_id] (t33_capture.py:234 `top5`, VERIFIED against the writer)",
    "bare logprob scalar",
    "{'logprob': float, ...} (the OpenAI/SGLang serving shape, UNVERIFIED - no "
    "capture rerun has landed yet)",
)


def _top_logprob_value(row: Any) -> float:
    """One row of a ``top_logprobs`` list -> its logprob, or a loud failure.

    The local dump shape is fixed and verified (t33_capture.py:234 writes
    ``[float(logprob), int(token_id)]``); the serving shape is NOT, so an
    unrecognised row raises a message that names what it actually saw instead of
    silently reading element 0 of something else.
    """
    if isinstance(row, dict):
        for key in ("logprob", "logprobs", "value"):
            if key in row:
                return float(row[key])
        raise ValueError(
            "truncated_entropy: dict row without a 'logprob' key: keys="
            f"{sorted(row)}; accepted shapes are {TOP_LOGPROB_ROW_SHAPES}")
    if isinstance(row, (int, float)) and not isinstance(row, bool):
        return float(row)
    if isinstance(row, (list, tuple)):
        if not row:
            raise ValueError("truncated_entropy: empty top_logprobs row")
        head = row[0]
        if isinstance(head, (int, float)) and not isinstance(head, bool):
            return float(head)
        # [token_id, logprob] would put an int first and a float second; refuse
        # rather than guess which way round the pair is.
        raise ValueError(
            f"truncated_entropy: unrecognised pair row {row!r}; accepted shapes "
            f"are {TOP_LOGPROB_ROW_SHAPES}")
    raise ValueError(
        f"truncated_entropy: unrecognised row type {type(row).__name__}; accepted "
        f"shapes are {TOP_LOGPROB_ROW_SHAPES}")


def truncated_entropy(top_logprobs: Sequence[Any]) -> Optional[float]:
    """Top-k renormalised entropy (NOT the full-vocabulary entropy).

    Serving face only exposes ``top_logprobs``; 2607.11317 sec.3.1 uses the entropy
    of the sampling distribution.  Rows are parsed by :func:`_top_logprob_value`,
    which accepts :data:`TOP_LOGPROB_ROW_SHAPES` and raises (naming the row) on
    anything else - the serving row shape has never been seen on this machine, so
    a wrong guess must abort rather than produce a plausible number.
    """
    if not top_logprobs:
        return None
    lp = np.array([_top_logprob_value(r) for r in top_logprobs], dtype=float)
    p = np.exp(lp)
    z = p.sum()
    if not np.isfinite(z) or z <= 0:
        return None
    p = p / z
    return float(-(p * np.log(np.maximum(p, 1e-300))).sum())


def entropy_spike_relative(step_means: Sequence[Optional[float]], *,
                           window: int = ECUSUM_ROLL_W) -> List[Optional[float]]:
    """u_n: entropy spike vs the ROLLING history of previous steps (2607.11317 sec.3.1).

    ``u_n = clip((H_n - m)/max(m, eps), 0, 1)`` with ``m`` the mean of the previous
    ``window`` steps' mean entropy inside the same session.  The first step of a
    session has no history and returns None (never 0 - that would be a sentinel).
    """
    out: List[Optional[float]] = []
    hist: List[float] = []
    for h in step_means:
        if h is None:
            out.append(None)
            continue
        if not hist:
            out.append(None)
        else:
            m = float(np.mean(hist[-window:]))
            out.append(float(min(1.0, max(0.0, (float(h) - m) / max(m, 1e-6)))))
        hist.append(float(h))
    return out


def entropy_spike_absolute(step_means: Sequence[Optional[float]], *,
                           scale: float = ENTROPY_ABS_SCALE_NATS) -> List[Optional[float]]:
    """Absolute variant of u (separate column, per the digest)."""
    return [None if h is None else float(min(1.0, max(0.0, float(h) / scale)))
            for h in step_means]


def alarm_score(r: float, u: Optional[float], *, w_rep: float = ECUSUM_W_REP,
                w_ent: float = ECUSUM_W_ENT) -> Optional[float]:
    """a_t = min(1, w_rep*r_t + w_ent*u_t) (2607.11317 sec.3.1).

    Returns None when u is unavailable - the rep-only variant has its own name
    (:func:`alarm_score_rep_only`) so no fused column is ever back-filled with u=0.
    """
    if u is None:
        return None
    return float(min(1.0, w_rep * float(r) + w_ent * float(u)))


def alarm_score_rep_only(r: float, *, w_rep: float = ECUSUM_W_REP) -> float:
    """The honestly-named repetition-only alarm score (u channel absent)."""
    return float(min(1.0, w_rep * float(r)))


def mu0_from_pool(a_values: Sequence[float], *,
                  quantile: float = ECUSUM_MU0_QUANTILE,
                  unit: str = "per-token a_t over C->C rows") -> Dict[str, Any]:
    """mu_0 = 90th percentile of a_t pooled over HEALTHY traces (2607.11317 sec.3.3).

    The pooling unit is stated in the return value: per-token a_t pooled across the
    C->C rows (a 90th percentile of 68 row-level scalars would not be estimable, per
    the card's warning).
    """
    arr = np.asarray([v for v in a_values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mu0": None, "pooling_unit": unit, "n": 0}
    return {
        "mu0": float(np.quantile(arr, quantile)),
        "quantile": quantile,
        "pooling_unit": unit,
        "n": int(arr.size),
        "healthy_mean": float(arr.mean()),
    }


def betting_process(a: Sequence[float], *, lam: float = ECUSUM_LAMBDA,
                    mu0: float) -> np.ndarray:
    """E_n = prod_t (1 + lambda (a_t - mu_0)) (2607.11317 sec.3.2).

    Nonnegative supermartingale under the conditional null; Ville then gives
    ``P(exists n : E_n >= 1/delta) <= delta``.
    """
    vals = np.asarray(a, dtype=float)
    return np.cumprod(1.0 + lam * (vals - mu0))


def ecusum_log_path(a: Sequence[float], *, lam: float = ECUSUM_LAMBDA,
                    mu0: float) -> np.ndarray:
    """S_n = max(0, S_{n-1} + log(1 + lambda (a_n - mu_0))) (2607.11317 sec.3.2).

    DECLARED, as the paper declares it: the reset destroys the supermartingale
    property, so this is an e-process-INSPIRED change detector whose false-alarm
    behaviour must be calibrated and reported empirically.
    """
    s = 0.0
    out = np.empty(len(a), dtype=float)
    for i, x in enumerate(a):
        s = max(0.0, s + math.log(max(1e-12, 1.0 + lam * (float(x) - mu0))))
        out[i] = s
    return out


ECUSUM_SUPERMARTINGALE_NOTE = (
    "2607.11317 sec.3.2/3.3/7: the CUSUM floor (reset to 0) destroys the "
    "supermartingale property; the nominal delta is NOT a guarantee for the deployed "
    "statistic and the achieved false-alarm rate is reported beside it."
)


def ecusum_evaluate(a_by_session: Dict[str, List[float]],
                    healthy_sessions: Sequence[str], *, mu0: float,
                    lam: float = ECUSUM_LAMBDA, tau: float,
                    delta_nominal: float = ECUSUM_DELTA_NOMINAL) -> Dict[str, Any]:
    """Nominal delta vs ACHIEVED false-alarm rate on held-out healthy sessions.

    2607.11317 sec.6.3 reports exactly this pair (nominal 0.05 vs achieved 0.20).
    """
    fired = 0
    n = 0
    max_len = 0
    max_S = 0.0
    for sid in healthy_sessions:
        vals = a_by_session.get(sid)
        if not vals:
            continue
        n += 1
        max_len = max(max_len, len(vals))
        path = ecusum_log_path(vals, lam=lam, mu0=mu0)
        max_S = max(max_S, float(path.max()))
        if first_alarm(path, tau) is not None:
            fired += 1
    # a_n is bounded by 1, so the largest increment any step can contribute is
    # log(1 + lam(1 - mu_0)); over an episode of max_len steps the statistic cannot
    # exceed max_len times that.  When that ceiling is below tau the detector is
    # SILENT BY CONSTRUCTION and an achieved rate of 0 says nothing about calibration.
    per_step_ceiling = math.log(max(1e-12, 1.0 + lam * (1.0 - mu0)))
    ceiling = max_len * per_step_ceiling
    return {
        "nominal_delta": delta_nominal,
        "ville_threshold_equivalent": math.log(1.0 / delta_nominal),
        "tau": tau,
        "achieved_false_alarm_rate": (fired / n) if n else None,
        "n_healthy_sessions": n,
        "n_fired": fired,
        "max_episode_len": max_len,
        "max_attainable_S": ceiling,
        "max_observed_S": max_S,
        "tau_attainable_at_this_episode_length": bool(ceiling >= tau),
        "silence_note": (
            None if ceiling >= tau else
            f"tau={tau:.4g} is UNREACHABLE: a_n <= 1 caps a step's increment at "
            f"log(1+lam(1-mu_0))={per_step_ceiling:.4g} and the longest healthy "
            f"episode has {max_len} steps, so S_n <= {ceiling:.4g} < tau. The achieved "
            "false-alarm rate of 0 is a STRUCTURAL SILENCE, not evidence of "
            "calibration; the cross-step e-CUSUM needs longer episodes."),
        "note": ECUSUM_SUPERMARTINGALE_NOTE,
    }


# ==========================================================================
# (C) CURA - 2608.27808
# ==========================================================================

def ltt_alpha_grid(n: int, *, alphas: Sequence[float] = (0.05, 0.10, 0.20),
                   delta: float = CURA_DELTA) -> Dict[str, Any]:
    """The ten-line feasibility check (2608.27808 sec.4 + card open question 1).

    With ``n`` calibration points the smallest attainable exact-binomial p-value for
    ``H_j : FPR(theta_j) > alpha`` is ``P(X<=0) = (1-alpha)^n``.  An alpha can be
    certified at level ``delta`` only if that is <= delta.
    """
    from scipy.stats import binom
    rows = []
    for a in alphas:
        p_at_zero = float((1.0 - a) ** n)
        kmax = None
        for k in range(0, n + 1):
            if float(binom.cdf(k, n, a)) <= delta:
                kmax = k
            else:
                break
        rows.append({
            "alpha": a,
            "min_pvalue_at_zero_false_alarms": p_at_zero,
            "certifiable": bool(p_at_zero <= delta),
            "max_false_alarms_allowed": kmax,
            "n_required_for_alpha": int(math.ceil(math.log(delta) / math.log(1.0 - a))),
        })
    return {"n": n, "delta": delta, "grid": rows}


def ltt_select_threshold(cal_scores: Sequence[float], *,
                         alpha: float, delta: float = CURA_DELTA,
                         thresholds: Optional[Sequence[float]] = None) -> Dict[str, Any]:
    """Learn-then-Test fixed-sequence threshold selection (2608.27808 sec.4).

    Candidates are tested in DESCENDING order with the exact binomial test of
    ``H_j : FPR(theta_j) > alpha``; testing stops at the first non-rejection and the
    LAST REJECTED threshold is returned.  If none is rejected the result is
    ``"no threshold certified"`` - a legitimate outcome, and stronger than an AUROC.

    ``cal_scores`` are the monitor's scores on the healthy calibration pool only
    (C->C rows); no failure label is used.
    """
    from scipy.stats import binom
    s = np.asarray([v for v in cal_scores if v is not None and np.isfinite(v)],
                   dtype=float)
    n = int(s.size)
    if n == 0:
        return {"certified": False, "theta": None, "reason": "empty calibration pool"}
    if thresholds is None:
        obs = np.unique(s)
        # The fixed sequence must be able to express ZERO false alarms: with the
        # candidate grid taken from the calibration scores alone the top rung is
        # theta = max(s), which already has k = 1, so a small alpha (at n = 68,
        # alpha = 0.05 admits at most 0 false alarms) could never be certified for
        # a purely arithmetic reason.  One rung just above the observed maximum is
        # added; it is flagged in the result because it is vacuous as a detector.
        top = float(np.nextafter(obs[-1], np.inf))
        cands = np.concatenate([[top], obs[::-1]])
    else:
        cands = np.asarray(sorted(set(float(t) for t in thresholds)), dtype=float)[::-1]
    tested: List[Dict[str, Any]] = []
    last_rejected: Optional[float] = None
    for theta in cands:
        k = int((s >= theta).sum())
        pval = float(binom.cdf(k, n, alpha))
        rejected = bool(pval <= delta)
        tested.append({"theta": float(theta), "false_alarms": k, "pvalue": pval,
                       "rejected": rejected})
        if not rejected:
            break
        last_rejected = float(theta)
    return {
        "certified": last_rejected is not None,
        "theta": last_rejected,
        "alpha": alpha,
        "delta": delta,
        "n_calibration": n,
        "n_tested": len(tested),
        "sequence": tested,
        "theta_above_all_calibration_scores": bool(
            last_rejected is not None and last_rejected > float(s.max())),
        "reason": None if last_rejected is not None else "no threshold certified",
        "scope": ("2608.27808 sec.4: the certificate controls FALSE ALARMS only under the "
                  "calibrated healthy distribution - not detection, task correctness or "
                  "safety; it must be recalibrated under distribution shift."),
    }


def ltt_three_way(scores_by_qid: Optional[Dict[str, float]], sessions: Dict[str, str], *,
                  alpha: float, delta: float = CURA_DELTA, repeats: int = 200,
                  seed: int = 20260905,
                  score_fn: Optional[Callable[[Sequence[str]], Dict[str, float]]] = None,
                  qids: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """200 repeated session-grouped fit/calibration/test splits (2608.27808 sec.5).

    All three parts come from the HEALTHY class; the reported quantity is the share
    of replicates whose TEST false-alarm rate exceeds the target alpha (their 2.0 %
    / 4.5 % / 6.5 % at alpha 0.05/0.10/0.20 against a nominal delta of 5 %).

    The point of the protocol (2608.27808 sec.4: "Physiology models, the screen
    predictor, normalization, and step-score constants are fit only on the fit set
    and then frozen"; sec.5: "refit all learned components on fit") is that NOTHING
    is fitted outside the fit third.  Pass ``score_fn(fit_sessions) -> {qid: score}``
    so the monitor is genuinely refitted per replicate; ``scores_by_qid`` (a single
    globally-fitted score vector) is only accepted for the degenerate case where the
    score carries no fitted constant, and the result then says
    ``refit_inside_fit_fold: False``.
    """
    if score_fn is None and scores_by_qid is None:
        raise ValueError("ltt_three_way needs either score_fn or scores_by_qid")
    rng = np.random.default_rng(seed)
    if qids is None:
        qids = [q for q in (scores_by_qid or {}) if (scores_by_qid or {})[q] is not None]
    qids = list(qids)
    sess = sorted({sessions[q] for q in qids})
    exceed = 0
    used = 0
    certified = 0
    test_fprs: List[float] = []
    for _ in range(repeats):
        perm = rng.permutation(len(sess))
        n1, n2 = len(sess) // 3, 2 * len(sess) // 3
        fit = {sess[i] for i in perm[:n1]}
        cal = {sess[i] for i in perm[n1:n2]}
        tst = {sess[i] for i in perm[n2:]}
        scores = score_fn(sorted(fit)) if score_fn is not None else scores_by_qid
        cal_s = [scores[q] for q in qids
                 if sessions[q] in cal and scores.get(q) is not None]
        tst_s = [scores[q] for q in qids
                 if sessions[q] in tst and scores.get(q) is not None]
        if not cal_s or not tst_s:
            continue
        used += 1
        sel = ltt_select_threshold(cal_s, alpha=alpha, delta=delta)
        if not sel["certified"]:
            continue
        certified += 1
        fpr = float(np.mean(np.asarray(tst_s, dtype=float) >= sel["theta"]))
        test_fprs.append(fpr)
        if fpr > alpha:
            exceed += 1
    return {
        "alpha": alpha,
        "delta": delta,
        "repeats_used": used,
        "n_certified": certified,
        "exceedance_rate": (exceed / certified) if certified else None,
        "mean_test_fpr": float(np.mean(test_fprs)) if test_fprs else None,
        "n_sessions": len(sess),
        "refit_inside_fit_fold": bool(score_fn is not None),
    }


@dataclass
class PhysiologyModel:
    """Per-tool diagonal Gaussian over z-scored vitals (2608.27808 sec.4).

    Fitted on HEALTHY (C->C) rows only; ``NLL_t = 1/2 sum_d ((v-mu)/sigma)^2 +
    sum_d log sigma``.
    """

    vitals: Tuple[str, ...]
    z_mean: np.ndarray
    z_std: np.ndarray
    per_tool: Dict[str, Tuple[np.ndarray, np.ndarray]]
    pooled: Optional[Tuple[np.ndarray, np.ndarray]]
    n_rows: Dict[str, int]
    min_tool_rows: int

    def vector(self, row: Dict[str, Any]) -> Optional[np.ndarray]:
        vals = []
        for key in self.vitals:
            v = row.get(key)
            if v is None:
                return None
            vals.append(float(v))
        return (np.asarray(vals, dtype=float) - self.z_mean) / self.z_std

    def nll(self, row: Dict[str, Any], tool: Optional[str]) -> Tuple[Optional[float], bool]:
        """Returns (NLL, tool_seen).  None when the vitals or the model are missing."""
        v = self.vector(row)
        if v is None:
            return None, False
        seen = bool(tool is not None and self.n_rows.get(tool, 0) >= self.min_tool_rows)
        mu, sd = (self.per_tool[tool] if seen and tool in self.per_tool
                  else (self.pooled if self.pooled is not None else (None, None)))
        if mu is None:
            return None, seen
        z = (v - mu) / sd
        return float(0.5 * np.sum(z ** 2) + np.sum(np.log(sd))), seen


def physiology_fit(rows: Sequence[Dict[str, Any]], tools: Sequence[Optional[str]], *,
                   vitals: Sequence[str] = PHYS_VITALS,
                   min_tool_rows: int = PHYS_MIN_TOOL_ROWS,
                   var_floor: float = 1e-6) -> Optional[PhysiologyModel]:
    """Fit the physiology channel on HEALTHY rows only (2608.27808 sec.4).

    ``rows`` must be C->C rows; ``tools`` the tool name each row's own arm emitted
    (parsed from its own text - never the scoring column).
    """
    mat: List[List[float]] = []
    keep_tools: List[Optional[str]] = []
    for row, tool in zip(rows, tools):
        vals = [row.get(k) for k in vitals]
        if any(v is None for v in vals):
            continue
        mat.append([float(v) for v in vals])
        keep_tools.append(tool)
    if not mat:
        return None
    X = np.asarray(mat, dtype=float)
    z_mean = X.mean(axis=0)
    z_std = np.maximum(X.std(axis=0, ddof=0), math.sqrt(var_floor))
    Z = (X - z_mean) / z_std
    per_tool: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    n_rows: Dict[str, int] = {}
    for tool in set(t for t in keep_tools if t):
        idx = [i for i, t in enumerate(keep_tools) if t == tool]
        n_rows[tool] = len(idx)
        if len(idx) >= min_tool_rows:
            sub = Z[idx]
            per_tool[tool] = (sub.mean(axis=0),
                              np.maximum(sub.std(axis=0, ddof=0), math.sqrt(var_floor)))
    pooled = (Z.mean(axis=0), np.maximum(Z.std(axis=0, ddof=0), math.sqrt(var_floor)))
    return PhysiologyModel(vitals=tuple(vitals), z_mean=z_mean, z_std=z_std,
                           per_tool=per_tool, pooled=pooled, n_rows=n_rows,
                           min_tool_rows=min_tool_rows)


def prefix_signals(row: Dict[str, Any], sidecar: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Optional[float]]:
    """The prefix-only ("gate-stage") signal vector (2608.27808 sec.4 gate stage).

    Every value is available BEFORE generation.  Fields that live only in the
    sidecar / proxy row are None on the battery face and are dropped from S_t
    rather than entering it as a zero.

    INDEX SPACES (t34_dump_sidecar "the trap this dump exists to make explicit"):
    ``docs`` / ``doc_lengths`` hold ONLY THE KEPT blocks, while ``dropped_docs``
    indexes the POST-SPLIT HISTORY list.  The declared signal ``n_docs`` is "how
    many history blocks could be damaged", i.e. the POST-SPLIT total = kept +
    dropped; ``len(doc_lengths)`` alone is the visible count and is NOT that
    number.  When the dropped side is unavailable the total is unknown and the
    signal is None (it never falls back to the kept count).
    """
    sc = sidecar or {}
    doc_lengths = sc.get("doc_lengths")
    dropped = sc.get("dropped_docs")
    n_kept = len(doc_lengths) if isinstance(doc_lengths, list) else None
    n_dropped = (float(len(dropped)) if isinstance(dropped, list) else _num(dropped))
    if n_kept is not None and n_dropped is not None:
        n_docs: Optional[float] = float(n_kept) + float(n_dropped)
    else:
        n_docs = _num(sc.get("n_docs"))       # a dump that states the total itself
    # Saturation of the tail window: the direct evidence is a non-empty dropped
    # list; the caliber comparison is the fallback when only the total is known.
    if n_dropped is not None:
        sat: Optional[float] = 1.0 if n_dropped > 0 else 0.0
    elif n_docs is not None:
        sat = 1.0 if float(n_docs) > MAX_DOC_NUM_CALIBER else 0.0
    else:
        sat = None
    return {
        "gist_tokens": _num(row.get("gist_tokens")),
        "actual_compression_ratio": _num(row.get("actual_compression_ratio")),
        "n_docs": _num(n_docs),
        "doc_chunks": _num(row.get("doc_chunks")),
        "kept_history_tokens": _num(row.get("kept_history_tokens")),
        "hybrid_top_k": _num(row.get("hybrid_top_k")),
        "max_doc_num_saturated": sat,
        "dropped_docs": n_dropped,
        "repair_frame_delta": _num(row.get("repair_frame_delta")),
    }


def _num(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None if v is None else float(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def prefix_norm_constants(sig_rows: Sequence[Dict[str, Optional[float]]]
                          ) -> Dict[str, Tuple[float, float]]:
    """Frozen (mu_j, sigma_j) for the prefix gate, fit on the calibration pool only."""
    out: Dict[str, Tuple[float, float]] = {}
    for key in CURA_PREFIX_SIGNS:
        vals = [r[key] for r in sig_rows if r.get(key) is not None]
        if len(vals) < 2:
            continue
        arr = np.asarray(vals, dtype=float)
        sd = float(arr.std(ddof=0))
        if sd <= 0:
            continue
        out[key] = (float(arr.mean()), sd)
    return out


def prefix_step_score(sig: Dict[str, Optional[float]],
                      norm: Dict[str, Tuple[float, float]]) -> Tuple[Optional[float], int]:
    """S_t = sum_j sign_j (s_j - mu_j)/sigma_j (2608.27808 sec.4), signs fixed a priori.

    Returns (S_t, n_signals_used); None when no signal participates.
    """
    total = 0.0
    used = 0
    for key, sign in CURA_PREFIX_SIGNS.items():
        v = sig.get(key)
        if v is None or key not in norm:
            continue
        mu, sd = norm[key]
        total += sign * (v - mu) / sd
        used += 1
    return (total if used else None), used


def prefix_gate_path(step_scores: Sequence[Optional[float]], *,
                     k: float = CURA_K) -> List[Optional[float]]:
    """W_t = max(0, W_{t-1} + S_t - k), k = 0.5 (2608.27808 sec.4).

    Steps whose S_t is undefined leave W unchanged and report None for that step.
    """
    w = 0.0
    out: List[Optional[float]] = []
    for s in step_scores:
        if s is None:
            out.append(None)
            continue
        w = max(0.0, w + float(s) - k)
        out.append(w)
    return out


def matched_fpr_comparison(scores_a: Sequence[float], scores_b: Sequence[float],
                           y: Sequence[int], *, target_fpr: float,
                           groups: Sequence[str], tol: float = 0.02,
                           seed: int = 20260905,
                           cal_frac: float = 0.5) -> Dict[str, Any]:
    """Recall of two detectors at a MATCHED false-positive rate (2608.27808 sec.5 C).

    The digest makes this mandatory as the S0 control family's length-only baseline:
    any composite must be shown to beat length at the same false-alarm budget.

    Each arm's operating point is chosen on a session-grouped CALIBRATION half, from
    ITS OWN NEGATIVES ONLY, and every reported rate - FPR, recall, fires, tie mass -
    is measured on the DISJOINT held-out half.  Taking the quantile of the very
    negatives the arm is then scored against is the selection leak this control
    exists to exclude (2608.27808 sec.4: "split into disjoint fit and calibration
    sets BEFORE threshold selection").  ``groups`` are the session ids - the grouping
    unit of every split on this face - and are REQUIRED; there is no in-sample path.

    A heavily TIED score (on the r2 set ``generated_tokens`` piles up on the
    128-token cap) still cannot realise an arbitrary FPR: the achieved held-out rate
    of each arm is always reported, ``matched`` says whether the two rates actually
    agree within ``tol``, and the tie mass at each threshold is printed so an
    unmatched comparison cannot be read as a matched one.
    """
    yy = np.asarray(y, dtype=int)
    gg = np.asarray([str(g) for g in groups])
    if gg.size != yy.size:
        raise ValueError("groups must be one session id per row")
    sess = np.unique(gg)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(sess.size)
    n_cal = int(round(cal_frac * sess.size))
    cal_sessions = {sess[i] for i in perm[:n_cal]}
    cal_mask = np.array([s in cal_sessions for s in gg], dtype=bool)
    ev_mask = ~cal_mask
    neg, pos = (yy == 0), (yy == 1)
    n_cal_neg = int((cal_mask & neg).sum())
    out: Dict[str, Any] = {
        "target_fpr": target_fpr, "n": int(yy.size),
        "n_pos": int(pos.sum()), "n_neg": int(neg.sum()), "tol": tol,
        "threshold_selected_on": ("session-grouped CALIBRATION half, negatives only; "
                                  "every rate below is measured on the disjoint "
                                  "held-out half"),
        "split": {
            "unit": "session", "seed": seed, "cal_frac": cal_frac,
            "n_sessions": int(sess.size),
            "n_calibration_sessions": len(cal_sessions),
            "n_calibration_rows": int(cal_mask.sum()),
            "n_calibration_negatives": n_cal_neg,
            "n_evaluation_rows": int(ev_mask.sum()),
            "n_evaluation_negatives": int((ev_mask & neg).sum()),
            "n_evaluation_positives": int((ev_mask & pos).sum()),
        },
        "calibration_resolution_note": (
            None if n_cal_neg * target_fpr >= 1.0 else
            f"the calibration half holds {n_cal_neg} negatives, so a target FPR of "
            f"{target_fpr} is finer than the {1.0 / n_cal_neg if n_cal_neg else float('nan'):.4g} "
            "resolution of the pool the threshold is read from"),
    }
    n_ev_neg = int((ev_mask & neg).sum())
    out["evaluation_resolution_note"] = (
        None if n_ev_neg and n_ev_neg * target_fpr >= 1.0 else
        f"the held-out half holds {n_ev_neg} negatives, so the reported FPRs are "
        f"quantised to {1.0 / n_ev_neg if n_ev_neg else float('nan'):.4g}; a "
        f"{target_fpr} budget is finer than this frame can measure")
    for name, s in (("a", scores_a), ("b", scores_b)):
        arr = np.asarray(s, dtype=float)
        cal_neg = arr[cal_mask & neg]
        if cal_neg.size == 0 or not ev_mask.any():
            out[name] = None
            out[name + "_unavailable_reason"] = (
                "empty calibration negatives" if cal_neg.size == 0
                else "empty evaluation half")
            continue
        thr = float(np.quantile(cal_neg, 1.0 - target_fpr))
        fire = arr >= thr
        ev_neg, ev_pos = (ev_mask & neg), (ev_mask & pos)
        out[name] = {
            "threshold": thr,
            "fpr": float(fire[ev_neg].mean()) if ev_neg.any() else None,
            "recall": float(fire[ev_pos].mean()) if ev_pos.any() else None,
            "fires": int(fire[ev_mask].sum()),
            "tie_mass_at_threshold": float(np.mean(arr[ev_mask] == thr)),
            "calibration_fpr_in_sample": float((cal_neg >= thr).mean()),
            "n_evaluation_negatives": int(ev_neg.sum()),
            "n_evaluation_positives": int(ev_pos.sum()),
        }
    a, b = out.get("a"), out.get("b")
    have = bool(a and b and a["fpr"] is not None and b["fpr"] is not None)
    out["matched"] = bool(have and abs(a["fpr"] - b["fpr"]) <= tol)
    if have and not out["matched"]:
        out["unmatched_note"] = (
            f"held-out FPRs differ by {abs(a['fpr'] - b['fpr']):.3f} > tol={tol}: at "
            "least one score is heavily tied (see tie_mass_at_threshold) and cannot "
            "realise the target budget. Recalls below are NOT at a matched budget. "
            + CAP_CONFOUND_NOTE
        )
    elif not have:
        out["unmatched_note"] = ("at least one arm has no held-out false-alarm rate; "
                                 "see the *_unavailable_reason keys and the split "
                                 "denominators")
    return out


def fold_internal_floor(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                        **kw: Any) -> Dict[str, Any]:
    """Fold-internal floor (2608.27808 sec.4/5): refit EVERYTHING inside each fold.

    Thin wrapper around ``t34_common.nested_cv_logistic`` so that standardisation,
    C and any selector are chosen in inner folds only - "a floor no bias can inflate".
    """
    res = nested_cv_logistic(X, y, groups, **kw)
    return {
        "auprc": res["auprc"],
        "auroc": res["auroc"],
        "n_scored": res["n_scored"],
        "n_pos_scored": res["n_pos_scored"],
        "prevalence": res["prevalence"],
        "chosen": res["chosen"],
        "n_dropped_nan": res["n_dropped_nan"],
    }


# ==========================================================================
# (D) ESN + per-channel-max CUSUM telemetry - 2608.02464
# ==========================================================================

ESN_CHANNELS: Dict[str, Tuple[str, ...]] = {
    # numeric size / packing fields
    "size": ("gist_tokens", "original_tokens", "n_docs", "dropped_docs",
             "doc_packing_turn", "doc_packing_message", "repair_frame_delta"),
    # timing
    "timing": ("wall_sec", "completion_tokens", "prompt_tokens"),
    # protocol / outcome
    "protocol": ("finish_reason_stop", "finish_reason_length", "finish_reason_other",
                 "error_kind_none", "error_kind_other", "has_tool_call",
                 "protocol_legal", "n_chars"),
}


def observation_vector(row: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """x_t per channel from a proxy request-log row (2608.02464 sec.3 eq.1).

    NO ``u`` channel (declared: the paper ablates its token-surprisal channel to
    +0.000 AUROC and we have never emitted a logprob on the serving face) and no
    ``e`` channel.  Missing numeric fields become 0.0 AFTER the healthy-fit
    standardisation is applied by the caller; the presence mask is not modelled -
    a row missing a whole channel should be excluded upstream.
    """
    usage = row.get("usage") or {}
    finish = (row.get("finish_reason") or "other")
    err = row.get("error_kind")
    text = row.get("text") or ""
    flat: Dict[str, float] = {
        "gist_tokens": float(row.get("gist_tokens") or 0.0),
        "original_tokens": float(row.get("original_tokens") or 0.0),
        "n_docs": float(row.get("n_docs") or 0.0),
        "dropped_docs": float(len(row["dropped_docs"]) if isinstance(row.get("dropped_docs"), list)
                              else (row.get("dropped_docs") or 0.0)),
        "doc_packing_turn": 1.0 if row.get("doc_packing") == "turn" else 0.0,
        "doc_packing_message": 1.0 if row.get("doc_packing") == "message" else 0.0,
        "repair_frame_delta": float(row.get("repair_frame_delta") or 0.0),
        "wall_sec": float(row.get("wall_sec") or 0.0),
        "completion_tokens": float(usage.get("completion_tokens") or 0.0),
        "prompt_tokens": float(usage.get("prompt_tokens") or 0.0),
        "finish_reason_stop": 1.0 if finish == "stop" else 0.0,
        "finish_reason_length": 1.0 if finish == "length" else 0.0,
        "finish_reason_other": 1.0 if finish not in ("stop", "length") else 0.0,
        "error_kind_none": 1.0 if not err else 0.0,
        "error_kind_other": 1.0 if err else 0.0,
        "has_tool_call": 1.0 if row.get("has_tool_call") else 0.0,
        "protocol_legal": 1.0 if row.get("protocol_legal") else 0.0,
        "n_chars": float(len(text)),
    }
    return {ch: np.asarray([flat[k] for k in keys], dtype=float)
            for ch, keys in ESN_CHANNELS.items()}


def episodes_from_proxy(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, np.ndarray]],
                                                                 Dict[str, int]]:
    """Group proxy rows into per-conversation per-channel matrices.

    Rows whose ``error_kind`` is in :data:`ESN_EXCLUDED_ERROR_KINDS` form an explicit
    exclusion class (2608.02464 card pitfall: the gist-pool LRU eviction would turn
    the monitor into a cache-miss detector) and are counted, never silently kept.
    """
    per_conv: Dict[str, List[Dict[str, np.ndarray]]] = {}
    counts = {"rows": 0, "excluded_cache_miss": 0, "no_conv": 0}
    for row in rows:
        counts["rows"] += 1
        if (row.get("error_kind") or row.get("status")) in ESN_EXCLUDED_ERROR_KINDS:
            counts["excluded_cache_miss"] += 1
            continue
        conv = row.get("conv_id")
        if not conv:
            counts["no_conv"] += 1
            continue
        per_conv.setdefault(conv, []).append(observation_vector(row))
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for conv, steps in per_conv.items():
        out[conv] = {ch: np.vstack([s[ch] for s in steps]) for ch in ESN_CHANNELS}
    return out, counts


@dataclass
class DeltaMahalanobis:
    """Memoryless per-step surprise baseline (2608.02464 sec.3 baselines).

    Fitted on healthy episodes only.  ``delta=True`` uses first differences
    (Delta-Mahalanobis, which WINS on their short-episode corpora, 0.848 vs 0.777);
    ``delta=False`` is the plain memoryless Mahalanobis.
    """

    mean: np.ndarray
    prec: np.ndarray
    delta: bool

    @staticmethod
    def fit(mats: Sequence[np.ndarray], *, delta: bool = True,
            ridge: float = 1e-6) -> "DeltaMahalanobis":
        rows = []
        for m in mats:
            x = np.diff(m, axis=0) if delta else m
            if x.shape[0]:
                rows.append(x)
        X = np.vstack(rows) if rows else np.zeros((0, mats[0].shape[1]))
        mu = X.mean(axis=0) if X.shape[0] else np.zeros(mats[0].shape[1])
        cov = np.cov(X, rowvar=False) if X.shape[0] > 1 else np.eye(mu.size)
        cov = np.atleast_2d(cov) + ridge * np.eye(mu.size)
        return DeltaMahalanobis(mean=mu, prec=np.linalg.pinv(cov), delta=delta)

    def scores(self, mat: np.ndarray) -> np.ndarray:
        x = np.diff(mat, axis=0) if self.delta else mat
        if x.shape[0] == 0:
            return np.zeros(0)
        d = x - self.mean
        return np.einsum("ij,jk,ik->i", d, self.prec, d)


@dataclass
class Reservoir:
    """Fixed-seed sparse random recurrent map, NEVER trained (2608.02464 sec.3 eq.2).

    ``h_t = tanh(W h_{t-1} + W_in x_t)``.  Determinism: the same seed gives the same
    map, which is why the monitor does not break the repair-determinism gate.
    """

    n_in: int
    size: int = 32
    sparsity: float = 0.1
    spectral_radius: float = 0.9
    seed: int = 20260905
    #: leak rate alpha of eq.2; the paper writes the leaky form but fixes no value,
    #: so we declare alpha = 1.0 (the plain tanh update) - see DEVIATIONS.
    leak: float = 1.0
    W: np.ndarray = field(init=False)
    W_in: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        W = rng.normal(size=(self.size, self.size))
        mask = rng.random((self.size, self.size)) < self.sparsity
        W = W * mask
        eig = np.max(np.abs(np.linalg.eigvals(W))) if self.size else 0.0
        if eig > 0:
            W = W * (self.spectral_radius / eig)
        self.W = W
        self.W_in = rng.normal(size=(self.size, self.n_in)) * 0.5

    def run(self, X: np.ndarray) -> np.ndarray:
        """States h_1..h_T for the step matrix X (T x n_in)."""
        h = np.zeros(self.size)
        out = np.empty((X.shape[0], self.size), dtype=float)
        a = float(self.leak)
        for t in range(X.shape[0]):
            h = (1.0 - a) * h + a * np.tanh(self.W @ h + self.W_in @ X[t])
            out[t] = h
        return out


def ridge_fit(Z: np.ndarray, Y: np.ndarray, lam: float = 1e-3) -> np.ndarray:
    """Closed-form ridge readout A = (Z^T Z + lam I)^-1 Z^T Y (2608.02464 sec.3 eq.3)."""
    d = Z.shape[1]
    return np.linalg.solve(Z.T @ Z + lam * np.eye(d), Z.T @ Y)


@dataclass
class EsnChannel:
    """One channel's frozen reservoir + healthy-fitted ridge readout + z constants."""

    reservoir: Reservoir
    A: np.ndarray
    sigma_err: np.ndarray
    q_mean: float
    q_std: float
    #: where sigma_err / (q_mean, q_std) were measured - 2608.02464 sec.3 measures
    #: them on HELD-OUT healthy runs, and an in-sample fallback must say so.
    sigma_source: str = "held-out healthy episodes"
    sigma_in_sample: bool = False
    n_sigma_episodes: int = 0

    def surprise(self, X: np.ndarray) -> np.ndarray:
        """q_t: mean over dims of the squared normalised one-step prediction error."""
        if X.shape[0] < 2:
            return np.zeros(0)
        H = self.reservoir.run(X)
        Z = np.hstack([H[:-1], X[:-1], np.ones((X.shape[0] - 1, 1))])
        pred = Z @ self.A
        err = (X[1:] - pred) / self.sigma_err
        return np.mean(err ** 2, axis=1)

    def cusum(self, X: np.ndarray, *, kappa: float) -> np.ndarray:
        """S_t = max(0, S_{t-1} + z(q_t) - kappa) (2608.02464 sec.3 eq.5)."""
        q = self.surprise(X)
        z = (q - self.q_mean) / max(self.q_std, 1e-9)
        s = 0.0
        out = np.empty(z.size, dtype=float)
        for i, v in enumerate(z):
            s = max(0.0, s + float(v) - kappa)
            out[i] = s
        return out


def esn_channel_fit(healthy_mats: Sequence[np.ndarray], *, seed: int = 20260905,
                    size: int = 32, lam: float = 1e-3,
                    sigma_mats: Optional[Sequence[np.ndarray]] = None
                    ) -> Optional[EsnChannel]:
    """Fit ONE channel's ridge readout on healthy episodes only (2608.02464 sec.3).

    The recurrent map is frozen at initialisation; only the readout is solved, in
    closed form (their 1.7 s vs a GRU's 68 s).

    ``sigma_mats`` are HELD-OUT healthy episodes (disjoint from ``healthy_mats``):
    2608.02464 sec.3 measures the residual scale ``sigma_err`` of eq.4 - and, here,
    the (q_mean, q_std) normalisation eq.5 z-scores with - on runs the readout has
    not seen.  Measured on the fit residuals instead they are optimistically small,
    which inflates every q_t on unseen data.  When no held-out pool is supplied the
    fit residuals are used and the channel says so
    (``sigma_in_sample=True`` / ``sigma_source``), never silently.
    """
    mats = [m for m in healthy_mats if m.shape[0] >= 2]
    if not mats:
        return None
    res = Reservoir(n_in=mats[0].shape[1], size=size, seed=seed)

    def _design(ms: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        Zs, Ys = [], []
        for m in ms:
            H = res.run(m)
            Zs.append(np.hstack([H[:-1], m[:-1], np.ones((m.shape[0] - 1, 1))]))
            Ys.append(m[1:])
        return np.vstack(Zs), np.vstack(Ys)

    Z, Y = _design(mats)
    A = ridge_fit(Z, Y, lam)

    held = [m for m in (sigma_mats or []) if m.shape[0] >= 2]
    if held:
        Zh, Yh = _design(held)
        resid = Yh - Zh @ A
        source = "held-out healthy episodes (2608.02464 sec.3)"
        in_sample = False
    else:
        resid = Y - Z @ A
        source = ("IN-SAMPLE fit residuals - no held-out healthy pool was supplied; "
                  "2608.02464 sec.3 measures sigma_err on held-out healthy runs")
        in_sample = True
    sigma = np.maximum(resid.std(axis=0, ddof=0), 1e-6)
    ch = EsnChannel(reservoir=res, A=A, sigma_err=sigma, q_mean=0.0, q_std=1.0,
                    sigma_source=source, sigma_in_sample=in_sample,
                    n_sigma_episodes=len(held))
    q_pool = held if held else mats
    qs = np.concatenate([ch.surprise(m) for m in q_pool]) if q_pool else np.zeros(0)
    ch.q_mean = float(qs.mean()) if qs.size else 0.0
    ch.q_std = float(qs.std(ddof=0)) if qs.size else 1.0
    return ch


def fuse_per_channel_max(paths: Dict[str, np.ndarray]) -> np.ndarray:
    """s_t = max_c S_t^(c) (2608.02464 sec.3 eq.6) - NEVER the mean.

    Their own ablation says this wrapper, not the reservoir, carries most of the
    margin: a drift confined to one channel is diluted by averaging.
    """
    arrs = [p for p in paths.values() if p.size]
    if not arrs:
        return np.zeros(0)
    n = min(a.size for a in arrs)
    return np.max(np.vstack([a[:n] for a in arrs]), axis=0)


def fuse_mean(paths: Dict[str, np.ndarray]) -> np.ndarray:
    """Pooled-mean fusion - computed ONLY as the contrast their ablation demands."""
    arrs = [p for p in paths.values() if p.size]
    if not arrs:
        return np.zeros(0)
    n = min(a.size for a in arrs)
    return np.mean(np.vstack([a[:n] for a in arrs]), axis=0)


def threshold_from_validation(healthy_val_paths: Sequence[np.ndarray], *,
                              beta: float) -> Optional[float]:
    """theta = Q_{1-beta}({max_t s_t : healthy VALIDATION episodes}) (2608.02464 sec.3).

    Read from healthy validation episodes and never from test data.
    """
    maxima = [float(p.max()) for p in healthy_val_paths if p.size]
    if not maxima:
        return None
    return float(np.quantile(np.asarray(maxima, dtype=float), 1.0 - beta))


def esn_monitor(episodes: Dict[str, Dict[str, np.ndarray]],
                healthy_ids: Sequence[str], *, beta: float = 0.05,
                kappa: float = 0.5, seed: int = 20260905,
                val_frac: float = 0.34, sigma_frac: float = 0.2) -> Dict[str, Any]:
    """Full Layer-A pipeline (2608.02464 sec.3), Delta-Mahalanobis baseline FIRST.

    Aborts with a message when fewer than :data:`ESN_MIN_HEALTHY_EPISODES` healthy
    episodes are supplied - their stated data requirement ("provoking enough
    fabrication leaves 2 healthy episodes against the 15 a null needs").
    ``healthy`` MUST be bench task success, never "compressed matched full".

    The healthy pool is cut THREE ways by episode: the readout fit split, a
    sigma/normalisation split held out from the readout (2608.02464 sec.3 measures
    sigma_err on held-out healthy runs), and the validation split eq.6 reads theta
    from.  ``sigma_frac=0`` collapses the middle split and the result then carries
    ``sigma_err_source.held_out = False``.
    """
    healthy = [c for c in healthy_ids if c in episodes]
    if len(healthy) < ESN_MIN_HEALTHY_EPISODES:
        return {
            "aborted": True,
            "reason": (f"healthy null needs >= {ESN_MIN_HEALTHY_EPISODES} healthy episodes "
                       f"(2608.02464 sec.10); got {len(healthy)}. This is a DATA problem, "
                       "not a method problem: tau2 is CONTAMINATED, BFCL is at the format "
                       "floor, TS is n=3."),
            "n_healthy": len(healthy),
            "cpu_cost_note": ESN_CPU_COST_NOTE,
        }
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(healthy))
    n_val = max(1, int(round(val_frac * len(healthy))))
    n_sigma = int(round(sigma_frac * len(healthy)))
    # three disjoint HEALTHY splits: readout fit / sigma+normalisation (held out from
    # the readout, 2608.02464 sec.3) / threshold validation (eq.6).
    val_ids = [healthy[i] for i in perm[:n_val]]
    sigma_ids = [healthy[i] for i in perm[n_val:n_val + n_sigma]]
    fit_ids = [healthy[i] for i in perm[n_val + n_sigma:]]
    if not fit_ids:  # never fit on nothing; give the readout the sigma split back
        fit_ids, sigma_ids = sigma_ids, []

    # (1) memoryless Delta-Mahalanobis baseline, fitted on the healthy FIT split
    baseline: Dict[str, Any] = {}
    for name, is_delta in (("delta_mahalanobis", True), ("mahalanobis", False)):
        per_channel_models = {ch: DeltaMahalanobis.fit([episodes[c][ch] for c in fit_ids],
                                                       delta=is_delta)
                              for ch in ESN_CHANNELS}
        paths = {conv: fuse_per_channel_max(
            {ch: np.maximum.accumulate(per_channel_models[ch].scores(episodes[conv][ch]))
             for ch in ESN_CHANNELS}) for conv in episodes}
        thr = threshold_from_validation([paths[c] for c in val_ids], beta=beta)
        baseline[name] = {
            "threshold_from_validation_only": thr,
            "alarms": {conv: (first_alarm(paths[conv], thr) if thr is not None else None)
                       for conv in episodes},
        }

    # (2) the ESN layer
    channels: Dict[str, EsnChannel] = {}
    for ch in ESN_CHANNELS:
        fitted = esn_channel_fit([episodes[c][ch] for c in fit_ids], seed=seed,
                                 sigma_mats=[episodes[c][ch] for c in sigma_ids])
        if fitted is not None:
            channels[ch] = fitted
    if not channels:
        return {"aborted": True, "reason": "no channel had >= 2 steps in the healthy fit split",
                "cpu_cost_note": ESN_CPU_COST_NOTE}
    paths_max: Dict[str, np.ndarray] = {}
    paths_mean: Dict[str, np.ndarray] = {}
    for conv, mats in episodes.items():
        per_ch = {ch: channels[ch].cusum(mats[ch], kappa=kappa) for ch in channels}
        paths_max[conv] = fuse_per_channel_max(per_ch)
        paths_mean[conv] = fuse_mean(per_ch)
    theta = threshold_from_validation([paths_max[c] for c in val_ids], beta=beta)
    theta_mean = threshold_from_validation([paths_mean[c] for c in val_ids], beta=beta)

    fp_fit = sum(1 for c in fit_ids
                 if theta is not None and first_alarm(paths_max[c], theta) is not None)
    fp_val = sum(1 for c in val_ids
                 if theta is not None and first_alarm(paths_max[c], theta) is not None)
    fp_sigma = sum(1 for c in sigma_ids
                   if theta is not None and first_alarm(paths_max[c], theta) is not None)
    sigma_in_sample = any(c.sigma_in_sample for c in channels.values())
    return {
        "aborted": False,
        "n_episodes": len(episodes),
        "n_healthy": len(healthy),
        "n_healthy_fit": len(fit_ids),
        "n_healthy_sigma": len(sigma_ids),
        "n_healthy_val": len(val_ids),
        "sigma_err_source": {
            "held_out": not sigma_in_sample,
            "n_episodes": len(sigma_ids),
            "per_channel": {ch: channels[ch].sigma_source for ch in channels},
            "note": ("2608.02464 sec.3 measures sigma_err (eq.4) on held-out healthy "
                     "runs; the same held-out split supplies the (q_mean, q_std) the "
                     "CUSUM z-scores with. sigma_frac=0 or a pool too small to hold "
                     "an episode falls back to the fit residuals and sets "
                     "held_out=False."),
        },
        "beta": beta,
        "kappa": kappa,
        "channels": sorted(channels),
        "theta_per_channel_max": theta,
        "theta_mean_fusion_contrast": theta_mean,
        "alarms": {conv: (first_alarm(paths_max[conv], theta) if theta is not None else None)
                   for conv in episodes},
        "false_positives_raw": {
            "healthy_fit_split_IN_SAMPLE": f"{fp_fit}/{len(fit_ids)}",
            "healthy_validation_split_THRESHOLD_SOURCE": f"{fp_val}/{len(val_ids)}",
            "healthy_sigma_split_UNSEEN_BY_READOUT_AND_THRESHOLD":
                (f"{fp_sigma}/{len(sigma_ids)}" if sigma_ids else None),
            "note": ("2608.02464 eq.6 reads theta from the healthy VALIDATION episodes "
                     "and reports false alarms on unseen episodes. The fit count is "
                     "in-sample for the readout, the validation count is in-sample for "
                     "the threshold; the sigma split is unseen by BOTH but is the pool "
                     "sigma_err and (q_mean, q_std) were measured on, so it is not a "
                     "clean held-out false-alarm rate either. Raw counts with their "
                     "denominators, as the paper reports (0/63, 0 of 1825)."),
        },
        "baseline": baseline,
        "u_channel": "ABSENT BY DESIGN (2608.02464 ablates its surprisal channel to +0.000)",
        "exclusion_class": list(ESN_EXCLUDED_ERROR_KINDS),
        "cpu_cost_note": ESN_CPU_COST_NOTE,
        "healthy_definition": "bench TASK SUCCESS only (never 'compressed matched full')",
    }


# ==========================================================================
# battery-face feature assembly
# ==========================================================================

def _parse_tool_name(text: str) -> Optional[str]:
    from t33_spanmap import parse_tool_call
    parsed = parse_tool_call(text or "")
    name = parsed.get("name") if isinstance(parsed, dict) else None
    return name or None


class CaptureUnavailable(RuntimeError):
    """Raised when --capture_dir is given but the u-channel dump is not there."""


def capture_entropy_availability(capture_dir: Optional[Path], arm: str) -> Dict[str, Any]:
    """Named status of the (B) ``u`` channel input - never a silent absence.

    The u channel of 2607.11317 needs the capture rerun (RUNBOOK step 0).  Without
    it every ``ecusum_u_*`` and ``ecusum_a_fused`` column is null BY DATA, and this
    dict says so under its own key rather than leaving the caller to infer it from
    a column of nulls.
    """
    if capture_dir is None:
        return {
            "u_channel_available": False,
            "reason": "no --capture_dir given; the capture rerun (RUNBOOK step 0) "
                      "has not landed on this machine",
            "path": None,
            "n_qids": 0,
            "entropy_source": None,
        }
    path = Path(capture_dir) / arm / "p0.steps.jsonl"
    return {
        "u_channel_available": bool(path.exists()),
        "reason": None if path.exists() else f"{path} does not exist",
        "path": str(path),
        "n_qids": 0,
        "entropy_source": None,
    }


def _capture_step_entropies(capture_dir: Optional[Path], arm: str,
                            *, status: Optional[Dict[str, Any]] = None
                            ) -> Dict[str, List[Optional[float]]]:
    """qid -> per-token entropy from the t33 capture dump (RUNBOOK step 0).

    ``entropy_full`` (t33_capture.py:240) is preferred; when a record carries only
    ``top5`` / ``top_logprobs`` the TRUNCATED entropy is used instead and the
    status dict says which - the two are not the same quantity and are never
    reported under one name.

    A ``capture_dir`` that does not hold the dump raises :class:`CaptureUnavailable`
    instead of returning an empty map: silently degrading to a null u channel while
    the caller believes it passed the capture in is the failure mode this unit's
    fused column exists to avoid.
    """
    if capture_dir is None:
        return {}
    path = Path(capture_dir) / arm / "p0.steps.jsonl"
    if not path.exists():
        raise CaptureUnavailable(
            f"--capture_dir given but {path} does not exist; the (B) u channel is "
            "unavailable. Re-run without --capture_dir to get the honestly-named "
            "rep-only columns (ecusum_a_fused stays null), or land the capture "
            "rerun first (RUNBOOK step 0).")
    out: Dict[str, List[Optional[float]]] = {}
    sources = {"entropy_full": 0, "truncated_top_logprobs": 0, "missing": 0}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            vals: List[Optional[float]] = []
            for s in (rec.get("steps") or []):
                if s.get("entropy_full") is not None:
                    sources["entropy_full"] += 1
                    vals.append(float(s["entropy_full"]))
                    continue
                top = s.get("top5") or s.get("top_logprobs")
                if top:
                    sources["truncated_top_logprobs"] += 1
                    vals.append(truncated_entropy(top))
                else:
                    sources["missing"] += 1
                    vals.append(None)
            out[rec["qid"]] = vals
    if status is not None:
        status["n_qids"] = len(out)
        status["entropy_source"] = sources
        status["u_channel_available"] = bool(
            sources["entropy_full"] or sources["truncated_top_logprobs"])
        if not status["u_channel_available"]:
            status["reason"] = (f"{path} exists but carries neither entropy_full nor "
                                "top_logprobs on any step")
    return out


def build_sequential_features(frame: FrozenFrame, *, arm: str = "c2kv",
                              capture_dir: Optional[Path] = None,
                              sidecar: Optional[Dict[str, Dict[str, Any]]] = None,
                              fit_qids: Optional[Sequence[str]] = None,
                              capture_status: Optional[Dict[str, Any]] = None
                              ) -> List[Dict[str, Any]]:
    """Per-qid scalar features for the shared winner table.

    Only the named arm's own row / text / prefix scalars are read (rule 4).  With
    ``arm='full'`` this is the S0 twin control and must be written to its own file.
    All cross-step columns are CAUSAL (they read steps < t of the same session only)
    and are None for the first step of a session, with the denominator reported by
    ``seq_n_prior_steps``.

    ``fit_qids`` restricts the pool the physiology Gaussian and the prefix
    normalisation constants are fitted on (2608.27808 sec.4: those constants are
    "fit only on the fit set and then frozen").  ``None`` fits on every C->C row,
    which is in-sample for any C->C row that is later scored - use it only for the
    descriptive feature dump, never for a calibration/floor claim; the emitted
    ``cura_constants_out_of_fold`` column is 0.0 on exactly those rows.  The
    session-grouped out-of-fold wrapper is
    :func:`build_sequential_features_out_of_fold`, which is what the ``features``
    subcommand runs.
    """
    rows_by_qid = frame.c2kv_by_qid if arm == "c2kv" else frame.full_by_qid
    ent_by_qid = _capture_step_entropies(capture_dir, arm, status=capture_status)

    # FIT pool = C->C rows of THIS arm (healthy, no failure labels)
    cc = list(fit_qids) if fit_qids is not None else frame.cc_qids()
    phys_rows = [rows_by_qid[q] for q in cc if q in rows_by_qid]
    phys_tools = [_parse_tool_name(rows_by_qid[q].get("prediction", "")) for q in cc
                  if q in rows_by_qid]
    phys = physiology_fit(phys_rows, phys_tools)
    sig_cal = [prefix_signals(rows_by_qid[q], (sidecar or {}).get(q)) for q in cc
               if q in rows_by_qid]
    norm = prefix_norm_constants(sig_cal)
    # sessions the constants above were fitted on: a row from one of them is
    # IN-SAMPLE for cura_phys_* / cura_prefix_*, and says so per row.
    fit_sessions_used = {session_of(q) for q in cc if q in rows_by_qid}

    # order every session causally
    by_session: Dict[str, List[str]] = {}
    for qid in rows_by_qid:
        by_session.setdefault(session_of(qid), []).append(qid)
    for sid in by_session:
        by_session[sid].sort(key=step_index)

    out: List[Dict[str, Any]] = []
    for sid, qids in by_session.items():
        step_ent_means: List[Optional[float]] = []
        for qid in qids:
            ents = [e for e in ent_by_qid.get(qid, []) if e is not None]
            step_ent_means.append(float(np.mean(ents)) if ents else None)
        u_rel = entropy_spike_relative(step_ent_means)
        u_abs = entropy_spike_absolute(step_ent_means)

        r_steps: List[float] = []
        for qid in qids:
            toks = tokenize_words(rows_by_qid[qid].get("prediction", ""))
            cov = causal_repeat_coverage(toks)
            r_steps.append(float(cov.max()) if cov.size else 0.0)

        a_rep = [alarm_score_rep_only(r) for r in r_steps]
        a_full = [alarm_score(r, u) for r, u in zip(r_steps, u_rel)]

        sigs = [prefix_signals(rows_by_qid[q], (sidecar or {}).get(q)) for q in qids]
        s_steps: List[Optional[float]] = []
        used_steps: List[int] = []
        for sig in sigs:
            s, used = prefix_step_score(sig, norm)
            s_steps.append(s)
            used_steps.append(used)
        w_path = prefix_gate_path(s_steps)

        # causal running physiology mean/max over the session
        nll_hist: List[float] = []
        for i, qid in enumerate(qids):
            row = rows_by_qid[qid]
            tool = _parse_tool_name(row.get("prediction", ""))
            nll, seen = (phys.nll(row, tool) if phys is not None else (None, False))
            if nll is not None:
                nll_hist.append(nll)
            out.append({
                "qid": qid,
                "arm": arm,
                "session_id": sid,
                "seq_step_index": step_index(qid),
                "seq_n_prior_steps": i,
                # --- (B) e-CUSUM channels
                "ecusum_r": r_steps[i],
                "ecusum_a_rep_only": a_rep[i],
                "ecusum_a_fused": a_full[i],
                "ecusum_u_rel": u_rel[i],
                "ecusum_u_abs": u_abs[i],
                # --- (C) CURA physiology + prefix gate
                "cura_phys_nll": nll,
                "cura_phys_nll_mean_causal": (float(np.mean(nll_hist)) if nll_hist else None),
                "cura_phys_nll_max_causal": (float(np.max(nll_hist)) if nll_hist else None),
                "cura_phys_tool_seen": (1.0 if seen else 0.0),
                "cura_prefix_S": s_steps[i],
                "cura_prefix_W": w_path[i],
                "cura_prefix_n_signals": float(used_steps[i]),
                # provenance of the FITTED constants of the two cura channels
                "cura_constants_out_of_fold": (
                    1.0 if (fit_sessions_used and sid not in fit_sessions_used) else 0.0),
            })
    out.sort(key=lambda r: r["qid"])
    return out


#: folds of the session-grouped out-of-fold feature dump.  Three folds = the
#: three-way session split of 2608.27808 sec.5, rotated so that every row is
#: scored by constants fitted on the other two thirds.
FEATURES_OOF_FOLDS = 3


def build_sequential_features_out_of_fold(
        frame: FrozenFrame, *, arm: str = "c2kv",
        capture_dir: Optional[Path] = None,
        sidecar: Optional[Dict[str, Dict[str, Any]]] = None,
        n_folds: int = FEATURES_OOF_FOLDS, seed: int = 20260905,
        capture_status: Optional[Dict[str, Any]] = None
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The features dump with every FITTED column computed OUT OF FOLD.

    2608.27808 sec.4 fits the physiology Gaussian and the step-score normalisation
    on the fit split only; sec.5 refits them per replicate.  Building them once on
    every C->C row makes ``cura_phys_nll*`` and ``cura_prefix_S/W`` in-sample for
    every C->C row in the dump, and any downstream winner table that scores those
    columns inherits the leakage.  Here the sessions are cut into ``n_folds``
    grouped folds; for each fold the constants are fitted on the C->C rows of the
    OTHER folds' sessions and only the held-out fold's rows are kept, so no row
    contributed to its own normalisation.

    Rows whose fold had an empty C->C fit pool keep their unfitted columns (the
    ecusum / seq_* channels, which are frozen row facts) and get None - never a
    zero - for the fitted ones; ``cura_constants_out_of_fold`` is 0.0 there and the
    returned provenance dict counts them.
    """
    rows_by_qid = frame.c2kv_by_qid if arm == "c2kv" else frame.full_by_qid
    all_qids = sorted(rows_by_qid)
    sessions = np.array([session_of(q) for q in all_qids])
    cc = set(frame.cc_qids())
    folds = grouped_folds(sessions, n_folds, seed)
    by_qid: Dict[str, Dict[str, Any]] = {}
    prov: Dict[str, Any] = {
        "n_folds": n_folds, "seed": seed, "unit": "session",
        "folds": [], "n_rows_without_fit_pool": 0,
        "rule": ("cura_phys_* and cura_prefix_* constants are fitted on the C->C "
                 "rows of the OTHER folds' sessions only (2608.27808 sec.4/5)"),
    }
    for f, mask in enumerate(folds):
        held_sessions = {sessions[i] for i in np.flatnonzero(np.asarray(mask, dtype=bool))}
        if not held_sessions:
            continue
        fit_cc = [q for q in all_qids if q in cc and session_of(q) not in held_sessions]
        recs = build_sequential_features(frame, arm=arm, capture_dir=capture_dir,
                                         sidecar=sidecar, fit_qids=fit_cc,
                                         capture_status=capture_status)
        n_kept = 0
        for rec in recs:
            if rec["session_id"] in held_sessions:
                by_qid[rec["qid"]] = rec
                n_kept += 1
        prov["folds"].append({
            "fold": f, "n_heldout_sessions": len(held_sessions),
            "n_heldout_rows": n_kept, "n_fit_cc_rows": len(fit_cc),
            "fit_pool_empty": not fit_cc,
        })
        if not fit_cc:
            prov["n_rows_without_fit_pool"] += n_kept
    missing = [q for q in all_qids if q not in by_qid]
    if missing:  # defensive: grouped_folds covers every session, but never drop rows
        for rec in build_sequential_features(frame, arm=arm, capture_dir=capture_dir,
                                             sidecar=sidecar, fit_qids=[],
                                             capture_status=capture_status):
            if rec["qid"] in set(missing):
                by_qid[rec["qid"]] = rec
        prov["n_rows_without_fit_pool"] += len(missing)
    rows = [by_qid[q] for q in all_qids if q in by_qid]
    prov["n_rows"] = len(rows)
    prov["n_rows_out_of_fold"] = sum(1 for r in rows
                                     if r.get("cura_constants_out_of_fold") == 1.0)
    return rows, prov


SEQ_ORIENTATIONS: Dict[str, int] = {
    # +1 = higher is riskier (declared BEFORE any scoring)
    "ecusum_r": +1,
    "ecusum_a_rep_only": +1,
    "ecusum_a_fused": +1,
    "ecusum_u_rel": +1,
    "ecusum_u_abs": +1,
    "cura_phys_nll": +1,
    "cura_phys_nll_mean_causal": +1,
    "cura_phys_nll_max_causal": +1,
    "cura_phys_tool_seen": -1,      # an unseen tool is the anomalous case
    "cura_prefix_S": +1,
    "cura_prefix_W": +1,
    "cura_prefix_n_signals": +1,    # nuisance / availability control, direction declared
    # provenance control: 0.0 = the cura constants were fitted on this row's own
    # session (in-sample, deflated NLL / S_t), 1.0 = out of fold.  The riskier-
    # looking reading is the in-sample one, hence -1.
    "cura_constants_out_of_fold": -1,
    "seq_step_index": +1,           # position nuisance control
    "seq_n_prior_steps": +1,        # cross-step denominator control
}


# ==========================================================================
# CLI
# ==========================================================================

def _load_sidecar(path: Optional[str]) -> Optional[Dict[str, Dict[str, Any]]]:
    if not path:
        return None
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["qid"]] = rec
    return out


def _cmd_qp(args: argparse.Namespace) -> int:
    frame = FrozenAssets(args.root).load()
    diag = qp_persistence_diagnostic(frame, reps=args.reps)
    diag["deviations"] = DEVIATIONS
    if args.out:
        freeze_json(Path(args.out), diag)
    print(json.dumps({k: diag[k] for k in
                      ("transition", "ci95", "steps_per_session",
                       "sequential_licensed", "verdict")}, indent=1))
    return 0


def _cmd_ltt_grid(args: argparse.Namespace) -> int:
    grid = ltt_alpha_grid(args.n, delta=args.delta)
    if args.out:
        freeze_json(Path(args.out), grid)
    print(json.dumps(grid, indent=1))
    return 0


def ecusum_calibrate_and_evaluate(frame: FrozenFrame, *, tau: float,
                                  capture_dir: Optional[Path] = None,
                                  seed: int = 20260905) -> Dict[str, Any]:
    """mu_0 on a calibration half of C->C, achieved false alarm on the HELD-OUT half.

    2607.11317 sec.3.3 calibrates ``mu_0`` on healthy reference traces; sec.7 admits the
    split was not strict in the paper, so ours is: the C->C sessions are cut in two by
    SESSION (never by row), mu_0 comes from the calibration half only, and the achieved
    false-alarm rate is measured on the held-out half beside the nominal delta.

    "Healthy session" is the strict reading: a session with at least one C->C row and
    NO C->W row.  Both denominators are returned.
    """
    rows = frame.c2kv_by_qid
    cc = set(frame.cc_qids())
    cw_sessions = {session_of(q) for q in frame.cw_qids()}
    cc_sessions = sorted({session_of(q) for q in cc})
    pure_healthy = [s for s in cc_sessions if s not in cw_sessions]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pure_healthy))
    half = len(pure_healthy) // 2
    cal_sessions = {pure_healthy[i] for i in perm[:half]}
    held_sessions = [pure_healthy[i] for i in perm[half:]]

    # mu_0 companion, in the PAPER's unit: per-token a_t over the calibration half.
    pool_tok: List[float] = []
    for qid in sorted(cc):
        if session_of(qid) not in cal_sessions:
            continue
        toks = tokenize_words(rows[qid].get("prediction", ""))
        pool_tok.extend(alarm_score_rep_only(v) for v in causal_repeat_coverage(toks))
    cal_token_unit = mu0_from_pool(
        pool_tok, unit="per-token a_t over the C->C CALIBRATION-half rows "
                       "(the paper's unit; reported as a companion only)")

    # a_n = alarm_score_rep_only(r_n) is a frozen row fact (no fitted constant), so
    # the feature build runs with an EMPTY fit pool: nothing here is fitted on any
    # C->C row, and the cura_* columns it would otherwise carry stay None.
    capture_status = capture_entropy_availability(capture_dir, "c2kv")
    feats = build_sequential_features(frame, arm="c2kv", capture_dir=capture_dir,
                                      fit_qids=[], capture_status=capture_status)
    by_session: Dict[str, List[Tuple[int, float]]] = {}
    for rec in feats:
        by_session.setdefault(rec["session_id"], []).append(
            (rec["seq_step_index"], rec["ecusum_a_rep_only"]))
    a_by_session = {sid: [a for _, a in sorted(v, key=lambda t: t[0])]
                    for sid, v in by_session.items()}  # NUMERIC step order, not lexicographic

    # mu_0 ACTUALLY USED: the same unit as the accumulated statistic.  Our a_n is a
    # per-STEP score (the max of the step's causal per-token r_t stream), so a 90th
    # percentile of per-TOKEN a_t is not an upper bound on the healthy conditional
    # mean of a_n (2607.11317 sec.3.3 requires mu_0 to bound E[a_n | F_{n-1}]).
    pool_step: List[float] = []
    for sid in sorted(cal_sessions):
        pool_step.extend(v for v in a_by_session.get(sid, []) if v is not None)
    cal = mu0_from_pool(
        pool_step, unit="per-STEP a_n (max over the step's causal per-token r_t "
                        "stream) over the C->C CALIBRATION-half sessions - the unit "
                        "of the statistic that is accumulated")

    if cal["mu0"] is None:
        return {
            "calibration": cal,
            "calibration_paper_unit_companion": cal_token_unit,
            "evaluation": {"not_calibrated": True,
                           "reason": ("the calibration half produced no per-step a_n; "
                                      "mu_0 is undefined and is NOT back-filled with 0 "
                                      "(a mu_0 of 0 would make the detector fire on "
                                      "every trace)")},
            "denominators": {
                "n_cc_rows": len(cc),
                "n_cc_sessions": len(cc_sessions),
                "n_pure_healthy_sessions": len(pure_healthy),
                "n_calibration_sessions": len(cal_sessions),
                "n_heldout_sessions": len(held_sessions),
            },
            "lambda": ECUSUM_LAMBDA,
            "w_rep": ECUSUM_W_REP,
            "w_ent": ECUSUM_W_ENT,
            "u_channel_present": bool(capture_status.get("u_channel_available")),
        "u_channel_status": capture_status,
            "note": CAP_CONFOUND_NOTE,
        }

    ev = ecusum_evaluate(a_by_session, held_sessions, mu0=float(cal["mu0"]), tau=tau)
    return {
        "calibration": cal,
        "calibration_paper_unit_companion": cal_token_unit,
        "evaluation": ev,
        "denominators": {
            "n_cc_rows": len(cc),
            "n_cc_sessions": len(cc_sessions),
            "n_pure_healthy_sessions": len(pure_healthy),
            "n_calibration_sessions": len(cal_sessions),
            "n_heldout_sessions": len(held_sessions),
        },
        "lambda": ECUSUM_LAMBDA,
        "w_rep": ECUSUM_W_REP,
        "w_ent": ECUSUM_W_ENT,
        "u_channel_present": bool(capture_status.get("u_channel_available")),
        "u_channel_status": capture_status,
        "note": CAP_CONFOUND_NOTE,
    }


def _cmd_ecusum(args: argparse.Namespace) -> int:
    frame = FrozenAssets(args.root).load()
    result = ecusum_calibrate_and_evaluate(
        frame, tau=args.tau,
        capture_dir=Path(args.capture_dir) if args.capture_dir else None)
    result["deviations"] = DEVIATIONS
    if args.out:
        freeze_json(Path(args.out), result)
    print(json.dumps({k: result[k] for k in ("calibration", "evaluation",
                                             "denominators")}, indent=1))
    return 0


def cura_fold_internal_floor(frame: FrozenFrame, qids: Sequence[str],
                             y: np.ndarray, sessions: Dict[str, str], *,
                             sidecar: Optional[Dict[str, Dict[str, Any]]] = None,
                             n_folds: int = 5, seed: int = 20260905,
                             columns: Sequence[str] = ("cura_prefix_S", "cura_prefix_W",
                                                       "cura_phys_nll",
                                                       "cura_phys_nll_max_causal"),
                             ) -> Dict[str, Any]:
    """Fold-internal floor with the FEATURES themselves refitted inside each fold.

    2608.27808 sec.4/5 fit the physiology Gaussian and the normalisation constants on
    the fit split only.  Building the columns once on every C->C row and then running
    a nested CV over them leaks the negative class of the evaluation frame into the
    features, which is exactly what a floor is supposed to exclude - so here every
    outer fold refits the physiology model and the prefix (mu_j, sigma_j) on the
    TRAINING sessions' C->C rows alone, rebuilds the columns, and fits the logistic
    readout on the training rows only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    groups = np.array([sessions[q] for q in qids])
    folds = grouped_folds(groups, n_folds, seed)
    label_by_qid = frame.label_by_qid
    oof = np.full(len(qids), np.nan, dtype=float)
    idx_of = {q: i for i, q in enumerate(qids)}
    for mask in folds:
        mask = np.asarray(mask, dtype=bool)
        test_idx = np.flatnonzero(mask)
        if test_idx.size == 0:
            continue
        test_set = {qids[i] for i in test_idx}
        train_qids = [q for q in qids if q not in test_set]
        fit_cc = [q for q in train_qids if label_by_qid.get(q) == 0]
        if not fit_cc:
            continue
        feats = {r["qid"]: r for r in build_sequential_features(
            frame, arm="c2kv", sidecar=sidecar, fit_qids=fit_cc)}

        def _mat(qs: Sequence[str]) -> np.ndarray:
            return np.array([[feats[q][c] if feats[q].get(c) is not None else np.nan
                              for c in columns] for q in qs], dtype=float)

        Xtr, Xte = _mat(train_qids), _mat([qids[i] for i in test_idx])
        ytr = np.array([y[idx_of[q]] for q in train_qids], dtype=int)
        keep = np.isfinite(Xtr).all(axis=1)
        if keep.sum() < 4 or len(set(ytr[keep].tolist())) < 2:
            continue
        med = np.nanmedian(Xtr[keep], axis=0)
        Xte = np.where(np.isfinite(Xte), Xte, med)
        scaler = StandardScaler().fit(Xtr[keep])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(Xtr[keep]),
                                                           ytr[keep])
        oof[test_idx] = clf.decision_function(scaler.transform(Xte))
    ok = np.isfinite(oof)
    return {
        "auprc": average_precision(oof[ok], y[ok]) if ok.any() else None,
        "auroc": auroc(oof[ok], y[ok]) if ok.any() else None,
        "n_scored": int(ok.sum()),
        "n_pos_scored": int(y[ok].sum()) if ok.any() else 0,
        "prevalence": prevalence(y[ok]) if ok.any() else None,
        "n_folds": n_folds,
        "refit_inside_fold": ("physiology Gaussian + prefix (mu_j, sigma_j) + scaler + "
                              "logistic readout, all on the training sessions only"),
    }


def _cmd_cura(args: argparse.Namespace) -> int:
    frame = FrozenAssets(args.root).load()
    sidecar = _load_sidecar(args.sidecar)
    sub = frame.trigger_subset()
    qids = [r["qid"] for r in sub]
    y = np.array([r["label_cw"] for r in sub], dtype=int)
    sessions = {r["qid"]: r["session_id"] for r in sub}
    cc = [q for q in qids if frame.label_by_qid[q] == 0]

    # 2608.27808 sec.4: "Successful trajectories are split into disjoint fit and
    # calibration sets BEFORE threshold selection.  Physiology models, ...,
    # normalization, and step-score constants are fit only on the fit set."
    rng = np.random.default_rng(20260905)
    cc_sessions = sorted({sessions[q] for q in cc})
    perm = rng.permutation(len(cc_sessions))
    n_fit = len(cc_sessions) // 2
    fit_sessions = {cc_sessions[i] for i in perm[:n_fit]}
    fit_cc = [q for q in cc if sessions[q] in fit_sessions]
    cal_cc = [q for q in cc if sessions[q] not in fit_sessions]

    # The prefix SIGNALS are frozen row facts and do not depend on any fit; only the
    # (mu_j, sigma_j) do.  Compute the signals once and refit the constants per call
    # so the 200-replicate three-way split is affordable without ever reusing a
    # normalisation fitted outside its own fit third.
    rows_by_qid = frame.c2kv_by_qid
    all_qids = sorted(rows_by_qid)
    sig_by_qid = {q: prefix_signals(rows_by_qid[q], (sidecar or {}).get(q))
                  for q in all_qids}
    sess_order: Dict[str, List[str]] = {}
    for q in all_qids:
        sess_order.setdefault(session_of(q), []).append(q)
    for sid in sess_order:
        sess_order[sid].sort(key=step_index)

    cc_set = set(cc)

    def _scores(fit_sessions_arg: Sequence[str]) -> Dict[str, Optional[float]]:
        """``fit_sessions_arg`` are SESSION ids (the grouping unit of every split)."""
        fit = set(fit_sessions_arg)
        fit_pool = [q for q in cc_set if sessions[q] in fit]
        norm = prefix_norm_constants([sig_by_qid[q] for q in fit_pool
                                      if q in sig_by_qid])
        out: Dict[str, Optional[float]] = {}
        for sid, qs in sess_order.items():
            steps = [prefix_step_score(sig_by_qid[q], norm)[0] for q in qs]
            for q, w in zip(qs, prefix_gate_path(steps)):
                out[q] = w
        return {q: out.get(q) for q in qids}

    score_fit = _scores(sorted(fit_sessions))
    cal_scores = [score_fit[q] for q in cal_cc if score_fit[q] is not None]
    grid = ltt_alpha_grid(len(cal_scores))
    ltt = {str(a): ltt_select_threshold(cal_scores, alpha=a) for a in (0.05, 0.10, 0.20)}
    for a in ltt:
        ltt[a]["fit_calibration_disjoint"] = True
        ltt[a]["n_fit_sessions"] = len(fit_sessions)
        ltt[a]["n_fit_rows"] = len(fit_cc)
        ltt[a]["n_calibration_rows"] = len(cal_scores)
    three = {str(a): ltt_three_way(None, sessions, alpha=a, repeats=args.repeats,
                                   score_fn=_scores, qids=cc)
             for a in (0.05, 0.10, 0.20)}

    # the score used for the matched-FPR contrast: constants frozen on the fit half,
    # and the contrast itself runs on the rows OUTSIDE that half only, so no row it
    # scores contributed to its own normalisation.  matched_fpr_comparison then cuts
    # those rows again into its own session-grouped calibration / held-out halves.
    score = score_fit
    length = np.array([float(frame.c2kv_by_qid[q].get("generated_tokens") or 0.0)
                       for q in qids], dtype=float)
    comp = np.array([(score[q] if score[q] is not None else np.nan) for q in qids],
                    dtype=float)
    grp = np.array([sessions[q] for q in qids])
    outside_fit = np.array([sessions[q] not in fit_sessions for q in qids], dtype=bool)
    ok = np.isfinite(comp) & outside_fit
    if ok.any() and len(set(y[ok].tolist())) == 2:
        matched = matched_fpr_comparison(comp[ok], length[ok], y[ok],
                                         target_fpr=0.10, groups=grp[ok])
    else:
        matched = {
            "target_fpr": 0.10, "tol": 0.02, "matched": False,
            "n": int(ok.sum()), "n_pos": int(y[ok].sum()) if ok.any() else 0,
            "unmatched_note": ("no comparison: the rows outside the CURA fit half are "
                               "one-class or empty; a matched-FPR contrast has no "
                               "denominator here"),
        }
    floor = cura_fold_internal_floor(frame, qids, y, sessions, sidecar=sidecar)
    # a certificate that fires on nothing is not a detector: say so at the top level
    ltt_summary = {
        str(a): {
            "certified": ltt[str(a)]["certified"],
            "fires_on_nothing": bool(ltt[str(a)]["theta_above_all_calibration_scores"]),
            "usable": bool(ltt[str(a)]["certified"]
                           and not ltt[str(a)]["theta_above_all_calibration_scores"]),
            "n_calibration_rows": len(cal_scores),
            "n_required_for_alpha": next(r["n_required_for_alpha"]
                                         for r in grid["grid"] if r["alpha"] == a),
        } for a in (0.05, 0.10, 0.20)
    }
    result = {
        "alpha_grid": grid,
        "ltt": ltt,
        "ltt_usability": ltt_summary,
        "three_way": three,
        "matched_fpr_10pct": {
            "composite": matched.get("a"),
            "length_only_baseline": matched.get("b"),
            "target_fpr": matched["target_fpr"],
            "tol": matched["tol"],
            "matched": matched["matched"],
            "unmatched_note": matched.get("unmatched_note"),
            "frame": {"n": matched["n"], "n_pos": matched["n_pos"]},
            "split": matched.get("split"),
            "threshold_selected_on": matched.get("threshold_selected_on"),
            "calibration_resolution_note": matched.get("calibration_resolution_note"),
            "evaluation_resolution_note": matched.get("evaluation_resolution_note"),
            "rows_used": ("trigger-subset rows OUTSIDE the CURA fit half "
                          f"({int(ok.sum())} of {len(qids)}); the fit half's rows are "
                          "in-sample for the prefix normalisation"),
        },
        "fold_internal_floor": floor,
        "prefix_signs": CURA_PREFIX_SIGNS,
        "cap_confound": CAP_CONFOUND_NOTE,
        "caliber_requirement": "must be re-run at the 4096 caliber before publication",
        "deviations": DEVIATIONS,
    }
    if args.out:
        freeze_json(Path(args.out), result)
    print(json.dumps({"alpha_grid": grid, "ltt_alpha_0.10": ltt["0.1"],
                      "matched_fpr": result["matched_fpr_10pct"],
                      "fold_internal_floor": floor}, indent=1))
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    frame = FrozenAssets(args.root).load()
    capture_dir = Path(args.capture_dir) if args.capture_dir else None
    status = capture_entropy_availability(capture_dir, args.arm)
    sidecar = _load_sidecar(args.sidecar)
    if args.in_sample_dump:
        rows = build_sequential_features(frame, arm=args.arm, sidecar=sidecar,
                                         capture_dir=capture_dir,
                                         capture_status=status)
        prov = {"mode": "IN-SAMPLE DESCRIPTIVE DUMP",
                "warning": ("cura_phys_* and cura_prefix_* are fitted on every C->C "
                            "row and are IN-SAMPLE for every C->C row in this file; "
                            "cura_constants_out_of_fold is 0.0 on those rows. Do not "
                            "score these columns in a winner table."),
                "n_rows": len(rows)}
    else:
        rows, prov = build_sequential_features_out_of_fold(
            frame, arm=args.arm, sidecar=sidecar, capture_dir=capture_dir,
            n_folds=args.oof_folds, capture_status=status)
        prov["mode"] = "OUT-OF-FOLD (session-grouped)"
    n = write_features_jsonl(Path(args.out), rows, context=f"t34 U7b features ({args.arm})")
    print(json.dumps({"wrote": n, "path": args.out, "provenance": prov,
                      "capture_u_channel": status}, indent=1))
    return 0


def _cmd_arl(args: argparse.Namespace) -> int:
    diag = json.loads(Path(args.gate).read_text(encoding="utf-8"))
    require_gate(diag, force=args.force)
    frame = FrozenAssets(args.root).load()
    sub = frame.trigger_subset()
    qids = [r["qid"] for r in sub]
    y = np.array([r["label_cw"] for r in sub], dtype=int)
    groups = np.array([r["session_id"] for r in sub])
    # the probe's prefix column is refitted out of fold (2608.27808 sec.4): built
    # once on every C->C row it would carry the evaluation frame's negatives.
    feats = {r["qid"]: r for r in
             build_sequential_features_out_of_fold(frame, arm="c2kv")[0]}
    X = np.array([[feats[q]["cura_prefix_S"] if feats[q]["cura_prefix_S"] is not None
                   else np.nan,
                   feats[q]["ecusum_a_rep_only"]] for q in qids], dtype=float)
    probe = nested_cv_logistic(X, y, groups)
    oof = probe["oof_scores"]
    ok = probe["scored_mask"]
    inc = {q: float(v) for q, v, m in zip(qids, oof, ok) if m}

    # 2606.12476 sec.3.3/4.1: k and the ARL null pool are CALIBRATION quantities and
    # must not be read off the rows the arms are then scored on.  The split is by
    # SESSION (the grouping unit of every split on this face); every arm below is
    # evaluated on the held-out sessions only.
    rng = np.random.default_rng(args.seed)
    scored_sessions = sorted({session_of(q) for q in inc})
    perm = rng.permutation(len(scored_sessions))
    n_cal = int(round(args.cal_frac * len(scored_sessions)))
    cal_sessions = {scored_sessions[i] for i in perm[:n_cal]}
    cal_qids = [q for q in qids if q in inc and session_of(q) in cal_sessions]
    ev_qids = [q for q in qids if q in inc and session_of(q) not in cal_sessions]
    k = cusum_reference_k([inc[q] for q in cal_qids],
                          [int(frame.label_by_qid[q]) for q in cal_qids])
    neg_pool = [inc[q] for q in cal_qids if frame.label_by_qid[q] == 0]
    pos_cal = [q for q in cal_qids if frame.label_by_qid[q] == 1]
    calibration = {
        "unit": "session", "seed": args.seed, "cal_frac": args.cal_frac,
        "n_sessions_scored": len(scored_sessions),
        "n_calibration_sessions": len(cal_sessions),
        "n_calibration_rows": len(cal_qids),
        "n_calibration_negatives": len(neg_pool),
        "n_calibration_positives": len(pos_cal),
        "n_evaluation_rows": len(ev_qids),
        "n_evaluation_negatives": sum(1 for q in ev_qids if frame.label_by_qid[q] == 0),
        "n_evaluation_positives": sum(1 for q in ev_qids if frame.label_by_qid[q] == 1),
        "min_null_pool": ARL_MIN_NULL_POOL,
        "null_pool_underpowered": bool(len(neg_pool) < ARL_MIN_NULL_POOL),
        "k_estimable": bool(neg_pool and pos_cal),
        "note": ("k = (mu_0+mu_1)/2 and the ARL null pool come from the calibration "
                 "sessions ONLY; the arms are scored on the disjoint held-out "
                 "sessions. With fewer than "
                 f"{ARL_MIN_NULL_POOL} calibration negatives the bisected threshold "
                 "has a resolution coarser than the ARL targets and the alignment "
                 "must not be read as a matched operating point."),
    }
    if not neg_pool or not pos_cal:
        out = {
            "gate": {"licensed": diag.get("sequential_licensed"),
                     "forced": bool(args.force)},
            "aborted": True,
            "reason": ("the calibration half is one-class: k = (mu_0+mu_1)/2 and the "
                       "ARL null pool are undefined and are NOT back-filled from the "
                       "evaluation rows"),
            "calibration": calibration,
            "deviations": DEVIATIONS,
        }
        if args.out:
            freeze_json(Path(args.out), out)
        print(json.dumps({k2: out[k2] for k2 in ("aborted", "reason", "calibration")},
                         indent=1))
        return 0

    # causal per-session ordering, built ONCE (positions index into these lists);
    # only the HELD-OUT sessions are scored.
    qids_by_session: Dict[str, List[str]] = {}
    for q in ev_qids:
        qids_by_session.setdefault(session_of(q), []).append(q)
    for sid in qids_by_session:
        qids_by_session[sid].sort(key=step_index)
    by_session = {sid: [inc[q] for q in qs] for sid, qs in qids_by_session.items()}
    max_len = max((len(v) for v in by_session.values()), default=0)
    aligned = align_thresholds_at_arl({"learned_cusum": neg_pool}, {"learned_cusum": k},
                                      max_episode_len=max_len, n_runs=args.n_runs)
    onset: Dict[str, Optional[int]] = {}
    last: Dict[str, int] = {}
    for sid, qs in qids_by_session.items():
        last[sid] = len(qs) - 1
        onset[sid] = next((i for i, q in enumerate(qs)
                           if frame.label_by_qid[q] == 1), None)
    h = aligned["detectors"]["learned_cusum"]["100"]["h"]
    # 2606.12476 sec.4.1 matches every detector at the SAME ARL by sweeping ITS OWN
    # threshold; the no-accumulation arm therefore gets its own h', not the CUSUM's.
    memoryless = {str(g): threshold_for_arl_memoryless(neg_pool, target_arl=float(g))
                  for g in ARL_TARGETS}
    h_step = memoryless["100"]["h"]
    arms = {
        "cusum": detection_delays(cusum_arm(by_session, k=k, h=h), onset, last),
        "per_step_threshold": detection_delays(per_step_threshold_arm(by_session, h_step),
                                               onset, last),
        "shuffled_order": detection_delays(
            cusum_arm(shuffle_within_sessions(by_session), k=k, h=h), onset, last),
    }
    pf_by_qid = {r["qid"]: bool(r["parse_fail_fire"]) for r in frame.labels}
    pf_fire: Dict[str, Optional[int]] = {
        sid: next((i for i, q in enumerate(qs) if pf_by_qid.get(q)), None)
        for sid, qs in qids_by_session.items()
    }
    arms["parse_fail_baseline"] = detection_delays(pf_fire, onset, last)
    fires = np.array([inc[q] >= h_step for q in ev_qids], dtype=bool)
    ylab = np.array([frame.label_by_qid[q] for q in ev_qids], dtype=int)
    result = {
        "gate": {"licensed": diag.get("sequential_licensed"), "forced": bool(args.force)},
        "k_reference": k,
        "k_pool": ("CALIBRATION sessions only (session-grouped half); every arm below "
                   "is scored on the disjoint held-out sessions. See `calibration` for "
                   "the denominators and the null_pool_underpowered flag."),
        "k_all_rows_companion_NOT_USED": cusum_reference_k(
            [inc[q] for q in qids if q in inc],
            [int(frame.label_by_qid[q]) for q in qids if q in inc]),
        "calibration": calibration,
        "arl_alignment": aligned,
        "arl_alignment_memoryless_arm": memoryless,
        "thresholds_used_at_arl_100": {"cusum": h, "per_step_threshold": h_step},
        "arms": arms,
        "per_step_false_fire_rate": per_step_false_fire_rate(fires, ylab),
        "episode_length_hist": {str(n): int(c) for n, c in
                                zip(*np.unique([len(v) for v in by_session.values()],
                                               return_counts=True))},
        "probe": {kk: probe[kk] for kk in ("auprc", "auroc", "n_scored",
                                           "n_pos_scored", "prevalence", "chosen")},
        "deviations": DEVIATIONS,
    }
    if args.out:
        freeze_json(Path(args.out), result)
    print(json.dumps({"k": k, "arms": arms,
                      "arl": aligned.get("note")}, indent=1))
    return 0


def _cmd_esn(args: argparse.Namespace) -> int:
    rows = load_proxy_log(Path(args.proxy_log))
    episodes, counts = episodes_from_proxy(rows)
    healthy = json.loads(Path(args.healthy).read_text(encoding="utf-8")) if args.healthy else []
    res = esn_monitor(episodes, healthy, beta=args.beta)
    res["row_counts"] = counts
    res["deviations"] = DEVIATIONS
    if args.out:
        freeze_json(Path(args.out), res)
    print(json.dumps({k: res[k] for k in res if k not in ("alarms", "baseline", "deviations")},
                     indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="t34_sequential",
        description="t34 U7b: survey 4.9 sequential family (2606.12476 / 2607.11317 / "
                    "2608.27808 / 2608.02464). Torch-free, zero GPU.")
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("qp", help="q/p persistence diagnostic - THE GATE, run first")
    q.add_argument("--root", default=".")
    q.add_argument("--reps", type=int, default=2000)
    q.add_argument("--out", default=None)
    q.set_defaults(func=_cmd_qp)

    g = sub.add_parser("ltt-grid", help="achievable alpha grid for the exact-binomial LTT")
    g.add_argument("--n", type=int, required=True, help="calibration pool size (68 = C->C)")
    g.add_argument("--delta", type=float, default=CURA_DELTA)
    g.add_argument("--out", default=None)
    g.set_defaults(func=_cmd_ltt_grid)

    e = sub.add_parser("ecusum", help="mu_0 calibration + nominal vs achieved false alarm")
    e.add_argument("--root", default=".")
    e.add_argument("--capture_dir", default=None)
    e.add_argument("--tau", type=float, default=math.log(1.0 / ECUSUM_DELTA_NOMINAL))
    e.add_argument("--out", default=None)
    e.set_defaults(func=_cmd_ecusum)

    c = sub.add_parser("cura", help="LTT threshold + physiology + prefix gate + length control")
    c.add_argument("--root", default=".")
    c.add_argument("--sidecar", default=None)
    c.add_argument("--repeats", type=int, default=200)
    c.add_argument("--out", default=None)
    c.set_defaults(func=_cmd_cura)

    f = sub.add_parser("features", help="per-qid feature frame (arm=c2kv, or full for S0)")
    f.add_argument("--root", default=".")
    f.add_argument("--arm", choices=["c2kv", "full"], default="c2kv")
    f.add_argument("--sidecar", default=None)
    f.add_argument("--capture_dir", default=None)
    f.add_argument("--oof_folds", type=int, default=FEATURES_OOF_FOLDS,
                   help="session-grouped folds the cura constants are refitted in")
    f.add_argument("--in_sample_dump", action="store_true",
                   help="fit the cura constants on ALL C->C rows (descriptive dump "
                        "only; the emitted columns are in-sample and must not be "
                        "scored in a winner table)")
    f.add_argument("--out", required=True)
    f.set_defaults(func=_cmd_features)

    a = sub.add_parser("arl", help="learned CUSUM at aligned ARL (needs the gate)")
    a.add_argument("--root", default=".")
    a.add_argument("--gate", required=True, help="seq_qp_diagnostic.json from `qp`")
    a.add_argument("--force", action="store_true",
                   help="run despite a negative gate (prereg violation; say so in the report)")
    a.add_argument("--n_runs", type=int, default=400)
    a.add_argument("--cal_frac", type=float, default=0.5,
                   help="session-grouped share used to fit k and the ARL null pool")
    a.add_argument("--seed", type=int, default=20260905)
    a.add_argument("--out", default=None)
    a.set_defaults(func=_cmd_arl)

    s = sub.add_parser("esn", help="bench-face ESN + Delta-Mahalanobis telemetry monitor")
    s.add_argument("--proxy_log", required=True)
    s.add_argument("--healthy", default=None, help="json list of task-SUCCESS conv_ids")
    s.add_argument("--beta", type=float, default=0.05)
    s.add_argument("--out", default=None)
    s.set_defaults(func=_cmd_esn)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
