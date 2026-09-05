# -*- coding: utf-8 -*-
"""t34 / U7a -- bench-side rescoring for digest section 4.9.

Four migrations, all torch-free and all zero-GPU rescorings of artefacts that
are already on disk:

  (A) BENCH2ROBUST / BTM        arXiv 2608.11977   solvability classes S1'/S2'/S3'
                                                   + the explicit/silent taxonomy
  (B) IRBench                   arXiv 2608.16370   (Q, C_R, C_E) cost triples and
                                                   the D*/R* prefix feature
  (C) Last Step Matters         arXiv 2608.29685   backward truncation, 5x6 grid,
                                                   path-switch analogue as a control
  (D) Context Equilibria        arXiv 2510.07777   dZ = a + b Z, Z* = -a/b, and the
                                                   MANDATORY b != -1 pre-gate

RUNBOOK (execution order; everything here runs on THIS box -- no GPU, no torch)
------------------------------------------------------------------------------
Run from the worktree root C:/Users/yl998/Documents/programming/c2kv/tmp/t34-migration
with PYTHONIOENCODING=utf-8.

  0. (server, already done elsewhere)  copy the D-line k-sweep flip table off the
     NPU box to results/t34/flip_table.jsonl and the decoded-doc sidecars to
     results/t34/sidecar_c2kv.jsonl  (unit U2 writes the dumper; U7a only reads).

  1. python agent/t34_bench_signals.py classes \
         --root . --flip-table results/t34/flip_table.jsonl \
         [--none-arm results/bdf_pilot/d_r2/<none-arm>.jsonl] \
         --out configs/t34/solvability_classes_bencha.json
     -> assigns S1'/S2'/S3' ONCE from frozen artefacts and freezes the manifest
        with its sha256.  Never re-run after a detector exists.

  2. python agent/t34_bench_signals.py l1-cut \
         --root . [--sidecar results/t34/sidecar_c2kv.jsonl]
     -> L1-visible vs silent cut + the 128-token caliber caveat.

  3. python agent/t34_bench_signals.py partition-check \
         --partition configs/t34/tool_partition_template.json
     -> validates the static tool partition (the runner fills the three buckets
        from the benchmark manifests; this file ships EMPTY on purpose).

  4. python agent/t34_bench_signals.py triples \
         --proxy-log <proxy_requests.jsonl> --partition configs/t34/tool_partition_template.json \
         --max-turn <fixed horizon> --family c2kv:full:C_R,hybrid:full:C_R \
         --out results/t34/irbench_triples_bencha.json
     -> per-arm (Q, C_R, C_E) from turn 1 to the declared horizon + paired
        Wilcoxon + Holm ACROSS the declared family of comparison cells.

  5. python agent/t34_bench_signals.py dstar \
         --root . --sidecar results/t34/sidecar_c2kv.jsonl \
         --partition configs/t34/tool_partition_template.json \
         --out results/t34/features_dstar_bencha.jsonl
     -> prefix-only D*/R* features (pre-generation, zero forward).
        REQUIRES a sidecar dumped with --with_dropped_text for the DROPPED half of
        s_t; without it the command prints n_rows_missing_dropped_doc_texts > 0,
        dropped_side_available=false, and every such row's irb_s_t_dropped_only is
        null with irb_s_t_scope='kept_only_partial'.

  6. python agent/t34_bench_signals.py truncation \
         --proxy-log <proxy_requests.jsonl> --labels <conv->label json> \
         --out results/t34/lsm_truncation_bencha.json
     -> AUROC vs progress + the 5x6 reducer grid + the base rate in the SAME table.

  7. python agent/t34_bench_signals.py selfconv --proxy-log <proxy_requests.jsonl>
     -> post-divergence self-convergence rate at t+k (SPEC 7.2 Q2 persistence K)
        AND the bench-face FULL-ROLLBACK rescue rate -- an UPPER BOUND on S1',
        never S1' itself (the recover arm re-prefills the raw history).

  8. python agent/t34_bench_signals.py equilibrium \
         --proxy-log <proxy_requests.jsonl> --z compression_ratio \
         --labels <conv->label json> [--kappa-grid 0.5,1.0,1.5,2.0,3.0] \
         [--kappa-folds 3] [--target-fire-rate <q>] \
         --out results/t34/ceq_bencha.json
     -> fits dZ = a + b Z on the C->C pool (--labels selects it; WITHOUT it the
        fit also uses the evaluation rows and stamps a warning saying so), runs
        the white-noise b != -1 pre-gate FIRST and refuses to report a band when
        the gate fails, then SELECTS kappa in inner grouped folds (precision at
        the declared fire rate) and emits the completed fire rule
        Z_t - Z* > kappa*sigma_eta.  Without --labels kappa stays null and the
        fire rule is reported as incomplete.  z_pre_coverage on the output says
        how many of the six Z^pre components this log actually carries; the
        command aborts with rc=2 when the chosen --z is null on every row.

  9. python agent/t34_bench_signals.py battery-z --root . \
         --out results/t34/features_zpre_bencha.jsonl
     -> the per-step (no t axis) form of Z^pre on the frozen 900 battery rows.

INPUT REQUIREMENTS -- what each command needs on a row, and who writes it
------------------------------------------------------------------------------
Two proxies exist and they are NOT the same file.  The bench branch's
``c2kv/benchmarks/proxy.py`` is the rich one; the ``benchmarks/proxy.py``
vendored in THIS worktree is a 344-line copy whose ``_log_request`` (:300-322)
writes only ``ts / arm / n_messages / n_tools / gist_tokens / original_tokens /
n_gist_messages / wall_sec / status / usage / finish_reason``.  Nothing below
that needs conv_id, turn, action or a recover flag can run against a log this
worktree's copy produced -- the commands report the missing columns rather than
returning a zero, but the number itself is unavailable until the rich proxy's
log is copied in.

  command      needs on each row                     written by
  -----------  -----------------------------------  ---------------------------
  triples      conv_id, arm, turn                   bench proxy _log_request
               action (tool_calls)                  ONLY the --record-reference
                                                    sidecar (join on ``fp``) --
                                                    rows without it land in
                                                    n_no_action
               finish_reason|error_kind|status      both proxies
               Q                                    NOT in any log: pass
                                                    completion_by_conv from the
                                                    benchmark harness or Q stays
                                                    null (a two-element triple)
  selfconv     conv_id, match, diverged_now,        bench proxy RecoverState
               repaired, repair_fidelity,           (absent here -> reported in
               recovered_action_match               columns_missing)
  truncation   conv_id, turn + per-step logprob     NOT written today
               columns (ppl / nll / entropy)        (normalize_response drops
                                                    logprobs) -> every row reads
                                                    reason='signal absent'
  equilibrium  conv_id + the chosen --z component   see z_pre_coverage: a bench
                                                    log carries 4 of the 6
                                                    Z^pre components
  dstar        sidecar docs / dropped_docs and,     agent/t34_dump_sidecar.py
               for the dropped half,                (--with_dropped_text)
               dropped_doc_texts
  classes      results/t34/flip_table.jsonl         the D-line per-(qid,k) sweep
                                                    on the NPU box

SIDECAR COORDINATE SYSTEMS (bites every reader once).  ``docs`` holds ONLY the
KEPT blocks, in the order the model saw them -- every entry of ``docs`` is
visible.  ``dropped_docs`` indexes the POST-SPLIT history list, a DIFFERENT list,
so it can never index ``docs``; a block is dropped precisely by NOT being in
``docs``.  Dropped text lives only under ``dropped_doc_texts``.

WIRING (bench face) -- this module never imports or edits anything under
benchmarks/.  The pure functions plug in as follows:

  * ``classify_observation(text)``            -> called on the tool-response
    message text inside the bench harness one turn AFTER the action, i.e. where
    ``benchmarks/proxy.py`` builds the next request's message list.  It is a
    read-only classifier; nothing in proxy.py changes.
  * ``cost_triples(...)``                     -> offline groupby over the rows
    ``_log_request`` already appends (c2kv/benchmarks/proxy.py:1338; the short
    proxy.py vendored in THIS worktree is an older 344-line copy with no
    conv_id / turn / action columns).  It needs the emitted action's tool name,
    written only to the --record-reference sidecar (c2kv/benchmarks/proxy.py:
    1254-1266).  So either join the request log to that sidecar on ``fp``, or
    add ``"action": action_canonical(...)`` to the row.  Both shapes are handled;
    rows with no turn column at all are counted, never silently dropped.
  * ``l1_visibility(...)``                    -> mirrors
    ``benchmarks/metrics.py:protocol_columns_for_turn`` /
    ``_schema_violations`` (:66) for the battery face, which has no proxy.  On
    the bench face pass ``schema_violation_fn=benchmarks.metrics._schema_violations``
    so there is exactly one implementation in play.
  * ``post_divergence_convergence(...)``      -> offline over the recover flags
    ``RecoverState.check`` (c2kv/benchmarks/proxy.py:494) merges into the log rows
    (``match / diverged_now / re_diverged / tracking_lost``).
  * ``z_pre_components(row)`` / equilibrium    -> HOOK 2, ``RecoverState.check``
    (c2kv/benchmarks/proxy.py:476/494): every component is a prefix scalar in the
    log row before generation, so the fire decision is causal.

ESTIMANDS.  Trigger features are per-qid scalars written through
``t34_common.write_features_jsonl`` with ``None`` (json null) where undefined --
never a sentinel.  Chance AP on the 161-row trigger subset is that subset's own
prevalence (0.578), never the 900-frame base rate 0.1033; the 0.1033 figure is
printed only as the ``base_rate_900`` reference column the Last-Step migration
requires in the same table.  Locators are not part of this unit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import t34_common as C  # noqa: E402
from d_witness_core import leaves as witness_leaves, target_values  # noqa: E402
from t33_spanmap import parse_tool_call  # noqa: E402

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "BTM solvability classes",
        "paper": "2608.11977",
        "what": "S1/S2/S3 are assigned from a FROZEN best-k flip table (+ an optional "
                "none-arm results file), not post hoc from the evaluated policy's own "
                "trajectory; S3' is scored, not training-only.",
        "why": "The paper states its own counts are policy-dependent and cannot isolate a "
               "causal effect (Table tab:strategy note).  Assigning once from an artefact "
               "that no detector produced removes the circularity; the class manifest is "
               "sha256-frozen so a detector can never redefine its own denominator.",
    },
    {
        "method": "BTM solvability classes",
        "paper": "2608.11977",
        "what": "S1' is 'the none arm flips the row' rather than 'no path permanently "
                "blocked'.  With no none-arm file supplied the count is reported as "
                "constructively 0/93 with the reason recorded, not silently zeroed.",
        "why": "Our battery is greedy + single-step teacher-forced, so a bare retry is a "
               "no-op by construction; the paper's episode-level blocking has no analogue "
               "on a face without an environment.",
    },
    {
        "method": "BTM solvability classes",
        "paper": "2608.11977",
        "what": "The bench-face regenerate count is reported as the FULL-ROLLBACK rescue "
                "rate (bench_regenerate_rescue), explicitly NOT as bench S1'.",
        "why": "Their S1 retry_works is a retry on the same context.  Our proxy's "
               "oracle-recover rebuilds the request on FULL_ASSEMBLY "
               "(Arm(compress_history=False), c2kv/benchmarks/proxy.py:545 used at :1226), "
               "i.e. it re-prefills the uncompressed history -- the bottom rung of the "
               "recovery ladder, not a retry.  Calling it S1' would credit 'retry "
               "suffices' for rows only a full rebuild rescues and would soften C-K5 "
               "with the wrong number.  Bench S1' stays unmeasured until a bare-retry arm "
               "exists.",
    },
    {
        "method": "IRBench cost triples",
        "paper": "2608.16370",
        "what": "The fixed interaction horizon is a caller-supplied max_turn, not the "
                "paper's 24; when it is omitted the triples are flagged as NOT "
                "horizon-bounded rather than silently reported as Definition 1's Y(m).",
        "why": "Definition 1 records (Q, C_R, C_E) 'under a fixed interaction horizon' and "
               "their section 4.8 shows the horizon binds in every High cell.  Their 24 is "
               "a property of their 10-task world and does not port; the requirement that "
               "both arms be read at the SAME horizon does.",
    },
    {
        "method": "IRBench cost triples",
        "paper": "2608.16370",
        "what": "Holm across the declared family is available at the COMPARISON-CELL level "
                "(irbench_family_report); irbench_report's per-arm-pair correction is "
                "labelled as the narrower scope it is.",
        "why": "Their family is the six retrieval comparisons (3 models x 2 regimes) "
               "corrected jointly (section 3.4).  Correcting metrics inside one arm pair "
               "with a single primary metric applies no correction at all, which would "
               "quietly drop the multiplicity the protocol exists to control.",
    },
    {
        "method": "Context Equilibria",
        "paper": "2510.07777",
        "what": "The equilibrium CLI fits (a, b) on the C->C pool when labels are supplied "
                "and, when they are not, stamps the output with a warning that the fit "
                "used the evaluation rows.",
        "why": "The digest fixes the fit pool as the C->C training pool.  The paper fits "
               "descriptively on the same trajectories it describes (section 12, no "
               "split); inheriting that silently would be selection on evaluation rows.",
    },
    {
        "method": "BTM explicit/silent taxonomy",
        "paper": "2608.11977",
        "what": "The 3 silent modes (partial/stale/factual_error) are never emitted by the "
                "text classifier; only the 6 explicit modes plus 'unclassified' are.",
        "why": "The paper defines silent modes as observationally indistinguishable from a "
               "clean response; a text rule that claimed to detect them would be inventing "
               "a signal the generative process does not expose.",
    },
    {
        "method": "BTM explicit/silent cut on our rows",
        "paper": "2608.11977",
        "what": "The L1-visible vs silent stratification is emitted with a hard "
                "'not computable' flag whenever the cap rate exceeds CAP_RATE_CAVEAT.",
        "why": "Under the frozen 128-token cap 'protocol legal' is largely 'finished inside "
               "128 tokens' (digest 4.9: capped 0.9% vs uncapped 37.1%), so the stratum "
               "would be measuring the cap.",
    },
    {
        "method": "IRBench cost triples",
        "paper": "2608.16370",
        "what": "A third bucket 'prose_or_other' is mandatory alongside retrieval/execution, "
                "and cost is accumulated from turn 1 onward (turn 0 excluded).",
        "why": "~45% of BFCL steps emit prose rather than a tool call, and at turn-0 step-0 "
               "all four arms are byte-identical (_history_cutoff returns 0), so turn 0 is "
               "a constructive tie rather than a measurement.",
    },
    {
        "method": "IRBench cost triples",
        "paper": "2608.16370",
        "what": "Their paired unit is a SEED (a seed fully determines their world, so a "
                "condition replays an identical task set); ours is a conversation "
                "(conv_id), which is what our arms replay identically.",
        "why": "The bench face has no seed axis -- each arm re-runs the same benchmark "
               "conversations, so conv_id is the exchangeable pairing unit.  Stated rather "
               "than relabelled: a per-conversation Wilcoxon is not a per-seed Wilcoxon.",
    },
    {
        "method": "IRBench D*/R* feature",
        "paper": "2608.16370",
        "what": "Per-block compression ratio is not on disk (only aggregate gist/original "
                "token counts), so kept blocks share one aggregate loss weight and the "
                "dropped-only variant s_t_dropped_only is emitted beside s_t.",
        "why": "Their per-atom digest budget has no per-block analogue in our logs; emitting "
               "both makes the aggregate-weight assumption checkable instead of hidden.",
    },
    {
        "method": "IRBench D*/R* feature",
        "paper": "2608.16370",
        "what": "s_t's DROPPED half is computed from the sidecar's optional "
                "dropped_doc_texts, never from the sidecar's docs list.  With no dropped "
                "text on disk the dropped half is null and s_t is emitted as the "
                "kept-only partial, labelled by irb_s_t_scope.",
        "why": "The sidecar's docs holds ONLY the kept blocks and dropped_docs indexes the "
               "post-split history list, a different coordinate system.  Selecting docs by "
               "those indices would score visible blocks as if they had been dropped, and "
               "reporting 0 for an absent dropped half would read as 'nothing was lost'.",
    },
    {
        "method": "IRBench D*/R* feature",
        "paper": "2608.16370",
        "what": "When the partition carries no declared return-field schemas the D/R call "
                "falls back to a token-overlap heuristic against retrieval TOOL NAMES, and "
                "every row records which mode produced it.",
        "why": "The card's 'schema question, not a model question' needs return schemas; our "
               "benchmark manifests may not declare them.  The fallback is declared rather "
               "than silently mixed with schema-grounded rows.",
    },
    {
        "method": "Last Step Matters backward truncation",
        "paper": "2608.29685",
        "what": "Combination classifiers use session/conversation-GROUPED nested CV, not the "
                "paper's 5-fold stratified CV.",
        "why": "Stratified folds leak across steps of the same conversation.  The paper's own "
               "8-rollouts-per-task structure has the same exposure; we do not copy it.",
    },
    {
        "method": "Last Step Matters signal battery",
        "paper": "2608.29685",
        "what": "Verbal confidence is always None (never elicited by our harness) and the "
                "logprob-derived signals are computed only when the corresponding log "
                "columns exist; absent signals are reported as absent, never imputed.",
        "why": "proxy.normalize_response drops logprobs today, so those columns hold zero "
               "values on disk.  Fabricating them would manufacture the study's own signal.",
    },
    {
        "method": "Context Equilibria",
        "paper": "2510.07777",
        "what": "The reference policy p_t and D_t = KL(q_t || p_t) are DISCARDED; only the "
                "dynamics layer is ported, onto a reference-free Z_t^pre.",
        "why": "p_t is an extra forward over the uncompressed history -- exactly the cost the "
               "trigger exists to avoid, and the family already killed as S2 norm_logp_gap.",
    },
    {
        "method": "Context Equilibria",
        "paper": "2510.07777",
        "what": "A white-noise pre-gate on b != -1 runs BEFORE any equilibrium band is "
                "reported, and the band is withheld when the gate fails.",
        "why": "Regressing dD_t on a noisy D_t yields b = -1 mechanically; four of the "
               "paper's six fitted b values sit in [-1.05, -0.96] (Table tab:equilibrium_core), "
               "so the restoring-force reading is not established for them.",
    },
    {
        "method": "Context Equilibria",
        "paper": "2510.07777",
        "what": "kappa is chosen in inner folds of grouped CV; the paper fits (a, b) "
                "descriptively on the same trajectories it describes, with no split.",
        "why": "A threshold chosen on the evaluation rows is not a threshold.",
    },
    {
        "method": "Context Equilibria",
        "paper": "2510.07777",
        "what": "kappa is selected by PRECISION at a declared fire rate (default: the "
                "labelled episodes' own prevalence), reported next to that prevalence as "
                "chance_precision and next to the kappa-free AUROC of the standardised "
                "deviation.  No ranking metric is computed on the binary fired indicator.",
        "why": "kappa only slides an operating point along one fixed score, so an AP or "
                "AUROC computed on the two-valued fired indicator is a degenerate "
                "one-threshold summary that is not comparable to the AUPRC numbers the "
                "rest of the contract reports -- quoting it beside them would compare two "
                "different quantities.",
    },
]

# --------------------------------------------------------------------------
# (A) BENCH2ROBUST / BTM -- arXiv 2608.11977
# --------------------------------------------------------------------------

#: 2608.11977 section 2, observation kernel Z_inj: 6 explicit-signal failure modes.
EXPLICIT_FAILURE_MODES: Tuple[str, ...] = (
    "timeout", "rate_limit", "server_error", "auth_error",
    "malformed_response", "schema_drift",
)
#: ... and 3 silent ones.  Never emitted by the text classifier (see DEVIATIONS).
SILENT_FAILURE_MODES: Tuple[str, ...] = ("partial", "stale", "factual_error")

#: 2608.11977 section 2: P(clean) = 0.60 in their injection budget.  Recorded for
#: provenance only -- the card is explicit that this is a stress-test setting and
#: must NOT be imported as a prior.  Nothing in this module reads it.
PAPER_CLEAN_INJECTION_RATE = 0.60

#: Cap rate above which the explicit/silent cut is declared not computable.
#: Declared here, before any run, per the digest's caliber caveat.
CAP_RATE_CAVEAT = 0.20

_OBSERVATION_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("timeout", re.compile(r"\b(timed?[ _-]?out|timeout|deadline exceeded|etimedout)\b", re.I)),
    ("rate_limit", re.compile(r"\b(rate[ _-]?limit\w*|too many requests|429|quota exceeded|throttl\w*)\b", re.I)),
    ("auth_error", re.compile(r"\b(unauthori[sz]ed|forbidden|invalid (?:api[ _-]?key|token|credentials)"
                              r"|authentication failed|401|403)\b", re.I)),
    ("server_error", re.compile(r"\b(internal server error|service unavailable|bad gateway|"
                                r"upstream error|50[0234])\b", re.I)),
    ("schema_drift", re.compile(r"\b(unknown (?:field|argument|parameter)|unexpected (?:field|key)|"
                                r"missing required (?:field|argument|parameter)|schema mismatch|"
                                r"schema drift|deprecated field)\b", re.I)),
    ("malformed_response", re.compile(r"\b(malformed|could not (?:parse|decode)|json ?decode ?error|"
                                      r"unparse\w*|invalid json|truncated response)\b", re.I)),
)


def classify_observation(text: Optional[str]) -> Dict[str, Any]:
    """Bench-side observation classifier: tool-response text -> failure class.

    Implements the OBSERVABLE half of the 9-mode taxonomy of 2608.11977 section 2
    (``Z_inj``; 6 explicit-signal modes, 3 silent).  Silent modes are by the
    paper's own construction indistinguishable from a clean response at the text
    level, so this function never returns one -- it returns ``class=None`` with
    ``channel="unclassified"`` and leaves the silent question to a label-side
    ground truth.

    Returns ``{class, channel, matched, silent_modes_undetectable}``.
    """
    body = (text or "")
    for name, pattern in _OBSERVATION_PATTERNS:
        m = pattern.search(body)
        if m:
            return {"class": name, "channel": "explicit", "matched": m.group(0),
                    "silent_modes_undetectable": True}
    return {"class": None, "channel": "unclassified", "matched": None,
            "silent_modes_undetectable": True}


def assign_solvability_classes(
    cw_qids: Sequence[str],
    flip_table: Dict[str, Dict[int, bool]],
    none_arm_hits: Optional[Dict[str, bool]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Assign S1'/S2'/S3' once, from frozen artefacts only (2608.11977 section 2.1).

    * ``S1'`` -- the ``none`` arm (bare regenerate) already flips the row.  On the
      battery this is constructively empty (greedy + teacher-forced): pass
      ``none_arm_hits`` from an arm results file to measure it rather than assume it.
    * ``S2'`` -- some block index ``k`` in the frozen flip table flips the row.
      ``single_k`` marks the rows where exactly one k works (their "switch" case).
    * ``S3'`` -- neither.  Scored, not dropped: it is the ceiling statement.

    ``flip_table`` is ``qid -> {k: correct}`` from ``t34_common.load_flip_table``.
    A qid absent from the flip table is S3' with ``flip_table_missing=True`` so a
    truncated dump can never masquerade as an unsolvable row.
    """
    none_arm_hits = none_arm_hits or {}
    out: Dict[str, Dict[str, Any]] = {}
    for qid in cw_qids:
        per_k = flip_table.get(qid)
        flip_ks = sorted(k for k, ok in (per_k or {}).items() if ok)
        none_flip = bool(none_arm_hits.get(qid, False))
        if none_flip:
            cls = "S1'"
        elif flip_ks:
            cls = "S2'"
        else:
            cls = "S3'"
        out[qid] = {
            "class": cls,
            "none_flips": none_flip,
            "flip_ks": flip_ks,
            "n_flip_k": len(flip_ks),
            "single_k": bool(len(flip_ks) == 1),
            "flip_table_missing": per_k is None,
        }
    return out


def solvability_census(classes: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Counts + the two denominators the digest requires (93 and |S2'|)."""
    n = len(classes)
    by = {c: [q for q, v in classes.items() if v["class"] == c] for c in ("S1'", "S2'", "S3'")}
    s2 = by["S2'"]
    single = [q for q in s2 if classes[q]["single_k"]]
    return {
        "n_rows": n,
        "n_S1": len(by["S1'"]),
        "n_S2": len(s2),
        "n_S3": len(by["S3'"]),
        "n_S2_single_k": len(single),
        "n_S2_multi_k": len(s2) - len(single),
        "n_flip_table_missing": sum(1 for v in classes.values() if v["flip_table_missing"]),
        "denominator_all": n,
        "denominator_S2": len(s2),
        # the card's "report coverage against 81, not 93" denominator is the set of
        # rows SOMETHING can rescue, i.e. S1' + S2'.  It coincides with |S2'| only
        # while S1' is empty (the battery's constructive case); the moment a
        # none-arm file is supplied it does not, and |S2'| alone would then be the
        # wrong ceiling.
        "denominator_recoverable": len(by["S1'"]) + len(s2),
        "note": ("coverage must be printed against BOTH denominators (all C->W rows and "
                 "the recoverable set S1'+S2'); no trigger can rescue an S3' row"),
    }


def freeze_solvability_manifest(path: Path, classes: Dict[str, Dict[str, Any]],
                                provenance: Dict[str, Any]) -> Dict[str, Any]:
    """Write the class assignment ONCE and return it with its sha256.

    2608.11977's own class counts are policy-dependent; ours must not be.  The
    frozen file is the contract that a detector never redefines its denominator.
    """
    payload = {
        "schema": "t34.bencha.solvability/1",
        "paper": "2608.11977",
        "classes": {q: dict(v) for q, v in sorted(classes.items())},
        "census": solvability_census(classes),
        "provenance": provenance,
    }
    sha = C.freeze_json(Path(path), payload)
    return {"path": str(path), "sha256": sha, "census": payload["census"],
            "provenance": payload["provenance"]}


def metrics_by_solvability_class(
    scores: Sequence[float],
    y: Sequence[int],
    qids: Sequence[str],
    classes: Dict[str, Dict[str, Any]],
    n_fires: int,
) -> Dict[str, Any]:
    """Re-report the three metrics stratified by solvability class.

    Coverage is per class and additionally against both frozen denominators
    (all C->W rows, and |S2'|).  Precision and false-reset are GLOBAL, because
    the class is defined only on positives (C->C rows carry no class) -- stated
    rather than silently recomputed on a mixed denominator.
    """
    s = np.asarray(scores, dtype=float)
    yy = np.asarray(y, dtype=int)
    op = C.operating_point(s, yy, n_fires)
    order = np.argsort(-s, kind="mergesort")
    fire = np.zeros(len(s), dtype=bool)
    fire[order[: max(0, int(n_fires))]] = True
    census = solvability_census(classes)
    rows = []
    for cls in ("S1'", "S2'", "S3'"):
        idx = [i for i, q in enumerate(qids) if yy[i] == 1 and classes.get(q, {}).get("class") == cls]
        cov = int(sum(bool(fire[i]) for i in idx))
        rows.append({
            "class": cls,
            "n_pos_in_class": len(idx),
            "coverage": cov,
            "coverage_over_class": (cov / len(idx)) if idx else None,
            "coverage_over_all_cw": cov / census["denominator_all"] if census["denominator_all"] else None,
            "coverage_over_S2": (cov / census["denominator_S2"]) if census["denominator_S2"] else None,
            "coverage_over_recoverable": ((cov / census["denominator_recoverable"])
                                          if census["denominator_recoverable"] else None),
        })
    return {
        "global": op,
        "by_class": rows,
        "census": census,
        "note": "precision / false-reset are global; class is defined on positives only",
    }


#: the recover-bookkeeping columns this function reads.  Their presence is counted
#: and returned, so an all-zero result can never be mistaken for a measurement on a
#: log that simply does not carry them (the short proxy in this worktree does not).
_RECOVER_COLUMNS: Tuple[str, ...] = ("diverged_now", "repaired", "repair_fidelity",
                                     "recovered_action_match", "match")


def bench_regenerate_rescue(convs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Bench-face regenerate-rescue rate -- the FULL-ROLLBACK rung, NOT BTM's S1'.

    The battery's S1' (2608.11977 section 2.1 ``retry_works``) is a bare retry on
    the SAME prefix, and on our battery it is constructively empty (greedy +
    single-step teacher-forced).  The bench face has a regeneration that really
    regenerates -- but the proxy's oracle-recover rebuilds the request on
    ``FULL_ASSEMBLY`` (``Arm(compress_history=False)``, c2kv/benchmarks/proxy.py:545
    used at :1226), i.e. it re-prefills the UNCOMPRESSED history.  That is the
    bottom rung of our recovery ladder (full rollback), not a retry, so this count
    is an upper bound on S1' and must never be reported as S1' itself: doing so
    would credit "retry suffices" for rows only a full rebuild rescues, which is
    exactly the C-K5 defence the class stratification exists to make honestly.

    Reads only bookkeeping the proxy already logs (c2kv/benchmarks/proxy.py:1241-1249):
    ``repair_fidelity`` = the regenerated action reproduced the REFERENCE action;
    ``recovered_action_match`` = it equalled the compressed arm's own action.
    """
    n_div = n_repaired = n_flipped = n_fidelity = 0
    convs_with_div = set()
    convs_rescued = set()
    present = {col: 0 for col in _RECOVER_COLUMNS}
    n_rows = 0
    for conv, rows in convs.items():
        for row in rows:
            n_rows += 1
            for col in _RECOVER_COLUMNS:
                if col in row:
                    present[col] += 1
            if row.get("diverged_now"):
                n_div += 1
                convs_with_div.add(conv)
            if row.get("repaired"):
                n_repaired += 1
                if row.get("recovered_action_match") is False:
                    # the regenerated action differs from the compressed one, i.e. the
                    # regenerate actually changed something
                    n_flipped += 1
                if row.get("repair_fidelity"):
                    n_fidelity += 1
                    convs_rescued.add(conv)
    return {
        "paper": "2608.11977",
        "face": "bench",
        "rung": "full_rollback_regeneration",
        "is_s1_prime": False,
        "n_rows": n_rows,
        "columns_present": present,
        "columns_missing": sorted(c for c, k in present.items() if k == 0),
        "n_divergence_events": n_div,
        "n_convs_with_divergence": len(convs_with_div),
        "n_repair_events": n_repaired,
        "n_regenerate_changed_action": n_flipped,
        "n_regenerate_matched_reference": n_fidelity,
        "full_rollback_conv_rescue_rate": (len(convs_rescued) / len(convs_with_div)
                                           if convs_with_div else None),
        "note": ("battery S1' is constructively 0 (greedy + single-step teacher-forced). "
                 "This bench-face number is the FULL-ROLLBACK rescue rate and is an UPPER "
                 "BOUND on S1', not S1': the proxy's recover arm re-prefills the "
                 "uncompressed history (FULL_ASSEMBLY). A bare-retry arm on the bench face "
                 "does not exist today, so bench S1' remains unmeasured."),
    }


def default_schema_violation(name: Optional[str], args: Any,
                             tools: Sequence[Dict[str, Any]]) -> Optional[str]:
    """Torch-free mirror of ``benchmarks/metrics.py:_schema_violations`` (:66).

    Kept local because the battery face has no proxy and this module must not
    import anything under ``benchmarks/``.  On the bench face pass the real
    function in as ``schema_violation_fn`` so only one implementation is live.
    Degrades to ``None`` (unknown) when no tool pool is advertised, exactly as
    the original does.
    """
    if not tools:
        return None
    tool = next((t for t in tools if (t.get("function") or {}).get("name") == name), None)
    if tool is None:
        return "unknown tool name %r" % (name,)
    if args is None or not isinstance(args, dict):
        return "arguments are not a JSON object"
    schema = (tool.get("function") or {}).get("parameters") or {}
    for key in schema.get("required") or []:
        if key not in args:
            return "missing required argument %r" % (key,)
    props = schema.get("properties") or {}
    checks = {"integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
              "string": lambda v: isinstance(v, str),
              "boolean": lambda v: isinstance(v, bool),
              "array": lambda v: isinstance(v, list),
              "object": lambda v: isinstance(v, dict)}
    for key, value in args.items():
        expected = (props.get(key) or {}).get("type")
        check = checks.get(expected)
        if check is not None and not check(value):
            return "argument %r must be %s" % (key, expected)
    return None


def l1_visibility(
    prediction: str,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
    *,
    schema_violation_fn: Optional[Callable[[Optional[str], Any, Sequence[Dict[str, Any]]], Optional[str]]] = None,
) -> Dict[str, Any]:
    """L1-visible vs silent cut (2608.11977 explicit/silent, ported to our L1 tier).

    ``l1_visible`` is True when the compressed arm's OWN emission fails any of
    has_tool_call / strict parse / schema legality; None when legality is not
    computable (no tool pool advertised) and nothing else fired.
    """
    parsed = parse_tool_call(prediction or "")
    violation: Optional[str] = None
    legality_known = True
    if parsed["parse_ok"]:
        fn = schema_violation_fn or default_schema_violation
        violation = fn(parsed["name"], parsed["arguments"], tools or [])
        legality_known = bool(tools)
    if not parsed["has_tool_call"] or not parsed["parse_ok"]:
        visible: Optional[bool] = True
    elif violation:
        visible = True
    elif legality_known:
        visible = False
    else:
        visible = None
    return {
        "has_tool_call": bool(parsed["has_tool_call"]),
        "parse_ok": bool(parsed["parse_ok"]),
        "closed": bool(parsed["closed"]),
        "first_violation": violation,
        "protocol_legal": (violation is None) if (parsed["parse_ok"] and legality_known) else None,
        "l1_visible": visible,
    }


def cap_caveat(censored_flags: Sequence[bool], cap_tokens: int) -> Dict[str, Any]:
    """Emit the 128-token caliber caveat the digest pre-registers for the cut."""
    flags = [bool(x) for x in censored_flags]
    rate = (sum(flags) / len(flags)) if flags else None
    blocked = bool(rate is not None and rate > CAP_RATE_CAVEAT)
    return {
        "cap_tokens": int(cap_tokens),
        "n_rows": len(flags),
        "cap_rate": rate,
        "threshold": CAP_RATE_CAVEAT,
        "explicit_silent_computable": (None if rate is None else not blocked),
        "caveat": ("explicit/silent not computable at this caliber: 'protocol legal' is "
                   "largely 'finished within %d tokens'; report the cut only at a larger "
                   "caliber" % int(cap_tokens)) if blocked else None,
    }


# --------------------------------------------------------------------------
# (B) IRBench -- arXiv 2608.16370
# --------------------------------------------------------------------------

TOOL_BUCKETS: Tuple[str, ...] = ("retrieval", "execution", "prose_or_other")


def partition_sha256(obj: Dict[str, Any]) -> str:
    """sha256 of the canonical partition WITHOUT its own sha256 field."""
    body = {k: v for k, v in obj.items() if k != "sha256"}
    text = json.dumps(body, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    return C.sha256_bytes(text.encode("utf-8"))


def load_tool_partition(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_tool_partition(obj: Dict[str, Any],
                            declared_tools: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Validate the checked-in static partition (2608.16370 section 3.2).

    The third bucket is mandatory (their ALFWorld probe needed one, and ~45% of
    our BFCL steps emit prose).  Buckets must be disjoint.  ``declared_tools``,
    when given, must be fully covered -- an unclassified tool silently becoming
    "not retrieval" would bias ``C_R`` downward.
    """
    errors: List[str] = []
    if not obj.get("benchmark"):
        errors.append("missing benchmark name")
    buckets = {b: list(obj.get(b) or []) for b in TOOL_BUCKETS}
    for b in TOOL_BUCKETS:
        if b not in obj:
            errors.append("missing bucket %r" % b)
    seen: Dict[str, str] = {}
    for b, names in buckets.items():
        for name in names:
            if name in seen:
                errors.append("tool %r in both %s and %s" % (name, seen[name], b))
            seen[name] = b
    unclassified: List[str] = []
    if declared_tools is not None:
        unclassified = sorted(set(declared_tools) - set(seen))
        if unclassified:
            errors.append("unclassified declared tools: %s" % unclassified)
    sha_ok = None
    if obj.get("sha256"):
        sha_ok = (obj["sha256"] == partition_sha256(obj))
        if not sha_ok:
            errors.append("sha256 mismatch")
    returns = obj.get("retrieval_returns") or {}
    for name in returns:
        if name not in set(buckets["retrieval"]):
            errors.append("retrieval_returns names a non-retrieval tool %r" % name)
    return {
        "ok": not errors,
        "errors": errors,
        "sha256_ok": sha_ok,
        "filled": bool(seen),
        "n_tools": len(seen),
        "unclassified": unclassified,
        "has_return_schemas": bool(returns),
        "ready_for_measurement": bool(seen) and not errors,
        "note": (None if seen else
                 "EMPTY partition: with no tool names declared, every value falls to the "
                 "token-overlap heuristic and is typed R by default (irb_dr_mode="
                 "'heuristic').  That is a declared fallback, NOT a measurement -- fill "
                 "retrieval / execution / prose_or_other from the benchmark manifest, "
                 "optionally add retrieval_returns, and re-freeze sha256 before quoting "
                 "any D*/R* number."),
    }


def tool_bucket(name: Optional[str], partition: Dict[str, Any]) -> Optional[str]:
    for b in TOOL_BUCKETS:
        if name in set(partition.get(b) or []):
            return b
    return None


def _row_tool_names(row: Dict[str, Any]) -> List[str]:
    """Tool names of the emitted action of one proxy row.

    Accepts either a row carrying ``action`` (the ``action_canonical`` dict the
    ``--record-reference`` sidecar writes, joined on ``fp``) or a row whose
    ``action`` was merged into ``_log_request``.  Rows without an action carry no
    action information and are counted as such by the caller.
    """
    action = row.get("action")
    if not isinstance(action, dict):
        return []
    names = []
    for call in action.get("tool_calls") or []:
        name = call.get("name")
        if name:
            names.append(str(name))
    return names


#: 2608.16370 section 3.2 / section 4.8: their world runs under a FIXED interaction
#: horizon of 24 turns, and Definition 1's triple is defined "under a fixed
#: interaction horizon".  Ours is a different environment, so the number does not
#: port -- but the *requirement* does: a cost triple compared across arms must be
#: read at a horizon that is the same for both, or one arm is simply given more
#: turns to spend.  ``cost_triples(max_turn=...)`` enforces it; the paper's own
#: value is recorded here as provenance, never used as our default.
PAPER_INTERACTION_HORIZON_TURNS = 24


def cost_triples(
    rows: Sequence[Dict[str, Any]],
    partition: Dict[str, Any],
    *,
    min_turn: int = 1,
    max_turn: Optional[int] = None,
    completion_by_conv: Optional[Dict[str, float]] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Per (arm, conv_id) interaction-cost triple ``Y = (Q, C_R, C_E)``.

    2608.16370 Definition 1.  ``min_turn=1`` excludes turn 0: at turn-0 step-0 the
    arms are byte-identical (``_history_cutoff`` returns 0), so turn 0 is a
    constructive tie, not a measurement.  ``max_turn`` is the paper's FIXED
    interaction horizon (theirs: 24 turns); pass it whenever the arms being
    compared can run for different numbers of turns, or the triple is not the
    horizon-bounded quantity Definition 1 defines.  Steps that emitted no tool call
    land in ``C_other`` (the mandatory third bucket) and are never dropped.  ``Q``
    comes from the harness when supplied; it is never inferred from the log.

    Rows carrying no ``turn`` are counted in ``n_rows_missing_turn`` on the result's
    ``__meta__`` entry rather than silently vanishing -- the short proxy log schema
    has no ``turn`` column at all, and an empty result must be readable as "the log
    lacks the column", not as "the arms cost nothing".
    """
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    n_missing_turn = 0
    n_over_horizon = 0
    for row in rows:
        turn = row.get("turn")
        if turn is None:
            n_missing_turn += 1
            continue
        if int(turn) < int(min_turn):
            continue
        if max_turn is not None and int(turn) > int(max_turn):
            n_over_horizon += 1
            continue
        key = (row.get("arm"), row.get("conv_id"))
        rec = out.setdefault(key, {"arm": key[0], "conv_id": key[1], "Q": None,
                                   "C_R": 0, "C_E": 0, "C_other": 0,
                                   "n_steps": 0, "n_unbucketed": 0, "n_no_action": 0,
                                   "max_turn_seen": None, "termination": {}})
        rec["n_steps"] += 1
        rec["max_turn_seen"] = (int(turn) if rec["max_turn_seen"] is None
                                else max(int(rec["max_turn_seen"]), int(turn)))
        # their section 4.8 requires the termination reason per run so a failure can
        # be audited as budget exhaustion rather than as an error
        term = row.get("finish_reason") or row.get("error_kind") or row.get("status")
        if term is not None:
            rec["termination"][str(term)] = rec["termination"].get(str(term), 0) + 1
        names = _row_tool_names(row)
        if not names:
            if row.get("action") is None:
                rec["n_no_action"] += 1
            rec["C_other"] += 1
            continue
        for name in names:
            bucket = tool_bucket(name, partition)
            if bucket == "retrieval":
                rec["C_R"] += 1
            elif bucket == "execution":
                rec["C_E"] += 1
            elif bucket == "prose_or_other":
                rec["C_other"] += 1
            else:
                rec["n_unbucketed"] += 1
    for key, rec in out.items():
        rec["C_total"] = rec["C_R"] + rec["C_E"] + rec["C_other"]
        if completion_by_conv is not None:
            rec["Q"] = completion_by_conv.get(key[1])
    out[("__meta__", "__meta__")] = {
        "arm": None, "conv_id": None, "meta": True,
        "n_rows": len(rows), "n_rows_missing_turn": n_missing_turn,
        "n_rows_over_horizon": n_over_horizon,
        "min_turn": int(min_turn), "max_turn": max_turn,
        "paper_horizon_turns": PAPER_INTERACTION_HORIZON_TURNS,
        "horizon_declared": max_turn is not None,
        "note": ("no max_turn given: the triples are NOT horizon-bounded, so an arm that "
                 "ran longer is charged for the extra turns (2608.16370 Definition 1)"
                 if max_turn is None else None),
    }
    return out


def paired_differences(a: Dict[str, float], b: Dict[str, float]) -> Tuple[List[str], np.ndarray]:
    """Per-seed (here: per-conversation) paired differences a - b on shared keys."""
    keys = sorted(set(a) & set(b))
    return keys, np.array([float(a[k]) - float(b[k]) for k in keys], dtype=float)


def wilcoxon_paired(diffs: Sequence[float]) -> Dict[str, Any]:
    """Per-seed paired Wilcoxon signed-rank (2608.16370 section 3.4)."""
    d = np.asarray(list(diffs), dtype=float)
    d = d[np.isfinite(d)]
    nz = d[d != 0]
    if nz.size == 0:
        return {"n": int(d.size), "n_nonzero": 0, "stat": None, "p": 1.0,
                "note": "all paired differences are zero"}
    from scipy.stats import wilcoxon
    stat, p = wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return {"n": int(d.size), "n_nonzero": int(nz.size), "stat": float(stat), "p": float(p)}


def bootstrap_diff_ci(diffs: Sequence[float], reps: int = 2000, seed: int = 20260905,
                      alpha: float = 0.05) -> Dict[str, Any]:
    """Bootstrap 95% CI on the MEAN of the per-seed differences (their section 3.4)."""
    d = np.asarray(list(diffs), dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    draws = rng.choice(d, size=(reps, d.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(d.size)}


def holm_correct(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, Any]]:
    """Holm-Bonferroni within a DECLARED family (2608.16370 section 3.4, alpha=0.05).

    Step-down: sorted ascending, threshold alpha/(m-i), and once a comparison
    fails every later one fails too (monotone enforcement).
    """
    items = sorted(pvalues.items(), key=lambda kv: (kv[1], kv[0]))
    m = len(items)
    out: Dict[str, Dict[str, Any]] = {}
    running = 0.0
    failed = False
    for i, (name, p) in enumerate(items):
        thresh = alpha / (m - i) if m - i > 0 else alpha
        running = max(running, min(1.0, p * (m - i)))
        reject = (p <= thresh) and not failed
        if not reject:
            failed = True
        out[name] = {"p": float(p), "rank": i + 1, "m": m, "threshold": thresh,
                     "p_adj": float(running), "reject": bool(reject)}
    return out


def irbench_report(
    triples: Dict[Tuple[str, str], Dict[str, Any]],
    *,
    arm_a: str,
    arm_b: str,
    primary_family: Sequence[str] = ("C_R",),
    post_hoc: Sequence[str] = ("C_E", "C_total", "Q"),
    alpha: float = 0.05,
    reps: int = 2000,
    seed: int = 20260905,
) -> Dict[str, Any]:
    """Paired arm comparison with a declared primary family and Holm correction.

    The family is declared by the caller BEFORE the comparison; everything else is
    reported with ``post_hoc=True``, matching their D-Irrelevant discipline.
    """
    def series(arm: str, metric: str) -> Dict[str, float]:
        return {conv: rec[metric] for (a, conv), rec in triples.items()
                if a == arm and not rec.get("meta") and rec.get(metric) is not None}

    results: Dict[str, Any] = {"arm_a": arm_a, "arm_b": arm_b,
                               "primary_family": list(primary_family),
                               "post_hoc_metrics": list(post_hoc), "metrics": {}}
    pvals: Dict[str, float] = {}
    for metric in list(primary_family) + list(post_hoc):
        keys, d = paired_differences(series(arm_a, metric), series(arm_b, metric))
        if d.size == 0:
            results["metrics"][metric] = {"n_pairs": 0, "post_hoc": metric not in primary_family}
            continue
        w = wilcoxon_paired(d)
        ci = bootstrap_diff_ci(d, reps=reps, seed=seed)
        results["metrics"][metric] = {
            "n_pairs": len(keys), "wilcoxon": w, "bootstrap_mean_diff": ci,
            "post_hoc": metric not in primary_family,
        }
        if metric in primary_family and w["p"] is not None:
            pvals[metric] = w["p"]
    results["holm"] = holm_correct(pvals, alpha=alpha) if pvals else {}
    results["alpha"] = alpha
    results["holm_family_scope"] = "metrics within this ONE arm pair"
    results["holm_warning"] = (
        "2608.16370 section 3.4's family is the SIX retrieval COMPARISONS (3 models x 2 "
        "regimes), i.e. it spans comparison cells, not metrics inside one cell.  If more "
        "than one arm pair is being compared, this per-pair correction is not the paper's "
        "family: use irbench_family_report() so Holm runs across all declared cells."
        if len(pvals) <= 1 else None)
    return results


def irbench_family_report(
    triples: Dict[Tuple[str, str], Dict[str, Any]],
    comparisons: Sequence[Tuple[str, str, str]],
    *,
    alpha: float = 0.05,
    reps: int = 2000,
    seed: int = 20260905,
    post_hoc: Sequence[Tuple[str, str, str]] = (),
) -> Dict[str, Any]:
    """Holm across a DECLARED family of comparison CELLS (2608.16370 section 3.4).

    Their family is the six retrieval comparisons -- three models x two regimes --
    corrected jointly at alpha = 0.05, after which five of six survive.  The family
    therefore spans *cells*, not metrics within one cell, so a per-arm-pair
    correction (``irbench_report``) with a single primary metric applies no
    correction at all.  ``comparisons`` is a list of ``(arm_a, arm_b, metric)``
    triples declared BEFORE the scan; anything in ``post_hoc`` is tested and
    reported with ``post_hoc=True`` and is excluded from the family.
    """
    def series(arm: str, metric: str) -> Dict[str, float]:
        return {conv: rec[metric] for (a, conv), rec in triples.items()
                if a == arm and not rec.get("meta") and rec.get(metric) is not None}

    def cell(arm_a: str, arm_b: str, metric: str) -> Dict[str, Any]:
        keys, d = paired_differences(series(arm_a, metric), series(arm_b, metric))
        if d.size == 0:
            return {"arm_a": arm_a, "arm_b": arm_b, "metric": metric, "n_pairs": 0}
        return {"arm_a": arm_a, "arm_b": arm_b, "metric": metric, "n_pairs": len(keys),
                "wilcoxon": wilcoxon_paired(d),
                "bootstrap_mean_diff": bootstrap_diff_ci(d, reps=reps, seed=seed)}

    cells: Dict[str, Dict[str, Any]] = {}
    pvals: Dict[str, float] = {}
    for arm_a, arm_b, metric in comparisons:
        name = "%s_vs_%s::%s" % (arm_a, arm_b, metric)
        rec = cell(arm_a, arm_b, metric)
        rec["post_hoc"] = False
        cells[name] = rec
        if rec.get("wilcoxon") and rec["wilcoxon"]["p"] is not None:
            pvals[name] = rec["wilcoxon"]["p"]
    for arm_a, arm_b, metric in post_hoc:
        name = "%s_vs_%s::%s" % (arm_a, arm_b, metric)
        rec = cell(arm_a, arm_b, metric)
        rec["post_hoc"] = True
        cells[name] = rec
    holm = holm_correct(pvals, alpha=alpha) if pvals else {}
    return {
        "paper": "2608.16370",
        "primary_family": ["%s_vs_%s::%s" % c for c in comparisons],
        "post_hoc": ["%s_vs_%s::%s" % c for c in post_hoc],
        "family_size": len(pvals),
        "cells": cells,
        "holm": holm,
        "alpha": alpha,
        "n_survive_holm": sum(1 for v in holm.values() if v["reject"]),
        "note": ("the family spans comparison cells and is declared before the scan; "
                 "post-hoc cells are tested but never enter the correction"),
    }


# ---- D*/R* recoverability feature (prefix-only, pre-generation) ------------

_CAMEL = re.compile(r"(?<!^)(?=[A-Z])")


def _tokens(name: str) -> set:
    return {t for t in re.split(r"[^A-Za-z0-9]+", _CAMEL.sub("_", str(name))) if t}


def keyed_leaves(value: Any, key: Optional[str] = None) -> List[Tuple[Optional[str], Any]]:
    """JSON leaves paired with the object key that carried them.

    ``d_witness_core.leaves`` drops keys (it only needs the values for IDF); the
    D*/R* schema question is asked about the FIELD, so the key must survive.
    """
    if isinstance(value, dict):
        out: List[Tuple[Optional[str], Any]] = []
        for k, v in value.items():
            out.extend(keyed_leaves(v, str(k)))
        return out
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(keyed_leaves(v, key))
        return out
    return [(key, value)]


_JSON_TYPE = {str: "string", bool: "boolean", int: "integer", float: "number"}


def _json_type(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "boolean"
    for py, name in _JSON_TYPE.items():
        if isinstance(value, py):
            return name
    if value is None:
        return "null"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return None


def value_recoverability(key: Optional[str], value: Any, partition: Dict[str, Any]) -> Dict[str, Any]:
    """D*/R* type of one JSON leaf (2608.16370 section 3.3).

    The SCHEMA question, not a model question: is there a declared retrieval tool
    whose RETURN field could carry this value?

    * ``mode="schema"`` -- the partition declares ``retrieval_returns``:
      ``{tool: {field: type}}`` or ``{tool: [field, ...]}``.  D iff some retrieval
      tool declares a field of this name (case-insensitive) whose declared type,
      when present, matches the value's JSON type.
    * ``mode="heuristic"`` -- no return schemas on disk: D iff the key's tokens
      overlap some retrieval tool NAME's tokens.  Declared, never mixed silently.
    """
    returns = partition.get("retrieval_returns") or {}
    if returns:
        want = (key or "").lower()
        for tool, fields in returns.items():
            if isinstance(fields, dict):
                for field, ftype in fields.items():
                    if str(field).lower() == want and (
                            not ftype or _json_type(value) in (None, str(ftype))):
                        return {"type": "D", "mode": "schema", "via": "%s.%s" % (tool, field)}
            else:
                for field in fields or []:
                    if str(field).lower() == want:
                        return {"type": "D", "mode": "schema", "via": "%s.%s" % (tool, field)}
        return {"type": "R", "mode": "schema", "via": None}
    ktok = _tokens(key or "")
    if ktok:
        for tool in partition.get("retrieval") or []:
            if ktok & _tokens(tool):
                return {"type": "D", "mode": "heuristic", "via": tool}
    return {"type": "R", "mode": "heuristic", "via": None}


def block_values(text: str) -> List[Tuple[Optional[str], Any]]:
    """Tool name + keyed JSON leaves of every ``<tool_call>`` block in a doc.

    Mirrors ``d_witness_core.target_values``' value set (tool name + argument
    leaves) but keeps keys.  Blocks with no parseable tool call contribute
    nothing; the caller reports ``n_values=0`` as None, never as 0.0.
    """
    out: List[Tuple[Optional[str], Any]] = []
    body = text or ""
    for opener in re.finditer(r"<tool_call>", body):
        close = body.find("</tool_call>", opener.end())
        chunk = body[opener.end(): close if close != -1 else len(body)]
        try:
            obj = json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if isinstance(name, str) and name:
            out.append(("name", name))
        out.extend(keyed_leaves(obj.get("arguments")))
    # de-duplicate while preserving order (a repeated value must not double-count,
    # same discipline as d_witness_core.target_values)
    seen = set()
    dedup: List[Tuple[Optional[str], Any]] = []
    for k, v in out:
        sig = (k, json.dumps(v, ensure_ascii=False, sort_keys=True, default=str))
        if sig in seen:
            continue
        seen.add(sig)
        dedup.append((k, v))
    return dedup


def block_value_strings(text: str) -> List[str]:
    """The FROZEN witness-style value set of a block: tool name + JSON leaves.

    Uses ``d_witness_core.target_values`` / ``leaves`` verbatim so the D*/R*
    denominator is the same value set the witness-IDF locator scores on.
    ``block_values`` is the keyed twin of this list -- the schema question of
    2608.16370 is asked about the FIELD, which ``leaves`` deliberately drops.
    """
    out: List[str] = []
    body = text or ""
    for opener in re.finditer(r"<tool_call>", body):
        close = body.find("</tool_call>", opener.end())
        chunk = body[opener.end(): close if close != -1 else len(body)]
        try:
            obj = json.loads(chunk)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            out.extend(target_values(obj.get("name"), obj.get("arguments")))
        else:
            out.extend(witness_leaves(obj))
    return list(dict.fromkeys(out))


def block_r_frac(text: str, partition: Dict[str, Any]) -> Dict[str, Any]:
    """``r_frac_k = |R-type| / |all|`` for one history block."""
    vals = block_values(text)
    if not vals:
        return {"r_frac": None, "n_values": 0, "n_r": 0, "modes": []}
    types = [value_recoverability(k, v, partition) for k, v in vals]
    n_r = sum(1 for t in types if t["type"] == "R")
    modes = sorted({t["mode"] for t in types})
    return {"r_frac": n_r / len(types), "n_values": len(types), "n_r": n_r, "modes": modes}


#: Keys ``dstar_features_for_qid`` always emits (so a row with no sidecar can be
#: written with every one of them null instead of with a ragged schema).
DSTAR_FEATURE_KEYS: Tuple[str, ...] = (
    "irb_s_t_r_weighted", "irb_s_t_dropped_only", "irb_s_t_scope",
    "irb_r_frac_dropped_mean", "irb_r_frac_all_mean",
    "irb_n_dropped_blocks", "irb_n_blocks_with_values",
    "irb_dropped_text_available", "irb_dr_mode", "irb_aggregate_loss_frac",
)
#: Emitted beside the D*/R* features but NOT risk scores: they say which halves of
#: s_t exist and which mode produced the typing.  They carry no orientation and
#: must never be scored -- the orientations config declares only the seven scored
#: columns, and a column with no declared orientation raises in ``orient``.
DSTAR_DIAGNOSTIC_COLUMNS: Tuple[str, ...] = (
    "irb_s_t_scope", "irb_dropped_text_available", "irb_dr_mode",
)


def dstar_features_for_qid(
    docs: Sequence[str],
    partition: Dict[str, Any],
    *,
    dropped_docs: Sequence[int] = (),
    dropped_doc_texts: Optional[Sequence[str]] = None,
    gist_tokens: Optional[float] = None,
    original_tokens: Optional[float] = None,
) -> Dict[str, Any]:
    """Prefix-only D*/R* features for one decision step (2608.16370 section 3.3).

    TWO COORDINATE SYSTEMS -- read before touching this function.  In the sidecar
    schema (``agent/t34_dump_sidecar.py``) ``docs`` holds ONLY the KEPT blocks, in
    order: every entry of ``docs`` is a block the model actually saw.
    ``dropped_docs`` holds indices into the POST-SPLIT history list, which is a
    DIFFERENT list, so those indices can never be used to index ``docs`` -- a
    block is dropped precisely by NOT being in ``docs``.  The dropped blocks'
    text exists only under the optional ``dropped_doc_texts`` key (the dumper's
    ``--with_dropped_text``).  When it is absent the dropped half of ``s_t`` is
    UNAVAILABLE and is reported as None: never as a filtered subset of ``docs``,
    never as 0.

    ``s_t`` = sum over lost-or-compressed blocks of ``r_frac_k`` weighted by how
    much of the block was destroyed: weight 1.0 for a dropped block, and the
    aggregate information-loss fraction ``1 - gist/original`` for a kept (i.e.
    merely compressed) one -- per-block ratios are not on disk, see DEVIATIONS.
    ``irb_s_t_scope`` names which halves the emitted sum actually contains:

      ``kept_and_dropped``     both halves defined (complete s_t)
      ``kept_only_partial``    dropped text absent -> dropped half missing
      ``dropped_only_partial`` no aggregate loss fraction -> kept half unweighted
      ``None``                 no block anywhere carries a value

    ``s_t_dropped_only`` is the dropped half alone, so the aggregate-weight
    assumption stays checkable.  Zero forward passes, computed before generation.
    """
    dropped_idx = [int(k) for k in (dropped_docs or [])]
    n_dropped = len(dropped_idx)
    kept_blocks = [block_r_frac(t, partition) for t in (docs or [])]
    have_dropped_text = dropped_doc_texts is not None
    dropped_blocks = ([block_r_frac(t, partition) for t in dropped_doc_texts]
                      if have_dropped_text else [])
    loss_frac: Optional[float] = None
    if gist_tokens is not None and original_tokens:
        try:
            loss_frac = max(0.0, min(1.0, 1.0 - float(gist_tokens) / float(original_tokens)))
        except ZeroDivisionError:
            loss_frac = None

    kept_r = [b["r_frac"] for b in kept_blocks if b["r_frac"] is not None]
    dropped_r = [b["r_frac"] for b in dropped_blocks if b["r_frac"] is not None]
    n_defined = len(kept_r) + len(dropped_r)

    # kept half: the block survived but was compressed, so its weight is the
    # aggregate loss fraction.  Undefined (not 0) when that fraction is unknown.
    if not kept_r:
        s_kept: Optional[float] = 0.0
    elif loss_frac is None:
        s_kept = None
    else:
        s_kept = float(sum(r * loss_frac for r in kept_r))
    # dropped half: weight 1.0, computable ONLY from dropped_doc_texts.  With no
    # dropped blocks at all the half is trivially complete and equal to 0.
    dropped_side_available = bool(have_dropped_text or n_dropped == 0)
    s_drop: Optional[float] = float(sum(dropped_r)) if dropped_side_available else None

    if n_defined == 0:
        s_t: Optional[float] = None
        s_kept = None
        s_drop = None
        scope: Optional[str] = None
    elif s_kept is not None and s_drop is not None:
        s_t, scope = s_kept + s_drop, "kept_and_dropped"
    elif s_kept is not None:
        s_t, scope = s_kept, "kept_only_partial"
    elif s_drop is not None:
        s_t, scope = s_drop, "dropped_only_partial"
    else:
        s_t, scope = None, None

    modes = sorted({m for blk in (kept_blocks + dropped_blocks) for m in blk["modes"]})
    return {
        "irb_s_t_r_weighted": s_t,
        "irb_s_t_dropped_only": s_drop,
        "irb_s_t_scope": scope,
        "irb_r_frac_dropped_mean": (float(np.mean(dropped_r)) if dropped_r else None),
        "irb_r_frac_all_mean": (float(np.mean(kept_r + dropped_r)) if n_defined else None),
        "irb_n_dropped_blocks": n_dropped,
        "irb_n_blocks_with_values": n_defined,
        "irb_dropped_text_available": dropped_side_available,
        "irb_dr_mode": ("+".join(modes) if modes else None),
        "irb_aggregate_loss_frac": loss_frac,
    }


# --------------------------------------------------------------------------
# (C) Last Step Matters -- arXiv 2608.29685
# --------------------------------------------------------------------------

N_PROGRESS_POINTS = 11               # section 3.4: 0%, 10%, ..., 100%
MIN_STEPS_FOR_TRUNCATION = 11        # section 3.4: trajectories with >= 11 steps only
REDUCERS: Tuple[str, ...] = ("mean", "running_max", "running_min", "last", "last3", "last5")
#: section 3 signal battery.  verbal_confidence is never elicited by our harness;
#: the four logprob-derived ones exist only if the proxy log carries the columns.
BASE_STEP_SIGNALS: Tuple[str, ...] = (
    "verbal_confidence", "ppl", "max_nll", "entropy", "max_token_entropy",
)
#: column each base signal would be read from on a proxy row.  Nothing is derived
#: from a column that does not exist -- absent stays absent.
SIGNAL_COLUMNS: Dict[str, str] = {
    "verbal_confidence": "verbal_confidence",
    "ppl": "ppl",
    "max_nll": "max_nll",
    "entropy": "entropy",
    "max_token_entropy": "max_token_entropy",
}


ORIENTATIONS_PATH = _HERE.parent / "configs/t34/orientations_bencha.json"

#: why each declared orientation has the sign it has.  Flat name -> reason; the
#: signs themselves live in configs/t34/orientations_bencha.json (the config is
#: the declaration, this is its rationale, and a unit test keeps them in step).
ORIENTATION_RATIONALE: Dict[str, str] = {
    "irb_s_t_r_weighted": "more history-only (R-type) content destroyed = riskier",
    "irb_s_t_dropped_only": "same, counting only outright dropped blocks",
    "irb_r_frac_dropped_mean": "dropped blocks that were mostly R-type are unrecoverable by re-query",
    "irb_r_frac_all_mean": "an R-heavy history has no self-service recovery path",
    "irb_n_dropped_blocks": "length / cap control comparator: more blocks lost = riskier",
    "irb_n_blocks_with_values": "denominator health: more parseable blocks = better grounded",
    "irb_aggregate_loss_frac": "larger aggregate information loss = riskier",
    "ceq_compression_ratio": "more aggressive compression = riskier",
    "ceq_dropped_docs_n": "more blocks dropped by the tail window = riskier",
    "ceq_kept_history_tokens": "more raw history kept = safer",
    "ceq_gist_tokens": "more gist budget for the same history = less loss = safer",
    "ceq_position_gap": "a larger position-ledger / repair-frame gap = riskier",
    "ceq_hybrid_tail": "a larger raw tail = safer",
    "ceq_z_dev": "Z_t above its own equilibrium = riskier (2510.07777 band)",
    "lsm_verbal_confidence_*": "higher self-reported confidence = safer",
    "lsm_ppl_*": "higher perplexity = riskier",
    "lsm_max_nll_*": "higher max NLL = riskier",
    "lsm_entropy_*": "higher entropy = riskier",
    "lsm_max_token_entropy_*": "higher max token entropy = riskier",
}


def lsm_feature_name(signal: str, reducer: str) -> str:
    """Canonical feature name for a (base signal, temporal reducer) pair."""
    return "lsm_%s_%s" % (signal, reducer)


def declared_orientations() -> Dict[str, int]:
    """Pre-declared risk orientations for this unit (+1 higher = riskier)."""
    return C.load_orientations(ORIENTATIONS_PATH)


def orient(name: str, values: np.ndarray, orientations: Optional[Dict[str, int]] = None) -> np.ndarray:
    """Multiply a raw score by its PRE-DECLARED orientation.

    Pitfall guard: an un-oriented score must never be scored or residualised.
    A missing declaration is an error, not a default of +1.
    """
    table = orientations if orientations is not None else declared_orientations()
    if name not in table:
        raise KeyError("no declared orientation for feature %r" % (name,))
    return np.asarray(values, dtype=float) * float(table[name])


def reduce_series(values: Sequence[Optional[float]], reducer: str) -> Optional[float]:
    """One of the six temporal reduction operators (2608.29685 section 3)."""
    vals = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not vals:
        return None
    if reducer == "mean":
        return float(np.mean(vals))
    if reducer == "running_max":
        return float(np.max(vals))
    if reducer == "running_min":
        return float(np.min(vals))
    if reducer == "last":
        return float(vals[-1])
    if reducer == "last3":
        return float(np.mean(vals[-3:]))
    if reducer == "last5":
        return float(np.mean(vals[-5:]))
    raise ValueError("unknown reducer %r" % (reducer,))


def progress_prefix_lengths(n_steps: int, n_points: int = N_PROGRESS_POINTS) -> List[int]:
    """Prefix length at each of the 11 equally spaced progress points.

    ``length = max(1, round(f * n))`` with ``f = i/(n_points-1)``.  The prefix at
    point i contains ONLY steps ``[0, length)`` -- a_t may never read a token
    after t (causality rule).  Point 0 is the first step, point 10 the whole
    trajectory.
    """
    out = []
    for i in range(n_points):
        f = i / (n_points - 1)
        out.append(max(1, int(round(f * n_steps))))
    return out


def group_by_conversation(rows: Sequence[Dict[str, Any]], *, arm: Optional[str] = None
                          ) -> Dict[str, List[Dict[str, Any]]]:
    """conv_id -> rows sorted by turn (the causal step order on the bench face)."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        if arm is not None and row.get("arm") != arm:
            continue
        conv = row.get("conv_id")
        if conv is None:
            continue
        out.setdefault(str(conv), []).append(row)
    for conv in out:
        out[conv].sort(key=lambda r: (r.get("turn") if r.get("turn") is not None else 0,
                                      r.get("ts") or 0.0))
    return out


def available_step_signals(rows: Sequence[Dict[str, Any]]) -> Dict[str, bool]:
    """Which of the five per-step signals actually exist in these log rows.

    Absent means absent: the caller must report None for the signal, not impute a
    value.  verbal_confidence is expected to be absent (our harness never elicits
    it) and PPL/NLL/entropy are absent until ``normalize_response`` stops dropping
    logprobs.
    """
    out = {}
    for sig, col in SIGNAL_COLUMNS.items():
        out[sig] = any(r.get(col) is not None for r in rows)
    return out


def extract_step_signal(row: Dict[str, Any], signal: str) -> Optional[float]:
    value = row.get(SIGNAL_COLUMNS[signal])
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def backward_truncation_table(
    convs: Dict[str, List[Dict[str, Any]]],
    labels: Dict[str, int],
    *,
    signals: Sequence[str] = BASE_STEP_SIGNALS,
    reducers: Sequence[str] = REDUCERS,
    min_steps: int = MIN_STEPS_FOR_TRUNCATION,
    n_points: int = N_PROGRESS_POINTS,
    label_kind: str = "final_step_cw",
) -> Dict[str, Any]:
    """Backward truncation protocol (2608.29685 section 3.4).

    For every qualifying trajectory and every progress point, reduce the PREFIX of
    each per-step signal with each of the six operators and score the reduction
    against the trajectory-level label with AUROC.  The base rate is returned in
    the same object -- the paper's cleanest lesson is that precision must be read
    against prevalence.

    ``labels`` maps conv_id -> 1/0 (the final step's C->W, or the task-level
    oracle).  Both are LABEL-side quantities computed by the caller from the full
    arm; nothing here reads them as a feature.
    """
    orientations = declared_orientations()
    eligible = {c: rows for c, rows in convs.items()
                if len(rows) >= min_steps and c in labels}
    avail = available_step_signals([r for rows in convs.values() for r in rows])
    y = np.array([int(labels[c]) for c in sorted(eligible)], dtype=int)
    table: List[Dict[str, Any]] = []
    for p in range(n_points):
        for sig in signals:
            if not avail.get(sig):
                table.append({"progress_pct": p * 10, "signal": sig, "reducer": None,
                              "auroc": None, "n": len(eligible), "reason": "signal absent"})
                continue
            for red in reducers:
                scores = []
                for conv in sorted(eligible):
                    rows = eligible[conv]
                    cut = progress_prefix_lengths(len(rows), n_points)[p]
                    prefix = rows[:cut]
                    scores.append(reduce_series([extract_step_signal(r, sig) for r in prefix], red))
                arr = np.array([np.nan if s is None else s for s in scores], dtype=float)
                name = lsm_feature_name(sig, red)
                arr = orient(name, arr, orientations)
                ok = np.isfinite(arr)
                a = C.auroc(arr[ok], y[ok]) if ok.sum() and 0 < y[ok].sum() < ok.sum() else None
                table.append({"progress_pct": p * 10, "signal": sig, "reducer": red,
                              "feature": name, "orientation": orientations[name],
                              "auroc": a, "n": int(ok.sum()), "n_pos": int(y[ok].sum())})
    return {
        "paper": "2608.29685",
        "label_kind": label_kind,
        "n_trajectories_total": len(convs),
        "n_trajectories_eligible": len(eligible),
        "min_steps": min_steps,
        "signals_available": avail,
        "base_rate": (float(y.mean()) if y.size else None),
        "base_rate_900_reference": C.BASE_RATE_900,
        "rows": table,
        "note": ("running_min is reported beside last, never instead of it -- that is the "
                 "comparison 2608.29685 section 4.3 says you lose"),
        "causality_note": ("progress point 100% includes the final step itself, whose "
                           "generation is what the C->W label is computed from; that "
                           "column is the paper's completion column (their AUROC 0.85) "
                           "and is POST-generation, so it is a measurement, never a "
                           "decision-legal pre-generation trigger feature.  Only points "
                           "strictly before the labelled step are gate-legal."),
    }


def combination_classifier(
    convs: Dict[str, List[Dict[str, Any]]],
    labels: Dict[str, int],
    *,
    progress_index: int,
    signals: Sequence[str] = BASE_STEP_SIGNALS,
    reducers: Sequence[str] = REDUCERS,
    min_steps: int = MIN_STEPS_FOR_TRUNCATION,
    n_points: int = N_PROGRESS_POINTS,
) -> Dict[str, Any]:
    """Their section 4.3 combination test, with GROUPED CV instead of stratified.

    Groups are conversations, so no two steps of one trajectory straddle a fold.
    The paper's own 8-rollouts-per-task structure has the same exposure and it
    used stratified folds; we declare the difference rather than copy it.
    """
    eligible = sorted(c for c, rows in convs.items() if len(rows) >= min_steps and c in labels)
    if not eligible:
        return {"n": 0, "cv": "grouped (conversation)", "paper_cv": "stratified"}
    avail = available_step_signals([r for rows in convs.values() for r in rows])
    cols = [(s, r) for s in signals if avail.get(s) for r in reducers]
    if not cols:
        return {"n": len(eligible), "n_features": 0, "reason": "no per-step signal present",
                "cv": "grouped (conversation)", "paper_cv": "stratified"}
    X = np.zeros((len(eligible), len(cols)), dtype=float)
    for i, conv in enumerate(eligible):
        rows = convs[conv]
        cut = progress_prefix_lengths(len(rows), n_points)[progress_index]
        prefix = rows[:cut]
        for j, (sig, red) in enumerate(cols):
            v = reduce_series([extract_step_signal(r, sig) for r in prefix], red)
            X[i, j] = np.nan if v is None else v
    y = np.array([int(labels[c]) for c in eligible], dtype=int)
    groups = np.array(eligible)
    res = C.nested_cv_logistic(X, y, groups, outer_folds=min(5, len(set(groups))))
    res["feature_columns"] = ["%s__%s" % (s, r) for s, r in cols]
    res["cv"] = "grouped (conversation)"
    res["paper_cv"] = "stratified (2608.29685 section 4.3) -- deliberately NOT copied"
    res["progress_pct"] = progress_index * 10
    res.pop("oof_scores", None)
    res.pop("labels", None)
    res.pop("groups", None)
    res.pop("scored_mask", None)
    return res


def post_divergence_convergence(
    convs: Dict[str, List[Dict[str, Any]]],
    *,
    k_max: int = 5,
) -> Dict[str, Any]:
    """Path-switch analogue as a CONFOUND CONTROL (2608.29685 section 5, migration 6).

    From the recover bookkeeping ``RecoverState.check`` merges into each log row
    (``match / diverged_now / re_diverged / tracking_lost``): after the compressed
    arm diverges at step t, how often does it come back to the reference action by
    step t+k?  This is SPEC 7.2 Q2's downstream persistence K, which has zero rows
    today.  It is a control, never a feature.
    """
    per_k = {k: {"n_divergences": 0, "converged": 0} for k in range(1, k_max + 1)}
    n_div = 0
    n_lost = 0
    n_redi = 0
    for rows in convs.values():
        for i, row in enumerate(rows):
            if not row.get("diverged_now"):
                continue
            n_div += 1
            for k in range(1, k_max + 1):
                if i + k >= len(rows):
                    continue
                per_k[k]["n_divergences"] += 1
                window = rows[i + 1: i + 1 + k]
                if any(w.get("match") is True for w in window):
                    per_k[k]["converged"] += 1
        n_lost += sum(1 for r in rows if r.get("tracking_lost"))
        n_redi += sum(1 for r in rows if r.get("re_diverged"))
    for k, rec in per_k.items():
        rec["rate"] = (rec["converged"] / rec["n_divergences"]) if rec["n_divergences"] else None
    return {
        "paper": "2608.29685",
        "n_divergence_events": n_div,
        "n_tracking_lost_rows": n_lost,
        "n_re_diverged_rows": n_redi,
        "self_convergence": per_k,
        "note": ("tracking_lost / re_diverged rows carry no reference action afterwards, so "
                 "they can never count as converged -- report them beside the rate"),
    }


# --------------------------------------------------------------------------
# (D) Context Equilibria -- arXiv 2510.07777
# --------------------------------------------------------------------------

#: prefix-only, pre-generation components of Z_t (S8/S9/S10 fields already logged).
Z_PRE_COMPONENTS: Tuple[str, ...] = (
    "compression_ratio", "dropped_docs_n", "kept_history_tokens",
    "gist_tokens", "position_gap", "hybrid_tail",
)

#: Which of the six a BENCH proxy request row can carry.  ``_log_request``
#: (c2kv/benchmarks/proxy.py:1344-1381) writes gist_tokens, original_tokens,
#: dropped_docs and repair_frame -- so compression_ratio, dropped_docs_n,
#: gist_tokens and position_gap are computable on that face.  It writes NEITHER
#: kept_history_tokens NOR hybrid_top_k: those two exist only on BATTERY rows
#: (agent/d1_arms.py:347 and the manifest kv_recipe).  The bench-face Z^pre is
#: therefore a FOUR-component vector; making it six needs one line in proxy.py,
#: which this unit may not edit.  ``z_pre_coverage`` measures it per log rather
#: than asserting it, and every absent component stays None.
BENCH_LOG_Z_PRE_COMPONENTS: Tuple[str, ...] = (
    "compression_ratio", "dropped_docs_n", "gist_tokens", "position_gap",
)
BATTERY_ONLY_Z_PRE_COMPONENTS: Tuple[str, ...] = ("kept_history_tokens", "hybrid_tail")


def z_pre_components(row: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Reference-free ``Z_t^pre`` components from one prefix (2510.07777 migration).

    Every value is available BEFORE generation, so a fire decision built on them is
    causal.  ``None`` where the field is absent -- ``metadata.sglang_runtime`` was
    null for the whole September matrix and is not back-fillable, so the position
    gap is forward-only.
    """
    def num(*keys: str) -> Optional[float]:
        for k in keys:
            v = row.get(k)
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                return float(len(v))
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
        return None

    gist = num("gist_tokens")
    # the ORIGINAL history size is ``original_tokens`` on a proxy row and
    # ``doc_tokens`` on a battery row.  ``compressed_history_tokens`` is the
    # POST-compression count (it equals gist_tokens on the frozen battery), so
    # using it as the numerator would silently report a ratio of 1.0.
    original = num("original_tokens", "doc_tokens")
    dropped = row.get("dropped_docs")
    if isinstance(dropped, (list, tuple)):
        dropped_n: Optional[float] = float(len(dropped))
    else:
        dropped_n = num("dropped_docs")
    ratio: Optional[float] = None
    if row.get("actual_compression_ratio") is not None:
        ratio = num("actual_compression_ratio")   # the harness' own measured ratio
    elif gist and original:
        ratio = float(original) / float(gist)
    return {
        "compression_ratio": ratio,
        "dropped_docs_n": dropped_n,
        "kept_history_tokens": num("kept_history_tokens"),
        "gist_tokens": gist,
        "position_gap": num("repair_frame_delta", "repair_frame", "position_ledger_gap"),
        "hybrid_tail": num("hybrid_top_k"),
    }


def z_pre_coverage(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-component defined counts for a log, and which components it lacks.

    A runner reading ``Z_PRE_COMPONENTS`` expects six numbers; a bench proxy log
    can only supply four (see ``BENCH_LOG_Z_PRE_COMPONENTS``).  This reports the
    shortfall as a named, countable fact instead of leaving the caller to notice
    that two columns are all-null.
    """
    per: Dict[str, Dict[str, Any]] = {}
    for name in Z_PRE_COMPONENTS:
        vals = [z_pre_components(r).get(name) for r in rows]
        per[name] = {
            "n_defined": sum(1 for v in vals if v is not None),
            "n_rows": len(vals),
            "battery_only_on_the_bench_face": name in BATTERY_ONLY_Z_PRE_COMPONENTS,
        }
    absent = sorted(n for n, v in per.items() if v["n_defined"] == 0)
    return {
        "components": per,
        "n_components_declared": len(Z_PRE_COMPONENTS),
        "n_components_present": len(Z_PRE_COMPONENTS) - len(absent),
        "components_absent_from_this_log": absent,
        "bench_face_expected": list(BENCH_LOG_Z_PRE_COMPONENTS),
        "note": ("kept_history_tokens and hybrid_tail are not written by "
                 "benchmarks/proxy.py:_log_request; on a bench log Z^pre is a "
                 "4-component vector, not 6."),
    }


def fit_delta_regression(series: Sequence[Sequence[float]]) -> Dict[str, Any]:
    """OLS ``dZ_t = a + b Z_t`` pooled over trajectories (2510.07777 section 11.2).

    Returns a, b, the empirical equilibrium ``Z* = -a/b``, residual sd sigma_eta,
    R^2 and the pair count.  ``Z*`` is None when b is 0 (no fixed point).
    """
    zs: List[float] = []
    dz: List[float] = []
    n_traj = 0
    for s in series:
        vals = [float(v) for v in s if v is not None and np.isfinite(float(v))]
        if len(vals) < 2:
            continue
        n_traj += 1
        for t in range(len(vals) - 1):
            zs.append(vals[t])
            dz.append(vals[t + 1] - vals[t])
    if len(zs) < 3:
        return {"a": None, "b": None, "z_star": None, "sigma_eta": None,
                "r2": None, "n_pairs": len(zs), "n_trajectories": n_traj}
    X = np.column_stack([np.ones(len(zs)), np.asarray(zs, dtype=float)])
    yv = np.asarray(dz, dtype=float)
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = yv - X @ coef
    dof = max(1, len(yv) - 2)
    sigma = float(np.sqrt(float(resid @ resid) / dof))
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = float(1.0 - float(resid @ resid) / ss_tot) if ss_tot > 0 else None
    return {"a": a, "b": b, "z_star": (-a / b) if b else None, "sigma_eta": sigma,
            "r2": r2, "n_pairs": len(zs), "n_trajectories": n_traj}


def bootstrap_equilibrium(series: Sequence[Sequence[float]], reps: int = 2000,
                          seed: int = 20260905, alpha: float = 0.05) -> Dict[str, Any]:
    """Trajectory-level bootstrap CIs for (a, b, Z*) (2510.07777 section 12)."""
    rng = np.random.default_rng(seed)
    pool = [list(s) for s in series]
    if not pool:
        return {"n_trajectories": 0}
    keep: Dict[str, List[float]] = {"a": [], "b": [], "z_star": []}
    for _ in range(reps):
        pick = rng.integers(0, len(pool), size=len(pool))
        fit = fit_delta_regression([pool[i] for i in pick])
        if fit["b"] is None:
            continue
        keep["a"].append(fit["a"])
        keep["b"].append(fit["b"])
        if fit["z_star"] is not None:
            keep["z_star"].append(fit["z_star"])
    out: Dict[str, Any] = {"n_trajectories": len(pool), "reps": reps}
    for name, vals in keep.items():
        if vals:
            lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            out[name + "_ci"] = [float(lo), float(hi)]
        else:
            out[name + "_ci"] = None
    return out


def white_noise_b_null(series: Sequence[Sequence[float]], reps: int = 2000,
                       seed: int = 20260905, alpha: float = 0.05) -> Dict[str, Any]:
    """MANDATORY pre-gate: is the fitted b distinguishable from -1?

    Regressing ``dZ_t = Z_{t+1} - Z_t`` on ``Z_t`` produces b = -1 mechanically when
    Z is white noise, so a b near -1 is a mean-reversion artefact, not a restoring
    force (2510.07777 Table tab:equilibrium_core has four of six fitted b in
    [-1.05, -0.96]).  Simulate white noise with the OBSERVED variance and the same
    trajectory lengths, fit b the same way, and compare.
    """
    lengths = [len([v for v in s if v is not None and np.isfinite(float(v))]) for s in series]
    lengths = [n for n in lengths if n >= 2]
    flat = np.array([float(v) for s in series for v in s
                     if v is not None and np.isfinite(float(v))], dtype=float)
    if not lengths or flat.size < 3:
        return {"n_trajectories": 0, "distinguishable_from_minus_one": None}
    mu, sd = float(flat.mean()), float(flat.std(ddof=1))
    rng = np.random.default_rng(seed)
    bs: List[float] = []
    for _ in range(reps):
        sim = [rng.normal(mu, sd if sd > 0 else 1.0, size=n).tolist() for n in lengths]
        fit = fit_delta_regression(sim)
        if fit["b"] is not None:
            bs.append(fit["b"])
    if not bs:
        return {"n_trajectories": len(lengths), "distinguishable_from_minus_one": None}
    lo, hi = np.percentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n_trajectories": len(lengths), "reps": reps,
            "null_b_mean": float(np.mean(bs)), "null_b_ci": [float(lo), float(hi)],
            "matched_variance": sd ** 2}


def equilibrium_gate(series: Sequence[Sequence[float]], *, reps: int = 1000,
                     seed: int = 20260905) -> Dict[str, Any]:
    """Fit + pre-gate.  The band is WITHHELD when b is not distinguishable from -1.

    Gate rule, pre-registered: b is distinguishable from -1 iff its trajectory
    bootstrap CI excludes -1 AND the point estimate falls outside the white-noise
    null CI.  On failure the card's own demotion condition applies: the equilibrium
    layer contributes nothing and the design collapses to a per-step threshold.
    """
    fit = fit_delta_regression(series)
    boot = bootstrap_equilibrium(series, reps=reps, seed=seed)
    null = white_noise_b_null(series, reps=reps, seed=seed)
    b_ci = boot.get("b_ci")
    null_ci = null.get("null_b_ci")
    ci_excludes = bool(b_ci and not (b_ci[0] <= -1.0 <= b_ci[1]))
    outside_null = bool(null_ci and fit["b"] is not None
                        and not (null_ci[0] <= fit["b"] <= null_ci[1]))
    passed = bool(ci_excludes and outside_null)
    return {
        "paper": "2510.07777",
        "fit": fit,
        "bootstrap": boot,
        "white_noise_null": null,
        "b_ci_excludes_minus_one": ci_excludes,
        "b_outside_white_noise_null": outside_null,
        "gate": "PASS" if passed else "FAIL",
        "z_star": fit["z_star"] if passed else None,
        "sigma_eta": fit["sigma_eta"] if passed else None,
        "action_on_fail": ("b indistinguishable from -1: mean reversion, not a restoring "
                           "force.  Do not report an equilibrium band; fall back to the "
                           "per-step threshold form."),
    }


def fire_deviation(z: Optional[float], z_star: Optional[float], sigma: Optional[float],
                   kappa: float) -> Optional[bool]:
    """``fire iff Z_t - Z* > kappa * sigma_eta`` (2510.07777 migration)."""
    if z is None or z_star is None or sigma is None or not np.isfinite(sigma):
        return None
    return bool(z - z_star > kappa * sigma)


def _mean_or_none(values: Sequence[float]) -> Optional[float]:
    return float(np.mean(values)) if len(values) else None


def select_kappa(
    series_by_group: Dict[str, Sequence[float]],
    labels: Dict[str, int],
    *,
    kappa_grid: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
    folds: int = 3,
    seed: int = 20260905,
    target_fire_rate: Optional[float] = None,
) -> Dict[str, Any]:
    """Choose kappa in INNER folds of grouped CV; never on the evaluation rows.

    Each group (conversation) contributes its last Z and its label; within each
    inner fold the equilibrium is fitted on that fold's TRAINING groups only.

    CRITERION.  ``kappa`` only moves an operating point along one fixed score,
    the standardised deviation ``d = (Z_t - Z*) / sigma_eta``, so it must be
    chosen the way an operating point is chosen: PRECISION at a declared fire
    rate.  (Ranking metrics computed on the two-valued fired indicator -- AP or
    AUROC on ``fired`` -- are degenerate one-threshold summaries and are NOT
    comparable to the AUPRC/AUROC numbers the rest of the contract reports, so
    none is computed here.)  Candidates are restricted to those whose inner-fold
    fire rate is at or below ``target_fire_rate`` (default: the label prevalence
    of the supplied groups, i.e. fire about as often as the event happens) and
    the highest inner precision wins, ties going to the LARGER kappa (fires
    less).  ``chance_precision`` is that same prevalence -- the evaluation-frame
    chance level -- so a precision is never readable without its chance line.
    ``deviation_auroc`` is the kappa-FREE AUROC of ``d`` itself: it says whether
    any kappa could work, and it is the only ranking number in this result.
    """
    groups = sorted(g for g in series_by_group if g in labels)
    grid = [float(k) for k in kappa_grid]
    empty = {"kappa": None, "inner_precision": None, "inner_fire_rate": None,
             "inner_recall": None, "deviation_auroc": None, "per_kappa": {},
             "chance_precision": None, "target_fire_rate": None,
             "target_fire_rate_source": None, "constraint_met": None,
             "n_groups": len(groups), "criterion": "precision at a fixed fire rate",
             "note": "selected on inner folds of session-grouped CV only"}
    if len(groups) < 2:
        return dict(empty, reason="not enough groups")
    y_all = [int(labels[g]) for g in groups]
    prevalence = float(np.mean(y_all))
    if target_fire_rate is None:
        target = prevalence
        target_source = "label prevalence of the supplied groups"
    else:
        target = float(target_fire_rate)
        target_source = "caller-supplied"
    empty = dict(empty, chance_precision=prevalence, target_fire_rate=target,
                 target_fire_rate_source=target_source)

    masks = C.grouped_folds(np.array(groups), min(folds, len(groups)), seed)
    per: Dict[float, Dict[str, List[float]]] = {
        k: {"fire_rate": [], "precision": [], "recall": []} for k in grid}
    dev_auroc: List[float] = []
    n_folds_used = 0
    for mask in masks:
        tr = [g for g, m in zip(groups, mask) if not m]
        te = [g for g, m in zip(groups, mask) if m]
        if not tr or not te:
            continue
        fit = fit_delta_regression([series_by_group[g] for g in tr])
        if fit["z_star"] is None or fit["sigma_eta"] is None or not fit["sigma_eta"]:
            continue
        z_last: List[float] = []
        y: List[int] = []
        for g in te:
            vals = [v for v in series_by_group[g] if v is not None]
            if not vals:
                continue
            z_last.append(float(vals[-1]))
            y.append(int(labels[g]))
        if not z_last:
            continue
        n_folds_used += 1
        yy = np.asarray(y, dtype=int)
        dev = (np.asarray(z_last, dtype=float) - float(fit["z_star"])) / float(fit["sigma_eta"])
        if len(set(y)) == 2:
            a = C.auroc(dev, yy)
            if a is not None:
                dev_auroc.append(a)
        for kappa in grid:
            fired = np.array([bool(fire_deviation(z, fit["z_star"], fit["sigma_eta"], kappa))
                              for z in z_last], dtype=bool)
            n_fire = int(fired.sum())
            per[kappa]["fire_rate"].append(n_fire / len(fired))
            if n_fire:
                per[kappa]["precision"].append(float((fired & (yy == 1)).sum()) / n_fire)
            if int(yy.sum()):
                per[kappa]["recall"].append(float((fired & (yy == 1)).sum()) / int(yy.sum()))
    if not n_folds_used:
        return dict(empty, reason="no inner fold could be fitted")

    means = {k: {"fire_rate": _mean_or_none(v["fire_rate"]),
                 "precision": _mean_or_none(v["precision"]),
                 "recall": _mean_or_none(v["recall"]),
                 "n_folds_with_fires": len(v["precision"])}
             for k, v in per.items()}
    firing = [k for k in grid if (means[k]["fire_rate"] or 0.0) > 0.0]
    if not firing:
        return dict(empty, per_kappa=means, reason="no kappa on the grid ever fires",
                    deviation_auroc=_mean_or_none(dev_auroc))
    eligible = [k for k in firing if means[k]["fire_rate"] <= target]
    constraint_met = bool(eligible)
    if not eligible:
        # nothing fires rarely enough: fall back to the closest rate, and say so
        closest = min(abs(means[k]["fire_rate"] - target) for k in firing)
        eligible = [k for k in firing
                    if abs(means[k]["fire_rate"] - target) == closest]
    # precision first, then the LARGER kappa (fires less) -- a fixed, declared rule
    best = max(eligible, key=lambda k: ((means[k]["precision"]
                                         if means[k]["precision"] is not None else -1.0), k))
    return dict(empty,
                kappa=best,
                inner_precision=means[best]["precision"],
                inner_fire_rate=means[best]["fire_rate"],
                inner_recall=means[best]["recall"],
                deviation_auroc=_mean_or_none(dev_auroc),
                per_kappa=means,
                constraint_met=constraint_met,
                n_folds_used=n_folds_used)


def battery_z_rows(frame: "C.FrozenFrame") -> List[Dict[str, Any]]:
    """Per-step (NO t axis) form of Z^pre on the frozen battery rows.

    The battery is single-step teacher-forced, so the equilibrium form is not
    defined here; this output is deliberately a separate artefact from the bench
    episode fit.  Only the compressed arm's own prefix scalars are read.
    """
    rows = []
    for qid, c in frame.c2kv_by_qid.items():
        comp = z_pre_components(c)
        rows.append({
            "qid": qid,
            "session_id": c.get("session_id"),
            "form": "per_step",
            **{"ceq_" + k: v for k, v in comp.items()},
        })
    return sorted(rows, key=lambda r: r["qid"])


def z_pre_s0_report(frame: "C.FrozenFrame") -> Dict[str, Any]:
    """S0 control for Z^pre: the SAME components computed on the full arm.

    The default scoring contract wants every feature re-computed on the full arm.
    Z^pre is a statistic of the COMPRESSED prefix, so on the full arm most
    components are structurally undefined or constant -- which is an anchor that
    is structurally zero, not a usable control.  This function measures that
    rather than assuming it, and says so per component.
    """
    per: Dict[str, Dict[str, Any]] = {}
    for name in Z_PRE_COMPONENTS:
        vals = [z_pre_components(f).get(name) for f in frame.full_by_qid.values()]
        defined = [v for v in vals if v is not None]
        uniq = sorted(set(defined))
        per["ceq_" + name] = {
            "n_defined": len(defined), "n_rows": len(vals),
            "n_distinct": len(uniq),
            "usable_as_s0": bool(len(defined) > 0 and len(uniq) > 1),
        }
    return {
        "arm": "full",
        "components": per,
        "note": ("a component with n_distinct <= 1 is a structurally constant anchor and "
                 "must NOT be reported as an S0 control"),
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _load_none_arm_hits(path: Optional[str]) -> Optional[Dict[str, bool]]:
    if not path:
        return None
    rows = C.load_jsonl(path)
    return {r["qid"]: bool(r.get("tool_name_match")) for r in rows if "qid" in r}


def _cmd_classes(args: argparse.Namespace) -> int:
    frame = C.FrozenAssets(Path(args.root)).load()
    flips = C.load_flip_table(Path(args.flip_table)) if args.flip_table else {}
    none_hits = _load_none_arm_hits(args.none_arm)
    classes = assign_solvability_classes(frame.cw_qids(), flips, none_hits)
    n_missing = sum(1 for v in classes.values() if v["flip_table_missing"])
    provenance = {
        "flip_table": str(args.flip_table) if args.flip_table else None,
        "n_rows_without_flip_table": n_missing,
        "flip_table_note": (None if n_missing == 0 else
                            "MISSING INPUT: %d of %d C->W rows have no entry in the flip "
                            "table, so they fall to S3' as UNMEASURED, not as "
                            "measured-unsolvable.  Copy the D-line per-(qid,k) sweep off "
                            "the NPU box before quoting any S1'/S2'/S3' count."
                            % (n_missing, len(classes))),
        "flip_table_sha256": (C.sha256_file(Path(args.flip_table))
                              if args.flip_table and Path(args.flip_table).exists() else None),
        "none_arm": str(args.none_arm) if args.none_arm else None,
        "none_arm_note": (None if none_hits else
                          "no none-arm file supplied: S1' is reported constructively empty "
                          "(greedy + single-step teacher-forced retry is a no-op)"),
        "manifest": str(C.FrozenAssets(Path(args.root)).manifest_path),
    }
    out = freeze_solvability_manifest(Path(args.out), classes, provenance)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


def _cmd_l1_cut(args: argparse.Namespace) -> int:
    frame = C.FrozenAssets(Path(args.root)).load()
    sub = frame.trigger_subset()
    # the decoded-doc sidecar carries the tool schemas the model was shown; without
    # it schema legality is NOT computable and the cut reports "unknown", never
    # a silently-passed row.
    tools_by_qid: Dict[str, List[Dict[str, Any]]] = {}
    if args.sidecar:
        for row in C.load_jsonl(args.sidecar):
            tools_by_qid[row["qid"]] = list(row.get("tools") or [])
    vis = {}
    for rec in sub:
        c = frame.c2kv_by_qid[rec["qid"]]
        vis[rec["qid"]] = l1_visibility(c.get("prediction", ""),
                                        tools_by_qid.get(rec["qid"]))["l1_visible"]
    caveat = cap_caveat([r["censored_at_cap"] for r in sub], frame.cap_tokens())
    n_visible = sum(1 for v in vis.values() if v is True)
    print(json.dumps({
        "n_rows": len(sub),
        "n_l1_visible": n_visible,
        "n_silent": sum(1 for v in vis.values() if v is False),
        "n_unknown": sum(1 for v in vis.values() if v is None),
        "tool_pool_available": bool(tools_by_qid),
        "cap": caveat,
        "explicit_modes": list(EXPLICIT_FAILURE_MODES),
        "silent_modes": list(SILENT_FAILURE_MODES),
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_partition_check(args: argparse.Namespace) -> int:
    obj = load_tool_partition(Path(args.partition))
    report = validate_tool_partition(obj, args.tools.split(",") if args.tools else None)
    report["expected_sha256"] = partition_sha256(obj)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def _cmd_triples(args: argparse.Namespace) -> int:
    rows = C.load_proxy_log(Path(args.proxy_log))
    partition = load_tool_partition(Path(args.partition))
    triples = cost_triples(rows, partition, min_turn=args.min_turn,
                           max_turn=args.max_turn)
    meta = triples[("__meta__", "__meta__")]
    real = {k: v for k, v in triples.items() if k != ("__meta__", "__meta__")}
    payload: Dict[str, Any] = {
        "paper": "2608.16370",
        "min_turn": args.min_turn,
        "max_turn": args.max_turn,
        "turn0_note": "turn 0 excluded: arms are byte-identical at turn-0 step-0",
        "horizon_note": meta["note"],
        "row_accounting": meta,
        "partition_sha256": partition_sha256(partition),
        "triples": [dict(v) for v in real.values()],
    }
    if args.arm_a and args.arm_b:
        payload["comparison"] = irbench_report(triples, arm_a=args.arm_a, arm_b=args.arm_b)
    if args.family:
        comparisons = []
        for item in args.family.split(","):
            a, b, metric = item.split(":")
            comparisons.append((a.strip(), b.strip(), metric.strip()))
        payload["family"] = irbench_family_report(triples, comparisons)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(json.dumps({"n_conversation_arms": len(real),
                      "arms": sorted({k[0] for k in real}),
                      "row_accounting": meta,
                      "partition_sha256": payload["partition_sha256"],
                      "comparison": payload.get("comparison", {}).get("holm"),
                      "family_holm": payload.get("family", {}).get("holm"),
                      "out": args.out}, ensure_ascii=False, indent=2))
    return 0


def _cmd_dstar(args: argparse.Namespace) -> int:
    partition = load_tool_partition(Path(args.partition))
    partition_report = validate_tool_partition(partition)
    frame = C.FrozenAssets(Path(args.root)).load()
    sidecar = {r["qid"]: r for r in C.load_jsonl(args.sidecar)}
    rows = []
    n_no_sidecar = 0
    n_with_dropped = 0
    n_missing_dropped_text = 0
    for rec in frame.trigger_subset():
        qid = rec["qid"]
        side = sidecar.get(qid)
        c = frame.c2kv_by_qid[qid]
        if side is None:
            n_no_sidecar += 1
            feats = {k: None for k in DSTAR_FEATURE_KEYS}
        else:
            # ``docs`` = the KEPT blocks only; ``dropped_docs`` indexes the
            # post-split history list and is a COUNT here, never a selector into
            # ``docs``.  The dropped side needs ``dropped_doc_texts``.
            dropped_idx = side.get("dropped_docs") or []
            dropped_texts = side.get("dropped_doc_texts")
            if dropped_idx:
                n_with_dropped += 1
                if dropped_texts is None:
                    n_missing_dropped_text += 1
            feats = dstar_features_for_qid(
                side.get("docs") or [], partition,
                dropped_docs=dropped_idx,
                dropped_doc_texts=dropped_texts,
                gist_tokens=c.get("gist_tokens"),
                original_tokens=c.get("original_tokens") or c.get("doc_tokens"),
            )
        rows.append({"qid": qid, "session_id": rec["session_id"], **feats})
    n = C.write_features_jsonl(Path(args.out), rows, context="t34 U7a D*/R* features")
    print(json.dumps({
        "written": n, "path": args.out,
        "partition_sha256": partition_sha256(partition),
        "partition_filled": partition_report["filled"],
        "partition_note": partition_report["note"],
        "n_rows_missing_sidecar": n_no_sidecar,
        "n_rows_with_dropped_blocks": n_with_dropped,
        "n_rows_missing_dropped_doc_texts": n_missing_dropped_text,
        "dropped_side_available": bool(n_missing_dropped_text == 0),
        "note": (None if n_missing_dropped_text == 0 else
                 "MISSING INPUT: %d rows have dropped blocks but no dropped_doc_texts, so "
                 "irb_s_t_r_weighted is the KEPT-ONLY partial (irb_s_t_scope="
                 "'kept_only_partial') and irb_s_t_dropped_only is null.  Re-dump the "
                 "sidecar with: python agent/t34_dump_sidecar.py --arm c2kv "
                 "--with_dropped_text" % n_missing_dropped_text),
    }, ensure_ascii=False, indent=2))
    return 0


def _cmd_truncation(args: argparse.Namespace) -> int:
    rows = C.load_proxy_log(Path(args.proxy_log))
    convs = group_by_conversation(rows, arm=args.arm)
    labels = {k: int(v) for k, v in json.loads(Path(args.labels).read_text(encoding="utf-8")).items()}
    table = backward_truncation_table(convs, labels)
    table["combination_at_50pct"] = combination_classifier(convs, labels, progress_index=5)
    text = json.dumps(table, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    best = [r for r in table["rows"] if r.get("auroc") is not None]
    best.sort(key=lambda r: (-r["progress_pct"], -r["auroc"]))
    print(json.dumps({"n_trajectories_eligible": table["n_trajectories_eligible"],
                      "signals_available": table["signals_available"],
                      "base_rate": table["base_rate"],
                      "base_rate_900_reference": table["base_rate_900_reference"],
                      "best_row_at_100pct": (best[0] if best else None),
                      "out": args.out}, ensure_ascii=False, indent=2))
    return 0


def _cmd_selfconv(args: argparse.Namespace) -> int:
    rows = C.load_proxy_log(Path(args.proxy_log))
    convs = group_by_conversation(rows, arm=args.arm)
    print(json.dumps({"self_convergence": post_divergence_convergence(convs, k_max=args.k_max),
                      "bench_regenerate_rescue": bench_regenerate_rescue(convs)},
                     ensure_ascii=False, indent=2))
    return 0


def _cmd_equilibrium(args: argparse.Namespace) -> int:
    rows = C.load_proxy_log(Path(args.proxy_log))
    convs = group_by_conversation(rows, arm=args.arm)
    # The digest fixes the fit pool: "在 C->C 训练池上拟 dZ = a + bZ".  Fitting on
    # every episode, positives included, chooses the band on the rows it is later
    # scored on -- the paper does exactly that (no train/test split, section 12) and
    # that is precisely what we do not inherit.
    labels: Optional[Dict[str, int]] = None
    if args.labels:
        labels = {k: int(v) for k, v in
                  json.loads(Path(args.labels).read_text(encoding="utf-8")).items()}
    coverage = z_pre_coverage(rows)
    if coverage["components"][args.z]["n_defined"] == 0:
        print(json.dumps({
            "abort": True,
            "reason": ("MISSING INPUT: the Z component %r is null on every row of %s, so "
                       "no series can be built." % (args.z, args.proxy_log)),
            "z_pre_coverage": coverage,
        }, ensure_ascii=False, indent=2))
        return 2
    series = []
    used: List[str] = []
    skipped_labeled_positive = 0
    skipped_unlabeled = 0
    series_by_group: Dict[str, List[float]] = {}
    for conv, crows in sorted(convs.items()):
        vals_all = [v for v in (z_pre_components(r).get(args.z) for r in crows)
                    if v is not None]
        if labels is not None and conv in labels and len(vals_all) >= 2:
            # kappa is selected over the WHOLE labeled set (both classes) but only
            # ever inside inner folds, which refit (a, b) on their own training
            # groups -- the outer fit below never sees a positive episode.
            series_by_group[conv] = vals_all
        if labels is not None:
            if conv not in labels:
                skipped_unlabeled += 1
                continue
            if labels[conv] != 0:
                skipped_labeled_positive += 1
                continue
        if len(vals_all) >= 2:
            series.append(vals_all)
            used.append(conv)
    out = equilibrium_gate(series, reps=args.reps)
    out["z_component"] = args.z
    out["z_pre_coverage"] = coverage
    out["fit_pool"] = {
        "labels_supplied": labels is not None,
        "pool": "C->C (label 0) training episodes" if labels is not None else "ALL episodes",
        "n_episodes_fitted": len(series),
        "n_skipped_positive": skipped_labeled_positive,
        "n_skipped_unlabeled": skipped_unlabeled,
        "warning": (None if labels is not None else
                    "no --labels given: (a, b) were fitted on every episode INCLUDING the "
                    "ones the band would later be scored on.  That is the paper's own "
                    "descriptive fit (2510.07777 section 12, no split) and is NOT a "
                    "validated threshold; supply --labels to fit on the C->C pool."),
    }

    # kappa: the fire rule Z_t - Z* > kappa * sigma_eta has no kappa until one is
    # selected, and it can only be selected against labels.
    if labels is None:
        out["kappa_selection"] = {
            "kappa": None,
            "reason": ("MISSING INPUT: no --labels supplied, so kappa cannot be selected; "
                       "the fire rule Z_t - Z* > kappa * sigma_eta is incomplete."),
        }
    else:
        grid = tuple(float(x) for x in str(args.kappa_grid).split(",") if x.strip())
        out["kappa_selection"] = select_kappa(
            series_by_group, labels, kappa_grid=grid, folds=args.kappa_folds,
            target_fire_rate=args.target_fire_rate)
    kappa = out["kappa_selection"].get("kappa")
    if out["gate"] == "PASS" and kappa is not None and out["z_star"] is not None \
            and out["sigma_eta"] is not None:
        out["fire_rule"] = {
            "rule": "fire iff Z_t - Z* > kappa * sigma_eta",
            "z_star": out["z_star"], "sigma_eta": out["sigma_eta"], "kappa": kappa,
            "threshold": float(out["z_star"]) + float(kappa) * float(out["sigma_eta"]),
        }
    else:
        out["fire_rule"] = {
            "rule": None,
            "reason": ("gate FAIL: no band, so no fire rule (fall back to the per-step "
                       "threshold form)" if out["gate"] != "PASS"
                       else "no kappa selected: %s"
                            % out["kappa_selection"].get("reason", "see kappa_selection")),
        }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


def _cmd_battery_z(args: argparse.Namespace) -> int:
    frame = C.FrozenAssets(Path(args.root)).load()
    rows = battery_z_rows(frame)
    n = C.write_features_jsonl(Path(args.out), rows, context="t34 U7a Z^pre per-step features")
    print(json.dumps({"written": n, "path": args.out, "form": "per_step (no t axis)",
                      "s0_full_arm_control": z_pre_s0_report(frame)},
                     ensure_ascii=False, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="t34 U7a bench-side rescoring (BTM / IRBench / Last Step / Context Equilibria)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("classes", help="assign + freeze the S1'/S2'/S3' solvability manifest")
    p.add_argument("--root", default=".")
    p.add_argument("--flip-table", default=None, help="results/t34/flip_table.jsonl")
    p.add_argument("--none-arm", default=None, help="arm results jsonl for the none arm")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_classes)

    p = sub.add_parser("l1-cut", help="L1-visible vs silent cut + the caliber caveat")
    p.add_argument("--root", default=".")
    p.add_argument("--sidecar", default=None,
                   help="results/t34/sidecar_c2kv.jsonl (supplies the tool schemas; "
                        "without it schema legality is reported as unknown)")
    p.set_defaults(func=_cmd_l1_cut)

    p = sub.add_parser("partition-check", help="validate the static tool partition")
    p.add_argument("--partition", required=True)
    p.add_argument("--tools", default=None, help="comma-separated declared tool names")
    p.set_defaults(func=_cmd_partition_check)

    p = sub.add_parser("triples", help="(Q, C_R, C_E) per arm from a proxy log")
    p.add_argument("--proxy-log", required=True)
    p.add_argument("--partition", required=True)
    p.add_argument("--min-turn", type=int, default=1)
    p.add_argument("--max-turn", type=int, default=None,
                   help="fixed interaction horizon (2608.16370 Definition 1; theirs is "
                        "24). Without it the triples are not horizon-bounded and an arm "
                        "that ran longer is charged for the extra turns.")
    p.add_argument("--arm-a", default=None)
    p.add_argument("--arm-b", default=None)
    p.add_argument("--family", default=None,
                   help="declared primary family as armA:armB:metric[,...] -- Holm runs "
                        "ACROSS these cells, which is the paper's family shape")
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_triples)

    p = sub.add_parser("dstar", help="D*/R* prefix features into a features jsonl")
    p.add_argument("--root", default=".")
    p.add_argument("--sidecar", required=True, help="results/t34/sidecar_c2kv.jsonl")
    p.add_argument("--partition", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_dstar)

    p = sub.add_parser("truncation", help="backward truncation AUROC vs progress")
    p.add_argument("--proxy-log", required=True)
    p.add_argument("--labels", required=True, help='json {"conv_id": 0|1}')
    p.add_argument("--arm", default=None)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_truncation)

    p = sub.add_parser("selfconv", help="post-divergence self-convergence rate at t+k")
    p.add_argument("--proxy-log", required=True)
    p.add_argument("--arm", default=None)
    p.add_argument("--k-max", type=int, default=5)
    p.set_defaults(func=_cmd_selfconv)

    p = sub.add_parser("equilibrium", help="dZ = a + bZ fit with the b != -1 pre-gate")
    p.add_argument("--proxy-log", required=True)
    p.add_argument("--z", default="compression_ratio", choices=list(Z_PRE_COMPONENTS))
    p.add_argument("--arm", default=None)
    p.add_argument("--labels", default=None,
                   help='json {"conv_id": 0|1}; the fit uses the C->C (0) pool only. '
                        "Without it (a, b) are fitted on the evaluation rows too, and "
                        "kappa cannot be selected at all.")
    p.add_argument("--kappa-grid", default="0.5,1.0,1.5,2.0,3.0",
                   help="candidate kappas for the fire rule Z_t - Z* > kappa*sigma_eta")
    p.add_argument("--kappa-folds", type=int, default=3,
                   help="inner grouped folds kappa is selected in (never the eval rows)")
    p.add_argument("--target-fire-rate", type=float, default=None,
                   help="fixed fire rate kappa is chosen at; default = the label "
                        "prevalence of the labelled episodes")
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--out", default=None)
    p.set_defaults(func=_cmd_equilibrium)

    p = sub.add_parser("battery-z", help="per-step Z^pre features on the frozen battery")
    p.add_argument("--root", default=".")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_battery_z)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
