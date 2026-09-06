# -*- coding: utf-8 -*-
"""t34 unit U5 / digest section 4.8 -- one extra forward, draft-verify, paired-config
spread, and the SelfCheckGPT N=1 substitution.

Four independent pieces live here because they share one thing: each buys its
statistic with EXACTLY ONE extra deterministic forward (or two generations), and
none of them resamples.

  (A) VeriCache (2605.17613 section 5.1) -- LABEL pass.  Hard token equality between
      the compressed arm's emitted ids and the full-prefix argmax ids.  Produces
      first_div_idx / accept_len / div_region / cap_hit.  These are LABELS: they
      never enter a feature frame, they go to results/t34/vericache_labels.jsonl,
      and their GPU cost is booked offline only.
  (B) AsymSpec (2608.26004 eq:delta / eq:cda) -- FEATURE pass.  D_i = JSD(softmax a_i
      || softmax b_i) in nats between a small drafter reading x_full and the same
      drafter reading a TEXT STAND-IN for the gist view.  Every artifact row is
      stamped proxy="text-stand-in for the gist view"; the divergence is not measured
      against the compressor we ship.
  (C) KV-Eviction Error Certificates (2607.21475 section 6.4 aside) -- FEATURE pass.
      The certificate itself is not portable (it needs a randomized compressor);
      what ports is its corollary: one deterministic compression cannot identify its
      own error, two different deterministic compressions can.  Paired-config spread
      over the SAME frozen history, plus the paper's mandatory baseline (the
      compressed arm's own mean output log-probability) and its two estimands
      (prediction / attribution) reported separately.
  (D) SelfCheckGPT (2303.08896 section 7.3.1) -- FEATURE pass.  Not the sampling
      method (N=4..20 extra generations cost 10-60x the recovery action it would
      gate); only the external-knowledge substitution at N=1, with the decoded
      compressed docs + query as the single reference.

WIRING (bench face, for the pieces that are deployable).  Nothing here edits
benchmarks/.  The pure functions take (request-ish dicts, response text, tool
schemas) and plug in as follows:

  * ``selfcheck_row`` is the HOOK 2 body: benchmarks/proxy.py ``RecoverState.check``
    (:494), after the compressed arm's answer exists and before it is committed.
    Inputs it needs from that scope: ``action_canonical(response)`` (proxy.py:466)
    for the emitted call, and the decoded doc plaintext + current query the proxy
    already holds.  It returns a dict of scalars; the proxy would log them beside
    ``diverged_now`` and never branch on them without a threshold chosen in CV.
    CAVEAT: those two line numbers are the BENCH FACE's, not this worktree's.  The
    benchmarks/proxy.py checked out here (344 lines) defines neither
    ``RecoverState`` nor ``action_canonical``, so the promised diff between the bench
    face's canonicalisation and this unit's local rebuild cannot be run here;
    :func:`bench_face_action_canonical` resolves the symbol when a worktree HAS it and
    the unit test diffs the two definitions then (and skips, loudly, when it does not).
  * ``paired_config_spread_row`` consumes two ALREADY-LOGGED runs; on the bench face
    the two runs are two proxy request-log rows for the same ``(conv_id, turn, fp)``
    under two arms whose configs differ only in the declared axis.  ``Arm.validate``
    (benchmarks/arms.py:178) is not touched: neither run enables ``recover``.
  * (A) and (B) are battery-only and have no bench-face landing point.

RUNBOOK (execution order; [NPU] runs on the Ascend server, [HERE] on this box).

  [HERE] 1. sanity + unit tests
        PYTHONIOENCODING=utf-8 python -m pytest agent/test_t34_extra_forward.py -q

  [NPU]  2. VeriCache label pass (section A).  ~12-30 min GPU for 900 rows.
        python agent/t34_extra_forward.py vericache \
          --model_path <fixed_joint> --tokenizer_path <tok> --dataset_path <ds> \
          --battery_full results/bdf_pilot/d_r2/battery_full.jsonl \
          --battery_c2kv results/bdf_pilot/d_r2/battery_c2kv.jsonl \
          --manifest configs/bdf_pilot/d_cw_manifest_r2.json \
          --capture_steps <capture_dir>/c2kv/p0.steps.jsonl \
          --out results/t34/vericache_labels.jsonl --repeat

  [HERE] 3. VeriCache self-consistency read-out (bf16 shape-dependent rounding gate).
        python agent/t34_extra_forward.py vericache-agree \
          --pass_a results/t34/vericache_labels.jsonl \
          --pass_b results/t34/vericache_labels.repeat.jsonl

  [NPU]  4. AsymSpec drafter pass (section B).  Needs Qwen/Qwen3-1.7B; < 1 h.
        PREREQUISITE (quality, not liveness): dump the sidecar WITH
        --with_dropped_text.  The sidecar's ``docs`` are the KEPT grid rows only, so
        without the dropped blocks' plaintext the tail-window drop is NOT in x_full.
        The pass still scores those rows -- skipping them would delete every row that
        has a drop, i.e. the whole informative subset -- but under the degraded
        proxy_variant='truncation_only_missing_dropped_text', which carries the
        truncation axis alone.  The three proxy_variants must be stratified, never
        pooled.
        python agent/t34_extra_forward.py asymspec \
          --sidecar results/t34/sidecar_c2kv.jsonl \
          --battery_c2kv results/bdf_pilot/d_r2/battery_c2kv.jsonl \
          --drafter Qwen/Qwen3-1.7B --target_tokenizer <tok of the 4B> \
          --out results/t34/features_asymspec.jsonl

  [NPU]  5. paired-config spread (section C).  Two generations per row.
        python agent/t34_extra_forward.py spread-plan --mode doc_permute \
          --witness configs/bdf_pilot/d_witness_r2.json --out configs/t34/spread_plan.json
        # then two capture runs of the frozen battery under config A / config B
        # (t33_capture_npu.sh with the plan applied); afterwards, [HERE]:

  [HERE] 6. spread features from the two capture files.  --sidecar is what attaches
           2607.21475's mandatory baseline (3) -- the dropped_docs / evicted-score-mass
           predicted-weak control -- and the retained-entropy control; without it those
           columns are None and the run prints missing_mandatory_baselines.
        python agent/t34_extra_forward.py spread \
          --capture_a <capA>/c2kv/p0.steps.jsonl --capture_b <capB>/c2kv/p0.steps.jsonl \
          --sidecar results/t34/sidecar_c2kv.jsonl \
          --out results/t34/features_spread.jsonl

  [HERE] 7. SelfCheckGPT N=1 (section D).  (a) is zero GPU; (b) needs a DeBERTa-MNLI.
        python agent/t34_extra_forward.py selfcheck \
          --sidecar results/t34/sidecar_c2kv.jsonl \
          --battery_c2kv results/bdf_pilot/d_r2/battery_c2kv.jsonl \
          --out results/t34/features_selfcheck.jsonl [--nli_model microsoft/deberta-...]

  [HERE] 8. score any feature file on the 161-row trigger subset (same rows, n / n_pos
           printed; chance AP = the frame's own prevalence, never 0.1033)
        python agent/t34_extra_forward.py score --features results/t34/features_*.jsonl \
          --root . --orientations configs/t34/orientations_extra.json

ORIENTATIONS.  configs/t34/orientations_extra.json declares, before any scoring, the risk
orientation of every feature this unit emits (+1 = higher is riskier for C->W, -1 = higher
is safer).  ``score_feature_table`` multiplies by it, so no un-oriented score is ever
compared or residualised.  The VeriCache columns are LABELS and deliberately carry no
orientation; so does everything in t34_contextcite.py.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import t34_common as C  # noqa: E402
from d_witness_core import leaves as _witness_leaves  # noqa: E402
from d_witness_core import occurs as _witness_occurs  # noqa: E402
from t33_labels import load_jsonl  # noqa: E402
from t33_spanmap import parse_tool_call, spans_from_generation  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered constants (from the digest / the cards; never invented here)
# ---------------------------------------------------------------------------

#: JSD of two distributions is bounded by ln 2 nats (2608.26004 section 9: D_i =
#: I(X_i; Z_i) <= H(Z_i) = ln 2 for the binary context-source indicator).
JSD_MAX_NATS = math.log(2.0)

#: NPU bf16 shape-dependent rounding, max|d| recorded by SPEC 5.5.4.  The VeriCache
#: label is exact token equality, which is the construct most easily broken by it;
#: --repeat measures how often it actually breaks.
BF16_SHAPE_ROUNDING_MAXABS = 0.0078125

#: Cost kill line for section C: the paired-config spread costs 2x generation, and
#: the F line already priced that bargain (174 -> 348 rollouts moved success/GPU-s
#: from 0.0320 to 0.0162).  The spread must clear a 2x efficiency bar, not only an
#: AUROC bar (digest 4.8 / card 2607.21475 E1).
SPREAD_COST_MULTIPLIER = 2.0
F_LINE_SUCCESS_PER_GPUS = (0.0320, 0.0162)

#: The forbidden config pair for section C: 512/12 vs 768/16 doc packing pushes the
#: compressed arm out of its training regime, so the spread would measure regime
#: mismatch instead of compression error (digest 4.8; proxy.py's own comment).
FORBIDDEN_PACKING_PAIR = ({"max_doc_length": 512, "max_doc_num": 12},
                          {"max_doc_length": 768, "max_doc_num": 16})

PROXY_STAMP = "text-stand-in for the gist view"

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "VeriCache label pass",
        "paper": "2605.17613 section 5.1",
        "what": "accept_len is reported as the 0-based first_div_idx (count of accepted "
                "tokens), not as the paper's 1-based j-1.",
        "why": "Identical quantity; our token indices are 0-based everywhere "
               "(t33_spanmap spans, t34_common.first_divergence_index).",
    },
    {
        "method": "VeriCache label pass",
        "paper": "2605.17613 section 5.1",
        "what": "No drafting loop and no acceptance/correction: we score the ALREADY "
                "emitted compressed-arm sequence in one teacher-forced pass and stop at "
                "the first divergence.",
        "why": "The frozen battery is single-step teacher-forced; re-drafting would "
               "change the emission and void the frozen 900-row set.  The accept rule "
               "reduces to first_div_idx, which is all the label needs.",
    },
    {
        "method": "VeriCache label pass",
        "paper": "2605.17613 section 5.1",
        "what": "div_region takes two values beyond the digest's {tool_name, arguments, "
                "closing, none}: 'preamble' (divergence before the name span) and "
                "'unparsed' (no tool call span at all).",
        "why": "The emission begins with free text ('...\\nAction:\\n<tool_call>'), so a "
               "divergence can land before any span; collapsing it into tool_name would "
               "be a prefix heuristic, which the digest forbids.  41/93 C->W rows are "
               "cap-censored and some do not parse at all.",
    },
    {
        "method": "VeriCache label pass",
        "paper": "2605.17613 section 5.1",
        "what": "first_div_idx is None (not the paper's implicit infinity) when no "
                "divergence exists, and cap_hit is emitted alongside, never merged.",
        "why": "Card pitfall 4: infinity conflates 'genuinely identical' with 'identical "
               "so far, cap hit at 128'.",
    },
    {
        "method": "VeriCache label pass",
        "paper": "2605.17613 card, pitfall 1 (downgrade bucket)",
        "what": "div_bucket carries a fifth value 'other' beyond the card's "
                "{no_divergence, name_span, arguments, after_cap}, and the downgrade flag "
                "fires at ANY non-zero index-disagreement rate rather than at a named "
                "'non-trivial' threshold.",
        "why": "'other' is where the closing / preamble / unparsed regions land; folding "
               "them into an existing bucket would be the prefix heuristic again.  The "
               "card says 'non-trivial' without a number, and the conservative direction "
               "for a LABEL is to downgrade, so no number is invented in the permissive "
               "direction.",
    },
    {
        "method": "AsymSpec text views",
        "paper": "2608.26004 card, substitution (A) ('truncated to its logged "
                 "kept_history_tokens share')",
        "what": "Each kept doc is HEAD-truncated to kept_history_tokens / n_kept tokens "
                "(the equal share the unit brief prescribes), not to its own logged "
                "doc_lengths[i].",
        "why": "The card's phrase is ambiguous and the consequence must not be read away: "
               "kept_history_tokens is the SUM of the kept rows' token lengths, so "
               "truncating each to the mean chops any above-average block the compressor "
               "actually kept in full, adding a divergence component the tail window did "
               "not create.  The per-doc alternative (doc_lengths, in the U2 fixed schema) "
               "would make the truncation a no-op and leave dropped_docs as the only axis. "
               "per_doc_token_budget is emitted per row so the choice is auditable.",
    },
    {
        "method": "AsymSpec context divergence",
        "paper": "2608.26004 eq:cda",
        "what": "b_i is computed from a TEXT stand-in x_comp_tilde (dropped_docs removed, "
                "each kept doc truncated to kept_history_tokens/n_kept), not from the gist "
                "view; and the drafter scores an already emitted span rather than drafting.",
        "why": "Our compressed view is learned gist KV inside the 4B's own space; a 1.7B "
               "drafter cannot consume it.  The card's substitution (A) is the only "
               "full-free one; substitution (B) (target model on both contexts) is a "
               "verbatim re-run of the killed S2.  Teacher forcing keeps the pass "
               "deterministic.",
    },
    {
        "method": "AsymSpec context divergence",
        "paper": "2608.26004 eq:accept / eq:fusion",
        "what": "The acceptance gate, the delta-fusion emission, gamma=0.5, beta=1.0 and "
                "K=2 are NOT implemented.",
        "why": "We detect, we do not steer.  Porting gamma_eff as a decoding gate would be "
               "an adaptive-compression reading that the project bans; only D_i and delta_i "
               "are migrated, as features.",
    },
    {
        "method": "paired-config spread",
        "paper": "2607.21475 section 6.4",
        "what": "The certificate r_t, the Poisson design and the Hajek correction are NOT "
                "implemented; only the certificate-free two-draw attributor generalised "
                "from two Poisson draws to two deterministic configurations.",
        "why": "The certificate requires replacing the trained gist compressor with a "
               "randomized design (out of scope) and a per-head reduction the NPU fused "
               "kernel does not return.  Two different deterministic F_t suffice for the "
               "impossibility proof's escape.",
    },
    {
        "method": "paired-config spread",
        "paper": "2607.21475 section 6.4 (action_disagree = 1[action_canonical differs])",
        "what": "action_canonical is rebuilt LOCALLY (t33_spanmap.parse_tool_call -> name + "
                "key-sorted arguments) instead of imported from benchmarks/proxy.py, and an "
                "emission that does not parse falls back to the whitespace-normalised text "
                "so the column stays total.  The raw text comparison is kept beside it as "
                "spread_text_disagree.",
        "why": "proxy.action_canonical is a bench-face object this unit may not import "
               "into its own logic, and the benchmarks/proxy.py checked out in this "
               "worktree defines no such symbol at all, so the two definitions cannot be "
               "diffed here; bench_face_action_canonical resolves it where a worktree HAS "
               "it and the unit test compares the two partitions then.  Comparing raw text "
               "would count JSON key order and payload whitespace as an action change, "
               "which is not the estimand the card names.",
    },
    {
        "method": "AsymSpec text views",
        "paper": "2608.26004 card, substitution (A)",
        "what": "x_full is assembled from the KEPT grid rows plus the sidecar's optional "
                "dropped_doc_texts, reconstructed into post-split order.  When the dropped "
                "plaintext is absent the row is still SCORED, with x_full = the visible "
                "blocks at full length and x_comp = the same blocks under the declared "
                "kept_history_tokens truncation, stamped "
                "proxy_variant='truncation_only_missing_dropped_text'.",
        "why": "The U2 sidecar's ``docs`` holds the kept rows only while ``dropped_docs`` "
               "indexes the post-split list (t34_dump_sidecar DEVIATIONS).  Filtering docs "
               "by dropped_docs deletes KEPT blocks and never removes a dropped one, so the "
               "divergence would measure an arbitrary deletion instead of the tail window.  "
               "Skipping the incomplete rows instead would delete every row that HAS a drop "
               "under the default dump -- the whole informative subset -- so the degraded "
               "rows carry a narrower, declared estimand (truncation only) and the three "
               "proxy_variants are stratified, never pooled.",
    },
    {
        "method": "paired-config spread -- mandatory baseline (3) and retained entropy",
        "paper": "2607.21475 card, 'Mandatory baselines' item 3; results table row "
                 "'retained entropy' (0.43-0.51, at chance)",
        "what": "The evicted-score-mass control is the sidecar's dropped-block COUNT "
                "fraction plus a CHARACTER-mass fraction (the latter only when "
                "dropped_doc_texts is present); the retained-entropy control is the Shannon "
                "entropy of the KEPT blocks' token-length distribution, not of an attention "
                "distribution.",
        "why": "The U2 fixed schema logs doc_lengths for the kept rows only and no lengths "
               "at all for the dropped ones, and the retained-ATTENTION entropy the paper "
               "measures is kernel-blocked on serving (S6).  These are the deterministic, "
               "already-logged analogues; both are PREDICTED-weak controls, which is what "
               "the card asks them to be, and each is None -- never 0 -- when undefined.",
    },
    {
        "method": "paired-config spread",
        "paper": "2607.21475 section 6.4",
        "what": "logit_spread is computed from the capture files' top-5 log-probabilities "
                "renormalised over the union support, not over the full vocabulary.",
        "why": "The frozen capture stores top-5 only.  The JSD of two proper (renormalised) "
               "distributions is still bounded by ln 2, but tokens outside a run's top-5 are "
               "assigned zero mass, which OVER-estimates divergence; flagged per row.",
    },
    {
        "method": "SelfCheckGPT N=1",
        "paper": "2303.08896 section 7.3.1",
        "what": "None of the five sampling variants is implemented; N=20 stochastic samples "
                "are replaced by ONE reference (decoded compressed docs + query) and only "
                "the deterministic lexical and NLI scores are computed.",
        "why": "Sampling breaks the determinism gate and costs 44-222 GPU-s against a "
               "3.69-5.57 s recovery action.  section 7.3.1 reports NLI/Prompt IMPROVE under "
               "the single-reference substitution.",
    },
    {
        "method": "SelfCheckGPT N=1",
        "paper": "2303.08896 section 5 (unigram variant) / section 7.3.1",
        "what": "Lexical grounding matches case-folded, whereas d_witness_core.occurs (the "
                "frozen witness semantics) matches case-sensitively.",
        "why": "The digest specifies case-folded verbatim occurrence for this feature; the "
               "frozen witness table is untouched (a separate, gold-scored object).",
    },
    {
        "method": "SelfCheckGPT N=1",
        "paper": "2303.08896 section 5 (NLI variant)",
        "what": "Sentence decomposition is dropped: the hypothesis is one templated "
                "verbalisation of the whole emitted call.",
        "why": "Our unit is a single <tool_call>, not a paragraph; 'average over sentences' "
                "has no analogue.  Their AUC-PR (73 % prevalence) is therefore not "
                "comparable and is never printed beside ours.",
    },
]


# ===========================================================================
# (A) VeriCache -- LABEL pass (2605.17613 section 5.1)
# ===========================================================================

#: The digest's canonical region codomain plus the two honest extras (see DEVIATIONS).
DIV_REGIONS = ("none", "tool_name", "arguments", "closing", "preamble", "unparsed")


def _in_span(idx: int, first: Optional[int], last: Optional[int]) -> bool:
    return first is not None and last is not None and first <= idx <= last


def vericache_label_div_region(first_div_idx: Optional[int],
                               spans: Dict[str, Any]) -> str:
    """Region of the first divergence (2605.17613 section 5.1 accept rule, mapped
    through the real span map rather than a prefix heuristic).

    ``spans`` is a ``t33_spanmap.spans_from_generation`` record of the COMPRESSED
    arm's own emission.  Returns one of :data:`DIV_REGIONS`.
    """
    if first_div_idx is None:
        return "none"
    if not spans.get("has_tool_call"):
        return "unparsed"
    if _in_span(first_div_idx, spans.get("name_first"), spans.get("name_last")):
        return "tool_name"
    if _in_span(first_div_idx, spans.get("args_first"), spans.get("args_last")):
        return "arguments"
    ends = [spans.get(k) for k in ("args_last", "name_last", "payload_last")]
    ends = [e for e in ends if e is not None]
    if ends and first_div_idx > max(ends):
        return "closing"
    starts = [spans.get(k) for k in ("name_first", "args_first", "payload_first")]
    starts = [s for s in starts if s is not None]
    if starts and first_div_idx < min(starts):
        return "preamble"
    return "unparsed"


def vericache_label_bucket(row: Dict[str, Any]) -> str:
    """Coarse label the card prescribes as the downgrade when the index is not
    self-consistent (2605.17613 card, pitfall 1):
    {no_divergence, name_span, arguments, after_cap}."""
    if row.get("first_div_idx") is None:
        return "after_cap" if row.get("cap_hit") else "no_divergence"
    region = row.get("div_region")
    if region == "tool_name":
        return "name_span"
    if region == "arguments":
        return "arguments"
    return "other"


def vericache_label_row(
    qid: str,
    emitted_ids: Sequence[int],
    verifier_argmax_ids: Sequence[int],
    spans: Dict[str, Any],
    *,
    cap_tokens: int,
    ids_source: str = "capture",
) -> Dict[str, Any]:
    """One VeriCache verify row (2605.17613 section 5.1).

    ``verifier_argmax_ids`` are t*_1..t*_{x+1}: the argmax of ONE teacher-forced
    parallel forward under the FULL prefix at the same positions.  Hard token
    equality, no threshold, no calibration set.

    Returns the label columns ``first_div_idx`` (0-based, None = no divergence
    within the compared prefix), ``accept_len`` (= first_div_idx, or the compared
    length when there is no divergence), ``div_region``, ``div_bucket``,
    ``cap_hit`` and the bonus token t*_{x+1} when the pass supplied it.
    """
    emitted = [int(t) for t in emitted_ids]
    star = [int(t) for t in verifier_argmax_ids]
    n_cmp = min(len(emitted), len(star))
    j = C.first_divergence_index(emitted[:n_cmp], star[:n_cmp])
    row = {
        "qid": qid,
        "session_id": C.session_of(qid),
        "n_emitted": len(emitted),
        "n_verified": len(star),
        "n_compared": n_cmp,
        "first_div_idx": j,
        "accept_len": j if j is not None else n_cmp,
        "cap_hit": bool(len(emitted) >= int(cap_tokens)),
        "bonus_token_id": star[len(emitted)] if len(star) > len(emitted) else None,
        "ids_source": ids_source,
        "label_family": "vericache_verify",
    }
    row["div_region"] = vericache_label_div_region(j, spans)
    row["div_bucket"] = vericache_label_bucket(row)
    return row


def vericache_label_disagreement(rows_a: Sequence[Dict[str, Any]],
                                 rows_b: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Self-consistency of the label across two identical passes (2605.17613 card,
    pitfall 1; SPEC 5.5.4 records NPU bf16 shape-dependent rounding max|d| =
    0.0078125 and 'bit-exact sentinels unattainable').

    Reports the disagreement rate of ``first_div_idx`` AND of the coarse bucket.
    A non-trivial index rate is the documented trigger to downgrade the label from
    an index to a bucket; the caller must act on it, this function only measures.
    """
    a = {r["qid"]: r for r in rows_a}
    b = {r["qid"]: r for r in rows_b}
    common = sorted(set(a) & set(b))
    idx_bad = [q for q in common if a[q].get("first_div_idx") != b[q].get("first_div_idx")]
    buck_bad = [q for q in common if a[q].get("div_bucket") != b[q].get("div_bucket")]
    n = len(common)
    return {
        "n_common": n,
        "n_only_a": len(set(a) - set(b)),
        "n_only_b": len(set(b) - set(a)),
        "index_disagree": len(idx_bad),
        "index_disagree_rate": (len(idx_bad) / n) if n else None,
        "bucket_disagree": len(buck_bad),
        "bucket_disagree_rate": (len(buck_bad) / n) if n else None,
        "bf16_shape_rounding_maxabs": BF16_SHAPE_ROUNDING_MAXABS,
        "downgrade_to_bucket": bool(n and len(idx_bad) / n > 0.0),
        "disagreeing_qids": idx_bad[:20],
    }


def vericache_label_census(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Distribution of the label over a row set (2605.17613 card deliverable 1:
    the mechanical read-out of how much of the trigger set is a 128-cap truncation
    flip rather than a semantic divergence)."""
    out: Dict[str, Any] = {"n": len(rows)}
    for key in ("div_region", "div_bucket"):
        counts: Dict[str, int] = {}
        for r in rows:
            counts[str(r.get(key))] = counts.get(str(r.get(key)), 0) + 1
        out[key] = dict(sorted(counts.items()))
    acc = [r["accept_len"] for r in rows if r.get("accept_len") is not None]
    out["accept_len_mean"] = float(np.mean(acc)) if acc else None
    out["accept_len_median"] = float(np.median(acc)) if acc else None
    out["n_no_divergence"] = sum(1 for r in rows if r.get("first_div_idx") is None)
    out["n_no_divergence_and_cap_hit"] = sum(
        1 for r in rows if r.get("first_div_idx") is None and r.get("cap_hit"))
    # The two id sources are not comparable: a re-tokenised sequence can be shifted
    # against the emitted one, which shifts first_div_idx.  Make the mix visible here
    # instead of leaving "never pooled" as a promise in a docstring.
    src: Dict[str, int] = {}
    for r in rows:
        key = str(r.get("ids_source"))
        src[key] = src.get(key, 0) + 1
    out["by_ids_source"] = dict(sorted(src.items()))
    out["mixed_ids_sources"] = len(src) > 1
    out["n_retokenized_lossy"] = src.get("retokenized_lossy", 0)
    return out


def load_capture_steps(path: Path) -> Dict[str, Dict[str, Any]]:
    """Index a t33 capture ``<arm>/<part>.steps.jsonl`` by qid."""
    return {r["qid"]: r for r in load_jsonl(str(path)) if "qid" in r}


def emitted_ids_for_row(qid: str,
                        capture_index: Dict[str, Dict[str, Any]],
                        prediction_text: str,
                        encode_fn: Optional[Callable[[str], List[int]]],
                        decode_fn: Optional[Callable[[Sequence[int]], str]] = None
                        ) -> Tuple[List[int], str]:
    """Compressed-arm emitted token ids, capture first.

    Falls back to re-tokenising ``prediction_text``.  decode->encode is NOT guaranteed
    to reproduce the original ids and a shifted id sequence shifts ``first_div_idx``,
    so the fallback is stamped and, when ``decode_fn`` is supplied, actually CHECKED:

      ``capture``              the ids the model emitted, from the t33 capture
      ``retokenized_checked``  re-encoded and the roundtrip reproduced the text
      ``retokenized_lossy``    re-encoded and the roundtrip did NOT reproduce it:
                               this row's index is suspect and must not be pooled
      ``retokenized``          re-encoded with no tokenizer to check the roundtrip
      ``unavailable``          no ids at all

    ``vericache_label_census`` reports the mix and flags it, so the sources are never
    pooled silently.
    """
    rec = capture_index.get(qid)
    if rec and rec.get("generated_ids"):
        return [int(t) for t in rec["generated_ids"]], "capture"
    if rec and rec.get("steps"):
        return [int(s["token_id"]) for s in rec["steps"]], "capture"
    if encode_fn is None:
        return [], "unavailable"
    text = prediction_text or ""
    ids = list(encode_fn(text))
    if decode_fn is None:
        return ids, "retokenized"
    try:
        ok = decode_fn(ids) == text
    except Exception:  # noqa: BLE001 -- an unusable decoder is "unchecked", not "ok"
        return ids, "retokenized"
    return ids, ("retokenized_checked" if ok else "retokenized_lossy")


def _forward_argmax(model: Any, prefix: Dict[str, Any], prompt_ids: Sequence[int],
                    emitted_ids: Sequence[int], attn_impl: str) -> Optional[List[int]]:
    """ONE teacher-forced parallel forward under ``prefix``; returns the argmax id at
    each of the ``len(emitted_ids)+1`` scored positions (2605.17613 section 5.1 verify
    step: t*_1..t*_{x+1}).

    Router convention copied verbatim from ``t33_svip_gamma._forward_stats`` (input_ids
    = new tokens only, attention_mask spans cache+new, position_ids at the logical
    offsets).  A local copy is used because ``_forward_stats`` has no ``want='argmax'``
    branch and t33_* files are frozen.
    """
    import torch  # lazy: the NPU server has torch, this box does not

    device = model.device
    cache_length = prefix["cache"].get_seq_length()
    real = list(prompt_ids) + list(emitted_ids)
    real_t = torch.tensor([real], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, cache_length + len(real)), dtype=torch.long, device=device)
    original_prefix_length = prefix["system_length"] + prefix["history_length"]
    position_ids = torch.arange(
        original_prefix_length, original_prefix_length + len(real),
        dtype=torch.long, device=device).unsqueeze(0)
    original_attn = model.model.config._attn_implementation
    model.model.config._attn_implementation = attn_impl
    try:
        with torch.inference_mode():
            out = model(
                input_ids=real_t,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=prefix["cache"],
                use_gist=bool(prefix.get("use_gist", False)),
                use_cache=False,
                logits_to_keep=len(emitted_ids) + 1,
            )
        return [int(v) for v in out.logits[0].to(torch.float32).argmax(dim=-1).tolist()]
    except Exception:  # noqa: BLE001  -- caller records the row as skipped
        return None
    finally:
        model.model.config._attn_implementation = original_attn


# ===========================================================================
# (B) AsymSpec -- context divergence on a TEXT proxy (2608.26004)
# ===========================================================================

def softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax (2608.26004 eq:cda operates on softmax(a_i), softmax(b_i))."""
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    mask = p > 0
    logp = np.where(mask, np.log(np.where(mask, p, 1.0)), 0.0)
    logq = np.where(mask, np.log(np.where(q > 0, q, 1.0)), 0.0)
    return np.sum(np.where(mask, p * (logp - logq), 0.0), axis=-1)


def jensen_shannon_nats(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """JSD(p || q) in NATS, row-wise (2608.26004 eq:cda).

    Bounded by ln 2 by construction; section 9 reads it as I(X_i; Z_i) <= H(Z_i) = ln 2
    for the binary latent selecting x_full vs x_comp.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = 0.5 * (p + q)
    return 0.5 * _kl_rows(p, m) + 0.5 * _kl_rows(q, m)


def asymspec_position_stats(logits_a: np.ndarray, logits_b: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-position D_i and delta_i (2608.26004 eq:delta, eq:cda).

    ``logits_a`` = S(x_full + y), ``logits_b`` = S(x_comp + y): (n_positions, |V|).
    Returns ``jsd`` (nats, <= ln 2), ``delta_l1`` = ||a_i - b_i||_1, and
    ``entropy_b`` = H(softmax b_i) in nats.

    ``entropy_b`` is the card's MANDATORY S0 control (b): the drafter's own
    single-ended entropy on x_comp_tilde, i.e. the "generic sample difficulty"
    confound that killed S1 at 0.5313.  It is emitted as its own feature column so
    D_i can be reported as a delta against it, never instead of it.
    """
    a = np.asarray(logits_a, dtype=np.float64)
    b = np.asarray(logits_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"logit shapes differ: {a.shape} vs {b.shape}")
    pb = softmax_rows(b)
    jsd = jensen_shannon_nats(softmax_rows(a), pb)
    ent = -np.sum(np.where(pb > 0, pb * np.log(np.where(pb > 0, pb, 1.0)), 0.0), axis=-1)
    return {"jsd": jsd, "delta_l1": np.abs(a - b).sum(axis=-1), "entropy_b": ent}


def asymspec_aggregate(stats: Dict[str, np.ndarray],
                       name_positions: Optional[Sequence[int]] = None,
                       *, prefix: str = "asym") -> Dict[str, Optional[float]]:
    """Step-level aggregation of the per-position divergence (2608.26004 card,
    migration recipe E1 step 5): max_i D_i, mean_i D_i, D on the tool-name token
    positions, ||delta||_1 (mean, max, and on the name positions), D_1, and the S0
    control H(softmax b_i) (mean/max).

    ``name_positions`` are indices into the scored span (from the span map).  When
    the row has no name span the name features are ``None`` -- never a whole-span
    fallback under a span-specific name, and never a sentinel.
    """
    jsd = np.asarray(stats["jsd"], dtype=float)
    dl1 = np.asarray(stats["delta_l1"], dtype=float)
    ent = np.asarray(stats.get("entropy_b", np.empty(0)), dtype=float)
    out: Dict[str, Optional[float]] = {
        f"{prefix}_jsd_max": float(jsd.max()) if jsd.size else None,
        f"{prefix}_jsd_mean": float(jsd.mean()) if jsd.size else None,
        f"{prefix}_jsd_first": float(jsd[0]) if jsd.size else None,
        f"{prefix}_delta_l1_mean": float(dl1.mean()) if dl1.size else None,
        f"{prefix}_delta_l1_max": float(dl1.max()) if dl1.size else None,
        f"{prefix}_n_positions": int(jsd.size),
    }
    # S0 control (b): single-ended drafter entropy on x_comp_tilde.
    out[f"{prefix}_entropy_comp_mean"] = float(ent.mean()) if ent.size else None
    out[f"{prefix}_entropy_comp_max"] = float(ent.max()) if ent.size else None
    pos = [int(i) for i in (name_positions or []) if 0 <= int(i) < jsd.size]
    out[f"{prefix}_jsd_name_max"] = float(jsd[pos].max()) if pos else None
    out[f"{prefix}_jsd_name_mean"] = float(jsd[pos].mean()) if pos else None
    out[f"{prefix}_delta_l1_name_max"] = float(dl1[pos].max()) if pos else None
    out[f"{prefix}_n_name_positions"] = len(pos)
    return out


def _ws_encode(text: str) -> List[str]:
    return text.split()


def _ws_decode(toks: Sequence[str]) -> str:
    return " ".join(toks)


def truncate_to_tokens(text: str, budget: int,
                       encode_fn: Optional[Callable[[str], Sequence[Any]]] = None,
                       decode_fn: Optional[Callable[[Sequence[Any]], str]] = None) -> str:
    """Head-truncate ``text`` to ``budget`` tokens.

    With no tokenizer the whitespace fallback is used; the caller records
    ``truncation_mode`` so a whitespace-truncated artifact is never read as a
    token-budget one.  (2608.26004 card: x_comp_tilde truncates each kept doc to its
    logged ``kept_history_tokens`` share.)
    """
    if budget is None or budget <= 0:
        return ""
    enc = encode_fn or _ws_encode
    dec = decode_fn or _ws_decode
    toks = list(enc(text or ""))
    if len(toks) <= budget:
        return text or ""
    return dec(toks[:budget])


def split_history_blocks(sidecar_row: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct the POST-SPLIT history ordering from a U2 sidecar row.

    THE TWO INDEX BASES ARE DIFFERENT and conflating them silently corrupts every
    view built here:

    * ``docs`` holds ONLY the KEPT grid rows.  ``t34_dump_sidecar`` builds them from
      ``eval_agent_history_c2kv._build_history_chunks`` -> ``_history_messages`` ->
      ``train_data_multiturn._fit_reused_history``, and ``_fit_reused_history``
      SELECTS (``_select_history_with_indices``) before returning.
    * ``dropped_docs`` indexes the POST-SPLIT list the tail window selected over --
      stated verbatim in ``t34_dump_sidecar.DEVIATIONS`` ('dropped_docs index base')
      and produced by ``kept_and_dropped_indices(len(split), ...)``.

    So ``dropped_docs`` are NOT positions in ``docs``; ``[d for i, d in
    enumerate(docs) if i not in dropped]`` deletes KEPT blocks and never removes any
    dropped one (whose text is not in ``docs`` at all).

    Returns the reconstructed ordering plus ``dropped_text_available``: the dropped
    blocks' plaintext exists only when the sidecar was dumped with
    ``--with_dropped_text`` (key ``dropped_doc_texts``, aligned with
    ``sorted(dropped_docs)``).  Without it the FULL post-split text cannot be
    reconstructed -- ``blocks`` carries ``None`` at every dropped position -- and the
    caller must narrow its estimand accordingly (see :func:`build_text_views`, which
    falls back to the visible blocks and stamps ``proxy_variant``), never silently
    treat the kept blocks as the whole history.
    """
    docs = [str(d) for d in (sidecar_row.get("docs") or [])]
    dropped = sorted({int(i) for i in (sidecar_row.get("dropped_docs") or [])})
    n_split = len(docs) + len(dropped)
    if dropped and (dropped[0] < 0 or dropped[-1] >= n_split):
        raise ValueError(
            f"dropped_docs {dropped} out of range for a post-split list of "
            f"{n_split} blocks (len(docs)={len(docs)})")
    kept_positions = [i for i in range(n_split) if i not in set(dropped)]
    if len(kept_positions) != len(docs):  # pragma: no cover - arithmetic invariant
        raise ValueError("kept-position count does not match len(docs)")
    texts = list(sidecar_row.get("dropped_doc_texts") or [])
    have_dropped_text = len(texts) == len(dropped)
    blocks: List[Optional[str]] = [None] * n_split
    for rank, pos in enumerate(kept_positions):
        blocks[pos] = docs[rank]
    for rank, pos in enumerate(dropped):
        blocks[pos] = texts[rank] if have_dropped_text else None
    return {
        "n_split_blocks": n_split,
        "kept_positions": kept_positions,
        "dropped_positions": dropped,
        "blocks": blocks,
        "kept_texts": docs,
        "dropped_text_available": bool(have_dropped_text or not dropped),
    }


def build_text_views(sidecar_row: Dict[str, Any],
                     *,
                     encode_fn: Optional[Callable[[str], Sequence[Any]]] = None,
                     decode_fn: Optional[Callable[[Sequence[Any]], str]] = None,
                     include_system: bool = True,
                     sep: str = "\n\n") -> Dict[str, Any]:
    """x_full and x_comp_tilde from the frozen sidecar (2608.26004 card, substitution
    (A); S8/S13 fields only).

    x_full          = system_prompt + EVERY post-split history block, in order
                      (kept blocks from ``docs``, dropped blocks from
                      ``dropped_doc_texts``) + query
    x_comp_tilde    = system_prompt + the kept blocks only, each truncated to
                      ``kept_history_tokens / n_kept`` tokens, + query

    The system prompt and the query are identical in both views by construction, so
    the divergence is attributable to the history only.  Every returned record carries
    ``proxy`` = PROXY_STAMP: this is a text stand-in, not the gist view we ship.

    ``view_complete`` is False when the row has dropped blocks whose plaintext the
    sidecar does not carry (``--with_dropped_text`` was not passed).  Such a row is NOT
    skipped -- skipping would delete every row that has a drop, i.e. the whole
    informative subset, whenever the default dump is used.  It falls back to the
    sidecar-declared ``kept_history_tokens`` truncation alone:

        x_full = system + the KEPT blocks at full length + query
        x_comp = system + the same blocks truncated to the kept share + query

    so the divergence still measures a real, declared axis (truncation) but NOT the
    tail-window drop.  ``proxy_variant`` says which axes a row's D_i contains and the
    three variants must be stratified, never pooled:

      ``drop_and_truncation``                 dropped plaintext present: both axes
      ``truncation_only_no_drop``             the row has no dropped block at all
      ``truncation_only_missing_dropped_text``  degraded fallback; re-dump the sidecar
                                              with --with_dropped_text to recover the
                                              drop axis on these rows
    """
    layout = split_history_blocks(sidecar_row)
    docs = layout["kept_texts"]
    query = sidecar_row.get("query") or ""
    system = (sidecar_row.get("system_prompt") or "") if include_system else ""
    kept_tokens = int(sidecar_row.get("kept_history_tokens") or 0)
    per_doc = (kept_tokens // len(docs)) if docs and kept_tokens else 0
    comp_docs = [truncate_to_tokens(d, per_doc, encode_fn, decode_fn) for d in docs]
    if layout["dropped_text_available"]:
        full_blocks = [b for b in layout["blocks"] if b is not None]
        variant = ("drop_and_truncation" if layout["dropped_positions"]
                   else "truncation_only_no_drop")
    else:
        # x_full restricted to the VISIBLE blocks (every entry of docs), untruncated.
        full_blocks = list(docs)
        variant = "truncation_only_missing_dropped_text"

    def _join(parts: Sequence[str]) -> str:
        return sep.join([p for p in parts if p])

    return {
        "qid": sidecar_row.get("qid"),
        "x_full": _join([system] + full_blocks + [query]),
        "x_comp": _join([system] + comp_docs + [query]),
        "n_docs": layout["n_split_blocks"],
        "n_kept_docs": len(docs),
        "n_dropped_docs": len(layout["dropped_positions"]),
        "per_doc_token_budget": per_doc,
        "truncation_mode": "tokenizer" if encode_fn is not None else "whitespace",
        "view_complete": bool(layout["dropped_text_available"]),
        "proxy_variant": variant,
        "proxy": PROXY_STAMP,
    }


def assert_drafter_family(drafter_tokenizer: Any, target_tokenizer: Any,
                          drafter_name: str, target_name: str) -> None:
    """Refuse a drafter that is our own 4B (that is the killed S2 verbatim) or one
    from a different tokenizer family (2608.26004 section 6.4: cross-family runs need a
    109,566-token restriction; card pitfall 5 says use Qwen3-1.7B against Qwen3-4B)."""
    if drafter_name.strip() == target_name.strip():
        raise ValueError("drafter must not be the target model: S = L is the killed S2")
    v_d = int(getattr(drafter_tokenizer, "vocab_size", -1))
    v_t = int(getattr(target_tokenizer, "vocab_size", -2))
    if v_d != v_t:
        raise ValueError(f"tokenizer family mismatch: drafter vocab {v_d} != target {v_t}")
    probe = 'Action:\n<tool_call>\n{"name":"a__b","arguments":{"x":1}}\n</tool_call>'
    if list(drafter_tokenizer.encode(probe)) != list(target_tokenizer.encode(probe)):
        raise ValueError("tokenizer family mismatch: probe encoding differs")


def _score_span_logits(model: Any, tokenizer: Any, context_text: str,
                       span_text: str, max_ctx_tokens: int) -> Optional[np.ndarray]:
    """Teacher-forced logits of ``span_text`` under ``context_text`` (float32 numpy,
    (n_span, |V|)).  Deterministic scoring pass; the drafter never generates
    (2608.26004 card pitfall 7)."""
    import torch

    ctx = tokenizer.encode(context_text, add_special_tokens=False)
    span = tokenizer.encode(span_text, add_special_tokens=False)
    if not span or not ctx:
        return None
    if max_ctx_tokens and len(ctx) > max_ctx_tokens:
        ctx = ctx[-max_ctx_tokens:]
    ids = torch.tensor([ctx + span], dtype=torch.long, device=model.device)
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=False)
    logits = out.logits[0, len(ctx) - 1: len(ctx) - 1 + len(span), :]
    return logits.to(torch.float32).cpu().numpy()


# ===========================================================================
# (C) KV-Eviction certificates -> paired-config spread (2607.21475)
# ===========================================================================

def config_pair_for_qid(qid: str, n_docs: int, mode: str,
                        hybrid_top_k_pair: Tuple[int, int] = (2, 3)) -> Dict[str, Any]:
    """Two DETERMINISTIC but different compression configurations for one row
    (2607.21475 card, E1: 'you cannot identify the compression error from one
    deterministic compression, but you can from two').

    ``mode='doc_permute'``: config A keeps the natural doc order, config B applies a
    permutation seeded by ``sha256(qid)`` -- reproducible, and the same request under
    the same config always returns the same content, so the ``repair_fidelity``
    determinism gate is untouched.  This is NOT resampling.

    ``mode='hybrid_top_k'``: two hybrid_top_k values.  The doc-packing pair
    512/12 vs 768/16 is refused (see FORBIDDEN_PACKING_PAIR).
    """
    if mode == "doc_permute":
        seed = int(hashlib.sha256(qid.encode("utf-8")).hexdigest()[:16], 16) % (2 ** 32)
        perm = list(np.random.default_rng(seed).permutation(int(n_docs)).tolist())
        return {"qid": qid, "mode": mode,
                "config_a": {"doc_order": list(range(int(n_docs)))},
                "config_b": {"doc_order": [int(i) for i in perm]},
                "seed": seed, "resampling": False}
    if mode == "hybrid_top_k":
        ka, kb = int(hybrid_top_k_pair[0]), int(hybrid_top_k_pair[1])
        if ka == kb:
            raise ValueError("hybrid_top_k pair must differ")
        return {"qid": qid, "mode": mode,
                "config_a": {"hybrid_top_k": ka},
                "config_b": {"hybrid_top_k": kb},
                "resampling": False}
    raise ValueError(f"unknown paired-config mode: {mode}")


def assert_config_pair_legal(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any]) -> None:
    """Refuse the 512/12 vs 768/16 doc-packing pair: it pushes the compressed arm out
    of its training regime, so the spread would measure regime mismatch instead of
    compression error (digest 4.8)."""
    def _pack(c: Dict[str, Any]) -> Dict[str, Any]:
        return {k: c[k] for k in ("max_doc_length", "max_doc_num") if k in c}
    pa, pb = _pack(cfg_a), _pack(cfg_b)
    if pa and pb and pa != pb:
        raise ValueError(
            "illegal paired-config axis: doc packing differs "
            f"({pa} vs {pb}); the frozen recipe is 768/16 and 512/12 is out of "
            "training regime (digest 4.8)")


def top5_to_dense(top5: Sequence[Sequence[float]]) -> Dict[int, float]:
    """A capture step's ``top5`` -> {token_id: probability}, renormalised.

    The capture writes pairs ``[logprob, token_id]`` (t33_capture.py:234).  Mass
    outside the top 5 is dropped, so the resulting JSD OVER-estimates divergence;
    every row that uses it is flagged ``logit_spread_truncated=True``.
    """
    d: Dict[int, float] = {}
    for pair in top5 or []:
        lp, tid = float(pair[0]), int(pair[1])
        d[tid] = d.get(tid, 0.0) + math.exp(lp)
    z = sum(d.values())
    return {k: v / z for k, v in d.items()} if z > 0 else {}


def truncated_jsd(a: Dict[int, float], b: Dict[int, float]) -> Optional[float]:
    """JSD in nats between two truncated, renormalised top-k distributions."""
    if not a or not b:
        return None
    support = sorted(set(a) | set(b))
    pa = np.array([[a.get(t, 0.0) for t in support]], dtype=np.float64)
    pb = np.array([[b.get(t, 0.0) for t in support]], dtype=np.float64)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(jensen_shannon_nats(pa, pb)[0])


def canonical_action_key(text: str) -> Tuple[str, str]:
    """A canonical key for the emitted action (2607.21475 card, E1:
    ``action_disagree = 1[action_canonical differs]``).

    ``benchmarks/proxy.py``'s ``action_canonical`` is a bench-face object this unit
    may not import, so the key is rebuilt locally from the SAME parse the rest of the
    unit uses (``t33_spanmap.parse_tool_call``): ``(name, arguments)`` with the
    argument keys sorted, so JSON key order and whitespace inside the payload do not
    count as an action change.

    An emission that does not parse to a name + arguments object canonicalises to
    ``("unparsed", <whitespace-normalised text>)`` so the column stays TOTAL -- an
    unparseable pair is compared as text rather than silently declared equal.
    """
    parsed = parse_tool_call(text or "")
    name, args = parsed.get("name"), parsed.get("arguments")
    if parsed.get("parse_ok") and isinstance(name, str) and isinstance(args, dict):
        return ("parsed", json.dumps({"name": name, "arguments": args},
                                     sort_keys=True, ensure_ascii=False))
    return ("unparsed", normalize_ws_local(text))


def normalize_ws_local(text: str) -> str:
    return C.normalize_ws(text or "")


def bench_face_action_canonical(root: Optional[Path] = None) -> Optional[Callable[[str], Any]]:
    """The BENCH FACE's own ``action_canonical``, when this worktree has one.

    :func:`canonical_action_key` is a local rebuild (see DEVIATIONS) because
    ``benchmarks/proxy.py`` is a face this unit may not import into its own logic.
    The two definitions must agree before a spread ``action_disagree`` is ever read
    next to a bench-face one -- and in THIS worktree the bench proxy defines neither
    ``action_canonical`` nor ``RecoverState``, so nothing can diff them here.

    Returns the callable when it exists, ``None`` otherwise.  The unit test skips with
    that reason when it is absent and diffs the two partitions when it appears, so the
    check turns itself on the moment the symbol lands instead of drifting silently.
    """
    bench = (Path(root) if root else _HERE.parent) / "benchmarks"
    src = bench / "proxy.py"
    if not src.exists() or "def action_canonical" not in src.read_text(
            encoding="utf-8", errors="replace"):
        return None
    import importlib.util
    added = str(bench) not in sys.path
    if added:
        sys.path.insert(0, str(bench))
    try:
        spec = importlib.util.spec_from_file_location("_t34_bench_proxy", src)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # noqa: S102 -- read-only face import
    except Exception:  # noqa: BLE001 -- an unimportable face is simply "not available"
        return None
    finally:
        if added and str(bench) in sys.path:
            sys.path.remove(str(bench))
    fn = getattr(mod, "action_canonical", None)
    return fn if callable(fn) else None


def _span_nll(record: Dict[str, Any]) -> Optional[float]:
    """Negative log-probability of the emitted <tool_call> span from a capture record.

    Uses the span map's payload range when it exists; returns None when the row has
    no tool call (never a whole-output fallback under a span-specific name)."""
    spans = record.get("spans") or {}
    steps = record.get("steps") or []
    first, last = spans.get("payload_first"), spans.get("payload_last")
    if first is None or last is None:
        return None
    sel = [s for s in steps if first <= int(s.get("step", -1)) <= last]
    if not sel:
        return None
    return float(-sum(float(s["chosen_logprob"]) for s in sel))


def mean_output_logprob(record: Dict[str, Any]) -> Optional[float]:
    """The compressed arm's own MEAN output log-probability -- 2607.21475's mandatory
    baseline (section 6.2: 'a trust signal evaluated without this baseline overstates its
    case'; it beat every cache-side signal at PREDICTION, 0.73-0.80 vs 0.56-0.57, and
    was at chance at ATTRIBUTION, 0.542/0.469/0.519)."""
    steps = record.get("steps") or []
    if not steps:
        return None
    return float(np.mean([float(s["chosen_logprob"]) for s in steps]))


#: Columns of the predicted-weak control family (2607.21475 card, mandatory baseline
#: 3 and the paper's own "retained entropy" row, which sat at chance, 0.43-0.51).  They
#: are DECLARED here so ``score_feature_table`` can say whether the spread table it is
#: about to print contains them.
EVICTED_CONTROL_COLS = ("baseline_evicted_doc_frac", "baseline_evicted_char_mass")
RETAINED_ENTROPY_COLS = ("baseline_retained_len_entropy",)


def evicted_mass_control_row(sidecar_row: Dict[str, Any]) -> Dict[str, Any]:
    """2607.21475's mandatory baseline (3) -- the ``dropped_docs`` / evicted-score-mass
    analogue -- and the paper's retained-entropy control, from the U2 sidecar alone.

    The card calls (3) a PREDICTED-weak control ("the theorem says it should be weak,
    so it is a predicted-weak control, which is worth more than an unpredicted one"),
    and its own table puts retained entropy at chance (0.43-0.51).  Both are emitted
    as ordinary oriented feature columns so they are scored on exactly the same rows,
    with the same bootstrap, as the spread features they are the floor for.

    Columns (see DEVIATIONS for why these are the available analogues):

      ``baseline_evicted_doc_frac``      dropped blocks / post-split blocks.  Always
                                         defined from ``dropped_docs`` + ``docs``.
      ``baseline_evicted_char_mass``     dropped chars / (dropped + kept) chars.
                                         ``None`` when the row HAS drops but the
                                         sidecar was dumped without
                                         ``--with_dropped_text`` -- never 0.0, which
                                         would read as "nothing was evicted".  The
                                         token-mass version needs per-dropped-block
                                         lengths, which the U2 fixed schema does not
                                         carry.
      ``baseline_retained_len_entropy``  Shannon entropy (nats) of the KEPT blocks'
                                         token-length distribution from
                                         ``doc_lengths``; ``None`` when
                                         ``doc_lengths`` is absent or degenerate.
                                         The paper's object is a retained-ATTENTION
                                         entropy, which is kernel-blocked on serving.
    """
    layout = split_history_blocks(sidecar_row)
    n_split = layout["n_split_blocks"]
    dropped = layout["dropped_positions"]
    kept_texts = layout["kept_texts"]
    row: Dict[str, Any] = {
        "qid": sidecar_row.get("qid"),
        "baseline_evicted_doc_frac": (float(len(dropped)) / n_split) if n_split else None,
        "baseline_n_dropped_docs": len(dropped),
        "baseline_n_kept_docs": len(kept_texts),
        "evicted_control_available": True,
    }
    if not dropped:
        row["baseline_evicted_char_mass"] = 0.0
    elif layout["dropped_text_available"]:
        drop_chars = sum(len(b) for i, b in enumerate(layout["blocks"])
                         if i in set(dropped) and b is not None)
        keep_chars = sum(len(t) for t in kept_texts)
        total = drop_chars + keep_chars
        row["baseline_evicted_char_mass"] = (float(drop_chars) / total) if total else None
    else:
        # HAS drops, no plaintext for them: undefined, never 0.0.
        row["baseline_evicted_char_mass"] = None
    row["baseline_retained_len_entropy"] = retained_length_entropy(
        sidecar_row.get("doc_lengths"))
    return row


def retained_length_entropy(doc_lengths: Optional[Sequence[Any]]) -> Optional[float]:
    """Shannon entropy in nats of the KEPT blocks' token-length distribution
    (2607.21475's "retained entropy" control, in the only retained-mass quantity the
    U2 fixed schema carries).

    ``None`` when there are no lengths, when they sum to zero, or when there is a
    single kept block (entropy 0 there is a degenerate constant, not a measurement).
    """
    vals = [float(x) for x in (doc_lengths or []) if float(x) > 0.0]
    if len(vals) < 2:
        return None
    total = sum(vals)
    if total <= 0:
        return None
    ps = np.array([v / total for v in vals], dtype=float)
    return float(-np.sum(ps * np.log(ps)))


def paired_config_spread_row(rec_a: Dict[str, Any], rec_b: Dict[str, Any]) -> Dict[str, Any]:
    """Spread features for one qid from two capture records (2607.21475 card, E1).

    Emits ``action_disagree``, ``name_disagree``, ``span_nll_spread`` and
    ``logit_spread`` (mean/max per-position JSD on the COMMON emitted prefix), plus
    the mandatory ``mean_output_logprob`` baseline from config A.
    """
    qid = rec_a.get("qid") or rec_b.get("qid")
    text_a = rec_a.get("text") or ""
    text_b = rec_b.get("text") or ""
    pa, pb = parse_tool_call(text_a), parse_tool_call(text_b)
    name_a, name_b = pa.get("name"), pb.get("name")

    steps_a, steps_b = rec_a.get("steps") or [], rec_b.get("steps") or []
    ids_a = [int(s["token_id"]) for s in steps_a]
    ids_b = [int(s["token_id"]) for s in steps_b]
    common = C.first_divergence_index(ids_a, ids_b)
    n_common = common if common is not None else min(len(ids_a), len(ids_b))

    jsds: List[float] = []
    for i in range(n_common):
        v = truncated_jsd(top5_to_dense(steps_a[i].get("top5")),
                          top5_to_dense(steps_b[i].get("top5")))
        if v is not None:
            jsds.append(v)

    nll_a, nll_b = _span_nll(rec_a), _span_nll(rec_b)
    row: Dict[str, Any] = {
        "qid": qid,
        "session_id": C.session_of(qid) if qid else None,
        "spread_action_disagree": float(canonical_action_key(text_a)
                                        != canonical_action_key(text_b)),
        "spread_text_disagree": float(C.normalize_ws(text_a) != C.normalize_ws(text_b)),
        "spread_name_disagree": (None if (name_a is None and name_b is None)
                                 else float(name_a != name_b)),
        "spread_span_nll": nll_a,
        "spread_span_nll_spread": (None if (nll_a is None or nll_b is None)
                                   else float(abs(nll_a - nll_b))),
        "spread_logit_jsd_mean": float(np.mean(jsds)) if jsds else None,
        "spread_logit_jsd_max": float(np.max(jsds)) if jsds else None,
        "spread_n_common_positions": int(n_common),
        "spread_first_token_divergence": common,
        "spread_mean_output_logprob": mean_output_logprob(rec_a),
        "logit_spread_truncated": True,
        "resampling": False,
        "cost_multiplier": SPREAD_COST_MULTIPLIER,
    }
    return row


def paired_config_label_frames(frame: "C.FrozenFrame") -> Dict[str, Any]:
    """The two estimands 2607.21475 insists on reporting separately (section 6.2/6.4).

    * PREDICTION: over ALL 900 paired rows, y = 1[compressed arm wrong].
    * ATTRIBUTION: restricted to the rows where the compressed arm is wrong
      (93 C->W + 619 W->W = 712), y = 1[full arm right].

    LABEL side: both read full-arm / tool_name_match fields, so this function is
    ``*_label_*`` and its output must never reach a feature frame.
    """
    pred_qids, pred_y, attr_qids, attr_y = [], [], [], []
    for f, c in frame.pairs:
        qid = f["qid"]
        c_wrong = not bool(c.get("tool_name_match"))
        pred_qids.append(qid)
        pred_y.append(int(c_wrong))
        if c_wrong:
            attr_qids.append(qid)
            attr_y.append(int(bool(f.get("tool_name_match"))))
    return {
        "prediction": {"qids": pred_qids, "y": np.array(pred_y, dtype=int),
                       "n": len(pred_qids), "n_pos": int(sum(pred_y))},
        "attribution": {"qids": attr_qids, "y": np.array(attr_y, dtype=int),
                        "n": len(attr_qids), "n_pos": int(sum(attr_y))},
    }


# ===========================================================================
# (D) SelfCheckGPT N=1 (2303.08896 section 7.3.1)
# ===========================================================================

def string_number_leaves(arguments: Any) -> List[str]:
    """String and number argument leaves in the frozen witness literal form.

    ``d_witness_core.leaves`` is the frozen stringification (strings verbatim, numbers
    via ``str()``); booleans/None are excluded here because the digest names
    'string/number argument leaves' only, and 'true'/'null' occur in almost any JSON
    text, which would inflate the grounding score.
    """
    out: List[str] = []
    for v in C.json_leaves(arguments):
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, (str, int, float)):
            got = _witness_leaves(v)
            out.extend(got)
    return [s for s in dict.fromkeys(out) if s]


def build_reference_text(sidecar_row: Dict[str, Any]) -> str:
    """The single N=1 reference (2303.08896 section 7.3.1 substitution): the VISIBLE
    decoded doc plaintext plus the current query.

    Pitfall carried in the docstring on purpose: their reference is an authoritative
    WikiBio paragraph; ours is the decoded compressed history -- the very object
    suspected of being lossy.  The substitution's validity under that condition has
    no analogue in the paper.

    Every entry of ``docs`` is VISIBLE by construction: the U2 sidecar dumps the KEPT
    grid rows only, and ``dropped_docs`` indexes the post-split list, not ``docs``
    (see :func:`split_history_blocks`).  Filtering ``docs`` by ``dropped_docs`` would
    delete visible blocks and understate grounding on every row that has a drop, so
    the reference is all of ``docs`` plus the query.
    """
    layout = split_history_blocks(sidecar_row)
    visible = list(layout["kept_texts"])
    return "\n\n".join(visible + [sidecar_row.get("query") or ""])


def lexical_grounding(tool_name: Optional[str], arguments: Any, reference: str
                      ) -> Dict[str, Optional[float]]:
    """Deterministic N=1 lexical grounding (2303.08896 section 5 unigram variant read
    through section 7.3.1; identical in substance to Tracy's ``_argument_grounding_score``,
    bfcl_history_kv_repair.py:484-516).

    Fraction of {tool name} + {string/number argument leaves} occurring case-folded
    verbatim in ``reference``.  ``occurs`` is the frozen witness predicate
    (``d_witness_core.occurs``: substring for values >= 8 chars, word-boundary regex
    below that), applied to case-folded text.

    Two denominators, reported separately and never merged: ``_all`` (name + args) and
    ``_args`` (args only; ``None`` when the call carries no string/number argument --
    48 of the 93 C->W rows are tool-name-only, so the active args denominator is 42/93
    and must never be silently divided by 93).
    """
    ref = (reference or "").casefold()
    args_vals = string_number_leaves(arguments)
    name_vals = [tool_name] if tool_name else []
    all_vals = list(dict.fromkeys([v for v in name_vals + args_vals if v]))

    def _frac(vals: Sequence[str]) -> Optional[float]:
        if not vals:
            return None
        hit = sum(1 for v in vals if _witness_occurs(str(v).casefold(), ref))
        return float(hit / len(vals))

    return {
        "selfcheck_grounding_all": _frac(all_vals),
        "selfcheck_grounding_args": _frac(args_vals),
        "selfcheck_n_leaves_all": len(all_vals),
        "selfcheck_n_leaves_args": len(args_vals),
        "selfcheck_name_only_row": bool(not args_vals),
    }


def verbalize_call(tool_name: Optional[str], arguments: Any) -> Optional[str]:
    """Templated NLI hypothesis for the emitted call (2303.08896 section 5 NLI variant;
    our unit is one <tool_call>, not a sentence of a paragraph).

    'The assistant calls <tool> with <k>=<v>, <k>=<v>.'
    """
    if not tool_name:
        return None
    if isinstance(arguments, dict) and arguments:
        parts = []
        for k, v in arguments.items():
            parts.append(f"{k}={v if isinstance(v, (str, int, float)) else json.dumps(v, ensure_ascii=False)}")
        return f"The assistant calls {tool_name} with " + ", ".join(parts) + "."
    return f"The assistant calls {tool_name} with no arguments."


def contradiction_from_logits(logits: Sequence[float], id2label: Dict[Any, str]
                              ) -> Optional[float]:
    """P(contradict) = exp(z_c) / (exp(z_e) + exp(z_c)) -- 2303.08896 section 5, NLI
    variant (the neutral class is dropped so the score lies in [0,1]).

    The entailment/contradiction indices are read from the checkpoint's ``id2label``
    rather than hardcoded, because MNLI head orderings differ between releases.
    """
    lab = {int(k): str(v).lower() for k, v in id2label.items()}
    idx_e = next((i for i, v in lab.items() if v.startswith("entail")), None)
    idx_c = next((i for i, v in lab.items() if v.startswith("contra")), None)
    if idx_e is None or idx_c is None:
        return None
    z = np.asarray(logits, dtype=np.float64)
    m = max(float(z[idx_e]), float(z[idx_c]))
    ee, ec = math.exp(float(z[idx_e]) - m), math.exp(float(z[idx_c]) - m)
    return float(ec / (ee + ec)) if (ee + ec) > 0 else None


def selfcheck_row(prediction_text: str, sidecar_row: Dict[str, Any],
                  *, nli_fn: Optional[Callable[[str, str], Optional[float]]] = None
                  ) -> Dict[str, Any]:
    """One HOOK-2 SelfCheckGPT N=1 record (2303.08896 section 7.3.1).

    Reads ONLY the compressed arm's own emitted text and the sidecar (decoded docs +
    query).  It must not read target / gold / tool_name_match / any full-arm field:
    the label lives in ``t33_labels.cw_label`` and shares no variable with this
    function (the same-event trap the digest names).

    ``nli_fn(premise, hypothesis) -> P(contradict)`` is injected so the CPU path is
    testable and the DeBERTa-MNLI checkpoint stays lazy.
    """
    parsed = parse_tool_call(prediction_text or "")
    reference = build_reference_text(sidecar_row)
    row: Dict[str, Any] = {"qid": sidecar_row.get("qid")}
    if row["qid"]:
        row["session_id"] = C.session_of(row["qid"])
    row.update(lexical_grounding(parsed.get("name"), parsed.get("arguments"), reference))
    row["selfcheck_parse_ok"] = bool(parsed.get("parse_ok"))
    row["selfcheck_reference_chars"] = len(reference)
    hypothesis = verbalize_call(parsed.get("name"), parsed.get("arguments"))
    row["selfcheck_nli_contradiction"] = (
        nli_fn(reference, hypothesis) if (nli_fn is not None and hypothesis) else None)
    return row


def make_nli_fn(model_name: str, *, max_length: int = 512, device: str = "cpu"
                ) -> Callable[[str, str], Optional[float]]:
    """Lazy DeBERTa-MNLI scorer (2303.08896 section 5 NLI variant).

    Cost columns the caller must book: one small-encoder forward per decision, and the
    checkpoint amortised over the session's block count -- 'neither double-billed nor
    free'.  The model name is an argument; nothing is hardcoded.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer  # lazy
    import torch

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    id2label = dict(mdl.config.id2label)

    def _fn(premise: str, hypothesis: str) -> Optional[float]:
        enc = tok(premise, hypothesis, truncation=True, max_length=max_length,
                  return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = mdl(**enc).logits[0].to(torch.float32).cpu().numpy()
        return contradiction_from_logits(logits, id2label)

    return _fn


# ===========================================================================
# scoring (same row subset for every feature; chance AP = the frame's prevalence)
# ===========================================================================

#: Numeric columns this unit emits as bookkeeping, NOT as candidate detectors.  They
#: are reported with an explicit note rather than silently scored under a default
#: orientation.  ``spread_first_token_divergence`` in particular is undefined (None)
#: exactly on the rows where the two configs agree, so scoring it would select on the
#: outcome; ``spread_action_disagree`` is the total column that covers that contrast.
DIAGNOSTIC_COLS = frozenset({
    "asym_n_positions", "asym_n_name_positions",
    "asymnull_n_positions", "asymnull_n_name_positions",
    "selfcheck_n_leaves_all", "selfcheck_n_leaves_args", "selfcheck_reference_chars",
    "spread_n_common_positions", "spread_first_token_divergence",
    "cost_multiplier", "baseline_n_dropped_docs", "baseline_n_kept_docs",
    "n_dropped_docs_in_view",
    "n_emitted", "n_verified", "n_compared", "first_div_idx", "accept_len",
    "bonus_token_id", "pass_index",
})


#: What a reader must be told about a column at the moment it is ranked, not only in
#: the module docstring.  ``score_feature_table`` stamps these onto the scored rows.
COLUMN_NOTES: Dict[str, str] = {
    "asym_entropy_comp_mean":
        "S0 control (b) measured on the PROXY view: the 1.7B drafter's own entropy "
        "under x_comp_tilde, NOT the 4B's decode entropy on the gist view",
    "asym_entropy_comp_max":
        "S0 control (b) measured on the PROXY view: the 1.7B drafter's own entropy "
        "under x_comp_tilde, NOT the 4B's decode entropy on the gist view",
    "asym_jsd_max_nullctl":
        "S0 control (a) = both passes on x_full: exactly 0 under a deterministic pass, "
        "so this is a determinism sentinel and carries no ranking information",
    "asym_jsd_mean_nullctl":
        "S0 control (a) = both passes on x_full: exactly 0 under a deterministic pass, "
        "so this is a determinism sentinel and carries no ranking information",
    "baseline_evicted_doc_frac":
        "2607.21475 mandatory baseline (3): PREDICTED-weak by the paper's theorem",
    "baseline_evicted_char_mass":
        "2607.21475 mandatory baseline (3): PREDICTED-weak by the paper's theorem",
    "baseline_retained_len_entropy":
        "2607.21475 retained-entropy control (their own row sat at chance, 0.43-0.51); "
        "length analogue, the attention version is kernel-blocked on serving",
    "spread_mean_output_logprob":
        "2607.21475's strongest baseline, not one of our signals",
    "spread_logit_jsd_mean":
        "top-5-truncated: mass outside the capture's top 5 is dropped, so this "
        "OVER-estimates the per-position divergence",
    "spread_logit_jsd_max":
        "top-5-truncated: mass outside the capture's top 5 is dropped, so this "
        "OVER-estimates the per-position divergence",
}


def score_feature_table(feature_rows: Sequence[Dict[str, Any]],
                        frame: "C.FrozenFrame",
                        orientations: Dict[str, int],
                        *, reps: int = 2000) -> Dict[str, Any]:
    """Score every numeric feature column on the 161-row trigger subset.

    Every column is scored on ITS OWN defined rows and reports ``n`` / ``n_pos`` so no
    two columns are compared across different subsets.  Chance AP is the prevalence of
    the scored rows (0.578 on the full 161-row frame), never the 900-frame 0.1033.
    The operating point is matched to the parse-failure baseline's fire count, which is
    the line every signal must beat.
    """
    subset = frame.trigger_subset()
    label = {r["qid"]: int(r["label_cw"]) for r in subset}
    parse_fire = {r["qid"]: bool(r["parse_fail_fire"]) for r in subset}
    by_qid = {r["qid"]: r for r in feature_rows if r.get("qid") in label}

    cols = sorted({k for r in by_qid.values() for k, v in r.items()
                   if k not in C.META_COLS and isinstance(v, (int, float))
                   and not isinstance(v, bool)})
    n_fires = sum(1 for q in by_qid if parse_fire.get(q))
    out: Dict[str, Any] = {
        "n_rows_available": len(by_qid),
        "n_frame": len(subset),
        "baseline_parse_fail_fires": n_fires,
        "columns": {},
    }
    out.update(mandatory_baseline_ledger(cols, [label[q] for q in by_qid], n_fires))
    for col in cols:
        qids = [q for q, r in by_qid.items()
                if isinstance(r.get(col), (int, float)) and not isinstance(r.get(col), bool)
                and np.isfinite(float(r[col]))]
        if col in DIAGNOSTIC_COLS:
            out["columns"][col] = {"n": len(qids),
                                   "note": "diagnostic column, not a scored feature"}
            continue
        if col not in orientations:
            # The module docstring promises no un-oriented score is ever compared or
            # residualised; a silent +1 default would break that promise for any
            # numeric column added later.  Refuse to score instead.
            out["columns"][col] = {"n": len(qids),
                                   "note": "no declared orientation, not scored"}
            continue
        if len(qids) < 4:
            out["columns"][col] = {"n": len(qids), "note": "too few defined rows"}
            continue
        sign = int(orientations[col])
        s = np.array([sign * float(by_qid[q][col]) for q in qids], dtype=float)
        if float(np.std(s)) == 0.0:
            out["columns"][col] = {"n": len(qids), "note": "constant column, not scored"}
            continue
        y = np.array([label[q] for q in qids], dtype=int)
        clusters = C.session_clusters([C.session_of(q) for q in qids])
        ap = C.average_precision(s, y)
        lo, hi, n_cl = C.clustered_bootstrap(C.average_precision, s, y, clusters, reps=reps)
        out["columns"][col] = {
            "orientation": sign,
            "declared_note": COLUMN_NOTES.get(col),
            "n": len(qids), "n_pos": int(y.sum()),
            "chance_ap": C.prevalence(y),
            "auprc": ap, "auprc_ci95": [lo, hi], "n_clusters": n_cl,
            "auroc": C.auroc(s, y),
            "operating_point_at_baseline_fires": C.operating_point(s, y, min(n_fires, len(s))),
        }
    return out


def mandatory_baseline_ledger(cols: Sequence[str], labels: Sequence[int],
                              n_fires: int) -> Dict[str, Any]:
    """Which of 2607.21475's four MANDATORY baselines this table actually contains
    (card, "Mandatory baselines", adopted verbatim).

    The card's discipline is that "a trust signal evaluated without this baseline
    overstates its case", so the read-out names the missing ones instead of letting a
    spread table be published with two of the four absent.  The matched-fire coin is
    owned by another unit's file (``agent/t34_controls.random_matched_rate_table``) and
    is therefore always reported as absent HERE, with the arithmetic a coin would
    achieve (its expected precision at any fire rate is the frame's prevalence).
    """
    have = set(cols)
    y = np.array([int(v) for v in labels], dtype=int)
    prevalence = C.prevalence(y) if y.size else None
    ledger: Dict[str, Any] = {
        "mean_output_logprob": {
            "present": "spread_mean_output_logprob" in have,
            "columns": ["spread_mean_output_logprob"],
            "note": "2607.21475's strongest baseline (0.73-0.80 prediction, "
                    "0.542/0.469/0.519 attribution)",
        },
        "parse_failure_alone": {"present": True, "n_fires": int(n_fires)},
        "evicted_score_mass_predicted_weak": {
            "present": any(c in have for c in EVICTED_CONTROL_COLS),
            "columns": list(EVICTED_CONTROL_COLS),
            "note": "attach with `spread --sidecar results/t34/sidecar_<arm>.jsonl`; "
                    "the theorem predicts it weak, which is worth more than an "
                    "unpredicted weak control",
        },
        "matched_fire_rate_coin": {
            "present": False,
            "owner": "agent/t34_controls.random_matched_rate_table (another unit's file)",
            "expected_precision_at_matched_fire_rate": prevalence,
        },
    }
    extra = {
        "retained_entropy_analogue": {
            "present": any(c in have for c in RETAINED_ENTROPY_COLS),
            "columns": list(RETAINED_ENTROPY_COLS),
            "note": "2607.21475's retained-entropy row sat at chance (0.43-0.51); the "
                    "attention version is kernel-blocked, this is the length analogue",
        },
    }
    missing = [k for k, v in ledger.items() if not v["present"]]
    return {
        "mandatory_baselines_2607_21475": ledger,
        "declared_controls": extra,
        "missing_mandatory_baselines": missing,
        "spread_table_publishable": not missing,
    }


# ===========================================================================
# CLIs
# ===========================================================================

def _cmd_vericache(args: argparse.Namespace) -> int:
    """[NPU] one teacher-forced full-prefix verify pass over the frozen rows."""
    sys.path.insert(0, str(_HERE.parent / "python"))
    sys.path.insert(0, str(_HERE.parent / "python" / "inference"))
    import t33_svip_gamma as SV
    if SV.IMPORT_ERROR is not None:
        print(f"needs torch/transformers (server): {SV.IMPORT_ERROR}", file=sys.stderr)
        return 2
    from eval_agent_history_c2kv import (  # type: ignore
        _build_full_or_truncate_prefix, _clear_device_cache, _current_messages,
        _load_examples, _load_tokenizer, _resolve_model_checkpoint)
    from eval_agent_tool_definition_c2kv import _load_model, _setup_device  # type: ignore
    from train.train_data_multiturn import _chat_template_ids  # type: ignore

    assets = C.FrozenAssets(Path(args.root))
    for given, expected in ((args.battery_full, assets.battery_full),
                            (args.battery_c2kv, assets.battery_c2kv),
                            (args.manifest, assets.manifest_path)):
        if Path(given).resolve() != Path(expected).resolve():
            raise SystemExit(f"frozen-asset mismatch: {given} != {expected}")
    frame = assets.load()
    cap_tokens = frame.cap_tokens()
    c2kv = frame.c2kv_by_qid
    qids = sorted(c2kv)
    max_rows = int(getattr(args, "max_rows", 0) or 0)
    if max_rows:
        # smoke only: a truncated label file is stamped as such in the cost
        # sidecar so it can never be mistaken for the 900-row pass
        qids = qids[:max_rows]
    capture_index = load_capture_steps(Path(args.capture_steps)) if args.capture_steps else {}

    eval_args = SV._build_eval_args(args)
    device = _setup_device(args.device_type)
    eval_args.model = _resolve_model_checkpoint(eval_args.model)
    tokenizer = _load_tokenizer(eval_args)
    model = _load_model(eval_args, tokenizer, device)
    eval_args.qid_allowlist = set(qids)   # start-up cost: only the frozen rows
    examples = {e.qid: e for e in _load_examples(eval_args, tokenizer)[0] if e.qid in set(qids)}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    passes = [out_path] + ([out_path.with_suffix(".repeat.jsonl")] if args.repeat else [])
    for pass_idx, path in enumerate(passes):
        handle = path.open("w", encoding="utf-8")
        t0 = time.time()
        for qid in qids:
            row_c = c2kv[qid]
            ids, src = emitted_ids_for_row(
                qid, capture_index, row_c.get("prediction") or "",
                lambda t: tokenizer.encode(t, add_special_tokens=False),
                decode_fn=lambda i: tokenizer.decode(i, skip_special_tokens=True))
            if not ids or qid not in examples:
                handle.write(json.dumps({"qid": qid, "skipped": "no_ids_or_example"}) + "\n")
                continue
            example = examples[qid]
            prompt_ids = _chat_template_ids(tokenizer, _current_messages(example),
                                            add_generation_prompt=True)
            if eval_args.max_prompt_tokens and len(prompt_ids) > eval_args.max_prompt_tokens:
                prompt_ids = prompt_ids[-eval_args.max_prompt_tokens:]
            prefix, skip = _build_full_or_truncate_prefix(model, tokenizer, example,
                                                          eval_args, "full")
            if prefix is None:
                handle.write(json.dumps({"qid": qid, "skipped": f"full:{skip}"}) + "\n")
                continue
            star = _forward_argmax(model, prefix, prompt_ids, ids, args.attn_impl)
            del prefix
            _clear_device_cache(device)
            if star is None:
                handle.write(json.dumps({"qid": qid, "skipped": "forward_failed"}) + "\n")
                continue
            spans = spans_from_generation(
                lambda t: tokenizer.decode(t, skip_special_tokens=True), ids)
            row = vericache_label_row(qid, ids, star, spans,
                                      cap_tokens=cap_tokens, ids_source=src)
            row["pass_index"] = pass_idx
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.close()
        cost = {"pass": pass_idx, "wall_sec": round(time.time() - t0, 2), "n_rows": len(qids),
                "n_rows_frame": len(c2kv), "max_rows": max_rows,
                "smoke_truncated": bool(max_rows and max_rows < len(c2kv)),
                "booking": "OFFLINE ONLY -- never amortised into the online cost ledger"}
        (path.with_suffix(".cost.json")).write_text(
            json.dumps(cost, indent=1), encoding="utf-8")
        print(json.dumps(cost))
    return 0


def _cmd_vericache_agree(args: argparse.Namespace) -> int:
    a = [r for r in load_jsonl(args.pass_a) if "first_div_idx" in r]
    b = [r for r in load_jsonl(args.pass_b) if "first_div_idx" in r]
    report = vericache_label_disagreement(a, b)
    report["census_a"] = vericache_label_census(a)
    report["census_b"] = vericache_label_census(b)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


def _cmd_asymspec(args: argparse.Namespace) -> int:
    """[NPU] two drafter passes per row; feature file + null control."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy
    import torch  # noqa: F401

    tok = AutoTokenizer.from_pretrained(args.drafter)
    if args.target_tokenizer:
        assert_drafter_family(tok, AutoTokenizer.from_pretrained(args.target_tokenizer),
                              args.drafter, args.target_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.drafter).eval()
    if args.device != "cpu":
        model = model.to(args.device)

    sidecar = {r["qid"]: r for r in load_jsonl(args.sidecar)}
    preds = {r["qid"]: (r.get("prediction") or "") for r in load_jsonl(args.battery_c2kv)}
    enc = lambda t: tok.encode(t, add_special_tokens=False)
    dec = lambda ids: tok.decode(ids, skip_special_tokens=True)

    rows: List[Dict[str, Any]] = []
    variants: Dict[str, int] = {}
    for qid, side in sorted(sidecar.items()):
        parsed = parse_tool_call(preds.get(qid, ""))
        if not parsed.get("has_tool_call"):
            rows.append({"qid": qid, "session_id": C.session_of(qid), "proxy_stamp": PROXY_STAMP})
            continue
        cs, ce = parsed["payload_span"]
        span_text = preds[qid][cs:ce]
        views = build_text_views(side, encode_fn=enc, decode_fn=dec)
        # A row whose dropped blocks have no plaintext is NOT skipped: under the
        # default dump (t34_dump_sidecar without --with_dropped_text) that would be
        # every row that has a drop, i.e. the whole informative subset.  It is scored
        # on the narrower, declared axis (truncation only) and stamped with its
        # proxy_variant; the variants must be stratified downstream, never pooled.
        variants[views["proxy_variant"]] = variants.get(views["proxy_variant"], 0) + 1
        la = _score_span_logits(model, tok, views["x_full"], span_text, args.max_ctx_tokens)
        lb = _score_span_logits(model, tok, views["x_comp"], span_text, args.max_ctx_tokens)
        if la is None or lb is None:
            rows.append({"qid": qid, "session_id": C.session_of(qid), "proxy_stamp": PROXY_STAMP})
            continue
        n = min(la.shape[0], lb.shape[0])
        stats = asymspec_position_stats(la[:n], lb[:n])
        name_pos = _name_positions_in_span(preds[qid], parsed, enc, cs)
        row = {"qid": qid, "session_id": C.session_of(qid), "proxy_stamp": PROXY_STAMP,
               "truncation_mode": views["truncation_mode"],
               "proxy_variant": views["proxy_variant"],
               "view_complete": bool(views["view_complete"]),
               "n_dropped_docs_in_view": int(views["n_dropped_docs"])}
        row.update(asymspec_aggregate(stats, name_pos))
        # S0 null control (card, mandatory control (a)): BOTH passes on x_full.
        # Under a deterministic scoring pass the two logit arrays are IDENTICAL, so
        # the expected value of this column is exactly 0 on every row -- it is a
        # determinism/plumbing sentinel, and `score_feature_table` will report it as a
        # constant column.  Any non-zero value means the drafter pass is not
        # deterministic and every D_i in this file is contaminated by that jitter.
        lb0 = _score_span_logits(model, tok, views["x_full"], span_text, args.max_ctx_tokens)
        if lb0 is not None:
            m = min(la.shape[0], lb0.shape[0])
            null = asymspec_aggregate(asymspec_position_stats(la[:m], lb0[:m]),
                                      prefix="asymnull")
            row["asym_jsd_max_nullctl"] = null["asymnull_jsd_max"]
            row["asym_jsd_mean_nullctl"] = null["asymnull_jsd_mean"]
        rows.append(row)
    n = C.write_features_jsonl(Path(args.out), rows, context="t34 asymspec features")
    degraded = variants.get("truncation_only_missing_dropped_text", 0)
    if degraded:
        print(f"WARNING: {degraded} rows carry the TRUNCATION AXIS ONLY -- the sidecar "
              "was dumped without --with_dropped_text, so the tail-window drop is not "
              "in x_full on those rows.  Stratify by proxy_variant; re-dump with "
              "--with_dropped_text to recover the drop axis.", file=sys.stderr)
    print(json.dumps({"rows": n, "out": args.out, "proxy": PROXY_STAMP,
                      "proxy_variants": dict(sorted(variants.items())),
                      "n_degraded_truncation_only": degraded,
                      "s0_controls": ["asym_jsd_*_nullctl (both passes on x_full)",
                                      "asym_entropy_comp_* (single-ended drafter "
                                      "entropy on x_comp_tilde)"]}))
    return 0


def _name_positions_in_span(text: str, parsed: Dict[str, Any],
                            encode_fn: Callable[[str], Sequence[int]],
                            span_char_start: int) -> List[int]:
    """Token positions of the tool-name value inside the scored span, via the real
    parse (never a prefix heuristic)."""
    ns = parsed.get("name_span")
    if not ns:
        return []
    pre = len(encode_fn(text[span_char_start:ns[0]]))
    inside = len(encode_fn(text[ns[0]:ns[1]]))
    return list(range(pre, pre + max(1, inside)))


def _cmd_spread_plan(args: argparse.Namespace) -> int:
    witness = json.loads(Path(args.witness).read_text(encoding="utf-8"))
    entries = witness.get("entries") or {}
    plan = {"mode": args.mode, "resampling": False,
            "note": "two deterministic configurations of the same content; the same "
                    "request under the same config returns identical content, so the "
                    "repair_fidelity determinism gate is untouched",
            "forbidden_pair": [dict(FORBIDDEN_PACKING_PAIR[0]), dict(FORBIDDEN_PACKING_PAIR[1])],
            "per_qid": {}}
    for qid, ent in sorted(entries.items()):
        pair = config_pair_for_qid(qid, int(ent["n_docs"]), args.mode,
                                   (args.top_k_a, args.top_k_b))
        assert_config_pair_legal(pair["config_a"], pair["config_b"])
        plan["per_qid"][qid] = pair
    sha = C.freeze_json(Path(args.out), plan)
    print(json.dumps({"out": args.out, "sha256": sha, "n_qids": len(plan["per_qid"])}))
    return 0


def _cmd_spread(args: argparse.Namespace) -> int:
    a = load_capture_steps(Path(args.capture_a))
    b = load_capture_steps(Path(args.capture_b))
    sidecar = ({r["qid"]: r for r in load_jsonl(args.sidecar)} if args.sidecar else {})
    rows = []
    n_control = 0
    for q in sorted(set(a) & set(b)):
        row = paired_config_spread_row(a[q], b[q])
        side = sidecar.get(q)
        if side is not None:
            ctl = evicted_mass_control_row(side)
            ctl.pop("qid", None)
            row.update(ctl)
            n_control += 1
        else:
            # DATA AVAILABILITY, said out loud: the mandatory predicted-weak baseline
            # is None on this row, never 0.0, and the flag names why.
            row.update({c: None for c in EVICTED_CONTROL_COLS})
            row.update({c: None for c in RETAINED_ENTROPY_COLS})
            row["evicted_control_available"] = False
        rows.append(row)
    n = C.write_features_jsonl(Path(args.out), rows, context="t34 paired-config spread")
    frame = C.FrozenAssets(Path(args.root)).load()
    frames = paired_config_label_frames(frame)
    missing = [] if n_control == n else ["evicted_score_mass_predicted_weak"]
    missing.append("matched_fire_rate_coin")     # agent/t34_controls, another unit
    if n_control < n:
        print(f"WARNING: {n - n_control} of {n} spread rows have NO evicted-score-mass "
              "control (2607.21475 mandatory baseline 3): pass --sidecar "
              "results/t34/sidecar_<arm>.jsonl.  The spread table must not be "
              "published without it.", file=sys.stderr)
    print(json.dumps({
        "rows": n, "out": args.out,
        "n_rows_with_evicted_control": n_control,
        "missing_mandatory_baselines": missing,
        "estimands": {k: {"n": v["n"], "n_pos": v["n_pos"]} for k, v in frames.items()},
        "cost_multiplier": SPREAD_COST_MULTIPLIER,
        "f_line_success_per_gpus": F_LINE_SUCCESS_PER_GPUS,
    }))
    return 0


def _cmd_selfcheck(args: argparse.Namespace) -> int:
    sidecar = {r["qid"]: r for r in load_jsonl(args.sidecar)}
    preds = {r["qid"]: (r.get("prediction") or "") for r in load_jsonl(args.battery_c2kv)}
    nli_fn = make_nli_fn(args.nli_model, device=args.device) if args.nli_model else None
    t0 = time.time()
    rows = [selfcheck_row(preds.get(q, ""), sidecar[q], nli_fn=nli_fn)
            for q in sorted(sidecar)]
    n = C.write_features_jsonl(Path(args.out), rows, context="t34 selfcheck N=1 features")
    print(json.dumps({
        "rows": n, "out": args.out,
        "n_name_only_rows": sum(1 for r in rows if r.get("selfcheck_name_only_row")),
        "n_args_defined": sum(1 for r in rows if r.get("selfcheck_grounding_args") is not None),
        "nli_model": args.nli_model,
        "cost": {"lexical_gpu_sec": 0.0, "wall_sec": round(time.time() - t0, 2),
                 "nli_amortisation": "checkpoint amortised over the session's block count"},
    }))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    frame = C.FrozenAssets(Path(args.root)).load()
    orient = C.load_orientations(Path(args.orientations)) if args.orientations else {}
    rows: List[Dict[str, Any]] = []
    for p in args.features:
        rows.extend(load_jsonl(p))
    report = score_feature_table(rows, frame, orient, reps=args.reps)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("vericache", help="[NPU] VeriCache full-prefix verify LABEL pass")
    v.add_argument("--root", default=".")
    v.add_argument("--model_path", required=True)
    v.add_argument("--tokenizer_path", required=True)
    v.add_argument("--dataset_path", required=True)
    v.add_argument("--battery_full", required=True)
    v.add_argument("--battery_c2kv", required=True)
    v.add_argument("--manifest", required=True)
    v.add_argument("--max_rows", type=int, default=0,
                   help="smoke only: first N qids; the cost json is stamped smoke_truncated")
    v.add_argument("--capture_steps", default=None,
                   help="t33 capture <arm>/p0.steps.jsonl (preferred id source)")
    v.add_argument("--out", required=True)
    v.add_argument("--ratio", type=int, default=8)
    v.add_argument("--attn_impl", default="eager")
    v.add_argument("--device_type", default="npu")
    v.add_argument("--repeat", action="store_true",
                   help="run the pass twice and write <out>.repeat.jsonl (determinism gate)")
    v.set_defaults(func=_cmd_vericache)

    a = sub.add_parser("vericache-agree", help="[HERE] self-consistency of the label")
    a.add_argument("--pass_a", required=True)
    a.add_argument("--pass_b", required=True)
    a.add_argument("--out", default=None)
    a.set_defaults(func=_cmd_vericache_agree)

    s = sub.add_parser("asymspec", help="[NPU] AsymSpec D_i on the text proxy")
    s.add_argument("--sidecar", required=True)
    s.add_argument("--battery_c2kv", required=True)
    s.add_argument("--drafter", default="Qwen/Qwen3-1.7B")
    s.add_argument("--target_tokenizer", default=None)
    s.add_argument("--device", default="cpu")
    s.add_argument("--max_ctx_tokens", type=int, default=12288)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_asymspec)

    sp = sub.add_parser("spread-plan", help="[HERE] freeze the paired-config plan")
    sp.add_argument("--witness", required=True)
    sp.add_argument("--mode", default="doc_permute", choices=("doc_permute", "hybrid_top_k"))
    sp.add_argument("--top_k_a", type=int, default=2)
    sp.add_argument("--top_k_b", type=int, default=3)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=_cmd_spread_plan)

    d = sub.add_parser("spread", help="[HERE] spread features from two capture files")
    d.add_argument("--capture_a", required=True)
    d.add_argument("--capture_b", required=True)
    d.add_argument("--sidecar", default=None,
                   help="U2 sidecar; attaches 2607.21475's mandatory evicted-score-mass "
                        "control and the retained-entropy control.  Without it those "
                        "columns are None and the run warns.")
    d.add_argument("--root", default=".")
    d.add_argument("--out", required=True)
    d.set_defaults(func=_cmd_spread)

    c = sub.add_parser("selfcheck", help="[HERE] SelfCheckGPT N=1 features")
    c.add_argument("--sidecar", required=True)
    c.add_argument("--battery_c2kv", required=True)
    c.add_argument("--nli_model", default=None,
                   help="e.g. a DeBERTa-MNLI checkpoint; omitted -> lexical scores only")
    c.add_argument("--device", default="cpu")
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_selfcheck)

    sc = sub.add_parser("score", help="[HERE] score a feature file on the 161-row subset")
    sc.add_argument("--features", nargs="+", required=True)
    sc.add_argument("--root", default=".")
    sc.add_argument("--orientations", default="configs/t34/orientations_extra.json")
    sc.add_argument("--reps", type=int, default=2000)
    sc.add_argument("--out", default=None)
    sc.set_defaults(func=_cmd_score)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
