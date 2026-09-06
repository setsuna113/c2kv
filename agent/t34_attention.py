# -*- coding: utf-8 -*-
"""t34 U1a - S6 attention family: capture primitive + Lookback Lens + Elastic-Cache
+ Retrieval-Head ports 1/2 (digest 33_..._2026-09-04.md, section 4.5, lines 657-712).

RUNBOOK (execution order; "SERVER" = NPU box with torch, "LOCAL" = this Windows box)
-----------------------------------------------------------------------------------
0. SERVER, once per arm - re-run the 161-row trigger subset with the attention
   capture installed (eager attention is mandatory; the Ascend fused kernel
   returns outputs only and exposes no probabilities):

     PYTHONIOENCODING=utf-8 python agent/t34_attention.py run-battery \\
       --arm c2kv --model_path <fixed_joint ckpt> --base_model <Qwen3-4B-Instruct-2507-FC> \\
       --tokenizer_path <tok> --dataset_path <agent-llm-traces> \\
       --battery_full results/bdf_pilot/d_r2/battery_full.jsonl \\
       --battery_c2kv results/bdf_pilot/d_r2/battery_c2kv.jsonl \\
       --manifest configs/bdf_pilot/d_cw_manifest_r2.json \\
       --out_dir results/t34 --attn_impl eager

     ... and again with --arm full (S0 twin; LR_gist is undefined there and is
     written as null, never as 0.0).

1. SERVER, optional determinism probe for the Elastic-Cache drift features -
   two identical forwards of the same rows, then the bf16 noise-floor report:

     PYTHONIOENCODING=utf-8 python agent/t34_attention.py run-battery \\
       --arm c2kv ... --out_dir results/t34/noise_b
     PYTHONIOENCODING=utf-8 python agent/t34_attention.py noise-floor \\
       --a results/t34/attn_c2kv.npz --b results/t34/noise_b/attn_c2kv.npz

2. LOCAL - per-qid scalar features (writes through write_features_jsonl, so the
   leakage guard runs on every column name):

     PYTHONIOENCODING=utf-8 python agent/t34_attention.py extract-features \\
       --capture results/t34/attn_c2kv.npz --meta results/t34/attn_c2kv.jsonl \\
       --arm c2kv --root . --out results/t34/features_attn_c2kv.jsonl \\
       --docs-sidecar results/t34/sidecar_c2kv.jsonl
     (repeat with --arm full --capture results/t34/attn_full.npz ... for the S0 twin)

     WITHOUT --docs-sidecar the S8 control column n_dropped_docs is null for every
     row (the battery generation row does not carry dropped_docs) and the CLI
     prints the S8_CONTROL_INCOMPLETE flag naming it; the control block then runs
     on four of its five pre-registered columns.

3. LOCAL - Lookback-Lens probe on the L*H*3 vector, with the S8 control block and
   the increment over it (session-grouped nested CV, selection in inner folds only):

     PYTHONIOENCODING=utf-8 python agent/t34_attention.py probe \\
       --capture results/t34/attn_c2kv.npz --meta results/t34/attn_c2kv.jsonl \\
       --features results/t34/features_attn_c2kv.jsonl --root . \\
       --out results/t34/probe_lookback_c2kv.json

4. LOCAL - Elastic-Cache drift block.  Layer band x score mode are chosen in
   inner folds, and the cross-step arm's own denominator (rows with a predecessor
   decision step IN THE FRAME) is printed next to the full one, never multiplied:

     PYTHONIOENCODING=utf-8 python agent/t34_attention.py drift-probe \\
       --capture results/t34/attn_c2kv.npz --meta results/t34/attn_c2kv.jsonl \\
       --root . --out results/t34/probe_drift_c2kv.json

5. LOCAL - Retrieval-Head port 1 (locator) and port 2 (trigger scalars).  The head
   set comes from unit U1b's port 0 (needle detection on raw contexts):

     PYTHONIOENCODING=utf-8 python agent/t34_attention.py locate \\
       --capture results/t34/attn_c2kv.npz --meta results/t34/attn_c2kv.jsonl \\
       --head-set configs/t34/retrieval_heads_port0.json --root . \\
       --out results/t34/locate_retrieval_heads.json

MEMORY.  Our cost is Q x n_keys per layer, not L x L: the reduction is streamed
one layer at a time and, inside a layer, one query-row chunk at a time, so the
peak float32 buffer is [H, qrow_chunk, n_keys].  Nothing of size [L, ...] over
keys is ever materialised, and the on-disk artifact is span-reduced
([L, H, 5] + [L, H, D] + [L, H, 3]), about 120 KB per battery row.

CROSS-UNIT INTERFACE.  ``CLASSES``, ``KeyClassMap`` and ``AttentionRowCapture``
are the contract unit U1b (QRHead / Retrieval-Head port 0 / CacheBlend) codes
against.  Do not change their names or shapes without telling U1b.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from t34_common import (  # noqa: E402
    FrozenAssets,
    average_precision,
    auroc,
    chooser_argmax,
    clustered_bootstrap,
    freeze_json,
    inverted_score_control,
    locate_table,
    load_decoded_docs,
    load_flip_table,
    mcnemar_exact,
    nested_cv_logistic,
    operating_point,
    paired_delta_bootstrap,
    prevalence,
    session_clusters,
    session_of,
    step_index,
    write_features_jsonl,
)
from t33_spanmap import spans_from_generation  # noqa: E402


DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "Lookback Lens",
        "paper": "2407.07071",
        "what": "A third ratio LR_gist_vs_raw = A(history_gist) / (A(history_gist) + A(history_raw_tail)) "
                "is added; the paper's split is binary (context vs newly generated) and has no counterpart.",
        "why": "digest 4.5: the compression-specific quantity the paper cannot have. It is declared as a "
               "hypothesis-bearing feature, not as a reproduction of the paper.",
    },
    {
        "method": "Lookback Lens",
        "paper": "2407.07071",
        "what": "AUPRC against the evaluation frame's own prevalence is the primary metric; the paper "
                "reports AUROC only and offers no threshold-calibration protocol (card: 'Decision rule').",
        "why": "digest 4.0 winner rule; the three deployment metrics all need an operating point, which is "
               "taken here from fixed_rate_threshold inside the CV, never from the evaluation rows.",
    },
    {
        "method": "Lookback Lens",
        "paper": "2407.07071",
        "what": "The paper's top-100-by-|coef| head study is a POST-HOC importance analysis; here it is a "
                "selector re-fitted inside every inner training fold (make_topk_coef_selector).",
        "why": "hard rule 6 - nothing may be chosen on evaluation rows. Card, 'Head selection': the paper's "
               "top-k is 'not a selection procedure applied before training'.",
    },
    {
        "method": "Lookback Lens",
        "paper": "2407.07071",
        "what": "The S8 prefix scalars (actual_compression_ratio, gist_tokens, kept_history_tokens, "
                "n_dropped_docs, n_docs) are fitted as a control block and the attention features are only "
                "credited with the increment over it.",
        "why": "digest 4.5 risk: gist tokens are not natural tokens, so 1/N normalisation can make LR_gist a "
               "re-derivation of actual_compression_ratio, which is already free from S8.",
    },
    {
        "method": "Elastic-Cache",
        "paper": "2510.14973",
        "what": "The paper's drift test is between two DENOISING steps of a masked diffusion LLM, where "
                "committed KV genuinely changes. Here the two reference points are two adjacent DECISION "
                "STEPS of the same session (joined by session_of/step_index), and the vector compared is the "
                "per-doc attention mass vector, not an attention row over a single most-attended token.",
        "why": "card, 'Mechanism' domain caveat: in causal autoregressive decoding the literal drift does not "
               "exist, so only the shape of the test transfers. digest 4.5 says to re-anchor the step.",
    },
    {
        "method": "Elastic-Cache",
        "paper": "2510.14973",
        "what": "gamma is never swept on the evaluation rows; no gamma is used at all here - the drift "
                "statistics are emitted as continuous features and any operating point is chosen in inner "
                "folds.",
        "why": "card flags the paper's gamma sweep directly on GSM8K as a defect not to copy (no held-out "
               "calibration set, two models disagree on the optimum).",
    },
    {
        "method": "Elastic-Cache",
        "paper": "2510.14973",
        "what": "The recompute action (layers l*+1..L) is not implemented; only the WHEN half is ported.",
        "why": "this unit delivers features, not a repair arm.",
    },
    {
        "method": "Retrieval Head",
        "paper": "2404.15574",
        "what": "The retrieval SCORE (token-identity copy test, |g_h & k| / |k|) is not ported: it is "
                "undefined on gist spans, which have no token identity. Only the HEAD SET transfers, and it "
                "is produced by unit U1b's port 0 on raw needle contexts.",
        "why": "digest 4.5, verbatim: 'only the head set is transferable, the score is not'.",
    },
    {
        "method": "Retrieval Head",
        "paper": "2404.15574",
        "what": "The paper's 0.1 threshold on the retrieval score is not used here (it belongs to port 0). "
                "Ports 1/2 consume whatever head list the port-0 json declares and record its provenance.",
        "why": "card: the 0.1 cut has 'no sweep, no validation set, no sensitivity analysis'.",
    },
    {
        "method": "Retrieval Head",
        "paper": "2404.15574",
        "what": "sink_ratio is emitted in two forms - mass on key 0 exactly, and mass on the whole system "
                "prefix class - because section 4.1's observation is about 'the initial token of the input' "
                "while our system prefix is many tokens long.",
        "why": "keeping only one of them would silently pick an interpretation; both orientations are +1 and "
               "pre-registered in configs/t34/orientations_attn.json.",
    },
    {
        "method": "Lookback Lens",
        "paper": "2407.07071",
        "what": "The section-5 LAYER ablation (middle layers best, late layers worst) is a reported "
                "table in the paper; here it is an inner-fold selector (layers_early / middle / late) "
                "chosen jointly with C, never a band picked after seeing the result.",
        "why": "hard rule 6 - nothing may be chosen on evaluation rows.",
    },
    {
        "method": "Lookback Lens",
        "paper": "2407.07071",
        "what": "Feature COLUMNS non-finite for any row are dropped and counted "
                "(drop_undefined_columns) rather than the rows being dropped by nested_cv_logistic. "
                "LR_gist_vs_raw is undefined for every row of the pure c2kv recipe (no raw tail) and "
                "LR_gist for every row of the full arm, but the rule is applied to ANY non-finite "
                "entry, which is stricter than 'undefined everywhere'.",
        "why": "keeping them would drop ROWS, not columns. Nothing is imputed; the count is reported "
               "next to every metric, and for the S8 control block the dropped columns are "
               "reported BY NAME (s8_columns_dropped_undefined). RESIDUAL: this is one global, "
               "LABEL-FREE decision taken over all rows including the evaluation ones - it reads "
               "the missingness pattern only, never a label - rather than a per-fold one; a "
               "per-fold column set would make the feature space fold-dependent and would drop "
               "rows in the folds where an all-undefined column reappears.",
    },
    {
        "method": "Lookback Lens",
        "paper": "2407.07071",
        "what": "The mitigation half (classifier-guided decoding, 8 sampled candidates per 8-token "
                "chunk) is not ported at all.",
        "why": "it needs temperature sampling, which the determinism gate forbids; the paper itself "
               "calls it a robustness test of the detector rather than the contribution.",
    },
    {
        "method": "Lookback Lens",
        "paper": "2407.07071",
        "what": "The paper's `context` is one undifferentiated block of length N; ours is the union of "
                "four prompt classes (system_raw / history_gist / history_raw_tail / current_query). "
                "A(context) pools their mass and divides ONCE by the pooled key count, exactly as "
                "eq. (1) - it is not the sum of the four classes' individual per-key means.",
        "why": "summing per-class means would make LR_context depend on how many prompt classes "
               "happen to be non-empty (0.75 without a raw tail vs 0.80 with one under a uniform "
               "attention distribution) and would bias the S0 full-arm twin by class composition.",
    },
    {
        "method": "Elastic-Cache",
        "paper": "2510.14973",
        "what": "Cross-step doc alignment is pre-registered: sha256 when both steps carry per-doc "
                "hashes (and, when they share no block, the row is reported unalignable rather than "
                "silently re-joined by index), else block index truncated to min(n_docs). "
                "attn_cos_prev_step / "
                "attn_drop_maxdoc are None (never 0.0) for rows with no predecessor in the frame, and "
                "the cross-step denominator is reported next to the within-step one, never multiplied.",
        "why": "digest 4.5: the cross-step denominator has to be counted first and both denominators "
               "reported, never multiplied.",
    },
    {
        "method": "Retrieval Head",
        "paper": "2404.15574",
        "what": "RESIDUAL RISK, stated not smoothed over: the head set is selected on RAW needle "
                "tokens by port 0, while gist vectors are objects the base model never saw. Port 0's "
                "raw-input correlation check (r > 0.8) verifies head-set stability on raw inputs "
                "only; nothing here verifies those heads do the same thing on gist spans.",
        "why": "digest 4.5 names this as the port's most dangerous pitfall and requires it be written "
               "down rather than smoothed over.",
    },
    {
        "method": "Retrieval Head",
        "paper": "2404.15574",
        "what": "ONE truth source per locator table (locator_truth). With a --flip-table, a qid that "
                "flips at more than one k AND a qid the table does not cover both ABSTAIN; neither "
                "is resolved to min(k) nor replaced by the frozen witness k*. locate_table counts "
                "an abstention in the denominator as a miss and reports it separately, and the "
                "report carries truth_provenance (source + the three abstention counts).",
        "why": "resolving a multi-flip qid to min(k) would bias S@k toward the first-block prior, "
               "which is one of the paired comparison arms (k_first); per-qid fallback to the "
               "gold-scored witness would make one S@k denominator a blend of two truth "
               "definitions (repair-flip vs gold) under a single reported source.",
    },
    {
        "method": "Retrieval Head",
        "paper": "2404.15574",
        "what": "The locator's chooser is the frozen d_witness_core.select_k_star semantics "
                "(t34_common.chooser_argmax: abstain when nothing scores above zero, ties to the "
                "lowest index), not a bare argmax over the per-doc score vector.",
        "why": "the inverted-score specificity control already runs that chooser; a bare argmax "
               "would answer rows the control abstains on, so the locator table and its own "
               "control's forward arm would be computed over different qid sets.",
    },
    {
        "method": "Elastic-Cache / shared feature contract",
        "paper": "2510.14973",
        "what": "The history-slot share of TOTAL attention mass is written as ONE column, "
                "attn_gist_frac (digest 4.5's own name), with one declared orientation (-1). An "
                "earlier draft also wrote attn_history_gist_frac - the same number by construction "
                "- with its own orientation entry; the duplicate is deleted from both the jsonl and "
                "configs/t34/orientations_attn.json.",
        "why": "two perfectly collinear columns with independently declared orientations are a "
               "footgun for anything that ranks or residualises features. The -1 is the "
               "context-vs-self direction of 2407.07071 (its complement, "
               "attn_generated_so_far_frac, is +1); it does not contradict attn_gist_ratio / "
               "attn_rh_gist_ratio (+1), whose denominator is gist + raw tail only and which "
               "therefore declare a WITHIN-history claim, not a context-vs-self one.",
    },
    {
        "method": "capture primitive",
        "paper": "2407.07071",
        "what": "On the eager path the recomputed probabilities are compared against the kernel's own "
                "returned attn_weights on every captured chunk; the running max |d| and the number of "
                "checks go into every capture meta line (recompute_max_abs_diff / "
                "recompute_n_checked), and a difference above recompute_tol=1e-2 (dtype-matched replica of the eager kernel; the fp32 gap is reported separately) raises the "
                "'recompute_mismatch' error counter.",
        "why": "the recomputation deviation below is only safe if it reproduces the eager kernel's "
               "alpha; without this check a wrong post-RoPE query would make every downstream number "
               "uninterpretable AND silent.",
    },
    {
        "method": "capture primitive",
        "paper": "2407.07071 / 2404.15574 / 2510.14973 (shared instrument)",
        "what": "Attention probabilities are recomputed from the layer's post-RoPE query states and the "
                "post-update cache keys AFTER the original forward has run, rather than read out of the "
                "eager kernel's returned attn_weights.",
        "why": "Qwen3Attention.forward_with_gist (modeling_qwen3.py:350) discards attn_weights entirely, and "
               "nn.Module forward hooks are bypassed on that path; recomputation is the only route that is "
               "identical on both paths. Cost: one extra q_proj per captured layer, no extra attention.",
    },
    {
        "method": "capture primitive",
        "paper": "shared instrument",
        "what": "Gist spans from _gist_spans_from_doc_lengths can overlap by one token at block boundaries "
                "(end_i = ceil(c*g/T) >= start_{i+1} = floor(c*g/T)); KeyClassMap truncates each span's "
                "start to the previous span's end and counts the adjustments in n_span_adjustments.",
        "why": "the per-doc mass vector must be a partition or the masses double-count; the alternative "
               "(dropping the shared token) loses mass. The count is reported, never hidden.",
    },
    {
        "method": "capture primitive",
        "paper": "shared instrument",
        "what": "The on-disk artifact stores SPAN-REDUCED arrays ([L,H,5] + [L,H,D] + [L,H,3], about "
                "120 KB per row) rather than the per-step [L,H,Q,K] block; the lookback ratio is "
                "averaged over the <tool_call> span AS A RATIO before that reduction.",
        "why": "2407.07071's v-bar is the span mean of the ratio and LR is non-linear, so the ratio "
               "must be formed per step and averaged afterwards - which is what is stored.",
    },
    {
        "method": "capture primitive",
        "paper": "shared instrument",
        "what": "forward_with_gist is patched with a COUNTING wrapper only; the gist-path reduction "
                "is NOT implemented and capture_gist_path=True is refused at construction "
                "(NotImplementedError), not accepted and silently reduced to an error counter.",
        "why": "the gist path is the compression build pass, not a decision-step forward, and re-running "
               "apply_gist_residual to recover its query states could have side effects. The counter makes "
               "a bypassed capture loud instead of silent (t33 audit pitfall: 'hooks that the gist path "
               "bypasses'); refusing the flag makes an unimplemented capture loud instead of empty.",
    },
]


# ---------------------------------------------------------------------------
# (A) the shared attention-capture primitive
# ---------------------------------------------------------------------------

#: Cross-unit interface: the five key classes, in tensor-column order.
CLASSES: Tuple[str, ...] = (
    "system_raw",
    "history_gist",
    "history_raw_tail",
    "current_query",
    "generated_so_far",
)
CLASS_INDEX: Dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

#: Feature-column name written for each key class's share of the TOTAL attention
#: mass.  The ``history_gist`` class is written under the digest's own name
#: ``attn_gist_frac`` and under that name ONLY: an earlier draft also wrote
#: ``attn_history_gist_frac`` (the same number by construction) with a separately
#: declared orientation, which is two perfectly collinear columns telling a
#: ranker two different stories.  ``system_raw`` is not in this map because its
#: share is written as ``attn_sink_ratio_system`` (2404.15574 section 4.1).
CLASS_FRAC_COLUMNS: Dict[str, str] = {
    "history_gist": "attn_gist_frac",
    "history_raw_tail": "attn_history_raw_tail_frac",
    "current_query": "attn_current_query_frac",
    "generated_so_far": "attn_generated_so_far_frac",
}

#: Documented NPU bf16 matmul rounding floor (controlled probe, digest 4.5 risk
#: paragraph).  Any drift threshold sitting below this is inside the noise.
BF16_NOISE_FLOOR = 0.0078125

SCORE_MODES: Tuple[str, ...] = ("sum", "sqrt_len", "mean")


def gist_spans_from_doc_lengths(doc_lengths: Sequence[int], gist_tokens: int) -> List[Tuple[int, int]]:
    """Torch-free mirror of ``eval_agent_history_c2kv._gist_spans_from_doc_lengths``
    (eval_agent_history_c2kv.py:2568).

    Copied rather than imported because that module imports torch, which is not
    installed on the analysis box.  ``test_t34_attention`` asserts equality
    against the original whenever torch is importable.
    """
    total = int(sum(doc_lengths))
    if total <= 0 or gist_tokens <= 0:
        return [(0, 0) for _ in doc_lengths]
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for length in doc_lengths:
        start = int(cursor * gist_tokens / total)
        cursor += length
        end = int((cursor * gist_tokens + total - 1) / total)
        if end <= start:
            end = min(gist_tokens, start + 1)
        spans.append((max(0, start), min(gist_tokens, end)))
    return spans


class KeyClassMap:
    """Absolute key position -> (class, history block).

    Parameters
    ----------
    system_len:
        number of system-prefix keys; class ``system_raw`` covers ``[0, system_len)``.
    doc_spans:
        absolute ``(start, end)`` key spans of each history block, in block order.
        On the compressed arm these are GIST spans
        (``gist_spans_from_doc_lengths(doc_lengths, gist_tokens)`` shifted by
        ``system_len``); on the full arm they are the raw per-doc spans.  Class
        ``history_gist`` on both arms - the class name marks the SLOT, the arm
        marks what lives in it.
    raw_tail_span:
        absolute span of the uncompressed hybrid tail, or None.
    query_span:
        absolute span of the current-query tokens.
    generated_start:
        first absolute key position of the model's own continuation.

    Overlapping ``doc_spans`` (which the gist span construction can produce at
    block boundaries) are normalised by truncating each span's start to the
    previous span's end; the number of adjustments is exposed as
    ``n_span_adjustments``.
    """

    def __init__(
        self,
        system_len: int,
        doc_spans: Sequence[Tuple[int, int]],
        raw_tail_span: Optional[Tuple[int, int]],
        query_span: Tuple[int, int],
        generated_start: int,
    ) -> None:
        self.system_len = int(system_len)
        self.raw_tail_span = tuple(raw_tail_span) if raw_tail_span else None
        self.query_span = (int(query_span[0]), int(query_span[1]))
        self.generated_start = int(generated_start)

        spans: List[Tuple[int, int]] = []
        adjust = 0
        prev_end = self.system_len
        for start, end in doc_spans:
            s, e = int(start), int(end)
            if s < prev_end:
                adjust += 1
                s = prev_end
            if e < s:
                e = s
            spans.append((s, e))
            prev_end = max(prev_end, e)
        self.doc_spans: List[Tuple[int, int]] = spans
        self.n_span_adjustments = adjust

        self._validate()

    # -- construction helpers ------------------------------------------------

    @classmethod
    def from_prefix(
        cls,
        *,
        arm: str,
        system_length: int,
        doc_lengths: Sequence[int],
        gist_tokens: int = 0,
        raw_tail_tokens: int = 0,
        query_len: int,
        history_length: Optional[int] = None,
    ) -> "KeyClassMap":
        """Build from the battery assembly bookkeeping.

        ``arm='c2kv'`` -> block spans are gist spans of width ``gist_tokens``;
        ``arm='full'`` -> block spans are the raw per-doc spans (cumulative
        ``doc_lengths``).  ``raw_tail_tokens`` is the hybrid uncompressed tail
        appended after the compressed region (0 in the pure c2kv recipe).
        """
        sl = int(system_length)
        if arm == "c2kv":
            rel = gist_spans_from_doc_lengths(doc_lengths, int(gist_tokens))
            spans = [(sl + a, sl + b) for a, b in rel]
            compressed_len = int(gist_tokens)
        else:
            spans = []
            cur = sl
            for length in doc_lengths:
                spans.append((cur, cur + int(length)))
                cur += int(length)
            compressed_len = int(sum(int(x) for x in doc_lengths))
        tail_start = sl + compressed_len
        raw_tail = (tail_start, tail_start + int(raw_tail_tokens)) if raw_tail_tokens else None
        hist_end = tail_start + int(raw_tail_tokens)
        if history_length is not None:
            hist_end = sl + int(history_length)
        q0 = hist_end
        q1 = q0 + int(query_len)
        return cls(sl, spans, raw_tail, (q0, q1), q1)

    # -- contract ------------------------------------------------------------

    @property
    def n_docs(self) -> int:
        return len(self.doc_spans)

    def _regions(self) -> List[Tuple[int, int, int]]:
        out: List[Tuple[int, int, int]] = []
        if self.system_len > 0:
            out.append((0, self.system_len, CLASS_INDEX["system_raw"]))
        for s, e in self.doc_spans:
            if e > s:
                out.append((s, e, CLASS_INDEX["history_gist"]))
        if self.raw_tail_span and self.raw_tail_span[1] > self.raw_tail_span[0]:
            out.append((self.raw_tail_span[0], self.raw_tail_span[1],
                        CLASS_INDEX["history_raw_tail"]))
        if self.query_span[1] > self.query_span[0]:
            out.append((self.query_span[0], self.query_span[1],
                        CLASS_INDEX["current_query"]))
        return out

    def _validate(self) -> None:
        regions = sorted(self._regions())
        for (s0, e0, c0), (s1, e1, c1) in zip(regions, regions[1:]):
            if s1 < e0:
                raise ValueError(
                    f"KeyClassMap: overlapping regions {CLASSES[c0]}[{s0},{e0}) and "
                    f"{CLASSES[c1]}[{s1},{e1})"
                )
        if regions and regions[-1][1] > self.generated_start:
            raise ValueError(
                f"KeyClassMap: prompt regions run past generated_start={self.generated_start}"
            )

    def class_ids(self, n_keys: int) -> np.ndarray:
        """``int`` array of length ``n_keys``; values index ``CLASSES``, -1 = unclassified."""
        out = np.full(int(n_keys), -1, dtype=np.int64)
        for s, e, c in self._regions():
            s2, e2 = max(0, s), min(int(n_keys), e)
            if e2 > s2:
                out[s2:e2] = c
        g = max(0, min(int(n_keys), self.generated_start))
        if g < int(n_keys):
            out[g:] = CLASS_INDEX["generated_so_far"]
        return out

    def doc_ids(self, n_keys: int) -> np.ndarray:
        """``int`` array of length ``n_keys``; history block index, -1 elsewhere."""
        out = np.full(int(n_keys), -1, dtype=np.int64)
        for k, (s, e) in enumerate(self.doc_spans):
            s2, e2 = max(0, s), min(int(n_keys), e)
            if e2 > s2:
                out[s2:e2] = k
        return out

    def doc_span_lens(self) -> np.ndarray:
        return np.array([max(0, e - s) for s, e in self.doc_spans], dtype=np.int64)

    def coverage(self, n_keys: int) -> Dict[str, int]:
        cid = self.class_ids(n_keys)
        return {
            "n_keys": int(n_keys),
            "n_unclassified": int((cid < 0).sum()),
            **{name: int((cid == i).sum()) for i, name in enumerate(CLASSES)},
        }


# -- the reduction, shared by the numpy reference and the torch streaming path --

def reduce_probs(
    probs: np.ndarray,
    class_ids: np.ndarray,
    doc_ids: np.ndarray,
    n_docs: int,
) -> Dict[str, np.ndarray]:
    """Reduce a ``[H, Q, K]`` probability block to the per-(head, query-row) record.

    Returns ``class_mass [H,Q,5]``, ``doc_mass [H,Q,D]``, ``argmax_key [H,Q]``,
    ``argmax_val [H,Q]``, ``doc_entropy [H,Q]`` (Shannon entropy in nats of the
    normalised per-doc vector; nan when the doc mass is zero) and
    ``sink_mass [H,Q]`` (probability on absolute key 0, section 4.1 of 2404.15574).
    """
    probs = np.asarray(probs, dtype=np.float32)
    h, q, k = probs.shape
    class_mass = np.zeros((h, q, N_CLASSES), dtype=np.float32)
    for c in range(N_CLASSES):
        sel = class_ids[:k] == c
        if sel.any():
            class_mass[:, :, c] = probs[:, :, sel].sum(axis=2)
    doc_mass = np.zeros((h, q, int(n_docs)), dtype=np.float32)
    for d in range(int(n_docs)):
        sel = doc_ids[:k] == d
        if sel.any():
            doc_mass[:, :, d] = probs[:, :, sel].sum(axis=2)
    argmax_key = probs.argmax(axis=2).astype(np.int64)
    argmax_val = np.take_along_axis(probs, argmax_key[:, :, None], axis=2)[:, :, 0]
    tot = doc_mass.sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = doc_mass / tot[:, :, None]
    ent = np.where(p > 0, -p * np.log(np.where(p > 0, p, 1.0)), 0.0).sum(axis=2)
    ent = np.where(tot > 0, ent, np.nan).astype(np.float32)
    sink = probs[:, :, 0] if k > 0 else np.zeros((h, q), dtype=np.float32)
    return {
        "class_mass": class_mass,
        "doc_mass": doc_mass.astype(np.float32),
        "argmax_key": argmax_key,
        "argmax_val": argmax_val.astype(np.float32),
        "doc_entropy": ent,
        "sink_mass": np.asarray(sink, dtype=np.float32),
    }


def reduce_attention_rows(
    q: np.ndarray,
    k: np.ndarray,
    key_map: KeyClassMap,
    causal_mask: Optional[np.ndarray] = None,
    *,
    scaling: Optional[float] = None,
    query_chunk: Optional[int] = None,
    return_probs: bool = False,
) -> Dict[str, np.ndarray]:
    """Pure-numpy reference implementation of the streaming reduction.

    ``q``: ``[H, Q, D]`` post-RoPE query states of the SELECTED query rows.
    ``k``: ``[H_kv, K, D]`` post-RoPE key states over ALL keys (cache + current);
    KV heads are repeated for GQA exactly as ``modeling_qwen3.repeat_kv``
    (head ``h`` reads KV head ``h // (H // H_kv)``).
    ``causal_mask``: additive ``[Q, K]`` mask (0 / -inf), or None for full visibility.

    ``query_chunk`` splits the query rows so the reference exercises the same
    chunked path as ``AttentionRowCapture``; the result is identical for every
    chunk size (asserted in the tests).
    """
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    n_h, n_q, dim = q.shape
    n_kv, n_k, dim_k = k.shape
    if dim != dim_k:
        raise ValueError(f"head dim mismatch: q {dim} vs k {dim_k}")
    if n_h % n_kv:
        raise ValueError(f"GQA mismatch: {n_h} query heads, {n_kv} kv heads")
    n_rep = n_h // n_kv
    k_rep = np.repeat(k, n_rep, axis=0)
    scale = float(dim ** -0.5) if scaling is None else float(scaling)

    cid = key_map.class_ids(n_k)
    did = key_map.doc_ids(n_k)
    chunk = int(query_chunk) if query_chunk else n_q
    chunk = max(1, chunk)

    parts: List[Dict[str, np.ndarray]] = []
    probs_all: List[np.ndarray] = []
    for lo in range(0, n_q, chunk):
        hi = min(n_q, lo + chunk)
        logits = np.einsum("hqd,hkd->hqk", q[:, lo:hi, :], k_rep, dtype=np.float32)
        logits = (logits * np.float32(scale)).astype(np.float32)
        if causal_mask is not None:
            logits = logits + np.asarray(causal_mask, dtype=np.float32)[None, lo:hi, :]
        mx = logits.max(axis=2, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, np.float32(0.0))
        ex = np.exp((logits - mx).astype(np.float32))
        ex = np.where(np.isfinite(logits), ex, np.float32(0.0))
        denom = ex.sum(axis=2, keepdims=True)
        probs = np.where(denom > 0, ex / denom, np.float32(0.0)).astype(np.float32)
        parts.append(reduce_probs(probs, cid, did, key_map.n_docs))
        if return_probs:
            probs_all.append(probs)
        del logits, ex, probs

    out = {key: np.concatenate([p[key] for p in parts], axis=1) for key in parts[0]}
    if return_probs:
        out["probs"] = np.concatenate(probs_all, axis=1)
    return out


class CaptureContractError(RuntimeError):
    """The capture's declared shape contract was violated by a real forward.

    Raised inside ``_capture`` (whose exceptions are counted, never propagated,
    so instrumentation can never kill a battery row).  It gets its own error
    counter name, so a batched harness or an unexpected cache layout shows up as
    ``capture_errors={'CaptureContractError': n}`` with an empty capture, instead
    of silently reducing batch element 0 only.
    """


def check_key_layout(keys: Any) -> Any:
    """Assert the cache layout the reduction indexes: ``[1, n_kv_heads, n_keys,
    head_dim]`` (a DynamicCache-like ``.layers[i].keys`` / ``.key_cache[i]``).

    Separated from ``_capture`` so the contract is a named, unit-tested function
    rather than an inline branch that only a real forward can reach: a batched
    cache would otherwise be reduced as batch element 0 while the meta line
    claimed to describe the whole batch.
    """
    if keys is None:
        raise RuntimeError("capture: post-update cache keys unavailable")
    if getattr(keys, "ndim", 0) != 4 or int(keys.shape[0]) != 1:
        raise CaptureContractError(
            "capture: cache layout contract violated - keys have shape "
            f"{tuple(getattr(keys, 'shape', ()))}, expected [1, n_kv_heads, n_keys, head_dim] "
            "(DynamicCache-like .layers[i].keys / .key_cache[i], batch size 1)."
        )
    return keys


class AttentionRowCapture:
    """Monkeypatching capture of per-(layer, head, query-row) attention reductions.

    Patches BOTH ``Qwen3Attention.forward`` (modeling_qwen3.py:247) and
    ``Qwen3Attention.forward_with_gist`` (:303) on every attention module of the
    model, because nn.Module forward hooks are bypassed on the gist path and
    ``forward_with_gist`` additionally throws its attention weights away
    (``attn_output, _ = attention_interface(...)``, :350).

    ``query_mode``:
      ``"decode"``          - record every row of a 1-token forward;
      ``"prefill_last_n"``  - record the last ``last_n`` rows of a multi-token
                              forward (the query segment) and every decode row.

    Query-row indexing is consistent across layers: rows are ordered by
    (forward id, local row) and a forward is detected as new when an already-seen
    layer arrives again.  ``emit_index`` is the index of the generated token whose
    logits that row produced: the LAST row of the first captured multi-token
    forward emits token 0, and the i-th (0-based) 1-token forward emits token
    ``i + 1``.

    SHAPE CONTRACT (asserted, not assumed).  This is a BATCH-SIZE-1 instrument:
    the reduction indexes ``hidden_states[0]`` and ``keys[0]``, so a batched
    harness would otherwise capture batch element 0 while every meta line
    claimed to describe the whole batch.  ``install`` refuses an
    ``expected_batch_size != 1`` up front, and every forward checks the real
    tensors (``hidden_states.ndim == 3``, leading dim 1) and the cache layout
    (``keys.ndim == 4`` = ``[B, H_kv, K, D]``, leading dim 1); a violation raises
    :class:`CaptureContractError`, which lands in ``capture_errors`` and leaves
    the row's capture EMPTY rather than half-right.
    """

    def __init__(
        self,
        key_map: KeyClassMap,
        query_mode: str = "decode",
        last_n: int = 64,
        layers: Optional[Sequence[int]] = None,
        *,
        qrow_chunk: int = 16,
        capture_gist_path: bool = False,
        recompute_tol: float = 1e-2,
        expected_batch_size: int = 1,
    ) -> None:
        if query_mode not in ("decode", "prefill_last_n"):
            raise ValueError(f"query_mode must be decode|prefill_last_n, got {query_mode!r}")
        if capture_gist_path:
            # Refused at construction rather than accepted and silently reduced to
            # an error counter: the gist-path reduction is NOT implemented (see
            # DEVIATIONS, 'forward_with_gist is patched with a COUNTING wrapper').
            raise NotImplementedError(
                "capture_gist_path=True is not implemented: forward_with_gist is patched with a "
                "counting wrapper only, so enabling it would produce an EMPTY capture on the gist "
                "path. Leave it False and read gist_path_forwards to see whether the path was hit."
            )
        self.expected_batch_size = int(expected_batch_size)
        self.key_map = key_map
        self.query_mode = query_mode
        self.last_n = int(last_n)
        self.layers = None if layers is None else sorted(int(x) for x in layers)
        self.qrow_chunk = max(1, int(qrow_chunk))
        self.capture_gist_path = bool(capture_gist_path)
        self.recompute_tol = float(recompute_tol)

        self.errors: Dict[str, int] = {}
        self.gist_path_forwards = 0
        self._patched: List[Tuple[Any, bool, bool]] = []
        self._installed = False
        self.reset()

    # -- lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        """Clear the per-row buffers (call before each battery row)."""
        self.recompute_max_abs_diff: Optional[float] = None
        self.recompute_max_abs_diff_fp32: Optional[float] = None
        self.recompute_n_checked = 0
        #: distinct shape-contract violation messages seen on this row (batch size
        #: / cache layout), so the meta line says WHY the capture is empty.
        self.contract_errors: List[str] = []
        self._blocks: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
        self._forward_rows: Dict[int, List[int]] = {}   # forward id -> captured qrows
        self._forward_nq: Dict[int, int] = {}           # forward id -> forward width
        self._forward_nkeys: Dict[int, int] = {}
        self._forward_id = -1
        self._seen_layers: set = set()
        self.gist_path_forwards = 0

    def _bump_error(self, kind: str) -> None:
        self.errors[kind] = self.errors.get(kind, 0) + 1

    @staticmethod
    def _iter_attention_modules(model: Any) -> List[Any]:
        layers = None
        for path in ("model.layers", "layers", "model.model.layers"):
            obj = model
            ok = True
            for part in path.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    ok = False
                    break
            if ok:
                layers = obj
                break
        if layers is None:
            raise RuntimeError("AttentionRowCapture: cannot locate the decoder layer list")
        mods = []
        for layer in layers:
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                raise RuntimeError("AttentionRowCapture: decoder layer has no .self_attn")
            mods.append(attn)
        return mods

    def install(self, model: Any) -> "AttentionRowCapture":
        if self._installed:
            raise RuntimeError("AttentionRowCapture already installed")
        if self.expected_batch_size != 1:
            raise RuntimeError(
                "AttentionRowCapture is a batch-size-1 instrument (it reduces hidden_states[0] "
                f"and keys[0]); expected_batch_size={self.expected_batch_size} would silently "
                "capture batch element 0 only. Drive the battery one row at a time."
            )
        impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
        if impl is not None and impl != "eager":
            raise RuntimeError(
                f"AttentionRowCapture needs attn_implementation='eager', got {impl!r}. "
                "The Ascend fused kernel (npu_fusion_attention) returns outputs only and "
                "exposes no attention probabilities; sdpa/flash do not return them either."
            )
        for idx, mod in enumerate(self._iter_attention_modules(model)):
            layer_idx = int(getattr(mod, "layer_idx", idx))
            if self.layers is not None and layer_idx not in self.layers:
                continue
            had_fwd = "forward" in getattr(mod, "__dict__", {})
            orig_fwd = mod.forward
            mod.forward = self._wrap_forward(mod, layer_idx, orig_fwd)
            had_gist = "forward_with_gist" in getattr(mod, "__dict__", {})
            orig_gist = getattr(mod, "forward_with_gist", None)
            if orig_gist is not None:
                mod.forward_with_gist = self._wrap_gist(mod, layer_idx, orig_gist)
            self._patched.append((mod, had_fwd, had_gist and orig_gist is not None))
        self._installed = True
        return self

    def remove(self) -> None:
        for mod, had_fwd, had_gist in self._patched:
            if not had_fwd:
                mod.__dict__.pop("forward", None)
            if not had_gist:
                mod.__dict__.pop("forward_with_gist", None)
        self._patched = []
        self._installed = False

    def __enter__(self) -> "AttentionRowCapture":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()

    # -- patched callables ---------------------------------------------------

    def _wrap_forward(self, mod: Any, layer_idx: int, orig: Callable[..., Any]) -> Callable[..., Any]:
        names = ("hidden_states", "position_embeddings", "attention_mask", "past_key_values")

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = dict(zip(names, args))
            bound.update({k: v for k, v in kwargs.items() if k in names})
            out = orig(*args, **kwargs)
            try:
                self._capture(mod, layer_idx, bound.get("hidden_states"),
                              bound.get("position_embeddings"),
                              bound.get("past_key_values"), kwargs, out)
            except Exception as exc:  # never kill a battery row over instrumentation
                self._bump_error(type(exc).__name__)
                if isinstance(exc, CaptureContractError) and str(exc) not in self.contract_errors:
                    self.contract_errors.append(str(exc))
            return out

        return wrapper

    def _wrap_gist(self, mod: Any, layer_idx: int, orig: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            self.gist_path_forwards += 1
            out = orig(*args, **kwargs)
            if self.capture_gist_path:
                self._bump_error("gist_path_capture_not_implemented")
            return out

        return wrapper

    # -- the streaming reduction --------------------------------------------

    def _new_forward(self, layer_idx: int, n_q: int) -> int:
        """A repeat of an already-seen layer index means a new forward pass began."""
        if self._forward_id < 0 or layer_idx in self._seen_layers:
            self._seen_layers = set()
            self._forward_id += 1
        self._seen_layers.add(layer_idx)
        self._forward_nq.setdefault(self._forward_id, int(n_q))
        return self._forward_id

    def _select_rows(self, n_q: int) -> Optional[np.ndarray]:
        if n_q == 1:
            return np.array([0], dtype=np.int64)
        if self.query_mode == "decode":
            return None
        n = min(self.last_n, n_q)
        return np.arange(n_q - n, n_q, dtype=np.int64)

    def _capture(self, mod: Any, layer_idx: int, hidden_states: Any,
                 position_embeddings: Any, past_key_values: Any,
                 kwargs: Dict[str, Any], returned: Any = None) -> None:
        # The shape contract is checked BEFORE torch is imported, so a batched
        # harness is refused identically on a box without torch (where the import
        # would otherwise raise first and hide the real violation behind an
        # ImportError counter).
        if hidden_states is None or position_embeddings is None:
            raise RuntimeError("capture: hidden_states / position_embeddings unavailable")
        if getattr(hidden_states, "ndim", 0) != 3 or int(hidden_states.shape[0]) != 1:
            raise CaptureContractError(
                "capture: batch-size-1 contract violated - hidden_states has shape "
                f"{tuple(getattr(hidden_states, 'shape', ()))}, expected [1, Q, hidden]. "
                "The reduction indexes element 0 only; a batched harness must be driven row by row."
            )

        import torch  # lazy: torch lives on the server only

        n_q = int(hidden_states.shape[1])
        fid = self._new_forward(layer_idx, n_q)
        rows = self._select_rows(n_q)
        if rows is None or rows.size == 0:
            return

        apply_rotary_pos_emb = _import_rope()

        head_dim = int(mod.head_dim)
        use_gist = bool(kwargs.get("use_gist", False))
        gist_param = (getattr(mod, "gist_param", "") or "").lower()
        q_proj = mod.gist_q_proj if (use_gist and "q" in gist_param) else mod.q_proj

        hs = hidden_states[:, rows, :]
        shape = (hs.shape[0], hs.shape[1], -1, head_dim)
        qs = mod.q_norm(q_proj(hs).view(shape)).transpose(1, 2)
        cos, sin = position_embeddings
        cos_s = cos[:, rows, :]
        sin_s = sin[:, rows, :]
        qs, _ = apply_rotary_pos_emb(qs, qs, cos_s, sin_s)

        keys = _cache_keys(past_key_values, int(getattr(mod, "layer_idx", layer_idx)))
        check_key_layout(keys)

        n_h = int(qs.shape[1])
        n_kv = int(keys.shape[1])
        n_rep = max(1, n_h // n_kv)
        n_keys = int(keys.shape[2])
        scale = float(mod.scaling)

        cid = self.key_map.class_ids(n_keys)
        did = self.key_map.doc_ids(n_keys)
        # absolute key position of each captured query row (causal cut)
        abs_pos = n_keys - n_q + rows

        # Fidelity self-check: on the eager path ``forward`` RETURNS the kernel's
        # own attn_weights, so the recomputed probabilities can be compared against
        # them for free (no extra memory - the tensor already exists).  If this
        # ever exceeds ``recompute_tol`` the whole capture is measuring something
        # other than 2407.07071's alpha and nothing downstream is interpretable.
        ref_weights = None
        if returned is not None:
            cand = returned[1] if isinstance(returned, (tuple, list)) and len(returned) > 1 else None
            if (cand is not None and getattr(cand, "ndim", 0) == 4
                    and int(cand.shape[-1]) == n_keys
                    and int(cand.shape[1]) == n_h):
                ref_weights = cand

        parts: List[Dict[str, np.ndarray]] = []
        k32 = keys[0].to(torch.float32)
        k32 = k32.repeat_interleave(n_rep, dim=0) if n_rep > 1 else k32
        for lo in range(0, rows.size, self.qrow_chunk):
            hi = min(rows.size, lo + self.qrow_chunk)
            q32 = qs[0, :, lo:hi, :].to(torch.float32)
            logits = torch.matmul(q32, k32.transpose(1, 2)) * scale
            cut = torch.as_tensor(abs_pos[lo:hi], device=logits.device)
            keep = torch.arange(n_keys, device=logits.device)[None, :] <= cut[:, None]
            logits = logits.masked_fill(~keep[None, :, :], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            if ref_weights is not None:
                sel = torch.as_tensor(rows[lo:hi], device=ref_weights.device)
                ref = ref_weights[0].index_select(1, sel).to(torch.float32).to(probs.device)
                # (i) fp32 recompute vs the kernel: informational.  The eager kernel
                # forms the logits in the MODEL dtype (bf16: ~3 significant digits,
                # so logits of magnitude 10-30 carry ~0.05-0.1 absolute rounding)
                # and rounds the fp32 softmax back to bf16, so this difference is
                # dominated by dtype, not by construction (measured 0.07 on the
                # NPU smoke with 4572/4572 rows over 1e-3).
                d32 = float((probs - ref).abs().max())
                if self.recompute_max_abs_diff_fp32 is None or d32 > self.recompute_max_abs_diff_fp32:
                    self.recompute_max_abs_diff_fp32 = d32
                # (ii) the ENFORCED check replicates the kernel's own dtype path
                # (eager_attention_forward: bf16 matmul * scaling, fp32 softmax,
                # cast back) on the same q/k, so what remains is accumulation-
                # order noise; a construction error (wrong projection, wrong
                # keys, wrong mask) still shows up as a large difference.
                qm = qs[0, :, lo:hi, :]
                km = keys[0]
                km = km.repeat_interleave(n_rep, dim=0) if n_rep > 1 else km
                lm = torch.matmul(qm, km.transpose(1, 2).to(qm.dtype)) * scale
                lm = lm.masked_fill(~keep[None, :, :], float("-inf"))
                pm = torch.softmax(lm, dim=-1, dtype=torch.float32).to(qm.dtype).to(torch.float32)
                pm = torch.nan_to_num(pm, nan=0.0)
                d = float((pm - ref).abs().max())
                self.recompute_n_checked += 1
                if self.recompute_max_abs_diff is None or d > self.recompute_max_abs_diff:
                    self.recompute_max_abs_diff = d
                if d > self.recompute_tol:
                    self._bump_error("recompute_mismatch")
                del ref, sel, qm, km, lm, pm
            parts.append(reduce_probs(probs.cpu().numpy(), cid, did, self.key_map.n_docs))
            del logits, probs, q32
        del k32

        block = {key: np.concatenate([p[key] for p in parts], axis=1) for key in parts[0]}
        self._blocks[(layer_idx, fid)] = block
        self._forward_rows.setdefault(fid, rows.tolist())
        self._forward_nkeys.setdefault(fid, int(n_keys))

    # -- accessors -----------------------------------------------------------

    @property
    def layer_ids(self) -> List[int]:
        return sorted({l for (l, _) in self._blocks})

    def _row_index(self) -> List[Tuple[int, int]]:
        """Global query-row order: (forward id, local qrow), forwards in id order."""
        out: List[Tuple[int, int]] = []
        for fid in sorted(self._forward_rows):
            for r in self._forward_rows[fid]:
                out.append((fid, int(r)))
        return out

    def _stack(self, field: str, dtype: Any) -> np.ndarray:
        layers = self.layer_ids
        index = self._row_index()
        if not layers or not index:
            return np.zeros((0, 0, 0), dtype=dtype)
        ref = self._blocks[(layers[0], index[0][0])][field]
        n_h = ref.shape[0]
        tail = tuple(ref.shape[2:])
        out = np.full((len(layers), n_h, len(index)) + tail, np.nan, dtype=np.float64)
        pos_of: Dict[Tuple[int, int], int] = {}
        for fid, rows in self._forward_rows.items():
            for pos, r in enumerate(rows):
                pos_of[(fid, int(r))] = pos
        for li, layer in enumerate(layers):
            for ri, (fid, r) in enumerate(index):
                block = self._blocks.get((layer, fid))
                if block is None:
                    continue
                out[li, :, ri, ...] = block[field][:, pos_of[(fid, r)], ...]
        return out.astype(dtype)

    def class_mass_tensor(self) -> np.ndarray:
        """``[L, H, Q, 5]`` float32."""
        return self._stack("class_mass", np.float32)

    def doc_mass_tensor(self) -> np.ndarray:
        """``[L, H, Q, D]`` float32."""
        return self._stack("doc_mass", np.float32)

    def argmax_valid_mask(self) -> np.ndarray:
        """``[L, H, Q]`` bool: True where that (layer, query row) was actually
        captured.  False entries carry the -1 sentinel in :meth:`argmax_tensor`."""
        return np.isfinite(self._stack("argmax_key", np.float64))

    def argmax_tensor(self) -> np.ndarray:
        """``[L, H, Q]`` int64 absolute key indices, with **-1** where the
        (layer, query row) was not captured.

        ``_stack`` fills missing entries with NaN (a layer that failed to capture
        on one forward); casting that straight to int64 would turn them into
        INT64_MIN, a silent sentinel that looks like a key index.  The validity
        mask is :meth:`argmax_valid_mask` - callers must not read a -1 as a key.
        """
        raw = self._stack("argmax_key", np.float64)
        valid = np.isfinite(raw)
        out = np.where(valid, raw, -1.0).astype(np.int64)
        out[~valid] = -1
        return out

    def sink_tensor(self) -> np.ndarray:
        return self._stack("sink_mass", np.float32)

    def entropy_tensor(self) -> np.ndarray:
        return self._stack("doc_entropy", np.float32)

    @property
    def records(self) -> List[Dict[str, Any]]:
        """Flat per-(layer, head, query-row) view.  O(L*H*Q) dicts - inspection and
        tests only; the driver uses the tensor accessors."""
        out: List[Dict[str, Any]] = []
        for (layer, fid), block in sorted(self._blocks.items()):
            rows = self._forward_rows.get(fid, [])
            for h in range(block["class_mass"].shape[0]):
                for pos, r in enumerate(rows):
                    out.append({
                        "layer": int(layer),
                        "head": int(h),
                        "forward_id": int(fid),
                        "qrow": int(r),
                        "class_mass": block["class_mass"][h, pos],
                        "doc_mass": block["doc_mass"][h, pos],
                        "argmax_key": int(block["argmax_key"][h, pos]),
                        "argmax_val": float(block["argmax_val"][h, pos]),
                        "doc_entropy": float(block["doc_entropy"][h, pos]),
                        "sink_mass": float(block["sink_mass"][h, pos]),
                    })
        return out

    @property
    def row_meta(self) -> List[Dict[str, Any]]:
        """One entry per captured query row, in the tensor accessors' row order.

        ``emit_index`` is the index of the GENERATED token whose logits that row
        produced: the last row of the first captured multi-token forward emits
        token 0, and the i-th (0-based) 1-token forward emits token ``i + 1``.
        Other prompt rows have ``emit_index = None`` and never enter a span mean.
        """
        out: List[Dict[str, Any]] = []
        decode_seen = 0
        prefill_seen = False
        for fid in sorted(self._forward_rows):
            n_q = self._forward_nq.get(fid, 1)
            rows = self._forward_rows[fid]
            for r in rows:
                if n_q == 1:
                    emit: Optional[int] = decode_seen + 1
                elif not prefill_seen and r == n_q - 1:
                    emit = 0
                else:
                    emit = None
                out.append({
                    "forward_id": int(fid),
                    "qrow": int(r),
                    "emit_index": emit,
                    "n_keys": int(self._forward_nkeys.get(fid, 0)),
                    "kind": "decode" if n_q == 1 else "prefill",
                })
            if n_q == 1:
                decode_seen += 1
            else:
                prefill_seen = True
        return out


def emit_index_consistency(
    row_meta: Sequence[Dict[str, Any]],
    n_generated: Optional[int] = None,
) -> Dict[str, Any]:
    """Check the ``emit_index`` convention against the forwards actually observed.

    The convention (last row of the first captured multi-token forward emits
    token 0; the i-th 1-token forward emits token i+1) is INFERRED from how
    ``_generate_with_prefix`` drives a prefix cache.  If ``generate()`` ever
    issues an extra warm-up forward, every ``<tool_call>`` span mean silently
    shifts by one token, so the assumption is verified from evidence the capture
    already carries rather than trusted:

    * the first captured forward must be the multi-token prefill;
    * the key count must advance by exactly one per 1-token forward (a warm-up
      or re-run forward breaks the +1 chain);
    * the emitted indices must be 0,1,2,... without gaps or repeats;
    * ``max(emit_index)`` must not run past the generated token count.

    Returns ``{"ok": bool, "problems": [...], ...}``; the result goes on every
    capture meta line, so a violated assumption is loud instead of a quiet
    one-token offset.
    """
    rows = list(row_meta)
    emits = [m["emit_index"] for m in rows if m.get("emit_index") is not None]
    problems: List[str] = []
    if not rows:
        problems.append("no captured query rows")
    elif rows[0].get("kind") != "prefill":
        problems.append("first captured forward is not a multi-token prefill")

    seen: Dict[int, int] = {}
    order: List[int] = []
    for m in rows:
        fid = int(m.get("forward_id", -1))
        if fid not in seen:
            seen[fid] = int(m.get("n_keys") or 0)
            order.append(fid)
    prev_kind: Dict[int, str] = {}
    for m in rows:
        prev_kind.setdefault(int(m.get("forward_id", -1)), str(m.get("kind")))
    for a, b in zip(order, order[1:]):
        if prev_kind.get(b) == "decode" and seen[b] - seen[a] != 1:
            problems.append(
                f"key count jumps {seen[a]} -> {seen[b]} into a 1-token forward "
                "(expected +1; an extra warm-up forward shifts every emit_index)")
            break
    if emits != list(range(len(emits))):
        problems.append(f"emit indices are not 0..n-1 (got {emits[:8]}...)")
    if n_generated is not None and emits and max(emits) > int(n_generated):
        problems.append(f"max emit_index {max(emits)} > n_generated {int(n_generated)}")
    return {
        "ok": not problems,
        "problems": problems,
        "n_emitting_rows": len(emits),
        "max_emit_index": (max(emits) if emits else None),
        "n_generated": (None if n_generated is None else int(n_generated)),
        "note": "the emit_index convention is inferred from _generate_with_prefix; "
                "this check is what makes a wrong inference loud",
    }


def _import_rope() -> Callable[..., Any]:
    """``modeling_qwen3.apply_rotary_pos_emb``; the harness puts <root>/python on
    sys.path, so both spellings are tried."""
    last: Optional[BaseException] = None
    for mod in ("models.qwen3.modeling_qwen3", "python.models.qwen3.modeling_qwen3"):
        try:
            return __import__(mod, fromlist=["apply_rotary_pos_emb"]).apply_rotary_pos_emb
        except ImportError as exc:  # pragma: no cover - server-side path
            last = exc
    raise RuntimeError(f"cannot import apply_rotary_pos_emb: {last}")


def _cache_keys(past_key_values: Any, layer_idx: int) -> Any:
    if past_key_values is None:
        return None
    layers = getattr(past_key_values, "layers", None)
    if layers is not None and layer_idx < len(layers):
        return getattr(layers[layer_idx], "keys", None)
    kc = getattr(past_key_values, "key_cache", None)
    if kc is not None and layer_idx < len(kc):
        return kc[layer_idx]
    return None


# ---------------------------------------------------------------------------
# (B) Lookback Lens  (2407.07071)
# ---------------------------------------------------------------------------

LOOKBACK_RATIOS: Tuple[str, ...] = ("lr_context", "lr_gist", "lr_gist_vs_raw")


def lookback_ratios(
    class_mass: np.ndarray,
    class_key_counts: Sequence[int],
) -> np.ndarray:
    """Lookback ratios per (layer, head, query row).  2407.07071 section 2, eq. for
    ``A^{l,h}_t(context)``, ``A^{l,h}_t(new)`` and ``LR^{l,h}_t``.

    ``class_mass``: ``[L, H, Q, 5]`` summed attention mass per key class.
    ``class_key_counts``: ``[5]`` (or ``[Q, 5]``) number of KEYS in each class, so
    each sum is normalised by ITS OWN length - that is what makes the ratio
    length-invariant (card, 'Signals used', property 1).

    Returns ``[L, H, Q, 3]`` in ``LOOKBACK_RATIOS`` order; ``nan`` where a ratio
    is undefined (no generated keys yet at t=1; no gist class on the full arm;
    no raw tail in the pure c2kv recipe).  Heads are never folded.
    """
    cm = np.asarray(class_mass, dtype=np.float64)
    cnt = np.asarray(class_key_counts, dtype=np.float64)
    if cnt.ndim == 1:
        cnt = cnt[None, :]
    if cnt.shape[-1] != N_CLASSES:
        raise ValueError("class_key_counts must have 5 columns")
    with np.errstate(divide="ignore", invalid="ignore"):
        # cnt is [1,5] (shared across rows) or [Q,5] (per query row: the
        # generated-key count grows with t, so the per-row form is the faithful
        # one and is what finalize_row passes).
        a = cm / np.where(cnt > 0, cnt, np.nan)[None, None, :, :]

    ctx_ids = [CLASS_INDEX[c] for c in
               ("system_raw", "history_gist", "history_raw_tail", "current_query")]
    gen = CLASS_INDEX["generated_so_far"]
    gist = CLASS_INDEX["history_gist"]
    raw = CLASS_INDEX["history_raw_tail"]

    def ratio(num: np.ndarray, den_other: np.ndarray) -> np.ndarray:
        tot = num + den_other
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.isfinite(tot) & (tot > 0), num / tot, np.nan)

    # 2407.07071 eq. (1): A(context) = (1/N) * sum over the WHOLE context, one
    # aggregate normalisation by the context's own length N - NOT the sum of the
    # four prompt classes' individual per-key means.  Summing per-class means
    # would make LR depend on how many prompt classes happen to be non-empty
    # (0.75 with no raw tail vs 0.80 with one, under a uniform attention
    # distribution), which would bias the S0 full-arm twin by class composition
    # alone.  The paper's `context` is one undifferentiated block; our four
    # prompt classes are its parts, so they are pooled before the division.
    ctx_mass = cm[..., ctx_ids].sum(axis=-1)                 # [L, H, Q]
    ctx_cnt = cnt[..., ctx_ids].sum(axis=-1)                 # [Q] (or [1])
    with np.errstate(divide="ignore", invalid="ignore"):
        a_ctx = np.where(ctx_cnt > 0,
                         ctx_mass / np.where(ctx_cnt > 0, ctx_cnt, 1.0),
                         np.nan)
    out = np.stack([
        ratio(a_ctx, a[..., gen]),
        ratio(a[..., gist], a[..., gen]),
        ratio(a[..., gist], a[..., raw]),
    ], axis=-1)
    return out.astype(np.float32)


def tool_call_span_rows(
    row_meta: Sequence[Dict[str, Any]],
    span_first: Optional[int],
    span_last: Optional[int],
) -> np.ndarray:
    """Indices into the captured query rows whose ``emit_index`` falls inside the
    ``<tool_call>`` payload span (2407.07071: v-bar is the mean of v_t over a span).
    Empty when the span is None - the caller then writes None, not 0.0."""
    if span_first is None or span_last is None:
        return np.zeros(0, dtype=np.int64)
    keep = [i for i, m in enumerate(row_meta)
            if m.get("emit_index") is not None and span_first <= m["emit_index"] <= span_last]
    return np.asarray(keep, dtype=np.int64)


def span_mean_lookback(lr: np.ndarray, rows: np.ndarray) -> Optional[np.ndarray]:
    """Mean of ``lr [L,H,Q,3]`` over the span rows -> ``[L,H,3]``; None when the
    span is empty (never a 0.0 fallback)."""
    if rows.size == 0:
        return None
    with np.errstate(invalid="ignore"):
        return np.nanmean(lr[:, :, rows, :], axis=2).astype(np.float32)


def lookback_feature_matrix(
    span_lr: Dict[str, Optional[np.ndarray]],
    qids: Sequence[str],
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Stack the per-qid ``[L,H,3]`` span means into ``X [n, L*H*3]``.

    Returns ``(X, feature_names, kept_qids)``.  qids whose span was empty are
    dropped and reported by the caller as a denominator, not imputed.
    """
    kept = [q for q in qids if span_lr.get(q) is not None]
    if not kept:
        return np.zeros((0, 0)), [], []
    ref = span_lr[kept[0]]
    n_l, n_h, _ = ref.shape
    names = [f"lr__l{l}__h{h}__{LOOKBACK_RATIOS[r]}"
             for l in range(n_l) for h in range(n_h) for r in range(3)]
    X = np.stack([np.asarray(span_lr[q], dtype=float).reshape(-1) for q in kept], axis=0)
    return X, names, kept


def make_topk_coef_selector(
    X_all: np.ndarray,
    y_all: np.ndarray,
    k: int = 100,
    C: float = 1.0,
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """2407.07071 section 5's top-k-by-|coef| head study, turned into a legal
    selector: the coefficients are re-fitted on the rows of the CURRENT training
    fold only (matched back to their labels by exact row identity), so nothing is
    chosen with the held-out labels.  See DEVIATIONS.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    Xc = np.ascontiguousarray(np.asarray(X_all, dtype=float))
    lut: Dict[bytes, int] = {}
    for i in range(Xc.shape[0]):
        key = Xc[i].tobytes()
        if key in lut:
            raise ValueError(
                "top-k selector: two feature rows are byte-identical, so a training row "
                "cannot be matched back to its own label unambiguously"
            )
        lut[key] = i

    def selector(X_tr: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Xt = np.ascontiguousarray(np.asarray(X_tr, dtype=float))
        idx = []
        for i in range(Xt.shape[0]):
            j = lut.get(Xt[i].tobytes())
            if j is None:
                raise RuntimeError("top-k selector: training row not found in X_all")
            idx.append(j)
        y = np.asarray(y_all, dtype=int)[np.asarray(idx, dtype=int)]
        if y.sum() in (0, len(y)):
            cols = np.arange(min(k, X_tr.shape[1]))
            return X_tr[:, cols], X_te[:, cols]
        sc = StandardScaler().fit(X_tr)
        clf = LogisticRegression(C=C, solver="liblinear", max_iter=2000).fit(sc.transform(X_tr), y)
        order = np.argsort(-np.abs(clf.coef_.ravel()))
        cols = np.sort(order[: min(k, X_tr.shape[1])])
        return X_tr[:, cols], X_te[:, cols]

    return selector


def layer_subset_selector(names: Sequence[str], lo: float, hi: float) -> Callable[..., Tuple[np.ndarray, np.ndarray]]:
    """Column selector keeping layers in the fractional depth band ``[lo, hi)``.
    2407.07071 section 5's layer ablation (middle layers best, late layers worst)
    becomes an inner-fold selector rather than a post-hoc table."""
    layers = np.array([int(n.split("__l")[1].split("__")[0]) for n in names])
    n_l = layers.max() + 1 if layers.size else 1
    keep = np.where((layers >= math.floor(lo * n_l)) & (layers < max(1, math.ceil(hi * n_l))))[0]
    if keep.size == 0:
        keep = np.arange(len(names))

    def selector(X_tr: np.ndarray, X_te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return X_tr[:, keep], X_te[:, keep]

    return selector


S8_CONTROL_COLUMNS: Tuple[str, ...] = (
    "actual_compression_ratio",
    "gist_tokens",
    "kept_history_tokens",
    "n_dropped_docs",
    "n_docs",
)


def drop_undefined_columns(X: np.ndarray, names: Sequence[str]) -> Tuple[np.ndarray, List[str], int]:
    """Drop feature columns that are non-finite for ANY row.

    LR_gist_vs_raw is undefined for every row of the pure c2kv recipe (no raw
    tail) and LR_gist for every row of the full arm; keeping those columns would
    make ``nested_cv_logistic`` drop all ROWS instead.  Nothing is imputed - the
    count of dropped columns is returned and reported.
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return X, list(names), 0
    keep = np.isfinite(X).all(axis=0)
    return X[:, keep], [n for n, k in zip(names, keep) if k], int((~keep).sum())


def lookback_probe(
    X: np.ndarray,
    names: Sequence[str],
    y: np.ndarray,
    groups: np.ndarray,
    *,
    s8: Optional[np.ndarray] = None,
    s8_names: Optional[Sequence[str]] = None,
    seed: int = 20260905,
) -> Dict[str, Any]:
    """Nested, session-grouped probe on the L*H*3 lookback vector, with the S8
    control block and the increment over it.

    The chance level printed is the EVALUATION FRAME's prevalence (0.578 on the
    161-row trigger subset), never the 900-frame base rate 0.1033.  When an S8
    block is supplied, BOTH arms are restricted to the rows finite in both, so
    the comparison is on the same row subset.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    X, names, n_dropped_cols = drop_undefined_columns(X, names)
    n_dropped_s8_cols = 0
    s8_kept: List[str] = []
    s8_dropped: List[str] = []
    if s8 is not None and np.asarray(s8).size:
        s8 = np.asarray(s8, dtype=float)
        # The control columns are named, so a control block running on four of its
        # five pre-registered columns says WHICH one is missing (n_dropped_docs
        # until unit U2's sidecar is wired in), not just how many.
        raw_names = list(s8_names) if s8_names is not None else list(S8_CONTROL_COLUMNS)
        if len(raw_names) != s8.shape[1]:
            raw_names = [f"s8_{i}" for i in range(s8.shape[1])]
        s8, s8_kept, n_dropped_s8_cols = drop_undefined_columns(s8, raw_names)
        s8_dropped = [n for n in raw_names if n not in set(s8_kept)]
        keep = np.isfinite(X).all(axis=1) & np.isfinite(s8).all(axis=1)
        X, s8, y, groups = X[keep], s8[keep], y[keep], groups[keep]
    else:
        s8 = None

    selectors: Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray]]] = {
        "all_heads": lambda a, b: (a, b),
        "top100_abs_coef": make_topk_coef_selector(X, y, k=100),
        "layers_early": layer_subset_selector(names, 0.0, 1 / 3),
        "layers_middle": layer_subset_selector(names, 1 / 3, 2 / 3),
        "layers_late": layer_subset_selector(names, 2 / 3, 1.0),
    }
    attn = nested_cv_logistic(X, y, groups, extra_selectors=selectors, seed=seed)
    out: Dict[str, Any] = {
        "attention": _probe_report(attn),
        "n_features": int(X.shape[1]),
        "n_dropped_undefined_columns": n_dropped_cols,
        "n_dropped_undefined_s8_columns": n_dropped_s8_cols,
        "s8_columns_used": s8_kept,
        "s8_columns_dropped_undefined": s8_dropped,
    }
    if s8 is not None and s8.size:
        ctrl = nested_cv_logistic(s8, y, groups, seed=seed)
        joint = nested_cv_logistic(np.hstack([s8, X]), y, groups,
                                   extra_selectors={"all": lambda a, b: (a, b)}, seed=seed)
        out["s8_control"] = _probe_report(ctrl)
        out["s8_plus_attention"] = _probe_report(joint)
        both = attn["scored_mask"] & ctrl["scored_mask"]
        if both.any():
            d, lo, hi = paired_delta_bootstrap(
                average_precision, attn["oof_scores"][both], ctrl["oof_scores"][both],
                attn["labels"][both],
                session_clusters([str(g) for g in attn["groups"][both]]))
            out["increment_over_s8"] = {
                "delta_auprc": d, "ci95": [lo, hi],
                "n": int(both.sum()), "n_pos": int(attn["labels"][both].sum()),
                "note": "same row subset for both arms; CI lower bound > 0 is the digest 4.0 gate",
            }
    return out


def _probe_report(res: Dict[str, Any]) -> Dict[str, Any]:
    ok = res["scored_mask"]
    s, y = res["oof_scores"][ok], res["labels"][ok]
    cl = session_clusters([str(g) for g in np.asarray(res["groups"])[ok]])
    ap_lo, ap_hi, n_cl = clustered_bootstrap(average_precision, s, y, cl)
    au_lo, au_hi, _ = clustered_bootstrap(auroc, s, y, cl)
    # Pre-declared fire rates (not tuned): the three deployment metrics need an
    # operating point, and 2407.07071 supplies none (card, 'Decision rule').
    ops = {f"fire_rate_{q:.2f}": operating_point(s, y, int(round(q * len(s))))
           for q in (0.10, 0.20, 0.30)}
    return {
        "n": int(ok.sum()), "n_pos": int(y.sum()),
        "prevalence_chance_ap": prevalence(y),
        "auprc": res["auprc"], "auprc_ci95": [ap_lo, ap_hi],
        "auroc": res["auroc"], "auroc_ci95": [au_lo, au_hi],
        "n_clusters": n_cl, "n_dropped_nan": res["n_dropped_nan"],
        "operating_points": ops,
        "chosen": res["chosen"],
    }


# ---------------------------------------------------------------------------
# (C) Elastic-Cache drift features  (2510.14973)
# ---------------------------------------------------------------------------

def score_doc_vector(doc_mass: np.ndarray, span_lens: Sequence[int], mode: str) -> np.ndarray:
    """Per-doc score vector under the declared variants, mirroring
    ``eval_agent_history_c2kv._score_history_attention`` (:2800):
    ``sum`` = summed mass, ``sqrt_len`` = mass / sqrt(span_len),
    ``mean`` = mass / span_len.  Zero-length spans give 0.0."""
    if mode not in SCORE_MODES:
        raise ValueError(f"score mode must be one of {SCORE_MODES}, got {mode!r}")
    v = np.asarray(doc_mass, dtype=np.float64)
    L = np.asarray(span_lens, dtype=np.float64)
    if mode == "sum":
        return v.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.sqrt(L) if mode == "sqrt_len" else L
        out = np.where(L > 0, v / np.where(d > 0, d, 1.0), 0.0)
    return out.astype(np.float32)


def doc_vector_stats(vec: np.ndarray) -> Dict[str, Optional[float]]:
    """attn_entropy_over_docs and attn_top1_top2_margin_docs for one per-doc vector."""
    v = np.asarray(vec, dtype=np.float64)
    tot = float(np.nansum(v))
    if not np.isfinite(tot) or tot <= 0 or v.size == 0:
        return {"entropy": None, "margin": None, "top1": None, "argmax": None}
    p = v / tot
    ent = float(-np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)))
    order = np.argsort(-v)
    top1 = float(v[order[0]])
    top2 = float(v[order[1]]) if v.size > 1 else 0.0
    return {"entropy": ent, "margin": float(top1 - top2), "top1": top1,
            "argmax": int(order[0])}


def align_doc_vectors(
    prev: np.ndarray,
    cur: np.ndarray,
    prev_shas: Optional[Sequence[str]] = None,
    cur_shas: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Align two adjacent decision steps' per-doc vectors.

    POLICY (pre-registered): when both steps carry per-doc sha256 (from the
    decoded-docs sidecar / the frozen witness table), align on the shas that
    occur in BOTH, in the current step's block order - a decision step appends
    history, so the previous step's blocks are a prefix by content, not by index.
    Without shas, fall back to block index truncated to ``min(n_docs)``.  The
    policy actually used is returned so the caller can report it.
    """
    prev = np.asarray(prev, dtype=np.float64)
    cur = np.asarray(cur, dtype=np.float64)
    if prev_shas and cur_shas:
        pos_prev = {s: i for i, s in enumerate(prev_shas)}
        pairs = [(pos_prev[s], j) for j, s in enumerate(cur_shas) if s in pos_prev]
        if pairs:
            pi = np.array([p for p, _ in pairs], dtype=int)
            ci = np.array([c for _, c in pairs], dtype=int)
            return prev[pi], cur[ci], "sha256"
        # Both steps carry hashes and share no block: there is nothing to align.
        # Falling back to block index here would compare unrelated blocks under a
        # name that claims a content join, so the row is reported as unalignable
        # (the caller then writes None, never 0.0).
        return prev[:0], cur[:0], "sha256_no_overlap"
    n = min(prev.size, cur.size)
    return prev[:n], cur[:n], "block_index"


def drift_features(
    cur_vec: np.ndarray,
    prev_vec: Optional[np.ndarray],
    *,
    prev_shas: Optional[Sequence[str]] = None,
    cur_shas: Optional[Sequence[str]] = None,
) -> Dict[str, Optional[float]]:
    """``attn_cos_prev_step`` (the transplanted sigma of 2510.14973 section 3.3) and
    ``attn_drop_maxdoc``.  Both are None when the row has no predecessor decision
    step in the frame - never 0.0, never imputed."""
    if prev_vec is None:
        return {"attn_cos_prev_step": None, "attn_drop_maxdoc": None,
                "align_policy": None, "n_aligned_docs": 0}
    p, c, policy = align_doc_vectors(prev_vec, cur_vec, prev_shas, cur_shas)
    if p.size == 0:
        return {"attn_cos_prev_step": None, "attn_drop_maxdoc": None,
                "align_policy": policy, "n_aligned_docs": 0}
    np_, nc = float(np.linalg.norm(p)), float(np.linalg.norm(c))
    cos = float(np.dot(p, c) / (np_ * nc)) if np_ > 0 and nc > 0 else None
    k = int(np.argmax(p))
    drop = float(1.0 - (c[k] / p[k])) if p[k] > 0 else None
    return {"attn_cos_prev_step": cos, "attn_drop_maxdoc": drop,
            "align_policy": policy, "n_aligned_docs": int(p.size)}


def predecessor_map(qids: Sequence[str]) -> Dict[str, Optional[str]]:
    """qid -> the qid of the immediately preceding decision step of the SAME
    session that is also IN THIS FRAME, or None.

    Ordering is ``step_index(qid)`` (the qid suffix) - never ``decision_step``
    (= doc_chunks + 1), which is a length, not an order.
    """
    by_session: Dict[str, List[str]] = {}
    for q in qids:
        by_session.setdefault(session_of(q), []).append(q)
    out: Dict[str, Optional[str]] = {}
    for sess, qs in by_session.items():
        qs = sorted(qs, key=step_index)
        prev: Optional[str] = None
        for q in qs:
            out[q] = prev
            prev = q
    return out


def predecessor_denominator(qids: Sequence[str], y: Optional[Sequence[int]] = None) -> Dict[str, int]:
    """The cross-step denominator, counted BEFORE any number is promised
    (digest 4.5: 'this denominator has to be counted first, and both denominators
    reported, never multiplied')."""
    pm = predecessor_map(qids)
    have = [q for q in qids if pm.get(q) is not None]
    out = {"n_rows": len(qids), "n_with_predecessor": len(have)}
    if y is not None:
        ymap = {q: int(v) for q, v in zip(qids, y)}
        out["n_pos_rows"] = int(sum(ymap.values()))
        out["n_pos_with_predecessor"] = int(sum(ymap[q] for q in have))
    return out


#: Pre-declared layer bands (fractional depth).  2407.07071 section 5's layer
#: ablation motivates the split; which band wins is decided in inner folds.
LAYER_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("all", 0.0, 1.0), ("early", 0.0, 1 / 3), ("middle", 1 / 3, 2 / 3), ("late", 2 / 3, 1.0),
)


def band_gist_frac(class_mass: Optional[np.ndarray], lo: float, hi: float) -> float:
    """``attn_gist_frac`` restricted to a fractional layer band - the first of the
    five features digest 4.5 asks the Elastic-Cache block to emit.  ``nan`` (never
    0.0) when the row carries no class mass."""
    if class_mass is None or np.asarray(class_mass).size == 0:
        return float("nan")
    cm = np.asarray(class_mass, dtype=np.float64)
    n_l = cm.shape[0]
    blk = cm[slice(math.floor(lo * n_l), max(1, math.ceil(hi * n_l)))]
    tot = np.nansum(blk, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(tot > 0,
                        blk[..., CLASS_INDEX["history_gist"]] / np.where(tot > 0, tot, 1.0),
                        np.nan)
    v = _nanmean_or_none(frac)
    return float("nan") if v is None else float(v)


def cap_control_matrix(
    metas: Dict[str, Dict[str, Any]],
    qids: Sequence[str],
) -> Tuple[np.ndarray, List[str], int]:
    """The cap-flip control block digest 4.5 pre-registers for this method:
    ``generated_tokens`` and ``finish_reason == 'length'``.

    The 128-token cap is a known confound of the r2 trigger set (digest 1.3), so
    the drift features have to be credited only with the increment over it, the
    same way the Lookback block is credited only over S8.  Columns that are
    undefined for any row are dropped and counted, never imputed.
    """
    rows: List[List[float]] = []
    for q in qids:
        r = (metas.get(q) or {}).get("row") or {}
        gt = r.get("generated_tokens")
        fr = r.get("finish_reason")
        rows.append([
            float(gt) if isinstance(gt, (int, float)) and not isinstance(gt, bool) else float("nan"),
            float("nan") if fr is None else float(fr == "length"),
        ])
    X = np.asarray(rows, dtype=float).reshape(len(list(qids)), 2)
    return drop_undefined_columns(X, ["ctl_generated_tokens", "ctl_cap_hit"])


def drift_feature_matrix(
    arrays: Dict[str, Dict[str, np.ndarray]],
    metas: Dict[str, Dict[str, Any]],
    qids: Sequence[str],
    doc_shas: Optional[Dict[str, List[str]]] = None,
) -> Tuple[np.ndarray, List[str], List[str], Dict[str, np.ndarray]]:
    """Design matrix for the Elastic-Cache block over (layer band x score mode).

    The digest requires the layer range AND the score mode to be chosen in inner
    folds, which a flat scalar jsonl cannot express; this builds the full
    ``4 bands x 3 modes x 4 statistics`` block, plus one mode-independent
    ``gist_frac`` column per band (digest 4.5 lists ``attn_gist_frac`` as the
    first of the five features), plus the column groups a caller hands to
    ``nested_cv_logistic(extra_selectors=...)``.

    ``doc_shas`` (qid -> per-block sha256) activates the pre-registered sha256
    alignment policy across decision steps; without it the fallback is block
    index truncated to ``min(n_docs)``, exactly as in ``extract_features``.

    Returns ``(X, names, kept_qids, selector_groups)``; rows whose predecessor is
    absent carry ``nan`` in the two cross-step columns and are handled by the
    caller's own denominator report - they are never imputed here.
    """
    kept = [q for q in qids if arrays.get(q, {}).get("doc_mass") is not None]
    pm = predecessor_map(kept)
    per: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {}
    for q in kept:
        dm = arrays[q]["doc_mass"]
        lens = metas[q].get("doc_span_lens") or []
        n_l = dm.shape[0]
        per[q] = {}
        for band, lo, hi in LAYER_BANDS:
            sl = slice(math.floor(lo * n_l), max(1, math.ceil(hi * n_l)))
            pooled = np.nanmean(dm[sl], axis=(0, 1))
            for mode in SCORE_MODES:
                per[q][(band, mode)] = score_doc_vector(pooled, lens, mode)

    names: List[str] = []
    cols: List[List[float]] = []
    groups: Dict[str, List[int]] = {}
    for band, lo, hi in LAYER_BANDS:
        for mode in SCORE_MODES:
            base = len(names)
            names += [f"{s}__{band}__{mode}" for s in
                      ("entropy", "margin", "cos_prev_step", "drop_maxdoc")]
            groups.setdefault(f"band_{band}", []).extend(range(base, base + 4))
            groups.setdefault(f"mode_{mode}", []).extend(range(base, base + 4))
            vals: List[List[float]] = []
            for q in kept:
                v = per[q][(band, mode)]
                st = doc_vector_stats(v)
                prev_qid = pm.get(q)
                prev = per.get(prev_qid or "", {}).get((band, mode))
                d = drift_features(
                    v, prev,
                    prev_shas=(doc_shas or {}).get(prev_qid) if prev_qid else None,
                    cur_shas=(doc_shas or {}).get(q))
                vals.append([
                    st["entropy"] if st["entropy"] is not None else np.nan,
                    st["margin"] if st["margin"] is not None else np.nan,
                    d["attn_cos_prev_step"] if d["attn_cos_prev_step"] is not None else np.nan,
                    d["attn_drop_maxdoc"] if d["attn_drop_maxdoc"] is not None else np.nan,
                ])
            arr = np.asarray(vals, dtype=float)
            cols.extend(arr.T.tolist())
        # attn_gist_frac is mode-independent, so it is emitted once per band
        # rather than duplicated across the three score modes.
        gf = [band_gist_frac(arrays.get(q, {}).get("class_mass"), lo, hi) for q in kept]
        if np.isfinite(np.asarray(gf, dtype=float)).all() and kept:
            groups.setdefault(f"band_{band}", []).append(len(names))
            names.append(f"gist_frac__{band}")
            cols.append(gf)
    X = np.asarray(cols, dtype=float).T if cols else np.zeros((len(kept), 0))
    sel = {"all": np.arange(len(names))}
    sel.update({k: np.asarray(sorted(set(v)), dtype=int) for k, v in groups.items()})
    return X, names, kept, sel


def column_group_selectors(groups: Dict[str, np.ndarray]) -> Dict[str, Callable[..., Tuple[np.ndarray, np.ndarray]]]:
    """Turn column-index groups into ``nested_cv_logistic`` inner-fold selectors."""
    def make(idx: np.ndarray) -> Callable[..., Tuple[np.ndarray, np.ndarray]]:
        def sel(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            return a[:, idx], b[:, idx]
        return sel
    return {name: make(idx) for name, idx in groups.items() if len(idx)}


def bf16_noise_floor_check(
    a: Dict[str, np.ndarray],
    b: Dict[str, np.ndarray],
    floor: float = BF16_NOISE_FLOOR,
) -> Dict[str, Any]:
    """Compare two identical forwards' per-doc vectors against the documented
    NPU bf16 rounding floor (0.0078125).  A cosine-drift threshold that sits
    below this floor is measuring shape-dependent rounding, not drift
    (digest 4.5 risk paragraph)."""
    common = sorted(set(a) & set(b))
    worst = 0.0
    worst_qid: Optional[str] = None
    per: Dict[str, float] = {}
    for q in common:
        va, vb = np.asarray(a[q], dtype=np.float64), np.asarray(b[q], dtype=np.float64)
        n = min(va.size, vb.size)
        d = float(np.max(np.abs(va.ravel()[:n] - vb.ravel()[:n]))) if n else 0.0
        per[q] = d
        if d > worst:
            worst, worst_qid = d, q
    return {
        "n_compared": len(common),
        "n_only_a": len(set(a) - set(b)),
        "n_only_b": len(set(b) - set(a)),
        "max_abs_diff": worst,
        "worst_qid": worst_qid,
        "floor": float(floor),
        "within_floor": bool(worst <= floor),
        "n_above_floor": int(sum(1 for v in per.values() if v > floor)),
    }


# ---------------------------------------------------------------------------
# (D) Retrieval Head ports 1 / 2  (2404.15574)
# ---------------------------------------------------------------------------

def load_head_set(path: Path) -> Dict[str, Any]:
    """Read unit U1b's port-0 output.

    Schema (fixed here so U1b and U1a agree)::

        {"model": str, "checkpoint": str, "threshold": float,
         "n_heads_total": int, "heads": [[layer, head], ...],
         "detection": {...provenance...}}

    Port 0 selects the heads on RAW needle contexts (token identity is required
    by 2404.15574 section 2 and is undefined on gist spans); this module only
    consumes the list.
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    heads = [(int(l), int(h)) for l, h in obj["heads"]]
    if not heads:
        raise ValueError(f"{path}: empty head set")
    return {"heads": heads, "meta": {k: v for k, v in obj.items() if k != "heads"}}


def head_mask(n_layers: int, n_heads: int, heads: Sequence[Tuple[int, int]]) -> np.ndarray:
    m = np.zeros((n_layers, n_heads), dtype=bool)
    for l, h in heads:
        if 0 <= l < n_layers and 0 <= h < n_heads:
            m[l, h] = True
    if not m.any():
        raise ValueError("head set does not intersect this model's (layer, head) grid")
    return m


def retrieval_head_doc_scores(
    doc_mass: np.ndarray,
    span_lens: Sequence[int],
    mask: np.ndarray,
    mode: str = "sum",
) -> np.ndarray:
    """Port 1: restrict the per-doc aggregation to the retrieval-head subset.

    ``doc_mass``: ``[L, H, D]`` span-mean per-doc mass.  Returns ``[D]``, the mean
    over the selected heads of the per-head score under ``mode`` (the three score
    modes of ``_rank_history_by_attention`` are kept as declared variants).
    """
    dm = np.asarray(doc_mass, dtype=np.float64)
    if dm.ndim != 3:
        raise ValueError("doc_mass must be [L, H, D]")
    sel = dm[mask[: dm.shape[0], : dm.shape[1]]]
    if sel.size == 0:
        return np.zeros(dm.shape[2], dtype=np.float32)
    scored = np.stack([score_doc_vector(v, span_lens, mode) for v in sel], axis=0)
    return np.nanmean(scored, axis=0).astype(np.float32)


def retrieval_head_trigger_scalars(
    class_mass: np.ndarray,
    sink_mass: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Optional[float]]:
    """Port 2: the two free scalars of section 4.1's mechanism hypothesis
    ('during hallucinated generation the retrieval heads predominantly attend to
    the initial token of the input, the attention sink').

    ``sink_ratio``   - mass on key 0 (the sink proper) over total, by retrieval heads;
    ``sink_ratio_system`` - the same with the whole system-prefix class;
    ``gist_ratio``   - gist mass / (gist + raw tail), None when there is no raw tail.
    """
    cm = np.asarray(class_mass, dtype=np.float64)
    sm = np.asarray(sink_mass, dtype=np.float64)
    m = mask[: cm.shape[0], : cm.shape[1]]
    sel = cm[m]
    sel_sink = sm[m] if sm.ndim == 2 else None
    if sel.size == 0:
        return {"attn_rh_sink_ratio": None, "attn_rh_sink_ratio_system": None,
                "attn_rh_gist_ratio": None}
    tot = sel.sum(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sys_ratio = np.where(tot > 0, sel[:, CLASS_INDEX["system_raw"]] / tot, np.nan)
    gist = sel[:, CLASS_INDEX["history_gist"]]
    raw = sel[:, CLASS_INDEX["history_raw_tail"]]
    den = gist + raw
    with np.errstate(divide="ignore", invalid="ignore"):
        gr = np.where(den > 0, gist / den, np.nan)
    return {
        "attn_rh_sink_ratio": _nanmean_or_none(sel_sink) if sel_sink is not None else None,
        "attn_rh_sink_ratio_system": _nanmean_or_none(sys_ratio),
        "attn_rh_gist_ratio": _nanmean_or_none(gr),
    }


def _nanmean_or_none(a: Optional[np.ndarray]) -> Optional[float]:
    if a is None:
        return None
    arr = np.asarray(a, dtype=np.float64)
    if arr.size == 0 or not np.isfinite(arr).any():
        return None
    with np.errstate(invalid="ignore"):
        return float(np.nanmean(arr))


def legacy_locator_hits(
    truth: Dict[str, Optional[int]],
    n_docs: Dict[str, int],
    witness: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Optional[bool]]]:
    """The frozen legacy comparison arms: ``k_first`` = block 0, ``k_last`` = the
    last block, ``k_median`` = the witness table's own ``k_median`` column."""
    out: Dict[str, Dict[str, Optional[bool]]] = {"k_first": {}, "k_median": {}, "k_last": {}}
    for qid, ref in truth.items():
        n = int(n_docs.get(qid, 0))
        ent = witness.get(qid) or {}
        km = ent.get("k_median")
        out["k_first"][qid] = None if ref is None else bool(ref == 0)
        out["k_last"][qid] = None if (ref is None or n <= 0) else bool(ref == n - 1)
        out["k_median"][qid] = None if (ref is None or km is None) else bool(int(km) == ref)
    return out


def flip_table_truth(flips_for_qid: Dict[int, bool]) -> Tuple[Optional[int], str]:
    """Reference block for one qid from the D-line per-(qid, k) flip table.

    POLICY (declared before any locator number is reported): a qid that flips at
    more than one k has no unique reference block and ABSTAINS - ``locate_table``
    counts an abstention in the denominator as a miss and reports it separately.
    Resolving it to ``min(k)`` would bias the table toward the strong first-block
    prior, which is one of the paired comparison arms (``k_first``).

    Returns ``(k, status)`` with status in ``{"unique", "multi", "none"}``.
    """
    ok = sorted(int(k) for k, v in (flips_for_qid or {}).items() if v)
    if len(ok) == 1:
        return ok[0], "unique"
    if len(ok) > 1:
        return None, "multi"
    return None, "none"


def locator_truth(
    qids: Sequence[str],
    flips: Dict[str, Dict[int, bool]],
    witness: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Optional[int]], Dict[str, Any]]:
    """Reference block per qid, from ONE source for the whole table.

    POLICY.  When a D-line flip table is supplied it is the truth source for
    EVERY qid: a qid the table does not cover ABSTAINS (None) and is counted,
    it does NOT quietly fall back to the frozen witness k* while the report's
    ``truth_source`` says ``flip_table``.  Mixing a repair-outcome truth with a
    gold-scored one inside one S@k denominator would make the number a blend of
    two definitions.  Without a flip table the frozen witness k* is the source
    for every qid (also reported).

    Multi-flip qids abstain via :func:`flip_table_truth`.  The second return
    value is the provenance block that goes into the report.
    """
    qs = sorted(set(str(q) for q in qids))
    truth: Dict[str, Optional[int]] = {}
    n_multi = n_none = n_missing = 0
    if flips:
        for q in qs:
            per_k = flips.get(q)
            if per_k is None:
                truth[q] = None
                n_missing += 1
                continue
            truth[q], status = flip_table_truth(per_k)
            n_multi += int(status == "multi")
            n_none += int(status == "none")
        source = "flip_table"
    else:
        for q in qs:
            truth[q] = (witness.get(q) or {}).get("k_witness")
        source = "witness_k_star"
    return truth, {
        "truth_source": source,
        "n_qids": len(qs),
        "n_with_reference": int(sum(1 for v in truth.values() if v is not None)),
        "n_multi_flip": n_multi,
        "n_no_flip": n_none,
        "n_missing_from_flip_table": n_missing,
        "policy": ("one truth source for the whole table: with a flip table, a qid that flips at "
                   "more than one k and a qid the table does not cover both abstain (counted in "
                   "the S@k denominator as misses and reported here); neither is resolved to "
                   "min(k) nor silently replaced by the gold witness k*"),
    }


def retrieval_head_locator_report(
    score_vectors: Dict[str, Sequence[float]],
    truth: Dict[str, Optional[int]],
    n_docs: Dict[str, int],
    witness: Dict[str, Dict[str, Any]],
    *,
    label: str = "retrieval_head_argmax",
) -> Dict[str, Any]:
    """S@k table + inverted-score control + paired McNemar against k_first /
    k_median / k_last, all on the SAME qid set."""
    ctrl = inverted_score_control(score_vectors, truth)
    # The chooser is the FROZEN select_k_star semantics (t34_common.chooser_argmax:
    # abstain when nothing scores above zero, ties to the lowest index) - the same
    # callable the inverted control's forward arm uses, so the locator table and
    # ctrl["forward"] cannot disagree about which qids were even answered.  A bare
    # np.argmax would answer rows the control abstains on.
    ours: Dict[str, Optional[bool]] = {}
    for q, v in score_vectors.items():
        ref = truth.get(q)
        k = chooser_argmax(v)
        ours[q] = None if (ref is None or k is None) else bool(k == ref)
    legacy = legacy_locator_hits({q: truth.get(q) for q in score_vectors}, n_docs, witness)
    comparisons = {}
    for name, hits in legacy.items():
        b = sum(1 for q in ours if ours[q] is True and hits.get(q) is not True)
        c = sum(1 for q in ours if hits.get(q) is True and ours[q] is not True)
        comparisons[name] = {
            "table": locate_table(hits, label=name),
            "mcnemar_p": mcnemar_exact(b, c), "b": b, "c": c,
        }
    return {
        "locator": locate_table(ours, label=label),
        "inverted_control": ctrl,
        "legacy": comparisons,
        "n_qids": len(score_vectors),
    }


# ---------------------------------------------------------------------------
# (E) on-disk capture schema + feature extraction
# ---------------------------------------------------------------------------

NPZ_FIELDS: Tuple[str, ...] = ("class_mass", "doc_mass", "lookback", "sink_mass", "doc_entropy")


def _npz_key(qid: str, field: str) -> str:
    return f"{qid.replace(':', '__')}||{field}"


def finalize_row(
    capture: "AttentionRowCapture",
    key_map: KeyClassMap,
    *,
    qid: str,
    arm: str,
    generated_ids: Sequence[int],
    decode_fn: Callable[[Sequence[int]], str],
    prefix_meta: Dict[str, Any],
    row_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Reduce one battery row's capture to the on-disk record.

    The lookback ratio is averaged over the ``<tool_call>`` span AS A RATIO
    (LR is non-linear, so the span mean of LR is not LR of the span means);
    class / doc masses are stored as span means so the artifact stays ~120 KB
    per row instead of ~10 MB.
    """
    spans = spans_from_generation(decode_fn, list(generated_ids))
    cm = capture.class_mass_tensor()
    dm = capture.doc_mass_tensor()
    sk = capture.sink_tensor()
    en = capture.entropy_tensor()
    meta_rows = capture.row_meta
    # per query row: the generated-key count grows with t, so 1/N and 1/(t-1) are
    # taken from THAT row's own key partition (2407.07071's length invariance).
    counts = np.array(
        [[int((key_map.class_ids(m["n_keys"]) == c).sum()) for c in range(N_CLASSES)]
         for m in meta_rows],
        dtype=np.int64,
    ) if meta_rows else np.zeros((0, N_CLASSES), dtype=np.int64)
    lr = lookback_ratios(cm, counts) if cm.size and counts.size else np.zeros(
        (0, 0, 0, 3), dtype=np.float32)
    rows = tool_call_span_rows(meta_rows, spans.get("payload_first"), spans.get("payload_last"))
    span_lr = span_mean_lookback(lr, rows) if lr.size else None

    def span_mean(t: np.ndarray) -> Optional[np.ndarray]:
        if t.size == 0 or rows.size == 0:
            return None
        with np.errstate(invalid="ignore"):
            return np.nanmean(t[:, :, rows, ...], axis=2).astype(np.float32)

    arrays = {
        "class_mass": span_mean(cm),
        "doc_mass": span_mean(dm),
        "sink_mass": span_mean(sk),
        "doc_entropy": span_mean(en),
        "lookback": span_lr,
    }
    meta = {
        "qid": qid,
        "arm": arm,
        "n_layers": int(cm.shape[0]) if cm.size else 0,
        "n_heads": int(cm.shape[1]) if cm.size else 0,
        "n_docs": int(key_map.n_docs),
        "doc_span_lens": key_map.doc_span_lens().tolist(),
        "class_key_counts_last_row": counts[-1].tolist() if counts.size else [],
        "n_span_adjustments": int(key_map.n_span_adjustments),
        "n_query_rows": len(meta_rows),
        "n_span_rows": int(rows.size),
        "span_first": spans.get("payload_first"),
        "span_last": spans.get("payload_last"),
        "has_tool_call": bool(spans.get("has_tool_call")),
        "parse_ok": bool(spans.get("parse_ok")),
        "closed": bool(spans.get("closed")),
        "n_generated": int(spans.get("n_generated") or 0),
        # The emit_index convention is INFERRED from how _generate_with_prefix
        # drives a prefix cache; this check is what makes a wrong inference loud
        # (an extra warm-up forward would shift every <tool_call> span mean by one
        # token and silently change every LR number).  It rides on every meta line
        # so no capture can be read without seeing whether it held.
        "emit_index_check": emit_index_consistency(meta_rows, spans.get("n_generated")),
        "gist_path_forwards": int(capture.gist_path_forwards),
        "recompute_max_abs_diff": (None if getattr(capture, "recompute_max_abs_diff", None) is None
                                   else float(capture.recompute_max_abs_diff)),
        "recompute_max_abs_diff_fp32": (None if getattr(capture, "recompute_max_abs_diff_fp32", None) is None
                                   else float(capture.recompute_max_abs_diff_fp32)),
        "recompute_n_checked": int(getattr(capture, "recompute_n_checked", 0) or 0),
        "capture_errors": dict(capture.errors),
        "query_rows": meta_rows,
        "prefix": {k: prefix_meta.get(k) for k in (
            "system_length", "history_length", "doc_chunks", "gist_tokens",
            "kept_history_tokens", "actual_compression_ratio", "doc_tokens",
            "dropped_docs")},
    }
    if row_meta:
        meta["row"] = {k: row_meta.get(k) for k in (
            "generated_tokens", "finish_reason", "session_id", "dropped_docs")}
    return {k: v for k, v in arrays.items() if v is not None}, meta


class AttentionCaptureWriter:
    """``results/t34/attn_<arm>.npz`` + ``attn_<arm>.jsonl`` (one meta line per qid)."""

    def __init__(self, out_dir: Path, arm: str) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.arm = arm
        self._arrays: Dict[str, np.ndarray] = {}
        self._meta_fh = io.open(self.out_dir / f"attn_{arm}.jsonl", "a", encoding="utf-8")

    def add(self, qid: str, arrays: Dict[str, np.ndarray], meta: Dict[str, Any]) -> None:
        for field, value in arrays.items():
            self._arrays[_npz_key(qid, field)] = np.asarray(value, dtype=np.float32)
        self._meta_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        self._meta_fh.flush()

    def close(self) -> None:
        np.savez_compressed(self.out_dir / f"attn_{self.arm}.npz", **self._arrays)
        self._meta_fh.close()


def load_capture(npz_path: Path, meta_path: Path) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, Any]]]:
    """Read a capture back.  Returns ``qid -> {field: array}`` and ``qid -> meta``."""
    metas: Dict[str, Dict[str, Any]] = {}
    for line in Path(meta_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            m = json.loads(line)
            metas[m["qid"]] = m
    arrays: Dict[str, Dict[str, np.ndarray]] = {q: {} for q in metas}
    with np.load(npz_path) as z:
        lut = {_npz_key(q, ""): q for q in metas}
        for key in z.files:
            stem, field = key.split("||", 1)
            qid = lut.get(stem + "||")
            if qid is None:
                continue
            arrays[qid][field] = z[key]
    return arrays, metas


def dropped_doc_counts_from_sidecar(path: Optional[str]) -> Optional[Dict[str, int]]:
    """``qid -> number of history blocks dropped by the tail window``, read from
    unit U2's sidecar (``agent/t34_dump_sidecar.py``).  None when no sidecar is
    given - the caller then writes ``n_dropped_docs`` as null, never as 0.

    SIDECAR SHAPE, stated because getting it wrong is silent: ``docs`` holds ONLY
    the KEPT blocks (the decoded grid rows the model actually saw, in order),
    while ``dropped_docs`` holds indices into the POST-SPLIT history list, NOT
    into ``docs``.  The count is therefore ``len(dropped_docs)`` and nothing here
    filters ``docs`` by those indices; the dropped blocks' text exists only under
    the optional ``dropped_doc_texts`` key and is not needed for this count.
    """
    if not path:
        return None
    out: Dict[str, int] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        dd = r.get("dropped_docs")
        if isinstance(dd, list):
            out[str(r["qid"])] = len(dd)
    return out


def extract_features(
    arrays: Dict[str, Dict[str, np.ndarray]],
    metas: Dict[str, Dict[str, Any]],
    *,
    arm: str,
    doc_shas: Optional[Dict[str, List[str]]] = None,
    head_mask_arr: Optional[np.ndarray] = None,
    dropped_counts: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Per-qid scalar feature rows (hard rule 3: scalars, ``None`` when undefined,
    never a sentinel and never a whole-output fallback under a span-specific name).

    Only the compressed arm's own capture, its own battery row and the sidecar are
    read; no target / gold / tool_name_match / full-arm field is touched.

    On ``arm='full'`` (the S0 twin) the ``history_gist`` CLASS holds raw history
    KV, so the ``attn_gist_frac`` column is a HISTORY-SLOT fraction there, not a
    gist fraction; the genuinely
    gist-specific columns (``lr_gist_mean``, ``lr_gist_vs_raw_mean``,
    ``attn_gist_ratio``, ``attn_rh_gist_ratio``, ``gist_tokens``,
    ``actual_compression_ratio``) are written as null.
    """
    qids = sorted(metas)
    pm = predecessor_map(qids)
    per_doc: Dict[str, Dict[str, np.ndarray]] = {}
    for qid in qids:
        dm = arrays.get(qid, {}).get("doc_mass")
        if dm is None or dm.size == 0:
            continue
        lens = metas[qid].get("doc_span_lens") or []
        pooled = np.nanmean(dm, axis=(0, 1))
        per_doc[qid] = {m: score_doc_vector(pooled, lens, m) for m in SCORE_MODES}

    rows: List[Dict[str, Any]] = []
    for qid in qids:
        meta = metas[qid]
        arr = arrays.get(qid, {})
        row: Dict[str, Any] = {
            "qid": qid, "arm": arm,
            "session_id": (meta.get("row") or {}).get("session_id") or session_of(qid),
        }
        lr = arr.get("lookback")
        for i, name in enumerate(LOOKBACK_RATIOS):
            row[f"{name}_mean"] = (_nanmean_or_none(lr[..., i]) if lr is not None and lr.size else None)

        cm = arr.get("class_mass")
        if cm is not None and cm.size:
            tot = np.nansum(cm, axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(tot[..., None] > 0, cm / tot[..., None], np.nan)
            for cls, col in CLASS_FRAC_COLUMNS.items():
                row[col] = _nanmean_or_none(frac[..., CLASS_INDEX[cls]])
            row["attn_sink_ratio_system"] = _nanmean_or_none(frac[..., CLASS_INDEX["system_raw"]])
            g = cm[..., CLASS_INDEX["history_gist"]]
            r = cm[..., CLASS_INDEX["history_raw_tail"]]
            den = g + r
            with np.errstate(divide="ignore", invalid="ignore"):
                row["attn_gist_ratio"] = _nanmean_or_none(np.where(den > 0, g / den, np.nan))
        else:
            for col in CLASS_FRAC_COLUMNS.values():
                row[col] = None
            row["attn_sink_ratio_system"] = None
            row["attn_gist_ratio"] = None
        row["attn_sink_ratio"] = _nanmean_or_none(arr.get("sink_mass"))

        for mode in SCORE_MODES:
            vec = per_doc.get(qid, {}).get(mode)
            st = doc_vector_stats(vec) if vec is not None else {"entropy": None, "margin": None}
            row[f"attn_entropy_over_docs_{mode}"] = st["entropy"]
            row[f"attn_top1_top2_margin_docs_{mode}"] = st["margin"]
            prev_qid = pm.get(qid)
            prev_vec = per_doc.get(prev_qid, {}).get(mode) if prev_qid else None
            d = drift_features(
                vec if vec is not None else np.zeros(0), prev_vec,
                prev_shas=(doc_shas or {}).get(prev_qid) if prev_qid else None,
                cur_shas=(doc_shas or {}).get(qid))
            row[f"attn_cos_prev_step_{mode}"] = d["attn_cos_prev_step"]
            row[f"attn_drop_maxdoc_{mode}"] = d["attn_drop_maxdoc"]
        row["has_prev_step"] = int(pm.get(qid) is not None)

        if head_mask_arr is not None and cm is not None and cm.size:
            row.update(retrieval_head_trigger_scalars(cm, arr.get("sink_mass"), head_mask_arr))
        else:
            row.update({"attn_rh_sink_ratio": None, "attn_rh_sink_ratio_system": None,
                        "attn_rh_gist_ratio": None})

        prefix = meta.get("prefix") or {}
        battery = meta.get("row") or {}
        row["actual_compression_ratio"] = prefix.get("actual_compression_ratio")
        row["gist_tokens"] = prefix.get("gist_tokens")
        row["kept_history_tokens"] = prefix.get("kept_history_tokens")
        row["n_docs"] = meta.get("n_docs")
        # n_dropped_docs: the battery generation row does not carry dropped_docs,
        # so the declared source is unit U2's sidecar (--docs-sidecar).  Order:
        # battery row -> prefix bookkeeping -> sidecar; null (never 0) when none
        # of the three has it, and the CLI reports how many rows are null.
        dropped = battery.get("dropped_docs")
        if dropped is None:
            dropped = prefix.get("dropped_docs")
        if isinstance(dropped, list):
            n_drop: Optional[int] = len(dropped)
        elif isinstance(dropped, int) and not isinstance(dropped, bool):
            n_drop = int(dropped)
        else:
            n_drop = None
        if n_drop is None and dropped_counts is not None:
            v = dropped_counts.get(qid)
            n_drop = None if v is None else int(v)
        row["n_dropped_docs"] = n_drop
        row["generated_tokens"] = battery.get("generated_tokens")
        fr = battery.get("finish_reason")
        row["cap_hit"] = None if fr is None else int(fr == "length")
        row["n_span_rows"] = meta.get("n_span_rows")

        if arm == "full":
            # S0 twin: LR_context is defined on the full arm, the gist-specific
            # quantities are NOT (the history slot holds raw KV there).  They are
            # written as null, never as 0.0 and never as a whole-output fallback.
            for col in ("lr_gist_mean", "lr_gist_vs_raw_mean", "attn_gist_ratio",
                        "attn_rh_gist_ratio", "gist_tokens", "actual_compression_ratio"):
                row[col] = None
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# CLIs
# ---------------------------------------------------------------------------

def _head_mask_for(metas: Dict[str, Dict[str, Any]], heads: Sequence[Tuple[int, int]]) -> np.ndarray:
    n_l = max((int(m.get("n_layers") or 0) for m in metas.values()), default=0)
    n_h = max((int(m.get("n_heads") or 0) for m in metas.values()), default=0)
    if n_l <= 0 or n_h <= 0:
        raise ValueError("capture meta carries no (n_layers, n_heads); cannot map a head set")
    return head_mask(n_l, n_h, heads)


def _doc_shas_from_sidecar(path: Optional[str]) -> Optional[Dict[str, List[str]]]:
    """qid -> per-block sha256 of the decoded doc text, for the pre-registered
    sha256 cross-step alignment policy.  None (-> block-index fallback) when no
    sidecar is given."""
    if not path:
        return None
    import hashlib
    docs = load_decoded_docs(Path(path))
    return {q: [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in ts]
            for q, ts in docs.items()}


def _cmd_extract_features(args: argparse.Namespace) -> int:
    arrays, metas = load_capture(Path(args.capture), Path(args.meta))
    doc_shas = _doc_shas_from_sidecar(args.docs_sidecar)
    hm = None
    if args.head_set:
        hs = load_head_set(Path(args.head_set))
        hm = _head_mask_for(metas, hs["heads"])
    rows = extract_features(arrays, metas, arm=args.arm, doc_shas=doc_shas, head_mask_arr=hm,
                            dropped_counts=dropped_doc_counts_from_sidecar(args.docs_sidecar))
    n = write_features_jsonl(Path(args.out), rows, context=f"t34_attention/{args.arm}")
    qids = [r["qid"] for r in rows]
    # Loud missing-input report: every S8 control column that is null for some row
    # is named here, so the S8 block can never be read as complete when it is not.
    s8_null = {c: int(sum(1 for r in rows if r.get(c) is None)) for c in S8_CONTROL_COLUMNS}
    incomplete = sorted(c for c, k in s8_null.items() if k)
    out = {"written": n, "out": str(args.out), "arm": args.arm,
           "predecessor_denominator": predecessor_denominator(qids),
           "s8_null_counts": s8_null,
           "s8_control_incomplete_columns": incomplete,
           "docs_sidecar": args.docs_sidecar,
           "flags": (["S8_CONTROL_INCOMPLETE"] if incomplete else [])}
    print(json.dumps(out, ensure_ascii=False))
    if incomplete:
        print("WARNING: S8_CONTROL_INCOMPLETE - null for some rows: "
              + ", ".join(f"{c} ({s8_null[c]}/{len(rows)})" for c in incomplete)
              + ". n_dropped_docs comes from unit U2's sidecar (--docs-sidecar "
                "results/t34/sidecar_<arm>.jsonl); the control block runs on the "
                "columns that survive drop_undefined_columns and that count is "
                "reported next to every metric.", file=sys.stderr)
    return 0


def _cmd_probe(args: argparse.Namespace) -> int:
    arrays, metas = load_capture(Path(args.capture), Path(args.meta))
    frame = FrozenAssets(Path(args.root)).load()
    subset = {r["qid"]: int(r["label_cw"]) for r in frame.trigger_subset()}
    span_lr = {q: arrays.get(q, {}).get("lookback") for q in metas if q in subset}
    qids = sorted(q for q in span_lr if span_lr[q] is not None)
    X, names, kept = lookback_feature_matrix(span_lr, qids)
    if X.size == 0:
        print(json.dumps({"error": "no rows with a <tool_call> span"}))
        return 1
    y = np.array([subset[q] for q in kept], dtype=int)
    groups = np.array([session_of(q) for q in kept])
    s8 = None
    if args.features:
        feats = {json.loads(l)["qid"]: json.loads(l)
                 for l in Path(args.features).read_text(encoding="utf-8").splitlines() if l.strip()}
        def _f(q: str, c: str) -> float:
            v = feats.get(q, {}).get(c)
            return float(v) if isinstance(v, (int, float)) else float("nan")
        s8 = np.array([[_f(q, c) for c in S8_CONTROL_COLUMNS] for q in kept], dtype=float)
    rep = lookback_probe(X, names, y, groups, s8=s8, s8_names=S8_CONTROL_COLUMNS)
    rep["frame"] = {"n": len(kept), "n_pos": int(y.sum()),
                    "chance_ap": float(y.mean()),
                    "n_dropped_no_span": len(qids) - len(kept),
                    "note": "chance AP is this frame's prevalence, not 0.1033"}
    rep["deviations"] = DEVIATIONS
    sha = freeze_json(Path(args.out), rep)
    print(json.dumps({"out": str(args.out), "sha256": sha,
                      "auprc": rep["attention"]["auprc"],
                      "chance_ap": rep["frame"]["chance_ap"],
                      "s8_columns_used": rep.get("s8_columns_used"),
                      "s8_columns_dropped_undefined": rep.get("s8_columns_dropped_undefined")},
                     ensure_ascii=False))
    if rep.get("s8_columns_dropped_undefined"):
        print("WARNING: S8_CONTROL_INCOMPLETE - the control block ran without "
              + ", ".join(rep["s8_columns_dropped_undefined"])
              + " (undefined for at least one row). The increment over S8 is an "
                "increment over the columns actually present.", file=sys.stderr)
    return 0


def increment_over_control(
    X: np.ndarray,
    X_control: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    min_rows: int = 8,
    seed: int = 20260905,
) -> Optional[Dict[str, Any]]:
    """Delta-AUPRC of a feature block over a control block on the SAME row subset.

    Both arms are restricted to the rows finite in BOTH matrices before either is
    fitted, so the difference cannot be an artefact of one arm being scored on an
    easier subset.  The CI is a session-clustered paired bootstrap; digest 4.0's
    winner rule is that its lower bound must exceed 0.  Returns None (never a 0.0)
    when the shared subset is too small or single-class to support a fit.
    """
    X = np.asarray(X, dtype=float)
    Xc = np.asarray(X_control, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if X.size == 0 or Xc.size == 0:
        return None
    keep = np.isfinite(X).all(axis=1) & np.isfinite(Xc).all(axis=1)
    n_keep = int(keep.sum())
    if n_keep < min_rows or int(y[keep].sum()) in (0, n_keep):
        return None
    a = nested_cv_logistic(X[keep], y[keep], groups[keep], seed=seed)
    b = nested_cv_logistic(Xc[keep], y[keep], groups[keep], seed=seed)
    both = a["scored_mask"] & b["scored_mask"]
    if not both.any() or int(a["labels"][both].sum()) in (0, int(both.sum())):
        return None
    d, lo_ci, hi_ci = paired_delta_bootstrap(
        average_precision, a["oof_scores"][both], b["oof_scores"][both],
        a["labels"][both],
        session_clusters([str(x) for x in a["groups"][both]]))
    return {
        "delta_auprc": d, "ci95": [lo_ci, hi_ci],
        "n": int(both.sum()), "n_pos": int(a["labels"][both].sum()),
        "n_rows_shared": n_keep,
        "note": "same row subset for both arms; CI lower bound > 0 is the digest 4.0 gate",
    }


def _cmd_drift_probe(args: argparse.Namespace) -> int:
    arrays, metas = load_capture(Path(args.capture), Path(args.meta))
    frame = FrozenAssets(Path(args.root)).load()
    subset = {r["qid"]: int(r["label_cw"]) for r in frame.trigger_subset()}
    qids = sorted(q for q in metas if q in subset)
    doc_shas = _doc_shas_from_sidecar(getattr(args, "docs_sidecar", None))
    X, names, kept, groups = drift_feature_matrix(arrays, metas, qids, doc_shas)
    if X.size == 0:
        print(json.dumps({"error": "no rows with a per-doc mass vector"}))
        return 1
    y = np.array([subset[q] for q in kept], dtype=int)
    g = np.array([session_of(q) for q in kept])
    den = predecessor_denominator(kept, y)

    # Pre-registered cap-flip control (digest 4.5): generated_tokens and
    # finish_reason == "length".  The drift block is credited only with the
    # increment over it, on the SAME row subset - the 128-token cap is a known
    # confound of the r2 trigger set.
    Xc, ctl_names, n_dropped_ctl = cap_control_matrix(metas, kept)

    # Two arms, never multiplied: the full block (cross-step columns force the
    # predecessor subset) and the within-step-only block on ALL rows.
    within = np.asarray([i for i, n in enumerate(names)
                         if n.startswith(("entropy", "margin", "gist_frac"))], dtype=int)
    res_all = nested_cv_logistic(X, y, g, extra_selectors=column_group_selectors(groups))
    res_within = nested_cv_logistic(X[:, within], y, g,
                                    extra_selectors=column_group_selectors(
                                        {"all": np.arange(within.size)}))
    rep = {
        "cross_step_block": _probe_report(res_all),
        "within_step_block": _probe_report(res_within),
        "predecessor_denominator": den,
        "n_features": int(X.shape[1]),
        "bands": [b[0] for b in LAYER_BANDS],
        "score_modes": list(SCORE_MODES),
        "doc_align_policy": "sha256" if doc_shas else "block_index",
        "note": ("the two denominators are reported side by side and never multiplied; "
                 "the cross-step arm is defined only on rows with a predecessor step "
                 "IN THIS FRAME"),
        "deviations": [d for d in DEVIATIONS if d["paper"].startswith("2510")],
    }
    rep["cap_control_columns"] = ctl_names
    rep["n_dropped_undefined_cap_columns"] = n_dropped_ctl
    if Xc.size:
        rep["cap_control_block"] = _probe_report(nested_cv_logistic(Xc, y, g))
        for arm_name, cols in (("cross_step", np.arange(X.shape[1])), ("within_step", within)):
            rep[f"increment_over_cap_control__{arm_name}"] = increment_over_control(
                X[:, cols], Xc, y, g)
    else:
        rep["cap_control_block"] = None
    sha = freeze_json(Path(args.out), rep)
    print(json.dumps({"out": str(args.out), "sha256": sha,
                      "cross_step_n": rep["cross_step_block"]["n"],
                      "within_step_n": rep["within_step_block"]["n"],
                      "predecessor_denominator": den}, ensure_ascii=False))
    return 0


def _cmd_locate(args: argparse.Namespace) -> int:
    arrays, metas = load_capture(Path(args.capture), Path(args.meta))
    frame = FrozenAssets(Path(args.root)).load()
    hs = load_head_set(Path(args.head_set))
    hm = _head_mask_for(metas, hs["heads"])
    flips = load_flip_table(Path(args.flip_table)) if args.flip_table else {}
    witness = (frame.witness or {}).get("entries") or {}

    n_docs: Dict[str, int] = {}
    vectors: Dict[str, Sequence[float]] = {}
    n_no_capture = 0
    for qid in frame.cw_qids():
        arr = arrays.get(qid, {})
        dm = arr.get("doc_mass")
        if dm is None or dm.size == 0:
            n_no_capture += 1
            continue
        lens = metas[qid].get("doc_span_lens") or []
        vectors[qid] = retrieval_head_doc_scores(dm, lens, hm, mode=args.score_mode).tolist()
        n_docs[qid] = int(metas[qid].get("n_docs") or 0)
    # ONE truth source for the whole table (see locator_truth): with a flip table
    # a qid it does not cover abstains, it does not silently revert to the gold
    # witness k*.
    truth, truth_report = locator_truth(list(vectors), flips, witness)
    rep = retrieval_head_locator_report(vectors, truth, n_docs, witness,
                                        label=f"retrieval_head_argmax_{args.score_mode}")
    rep["head_set"] = hs["meta"]
    rep["n_heads_selected"] = int(hm.sum())
    rep["truth_source"] = truth_report["truth_source"]
    rep["truth_provenance"] = truth_report
    rep["n_qids_without_capture"] = n_no_capture
    rep["deviations"] = [d for d in DEVIATIONS if d["paper"].startswith("2404")]
    rep["flip_table_available"] = bool(args.flip_table)
    if not args.flip_table:
        # DATA AVAILABILITY, stated loudly rather than absorbed: the per-(qid, k)
        # flip table lives on the NPU (~/bench_results/d_v2/).  Without it the
        # reference block is the frozen GOLD-SCORED witness k*, which answers a
        # different question than "which block, when repaired, flips the row".
        print("WARNING: no --flip-table; the reference block is the frozen witness k* "
              "(gold-scored), not the D-line repair-flip block. Copy "
              "~/bench_results/d_v2/ from the NPU before reporting a locator number.",
              file=sys.stderr)
    sha = freeze_json(Path(args.out), rep)
    print(json.dumps({"out": str(args.out), "sha256": sha,
                      "s_at_k": rep["locator"]["s_at_k"], "n": rep["locator"]["n"],
                      "truth_source": rep["truth_source"],
                      "n_with_reference": truth_report["n_with_reference"],
                      "n_multi_flip": truth_report["n_multi_flip"],
                      "n_missing_from_flip_table": truth_report["n_missing_from_flip_table"]},
                     ensure_ascii=False))
    return 0


def _cmd_noise_floor(args: argparse.Namespace) -> int:
    def load_docs(npz: Path) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        with np.load(npz) as z:
            for key in z.files:
                stem, field = key.split("||", 1)
                if field == "doc_mass":
                    out[stem] = z[key]
        return out

    rep = bf16_noise_floor_check(load_docs(Path(args.a)), load_docs(Path(args.b)),
                                 floor=args.floor)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0 if rep["n_compared"] else 1


def _cmd_run_battery(args: argparse.Namespace) -> int:
    """SERVER.  Mirrors t33_hidden_topup.build_args / main so the rows are the
    same assembly the frozen battery used, with AttentionRowCapture installed."""
    try:
        import torch  # noqa: F401
        from eval_agent_history_c2kv import (  # type: ignore
            _build_c2kv_prefix, _build_full_or_truncate_prefix, _clear_device_cache,
            _current_messages, _chat_template_ids, _generate_with_prefix, _is_oom_error,
            _load_examples, _load_tokenizer, _resolve_model_checkpoint,
        )
        from eval_agent_tool_definition_c2kv import _load_model, _setup_device  # type: ignore
        from t33_hidden_topup import build_args  # type: ignore
        from t33_labels import build_label_frame, join_arms, load_jsonl  # type: ignore
    except ImportError as exc:
        print(f"run-battery needs torch/transformers (server side): {exc}", file=sys.stderr)
        return 2

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    frame = build_label_frame(
        join_arms(load_jsonl(args.battery_full), load_jsonl(args.battery_c2kv)), manifest)
    subset = [r["qid"] for r in frame if r["label_cw"] in (0, 1)]
    if args.max_rows:
        subset = subset[: args.max_rows]
    else:
        # Memory is UNTESTED on the server: the peak float32 buffer is
        # [H, qrow_chunk, n_keys] per layer and the digest records eager attention
        # OOM-ing at 16K (37.17 GiB / 60.96 GiB).
        print("WARNING: --max_rows 0 runs the whole subset; eager attention has not been "
              "memory-profiled on this box. Do a --max_rows 2 pass first and read the peak "
              "before committing to a full run.", file=sys.stderr)

    eval_args = build_args(argparse.Namespace(
        arm=args.arm, model_path=args.model_path, base_model=args.base_model,
        tokenizer_path=args.tokenizer_path, dataset_path=args.dataset_path,
        ratio=args.ratio, attn_impl=args.attn_impl))
    device = _setup_device(args.device_type)
    eval_args.model = _resolve_model_checkpoint(eval_args.model)
    tokenizer = _load_tokenizer(eval_args)
    model = _load_model(eval_args, tokenizer, device)

    writer = AttentionCaptureWriter(Path(args.out_dir), args.arm)
    eval_args.qid_allowlist = set(subset)   # start-up cost: only the frozen rows
    examples = {e.qid: e for e in _load_examples(eval_args, tokenizer)[0] if e.qid in set(subset)}
    n_emit_bad = 0
    n_capture_errors = 0
    for i, qid in enumerate(subset):
        example = examples.get(qid)
        if example is None:
            continue
        try:
            if args.arm == "full":
                prefix, skip = _build_full_or_truncate_prefix(model, tokenizer, example, eval_args, "full")
            else:
                prefix, skip = _build_c2kv_prefix(model, tokenizer, example, eval_args)
            if prefix is None:
                continue
            query_ids = _chat_template_ids(tokenizer, _current_messages(example),
                                           add_generation_prompt=True)
            doc_lengths = list(prefix.get("doc_token_lengths") or [])
            if not doc_lengths:
                n = int(prefix.get("doc_chunks") or 0)
                doc_lengths = [max(1, int(prefix.get("doc_tokens") or n) // max(1, n))] * n
            key_map = KeyClassMap.from_prefix(
                arm=args.arm, system_length=int(prefix["system_length"]),
                doc_lengths=doc_lengths, gist_tokens=int(prefix.get("gist_tokens") or 0),
                raw_tail_tokens=int(prefix.get("top_full_tokens") or 0),
                query_len=len(query_ids), history_length=int(prefix.get("history_length") or 0))
            cap = AttentionRowCapture(key_map, query_mode=args.query_mode,
                                      last_n=args.last_n, qrow_chunk=args.qrow_chunk)
            cap.install(model)
            try:
                row = _generate_with_prefix(model, tokenizer, example, prefix, eval_args, args.arm)
            finally:
                cap.remove()
            gen_ids = row.get("generated_ids") or []
            ids_source = "harness_generated_ids"
            if not gen_ids:
                # the harness row carries the decoded text only; re-tokenise it
                # (decode->encode is not guaranteed to reproduce the emitted ids,
                # so the source is stamped and the count is cross-checked against
                # the harness's own generated_tokens)
                gen_ids = list(tokenizer.encode(row.get("prediction") or "", add_special_tokens=False))
                ids_source = "retokenized_prediction"
            n_gen_harness = row.get("generated_tokens")
            arrays, meta = finalize_row(
                cap, key_map, qid=qid, arm=args.arm, generated_ids=gen_ids,
                decode_fn=lambda ids: tokenizer.decode(list(ids), skip_special_tokens=True),
                prefix_meta=prefix, row_meta=row)
            meta["generated_ids_source"] = ids_source
            meta["n_generated_harness"] = None if n_gen_harness is None else int(n_gen_harness)
            meta["n_generated_retokenized_matches_harness"] = (
                None if n_gen_harness is None else bool(len(gen_ids) == int(n_gen_harness)))
            writer.add(qid, arrays, meta)
            if not meta["emit_index_check"]["ok"]:
                n_emit_bad += 1
            if meta["capture_errors"]:
                n_capture_errors += 1
            print(f"[{i + 1}/{len(subset)}] {qid} rows={meta['n_query_rows']} "
                  f"span={meta['n_span_rows']} gist_path={meta['gist_path_forwards']} "
                  f"emit_ok={meta['emit_index_check']['ok']} "
                  f"recompute_max_abs_diff={meta['recompute_max_abs_diff']} "
                  f"errors={meta['capture_errors'] or '{}'}")
        except RuntimeError as error:
            if _is_oom_error(error):
                _clear_device_cache(device)
                continue
            raise
        _clear_device_cache(device)
    writer.close()
    if n_emit_bad or n_capture_errors:
        print(f"WARNING: {n_emit_bad} row(s) failed the emit_index consistency check and "
              f"{n_capture_errors} row(s) recorded capture errors; read emit_index_check / "
              "capture_errors on those meta lines BEFORE reading any number off this capture.",
              file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="t34_attention", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract-features", help="LOCAL: capture -> per-qid scalar features jsonl")
    e.add_argument("--capture", required=True)
    e.add_argument("--meta", required=True)
    e.add_argument("--arm", required=True, choices=["full", "c2kv"])
    e.add_argument("--out", required=True)
    e.add_argument("--docs-sidecar", default=None,
                   help="results/t34/sidecar_<arm>.jsonl (unit U2); enables sha-based cross-step "
                        "doc alignment AND supplies n_dropped_docs, the fifth S8 control column, "
                        "which the battery generation row does not carry")
    e.add_argument("--head-set", default=None, help="U1b port-0 retrieval head json")
    e.set_defaults(fn=_cmd_extract_features)

    pr = sub.add_parser("probe", help="LOCAL: Lookback-Lens probe + S8 control block")
    pr.add_argument("--capture", required=True)
    pr.add_argument("--meta", required=True)
    pr.add_argument("--root", default=".")
    pr.add_argument("--features", default=None, help="features jsonl carrying the S8 scalars")
    pr.add_argument("--out", required=True)
    pr.set_defaults(fn=_cmd_probe)

    dp = sub.add_parser("drift-probe",
                        help="LOCAL: Elastic-Cache block, layer band x score mode in inner folds")
    dp.add_argument("--capture", required=True)
    dp.add_argument("--meta", required=True)
    dp.add_argument("--root", default=".")
    dp.add_argument("--docs-sidecar", default=None,
                   help="results/t34/sidecar_<arm>.jsonl; enables the sha256 cross-step "
                        "doc-alignment policy instead of the block-index fallback")
    dp.add_argument("--out", required=True)
    dp.set_defaults(fn=_cmd_drift_probe)

    lo = sub.add_parser("locate", help="LOCAL: Retrieval-Head port 1 locator table")
    lo.add_argument("--capture", required=True)
    lo.add_argument("--meta", required=True)
    lo.add_argument("--head-set", required=True)
    lo.add_argument("--root", default=".")
    lo.add_argument("--flip-table", default=None)
    lo.add_argument("--score-mode", default="sum", choices=list(SCORE_MODES))
    lo.add_argument("--out", required=True)
    lo.set_defaults(fn=_cmd_locate)

    nf = sub.add_parser("noise-floor", help="LOCAL: bf16 noise-floor check of two identical captures")
    nf.add_argument("--a", required=True)
    nf.add_argument("--b", required=True)
    nf.add_argument("--floor", type=float, default=BF16_NOISE_FLOOR)
    nf.set_defaults(fn=_cmd_noise_floor)

    rb = sub.add_parser("run-battery", help="SERVER: rerun the 161-row subset with the capture installed")
    rb.add_argument("--arm", required=True, choices=["full", "c2kv"])
    rb.add_argument("--model_path", required=True)
    rb.add_argument("--base_model", required=True)
    rb.add_argument("--tokenizer_path", required=True)
    rb.add_argument("--dataset_path", required=True)
    rb.add_argument("--battery_full", required=True)
    rb.add_argument("--battery_c2kv", required=True)
    rb.add_argument("--manifest", required=True)
    rb.add_argument("--out_dir", default="results/t34")
    rb.add_argument("--ratio", type=int, default=8)
    rb.add_argument("--attn_impl", default="eager")
    rb.add_argument("--device_type", default="npu")
    rb.add_argument("--query_mode", default="prefill_last_n", choices=["decode", "prefill_last_n"],
                    help="prefill_last_n + --last_n 1 (default) captures the prompt forward last "
                         "row, which emits generated token 0; plain decode skips it and the "
                         "emit_index convention check fails by construction (NPU smoke 2026-09-06)")
    rb.add_argument("--last_n", type=int, default=1)
    rb.add_argument("--qrow_chunk", type=int, default=16)
    rb.add_argument("--max_rows", type=int, default=0)
    rb.set_defaults(fn=_cmd_run_battery)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
