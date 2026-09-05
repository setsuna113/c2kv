# -*- coding: utf-8 -*-
"""t34 unit U3 -- digest section 4.7: localisation (which history block to repair).

Five migrations, all torch-free and zero-GPU, over the frozen r2 battery +
the server-side decoded-doc sidecar:

  (A) CausalCache proposal-as-reference   -- 2608.22577 sec 3.4 "Reference actions"
  (B) LANTERN RRF hybrid ranking          -- 2606.05182 sec 3.2 (RRF k=60)
  (C) sigma cutoff-margin abstain         -- 2607.07724 sec 3.3 / 3.4
  (D) gold-free dependency-edge oracle    -- 2605.25310 sec 3 ("Oracle tool-call DAG")
  (E) Lost-in-the-Middle position priors  -- 2307.03172 (NO transfer card; qualitative only)

Everything here consumes the compressed arm's OWN emission, the decoded doc
plaintext and free prefix scalars.  The one gold-side construct (the k*_gold
rung of the reference ladder) lives in a ``*_label_*`` function and is written
to the chooser file as an ENVELOPE row, never to the feature frame.

RUNBOOK (execution order; every step runs HERE on the Windows box, zero GPU)
---------------------------------------------------------------------------
0. (NPU SERVER, unit U2 owns this script) dump the decoded-doc sidecar for the
   compressed arm, and copy the D-line k-sweep flip table off the server:

     python agent/t34_dump_sidecar.py --arm c2kv --out results/t34/sidecar_c2kv.jsonl
     # scp ~/bench_results/d_v2/flip_table.jsonl -> results/t34/flip_table.jsonl

1. (LOCAL) build every chooser / ranker / oracle / prior arm into one file:

     PYTHONIOENCODING=utf-8 python agent/t34_localize.py choosers \
       --root . --sidecar results/t34/sidecar_c2kv.jsonl \
       --out results/t34/choosers_localize.jsonl

   Add ``--semantic`` to also run the MiniLM ranker (S14, a NEW dependency);
   its bytes + ms land in ``<out>.cost.json`` (per turn AND amortised per
   session, plus which encoder produced them) and it produces the second RRF
   arm (rankers 1-4).  Without it only the free arm (rankers 1-3) is written.
   On a box without torch ``--semantic`` ABORTS naming the dependency; the
   4-ranker row is then simply absent, never a degraded 3-ranker stand-in.

2. (LOCAL) score the locator table (S@k against the 25.0 % wrong-block floor,
   witness k* and flip-table hit columns kept separate):

     PYTHONIOENCODING=utf-8 python agent/t34_locate_score.py \
       --choosers results/t34/choosers_localize.jsonl \
       --witness configs/bdf_pilot/d_witness_r2.json \
       --flip results/t34/flip_table.jsonl \
       --out results/t34/locate_table_localize.json --print

3. (LOCAL) emit the sigma / edge / margin TRIGGER features into the t34 frame:

     PYTHONIOENCODING=utf-8 python agent/t34_localize.py features \
       --root . --sidecar results/t34/sidecar_c2kv.jsonl \
       --out results/t34/features_localize.jsonl

   Orientations for those columns: configs/t34/orientations_localize.json.
   Merge + score with agent/t34_score.py (owned by another unit).

WIRING (bench face)
-------------------
Nothing here edits ``benchmarks/``.  The deployable half -- ``values_from_proposal``
-> ``block_scores`` -> ``rrf_fuse`` -> ``sigma_top1`` -> abstain -- is written as
pure functions of (decoded block texts, query text, the arm's own emission), so
on the bench face it plugs into ``benchmarks/proxy.py:plan_repair`` (:827), which
already holds every doc's plaintext and runs BEFORE the model call.  The edge
oracle (D) additionally needs the emitted action, so its deployable variant
belongs in ``RecoverState.check`` (``benchmarks/proxy.py:494``), after the
compressed arm's answer exists and before it is committed.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from t34_common import (  # noqa: E402
    FrozenAssets,
    check_docs_against_witness,
    chooser_argmax,
    fixed_rate_threshold,
    grouped_folds,
    locate_table,
    normalize_ws,
    session_of,
    step_index,
    write_features_jsonl,
)
from d_witness_core import select_k_star, target_values, witness_scores  # noqa: E402
from t33_labels import load_jsonl  # noqa: E402
from t33_spanmap import parse_tool_call  # noqa: E402

# --------------------------------------------------------------------------
# declared constants (NOT fitted; every one of them is a DEVIATIONS entry)
# --------------------------------------------------------------------------

#: LANTERN's Reciprocal Rank Fusion constant (2606.05182 sec 3.2).  Their Exp. 6
#: (N=46) shows recovery is FLAT for k in [10, 200], so this is declared, never
#: tuned.
RRF_K = 60

#: LANTERN recency half-life.  The paper uses T_half = 7 days of wall-clock;
#: our history blocks have no timestamps, so Delta t is measured in HISTORY
#: BLOCK STEPS and T_half is declared as 8 = max_doc_num / 2 (16 / 2).
T_HALF_STEPS = 8.0

#: 2605.25310 sec 3: the dependency-edge oracle's minimum verbatim substring
#: length (contiguous, whitespace-normalised, case-SENSITIVE).
EDGE_MIN_CHARS = 4

#: chooser names (stable strings; t34_locate_score keys its table on them)
CH_GOLD = "k_star_gold_label_envelope"
CH_PROPOSAL = "k_star_proposal"
CH_QUERY_ONLY = "k_star_query_only"
CH_NONE = "k_star_none"
CH_FIRST = "k_first"
CH_MEDIAN = "k_median"
CH_LAST = "k_last"
CH_TWOPOINT = "k_first_or_last_twopoint"
CH_RRF3 = "rrf_free3"
CH_RRF4 = "rrf_semantic4"
CH_EDGE_OWN = "edge_own_argmax"
CH_EDGE_GOLD = "edge_gold_label_envelope"
CH_EDGE_TYPED = "edge_typed_equality_own"

PRIOR_CHOOSERS = (CH_FIRST, CH_MEDIAN, CH_LAST)


DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "CausalCache proposal-as-reference",
        "paper": "2608.22577 sec 3.4",
        "what": "Their reference is a GUI action compared against archived events by a trained "
                "two-tower selector; ours is [predicted tool name] + JSON leaves of the predicted "
                "arguments fed to the FROZEN witness-IDF scorer (d_witness_core, sum 1/df), with no "
                "trained selector, no budget B and no beam search.",
        "why": "The transferable claim is the oracle SUBSTITUTION (gold reference -> policy's own "
               "tentative action), not the HGKV adapter; the card records that Frozen+selector "
               "carries 33.9 of the 36.8, and porting the adapter would create a third checkpoint.",
    },
    {
        "method": "CausalCache proposal-as-reference",
        "paper": "2608.22577 sec 3.4",
        "what": "41/93 C->W predictions are censored at the 128-token cap and carry no closing "
                "</tool_call>; for those rows the proposal values are SALVAGED leniently from the "
                "truncated arguments region (complete quoted strings and numbers only) instead of "
                "being dropped.",
        "why": "Their GUI proposals are always complete actions.  Dropping the censored rows would "
               "silently delete 44 % of the trigger set; salvage keeps the row and the chooser "
               "note's `salvaged` / `parse_ok` flags make the difference visible.",
    },
    {
        "method": "CausalCache reference ladder",
        "paper": "2608.22577 Table `chooser`",
        "what": "Their ladder is scored by log-likelihood margin over Recent-B; ours is scored by "
                "S@k against the frozen 25.0 % wrong-block floor plus the inverted-score control.",
        "why": "We have no budgeted-allocation object (Arm.validate refuses repair+recover), so the "
               "only shared estimand is 'did the chooser pick the reference block'.",
    },
    {
        "method": "LANTERN RRF",
        "paper": "2606.05182 sec 3.2",
        "what": "Four rankers become: witness-IDF over query+proposal leaves (their lexical/FTS pair "
                "collapsed into ONE ranker), tag-Jaccard, importance, optional MiniLM cosine.  "
                "Their FTS5 full-text ranker has no analogue (no SQLite index) and is not a fifth list.",
        "why": "Both of their lexical lists score the same query-vs-turn text overlap; duplicating it "
               "would double-weight lexical evidence inside a fusion that has no weights.",
    },
    {
        "method": "LANTERN tag extraction",
        "paper": "2606.05182 sec 3.1 step 4",
        "what": "Their tags come from structured per-turn metadata (file paths, error codes, "
                "function names).  Ours adds bare (unquoted) argument literals of >= 8 characters "
                "because our blocks are chat-rendered grid rows, not structured turn records.",
        "why": "Without it the tag ranker is identically 0 on prose-only blocks and the RRF arm "
               "silently degenerates to two lists.  The >= 8 threshold is not new: it is the one "
               "already frozen in d_witness_core.occurs.",
    },
    {
        "method": "LANTERN importance",
        "paper": "2606.05182 sec 3.2",
        "what": "I(e) = R(e)*F(e)*D(e) with c_e and sigma_e DROPPED; R uses Delta t in history-block "
                "steps with T_half = 8 steps instead of 7 days; F(e)=log2(a_e+1)+1 with a_e = number "
                "of LATER blocks in the same qid that quote one of block k's tags; D(e) = block token "
                "length / max block token length.",
        "why": "c_e and sigma_e need a cross-session persistence store we do not have; our blocks have "
               "no timestamps, only an order.  Declared, not fitted.",
    },
    {
        "method": "LANTERN RRF",
        "paper": "2606.05182 sec 3.2",
        "what": "MMR (lambda=0.7) and the 6000-char budget fill are skipped; k=60 is declared, not tuned.",
        "why": "We select exactly ONE block, so diversification has no meaning; their Exp. 6 shows "
               "k in [10,200] is flat at 0.730, so tuning it would be selection on nothing.",
    },
    {
        "method": "sigma cutoff margin",
        "paper": "2607.07724 sec 3.3",
        "what": "Their sigma = (s_(k-1)-s_(k))/(s_(0)-s_(k)) is re-based for top-1 selection to "
                "sigma_top1 = (s_(0)-s_(1))/(s_(0)-s_(min)); raw gap s_(0)-s_(1) and second-order gap "
                "s_(1)-s_(2) are carried as declared alternatives, and the paper-faithful sigma at a "
                "given k is kept as `sigma_paper`.",
        "why": "With k=1 their numerator and denominator collapse onto the same gap.  The card names "
               "this re-basing as THE one non-mechanical step of the port; it is pre-registered here.",
    },
    {
        "method": "sigma quantile trigger",
        "paper": "2607.07724 sec 3.4",
        "what": "Their quantile is taken over hundreds of Q-tiles inside ONE forward; ours is taken "
                "across sessions/steps of a held-out calibration fold (chosen in the inner folds of a "
                "session-grouped nested CV).",
        "why": "We have <=16 docs and one decision per step, so the in-forward population does not "
               "exist.  This converts a self-normalising rule into one that needs held-out "
               "calibration -- declared here, not in a footnote.",
    },
    {
        "method": "sigma quantile trigger (q = 0 endpoint)",
        "paper": "2607.07724 sec 3.4 / App. A.8",
        "what": "Their rule is `trigger[t] = 1[sigma_bar <= quantile_q]`, a CLOSED boundary, so at "
                "q = 0 the single minimum-sigma row still fires.  The boundary is kept verbatim for "
                "every q > 0; q = 0 alone is handled by an explicit branch that abstains on NOTHING, "
                "so the abstain curve starts at a true zero-abstention left endpoint.",
        "why": "Their q is swept over {0.10 ... 0.40} and never reaches 0, so the endpoint is outside "
               "the range they calibrated; keeping the closed boundary there would make the curve's "
               "first point an abstention rate of 1/n rather than 0 and misdraw the whole plot.  The "
               "departure is confined to q <= 0 and is reported by `never_abstain: True` on that row.",
    },
    {
        "method": "sigma head aggregation",
        "paper": "2607.07724 sec 3.4",
        "what": "Their uniform mean over attention HEADS becomes a uniform mean over (score-mode, "
                "layer-group) score vectors, still applied BEFORE thresholding.",
        "why": "Their negative result -- per-cell gating does not work, aggregate first -- is the "
               "transferable part; our cells are score modes, not heads.",
    },
    {
        "method": "dependency-edge oracle",
        "paper": "2605.25310 sec 3",
        "what": "Their edge is between two TOOL CALLS inside one trajectory; ours is between a decoded "
                "HISTORY BLOCK and the current step's action.  Locator strength (their oracle is "
                "binary) is defined as the LONGEST verbatim matched substring length.",
        "why": "Our compressed blocks are grid rows, not calls with a recoverable output field; a "
               "binary edge row cannot pick one block, and argmax needs an ordering.",
    },
    {
        "method": "dependency-edge oracle",
        "paper": "2605.25310 sec 3",
        "what": "The 'serialised JSON arguments' of the action are json.dumps(args, ensure_ascii=False) "
                "when the emission parses strictly, and the raw arguments-region substring verbatim "
                "when it does not (128-token cap).  Both sides are whitespace-normalised, case kept.",
        "why": "The cap truncates the exact string the substring oracle matches into; edge_own "
               "therefore UNDER-counts on C->W rows, so results are stratified by censored_at_cap "
               "(their setting has no truncation).",
    },
    {
        "method": "dependency-edge oracle",
        "paper": "2605.25310 App. 'Oracle construction (full specification)'",
        "what": "Their a_j is json.dumps(arguments) at DEFAULT settings, i.e. ensure_ascii=True; we "
                "serialise with ensure_ascii=False.",
        "why": "Under their setting a non-ASCII argument is escaped to \\uXXXX and can never match the "
               "block's literal non-ASCII text, so the oracle would silently drop every non-Latin "
               "argument in our BFCL / tau2 mix.  Declared because it is a paper-fixed constant we "
               "deliberately changed; the >= 4-char, whitespace-normalised, case-sensitive rule and "
               "the re.sub(r'\\s+',' ').strip() normalisation are kept verbatim on BOTH sides.",
    },
    {
        "method": "schema-typed value-equality oracle",
        "paper": "2605.25310 sec 4.1.2 / App. 'Independent (non-substring) oracle replication'",
        "what": "Their independent oracle fires only when a TYPED-ID value produced by call i (a "
                "structured `_id` field or a bare entity return) is passed under a TYPED-ID KEY of "
                "call j.  Ours compares EVERY typed JSON leaf of the block's plaintext against every "
                "typed leaf of the action's arguments, with no key schema and no id-field filter.",
        "why": "Our history blocks are chat-rendered grid rows with no tau-bench retail schema behind "
               "them, so there is no typed-ID key list to condition on.  The consequence is one-sided "
               "and must be reported as such: our variant is BROADER than theirs, so it cannot "
               "inherit their precision 1.000 / strict-subset claim -- it is reported as a separate "
               "cross-check arm, never as the paper's number.",
    },
    {
        "method": "dependency-edge oracle (gold envelope)",
        "paper": "2605.25310 sec 3",
        "what": "`edge_gold` should match into the serialised JSON arguments of the GOLD action; the "
                "frozen witness table only stores `tool_name` + `arg_leaf_values`, so we match into "
                "json.dumps of that leaf LIST instead of the original argument object.",
        "why": "The gold argument object is not in any frozen asset on this box.  The list "
               "serialisation preserves every leaf verbatim (only the keys and nesting are lost), and "
               "the arm is an envelope, never a point estimate, so a small recall loss on key-adjacent "
               "substrings cannot inflate any reported locator.",
    },
    {
        "method": "dependency-edge probe",
        "paper": "2605.25310 sec 3 'Probe'",
        "what": "The 10,240-dim residual-stream edge probe is NOT ported; only the label/oracle half is.",
        "why": "Their H_i is the residual at an EARLIER call's boundary token, which on the compressed "
               "arm has been replaced by a gist; the card's own tier note says everything above the "
               "label needs redesign.  Their residual beat the 36-dim surface decoder by only +0.039.",
    },
    {
        "method": "position priors",
        "paper": "2307.03172",
        "what": "NO transfer card exists for this paper (phase2_literature/reads/2307.03172.md is "
                "absent); only the qualitative 'beginning or end is best' claim is used.  No number "
                "from that paper is quoted anywhere, and the two-point rule's threshold is fitted on "
                "session-grouped training folds only.",
        "why": "CRITIC_phase2 marks the entry venue/year unverified and second-hand; quoting any of "
               "its numbers would be fabrication.",
    },
    {
        "method": "LANTERN semantic ranker (S14)",
        "paper": "2606.05182 sec 3.1 / 3.2",
        "what": "Their ranker 3 is an all-MiniLM-L6-v2 cosine over archived turn embeddings and is "
                "always present.  Ours is OPTIONAL (--semantic) and is simply ABSENT from the table "
                "on a box without torch; when it is absent the RRF arm is the free 3-ranker fusion "
                "and is labelled as such, never a 4-ranker fusion with a degraded stand-in.  The "
                "cost row records which encoder produced it (`encoder`: the model name, or "
                "`injected` for a test double).",
        "why": "sentence_transformers + torch do not exist on the analysis box, so the arm has never "
               "been measured against the real package here; reporting an injected encoder's bytes "
               "and ms as a MiniLM measurement would be a fabricated number.",
    },
    {
        "method": "LANTERN importance D(e)",
        "paper": "2606.05182 sec 3.2 eq. I(e)",
        "what": "Their D(e) is 'richness (bonuses for tool calls and file references)' -- an "
                "unspecified bonus schedule.  Ours is block token length / max block token length.",
        "why": "The bonus schedule is not given anywhere in the paper, so copying it is impossible; "
               "the digest's migration line fixes richness as normalised token length.  Declared "
               "because it is a substitution, not a reading of their formula.",
    },
]


# ==========================================================================
# sidecar / frozen-asset plumbing
# ==========================================================================

@dataclass
class LocalizeRow:
    """One decision step: everything a chooser is allowed to read.

    SIDECAR SEMANTICS (``agent/t34_dump_sidecar.py``), because getting them
    wrong silently inverts every visibility statement:

    * ``docs`` holds ONLY the KEPT blocks -- the decoded grid rows the model
      actually saw, in order.  EVERY entry of ``docs`` is visible; there is no
      such thing as a dropped entry inside it, so a filter of the form
      ``enumerate(docs) ... if i not in dropped_docs`` is always a bug.
    * ``dropped_docs`` indexes the POST-SPLIT history list, NOT ``docs``.  It is
      carried for provenance only and is never used as an index into ``docs``.
    * the dropped blocks' TEXT exists only when the dump was taken with
      ``--with_dropped_text``; ``dropped_doc_texts`` is then that text and
      ``None`` otherwise.  ``None`` means UNAVAILABLE and must be reported as
      such -- never approximated by a subset of ``docs``.
    """

    qid: str
    session_id: str
    docs: List[str]
    query: str
    prediction: str
    doc_lengths: List[int]
    decision_step: Optional[int] = None
    #: indices into the POST-SPLIT history list (see the class docstring)
    dropped_docs: List[int] = field(default_factory=list)
    #: decoded text of the dropped blocks; None = the dump did not carry it
    dropped_doc_texts: Optional[List[str]] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    censored_at_cap: Optional[bool] = None
    #: gold values for the label-side envelope rung ONLY (witness table fields)
    gold_values: Optional[List[str]] = None
    #: reference block (frozen witness k*), None when the target has no witness
    truth_k: Optional[int] = None

    @property
    def n_docs(self) -> int:
        """Number of VISIBLE blocks -- every element of ``docs``."""
        return len(self.docs)

    @property
    def visible_docs(self) -> List[str]:
        """All of ``docs``: the sidecar already dropped what the model lost."""
        return list(self.docs)

    @property
    def dropped_side_available(self) -> bool:
        """Whether the dropped blocks' text is present at all (``--with_dropped_text``)."""
        return self.dropped_doc_texts is not None

    @property
    def step(self) -> int:
        return step_index(self.qid)


def load_sidecar(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read ``results/t34/sidecar_<arm>.jsonl`` (schema fixed by the runner).

    Lines: {qid, session_id, docs, query, tools, system_prompt, doc_lengths,
    dropped_docs, kept_history_tokens} plus the OPTIONAL ``dropped_doc_texts``
    (only with ``t34_dump_sidecar.py --with_dropped_text``).  ``docs`` is the
    KEPT set; ``dropped_docs`` indexes the post-split history list, not ``docs``
    (see :class:`LocalizeRow`).  Returns qid -> row.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for r in load_jsonl(str(path)):
        out[str(r["qid"])] = r
    return out


def build_rows(root: Path, sidecar_path: Path,
               *, qids: Optional[Iterable[str]] = None) -> Tuple[List[LocalizeRow], Dict[str, Any]]:
    """Join the frozen battery + witness table + decoded-doc sidecar.

    The sidecar is cross-checked against the witness table's per-doc sha256
    (``t34_common.check_docs_against_witness``) so a stale dump cannot be
    scored silently.
    """
    frame = FrozenAssets(root).load()
    side = load_sidecar(sidecar_path)
    c2kv = frame.c2kv_by_qid
    labels = {r["qid"]: r for r in frame.labels}
    docs_only = {q: list(r.get("docs") or []) for q, r in side.items()}
    audit = check_docs_against_witness(docs_only, frame)

    want = set(qids) if qids is not None else set(side)
    rows: List[LocalizeRow] = []
    for qid in sorted(want):
        s = side.get(qid)
        c = c2kv.get(qid)
        if s is None or c is None:
            continue
        ent = frame.witness_entry(qid) or {}
        gold_vals = None
        if ent:
            # frozen gold `values` exactly as d_witness_core built them:
            # [target tool name] + JSON leaves of the target arguments.
            raw = ([str(ent["tool_name"])] if ent.get("tool_name") else []) \
                + [str(v) for v in (ent.get("arg_leaf_values") or [])]
            gold_vals = list(dict.fromkeys(v for v in raw if v))
        lab = labels.get(qid) or {}
        rows.append(LocalizeRow(
            qid=qid,
            session_id=str(s.get("session_id") or c.get("session_id") or session_of(qid)),
            docs=list(s.get("docs") or []),
            query=str(s.get("query") or ""),
            prediction=str(c.get("prediction") or ""),
            doc_lengths=[int(x) for x in (s.get("doc_lengths") or [])],
            decision_step=c.get("decision_step"),
            dropped_docs=[int(x) for x in (s.get("dropped_docs") or [])],
            dropped_doc_texts=(list(s["dropped_doc_texts"])
                               if s.get("dropped_doc_texts") is not None else None),
            tools=list(s.get("tools") or []),
            censored_at_cap=lab.get("censored_at_cap"),
            gold_values=gold_vals,
            truth_k=ent.get("k_witness") if ent else None,
        ))
    audit["n_rows"] = len(rows)
    audit["n_sidecar"] = len(side)
    # loud, named availability flags: a missing input is reported, never
    # approximated.  The dropped side is UNAVAILABLE unless the dump was taken
    # with --with_dropped_text; nothing here ever reconstructs it from `docs`.
    n_dropped_text = sum(1 for r in rows if r.dropped_side_available)
    audit["dropped_side_available"] = bool(rows) and n_dropped_text == len(rows)
    audit["n_rows_with_dropped_text"] = n_dropped_text
    audit["n_rows_with_dropped_blocks"] = sum(1 for r in rows if r.dropped_docs)
    audit["docs_are_all_visible"] = True
    # check_docs_against_witness can only sha-verify qids the frozen witness
    # table covers (the 93 C->W rows).  A sidecar row for a C->C qid -- which
    # still feeds the FEATURE frame -- has nothing to be checked against, so a
    # stale dump there is invisible to --allow_stale_sidecar.  Report the size
    # of that blind spot by name instead of letting "0 mismatches" read as
    # "everything verified".
    unverifiable = sorted(q for q in side if not frame.witness_entry(q))
    audit["n_qids_without_witness_sha"] = len(unverifiable)
    audit["qids_without_witness_sha_sample"] = unverifiable[:5]
    audit["sha_coverage_complete"] = not unverifiable
    return rows, audit


# ==========================================================================
# (A) CausalCache proposal-as-reference -- 2608.22577 sec 3.4
# ==========================================================================

_QUOTED_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")
_ARG_KEY_RE = re.compile(r'"(?:[^"\\]|\\.)*"\s*:\s*')


def _mask_quoted(text: str) -> str:
    """Blank out every CLOSED quoted region (quotes included), same length.

    The bare-number scan must not see digits that live inside a JSON string:
    ``{"user_id":"123"`` has a STRING leaf, and a number scan over the raw text
    would additionally manufacture the int ``123`` -- which the schema-typed
    oracle (2605.25310 sec 4.1.2) would then match against an int in a block,
    exactly the type confusion that oracle exists to avoid.
    """
    out = list(text or "")
    for m in _QUOTED_RE.finditer(text or ""):
        for i in range(m.start(), m.end()):
            out[i] = " "
    return "".join(out)


def _salvage_arg_values(args_text: str) -> List[str]:
    """Lenient JSON-leaf salvage from a TRUNCATED arguments region.

    2608.22577 sec 3.4 assumes a complete proposal action; our 128-token cap
    leaves 41/93 C->W emissions without a closing tag, so the arguments object
    never balances.  We keep only leaves that are unambiguously complete:
    fully closed quoted strings that are NOT immediately followed by ``:``
    (those are keys) and bare numbers.  Order preserved, no dedup here.
    """
    values: List[str] = []
    for m in _QUOTED_RE.finditer(args_text):
        tail = args_text[m.end():]
        if re.match(r"\s*:", tail):
            continue  # this quoted token is a key, not a leaf
        try:
            values.append(json.loads('"' + m.group(1) + '"'))
        except Exception:
            values.append(m.group(1))
    for m in _NUMBER_RE.finditer(_mask_quoted(args_text)):
        values.append(m.group(0))
    return [str(v) for v in values if str(v)]


def _salvage_typed_arg_values(args_text: str) -> List[Any]:
    """``_salvage_arg_values`` but keeping the JSON TYPE of each salvaged leaf.

    The schema-typed value-equality oracle (2605.25310 sec 4.1.2) compares by
    type-aware equality, so a bare ``123`` inside a truncated arguments region
    has to stay an ``int``.  Stringifying it (as the witness-IDF path must,
    because ``d_witness_core.occurs`` matches text) would make the typed oracle
    structurally unable to match ANY numeric leaf on the 41/93 censored rows --
    a silent, one-sided under-count on exactly the rows the cap pitfall warns
    about.
    """
    values: List[Any] = []
    for m in _QUOTED_RE.finditer(args_text):
        tail = args_text[m.end():]
        if re.match(r"\s*:", tail):
            continue  # key, not a leaf
        try:
            values.append(json.loads('"' + m.group(1) + '"'))
        except Exception:
            values.append(m.group(1))
    for m in _NUMBER_RE.finditer(_mask_quoted(args_text)):
        try:
            values.append(json.loads(m.group(0)))
        except Exception:
            values.append(m.group(0))
    return [v for v in values if v != ""]


def values_from_proposal(prediction_text: str) -> Dict[str, Any]:
    """CausalCache's reference substitution (2608.22577 sec 3.4, "Reference actions").

    ``values = [predicted tool name] + JSON leaves of the predicted arguments``,
    taken from the compressed arm's OWN first-pass emission -- the deployable
    stand-in for ``d_witness_core``'s gold ``values``.  Returns a dict with the
    values plus the provenance flags the pitfall section demands
    (``parse_ok`` / ``closed`` / ``salvaged``), never a bare list.
    """
    parsed = parse_tool_call(prediction_text or "")
    name = parsed.get("name")
    salvaged = False
    if parsed.get("parse_ok"):
        vals = target_values(name, parsed.get("arguments"))
    else:
        span = parsed.get("args_span")
        raw = prediction_text[span[0]:span[1]] if span else ""
        leaves = _salvage_arg_values(raw)
        salvaged = bool(leaves) or bool(span)
        vals = list(dict.fromkeys(v for v in (([str(name)] if name else []) + leaves) if v))
    return {
        "values": vals,
        "name": name,
        "parse_ok": bool(parsed.get("parse_ok")),
        "closed": bool(parsed.get("closed")),
        "has_tool_call": bool(parsed.get("has_tool_call")),
        "salvaged": salvaged,
    }


_WORD_RE = re.compile(r"[A-Za-z0-9_.:/@-]{2,}")


def values_from_query(query_text: str) -> List[str]:
    """CausalCache's weak "instruction-only similarity" reference (2608.22577 sec 3.4).

    values = JSON leaves of the query when it parses as JSON, else its word-ish
    tokens (``[A-Za-z0-9_.:/@-]{2,}``), deduplicated and order-preserving.
    """
    text = query_text or ""
    try:
        obj = json.loads(text)
    except Exception:
        obj = None
    if obj is not None and isinstance(obj, (dict, list)):
        from d_witness_core import leaves as _leaves
        vals = _leaves(obj)
    else:
        vals = _WORD_RE.findall(text)
    return list(dict.fromkeys(v for v in vals if v))


def values_from_target_label(row: LocalizeRow) -> List[str]:
    """LABEL-SIDE ONLY: the frozen gold ``values`` (witness table fields).

    This is the k*_gold rung of the CausalCache ladder (2608.22577 sec 3.4) and
    the 76.3 % envelope.  It reads the target and therefore MUST NOT reach the
    feature frame -- the name carries ``_label_`` for exactly that reason.
    """
    return list(row.gold_values or [])


def block_scores(texts: Sequence[str], values: Sequence[str]) -> List[float]:
    """Frozen witness-IDF per-block score (d_witness_core.witness_scores, sum 1/df).

    Unchanged from the prereg v2.2 algorithm; only the ``values`` constructor
    varies across the ladder (2608.22577 sec 3.4).
    """
    if not texts:
        return []
    _, scores = witness_scores(list(texts), list(values))
    return [float(s) for s in scores]


def chooser_from_values(texts: Sequence[str], values: Sequence[str]) -> Tuple[Optional[int], List[float]]:
    """(k_hat, score vector) using the FROZEN selector semantics.

    ``d_witness_core.select_k_star`` decides ``None`` (max score <= 0) and the
    lowest-index tie break; we never re-implement either.
    """
    scores = block_scores(texts, values)
    khat = select_k_star(list(texts), list(values)) if texts else None
    return khat, scores


def positional_scores_at(n: int, k: Optional[int]) -> List[float]:
    """Tent score vector peaked at block ``k``: ``s_i = n - |i - k|``.

    This is THE constructor for every position-prior score vector, because the
    inverted-score control inverts exactly this vector: its argmax has to be the
    block the prior actually picked, whatever the prior's own rule was.  At
    ``k = 0`` the tent is the descending "first" vector and at ``k = n - 1`` the
    ascending "last" vector, so the three priors are one formula.  Returns [] for
    an empty block set or an undefined ``k``.
    """
    if n <= 0 or k is None:
        return []
    kk = int(k)
    if kk < 0 or kk >= n:
        return []
    return [float(n - abs(i - kk)) for i in range(n)]


def positional_scores(n: int, prior: str) -> List[float]:
    """Score vector whose argmax is the position prior's choice (2307.03172).

    ``first`` -> descending, ``last`` -> ascending, ``median`` -> a tent peaked
    at ``(n - 1) // 2``, which is the FROZEN ``k_median`` convention
    (``d_witness_select.py:150``) and NOT ``n // 2``: on the 41/93 qids that
    saturate ``max_doc_num = 16`` the two differ (7 vs 8), and the argmax of
    this vector must agree with ``position_prior_khat`` or the inverted-score
    control would score a different block than the chooser reported.  Built
    through :func:`positional_scores_at` so that agreement is structural rather
    than a coincidence of three hand-written formulas.
    """
    if n <= 0:
        return []
    if prior not in ("first", "median", "last"):
        raise ValueError(f"unknown prior {prior!r}")
    return positional_scores_at(n, position_prior_khat(n, prior))


def ladder_choosers(row: LocalizeRow) -> List[Dict[str, Any]]:
    """The full CausalCache reference ladder for one row (2608.22577 Table `chooser`).

    Rungs: k*_gold (LABEL envelope) / k*_proposal / k*_query_only / k*_none /
    k_first.  ``k*_none`` emits an EMPTY score vector so the inverted-score
    control reports it as undefined rather than silently choosing block 0.
    """
    texts = row.docs
    n = len(texts)
    out: List[Dict[str, Any]] = []

    prop = values_from_proposal(row.prediction)
    for name, vals in (
        (CH_GOLD, values_from_target_label(row)),
        (CH_PROPOSAL, prop["values"]),
        (CH_QUERY_ONLY, values_from_query(row.query)),
    ):
        khat, scores = chooser_from_values(texts, vals)
        out.append(_chooser_row(row, name, khat, scores,
                                note={"n_values": len(vals)}))
    out.append(_chooser_row(row, CH_NONE, None, [], note={"n_values": 0}))
    first_vec = positional_scores(n, "first")
    # chooser_argmax (frozen select_k_star semantics: None when max <= 0, lowest
    # index on ties) rather than a bare argmax, so every arm in this file abstains
    # by the same rule the witness selector uses.
    out.append(_chooser_row(row, CH_FIRST, chooser_argmax(first_vec), first_vec))
    for r in out:
        if r["chooser"] == CH_PROPOSAL:
            r["note"].update({"parse_ok": prop["parse_ok"], "closed": prop["closed"],
                              "salvaged": prop["salvaged"]})
    return out


def _chooser_row(row: LocalizeRow, chooser: str, khat: Optional[int],
                 scores: Sequence[float], note: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "qid": row.qid,
        "session_id": row.session_id,
        "chooser": chooser,
        "khat": (int(khat) if khat is not None else None),
        "score_vector": [float(s) for s in scores],
        "n_docs": row.n_docs,
        "censored_at_cap": row.censored_at_cap,
        "note": dict(note or {}),
    }


# ==========================================================================
# (B) LANTERN RRF hybrid ranking -- 2606.05182 sec 3.2
# ==========================================================================

_TAG_TOOLNAME_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:__[A-Za-z0-9_]+)+\b")
_TAG_PATH_RE = re.compile(r"\b(?:[A-Za-z]:)?(?:[\w.-]+[\\/])+[\w.-]+\b")
_TAG_ERRCODE_RE = re.compile(r"\b(?:[45]\d{2}|E\d{3,}|[A-Z][A-Z0-9_]{3,})\b")
#: bare (unquoted) argument literals.  The >= 8 length threshold is the one
#: already frozen in ``d_witness_core.occurs`` (substring match at >= 8 chars,
#: word-boundary regex below it), so the tag ranker and the witness-IDF ranker
#: agree on what counts as a long literal.
_TAG_BARE_LITERAL_RE = re.compile(r"[A-Za-z0-9_.:/@-]{8,}")


def extract_tags(text: str) -> set:
    """LANTERN's pattern-matched lookup hints (2606.05182 sec 3.1 step 4).

    Four families, exactly the ones the digest names: tool names (our
    ``ns__tool`` dialect), file paths, error codes, and argument literals
    (complete quoted strings and bare numbers).  Case is preserved; the tag set
    is what the keyword ranker takes a Jaccard over.
    """
    t = text or ""
    tags = set()
    tags.update(_TAG_TOOLNAME_RE.findall(t))
    tags.update(_TAG_PATH_RE.findall(t))
    tags.update(_TAG_ERRCODE_RE.findall(t))
    for m in _QUOTED_RE.finditer(t):
        v = m.group(1)
        if len(v) >= 2:
            tags.add(v)
    tags.update(_NUMBER_RE.findall(t))
    tags.update(_TAG_BARE_LITERAL_RE.findall(t))
    return {v for v in tags if v}


def ranker_witness_idf(row: LocalizeRow) -> List[float]:
    """LANTERN ranker 1 (lexical), 2606.05182 sec 3.2.

    Witness-IDF with ``values`` = query leaves + proposal leaves -- i.e. the
    (A) constructor reused, never gold.
    """
    vals = list(dict.fromkeys(values_from_query(row.query)
                              + values_from_proposal(row.prediction)["values"]))
    return block_scores(row.docs, vals)


def ranker_tag_jaccard(row: LocalizeRow) -> List[float]:
    """LANTERN ranker 2 (keyword/tag overlap), 2606.05182 sec 3.2.

    Jaccard between a block's tag set and the tag set of (emitted action ||
    query).  Same construction family as Tracy's ``argument_grounding_score``
    (bfcl_history_kv_repair.py:484) but scored per block, not per step.
    """
    need = extract_tags((row.prediction or "") + "\n" + (row.query or ""))
    out: List[float] = []
    for t in row.docs:
        have = extract_tags(t)
        union = need | have
        out.append(float(len(need & have) / len(union)) if union else 0.0)
    return out


def ranker_importance(row: LocalizeRow, t_half: float = T_HALF_STEPS) -> List[float]:
    """LANTERN ranker 4 (importance), 2606.05182 sec 3.2: I(e) = R(e)*F(e)*D(e).

    R(e) = exp(-0.693*Delta t / T_half) with Delta t = (n_docs-1) - k in history
    block steps; F(e) = log2(a_e+1)+1 with a_e = the number of LATER blocks that
    quote one of block k's tags; D(e) = block token length / max block token
    length.  ``c_e`` and ``sigma_e`` are DROPPED (declared: no cross-session
    store).
    """
    texts = row.docs
    n = len(texts)
    if n == 0:
        return []
    lengths = row.doc_lengths if len(row.doc_lengths) == n else [len(t) for t in texts]
    max_len = max(lengths) if lengths and max(lengths) > 0 else 1
    tag_sets = [extract_tags(t) for t in texts]
    out: List[float] = []
    for k in range(n):
        dt = (n - 1) - k
        recency = math.exp(-0.693 * dt / float(t_half)) if t_half > 0 else 1.0
        a_e = sum(1 for j in range(k + 1, n) if tag_sets[k] & tag_sets[j])
        freq = math.log2(a_e + 1) + 1.0
        richness = float(lengths[k]) / float(max_len)
        out.append(float(recency * freq * richness))
    return out


#: S14 is a NEW dependency and this box has no torch, so the MiniLM arm has
#: never been run against the real package here.  Missing it is an INPUT
#: problem, not a modelling one: say so by name rather than surfacing a bare
#: ImportError from inside a ranker.
SEMANTIC_DEP_HELP = (
    "the MiniLM ranker (S14, LANTERN ranker 3) needs sentence_transformers + torch, "
    "which are absent on this box. Install them where torch exists (the NPU server) "
    "or drop --semantic: without it the free arm (rankers 1-3) is written alone and "
    "the rrf_semantic4 row is simply ABSENT from the table, never a degraded stand-in."
)


def load_semantic_encoder(model_name: str = "all-MiniLM-L6-v2") -> Callable[[List[str]], Any]:
    """Lazily build the MiniLM encoder (never at module import: no torch here).

    Aborts with a NAMED message when the optional dependency is missing, so a
    ``--semantic`` run on a box without torch fails on the dependency instead
    of half-way through the chooser loop with a bare ImportError.
    """
    try:
        from sentence_transformers import SentenceTransformer  # lazy: S14
    except Exception as exc:  # ImportError, or torch blowing up underneath it
        raise SystemExit("MISSING DEPENDENCY (semantic): %s [%s]" % (SEMANTIC_DEP_HELP, exc))
    model = SentenceTransformer(model_name)
    return lambda batch: model.encode(batch)


def ranker_semantic(row: LocalizeRow,
                    encode_fn: Optional[Callable[[List[str]], Any]] = None,
                    model_name: str = "all-MiniLM-L6-v2",
                    encoder_label: Optional[str] = None) -> Tuple[List[float], Dict[str, Any]]:
    """LANTERN ranker 3 (semantic cosine), 2606.05182 sec 3.2 -- S14, NEW dependency.

    ``encode_fn`` is injected in tests; in production it comes from
    :func:`load_semantic_encoder`.  Returns (cosine scores, cost dict with
    bytes + ms) because the free arm and this arm must be reported separately;
    ``cost["encoder"]`` records WHICH encoder produced the numbers, since the
    injected one is not the real MiniLM.
    """
    texts = row.docs
    encoder = encoder_label or ("injected" if encode_fn is not None else model_name)
    if not texts:
        return [], {"model": model_name, "encoder": encoder, "ms": 0.0, "bytes": 0, "n_texts": 0}
    if encode_fn is None:  # pragma: no cover - needs the optional dependency
        encode_fn = load_semantic_encoder(model_name)
    t0 = time.perf_counter()
    emb = np.asarray(encode_fn(list(texts) + [row.query or ""]), dtype=float)
    ms = (time.perf_counter() - t0) * 1000.0
    doc_emb, q_emb = emb[:-1], emb[-1]
    dn = np.linalg.norm(doc_emb, axis=1)
    qn = float(np.linalg.norm(q_emb))
    denom = np.where(dn > 0, dn, 1.0) * (qn if qn > 0 else 1.0)
    scores = (doc_emb @ q_emb) / denom
    cost = {"model": model_name, "encoder": encoder, "ms": float(ms),
            "bytes": int(emb.nbytes), "n_texts": len(texts) + 1}
    return [float(s) for s in scores], cost


def ranks_from_scores(scores: Sequence[float]) -> List[int]:
    """1-based competition ranking (``1,2,2,4``) of ``scores``, descending.

    Ties share the best rank so RRF cannot be gamed by float noise
    (2606.05182 sec 3.2 does not specify tie handling; declared here).
    """
    s = list(scores)
    order = sorted(range(len(s)), key=lambda i: (-s[i], i))
    ranks = [0] * len(s)
    prev_val: Optional[float] = None
    prev_rank = 0
    for pos, idx in enumerate(order, start=1):
        if prev_val is not None and s[idx] == prev_val:
            ranks[idx] = prev_rank
        else:
            ranks[idx] = pos
            prev_rank = pos
            prev_val = s[idx]
    return ranks


def rrf_fuse(rank_lists: Sequence[Sequence[int]], k: int = RRF_K) -> List[float]:
    """Reciprocal Rank Fusion (2606.05182 sec 3.2): RRF(e) = sum_L 1/(k + rank_L(e)).

    ``k = 60`` fixed -- their Exp. 6 (N=46) shows recovery is flat for
    k in [10,200], so it is declared, never tuned.  MMR is skipped (we pick one
    block).
    """
    if not rank_lists:
        return []
    n = len(rank_lists[0])
    if any(len(r) != n for r in rank_lists):
        raise ValueError("rank lists must have equal length")
    return [float(sum(1.0 / (k + r[i]) for r in rank_lists)) for i in range(n)]


def rrf_choosers(row: LocalizeRow,
                 encode_fn: Optional[Callable[[List[str]], Any]] = None,
                 with_semantic: bool = False,
                 encoder_label: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Both LANTERN arms: free rankers 1-3, and 1-4 when the MiniLM arm is on."""
    n = row.n_docs
    if n == 0:
        return [_chooser_row(row, CH_RRF3, None, [])], None
    r1 = ranker_witness_idf(row)
    r2 = ranker_tag_jaccard(row)
    r3 = ranker_importance(row)
    free = rrf_fuse([ranks_from_scores(r1), ranks_from_scores(r2), ranks_from_scores(r3)])
    # chooser_argmax, not np.argmax: the frozen select_k_star semantics (abstain
    # when nothing scores above zero, lowest index on ties) are the ones the
    # inverted-score control assumes on the forward arm.
    rows = [_chooser_row(row, CH_RRF3, chooser_argmax(free), free,
                         note={"rankers": ["witness_idf", "tag_jaccard", "importance"]})]
    cost = None
    if with_semantic:
        r4, cost = ranker_semantic(row, encode_fn=encode_fn, encoder_label=encoder_label)
        fused = rrf_fuse([ranks_from_scores(r1), ranks_from_scores(r2),
                          ranks_from_scores(r3), ranks_from_scores(r4)])
        rows.append(_chooser_row(row, CH_RRF4, chooser_argmax(fused), fused,
                                 note={"rankers": ["witness_idf", "tag_jaccard",
                                                   "importance", "semantic"],
                                       "cost": cost}))
    return rows, cost


# ==========================================================================
# (C) sigma margin abstain -- 2607.07724 sec 3.3 / 3.4
# ==========================================================================

def sigma_paper(scores: Sequence[float], k: int) -> Optional[float]:
    """The paper's cutoff margin, 2607.07724 sec 3.3, kept verbatim.

    ``sigma = (s_(k-1) - s_(k)) / (s_(0) - s_(k))`` on the descending sort:
    the gap between the last KEPT and the first DROPPED block, normalised by
    the spread from the best kept block down to the first dropped one.
    Returns None when the vector is too short for a k-cut; 0.0 when the
    denominator vanishes (a coin-flip cut).
    """
    s = sorted((float(x) for x in scores), reverse=True)
    if k < 1 or len(s) <= k:
        return None
    num = s[k - 1] - s[k]
    den = s[0] - s[k]
    if den <= 0:
        return 0.0
    return float(num / den)


def sigma_top1(scores: Sequence[float]) -> Optional[float]:
    """Top-1 re-based margin (2607.07724 sec 3.3, re-based -- see DEVIATIONS).

    ``sigma_top1 = (s_(0) - s_(1)) / (s_(0) - s_(min))``.  With top-1 selection
    the paper's numerator and denominator collapse onto the same gap, so the
    denominator is re-based to the full spread.  This is the ONE non-mechanical
    step of the port and is pre-registered.
    """
    s = sorted((float(x) for x in scores), reverse=True)
    if len(s) < 2:
        return None
    den = s[0] - s[-1]
    if den <= 0:
        return 0.0
    return float((s[0] - s[1]) / den)


def gap_raw(scores: Sequence[float]) -> Optional[float]:
    """Declared alternative 1 (2607.07724 sec 3.3): the raw gap s_(0) - s_(1)."""
    s = sorted((float(x) for x in scores), reverse=True)
    return float(s[0] - s[1]) if len(s) >= 2 else None


def gap_second(scores: Sequence[float]) -> Optional[float]:
    """Declared alternative 2 (2607.07724 sec 3.3): second-order gap s_(1) - s_(2)."""
    s = sorted((float(x) for x in scores), reverse=True)
    return float(s[1] - s[2]) if len(s) >= 3 else None


def aggregate_sigma(score_vectors: Dict[str, Sequence[float]],
                    fn: Callable[[Sequence[float]], Optional[float]] = sigma_top1) -> Optional[float]:
    """Uniform mean of per-cell sigma BEFORE thresholding (2607.07724 sec 3.4).

    Their negative result is the transferable part: per-cell gating does not
    work, the mean across cells does.  Cells here are score modes / layer
    groups instead of attention heads.  Returns None when no cell is defined.
    """
    vals = [v for v in (fn(vec) for vec in score_vectors.values()) if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def sigma_threshold(sigma_values: Sequence[float], q: float) -> float:
    """Fixed-rate cut: fire on the bottom-``q`` fraction of sigma (2607.07724 sec 3.4).

    Implemented through ``t34_common.fixed_rate_threshold`` on the NEGATED
    vector so the one frozen quantile helper is the only place a rate becomes
    a threshold.  Rows with sigma <= the returned value are the fired ones.
    """
    return float(-fixed_rate_threshold(np.asarray([-float(v) for v in sigma_values], dtype=float), q))


def sigma_abstain_curve(sigma_by_qid: Dict[str, Optional[float]],
                        hit_by_qid: Dict[str, Optional[bool]],
                        q_grid: Sequence[float]) -> List[Dict[str, Any]]:
    """Estimand (a): sigma as an ABSTAIN gate on the locator (2607.07724 sec 3.4).

    For each q, blocks with sigma <= threshold get ``k*=None``.  Reports S@k
    with abstentions counted as misses (the ``locate_table`` convention), the
    SELECTIVE accuracy over the kept decisions, and the abstain rate -- three
    numbers that must never be merged.

    The paper's boundary is CLOSED (``sigma <= quantile_q``) and is kept
    verbatim for every ``q > 0``.  ``q <= 0`` gets an explicit branch that
    abstains on NOTHING -- see the DEVIATIONS entry "sigma quantile trigger
    (q = 0 endpoint)": under the closed boundary q = 0 would still abstain on
    the single minimum-sigma row, which makes the left endpoint of a curve
    plotted from zero 1/n instead of 0.  Rows produced by that branch carry
    ``never_abstain: True``; rows with an UNDEFINED sigma stay ``None`` there
    too, because "no sigma" is a missing input, not an abstain decision.
    """
    qids = [q for q in sorted(sigma_by_qid) if sigma_by_qid[q] is not None]
    vals = [float(sigma_by_qid[q]) for q in qids]
    rows: List[Dict[str, Any]] = []
    for q in q_grid:
        never = float(q) <= 0.0
        # None, never -inf / nan: an undefined threshold is a missing value, and
        # a sentinel here would also make the row un-serialisable as strict JSON.
        thr = None if (never or not vals) else sigma_threshold(vals, q)
        gated: Dict[str, Optional[bool]] = {}
        kept_hits = kept_n = n_undef = 0
        for qid in sorted(hit_by_qid):
            s = sigma_by_qid.get(qid)
            if s is None or (thr is not None and s <= thr):
                gated[qid] = None
                n_undef += int(s is None)
                continue
            gated[qid] = hit_by_qid[qid]
            if hit_by_qid[qid] is not None:
                kept_n += 1
                kept_hits += int(bool(hit_by_qid[qid]))
        tab = locate_table(gated, label=f"sigma_abstain_q{q:g}")
        rows.append({
            "q": float(q),
            "threshold": (float(thr) if thr is not None else None),
            "never_abstain": bool(never), **tab,
            # `abstained` is the locate_table convention (gate abstentions AND
            # rows with no sigma at all); the second group is a MISSING INPUT,
            # not a decision, so its size is reported by name next to it.
            "n_sigma_undefined": n_undef,
            "abstain_rate": (tab["abstained"] / tab["n"]) if tab["n"] else None,
            "gate_abstain_rate": (((tab["abstained"] - n_undef) / tab["n"]) if tab["n"] else None),
            "selective_n": kept_n,
            "selective_accuracy": (kept_hits / kept_n) if kept_n else None,
        })
    return rows


def select_q_grouped(sigma_by_qid: Dict[str, Optional[float]],
                     y_by_qid: Dict[str, int],
                     groups_by_qid: Dict[str, str],
                     q_grid: Sequence[float] = (0.10, 0.20, 0.30, 0.40),
                     outer_folds: int = 5,
                     inner_folds: int = 3,
                     seed: int = 20260905,
                     objective: str = "f1") -> Dict[str, Any]:
    """Estimand (b): sigma as a TRIGGER, q chosen in INNER folds only.

    Session-grouped nested CV in the ``t34_common.nested_cv_logistic`` pattern:
    q is picked on inner folds of the outer-train fold, the threshold is then
    refit on the whole outer-train fold, and the outer-test rows are fired
    against it.  Nothing is chosen on evaluation rows (rule 6).  Objective is
    declared (``f1`` over coverage/precision by default); the paper's own
    q=0.40 grid is the default grid.
    """
    qids = [q for q in sorted(sigma_by_qid)
            if sigma_by_qid[q] is not None and q in y_by_qid and q in groups_by_qid]
    if not qids:
        return {"n_scored": 0, "chosen": [], "fires": {}}
    sig = np.asarray([float(sigma_by_qid[q]) for q in qids])
    y = np.asarray([int(y_by_qid[q]) for q in qids])
    groups = np.asarray([groups_by_qid[q] for q in qids])

    def _obj(fire: np.ndarray, yy: np.ndarray) -> Optional[float]:
        n_f = int(fire.sum())
        n_p = int(yy.sum())
        if n_p == 0:
            return None
        cov = float((fire & (yy == 1)).sum()) / n_p
        prec = (float((fire & (yy == 1)).sum()) / n_f) if n_f else 0.0
        if objective == "precision":
            return prec
        if objective == "coverage":
            return cov
        return (2 * cov * prec / (cov + prec)) if (cov + prec) > 0 else 0.0

    fires = np.zeros(len(qids), dtype=bool)
    scored = np.zeros(len(qids), dtype=bool)
    chosen: List[Dict[str, Any]] = []
    for outer in grouped_folds(groups, outer_folds, seed):
        tr, te = ~outer, outer
        if te.sum() == 0 or tr.sum() == 0 or y[tr].sum() == 0:
            continue
        best: Optional[Tuple[float, float]] = None
        for q in q_grid:
            vals: List[float] = []
            for inner in grouped_folds(groups[tr], inner_folds, seed + 1):
                itr, ite = ~inner, inner
                if itr.sum() == 0 or ite.sum() == 0:
                    continue
                thr = sigma_threshold(sig[tr][itr], q)
                m = _obj(sig[tr][ite] <= thr, y[tr][ite])
                if m is not None:
                    vals.append(m)
            if vals:
                mean = float(np.mean(vals))
                if best is None or mean > best[0]:
                    best = (mean, float(q))
        if best is None:
            continue
        thr = sigma_threshold(sig[tr], best[1])
        fires[te] = sig[te] <= thr
        scored[te] = True
        chosen.append({"q": best[1], "inner_objective": best[0], "threshold": float(thr),
                       "n_test": int(te.sum())})
    idx = {qid: i for i, qid in enumerate(qids)}
    return {
        "n_scored": int(scored.sum()),
        "n_pos_scored": int(y[scored].sum()),
        "objective": objective,
        "chosen": chosen,
        "fires": {qid: bool(fires[idx[qid]]) for qid in qids if scored[idx[qid]]},
    }


# ==========================================================================
# (D) gold-free dependency-edge oracle -- 2605.25310 sec 3
# ==========================================================================

def longest_common_substring_len(a: str, b: str, min_len: int = EDGE_MIN_CHARS) -> int:
    """Longest contiguous, case-SENSITIVE substring of ``a`` occurring in ``b``.

    2605.25310 sec 3 ("Oracle tool-call DAG") only needs existence at length
    >= 4; we return the LONGEST match because a locator needs an ordering
    (declared in DEVIATIONS).  Returns 0 when no match reaches ``min_len``.
    Implemented by seeding on the ``min_len``-grams of ``b`` and extending, so
    the cost is linear in |a| for the (many) non-matching rows.
    """
    if not a or not b or min_len <= 0 or len(a) < min_len or len(b) < min_len:
        return 0
    seeds = {b[i:i + min_len] for i in range(len(b) - min_len + 1)}
    best = 0
    for i in range(len(a) - min_len + 1):
        if a[i:i + min_len] not in seeds:
            continue
        if best >= len(a) - i:
            continue
        length = min_len
        while i + length < len(a) and a[i:i + length + 1] in b:
            length += 1
        if length > best:
            best = length
    return best


def action_arguments_text(prediction_text: str) -> Tuple[str, bool]:
    """Serialised JSON arguments of the emitted action (2605.25310 sec 3).

    Returns (whitespace-normalised text, strict_parse_ok).  When the emission
    parses strictly we serialise the parsed arguments with
    ``json.dumps(..., ensure_ascii=False)``; when the 128-token cap left it
    unterminated we take the raw arguments-region substring verbatim.  Case is
    preserved, whitespace normalised -- exactly the oracle's normalisation.
    """
    parsed = parse_tool_call(prediction_text or "")
    if parsed.get("parse_ok") and parsed.get("arguments") is not None:
        return normalize_ws(json.dumps(parsed["arguments"], ensure_ascii=False)), True
    span = parsed.get("args_span")
    raw = prediction_text[span[0]:span[1]] if span else ""
    return normalize_ws(raw), False


def edge_strengths(doc_texts: Sequence[str], args_text: str,
                   min_len: int = EDGE_MIN_CHARS) -> List[int]:
    """Per-block edge strength (2605.25310 sec 3).

    ``edge(k,t) = 1`` iff a contiguous, whitespace-normalised, case-SENSITIVE
    substring of block k's decoded plaintext of length >= 4 occurs verbatim in
    the serialised JSON arguments of step t's action.  Strength = the longest
    such substring's length (0 when there is no edge).
    """
    tgt = args_text or ""
    return [longest_common_substring_len(normalize_ws(t), tgt, min_len) for t in doc_texts]


def edge_locator(strengths: Sequence[int]) -> Optional[int]:
    """argmax_k of edge strength; None when every block has strength 0.

    2605.25310 sec 4.2.1 is explicit that a good AUROC does NOT give coherent
    thresholded decisions (their P(i->k | i->j, j->k) = 0.307 < the 0.429
    independence null), so the primary metric here is the argmax hit rate,
    never AUROC.

    Delegates to ``t34_common.chooser_argmax`` so the abstain rule (no block
    scores above zero) and the lowest-index tie break are the FROZEN
    ``select_k_star`` ones rather than a second hand-written argmax.
    """
    return chooser_argmax([float(s) for s in strengths])


_JSON_OBJ_RE = re.compile(r"\{")


def typed_values_from_text(text: str, max_objects: int = 64) -> set:
    """Typed JSON leaves parsed out of a block's decoded plaintext.

    Support for the schema-typed value-equality oracle (2605.25310 sec 4.1.2),
    which agrees with the substring oracle at precision 1.0 and is a strict
    subset.  Leaves are kept as ``(type name, value)`` pairs so ``1`` and
    ``"1"`` and ``True`` never compare equal.
    """
    from t34_common import json_leaves

    out = set()
    if not text:
        return out
    n_obj = 0
    for m in _JSON_OBJ_RE.finditer(text):
        if n_obj >= max_objects:
            break
        obj = _balanced_json(text, m.start())
        if obj is None:
            continue
        n_obj += 1
        for leaf in json_leaves(obj):
            out.add(_typed_key(leaf))
    return out


def _balanced_json(text: str, start: int) -> Optional[Any]:
    depth = 0
    in_str = False
    esc = False
    for j in range(start, len(text)):
        c = text[j]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:j + 1])
                except Exception:
                    return None
    return None


def _typed_key(value: Any) -> Tuple[str, Any]:
    if isinstance(value, bool):
        return ("bool", value)
    if value is None:
        return ("null", None)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value)
    return ("str", str(value))


def typed_equality_edges(doc_texts: Sequence[str], prediction_text: str) -> List[int]:
    """Schema-typed value-equality oracle (2605.25310 sec 4.1.2), the precision cross-check.

    No substring matching: a block gets an edge iff one of its typed JSON leaf
    values is EXACTLY equal (type-aware) to one of the action's argument leaf
    values.  Returns the number of matching leaves per block (0 = no edge).
    """
    parsed = parse_tool_call(prediction_text or "")
    if parsed.get("parse_ok") and parsed.get("arguments") is not None:
        from t34_common import json_leaves
        arg_keys = {_typed_key(v) for v in json_leaves(parsed["arguments"])}
    else:
        span = parsed.get("args_span")
        raw = prediction_text[span[0]:span[1]] if span else ""
        arg_keys = {_typed_key(v) for v in _salvage_typed_arg_values(raw)}
    if not arg_keys:
        return [0] * len(doc_texts)
    return [len(typed_values_from_text(t) & arg_keys) for t in doc_texts]


def edge_choosers(row: LocalizeRow) -> List[Dict[str, Any]]:
    """The three edge-oracle arms for one row (2605.25310 sec 3 / 4.1.2).

    ``edge_own``   -- action = the compressed arm's OWN emission (deployable);
    ``edge_gold``  -- action = the gold values (LABEL envelope, never a point
                      estimate; the name carries ``_label_``);
    ``edge_typed`` -- the schema-typed value-equality cross-check on the own
                      emission.
    """
    args_text, strict = action_arguments_text(row.prediction)
    own = edge_strengths(row.docs, args_text)
    gold_text = normalize_ws(json.dumps(list(row.gold_values or []), ensure_ascii=False))
    gold = edge_strengths(row.docs, gold_text)
    typed = typed_equality_edges(row.docs, row.prediction)
    return [
        _chooser_row(row, CH_EDGE_OWN, edge_locator(own), [float(x) for x in own],
                     note={"strict_args": strict, "n_edges": sum(1 for x in own if x > 0)}),
        _chooser_row(row, CH_EDGE_GOLD, edge_locator(gold), [float(x) for x in gold],
                     note={"envelope": True, "n_edges": sum(1 for x in gold if x > 0)}),
        _chooser_row(row, CH_EDGE_TYPED, edge_locator(typed), [float(x) for x in typed],
                     note={"n_edges": sum(1 for x in typed if x > 0),
                           # 2605.25310's own typed oracle is conditioned on a
                           # typed-ID KEY schema we do not have, so ours is
                           # strictly BROADER and cannot inherit their precision
                           # 1.000 / strict-subset relation to the substring
                           # oracle.  The flag travels with the row so a table
                           # writer cannot lose it between here and the prose.
                           "broader_than_paper_typed_oracle": True,
                           "inherits_paper_precision_claim": False}),
    ]


# ==========================================================================
# (E) position priors -- 2307.03172 (NO card; qualitative claim only)
# ==========================================================================

def position_prior_khat(n_docs: int, prior: str, k_median: Optional[int] = None) -> Optional[int]:
    """k_first / k_median / k_last as REGISTERED prior arms (2307.03172).

    The only claim taken from that paper is qualitative -- performance is
    highest when the relevant information sits at the beginning or the end of
    the context -- because no transfer card exists for it.  No number from it
    is quoted anywhere in this repo.
    """
    if n_docs <= 0:
        return None
    if prior == "first":
        return 0
    if prior == "last":
        return n_docs - 1
    if prior == "median":
        # frozen convention: k_median = (n_docs - 1) // 2 (d_witness_select.py:150).
        # The witness table's own column wins when it is available.
        return int(k_median) if k_median is not None else (n_docs - 1) // 2
    raise ValueError(f"unknown prior {prior!r}")


def prior_choosers(row: LocalizeRow, k_median: Optional[int] = None) -> List[Dict[str, Any]]:
    """The three mandatory position-prior rows for one qid (2307.03172).

    Each row's score vector is built from the k it ACTUALLY chose
    (``positional_scores_at``), not from the prior's name: when the witness
    table supplies ``k_median`` and the sidecar's ``n_docs`` disagrees with the
    witness table's, a name-built tent would peak somewhere else and the
    inverted-score control would then invert a vector the chooser never used.
    """
    n = row.n_docs
    k_med = position_prior_khat(n, "median", k_median)
    k_last = position_prior_khat(n, "last")
    return [
        _chooser_row(row, CH_MEDIAN, k_med, positional_scores_at(n, k_med),
                     note={"k_median_source": ("witness" if k_median is not None else "frozen_rule")}),
        _chooser_row(row, CH_LAST, k_last, positional_scores_at(n, k_last)),
    ]


def two_point_prior_oof(rows: Sequence[LocalizeRow],
                        truth: Dict[str, Optional[int]],
                        scalar: Dict[str, Optional[float]],
                        *,
                        outer_folds: int = 5,
                        seed: int = 20260905) -> Dict[str, Any]:
    """First-or-last two-point rule keyed on a free scalar (2307.03172).

    ``k_hat = 0 if scalar <= threshold else n_docs-1``.  The threshold is a
    SELECTED hyper-parameter, so it is fitted on session-grouped TRAINING folds
    only and applied out-of-fold (rule 6); the evaluation rows never see it.
    """
    usable = [r for r in rows if r.n_docs > 0 and scalar.get(r.qid) is not None]
    if not usable:
        return {"khat": {}, "chosen": []}
    groups = np.asarray([r.session_id for r in usable])
    vals = np.asarray([float(scalar[r.qid]) for r in usable])
    khat: Dict[str, Optional[int]] = {}
    chosen: List[Dict[str, Any]] = []
    for outer in grouped_folds(groups, outer_folds, seed):
        tr, te = ~outer, outer
        if te.sum() == 0 or tr.sum() == 0:
            continue
        # the candidate grid itself is built from TRAINING scalars only: taking
        # it over the pooled vector would let the evaluation rows choose which
        # cut points exist (rule 6).
        grid = sorted(set(vals[tr].tolist())) + [float("inf")]
        best: Optional[Tuple[float, float]] = None
        for thr in grid:
            hits = tot = 0
            for i, r in enumerate(usable):
                if not tr[i] or truth.get(r.qid) is None:
                    continue
                pick = 0 if vals[i] <= thr else r.n_docs - 1
                tot += 1
                hits += int(pick == truth[r.qid])
            if tot == 0:
                continue
            rate = hits / tot
            if best is None or rate > best[0]:
                best = (rate, float(thr))
        if best is None:
            continue
        for i, r in enumerate(usable):
            if te[i]:
                khat[r.qid] = 0 if vals[i] <= best[1] else r.n_docs - 1
        chosen.append({"threshold": best[1], "train_hit_rate": best[0], "n_test": int(te.sum())})
    return {"khat": khat, "chosen": chosen}


# ==========================================================================
# assembly: choosers + trigger features
# ==========================================================================

def all_choosers(rows: Sequence[LocalizeRow],
                 *,
                 with_semantic: bool = False,
                 encode_fn: Optional[Callable[[List[str]], Any]] = None,
                 k_median_by_qid: Optional[Dict[str, int]] = None,
                 two_point_scalar: str = "decision_step") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Every arm of section 4.7 on one row set, as chooser records.

    Records are ``{qid, session_id, chooser, khat, score_vector, n_docs,
    censored_at_cap, note}`` -- the input format of ``t34_locate_score``.
    """
    kmed = k_median_by_qid or {}
    out: List[Dict[str, Any]] = []
    costs: List[Dict[str, Any]] = []
    # load the MiniLM encoder ONCE (it used to be rebuilt inside every row, which
    # both re-paid the model load per turn and folded that load into the per-turn
    # ms the digest's cost line asks for).
    encoder_label: Optional[str] = None
    if with_semantic and encode_fn is None:
        encode_fn = load_semantic_encoder()
        encoder_label = "all-MiniLM-L6-v2"
    for row in rows:
        out.extend(ladder_choosers(row))
        rrf_rows, cost = rrf_choosers(row, encode_fn=encode_fn, with_semantic=with_semantic,
                                      encoder_label=encoder_label)
        out.extend(rrf_rows)
        if cost:
            costs.append(cost)
        out.extend(edge_choosers(row))
        out.extend(prior_choosers(row, kmed.get(row.qid)))

    scalar: Dict[str, Optional[float]] = {}
    for r in rows:
        if two_point_scalar == "decision_step":
            scalar[r.qid] = float(r.decision_step) if r.decision_step is not None else float(r.step)
        else:
            scalar[r.qid] = float(sum(r.doc_lengths)) if r.doc_lengths else None
    truth = {r.qid: r.truth_k for r in rows}
    tp = two_point_prior_oof(rows, truth, scalar)
    by_qid = {r.qid: r for r in rows}
    for qid, k in sorted(tp["khat"].items()):
        r = by_qid[qid]
        # the score vector has to be the one whose argmax IS this row's choice:
        # the two-point rule picks first OR last per row, so emitting the
        # "first" vector unconditionally would make the inverted-score control
        # score k_first under the two-point label on every last-picking row.
        side = "first" if (k is not None and int(k) == 0) else "last"
        out.append(_chooser_row(r, CH_TWOPOINT, k, positional_scores_at(r.n_docs, k),
                                note={"scalar": two_point_scalar, "oof": True, "side": side}))
    n_sessions = len({r.session_id for r in rows})
    total_ms = float(sum(c["ms"] for c in costs)) if costs else 0.0
    total_bytes = int(sum(c["bytes"] for c in costs)) if costs else 0
    meta = {
        "n_rows": len(rows),
        "with_semantic": bool(with_semantic),
        "two_point_scalar": two_point_scalar,
        "two_point_folds": tp["chosen"],
        # 2606.05182 costing line: the MiniLM arm is the one non-free ranker and
        # the digest asks for it "bytes + ms per turn, AMORTISED PER SESSION".
        # Both denominators are reported and neither replaces the other.
        "semantic_cost": {
            # which encoder actually produced these numbers: "injected" means a
            # test double, NOT the real MiniLM (this box has no torch), and the
            # cost line must not be reported as a MiniLM measurement.
            "encoder": (sorted({str(c.get("encoder")) for c in costs})[0] if costs else None),
            "n_calls": len(costs),
            "total_ms": total_ms,
            "total_bytes": total_bytes,
            "n_sessions": n_sessions,
            "n_turns": len(rows),
            "ms_per_turn": (total_ms / len(rows)) if rows else None,
            "bytes_per_turn": (total_bytes / len(rows)) if rows else None,
            "ms_per_session": (total_ms / n_sessions) if n_sessions else None,
            "bytes_per_session": (total_bytes / n_sessions) if n_sessions else None,
        },
    }
    return out, meta


#: pre-declared risk orientations of the columns emitted by ``features_for_row``
#: (+1 = higher is riskier).  Mirrored in configs/t34/orientations_localize.json.
FEATURE_ORIENTATIONS: Dict[str, int] = {
    "loc_sigma_top1_proposal": -1,
    "loc_sigma_paper_k1_proposal": -1,
    "loc_gap_raw_proposal": -1,
    "loc_gap_second_proposal": -1,
    "loc_sigma_bar_agg": -1,
    "loc_rrf3_sigma_top1": -1,
    "loc_rrf3_margin": -1,
    "loc_edge_own_max_strength": -1,
    "loc_edge_own_block_count": -1,
    "loc_edge_typed_block_count": -1,
    "loc_proposal_n_values": -1,
    "loc_tag_jaccard_max": -1,
}


def features_for_row(row: LocalizeRow) -> Dict[str, Any]:
    """Per-qid TRIGGER features from section 4.7 (estimand (b) of 2607.07724 plus
    the grounding scalars of 2605.25310 sec 3).

    Reads only the compressed arm's own emission, the decoded blocks and the
    query -- never target / gold / tool_name_match / any full-arm field.  Every
    undefined value is ``None`` (json null), never a sentinel: a row with one
    block has no margin, and a row with no parsed action has no grounding.
    """
    prop = values_from_proposal(row.prediction)
    s_prop = block_scores(row.docs, prop["values"]) if row.docs else []
    r1 = ranker_witness_idf(row) if row.docs else []
    r2 = ranker_tag_jaccard(row) if row.docs else []
    r3 = ranker_importance(row) if row.docs else []
    rrf3 = rrf_fuse([ranks_from_scores(r1), ranks_from_scores(r2), ranks_from_scores(r3)]) if row.docs else []
    args_text, _ = action_arguments_text(row.prediction)
    own = edge_strengths(row.docs, args_text) if row.docs else []
    typed = typed_equality_edges(row.docs, row.prediction) if row.docs else []
    cells = {"witness_proposal": s_prop, "tag_jaccard": r2, "importance": r3}
    cells = {k: v for k, v in cells.items() if v}

    return {
        "qid": row.qid,
        "session_id": row.session_id,
        "loc_sigma_top1_proposal": sigma_top1(s_prop) if s_prop else None,
        "loc_sigma_paper_k1_proposal": sigma_paper(s_prop, 1) if s_prop else None,
        "loc_gap_raw_proposal": gap_raw(s_prop) if s_prop else None,
        "loc_gap_second_proposal": gap_second(s_prop) if s_prop else None,
        "loc_sigma_bar_agg": aggregate_sigma(cells) if cells else None,
        "loc_rrf3_sigma_top1": sigma_top1(rrf3) if rrf3 else None,
        "loc_rrf3_margin": gap_raw(rrf3) if rrf3 else None,
        "loc_edge_own_max_strength": (float(max(own)) if own else None),
        "loc_edge_own_block_count": (float(sum(1 for x in own if x > 0)) if own else None),
        "loc_edge_typed_block_count": (float(sum(1 for x in typed if x > 0)) if typed else None),
        "loc_proposal_n_values": (float(len(prop["values"])) if prop["has_tool_call"] else None),
        "loc_tag_jaccard_max": (float(max(r2)) if r2 else None),
    }


# ==========================================================================
# CLI
# ==========================================================================

def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


#: the two server-side inputs this unit cannot produce on the Windows box.
MISSING_INPUT_HELP: Dict[str, str] = {
    "sidecar": "run agent/t34_dump_sidecar.py on the NPU server: "
               "python agent/t34_dump_sidecar.py --arm c2kv "
               "--out results/t34/sidecar_c2kv.jsonl  (add --with_dropped_text to "
               "also carry the DROPPED blocks' text; without it the dropped side "
               "is reported as unavailable, never approximated from `docs`)",
    "choosers": "build it first: PYTHONIOENCODING=utf-8 python agent/t34_localize.py choosers "
                "--root . --sidecar results/t34/sidecar_c2kv.jsonl "
                "--out results/t34/choosers_localize.jsonl",
    "flip": "copy the D-line k-sweep table off the server: "
            "scp <server>:~/bench_results/d_v2/flip_table.jsonl results/t34/flip_table.jsonl",
}


def require_input(path: Path, kind: str) -> Path:
    """Abort with a NAMED, actionable message when a server-side input is absent.

    Both of this unit's real inputs are produced elsewhere (NPU server / D-line
    sweep).  Falling through to ``load_jsonl`` would raise a bare FileNotFound
    that says nothing about which artefact is missing or how to get it.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            "MISSING INPUT (%s): %s does not exist. Every number in this unit's "
            "table is unproduced until it does. To produce it: %s"
            % (kind, p, MISSING_INPUT_HELP.get(kind, "see the module RUNBOOK")))
    return p


def _guard_sidecar(audit: Dict[str, Any], allow_stale: bool) -> None:
    """Refuse to score a sidecar whose decoded docs disagree with the frozen
    witness table's per-doc sha256.  ``build_rows`` only RECORDS the mismatch;
    without this the whole locator table could be produced from a stale dump
    and nothing in the output would be wrong-looking."""
    bad = list(audit.get("mismatched_qids") or [])
    if bad and not allow_stale:
        raise SystemExit(
            "stale sidecar: %d qid(s) disagree with the frozen witness table's "
            "doc sha256 (first: %s). Re-dump it, or pass --allow_stale_sidecar "
            "to score anyway (the count is stamped into the .cost.json)."
            % (len(bad), ", ".join(bad[:5])))


def _cmd_choosers(args: argparse.Namespace) -> int:
    root = Path(args.root)
    rows, audit = build_rows(root, require_input(Path(args.sidecar), "sidecar"))
    _guard_sidecar(audit, bool(getattr(args, "allow_stale_sidecar", False)))
    frame = FrozenAssets(root).load()
    kmed = {}
    for r in rows:
        ent = frame.witness_entry(r.qid) or {}
        if ent.get("k_median") is not None:
            kmed[r.qid] = int(ent["k_median"])
    recs, meta = all_choosers(rows, with_semantic=bool(args.semantic),
                              k_median_by_qid=kmed, two_point_scalar=args.two_point_scalar)
    n = _write_jsonl(Path(args.out), recs)
    meta["sidecar_audit"] = audit
    Path(str(args.out) + ".cost.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"chooser_rows": n, "qids": len(rows),
                      "mismatched_sidecar_qids": len(audit.get("mismatched_qids") or []),
                      "qids_without_witness_sha": audit.get("n_qids_without_witness_sha"),
                      "dropped_side_available": bool(audit.get("dropped_side_available")),
                      "n_rows_with_dropped_blocks": audit.get("n_rows_with_dropped_blocks"),
                      "with_semantic": bool(args.semantic),
                      "semantic_encoder": meta["semantic_cost"]["encoder"]}, indent=2))
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    rows, audit = build_rows(Path(args.root), require_input(Path(args.sidecar), "sidecar"))
    _guard_sidecar(audit, bool(getattr(args, "allow_stale_sidecar", False)))
    recs = [features_for_row(r) for r in rows]
    n = write_features_jsonl(Path(args.out), recs, context="t34_localize features")
    print(json.dumps({"feature_rows": n, "columns": sorted(FEATURE_ORIENTATIONS),
                      "mismatched_sidecar_qids": len(audit.get("mismatched_qids") or []),
                      "qids_without_witness_sha": audit.get("n_qids_without_witness_sha"),
                      "dropped_side_available": bool(audit.get("dropped_side_available"))},
                     indent=2))
    return 0


def _cmd_orientations(args: argparse.Namespace) -> int:
    from t34_common import freeze_json
    sha = freeze_json(Path(args.out), FEATURE_ORIENTATIONS)
    print(json.dumps({"out": str(args.out), "sha256": sha, "n": len(FEATURE_ORIENTATIONS)}, indent=2))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="t34 U3 section 4.7 localisation arms")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("choosers", help="build every locator arm into a chooser jsonl")
    c.add_argument("--root", default=".", help="worktree root holding results/ and configs/")
    c.add_argument("--sidecar", required=True, help="results/t34/sidecar_c2kv.jsonl")
    c.add_argument("--out", required=True)
    c.add_argument("--semantic", action="store_true",
                   help="also run the MiniLM ranker (S14, new dependency; costed separately)")
    c.add_argument("--two_point_scalar", default="decision_step",
                   choices=["decision_step", "doc_chars"])
    c.add_argument("--allow_stale_sidecar", action="store_true",
                   help="score even when decoded docs disagree with the frozen witness sha256")
    c.set_defaults(func=_cmd_choosers)

    f = sub.add_parser("features", help="write the section 4.7 trigger features")
    f.add_argument("--root", default=".")
    f.add_argument("--sidecar", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--allow_stale_sidecar", action="store_true",
                   help="score even when decoded docs disagree with the frozen witness sha256")
    f.set_defaults(func=_cmd_features)

    o = sub.add_parser("orientations", help="(re)write configs/t34/orientations_localize.json")
    o.add_argument("--out", default="configs/t34/orientations_localize.json")
    o.set_defaults(func=_cmd_orientations)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
