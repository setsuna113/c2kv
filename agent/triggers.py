# -*- coding: utf-8 -*-
"""t34 §4.6 — deterministic L1 trigger checks (SIEVE / SAAG / gates / DART / Tracy).

Torch-free, CPU strings only, in the style of ``d_witness_core.py``: every
public function is a pure function of (emitted action, tool schemas, visible
text), so the whole module is unit-testable on the Windows box and importable
by the server-side dump script.

RUNBOOK (execution order)
-------------------------
0. [SERVER, one-off]  dump the per-doc plaintext sidecars that every grounding
   feature reads (this repo has no plaintext on disk):

     python agent/t34_dump_sidecar.py --arm c2kv --with_dropped_text --out results/t34/sidecar_c2kv.jsonl
     python agent/t34_dump_sidecar.py --arm full --with_dropped_text --out results/t34/sidecar_full.jsonl
     python agent/t34_dump_sidecar.py --arm c2kv --out results/t34/sidecar_c2kv.jsonl --check

   ``--check`` re-runs ``t34_common.check_docs_against_witness`` so a stale
   dump cannot be scored silently.  Copy both jsonl files back to this box.

   ``--with_dropped_text`` is the flag the digest's
   ``grounded_in_visible_history`` vs ``grounded_in_dropped_docs`` delta needs:
   WITHOUT it ``grounding_dropped_delta`` is null on every step that dropped a
   block and the frame says so through ``grounding_dropped_side_available = 0``
   and ``data_availability.n_rows_dropped_text_not_dumped``.  Turn it on
   deliberately and record it in the prereg.  ``docs`` holds ONLY the kept
   blocks; ``dropped_docs`` indexes the post-split history list and never
   ``docs`` (the bridge is the extension key ``kept_history_indices``).

1. [LOCAL, zero GPU]  argument-bearing census FIRST — the digest warns that
   48/93 C->W rows are tool-name-only, and stage 3 of SAAG / Tracy's grounding
   are undefined on those rows:

     PYTHONIOENCODING=utf-8 python agent/triggers.py census --out results/t34/l1_census.json

2. [LOCAL, zero GPU]  emit the feature frame (compressed arm) and the S0 twin
   (the SAME features recomputed on the full arm's own prediction against ITS
   own visible text — never a full-arm field read from the compressed row):

     PYTHONIOENCODING=utf-8 python agent/triggers.py build-features \
        --arm c2kv --sidecar results/t34/sidecar_c2kv.jsonl \
        --out results/t34/features_l1.jsonl
     PYTHONIOENCODING=utf-8 python agent/triggers.py build-features \
        --arm full --sidecar results/t34/sidecar_full.jsonl \
        --out results/t34/features_l1_full.jsonl

3. [LOCAL, zero GPU]  the SIEVE 2x3 cross-tab, the escalation accounting, the
   dropped_docs sub-analysis and the per-signal precision audit:

     PYTHONIOENCODING=utf-8 python agent/triggers.py report \
        --features results/t34/features_l1.jsonl \
        --sidecar results/t34/sidecar_c2kv.jsonl \
        --out results/t34/l1_report.json

4. [LOCAL]  scoring is NOT done here.  Hand the feature files to
   ``agent/t34_score.py`` (runner-owned) together with
   ``configs/t34/orientations_triggers.json``.  Locators are not scored here
   either: nothing in §4.6 chooses a block.

WIRING (bench face — this module never imports from ``benchmarks/``)
--------------------------------------------------------------------
Everything below is a pure function over ``(request dict, response dict, proxy
log rows, tool schemas)``.  The bench-branch plug points, for whoever wires it:

* ``route`` / ``saag_cascade`` / ``required_coverage`` — HOOK 2,
  ``benchmarks/proxy.py:RecoverState.check`` (:494), on the response message
  after ``action_canonical`` (:466) has canonicalised it.  Feed
  ``action=response_message``, ``tool_schemas=request["tools"]``,
  ``visible_text`` = the concatenated plaintext of the docs the compressor kept
  (proxy holds them), ``query`` = the last user message content.
* ``MustReadBeforeWriteGate`` — also HOOK 2 (it must see the proposed action);
  its raw/gist bookkeeping is fed at HOOK 1, ``benchmarks/proxy.py:plan_repair``
  (:827), where the compressed-record list is known.
* ``tool_contract`` — one step late by construction (S12): it reads the
  PREVIOUS turn's trailing ``role="tool"`` message out of ``request["messages"]``.
* ``silent_failure_census`` / ``firing_share_decomposition`` — offline over the
  request log written by ``benchmarks/proxy.py:_log_request`` (:1285); the row
  fields used are ``arm / status / error_kind / finish_reason / conv_id / turn``.
* ``admissible_recover`` — G2 (post-execution), the downstream-persistence K
  experiment.  It emits no ``d_t`` and MUST NOT enter the trigger table.

Nothing here edits ``benchmarks/``; nothing here imports it.

Faithfulness notes
------------------
Every function docstring cites the arXiv id and the section / equation it
implements.  ``DEVIATIONS`` below enumerates every departure from the papers.
No number from any paper is hard-coded as a result; the only paper constants
reproduced are the ones the papers themselves fix as part of the algorithm
(SAAG's ``0.8`` drift cut, ``0.6/0.4`` NCS weights, ``0.6`` hard/soft string
cut; Tracy's ``len>=3`` and last-12-messages window).  SAAG's ``0.34``-style
operating points are NOT applied: any cut is chosen in inner folds (§4.0).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "SIEVE Layer 1",
        "paper": "2512.06716 §3.2",
        "what": "The Intent Graph G_t = (V_t, E_t, Psi_t), the typed bindings "
                "Literal/Field/Output, and the control-flow match |M_t(a_t)| are "
                "NOT migrated.  Our PASS/DEVIATION/UNRESOLVED is decided by "
                "(parse ok, name-in-pool, required keys/types) and by lexical "
                "grounding of argument values in the visible decoded plaintext, "
                "not by graph-scoped output records.",
        "why": "The graph needs a trusted plan we do not have (and costs "
               "2,651-2,688 graph tokens/task, §4.6); our sessions carry no "
               "dataflow annotation.  Digest §4.6 instructs exactly this mapping.",
    },
    {
        "method": "SIEVE Layer 2",
        "paper": "2512.06716 §3.3",
        "what": "S_align, S_sem, S_causal, S_prov, S_risk, tau_low/tau_high and "
                "the corrective hint on Block are NOT migrated.",
        "why": "S_prov/S_risk have no analogue in a single-agent, uniform-repair-"
               "cost setting; the corrective hint is our text-erratum channel, "
               "measured dead (9.5% vs KV transplant 76-81%).  Only the "
               "escalation ACCOUNTING is ported.",
    },
    {
        "method": "SIEVE routing semantics",
        "paper": "2512.06716 §3.2",
        "what": "An action carrying no parseable tool call is routed DEVIATION "
                "(the paper's action a_t always exists).",
        "why": "Makes the parse-failure baseline (kill line T1) explicit inside "
               "the ladder rather than a separate column; equals SAAG FNEM=-1.",
    },
    {
        "method": "SIEVE DEVIATION vs the frozen parse-failure baseline",
        "paper": "2512.06716 §3.2 / kill line T1",
        "what": "``route == DEVIATION`` is NOT identical to "
                "``t33_labels.parse_fail_baseline``: the frozen baseline is "
                "gated on ``target_has_tool_call`` (a TARGET field, legal only "
                "on the label side) and looks at parseability alone, whereas "
                "the route is target-free and also fails on name-not-in-pool "
                "and required-key / type violations.",
        "why": "A feature may not read a target field (§4.0 item 1).  The two "
               "columns are reported side by side and must never be swapped: "
               "the frozen baseline stays the comparison line, the route is the "
               "candidate.  ``per_signal_precision_audit`` enforces this: the "
               "baseline is audited on its own row and is EXCLUDED from the "
               "combined ``_OR`` unless a caller names it in ``or_members``, so "
               "the combined detector can never silently read the target.",
    },
    {
        "method": "SAAG grounding target",
        "paper": "2607.18245 §2.3",
        "what": "AVEM / QSLO / VHR are computed against (visible decoded "
                "compressed-history plaintext UNION current query), not against "
                "the user query Q alone.  The paper-faithful Q-only variant is "
                "kept under the *_query_only names and reported beside it.",
        "why": "Our failures are caused by compressed history; the query is "
               "uncompressed and always present, so Q-only grounding is "
               "structurally uninformative here (card 2607.18245, 'The critical "
               "adaptation').  Digest §4.6 mandates the history version plus the "
               "grounded_in_visible_history vs grounded_in_dropped_docs delta.",
    },
    {
        "method": "SAAG fuzzy backend",
        "paper": "2607.18245 §2.1-2.3",
        "what": "RapidFuzz WRatio / PartialRatio are used when importable; this "
                "box has no rapidfuzz, so a difflib approximation is used and "
                "every row is stamped fuzz_backend='difflib'.  In the difflib "
                "backend the haystack is truncated to DIFFLIB_HAYSTACK_CHARS.",
        "why": "Numbers computed under the fallback are NOT comparable to the "
               "paper's; the stamp makes that visible in the artifact instead of "
               "silently mixing backends.",
    },
    {
        "method": "SAAG cascade",
        "paper": "2607.18245 §2.4",
        "what": "The correction loop psi_k (D_max = 15) is NOT migrated.",
        "why": "It is our text-erratum channel, already measured dead; SAAG's own "
               "results show it overcorrects (RPR .984->.886) and loses to a "
               "content-free ablation on solve rate for 2/3 models.",
    },
    {
        "method": "SAAG typed rules — unspecified types",
        "paper": "2607.18245 §2.3",
        "what": "Booleans, nulls and nested objects are treated as 'assumed' "
                "(excluded from VHR, counted in the AVEM denominator as "
                "not-verbatim).  A list is exact iff every element is exact, "
                "assumed iff no element fails and at least one is assumed; a "
                "non-exact list is charged h = 1 in VHR (the paper gives h only "
                "for strings and numerics).",
        "why": "The paper types only strings, numerics and lists; the remaining "
               "JSON types must be declared before running rather than decided "
               "row by row.  The excluded fraction is reported (vhr_excluded_frac).",
    },
    {
        "method": "SAAG VHR_ctx as a feature column",
        "paper": "2607.18245 §2.3, Table `intrinsic`",
        "what": "The pass criterion uses the paper's SUM (``VHR_ctx = Σ_{i∈G} h "
                "= 0``), but the feature frame emits BOTH ``vhr_ctx`` (that sum) "
                "and ``vhr_ctx_mean`` (the same quantity divided by |G|).",
        "why": "The paper's reported VHR (0.166-0.336) is a rate, not a sum; the "
               "sum grows with the number of arguments, so scoring the sum alone "
               "would confound hallucination with call arity.  Both are declared "
               "up front rather than one being picked after seeing the table.",
    },
    {
        "method": "SAAG empty-set conventions",
        "paper": "2607.18245 §2.1-2.2",
        "what": "Three cases the paper leaves undefined are fixed here: NCS "
                "with no trigram-sharing candidate is 0.0 (their 'fabrication' "
                "end of the scale); SPR with |P| = 0 is 0.0; and when a tool "
                "declares neither properties nor required parameters, A_all is "
                "empty so every emitted parameter has PDS = 0 and counts as "
                "spurious.",
        "why": "Undefined denominators must resolve to a declared constant "
               "before running, not to whatever the first row happens to hit.",
    },
    {
        "method": "SIEVE UNRESOLVED on unspecified types",
        "paper": "2512.06716 §3.2",
        "what": "Values classified ASSUMED (booleans, nulls, nested objects, "
                "empty lists, numerics with no numbers in the text) are counted "
                "as UNGROUNDED by ``route``, i.e. they escalate — unlike SAAG's "
                "VHR, which excludes them.",
        "why": "SIEVE routes every binding it cannot deterministically resolve "
               "to UNRESOLVED, explicitly including every ``Output(u)`` binding "
               "('Layer 1 does not treat arbitrary content from the recorded "
               "output as authorized').  The two ladders therefore disagree on "
               "these values by construction, and that is the papers' own "
               "difference, not an inconsistency in the port.",
    },
    {
        "method": "must_read_before_write",
        "paper": "2607.07405 §3.3",
        "what": "The gate is specialised to compression: it does not read "
                "db_state, and it fires when an identifier-shaped leaf of the "
                "proposed action occurs in history ONLY inside gisted docs.  The "
                "other three gates (cancellation_eligibility, baggage_allowance, "
                "passenger_count) are not migrated at all.",
        "why": "Those three encode airline policy; db_state is not available "
               "pre-execution in our proxy.  Digest §4.6 ports exactly one gate.  "
               "The identifier shape is a pre-registered regex (IDENTIFIER_RE), "
               "not a per-row judgement.",
    },
    {
        "method": "must_read_before_write — rejection",
        "paper": "2607.07405 §3.2",
        "what": "The gate returns a boolean fire; it does not short-circuit the "
                "call and emits no structured rejection message.",
        "why": "Their +12.4pp confounds the block with the rejection text (§7(9)); "
               "our text channel is dead, so that half is predicted not to "
               "transfer.  We keep the detector, not the remedy.",
    },
    {
        "method": "2608.02464 required_coverage",
        "paper": "2608.02464 §10",
        "what": "The paper's version reads 'every call the task requires', which "
                "is task-side ground truth.  Ours checks the SESSION tool schema "
                "against the calls actually emitted this turn (name resolves in "
                "the pool AND the tool's declared required parameters are all "
                "covered by the emitted arguments).",
        "why": "Task requirements are gold; the session schema is not.  Named "
               "differently in the report so the substitution is visible.  It "
               "returns None (not False) on an unparseable emission and when no "
               "tool pool is advertised, so the parse-failure baseline is not "
               "silently re-counted inside a second column.",
    },
    {
        "method": "2608.02464 total_consistency",
        "paper": "2608.02464 §10",
        "what": "NOT migrated.  The stand-in is argument grounding against "
                "visible history, exported under the explicit name "
                "grounding_standin_for_total_consistency.",
        "why": "total_consistency is arithmetic re-derivation specific to their "
               "research/calculator family.  We must never claim we ported it.",
    },
    {
        "method": "2608.02464 tool_contract",
        "paper": "2608.02464 §10",
        "what": "Uses the tool's declared return/response schema when the schema "
                "provides one; otherwise a documented name-typed heuristic that "
                "returns None (unknown) rather than False when it cannot type "
                "the return.",
        "why": "Our tool schemas mostly declare parameters only.  A heuristic "
               "that guessed False would manufacture the 0-false-positive result "
               "the paper reports; unknown must stay unknown.",
    },
    {
        "method": "DART AdmissibleRecover",
        "paper": "2605.23311 §4.5 Def. 6",
        "what": "Identified() is taken as GIVEN (an input), never computed. "
                "NoCommittedConflict is reported on two channels: (a) the "
                "gist-extraction channel, which is vacuous under a left-to-right "
                "compressor and is reported as a vacuous bound, and (b) the "
                "execution channel via d_witness_core's 1/df machinery with the "
                "MODEL'S OWN emitted actions substituted for gold.",
        "why": "Deployment-time localisation is out of scope (red line); the "
               "vacuous-bound report is mandated by the digest.  DART emits no "
               "d_t, so none of this enters the trigger table.",
    },
    {
        "method": "DART EffectAllowed",
        "paper": "2605.23311 §4.5",
        "what": "Loaded from a frozen effect policy file; the shipped skeleton "
                "(configs/t34/effect_policy_skeleton.json) contains NO tool "
                "names, only documented placeholders.",
        "why": "Inventing tau2/BFCL tool names would put unverified strings into "
               "a frozen policy.  A deployment copies the skeleton to "
               "effect_policy_<bench>.json and fills it from that bench's registry.",
    },
    {
        "method": "Tracy argument_grounding_score",
        "paper": "no arXiv (tmp/bfcl-c2kv bfcl_history_kv_repair.py:484-516)",
        "what": "Ported verbatim (last 12 messages, roles user/tool/assistant, "
                "values len>=3, lower-case substring, 1.0 when no values, 0.0 "
                "when parse fails and not an empty-execute response) as the "
                "reference row ONLY.  The 0.34 cut is not applied.",
        "why": "Digest §4.6: keep her rule under its own name, but in the feature "
               "frame rows without gradable values are emitted as None, not 1.0 "
               "— her 1.0 collapses the denominator on the 48/93 name-only rows.",
    },
    {
        "method": "grounded_in_dropped_docs convention",
        "paper": "2607.18245 card ('The critical adaptation'); digest §4.6",
        "what": "When a step dropped NO block, ``grounded_in_dropped_docs`` is "
                "fixed at 0.0 (no value can be grounded in an empty document "
                "set) and the delta is exactly ``grounded_in_visible_history`` "
                "— a DEFINED cell, not a missing one.  Only 'blocks were "
                "dropped but the sidecar carries no ``dropped_doc_texts``' and "
                "'the dropped count is unknown' are missing inputs; they emit "
                "delta=None with ``grounding_dropped_side_available = 0``.",
        "why": "The two states have different meanings and were previously both "
               "None, which made the column's missingness uninterpretable.  The "
               "convention is declared here rather than decided per row, and "
               "the availability flag travels with the column so the delta can "
               "never be read without it.",
    },
    {
        "method": "SIEVE route under the frozen 128-token generation cap",
        "paper": "2512.06716 §3.2 (caliber caveat: digest §4.0 requires >= 512)",
        "what": "On the frozen r2 battery ``max_new_tokens`` is 128, so a call "
                "can be routed DEVIATION merely because generation was cut off "
                "mid-JSON.  The arm's OWN ``generated_tokens`` is therefore "
                "emitted as the non-candidate stratifier ``censored_at_cap`` "
                "(1 iff generated_tokens >= the manifest's cap) and the "
                "cross-tab is reported stratified by it.",
        "why": "'Protocol legal' degenerates towards 'finished inside the cap' "
               "at caliber 128 (capped 0.9% vs uncapped 37.1%); the route may "
               "not be offered as evidence without the stratification, and the "
               "ladder still has to be re-run at caliber >= 512.  The column is "
               "target-free: it reads only the compressed arm's own row.",
    },
]

# --------------------------------------------------------------------------
# fuzzy string backend (rapidfuzz when importable, difflib otherwise)
# --------------------------------------------------------------------------

#: In the difflib fallback the haystack is truncated before the O(n*m) search.
DIFFLIB_HAYSTACK_CHARS = 20000

_FUZZ_BACKEND: Optional[str] = None


def fuzz_backend() -> str:
    """'rapidfuzz' when the library is importable, else 'difflib'.

    2607.18245 §2.1/§2.3 specify RapidFuzz WRatio / PartialRatio by name; the
    digest requires that any row computed under the fallback carries the stamp
    so its numbers are never compared with the paper's.
    """
    global _FUZZ_BACKEND
    if _FUZZ_BACKEND is None:
        try:
            import rapidfuzz  # noqa: F401  (lazy: not installed on this box)
            _FUZZ_BACKEND = "rapidfuzz"
        except Exception:
            _FUZZ_BACKEND = "difflib"
    return _FUZZ_BACKEND


def _difflib_partial_ratio(needle: str, haystack: str) -> float:
    """Best contiguous alignment of ``needle`` inside ``haystack``, 0..100.

    Approximation of RapidFuzz ``fuzz.partial_ratio``: locate the longest
    common block, then score ``needle`` against the equally long window of the
    haystack anchored on it.  Bounded work: one find_longest_match plus one
    ratio over ``len(needle)`` characters.
    """
    from difflib import SequenceMatcher

    if not needle:
        return 0.0
    hay = haystack[:DIFFLIB_HAYSTACK_CHARS]
    if not hay:
        return 0.0
    if len(needle) >= len(hay):
        return 100.0 * SequenceMatcher(None, needle, hay).ratio()
    matcher = SequenceMatcher(None, needle, hay)
    block = matcher.find_longest_match(0, len(needle), 0, len(hay))
    start = max(0, block.b - block.a)
    window = hay[start:start + len(needle)]
    best = SequenceMatcher(None, needle, window).ratio()
    # a second anchor at the raw block start guards the off-by-alignment case
    window2 = hay[block.b:block.b + len(needle)]
    best = max(best, SequenceMatcher(None, needle, window2).ratio())
    return 100.0 * best


def partial_ratio(a: str, b: str) -> float:
    """RapidFuzz ``fuzz.partial_ratio`` (0..100) with a difflib fallback.

    Used by 2607.18245 §2.3 for ``QSLO`` and for the string branch of ``VHR``.
    """
    a, b = str(a or ""), str(b or "")
    if fuzz_backend() == "rapidfuzz":
        from rapidfuzz import fuzz  # noqa: PLC0415
        return float(fuzz.partial_ratio(a, b))
    needle, hay = (a, b) if len(a) <= len(b) else (b, a)
    return _difflib_partial_ratio(needle, hay)


def w_ratio(a: str, b: str) -> float:
    """RapidFuzz ``fuzz.WRatio`` (0..100) with a difflib approximation.

    Used by 2607.18245 §2.1 (``NCS``) and §2.2 (``PDS``).  The fallback blends
    a whole-string ratio, a discounted partial ratio and a discounted
    token-sort ratio; it is an approximation and is stamped as such.
    """
    from difflib import SequenceMatcher

    a, b = str(a or ""), str(b or "")
    if fuzz_backend() == "rapidfuzz":
        from rapidfuzz import fuzz  # noqa: PLC0415
        return float(fuzz.WRatio(a, b))
    if not a or not b:
        return 0.0
    la, lb = a.lower(), b.lower()
    plain = SequenceMatcher(None, la, lb).ratio() * 100.0
    partial = partial_ratio(la, lb) * 0.90
    ts_a = " ".join(sorted(re.findall(r"\w+", la)))
    ts_b = " ".join(sorted(re.findall(r"\w+", lb)))
    token_sort = SequenceMatcher(None, ts_a, ts_b).ratio() * 100.0 * 0.95
    return float(max(plain, partial, token_sort))


def jaccard(a: str, b: str) -> float:
    """Jaccard over whitespace-tokenised lowercased strings (2607.18245 §2.1)."""
    sa = set(str(a or "").lower().split())
    sb = set(str(b or "").lower().split())
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _trigrams(text: str) -> Set[str]:
    t = str(text or "").lower()
    return {t[i:i + 3] for i in range(max(0, len(t) - 2))}


# --------------------------------------------------------------------------
# parsing + schema (LOCAL re-implementation; benchmarks/ is never imported)
# --------------------------------------------------------------------------

def parse_emitted_call(action: Any) -> Dict[str, Any]:
    """Normalise an emitted action into ``{name, arguments, parse_ok, ...}``.

    Accepts (a) the raw generated text of a battery row, (b) a proxy response
    message ``{"content":..., "tool_calls":[...]}``, or (c) an already-parsed
    ``{"name":..., "arguments":...}``.  Text parsing delegates to
    ``t33_spanmap.parse_tool_call`` so 'unparseable' means exactly what the
    frozen span map could not strictly parse (cap-censored rows whose JSON
    object still balanced count as parseable).
    """
    from t33_spanmap import parse_tool_call  # noqa: PLC0415

    if isinstance(action, dict) and ("tool_calls" in action or "content" in action):
        calls = action.get("tool_calls") or []
        if calls:
            first = calls[0]
            fn = first.get("function") or {}
            name = fn.get("name") or first.get("name")
            raw = fn.get("arguments", first.get("arguments"))
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    return {"name": name, "arguments": None, "parse_ok": False,
                            "has_tool_call": True, "closed": True, "text": ""}
            return {"name": name, "arguments": raw if isinstance(raw, dict) else None,
                    "parse_ok": bool(name) and isinstance(raw, dict),
                    "has_tool_call": True, "closed": True,
                    "text": action.get("content") or ""}
        action = action.get("content") or ""
    if isinstance(action, dict):
        name = action.get("name")
        args = action.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        return {"name": name, "arguments": args if isinstance(args, dict) else None,
                "parse_ok": bool(action.get("parse_ok", bool(name) and isinstance(args, dict))),
                "has_tool_call": bool(name), "closed": True, "text": ""}
    text = str(action or "")
    parsed = parse_tool_call(text)
    return {
        "name": parsed.get("name"),
        "arguments": parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else None,
        "parse_ok": bool(parsed.get("parse_ok")),
        "has_tool_call": bool(parsed.get("has_tool_call")),
        "closed": bool(parsed.get("closed")),
        "text": text,
    }


def tool_index(tool_schemas: Optional[Sequence[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    """name -> {required, properties, returns} over the session's tool pool.

    Accepts the three shapes our stacks emit: OpenAI
    ``{"type":"function","function":{"name","parameters"}}``, flat
    ``{"name","parameters"}`` and Anthropic-style ``{"name","input_schema"}``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for tool in tool_schemas or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters") or fn.get("input_schema") or {}
        if not isinstance(schema, dict):
            schema = {}
        returns = (fn.get("returns") or fn.get("response") or fn.get("output_schema")
                   or fn.get("return_schema"))
        out[str(name)] = {
            "required": list(schema.get("required") or []),
            "properties": dict(schema.get("properties") or {}),
            "returns": returns if isinstance(returns, dict) else None,
        }
    return out


_JSON_TYPE_CHECK: Dict[str, Callable[[Any], bool]] = {
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def schema_violation(name: Optional[str], args: Any,
                     tool_schemas: Optional[Sequence[Dict[str, Any]]]) -> Optional[str]:
    """First schema violation, or None.  Local re-implementation of the
    bench-branch ``benchmarks/metrics._schema_violations`` semantics
    (name-in-pool, arguments-are-an-object, required keys, declared types).

    Deliberately NOT imported from ``benchmarks/`` (that tree is another
    branch and is never edited or imported from here).  Same degradation rule:
    with no advertised tool pool legality is not computable -> None.
    """
    index = tool_index(tool_schemas)
    if not index:
        return None
    if name is None:
        return "no tool name"
    if name not in index:
        return f"unknown tool name {name!r}"
    if args is None or not isinstance(args, dict):
        return "arguments are not a JSON object"
    entry = index[name]
    for key in entry["required"]:
        if key not in args:
            return f"missing required argument {key!r}"
    for key, value in args.items():
        expected = entry["properties"].get(key)
        if not isinstance(expected, dict):
            continue
        json_type = expected.get("type")
        check = _JSON_TYPE_CHECK.get(json_type)
        if check is not None and not check(value):
            return f"argument {key!r} must be {json_type}"
    return None


# --------------------------------------------------------------------------
# typed value grounding (2607.18245 §2.3), shared by SIEVE / SAAG / Tracy-visible
# --------------------------------------------------------------------------

EXACT = "exact"
PARTIAL = "partial"      # string present in the pool but not verbatim
ASSUMED = "assumed"      # no grounding context -> excluded from VHR

_FLOAT_TOKEN_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_HAS_NUMBER_RE = re.compile(r"\d")


def has_numbers(text: str) -> bool:
    """``HasNumbers(Q)`` of 2607.18245 §2.3."""
    return bool(_HAS_NUMBER_RE.search(text or ""))


def _string_verbatim(value: str, text: str) -> bool:
    """Case-insensitive match on word boundaries (2607.18245 §2.3, strings)."""
    if not value:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(value)}(?!\w)", text or "", re.IGNORECASE))


def _numeric_verbatim(value: float, text: str) -> bool:
    """A numeric is grounded iff it equals a float token of the text
    (2607.18245 §2.3: 'exact match is the only grounding check applied to
    numeric arguments, there is no partial-credit fallback')."""
    for token in _FLOAT_TOKEN_RE.findall(text or ""):
        try:
            if float(token) == float(value):
                return True
        except ValueError:
            continue
    return False


def classify_value(value: Any, text: str) -> str:
    """EXACT / PARTIAL / ASSUMED for one predicted argument value.

    2607.18245 §2.3.  Strings: word-boundary case-insensitive verbatim match,
    else PARTIAL (graded by PartialRatio downstream).  Numerics: float-token
    equality, else PARTIAL when the text contains numbers (h = 1) and ASSUMED
    when it does not.  Lists: EXACT iff every element is EXACT; PARTIAL if any
    element is PARTIAL; ASSUMED otherwise.  Booleans / null / nested objects:
    ASSUMED (declared deviation).
    """
    if isinstance(value, bool) or value is None:
        return ASSUMED
    if isinstance(value, str):
        return EXACT if _string_verbatim(value, text) else PARTIAL
    if isinstance(value, (int, float)):
        if _numeric_verbatim(value, text):
            return EXACT
        return PARTIAL if has_numbers(text) else ASSUMED
    if isinstance(value, list):
        if not value:
            return ASSUMED
        kinds = [classify_value(v, text) for v in value]
        if all(k == EXACT for k in kinds):
            return EXACT
        if any(k == PARTIAL for k in kinds):
            return PARTIAL
        return ASSUMED
    return ASSUMED


def argument_values(args: Optional[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    """Top-level (parameter, value) pairs of the emitted call.

    2607.18245 scores 'predicted values', i.e. the argument assignment, not
    every nested JSON leaf.  Nested structure is handled inside
    ``classify_value`` (lists element-wise, objects ASSUMED).
    """
    if not isinstance(args, dict):
        return []
    return [(str(k), v) for k, v in args.items()]


def argument_bearing(args: Optional[Dict[str, Any]], *, min_chars: int = 1) -> bool:
    """Does this call carry any gradable value at all?

    The digest warns 48/93 C->W rows are tool-name-only; stage 3 of SAAG,
    Tracy's grounding and the UNRESOLVED route are undefined on those rows and
    must be emitted as None rather than as a pass.
    """
    for _, value in argument_values(args):
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str) and len(value.strip()) >= min_chars:
            return True
        if isinstance(value, (list, dict)) and value:
            return True
    return False


# --------------------------------------------------------------------------
# (A) SIEVE Layer 1 — three-valued routing (2512.06716 §3.2)
# --------------------------------------------------------------------------

ROUTE_PASS = "PASS"
ROUTE_DEVIATION = "DEVIATION"
ROUTE_UNRESOLVED = "UNRESOLVED"
ROUTES = (ROUTE_PASS, ROUTE_DEVIATION, ROUTE_UNRESOLVED)


def route_detail(action: Any,
                 tool_schemas: Optional[Sequence[Dict[str, Any]]],
                 visible_text: str,
                 query: str) -> Dict[str, Any]:
    """SIEVE Layer-1 routing with its reason (2512.06716 §3.2).

    * DEVIATION — a call was emitted but does not survive the deterministic
      checks: parse failure, name not in the session tool pool, missing
      required key, or declared-type mismatch.  This is the parse-failure
      baseline (kill line T1) made explicit inside the ladder.
    * UNRESOLVED — structurally legal, but at least one argument VALUE cannot
      be grounded in ``visible_text UNION query``.  SIEVE routes every
      non-deterministically-resolvable binding here, so ASSUMED values count
      as ungrounded (unlike SAAG's VHR, which excludes them).
    * PASS — everything else, including a legal call with no arguments.

    Layer 1 has NO thresholds in the paper and none here.
    """
    parsed = parse_emitted_call(action)
    pool = tool_index(tool_schemas)
    if not parsed["parse_ok"]:
        reason = "no tool call emitted" if not parsed["has_tool_call"] else "parse failure"
        return {"route": ROUTE_DEVIATION, "reason": reason, "parsed": parsed,
                "ungrounded": [], "n_values": 0}
    violation = schema_violation(parsed["name"], parsed["arguments"], tool_schemas)
    if violation is not None:
        return {"route": ROUTE_DEVIATION, "reason": violation, "parsed": parsed,
                "ungrounded": [], "n_values": 0}
    if not pool:
        # no advertised pool: legality is not computable (same degradation as
        # the bench column).  We still run the grounding rung.
        pass
    text = (visible_text or "") + "\n" + (query or "")
    values = argument_values(parsed["arguments"])
    ungrounded = [k for k, v in values if classify_value(v, text) != EXACT]
    if ungrounded:
        return {"route": ROUTE_UNRESOLVED, "reason": f"ungrounded values {ungrounded[:3]}",
                "parsed": parsed, "ungrounded": ungrounded, "n_values": len(values)}
    return {"route": ROUTE_PASS, "reason": None, "parsed": parsed,
            "ungrounded": [], "n_values": len(values)}


def route(action: Any,
          tool_schemas: Optional[Sequence[Dict[str, Any]]],
          visible_text: str,
          query: str) -> str:
    """Three-valued SIEVE Layer-1 route (2512.06716 §3.2).  See ``route_detail``."""
    return route_detail(action, tool_schemas, visible_text, query)["route"]


def escalation_accounting(routes: Sequence[str],
                          *,
                          cpu_ms_total: Optional[float] = None,
                          rescued_steps: Optional[int] = None,
                          cost_per_escalation: Optional[float] = None) -> Dict[str, Any]:
    """SIEVE's escalation ledger (2512.06716 §4.6, Table ``selective_escalation``).

    Reproduces the SHAPE of their accounting — fire rate, escalation rate,
    per-step cost, and cost per RESCUED step (their 'tokens per successful
    task' column) — never their numbers.  ``rescued_steps`` and
    ``cost_per_escalation`` come from the caller's own frozen cost ledger; when
    either is missing the derived cell is None, never a filled-in default.
    """
    n = len(routes)
    fires = sum(1 for r in routes if r != ROUTE_PASS)
    esc = fires  # DEVIATION and UNRESOLVED both escalate (§3.2)
    cost = (cost_per_escalation * esc) if cost_per_escalation is not None else None
    return {
        "n_steps": n,
        "fires": fires,
        "fire_rate": (fires / n) if n else None,
        "escalation_rate": (esc / n) if n else None,
        "cpu_ms_per_step": (cpu_ms_total / n) if (cpu_ms_total is not None and n) else None,
        "escalation_cost_total": cost,
        "rescued_steps": rescued_steps,
        "cost_per_rescued_step": (cost / rescued_steps)
        if (cost is not None and rescued_steps) else None,
    }


def always_on_vs_selective(routes: Sequence[str],
                           *,
                           cost_per_escalation: Optional[float] = None,
                           rescued_selective: Optional[int] = None,
                           rescued_always_on: Optional[int] = None) -> Dict[str, Any]:
    """Always-on-Audit vs Selective comparison scaffolding
    (2512.06716 §4.6: everything held fixed, every step routed to the
    expensive tier).

    Returns both ledgers plus the absorbed share — the analogue of their
    'L1 absorbs 43.1% of Layer-2 calls at zero utility cost'.  Utility is NOT
    computed here (it needs a closed-loop run); the field is carried through
    from the caller so the comparison cannot be reported without it.
    """
    n = len(routes)
    selective = escalation_accounting(routes, rescued_steps=rescued_selective,
                                      cost_per_escalation=cost_per_escalation)
    always = escalation_accounting([ROUTE_DEVIATION] * n, rescued_steps=rescued_always_on,
                                   cost_per_escalation=cost_per_escalation)
    absorbed = (n - selective["fires"]) / n if n else None
    return {"selective": selective, "always_on": always,
            "absorbed_share_of_escalations": absorbed,
            "note": "utility must be supplied by a closed-loop run; not computed here"}


def crosstab_2x3(routes_by_qid: Dict[str, str],
                 labels_by_qid: Dict[str, Optional[int]]) -> Dict[str, Any]:
    """{C->W, C->C} x {PASS, DEVIATION, UNRESOLVED} (digest §4.6, SIEVE entry).

    The ``C->W AND PASS`` cell is the reason a Layer 2 has to exist: those are
    the failures no deterministic check can see.  Denominators are printed with
    the table; rows outside the trigger frame (label None) are excluded and
    counted separately.
    """
    table = {"C->W": {r: 0 for r in ROUTES}, "C->C": {r: 0 for r in ROUTES}}
    n_outside = 0
    for qid, r in routes_by_qid.items():
        y = labels_by_qid.get(qid)
        if y == 1:
            table["C->W"][r] += 1
        elif y == 0:
            table["C->C"][r] += 1
        else:
            n_outside += 1
    n_cw = sum(table["C->W"].values())
    n_cc = sum(table["C->C"].values())
    fires_cw = n_cw - table["C->W"][ROUTE_PASS]
    fires_cc = n_cc - table["C->C"][ROUTE_PASS]
    return {
        "table": table,
        "n_cw": n_cw,
        "n_cc": n_cc,
        "n_outside_trigger_frame": n_outside,
        "coverage": (fires_cw / n_cw) if n_cw else None,
        "false_reset": (fires_cc / n_cc) if n_cc else None,
        "precision": (fires_cw / (fires_cw + fires_cc)) if (fires_cw + fires_cc) else None,
        "cw_and_pass": table["C->W"][ROUTE_PASS],
        "cw_and_pass_note": "the silent-failure cell: no L1 signal exists on these rows",
        # The two 'silent failure' bookkeepings in this module are NOT the same
        # quantity and must never be quoted against each other unaligned.
        "cw_and_pass_definition": (
            "SILENT-L1 = label C->W AND this module's L1 route == PASS, over the "
            "trigger frame (label in {0,1}) of the frozen battery. It is NOT "
            "silent_failure_census()['neither'], which is 'neither a tool error "
            "nor a parse failure' over PROXY REQUEST ROWS with no C->W label at "
            "all. Different denominators, different frames: SPEC §5.1 item 13's "
            "own silent-failure bookkeeping has to be read against this "
            "definition before either number is quoted."),
    }


def dropped_docs_subanalysis(routes_by_qid: Dict[str, str],
                             labels_by_qid: Dict[str, Optional[int]],
                             dropped_nonempty_by_qid: Dict[str, bool]) -> Dict[str, Any]:
    """The pre-registered named sub-analysis on rows with non-empty
    ``dropped_docs`` (2512.06716 §4.3 failure analysis).

    SIEVE's 19 failures all sit where 'the legitimate content is absent from
    the runtime context, leaving no trusted reference' — which is the
    definition of gist compression.  So a context-internal grounding check is
    predicted to MISS systematically exactly on the dropped-doc rows.  This
    registers the prediction as a table, before running.
    """
    out: Dict[str, Any] = {}
    for stratum, want in (("dropped_nonempty", True), ("dropped_empty", False)):
        sub = {q: r for q, r in routes_by_qid.items()
               if bool(dropped_nonempty_by_qid.get(q)) is want}
        out[stratum] = crosstab_2x3(sub, labels_by_qid)
    out["prereg"] = ("prediction: coverage on dropped_nonempty <= coverage on "
                     "dropped_empty (2512.06716 §4.3, 17/19 failures had no "
                     "trusted reference in the runtime context)")
    return out


# --------------------------------------------------------------------------
# (B) SAAG — strict three-stage cascade (2607.18245 §2.1-2.3)
# --------------------------------------------------------------------------

PDS_DRIFT_CUT = 0.8          # 2607.18245 §2.2 (paper-fixed, provenance not stated)
NCS_W_RATIO = 0.6            # 2607.18245 §2.1
NCS_W_JACCARD = 0.4          # 2607.18245 §2.1


def fnem(name: Optional[str], registry_names: Sequence[str]) -> int:
    """Function-Name Exact Match, 2607.18245 §2.1.

    -1 when no name was produced at all (complete parse failure), 0 when the
    name is not in the registry, 1 when it is.
    """
    if not name:
        return -1
    return 1 if str(name) in set(registry_names) else 0


def ncs(name: Optional[str], registry_names: Sequence[str]) -> Optional[float]:
    """Nearest Candidate Score, 2607.18245 §2.1:
    ``max_c 0.6*WRatio(name,c)/100 + 0.4*Jaccard(name,c)`` over candidates
    sharing at least one trigram with the emitted name.

    Defined only when ``FNEM == 0`` (the paper computes it exactly there);
    returns None otherwise.  WRatio is normalised to [0,1] as the card states.
    """
    if not name:
        return None
    if fnem(name, registry_names) != 0:
        return None
    tri = _trigrams(name)
    cands = [c for c in registry_names if _trigrams(c) & tri]
    if not cands:
        return 0.0
    return max(NCS_W_RATIO * (w_ratio(name, c) / 100.0) + NCS_W_JACCARD * jaccard(name, c)
               for c in cands)


def rpr(args: Optional[Dict[str, Any]], required: Sequence[str]) -> Optional[float]:
    """Required-Parameter Recall ``|A_req ∩ P| / |A_req|``, 2607.18245 §2.2.

    ``|A_req| == 0`` is undefined in the paper; we return 1.0 (nothing required
    is trivially covered) and the census reports how many rows are in that
    state.
    """
    if args is None:
        return None
    req = list(required or [])
    if not req:
        return 1.0
    return sum(1 for k in req if k in args) / len(req)


def pds(param: str, admissible: Sequence[str]) -> float:
    """Parameter Drift Score ``max_a WRatio(p,a)/100``, 2607.18245 §2.2."""
    if not admissible:
        return 0.0
    return max(w_ratio(param, a) / 100.0 for a in admissible)


def spr(args: Optional[Dict[str, Any]], admissible: Sequence[str]) -> Optional[float]:
    """Spurious-Parameter Rate ``|{p in P\\A_all : PDS(p) < 0.8}| / |P|``,
    2607.18245 §2.2.  Denominator is ``|P|`` (all emitted parameters), as
    printed in the paper.  ``|P| == 0`` -> 0.0.
    """
    if args is None:
        return None
    keys = list(args.keys())
    if not keys:
        return 0.0
    allowed = set(admissible or [])
    spurious = [p for p in keys if p not in allowed and pds(p, admissible) < PDS_DRIFT_CUT]
    return len(spurious) / len(keys)


def drifted_parameters(args: Optional[Dict[str, Any]],
                       admissible: Sequence[str]) -> List[str]:
    """Out-of-schema parameters with ``PDS >= 0.8`` whose nearest match is not
    itself already emitted (2607.18245 §2.2)."""
    if args is None:
        return []
    allowed = set(admissible or [])
    out = []
    for p in args:
        if p in allowed:
            continue
        if pds(p, admissible) < PDS_DRIFT_CUT:
            continue
        nearest = max(admissible, key=lambda a: w_ratio(p, a)) if admissible else None
        if nearest is not None and nearest not in args:
            out.append(p)
    return out


def avem(args: Optional[Dict[str, Any]], text: str) -> Optional[float]:
    """Argument-Value Exact Match, 2607.18245 §2.3: the fraction of predicted
    values appearing verbatim in the grounding text, type-aware (strings on
    word boundaries case-insensitively, numerics against a float token, lists
    element-wise).  None when the call carries no values (denominator 0)."""
    values = argument_values(args)
    if not values:
        return None
    return sum(1 for _, v in values if classify_value(v, text) == EXACT) / len(values)


def qslo(args: Optional[Dict[str, Any]], text: str) -> Optional[float]:
    """Query-String Lexical Overlap, 2607.18245 §2.3: mean
    ``PartialRatio(v, text)/100`` over STRING parameters that failed exact
    match.  None when there is no such parameter."""
    vals = [v for _, v in argument_values(args)
            if isinstance(v, str) and classify_value(v, text) != EXACT]
    if not vals:
        return None
    return sum(partial_ratio(v, text) / 100.0 for v in vals) / len(vals)


def vhr_ctx(args: Optional[Dict[str, Any]], text: str) -> Dict[str, Any]:
    """Value Hallucination Rate over the contextual subset, 2607.18245 §2.3.

    Per parameter ``h(v, text)``: 0 when exact; ``1 - PartialRatio/100`` for a
    string; 1 for a numeric when ``HasNumbers(text)``; otherwise the parameter
    is 'assumed' and EXCLUDED from both numerator and denominator.  Pass iff
    the sum over the graded subset is 0.

    The excluded fraction is returned (``excluded_frac``) because the exclusion
    is a load-bearing degree of freedom that must be pre-registered and
    reported (card 2607.18245, pitfall 3).
    """
    values = argument_values(args)
    total = 0.0
    n_graded = 0
    n_excluded = 0
    for _, v in values:
        kind = classify_value(v, text)
        if kind == EXACT:
            n_graded += 1
            continue
        if kind == ASSUMED:
            n_excluded += 1
            continue
        n_graded += 1
        if isinstance(v, str):
            total += 1.0 - partial_ratio(v, text) / 100.0
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            total += 1.0
        else:  # list with a failing element
            total += 1.0
    n = len(values)
    return {
        "vhr_sum": total if n_graded else None,
        "vhr_mean": (total / n_graded) if n_graded else None,
        "n_values": n,
        "n_graded": n_graded,
        "n_excluded": n_excluded,
        "excluded_frac": (n_excluded / n) if n else None,
        "pass": (total == 0.0) if n_graded else None,
    }


def saag_cascade(action: Any,
                 tool_schemas: Optional[Sequence[Dict[str, Any]]],
                 grounding_text: str) -> Dict[str, Any]:
    """The strict three-stage cascade, halting at the first failing stage
    (2607.18245 §2.1-2.3).

    Stage 1 passes iff ``FNEM == 1``; stage 2 iff ``RPR == 1 and SPR == 0``;
    stage 3 iff ``VHR_ctx == 0``.  Stage-2 and stage-3 diagnostics are
    CONDITIONAL ON THE SURVIVORS of the earlier stages; every row is stamped
    ``transition on trigger set, not full set`` so that discipline travels with
    the number.  The correction loop is not implemented (see DEVIATIONS).
    """
    parsed = parse_emitted_call(action)
    index = tool_index(tool_schemas)
    registry = list(index.keys())
    name = parsed["name"]
    args = parsed["arguments"]
    row: Dict[str, Any] = {
        "fuzz_backend": fuzz_backend(),
        "stamp": "transition on trigger set, not full set",
        "stage_reached": 1,
        "fnem": fnem(name if parsed["parse_ok"] else None, registry),
        "ncs": None, "rpr": None, "spr": None, "drifted": None,
        "avem": None, "qslo": None, "vhr_sum": None, "vhr_mean": None,
        "vhr_excluded_frac": None, "n_values": None, "n_graded": None,
        "argument_bearing": argument_bearing(args),
        "passed": False, "failed_stage": 1,
    }
    if row["fnem"] != 1:
        row["ncs"] = ncs(name, registry)
        return row
    entry = index.get(str(name), {"required": [], "properties": {}})
    admissible = list(entry["properties"].keys()) or list(entry["required"])
    row["stage_reached"] = 2
    row["rpr"] = rpr(args, entry["required"])
    row["spr"] = spr(args, admissible)
    row["drifted"] = drifted_parameters(args, admissible)
    if row["rpr"] is None or row["spr"] is None or row["rpr"] < 1.0 or row["spr"] > 0.0:
        row["failed_stage"] = 2
        return row
    row["stage_reached"] = 3
    row["avem"] = avem(args, grounding_text)
    row["qslo"] = qslo(args, grounding_text)
    vhr = vhr_ctx(args, grounding_text)
    row.update({"vhr_sum": vhr["vhr_sum"], "vhr_mean": vhr["vhr_mean"],
                "vhr_excluded_frac": vhr["excluded_frac"],
                "n_values": vhr["n_values"], "n_graded": vhr["n_graded"]})
    if vhr["pass"] is None:
        # nothing gradable: stage 3 is undefined, NOT passed (digest: rows with
        # no gradable value must be single-columned, never counted as a pass)
        row["failed_stage"] = None
        row["passed"] = None
        return row
    row["passed"] = bool(vhr["pass"])
    row["failed_stage"] = None if row["passed"] else 3
    return row


def saag_cascade_query_only(action: Any,
                            tool_schemas: Optional[Sequence[Dict[str, Any]]],
                            query: str) -> Dict[str, Any]:
    """The PAPER-FAITHFUL variant of 2607.18245 §2.3: ground argument values in
    the user query ``Q`` alone.

    Kept because it is cheap and because it is the control that shows the
    history version is doing the work: our query is uncompressed and always
    present, so a difference between this and ``saag_cascade`` over the visible
    history is attributable to the compression, not to the rule.
    """
    return saag_cascade(action, tool_schemas, query)


def argument_census(rows: Iterable[Tuple[str, Any, Optional[int]]]) -> Dict[str, Any]:
    """Argument-bearing census BEFORE any stage-3 number is computed.

    ``rows`` is an iterable of ``(qid, action, label)``.  Reports, per label
    stratum, how many rows carry any gradable value at all and how many carry a
    Tracy-gradable value (``len(str(v).strip()) >= 3``) — the digest's warning
    is that 48/93 C->W rows are tool-name-only, in which case stage 3 and the
    grounding features are being computed on an empty set.
    """
    out = {"C->W": {"n": 0, "argument_bearing": 0, "tracy_gradable": 0},
           "C->C": {"n": 0, "argument_bearing": 0, "tracy_gradable": 0},
           "outside": {"n": 0, "argument_bearing": 0, "tracy_gradable": 0}}
    for _qid, action, label in rows:
        key = "C->W" if label == 1 else ("C->C" if label == 0 else "outside")
        parsed = parse_emitted_call(action)
        out[key]["n"] += 1
        if argument_bearing(parsed["arguments"]):
            out[key]["argument_bearing"] += 1
        if argument_bearing(parsed["arguments"], min_chars=3):
            out[key]["tracy_gradable"] += 1
    return out


# --------------------------------------------------------------------------
# (C) must_read_before_write, specialised to gists (2607.07405 §3.3)
# --------------------------------------------------------------------------

#: Pre-registered identifier shape.  Fixed BEFORE looking at the trigger rows
#: (2607.07405 §7(8): gates written from the tasks they are evaluated on is the
#: paper's own admitted weakness).  An identifier-shaped leaf is a token that
#: is not an ordinary word: it mixes digits with letters, or is a long hex /
#: uuid run, or carries an id-ish separator.
IDENTIFIER_RE = re.compile(
    r"^(?:"
    r"[0-9a-fA-F]{8,}"                                  # hex / sha / uuid chunk
    r"|[0-9a-fA-F]{8}-[0-9a-fA-F-]{8,}"                 # dashed uuid
    r"|(?=[^\W\d_]*\d)(?=\d*[^\W\d_])[A-Za-z0-9_.:-]{6,}"  # letters AND digits
    r"|[A-Za-z]+[_:][A-Za-z0-9_:.-]{3,}"                # ns_id / ns:id
    r")$"
)


def identifier_leaves(args: Optional[Dict[str, Any]]) -> List[str]:
    """Identifier-shaped JSON leaves of a proposed action (2607.07405 §3.3).

    Uses ``d_witness_core.leaves`` so the leaf semantics are the frozen ones
    (strings verbatim, numbers via ``str()``, booleans as ``true``/``false``).
    """
    from d_witness_core import leaves  # noqa: PLC0415

    out: List[str] = []
    for leaf in leaves(args if isinstance(args, dict) else {}):
        token = str(leaf).strip()
        if IDENTIFIER_RE.match(token):
            out.append(token)
    return list(dict.fromkeys(out))


@dataclass
class MustReadBeforeWriteGate:
    """``must_read_before_write`` specialised to compression (2607.07405 §3.3).

    The paper's gate blocks a write to a record the agent has not read in this
    session.  Under compression the interesting event is sharper: the record
    WAS in history, but only inside a gisted doc — the evidence for the write
    was compressed away.  So the gate keeps two per-session sets and fires when
    an identifier of the proposed action is in ``gisted`` and not in ``raw``.

    On the full arm every doc is raw, so the gate never fires there: that
    asymmetry is the compression-specific invariant, and it is what makes the
    S0 twin a real control rather than a copy.

    Fail-open by construction: any exception in ``fire`` is the caller's to
    catch; the gate itself never blocks generation (2607.07405 §3.2).
    """

    raw_ids: Set[str] = field(default_factory=set)
    gisted_ids: Set[str] = field(default_factory=set)

    def note_raw_text(self, text: str) -> None:
        """Record identifiers visible in a RAW message / hybrid raw tail."""
        self.raw_ids.update(_identifiers_in_text(text))

    def note_gisted_text(self, text: str) -> None:
        """Record identifiers that exist in history only as a GISTED doc."""
        self.gisted_ids.update(_identifiers_in_text(text))

    def fire(self, args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fire iff some identifier of the action occurs in history ONLY inside
        a gisted doc.  Returns the offending identifiers so the per-signal
        precision audit can be raw counts, never an OR."""
        ids = identifier_leaves(args)
        gist_only = [i for i in ids if i in self.gisted_ids and i not in self.raw_ids]
        return {"fire": bool(gist_only), "identifiers": ids, "gist_only": gist_only}


_ID_TOKEN_RE = re.compile(r"[A-Za-z0-9_.:@-]{4,}")


def _identifiers_in_text(text: str) -> Set[str]:
    return {t for t in _ID_TOKEN_RE.findall(text or "") if IDENTIFIER_RE.match(t)}


def silent_failure_census(proxy_rows: Sequence[Dict[str, Any]],
                          *, arm: Optional[str] = None) -> Dict[str, Any]:
    """Census of compressed-arm failures that are NEITHER a tool error NOR a
    parse failure (2607.07405 §1.1: 78% of observed failures are silent
    wrong-state failures).

    Reads only the proxy request-log columns ``arm / status / error_kind /
    finish_reason``.  This is an internal sizing number: the tau2 airline face
    is CONTAMINATED (31.3% inside the ckpt-1088 training pool) and the caller
    must stamp it as such before it appears anywhere.
    """
    rows = [r for r in proxy_rows if arm is None or r.get("arm") == arm]
    n = len(rows)
    tool_error = 0
    parse_fail = 0
    silent = 0
    for r in rows:
        status = str(r.get("status") or "")
        kind = str(r.get("error_kind") or "")
        finish = str(r.get("finish_reason") or "")
        is_tool_error = bool(kind) or status not in ("", "ok")
        is_parse_fail = finish in ("length", "content_filter") or kind == "parse_error"
        if is_tool_error:
            tool_error += 1
        if is_parse_fail:
            parse_fail += 1
        if not is_tool_error and not is_parse_fail:
            silent += 1
    return {
        "n_rows": n, "tool_error": tool_error, "parse_failure": parse_fail,
        "neither": silent, "silent_share": (silent / n) if n else None,
        "overlap_note": "tool_error and parse_failure OVERLAP (error_kind="
                        "'parse_error' satisfies both); only 'neither' is a "
                        "disjoint count — do not add the three columns",
        "denominator_note": "request rows, not failures: intersect with an "
                            "outcome column before quoting a failure share",
        "not_the_same_as_note": "this 'neither' count is NOT the cross-tab's "
                                "cw_and_pass cell (SILENT-L1): that one is "
                                "label C->W AND route == PASS over the labelled "
                                "trigger frame. Align the definitions (see "
                                "crosstab_2x3()['cw_and_pass_definition'] and "
                                "SPEC §5.1 item 13) before quoting either.",
        "contamination_stamp": "tau2 airline is CONTAMINATED (31.3% in the "
                               "ckpt-1088 training pool) — internal sizing only",
    }


def firing_share_decomposition(fires: Sequence[bool],
                               outcomes: Sequence[float]) -> Dict[str, Any]:
    """``Delta_aggregate ~= p_fire * Delta_fire`` (2607.07405 §3.4).

    Splits an outcome column into the firing and the non-firing stratum and
    reports both means with their n.  The paper's discipline — refuse to claim
    movement on the non-firing stratum when its interval includes zero — is the
    caller's; this returns the strata, never a single product.
    """
    f = [bool(x) for x in fires]
    o = [float(x) for x in outcomes]
    if len(f) != len(o):
        raise ValueError("fires and outcomes must be the same length")
    fired = [o[i] for i in range(len(o)) if f[i]]
    nonfired = [o[i] for i in range(len(o)) if not f[i]]
    n = len(o)
    return {
        "n": n,
        "p_fire": (len(fired) / n) if n else None,
        "firing_stratum": {"n": len(fired),
                           "mean": (sum(fired) / len(fired)) if fired else None},
        "nonfiring_stratum": {"n": len(nonfired),
                              "mean": (sum(nonfired) / len(nonfired)) if nonfired else None},
        "note": "report both strata; the product p_fire * Delta_fire alone is forbidden",
    }


def per_signal_precision_audit(signals: Dict[str, Sequence[bool]],
                               y: Sequence[Optional[int]],
                               *,
                               baselines: Optional[Dict[str, Sequence[bool]]] = None,
                               or_members: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Per-signal precision audit in raw counts (2607.07405 Table
    ``tab:gate-audit``: 161 fires / 100% precision, 90 fires / 78%, ...).

    Each signal is audited alone AND the OR is reported beside them, so the OR
    can never be the only number shown.

    ``baselines`` are audited per signal exactly like the candidates but are
    EXCLUDED from ``_OR``: the frozen parse-failure baseline is gated on
    ``target_has_tool_call`` (a target field, ``t33_labels.py:126``), so an OR
    containing it would be a detector that reads the target (§4.0 item 1).  The
    exclusion is enforced here, not left to the caller.  ``or_members`` overrides
    the OR membership explicitly — pass a baseline name in it to opt in on
    purpose; the members are always printed in ``_OR_members``.
    """
    yy = list(y)
    base = dict(baselines or {})
    out: Dict[str, Any] = {}
    overlap = sorted(set(signals) & set(base))
    if overlap:
        raise ValueError(f"names appear as both candidate and baseline: {overlap}")
    everything: Dict[str, Sequence[bool]] = {**signals, **base}
    for name in sorted(everything):
        col = list(everything[name])
        if len(col) != len(yy):
            raise ValueError(f"signal {name!r} length {len(col)} != labels {len(yy)}")
        fires = sum(1 for i in range(len(col)) if col[i])
        tp = sum(1 for i in range(len(col)) if col[i] and yy[i] == 1)
        fp = sum(1 for i in range(len(col)) if col[i] and yy[i] == 0)
        out[name] = {"fires": fires, "true": tp, "false": fp,
                     "precision": (tp / (tp + fp)) if (tp + fp) else None,
                     "role": "baseline" if name in base else "candidate"}
    if or_members is None:
        members = sorted(signals)
    else:
        members = sorted(or_members)
        unknown = [m for m in members if m not in everything]
        if unknown:
            raise ValueError(f"or_members names no such signal: {unknown}")
    if members:
        ored = [any(bool(everything[n][i]) for n in members) for i in range(len(yy))]
        tp = sum(1 for i in range(len(yy)) if ored[i] and yy[i] == 1)
        fp = sum(1 for i in range(len(yy)) if ored[i] and yy[i] == 0)
        out["_OR"] = {"fires": sum(ored), "true": tp, "false": fp,
                      "precision": (tp / (tp + fp)) if (tp + fp) else None}
    out["_OR_members"] = members
    out["_OR_note"] = ("_OR is the disjunction of _OR_members only; baseline "
                       "rows (role='baseline') read a target field and are the "
                       "comparison line, never members unless named explicitly "
                       "in or_members")
    return out


# --------------------------------------------------------------------------
# (D) deterministic checks (2608.02464 §10)
# --------------------------------------------------------------------------

def required_coverage(action: Any,
                      tool_schemas: Optional[Sequence[Dict[str, Any]]]) -> Optional[bool]:
    """``required_coverage``, re-specified against the SESSION tool schema
    (2608.02464 §10; see DEVIATIONS — the paper's version reads task-side
    ground truth, which we may not touch).

    True iff the emitted call resolves to a declared tool AND every parameter
    that tool declares required is present in the emitted arguments.  None when
    no pool is advertised (legality not computable) or nothing was emitted.
    """
    parsed = parse_emitted_call(action)
    index = tool_index(tool_schemas)
    if not index:
        return None
    if not parsed["parse_ok"] or not parsed["name"]:
        return None
    entry = index.get(str(parsed["name"]))
    if entry is None:
        return False
    args = parsed["arguments"] or {}
    return all(k in args for k in entry["required"])


#: Name-typed heuristic families used ONLY when a tool declares no return
#: schema (documented deviation).  Read-family tools are expected to return
#: content; the heuristic never returns False on an unrecognised name.
#: A read-family verb at a name-segment boundary (start, ``__``, ``.``, ``_``,
#: ``-``) followed by a segment break (end, separator, or a camelCase capital).
#: Deliberately case-sensitive on the verb: ``getting`` must not match, and an
#: unmatched name stays UNKNOWN rather than being guessed.
_READ_FAMILY_RE = re.compile(
    r"(?:^|__|[._-])(?:get|list|search|read|fetch|find|query|show)(?=$|[_.\-A-Z])")
_ERROR_TEXT_RE = re.compile(r"\b(error|exception|traceback|not found|failed|denied|invalid)\b",
                            re.IGNORECASE)


def tool_contract(prev_tool_message: Optional[Dict[str, Any]],
                  tool_name: Optional[str],
                  tool_schemas: Optional[Sequence[Dict[str, Any]]]) -> Optional[bool]:
    """``tool_contract``: does the PREVIOUS step's trailing tool message carry a
    return shape that belongs to that tool's legal return set (2608.02464 §10)?

    One step late by construction (S12).  Two branches:

    * the tool declares a return / response / output schema -> validate the
      parsed content's top-level JSON type and required keys against it;
    * it does not -> a documented name-typed heuristic: for a read-family tool
      (get/list/search/read/fetch/find/query/show) an empty body or an
      error-shaped string is a contract violation; everything else is
      unknown -> None.

    Returns None (unknown), never False, when the return cannot be typed:
    manufacturing False would manufacture the paper's 0-false-positive result.
    """
    if not prev_tool_message:
        return None
    content = prev_tool_message.get("content")
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    text = "" if content is None else str(content)
    entry = tool_index(tool_schemas).get(str(tool_name or ""))
    returns = entry.get("returns") if entry else None
    if isinstance(returns, dict) and returns:
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return returns.get("type") in (None, "string")
        want = returns.get("type")
        check = _JSON_TYPE_CHECK.get(want) if want else None
        if check is not None and not check(parsed):
            return False
        if isinstance(parsed, dict):
            for key in returns.get("required") or []:
                if key not in parsed:
                    return False
        return True
    if tool_name and _READ_FAMILY_RE.search(str(tool_name)):
        if not text.strip():
            return False
        if _ERROR_TEXT_RE.search(text[:400]):
            return False
        return True
    return None


def grounding_standin_for_total_consistency(action: Any, visible_text: str) -> Optional[float]:
    """STAND-IN for ``total_consistency`` (2608.02464 §10) — explicitly named as
    a stand-in, never as the check itself.

    Their ``total_consistency`` recomputes a stated total from the tool results
    the run actually received; that is arithmetic specific to their
    research/calculator family and is NOT portable.  Our analogue is argument
    grounding against the visible decoded history (Tracy's family, SAAG's typed
    rules), returned here so the report can print it under a name that cannot
    be mistaken for the original check.  None when there is nothing gradable.
    """
    parsed = parse_emitted_call(action)
    return avem(parsed["arguments"], visible_text)


def false_positive_counts(fires: Sequence[bool],
                          healthy: Sequence[bool]) -> Dict[str, Any]:
    """False positives as RAW counts with the denominator visible
    (2608.02464 §10: '0/63', '0 of 1825'), never as a rate alone."""
    n_healthy = sum(1 for h in healthy if h)
    fp = sum(1 for i in range(len(fires)) if fires[i] and healthy[i])
    return {"false_positives": fp, "healthy": n_healthy,
            "as_printed": f"{fp}/{n_healthy}",
            "rate": (fp / n_healthy) if n_healthy else None}


# --------------------------------------------------------------------------
# (E) DART AdmissibleRecover (2605.23311 §4.5, Definition 6)
# --------------------------------------------------------------------------

def gist_extraction_consumers(k: int, n_docs: int) -> List[int]:
    """Channel (a): every LATER gist extracted from a prefix containing block
    ``k`` (2605.23311 §4.5, conservative producer-consumer relation).

    Under a left-to-right gist compressor this is ``[k+1 .. n_docs-1]`` for
    every ``k``, so ``NoCommittedConflict(k)`` is false for every block but the
    last and ``A(f)`` is empty always — DART degenerates to whole-task rerun.
    That is the VACUOUS BOUND the digest requires to be reported alongside the
    tight channel, not a usable relation.
    """
    return list(range(int(k) + 1, int(n_docs)))


def execution_consumers(k: int,
                        doc_texts: Sequence[str],
                        later_actions: Sequence[Any],
                        *, mode: str = "occurs") -> List[int]:
    """Channel (b): later EXECUTED steps whose arguments lexically ground in
    block ``k``'s plaintext (2605.23311 §4.5).

    Computed with ``d_witness_core``'s frozen ``sum 1/df(v)`` machinery, with
    the MODEL'S OWN emitted action substituted for the gold action (the one
    deployable substitution the SPEC names and nobody had tried).

    ``mode='occurs'`` — conservative: a step consumes ``k`` when any of its
    values occurs in doc ``k`` (df-weighted score at ``k`` > 0).
    ``mode='argmax'`` — tight: it consumes ``k`` only when ``k`` is the argmax
    of the witness score, i.e. block ``k`` is where the evidence is most
    specific.  Both are reported; neither is chosen on the evaluation rows.
    """
    from d_witness_core import target_values, witness_scores  # noqa: PLC0415

    if mode not in ("occurs", "argmax"):
        raise ValueError("mode must be 'occurs' or 'argmax'")
    out: List[int] = []
    texts = list(doc_texts)
    if not texts or not (0 <= int(k) < len(texts)):
        return out
    for j, action in enumerate(later_actions):
        parsed = parse_emitted_call(action)
        values = target_values(parsed["name"], parsed["arguments"] or {})
        if not values:
            continue
        _df, scores = witness_scores(texts, values)
        if max(scores) <= 0:
            continue
        if mode == "occurs":
            if scores[int(k)] > 0:
                out.append(j)
        else:
            best = max(range(len(scores)), key=lambda i: scores[i])
            if best == int(k):
                out.append(j)
    return out


def no_committed_conflict(k: int,
                          n_docs: int,
                          doc_texts: Sequence[str],
                          later_actions: Sequence[Any],
                          committed: Sequence[bool]) -> Dict[str, Any]:
    """``NoCommittedConflict(k*)`` on both channels (2605.23311 §4.5).

    ``committed[j]`` says whether later step ``j`` actually executed against
    the environment.  Returns the per-channel verdicts; the caller must report
    channel (a) as the vacuous bound and use channel (b) as the tight relation.
    """
    chan_a = gist_extraction_consumers(k, n_docs)
    chan_b_occurs = [j for j in execution_consumers(k, doc_texts, later_actions, mode="occurs")
                     if j < len(committed) and committed[j]]
    chan_b_argmax = [j for j in execution_consumers(k, doc_texts, later_actions, mode="argmax")
                     if j < len(committed) and committed[j]]
    return {
        "channel_a_gist": {"consumers": chan_a, "no_conflict": not chan_a,
                           "bound": "vacuous under a left-to-right compressor"},
        "channel_b_occurs": {"consumers": chan_b_occurs, "no_conflict": not chan_b_occurs},
        "channel_b_argmax": {"consumers": chan_b_argmax, "no_conflict": not chan_b_argmax},
    }


def load_effect_policy(path: Path) -> Dict[str, Any]:
    """Load the FROZEN effect policy (2605.23311 §4.5, ``EffectAllowed``).

    The shipped skeleton carries no tool names: a deployment copies it to
    ``configs/t34/effect_policy_<bench>.json`` and fills ``irreversible`` /
    ``read_only`` from that benchmark's own registry.  Any name not listed is
    ``unknown`` and is treated as irreversible (fail-closed on effects, which is
    the opposite of the gate's fail-open on exceptions — stated explicitly).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.setdefault("irreversible", [])
    data.setdefault("read_only", [])
    # NAMED MISSING INPUT: the shipped skeleton lists no tool names, so every
    # candidate is 'unknown' and every repair is blocked.  That is the correct
    # fail-closed behaviour, but it must never be mistaken for a policy verdict.
    data["is_unfilled_skeleton"] = not (data["irreversible"] or data["read_only"])
    if data["is_unfilled_skeleton"]:
        data["unfilled_note"] = (
            f"{Path(path).name} lists NO tool names: effect_allowed() will block "
            f"every candidate as 'unknown_effect' and no DART admissibility "
            f"panel can be quoted until someone with the bench registry copies "
            f"it to effect_policy_<bench>.json and fills irreversible/read_only")
    return data


def effect_allowed(tool_names: Sequence[str], policy: Dict[str, Any]) -> Dict[str, Any]:
    """``EffectAllowed(I, c)``: no replayed call crosses a disallowed effect
    boundary (2605.23311 §4.5)."""
    irr = set(policy.get("irreversible") or [])
    ro = set(policy.get("read_only") or [])
    blocked = [n for n in tool_names if n in irr]
    unknown = [n for n in tool_names if n not in irr and n not in ro]
    out = {"allowed": not blocked and not unknown,
           "blocked_by_effect": blocked, "unknown_effect": unknown,
           "policy_note": "unknown names are treated as irreversible (fail-closed)"}
    if policy.get("is_unfilled_skeleton"):
        # the verdict is a statement about the POLICY FILE, not about the tools
        out["policy_unfilled"] = True
        out["policy_note"] = ("policy file is the UNFILLED skeleton: every name "
                              "is unknown and every candidate is blocked. This "
                              "is a missing input, not an effect verdict")
    return out


def admissible_recover(*,
                       identified: bool,
                       checkpoint: Any,
                       stable_checkpoints: Sequence[Any],
                       instance_checkpoints: Sequence[Any],
                       conflict: Dict[str, Any],
                       effects: Dict[str, Any],
                       channel: str = "channel_b_occurs") -> Dict[str, Any]:
    """Definition 6 of 2605.23311 §4.5::

        AdmissibleRecover(f, I, c) <=> Identified(f, I) /\\ Stable(c, I)
                                    /\\ ScopeOK(I, c) /\\ NoCommittedConflict(I)
                                    /\\ EffectAllowed(I, c)

    ``Identified`` is an INPUT here and is never computed (deployment-time
    localisation is out of scope).  ``Stable`` and ``ScopeOK`` are membership
    checks against the failed instance's own checkpoint sets, matching the
    paper's 'admissibility is defined over the failed instance's own stable
    checkpoints'.  Emits no ``d_t``: this belongs in the recovery/K table, never
    in the trigger table.
    """
    if channel not in conflict:
        raise KeyError(f"unknown conflict channel {channel!r}")
    stable = checkpoint in list(stable_checkpoints)
    scope_ok = checkpoint in list(instance_checkpoints)
    no_conflict = bool(conflict[channel]["no_conflict"])
    eff_ok = bool(effects.get("allowed"))
    predicates = {"Identified": bool(identified), "Stable": stable, "ScopeOK": scope_ok,
                  "NoCommittedConflict": no_conflict, "EffectAllowed": eff_ok}
    admissible = all(predicates.values())
    reason = None if admissible else sorted(k for k, v in predicates.items() if not v)
    return {"admissible": admissible, "predicates": predicates,
            "blocked_by": reason, "channel": channel,
            "note": "one candidate checkpoint only; the whole-task-rerun "
                    "fallback is decided by select_checkpoint over A(f), not by "
                    "a single inadmissible c"}


def admissible_set(*,
                   identified: bool,
                   candidates: Sequence[Any],
                   stable_checkpoints: Sequence[Any],
                   instance_checkpoints: Sequence[Any],
                   conflict_by_checkpoint: Dict[Any, Dict[str, Any]],
                   effects_by_checkpoint: Dict[Any, Dict[str, Any]],
                   channel: str = "channel_b_occurs") -> List[Any]:
    """``A(f) = { c in C(I_f) : AdmissibleRecover(f, I_f, c) }`` (2605.23311 §4.5).

    Evaluated over the failed instance's OWN candidate checkpoints, which is the
    paper's scoping ("admissibility is defined over the failed instance's own
    stable checkpoints, not over the whole task").
    """
    out: List[Any] = []
    for c in candidates:
        verdict = admissible_recover(
            identified=identified, checkpoint=c,
            stable_checkpoints=stable_checkpoints,
            instance_checkpoints=instance_checkpoints,
            conflict=conflict_by_checkpoint[c],
            effects=effects_by_checkpoint[c],
            channel=channel)
        if verdict["admissible"]:
            out.append(c)
    return out


def select_checkpoint(admissible: Sequence[Any],
                      *, key: Optional[Callable[[Any], Any]] = None) -> Dict[str, Any]:
    """``c*(f) = max A(f)`` — checkpoint RECENCY within the failed instance
    (2605.23311 §4.5).  ``A(f)`` empty => local rollback is rejected and the
    runtime falls back to whole-task rerun.

    ``key`` maps a checkpoint to its recency order when the checkpoint objects
    are not themselves ordered; the default is the identity, matching the
    paper's integer-indexed checkpoints.
    """
    items = list(admissible)
    if not items:
        return {"c_star": None, "n_admissible": 0,
                "fallback": "whole-task rerun (A(f) empty)"}
    c_star = max(items, key=key) if key is not None else max(items)
    return {"c_star": c_star, "n_admissible": len(items), "fallback": None}


def admissibility_panel(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The admitted / blocked / unsafe / false-blocked panel shape
    (2605.23311 Table ``tab:correctness-safety``).

    Shape only.  ``unsafe_admitted`` and ``false_blocked`` need an audit label
    supplied by the caller (``event['unsafe']`` / ``event['ground_truth_ok']``);
    when absent they are reported as None, never as zero.  No number from the
    paper is reproduced here.
    """
    admitted = [e for e in events if e.get("admissible")]
    blocked = [e for e in events if not e.get("admissible")]
    unsafe = [e for e in admitted if e.get("unsafe") is True]
    false_blocked = [e for e in blocked if e.get("ground_truth_ok") is True]
    have_audit = any("unsafe" in e or "ground_truth_ok" in e for e in events)
    reasons: Dict[str, int] = {}
    for e in blocked:
        for r in (e.get("blocked_by") or ["unspecified"]):
            reasons[r] = reasons.get(r, 0) + 1
    return {
        "n_events": len(events),
        "admitted": len(admitted),
        "blocked": len(blocked),
        "unsafe_admitted": len(unsafe) if have_audit else None,
        "false_blocked": len(false_blocked) if have_audit else None,
        "block_reasons": reasons,
        "audit_note": "unsafe/false-blocked require an audit label; None means "
                      "no audit was supplied, not zero",
    }


# --------------------------------------------------------------------------
# (F) Tracy's argument_grounding_score
# --------------------------------------------------------------------------

def is_empty_execute_response(action: Any) -> bool:
    """Verbatim port of ``bfcl_eval...multi_turn_utils.is_empty_execute_response``
    (used by Tracy's rule at ``bfcl_history_kv_repair.py:492``)."""
    if not isinstance(action, list):
        return False
    if len(action) == 0:
        return True
    if len(action) == 1 and len(action[0]) == 0:
        return True
    return False


def argument_grounding_score_verbatim(action: Sequence[Any],
                                      messages: Sequence[Dict[str, Any]]) -> float:
    """Tracy's ``_argument_grounding_score`` ported verbatim
    (``tmp/bfcl-c2kv/c2kv_eval/adapters/bfcl_history_kv_repair.py:484-516``).

    grounding@12msg: ``recent_text`` = lower-cased join of the CONTENT of the
    last 12 messages with ``role in {user, tool, assistant}``; values are the
    str/int/float argument values with ``len(str(v).strip()) >= 3``; hit iff
    ``value.lower() in recent_text``.  Her two edge rules are kept exactly:
    **1.0** when the action parsed but carries no such value, **0.0** when it
    did not parse and is not an empty-execute response.  The 0.34 cut is NOT
    applied here (any cut is chosen in inner folds).
    """
    objs: List[Dict[str, Any]] = []
    for item in action or []:
        value: Any = item
        if isinstance(item, str):
            try:
                value = json.loads(item)
            except Exception:
                value = item
        if isinstance(value, dict):
            objs.append(value)
    if not objs:
        return 1.0 if is_empty_execute_response(list(action or [])) else 0.0
    recent_text = "\n".join(
        str(m.get("content") or "")
        for m in list(messages)[-12:]
        if m.get("role") in {"user", "tool", "assistant"}
    ).lower()
    values: List[str] = []
    for obj in objs:
        args = obj.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"value": args}
        if isinstance(args, dict):
            for value in args.values():
                if isinstance(value, (str, int, float)) and str(value).strip():
                    text = str(value).strip().lower()
                    if len(text) >= 3:
                        values.append(text)
    if not values:
        return 1.0
    hits = sum(1 for v in values if v in recent_text)
    return hits / len(values)


def grounding_visible(action: Any, docs: Sequence[str], query: str) -> Optional[float]:
    """The version we actually want (digest §4.6, Tracy entry).

    ``recent_text`` becomes the decoded plaintext of the VISIBLE compressed
    docs UNION the current query, and the hit rule is upgraded to SAAG's typed
    rules (strings on word boundaries, numerics against a float token, lists
    element-wise) instead of a raw lower-case substring.

    Returns **None** when the call carries no gradable value — the digest is
    explicit that Tracy's 1.0 on that case collapses the denominator on the
    ~48/93 tool-name-only rows.  The verbatim 1.0 rule survives only under
    ``argument_grounding_score_verbatim``.
    """
    parsed = parse_emitted_call(action)
    text = "\n".join(list(docs or [])) + "\n" + (query or "")
    return avem(parsed["arguments"], text)


#: The four states of the dropped-side grounding comparison.  They must never
#: collapse into a single ``None``: "the step dropped nothing" is a fact about
#: the run, "the dump did not carry the dropped text" is a fact about the
#: artifact, and only the second one is a missing input.
DROPPED_DEFINED = "defined"
DROPPED_NO_BLOCKS = "no_blocks_dropped"
DROPPED_TEXT_NOT_DUMPED = "dropped_text_not_dumped"
DROPPED_NO_VALUES = "no_gradable_values"
#: neither the dropped text nor the dropped COUNT was supplied: we cannot
#: even say whether anything was dropped, so nothing may be assumed (this
#: is NOT 'nothing was dropped').
DROPPED_COUNT_UNKNOWN = "dropped_count_unknown"
#: the artifact contradicts itself (len(dropped_doc_texts) != len(dropped_docs)).
DROPPED_TEXT_COUNT_MISMATCH = "dropped_text_count_mismatch"


def grounding_dropped_detail(action: Any,
                             visible_docs: Sequence[str],
                             dropped_doc_texts: Optional[Sequence[str]],
                             query: str,
                             *, n_dropped_blocks: Optional[int] = None) -> Dict[str, Any]:
    """``grounded_in_visible_history`` minus ``grounded_in_dropped_docs``, with
    its state (card 2607.18245, 'The critical adaptation'; digest §4.6).

    Positive means the values are found where the model could see them;
    negative means the evidence for them sits in the blocks the tail window
    dropped — the compression-specific diagnostic.

    ``dropped_doc_texts`` is ``None`` when the sidecar was dumped WITHOUT
    ``--with_dropped_text`` (the dropped side is not retrievable from the
    artifact) and a list (possibly empty) when it was dumped with it.
    ``n_dropped_blocks`` is ``len(sidecar['dropped_docs'])``; those are indices
    into the POST-SPLIT history list, never into ``docs``, so they are used
    here only as a count.

    Four states, reported in ``state`` and summarised by the boolean
    ``dropped_side_available``:

    * ``no_blocks_dropped`` — the step dropped nothing, so
      ``grounded_in_dropped_docs`` is 0 over an empty document set BY
      CONSTRUCTION and the delta is exactly ``grounded_in_visible``.  Defined,
      not missing (declared convention, see DEVIATIONS).
    * ``dropped_text_not_dumped`` — blocks were dropped but their plaintext is
      absent from the artifact.  ``delta`` is None and
      ``dropped_side_available`` is False: a MISSING INPUT, not a zero.
    * ``no_gradable_values`` — the call carries no gradable value, so both
      sides are undefined (the dropped side may still be available).
    * ``dropped_count_unknown`` — neither the dropped texts nor
      ``n_dropped_blocks`` was supplied, so it is not even known whether
      anything was dropped.  A MISSING INPUT, never "nothing was dropped".
    * ``dropped_text_count_mismatch`` — the artifact contradicts itself
      (``len(dropped_doc_texts) != len(dropped_docs)``); the dropped side is
      refused rather than scored over a partial list.
    * ``defined`` — both sides computed.
    """
    parsed = parse_emitted_call(action)
    texts = None if dropped_doc_texts is None else list(dropped_doc_texts)
    unknown_count = n_dropped_blocks is None and texts is None
    n_dropped = int(n_dropped_blocks) if n_dropped_blocks is not None else (
        len(texts) if texts is not None else 0)
    mismatch = (texts is not None and n_dropped_blocks is not None
                and len(texts) != n_dropped)
    available = (not unknown_count) and (not mismatch) and (
        (n_dropped == 0) or (texts is not None))
    vis = avem(parsed["arguments"],
               "\n".join(list(visible_docs or [])) + "\n" + (query or ""))
    out: Dict[str, Any] = {
        "delta": None,
        "grounded_in_visible_history": vis,
        "grounded_in_dropped_docs": None,
        "n_dropped_blocks": None if unknown_count else n_dropped,
        "dropped_side_available": available,
        "state": None,
    }
    if not available:
        out["state"] = (DROPPED_COUNT_UNKNOWN if unknown_count else
                        DROPPED_TEXT_COUNT_MISMATCH if mismatch else
                        DROPPED_TEXT_NOT_DUMPED)
        return out
    if vis is None:
        out["state"] = DROPPED_NO_VALUES
        return out
    if n_dropped == 0:
        # nothing was dropped: no value can be grounded in an empty document
        # set, so grounded_in_dropped_docs is 0.0 by construction.
        out.update({"grounded_in_dropped_docs": 0.0, "delta": vis,
                    "state": DROPPED_NO_BLOCKS})
        return out
    drop = avem(parsed["arguments"], "\n".join(texts or []))
    if drop is None:                                   # unreachable: same args
        out["state"] = DROPPED_NO_VALUES
        return out
    out.update({"grounded_in_dropped_docs": drop, "delta": vis - drop,
                "state": DROPPED_DEFINED})
    return out


def grounding_dropped_delta(action: Any,
                            visible_docs: Sequence[str],
                            dropped_doc_texts: Optional[Sequence[str]],
                            query: str,
                            *, n_dropped_blocks: Optional[int] = None) -> Optional[float]:
    """The ``delta`` cell of ``grounding_dropped_detail`` (see there for every
    state).  None means the dropped side was NOT DUMPED / not countable / the
    artifact disagrees with itself, or the call has no gradable value — read the
    companion flag column ``grounding_dropped_side_available`` (and the
    ``state`` cell of ``grounding_dropped_detail``) to tell those apart.  It is
    NOT None when the step simply dropped nothing: that case is a defined
    delta equal to ``grounded_in_visible_history``."""
    return grounding_dropped_detail(action, visible_docs, dropped_doc_texts, query,
                                    n_dropped_blocks=n_dropped_blocks)["delta"]


def grounding_strata(scores_by_qid: Dict[str, Optional[float]],
                     labels_by_qid: Dict[str, Optional[int]],
                     dropped_nonempty_by_qid: Dict[str, bool]) -> Dict[str, Any]:
    """dropped_docs empty / non-empty strata with the argument-bearing subset
    size reported (digest §4.6, Tracy entry: 'denominator collapse')."""
    out: Dict[str, Any] = {}
    for stratum, want in (("dropped_nonempty", True), ("dropped_empty", False)):
        sel = [q for q in scores_by_qid
               if bool(dropped_nonempty_by_qid.get(q)) is want]
        for lab_name, lab in (("C->W", 1), ("C->C", 0)):
            rows = [q for q in sel if labels_by_qid.get(q) == lab]
            defined = [scores_by_qid[q] for q in rows if scores_by_qid[q] is not None]
            out[f"{stratum}/{lab_name}"] = {
                "n": len(rows),
                "n_defined": len(defined),
                "mean": (sum(defined) / len(defined)) if defined else None,
            }
    return out


# --------------------------------------------------------------------------
# (G) feature emission
# --------------------------------------------------------------------------

#: Feature columns this module writes.  Kept explicit so the orientations file
#: and the leakage guard can be checked against it in the unit test.
FEATURE_COLUMNS = (
    "route_fire", "route_deviation", "route_unresolved",
    "fnem", "ncs", "rpr", "spr",
    "avem", "qslo", "vhr_ctx", "vhr_ctx_mean", "vhr_excluded_frac",
    "avem_query_only",
    "grounding_12msg", "grounding_visible", "grounding_dropped_delta",
    "grounding_dropped_side_available",
    "mrbw_fire", "required_coverage_ok", "dropped_nonempty",
    "argument_bearing", "censored_at_cap", "cpu_ms",
)

#: Columns that are emitted but are NOT candidate triggers: a stratifier, a
#: cost column and a data-availability flag.  They carry a declared orientation
#: only so the scorer's default of +1 cannot silently apply to them; the scorer
#: must drop them from the winner table (digest §4.0: "活下来的信号之间不排序
#: ... 挑进闭环的按成本").  The list is written into the features
#: ``.meta.json`` (key ``non_candidate_columns``) so the scorer has a
#: machine-readable declaration beside the frame rather than a comment.
NON_CANDIDATE_COLUMNS = ("argument_bearing", "censored_at_cap", "cpu_ms",
                         "grounding_dropped_side_available")


#: Pre-declared risk orientation of every emitted feature: +1 = higher is
#: riskier (more likely C->W), -1 = higher is safer.  Declared BEFORE scoring;
#: ``configs/t34/orientations_triggers.json`` is generated from this mapping
#: (``python agent/triggers.py orientations --write``) and the unit test asserts
#: the two agree, so the file can never drift from the rationale.
ORIENTATION_RATIONALE: Dict[str, Tuple[int, str]] = {
    "route_fire": (+1, "SIEVE L1 escalates on DEVIATION or UNRESOLVED (2512.06716 §3.2)"),
    "route_deviation": (+1, "the parse-failure baseline made explicit (kill line T1)"),
    "route_unresolved": (+1, "a value that cannot be grounded is the escalation case"),
    "fnem": (-1, "FNEM=1 is registry-conformant; -1/0 are the failure states (2607.18245 §2.1)"),
    "ncs": (-1, "high NCS = recoverable misspelling, low NCS = fabrication (2607.18245 §2.1)"),
    "rpr": (-1, "RPR=1 is the pass criterion; RPR<1 is a hard failure (2607.18245 §2.2)"),
    "spr": (+1, "SPR>0 fails stage 2 (2607.18245 §2.2)"),
    "avem": (-1, "more values grounded verbatim = safer (2607.18245 §2.3)"),
    "qslo": (-1, "higher lexical overlap for non-exact strings = safer (2607.18245 §2.3)"),
    "vhr_ctx": (+1, "VHR_ctx=0 is the pass criterion; any mass is hallucination (2607.18245 §2.3)"),
    "vhr_ctx_mean": (+1, "the same quantity as a RATE (the paper reports VHR as a rate, "
                         "0.166-0.336); the sum alone is confounded with call arity"),
    "vhr_excluded_frac": (+1, "more 'assumed' values = fewer trusted references, "
                              "the regime SIEVE's failure analysis names (2512.06716 §4.3)"),
    "avem_query_only": (-1, "paper-faithful Q-only variant of AVEM"),
    "grounding_12msg": (-1, "Tracy's score: higher = arguments found in recent text"),
    "grounding_visible": (-1, "same, over the visible decoded compressed history"),
    "grounding_dropped_delta": (-1, "positive = evidence is where the model could see it; "
                                    "negative = evidence sits in the dropped blocks"),
    "grounding_dropped_side_available": (
        -1, "AVAILABILITY FLAG, not a candidate trigger: 1 when the dropped side "
            "of grounding_dropped_delta was computable (nothing dropped, or the "
            "sidecar was dumped with --with_dropped_text), 0 when blocks were "
            "dropped and their text is absent from the artifact"),
    "mrbw_fire": (+1, "write whose identifier exists only inside a gisted doc (2607.07405 §3.3)"),
    "required_coverage_ok": (-1, "coverage of the declared contract = safer (2608.02464 §10)"),
    "dropped_nonempty": (+1, "S8: the tail window dropped blocks on this step"),
    "argument_bearing": (-1, "STRATIFIER, not a candidate trigger: it exists to "
                             "single-column the tool-name-only rows whose stage-3 "
                             "features are undefined"),
    "censored_at_cap": (+1, "STRATIFIER, not a candidate trigger: generation hit "
                            "the frozen 128-token cap, where 'protocol legal' "
                            "degenerates towards 'finished inside the cap' — the "
                            "route columns must be read stratified by it and "
                            "re-run at caliber >= 512 (digest §4.0)"),
    "cpu_ms": (+1, "COST column, not a candidate trigger; excluded from the winner table"),
}

ORIENTATIONS: Dict[str, int] = {k: v[0] for k, v in ORIENTATION_RATIONALE.items()}


def load_sidecar(path: Path) -> Dict[str, Dict[str, Any]]:
    """Read a ``results/t34/sidecar_<arm>.jsonl`` dump into qid -> row.

    Schema (fixed, server-side): ``{qid, session_id, docs, query, tools,
    system_prompt, doc_lengths, dropped_docs, kept_history_tokens}``.

    ``docs`` holds ONLY the KEPT blocks — every entry is text the model saw.
    ``dropped_docs`` holds indices into the POST-SPLIT history list, NOT into
    ``docs``: it may never be used to slice or filter ``docs``.  The dropped
    blocks' plaintext exists only under the opt-in key ``dropped_doc_texts``
    (``t34_dump_sidecar --with_dropped_text``); when that key is ABSENT the
    dropped side is an unavailable input and is reported as None with the flag
    column ``grounding_dropped_side_available = 0``.  The optional extension key
    ``kept_history_indices`` gives the post-split index of each ``docs[i]`` so
    the two index spaces can be aligned (see ``sidecar_history_map``).

    Aborts loudly when the file is absent: the dump is a SERVER-side one-off
    (see the RUNBOOK at the top of this module) and nothing here can synthesise
    it.
    """
    from t33_labels import load_jsonl  # noqa: PLC0415

    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"FATAL: sidecar {path} does not exist. Every grounding feature "
            f"(SAAG stage 3, the SIEVE UNRESOLVED rung, grounding_visible, the "
            f"must_read_before_write identifier sets) reads it and nothing else. "
            f"Run on the SERVER:  python agent/t34_dump_sidecar.py --arm <arm> "
            f"--out {path} [--with_dropped_text]  and copy the file back.")
    out: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(str(path)):
        out[row["qid"]] = row
    return out


def sidecar_history_map(side: Dict[str, Any]) -> Optional[Dict[int, int]]:
    """post-split history index -> index into ``side['docs']``, or None.

    ``docs`` and ``dropped_docs`` live in two different index spaces (see
    ``load_sidecar``).  ``t34_dump_sidecar`` emits the bridge as the extension
    key ``kept_history_indices``; this returns it as a mapping, or None when the
    dump predates the key — in which case the two spaces CANNOT be aligned and
    the caller must treat ``docs`` as visible and ``dropped_doc_texts`` as the
    only handle on the dropped side.
    """
    kept = side.get("kept_history_indices")
    docs = list(side.get("docs") or [])
    if kept is None:
        return None
    kept = list(kept)
    if len(kept) != len(docs):
        raise ValueError(
            f"sidecar kept_history_indices has {len(kept)} entries but docs has "
            f"{len(docs)}: the dump is not co-indexed and must be re-run")
    return {int(h): i for i, h in enumerate(kept)}


def _messages_for_tracy(side: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruct a message list for Tracy's last-12 window from the sidecar.

    The battery is single-step teacher-forced, so the 'conversation' is the
    visible docs (role ``user``, in block order) followed by the current query.
    That is the closest honest analogue of her transcript window; it is a
    DIFFERENT window from hers and is reported as ``grounding_12msg`` computed
    over reconstructed messages, never as a bench-face number.
    """
    msgs = [{"role": "user", "content": d} for d in (side.get("docs") or [])]
    msgs.append({"role": "user", "content": side.get("query") or ""})
    return msgs


def build_feature_rows(battery_rows: Sequence[Dict[str, Any]],
                       sidecar: Dict[str, Dict[str, Any]],
                       *, arm: str = "c2kv",
                       cap_tokens: Optional[int] = None,
                       ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One feature row per battery qid + the side artifacts (routes, details).

    Only the arm's OWN row, its OWN sidecar and prefix scalars are read: no
    target, no gold, no ``tool_name_match``, no field of the other arm.  The
    same function computes the S0 twin when it is handed the full arm's rows
    and the full arm's sidecar with ``arm="full"``.

    ``cap_tokens`` is the run's ``max_new_tokens`` (``FrozenFrame.cap_tokens()``)
    and is used ONLY to emit the ``censored_at_cap`` stratifier from the arm's
    own ``generated_tokens``.  When it is None — or the row does not carry
    ``generated_tokens`` — the column is None, never 0: "not censored" and "we
    do not know the cap" are different statements.

    ``arm`` selects the ``must_read_before_write`` channel assignment
    (2607.07405 §3.3): on the compressed arm every visible history doc is a
    GISTED doc and the only raw channel is the current query, so the gate can
    fire; on the full arm every doc is raw, so it never fires.  That asymmetry
    is the control, not a bug.
    """
    rows: List[Dict[str, Any]] = []
    routes: Dict[str, str] = {}
    details: Dict[str, Any] = {}
    for r in battery_rows:
        qid = r["qid"]
        side = sidecar.get(qid)
        if not side:
            # No sidecar row: there is no tool pool and no visible text, so every
            # check would be computed against empty inputs and would MANUFACTURE
            # values (fnem=0 'name not in registry', route=UNRESOLVED 'nothing
            # grounds').  Emit the row as all-None instead — missingness is kept
            # visible by write_features_jsonl and is what --allow_missing_sidecar
            # promises.
            rows.append({"qid": qid, "session_id": r.get("session_id"),
                         **{c: None for c in FEATURE_COLUMNS}})
            continue
        # SIDECAR CO-INDEXING: ``docs`` holds ONLY the KEPT blocks (in order) —
        # every entry of ``docs`` is text the model saw.  ``dropped_docs`` holds
        # indices into the POST-SPLIT history list, NOT into ``docs``, so it may
        # never be used to slice ``docs``; it is used here as a COUNT only, and
        # the dropped side's text comes from the opt-in ``dropped_doc_texts``.
        # ``kept_history_indices`` (optional extension key) maps docs[i] back to
        # its post-split index for consumers that need the two aligned.
        docs = list(side.get("docs") or [])
        dropped_idx = list(side.get("dropped_docs") or [])
        # key ABSENT -> the dump did not carry the dropped text (missing input);
        # key present but empty -> it was dumped and nothing was dropped.
        dropped_texts = (list(side["dropped_doc_texts"])
                         if "dropped_doc_texts" in side else None)
        query = side.get("query") or ""
        tools = side.get("tools") or []
        visible = "\n".join(docs)
        prediction = r.get("prediction") or ""

        t0 = time.perf_counter()
        det = route_detail(prediction, tools, visible, query)
        cascade = saag_cascade(prediction, tools, visible + "\n" + query)
        # the paper-faithful Q-only control must run through the SAME cascade,
        # so it is defined on exactly the stage-3 survivors the history version
        # is defined on (stages 1-2 do not read the grounding text, so the two
        # survivor sets are identical by construction).  Computing avem(Q)
        # unconditionally would compare a filtered column with an unfiltered one.
        cascade_q = saag_cascade_query_only(prediction, tools, query)
        parsed = det["parsed"]
        g_vis = grounding_visible(prediction, docs, query)
        g_drop_detail = grounding_dropped_detail(prediction, docs, dropped_texts, query,
                                                 n_dropped_blocks=len(dropped_idx))
        # Tracy's rule distinguishes "parsed but no gradable value" (1.0) from
        # "did not parse and is not an empty-execute response" (0.0), so an
        # unparseable row must be handed the RAW text, never an empty list
        # (an empty list is her empty-execute case and scores 1.0).
        tracy_action = ([json.dumps({"name": parsed["name"],
                                     "arguments": parsed["arguments"] or {}})]
                        if parsed["parse_ok"] else [prediction])
        g_12 = argument_grounding_score_verbatim(tracy_action, _messages_for_tracy(side))
        gate = MustReadBeforeWriteGate()
        gate.note_raw_text(query)
        for text in docs:
            if arm == "full":
                gate.note_raw_text(text)
            else:
                gate.note_gisted_text(text)
        mrbw = gate.fire(parsed["arguments"])
        cov = required_coverage(prediction, tools)
        cpu_ms = (time.perf_counter() - t0) * 1000.0
        # censoring stratifier, from the arm's OWN row (no target field): the
        # frozen battery caps generation at 128 tokens, where a DEVIATION route
        # can mean "cut off mid-JSON" rather than "illegal".
        gen_tokens = r.get("generated_tokens")
        censored = (None if (cap_tokens is None or gen_tokens is None)
                    else int(int(gen_tokens) >= int(cap_tokens)))

        routes[qid] = det["route"]
        details[qid] = {"route": det["route"], "reason": det["reason"],
                        "stage_reached": cascade["stage_reached"],
                        "failed_stage": cascade["failed_stage"],
                        "gist_only_identifiers": mrbw["gist_only"],
                        "grounding_dropped_state": g_drop_detail["state"],
                        "n_dropped_blocks": g_drop_detail["n_dropped_blocks"]}
        rows.append({
            "qid": qid,
            "session_id": r.get("session_id"),
            "route_fire": int(det["route"] != ROUTE_PASS),
            "route_deviation": int(det["route"] == ROUTE_DEVIATION),
            "route_unresolved": int(det["route"] == ROUTE_UNRESOLVED),
            "fnem": cascade["fnem"],
            "ncs": cascade["ncs"],
            "rpr": cascade["rpr"],
            "spr": cascade["spr"],
            "avem": cascade["avem"],
            "qslo": cascade["qslo"],
            "vhr_ctx": cascade["vhr_sum"],
            "vhr_ctx_mean": cascade["vhr_mean"],
            "vhr_excluded_frac": cascade["vhr_excluded_frac"],
            "avem_query_only": cascade_q["avem"],
            "grounding_12msg": g_12,
            "grounding_visible": g_vis,
            "grounding_dropped_delta": g_drop_detail["delta"],
            "grounding_dropped_side_available": int(g_drop_detail["dropped_side_available"]),
            "mrbw_fire": int(mrbw["fire"]),
            "required_coverage_ok": (None if cov is None else int(cov)),
            "dropped_nonempty": int(bool(dropped_idx)),
            "argument_bearing": int(argument_bearing(parsed["arguments"])),
            "censored_at_cap": censored,
            "cpu_ms": cpu_ms,
        })
    n_no_sidecar = sum(1 for r in rows if r.get("route_fire") is None)
    n_drop_unavail = sum(1 for r in rows
                         if r.get("grounding_dropped_side_available") == 0)
    n_censored = sum(1 for r in rows if r.get("censored_at_cap") == 1)
    meta = {
        "routes": routes, "details": details, "fuzz_backend": fuzz_backend(),
        "n_rows": len(rows),
        "non_candidate_columns": list(NON_CANDIDATE_COLUMNS),
        # NAMED DATA-AVAILABILITY FLAGS.  Every one of these is a missing INPUT,
        # never a value: the consumer must read them before quoting a column.
        "data_availability": {
            "rapidfuzz_available": fuzz_backend() == "rapidfuzz",
            "fuzz_backend": fuzz_backend(),
            "n_rows_without_sidecar_all_columns_null": n_no_sidecar,
            "n_rows_dropped_text_not_dumped": n_drop_unavail,
            "dropped_side_available_everywhere": n_drop_unavail == 0,
            "cap_tokens": cap_tokens,
            "censoring_stratifier_available": cap_tokens is not None,
            "n_rows_censored_at_cap": n_censored if cap_tokens is not None else None,
            "caliber_note": "the frozen battery is 128-token capped; the digest "
                            "requires caliber >= 512 for the L1 ladder. Read "
                            "every route column stratified by censored_at_cap "
                            "and do not offer it as evidence until it is re-run "
                            "at the required caliber.",
            "note": "grounding_dropped_delta is null on the "
                    "n_rows_dropped_text_not_dumped rows because the sidecar was "
                    "dumped without --with_dropped_text, NOT because the step "
                    "dropped nothing (that case is a defined delta); "
                    "n_rows_without_sidecar_all_columns_null rows carry no "
                    "feature at all and are excluded from every audit",
        },
    }
    return rows, meta


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _load_frame(root: Path):
    from t34_common import FrozenAssets  # noqa: PLC0415
    return FrozenAssets(root).load()


def _cmd_census(args: argparse.Namespace) -> int:
    frame = _load_frame(Path(args.root))
    labels = frame.label_by_qid
    c2kv = frame.c2kv_by_qid
    rows = [(qid, c2kv[qid].get("prediction") or "", labels.get(qid)) for qid in c2kv]
    out = argument_census(rows)
    out["fuzz_backend"] = fuzz_backend()
    text = json.dumps(out, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


def _cmd_build_features(args: argparse.Namespace) -> int:
    from t34_common import write_features_jsonl  # noqa: PLC0415

    frame = _load_frame(Path(args.root))
    battery = frame.c2kv_by_qid if args.arm == "c2kv" else frame.full_by_qid
    sidecar = load_sidecar(Path(args.sidecar))
    missing = sorted(set(battery) - set(sidecar))
    if missing and not args.allow_missing_sidecar:
        raise SystemExit(f"FATAL: {len(missing)} qids missing from the sidecar "
                         f"(e.g. {missing[:3]}); pass --allow_missing_sidecar to "
                         f"score them as all-None")
    rows, meta = build_feature_rows(list(battery.values()), sidecar, arm=args.arm,
                                    cap_tokens=frame.cap_tokens())
    n = write_features_jsonl(Path(args.out), rows, context=f"t34 §4.6 L1 ({args.arm})")
    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(
        {"arm": args.arm, "n_rows": n, "fuzz_backend": meta["fuzz_backend"],
         "n_missing_sidecar": len(missing), "missing_sidecar_qids": missing[:50],
         # machine-readable declarations the scorer must read beside the frame
         "non_candidate_columns": meta["non_candidate_columns"],
         "data_availability": meta["data_availability"],
         "routes": meta["routes"],
         "details": meta["details"], "deviations": DEVIATIONS},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"out": args.out, "n_rows": n, "arm": args.arm,
                      "fuzz_backend": meta["fuzz_backend"],
                      "n_missing_sidecar": len(missing),
                      "non_candidate_columns": meta["non_candidate_columns"],
                      "data_availability": meta["data_availability"]}, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from t33_labels import load_jsonl  # noqa: PLC0415

    frame = _load_frame(Path(args.root))
    labels = frame.label_by_qid
    feats = {r["qid"]: r for r in load_jsonl(args.features)}
    meta_path = Path(args.features).with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    routes = meta.get("routes") or {}
    if not routes:
        raise SystemExit(f"FATAL: {meta_path} missing; re-run build-features")
    dropped = {q: bool(feats[q].get("dropped_nonempty")) for q in feats}
    trigger_qids = [q for q in feats if labels.get(q) in (0, 1)]
    # Rows whose sidecar was missing carry NO feature at all (every column is
    # null) and no route.  They are counted, never silently routed as PASS and
    # never counted as a non-firing detector row: every audit below runs on
    # ``routed_qids`` so all of them share one denominator.
    routed_qids = [q for q in trigger_qids if q in routes]
    y = [labels[q] for q in routed_qids]
    signals = {
        # NOT the frozen parse-failure baseline: this one is target-free and
        # also fails on name-not-in-pool / required-key / type (see DEVIATIONS).
        "route_deviation_parse_or_schema": [feats[q]["route_deviation"] == 1
                                            for q in routed_qids],
        "route_unresolved": [feats[q]["route_unresolved"] == 1 for q in routed_qids],
        "route_any_fire": [feats[q]["route_fire"] == 1 for q in routed_qids],
        "mrbw_fire": [feats[q]["mrbw_fire"] == 1 for q in routed_qids],
        "required_coverage_fail": [feats[q].get("required_coverage_ok") == 0
                                   for q in routed_qids],
    }
    # the frozen baseline itself, from the label frame (it reads a target field
    # and therefore may never live in the feature frame NOR in the OR)
    baseline = {r["qid"]: bool(r["parse_fail_fire"]) for r in frame.labels}
    baselines = {"frozen_parse_fail_baseline": [baseline[q] for q in routed_qids]}
    audit = per_signal_precision_audit(signals, y, baselines=baselines)
    # censoring stratifier (digest §4.0 caliber caveat): the route columns must
    # be read stratified by it on the 128-token-capped frozen battery.
    censored = {q: feats[q].get("censored_at_cap") for q in routed_qids}
    have_censoring = any(v is not None for v in censored.values())
    censoring_source = "feature frame (this arm's own generated_tokens)"
    if not have_censoring:
        # older feature frames predate the column; the frozen label frame
        # carries the same quantity for the COMPRESSED arm (t33_labels.py:123),
        # and it is target-free, so it is a legal fallback as long as the
        # substitution is named.
        label_censor = {r["qid"]: (1 if r.get("censored_at_cap") else 0)
                        for r in frame.labels}
        if any(q in label_censor for q in routed_qids):
            censored = {q: label_censor.get(q) for q in routed_qids}
            have_censoring = any(v is not None for v in censored.values())
            censoring_source = "label frame (compressed arm, t33_labels)"
    crosstab_by_censoring = None
    if have_censoring:
        crosstab_by_censoring = {
            name: crosstab_2x3({q: routes[q] for q in routed_qids
                                if censored[q] == want}, labels)
            for name, want in (("censored_at_cap", 1), ("uncensored", 0))
        }
    report = {
        "n_rows": len(feats),
        "n_trigger_rows": len(trigger_qids),
        "n_trigger_rows_without_route": len(trigger_qids) - len(routed_qids),
        "audit_denominator_note": "every audit below is over the "
                                  "feature-defined trigger rows (n = "
                                  f"{len(routed_qids)}); rows with no sidecar "
                                  "carry null features and are excluded rather "
                                  "than counted as non-firing",
        "prevalence": (sum(y) / len(y)) if y else None,
        "fuzz_backend": fuzz_backend(),
        "non_candidate_columns": list(NON_CANDIDATE_COLUMNS),
        "data_availability": meta.get("data_availability"),
        "crosstab_2x3": crosstab_2x3({q: routes[q] for q in routed_qids}, labels),
        "crosstab_2x3_by_censoring": crosstab_by_censoring,
        "censoring_source": censoring_source if have_censoring else None,
        "n_censoring_unknown": sum(1 for q in routed_qids if censored[q] is None),
        "censoring_note": ("censored_at_cap is missing from the feature frame; "
                           "the route columns cannot be read free of the "
                           "128-token censoring confound until it is emitted"
                           if not have_censoring else
                           "DEVIATION on a censored row can mean 'cut off "
                           "mid-JSON'; compare the two strata before quoting "
                           "the route, and re-run at caliber >= 512"),
        "dropped_docs_subanalysis": dropped_docs_subanalysis(
            {q: routes[q] for q in routed_qids}, labels, dropped),
        "escalation": escalation_accounting(
            [routes[q] for q in routed_qids],
            cpu_ms_total=sum(feats[q].get("cpu_ms") or 0.0 for q in routed_qids)),
        "always_on_vs_selective": always_on_vs_selective([routes[q] for q in routed_qids]),
        "per_signal_precision_audit": audit,
        # 2608.02464 §10 reporting discipline: false positives as raw counts
        # with the denominator visible (their '0/63'), never as a rate alone.
        # 'healthy' here is the C->C stratum (label 0).
        "false_positives_raw": {
            name: false_positive_counts(sig, [lab == 0 for lab in y])
            for name, sig in {**signals, **baselines}.items()
        },
        "grounding_strata_visible": grounding_strata(
            {q: feats[q].get("grounding_visible") for q in routed_qids}, labels, dropped),
        "grounding_strata_12msg": grounding_strata(
            {q: feats[q].get("grounding_12msg") for q in routed_qids}, labels, dropped),
        "argument_bearing_subset": {
            "C->W": sum(1 for q in routed_qids
                        if labels[q] == 1 and feats[q].get("argument_bearing")),
            "C->C": sum(1 for q in routed_qids
                        if labels[q] == 0 and feats[q].get("argument_bearing")),
        },
        "deviations": DEVIATIONS,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("n_rows", "n_trigger_rows", "n_trigger_rows_without_route",
                       "prevalence", "fuzz_backend", "data_availability")}, indent=2))
    return 0


def _cmd_orientations(args: argparse.Namespace) -> int:
    """Write configs/t34/orientations_triggers.json from ORIENTATION_RATIONALE.

    The file must stay FLAT ``{name: int}`` — ``t34_common.load_orientations``
    and ``t34_score.merged_orientations`` both do ``int(v)`` on every value, so
    a nested notes block would crash them.  The rationale therefore lives in
    the module, and this command is the only writer.
    """
    path = Path(args.out) if args.out else Path(args.root) / "configs/t34/orientations_triggers.json"
    text = json.dumps(ORIENTATIONS, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    if args.write:
        path.parent.mkdir(parents=True, exist_ok=True)
        with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    print(text)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="t34 §4.6 deterministic L1 checks (SIEVE / SAAG / gates / "
                    "DART / Tracy). Zero GPU, CPU strings only.")
    parser.add_argument("--root", default=".", help="worktree root (frozen assets live under it)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("census", help="argument-bearing census (run BEFORE stage 3)")
    p.add_argument("--sidecar", default=None, help="unused; census reads predictions only")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=_cmd_census)

    p = sub.add_parser("build-features", help="emit results/t34/features_l1.jsonl")
    p.add_argument("--arm", choices=("c2kv", "full"), default="c2kv")
    p.add_argument("--sidecar", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow_missing_sidecar", action="store_true",
                   help="emit qids with no sidecar row as ALL-None (every "
                        "sidecar-dependent column null, no route recorded) "
                        "instead of aborting; they are excluded from every "
                        "audit in `report`, never counted as non-firing")
    p.set_defaults(fn=_cmd_build_features)

    p = sub.add_parser("orientations", help="print / write orientations_triggers.json")
    p.add_argument("--out", default=None)
    p.add_argument("--write", action="store_true")
    p.set_defaults(fn=_cmd_orientations)

    p = sub.add_parser("report", help="2x3 cross-tab, escalation ledger, audits")
    p.add_argument("--features", required=True)
    p.add_argument("--sidecar", default=None,
                   help="unused; the report reads the feature frame and its "
                        "sibling .meta.json (routes / data_availability) only")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=_cmd_report)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
