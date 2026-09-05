# -*- coding: utf-8 -*-
"""t34 §4.12 — LLM-judge detector: MIRAGE-Bench judge protocol on frozen battery
rows, ported from Tracy's ``judge_detector_segments.py``, plus the
Verify-when-Uncertain judge cascade that prices it.

RUNBOOK ("SERVER" needs the model endpoint; everything else runs HERE)
----------------------------------------------------------------------
0. HERE   python agent/t34_judge.py sample-validation --manifest ... --n 160 \
              --out configs/t34/judge_human_validation.json
          # seeded human-validation subset, drawn BEFORE any judge runs.
1. SERVER (or wherever the endpoint is reachable)
          python agent/t34_judge.py run --battery-c2kv ... --battery-full ... \
              --manifest ... --sidecar results/t34/sidecar_c2kv.jsonl \
              --base-url http://127.0.0.1:30000 --model <big-judge> \
              --judge-name o4mini_like --out results/t34/judge_rows.jsonl
2. SERVER same command with our own 4B behind the endpoint and
          --judge-name small_judge_qwen3_4b     # the mandatory small-judge variant
3. HERE   python agent/t34_judge.py score --judge-rows results/t34/judge_rows.jsonl \
              --battery-c2kv ... --battery-full ... --manifest ... \
              --features-out results/t34/features_judge.jsonl \
              --out results/t34/judge_score.json
4. HERE   python agent/t34_judge.py cascade --judge-rows ... --battery-* ... \
              --manifest ... --out results/t34/judge_cascade.json
          # sweeps p in {0,.05,.1,.2,.4,.8,1}: cost first, AUROC second.

Papers implemented here
-----------------------
* 2507.21017 (MIRAGE-Bench) §4.2 freeze-frame contextual snapshot, §4.3 the
  two-step zero-shot judge prompt, the Utility(c) in {1, 0.5, 0} rubric with the
  explicit incomplete middle grade, and the US / HR aggregates.
* 2502.15845 (Verify when Uncertain) Algorithm 1: the (t1, t*, p, t2) cascade,
  with ``p`` — the fraction of instances that land in the uncertainty band — as
  the budget dial.  The rule itself is implemented ONCE on this branch, in
  ``agent/t34_cascade.py``; ``cascade_star_threshold`` / ``cascade_decide`` /
  ``cascade_apply`` / ``fit_cascade`` / ``cascade_sweep`` here are thin
  (lazily-importing) wrappers over it, so the judge cascade and the U6 cascade
  cannot drift apart.

WIRING (bench face; those files are on another branch and are NOT edited here)
-----------------------------------------------------------------------------
The judge is a HOOK 2 consumer: it reads the compressed context summary plus the
action that already exists.  Call ``judge_one`` from
``tmp/bench-recover/benchmarks/proxy.py`` right before ``RecoverState.check``
(class :476, method :494), building the segment from the assembled request's
compressed messages and ``action_canonical`` (:466); write
``judge_seconds`` / ``judge_prompt_tokens`` / ``judge_completion_tokens`` into
the request log through ``_log_request`` (:1285) so the judge's cost lands in
the frozen GPU-sec sum instead of being invisible.  Nothing here edits
``proxy.py``, ``metrics.py`` or ``arms.py``.

LABEL DISCIPLINE.  The judge's score is a FEATURE.  The label stays ``d_cw_v1``
(C->W under ``tool_name_match``, computed in ``t33_labels``).  Importing the
judge's own grade as a label would import 0.756-0.769-accuracy noise into the
target; :func:`assert_segment_is_feature_side` is the mechanical form of that.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import t34_common as C  # noqa: E402
from t33_labels import load_jsonl  # noqa: E402

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "MIRAGE judge input",
        "paper": "2507.21017 §4.2",
        "what": "The snapshot is instruction + history + observation of a live "
                "browser/OS agent.  Ours is a compressed-prefix SUMMARY built "
                "from the sidecar's decoded doc texts plus the compressed arm's "
                "own prediction; there is no observation (the battery is "
                "single-step teacher-forced, S12 is unavailable).",
        "why": "The battery has no environment step; the digest specifies the "
               "battery-row adapter replacing Tracy's ``_compact_segment``.",
    },
    {
        "method": "MIRAGE judge sees CoT",
        "paper": "2507.21017 §4.3",
        "what": "The paper's judge reads the agent's interleaved [thinking] and "
                "[action] and argues CoT is what makes the judgement possible.  "
                "Our arm runs with enable_thinking False, so the judge sees the "
                "action only.",
        "why": "The frozen battery rows carry no thinking trace; SPEC §5.5.9's "
               "determinism gate keeps thinking off.",
    },
    {
        "method": "US / HR",
        "paper": "2507.21017 §4.3",
        "what": "US and HR are computed and reported, but they are NOT the "
                "trigger metrics; the three metrics stay coverage / precision / "
                "false-reset against the parse-failure baseline.",
        "why": "US/HR are prevalence rates on a curated set with no precision, "
               "recall or false-positive budget (card: 'Do NOT import US/HR').",
    },
    {
        "method": "US / HR denominator",
        "paper": "2507.21017 §4.3 (US = (1/|C|) sum Utility(c); "
                 "HR = (1/|C|) sum 1[Utility(c)=0])",
        "what": "The paper divides by |C|, every snapshot context.  We divide "
                "by the GRADED rows and report ``n_ungraded`` beside them: an "
                "unparsed judge reply has no Utility and is not silently "
                "scored 0 (which would inflate HR) or 1 (which would inflate "
                "US).",
        "why": "The paper never has an ungraded case to rule on.  Imputing "
               "either end would be a sentinel; the split denominator is "
               "reported so both readings are recoverable.",
    },
    {
        "method": "cascade candidate grids",
        "paper": "2502.15845 §6 (grid not specified; fitted on a separate "
                 "400-sample validation split)",
        "what": "The search runs over quantile LEVELS (t34_cascade.Q1_GRID / "
                "Q2_GRID), and every level is materialised into an absolute "
                "(t1, t2) on the calibration rows of the fold that is being "
                "fitted — never on quantiles of the whole 161-row frame.",
        "why": "A grid quantiled over the whole frame lets the evaluation "
               "rows' score distribution pick the values that get tried, which "
               "is selection on evaluation rows even though the selection "
               "itself is out-of-fold (§4.0).",
    },
    {
        "method": "cascade implementation",
        "paper": "2502.15845 Algorithm 1",
        "what": "``cascade_star_threshold`` / ``cascade_decide`` / "
                "``cascade_apply`` / ``fit_cascade`` / ``cascade_sweep`` are "
                "thin wrappers over agent/t34_cascade.py; Algorithm 1 is "
                "implemented once on this branch.  The wrappers inherit that "
                "module's band construction at the edges (p = 0 puts t* just "
                "below t1; an over-large p clamps to the largest score instead "
                "of +inf) and its stage vocabulary.",
        "why": "Two implementations of the same (t1, t*, p, t2) rule with "
               "different selection objectives cannot both be cited, and the "
               "cost curve would depend on which one produced it.",
    },
    {
        "method": "judge harmful_probability range",
        "paper": "2507.21017 §4.3 (asks for a probability; states no range "
                 "check)",
        "what": "The field is RANGE-CHECKED onto [0, 1].  A value outside it "
                "is rescaled only on explicit percent evidence (a '%' in the "
                "emitted value), otherwise it is None with the verbatim string "
                "kept in ``judge_score_raw`` and the verdict in "
                "``judge_score_scale``.  Tracy's judge_detector_segments.py "
                "(:103-105) stores the field verbatim.",
        "why": "AP/AUROC are rank-based and survive a percent scale, but a "
               "threshold or any absolute reading does not; dividing every "
               "out-of-range number by 100 would invent a scale the reply "
               "never stated.",
    },
    {
        "method": "MIRAGE grading scale",
        "paper": "2507.21017 §4.3 vs tab:llm-judge-eval-example",
        "what": "Only the {1, 0.5, 0} Utility rubric is implemented; the worked "
                "example's 0/1/2 rubric is NOT, and the two are not reconciled "
                "by the paper.",
        "why": "Implementing an unreconciled second scale would invent a "
               "mapping the source does not state.",
    },
    {
        "method": "Verify-when-Uncertain cascade",
        "paper": "2502.15845 Algorithm 1",
        "what": "Stage 1 is a deterministic free signal (parse failure alone, "
                "the baseline every candidate must beat) instead of "
                "MPD(M_self) over m=10 sampled answers; stage 2 is the judge "
                "score instead of MPD(M_cross) from a verifier model.",
        "why": "m=10 samples at tau'=1.0 breaks the determinism gate and costs "
               "~111 GPU-s/step (card); the digest keeps the architecture and "
               "the p budget dial, not the signals.",
    },
    {
        "method": "cascade threshold fitting",
        "paper": "2502.15845 §6 (400-sample validation set, 5 seeds)",
        "what": "t1 / t2 are fitted in INNER folds of session-grouped nested CV "
                "on our 161-row frame, and the number of thresholds tried is "
                "reported; there is no separate 400-sample validation split.",
        "why": "We have ~100 session clusters on the trigger subset; a held-out "
               "validation split of the paper's size does not exist, and "
               "selecting on evaluation rows is forbidden (§4.0).",
    },
    {
        "method": "judge determinism",
        "paper": "2507.21017 §4.3",
        "what": "Temperature is pinned to 0 with no sampled variant, and the "
                "self-consistency figure (0.849 / 0.819 at T=1 against its own "
                "T=0) is reported as a caveat rather than reproduced.",
        "why": "SPEC §5.5.9 determinism gate; reproducing T=1 would break it.",
    },
]

# --------------------------------------------------------------------------
# HTTP client (OpenAI-compatible; lazy import, injectable for tests)
# --------------------------------------------------------------------------

#: Tracy's frozen generation contract (bfcl-c2kv/c2kv_eval/analysis/
#: judge_detector_segments.py) — kept verbatim.
JUDGE_GEN = {"temperature": 0, "max_completion_tokens": 256,
             "chat_template_kwargs": {"enable_thinking": False}}


class JudgeClient:
    """Minimal OpenAI-compatible chat client (``urllib`` imported lazily).

    Tests never touch the network: pass ``post_fn`` (any callable
    ``(url, payload, timeout) -> dict``) to inject a mock.
    """

    def __init__(self, base_url: str, model: str, *, timeout: int = 300,
                 post_fn: Optional[Callable[[str, Dict[str, Any], int], Dict[str, Any]]] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._post_fn = post_fn

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url + "/v1/chat/completions"
        if self._post_fn is not None:
            return self._post_fn(url, payload, self.timeout)
        import urllib.request  # lazy: never imported in the pure-python tests
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def complete(self, prompt: str) -> Dict[str, Any]:
        """One judge generation; returns text + the three cost columns."""
        payload = dict(JUDGE_GEN)
        payload.update({"model": self.model,
                        "messages": [{"role": "user", "content": prompt}]})
        start = time.perf_counter()
        data = self._post(payload)
        elapsed = time.perf_counter() - start
        text = (((data.get("choices") or [{}])[0] or {}).get("message", {})
                .get("content") or "")
        usage = data.get("usage") or {}
        return {
            "text": text,
            "judge_seconds": elapsed,
            "judge_prompt_tokens": usage.get("prompt_tokens"),
            "judge_completion_tokens": usage.get("completion_tokens"),
        }


# --------------------------------------------------------------------------
# battery-row adapter (replaces Tracy's _compact_segment)
# --------------------------------------------------------------------------

#: Fields a judge segment may read from a frozen battery row.  Everything else
#: — target, target_tool_name, tool_name_match, exact_match, any full-arm row —
#: is label side and refused.
SEGMENT_ALLOWED_ROW_FIELDS: Tuple[str, ...] = (
    "qid", "session_id", "prediction", "doc_chunks", "gist_tokens",
    "kept_history_tokens", "actual_compression_ratio", "generated_tokens",
    "prompt_tokens", "hybrid_top_k",
)


def assert_segment_is_feature_side(fields_used: Sequence[str]) -> None:
    """Mechanical assertion that the judge segment reads no label-side field."""
    from t33_labels import guard_columns
    guard_columns([f for f in fields_used if f not in ("qid", "session_id")],
                  context="t34_judge segment")
    bad = sorted(set(fields_used) - set(SEGMENT_ALLOWED_ROW_FIELDS))
    if bad:
        raise ValueError(f"judge segment may not read {bad}")


#: The sidecar key that carries the DROPPED blocks' text.  It is written ONLY
#: under ``--with_dropped_text`` (agent/t34_dump_sidecar.py:31-34, :180), so it
#: is absent from a default dump and its absence is a data-availability fact,
#: not a reason to substitute anything.
DROPPED_TEXT_KEY = "dropped_doc_texts"


def segment_history_entries(sidecar: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """History blocks for the judge segment, in POST-SPLIT history order.

    SIDECAR CONTRACT (agent/t34_dump_sidecar.py:19-33 and :286-322).  ``docs``
    holds ONLY the KEPT blocks — the decoded grid rows the model actually saw,
    in order — while ``dropped_docs`` holds indices into the POST-SPLIT history
    list, NOT into ``docs``.  Therefore:

    * every entry of ``docs`` is visible to the model (``available`` True);
    * the dropped blocks are exactly the ones ABSENT from ``docs``, keyed by
      their post-split history index;
    * a dropped block's text exists only under :data:`DROPPED_TEXT_KEY`; when
      the dump did not carry it the entry is rendered ``available: False`` with
      ``text: None`` — never reconstructed from ``docs``.

    ``enumerate(docs)`` filtered by ``dropped_docs`` — the shape this function
    replaces — is wrong in both directions: it marks kept blocks unavailable and
    it never shows the reader that a block is missing at all.
    """
    sc = sidecar or {}
    docs = list(sc.get("docs") or [])
    dropped = sorted({int(d) for d in (sc.get("dropped_docs") or [])})
    dropped_set = set(dropped)
    texts = sc.get(DROPPED_TEXT_KEY)
    have_text = isinstance(texts, (list, tuple)) and len(texts) == len(dropped)
    text_by_index = dict(zip(dropped, texts)) if have_text else {}
    n_split = max(len(docs) + len(dropped), (dropped[-1] + 1) if dropped else 0)
    kept_idx = [i for i in range(n_split) if i not in dropped_set]
    entries = [{"history_index": int(i), "earlier_turn": int(i) + 1,
                "available": True, "text": t}
               for i, t in zip(kept_idx, docs)]
    entries += [{"history_index": int(i), "earlier_turn": int(i) + 1,
                 "available": False, "text": text_by_index.get(i)}
                for i in dropped]
    entries.sort(key=lambda e: e["history_index"])
    return entries


def dropped_text_availability(sidecar: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Named flag for the dropped side of the history.

    ``dropped_doc_texts_available`` is None when the row dropped nothing (the
    question is undefined), False when blocks were dropped but the sidecar was
    dumped without ``--with_dropped_text`` (the judge sees the gap named but not
    its content), True when the text is there.
    """
    sc = sidecar or {}
    dropped = sorted({int(d) for d in (sc.get("dropped_docs") or [])})
    texts = sc.get(DROPPED_TEXT_KEY)
    have = isinstance(texts, (list, tuple)) and len(texts) == len(dropped)
    return {
        "n_dropped_blocks": len(dropped),
        "n_visible_blocks": len(list(sc.get("docs") or [])),
        "dropped_doc_texts_available": (bool(have) if dropped else None),
    }


def compact_battery_segment(row: Dict[str, Any],
                            sidecar: Optional[Dict[str, Any]] = None,
                            *, max_doc_chars: int = 400,
                            max_docs: int = 16) -> str:
    """Battery-row adapter for the judge prompt (2507.21017 §4.2 freeze-frame).

    Renders: the current query, a compressed-prefix summary (one line per
    history block in post-split order, kept blocks with their decoded text and
    dropped blocks named as unavailable — see :func:`segment_history_entries`),
    and the compressed arm's emitted action.  Reads only
    :data:`SEGMENT_ALLOWED_ROW_FIELDS` from the battery row.

    ``max_docs`` caps the number of rendered BLOCK LINES (kept and dropped
    together), so the cap never silently hides the fact that a block is gone.
    """
    used = [k for k in SEGMENT_ALLOWED_ROW_FIELDS if k in row]
    assert_segment_is_feature_side(used)
    sc = sidecar or {}
    lines: List[str] = []
    query = C.normalize_ws(sc.get("query") or "")
    if query:
        lines.append(json.dumps({"current_request": query}, ensure_ascii=False))
    for e in segment_history_entries(sc)[:max_docs]:
        text = e["text"]
        lines.append(json.dumps({
            "earlier_turn": e["earlier_turn"],
            "available": e["available"],
            "text": (None if text is None
                     else C.normalize_ws(text)[:max_doc_chars]),
        }, ensure_ascii=False))
    lines.append(json.dumps({"agent_action": (row.get("prediction") or "")},
                            ensure_ascii=False))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# MIRAGE two-step prompt + grading
# --------------------------------------------------------------------------

#: 2507.21017 §4.3: Utility(c) = 1 faithful, 0.5 incomplete, 0 hallucinated.
UTILITY_GRADES: Dict[str, float] = {"faithful": 1.0, "incomplete": 0.5,
                                    "hallucinated": 0.0}


def utility_from_grade(grade: Optional[str]) -> Optional[float]:
    """Map a judge grade string onto Utility in {1, 0.5, 0} (2507.21017 §4.3).

    The middle grade is explicit and is never collapsed into 1 or 0: it is the
    'neither clearly faithful nor overtly hallucinatory' class.
    """
    if grade is None:
        return None
    key = str(grade).strip().lower()
    if key in UTILITY_GRADES:
        return UTILITY_GRADES[key]
    aliases = {
        "faithful action": "faithful", "correct": "faithful", "1": "faithful",
        "1.0": "faithful",
        "incomplete action": "incomplete", "partial": "incomplete",
        "uncertain": "incomplete", "0.5": "incomplete",
        "hallucinated action": "hallucinated", "hallucination": "hallucinated",
        "harmful": "hallucinated", "0": "hallucinated", "0.0": "hallucinated",
    }
    if key in aliases:
        return UTILITY_GRADES[aliases[key]]
    return None


def mirage_prompt(segment_text: str) -> str:
    """Two-step, per-setting, zero-shot judge prompt (2507.21017 §4.3).

    Step 1 identifies the risk trigger in the frozen snapshot and explains why
    it is a risk; step 2 categorises and scores the agent's response with a
    justification.  The output keeps Tracy's ``harmful_probability`` key so her
    downstream columns keep working.
    """
    return (
        "You are judging one frozen decision point of a tool-using agent. The "
        "agent's earlier conversation was compressed before it reached the "
        "agent; some earlier turns may be summarised or missing. You see "
        "exactly what the agent saw, plus the action it emitted. You have no "
        "reference trajectory and no ground truth.\n\n"
        "Work in two steps.\n"
        "Step 1: identify the risk trigger in this snapshot - the specific "
        "thing that could make the emitted action wrong (for example an "
        "argument that no visible earlier turn grounds) - and explain why it is "
        "a risk. If there is none, say so.\n"
        "Step 2: categorise and score the agent's action, with a "
        "justification. Use exactly one category: 'faithful' (the action is "
        "supported by what is visible), 'incomplete' (neither clearly faithful "
        "nor overtly wrong - the agent neither grounds the action nor reports "
        "that it cannot), or 'hallucinated' (the action asserts something the "
        "visible context does not support).\n\n"
        "Return ONLY a JSON object with keys: risk_trigger (string), why "
        "(string), category (faithful|incomplete|hallucinated), "
        "harmful_probability (number in [0,1]), justification (string).\n\n"
        f"Snapshot:\n{segment_text}"
    )


_JSON_OBJ = re.compile(r"\{.*\}", re.S)

#: A judge that answers ``82`` for 0.82 is not rare, but silently dividing every
#: out-of-range number by 100 invents a scale the reply never stated.  A value
#: outside [0, 1] is rescaled ONLY on explicit percent evidence: the value was a
#: string ending in '%', or the raw reply writes the harmful_probability value
#: with a '%' sign attached.  Anything else out of range is dropped to None and
#: the raw string is kept so the drop is auditable.
_PCT_NEAR_HP = re.compile(
    r"harmful[_ ]?probability\"?\s*[:=]\s*\"?\s*(-?\d+(?:\.\d+)?)\s*%", re.I)

#: Values the range check can report under ``harmful_probability_scale``.
HP_SCALES: Tuple[str, ...] = ("unit", "percent_rescaled", "out_of_range",
                              "not_a_number")


def coerce_harmful_probability(value: Any, raw_text: str = "") -> Dict[str, Any]:
    """Range-check the judge's ``harmful_probability`` onto [0, 1].

    Returns ``{"value", "raw", "scale"}``.  ``value`` is None whenever the reply
    did not commit to a probability this function can place on the unit
    interval; ``raw`` is the verbatim string the judge emitted for the field (so
    an absolute reading can always be audited against what was said) and
    ``scale`` is one of :data:`HP_SCALES`.

    Tracy's ``judge_detector_segments.py`` (:103-105) stores the field verbatim.
    AP/AUROC are rank-based and survive a percent scale, but any ABSOLUTE
    reading (a threshold, a calibration plot, a cost-weighted rule) does not, so
    the scale is recorded rather than assumed.
    """
    out: Dict[str, Any] = {"value": None, "raw": None, "scale": None}
    if value is None or isinstance(value, bool):
        if value is not None:
            out["raw"] = json.dumps(value)
            out["scale"] = "not_a_number"
        return out
    pct_evidence = False
    num: Optional[float] = None
    if isinstance(value, str):
        out["raw"] = value
        s = value.strip()
        if s.endswith("%"):
            pct_evidence = True
            s = s[:-1].strip()
        try:
            num = float(s)
        except ValueError:
            out["scale"] = "not_a_number"
            return out
    elif isinstance(value, (int, float)):
        out["raw"] = repr(value)
        num = float(value)
    else:
        out["raw"] = json.dumps(value, ensure_ascii=False, default=str)
        out["scale"] = "not_a_number"
        return out
    if not np.isfinite(num):
        out["scale"] = "not_a_number"
        return out
    if 0.0 <= num <= 1.0:
        out["value"] = float(num)
        out["scale"] = "unit"
        return out
    if not pct_evidence:
        m = _PCT_NEAR_HP.search(raw_text or "")
        pct_evidence = m is not None and float(m.group(1)) == num
    if pct_evidence and 0.0 <= num <= 100.0:
        out["value"] = float(num) / 100.0
        out["scale"] = "percent_rescaled"
        return out
    out["scale"] = "out_of_range"
    return out


def parse_judge_response(text: str) -> Dict[str, Any]:
    """Parse the judge's JSON (tolerating prose around it).

    ``utility`` follows :func:`utility_from_grade`; when the category is absent
    but ``harmful_probability`` is present, utility stays None rather than being
    inferred — no sentinel, no fallback.  ``harmful_probability`` is
    RANGE-CHECKED by :func:`coerce_harmful_probability`: a value outside [0, 1]
    is None unless the reply itself said percent, and the verbatim string plus
    the scale verdict travel beside it.
    """
    out: Dict[str, Any] = {"harmful_probability": None, "category": None,
                           "utility": None, "risk_trigger": None,
                           "harmful_probability_raw": None,
                           "harmful_probability_scale": None,
                           "parse_ok": False, "raw": text}
    if not text:
        return out
    blob = None
    try:
        blob = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        m = _JSON_OBJ.search(text)
        if m is not None:
            try:
                blob = json.loads(m.group(0))
            except json.JSONDecodeError:
                blob = None
    if not isinstance(blob, dict):
        return out
    hp = coerce_harmful_probability(blob.get("harmful_probability"), text)
    out["harmful_probability"] = hp["value"]
    out["harmful_probability_raw"] = hp["raw"]
    out["harmful_probability_scale"] = hp["scale"]
    out["category"] = blob.get("category")
    out["utility"] = utility_from_grade(blob.get("category"))
    out["risk_trigger"] = blob.get("risk_trigger")
    out["parse_ok"] = out["harmful_probability"] is not None or out["utility"] is not None
    return out


def us_hr(utilities: Sequence[Optional[float]]) -> Dict[str, Any]:
    """MIRAGE aggregates (2507.21017 §4.3): US = mean Utility, HR = share of
    Utility == 0.

    Reported for comparability with the paper ONLY.  They are prevalence rates
    on a curated set and are not divided by any of our three denominators.
    """
    vals = [float(u) for u in utilities if u is not None]
    n = len(vals)
    return {
        "US": float(np.mean(vals)) if n else None,
        "HR": float(np.mean([1.0 if v == 0.0 else 0.0 for v in vals])) if n else None,
        "n_graded": n,
        "n_ungraded": len(utilities) - n,
        "not_a_trigger_metric": True,
    }


# --------------------------------------------------------------------------
# running the judge
# --------------------------------------------------------------------------

def judge_one(row: Dict[str, Any], sidecar: Optional[Dict[str, Any]],
              client: JudgeClient, *, judge_name: str = "judge") -> Dict[str, Any]:
    """One judged battery row: segment -> two-step prompt -> parsed grade.

    Carries Tracy's three cost columns (``judge_seconds``,
    ``judge_prompt_tokens``, ``judge_completion_tokens``) so the judge's own
    generation is charged in the frozen GPU-sec sum, the range columns
    (``judge_score_raw`` / ``judge_score_scale``) so an absolute reading of the
    probability is auditable, and the sidecar-availability flags so a row judged
    with no visible history — or with dropped blocks whose text the dump did not
    carry — says so instead of looking like a row with a short history.
    """
    segment = compact_battery_segment(row, sidecar)
    result = client.complete(mirage_prompt(segment))
    parsed = parse_judge_response(result["text"])
    return {
        "qid": row["qid"],
        "session_id": row.get("session_id") or C.session_of(row["qid"]),
        "judge_name": judge_name,
        "judge_model": client.model,
        "judge_raw": result["text"],
        "judge_score": parsed["harmful_probability"],
        "judge_score_raw": parsed["harmful_probability_raw"],
        "judge_score_scale": parsed["harmful_probability_scale"],
        "judge_category": parsed["category"],
        "judge_utility": parsed["utility"],
        "judge_parse_ok": parsed["parse_ok"],
        "judge_seconds": result["judge_seconds"],
        "judge_prompt_tokens": result["judge_prompt_tokens"],
        "judge_completion_tokens": result["judge_completion_tokens"],
        "sidecar_available": sidecar is not None,
        **dropped_text_availability(sidecar),
    }


def judge_feature_rows(judge_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """FEATURE rows from judged battery rows.

    ``judge_harmful_probability`` (+1) and ``judge_utility_risk`` = 1 - Utility
    (+1); None stays None.  The judge's grade is never a label.
    """
    out: List[Dict[str, Any]] = []
    for r in judge_rows:
        util = r.get("judge_utility")
        out.append({
            "qid": r["qid"],
            "session_id": r.get("session_id") or C.session_of(r["qid"]),
            "judge_harmful_probability": r.get("judge_score"),
            "judge_utility_risk": (None if util is None else 1.0 - float(util)),
            "judge_answer_parse_ok": 1.0 if r.get("judge_parse_ok") else 0.0,
            "judge_seconds": r.get("judge_seconds"),
            "judge_prompt_tokens": r.get("judge_prompt_tokens"),
            "judge_completion_tokens": r.get("judge_completion_tokens"),
        })
    return out


def sample_human_validation(qids: Sequence[str], n: int = 160,
                            seed: int = 20260905) -> Dict[str, Any]:
    """Seeded human-validation subset (2507.21017 validates its judge on 160
    human-annotated samples).

    Drawn before the judge runs so the subset cannot be chosen on the judge's
    own output.  Returns the qids and the seed that produced them.
    """
    pool = sorted(set(qids))
    rng = np.random.default_rng(seed)
    k = min(int(n), len(pool))
    idx = rng.choice(len(pool), size=k, replace=False)
    return {"seed": seed, "n_requested": int(n), "n_sampled": k,
            "qids": sorted(pool[int(i)] for i in idx),
            "n_pool": len(pool)}


# --------------------------------------------------------------------------
# Verify-when-Uncertain cascade — 2502.15845 Algorithm 1
#
# THIN WRAPPERS.  Algorithm 1 has exactly ONE implementation on this branch,
# ``agent/t34_cascade.py`` (its module docstring names these wrappers).  The
# functions below keep this module's public names and argument order and
# delegate the decision rule, the band construction and the threshold fitting
# to that module; nothing here re-derives them.  The import is LAZY so the
# judge's parsing/prompting half stays importable without pulling the cascade
# unit's dependencies in.
# --------------------------------------------------------------------------

#: The digest's pre-registered budget sweep for the escalation fraction.
#: Mirrors ``t34_cascade.P_GRID``; :func:`cascade_sweep` asserts they agree so
#: the two entry points can never sweep different budgets.
P_GRID: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0)


def _cascade_module():
    """Lazy handle on the single Algorithm-1 implementation."""
    import t34_cascade as K  # noqa: PLC0415
    return K


def _thresholds(t1: float, t_star: float, t2: float):
    """Three absolute thresholds as a :class:`t34_cascade.CascadeThresholds`.

    ``p_target`` / ``p_realized`` stay NaN: a caller that hands over three
    numbers measured no budget, and inventing one here would be a sentinel.
    """
    K = _cascade_module()
    return K.CascadeThresholds(t1=float(t1), t_star=float(t_star), t2=float(t2))


def cascade_star_threshold(s_self: Sequence[Optional[float]], t1: float,
                           p: float) -> float:
    """``t*`` is not chosen directly (2502.15845 Algorithm 1): given ``t1`` it is
    set so that a fraction ``p`` of the stage-1 scores falls inside ``[t1, t*]``.

    Wrapper over :func:`t34_cascade.band_upper_threshold`, whose semantics this
    function now inherits: ``p = 0`` returns a ``t*`` just BELOW ``t1`` (every
    score at or above ``t1`` fires without escalation, which is the single-stage
    degenerate case) and a ``p`` larger than the mass at or above ``t1`` returns
    the largest such score rather than ``+inf``.  The realized fraction (ties
    can make the requested one unattainable) is the second element of the
    wrapped call and is reported by :func:`fit_cascade` per fold.
    """
    K = _cascade_module()
    s = [float(v) for v in s_self
         if v is not None and np.isfinite(float(v))]
    t_star, _realized, _k = K.band_upper_threshold(s, float(t1), float(p))
    return float(t_star)


def cascade_decide(s_self: Optional[float], t1: float, t_star: float,
                   s_cross: Optional[float], t2: float) -> Dict[str, Any]:
    """Algorithm 1 for one row — wrapper over :func:`t34_cascade.cascade_decide`.

    ``s_self < t1`` -> negative; ``s_self > t*`` -> positive; in between ->
    escalate and fire iff ``s_cross >= t2``.  The returned dict is the wrapped
    module's own (``fire`` / ``escalated`` / ``stage`` / ``fallback``), with its
    stage vocabulary (``cheap_negative`` / ``cheap_positive`` / ``band`` /
    ``band_unavailable`` / ``undefined``); a band row with no stage-2 score
    takes ``t34_cascade.BAND_FALLBACK_FIRE`` (declared False: an escalation that
    could not be paid for does not fire) and is counted separately.
    """
    K = _cascade_module()
    s = np.nan if s_self is None else float(s_self)
    return K.cascade_decide(s, _thresholds(t1, t_star, t2),
                            None if s_cross is None else float(s_cross))


def cascade_apply(s_self: Sequence[Optional[float]], s_cross: Sequence[Optional[float]],
                  t1: float, t_star: float, t2: float) -> Dict[str, np.ndarray]:
    """Vectorised :func:`cascade_decide` — wrapper over
    :func:`t34_cascade.apply_cascade` (adds the ``fallback`` mask)."""
    K = _cascade_module()
    a = [np.nan if v is None else float(v) for v in s_self]
    b = [None if v is None else float(v) for v in s_cross]
    return K.apply_cascade(a, _thresholds(t1, t_star, t2), b)


def fit_cascade(s_self: Sequence[Optional[float]], s_cross: Sequence[Optional[float]],
                y: Sequence[int], groups: Sequence[str], *,
                p: float, outer_folds: int = 5, inner_folds: int = 3,
                seed: int = 20260905, select_by: str = "youden",
                q1_grid: Optional[Sequence[float]] = None,
                q2_grid: Optional[Sequence[float]] = None,
                stage2_seconds: Optional[Sequence[Optional[float]]] = None
                ) -> Dict[str, Any]:
    """Session-grouped nested CV for (t1, t*, t2) at a fixed budget ``p``.

    Wrapper over :func:`t34_cascade.cascade_nested_cv`: the thresholds are
    materialised from quantile LEVELS chosen on INNER folds of the outer-train
    fold and are then re-materialised on that outer-train fold alone, so both
    the levels tried and the values they resolve to come from training rows
    only — an evaluation row never informs a threshold or the grid that
    produced it.  ``q1_grid`` / ``q2_grid`` override
    ``t34_cascade.Q1_GRID`` / ``Q2_GRID``; the absolute-value ``t1_grid`` /
    ``t2_grid`` arguments of the old local implementation are gone with it.

    The returned dict keeps this module's column names, and adds
    ``n_band_unavailable`` (band rows whose judge score was missing) and the
    per-fold ``p_realized`` that the wrapped module measures.
    """
    K = _cascade_module()
    a = [np.nan if v is None else float(v) for v in s_self]
    b = [None if v is None else float(v) for v in s_cross]
    yy = np.asarray(list(y), dtype=int)
    fit_fn = None
    if q1_grid is not None or q2_grid is not None:
        grids: Dict[str, Any] = {}
        if q1_grid is not None:
            grids["q1_grid"] = list(q1_grid)
        if q2_grid is not None:
            grids["q2_grid"] = list(q2_grid)

        def fit_fn(*args, **kwargs):  # noqa: F811 - injected grid override
            kwargs.update(grids)
            return K.fit_thresholds(*args, **kwargs)

    res = K.cascade_nested_cv(a, yy, b, list(groups), float(p),
                              outer_folds=outer_folds, inner_folds=inner_folds,
                              seed=seed, select_by=select_by, fit_fn=fit_fn)
    scored = res["scored"]
    fire = res["fire"][scored]
    esc = res["escalated"][scored]
    fallback = res["fallback"][scored]
    ys = yy[scored]
    m = K.trigger_metrics(fire, ys)
    sec = None
    if stage2_seconds is not None:
        s2 = np.asarray([0.0 if v is None else float(v) for v in stage2_seconds])
        sec = float(s2[scored][esc].sum())
    cov = m["coverage"] if m["coverage"] is not None else 0.0
    fr = m["false_reset"] if m["false_reset"] is not None else 0.0
    return {
        "p": p,
        "n_scored": int(scored.sum()),
        "n_pos": m["n_pos"], "n_neg": m["n_neg"],
        "escalation_fraction": float(esc.mean()) if esc.size else None,
        "coverage": m["coverage"],
        "n_fires": m["fires"],
        "precision": m["precision"],
        "false_reset_rate": m["false_reset"],
        "objective_coverage_minus_false_reset": float(cov - fr),
        "judge_seconds_spent": sec,
        "n_band_unavailable": int(fallback.sum()),
        # With the declared stage-1 substitution (parse failure alone) s_self is
        # BINARY, so every threshold lands on the same two values and the band
        # is either empty or everything: the p sweep is nearly uninformative
        # until a continuous free signal replaces it.  Reported, not silently
        # swept.
        "stage1_distinct_values": int(len({float(v) for v in a
                                           if np.isfinite(v)})),
        "stage1_degenerate_band": bool(len({float(v) for v in a
                                            if np.isfinite(v)}) <= 2),
        "n_threshold_pairs_tried": int(sum(int(f.get("n_evaluations") or 0)
                                           for f in res["folds"])),
        "chosen_per_outer_fold": res["folds"],
        "regime": res["regime"],
        "implementation": "t34_cascade.cascade_nested_cv",
    }


def cascade_sweep(s_self, s_cross, y, groups, *, p_grid: Sequence[float] = P_GRID,
                  stage2_seconds=None, **kwargs) -> Dict[str, Any]:
    """Sweep the budget dial (digest §4.12: cost column first, AUROC second)."""
    K = _cascade_module()
    if tuple(P_GRID) != tuple(K.P_GRID):
        raise AssertionError(
            "t34_judge.P_GRID and t34_cascade.P_GRID have diverged: "
            f"{P_GRID} vs {K.P_GRID}")
    rows = [fit_cascade(s_self, s_cross, y, groups, p=float(p),
                        stage2_seconds=stage2_seconds, **kwargs) for p in p_grid]
    return {"p_grid": list(p_grid), "rows": rows,
            "implementation": "t34_cascade (single Algorithm-1 implementation)",
            "stage1_degenerate_band": bool(rows and rows[0]["stage1_degenerate_band"]),
            "stage1_note": ("stage 1 is the free parse-failure indicator "
                            "(declared substitution): a binary s_self makes the "
                            "uncertainty band collapse, so read the p sweep as "
                            "a cost curve, not as a threshold study"),
            "oracle_ceiling_note": "the oracle-trigger row is a horizontal "
                                   "ceiling on this curve, not a cascade point"}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _scale_counts(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """How many judged rows landed on each :data:`HP_SCALES` verdict.

    ``unrecorded`` counts rows written before the range check existed, so an
    old judge file is never silently read as if it had been checked.
    """
    counts: Dict[str, int] = {}
    for r in rows:
        key = r.get("judge_score_scale")
        if key is None and "judge_score_scale" not in r:
            key = "unrecorded"
        counts[str(key)] = counts.get(str(key), 0) + 1
    return dict(sorted(counts.items()))


def _frame(args) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    from t33_labels import build_label_frame, join_arms
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pairs = join_arms(load_jsonl(args.battery_full), load_jsonl(args.battery_c2kv))
    frame = build_label_frame(pairs, manifest)
    return frame, {c["qid"]: c for _, c in pairs}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("RUNBOOK")[0].strip())
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _battery_args(p):
        p.add_argument("--battery-c2kv", required=True)
        p.add_argument("--battery-full", required=True)
        p.add_argument("--manifest", required=True)

    p_s = sub.add_parser("sample-validation", help="seeded human-validation subset")
    _battery_args(p_s)
    p_s.add_argument("--n", type=int, default=160)
    p_s.add_argument("--seed", type=int, default=20260905)
    p_s.add_argument("--out", required=True)

    p_r = sub.add_parser("run", help="run the judge over the 161-row trigger subset")
    _battery_args(p_r)
    p_r.add_argument("--sidecar", required=True)
    p_r.add_argument("--base-url", required=True)
    p_r.add_argument("--model", required=True)
    p_r.add_argument("--judge-name", default="judge")
    p_r.add_argument("--timeout", type=int, default=300)
    p_r.add_argument("--max-rows", type=int, default=0)
    p_r.add_argument("--out", required=True)

    p_sc = sub.add_parser("score", help="features + US/HR + the winner-table row")
    _battery_args(p_sc)
    p_sc.add_argument("--judge-rows", required=True)
    p_sc.add_argument("--features-out", required=True)
    p_sc.add_argument("--out", required=True)

    p_c = sub.add_parser("cascade", help="Verify-when-Uncertain p sweep")
    _battery_args(p_c)
    p_c.add_argument("--judge-rows", required=True)
    p_c.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    frame, c2kv_by_qid = _frame(args)
    subset = [r for r in frame if r["label_cw"] in (0, 1)]

    if args.cmd == "sample-validation":
        res = sample_human_validation([r["qid"] for r in subset], n=args.n,
                                      seed=args.seed)
        C.freeze_json(Path(args.out), res)
        print(json.dumps({k: v for k, v in res.items() if k != "qids"}, indent=2))
        return 0

    if args.cmd == "run":
        if not Path(args.sidecar).exists():
            raise SystemExit(
                f"FATAL: sidecar {args.sidecar} does not exist. The judge "
                "segment is built from the decoded per-doc sidecar (unit U2, "
                "agent/t34_dump_sidecar.py); without it every snapshot would "
                "carry no history at all. Dump it first, with "
                "--with_dropped_text if the dropped side is to be judged.")
        sidecar = {r["qid"]: r for r in load_jsonl(args.sidecar)}
        client = JudgeClient(args.base_url, args.model, timeout=args.timeout)
        qids = [r["qid"] for r in subset]
        if args.max_rows:
            qids = qids[: args.max_rows]
        avail = [dropped_text_availability(sidecar.get(q)) for q in qids]
        n_missing_sidecar = sum(1 for q in qids if q not in sidecar)
        n_text_missing = sum(1 for a in avail
                             if a["dropped_doc_texts_available"] is False)
        if n_missing_sidecar == len(qids):
            raise SystemExit(
                f"FATAL: none of the {len(qids)} trigger qids appear in "
                f"{args.sidecar}; the sidecar was dumped for a different "
                "rowset.")
        print(json.dumps({
            "sidecar_rows_missing": n_missing_sidecar,
            "rows_with_dropped_blocks_but_no_dropped_text": n_text_missing,
            "note": ("dropped blocks are rendered available:false with text "
                     "null unless the dump carried dropped_doc_texts"),
        }, indent=2))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            done = {r["qid"] for r in load_jsonl(str(out))}
        with open(out, "a", encoding="utf-8", newline="\n") as fh:
            for qid in qids:
                if qid in done:
                    continue
                rec = judge_one(c2kv_by_qid[qid], sidecar.get(qid), client,
                                judge_name=args.judge_name)
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
        print(json.dumps({"judged": len(qids), "out": str(out)}, indent=2))
        return 0

    rows = load_jsonl(args.judge_rows)
    by_qid = {r["qid"]: r for r in rows}
    label_by_qid = {r["qid"]: r["label_cw"] for r in subset}
    pf_by_qid = {r["qid"]: bool(r["parse_fail_fire"]) for r in subset}
    order = [r["qid"] for r in subset if r["qid"] in by_qid]
    y = [label_by_qid[q] for q in order]
    sess = [C.session_of(q) for q in order]

    if args.cmd == "score":
        feats = judge_feature_rows([by_qid[q] for q in order])
        C.write_features_jsonl(Path(args.features_out), feats,
                               context="t34_judge features")
        import t34_selfreport as SR
        table = SR.score_feature_table(
            [f["judge_harmful_probability"] for f in feats], y, sess,
            name="judge_harmful_probability",
            orientation=SR.orientation_of("judge_harmful_probability"),
            baselines={"parse_fail": [1.0 if pf_by_qid[q] else 0.0 for q in order]})
        res = {
            "n_judged": len(order),
            "table": table,
            "us_hr_reported_for_comparability_only":
                us_hr([by_qid[q].get("judge_utility") for q in order]),
            "cost": {
                "judge_seconds_total": float(sum(
                    float(by_qid[q].get("judge_seconds") or 0.0) for q in order)),
                "judge_prompt_tokens_total": int(sum(
                    int(by_qid[q].get("judge_prompt_tokens") or 0) for q in order)),
                "judge_completion_tokens_total": int(sum(
                    int(by_qid[q].get("judge_completion_tokens") or 0) for q in order)),
            },
            "judge_names": sorted({by_qid[q].get("judge_name") for q in order}),
            "harmful_probability_scales": _scale_counts(
                [by_qid[q] for q in order]),
            "inputs": {
                "rows_without_sidecar": sum(
                    1 for q in order if by_qid[q].get("sidecar_available") is False),
                "rows_with_dropped_blocks_but_no_dropped_text": sum(
                    1 for q in order
                    if by_qid[q].get("dropped_doc_texts_available") is False),
                "note": ("a row judged without its sidecar saw no history at "
                         "all; a row whose dropped text is unavailable saw the "
                         "gap named but not its content"),
            },
        }
        C.freeze_json(Path(args.out), res)
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "cascade":
        stage1 = [1.0 if pf_by_qid[q] else 0.0 for q in order]
        stage2 = [by_qid[q].get("judge_score") for q in order]
        secs = [by_qid[q].get("judge_seconds") for q in order]
        res = cascade_sweep(stage1, stage2, y, sess, stage2_seconds=secs)
        C.freeze_json(Path(args.out), res)
        print(json.dumps(res, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
