# -*- coding: utf-8 -*-
"""t34 §4.12 — self-report: VISTA proprioception probes + ledger, MemGPT page-in,
same-context selfcheck arm.

RUNBOOK (execution order; "SERVER" = NPU box with torch, "HERE" = this Windows box)
------------------------------------------------------------------------------
0. HERE   python agent/t34_selfreport.py cost --rows 161 --probes 4 --conditions 2
          # prints the pre-registered generation budget before anything is spent.
1. SERVER python agent/t34_dump_sidecar.py ...            (unit U2; produces
          results/t34/sidecar_c2kv.jsonl — this module only READS it)
2. HERE   python agent/t34_selfreport.py plan-probes \
              --sidecar results/t34/sidecar_c2kv.jsonl --condition minus_ledger \
              --out results/t34/probe_plan_minus_ledger.jsonl
          # deterministic (qid, probe, sampled blocks, query text) plan; the
          # server driver only executes it.  Also runs the leak scan.
3. SERVER python agent/t34_probe_mode.py --plan results/t34/probe_plan_minus_ledger.jsonl \
              --arm c2kv ... --out results/t34/probes_vista_minus_ledger_c2kv.jsonl
4. HERE   python agent/t34_selfreport.py score-probes \
              --probes results/t34/probes_vista_minus_ledger_c2kv.jsonl \
              --sidecar results/t34/sidecar_c2kv.jsonl \
              --battery-c2kv results/bdf_pilot/d_r2/battery_c2kv.jsonl \
              --battery-full results/bdf_pilot/d_r2/battery_full.jsonl \
              --manifest configs/bdf_pilot/d_cw_manifest_r2.json \
              --out results/t34/vista_minus_ledger.json
          # emits the gate verdict: run +ledger only if -ledger passes.
4b. SERVER+HERE  repeat steps 1-3 for --arm full (its own sidecar, its own
          plan) and pool the two P4 files:
              python agent/t34_selfreport.py score-p4-pooled \
                  --probes results/t34/probes_vista_minus_ledger_c2kv.jsonl \
                           results/t34/probes_vista_minus_ledger_full.jsonl \
                  --out results/t34/vista_p4_pooled.json
          # MANDATORY for the "was anything condensed" half: on the compressed
          # arm alone that truth is constant True and no accuracy can beat it.
5. HERE   (only if step 4 says pass) repeat 2-4 with --condition plus_ledger.
6. SERVER python agent/t34_probe_mode.py --mode selfcheck --p4-result \
              results/t34/vista_minus_ledger.json ...      (refuses without it)
7. HERE   python agent/t34_selfreport.py score-selfcheck --probes ... --out ...
          # features jsonl + the co-firing-with-parse-failure "was it worth it"
          # diagnostic + the S0 full-arm label control.
8. HERE   python agent/t34_selfreport.py pagein-smoke --proxy-log <bench log>
          # MemGPT fire-rate smoke test; fire rate 0 over >=30 conversations
          # is the stopping rule and prints STOP.

Papers implemented here
-----------------------
* 2606.30005 (VISTA) §14 proprioceptive-blindness diagnostic + §4 discussion,
  §2.4/§3.2 ledger and its prefix-cache placement constraint.
* 2310.08560 (MemGPT) §2.2 page-out counters, §2.3 self-emitted page-in,
  §3.2.1/§3.2.2 under-firing failure modes.
* SPEC §8.2 item 6 — the same-compressed-context `selfcheck` arm (designed,
  never run; no paper).

WIRING (bench face; do NOT edit those files, they live on another branch)
------------------------------------------------------------------------
``intercept()`` is the pure form of the MemGPT interception rule.  It plugs into
``tmp/bench-recover/benchmarks/proxy.py`` next to ``normalize_response``:
feed it ``action_canonical(message)`` (proxy.py:466, built from ``_canon_calls``
proxy.py:406) *before* ``RecoverState.check`` (class proxy.py:476, method
``check`` proxy.py:494).  When it returns ``fired``, the step must NOT be
counted as a task action; instead fire the existing recovery primitive with the
returned ``recovery`` dict and re-issue the same request, then log the fire in
``_log_request`` (proxy.py:1285) as two ``generate_sec`` charges.
``benchmarks/arms.py`` ``Arm.validate`` (def at :178) rejects ``repair`` and
``recover`` together at :189-190 — that guard has to be relaxed before this arm
can run.  Nothing in this module edits any of those files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import t34_common as C  # noqa: E402
from t33_labels import load_jsonl  # noqa: E402

DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "VISTA proprioception diagnostic (P1/P2/P3)",
        "paper": "2606.30005 §14",
        "what": "The paper anchors at the first archive event of a live LOCA run "
                "(n=29 runs) and asks about text blocks the model can still read. "
                "We anchor at the 161 frozen battery decision points (93 C->W + 68 "
                "C->C) and ask about history blocks that have been compressed into "
                "gist KV and are NOT readable at all.",
        "why": "Our frozen frame is the only rowset with the C->W label; the "
               "battery has no agent loop and no archive event to anchor on.",
    },
    {
        "method": "VISTA P1 ground truth",
        "paper": "2606.30005 §14",
        "what": "Truth for the total-size probe is the frozen row's "
                "``kept_history_tokens`` (pre-compression tokens of the history "
                "actually fed in), cross-checked against sum(sidecar doc_lengths); "
                "the card's ``original_seq_len`` is not a column of the frozen "
                "battery rows.",
        "why": "Never invent a column: the r2 battery rows carry doc_tokens / "
               "kept_history_tokens / compressed_history_tokens and no "
               "original_seq_len.",
    },
    {
        "method": "VISTA P1 has two truths on a compressed arm",
        "paper": "2606.30005 §14 (one transcript, one exact token count)",
        "what": "VISTA's transcript has a single size.  Ours has two — the "
                "pre-compression size of the history that was fed in "
                "(``kept_history_tokens``) and the compressed size actually "
                "occupying the prefix (``compressed_history_tokens``) — and the "
                "neutral question wording does not pick between them.  BOTH are "
                "scored and printed (``P1`` and "
                "``P1_vs_compressed_history_tokens``); the pre-compression "
                "reading is the declared headline.",
        "why": "Scoring only one of them, or choosing after seeing which came "
               "out better, would be selection on the result; the card itself "
               "names both columns as the truth.",
    },
    {
        "method": "VISTA P4 (loss awareness)",
        "paper": "2606.30005 — no counterpart",
        "what": "P4 has no VISTA analogue; it is ours.  Detection half scored "
                "against the frame's own measured base rate (the card gives no "
                "number for it), localisation half against the 25.0 % wrong-block "
                "floor via t34_common.locate_table.  Its 'was anything "
                "condensed' truth is CONSTANT on the compressed arm, so that "
                "half is scored only POOLED over the c2kv and full probe files "
                "(``pooled_p4_detection`` / ``score-p4-pooled``); within one "
                "arm it is reported as degenerate with no p-value.",
        "why": "The card names P4 as the missing probe and explicitly says its "
               "base rate is not given by the paper; the negatives for the "
               "condensed half exist only on the full arm, which is why that "
               "probe pass is mandatory rather than a control.",
    },
    {
        "method": "VISTA P2 request shape",
        "paper": "2606.30005 §4 / §14 (block-size probe over four sampled blocks)",
        "what": "The paper does not say whether its four sampled blocks are "
                "asked in one request or four; we ask all four in ONE request "
                "(one integer per turn, comma separated) to keep the digest's "
                "frozen 4-probes x 161-rows x 2-conditions = 1288 generation "
                "budget.  The SCORE is the paper's own: median relative error "
                "PER BLOCK (its +dash block-size error is 0-0.02 per block), "
                "never over the four blocks' sum.  A request whose answer does "
                "not yield exactly four aligned integers is unparsed for all "
                "four blocks and counted, never padded or re-aligned.  A row "
                "with fewer than four VISIBLE blocks is asked about as many as "
                "it has (``sample_blocks`` caps k at that count), so the "
                "per-block denominator varies by row and the histogram of "
                "blocks-per-request is printed beside every P2 number.",
        "why": "Summing the four would silently turn the block-size probe into "
               "a second total-size probe and change the estimand the paper "
               "reports.",
    },
    {
        "method": "VISTA probe turn numbering",
        "paper": "2606.30005 §14 (its transcript has no dropped blocks, so the "
                 "paper never has two numbering frames to reconcile)",
        "what": "P1/P2/P3 number the turns the model can STILL SEE (the "
                "sidecar's kept ``docs`` / ``doc_lengths``), P4 and the "
                "ledger's dropped list number the ORIGINAL post-split "
                "conversation (``dropped_docs``), and every rendered string "
                "says which frame it is using.",
        "why": "``docs`` holds only the kept blocks while ``dropped_docs`` "
               "indexes the post-split history: unlabelled, a +ledger prompt "
               "asserts 'turn 2 was dropped' and asks for the size of a "
               "different 'turn 2' in the same breath, and the size probes "
               "would be answered in a frame the truth is not in.",
    },
    {
        "method": "VISTA P3 / P4 unparsed handling",
        "paper": "2606.30005 §14 (no rule given)",
        "what": "Declared before running: P1/P2 unparsed answers are excluded "
                "from the median and counted (``n_unparsed``); P3 unparsed "
                "answers are excluded from the accuracy denominator and "
                "counted; P4 detection counts an unparsed answer as a MISS "
                "inside the denominator; P4 localisation excludes rows whose "
                "truth block set is empty (undefined, the k*=None analogue) and "
                "counts a defined-truth abstention in the denominator as a miss.",
        "why": "Card pitfall (f) requires the None case be fixed before the run; "
               "the paper only notes that one backbone returns valid structured "
               "output less often and gives no exclusion rule.",
    },
    {
        "method": "VISTA P4 gate vs +ledger gate",
        "paper": "2606.30005 — neither gate is in the paper",
        "what": "Two DIFFERENT pre-registered gates: ``ledger_gate`` decides "
                "whether to spend the +ledger half of the budget and may pass "
                "on P4 OR on a size probe; ``p4_selfcheck_gate`` decides "
                "whether the SPEC §8.2 item 6 selfcheck arm gets built and "
                "keys on the P4 detection half alone.  A degenerate P4 is "
                "reported as undetermined, not as a pass or a fail.",
        "why": "The card licenses the selfcheck arm on P4 specifically ('if P4 "
               "is materially above the base rate, the self-check arm is "
               "alive'); size proprioception is a different construct and must "
               "not stand in for it.",
    },
    {
        "method": "VISTA ledger placement",
        "paper": "2606.30005 §4 / §14 (+dash prepends)",
        "what": "The ledger is APPENDED at the tail (after the compressed history, "
                "before the current query); VISTA prepends it.",
        "why": "Card pitfall (c): a per-turn prompt edit at the head invalidates "
               "the reusable KV prefix, and our compressed prefix is the product. "
               "Head placement would void the cost column.  Effect of tail vs head "
               "on accuracy is unmeasured by anyone (card open question 5).",
    },
    {
        "method": "VISTA -ledger/+ledger gate",
        "paper": "2606.30005 — no gate in the paper",
        "what": "A pre-registered pass/fail gate decides whether +ledger is run at "
                "all (LEDGER_GATE constants below).  The paper runs both arms "
                "unconditionally.",
        "why": "Digest §4.12: run +ledger only if -ledger passes; halves the 1288 "
               "generation budget.  The criterion is ours and is declared in code.",
    },
    {
        "method": "MemGPT page-in",
        "paper": "2310.08560 §2.3",
        "what": "The reload tool is PARAMETERLESS (``request_history_reload()``); "
                "the block-id variant exists only as an offline diagnostic and is "
                "never a positive arm.  MemGPT's recall functions take arguments.",
        "why": "Digest §4.12 / SPEC §5.3: an argument-free request is append-only "
               "erratum with zero localisation exposure.",
    },
    {
        "method": "MemGPT page-out thresholds",
        "paper": "2310.08560 §2.2",
        "what": "70 % warning / 100 % flush / 50 % eviction are carried over as "
                "the paper's own 'e.g.' defaults and are exposed as arguments; no "
                "sweep and no calibration set is claimed, because the paper has "
                "neither.",
        "why": "Faithfulness: these are illustrative defaults in the source, not "
               "tuned values.",
    },
    {
        "method": "MemGPT reload budget",
        "paper": "2310.08560 §2.3 (no bound on recall calls)",
        "what": "``intercept`` enforces at most ONE self-emitted reload per "
                "conversation (``max_reloads_per_conv``).  MemGPT places no "
                "bound on how often the model may page memory back in — its "
                "§3.2.1 failure mode is under-firing, not over-firing.",
        "why": "The bench's arm-alignment bookkeeping (match / diverged_now / "
               "re_diverged / tracking_lost) is defined for one repair per "
               "conversation; more than one reload makes those columns "
               "uninterpretable.  Declared, exposed as an argument, never "
               "presented as the paper's rule.",
    },
    {
        "method": "MemGPT evaluation",
        "paper": "2310.08560 — no precision accounting anywhere",
        "what": "coverage/precision/false-reset and a 2x generate_sec cost column "
                "are added; the paper reports only a coverage-shaped accuracy "
                "delta (DMR 32.1 -> 92.5).",
        "why": "SPEC §1.2 three-metric contract; the paper supplies no "
               "false-positive number to inherit.",
    },
    {
        "method": "selfcheck arm",
        "paper": "no paper (SPEC §8.2 item 6)",
        "what": "Implemented as designed in the SPEC: same compressed prefix, "
                "question appended after the emitted call, temperature 0, "
                "enable_thinking False, max 32 new tokens.  No paper-faithful "
                "variant exists to keep.",
        "why": "The card for it is 'no arXiv id'; VISTA P4 is its gate and MIRAGE "
               "0.756-0.769 its soft upper bound.",
    },
]

# --------------------------------------------------------------------------
# (A) VISTA proprioception probes — 2606.30005 §14
# --------------------------------------------------------------------------

PROBE_IDS: Tuple[str, ...] = ("P1", "P2", "P3", "P4")

#: VISTA §14 samples four blocks for the block-size probe and two for the
#: pairwise probe.  Both are asked in SEPARATE requests ("so the measurements
#: stay independent").
P2_N_BLOCKS = 4
P3_N_PAIRS = 1

#: Blocks are addressed 1-based in the probe text (natural language), truth
#: indices are 0-based.  The conversion happens once, in the scorers.
BLOCK_NUMBERING = "one_based"

#: TWO NUMBERING FRAMES EXIST AND EVERY STRING NAMES THE ONE IT USES.
#: The sidecar's ``docs`` / ``doc_lengths`` hold only the KEPT blocks (the rows
#: the model actually saw, in order), while ``dropped_docs`` indexes the
#: POST-SPLIT history list.  So:
#:   * P1/P2/P3 address the turns the model can STILL SEE — index into
#:     ``doc_lengths`` plus one — because a turn it cannot see has no size it
#:     could report;
#:   * P4 and the ledger's dropped list address the ORIGINAL conversation —
#:     ``dropped_docs`` index plus one — because a dropped turn has no visible
#:     position at all.
#: Leaving that implicit put the two frames in the SAME +ledger prompt with no
#: label ("earlier turns dropped entirely: 2, 3" beside "estimate the size of
#: earlier turn 2"), which is a silent contradiction, so both frames are now
#: spelled out in the rendered text.
NUMBERING_FRAMES = {
    "P1": "visible_kept_order",
    "P2": "visible_kept_order",
    "P3": "visible_kept_order",
    "P4": "original_post_split_order",
    "ledger_dropped_docs": "original_post_split_order",
}

#: Sentence prefix that fixes the visible-turn frame for P2/P3.
VISIBLE_FRAME_PREAMBLE = ("The earlier turns you can still see are numbered in "
                          "order, starting at 1. ")

#: VISTA §4: without the dashboard every backbone's median relative error lands
#: in this band and estimates are essentially uncorrelated with truth.
VISTA_BLINDNESS_BAND = (0.43, 0.84)

#: Pre-registered gate for running the +ledger condition (ours, not the paper's).
LEDGER_GATE = {
    "min_parse_rate": 0.50,      # VISTA's own caveat: backbones differ in how
                                 # often they return valid structured output
    "p4_alpha": 0.05,            # detection half must beat its own base rate
    "size_error_max": VISTA_BLINDNESS_BAND[0],   # or size error better than 0.43
}

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_NUM = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*([kKmM])?")
_BLOCK_REF = re.compile(r"(?:block|turn|message|chunk|item)\s*#?\s*(\d+)", re.I)
#: VISTA §14 sanitisation list: no token/budget/ratio string may reach the prompt.
_STATE_LEAK = re.compile(
    r"\b\d+\s*tokens?\b"
    r"|\bcompression\s*ratio\b|\bratio\s*[=:]\s*\d"
    r"|\bdropped[_ ]docs?\b|\bgist[_ ]tokens?\b|\bkept[_ ]history\b"
    r"|\bbudget\b\s*[=:]\s*\d|\bcontext window\b\s*[=:]?\s*\d",
    re.I,
)


def scan_prompt_for_state_leak(text: str) -> List[str]:
    """VISTA §14 sanitisation scan (2606.30005 §14: 'no residual token, budget,
    or usage strings remain').

    Returns the offending substrings; empty list = clean.  Used both as a test
    and as a runtime assertion on every -ledger probe query.
    """
    return [m.group(0) for m in _STATE_LEAK.finditer(text or "")]


def build_probe_query(probe: str, *, blocks: Optional[Sequence[int]] = None,
                      n_blocks: Optional[int] = None) -> str:
    """The four probe QUERIES (2606.30005 §14: three size probes + our P4).

    ``blocks`` are 0-based block indices; the rendered text is 1-based
    (``BLOCK_NUMBERING``).  Wording is deliberately neutral: it names no count,
    no ratio and no budget (see :func:`scan_prompt_for_state_leak`).

    Each probe states WHICH numbering frame its turn numbers use
    (:data:`NUMBERING_FRAMES`): P1/P2/P3 count the turns still visible, P4
    counts turns of the original conversation, which is the frame its truth
    (``dropped_docs``) and the ledger's dropped list are in.
    """
    probe = probe.upper()
    if probe == "P1":
        return ("Estimate how much of the earlier conversation is currently "
                "available to you, measured in tokens. Reply with a single "
                "integer and nothing else.")
    if probe == "P2":
        if not blocks:
            raise ValueError("P2 needs block indices")
        b = ", ".join(str(int(i) + 1) for i in blocks)
        # 2606.30005 §4 scores INDIVIDUAL block size (its +dash error is
        # 0-0.02 per block), so the request must return one estimate PER
        # block, not one number for the group.
        return (VISIBLE_FRAME_PREAMBLE +
                "Estimate the size in tokens of each of these "
                f"earlier turns: {b}. Reply with one integer per turn, in that "
                "order, separated by commas, and nothing else.")
    if probe == "P3":
        if not blocks or len(blocks) != 2:
            raise ValueError("P3 needs exactly two block indices")
        i, j = int(blocks[0]) + 1, int(blocks[1]) + 1
        return (VISIBLE_FRAME_PREAMBLE +
                f"Which is larger, earlier turn {i} (answer A) or "
                f"earlier turn {j} (answer B)? Reply with A or B and nothing else.")
    if probe == "P4":
        # ``n_blocks`` is accepted for symmetry and deliberately NOT rendered:
        # disclosing the block count would itself be a state leak (see
        # :func:`scan_prompt_for_state_leak`).
        return ("Has any part of the earlier conversation been removed or "
                "condensed before it reached you? Reply yes or no; if yes, also "
                "give the number of the earliest affected turn, counting from 1 "
                "in the original conversation, before anything was removed.")
    raise ValueError(f"unknown probe {probe!r}")


def sample_blocks(qid: str, n_docs: int, k: int, seed: int = 20260905) -> List[int]:
    """Deterministic per-qid block sample (2606.30005 §14 samples four blocks).

    Sampling is independent of the truth values, so the within-2x hard subset
    stays a post-hoc restriction the way the paper defines it.
    """
    if n_docs <= 0 or k <= 0:
        return []
    h = hashlib.sha256(f"{seed}:{qid}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    k = min(k, n_docs)
    return sorted(int(v) for v in rng.choice(n_docs, size=k, replace=False))


# ---- parsers (robust to prose) -------------------------------------------

def _strip_thinking(text: str) -> str:
    return _THINK.sub(" ", text or "")


def _num_value(match: "re.Match[str]") -> float:
    raw = match.group(1).replace(",", "")
    val = float(raw)
    suf = (match.group(2) or "").lower()
    if suf == "k":
        val *= 1e3
    elif suf == "m":
        val *= 1e6
    return val


def parse_integer_answer(text: str) -> Optional[int]:
    """Extract an integer size estimate from free text (2606.30005 §14 asks for
    a bare number; real answers are prose).

    Precedence, fixed before running: (1) a number adjacent to the word
    'token'; (2) a number the answer starts with; (3) the last number in the
    text.  ``None`` when no number is present — never a sentinel.
    """
    t = _strip_thinking(text)
    if not t.strip():
        return None
    for m in _NUM.finditer(t):
        tail = t[m.end(): m.end() + 12].lower()
        head = t[max(0, m.start() - 12): m.start()].lower()
        if "token" in tail or "token" in head:
            return int(round(_num_value(m)))
    stripped = t.lstrip(" \t\n:-*>")
    m0 = _NUM.match(stripped)
    if m0 is not None:
        return int(round(_num_value(m0)))
    last = None
    for m in _NUM.finditer(t):
        last = m
    return int(round(_num_value(last))) if last is not None else None


def parse_integer_list_answer(text: str, n: int) -> Optional[List[Optional[int]]]:
    """Extract ``n`` per-block size estimates from one P2 answer.

    2606.30005 §4 reports block-size error PER BLOCK, so the four sampled
    blocks of §14's block-size probe need four numbers out of one generation.

    Precedence, fixed before running:
      0. explicit block references ("earlier turn 3", "block #1") are removed
         first, so the turn INDEX is never mistaken for a size;
      1. numbers adjacent to the word 'token', when exactly ``n`` of them;
      2. all remaining numbers, when exactly ``n`` of them;
      3. the first ``n`` numbers when there are more;
      4. otherwise ``None`` — the whole row is unparsed, never padded with a
         sentinel and never re-aligned by guesswork.
    """
    if n <= 0:
        return None
    t = _BLOCK_REF.sub(" <ref> ", _strip_thinking(text))
    if not t.strip():
        return None
    tokenish: List[int] = []
    every: List[int] = []
    for m in _NUM.finditer(t):
        val = int(round(_num_value(m)))
        every.append(val)
        tail = t[m.end(): m.end() + 12].lower()
        head = t[max(0, m.start() - 12): m.start()].lower()
        if "token" in tail or "token" in head:
            tokenish.append(val)
    if len(tokenish) == n:
        return list(tokenish)
    if len(every) == n:
        return list(every)
    if len(every) > n:
        return every[:n]
    return None


def parse_choice_answer(text: str, options: Sequence[str] = ("A", "B")) -> Optional[str]:
    """Extract a single-letter choice (VISTA's pairwise probe)."""
    t = _strip_thinking(text)
    opts = [o.upper() for o in options]
    for m in re.finditer(r"\b([A-Za-z])\b", t):
        tok = m.group(1).upper()
        if tok in opts:
            return tok
    low = t.lower()
    for o in opts:
        if re.search(rf"\boption\s*{o.lower()}\b", low) or re.search(rf"\b{o.lower()}\)", low):
            return o
    return None


def parse_yes_no_answer(text: str) -> Optional[bool]:
    """yes/no out of prose; ``None`` when the answer commits to neither."""
    t = _strip_thinking(text).lower()
    m = re.search(r"\b(yes|no|yep|nope|affirmative|negative)\b", t)
    if m is None:
        if re.search(r"\b(nothing (was|has been) (removed|condensed))\b", t):
            return False
        if re.search(r"\b(part of|some of) the .*(was|has been) (removed|condensed)\b", t):
            return True
        return None
    return m.group(1) in ("yes", "yep", "affirmative")


def parse_loss_answer(text: str) -> Dict[str, Optional[int]]:
    """P4 parser: yes/no + the 1-based turn number, converted to a 0-based
    block index.  ``block`` is None when the answer names no turn."""
    detected = parse_yes_no_answer(text)
    t = _strip_thinking(text)
    block: Optional[int] = None
    m = _BLOCK_REF.search(t)
    if m is not None:
        block = int(m.group(1)) - 1
    elif detected:
        # "yes, 3" style: take the first bare integer after the yes
        after = t[t.lower().find("yes") + 3:] if "yes" in t.lower() else ""
        m2 = _NUM.search(after)
        if m2 is not None:
            block = int(round(_num_value(m2))) - 1
    if block is not None and block < 0:
        block = None
    return {"detected": detected, "block": block}


# ---- scorers (VISTA's own rules) -----------------------------------------

def median_relative_error(pairs: Sequence[Tuple[Optional[float], Optional[float]]]) -> Dict[str, Any]:
    """VISTA §14 scoring rule for the size probes: MEDIAN RELATIVE ERROR.

    ``pairs`` are (predicted, truth).  Rows with an unparsed prediction or a
    non-positive truth are excluded from the median and reported separately —
    never imputed.  Spearman rho is the companion the paper's prose reports
    ('estimates essentially uncorrelated with truth').
    """
    errs: List[float] = []
    preds: List[float] = []
    truths: List[float] = []
    n_unparsed = 0
    n_bad_truth = 0
    for pred, truth in pairs:
        if truth is None or not np.isfinite(float(truth)) or float(truth) <= 0:
            n_bad_truth += 1
            continue
        if pred is None:
            n_unparsed += 1
            continue
        errs.append(abs(float(pred) - float(truth)) / float(truth))
        preds.append(float(pred))
        truths.append(float(truth))
    rho: Optional[float] = None
    if len(errs) >= 3 and len(set(preds)) > 1 and len(set(truths)) > 1:
        from scipy.stats import spearmanr
        rho = float(spearmanr(preds, truths).statistic)
    return {
        "median_relative_error": float(np.median(errs)) if errs else None,
        "n_scored": len(errs),
        "n_unparsed": n_unparsed,
        "n_undefined_truth": n_bad_truth,
        "spearman_rho_vs_truth": rho,
        "vista_blindness_band": list(VISTA_BLINDNESS_BAND),
    }


def pairwise_hard_subset_accuracy(
    items: Sequence[Tuple[Optional[str], Optional[float], Optional[float]]],
    *, hard_ratio: float = 2.0,
) -> Dict[str, Any]:
    """VISTA §14: pairwise accuracy REPORTED ON THE HARD SUBSET where the two
    blocks are within 2x in true size.

    ``items`` are (predicted_choice, size_A, size_B).  The correct answer is
    'A' when size_A > size_B.  Exact ties are undefined and excluded.
    """
    all_hits = all_n = hard_hits = hard_n = 0
    n_unparsed = n_tie = 0
    for choice, a, b in items:
        if a is None or b is None or float(a) <= 0 or float(b) <= 0:
            continue
        a, b = float(a), float(b)
        if a == b:
            n_tie += 1
            continue
        if choice is None:
            n_unparsed += 1
            continue
        truth = "A" if a > b else "B"
        hit = 1 if choice.upper() == truth else 0
        all_hits += hit
        all_n += 1
        if max(a, b) / min(a, b) <= hard_ratio:
            hard_hits += hit
            hard_n += 1
    return {
        "accuracy_all": (all_hits / all_n) if all_n else None,
        "n_all": all_n,
        "accuracy_hard_subset": (hard_hits / hard_n) if hard_n else None,
        "n_hard_subset": hard_n,
        "hard_ratio": hard_ratio,
        "n_unparsed": n_unparsed,
        "n_ties_excluded": n_tie,
    }


def p4_detection_score(detected: Dict[str, Optional[bool]],
                       truth_lost: Dict[str, Optional[bool]]) -> Dict[str, Any]:
    """P4 detection half: accuracy of 'was anything removed/condensed' against
    the frame's OWN base rate (2606.30005 gives no base rate for this probe).

    The base rate is the majority-class rate of ``truth_lost`` on the scored
    rows; the one-sided exact binomial tests accuracy > that rate.
    Unparsed answers are counted as misses AND reported.

    DEGENERACY, declared before running: on the compressed arm alone the
    'anything condensed?' truth is constant True (every c2kv row is compressed),
    so the majority base rate is 1.0 and no accuracy can beat it.  That variant
    is only scorable on the POOLED set of compressed + full-arm rows, where the
    full arm supplies the negatives; the within-arm variant that does vary is
    'was a turn DROPPED entirely' (``dropped_docs`` non-empty).  When the truth
    is constant the function reports ``degenerate: True`` and returns
    ``p_vs_base_rate: None`` rather than a meaningless p-value.
    """
    qids = [q for q in truth_lost if truth_lost.get(q) is not None]
    n = len(qids)
    n_unparsed = sum(1 for q in qids if detected.get(q) is None)
    hits = sum(1 for q in qids if detected.get(q) is not None
               and bool(detected[q]) == bool(truth_lost[q]))
    pos = sum(1 for q in qids if bool(truth_lost[q]))
    base = max(pos, n - pos) / n if n else float("nan")
    degenerate = bool(n == 0 or pos == 0 or pos == n)
    lo, hi = C.clopper_pearson(hits, n)
    return {
        "n": n,
        "n_positive_truth": pos,
        "prevalence_truth": (pos / n) if n else None,
        "majority_base_rate": base,
        "accuracy": (hits / n) if n else None,
        "ci95": [lo, hi],
        "degenerate": degenerate,
        "p_vs_base_rate": (None if degenerate
                           else C.exact_binom_one_sided(hits, n, base)),
        "n_unparsed_counted_as_miss": n_unparsed,
        "note": ("truth is constant on this row subset: pool the full-arm S0 "
                 "rows to score this half" if degenerate else None),
    }


def p4_localisation_table(reported_block: Dict[str, Optional[int]],
                          truth_blocks: Dict[str, Optional[Sequence[int]]]) -> Dict[str, Any]:
    """P4 'which turn' half, scored with :func:`t34_common.locate_table` against
    the 25.0 % wrong-block floor.

    Pre-registered None handling (card pitfall (f), the k*=None analogue):
    a row whose truth block set is EMPTY or None has no defined answer and is
    excluded from the denominator (reported as ``n_undefined``); a row with a
    defined truth where the model names no block is an ABSTENTION, which
    :func:`locate_table` counts in the denominator as a miss and reports.
    """
    hits: Dict[str, Optional[bool]] = {}
    n_undefined = 0
    for qid, truth in truth_blocks.items():
        if not truth:
            n_undefined += 1
            continue
        got = reported_block.get(qid)
        hits[qid] = None if got is None else bool(int(got) in set(int(t) for t in truth))
    table = C.locate_table(hits, label="vista_p4_reported_turn")
    table["n_undefined_excluded"] = n_undefined
    return table


# ---- the ledger (2606.30005 §2.4 / §4, tail placement) --------------------

#: The ONLY fields the ledger may render.  All are compressed-arm side (S8/S9/
#: S10); none is a target, gold, scoring or full-arm quantity.
LEDGER_FIELDS: Tuple[str, ...] = (
    "doc_chunks", "kept_history_tokens", "gist_tokens",
    "actual_compression_ratio", "dropped_docs", "hybrid_top_k",
    "repair_frame_delta",
)


def build_ledger_line(fields: Dict[str, Any]) -> str:
    """One factual ledger line (2606.30005 §2.4 dashboard, ported to our S8/S9/
    S10 columns).

    Reads ONLY :data:`LEDGER_FIELDS`; any other key raises, which is the coded
    form of card pitfall (a) ("write the label as a function and mechanically
    assert the ledger builder reads none of its inputs" — the ``a_made_call``
    precedent is why this is an assertion and not an intention).
    """
    unknown = sorted(set(fields) - set(LEDGER_FIELDS))
    if unknown:
        raise ValueError(f"ledger may not render non-S8/S9/S10 fields: {unknown}")
    parts: List[str] = []
    n = fields.get("doc_chunks")
    if n is not None:
        parts.append(f"earlier turns held: {int(n)}")
    kept = fields.get("kept_history_tokens")
    if kept is not None:
        parts.append(f"their original size: {int(kept)} tokens")
    gist = fields.get("gist_tokens")
    if gist is not None:
        parts.append(f"stored compressed as: {int(gist)} slots")
    ratio = fields.get("actual_compression_ratio")
    if ratio is not None:
        parts.append(f"compression ratio: {float(ratio):.2f}")
    dropped = fields.get("dropped_docs")
    if dropped is not None:
        # ``dropped_docs`` indexes the POST-SPLIT history, so these numbers are
        # in the ORIGINAL-conversation frame while P2/P3 count visible turns
        # (:data:`NUMBERING_FRAMES`).  The frame is named in the line itself:
        # unlabelled, the two frames contradict each other inside the same
        # +ledger prompt.
        rendered = ", ".join(str(int(d) + 1) for d in dropped) if dropped else "none"
        parts.append("earlier turns dropped entirely, counted in the original "
                     f"conversation: {rendered}")
    topk = fields.get("hybrid_top_k")
    if topk is not None:
        parts.append(f"turns kept uncompressed: {int(topk)}")
    delta = fields.get("repair_frame_delta")
    if delta is not None:
        parts.append(f"position ledger delta: {int(delta)}")
    return "Context state — " + "; ".join(parts) + "."


def ledger_message(ledger_line: str, *, role: str = "system") -> Dict[str, str]:
    return {"role": role, "content": ledger_line}


def current_messages_for_probe(query_text: str, *, ledger_line: Optional[str] = None,
                               role: str = "system") -> List[Dict[str, str]]:
    """The tail segment handed to the harness as ``prefix['current_messages']``.

    Everything before it (system prompt + compressed history) stays in the KV
    prefix untouched, so the ledger lands AFTER the compressed history and
    BEFORE the current query — 2606.30005 card pitfall (c), the single hard
    engineering constraint transferred from that paper.
    """
    out: List[Dict[str, str]] = []
    if ledger_line:
        out.append(ledger_message(ledger_line, role=role))
    out.append({"role": "user", "content": query_text})
    return out


def assemble_probe_messages(history_messages: Sequence[Dict[str, Any]],
                            query_text: str,
                            *, ledger_line: Optional[str] = None,
                            role: str = "system") -> List[Dict[str, Any]]:
    """Logical message assembly (history, then ledger, then query).

    On the server the history is a KV prefix, not a message list; this function
    is the testable statement of the ORDER the prefix + tail produce.
    """
    return list(history_messages) + current_messages_for_probe(
        query_text, ledger_line=ledger_line, role=role)


def ledger_position(messages: Sequence[Dict[str, Any]], ledger_line: str) -> Dict[str, Any]:
    """Where the ledger sits in an assembled list — the assertion the test makes."""
    idx = next((i for i, m in enumerate(messages)
                if (m.get("content") or "") == ledger_line), None)
    return {
        "index": idx,
        "n_messages": len(messages),
        "after_all_history": idx is not None and idx == len(messages) - 2,
        "before_query": idx is not None and idx < len(messages) - 1,
    }


# ---- plan / gate / cost ---------------------------------------------------

def probe_plan_rows(sidecar_rows: Iterable[Dict[str, Any]],
                    battery_by_qid: Dict[str, Dict[str, Any]],
                    *, condition: str, seed: int = 20260905,
                    qids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Deterministic (qid, probe) plan: query text, sampled blocks, truths.

    ``condition`` is ``minus_ledger`` or ``plus_ledger`` (2606.30005 §14's
    -dash/+dash).  The truths travel with the plan so scoring never has to
    re-sample.  Every -ledger query is leak-scanned here, not later.
    """
    if condition not in ("minus_ledger", "plus_ledger"):
        raise ValueError("condition must be minus_ledger or plus_ledger")
    keep = set(qids) if qids is not None else None
    rows: List[Dict[str, Any]] = []
    for sc in sidecar_rows:
        qid = sc["qid"]
        if keep is not None and qid not in keep:
            continue
        lens = [int(v) for v in (sc.get("doc_lengths") or [])]
        dropped = [int(v) for v in (sc.get("dropped_docs") or [])]
        bat = battery_by_qid.get(qid, {})
        ledger_line = None
        if condition == "plus_ledger":
            ledger_line = build_ledger_line({
                "doc_chunks": bat.get("doc_chunks"),
                "kept_history_tokens": bat.get("kept_history_tokens"),
                "gist_tokens": bat.get("gist_tokens"),
                "actual_compression_ratio": bat.get("actual_compression_ratio"),
                "dropped_docs": dropped,
                "hybrid_top_k": bat.get("hybrid_top_k"),
            })
        # ``doc_lengths`` parallels the sidecar's ``docs``, i.e. the KEPT blocks
        # the model actually saw (t34_dump_sidecar.py:19-33).  P2/P3 therefore
        # sample and address VISIBLE turns (:data:`NUMBERING_FRAMES`), and
        # ``sample_blocks`` caps k at that count: a row with one visible block
        # yields ONE P2 pair and no P3 request at all.
        n_docs = len(lens)
        p2_blocks = sample_blocks(qid + ":P2", n_docs, P2_N_BLOCKS, seed)
        p3_blocks = sample_blocks(qid + ":P3", n_docs, 2, seed)
        for probe in PROBE_IDS:
            if probe == "P2" and not p2_blocks:
                continue
            if probe == "P3" and len(p3_blocks) != 2:
                continue
            blocks = p2_blocks if probe == "P2" else (p3_blocks if probe == "P3" else None)
            query = build_probe_query(probe, blocks=blocks)
            leaks = scan_prompt_for_state_leak(query)
            if leaks:
                raise ValueError(f"probe query leaks context state {leaks!r}")
            truth: Dict[str, Any] = {}
            if probe == "P1":
                truth = {"total_tokens": bat.get("kept_history_tokens"),
                         "sidecar_total_tokens": sum(lens) if lens else None,
                         "compressed_history_tokens": bat.get("compressed_history_tokens")}
            elif probe == "P2":
                truth = {"blocks": blocks,
                         "block_tokens": [lens[i] for i in blocks]}
            elif probe == "P3":
                truth = {"blocks": p3_blocks,
                         "size_a": lens[p3_blocks[0]], "size_b": lens[p3_blocks[1]]}
            else:
                truth = {"lost_dropped": bool(dropped),
                         "lost_any": bool(dropped) or _ratio_indicates_loss(bat),
                         "dropped_docs": dropped}
            rows.append({
                "qid": qid,
                "session_id": sc.get("session_id") or C.session_of(qid),
                "probe": probe,
                "condition": condition,
                "query": query,
                "ledger": ledger_line,
                "blocks": blocks,
                "truth": truth,
            })
    return rows


def _ratio_indicates_loss(battery_row: Dict[str, Any]) -> bool:
    """P4 truth also counts 'condensed', not only 'dropped': a compression
    ratio > 1 means the earlier conversation reached the model condensed."""
    ratio = battery_row.get("actual_compression_ratio")
    try:
        return ratio is not None and float(ratio) > 1.0
    except (TypeError, ValueError):
        return False


def score_probe_file(probe_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Score an executed probe file with VISTA's own rules (2606.30005 §14)."""
    by_probe: Dict[str, List[Dict[str, Any]]] = {p: [] for p in PROBE_IDS}
    for r in probe_rows:
        by_probe.setdefault(r.get("probe"), []).append(r)

    out: Dict[str, Any] = {"conditions": sorted({r.get("condition") for r in probe_rows}),
                           "n_rows": len(probe_rows)}

    # P1 has TWO defensible truths on a compressed arm and the question text
    # ("how much of the earlier conversation is currently available to you")
    # does not disambiguate them, so both are scored and both are printed:
    # the pre-compression size of the history that was fed in (headline, the
    # card's ``original_seq_len`` slot) and the compressed size actually
    # occupying the prefix.  Picking whichever came out better afterwards would
    # be selection on the result.
    p1_gen = [parse_integer_answer(r.get("generation") or "")
              for r in by_probe.get("P1", [])]
    p1_truth = [(r.get("truth") or {}) for r in by_probe.get("P1", [])]
    out["P1"] = median_relative_error(
        list(zip(p1_gen, [t.get("total_tokens") for t in p1_truth])))
    out["P1"]["truth_column"] = "kept_history_tokens (pre-compression)"
    out["P1_vs_compressed_history_tokens"] = median_relative_error(
        list(zip(p1_gen, [t.get("compressed_history_tokens") for t in p1_truth])))
    out["P1_vs_compressed_history_tokens"]["truth_column"] = \
        "compressed_history_tokens (post-compression)"

    p2_pairs: List[Tuple[Optional[float], Optional[float]]] = []
    n_p2_rows_unparsed = 0
    p2_hist: Dict[str, int] = {}
    for r in by_probe.get("P2", []):
        toks = (r.get("truth") or {}).get("block_tokens") or []
        # 2606.30005 §4 scores INDIVIDUAL block size: one pair per sampled
        # block, never one pair for their sum.
        preds = parse_integer_list_answer(r.get("generation") or "", len(toks))
        if preds is None:
            n_p2_rows_unparsed += 1
        p2_hist[str(len(toks))] = p2_hist.get(str(len(toks)), 0) + 1
        for i, tok in enumerate(toks):
            p2_pairs.append((None if preds is None else preds[i], float(tok)))
    out["P2"] = median_relative_error(p2_pairs)
    out["P2"]["n_requests"] = len(by_probe.get("P2", []))
    out["P2"]["n_requests_unparsed"] = n_p2_rows_unparsed
    out["P2"]["blocks_per_request_requested"] = P2_N_BLOCKS
    # ``sample_blocks`` caps k at the number of VISIBLE blocks, so a row with
    # one visible block contributes ONE pair, not four: the per-block
    # denominator varies by row and the histogram travels with the number.
    out["P2"]["blocks_per_request_histogram"] = dict(sorted(p2_hist.items()))
    out["P2"]["n_blocks_asked"] = sum(int(k) * v for k, v in p2_hist.items())
    out["P2"]["estimand"] = "per_block_relative_error"

    p3_items = []
    for r in by_probe.get("P3", []):
        t = r.get("truth") or {}
        p3_items.append((parse_choice_answer(r.get("generation") or ""),
                         t.get("size_a"), t.get("size_b")))
    out["P3"] = pairwise_hard_subset_accuracy(p3_items)
    # P3 needs two visible blocks, so its request count is BELOW the row count
    # of the other probes; print it beside every P3 number.
    out["P3"]["n_requests"] = len(by_probe.get("P3", []))
    out["P3"]["n_qids"] = len({r.get("qid") for r in by_probe.get("P3", [])})

    detected: Dict[str, Optional[bool]] = {}
    reported_block: Dict[str, Optional[int]] = {}
    truth_dropped: Dict[str, Optional[bool]] = {}
    truth_any: Dict[str, Optional[bool]] = {}
    truth_blocks: Dict[str, Optional[Sequence[int]]] = {}
    for r in by_probe.get("P4", []):
        qid = r["qid"]
        parsed = parse_loss_answer(r.get("generation") or "")
        detected[qid] = parsed["detected"]
        reported_block[qid] = parsed["block"]
        t = r.get("truth") or {}
        truth_dropped[qid] = bool(t.get("lost_dropped"))
        truth_any[qid] = bool(t.get("lost_any"))
        truth_blocks[qid] = t.get("dropped_docs")
    # within-arm variant (varies): was a turn dropped entirely
    out["P4_detection"] = p4_detection_score(detected, truth_dropped)
    # pooled variant (needs the full-arm S0 rows to have negatives)
    out["P4_detection_any_condensed"] = p4_detection_score(detected, truth_any)
    out["P4_detection_any_condensed"]["pooled_scorer"] = (
        "degenerate within one arm: score it with pooled_p4_detection() over "
        "this file PLUS the --arm full probe file (score-p4-pooled)")
    out["arms_present"] = sorted({str(r.get("arm")) for r in probe_rows})
    out["P4_localisation"] = p4_localisation_table(reported_block, truth_blocks)

    n_parsed = sum(1 for r in probe_rows if _row_parsed(r))
    out["parse_rate"] = (n_parsed / len(probe_rows)) if probe_rows else None
    # VISTA card pitfall (b): the probes run against a prefix built by the
    # FROZEN battery run, so every P1/P2 number inherits that run's generation
    # caliber and is not comparable to one taken at another.  Unmeasurable from
    # here; it travels with the numbers so it cannot be dropped in a table.
    out["caliber_caveat"] = (
        "P1/P2 are proprioception of the frozen battery's own compressed "
        "prefix: label every number with the battery caliber it was taken at, "
        "or re-run per battery4096_adjudication.md")
    out["generate_sec_total"] = float(sum(float(r.get("generate_sec") or 0.0)
                                          for r in probe_rows))
    return out


def pooled_p4_detection(probe_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """P4 detection scored on the POOLED arms — the only way the 'was anything
    condensed?' half is scorable at all.

    On the compressed arm alone that truth is constant True (every c2kv row is
    compressed), so :func:`p4_detection_score` reports ``degenerate`` and no
    accuracy can beat the base rate.  The negatives live on the FULL arm, which
    is why the RUNBOOK's ``--arm full`` probe pass is mandatory rather than
    optional.  This function pools executed P4 rows from both arms, keyed
    ``arm|qid``, and scores them with the same rule.

    EACH ROW BRINGS ITS OWN TRUTH from its own plan (built against that arm's
    sidecar): nothing here assumes the full arm dropped nothing — if the
    harness's tail window drops blocks in the full arm too, that shows up in
    the full-arm plan's ``lost_any`` and is scored as it is.

    Returns ``{"available": False, ...}`` with no numbers when only one arm is
    present or when the pooled truth is still constant, instead of a p-value
    that would be meaningless.
    """
    rows = [r for r in probe_rows if r.get("probe") == "P4"]
    detected: Dict[str, Optional[bool]] = {}
    truth_any: Dict[str, Optional[bool]] = {}
    by_arm: Dict[str, int] = {}
    for r in rows:
        arm = str(r.get("arm") or "unknown_arm")
        key = f"{arm}|{r['qid']}"
        detected[key] = parse_loss_answer(r.get("generation") or "")["detected"]
        truth_any[key] = bool((r.get("truth") or {}).get("lost_any"))
        by_arm[arm] = by_arm.get(arm, 0) + 1
    n_pos = sum(1 for v in truth_any.values() if v)
    n = len(truth_any)
    if len(by_arm) < 2 or n_pos in (0, n):
        return {
            "available": False,
            "detection": None,
            "n_rows_by_arm": dict(sorted(by_arm.items())),
            "n_positive_truth": n_pos,
            "n": n,
            "reason": ("pooled P4 needs BOTH arms and a truth that varies: "
                       "run agent/t34_probe_mode.py with --arm full over the "
                       "full-arm plan (RUNBOOK step 2) before reading this "
                       "half"),
        }
    res = p4_detection_score(detected, truth_any)
    return {
        "available": True,
        "detection": res,
        "n_rows_by_arm": dict(sorted(by_arm.items())),
        "estimand": "was anything removed or condensed (pooled over arms)",
        "caveat": ("the arm is a perfectly predictive cue for an observer who "
                   "knows it; the model is not told which arm it is running "
                   "in, and the two arms share the same query text"),
    }


def _row_parsed(row: Dict[str, Any]) -> bool:
    probe = row.get("probe")
    gen = row.get("generation") or ""
    if probe == "P1":
        return parse_integer_answer(gen) is not None
    if probe == "P2":
        n = len((row.get("truth") or {}).get("block_tokens") or [])
        return parse_integer_list_answer(gen, n) is not None
    if probe == "P3":
        return parse_choice_answer(gen) is not None
    if probe == "P4":
        return parse_loss_answer(gen)["detected"] is not None
    return False


def ledger_gate(minus_ledger_result: Dict[str, Any],
                gate: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """'Run +ledger only if -ledger passes' (digest §4.12).

    PASS iff the -ledger condition produced a usable measurement (parse rate
    >= ``min_parse_rate``) AND the model shows some proprioception, i.e. either
    the P4 detection half beats its own base rate at ``p4_alpha`` or a size
    probe's median relative error is better than VISTA's 0.43 blindness edge.
    A FAIL is the SPEC §8.2 item 6 stopping condition: no extra LLM call is
    worth spending, and the -ledger table is the deliverable.
    """
    g = dict(LEDGER_GATE)
    g.update(gate or {})
    parse_rate = minus_ledger_result.get("parse_rate")
    p4 = minus_ledger_result.get("P4_detection") or {}
    p4_p = p4.get("p_vs_base_rate")
    errs = [(minus_ledger_result.get(k) or {}).get("median_relative_error")
            for k in ("P1", "P2")]
    errs = [e for e in errs if e is not None]
    best_err = min(errs) if errs else None
    reasons: List[str] = []
    usable = parse_rate is not None and parse_rate >= g["min_parse_rate"]
    if not usable:
        reasons.append(f"parse_rate {parse_rate} < {g['min_parse_rate']}")
    p4_ok = p4_p is not None and p4_p < g["p4_alpha"]
    size_ok = best_err is not None and best_err < g["size_error_max"]
    if not p4_ok:
        reasons.append(f"P4 detection p={p4_p} not < {g['p4_alpha']}")
    if not size_ok:
        reasons.append(f"best size median relative error {best_err} "
                       f"not < {g['size_error_max']}")
    passed = bool(usable and (p4_ok or size_ok))
    return {
        "run_plus_ledger": passed,
        "passed": passed,
        "gate": g,
        "parse_rate": parse_rate,
        "p4_p_vs_base_rate": p4_p,
        "best_size_median_relative_error": best_err,
        "reasons": reasons,
        "verdict": ("selfcheck arm alive: P4/size probes show proprioception"
                    if passed else
                    "stopping condition: the compressed arm cannot perceive its "
                    "own compression state; SPEC 8.2 item 6 selfcheck arm is not "
                    "worth an extra LLM call"),
    }


def p4_selfcheck_gate(minus_ledger_result: Dict[str, Any],
                      gate: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """The SELFCHECK precondition — P4 only (2606.30005 card, deliverable (A)).

    The card is explicit about which half of the diagnostic gates the arm:
    *"If P1/P2 median relative error lands in VISTA's 0.43-0.84 band AND P4 is
    at or near the base rate, then the model cannot perceive its own
    compression state and SPEC §8.2.6's I1 selfcheck arm is not worth an extra
    LLM call... If P4 is materially above the base rate, the self-check arm is
    alive."*  Size proprioception alone therefore does NOT license the arm, so
    this gate keys on the P4 detection half and nothing else — unlike
    :func:`ledger_gate`, which decides only whether to spend the +ledger half of
    the budget and may pass on either signal.

    A DEGENERATE P4 (constant truth on the compressed arm) is reported as
    ``undetermined`` rather than pass or fail: the pooled full-arm S0 probe rows
    are needed before the question has an answer at all.
    """
    g = dict(LEDGER_GATE)
    g.update(gate or {})
    parse_rate = minus_ledger_result.get("parse_rate")
    p4 = minus_ledger_result.get("P4_detection") or {}
    p4_p = p4.get("p_vs_base_rate")
    degenerate = bool(p4.get("degenerate"))
    reasons: List[str] = []
    usable = parse_rate is not None and parse_rate >= g["min_parse_rate"]
    if not usable:
        reasons.append(f"parse_rate {parse_rate} < {g['min_parse_rate']}")
    if degenerate:
        reasons.append("P4 detection truth is constant on this arm: pool the "
                       "full-arm S0 probe rows before this gate can be read")
    p4_ok = (not degenerate) and p4_p is not None and p4_p < g["p4_alpha"]
    if not p4_ok and not degenerate:
        reasons.append(f"P4 detection p={p4_p} not < {g['p4_alpha']}")
    passed = bool(usable and p4_ok)
    return {
        "passed": passed,
        "undetermined": bool(degenerate),
        "gate": g,
        "parse_rate": parse_rate,
        "p4_p_vs_base_rate": p4_p,
        "p4_degenerate": degenerate,
        "reasons": reasons,
        "verdict": ("P4 above its base rate: the selfcheck arm's precondition "
                    "holds" if passed else
                    "P4 gate not met: the compressed arm shows no measured loss "
                    "awareness, so SPEC 8.2 item 6's selfcheck arm is not worth "
                    "an extra LLM call"),
    }


def probe_cost_report(*, n_rows: int = 161, n_probes: int = 4, n_conditions: int = 2,
                      sec_per_generation: float = 11.1) -> Dict[str, Any]:
    """Pre-run cost (digest §4.12: 4 probes x 161 rows x 2 conditions = 1288
    generations; 11.1 s/gen is the measured battery rate cited by the card)."""
    gens = n_rows * n_probes * n_conditions
    sec = gens * sec_per_generation
    return {
        "generations": gens,
        "sec_per_generation": sec_per_generation,
        "gpu_seconds": sec,
        "gpu_hours": sec / 3600.0,
        "halved_by_gate_gpu_hours": sec / 7200.0,
        "note": "run minus_ledger first; +ledger only if ledger_gate() passes",
    }


# --------------------------------------------------------------------------
# (B) MemGPT self-emitted page-in — 2310.08560
# --------------------------------------------------------------------------

RELOAD_TOOL_NAME = "request_history_reload"

#: 2310.08560 §2.2 gives these as "e.g." defaults with no calibration set,
#: no sweep and no sensitivity analysis.  Declared, never presented as tuned.
MEMGPT_WARNING_FRAC = 0.70
MEMGPT_FLUSH_FRAC = 1.00
MEMGPT_EVICTION_FRAC = 0.50


def reload_tool_schema() -> Dict[str, Any]:
    """The parameterless page-in tool (2310.08560 §2.3, argument-free per the
    digest so the request carries zero localisation)."""
    return {
        "type": "function",
        "function": {
            "name": RELOAD_TOOL_NAME,
            "description": ("Request that the earlier conversation be made "
                            "available again in full. Takes no arguments."),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }


def reload_tool_schema_block_variant_offline_diagnostic() -> Dict[str, Any]:
    """OFFLINE DIAGNOSTIC ONLY — never a positive arm.

    2310.08560 §2.3's recall functions take arguments; the digest keeps the
    block-id form out of every headline arm because it re-exposes the
    localisation red line.  Kept so the diagnostic can be scored, and named so
    it cannot be wired in by accident.
    """
    schema = reload_tool_schema()
    schema["function"]["parameters"] = {
        "type": "object",
        "properties": {"block_id": {"type": "integer",
                                    "description": "1-based earlier turn to restore"}},
        "required": ["block_id"],
    }
    return schema


def intercept(action_canonical: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    """MemGPT interception rule as a pure function (2310.08560 §2.3).

    ``action_canonical`` is ``benchmarks/proxy.py:466``'s canonical response
    form ``{"tool_calls": [{"name", "arguments"}], "text"}``.  ``state`` is
    ``{"conv": <conv id>, "reloaded_convs": [...], "max_reloads_per_conv": 1}``.

    Returns ``fired`` plus, when fired, the recovery primitive to run instead of
    counting the step as a task action: append-only tail placement at the fixed
    first/offset:0 target (the digest's §3.2 best fixed policy), and a
    ``state_update`` the caller applies.  The function mutates nothing.
    """
    names = [str((c or {}).get("name") or "")
             for c in (action_canonical or {}).get("tool_calls") or []]
    conv = state.get("conv")
    done = set(state.get("reloaded_convs") or [])
    budget = int(state.get("max_reloads_per_conv", 1))
    if RELOAD_TOOL_NAME not in names:
        return {"fired": False, "recovery": None, "reason": "no_reload_call",
                "state_update": None}
    if budget <= 0 or conv in done:
        # 2310.08560 §2.2's bookkeeping analogue: one reload per conversation
        # keeps match / diverged_now / re_diverged / tracking_lost meaningful.
        return {"fired": False, "recovery": None, "reason": "reload_budget_exhausted",
                "state_update": None}
    return {
        "fired": True,
        "recovery": {"placement": "append_tail", "target": "first",
                     "target_spec": "offset:0", "reissue_same_request": True},
        "reason": "self_emitted_reload",
        "state_update": {"reloaded_convs": sorted(done | {conv})},
    }


def page_out_signal(prompt_tokens: int, window: int, *,
                    warning_frac: float = MEMGPT_WARNING_FRAC,
                    flush_frac: float = MEMGPT_FLUSH_FRAC,
                    eviction_frac: float = MEMGPT_EVICTION_FRAC) -> Dict[str, Any]:
    """2310.08560 §2.2 queue-manager counters (optional helper).

    ``prompt_tokens > warning_token_count`` injects the memory-pressure notice;
    ``> flush_token_count`` flushes ~``eviction_frac`` of the window.  All three
    fractions are the paper's illustrative defaults.
    """
    if window <= 0:
        raise ValueError("window must be positive")
    frac = float(prompt_tokens) / float(window)
    return {
        "pressure_ratio": frac,
        "warning": frac > warning_frac,
        "flush": frac > flush_frac,
        "eviction_tokens": int(round(eviction_frac * window)) if frac > flush_frac else 0,
        "defaults_are_paper_examples": True,
    }


def memory_pressure_notice(pressure_ratio: float) -> str:
    """MemGPT's in-band notice, ported in spirit (2310.08560 §2.2).

    WARNING — injecting this changes the raw message bytes, so the bench face's
    ``RecoverState.check`` fingerprint (raw-message sha256, proxy.py:494) will
    no longer match the reference run and the pairing degrades silently to
    ``tracking_lost``.  Either inject into BOTH arms or fingerprint the
    pre-injection message list; decide in code before any run (card pitfall (d)).
    """
    return ("System notice: part of the earlier conversation is held in "
            f"compressed form (working set at {pressure_ratio:.0%} of budget). "
            f"You may call {RELOAD_TOOL_NAME}() to have it restored.")


def fire_rate_smoke(proxy_rows: Sequence[Dict[str, Any]], *, min_convs: int = 30
                    ) -> Dict[str, Any]:
    """MemGPT smoke test with its stopping rule (digest §4.12 / card pitfall (c)).

    Counts conversations in a bench proxy log and how many emitted a reload
    call.  Fire rate 0 over >= ``min_convs`` conversations => STOP (stopping
    condition (1) implementation-invalid / (2) no headroom).
    """
    convs: Dict[str, int] = {}
    fires = 0
    for r in proxy_rows:
        conv = str(r.get("conv_id"))
        convs.setdefault(conv, 0)
        if _row_emitted_reload(r):
            convs[conv] += 1
            fires += 1
    n_convs = len(convs)
    fired_convs = sum(1 for v in convs.values() if v > 0)
    rate = (fired_convs / n_convs) if n_convs else None
    stop = bool(n_convs >= min_convs and fired_convs == 0)
    return {
        "n_conversations": n_convs,
        "n_conversations_with_fire": fired_convs,
        "n_fires": fires,
        "fire_rate_per_conversation": rate,
        "min_convs": min_convs,
        "stop": stop,
        "stop_reason": ("fire rate 0 on >= %d conversations: stop (MemGPT card "
                        "pitfall (c))" % min_convs) if stop else None,
        "tracking_lost_rows": sum(1 for r in proxy_rows if r.get("tracking_lost")),
    }


def _row_emitted_reload(row: Dict[str, Any]) -> bool:
    action = row.get("action") or {}
    if isinstance(action, dict):
        names = [str((c or {}).get("name") or "")
                 for c in (action.get("tool_calls") or [])]
        return RELOAD_TOOL_NAME in names
    return False


def page_in_metrics(proxy_rows: Sequence[Dict[str, Any]],
                    oracle_cw_steps: Dict[str, Sequence[int]],
                    *, cc_convs: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """The three metrics + cost for the page-in arm (digest §4.12).

    coverage = share of oracle C->W steps at which a reload was emitted AT that
    step or the step BEFORE; precision = share of fires landing on an oracle
    C->W step; false-reset = share of C->C conversations carrying >= 1 fire;
    cost = 2 x ``generate_sec`` per fire (the fired generation + the re-issue).
    Every denominator is returned.
    """
    fires: Dict[str, List[int]] = {}
    cost = 0.0
    for r in proxy_rows:
        if not _row_emitted_reload(r):
            continue
        conv = str(r.get("conv_id"))
        turn = r.get("turn")
        fires.setdefault(conv, []).append(int(turn) if turn is not None else -1)
        cost += 2.0 * float(r.get("generate_sec") or r.get("wall_sec") or 0.0)
    n_cw = sum(len(v) for v in oracle_cw_steps.values())
    covered = 0
    for conv, steps in oracle_cw_steps.items():
        got = set(fires.get(conv, []))
        for s in steps:
            if int(s) in got or int(s) - 1 in got:
                covered += 1
    n_fires = sum(len(v) for v in fires.values())
    on_target = 0
    for conv, turns in fires.items():
        oracle = set(int(s) for s in oracle_cw_steps.get(conv, []))
        on_target += sum(1 for t in turns if t in oracle)
    cc = list(cc_convs or [])
    cc_fired = sum(1 for c in cc if fires.get(c))
    return {
        "coverage": (covered / n_cw) if n_cw else None,
        "n_cw_steps": n_cw,
        "precision": (on_target / n_fires) if n_fires else None,
        "n_fires": n_fires,
        "false_reset_rate": (cc_fired / len(cc)) if cc else None,
        "n_cc_conversations": len(cc),
        "cost_generate_sec_2x": cost,
    }


# --------------------------------------------------------------------------
# (C) same-compressed-context selfcheck arm — SPEC §8.2 item 6
# --------------------------------------------------------------------------

SELFCHECK_QUESTION = (
    "Is the tool call you just produced consistent with the earlier conversation "
    "you can see? Reply with yes or no, then a consistency score between 0 and 1."
)

#: SPEC §8.2 item 6 generation contract for the second call.
SELFCHECK_GEN = {"temperature": 0.0, "enable_thinking": False, "max_new_tokens": 32}


def selfcheck_current_messages(current_messages: Sequence[Dict[str, Any]],
                               prediction: str,
                               *, prefix_reuse: bool = True) -> List[Dict[str, Any]]:
    """Append the selfcheck question after the emitted call, IN THE SAME
    compressed context (SPEC §8.2 item 6).

    With ``prefix_reuse=True`` the returned list starts with the original
    current messages verbatim, so the system + compressed-history KV prefix and
    the already-prefilled query tokens stay byte-identical and only the short
    tail is re-prefilled.  ``prefix_reuse=False`` returns the question alone and
    is NOT the arm (kept only so a test can show the difference).
    """
    tail = [{"role": "assistant", "content": prediction or ""},
            {"role": "user", "content": SELFCHECK_QUESTION}]
    if not prefix_reuse:
        return tail
    return list(current_messages) + tail


def parse_selfcheck(text: str) -> Dict[str, Any]:
    """Parse the yes/no + 0-1 score answer.  Unparsed halves stay None."""
    consistent = parse_yes_no_answer(text)
    t = _strip_thinking(text)
    score: Optional[float] = None
    for m in re.finditer(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)(?![\d])", t):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if 0.0 <= val <= 1.0:
            score = val
            break
    if score is None:
        m = re.search(r"(\d{1,3})\s*%", t)
        if m is not None:
            score = min(1.0, int(m.group(1)) / 100.0)
    return {"consistent": consistent, "score": score,
            "parse_ok": consistent is not None or score is not None}


def selfcheck_feature_rows(probe_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-qid FEATURE row for the selfcheck arm.

    Columns (orientations in configs/t34/orientations_selfreport.json):
      ``selfcheck_risk``            1 - self-reported consistency score  (+1)
      ``selfcheck_says_inconsistent`` 1 when the model answered 'no'      (+1)
      ``selfcheck_answer_parse_ok`` the selfcheck answer itself parsed    (-1)
      ``selfcheck_generate_sec``    its own cost column, never merged into
                                    the model's generate_sec.
    None stays None: an unparsed selfcheck answer is NOT imputed and NOT
    given a whole-output fallback.
    """
    rows: List[Dict[str, Any]] = []
    for r in probe_rows:
        parsed = parse_selfcheck(r.get("generation") or "")
        score = parsed["score"]
        rows.append({
            "qid": r["qid"],
            "session_id": r.get("session_id") or C.session_of(r["qid"]),
            "selfcheck_risk": (1.0 - float(score)) if score is not None else None,
            "selfcheck_says_inconsistent": (
                None if parsed["consistent"] is None
                else (0.0 if parsed["consistent"] else 1.0)),
            "selfcheck_answer_parse_ok": 1.0 if parsed["parse_ok"] else 0.0,
            "selfcheck_generate_sec": r.get("generate_sec"),
        })
    return rows


def cofire_with_parse_failure(selfcheck_fires: Dict[str, Optional[bool]],
                              parse_fail_fires: Dict[str, bool]) -> Dict[str, Any]:
    """The 'was it worth it' criterion (digest §4.12): the fraction of selfcheck
    fires that coincide with parse-failure fires.

    A selfcheck arm that fires on the same steps where the call already fails to
    parse has added nothing over the L1 baseline it must beat.

    A ``None`` selfcheck value means the answer did not parse.  It is EXCLUDED
    from every denominator and reported as ``n_undefined_selfcheck`` — an
    unparsed self-report is not evidence of "did not fire", and silently
    reading it as False would understate the co-firing fraction.
    """
    qids = sorted(set(selfcheck_fires) & set(parse_fail_fires))
    undefined = [q for q in qids if selfcheck_fires.get(q) is None]
    defined = [q for q in qids if selfcheck_fires.get(q) is not None]
    sc = [q for q in defined if selfcheck_fires[q]]
    pf = [q for q in defined if parse_fail_fires.get(q)]
    both = [q for q in sc if parse_fail_fires.get(q)]
    union = set(sc) | set(pf)
    return {
        "n_rows": len(defined),
        "n_undefined_selfcheck": len(undefined),
        "n_selfcheck_fires": len(sc),
        "n_parse_fail_fires": len(pf),
        "n_cofire": len(both),
        "fraction_of_selfcheck_fires_cofiring": (len(both) / len(sc)) if sc else None,
        "jaccard": (len(both) / len(union)) if union else None,
        "n_selfcheck_only": len(sc) - len(both),
    }


def selfcheck_label_control_s0(scores_c2kv: Sequence[Optional[float]],
                               scores_full: Sequence[Optional[float]],
                               y: Sequence[int],
                               sessions: Sequence[str],
                               *, orientation: int = 1,
                               reps: int = 2000) -> Dict[str, Any]:
    """LABEL-SIDE CONTROL — the S0 twin (§4.0 winner rule).

    Runs the SAME feature on the FULL arm and reports Delta AUPRC with a
    session-clustered paired bootstrap.  This reads a full-arm quantity, so it
    lives in a ``*_label_*`` function and its output never enters a feature
    frame.  Orientation is applied BEFORE the comparison (never residualise or
    compare an un-oriented score).
    """
    a = np.asarray([np.nan if v is None else float(v) for v in scores_c2kv], dtype=float)
    b = np.asarray([np.nan if v is None else float(v) for v in scores_full], dtype=float)
    yy = np.asarray(list(y), dtype=int)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b, yy = orientation * a[ok], orientation * b[ok], yy[ok]
    clusters = C.session_clusters([s for s, k in zip(sessions, ok) if k])
    point, lo, hi = C.paired_delta_bootstrap(C.average_precision, a, b, yy, clusters,
                                             reps=reps)
    return {
        "n_scored": int(ok.sum()),
        "n_pos": int(yy.sum()),
        "prevalence": C.prevalence(yy),
        "auprc_c2kv": C.average_precision(a, yy),
        "auprc_full_s0": C.average_precision(b, yy),
        "delta_auprc": point,
        "delta_ci95": [lo, hi],
        "alive_by_s0_rule": bool(lo is not None and lo > 0),
    }


def score_feature_table(scores: Sequence[Optional[float]], y: Sequence[int],
                        sessions: Sequence[str], *, name: str, orientation: int,
                        baselines: Optional[Dict[str, Sequence[Optional[float]]]] = None,
                        reps: int = 2000) -> Dict[str, Any]:
    """One row of the §4.0 winner table for a scalar feature.

    Orientation is applied first; AP/AUROC come with a session-clustered
    bootstrap CI; the operating point fires at the parse-failure baseline's own
    fire count so coverage/precision/false-reset are comparable; every baseline
    delta is a paired bootstrap on the SAME rows, and ``n``/``n_pos`` are
    reported so no comparison silently changes the row subset.
    """
    s = np.asarray([np.nan if v is None else float(v) for v in scores], dtype=float)
    yy = np.asarray(list(y), dtype=int)
    ok = np.isfinite(s)
    s_ok, y_ok = orientation * s[ok], yy[ok]
    clusters = C.session_clusters([g for g, k in zip(sessions, ok) if k])
    ap = C.average_precision(s_ok, y_ok)
    ap_lo, ap_hi, n_clusters = C.clustered_bootstrap(C.average_precision, s_ok, y_ok,
                                                     clusters, reps=reps)
    au = C.auroc(s_ok, y_ok)
    au_lo, au_hi, _ = C.clustered_bootstrap(C.auroc, s_ok, y_ok, clusters, reps=reps)
    out: Dict[str, Any] = {
        "feature": name,
        "orientation": orientation,
        "n": int(ok.sum()),
        "n_pos": int(y_ok.sum()),
        "n_dropped_undefined": int((~ok).sum()),
        "prevalence_chance_ap": C.prevalence(y_ok),
        "auprc": ap, "auprc_ci95": [ap_lo, ap_hi],
        "auroc": au, "auroc_ci95": [au_lo, au_hi],
        "n_session_clusters": n_clusters,
        "deltas": {},
    }
    for bname, bvals in (baselines or {}).items():
        b = np.asarray([np.nan if v is None else float(v) for v in bvals], dtype=float)
        b_ok = b[ok]
        both = np.isfinite(b_ok)
        if not both.all():
            # a baseline defined on fewer rows would change the subset
            out["deltas"][bname] = {"skipped": "baseline undefined on some rows",
                                    "n_common": int(both.sum())}
            continue
        point, lo, hi = C.paired_delta_bootstrap(C.average_precision, s_ok, b_ok,
                                                 y_ok, clusters, reps=reps)
        n_fires = int((b_ok > 0).sum())
        out["deltas"][bname] = {
            "delta_auprc": point, "ci95": [lo, hi],
            "beats_baseline": bool(lo is not None and lo > 0),
            "operating_point_at_baseline_fire_count":
                C.operating_point(s_ok, y_ok, n_fires),
            "baseline_operating_point": C.operating_point(b_ok, y_ok, n_fires),
        }
    return out


#: Pre-declared risk orientations for every feature this unit emits.
ORIENTATIONS_PATH = _HERE.parent / "configs/t34/orientations_selfreport.json"


def orientation_of(feature: str) -> int:
    """The feature's PRE-DECLARED risk orientation (+1 higher = riskier).

    Raises for an undeclared feature: a score may never be oriented (or
    residualised, or compared) on the basis of how it happened to come out.
    """
    orients = C.load_orientations(ORIENTATIONS_PATH)
    if feature not in orients:
        raise KeyError(
            f"{feature!r} has no declared orientation in {ORIENTATIONS_PATH.name}; "
            "declare it before scoring, never after looking at the result")
    return int(orients[feature])


def require_p4_gate(p4_result_path: Optional[str], *, force: bool = False) -> Dict[str, Any]:
    """The selfcheck arm is gated on VISTA P4 (digest §4.12: 'sequence is the
    only meaningful part — run VISTA's P4 first, then decide').

    Raises unless a P4 result file exists and :func:`p4_selfcheck_gate` passes
    on it, or ``force``.  The gate read here is the P4-ONLY one: a result whose
    ``ledger_gate`` passed on the size probes alone does not license the arm.
    """
    if force:
        return {"gated": False, "forced": True,
                "note": "P4 gate bypassed with --force; record it in the prereg"}
    if not p4_result_path or not Path(p4_result_path).exists():
        raise RuntimeError(
            "selfcheck arm is gated on the VISTA P4 result: run score-probes on "
            "the minus_ledger condition first and pass --p4-result, or --force")
    res = json.loads(Path(p4_result_path).read_text(encoding="utf-8"))
    gate = res.get("p4_gate") or p4_selfcheck_gate(res)
    if not gate.get("passed"):
        raise RuntimeError(
            "VISTA P4 gate FAILED — the compressed arm shows no measured loss "
            "awareness, so the selfcheck arm is not worth an extra LLM call: "
            f"{gate.get('reasons')}")
    return {"gated": True, "forced": False, "gate": gate}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _load_frame(args: argparse.Namespace):
    from t33_labels import build_label_frame, join_arms
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pairs = join_arms(load_jsonl(args.battery_full), load_jsonl(args.battery_c2kv))
    return manifest, pairs, build_label_frame(pairs, manifest)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("RUNBOOK")[0].strip())
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_cost = sub.add_parser("cost", help="print the pre-registered probe budget")
    p_cost.add_argument("--rows", type=int, default=161)
    p_cost.add_argument("--probes", type=int, default=4)
    p_cost.add_argument("--conditions", type=int, default=2)
    p_cost.add_argument("--sec-per-generation", type=float, default=11.1)

    p_plan = sub.add_parser("plan-probes", help="deterministic probe plan (no GPU)")
    p_plan.add_argument("--sidecar", required=True)
    p_plan.add_argument("--battery-c2kv", required=True)
    p_plan.add_argument("--battery-full", required=True)
    p_plan.add_argument("--manifest", required=True)
    p_plan.add_argument("--condition", choices=["minus_ledger", "plus_ledger"],
                        default="minus_ledger")
    p_plan.add_argument("--seed", type=int, default=20260905)
    p_plan.add_argument("--out", required=True)

    p_score = sub.add_parser("score-probes", help="score an executed probe file")
    p_score.add_argument("--probes", required=True)
    p_score.add_argument("--out", required=True)

    p_pool = sub.add_parser(
        "score-p4-pooled",
        help="P4 detection pooled over the c2kv and full probe files")
    p_pool.add_argument("--probes", required=True, nargs="+",
                        help="one executed probe file per ARM (c2kv and full)")
    p_pool.add_argument("--out", required=True)

    p_sc = sub.add_parser("score-selfcheck", help="selfcheck features + diagnostics")
    p_sc.add_argument("--probes", required=True)
    p_sc.add_argument("--battery-c2kv", required=True)
    p_sc.add_argument("--battery-full", required=True)
    p_sc.add_argument("--manifest", required=True)
    p_sc.add_argument("--p4-result", default=None)
    p_sc.add_argument("--force", action="store_true")
    p_sc.add_argument("--features-out", required=True)
    p_sc.add_argument("--out", required=True)

    p_pg = sub.add_parser("pagein-smoke", help="MemGPT fire-rate smoke test")
    p_pg.add_argument("--proxy-log", required=True)
    p_pg.add_argument("--min-convs", type=int, default=30)
    p_pg.add_argument("--out", default=None)

    args = ap.parse_args(argv)

    if args.cmd == "cost":
        rep = probe_cost_report(n_rows=args.rows, n_probes=args.probes,
                                n_conditions=args.conditions,
                                sec_per_generation=args.sec_per_generation)
        print(json.dumps(rep, indent=2))
        return 0

    if args.cmd == "plan-probes":
        if not Path(args.sidecar).exists():
            raise SystemExit(
                f"FATAL: sidecar {args.sidecar} does not exist. The probe plan "
                "reads the per-doc sidecar (unit U2, agent/t34_dump_sidecar.py) "
                "for the visible block lengths and the dropped-block truth; "
                "there is no substitute for it and nothing here fabricates one.")
        manifest, pairs, frame = _load_frame(args)
        subset = [r["qid"] for r in frame if r["label_cw"] in (0, 1)]
        c2kv_by_qid = {c["qid"]: c for _, c in pairs}
        sidecar_rows = load_jsonl(args.sidecar)
        covered = {r["qid"] for r in sidecar_rows} & set(subset)
        if not covered:
            raise SystemExit(
                f"FATAL: none of the {len(subset)} trigger qids appear in "
                f"{args.sidecar}; that sidecar was dumped for another rowset.")
        if len(covered) < len(subset):
            print(f"WARNING: {len(subset) - len(covered)} of {len(subset)} "
                  "trigger qids have no sidecar row and are NOT planned; "
                  "every probe denominator below is out of that many.")
        rows = probe_plan_rows(sidecar_rows, c2kv_by_qid,
                               condition=args.condition, seed=args.seed, qids=subset)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        # per-probe denominators: P2 loses blocks and P3 loses whole qids on
        # rows with fewer than two visible history blocks, so the counts are
        # NOT all equal to the qid count and every downstream number has to
        # carry its own.
        by_probe = {p: [r for r in rows if r["probe"] == p] for p in PROBE_IDS}
        p2_hist: Dict[str, int] = {}
        for r in by_probe["P2"]:
            k = str(len(r["blocks"] or []))
            p2_hist[k] = p2_hist.get(k, 0) + 1
        print(json.dumps({
            "plan_rows": len(rows),
            "qids": len(set(r["qid"] for r in rows)),
            "rows_by_probe": {p: len(v) for p, v in by_probe.items()},
            "qids_by_probe": {p: len({r["qid"] for r in v})
                              for p, v in by_probe.items()},
            "p2_blocks_per_request_histogram": dict(sorted(p2_hist.items())),
            "condition": args.condition, "out": str(out)}, indent=2))
        return 0

    if args.cmd == "score-probes":
        rows = load_jsonl(args.probes)
        res = score_probe_file(rows)
        res["gate"] = ledger_gate(res)            # spend the +ledger half?
        res["p4_gate"] = p4_selfcheck_gate(res)   # build the selfcheck arm?
        if res["p4_gate"]["undetermined"]:
            print("P4 GATE UNDETERMINED: the detection truth does not vary in "
                  "this file; run the --arm full probe pass and score it with "
                  "score-p4-pooled before reading the selfcheck precondition.")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        C.freeze_json(Path(args.out), res)
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "score-p4-pooled":
        rows: List[Dict[str, Any]] = []
        for path in args.probes:
            rows.extend(load_jsonl(path))
        pooled = pooled_p4_detection(rows)
        parse_rate = None
        p4_rows = [r for r in rows if r.get("probe") == "P4"]
        if p4_rows:
            parse_rate = sum(1 for r in p4_rows if _row_parsed(r)) / len(p4_rows)
        res: Dict[str, Any] = {
            "sources": list(args.probes),
            "parse_rate": parse_rate,
            "P4_detection_pooled": pooled,
        }
        if pooled["available"]:
            # the pooled table is the one the selfcheck gate can be read on
            res["P4_detection"] = pooled["detection"]
            res["p4_gate"] = p4_selfcheck_gate(res)
        else:
            res["p4_gate"] = None
            print("POOLED P4 UNAVAILABLE: " + str(pooled["reason"]))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        C.freeze_json(Path(args.out), res)
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "score-selfcheck":
        gate = require_p4_gate(args.p4_result, force=args.force)
        manifest, pairs, frame = _load_frame(args)
        rows = load_jsonl(args.probes)
        feats = selfcheck_feature_rows(rows)
        C.write_features_jsonl(Path(args.features_out), feats,
                               context="t34_selfreport selfcheck features")
        label_by_qid = {r["qid"]: r["label_cw"] for r in frame}
        pf_by_qid = {r["qid"]: bool(r["parse_fail_fire"]) for r in frame}
        sub_rows = [f for f in feats if label_by_qid.get(f["qid"]) in (0, 1)]
        y = [label_by_qid[f["qid"]] for f in sub_rows]
        sess = [f["session_id"] for f in sub_rows]
        scores = [f["selfcheck_risk"] for f in sub_rows]
        base = [1.0 if pf_by_qid.get(f["qid"]) else 0.0 for f in sub_rows]
        table = score_feature_table(scores, y, sess, name="selfcheck_risk",
                                    orientation=orientation_of("selfcheck_risk"),
                                    baselines={"parse_fail": base})
        fires = {f["qid"]: (None if f["selfcheck_says_inconsistent"] is None
                            else bool(f["selfcheck_says_inconsistent"]))
                 for f in sub_rows}
        res = {
            "gate": gate,
            "n_feature_rows": len(feats),
            "table": table,
            "cofire_with_parse_failure": cofire_with_parse_failure(
                fires, {q: pf_by_qid.get(q, False) for q in fires}),
            "selfcheck_generate_sec_total": float(sum(
                float(f["selfcheck_generate_sec"] or 0.0) for f in feats)),
        }
        C.freeze_json(Path(args.out), res)
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "pagein-smoke":
        rows = C.load_proxy_log(Path(args.proxy_log))
        res = fire_rate_smoke(rows, min_convs=args.min_convs)
        if res["stop"]:
            print("STOP: " + str(res["stop_reason"]))
        if args.out:
            C.freeze_json(Path(args.out), res)
        print(json.dumps(res, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
