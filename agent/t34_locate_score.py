# -*- coding: utf-8 -*-
"""t34 unit U3 -- the locator table for digest section 4.7.

Takes the chooser file written by ``agent/t34_localize.py choosers`` and turns
it into the S@k table the section-4.0 winner rule asks for:

  * S@k against the frozen 25.0 % wrong-block floor, one-sided exact binomial,
    Clopper-Pearson 95 % CI (``t34_common.locate_table``);
  * the witness oracle (71/93 = 76.3 %) and best-k ceiling (81/93 = 87.1 %)
    printed as FIXED reference rows -- position only, never a point estimate;
  * the position priors k_first / k_median / k_last as MANDATORY rows, added
    from the witness table itself when the chooser file omits them
    (2307.03172; only its qualitative first-or-last claim is used);
  * the inverted-score control per chooser (2608.22577 Table `chooser`) --
    argmax vs argmin on the SAME score vectors, with an exact McNemar;
  * an exact McNemar of each chooser against each position prior AND against the
    surface-form ceiling (``k_star_proposal``: witness-IDF scored with the arm's
    own action) on the SHARED qid subset, reported as a DELTA with its n;
  * abstain counts (``k*=None`` rows) reported separately, never dropped;
  * ``khat_argmax_disagreements``: rows whose ``khat`` is not the argmax of their
    own ``score_vector`` under the frozen selector semantics -- the inverted-score
    control is only interpretable where that count is 0.

Hits are computed against TWO references and the two columns are NEVER merged:

  ``witness``  -- the frozen gold-scored k* (configs/bdf_pilot/d_witness_r2.json,
                  field ``k_witness``); this is the 76.3 % reference locator.
  ``flip``     -- the D-line per-(qid, k) repair sweep (results/t34/flip_table.jsonl,
                  ``{qid, k, correct}``): a hit iff repairing at k_hat actually
                  flipped the row.  Its denominator is the qids present in the
                  flip table AND swept at the chosen k, which is smaller; it is
                  printed with its own n plus ``n_khat_outside_flip_sweep`` --
                  an unswept k is unmeasured, never scored as a miss.

RUNBOOK (runs HERE; zero GPU)
-----------------------------
  PYTHONIOENCODING=utf-8 python agent/t34_locate_score.py \
      --choosers results/t34/choosers_localize.jsonl \
      --witness configs/bdf_pilot/d_witness_r2.json \
      --flip results/t34/flip_table.jsonl \
      --out results/t34/locate_table_localize.json --print

  # restrict to the uncensored slice (the 128-token cap under-counts edge_own):
  PYTHONIOENCODING=utf-8 python agent/t34_locate_score.py ... --stratum uncensored

``--frame witness`` (the default) keeps only the qids the frozen witness table
covers -- the 93 C->W rows that ARE the locator estimand.  ``--frame all``
scores every qid in the chooser file and is for diagnostics only: a C->C qid has
no reference block, so it would enter the denominator as a guaranteed miss.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from t34_common import (  # noqa: E402
    LOCATE_BESTK_HITS,
    LOCATE_FLOOR_WRONG_BLOCK,
    LOCATE_WITNESS_HITS,
    chooser_argmax,
    inverted_score_control,
    load_flip_table,
    locate_table,
    mcnemar_exact,
)
from t33_labels import load_jsonl  # noqa: E402
from t34_localize import (  # noqa: E402
    CH_FIRST,
    CH_LAST,
    CH_MEDIAN,
    CH_PROPOSAL,
    PRIOR_CHOOSERS,
    position_prior_khat,
    positional_scores_at,
    require_input,
)


DEVIATIONS: List[Dict[str, str]] = [
    {
        "method": "locator scoring table",
        "paper": "2608.22577 Table `chooser`",
        "what": "Their chooser ladder is scored by log-likelihood margin over Recent-B; this table "
                "scores S@k against the frozen 25.0 % wrong-block floor with an exact one-sided "
                "binomial and a Clopper-Pearson CI, and carries their inverted-score control "
                "verbatim.",
        "why": "We have no budgeted-allocation object, so log-likelihood margin over a budget "
               "baseline is not defined here; S@k against the frozen floor is the D-line's own "
               "R-gate and is what the section-4.0 winner rule asks for.",
    },
    {
        "method": "locator scoring table",
        "paper": "2605.25310 sec 4.2.1",
        "what": "Abstentions (k*=None, 3/93 by construction) stay in the S@k denominator as misses "
                "AND are reported separately; the witness-k* column and the flip-table column are "
                "kept as two columns with two denominators and are never merged.  In the flip "
                "column a k_hat the D-line sweep never tried for that qid is UNMEASURED: it leaves "
                "that column's denominator and is counted as `n_khat_outside_flip_sweep`, never "
                "scored as a failed flip.",
        "why": "Their own negative result (thresholded coherence 0.307 below the 0.429 independence "
               "null) is that a good ranking does not give coherent decisions; hiding abstentions or "
               "pooling two references would reproduce exactly that error.",
    },
    {
        "method": "position-prior rows",
        "paper": "2307.03172",
        "what": "k_first / k_median / k_last are synthesised from the witness table when the chooser "
                "file omits them, so they appear in EVERY locator table.  No number from that paper "
                "is quoted -- it has no transfer card.",
        "why": "The digest makes the priors mandatory rows rather than a footnote; the paper itself "
               "is a second-hand CRITIC entry with unverified venue/year.",
    },
]


def witness_truth(witness: Dict[str, Any]) -> Dict[str, Optional[int]]:
    """qid -> the frozen gold-scored reference block ``k_witness`` (None = no witness)."""
    return {q: (int(e["k_witness"]) if e.get("k_witness") is not None else None)
            for q, e in (witness.get("entries") or {}).items()}


def witness_ndocs(witness: Dict[str, Any]) -> Dict[str, int]:
    return {q: int(e["n_docs"]) for q, e in (witness.get("entries") or {}).items()}


def witness_kmedian(witness: Dict[str, Any]) -> Dict[str, int]:
    return {q: int(e["k_median"]) for q, e in (witness.get("entries") or {}).items()
            if e.get("k_median") is not None}


def prior_rows_from_witness(witness: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Synthesise the three mandatory position-prior chooser rows (2307.03172).

    They must appear in EVERY locator table, so the scorer builds them itself
    from ``n_docs`` / ``k_median`` rather than trusting the chooser file to
    carry them.
    """
    nd = witness_ndocs(witness)
    km = witness_kmedian(witness)
    rows: List[Dict[str, Any]] = []
    for qid, n in sorted(nd.items()):
        for name, prior in ((CH_FIRST, "first"), (CH_MEDIAN, "median"), (CH_LAST, "last")):
            khat = position_prior_khat(n, prior, km.get(qid))
            rows.append({
                "qid": qid, "chooser": name,
                "khat": khat,
                # the tent peaked at the k this row ACTUALLY chose, so a
                # synthesised prior gets the same inverted-score control as a
                # chooser that came from the chooser file (an empty vector would
                # silently drop the control for the mandatory baseline rows).
                "score_vector": positional_scores_at(n, khat),
                "n_docs": n, "note": {"synthesised_prior": True},
            })
    return rows


def _hits_vs_witness(rows: Sequence[Dict[str, Any]],
                     truth: Dict[str, Optional[int]]) -> Dict[str, Optional[bool]]:
    out: Dict[str, Optional[bool]] = {}
    for r in rows:
        qid = r["qid"]
        ref = truth.get(qid)
        if ref is None or r.get("khat") is None:
            out[qid] = None          # abstention / no reference: a miss, counted apart
        else:
            out[qid] = bool(int(r["khat"]) == int(ref))
    return out


def _hits_vs_flip(rows: Sequence[Dict[str, Any]],
                  flip: Dict[str, Dict[int, bool]]) -> Tuple[Dict[str, Optional[bool]], int]:
    """(qid -> flip hit, number of rows whose k_hat the sweep never tried).

    A qid outside the flip table is outside this column's denominator.  So is a
    row whose chosen block was never swept FOR that qid: the sweep says nothing
    about it, and scoring it as ``False`` would turn a missing measurement into
    a miss and deflate every flip S@k.  Both exclusions are counted, never
    silent.
    """
    out: Dict[str, Optional[bool]] = {}
    n_unswept = 0
    for r in rows:
        qid = r["qid"]
        table = flip.get(qid)
        if not table:
            continue                  # outside the flip table's denominator
        if r.get("khat") is None:
            out[qid] = None           # abstention: stays in, counted apart
            continue
        k = int(r["khat"])
        if k not in table:
            n_unswept += 1            # unmeasured, NOT a miss
            continue
        out[qid] = bool(table[k])
    return out, n_unswept


def _mcnemar_vs(a: Dict[str, Optional[bool]], b: Dict[str, Optional[bool]]) -> Dict[str, Any]:
    """Exact McNemar on the SHARED qid subset; n, both marginals and the DELTA.

    The digest's 4.7 line for the edge oracle is explicit that what gets
    reported against the position baselines and the surface-form ceiling is the
    DIFFERENCE, not a bare per-arm number, so the delta is computed here rather
    than left to whoever writes the table.
    """
    shared = sorted(set(a) & set(b))
    bb = sum(1 for q in shared if a[q] is True and b[q] is not True)
    cc = sum(1 for q in shared if b[q] is True and a[q] is not True)
    a_hits = sum(1 for q in shared if a[q] is True)
    b_hits = sum(1 for q in shared if b[q] is True)
    n = len(shared)
    return {"n_shared": n, "a_only": bb, "b_only": cc,
            "a_hits": a_hits, "b_hits": b_hits,
            "a_s_at_k": (a_hits / n) if n else None,
            "b_s_at_k": (b_hits / n) if n else None,
            "delta_s_at_k": ((a_hits - b_hits) / n) if n else None,
            "mcnemar_p": mcnemar_exact(bb, cc)}


def build_table(chooser_rows: Sequence[Dict[str, Any]],
                witness: Dict[str, Any],
                flip: Optional[Dict[str, Dict[int, bool]]] = None,
                *,
                stratum: str = "all",
                frame: str = "witness",
                floor: float = LOCATE_FLOOR_WRONG_BLOCK) -> Dict[str, Any]:
    """The full locator table.

    ``frame``: ``witness`` (default) restricts the denominator to the qids the
    frozen witness table actually covers -- the 93 C->W rows, of which 3 have
    ``k_witness = None`` and stay in as declared abstentions.  A qid with no
    witness entry at all is NOT part of the locator estimand and would silently
    inflate the denominator, so it is dropped and counted.

    ``stratum``: ``all`` | ``censored`` | ``uncensored`` -- the 128-token cap
    under-counts the ``edge_own`` oracle (2605.25310 pitfall: 41/93 emissions
    lack a closing tag), so both slices are reportable and the chosen one is
    stamped into the output.
    """
    truth = witness_truth(witness)
    rows = list(chooser_rows)
    n_in = len(rows)
    if frame == "witness":
        rows = [r for r in rows if r["qid"] in truth]
    n_outside = n_in - len(rows)
    if stratum != "all":
        want = (stratum == "censored")
        rows = [r for r in rows if bool(r.get("censored_at_cap")) == want]

    by_ch: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_ch.setdefault(str(r["chooser"]), []).append(r)
    # mandatory prior rows: whole groups, added only for priors the chooser
    # file did not already supply (never a half-populated prior row set).  They
    # are restricted to the qids that survived the frame/stratum filter, so a
    # synthesised prior can never be scored on a LARGER denominator than the
    # candidate it is the control for (e.g. --stratum uncensored).
    scored_qids = {r["qid"] for r in rows}
    synth: Dict[str, List[Dict[str, Any]]] = {}
    if scored_qids:
        for pr in prior_rows_from_witness(witness):
            if pr["qid"] in scored_qids:
                synth.setdefault(pr["chooser"], []).append(pr)
    for name, group in synth.items():
        if name not in by_ch:
            by_ch[name] = group

    prior_hits = {name: _hits_vs_witness(by_ch.get(name, []), truth) for name in PRIOR_CHOOSERS}
    # the surface-form ceiling (witness-IDF scored with the arm's OWN action):
    # the digest's 4.7 edge-oracle line requires it in the SAME table as the
    # position baselines, reported as a difference.  Absent from the chooser
    # file -> the comparison is reported as absent, never silently skipped.
    surface_hits = (_hits_vs_witness(by_ch[CH_PROPOSAL], truth)
                    if CH_PROPOSAL in by_ch else None)

    out_rows: List[Dict[str, Any]] = []
    for name in sorted(by_ch):
        crows = by_ch[name]
        hw = _hits_vs_witness(crows, truth)
        tab_w = locate_table(hw, floor=floor, label=name)
        entry: Dict[str, Any] = {
            "chooser": name,
            "n_rows": len(crows),
            "hits_vs_witness": tab_w,
            "abstained": tab_w["abstained"],
        }
        if flip:
            hf, n_unswept = _hits_vs_flip(crows, flip)
            tab_f = locate_table(hf, floor=floor, label=name) if hf else None
            if tab_f is not None:
                # rows the sweep never tried at this k are OUTSIDE this column's
                # denominator and are named, not folded into the misses.
                tab_f["n_khat_outside_flip_sweep"] = n_unswept
            entry["hits_vs_flip"] = tab_f
            entry["n_khat_outside_flip_sweep"] = n_unswept
        # The specificity control (2608.22577 Table `chooser`) inverts the
        # chooser's OWN score vector; t34_common.inverted_score_control now uses
        # the frozen select_k_star semantics on both arms (chooser_argmax /
        # chooser_argmin), so the vector is passed through verbatim -- blanking
        # it on abstained rows would also silence the ARGMIN arm and hide a
        # control that beat the chooser.  What the control cannot do is notice a
        # chooser whose k_hat was not produced by an argmax of this vector, so
        # that disagreement is counted and reported instead of papered over.
        vecs = {r["qid"]: list(r.get("score_vector") or []) for r in crows}
        disagree = [r["qid"] for r in crows
                    if chooser_argmax([float(x) for x in (r.get("score_vector") or [])])
                    != (int(r["khat"]) if r.get("khat") is not None else None)]
        entry["khat_argmax_disagreements"] = len(disagree)
        entry["khat_argmax_disagreement_sample"] = sorted(disagree)[:5]
        if any(len(v) > 0 for v in vecs.values()):
            entry["inverted_control"] = inverted_score_control(vecs, truth)
        entry["vs_priors"] = {p: _mcnemar_vs(hw, prior_hits[p])
                              for p in PRIOR_CHOOSERS if prior_hits.get(p)}
        entry["vs_surface_form_ceiling"] = (
            _mcnemar_vs(hw, surface_hits) if (surface_hits and name != CH_PROPOSAL) else None)
        out_rows.append(entry)

    return {
        "stratum": stratum,
        "frame": frame,
        "n_rows_outside_witness_frame": n_outside,
        "floor_wrong_block": floor,
        "reference_rows": {
            "witness_oracle_gold_scored": {"hits": LOCATE_WITNESS_HITS[0],
                                           "n": LOCATE_WITNESS_HITS[1],
                                           "s_at_k": LOCATE_WITNESS_HITS[0] / LOCATE_WITNESS_HITS[1]},
            "best_k_ceiling": {"hits": LOCATE_BESTK_HITS[0], "n": LOCATE_BESTK_HITS[1],
                               "s_at_k": LOCATE_BESTK_HITS[0] / LOCATE_BESTK_HITS[1]},
        },
        "n_chooser_rows": len(rows),
        "surface_form_ceiling_present": bool(surface_hits),
        "choosers": out_rows,
    }


def render(table: Dict[str, Any]) -> str:
    """ASCII-only rendering (never write non-ASCII to stdout)."""
    lines: List[str] = []
    lines.append("frame=%s(dropped %d)  stratum=%s  floor=%.3f  witness_oracle=%.3f  bestk_ceiling=%.3f"
                 % (table.get("frame", "?"), table.get("n_rows_outside_witness_frame", 0),
                    table["stratum"], table["floor_wrong_block"],
                    table["reference_rows"]["witness_oracle_gold_scored"]["s_at_k"],
                    table["reference_rows"]["best_k_ceiling"]["s_at_k"]))
    head = ("%-34s %5s %5s %6s %8s %8s %8s %8s" %
            ("chooser", "n", "hits", "S@k", "p_floor", "abstain", "inv_S@k", "flip_S@k"))
    lines.append(head)
    lines.append("-" * len(head))
    for e in table["choosers"]:
        w = e["hits_vs_witness"]
        inv = e.get("inverted_control")
        flip = e.get("hits_vs_flip")
        lines.append("%-34s %5d %5d %6.3f %8.2e %8d %8s %8s" % (
            e["chooser"][:34], w["n"], w["hits"], (w["s_at_k"] or 0.0), w["p_vs_floor"],
            w["abstained"],
            ("%.3f" % inv["inverted"]["s_at_k"]) if inv and inv["inverted"]["s_at_k"] is not None else "-",
            ("%.3f" % flip["s_at_k"]) if flip and flip["s_at_k"] is not None else "-",
        ))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--choosers", required=True, help="chooser jsonl from t34_localize choosers")
    p.add_argument("--witness", required=True, help="configs/bdf_pilot/d_witness_r2.json")
    p.add_argument("--flip", default=None, help="results/t34/flip_table.jsonl (optional)")
    p.add_argument("--stratum", default="all", choices=["all", "censored", "uncensored"])
    p.add_argument("--frame", default="witness", choices=["witness", "all"],
                   help="witness = only qids the frozen witness table covers (the locator estimand)")
    p.add_argument("--out", required=True)
    p.add_argument("--print", dest="do_print", action="store_true")
    args = p.parse_args(argv)

    # both of this table's non-local inputs are produced elsewhere (the chooser
    # file by the sibling CLI, the flip table by the D-line sweep on the server):
    # name the missing one instead of a bare FileNotFoundError.
    rows = load_jsonl(str(require_input(Path(args.choosers), "choosers")))
    witness = json.loads(require_input(Path(args.witness), "witness").read_text(encoding="utf-8"))
    flip = load_flip_table(require_input(Path(args.flip), "flip")) if args.flip else None
    table = build_table(rows, witness, flip, stratum=args.stratum, frame=args.frame)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(table, ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    if args.do_print:
        print(render(table))
    else:
        print(json.dumps({"out": args.out, "n_choosers": len(table["choosers"]),
                          "n_rows": table["n_chooser_rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
