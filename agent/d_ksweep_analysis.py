"""D1 k-sweep analysis (prereg v2.2/v2.4) — per-k curves + corrected best-k.

Reads the k-sweep jsonl (one row per (qid, k), arm raw_keepG), the frozen
witness table, and the c2kv baseline battery; emits the pre-registered
readouts:

- MAIN point estimate: S at the frozen k_witness (per-qid witness-IDF k*);
  k_witness=None qids form their own stratum (no repair channel: their
  paired delta is 0 by construction and they stay in the denominator).
- legacy column: S at k_median (comparability with the v1 d_corr arms).
- best-k as an oracle upper envelope ONLY, with the two mandatory
  corrections: (a) the pure-random null E[max] = 1 − (1−p)^n_docs where p
  is the empirical single-k flip rate over NON-witness ks; (b) the
  concentration diagnostic — how many ks flip per qid and whether flips
  concentrate at the witness block (item-specific repair) or spread
  (any-extra-KV effect).  This diagnostic outranks the best-k number.
- per-k curve: correct rate by k, plus the wrong-block distribution
  (non-witness ks) that replaces the cancelled sham arms (prereg v2.3).

Usage:
  python agent/d_ksweep_analysis.py \
    --sweep <sweep.jsonl> --witness configs/bdf_pilot/d_witness_r2.json \
    --baseline results/bdf_pilot/d_r2/battery_c2kv.jsonl \
    --output <report.json> [--md <report.md>]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--witness", default="./configs/bdf_pilot/d_witness_r2.json")
    parser.add_argument("--baseline", default="./results/bdf_pilot/d_r2/battery_c2kv.jsonl")
    parser.add_argument("--manifest", default="./configs/bdf_pilot/d_cw_manifest_r2.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--md", default="")
    return parser.parse_args(argv)


def _norm(v) -> Optional[float]:
    return float(v) if v is not None else None


def main(argv=None):
    args = parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    qids: List[str] = list(manifest["cw_qids"])
    witness_doc = json.loads(Path(args.witness).read_text(encoding="utf-8"))
    witness = witness_doc.get("entries", witness_doc)

    sweep: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    skipped = 0
    for line in Path(args.sweep).open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("skipped"):
            skipped += 1
            continue
        sweep[row["qid"]][int(row.get("d_ksweep_k", row.get("d_corr_doc_index")))] = row

    baseline: Dict[str, bool] = {}
    for line in Path(args.baseline).open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("skipped") and row["qid"] in set(qids):
            baseline[row["qid"]] = bool(row.get("tool_name_match"))

    n_total = len(qids)
    per_qid: Dict[str, Dict[str, Any]] = {}
    s_witness = s_median = n_witness_rows = n_median_rows = 0
    n_none_stratum = 0
    n_missing_witness_row = 0
    flips_by_qid: Dict[str, List[int]] = {}
    nonwitness_trials = nonwitness_correct = 0
    correct_at_k: Dict[int, int] = defaultdict(int)
    trials_at_k: Dict[int, int] = defaultdict(int)

    for qid in qids:
        rows_k = sweep.get(qid, {})
        w = witness.get(qid, {})
        k_w, k_med = w.get("k_witness"), w.get("k_median", 0)
        n_docs = w.get("n_docs", len(rows_k))
        correct = {k: bool(r.get("tool_name_match")) for k, r in rows_k.items()}
        for k, c in correct.items():
            trials_at_k[k] += 1
            correct_at_k[k] += int(c)
        flips = sorted(k for k, c in correct.items() if c)
        flips_by_qid[qid] = flips
        if k_w is None:
            n_none_stratum += 1
        else:
            nonwitness = [correct[k] for k in correct if k != k_w]
            nonwitness_trials += len(nonwitness)
            nonwitness_correct += sum(nonwitness)
            if k_w in correct:
                n_witness_rows += 1
                s_witness += int(correct[k_w])
            else:
                n_missing_witness_row += 1
        if k_med in correct:
            n_median_rows += 1
            s_median += int(correct[k_med])
        per_qid[qid] = {
            "n_docs": n_docs, "k_witness": k_w, "k_median": k_med,
            "correct_by_k": {str(k): int(c) for k, c in sorted(correct.items())},
            "baseline_c2kv": baseline.get(qid),
            "witness_flips": (k_w in flips) if k_w is not None else None,
            "n_flips": len(flips),
        }

    # ---- best-k envelope + null correction (prereg v2.4) ----
    p_flip = (nonwitness_correct / nonwitness_trials) if nonwitness_trials else 0.0
    best_k_hits = 0
    best_k_expected_random = 0.0
    for qid in qids:
        w = witness.get(qid, {})
        n_docs = max(1, w.get("n_docs", 1))
        if flips_by_qid.get(qid):
            best_k_hits += 1
        best_k_expected_random += 1.0 - (1.0 - p_flip) ** n_docs

    # ---- flip concentration (the diagnostic that outranks best-k) ----
    n_any_flip = sum(1 for qid in qids if flips_by_qid.get(qid))
    n_exactly_one = sum(1 for qid in qids if len(flips_by_qid.get(qid, [])) == 1)
    n_witness_flips = sum(
        1 for qid in qids
        if witness.get(qid, {}).get("k_witness") is not None
        and witness[qid]["k_witness"] in flips_by_qid.get(qid, [])
    )
    n_median_flips = sum(
        1 for qid in qids
        if witness.get(qid, {}).get("k_median") in flips_by_qid.get(qid, [])
    )
    flip_frac_of_docs = []
    for qid in qids:
        w = witness.get(qid, {})
        n_docs = max(1, w.get("n_docs", 1))
        if flips_by_qid.get(qid):
            flip_frac_of_docs.append(len(flips_by_qid[qid]) / n_docs)

    curve = {
        str(k): {"correct": correct_at_k[k], "trials": trials_at_k[k],
                 "rate": round(correct_at_k[k] / trials_at_k[k], 4) if trials_at_k[k] else None}
        for k in sorted(trials_at_k)
    }

    report = {
        "n_trigger_set": n_total,
        "n_rows_skipped": skipped,
        "main_estimate": {
            "S_at_k_witness": s_witness,
            "denominator": n_total,
            "rate": round(s_witness / n_total, 4),
            "witness_rows_present": n_witness_rows,
            "witness_rows_missing": n_missing_witness_row,
            "none_stratum_n": n_none_stratum,
            "note": "k_witness=None qids stay in the denominator with paired delta 0 "
                    "(no repair channel; synthesized-argument qids marked by the "
                    "witness algorithm itself)",
        },
        "legacy_median_column": {
            "S_at_k_median": s_median, "rows": n_median_rows,
            "rate": round(s_median / n_total, 4),
        },
        "baseline_c2kv": {
            "rate_on_trigger_set": round(
                sum(1 for q in qids if baseline.get(q)) / n_total, 4),
            "note": "C->W trigger set: the c2kv battery is wrong on these qids by "
                    "construction; paired McNemar degenerates to a one-sided "
                    "binomial on the arm's repairs",
        },
        "best_k_envelope": {
            "observed_any_flip": best_k_hits,
            "observed_rate": round(best_k_hits / n_total, 4),
            "expected_random_sum": round(best_k_expected_random, 2),
            "expected_random_rate": round(best_k_expected_random / n_total, 4),
            "p_nonwitness_flip": round(p_flip, 4),
            "nonwitness_trials": nonwitness_trials,
            "note": "best-k is an upper envelope, never a point estimate; it must "
                    "beat the random envelope E[max]=1-(1-p)^n_docs to mean anything",
        },
        "flip_concentration": {
            "qids_with_any_flip": n_any_flip,
            "qids_with_exactly_one_flip": n_exactly_one,
            "witness_k_flips": n_witness_flips,
            "median_k_flips": n_median_flips,
            "mean_flip_fraction_of_docs": (
                round(sum(flip_frac_of_docs) / len(flip_frac_of_docs), 4)
                if flip_frac_of_docs else None
            ),
            "note": "one flip at a non-uniform position => item-specific repair; "
                    "many flips => any-extra-KV effect, not content",
        },
        "wrong_block_distribution": {
            "rate": round(p_flip, 4),
            "note": "correct rate over non-witness ks; replaces the cancelled sham "
                    "arms (prereg v2.3)",
        },
        "per_k_curve": curve,
        "per_qid": per_qid,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")

    if args.md:
        r = report
        lines = [
            "# D1 k-sweep analysis (prereg v2)",
            "",
            f"- trigger set n={r['n_trigger_set']} (skipped rows: {r['n_rows_skipped']})",
            f"- **MAIN S@k_witness = {r['main_estimate']['S_at_k_witness']}/{r['n_trigger_set']}"
            f" ({r['main_estimate']['rate']})**; None-stratum n={r['main_estimate']['none_stratum_n']}",
            f"- legacy S@k_median = {r['legacy_median_column']['S_at_k_median']}"
            f" ({r['legacy_median_column']['rate']})",
            f"- baseline c2kv on trigger set = {r['baseline_c2kv']['rate_on_trigger_set']}",
            f"- best-k envelope: observed {r['best_k_envelope']['observed_rate']} vs random "
            f"{r['best_k_envelope']['expected_random_rate']} (p={r['best_k_envelope']['p_nonwitness_flip']})",
            f"- flip concentration: any={r['flip_concentration']['qids_with_any_flip']}, "
            f"exactly-one={r['flip_concentration']['qids_with_exactly_one_flip']}, "
            f"witness-flips={r['flip_concentration']['witness_k_flips']}, "
            f"median-flips={r['flip_concentration']['median_k_flips']}",
            f"- wrong-block distribution rate = {r['wrong_block_distribution']['rate']}",
        ]
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({k: report[k] for k in (
        "main_estimate", "legacy_median_column", "best_k_envelope",
        "flip_concentration", "wrong_block_distribution")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
