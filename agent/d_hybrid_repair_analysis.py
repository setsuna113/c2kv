"""hybrid-x-D combo analysis: additivity, rescue rates, block-position scan.

Inputs are battery/D jsonl rows (agent-llm-traces face, ckpt-1088).  Correct
= tool_name_match on non-skipped rows (the D-line s_metric); trigger sets are
re-derived from the battery rows with the same C->W rule the extractor froze
(full correct AND compressed wrong).

Answers the three combo questions on the mechanism face:
  1. additivity  — hybrid-fixed set vs corr@first-fixed sets, overlap and
                   combined coverage of the c2kv error mass;
  2. rescue rate — corr@first - sham on the hybrid base (net), with the
                   pure-c2kv base measured on the same checkpoint as the
                   reference, plus the full-arm ceiling;
  3. position    — per-qid rescuing-block sets from the offset:j scan:
                   histogram, first-block prior strength, any-block ceiling.

Usage:
  python agent/d_hybrid_repair_analysis.py \
    --full-battery  results/hxd/battery.jsonl \
    --c2kv-battery  results/hxd/battery.jsonl \
    --hybrid-battery results/hxd/battery.jsonl \
    --c2kv-sham results/hxd/d_c2kv_sham.jsonl --c2kv-corr results/hxd/d_c2kv_corr.jsonl \
    --hyb-sham  results/hxd/d_hyb_sham.jsonl  --hyb-corr  results/hxd/d_hyb_corr.jsonl \
    [--hyb-full results/hxd/d_hyb_full.jsonl] [--scan results/hxd/d_hyb_scan.jsonl] \
    [--json-out results/hxd/combo_analysis.json]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def load_rows(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def mode_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{mode: {qid: row}} for non-skipped rows (later rows win)."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("skipped"):
            continue
        mode = str(row.get("mode"))
        out[mode][str(row.get("qid"))] = row
    return out


def correct(row: Optional[Dict[str, Any]]) -> Optional[bool]:
    if row is None:
        return None
    value = row.get("tool_name_match")
    return None if value is None else bool(value)


def n_docs(row: Dict[str, Any]) -> Optional[int]:
    chunks = row.get("doc_chunks")
    return int(chunks) if chunks is not None else None


def mcnemar_p(b: int, c: int) -> float:
    """Exact binomial McNemar (two-sided) on the discordant pairs."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def cluster_bootstrap_delta(
    pairs: List[tuple], sessions: List[str], reps: int = 20000, seed: int = 0
) -> List[float]:
    """Session-clustered bootstrap of the paired success-rate delta."""
    by_session: Dict[str, List[int]] = defaultdict(list)
    for (a, b_val), sess in zip(pairs, sessions):
        by_session[sess].append((int(a) - int(b_val)))
    keys = sorted(by_session)
    rng = random.Random(seed)
    deltas = []
    for _ in range(reps):
        sample = [by_session[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [d for group in sample for d in group]
        deltas.append(sum(flat) / len(flat))
    deltas.sort()
    return deltas


def rescue_table(name: str, base_rows: Dict[str, Dict[str, Any]],
                 arm_rows: Dict[str, Dict[str, Any]], triggers: List[str],
                 session_of) -> Dict[str, Any]:
    """L2-style rescue stats of `arm` over `base` on the trigger qids."""
    pairs = []
    sessions = []
    n_resc = 0
    rescued: Set[str] = set()
    for qid in triggers:
        base_ok = correct(base_rows.get(qid))
        arm_ok = correct(arm_rows.get(qid))
        if base_ok is None or arm_ok is None:
            continue
        # trigger qids are base-wrong by construction; arm success = rescue
        pairs.append((1 if arm_ok else 0, 0))
        sessions.append(session_of(qid))
        if arm_ok:
            n_resc += 1
            rescued.add(qid)
    n = len(pairs)
    rate = n_resc / n if n else None
    return {
        "arm": name, "n": n, "rescued": n_resc,
        "rescue_rate": round(rate, 4) if rate is not None else None,
        "rescued_qids": sorted(rescued),
        "n_pairs": n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-battery", required=True)
    parser.add_argument("--c2kv-battery", required=True)
    parser.add_argument("--hybrid-battery", required=True)
    parser.add_argument("--hybrid-top-k", type=int, default=3)
    parser.add_argument("--c2kv-sham", required=True)
    parser.add_argument("--c2kv-corr", required=True)
    parser.add_argument("--hyb-sham", required=True)
    parser.add_argument("--hyb-corr", required=True)
    parser.add_argument("--hyb-full", default=None)
    parser.add_argument("--scan", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    battery = load_rows(args.full_battery)
    idx = mode_index(battery)
    full_rows, c2kv_rows, hyb_rows = idx.get("full", {}), idx.get("c2kv", {}), idx.get("hybrid", {})

    def session_of(qid: str) -> str:
        return qid.rpartition(":")[0]

    # ---- trigger sets (re-derived; identical to the frozen C->W rule)
    paired_all = sorted(set(full_rows) & set(c2kv_rows) & set(hyb_rows))
    cw_c2kv = [q for q in paired_all if correct(full_rows[q]) and not correct(c2kv_rows[q])]
    cw_hyb = [q for q in paired_all if correct(full_rows[q]) and not correct(hyb_rows[q])]
    hyb_regression = [q for q in paired_all if correct(c2kv_rows[q]) and not correct(hyb_rows[q])]

    print(f"battery paired n={len(paired_all)}  full-acc="
          f"{sum(bool(correct(r)) for r in full_rows.values())/max(len(full_rows),1):.4f}")
    print(f"C->W(c2kv) n={len(cw_c2kv)}   C->W(hybrid k={args.hybrid_top_k}) n={len(cw_hyb)}   "
          f"hybrid regression (c2kv right, hybrid wrong) n={len(hyb_regression)}")

    report: Dict[str, Any] = {
        "n_paired": len(paired_all),
        "n_cw_c2kv": len(cw_c2kv), "n_cw_hybrid": len(cw_hyb),
        "n_hybrid_regression": len(hyb_regression),
        "hybrid_regression_qids": sorted(hyb_regression),
    }

    # ---- rescue arms
    sham_c = mode_index(load_rows(args.c2kv_sham)).get("d_sham_neutral", {})
    corr_c = mode_index(load_rows(args.c2kv_corr)).get("d_corr", {})
    sham_h = mode_index(load_rows(args.hyb_sham)).get("d_sham_neutral", {})
    corr_h = mode_index(load_rows(args.hyb_corr)).get("d_corr", {})

    t_sham_c = rescue_table("c2kv-base sham@first", c2kv_rows, sham_c, cw_c2kv, session_of)
    t_corr_c = rescue_table("c2kv-base corr@first", c2kv_rows, corr_c, cw_c2kv, session_of)
    t_sham_h = rescue_table("hybrid-base sham@first", hyb_rows, sham_h, cw_hyb, session_of)
    t_corr_h = rescue_table("hybrid-base corr@first", hyb_rows, corr_h, cw_hyb, session_of)

    def net(corr_t, sham_t):
        if corr_t["rescue_rate"] is None or sham_t["rescue_rate"] is None:
            return None
        return round(corr_t["rescue_rate"] - sham_t["rescue_rate"], 4)

    for t in (t_sham_c, t_corr_c, t_sham_h, t_corr_h):
        print(f"{t['arm']}: n={t['n']} rescued={t['rescued']} rate={t['rescue_rate']}")
    print(f"net repair (c2kv base)  corr-sham = {net(t_corr_c, t_sham_c)}")
    print(f"net repair (hybrid base) corr-sham = {net(t_corr_h, t_sham_h)}")
    report["rescue"] = {
        "c2kv_sham": t_sham_c, "c2kv_corr": t_corr_c,
        "hyb_sham": t_sham_h, "hyb_corr": t_corr_h,
        "net_c2kv": net(t_corr_c, t_sham_c), "net_hybrid": net(t_corr_h, t_sham_h),
    }

    if args.hyb_full:
        full_h = mode_index(load_rows(args.hyb_full)).get("full", {})
        t_full_h = rescue_table("full ceiling on hybrid triggers", hyb_rows, full_h, cw_hyb, session_of)
        print(f"full ceiling: rate={t_full_h['rescue_rate']} (n={t_full_h['n']})")
        report["rescue"]["hyb_full_ceiling"] = t_full_h

    # ---- additivity
    hybrid_fixed = {q for q in cw_c2kv if correct(hyb_rows.get(q))}  # hybrid alone fixed it
    corr_c_fixed = set(t_corr_c["rescued_qids"]) & set(cw_c2kv)     # corr on c2kv base fixed it
    corr_h_fixed = set(t_corr_h["rescued_qids"]) & set(cw_hyb)      # corr on hybrid base fixed it
    union_cov = (hybrid_fixed | corr_h_fixed)
    overlap = hybrid_fixed & corr_c_fixed
    denom = len(cw_c2kv) or 1
    print(f"\nadditivity on C->W(c2kv) n={len(cw_c2kv)}:")
    print(f"  hybrid-fixed            {len(hybrid_fixed)} ({len(hybrid_fixed)/denom:.1%})")
    print(f"  corr@first-fixed(c2kv)  {len(corr_c_fixed)} ({len(corr_c_fixed)/denom:.1%})")
    if hybrid_fixed or corr_c_fixed:
        jac = len(overlap) / max(len(hybrid_fixed | corr_c_fixed), 1)
        print(f"  overlap                 {len(overlap)}  jaccard={jac:.3f} "
              f"({'REDUNDANT' if jac > 0.5 else 'additive-leaning'})")
    combo = {q for q in cw_c2kv if correct(hyb_rows.get(q)) or q in corr_h_fixed}
    print(f"  combo coverage (hybrid ∪ hybrid+corr) {len(combo)}/{len(cw_c2kv)} "
          f"({len(combo)/denom:.1%})")
    report["additivity"] = {
        "hybrid_fixed": sorted(hybrid_fixed),
        "corr_c2kv_fixed": sorted(corr_c_fixed),
        "corr_hybrid_fixed": sorted(corr_h_fixed),
        "overlap": sorted(overlap),
        "combo_coverage": round(len(combo) / denom, 4),
    }

    # ---- block-position scan (hybrid base)
    if args.scan:
        scan_rows = [r for r in load_rows(args.scan) if not r.get("skipped")]
        by_qid: Dict[str, Dict[int, bool]] = defaultdict(dict)
        for row in scan_rows:
            j = row.get("d_corr_doc_index")
            if j is not None:
                by_qid[str(row["qid"])][int(j)] = bool(correct(row))
        hist: Counter = Counter()
        n_any = 0
        n_zero = 0
        n_q = 0
        norm_positions: List[float] = []
        for qid in cw_hyb:
            blocks = by_qid.get(qid)
            if not blocks:
                continue
            winners = sorted(j for j, ok in blocks.items() if ok)
            n_q += 1
            if winners:
                n_any += 1
                if 0 in winners:
                    n_zero += 1
                t_total = n_docs(hyb_rows[qid]) or max(blocks) + 1
                compressed = max(t_total - args.hybrid_top_k, 1)
                hist.update(winners)
                norm_positions.extend(w / compressed for w in winners)
        print(f"\nscan on C->W(hybrid) qids with scan rows: {n_q}")
        if n_q:
            print(f"  any-block rescues (locator ceiling): {n_any}/{n_q} = {n_any/n_q:.1%}")
            print(f"  first block among rescuing: P(j=0 rescues | any) = {n_zero}/{n_any} = "
                  f"{(n_zero/n_any if n_any else 0):.1%}")
            print(f"  rescuing-block histogram: {dict(sorted(hist.items()))}")
            if norm_positions:
                buckets = Counter(int(p * 4) for p in norm_positions)
                print(f"  normalized position quartiles: "
                      f"{ {k: buckets[k] for k in sorted(buckets)} }")
        report["scan"] = {
            "n_qids": n_q, "n_any_rescue": n_any, "n_first_rescue": n_zero,
            "histogram": dict(sorted(hist.items())),
        }

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
