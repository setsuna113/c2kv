"""hybrid-x-D combo analysis: additivity, rescue rates, block-position scan.

Inputs are battery/D jsonl rows (agent-llm-traces face, ckpt-1088).  Correct
= tool_name_match on non-skipped rows (the D-line s_metric); trigger sets are
re-derived from the battery rows with the same C->W rule the extractor froze
(full correct AND compressed wrong).  Rows whose metric is missing are
three-state: they never enter a trigger set (counted as missing_metric).

Answers the three combo questions on the mechanism face:
  1. additivity  — hybrid-fixed set vs corr@first-fixed sets, overlap tested
                   with Fisher exact against the independence expectation
                   (the old hardcoded jaccard>0.5 verdict is retired);
  2. rescue rate — corr@first - sham on each base (paired McNemar +
                   session-clustered bootstrap CI; per-arm rates carry a
                   bootstrap CI too), plus the full-arm ceiling;
  3. position    — per-qid rescuing-block sets from the offset:j scan:
                   histogram, first-block prior strength, any-block ceiling
                   reported as headroom over fixed @first (no noise-floor
                   extrapolation).

Usage:
  python agent/d_hybrid_repair_analysis.py \
    --full-battery  results/hxd/battery.jsonl \
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
from math import comb
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


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
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def cluster_bootstrap_delta(
    pairs: List[Tuple[int, int]], sessions: List[str], reps: int = 20000, seed: int = 0
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


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact on [[a, b], [c, d]] (hypergeometric tail sum)."""
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0:
        return 1.0

    def prob(x: int) -> float:
        return comb(r1, x) * comb(n - r1, c1 - x) / comb(n, c1)

    p0 = prob(a)
    lo = max(0, c1 - (n - r1))
    hi = min(r1, c1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p0 * (1 + 1e-9)))


def ci_of(deltas: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not deltas:
        return None, None
    return deltas[int(0.025 * len(deltas))], deltas[int(0.975 * len(deltas)) - 1]


def rescue_table(name: str, base_rows: Dict[str, Dict[str, Any]],
                 arm_rows: Dict[str, Dict[str, Any]], triggers: List[str],
                 session_of) -> Dict[str, Any]:
    """Rescue stats of `arm` over `base` on the trigger qids.

    Returns the per-qid 0/1 vector + sessions so the caller can bootstrap;
    base is 0 by construction on triggers (degenerate pairs), which the
    session-clustered bootstrap tolerates.
    """
    pairs: List[Tuple[int, int]] = []
    sessions: List[str] = []
    qids: List[str] = []
    n_resc = 0
    rescued: Set[str] = set()
    missing = 0
    for qid in triggers:
        base_ok = correct(base_rows.get(qid))
        arm_ok = correct(arm_rows.get(qid))
        if base_ok is None or arm_ok is None:
            missing += 1
            continue
        # trigger qids are base-wrong by construction; arm success = rescue
        pairs.append((1 if arm_ok else 0, 0))
        sessions.append(session_of(qid))
        qids.append(qid)
        if arm_ok:
            n_resc += 1
            rescued.add(qid)
    n = len(pairs)
    rate = n_resc / n if n else None
    deltas = cluster_bootstrap_delta(pairs, sessions)
    lo, hi = ci_of(deltas)
    return {
        "arm": name, "n": n, "rescued": n_resc, "missing_metric": missing,
        "rescue_rate": round(rate, 4) if rate is not None else None,
        "ci_low": round(lo, 4) if lo is not None else None,
        "ci_high": round(hi, 4) if hi is not None else None,
        # degenerate McNemar (base constant 0): tests rate>0, reported for
        # completeness only — the CI above is the meaningful uncertainty
        "p_vs_zero": mcnemar_p(n_resc, 0) if n else None,
        "rescued_qids": sorted(rescued),
        "qids": qids,
        "arm_ok": [p[0] for p in pairs],
        "sessions": sessions,
    }


def paired_arm_stats(corr_t: Dict[str, Any], sham_t: Dict[str, Any],
                     session_of) -> Dict[str, Any]:
    """corr vs sham on the shared qids: paired McNemar + clustered bootstrap."""
    sham_by_qid = {q: ok for q, ok in zip(sham_t["qids"], sham_t["arm_ok"])}
    pairs, sessions = [], []
    for q, corr_ok in zip(corr_t["qids"], corr_t["arm_ok"]):
        if q not in sham_by_qid:
            continue
        pairs.append((int(corr_ok), int(sham_by_qid[q])))
        sessions.append(session_of(q))
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    deltas = cluster_bootstrap_delta(pairs, sessions)
    lo, hi = ci_of(deltas)
    point = (sum(x for x, _ in pairs) - sum(y for _, y in pairs)) / len(pairs) if pairs else None
    return {
        "n_paired": len(pairs), "corr_only": b, "sham_only": c,
        "net_delta": round(point, 4) if point is not None else None,
        "mcnemar_p": mcnemar_p(b, c),
        "ci_low": round(lo, 4) if lo is not None else None,
        "ci_high": round(hi, 4) if hi is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-battery", required=True)
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

    # ---- trigger sets (re-derived; identical to the frozen C->W rule).
    # three-state: rows with a missing metric never enter any trigger set.
    paired_all = sorted(set(full_rows) & set(c2kv_rows) & set(hyb_rows))
    missing = {"full": 0, "c2kv": 0, "hybrid": 0}

    def ok(arm_rows, qid, key) -> Optional[bool]:
        value = correct(arm_rows.get(qid))
        if value is None:
            missing[key] += 1
        return value

    cw_c2kv = [q for q in paired_all
               if ok(full_rows, q, "full") and ok(c2kv_rows, q, "c2kv") is False]
    cw_hyb = [q for q in paired_all
              if ok(full_rows, q, "full") and ok(hyb_rows, q, "hybrid") is False]
    hyb_regression = [q for q in paired_all
                      if ok(c2kv_rows, q, "c2kv") and ok(hyb_rows, q, "hybrid") is False]
    if any(missing.values()):
        print(f"WARNING: missing tool_name_match rows excluded from triggers: {missing}")

    print(f"battery paired n={len(paired_all)}  full-acc="
          f"{sum(bool(correct(r)) for r in full_rows.values())/max(len(full_rows),1):.4f}")
    print(f"C->W(c2kv) n={len(cw_c2kv)}   C->W(hybrid k={args.hybrid_top_k}) n={len(cw_hyb)}   "
          f"hybrid regression (c2kv right, hybrid wrong) n={len(hyb_regression)}")

    report: Dict[str, Any] = {
        "n_paired": len(paired_all),
        "n_cw_c2kv": len(cw_c2kv), "n_cw_hybrid": len(cw_hyb),
        "n_hybrid_regression": len(hyb_regression),
        "hybrid_regression_qids": sorted(hyb_regression),
        "missing_metric": missing,
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
    net_c = paired_arm_stats(t_corr_c, t_sham_c, session_of)
    net_h = paired_arm_stats(t_corr_h, t_sham_h, session_of)

    for t in (t_sham_c, t_corr_c, t_sham_h, t_corr_h):
        print(f"{t['arm']}: n={t['n']} rescued={t['rescued']} rate={t['rescue_rate']} "
              f"ci=[{t['ci_low']},{t['ci_high']}] (missing={t['missing_metric']})")
    print(f"net repair (c2kv base)  corr-sham = {net_c['net_delta']} "
          f"ci=[{net_c['ci_low']},{net_c['ci_high']}] mcnemar_p={net_c['mcnemar_p']}")
    print(f"net repair (hybrid base) corr-sham = {net_h['net_delta']} "
          f"ci=[{net_h['ci_low']},{net_h['ci_high']}] mcnemar_p={net_h['mcnemar_p']}")
    # strip the heavy vectors from the persisted tables, keep the stats
    def slim(t):
        return {k: v for k, v in t.items() if k not in ("qids", "arm_ok", "sessions")}
    report["rescue"] = {
        "c2kv_sham": slim(t_sham_c), "c2kv_corr": slim(t_corr_c),
        "hyb_sham": slim(t_sham_h), "hybrid_corr": slim(t_corr_h),
        "net_c2kv": net_c, "net_hybrid": net_h,
    }

    if args.hyb_full:
        full_h = mode_index(load_rows(args.hyb_full)).get("full", {})
        t_full_h = rescue_table("full ceiling on hybrid triggers", hyb_rows, full_h, cw_hyb, session_of)
        print(f"full ceiling: rate={t_full_h['rescue_rate']} ci=[{t_full_h['ci_low']},{t_full_h['ci_high']}] "
              f"(n={t_full_h['n']})")
        report["rescue"]["hyb_full_ceiling"] = slim(t_full_h)

    # ---- additivity (Fisher exact on the overlap; jaccard kept descriptive)
    hybrid_fixed = {q for q in cw_c2kv if correct(hyb_rows.get(q))}  # hybrid alone fixed it
    corr_c_fixed = set(t_corr_c["rescued_qids"]) & set(cw_c2kv)     # corr on c2kv base fixed it
    corr_h_fixed = set(t_corr_h["rescued_qids"]) & set(cw_hyb)      # corr on hybrid base fixed it
    overlap = hybrid_fixed & corr_c_fixed
    denom = len(cw_c2kv) or 1
    H, C, O, N = len(hybrid_fixed), len(corr_c_fixed), len(overlap), len(cw_c2kv)
    expected_overlap = H * C / N if N else None
    # [[overlap, hybrid-only], [corr-only, neither]]
    fisher = fisher_exact_2x2(O, H - O, C - O, N - H - C + O) if N else None
    odds_ratio = (O * (N - H - C + O)) / ((H - O) * (C - O)) if H > O and C > O else None
    combo = {q for q in cw_c2kv if correct(hyb_rows.get(q)) or q in corr_h_fixed}
    marginal_of_corr = len(set(t_corr_h["rescued_qids"]) & set(cw_c2kv))
    print(f"\nadditivity on C->W(c2kv) n={N}:")
    print(f"  hybrid-fixed            {H} ({H/denom:.1%})")
    print(f"  corr@first-fixed(c2kv)  {C} ({C/denom:.1%})")
    print(f"  overlap                 {O} (expected under independence {expected_overlap:.2f})")
    if H or C:
        jac = O / max(len(hybrid_fixed | corr_c_fixed), 1)
        direction = "above" if O > expected_overlap else "at-or-below"
        verdict = (f"overlap {direction} chance"
                   + (f", p={fisher:.4f} significant" if fisher is not None and fisher < 0.05
                      else f", p={fisher if fisher is not None else float('nan'):.4f} not significant"))
        print(f"  jaccard={jac:.3f}  fisher_exact p={fisher}  odds_ratio={odds_ratio}  -> {verdict}")
    print(f"  combo coverage (hybrid ∪ hybrid+corr) {len(combo)}/{N} ({len(combo)/denom:.1%})")
    print(f"  corr-on-hybrid marginal over hybrid alone: {marginal_of_corr}/{N} "
          f"(= combo - hybrid_fixed)")
    report["additivity"] = {
        "n_universe": N,
        "hybrid_fixed": sorted(hybrid_fixed),
        "corr_c2kv_fixed": sorted(corr_c_fixed),
        "corr_hybrid_fixed": sorted(corr_h_fixed),
        "overlap": sorted(overlap),
        "expected_overlap": expected_overlap,
        "odds_ratio": odds_ratio,
        "fisher_p": fisher,
        "overlap_count": O,
        "combo_coverage": round(len(combo) / denom, 4),
        "corr_marginal_over_hybrid": marginal_of_corr,
    }

    # ---- block-position scan (hybrid base).  Ceiling is reported as
    # headroom over fixed @first; no independent-blocks noise-floor formula.
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
            winners = sorted(j for j, ok_block in blocks.items() if ok_block)
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
            headroom = (n_any - n_zero) / n_q
            print(f"  any-block rescues (perfect locator): {n_any}/{n_q} = {n_any/n_q:.1%}")
            print(f"  first block among rescuing: {n_zero}/{n_any} = "
                  f"{(n_zero/n_any if n_any else 0):.1%}")
            print(f"  headroom of perfect locator over fixed @first: "
                  f"({n_any}-{n_zero})/{n_q} = {headroom:.1%}")
            # consistency cross-check: scan j=0 winners should equal the
            # corr@first arm's rescued set on the same triggers
            scan_j0 = {qid for qid in cw_hyb
                       if by_qid.get(qid) and by_qid[qid].get(0)}
            corr_h_set = set(t_corr_h["rescued_qids"])
            print(f"  cross-check scan(j=0 winners) vs corr@first rescued: "
                  f"scan={len(scan_j0)} corr={len(corr_h_set)} "
                  f"sym-diff={len(scan_j0 ^ corr_h_set)}")
            print(f"  rescuing-block histogram: {dict(sorted(hist.items()))}")
            if norm_positions:
                buckets = Counter(int(p * 4) for p in norm_positions)
                print(f"  normalized position quartiles: "
                      f"{ {k: buckets[k] for k in sorted(buckets)} }")
        report["scan"] = {
            "n_qids": n_q, "n_any_rescue": n_any, "n_first_rescue": n_zero,
            "headroom_vs_first": round((n_any - n_zero) / n_q, 4) if n_q else None,
            "scan_j0_vs_corr_symdiff": len(
                {qid for qid in cw_hyb if by_qid.get(qid) and by_qid[qid].get(0)}
                ^ set(t_corr_h["rescued_qids"])),
            "histogram": dict(sorted(hist.items())),
        }

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
