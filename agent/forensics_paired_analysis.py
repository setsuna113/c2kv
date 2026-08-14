#!/usr/bin/env python
"""Task T3: paired full-vs-topk comparison over the frozen round-1 archive.

Loads two per-sample jsonl files, indexes rows by qid (first row wins on
duplicates, duplicates counted; rows flagged skipped are excluded and
counted), and reports:
  - per-arm N and the binary metric rate with a Wilson 95% CI (k/n);
  - over the shared-qid intersection: n_pairs, n11/n10/n01/n00 (x=topk,
    y=full), McNemar (exact binomial for small discordance, chi-square with
    continuity correction otherwise), the paired difference topk-minus-full
    with a Newcombe method-10 95% CI (Wald fallback included), the observed
    discordance rate psi = (n10+n01)/n_pairs and the measured MDE at 80%
    power, matching agent/analyze_s4_forced_prefix.py;
  - a session-cluster bootstrap 95% CI of the paired difference (session id
    from the session_id field, else the qid prefix before the last ':').
A non-empty qid symmetric difference (files covering different samples) is
warned about loudly with counts in both directions; an empty intersection is
fatal.

Outputs <out_prefix>.json and <out_prefix>.md.

Example:
  python agent/forensics_paired_analysis.py \
    --full_jsonl outputs/r1_frozen/history_agent_full.jsonl \
    --topk_jsonl outputs/r1_frozen/history_agent_c2kv_topk.jsonl \
    --metric_key tool_name_match --out_prefix outputs/r2/paired_analysis
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from stats_paired import (
        as_bool,
        cluster_bootstrap_ci,
        fmt_ci,
        fmt_p,
        fmt_prop,
        git_commit,
        mcnemar,
        mde,
        paired_binary_diff_ci,
        psi,
        wilson_ci,
    )
except ImportError:  # package-style import
    from agent.stats_paired import (
        as_bool,
        cluster_bootstrap_ci,
        fmt_ci,
        fmt_p,
        fmt_prop,
        git_commit,
        mcnemar,
        mde,
        paired_binary_diff_ci,
        psi,
        wilson_ci,
    )


def _load_rows(path: str, qid_key: str) -> Dict[str, Any]:
    """Index valid (non-skipped) rows by qid; first row wins on duplicate qids."""
    rows: Dict[str, Dict[str, Any]] = {}
    n_rows = 0
    n_skipped = 0
    n_missing_qid = 0
    n_dup_qids = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            if as_bool(row.get("skipped")):
                n_skipped += 1
                continue
            qid = row.get(qid_key)
            if qid is None:
                n_missing_qid += 1
                continue
            qid = str(qid)
            if qid in rows:
                n_dup_qids += 1
                continue
            rows[qid] = row
    return {
        "rows": rows,
        "n_rows": n_rows,
        "n_skipped": n_skipped,
        "n_missing_qid": n_missing_qid,
        "n_dup_qids": n_dup_qids,
    }


def _metric(row: Dict[str, Any], metric_key: str) -> Optional[bool]:
    return as_bool(row.get(metric_key))


def _session_id(qid: str, row: Dict[str, Any]) -> str:
    """Cluster id: session_id field, else the qid prefix before the last ':'."""
    session = row.get("session_id")
    if session is not None:
        return str(session)
    if ":" in qid:
        return qid.rsplit(":", 1)[0]
    return qid


def _arm_stats(loaded: Dict[str, Any], metric_key: str) -> Dict[str, Any]:
    rows = loaded["rows"]
    n = len(rows)
    k = sum(1 for row in rows.values() if _metric(row, metric_key))
    n_missing = sum(1 for row in rows.values() if _metric(row, metric_key) is None)
    lo, hi = wilson_ci(k, n)
    return {
        "n_rows": loaded["n_rows"],
        "n_valid": n,
        "n_skipped": loaded["n_skipped"],
        "n_missing_qid": loaded["n_missing_qid"],
        "n_dup_qids": loaded["n_dup_qids"],
        "metric": {"k": k, "n": n, "n_missing": n_missing, "rate": round(k / n, 4) if n else 0.0,
                   "wilson95": [round(lo, 4), round(hi, 4)]},
    }


def _render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Paired analysis (T3): topk minus full")
    lines.append("")
    lines.append(f"- git_commit: `{report['git_commit']}`")
    lines.append(f"- full_jsonl: `{report['full_jsonl']}`")
    lines.append(f"- topk_jsonl: `{report['topk_jsonl']}`")
    lines.append(f"- metric_key: `{report['metric_key']}`, qid_key: `{report['qid_key']}`")
    lines.append("")

    lines.append("## Warnings")
    lines.append("")
    if not report["warnings"]:
        lines.append("None.")
    for warning in report["warnings"]:
        lines.append(f"- **WARN** {warning}")
    lines.append("")

    lines.append("## Per-arm metric rates")
    lines.append("")
    lines.append("| arm | N valid | skipped | dup qids | metric rate [wilson95] (k/n) |")
    lines.append("|---|---:|---:|---:|---|")
    for arm in ("full", "topk"):
        stats = report["arms"][arm]
        metric = stats["metric"]
        lines.append(
            f"| {arm} | {stats['n_valid']} | {stats['n_skipped']} | {stats['n_dup_qids']} "
            f"| {fmt_prop(metric['k'], metric['n'])} |"
        )
    lines.append("")

    paired = report["paired"]
    lines.append("## Paired comparison over shared qids")
    lines.append("")
    lines.append("| quantity | value |")
    lines.append("|---|---|")
    lines.append(f"| n_pairs | {paired['n_pairs']} |")
    lines.append(f"| n11 (both correct) | {paired['n11']} |")
    lines.append(f"| n10 (topk only) | {paired['n10']} |")
    lines.append(f"| n01 (full only) | {paired['n01']} |")
    lines.append(f"| n00 (both wrong) | {paired['n00']} |")
    lines.append(f"| n_excluded_missing_metric | {paired['n_excluded_missing_metric']} |")
    lines.append(
        f"| diff (topk - full) [{paired['diff_method']}] "
        f"| {fmt_ci(paired['diff'], paired['diff_lo'], paired['diff_hi'])} |"
    )
    lines.append(
        f"| diff (topk - full) [wald fallback] "
        f"| {fmt_ci(paired['diff'], paired['wald_lo'], paired['wald_hi'])} |"
    )
    lines.append(
        f"| diff (topk - full) [session-cluster bootstrap, "
        f"{paired['n_clusters']} clusters] "
        f"| {fmt_ci(paired['diff'], paired['boot_lo'], paired['boot_hi'])} |"
    )
    lines.append(
        f"| McNemar ({paired['mcnemar_method']}) "
        f"| stat {paired['mcnemar_stat']:.4f}, p {fmt_p(paired['mcnemar_p'])} |"
    )
    lines.append(f"| psi (discordance rate) | {paired['psi']:.4f} ({paired['n10'] + paired['n01']}/{paired['n_pairs']}) |")
    lines.append(f"| measured MDE (80% power) | {paired['mde80']:.4f} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full_jsonl", required=True)
    parser.add_argument("--topk_jsonl", required=True)
    parser.add_argument("--metric_key", default="tool_name_match")
    parser.add_argument("--qid_key", default="qid")
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    full = _load_rows(args.full_jsonl, args.qid_key)
    topk = _load_rows(args.topk_jsonl, args.qid_key)
    full_qids = set(full["rows"])
    topk_qids = set(topk["rows"])
    shared = sorted(full_qids & topk_qids)
    full_only = sorted(full_qids - topk_qids)
    topk_only = sorted(topk_qids - full_qids)

    warnings: List[str] = []
    if full_only or topk_only:
        warnings.append(
            f"qid symmetric difference non-empty: "
            f"{len(full_only)} only in full, {len(topk_only)} only in topk "
            f"(examples full-only {full_only[:5]}, topk-only {topk_only[:5]})"
        )
    if not shared:
        for warning in warnings:
            print(f"[warn] {warning}", file=sys.stderr)
        raise SystemExit(
            "error: empty qid-set intersection between --full_jsonl and --topk_jsonl; "
            "paired analysis impossible (check --qid_key and file pairing)"
        )

    n11 = n10 = n01 = n00 = 0
    n_excluded = 0
    diffs: List[float] = []
    clusters: List[str] = []
    for qid in shared:
        full_row = full["rows"][qid]
        topk_row = topk["rows"][qid]
        x = _metric(topk_row, args.metric_key)  # x = topk
        y = _metric(full_row, args.metric_key)  # y = full
        if x is None or y is None:
            n_excluded += 1
            continue
        if x and y:
            n11 += 1
        elif x and not y:
            n10 += 1
        elif not x and y:
            n01 += 1
        else:
            n00 += 1
        diffs.append(float(x) - float(y))
        clusters.append(_session_id(qid, topk_row if topk_row.get("session_id") is not None else full_row))

    n_pairs = len(diffs)
    stat, p_value, method = mcnemar(n10, n01)
    diff_ci = paired_binary_diff_ci(n11, n10, n01, n00)
    psi_value = psi(n10, n01, n_pairs)
    mde_value = mde(psi_value, n_pairs)
    boot_lo = boot_hi = None
    try:
        boot_lo, boot_hi = cluster_bootstrap_ci(diffs, clusters)
    except ImportError:
        warnings.append("numpy unavailable; session-cluster bootstrap CI skipped")
    for warning in warnings:
        print(f"[warn] {warning}", file=sys.stderr)

    report: Dict[str, Any] = {
        "git_commit": git_commit(),
        "full_jsonl": args.full_jsonl,
        "topk_jsonl": args.topk_jsonl,
        "metric_key": args.metric_key,
        "qid_key": args.qid_key,
        "arms": {
            "full": _arm_stats(full, args.metric_key),
            "topk": _arm_stats(topk, args.metric_key),
        },
        "qid_overlap": {
            "n_shared": len(shared),
            "n_full_only": len(full_only),
            "n_topk_only": len(topk_only),
            "full_only_examples": full_only[:5],
            "topk_only_examples": topk_only[:5],
        },
        "paired": {
            "n_pairs": n_pairs,
            "n11": n11,
            "n10": n10,
            "n01": n01,
            "n00": n00,
            "n_excluded_missing_metric": n_excluded,
            "diff": round(diff_ci["diff"], 4),
            "diff_lo": round(diff_ci["lo"], 4),
            "diff_hi": round(diff_ci["hi"], 4),
            "diff_method": diff_ci["method"],
            "wald_lo": round(diff_ci.get("wald_lo", 0.0), 4),
            "wald_hi": round(diff_ci.get("wald_hi", 0.0), 4),
            "boot_lo": round(boot_lo, 4) if boot_lo is not None else None,
            "boot_hi": round(boot_hi, 4) if boot_hi is not None else None,
            "n_clusters": len(set(clusters)),
            "mcnemar_stat": round(stat, 4),
            "mcnemar_p": round(p_value, 6),
            "mcnemar_method": method,
            "psi": round(psi_value, 4),
            "mde80": round(mde_value, 4),
        },
        "warnings": warnings,
    }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_text = _render_md(report)
    out_prefix.with_suffix(".md").write_text(md_text + "\n", encoding="utf-8")
    print(md_text)


if __name__ == "__main__":
    main()
