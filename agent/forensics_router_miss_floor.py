#!/usr/bin/env python
"""Task T4: router-miss floor -- prior-guessing noise when the target tool is
not in the retained context.

Slices rows where router_hit is False (key aliases router_hit /
router_target_hit; string values like "False" tolerated) and reports:
  - the slice size vs the total row count (k/n, Wilson 95% CI);
  - the tool_name_match rate (aliases tool_name_match / tool_name_correct)
    over the slice, overall and stratified by num_tools buckets
    (<=b1, b1+1..b2, ..., >bk, plus an "unknown" bucket for missing
    num_tools), each with k/n and a Wilson 95% CI.
This is the noise floor of guessing the target tool from the prior when the
router failed to retain it.

Outputs <out_prefix>.json and <out_prefix>.md.

Example:
  python agent/forensics_router_miss_floor.py \
    --jsonl outputs/r1_frozen/tooldef_agent_router_*.jsonl \
    --buckets 8,16,32 --out_prefix outputs/r2/router_miss_floor
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from stats_paired import as_bool, first_present, fmt_prop, git_commit, wilson_ci
except ImportError:  # package-style import
    from agent.stats_paired import as_bool, first_present, fmt_prop, git_commit, wilson_ci

ROUTER_HIT_ALIASES = ("router_hit", "router_target_hit")
METRIC_ALIASES = ("tool_name_match", "tool_name_correct")
NUM_TOOLS_ALIASES = ("num_tools", "n_tools", "num_tools_total")


def _bucket_label(num_tools: Optional[int], bounds: List[int]) -> str:
    if num_tools is None:
        return "unknown"
    previous = 0
    for bound in bounds:
        if num_tools <= bound:
            return f"<={bound}" if previous == 0 else f"{previous + 1}-{bound}"
        previous = bound
    return f">{bounds[-1]}" if bounds else "all"


def _bucket_order(bounds: List[int]) -> List[str]:
    labels = []
    previous = 0
    for bound in bounds:
        labels.append(f"<={bound}" if previous == 0 else f"{previous + 1}-{bound}")
        previous = bound
    if bounds:
        labels.append(f">{bounds[-1]}")
    labels.append("unknown")
    return labels


def _num_tools(row: Dict[str, Any]) -> Optional[int]:
    _, value = first_present(row, NUM_TOOLS_ALIASES)
    if isinstance(value, bool):  # bool is an int subclass; guard first
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _rate_entry(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """tool_name_match k/n over the given rows (missing metric tolerated)."""
    n = len(rows)
    k = 0
    n_missing = 0
    for row in rows:
        _, value = first_present(row, METRIC_ALIASES)
        metric = as_bool(value)
        if metric is None:
            n_missing += 1
        elif metric:
            k += 1
    lo, hi = wilson_ci(k, n)
    return {
        "k": k,
        "n": n,
        "n_missing_metric": n_missing,
        "rate": round(k / n, 4) if n else 0.0,
        "wilson95": [round(lo, 4), round(hi, 4)],
    }


def _render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Router-miss floor (T4)")
    lines.append("")
    lines.append(f"- git_commit: `{report['git_commit']}`")
    lines.append(f"- files: {len(report['files'])}")
    for name, count in report["files"].items():
        lines.append(f"  - `{name}`: {count} rows")
    lines.append(
        f"- router_hit aliases: {', '.join(ROUTER_HIT_ALIASES)}; "
        f"metric aliases: {', '.join(METRIC_ALIASES)}"
    )
    lines.append("")

    lines.append("## Slice size")
    lines.append("")
    lines.append("| slice | count |")
    lines.append("|---|---|")
    lines.append(f"| total rows | {report['n_total']} |")
    lines.append(
        f"| router_hit == False | {fmt_prop(report['n_miss'], report['n_total'])} |"
    )
    lines.append(f"| router_hit == True | {report['n_hit']} |")
    lines.append(f"| router_hit missing/unknown | {report['n_router_unknown']} |")
    lines.append("")

    lines.append("## tool_name_match rate on the router-miss slice")
    lines.append("")
    lines.append("| bucket (num_tools) | rate [wilson95] (k/n) | n_missing_metric |")
    lines.append("|---|---|---:|")
    overall = report["overall"]
    lines.append(
        f"| **all** | {fmt_prop(overall['k'], overall['n'])} | {overall['n_missing_metric']} |"
    )
    for label in report["bucket_order"]:
        entry = report["buckets"].get(label)
        if entry is None or entry["n"] == 0:
            continue
        lines.append(
            f"| {label} | {fmt_prop(entry['k'], entry['n'])} | {entry['n_missing_metric']} |"
        )
    lines.append("")
    if report["warnings"]:
        lines.append("## Warnings")
        lines.append("")
        for warning in report["warnings"]:
            lines.append(f"- **WARN** {warning}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", nargs="+", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--buckets", default="8,16,32")
    args = parser.parse_args()

    bounds = [int(item.strip()) for item in args.buckets.split(",") if item.strip()]
    if not bounds:
        raise SystemExit("error: --buckets must contain at least one integer")

    warnings: List[str] = []
    rows: List[Dict[str, Any]] = []
    files: Dict[str, int] = {}
    for jsonl_path in args.jsonl:
        path = Path(jsonl_path)
        if not path.is_file():
            warnings.append(f"input file not found: {path}")
            print(f"[warn] input file not found: {path}", file=sys.stderr)
            continue
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                rows.append(row)
                count += 1
        files[str(path)] = count

    n_total = len(rows)
    miss_rows: List[Dict[str, Any]] = []
    n_hit = 0
    n_router_unknown = 0
    for row in rows:
        key, value = first_present(row, ROUTER_HIT_ALIASES)
        router_hit = as_bool(value)
        if key is None or router_hit is None:
            n_router_unknown += 1
        elif router_hit:
            n_hit += 1
        else:
            miss_rows.append(row)

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for row in miss_rows:
        buckets.setdefault(_bucket_label(_num_tools(row), bounds), []).append(row)

    report: Dict[str, Any] = {
        "git_commit": git_commit(),
        "files": files,
        "bucket_bounds": bounds,
        "bucket_order": _bucket_order(bounds),
        "n_total": n_total,
        "n_miss": len(miss_rows),
        "n_hit": n_hit,
        "n_router_unknown": n_router_unknown,
        "overall": _rate_entry(miss_rows),
        "buckets": {label: _rate_entry(bucket_rows) for label, bucket_rows in buckets.items()},
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
