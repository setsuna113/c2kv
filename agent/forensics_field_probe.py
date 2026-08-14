#!/usr/bin/env python
"""Archive field probe: schema census of the frozen round-1 agent-eval files.

Scans *.jsonl and *.summary.json directly under --archive_dir (non-recursive)
and reports, per .jsonl file:
  - row count and per-key coverage fraction over rows (top-level keys, rows
    are JSON objects; non-object / unparseable lines are counted, not fatal);
  - an explicit presence matrix for the probe fields (qid, session_id, mode,
    arm, prediction, target, target_tool_name, tool_name_match, tool_call,
    has_tool_call, router_strategy, router_hit, num_tools, doc_tokens,
    logp_prefix_c2kv, logp_prefix_full, delta_logp_prefix, skipped,
    skip_reason);
  - for files with a mode (or arm) key: per-mode qid sets and the pairwise
    intersection-count matrix, flagging any mode pair whose intersection is
    below 80% of the smaller set as "subset drift".
Per .summary.json file: top-level keys plus the keys of any embedded
args/config/run_args/kwargs dict.

Outputs <out_prefix>.json and <out_prefix>.md.

Example:
  python agent/forensics_field_probe.py \
    --archive_dir outputs/r1_frozen --out_prefix outputs/r2/field_probe
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from stats_paired import fmt_prop, git_commit, wilson_ci
except ImportError:  # package-style import
    from agent.stats_paired import fmt_prop, git_commit, wilson_ci

PROBE_FIELDS = (
    "qid",
    "session_id",
    "mode",
    "arm",
    "prediction",
    "target",
    "target_tool_name",
    "tool_name_match",
    "tool_call",
    "has_tool_call",
    "router_strategy",
    "router_hit",
    "num_tools",
    "doc_tokens",
    "logp_prefix_c2kv",
    "logp_prefix_full",
    "delta_logp_prefix",
    "skipped",
    "skip_reason",
)

ARGS_DICT_KEYS = ("args", "config", "run_args", "kwargs")
DRIFT_MIN_OVERLAP = 0.8  # flag mode pairs with |intersection| < 0.8 * smaller set


def _read_rows(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Read a jsonl file tolerantly; returns (object rows, n_bad_lines)."""
    rows: List[Dict[str, Any]] = []
    n_bad = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            if isinstance(value, dict):
                rows.append(value)
            else:
                n_bad += 1
    return rows, n_bad


def _coverage(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-key coverage over rows: key present and value not None."""
    counts: Counter = Counter()
    for row in rows:
        for key, value in row.items():
            if value is not None:
                counts[key] += 1
    n = len(rows)
    return {
        key: {"k": counts[key], "n": n, "frac": round(counts[key] / n, 4) if n else 0.0}
        for key in sorted(counts)
    }


def _mode_analysis(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Per-mode qid sets, pairwise intersection counts, and subset-drift flags."""
    mode_key = None
    for candidate in ("mode", "arm"):
        if any(candidate in row for row in rows):
            mode_key = candidate
            break
    if mode_key is None:
        return None
    qids_by_mode: Dict[str, set] = {}
    n_rows_by_mode: Counter = Counter()
    n_missing_qid = 0
    for row in rows:
        mode_value = row.get(mode_key)
        if mode_value is None:
            continue
        mode = str(mode_value)
        n_rows_by_mode[mode] += 1
        qid = row.get("qid")
        if qid is None:
            n_missing_qid += 1
            continue
        qids_by_mode.setdefault(mode, set()).add(str(qid))
    modes = sorted(set(n_rows_by_mode) | set(qids_by_mode))
    intersection: Dict[str, Dict[str, int]] = {mode: {} for mode in modes}
    drift: List[Dict[str, Any]] = []
    for i, mode_a in enumerate(modes):
        set_a = qids_by_mode.get(mode_a, set())
        for mode_b in modes[i:]:
            set_b = qids_by_mode.get(mode_b, set())
            count = len(set_a & set_b)
            intersection[mode_a][mode_b] = count
            intersection[mode_b][mode_a] = count
            if mode_a == mode_b:
                continue
            smaller = min(len(set_a), len(set_b))
            if smaller > 0 and count < DRIFT_MIN_OVERLAP * smaller:
                drift.append(
                    {
                        "mode_a": mode_a,
                        "mode_b": mode_b,
                        "n_a": len(set_a),
                        "n_b": len(set_b),
                        "intersection": count,
                        "overlap_ratio_of_smaller": round(count / smaller, 4),
                    }
                )
    return {
        "mode_key": mode_key,
        "n_missing_mode": sum(1 for row in rows if row.get(mode_key) is None),
        "n_missing_qid": n_missing_qid,
        "modes": {
            mode: {
                "n_rows": n_rows_by_mode.get(mode, 0),
                "n_qids": len(qids_by_mode.get(mode, set())),
            }
            for mode in modes
        },
        "intersection": intersection,
        "subset_drift": drift,
    }


def probe_jsonl(path: Path) -> Dict[str, Any]:
    rows, n_bad = _read_rows(path)
    n = len(rows)
    key_coverage = _coverage(rows)
    probe_fields: Dict[str, Any] = {}
    for field in PROBE_FIELDS:
        entry = key_coverage.get(field, {"k": 0, "n": n, "frac": 0.0})
        lo, hi = wilson_ci(entry["k"], n)
        probe_fields[field] = {**entry, "wilson95": [round(lo, 4), round(hi, 4)]}
    n_present = sum(1 for field in PROBE_FIELDS if probe_fields[field]["k"] > 0)
    return {
        "file": path.name,
        "rows": n,
        "n_bad_lines": n_bad,
        "key_coverage": key_coverage,
        "probe_fields": probe_fields,
        "probe_fields_present": {"k": n_present, "n": len(PROBE_FIELDS)},
        "mode_analysis": _mode_analysis(rows),
    }


def probe_summary(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return {"file": path.name, "error": str(error), "top_level_keys": [], "args_dict_keys": {}}
    if not isinstance(data, dict):
        return {
            "file": path.name,
            "error": f"top-level JSON is {type(data).__name__}, not an object",
            "top_level_keys": [],
            "args_dict_keys": {},
        }
    args_dict_keys = {
        key: sorted(data[key]) for key in ARGS_DICT_KEYS if isinstance(data.get(key), dict)
    }
    return {"file": path.name, "top_level_keys": sorted(data), "args_dict_keys": args_dict_keys}


def _render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Archive field probe")
    lines.append("")
    lines.append(f"- git_commit: `{report['git_commit']}`")
    lines.append(f"- archive_dir: `{report['archive_dir']}`")
    lines.append(
        f"- files: {report['n_jsonl_files']} *.jsonl, {report['n_summary_files']} *.summary.json"
    )
    lines.append("")

    lines.append("## Subset drift warnings")
    lines.append("")
    warnings = report["warnings"]
    if not warnings:
        lines.append("None.")
    for warning in warnings:
        lines.append(f"- {warning}")
    lines.append("")

    lines.append("## Per-jsonl overview")
    lines.append("")
    lines.append(
        "| file | rows | bad lines | probe fields present [wilson95] | mode key | modes |"
    )
    lines.append("|---|---:|---:|---|---|---:|")
    for entry in report["jsonl_files"]:
        present = entry["probe_fields_present"]
        mode_analysis = entry["mode_analysis"]
        if mode_analysis:
            mode_key = mode_analysis["mode_key"]
            n_modes = len(mode_analysis["modes"])
        else:
            mode_key = "-"
            n_modes = 0
        lines.append(
            f"| {entry['file']} | {entry['rows']} | {entry['n_bad_lines']} "
            f"| {fmt_prop(present['k'], present['n'])} "
            f"| {mode_key} | {n_modes} |"
        )
    lines.append("")

    lines.append("## Probe-field presence matrix (k/n rows with the key non-null)")
    lines.append("")
    header = "| file | " + " | ".join(PROBE_FIELDS) + " |"
    lines.append(header)
    lines.append("|---" * (len(PROBE_FIELDS) + 1) + "|")
    for entry in report["jsonl_files"]:
        cells = []
        for field in PROBE_FIELDS:
            field_entry = entry["probe_fields"][field]
            cells.append(
                f"{field_entry['k']}/{field_entry['n']}" if field_entry["n"] else "-"
            )
        lines.append(f"| {entry['file']} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-mode qid-set intersections")
    lines.append("")
    any_modes = False
    for entry in report["jsonl_files"]:
        mode_analysis = entry["mode_analysis"]
        if not mode_analysis:
            continue
        any_modes = True
        modes = sorted(mode_analysis["modes"])
        lines.append(f"### {entry['file']} (mode key: `{mode_analysis['mode_key']}`)")
        lines.append("")
        lines.append("| mode | " + " | ".join(modes) + " |")
        lines.append("|---" * (len(modes) + 1) + "|")
        for mode_a in modes:
            row_cells = [
                str(mode_analysis["intersection"].get(mode_a, {}).get(mode_b, 0))
                for mode_b in modes
            ]
            lines.append(f"| {mode_a} | " + " | ".join(row_cells) + " |")
        lines.append("")
        drift = mode_analysis["subset_drift"]
        if drift:
            for flag in drift:
                lines.append(
                    f"- **subset drift**: `{flag['mode_a']}` vs `{flag['mode_b']}`: "
                    f"intersection {flag['intersection']} < 0.8 * min("
                    f"{flag['n_a']}, {flag['n_b']}) "
                    f"(overlap {flag['overlap_ratio_of_smaller']:.4f} of smaller set)"
                )
            lines.append("")
    if not any_modes:
        lines.append("No file carries a mode/arm key.")
        lines.append("")

    lines.append("## Summary-file key census")
    lines.append("")
    lines.append("| file | top-level keys | embedded args-dict keys |")
    lines.append("|---|---|---|")
    for entry in report["summary_files"]:
        if entry.get("error"):
            lines.append(f"| {entry['file']} | ERROR: {entry['error']} | |")
            continue
        args_keys = [
            f"{dict_key}.{sub_key}"
            for dict_key, sub_keys in sorted(entry["args_dict_keys"].items())
            for sub_key in sub_keys
        ]
        lines.append(
            f"| {entry['file']} | {', '.join(entry['top_level_keys'])} "
            f"| {', '.join(args_keys) if args_keys else '-'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive_dir", required=True)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir)
    if not archive_dir.is_dir():
        raise SystemExit(f"error: --archive_dir not found: {archive_dir}")

    jsonl_paths = sorted(
        path
        for path in archive_dir.iterdir()
        if path.is_file() and path.suffix == ".jsonl" and not path.name.endswith(".summary.json")
    )
    summary_paths = sorted(
        path for path in archive_dir.iterdir() if path.is_file() and path.name.endswith(".summary.json")
    )

    jsonl_entries = [probe_jsonl(path) for path in jsonl_paths]
    summary_entries = [probe_summary(path) for path in summary_paths]

    warnings: List[str] = []
    for entry in jsonl_entries:
        for flag in (entry["mode_analysis"] or {}).get("subset_drift", []):
            warnings.append(
                f"{entry['file']}: subset drift {flag['mode_a']} vs {flag['mode_b']} "
                f"(intersection {flag['intersection']}, smaller set "
                f"{min(flag['n_a'], flag['n_b'])})"
            )
        if entry["rows"] == 0:
            warnings.append(f"{entry['file']}: 0 parseable rows")
        if entry["n_bad_lines"]:
            warnings.append(f"{entry['file']}: {entry['n_bad_lines']} unparseable/non-object lines")

    report: Dict[str, Any] = {
        "git_commit": git_commit(),
        "archive_dir": str(archive_dir),
        "n_jsonl_files": len(jsonl_entries),
        "n_summary_files": len(summary_entries),
        "probe_fields": list(PROBE_FIELDS),
        "jsonl_files": jsonl_entries,
        "summary_files": summary_entries,
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
