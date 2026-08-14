#!/usr/bin/env python
"""Tasks T1/T2: registry of the run args embedded in round-1 *.summary.json
files, with back-inference evidence for runs whose thresholds are missing.

For every *.summary.json directly under --archive_dir (non-recursive):
  - registers the embedded run args, looked up under args / config /
    run_args / kwargs dicts first, then flat scalar top-level keys;
  - records the modes/arms covered (from the summary results rows, a
    top-level modes list, or the same-stem .jsonl row values);
  - checks for threshold-relevant fields (max_doc_length, min_target_tokens,
    max_samples_per_session, eval_ratio, ratio, ratios, max_new_tokens,
    max_baseline_input_tokens, max_tool_definition_tokens, max_doc_num);
  - for files where none of these are present, loads the same-stem .jsonl
    (if present) and reports the row-level doc_tokens distribution
    (n, p50, p90, max, min) as back-inference evidence, marking
    threshold_source=inferred (vs explicit / unknown).

Outputs <out_prefix>.json and <out_prefix>.md.

Example:
  python agent/forensics_summary_args_registry.py \
    --archive_dir outputs/r1_frozen --out_prefix outputs/r2/args_registry
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
    from stats_paired import fmt_prop, git_commit
except ImportError:  # package-style import
    from agent.stats_paired import fmt_prop, git_commit

ARGS_DICT_KEYS = ("args", "config", "run_args", "kwargs")
THRESHOLD_FIELDS = (
    "max_doc_length",
    "min_target_tokens",
    "max_samples_per_session",
    "eval_ratio",
    "ratio",
    "ratios",
    "max_new_tokens",
    "max_baseline_input_tokens",
    "max_tool_definition_tokens",
    "max_doc_num",
)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _stat(values: List[float]) -> Dict[str, Any]:
    """Distribution summary in the _stat idiom of the inspect scripts."""
    if not values:
        return {"n": 0, "min": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(values)
    p90_index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.9)))
    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[p90_index],
        "max": ordered[-1],
    }


def _find_threshold_fields(
    data: Dict[str, Any], args_dicts: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Locate threshold-relevant fields: embedded args dicts first, then flat."""
    found: Dict[str, Any] = {}
    for field in THRESHOLD_FIELDS:
        for dict_key in ARGS_DICT_KEYS:
            args_dict = args_dicts.get(dict_key)
            if args_dict is not None and field in args_dict:
                found[field] = {
                    "value": args_dict[field],
                    "source": f"{dict_key}.{field}",
                }
                break
        else:
            if field in data and _is_scalar(data[field]):
                found[field] = {"value": data[field], "source": field}
    return found


def _modes_from_summary(data: Dict[str, Any]) -> List[str]:
    """Modes/arms from the results rows or a top-level modes list."""
    modes = set()
    results = data.get("results")
    if isinstance(results, list):
        for row in results:
            if not isinstance(row, dict):
                continue
            mode = row.get("mode", row.get("arm"))
            if mode is None:
                continue
            ratio = row.get("ratio")
            modes.add(f"{mode}@{ratio}" if ratio is not None else str(mode))
    top_modes = data.get("modes", data.get("arms"))
    if isinstance(top_modes, list):
        for mode in top_modes:
            if _is_scalar(mode):
                modes.add(str(mode))
    return sorted(modes)


def _scan_same_stem_jsonl(summary_path: Path) -> Optional[Dict[str, Any]]:
    """Load the same-stem .jsonl (foo.summary.json -> foo.jsonl) if present."""
    name = summary_path.name
    if not name.endswith(".summary.json"):
        return None
    jsonl_path = summary_path.with_name(name[: -len(".summary.json")] + ".jsonl")
    if not jsonl_path.is_file():
        return None
    modes = set()
    doc_tokens: List[float] = []
    n_rows = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            n_rows += 1
            mode = row.get("mode", row.get("arm"))
            if mode is not None:
                modes.add(str(mode))
            value = row.get("doc_tokens")
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                doc_tokens.append(value)
    return {
        "jsonl_file": jsonl_path.name,
        "n_rows": n_rows,
        "modes": sorted(modes),
        "doc_tokens": _stat(doc_tokens),
    }


def register_summary(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return {
            "file": path.name,
            "error": str(error),
            "modes": [],
            "threshold_fields": {},
            "threshold_source": "unknown",
        }
    if not isinstance(data, dict):
        return {
            "file": path.name,
            "error": f"top-level JSON is {type(data).__name__}, not an object",
            "modes": [],
            "threshold_fields": {},
            "threshold_source": "unknown",
        }

    args_dicts = {key: data[key] for key in ARGS_DICT_KEYS if isinstance(data.get(key), dict)}
    flat_scalar_keys = sorted(key for key, value in data.items() if _is_scalar(value))
    threshold_fields = _find_threshold_fields(data, args_dicts)

    stem = _scan_same_stem_jsonl(path)
    modes = _modes_from_summary(data)
    modes_from = "summary"
    if not modes and stem:
        modes = stem["modes"]
        modes_from = "jsonl"

    if threshold_fields:
        threshold_source = "explicit"
    elif stem is not None:
        threshold_source = "inferred"
    else:
        threshold_source = "unknown"

    entry: Dict[str, Any] = {
        "file": path.name,
        "modes": modes,
        "modes_from": modes_from,
        "args_dict_keys": {key: sorted(value) for key, value in args_dicts.items()},
        "flat_scalar_keys": flat_scalar_keys,
        "threshold_fields": threshold_fields,
        "threshold_fields_found": len(threshold_fields),
        "threshold_fields_total": len(THRESHOLD_FIELDS),
        "threshold_source": threshold_source,
    }
    if stem is not None:
        entry["same_stem_jsonl"] = stem
    if threshold_source == "inferred":
        entry["doc_tokens"] = stem["doc_tokens"]
    return entry


def _render_md(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Summary-args registry (T1/T2)")
    lines.append("")
    lines.append(f"- git_commit: `{report['git_commit']}`")
    lines.append(f"- archive_dir: `{report['archive_dir']}`")
    lines.append(f"- summary files: {report['n_summary_files']}")
    counts = report["threshold_source_counts"]
    total = report["n_summary_files"]
    lines.append(
        f"- threshold_source: explicit {fmt_prop(counts['explicit'], total)}; "
        f"inferred {fmt_prop(counts['inferred'], total)}; "
        f"unknown {fmt_prop(counts['unknown'], total)}"
    )
    lines.append("")

    lines.append("## Registry")
    lines.append("")
    lines.append(
        "| file | modes/arms | threshold fields (k/K) | found values | threshold_source "
        "| doc_tokens back-inference (n, p50, p90, max, min) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for entry in report["entries"]:
        if entry.get("error"):
            lines.append(f"| {entry['file']} | | | ERROR: {entry['error']} | unknown | |")
            continue
        found = entry["threshold_fields"]
        found_text = (
            ", ".join(f"{field}={found[field]['value']}" for field in THRESHOLD_FIELDS if field in found)
            if found
            else "-"
        )
        doc = entry.get("doc_tokens")
        doc_text = (
            f"n={doc['n']}, p50={doc['p50']}, p90={doc['p90']}, max={doc['max']}, min={doc['min']}"
            if doc
            else "-"
        )
        lines.append(
            f"| {entry['file']} | {', '.join(entry['modes']) if entry['modes'] else '-'} "
            f"| {entry['threshold_fields_found']}/{entry['threshold_fields_total']} "
            f"| {found_text} | {entry['threshold_source']} | {doc_text} |"
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

    summary_paths = sorted(
        path
        for path in archive_dir.iterdir()
        if path.is_file() and path.name.endswith(".summary.json")
    )
    entries = [register_summary(path) for path in summary_paths]
    counts = {"explicit": 0, "inferred": 0, "unknown": 0}
    for entry in entries:
        counts[entry.get("threshold_source", "unknown")] += 1

    report: Dict[str, Any] = {
        "git_commit": git_commit(),
        "archive_dir": str(archive_dir),
        "n_summary_files": len(entries),
        "threshold_fields": list(THRESHOLD_FIELDS),
        "threshold_source_counts": counts,
        "entries": entries,
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
