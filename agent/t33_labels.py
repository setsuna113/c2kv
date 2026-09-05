# -*- coding: utf-8 -*-
"""Frozen battery labels, rowset join and the feature-frame leakage guard.

Survey item 4.0-1.  The C->W unit is defined ONCE here:

    label(qid) = 1  iff  full_row.tool_name_match  AND NOT c2kv_row.tool_name_match
              = 0  iff  full_row.tool_name_match  AND     c2kv_row.tool_name_match  (C->C)
              = None otherwise (W->C / W->W — not in the trigger table's
                                 denominators; kept for the Diff-01 three-valued arm)

and cross-checked against the frozen manifest ``cw_qids`` (93).  Features NEVER
read: ``target``, ``tool_name_match``, any scoring column, any full-arm field.
The guard below is mechanical, and the unit test feeds it the two historical
leak specimens (``a_made_call``; a raw ``tool_name_match`` column) and requires
both to be rejected.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Columns that may appear in a FEATURE frame.  Everything from the compressed
# arm's own row is allowed EXCEPT scoring columns; qid/session_id are keys;
# mode/ratio are constants of the run.
_SCORING_COL = re.compile(
    r"(^|_)(target|gold|prediction_tool_name)"      # target-side identity
    r"|tool_name_match|exact_match|response_type_match"
    r"|rouge|token_f1|_f1$|^f1_"                    # scoring family
    r"|a_made_call|made_call|a_called"              # historical leak specimens
)
_FULL_ARM_MARKER = ("full_", "_full")

# Historical leak specimens that the guard MUST catch (see test):
#   a_made_call        -> label ingredient (requires a_called on the label side)
#   tool_name_match    -> the label itself
KNOWN_LEAK_COLUMNS = ("a_made_call", "tool_name_match", "target_tool_name",
                      "exact_match", "target", "text_token_f1")


def guard_columns(columns: Iterable[str], *, context: str = "feature frame") -> None:
    """Raise if any column name can carry label or full-arm information.

    ``full_``-prefixed columns are full-arm fields by construction; the join
    keeps them in the label frame only, never in the feature frame.
    """
    bad: List[str] = []
    for col in columns:
        if _SCORING_COL.search(col):
            bad.append(col)
            continue
        if col.startswith(_FULL_ARM_MARKER) or col.endswith(_FULL_ARM_MARKER):
            bad.append(col)
    if bad:
        raise ValueError(
            f"{context}: label-leaking columns refused: {sorted(set(bad))}"
        )


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with io.open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def join_arms(full_rows: List[Dict[str, Any]],
              c2kv_rows: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Join the two battery files on qid; both sides keep their own row."""
    by_qid_full = {r["qid"]: r for r in full_rows}
    by_qid_c2kv = {r["qid"]: r for r in c2kv_rows}
    if set(by_qid_full) != set(by_qid_c2kv):
        only_full = set(by_qid_full) - set(by_qid_c2kv)
        only_c2kv = set(by_qid_c2kv) - set(by_qid_full)
        raise ValueError(f"qid sets differ: only_full={len(only_full)} only_c2kv={len(only_c2kv)}")
    pairs = []
    for qid in sorted(by_qid_full):
        f, c = by_qid_full[qid], by_qid_c2kv[qid]
        if f.get("session_id") != c.get("session_id"):
            raise ValueError(f"session_id mismatch at {qid}")
        if f.get("skipped") or c.get("skipped"):
            raise ValueError(f"skipped row in frozen battery at {qid}")
        pairs.append((f, c))
    return pairs


def cw_label(full_row: Dict[str, Any], c2kv_row: Dict[str, Any]) -> Optional[int]:
    """1 = C->W (trigger), 0 = C->C, None = outside the trigger denominators."""
    if not full_row.get("tool_name_match"):
        return None
    return 0 if c2kv_row.get("tool_name_match") else 1


def build_label_frame(pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]],
                      manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One record per qid with the label and the shared/neutral columns.

    Deliberately tiny: features live in a SEPARATE frame whose column names go
    through :func:`guard_columns`.  This frame is allowed to hold scoring
    columns because it is never a model input.
    """
    cw_qids = set(manifest["cw_qids"])
    records = []
    recomputed_cw = set()
    for f, c in pairs:
        label = cw_label(f, c)
        if label == 1:
            recomputed_cw.add(f["qid"])
        records.append({
            "qid": f["qid"],
            "session_id": f["session_id"],
            "label_cw": label,
            # denominators / stratifiers (not features)
            "censored_at_cap": bool(c.get("generated_tokens", 0) >= _cap_tokens(manifest)),
            "censored_at_cap_full": bool(f.get("generated_tokens", 0) >= _cap_tokens(manifest)),
            # parse-failure baseline over the COMPRESSED arm's own text only
            "parse_fail_fire": parse_fail_baseline(c.get("prediction", ""),
                                                   bool(c.get("target_has_tool_call"))),
            # three-valued Diff-01 target (needs the full arm — label side only)
            "z_deferral": (1 if f.get("tool_name_match") and not c.get("tool_name_match")
                           else (-1 if not f.get("tool_name_match") and c.get("tool_name_match")
                                 else 0)),
        })
    if recomputed_cw != cw_qids:
        missing = cw_qids - recomputed_cw
        extra = recomputed_cw - cw_qids
        raise ValueError(
            "C->W recomputation disagrees with the frozen manifest: "
            f"manifest_only={sorted(missing)[:5]} recomputed_only={sorted(extra)[:5]}"
        )
    return records


def _cap_tokens(manifest: Dict[str, Any]) -> int:
    return int(manifest.get("kv_recipe", {}).get("max_new_tokens", 128))


def parse_fail_baseline(prediction: str, target_has_tool_call: bool) -> bool:
    """The L1 baseline: fire iff the compressed arm's own emission is
    unparseable while a tool call was expected.

    Uses the shared parser (t33_spanmap) so 'unparseable' means exactly what
    the span map could not strictly parse — censoring without a closing tag
    still counts as parseable when the JSON object itself balanced.
    """
    if not target_has_tool_call:
        return False
    from t33_spanmap import parse_tool_call
    parsed = parse_tool_call(prediction or "")
    return not parsed["parse_ok"]


def census(label_frame: List[Dict[str, Any]], manifest: Dict[str, Any]) -> Dict[str, Any]:
    n = len(label_frame)
    cw = [r for r in label_frame if r["label_cw"] == 1]
    cc = [r for r in label_frame if r["label_cw"] == 0]
    sessions = {r["session_id"] for r in label_frame}
    stats = {
        "n_paired": n,
        "n_sessions": len(sessions),
        "n_cw": len(cw),
        "n_cw_sessions": len({r["session_id"] for r in cw}),
        "n_cc": len(cc),
        "n_cc_sessions": len({r["session_id"] for r in cc}),
        "base_rate": round(len(cw) / n, 4) if n else None,
        "censored_cw": sum(1 for r in cw if r["censored_at_cap"]),
        "censored_cc": sum(1 for r in cc if r["censored_at_cap"]),
        "parse_fail_fire_cw": sum(1 for r in cw if r["parse_fail_fire"]),
        "parse_fail_fire_cc": sum(1 for r in cc if r["parse_fail_fire"]),
        "manifest_transitions": manifest.get("transitions"),
    }
    guard_columns([], context="census")  # no-op; keeps the guard imported/used
    return stats


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", required=True, help="battery_full.jsonl (frozen r2)")
    parser.add_argument("--c2kv", required=True, help="battery_c2kv.jsonl (frozen r2)")
    parser.add_argument("--manifest", required=True, help="d_cw_manifest_r2.json")
    parser.add_argument("--out", default=None, help="optional stats json path")
    args = parser.parse_args(argv)

    full_rows = load_jsonl(args.full)
    c2kv_rows = load_jsonl(args.c2kv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pairs = join_arms(full_rows, c2kv_rows)
    frame = build_label_frame(pairs, manifest)
    stats = census(frame, manifest)
    stats["label_frame_columns"] = sorted(frame[0].keys())

    text = json.dumps(stats, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
