#!/usr/bin/env python
"""Freeze the S4 qid subset from round-1 arm jsonls.

Round-1 S4 arm runs (agent/eval_agent_history_s4_npu.sh) wrote per-sample rows
with `skipped` and `prediction` fields. The frozen subset consumed by the clean
logp recompute (agent/recompute_s4_logp.py) and the trigger-signal analysis
(agent/analyze_trigger_signals.py) is the set of qids with a valid row --
`skipped == False` and a non-empty `prediction` string -- in EVERY supplied arm
file (typically arms B and C, the two c2kv@4 arms).

Example:
  python agent/extract_s4_frozen_qids.py \
    --arm_jsonl outputs/s4_armB_c2kv.jsonl outputs/s4_armC_c2kv_forced.jsonl \
    --out configs/s4_frozen_qids.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def _git_commit() -> str:
    """Best-effort git commit hash of this repo; 'unknown' on any failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[1],
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _valid_qids(path: str) -> Tuple[Set[str], int]:
    """Qids with a valid row (not skipped, non-empty prediction) in one arm file.

    Returns (valid_qids, total_rows). A row counts as valid iff
    `row["skipped"] is False` and `row["prediction"]` is a non-blank string;
    rows missing either field are treated as invalid (defensive: the eval
    writer always emits both).
    """
    qids: Set[str] = set()
    n_rows = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            n_rows += 1
            row: Dict[str, Any] = json.loads(line)
            if row.get("skipped", True):
                continue
            prediction = row.get("prediction")
            if not isinstance(prediction, str) or not prediction.strip():
                continue
            qid = row.get("qid")
            if qid:
                qids.add(str(qid))
    return qids, n_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm_jsonl",
        nargs="+",
        required=True,
        help="Round-1 S4 arm jsonl files (e.g. arms B and C).",
    )
    parser.add_argument("--out", required=True, help="Output JSON path (e.g. configs/s4_frozen_qids.json).")
    args = parser.parse_args()

    per_file: List[Dict[str, Any]] = []
    valid_sets: List[Set[str]] = []
    for path in args.arm_jsonl:
        qids, n_rows = _valid_qids(path)
        valid_sets.append(qids)
        per_file.append({"file": path, "n_rows": n_rows, "n_valid": len(qids)})
        print(f"{path}: valid {len(qids)}/{n_rows} rows")

    print("pairwise valid-qid intersections:")
    for (i, left), (j, right) in combinations(enumerate(valid_sets), 2):
        print(
            f"  [{i}] {args.arm_jsonl[i]} ∩ [{j}] {args.arm_jsonl[j]}:"
            f" {len(left & right)} qids"
        )

    frozen = set.intersection(*valid_sets) if valid_sets else set()
    payload = {
        "created_from": list(args.arm_jsonl),
        "n_qids": len(frozen),
        "qids": sorted(frozen),
        "git_commit": _git_commit(),
        "per_file_valid": per_file,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"froze {payload['n_qids']} qids -> {out_path} (git {payload['git_commit']})")


if __name__ == "__main__":
    main()
