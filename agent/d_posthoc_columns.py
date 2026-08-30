"""Post-hoc row columns for D-arm jsonl files (S1.6).

Adds, per row, computed entirely from stored fields:
  - strict_action_match per qid (the file-level aggregate cannot do paired
    tests; the harness-parser-nested strict metric from d_strict_metric)
  - prediction_sha256 / target_sha256 (row-level provenance)

warm repair latency is NOT here by design: it needs the same qid run a
second time inside the process, which cannot be reconstructed post hoc —
recorded as a limitation (prereg v2.10), not re-run for.

Usage: python agent/d_posthoc_columns.py <in.jsonl> [--out <out.jsonl>]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from d_strict_metric import strict_action_match  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    src = Path(args.jsonl)
    out = Path(args.out) if args.out else src.with_suffix(".aug.jsonl")
    n = 0
    with out.open("w", encoding="utf-8") as handle:
        for line in src.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("skipped") and "prediction" in row:
                row["strict_action_match"] = strict_action_match(
                    row.get("prediction", ""), row.get("target", ""))
                row["prediction_sha256"] = hashlib.sha256(
                    row["prediction"].encode("utf-8")).hexdigest()
                row["target_sha256"] = hashlib.sha256(
                    row["target"].encode("utf-8")).hexdigest()
                n += 1
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    print(f"augmented {n} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
