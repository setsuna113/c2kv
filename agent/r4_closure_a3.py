"""R4 A3 closure (zero-compute): finish_reason/completion_tokens distribution
table across the three 76k arms + append/resume-channel file audit.

① distribution table: T-A trusted (sglang, base), T-E c2kv (r3), R4 new full
   arm (checkpoint-250). finish_reason semantics differ per arm and are
   normalized: "length" iff completion hit the 128-token cap, else "stop".
② append/resume audit: per file — line count, unique qids, duplicates,
   skipped rows.

Usage:
  python agent/r4_closure_a3.py --ta t_a_generations.jsonl \
      --te t_e_c2kv_r4.jsonl --r4 r4_full_76k.jsonl --out results/r4/a3_closure.json
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("r4_closure_a3")
MAX_NEW_TOKENS = 128


def _load_lines(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in open(path, "r", encoding="utf-8") if l.strip()]


def _norm_finish(row: Dict[str, Any], source: str) -> str:
    if source == "ta":
        fr = row.get("finish_reason") or ""
        return "length" if "length" in str(fr) else "stop"
    tokens = row.get("completion_tokens", row.get("generated_tokens"))
    if tokens is None:
        return "unknown"
    return "length" if int(tokens) >= MAX_NEW_TOKENS else "stop"


def _arm_block(path: str, source: str) -> Dict[str, Any]:
    rows = _load_lines(path)
    qids = [r.get("qid") for r in rows]
    dup = len(qids) - len(set(qids))
    valid = [r for r in rows if not r.get("skipped")]
    finishes = Counter(_norm_finish(r, source) for r in valid)
    comp = [int(r.get("completion_tokens", r.get("generated_tokens", 0))) for r in valid]
    comp_hist = Counter(min(c, MAX_NEW_TOKENS) for c in comp)
    return {
        "file": path,
        "rows_total": len(rows),
        "rows_valid": len(valid),
        "rows_skipped": len(rows) - len(valid),
        "unique_qids": len(set(qids)),
        "duplicate_qids": dup,
        "finish_reason": dict(finishes),
        "completion_tokens": {
            "min": min(comp) if comp else None,
            "max": max(comp) if comp else None,
            "at_cap_128": comp_hist.get(MAX_NEW_TOKENS, 0),
            "hist": {str(k): v for k, v in sorted(comp_hist.items())},
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ta", required=True, help="T-A trusted arm generations jsonl (48)")
    p.add_argument("--te", required=True, help="T-E c2kv arm jsonl (48)")
    p.add_argument("--r4", required=True, help="R4 new full arm jsonl")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    report = {
        "note": "finish_reason normalized: 'length' iff completion hit the 128-token cap. T-A field is the sglang meta_info; T-E/R4 derived from completion_tokens.",
        "arms": {
            "t_a_trusted_sglang_base": _arm_block(args.ta, "ta"),
            "t_e_c2kv_checkpoint250": _arm_block(args.te, "te"),
            "r4_full_checkpoint250": _arm_block(args.r4, "r4"),
        },
    }
    for name, block in report["arms"].items():
        logger.info(
            "%s: rows=%d valid=%d dup=%d finish=%s",
            name, block["rows_total"], block["rows_valid"], block["duplicate_qids"], block["finish_reason"],
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
