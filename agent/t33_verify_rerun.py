# -*- coding: utf-8 -*-
"""Determinism gate for the t33 capture rerun (survey item 4.0-2).

The frozen d_r2 battery rows were produced by commit f84dc74 (an ancestor of
this branch).  The capture rerun adds instrumentation that must not change the
generation path; this script compares every rerun row against its frozen
counterpart on the fields that define the battery: prediction text,
generated_tokens, tool_name_match, has_tool_call.  Any prediction mismatch
means the rerun is NOT the frozen battery and downstream feature numbers do
not inherit the manifest labels — report and stop.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t33_labels import load_jsonl  # noqa: E402


def compare(frozen_rows: List[Dict[str, Any]], rerun_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    frozen = {r["qid"]: r for r in frozen_rows}
    rerun = {r["qid"]: r for r in rerun_rows}
    missing = sorted(set(frozen) - set(rerun))
    extra = sorted(set(rerun) - set(frozen))

    pred_mismatch: List[Dict[str, Any]] = []
    tok_mismatch: List[str] = []
    match_mismatch: List[str] = []
    call_mismatch: List[str] = []
    capture_missing: List[str] = []
    for qid in sorted(set(frozen) & set(rerun)):
        f, r = frozen[qid], rerun[qid]
        if (f.get("prediction") or "") != (r.get("prediction") or ""):
            fp = f.get("prediction") or ""
            rp = r.get("prediction") or ""
            first_diff = next(
                (i for i in range(min(len(fp), len(rp))) if fp[i] != rp[i]),
                min(len(fp), len(rp)),
            )
            pred_mismatch.append({
                "qid": qid,
                "frozen_len": len(fp), "rerun_len": len(rp),
                "first_diff_char": first_diff,
                "frozen_head": fp[max(0, first_diff - 20):first_diff + 40],
                "rerun_head": rp[max(0, first_diff - 20):first_diff + 40],
            })
        if f.get("generated_tokens") != r.get("generated_tokens"):
            tok_mismatch.append(qid)
        if bool(f.get("tool_name_match")) != bool(r.get("tool_name_match")):
            match_mismatch.append(qid)
        if bool(f.get("has_tool_call")) != bool(r.get("has_tool_call")):
            call_mismatch.append(qid)
        if r.get("skipped"):
            capture_missing.append(qid)
        elif not isinstance(r.get("capture"), dict):
            capture_missing.append(qid)

    n = len(frozen)
    return {
        "n_frozen": n,
        "n_rerun": len(rerun),
        "missing_qids": missing[:10],
        "n_missing": len(missing),
        "extra_qids": extra[:10],
        "n_extra": len(extra),
        "n_prediction_mismatch": len(pred_mismatch),
        "prediction_mismatch_examples": pred_mismatch[:5],
        "n_generated_tokens_mismatch": len(tok_mismatch),
        "generated_tokens_mismatch_examples": tok_mismatch[:5],
        "n_tool_name_match_mismatch": len(match_mismatch),
        "tool_name_match_mismatch_examples": match_mismatch[:5],
        "n_has_tool_call_mismatch": len(call_mismatch),
        "n_capture_missing": len(capture_missing),
        "capture_missing_examples": capture_missing[:5],
        "gate": "PASS" if (not pred_mismatch and not tok_mismatch and not match_mismatch
                           and not missing and not extra and n == len(rerun)) else "FAIL",
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--rerun", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    result = compare(load_jsonl(args.frozen), load_jsonl(args.rerun))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0 if result["gate"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
