"""R4 task D: freeze the PR#1 sample set from the merged per-arm archives.

The first-round (PR#1, task/s4-forced-prefix) four-arm eval has no manifest
file on disk; its sample set is reconstructed here as the union of valid
(non-skipped) qids across the merged_{A,B,C,D}.jsonl archives, each bound by
sha256. Also computes the regime census (history token distribution) and a
latency-based GPU-hour estimate that drives the pre-authorized subset rule:

  if estimated 3-arm GPU time > 35 h -> freeze only the first 300 qids
  (sorted by qid); otherwise freeze the full union.

The estimate uses arm-B (c2kv, free generation) latency as the per-arm proxy
times 3 arms, times a 1.25 margin for anchor-arm overhead.

Usage (anywhere the merged archives are available):
  python agent/r4_freeze_d_qids.py --merged_dir <dir> --out configs/r4_d_qids.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("r4_freeze_d_qids")

SUBSET_RULE_THRESHOLD_H = 35.0
SUBSET_RULE_N = 300
LATENCY_MARGIN = 1.25


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--merged_dir", required=True, help="dir with merged_{A,B,C,D}.jsonl")
    p.add_argument("--out", default="./configs/r4_d_qids.json")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    merged_dir = Path(args.merged_dir)
    per_arm: Dict[str, List[Dict[str, Any]]] = {}
    sources: Dict[str, Any] = {}
    for arm in "ABCD":
        path = merged_dir / f"merged_{arm}.jsonl"
        rows = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
        per_arm[arm] = rows
        valid = [r for r in rows if not r.get("skipped")]
        sources[arm] = {
            "path": str(path),
            "sha256": _sha256(path),
            "rows": len(rows),
            "valid_rows": len(valid),
        }
        logger.info("arm %s: rows=%d valid=%d sha256=%s", arm, len(rows), len(valid), sources[arm]["sha256"][:12])

    # Union of valid qids; per-qid session + per-arm membership.
    qid_info: Dict[str, Dict[str, Any]] = {}
    for arm, rows in per_arm.items():
        for r in rows:
            if r.get("skipped"):
                continue
            info = qid_info.setdefault(r["qid"], {"session_id": r["session_id"], "arms": []})
            info["arms"].append(arm)
    qids = sorted(qid_info)
    logger.info("union valid qids: %d", len(qids))

    # Regime census over the union (from arm B rows where present, else any arm).
    doc_tokens: List[int] = []
    for q in qids:
        for arm in "BCDA":
            hit = [r for r in per_arm[arm] if r["qid"] == q and not r.get("skipped")]
            if hit:
                doc_tokens.append(int(hit[0]["doc_tokens"]))
                break
    regime = {
        "doc_tokens_min": min(doc_tokens),
        "doc_tokens_median": int(statistics.median(doc_tokens)),
        "doc_tokens_max": max(doc_tokens),
        "note": "doc_tokens == kept_history_tokens in the PR#1 archives (history regime, not a 32k tool pool)",
    }

    # Latency estimate: arm-B per-qid latency where available, else arm median.
    b_lat = {r["qid"]: float(r["latency_sec"]) for r in per_arm["B"] if not r.get("skipped") and r.get("latency_sec")}
    b_median = statistics.median(b_lat.values())
    est_sec_per_arm = sum(b_lat.get(q, b_median) for q in qids)
    est_h_total = 3 * est_sec_per_arm * LATENCY_MARGIN / 3600.0
    logger.info("latency estimate: %.1f s/arm -> %.2f GPU-h total (3 arms x%.2f margin)", est_sec_per_arm, est_h_total, LATENCY_MARGIN)

    if est_h_total > SUBSET_RULE_THRESHOLD_H:
        frozen = qids[:SUBSET_RULE_N]
        subset = {"applied": True, "rule": f"estimated {est_h_total:.1f} GPU-h > {SUBSET_RULE_THRESHOLD_H} h -> first {SUBSET_RULE_N} qids sorted by qid"}
    else:
        frozen = qids
        subset = {"applied": False, "rule": f"estimated {est_h_total:.1f} GPU-h <= {SUBSET_RULE_THRESHOLD_H} h -> full union"}
    logger.info("subset decision: %s (frozen n=%d)", subset["rule"], len(frozen))

    out = {
        "description": "R4 task D frozen sample set: union of valid qids across PR#1 merged_{A,B,C,D}.jsonl (sha256-bound). Order: sorted by qid.",
        "source_archives": sources,
        "regime": regime,
        "selection": {
            "pr1_config": "subset_disjoint, eval_ratio=0.1, split_seed=42, max_samples_per_session=4, include_tools=True, greedy, max_new_tokens=128, enable_thinking=False, checkpoint-2678, ratio=4, eager",
            "union_n": len(qids),
            "frozen_n": len(frozen),
            "subset_rule": subset,
            "latency_estimate": {
                "proxy": "arm-B (c2kv free) latency_sec per qid, arm median for qids missing in B",
                "est_h_total_3_arms": round(est_h_total, 2),
                "margin": LATENCY_MARGIN,
            },
        },
        "n": len(frozen),
        "n_sessions": len({qid_info[q]["session_id"] for q in frozen}),
        "qids": frozen,
        "per_qid": {q: {"session_id": qid_info[q]["session_id"], "pr1_arms": sorted(qid_info[q]["arms"])} for q in frozen},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s (n=%d, sessions=%d)", out_path, out["n"], out["n_sessions"])


if __name__ == "__main__":
    main()
