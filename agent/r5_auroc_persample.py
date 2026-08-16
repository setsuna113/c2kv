#!/usr/bin/env python
"""S7.2: recover the per-sample data behind the round-3 ΔAUROC n=390 result.

Pure-CPU. Imports agent/analyze_trigger_signals.py (the evidentiary recompute
script; it has an `if __name__ == "__main__"` guard, so plain `import` is side-
effect free) and reuses its loaders, labeler, signal assembly, and paired
session-cluster bootstrap. Nothing here re-derives scoring logic.

Anchor (r3 aggregation product): .foreman/staging_s7/trigger_signals.json
  - best candidate = S2.norm_logp_gap, AUROC 0.7089
  - best control   = S0.d_forced_tool_name_match, AUROC 0.6649
  - ΔAUROC point 0.0439, CI95 [-0.0613, 0.1425], n_valid_reps=20000
  - definition: best(S1..S3) − best(S0), paired session-cluster bootstrap,
    bootstrap reps=20000, seed=0, clusters=session_id.

Outputs (writes only these):
  results/r5/provenance/auroc_n390_per_sample.json

No git operations.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import analyze_trigger_signals as T

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / ".foreman" / "staging_s7"
OUT_DIR = ROOT / "results" / "r5" / "provenance"
OUT_PATH = OUT_DIR / "auroc_n390_per_sample.json"

BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 0
TOL = 0.0005

# Anchor values from .foreman/staging_s7/trigger_signals.json (r3 aggregation).
ANCHOR = {
    "n": 390,
    "n_sessions": 166,
    "best_candidate": {"name": "S2.norm_logp_gap", "auroc": 0.7089},
    "best_control": {"name": "S0.d_forced_tool_name_match", "auroc": 0.6649},
    "delta_auroc": {"point": 0.0439, "ci95": [-0.0613, 0.1425], "n_valid_reps": 20000},
    "pr_claimed_n": 390,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _f(value: Any) -> Optional[float]:
    """nan -> None for JSON; finite floats pass through."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _close(value: Optional[float], target: float) -> bool:
    return value is not None and abs(value - target) <= TOL


def main() -> int:
    logp_rows = T._load_logp_rows(str(STAGING / "s4_logp_clean.jsonl"))
    arm_rows = {
        arm: T._load_rows(str(STAGING / f"merged_{arm}.jsonl"))
        for arm in ("A", "B", "C", "D")
    }

    # Same funnel as T.main(): A∩B qids with clean logp coverage.
    a_intersect_b = sorted(set(arm_rows["A"]) & set(arm_rows["B"]))
    eval_qids = [qid for qid in a_intersect_b if T._logp_clean(logp_rows.get(qid))]
    labels = np.asarray(
        [T._paired_action_change_label(arm_rows["A"][qid], arm_rows["B"][qid]) for qid in eval_qids],
        dtype=int,
    )
    clusters = np.asarray(
        [
            (logp_rows[qid].get("session_id") or arm_rows["B"][qid].get("session_id") or qid.rsplit(":", 1)[0])
            for qid in eval_qids
        ]
    )
    draws, unique_sessions = T._cluster_index_draws(clusters, BOOTSTRAP_REPS, BOOTSTRAP_SEED)

    session_tools: Optional[Dict[str, set]] = None  # anchor ran without --dataset_path
    signals, notes = T._build_signals(eval_qids, logp_rows, arm_rows, session_tools)
    scored = T._scored_signals(signals, labels, draws)

    def _valid(family: str) -> Dict[str, Dict[str, Any]]:
        return {
            name: entry
            for name, entry in scored.items()
            if entry["family"] == family and entry.get("auroc") is not None
        }

    s3 = _valid("candidate_s3")
    s0 = _valid("control_s0")
    best_s3_name = max(s3, key=lambda name: s3[name]["auroc_raw"]) if s3 else None
    best_s0_name = max(s0, key=lambda name: s0[name]["auroc_raw"]) if s0 else None

    candidates: Dict[str, Dict[str, Any]] = {
        name: entry
        for name, entry in scored.items()
        if entry["family"] == "candidate" and entry.get("auroc") is not None
    }
    if best_s3_name is not None:
        candidates[f"S3_best({best_s3_name})"] = s3[best_s3_name]
    best_cand_name = max(candidates, key=lambda name: candidates[name]["auroc_raw"]) if candidates else None

    cand_clean = {
        name: T._clean_signal(entry, labels, draws)
        for name, entry in scored.items()
        if entry["family"] in {"candidate", "candidate_s3"} and entry.get("auroc") is not None
    }
    s0_clean = {
        name: T._clean_signal(entry, labels, draws)
        for name, entry in s0.items()
        if entry.get("auroc") is not None
    }
    delta_point, delta_ci, delta_reps = T._delta_best_bootstrap(cand_clean, s0_clean)

    # ---- per-sample table for the two best signals (aligned with eval_qids) ----
    s2_spec = signals["S2.norm_logp_gap"]
    s0d_spec = signals["S0.d_forced_tool_name_match"]
    s2_vals = np.asarray(s2_spec["values"], dtype=float) * s2_spec["orient"]
    s0d_vals = np.asarray(s0d_spec["values"], dtype=float) * s0d_spec["orient"]
    per_sample: List[Dict[str, Any]] = []
    for pos, qid in enumerate(eval_qids):
        per_sample.append(
            {
                "sample_id": qid,
                "session_id": str(clusters[pos]),
                "label": int(labels[pos]),
                "s2_norm_logp_gap": _f(s2_vals[pos]),
                "s0d_forced_tool_name_match": _f(s0d_vals[pos]),
            }
        )

    n = int(len(eval_qids))
    n_pos = int(labels.sum())
    s2_auroc_raw = candidates[best_cand_name]["auroc_raw"] if best_cand_name else float("nan")
    s0d_auroc_raw = s0[best_s0_name]["auroc_raw"] if best_s0_name else float("nan")
    delta_raw = float(delta_point)

    # ---- assertions (tolerance ±0.0005 for rounding differences) ----
    checks = {
        "n_matches": n == ANCHOR["n"],
        "best_candidate_name": best_cand_name == ANCHOR["best_candidate"]["name"],
        "best_candidate_auroc": _close(s2_auroc_raw, ANCHOR["best_candidate"]["auroc"]),
        "best_control_name": best_s0_name == ANCHOR["best_control"]["name"],
        "best_control_auroc": _close(s0d_auroc_raw, ANCHOR["best_control"]["auroc"]),
        "delta_point": _close(delta_raw, ANCHOR["delta_auroc"]["point"]),
        "delta_ci_lo": bool(delta_ci) and _close(delta_ci[0], ANCHOR["delta_auroc"]["ci95"][0]),
        "delta_ci_hi": bool(delta_ci) and _close(delta_ci[1], ANCHOR["delta_auroc"]["ci95"][1]),
        "delta_n_valid_reps": delta_reps == ANCHOR["delta_auroc"]["n_valid_reps"],
        "n_sessions": int(len(unique_sessions)) == ANCHOR["n_sessions"],
    }
    match = all(checks.values())

    delta_point_r = None if math.isnan(delta_raw) else round(delta_raw, 4)
    delta_ci_r = [round(delta_ci[0], 4), round(delta_ci[1], 4)] if delta_ci else None
    anchors = {
        "source": ".foreman/staging_s7/trigger_signals.json",
        "best_candidate": ANCHOR["best_candidate"],
        "best_control": ANCHOR["best_control"],
        "delta_auroc": ANCHOR["delta_auroc"],
        "bootstrap": {"reps": 20000, "seed": 0, "cluster": "session_id"},
    }
    recomputed = {
        "n": n,
        "n_pos": n_pos,
        "n_sessions": int(len(unique_sessions)),
        "best_candidate": best_cand_name,
        "best_control": best_s0_name,
        "s2_auroc": None if math.isnan(s2_auroc_raw) else round(s2_auroc_raw, 4),
        "s2_auroc_raw": None if math.isnan(s2_auroc_raw) else s2_auroc_raw,
        "s0d_auroc": None if math.isnan(s0d_auroc_raw) else round(s0d_auroc_raw, 4),
        "s0d_auroc_raw": None if math.isnan(s0d_auroc_raw) else s0d_auroc_raw,
        "delta_point": delta_point_r,
        "delta_point_raw": None if math.isnan(delta_raw) else delta_raw,
        "delta_ci95": delta_ci_r,
        "delta_n_valid_reps": delta_reps,
        "match": match,
        "checks": checks,
        "diffs": {
            "s2_auroc": None if math.isnan(s2_auroc_raw) else round(s2_auroc_raw - ANCHOR["best_candidate"]["auroc"], 6),
            "s0d_auroc": None if math.isnan(s0d_auroc_raw) else round(s0d_auroc_raw - ANCHOR["best_control"]["auroc"], 6),
            "delta_point": None if math.isnan(delta_raw) else round(delta_raw - ANCHOR["delta_auroc"]["point"], 6),
        },
    }

    report = {
        "task": "S7.2 per-sample data recovery: ΔAUROC n=390",
        "inputs": {
            "s4_logp_clean.jsonl": _sha256(STAGING / "s4_logp_clean.jsonl"),
            "merged_A.jsonl": _sha256(STAGING / "merged_A.jsonl"),
            "merged_B.jsonl": _sha256(STAGING / "merged_B.jsonl"),
            "merged_C.jsonl": _sha256(STAGING / "merged_C.jsonl"),
            "merged_D.jsonl": _sha256(STAGING / "merged_D.jsonl"),
            "trigger_signals.json": _sha256(STAGING / "trigger_signals.json"),
        },
        "anchors": anchors,
        "recomputed": recomputed,
        "per_sample_n": len(per_sample),
        "per_sample": per_sample,
        "notes": notes,
        "method": {
            "import": "plain `import analyze_trigger_signals`; module has a __main__ guard, no importlib workaround needed",
            "loader": "T._load_logp_rows / T._load_rows",
            "labels": "T._paired_action_change_label (A vs B)",
            "signals": "T._build_signals(eval_qids, logp_rows, arm_rows, session_tools=None)",
            "bootstrap": "T._cluster_index_draws(reps=20000, seed=0) + T._delta_best_bootstrap",
            "per_sample_columns": "sample_id=qid, session_id, label, s2_norm_logp_gap (oriented, orient=1.0), s0d_forced_tool_name_match (oriented, orient=1.0); 值为 null 表示该 qid 该信号未定义（如 D 行缺失），与逐信号 nan-drop 口径一致",
        },
        "zh_note": [
            "本文件是 S7.2 逐样本数据恢复产物：用入库的 analyze_trigger_signals.py 全部复用其 loader、打标签与打分逻辑，在本地纯 CPU 重建 r3 聚合锚点。",
            "ΔAUROC 定义：best(S1..S3) − best(S0)，paired session-cluster bootstrap（按 session_id 整簇有放回重采样，两边共用同一组重采样索引，per-rep argmax 允许选不同信号）。",
            "该定义与锚点出处：.foreman/staging_s7/trigger_signals.json（r3 聚合产物），其 bootstrap.reps=20000、seed=0、cluster=session_id。",
            "锚点：best candidate = S2.norm_logp_gap（AUROC 0.7089）、best control = S0.d_forced_tool_name_match（AUROC 0.6649）、Δ point=0.0439、CI95=[−0.0613, 0.1425]、n_valid_reps=20000、n=390。",
            "PR 声称 n=390；本次重建实际 n 为 {n}（若与 390 不一致，以此处实际值为准）。".format(n=n),
            "逐样本表共 {m} 行，与 eval_qids（A∩B 且 logp 干净的 qid，字典序）逐位对齐；s2_norm_logp_gap 与 s0d_forced_tool_name_match 均为已按 orient 定向后的值（两者 orient 均为 +1.0，故与原始值一致）。".format(m=len(per_sample)),
            "四舍五入差异说明：锚点中各 auroc/CI 为 4 位小数舍入值，Δ point 由未舍入的 auroc_raw 相减后舍入得到，因此 0.7089−0.6649=0.0440 与 point=0.0439 的差异属正常舍入，非数值矛盾（容差 ±0.0005）。",
            "本次运行未做任何 git 操作；analyze_trigger_signals.py 作为物证未做任何修改。",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"n (per-sample rows)          : {n}")
    print(f"n_pos / n_sessions           : {n_pos} / {len(unique_sessions)}")
    print(f"best candidate               : {best_cand_name}  AUROC={recomputed['s2_auroc']}")
    print(f"best control                 : {best_s0_name}  AUROC={recomputed['s0d_auroc']}")
    print(f"ΔAUROC point                 : {delta_point_r}")
    print(f"ΔAUROC CI95                  : {delta_ci_r}")
    print(f"ΔAUROC valid reps            : {delta_reps}")
    print(f"anchor match                 : {match}")
    for name, ok in checks.items():
        print(f"  - {name:26s}: {'PASS' if ok else 'FAIL'}")
    print(f"wrote {OUT_PATH}")

    if not match:
        sys.exit(1)
    return 0


if __name__ == "__main__":
    main()
