# -*- coding: utf-8 -*-
"""Diff-01 / MaxProb three-valued deferral regression (survey 4.2).

z in {+1 C->W, 0 (C->C | W->W), -1 W->C} regressed with an L2 linear scorer
on Tier-0 (pre-generation) compressed-arm features — <=6 z-scored features,
capacity chosen in inner folds only, session-grouped CV.  Deliverable is the
DEFERRAL CURVE over thresholds with the x-axis in frozen GPU-seconds (the
compressed arm's own generate_sec per row; recovery cost scales with length
per SPEC 3.6), plus the non-predictability gate: R^2 of Tier-0 features
regressing the full-arm span NLL is reported against Var(z) BEFORE any
deferral claim (their section 4.2 protocol adapted).

Oracle rows (r_hat_rel / r_hat_01) are registered family=oracle and reported
as headroom only, never candidates.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t33_labels import build_label_frame, join_arms, load_jsonl  # noqa: E402
from t33_score import session_clusters  # noqa: E402

TIER0 = [
    "s8_n_docs_kept", "s8_dropped_docs", "s8_doc_tokens_sum", "s8_n_ctx",
    "s8_packing_sat", "boundary_max_doc_len", "boundary_mean_doc_len",
    "gzip_ratio_mean", "surprise_max_k", "surprise_mean_k", "n_generated",
]
LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0]


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_c2kv", required=True)
    parser.add_argument("--features_full", required=True)
    parser.add_argument("--battery_full", required=True)
    parser.add_argument("--battery_c2kv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rows_c2kv", required=True, help="RERUN c2kv rows (generate_sec axis)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    label_frame = build_label_frame(
        join_arms(load_jsonl(args.battery_full), load_jsonl(args.battery_c2kv)), manifest)
    lab = {r["qid"]: r for r in label_frame}
    feat_c = {r["qid"]: r for r in load_jsonl(args.features_c2kv)}
    feat_f = {r["qid"]: r for r in load_jsonl(args.features_full)}
    rows = {r["qid"]: r for r in load_jsonl(args.rows_c2kv)}

    qids = [q for q in lab if q in feat_c and q in feat_f]
    z = np.array([lab[q]["z_deferral"] for q in qids], dtype=float)
    clusters_all = session_clusters([lab[q]["session_id"] for q in qids])

    def matrix(qid_list):
        cols = [c for c in TIER0 if all(feat_c[q].get(c) is not None for q in qid_list)]
        X = np.array([[float(feat_c[q][c]) for c in cols] for q in qid_list], dtype=np.float64)
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        return (X - mu) / sd, cols

    X, cols = matrix(qids)

    # ---- non-predictability gate: Tier-0 -> full-arm span NLL R^2 ----
    gnll = np.array([feat_f[q].get("fc_gnll_all") for q in qids], dtype=float)
    gate_ok = None
    if not np.isnan(gnll).any():
        pred = np.column_stack([X, np.ones(len(X))]) @ ridge_fit(np.column_stack([X, np.ones(len(X))]), gnll, 1.0)
        gate = {"r2_tier0_vs_full_gnll": round(r2(gnll, pred), 4),
                "var_z": round(float(z.var()), 4),
                "verdict": "gate-fail (Tier-0 carries no signal about full-arm difficulty)"
                if r2(gnll, pred) < 0.02 else "gate-pass"}

    # ---- deferral curve: session-grouped CV, inner lambda ----
    n = len(qids)
    scores = np.full(n, np.nan)
    folds = [np.isin(clusters_all, u) for u in np.unique(clusters_all)[::5]] and None
    # simple session-grouped 5-fold
    uniq = np.unique(clusters_all)
    rng = np.random.default_rng(20260905)
    rng.shuffle(uniq)
    folds = [np.isin(clusters_all, uniq[i::5]) for i in range(5)]
    for test in folds:
        train = ~test
        best_lam, best_r2 = LAMBDA_GRID[0], -1e9
        inner = np.unique(clusters_all[train])
        rng.shuffle(inner)
        inner_folds = [np.isin(clusters_all[train], inner[i::3]) for i in range(3)]
        for lam in LAMBDA_GRID:
            r2s = []
            for it in inner_folds:
                itr = ~it
                Xtr = np.column_stack([X[train][itr], np.ones(int(itr.sum()))])
                w = ridge_fit(Xtr, z[train][itr], lam)
                pred_i = np.column_stack([X[train][it], np.ones(int(it.sum()))]) @ w
                r2s.append(r2(z[train][it], pred_i))
            m = float(np.mean(r2s)) if r2s else -1e9
            if m > best_r2:
                best_r2, best_lam = m, lam
        Xtr = np.column_stack([X[train], np.ones(int(train.sum()))])
        w = ridge_fit(Xtr, z[train], best_lam)
        scores[test] = np.column_stack([X[test], np.ones(int(test.sum()))]) @ w

    # ---- curve with GPU-sec axis ----
    gsec = np.array([float(rows.get(q, {}).get("generate_sec") or 0.0) for q in qids])
    order = np.argsort(-scores)
    curve = []
    cov_cw = 0
    seen_wc = 0
    for frac in np.linspace(0.02, 1.0, 20):
        k = max(1, int(n * frac))
        idx = order[:k]
        yy = z[idx]
        curve.append({
            "deferred_frac": round(float(frac), 3),
            "gpu_sec": round(float(gsec[idx].sum()), 2),
            "z_pos_covered": int((yy == 1).sum()),          # C->W deferred
            "z_neg_covered": int((yy == -1).sum()),          # W->C deferred
            "z_zero_deferred": int((yy == 0).sum()),         # budget wasted on stable rows
        })

    # ---- oracle headroom (family=oracle) ----
    oracle = {
        "r_hat_01": {"note": "fires iff z==+1", "coverage_of_cw": 1.0, "fire_rate": round(float((z == 1).mean()), 4)},
        "r_hat_conf": {"note": "fires iff compressed arm unparseable (confidence proxy)",
                       "coverage_of_cw": None, "family": "oracle"},
        "family": "oracle — headroom only, never a candidate",
    }

    out = {
        "n": n, "tier0_columns": cols,
        "var_z": round(float(z.var()), 4),
        "nonpredictability_gate": gate,
        "deferral_curve": curve,
        "oracle": oracle,
        "scoring": "session-grouped 5-fold outer, 3-fold inner over lambda grid",
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[t33] diff01 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
