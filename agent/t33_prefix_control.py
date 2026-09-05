# -*- coding: utf-8 -*-
"""4.1 adjudication control: incremental value of arm-invariant / S0-undefined
prefix features over the cheap baseline (survey 4.1 + audit 2026-09-05).

The 4.1 prefix family cannot be adjudicated by the S0 twin:
  * s8 / boundary / gzip / surprise are IDENTICAL in both arms by
    construction (they read the frozen doc sidecar, not the compressed
    prefix), so Delta-vs-S0 == 0 is an identity, not evidence;
  * gist/sat features are undefined on the full arm (no gist pass there).

Adjudication here: session-grouped CV with a cheap baseline (length +
parse-failure indicator) versus baseline + feature.  Report Delta-AUPRC on
the SAME folds, a session-clustered bootstrap CI, and a label-permutation
p-value.  Features that add nothing over the cheap baseline are dominated
and die on data, not on an undefined control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t33_labels import build_label_frame, join_arms, load_jsonl  # noqa: E402
from t33_score import auprc, session_clusters  # noqa: E402

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False


def grouped_folds(clusters: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    rng.shuffle(uniq)
    return [np.isin(clusters, uniq[i::n_folds]) for i in range(n_folds)]


def oof_lr(X: np.ndarray, y: np.ndarray, clusters: np.ndarray, folds) -> np.ndarray:
    preds = np.full(len(y), np.nan)
    for test in folds:
        train = ~test
        if len(np.unique(y[train])) < 2:
            continue
        mu, sd = X[train].mean(axis=0), X[train].std(axis=0) + 1e-9
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=1e-3, max_iter=2000, solver="liblinear"))
        pipe.fit((X[train] - mu) / sd, y[train])
        preds[test] = pipe.decision_function((X[test] - mu) / sd)
    return preds


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_c2kv", required=True)
    parser.add_argument("--battery_full", required=True)
    parser.add_argument("--battery_c2kv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if not HAS_SKLEARN:
        print("sklearn unavailable", file=sys.stderr)
        return 2

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    label_frame = build_label_frame(
        join_arms(load_jsonl(args.battery_full), load_jsonl(args.battery_c2kv)), manifest)
    lab = {r["qid"]: r for r in label_frame}
    feats = {r["qid"]: r for r in load_jsonl(args.features_c2kv)}

    qids = [q for q, r in lab.items()
            if r["label_cw"] in (0, 1) and q in feats]
    y = np.array([1 if lab[q]["label_cw"] == 1 else 0 for q in qids])
    clusters = session_clusters([lab[q]["session_id"] for q in qids])

    n_len = np.array([float(feats[q].get("n_generated") or 0) for q in qids])
    pf = np.array([1.0 if lab[q].get("parse_fail_fire") else 0.0 for q in qids])
    base_cols = np.column_stack([n_len, pf])

    # the arm-invariant / S0-undefined families under adjudication
    FAMILIES = {
        "s8": ["s8_n_docs_kept", "s8_dropped_docs", "s8_packing_sat",
               "s8_doc_tokens_sum", "s8_kept_frac"],
        "boundary": ["boundary_max_doc_len", "boundary_mean_doc_len",
                     "boundary_longest_doc_pos_frac"],
        "gzip": ["gzip_ratio_mean", "gzip_ratio_min", "gzip_ratio_max"],
        "surprise": ["surprise_max_k", "surprise_mean_k", "surprise_hit_rate_mean"],
        "gist": ["gist_gists_per_doc_mean", "gist_gists_per_doc_max"],
        "sat": ["sat_hoyer_k", "sat_spec_ent_k", "sat_norm_k", "sat_kurt_k"],
        "rung0": ["rung0_dropped_any"],
    }

    SEED = 20260905
    folds = grouped_folds(clusters, 5, SEED)
    base_oof = oof_lr(base_cols, y, clusters, folds)
    base_ap = auprc(base_oof, y)

    rng = np.random.default_rng(SEED + 9)
    PERMS = 200

    out: Dict[str, Any] = {
        "n": len(qids), "n_pos": int(y.sum()),
        "baseline_cols": ["n_generated", "parse_fail_fire"],
        "baseline_auprc": round(base_ap, 4),
        "note": ("arm-invariant features have S0 == identity-zero by "
                 "construction and gist/sat have no S0 (undefined on the "
                 "full arm); adjudication is increment over the cheap "
                 "baseline on the SAME session-grouped folds"),
        "entries": [],
    }

    for fam, cols in FAMILIES.items():
        avail = [c for c in cols
                 if sum(1 for q in qids if feats[q].get(c) is not None) >= 0.8 * len(qids)]
        if not avail:
            out["entries"].append({"family": fam, "verdict": "n/a (no complete-case columns)"})
            continue
        # per-family matrix: impute family medians for the <20% missing
        fam_cols = []
        for c in avail:
            v = np.array([float(feats[q][c]) if feats[q].get(c) is not None else np.nan
                          for q in qids])
            med = np.nanmedian(v)
            v = np.where(np.isnan(v), med, v)
            fam_cols.append(v)
        Xf = np.column_stack(fam_cols)
        plus = np.column_stack([base_cols, Xf])
        plus_oof = oof_lr(plus, y, clusters, folds)
        plus_ap = auprc(plus_oof, y)
        delta = plus_ap - base_ap

        # session-clustered bootstrap CI of the delta (paired resamples)
        uniq = np.unique(clusters)
        deltas = []
        for _ in range(1000):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([np.where(clusters == c)[0] for c in pick])
            if len(np.unique(y[idx])) < 2:
                continue
            deltas.append(auprc(plus_oof[idx], y[idx]) - auprc(base_oof[idx], y[idx]))
        ci_lo = float(np.percentile(deltas, 2.5)) if deltas else None
        ci_hi = float(np.percentile(deltas, 97.5)) if deltas else None

        # label permutation p (same folds, same models, permuted labels)
        perm_deltas = []
        for _ in range(PERMS):
            yp = rng.permutation(y)
            b = oof_lr(base_cols, yp, clusters, folds)
            p = oof_lr(plus, yp, clusters, folds)
            perm_deltas.append(auprc(p, yp) - auprc(b, yp))
        p_gt = float(np.mean([d >= delta for d in perm_deltas])) if perm_deltas else None

        verdict = "LIVE-eligible" if (ci_lo is not None and ci_lo > 0 and p_gt is not None
                                      and p_gt < 0.05) else "dominated-by-cheap-baseline"
        out["entries"].append({
            "family": fam, "columns": avail,
            "baseline_auprc": round(base_ap, 4),
            "plus_feature_auprc": round(plus_ap, 4),
            "delta_auprc": round(delta, 4),
            "delta_ci": [round(ci_lo, 4) if ci_lo is not None else None,
                         round(ci_hi, 4) if ci_hi is not None else None],
            "perm_p": round(p_gt, 4) if p_gt is not None else None,
            "verdict": verdict,
        })

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[t33] prefix control -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
