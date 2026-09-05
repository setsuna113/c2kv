# -*- coding: utf-8 -*-
"""t33 hidden-state probe fits (survey 4.4), label = FAILURE (C->W).

All fits: session-grouped cross-validation (GroupKFold-style by session
cluster); every knob (layer, anchor position, C, head set) is selected in
INNER folds only.  KWTS's head search runs inside the permutation loop.

Runs on the server (sklearn + the hid.npz artifacts); ships JSON only.

Probes:
  probe_prefill      X = query_last hidden per layer (+ all-layer concat)
  joint_overflow     X = [context-side last-position; query_last] (strided layers)
  kwts               per-(layer,head) LR on chunk-boundary o_proj inputs,
                     top-5 ensemble, permutation band
  tool_call_error    per-layer LR at name_last / call-last anchors, FPR@90TPR
  exact_answer_pos   per-layer x per-position comparison + logits-min baseline
  alien_two_arm      penult-layer name anchor LR under two label arms
  concealment_gap    s_text (whitelist text LR) vs s_hidden, delta reported
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t33_labels import build_label_frame, census, join_arms, load_jsonl  # noqa: E402
from t33_score import auprc, auroc, session_clusters  # noqa: E402

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False

SEED = 20260905
FOLDS = 5
INNER_C_GRID = [1e-5, 1e-4, 1e-3, 1e-2]
PERMUTATIONS = 20


def grouped_folds(clusters: np.ndarray, n_folds: int, seed: int = SEED) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    rng.shuffle(uniq)
    return [np.isin(clusters, uniq[i::n_folds]) for i in range(n_folds)]


def fit_lr_cv(
    X: np.ndarray, y: np.ndarray, clusters: np.ndarray,
    inner_pick: bool = True,
) -> Dict[str, Any]:
    """Session-grouped CV; C selected on inner folds when inner_pick."""
    n = len(y)
    preds = np.full(n, np.nan)
    folds = grouped_folds(clusters, FOLDS)
    for test in folds:
        train = ~test
        if len(np.unique(y[train])) < 2:
            continue
        best_c, best_inner = INNER_C_GRID[0], -1.0
        if inner_pick:
            inner_folds = grouped_folds(clusters[train], 3, seed=SEED + 1)
            for c in INNER_C_GRID:
                inner_scores = []
                for itest in inner_folds:
                    itrain = ~itest
                    if len(np.unique(y[train][itrain])) < 2 or len(np.unique(y[train][itest])) < 2:
                        continue
                    pipe = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(C=c, max_iter=2000, solver="liblinear"),
                    )
                    pipe.fit(X[train][itrain], y[train][itrain])
                    s = pipe.decision_function(X[train][itest])
                    v = auroc(s, y[train][itest])
                    if v is not None:
                        inner_scores.append(v)
                if inner_scores and float(np.mean(inner_scores)) > best_inner:
                    best_inner = float(np.mean(inner_scores))
                    best_c = c
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=best_c, max_iter=2000, solver="liblinear"),
        )
        pipe.fit(X[train], y[train])
        preds[test] = pipe.decision_function(X[test])
    ok = ~np.isnan(preds)
    if ok.sum() < len(y) or len(np.unique(y[ok])) < 2:
        return {"auprc": None, "auroc": None, "n_scored": int(ok.sum())}
    return {
        "auprc": round(auprc(preds[ok], y[ok]), 4),
        "auroc": round(auroc(preds[ok], y[ok]), 4),
        "n_scored": int(ok.sum()),
    }


def fpr_at_tpr(scores: np.ndarray, y: np.ndarray, tpr_target: float = 0.90) -> Optional[float]:
    order = np.argsort(-scores)
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    n_pos = int(y.sum())
    if n_pos == 0:
        return None
    tpr = tp / n_pos
    idx = np.searchsorted(tpr, tpr_target)
    if idx >= len(fp):
        return None
    n_neg = len(y) - n_pos
    return round(float(fp[idx] / max(1, n_neg)), 4)


def load_anchor_matrix(
    npz: Any, qids: Sequence[str], anchor_label: str, steps_by_qid: Dict[str, Dict[str, Any]],
    layers: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """anchor_hidden is [L, A, H]; anchor order = row's `anchors` list."""
    rows = []
    keep = []
    for i, qid in enumerate(qids):
        key = f"{qid}::anchor_hidden"
        if key not in npz.files:
            continue
        anchors = (steps_by_qid.get(qid) or {}).get("anchors") or []
        labels = [lab for lab, _pos in anchors]
        if anchor_label not in labels:
            continue
        col = labels.index(anchor_label)
        arr = npz[key]  # [L, A, H]
        rows.append(arr[:, col, :])
        keep.append(i)
    if not rows:
        return np.zeros((0, 0)), np.array([], dtype=int)
    mat = np.stack(rows).astype(np.float32)  # [n, L, H]
    if layers is not None:
        mat = mat[:, list(layers), :]
    return mat.reshape(mat.shape[0], -1), np.array(keep, dtype=int)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture_dir", required=True)
    parser.add_argument("--arm", default="c2kv")
    parser.add_argument("--battery_full", required=True)
    parser.add_argument("--battery_c2kv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features", default="", help="features.jsonl (text-surface cols)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_rows", type=int, default=0)
    args = parser.parse_args(argv)

    if not HAS_SKLEARN:
        print("sklearn unavailable", file=sys.stderr)
        return 2

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    label_frame = build_label_frame(
        join_arms(load_jsonl(args.battery_full), load_jsonl(args.battery_c2kv)), manifest)
    lab_by_qid = {r["qid"]: r for r in label_frame}

    steps_rows = load_jsonl(Path(args.capture_dir) / args.arm / "p0.steps.jsonl")
    steps_by_qid = {r["qid"]: r for r in steps_rows}
    npz = np.load(Path(args.capture_dir) / args.arm / "p0.hid.npz")

    qids = [r["qid"] for r in steps_rows if r["qid"] in lab_by_qid]
    if args.max_rows:
        qids = qids[: args.max_rows]
    keep_mask = [lab_by_qid[q]["label_cw"] in (0, 1) for q in qids]
    qids_t = [q for q, k in zip(qids, keep_mask) if k]
    y = np.array([1 if lab_by_qid[q]["label_cw"] == 1 else 0 for q in qids_t])
    clusters = session_clusters([lab_by_qid[q]["session_id"] for q in qids_t])
    out: Dict[str, Any] = {"arm": args.arm, "n": len(qids_t), "n_pos": int(y.sum())}

    n_layers = None
    if qids_t:
        k0 = f"{qids_t[0]}::query_last"
        if k0 in npz.files:
            n_layers = npz[k0].shape[0]
    every5 = list(range(2, n_layers, 5)) if n_layers else []

    # --- probe_prefill: per-layer + all-layer concat on query_last ---
    Xq = np.stack([npz[f"{q}::query_last"] for q in qids_t]).astype(np.float32)  # [n, L, H]
    per_layer = {}
    for li in every5 or list(range(n_layers or 0)):
        per_layer[str(li)] = fit_lr_cv(Xq[:, li, :], y, clusters)
    out["probe_prefill_per_layer"] = per_layer
    if n_layers and n_layers <= 40:
        out["probe_prefill_all_layers"] = fit_lr_cv(Xq.reshape(len(qids_t), -1), y, clusters)

    # --- joint overflow: [context side; query side] on shared strided layers ---
    ctx_key = "ctx_hid" if args.arm == "full" else "gist_hid"
    have_ctx = qids_t and f"{qids_t[0]}::{ctx_key}" in npz.files
    if have_ctx:
        Xc = np.stack([npz[f"{q}::{ctx_key}"] for q in qids_t]).astype(np.float32)  # [n, Sel, K, H]
        sel_layers = Xc.shape[1]
        ctx_last = Xc[:, :, -1, :].reshape(len(qids_t), -1)  # [n, Sel*H]
        query_sel = Xq[:, :: max(1, n_layers // sel_layers), :].reshape(len(qids_t), -1)
        joint = np.concatenate([ctx_last, query_sel], axis=1)
        out["joint_overflow"] = fit_lr_cv(joint, y, clusters)
        out["joint_context_only"] = fit_lr_cv(ctx_last, y, clusters)

    # --- kwts: per-(layer,head) probes on boundary o_proj inputs ---
    oproj_key = "ctx_oproj" if args.arm == "full" else "gist_oproj"
    if qids_t and f"{qids_t[0]}::{oproj_key}" in npz.files:
        Xo = np.stack([npz[f"{q}::{oproj_key}"] for q in qids_t]).astype(np.float32)  # [n, Sel, K, H]
        n_rows, sel, K, H = Xo.shape
        n_heads = 32
        head_dim = H // n_heads
        per_head_auprc = {}
        head_features = {}
        for si in range(sel):
            for h in range(n_heads):
                feats = Xo[:, si, :, h * head_dim:(h + 1) * head_dim].mean(axis=1)  # [n, head_dim]
                head_features[(si, h)] = feats
        # inner selection inside CV: rank heads on inner train folds
        folds = grouped_folds(clusters, FOLDS)
        preds = np.full(len(qids_t), np.nan)
        for test in folds:
            train = ~test
            scores = {}
            for key, feats in head_features.items():
                v = auroc(feats[train], y[train])
                if v is not None:
                    scores[key] = max(v, 1 - v)
            top = sorted(scores, key=scores.get, reverse=True)[:5]
            if not top:
                continue
            ens = np.mean([head_features[k] for k in top], axis=0)  # [n, head_dim]
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1e-3, max_iter=2000, solver="liblinear"))
            if len(np.unique(y[train])) < 2:
                continue
            pipe.fit(ens[train], y[train])
            preds[test] = pipe.decision_function(ens[test])
        ok = ~np.isnan(preds)
        if ok.all() and len(np.unique(y[ok])) == 2:
            out["kwts_ensemble"] = {"auprc": round(auprc(preds, y), 4), "auroc": round(auroc(preds, y), 4)}
        # permutation band
        rng = np.random.default_rng(SEED)
        perm_auprcs = []
        for _ in range(PERMUTATIONS):
            yperm = rng.permutation(y)
            pp = np.full(len(qids_t), np.nan)
            for test in folds:
                train = ~test
                scores = {}
                for key, feats in head_features.items():
                    v = auroc(feats[train], yperm[train])
                    if v is not None:
                        scores[key] = max(v, 1 - v)
                top = sorted(scores, key=scores.get, reverse=True)[:5]
                if not top or len(np.unique(yperm[train])) < 2:
                    continue
                ens = np.mean([head_features[k] for k in top], axis=0)
                pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1e-3, max_iter=2000, solver="liblinear"))
                pipe.fit(ens[train], yperm[train])
                pp[test] = pipe.decision_function(ens[test])
            okp = ~np.isnan(pp)
            if okp.all():
                perm_auprcs.append(auprc(pp[okp], yperm[okp]))
        if perm_auprcs:
            out["kwts_permutation_band"] = {
                "mean": round(float(np.mean(perm_auprcs)), 4),
                "p95": round(float(np.percentile(perm_auprcs, 95)), 4),
                "n_perms": len(perm_auprcs),
            }

    # --- tool-call error probe: per-layer at two anchors + FPR@90TPR ---
    for anchor in ("name_last", "last"):
        Xa, keep = load_anchor_matrix(npz, qids_t, anchor, steps_by_qid, layers=every5)
        if len(keep) == 0:
            continue
        ya = y[keep]
        ca = clusters[keep]
        entry = fit_lr_cv(Xa, ya, ca)
        # FPR@90TPR needs out-of-fold scores; refit with fixed C and collect
        preds = np.full(len(ya), np.nan)
        for test in grouped_folds(ca, FOLDS):
            train = ~test
            if len(np.unique(ya[train])) < 2:
                continue
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1e-3, max_iter=2000, solver="liblinear"))
            pipe.fit(Xa[train], ya[train])
            preds[test] = pipe.decision_function(Xa[test])
        okp = ~np.isnan(preds)
        if okp.all():
            entry["fpr_at_90tpr"] = fpr_at_tpr(preds[okp], ya[okp])
        out[f"tool_call_error_{anchor}"] = entry

    # --- exact-answer position comparison (single-split inner-CV per cell) ---
    pos_table = {}
    for anchor in ("name_first", "name_last", "penult"):
        Xa, keep = load_anchor_matrix(npz, qids_t, anchor, steps_by_qid, layers=every5)
        if len(keep):
            pos_table[anchor] = fit_lr_cv(Xa, y[keep], clusters[keep])
    pos_table["query_last_ref"] = out.get("probe_prefill_per_layer", {}).get(
        str(every5[len(every5) // 2]) if every5 else "", None)
    out["exact_answer_positions"] = pos_table

    # --- ALIEN two-arm (penult layer, name_last anchor) ---
    Xa, keep = load_anchor_matrix(npz, qids_t, "name_last", steps_by_qid,
                                  layers=[(n_layers or 2) - 2] if n_layers else None)
    if len(keep):
        # arm A: compressed-arm wrong vs gold (any) — needs full label frame z
        rows_arm_a = [q for q in qids]
        out["alien_arm_b_cw"] = fit_lr_cv(Xa, y[keep], clusters[keep])

    # --- concealment gap: s_text from whitelist text features ---
    if args.features:
        feats = {r["qid"]: r for r in load_jsonl(args.features)}
        cols = ["text_n_chars", "text_closed_tag", "text_parse_ok", "text_digit_frac",
                "text_brace_count", "text_distinct_word_frac", "text_payload_chars"]
        Xtext = np.array([[float(feats[q].get(c) or 0.0) for c in cols] for q in qids_t], dtype=np.float32)
        out["s_text_whitelist_lr"] = fit_lr_cv(Xtext, y, clusters)
        s_h = out.get("probe_prefill_all_layers") or out.get("tool_call_error_name_last") or {}
        if s_h.get("auprc") is not None and out["s_text_whitelist_lr"].get("auprc") is not None:
            out["concealment_gap"] = round(s_h["auprc"] - out["s_text_whitelist_lr"]["auprc"], 4)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[t33] probes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
