# -*- coding: utf-8 -*-
"""t33 hidden-state probe fits (survey 4.4), label = FAILURE (C->W).

All fits: session-grouped cross-validation; every knob (layer, anchor
position, C, head set) is selected in INNER folds only.  The previous build
selected the reported layer on OUTER results, scored AUPRC and FPR@90TPR
from two differently-trained models, and its KWTS loop stored only the last
head per strided layer (an indentation bug) while feeding activation
matrices to a 1-D score metric — all fixed here.

Probes:
  probe_prefill      X = query_last hidden per layer; layer+C selected in
                     inner folds; per-layer OOF scores shipped for the
                     same-layer twin comparison against the full arm
  joint_overflow     X = [context-side boundary; query side] (strided layers)
  kwts               per-(layer,head) LR probes, top-5 ensemble by inner
                     AUROC, permutation band with the search re-run per perm
  tool_call_error    per-layer LR at name_last / call-last anchors; FPR@90TPR
                     from the SAME OOF scores; parse-failure-excluded variant
  exact_answer_pos   per-position comparison; cheap Logits-min baselines
                     reported alongside (probe residual, not replacement)
  alien_two_arm      penult-layer name anchor: arm-B C->W labels vs arm-A
                     wrong-any labels (c2kv wrong vs gold on the 900 frame),
                     both scored on C->W; the gap is the reported quantity
  concealment_gap    s_text (whitelist text LR) vs s_hidden, delta reported
"""

from __future__ import annotations

import argparse
import io
import json
import math
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
# prereg (tool-call error probe) fixes C=1 verbatim; the rest of the grid is
# the original search space
INNER_C_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1.0]
PERMUTATIONS = 20
KWTS_HEAD_C = 1e-3


_FOLD_MODE: Dict[str, Any] = {"mode": "default", "signatures": {}}


def grouped_folds(clusters: np.ndarray, n_folds: int, seed: int = SEED) -> List[np.ndarray]:
    if _FOLD_MODE["mode"] == "toolset_disjoint":
        return toolset_disjoint_folds(clusters, _FOLD_MODE["signatures"], n_folds, seed)
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    rng.shuffle(uniq)
    return [np.isin(clusters, uniq[i::n_folds]) for i in range(n_folds)]


def toolset_disjoint_folds(clusters: np.ndarray, signatures: Dict[str, int],
                           n_folds: int, seed: int = SEED) -> List[np.ndarray]:
    """toolset_disjoint split (prereg): sessions sharing a tool-set signature
    land in the SAME fold, so no tool set spans train/test.  signatures maps
    session-label -> signature id (built from the capture's per-session
    candidate pools)."""
    rng = np.random.default_rng(seed)
    sig_groups: Dict[int, List[int]] = {}
    for c in np.unique(clusters):
        sig_groups.setdefault(signatures.get(int(c), -1), []).append(int(c))
    groups = list(sig_groups.values())
    rng.shuffle(groups)
    folds = []
    for i in range(n_folds):
        members = [c for g in groups[i::n_folds] for c in g]
        folds.append(np.isin(clusters, members) if members else np.zeros(len(clusters), dtype=bool))
    return folds


def _lr(c: float, n_features: int = 0):
    if n_features and n_features > 8000:
        # liblinear crawls on the ~92k-dim all-layer concat when it fails to
        # converge (2000 iters x dozens of fits stalled the full arm for
        # hours); lbfgs handles wide inputs far better
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c, max_iter=500, solver="lbfgs"),
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, max_iter=2000, solver="liblinear"),
    )


def _pick_c_inner(X: np.ndarray, y: np.ndarray, clusters: np.ndarray) -> float:
    best_c, best = INNER_C_GRID[0], -1.0
    grid = INNER_C_GRID if X.shape[1] <= 8000 else INNER_C_GRID[:2]
    inner_folds = grouped_folds(clusters, 3, seed=SEED + 1)
    for c in grid:
        scores = []
        for itest in inner_folds:
            itrain = ~itest
            if len(np.unique(y[itrain])) < 2 or len(np.unique(y[itest])) < 2:
                continue
            pipe = _lr(c, X.shape[1])
            pipe.fit(X[itrain], y[itrain])
            v = auroc(pipe.decision_function(X[itest]), y[itest])
            if v is not None:
                scores.append(v)
        if scores and float(np.mean(scores)) > best:
            best = float(np.mean(scores))
            best_c = c
    return best_c


def fit_lr_cv_detailed(
    X: np.ndarray, y: np.ndarray, clusters: np.ndarray,
    inner_pick: bool = True,
) -> Dict[str, Any]:
    """Session-grouped CV.  Returns metrics AND the out-of-fold scores (plus
    the per-fold chosen C) so downstream consumers (FPR@90TPR, twin
    comparisons, residual-on-baseline) use the SAME model, not a refit."""
    n = len(y)
    preds = np.full(n, np.nan)
    chosen_c: List[float] = []
    folds = grouped_folds(clusters, FOLDS)
    for test in folds:
        train = ~test
        if len(np.unique(y[train])) < 2:
            continue
        c = _pick_c_inner(X[train], y[train], clusters[train]) if inner_pick else INNER_C_GRID[0]
        chosen_c.append(c)
        pipe = _lr(c, X.shape[1])
        pipe.fit(X[train], y[train])
        preds[test] = pipe.decision_function(X[test])
    ok = ~np.isnan(preds)
    out: Dict[str, Any] = {
        "auprc": None, "auroc": None, "n_scored": int(ok.sum()),
        "chosen_c": chosen_c, "_oof": preds, "_ok": ok,
    }
    if ok.sum() >= len(y) * 0.5 and len(np.unique(y[ok])) == 2:
        out["auprc"] = round(auprc(preds[ok], y[ok]), 4)
        out["auroc"] = round(auroc(preds[ok], y[ok]), 4)
    return out


def fit_lr_cv(X: np.ndarray, y: np.ndarray, clusters: np.ndarray,
              inner_pick: bool = True) -> Dict[str, Any]:
    d = fit_lr_cv_detailed(X, y, clusters, inner_pick)
    d.pop("_oof", None)
    d.pop("_ok", None)
    return d


def fit_layer_select_cv(
    X_layers: np.ndarray, y: np.ndarray, clusters: np.ndarray,
    c_grid: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Layer AND C selected on inner folds only; outer OOF scores for the
    selected (layer, C) per fold.  X_layers: [n, L, H]."""
    n, L, _H = X_layers.shape
    preds = np.full(n, np.nan)
    sel_layers: List[int] = []
    sel_cs: List[float] = []
    for test in grouped_folds(clusters, FOLDS):
        train = ~test
        if len(np.unique(y[train])) < 2:
            continue
        inner_folds = grouped_folds(clusters[train], 3, seed=SEED + 2)
        best = (-1.0, 0, INNER_C_GRID[0])
        wide = X_layers.shape[-1] > 8000
        for li in range(L):
            for c in (c_grid or (INNER_C_GRID[:2] if wide else INNER_C_GRID)):
                scores = []
                for itest in inner_folds:
                    itrain = ~itest
                    if len(np.unique(y[train][itrain])) < 2 or len(np.unique(y[train][itest])) < 2:
                        continue
                    pipe = _lr(c, X_layers.shape[-1])
                    pipe.fit(X_layers[train][itrain, li, :], y[train][itrain])
                    v = auroc(pipe.decision_function(X_layers[train][itest, li, :]),
                              y[train][itest])
                    if v is not None:
                        scores.append(v)
                if scores and float(np.mean(scores)) > best[0]:
                    best = (float(np.mean(scores)), li, c)
        _, li, c = best
        sel_layers.append(li)
        sel_cs.append(c)
        pipe = _lr(c, X_layers.shape[-1])
        pipe.fit(X_layers[train, li, :], y[train])
        preds[test] = pipe.decision_function(X_layers[test, li, :])
    ok = ~np.isnan(preds)
    out: Dict[str, Any] = {
        "n_scored": int(ok.sum()),
        "selected_layers": sel_layers, "selected_c": sel_cs,
        "_oof": preds, "_ok": ok,
    }
    if ok.sum() >= len(y) * 0.5 and len(np.unique(y[ok])) == 2:
        out["auprc"] = round(auprc(preds[ok], y[ok]), 4)
        out["auroc"] = round(auroc(preds[ok], y[ok]), 4)
    return out


def fpr_at_tpr(scores: np.ndarray, y: np.ndarray, tpr_target: float = 0.90) -> Optional[float]:
    order = np.argsort(-scores, kind="mergesort")
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
    """anchor_hidden is [L, A, H]; anchor order from the PERSISTED
    anchor_labels array when present (topup shards previously had no way to
    be verified), falling back to the steps-file anchors list."""
    rows = []
    keep = []
    for i, qid in enumerate(qids):
        key = f"{qid}::anchor_hidden"
        if key not in npz:
            continue
        labels_key = f"{qid}::anchor_labels"
        if labels_key in npz:
            labels = [str(x) for x in npz[labels_key]]
        else:
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


def kwts_head_ensemble(
    Xo: np.ndarray, y: np.ndarray, clusters: np.ndarray,
    folds: List[np.ndarray], n_heads: int = 32, top_k: int = 5,
) -> np.ndarray:
    """Xo: [n, Sel, H] boundary o_proj inputs (mean over boundaries per layer).
    Head search INSIDE the CV (and inside the permutation loop when the
    caller re-runs this on permuted labels): per outer fold, rank
    (strided-layer, head) pairs by INNER-fold AUROC of a per-head LR, take
    the top-k heads, fit one LR per selected head on the outer-train
    activations, orient each by its inner AUROC sign, and average the
    out-of-fold decision scores."""
    n, sel, H = Xo.shape
    head_dim = H // n_heads
    pairs = [(si, h) for si in range(sel) for h in range(n_heads)]
    preds = np.full(n, np.nan)
    for test in folds:
        train = ~test
        if len(np.unique(y[train])) < 2:
            continue
        inner_folds = grouped_folds(clusters[train], 3, seed=SEED + 3)
        ranked: List[Tuple[float, Tuple[int, int]]] = []
        for (si, h) in pairs:
            feats = Xo[:, si, h * head_dim:(h + 1) * head_dim]
            scores = []
            for itest in inner_folds:
                itrain = ~itest
                if len(np.unique(y[train][itrain])) < 2 or len(np.unique(y[train][itest])) < 2:
                    continue
                pipe = _lr(KWTS_HEAD_C)
                pipe.fit(feats[train][itrain], y[train][itrain])
                v = auroc(pipe.decision_function(feats[train][itest]), y[train][itest])
                if v is not None:
                    scores.append(v)
            if scores:
                m = float(np.mean(scores))
                ranked.append((max(m, 1 - m), (si, h), 1.0 if m >= 0.5 else -1.0))
        ranked.sort(key=lambda t: -t[0])
        top = ranked[:top_k]
        if not top:
            continue
        member_preds = []
        for _score, (si, h), sign in top:
            feats = Xo[:, si, h * head_dim:(h + 1) * head_dim]
            pipe = _lr(KWTS_HEAD_C)
            pipe.fit(feats[train], y[train])
            member_preds.append(sign * pipe.decision_function(feats[test]))
        preds[test] = np.mean(member_preds, axis=0)
    return preds


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
    parser.add_argument("--split", default="default", choices=["default", "toolset_disjoint"],
                        help="toolset_disjoint: sessions sharing a tool-pool signature land "
                             "in the same fold (prereg's second split)")
    args = parser.parse_args(argv)

    if not HAS_SKLEARN:
        print("sklearn unavailable", file=sys.stderr)
        return 2

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    pairs = join_arms(load_jsonl(args.battery_full), load_jsonl(args.battery_c2kv))
    label_frame = build_label_frame(pairs, manifest)
    lab_by_qid = {r["qid"]: r for r in label_frame}
    # ALIEN arm-A labels (c2kv wrong vs gold on the whole 900 frame) come from
    # the label side; the label frame is never a model input elsewhere
    cw_wrong_any = {f["qid"]: (0 if c.get("tool_name_match") else 1) for f, c in pairs}

    steps_rows = load_jsonl(Path(args.capture_dir) / args.arm / "p0.steps.jsonl")
    steps_by_qid = {r["qid"]: r for r in steps_rows}
    npz: Dict[str, Any] = {}
    npz_shards: List[str] = []
    for path in sorted((Path(args.capture_dir) / args.arm).glob("*.hid.npz")):
        npz_shards.append(path.name)
        with np.load(path, allow_pickle=True) as z:
            for k in z.files:
                if k not in npz:
                    npz[k] = z[k]

    qids = [r["qid"] for r in steps_rows if r["qid"] in lab_by_qid]
    if args.max_rows:
        qids = qids[: args.max_rows]
    keep_mask = [lab_by_qid[q]["label_cw"] in (0, 1) for q in qids]
    qids_t = [q for q, k in zip(qids, keep_mask) if k]
    y = np.array([1 if lab_by_qid[q]["label_cw"] == 1 else 0 for q in qids_t])
    clusters = session_clusters([lab_by_qid[q]["session_id"] for q in qids_t])
    pf = np.array([bool(lab_by_qid[q].get("parse_fail_fire")) for q in qids_t])
    out: Dict[str, Any] = {"arm": args.arm, "n": len(qids_t), "n_pos": int(y.sum()),
                           "qids": qids_t, "npz_shards": npz_shards, "split": args.split}

    if args.split == "toolset_disjoint":
        # signature = the session's candidate tool pool (first-token ids from
        # the capture's IC record); sessions sharing a pool go to one fold
        sess_sorted = sorted({r["session_id"] for r in label_frame})
        sess_to_pool: Dict[str, tuple] = {}
        for rec in steps_rows:
            sess = (rec.get("meta") or {}).get("session_id") or rec["qid"].rsplit(":", 1)[0]
            pool = tuple(sorted(((rec.get("ic") or {}).get("candidate_token_ids")) or []))
            if pool:
                sess_to_pool[sess] = pool
        sig_ids: Dict[tuple, int] = {}
        signatures: Dict[int, int] = {}
        for si, sess in enumerate(sess_sorted):
            pool = sess_to_pool.get(sess)
            signatures[si] = sig_ids.setdefault(pool, len(sig_ids)) if pool is not None else -1
        _FOLD_MODE["mode"] = "toolset_disjoint"
        _FOLD_MODE["signatures"] = signatures
        out["n_toolset_signatures"] = len(sig_ids)

    n_layers = None
    if qids_t:
        k0 = f"{qids_t[0]}::query_last"
        if k0 in npz:
            n_layers = npz[k0].shape[0]
    every5 = list(range(2, n_layers, 5)) if n_layers else []

    # --- probe_prefill: layer selected in INNER folds; per-layer OOF scores
    # shipped so the report can do the same-layer twin (c2kv vs full at the
    # SAME layer) instead of comparing each arm's own best layer
    Xq = np.stack([npz[f"{q}::query_last"] for q in qids_t]).astype(np.float32)  # [n, L, H]
    sel = fit_layer_select_cv(Xq[:, every5 or list(range(n_layers or 0)), :], y, clusters)
    out["probe_prefill_layer_select"] = {k: v for k, v in sel.items() if not k.startswith("_")}
    out["probe_prefill_layer_select"]["oof_scores"] = [None if np.isnan(v) else round(float(v), 5)
                                                       for v in sel["_oof"]]
    out["probe_prefill_layer_select"]["layers_grid"] = list(every5)
    per_layer_scores: Dict[str, List[Optional[float]]] = {}
    for li in every5 or list(range(n_layers or 0)):
        d = fit_lr_cv_detailed(Xq[:, li, :], y, clusters)
        per_layer_scores[str(li)] = {
            "auprc": d["auprc"], "auroc": d["auroc"],
            # descriptive only — outer-selected; never quoted as the headline
            "oof_scores": [None if np.isnan(v) else round(float(v), 5) for v in d["_oof"]],
        }
    out["probe_prefill_per_layer"] = per_layer_scores
    if n_layers and n_layers <= 40:
        d = fit_lr_cv(Xq.reshape(len(qids_t), -1), y, clusters)
        out["probe_prefill_all_layers"] = d

    # --- joint overflow: [context side; query side] on shared strided layers ---
    ctx_key = "ctx_hid" if args.arm == "full" else "gist_hid"
    try:
        have_ctx = qids_t and f"{qids_t[0]}::{ctx_key}" in npz
        if have_ctx:
            ctx_rows = [npz[f"{q}::{ctx_key}"][:, -1, :] for q in qids_t]  # [Sel, H]
            Xc = np.stack(ctx_rows).astype(np.float32)  # [n, Sel, H]
            sel_layers = Xc.shape[1]
            ctx_last = Xc.reshape(len(qids_t), -1)
            query_sel = Xq[:, :: max(1, n_layers // sel_layers), :].reshape(len(qids_t), -1)
            joint = np.concatenate([ctx_last, query_sel], axis=1)
            dj = fit_lr_cv_detailed(joint, y, clusters)
            out["joint_overflow"] = {k: v for k, v in dj.items() if not k.startswith("_")}
            out["joint_overflow"]["oof_scores"] = [None if np.isnan(v) else round(float(v), 5)
                                                   for v in dj["_oof"]]
            dc = fit_lr_cv_detailed(ctx_last, y, clusters)
            out["joint_context_only"] = {k: v for k, v in dc.items() if not k.startswith("_")}
        else:
            out["joint_overflow_missing"] = (f"no {ctx_key} arrays in capture; the gist-path "
                                             "hooks were structurally dead before the fix")
    except Exception as exc:  # noqa: BLE001
        out["joint_overflow_error"] = repr(exc)[:200]

    # --- kwts: per-(layer,head) probes on boundary o_proj inputs ---
    oproj_key = "ctx_oproj" if args.arm == "full" else "gist_oproj"
    try:
        if qids_t and f"{qids_t[0]}::{oproj_key}" in npz:
            row_means = [npz[f"{q}::{oproj_key}"].mean(axis=1) for q in qids_t]  # [Sel, H]
            Xo = np.stack(row_means).astype(np.float32)  # [n, Sel, H]
            folds = grouped_folds(clusters, FOLDS)
            preds = kwts_head_ensemble(Xo, y, clusters, folds)
            ok = ~np.isnan(preds)
            if ok.all() and len(np.unique(y[ok])) == 2:
                out["kwts_ensemble"] = {"auprc": round(auprc(preds, y), 4),
                                        "auroc": round(auroc(preds, y), 4),
                                        "oof_scores": [round(float(v), 5) for v in preds]}
            else:
                out["kwts_ensemble"] = {"n_scored": int(ok.sum()),
                                        "note": "some folds produced no predictions"}
            # permutation band with the head search INSIDE the loop
            rng = np.random.default_rng(SEED)
            perm_auprcs = []
            for _p in range(PERMUTATIONS):
                yperm = rng.permutation(y)
                pp = kwts_head_ensemble(Xo, yperm, clusters, folds)
                okp = ~np.isnan(pp)
                if okp.all():
                    perm_auprcs.append(auprc(pp[okp], yperm[okp]))
            if perm_auprcs:
                out["kwts_permutation_band"] = {
                    "mean": round(float(np.mean(perm_auprcs)), 4),
                    "p95": round(float(np.percentile(perm_auprcs, 95)), 4),
                    "n_perms": len(perm_auprcs),
                }
        else:
            out["kwts_missing"] = (f"no {oproj_key} arrays in capture")
    except Exception as exc:  # noqa: BLE001
        out["kwts_error"] = repr(exc)[:200]

    # --- tool-call error probe: per-layer at two anchors; FPR@90TPR from the
    # SAME OOF scores (previously AUPRC and FPR came from two different
    # models); parse-failure-excluded variant per prereg ---
    for anchor in ("name_last", "last"):
        Xa, keep = load_anchor_matrix(npz, qids_t, anchor, steps_by_qid, layers=every5)
        if len(keep) == 0:
            continue
        ya = y[keep]
        ca = clusters[keep]
        d = fit_lr_cv_detailed(Xa, ya, ca)
        entry = {k: v for k, v in d.items() if not k.startswith("_")}
        entry["oof_scores"] = [None if np.isnan(v) else round(float(v), 5) for v in d["_oof"]]
        okp = d["_ok"]
        if okp.sum() == len(ya) and len(np.unique(ya[okp])) == 2:
            entry["fpr_at_90tpr"] = fpr_at_tpr(d["_oof"][okp], ya[okp])
        nopa = ~pf[keep]
        if nopa.sum() > 20 and len(np.unique(ya[nopa])) == 2:
            de = fit_lr_cv_detailed(Xa[nopa], ya[nopa], ca[nopa])
            entry["parse_fail_excluded"] = {k: v for k, v in de.items()
                                            if not k.startswith("_")}
            entry["parse_fail_excluded"]["n"] = int(nopa.sum())
        out[f"tool_call_error_{anchor}"] = entry

    # --- exact-answer position comparison + cheap Logits-min baselines ---
    pos_table = {}
    for anchor in ("name_first", "name_last", "penult"):
        Xa, keep = load_anchor_matrix(npz, qids_t, anchor, steps_by_qid, layers=every5)
        if len(keep):
            pos_table[anchor] = fit_lr_cv(Xa, y[keep], clusters[keep])
            pos_table[anchor]["n"] = int(len(keep))
    # Logits-min baselines from the capture steps (chosen-token logprobs):
    # min p over the name span — the cheap baseline the probe must beat
    try:
        lmin, lmin_keep = [], []
        for i, q in enumerate(qids_t):
            rec = steps_by_qid.get(q) or {}
            spans = rec.get("spans") or {}
            steps = rec.get("steps") or []
            nf, nl = spans.get("name_first"), spans.get("name_last")
            if nf is None or nl is None or not steps:
                continue
            ps = [math.exp(steps[j]["chosen_logprob"]) for j in range(nf, nl + 1) if j < len(steps)]
            if ps:
                lmin.append(min(ps))
                lmin_keep.append(i)
        if lmin:
            lmin = np.array(lmin)
            keep = np.array(lmin_keep)
            risk = -lmin  # lower min-p => higher risk
            pos_table["logits_min_name_span"] = {
                "auprc": round(auprc(risk, y[keep]), 4),
                "auroc": round(auroc(risk, y[keep]), 4),
                "n": int(len(keep)),
                "note": "cheap baseline; probe rows must be read against this",
            }
    except Exception as exc:  # noqa: BLE001
        pos_table["logits_min_name_span"] = {"error": repr(exc)[:120]}
    out["exact_answer_positions"] = pos_table
    # single-layer reference kept OUT of the position table (it was previously
    # compared side-by-side with 7-layer-concat probe rows)
    if every5:
        d = fit_lr_cv(Xq[:, every5[len(every5) // 2], :], y, clusters)
        out["query_last_single_layer_ref"] = {**d, "layer": every5[len(every5) // 2],
                                              "note": "single-layer diagnostic, not comparable to concat probes"}

    # --- ALIEN two-arm (penult layer, name_last anchor) ---
    Xa, keep = load_anchor_matrix(npz, qids_t, "name_last", steps_by_qid,
                                  layers=[(n_layers or 2) - 2] if n_layers else None)
    if len(keep):
        # arm B: C->W labels on the trigger subset (as before)
        db = fit_lr_cv_detailed(Xa, y[keep], clusters[keep])
        out["alien_arm_b_cw"] = {k: v for k, v in db.items() if not k.startswith("_")}
        out["alien_arm_b_cw"]["oof_scores"] = [None if np.isnan(v) else round(float(v), 5)
                                               for v in db["_oof"]]
        out["alien_arm_b_cw"]["oof_row_indices"] = [int(i) for i in keep]
        # arm A: wrong-any labels (c2kv wrong vs gold) over ALL hidden rows,
        # scored on the C->W subset via the OOF predictions — the two-arm gap
        # on C->W is the prereg's reported quantity.  Previously rows_arm_a
        # was assigned and never used.
        try:
            qids_all = [q for q in qids if f"{q}::anchor_hidden" in npz]
            Xall, keep_all = load_anchor_matrix(
                npz, qids_all, "name_last", steps_by_qid,
                layers=[(n_layers or 2) - 2] if n_layers else None)
            y_all = np.array([cw_wrong_any[q] for q in qids_all])[keep_all]
            c_all = session_clusters([lab_by_qid[q]["session_id"] for q in qids_all])[keep_all]
            trig = np.array([lab_by_qid[q]["label_cw"] == 1 for q in qids_all])[keep_all]
            da = fit_lr_cv_detailed(Xall, y_all, c_all)
            oof, okm = da["_oof"], da["_ok"]
            cw_rows = trig & okm
            arm_a_on_cw = None
            if cw_rows.sum() > 10 and len(np.unique(y[cw_rows])) == 2:
                arm_a_on_cw = round(auprc(oof[cw_rows], y_all[cw_rows]), 4)
            out["alien_arm_a_wrong_any"] = {
                "n_900": int(len(y_all)), "n_wrong_any": int(y_all.sum()),
                "auprc_on_cw": arm_a_on_cw,
            }
            if arm_a_on_cw is not None and db["auprc"] is not None:
                out["alien_two_arm_gap_on_cw"] = round(db["auprc"] - arm_a_on_cw, 4)
        except Exception as exc:  # noqa: BLE001
            out["alien_arm_a_wrong_any"] = {"error": repr(exc)[:200]}

    # --- concealment gap: s_text from whitelist text features ---
    if args.features:
        feats = {r["qid"]: r for r in load_jsonl(args.features)}
        cols = ["text_n_chars", "text_closed_tag", "text_parse_ok", "text_digit_frac",
                "text_brace_count", "text_distinct_word_frac", "text_payload_chars"]
        Xtext = np.array([[float(feats[q].get(c) or 0.0) for c in cols] for q in qids_t], dtype=np.float32)
        out["s_text_whitelist_lr"] = fit_lr_cv(Xtext, y, clusters)
        s_h = out.get("probe_prefill_all_layers") or out.get("tool_call_error_name_last") or {}
        if isinstance(s_h, dict) and s_h.get("auprc") is not None and out["s_text_whitelist_lr"].get("auprc") is not None:
            out["concealment_gap"] = round(s_h["auprc"] - out["s_text_whitelist_lr"]["auprc"], 4)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[t33] probes -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
