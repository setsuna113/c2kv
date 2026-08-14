#!/usr/bin/env python
"""Gist readability probe analysis (pure CPU, numpy + stdlib only).

Question: is the target tool NAME still linearly readable from the pooled KV
features of the compressed (gist) prefix? Loads features_index.jsonl + per
sample .npz files written by agent/extract_kv_pool_features.py and runs the
pre-registered probe pipeline per arm (gist / full).

PROBE (candidate-scoring, closed-set): this is a NEAREST-CLASS-MEAN linear
probe. Per session the candidate set is that session's tool names. Each
candidate tool is represented by its feature prototype = mean pooled feature
of the TRAINING samples whose target is that tool (session-grouped training
folds only). Score = cosine similarity between the sample feature and each
candidate prototype; prediction = argmax over candidates; accuracy is computed
over samples whose session has >= 2 candidates.

FLOOR (dual estimator, MAX of the two):
  (a) permutation floor: --n_perm repetitions of qid x feature mismatch
      permutation (features shuffled across qids WITHIN session, so per-session
      priors stay matched), mean accuracy;
  (b) session prior floor: sample-level CV; always predict the most frequent
      target tool of the same session in the TRAIN folds.
  If (a) and (b) disagree by > 0.10 absolute, a sensitivity note is emitted
  with both values.

VALIDATION: session-grouped nested CV. Outer --n_outer folds by session_id.
ALL feature choices (which pooling arrays: k_last / v_last / k_mean / v_mean /
concatenations; layer scope: per-layer vs all-layer-concat vs all-layer-mean)
are selected ONLY inside the inner --n_inner-fold CV (by inner accuracy).
Reported: outer-fold accuracy mean +/- sd; final number = mean outer accuracy
on held-out sessions.

GATES (pre-registered, printed verbatim + verdict):
  (1) sensitivity precondition: full-arm probe - floor >= 0.15 AND full-arm
      absolute accuracy >= 0.302, else "paradigm INCONCLUSIVE (early stop)";
  (2) positive: gist-arm probe - floor >= 0.10
      => "information linearly readable (ALIVE)";
  (3) negative wording FIXED: "未检出线性可读动作信息（可判界<=X）" where
      X = upper 95% CI bound of (gist - floor) — never write "信息不在".
      CI: Wilson score interval on the paired per-sample correctness
      difference (probe vs floor), the bounded [-1,1] differences mapped to
      [0,1] for the Wilson computation and mapped back.

Outputs: <out_prefix>.json (full results) and <out_prefix>.md (gates table:
pre-registered text | measured | verdict).

Example:
  python agent/analyze_gist_probe.py \
    --features_dir ./outputs/gist_probe_features \
    --out_prefix ./outputs/gist_probe --n_outer 5 --n_inner 4 --n_perm 5 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

Z_975 = 1.959964

GATE1_TEXT = (
    "(1) sensitivity precondition: full-arm probe - floor >= 0.15 AND full-arm "
    'absolute accuracy >= 0.302, else "paradigm INCONCLUSIVE (early stop)"'
)
GATE2_TEXT = (
    "(2) positive: gist-arm probe - floor >= 0.10 "
    '=> "information linearly readable (ALIVE)"'
)
GATE3_TEXT = (
    "(3) negative wording FIXED: \"未检出线性可读动作信息（可判界<=X）\" where X = the "
    'upper 95% CI bound of (gist - floor) — never write "信息不在"'
)
VERDICT_INCONCLUSIVE = "paradigm INCONCLUSIVE (early stop)"
VERDICT_ALIVE = "information linearly readable (ALIVE)"

# Pooling-array candidates for the inner-CV feature choice.
ARRAY_SETS: List[Tuple[str, Tuple[str, ...]]] = [
    ("k_last", ("k_last",)),
    ("v_last", ("v_last",)),
    ("k_mean", ("k_mean",)),
    ("v_mean", ("v_mean",)),
    ("kv_last", ("k_last", "v_last")),
    ("kv_mean", ("k_mean", "v_mean")),
    ("all4", ("k_last", "v_last", "k_mean", "v_mean")),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_arm_samples(features_dir: Path, arm: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Index rows for one arm, filtered to probe-eligible samples."""
    index_path = features_dir / "features_index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"{index_path} not found; run extract_kv_pool_features.py first")
    excluded = Counter()
    samples: List[Dict[str, Any]] = []
    seen_qids = set()
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("arm") != arm:
                continue
            if row.get("skipped"):
                excluded["skipped_extraction"] += 1
                continue
            qid = row.get("qid")
            if qid in seen_qids:
                continue
            seen_qids.add(qid)
            target = row.get("target_tool_name")
            candidates = list(dict.fromkeys(row.get("session_tool_names") or []))
            if not target:
                excluded["no_target_tool_name"] += 1
                continue
            if len(candidates) < 2:
                excluded["lt2_candidates"] += 1
                continue
            if target not in candidates:
                excluded["target_not_in_candidates"] += 1
                continue
            if not row.get("features_file") or not (features_dir / row["features_file"]).exists():
                excluded["missing_features_file"] += 1
                continue
            samples.append({
                "qid": qid,
                "session_id": row.get("session_id") or qid.rsplit(":", 1)[0],
                "target": target,
                "candidates": candidates,
                "features_file": row["features_file"],
                "n_layers": row.get("n_layers"),
                "n_kv_heads": row.get("n_kv_heads"),
                "head_dim": row.get("head_dim"),
            })
    return samples, dict(excluded)


def _build_blocks(
    features_dir: Path,
    samples: List[Dict[str, Any]],
) -> Tuple[Dict[Tuple[str, int], np.ndarray], Dict[str, np.ndarray], Dict[str, int]]:
    """Materialize per-sample pooled features as float32 block matrices.

    blocks[(array, layer)] : (n_samples, n_kv_heads * head_dim) float32
    mblocks[array]         : mean of blocks[(array, layer)] over layers
    """
    meta_rows = [s for s in samples if s.get("n_layers") and s.get("n_kv_heads") and s.get("head_dim")]
    if not meta_rows:
        raise ValueError("No samples carry n_layers/n_kv_heads/head_dim metadata")
    n_layers = int(meta_rows[0]["n_layers"])
    n_kv_heads = int(meta_rows[0]["n_kv_heads"])
    head_dim = int(meta_rows[0]["head_dim"])
    block_dim = n_kv_heads * head_dim
    n = len(samples)
    arrays = sorted({array for _, names in ARRAY_SETS for array in names})
    blocks = {
        (array, layer): np.empty((n, block_dim), dtype=np.float32)
        for array in arrays
        for layer in range(n_layers)
    }
    for row_idx, sample in enumerate(samples):
        npz = np.load(features_dir / sample["features_file"])
        for array in arrays:
            for layer in range(n_layers):
                key = (array, layer)
                blocks[key][row_idx] = np.concatenate(
                    [npz[f"layer{layer}_head{head}_{array}"] for head in range(n_kv_heads)]
                ).astype(np.float32, copy=False)
        npz.close()
    mblocks = {
        array: np.mean(np.stack([blocks[(array, layer)] for layer in range(n_layers)]), axis=0).astype(np.float32)
        for array in arrays
    }
    meta = {"n_layers": n_layers, "n_kv_heads": n_kv_heads, "head_dim": head_dim, "block_dim": block_dim}
    return blocks, mblocks, meta


# ---------------------------------------------------------------------------
# Feature views (the choices selected inside the inner CV)
# ---------------------------------------------------------------------------

def _feature_views(n_layers: int) -> List[Tuple[str, List[Tuple[str, Any]]]]:
    """(view_id, block keys) candidates.

    Block keys: ("layer", (array, layer_idx)) for per-layer / all-layer-concat
    scopes, ("mean", array) for the all-layer-mean scope. Concatenation views
    are scored without materializing the concatenated matrix: cosine
    numerators and squared norms are summed over blocks.
    """
    views: List[Tuple[str, List[Tuple[str, Any]]]] = []
    for set_name, arrays in ARRAY_SETS:
        for layer in range(n_layers):
            views.append((
                f"{set_name}|layer{layer}",
                [("layer", (array, layer)) for array in arrays],
            ))
        views.append((
            f"{set_name}|all_layer_mean",
            [("mean", array) for array in arrays],
        ))
        views.append((
            f"{set_name}|all_layer_concat",
            [("layer", (array, layer)) for array in arrays for layer in range(n_layers)],
        ))
    return views


# ---------------------------------------------------------------------------
# Nearest-class-mean probe with cosine scoring
# ---------------------------------------------------------------------------

class _FoldStats:
    """Lazily computed per-block statistics for one train/test split.

    For block key k: prototypes P (n_classes x dim), squared prototype norms,
    raw test scores X_test @ P.T, squared test row norms. Train-side stats can
    be shared across folds with identical train_idx via `shared_proto` (used by
    the permutation repetitions, which only re-slice the test rows).
    """

    def __init__(
        self,
        blocks: Dict[Any, np.ndarray],
        mblocks: Dict[str, np.ndarray],
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        targets: Sequence[str],
        shared_proto: Optional[Dict[Any, Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        self._blocks = blocks
        self._mblocks = mblocks
        self.train_idx = np.asarray(train_idx)
        self.test_idx = np.asarray(test_idx)
        train_targets = [targets[i] for i in self.train_idx]
        self.classes = sorted(set(train_targets))
        self.class_index = {name: col for col, name in enumerate(self.classes)}
        self._y_train = np.array([self.class_index[t] for t in train_targets], dtype=np.int64)
        self._onehot = None
        self._counts = None
        self._proto = shared_proto if shared_proto is not None else {}
        self._test_stats: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}

    def _matrix(self, key: Tuple[str, Any]) -> np.ndarray:
        kind, ident = key
        return self._mblocks[ident] if kind == "mean" else self._blocks[ident]

    def _train_proto(self, key: Tuple[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        if key not in self._proto:
            X = self._matrix(key)[self.train_idx]
            n_classes = len(self.classes)
            if self._onehot is None:
                self._onehot = (
                    self._y_train[None, :] == np.arange(n_classes)[:, None]
                ).astype(np.float32)
                self._counts = np.bincount(self._y_train, minlength=n_classes).astype(np.float32)
            proto = (self._onehot @ X) / np.maximum(self._counts, 1.0)[:, None]
            proto[self._counts == 0] = 0.0
            pnorm2 = np.einsum("ij,ij->i", proto, proto)
            self._proto[key] = (proto, pnorm2)
        return self._proto[key]

    def block_stats(self, key: Tuple[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(test scores n_test x n_classes, test row sq norms, proto sq norms)."""
        if key not in self._test_stats:
            proto, pnorm2 = self._train_proto(key)
            Xte = self._matrix(key)[self.test_idx]
            scores = Xte @ proto.T
            xnorm2 = np.einsum("ij,ij->i", Xte, Xte)
            self._test_stats[key] = (scores, xnorm2)
        _, pnorm2 = self._train_proto(key)
        scores, xnorm2 = self._test_stats[key]
        return scores, xnorm2, pnorm2


def _view_correctness(
    stats: _FoldStats,
    block_keys: List[Tuple[str, Any]],
    cand_cols: Sequence[List[int]],
    target_cols: np.ndarray,
) -> np.ndarray:
    """Per-test-sample 0/1 correctness of the NCM-cosine probe for a view."""
    num: Optional[np.ndarray] = None
    xnorm2: Optional[np.ndarray] = None
    pnorm2: Optional[np.ndarray] = None
    for key in block_keys:
        scores, x2, p2 = stats.block_stats(key)
        num = scores if num is None else num + scores
        xnorm2 = x2 if xnorm2 is None else xnorm2 + x2
        pnorm2 = p2 if pnorm2 is None else pnorm2 + p2
    denom = np.sqrt(np.maximum(xnorm2, 0.0))[:, None] * np.sqrt(np.maximum(pnorm2, 0.0))[None, :]
    cosine = num / (denom + 1e-12)
    correct = np.zeros(len(target_cols), dtype=np.float64)
    for j in range(len(target_cols)):
        cols = cand_cols[j]
        if not cols:
            continue  # no candidate has a prototype -> cannot be correct
        best = cols[int(np.argmax(cosine[j, cols]))]
        correct[j] = float(best == target_cols[j])
    return correct


def _cand_cols(stats: _FoldStats, candidates: Sequence[List[str]], test_idx: np.ndarray) -> List[List[int]]:
    return [
        [stats.class_index[c] for c in candidates[i] if c in stats.class_index]
        for i in test_idx
    ]


def _target_cols(stats: _FoldStats, targets: Sequence[str], test_idx: np.ndarray) -> np.ndarray:
    return np.array([stats.class_index.get(targets[i], -1) for i in test_idx], dtype=np.int64)


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------

def _session_folds(sessions: Sequence[str], n_folds: int, seed: int) -> List[np.ndarray]:
    """Session-grouped folds: returns per-fold arrays of session ids."""
    unique = sorted(set(sessions))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(unique))
    folds = np.array_split(np.array(unique, dtype=object)[order], min(n_folds, len(unique)))
    return [np.asarray(list(fold), dtype=object) for fold in folds if len(fold)]


def _sample_folds(n: int, n_folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.RandomState(seed)
    order = rng.permutation(n)
    return [fold for fold in np.array_split(order, min(n_folds, n)) if len(fold)]


def _idx_for_sessions(sessions: Sequence[str], keep: np.ndarray) -> np.ndarray:
    keep_set = set(keep.tolist())
    return np.array([i for i, s in enumerate(sessions) if s in keep_set], dtype=np.int64)


def _within_session_permutation(test_idx: np.ndarray, sessions: Sequence[str], rng: np.random.RandomState) -> np.ndarray:
    """Permutation of test positions that shuffles features across qids WITHIN
    session (per-session candidate priors stay matched)."""
    perm = np.arange(len(test_idx))
    by_session: Dict[str, List[int]] = {}
    for pos, sample_idx in enumerate(test_idx):
        by_session.setdefault(sessions[sample_idx], []).append(pos)
    for positions in by_session.values():
        if len(positions) > 1:
            perm[positions] = np.asarray(positions)[rng.permutation(len(positions))]
    return perm


def _session_prior_correctness(
    targets: Sequence[str],
    sessions: Sequence[str],
    candidates: Sequence[List[str]],
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Floor (b): predict the session's most frequent TRAIN-fold target.

    Sample-level (not session-grouped) folds: this estimator intentionally
    allows same-session train leakage, because it is meant to measure what a
    per-session class prior alone would buy. Ties break alphabetically;
    sessions unseen in train fall back to the global train majority.
    """
    n = len(targets)
    correct = np.zeros(n, dtype=np.float64)
    all_idx = np.arange(n)
    for test_idx in _sample_folds(n, n_folds, seed):
        train_idx = np.setdiff1d(all_idx, test_idx)
        session_counts: Dict[str, Counter] = {}
        global_counts: Counter = Counter()
        for i in train_idx:
            session_counts.setdefault(sessions[i], Counter())[targets[i]] += 1
            global_counts[targets[i]] += 1
        for j in test_idx:
            cands = sorted(candidates[j])
            prediction = _majority_pick(session_counts.get(sessions[j], Counter()), cands)
            if prediction is None:
                prediction = _majority_pick(global_counts, cands)
            correct[j] = float(prediction is not None and prediction == targets[j])
    return correct


def _majority_pick(counts: Counter, cands: List[str]) -> Optional[str]:
    best: Optional[str] = None
    best_count = 0
    for cand in cands:
        count = counts.get(cand, 0)
        if count > best_count:
            best, best_count = cand, count
    return best


def _wilson_upper_diff(diffs: np.ndarray) -> float:
    """Upper 95% Wilson bound of the mean paired difference in [-1, 1].

    Paired per-sample differences d_i in [-1,1] are mapped to q_i=(d_i+1)/2 in
    [0,1]; the Wilson score interval is computed on mean(q) and the upper
    bound mapped back: X = 2*U - 1.
    """
    n = len(diffs)
    if n == 0:
        return float("nan")
    q = (np.asarray(diffs, dtype=np.float64) + 1.0) / 2.0
    p = float(q.mean())
    denom = 1.0 + Z_975**2 / n
    center = (p + Z_975**2 / (2.0 * n)) / denom
    half = Z_975 * math.sqrt(p * (1.0 - p) / n + Z_975**2 / (4.0 * n**2)) / denom
    return 2.0 * (center + half) - 1.0


# ---------------------------------------------------------------------------
# Per-arm nested-CV pipeline
# ---------------------------------------------------------------------------

def _run_arm(
    arm: str,
    features_dir: Path,
    n_outer: int,
    n_inner: int,
    n_perm: int,
    seed: int,
) -> Dict[str, Any]:
    samples, excluded = _load_arm_samples(features_dir, arm)
    result: Dict[str, Any] = {"arm": arm, "n_index_excluded": excluded}
    if not samples:
        result["error"] = "no eligible samples"
        return result
    blocks, mblocks, meta = _build_blocks(features_dir, samples)
    n_layers = meta["n_layers"]
    views = _feature_views(n_layers)

    targets = [s["target"] for s in samples]
    sessions = [s["session_id"] for s in samples]
    candidates = [s["candidates"] for s in samples]
    n = len(samples)
    result.update({
        "n_samples": n,
        "n_sessions": len(set(sessions)),
        "feature_meta": meta,
        "n_views": len(views),
    })

    probe_correct = np.zeros(n, dtype=np.float64)  # filled once per sample (its outer test fold)
    perm_correct = np.zeros(n, dtype=np.float64)   # mean over reps, same alignment
    fold_rows: List[Dict[str, Any]] = []

    outer_folds = _session_folds(sessions, n_outer, seed)
    for outer_i, test_sessions in enumerate(outer_folds):
        test_idx = _idx_for_sessions(sessions, test_sessions)
        train_idx = np.setdiff1d(np.arange(n), test_idx)
        train_sessions = sorted(set(sessions[i] for i in train_idx))

        # --- inner CV: select the feature view on session-grouped inner folds ---
        inner_folds = _session_folds([sessions[i] for i in train_idx], n_inner, seed * 1000 + outer_i + 1)
        view_scores: List[List[float]] = [[] for _ in views]
        for val_sessions in inner_folds:
            val_local = _idx_for_sessions([sessions[i] for i in train_idx], val_sessions)
            val_idx = train_idx[val_local]
            inner_train_idx = np.setdiff1d(train_idx, val_idx)
            inner_stats = _FoldStats(blocks, mblocks, inner_train_idx, val_idx, targets)
            inner_cands = _cand_cols(inner_stats, candidates, val_idx)
            inner_target_cols = _target_cols(inner_stats, targets, val_idx)
            for view_i, (_view_id, block_keys) in enumerate(views):
                correct = _view_correctness(inner_stats, block_keys, inner_cands, inner_target_cols)
                view_scores[view_i].append(float(correct.mean()))
        view_inner_acc = [float(np.mean(scores)) if scores else 0.0 for scores in view_scores]
        best_view_i = int(np.argmax(view_inner_acc))  # first max wins (fixed view order)
        best_view_id, best_block_keys = views[best_view_i]

        # --- outer evaluation with the selected view ---
        shared_proto: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        stats = _FoldStats(blocks, mblocks, train_idx, test_idx, targets, shared_proto=shared_proto)
        cands = _cand_cols(stats, candidates, test_idx)
        tcols = _target_cols(stats, targets, test_idx)
        fold_probe = _view_correctness(stats, best_block_keys, cands, tcols)
        probe_correct[test_idx] = fold_probe

        # --- floor (a): within-session feature permutation, same view ---
        rng = np.random.RandomState(seed * 100000 + outer_i)
        fold_perm = np.zeros(len(test_idx), dtype=np.float64)
        for _rep in range(n_perm):
            perm = _within_session_permutation(test_idx, sessions, rng)
            perm_stats = _FoldStats(
                blocks, mblocks, train_idx, test_idx[perm], targets, shared_proto=shared_proto
            )
            # test_idx[perm] rows = mismatched features; targets/candidates stay
            # in test_idx order, so correctness[j] pairs sample j's target with
            # another (same-session) sample's features.
            fold_perm += _view_correctness(perm_stats, best_block_keys, cands, tcols)
        fold_perm /= max(n_perm, 1)
        perm_correct[test_idx] = fold_perm

        fold_rows.append({
            "outer_fold": outer_i,
            "n_test": int(len(test_idx)),
            "n_train": int(len(train_idx)),
            "n_train_sessions": len(train_sessions),
            "selected_view": best_view_id,
            "selected_view_inner_acc": view_inner_acc[best_view_i],
            "probe_acc": float(fold_probe.mean()),
            "perm_floor_acc": float(fold_perm.mean()),
        })
        print(
            f"[{arm}] outer {outer_i + 1}/{len(outer_folds)}: probe={fold_probe.mean():.4f} "
            f"perm_floor={fold_perm.mean():.4f} view={best_view_id} "
            f"(inner={view_inner_acc[best_view_i]:.4f})",
            flush=True,
        )

    # --- floor (b): session prior on sample-level folds ---
    prior_correct = _session_prior_correctness(targets, sessions, candidates, n_outer, seed * 7 + 777)

    fold_accs = [row["probe_acc"] for row in fold_rows]
    fold_perms = [row["perm_floor_acc"] for row in fold_rows]
    probe_mean = float(np.mean(fold_accs))
    probe_sd = float(np.std(fold_accs, ddof=1)) if len(fold_accs) > 1 else 0.0
    perm_mean = float(np.mean(fold_perms))
    prior_mean = float(prior_correct.mean())
    floor_max = max(perm_mean, prior_mean)
    floor_estimator = "permutation" if perm_mean >= prior_mean else "session_prior"
    floor_per_sample = perm_correct if floor_estimator == "permutation" else prior_correct
    sensitivity_note = None
    if abs(perm_mean - prior_mean) > 0.10:
        sensitivity_note = (
            f"SENSITIVITY: permutation floor ({perm_mean:.4f}) and session-prior floor "
            f"({prior_mean:.4f}) disagree by {abs(perm_mean - prior_mean):.4f} > 0.10; "
            f"using MAX ({floor_estimator}) as the floor."
        )
        print(sensitivity_note, flush=True)

    diffs = probe_correct - floor_per_sample
    result.update({
        "outer_folds": fold_rows,
        "probe_acc_mean": probe_mean,
        "probe_acc_sd": probe_sd,
        "probe_acc_pooled": float(probe_correct.mean()),
        "floor_permutation_mean": perm_mean,
        "floor_session_prior_mean": prior_mean,
        "floor_max": floor_max,
        "floor_max_estimator": floor_estimator,
        "sensitivity_note": sensitivity_note,
        "probe_minus_floor": probe_mean - floor_max,
        "wilson95_upper_diff": _wilson_upper_diff(diffs),
        "n_perm": n_perm,
    })
    return result


# ---------------------------------------------------------------------------
# Gates + outputs
# ---------------------------------------------------------------------------

def _gate_rows(gist: Dict[str, Any], full: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    full_probe = full.get("probe_acc_mean", float("nan"))
    full_diff = full.get("probe_minus_floor", float("nan"))
    gist_probe = gist.get("probe_acc_mean", float("nan"))
    gist_diff = gist.get("probe_minus_floor", float("nan"))
    gist_upper = gist.get("wilson95_upper_diff", float("nan"))

    gate1_pass = bool(
        not math.isnan(full_diff)
        and full_diff >= 0.15
        and not math.isnan(full_probe)
        and full_probe >= 0.302
    )
    gate1_measured = (
        f"full probe={full_probe:.4f}, full floor={full.get('floor_max', float('nan')):.4f}, "
        f"diff={full_diff:.4f} (>=0.15: {'PASS' if full_diff >= 0.15 else 'FAIL'}); "
        f"probe>=0.302: {'PASS' if full_probe >= 0.302 else 'FAIL'}"
    )
    rows = [{
        "gate": "1 sensitivity precondition",
        "pre_registered": GATE1_TEXT,
        "measured": gate1_measured,
        "verdict": "PASS" if gate1_pass else VERDICT_INCONCLUSIVE,
    }]

    gate2_pass = bool(gate1_pass and not math.isnan(gist_diff) and gist_diff >= 0.10)
    gate2_measured = (
        f"gist probe={gist_probe:.4f}, gist floor={gist.get('floor_max', float('nan')):.4f}, "
        f"diff={gist_diff:.4f} (>=0.10: {'PASS' if not math.isnan(gist_diff) and gist_diff >= 0.10 else 'FAIL'})"
    )
    rows.append({
        "gate": "2 positive",
        "pre_registered": GATE2_TEXT,
        "measured": gate2_measured,
        "verdict": (
            VERDICT_ALIVE if gate2_pass
            else "not met" if gate1_pass
            else "not evaluated (gate 1 failed)"
        ),
    })

    gate3_measured = f"X = upper 95% CI bound of (gist - floor) = {gist_upper:.4f}"
    rows.append({
        "gate": "3 negative (fixed wording)",
        "pre_registered": GATE3_TEXT,
        "measured": gate3_measured,
        "verdict": (
            f"未检出线性可读动作信息（可判界<={gist_upper:.3f}）" if gate1_pass and not gate2_pass
            else "not triggered"
        ),
    })

    if not gate1_pass:
        final = VERDICT_INCONCLUSIVE
    elif gate2_pass:
        final = VERDICT_ALIVE
    else:
        final = f"未检出线性可读动作信息（可判界<={gist_upper:.3f}）"
    return rows, final


def _write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Gist readability probe",
        "",
        "Can the target tool NAME still be linearly read out from the compressed (gist)",
        "KV representation? Probe: candidate-scoring closed-set NEAREST-CLASS-MEAN",
        "linear probe with cosine similarity; validation: session-grouped nested CV;",
        "all feature choices were selected inside the inner CV only.",
        "",
        "## Arms",
        "",
        "| arm | n samples | n sessions | probe acc (mean ± sd) | perm floor | prior floor | floor (MAX) | probe - floor | Wilson95 upper(diff) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in ("gist", "full"):
        res = summary["arms"].get(arm, {})
        if "probe_acc_mean" not in res:
            lines.append(f"| {arm} | — | — | not evaluable | | | | | |")
            continue
        lines.append(
            f"| {arm} | {res.get('n_samples')} | {res.get('n_sessions')} | "
            f"{res['probe_acc_mean']:.4f} ± {res['probe_acc_sd']:.4f} | "
            f"{res['floor_permutation_mean']:.4f} | {res['floor_session_prior_mean']:.4f} | "
            f"{res['floor_max']:.4f} ({res['floor_max_estimator']}) | "
            f"{res['probe_minus_floor']:.4f} | {res['wilson95_upper_diff']:.4f} |"
        )
    for arm in ("gist", "full"):
        note = summary["arms"].get(arm, {}).get("sensitivity_note")
        if note:
            lines += ["", f"> {note}"]
    lines += [
        "",
        "## Selected views (per outer fold)",
        "",
        "| arm | outer fold | view | inner acc |",
        "| --- | --- | --- | --- |",
    ]
    for arm in ("gist", "full"):
        for fold in summary["arms"].get(arm, {}).get("outer_folds", []):
            lines.append(
                f"| {arm} | {fold['outer_fold']} | {fold['selected_view']} | "
                f"{fold['selected_view_inner_acc']:.4f} |"
            )
    lines += [
        "",
        "## Gates (pre-registered)",
        "",
        "| gate | pre-registered text | measured | verdict |",
        "| --- | --- | --- | --- |",
    ]
    for row in summary["gates"]:
        lines.append(
            f"| {row['gate']} | {row['pre_registered']} | {row['measured']} | {row['verdict']} |"
        )
    lines += [
        "",
        f"**Final verdict: {summary['final_verdict']}**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gist readability probe analysis (pure CPU).")
    parser.add_argument("--features_dir", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--n_outer", type=int, default=5)
    parser.add_argument("--n_inner", type=int, default=4)
    parser.add_argument("--n_perm", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    print("Pre-registered gates (verbatim):", flush=True)
    for text in (GATE1_TEXT, GATE2_TEXT, GATE3_TEXT):
        print(f"  {text}", flush=True)

    arms: Dict[str, Any] = {}
    for arm in ("gist", "full"):
        try:
            arms[arm] = _run_arm(arm, features_dir, args.n_outer, args.n_inner, args.n_perm, args.seed)
        except (FileNotFoundError, ValueError) as error:
            print(f"[{arm}] not evaluable: {error}", flush=True)
            arms[arm] = {"arm": arm, "error": str(error)}

    gate_rows, final_verdict = _gate_rows(arms.get("gist", {}), arms.get("full", {}))
    summary = {
        "features_dir": str(features_dir),
        "config": {
            "n_outer": args.n_outer,
            "n_inner": args.n_inner,
            "n_perm": args.n_perm,
            "seed": args.seed,
        },
        "gate_texts": [GATE1_TEXT, GATE2_TEXT, GATE3_TEXT],
        "arms": arms,
        "gates": gate_rows,
        "final_verdict": final_verdict,
    }

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    md_path = out_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(md_path, summary)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {md_path}", flush=True)
    print(f"Final verdict: {final_verdict}", flush=True)


if __name__ == "__main__":
    main()
