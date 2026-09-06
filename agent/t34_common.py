# -*- coding: utf-8 -*-
"""t34 — shared conventions for the §4.5–4.12 trigger-detector migrations.

Every t34 module imports from here so that:

* labels come from ONE place (``t33_labels.build_label_frame`` on the frozen
  r2 battery + ``d_cw_manifest_r2.json``), never re-derived ad hoc;
* the metric code is tie-correct (distinct-threshold average precision,
  average-rank AUROC) and prevalence-aware (chance AP = prevalence of the
  evaluation frame, NOT the 900-frame base rate 0.1033);
* every confidence interval is a session-clustered bootstrap over the
  sessions PRESENT in the evaluation frame (100 clusters on the 161-row
  trigger subset, not "227");
* locators are scored with S@k against the frozen floor / oracle / ceiling
  (25.0 % / 71/93 / 81/93) with an exact one-sided binomial test and an
  inverted-score control;
* hyper-parameters are selected in INNER folds of a session-grouped nested
  CV; nothing is chosen on outer-fold test scores;
* features are written through ``t33_labels.guard_columns`` so no
  target/gold/scoring/full-arm column can leak into a feature frame.

Torch-free.  Depends on numpy, scipy, scikit-learn only.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from t33_labels import (  # noqa: E402
    build_label_frame,
    cw_label,
    guard_columns,
    join_arms,
    load_jsonl,
    parse_fail_baseline,
)

# --------------------------------------------------------------------------
# frozen assets
# --------------------------------------------------------------------------

#: Frozen D-line reference values the locators are scored against
#: (docs/research/k1_c3_locate_cost.md, transfer manual).  Never recomputed
#: here; a locator table prints them as fixed rows.
LOCATE_FLOOR_WRONG_BLOCK = 0.25          # wrong-block floor
LOCATE_WITNESS_HITS = (71, 93)           # witness-IDF k* gold-scored: 76.3 %
LOCATE_BESTK_HITS = (81, 93)             # best-k ceiling: 87.1 %
LOCATE_BEST_ARM_HITS = (75, 93)          # raw_erratum_tail: 80.6 %
BASE_RATE_900 = 93 / 900                 # 0.1033, the 900-frame rate (reference only)
MDE_PP = (17, 25)                        # minimum detectable effect at n=93


def session_of(qid: str) -> str:
    """qids are ``<session>:<step>``."""
    return qid.rsplit(":", 1)[0]


def step_index(qid: str) -> int:
    tail = qid.rsplit(":", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return -1


@dataclass
class FrozenAssets:
    """Loader for the frozen r2 battery + manifest (+ optional witness table)."""

    root: Path
    battery_full: Path = field(init=False)
    battery_c2kv: Path = field(init=False)
    manifest_path: Path = field(init=False)
    witness_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.battery_full = self.root / "results/bdf_pilot/d_r2/battery_full.jsonl"
        self.battery_c2kv = self.root / "results/bdf_pilot/d_r2/battery_c2kv.jsonl"
        self.manifest_path = self.root / "configs/bdf_pilot/d_cw_manifest_r2.json"
        self.witness_path = self.root / "configs/bdf_pilot/d_witness_r2.json"

    def load(self) -> "FrozenFrame":
        full_rows = load_jsonl(str(self.battery_full))
        c2kv_rows = load_jsonl(str(self.battery_c2kv))
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        pairs = join_arms(full_rows, c2kv_rows)
        labels = build_label_frame(pairs, manifest)
        witness = None
        if self.witness_path.exists():
            witness = json.loads(self.witness_path.read_text(encoding="utf-8"))
        return FrozenFrame(pairs=pairs, labels=labels, manifest=manifest, witness=witness)


@dataclass
class FrozenFrame:
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]]
    labels: List[Dict[str, Any]]
    manifest: Dict[str, Any]
    witness: Optional[Dict[str, Any]] = None

    @property
    def full_by_qid(self) -> Dict[str, Dict[str, Any]]:
        return {f["qid"]: f for f, _ in self.pairs}

    @property
    def c2kv_by_qid(self) -> Dict[str, Dict[str, Any]]:
        return {c["qid"]: c for _, c in self.pairs}

    @property
    def label_by_qid(self) -> Dict[str, Optional[int]]:
        return {r["qid"]: r["label_cw"] for r in self.labels}

    def trigger_subset(self) -> List[Dict[str, Any]]:
        """C->W (1) and C->C (0) rows only — the 161-row scoring frame."""
        return [r for r in self.labels if r["label_cw"] in (0, 1)]

    def cw_qids(self) -> List[str]:
        return [r["qid"] for r in self.labels if r["label_cw"] == 1]

    def cc_qids(self) -> List[str]:
        return [r["qid"] for r in self.labels if r["label_cw"] == 0]

    def witness_entry(self, qid: str) -> Optional[Dict[str, Any]]:
        if not self.witness:
            return None
        entries = self.witness.get("entries") or {}
        return entries.get(qid)

    def cap_tokens(self) -> int:
        return int((self.manifest.get("kv_recipe") or {}).get("max_new_tokens", 128))


# --------------------------------------------------------------------------
# metrics (tie-correct, prevalence-aware)
# --------------------------------------------------------------------------

def prevalence(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(y.mean()) if y.size else float("nan")


def average_precision(scores: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Distinct-threshold average precision (sklearn-equivalent).

    One threshold per DISTINCT score value; tied rows enter together, so the
    result does not depend on row order inside a tie block.  Higher score =
    more risk.  Returns None when the labels are one-class.
    """
    s = np.asarray(scores, dtype=float)
    yy = np.asarray(y, dtype=int)
    n_pos = int(yy.sum())
    if n_pos == 0 or n_pos == len(yy):
        return None
    order = np.argsort(-s, kind="mergesort")
    s = s[order]
    yy = yy[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    # last index of each tie block
    last = np.r_[s[1:] != s[:-1], True]
    tp_b = tp[last]
    fp_b = fp[last]
    prec = tp_b / np.maximum(1, tp_b + fp_b)
    rec = tp_b / n_pos
    rec_prev = np.r_[0.0, rec[:-1]]
    return float(np.sum(prec * (rec - rec_prev)))


def auroc(scores: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Mann–Whitney AUROC with average ranks for ties."""
    s = np.asarray(scores, dtype=float)
    yy = np.asarray(y, dtype=int)
    n_pos = int(yy.sum())
    n_neg = len(yy) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    from scipy.stats import rankdata
    ranks = rankdata(s, method="average")
    return float((ranks[yy == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def session_clusters(sessions: Sequence[str]) -> np.ndarray:
    uniq = {s: i for i, s in enumerate(sorted(set(sessions)))}
    return np.array([uniq[s] for s in sessions], dtype=int)


def clustered_bootstrap(
    metric: Callable[[np.ndarray, np.ndarray], Optional[float]],
    scores: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    reps: int = 2000,
    seed: int = 20260905,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float], int]:
    """Percentile CI of ``metric`` resampling CLUSTERS (sessions) with replacement.

    Returns (lo, hi, n_clusters).  Clusters are the ones present in the
    evaluation frame — on the 161-row trigger subset that is ~100, never 227.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {c: np.where(clusters == c)[0] for c in uniq}
    vals: List[float] = []
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in pick])
        m = metric(scores[idx], y[idx])
        if m is not None:
            vals.append(m)
    if not vals:
        return None, None, len(uniq)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), len(uniq)


def paired_delta_bootstrap(
    metric: Callable[[np.ndarray, np.ndarray], Optional[float]],
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    reps: int = 2000,
    seed: int = 20260905,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """(point, lo, hi) of metric(a) − metric(b) on the SAME cluster resamples."""
    rng = np.random.default_rng(seed)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    y = np.asarray(y, dtype=int)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = {c: np.where(clusters == c)[0] for c in uniq}
    ma, mb = metric(a, y), metric(b, y)
    if ma is None or mb is None:
        return None, None, None
    deltas: List[float] = []
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[c] for c in pick])
        xa, xb = metric(a[idx], y[idx]), metric(b[idx], y[idx])
        if xa is not None and xb is not None:
            deltas.append(xa - xb)
    if not deltas:
        return float(ma - mb), None, None
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(ma - mb), float(lo), float(hi)


def operating_point(scores: np.ndarray, y: np.ndarray, n_fires: int) -> Dict[str, Any]:
    """Fire on the top ``n_fires`` rows (ties broken by stable order).

    coverage = fires ∩ positives, false_resets = fires ∩ negatives,
    precision = coverage / fires.  The denominators (93 / 68) are the frame's
    own counts and are returned so a caller can print them.
    """
    s = np.asarray(scores, dtype=float)
    yy = np.asarray(y, dtype=int)
    order = np.argsort(-s, kind="mergesort")
    fire = np.zeros(len(s), dtype=bool)
    fire[order[: max(0, int(n_fires))]] = True
    cov = int((fire & (yy == 1)).sum())
    fr = int((fire & (yy == 0)).sum())
    n_f = int(fire.sum())
    return {
        "fires": n_f,
        "coverage": cov,
        "n_pos": int(yy.sum()),
        "false_resets": fr,
        "n_neg": int((yy == 0).sum()),
        "precision": (cov / n_f) if n_f else None,
        "threshold": float(s[order[n_f - 1]]) if n_f else None,
    }


def per_step_false_fire_rate(fires: np.ndarray, y: np.ndarray) -> Optional[float]:
    """Companion column to the per-episode false-reset rate (Quickest Detection
    migration): false fires / all negative steps."""
    f = np.asarray(fires, dtype=bool)
    yy = np.asarray(y, dtype=int)
    n_neg = int((yy == 0).sum())
    return float((f & (yy == 0)).sum() / n_neg) if n_neg else None


# --------------------------------------------------------------------------
# locators: S@k against the frozen floor / oracle / ceiling
# --------------------------------------------------------------------------

def exact_binom_one_sided(k: int, n: int, p0: float) -> float:
    """P[X >= k] under Binomial(n, p0) — one-sided test that hit rate > p0."""
    from scipy.stats import binom
    if n <= 0:
        return float("nan")
    return float(binom.sf(k - 1, n, p0))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    from scipy.stats import beta
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value on the discordant counts (b, c)."""
    from scipy.stats import binom
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return float(min(1.0, 2.0 * binom.cdf(k, n, 0.5)))


def locate_table(
    hits: Dict[str, Optional[bool]],
    *,
    floor: float = LOCATE_FLOOR_WRONG_BLOCK,
    label: str = "locator",
) -> Dict[str, Any]:
    """S@k table row for a locator.

    ``hits`` maps qid -> True (chosen block is the witness/flip block), False,
    or None (abstained / k*=None).  Abstentions are counted in the denominator
    as misses AND reported separately, never silently dropped.
    """
    n = len(hits)
    n_abstain = sum(1 for v in hits.values() if v is None)
    k = sum(1 for v in hits.values() if v is True)
    lo, hi = clopper_pearson(k, n)
    return {
        "locator": label,
        "hits": k,
        "n": n,
        "abstained": n_abstain,
        "s_at_k": (k / n) if n else None,
        "ci95": [lo, hi],
        "p_vs_floor": exact_binom_one_sided(k, n, floor),
        "floor": floor,
        "witness_oracle": LOCATE_WITNESS_HITS[0] / LOCATE_WITNESS_HITS[1],
        "bestk_ceiling": LOCATE_BESTK_HITS[0] / LOCATE_BESTK_HITS[1],
    }


def chooser_argmax(scores: Sequence[float]) -> Optional[int]:
    """The frozen ``d_witness_core.select_k_star`` chooser semantics for ANY
    per-block score vector: abstain (None) when no block scores above zero,
    ties resolve to the LOWEST index.  NaNs are treated as no score."""
    v = np.asarray(scores, dtype=float)
    if v.size == 0:
        return None
    v = np.where(np.isfinite(v), v, -np.inf)
    if not np.isfinite(v).any() or float(v.max()) <= 0.0:
        return None
    return int(np.argmax(v))  # np.argmax returns the first maximum


def chooser_argmin(scores: Sequence[float]) -> Optional[int]:
    """Mirror of :func:`chooser_argmax` for the inverted-score control:
    lowest-index minimum among finite scores; abstain when the vector is
    constant (an inverted constant score has no ranking to invert)."""
    v = np.asarray(scores, dtype=float)
    if v.size == 0:
        return None
    fin = np.isfinite(v)
    if not fin.any() or float(v[fin].max()) == float(v[fin].min()):
        return None
    v = np.where(fin, v, np.inf)
    return int(np.argmin(v))


def inverted_score_control(score_vectors: Dict[str, Sequence[float]],
                           truth: Dict[str, Optional[int]]) -> Dict[str, Any]:
    """CausalCache's specificity control: run the SAME chooser with the score
    negated.  If argmin does not do significantly worse than argmax the score
    is a rendering artefact.  ``truth`` maps qid -> the reference block index
    (None = no reference)."""
    fwd: Dict[str, Optional[bool]] = {}
    inv: Dict[str, Optional[bool]] = {}
    for qid, vec in score_vectors.items():
        ref = truth.get(qid)
        v = np.asarray(vec, dtype=float)
        if ref is None or v.size == 0 or not np.isfinite(v).any():
            fwd[qid] = None
            inv[qid] = None
            continue
        k_fwd = chooser_argmax(v)
        k_inv = chooser_argmin(v)
        fwd[qid] = None if k_fwd is None else bool(k_fwd == ref)
        inv[qid] = None if k_inv is None else bool(k_inv == ref)
    b = sum(1 for q in fwd if fwd[q] is True and inv[q] is not True)
    c = sum(1 for q in fwd if inv[q] is True and fwd[q] is not True)
    return {
        "forward": locate_table(fwd, label="argmax"),
        "inverted": locate_table(inv, label="argmin"),
        "mcnemar_p": mcnemar_exact(b, c),
    }


# --------------------------------------------------------------------------
# nested, session-grouped cross-validation
# --------------------------------------------------------------------------

def grouped_folds(groups: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    """Deterministic group-level folds: returns a list of boolean masks."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    perm = rng.permutation(uniq)
    masks = []
    for f in range(n_folds):
        held = set(perm[f::n_folds].tolist())
        masks.append(np.array([g in held for g in groups]))
    return masks


def nested_cv_logistic(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    c_grid: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0),
    outer_folds: int = 5,
    inner_folds: int = 3,
    seed: int = 20260905,
    class_weight: Optional[str] = None,
    select_by: str = "auroc",
    extra_selectors: Optional[Dict[str, Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]] = None,
) -> Dict[str, Any]:
    """Session-grouped nested CV for an L2 logistic probe.

    * Standardisation is fit on the outer-train fold only.
    * ``C`` (and, if ``extra_selectors`` is given, any other discrete knob such
      as a layer / head subset) is chosen on INNER folds of the outer-train
      fold by ``select_by`` (auroc | ap).  Nothing is chosen on outer test.
    * Returns out-of-fold scores (one per row), chosen knobs per fold, and the
      pooled OOF metrics.  Rows with NaN features are dropped and reported.

    ``extra_selectors``: name -> callable(X_train, X_test) returning the
    column-subset views to try; the winner is picked with C jointly.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    keep = np.isfinite(X).all(axis=1)
    dropped = int((~keep).sum())
    X, y, groups = X[keep], y[keep], groups[keep]
    oof = np.full(len(y), np.nan)
    chosen: List[Dict[str, Any]] = []
    metric = auroc if select_by == "auroc" else average_precision
    selectors = extra_selectors or {"all": (lambda a, b: (a, b))}

    for outer_mask in grouped_folds(groups, outer_folds, seed):
        tr, te = ~outer_mask, outer_mask
        if y[tr].sum() == 0 or y[tr].sum() == tr.sum() or te.sum() == 0:
            continue
        best = None
        for sel_name, sel in selectors.items():
            for C in c_grid:
                inner_scores = np.full(int(tr.sum()), np.nan)
                Xtr, ytr, gtr = X[tr], y[tr], groups[tr]
                for inner_mask in grouped_folds(gtr, inner_folds, seed + 1):
                    itr, ite = ~inner_mask, inner_mask
                    if ytr[itr].sum() == 0 or ytr[itr].sum() == itr.sum() or ite.sum() == 0:
                        continue
                    a, b = sel(Xtr[itr], Xtr[ite])
                    sc = StandardScaler().fit(a)
                    clf = LogisticRegression(C=C, solver="liblinear", max_iter=2000,
                                             class_weight=class_weight).fit(sc.transform(a), ytr[itr])
                    inner_scores[ite] = clf.decision_function(sc.transform(b))
                ok = np.isfinite(inner_scores)
                m = metric(inner_scores[ok], ytr[ok]) if ok.sum() and ytr[ok].sum() not in (0, ok.sum()) else None
                if m is not None and (best is None or m > best[0]):
                    best = (m, sel_name, C)
        if best is None:
            continue
        _, sel_name, C = best
        a, b = selectors[sel_name](X[tr], X[te])
        sc = StandardScaler().fit(a)
        clf = LogisticRegression(C=C, solver="liblinear", max_iter=2000,
                                 class_weight=class_weight).fit(sc.transform(a), y[tr])
        oof[te] = clf.decision_function(sc.transform(b))
        chosen.append({"selector": sel_name, "C": C, "inner_metric": float(best[0]),
                       "n_test": int(te.sum())})
    ok = np.isfinite(oof)
    return {
        "oof_scores": oof,
        "labels": y,
        "groups": groups,
        "scored_mask": ok,
        "n_scored": int(ok.sum()),
        "n_dropped_nan": dropped,
        "n_pos_scored": int(y[ok].sum()),
        "prevalence": prevalence(y[ok]) if ok.sum() else None,
        "auprc": average_precision(oof[ok], y[ok]) if ok.sum() else None,
        "auroc": auroc(oof[ok], y[ok]) if ok.sum() else None,
        "chosen": chosen,
    }


def permutation_band(
    fit_fn: Callable[[np.ndarray], float],
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_perm: int = 200,
    seed: int = 20260905,
) -> Dict[str, Any]:
    """Permutation null for a probe metric.  Labels are permuted WITHIN
    session-preserving structure by permuting whole sessions' label vectors
    across sessions of equal size when possible, else globally; ``fit_fn``
    must run the ENTIRE pipeline (including any head/layer search) on the
    permuted labels — that is what makes the band honest."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    vals: List[float] = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        m = fit_fn(yp)
        if m is not None and np.isfinite(m):
            vals.append(float(m))
    if not vals:
        return {"n": 0}
    arr = np.asarray(vals)
    return {"n": len(arr), "mean": float(arr.mean()), "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99))}


# --------------------------------------------------------------------------
# thresholds: fixed fire rate, matched to a baseline fire count
# --------------------------------------------------------------------------

def fixed_rate_threshold(scores: np.ndarray, q: float) -> float:
    """σ-abstain style: fire on the ``q`` fraction with the highest risk."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan")
    return float(np.quantile(s, 1.0 - q))


# --------------------------------------------------------------------------
# feature frames, freezing, hashing
# --------------------------------------------------------------------------

META_COLS = {"qid", "arm", "session_id", "mode", "ratio"}


def write_features_jsonl(path: Path, rows: Iterable[Dict[str, Any]], *, context: str) -> int:
    """Write one row per qid.  Column names go through the leakage guard;
    None values are KEPT (json null) so missingness stays visible — the
    scorer must report ``n_scored`` rather than silently imputing."""
    rows = list(rows)
    cols = sorted({k for r in rows for k in r} - META_COLS)
    guard_columns(cols, context=context)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    with open(path, "rb") as fh:
        return sha256_bytes(fh.read())


def freeze_json(path: Path, obj: Any) -> str:
    """Write canonical JSON (sorted keys, LF) and return its sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    with io.open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return sha256_bytes(text.encode("utf-8"))


def load_orientations(path: Optional[Path] = None) -> Dict[str, int]:
    """Pre-declared risk orientations for t34 features (+1 = higher is riskier)."""
    p = Path(path) if path else _HERE.parent / "configs/t34/orientations.json"
    if not p.exists():
        return {}
    out: Dict[str, int] = {}
    for k, v in json.loads(p.read_text(encoding="utf-8")).items():
        # "_..." keys and non-integer values are declarations' metadata
        if k.startswith("_") or isinstance(v, bool) or not isinstance(v, (int, str)):
            continue
        try:
            out[k] = int(v)
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------
# server-side sidecars the t34 modules consume (schemas fixed here)
# --------------------------------------------------------------------------

def load_decoded_docs(path: Path) -> Dict[str, List[str]]:
    """``{qid, docs: [decoded grid row text, ...]}`` per line — the text the
    model actually saw (post max_doc_length truncation, post chat template),
    dumped server-side by the witness selector.  Returns qid -> [text_k]."""
    out: Dict[str, List[str]] = {}
    for r in load_jsonl(str(path)):
        out[r["qid"]] = list(r.get("docs") or [])
    return out


def check_docs_against_witness(docs: Dict[str, List[str]], frame: FrozenFrame) -> Dict[str, Any]:
    """Cross-check decoded doc texts against the frozen witness table's
    per-doc sha256 so a stale dump cannot be scored silently."""
    bad: List[str] = []
    n_checked = 0
    for qid, texts in docs.items():
        ent = frame.witness_entry(qid)
        if not ent:
            continue
        shas = ent.get("doc_text_sha256") or []
        if len(shas) != len(texts):
            bad.append(qid)
            continue
        for s, t in zip(shas, texts):
            n_checked += 1
            if sha256_bytes(t.encode("utf-8")) != s:
                bad.append(qid)
                break
    return {"n_checked_docs": n_checked, "mismatched_qids": sorted(set(bad))}


def load_flip_table(path: Path) -> Dict[str, Dict[int, bool]]:
    """Per-(qid, k) repair outcome from the D-line k-sweep.  Two row shapes
    are accepted:

    * the reduced form ``{qid, k, correct}``;
    * the RAW sweep rows the server holds (``~/bench_results/d_v2/
      d_ksweep_r2.jsonl``: 928 rows over the 93 C->W qids, arm
      ``raw_keepG_sweep``), where ``k`` is ``d_ksweep_k`` (== the block index
      ``d_corr_doc_index`` on every row) and ``correct`` is ``tool_name_match``
      -- the same reading ``agent/d_ksweep_analysis.py`` uses.  Rows flagged
      ``skipped`` are dropped, as there.

    Returns qid -> {k: correct}."""
    out: Dict[str, Dict[int, bool]] = {}
    for r in load_jsonl(str(path)):
        if r.get("skipped"):
            continue
        if "k" in r:
            k = int(r["k"])
        elif r.get("d_ksweep_k") is not None or r.get("d_corr_doc_index") is not None:
            k = int(r["d_ksweep_k"] if r.get("d_ksweep_k") is not None
                    else r["d_corr_doc_index"])
        else:
            raise ValueError(f"flip table row without k / d_ksweep_k: {sorted(r)[:8]}")
        correct = r["correct"] if "correct" in r else r.get("tool_name_match")
        out.setdefault(r["qid"], {})[k] = bool(correct)
    return out


def proxy_backend_census(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Count of the ``backend`` field over proxy rows (``"missing"`` when absent)."""
    out: Dict[str, int] = {}
    for r in rows:
        key = r.get("backend")
        key = "missing" if key in (None, "") else str(key)
        out[key] = out.get(key, 0) + 1
    return out


def load_proxy_log(path: Path, expect_backend: Optional[str] = None) -> List[Dict[str, Any]]:
    """bench proxy request log (one JSON row per request; fields such as
    arm / conv_id / turn / fp / status / error_kind / finish_reason / usage /
    action / match / diverged_now / re_diverged / tracking_lost ...).

    BACKEND GUARD (2026-09-06 ruling: the bench side uses the sglang serving
    backend only; the hf_server backend is not to be used).  Rows are checked
    against ``expect_backend`` -- default ``"sglang"``, overridable with the
    env var ``T34_PROXY_BACKEND`` (``any`` disables the guard).  On the server
    the sglang runs stamp ``backend: "sglang"``; the older hf_server-era logs
    (``proxy_task_{bx,f2*,f3,f4,hr*,up*,z4}_*.jsonl``) carry no backend field at
    all, so a missing field is treated as NOT sglang and refused with the census.
    """
    rows = load_jsonl(str(path))
    want = expect_backend if expect_backend is not None else os.environ.get("T34_PROXY_BACKEND", "sglang")
    if want and want != "any" and rows:
        census = proxy_backend_census(rows)
        bad = {k: v for k, v in census.items() if k != want}
        if bad:
            raise ValueError(
                f"{path}: proxy log backend census {census} is not all {want!r}; the bench face "
                "reads sglang-backend runs only (hf_server logs are refused).  Set "
                "T34_PROXY_BACKEND=any to override deliberately and say so in the report.")
    return rows


# --------------------------------------------------------------------------
# small text helpers shared by the deterministic modules
# --------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def normalize_ws(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def json_leaves(value: Any) -> List[Any]:
    """JSON leaves (typed, not stringified) depth-first; keys excluded."""
    if isinstance(value, dict):
        out: List[Any] = []
        for v in value.values():
            out.extend(json_leaves(v))
        return out
    if isinstance(value, list):
        out = []
        for v in value:
            out.extend(json_leaves(v))
        return out
    return [value]


def first_divergence_index(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    """min{j : a_j != b_j} (0-based) or None when one is a prefix of the other
    up to min length (callers must pair this with cap_hit, never merge)."""
    for j, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return j
    return None
