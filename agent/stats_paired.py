#!/usr/bin/env python
"""Shared paired-stats helpers for the round-2 archive forensics scripts.

The closed-form stats (Wilson, McNemar, Newcombe, AUROC) are pure stdlib so
this module imports cleanly on a minimal CPU box; numpy is imported lazily
only inside the cluster-bootstrap helpers (numpy is available on the eval
server, scipy/pandas are not assumed).

Constants and formulas (Z_975, Z_80, MDE) match
agent/analyze_s4_forced_prefix.py so forensics reports stay comparable with
the pre-registered S4 paired analysis.
"""
from __future__ import annotations

import math
import subprocess
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

Z_975 = 1.959964
Z_80 = 0.841621


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar(b: int, c: int) -> Tuple[float, float, str]:
    """McNemar test on the discordant counts b and c.

    Exact two-sided binomial when b + c <= 25, otherwise chi-square with
    continuity correction (p from the normal tail, chi2_1 == z**2, so no
    scipy dependency). Returns (statistic, p_value, method).
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0, "empty"
    if n <= 25:
        stat = float(min(b, c))
        p = 2.0 * sum(math.comb(n, i) for i in range(int(stat) + 1)) / 2**n
        return stat, min(1.0, p), "exact_binomial"
    stat = (abs(b - c) - 1) ** 2 / n
    return stat, math.erfc(math.sqrt(stat / 2)), "chi2_cc"


def paired_binary_diff_ci(
    n11: int,
    n10: int,
    n01: int,
    n00: int,
    z: float = Z_975,
) -> Dict[str, Any]:
    """95% CI for the difference of paired proportions (arm x minus arm y).

    Newcombe method 10 (square-and-add of the Wilson limits of the two
    marginal rates); counts follow the (x, y) convention, n10 = x=1,y=0.
    A simple Wald interval on the paired difference (var = (psi - d**2)/n)
    is included as fallback and used when the table is empty.
    """
    n = n11 + n10 + n01 + n00
    if n == 0:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "method": "wald_fallback"}
    p1 = (n11 + n10) / n
    p2 = (n11 + n01) / n
    diff = p1 - p2
    psi_hat = (n10 + n01) / n
    # Wald fallback on the paired difference.
    se = math.sqrt(max(0.0, psi_hat - diff**2) / n)
    wald_lo = max(-1.0, diff - z * se)
    wald_hi = min(1.0, diff + z * se)
    # Newcombe method 10.
    l1, u1 = wilson_ci(n11 + n10, n, z)
    l2, u2 = wilson_ci(n11 + n01, n, z)
    lo = diff - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = diff + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return {
        "diff": diff,
        "lo": max(-1.0, lo),
        "hi": min(1.0, hi),
        "method": "newcombe10",
        "wald_lo": wald_lo,
        "wald_hi": wald_hi,
    }


def psi(n10: int, n01: int, n_pairs: int) -> float:
    """Observed discordance rate: fraction of pairs where the two arms differ."""
    if n_pairs <= 0:
        return 0.0
    return (n10 + n01) / n_pairs


discordance_rate = psi


def mde(psi_value: float, n_pairs: int) -> float:
    """Minimum detectable effect at 80% power: (z_0.975 + z_0.80) * sqrt(psi / n)."""
    if n_pairs <= 0:
        return 0.0
    return (Z_975 + Z_80) * math.sqrt(psi_value / n_pairs)


def _cluster_groups(clusters: Sequence[Hashable]) -> List[List[int]]:
    """Row indices grouped by cluster id (e.g. session id), deterministically ordered."""
    by_cluster: Dict[Hashable, List[int]] = {}
    for index, cluster in enumerate(clusters):
        by_cluster.setdefault(cluster, []).append(index)
    return [by_cluster[key] for key in sorted(by_cluster, key=str)]


def cluster_bootstrap_ci(
    diffs: Sequence[float],
    clusters: Sequence[Hashable],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """Percentile CI for the mean of per-sample diffs, resampling whole clusters.

    Clusters (session ids) are resampled with replacement to respect the
    within-session correlation of eval samples. Requires numpy (lazy import).
    Returns (lo, hi), or (None, None) when there is nothing to resample.
    """
    import numpy as np

    diffs_arr = np.asarray(list(diffs), dtype=float)
    if diffs_arr.size == 0:
        return None, None
    groups = [np.asarray(indices, dtype=int) for indices in _cluster_groups(clusters)]
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot, dtype=float)
    for boot in range(n_boot):
        picks = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[pick] for pick in picks])
        estimates[boot] = float(diffs_arr[sample].mean())
    lo, hi = np.percentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """AUROC via the rank (Mann-Whitney U) statistic with tie-averaged ranks."""
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    n_pos = sum(1 for _, label in pairs if label == 1)
    n_neg = len(pairs) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum_pos = 0.0
    index = 0
    while index < len(pairs):
        end = index
        while end + 1 < len(pairs) and pairs[end + 1][0] == pairs[index][0]:
            end += 1
        mean_rank = (index + end) / 2 + 1  # 1-based average rank of the tie block
        for j in range(index, end + 1):
            if pairs[j][1] == 1:
                rank_sum_pos += mean_rank
        index = end + 1
    u_pos = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u_pos / (n_pos * n_neg)


def auroc_bootstrap_ci(
    scores: Sequence[float],
    labels: Sequence[int],
    clusters: Sequence[Hashable],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Tuple[Optional[float], Optional[float]]:
    """Cluster-aware percentile bootstrap CI for AUROC.

    Resamples clusters with replacement; resamples containing a single class
    are skipped. Requires numpy (lazy import). Returns (lo, hi), or
    (None, None) when no valid resample exists.
    """
    import numpy as np

    scores = list(scores)
    labels = list(labels)
    if not scores:
        return None, None
    groups = _cluster_groups(clusters)
    rng = np.random.default_rng(seed)
    estimates: List[float] = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(groups), size=len(groups))
        indices = [index for pick in picks for index in groups[pick]]
        value = auroc([scores[i] for i in indices], [labels[i] for i in indices])
        if value is not None:
            estimates.append(value)
    if not estimates:
        return None, None
    lo, hi = np.percentile(
        np.asarray(estimates, dtype=float), [100 * alpha / 2, 100 * (1 - alpha / 2)]
    )
    return float(lo), float(hi)


def as_bool(value: Any) -> Optional[bool]:
    """Tolerant bool coercion for jsonl fields (handles 'False'/'true'/0/1)."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "t"}:
            return True
        if lowered in {"false", "0", "no", "n", "f", ""}:
            return False
    return None


def first_present(row: Dict[str, Any], aliases: Sequence[str]) -> Tuple[Optional[str], Any]:
    """Return (key, value) of the first alias present in the row, else (None, None)."""
    for key in aliases:
        if key in row:
            return key, row[key]
    return None, None


def fmt_prop(k: int, n: int, digits: int = 4) -> str:
    """'0.1234 [0.1000,0.1500] (k/n)' with a Wilson 95% CI."""
    if n <= 0:
        return "n/a (0/0)"
    lo, hi = wilson_ci(k, n, z=Z_975)
    return f"{k / n:.{digits}f} [{lo:.{digits}f},{hi:.{digits}f}] ({k}/{n})"


def fmt_ci(value: Optional[float], lo: Optional[float], hi: Optional[float], digits: int = 4) -> str:
    """'0.1234 [0.1000,0.1500]' for a precomputed interval; tolerates None."""
    if value is None or lo is None or hi is None:
        return "n/a"
    return f"{value:.{digits}f} [{lo:.{digits}f},{hi:.{digits}f}]"


def fmt_p(p: Optional[float]) -> str:
    """Compact p-value rendering for report tables."""
    if p is None:
        return "n/a"
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def git_commit(cwd: Optional[str] = None) -> str:
    """Current git commit hash for report provenance; 'unknown' on any failure."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"
