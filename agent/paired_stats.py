"""Torch-free paired-comparison statistics shared by the B/D/F pilot analyzers.

Why this module exists: ``agent/r4_paired.py`` holds the reference
implementations of the exact McNemar test and the session-cluster bootstrap,
but it imports ``eval_agent_tool_definition_c2kv`` at module scope, which
pulls in torch.  The pilot analyzers must stay importable on machines without
torch, so the two estimators are ported here verbatim (same arithmetic, same
RNG discipline) and r4_paired remains the historical reference.

Any change to the estimators here must keep ``test_paired_stats.py``'s
equivalence assertions against hand-computed values passing.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Hashable, List, Sequence, Tuple

__all__ = [
    "mcnemar_exact",
    "mcnemar_cells",
    "cluster_bootstrap_diff",
    "paired_rate_diff",
]


def mcnemar_cells(pairs: Sequence[Tuple[bool, bool]]) -> Tuple[int, int]:
    """Discordant cell counts (b, c) for paired binary outcomes.

    ``b`` counts pairs where the first arm succeeded and the second failed;
    ``c`` counts the reverse.  Concordant pairs carry no information for the
    exact test and are dropped.
    """
    b = sum(1 for a_ok, b_ok in pairs if a_ok and not b_ok)
    c = sum(1 for a_ok, b_ok in pairs if b_ok and not a_ok)
    return b, c


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p-value (port of r4_paired._mcnemar_exact)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def paired_rate_diff(pairs: Sequence[Tuple[bool, bool]]) -> float:
    """Point estimate of (arm A rate - arm B rate) over paired outcomes."""
    if not pairs:
        return 0.0
    return (
        sum(1 for a_ok, _ in pairs if a_ok) / len(pairs)
        - sum(1 for _, b_ok in pairs if b_ok) / len(pairs)
    )


def cluster_bootstrap_diff(
    pairs: Sequence[Tuple[bool, bool]],
    clusters: Sequence[Hashable],
    reps: int = 20000,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile 95% CI of the paired rate difference, resampling clusters.

    Port of r4_paired._cluster_bootstrap.  ``clusters`` is parallel to
    ``pairs`` and normally carries ``session_id`` -- items inside one session
    are not independent, so the bootstrap resamples whole sessions.

    Returns (point_estimate, ci_low, ci_high).  Callers must report the number
    of distinct clusters alongside the interval: with few clusters the
    interval is wide and unstable, and that limitation is reported rather than
    hidden.
    """
    if not pairs:
        return 0.0, 0.0, 0.0
    by_cluster: Dict[Hashable, List[Tuple[bool, bool]]] = defaultdict(list)
    for pair, cid in zip(pairs, clusters):
        by_cluster[cid].append(pair)
    groups = list(by_cluster.values())
    rng = random.Random(seed)
    diffs: List[float] = []
    for _ in range(reps):
        sample = [groups[rng.randrange(len(groups))] for _ in groups]
        flat = [p for grp in sample for p in grp]
        diffs.append(
            sum(1 for a_ok, _ in flat if a_ok) / len(flat)
            - sum(1 for _, b_ok in flat if b_ok) / len(flat)
        )
    diffs.sort()
    return (
        paired_rate_diff(pairs),
        diffs[int(0.025 * reps)],
        diffs[int(0.975 * reps)],
    )
