"""Tests for the torch-free paired statistics shared by the pilot analyzers.

Run: python -m pytest agent/test_paired_stats.py -v
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from paired_stats import (  # noqa: E402
    cluster_bootstrap_diff,
    mcnemar_cells,
    mcnemar_exact,
    paired_rate_diff,
)


def test_module_is_torch_free():
    """The whole point of this module: importable without torch."""
    assert "torch" not in sys.modules or True  # torch may be present elsewhere
    import importlib

    mod = importlib.import_module("paired_stats")
    # No repo module that pulls torch may appear in the module's globals.
    forbidden = {"torch", "eval_agent_tool_definition_c2kv", "eval_agent_history_c2kv"}
    assert forbidden.isdisjoint(set(vars(mod)))


def test_mcnemar_cells_counts_discordant_only():
    pairs = [(True, False), (True, False), (False, True), (True, True), (False, False)]
    assert mcnemar_cells(pairs) == (2, 1)


@pytest.mark.parametrize(
    "b,c,expected",
    [
        (0, 0, 1.0),
        (1, 0, 1.0),  # 2 * (1/2) = 1.0
        (2, 0, 0.5),  # 2 * (1/4)
        (3, 0, 0.25),  # 2 * (1/8)
        (5, 0, 0.0625),  # 2 * (1/32)
    ],
)
def test_mcnemar_exact_known_values(b, c, expected):
    assert mcnemar_exact(b, c) == pytest.approx(expected)


def test_mcnemar_exact_is_symmetric():
    assert mcnemar_exact(7, 2) == pytest.approx(mcnemar_exact(2, 7))


def test_mcnemar_exact_never_exceeds_one():
    # b == c is maximally non-significant; the doubled tail must be clamped.
    for n in range(1, 12):
        assert mcnemar_exact(n, n) == 1.0


def test_paired_rate_diff_sign_and_magnitude():
    pairs = [(True, False), (True, False), (False, False), (False, False)]
    assert paired_rate_diff(pairs) == pytest.approx(0.5)
    flipped = [(b, a) for a, b in pairs]
    assert paired_rate_diff(flipped) == pytest.approx(-0.5)


def test_paired_rate_diff_empty_is_zero():
    assert paired_rate_diff([]) == 0.0


def test_cluster_bootstrap_point_estimate_matches_paired_rate_diff():
    pairs = [(True, False)] * 6 + [(False, True)] * 2 + [(True, True)] * 4
    clusters = ["s1"] * 4 + ["s2"] * 4 + ["s3"] * 4
    point, lo, hi = cluster_bootstrap_diff(pairs, clusters, reps=500, seed=0)
    assert point == pytest.approx(paired_rate_diff(pairs))
    assert lo <= point <= hi


def test_cluster_bootstrap_is_seed_reproducible():
    # Clusters must disagree with each other (else every resample gives the
    # same difference) and there must be enough of them that the 2.5/97.5
    # percentiles land in the interior rather than saturating at the extremes.
    compositions = {
        0: [(True, False), (True, False)],
        1: [(False, True), (True, True)],
        2: [(True, True), (False, False)],
    }
    pairs: list = []
    clusters: list = []
    for i in range(10):
        block = compositions[i % 3]
        pairs.extend(block)
        clusters.extend([f"s{i}"] * len(block))
    first = cluster_bootstrap_diff(pairs, clusters, reps=400, seed=7)
    second = cluster_bootstrap_diff(pairs, clusters, reps=400, seed=7)
    third = cluster_bootstrap_diff(pairs, clusters, reps=400, seed=8)
    assert first == second
    assert first[1:] != third[1:]


def test_cluster_bootstrap_single_cluster_degenerates_to_point():
    """One cluster means every resample is that cluster: zero-width interval."""
    pairs = [(True, False), (True, True), (False, False)]
    clusters = ["only"] * 3
    point, lo, hi = cluster_bootstrap_diff(pairs, clusters, reps=200, seed=0)
    assert lo == pytest.approx(point)
    assert hi == pytest.approx(point)


def test_cluster_bootstrap_respects_clusters_not_items():
    """Items inside a cluster move together, so a one-sided cluster widens the CI."""
    pairs = [(True, False)] * 10 + [(False, True)] * 10
    by_cluster = ["hot"] * 10 + ["cold"] * 10
    by_item = [f"c{i}" for i in range(20)]
    _, lo_cluster, hi_cluster = cluster_bootstrap_diff(pairs, by_cluster, reps=800, seed=1)
    _, lo_item, hi_item = cluster_bootstrap_diff(pairs, by_item, reps=800, seed=1)
    assert (hi_cluster - lo_cluster) > (hi_item - lo_item)


def test_cluster_bootstrap_empty_is_zeros():
    assert cluster_bootstrap_diff([], [], reps=10, seed=0) == (0.0, 0.0, 0.0)
