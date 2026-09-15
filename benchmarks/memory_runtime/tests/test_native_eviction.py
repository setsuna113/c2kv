"""Focused CPU contracts for offline native eviction diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from benchmarks.memory_runtime.native_eviction import (
    _prefix_plan,
    gather_selected_cache,
    select_history_indices,
)


def test_prefix_plan_exposes_no_future_query_and_charges_recent_to_budget() -> None:
    plan = _prefix_plan(
        prefix_tokens=20,
        history_start=4,
        history_budget_tokens=6,
        recent_window=64,
    )

    assert plan["shared_prefix_range"] == [0, 4]
    assert plan["eligible_history_range"] == [4, 20]
    assert plan["query_indices"] == list(range(4, 20))
    assert plan["recent_indices"] == plan["query_indices"]
    assert max(plan["query_indices"]) < plan["prefix_tokens"]
    assert len(plan["recent_indices"]) == 16
    assert plan["kept_history_tokens"] == 6

    larger_budget = _prefix_plan(
        prefix_tokens=20,
        history_start=4,
        history_budget_tokens=12,
        recent_window=64,
    )
    assert larger_budget["query_indices"] == plan["query_indices"]


@pytest.mark.parametrize("method", ["snapkv_style", "h2o_style"])
def test_selector_obeys_exact_history_budget_and_original_coordinates(method: str) -> None:
    selected = select_history_indices(
        [0.1, 0.9, 0.2, 0.3, 0.8, 0.4],
        candidate_indices=[10, 11, 12, 13, 14, 15],
        recent_indices=[16, 17],
        history_budget_tokens=5,
        method=method,
    )

    assert len(selected) == 5
    assert tuple(sorted(selected)) == selected
    assert {16, 17} <= set(selected)
    assert set(selected) <= set(range(10, 18))


def test_snapkv_style_pooling_and_h2o_style_can_select_different_tokens() -> None:
    scores = [0.0, 0.0, 1.0, 0.0, 0.0, 0.8, 0.0, 0.0]
    coordinates = list(range(10, 18))
    snap = select_history_indices(
        scores,
        candidate_indices=coordinates,
        recent_indices=[18],
        history_budget_tokens=3,
        method="snapkv_style",
    )
    h2o = select_history_indices(
        scores,
        candidate_indices=coordinates,
        recent_indices=[18],
        history_budget_tokens=3,
        method="h2o_style",
    )

    assert snap != h2o
    assert h2o == (12, 15, 18)
    assert snap == (13, 14, 18)
    assert snap[-1] == h2o[-1] == 18


def test_head_specific_gather_preserves_each_heads_original_coordinates() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    keys = torch.tensor(
        [[
            [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
            [[10.0], [11.0], [12.0], [13.0], [14.0], [15.0]],
        ]]
    )
    values = keys + 100.0
    cache = transformers.DynamicCache([(keys, values)])
    selected = gather_selected_cache(
        cache,
        [[(2, 5), (3, 4)]],
        history_start=2,
        config=None,
    )

    assert selected.get_seq_length() == 4
    assert selected.layers[0].keys[0, 0, :, 0].tolist() == [0.0, 1.0, 2.0, 5.0]
    assert selected.layers[0].keys[0, 1, :, 0].tolist() == [10.0, 11.0, 13.0, 14.0]
    assert selected.layers[0].values[0, 0, :, 0].tolist() == [100.0, 101.0, 102.0, 105.0]


def test_budget_smaller_than_query_window_keeps_only_newest_budget_tokens() -> None:
    selected = select_history_indices(
        [1.0],
        candidate_indices=[3],
        recent_indices=[4, 5, 6],
        history_budget_tokens=2,
        method="h2o_style",
    )

    assert selected == (5, 6)
