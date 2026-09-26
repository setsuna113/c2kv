"""CPU tests for request-bound backend capacity constraints."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.backend_capacity import (
    BackendCapacityConstraints,
    capacity_scope,
    current_constraints,
)


def constraints(**changes):
    fields = {
        "session_id": "task-155",
        "decision_key": "turn-1/step-1",
        "stage": "regeneration",
        "history_budget_tokens": 256,
        "mandatory_history_tokens": 28,
        "mandatory_source_indices": (3, 5),
        "release": "replaced_source_message",
        "provenance": "engine_held_checkpoint_receipt",
    }
    fields.update(changes)
    return BackendCapacityConstraints(**fields)


def test_any_replaced_source_releases_the_entire_held_window():
    held = constraints()
    assert held.minimum_history_tokens(()) == 28
    assert held.minimum_history_tokens((1, 2)) == 28
    assert held.minimum_history_tokens((5,)) == 0
    assert held.minimum_history_tokens((3, 5)) == 0
    assert held.minimum_history_tokens((3,)) == 0
    assert 232 + held.minimum_history_tokens((1,)) > held.history_budget_tokens
    assert 232 + held.minimum_history_tokens((5,)) <= held.history_budget_tokens


def test_never_release_preserves_the_pending_floor():
    current = constraints(stage="draft", release="never",
                          provenance="engine_current_resident_receipt")
    assert current.minimum_history_tokens((3, 5)) == 28
    assert 200 + current.minimum_history_tokens(()) <= current.history_budget_tokens
    assert 232 + current.minimum_history_tokens(()) > current.history_budget_tokens


def test_context_is_bound_to_session_decision_and_stage_and_restored():
    outer = constraints()
    inner = constraints(session_id="task-156", decision_key="turn-2/step-0", stage="draft")
    assert current_constraints("task-155") is None
    with capacity_scope(outer):
        assert current_constraints("task-155", "turn-1/step-1", "regeneration") is outer
        assert current_constraints("task-156") is None
        assert current_constraints("task-155", "turn-2/step-0") is None
        assert current_constraints("task-155", stage="draft") is None
        with capacity_scope(inner):
            assert current_constraints("task-155") is None
            assert current_constraints("task-156", "turn-2/step-0", "draft") is inner
            with capacity_scope(None):
                assert current_constraints("task-156") is None
            assert current_constraints("task-156") is inner
        assert current_constraints("task-155") is outer
    assert current_constraints("task-155") is None


def test_context_resets_after_error():
    with pytest.raises(RuntimeError, match="failed"):
        with capacity_scope(constraints()):
            raise RuntimeError("failed")
    assert current_constraints("task-155") is None


@pytest.mark.parametrize("changes", [
    {"session_id": ""},
    {"decision_key": ""},
    {"stage": "commit"},
    {"history_budget_tokens": True},
    {"history_budget_tokens": 0},
    {"mandatory_history_tokens": -1},
    {"mandatory_history_tokens": False},
    {"mandatory_source_indices": (True,)},
    {"mandatory_source_indices": (-1,)},
    {"mandatory_source_indices": (3, 3)},
    {"release": "partial"},
    {"provenance": ""},
])
def test_invalid_constraints_are_rejected(changes):
    with pytest.raises(ValueError):
        constraints(**changes)


def test_invalid_lookup_and_source_indices_are_rejected():
    held = constraints()
    with pytest.raises(ValueError):
        held.minimum_history_tokens((True,))
    with pytest.raises(ValueError):
        current_constraints("")
    with pytest.raises(ValueError):
        current_constraints("task-155", decision_key="")
    with pytest.raises(ValueError):
        current_constraints("task-155", stage="unknown")
