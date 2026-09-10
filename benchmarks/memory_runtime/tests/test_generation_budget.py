from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from benchmarks.memory_runtime.generation_budget import (
    GenerationBudget,
    GenerationBudgetExceeded,
    TaskGenerationBudgetExceeded,
)


def test_reservations_are_atomic_at_the_process_limit():
    budget = GenerationBudget(64)

    def reserve_once():
        try:
            return budget.reserve()
        except GenerationBudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: reserve_once(), range(128)))

    assert sorted(index for index in results if index is not None) == list(range(1, 65))
    assert results.count(None) == 64
    assert budget.consumed == 64


def test_request_metadata_uses_process_snapshots_and_owned_indices():
    budget = GenerationBudget(2)
    with budget.request_scope() as first:
        assert budget.reserve() == 1
        assert first.metadata() == {
            "limit": 2,
            "consumed_before": 0,
            "consumed_after": 1,
            "attempt_indices": [1],
        }
    with budget.request_scope() as second:
        assert budget.reserve() == 2
        with pytest.raises(GenerationBudgetExceeded):
            budget.reserve()
        assert second.metadata() == {
            "limit": 2,
            "consumed_before": 1,
            "consumed_after": 2,
            "attempt_indices": [2],
        }


def test_disabled_budget_keeps_metadata_absent():
    budget = GenerationBudget(None)
    with budget.request_scope() as request:
        assert budget.reserve() == 1
        assert request.metadata() is None


def test_per_task_limit_is_atomic_and_independent_of_process_limit():
    budget = GenerationBudget(5, per_task_limit=2)

    assert budget.reserve("task-a") == 1
    assert budget.reserve("task-a") == 2
    with pytest.raises(TaskGenerationBudgetExceeded) as error:
        budget.reserve("task-a")
    assert error.value.kind == "generation_task_budget_exhausted"
    assert error.value.task_id == "task-a"
    assert budget.reserve("task-b") == 3
    assert budget.consumed_for_task("task-a") == 2
    assert budget.consumed_for_task("task-b") == 1
    assert budget.consumed == 3


def test_per_task_metadata_binds_request_to_task_counter():
    budget = GenerationBudget(4, per_task_limit=2)
    with budget.request_scope() as first:
        assert budget.reserve("task-a") == 1
        assert budget.reserve("task-a") == 2
        assert first.metadata() == {
            "limit": 4,
            "consumed_before": 0,
            "consumed_after": 2,
            "attempt_indices": [1, 2],
            "per_task_limit": 2,
            "task_id": "task-a",
            "task_consumed_before": 0,
            "task_consumed_after": 2,
        }
    with budget.request_scope() as second:
        assert budget.reserve("task-b") == 3
        assert second.metadata()["task_consumed_before"] == 0
        assert second.metadata()["task_consumed_after"] == 1


def test_per_task_limit_requires_task_identity_before_reservation():
    budget = GenerationBudget(None, per_task_limit=2)
    with pytest.raises(ValueError, match="task_id must be a non-empty string"):
        budget.reserve()
    assert budget.consumed == 0


def test_one_request_cannot_charge_two_tasks_or_consume_on_rejection():
    budget = GenerationBudget(4, per_task_limit=2)
    with budget.request_scope():
        assert budget.reserve("task-a") == 1
        with pytest.raises(RuntimeError, match="multiple task IDs"):
            budget.reserve("task-b")
    assert budget.consumed == 1
    assert budget.consumed_for_task("task-a") == 1
    assert budget.consumed_for_task("task-b") == 0
