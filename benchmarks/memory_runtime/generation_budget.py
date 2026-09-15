"""Thread-safe process-wide accounting for chat generation attempts."""

from __future__ import annotations

import argparse
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional


def positive_generation_limit(value: str) -> int:
    """Parse a strictly positive generation-attempt limit for argparse."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


class GenerationBudgetExceeded(RuntimeError):
    """Raised before transport when the process-wide limit is exhausted."""

    kind = "generation_budget_exhausted"

    def __init__(self, limit: int):
        super().__init__(f"generation attempt budget exhausted ({limit}/{limit})")
        self.limit = limit


class TaskGenerationBudgetExceeded(GenerationBudgetExceeded):
    """Raised before transport when one task exhausts its independent cap."""

    kind = "generation_task_budget_exhausted"

    def __init__(self, task_id: str, limit: int):
        RuntimeError.__init__(
            self,
            f"generation attempt budget exhausted for task {task_id!r} "
            f"({limit}/{limit})",
        )
        self.task_id = task_id
        self.limit = limit


@dataclass
class RequestGenerationBudget:
    """The process counter observed by one proxy request."""

    budget: "GenerationBudget"
    consumed_before: int
    attempt_indices: list[int] = field(default_factory=list)
    task_id: Optional[str] = None
    task_consumed_before: Optional[int] = None

    @property
    def attempt_count(self) -> int:
        return len(self.attempt_indices)

    def metadata(self) -> Optional[dict]:
        if not self.budget.enabled:
            return None
        return {
            "limit": self.budget.limit,
            "consumed_before": self.consumed_before,
            "consumed_after": self.budget.consumed,
            "attempt_indices": list(self.attempt_indices),
            **({
                "per_task_limit": self.budget.per_task_limit,
                "task_id": self.task_id,
                "task_consumed_before": self.task_consumed_before,
                "task_consumed_after": self.budget.consumed_for_task(self.task_id),
            } if self.budget.per_task_limit is not None else {}),
        }


class GenerationBudget:
    """A process-wide counter whose reservations precede network calls."""

    def __init__(self, limit: Optional[int], per_task_limit: Optional[int] = None):
        for value, label in (
            (limit, "generation attempt limit"),
            (per_task_limit, "per-task generation attempt limit"),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{label} must be a positive integer")
        self.limit = limit
        self.per_task_limit = per_task_limit
        self._consumed = 0
        self._consumed_by_task: dict[str, int] = {}
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def enabled(self) -> bool:
        return self.limit is not None or self.per_task_limit is not None

    @property
    def consumed(self) -> int:
        with self._lock:
            return self._consumed

    def consumed_for_task(self, task_id: Optional[str]) -> Optional[int]:
        if self.per_task_limit is None:
            return None
        if not isinstance(task_id, str) or not task_id:
            return None
        with self._lock:
            return self._consumed_by_task.get(task_id, 0)

    def reserve(self, task_id: Optional[str] = None) -> int:
        """Reserve and return a 1-based attempt index atomically."""
        if self.per_task_limit is not None and (
            not isinstance(task_id, str) or not task_id
        ):
            raise ValueError(
                "task_id must be a non-empty string when a per-task generation "
                "attempt limit is enabled"
            )
        scope = self.current_request()
        if (self.per_task_limit is not None and scope is not None
                and scope.task_id is not None and scope.task_id != task_id):
            raise RuntimeError(
                "one proxy request cannot reserve generation attempts for "
                "multiple task IDs"
            )
        with self._lock:
            if self.limit is not None and self._consumed >= self.limit:
                raise GenerationBudgetExceeded(self.limit)
            task_consumed = (
                self._consumed_by_task.get(task_id, 0)
                if self.per_task_limit is not None else None
            )
            if (self.per_task_limit is not None
                    and task_consumed is not None
                    and task_consumed >= self.per_task_limit):
                raise TaskGenerationBudgetExceeded(task_id, self.per_task_limit)
            self._consumed += 1
            index = self._consumed
            if self.per_task_limit is not None:
                assert task_id is not None and task_consumed is not None
                self._consumed_by_task[task_id] = task_consumed + 1
        if scope is not None:
            if self.per_task_limit is not None:
                if scope.task_id is None:
                    scope.task_id = task_id
                    scope.task_consumed_before = task_consumed
            scope.attempt_indices.append(index)
        return index

    def current_request(self) -> Optional[RequestGenerationBudget]:
        stack = getattr(self._local, "request_stack", ())
        return stack[-1] if stack else None

    @contextmanager
    def request_scope(self) -> Iterator[RequestGenerationBudget]:
        record = RequestGenerationBudget(self, self.consumed)
        stack = getattr(self._local, "request_stack", None)
        if stack is None:
            stack = []
            self._local.request_stack = stack
        stack.append(record)
        try:
            yield record
        finally:
            popped = stack.pop()
            assert popped is record
