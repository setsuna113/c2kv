"""Request-bound backend history constraints for capacity planning."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class BackendCapacityConstraints:
    """Mandatory history in one backend source-message frame.

    ``mandatory_source_indices`` and ``exact_sources`` use the same request's
    message indices. The caller is responsible for mapping canonical source
    indices into that frame before constructing or querying this value.
    """

    session_id: str
    decision_key: str
    stage: str
    history_budget_tokens: int
    mandatory_history_tokens: int
    mandatory_source_indices: tuple[int, ...]
    release: str
    provenance: str

    def __post_init__(self) -> None:
        _nonempty_id(self.session_id, "session_id")
        _nonempty_id(self.decision_key, "decision_key")
        _stage(self.stage)
        if type(self.history_budget_tokens) is not int or self.history_budget_tokens <= 0:
            raise ValueError("history_budget_tokens must be a positive integer")
        if type(self.mandatory_history_tokens) is not int or self.mandatory_history_tokens < 0:
            raise ValueError("mandatory_history_tokens must be a nonnegative integer")
        if not isinstance(self.mandatory_source_indices, tuple) or any(
            type(index) is not int or index < 0 for index in self.mandatory_source_indices
        ):
            raise ValueError("mandatory_source_indices must be nonnegative integer indices")
        if len(set(self.mandatory_source_indices)) != len(self.mandatory_source_indices):
            raise ValueError("mandatory_source_indices must be unique")
        if self.release not in {"replaced_source_message", "never"}:
            raise ValueError("unsupported release condition")
        _nonempty_id(self.provenance, "provenance")

    def minimum_history_tokens(self, exact_sources: Iterable[int]) -> int:
        """Release the whole mandatory window when any protected source is replaced."""
        sources = tuple(exact_sources)
        if any(type(index) is not int or index < 0 for index in sources):
            raise ValueError("exact_sources must contain nonnegative integer indices")
        if self.release == "replaced_source_message" and set(sources).intersection(
            self.mandatory_source_indices
        ):
            return 0
        return self.mandatory_history_tokens


def _nonempty_id(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _stage(value: str) -> None:
    if value not in {"draft", "regeneration"}:
        raise ValueError("stage must be draft or regeneration")


_CURRENT: ContextVar[BackendCapacityConstraints | None] = ContextVar(
    "backend_capacity_constraints", default=None
)


@contextmanager
def capacity_scope(
    constraints: BackendCapacityConstraints | None,
) -> Iterator[BackendCapacityConstraints | None]:
    """Install one request's constraints and restore the prior context on exit."""
    if constraints is not None and not isinstance(constraints, BackendCapacityConstraints):
        raise TypeError("capacity_scope expects BackendCapacityConstraints or None")
    token = _CURRENT.set(constraints)
    try:
        yield constraints
    finally:
        _CURRENT.reset(token)


def current_constraints(
    session_id: str, decision_key: str | None = None, stage: str | None = None,
) -> BackendCapacityConstraints | None:
    """Return only a constraint bound to this request, or None for a mismatch."""
    _nonempty_id(session_id, "session_id")
    if decision_key is not None:
        _nonempty_id(decision_key, "decision_key")
    if stage is not None:
        _stage(stage)
    constraints = _CURRENT.get()
    if constraints is None or constraints.session_id != session_id:
        return None
    if decision_key is not None and constraints.decision_key != decision_key:
        return None
    if stage is not None and constraints.stage != stage:
        return None
    return constraints
