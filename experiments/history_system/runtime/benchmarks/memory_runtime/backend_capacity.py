"""Request-bound backend history constraints for capacity planning."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True)
class BackendCapacityConstraints:
    """Mandatory history in one backend source-message frame.

    ``mandatory_source_indices`` and ``exact_sources`` use canonical source
    message indices. The transport maps its engine receipt into this frame
    before constructing this value; planners never infer indices from tokens.
    """

    session_id: str
    decision_key: str
    stage: str
    history_budget_tokens: int
    mandatory_history_tokens: int
    mandatory_source_indices: tuple[int, ...]
    release: str
    provenance: str
    # A draft whose backend reconfigures its lifecycle from new source
    # messages (CommitKV): any new message closes the resumed window and a new
    # tool message opens a window of ``tool_event_mandatory_history_tokens``.
    # ``None`` keeps the resumed window for every draft.
    new_source_start: int | None = None
    tool_event_mandatory_history_tokens: int = 0
    tool_event_mandatory_source_indices: tuple[int, ...] = ()

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
        if not isinstance(self.release, str) or self.release not in {
            "replaced_source_message", "never"
        }:
            raise ValueError("unsupported release condition")
        _nonempty_id(self.provenance, "provenance")
        if self.new_source_start is not None and (
            type(self.new_source_start) is not int or self.new_source_start < 0
        ):
            raise ValueError("new_source_start must be a nonnegative integer")
        if type(self.tool_event_mandatory_history_tokens) is not int or self.tool_event_mandatory_history_tokens < 0:
            raise ValueError("tool_event_mandatory_history_tokens must be a nonnegative integer")
        if not isinstance(self.tool_event_mandatory_source_indices, tuple) or any(
            type(index) is not int or index < 0 for index in self.tool_event_mandatory_source_indices
        ) or len(set(self.tool_event_mandatory_source_indices)) != len(self.tool_event_mandatory_source_indices):
            raise ValueError("tool_event_mandatory_source_indices must be unique nonnegative indices")

    def minimum_history_tokens(self, exact_sources: Iterable[int], *, source_phases=None) -> int:
        """Only regeneration may release a window by replacing a protected source.

        ``source_phases`` gives the planned draft's per-source event phase
        ("tool" for a tool message).  With it, a lifecycle draft keeps what its
        new source messages leave mandatory rather than the resumed window.
        """
        sources = tuple(exact_sources)
        if any(type(index) is not int or index < 0 for index in sources):
            raise ValueError("exact_sources must contain nonnegative integer indices")
        if self.stage == "regeneration" and self.release == "replaced_source_message" and set(sources).intersection(
            self.mandatory_source_indices
        ):
            return 0
        if self.stage == "draft" and self.new_source_start is not None and source_phases is not None:
            new = tuple(source_phases)[self.new_source_start:]
            if "tool" in new:
                return self.tool_event_mandatory_history_tokens
            if new:
                return 0
        return self.mandatory_history_tokens


def _nonempty_id(value: object, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")


def _stage(value: str) -> None:
    if not isinstance(value, str) or value not in {"draft", "regeneration"}:
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
    """Return the bound constraint; reject stale bindings and absent context separately."""
    _nonempty_id(session_id, "session_id")
    if decision_key is not None:
        _nonempty_id(decision_key, "decision_key")
    if stage is not None:
        _stage(stage)
    constraints = _CURRENT.get()
    if constraints is None:
        return None
    if constraints.session_id != session_id:
        raise ValueError("backend capacity constraint belongs to another session")
    if decision_key is not None and constraints.decision_key != decision_key:
        raise ValueError("backend capacity constraint belongs to another decision")
    if stage is not None and constraints.stage != stage:
        raise ValueError("backend capacity constraint belongs to another stage")
    return constraints
