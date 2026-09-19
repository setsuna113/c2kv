"""Device-independent proposal and commit contracts for opt-in repair policies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


REPAIR_VERSION = "c2kv-source-repair-v1"


@dataclass(frozen=True)
class RepairContext:
    prepared: Any
    draft_tool_calls: tuple[dict[str, Any], ...]
    draft_text: str
    parse_error: str | None
    token_counter: Callable
    token_budget: int


@dataclass(frozen=True)
class RepairProposal:
    reason: str
    messages: tuple[dict[str, Any], ...]
    receipt: dict[str, Any]
    guard: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GuardVerdict:
    accepted: bool
    reason: str
    fallback: str = "original"


class RepairPolicy(Protocol):
    def propose(self, context: RepairContext) -> RepairProposal | None: ...

    def validate(self, context: RepairContext, proposal: RepairProposal,
                 candidate_calls, *, draft_text: str,
                 parse_error: str | None = None) -> GuardVerdict: ...


def policy_for(variant: str) -> RepairPolicy:
    # Imports stay local: legacy routes never import or initialize new policies.
    if variant == "request_contract":
        from .request_contract import Policy
    elif variant == "argument_binding":
        from .argument_binding import Policy
    elif variant == "no_progress":
        from .no_progress import Policy
    else:
        raise ValueError(f"Unknown repair policy: {variant!r}")
    return Policy()
