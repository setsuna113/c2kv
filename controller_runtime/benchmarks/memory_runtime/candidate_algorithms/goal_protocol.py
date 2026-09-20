"""Policy seams for additive Goal recovery, independent of device and harness."""
from __future__ import annotations

from typing import Protocol

from .repair_protocol import GuardVerdict, RepairContext, RepairProposal


class AdditionalGoalPolicy(Protocol):
    def propose(self, context: RepairContext) -> RepairProposal | None: ...

    def validate(self, context: RepairContext, proposal: RepairProposal,
                 candidate_calls, *, draft_text: str,
                 parse_error: str | None = None) -> GuardVerdict: ...


def modules_for(variant):
    """Load only selected modules; historical profiles never use this registry."""
    from . import GOAL_VARIANTS
    if variant not in GOAL_VARIANTS:
        raise ValueError("Unknown Goal composition variant")
    policies = []
    if variant in {"goal_source", "goal_joint"}:
        from .goal_source import Policy
        policies.append(("source", Policy()))
    if variant in {"goal_progress", "goal_joint"}:
        from .goal_progress import Policy
        policies.append(("progress", Policy()))
    return tuple(policies)
