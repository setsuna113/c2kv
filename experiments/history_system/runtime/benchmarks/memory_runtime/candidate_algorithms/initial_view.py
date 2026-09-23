"""Compose an explicit first view with an unchanged recovery backbone."""

from __future__ import annotations

import copy
from typing import Any, Mapping

from . import (
    INITIAL_VIEW_BACKBONES,
    INITIAL_VIEW_POLICY_VERSION,
    INITIAL_VIEW_VERSION,
    VERIFIED_VARIANTS,
)
from .allocation import CandidateAllocator
from .controller import wrap_with_candidate_recovery


STATIC_INITIAL_VIEW = {
    "policy": "static_gist",
    "version": INITIAL_VIEW_POLICY_VERSION,
}


def validate_initial_view_config(candidate: Mapping[str, Any]) -> str:
    """Require the public variant, first view, and recovery policy to agree."""
    variant = candidate.get("variant")
    if variant not in INITIAL_VIEW_BACKBONES:
        raise ValueError(f"Unknown initial-view variant: {variant!r}")
    backbone = INITIAL_VIEW_BACKBONES[variant]
    if candidate.get("recovery_backbone") != backbone:
        raise ValueError(
            f"{variant} requires recovery_backbone={backbone!r}"
        )
    if candidate.get("initial_view") != STATIC_INITIAL_VIEW:
        raise ValueError(
            f"{variant} requires initial_view={STATIC_INITIAL_VIEW!r}"
        )
    if backbone in VERIFIED_VARIANTS:
        from .verified_binding import PROOF_REGISTRY_VERSION
        if candidate.get("proof_registry_version") != PROOF_REGISTRY_VERSION:
            raise ValueError(f"{variant} requires the frozen proof registry version")
    if "risk_artifact" not in candidate:
        raise ValueError(f"{variant} requires risk_artifact")
    threshold = candidate.get("risk_threshold")
    if isinstance(threshold, bool) or threshold != 0.5:
        raise ValueError(f"{variant} requires risk_threshold=0.5")
    return backbone


def build_initial_view_allocator(
    tokenizer: Any,
    *,
    initial_view: Mapping[str, str],
    packing: Mapping[str, Any],
    policy: Mapping[str, Any],
    model_context: int | None,
    s0_config: Mapping[str, Any],
    benchmark: str,
    initial_allocator_factory=None,
) -> CandidateAllocator:
    """Select the first-view policy independently of post-draft recovery."""
    if initial_view != STATIC_INITIAL_VIEW:
        raise ValueError("Unsupported initial_view policy")
    from ..initial_factory import instantiate_initial
    return instantiate_initial(CandidateAllocator,
        tokenizer,
        initial_allocator_factory=initial_allocator_factory,
        packing=packing,
        policy=policy,
        variant="static_t02",
        model_context=model_context,
        s0_config=s0_config,
        benchmark=benchmark,
    )


class InitialViewCompositionController:
    """Keep recovery logic in its existing controller and expose a new identity."""

    def __init__(self, recovery, *, variant: str, recovery_backbone: str):
        self.recovery = recovery
        self.variant = variant
        self.recovery_backbone = recovery_backbone

    def __getattr__(self, name: str) -> Any:
        return getattr(self.recovery, name)

    def _identify_metadata(self, metadata: dict[str, Any]) -> None:
        metadata["candidate_algorithm"].update(
            schema=INITIAL_VIEW_VERSION,
            variant=self.variant,
            recovery_backbone=self.recovery_backbone,
            initial_view=copy.deepcopy(STATIC_INITIAL_VIEW),
        )
        metadata["route"]["baseline_identity"] = (
            f"{INITIAL_VIEW_VERSION}:{self.variant}"
        )

    def prepare(self, payload, *, ratio, max_new_tokens):
        prepared = self.recovery.prepare(
            payload, ratio=ratio, max_new_tokens=max_new_tokens
        )
        self._identify_metadata(prepared.metadata)
        return prepared

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error=None):
        result = self.recovery.reconsider(
            prepared,
            draft_tool_calls,
            draft_text=draft_text,
            parse_error=parse_error,
        )
        result = {**result, "metadata": copy.deepcopy(result["metadata"]),
                  "decision": copy.deepcopy(result["decision"])}
        self._identify_metadata(result["metadata"])
        decision = result["decision"]
        decision["backbone_decision_version"] = decision["version"]
        decision.update(
            version=INITIAL_VIEW_VERSION,
            variant=self.variant,
            recovery_backbone=self.recovery_backbone,
            initial_view=copy.deepcopy(STATIC_INITIAL_VIEW),
        )
        result["metadata"]["exact_recovery"] = copy.deepcopy(decision)
        return result


def build_initial_view_composition(
    tokenizer: Any,
    *,
    candidate: Mapping[str, Any],
    packing: Mapping[str, Any],
    policy: Mapping[str, Any],
    model_context: int | None,
    s0_config: Mapping[str, Any],
    benchmark: str,
    initial_allocator_factory=None,
) -> InitialViewCompositionController:
    backbone = validate_initial_view_config(candidate)
    base = build_initial_view_allocator(
        tokenizer,
        initial_allocator_factory=initial_allocator_factory,
        initial_view=candidate["initial_view"],
        packing=packing,
        policy=policy,
        model_context=model_context,
        s0_config=s0_config,
        benchmark=benchmark,
    )
    recovery = wrap_with_candidate_recovery(
        base, {**candidate, "variant": backbone}
    )
    return InitialViewCompositionController(
        recovery, variant=candidate["variant"], recovery_backbone=backbone
    )
