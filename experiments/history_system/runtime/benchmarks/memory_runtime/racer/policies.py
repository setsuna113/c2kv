"""Compose existing recovery/commit policies with a history adapter."""
from __future__ import annotations

from .allocator import PersistentHistoryAllocator
from .config import BackendConfig


def build_controller(tokenizer, *, config, packing, policy, model_context, benchmark):
    """Keep detector artifacts and policy implementations; substitute only KV admission."""
    from ..candidate_algorithms import INITIAL_VIEW_VARIANTS, STATIC_EXTENSION_VARIANTS, C1_V2_VARIANTS
    from ..candidate_algorithms.controller import wrap_with_candidate_recovery
    from ..event_native_s0_policy import S0_CONFIG_DEFAULTS

    backend = BackendConfig.parse(config["racer_backend"])
    if backend.backend == "c2kv":
        raise ValueError("C2KV uses its unchanged controller factory")
    base = PersistentHistoryAllocator(tokenizer, backend_config=backend,
        packing=packing, policy=policy, model_context=model_context, benchmark=benchmark,
        s0_config={key: value for key, value in config.items() if key in S0_CONFIG_DEFAULTS} or None)
    if backend.policy == "off":
        if any(key in config for key in ("candidate_algorithm", "post_draft_recovery",
                                         "gp_experiments", "d3_hybrid_recovery")):
            raise ValueError("RACER off cannot silently ignore a recovery policy")
        controller = base
    elif "candidate_algorithm" in config:
        candidate = config["candidate_algorithm"]
        if candidate.get("variant") != backend.policy:
            raise ValueError("RACER backend policy differs from the configured policy identity")
        if backend.policy in C1_V2_VARIANTS:
            from ..candidate_algorithms.c1_v2 import C1V2VerifiedController
            controller = C1V2VerifiedController(base, candidate)
        elif backend.policy in STATIC_EXTENSION_VARIANTS:
            from ..candidate_algorithms.static_extensions import (
                StaticVerifiedController, StaticVerifiedV2Controller, StaticActionLedgerController)
            controller = {"static_verified": StaticVerifiedController,
                          "static_verified_v2": StaticVerifiedV2Controller,
                          "static_action_ledger": StaticActionLedgerController}[backend.policy](base, candidate)
        elif backend.policy in INITIAL_VIEW_VARIANTS:
            from ..candidate_algorithms.initial_view import (
                validate_initial_view_config, InitialViewCompositionController)
            backbone = validate_initial_view_config(candidate)
            controller = InitialViewCompositionController(
                wrap_with_candidate_recovery(base, {**candidate, "variant": backbone}),
                variant=backend.policy, recovery_backbone=backbone)
        else:
            controller = wrap_with_candidate_recovery(base, candidate)
    elif backend.policy == "t02":
        if "gp_experiments" in config:
            from ..recovery.experiment import GPRecoveryController
            controller = GPRecoveryController(base, config["post_draft_recovery"],
                                              config["gp_experiments"], benchmark=benchmark)
        elif config.get("d3_hybrid_recovery"):
            from ..recovery.hybrid import wrap_with_d3_hybrid_recovery
            controller = wrap_with_d3_hybrid_recovery(base, config["post_draft_recovery"], benchmark=benchmark)
        else:
            from ..recovery.orchestrator import EventNativeRecoveryController
            controller = EventNativeRecoveryController(base, config["post_draft_recovery"], benchmark=benchmark)
    else:
        raise ValueError("RACER policy configuration is missing")
    return BackendPolicy(controller, backend)


class BackendPolicy:
    """Attach backend identity without changing policy state or commit decisions."""

    def __init__(self, inner, backend):
        self.inner, self.backend = inner, backend
        self.base = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _identity(self, metadata):
        metadata["racer_backend"] = self.backend.receipt()
        metadata["allocation"] = self.backend.allocation
        metadata["accounting_status"] = "admission_upper_bound_until_engine_receipt"
        candidate = metadata.get("candidate_algorithm")
        if isinstance(candidate, dict):
            candidate["history_allocation"] = self.backend.allocation
            candidate["c2kv_initial_allocation_applied"] = False
            if "original_static_initial_view_preserved" in candidate:
                candidate["original_static_initial_view_preserved"] = False
            if "initial_view" in candidate:
                candidate.setdefault("policy_source_initial_view", candidate["initial_view"])
                candidate["initial_view"] = {"policy": self.backend.allocation,
                                              "backend": self.backend.backend}
        metadata["route"]["baseline_identity"] = self.backend.receipt()["identity"]

    def prepare(self, *args, **kwargs):
        result = self.inner.prepare(*args, **kwargs)
        self._identity(result.metadata)
        return result

    def reconsider(self, *args, **kwargs):
        result = self.inner.reconsider(*args, **kwargs)
        self._identity(result["metadata"])
        result["decision"]["racer_backend"] = self.backend.receipt()
        return result
