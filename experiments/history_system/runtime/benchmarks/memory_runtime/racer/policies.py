"""Compose existing recovery/commit policies with a history adapter."""
from __future__ import annotations

import copy

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
    if backend.allocation == "racer_s0":
        return _build_shared_initial_controller(tokenizer, config=config, backend=backend,
            packing=packing, policy=policy, model_context=model_context, benchmark=benchmark)
    allocator_type = PersistentHistoryAllocator
    if backend.extra_protection == "on":
        from .native_protection import NativeProtectionAllocator
        allocator_type = NativeProtectionAllocator
    base = allocator_type(tokenizer, backend_config=backend,
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


def _build_shared_initial_controller(tokenizer, *, config, backend, packing,
                                     policy, model_context, benchmark):
    from ..always_compress import ALWAYS_COMPRESSION_POLICY
    from ..event_native_always import NATIVE_S0_MODE
    from ..event_native_controls import build_event_native_controller
    from ..event_native_s0_policy import S0_CONFIG_DEFAULTS
    from .native_initial import NativeInitialFactory

    initial_config = {**S0_CONFIG_DEFAULTS,
                      **{key: value for key, value in config.items() if key != "racer_backend"}}
    candidate = initial_config.get("candidate_algorithm")
    if candidate is not None and candidate.get("variant") != backend.policy:
        raise ValueError("RACER backend policy differs from the configured policy identity")
    if backend.policy not in ("off", "t02") and candidate is None:
        raise ValueError("RACER candidate policy configuration is missing")
    if backend.policy == "t02" and "post_draft_recovery" not in initial_config:
        raise ValueError("RACER t02 detector configuration is missing")
    native_factory = NativeInitialFactory(backend)
    controller = build_event_native_controller(tokenizer, packing=packing, policy=policy,
        view_mode=NATIVE_S0_MODE, model_context=model_context,
        compression_policy=ALWAYS_COMPRESSION_POLICY, s0_config=initial_config,
        benchmark=benchmark, initial_allocator_factory=native_factory)
    initial = native_factory.initial
    if initial is None:
        raise ValueError("RACER requires a configured initial allocation controller")
    if getattr(backend, "mode", None) == "protected_off":
        controller = InitialOnlyPolicy(controller, candidate)
    return BackendPolicy(controller, backend, native_allocator=initial)


class InitialOnlyPolicy:
    """Keep the configured initial allocator and omit post-draft/commit recovery."""

    _disabled_hooks = frozenset({"observe_selection_draft", "observe_draft_features",
        "validate_commit", "finalize_commit", "calibration_risk", "advance_recovery"})

    def __init__(self, inner, candidate=None):
        self.inner = inner
        self.candidate = candidate
        self._prepared = {}
        self._checked = {}

    def __getattr__(self, name):
        if name in self._disabled_hooks:
            raise AttributeError(name)
        return getattr(self.inner, name)

    def prepare(self, *args, **kwargs):
        prepared = self.inner.prepare(*args, **kwargs)
        prepared.metadata["recovery_disabled_ablation"] = {
            "initial_policy_preserved": True, "post_draft_recovery_enabled": False,
            "commit_recovery_enabled": False,
            "candidate_variant": self.candidate.get("variant") if self.candidate else None}
        prepared.metadata["route"].update(recovery_enabled=False, max_generations_per_decision=1)
        key = (prepared.metadata["session_id"], prepared.metadata["decision_key"])
        self._prepared[key] = prepared
        return prepared

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error=None):
        from ..policy import PolicyInputError
        from ..recovery.gate import canonical
        key = (prepared.metadata["session_id"], prepared.metadata["decision_key"])
        if self._prepared.get(key) is not prepared:
            raise PolicyInputError("Recovery-off decision belongs to another controller")
        signature = canonical([draft_tool_calls, draft_text, parse_error])
        previous = self._checked.get(key)
        if previous is not None:
            if previous[0] != signature:
                raise PolicyInputError("Recovery-off decision cannot inspect another held draft")
            return copy.deepcopy(previous[1])
        decision = {"version": "racer-recovery-off-v2", "status": "no_op",
                    "reason": "post_draft_and_commit_recovery_disabled",
                    "regeneration_allowed": False, "upgrade_count": 0,
                    "candidate_event_id": None, "judges_action_correctness": False}
        metadata = copy.deepcopy(prepared.metadata)
        metadata["exact_recovery"] = copy.deepcopy(decision)
        metadata["post_draft_exact_recovery_applied"] = False
        result = {"regenerate": False, "memory": prepared.memory,
                  "metadata": metadata, "decision": decision}
        self._checked[key] = (signature, copy.deepcopy(result))
        return result


class BackendPolicy:
    """Attach backend identity without changing policy state or commit decisions."""

    def __init__(self, inner, backend, *, native_allocator=None):
        self.inner, self.backend = inner, backend
        self.base = inner
        self.native_allocator = native_allocator

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
            if self.backend.allocation == "racer_s0":
                candidate["shared_initial_allocation_applied"] = True
            elif "original_static_initial_view_preserved" in candidate:
                candidate["original_static_initial_view_preserved"] = False
            if "initial_view" in candidate and self.backend.allocation != "racer_s0":
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
        native_metadata = getattr(self.native_allocator, "_native_metadata", None)
        if callable(native_metadata):
            native_metadata(result["metadata"], result["memory"],
                            stage="recovery" if result["regenerate"] else "initial_s0")
        self._identity(result["metadata"])
        result["decision"]["racer_backend"] = self.backend.receipt()
        return result
