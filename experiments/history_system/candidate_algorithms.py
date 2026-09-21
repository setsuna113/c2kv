"""Build explicit ratio-8 candidate controllers from the frozen C1000 delivery."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable


VARIANTS = ("static_t02", "turn_c1", "goal_rescue", "dependency_first")
REPAIR_VARIANTS = ("request_contract", "argument_binding", "no_progress")
GOAL_VARIANTS = ("goal_pending", "goal_source", "goal_progress", "goal_joint")
GOAL_VERSION = "c2kv-goal-composition-v1"
VERIFIED_VARIANTS = ("goal_verified", "pending_verified")
VERIFIED_VERSION = "c2kv-verified-binding-v1"
PROOF_REGISTRY_VERSION = "verified-binding-rules-v1"
RELATIONAL_PROOF_REGISTRY_VERSION = "verified-binding-relations-v2"
INITIAL_VIEW_BACKBONES = {
    "goal_static": "goal_rescue", "pending_static": "goal_pending",
    "goal_verified_static": "goal_verified",
    "pending_verified_static": "pending_verified",
}
INITIAL_VIEW_VARIANTS = tuple(INITIAL_VIEW_BACKBONES)
INITIAL_VIEW_VERSION = "c2kv-initial-view-composition-v1"
INITIAL_VIEW_POLICY_VERSION = "c2kv-static-initial-view-v1"
STATIC_EXTENSION_VARIANTS = ("static_verified", "static_action_ledger", "static_verified_v2")
STATIC_EXTENSION_VERSION = "c2kv-static-extension-v1"
C1_V2_VARIANTS = ("c1_v2_verified",)
C1_V2_VERSION = "c2kv-c1-v2-verified-v1"
ALL_VARIANTS = (VARIANTS + REPAIR_VARIANTS + GOAL_VARIANTS + VERIFIED_VARIANTS
                + INITIAL_VIEW_VARIANTS + STATIC_EXTENSION_VARIANTS + C1_V2_VARIANTS)
RATIO = 8


def c1_v2_fields(variant: str) -> dict[str, Any]:
    if variant not in C1_V2_VARIANTS:
        raise ValueError("unknown C1 v2 candidate")
    return {
        "initial_view": {
            "policy": "s0_capacity_fallback",
            "version": "c2kv-s0-capacity-fallback-v1",
        },
        "recovery_backbone": "t02_complete_event",
        "completion_review": False,
        "proof_registry_version": PROOF_REGISTRY_VERSION,
    }


def initial_view_fields(variant: str) -> dict[str, Any]:
    """Describe only opt-in compositions; keep historical contracts byte-stable."""
    if variant in STATIC_EXTENSION_VARIANTS:
        if variant == "static_verified_v2":
            return {
                "recovery_backbone": "static_t02",
                "initial_view": {"policy": "static_gist", "version": INITIAL_VIEW_POLICY_VERSION},
                "commit_policy": "verified_binding_v2",
                "proof_registry_version": RELATIONAL_PROOF_REGISTRY_VERSION,
                "base_proof_registry_version": PROOF_REGISTRY_VERSION,
            }
        return {
            "recovery_backbone": "static_t02",
            "initial_view": {"policy": "static_gist", "version": INITIAL_VIEW_POLICY_VERSION},
            "commit_policy": "verified_binding" if variant == "static_verified" else "action_ledger",
            **({"proof_registry_version": PROOF_REGISTRY_VERSION}
               if variant == "static_verified" else {
                   "action_ledger_version": "static-action-ledger-v1",
                   "action_rules_version": "action-ledger-rules-v1"}),
        }
    if variant not in INITIAL_VIEW_BACKBONES:
        return {}
    return {
        "recovery_backbone": INITIAL_VIEW_BACKBONES[variant],
        "initial_view": {"policy": "static_gist", "version": INITIAL_VIEW_POLICY_VERSION},
        **({"proof_registry_version": PROOF_REGISTRY_VERSION}
           if INITIAL_VIEW_BACKBONES[variant] in VERIFIED_VARIANTS else {}),
    }


def build_profile(
    args: Any,
    *,
    base_controller: Mapping[str, Any],
    selected: Mapping[str, Any],
    risk_artifact_path: Path,
    risk_artifact_sha256: str,
    bind_artifact: Callable[[dict[str, Any], Path], tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep candidate experiments separate from C1 and D3 delivery contracts."""
    variant = args.candidate_algorithm
    if variant not in ALL_VARIANTS:
        raise ValueError(f"unknown candidate algorithm: {variant!r}")
    if args.method != "proposed":
        raise ValueError("--candidate-algorithm requires --method proposed")
    if getattr(args, "benchmark", "bfcl") not in {"bfcl", "acebench", "acon_appworld", "tau2", "toolsandbox"}:
        raise ValueError("candidate algorithms support BFCL, ACEBench Agent, AppWorld, tau2 and ToolSandbox only")
    if args.selector_artifact is not None:
        if variant not in REPAIR_VARIANTS:
            raise ValueError("candidate algorithms use the bundled T02 risk artifact")
        raise ValueError("repair candidates do not use a selector artifact")
    ratio = int(args.ratio) if args.ratio is not None else int(selected["ratio"])
    if ratio != RATIO:
        raise ValueError("candidate algorithms require ratio 8")

    actual_checkpoint = hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest()
    if actual_checkpoint != selected["checkpoint_selection"]["config_sha256"]:
        raise ValueError("candidate algorithms require the selected C1000 checkpoint config")
    controller = copy.deepcopy(dict(base_controller))
    for key in ("gp_experiments", "post_draft_recovery", "d3_hybrid_recovery"):
        controller.pop(key, None)
    if variant in REPAIR_VARIANTS:
        controller["candidate_algorithm"] = {"variant": variant}
        profile = {
            "schema": "c2kv-candidate-delivery-profile-v2",
            "method": "proposed",
            "detector": "candidate_algorithm",
            "candidate_algorithm": variant,
            "algorithm": f"C2KV candidate {variant}",
            "new_c1_training_claimed": False,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_config_sha256": actual_checkpoint,
            "selection_protocol": "c2kv-source-repair-v1",
            "history_variant": "H0",
            "ratio": ratio,
            "controller_sha256": hashlib.sha256(
                json.dumps(controller, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "automatic_reruns": 0,
        }
        return controller, profile

    if args.detector != "t02_risk":
        raise ValueError("candidate algorithms use the frozen T02 risk detector")
    if args.selector_threshold != 0.5:
        raise ValueError("candidate algorithms require the frozen risk threshold 0.5")
    artifact_file_sha256 = hashlib.sha256(risk_artifact_path.read_bytes()).hexdigest()
    if artifact_file_sha256 != risk_artifact_sha256:
        raise ValueError("bundled T02 risk artifact differs from the evaluated release")
    source_artifact = json.loads(risk_artifact_path.read_text(encoding="utf-8"))
    bound_artifact, binding = bind_artifact(source_artifact, args.checkpoint)
    controller["candidate_algorithm"] = {
        "variant": variant,
        "risk_artifact": bound_artifact,
        "risk_threshold": 0.5,
        **({"proof_registry_version": PROOF_REGISTRY_VERSION}
           if variant in VERIFIED_VARIANTS else {}),
        **initial_view_fields(variant),
        **(c1_v2_fields(variant) if variant in C1_V2_VARIANTS else {}),
    }
    profile = {
        "schema": "c2kv-candidate-delivery-profile-v1",
        "method": "proposed",
        "detector": "candidate_algorithm",
        "candidate_algorithm": variant,
        "algorithm": f"C2KV candidate {variant}",
        "new_c1_training_claimed": False,
        "selector_artifact": str(risk_artifact_path.resolve()),
        "selector_artifact_sha256": artifact_file_sha256,
        "selector_artifact_binding": binding,
        "selector_threshold": 0.5,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config_sha256": actual_checkpoint,
        "selection_protocol": "candidate_algorithm_v1",
        "history_variant": "H0",
        "ratio": ratio,
        "controller_sha256": hashlib.sha256(
            json.dumps(controller, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "automatic_reruns": 0,
    }
    if variant in GOAL_VARIANTS:
        profile["schema"] = "c2kv-candidate-delivery-profile-v3"
        profile["selection_protocol"] = GOAL_VERSION
    elif variant in VERIFIED_VARIANTS:
        profile["schema"] = "c2kv-candidate-delivery-profile-v4"
        profile["selection_protocol"] = VERIFIED_VERSION
        profile["proof_registry_version"] = PROOF_REGISTRY_VERSION
    elif variant in INITIAL_VIEW_VARIANTS:
        profile["schema"] = "c2kv-candidate-delivery-profile-v5"
        profile["selection_protocol"] = INITIAL_VIEW_VERSION
        profile.update(initial_view_fields(variant))
    elif variant in STATIC_EXTENSION_VARIANTS:
        profile["schema"] = "c2kv-candidate-delivery-profile-v6"
        profile["selection_protocol"] = STATIC_EXTENSION_VERSION
        profile.update(initial_view_fields(variant))
    elif variant in C1_V2_VARIANTS:
        profile["schema"] = "c2kv-candidate-delivery-profile-v7"
        profile["selection_protocol"] = C1_V2_VERSION
        profile.update(c1_v2_fields(variant))
    return controller, profile
