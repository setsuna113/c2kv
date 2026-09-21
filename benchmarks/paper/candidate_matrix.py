"""Explicit paper-matrix overlay for ratio-8 candidate arms."""

from __future__ import annotations

import copy

from experiments.history_system.candidate_algorithms import (
    C1_V2_VARIANTS, INITIAL_VIEW_VARIANTS, STATIC_EXTENSION_VARIANTS,
)

VARIANT_TO_ARM = {
    "static_t02": "c2kv_static_t02_r8",
    "turn_c1": "c2kv_turn_c1_r8",
    "goal_rescue": "c2kv_goal_rescue_r8",
    "dependency_first": "c2kv_dependency_first_r8",
    "request_contract": "c2kv_request_contract_r8",
    "argument_binding": "c2kv_argument_binding_r8",
    "no_progress": "c2kv_no_progress_r8",
    "goal_pending": "c2kv_goal_pending_r8",
    "goal_source": "c2kv_goal_source_r8",
    "goal_progress": "c2kv_goal_progress_r8",
    "goal_joint": "c2kv_goal_joint_r8",
    "goal_verified": "c2kv_goal_verified_r8",
    "pending_verified": "c2kv_pending_verified_r8",
    "goal_static": "c2kv_goal_static_r8",
    "pending_static": "c2kv_pending_static_r8",
    "goal_verified_static": "c2kv_goal_verified_static_r8",
    "pending_verified_static": "c2kv_pending_verified_static_r8",
    "static_verified": "c2kv_static_verified_r8",
    "static_action_ledger": "c2kv_static_action_ledger_r8",
    "static_verified_v2": "c2kv_static_verified_v2_r8",
    "c1_v2_verified": "c2kv_c1_v2_verified_r8",
}
REPAIR_VARIANTS = frozenset({"request_contract", "argument_binding", "no_progress"})
GOAL_VARIANTS = ("goal_pending", "goal_source", "goal_progress", "goal_joint")
VERIFIED_VARIANTS = ("goal_verified", "pending_verified")
LEGACY_VARIANTS = tuple(variant for variant in VARIANT_TO_ARM
                        if variant not in (VERIFIED_VARIANTS + INITIAL_VIEW_VARIANTS
                                           + STATIC_EXTENSION_VARIANTS + C1_V2_VARIANTS))
ARM_TO_VARIANT = {arm: variant for variant, arm in VARIANT_TO_ARM.items()}
SUPPORTED_BENCHMARKS = frozenset({
    "bfcl_base", "bfcl_long_context", "appworld", "acebench_agent", "tau2", "toolsandbox",
})


def parse_candidate_arms(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    requested = LEGACY_VARIANTS if value == "all" else tuple(value.split(","))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("candidate arms must be unique")
    unknown = set(requested) - set(VARIANT_TO_ARM)
    if unknown:
        raise ValueError(f"unknown candidate arms: {sorted(unknown)}")
    return requested


def with_candidate_methods(config: dict, variants: tuple[str, ...],
                           benchmarks: tuple[str, ...] = ("bfcl_base",)) -> dict:
    if not variants:
        return config
    if (not benchmarks or len(benchmarks) != len(set(benchmarks))
            or not set(benchmarks) <= SUPPORTED_BENCHMARKS):
        raise ValueError("candidate benchmarks must be a nonempty subset of "
                         + ",".join(sorted(SUPPORTED_BENCHMARKS)))
    configured = {row["name"] for row in config["benchmarks"]}
    if not set(benchmarks) <= configured:
        raise ValueError("candidate benchmarks must exist in the configured paper matrix")
    resolved = copy.deepcopy(config)
    existing = {row["arm"] for row in resolved["methods"]}
    for variant in variants:
        arm = VARIANT_TO_ARM[variant]
        if arm in existing:
            raise ValueError(f"candidate arm already exists in methods: {arm}")
        resolved["methods"].append({
            "method": f"C2KV {variant}",
            "arm": arm,
            "group": "candidate",
            "ratio": 8,
            "benchmarks": list(benchmarks),
        })
    return resolved
