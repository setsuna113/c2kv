"""Explicit paper-matrix overlay for the four ratio-8 BFCL candidate arms."""

from __future__ import annotations

import copy


VARIANT_TO_ARM = {
    "static_t02": "c2kv_static_t02_r8",
    "turn_c1": "c2kv_turn_c1_r8",
    "goal_rescue": "c2kv_goal_rescue_r8",
    "dependency_first": "c2kv_dependency_first_r8",
}
ARM_TO_VARIANT = {arm: variant for variant, arm in VARIANT_TO_ARM.items()}


def parse_candidate_arms(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    requested = tuple(VARIANT_TO_ARM) if value == "all" else tuple(value.split(","))
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("candidate arms must be unique")
    unknown = set(requested) - set(VARIANT_TO_ARM)
    if unknown:
        raise ValueError(f"unknown candidate arms: {sorted(unknown)}")
    return requested


def with_candidate_methods(config: dict, variants: tuple[str, ...]) -> dict:
    if not variants:
        return config
    if "bfcl_base" not in {row["name"] for row in config["benchmarks"]}:
        raise ValueError("candidate arms require the bfcl_base benchmark")
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
            "benchmarks": ["bfcl_base"],
        })
    return resolved
