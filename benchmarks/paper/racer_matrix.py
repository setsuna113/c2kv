"""Explicit paper-matrix overlay for modular RACER compositions."""

from __future__ import annotations

import copy
import re
from typing import Mapping

from .candidate_matrix import REPAIR_VARIANTS, SUPPORTED_BENCHMARKS, VARIANT_TO_ARM


RACER_BACKENDS = ("c2kv", "commitkv", "h2o", "snapkv", "streamingllm")
RACER_CANDIDATE_POLICIES = tuple(VARIANT_TO_ARM)
RACER_POLICIES = ("off", "t02", *RACER_CANDIDATE_POLICIES)

_BACKEND_CONFIG = {
    "c2kv": {"method": "c2kv"},
    "commitkv": {
        "method": "commitkv",
        "backend": "reference_attention",
        "persistent_session": True,
    },
    "h2o": {
        "method": "h2o",
        "backend": "physical_eviction",
        "recent_window": 64,
        "h2o_recent_fraction": 0.5,
        "persistent_session": True,
    },
    "snapkv": {
        "method": "snapkv_persistent",
        "backend": "physical_eviction",
        "recent_window": 64,
        "kernel_size": 5,
        "pooling": "avgpool",
        "persistent_session": True,
    },
    "streamingllm": {
        "method": "streamingllm",
        "backend": "physical_eviction",
        "recent_window": 64,
        "persistent_session": True,
    },
}

_ARM_PATTERN = re.compile(
    r"^racer_(?P<backend>" + "|".join(map(re.escape, RACER_BACKENDS)) + ")_"
    r"(?P<policy>" + "|".join(map(re.escape, RACER_POLICIES)) + ")_"
    r"b(?P<budget>[1-9][0-9]*)$"
)


def _unique_csv(value: str, *, label: str) -> tuple[str, ...]:
    values = tuple(item for item in value.split(",") if item)
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must be a nonempty unique comma-separated list")
    return values


def parse_racer_backends(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    if value == "all":
        return RACER_BACKENDS
    requested = _unique_csv(value, label="RACER backends")
    unknown = set(requested) - set(RACER_BACKENDS)
    if unknown:
        raise ValueError(f"unknown RACER backends: {sorted(unknown)}")
    return requested


def parse_racer_policies(value: str) -> tuple[str, ...]:
    """Resolve policy names and include the paired recovery-off control."""
    if not value:
        return ()
    if value == "all":
        return RACER_POLICIES
    requested = _unique_csv(value, label="RACER policies")
    unknown = set(requested) - set(RACER_POLICIES)
    if unknown:
        raise ValueError(f"unknown RACER policies: {sorted(unknown)}")
    if any(policy != "off" for policy in requested):
        requested = ("off", *(policy for policy in requested if policy != "off"))
    return requested


def racer_arm_name(backend: str, policy: str, history_budget_tokens: int) -> str:
    if backend not in RACER_BACKENDS or policy not in RACER_POLICIES:
        raise ValueError("unknown RACER backend or policy")
    if type(history_budget_tokens) is not int or history_budget_tokens < 1:
        raise ValueError("RACER history budget must be a positive integer")
    return f"racer_{backend}_{policy}_b{history_budget_tokens}"


def parse_racer_arm_name(name: str) -> tuple[str, str, int]:
    match = _ARM_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid RACER arm name: {name!r}")
    return match["backend"], match["policy"], int(match["budget"])


def is_racer_arm(name: str | None) -> bool:
    if not isinstance(name, str):
        return False
    return _ARM_PATTERN.fullmatch(name) is not None


def resolve_racer_backend(backend: str, policy: str,
                          history_budget_tokens: int) -> dict:
    racer_arm_name(backend, policy, history_budget_tokens)
    backend_config = copy.deepcopy(_BACKEND_CONFIG[backend])
    if backend != "c2kv":
        backend_config["target_tokens"] = history_budget_tokens
    return {
        "schema": "racer-backend-v1",
        "backend": backend,
        "policy": policy,
        "history_budget_tokens": history_budget_tokens,
        "backend_config": backend_config,
        "detector_calibration": (
            "not_used" if policy == "off" or policy in REPAIR_VARIANTS else
            "reference" if backend == "c2kv" else
            "frozen_c2kv_unvalidated_transfer"
        ),
        "allocation": "c2kv_s0" if backend == "c2kv" else "backend_native_persistent",
    }


def validate_racer_backend(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("RACER backend config must be an object")
    backend = value.get("backend")
    policy = value.get("policy")
    budget = value.get("history_budget_tokens")
    expected = resolve_racer_backend(backend, policy, budget)
    if dict(value) != expected:
        raise ValueError("RACER backend config differs from its resolved arm contract")
    return expected


def racer_config_for_arm(config: Mapping, arm_name: str) -> dict:
    matches = [method for method in config.get("methods", ()) if method.get("arm") == arm_name]
    if len(matches) != 1:
        raise ValueError(f"RACER arm requires one configured method: {arm_name}")
    method = matches[0]
    resolved = validate_racer_backend(method.get("racer_backend"))
    backend, policy, budget = parse_racer_arm_name(arm_name)
    if resolved != resolve_racer_backend(backend, policy, budget):
        raise ValueError("RACER arm identity differs from its resolved backend config")
    if method.get("history_budget_tokens") != budget:
        raise ValueError("RACER matrix history budget differs from its backend config")
    if method.get("ratio") is not None or method.get("retention") is not None:
        raise ValueError("RACER uses an explicit history token budget, not a nominal ratio")
    return resolved


def with_racer_methods(config: dict, backends: tuple[str, ...],
                       policies: tuple[str, ...], history_budget_tokens: int | None) -> dict:
    """Add an opt-in backend x policy matrix without changing legacy methods."""
    if not backends and not policies and history_budget_tokens is None:
        return config
    if not backends or not policies or history_budget_tokens is None:
        raise ValueError("RACER requires backends, policies, and an explicit history budget")
    if len(backends) != len(set(backends)) or not set(backends) <= set(RACER_BACKENDS):
        raise ValueError("RACER backends must be unique supported names")
    if len(policies) != len(set(policies)) or not set(policies) <= set(RACER_POLICIES):
        raise ValueError("RACER policies must be unique supported names")
    if any(policy != "off" for policy in policies):
        policies = ("off", *(policy for policy in policies if policy != "off"))
    if type(history_budget_tokens) is not int or history_budget_tokens < 1:
        raise ValueError("RACER history budget must be a positive integer")

    configured_benchmarks = tuple(row["name"] for row in config["benchmarks"])
    if not configured_benchmarks or not set(configured_benchmarks) <= SUPPORTED_BENCHMARKS:
        raise ValueError("RACER requires the configured portable paper benchmarks")
    resolved = copy.deepcopy(config)
    existing = {row["arm"] for row in resolved["methods"]}
    for backend in backends:
        for policy in policies:
            arm = racer_arm_name(backend, policy, history_budget_tokens)
            if arm in existing:
                raise ValueError(f"RACER arm already exists in methods: {arm}")
            resolved["methods"].append({
                "method": f"RACER {backend} {policy}",
                "arm": arm,
                "group": "racer",
                "history_budget_tokens": history_budget_tokens,
                "racer_backend": resolve_racer_backend(
                    backend, policy, history_budget_tokens),
                "benchmarks": list(configured_benchmarks),
            })
            existing.add(arm)
    return resolved
