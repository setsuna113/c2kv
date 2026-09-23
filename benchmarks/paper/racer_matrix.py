"""Explicit paper-matrix overlay for modular RACER compositions."""

from __future__ import annotations

import copy
import re
from typing import Mapping

from .candidate_matrix import REPAIR_VARIANTS, SUPPORTED_BENCHMARKS, VARIANT_TO_ARM


RACER_BACKENDS = ("c2kv", "commitkv", "agentkv", "h2o", "snapkv", "pyramidkv", "streamingllm")
RACER_CANDIDATE_POLICIES = tuple(VARIANT_TO_ARM)
RACER_POLICIES = ("off", "t02", *RACER_CANDIDATE_POLICIES)

_BACKEND_CONFIG = {
    "c2kv": {"method": "c2kv"},
    "commitkv": {
        "method": "commitkv",
        "backend": "reference_attention",
        "persistent_session": True,
    },
    "agentkv": {
        "method": "agentkv",
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
    "pyramidkv": {
        "method": "pyramidkv",
        "backend": "reference_attention",
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

_UNIFIED_KV_ARMS = {
    "commitkv": "commitkv",
    "agentkv": "agentkv",
    "history_kv_h2o_r25_persistent": "h2o",
    "history_kv_snapkv_r25_persistent": "snapkv",
    "history_kv_pyramidkv_r25_persistent": "pyramidkv",
    "history_kv_streamingllm_r25_persistent": "streamingllm",
}

_UNIFIED_C2KV_ARMS = {"c2kv_native_r8": 8}


def unified_backend_for_arm(arm: str) -> str | None:
    return _UNIFIED_KV_ARMS.get(arm)

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


def _scope(method: Mapping) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (tuple(method.get("benchmarks") or ()),
            tuple(method.get("tool_contexts") or ("raw",)))


def resolve_unified_runtime_methods(config: dict) -> dict:
    """Route explicitly marked primary methods through the native runtime.

    Historical, unmarked methods keep their original arms and result identities.
    The C2KV recovery-off primary is the bare native arm, never the historical
    S0-allocated racer_c2kv_off arm.
    """
    marked = [row for row in config["methods"] if row.get("history_runtime") == "racer"]
    if not marked:
        return config
    budget = config.get("history_kv_budget_tokens")
    if type(budget) is not int or budget < 1:
        raise ValueError("Unified history runtime requires a positive history_kv_budget_tokens B")
    configured_benchmarks = {row["name"] for row in config["benchmarks"]}
    methods = []
    for source in config["methods"]:
        if source.get("history_runtime") != "racer":
            methods.append(source)
            continue
        arm = source["arm"]
        expected_backend = (_UNIFIED_KV_ARMS.get(arm) or
                            ("c2kv" if arm in _UNIFIED_C2KV_ARMS else None))
        if expected_backend is None:
            if is_racer_arm(arm):
                backend, policy, arm_budget = parse_racer_arm_name(arm)
                if (source.get("history_backend") != backend
                        or source.get("recovery_policy", policy) != policy):
                    raise ValueError(f"Unified RACER arm differs from its declared identity: {arm}")
                if (source.get("history_budget_tokens") != arm_budget
                        or source.get("racer_backend") != resolve_racer_backend(
                            backend, policy, arm_budget)):
                    raise ValueError(f"Unified RACER arm differs from its resolved backend: {arm}")
                if source.get("budget_variant"):
                    methods.append(source)
                    continue
                if arm_budget != budget and source.get("history_budget_source") != "shared":
                    raise ValueError(f"Unified RACER arm has a fixed budget different from B: {arm}")
                row = copy.deepcopy(source)
                row.update(arm=racer_arm_name(backend, policy, budget),
                           history_budget_tokens=budget,
                           racer_backend=resolve_racer_backend(backend, policy, budget))
                methods.append(row)
                continue
            raise ValueError(f"Unsupported unified history arm: {arm}")
        if source.get("history_backend") != expected_backend:
            raise ValueError(f"Unified history backend differs from source arm: {arm}")
        if source.get("recovery_policy", "off") != "off":
            raise ValueError("Primary unified history rows must have recovery_policy=off")
        source_budget = source.get("history_budget_tokens")
        shared = (source_budget == "shared"
                  or source.get("history_budget_source") == "shared")
        if (source_budget not in (None, "shared", budget) and not shared):
            raise ValueError(f"Unified history budget differs from global B: {arm}")
        benchmarks, _ = _scope(source)
        if (not benchmarks or len(benchmarks) != len(set(benchmarks))
                or not set(benchmarks) <= configured_benchmarks
                or not set(benchmarks) <= SUPPORTED_BENCHMARKS):
            raise ValueError(f"Unified history method needs an explicit portable benchmark scope: {arm}")
        row = copy.deepcopy(source)
        row.pop("retention", None)
        if shared:
            row["history_budget_source"] = "shared"
        else:
            row.pop("history_budget_source", None)
        row["history_budget_tokens"] = budget
        row["recovery_policy"] = "off"
        if expected_backend == "c2kv":
            ratio = _UNIFIED_C2KV_ARMS[arm]
            if row.get("compression_ratio", ratio) != ratio or row.get("ratio", ratio) != ratio:
                raise ValueError("C2KV bare primary must preserve its explicit compression ratio")
            row["compression_ratio"] = ratio
            row["ratio"] = ratio
            row["history_allocation"] = "c2kv_bare"
        else:
            row.pop("ratio", None)
            row["arm"] = racer_arm_name(expected_backend, "off", budget)
            row["group"] = "racer"
            row["racer_backend"] = resolve_racer_backend(expected_backend, "off", budget)
        methods.append(row)
    identities = [(row["arm"], row.get("history_budget_tokens"), _scope(row))
                  for row in methods]
    if len(identities) != len(set(identities)):
        raise ValueError("Unified history runtime produced duplicate cell identities")
    return dict(config, methods=methods)


def with_racer_methods(config: dict, backends: tuple[str, ...],
                       policies: tuple[str, ...], history_budget_tokens: int | None) -> dict:
    """Add exact off/on pairs, reusing marked native primaries when present."""
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
    existing = {}
    for row in resolved["methods"]:
        existing.setdefault(row["arm"], []).append(row)

    def unique(arm: str) -> dict | None:
        rows = existing.get(arm, ())
        if len(rows) > 1:
            raise ValueError(f"RACER arm appears more than once in methods: {arm}")
        return rows[0] if rows else None

    def validate_existing(row: dict, backend: str, policy: str,
                          expected_scope: tuple[tuple[str, ...], tuple[str, ...]]) -> None:
        arm = racer_arm_name(backend, policy, history_budget_tokens)
        if (row.get("arm") != arm or row.get("group") != "racer"
                or row.get("history_budget_tokens") != history_budget_tokens
                or row.get("racer_backend") != resolve_racer_backend(
                    backend, policy, history_budget_tokens)
                or _scope(row) != expected_scope
                or row.get("ratio") is not None or row.get("retention") is not None):
            raise ValueError(f"Existing RACER arm differs from its paired contract: {arm}")

    for backend in backends:
        off_arm = racer_arm_name(backend, "off", history_budget_tokens)
        off = unique(off_arm)
        bare = None
        if backend == "c2kv":
            bare_rows = [row for row in resolved["methods"]
                         if row.get("history_runtime") == "racer"
                         and row.get("history_backend") == "c2kv"
                         and row.get("recovery_policy") == "off"
                         and row.get("arm") in _UNIFIED_C2KV_ARMS]
            if len(bare_rows) > 1:
                raise ValueError("Unified C2KV has more than one bare primary")
            bare = bare_rows[0] if bare_rows else None
            if bare and off:
                raise ValueError("Historical S0-off RACER arm cannot replace bare C2KV primary")
            if bare and bare.get("history_budget_tokens") != history_budget_tokens:
                raise ValueError("Bare C2KV primary differs from RACER history budget")
        if off:
            validate_existing(off, backend, "off", _scope(off))
        if bare:
            off = bare
        if off is None:
            off = {
                "method": f"RACER {backend} off",
                "arm": off_arm,
                "group": "racer",
                "history_budget_tokens": history_budget_tokens,
                "racer_backend": resolve_racer_backend(
                    backend, "off", history_budget_tokens),
                "benchmarks": list(configured_benchmarks),
            }
            resolved["methods"].append(off)
            existing[off_arm] = [off]
        paired_scope = _scope(off)
        for policy in policies:
            if policy == "off":
                continue
            arm = racer_arm_name(backend, policy, history_budget_tokens)
            current = unique(arm)
            if current:
                validate_existing(current, backend, policy, paired_scope)
                continue
            method = copy.deepcopy(off)
            method.update(method=(f"{off['method']}+RACER {policy}"
                                  if off.get("history_runtime") == "racer" else
                                  f"RACER {backend} {policy}"),
                          arm=arm, group="racer", recovery_policy=policy,
                          racer_backend=resolve_racer_backend(
                              backend, policy, history_budget_tokens))
            method.pop("history_allocation", None)
            if backend == "c2kv":
                method["compression_ratio"] = _UNIFIED_C2KV_ARMS.get(off["arm"], 8)
                method.pop("ratio", None)
            resolved["methods"].append(method)
            existing[arm] = [method]
    return resolved
