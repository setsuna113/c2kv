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
_V2_ARM_PATTERN = re.compile(
    r"^racer_v2_(?P<backend>" + "|".join(map(re.escape, RACER_BACKENDS)) + ")_"
    r"(?P<mode>bare|protected_off|(?:(?:" + "|".join(map(re.escape, RACER_POLICIES[1:]))
    + r")_protected_off)|(?:" + "|".join(map(re.escape, RACER_POLICIES[1:]))
    + r"))_b(?P<budget>[1-9][0-9]*)$"
)
_V3_ARM_PATTERN = re.compile(
    r"^racer_v3_(?P<backend>" + "|".join(map(re.escape, RACER_BACKENDS)) + ")_"
    r"(?P<policy>" + "|".join(map(re.escape, RACER_POLICIES)) + ")_"
    r"protection_(?P<protection>off|on)_b(?P<budget>[1-9][0-9]*)$"
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


def parse_racer_policies(value: str, *, paired_off: bool = True) -> tuple[str, ...]:
    """Resolve exact policies; legacy v2 callers retain their recovery-off pair."""
    if not value:
        return ()
    if value == "all":
        return RACER_POLICIES
    requested = _unique_csv(value, label="RACER policies")
    unknown = set(requested) - set(RACER_POLICIES)
    if unknown:
        raise ValueError(f"unknown RACER policies: {sorted(unknown)}")
    if paired_off and any(policy != "off" for policy in requested):
        requested = ("off", *(policy for policy in requested if policy != "off"))
    return requested


def parse_racer_protections(value: str) -> tuple[str, ...]:
    protections = _unique_csv(value, label="RACER protections")
    if not set(protections) <= {"off", "on"}:
        raise ValueError("RACER protections must be off,on or a subset")
    return protections


def racer_arm_name(backend: str, policy: str, history_budget_tokens: int) -> str:
    """Frozen v1 arm identity; v2/v3 cells have separate names."""
    if backend not in RACER_BACKENDS or policy not in RACER_POLICIES:
        raise ValueError("unknown RACER backend or policy")
    if type(history_budget_tokens) is not int or history_budget_tokens < 1:
        raise ValueError("RACER history budget must be a positive integer")
    return f"racer_{backend}_{policy}_b{history_budget_tokens}"


def racer_v2_arm_name(backend: str, policy: str, history_budget_tokens: int,
                      mode: str) -> str:
    racer_arm_name(backend, policy, history_budget_tokens)
    if mode not in {"bare", "protected_off", "on"}:
        raise ValueError("unknown RACER v2 mode")
    if mode == "bare" and policy != "off":
        raise ValueError("RACER bare mode requires policy=off")
    if backend == "c2kv" and mode == "bare":
        raise ValueError("C2KV bare mode uses c2kv_native_r8")
    if mode == "on" and policy == "off":
        raise ValueError("RACER on mode requires a recovery policy")
    suffix = ("bare" if mode == "bare" else
              "protected_off" if policy == "off" else
              f"{policy}_protected_off" if mode == "protected_off" else policy)
    return f"racer_v2_{backend}_{suffix}_b{history_budget_tokens}"


def racer_v3_arm_name(backend: str, policy: str, history_budget_tokens: int,
                      extra_protection: str) -> str:
    racer_arm_name(backend, policy, history_budget_tokens)
    if extra_protection not in {"off", "on"}:
        raise ValueError("unknown RACER extra protection")
    return (f"racer_v3_{backend}_{policy}_protection_{extra_protection}_"
            f"b{history_budget_tokens}")


def parse_racer_arm_identity(name: str) -> tuple[str, str, int, str | None]:
    match = _V3_ARM_PATTERN.fullmatch(name)
    if match is not None:
        return (match["backend"], match["policy"], int(match["budget"]),
                f"protection_{match['protection']}")
    match = _ARM_PATTERN.fullmatch(name)
    if match is not None:
        return match["backend"], match["policy"], int(match["budget"]), None
    match = _V2_ARM_PATTERN.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid RACER arm name: {name!r}")
    token = match["mode"]
    mode = "bare" if token == "bare" else "protected_off" if token.endswith(
        "protected_off") else "on"
    policy = ("off" if token in {"bare", "protected_off"} else
              token.removesuffix("_protected_off") if mode == "protected_off" else token)
    return match["backend"], policy, int(match["budget"]), mode


def parse_racer_arm_name(name: str) -> tuple[str, str, int]:
    backend, policy, budget, _ = parse_racer_arm_identity(name)
    return backend, policy, budget


def is_racer_arm(name: str | None) -> bool:
    if not isinstance(name, str):
        return False
    return any(pattern.fullmatch(name) is not None for pattern in
               (_ARM_PATTERN, _V2_ARM_PATTERN, _V3_ARM_PATTERN))


def resolve_racer_backend(backend: str, policy: str,
                          history_budget_tokens: int, *, mode: str | None = None,
                          extra_protection: str | None = None) -> dict:
    racer_arm_name(backend, policy, history_budget_tokens)
    if mode is not None and extra_protection is not None:
        raise ValueError("RACER v2 mode and v3 extra protection cannot be combined")
    if mode is not None:
        racer_v2_arm_name(backend, policy, history_budget_tokens, mode)
    if extra_protection is not None:
        racer_v3_arm_name(backend, policy, history_budget_tokens, extra_protection)
    backend_config = copy.deepcopy(_BACKEND_CONFIG[backend])
    if backend != "c2kv":
        backend_config["target_tokens"] = history_budget_tokens
    return {
        "schema": ("racer-backend-v3" if extra_protection is not None else
                   "racer-backend-v2" if mode is not None else "racer-backend-v1"),
        "backend": backend,
        "policy": policy,
        **({"mode": mode} if mode is not None else {}),
        **({"extra_protection": extra_protection} if extra_protection is not None else {}),
        "history_budget_tokens": history_budget_tokens,
        "backend_config": backend_config,
        "detector_calibration": (
            "not_used" if mode in {"bare", "protected_off"} or policy == "off"
            or policy in REPAIR_VARIANTS else
            "reference" if backend == "c2kv" else
            "frozen_c2kv_unvalidated_transfer"
        ),
        "allocation": ("c2kv_s0" if backend == "c2kv" else "backend_native_persistent")
        if mode is None else ("c2kv_bare" if backend == "c2kv" else
                              "backend_native_persistent") if mode == "bare" else "racer_s0",
    }


def validate_racer_backend(value: Mapping) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError("RACER backend config must be an object")
    backend = value.get("backend")
    policy = value.get("policy")
    budget = value.get("history_budget_tokens")
    schema = value.get("schema")
    if schema not in {"racer-backend-v1", "racer-backend-v2", "racer-backend-v3"}:
        raise ValueError("unknown RACER backend schema")
    expected = resolve_racer_backend(
        backend, policy, budget,
        mode=value.get("mode") if schema == "racer-backend-v2" else None,
        extra_protection=value.get("extra_protection") if schema == "racer-backend-v3" else None)
    if dict(value) != expected:
        raise ValueError("RACER backend config differs from its resolved arm contract")
    return expected


def racer_config_for_arm(config: Mapping, arm_name: str) -> dict:
    matches = [method for method in config.get("methods", ()) if method.get("arm") == arm_name]
    if len(matches) != 1:
        raise ValueError(f"RACER arm requires one configured method: {arm_name}")
    method = matches[0]
    resolved = validate_racer_backend(method.get("racer_backend"))
    backend, policy, budget, mode = parse_racer_arm_identity(arm_name)
    expected = (resolve_racer_backend(backend, policy, budget,
                                      extra_protection=mode.removeprefix("protection_"))
                if mode is not None and mode.startswith("protection_") else
                resolve_racer_backend(backend, policy, budget, mode=mode))
    if resolved != expected:
        raise ValueError("RACER arm identity differs from its resolved backend config")
    if method.get("history_budget_tokens") != budget:
        raise ValueError("RACER matrix history budget differs from its backend config")
    if (resolved["schema"] == "racer-backend-v3"
            and method.get("recovery_policy") != policy):
        raise ValueError("RACER v3 recovery policy differs from its arm identity")
    if method.get("ratio") is not None or method.get("retention") is not None:
        raise ValueError("RACER uses an explicit history token budget, not a nominal ratio")
    return resolved


def _scope(method: Mapping) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (tuple(method.get("benchmarks") or ()),
            tuple(method.get("tool_contexts") or ("raw",)))


def resolve_unified_runtime_methods(config: dict) -> dict:
    """Route explicitly marked primary methods through the native runtime.

    Historical, unmarked methods keep their original arms and result identities.
    The C2KV primary stays the native bare arm. Other marked primaries get an
    explicit v2 bare identity, distinct from historical v1 recovery-off cells.
    """
    marked = [row for row in config["methods"] if row.get("history_runtime") == "racer"]
    if not marked:
        return config
    budget = config.get("history_kv_budget_tokens")
    shared_marked = any(row.get("history_runtime") == "racer" and (
        row.get("history_budget_tokens") == "shared" or
        row.get("history_budget_source") == "shared") for row in marked)
    if shared_marked and (type(budget) is not int or budget < 1):
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
                backend, policy, arm_budget, mode = parse_racer_arm_identity(arm)
                expected_recovery = "off" if mode in {"bare", "protected_off"} else policy
                if (source.get("history_backend") != backend
                        or source.get("recovery_policy", expected_recovery) != expected_recovery):
                    raise ValueError(f"Unified RACER arm differs from its declared identity: {arm}")
                if (source.get("history_budget_tokens") != arm_budget
                        or source.get("racer_backend") != (
                            resolve_racer_backend(backend, policy, arm_budget,
                                                  extra_protection=mode.removeprefix("protection_"))
                            if mode is not None and mode.startswith("protection_") else
                            resolve_racer_backend(backend, policy, arm_budget, mode=mode))):
                    raise ValueError(f"Unified RACER arm differs from its resolved backend: {arm}")
                if source.get("budget_variant") or source.get("history_budget_source") != "shared":
                    methods.append(source)
                    continue
                row = copy.deepcopy(source)
                row.update(arm=(racer_arm_name(backend, policy, budget) if mode is None else
                                racer_v3_arm_name(backend, policy, budget,
                                                  mode.removeprefix("protection_"))
                                if mode.startswith("protection_") else
                                racer_v2_arm_name(backend, policy, budget, mode)),
                           history_budget_tokens=budget,
                           racer_backend=(resolve_racer_backend(
                               backend, policy, budget,
                               extra_protection=mode.removeprefix("protection_"))
                               if mode is not None and mode.startswith("protection_") else
                               resolve_racer_backend(backend, policy, budget, mode=mode)))
                methods.append(row)
                continue
            raise ValueError(f"Unsupported unified history arm: {arm}")
        if source.get("history_backend") != expected_backend:
            raise ValueError(f"Unified history backend differs from source arm: {arm}")
        if source.get("budget_variant"):
            methods.append(source)
            continue
        if source.get("recovery_policy", "off") != "off":
            raise ValueError("Primary unified history rows must have recovery_policy=off")
        source_budget = source.get("history_budget_tokens")
        shared = (source_budget == "shared"
                  or source.get("history_budget_source") == "shared")
        fixed = type(source_budget) is int and source_budget > 0 and not shared
        if (source_budget not in (None, "shared", budget) and not shared):
            if not fixed:
                raise ValueError(f"Unified history budget differs from global B: {arm}")
        benchmarks, _ = _scope(source)
        if (not benchmarks or len(benchmarks) != len(set(benchmarks))
                or not set(benchmarks) <= configured_benchmarks
                or not set(benchmarks) <= SUPPORTED_BENCHMARKS):
            raise ValueError(f"Unified history method needs an explicit portable benchmark scope: {arm}")
        if fixed:
            methods.append(source)
            continue
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
            row["arm"] = racer_v2_arm_name(expected_backend, "off", budget, "bare")
            row["group"] = "racer"
            row["history_allocation"] = "backend_native_persistent"
            row["racer_backend"] = resolve_racer_backend(expected_backend, "off", budget,
                                                            mode="bare")
        methods.append(row)
    identities = [(row["arm"], row.get("history_budget_tokens"), _scope(row))
                  for row in methods]
    if len(identities) != len(set(identities)):
        raise ValueError("Unified history runtime produced duplicate cell identities")
    return dict(config, methods=methods)


def with_racer_methods(config: dict, backends: tuple[str, ...],
                       policies: tuple[str, ...], history_budget_tokens: int | None,
                       *, protections: tuple[str, ...] | None = None) -> dict:
    """Add v3 independent axes, or explicitly retain the frozen v2 overlay."""
    if protections is not None:
        return _with_racer_v3_methods(config, backends, policies,
                                      history_budget_tokens, protections)
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

    def validate_existing(row: dict, backend: str, policy: str, mode: str,
                          expected_scope: tuple[tuple[str, ...], tuple[str, ...]]) -> None:
        arm = racer_v2_arm_name(backend, policy, history_budget_tokens, mode)
        if (row.get("arm") != arm or row.get("group") != "racer"
                or row.get("history_budget_tokens") != history_budget_tokens
                or row.get("racer_backend") != resolve_racer_backend(
                    backend, policy, history_budget_tokens, mode=mode)
                or _scope(row) != expected_scope
                or row.get("ratio") is not None or row.get("retention") is not None):
            raise ValueError(f"Existing RACER arm differs from its paired contract: {arm}")

    for backend in backends:
        bare_arm = ("c2kv_native_r8" if backend == "c2kv" else
                    racer_v2_arm_name(backend, "off", history_budget_tokens, "bare"))
        bare = None if backend == "c2kv" else unique(bare_arm)
        if backend == "c2kv":
            bare_rows = [row for row in resolved["methods"]
                         if row.get("history_runtime") == "racer"
                         and row.get("history_backend") == "c2kv"
                         and row.get("recovery_policy") == "off"
                         and not row.get("budget_variant")
                         and row.get("arm") in _UNIFIED_C2KV_ARMS]
            if len(bare_rows) > 1:
                raise ValueError("Unified C2KV has more than one bare primary")
            if bare_rows and bare:
                raise ValueError("C2KV has two bare primary identities")
            bare = bare_rows[0] if bare_rows else bare
            if bare and bare.get("history_budget_tokens") != history_budget_tokens:
                raise ValueError("Bare C2KV primary differs from RACER history budget")
        elif bare:
            validate_existing(bare, backend, "off", "bare", _scope(bare))
        if bare is None:
            bare = {
                "method": "C2KV" if backend == "c2kv" else f"RACER {backend} bare",
                "arm": "c2kv_native_r8" if backend == "c2kv" else bare_arm,
                "group": "main" if backend == "c2kv" else "racer",
                "history_runtime": "racer",
                "history_backend": backend,
                "recovery_policy": "off",
                "history_budget_tokens": history_budget_tokens,
                "benchmarks": list(configured_benchmarks),
            }
            if backend == "c2kv":
                if unique("c2kv_native_r8"):
                    raise ValueError("C2KV bare primary must be explicitly marked for RACER")
                bare["compression_ratio"] = 8
                bare["ratio"] = 8
                bare["history_allocation"] = "c2kv_bare"
            else:
                bare["racer_backend"] = resolve_racer_backend(
                    backend, "off", history_budget_tokens, mode="bare")
            resolved["methods"].append(bare)
            existing[bare["arm"]] = [bare]
        paired_scope = _scope(bare)
        for policy in policies:
            for mode in (("protected_off",) if policy == "off" else
                         ("protected_off", "on")):
                arm = racer_v2_arm_name(backend, policy, history_budget_tokens, mode)
                current = unique(arm)
                if current:
                    validate_existing(current, backend, policy, mode, paired_scope)
                    continue
                method = copy.deepcopy(bare)
                method.update(method=f"RACER {backend} {policy} {mode}",
                              arm=arm, group="racer", history_runtime="racer",
                              history_backend=backend,
                              recovery_policy=policy if mode == "on" else "off",
                              history_budget_tokens=history_budget_tokens,
                              history_allocation="racer_s0",
                              racer_backend=resolve_racer_backend(
                                  backend, policy, history_budget_tokens, mode=mode))
                method.pop("ratio", None)
                method.pop("history_budget_source", None)
                method.pop("budget_variant", None)
                if backend == "c2kv":
                    method["compression_ratio"] = _UNIFIED_C2KV_ARMS.get(bare["arm"], 8)
                resolved["methods"].append(method)
                existing[arm] = [method]
    return resolved


def _with_racer_v3_methods(config: dict, backends: tuple[str, ...],
                           policies: tuple[str, ...], history_budget_tokens: int | None,
                           protections: tuple[str, ...]) -> dict:
    if not backends and not policies and history_budget_tokens is None:
        return config
    if not backends or not policies or history_budget_tokens is None:
        raise ValueError("RACER requires backends, policies, and an explicit history budget")
    if len(backends) != len(set(backends)) or not set(backends) <= set(RACER_BACKENDS):
        raise ValueError("RACER backends must be unique supported names")
    if len(policies) != len(set(policies)) or not set(policies) <= set(RACER_POLICIES):
        raise ValueError("RACER policies must be unique supported names")
    if (not protections or len(protections) != len(set(protections))
            or not set(protections) <= {"off", "on"}):
        raise ValueError("RACER protections must be a nonempty unique subset of off,on")
    if type(history_budget_tokens) is not int or history_budget_tokens < 1:
        raise ValueError("RACER history budget must be a positive integer")
    configured_benchmarks = tuple(row["name"] for row in config["benchmarks"])
    if not configured_benchmarks or not set(configured_benchmarks) <= SUPPORTED_BENCHMARKS:
        raise ValueError("RACER requires the configured portable paper benchmarks")

    resolved = copy.deepcopy(config)
    existing = {}
    for row in resolved["methods"]:
        existing.setdefault(row["arm"], []).append(row)
    for backend in backends:
        native = next((row for row in resolved["methods"]
                       if row.get("history_runtime") == "racer"
                       and row.get("history_backend") == backend
                       and row.get("recovery_policy", "off") == "off"
                       and not row.get("budget_variant")
                       and (row.get("arm") in _UNIFIED_C2KV_ARMS
                            or (backend != "c2kv" and row.get("arm") == racer_v2_arm_name(
                                backend, "off", history_budget_tokens, "bare")))), None)
        scope = _scope(native) if native else (configured_benchmarks, ("raw",))
        for policy in policies:
            for protection in protections:
                arm = racer_v3_arm_name(backend, policy, history_budget_tokens, protection)
                racer = resolve_racer_backend(backend, policy, history_budget_tokens,
                                              extra_protection=protection)
                rows = existing.get(arm, ())
                if len(rows) > 1:
                    raise ValueError(f"RACER arm appears more than once in methods: {arm}")
                if rows:
                    row = rows[0]
                    if (row.get("group") != "racer" or
                            row.get("history_budget_tokens") != history_budget_tokens or
                            row.get("racer_backend") != racer or _scope(row) != scope or
                            row.get("recovery_policy") != policy or
                            row.get("ratio") is not None or row.get("retention") is not None):
                        raise ValueError(f"Existing RACER arm differs from its paired contract: {arm}")
                    continue
                row = {
                    "method": f"RACER {backend} {policy} protection {protection}",
                    "arm": arm, "group": "racer", "history_runtime": "racer",
                    "history_backend": backend, "recovery_policy": policy,
                    "history_budget_tokens": history_budget_tokens,
                    "history_allocation": racer["allocation"], "racer_backend": racer,
                    "benchmarks": list(scope[0]),
                }
                if scope[1] != ("raw",):
                    row["tool_contexts"] = list(scope[1])
                if backend == "c2kv":
                    row["compression_ratio"] = 8
                resolved["methods"].append(row)
                existing[arm] = [row]
    return resolved
