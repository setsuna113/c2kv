"""Read generation cost from proxy traces without losing a discarded draft."""
from __future__ import annotations

import math
from collections.abc import Mapping


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens")
BACKEND_PEAK_FIELDS = (
    "kv_resident_tokens",
    "kv_peak_resident_tokens",
    "total_gpu_kv_bytes",
    "peak_total_gpu_kv_bytes",
)
RUNTIME_PEAK_FIELDS = ("active_history_bytes", "evidence_bytes")
EXTRACTION_COUNT_FIELDS = (
    "lookups",
    "client_cache_hits",
    "producer_calls",
    "producer_successes",
    "producer_failures",
)
EXTRACTION_WALL_FIELDS = ("lookup_wall_sec", "producer_wall_sec")
SERVER_ALLOCATOR_SCOPE = (
    "process allocator snapshot including shared caches and concurrent requests; "
    "maxima are not request-exclusive"
)
EXTRACTION_SCOPE = (
    "whole proxy HTTP request; lookup_wall_seconds and producer_wall_seconds "
    "overlap and must not be added"
)


def _tokens(value):
    return value if type(value) is int and value >= 0 else None


def _resource_number(value, label, *, integer=False):
    if value is None:
        return None
    valid = (
        type(value) is int
        if integer
        else isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    if not valid or not math.isfinite(value) or value < 0:
        kind = "nonnegative integer" if integer else "finite nonnegative number"
        raise ValueError(f"{label} must be a {kind} or null")
    return value


def _optional_mapping(value, label):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object or null")
    return value


def _strict_sum(values):
    known = [value for value in values if value is not None]
    strict = sum(known) if values and len(known) == len(values) else None
    return strict, sum(known) if known else None


def _strict_max(values):
    known = [value for value in values if value is not None]
    strict = max(known) if values and len(known) == len(values) else None
    return strict, max(known) if known else None


def _set_strict_pair(result, key, values, reducer):
    strict, known = reducer(values)
    result[key] = strict
    result[f"{key}_known_lower_bound"] = known


def _extraction_summary(row):
    telemetry = _optional_mapping(row.get("extraction_telemetry"),
                                  "extraction_telemetry")
    return _optional_mapping(telemetry.get("summary"),
                             "extraction_telemetry.summary")


def request_generation_cost(row):
    """Validate trace totals and preserve unknown cost on incomplete attempts.

    Standard response usage remains the final generation. A traced request's
    billable usage is the sum across its draft and optional regeneration.
    Historical untraced rows retain their existing single-response usage.
    """
    if "generation_trace" not in row:
        if "generation_attempts" in row or "generation_usage_total" in row:
            raise ValueError("Generation totals lack their per-attempt trace")
        usage = row.get("usage") or {}
        values = {field: _tokens(usage.get(field)) for field in TOKEN_FIELDS}
        return {"usage": values, "known_usage": {k: v or 0 for k, v in values.items()},
                "attempts": 1 if row.get("status") == "ok" else None,
                "scope": "legacy_final_response"}

    trace = row["generation_trace"]
    if not isinstance(trace, list) or len(trace) > 2:
        raise ValueError("Exact recovery needs a trace of at most two attempts")
    if type(row.get("generation_attempts")) is not int or row["generation_attempts"] != len(trace):
        raise ValueError("Generation attempt count disagrees with trace")
    totals = row.get("generation_usage_total")
    if not isinstance(totals, Mapping):
        raise ValueError("Generation trace lacks its usage totals")
    completed = 0
    for index, record in enumerate(trace):
        if not isinstance(record, Mapping):
            raise ValueError("Generation trace record must be an object")
        expected_phase = "draft" if index == 0 else "regeneration"
        if record.get("phase") != expected_phase:
            raise ValueError("Generation phases are not draft then regeneration")
        completed += record.get("status") == "completed"
        if row.get("status") == "ok" and (
                record.get("status") != "completed" or record.get("backend_verified") is not True):
            raise ValueError("Successful exact request has an incomplete/unverified generation")
    if row.get("generation_completed") != completed:
        raise ValueError("Generation completion count disagrees with trace")
    if row.get("status") == "ok" and not trace:
        raise ValueError("Successful exact request has no generation")
    if len(trace) == 2 and trace[0].get("discarded") is not True:
        raise ValueError("Regenerated request did not mark its draft discarded")
    values, known = {}, {}
    for field in TOKEN_FIELDS:
        parts = [_tokens((record.get("usage") or {}).get(field)) for record in trace]
        values[field] = sum(parts) if parts and all(v is not None for v in parts) else None
        known[field] = sum(v for v in parts if v is not None)
        if totals.get(field) != values[field]:
            raise ValueError(f"Generation total {field} disagrees with per-attempt usage")
    if row.get("status") == "ok":
        final_usage = row.get("usage") or {}
        recorded_final = trace[-1].get("usage") or {}
        if any(final_usage.get(field) != recorded_final.get(field) for field in TOKEN_FIELDS):
            raise ValueError("Standard usage differs from the final generation")
    return {"usage": values, "known_usage": known, "attempts": len(trace),
            "scope": "all_generation_attempts"}


def summed_generation_cost(rows):
    costs = [request_generation_cost(row) for row in rows]
    usage = {}
    for field in TOKEN_FIELDS:
        parts = [cost["usage"][field] for cost in costs]
        usage[field] = sum(parts) if all(v is not None for v in parts) else None
    attempts = [cost["attempts"] for cost in costs]
    return {
        **usage,
        "generation_attempts": sum(attempts) if all(v is not None for v in attempts) else None,
        "known_prompt_tokens": sum(cost["known_usage"]["prompt_tokens"] for cost in costs),
        "known_completion_tokens": sum(cost["known_usage"]["completion_tokens"] for cost in costs),
        "generation_cost_scope": "all traced attempts plus legacy final responses; null totals are unknown, known tokens are lower bounds",
    }


def request_generation_resources(row):
    """Return resource costs for every generation made by one proxy request.

    A traced exact request carries one backend/runtime snapshot per generation.
    The initial controller time and the final exact reconsideration time are
    disjoint; the second generation's controller snapshot is contained in the
    latter and must not be added again. Extraction telemetry already spans the
    complete proxy HTTP request.
    """
    request_generation_cost(row)  # Validate trace/header/final usage first.
    traced = "generation_trace" in row
    if traced:
        trace = row["generation_trace"]
        costs = [
            _optional_mapping(record.get("cost"),
                              f"generation_trace[{index}].cost")
            for index, record in enumerate(trace)
        ]
        runtimes = [
            _optional_mapping(record.get("memory_runtime"),
                              f"generation_trace[{index}].memory_runtime")
            for index, record in enumerate(trace)
        ]
        scope = "all_generation_attempts"
    else:
        trace = []
        costs = [row]
        runtimes = [_optional_mapping(row.get("memory_runtime"), "memory_runtime")]
        scope = "legacy_final_response"

    result = {
        "resource_scope": scope,
        "server_allocator_scope": SERVER_ALLOCATOR_SCOPE,
        "extraction_scope": EXTRACTION_SCOPE,
    }
    for field in BACKEND_PEAK_FIELDS:
        values = [
            _resource_number(cost.get(field),
                             f"generation cost {field}", integer=True)
            for cost in costs
        ]
        _set_strict_pair(result, f"{field}_max", values, _strict_max)
    for field in RUNTIME_PEAK_FIELDS:
        values = [
            _resource_number(runtime.get(field),
                             f"generation memory_runtime {field}", integer=True)
            for runtime in runtimes
        ]
        _set_strict_pair(result, f"{field}_max", values, _strict_max)

    if traced:
        gist_values = [
            _resource_number(runtime.get("gist_tokens"),
                             "generation memory_runtime gist_tokens", integer=True)
            for runtime in runtimes
        ]
    else:
        gist_values = [
            _resource_number(row.get("gist_tokens"), "gist_tokens", integer=True)
        ]
    _set_strict_pair(result, "gist_tokens_sum", gist_values, _strict_sum)
    _set_strict_pair(result, "gist_tokens_max", gist_values, _strict_max)

    if traced:
        initial_runtime = runtimes[0] if runtimes else {}
        final_runtime = _optional_mapping(row.get("memory_runtime"), "memory_runtime")
        exact_recovery = _optional_mapping(final_runtime.get("exact_recovery"),
                                           "memory_runtime.exact_recovery")
        controller_values = [
            _resource_number(initial_runtime.get("controller_wall_sec"),
                             "initial controller_wall_sec"),
            _resource_number(exact_recovery.get("controller_wall_sec"),
                             "exact_recovery.controller_wall_sec"),
        ]
        assembly_values = [
            _resource_number(initial_runtime.get("compressed_assembly_wall_sec"),
                             "initial compressed_assembly_wall_sec")
        ]
    else:
        runtime = runtimes[0]
        controller_values = [
            _resource_number(runtime.get("controller_wall_sec"),
                             "memory_runtime.controller_wall_sec")
        ]
        assembly_values = [
            _resource_number(runtime.get("compressed_assembly_wall_sec"),
                             "memory_runtime.compressed_assembly_wall_sec")
        ]
    _set_strict_pair(result, "controller_wall_seconds", controller_values,
                     _strict_sum)
    _set_strict_pair(result, "compressed_assembly_wall_seconds", assembly_values,
                     _strict_sum)

    extraction = _extraction_summary(row)
    for field in EXTRACTION_COUNT_FIELDS:
        values = [
            _resource_number(extraction.get(field),
                             f"extraction_telemetry.summary.{field}", integer=True)
        ]
        _set_strict_pair(result, f"extraction_{field}", values, _strict_sum)
    for field in EXTRACTION_WALL_FIELDS:
        key = f"extraction_{field.removesuffix('_sec')}_seconds"
        values = [
            _resource_number(extraction.get(field),
                             f"extraction_telemetry.summary.{field}")
        ]
        _set_strict_pair(result, key, values, _strict_sum)
    return result


def summed_generation_resources(rows):
    """Aggregate request resources without zero-imputing missing telemetry."""
    resources = [request_generation_resources(row) for row in rows]
    scopes = {resource["resource_scope"] for resource in resources}
    if not scopes:
        scope = "no_requests"
    elif len(scopes) == 1:
        scope = next(iter(scopes))
    else:
        scope = "all_generation_attempts_plus_legacy_final_responses"
    result = {
        "resource_scope": scope,
        "server_allocator_scope": SERVER_ALLOCATOR_SCOPE,
        "extraction_scope": EXTRACTION_SCOPE,
    }
    peak_keys = tuple(f"{field}_max" for field in (
        *BACKEND_PEAK_FIELDS, *RUNTIME_PEAK_FIELDS, "gist_tokens"))
    sum_keys = (
        "gist_tokens_sum",
        "controller_wall_seconds",
        "compressed_assembly_wall_seconds",
        *(f"extraction_{field}" for field in EXTRACTION_COUNT_FIELDS),
        *(f"extraction_{field.removesuffix('_sec')}_seconds"
          for field in EXTRACTION_WALL_FIELDS),
    )
    for key in peak_keys:
        strict_values = [resource[key] for resource in resources]
        known_values = [
            resource[f"{key}_known_lower_bound"] for resource in resources
            if resource[f"{key}_known_lower_bound"] is not None
        ]
        result[key] = (
            max(strict_values)
            if strict_values and all(value is not None for value in strict_values)
            else None
        )
        result[f"{key}_known_lower_bound"] = (
            max(known_values) if known_values else None
        )
    for key in sum_keys:
        strict_values = [resource[key] for resource in resources]
        known_values = [
            resource[f"{key}_known_lower_bound"] for resource in resources
            if resource[f"{key}_known_lower_bound"] is not None
        ]
        result[key] = (
            sum(strict_values)
            if strict_values and all(value is not None for value in strict_values)
            else None
        )
        result[f"{key}_known_lower_bound"] = (
            sum(known_values) if known_values else None
        )
    return result
