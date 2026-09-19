"""Check the assembled history representation immediately before generation."""
from __future__ import annotations

from .event_native_exact_policy import EventNativeExactController


def history_budget_receipt(memory, metadata, controller, *, ratio, phase):
    policy = getattr(controller, "policy_config", None)
    unit = getattr(controller, "kv_bytes_per_token", None)
    common = metadata.get("common_raw_prompt_tokens")
    if policy is None or type(unit) is not int:
        raise ValueError("Hybrid search requires an explicit native history budget contract")
    base_controller = getattr(controller, "inner", controller)
    if isinstance(base_controller, EventNativeExactController):
        # Exact controllers enforce their own history capacity during prepare and
        # report Full-render common_live_tokens, not the S0 common-input boundary.
        # AppWorld also reports a task-packet common value, but it has different
        # accounting and must not select this S0 check.
        if "common_raw_prompt_tokens" in metadata and type(common) is not int:
            raise ValueError("Hybrid search requires an explicit native history budget contract")
        return {
            "schema": "a-hybrid-pre-generation-budget-v1", "phase": phase,
            "status": "not_applicable", "errors": [],
            "reason": "exact controller uses its own capacity accounting, not S0 common-input accounting",
            "controller": type(controller).__name__,
            "declared_active_history_bytes": metadata.get("actual_history_bytes"),
            "kv_bytes_per_token": unit,
        }
    if type(common) is not int:
        raise ValueError("Hybrid search requires an explicit native history budget contract")
    costs = memory.costs(ratio)
    history_gist_tokens = metadata.get("history_only_gist_tokens", costs["gist_tokens"])
    if type(history_gist_tokens) is not int or history_gist_tokens > costs["gist_tokens"]:
        raise ValueError("Invalid separate tool/history gist accounting")
    resident = metadata.get("history_only_resident_kv_tokens", costs["resident_kv_tokens"])
    if type(resident) is not int or resident < 0 or (
        "history_only_resident_kv_tokens" not in metadata
        and resident > costs["resident_kv_tokens"]
    ):
        raise ValueError("Invalid separate tool/history resident accounting")
    history_tokens = resident - common
    active = history_tokens * unit
    declared = metadata.get("actual_history_bytes")
    cap = min(policy.history_budget_bytes, policy.workspace_budget_bytes)
    errors = []
    if history_tokens < 0:
        errors.append("negative_history_after_common_boundary")
    if active != declared:
        errors.append("assembled_history_differs_from_controller_accounting")
    if active > cap:
        errors.append("active_history_exceeds_budget")
    return {
        "schema": "a-hybrid-pre-generation-budget-v1", "phase": phase,
        "status": "rejected" if errors else "passed", "errors": errors,
        "history_budget_bytes": cap, "active_history_bytes": active,
        "declared_active_history_bytes": declared, "kv_bytes_per_token": unit,
        "common_prompt_tokens": common, "resident_prompt_tokens": resident,
        **({"tool_resident_delta_tokens": costs["resident_kv_tokens"] - resident,
            "combined_resident_prompt_tokens": costs["resident_kv_tokens"]}
           if "history_only_resident_kv_tokens" in metadata else {}),
        "resident_prompt_kv_bytes": resident * unit,
        "active_gist_bytes": history_gist_tokens * unit,
        "active_raw_and_derived_history_bytes": (history_tokens - history_gist_tokens) * unit,
        "scope": "Assembled BF16 KV representation at loaded-model geometry; excludes weights, allocator overhead and CPU/disk archives.",
        "all_derived_workspace_content_charged": True,
    }
