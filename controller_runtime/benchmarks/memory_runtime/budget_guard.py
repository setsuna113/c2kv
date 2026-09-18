"""Check the assembled history representation immediately before generation."""
from __future__ import annotations


def history_budget_receipt(memory, metadata, controller, *, ratio, phase):
    policy = getattr(controller, "policy_config", None)
    unit = getattr(controller, "kv_bytes_per_token", None)
    common = metadata.get("common_raw_prompt_tokens")
    if policy is None or type(unit) is not int or type(common) is not int:
        raise ValueError("Hybrid search requires an explicit native history budget contract")
    costs = memory.costs(ratio)
    resident = costs["resident_kv_tokens"]
    history_tokens = resident - common
    active = history_tokens * unit
    declared = metadata.get("actual_history_bytes")
    # Generality extension: a regeneration follows a recovery admission, so its
    # assembled view (initial memory + admitted evidence) is checked against
    # the post-recovery caps B; drafts always use the initial caps K.
    recovery_history = getattr(policy, "recovery_history_bytes", None)
    recovery_workspace = getattr(policy, "recovery_workspace_bytes", None)
    if phase == "regeneration" and recovery_history is not None and recovery_workspace is not None:
        cap = min(recovery_history, recovery_workspace)
        budget_mode = "recovery"
    else:
        cap = min(policy.history_budget_bytes, policy.workspace_budget_bytes)
        budget_mode = "initial"
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
        "history_budget_bytes": cap, "budget_mode": budget_mode,
        "active_history_bytes": active,
        "declared_active_history_bytes": declared, "kv_bytes_per_token": unit,
        "common_prompt_tokens": common, "resident_prompt_tokens": resident,
        "resident_prompt_kv_bytes": resident * unit,
        "active_gist_bytes": costs["gist_tokens"] * unit,
        "active_raw_and_derived_history_bytes": (history_tokens - costs["gist_tokens"]) * unit,
        "scope": "Assembled BF16 KV representation at loaded-model geometry; excludes weights, allocator overhead and CPU/disk archives.",
        "all_derived_workspace_content_charged": True,
    }
