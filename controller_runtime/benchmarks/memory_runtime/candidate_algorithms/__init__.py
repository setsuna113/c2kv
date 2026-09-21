"""Opt-in whole-system candidates; legacy routes remain unchanged."""

VARIANTS = ("static_t02", "turn_c1", "goal_rescue", "dependency_first")
VERSION = "c2kv-paper-candidates-v1"

# Keep the historical collection frozen for old profiles and callers.
REPAIR_VARIANTS = ("request_contract", "argument_binding", "no_progress")
GOAL_VARIANTS = ("goal_pending", "goal_source", "goal_progress", "goal_joint")
GOAL_VERSION = "c2kv-goal-composition-v1"
VERIFIED_VARIANTS = ("goal_verified", "pending_verified")
VERIFIED_VERSION = "c2kv-verified-binding-v1"
INITIAL_VIEW_BACKBONES = {
    "goal_static": "goal_rescue", "pending_static": "goal_pending",
    "goal_verified_static": "goal_verified",
    "pending_verified_static": "pending_verified",
}
INITIAL_VIEW_VARIANTS = tuple(INITIAL_VIEW_BACKBONES)
INITIAL_VIEW_VERSION = "c2kv-initial-view-composition-v1"
INITIAL_VIEW_POLICY_VERSION = "c2kv-static-initial-view-v1"
STATIC_EXTENSION_VARIANTS = ("static_verified", "static_action_ledger")
STATIC_EXTENSION_VERSION = "c2kv-static-extension-v1"
ALL_VARIANTS = (VARIANTS + REPAIR_VARIANTS + GOAL_VARIANTS
                + VERIFIED_VARIANTS + INITIAL_VIEW_VARIANTS + STATIC_EXTENSION_VARIANTS)
