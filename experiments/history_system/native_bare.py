"""Device-independent native bare C2KV contract, separate from S0/C1."""
from __future__ import annotations

import copy
import json

ARM = "c2kv_native_r4"
ARM_RATIOS = {"c2kv_native_r4": 4, "c2kv_native_r8": 8}
METHOD = "c2kv_native"


def arm_for_ratio(ratio: int) -> str:
    if type(ratio) is not int or ratio not in ARM_RATIOS.values():
        raise ValueError("Native bare C2KV supports ratios 4 and 8")
    return f"c2kv_native_r{ratio}"


def configure_design(design: dict, ratio: int = 4) -> dict:
    """Keep explicit capacity limits; remove all S0 selection and recovery."""
    design = copy.deepcopy(design)
    arm = arm_for_ratio(ratio)
    design.update(route="ac_gist_static", ratio=ratio,
                  candidate_id=arm, run_id_template=arm,
                  compression_policy="always-compress-v1",
                  history_view_protocol="fixed-budget-main")
    design["runtime"].pop("controller", None)
    design["runtime"].pop("shadow_feature_config", None)
    return design


def profile(ratio: int = 4) -> dict:
    return {"method": METHOD, "arm": arm_for_ratio(ratio), "ratio": ratio,
            "algorithm": "event-native static gist compression",
            "view_mode": "ac_gist_static", "detector": "disabled",
            "recovery_enabled": False, "s0_allocator_enabled": False,
            "max_generations_per_decision": 1,
            "comparison": "Independent native compression baseline; not a detector-only C1 ablation"}


def validate_manifest(path, ratio: int = 4):
    arm = arm_for_ratio(ratio)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    route = manifest.get("route_contract", {})
    if (manifest.get("view_mode") != "ac_gist_static" or manifest.get("ratio") != ratio
            or manifest.get("model_name") != arm
            or route.get("recovery_enabled") is not False
            or route.get("max_generations_per_decision") != 1
            or manifest.get("s0_controller_contract")):
        raise RuntimeError("Native bare server manifest differs from its declared method")
    return route
