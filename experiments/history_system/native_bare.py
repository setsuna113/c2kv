"""Device-independent native bare C2KV contract, separate from S0/C1."""
from __future__ import annotations

import copy
import json

ARM = "c2kv_native_r4"
METHOD = "c2kv_native"


def configure_design(design: dict) -> dict:
    """Keep explicit capacity limits; remove all S0 selection and recovery."""
    design = copy.deepcopy(design)
    design.update(route="ac_gist_static", ratio=4,
                  candidate_id=ARM, run_id_template=ARM,
                  compression_policy="always-compress-v1",
                  history_view_protocol="fixed-budget-main")
    design["runtime"].pop("controller", None)
    design["runtime"].pop("shadow_feature_config", None)
    return design


def profile() -> dict:
    return {"method": METHOD, "arm": ARM, "ratio": 4,
            "algorithm": "event-native static gist compression",
            "view_mode": "ac_gist_static", "detector": "disabled",
            "recovery_enabled": False, "s0_allocator_enabled": False,
            "max_generations_per_decision": 1,
            "comparison": "Independent native compression baseline; not a detector-only C1 ablation"}


def validate_manifest(path):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    route = manifest.get("route_contract", {})
    if (manifest.get("view_mode") != "ac_gist_static" or manifest.get("ratio") != 4
            or route.get("recovery_enabled") is not False
            or route.get("max_generations_per_decision") != 1
            or manifest.get("s0_controller_contract")):
        raise RuntimeError("Native bare server manifest differs from its declared method")
    return route
