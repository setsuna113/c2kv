"""Build explicit ratio-8 candidate controllers from the frozen C1000 delivery."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable


VARIANTS = ("static_t02", "turn_c1", "goal_rescue", "dependency_first")
REPAIR_VARIANTS = ("request_contract", "argument_binding", "no_progress")
ALL_VARIANTS = VARIANTS + REPAIR_VARIANTS
RATIO = 8


def build_profile(
    args: Any,
    *,
    base_controller: Mapping[str, Any],
    selected: Mapping[str, Any],
    risk_artifact_path: Path,
    risk_artifact_sha256: str,
    bind_artifact: Callable[[dict[str, Any], Path], tuple[dict[str, Any], dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep candidate experiments separate from C1 and D3 delivery contracts."""
    variant = args.candidate_algorithm
    if variant not in ALL_VARIANTS:
        raise ValueError(f"unknown candidate algorithm: {variant!r}")
    if args.method != "proposed":
        raise ValueError("--candidate-algorithm requires --method proposed")
    if getattr(args, "benchmark", "bfcl") not in {"bfcl", "acebench", "acon_appworld"}:
        raise ValueError("candidate algorithms support BFCL, ACEBench Agent and AppWorld only")
    if args.selector_artifact is not None:
        if variant in VARIANTS:
            raise ValueError("candidate algorithms use the bundled T02 risk artifact")
        raise ValueError("repair candidates do not use a selector artifact")
    ratio = int(args.ratio) if args.ratio is not None else int(selected["ratio"])
    if ratio != RATIO:
        raise ValueError("candidate algorithms require ratio 8")

    actual_checkpoint = hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest()
    if actual_checkpoint != selected["checkpoint_selection"]["config_sha256"]:
        raise ValueError("candidate algorithms require the selected C1000 checkpoint config")
    controller = copy.deepcopy(dict(base_controller))
    for key in ("gp_experiments", "post_draft_recovery", "d3_hybrid_recovery"):
        controller.pop(key, None)
    if variant in REPAIR_VARIANTS:
        controller["candidate_algorithm"] = {"variant": variant}
        profile = {
            "schema": "c2kv-candidate-delivery-profile-v2",
            "method": "proposed",
            "detector": "candidate_algorithm",
            "candidate_algorithm": variant,
            "algorithm": f"C2KV candidate {variant}",
            "new_c1_training_claimed": False,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_config_sha256": actual_checkpoint,
            "selection_protocol": "c2kv-source-repair-v1",
            "history_variant": "H0",
            "ratio": ratio,
            "controller_sha256": hashlib.sha256(
                json.dumps(controller, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "automatic_reruns": 0,
        }
        return controller, profile

    if args.detector != "t02_risk":
        raise ValueError("candidate algorithms use the frozen T02 risk detector")
    if args.selector_threshold != 0.5:
        raise ValueError("candidate algorithms require the frozen risk threshold 0.5")
    artifact_file_sha256 = hashlib.sha256(risk_artifact_path.read_bytes()).hexdigest()
    if artifact_file_sha256 != risk_artifact_sha256:
        raise ValueError("bundled T02 risk artifact differs from the evaluated release")
    source_artifact = json.loads(risk_artifact_path.read_text(encoding="utf-8"))
    bound_artifact, binding = bind_artifact(source_artifact, args.checkpoint)
    controller["candidate_algorithm"] = {
        "variant": variant,
        "risk_artifact": bound_artifact,
        "risk_threshold": 0.5,
    }
    profile = {
        "schema": "c2kv-candidate-delivery-profile-v1",
        "method": "proposed",
        "detector": "candidate_algorithm",
        "candidate_algorithm": variant,
        "algorithm": f"C2KV candidate {variant}",
        "new_c1_training_claimed": False,
        "selector_artifact": str(risk_artifact_path.resolve()),
        "selector_artifact_sha256": artifact_file_sha256,
        "selector_artifact_binding": binding,
        "selector_threshold": 0.5,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config_sha256": actual_checkpoint,
        "selection_protocol": "candidate_algorithm_v1",
        "history_variant": "H0",
        "ratio": ratio,
        "controller_sha256": hashlib.sha256(
            json.dumps(controller, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "automatic_reruns": 0,
    }
    return controller, profile
