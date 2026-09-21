"""Explicit ratio-8 candidate cell contract outside the legacy matrix."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

try:
    from .design import T02_RISK_ARTIFACT_SHA256
except ImportError:  # Direct file launch on ascend03.
    from design import T02_RISK_ARTIFACT_SHA256


LEGACY_VARIANTS = ("static_t02", "turn_c1", "goal_rescue", "dependency_first")
REPAIR_VARIANTS = ("request_contract", "argument_binding", "no_progress")
GOAL_VARIANTS = ("goal_pending", "goal_source", "goal_progress", "goal_joint")
VERIFIED_VARIANTS = ("goal_verified", "pending_verified")
STATIC_BACKBONES = {
    "goal_static": "goal_rescue",
    "pending_static": "goal_pending",
    "goal_verified_static": "goal_verified",
    "pending_verified_static": "pending_verified",
}
VERIFIED_STATIC_VARIANTS = ("goal_verified_static", "pending_verified_static")
STATIC_VARIANTS = tuple(STATIC_BACKBONES)
VARIANTS = LEGACY_VARIANTS + REPAIR_VARIANTS + GOAL_VARIANTS + VERIFIED_VARIANTS + STATIC_VARIANTS
REPAIR_VERSION = "c2kv-source-repair-v1"
GOAL_VERSION = "c2kv-goal-composition-v1"
VERIFIED_VERSION = "c2kv-verified-binding-v1"
PROOF_REGISTRY_VERSION = "verified-binding-rules-v1"
STATIC_VERSION = "c2kv-initial-view-composition-v1"
STATIC_INITIAL_VIEW_VERSION = "c2kv-static-initial-view-v1"
RATIO = 8
RISK_THRESHOLD = 0.5
RISK_ARTIFACT_SHA256 = T02_RISK_ARTIFACT_SHA256


def static_contract(variant: str) -> dict:
    contract = {
        "recovery_backbone": STATIC_BACKBONES[variant],
        "initial_view": {"policy": "static_gist", "version": STATIC_INITIAL_VIEW_VERSION},
    }
    if variant in VERIFIED_STATIC_VARIANTS:
        contract["proof_registry_version"] = PROOF_REGISTRY_VERSION
    return contract


def candidate_cell_from_source(
    source: dict, variant: str, backend_url: str,
    history_budget_tokens: int | None = None,
) -> dict:
    """Derive a separate, resumable cell from an existing BFCL B-budget manifest."""
    if variant not in VARIANTS:
        raise ValueError("Unknown candidate algorithm")
    if (source.get("backend") != "c2kv" or source.get("benchmark") != "bfcl"
            or source.get("benchmark_key") != "bfcl_base"):
        raise ValueError("Candidates require a C2KV BFCL base source cell")
    if source.get("condition") != "compression_full_budget":
        raise ValueError("Candidates require the B-budget compression_full_budget source cell")
    if source.get("ratio") != 4:
        raise ValueError("Candidates require a frozen ratio-4 source cell")
    if not isinstance(backend_url, str) or not backend_url.strip():
        raise ValueError("Candidates require an explicit SGLang backend URL")
    if history_budget_tokens is not None and (
            type(history_budget_tokens) is not int or history_budget_tokens <= 0):
        raise ValueError("history_budget_tokens must be a positive integer")
    source_dir = Path(source["cell_dir"])
    cell = copy.deepcopy(source)
    cell["schema"] = ("c2kv-generality-candidate-cell-v5"
                      if variant in STATIC_VARIANTS else
                      "c2kv-generality-candidate-cell-v4"
                      if variant in VERIFIED_VARIANTS else
                      "c2kv-generality-candidate-cell-v3"
                      if variant in GOAL_VARIANTS else
                      "c2kv-generality-candidate-cell-v2"
                      if variant in REPAIR_VARIANTS else
                      "c2kv-generality-candidate-cell-v1")
    cell["cell_id"] = f"{source['cell_id']}__candidate_{variant}"
    cell["cell_dir"] = str(source_dir.parent / "candidate_algorithms" / variant)
    cell["condition"] = "candidate_algorithm"
    cell["candidate_algorithm"] = variant
    cell["candidate_source_cell_id"] = source["cell_id"]
    cell["candidate_budget_source"] = "working_point.common_cap_bytes"
    cell["ratio"] = RATIO
    if history_budget_tokens is not None:
        suffix = f"b{history_budget_tokens}"
        cell["cell_id"] += f"__{suffix}"
        cell["cell_dir"] = str(Path(cell["cell_dir"]) / suffix)
        cell["model_name"] = f"{source.get('model_name', source['cell_id'])}__candidate_{variant}__{suffix}"
        cell["history_budget_tokens"] = history_budget_tokens
        cell["candidate_budget_source"] = "explicit.native_history_budget_tokens"
    if variant in REPAIR_VARIANTS:
        cell.pop("threshold", None)
        cell.pop("threshold_status", None)
        cell["candidate_protocol"] = REPAIR_VERSION
    else:
        cell["threshold"] = RISK_THRESHOLD
        cell["threshold_status"] = "frozen_candidate"
        if variant in GOAL_VARIANTS:
            cell["candidate_protocol"] = GOAL_VERSION
        elif variant in VERIFIED_VARIANTS:
            cell["candidate_protocol"] = VERIFIED_VERSION
            cell["proof_registry_version"] = PROOF_REGISTRY_VERSION
        elif variant in STATIC_VARIANTS:
            cell["candidate_protocol"] = STATIC_VERSION
            cell.update(static_contract(variant))
    cell["sglang_backend_url"] = backend_url.strip()
    cell.pop("controller_path", None)
    cell.pop("eval_policy_path", None)
    return cell


def controller_with_binding(
    cell: dict,
    *,
    base_controller: dict,
    selected: dict,
    risk_artifact_path: Path,
    bind_risk_artifact,
) -> tuple[dict, dict]:
    """Bind frozen T02 weights to the exact selected C1000 checkpoint."""
    if cell.get("candidate_algorithm") not in VARIANTS or cell.get("ratio") != RATIO:
        raise ValueError("Invalid ratio-8 candidate cell")
    variant = cell["candidate_algorithm"]
    if cell.get("benchmark") != "bfcl":
        raise ValueError("Candidate cells require BFCL")
    if variant in REPAIR_VARIANTS:
        if cell.get("candidate_protocol") != REPAIR_VERSION or "threshold" in cell:
            raise ValueError("Repair candidate requires the source-repair protocol without T02")
    elif variant in VERIFIED_VARIANTS:
        if (cell.get("candidate_protocol") != VERIFIED_VERSION
                or cell.get("proof_registry_version") != PROOF_REGISTRY_VERSION
                or cell.get("threshold") != RISK_THRESHOLD):
            raise ValueError("Verified candidate requires frozen T02 and proof registry")
    elif variant in STATIC_VARIANTS:
        contract = static_contract(variant)
        if (variant in VERIFIED_STATIC_VARIANTS
                and cell.get("proof_registry_version") != PROOF_REGISTRY_VERSION):
            raise ValueError("Verified static candidate requires proof registry")
        if (cell.get("schema") != "c2kv-generality-candidate-cell-v5"
                or cell.get("candidate_protocol") != STATIC_VERSION
                or cell.get("threshold") != RISK_THRESHOLD
                or any(cell.get(key) != value for key, value in contract.items())):
            raise ValueError("Static candidate requires matching initial view and recovery backbone")
    elif variant in GOAL_VARIANTS:
        if cell.get("candidate_protocol") != GOAL_VERSION or cell.get("threshold") != RISK_THRESHOLD:
            raise ValueError("Goal candidate requires the frozen T02 contract")
    elif cell.get("threshold") != RISK_THRESHOLD:
        raise ValueError("Candidate cells require the frozen T02 threshold")
    checkpoint = Path(cell["checkpoint"])
    config_sha256 = hashlib.sha256((checkpoint / "config.json").read_bytes()).hexdigest()
    expected = selected["checkpoint_selection"]["config_sha256"]
    if config_sha256 != expected:
        raise ValueError("Candidate checkpoint differs from selected C1000")
    controller = copy.deepcopy(base_controller)
    for key in ("gp_experiments", "post_draft_recovery", "d3_hybrid_recovery"):
        controller.pop(key, None)
    if variant in REPAIR_VARIANTS:
        controller["candidate_algorithm"] = {"variant": variant}
        return controller, None
    artifact_bytes = risk_artifact_path.read_bytes()
    if hashlib.sha256(artifact_bytes).hexdigest() != RISK_ARTIFACT_SHA256:
        raise ValueError("T02 risk artifact differs from evaluated release")
    artifact = json.loads(artifact_bytes)
    bound, binding = bind_risk_artifact(artifact, checkpoint)
    controller["candidate_algorithm"] = {
        "variant": variant,
        "risk_artifact": bound,
        "risk_threshold": RISK_THRESHOLD,
    }
    if variant in VERIFIED_VARIANTS:
        controller["candidate_algorithm"]["proof_registry_version"] = PROOF_REGISTRY_VERSION
    elif variant in STATIC_VARIANTS:
        controller["candidate_algorithm"].update(static_contract(variant))
    return controller, binding
