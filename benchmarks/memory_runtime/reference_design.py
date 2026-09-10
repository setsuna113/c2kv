"""Frozen reference-dev2 design and self-contained selection validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

try:
    from . import shared_exact_design as shared_exact
except ImportError:
    import shared_exact_design as shared_exact


HERE = Path(__file__).resolve().parent
DESIGN_NAME = "reference-dev2"
DESIGN_FILE = "benchmarks/memory_runtime/configs/reference_dev2.json"
VARIANTS = (
    "full", "capacity_protect", "capacity_exact_once", "capacity_exact_no_gist",
)
ARM_BY_VARIANT = {
    "full": "full",
    "capacity_protect": "c2kv4",
    "capacity_exact_once": "c2kv4",
    "capacity_exact_no_gist": "full",
}
EXACT_VARIANTS = frozenset({"capacity_exact_once", "capacity_exact_no_gist"})
CONFIG_SOURCES = {"capacity_protect": "protect"}
EXECUTION_SOURCE_FILES = (DESIGN_FILE, "benchmarks/memory_runtime/reference_design.py")


def _numeric_task_id(name: str) -> int:
    prefix = "multi_turn_base_"
    if not isinstance(name, str) or not name.startswith(prefix) or not name[len(prefix):].isdigit():
        raise ValueError(f"Invalid reference-dev2 task ID: {name!r}")
    return int(name[len(prefix):])


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"Invalid reference-dev2 {label}")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"Invalid reference-dev2 {label}") from error
    return value


def load_design() -> dict:
    path = HERE / "configs/reference_dev2.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    validate_design(spec)
    return spec


def design_source() -> dict[str, str]:
    path = HERE / "configs/reference_dev2.json"
    return {"path": DESIGN_FILE, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_design(spec: dict) -> None:
    if (not isinstance(spec, dict)
            or spec.get("schema") != "a-reference-dev-design-v1"
            or spec.get("design") != DESIGN_NAME):
        raise ValueError("Unknown reference-dev2 design")
    if spec.get("variants") != [[name, ARM_BY_VARIANT[name]] for name in VARIANTS]:
        raise ValueError("Reference-dev2 variants or plain backend arms differ")

    selection = spec.get("task_selection")
    if not isinstance(selection, dict):
        raise ValueError("Reference-dev2 task selection is missing")
    source = selection.get("source_manifest")
    policy = selection.get("policy")
    result = selection.get("result")
    if not all(isinstance(value, dict) for value in (source, policy, result)):
        raise ValueError("Reference-dev2 must embed source, selection policy, and result")
    pool = source.get("ids")
    if (not isinstance(pool, list) or source.get("n_total") != len(pool)
            or len(pool) != len(set(pool))):
        raise ValueError("Reference-dev2 source pool lacks unique counted IDs")
    for task_id in pool:
        _numeric_task_id(task_id)
    if policy.get("schema") != "a-reference-dev2-static-selection-v1":
        raise ValueError("Reference-dev2 selection policy schema differs")
    if (policy.get("pool") != pool
            or policy.get("frozen_before_question_export") is not True
            or policy.get("held_out") is not False
            or policy.get("score_or_response_filtering") is not False):
        raise ValueError("Reference-dev2 selection policy is not the frozen dev-only policy")
    source_sha = _sha256(source.get("source_sha256"), "source SHA-256")
    if policy.get("expected_source_sha256") != source_sha:
        raise ValueError("Reference-dev2 policy does not bind the source manifest")

    prior = policy.get("prior_a_pilots")
    if not isinstance(prior, list) or any(not isinstance(item, dict) for item in prior):
        raise ValueError("Reference-dev2 prior-pilot exclusions are malformed")
    excluded = sorted(
        {task_id for item in prior for task_id in item.get("task_ids", [])},
        key=_numeric_task_id,
    )
    if sorted(policy.get("excluded_prior_a_ids", []), key=_numeric_task_id) != excluded:
        raise ValueError("Reference-dev2 prior-pilot exclusions do not reproduce")
    if result.get("schema") != "a-reference-dev2-static-selection-result-v1":
        raise ValueError("Reference-dev2 selection result schema differs")
    if (selection.get("held_out") is not False
            or selection.get("score_or_response_filtering") is not False
            or result.get("held_out") is not False
            or result.get("score_or_response_filtering") is not False):
        raise ValueError("Reference-dev2 result must not use outcomes or claim held-out status")

    policy_sha = _sha256(selection.get("policy_sha256"), "policy SHA-256")
    result_sha = _sha256(selection.get("result_sha256"), "result SHA-256")
    question_sha = _sha256(selection.get("question_export_sha256"), "question export SHA-256")
    if result.get("policy_sha256") != policy_sha or result.get("question_export_sha256") != question_sha:
        raise ValueError("Reference-dev2 selection result does not bind its policy/questions")
    for key in ("policy_path", "result_path", "question_export_path"):
        if not isinstance(selection.get(key), str) or not selection[key]:
            raise ValueError(f"Reference-dev2 {key} is missing")
    del result_sha  # Its presence binds the referenced selection artifact in the spec.

    rows = result.get("all_rows")
    if (not isinstance(rows, list) or len(rows) != len(pool)
            or {row.get("task_id") for row in rows if isinstance(row, dict)} != set(pool)):
        raise ValueError("Reference-dev2 result does not cover the source pool")
    eligible_rows = [row for row in rows if row.get("eligible") is True]
    expected_ranking = [row["task_id"] for row in sorted(
        eligible_rows,
        key=lambda row: (-row["reference_turn_count"], -row["user_turn_count"],
                         -row["total_user_characters"], _numeric_task_id(row["task_id"])),
    )]
    if (result.get("eligible_count") != len(eligible_rows)
            or result.get("ranked_eligible_ids") != expected_ranking):
        raise ValueError("Reference-dev2 embedded ranking does not reproduce")
    sample_size = min(policy.get("sample_size_max", 0), len(expected_ranking))
    selected = expected_ranking[:sample_size]
    if selected != result.get("task_ids") or selected != spec.get("task_ids"):
        raise ValueError("Reference-dev2 task IDs do not preserve the frozen rank order")

    positive_ints = (
        "history_budget_bytes", "workspace_budget_bytes", "bytes_per_kv_token",
        "lease_decisions", "max_retrieved_events", "maximum_tasks",
        "maximum_wall_seconds", "maximum_generation_attempts_per_arm",
        "maximum_generation_attempts", "maximum_extraction_attempts_per_arm",
        "maximum_extraction_attempts",
    )
    if any(type(spec.get(key)) is not int or spec[key] <= 0 for key in positive_ints):
        raise ValueError("Reference-dev2 contains an invalid positive budget")
    if spec["maximum_tasks"] != len(selected) * len(VARIANTS):
        raise ValueError("Reference-dev2 task-arm count differs")
    if spec["maximum_generation_attempts"] != spec["maximum_generation_attempts_per_arm"] * len(VARIANTS):
        raise ValueError("Reference-dev2 generation allocations do not sum to the global cap")
    if spec["maximum_extraction_attempts"] != spec["maximum_extraction_attempts_per_arm"] * len(VARIANTS):
        raise ValueError("Reference-dev2 extraction allocations do not sum to the global cap")
    sampling = spec.get("sampling")
    if not isinstance(sampling, dict) or sampling.get("max_completion_tokens") != 4096:
        raise ValueError("Reference-dev2 sampling contract differs")
    if (spec.get("budget_transfer_between_arms") is not False
            or spec.get("max_regenerations_per_decision") != 1
            or any(spec.get(key) != 0 for key in (
                "automatic_reruns", "sdk_retries", "proxy_transport_retries",
                "cache_miss_retries"))):
        raise ValueError("Reference-dev2 retry or fixed-allocation contract differs")
    if spec.get("exact_gap_version") != "exact-source-gap-v2":
        raise ValueError("Reference-dev2 must use exact-source-gap-v2")


def verify_current_bundle(bundle: dict, root: Path) -> None:
    shared_exact.verify_current_bundle(bundle, root, extra_files=EXECUTION_SOURCE_FILES)
