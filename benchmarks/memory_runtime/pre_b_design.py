"""Validated legacy official-runner contract for the pre-B P3/P4 stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from . import shared_exact_design as shared_exact
except ImportError:
    import shared_exact_design as shared_exact


HERE = Path(__file__).resolve().parent
DESIGN_NAME = "pre-b-always-compress-v1"
DESIGN_FILE = "benchmarks/memory_runtime/configs/pre_b_v1.json"
DESIGN_SCHEMA = "a-pre-b-official-design-v1"
COMPRESSION_POLICY = "always-compress-v1"
STAGE_NAMES = ("pre-b-p3", "pre-b-p4a", "pre-b-p4b")
P4_STAGE_NAMES = frozenset({"pre-b-p4a", "pre-b-p4b"})
PROVISIONAL_METHOD_STATUS = "provisional_p3_pending"
FROZEN_METHOD_STATUS = "frozen_after_p3"
FROZEN_CANDIDATE_STATUS = "frozen_after_p2"
CANDIDATE_VARIANT = "ac_acquire_for_next"
TASK_IDS = (
    "multi_turn_base_16", "multi_turn_base_105", "multi_turn_base_122",
    "multi_turn_base_157", "multi_turn_base_165", "multi_turn_base_172",
    "multi_turn_base_188", "multi_turn_base_192",
)
G460_TASK_IDS = TASK_IDS[:4]
P3_VARIANTS = (
    "full", "ac_full_shared", "ac_gist_static", "ac_protect",
    "ac_exact_once", "ac_exact_persistent", "raw_recency",
    "raw_exact_shared",
)
CANONICAL_MODE_BY_VARIANT = {
    "full": None,
    "ac_full_shared": "full_exact_shared",
    "ac_gist_static": "legacy",
    "ac_protect": "capacity_protect",
    "ac_exact_once": "capacity_exact_once",
    "ac_exact_persistent": "capacity_exact_persistent",
    "raw_recency": "raw_recency",
    "raw_exact_shared": "capacity_exact_no_gist",
    CANDIDATE_VARIANT: "capacity_exact_persistent",
}
ARM_BY_VARIANT = {
    "full": "full",
    "ac_full_shared": "full",
    "ac_gist_static": "c2kv4",
    "ac_protect": "c2kv4",
    "ac_exact_once": "c2kv4",
    "ac_exact_persistent": "c2kv4",
    "raw_recency": "full",
    "raw_exact_shared": "full",
    CANDIDATE_VARIANT: "c2kv4",
}
CONFIG_SOURCE_BY_VARIANT = {
    "ac_full_shared": "full_exact_shared",
    "ac_gist_static": "legacy",
    "ac_protect": "protect",
    "ac_exact_once": "capacity_exact_once",
    "ac_exact_persistent": "capacity_exact_persistent",
    "raw_recency": "full_shared",
    "raw_exact_shared": "capacity_exact_no_gist",
    CANDIDATE_VARIANT: "capacity_exact_persistent",
}
EXACT_VARIANTS = frozenset({
    "ac_full_shared", "ac_exact_once", "ac_exact_persistent",
    "raw_exact_shared", CANDIDATE_VARIANT,
})
BASE_GIST_VARIANTS = frozenset({
    "ac_gist_static", "ac_protect", "ac_exact_once",
    "ac_exact_persistent",
})
GIST_VARIANTS = BASE_GIST_VARIANTS | {CANDIDATE_VARIANT}
P4_METHOD_VARIANTS = frozenset({"ac_exact_persistent", CANDIDATE_VARIANT})
EXECUTION_SOURCE_FILES = (
    DESIGN_FILE,
    "benchmarks/memory_runtime/pre_b_design.py",
    "benchmarks/memory_runtime/official_pilot.py",
    "benchmarks/memory_runtime/collect_official.py",
    "benchmarks/run.py",
    "benchmarks/proxy.py",
    "benchmarks/checkpoint_profile.py",
    "benchmarks/adapters/bfcl_adapter.py",
    "benchmarks/reqlog.py",
    "benchmarks/memory_runtime/adapter.py",
    "benchmarks/memory_runtime/always_compress.py",
    "benchmarks/memory_runtime/capacity.py",
    "benchmarks/memory_runtime/policy.py",
    "benchmarks/memory_runtime/exact_policy.py",
    "benchmarks/memory_runtime/exact_gap.py",
    "benchmarks/memory_runtime/exact_raw.py",
    "benchmarks/memory_runtime/raw_recency.py",
    "benchmarks/memory_runtime/tokenization.py",
    "benchmarks/memory_runtime/generation_budget.py",
    "benchmarks/memory_runtime/extraction_telemetry.py",
    "benchmarks/memory_runtime/attempt_journal.py",
    "benchmarks/memory_runtime/configs/full_exact_shared.json",
    "benchmarks/memory_runtime/configs/legacy.json",
    "benchmarks/memory_runtime/configs/protect.json",
    "benchmarks/memory_runtime/configs/capacity_exact_once.json",
    "benchmarks/memory_runtime/configs/capacity_exact_persistent.json",
    "benchmarks/memory_runtime/configs/full_shared.json",
    "benchmarks/memory_runtime/configs/capacity_exact_no_gist.json",
    "benchmarks/backends/sglang.py",
    "benchmarks/arms.py",
    "python/history_memory/events.py",
    "python/history_memory/evidence.py",
    "python/history_memory/packing.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def design_source() -> dict[str, str]:
    path = HERE / "configs/pre_b_v1.json"
    return {"path": DESIGN_FILE, "sha256": _sha256(path)}


def execution_source_identity(root: Path) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for name in EXECUTION_SOURCE_FILES:
        path = root / name
        if path.is_file():
            hashes[name] = _sha256(path)
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Pre-B execution sources are missing: {missing}")
    return {
        "schema": "a-pre-b-execution-source-identity-v1",
        "files": list(EXECUTION_SOURCE_FILES),
        "source_files_sha256": hashes,
        "scope": "files that select, execute, meter, and collect the legacy pre-B routes",
    }


def load_design() -> dict[str, Any]:
    spec = json.loads((HERE / "configs/pre_b_v1.json").read_text(encoding="utf-8"))
    validate_design(spec)
    return spec


def load_stage(
    name: str,
    *,
    selected_method: str | None = None,
    selected_method_status: str | None = None,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = load_design()
    if name not in STAGE_NAMES:
        raise ValueError(f"Unknown pre-B stage: {name}")
    stage = dict(spec["stages"][name])
    candidate_active = candidate is not None
    if candidate_active:
        if (candidate.get("status") != FROZEN_CANDIDATE_STATUS
                or candidate.get("variant") != CANDIDATE_VARIANT
                or not isinstance(candidate.get("method_id"), str)
                or not candidate["method_id"]):
            raise ValueError("Conditional candidate lacks a frozen P2 method identity")
        resolved_candidate = dict(candidate)
    else:
        resolved_candidate = dict(spec["conditional_candidate"])
    stage["design"] = name
    stage["task_ids"] = list(spec[stage.pop("task_set")])
    if name in P4_STAGE_NAMES:
        if selected_method is None:
            selected_method = spec["provisional_selected_method"]
            selected_method_status = PROVISIONAL_METHOD_STATUS
        elif selected_method_status not in {
                PROVISIONAL_METHOD_STATUS, FROZEN_METHOD_STATUS}:
            raise ValueError(
                "P4 selected method status must be provisional_p3_pending or "
                "frozen_after_p3"
            )
        if (selected_method not in P4_METHOD_VARIANTS
                or selected_method == CANDIDATE_VARIANT and not candidate_active):
            raise ValueError(
                "P4 selected method must be the incumbent or the frozen P2 "
                f"candidate: {selected_method!r}"
            )
        stage["variants"] = [
            selected_method if variant == "selected_method" else variant
            for variant in stage["variants"]
        ]
        extraction_caps = dict(stage["maximum_extraction_attempts_by_arm"])
        extraction_caps[selected_method] = extraction_caps.pop("selected_method")
        stage["maximum_extraction_attempts_by_arm"] = extraction_caps
        stage["selected_method"] = selected_method
        stage["selected_method_status"] = selected_method_status
    else:
        if selected_method is not None or selected_method_status is not None:
            raise ValueError("P3 does not accept a preselected method")
        stage["selected_method"] = None
        stage["selected_method_status"] = "pending_p3"
        if candidate_active:
            stage["variants"] = [*stage["variants"], CANDIDATE_VARIANT]
            stage["maximum_extraction_attempts_by_arm"] = {
                **stage["maximum_extraction_attempts_by_arm"],
                CANDIDATE_VARIANT: 9216,
            }
    stage["variants"] = [
        [variant, ARM_BY_VARIANT[variant]] for variant in stage["variants"]
    ]
    budget = spec["budgets"][stage.pop("budget")]
    stage.update({
        "history_budget_bytes": budget,
        "workspace_budget_bytes": budget,
        "bytes_per_kv_token": spec["bytes_per_kv_token"],
        "sampling": dict(spec["sampling"]),
        "compression_policy": spec["compression_policy"],
        "scope": spec["scope"],
        "implementation_profile": spec["implementation_profile"],
        "data_contract": dict(spec["data_contract"]),
        "scorer_contract": dict(spec["scorer_contract"]),
        "policy": dict(spec["policy"]),
        "retry_contract": dict(spec["retry_contract"]),
        "profile_contract": dict(
            spec["profile_contracts"][stage["checkpoint_profile"]]),
        "provisional_selected_method": spec["provisional_selected_method"],
        "conditional_candidate": resolved_candidate,
    })
    stage["maximum_tasks"] = len(stage["task_ids"]) * len(stage["variants"])
    stage["maximum_generation_attempts"] = (
        stage["maximum_generation_attempts_per_arm"] * len(stage["variants"])
    )
    stage["maximum_extraction_attempts"] = sum(
        stage["maximum_extraction_attempts_by_arm"].values()
    )
    return stage


def validate_checkpoint_profile(stage: Mapping[str, Any], profile: Mapping[str, Any]) -> None:
    expected = stage["profile_contract"]
    observed = profile.get("serving")
    checkpoint = profile.get("checkpoint")
    if not isinstance(observed, Mapping) or not isinstance(checkpoint, Mapping):
        raise ValueError("Pre-B checkpoint profile lacks resolved serving/checkpoint identity")
    mismatches = {
        field: {"expected": value, "observed": observed.get(field)}
        for field, value in expected.items()
        if field != "checkpoint_name" and observed.get(field) != value
    }
    if checkpoint.get("name") != expected["checkpoint_name"]:
        mismatches["checkpoint.name"] = {
            "expected": expected["checkpoint_name"],
            "observed": checkpoint.get("name"),
        }
    if mismatches:
        raise ValueError(f"Checkpoint profile differs from pre-B stage: {mismatches}")


def validate_generation_budget_rows(
    rows: list[Mapping[str, Any]], arm_limit: int, task_limit: int,
) -> tuple[int, dict[str, int]]:
    """Validate the process counter and independent task counters together."""
    consumed = shared_exact.validate_generation_budget_rows(rows, arm_limit)
    by_task: dict[str, int] = {}
    for index, row in enumerate(rows, 1):
        context = row.get("eval_context")
        budget = row.get("generation_budget")
        task_id = context.get("task_id") if isinstance(context, Mapping) else None
        if not isinstance(task_id, str) or not task_id or not isinstance(budget, Mapping):
            raise ValueError(f"Request {index} lacks task-bound generation metadata")
        attempts = budget.get("attempt_indices")
        before = by_task.get(task_id, 0)
        after = before + len(attempts)
        expected = {
            "per_task_limit": task_limit,
            "task_id": task_id,
            "task_consumed_before": before,
            "task_consumed_after": after,
        }
        mismatches = {
            field: {"expected": value, "observed": budget.get(field)}
            for field, value in expected.items() if budget.get(field) != value
        }
        if mismatches or after > task_limit:
            raise ValueError(
                f"Request {index} per-task generation ledger differs: {mismatches}"
            )
        by_task[task_id] = after
    return consumed, by_task


def validate_zero_extraction_rows(rows: list[Mapping[str, Any]]) -> int:
    """Structural no-extraction routes must record zero producer calls."""
    for index, row in enumerate(rows, 1):
        if row.get("extraction_budget") is not None:
            raise ValueError(f"Request {index} unexpectedly enabled an extraction cap")
        telemetry = row.get("extraction_telemetry")
        summary = telemetry.get("summary") if isinstance(telemetry, Mapping) else None
        events = telemetry.get("events") if isinstance(telemetry, Mapping) else None
        if (not isinstance(summary, Mapping) or not isinstance(events, list)
                or summary.get("producer_calls") != 0
                or any(isinstance(event, Mapping)
                       and event.get("producer_called") is True for event in events)):
            raise ValueError(f"Request {index} used extraction on a zero-extraction route")
    return 0


def validate_design(spec: Mapping[str, Any]) -> None:
    if (not isinstance(spec, Mapping) or spec.get("schema") != DESIGN_SCHEMA
            or spec.get("design") != DESIGN_NAME):
        raise ValueError("Unknown pre-B design")
    shared = shared_exact.load_design()
    source = spec.get("source_design")
    source_path = HERE.parents[1] / str(source.get("path", "")) if isinstance(source, Mapping) else None
    if (not isinstance(source, Mapping) or source.get("design") != shared_exact.DESIGN_NAME
            or source_path is None or not source_path.is_file()
            or source.get("sha256") != _sha256(source_path)
            or list(spec.get("task_ids", [])) != shared["task_ids"]
            or tuple(spec.get("task_ids", ())) != TASK_IDS):
        raise ValueError("Pre-B design does not preserve the validated shared-exact dev8 source")
    if tuple(spec.get("g460_task_ids", ())) != G460_TASK_IDS:
        raise ValueError("G460 tasks must be the first four numeric pre-B task IDs")
    source_selection = shared["task_selection"]
    source_manifest = source_selection["source_manifest"]
    if spec.get("data_contract") != {
        "benchmark": "bfcl",
        "category": "multi_turn_base",
        "selection_manifest_path": source_selection["source_path"],
        "selection_manifest_sha256": source_selection["source_sha256"],
        "official_source_file": source_manifest["source_file"],
        "official_source_sha256": source_manifest["source_sha256"],
        "held_out": False,
        "score_or_response_filtering": False,
    }:
        raise ValueError("Pre-B BFCL data identity differs from the frozen dev8 source")
    if spec.get("scorer_contract") != {
        "implementation": (
            "official BFCL evaluator from the explicitly bound BFCL checkout"),
        "mode": "both",
        "category": "multi_turn_base",
        "runtime_identity_required": True,
    }:
        raise ValueError("Pre-B scorer contract differs")
    routes = spec.get("routes")
    if not isinstance(routes, Mapping) or tuple(routes) != P3_VARIANTS:
        raise ValueError("Pre-B routes differ from the frozen P3 order")
    for variant in P3_VARIANTS:
        route = routes.get(variant)
        if (not isinstance(route, Mapping)
                or route.get("arm") != ARM_BY_VARIANT[variant]
                or route.get("canonical_mode") != CANONICAL_MODE_BY_VARIANT[variant]
                or route.get("config_source") != CONFIG_SOURCE_BY_VARIANT.get(variant)):
            raise ValueError(f"Pre-B route mapping differs for {variant}")
    candidate = spec.get("conditional_candidate")
    if (not isinstance(candidate, Mapping)
            or candidate.get("status") != "pending_revision_choice"
            or candidate.get("implementation_status") != "not_implemented"
            or candidate.get("selection_status") != "not_selected"
            or candidate.get("variant") is not None):
        raise ValueError("Pre-B cannot expose a candidate before a frozen revision choice")
    if (spec.get("provisional_selected_method") != "ac_exact_persistent"
            or spec.get("selected_method_status") != PROVISIONAL_METHOD_STATUS):
        raise ValueError("Pre-B P4 provisional method/status differs")
    if spec.get("compression_policy") != COMPRESSION_POLICY:
        raise ValueError("Pre-B compression policy differs")
    if spec.get("bytes_per_kv_token") != 147456 or spec.get("budgets") != {
            "B0": 113246208, "B1": 226492416}:
        raise ValueError("Pre-B byte geometry or B0/B1 differs")
    if spec.get("profile_contracts") != {
        "checkpoint-1088": {
            "checkpoint_name": "checkpoint-1088", "query_projection": "base",
            "doc_packing": "turn", "max_doc_length": 512, "max_doc_num": 12,
        },
        "checkpoint-460": {
            "checkpoint_name": "checkpoint-460", "query_projection": "gist",
            "doc_packing": "turn", "max_doc_length": 768, "max_doc_num": 16,
        },
    }:
        raise ValueError("Pre-B checkpoint profile contracts differ")
    policy = spec.get("policy")
    policy_path = HERE.parents[1] / str(policy.get("source", "")) if isinstance(policy, Mapping) else None
    if (not isinstance(policy, Mapping) or policy_path is None or not policy_path.is_file()
            or policy.get("source_sha256") != _sha256(policy_path)
            or any(policy.get(key) != value for key, value in {
                "lease_decisions": 3, "max_retrieved_events": 1,
                "max_regenerations_per_decision": 1,
            }.items())):
        raise ValueError("Pre-B policy source or L3/retrieval contract differs")
    if spec.get("sampling") != {
            "temperature": 0.001, "seed": 0, "max_completion_tokens": 4096}:
        raise ValueError("Pre-B sampling contract differs")
    retry = spec.get("retry_contract")
    if (not isinstance(retry, Mapping)
            or retry.get("budget_transfer_between_arms") is not False
            or any(retry.get(key) != 0 for key in (
                "automatic_reruns", "sdk_retries", "proxy_transport_retries",
                "cache_miss_retries"))):
        raise ValueError("Pre-B retry contract differs")
    walls = spec.get("wall_allocation_seconds")
    if (not isinstance(walls, Mapping) or dict(walls) != {
            "P0": 3600, "P1": 7200, "P2": 10800, "P3": 93600,
            "P4a": 28800, "P4b": 28800}
            or sum(walls.values()) != spec.get("maximum_model_wall_seconds")
            or spec.get("maximum_model_wall_seconds") != 172800):
        raise ValueError("Pre-B 48-hour stage allocation differs")
    stages = spec.get("stages")
    if not isinstance(stages, Mapping) or tuple(stages) != STAGE_NAMES:
        raise ValueError("Pre-B official stages differ")
    expected_variants = {
        "pre-b-p3": P3_VARIANTS,
        "pre-b-p4a": ("selected_method", "raw_recency", "raw_exact_shared"),
        "pre-b-p4b": ("full", "selected_method", "raw_recency", "raw_exact_shared"),
    }
    expected_stage_fields = {
        "pre-b-p3": {
            "stage": "P3", "checkpoint_profile": "checkpoint-1088",
            "budget": "B0", "task_set": "task_ids",
            "maximum_wall_seconds": 93600,
        },
        "pre-b-p4a": {
            "stage": "P4a", "checkpoint_profile": "checkpoint-1088",
            "budget": "B1", "task_set": "task_ids",
            "maximum_wall_seconds": 28800,
        },
        "pre-b-p4b": {
            "stage": "P4b", "checkpoint_profile": "checkpoint-460",
            "budget": "B0", "task_set": "g460_task_ids",
            "maximum_wall_seconds": 28800,
        },
    }
    expected_extraction_caps = {
        "pre-b-p3": 9216,
        "pre-b-p4a": 9216,
        "pre-b-p4b": 6144,
    }
    for name in STAGE_NAMES:
        stage = stages[name]
        if any(stage.get(field) != value
               for field, value in expected_stage_fields[name].items()):
            raise ValueError(f"Pre-B stage identity differs for {name}")
        task_ids = spec[stage["task_set"]]
        variants = tuple(stage.get("variants", ()))
        if variants != expected_variants[name]:
            raise ValueError(f"Pre-B variants differ for {name}")
        per_task = stage.get("maximum_generation_attempts_per_task")
        per_arm = stage.get("maximum_generation_attempts_per_arm")
        if per_task != 96 or per_arm != len(task_ids) * per_task:
            raise ValueError(f"Pre-B per-task/per-arm generation caps differ for {name}")
        caps = stage.get("maximum_extraction_attempts_by_arm")
        if not isinstance(caps, Mapping) or set(caps) != set(variants):
            raise ValueError(f"Pre-B extraction cap coverage differs for {name}")
        if any(not isinstance(value, int) or value < 0 for value in caps.values()):
            raise ValueError(f"Pre-B extraction cap is invalid for {name}")
        if any((caps[variant] > 0) != (variant in GIST_VARIANTS)
               for variant in variants if variant != "selected_method"):
            raise ValueError(f"Pre-B extraction caps do not match gist routes for {name}")
        for variant, cap in caps.items():
            if variant == "selected_method":
                if cap != expected_extraction_caps[name]:
                    raise ValueError(f"Pre-B selected method extraction cap differs for {name}")
            elif variant in GIST_VARIANTS and cap != expected_extraction_caps[name]:
                raise ValueError(f"Pre-B gist extraction cap differs for {name}")
            elif variant not in GIST_VARIANTS and cap != 0:
                raise ValueError(f"Pre-B no-extraction cap differs for {name}")
