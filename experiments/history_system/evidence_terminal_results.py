"""Build and verify fixed-denominator terminal D128 result receipts.

The existing ``experiment3-complete-d128-result-v1`` schema keeps its strict
meaning: all 128 cells have successful runtimes and accepted official scores.
This module adds a separate terminal receipt for the Experiment 3 C0/C5 H0/H1
selection protocol.  A hash-bound, audited pre-generation CapacityInfeasible
cell is an operational non-success in the fixed denominator; its raw official
artifact is retained but is never promoted to an accepted quality score.
Unknown failures and pending cells remain selection-blocking.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence


BUILD_SCHEMA = "experiment3-terminal-d128-build-v1"
SOURCE_RESULT_SCHEMA = "experiment3-complete-d128-result-v1"
TERMINAL_RESULT_SCHEMA = "experiment3-terminal-d128-result-v1"
CAPACITY_AUDIT_SCHEMA = "experiment3-d128-h0-runtime-failure-audit-v1"
GENERIC_CAPACITY_AUDIT_SCHEMA = "experiment3-d128-runtime-failure-audit-v1"
CONTINUATION_OVERLAY_SCHEMA = "experiment3-expansion-remainder-overlay-v1"
SELECTION_SCOPE = "fixed_d128_operational_success_v1"
SUPPORTED_STAGES = {"H0_R1", "H1_R1"}
SUPPORTED_CONTROLLERS = {"C0", "C5"}
HEX64 = re.compile(r"[0-9a-f]{64}")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_sha(value: object) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def _bound_document(
    binding: Mapping[str, Any], expected_schema: str, label: str
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    path_value = binding.get("path")
    expected_sha = binding.get("sha256")
    if not isinstance(path_value, str) or not path_value or not _valid_sha(expected_sha):
        raise ValueError(f"Invalid {label} binding")
    path = Path(path_value).resolve()
    if not path.is_file() or _sha(path) != expected_sha:
        raise ValueError(f"{label} differs from its exact hash binding")
    value = _read(path)
    if value.get("schema") != expected_schema:
        raise ValueError(f"Unsupported {label} schema")
    return path, value, {
        "path": str(path), "sha256": expected_sha, "schema": expected_schema,
    }


def _bound_failure_audit(
    binding: Mapping[str, Any], label: str = "capacity_failure_audit"
) -> tuple[Path, dict[str, Any], dict[str, str]]:
    path_value = binding.get("path")
    expected_sha = binding.get("sha256")
    if not isinstance(path_value, str) or not path_value or not _valid_sha(expected_sha):
        raise ValueError(f"Invalid {label} binding")
    path = Path(path_value).resolve()
    if not path.is_file() or _sha(path) != expected_sha:
        raise ValueError(f"{label} differs from its exact hash binding")
    value = _read(path)
    schema = value.get("schema")
    if schema not in {CAPACITY_AUDIT_SCHEMA, GENERIC_CAPACITY_AUDIT_SCHEMA}:
        raise ValueError(f"Unsupported {label} schema")
    return path, value, {
        "path": str(path), "sha256": expected_sha, "schema": schema,
    }


def _official_valid(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("scored") is True
        and value.get("n_total") == 1
        and value.get("n_scored") == 1
        and type(value.get("correct_count")) is int
        and value.get("correct_count") in {0, 1}
        and not isinstance(value.get("semantic_score"), bool)
        and isinstance(value.get("semantic_score"), (int, float))
        and math.isfinite(float(value["semantic_score"]))
    )


def _evidence_list(value: Any, label: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        rows = []
        for role, binding in value.items():
            if not isinstance(binding, dict):
                raise ValueError(f"{label} contains malformed evidence")
            rows.append({"role": role, **binding})
    elif isinstance(value, list):
        rows = [dict(row) if isinstance(row, dict) else row for row in value]
    else:
        raise ValueError(f"{label} must contain evidence")
    if (
        not rows
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not row["path"]
            or not _valid_sha(row.get("sha256"))
            for row in rows
        )
    ):
        raise ValueError(f"{label} contains malformed evidence")
    return rows


def _source_receipt(value: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    cells = value.get("quality_cells")
    task_ids = [cell.get("task_id") for cell in cells] if isinstance(cells, list) else []
    if (
        value.get("schema") != SOURCE_RESULT_SCHEMA
        or value.get("stage") not in SUPPORTED_STAGES
        or value.get("controller") not in SUPPORTED_CONTROLLERS
        or not isinstance(value.get("source_algorithm_id"), str)
        or not value["source_algorithm_id"]
        or not _valid_sha(value.get("task_manifest_sha256"))
        or not isinstance(value.get("configuration_binding"), dict)
        or not isinstance(value.get("evidence"), list)
        or not isinstance(cells, list)
        or len(cells) != 128
        or len(task_ids) != len(set(task_ids))
        or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
    ):
        raise ValueError("Source D128 result receipt is malformed")
    _evidence_list(value.get("evidence"), "source D128 result receipt")
    for cell in cells:
        status = cell.get("status")
        if (
            status not in {"completed", "runtime_failed", "pending"}
            or not isinstance(cell.get("quality_source_id"), str)
            or not cell["quality_source_id"]
            or cell.get("cohort") not in {"historical20", "new108", "full128"}
        ):
            raise ValueError(f"Malformed source quality cell: {cell.get('task_id')}")
        _evidence_list(cell.get("evidence"), f"source cell {cell['task_id']}")
        if status == "completed" and (
            cell.get("runtime_completed") is not True
            or cell.get("worker_returncode") != 0
            or cell.get("server_returncode") != 0
            or not _official_valid(cell.get("official"))
        ):
            raise ValueError(f"Completed source cell is invalid: {cell['task_id']}")
    return [dict(cell) for cell in cells], task_ids


def _matching_expansion_contract_sha(source: Mapping[str, Any]) -> str | None:
    for row in source.get("evidence", []):
        if isinstance(row, dict) and row.get("role") == "expansion_contract":
            return row.get("sha256") if _valid_sha(row.get("sha256")) else None
    return None


def _matching_package_contract(source: Mapping[str, Any]) -> dict[str, Any] | None:
    role = "expansion_contract" if source.get("stage") == "H0_R1" else "combination_contract"
    rows = [row for row in source.get("evidence", [])
            if isinstance(row, dict) and row.get("role") == role]
    if len(rows) != 1 or not _valid_sha(rows[0].get("sha256")):
        return None
    return rows[0]


def _has_source_evidence_sha(source: Mapping[str, Any], role: str, sha256: Any) -> bool:
    return bool(_valid_sha(sha256) and any(
        isinstance(row, dict) and row.get("role") == role
        and row.get("sha256") == sha256
        for row in source.get("evidence", [])))


def _document_parent(path_value: str) -> str:
    return (str(PurePosixPath(path_value).parent)
            if path_value.startswith("/") else str(Path(path_value).parent))


def _audit_failures(
    audit: Mapping[str, Any], source: Mapping[str, Any], audit_sha256: str
) -> dict[tuple[str, str], dict[str, Any]]:
    scope = audit.get("scope")
    package = audit.get("package")
    failures = audit.get("failures")
    schema = audit.get("schema")
    if (
        schema not in {CAPACITY_AUDIT_SCHEMA, GENERIC_CAPACITY_AUDIT_SCHEMA}
        or audit.get("status") != "audited"
        or not isinstance(scope, dict)
        or scope.get("automatic_retry_or_rerun") is not False
        or scope.get("fixed_d128_runtime_outcomes") is not True
        or scope.get("official_scores_preserved") is not True
        or scope.get("official_zero_used_as_clean_runtime_completion") is not False
        or scope.get("selection_eligible") is not True
        or scope.get("unknown_failure_count") != 0
        or (scope.get("audited_capacity_failure_count") is not None
            and scope.get("audited_capacity_failure_count") != len(failures))
        or not isinstance(package, dict)
        or not isinstance(failures, list)
    ):
        raise ValueError("Capacity-failure audit does not preserve the fixed-D128 contract")
    if schema == CAPACITY_AUDIT_SCHEMA:
        source_contract_sha = _matching_expansion_contract_sha(source)
        audit_contract = package.get("expansion_contract")
        if (
            source.get("stage") != "H0_R1"
            or not isinstance(audit_contract, dict)
            or audit_contract.get("sha256") != source_contract_sha
        ):
            raise ValueError("H0 capacity-failure audit is not bound to the source H0 package")
    else:
        audit_contract = package.get("contract")
        audit_static = package.get("static_files")
        if (
            package.get("stage") != source.get("stage")
            or not isinstance(audit_contract, dict)
            or not _valid_sha(audit_contract.get("sha256"))
            or not isinstance(audit_static, dict)
            or not _valid_sha(audit_static.get("sha256"))
            or not isinstance(package.get("path"), str)
            or not package["path"]
        ):
            raise ValueError("Generic capacity-failure audit is not bound to the source package/stage")

    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(failures):
        if not isinstance(row, dict) or row.get("controller") != source.get("controller"):
            continue
        if schema == GENERIC_CAPACITY_AUDIT_SCHEMA and (
            row.get("stage") != source.get("stage")
            or row.get("source_algorithm_id") != source.get("source_algorithm_id")
        ):
            continue
        shard = row.get("shard")
        task_id = row.get("task_id")
        stage = row.get("stage_outcome")
        classification = row.get("classification")
        failure = row.get("failure")
        runtime = row.get("runtime_vs_official")
        measurement = row.get("declared_budget_and_failure_measurement")
        evidence = row.get("evidence")
        markers = measurement.get("failure_reason_markers") if isinstance(measurement, dict) else None
        budget = classification.get("budget_contract") if isinstance(classification, dict) else None
        checks = classification.get("checks") if isinstance(classification, dict) else None
        limit = budget.get("eval_capacity_max_sequence_tokens") if isinstance(budget, dict) else None
        design_limit = budget.get("design_resolved_max_sequence_tokens") if isinstance(budget, dict) else None
        startup_limit = budget.get("startup_effective_max_sequence_tokens") if isinstance(budget, dict) else None
        backend_context = budget.get("backend_context_length") if isinstance(budget, dict) else None
        model_context = budget.get("model_context") if isinstance(budget, dict) else None
        candidate_min = budget.get("candidate_sequence_tokens_min") if isinstance(budget, dict) else None
        candidate_max = budget.get("candidate_sequence_tokens_max") if isinstance(budget, dict) else None
        source_binding = row.get("source_binding")
        generic_binding_valid = True
        if schema == GENERIC_CAPACITY_AUDIT_SCHEMA:
            generic_binding_valid = bool(
                isinstance(source_binding, dict)
                and source_binding.get("stage") == source.get("stage")
                and source_binding.get("controller") == source.get("controller")
                and source_binding.get("task_id") == task_id
                and source_binding.get("source_algorithm_id")
                == source.get("source_algorithm_id")
                and source_binding.get("package_root") == package.get("path")
                and source_binding.get("source_package_static_sha256")
                == package.get("static_files", {}).get("sha256")
                and source_binding.get("source_package_contract_sha256")
                == package.get("contract", {}).get("sha256")
                and source_binding.get("shard") == shard
                and isinstance(source_binding.get("lane"), str)
                and bool(source_binding["lane"])
                and source_binding.get("source_provenance_kind") in {
                    "shard_provenance", "remainder_source_binding"}
                and _valid_sha(source_binding.get("source_provenance_sha256"))
                and _valid_sha(source_binding.get("evaluated_design_sha256"))
                and _valid_sha(source_binding.get("server_startup_sha256"))
                and _valid_sha(source_binding.get("frozen_eval_capacity_sha256"))
            )
        if (
            not isinstance(shard, str)
            or not isinstance(task_id, str)
            or not generic_binding_valid
            or row.get("quality_status") != "runtime_failed"
            or not isinstance(stage, dict)
            or stage.get("task_id") != task_id
            or stage.get("outcome") != "runtime_failure_in_denominator"
            or stage.get("in_fixed_denominator") is not True
            or stage.get("runtime_completed") is not False
            or stage.get("worker_returncode") != 0
            or stage.get("server_returncode") != 1
            or not isinstance(classification, dict)
            or classification.get("status") != "audited_in_contract"
            or classification.get("selection_eligible") is not True
            or classification.get("budget_contract_verified") is not True
            or classification.get("family") != "pre_generation_capacity_infeasible"
            or classification.get("exception_type") != "CapacityInfeasible"
            or classification.get("source_exception") != "CapacityInfeasible"
            or classification.get("is_sglang_extraction_budget_exhausted") is not False
            or classification.get("is_sglang_http_failure") is not False
            or classification.get("is_endpoint_or_startup_failure") is not False
            or not isinstance(checks, dict)
            or not checks
            or any(value is not True for value in checks.values())
            or type(limit) is not int
            or limit <= 0
            or design_limit != limit
            or startup_limit != limit
            or type(backend_context) is not int
            or type(model_context) is not int
            or type(candidate_min) is not int
            or type(candidate_max) is not int
            or not (limit < candidate_min <= candidate_max <= backend_context)
            or candidate_max > model_context
            or budget.get("binding_limit") != "eval_capacity_max_sequence_tokens"
            or not isinstance(failure, dict)
            or failure.get("failed_step_generation_attempts") != 0
            or failure.get("failed_step_generation_completed") != 0
            or failure.get("failed_step_generation_trace_count") != 0
            or not _valid_sha(failure.get("error_object_sha256"))
            or not isinstance(runtime, dict)
            or runtime.get("runtime_completed") is not False
            or runtime.get("server_returncode") != 1
            or runtime.get("official_summary_exists") is not True
            or runtime.get("official_score_does_not_convert_runtime_failure_to_completed") is not True
            or runtime.get("official_zero_imputed") is not False
            or not _official_valid(runtime.get("official"))
            or runtime["official"].get("correct_count") != 0
            or not isinstance(markers, list)
            or "physical_sequence_budget:4" not in markers
            or "physical_sequence_budget:8" not in markers
        ):
            raise ValueError(f"Unusable audited CapacityInfeasible row: {shard}/{task_id}")
        evidence_rows = _evidence_list(evidence, f"audit row {shard}/{task_id}")
        evidence_roles = {item.get("role") for item in evidence_rows}
        required_roles = {
            "stage_manifest", "steps", "attempts", "server_final",
            "official_summary", "frozen_capacity_exception_source",
            "frozen_capacity_policy_source", "evaluated_design", "source_design",
            "frozen_eval_capacity", "server_startup", "engine_log", "engine_status",
        }
        if schema == GENERIC_CAPACITY_AUDIT_SCHEMA:
            required_roles.update({
                "package_contract", "package_static_files",
                "source_provenance", "evaluated_design",
            })
        if not required_roles <= evidence_roles:
            raise ValueError(f"Audited failure lacks raw proof: {shard}/{task_id}")
        key = (shard, task_id)
        if key in selected:
            raise ValueError(f"Duplicate audited failure: {shard}/{task_id}")
        selected[key] = {
            "row": row,
            "failure_index": index,
            "audit_sha256": audit_sha256,
            "audit_schema": schema,
            "audit_package": package,
            "evidence": evidence_rows,
        }
    return selected


def _audit_cell(
    source_cell: Mapping[str, Any], audited: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    row = audited["row"]
    stage = row["stage_outcome"]
    runtime = row["runtime_vs_official"]
    owner = source_cell.get("continuation_lane", source_cell.get("shard_id"))
    if (
        source_cell.get("status") != "unknown_failure"
        or owner != row.get("shard")
        or source_cell.get("runtime_completed") is not False
        or source_cell.get("worker_returncode") != stage.get("worker_returncode")
        or source_cell.get("server_returncode") != stage.get("server_returncode")
    ):
        raise ValueError(
            f"Audit does not match source runtime failure: {source_cell.get('task_id')}"
        )
    source_official = (
        source_cell.get("official")
        if isinstance(source_cell.get("official"), dict)
        else source_cell.get("raw_official"))
    raw_official = runtime["official"]
    audited_stage_sha = next(
        (item.get("sha256") for item in audited["evidence"]
         if item.get("role") == "stage_manifest"), None)
    cell_evidence_shas = {
        item.get("sha256") for item in source_cell.get("evidence", [])
        if isinstance(item, dict) and _valid_sha(item.get("sha256"))}
    if audited_stage_sha not in cell_evidence_shas:
        raise ValueError(
            f"Audit stage proof differs from source receipt: {source_cell.get('task_id')}"
        )
    if audited.get("audit_schema") == GENERIC_CAPACITY_AUDIT_SCHEMA:
        binding = row["source_binding"]
        package = audited["audit_package"]
        audit_evidence = {
            item.get("role"): item.get("sha256") for item in audited["evidence"]}
        continuation_package = source_cell.get("continuation_package")
        continuation_contract = source_cell.get("continuation_contract_sha256")
        if continuation_package is not None:
            package_matches = (
                binding.get("package_root") == continuation_package
                and binding.get("source_package_contract_sha256")
                == continuation_contract)
        else:
            source_contract = _matching_package_contract(source)
            source_root = (
                _document_parent(source_contract["path"])
                if isinstance(source_contract, dict) else None)
            package_matches = (
                source_contract is not None
                and binding.get("package_root") == source_root
                and binding.get("source_package_contract_sha256")
                == source_contract.get("sha256")
                and _has_source_evidence_sha(
                    source, binding.get("source_provenance_kind"),
                    binding.get("source_provenance_sha256"))
                and _has_source_evidence_sha(
                    source, "evaluated_design",
                    binding.get("evaluated_design_sha256")))
        if (
            not package_matches
            or package.get("path") != binding.get("package_root")
            or binding.get("lane") != owner
            or audit_evidence.get("source_provenance")
            != binding.get("source_provenance_sha256")
            or audit_evidence.get("evaluated_design")
            != binding.get("evaluated_design_sha256")
            or audit_evidence.get("server_startup")
            != binding.get("server_startup_sha256")
            or audit_evidence.get("frozen_eval_capacity")
            != binding.get("frozen_eval_capacity_sha256")
            or audit_evidence.get("package_contract")
            != binding.get("source_package_contract_sha256")
            or audit_evidence.get("package_static_files")
            != binding.get("source_package_static_sha256")
        ):
            raise ValueError(
                f"Generic audit source binding differs: {source_cell.get('task_id')}"
            )
    for key in ("scored", "n_total", "n_scored", "correct_count", "semantic_score"):
        if not isinstance(source_official, dict) or source_official.get(key) != raw_official.get(key):
            raise ValueError(
                f"Audit raw official differs from source artifact: {source_cell.get('task_id')}"
            )
    return {
        **dict(source_cell),
        "status": "audited_capacity_failure",
        "status_reason": "audited_pre_generation_capacity_infeasible",
        "official": None,
        "raw_official": dict(raw_official),
        "operational_success": False,
        "audit_binding": {
            "sha256": audited["audit_sha256"],
            "schema": audited["audit_schema"],
            "failure_index": audited["failure_index"],
            "error_object_sha256": row["failure"]["error_object_sha256"],
        },
        "evidence": [*source_cell.get("evidence", []), *audited["evidence"]],
    }


def _source_stage_sha(cell: Mapping[str, Any]) -> str | None:
    rows = cell.get("evidence")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("role") == "stage_manifest":
            return row.get("sha256") if _valid_sha(row.get("sha256")) else None
    return None


def _source_stage_binding(cell: Mapping[str, Any]) -> tuple[Path, str]:
    rows = cell.get("evidence")
    if not isinstance(rows, list):
        raise ValueError(f"Source cell lacks stage evidence: {cell.get('task_id')}")
    matches = [row for row in rows if isinstance(row, dict)
               and row.get("role") == "stage_manifest"]
    if len(matches) != 1 or not _valid_sha(matches[0].get("sha256")):
        raise ValueError(f"Source cell lacks exact stage evidence: {cell.get('task_id')}")
    return Path(matches[0]["path"]), matches[0]["sha256"]


def _overlay_cell(source_cell: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    if source_cell.get("status") != "pending":
        raise ValueError(
            f"Continuation overlay may replace only original pending cells: "
            f"{source_cell.get('task_id')}"
        )
    source_outcome = row.get("source_outcome")
    source_shard = row.get("source_shard")
    stage_path, stage_sha = _source_stage_binding(source_cell)
    status_path = stage_path.parent.parent / "run/status.json"
    if (not stage_path.is_file() or _sha(stage_path) != stage_sha
            or not status_path.is_file()
            or _sha(status_path) != row.get("source_status_sha256")):
        raise ValueError(
            f"Continuation overlay cannot verify original stage/status hashes: "
            f"{source_cell.get('task_id')}"
        )
    stage_document = _read(stage_path)
    source_rows = stage_document.get("task_outcomes")
    if not isinstance(source_rows, list):
        raise ValueError(
            f"Continuation overlay cannot resolve original outcome: "
            f"{source_cell.get('task_id')}"
        )
    exact_rows = [item for item in source_rows if isinstance(item, dict)
                  and item.get("task_id") == source_cell.get("task_id")]
    if len(exact_rows) != 1:
        raise ValueError(
            f"Continuation overlay cannot resolve original outcome: "
            f"{source_cell.get('task_id')}"
        )
    if (
        not isinstance(source_shard, str)
        or Path(source_shard).name != source_cell.get("shard_id")
        or row.get("source_lane") != source_cell.get("shard_id")
        or row.get("task_id") != source_cell.get("task_id")
        or row.get("source_stage_manifest_sha256") != stage_sha
        or not _valid_sha(row.get("source_status_sha256"))
        or not isinstance(source_outcome, dict)
        or source_outcome != exact_rows[0]
        or source_outcome.get("task_id") != source_cell.get("task_id")
        or source_outcome.get("outcome") != "not_started"
        or source_outcome.get("official_summary", object()) is not None
        or source_outcome.get("in_fixed_denominator") is not True
        or not isinstance(row.get("continuation_lane"), str)
        or not row["continuation_lane"]
        or row.get("status") not in {"completed", "runtime_failed", "pending"}
    ):
        raise ValueError(f"Continuation overlay source binding differs: {source_cell.get('task_id')}")
    evidence = _evidence_list(row.get("evidence"), f"overlay row {row.get('task_id')}")
    stage = row["stage_outcome"]
    status = row["status"]
    if status == "pending":
        if stage is not None and (
            not isinstance(stage, dict) or stage.get("outcome") != "not_started"
        ):
            raise ValueError(f"Pending continuation cell is invalid: {source_cell.get('task_id')}")
        return {
            **dict(source_cell),
            "status_reason": "continuation_pending",
            "continuation_lane": row["continuation_lane"],
            "continuation_source_binding": {
                "source_stage_manifest_sha256": row["source_stage_manifest_sha256"],
                "source_status_sha256": row["source_status_sha256"],
            },
            "evidence": [*source_cell.get("evidence", []), *evidence],
        }
    if status == "completed":
        official = row.get("official")
        if (
            not isinstance(stage, dict)
            or stage.get("task_id") != source_cell.get("task_id")
            or stage.get("outcome") != "official_completed"
            or stage.get("runtime_completed") is not True
            or stage.get("worker_returncode") != 0
            or stage.get("server_returncode") != 0
            or not _official_valid(official)
            or not isinstance(row.get("measured_cost", {}), dict)
        ):
            raise ValueError(f"Completed continuation cell is invalid: {source_cell.get('task_id')}")
        return {
            **dict(source_cell),
            "status": "completed",
            "status_reason": "official_completed_by_never_started_continuation",
            "runtime_completed": True,
            "worker_returncode": 0,
            "server_returncode": 0,
            "official": dict(official),
            "raw_official": None,
            "operational_success": official["correct_count"] == 1,
            "measured_cost": dict(row.get("measured_cost", {})),
            "continuation_lane": row["continuation_lane"],
            "continuation_source_binding": {
                "source_stage_manifest_sha256": row["source_stage_manifest_sha256"],
                "source_status_sha256": row["source_status_sha256"],
            },
            "evidence": [*source_cell.get("evidence", []), *evidence],
        }
    return {
        **dict(source_cell),
        "status": "unknown_failure",
        "status_reason": "unaudited_continuation_runtime_failure",
        "runtime_completed": stage.get("runtime_completed") is True
        if isinstance(stage, dict) else False,
        "worker_returncode": stage.get("worker_returncode")
        if isinstance(stage, dict) else None,
        "server_returncode": stage.get("server_returncode")
        if isinstance(stage, dict) else None,
        "official": None,
        "raw_official": dict(row["official"])
        if isinstance(row.get("official"), dict) else None,
        "operational_success": None,
        "continuation_lane": row["continuation_lane"],
        "continuation_source_binding": {
            "source_stage_manifest_sha256": row["source_stage_manifest_sha256"],
            "source_status_sha256": row["source_status_sha256"],
        },
        "evidence": [*source_cell.get("evidence", []), *evidence],
    }


def _apply_overlays(
    cells: list[dict[str, Any]], overlays: Sequence[tuple[Mapping[str, Any], str]]
) -> list[dict[str, Any]]:
    by_key = {(cell.get("shard_id"), cell["task_id"]): index
              for index, cell in enumerate(cells)}
    source_shards = {key[0] for key in by_key if key[0] is not None}
    applied: set[tuple[str, str]] = set()
    for overlay, overlay_sha in overlays:
        rows = overlay.get("rows")
        pending_count = (
            sum(isinstance(row, dict) and row.get("status") == "pending" for row in rows)
            if isinstance(rows, list) else None)
        if (overlay.get("schema") != CONTINUATION_OVERLAY_SCHEMA
                or not isinstance(rows, list)
                or overlay.get("total_original_not_started_cells") != len(rows)
                or overlay.get("pending_cells") != pending_count
                or overlay.get("status")
                != ("completed" if pending_count == 0 else "partial_pending")
                or overlay.get("automatic_retries") != 0
                or overlay.get("automatic_reruns") != 0
                or not _valid_sha(overlay.get("continuation_contract_sha256"))):
            raise ValueError("Continuation overlay is malformed")
        for row in rows:
            if not isinstance(row, dict) or row.get("source_lane") not in source_shards:
                continue
            key = (row.get("source_lane"), row.get("task_id"))
            if key not in by_key:
                raise ValueError(f"Continuation overlay names unknown source cell: {key}")
            if key in applied:
                raise ValueError(f"Continuation overlay repeats source cell: {key}")
            index = by_key[key]
            cells[index] = _overlay_cell(cells[index], row)
            cells[index]["continuation_overlay_sha256"] = overlay_sha
            cells[index]["continuation_package"] = overlay.get("package")
            cells[index]["continuation_contract_sha256"] = overlay[
                "continuation_contract_sha256"]
            applied.add(key)
    return cells


def _terminal_document(
    source: Mapping[str, Any], source_binding: Mapping[str, str],
    audits: Sequence[tuple[Mapping[str, Any], Mapping[str, str]]],
    overlays: Sequence[tuple[Mapping[str, Any], Mapping[str, str]]], *,
    generated_at: str | None = None,
) -> dict[str, Any]:
    source_cells, task_ids = _source_receipt(source)
    audited: dict[tuple[str, str], dict[str, Any]] = {}
    for audit, binding in audits:
        rows = _audit_failures(audit, source, binding["sha256"])
        duplicates = set(audited) & set(rows)
        if duplicates:
            raise ValueError(f"Capacity audits repeat source failures: {sorted(duplicates)}")
        audited.update(rows)

    cells: list[dict[str, Any]] = []
    for cell in source_cells:
        if cell.get("status") == "completed":
            cells.append({
                **cell,
                "raw_official": None,
                "operational_success": cell["official"]["correct_count"] == 1,
            })
        elif cell.get("status") == "runtime_failed":
            cells.append({
                **cell,
                "status": "unknown_failure",
                "source_status": "runtime_failed",
                "official": None,
                "raw_official": dict(cell["official"])
                if isinstance(cell.get("official"), dict) else None,
                "operational_success": None,
            })
        else:
            cells.append({**cell, "raw_official": None, "operational_success": None})
    cells = _apply_overlays(
        cells, [(value, binding["sha256"]) for value, binding in overlays])
    cell_by_audit_key = {
        (cell.get("continuation_lane", cell.get("shard_id")), cell["task_id"]): index
        for index, cell in enumerate(cells)
        if cell.get("status") == "unknown_failure"
    }
    for key, audited_row in audited.items():
        if key not in cell_by_audit_key:
            raise ValueError(f"Audit may classify only matching unknown failures: {key}")
        index = cell_by_audit_key[key]
        cells[index] = _audit_cell(cells[index], audited_row, source)

    normal = [cell for cell in cells if cell.get("status") == "completed"]
    capacity = [cell for cell in cells
                if cell.get("status") == "audited_capacity_failure"]
    unknown = [cell for cell in cells if cell.get("status") == "unknown_failure"]
    pending = [cell for cell in cells if cell.get("status") == "pending"]
    if len(normal) + len(capacity) + len(unknown) + len(pending) != 128:
        raise ValueError("Terminal cells do not form the fixed D128 partition")
    terminal_count = len(normal) + len(capacity)
    eligible = terminal_count == 128 and not unknown and not pending
    operational_success_count = sum(
        1 for cell in normal if cell.get("operational_success") is True)
    return {
        "schema": TERMINAL_RESULT_SCHEMA,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "status": "terminal" if eligible else "incomplete",
        "selection_eligible": eligible,
        "selection_scope": SELECTION_SCOPE,
        "quality_label": "preliminary, n=1",
        "package_kind": source.get("package_kind"),
        "stage": source["stage"],
        "controller": source["controller"],
        "source_algorithm_id": source["source_algorithm_id"],
        "task_manifest_sha256": source["task_manifest_sha256"],
        "configuration_binding": dict(source["configuration_binding"]),
        "source_result_receipt": dict(source_binding),
        "capacity_failure_audits": [dict(binding) for _, binding in audits],
        "continuation_overlays": [dict(binding) for _, binding in overlays],
        "expected_task_cells": 128,
        "terminal_task_cells": terminal_count,
        "normal_completed_task_cells": len(normal),
        "audited_capacity_failure_task_cells": len(capacity),
        "unknown_failure_task_cells": len(unknown),
        "pending_task_cells": len(pending),
        "operational_denominator": 128,
        "operational_success_count": operational_success_count,
        "operational_non_success_count": (
            128 - operational_success_count if eligible else None),
        "accepted_official_score_task_cells": len(normal),
        "operational_official_unobserved_task_cells": (
            len(capacity) + len(unknown) + len(pending)),
        "official_zero_imputed": False,
        "automatic_retries": 0,
        "automatic_reruns": 0,
        "task_ids": task_ids,
        "evidence": list(source["evidence"]),
        "quality_cells": cells,
    }


def build_terminal_receipt(spec_path: Path) -> dict[str, Any]:
    spec = _read(spec_path.resolve())
    if spec.get("schema") != BUILD_SCHEMA:
        raise ValueError("Unknown terminal D128 build spec")
    source_raw = spec.get("source_result_receipt")
    if not isinstance(source_raw, dict):
        raise ValueError("Build spec requires source_result_receipt")
    _, source, source_binding = _bound_document(
        source_raw, SOURCE_RESULT_SCHEMA, "source_result_receipt")
    audit_raw = spec.get("capacity_failure_audit")
    audit_list_raw = spec.get("capacity_failure_audits")
    if audit_raw is not None and audit_list_raw is not None:
        raise ValueError("Use only one capacity-failure audit binding form")
    if audit_list_raw is None:
        audit_values = [] if audit_raw is None else [audit_raw]
    elif isinstance(audit_list_raw, list):
        audit_values = audit_list_raw
    else:
        raise ValueError("capacity_failure_audits must be a list")
    audits = []
    for raw in audit_values:
        if not isinstance(raw, dict):
            raise ValueError("Capacity-failure audit binding is malformed")
        _, audit, binding = _bound_failure_audit(raw)
        audits.append((audit, binding))
    overlay_raw = spec.get("continuation_overlays", [])
    if not isinstance(overlay_raw, list):
        raise ValueError("continuation_overlays must be a list")
    overlays = []
    for raw in overlay_raw:
        if not isinstance(raw, dict):
            raise ValueError("Continuation overlay binding is malformed")
        _, value, binding = _bound_document(
            raw, CONTINUATION_OVERLAY_SCHEMA, "continuation_overlay")
        overlays.append((value, binding))
    return _terminal_document(source, source_binding, audits, overlays)


def _verify_terminal_document(
    value: Mapping[str, Any], *, expected_stage: str | None = None,
    expected_controller: str | None = None,
    expected_algorithm_id: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_task_ids: Sequence[str] | None = None,
) -> None:
    cells = value.get("quality_cells")
    task_ids = value.get("task_ids")
    audit_bindings = value.get("capacity_failure_audits")
    overlay_bindings = value.get("continuation_overlays")
    if (
        value.get("schema") != TERMINAL_RESULT_SCHEMA
        or value.get("selection_scope") != SELECTION_SCOPE
        or value.get("stage") not in SUPPORTED_STAGES
        or value.get("controller") not in SUPPORTED_CONTROLLERS
        or not isinstance(value.get("source_algorithm_id"), str)
        or not value["source_algorithm_id"]
        or not _valid_sha(value.get("task_manifest_sha256"))
        or value.get("expected_task_cells") != 128
        or value.get("operational_denominator") != 128
        or value.get("official_zero_imputed") is not False
        or value.get("automatic_retries") != 0
        or value.get("automatic_reruns") != 0
        or not isinstance(cells, list)
        or len(cells) != 128
        or not isinstance(task_ids, list)
        or [cell.get("task_id") for cell in cells] != task_ids
        or len(task_ids) != len(set(task_ids))
        or not isinstance(audit_bindings, list)
        or any(
            not isinstance(binding, dict)
            or not isinstance(binding.get("path"), str)
            or not binding["path"]
            or not _valid_sha(binding.get("sha256"))
            or binding.get("schema") not in {
                CAPACITY_AUDIT_SCHEMA, GENERIC_CAPACITY_AUDIT_SCHEMA}
            for binding in audit_bindings)
        or len({binding["sha256"] for binding in audit_bindings})
        != len(audit_bindings)
        or not isinstance(overlay_bindings, list)
        or any(
            not isinstance(binding, dict)
            or not isinstance(binding.get("path"), str)
            or not binding["path"]
            or not _valid_sha(binding.get("sha256"))
            or binding.get("schema") != CONTINUATION_OVERLAY_SCHEMA
            for binding in overlay_bindings)
    ):
        raise ValueError("Terminal D128 receipt is malformed")
    if (
        (expected_stage is not None and value.get("stage") != expected_stage)
        or (expected_controller is not None
            and value.get("controller") != expected_controller)
        or (expected_algorithm_id is not None
            and value.get("source_algorithm_id") != expected_algorithm_id)
        or (expected_manifest_sha256 is not None
            and value.get("task_manifest_sha256") != expected_manifest_sha256)
        or (expected_task_ids is not None and task_ids != list(expected_task_ids))
    ):
        raise ValueError("Terminal D128 receipt identity differs from expected source")

    counts = {name: 0 for name in (
        "completed", "audited_capacity_failure", "unknown_failure", "pending")}
    successes = 0
    for cell in cells:
        status = cell.get("status")
        if status not in counts:
            raise ValueError(f"Unknown terminal cell status: {status}")
        counts[status] += 1
        _evidence_list(cell.get("evidence"), f"terminal cell {cell.get('task_id')}")
        if status == "completed":
            official = cell.get("official")
            if (
                cell.get("runtime_completed") is not True
                or cell.get("worker_returncode") != 0
                or cell.get("server_returncode") != 0
                or not _official_valid(official)
                or cell.get("raw_official") is not None
                or cell.get("operational_success")
                is not (official.get("correct_count") == 1)
            ):
                raise ValueError(f"Invalid normal terminal cell: {cell.get('task_id')}")
            successes += int(cell["operational_success"])
        elif status == "audited_capacity_failure":
            binding = cell.get("audit_binding")
            if (
                cell.get("runtime_completed") is not False
                or cell.get("server_returncode") != 1
                or cell.get("official") is not None
                or not _official_valid(cell.get("raw_official"))
                or cell["raw_official"].get("correct_count") != 0
                or cell.get("operational_success") is not False
                or not isinstance(binding, dict)
                or not _valid_sha(binding.get("sha256"))
                or type(binding.get("failure_index")) is not int
                or not _valid_sha(binding.get("error_object_sha256"))
            ):
                raise ValueError(f"Invalid audited capacity failure: {cell.get('task_id')}")
        else:
            if cell.get("official") is not None or cell.get("operational_success") is not None:
                raise ValueError(f"Unresolved terminal cell claims a result: {cell.get('task_id')}")

    terminal = counts["completed"] + counts["audited_capacity_failure"]
    eligible = terminal == 128 and counts["unknown_failure"] == counts["pending"] == 0
    expected = {
        "terminal_task_cells": terminal,
        "normal_completed_task_cells": counts["completed"],
        "audited_capacity_failure_task_cells": counts["audited_capacity_failure"],
        "unknown_failure_task_cells": counts["unknown_failure"],
        "pending_task_cells": counts["pending"],
        "operational_success_count": successes,
        "accepted_official_score_task_cells": counts["completed"],
        "operational_official_unobserved_task_cells": (
            counts["audited_capacity_failure"]
            + counts["unknown_failure"] + counts["pending"]),
    }
    if any(value.get(key) != observed for key, observed in expected.items()):
        raise ValueError("Terminal D128 receipt counts differ from its cells")
    if (
        value.get("selection_eligible") is not eligible
        or value.get("status") != ("terminal" if eligible else "incomplete")
        or value.get("operational_non_success_count")
        != (128 - successes if eligible else None)
    ):
        raise ValueError("Terminal D128 eligibility differs from its cells")


def verify_terminal_receipt(
    path: Path, *, expected_sha256: str | None = None,
    expected_stage: str | None = None,
    expected_controller: str | None = None,
    expected_algorithm_id: str | None = None,
    expected_manifest_sha256: str | None = None,
    expected_task_ids: Sequence[str] | None = None,
    verify_inputs: bool = True,
) -> dict[str, Any]:
    path = path.resolve()
    observed_sha = _sha(path)
    if expected_sha256 is not None and observed_sha != expected_sha256:
        raise ValueError("Terminal D128 receipt differs from its exact hash binding")
    value = _read(path)
    _verify_terminal_document(
        value, expected_stage=expected_stage,
        expected_controller=expected_controller,
        expected_algorithm_id=expected_algorithm_id,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_task_ids=expected_task_ids,
    )
    if verify_inputs:
        source_raw = value.get("source_result_receipt")
        if not isinstance(source_raw, dict):
            raise ValueError("Terminal receipt lacks source_result_receipt binding")
        _, source, source_binding = _bound_document(
            source_raw, SOURCE_RESULT_SCHEMA, "source_result_receipt")
        audit_raw = value.get("capacity_failure_audits")
        if not isinstance(audit_raw, list):
            raise ValueError("Terminal receipt capacity audit bindings are malformed")
        audits = []
        for raw in audit_raw:
            if not isinstance(raw, dict):
                raise ValueError("Terminal receipt capacity audit binding is malformed")
            _, audit, binding = _bound_failure_audit(raw)
            audits.append((audit, binding))
        overlay_raw = value.get("continuation_overlays")
        if not isinstance(overlay_raw, list):
            raise ValueError("Terminal receipt continuation bindings are malformed")
        overlays = []
        for raw in overlay_raw:
            if not isinstance(raw, dict):
                raise ValueError("Terminal receipt continuation binding is malformed")
            _, overlay, binding = _bound_document(
                raw, CONTINUATION_OVERLAY_SCHEMA, "continuation_overlay")
            overlays.append((overlay, binding))
        rebuilt = _terminal_document(
            source, source_binding, audits, overlays,
            generated_at=value.get("generated_at"),
        )
        if rebuilt != value:
            raise ValueError("Terminal receipt differs from its bound source inputs")
    return {"path": path, "document": value, "sha256": observed_sha}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--spec", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        value = build_terminal_receipt(args.spec)
        _save(args.output.resolve(), value)
        result = verify_terminal_receipt(args.output.resolve())
        print(json.dumps({
            "schema": TERMINAL_RESULT_SCHEMA,
            "status": value["status"],
            "selection_eligible": value["selection_eligible"],
            "receipt": str(result["path"]),
            "sha256": result["sha256"],
        }, ensure_ascii=False, indent=2))
        return 0
    result = verify_terminal_receipt(args.receipt)
    print(json.dumps({
        "schema": TERMINAL_RESULT_SCHEMA,
        "status": "verified",
        "receipt_status": result["document"]["status"],
        "selection_eligible": result["document"]["selection_eligible"],
        "receipt": str(result["path"]),
        "sha256": result["sha256"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
