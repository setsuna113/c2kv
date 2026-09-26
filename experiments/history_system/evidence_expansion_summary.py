"""Aggregate raw Experiment 3 D128 shard outputs into bound result receipts.

This module is read-only with respect to frozen evaluation packages.  It reads
their contracts, task manifests, stage manifests, server finals, and official
BFCL summaries.  A receipt is complete only when every one of its 128 exact
tasks has a successful runtime outcome and a valid one-task official score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


COMPLETE_RESULT_SCHEMA = "experiment3-complete-d128-result-v1"
INDEX_SCHEMA = "experiment3-complete-d128-result-index-v1"
EXPANSION_PACKAGE_SCHEMA = "experiment3-d128-expansion-package-v1"
COMBINATION_PACKAGE_SCHEMA = "experiment3-h1-r3-package-v1"
EXPANSION_READINESS_SCHEMA = "experiment3-expansion-readiness-v1"
EXPANSION_BINDINGS_SCHEMA = "experiment3-expansion-source-bindings-v1"
EXPANSION_AUDIT_SCHEMA = "evidence-sets-d20-runtime-compatibility-v1"
H1_SHARDS_PER_CONTROLLER = 3
R3_SHARD_COUNT = 6
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


def _evidence(path: Path, role: str) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"role": role, "path": str(path.resolve()), "sha256": _sha(path)}


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


def _slim_official(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "benchmark", "categories", "mode", "n_total", "n_scored",
        "correct_count", "semantic_score", "scored",
        "total_gold_checker_seconds", "total_handler_http_calls",
    )
    return {key: value.get(key) for key in keys if key in value}


def _strict_total(value: Any, key: str) -> int | float | None:
    row = value.get(key) if isinstance(value, dict) else None
    total = row.get("strict_total") if isinstance(row, dict) else None
    return total if isinstance(total, (int, float)) and not isinstance(total, bool) else None


def _cost_fields(final: Mapping[str, Any], official: Mapping[str, Any]) -> dict[str, Any]:
    summary = final.get("cost_summary")
    summary = summary if isinstance(summary, dict) else {}
    costs = summary.get("costs")
    costs = costs if isinstance(costs, dict) else {}
    resident = costs.get("openai_resident_usage")
    resident = resident if isinstance(resident, dict) else {}
    work = costs.get("actual_model_work")
    work = work if isinstance(work, dict) else {}
    flat = final.get("cost")
    flat = flat if isinstance(flat, dict) else {}

    def number(value: Any) -> int | float | None:
        return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    attempts = summary.get("generation_attempts", flat.get("generation_attempts"))
    return {
        "wall_seconds": number(final.get("wall_seconds")),
        "gold_checker_seconds": number(official.get("total_gold_checker_seconds")),
        "handler_http_calls": (
            official.get("total_handler_http_calls")
            if type(official.get("total_handler_http_calls")) is int else None
        ),
        "generation_attempts": attempts if type(attempts) is int else None,
        "prompt_tokens": _strict_total(resident, "prompt_tokens")
        if resident else number(flat.get("prompt_tokens")),
        "completion_tokens": _strict_total(resident, "completion_tokens")
        if resident else number(flat.get("completion_tokens")),
        "total_tokens": _strict_total(resident, "total_tokens")
        if resident else number(flat.get("total_tokens")),
        "materialized_encoder_tokens": _strict_total(
            work, "materialized_encoder_tokens")
        if work else number(flat.get("materialized_encoder_tokens")),
    }


def _unique_task_ids(value: Any, expected: int, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != expected
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != expected
    ):
        raise ValueError(f"{label} must contain {expected} unique task IDs")
    return list(value)


def _outcomes(stage: Mapping[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    rows = stage.get("task_outcomes")
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("task_id"), str):
            raise ValueError(f"Malformed task outcome in {path}")
        task_id = row["task_id"]
        if task_id in result:
            raise ValueError(f"Duplicate task outcome for {task_id} in {path}")
        result[task_id] = row
    return result


def _pending_cell(task_id: str, cohort: str, quality_source_id: str,
                  shard_id: str | None, reason: str,
                  evidence: Sequence[dict[str, str]] = ()) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "cohort": cohort,
        "quality_source_id": quality_source_id,
        "shard_id": shard_id,
        "status": "pending",
        "status_reason": reason,
        "runtime_completed": None,
        "worker_returncode": None,
        "server_returncode": None,
        "official": None,
        "measured_cost": {},
        "evidence": list(evidence),
    }


def _raw_task_cell(lane_root: Path, task_id: str, *, cohort: str,
                   quality_source_id: str, shard_id: str,
                   outcome: Mapping[str, Any] | None,
                   stage_evidence: dict[str, str]) -> dict[str, Any]:
    if outcome is None or outcome.get("outcome") == "not_started":
        return _pending_cell(
            task_id, cohort, quality_source_id, shard_id,
            "not_started_or_not_yet_recorded", [stage_evidence],
        )
    shard = lane_root / "results" / "task_shards" / task_id
    final_path = shard / "server" / "final.json"
    official_path = shard / "bfcl" / "official_summary.json"
    row_evidence = [stage_evidence]
    final: dict[str, Any] = {}
    official: dict[str, Any] | None = None
    if final_path.is_file():
        row_evidence.append(_evidence(final_path, "server_final"))
        final = _read(final_path)
    if official_path.is_file():
        row_evidence.append(_evidence(official_path, "official_summary"))
        official = _read(official_path)
    strict = (
        outcome.get("outcome") == "official_completed"
        and outcome.get("runtime_completed") is True
        and outcome.get("worker_returncode") == 0
        and outcome.get("server_returncode") == 0
        and final_path.is_file()
        and official is not None
        and _official_valid(official)
    )
    if strict:
        return {
            "task_id": task_id,
            "cohort": cohort,
            "quality_source_id": quality_source_id,
            "shard_id": shard_id,
            "status": "completed",
            "status_reason": "official_completed",
            "runtime_completed": True,
            "worker_returncode": 0,
            "server_returncode": 0,
            "official": _slim_official(official),
            "measured_cost": _cost_fields(final, official),
            "evidence": row_evidence,
        }
    reasons = []
    if outcome.get("outcome") != "official_completed":
        reasons.append(str(outcome.get("outcome") or "missing_terminal_outcome"))
    if outcome.get("runtime_completed") is not True:
        reasons.append("runtime_not_completed")
    if outcome.get("worker_returncode") != 0:
        reasons.append("worker_returncode_not_zero")
    if outcome.get("server_returncode") != 0:
        reasons.append("server_returncode_not_zero")
    if not final_path.is_file():
        reasons.append("server_final_missing")
    if official is None or not _official_valid(official):
        reasons.append("official_summary_missing_or_invalid")
    return {
        "task_id": task_id,
        "cohort": cohort,
        "quality_source_id": quality_source_id,
        "shard_id": shard_id,
        "status": "runtime_failed",
        "status_reason": ";".join(dict.fromkeys(reasons)),
        "runtime_completed": outcome.get("runtime_completed") is True,
        "worker_returncode": outcome.get("worker_returncode"),
        "server_returncode": outcome.get("server_returncode"),
        "official": _slim_official(official) if official is not None else None,
        "measured_cost": _cost_fields(final, official or {}),
        "evidence": row_evidence,
    }


def _historical_cell(cell: Any, task_id: str, readiness_evidence: dict[str, str],
                     audit_evidence: dict[str, str] | None) -> dict[str, Any]:
    evidence = [readiness_evidence]
    if audit_evidence is not None:
        evidence.append(audit_evidence)
    owner_value = cell.get("quality_source_id") if isinstance(cell, dict) else None
    valid_owner = isinstance(owner_value, str) and bool(owner_value)
    owner = owner_value if valid_owner else "unknown_historical_owner"
    if (
        isinstance(cell, dict)
        and cell.get("task_id") == task_id
        and cell.get("status") == "completed"
        and valid_owner
        and _official_valid(cell.get("official"))
    ):
        # Frozen readiness was produced only after _task_state verified
        # runtime_completed=True and a valid official summary.  Return codes are
        # normalized to the successful runner contract required downstream.
        return {
            "task_id": task_id,
            "cohort": "historical20",
            "quality_source_id": owner,
            "shard_id": None,
            "status": "completed",
            "status_reason": "validated_completed_by_frozen_d20_summary",
            "runtime_completed": True,
            "worker_returncode": 0,
            "server_returncode": 0,
            "official": _slim_official(cell["official"]),
            "measured_cost": dict(cell.get("measured_cost") or {}),
            "evidence": evidence,
        }
    source_status = cell.get("status") if isinstance(cell, dict) else None
    status = "pending" if source_status == "pending" else "runtime_failed"
    return {
        **_pending_cell(task_id, "historical20", owner, None,
                        str(cell.get("status_reason") or source_status or "missing_d20_cell")
                        if isinstance(cell, dict) else "missing_d20_cell", evidence),
        "status": status,
        "runtime_completed": False if status == "runtime_failed" else None,
        "official": _slim_official(cell["official"])
        if isinstance(cell, dict) and isinstance(cell.get("official"), dict) else None,
        "measured_cost": dict(cell.get("measured_cost") or {})
        if isinstance(cell, dict) else {},
    }


def _view(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed"]
    failed = [row for row in rows if row.get("status") == "runtime_failed"]
    pending = [row for row in rows if row.get("status") == "pending"]
    correct = [row["official"]["correct_count"] for row in completed]
    semantic = [float(row["official"]["semantic_score"]) for row in completed]
    return {
        "expected_task_cells": len(rows),
        "completed_task_cells": len(completed),
        "runtime_failed_task_cells": len(failed),
        "pending_task_cells": len(pending),
        "correct_count_over_completed": sum(correct),
        "semantic_score_sum_over_completed": sum(semantic),
        "task_ids": [row["task_id"] for row in rows],
        "quality_cells": list(rows),
    }


def _top_evidence(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    result = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (row["path"], row["sha256"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _receipt(*, package_kind: str, stage: str, controller: str,
             algorithm_id: str, manifest_sha256: str,
             configuration_binding: Mapping[str, Any],
             quality_cells: Sequence[dict[str, Any]],
             evidence: Sequence[dict[str, str]],
             historical: Sequence[dict[str, Any]] = (),
             new: Sequence[dict[str, Any]] = ()) -> dict[str, Any]:
    if len(quality_cells) != 128 or len({row["task_id"] for row in quality_cells}) != 128:
        raise ValueError(f"{controller}: combined result does not cover 128 unique tasks")
    combined = _view(quality_cells)
    complete = combined["completed_task_cells"] == 128
    return {
        "schema": COMPLETE_RESULT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package_kind": package_kind,
        "stage": stage,
        "status": "completed" if complete else "incomplete",
        "quality_label": "preliminary, n=1",
        "controller": controller,
        "source_algorithm_id": algorithm_id,
        "task_manifest_sha256": manifest_sha256,
        "expected_task_cells": 128,
        "completed_task_cells": combined["completed_task_cells"],
        "runtime_failed_task_cells": combined["runtime_failed_task_cells"],
        "pending_task_cells": combined["pending_task_cells"],
        "configuration_binding": dict(configuration_binding),
        "evidence": _top_evidence(evidence),
        "historical20": _view(historical) if historical else None,
        "new108": _view(new) if new else None,
        "combined_with_provenance": combined,
        "quality_cells": list(quality_cells),
    }


def _shard_paths(package: Path, row: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    shard_id = row.get("shard_id")
    relative = row.get("relative_package", shard_id)
    if not isinstance(shard_id, str) or not isinstance(relative, str):
        raise ValueError("Malformed shard row")
    root = (package / "shards" / relative).resolve()
    expected_parent = (package / "shards").resolve()
    try:
        root.relative_to(expected_parent)
    except ValueError as error:
        raise ValueError(f"Shard path escapes package: {relative}") from error
    return root, root / "lanes" / shard_id, root / "tasks.json"


def _raw_shard(package: Path, row: Mapping[str, Any], expected_tasks: int,
               cohort: str) -> dict[str, Any]:
    shard_id = row["shard_id"]
    root, lane_root, tasks_path = _shard_paths(package, row)
    tasks_doc = _read(tasks_path)
    tasks = _unique_task_ids(tasks_doc.get("task_ids"), expected_tasks, f"{shard_id} tasks")
    design_path = lane_root / "design.json"
    controller_path = lane_root / "runtime" / "configs" / "controller.json"
    provenance_path = root / "provenance.json"
    design = _read(design_path)
    _read(controller_path)
    provenance = _read(provenance_path)
    algorithm_id = design.get("candidate_id")
    if not isinstance(algorithm_id, str) or not algorithm_id:
        raise ValueError(f"{shard_id}: design has no candidate_id")
    if design.get("task_ids") != tasks:
        raise ValueError(f"{shard_id}: design tasks differ from shard manifest")
    if provenance.get("controller") != row.get("controller"):
        raise ValueError(f"{shard_id}: provenance controller differs")
    expected_algorithm = (
        provenance.get("output_algorithm_id")
        if row.get("stage") in {"H1_R1", "R3"}
        else provenance.get("source_algorithm_id")
    )
    if expected_algorithm != algorithm_id:
        raise ValueError(f"{shard_id}: provenance does not bind evaluated candidate_id")
    stage_path = lane_root / "results" / "stage_manifest.json"
    if not stage_path.is_file():
        stage = {}
        stage_evidence = _evidence(design_path, "design_without_stage_manifest")
    else:
        stage = _read(stage_path)
        stage_evidence = _evidence(stage_path, "stage_manifest")
    outcomes = _outcomes(stage, stage_path)
    cells = {
        task_id: _raw_task_cell(
            lane_root, task_id, cohort=cohort,
            quality_source_id=algorithm_id, shard_id=shard_id,
            outcome=outcomes.get(task_id), stage_evidence=stage_evidence,
        )
        for task_id in tasks
    }
    return {
        "shard_id": shard_id,
        "controller": row.get("controller"),
        "stage": row.get("stage", "H0_R1"),
        "tasks": tasks,
        "algorithm_id": algorithm_id,
        "provenance": provenance,
        "evaluated_controller_sha256": _sha(controller_path),
        "cells": cells,
        "evidence": [
            _evidence(tasks_path, "shard_task_manifest"),
            _evidence(design_path, "evaluated_design"),
            _evidence(controller_path, "evaluated_controller_config"),
            _evidence(provenance_path, "shard_provenance"),
            *([_evidence(stage_path, "stage_manifest")] if stage_path.is_file() else []),
        ],
    }


def _expansion_receipts(package: Path) -> dict[str, dict[str, Any]]:
    contract_path = package / "expansion_contract.json"
    readiness_path = package / "readiness.json"
    bindings_path = package / "source_bindings.json"
    audit_path = package / "d20_compatibility_audit.json"
    contract = _read(contract_path)
    readiness = _read(readiness_path)
    bindings_doc = _read(bindings_path)
    audit = _read(audit_path)
    if contract.get("schema") != EXPANSION_PACKAGE_SCHEMA:
        raise ValueError("Unknown expansion package schema")
    if readiness.get("schema") != EXPANSION_READINESS_SCHEMA:
        raise ValueError("Unknown expansion readiness schema")
    if bindings_doc.get("schema") != EXPANSION_BINDINGS_SCHEMA:
        raise ValueError("Unknown expansion source bindings schema")
    if audit.get("schema") != EXPANSION_AUDIT_SCHEMA:
        raise ValueError("Unknown D20 compatibility audit schema")
    if (
        audit.get("conclusion", {}).get("status")
        != "semantic_nonactivation_compatible"
        or audit.get("conclusion", {}).get("result_reuse_supported") is not True
    ):
        raise ValueError("D20 compatibility audit does not authorize reuse")
    selected = contract.get("selected_controllers")
    if not isinstance(selected, list) or len(selected) != 2 or len(set(selected)) != 2:
        raise ValueError("Expansion contract must contain two selected controllers")
    if readiness.get("promotion", {}).get("selected") != selected:
        raise ValueError("Expansion contract and readiness select different controllers")
    manifest_sha = readiness.get("inputs", {}).get("d128", {}).get("sha256")
    if not _valid_sha(manifest_sha):
        raise ValueError("Readiness has no valid D128 manifest SHA256")
    audit_sha = _sha(audit_path)
    if (
        contract.get("d20_compatibility_audit_sha256") != audit_sha
        or readiness.get("inputs", {}).get("reuse_audit", {}).get("sha256") != audit_sha
        or bindings_doc.get("reuse_audit", {}).get("sha256") != audit_sha
    ):
        raise ValueError("Expansion package does not bind its copied D20 audit")
    rows = contract.get("shards")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("Expansion package must contain six shards")
    bindings = bindings_doc.get("bindings")
    if not isinstance(bindings, dict) or set(bindings) != set(selected):
        raise ValueError("Source bindings do not cover selected controllers")
    readiness_evidence = _evidence(readiness_path, "frozen_promotion_readiness")
    audit_evidence = _evidence(audit_path, "d20_semantic_compatibility_audit")
    base_evidence = [
        _evidence(contract_path, "expansion_contract"), readiness_evidence,
        _evidence(bindings_path, "source_bindings"), audit_evidence,
    ]
    audited = set(readiness.get("reuse_validation", {}).get("audited_lanes", []))
    receipts: dict[str, dict[str, Any]] = {}
    for controller in selected:
        manifest = readiness.get("next_manifests", {}).get(controller)
        if not isinstance(manifest, dict):
            raise ValueError(f"{controller}: missing remaining-108 manifest")
        reused = _unique_task_ids(manifest.get("reused_task_ids"), 20, f"{controller} reused")
        new_tasks = _unique_task_ids(manifest.get("task_ids"), 108, f"{controller} new")
        full = _unique_task_ids(manifest.get("full_task_ids"), 128, f"{controller} full")
        if set(reused) & set(new_tasks) or set(reused) | set(new_tasks) != set(full):
            raise ValueError(f"{controller}: historical20 and new108 do not form D128")
        lane = readiness.get("lanes", {}).get(controller)
        cells = lane.get("quality_cells") if isinstance(lane, dict) else None
        if not isinstance(cells, list) or [row.get("task_id") for row in cells] != reused:
            raise ValueError(f"{controller}: frozen D20 quality cells differ from reuse order")
        historical = [
            _historical_cell(row, task_id, readiness_evidence,
                             audit_evidence if controller in audited else None)
            for task_id, row in zip(reused, cells)
        ]
        shard_rows = [row for row in rows if row.get("controller") == controller]
        if len(shard_rows) != 3:
            raise ValueError(f"{controller}: expected three 36-task shards")
        shards = [_raw_shard(package, row, 36, "new108") for row in shard_rows]
        algorithm_ids = {row["algorithm_id"] for row in shards}
        if len(algorithm_ids) != 1:
            raise ValueError(f"{controller}: shards evaluate different candidate IDs")
        algorithm_id = next(iter(algorithm_ids))
        binding = bindings[controller]
        if (
            not isinstance(binding, dict)
            or binding.get("controller") != controller
            or binding.get("source_algorithm_id") != algorithm_id
            or not _valid_sha(binding.get("source_design_sha256"))
            or not _valid_sha(binding.get("source_controller_sha256"))
        ):
            raise ValueError(f"{controller}: source binding differs from evaluated candidate")
        controller_hashes = {row["evaluated_controller_sha256"] for row in shards}
        if (
            len(controller_hashes) != 1
            or next(iter(controller_hashes)) != binding["source_controller_sha256"]
        ):
            raise ValueError(f"{controller}: evaluated controller config differs from source binding")
        new_by_task: dict[str, dict[str, Any]] = {}
        for shard in shards:
            for task_id, cell in shard["cells"].items():
                if task_id in new_by_task:
                    raise ValueError(f"{controller}: duplicate new task owner {task_id}")
                new_by_task[task_id] = cell
        if set(new_by_task) != set(new_tasks):
            raise ValueError(f"{controller}: shard tasks do not exactly cover remaining108")
        new = [new_by_task[task_id] for task_id in new_tasks]
        all_by_task = {row["task_id"]: row for row in [*historical, *new]}
        combined = [all_by_task[task_id] for task_id in full]
        shard_evidence = [item for shard in shards for item in shard["evidence"]]
        config_status = (
            "audited_semantic_compatibility" if controller in audited
            else "exact_frozen_config"
        )
        if controller in audited:
            audit_lane = audit.get("lanes", {}).get(controller)
            audit_tasks = audit_lane.get("tasks") if isinstance(audit_lane, dict) else None
            if (
                not isinstance(audit_lane, dict)
                or audit_lane.get("classification")
                != "semantic_nonactivation_compatible_20_of_20"
                or not isinstance(audit_tasks, list)
                or len(audit_tasks) != 20
                or {row.get("task_id") for row in audit_tasks} != set(reused)
            ):
                raise ValueError(f"{controller}: copied audit does not bind its historical20")
        configuration = {
            "status": config_status,
            "source_algorithm_id": algorithm_id,
            "source_design_sha256": binding.get("source_design_sha256"),
            "source_controller_sha256": binding.get("source_controller_sha256"),
            "evaluated_controller_sha256": next(iter(controller_hashes)),
            "d20_compatibility_audit_sha256": audit_sha
            if controller in audited else None,
        }
        receipts[controller] = _receipt(
            package_kind="H0_remaining108_expansion", stage="H0_R1",
            controller=controller, algorithm_id=algorithm_id,
            manifest_sha256=manifest_sha, configuration_binding=configuration,
            quality_cells=combined, evidence=[*base_evidence, *shard_evidence],
            historical=historical, new=new,
        )
    return receipts


def _combination_receipts(package: Path) -> dict[str, dict[str, Any]]:
    contract_path = package / "combination_contract.json"
    manifest_path = package / "tasks.d128.json"
    contract = _read(contract_path)
    manifest = _read(manifest_path)
    if contract.get("schema") != COMBINATION_PACKAGE_SCHEMA:
        raise ValueError("Unknown H1/R3 package schema")
    tasks = _unique_task_ids(manifest.get("task_ids"), 128, "combination D128")
    manifest_sha = _sha(manifest_path)
    if contract.get("d128_manifest_sha256") != manifest_sha:
        raise ValueError("Combination package D128 manifest hash differs")
    rows = contract.get("shards")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Combination package contains no shards")
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Malformed combination shard row")
        stage = row.get("stage")
        controller = row.get("controller")
        if stage not in {"H1_R1", "R3"} or not isinstance(controller, str):
            raise ValueError("Combination shard has invalid stage/controller")
        groups.setdefault((stage, controller), []).append(row)
    base_evidence = [
        _evidence(contract_path, "combination_contract"),
        _evidence(manifest_path, "d128_task_manifest"),
    ]
    receipts: dict[str, dict[str, Any]] = {}
    for (stage, controller), group_rows in groups.items():
        shard_count = H1_SHARDS_PER_CONTROLLER if stage == "H1_R1" else R3_SHARD_COUNT
        if len(group_rows) != shard_count:
            raise ValueError(f"{stage}/{controller}: expected {shard_count} D128 shards")
        try:
            indexed_rows = sorted(
                ((int(row["shard_id"].rsplit("part", 1)[1]), row)
                 for row in group_rows),
                key=lambda item: item[0],
            )
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise ValueError(f"{stage}/{controller}: malformed shard part") from error
        if [part for part, _ in indexed_rows] != list(range(shard_count)):
            raise ValueError(f"{stage}/{controller}: shard parts do not exactly cover partition")
        expected_sizes = [len(range(part, len(tasks), shard_count))
                          for part in range(shard_count)]
        if any(row.get("task_budget") != expected_sizes[part]
               for part, row in indexed_rows):
            raise ValueError(f"{stage}/{controller}: shard task budgets differ from D128")
        shards = [_raw_shard(package, row, expected_sizes[part], "full128")
                  for part, row in indexed_rows]
        algorithms = {row["algorithm_id"] for row in shards}
        if len(algorithms) != 1:
            raise ValueError(f"{stage}/{controller}: shards evaluate different candidate IDs")
        algorithm_id = next(iter(algorithms))
        by_task: dict[str, dict[str, Any]] = {}
        source_design_hashes = set()
        source_controller_hashes = set()
        evaluated_controller_hashes = set()
        for shard in shards:
            provenance = shard["provenance"]
            source_design_hashes.add(provenance.get("source_design_sha256"))
            source_controller_hashes.add(provenance.get("source_controller_sha256"))
            evaluated_controller_hashes.add(shard["evaluated_controller_sha256"])
            for task_id, cell in shard["cells"].items():
                if task_id in by_task:
                    raise ValueError(f"{stage}/{controller}: duplicate task owner {task_id}")
                by_task[task_id] = cell
        if set(by_task) != set(tasks):
            raise ValueError(f"{stage}/{controller}: shards do not exactly cover D128")
        if (
            len(source_design_hashes) != 1
            or len(source_controller_hashes) != 1
            or len(evaluated_controller_hashes) != 1
            or not all(_valid_sha(value) for value in
                       [*source_design_hashes, *source_controller_hashes])
        ):
            raise ValueError(f"{stage}/{controller}: frozen source identity differs across shards")
        combined = [by_task[task_id] for task_id in tasks]
        shard_evidence = [item for shard in shards for item in shard["evidence"]]
        key = f"{stage}__{controller}"
        receipts[key] = _receipt(
            package_kind="H1_R3_full128", stage=stage, controller=controller,
            algorithm_id=algorithm_id, manifest_sha256=manifest_sha,
            configuration_binding={
                "status": "exact_frozen_config",
                "stage": stage,
                "evaluated_candidate_id": algorithm_id,
                "source_design_sha256": next(iter(source_design_hashes)),
                "source_controller_sha256": next(iter(source_controller_hashes)),
                "evaluated_controller_sha256": next(iter(evaluated_controller_hashes)),
            },
            quality_cells=combined, evidence=[*base_evidence, *shard_evidence],
        )
    return receipts


def summarize_package(package: Path) -> dict[str, dict[str, Any]]:
    package = package.resolve()
    expansion = (package / "expansion_contract.json").is_file()
    combination = (package / "combination_contract.json").is_file()
    if expansion == combination:
        raise ValueError("Package must contain exactly one supported Experiment 3 contract")
    return _expansion_receipts(package) if expansion else _combination_receipts(package)


def write_receipts(package: Path, output_dir: Path) -> dict[str, Any]:
    package = package.resolve()
    output_dir = output_dir.resolve()
    receipts = summarize_package(package)
    rows = []
    for receipt_id, receipt in sorted(receipts.items()):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", receipt_id)
        path = output_dir / f"{safe}.complete_d128.json"
        _save(path, receipt)
        rows.append({
            "receipt_id": receipt_id,
            "controller": receipt["controller"],
            "stage": receipt["stage"],
            "status": receipt["status"],
            "completed_task_cells": receipt["completed_task_cells"],
            "path": str(path),
            "sha256": _sha(path),
        })
    index = {
        "schema": INDEX_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package": str(package),
        "receipts": rows,
    }
    _save(output_dir / "index.json", index)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_receipts(args.package, args.output_dir),
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
