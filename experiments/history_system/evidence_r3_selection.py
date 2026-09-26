"""Select the one predeclared Experiment 3 configuration eligible for R3.

This module is CPU-only.  It validates four mutually exclusive strict-complete
or fixed-denominator terminal H0/H1 D128 receipts,
uses raw step journals only when an identical-success-set tie requires the
extra-generation tiebreak, and emits the existing R3-only combination spec.
It never launches evaluators or substitutes wall time for selector cost.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import evidence_eval_combinations as combinations
import evidence_eval_expansion as expansion
import evidence_terminal_results as terminal_results
from evidence_expansion_costs import step_costs


INPUT_SCHEMA = "experiment3-r3-selection-input-v1"
LEADING_SCHEMA = combinations.LEADING_SCHEMA
TERMINAL_LEADING_SCHEMA = combinations.TERMINAL_LEADING_SCHEMA
COMPLETE_RESULT_SCHEMA = combinations.COMPLETE_RESULT_SCHEMA
TERMINAL_RESULT_SCHEMA = terminal_results.TERMINAL_RESULT_SCHEMA
EXPECTED_CONFIGS = {
    ("H0_R1", "C0"),
    ("H0_R1", "C5"),
    ("H1_R1", "C0"),
    ("H1_R1", "C5"),
}
SOURCE_CATALOG_SCHEMA = "experiment3-source-catalog-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _save_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")


def _binding(value: Any, label: str) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise ValueError(f"Missing {label} binding")
    path_value = value.get("path")
    expected = value.get("sha256")
    if (not isinstance(path_value, str) or not path_value
            or not expansion._valid_sha256(expected)):
        raise ValueError(f"Invalid {label} binding")
    path = Path(path_value).resolve()
    if not path.is_file() or _sha(path) != expected:
        raise ValueError(f"{label} differs from its explicit hash")
    return path, expected


def _config_source(value: Any) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise ValueError("Missing config_source_package binding")
    path_value = value.get("path")
    expected = value.get("source_evidence_sets_sha256")
    if (not isinstance(path_value, str) or not path_value
            or not expansion._valid_sha256(expected)):
        raise ValueError("Invalid config_source_package binding")
    root = Path(path_value).resolve()
    evidence_sets = root / "history_system/evidence_sets.py"
    if not evidence_sets.is_file() or _sha(evidence_sets) != expected:
        raise ValueError("config source evidence_sets.py differs")
    return root, expected


def _validate_manifest(binding: Any) -> tuple[Path, str, dict[str, Any]]:
    path, expected = _binding(binding, "d128_manifest")
    value = combinations.validate_d128_manifest(path, expected)
    return path, expected, value


def _receipt_candidate(binding: Mapping[str, Any], manifest: Mapping[str, Any],
                       manifest_sha256: str) -> dict[str, Any]:
    stage = binding.get("stage")
    controller = binding.get("controller")
    if (stage, controller) not in EXPECTED_CONFIGS:
        raise ValueError("Complete receipt has an unexpected stage/controller")
    path, expected = _binding(binding, f"{stage}/{controller} complete receipt")
    receipt = _read(path)
    cells = receipt.get("quality_cells")
    evidence = receipt.get("evidence")
    if (
        receipt.get("schema") != COMPLETE_RESULT_SCHEMA
        or receipt.get("status") != "completed"
        or receipt.get("stage") != stage
        or receipt.get("controller") != controller
        or receipt.get("task_manifest_sha256") != manifest_sha256
        or receipt.get("expected_task_cells") != 128
        or receipt.get("completed_task_cells") != 128
        or receipt.get("runtime_failed_task_cells") != 0
        or receipt.get("pending_task_cells") != 0
        or not isinstance(receipt.get("source_algorithm_id"), str)
        or not receipt["source_algorithm_id"]
        or not isinstance(evidence, list)
        or not evidence
        or not isinstance(cells, list)
        or len(cells) != 128
        or [row.get("task_id") for row in cells] != manifest["task_ids"]
    ):
        raise ValueError(f"{stage}/{controller}: incomplete or malformed D128 receipt")
    for cell in cells:
        official = cell.get("official")
        cell_evidence = cell.get("evidence")
        if (
            cell.get("status") != "completed"
            or cell.get("runtime_completed") is not True
            or cell.get("worker_returncode") != 0
            or cell.get("server_returncode") != 0
            or not isinstance(cell.get("quality_source_id"), str)
            or not isinstance(official, dict)
            or official.get("scored") is not True
            or official.get("n_total") != 1
            or official.get("n_scored") != 1
            or type(official.get("correct_count")) is not int
            or official["correct_count"] not in {0, 1}
            or not isinstance(cell_evidence, list)
            or not cell_evidence
        ):
            raise ValueError(f"{stage}/{controller}: D128 cell is not completed")
    success = [row["task_id"] for row in cells
               if row["official"]["correct_count"] == 1]
    return {
        "config_id": f"{stage}__{controller}",
        "stage": stage,
        "controller": controller,
        "receipt_path": path,
        "receipt_sha256": expected,
        "receipt": receipt,
        "algorithm_id": receipt["source_algorithm_id"],
        "success_task_ids": success,
        "success_set_sha256": _json_sha(success),
        "correct_count": len(success),
        "result_gate": "complete_d128",
    }


def _terminal_candidate(binding: Mapping[str, Any], manifest: Mapping[str, Any],
                        manifest_sha256: str) -> dict[str, Any]:
    stage = binding.get("stage")
    controller = binding.get("controller")
    if (stage, controller) not in EXPECTED_CONFIGS:
        raise ValueError("Terminal receipt has an unexpected stage/controller")
    path, expected = _binding(binding, f"{stage}/{controller} terminal receipt")
    verified = terminal_results.verify_terminal_receipt(
        path, expected_sha256=expected, expected_stage=stage,
        expected_controller=controller,
        expected_manifest_sha256=manifest_sha256,
        expected_task_ids=manifest["task_ids"], verify_inputs=True)
    receipt = verified["document"]
    cells = receipt.get("quality_cells")
    if (
        receipt.get("schema") != TERMINAL_RESULT_SCHEMA
        or receipt.get("status") != "terminal"
        or receipt.get("selection_eligible") is not True
        or receipt.get("selection_scope")
        != terminal_results.SELECTION_SCOPE
        or receipt.get("terminal_task_cells") != 128
        or receipt.get("unknown_failure_task_cells") != 0
        or receipt.get("pending_task_cells") != 0
        or receipt.get("operational_denominator") != 128
        or receipt.get("official_zero_imputed") is not False
        or not isinstance(cells, list)
        or len(cells) != 128
        or [row.get("task_id") for row in cells] != manifest["task_ids"]
    ):
        raise ValueError(f"{stage}/{controller}: ineligible terminal D128 receipt")
    success = [row["task_id"] for row in cells
               if row.get("operational_success") is True]
    if len(success) != receipt.get("operational_success_count"):
        raise ValueError(f"{stage}/{controller}: terminal success count differs")
    return {
        "config_id": f"{stage}__{controller}",
        "stage": stage,
        "controller": controller,
        "receipt_path": path,
        "receipt_sha256": expected,
        "receipt": receipt,
        "algorithm_id": receipt["source_algorithm_id"],
        "success_task_ids": success,
        "success_set_sha256": _json_sha(success),
        "correct_count": len(success),
        "result_gate": "terminal_d128",
    }


def _h0_source(candidate: Mapping[str, Any], catalog: Mapping[str, Any],
               config_source: Path, evidence_sets_sha256: str) -> dict[str, Any]:
    binding = catalog.get("bindings", {}).get(candidate["controller"])
    if not isinstance(binding, dict):
        raise ValueError(f"Missing H0 source catalog binding: {candidate['controller']}")
    source = dict(binding)
    result_key = ("terminal_d128_receipt"
                  if candidate["result_gate"] == "terminal_d128"
                  else "complete_d128_receipt")
    source.update({
        "schema": combinations.SOURCE_SCHEMA,
        "config_source_package": str(config_source),
        "source_evidence_sets_sha256": evidence_sets_sha256,
        result_key: {
            "path": str(candidate["receipt_path"]),
            "sha256": candidate["receipt_sha256"],
        },
    })
    return source


def _verified_evidence_path(row: Any, label: str) -> Path:
    if not isinstance(row, dict):
        raise ValueError(f"Malformed {label} evidence")
    path_value = row.get("path")
    expected = row.get("sha256")
    if (not isinstance(path_value, str) or not path_value
            or not expansion._valid_sha256(expected)):
        raise ValueError(f"Malformed {label} evidence")
    path = Path(path_value).resolve()
    if not path.is_file() or _sha(path) != expected:
        raise ValueError(f"{label} evidence differs from its hash")
    return path


def _h1_source(candidate: Mapping[str, Any], config_source: Path,
               evidence_sets_sha256: str) -> dict[str, Any]:
    lane_name = f"h1_{candidate['controller']}_part0"
    matches = []
    for row in candidate["receipt"].get("evidence", []):
        if not isinstance(row, dict) or row.get("role") != "evaluated_design":
            continue
        raw_path = row.get("path")
        if isinstance(raw_path, str) and Path(raw_path).parent.name == lane_name:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError(f"{candidate['config_id']}: expected one exact part0 evaluated design")
    design_path = _verified_evidence_path(matches[0], "H1 part0 design")
    lane_root = design_path.parent
    if lane_root.parent.name != "lanes":
        raise ValueError("H1 evaluated design is outside a shard lane")
    source = lane_root.parent.parent
    provenance_path = source / "provenance.json"
    launch_path = source / "launch_contract.json"
    static_path = source / "static_files.json"
    sglang_path = source / "sglang_files.json"
    controller_path = lane_root / "runtime/configs/controller.json"
    for required in (provenance_path, launch_path, static_path, sglang_path,
                     controller_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    design = _read(design_path)
    provenance = _read(provenance_path)
    launch = _read(launch_path)
    if (design.get("candidate_id") != candidate["algorithm_id"]
            or provenance.get("output_algorithm_id") != candidate["algorithm_id"]
            or provenance.get("controller") != candidate["controller"]):
        raise ValueError("H1 part0 source identity differs from complete receipt")
    artifacts = provenance.get("copied_model_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("H1 source lacks copied_model_artifacts provenance")
    result_key = ("terminal_d128_receipt"
                  if candidate["result_gate"] == "terminal_d128"
                  else "complete_d128_receipt")
    return {
        "schema": combinations.SOURCE_SCHEMA,
        "controller": candidate["controller"],
        "source_package": str(source),
        "source_lane": lane_name,
        "source_algorithm_id": candidate["algorithm_id"],
        "source_static_files_sha256": _sha(static_path),
        "source_sglang_files_sha256": _sha(sglang_path),
        "source_design_sha256": _sha(design_path),
        "source_controller_sha256": _sha(controller_path),
        "required_env": launch.get("required_environment", {}),
        "model_artifacts": artifacts,
        "config_source_package": str(config_source),
        "source_evidence_sets_sha256": evidence_sets_sha256,
        result_key: {
            "path": str(candidate["receipt_path"]),
            "sha256": candidate["receipt_sha256"],
        },
    }


def _source_for(candidate: dict[str, Any], catalog: Mapping[str, Any],
                config_source: Path, evidence_sets_sha256: str,
                manifest: Mapping[str, Any], manifest_sha256: str) -> None:
    binding = (_h0_source(candidate, catalog, config_source, evidence_sets_sha256)
               if candidate["stage"] == "H0_R1"
               else _h1_source(candidate, config_source, evidence_sets_sha256))
    verified = combinations._verify_source(
        binding, manifest, manifest_sha256,
        result_gate=candidate["result_gate"])
    source_gp = verified["gp"]
    expected_history = "current" if candidate["stage"] == "H0_R1" else "record_bound"
    if source_gp.get("G") != expected_history or source_gp.get("R") != 1:
        raise ValueError(f"{candidate['config_id']}: source is not its complete R1 configuration")
    candidate["source_binding"] = binding
    candidate["source"] = verified
    candidate["history"] = expected_history


def _audit_steps(evidence: Sequence[Any], controller: str,
                 task_id: str) -> tuple[Path, str] | None:
    for row in evidence:
        if not isinstance(row, dict):
            continue
        role = row.get("role")
        raw_path = row.get("path")
        if (not isinstance(role, str) or "compatibility" not in role
                or not isinstance(raw_path, str)):
            continue
        path = _verified_evidence_path(row, "D20 compatibility audit")
        audit = _read(path)
        if audit.get("schema") != "evidence-sets-d20-runtime-compatibility-v1":
            continue
        lane = audit.get("lanes", {}).get(controller, {})
        tasks = lane.get("tasks") if isinstance(lane, dict) else None
        if not isinstance(tasks, list):
            continue
        matches = [item for item in tasks if isinstance(item, dict)
                   and item.get("task_id") == task_id]
        if len(matches) != 1:
            continue
        observed = matches[0].get("observed")
        if not isinstance(observed, dict):
            continue
        steps_value = observed.get("steps_path")
        expected = observed.get("steps_sha256")
        if (not isinstance(steps_value, str) or not steps_value
                or not expansion._valid_sha256(expected)):
            continue
        steps = Path(steps_value).resolve()
        if not steps.is_file() or _sha(steps) != expected:
            raise ValueError(f"{controller}/{task_id}: audited steps differ")
        return steps, expected
    return None


def _steps_for_cell(candidate: Mapping[str, Any], cell: Mapping[str, Any]) -> tuple[Path, str]:
    evidence = cell.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("Cell lacks evidence")
    for row in evidence:
        if not isinstance(row, dict):
            continue
        role = row.get("role")
        if role == "steps_jsonl":
            path = _verified_evidence_path(row, "steps")
            return path, row["sha256"]
        raw_path = row.get("path")
        if not isinstance(raw_path, str):
            continue
        source = Path(raw_path)
        inferred = None
        if role == "server_final" and source.name == "final.json":
            inferred = source.parent / "steps.jsonl"
        elif role == "official_summary" and source.name == "official_summary.json":
            inferred = source.parent.parent / "server/steps.jsonl"
        if inferred is not None:
            inferred = inferred.resolve()
            if inferred.is_file():
                return inferred, _sha(inferred)
    audited = _audit_steps(
        evidence, candidate["controller"], cell["task_id"])
    if audited is not None:
        return audited
    raise FileNotFoundError(
        f"{candidate['config_id']}/{cell['task_id']}: exact steps unavailable")


def _exact_cost(candidate: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    total_extra = 0
    for cell in candidate["receipt"]["quality_cells"]:
        steps, expected = _steps_for_cell(candidate, cell)
        raw = steps.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Steps changed while collecting cost: {steps}")
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines()
                   if line.strip()]
        cost = step_costs(records)
        total_extra += cost["extra_generations"]
        rows.append({
            "task_id": cell["task_id"],
            "steps_path": str(steps),
            "steps_sha256": expected,
            "step_rows": len(records),
            "extra_generations": cost["extra_generations"],
            "selector_model_calls": cost["selector_model_calls"],
            "selector_model_call_seconds": cost["selector_model_call_seconds"],
            "selector_input_preparation_seconds": cost[
                "selector_input_preparation_seconds"],
        })
    if len(rows) != 128:
        raise ValueError("Exact cost does not cover D128")
    return {
        "status": "exact_full128",
        "covered_tasks": 128,
        "extra_generations": total_extra,
        "selector_cost_status": "unavailable_cpu_selector_not_isolated",
        "selector_cost_used": False,
        "evidence": rows,
    }


def _public_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    receipt_key = ("terminal_d128_receipt"
                   if candidate["result_gate"] == "terminal_d128"
                   else "complete_d128_receipt")
    result = {
        "config_id": candidate["config_id"],
        "stage": candidate["stage"],
        "controller": candidate["controller"],
        "history": candidate["history"],
        "recovery_rounds": 1,
        "source_algorithm_id": candidate["algorithm_id"],
        "result_gate": candidate["result_gate"],
        receipt_key: {
            "path": str(candidate["receipt_path"]),
            "sha256": candidate["receipt_sha256"],
        },
        "success_task_ids": candidate["success_task_ids"],
        "success_set_sha256": candidate["success_set_sha256"],
        "source_identity": {
            key: candidate["source_binding"][key] for key in (
                "source_package", "source_lane", "source_static_files_sha256",
                "source_sglang_files_sha256", "source_design_sha256",
                "source_controller_sha256", "source_evidence_sets_sha256")
        },
    }
    if candidate["result_gate"] == "terminal_d128":
        result["operational_success_count"] = candidate["correct_count"]
        result["terminal_coverage"] = {
            key: candidate["receipt"][key] for key in (
                "terminal_task_cells", "normal_completed_task_cells",
                "audited_capacity_failure_task_cells",
                "unknown_failure_task_cells", "pending_task_cells",
                "operational_denominator")
        }
    else:
        result["correct_count"] = candidate["correct_count"]
    if "cost" in candidate:
        result["cost"] = candidate["cost"]
    return result


def evaluate(input_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    input_path = input_path.resolve()
    spec = _read(input_path)
    if spec.get("schema") != INPUT_SCHEMA:
        raise ValueError("Unknown R3 selection input schema")
    manifest_path, manifest_sha256, manifest = _validate_manifest(
        spec.get("d128_manifest"))
    catalog_path, catalog_sha256 = _binding(
        spec.get("h0_source_catalog"), "H0 source catalog")
    catalog = _read(catalog_path)
    if catalog.get("schema") != SOURCE_CATALOG_SCHEMA:
        raise ValueError("Unknown H0 source catalog schema")
    config_source, evidence_sets_sha256 = _config_source(
        spec.get("config_source_package"))
    complete_bindings = spec.get("complete_d128_receipts")
    terminal_bindings = spec.get("terminal_d128_receipts")
    has_complete = complete_bindings is not None
    has_terminal = terminal_bindings is not None
    if has_complete == has_terminal:
        raise ValueError(
            "R3 selection requires exactly one complete or terminal receipt set")
    bindings = terminal_bindings if has_terminal else complete_bindings
    if not isinstance(bindings, list) or len(bindings) != 4:
        raise ValueError("R3 selection requires exactly four D128 receipts")
    candidate_builder = _terminal_candidate if has_terminal else _receipt_candidate
    candidates = [candidate_builder(row, manifest, manifest_sha256)
                  for row in bindings if isinstance(row, dict)]
    observed = {(row["stage"], row["controller"]) for row in candidates}
    if len(candidates) != 4 or observed != EXPECTED_CONFIGS:
        raise ValueError("R3 selection receipts must be C0/C5 x H0/H1 exactly once")
    for candidate in candidates:
        _source_for(candidate, catalog, config_source, evidence_sets_sha256,
                    manifest, manifest_sha256)

    highest = max(row["correct_count"] for row in candidates)
    top = [row for row in candidates if row["correct_count"] == highest]
    decision: dict[str, Any]
    selected: dict[str, Any] | None = None
    if len(top) == 1:
        selected = top[0]
        decision = {
            "status": "selected",
            "reason": (
                "unique_highest_fixed_d128_operational_success_count"
                if has_terminal
                else "unique_highest_complete_task_success_count"),
            "cost_tiebreak_used": False,
        }
    elif len({row["success_set_sha256"] for row in top}) != 1:
        decision = {
            "status": "unresolved",
            "reason": "complementary_success_sets_tied_on_success_count",
            "cost_tiebreak_used": False,
            "tied_config_ids": [row["config_id"] for row in top],
        }
    else:
        cost_errors = {}
        for candidate in top:
            try:
                candidate["cost"] = _exact_cost(candidate)
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as error:
                cost_errors[candidate["config_id"]] = (
                    f"{type(error).__name__}: {error}")
        if cost_errors:
            decision = {
                "status": "unresolved",
                "reason": "missing_or_invalid_exact_full128_extra_generation_cost",
                "cost_tiebreak_used": True,
                "cost_errors": cost_errors,
                "tied_config_ids": [row["config_id"] for row in top],
            }
        else:
            fewest = min(row["cost"]["extra_generations"] for row in top)
            cheapest = [row for row in top
                        if row["cost"]["extra_generations"] == fewest]
            if len(cheapest) == 1:
                selected = cheapest[0]
                decision = {
                    "status": "selected",
                    "reason": "identical_success_set_fewer_exact_extra_generations",
                    "cost_tiebreak_used": True,
                    "selector_cost_used": False,
                }
            else:
                decision = {
                    "status": "unresolved",
                    "reason": "exact_extra_generation_tie_without_isolated_selector_cost",
                    "cost_tiebreak_used": True,
                    "selector_cost_used": False,
                    "tied_config_ids": [row["config_id"] for row in cheapest],
                }

    result_gate = "terminal_d128" if has_terminal else "complete_d128"
    receipt: dict[str, Any] = {
        "schema": (TERMINAL_LEADING_SCHEMA if has_terminal else LEADING_SCHEMA),
        "status": (("selected_after_terminal_d128" if has_terminal
                    else "selected_after_complete_d128") if selected is not None
                   else ("unresolved_after_terminal_d128" if has_terminal
                         else "unresolved_after_complete_d128")),
        "generated_at": _now(),
        "quality_label": "preliminary, n=1",
        "task_manifest_sha256": manifest_sha256,
        "selection_rule": {
            "primary": ("fixed_d128_operational_success_count"
                        if has_terminal else "complete_task_success_count"),
            "same_success_count_and_identical_success_set": (
                "exact_extra_generations_then_isolated_selector_cost"),
            "different_success_sets_at_equal_count": "unresolved",
            "wall_time_is_not_selector_cost": True,
        },
        "decision": decision,
        "candidates": [_public_candidate(row) for row in candidates],
        "inputs": {
            "selection_input": {"path": str(input_path), "sha256": _sha(input_path)},
            "d128_manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
            "h0_source_catalog": {"path": str(catalog_path), "sha256": catalog_sha256},
            "config_source_package": str(config_source),
            "source_evidence_sets_sha256": evidence_sets_sha256,
            "result_gate": result_gate,
        },
        "model_calls_executed_by_emitter": 0,
        "task_executions_by_emitter": 0,
    }
    r3_spec = None
    if selected is not None:
        target_controller, _ = combinations.build_controller(
            selected["source"], stage="R3")
        source_gp = selected["source"]["gp"]
        target_gp = target_controller["gp_experiments"]
        changed = sorted(key for key in set(source_gp) | set(target_gp)
                         if source_gp.get(key) != target_gp.get(key))
        if changed != ["R"] or target_gp.get("R") != 3:
            raise ValueError("R3 transition changes fields outside recovery rounds")
        receipt_sha_key = ("terminal_d128_receipt_sha256"
                           if has_terminal
                           else "complete_d128_receipt_sha256")
        receipt.update({
            "controller": selected["controller"],
            "source_algorithm_id": selected["algorithm_id"],
            receipt_sha_key: selected["receipt_sha256"],
            "selected_config_id": selected["config_id"],
            "selected_source": selected["source_binding"],
            "r3_transition": {
                "source_history": source_gp.get("G"),
                "source_recovery_rounds": source_gp.get("R"),
                "target_history": target_gp.get("G"),
                "target_recovery_rounds": target_gp.get("R"),
                "changed_controller_fields": changed,
            },
        })
        r3_spec = {
            "schema": combinations.SPEC_SCHEMA,
            "d128_manifest": {"path": str(manifest_path),
                              "sha256": manifest_sha256},
            "h1_sources": [],
            "c1_h1_calibrations": {},
            "r3": {"source": selected["source_binding"]},
        }
    return receipt, r3_spec


def write_outputs(input_path: Path, leading_receipt_path: Path,
                  r3_spec_path: Path) -> dict[str, Any]:
    leading_receipt_path = leading_receipt_path.resolve()
    r3_spec_path = r3_spec_path.resolve()
    if leading_receipt_path.exists() or r3_spec_path.exists():
        raise FileExistsError("Refusing to overwrite R3 selection outputs")
    receipt, r3_spec = evaluate(input_path)
    _save_exclusive(leading_receipt_path, receipt)
    result = {
        "status": receipt["status"],
        "leading_receipt": {"path": str(leading_receipt_path),
                            "sha256": _sha(leading_receipt_path)},
        "r3_spec": None,
    }
    if r3_spec is not None:
        r3_spec["r3"]["leading_receipt"] = dict(result["leading_receipt"])
        _save_exclusive(r3_spec_path, r3_spec)
        result["r3_spec"] = {"path": str(r3_spec_path),
                             "sha256": _sha(r3_spec_path)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--leading-receipt", type=Path, required=True)
    parser.add_argument("--r3-spec", type=Path, required=True)
    args = parser.parse_args()
    result = write_outputs(args.input, args.leading_receipt, args.r3_spec)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
