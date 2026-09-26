from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import evidence_terminal_results as terminal


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official(correct: int) -> dict:
    return {
        "scored": True,
        "n_total": 1,
        "n_scored": 1,
        "correct_count": correct,
        "semantic_score": float(correct),
    }


def _source(tmp_path: Path, *, failed: tuple[int, ...] = (),
            pending: tuple[int, ...] = ()) -> tuple[Path, dict]:
    expansion_contract_sha = "e" * 64
    stage_path = tmp_path / "frozen/C0_part0/results/stage_manifest.json"
    status_path = tmp_path / "frozen/C0_part0/run/status.json"
    outcomes = []
    for index in range(128):
        task_id = f"multi_turn_base_{index}"
        if index in failed:
            outcomes.append({
                "task_id": task_id,
                "outcome": "runtime_failure_in_denominator",
                "in_fixed_denominator": True,
                "runtime_completed": False,
                "worker_returncode": 0,
                "server_returncode": 1,
                "official_summary": f"/frozen/{task_id}/official_summary.json",
            })
        elif index in pending:
            outcomes.append({
                "task_id": task_id, "outcome": "not_started",
                "in_fixed_denominator": True, "official_summary": None,
            })
        else:
            outcomes.append({
                "task_id": task_id, "outcome": "official_completed",
                "in_fixed_denominator": True, "runtime_completed": True,
                "worker_returncode": 0, "server_returncode": 0,
                "official_summary": f"/frozen/{task_id}/official_summary.json",
            })
    _save(stage_path, {"task_outcomes": outcomes})
    _save(status_path, {"state": "failed" if failed else "completed"})
    stage_sha = _sha(stage_path)
    cells = []
    for index in range(128):
        task_id = f"multi_turn_base_{index}"
        evidence = [{
            "role": "stage_manifest",
            "path": str(stage_path),
            "sha256": stage_sha,
        }]
        if index in failed:
            cell = {
                "task_id": task_id, "cohort": "full128",
                "quality_source_id": "c0_h0", "shard_id": "C0_part0",
                "status": "runtime_failed", "status_reason": "runtime_failure",
                "runtime_completed": False, "worker_returncode": 0,
                "server_returncode": 1, "official": _official(0),
                "measured_cost": {}, "evidence": evidence,
            }
        elif index in pending:
            cell = {
                "task_id": task_id, "cohort": "full128",
                "quality_source_id": "c0_h0", "shard_id": "C0_part0",
                "status": "pending", "status_reason": "not_started",
                "runtime_completed": None, "worker_returncode": None,
                "server_returncode": None, "official": None,
                "measured_cost": {}, "evidence": evidence,
            }
        else:
            cell = {
                "task_id": task_id, "cohort": "full128",
                "quality_source_id": "c0_h0", "shard_id": "C0_part0",
                "status": "completed", "status_reason": "official_completed",
                "runtime_completed": True, "worker_returncode": 0,
                "server_returncode": 0, "official": _official(index % 2),
                "measured_cost": {"generation_attempts": 1},
                "evidence": evidence,
            }
        cells.append(cell)
    value = {
        "schema": terminal.SOURCE_RESULT_SCHEMA,
        "status": "completed" if not failed and not pending else "incomplete",
        "package_kind": "H0_remaining108_expansion",
        "stage": "H0_R1", "controller": "C0",
        "source_algorithm_id": "c0_h0",
        "task_manifest_sha256": "d" * 64,
        "completed_task_cells": 128 - len(failed) - len(pending),
        "runtime_failed_task_cells": len(failed),
        "pending_task_cells": len(pending),
        "configuration_binding": {"status": "exact_frozen_config"},
        "evidence": [{
            "role": "expansion_contract",
            "path": "/frozen/d128_h0_v2/expansion_contract.json",
            "sha256": expansion_contract_sha,
        }],
        "quality_cells": cells,
    }
    path = tmp_path / "source.json"
    _save(path, value)
    return path, value


def _audit_failure(index: int) -> dict:
    task_id = f"multi_turn_base_{index}"
    return {
        "shard": "C0_part0", "controller": "C0", "task_id": task_id,
        "quality_status": "runtime_failed",
        "stage_outcome": {
            "task_id": task_id, "outcome": "runtime_failure_in_denominator",
            "in_fixed_denominator": True, "runtime_completed": False,
            "worker_returncode": 0, "server_returncode": 1,
        },
        "classification": {
            "status": "audited_in_contract",
            "selection_eligible": True,
            "budget_contract_verified": True,
            "family": "pre_generation_capacity_infeasible",
            "exception_type": "CapacityInfeasible",
            "source_exception": "CapacityInfeasible",
            "is_sglang_extraction_budget_exhausted": False,
            "is_sglang_http_failure": False,
            "is_endpoint_or_startup_failure": False,
            "budget_contract": {
                "eval_capacity_max_sequence_tokens": 40960,
                "design_resolved_max_sequence_tokens": 40960,
                "startup_effective_max_sequence_tokens": 40960,
                "backend_context_length": 131072,
                "model_context": 262144,
                "candidate_sequence_tokens_min": 41283,
                "candidate_sequence_tokens_max": 49939,
                "binding_limit": "eval_capacity_max_sequence_tokens",
            },
            "checks": {"fixture_budget_contract": True},
        },
        "failure": {
            "failed_step_generation_attempts": 0,
            "failed_step_generation_completed": 0,
            "failed_step_generation_trace_count": 0,
            "error_object_sha256": f"{2000 + index:064x}",
        },
        "declared_budget_and_failure_measurement": {
            "failure_reason_markers": [
                "physical_sequence_budget:4", "physical_sequence_budget:8"],
        },
        "runtime_vs_official": {
            "runtime_completed": False, "server_returncode": 1,
            "official_summary_exists": True,
            "official_score_does_not_convert_runtime_failure_to_completed": True,
            "official_zero_imputed": False, "official": _official(0),
        },
        "evidence": {
            role: {"path": f"/proof/{task_id}/{role}",
                   "sha256": f"{offset + index:064x}"}
            for offset, role in enumerate((
                "stage_manifest", "steps", "attempts", "server_final",
                "official_summary", "frozen_capacity_exception_source",
                "frozen_capacity_policy_source", "evaluated_design", "source_design",
                "frozen_eval_capacity", "server_startup", "engine_log",
                "engine_status"), start=3000)
        },
    }


def _audit(tmp_path: Path, failed: tuple[int, ...]) -> Path:
    failures = [_audit_failure(index) for index in failed]
    stage_path = tmp_path / "frozen/C0_part0/results/stage_manifest.json"
    for row in failures:
        row["evidence"]["stage_manifest"] = {
            "path": str(stage_path), "sha256": _sha(stage_path)}
    value = {
        "schema": terminal.CAPACITY_AUDIT_SCHEMA, "status": "audited",
        "scope": {
            "automatic_retry_or_rerun": False,
            "fixed_d128_runtime_outcomes": True,
            "official_scores_preserved": True,
            "official_zero_used_as_clean_runtime_completion": False,
            "selection_eligible": True,
            "unknown_failure_count": 0,
            "audited_capacity_failure_count": len(failures),
        },
        "package": {"expansion_contract": {
            "path": "/frozen/d128_h0_v2/expansion_contract.json",
            "sha256": "e" * 64,
        }},
        "failures": failures,
    }
    path = tmp_path / "audit.json"
    _save(path, value)
    return path


def _generic_audit(tmp_path: Path, source: dict, index: int, *,
                   continuation: bool = False) -> Path:
    task_id = f"multi_turn_base_{index}"
    row = _audit_failure(index)
    owner = "C0_part0_remainder" if continuation else "C0_part0"
    package_root = str(tmp_path / "continuation_package") if continuation else (
        "/frozen/d128_h1_v1" if source["stage"] == "H1_R1"
        else "/frozen/d128_h0_v2")
    contract_sha = "7" * 64 if continuation else "e" * 64
    static_sha = "6" * 64
    provenance_sha = "5" * 64
    design_sha = "4" * 64
    startup_sha = "3" * 64
    capacity_sha = "2" * 64
    row.update(
        stage=source["stage"], shard=owner,
        source_algorithm_id=source["source_algorithm_id"])
    row["source_binding"] = {
        "stage": source["stage"], "controller": "C0", "task_id": task_id,
        "source_algorithm_id": source["source_algorithm_id"],
        "package_root": package_root,
        "source_package_static_sha256": static_sha,
        "source_package_contract_sha256": contract_sha,
        "shard": owner, "lane": owner,
        "source_provenance_kind": (
            "remainder_source_binding" if continuation else "shard_provenance"),
        "source_provenance_sha256": provenance_sha,
        "evaluated_design_sha256": design_sha,
        "server_startup_sha256": startup_sha,
        "frozen_eval_capacity_sha256": capacity_sha,
    }
    stage_sha = "8" * 64 if continuation else source["quality_cells"][index][
        "evidence"][0]["sha256"]
    row["evidence"].update({
        "stage_manifest": {"path": f"/proof/{owner}/stage.json",
                           "sha256": stage_sha},
        "source_provenance": {"path": f"/proof/{owner}/provenance.json",
                              "sha256": provenance_sha},
        "evaluated_design": {"path": f"/proof/{owner}/design.json",
                             "sha256": design_sha},
        "server_startup": {"path": f"/proof/{owner}/startup.json",
                           "sha256": startup_sha},
        "frozen_eval_capacity": {"path": f"/proof/{owner}/capacity.json",
                                 "sha256": capacity_sha},
        "package_contract": {"path": f"{package_root}/contract.json",
                             "sha256": contract_sha},
        "package_static_files": {"path": f"{package_root}/static_files.json",
                                 "sha256": static_sha},
    })
    value = {
        "schema": terminal.GENERIC_CAPACITY_AUDIT_SCHEMA, "status": "audited",
        "package": {
            "path": package_root, "stage": source["stage"],
            "static_files": {"path": f"{package_root}/static_files.json",
                             "sha256": static_sha},
            "contract": {"path": f"{package_root}/contract.json",
                         "sha256": contract_sha},
        },
        "scope": {
            "automatic_retry_or_rerun": False,
            "fixed_d128_runtime_outcomes": True,
            "official_scores_preserved": True,
            "official_zero_used_as_clean_runtime_completion": False,
            "selection_eligible": True, "unknown_failure_count": 0,
            "audited_capacity_failure_count": 1,
        },
        "failures": [row],
    }
    path = tmp_path / ("generic_continuation.json" if continuation
                       else "generic_direct.json")
    _save(path, value)
    return path


def _overlay(tmp_path: Path, source: dict, index: int, *,
             status: str = "completed") -> Path:
    source_cell = source["quality_cells"][index]
    task_id = source_cell["task_id"]
    stage_path = Path(source_cell["evidence"][0]["path"])
    status_path = stage_path.parent.parent / "run/status.json"
    stage = terminal._read(stage_path)
    source_outcome = next(
        row for row in stage["task_outcomes"] if row["task_id"] == task_id)
    row = {
        "source_shard": f"/frozen/shards/{source_cell['shard_id']}",
        "source_lane": source_cell["shard_id"],
        "task_id": task_id,
        "source_stage_manifest_sha256": source_cell["evidence"][0]["sha256"],
        "source_status_sha256": _sha(status_path),
        "source_outcome": source_outcome,
        "continuation_lane": "C0_part0_remainder",
        "status": status,
        "stage_outcome": {
            "task_id": task_id,
            "outcome": "official_completed" if status == "completed" else status,
            "runtime_completed": status == "completed",
            "worker_returncode": 0,
            "server_returncode": 0 if status == "completed" else 1,
        },
        "official": _official(1) if status == "completed" else (
            _official(0) if status == "runtime_failed" else None),
        "measured_cost": {"generation_attempts": 1},
        "evidence": [{
            "role": "continuation_stage_manifest",
            "path": f"/continuation/{task_id}.json", "sha256": "8" * 64,
        }],
    }
    path = tmp_path / "continuation.json"
    _save(path, {
        "schema": terminal.CONTINUATION_OVERLAY_SCHEMA,
        "package": str(tmp_path / "continuation_package"),
        "status": "completed" if status != "pending" else "partial_pending",
        "continuation_contract_sha256": "7" * 64,
        "total_original_not_started_cells": 1,
        "pending_cells": int(status == "pending"),
        "rows": [row], "automatic_retries": 0, "automatic_reruns": 0,
    })
    return path


def _spec(tmp_path: Path, source: Path, *, audit: Path | None = None,
          overlays: tuple[Path, ...] = ()) -> Path:
    value = {
        "schema": terminal.BUILD_SCHEMA,
        "source_result_receipt": {"path": str(source), "sha256": _sha(source)},
        "capacity_failure_audit": (
            {"path": str(audit), "sha256": _sha(audit)} if audit else None),
        "continuation_overlays": [
            {"path": str(path), "sha256": _sha(path)} for path in overlays],
    }
    path = tmp_path / "spec.json"
    _save(path, value)
    return path


def test_terminal_receipt_keeps_audited_failure_raw_official_out_of_success(
        tmp_path: Path) -> None:
    source_path, _ = _source(tmp_path, failed=(5, 7))
    audit_path = _audit(tmp_path, (5, 7))
    receipt = terminal.build_terminal_receipt(
        _spec(tmp_path, source_path, audit=audit_path))
    assert receipt["status"] == "terminal"
    assert receipt["selection_eligible"] is True
    assert receipt["terminal_task_cells"] == 128
    assert receipt["normal_completed_task_cells"] == 126
    assert receipt["audited_capacity_failure_task_cells"] == 2
    assert receipt["accepted_official_score_task_cells"] == 126
    assert receipt["operational_official_unobserved_task_cells"] == 2
    assert receipt["operational_success_count"] == 62
    assert receipt["operational_non_success_count"] == 66
    for index in (5, 7):
        cell = receipt["quality_cells"][index]
        assert cell["status"] == "audited_capacity_failure"
        assert cell["official"] is None
        assert cell["raw_official"]["correct_count"] == 0
        assert cell["operational_success"] is False
        assert cell["audit_binding"]["sha256"] == _sha(audit_path)

    receipt_path = tmp_path / "terminal.json"
    _save(receipt_path, receipt)
    verified = terminal.verify_terminal_receipt(
        receipt_path, expected_sha256=_sha(receipt_path),
        expected_stage="H0_R1", expected_controller="C0",
        expected_algorithm_id="c0_h0", expected_manifest_sha256="d" * 64,
        expected_task_ids=receipt["task_ids"],
    )
    assert verified["document"] == receipt


def test_real_frozen_v2_capacity_audit_has_exact_selection_contract() -> None:
    path = REPO / (
        "outputs/history_system_search/evidence_sets_v1/expansion/"
        "d128_h0_v2.failure_audit.v2.json")
    assert _sha(path) == "0e0b0ad1e4bd4dcffb14ddef57b9ed4a3b861247b3f3de208d1cbd425a3af4fb"
    audit = terminal._read(path)
    source = {
        "stage": "H0_R1", "controller": "C0",
        "evidence": [{
            "role": "expansion_contract",
            "path": audit["package"]["expansion_contract"]["path"],
            "sha256": audit["package"]["expansion_contract"]["sha256"],
        }],
    }
    rows = terminal._audit_failures(audit, source, _sha(path))
    assert set(rows) == {
        ("C0_part0", "multi_turn_long_context_102"),
        ("C0_part1", "multi_turn_long_context_103"),
    }
    assert all(row["row"]["classification"]["selection_eligible"] is True
               for row in rows.values())

    old_path = REPO / (
        "outputs/history_system_search/evidence_sets_v1/expansion/"
        "d128_h0_v2.failure_audit.json")
    with pytest.raises(ValueError, match="fixed-D128 contract"):
        terminal._audit_failures(terminal._read(old_path), source, _sha(old_path))


def test_real_generic_audit_binds_actual_h0_source_identity() -> None:
    audit_path = REPO / (
        "outputs/history_system_search/evidence_sets_v1/expansion/"
        "d128_h0_v2.failure_audit.generic.v1.json")
    assert _sha(audit_path) == (
        "a0c1e18ed55a3846102dc3eaeba4deec0959b3092a9e20ec8b123563c476a072")
    audit = terminal._read(audit_path)
    source_path = REPO / (
        "outputs/history_system_search/evidence_sets_v1/expansion/"
        "d128_h0_v2_summary/C0.complete_d128.json")
    source = terminal._read(source_path)
    failure = next(
        row for row in audit["failures"]
        if row["controller"] == "C0" and row["shard"] == "C0_part1")
    cell = next(row for row in source["quality_cells"]
                if row["task_id"] == failure["task_id"])
    # The checked-in native summary predates this second immutable failure.  Update
    # only that cell from the frozen audit so the integration test exercises both
    # real C0 failure bindings without claiming the remaining pending cells terminal.
    cell.update(
        status="runtime_failed", status_reason="runtime_failure_in_denominator",
        runtime_completed=False,
        worker_returncode=failure["stage_outcome"]["worker_returncode"],
        server_returncode=failure["stage_outcome"]["server_returncode"],
        official=copy.deepcopy(failure["runtime_vs_official"]["official"]),
        evidence=[{
            "role": "stage_manifest",
            "path": failure["evidence"]["stage_manifest"]["path"],
            "sha256": failure["evidence"]["stage_manifest"]["sha256"],
        }],
    )
    document = terminal._terminal_document(
        source,
        {"path": str(source_path), "sha256": _sha(source_path),
         "schema": terminal.SOURCE_RESULT_SCHEMA},
        [(audit, {"path": str(audit_path), "sha256": _sha(audit_path),
                  "schema": terminal.GENERIC_CAPACITY_AUDIT_SCHEMA})],
        [],
    )
    assert document["audited_capacity_failure_task_cells"] == 2
    assert document["pending_task_cells"] > 0
    assert document["selection_eligible"] is False
    assert {cell["audit_binding"]["schema"] for cell in document["quality_cells"]
            if cell["status"] == "audited_capacity_failure"} == {
                terminal.GENERIC_CAPACITY_AUDIT_SCHEMA}


def test_unknown_failure_and_pending_block_selection(tmp_path: Path) -> None:
    source_path, _ = _source(tmp_path, failed=(5,), pending=(9,))
    receipt = terminal.build_terminal_receipt(_spec(tmp_path, source_path))
    assert receipt["status"] == "incomplete"
    assert receipt["selection_eligible"] is False
    assert receipt["unknown_failure_task_cells"] == 1
    assert receipt["pending_task_cells"] == 1
    assert receipt["terminal_task_cells"] == 126
    assert receipt["operational_non_success_count"] is None


def test_continuation_overlay_replaces_only_exact_original_pending_cell(
        tmp_path: Path) -> None:
    source_path, source = _source(tmp_path, pending=(9,))
    overlay_path = _overlay(tmp_path, source, 9)
    receipt = terminal.build_terminal_receipt(
        _spec(tmp_path, source_path, overlays=(overlay_path,)))
    assert receipt["status"] == "terminal"
    assert receipt["normal_completed_task_cells"] == 128
    cell = receipt["quality_cells"][9]
    assert cell["status_reason"] == "official_completed_by_never_started_continuation"
    assert cell["continuation_overlay_sha256"] == _sha(overlay_path)
    assert cell["operational_success"] is True

    overlay = terminal._read(overlay_path)
    overlay["rows"][0]["task_id"] = source["quality_cells"][8]["task_id"]
    _save(overlay_path, overlay)
    with pytest.raises(ValueError, match="may replace only original pending"):
        terminal.build_terminal_receipt(
            _spec(tmp_path, source_path, overlays=(overlay_path,)))


def test_overlay_runtime_failure_remains_unknown_without_separate_audit(
        tmp_path: Path) -> None:
    source_path, source = _source(tmp_path, pending=(9,))
    overlay_path = _overlay(tmp_path, source, 9, status="runtime_failed")
    receipt = terminal.build_terminal_receipt(
        _spec(tmp_path, source_path, overlays=(overlay_path,)))
    assert receipt["selection_eligible"] is False
    assert receipt["unknown_failure_task_cells"] == 1
    assert receipt["quality_cells"][9]["official"] is None


def test_generic_audit_supports_h1_without_allowing_h0_audit_alias(
        tmp_path: Path) -> None:
    source_path, source = _source(tmp_path, failed=(5,))
    source["stage"] = "H1_R1"
    source["package_kind"] = "H1_R3_full128"
    source["evidence"] = [
        {"role": "combination_contract", "path": "/frozen/d128_h1_v1/contract.json",
         "sha256": "e" * 64},
        {"role": "shard_provenance", "path": "/frozen/C0_part0/provenance.json",
         "sha256": "5" * 64},
        {"role": "evaluated_design", "path": "/frozen/C0_part0/design.json",
         "sha256": "4" * 64},
    ]
    _save(source_path, source)
    h0_audit = _audit(tmp_path, (5,))
    with pytest.raises(ValueError, match="H0 capacity-failure audit"):
        terminal.build_terminal_receipt(
            _spec(tmp_path, source_path, audit=h0_audit))

    generic_audit = _generic_audit(tmp_path, source, 5)
    receipt = terminal.build_terminal_receipt(
        _spec(tmp_path, source_path, audit=generic_audit))
    assert receipt["selection_eligible"] is True
    assert receipt["audited_capacity_failure_task_cells"] == 1
    assert receipt["capacity_failure_audits"][0]["schema"] == (
        terminal.GENERIC_CAPACITY_AUDIT_SCHEMA)


def test_generic_audit_can_classify_exact_continuation_failure(
        tmp_path: Path) -> None:
    source_path, source = _source(tmp_path, pending=(9,))
    overlay_path = _overlay(tmp_path, source, 9, status="runtime_failed")
    audit_path = _generic_audit(tmp_path, source, 9, continuation=True)
    receipt = terminal.build_terminal_receipt(
        _spec(tmp_path, source_path, audit=audit_path,
              overlays=(overlay_path,)))
    assert receipt["selection_eligible"] is True
    assert receipt["normal_completed_task_cells"] == 127
    assert receipt["audited_capacity_failure_task_cells"] == 1
    cell = receipt["quality_cells"][9]
    assert cell["status"] == "audited_capacity_failure"
    assert cell["continuation_package"] == str(tmp_path / "continuation_package")
    assert cell["official"] is None
    assert cell["raw_official"]["correct_count"] == 0


def test_audit_cannot_reclassify_normal_cell_or_drift_raw_official(
        tmp_path: Path) -> None:
    source_path, _ = _source(tmp_path, failed=(5,))
    audit_path = _audit(tmp_path, (5, 7))
    with pytest.raises(ValueError, match="may classify only matching unknown failures"):
        terminal.build_terminal_receipt(
            _spec(tmp_path, source_path, audit=audit_path))

    audit = terminal._read(audit_path)
    audit["failures"] = audit["failures"][:1]
    audit["scope"]["audited_capacity_failure_count"] = 1
    audit["failures"][0]["runtime_vs_official"]["official"]["semantic_score"] = 1.0
    _save(audit_path, audit)
    with pytest.raises(ValueError, match="raw official differs"):
        terminal.build_terminal_receipt(
            _spec(tmp_path, source_path, audit=audit_path))


def test_verifier_rejects_zero_imputation_count_drift_and_input_drift(
        tmp_path: Path) -> None:
    source_path, _ = _source(tmp_path, failed=(5,))
    audit_path = _audit(tmp_path, (5,))
    receipt = terminal.build_terminal_receipt(
        _spec(tmp_path, source_path, audit=audit_path))
    receipt_path = tmp_path / "terminal.json"
    _save(receipt_path, receipt)

    mutated = copy.deepcopy(receipt)
    mutated["quality_cells"][5]["official"] = _official(0)
    _save(receipt_path, mutated)
    with pytest.raises(ValueError, match="Invalid audited capacity failure"):
        terminal.verify_terminal_receipt(receipt_path, verify_inputs=False)

    mutated = copy.deepcopy(receipt)
    mutated["operational_success_count"] += 1
    _save(receipt_path, mutated)
    with pytest.raises(ValueError, match="counts differ"):
        terminal.verify_terminal_receipt(receipt_path, verify_inputs=False)

    _save(receipt_path, receipt)
    source = terminal._read(source_path)
    source["quality_cells"][1]["official"]["correct_count"] = 0
    _save(source_path, source)
    with pytest.raises(ValueError, match="exact hash binding"):
        terminal.verify_terminal_receipt(receipt_path, verify_inputs=True)
