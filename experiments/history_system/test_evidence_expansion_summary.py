from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import evidence_eval_combinations as combinations
import evidence_expansion_summary as summary


def _save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official(index: int) -> dict:
    return {
        "benchmark": "bfcl",
        "categories": "multi_turn_base",
        "mode": "both",
        "n_total": 1,
        "n_scored": 1,
        "correct_count": index % 2,
        "semantic_score": float(index % 2),
        "scored": True,
        "total_gold_checker_seconds": 0.25,
        "total_handler_http_calls": 2,
    }


def _raw_shard(root: Path, *, shard_id: str, controller: str, stage: str,
               tasks: list[str], algorithm_id: str) -> dict:
    lane = root / "lanes" / shard_id
    _save(root / "tasks.json", {"task_ids": tasks})
    _save(lane / "design.json", {"candidate_id": algorithm_id, "task_ids": tasks})
    _save(lane / "runtime" / "configs" / "controller.json", {
        "gp_experiments": {"controller": controller, "stage": stage}
    })
    provenance = {
        "controller": controller,
        "source_algorithm_id": algorithm_id if stage == "H0_R1" else f"source_{controller}",
        "source_design_sha256": "a" * 64,
        "source_controller_sha256": _sha(
            lane / "runtime" / "configs" / "controller.json"),
    }
    if stage != "H0_R1":
        provenance.update({"stage": stage, "output_algorithm_id": algorithm_id})
    _save(root / "provenance.json", provenance)
    outcomes = []
    for index, task_id in enumerate(tasks):
        task = lane / "results" / "task_shards" / task_id
        _save(task / "server" / "final.json", {
            "wall_seconds": float(index + 1),
            "cost_summary": {
                "generation_attempts": index + 2,
                "costs": {
                    "openai_resident_usage": {
                        "prompt_tokens": {"strict_total": 100 + index},
                        "completion_tokens": {"strict_total": 10 + index},
                        "total_tokens": {"strict_total": 110 + 2 * index},
                    },
                    "actual_model_work": {
                        "materialized_encoder_tokens": {"strict_total": 20 + index}
                    },
                },
            },
        })
        _save(task / "bfcl" / "official_summary.json", _official(index))
        outcomes.append({
            "task_id": task_id,
            "outcome": "official_completed",
            "runtime_completed": True,
            "worker_returncode": 0,
            "server_returncode": 0,
        })
    _save(lane / "results" / "stage_manifest.json", {
        "status": "completed_fixed_manifest",
        "state": "completed",
        "task_outcomes": outcomes,
    })
    return {
        "stage": stage,
        "controller": controller,
        "shard_id": shard_id,
        "relative_package": shard_id,
        "task_budget": len(tasks),
    }


def _expansion_package(tmp_path: Path) -> tuple[Path, list[str], str]:
    package = tmp_path / "expansion"
    full = [f"multi_turn_base_{index}" for index in range(128)]
    reused = full[:20]
    new = full[20:]
    manifest_sha = "c" * 64
    audit_path = package / "d20_compatibility_audit.json"
    _save(audit_path, {
        "schema": summary.EXPANSION_AUDIT_SCHEMA,
        "conclusion": {
            "status": "semantic_nonactivation_compatible",
            "result_reuse_supported": True,
        },
        "lanes": {
            "C0": {
                "classification": "semantic_nonactivation_compatible_20_of_20",
                "tasks": [{"task_id": task_id} for task_id in reused],
            }
        },
    })
    audit_sha = _sha(audit_path)
    lanes = {}
    manifests = {}
    for controller in ("C0", "C1"):
        lanes[controller] = {
            "quality_cells": [
                {
                    "task_id": task_id,
                    "quality_source_id": f"owner_{controller}_{index}",
                    "status": "completed",
                    "status_reason": None,
                    "official": _official(index),
                    "measured_cost": {"generation_attempts": index + 1},
                }
                for index, task_id in enumerate(reused)
            ]
        }
        manifests[controller] = {
            "reused_task_ids": reused,
            "task_ids": new,
            "full_task_ids": full,
        }
    readiness_path = package / "readiness.json"
    _save(readiness_path, {
        "schema": summary.EXPANSION_READINESS_SCHEMA,
        "promotion": {"selected": ["C0", "C1"]},
        "next_manifests": manifests,
        "lanes": lanes,
        "reuse_validation": {"audited_lanes": ["C0", "C3", "C5"]},
        "inputs": {
            "d128": {"sha256": manifest_sha},
            "reuse_audit": {"sha256": audit_sha},
        },
    })
    bindings_path = package / "source_bindings.json"
    _save(bindings_path, {
        "schema": summary.EXPANSION_BINDINGS_SCHEMA,
        "reuse_audit": {"sha256": audit_sha},
        "bindings": {
            controller: {
                "controller": controller,
                "source_algorithm_id": f"algo_{controller}",
                "source_design_sha256": "d" * 64,
                "source_controller_sha256": "e" * 64,
            }
            for controller in ("C0", "C1")
        },
    })
    rows = []
    for controller in ("C0", "C1"):
        for part in range(3):
            shard_id = f"{controller}_part{part}"
            rows.append(_raw_shard(
                package / "shards" / shard_id,
                shard_id=shard_id, controller=controller, stage="H0_R1",
                tasks=new[part::3], algorithm_id=f"algo_{controller}",
            ))
    bindings = json.loads(bindings_path.read_text())
    for controller in ("C0", "C1"):
        controller_path = (
            package / "shards" / f"{controller}_part0" / "lanes"
            / f"{controller}_part0" / "runtime" / "configs" / "controller.json"
        )
        bindings["bindings"][controller]["source_controller_sha256"] = _sha(controller_path)
    _save(bindings_path, bindings)
    _save(package / "expansion_contract.json", {
        "schema": summary.EXPANSION_PACKAGE_SCHEMA,
        "selected_controllers": ["C0", "C1"],
        "d20_compatibility_audit_sha256": audit_sha,
        "shards": rows,
    })
    return package, full, manifest_sha


def _combination_package(tmp_path: Path) -> tuple[Path, list[str]]:
    package = tmp_path / "combinations"
    tasks = [f"multi_turn_base_{index}" for index in range(128)]
    _save(package / "tasks.d128.json", {"task_ids": tasks})
    rows = []
    for stage, controller, algorithm in (
        ("H1_R1", "C0", "algo_C0__h1_r1"),
        ("R3", "C0", "algo_C0__h1_r1__r3"),
    ):
        prefix = "h1_C0" if stage == "H1_R1" else "r3_C0"
        shard_count = 3 if stage == "H1_R1" else 6
        for part in range(shard_count):
            shard_id = f"{prefix}_part{part}"
            rows.append(_raw_shard(
                package / "shards" / shard_id,
                shard_id=shard_id, controller=controller, stage=stage,
                tasks=tasks[part::shard_count], algorithm_id=algorithm,
            ))
    _save(package / "combination_contract.json", {
        "schema": summary.COMBINATION_PACKAGE_SCHEMA,
        "d128_manifest_sha256": _sha(package / "tasks.d128.json"),
        "shards": rows,
    })
    return package, tasks


def test_h0_aggregates_historical20_and_raw108_in_d128_order(tmp_path: Path) -> None:
    package, tasks, manifest_sha = _expansion_package(tmp_path)
    output = tmp_path / "receipts"
    index = summary.write_receipts(package, output)
    assert len(index["receipts"]) == 2
    receipt = json.loads((output / "C0.complete_d128.json").read_text())
    assert receipt["schema"] == combinations.COMPLETE_RESULT_SCHEMA
    assert receipt["status"] == "completed"
    assert receipt["source_algorithm_id"] == "algo_C0"
    assert receipt["task_manifest_sha256"] == manifest_sha
    assert receipt["completed_task_cells"] == 128
    assert receipt["historical20"]["completed_task_cells"] == 20
    assert receipt["new108"]["completed_task_cells"] == 108
    assert [row["task_id"] for row in receipt["quality_cells"]] == tasks
    assert receipt["quality_cells"][0]["cohort"] == "historical20"
    assert receipt["quality_cells"][20]["cohort"] == "new108"
    assert receipt["configuration_binding"]["status"] == "audited_semantic_compatibility"
    assert json.loads((output / "C1.complete_d128.json").read_text())[
        "configuration_binding"]["status"] == "exact_frozen_config"
    verified = combinations._verify_complete_result(
        {"complete_d128_receipt": {
            "path": str(output / "C0.complete_d128.json"),
            "sha256": _sha(output / "C0.complete_d128.json"),
        }},
        "C0", "algo_C0", {"task_ids": tasks}, manifest_sha,
    )
    assert verified["document"]["completed_task_cells"] == 128


def test_failed_raw_task_remains_incomplete_with_real_failure(tmp_path: Path) -> None:
    package, _, _ = _expansion_package(tmp_path)
    stage_path = package / "shards" / "C0_part0" / "lanes" / "C0_part0" / "results" / "stage_manifest.json"
    stage = json.loads(stage_path.read_text())
    stage["task_outcomes"][0].update({
        "outcome": "runtime_failure_in_denominator",
        "runtime_completed": False,
        "server_returncode": 1,
    })
    _save(stage_path, stage)
    receipt = summary.summarize_package(package)["C0"]
    assert receipt["status"] == "incomplete"
    assert receipt["completed_task_cells"] == 127
    assert receipt["runtime_failed_task_cells"] == 1
    failed = [row for row in receipt["quality_cells"] if row["status"] == "runtime_failed"]
    assert len(failed) == 1
    assert failed[0]["server_returncode"] == 1
    assert "runtime_failure_in_denominator" in failed[0]["status_reason"]


def test_duplicate_shard_task_is_rejected_instead_of_hidden_by_dedup(tmp_path: Path) -> None:
    package, _, _ = _expansion_package(tmp_path)
    path = package / "shards" / "C0_part0" / "tasks.json"
    value = json.loads(path.read_text())
    value["task_ids"][-1] = value["task_ids"][0]
    _save(path, value)
    with pytest.raises(ValueError, match="36 unique task IDs"):
        summary.summarize_package(package)


def test_h1_three_way_and_r3_six_way_aggregate_without_d20_reuse(tmp_path: Path) -> None:
    package, tasks = _combination_package(tmp_path)
    receipts = summary.summarize_package(package)
    assert set(receipts) == {"H1_R1__C0", "R3__C0"}
    assert receipts["H1_R1__C0"]["source_algorithm_id"] == "algo_C0__h1_r1"
    assert receipts["R3__C0"]["source_algorithm_id"] == "algo_C0__h1_r1__r3"
    for receipt in receipts.values():
        assert receipt["status"] == "completed"
        assert receipt["historical20"] is None
        assert receipt["new108"] is None
        assert [row["task_id"] for row in receipt["quality_cells"]] == tasks
        assert {row["cohort"] for row in receipt["quality_cells"]} == {"full128"}
        assert all(row["quality_source_id"] == receipt["source_algorithm_id"]
                   for row in receipt["quality_cells"])


def test_invalid_official_summary_cannot_be_clean_completed128(tmp_path: Path) -> None:
    package, _ = _combination_package(tmp_path)
    official = (
        package / "shards" / "h1_C0_part0" / "lanes" / "h1_C0_part0"
        / "results" / "task_shards" / "multi_turn_base_0"
        / "bfcl" / "official_summary.json"
    )
    value = json.loads(official.read_text())
    value["n_scored"] = 0
    _save(official, value)
    receipt = summary.summarize_package(package)["H1_R1__C0"]
    assert receipt["status"] == "incomplete"
    assert receipt["completed_task_cells"] == 127
    cell = receipt["quality_cells"][0]
    assert cell["status"] == "runtime_failed"
    assert "official_summary_missing_or_invalid" in cell["status_reason"]
