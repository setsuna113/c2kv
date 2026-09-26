from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

import evidence_eval_combinations as combinations
import evidence_eval_expansion as expansion
import evidence_r3_selection as selection
import evidence_terminal_results as terminal_results


HERE = Path(__file__).resolve().parent
TASKS = [f"multi_turn_base_{index}" for index in range(128)]
CONFIGS = (("H0_R1", "C0"), ("H0_R1", "C5"),
           ("H1_R1", "C0"), ("H1_R1", "C5"))


def _save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_source(root: Path) -> Path:
    source = root / "prepared_v8"
    target = source / "history_system/evidence_sets.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(HERE / "evidence_sets.py", target)
    for relative in combinations.G04_SOURCE_HASHES:
        destination = source / "history_system/runtime" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / "runtime" / relative, destination)
    return source


def _sglang_manifest(source: Path) -> None:
    version = source / "sglang/version.py"
    version.parent.mkdir(parents=True, exist_ok=True)
    version.write_text("VERSION = 'frozen'\n", encoding="utf-8")
    _save(source / "sglang_files.json", {
        "schema": "synthetic-sglang-v1",
        "file_count": 1,
        "files": {"sglang/version.py": _sha(version)},
    })


def _lane(source: Path, lane: str, controller: str, history: str,
          algorithm: str, tasks: list[str]) -> tuple[Path, Path]:
    runtime = source / "lanes" / lane / "runtime"
    for relative in combinations.G04_SOURCE_HASHES:
        destination = runtime / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / "runtime" / relative, destination)
    gp = {
        "G": history,
        "R": 1,
        "set_selector": "candidate_rule" if controller == "C0" else "parameter_source",
    }
    controller_config = {"schema": "controller-v1", "gp_experiments": gp}
    controller_path = runtime / "configs/controller.json"
    _save(controller_path, controller_config)
    _save(runtime / "configs/eval_policy.json", {"policy": "frozen"})
    design = {
        "schema": "a-history-system-candidate-design-v1",
        "status": "frozen",
        "candidate_id": algorithm,
        "run_id_template": lane,
        "launch_authorized": True,
        "automatic_reruns": 0,
        "task_ids": tasks,
        "task_manifest_sha256": "a" * 64,
        "limits": {"tasks": len(tasks), "history_tokens": 768},
        "sampling": dict(combinations.STRICT_SAMPLING),
        "checkpoint_selection": {"path": "/checkpoint/frozen", "step": 1000},
        "runtime": {"sglang_backend_url": "http://127.0.0.1:39000",
                    "dtype": "bfloat16", "device": "npu"},
        "resolved_configs": {"controller": controller_config,
                             "eval_policy": {"policy": "frozen"}},
        "search_contract": {"history": "H0" if history == "current" else "H1",
                            "recovery_attempts": 1,
                            "fixed_denominator": "D128"},
        "source_files": expansion._tree_hashes(runtime),
    }
    design_path = source / "lanes" / lane / "design.json"
    _save(design_path, design)
    _save(source / "lanes" / lane / "lane.json", {
        "name": lane, "physical_device": 0, "engine_port": 39000,
        "task_port_base": 40000,
        "task_ports": list(range(40000, 40000 + len(tasks))),
    })
    return design_path, controller_path


def _source_shell(source: Path) -> None:
    source.mkdir(parents=True)
    shutil.copyfile(HERE / "evidence_eval.py", source / "evidence_eval.py")
    runner = source / "history_system/runner.py"
    runner.parent.mkdir()
    runner.write_text("# frozen runner\n", encoding="utf-8")
    _save(source / "launch_contract.json", {"required_environment": {}})
    _sglang_manifest(source)


def _freeze_static(source: Path) -> str:
    _save(source / "static_files.json", expansion._manifest(
        source, "synthetic-static-v1", skip_sglang=True,
        excluded=("static_files.json",)))
    return _sha(source / "static_files.json")


def _steps(path: Path, task_id: str, extra: int) -> dict[str, str]:
    trace = [{"phase": "draft", "status": "completed"}]
    trace.extend({"phase": "regeneration", "status": "completed"}
                 for _ in range(extra))
    record = {
        "status": "ok",
        "session_id": task_id,
        "decision_key": "turn0/step0",
        "generation_trace": trace,
        "recovery_checks": [],
        "controller_timing": {"reconsider_seconds": 1.0},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return {"role": "steps_jsonl", "path": str(path), "sha256": _sha(path)}


def _receipt(path: Path, stage: str, controller: str, algorithm: str,
             manifest_sha: str, successful: set[str], *,
             exact_cost: int | None, design_evidence: dict[str, str] | None) -> None:
    cells = []
    for index, task_id in enumerate(TASKS):
        evidence = [{"path": str(path.parent / "missing" / task_id),
                     "sha256": "d" * 64}]
        if exact_cost is not None:
            evidence = [_steps(
                path.parent / "steps" / stage / controller / task_id / "steps.jsonl",
                task_id, 1 if index < exact_cost else 0)]
        cells.append({
            "task_id": task_id,
            "cohort": "full128" if stage == "H1_R1" else (
                "historical20" if index < 20 else "new108"),
            "quality_source_id": algorithm,
            "shard_id": None,
            "status": "completed",
            "status_reason": "official_completed",
            "runtime_completed": True,
            "worker_returncode": 0,
            "server_returncode": 0,
            "official": {"scored": True, "n_total": 1, "n_scored": 1,
                         "correct_count": int(task_id in successful),
                         "semantic_score": float(task_id in successful)},
            "measured_cost": {},
            "evidence": evidence,
        })
    top_evidence = ([design_evidence] if design_evidence is not None else
                    [{"path": str(path.parent / "quality.json"),
                      "sha256": "b" * 64}])
    _save(path, {
        "schema": combinations.COMPLETE_RESULT_SCHEMA,
        "package_kind": "synthetic",
        "stage": stage,
        "status": "completed",
        "quality_label": "preliminary, n=1",
        "controller": controller,
        "source_algorithm_id": algorithm,
        "task_manifest_sha256": manifest_sha,
        "expected_task_cells": 128,
        "completed_task_cells": 128,
        "runtime_failed_task_cells": 0,
        "pending_task_cells": 0,
        "configuration_binding": {"status": "exact_frozen_config"},
        "evidence": top_evidence,
        "quality_cells": cells,
    })


def _case(tmp_path: Path, successes: dict[tuple[str, str], set[str]], *,
          extras: dict[tuple[str, str], int | None] | None = None) -> dict[str, Path]:
    extras = extras or {}
    manifest = tmp_path / "D128.json"
    _save(manifest, {"schema": "task-manifest-v1", "task_ids": TASKS})
    manifest_sha = _sha(manifest)
    config_source = _config_source(tmp_path)

    h0 = tmp_path / "h0_source"
    _source_shell(h0)
    h0_rows = {}
    for controller in ("C0", "C5"):
        algorithm = f"algo_{controller}_current"
        design, controller_path = _lane(
            h0, controller, controller, "current", algorithm, TASKS)
        h0_rows[controller] = (algorithm, design, controller_path)
    h0_static = _freeze_static(h0)
    h0_sglang = _sha(h0 / "sglang_files.json")

    catalog_bindings = {}
    for controller, (algorithm, design, controller_path) in h0_rows.items():
        catalog_bindings[controller] = {
            "controller": controller,
            "source_id": "synthetic_h0",
            "source_package": str(h0),
            "source_lane": controller,
            "source_algorithm_id": algorithm,
            "source_static_files_sha256": h0_static,
            "source_sglang_files_sha256": h0_sglang,
            "source_design_sha256": _sha(design),
            "source_controller_sha256": _sha(controller_path),
            "required_env": {},
            "model_artifacts": [],
        }
    catalog = tmp_path / "source_catalog.json"
    _save(catalog, {"schema": selection.SOURCE_CATALOG_SCHEMA,
                    "bindings": catalog_bindings})

    h1_sources = {}
    for controller in ("C0", "C5"):
        source = tmp_path / f"h1_{controller}_part0_source"
        _source_shell(source)
        lane = f"h1_{controller}_part0"
        algorithm = f"algo_{controller}_current__h1_r1"
        design, _ = _lane(source, lane, controller, "record_bound", algorithm,
                          TASKS[0::3])
        _save(source / "provenance.json", {
            "schema": "experiment3-combination-shard-provenance-v1",
            "controller": controller,
            "source_algorithm_id": f"algo_{controller}_current",
            "output_algorithm_id": algorithm,
            "copied_model_artifacts": [],
        })
        _freeze_static(source)
        h1_sources[controller] = (source, lane, algorithm, design)

    receipt_bindings = []
    receipts = {}
    for stage, controller in CONFIGS:
        if stage == "H0_R1":
            algorithm = h0_rows[controller][0]
            design_evidence = None
        else:
            algorithm = h1_sources[controller][2]
            design = h1_sources[controller][3]
            design_evidence = {"role": "evaluated_design", "path": str(design),
                               "sha256": _sha(design)}
        receipt = tmp_path / "receipts" / f"{stage}__{controller}.json"
        _receipt(receipt, stage, controller, algorithm, manifest_sha,
                 successes[(stage, controller)],
                 exact_cost=extras.get((stage, controller)),
                 design_evidence=design_evidence)
        receipts[(stage, controller)] = receipt
        receipt_bindings.append({"stage": stage, "controller": controller,
                                 "path": str(receipt), "sha256": _sha(receipt)})
    input_path = tmp_path / "selection_input.json"
    _save(input_path, {
        "schema": selection.INPUT_SCHEMA,
        "d128_manifest": {"path": str(manifest), "sha256": manifest_sha},
        "h0_source_catalog": {"path": str(catalog), "sha256": _sha(catalog)},
        "config_source_package": {
            "path": str(config_source),
            "source_evidence_sets_sha256": _sha(
                config_source / "history_system/evidence_sets.py"),
        },
        "complete_d128_receipts": receipt_bindings,
    })
    return {"input": input_path, "manifest": manifest,
            "catalog": catalog, "config_source": config_source,
            "receipts": receipts}


def _first(count: int, offset: int = 0) -> set[str]:
    return set(TASKS[offset:offset + count])


def _terminalize(case: dict[str, Path]) -> None:
    selection_input = json.loads(case["input"].read_text(encoding="utf-8"))
    terminal_bindings = []
    for binding in selection_input.pop("complete_d128_receipts"):
        source = Path(binding["path"])
        spec = source.with_name(source.stem + ".terminal_build.json")
        terminal_path = source.with_name(source.stem + ".terminal.json")
        _save(spec, {
            "schema": terminal_results.BUILD_SCHEMA,
            "source_result_receipt": {
                "path": str(source), "sha256": _sha(source),
                "schema": terminal_results.SOURCE_RESULT_SCHEMA,
            },
            "capacity_failure_audit": None,
            "continuation_overlays": [],
        })
        _save(terminal_path, terminal_results.build_terminal_receipt(spec))
        terminal_bindings.append({
            "stage": binding["stage"], "controller": binding["controller"],
            "path": str(terminal_path), "sha256": _sha(terminal_path),
        })
    selection_input["terminal_d128_receipts"] = terminal_bindings
    _save(case["input"], selection_input)


def test_unique_success_leader_emits_r3_only_spec(tmp_path: Path) -> None:
    case = _case(tmp_path, {
        ("H0_R1", "C0"): _first(9),
        ("H0_R1", "C5"): _first(7),
        ("H1_R1", "C0"): _first(8),
        ("H1_R1", "C5"): _first(6),
    })
    receipt_path = tmp_path / "leading.json"
    spec_path = tmp_path / "r3.json"
    result = selection.write_outputs(case["input"], receipt_path, spec_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert result["status"] == "selected_after_complete_d128"
    assert receipt["selected_config_id"] == "H0_R1__C0"
    assert receipt["decision"]["cost_tiebreak_used"] is False
    assert spec["h1_sources"] == []
    assert spec["r3"]["source"]["source_lane"] == "C0"
    assert spec["r3"]["leading_receipt"]["sha256"] == _sha(receipt_path)


def test_same_passset_uses_exact_extra_generation_tiebreak(tmp_path: Path) -> None:
    tied = _first(10)
    case = _case(tmp_path, {
        ("H0_R1", "C0"): tied,
        ("H0_R1", "C5"): tied,
        ("H1_R1", "C0"): _first(8),
        ("H1_R1", "C5"): _first(7),
    }, extras={("H0_R1", "C0"): 4, ("H0_R1", "C5"): 1})
    receipt, spec = selection.evaluate(case["input"])
    assert receipt["selected_config_id"] == "H0_R1__C5"
    assert receipt["decision"]["reason"] == (
        "identical_success_set_fewer_exact_extra_generations")
    assert spec is not None
    costs = {row["config_id"]: row["cost"]["extra_generations"]
             for row in receipt["candidates"] if "cost" in row}
    assert costs == {"H0_R1__C0": 4, "H0_R1__C5": 1}
    assert all(row["cost"]["covered_tasks"] == 128
               for row in receipt["candidates"] if "cost" in row)


def test_complementary_success_tie_is_unresolved_without_cost_lookup(tmp_path: Path) -> None:
    case = _case(tmp_path, {
        ("H0_R1", "C0"): _first(10),
        ("H0_R1", "C5"): _first(10, 10),
        ("H1_R1", "C0"): _first(8),
        ("H1_R1", "C5"): _first(7),
    })
    receipt_path = tmp_path / "leading.json"
    spec_path = tmp_path / "r3.json"
    result = selection.write_outputs(case["input"], receipt_path, spec_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert result["status"] == "unresolved_after_complete_d128"
    assert receipt["decision"]["reason"] == (
        "complementary_success_sets_tied_on_success_count")
    assert not spec_path.exists()


def test_missing_cost_is_unresolved_and_incomplete_receipt_is_rejected(tmp_path: Path) -> None:
    tied = _first(10)
    case = _case(tmp_path, {
        ("H0_R1", "C0"): tied,
        ("H0_R1", "C5"): tied,
        ("H1_R1", "C0"): _first(8),
        ("H1_R1", "C5"): _first(7),
    })
    receipt, spec = selection.evaluate(case["input"])
    assert receipt["status"] == "unresolved_after_complete_d128"
    assert receipt["decision"]["reason"] == (
        "missing_or_invalid_exact_full128_extra_generation_cost")
    assert spec is None

    broken_path = case["receipts"][("H1_R1", "C5")]
    broken = json.loads(broken_path.read_text(encoding="utf-8"))
    broken["status"] = "incomplete"
    _save(broken_path, broken)
    input_value = json.loads(case["input"].read_text(encoding="utf-8"))
    row = next(item for item in input_value["complete_d128_receipts"]
               if item["stage"] == "H1_R1" and item["controller"] == "C5")
    row["sha256"] = _sha(broken_path)
    _save(case["input"], input_value)
    with pytest.raises(ValueError, match="incomplete or malformed"):
        selection.evaluate(case["input"])


def test_exact_extra_tie_without_isolated_selector_cost_is_unresolved(tmp_path: Path) -> None:
    tied = _first(10)
    case = _case(tmp_path, {
        ("H0_R1", "C0"): tied,
        ("H0_R1", "C5"): tied,
        ("H1_R1", "C0"): _first(8),
        ("H1_R1", "C5"): _first(7),
    }, extras={("H0_R1", "C0"): 3, ("H0_R1", "C5"): 3})
    receipt, spec = selection.evaluate(case["input"])
    assert receipt["status"] == "unresolved_after_complete_d128"
    assert receipt["decision"]["reason"] == (
        "exact_extra_generation_tie_without_isolated_selector_cost")
    assert receipt["decision"]["selector_cost_used"] is False
    assert spec is None


def test_h1_leader_preserves_record_bound_and_r3_changes_only_r(tmp_path: Path) -> None:
    case = _case(tmp_path, {
        ("H0_R1", "C0"): _first(7),
        ("H0_R1", "C5"): _first(8),
        ("H1_R1", "C0"): _first(9),
        ("H1_R1", "C5"): _first(11),
    })
    receipt_path = tmp_path / "leading.json"
    spec_path = tmp_path / "r3.json"
    selection.write_outputs(case["input"], receipt_path, spec_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["selected_config_id"] == "H1_R1__C5"
    assert receipt["r3_transition"] == {
        "source_history": "record_bound",
        "source_recovery_rounds": 1,
        "target_history": "record_bound",
        "target_recovery_rounds": 3,
        "changed_controller_fields": ["R"],
    }
    package = tmp_path / "r3_package"
    combinations.prepare(package, spec_path)
    design = json.loads((package / "shards/r3_leading_part0/lanes/"
                         "r3_leading_part0/design.json").read_text(encoding="utf-8"))
    gp = design["resolved_configs"]["controller"]["gp_experiments"]
    assert gp["G"] == "record_bound"
    assert gp["R"] == 3


def test_terminal_receipts_select_and_build_with_terminal_gate(tmp_path: Path) -> None:
    case = _case(tmp_path, {
        ("H0_R1", "C0"): _first(7),
        ("H0_R1", "C5"): _first(8),
        ("H1_R1", "C0"): _first(9),
        ("H1_R1", "C5"): _first(11),
    })
    _terminalize(case)
    receipt_path = tmp_path / "leading_terminal.json"
    spec_path = tmp_path / "r3_terminal.json"
    selection.write_outputs(case["input"], receipt_path, spec_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == combinations.TERMINAL_LEADING_SCHEMA
    assert receipt["status"] == "selected_after_terminal_d128"
    assert receipt["selected_config_id"] == "H1_R1__C5"
    assert receipt["selection_rule"]["primary"] == (
        "fixed_d128_operational_success_count")
    assert receipt["decision"]["reason"] == (
        "unique_highest_fixed_d128_operational_success_count")
    assert all("operational_success_count" in row
               and "correct_count" not in row
               for row in receipt["candidates"])
    assert "terminal_d128_receipt_sha256" in receipt
    assert "complete_d128_receipt_sha256" not in receipt
    source = spec["r3"]["source"]
    assert "terminal_d128_receipt" in source
    assert "complete_d128_receipt" not in source
    package = tmp_path / "r3_terminal_package"
    combinations.prepare(package, spec_path)
    contract = json.loads((package / "combination_contract.json").read_text(
        encoding="utf-8"))
    assert contract["r3_terminal_d128_receipt_sha256"] == (
        receipt["terminal_d128_receipt_sha256"])


def test_incomplete_terminal_receipt_blocks_selection(tmp_path: Path) -> None:
    case = _case(tmp_path, {
        ("H0_R1", "C0"): _first(7),
        ("H0_R1", "C5"): _first(8),
        ("H1_R1", "C0"): _first(9),
        ("H1_R1", "C5"): _first(11),
    })
    source = case["receipts"][("H1_R1", "C5")]
    value = json.loads(source.read_text(encoding="utf-8"))
    value["quality_cells"][-1].update({
        "status": "pending", "status_reason": "not_started",
        "runtime_completed": False, "worker_returncode": None,
        "server_returncode": None, "official": None,
    })
    _save(source, value)
    _terminalize(case)
    with pytest.raises(ValueError, match="ineligible terminal D128 receipt"):
        selection.evaluate(case["input"])
