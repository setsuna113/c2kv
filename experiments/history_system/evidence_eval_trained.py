"""Evaluate each real T02 artifact once after training releases its device."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import evidence_eval as base

LANES = {
    "C1": {"selector": "risk", "physical_device": 4, "engine_port": 37440,
           "task_port_base": 37500, "model_smokes": ["embedding"],
           "artifact": "training/c1_risk.json", "kind": "c1_risk_logistic"},
    "C4_turn": {"selector": "gain_turn", "physical_device": 6, "engine_port": 37460,
                "task_port_base": 37520, "model_smokes": ["embedding", "reranker"],
                "artifact": "training/c4/c4_gain_turn.json", "kind": "c4_gain_turn"},
    "C4_task": {"selector": "gain_task", "physical_device": 0, "engine_port": 37400,
                "task_port_base": 37540, "model_smokes": ["embedding", "reranker"],
                "artifact": "training/c4/c4_gain_task.json", "kind": "c4_gain_task"},
}

EXPECTED_TRAINING_PACKAGE = "prepared_v8"
EXPECTED_TRAINING_STATE_COUNT = 118
EXPECTED_PRESERVED_STATE_COUNT = 10
EXPECTED_NEW_STATE_COUNT = 108
EXPECTED_STREAMING_TASK_COUNT = 83
EXPECTED_NEW_BRANCH_COUNT = 324
EXPECTED_CUMULATIVE_BRANCH_COUNT = 359
EXPECTED_TASK_GROUP_COUNT = 26
EXPECTED_SPLIT_COUNTS = {"calibration": 38, "train": 80}
TRAINING_BINDING_SCHEMA = "post-t02-streaming-training-binding-v2"
TRAINING_BINDING_FILES = (
    "source_files.json",
    "contract.json",
    "recovery/recovery_contract.json",
    "recovery/source_plan.json",
)


class TrainingPending(RuntimeError):
    """The frozen v8 run is valid so far but this lane is not trained yet."""


def lane_specs():
    return [{"name": name, **{k: v for k, v in row.items() if k not in {"artifact", "kind"}},
             "task_ports": list(range(row["task_port_base"], row["task_port_base"] + 20))}
            for name, row in LANES.items()]


def bind_base(package):
    base.LANES = LANES
    base.lane_specs = lane_specs
    base.DEFAULT_REMOTE_ROOT = package


def _canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _valid_sha256(value) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _bound_path(training: Path, value, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Missing {label} path")
    path = Path(value)
    path = path if path.is_absolute() else training / path
    path = path.resolve()
    try:
        path.relative_to(training)
    except ValueError as error:
        raise ValueError(f"{label} path escapes prepared_v8") from error
    return path


def _verify_file_sha256(path: Path, expected, label: str) -> None:
    if not _valid_sha256(expected) or not path.is_file() or base._sha(path) != expected:
        raise ValueError(f"{label} file differs from its streaming ledger hash")


def _verify_combined_labels(training: Path) -> dict:
    summary_path = training / "run/summary.json"
    labels_path = training / "run/labels.json"
    if not summary_path.is_file() or not labels_path.is_file():
        raise TrainingPending("Combined T02 labels are not ready")
    summary = base._read(summary_path)
    expected_summary = {
        "schema": "t02-streaming-recovery-summary-v1",
        "phase": "labels_completed",
        "status": "labels_completed",
        "combined_exact_state_count": EXPECTED_TRAINING_STATE_COUNT,
        "preserved_exact_state_count": EXPECTED_PRESERVED_STATE_COUNT,
        "new_exact_state_count": EXPECTED_NEW_STATE_COUNT,
        "actual_split_counts": EXPECTED_SPLIT_COUNTS,
        "task_group_count": EXPECTED_TASK_GROUP_COUNT,
        "source_task_count": EXPECTED_STREAMING_TASK_COUNT,
        "streaming_task_plan_count": EXPECTED_STREAMING_TASK_COUNT,
        "new_complete_branch_executions": EXPECTED_NEW_BRANCH_COUNT,
        "cumulative_complete_branch_executions": EXPECTED_CUMULATIVE_BRANCH_COUNT,
        "synthetic_global_plan_sha256": None,
        "training_allowed": True,
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise ValueError("T02 summary does not authorize the combined 118-state training set")
    labels = base._read(labels_path)
    rows = labels.get("rows")
    partitions = labels.get("partitions")
    if (labels.get("schema") != "t02-multi-plan-labeled-set-v1"
            or labels.get("state_count") != EXPECTED_TRAINING_STATE_COUNT
            or not isinstance(rows, list)
            or len(rows) != EXPECTED_TRAINING_STATE_COUNT
            or len({row.get("state_id") for row in rows if isinstance(row, dict)})
            != EXPECTED_TRAINING_STATE_COUNT
            or not isinstance(partitions, list)
            or len(partitions) != 1 + EXPECTED_STREAMING_TASK_COUNT
            or partitions[0].get("role") != "preserved_v6_completed_triplets"
            or partitions[0].get("selected_state_count") != EXPECTED_PRESERVED_STATE_COUNT
            or any(row.get("role") != "streaming_exact_recollection"
                   for row in partitions[1:])
            or sum(row.get("selected_state_count", -1) for row in partitions[1:])
            != EXPECTED_NEW_STATE_COUNT
            or any(not _valid_sha256(row.get("plan_sha256")) for row in partitions)
            or labels.get("plan_sha256s") != [row["plan_sha256"] for row in partitions]
            or labels.get("synthetic_global_plan_sha256") is not None
            or labels.get("old_result_plan_sha256_rewritten") is not False):
        raise ValueError("T02 labels lack the frozen 10+108 streaming multi-plan provenance")
    contract = base._read(training / "recovery/recovery_contract.json")
    bindings = contract.get("group_split_bindings")
    target = contract.get("target")
    budget = contract.get("budget")
    source_budget = contract.get("source_budget")
    if (not isinstance(target, dict)
            or target.get("combined_exact_state_count") != EXPECTED_TRAINING_STATE_COUNT
            or target.get("preserved_exact_state_count") != EXPECTED_PRESERVED_STATE_COUNT
            or target.get("new_exact_state_count") != EXPECTED_NEW_STATE_COUNT
            or target.get("split_counts") != EXPECTED_SPLIT_COUNTS
            or target.get("task_group_count") != EXPECTED_TASK_GROUP_COUNT
            or not isinstance(budget, dict)
            or budget.get("complete_branches_before_recovery") != 35
            or budget.get("new_complete_branch_executions") != EXPECTED_NEW_BRANCH_COUNT
            or budget.get("cumulative_complete_branches") != EXPECTED_CUMULATIVE_BRANCH_COUNT
            or budget.get("complete_branch_cap") != 360
            or not isinstance(source_budget, dict)
            or source_budget.get("prior_source_starts") != 241
            or source_budget.get("new_source_task_starts") != EXPECTED_STREAMING_TASK_COUNT
            or source_budget.get("cumulative_source_starts") != 324
            or source_budget.get("automatic_reruns") != 0
            or labels.get("recovery_contract_sha256") != _canonical_digest(contract)
            or not isinstance(bindings, dict)
            or len(bindings) != EXPECTED_TASK_GROUP_COUNT
            or any(bindings.get(row.get("task_group_id")) != row.get("split")
                   for row in rows)
            or len({row.get("task_group_id") for row in rows}) != EXPECTED_TASK_GROUP_COUNT
            or labels.get("rows_sha256") != _canonical_digest(rows)
            or _canonical_digest(labels) != summary.get("combined_labels_sha256")):
        raise ValueError("Combined labels differ from the frozen group/provenance contract")

    source_plan = base._read(training / "recovery/source_plan.json")
    source_plan_sha256 = _canonical_digest(source_plan)
    if (contract.get("source_plan", {}).get("sha256") != source_plan_sha256
            or partitions[0].get("plan_sha256") != source_plan_sha256
            or summary.get("source_plan_sha256") != source_plan_sha256):
        raise ValueError("Preserved labels are not bound to the frozen source plan")
    if (summary.get("actual_task_plan_sha256s")
            != [row["plan_sha256"] for row in partitions[1:]]):
        raise ValueError("Summary per-source plan provenance differs")

    preserved_path = training / "run/preserved_labels.json"
    if not preserved_path.is_file():
        raise ValueError("Preserved label artifact is missing")
    preserved = base._read(preserved_path)
    old_partition = partitions[0]
    _verify_file_sha256(preserved_path, old_partition.get("source_artifact_file_sha256"),
                        "preserved labels")
    preserved_rows = preserved.get("rows")
    preserved_ids = old_partition.get("state_ids")
    if (preserved.get("plan_sha256") != source_plan_sha256
            or not isinstance(preserved_rows, list)
            or not isinstance(preserved_ids, list)
            or len(preserved_ids) != EXPECTED_PRESERVED_STATE_COUNT
            or len(set(preserved_ids)) != EXPECTED_PRESERVED_STATE_COUNT
            or old_partition.get("source_artifact_sha256") != _canonical_digest(preserved)):
        raise ValueError("Preserved label partition provenance differs")
    combined_by_id = {row["state_id"]: row for row in rows}
    preserved_by_id = {row.get("state_id"): row for row in preserved_rows
                       if isinstance(row, dict)}
    if (set(preserved_ids) != {row.get("state_id") for row in contract.get("preserved_states", [])}
            or any(state_id not in preserved_by_id for state_id in preserved_ids)
            or any(combined_by_id.get(state_id) != {
                **preserved_by_id[state_id], "result_plan_sha256": source_plan_sha256}
                for state_id in preserved_ids)):
        raise ValueError("Preserved combined rows differ from their exact source artifact")

    ledger_path = training / "ledger.json"
    if not ledger_path.is_file():
        raise ValueError("Streaming recovery ledger is missing")
    ledger = base._read(ledger_path)
    ledger_tasks = ledger.get("tasks")
    source_tasks = contract.get("source_tasks")
    if (ledger.get("schema") != "t02-streaming-recovery-ledger-v1"
            or not isinstance(ledger_tasks, list)
            or not isinstance(source_tasks, list)
            or len(ledger_tasks) != EXPECTED_STREAMING_TASK_COUNT
            or len(source_tasks) != EXPECTED_STREAMING_TASK_COUNT
            or any(row.get("status") != "completed" for row in ledger_tasks)
            or ledger.get("complete_branches_before_recovery")
            != EXPECTED_CUMULATIVE_BRANCH_COUNT - EXPECTED_NEW_BRANCH_COUNT
            or ledger.get("new_complete_branch_cap") != EXPECTED_NEW_BRANCH_COUNT
            or ledger.get("cumulative_complete_branch_cap") != 360
            or ledger.get("prior_source_starts") != source_budget["prior_source_starts"]
            or ledger.get("cumulative_source_start_cap")
            != source_budget["cumulative_source_starts"]
            or sum(row.get("complete_branch_executions", -1) for row in ledger_tasks)
            != EXPECTED_NEW_BRANCH_COUNT):
        raise ValueError("Streaming recovery ledger is stale or partial")
    if ([row.get("task_id") for row in ledger_tasks]
            != [row.get("task_id") for row in source_tasks]):
        raise ValueError("Streaming ledger differs from the frozen source-task order")

    new_partitions = partitions[1:]
    if ([row.get("task_id") for row in new_partitions]
            != [row.get("task_id") for row in source_tasks]):
        raise ValueError("Streaming label partitions differ from the frozen source-task order")
    covered_new_ids = set()
    recovery_sha256 = _canonical_digest(contract)
    for ledger_row, source, partition in zip(ledger_tasks, source_tasks, new_partitions):
        task_id = source.get("task_id")
        expected_slots = source.get("expected_slots")
        if (ledger_row.get("task_id") != task_id
                or ledger_row.get("expected_slots") != expected_slots
                or ledger_row.get("expected_slot_count") != len(expected_slots)):
            raise ValueError("Streaming ledger changed a frozen source slot")
        expected_branches = 3 * len(expected_slots)
        if (ledger_row.get("complete_branch_executions") != expected_branches
                or partition.get("selected_state_count") != len(expected_slots)):
            raise ValueError("Streaming source did not persist three branches per exact slot")

        plan_path = _bound_path(training, ledger_row.get("plan_path"), "task plan")
        result_path = _bound_path(training, ledger_row.get("result_path"), "task result")
        label_path = _bound_path(training, ledger_row.get("label_path"), "task labels")
        _verify_file_sha256(plan_path, ledger_row.get("plan_file_sha256"), "task plan")
        _verify_file_sha256(result_path, ledger_row.get("result_file_sha256"), "task result")
        _verify_file_sha256(label_path, ledger_row.get("label_file_sha256"), "task labels")
        if (partition.get("plan_file_sha256") != ledger_row.get("plan_file_sha256")
                or partition.get("source_artifact_file_sha256")
                != ledger_row.get("label_file_sha256")):
            raise ValueError("Streaming partition file hashes differ from its ledger")

        plan = base._read(plan_path)
        result = base._read(result_path)
        task_labels = base._read(label_path)
        plan_sha256 = _canonical_digest(plan)
        state_ids = partition.get("state_ids")
        plan_states = plan.get("states")
        task_rows = task_labels.get("rows")
        if (plan.get("schema") != "t02-streaming-task-plan-v1"
                or plan.get("task_id") != task_id
                or plan.get("expected_slots") != expected_slots
                or plan.get("recovery_contract_sha256") != recovery_sha256
                or plan.get("source_plan_sha256") != source_plan_sha256
                or not isinstance(plan_states, list)
                or len(plan_states) != len(expected_slots)
                or ledger_row.get("plan_sha256") != plan_sha256
                or partition.get("plan_sha256") != plan_sha256):
            raise ValueError("Streaming task plan differs from its frozen slot provenance")
        plan_state_ids = [row.get("state_id") for row in plan_states]
        if (not isinstance(state_ids, list) or state_ids != plan_state_ids
                or len(set(state_ids)) != len(state_ids)
                or covered_new_ids.intersection(state_ids)):
            raise ValueError("Streaming partitions repeat or misown exact states")
        for state, slot in zip(plan_states, expected_slots):
            if (state.get("task_id") != task_id
                    or state.get("draft_kind") != slot.get("draft_kind")
                    or state.get("task_group_id") != slot.get("task_group_id")
                    or state.get("split") != slot.get("split")):
                raise ValueError("Streaming task plan changed group or split isolation")

        required_results = {(state_id, branch) for state_id in state_ids
                            for branch in ("A0", "A1", "A2")}
        result_rows = result.get("results")
        if (result.get("plan_sha256") != plan_sha256
                or result.get("complete_branch_executions") != expected_branches
                or not isinstance(result_rows, list)
                or {(row.get("state_id"), row.get("branch_id")) for row in result_rows}
                != required_results
                or any(row.get("execution_status") != "complete" for row in result_rows)):
            raise ValueError("Streaming task result is partial or bound to another plan")
        if (task_labels.get("schema") != "t02-labeled-dataset-v1"
                or task_labels.get("plan_schema") != "t02-streaming-task-plan-v1"
                or task_labels.get("plan_sha256") != plan_sha256
                or task_labels.get("state_count") != len(state_ids)
                or task_labels.get("observed_branch_count") != expected_branches
                or not isinstance(task_rows, list)
                or [row.get("state_id") for row in task_rows] != state_ids
                or any(row.get("result_plan_sha256") != plan_sha256 for row in task_rows)
                or partition.get("source_artifact_sha256") != _canonical_digest(task_labels)
                or any(combined_by_id.get(row["state_id"]) != row for row in task_rows)):
            raise ValueError("Streaming task labels differ from their actual per-source plan")
        covered_new_ids.update(state_ids)

    if covered_new_ids != set(combined_by_id) - set(preserved_ids):
        raise ValueError("Streaming partitions do not cover exactly the 108 new states")
    return labels


def verify_training_binding(training: Path, binding_path: Path, lane_name: str) -> dict:
    """Bind one lane to frozen v8 sources, streaming labels, and a successful fit."""
    training = training.resolve()
    if training.name != EXPECTED_TRAINING_PACKAGE:
        raise ValueError("Post-T02 evaluation requires prepared_v8; stale packages are forbidden")
    binding = base._read(binding_path)
    if binding.get("schema") != TRAINING_BINDING_SCHEMA:
        raise ValueError("Unknown post-T02 training binding schema")
    if Path(binding.get("training_package", "")).resolve() != training:
        raise ValueError("Training binding names a different package")
    hashes = binding.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(TRAINING_BINDING_FILES):
        raise ValueError("Training binding must hash every required frozen v8 source")
    for relative, expected in hashes.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Invalid SHA-256 for {relative}")
        path = training / relative
        if not path.is_file() or base._sha(path) != expected:
            raise ValueError(f"Frozen training artifact differs: {relative}")

    labels = _verify_combined_labels(training)
    lane = LANES[lane_name]
    artifact_path = training / lane["artifact"]
    receipt_path = training / "training/receipts" / f"{lane_name}.json"
    if not artifact_path.is_file():
        raise TrainingPending(f"{lane_name} artifact is not ready")
    artifact = base._read(artifact_path)
    if artifact.get("model_kind") != lane["kind"] or not isinstance(artifact.get("fit"), dict):
        raise ValueError(f"Missing actual fitted selector artifact: {lane['artifact']}")
    if receipt_path.is_file():
        receipt = base._read(receipt_path)
        expected = {
            "schema": "t02-training-lane-receipt-v1",
            "status": "completed",
            "lane": lane_name,
            "artifact_sha256": base._sha(artifact_path),
            "labels_sha256": base._sha(training / "run/labels.json"),
            "state_count": EXPECTED_TRAINING_STATE_COUNT,
            "frozen_manifest_sha256": base._sha(training / "source_files.json"),
        }
        receipt_artifact = Path(receipt.get("artifact", ""))
        if not receipt_artifact.is_absolute():
            receipt_artifact = training / receipt_artifact
        if (receipt_artifact.resolve() != artifact_path.resolve()
                or any(receipt.get(key) != value for key, value in expected.items())):
            raise ValueError(f"{lane_name} training receipt does not verify this fitted artifact")
    else:
        raise TrainingPending(f"{lane_name} has no successful training receipt yet")
    return binding


def trained_design(template, *, lane, controller, tasks, manifest_sha256, artifact_sha256):
    if len(tasks) != 20 or len(set(tasks)) != 20:
        raise ValueError("The trained selector requires the same fixed D20 denominator")
    result = copy.deepcopy(template)
    result.update(candidate_id="evidence_sets_v1_h0_" + lane["name"].lower(),
                  run_id_template="a_history_evidence_sets_v1_" + lane["name"].lower(),
                  task_ids=list(tasks), task_manifest_sha256=manifest_sha256,
                  launch_authorized=True, automatic_reruns=0)
    result["limits"]["tasks"] = 20
    result["runtime"]["sglang_backend_url"] = f"http://127.0.0.1:{lane['engine_port']}"
    result["resolved_configs"]["controller"] = controller
    result["search_contract"].update(fixed_denominator="r001_mixed20",
        trained_artifact_sha256=artifact_sha256,
        semantic_query_overflow_policy="task_head_tail_preserve_draft_v1")
    for key in ("repair_task_count", "clean_reuse_task_count", "clean_reuse_receipt"):
        result["search_contract"].pop(key, None)
    result["cpu_validation"] = {"source": "verified T02 runtime and actual fitted artifact",
                                "note": "No synthetic or placeholder training artifact"}
    return result


def frozen_sglang_manifest(training: Path) -> dict:
    """Derive the evaluator manifest from prepared_v8's frozen source manifest."""
    source_manifest_path = training / "source_files.json"
    source = training / "sglang"
    if not source_manifest_path.is_file() or not source.is_dir():
        raise FileNotFoundError("prepared_v8 frozen SGLang snapshot is missing")
    source_manifest = base._read(source_manifest_path)
    source_files = source_manifest.get("files")
    if not isinstance(source_files, dict):
        raise ValueError("prepared_v8 source manifest is invalid")
    files = {name: digest for name, digest in source_files.items()
             if name.startswith("sglang/")}
    if not files:
        raise ValueError("prepared_v8 source manifest has no SGLang files")
    for relative, digest in files.items():
        path = training / relative
        if not _valid_sha256(digest) or not path.is_file() or base._sha(path) != digest:
            raise ValueError(f"prepared_v8 SGLang file differs: {relative}")
    return {"schema": "evidence-sets-eval-sglang-files-v1",
            "file_count": len(files), "files": files,
            "copy_source": str(source)}


def copy_frozen_sglang(training: Path, package: Path) -> dict:
    """Copy the SGLang snapshot verified as part of prepared_v8."""
    manifest = frozen_sglang_manifest(training)
    source = training / "sglang"
    base._save(package / "sglang_files.json", manifest)
    shutil.copytree(source, package / "sglang",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return manifest


def prepare(package, source, training, lane_name, training_binding):
    if package.exists():
        raise FileExistsError("Refusing to overwrite a trained evaluation package")
    base.verify_package(source)
    sys.path.insert(0, str(training / "history_system"))
    from t02_parallel_launch import verify
    verify(training)
    verify_training_binding(training, training_binding, lane_name)
    lane = next(row for row in lane_specs() if row["name"] == lane_name)
    artifact_path = training / LANES[lane_name]["artifact"]
    artifact = base._read(artifact_path)
    if artifact.get("model_kind") != LANES[lane_name]["kind"]:
        raise ValueError("The actual artifact kind differs from the requested lane")
    tasks = base._read(source / "tasks.d20.json")
    base.validate_task_isolation(tasks, base._read(source / "tasks.t02.expanded104.json"))
    runtime_source = training / "history_system/runtime"
    sys.path[:0] = [str(runtime_source / "python"), str(runtime_source)]
    from benchmarks.memory_runtime.recovery.experiment_config import parse_gp_config
    from benchmarks.memory_runtime.recovery.local_selection_models import LocalSelectionModels
    from benchmarks.memory_runtime.recovery.set_models import load_set_selector, validate_selector_score_models
    controller = copy.deepcopy(base._read(training / "configs/runtime.production.json")
                               ["resolved_configs"]["controller"])
    gp = controller["gp_experiments"]
    gp.update(set_selector=lane["selector"], selector_artifact=artifact,
              semantic_query_overflow_policy="task_head_tail_preserve_draft_v1",
              selector_threshold=0.5, gain_delta=0.0, export_selection_state=False)
    if lane_name != "C1":
        gp["local_models"]["reranker"]["batch_size"] = 1
    gp = parse_gp_config(gp)
    selector = load_set_selector(artifact)
    if selector.kind != lane["selector"]:
        raise ValueError("Selector kind mismatch")
    if selector.kind != "risk":
        validate_selector_score_models(selector, LocalSelectionModels(gp["local_models"]),
            semantic_query_overflow_policy=gp["semantic_query_overflow_policy"])
    controller["gp_experiments"] = gp
    package.mkdir(parents=True)
    (package / "history_system").mkdir()
    shutil.copyfile(source / "history_system/runner.py", package / "history_system/runner.py")
    shutil.copyfile(source / "evidence_eval.py", package / "evidence_eval.py")
    shutil.copyfile(Path(__file__), package / Path(__file__).name)
    for filename in ("tasks.d20.json", "tasks.t02.expanded104.json", "remote_lineage.json"):
        shutil.copyfile(source / filename, package / filename)
    runtime = package / "lanes" / lane_name / "runtime"
    base._copy_tree(runtime_source, runtime)
    base._save(runtime / "configs/controller.json", controller)
    template = base._read(source / "lanes/C2/design.json")
    design = trained_design(template, lane=lane, controller=controller,
        tasks=tasks["task_ids"], manifest_sha256=base._sha(package / "tasks.d20.json"),
        artifact_sha256=base._sha(artifact_path))
    design["source_files"] = base._tree_hashes(runtime)
    lane_root = runtime.parent
    for filename, value in (("design.json", design), ("lane.json", lane),
                            ("gp.json", gp), ("tasks.json", tasks)):
        base._save(lane_root / filename, value)
    base._save(lane_root / "trained_artifact.json", artifact)
    base._save(lane_root / "preview.json", base._preview_lane(package, lane))
    sglang_manifest = copy_frozen_sglang(training, package)
    base._save(package / "launch_contract.json", {
        "schema": "evidence-sets-trained-eval-v1", "launch_authorized": True,
        "total_task_execution_budget": 20, "automatic_reruns": 0,
        "lanes": [lane], "artifact_source": str(artifact_path),
        "artifact_sha256": base._sha(artifact_path),
        "source_package": str(source), "t02_package": str(training),
        "sglang_source": str(training / "sglang"),
        "sglang_manifest_sha256": base._sha(package / "sglang_files.json"),
        "sglang_file_count": sglang_manifest["file_count"],
        "required_environment": {"C2KV_STRICT_NONFINITE_SAMPLING": "1"},
        "threshold_policy": "C1=0.5; C4 delta=0; no evaluation-set tuning"})
    base._save(package / "static_files.json", base._static_manifest(package))
    return base.verify_package(package)


def ascend_environment():
    command = ("source /usr/local/Ascend/ascend-toolkit/set_env.sh >/dev/null 2>&1; "
               "source /usr/local/Ascend/nnal/atb/set_env.sh >/dev/null 2>&1; env -0")
    result = subprocess.run(["bash", "-c", command], check=True, capture_output=True)
    for item in result.stdout.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            os.environ[key.decode()] = value.decode()


def enable_strict_sampling() -> None:
    os.environ["C2KV_STRICT_NONFINITE_SAMPLING"] = "1"


def wait_run(args):
    status_path = args.package.parent / (args.package.name + ".waiting.json")
    if status_path.exists():
        raise FileExistsError("Trained evaluation already scheduled")
    base._save(status_path, {"phase": "waiting_for_training", "pid": os.getpid(),
                            "lane": args.lane, "created_at": base._now()})
    try:
        while True:
            path = args.training / "status.json"
            status = base._read(path) if path.exists() else {}
            phase = status.get("phase")
            if phase in {"failed_no_retry", "partial_failed"}:
                raise RuntimeError("T02 failed; no artifact evaluation is permitted")
            try:
                verify_training_binding(args.training, args.training_binding, args.lane)
                break
            except TrainingPending:
                if phase == "training_partial_or_failed":
                    raise RuntimeError(f"{args.lane} training has no verified success receipt")
            time.sleep(20)
        for dependency in args.after_status:
            while True:
                value = base._read(dependency) if dependency.exists() else {}
                if value.get("state", value.get("phase")) in {"completed", "failed_no_rerun", "failed_no_retry"}:
                    break
                time.sleep(20)
        bind_base(args.package)
        prepare(args.package, args.source, args.training, args.lane, args.training_binding)
        lane = next(row for row in lane_specs() if row["name"] == args.lane)
        while True:
            try:
                base._assert_lane_free(lane)
                break
            except RuntimeError:
                time.sleep(20)
        ascend_environment()
        enable_strict_sampling()
        base._save(status_path, {"phase": "running", "pid": os.getpid(),
                                "lane": args.lane, "updated_at": base._now()})
        code = base.run_lane(args.package, args.lane)
        base._save(status_path, {"phase": "completed" if code == 0 else "failed_no_retry",
                                "returncode": code, "updated_at": base._now()})
        return code
    except Exception as error:
        base._save(status_path, {"phase": "failed_no_retry", "error": str(error),
                                "updated_at": base._now()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--training-binding", type=Path, required=True)
    parser.add_argument("--lane", choices=tuple(LANES), required=True)
    parser.add_argument("--after-status", type=Path, action="append", default=[])
    raise SystemExit(wait_run(parser.parse_args()))
