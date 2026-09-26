import copy
import importlib.util
import json
from pathlib import Path

import pytest

import evidence_eval_trained as trained
import runner


POST_SCRIPT = Path(__file__).parents[3] / "tmp/start_post_t02_v4.py"
POST_SPEC = importlib.util.spec_from_file_location("start_post_t02_v4", POST_SCRIPT)
post = importlib.util.module_from_spec(POST_SPEC)
assert POST_SPEC.loader is not None
POST_SPEC.loader.exec_module(post)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _prepared_v8(tmp_path: Path, lane_name: str = "C1") -> tuple[Path, Path]:
    root = tmp_path / "prepared_v8"
    bindings = {f"group-{index}": "train" if index < 17 else "calibration"
                for index in range(26)}
    source_plan = {"schema": "test-source-plan", "seed": 0}
    source_plan_sha256 = trained._canonical_digest(source_plan)
    old_rows = []
    for index in range(8):
        old_rows.append({"state_id": f"old-train-{index}",
                         "task_group_id": f"group-{index}", "split": "train"})
    for index in range(2):
        old_rows.append({"state_id": f"old-calibration-{index}",
                         "task_group_id": f"group-{17 + index}",
                         "split": "calibration"})
    preserved = {"schema": "t02-labeled-dataset-v1",
                 "plan_sha256": source_plan_sha256, "state_count": len(old_rows),
                 "rows": old_rows}

    source_tasks = []
    new_row_specs = []
    state_index = 0
    for task_index in range(83):
        slot_count = 2 if task_index < 25 else 1
        split = "train" if task_index < 47 else "calibration"
        slots = []
        for slot_index in range(slot_count):
            split_index = state_index if split == "train" else state_index - 72
            group_index = split_index % 17 if split == "train" else 17 + split_index % 9
            state_id = f"new-{state_index}"
            slot = {"task_id": f"task-{task_index}",
                    "draft_kind": "call" if slot_index == 0 else "stop",
                    "task_group_id": f"group-{group_index}", "split": split}
            slots.append(slot)
            new_row_specs.append((state_id, slot))
            state_index += 1
        source_tasks.append({"task_id": f"task-{task_index}", "expected_slots": slots})
    assert state_index == 108
    contract = {
        "source_plan": {"sha256": source_plan_sha256},
        "budget": {"complete_branches_before_recovery": 35,
                   "new_complete_branch_executions": 324,
                   "cumulative_complete_branches": 359,
                   "complete_branch_cap": 360},
        "source_budget": {"prior_source_starts": 241,
                          "new_source_task_starts": 83,
                          "cumulative_source_starts": 324,
                          "automatic_reruns": 0},
        "target": {"combined_exact_state_count": 118,
                   "preserved_exact_state_count": 10,
                   "new_exact_state_count": 108,
                   "split_counts": {"calibration": 38, "train": 80},
                   "task_group_count": 26},
        "group_split_bindings": bindings,
        "preserved_states": [{"state_id": row["state_id"]} for row in old_rows],
        "source_tasks": source_tasks,
    }
    contract_sha256 = trained._canonical_digest(contract)
    combined_rows = [{**row, "result_plan_sha256": source_plan_sha256} for row in old_rows]
    preserved_path = root / "run/preserved_labels.json"
    _write(preserved_path, preserved)
    partitions = [{"role": "preserved_v6_completed_triplets",
                   "plan_sha256": source_plan_sha256,
                   "source_artifact_sha256": trained._canonical_digest(preserved),
                   "source_artifact_file_sha256": trained.base._sha(preserved_path),
                   "selected_state_count": 10,
                   "state_ids": [row["state_id"] for row in old_rows]}]
    ledger_tasks = []
    offset = 0
    for task_index, source in enumerate(source_tasks):
        slots = source["expected_slots"]
        specs = new_row_specs[offset:offset + len(slots)]
        offset += len(slots)
        plan_states = [{"state_id": state_id, **slot}
                       for state_id, slot in specs]
        plan = {"schema": "t02-streaming-task-plan-v1",
                "task_id": source["task_id"], "target_states": len(plan_states),
                "expected_slots": slots, "states": plan_states,
                "source_plan_sha256": source_plan_sha256,
                "recovery_contract_sha256": contract_sha256}
        plan_sha256 = trained._canonical_digest(plan)
        result_rows = [{"state_id": state["state_id"], "branch_id": branch,
                        "execution_status": "complete"}
                       for state in plan_states for branch in ("A0", "A1", "A2")]
        result = {"schema": "t02-branch-results-v1", "plan_sha256": plan_sha256,
                  "complete_branch_executions": len(result_rows), "results": result_rows}
        task_rows = [{**state, "result_plan_sha256": plan_sha256}
                     for state in plan_states]
        task_labels = {"schema": "t02-labeled-dataset-v1",
                       "plan_schema": "t02-streaming-task-plan-v1",
                       "plan_sha256": plan_sha256, "state_count": len(task_rows),
                       "observed_branch_count": len(result_rows), "rows": task_rows}
        task_root = root / f"workers/worker-{task_index % 6}/tasks/{task_index:03d}"
        plan_path = task_root / "plan.json"
        result_path = task_root / "results.json"
        label_path = task_root / "labels.json"
        _write(plan_path, plan)
        _write(result_path, result)
        _write(label_path, task_labels)
        plan_file_sha256 = trained.base._sha(plan_path)
        result_file_sha256 = trained.base._sha(result_path)
        label_file_sha256 = trained.base._sha(label_path)
        ledger_tasks.append({"task_id": source["task_id"], "task_index": task_index,
                             "expected_slots": slots, "expected_slot_count": len(slots),
                             "status": "completed", "complete_branch_executions": len(result_rows),
                             "plan_path": str(plan_path), "plan_sha256": plan_sha256,
                             "plan_file_sha256": plan_file_sha256,
                             "result_path": str(result_path),
                             "result_file_sha256": result_file_sha256,
                             "label_path": str(label_path),
                             "label_file_sha256": label_file_sha256})
        partitions.append({"role": "streaming_exact_recollection",
                           "task_id": source["task_id"], "plan_sha256": plan_sha256,
                           "source_artifact_sha256": trained._canonical_digest(task_labels),
                           "plan_file_sha256": plan_file_sha256,
                           "source_artifact_file_sha256": label_file_sha256,
                           "selected_state_count": len(task_rows),
                           "state_ids": [row["state_id"] for row in task_rows]})
        combined_rows.extend(task_rows)

    labels = {"schema": "t02-multi-plan-labeled-set-v1", "state_count": 118,
              "rows": combined_rows, "partitions": partitions,
              "plan_sha256s": [row["plan_sha256"] for row in partitions],
              "recovery_contract_sha256": contract_sha256,
              "rows_sha256": trained._canonical_digest(combined_rows),
              "old_result_plan_sha256_rewritten": False,
              "synthetic_global_plan_sha256": None}
    summary = {"schema": "t02-streaming-recovery-summary-v1",
               "phase": "labels_completed", "status": "labels_completed",
               "source_task_count": 83, "combined_exact_state_count": 118,
               "preserved_exact_state_count": 10, "new_exact_state_count": 108,
               "actual_split_counts": {"calibration": 38, "train": 80},
               "task_group_count": 26, "streaming_task_plan_count": 83,
               "new_complete_branch_executions": 324,
               "cumulative_complete_branch_executions": 359,
               "source_plan_sha256": source_plan_sha256,
               "actual_task_plan_sha256s": [row["plan_sha256"] for row in partitions[1:]],
               "synthetic_global_plan_sha256": None,
               "combined_labels_sha256": trained._canonical_digest(labels),
               "training_allowed": True}
    _write(root / "source_files.json", {"files": {}})
    _write(root / "contract.json", {"schema": "test"})
    _write(root / "recovery/recovery_contract.json", contract)
    _write(root / "recovery/source_plan.json", source_plan)
    _write(root / "run/labels.json", labels)
    _write(root / "run/summary.json", summary)
    _write(root / "ledger.json", {"schema": "t02-streaming-recovery-ledger-v1",
                                   "complete_branches_before_recovery": 35,
                                   "new_complete_branch_cap": 324,
                                   "cumulative_complete_branch_cap": 360,
                                   "prior_source_starts": 241,
                                   "cumulative_source_start_cap": 324,
                                   "tasks": ledger_tasks})
    _write(root / "status.json", {"phase": "training"})
    lane = trained.LANES[lane_name]
    artifact_path = root / lane["artifact"]
    _write(artifact_path, {"model_kind": lane["kind"], "fit": {"estimator": "real"}})
    receipt = {"schema": "t02-training-lane-receipt-v1", "status": "completed",
               "lane": lane_name, "artifact": lane["artifact"],
               "artifact_sha256": trained.base._sha(artifact_path),
               "labels_sha256": trained.base._sha(root / "run/labels.json"),
               "state_count": 118,
               "frozen_manifest_sha256": trained.base._sha(root / "source_files.json")}
    _write(root / f"training/receipts/{lane_name}.json", receipt)
    static_files = ("source_files.json", "contract.json", "recovery/recovery_contract.json",
                    "recovery/source_plan.json")
    binding = tmp_path / "binding.json"
    _write(binding, {"schema": "post-t02-streaming-training-binding-v2",
                     "training_package": str(root.resolve()),
                     "sha256": {name: trained.base._sha(root / name) for name in static_files}})
    return root, binding


def test_trained_design_restores_the_entire_d20_after_partial_source_run(monkeypatch):
    source = Path(__file__).parents[2] / "outputs/history_system_search/evidence_sets_v1/eval/never_started_c2c3/prepared_v4"
    template = json.loads((source / "lanes/C2/design.json").read_text())
    tasks = json.loads((source / "tasks.d20.json").read_text())["task_ids"]
    original = copy.deepcopy(template)
    monkeypatch.setattr(runner, "ROOT", source / "lanes/C2/runtime")
    lane = trained.lane_specs()[1]
    result = trained.trained_design(template, lane=lane,
        controller=template["resolved_configs"]["controller"], tasks=tasks,
        manifest_sha256="a" * 64, artifact_sha256="b" * 64)
    runner.validate(result, allow_development=False)
    assert result["task_ids"] == tasks and len(tasks) == 20
    assert result["limits"]["tasks"] == 20
    assert result["search_contract"]["trained_artifact_sha256"] == "b" * 64
    assert template == original


def test_partial_or_duplicate_evaluation_denominator_is_rejected():
    for tasks in (["task"] * 20, [f"task-{x}" for x in range(19)]):
        with pytest.raises(ValueError, match="fixed D20"):
            trained.trained_design({}, lane={}, controller={}, tasks=tasks,
                                   manifest_sha256="a" * 64, artifact_sha256="b" * 64)


def test_trained_lanes_avoid_other_user_and_forbidden_devices_and_ports():
    lanes = trained.lane_specs()
    assert len({x["physical_device"] for x in lanes}) == 3
    assert not {5, 7}.intersection(x["physical_device"] for x in lanes)
    ports = [p for row in lanes for p in [row["engine_port"], *row["task_ports"]]]
    assert len(ports) == len(set(ports))


def test_v8_lane_receipt_allows_lane_before_overall_training_completes(tmp_path: Path):
    root, binding = _prepared_v8(tmp_path)
    value = trained.verify_training_binding(root, binding, "C1")
    assert value["training_package"] == str(root.resolve())


def test_v8_lane_without_receipt_waits_even_if_overall_training_succeeded(tmp_path: Path):
    root, binding = _prepared_v8(tmp_path)
    (root / "training/receipts/C1.json").unlink()
    with pytest.raises(trained.TrainingPending, match="no successful training receipt"):
        trained.verify_training_binding(root, binding, "C1")
    _write(root / "status.json", {"phase": "completed", "trainer_exit_code": 0})
    with pytest.raises(trained.TrainingPending, match="no successful training receipt"):
        trained.verify_training_binding(root, binding, "C1")


def test_v8_binding_rejects_changed_group_split_provenance(tmp_path: Path):
    root, binding = _prepared_v8(tmp_path)
    labels = json.loads((root / "run/labels.json").read_text())
    labels["rows"][0]["split"] = "calibration"
    _write(root / "run/labels.json", labels)
    with pytest.raises(ValueError, match="group/provenance"):
        trained.verify_training_binding(root, binding, "C1")


def test_v8_binding_rejects_partial_streaming_ledger(tmp_path: Path):
    root, binding = _prepared_v8(tmp_path)
    ledger = json.loads((root / "ledger.json").read_text())
    ledger["tasks"][0]["status"] = "failed_no_retry"
    _write(root / "ledger.json", ledger)
    with pytest.raises(ValueError, match="stale or partial"):
        trained.verify_training_binding(root, binding, "C1")


def test_v8_binding_rejects_stale_package_name(tmp_path: Path):
    root, binding = _prepared_v8(tmp_path)
    stale = tmp_path / "prepared_v7"
    root.rename(stale)
    with pytest.raises(ValueError, match="requires prepared_v8"):
        trained.verify_training_binding(stale, binding, "C1")


def test_v8_binding_rejects_per_source_plan_hash_substitution(tmp_path: Path):
    root, binding = _prepared_v8(tmp_path)
    labels = json.loads((root / "run/labels.json").read_text())
    labels["partitions"][1]["plan_sha256"] = "f" * 64
    labels["plan_sha256s"][1] = "f" * 64
    _write(root / "run/labels.json", labels)
    summary = json.loads((root / "run/summary.json").read_text())
    summary["combined_labels_sha256"] = trained._canonical_digest(labels)
    _write(root / "run/summary.json", summary)
    receipt = json.loads((root / "training/receipts/C1.json").read_text())
    receipt["labels_sha256"] = trained.base._sha(root / "run/labels.json")
    _write(root / "training/receipts/C1.json", receipt)
    with pytest.raises(ValueError, match="per-source plan provenance"):
        trained.verify_training_binding(root, binding, "C1")


def test_v4_schedule_keeps_original_d20_lane_and_device_contract(tmp_path: Path):
    root, binding = _prepared_v8(tmp_path)
    base = root.parent
    value = post.verify_static_binding(base, binding)
    commands = post.build_commands(base, base / "post_t02_v4", binding, "/python")
    assert value["training_package"] == str(root.resolve())
    assert [lane for lane, _ in commands] == ["C1", "C4_turn", "C4_task"]
    assert all("prepared_v8" in " ".join(command)
               and "eval_trained_v4" in " ".join(command) for _, command in commands)
    assert all("prepared_v7" not in " ".join(command) for _, command in commands)
    assert commands[2][1][-2:] == ["--after-status",
        str(base / "eval_failed_repair_v1/lanes/C0/run/status.json")]


def test_v4_binding_is_created_once_from_exact_frozen_inputs(tmp_path: Path):
    root, original = _prepared_v8(tmp_path)
    generated = tmp_path / "new.binding.json"
    value = post.write_static_binding(root.parent, generated)
    assert value["schema"] == post.BINDING_SCHEMA
    assert value["training_package"] == str(root.resolve())
    assert value["sha256"] == json.loads(original.read_text())["sha256"]
    with pytest.raises(FileExistsError, match="overwrite"):
        post.write_static_binding(root.parent, generated)


def test_trained_eval_copies_sglang_only_from_frozen_v8(tmp_path: Path):
    training = tmp_path / "prepared_v8"
    package = tmp_path / "eval"
    package.mkdir()
    frozen = training / "sglang/python/sglang/srt/sampling.py"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("numeric_guard = 'v8'\n")
    _write(training / "source_files.json", {
        "schema": "t02-prepared-files-v1", "file_count": 1,
        "files": {"sglang/python/sglang/srt/sampling.py": trained.base._sha(frozen)}})
    manifest = trained.copy_frozen_sglang(training, package)
    assert manifest["file_count"] == 1
    assert (package / "sglang/python/sglang/srt/sampling.py").read_text() == (
        "numeric_guard = 'v8'\n")
    generated = json.loads((package / "sglang_files.json").read_text())
    assert generated["files"] == manifest["files"]


def test_trained_eval_enables_strict_nonfinite_sampling(monkeypatch):
    monkeypatch.delenv("C2KV_STRICT_NONFINITE_SAMPLING", raising=False)
    trained.enable_strict_sampling()
    assert trained.os.environ["C2KV_STRICT_NONFINITE_SAMPLING"] == "1"
