"""CPU-only tests for exact-state T02 recovery planning."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "runtime" / "python"))
sys.path.insert(0, str(HERE / "runtime"))

import t02
import t02_recovery_plan as recovery
from benchmarks.memory_runtime.recovery.set_training import _read_dataset


def _candidate(index: int) -> dict:
    task_id = f"task-{index // 2}"
    draft_kind = "call" if index % 2 else "stop"
    first, second = f"candidate-{index}-0", f"candidate-{index}-1"
    return {
        "schema": t02.CANDIDATE_SCHEMA,
        "state_id": f"old-state-{index}",
        "benchmark": "bfcl",
        "task_id": task_id,
        "task_group_id": task_id,
        "decision_key": f"turn-0/step-{index}",
        "set_selector": "local_llm",
        "draft_kind": draft_kind,
        "risk_bucket": "high" if index % 2 else "low",
        "previous_turn_valid": True,
        "q": {"task": f"task {index}", "recent_feedback": "observed"},
        "draft": {
            "kind": draft_kind,
            "text": "" if draft_kind == "call" else "done",
            "tool_calls": ([{"name": "lookup", "arguments": {"id": index}}]
                           if draft_kind == "call" else []),
            "parse_ok": True,
            "token_logprobs": [-0.1],
        },
        "candidates": [
            {"candidate_id": first, "source_id": f"source-{index}-0", "rank": 0,
             "feasible": True, "text": "first"},
            {"candidate_id": second, "source_id": f"source-{index}-1", "rank": 1,
             "feasible": True, "text": "second"},
        ],
        "allowed_actions": [
            {"action_id": "none", "candidate_ids": []},
            {"action_id": "top", "candidate_ids": [first]},
            {"action_id": "other", "candidate_ids": [second]},
            {"action_id": "both", "candidate_ids": [first, second]},
        ],
        "local_llm_selected_ids": [first, second] if index % 3 == 0 else [second],
    }


def _manifests(tmp_path: Path) -> dict[str, Path]:
    d128, f128 = tmp_path / "D128.json", tmp_path / "F128.json"
    d128.write_text(json.dumps({"task_ids": ["forbidden-d"]}), encoding="utf-8")
    f128.write_text(json.dumps({"task_ids": ["forbidden-f"]}), encoding="utf-8")
    return {"D128": d128, "F128": f128}


def _source(tmp_path: Path):
    plan = t02.build_plan(
        [_candidate(index) for index in range(6)],
        forbidden_manifest_paths=_manifests(tmp_path),
        frozen_subsequent_policy={"name": "frozen", "version": 1},
        seed=0,
        target_states=6,
        train_states=4,
        min_task_groups=3,
        max_complete_branch_executions=18,
    )
    calibration = [state for state in plan["states"] if state["split"] == "calibration"]
    complete = calibration[0]
    partial = next(state for state in plan["states"] if state["split"] == "train")
    result_rows = [
        {"state_id": complete["state_id"], "branch_id": branch,
         "execution_status": "complete"}
        for branch in t02.BRANCH_IDS
    ] + [{"state_id": partial["state_id"], "branch_id": "A0", "execution_status": "complete"}]
    partials = {
        "npu0": {
            "results": result_rows,
            "complete_branch_executions": len(result_rows),
            "plan_sha256": t02._digest(plan),
        }
    }
    task_ids = sorted({state["task_id"] for state in plan["states"]})
    ledger = {
        "schema": "test-ledger",
        "tasks": [
            {"task_id": task_id, "task_index": index, "status": "completed"}
            for index, task_id in enumerate(task_ids)
        ],
        "candidates": [],
    }
    contract = recovery.derive_recovery_contract(
        plan, ledger, partials,
        complete_branch_cap=16,
        documented_prior_complete_branches=0,
        additional_v6_complete_branches=0,
        incomplete_started_branch_attempts=0,
        prior_source_starts=6,
    )
    return plan, contract


def _fresh_candidate(state: dict, index: int) -> dict:
    value = copy.deepcopy(state)
    value["schema"] = t02.CANDIDATE_SCHEMA
    value["state_id"] = f"new-state-{index}"
    for key in ("branches", "split", "branch_selection_uses_future_outcome"):
        value.pop(key, None)
    return value


def test_contract_drops_one_unexecuted_calibration_slot_and_accounts_branches(tmp_path):
    plan, contract = _source(tmp_path)
    assert contract["target"] == {
        "combined_exact_state_count": 5,
        "preserved_exact_state_count": 1,
        "new_exact_state_count": 4,
        "split_counts": {"train": 4, "calibration": 1},
        "task_group_count": 3,
    }
    assert contract["budget"]["v6_completed_branch_executions"] == 4
    assert contract["budget"]["v6_completed_triplet_branches_reused"] == 3
    assert contract["budget"]["v6_partial_completed_branches_cost_only"] == 1
    assert contract["budget"]["new_complete_branch_executions"] == 12
    assert contract["budget"]["cumulative_complete_branches"] == 16
    assert contract["dropped_state"]["split"] == "calibration"
    assert contract["dropped_state"]["source_plan_index"] == max(
        index for index, state in enumerate(plan["states"])
        if state["split"] == "calibration"
    )


def test_recovery_plan_requires_every_frozen_slot_and_uses_new_state_ids(tmp_path):
    plan, contract = _source(tmp_path)
    by_slot = {(state["task_id"], state["draft_kind"]): state for state in plan["states"]}
    candidates = [
        _fresh_candidate(by_slot[(slot["task_id"], slot["draft_kind"])], index)
        for index, slot in enumerate(contract["recapture_slots"])
    ]
    recovered = recovery.build_recovery_plan(
        contract, candidates, source_plan=plan,
        frozen_subsequent_policy={"name": "recovery", "version": 1},
    )
    assert recovered["target_states"] == 4
    assert recovered["target_train_states"] == 4
    assert recovered["max_complete_branch_executions"] == 12
    assert {state["state_id"] for state in recovered["states"]} == {
        f"new-state-{index}" for index in range(4)
    }
    preserved = contract["preserved_states"][0]
    preserved_source = by_slot[(preserved["task_id"], preserved["draft_kind"])]
    with_irrelevant_preserved = recovery.build_recovery_plan(
        contract, [*candidates, _fresh_candidate(preserved_source, 99)],
        source_plan=plan, frozen_subsequent_policy={"name": "recovery", "version": 1},
    )
    assert [row["state_id"] for row in with_irrelevant_preserved["states"]] == [
        row["state_id"] for row in recovered["states"]
    ]
    with pytest.raises(ValueError, match="missing or ambiguous recovery slots"):
        recovery.build_recovery_plan(
            contract, candidates[:-1], source_plan=plan,
            frozen_subsequent_policy={"name": "recovery", "version": 1},
        )
    changed_source = copy.deepcopy(plan)
    changed_source["extra"] = "digest drift"
    with pytest.raises(ValueError, match="source plan digest"):
        recovery.build_recovery_plan(
            contract, candidates, source_plan=changed_source,
            frozen_subsequent_policy={"name": "recovery", "version": 1},
        )


def test_label_merge_keeps_both_plan_provenances_and_filters_partial_old_rows(tmp_path):
    plan, contract = _source(tmp_path)
    preserved = contract["preserved_states"][0]
    preserved_row = {
        "state_id": preserved["state_id"], "task_id": preserved["task_id"],
        "draft_kind": preserved["draft_kind"], "task_group_id": preserved["task_group_id"],
        "split": preserved["split"],
    }
    extra_old = {
        "state_id": "partial-old", "task_id": "ignored", "draft_kind": "call",
        "task_group_id": "ignored", "split": "train",
    }
    new_rows = [
        {
            "state_id": f"fresh-{index}", "task_id": slot["task_id"],
            "draft_kind": slot["draft_kind"], "task_group_id": slot["task_group_id"],
            "split": slot["split"],
        }
        for index, slot in enumerate(contract["recapture_slots"])
    ]
    merged = recovery.merge_label_partitions(
        contract,
        {"plan_sha256": "old-plan", "rows": [preserved_row, extra_old]},
        {"plan_sha256": "new-plan", "rows": new_rows},
    )
    assert merged["schema"] == recovery.COMBINED_LABEL_SCHEMA
    assert merged["state_count"] == 5
    assert [part["plan_sha256"] for part in merged["partitions"]] == [
        "old-plan", "new-plan",
    ]
    assert merged["old_result_plan_sha256_rewritten"] is False
    assert "partial-old" not in {row["state_id"] for row in merged["rows"]}

    prefill = {"layer": 34, "readout": "decoder_layer_output"}
    for row in merged["rows"]:
        row["q"] = {"prefill_contract": prefill}
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(merged), encoding="utf-8")
    loaded_rows, source = _read_dataset(labels_path)
    assert len(loaded_rows) == 5
    assert source["dataset_schema"] == recovery.COMBINED_LABEL_SCHEMA
    assert recovery.extract_prefill_contract(merged) == prefill


def test_training_lane_receipt_binds_118_state_labels_and_manifest(tmp_path):
    labels = {"schema": recovery.COMBINED_LABEL_SCHEMA, "state_count": 118, "rows": []}
    labels_path, artifact, manifest = (
        tmp_path / "labels.json", tmp_path / "c1.json", tmp_path / "source_files.json"
    )
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    artifact.write_text("{}", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")
    receipt = recovery.training_lane_receipt(labels_path, artifact, manifest, "C1")
    assert receipt["schema"] == "t02-training-lane-receipt-v1"
    assert receipt["status"] == "completed"
    assert receipt["state_count"] == 118
