import copy
import hashlib
import json
from pathlib import Path

import pytest

from evidence_expansion import (FAILURE_AUDIT_SCHEMA, LANES, OPERATIONAL_SELECTION_SCOPE,
                                 TRAINED, build_plan, build_shards, select_top_two,
                                 summarize_lane, validate_manifests,
                                 validate_operational_failure_audit, validate_reuse_audit,
                                 trained_cells)


TASKS = [f"multi_turn_base_{index}" for index in range(20)]


def test_parallel_trained_sources_reconstruct_fixed_d20_and_preserve_failure():
    sources = {}
    for part in range(2):
        tasks = TASKS[part * 10:(part + 1) * 10]
        observations = {task: {"stage_outcome": {"runtime_completed": True},
            "official_summary": {"scored": True, "n_total": 1, "n_scored": 1,
                                 "correct_count": 0, "semantic_score": 0.0}} for task in tasks}
        sources[f"C1.part{part}"] = {"lanes": {"C1": {
            "declared_tasks": tasks, "task_observations": observations}}}
    sources["C1.part1"]["lanes"]["C1"]["task_observations"][TASKS[13]]["stage_outcome"] = {
        "runtime_completed": False, "outcome": "runtime_failure_in_denominator"}
    result = trained_cells({"sources": sources}, "C1", TASKS)
    assert [row["task_id"] for row in result] == TASKS
    assert sum(row["status"] == "completed" for row in result) == 19
    assert result[13]["status"] == "runtime_failed" and result[13]["official"] is None
    assert result[-1]["quality_source_id"] == "C1.part1"


def test_parallel_unprepared_sources_are_pending_and_wrong_partition_is_rejected():
    snapshot = {"sources": {"C1.part0": {"lanes": {}}, "C1.part1": {"lanes": {}}}}
    assert all(row["status"] == "pending" for row in trained_cells(snapshot, "C1", TASKS))
    snapshot["sources"]["C1.part0"]["lanes"]["C1"] = {"declared_tasks": TASKS[:9]}
    with pytest.raises(ValueError, match="denominator"):
        trained_cells(snapshot, "C1", TASKS)


def remaining_fixture():
    sources = {}
    for part in range(2):
        tasks = TASKS[part * 10:(part + 1) * 10]
        observations = {task: {"stage_outcome": {"runtime_completed": True},
            "official_summary": {"scored": True, "n_total": 1, "n_scored": 1,
                                 "correct_count": 0, "semantic_score": 0.0}} for task in tasks}
        sources[f"C4_turn.part{part}"] = {"lanes": {"C4_turn": {
            "declared_tasks": tasks, "status": {"state": "failed_no_rerun" if part else "completed"},
            "task_observations": observations}}}
    observations = sources["C4_turn.part1"]["lanes"]["C4_turn"]["task_observations"]
    observations[TASKS[13]]["stage_outcome"] = {"runtime_completed": False,
        "outcome": "runtime_failure_in_denominator"}
    for index, task in enumerate(TASKS[14:]):
        observations[task] = {"stage_outcome": {"outcome": "not_started"}}
        sources[f"C4_turn.remaining{index}"] = {"lanes": {"C4_turn": {
            "declared_tasks": [task], "task_observations": {task: {
                "stage_outcome": {"runtime_completed": True},
                "official_summary": {"scored": True, "n_total": 1, "n_scored": 1,
                                     "correct_count": 1, "semantic_score": 1.0}}}}}}
    return {"sources": sources}


def test_remaining_tasks_overlay_keeps_original_failure_and_quality_owner():
    snapshot = remaining_fixture()
    rows = trained_cells(snapshot, "C4_turn", TASKS)
    assert rows[13]["status"] == "runtime_failed" and rows[13]["official"] is None
    assert rows[13]["quality_source_id"] == "C4_turn.part1"
    assert all(row["status"] == "completed" for row in rows[14:])
    assert rows[14]["quality_source_id"] == "C4_turn.remaining0"
    assert rows[14]["original_never_started_source_id"] == "C4_turn.part1"
    assert [row["task_id"] for row in rows] == TASKS


@pytest.mark.parametrize("invalid", ["failed_task", "duplicate", "missing", "live_parent"])
def test_remaining_tasks_reject_retries_duplicates_missing_and_live_parent(invalid):
    snapshot = remaining_fixture()
    sources = snapshot["sources"]
    if invalid == "failed_task":
        sources["C4_turn.remaining0"]["lanes"]["C4_turn"]["declared_tasks"] = [TASKS[13]]
    elif invalid == "duplicate":
        sources["C4_turn.remaining1"] = copy.deepcopy(sources["C4_turn.remaining0"])
    elif invalid == "missing":
        del sources["C4_turn.remaining0"]
    else:
        sources["C4_turn.part1"]["lanes"]["C4_turn"]["status"]["state"] = "running_fixed_d20"
    with pytest.raises(ValueError, match="never-started"):
        trained_cells(snapshot, "C4_turn", TASKS)


def test_remaining_child_failure_is_not_overwritten_by_official_zero():
    snapshot = remaining_fixture()
    row = snapshot["sources"]["C4_turn.remaining0"]["lanes"]["C4_turn"]
    row["task_observations"][TASKS[14]]["stage_outcome"] = {
        "outcome": "runtime_failure_in_denominator", "runtime_completed": False}
    rows = trained_cells(snapshot, "C4_turn", TASKS)
    assert rows[14]["status"] == "runtime_failed" and rows[14]["official"] is None


def cells(successes, failed=(), pending=(), owner="fixture"):
    rows = []
    for index, task in enumerate(TASKS):
        status = "runtime_failed" if index in failed else "pending" if index in pending else "completed"
        rows.append({"task_id": task, "status": status, "quality_source_id": owner,
                     "status_reason": "runtime_failure_in_denominator" if status == "runtime_failed"
                     else "checker_or_execution_pending" if status == "pending" else None,
                     "official": None if status != "completed" else {
            "scored": True, "n_total": 1, "n_scored": 1, "correct_count": int(index in successes),
            "semantic_score": float(index in successes)}})
    return rows


def lanes():
    return {name: summarize_lane(cells(range(index)), TASKS) for index, name in enumerate(LANES)}


def cost(extra, seconds, *, successes=range(8), covered=20):
    return {"covered_tasks": covered, "extra_generations": extra, "selector_seconds": seconds,
            "evidence": [{"task_id": task, "owner": "fixture",
                          "expected_correct_count": int(index in successes),
                          "steps_sha256": "a" * 64, "official_sha256": "b" * 64}
                         for index, task in enumerate(TASKS[:covered])]}


def _write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def typed_failure_audit(tmp_path: Path, *, owner="fixture") -> dict:
    error = {"type": "SGLangExtractionBudgetExhausted",
             "message": "C2KV_EXTRACTION_BUDGET_EXHAUSTED: bounded"}
    stage = {"path": "/remote/C4_turn/stage_manifest.json", "sha256": "a" * 64}
    steps_source = {"path": "/remote/C4_turn/steps.jsonl", "sha256": "b" * 64}
    source = {"schema": "experiment3-trained-d20-runtime-failure-v1",
              "controller": "C4_turn", "task_id": TASKS[19],
              "decision_key": "turn-0/step-18", "status": "runtime_failed_not_quality_zero",
              "error": error, "stage_manifest": stage, "steps": steps_source,
              "source_shard_state": "failed_no_rerun", "model_artifact_unchanged": True}
    source_path = tmp_path / "typed_failure.json"
    source_sha = _write_json(source_path, source)
    return {"schema": FAILURE_AUDIT_SCHEMA, "status": "audited",
            "selection_scope": {"id": OPERATIONAL_SELECTION_SCOPE, "fixed_denominator": 20,
                                "counterfactual_bounds_used_for_promotion": False,
                                "automatic_retries": 0, "automatic_reruns": 0},
            "entries": [{"lane": "C4_turn", "task_id": TASKS[19],
                         "quality_source_id": owner,
                         "classification": "typed_extraction_budget_deployment_failure",
                         "selection_outcome": "operational_non_success",
                         "official_score_status": "unobserved_runtime_failure",
                         "official_zero_imputed": False, "retry_or_rerun": False,
                         "stage_manifest": stage,
                         "steps": {**steps_source, "final_status": "failed",
                                   "decision_key": "turn-0/step-18", "error": error},
                         "source_evidence": {"path": str(source_path), "sha256": source_sha}}]}


def historical_failure_audit(tmp_path: Path, *, owner="fixture") -> dict:
    error = {"type": "SGLangEventNativeError",
             "message": "SGLang native_generate returned HTTP 400"}
    stage_sha, steps_sha = "c" * 64, "d" * 64
    source = {"schema": "evidence-sets-c2-extraction-budget-failure-v1", "lane": "C2",
              "remote_root": "/remote/C2", "root_cause": "finite extraction budget exhausted",
              "failure": {"task_id": TASKS[19], "decision_key": "turn-1/step-12",
                          "stage_outcome": "runtime_failure_in_denominator",
                          "runtime_completed": False, "official_zero_accepted_for_quality": False,
                          "worker_returncode": 0, "server_returncode": 1,
                          "engine_error": "C2KV_EXTRACTION_BUDGET_EXHAUSTED: bounded",
                          "step_error": error},
              "extraction_budget": {"configured_task_cap": 1152,
                                    "physical_extractions_after_failed_attempt": 1152},
              "remote_artifact_sha256": {"stage_manifest.json": stage_sha,
                                         "server/steps.jsonl": steps_sha}}
    source_path = tmp_path / "historical_failure.json"
    source_sha = _write_json(source_path, source)
    return {"schema": FAILURE_AUDIT_SCHEMA, "status": "audited",
            "selection_scope": {"id": OPERATIONAL_SELECTION_SCOPE, "fixed_denominator": 20,
                                "counterfactual_bounds_used_for_promotion": False,
                                "automatic_retries": 0, "automatic_reruns": 0},
            "entries": [{"lane": "C2", "task_id": TASKS[19],
                         "quality_source_id": owner,
                         "classification": (
                             "historical_validated_extraction_budget_deployment_failure"),
                         "selection_outcome": "operational_non_success",
                         "official_score_status": "unobserved_runtime_failure",
                         "official_zero_imputed": False, "retry_or_rerun": False,
                         "stage_manifest": {"path": "/remote/C2/lanes/C2/results/stage_manifest.json",
                                            "sha256": stage_sha},
                         "steps": {"path": "/remote/C2/lanes/C2/results/task_shards/"
                                            + TASKS[19] + "/server/steps.jsonl",
                                   "sha256": steps_sha, "final_status": "failed",
                                   "decision_key": "turn-1/step-12", "error": error},
                         "source_evidence": {"path": str(source_path), "sha256": source_sha}}]}


def test_pending_methods_block_premature_selection():
    data = lanes()
    data["C1"] = summarize_lane(cells((), pending=(0,)), TASKS)
    result = select_top_two(data)
    assert result["phase"] == "waiting_for_d20"
    assert result["selected"] == []


def test_unaudited_failure_blocks_promotion_without_imputing_official_zero():
    row = summarize_lane(cells(range(6), failed=(19,)), TASKS)
    assert row["success_count_bounds"] == [6, 7]
    assert not row["clean_d20"]
    assert row["all_cells_terminal"]
    data = lanes()
    data["C2"] = row
    result = select_top_two(data)
    assert result["phase"] == "unaudited_runtime_failure_blocks_promotion"
    assert result["blocked_lanes"] == {"C2": [TASKS[19]]}
    assert row["quality_cells"][19]["official"] is None


@pytest.mark.parametrize("kind", ["typed", "historical"])
def test_audited_in_contract_failure_is_ranked_as_operational_non_success(
        tmp_path, kind):
    data = lanes()
    lane = "C4_turn" if kind == "typed" else "C2"
    audit = typed_failure_audit(tmp_path) if kind == "typed" else historical_failure_audit(tmp_path)
    data[lane] = summarize_lane(cells(range(10), failed=(19,)), TASKS)
    receipt = validate_operational_failure_audit(audit, data)
    result = select_top_two(data)
    assert result["phase"] == "promotion_ready" and result["selected"][0] == lane
    assert data[lane]["success_count_bounds"] == [10, 11]
    assert data[lane]["operational_success_count"] == 10
    assert data[lane]["operational_selection_eligible"] is True
    assert data[lane]["quality_cells"][19]["official"] is None
    assert data[lane]["quality_cells"][19]["operational_selection"]["outcome"] == "non_success"
    assert receipt["counterfactual_bounds_used_for_promotion"] is False


@pytest.mark.parametrize("mutation", ["owner", "hash", "unknown_contract"])
def test_failure_audit_rejects_wrong_owner_hash_and_unknown_contract(tmp_path, mutation):
    data = lanes()
    data["C4_turn"] = summarize_lane(cells(range(6), failed=(19,)), TASKS)
    audit = typed_failure_audit(tmp_path)
    if mutation == "owner":
        audit["entries"][0]["quality_source_id"] = "different"
    elif mutation == "hash":
        audit["entries"][0]["source_evidence"]["sha256"] = "0" * 64
    else:
        path = Path(audit["entries"][0]["source_evidence"]["path"])
        audit["entries"][0]["source_evidence"]["sha256"] = _write_json(
            path, {"schema": "unreviewed-runtime-error-v1"})
    with pytest.raises(ValueError, match="owner|hash|approved documented"):
        validate_operational_failure_audit(audit, data)


def test_two_complementary_leaders_both_advance():
    data = lanes()
    data["C0"] = summarize_lane(cells(range(8)), TASKS)
    data["C1"] = summarize_lane(cells(range(1, 9)), TASKS)
    assert select_top_two(data)["selected"] == ["C0", "C1"]


def test_three_complementary_ties_are_not_broken_by_lane_id():
    data = lanes()
    for index, lane in enumerate(("C0", "C1", "C2")):
        data[lane] = summarize_lane(cells(range(index, index + 8)), TASKS)
    result = select_top_two(data)
    assert result["phase"] == "complementary_tie_at_cutoff"
    assert result["selected"] == []


def test_identical_success_sets_use_extra_generation_then_selector_cost():
    data = lanes()
    for name in ("C0", "C1", "C2"):
        data[name] = summarize_lane(cells(range(8)), TASKS)
    assert select_top_two(data)["phase"] == "waiting_for_measured_tie_cost"
    result = select_top_two(data, {"C0": cost(10, 1), "C1": cost(9, 100), "C2": cost(9, 2)})
    assert result["selected"] == ["C2", "C1"]


def test_more_than_one_success_group_keeps_complementarity_before_cost():
    data = lanes()
    data["C0"] = summarize_lane(cells(range(8)), TASKS)
    data["C1"] = summarize_lane(cells(range(8)), TASKS)
    data["C2"] = summarize_lane(cells(range(1, 9)), TASKS)
    result = select_top_two(data, {"C0": cost(10, 1), "C1": cost(9, 1)})
    assert result["selected"] == ["C1", "C2"]


def test_distinct_extra_generation_counts_do_not_need_secondary_latency():
    data = lanes()
    for name in ("C0", "C1", "C2"):
        data[name] = summarize_lane(cells(range(8)), TASKS)
    result = select_top_two(data, {"C0": cost(10, None), "C1": cost(9, None), "C2": cost(8, None)})
    assert result["selected"] == ["C2", "C1"]


def _terminal_actual_tie_fixture(tmp_path):
    data = {name: summarize_lane(cells(range(6)), TASKS) for name in LANES}
    data["C0"] = summarize_lane(cells(range(8)), TASKS)
    data["C2"] = summarize_lane(cells(range(6), failed=(19,)), TASKS)
    data["C4_turn"] = summarize_lane(cells(range(6), failed=(19,)), TASKS)
    audit = historical_failure_audit(tmp_path)
    audit["entries"].extend(typed_failure_audit(tmp_path)["entries"])
    validate_operational_failure_audit(audit, data)
    costs = {
        "C1": cost(15, 0, successes=range(6)),
        "C2": cost(47, 82, successes=range(6), covered=19),
        "C3": cost(72, 195, successes=range(6)),
        "C4_turn": cost(61, None, successes=range(6), covered=19),
        "C4_task": cost(42, None, successes=range(6)),
        "C5": cost(0, 0, successes=range(6)),
    }
    return data, costs


def test_unique_full_extra_strictly_below_partial_bounds_resolves_tie(tmp_path):
    data, costs = _terminal_actual_tie_fixture(tmp_path)
    result = select_top_two(data, costs)
    assert result["phase"] == "promotion_ready"
    assert result["selected"] == ["C0", "C5"]
    proof = result["tie_resolution"][0]["cost_proof"]
    assert proof["winner"] == "C5"
    partial = {row["controller"]: row for row in proof["candidates"]
               if row["measurement"] == "partial_lower_bound"}
    assert partial["C2"]["coverage"] == partial["C4_turn"]["coverage"] == 19
    assert partial["C2"]["extra_generations_role"] == "lower_bound"
    assert partial["C2"]["missing_audited_runtime_failures"] == [TASKS[19]]
    assert len(partial["C2"]["evidence_rows_sha256"]) == 64


def test_equal_partial_extra_lower_bound_cannot_be_broken_by_selector_time(tmp_path):
    data, costs = _terminal_actual_tie_fixture(tmp_path)
    costs["C5"] = cost(47, 0, successes=range(6))
    costs["C1"] = cost(60, 0, successes=range(6))
    costs["C4_task"] = cost(70, None, successes=range(6))
    result = select_top_two(data, costs)
    assert result["phase"] == "waiting_for_measured_tie_cost"
    assert result["tie_cost_evidence"][
        "blocked_by_partial_lower_bound_at_or_below_best_full_extra"] is True


def manifests():
    return ({"task_ids": TASKS},
            {"task_ids": [f"multi_turn_base_{index}" for index in range(128)]},
            {"task_ids": [f"multi_turn_base_{index}" for index in range(128, 256)]},
            {"task_ids": ["multi_turn_base_300"]})


def test_only_remaining108_are_scheduled_and_evaluation_groups_are_isolated():
    docs = manifests()
    assert len(validate_manifests(*docs)) == 108
    assert not set(validate_manifests(*docs)) & set(TASKS)
    leaked = copy.deepcopy(docs)
    leaked[-1]["task_ids"] = ["multi_turn_long_context_0"]
    with pytest.raises(ValueError, match="Training overlaps"):
        validate_manifests(*leaked)


def test_realistic_pretraining_plan_waits_without_placeholder_models_or_launch():
    untrained = {"lanes": {name: {"quality_cells": cells(range(6))}
                           for name in LANES if name not in TRAINED}}
    result = build_plan(untrained, {"sources": {}}, *manifests())
    assert result["phase"] == "waiting_for_d20"
    assert result["promotion"]["pending_lanes"] == list(TRAINED)
    assert result["new_execution_count"] == 0
    assert result["launch_authorized"] is False
    assert result["f128_used_for_selection"] is False


def test_tie_costs_must_come_from_the_current_quality_owners():
    untrained = {"lanes": {name: {"quality_cells": [
        {**row, "quality_source_id": "current"} for row in cells(range(6))]}
        for name in LANES if name not in TRAINED}}
    costs = {"C0": {**cost(1, 1), "evidence": [
        {"task_id": task, "owner": "obsolete", "expected_correct_count": int(index < 6)}
        for index, task in enumerate(TASKS)]}}
    with pytest.raises(ValueError, match="owner differs"):
        build_plan(untrained, {"sources": {}}, *manifests(), costs=costs)
    for row in costs["C0"]["evidence"]:
        row["owner"] = "current"
    assert build_plan(untrained, {"sources": {}}, *manifests(), costs=costs)["phase"] == "waiting_for_d20"


def test_existing_development_release_group_overlap_is_recorded_not_repartitioned():
    docs = list(manifests())
    docs[2] = {"task_ids": [f"multi_turn_long_context_{index}" for index in range(128)]}
    untrained = {"lanes": {name: {"quality_cells": cells(range(6))}
                           for name in LANES if name not in TRAINED}}
    result = build_plan(untrained, {"sources": {}}, *docs)
    assert result["existing_d128_f128_shared_group_ids"] == list(range(128))
    assert result["next_manifests"] == {}


def test_duplicate_cells_and_boolean_scores_are_rejected():
    duplicate = cells(range(6))
    duplicate[1]["task_id"] = duplicate[0]["task_id"]
    with pytest.raises(ValueError, match="ordered D20"):
        summarize_lane(duplicate, TASKS)
    invalid = cells(range(6))
    invalid[0]["official"]["correct_count"] = True
    with pytest.raises(ValueError, match="not boolean"):
        summarize_lane(invalid, TASKS)


def test_six_shards_use_all_authorized_cards_without_duplicate_executions():
    remaining = validate_manifests(*manifests())
    expanded = {lane: {"task_ids": remaining} for lane in ("C0", "C4_task")}
    shards = build_shards(expanded)
    assert {row["preferred_physical_device"] for row in shards} == {0, 1, 2, 3, 4, 6}
    assert sum(row["task_budget"] for row in shards) == 216
    for lane in expanded:
        tasks = [task for row in shards if row["controller"] == lane for task in row["task_ids"]]
        assert len(tasks) == len(set(tasks)) == 108
        assert set(tasks) == set(remaining)
    assert all(row["launch_authorized"] is False for row in shards)


def test_reuse_audit_binds_each_task_owner_and_preserves_mixed_version_provenance():
    data = lanes()
    for cell in data["C0"]["quality_cells"]:
        cell["quality_source_id"] = "original"
    conclusion = "semantic_nonactivation_compatible_20_of_20"
    audit = {"schema": "evidence-sets-d20-runtime-compatibility-v1",
             "target": {"source_id": "failed_repair_v1"},
             "conclusion": {"result_reuse_supported": True, "lanes": {"C0": conclusion}},
             "lanes": {"C0": {"classification": conclusion, "tasks": [
                 {"task_id": task, "quality_source_id": "original",
                  "classification": "semantic_nonactivation_compatible"} for task in TASKS]}}}
    result = validate_reuse_audit(audit, data, TASKS)
    assert result["audited_lanes"] == ["C0"]
    assert result["single_frozen_config_d128_claim_allowed"] is False
    audit["lanes"]["C0"]["tasks"][0]["quality_source_id"] = "different_owner"
    with pytest.raises(ValueError, match="owner or conclusion differs"):
        validate_reuse_audit(audit, data, TASKS)


def test_incomplete_reuse_audit_does_not_approve_whole_d20():
    data = lanes()
    conclusion = "semantic_nonactivation_compatible_20_of_20"
    audit = {"schema": "evidence-sets-d20-runtime-compatibility-v1",
             "target": {"source_id": "failed_repair_v1"},
             "conclusion": {"result_reuse_supported": True, "lanes": {"C0": conclusion}},
             "lanes": {"C0": {"classification": conclusion, "tasks": []}}}
    with pytest.raises(ValueError, match="different tasks"):
        validate_reuse_audit(audit, data, TASKS)
