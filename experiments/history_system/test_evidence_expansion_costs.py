import copy

import pytest

from evidence_expansion_costs import aggregate, requests_for_plan, step_costs


def record():
    calls = [{"capability": "choose_action", "purpose": "recovery_action_selection",
              "latency_seconds": 5},
             {"capability": "model_load", "role": "selector", "latency_seconds": 3},
             {"capability": "embed", "purpose": "query", "latency_seconds": 100}]
    check = {"selection": {"selector": "local_llm"}, "selection_model_calls": calls}
    return {"status": "ok", "session_id": "task", "decision_key": "turn0/step0",
            "generation_trace": [{"phase": "draft", "status": "completed"},
                                 {"phase": "regeneration", "status": "completed"}],
            "recovery_checks": [check], "exact_recovery": copy.deepcopy(check),
            "recovery_rounds": [copy.deepcopy(check)],
            "controller_timing": {"reconsider_seconds": 123}}


def test_actual_extra_generation_and_selection_cost_are_not_double_counted():
    cost = step_costs([record()])
    assert cost["extra_generations"] == 1
    assert cost["selector_model_call_seconds"] == 5
    assert cost["selector_cold_load_seconds"] == 3
    assert cost["selector_model_calls"] == 1
    assert cost["reconsider_seconds_including_retrieval"] == 123


def test_runtime_failure_and_duplicate_decisions_cannot_enter_tie_costs():
    with pytest.raises(ValueError, match="Duplicate"):
        step_costs([record(), record()])
    broken = record()
    broken["status"] = "failed"
    with pytest.raises(ValueError, match="non-ok"):
        step_costs([broken])


def test_reranker_load_in_input_preparation_is_counted_once():
    row = record()
    row["recovery_checks"][0]["selection_model_calls"] = [
        {"capability": "model_load", "role": "reranker", "latency_seconds": 3},
        {"capability": "reranker_input_limit", "purpose": "recovery_evidence_relevance_input_limit",
         "latency_seconds": 4},
        {"capability": "rerank", "purpose": "recovery_evidence_relevance", "latency_seconds": 2}]
    costs = {"lane": "C2", "task_id": "one", **step_costs([row])}
    result = aggregate({"tasks": [costs]}, ["C2"])["C2"]
    assert result["selector_seconds"] == 6
    assert result["selector_cold_load_seconds"] == 3


def test_invalid_selection_latency_is_not_silently_zero():
    broken = record()
    broken["recovery_checks"][0]["selection_model_calls"][0]["latency_seconds"] = None
    with pytest.raises(ValueError, match="invalid measured latency"):
        step_costs([broken])


def test_partial_coverage_and_unisolated_c4_embedding_latency_remain_visible():
    row = {"lane": "C4_turn", "task_id": "one", **step_costs([record()])}
    result = aggregate({"tasks": [row]}, ["C4_turn", "C0"])
    assert result["C4_turn"]["covered_tasks"] == 1
    assert result["C4_turn"]["selector_seconds"] is None
    assert result["C0"]["covered_tasks"] == 0


def test_requests_follow_only_completed_quality_owners_and_isolated_c2_layout():
    plan = {"lanes": {"C2": {"quality_cells": [
        {"task_id": "multi_turn_base_0", "status": "completed", "quality_source_id": "c2_remaining_v1",
         "official": {"correct_count": 1}},
        {"task_id": "multi_turn_base_1", "status": "runtime_failed"}]}}, "trained_snapshot": {}}
    untrained = {"sources": {"c2_remaining_v1": {"remote_root": "/remote/c2"}}}
    rows = requests_for_plan(plan, untrained)
    assert len(rows) == 1
    assert "/task_attempts/multi_turn_base_0/results/task_shards/" in rows[0]["steps"]


@pytest.mark.parametrize("owner", ["C1", "C1.part1"])
def test_trained_costs_follow_the_actual_shard_owner(owner):
    plan = {"lanes": {"C1": {"quality_cells": [{"task_id": "multi_turn_base_0",
        "status": "completed", "quality_source_id": owner, "official": {"correct_count": 0}}]}},
        "trained_snapshot": {"sources": {owner: {"remote_root": "/remote/exact-owner"}}}}
    assert requests_for_plan(plan, {})[0]["steps"].startswith("/remote/exact-owner/lanes/C1/")
