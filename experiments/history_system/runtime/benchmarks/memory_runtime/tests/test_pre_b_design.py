from __future__ import annotations

import copy

import pytest

from benchmarks.memory_runtime import pre_b_design as design


def test_stages_freeze_routes_tasks_profiles_and_independent_caps():
    p3 = design.load_stage("pre-b-p3")
    assert p3["task_ids"] == list(design.TASK_IDS)
    assert [item[0] for item in p3["variants"]] == list(design.P3_VARIANTS)
    assert p3["history_budget_bytes"] == 113246208
    assert p3["maximum_wall_seconds"] == 93600
    assert p3["maximum_generation_attempts_per_task"] == 96
    assert p3["maximum_generation_attempts_per_arm"] == 768
    assert p3["maximum_generation_attempts"] == 6144
    assert p3["maximum_extraction_attempts"] == 36864
    assert p3["selected_method"] is None
    assert p3["selected_method_status"] == "pending_p3"

    p4a = design.load_stage("pre-b-p4a")
    assert [item[0] for item in p4a["variants"]] == [
        "ac_exact_persistent", "raw_recency", "raw_exact_shared"]
    assert p4a["selected_method_status"] == design.PROVISIONAL_METHOD_STATUS
    assert p4a["history_budget_bytes"] == 226492416
    assert p4a["maximum_extraction_attempts_by_arm"] == {
        "ac_exact_persistent": 9216, "raw_recency": 0,
        "raw_exact_shared": 0,
    }

    p4b = design.load_stage(
        "pre-b-p4b", selected_method="ac_exact_persistent",
        selected_method_status=design.FROZEN_METHOD_STATUS)
    assert p4b["task_ids"] == list(design.G460_TASK_IDS)
    assert [item[0] for item in p4b["variants"]] == [
        "full", "ac_exact_persistent", "raw_recency", "raw_exact_shared"]
    assert p4b["selected_method_status"] == design.FROZEN_METHOD_STATUS
    assert p4b["maximum_generation_attempts_per_arm"] == 384
    assert p4b["maximum_extraction_attempts_by_arm"]["ac_exact_persistent"] == 6144


@pytest.mark.parametrize("selected", ["full", "ac_full_shared", "raw_recency"])
def test_p4_rejects_non_gist_selected_methods(selected):
    with pytest.raises(ValueError, match="incumbent or the frozen P2 candidate"):
        design.load_stage(
            "pre-b-p4a", selected_method=selected,
            selected_method_status=design.FROZEN_METHOD_STATUS)


def test_design_validation_rejects_changed_stage_and_data_contracts():
    spec = design.load_design()
    changed_cap = copy.deepcopy(spec)
    changed_cap["stages"]["pre-b-p3"][
        "maximum_extraction_attempts_by_arm"]["ac_protect"] += 1
    with pytest.raises(ValueError, match="gist extraction cap"):
        design.validate_design(changed_cap)

    changed_data = copy.deepcopy(spec)
    changed_data["data_contract"]["official_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="data identity"):
        design.validate_design(changed_data)


def _generation_row(task_id, before, indices, *, task_before):
    return {
        "eval_context": {"task_id": task_id},
        "generation_budget": {
            "limit": 8,
            "consumed_before": before,
            "consumed_after": before + len(indices),
            "attempt_indices": indices,
            "per_task_limit": 2,
            "task_id": task_id,
            "task_consumed_before": task_before,
            "task_consumed_after": task_before + len(indices),
        },
    }


def test_generation_validator_closes_process_and_per_task_ledgers():
    rows = [
        _generation_row("a", 0, [1], task_before=0),
        _generation_row("b", 1, [2], task_before=0),
        _generation_row("a", 2, [3], task_before=1),
    ]
    assert design.validate_generation_budget_rows(rows, 8, 2) == (
        3, {"a": 2, "b": 1})

    invalid = copy.deepcopy(rows)
    invalid[-1]["generation_budget"]["task_consumed_before"] = 0
    with pytest.raises(ValueError, match="per-task generation ledger"):
        design.validate_generation_budget_rows(invalid, 8, 2)


def test_zero_extraction_validator_requires_explicit_zero_telemetry():
    rows = [{
        "extraction_telemetry": {
            "summary": {"producer_calls": 0}, "events": []},
    }]
    assert design.validate_zero_extraction_rows(rows) == 0
    rows[0]["extraction_telemetry"]["summary"]["producer_calls"] = 1
    with pytest.raises(ValueError, match="used extraction"):
        design.validate_zero_extraction_rows(rows)
