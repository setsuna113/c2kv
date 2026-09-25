import json
from pathlib import Path

import pytest

from benchmarks.paper import runner
from benchmarks.paper.experiment2_smoke import smoke_command, scored_result


def test_smoke_preserves_budget_and_routes_task_to_both_clients(tmp_path):
    config = json.loads(Path(__file__).with_name("experiment2_cuda.json").read_text())
    plan, profile = runner.prepare(config, tmp_path / "plan", tmp_path / "engine")
    for cell in plan:
        if cell["benchmark"] != "bfcl_base":
            continue
        command = smoke_command(config, cell, tmp_path / cell["cell_id"], profile,
                                "multi_turn_base_26")
        flag = "--task-ids" if runner.is_native_arm(cell["arm"]) else "--run-ids"
        assert command[command.index(flag) + 1] == "multi_turn_base_26"
        if cell.get("history_budget_tokens") and get_history_flag(command):
            assert command[command.index("--history-kv-target-tokens") + 1] == "768"


def get_history_flag(command):
    return "--history-kv-target-tokens" in command


@pytest.mark.parametrize("change", [{"n_scored": 0}, {"n_scored": 200},
                                    {"n_harness_failures": 1},
                                    {"completion_ledger": {"remaining": ["missing"]}}])
def test_smoke_does_not_accept_incomplete_or_full_suite_scores(tmp_path, change):
    summary = dict(n_scored=1, semantic_score=0.0, **{})
    summary.update(change)
    (tmp_path / "summary_full.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError):
        scored_result(tmp_path, "full")


def test_task_failure_is_reported_without_inventing_quality_success(tmp_path):
    (tmp_path / "summary_c2kv_native_r4.json").write_text(json.dumps(
        {"n_scored": 1, "semantic_score": 0.0, "n_method_failures": 1}))
    result = scored_result(tmp_path, "c2kv_native_r4")
    assert result["semantic_score"] == 0.0
    assert result["n_method_failures"] == 1
