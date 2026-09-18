import copy
import json
from pathlib import Path
import tempfile

import pytest

from benchmarks.arms import get_arm
from benchmarks.paper import c1 as paper_c1
from benchmarks.paper.c1 import (ARM, controller_oom_message, controller_step_failure, replay_task_id,
                                 selected_tasks, select_arm, summarize_scores)
from benchmarks.paper.runner import DEFAULT_CONFIG, prepare, server_command


def test_native_arm_requires_real_controller_and_keeps_bare_c2kv():
    assert get_arm(ARM).native_controller == "c1_t02"
    assert get_arm(ARM).ratio == 8
    assert get_arm("c2kv4").ratio == 4
    config = json.loads(DEFAULT_CONFIG.read_text())
    command = server_command(config, Path("sglang"), ARM)
    assert "--enable-return-hidden-states" in command
    assert "--c2kv-shadow-feature-layer" in command
    assert "--disable-cuda-graph" not in command
    assert "--enable-return-hidden-states" not in server_command(config, Path("sglang"), "full")


def test_append_final_arm_preserves_old_cells_and_completed_artifacts():
    config = json.loads(DEFAULT_CONFIG.read_text())
    previous = copy.deepcopy(config)
    previous["methods"] = previous["methods"][:-2]   # drop the C1 system and its ratio-4 ablation
    previous.pop("c1")
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        old_plan, _ = prepare(previous, output, output / "sglang")
        completed = output / "closed_loop" / old_plan[0]["cell_id"] / "complete.json"
        completed.parent.mkdir(parents=True)
        completed.write_text('{"old_result": true}\n')
        new_plan, _ = prepare(config, output, output / "sglang")
        assert [row["cell_id"] for row in new_plan[:len(old_plan)]] == [row["cell_id"] for row in old_plan]
        assert completed.read_text() == '{"old_result": true}\n'
        assert json.loads((output / "config.before_c1_extension.json").read_text())["methods"] == previous["methods"]
        assert all("benchmarks.paper.c1" in row["command"] for row in new_plan[-4:])
        r4 = next(row for row in new_plan if row["arm"] == "c2kv_c1_t02_r4")
        command = list(r4["command"])
        assert r4["benchmark"] == "bfcl_base"
        assert command[command.index("--arm") + 1] == "c2kv_c1_t02_r4"
        assert "--enable-return-hidden-states" in server_command(config, Path("sglang"), "c2kv_c1_t02_r4")
        # An algorithm change to a previous arm cannot masquerade as extension.
        changed = copy.deepcopy(config)
        changed["methods"][3]["ratio"] = 8
        with pytest.raises(ValueError):
            prepare(changed, output, output / "sglang")


def test_task_selection_uses_the_official_file_and_rejects_foreign_ids():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        data = root / "bfcl_eval" / "data"
        data.mkdir(parents=True)
        (data / "BFCL_v4_multi_turn_base.json").write_text(
            '{"id":"multi_turn_base_1"}\n{"id":"multi_turn_base_2"}\n')
        config = {"bfcl_dir": str(root)}
        assert selected_tasks(config, "bfcl_base") == ["multi_turn_base_1", "multi_turn_base_2"]
        assert selected_tasks(config, "bfcl_base", ["multi_turn_base_2"]) == ["multi_turn_base_2"]
        with pytest.raises(ValueError):
            selected_tasks(config, "bfcl_base", ["multi_turn_base_99"])


def test_replay_keeps_official_task_identity_for_native_evidence_ids():
    rows = [{"replay_payload": {"c2kv_measurement_session_id": "multi_turn_base_26"}}] * 2
    assert replay_task_id(rows, "hashed-proxy-conversation") == "multi_turn_base_26"
    assert replay_task_id([{"replay_payload": {}}], "synthetic").startswith("replay_")


def test_controller_oom_is_a_scored_zero_harness_failure(tmp_path):
    shard = tmp_path / "task_shards" / "multi_turn_long_context_100"
    (shard / "server").mkdir(parents=True)
    assert controller_oom_message(shard) is None
    rows = [
        {"status": "completed", "error": None},
        {"status": "failed", "error": "{'type': 'OutOfMemoryError', "
                                      "'message': 'CUDA out of memory. Tried to allocate 6.00 GiB'}"},
    ]
    (shard / "server" / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    message = controller_oom_message(shard)
    assert message and "OutOfMemoryError" in message
    receipts = [
        {"task_id": "a", "status": "completed", "unified_metrics": {"official_score": 1.0}},
        {"task_id": "multi_turn_long_context_100", "status": "harness_failure",
         "failure": {"kind": "cuda_oom", "message": message},
         "unified_metrics": {"official_score": 0.0, "harness_failure": "cuda_oom"}},
    ]
    summary = summarize_scores("bfcl_long_context", receipts)
    assert summary["n"] == 2 and summary["semantic_score"] == 0.5
    assert summary["n_harness_failures"] == 1
    assert summary["harness_failure_task_ids"] == ["multi_turn_long_context_100"]
    assert summary["n_method_failures"] == 0


def test_capacity_infeasible_is_a_scored_zero_method_failure(tmp_path):
    shard = tmp_path / "task_shards" / "multi_turn_long_context_101"
    (shard / "server").mkdir(parents=True)
    rows = [{"status": "failed", "error": "{'type': 'CapacityInfeasible', 'message': "
                                          "\"Native S0 mandatory raw input and minimum whole-event gist cannot fit\"}"}]
    (shard / "server" / "steps.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    status, kind, message = controller_step_failure(shard)
    assert (status, kind) == ("method_failure", "capacity_infeasible")
    assert controller_oom_message(shard) is None
    receipts = [{"task_id": "multi_turn_long_context_101", "status": status,
                 "failure": {"kind": kind, "message": message},
                 "unified_metrics": {"official_score": 0.0, status: kind}}]
    summary = summarize_scores("bfcl_long_context", receipts)
    assert summary["semantic_score"] == 0.0 and summary["n_method_failures"] == 1
    assert summary["method_failure_task_ids"] == ["multi_turn_long_context_101"]
    assert summary["n_harness_failures"] == 0


def test_ratio4_ablation_binds_arm_and_ratio_for_summaries():
    try:
        assert select_arm("c2kv_c1_t02_r4") == ("c2kv_c1_t02_r4", 4)
        summary = summarize_scores("bfcl_base", [
            {"task_id": "a", "status": "completed", "unified_metrics": {"official_score": 1.0}}])
        assert (summary["arm"], summary["ratio"]) == ("c2kv_c1_t02_r4", 4)
        with pytest.raises(ValueError):
            select_arm("c2kv_c1_t02_r16")
    finally:
        select_arm("c2kv_c1_t02_r8")
    assert (paper_c1.ARM, paper_c1.RATIO) == ("c2kv_c1_t02_r8", 8)
