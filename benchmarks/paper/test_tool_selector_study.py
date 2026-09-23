"""Run-plan checks for the tool-selector comparison."""
import copy
import json

from benchmarks.paper import runner
from benchmarks.paper.tool_selector_study import joint_selector_config


def test_two_selectors_keep_one_history_and_no_tool_budget(tmp_path):
    base = dict(json.loads(runner.DEFAULT_CONFIG.read_text()), history_kv_budget_tokens=768)
    base["tool_history_study"] = {"history_factor": ["c2kv_c1_off_r8"]}
    original = copy.deepcopy(base)
    config = joint_selector_config(base, tool_checkpoint="/checkpoints/T0",
                                   output_root=str(tmp_path / "results"))
    assert base == original
    assert "tool_history_study" not in config
    rows = runner.cells(config)
    assert len(rows) == 2 and len({row["cell_id"] for row in rows}) == 2
    assert {row["arm"] for row in rows} == {"c2kv_pending_verified_r8"}
    assert {row["benchmark"] for row in rows} == {"bfcl_base"}
    assert {row["tool_memory"] for row in rows} == {
        "t0:r8:hybrid3:schema",
        "t0:r8:hybrid3:schema:selector=latest_event_topk_v1",
    }
    assert all("tool_budget_tokens" not in row for row in rows)
    plan, _ = runner.prepare(config, tmp_path / "paper", tmp_path / "sglang")
    assert len(plan) == 2
    for row in plan:
        command = row["command"]
        assert "benchmarks.paper.c1" in command
        assert command[command.index("--arm") + 1] == "c2kv_pending_verified_r8"
        assert command[command.index("--tool-memory") + 1] == row["tool_memory"]
