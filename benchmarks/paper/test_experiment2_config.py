"""The shared CUDA experiment-2 example must resolve to its stated cells."""

import json
from pathlib import Path

from benchmarks.paper import runner


CONFIG = Path(__file__).with_name("experiment2_cuda.json")
BENCHMARKS = {"bfcl_base", "bfcl_long_context", "acebench_agent", "appworld"}
ARMS = {
    "full", "hiagent_full_b768", "acon_hist_ut_co_b768",
    "history_kv_h2o_r25_persistent", "history_kv_snapkv_r25_persistent",
    "history_kv_pyramidkv_r25_persistent", "commitkv", "agentkv",
    "c2kv_native_r4", "c2kv_goal_rescue_r8",
}
HISTORY_KV = {
    "history_kv_h2o_r25_persistent", "history_kv_snapkv_r25_persistent",
    "history_kv_pyramidkv_r25_persistent", "commitkv", "agentkv",
}


def test_experiment2_config_resolves_only_original_budget_cells(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    plan, _ = runner.prepare(config, tmp_path / "results", tmp_path / "sglang-paper")
    by_id = {cell["cell_id"]: cell for cell in plan}

    assert len(plan) == len(BENCHMARKS) * len(ARMS) == len(by_id)
    assert {cell["benchmark"] for cell in plan} == BENCHMARKS
    assert {cell["arm"] for cell in plan} == ARMS
    assert all(cell["tool_context"] == "raw" for cell in plan)
    assert config["device"] == "cuda"
    assert config["disable_cuda_graph"] is True
    assert config["checkpoint"].endswith("/selected-C1000/checkpoint-1000")
    assert config["c1"] == {"task_timeout": 10800}

    for cell in plan:
        arm = cell["arm"]
        if arm in HISTORY_KV:
            assert cell["history_budget_tokens"] == 768
            assert "retention" not in cell
            command = cell["command"]
            assert command[command.index("--history-kv-target-tokens") + 1] == "768"
            assert cell["cell_id"].endswith(f"{arm}_b768")
        elif arm in {"hiagent_full_b768", "acon_hist_ut_co_b768"}:
            assert cell["history_budget_tokens"] == 768
            assert "--history-kv-target-tokens" not in cell["command"]
        elif arm == "c2kv_native_r4":
            assert cell["ratio"] == 4
        elif arm == "c2kv_goal_rescue_r8":
            assert cell["ratio"] == 8
            assert cell["group"] == "candidate"
            assert cell["command"][2] == "benchmarks.paper.c1"
        else:
            assert arm == "full"
            assert "--record-prefixes" in cell["command"]

    resolved = json.loads((tmp_path / "results" / "config.resolved.json").read_text())
    assert resolved["methods"] == config["methods"]
    assert len(json.loads((tmp_path / "results" / "commands.json").read_text())) == len(plan)
