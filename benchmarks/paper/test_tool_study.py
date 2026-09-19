"""Observable contracts for the paired tool/history component experiment."""
import copy
import json
from pathlib import Path

from benchmarks.arms import get_arm
from benchmarks.paper import c1, native_extra, runner
from benchmarks.paper.tool_study import joint_config


def test_joint_matrix_has_four_paired_cells_and_three_anchors(tmp_path):
    base = json.loads(runner.DEFAULT_CONFIG.read_text())
    original = copy.deepcopy(base)
    config = joint_config(base, tool_checkpoint="/checkpoints/T0", tool_budget_tokens=1024)
    assert base == original
    assert config["output_root"] != base["output_root"]
    rows = runner.cells(config)
    assert len(rows) == 7 and len({row["cell_id"] for row in rows}) == 7
    factorial = [row for row in rows if row["tool_context"] != "raw" and row["arm"] != "full"]
    assert {(row["arm"], row["tool_context"]) for row in factorial} == {
        (arm, tool) for arm in ("c2kv_c1_off_r8", "c2kv_c1_t02_r8")
        for tool in ("uniform", "hybrid")}
    assert all(row["ratio"] == 8 and row["tool_budget_tokens"] == 1024 for row in factorial)
    assert config["c1"]["detector"] == "t02_risk"
    plan, _ = runner.prepare(config, tmp_path / "prepared", tmp_path / "sglang")
    assert len(plan) == 7
    for row in plan:
        if row["arm"] == "c2kv_c1_off_r8":
            assert "benchmarks.paper.c1" in row["command"]


def test_recovery_off_routes_to_existing_initial_allocation_ablation(tmp_path):
    base = json.loads(runner.DEFAULT_CONFIG.read_text())
    config = joint_config(base, tool_checkpoint="/checkpoints/T0")
    config["sglang_source"] = str(tmp_path)
    delivery = c1.load_delivery()
    old_arm = c1.ARM
    try:
        c1.select_arm("c2kv_c1_off_r8")
        off = c1.delivery_args(config, "bfcl_base", tmp_path, ["multi_turn_base_26"], delivery)
        c1.select_arm("c2kv_c1_t02_r8")
        on = c1.delivery_args(config, "bfcl_base", tmp_path, ["multi_turn_base_26"], delivery)
    finally:
        c1.select_arm(old_arm)
    assert off.method == "c2kv_only" and on.method == "proposed"
    assert off.ratio == on.ratio == 8
    for key in ("checkpoint", "selector_threshold", "embedding_model", "detector"):
        assert getattr(off, key) == getattr(on, key)
    assert get_arm("c2kv_c1_off_r8").native_controller == "c1_recovery_off"
    assert native_extra.arm_identity({"native_arm": "c2kv_c1_off_r8"})["method"] == "c2kv_only"
