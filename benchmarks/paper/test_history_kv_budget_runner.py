"""Capacity sweeps retain legacy identities and exact-output session contracts."""
import copy
import json
from pathlib import Path
from unittest import mock

import pytest

from benchmarks.paper import runner
from benchmarks.paper.report import write_comparison


def config():
    return json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_budget_cells_preserve_existing_commands_and_record_capacity(tmp_path):
    original = config()
    saved = copy.deepcopy(original)
    old_plan, profile = runner.prepare(original, tmp_path / "results", tmp_path / "engine")
    expanded = runner.with_history_kv_budget(original, "commitkv", 768)
    expanded = runner.with_history_kv_budget(expanded, "commitkv", 1024)
    expanded = runner.with_history_kv_budget(expanded, "agentkv", 512)
    plan, _ = runner.prepare(expanded, profile.parent, tmp_path / "engine")
    assert original == saved
    by_id = {cell["cell_id"]: cell for cell in plan}
    assert all(by_id[cell["cell_id"]] == cell for cell in old_plan)
    for arm, budget in (("commitkv", 768), ("commitkv", 1024), ("agentkv", 512)):
        for benchmark in original["benchmarks"]:
            cell = by_id[f"{benchmark['name']}__{arm}_b{budget}"]
            command = cell["command"]
            assert cell["arm"] == arm
            assert cell["history_budget_tokens"] == budget
            assert command[command.index("--arm") + 1] == arm
            assert command[command.index("--history-kv-target-tokens") + 1] == str(budget)
            server = runner.server_command(expanded, tmp_path / "engine", arm)
            assert server[2] == "sglang.launch_server"
            assert "--disable-radix-cache" in server
            assert "--disable-cuda-graph" in server
    assert "history_budget_tokens" in (profile.parent / "matrix.csv").read_text()
    assert json.loads((profile.parent / "config.resolved.json").read_text())["methods"] == expanded["methods"]
    assert len(list(profile.parent.glob("config.before_extension.*.json"))) == 1


def test_cli_repeated_budget_and_run_selection(tmp_path):
    output = tmp_path / "results"
    args = ["--output", str(output), "--sglang-source", str(tmp_path / "engine"),
            "--history-kv-budget", "commitkv=768", "--history-kv-budget", "commitkv=1024",
            "--history-kv-budget", "agentkv=512"]
    runner.main(["prepare", *args])
    selected = "bfcl_base__agentkv_b512"
    with mock.patch.object(runner, "execute") as execute:
        runner.main(["run", *args, "--stage", "closed_loop", "--cells", selected])
    _, plan, _, _, stages, ids = execute.call_args.args[:6]
    assert stages == ["closed_loop"] and ids == {selected}
    assert next(row for row in plan if row["cell_id"] == selected)["history_budget_tokens"] == 512


def test_budget_is_orthogonal_to_tool_context_and_fractional_baseline(tmp_path):
    base = config()
    expanded = runner.with_history_kv_budget(base, "history_kv_h2o_r25_persistent", 768)
    expanded = runner.with_history_kv_budget(expanded, "commitkv", 768)
    expanded["methods"][-1]["tool_contexts"] = ["raw", "t0_r8"]
    plan, _ = runner.prepare(expanded, tmp_path / "results", tmp_path / "engine")
    cell = next(row for row in plan if row["cell_id"] == "bfcl_base__commitkv_b768__tools-t0_r8")
    assert "--tool-memory" in cell["command"] and "--history-kv-target-tokens" in cell["command"]
    h2o = next(row for row in plan if row["cell_id"] == "bfcl_base__history_kv_h2o_r25_persistent_b768")
    assert "retention" not in h2o
    assert next(row for row in plan if row["cell_id"] == "bfcl_base__history_kv_h2o_r25_persistent")["retention"] == 0.25


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "768", None])
def test_prepare_rejects_bad_explicit_budget(tmp_path, budget):
    cfg = config()
    next(row for row in cfg["methods"] if row["arm"] == "commitkv")["history_budget_tokens"] = budget
    with pytest.raises(ValueError, match="positive integer"):
        runner.prepare(cfg, tmp_path / "results", tmp_path / "engine")
    assert not (tmp_path / "results" / "config.resolved.json").exists()


def test_budget_variants_reject_duplicates_and_unsupported_methods(tmp_path):
    cfg = runner.with_history_kv_budget(config(), "commitkv", 768)
    with pytest.raises(ValueError, match="already exists"):
        runner.with_history_kv_budget(cfg, "commitkv", 768)
    for arm in ("full", "hiagent_full", "c2kv_native_r4"):
        with pytest.raises(ValueError, match="does not support"):
            runner.with_history_kv_budget(config(), arm, 768)
    # A budget change cannot resume under the old result identity.
    output = tmp_path / "results"
    runner.prepare(cfg, output, tmp_path / "engine")
    changed = runner.with_history_kv_budget(config(), "commitkv", 512)
    with pytest.raises(RuntimeError, match="cells removed"):
        runner.prepare(changed, output, tmp_path / "engine")


@pytest.mark.parametrize("arm", ["commitkv", "agentkv"])
def test_budget_variant_retains_exact_prefix_replay_rejection(tmp_path, arm):
    cfg = runner.with_history_kv_budget(config(), arm, 768)
    plan, _ = runner.prepare(cfg, tmp_path / "results", tmp_path / "engine")
    selected = f"bfcl_base__{arm}_b768"
    with mock.patch.object(runner.subprocess, "Popen") as popen:
        with pytest.raises(RuntimeError, match="exact_generated_prefix"):
            runner.execute(cfg, plan, tmp_path / "results", tmp_path / "engine", ["common_prefix"], {selected})
    popen.assert_not_called()


def test_comparison_keeps_budget_points_separate(tmp_path):
    cfg = runner.with_history_kv_budget(config(), "commitkv", 768)
    plan = [row for row in runner.cells(cfg) if row["cell_id"] in
            {"bfcl_base__commitkv", "bfcl_base__commitkv_b768"}]
    for cell in plan:
        directory = tmp_path / "closed_loop" / cell["cell_id"]
        directory.mkdir(parents=True)
        (directory / "complete.json").write_text("{}")
        (directory / "measurement_summary.json").write_text("{}")
        (directory / "summary_commitkv.json").write_text('{"n": 1}')
    rows = write_comparison(tmp_path, plan)
    assert [row["history_budget_tokens"] for row in rows] == [None, 768]
    assert len({row["cell_id"] for row in rows}) == 2
    assert all(row["comparison_basis"] == "own_output_closed_loop" for row in rows)


def test_supported_replay_uses_the_same_absolute_capacity(tmp_path):
    cfg = runner.with_history_kv_budget(config(), "history_kv_h2o_r25_persistent", 768)
    output, source = tmp_path / "results", tmp_path / "engine"
    plan, _ = runner.prepare(cfg, output, source)
    selected = "bfcl_base__history_kv_h2o_r25_persistent_b768"
    cell = next(row for row in plan if row["cell_id"] == selected)
    prefixes = Path(cell["replay_source"])
    prefixes.parent.mkdir(parents=True)
    prefixes.write_text("")
    with mock.patch.object(runner.subprocess, "Popen", return_value=mock.Mock(pid=123)) as popen, \
            mock.patch.object(runner, "run_owned"), \
            mock.patch.object(runner, "wait_server"), \
            mock.patch.object(runner, "cleanup_cell_processes"), \
            mock.patch("socket.socket") as socket_type:
        socket_type.return_value.__enter__.return_value.connect_ex.return_value = 1
        runner.execute(cfg, plan, output, source, ["common_prefix"], {selected})
    proxy_command = popen.call_args_list[1].args[0]
    assert proxy_command[proxy_command.index("--history-kv-target-tokens") + 1] == "768"
    assert proxy_command[proxy_command.index("--arm") + 1] == "history_kv_h2o_r25_persistent"
