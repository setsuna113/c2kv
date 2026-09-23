"""Absolute history-KV capacities share one resolved paper budget."""
import copy
import json
from pathlib import Path
from unittest import mock

import pytest

from benchmarks.history_budget import HistoryKVBudget
from benchmarks.paper import runner
from benchmarks.paper.report import write_comparison


KV_ARMS = (
    "history_kv_h2o_r25_persistent",
    "history_kv_snapkv_r25_persistent",
    "history_kv_pyramidkv_r25_persistent",
    "commitkv",
    "agentkv",
)


def config(budget=None):
    cfg = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg["history_kv_budget_tokens"] = budget
    return cfg


@pytest.mark.parametrize("budget", [384, 1024])
def test_shared_budget_resolves_each_kv_arm_without_mutating_template(tmp_path, budget):
    template = config(budget)
    saved = copy.deepcopy(template)
    resolved = runner.resolve_history_kv_budgets(template)
    assert template == saved
    assert resolved["history_kv_budget_tokens"] == budget
    for arm in KV_ARMS:
        method = next(m for m in resolved["methods"] if m["arm"] == arm)
        assert method["history_budget_tokens"] == budget
        assert method["history_budget_source"] == "shared"
        assert "retention" not in method
    plan, _ = runner.prepare(template, tmp_path / "results", tmp_path / "engine")
    frozen = json.loads((tmp_path / "results" / "config.resolved.json").read_text())
    assert frozen["history_kv_budget_tokens"] == budget
    assert all(m.get("history_budget_tokens") != "shared" for m in frozen["methods"])
    by_id = {cell["cell_id"]: cell for cell in plan}
    for arm in KV_ARMS:
        cell_id = f"bfcl_base__{HistoryKVBudget(budget).variant_name(arm)}"
        cell = by_id[cell_id]
        assert cell["history_budget_tokens"] == budget
        assert "retention" not in cell
        assert cell["command"][cell["command"].index("--history-kv-target-tokens") + 1] == str(budget)
        replay = runner.run_command(resolved, cell, tmp_path / "replay", tmp_path / "profile.json",
                                    "common_prefix")
        assert replay[replay.index("--history-kv-target-tokens") + 1] == str(budget)
    assert not any("history_retention_ratio" in m for m in frozen["methods"])


@pytest.mark.parametrize("budget", [None, 0, -1, True, 1.5, "384"])
def test_unresolved_or_invalid_shared_budget_writes_no_artifacts(tmp_path, budget):
    output = tmp_path / "results"
    with pytest.raises(ValueError, match="history_kv_budget_tokens"):
        runner.prepare(config(budget), output, tmp_path / "engine")
    assert not output.exists()
    assert not list(tmp_path.glob("*.prepare.lock"))


def test_ratio_api_metadata_is_rejected_but_historical_retention_config_works(tmp_path):
    cfg = config(384)
    cfg["methods"][0]["history_retention_ratio"] = 0.5
    with pytest.raises(ValueError, match="history_retention_ratio is no longer supported"):
        runner.prepare(cfg, tmp_path / "bad", tmp_path / "engine")
    assert not (tmp_path / "bad").exists()

    legacy = config()
    legacy.pop("history_kv_budget_tokens")
    for method in legacy["methods"]:
        if method.get("history_budget_tokens") != "shared":
            continue
        method.pop("history_budget_tokens")
        if method["method"] in {"H2O", "SnapKV", "PyramidKV"}:
            method["retention"] = 0.25
    plan, _ = runner.prepare(legacy, tmp_path / "legacy", tmp_path / "engine")
    assert any(cell["cell_id"] == "bfcl_base__history_kv_h2o_r25_persistent" for cell in plan)
    assert any(cell["cell_id"] == "bfcl_base__commitkv" for cell in plan)


def test_cli_shared_budget_precedes_independent_arm_overlay(tmp_path):
    output = tmp_path / "results"
    args = ["--output", str(output), "--sglang-source", str(tmp_path / "engine"),
            "--history-kv-budget-tokens", "384", "--history-kv-budget", "commitkv=1024"]
    runner.main(["prepare", *args])
    by_id = {row["cell_id"]: row for row in json.loads((output / "commands.json").read_text())}
    assert by_id["bfcl_base__commitkv_b384"]["history_budget_tokens"] == 384
    assert by_id["bfcl_base__commitkv_b1024"]["history_budget_tokens"] == 1024
    for arm in KV_ARMS[:3]:
        assert f"bfcl_base__{HistoryKVBudget(384).variant_name(arm)}" in by_id
    with mock.patch.object(runner, "execute") as execute:
        runner.main(["run", *args, "--stage", "closed_loop", "--cells",
                     "bfcl_base__commitkv_b1024"])
    _, plan, _, _, stages, ids = execute.call_args.args[:6]
    assert stages == ["closed_loop"] and ids == {"bfcl_base__commitkv_b1024"}
    assert next(row for row in plan if row["cell_id"] == "bfcl_base__commitkv_b1024")[
        "history_budget_tokens"] == 1024


def test_frozen_numeric_config_can_change_shared_b_without_rewriting_sweep(tmp_path):
    initial = runner.with_history_kv_budget(config(384), "commitkv", 1024)
    first_output = tmp_path / "first"
    runner.prepare(initial, first_output, tmp_path / "engine")
    frozen = json.loads((first_output / "config.resolved.json").read_text())
    frozen["history_kv_budget_tokens"] = 768
    changed = runner.resolve_history_kv_budgets(frozen)
    for arm in KV_ARMS:
        primary = next(m for m in changed["methods"]
                       if m["arm"] == arm and m.get("history_budget_source") == "shared")
        assert primary["history_budget_tokens"] == 768
    sweep = next(m for m in changed["methods"]
                 if m["arm"] == "commitkv" and m.get("group") == "budget")
    assert sweep["history_budget_tokens"] == 1024
    assert "history_budget_source" not in sweep
    with pytest.raises(RuntimeError, match="cells removed"):
        runner.prepare(changed, first_output, tmp_path / "engine")
    plan, _ = runner.prepare(changed, tmp_path / "second", tmp_path / "engine")
    ids = {cell["cell_id"] for cell in plan}
    assert "bfcl_base__commitkv_b768" in ids
    assert "bfcl_base__commitkv_b1024" in ids


def test_overlay_requires_unique_primary_and_can_use_explicit_base_budget():
    base = runner.resolve_history_kv_budgets(config(384))
    variant = runner.with_history_kv_budget(base, "commitkv", 1024)
    assert variant["methods"][-1]["history_budget_tokens"] == 1024
    with pytest.raises(ValueError, match="already exists"):
        runner.with_history_kv_budget(base, "commitkv", 384)
    with pytest.raises(ValueError, match="does not support"):
        runner.with_history_kv_budget(base, "hiagent_full", 1024)
    ambiguous = copy.deepcopy(base)
    ambiguous["methods"].append(dict(next(m for m in base["methods"] if m["arm"] == "commitkv")))
    with pytest.raises(ValueError, match="one primary"):
        runner.with_history_kv_budget(ambiguous, "commitkv", 1024)
    only_sweep = copy.deepcopy(base)
    next(m for m in only_sweep["methods"] if m["arm"] == "commitkv")["group"] = "sweep"
    with pytest.raises(ValueError, match="one primary"):
        runner.with_history_kv_budget(only_sweep, "commitkv", 1024)


def test_comparison_keeps_absolute_budget_points_separate(tmp_path):
    cfg = runner.with_history_kv_budget(config(384), "commitkv", 1024)
    plan = [row for row in runner.cells(cfg) if row["cell_id"] in
            {"bfcl_base__commitkv_b384", "bfcl_base__commitkv_b1024"}]
    for cell in plan:
        directory = tmp_path / "closed_loop" / cell["cell_id"]
        directory.mkdir(parents=True)
        (directory / "complete.json").write_text("{}")
        (directory / "measurement_summary.json").write_text("{}")
        (directory / "summary_commitkv.json").write_text('{"n": 1}')
    rows = write_comparison(tmp_path, plan)
    assert [row["history_budget_tokens"] for row in rows] == [384, 1024]
    assert len({row["cell_id"] for row in rows}) == 2


def test_supported_replay_uses_the_same_absolute_capacity(tmp_path):
    cfg = runner.resolve_history_kv_budgets(config(384))
    output, source = tmp_path / "results", tmp_path / "engine"
    plan, _ = runner.prepare(cfg, output, source)
    selected = "bfcl_base__history_kv_h2o_persistent_b384"
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
    assert proxy_command[proxy_command.index("--history-kv-target-tokens") + 1] == "384"
    assert proxy_command[proxy_command.index("--arm") + 1] == "history_kv_h2o_r25_persistent"
