"""The tool-context axis of the paper matrix (benchmarks/paper/runner.py).

Raw-tool cells must stay byte-identical (ids, server and run commands) when
tool contexts are added, tool-context cells must carry the T0 flags on both
the server and the proxy side, including native recovery controllers.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.paper import runner

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / "sglang-paper"


def _config(**overrides):
    """The shipped config without its tool contexts (the tests add their own)."""
    config = json.loads((Path(runner.__file__).with_name("config.json")).read_text())
    config.pop("tool_contexts", None)
    for method in config["methods"]:
        method.pop("tool_contexts", None)
    config.update(overrides)
    return config


def test_shipped_config_tool_contexts_are_valid():
    shipped = json.loads((Path(runner.__file__).with_name("config.json")).read_text())
    contexts = runner.tool_contexts(shipped)
    assert set(contexts) >= {"raw", "t0_r8", "t0_r8_hybrid3"}
    assert contexts["t0_r8"]["checkpoint"].endswith("/T0/checkpoint-500")
    rows = runner.cells(shipped)
    ids = {row["cell_id"] for row in rows}
    assert "bfcl_base__full__tools-t0_r8" in ids and "bfcl_base__full__tools-t0_r8_hybrid3" in ids
    assert "bfcl_base__full" in ids
    assert not any(row.get("tool_memory") for row in rows if runner.is_native_arm(row["arm"]))


def _with_tool_context(config, arms=("full",), name="t0_r8", spec="t0:r8",
                       checkpoint="/ckpt/T0/checkpoint-1034"):
    config = copy.deepcopy(config)
    config["tool_contexts"] = [{"name": name, "spec": spec, "checkpoint": checkpoint}]
    for method in config["methods"]:
        if method["arm"] in arms:
            method["tool_contexts"] = [runner.RAW_TOOL_CONTEXT, name]
    return config


def test_raw_cells_are_unchanged_when_tool_contexts_are_added():
    base = _config()
    extended = _with_tool_context(base, arms=("full", "hiagent_full"))
    base_cells = {row["cell_id"]: row for row in runner.cells(base)}
    new_cells = {row["cell_id"]: row for row in runner.cells(extended)}
    assert set(base_cells) <= set(new_cells)
    for cell_id, old in base_cells.items():
        new = new_cells[cell_id]
        assert old == new, cell_id
        assert old["tool_context"] == runner.RAW_TOOL_CONTEXT
        assert "tool_memory" not in old
        profile = Path("/out/deployment_profile.json")
        for stage in ("closed_loop", "common_prefix"):
            assert (runner.run_command(base, old, Path("/out") / stage / cell_id, profile, stage)
                    == runner.run_command(extended, new, Path("/out") / stage / cell_id, profile, stage))
        assert (runner.server_command(base, SOURCE, old["arm"])
                == runner.server_command(extended, SOURCE, new["arm"], tool_checkpoint=new.get("tool_checkpoint")))
    added = sorted(set(new_cells) - set(base_cells))
    benches = {"bfcl_base", "bfcl_long_context", "appworld", "acebench_agent", "toolsandbox", "tau2"}
    assert added == sorted(f"{b}__{arm}__tools-t0_r8" for b in benches for arm in ("full", "hiagent_full"))


def test_tool_context_cells_carry_t0_flags_on_server_and_proxy():
    config = _with_tool_context(_config(), arms=("full",))
    cell = next(row for row in runner.cells(config) if row["cell_id"] == "bfcl_base__full__tools-t0_r8")
    assert cell["tool_context"] == "t0_r8"
    assert cell["tool_memory"] == "t0:r8" and cell["tool_checkpoint"] == "/ckpt/T0/checkpoint-1034"
    server = runner.server_command(config, SOURCE, cell["arm"], cell["benchmark"], tool_checkpoint=cell["tool_checkpoint"])
    assert server[server.index("--c2kv-tool-gist-weights") + 1] == "/ckpt/T0/checkpoint-1034"
    assert runner.server_command(config, SOURCE, "full") == server[:-2]
    command = runner.run_command(config, cell, Path("/out/closed_loop/x"), Path("/out/deployment_profile.json"))
    assert command[command.index("--tool-memory") + 1] == "t0:r8"
    assert command[command.index("--tool-checkpoint") + 1] == "/ckpt/T0/checkpoint-1034"
    # only the raw-tools Full cell records the canonical replay prefixes
    assert "--record-prefixes" not in command
    raw = next(row for row in runner.cells(config) if row["cell_id"] == "bfcl_base__full")
    assert "--record-prefixes" in runner.run_command(config, raw, Path("/out/closed_loop/y"),
                                                     Path("/out/deployment_profile.json"))


def test_tool_context_validation():
    with pytest.raises(ValueError, match="unique"):
        runner.tool_contexts({"tool_contexts": [{"name": "raw", "spec": "t0:r8", "checkpoint": "/c"}]})
    with pytest.raises(ValueError, match="non-raw spec"):
        runner.tool_contexts({"tool_contexts": [{"name": "x", "spec": "none", "checkpoint": "/c"}]})
    with pytest.raises(ValueError):
        runner.tool_contexts({"tool_contexts": [{"name": "x", "spec": "t0:r4", "checkpoint": "/c"}]})
    with pytest.raises(ValueError, match="checkpoint"):
        runner.tool_contexts({"tool_contexts": [{"name": "x", "spec": "t0:r8"}]})
    config = _with_tool_context(_config(), arms=("full",))
    config["methods"][0]["tool_contexts"] = ["raw", "missing"]
    with pytest.raises(ValueError, match="unknown tool context"):
        runner.cells(config)


def test_native_arms_receive_tool_contexts(tmp_path):
    config = _with_tool_context(_config(), arms=("c2kv_c1_t02_r8",))
    runner.prepare(config, tmp_path / "out", SOURCE)
    cell = next(row for row in runner.cells(config)
                if row["cell_id"] == "bfcl_base__c2kv_c1_t02_r8__tools-t0_r8")
    command = runner.run_command(config, cell, tmp_path, tmp_path / "profile.json")
    assert command[command.index("--tool-memory") + 1] == "t0:r8"
    assert command[command.index("--tool-checkpoint") + 1] == cell["tool_checkpoint"]


def test_budget_arms_refuse_tool_contexts(tmp_path):
    """Budget-adapted text arms measure the raw prompt (tool prologue included)
    through the server's chat budget renderer; the axis stays off them."""
    config = runner.with_hiagent_budget(_config(), 4096)
    config = _with_tool_context(config, arms=("hiagent_full_b4096",))
    with pytest.raises(ValueError, match="ACON/HiAgent"):
        runner.prepare(config, tmp_path / "out", SOURCE)
    # the raw cell of the budget arm is untouched by the axis
    rows = [row for row in runner.cells(runner.with_hiagent_budget(_config(), 4096))
            if row["arm"] == "hiagent_full_b4096"]
    assert rows and all(row["tool_context"] == "raw" and "tool_memory" not in row for row in rows)


def test_tool_overlay_preserves_old_cells_and_history_budgets():
    base = _with_tool_context(_config())
    extended = runner.with_tool_contexts(base, ["t0_r8"], "/restored/T0/checkpoint-1034")
    assert runner.with_tool_contexts(base, []) is base
    old = {row["cell_id"]: row for row in runner.cells(base) if row["tool_context"] == "raw"}
    new = {row["cell_id"]: row for row in runner.cells(extended)}
    assert all(new[key] == value for key, value in old.items())
    for row in new.values():
        if row["tool_context"] == "raw":
            continue
        assert not row["arm"].startswith(("hiagent", "acon"))
        raw = old[row["benchmark"] + "__" + row["arm"]]
        for field in ("history_budget_tokens", "ratio", "retention"):
            assert row.get(field) == raw.get(field)
        assert row["tool_checkpoint"] == "/restored/T0/checkpoint-1034"
    assert "bfcl_base__c2kv_c1_t02_r8__tools-t0_r8" in new
    assert "bfcl_base__history_kv_h2o_r25_persistent__tools-t0_r8" in new


def test_tool_budget_reaches_native_and_proxy_commands():
    config = _with_tool_context(_config(), arms=("full", "c2kv_c1_t02_r8"))
    config["tool_contexts"][0]["budget_tokens"] = 128
    for cell in runner.cells(config):
        if not cell.get("tool_memory"):
            continue
        command = runner.run_command(config, cell, Path("/out/cell"), Path("/out/profile.json"))
        assert command[command.index("--tool-budget-tokens") + 1] == "128"


def test_prepare_writes_tool_context_column_and_commands(tmp_path):
    config = _with_tool_context(_config(), arms=("full",))
    plan, _ = runner.prepare(config, tmp_path / "out", SOURCE)
    rows = (tmp_path / "out" / "matrix.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].endswith(",tool_context")
    assert any(line.startswith("bfcl_base__full__tools-t0_r8,") and line.endswith(",t0_r8") for line in rows)
    commands = json.loads((tmp_path / "out" / "commands.json").read_text())
    tool_cells = [c for c in commands if c["cell_id"].endswith("__tools-t0_r8")]
    assert len(tool_cells) == len(config["benchmarks"]) and all("--tool-memory" in c["command"] for c in tool_cells)


def test_guard_tool_context_checks_the_checkpoint_contract(tmp_path):
    checkpoint = tmp_path / "checkpoint-1034"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "history_memory_compression_domain": "tool", "history_memory_variant": "T0",
        "history_memory_render_profile": "next-compression-tool-explicit-protocol-v2",
        "history_memory_supported_ratios": [8, 12]}), encoding="utf-8")
    runner._guard_tool_context({"tool_memory": "t0:r8", "tool_checkpoint": str(checkpoint)})
    runner._guard_tool_context({"arm": "full"})
    from benchmarks.toolmemory import ToolMemoryError
    with pytest.raises(ToolMemoryError):
        runner._guard_tool_context({"tool_memory": "t0:r8", "tool_checkpoint": str(tmp_path / "nope")})
