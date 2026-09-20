"""Paper matrix contract for the explicit HiAgent history-budget adaptation."""

import json
from pathlib import Path
from unittest import mock
import urllib.error

import pytest

from benchmarks.arms import Arm, get_arm
from benchmarks.paper import runner


def _config():
    return json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_budget_selector_preserves_full_mode_and_original_identity():
    original = get_arm("hiagent_full")
    budget = get_arm("hiagent_full_b768")
    assert original.text_history_budget_tokens is None
    assert budget.name == "hiagent_full_b768"
    assert budget.text_policy == original.text_policy == "hiagent_full"
    assert budget.text_history_budget_tokens == 768
    assert budget.required_capabilities == original.required_capabilities
    assert get_arm("acon_hist_ut_co_b768").text_policy == "acon_hist_ut_co"
    for name in ("hiagent_full_b0", "hiagent_full_b0768"):
        with pytest.raises(ValueError):
            get_arm(name)
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            Arm(name="bad", compress_history=False,
                text_policy="hiagent_full", text_history_budget_tokens=value).validate()
    with pytest.raises(ValueError):
        Arm(name="bad", compress_history=False,
            text_policy="hiagent_summary", text_history_budget_tokens=768).validate()


def test_hiagent_and_acon_overlays_have_distinct_cells_and_server_modes(tmp_path):
    original = _config()
    config = runner.with_hiagent_budget(runner.with_acon_budget(original, 768), 512)
    assert runner.with_hiagent_budget(original, None) is original
    assert len(config["methods"]) == len(original["methods"]) + 2
    assert not any("_b512" in method["arm"] for method in original["methods"])
    plan, profile = runner.prepare(config, tmp_path / "output", tmp_path / "engine")
    assert profile.is_file()
    resolved = json.loads((profile.parent / "config.resolved.json").read_text())
    assert resolved["methods"][-2:] == config["methods"][-2:]
    assert [row["arm"] for row in resolved["methods"][-2:]] == [
        "acon_hist_ut_co_b768", "hiagent_full_b512"]
    for arm_name, policy in (("acon_hist_ut_co_b768", "acon_hist_ut_co"),
                             ("hiagent_full_b512", "hiagent_full")):
        rows = [row for row in plan if row["arm"] == arm_name]
        assert {row["benchmark"] for row in rows} == {"bfcl_base", "bfcl_long_context", "acebench_agent", "appworld", "tau2"}
        assert all(row["cell_id"].endswith("__" + arm_name) for row in rows)
        assert all(row["history_budget_tokens"] == get_arm(arm_name).text_history_budget_tokens
                   for row in rows)
        assert all("--arm" in row["command"] and
                   row["command"][row["command"].index("--arm") + 1] == arm_name
                   for row in rows)
        command = runner.server_command(config, Path("engine"), arm_name)
        assert command[2] == "benchmarks.paper.budget_server"
        assert "--disable-radix-cache" not in command
        assert get_arm(arm_name).text_policy == policy
    for arm_name in ("hiagent_full", "acon_hist_ut_co"):
        assert runner.server_command(config, Path("engine"), arm_name)[2] == "sglang.launch_server"
    for arm_name in ("acon_hist_ut_co_b768", "hiagent_full_b512"):
        ace = next(row for row in plan if row["cell_id"] == "acebench_agent__" + arm_name)
        command = ace["command"]
        assert command[command.index("--benchmark") + 1] == "acebench"
        assert command[command.index("--acebench-category") + 1] == "agent"
        assert command[command.index("--arm") + 1] == arm_name
        assert "acebench_role_history_v1" in command[command.index("--capability-features") + 1]
        assert "--tool-memory" not in command
        server = runner.server_command(config, Path("engine"), arm_name,
                                       benchmark="acebench_agent")
        assert server[2] == "benchmarks.paper.budget_server"
        assert server[server.index("--max-running-requests") + 1] == "2"
    assert len(json.loads((profile.parent / "commands.json").read_text())) == len(plan)
    assert "history_budget_tokens" in (profile.parent / "matrix.csv").read_text().splitlines()[0]


def test_budget_overlays_accept_ace_agent_without_bfcl():
    original = _config()
    original["benchmarks"] = [row for row in original["benchmarks"]
                              if row["name"] == "acebench_agent"]
    config = runner.with_hiagent_budget(runner.with_acon_budget(original, 768), 768)
    assert [row["benchmarks"] for row in config["methods"][-2:]] == [
        ["acebench_agent"], ["acebench_agent"]]
    assert {row["cell_id"] for row in runner.cells(config)
            if row["arm"].endswith("_b768")} == {
        "acebench_agent__acon_hist_ut_co_b768",
        "acebench_agent__hiagent_full_b768",
    }


@pytest.mark.parametrize("budget", [0, -1, True, 1.5, "768"])
def test_hiagent_overlay_requires_positive_integer(budget):
    with pytest.raises(ValueError, match="positive integer"):
        runner.with_hiagent_budget(_config(), budget)


def test_prepare_rejects_inconsistent_hiagent_budget_metadata(tmp_path):
    config = runner.with_hiagent_budget(_config(), 768)
    config["methods"][-1]["history_budget_tokens"] = 512
    with pytest.raises(ValueError, match="matching token cap"):
        runner.prepare(config, tmp_path / "output", tmp_path / "engine")


def test_cli_options_roundtrip_and_run_selected_identity(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_config()), encoding="utf-8")
    output = tmp_path / "output"
    arguments = ["--config", str(config_path), "--sglang-source", str(tmp_path / "engine"),
                 "--output", str(output), "--acon-budget-tokens", "768",
                 "--hiagent-budget-tokens", "512"]
    runner.main(["prepare", *arguments])
    resolved = json.loads((output / "config.resolved.json").read_text())
    assert {row["arm"] for row in resolved["methods"][-2:]} == {
        "acon_hist_ut_co_b768", "hiagent_full_b512"}
    selected = "bfcl_base__hiagent_full_b512"
    with mock.patch.object(runner, "execute") as execute:
        runner.main(["run", *arguments, "--stage", "closed_loop", "--cells", selected])
    config, plan, run_output, _, stages, cell_ids = execute.call_args.args[:6]
    assert config == {key: value for key, value in resolved.items() if key != "sglang_source"}
    assert run_output == output
    assert stages == ["closed_loop"]
    assert cell_ids == {selected}
    assert any(row["cell_id"] == selected for row in plan)


def test_budget_renderer_readiness_accepts_only_registered_post_route():
    def response(code):
        return urllib.error.HTTPError(
            "http://127.0.0.1:34000/v1/c2kv/chat_budget", code,
            "test", {}, None)

    with mock.patch.object(runner.urllib.request, "build_opener") as build:
        build.return_value.open.side_effect = response(405)
        runner.require_budget_renderer(34000)
        build.return_value.open.side_effect = response(404)
        with pytest.raises(RuntimeError, match="HTTP 404"):
            runner.require_budget_renderer(34000)


@pytest.mark.parametrize("arm_name", ["acon_hist_ut_co_b768", "hiagent_full_b512"])
def test_execute_checks_budget_renderer_before_harness(tmp_path, arm_name):
    config = runner.with_hiagent_budget(runner.with_acon_budget(_config(), 768), 512)
    output = tmp_path / "output"
    source = tmp_path / "engine"
    plan, _ = runner.prepare(config, output, source)
    cell = next(row for row in plan if row["cell_id"] == "bfcl_base__" + arm_name)
    with mock.patch.object(runner.subprocess, "Popen", return_value=mock.Mock(pid=123)), \
            mock.patch.object(runner, "run_owned") as harness, \
            mock.patch.object(runner, "wait_server") as health, \
            mock.patch.object(runner, "require_budget_renderer") as renderer, \
            mock.patch.object(runner, "cleanup_cell_processes"), \
            mock.patch("socket.socket") as socket_type:
        socket_type.return_value.__enter__.return_value.connect_ex.return_value = 1
        runner.execute(config, [cell], output, source, ["closed_loop"], {cell["cell_id"]})
    health.assert_called_once()
    renderer.assert_called_once_with(config["server_port"])
    harness.assert_called_once()
    started = json.loads((output / "closed_loop" / cell["cell_id"] / "started.json").read_text())
    assert started["server_command"][2] == "benchmarks.paper.budget_server"
    assert started["cell"]["arm"] == arm_name
