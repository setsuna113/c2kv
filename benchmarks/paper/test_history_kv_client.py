"""Single-cell client keeps paper adapter commands and output identities."""
import json
from pathlib import Path

import pytest

from benchmarks.paper import history_kv_client as client
from benchmarks.paper import runner


def config():
    return json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))


@pytest.mark.parametrize("benchmark,adapter,flag,value", [
    ("bfcl_base", "bfcl", "--categories", "multi_turn_base"),
    ("bfcl_long_context", "bfcl", "--categories", "multi_turn_long_context"),
    ("acebench_agent", "acebench", "--acebench-category", "agent"),
    ("appworld", "acon_appworld", "--split", "test_normal"),
    ("toolsandbox", "toolsandbox", "--toolsandbox-dir", config()["toolsandbox_dir"]),
])
@pytest.mark.parametrize("arm", ["commitkv", "agentkv"])
def test_plan_reuses_all_configured_adapter_routes(tmp_path, benchmark, adapter, flag, value, arm):
    resolved, cell, command, directory = client.plan(
        config(), benchmark, f"{arm}=768", tmp_path / "results",
        "http://npu.example:36200/", 37490)
    assert cell["cell_id"] == f"{benchmark}__{arm}_b768"
    assert cell["arm"] == arm and cell["history_budget_tokens"] == 768
    assert directory == tmp_path / "results" / "closed_loop" / cell["cell_id"]
    assert command[command.index("--benchmark") + 1] == adapter
    assert command[command.index("--arm") + 1] == arm
    assert command[command.index("--history-kv-target-tokens") + 1] == "768"
    assert command[command.index("--upstream") + 1] == "http://npu.example:36200"
    assert command[command.index("--proxy-port") + 1] == "37490"
    assert command[command.index(flag) + 1] == value
    assert command[command.index("--run-name") + 1] == cell["cell_id"]
    assert "--exact-out" in command and "--num-workers" in command
    assert "--checkpoint-profile" not in command
    assert "--shared-engine" in command
    assert resolved["proxy_port"] == 37490


def test_profile_override_and_budget_validation(tmp_path):
    profile = tmp_path / "profile.json"
    _, _, command, _ = client.plan(config(), "bfcl_base", "commitkv=768",
                                   tmp_path, "http://localhost:36200", checkpoint_profile=profile,
                                   bench_python="/npu/bench/python")
    assert command[0] == "/npu/bench/python"
    assert command[command.index("--checkpoint-profile") + 1] == str(profile)
    for budget in ("full=768", "commitkv=0", "commitkv=-1", "commitkv=xyz"):
        with pytest.raises(ValueError):
            client.plan(config(), "bfcl_base", budget, tmp_path, "http://localhost:36200")
    with pytest.raises(ValueError, match="Expected one configured"):
        client.plan(config(), "unknown", "agentkv=768", tmp_path, "http://localhost:36200")


def test_ratio_client_uses_same_runner_identity_and_command(tmp_path, capsys):
    cfg = config()
    resolved, cell, command, directory = client.plan(
        cfg, "bfcl_base", None, tmp_path / "results", "http://localhost:36200",
        retention_spec="commitkv=0.5")
    assert cell["cell_id"] == "bfcl_base__commitkv_r0p5"
    assert cell["history_retention_ratio"] == 0.5
    assert "history_budget_tokens" not in cell
    assert directory.name == cell["cell_id"]
    assert command[command.index("--history-kv-retention-ratio") + 1] == "0.5"
    assert "--history-kv-target-tokens" not in command
    assert resolved["methods"][-1]["history_retention_ratio"] == 0.5
    source = tmp_path / "config.json"
    source.write_text(json.dumps(cfg), encoding="utf-8")
    client.main(["--config", str(source), "--benchmark", "bfcl_base",
                 "--history-kv-retention", "commitkv=0.5",
                 "--upstream", "http://localhost:36200", "--out",
                 str(tmp_path / "dry"), "--dry-run"])
    assert json.loads(capsys.readouterr().out)["cell_id"] == cell["cell_id"]
    assert not (tmp_path / "dry").exists()


def test_ratio_client_selects_existing_configured_cell(tmp_path):
    cfg = config()
    resolved, cell, command, _ = client.plan(
        cfg, "bfcl_base", None, tmp_path / "results", "http://localhost:36200",
        retention_spec="commitkv=0.25")
    assert resolved == cfg
    assert cell["cell_id"] == "bfcl_base__commitkv_r0p25"
    assert command[command.index("--history-kv-retention-ratio") + 1] == "0.25"


def test_dry_run_does_not_launch_or_write(monkeypatch, tmp_path, capsys):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(config()), encoding="utf-8")
    monkeypatch.setattr(client, "run_owned", lambda *a, **k: pytest.fail("must not launch"))
    output = tmp_path / "results"
    client.main(["--config", str(source), "--benchmark", "bfcl_base",
                 "--history-kv-budget", "commitkv=768", "--upstream", "http://localhost:36200",
                 "--out", str(output), "--dry-run"])
    printed = json.loads(capsys.readouterr().out)
    assert printed["cell_id"] == "bfcl_base__commitkv_b768"
    assert printed["stage"] == "closed_loop"
    assert not output.exists()


def test_run_records_exact_cell_then_refuses_changed_or_partial_output(monkeypatch, tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(config()), encoding="utf-8")
    output = tmp_path / "results"
    launched = []
    monkeypatch.setattr(client, "run_owned", lambda cmd, **kw: launched.append((cmd, kw)))
    arguments = ["--config", str(source), "--benchmark", "acebench_agent",
                 "--history-kv-budget", "agentkv=768", "--upstream", "http://localhost:36200",
                 "--out", str(output)]
    client.main(arguments)
    cell_dir = output / "closed_loop" / "acebench_agent__agentkv_b768"
    assert len(launched) == 1
    assert (cell_dir / "started.json").is_file()
    assert (cell_dir / "complete.json").is_file()
    assert json.loads((output / "client_manifest.json").read_text())["cell"]["arm"] == "agentkv"
    assert json.loads((output / "config.resolved.json").read_text())["methods"][-1]["history_budget_tokens"] == 768
    assert json.loads((output / "commands.json").read_text())[0]["command"] == launched[0][0]
    assert launched[0][1]["env"]["BENCH_BFCL_DIR"] == config()["bfcl_dir"]
    client.main(arguments)
    assert len(launched) == 1
    with pytest.raises(RuntimeError, match="different resolved cell"):
        client.main([*arguments[:5], "agentkv=512", *arguments[6:]])
    (cell_dir / "complete.json").unlink()
    with pytest.raises(RuntimeError, match="Partial cell"):
        client.main(arguments)


def test_aggregate_requires_attributed_server_telemetry(monkeypatch, tmp_path):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(config()), encoding="utf-8")
    output = tmp_path / "results"
    monkeypatch.setattr(client, "run_owned", lambda *a, **k: None)
    client.main(["--config", str(source), "--benchmark", "bfcl_base",
                 "--history-kv-budget", "commitkv=768", "--upstream", "http://localhost:36200",
                 "--out", str(output)])
    cell_dir = output / "closed_loop" / "bfcl_base__commitkv_b768"
    (cell_dir / "proxy_telemetry.jsonl").write_text("", encoding="utf-8")
    (cell_dir / "summary_commitkv.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete"):
        runner.main(["aggregate", "--output", str(output), "--stage", "closed_loop",
                     "--cells", "bfcl_base__commitkv_b768"])
    coverage = json.loads((output / "aggregation_coverage.json").read_text(encoding="utf-8"))
    assert coverage["counts"]["missing"] == 1
    assert coverage["missing"] == [{
        "stage": "closed_loop", "cell_id": "bfcl_base__commitkv_b768",
        "missing_artifacts": [str(cell_dir / "server_telemetry.jsonl")],
    }]
