"""Single-cell client keeps paper adapter commands and output identities."""
import json
from pathlib import Path

import pytest

from benchmarks.paper import history_kv_client as client
from benchmarks.paper import runner


def config():
    # These tests freeze the legacy proxy client protocol. Marked native
    # defaults and their aliases are covered in test_unified_runtime.py.
    cfg = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for row in cfg["methods"]:
        for key in ("history_runtime", "history_backend", "recovery_policy", "compression_ratio"):
            row.pop(key, None)
        if row["method"] == "C2KV":
            row.pop("history_budget_tokens", None)
    return cfg


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
    assert resolved["history_kv_budget_tokens"] == 768
    assert len(resolved["methods"]) == len(config()["methods"])


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


def test_client_reuses_existing_absolute_cell_or_adds_independent_variant(tmp_path):
    cfg = config()
    cfg["history_kv_budget_tokens"] = 384
    resolved, cell, command, _ = client.plan(
        cfg, "bfcl_base", "commitkv=384", tmp_path, "http://localhost:36200")
    assert len(resolved["methods"]) == len(cfg["methods"])
    assert cell["cell_id"] == "bfcl_base__commitkv_b384"
    assert command[command.index("--history-kv-target-tokens") + 1] == "384"
    expanded, variant, command, _ = client.plan(
        cfg, "bfcl_base", "commitkv=1024", tmp_path, "http://localhost:36200")
    assert expanded["history_kv_budget_tokens"] == 384
    assert len(expanded["methods"]) == len(cfg["methods"]) + 1
    assert variant["cell_id"] == "bfcl_base__commitkv_b1024"
    assert command[command.index("--history-kv-target-tokens") + 1] == "1024"


def test_client_budget_spec_fills_missing_shared_b_without_second_argument(tmp_path):
    resolved, cell, _, _ = client.plan(
        config(), "bfcl_base", "history_kv_h2o_r25_persistent=1024",
        tmp_path, "http://localhost:36200")
    assert resolved["history_kv_budget_tokens"] == 1024
    assert cell["cell_id"] == "bfcl_base__history_kv_h2o_persistent_b1024"
    assert all(m.get("history_budget_tokens") == 1024 for m in resolved["methods"]
               if m["arm"] in {"commitkv", "agentkv", "history_kv_h2o_r25_persistent",
                               "history_kv_snapkv_r25_persistent",
                               "history_kv_pyramidkv_r25_persistent"})
    cfg = config()
    cfg["methods"][0]["history_retention_ratio"] = 0.5
    with pytest.raises(ValueError, match="history_retention_ratio"):
        client.plan(cfg, "bfcl_base", "commitkv=384", tmp_path, "http://localhost:36200")


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
    frozen = json.loads((output / "config.resolved.json").read_text())
    assert frozen["history_kv_budget_tokens"] == 768
    assert next(m for m in frozen["methods"] if m["arm"] == "agentkv")["history_budget_tokens"] == 768
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
