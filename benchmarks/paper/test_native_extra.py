import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from benchmarks.paper import native_extra


def _config(tmp_path):
    return {
        "checkpoint": str(tmp_path / "checkpoint"),
        "model": "c2kv-agent",
        "bench_python": sys.executable,
        "acebench_python": sys.executable,
        "toolsandbox_python": sys.executable,
        "acebench_dir": str(tmp_path / "acebench"),
        "toolsandbox_dir": str(tmp_path / "ToolSandbox"),
        "acebench_language": "en",
        "toolsandbox_suite": "full",
        "toolsandbox_scenarios": [],
        "upstream": "http://127.0.0.1:34000",
        "proxy_port": 34100,
        "c1": {"task_timeout": 120},
    }


def test_official_ace_tasks_and_subset(tmp_path):
    config = _config(tmp_path)
    root = Path(config["acebench_dir"])
    root.mkdir()
    (root / "category.py").write_text(
        "ACE_DATA_CATEGORY = {'agent': ['agent_multi_turn', 'agent_multi_step']}\n",
        encoding="utf-8",
    )
    data = root / "data_all" / "data_en"
    data.mkdir(parents=True)
    (data / "data_agent_multi_turn.json").write_text(
        '{"id":"agent_multi_turn_1"}\n', encoding="utf-8")
    (data / "data_agent_multi_step.json").write_text(
        '{"id":"agent_multi_step_2"}\n', encoding="utf-8")
    assert native_extra.selected_tasks(config, "acebench_agent") == [
        "agent_multi_turn_1", "agent_multi_step_2"]
    assert native_extra.selected_tasks(config, "acebench_agent", ["agent_multi_step_2"]) == [
        "agent_multi_step_2"]
    with pytest.raises(ValueError, match="official split"):
        native_extra.selected_tasks(config, "acebench_agent", ["unknown"])


def test_toolsandbox_uses_official_resolver_and_checks_configured_subset(tmp_path, monkeypatch):
    config = _config(tmp_path)
    Path(config["toolsandbox_dir"]).mkdir()
    observed = {}

    def resolve(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return SimpleNamespace(stdout=json.dumps(["wifi_off", "get_wifi"]))

    monkeypatch.setattr(native_extra, "run_owned", resolve)
    assert native_extra.selected_tasks(config, "toolsandbox") == ["wifi_off", "get_wifi"]
    assert "resolve_scenarios" in observed["command"][2]
    assert observed["kwargs"]["cwd"] == Path(config["toolsandbox_dir"])
    config["toolsandbox_scenarios"] = ["get_wifi"]
    assert native_extra.selected_tasks(config, "toolsandbox") == ["get_wifi"]
    with pytest.raises(ValueError, match="official split"):
        native_extra.selected_tasks(config, "toolsandbox", ["missing"])


@pytest.mark.parametrize("benchmark,namespace,profile", [
    ("acebench_agent", "acebench", "acebench-text-actions-v1"),
    ("toolsandbox", "toolsandbox", "openai-single-task-v1"),
])
def test_server_command_is_native_bare(tmp_path, benchmark, namespace, profile):
    config = _config(tmp_path)
    root = Path(__file__).resolve().parents[2] / "experiments" / "history_system"
    command = native_extra.server_command(
        config, benchmark, "task_1", tmp_path / "native", root,
        tmp_path / "controller.json")
    def value(flag):
        return command[command.index(flag) + 1]
    assert value("--view-mode") == "ac_gist_static"
    assert value("--ratio") == "4"
    assert value("--benchmark") == namespace
    assert value("--source-profile") == profile
    assert value("--model-name") == "c2kv_native_r4"
    assert "--s0-config" not in command
    assert "--shadow-feature-config" not in command
    if benchmark == "acebench_agent":
        assert value("--max-new-tokens") == "1000"


def test_ace_official_score_and_user_model_are_bound(tmp_path, monkeypatch):
    config = _config(tmp_path)
    task = "agent_multi_turn_1"
    output = tmp_path / "task"
    (output / "acebench").mkdir(parents=True)
    observed = {}

    def run_acebench(*args, **kwargs):
        observed.update(kwargs)
        return {"n": 1, "semantic_score": 1.0,
                "selection": {"sources": [{"selected_ids": [task]}]}}

    from benchmarks.adapters import acebench_adapter
    monkeypatch.setattr(acebench_adapter, "run_acebench", run_acebench)
    official = native_extra._run_official(config, "acebench_agent", task,
                                          output, "http://127.0.0.1:34100/v1",
                                          "c2kv_native_r4")
    assert observed["model"] == "c2kv_native_r4"
    assert observed["user_model"] == "c2kv-agent"
    assert official["task_rows"][0]["semantic_score"] == 1.0
    with mock.patch.object(acebench_adapter, "run_acebench", return_value={
        "n": 0, "semantic_score": 1.0, "selection": {"sources": []}}):
        with pytest.raises(RuntimeError, match="selection or scoring"):
            native_extra._run_official(config, "acebench_agent", task, output,
                                       "http://127.0.0.1:34100/v1", "c2kv_native_r4")


def test_replay_requires_real_ace_receipts():
    payload = {"messages": [{"role": "system", "content": "system"},
                            {"role": "user", "content": "question"}],
               "temperature": 0.001, "top_p": 1, "max_tokens": 1000,
               "stream": False, "c2kv_measurement_session_id": "old"}
    with pytest.raises(ValueError, match="recorded execution receipts"):
        native_extra.replay_payload(payload, "agent_multi_turn_1", 0)
    payload["c2kv_ace_source"] = {"version": "acebench-text-actions-v1", "receipts": []}
    normalized = native_extra.replay_payload(payload, "agent_multi_turn_1", 0)
    assert normalized["messages"] == payload["messages"]
    assert normalized["temperature"] == 0.001
    assert normalized["max_tokens"] == 1000
    assert normalized["store"] is False
    assert "c2kv_measurement_session_id" not in normalized
    assert normalized["c2kv_eval_context"]["task_id"] == "agent_multi_turn_1"
