"""Official tau2 selection, scoring, and agent-only identity seams."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from benchmarks.adapters import tau2_adapter as tau2
from benchmarks.measurement.telemetry import canonical_sha256


def _result(task_id: str, reward=None, *, termination="agent_stop"):
    return {"task_id": task_id, "trial": 0,
            "termination_reason": termination,
            "messages": [{"role": "assistant", "content": "Done."}],
            "reward_info": {"reward": reward} if reward is not None else None}


def test_selected_task_ids_uses_official_resolver_and_checkout(monkeypatch, tmp_path):
    source = tmp_path / "tau2"
    (source / "src" / "tau2").mkdir(parents=True)
    seen = []

    def fake_run(command, **kwargs):
        seen.append((command, kwargs))
        return types.SimpleNamespace(stdout='["11", "19"]\n')

    monkeypatch.setattr(tau2, "run_owned", fake_run)
    assert tau2.selected_task_ids(source, "/venv/python", task_set="airline",
                                  split="base", task_ids=["11", "19"]) == ["11", "19"]
    command, options = seen[0]
    assert command[:2] == ["/venv/python", "-c"]
    assert json.loads(command[3]) == ["airline", "base", ["11", "19"], None]
    assert options["cwd"] == source.resolve()
    assert str(source.resolve() / "src") in options["env"]["PYTHONPATH"]


@pytest.mark.parametrize("bad", ['["11", "11"]', '[]', '["12"]'])
def test_selected_task_ids_rejects_wrong_official_inventory(monkeypatch, tmp_path, bad):
    source = tmp_path / "tau2"
    (source / "src" / "tau2").mkdir(parents=True)
    monkeypatch.setattr(tau2, "run_owned",
                        lambda *_args, **_kwargs: types.SimpleNamespace(stdout=bad))
    with pytest.raises(RuntimeError, match="task resolver"):
        tau2.selected_task_ids(source, "/venv/python", task_ids=["11"])


def test_run_tau2_scores_exact_official_task_and_routes_user_raw(monkeypatch, tmp_path):
    source = tmp_path / "tau2"
    (source / "src" / "tau2").mkdir(parents=True)
    commands = []
    sims = source / "data" / "simulations" / "run_11"
    sims.mkdir(parents=True)

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        if "-c" in command:
            return types.SimpleNamespace(
                stdout='["11"]' if "get_tasks" in command[2] else "[]")
        if "run" in command:
            (sims / "results.json").write_text(
                json.dumps({"simulations": [_result("11")]}), encoding="utf-8")
            events = out / "measurement" / "harness_events.jsonl"
            events.parent.mkdir(parents=True, exist_ok=True)
            events.write_text("\n".join(json.dumps(row) for row in (
                {"event_type": "episode_start", "episode_id": "11"},
                {"event_type": "decision", "episode_id": "11"},
                {"event_type": "episode_end", "episode_id": "11", "status": "ok"},
            )) + "\n", encoding="utf-8")
        elif "evaluate-trajs" in command:
            (sims / "updated_results.json").write_text(
                json.dumps({"simulations": [_result("11", 1.0)]}), encoding="utf-8")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(tau2, "run_owned", fake_run)
    out = tmp_path / "out"
    summary = tau2.run_tau2(
        "http://agent", "http://raw", out, tau2_dir=source,
        python="/venv/python", run_name="run_11", task_ids=["11"],
        native=True, model="served")
    assert summary["n"] == 1
    assert summary["task_ids"] == ["11"]
    assert summary["task_rows"][0]["semantic_score"] == 1.0
    assert (out / "official" / "updated_results.json").is_file()
    run, options = next((command, opts) for command, opts in commands if "run" in command)
    assert run[run.index("--task-ids") + 1] == "11"
    agent = json.loads(run[run.index("--agent-llm-args") + 1])
    user = json.loads(run[run.index("--user-llm-args") + 1])
    assert agent["api_base"] == "http://agent/v1"
    assert agent["max_tokens"] == 4096
    assert agent["num_retries"] == user["num_retries"] == 0
    assert user["api_base"] == "http://raw/v1"
    assert run[run.index("--max-retries") + 1] == "0"
    assert run[run.index("--hallucination-retries") + 1] == "0"
    assert options["env"]["C2KV_TAU2_NATIVE"] == "1"
    evaluation_env = next(opts["env"] for command, opts in commands
                          if "evaluate-trajs" in command)
    assert "C2KV_TAU2_TELEMETRY_PATH" not in evaluation_env
    assert "C2KV_TAU2_NATIVE" not in evaluation_env


def test_terminal_gate_rejects_missing_or_failed_task(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"simulations": [_result("11", 1.0)]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="terminal-state mismatch"):
        tau2._terminal_results(path, ["11", "19"], 1)
    path.write_text(json.dumps({"simulations": [
        _result("11", 1.0, termination="infrastructure_error")]}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="infrastructure_error"):
        tau2._terminal_results(path, ["11"], 1)


def test_full_replay_source_requires_exact_official_task_coverage(tmp_path):
    path = tmp_path / "prefixes.jsonl"
    payload = {"messages": [{"role": "user", "content": "Help"}],
               "c2kv_measurement_session_id": "11"}
    row = {"event_type": "recorded_prefix", "source_arm": "full",
           "replay_payload": payload, "canonical_sha256": canonical_sha256(payload)}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    tau2._validate_recorded_prefixes(path, ["11"])
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        tau2._validate_recorded_prefixes(path, ["11", "19"])
    del payload["c2kv_measurement_session_id"]
    row["canonical_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="official task identity"):
        tau2._validate_recorded_prefixes(path, ["11"])


def test_harness_events_reject_restarted_or_failed_official_trial(tmp_path):
    path = tmp_path / "harness_events.jsonl"
    rows = [
        {"event_type": "episode_start", "episode_id": "11"},
        {"event_type": "decision", "episode_id": "11"},
        {"event_type": "episode_end", "episode_id": "11", "status": "ok"},
    ]
    path.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    tau2._validate_harness_events(path, ["11"])
    with pytest.raises(RuntimeError, match="retry or missing episode"):
        tau2._validate_harness_events(path, ["11", "19"])
    path.write_text("\n".join(map(json.dumps, rows + rows)) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="retry or missing episode"):
        tau2._validate_harness_events(path, ["11"])
    rows[-1]["status"] = "error"
    path.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ended in error"):
        tau2._validate_harness_events(path, ["11"])
    rows[-1]["status"] = "ok"
    rows[1]["error"] = "upstream request failed"
    path.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ended in error"):
        tau2._validate_harness_events(path, ["11"])


@pytest.mark.parametrize("native", [False, True])
def test_hook_adds_task_identity_only_to_proxy_agent_and_records_episode(
        monkeypatch, tmp_path, native):
    calls = []
    events = tmp_path / "events.jsonl"

    class Environment:
        def get_response(self, message):
            return types.SimpleNamespace(
                error=False, model_dump=lambda **_kwargs: {"content": "ok"})

    llm_utils = types.ModuleType("tau2.utils.llm_utils")
    llm_utils.completion = lambda *_args, **kwargs: calls.append(kwargs) or types.SimpleNamespace(id="req-1")
    llm_agent = types.ModuleType("tau2.agent.llm_agent")
    llm_agent.generate = lambda **kwargs: llm_utils.completion(**kwargs) or None
    batch = types.ModuleType("tau2.runner.batch")

    def run_single_task(_config, task, **_kwargs):
        llm_agent.generate(call_name="agent_response", model="openai/served")
        llm_utils.completion(model="openai/user")
        environment = Environment()
        message = types.SimpleNamespace(
            requestor="assistant", model_dump=lambda **_kwargs: {"name": "lookup"})
        environment.get_response(message)
        environment.get_response(types.SimpleNamespace(requestor="user"))
        return types.SimpleNamespace(task_id=task.id)

    batch.run_single_task = run_single_task
    modules = {
        "tau2": types.ModuleType("tau2"),
        "tau2.agent": types.ModuleType("tau2.agent"),
        "tau2.agent.llm_agent": llm_agent,
        "tau2.environment": types.ModuleType("tau2.environment"),
        "tau2.environment.environment": types.ModuleType("tau2.environment.environment"),
        "tau2.runner": types.ModuleType("tau2.runner"),
        "tau2.runner.batch": batch,
        "tau2.utils": types.ModuleType("tau2.utils"),
        "tau2.utils.llm_utils": llm_utils,
    }
    modules["tau2.environment.environment"].Environment = Environment
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setenv("C2KV_TAU2_TELEMETRY_PATH", str(events))
    monkeypatch.setenv("C2KV_TAU2_NATIVE", "1" if native else "0")
    hook_path = Path(__file__).resolve().parent / "tau2_instrumentation" / "c2kv_tau2_hook.py"
    spec = importlib.util.spec_from_file_location("fixture_tau2_hook", hook_path)
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    assert hook.install()
    assert batch.run_single_task(None, types.SimpleNamespace(id="11")).task_id == "11"
    if native:
        assert "extra_body" not in calls[0]
    else:
        assert calls[0]["extra_body"] == {"c2kv_measurement_session_id": "11"}
    assert "extra_body" not in calls[1]
    rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in rows] == [
        "episode_start", "decision", "tool_action", "episode_end"]
    assert all(row["episode_id"] == "11" for row in rows)
