"""Native tau2/ToolSandbox agent clients wait for one whole server decision.

box9 tau2__racer_v2_commitkv_c1_v2_verified_b256, task 1: the capacity-gated
source allocation made controller work grow to 90 s per prepare and over
9 minutes in one post-draft reconsider, while tau2's LiteLLM agent kept its
600 s default on the native route and ended the task as infrastructure_error.
"""
import json
import sys
from pathlib import Path

import pytest

from benchmarks.adapters.generation_deadline import native_decision_client_timeout
from benchmarks.paper import native_extra


def _config(tmp_path, **changes):
    config = {"model": "c2kv-agent", "bench_python": sys.executable,
              "toolsandbox_python": sys.executable, "tau2_dir": str(tmp_path / "tau2"),
              "toolsandbox_dir": str(tmp_path / "ToolSandbox"),
              "upstream": "http://127.0.0.1:34000", "c1": {"task_timeout": 120}}
    config.update(changes)
    return config


def _task(tmp_path, generations=2):
    task_out = tmp_path / "task"
    (task_out / "server").mkdir(parents=True)
    (task_out / "server" / "ready.json").write_text(json.dumps(
        {"route_contract": {"max_generations_per_decision": generations}}), encoding="utf-8")
    return task_out


def test_native_decision_timeout_covers_every_generation_and_keeps_the_default():
    assert native_decision_client_timeout(10800, 2) == 2 * 10800 + 90
    assert native_decision_client_timeout(600, 2) is None
    with pytest.raises(ValueError, match="max_generations"):
        native_decision_client_timeout(10800, 0)


@pytest.mark.parametrize("deadline,expected", [(None, None), (600.0, None), (10800.0, 21690.0)])
def test_tau2_native_agent_gets_the_decision_timeout_only_for_a_configured_deadline(
        tmp_path, monkeypatch, deadline, expected):
    from benchmarks.adapters import tau2_adapter

    config = _config(tmp_path, **({} if deadline is None else {"generation_timeout": deadline}))
    task_out = _task(tmp_path)
    observed = {}

    def run_tau2(*args, **kwargs):
        observed.update(kwargs)
        return {"n": 1, "task_ids": ["1"], "semantic_score": 1.0,
                "task_rows": [{"termination": "agent_stop"}]}

    monkeypatch.setattr(tau2_adapter, "run_tau2", run_tau2)
    (task_out / "tau2").mkdir()
    native_extra._run_official(config, "tau2", "1", task_out, "http://127.0.0.1:34100/v1",
                               "c2kv-agent")
    assert observed.get("agent_timeout") == expected
    assert ("agent_timeout" in observed) is (expected is not None)


def test_toolsandbox_native_agent_gets_the_same_decision_timeout(tmp_path, monkeypatch):
    from benchmarks.adapters import toolsandbox_adapter

    config = _config(tmp_path, generation_timeout=3600.0)
    task_out = _task(tmp_path, generations=3)
    observed = {}

    def run_ts(*args, **kwargs):
        observed.update(kwargs)
        return {"n": 1, "scenario_ids": ["scenario"], "semantic_score": 1.0}

    monkeypatch.setattr(toolsandbox_adapter, "run_ts", run_ts)
    (task_out / "toolsandbox").mkdir()
    native_extra._run_official(config, "toolsandbox", "scenario", task_out,
                               "http://127.0.0.1:34100/v1", "c2kv-agent")
    assert observed["agent_timeout"] == 3 * 3600.0 + 90
