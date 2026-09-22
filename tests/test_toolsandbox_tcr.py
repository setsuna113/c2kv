"""ToolSandbox official-task and T/C/R driver contracts without model execution."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from generality import (
    historykv_cell, scheduler_npu, session_tracer_cell, toolsandbox_harness,
)


def _c2kv_driver():
    sys.modules.setdefault("current", types.SimpleNamespace())
    sys.modules.setdefault("evidence_sets", types.SimpleNamespace())
    sys.modules.setdefault("c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell
    return c2kv_cell


def _official(path: Path, task_id: str, score: float, failure: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": toolsandbox_harness.SCHEMA, "task_id": task_id,
        "n": 1, "semantic_score": score,
        "official_scorer": "tool_sandbox official CLI",
        "adapter_summary": {"n": 1, "scenario_ids": [task_id],
                            "scenario_manifest": {"scenario_ids": [task_id]},
                            "task_failures": ({failure: [task_id]} if failure else {})},
        "task_failure_kind": failure,
    }))


@pytest.mark.parametrize("code", (
    "acon_history_budget_exceeded", "hiagent_history_budget_exceeded",
    "hiagent_retrieval_budget_exceeded",
))
def test_text_budget_receipt_completes_without_inflating_official_scored(tmp_path, code):
    task_id = "budget_3_distraction_tools"
    path = tmp_path / "batches" / "one" / "toolsandbox_worker" / task_id / "official_summary.json"
    _official(path, task_id, 0.0, code)
    assert toolsandbox_harness.completed_task(tmp_path, task_id)
    summary = toolsandbox_harness.score_summary({
        "cell_dir": str(tmp_path), "cell_id": "budget", "task_ids": [task_id],
    })
    assert summary["semantic_score"] == 0.0
    assert summary["n_method_failures"] == 1
    assert summary["n_official_scored"] == 0
    assert summary["pending_task_ids"] == []
    raw = json.loads(path.read_text())
    raw["adapter_summary"]["task_failures"] = {code: ["another_scenario"]}
    path.write_text(json.dumps(raw))
    assert not toolsandbox_harness.completed_task(tmp_path, task_id)


def _capacity_final(batch: Path, task_id: str, *, pending=0):
    server = batch / "server"
    server.mkdir(parents=True, exist_ok=True)
    (server / "ready.json").write_text(json.dumps({
        "schema": "a-event-native-server-v1", "status": "ready",
        "benchmark": "toolsandbox", "allowed_task_ids": [task_id],
        "run_id": "capacity-run",
    }))
    (server / "steps.jsonl").write_text(json.dumps({
        "schema": "a-event-native-exact-step-v1", "status": "failed",
        "session_id": f"toolsandbox/{task_id}/attempt-0",
        "failure_kind": "method_failure", "failure_code": "c2kv_capacity_infeasible",
        "error": {"type": "CapacityInfeasible", "message": "cannot fit"},
        "response": None,
    }) + "\n")
    (server / "final.json").write_text(json.dumps({
        "status": "stopped", "cost_summary": {"generation_attempts": 0},
        "journal_summary": {"schema": "a-runtime-attempt-journal-v1",
                            "started": 0, "completed": 0,
                            "failed": 0, "pending": pending},
        "api_health": {"allowed_task_ids": [task_id], "terminal": False,
                       "terminal_reason": None},
    }))


@pytest.mark.parametrize("backend", ("c2kv", "h2o", "snapkv", "pyramidkv"))
@pytest.mark.parametrize("condition", (
    "compression_full_budget", "tracer_history", "recovery_off_same_initial"))
def test_scheduler_admits_toolsandbox_tcr(backend, condition):
    assert scheduler_npu.driver_supports_cell({
        "backend": backend, "condition": condition,
        "benchmark_key": "toolsandbox",
    })


def test_scored_task_requires_matching_official_id_and_finite_score(tmp_path):
    cell = {"cell_id": "ts__c2kv__K0__tracer_history", "cell_dir": str(tmp_path),
            "task_ids": ["first", "second"]}
    path = (tmp_path / "batches" / "a" / "toolsandbox_worker" / "first" /
            "official_summary.json")
    _official(path, "other", 0.3)
    assert not toolsandbox_harness.completed_task(tmp_path, "first")
    _official(path, "first", float("nan"))
    assert not toolsandbox_harness.completed_task(tmp_path, "first")
    _official(path, "first", 0.3)
    assert toolsandbox_harness.completed_task(tmp_path, "first")
    summary = toolsandbox_harness.score_summary(cell)
    assert summary["n_official_scored"] == 1
    assert summary["pending_task_ids"] == ["second"]
    assert summary["semantic_score"] is None
    assert summary["task_rows"][0]["semantic_score"] == pytest.approx(0.3)


def test_historykv_requires_official_score_before_counting_done(tmp_path):
    task_id = "task_3_distraction_tools"
    task_dir = tmp_path / "tasks" / task_id
    source = (task_dir / "attempts" / "a1" / "toolsandbox" /
              "toolsandbox_worker" / task_id / "official_summary.json")
    source.parent.mkdir(parents=True)
    (task_dir / "done.json").write_text(json.dumps({
        "task_id": task_id, "status": "completed",
        "semantic_score": 0.5,
        "official_summary": str(source.relative_to(task_dir)),
    }))
    cell = {"cell_id": "ts__h2o__K0__compression_full_budget",
            "cell_dir": str(tmp_path), "task_ids": [task_id]}
    assert historykv_cell.toolsandbox_task_result(task_dir, task_id) is None
    assert historykv_cell.toolsandbox_score_summary(cell)["semantic_score"] is None
    _official(source, task_id, 0.5)
    assert historykv_cell.toolsandbox_task_result(task_dir, task_id) is not None
    summary = historykv_cell.toolsandbox_score_summary(cell)
    assert summary["n_official_scored"] == 1
    assert summary["semantic_score"] == pytest.approx(0.5)


def test_official_worker_selects_one_scenario_and_separates_agent_user_routes(
    tmp_path, monkeypatch,
):
    adapter_file = tmp_path / "benchmarks" / "adapters" / "toolsandbox_adapter.py"
    adapter_file.parent.mkdir(parents=True)
    adapter_file.write_text("")
    monkeypatch.setattr(toolsandbox_harness, "paper_source", lambda: tmp_path)
    calls = []

    def run_ts(agent_url, out, **kwargs):
        calls.append((agent_url, out, kwargs))
        return {"n": 1, "scenario_ids": ["task_3_distraction_tools"],
                "scenario_manifest": {"scenario_ids": ["task_3_distraction_tools"]},
                "semantic_score": 0.75}

    fake = types.SimpleNamespace(run_ts=run_ts)
    monkeypatch.setitem(sys.modules, "benchmarks.adapters.toolsandbox_adapter", fake)
    import benchmarks.adapters as adapters
    monkeypatch.setattr(adapters, "toolsandbox_adapter", fake, raising=False)
    out = tmp_path / "official"
    result = toolsandbox_harness.run_task(
        "task_3_distraction_tools", "http://agent/v1", "http://raw/v1",
        out, tmp_path / "ToolSandbox", "ts-python", "gen-c1000")
    assert result["semantic_score"] == 0.75
    assert toolsandbox_harness.official_result(
        out / "official_summary.json", "task_3_distraction_tools") is not None
    agent, selected_out, options = calls[0]
    assert agent == "http://agent/v1" and selected_out == out
    assert options["user_base_url"] == "http://raw/v1"
    assert options["scenarios"] == ["task_3_distraction_tools"]
    assert options["expected"] == 1 and options["parallel"] == 1


def test_c2kv_uses_one_task_official_worker_and_raw_user_endpoint(
    tmp_path, monkeypatch,
):
    driver = _c2kv_driver()
    task_id = "demo_3_distraction_tools"
    cell = {"cell_dir": str(tmp_path), "benchmark": "toolsandbox",
            "python_sgl": "sgl-python", "python_bench": "ts-python",
            "benchmark_dir": "/official/ToolSandbox",
            "model_name": "gen_c2kv_K0_tracer_history",
            "caps": {"task_timeout": 2}, "sglang_backend_url": "http://raw:36200"}
    monkeypatch.setattr(driver, "_runtime_retrieval_cell", lambda cell, out: cell)
    monkeypatch.setattr(driver, "server_command", lambda *args: ["fake-controller"])
    monkeypatch.setattr(driver, "cost_finalization", lambda out: {"status": "valid"})
    monkeypatch.setattr(driver, "stop_owned_group", lambda proc: None)
    monkeypatch.setattr(driver, "UpstreamLiveness", lambda url: lambda: None)

    class Process:
        returncode = None

        def poll(self):
            return None

    commands = []

    def start(command, *args, **kwargs):
        commands.append(command)
        out = tmp_path / "batches" / "one"
        if command == ["fake-controller"]:
            (out / "server").mkdir()
            (out / "server" / "ready.json").write_text("{}")
        else:
            _official(out / "toolsandbox_worker" / task_id / "official_summary.json",
                      task_id, 0.4)
        return Process()

    monkeypatch.setattr(driver.subprocess, "Popen", start)
    monkeypatch.setattr(driver, "wait_owned_worker", lambda *args, **kwargs: 0)
    result = driver.run_task(cell, [task_id], 45001, "one")
    assert result["status"] == "completed"
    assert result["healthy"] == [task_id]
    worker = commands[1]
    assert worker[worker.index("--base-url") + 1] == "http://127.0.0.1:45001/v1"
    assert worker[worker.index("--user-base-url") + 1] == "http://raw:36200/v1"
    assert worker[worker.index("--task-id") + 1] == task_id
    assert worker[worker.index("--native-server-dir") + 1] == str(
        tmp_path / "batches" / "one" / "server")


def test_c2kv_capacity_zero_generation_is_terminal_only_with_clean_cost(tmp_path):
    driver = _c2kv_driver()
    task_id = "capacity_3_distraction_tools"
    batch = tmp_path / "batches" / "one"
    path = batch / "toolsandbox_worker" / task_id / "official_summary.json"
    _official(path, task_id, 0.0, "c2kv_capacity_infeasible")
    _capacity_final(batch, task_id)
    cell = {"cell_id": "ts", "cell_dir": str(tmp_path), "task_ids": [task_id]}
    assert driver.c2kv_toolsandbox_task_completed(tmp_path, task_id)
    summary = driver.c2kv_toolsandbox_score_summary(cell)
    assert summary["n_official_scored"] == 0
    assert summary["n_method_failures"] == 1
    assert summary["method_failure_task_ids"] == [task_id]
    assert summary["semantic_score"] == 0.0

    _capacity_final(batch, task_id, pending=1)
    assert not driver.c2kv_toolsandbox_task_completed(tmp_path, task_id)
    assert driver.c2kv_toolsandbox_score_summary(cell)["pending_task_ids"] == [task_id]


def test_tracer_binds_one_server_owned_scenario_and_rejects_client_identity():
    payload = {"messages": [{"role": "user", "content": "first"},
                            {"role": "user", "content": "second"}],
               "c2kv_measurement_session_id": "opaque-episode-instance"}
    bound = session_tracer_cell.bind_toolsandbox_task(payload, "official_task", 3)
    assert bound["c2kv_eval_context"] == {
        "benchmark": "toolsandbox", "task_id": "official_task",
        "user_turn": 1, "step": 3, "attempt": 0,
    }
    assert "c2kv_eval_context" not in payload
    with pytest.raises(ValueError, match="server-owned"):
        session_tracer_cell.bind_toolsandbox_task(
            {**payload, "c2kv_eval_context": {"task_id": "other"}},
            "official_task", 3)
