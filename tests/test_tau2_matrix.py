"""CPU seams for the NPU tau2 cells across C/T/R and backends."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from generality import historykv_cell, scheduler_npu, session_tracer_cell


def _c2kv_driver():
    sys.modules.setdefault("current", types.SimpleNamespace())
    sys.modules.setdefault("evidence_sets", types.SimpleNamespace())
    sys.modules.setdefault("c1_artifact_binding", types.SimpleNamespace(
        bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})))
    from generality import c2kv_cell
    return c2kv_cell


@pytest.mark.parametrize("backend", ("c2kv", "h2o", "snapkv", "pyramidkv"))
@pytest.mark.parametrize("condition", (
    "compression_full_budget", "tracer_history", "recovery_off_same_initial"))
def test_scheduler_admits_each_tau2_tcr_cell(backend, condition):
    cell = {"backend": backend, "condition": condition, "benchmark_key": "tau2"}
    assert scheduler_npu.driver_supports_cell(cell)


def test_c2kv_worker_uses_single_task_agent_endpoint_and_raw_user_endpoint(
    tmp_path, monkeypatch,
):
    driver = _c2kv_driver()
    cell = {"cell_dir": str(tmp_path), "benchmark": "tau2", "caps": {"task_timeout": 2},
            "sglang_backend_url": "http://raw:36200"}
    monkeypatch.setattr(driver, "_runtime_retrieval_cell", lambda cell, out: cell)
    monkeypatch.setattr(driver, "server_command", lambda *args: ["fake-controller"])
    monkeypatch.setattr(driver, "cost_finalization", lambda out: {"status": "valid"})
    monkeypatch.setattr(driver, "stop_owned_group", lambda proc: None)
    monkeypatch.setattr(driver, "UpstreamLiveness", lambda url: lambda: None)

    class Process:
        returncode = None

        def poll(self):
            return None

    def start_server(*args, **kwargs):
        out = tmp_path / "batches" / "one"
        (out / "server").mkdir()
        (out / "server" / "ready.json").write_text("{}")
        return Process()

    monkeypatch.setattr(driver.subprocess, "Popen", start_server)
    seen = []

    def run_task(cell, task_id, agent, user, out):
        seen.append((task_id, agent, user, out))
        return {"status": "completed"}

    monkeypatch.setattr(driver, "run_tau2_task", run_task)
    monkeypatch.setattr(driver, "completed_tau2_task", lambda out, task_id: True)
    result = driver.run_task(cell, ["0"], 45001, "one")
    assert result["status"] == "completed"
    assert result["healthy"] == ["0"]
    assert seen == [("0", "http://127.0.0.1:45001", "http://raw:36200",
                     tmp_path / "batches" / "one" / "tau2_worker" / "0")]


@pytest.mark.parametrize("problem", (None, "missing", "pending", "failed", "server_failed"))
def test_typed_tau2_live_batch_requires_clean_final_journal(tmp_path, monkeypatch, problem):
    driver = _c2kv_driver()
    cell = {"cell_dir": str(tmp_path), "benchmark": "tau2",
            "caps": {"task_timeout": 2}, "sglang_backend_url": "http://raw:36200"}
    monkeypatch.setattr(driver, "_runtime_retrieval_cell", lambda cell, out: cell)
    monkeypatch.setattr(driver, "server_command", lambda *args: ["fake-controller"])
    monkeypatch.setattr(driver, "stop_owned_group", lambda proc: None)
    monkeypatch.setattr(driver, "UpstreamLiveness", lambda url: lambda: None)
    monkeypatch.setattr(driver, "completed_tau2_task", lambda out, task_id: True)
    monkeypatch.setattr(driver, "run_tau2_task", lambda *args: {
        "status": "completed", "task_failure_kind": "generation_cap_reached"})

    class Process:
        returncode = None

        def poll(self):
            return None

    def start_server(*args, **kwargs):
        server = tmp_path / "batches" / "one" / "server"
        server.mkdir(parents=True)
        (server / "ready.json").write_text("{}", encoding="utf-8")
        if problem != "missing":
            (server / "final.json").write_text(json.dumps({
                "status": "failed" if problem == "server_failed" else "stopped",
                "cost_summary": {"recorded": True},
                "journal_summary": {
                    "schema": "a-runtime-attempt-journal-v1", "started": 2,
                    "completed": 1 if problem in {"pending", "failed"} else 2,
                    "pending": int(problem == "pending"),
                    "failed": int(problem == "failed")},
                "api_health": {
                    "allowed_task_ids": ["0"],
                    "terminal_reason": "generation_cap_reached",
                    "generation_calls_reserved": 2, "max_generation_calls": 2},
            }), encoding="utf-8")
        return Process()

    monkeypatch.setattr(driver.subprocess, "Popen", start_server)
    result = driver.run_task(cell, ["0"], 45001, "one")
    batch = tmp_path / "batches" / "one"
    if problem is None:
        assert result["status"] == "completed"
        assert result["cost_finalization"]["status"] == "valid"
        assert (batch / "done.json").is_file()
    else:
        assert result["status"] == "failed"
        assert result["cost_finalization"]["status"] != "valid"
        assert (batch / "status.json").is_file()
        assert not (batch / "done.json").exists()


@pytest.mark.parametrize("backend,condition,arm,target", [
    ("h2o", "recovery_off_same_initial", "gen_h2o_k0", 768),
    ("h2o", "compression_full_budget", "gen_h2o_b0", 1792),
    ("snapkv", "recovery_off_same_initial", "gen_snapkv_persistent_k0", 768),
    ("snapkv", "compression_full_budget", "gen_snapkv_persistent_b0", 1792),
    ("pyramidkv", "recovery_off_same_initial", "gen_pyramidkv_k0", 768),
    ("pyramidkv", "compression_full_budget", "gen_pyramidkv_b0", 1792),
])
def test_historykv_tau2_uses_frozen_k_or_b_and_raw_user_endpoint(
    tmp_path, monkeypatch, backend, condition, arm, target,
):
    directory = tmp_path / "cell"
    directory.mkdir()
    cell = {"cell_id": "tau2-cell", "cell_dir": str(directory), "backend": backend,
            "working_point": "K0", "condition": condition, "benchmark": "tau2",
            "task_ids": ["0"], "budget_tokens": {"K": 768, "B": 1792},
            "sglang_backend_url": "http://raw:36200"}
    manifest = tmp_path / "cell.json"
    manifest.write_text(json.dumps(cell))
    monkeypatch.setattr(historykv_cell, "resolve_free_port", lambda *args: 45000)
    proxy_args = []
    monkeypatch.setattr(historykv_cell, "start_proxy", lambda *args: proxy_args.append(args))
    monkeypatch.setattr(historykv_cell, "stop", lambda proc: None)
    seen = []

    def run_task(cell, task_id, agent, user, out):
        seen.append((task_id, agent, user, out))
        return {"task_id": task_id, "status": "completed", "semantic_score": 1.0}

    monkeypatch.setattr(historykv_cell, "run_tau2_task", run_task)
    monkeypatch.setattr(historykv_cell, "completed_tau2_task", lambda out, task_id: True)
    statuses = []
    monkeypatch.setattr(historykv_cell, "write_cell_status",
                        lambda cell, status: statuses.append(status))
    assert historykv_cell.main(["--cell", str(manifest), "--proxy-port", "45000"]) == 0
    assert proxy_args[0][0] == arm and proxy_args[0][-1] == target
    assert seen == [("0", "http://127.0.0.1:45000", "http://raw:36200",
                     directory / "tasks" / "0")]
    assert statuses[0]["status"] == "complete" and statuses[0]["n_completed"] == 1


def test_tracer_tau2_binds_server_owned_task_and_runs_one_official_task(
    tmp_path, monkeypatch,
):
    bound = session_tracer_cell.bind_tau2_task(
        {"messages": [{"role": "system", "content": "agent"},
                      {"role": "user", "content": "book flight"}]}, "0", 2)
    assert bound["c2kv_eval_context"] == {
        "benchmark": "tau2", "task_id": "0", "user_turn": 0, "step": 2, "attempt": 0}
    with pytest.raises(ValueError, match="server-owned"):
        session_tracer_cell.bind_tau2_task(
            {"c2kv_eval_context": {}, "messages": []}, "0", 0)

    root = tmp_path / "generation"
    (root / "config").mkdir(parents=True)
    (root / "config" / "budgets_resolved.json").write_text(json.dumps({
        "working_points": {"K0": {"kv_token_equivalents": {"K": 768, "R": 1024, "B": 1792}}}}))
    monkeypatch.setattr(session_tracer_cell, "GENERATION_ROOT", root)
    directory = tmp_path / "cell"
    cell = {"cell_id": "tau2-tracer", "cell_dir": str(directory),
            "benchmark": "tau2", "backend": "h2o", "working_point": "K0",
            "threshold": 0.3, "task_ids": ["0"],
            "sglang_backend_url": "http://raw:36200",
            "caps": {"task_timeout": 2}}
    manifest = tmp_path / "cell.json"
    manifest.write_text(json.dumps(cell))

    class Server:
        def serve_forever(self):
            return None

    monkeypatch.setattr(session_tracer_cell, "run_server",
                        lambda cell, batch, port, out: (Server(), {}))
    monkeypatch.setattr(session_tracer_cell, "stop_server", lambda *args: None)
    seen = []
    monkeypatch.setattr(session_tracer_cell, "run_tau2_task",
                        lambda cell, task_id, agent, user, out: (
                            seen.append((task_id, agent, user, out)) or {"status": "completed"}))
    monkeypatch.setattr(session_tracer_cell, "completed_task_ids",
                        lambda directory, benchmark, expected: set(expected))
    statuses = []
    monkeypatch.setattr(session_tracer_cell, "write_cell_status",
                        lambda cell, status: statuses.append(status))
    assert session_tracer_cell.main(["--cell", str(manifest), "--port-base", "45100"]) == 0
    assert seen == [("0", "http://127.0.0.1:45100", "http://raw:36200",
                     directory / "batches" / "000_0" / "tau2_worker" / "0")]
    assert statuses[0]["status"] == "complete"
