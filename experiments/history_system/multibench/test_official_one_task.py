"""CPU-only contract tests for the one-task official harness worker."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.history_system.multibench import official_one_task as worker


def _source(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    file = root / "runner.py"
    file.write_text("# frozen\n", encoding="utf-8")
    return root, {
        "components": [{
            "path": str(root), "kind": "files",
            "files": {"runner.py": hashlib.sha256(file.read_bytes()).hexdigest()},
        }]
    }


def _task(tmp_path: Path, benchmark="acon_appworld", task_id="3d9a636_1"):
    root, binding = _source(tmp_path)
    cap = 2048 if benchmark == "acon_appworld" else 1200 if benchmark == "acebench" else 4096
    return {
        "schema": worker.TASK_SCHEMA, "benchmark": benchmark, "task_id": task_id,
        "benchmark_dir": str(root), "bench_python": "python",
        "user_base_url": "" if benchmark == "acon_appworld" else "http://127.0.0.1:2/v1",
        "source_binding": binding, "max_new_tokens": cap,
        **({"run_name": "frozen-task"} if benchmark == "tau2" else {}),
        **({"appworld_root": str(root)} if benchmark == "acon_appworld" else {}),
    }


def _server(task, model="c2kv-agent"):
    return {
        "status": "ready", "source_profile": worker.SOURCE_PROFILE,
        "benchmark": task["benchmark"], "allowed_task_ids": [task["task_id"]],
        "base_url": "http://127.0.0.1:31000/v1", "model_name": model,
        "max_new_tokens": task["max_new_tokens"],
    }


def test_file_source_binding_is_verified_and_tampering_fails(tmp_path):
    task = worker._validate_task(_task(tmp_path))
    assert worker._verify_source_binding(task)[0]["kind"] == "files"
    (Path(task["benchmark_dir"]) / "runner.py").write_text("changed\n")
    try:
        worker._verify_source_binding(task)
    except ValueError as error:
        assert "hash mismatch" in str(error)
    else:
        raise AssertionError("tampered source was accepted")


def test_frozen_layout_resolves_submitted_runtime_before_active_layout(tmp_path):
    submitted = tmp_path / "submitted"
    runtime = submitted / "runtime" / "benchmarks"
    for relative in (
        "adapters/tau2_adapter.py", "adapters/toolsandbox_adapter.py",
        "adapters/acon_adapter.py", "adapters/acebench_adapter.py", "metrics.py",
    ):
        path = runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen\n")
    assert worker._resolve_benchmark_root(submitted / "official_one_task.py") == runtime


def test_toolsandbox_server_is_bound_to_the_official_wire_model(tmp_path):
    task = _task(tmp_path, benchmark="toolsandbox", task_id="wifi_off_all_tools")
    bad = _server(task)
    try:
        worker._validate_server(bad, task, bad["base_url"])
    except ValueError as error:
        assert "gpt-4o-2024-05-13" in str(error)
    else:
        raise AssertionError("wrong ToolSandbox model alias was accepted")
    good = _server(task, model="gpt-4o-2024-05-13")
    worker._validate_server(good, task, good["base_url"])


def test_server_cap_must_equal_frozen_task_cap(tmp_path):
    task = _task(tmp_path, benchmark="acebench", task_id="agent_multi_turn_10")
    server = _server(task)
    server["max_new_tokens"] = 2048
    try:
        worker._validate_server(server, task, server["base_url"])
    except ValueError as error:
        assert "differs from frozen task" in str(error)
    else:
        raise AssertionError("a server cap that differs from the task was accepted")


def test_main_writes_completed_result_contract(tmp_path, monkeypatch):
    task = _task(tmp_path)
    server = _server(task)
    task_path, server_path = tmp_path / "task.json", tmp_path / "ready.json"
    task_path.write_text(json.dumps(task))
    server_path.write_text(json.dumps(server))
    artifact = tmp_path / "official.json"
    artifact.write_text('{"score": 1}\n')
    monkeypatch.setattr(
        worker, "_dispatch",
        lambda task, base_url, out, manifest: ({"n": 1, "semantic_score": 1.0}, [artifact]),
    )
    out = tmp_path / "out"
    code = worker.main([
        "--task", str(task_path), "--base-url", server["base_url"],
        "--server-manifest", str(server_path), "--out", str(out),
        "--max-wall-seconds", "300",
    ])
    result = json.loads((out / "result.json").read_text())
    assert code == 0
    assert result["schema"] == worker.RESULT_SCHEMA
    assert result["status"] == "completed" and result["scored"] is True
    assert result["benchmark"] == "acon_appworld" and result["task_id"] == "3d9a636_1"
    assert result["official_score"] == 1.0 and len(result["official_artifacts"]) == 1
    assert result["source_binding"][0]["kind"] == "files"


def test_main_materializes_infra_failure_in_denominator(tmp_path, monkeypatch):
    task = _task(tmp_path)
    server = _server(task)
    task_path, server_path = tmp_path / "task.json", tmp_path / "ready.json"
    task_path.write_text(json.dumps(task))
    server_path.write_text(json.dumps(server))
    monkeypatch.setattr(worker, "_dispatch",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("harness failed")))
    out = tmp_path / "out"
    code = worker.main([
        "--task", str(task_path), "--base-url", server["base_url"],
        "--server-manifest", str(server_path), "--out", str(out),
        "--max-wall-seconds", "300",
    ])
    result = json.loads((out / "result.json").read_text())
    assert code == 1
    assert result["status"] == "infra_failed" and result["scored"] is False
    assert result["official_score"] is None and result["error"]["type"] == "RuntimeError"


def test_toolsandbox_dispatch_uses_one_exact_scenario_and_split_user_endpoint(tmp_path, monkeypatch):
    from adapters import toolsandbox_adapter

    task = _task(tmp_path, benchmark="toolsandbox", task_id="wifi_off_all_tools")
    called = {}
    def fake_run(*args, **kwargs):
        called["args"], called["kwargs"] = args, kwargs
        return {"n": 1, "semantic_score": 0.75}
    monkeypatch.setattr(toolsandbox_adapter, "run_ts", fake_run)
    summary, _ = worker._dispatch(
        task, "http://127.0.0.1:3/v1", tmp_path / "out",
        {"model_name": "gpt-4o-2024-05-13", "max_new_tokens": 256},
    )
    assert summary["semantic_score"] == 0.75
    assert called["kwargs"]["scenarios"] == ["wifi_off_all_tools"]
    assert called["kwargs"]["expected_task_ids"] == ["wifi_off_all_tools"]
    assert called["kwargs"]["user_base_url"] == task["user_base_url"]
    assert called["kwargs"]["agent"] == called["kwargs"]["user"] == toolsandbox_adapter.AGENT


def test_appworld_dispatch_pins_formal_split_and_frozen_sources(tmp_path, monkeypatch):
    from adapters import acon_adapter

    task = _task(tmp_path)
    (Path(task["benchmark_dir"]) / "src").mkdir()
    called = {}
    def fake_run(*args, **kwargs):
        called["kwargs"] = kwargs
        called["appworld_root"] = Path(worker.os.environ["APPWORLD_ROOT"])
        called["pythonpath"] = worker.os.environ["PYTHONPATH"]
        run = tmp_path / "run"
        return {"n": 1, "semantic_score": 0.0,
                "evaluation_path": str(tmp_path / "eval.json"), "run_dir": str(run)}
    monkeypatch.setattr(acon_adapter, "run_appworld", fake_run)
    old_root = worker.os.environ.get("APPWORLD_ROOT")
    worker._dispatch(
        task, "http://127.0.0.1:3/v1", tmp_path / "out",
        {"model_name": "c2kv-agent", "max_new_tokens": 2048},
    )
    assert called["kwargs"]["split"] == "test_normal"
    assert called["kwargs"]["task_ids"] == ["3d9a636_1"]
    assert called["kwargs"]["max_iter"] == 50
    assert called["appworld_root"] == Path(task["appworld_root"]).resolve()
    assert str(Path(task["benchmark_dir"]).resolve() / "src") in called["pythonpath"]
    assert worker.os.environ.get("APPWORLD_ROOT") == old_root


def test_acebench_dispatch_uses_agent_en_one_id_and_server_cap(tmp_path, monkeypatch):
    from adapters import acebench_adapter

    task = _task(tmp_path, benchmark="acebench", task_id="agent_multi_turn_10")
    called = {}
    def fake_run(*args, **kwargs):
        called["args"], called["kwargs"] = args, kwargs
        return {"n": 1, "semantic_score": 1.0}
    monkeypatch.setattr(acebench_adapter, "run_acebench", fake_run)
    worker._dispatch(
        task, "http://127.0.0.1:3/v1", tmp_path / "out",
        {"model_name": "c2kv-agent", "max_new_tokens": 1200},
    )
    assert called["kwargs"]["category"] == "agent"
    assert called["kwargs"]["language"] == "en"
    assert called["kwargs"]["task_ids"] == "agent_multi_turn_10"
    assert called["kwargs"]["max_tokens"] == 1200
    assert called["kwargs"]["num_threads"] == 1


def test_tau2_dispatch_injects_exact_task_and_zero_retries(tmp_path, monkeypatch):
    from adapters import tau2_adapter

    task = _task(tmp_path, benchmark="tau2", task_id="2")
    bench = Path(task["benchmark_dir"])
    (bench / "src").mkdir()
    bound_run_name = task["run_name"] + "_" + hashlib.sha256(str((tmp_path / "out").resolve()).encode("utf-8")).hexdigest()[:12]
    simulation = bench / "data" / "simulations" / bound_run_name
    seen = []
    def fake_subprocess(command, **kwargs):
        seen.append((command, kwargs))
        simulation.mkdir(parents=True, exist_ok=True)
        (simulation / "results.json").write_text(json.dumps({
            "simulations": [{"task_id": "2", "termination_reason": "stop"}]}))
        (simulation / "updated_results.json").write_text('{"simulations": []}')
        class Result:
            returncode = 0
        return Result()
    monkeypatch.setattr(worker.subprocess, "run", fake_subprocess)
    monkeypatch.setattr(tau2_adapter, "collect",
                        lambda *args, **kwargs: {"n": 1, "semantic_score": 1.0})
    summary, artifacts = worker._run_tau2(
        task, "http://127.0.0.1:3/v1", tmp_path / "out", "c2kv-agent")
    command, run_kwargs = seen[0]
    assert command[command.index("--task-ids") + 1:] == ["2", "--max-retries", "0"]
    assert command[command.index("--num-tasks") + 1] == "1"
    assert command[command.index("--num-trials") + 1] == "1"
    assert "--max-steps" not in command
    assert run_kwargs["env"]["PYTHONPATH"].split(worker.os.pathsep)[0] == str((bench / "src").resolve())
    assert summary["semantic_score"] == 1.0 and len(artifacts) == 2


def test_real_adapter_endpoint_normalization_is_idempotent():
    from adapters.base import v1
    from adapters.acon_adapter import runner_env, BASE_URL_ENV
    from adapters.tau2_adapter import run_command
    endpoint = "http://127.0.0.1:43000/v1"
    for value in (endpoint, endpoint + "/", "http://127.0.0.1:43000"):
        assert v1(value) == endpoint
        assert runner_env(value)[BASE_URL_ENV] == endpoint
    argv = run_command(endpoint, endpoint, "airline", "model", 1, "unique-run")
    for field in ("--agent-llm-args", "--user-llm-args"):
        args = json.loads(argv[argv.index(field) + 1])
        assert args["api_base"] == endpoint
        assert args["num_retries"] == 0
