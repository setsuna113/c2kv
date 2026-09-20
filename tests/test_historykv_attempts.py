import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from io import BytesIO
from unittest.mock import patch

from generality import historykv_cell as driver
from generality import scheduler_npu as scheduler
from generality.completion_contract import invalidate_appworld_done
from generality.historykv_cell import (
    appworld_done_healthy, bfcl_row_healthy, cached_terminal,
    new_attempt_root, write_receipt,
)
from generality.upstream_liveness import UpstreamUnavailable


def test_retry_preserves_existing_outputs_and_receipt(tmp_path):
    old = tmp_path / "appworld"
    old.mkdir()
    (old / "evidence.txt").write_text("original")
    write_receipt(tmp_path, {"status": "failed", "error": "connection"})
    assert cached_terminal(tmp_path) is None
    first = new_attempt_root(tmp_path, "appworld")
    second = new_attempt_root(tmp_path, "appworld")
    assert first != second
    assert (old / "evidence.txt").read_text() == "original"
    write_receipt(tmp_path, {"status": "infra_error"})
    saved = json.loads((tmp_path / "status_history.jsonl").read_text())
    assert json.loads(saved["previous_raw"])["error"] == "connection"


def test_typed_method_failure_is_terminal_but_transport_is_not(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    path = result / "result.json"
    for failure, expected in [
        ('HTTP 422 {"code":"c2kv_capacity_infeasible"}', True),
        ("HTTP 502 connection refused", False),
        (None, True),
    ]:
        path.write_text(json.dumps({"id": "task_1", "result": [], "traceback": failure}))
        assert bfcl_row_healthy(tmp_path, "task_1") is expected


def test_bfcl_validation_from_direct_file_entry_without_package_path(tmp_path):
    result = tmp_path / "result"
    result.mkdir()
    (result / "row.json").write_text(json.dumps({"id": "task_1", "result": [[[]]]}))
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    code = (
        "import runpy,sys; from pathlib import Path; "
        "sys.path.insert(0,str(Path(sys.argv[1]).parent)); "
        "module=runpy.run_path(sys.argv[1]); "
        "assert module['bfcl_row_healthy'](Path(sys.argv[2]),'task_1')"
    )
    check = subprocess.run([sys.executable, "-I", "-c", code,
                            str(Path(driver.__file__).resolve()), str(tmp_path)],
                           cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)
    assert check.returncode == 0, check.stdout + check.stderr


def test_direct_historykv_driver_binds_paper_checkout_from_environment(tmp_path):
    env = os.environ.copy()
    env["C2KV_PAPER_SOURCE"] = str(tmp_path / "full_paper")
    code = ("import runpy,sys; from pathlib import Path; "
            "sys.path.insert(0,str(Path(sys.argv[1]).parent)); "
            "module=runpy.run_path(sys.argv[1]); "
            "print(module['PAPER'])")
    result = subprocess.run([sys.executable, "-I", "-c", code,
                             str(Path(driver.__file__).resolve())],
                            cwd=tmp_path, env=env, capture_output=True, text=True,
                            timeout=20, check=True)
    assert result.stdout.strip() == str(tmp_path / "full_paper")


def test_old_fc_handler_row_and_done_are_replaced_only_after_new_result(tmp_path):
    task_id = "multi_turn_base_1"
    out = tmp_path / "tasks" / task_id
    old_result = out / "bfcl" / "result" / "model" / "multi_turn" / "row.json"
    old_result.parent.mkdir(parents=True)
    old_result.write_text(json.dumps({
        "id": task_id, "result": [["<tool_call>...</tool_call>"]],
        "inference_log": [{"step_0": [{"role": "assistant", "content":
                                      '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'},
                                     {"role": "handler_log",
                                      "error": "'str' object has no attribute 'items'"}]}],
    }), encoding="utf-8")
    old_done = {"task_id": task_id, "status": "completed", "returncode": 0}
    (out / "done.json").write_text(json.dumps(old_done), encoding="utf-8")
    assert not driver.existing_bfcl_result(out, task_id)

    def fake_call(*args, **kwargs):
        attempts = list((out / "attempts").glob("*"))
        assert len(attempts) == 1
        result = attempts[0] / "bfcl" / "result" / "model" / "multi_turn" / "row.json"
        result.parent.mkdir(parents=True)
        result.write_text(json.dumps({"id": task_id, "result": [[[]]]}),
                          encoding="utf-8")
        return 0

    cell = {"cell_dir": str(tmp_path), "model_name": "test-model",
            "handler_name": "test-handler", "python_bench": "python",
            "sglang_backend_url": "http://127.0.0.1:1"}
    with patch.object(driver, "run_owned_worker", side_effect=fake_call):
        receipt = driver.run_bfcl_task(cell, task_id, 37401)

    assert receipt["status"] == "completed"
    assert json.loads(old_result.read_text())["result"] == [["<tool_call>...</tool_call>"]]
    history = json.loads((out / "done_history.jsonl").read_text().splitlines()[0])
    assert json.loads(history["previous_raw"]) == old_done
    assert json.loads((out / "done.json").read_text())["status"] == "completed"


def _appworld_result(root, task_id, reason):
    path = root / "appworld_harness" / "experiments" / "appworld" / "outputs" / "model" / "test_normal" / f"task_{task_id}" / "results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"task_id": task_id,
                                "termination_reason": reason}), encoding="utf-8")
    return path


def test_old_appworld_generation_error_done_is_replaced_only_after_new_result(tmp_path):
    task_id = "6b6ca61_3"
    out = tmp_path / "tasks" / task_id
    _appworld_result(out / "appworld", task_id, "max_interactions")
    (out / "benchmark.log").write_text(
        "ERROR:llm:vLLM generation Error: Error code: 502 - upstream failed",
        encoding="utf-8",
    )
    old_done = {"task_id": task_id, "status": "completed", "returncode": 0}
    (out / "done.json").write_text(json.dumps(old_done), encoding="utf-8")
    assert not appworld_done_healthy(out, task_id)

    def fake_call(*args, **kwargs):
        script = args[0][-1]
        assert "from process_lifecycle import interruptible" in script
        assert "@interruptible" in script
        assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(
            Path(driver.__file__).resolve().parent)
        attempts = list((out / "attempts").glob("*"))
        assert len(attempts) == 1
        _appworld_result(attempts[0] / "appworld", task_id, "max_interactions")
        return 0

    cell = {"cell_dir": str(tmp_path), "acon_dir": "acon", "model_name": "model",
            "python_appworld": "python", "python_sgl": "python",
            "appworld_root": "appworld", "sglang_backend_url": "http://127.0.0.1:1"}
    with patch.object(driver, "run_owned_worker", side_effect=fake_call):
        receipt = driver.run_appworld_task(cell, task_id, 37401)

    assert receipt["status"] == "completed"
    assert driver.Path(receipt["attempt_root"]).parts[0] == "attempts"
    assert appworld_done_healthy(out, task_id)
    assert "vLLM generation Error:" in (out / "benchmark.log").read_text()
    history = json.loads((out / "done_history.jsonl").read_text().splitlines()[0])
    assert json.loads(history["previous_raw"]) == old_done


def test_toolsandbox_nested_worker_installs_signal_unwind(tmp_path):
    cell = {"cell_dir": str(tmp_path), "benchmark_dir": "toolsandbox",
            "python_sgl": "python", "python_bench": "python",
            "model_name": "model", "sglang_backend_url": "http://127.0.0.1:1"}

    def fake_call(command, **kwargs):
        script = command[-1]
        assert "from process_lifecycle import interruptible" in script
        assert "@interruptible" in script
        assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(
            Path(driver.__file__).resolve().parent)
        compile(script, "<toolsandbox-worker>", "exec")
        return 0

    with patch.object(driver, "run_owned_worker", side_effect=fake_call):
        result = driver._run_adapter_task(cell, "scenario_1", 37401, "toolsandbox")
    assert result["status"] == "completed"


def test_appworld_empty_model_output_is_not_generation_error(tmp_path):
    task_id = "task_1"
    out = tmp_path / "tasks" / task_id
    record = _appworld_result(out / "appworld", task_id, "error")
    data = json.loads(record.read_text())
    data["output"] = ""
    record.write_text(json.dumps(data), encoding="utf-8")
    done = {"task_id": task_id, "status": "completed", "returncode": 0}
    (out / "done.json").write_text(json.dumps(done), encoding="utf-8")

    assert appworld_done_healthy(out, task_id)
    cell = {"cell_dir": str(tmp_path)}
    with patch.object(driver, "run_owned_worker", side_effect=AssertionError("reran")):
        assert driver.run_appworld_task(cell, task_id, 37401) == done


def test_appworld_new_generation_error_is_not_healthy(tmp_path):
    task_id = "task_1"
    out = tmp_path / "tasks" / task_id
    _appworld_result(out / "appworld", task_id, "generation_error")
    (out / "done.json").write_text(json.dumps({
        "task_id": task_id, "status": "completed", "returncode": 0,
    }), encoding="utf-8")
    assert not appworld_done_healthy(out, task_id)


def test_appworld_invalidation_matches_only_old_done_bytes(tmp_path):
    task_id = "6b6ca61_3"
    out = tmp_path / "tasks" / task_id
    _appworld_result(out / "appworld", task_id, "max_interactions")
    done = out / "done.json"
    done.write_text(json.dumps({"task_id": task_id, "status": "completed",
                                "returncode": 0}), encoding="utf-8")
    assert appworld_done_healthy(out, task_id)
    (out / "invalidated_done.json").write_text(json.dumps({
        "sha256": hashlib.sha256(done.read_bytes()).hexdigest(),
        "reason": "old proxy served a different arm",
        "evidence": "proxy request receipt",
    }), encoding="utf-8")
    assert not appworld_done_healthy(out, task_id)
    new_root = out / "attempts" / "new" / "appworld"
    _appworld_result(new_root, task_id, "max_interactions")
    done.write_text(json.dumps({"task_id": task_id, "status": "completed",
                                "returncode": 0,
                                "attempt_root": "attempts/new/appworld"}), encoding="utf-8")
    assert appworld_done_healthy(out, task_id)
    (tmp_path / "cell.json").write_text(json.dumps({"task_ids": [task_id]}))
    second = invalidate_appworld_done(tmp_path, task_id,
                                       "new attempt invalidated", "review receipt")
    preserved = json.loads((out / "invalidated_done_history.jsonl").read_text())
    assert json.loads(preserved["previous_raw"])["sha256"] != second["sha256"]
    assert not appworld_done_healthy(out, task_id)


def test_full_appworld_cell_writes_manifest_bound_completion(tmp_path):
    task_id = "6b6ca61_3"
    cell_dir = tmp_path / "cell"
    out = cell_dir / "tasks" / task_id
    _appworld_result(out / "appworld", task_id, "max_interactions")
    (out / "done.json").write_text(json.dumps({
        "task_id": task_id, "status": "completed", "returncode": 0}))
    cell = {"cell_id": "history-full", "cell_dir": str(cell_dir),
            "backend": "h2o", "working_point": "K0",
            "condition": "recovery_off_same_initial", "benchmark": "appworld",
            "task_ids": [task_id], "budget_tokens": {"K": 8},
            "sglang_backend_url": "http://127.0.0.1:1"}
    frozen = cell_dir / "cell.json"
    frozen.write_text(json.dumps(cell))
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps(cell))

    with (patch.object(driver, "resolve_free_port", return_value=37401),
          patch.object(driver, "start_proxy", return_value=object()),
          patch.object(driver, "stop")):
        assert driver.main(["--cell", str(launch), "--max-tasks", "0"]) == 0

    assert scheduler.cell_done(cell)
    old_status = json.loads((cell_dir / "cell_status.json").read_text())
    record = invalidate_appworld_done(cell_dir, task_id,
                                      "old proxy served a different arm",
                                      "proxy request receipt")
    assert record["sha256"] == hashlib.sha256((out / "done.json").read_bytes()).hexdigest()
    assert json.loads((out / "invalidated_done.json").read_text())["sha256"] == record["sha256"]
    assert not scheduler.cell_done(cell)
    assert not appworld_done_healthy(out, task_id)
    (out / "invalidated_done.json").unlink()
    assert not appworld_done_healthy(out, task_id)
    assert invalidate_appworld_done(cell_dir, task_id,
                                    "old proxy served a different arm",
                                    "proxy request receipt") == record
    assert len((cell_dir / "cell_invalidations.jsonl").read_text().splitlines()) == 1

    with (patch.object(driver, "resolve_free_port", return_value=37401),
          patch.object(driver, "start_proxy", return_value=object()),
          patch.object(driver, "stop")):
        assert driver.main(["--cell", str(launch), "--max-tasks", "0"]) == 0
    assert json.loads((cell_dir / "cell_status.json").read_text())["status"] == "incomplete"
    assert not scheduler.cell_done(cell)

    new_root = out / "attempts" / "new" / "appworld"
    _appworld_result(new_root, task_id, "max_interactions")
    (out / "done.json").write_text(json.dumps({
        "task_id": task_id, "status": "completed", "returncode": 0,
        "attempt_root": "attempts/new/appworld"}))
    with (patch.object(driver, "resolve_free_port", return_value=37401),
          patch.object(driver, "start_proxy", return_value=object()),
          patch.object(driver, "stop")):
        assert driver.main(["--cell", str(launch), "--max-tasks", "0"]) == 0
    new_status = json.loads((cell_dir / "cell_status.json").read_text())
    assert new_status["cell_invalidation_sha256"] != old_status["cell_invalidation_sha256"]
    assert scheduler.cell_done(cell)
    history = [json.loads(line) for line in
               (cell_dir / "cell_status_history.jsonl").read_text().splitlines()]
    assert json.loads(history[0]["previous_raw"]) == old_status


def test_appworld_stale_done_cannot_complete_cell(tmp_path):
    task_id = "6b6ca61_3"
    cell_dir = tmp_path / "cell"
    out = cell_dir / "tasks" / task_id
    _appworld_result(out / "appworld", task_id, "max_interactions")
    (out / "benchmark.log").write_text("vLLM generation Error: HTTP 502", encoding="utf-8")
    (out / "done.json").write_text(json.dumps({
        "task_id": task_id, "status": "completed", "returncode": 0,
    }), encoding="utf-8")
    cell = {"cell_id": "stale-appworld", "cell_dir": str(cell_dir),
            "backend": "h2o", "working_point": "K0",
            "condition": "recovery_off_same_initial", "benchmark": "appworld",
            "task_ids": [task_id], "budget_tokens": {"K": 8},
            "sglang_backend_url": "http://127.0.0.1:1"}
    manifest = tmp_path / "cell.json"
    manifest.write_text(json.dumps(cell), encoding="utf-8")

    with (patch.object(driver, "resolve_free_port", return_value=37401),
          patch.object(driver, "start_proxy", return_value=object()),
          patch.object(driver, "stop"),
          patch.object(driver, "run_appworld_task", return_value={
              "task_id": task_id, "status": "failed_validation"})):
        driver.main(["--cell", str(manifest), "--task-ids", task_id])

    status = json.loads((cell_dir / "cell_status.json").read_text())
    assert status["status"] == "incomplete"
    assert status["n_completed"] == 0
    assert status["pending_infra"] == [task_id]


def test_subset_settlement_uses_frozen_manifest(tmp_path):
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    task_id = "multi_turn_base_1"
    result = cell_dir / "tasks" / task_id / "bfcl" / "result" / "model" / "row.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"id": task_id, "result": [[[]]]}), encoding="utf-8")
    cell = {"cell_id": "test", "cell_dir": str(cell_dir),
            "backend": "h2o", "working_point": "K0",
            "condition": "recovery_off_same_initial", "benchmark": "bfcl",
            "task_ids": [task_id, "multi_turn_base_2"],
            "budget_tokens": {"K": 8}, "sglang_backend_url": "http://127.0.0.1:1"}
    manifest = tmp_path / "cell.json"
    manifest.write_text(json.dumps(cell), encoding="utf-8")

    with (patch.object(driver, "resolve_free_port", return_value=37401),
          patch.object(driver, "start_proxy", return_value=object()),
          patch.object(driver, "stop"),
          patch.object(driver, "run_bfcl_task", return_value={
              "task_id": task_id, "status": "completed"})):
        driver.main(["--cell", str(manifest), "--task-ids", task_id])

    status = json.loads((cell_dir / "cell_status.json").read_text())
    assert status["status"] == "incomplete"
    assert status["n_total"] == 2
    assert status["n_completed"] == 1
    assert status["pending_infra"] == ["multi_turn_base_2"]


def test_engine_loss_stops_cell_and_preserves_prior_bfcl_row(tmp_path):
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    task_ids = ["multi_turn_base_1", "multi_turn_base_2", "multi_turn_base_3"]
    first_row = (cell_dir / "tasks" / task_ids[0] / "bfcl" / "result"
                 / "model" / "multi_turn" / "row.json")
    first_row.parent.mkdir(parents=True)
    first_row.write_text(json.dumps({"id": task_ids[0], "result": [[[]]]}),
                         encoding="utf-8")
    cell = {"cell_id": "engine-loss", "cell_dir": str(cell_dir),
            "backend": "h2o", "working_point": "K0",
            "condition": "recovery_off_same_initial", "benchmark": "bfcl",
            "task_ids": task_ids, "budget_tokens": {"K": 8},
            "sglang_backend_url": "http://127.0.0.1:1"}
    manifest = cell_dir / "cell.json"
    manifest.write_text(json.dumps(cell), encoding="utf-8")
    called = []

    def run_task(_cell, task_id, _port):
        called.append(task_id)
        if task_id == task_ids[1]:
            raise UpstreamUnavailable("engine exited")
        return {"task_id": task_id, "status": "completed"}

    with (patch.object(driver, "resolve_free_port", return_value=37401),
          patch.object(driver, "start_proxy", return_value=object()),
          patch.object(driver, "stop"),
          patch.object(driver, "run_bfcl_task", side_effect=run_task)):
        assert driver.main(["--cell", str(manifest)]) == 1

    assert called == task_ids[:2]
    assert json.loads(first_row.read_text())["id"] == task_ids[0]
    interrupted = json.loads((cell_dir / "tasks" / task_ids[1] / "status.json").read_text())
    assert interrupted["kind"] == "upstream_unavailable"
    assert not (cell_dir / "tasks" / task_ids[1] / "done.json").exists()
    status = json.loads((cell_dir / "cell_status.json").read_text())
    assert status["status"] == "incomplete"
    assert status["stop_reason"] == "upstream_unavailable"
    assert status["n_completed"] == 1
    assert status["pending_infra"] == task_ids[1:]


def test_proxy_retries_keep_separate_logs(tmp_path):
    calls = []

    class Process:
        def poll(self):
            return None

    class Reply(BytesIO):
        status = 200

    class Opener:
        def open(self, request, timeout=None):
            if isinstance(request, str):
                return Reply(b"{}")
            if request.full_url.endswith("/v1/chat/completions"):
                payload = json.loads(request.data)
                log = calls[-1]["request_log"]
                log.write_text(json.dumps({
                    "request_id": payload["c2kv_measurement_session_id"],
                    "arm": "gen_h2o_k0"}) + "\n", encoding="utf-8")
                return Reply(b'{"choices":[{"message":{"content":"OK"}}]}')
            if request.full_url.endswith("/close_measurement_session"):
                assert "--shared-engine" in calls[-1]["command"]
                assert json.loads(request.data)["c2kv_measurement_session_id"].startswith("proxy-probe-")
                return Reply(b'{"closed_owned_sessions":true}')
            return Reply(b"{}")

    def popen(command, **kwargs):
        calls.append({"command": command,
                      "request_log": command[command.index("--request-log") + 1],
                      "telemetry_log": command[command.index("--telemetry-log") + 1]})
        calls[-1]["request_log"] = driver.Path(calls[-1]["request_log"])
        return Process()

    with (patch.object(driver, "_proxy_opener", return_value=Opener()),
          patch.object(driver.subprocess, "Popen", side_effect=popen)):
        driver.start_proxy("gen_h2o_k0", "http://127.0.0.1:1", 37401, tmp_path, 8)
        first = calls[0]["request_log"]
        first_text = first.read_text()
        driver.start_proxy("gen_h2o_k0", "http://127.0.0.1:1", 37402, tmp_path, 8)

    assert calls[0]["request_log"] != calls[1]["request_log"]
    assert calls[0]["telemetry_log"] != calls[1]["telemetry_log"]
    assert first.read_text() == first_text
    assert (first.parent / "proxy.log").exists()
    assert (calls[1]["request_log"].parent / "proxy.log").exists()


def test_production_holds_cannot_be_overridden(monkeypatch):
    from generality import scheduler_npu
    monkeypatch.setattr(scheduler_npu, "BLOCKED_BACKENDS",
                        {"c2kv", "h2o", "snapkv", "pyramidkv"})
    from generality.scheduler_npu import cells_ready
    for backend in ("c2kv", "h2o", "snapkv", "pyramidkv"):
        for benchmark in ("bfcl_base", "bfcl_long_context"):
            assert not cells_ready({"backend": backend, "benchmark_key": benchmark,
                                    "condition": "compression_full_budget"})
    assert not cells_ready({"backend": "pyramidkv", "benchmark_key": "appworld",
                            "condition": "compression_full_budget"})
