import json
from io import BytesIO
from unittest.mock import patch

from generality import historykv_cell as driver
from generality.historykv_cell import (
    appworld_done_healthy, bfcl_row_healthy, cached_terminal,
    new_attempt_root, write_receipt,
)


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


def test_old_fc_handler_row_and_done_are_replaced_only_after_new_result(tmp_path):
    task_id = "multi_turn_base_1"
    out = tmp_path / "tasks" / task_id
    old_result = out / "bfcl" / "result" / "model" / "multi_turn" / "row.json"
    old_result.parent.mkdir(parents=True)
    old_result.write_text(json.dumps({
        "id": task_id, "result": [["<tool_call>...</tool_call>"]],
        "inference_log": [{"step_0": [{"role": "handler_log",
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
            "handler_name": "test-handler", "python_bench": "python"}
    with patch.object(driver.subprocess, "call", side_effect=fake_call):
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
        attempts = list((out / "attempts").glob("*"))
        assert len(attempts) == 1
        _appworld_result(attempts[0] / "appworld", task_id, "max_interactions")
        return 0

    cell = {"cell_dir": str(tmp_path), "acon_dir": "acon", "model_name": "model",
            "python_appworld": "python", "python_sgl": "python",
            "appworld_root": "appworld"}
    with patch.object(driver.subprocess, "call", side_effect=fake_call):
        receipt = driver.run_appworld_task(cell, task_id, 37401)

    assert receipt["status"] == "completed"
    assert driver.Path(receipt["attempt_root"]).parts[0] == "attempts"
    assert appworld_done_healthy(out, task_id)
    assert "vLLM generation Error:" in (out / "benchmark.log").read_text()
    history = json.loads((out / "done_history.jsonl").read_text().splitlines()[0])
    assert json.loads(history["previous_raw"]) == old_done


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
    with patch.object(driver.subprocess, "call", side_effect=AssertionError("reran")):
        assert driver.run_appworld_task(cell, task_id, 37401) == done


def test_appworld_new_generation_error_is_not_healthy(tmp_path):
    task_id = "task_1"
    out = tmp_path / "tasks" / task_id
    _appworld_result(out / "appworld", task_id, "generation_error")
    (out / "done.json").write_text(json.dumps({
        "task_id": task_id, "status": "completed", "returncode": 0,
    }), encoding="utf-8")
    assert not appworld_done_healthy(out, task_id)


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


def test_production_holds_cannot_be_overridden():
    from generality.scheduler_npu import cells_ready
    for backend in ("c2kv", "h2o", "snapkv", "pyramidkv"):
        for benchmark in ("bfcl_base", "bfcl_long_context"):
            assert not cells_ready({"backend": backend, "benchmark_key": benchmark,
                                    "condition": "compression_full_budget"})
    assert not cells_ready({"backend": "pyramidkv", "benchmark_key": "appworld",
                            "condition": "compression_full_budget"})
