from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from generality.bfcl_results import collect_bfcl_results, completion_receipt
from generality import scheduler_npu as scheduler


def _result_file(cell_dir: Path, attempt: str) -> Path:
    path = (
        cell_dir / "batches" / attempt / "bfcl_worker" / "bfcl" / "result"
        / "model" / "multi_turn" / "BFCL_v4_multi_turn_base_result.json"
    )
    path.parent.mkdir(parents=True)
    return path


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _load_driver():
    sys.modules.setdefault("current", types.SimpleNamespace())
    sys.modules.setdefault("evidence_sets", types.SimpleNamespace())
    sys.modules.setdefault(
        "c1_artifact_binding",
        types.SimpleNamespace(bind_risk_artifact=lambda artifact, checkpoint: (artifact, {})),
    )
    from generality import c2kv_cell
    return c2kv_cell


def test_collector_prefers_latest_valid_and_emits_complete_refill_order(tmp_path):
    first = _result_file(tmp_path, "first")
    _write_rows(first, [
        {"id": "task_a", "result": ["old"]},
        {"id": "task_b", "result": "", "traceback": "network"},
        {"id": "foreign", "result": ["ignored"]},
        None,
        ["not", "a", "row"],
    ])
    os.utime(first, ns=(1_000, 1_000))

    second = _result_file(tmp_path, "second")
    _write_rows(second, [
        {"id": "task_a", "result": "", "traceback": "later failure"},
        {"id": "task_a", "result": ["new"]},
        {"id": "task_b", "traceback": None},
    ])
    os.utime(second, ns=(2_000, 2_000))

    result = collect_bfcl_results(
        tmp_path, ["task_a", "task_b", "task_c", "task_c"]
    )

    assert result["expected_task_ids"] == ["task_a", "task_b", "task_c"]
    assert result["canonical"]["task_a"]["row"]["result"] == ["new"]
    assert result["valid_task_ids"] == ["task_a"]
    assert result["invalid_task_ids"] == ["task_b"]
    assert result["missing_task_ids"] == ["task_c"]
    assert result["refill_task_ids"] == ["task_b", "task_c"]
    assert result["unexpected_task_ids"] == ["foreign"]
    assert result["expected_rows"] == 5
    assert result["duplicate_rows"] == 3
    assert result["malformed_rows"] == 2


def test_scored_failures_are_terminal_but_backend_failures_refill(tmp_path):
    rows = [
        {"id": "overflow", "result": "", "traceback":
         "The input (138237 tokens) is longer than the model's context length (131072 tokens)."},
        {"id": "missing_subgoal", "result": "", "traceback":
         "HiAgent requested nonexistent completed subgoals"},
        {"id": "repeated_retrieval", "result": "", "traceback":
         "HiAgent requested an already revealed trajectory without advancing"},
        {"id": "bad_gateway", "result": "", "traceback": "HTTP 502 Bad Gateway"},
        {"id": "timeout", "result": "", "traceback": "request timed out"},
        {"id": "engine_crash", "result": "", "traceback": "engine crashed"},
        {"id": "no_result", "traceback":
         "The input (138237 tokens) is longer than the model's context length (131072 tokens)."},
    ]
    _write_rows(_result_file(tmp_path, "attempt"), rows)
    _write_rows(_result_file(tmp_path, "later"), [
        {"id": "overflow", "result": "", "traceback": "HTTP 502 Bad Gateway"},
    ])
    expected = [row["id"] for row in rows] + ["never_started"]

    result = collect_bfcl_results(tmp_path, expected)

    assert result["valid_task_ids"] == expected[:3]
    assert result["refill_task_ids"] == expected[3:]
    assert result["terminal_failures"] == {
        "overflow": "context_overflow",
        "missing_subgoal": "hiagent_invalid_retrieval",
        "repeated_retrieval": "hiagent_invalid_retrieval",
    }
    assert completion_receipt(result)["terminal_failures"] == result["terminal_failures"]
    assert result["canonical"]["overflow"]["row"] == rows[0]


def test_fc_collector_refills_only_the_old_handler_decode_failure(tmp_path):
    legacy = {
        "id": "task_old", "result": [["<tool_call>...</tool_call>"]],
        "inference_log": [{"step_0": [{"role": "assistant", "content":
                                      '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'},
                                     {"role": "handler_log",
                                      "error": "'str' object has no attribute 'items'"}]}],
    }
    actor_failure = {
        "id": "task_actor", "result": [["<tool_call>bad draft</tool_call>"]],
        "inference_log": [{"step_0": [{"role": "handler_log",
                                      "error": "invalid model tool syntax"}]}],
    }
    _write_rows(_result_file(tmp_path, "old"), [legacy, actor_failure])

    fc = collect_bfcl_results(tmp_path, ["task_old", "task_actor"], fc_model=True)
    prompting = collect_bfcl_results(tmp_path, ["task_old", "task_actor"])

    assert fc["refill_task_ids"] == ["task_old"]
    assert fc["legacy_fc_decode_task_ids"] == ["task_old"]
    assert fc["valid_task_ids"] == ["task_actor"]
    assert prompting["valid_task_ids"] == ["task_old", "task_actor"]


def test_fc_collector_preserves_plain_final_text_from_old_native_handler(tmp_path):
    row = {"id": "task_final", "result": [["Here is the answer."]],
           "inference_log": [{"step_0": [
               {"role": "assistant", "content": "Here is the answer."},
               {"role": "handler_log", "error": "'str' object has no attribute 'items'"}]}]}
    _write_rows(_result_file(tmp_path, "old"), [row])
    result = collect_bfcl_results(tmp_path, ["task_final"], fc_model=True)
    assert result["valid_task_ids"] == ["task_final"]
    assert result["refill_task_ids"] == []
    assert result["canonical"]["task_final"]["row"] == row


def test_driver_and_bundled_bfcl_completion_contracts_match():
    root = Path(__file__).resolve().parents[1]
    assert (root / "generality/bfcl_completion.py").read_text() == (
        root / "controller_runtime/benchmarks/bfcl_completion.py").read_text()


def test_validate_chunk_deduplicates_and_rejects_foreign_or_malformed_rows(tmp_path):
    driver = _load_driver()
    result = (
        tmp_path / "bfcl_worker" / "bfcl" / "result" / "model"
        / "multi_turn" / "rows.json"
    )
    result.parent.mkdir(parents=True)
    result.write_text(
        "\n".join([
            json.dumps({"id": "task_a", "result": ["ok"]}),
            json.dumps({"id": "task_a", "result": "", "traceback": "later"}),
            json.dumps({"id": "task_b", "result": "", "traceback": "failed"}),
            json.dumps({"id": "task_c", "traceback": None}),
            json.dumps({"id": "foreign", "result": ["ignore"]}),
            "null",
            "[]",
            "{broken json",
        ]) + "\n",
        encoding="utf-8",
    )

    healthy, bad = driver.validate_chunk(
        tmp_path, ["task_a", "task_a", "task_b", "task_c"]
    )
    assert healthy == ["task_a"]
    assert bad == ["task_b", "task_c"]


def test_nonzero_worker_exit_preserves_rows_and_retries_only_missing(tmp_path):
    driver = _load_driver()
    cell = {
        "benchmark": "bfcl",
        "cell_dir": str(tmp_path),
        "caps": {"task_timeout": 1},
    }

    class FakeProcess:
        def __init__(self, rc, pid):
            self.returncode = None
            self._rc = rc
            self.pid = pid

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = self._rc
            return self._rc

        def terminate(self):
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    calls = 0

    def popen(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs["env"]["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
        out = tmp_path / "batches" / "attempt"
        if calls == 1:
            (out / "server").mkdir(parents=True, exist_ok=True)
            (out / "server" / "ready.json").write_text("{}", encoding="utf-8")
            return FakeProcess(0, 9001)
        result = _result_file(tmp_path, "attempt")
        _write_rows(result, [{"id": "task_a", "result": ["ok"]}])
        return FakeProcess(1, 9002)

    with (
        patch.object(driver, "server_command", return_value=["server"]),
        patch.object(driver, "bfcl_worker_command", return_value=["worker"]),
        patch.object(driver.subprocess, "Popen", side_effect=popen),
        patch.object(driver.os, "getpgid", side_effect=OSError, create=True),
        patch.object(driver.os, "killpg", create=True),
    ):
        status = driver.run_task(cell, ["task_a", "task_b"], 37201, "attempt")

    assert status["status"] == "partial"
    assert status["healthy"] == ["task_a"]
    assert status["bad"] == ["task_b"]


def test_valid_bfcl_row_preserved_when_server_cost_finalization_fails(tmp_path):
    driver = _load_driver()
    cell = {"benchmark": "bfcl", "cell_dir": str(tmp_path),
            "caps": {"task_timeout": 1}}
    out = tmp_path / "batches" / "attempt"

    class FakeProcess:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

    server = FakeProcess(9001)
    worker = FakeProcess(9002)
    calls = 0

    def popen(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            (out / "server").mkdir(parents=True)
            (out / "server" / "ready.json").write_text("{}", encoding="utf-8")
            return server
        _write_rows(_result_file(tmp_path, "attempt"),
                    [{"id": "task_a", "result": ["valid official row"]}])
        return worker

    def stop(proc):
        if proc is server:
            (out / "server" / "final.json").write_text(json.dumps({
                "status": "failed", "cost_summary_error": {
                    "type": "ValueError", "message": "invalid cost inventory"},
            }), encoding="utf-8")

    with (
        patch.object(driver, "server_command", return_value=["server"]),
        patch.object(driver, "bfcl_worker_command", return_value=["worker"]),
        patch.object(driver.subprocess, "Popen", side_effect=popen),
        patch.object(driver, "stop_owned_group", side_effect=stop),
    ):
        status = driver.run_task(cell, ["task_a"], 37201, "attempt")

    assert status["status"] == "completed"
    assert status["healthy"] == ["task_a"]
    assert status["cost_finalization"] == {
        "status": "failed", "reason": "cost_summary_error",
        "error": {"type": "ValueError", "message": "invalid cost inventory"},
    }
    assert (out / "status.json").exists()
    assert not (out / "done.json").exists()
    assert driver.validate_chunk(out, ["task_a"]) == (["task_a"], [])


def test_appworld_worker_uses_pinned_paper_source(tmp_path, monkeypatch):
    driver = _load_driver()
    paper = tmp_path / "paper-release"
    monkeypatch.setenv("C2KV_PAPER_SOURCE", str(paper))
    cell = {"benchmark": "acon_appworld", "cell_dir": str(tmp_path),
            "caps": {"task_timeout": 1}}
    out = tmp_path / "batches" / "attempt"
    worker_env = None

    class FakeProcess:
        def __init__(self, rc):
            self.rc = rc

        def poll(self):
            return None

        def wait(self, timeout=None):
            return self.rc

    calls = 0

    def popen(*args, **kwargs):
        nonlocal calls, worker_env
        calls += 1
        if calls == 1:
            (out / "server").mkdir(parents=True)
            (out / "server" / "ready.json").write_text("{}", encoding="utf-8")
            return FakeProcess(0)
        worker_env = kwargs["env"]
        return FakeProcess(1)

    with (
        patch.object(driver, "server_command", return_value=["server"]),
        patch.object(driver, "appworld_worker_command", return_value=["worker"]),
        patch.object(driver.subprocess, "Popen", side_effect=popen),
        patch.object(driver, "stop_owned_group"),
    ):
        status = driver.run_task(cell, ["task_a"], 37201, "attempt")

    assert worker_env is not None
    assert worker_env["PYTHONPATH"].split(os.pathsep)[0] == str(paper / "benchmarks")
    assert status["cost_finalization"] == {
        "status": "unavailable", "reason": "missing_final_receipt"}


def test_audit_only_writes_full_refill_manifest_without_preparing_runtime(tmp_path):
    driver = _load_driver()
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    cell = {
        "cell_id": "bfcl-test",
        "cell_dir": str(cell_dir),
        "benchmark": "bfcl",
        "task_ids": ["task_a", "task_b"],
    }
    cell_path = tmp_path / "cell.json"
    cell_path.write_text(json.dumps(cell), encoding="utf-8")
    result = _result_file(cell_dir, "existing")
    _write_rows(result, [{"id": "task_a", "result": []}])

    with (
        patch.object(driver, "prepare_cell_files") as prepare,
        patch.object(driver, "run_task") as run_task,
    ):
        rc = driver.main([
            "--cell", str(cell_path),
            "--budgets", str(tmp_path / "unused.json"),
            "--audit-results-only",
        ])

    assert rc == 0
    prepare.assert_not_called()
    run_task.assert_not_called()
    refill = json.loads((cell_dir / "bfcl_refill.json").read_text())
    assert refill["task_ids"] == ["task_b"]
    assert refill["n_tasks"] == 1


def test_audit_rejects_duplicate_frozen_ids_before_writing_receipts(tmp_path):
    driver = _load_driver()
    cell = {"cell_id": "duplicate", "cell_dir": str(tmp_path),
            "benchmark": "bfcl", "task_ids": ["task_a", "task_a"]}
    cell_path = tmp_path / "cell.json"
    cell_path.write_text(json.dumps(cell), encoding="utf-8")

    import pytest
    with pytest.raises(ValueError, match="unique nonempty task IDs"):
        driver.main(["--cell", str(cell_path), "--budgets", str(tmp_path / "none"),
                     "--audit-results-only"])
    assert not (tmp_path / "bfcl_completion.json").exists()


def test_existing_attempt_directory_is_never_overwritten(tmp_path):
    driver = _load_driver()
    attempt = tmp_path / "batches" / "fixed"
    attempt.mkdir(parents=True)
    sentinel = attempt / "evidence.json"
    sentinel.write_text("keep", encoding="utf-8")

    try:
        driver.run_task(
            {"cell_dir": str(tmp_path), "benchmark": "bfcl", "caps": {}},
            ["task_a"], 37201, "fixed",
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("run_task must reject an existing attempt directory")

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_main_subset_runs_only_valid_gap_and_settles_against_full_manifest(tmp_path):
    driver = _load_driver()
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    cell = {
        "cell_id": "bfcl-main-resume",
        "cell_dir": str(cell_dir),
        "benchmark": "bfcl",
        "task_ids": ["task_a", "task_b", "task_c"],
    }
    cell_path = tmp_path / "cell.json"
    cell_path.write_text(json.dumps(cell), encoding="utf-8")
    budgets = tmp_path / "budgets.json"
    budgets.write_text("{}", encoding="utf-8")
    _write_rows(_result_file(cell_dir, "old"), [
        {"id": "task_a", "result": ["already valid"]},
        {"id": "task_b", "result": "", "traceback": "old failure"},
    ])
    # Stale marker arithmetic used to make this cell look complete even though
    # task_b and task_c had no valid official result rows.
    for task_id, marker in (("task_b", "terminal.json"), ("task_c", "done.json")):
        path = cell_dir / "tasks" / task_id / marker
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    calls: list[list[str]] = []

    def run_task(cell_arg, task_ids, port, batch_name):
        calls.append(list(task_ids))
        _write_rows(
            _result_file(Path(cell_arg["cell_dir"]), batch_name),
            [{"id": task_id, "result": ["refilled"]} for task_id in task_ids],
        )
        return {"status": "completed", "healthy": list(task_ids), "bad": []}

    with (
        patch.object(driver, "prepare_cell_files", side_effect=lambda value, _: value),
        patch.object(driver, "run_task", side_effect=run_task),
    ):
        rc = driver.main([
            "--cell", str(cell_path),
            "--budgets", str(budgets),
            "--task-ids", "task_a", "task_b",
            "--port-base", "43900",
        ])

    assert rc == 0
    assert calls == [["task_b"]]
    status = json.loads((cell_dir / "cell_status.json").read_text())
    assert status["status"] == "incomplete"
    assert status["n_valid_unique"] == 2
    assert status["n_total"] == 3
    assert status["n_retryable"] == 1
    refill = json.loads((cell_dir / "bfcl_refill.json").read_text())
    assert refill["task_ids"] == ["task_c"]


def test_main_refreshes_late_valid_row_before_recursive_retry(tmp_path):
    driver = _load_driver()
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    cell = {
        "cell_id": "bfcl-late-row",
        "cell_dir": str(cell_dir),
        "benchmark": "bfcl",
        "task_ids": ["task_a", "task_b"],
    }
    cell_path = tmp_path / "cell.json"
    cell_path.write_text(json.dumps(cell), encoding="utf-8")
    (cell_dir / "cell.json").write_text(json.dumps(cell), encoding="utf-8")
    budgets = tmp_path / "budgets.json"
    budgets.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []

    def run_task(cell_arg, task_ids, port, batch_name):
        calls.append(list(task_ids))
        if len(calls) == 1:
            # Simulate a row that becomes visible only after run_task's own
            # validation snapshot, while timeout cleanup is completing.
            rows = [{"id": "task_a", "result": ["late but valid"]}]
            status = {"status": "failed", "healthy": [], "bad": list(task_ids)}
        else:
            rows = [{"id": task_id, "result": ["refilled"]} for task_id in task_ids]
            status = {"status": "completed", "healthy": list(task_ids), "bad": []}
        _write_rows(_result_file(Path(cell_arg["cell_dir"]), batch_name), rows)
        return status

    with (
        patch.object(driver, "prepare_cell_files", side_effect=lambda value, _: value),
        patch.object(driver, "run_task", side_effect=run_task),
    ):
        rc = driver.main([
            "--cell", str(cell_path),
            "--budgets", str(budgets),
            "--chunk", "2",
            "--port-base", "43920",
        ])

    assert rc == 0
    assert calls == [["task_a", "task_b"], ["task_b"]]
    status = json.loads((cell_dir / "cell_status.json").read_text())
    assert status["status"] == "complete"
    assert status["n_valid_unique"] == 2
    assert scheduler.cell_done(cell)


def test_appworld_marker_without_official_summary_does_not_complete_task(tmp_path):
    driver = _load_driver()
    task_id = "appworld_task_1"
    marker = tmp_path / "tasks" / task_id / "done.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    assert not driver.appworld_task_completed(tmp_path, task_id)

    summary = tmp_path / "batches" / "first" / "appworld_worker" / "official_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({
        "schema": "a-event-native-appworld-run-v1", "status": "completed",
        "task_id": task_id, "n": 1, "semantic_score": 0.0,
    }), encoding="utf-8")
    assert driver.appworld_task_completed(tmp_path, task_id)


def test_appworld_subset_status_uses_full_manifest(tmp_path):
    driver = _load_driver()
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    task_id = "appworld_task_1"
    cell = {"cell_id": "appworld-subset", "cell_dir": str(cell_dir),
            "benchmark": "acon_appworld", "task_ids": [task_id, "appworld_task_2"]}
    manifest = tmp_path / "cell.json"
    manifest.write_text(json.dumps(cell), encoding="utf-8")
    budgets = tmp_path / "budgets.json"
    budgets.write_text("{}", encoding="utf-8")

    def run_task(cell_arg, task_ids, port, batch_name):
        summary = (Path(cell_arg["cell_dir"]) / "batches" / batch_name
                   / "appworld_worker" / "official_summary.json")
        summary.parent.mkdir(parents=True)
        summary.write_text(json.dumps({
            "schema": "a-event-native-appworld-run-v1", "status": "completed",
            "task_id": task_ids[0], "n": 1, "semantic_score": 0.0,
        }), encoding="utf-8")
        return {"status": "completed", "healthy": list(task_ids), "bad": []}

    with (patch.object(driver, "prepare_cell_files", side_effect=lambda value, _: value),
          patch.object(driver, "run_task", side_effect=run_task)):
        driver.main(["--cell", str(manifest), "--budgets", str(budgets),
                     "--task-ids", task_id, "--port-base", "44001"])

    status = json.loads((cell_dir / "cell_status.json").read_text())
    assert status["status"] == "incomplete"
    assert status["n_total"] == 2
    assert status["n_completed"] == 1
    score = json.loads((cell_dir / "appworld_score_summary.json").read_text())
    assert score["score_denominator"] == 2
    assert score["semantic_score"] is None
    assert score["pending_task_ids"] == ["appworld_task_2"]


def test_appworld_capacity_failure_scores_zero_and_next_worker_continues(
        tmp_path, monkeypatch):
    driver = _load_driver()
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "controller_runtime"))
    from benchmarks.memory_runtime.always_compress import CapacityInfeasible
    from benchmarks.memory_runtime.attempt_journal import AttemptJournal
    from benchmarks.memory_runtime.event_native_api import EventNativeAPI, EventNativeAPIError
    from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    tasks = ["too_large", "next_task"]
    cell = {"cell_id": "appworld-capacity", "cell_dir": str(cell_dir),
            "benchmark": "acon_appworld", "task_ids": tasks}
    manifest = cell_dir / "cell.json"
    manifest.write_text(json.dumps(cell), encoding="utf-8")
    budgets = tmp_path / "budgets.json"
    budgets.write_text("{}", encoding="utf-8")
    calls = []

    def run_task(cell_arg, task_ids, port, batch_name):
        task_id = task_ids[0]
        calls.append(task_id)
        out = Path(cell_arg["cell_dir"]) / "batches" / batch_name
        worker = out / "appworld_worker"
        worker.mkdir(parents=True)
        if task_id == tasks[0]:
            server = out / "server"
            server.mkdir()
            (server / "ready.json").write_text(json.dumps({
                "schema": "a-event-native-server-v1", "benchmark": "acon_appworld",
                "allowed_task_ids": [task_id],
            }), encoding="utf-8")

            def fail_capacity(_payload, **_kwargs):
                raise CapacityInfeasible("mandatory history exceeds budget")

            runner = EventNativeDecisionRunner(
                types.SimpleNamespace(prepare=fail_capacity),
                types.SimpleNamespace(close_session=lambda: None,
                                      session_cache_info=lambda: {}),
                object(), ratio=4, max_new_tokens=1, max_generation_calls=3,
                journal=AttemptJournal(server / "attempts.jsonl"),
            )
            api = EventNativeAPI(
                runner, run_id="capacity-fixture", model_name="fixture",
                benchmark="acon_appworld", view_mode="static", max_new_tokens=1,
                allowed_task_ids=[task_id], max_decisions=3,
                deadline_monotonic=time.monotonic() + 60,
                steps_path=server / "steps.jsonl",
            )
            api._validate_request = lambda _request: (
                {"session_id": f"acon_appworld/{task_id}/attempt-0",
                 "decision_key": "turn-0/step-0", "messages": [], "tools": []},
                (task_id, 0, 0), "fixture-signature",
            )
            with pytest.raises(EventNativeAPIError) as failed:
                api.handle_chat({"task_id": task_id})
            assert (failed.value.status_code, failed.value.code) == (
                422, "c2kv_capacity_infeasible")
            assert api.health()["terminal"] is False
            summary = {"schema": "a-event-native-appworld-run-v1",
                       "status": "failed:CalledProcessError", "task_id": task_id,
                       "n": 0, "semantic_score": None}
            result = {"status": "failed", "error": "official worker exited rc=1"}
        else:
            summary = {"schema": "a-event-native-appworld-run-v1",
                       "status": "completed", "task_id": task_id,
                       "n": 1, "semantic_score": 1.0}
            result = {"status": "completed", "healthy": [task_id], "bad": []}
        (worker / "official_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return result

    with (patch.object(driver, "prepare_cell_files", side_effect=lambda value, _: value),
          patch.object(driver, "run_task", side_effect=run_task)):
        assert driver.main(["--cell", str(manifest), "--budgets", str(budgets)]) == 0
        assert driver.main(["--cell", str(manifest), "--budgets", str(budgets)]) == 0

    assert calls == tasks
    terminal = driver.appworld_method_failure_receipt(cell_dir, tasks[0])
    assert terminal is not None and terminal["score_source"] == "method_failure_zero"
    assert not driver.appworld_task_completed(cell_dir, tasks[0])
    score = json.loads((cell_dir / "appworld_score_summary.json").read_text())
    assert score["semantic_score"] == 0.5
    assert score["score_denominator"] == 2
    assert score["n_official_scored"] == 1
    assert score["n_method_failures"] == 1
    assert {row["task_id"]: row["score_source"] for row in score["task_rows"]} == {
        tasks[0]: "method_failure_zero", tasks[1]: "official_appworld"}
    status = json.loads((cell_dir / "cell_status.json").read_text())
    assert status["status"] == "complete"
    assert (status["n_completed"], status["n_terminal"], status["n_total"]) == (1, 1, 2)
    assert scheduler.cell_done(cell)


def test_appworld_method_failure_requires_typed_task_bound_evidence(tmp_path):
    driver = _load_driver()
    out = tmp_path / "batch"
    server = out / "server"
    server.mkdir(parents=True)
    ready = {"schema": "a-event-native-server-v1", "benchmark": "acon_appworld",
             "allowed_task_ids": ["task_a"]}
    (server / "ready.json").write_text(json.dumps(ready), encoding="utf-8")
    typed = {"schema": "a-event-native-exact-step-v1", "status": "failed",
             "session_id": "acon_appworld/task_a/attempt-0",
             "failure_kind": "method_failure",
             "failure_code": "c2kv_capacity_infeasible"}
    steps = server / "steps.jsonl"
    for row in (
        {**typed, "session_id": "acon_appworld/task_b/attempt-0"},
        {**typed, "failure_code": "runner_failed"},
        {"schema": typed["schema"], "status": "failed",
         "session_id": typed["session_id"],
         "error": {"message": "CapacityInfeasible mentioned by a 500 error"}},
    ):
        steps.write_text(json.dumps(row) + "\n", encoding="utf-8")
        assert driver.appworld_method_failure_evidence(out, "task_a") is None
    steps.write_text(json.dumps(typed) + "\n", encoding="utf-8")
    assert driver.appworld_method_failure_evidence(out, "task_a") is not None
    ready["allowed_task_ids"] = ["task_b"]
    (server / "ready.json").write_text(json.dumps(ready), encoding="utf-8")
    assert driver.appworld_method_failure_evidence(out, "task_a") is None


def test_appworld_capacity_failure_cannot_hide_cost_finalization_error(tmp_path):
    driver = _load_driver()
    cell_dir = tmp_path / "cell"
    cell = {"cell_id": "appworld-cost-error", "cell_dir": str(cell_dir),
            "benchmark": "acon_appworld", "task_ids": ["task_a"]}
    cell_dir.mkdir()
    manifest = cell_dir / "cell.json"
    manifest.write_text(json.dumps(cell), encoding="utf-8")
    budgets = tmp_path / "budgets.json"
    budgets.write_text("{}", encoding="utf-8")

    def run_task(cell_arg, task_ids, port, batch_name):
        out = Path(cell_arg["cell_dir"]) / "batches" / batch_name
        server = out / "server"
        server.mkdir(parents=True)
        (server / "ready.json").write_text(json.dumps({
            "schema": "a-event-native-server-v1", "benchmark": "acon_appworld",
            "allowed_task_ids": task_ids,
        }), encoding="utf-8")
        (server / "steps.jsonl").write_text(json.dumps({
            "schema": "a-event-native-exact-step-v1", "status": "failed",
            "session_id": "acon_appworld/task_a/attempt-0",
            "failure_kind": "method_failure",
            "failure_code": "c2kv_capacity_infeasible",
        }) + "\n", encoding="utf-8")
        (server / "final.json").write_text(json.dumps({
            "status": "failed", "cost_summary_error": {
                "type": "ValueError", "message": "invalid cost inventory"},
        }), encoding="utf-8")
        return {"status": "failed", "error": "controller exited rc=1"}

    with (patch.object(driver, "prepare_cell_files", side_effect=lambda value, _: value),
          patch.object(driver, "run_task", side_effect=run_task)):
        assert driver.main(["--cell", str(manifest), "--budgets", str(budgets),
                            "--port-base", "44100"]) == 0

    assert not list((cell_dir / "tasks").glob("*/terminal.json"))
    assert driver.appworld_method_failure_receipt(cell_dir, "task_a") is None
    score = json.loads((cell_dir / "appworld_score_summary.json").read_text())
    assert score["semantic_score"] is None
    assert score["pending_task_ids"] == ["task_a"]


def test_prepare_rejects_budget_drift_without_rewriting_frozen_files(tmp_path):
    driver = _load_driver()
    cell_dir = tmp_path / "cell"
    cell = {
        "cell_id": "budget-freeze", "cell_dir": str(cell_dir),
        "condition": "recovery_off_same_initial", "working_point": "K0",
        "sglang_backend_url": "http://127.0.0.1:8000",
    }
    budgets = {"working_points": {"K0": {
        "history_allowance_bytes": 1024, "common_cap_bytes": 2048,
    }}}
    with patch.object(driver, "_controller_with_binding", return_value=({"controller": 1}, None)):
        driver.prepare_cell_files(cell, budgets)
        (cell_dir / "batches" / "first").mkdir(parents=True)
        frozen = {name: (cell_dir / name).read_bytes() for name in
                  ("cell.json", "controller.json", "eval_policy.json")}

        changed = {"working_points": {"K0": {
            "history_allowance_bytes": 2048, "common_cap_bytes": 2048,
        }}}
        with pytest.raises(ValueError, match="different frozen eval_policy.json"):
            driver.prepare_cell_files(cell, changed)

    assert {name: (cell_dir / name).read_bytes() for name in frozen} == frozen


def test_prepare_rejects_threshold_drift_without_rewriting_frozen_files(tmp_path):
    driver = _load_driver()
    cell_dir = tmp_path / "cell"
    cell = {
        "cell_id": "threshold-freeze", "cell_dir": str(cell_dir),
        "condition": "tracer_history", "working_point": "K0",
        "threshold": 0.5, "sglang_backend_url": "http://127.0.0.1:8000",
    }
    budgets = {"working_points": {"K0": {
        "history_allowance_bytes": 1024, "common_cap_bytes": 2048,
    }}}

    def configured(value):
        return {"threshold": value["threshold"]}, {"bound_threshold": value["threshold"]}

    with patch.object(driver, "_controller_with_binding", side_effect=configured):
        driver.prepare_cell_files(cell, budgets)
        (cell_dir / "batches" / "first").mkdir(parents=True)
        frozen = {name: (cell_dir / name).read_bytes() for name in
                  ("cell.json", "controller.json", "eval_policy.json",
                   "risk_artifact_binding.json")}

        with pytest.raises(ValueError, match="different frozen controller.json"):
            driver.prepare_cell_files({**cell, "threshold": 0.6}, budgets)

    assert {name: (cell_dir / name).read_bytes() for name in frozen} == frozen


def test_prepare_matching_resume_preserves_frozen_bytes(tmp_path):
    driver = _load_driver()
    cell_dir = tmp_path / "cell"
    cell = {
        "cell_id": "matching-freeze", "cell_dir": str(cell_dir),
        "condition": "recovery_off_same_initial", "working_point": "K0",
        "sglang_backend_url": "http://127.0.0.1:8000",
    }
    budgets = {"working_points": {"K0": {
        "history_allowance_bytes": 1024, "common_cap_bytes": 2048,
    }}}
    with patch.object(driver, "_controller_with_binding", return_value=({"controller": 1}, None)):
        driver.prepare_cell_files(cell, budgets)
        (cell_dir / "batches" / "first").mkdir(parents=True)
        # Historical source manifests did not include the two derived paths.
        manifest_path = cell_dir / "cell.json"
        historical = json.loads(manifest_path.read_text(encoding="utf-8"))
        historical.pop("controller_path")
        historical.pop("eval_policy_path")
        manifest_path.write_text(json.dumps(historical), encoding="utf-8")
        frozen = {name: (cell_dir / name).read_bytes() for name in
                  ("cell.json", "controller.json", "eval_policy.json")}
        resumed = driver.prepare_cell_files(
            {**cell, "sglang_backend_url": "http://127.0.0.1:8001",
             "scheduler_port_slot": 2}, budgets
        )

    assert resumed["sglang_backend_url"] == "http://127.0.0.1:8001"
    assert {name: (cell_dir / name).read_bytes() for name in frozen} == frozen


def test_prepare_checks_recorded_derived_config_paths(tmp_path):
    driver = _load_driver()
    cell_dir = tmp_path / "cell"
    cell = {
        "cell_id": "recorded-path-freeze", "cell_dir": str(cell_dir),
        "condition": "recovery_off_same_initial", "working_point": "K0",
    }
    budgets = {"working_points": {"K0": {
        "history_allowance_bytes": 1024, "common_cap_bytes": 2048,
    }}}
    with patch.object(driver, "_controller_with_binding", return_value=({"controller": 1}, None)):
        driver.prepare_cell_files(cell, budgets)
        (cell_dir / "batches" / "first").mkdir(parents=True)
        for name, recorded in (("controller.json", "old_controller.json"),
                               ("eval_policy.json", "old_eval_policy.json")):
            (cell_dir / recorded).write_bytes((cell_dir / name).read_bytes())
        manifest_path = cell_dir / "cell.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["controller_path"] = "old_controller.json"
        manifest["eval_policy_path"] = "old_eval_policy.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        frozen = {name: (cell_dir / name).read_bytes() for name in
                  ("cell.json", "controller.json", "eval_policy.json")}

        driver.prepare_cell_files(cell, budgets)
        (cell_dir / "old_controller.json").write_text('{"controller": 2}', encoding="utf-8")
        with pytest.raises(ValueError, match="different frozen controller.json at recorded path"):
            driver.prepare_cell_files(cell, budgets)

    assert {name: (cell_dir / name).read_bytes() for name in frozen} == frozen


def test_prepare_rejects_attempts_with_missing_frozen_config(tmp_path):
    driver = _load_driver()
    cell_dir = tmp_path / "cell"
    cell = {
        "cell_id": "missing-config-freeze", "cell_dir": str(cell_dir),
        "condition": "recovery_off_same_initial", "working_point": "K0",
    }
    budgets = {"working_points": {"K0": {
        "history_allowance_bytes": 1024, "common_cap_bytes": 2048,
    }}}
    with patch.object(driver, "_controller_with_binding", return_value=({"controller": 1}, None)):
        driver.prepare_cell_files(cell, budgets)
        (cell_dir / "batches" / "first").mkdir(parents=True)
        (cell_dir / "controller.json").unlink()
        frozen = {name: (cell_dir / name).read_bytes() for name in
                  ("cell.json", "eval_policy.json")}
        with pytest.raises(ValueError, match="existing attempts require frozen controller.json"):
            driver.prepare_cell_files(cell, budgets)
    assert not (cell_dir / "controller.json").exists()
    assert {name: (cell_dir / name).read_bytes() for name in frozen} == frozen


def test_retrieval_device_override_preserves_freeze_and_records_effective_attempt(tmp_path, monkeypatch):
    driver = _load_driver()
    cell_dir = tmp_path / "cell"
    cell = {"cell_id": "placement", "cell_dir": str(cell_dir),
            "condition": "tracer_history", "working_point": "K0"}
    budgets = {"working_points": {"K0": {
        "history_allowance_bytes": 1024, "common_cap_bytes": 2048}}}
    controller = {"gp_experiments": {"selector_threshold": 0.6, "local_models": {
        "embedding": {"device": "cpu", "batch_size": 16,
                      "dtype": "bfloat16", "model_name_or_path": "/frozen/model"}}}}
    with patch.object(driver, "_controller_with_binding", return_value=(controller, None)):
        driver.prepare_cell_files(cell, budgets)
        (cell_dir / "batches" / "old_cpu").mkdir(parents=True)
        frozen = {name: (cell_dir / name).read_bytes() for name in
                  ("cell.json", "controller.json", "eval_policy.json")}
        prepared = driver.prepare_cell_files(
            {**cell, "embedding_device": "npu:0", "embedding_batch_size": 1}, budgets)
    assert {name: (cell_dir / name).read_bytes() for name in frozen} == frozen
    out = cell_dir / "batches" / "new_npu"
    out.mkdir()
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3")
    runtime = driver._runtime_retrieval_cell(prepared, out)
    effective = json.loads(Path(runtime["controller_path"]).read_text())
    embedding = effective["gp_experiments"]["local_models"]["embedding"]
    assert embedding.pop("device") == "npu:0"
    assert embedding.pop("batch_size") == 1
    embedding.update(device="cpu", batch_size=16)
    assert effective == controller
    sidecar = json.loads((out / "retrieval_execution.json").read_text())
    assert sidecar["ascend_visible_devices"] == "3"
    assert sidecar["frozen_controller_sha256"] != sidecar["effective_controller_sha256"]
    assert {name: (cell_dir / name).read_bytes() for name in frozen} == frozen
    with pytest.raises(FileExistsError):
        driver._runtime_retrieval_cell(prepared, out)
