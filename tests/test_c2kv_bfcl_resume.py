from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

from generality.bfcl_results import collect_bfcl_results, completion_receipt


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


def test_audit_only_writes_full_refill_manifest_without_preparing_runtime(tmp_path):
    driver = _load_driver()
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    cell = {
        "cell_id": "bfcl-test",
        "cell_dir": str(cell_dir),
        "benchmark": "bfcl",
        "task_ids": ["task_a", "task_a", "task_b"],
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
