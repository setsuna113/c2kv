from __future__ import annotations

import json

import pytest

from evidence_eval_c2_release_v2 import classify_release, verify_zero_started


def test_terminal_global_failure_is_a_truthful_release():
    receipt = classify_release(
        {
            "phase": "failed_no_retry",
            "worker_id": "npu0",
            "physical_device": 0,
            "finished_at_epoch": 123.0,
            "error": "Global run aborted",
        },
        {
            "phase": "failed_no_retry",
            "coordinator_exit_code": 1,
            "abort_receipt": "abort.json",
        },
        {"phase": "coordinator", "error_type": "RuntimeError", "error": "worker failed"},
    )
    assert receipt["release_kind"] == "terminal_global_failure"
    assert receipt["slot_phase"] == "failed_no_retry"


def test_failed_slot_requires_global_abort_receipt():
    slot = {
        "phase": "failed_no_retry",
        "worker_id": "npu0",
        "physical_device": 0,
        "finished_at_epoch": 123.0,
        "error": "Global run aborted",
    }
    with pytest.raises(RuntimeError, match="global abort"):
        classify_release(
            slot,
            {"phase": "failed_no_retry", "abort_receipt": "abort.json"},
            None,
        )


def test_nonterminal_slot_waits():
    assert classify_release(
        {"phase": "worker_running", "worker_id": "npu0", "physical_device": 0},
        None,
        None,
    ) is None


def test_zero_started_rejects_any_lane_artifact(tmp_path):
    package = tmp_path
    (package / "run").mkdir()
    (package / "relay").mkdir()
    (package / "lanes/C2").mkdir(parents=True)
    (package / "run/launch.json").write_text(
        json.dumps(
            {"approved_task_executions": 8, "automatic_retries": 0, "automatic_reruns": 0}
        ),
        encoding="utf-8",
    )
    (package / "relay/C2.json").write_text(
        json.dumps(
            {"state": "waiting_for_t02_npu0_terminal", "supervisor_pid": 849483}
        ),
        encoding="utf-8",
    )
    assert verify_zero_started(package)["task_starts"] == 0
    (package / "lanes/C2/run").mkdir()
    with pytest.raises(RuntimeError, match="already has run/results"):
        verify_zero_started(package)
