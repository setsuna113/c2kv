from __future__ import annotations

import json

import evidence_eval_c2_remaining as subject


def test_lane_is_fixed_to_npu0_with_nonoverlapping_ports():
    lane = subject.lane_specs()[0]
    assert lane["physical_device"] == 0
    assert lane["engine_port"] == 37800
    assert lane["task_ports"] == list(range(37900, 37908))
    assert len({lane["engine_port"], *lane["task_ports"]}) == 9
    assert lane["after_status_path"] == str(subject.AFTER_STATUS_PATH)


def test_terminal_predecessor_requires_exact_phase_device_and_finish(tmp_path):
    status = tmp_path / "slot.json"
    lane = {**subject.lane_specs()[0], "after_status_path": str(status)}
    status.write_text(
        json.dumps(
            {
                "phase": "worker_running",
                "physical_device": 0,
                "worker_id": "npu0",
            }
        ),
        encoding="utf-8",
    )
    assert subject._terminal_predecessor_receipt(lane) is None

    status.write_text(
        json.dumps(
            {
                "phase": "completed_no_work",
                "physical_device": 0,
                "worker_id": "npu0",
                "finished_at_epoch": 123.0,
            }
        ),
        encoding="utf-8",
    )
    receipt = subject._terminal_predecessor_receipt(lane)
    assert receipt["phase"] == "completed_no_work"
    assert receipt["physical_device"] == 0


def test_runner_command_uses_outer_dispatcher(tmp_path):
    lane = subject.lane_specs()[0]
    command = subject._runner_command(tmp_path, lane)
    assert command[1] == str(tmp_path / "c2_remaining_dispatcher.py")
    assert command[2] == "run"
    assert command[command.index("--port-base") + 1] == "37900"
