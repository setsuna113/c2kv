from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks/memory_runtime"))
import official_pilot


def test_designs_preserve_default_and_freeze_lease_dev2():
    default = official_pilot.design_spec("first-dev4")
    assert default["task_ids"] == [f"multi_turn_base_{i}" for i in range(4)]
    assert default["variants"] == [
        ("full", "full"), ("legacy", "c2kv4"), ("protect", "c2kv4")]
    assert default["command_extra"] == []

    lease = official_pilot.design_spec("lease-dev2")
    assert lease["task_ids"] == ["multi_turn_base_1", "multi_turn_base_30"]
    assert lease["variants"] == [
        ("full", "full"), ("legacy", "c2kv4"), ("protect", "c2kv4"),
        ("recover_once", "c2kv4"), ("persistent", "c2kv4"),
        ("no_gist", "full")]
    assert lease["command_extra"] == [
        "--capture-request-views", "--bfcl-temperature", "0.001", "--bfcl-seed", "0"]


def write_gate(root, *, bad_cell=None):
    directory = root / "utilization_probe_v1"
    directory.mkdir()
    (directory / "receipt.json").write_text(json.dumps({
        "status": "completed", "chat_attempts": 24, "chat_completed": 24,
        "lease_gate_passed": True}))
    for index in range(24):
        row = {"cell_id": f"c{index}", "status": "completed", "counts": {
            "memory_runtime": {"raw_prompt_tokens_verified_by_backend": True,
                               "byte_geometry_verified_by_backend": True}}}
        if index == bad_cell:
            row["counts"]["memory_runtime"]["raw_prompt_tokens_verified_by_backend"] = False
        (directory / f"c{index}.json").write_text(json.dumps(row))
    return directory


def test_utilization_gate_reads_original_cells(tmp_path):
    directory = write_gate(tmp_path)
    assert official_pilot.require_utilization_gate(tmp_path) == directory / "receipt.json"
    (directory / "c23.json").unlink()
    with pytest.raises(SystemExit, match="exactly 24"):
        official_pilot.require_utilization_gate(tmp_path)


def test_utilization_gate_rejects_unverified_cell(tmp_path):
    write_gate(tmp_path, bad_cell=7)
    with pytest.raises(SystemExit, match="token/byte verification"):
        official_pilot.require_utilization_gate(tmp_path)


def test_timeout_termination_escalates_to_owned_process_group(monkeypatch):
    signals = []
    monkeypatch.setattr(official_pilot.os, "killpg",
                        lambda pid, sig: signals.append((pid, sig)), raising=False)
    monkeypatch.setattr(official_pilot.signal, "SIGKILL", 9, raising=False)
    waits = iter([official_pilot.subprocess.TimeoutExpired("pilot", 10), 9])

    def wait(timeout=None):
        result = next(waits)
        if isinstance(result, BaseException):
            raise result
        return result

    proc = SimpleNamespace(pid=123, wait=wait)
    assert official_pilot.terminate_process_group(proc) == 9
    assert signals == [(123, official_pilot.signal.SIGTERM),
                       (123, official_pilot.signal.SIGKILL)]
