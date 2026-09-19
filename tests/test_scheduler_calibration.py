"""A live calibration owns its card but does not hide detached controllers."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from generality import scheduler_npu as scheduler


def _process(root: Path, pid: int, parent: int, argv: list[str]) -> None:
    proc = root / str(pid)
    proc.mkdir()
    (proc / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")
    (proc / "stat").write_text(
        f"{pid} (python) " + " ".join(["S", str(parent)] + ["0"] * 18))


def test_calibration_owner_reserves_card_and_orphan_remains_visible(tmp_path, monkeypatch):
    root = tmp_path / "project"
    source = root / "src"
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    script = source / "generality" / "c2kv_calibration.py"
    out = root / "calibration" / "c2kv" / "K0" / "attempt-1"
    server_out = out / "controller" / "server"
    monkeypatch.setattr(scheduler, "GENERATION_ROOT", root)
    monkeypatch.setattr(scheduler, "SRC", source)
    _process(proc_root, 100, 1, ["python", script.as_posix(), "--run",
                              "--out", out.as_posix(),
                              "--sglang-backend-url", "http://127.0.0.1:36200"])
    _process(proc_root, 200, 100, ["python", "-m",
                               "benchmarks.memory_runtime.event_native_server",
                               "--out", server_out.as_posix()])
    _process(proc_root, 201, 200, ["python", "-m",
                               "benchmarks.memory_runtime.event_native_server",
                               "--serve-child", "--out", server_out.as_posix()])

    assert scheduler.live_calibration_owners(proc_root) == {
        100: {"card": 0, "out": str(out.resolve())}}
    assert scheduler.unmanaged_event_native_servers(proc_root) == []

    (proc_root / "100" / "cmdline").unlink()
    (proc_root / "100" / "stat").unlink()
    assert {row["pid"] for row in scheduler.unmanaged_event_native_servers(proc_root)} == {200, 201}


def test_scheduler_refuses_card_with_live_calibration(monkeypatch):
    monkeypatch.setattr(scheduler, "live_calibration_owners",
                        lambda: {100: {"card": 0, "out": "/staged"}})
    monkeypatch.setattr(scheduler, "unmanaged_event_native_servers", lambda: [])
    with pytest.raises(RuntimeError, match="calibration owns requested cards"):
        scheduler.run_scheduler(SimpleNamespace(cards=[0]))
