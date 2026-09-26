import copy
import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[3] / "tmp/reassign_c4_task_npu2.py"
SPEC = importlib.util.spec_from_file_location("reassign_c4_task_npu2", SCRIPT)
helper = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(helper)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _fixture(tmp_path: Path, pid: int = 1011596):
    base = tmp_path / "root"
    code = base / "post_t02_v4"
    command = [
        "/python", str(code / "evidence_eval_trained.py"),
        "--package", str(base / "eval_trained_v4/C4_task"),
        "--training", str(base / "prepared_v8"),
        "--training-binding", str(code / "prepared_v8.binding.json"),
        "--source", str(base / "eval_never_started_c2c3_v2"),
        "--lane", "C4_task", "--after-status",
        str(base / "eval_failed_repair_v1/lanes/C0/run/status.json"),
    ]
    scheduled = {"schema": "evidence-post-t02-scheduled-v4",
                 "children": [{"name": "C4_task", "pid": pid, "command": command}]}
    _write(code / "scheduled.json", scheduled)
    _write(base / "eval_trained_v4/C4_task.waiting.json",
           {"phase": "waiting_for_training", "pid": pid, "lane": "C4_task"})
    proc_root = tmp_path / "proc"
    process = proc_root / str(pid)
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"\0".join(x.encode() for x in command) + b"\0")
    verified = {"code": code, "scheduled_path": code / "scheduled.json",
                "scheduled": scheduled, "binding": code / "prepared_v8.binding.json"}
    return base, proc_root, verified, command


def test_inspection_accepts_only_untouched_waiting_c4_task(tmp_path: Path):
    base, proc_root, verified, command = _fixture(tmp_path)
    value = helper.inspect_old_waiter(base, 1011596, proc_root=proc_root,
                                      verified=verified)
    assert value["command"] == command
    assert value["waiting"]["phase"] == "waiting_for_training"
    assert Path(value["new_package"]) == base / "eval_trained_v4/C4_task_npu2"


def test_inspection_rejects_started_package_and_wrong_lane(tmp_path: Path):
    base, proc_root, verified, command = _fixture(tmp_path)
    (base / "eval_trained_v4/C4_task").mkdir()
    with pytest.raises(FileExistsError, match="already started"):
        helper.inspect_old_waiter(base, 1011596, proc_root=proc_root, verified=verified)
    (base / "eval_trained_v4/C4_task").rmdir()
    wrong = ["C1" if value == "C4_task" else value for value in command]
    (proc_root / "1011596/cmdline").write_bytes(
        b"\0".join(x.encode() for x in wrong) + b"\0")
    with pytest.raises(ValueError, match="not the frozen"):
        helper.inspect_old_waiter(base, 1011596, proc_root=proc_root, verified=verified)


def test_stop_signals_only_verified_pid_and_never_process_group(tmp_path: Path):
    base, proc_root, verified, _ = _fixture(tmp_path)
    snapshot = helper.inspect_old_waiter(base, 1011596, proc_root=proc_root,
                                         verified=verified)
    calls = []

    def kill(pid, sig):
        calls.append((pid, sig))
        (proc_root / str(pid) / "cmdline").unlink()
        (proc_root / str(pid)).rmdir()

    helper.stop_old_waiter(snapshot, proc_root=proc_root, kill_fn=kill,
                           sleep_fn=lambda _: None)
    assert calls == [(1011596, signal.SIGTERM)]


def test_hardware_override_changes_only_device_and_engine_port():
    lanes = {
        "C1": {"physical_device": 4, "engine_port": 37440, "task_port_base": 37500},
        "C4_turn": {"physical_device": 6, "engine_port": 37460, "task_port_base": 37520},
        "C4_task": {"physical_device": 0, "engine_port": 37400,
                    "task_port_base": 37540, "selector": "gain_task",
                    "artifact": "training/c4/c4_gain_task.json", "kind": "c4_gain_task"},
    }
    module = SimpleNamespace(LANES=copy.deepcopy(lanes))
    value = helper.apply_hardware_override(module)
    assert value["changed_fields"] == ["physical_device", "engine_port"]
    assert module.LANES["C4_task"]["physical_device"] == 2
    assert module.LANES["C4_task"]["engine_port"] == 37420
    assert module.LANES["C4_task"]["task_port_base"] == 37540
    assert module.LANES["C1"] == lanes["C1"]
    assert module.LANES["C4_turn"] == lanes["C4_turn"]
