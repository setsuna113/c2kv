"""CPU checks for the NPU scheduler's ownership and hold contracts."""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from generality import scheduler_npu as scheduler
from generality import scheduler as entry
from generality.completion_contract import write_cell_status


@pytest.fixture(autouse=True)
def _isolate_live_calibration_owners(monkeypatch):
    # Unit fixtures must never depend on jobs running on the test host.
    # Calibration ownership itself is exercised with explicit fake processes.
    monkeypatch.setattr(scheduler, "live_calibration_owners", lambda *args: {})


def cell(tmp_path, backend="h2o", benchmark="appworld"):
    directory = tmp_path / "cell"
    directory.mkdir(exist_ok=True)
    return {
        "cell_id": f"{benchmark}__{backend}__K0__compression_full_budget",
        "cell_dir": str(directory), "backend": backend,
        "benchmark_key": benchmark, "condition": "compression_full_budget",
    }


def test_production_hold_cannot_be_overridden_or_start_engines(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(scheduler, "BLOCKED_BACKENDS", {"c2kv", "h2o", "snapkv", "pyramidkv"})
    rows = [cell(tmp_path, backend=backend, benchmark=benchmark)
            for backend in scheduler.BLOCKED_BACKENDS
            for benchmark in ("appworld", "bfcl_base", "bfcl_long_context")]
    assert scheduler.queue_cells(rows, only_ready=True) == []
    assert scheduler.queue_cells(rows, only_ready=False) == []
    monkeypatch.setattr(scheduler, "enumerate_cells", lambda: rows)
    monkeypatch.setattr(scheduler, "ensure_engines", lambda cards: pytest.fail("engine started"))
    monkeypatch.setattr(scheduler, "acquire_scheduler_lock", lambda: pytest.fail("lock acquired"))
    assert scheduler.main(["--dry-run", "--include-pending", "--cards", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["cells"] == []
    assert scheduler.main(["--cards", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["event"] == "scheduler_no_runnable_cells"


def test_completed_or_exhausted_cells_do_not_start_engines(tmp_path, monkeypatch, capsys):
    complete = cell(tmp_path, backend="fixture")
    complete["cell_id"] = "complete"
    complete["task_ids"] = ["task_1"]
    Path(complete["cell_dir"], "cell.json").write_text(json.dumps(complete))
    write_cell_status(complete, {"cell_id": "complete", "status": "complete",
                                 "n_completed": 1, "n_total": 1})
    exhausted = dict(complete, cell_id="exhausted", cell_dir=str(tmp_path / "exhausted"))
    Path(exhausted["cell_dir"]).mkdir()
    Path(exhausted["cell_dir"], "scheduler_attempts.json").write_text(
        json.dumps({"cell_id": exhausted["cell_id"],
                    "launch_attempts": scheduler.MAX_ATTEMPTS}))
    monkeypatch.setattr(scheduler, "enumerate_cells", lambda: [complete, exhausted])
    monkeypatch.setattr(scheduler, "ensure_engines", lambda cards: pytest.fail("engine started"))
    assert scheduler.main(["--cards", "1"]) == 0
    assert json.loads(capsys.readouterr().out)["event"] == "scheduler_no_runnable_cells"


def test_only_manifest_bound_complete_status_skips_driver(tmp_path):
    complete = cell(tmp_path, backend="fixture")
    complete["task_ids"] = ["task_1"]
    manifest = Path(complete["cell_dir"], "cell.json")
    manifest.write_text(json.dumps(complete))
    status_path = manifest.with_name("cell_status.json")
    status_path.write_text(json.dumps({"cell_id": complete["cell_id"],
                                       "status": "complete", "n_completed": 1,
                                       "n_total": 1}))
    assert not scheduler.cell_done(complete)
    write_cell_status(complete, {"cell_id": complete["cell_id"],
                                 "status": "complete", "n_completed": 1,
                                 "n_total": 1})
    assert scheduler.cell_done(complete)
    assert "completion_contract" in json.loads(status_path.read_text())
    assert json.loads(status_path.with_name("cell_status_history.jsonl").read_text())[
        "previous_raw"] == '{"cell_id": "' + complete["cell_id"] + '", "status": "complete", "n_completed": 1, "n_total": 1}'
    changed = dict(complete, task_ids=["task_1", "task_2"])
    manifest.write_text(json.dumps(changed))
    assert not scheduler.cell_done(changed)


def test_empty_card_list_is_rejected():
    with pytest.raises(SystemExit, match="2"):
        scheduler.main(["--cards"])


def test_card_list_is_required():
    with pytest.raises(SystemExit, match="2"):
        scheduler.main([])


def test_unimplemented_extra_benchmark_routes_stay_held(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "BLOCKED_BACKENDS", set())
    monkeypatch.setattr(scheduler, "BLOCKED_CELL_KEYS", set())
    for benchmark in ("toolsandbox", "acebench"):
        assert scheduler.cell_blocked(cell(tmp_path, backend="c2kv", benchmark=benchmark))
        tracer = cell(tmp_path, backend="h2o", benchmark=benchmark)
        tracer["condition"] = "tracer_history"
        assert scheduler.cell_blocked(tracer)
        assert not scheduler.cell_blocked(cell(tmp_path, backend="h2o", benchmark=benchmark))


@pytest.mark.parametrize("backend,threshold", [
    ("c2kv", 0.6), ("h2o", 0.3), ("snapkv", 0.3), ("pyramidkv", 0.3),
])
def test_tau2_tracer_uses_frozen_threshold_without_calibration(
        tmp_path, monkeypatch, backend, threshold):
    task = cell(tmp_path, backend=backend, benchmark="tau2")
    task.update(condition="tracer_history", threshold=threshold,
                threshold_status="frozen_t02")
    monkeypatch.setattr(scheduler, "calibration_receipt", lambda *_:
                        pytest.fail("tau2 T must not read historical calibration"))
    monkeypatch.setattr(scheduler, "LOGS", tmp_path / "logs")
    assert scheduler.cells_ready(task)
    captured = {}
    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda command, **kwargs:
                        captured.update(command=command) or SimpleNamespace(pid=123))
    scheduler.launch_cell(task, card=1, slot=0)
    command = captured["command"]
    manifest = json.loads(Path(command[command.index("--cell") + 1]).read_text())
    assert manifest["threshold"] == threshold
    assert manifest["threshold_status"] == "frozen_t02"
    assert "calibration_receipt" not in manifest
    wrong = {**task, "threshold": 0.5}
    assert not scheduler.cells_ready(wrong)
    with pytest.raises(RuntimeError, match="frozen T02 threshold"):
        scheduler.launch_cell(wrong, card=1, slot=0)


def test_canonical_entry_delegates_and_runs_as_a_script():
    assert entry.main is scheduler.main
    assert entry.BLOCKED_BACKENDS is scheduler.BLOCKED_BACKENDS
    assert entry.BACKEND_ORDER == {"c2kv": 0, "h2o": 1, "snapkv": 2, "pyramidkv": 9}
    result = subprocess.run(
        [sys.executable, str(Path(entry.__file__)), "--help"],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "--max-drivers-per-card" in result.stdout


def test_live_driver_scan_reserves_orphan_cards_and_fails_closed(tmp_path, monkeypatch):
    first = cell(tmp_path)
    path = Path(first["cell_dir"]) / "cell_launch.json"
    path.write_text(json.dumps({**first, "sglang_backend_url": "http://127.0.0.1:36203"}))
    second = dict(first, cell_id="another", cell_dir=str(tmp_path / "another"))
    Path(second["cell_dir"]).mkdir()
    second_path = Path(second["cell_dir"]) / "cell_launch_attempts" / "launch-1.json"
    second_path.parent.mkdir()
    second_path.write_text(json.dumps({**second, "sglang_backend_url": "http://127.0.0.1:36203",
                                       "scheduler_port_slot": 1}))
    output = (f"123 python historykv_cell.py --cell {path}\n"
              f"124 python historykv_cell.py --cell {second_path}\n")
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=output))
    live = scheduler.live_driver_assignments()
    assert set(live[3]) == {first["cell_dir"], second["cell_dir"]}
    assert live[3][first["cell_dir"]].get("scheduler_port_slot") is None
    assert live[3][second["cell_dir"]]["scheduler_port_slot"] == 1
    running = {10: (object(), cell(tmp_path), 1, 0)}
    assert scheduler.free_healthy_slots([1, 2, 3], running,
                                        live, 1) == [(2, 0)]
    assert scheduler.free_healthy_slots([3], {},
                                        {3: {first["cell_dir"]: first}}, 2) == [(3, 1)]
    path.write_text("{")
    with pytest.raises(RuntimeError, match="Cannot identify live driver card"):
        scheduler.live_driver_assignments()
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=2, stdout=""))
    with pytest.raises(RuntimeError, match="Live driver scan failed"):
        scheduler.live_driver_assignments()


def test_direct_driver_manifest_reserves_its_card(tmp_path, monkeypatch):
    direct = cell(tmp_path)
    path = Path(direct["cell_dir"]) / "cell.json"
    path.write_text(json.dumps({**direct, "sglang_backend_url": "http://127.0.0.1:36203"}))
    output = f"123 python session_tracer_cell.py --cell {path}\n"
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **k:
                        SimpleNamespace(returncode=0, stdout=output))
    live = scheduler.live_driver_assignments()
    assert list(live) == [3]
    assert scheduler.free_healthy_slots([2, 3], {}, live, 1) == [(2, 0)]
    path.write_text(json.dumps({**direct, "cell_dir": str(tmp_path / "wrong"),
                                "sglang_backend_url": "http://127.0.0.1:36203"}))
    with pytest.raises(RuntimeError, match="mismatched cell_dir"):
        scheduler.live_driver_assignments()


def test_old_scheduler_process_is_detected(tmp_path, monkeypatch):
    proc_root = tmp_path / "proc"
    proc = proc_root / "123456"
    cwd = proc / "cwd"
    script = cwd / "src" / "generality" / "scheduler.py"
    script.parent.mkdir(parents=True)
    script.write_text("# fixture")
    (proc / "cmdline").write_bytes(b"/python\0src/generality/scheduler.py\0--cards\01\0")
    monkeypatch.setattr(scheduler, "SRC", cwd / "src")
    assert scheduler.other_scheduler_pids(proc_root) == [123456]


def test_old_and_pinned_release_schedulers_are_both_detected(tmp_path, monkeypatch):
    proc_root = tmp_path / "proc"
    legacy = tmp_path / "legacy" / "src" / "generality" / "scheduler.py"
    release = tmp_path / "release" / "generality" / "scheduler_npu.py"
    for script in (legacy, release):
        script.parent.mkdir(parents=True)
        script.write_text("# fixture")
    monkeypatch.setattr(scheduler, "SRC", legacy.parents[1])
    monkeypatch.setenv("C2KV_GENERALITY_SOURCE", str(release.parents[1]))
    for pid, script in ((123456, legacy), (234567, release)):
        proc = proc_root / str(pid)
        proc.mkdir(parents=True)
        (proc / "cmdline").write_bytes(
            b"python\0" + script.as_posix().encode() + b"\0--cards\01\0")
        (proc / "cwd").mkdir()
    assert scheduler.other_scheduler_pids(proc_root) == [123456, 234567]


def test_detached_event_native_server_blocks_startup(tmp_path, monkeypatch):
    root = tmp_path / "root"
    proc_root = tmp_path / "proc"
    server = proc_root / "456"
    server.mkdir(parents=True)
    out = root / "results" / "closed_loop" / "appworld" / "cell" / "server"
    command = f"/python\0-m\0benchmarks.memory_runtime.event_native_server\0--out\0{out}\0"
    (server / "cmdline").write_bytes(command.encode())
    (server / "stat").write_text("456 (python) " + " ".join(["S", "1"] + ["0"] * 17 + ["123"]))
    monkeypatch.setattr(scheduler, "GENERATION_ROOT", root)
    monkeypatch.setattr(scheduler, "RESULTS", root / "results" / "closed_loop")
    assert scheduler.unmanaged_event_native_servers(proc_root) == [
        {"pid": 456, "ppid": 1, "start_ticks": 123,
         "out": str(out), "role": "supervisor"}]
    driver = proc_root / "123"
    driver.mkdir()
    (driver / "cmdline").write_bytes(b"/python\0/src/generality/c2kv_cell.py\0")
    (driver / "stat").write_text("123 (python) " + " ".join(["S", "1"] + ["0"] * 17 + ["55"]))
    (server / "stat").write_text("456 (python) " + " ".join(["S", "123"] + ["0"] * 17 + ["123"]))
    assert scheduler.unmanaged_event_native_servers(proc_root) == []


def test_launch_limit_persists_across_scheduler_instances(tmp_path):
    task = cell(tmp_path)
    for expected in range(1, scheduler.MAX_ATTEMPTS + 1):
        assert scheduler.reserve_attempt(task) == expected
        assert scheduler.attempt_count(task) == expected
    with pytest.raises(RuntimeError, match="exceeded scheduler attempt limit"):
        scheduler.reserve_attempt(task)
    receipt = scheduler.attempt_state_path(task)
    receipt.write_text('{"cell_id":"wrong","launch_attempts":0}')
    with pytest.raises(RuntimeError, match="Invalid scheduler attempt receipt"):
        scheduler.attempt_count(task)


def test_cell_driver_lock_and_port_are_isolated_across_slots(tmp_path, monkeypatch):
    task = cell(tmp_path, backend="fixture")
    monkeypatch.setattr(scheduler, "LOGS", tmp_path / "logs")
    captured = {}

    def popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(pid=123)

    monkeypatch.setattr(scheduler.subprocess, "Popen", popen)
    scheduler.launch_cell(task, card=3, slot=1)
    assert captured["command"][:2] == ["flock", "-n"]
    assert Path(captured["command"][2]).parent == tmp_path / "logs" / "driver_locks"
    assert Path(captured["command"][2]).name.startswith("cell-")
    assert captured["command"][3:6] == [
        "bash",
        str(Path(scheduler.__file__).resolve().parents[1] / "tools" / "launch_cpu_controller.sh"),
        scheduler.PY_SGL,
    ]
    assert "--proxy-port" in captured["command"]
    assert captured["command"][captured["command"].index("--proxy-port") + 1] == "52000"
    manifest_path = Path(captured["command"][captured["command"].index("--cell") + 1])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["scheduler_port_slot"] == 1
    assert captured["kwargs"]["start_new_session"] is True

    first_lock = captured["command"][2]
    scheduler.launch_cell(task, card=3, slot=0)
    second_path = Path(captured["command"][captured["command"].index("--cell") + 1])
    assert second_path != manifest_path
    assert captured["command"][2] == first_lock
    assert json.loads(manifest_path.read_text())["scheduler_port_slot"] == 1
    assert json.loads(second_path.read_text())["scheduler_port_slot"] == 0


@pytest.mark.parametrize("backend,condition,driver_name", [
    ("c2kv", "compression_full_budget", "c2kv_cell.py"),
    ("h2o", "compression_full_budget", "historykv_cell.py"),
    ("h2o", "tracer_history", "session_tracer_cell.py"),
])
def test_launch_manifest_and_command_bind_complete_sources(
        tmp_path, monkeypatch, backend, condition, driver_name):
    sources = {name: tmp_path / name for name in ("generality", "paper", "sglang")}
    for name, marker in (("generality", "generality/scheduler_npu.py"),
                         ("paper", "benchmarks/proxy.py"),
                         ("sglang", "python/sglang/__init__.py")):
        path = sources[name] / marker
        path.parent.mkdir(parents=True)
        path.write_text("# fixture")
    for name, source in sources.items():
        monkeypatch.setenv(f"C2KV_{'SGLANG' if name == 'sglang' else name.upper()}_SOURCE",
                           str(source))
    task = cell(tmp_path, backend=backend)
    task["condition"] = condition
    monkeypatch.setattr(scheduler, "LOGS", tmp_path / "logs")
    if condition == "tracer_history":
        monkeypatch.setattr(scheduler, "calibration_receipt", lambda cell:
                            (tmp_path / "threshold.json", {"threshold": 0.6}))
        monkeypatch.setattr(scheduler, "calibration_is_ready", lambda *args: True)
    captured = {}
    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda command, **kwargs:
                        captured.update(command=command, kwargs=kwargs) or
                        SimpleNamespace(pid=123))
    scheduler.launch_cell(task, card=3, slot=0)
    command, kwargs = captured["command"], captured["kwargs"]
    manifest = json.loads(Path(command[command.index("--cell") + 1]).read_text())
    assert manifest["source_checkouts"] == {name: str(path) for name, path in sources.items()}
    assert str(sources["generality"] / "generality" / driver_name) in command
    assert command[4] == str(sources["generality"] / "tools" / "launch_cpu_controller.sh")
    assert kwargs["env"]["C2KV_PAPER_SOURCE"] == str(sources["paper"])
    assert kwargs["env"]["C2KV_SGLANG_SOURCE"] == str(sources["sglang"])
    assert task.get("source_checkouts") is None


def test_invalid_explicit_source_fails_before_manifest(tmp_path, monkeypatch):
    monkeypatch.setenv("C2KV_PAPER_SOURCE", str(tmp_path / "missing"))
    task = cell(tmp_path)
    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda *a, **k:
                        pytest.fail("driver started"))
    with pytest.raises(RuntimeError, match="Invalid C2KV_PAPER_SOURCE"):
        scheduler.launch_cell(task, card=1, slot=0)
    assert not list(Path(task["cell_dir"]).glob("cell_launch_attempts/*.json"))


def test_pinned_engine_source_must_match_live_port(tmp_path, monkeypatch):
    checkout = tmp_path / "sglang"
    (checkout / "python" / "sglang").mkdir(parents=True)
    monkeypatch.setenv("C2KV_SGLANG_SOURCE", str(checkout))
    proc_root = tmp_path / "proc"
    proc = proc_root / "123"
    proc.mkdir(parents=True)
    (proc / "cmdline").write_bytes(
        b"python\0-m\0sglang.launch_server\0--port\0" + b"36203\0")
    environment = proc / "environ"
    environment.write_bytes(b"PYTHONPATH=/old/engine/python\0")
    assert not scheduler.engine_source_matches(3, proc_root)
    environment.write_bytes(f"PYTHONPATH={checkout / 'python'}\0".encode())
    assert scheduler.engine_source_matches(3, proc_root)
    (proc / "cmdline").write_bytes(
        b"python\0-m\0sglang.launch_server\0--port\0" + b"36204\0")
    assert not scheduler.engine_source_matches(3, proc_root)


def test_healthy_old_engine_is_left_for_manual_migration(tmp_path, monkeypatch, capsys):
    checkout = tmp_path / "sglang"
    (checkout / "python" / "sglang").mkdir(parents=True)
    monkeypatch.setenv("C2KV_SGLANG_SOURCE", str(checkout))
    monkeypatch.setattr(scheduler, "engine_source_matches", lambda card: False)
    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda *args, **kwargs:
                        pytest.fail("old engine must not be replaced here"))

    class Reply:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("urllib.request.build_opener", lambda *args:
                        SimpleNamespace(open=lambda *args, **kwargs: Reply()))
    scheduler.ensure_engines([3])
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "engine_source_migration_pending"
    assert event["expected"] == str(checkout)


def test_vacant_engine_uses_release_launcher_and_same_source_pin(tmp_path, monkeypatch):
    release = tmp_path / "release"
    engine = tmp_path / "engine"
    (release / "generality").mkdir(parents=True)
    (release / "generality" / "scheduler_npu.py").write_text("# fixture")
    (engine / "python" / "sglang").mkdir(parents=True)
    monkeypatch.setenv("C2KV_GENERALITY_SOURCE", str(release))
    monkeypatch.setenv("C2KV_SGLANG_SOURCE", str(engine))
    monkeypatch.setattr(scheduler, "LOGS", tmp_path / "logs")
    (tmp_path / "logs" / "engines").mkdir(parents=True)
    monkeypatch.setattr("urllib.request.build_opener", lambda *args:
                        SimpleNamespace(open=lambda *args, **kwargs:
                                        (_ for _ in ()).throw(OSError("vacant"))))
    captured = {}
    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda command, **kwargs:
                        captured.update(command=command, kwargs=kwargs))
    scheduler.ensure_engines([3])
    assert captured["command"][1] == str(release / "tools" / "launch_engine.sh")
    assert captured["kwargs"]["env"]["C2KV_SGLANG_SOURCE"] == str(engine)


def test_launch_boundary_refuses_a_held_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "BLOCKED_BACKENDS", {"h2o"})
    task = cell(tmp_path, backend="h2o")
    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda *a, **k:
                        pytest.fail("driver started"))
    with pytest.raises(RuntimeError, match="held by scheduler policy"):
        scheduler.launch_cell(task, card=1, slot=0)


@pytest.mark.parametrize("backend", ["c2kv", "h2o"])
def test_tracer_retrieval_is_bound_to_its_exclusive_engine_card(tmp_path, monkeypatch, backend):
    task = cell(tmp_path, backend=backend)
    task.update(condition="tracer_history", working_point="K0")
    original = dict(task)
    monkeypatch.setattr(scheduler, "LOGS", tmp_path / "logs")
    monkeypatch.setattr(scheduler, "calibration_receipt", lambda cell:
                        (tmp_path / "threshold.json", {"threshold": 0.6}))
    monkeypatch.setattr(scheduler, "calibration_is_ready", lambda *args: True)
    launches = []
    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda command, **kwargs:
                        launches.append((command, kwargs)) or SimpleNamespace(pid=123))
    scheduler.launch_cell(task, card=3, slot=0)
    command, kwargs = launches[0]
    manifest = json.loads(Path(command[command.index("--cell") + 1]).read_text())
    assert manifest["sglang_backend_url"].endswith(":36203")
    assert manifest["embedding_device"] == "npu:0"
    assert manifest["embedding_batch_size"] == 1
    assert kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"] == "3"
    assert task == original


def test_singleton_lock_refuses_second_scheduler(tmp_path, monkeypatch):
    state = {"held": False}

    def flock(fd, flags):
        if state["held"]:
            raise BlockingIOError("busy")
        state["held"] = True

    monkeypatch.setitem(sys.modules, "fcntl", SimpleNamespace(
        flock=flock, LOCK_EX=1, LOCK_NB=2))
    monkeypatch.setattr(scheduler, "LOGS", tmp_path)
    first = scheduler.acquire_scheduler_lock()
    try:
        with pytest.raises(RuntimeError, match="singleton lock"):
            scheduler.acquire_scheduler_lock()
    finally:
        first.close()


def test_orphan_card_respects_configured_capacity(tmp_path, monkeypatch):
    task = cell(tmp_path)
    task["backend"] = "fixture"
    old = tmp_path / "orphan"
    observed = []
    monkeypatch.setattr(scheduler, "ensure_engines", lambda cards: None)
    monkeypatch.setattr(scheduler, "engine_healthy", lambda card: True)
    monkeypatch.setattr(scheduler, "enumerate_cells", lambda: [task])
    monkeypatch.setattr(scheduler, "queue_cells", lambda cells, only_ready: list(cells))
    orphan = {**task, "cell_dir": str(old)}
    monkeypatch.setattr(scheduler, "live_driver_assignments", lambda: {1: {str(old): orphan}})
    monkeypatch.setattr(scheduler, "unmanaged_event_native_servers", lambda: [])
    monkeypatch.setattr(scheduler, "launch_cell", lambda c, card, slot:
                        observed.append((card, slot)) or SimpleNamespace(pid=42))
    def stop_loop(seconds):
        raise StopIteration
    monkeypatch.setattr(scheduler.time, "sleep", stop_loop)
    args = SimpleNamespace(cards=[1, 2], include_pending=False, max_cells=1,
                           max_drivers_per_card=1)
    with pytest.raises(StopIteration):
        scheduler.run_scheduler(args)
    assert observed == [(2, 0)]


def test_two_drivers_can_share_a_card_without_repeating_a_cell(tmp_path, monkeypatch):
    first = cell(tmp_path, backend="fixture")
    first["condition"] = "compression_full_budget"
    first["task_ids"] = ["first-task"]
    second = dict(first, cell_id="second", cell_dir=str(tmp_path / "second"))
    second["task_ids"] = ["second-task"]
    Path(second["cell_dir"]).mkdir()
    observed = []
    monkeypatch.setattr(scheduler, "ensure_engines", lambda cards: None)
    monkeypatch.setattr(scheduler, "engine_healthy", lambda card: True)
    monkeypatch.setattr(scheduler, "enumerate_cells", lambda: [first, first, second])
    monkeypatch.setattr(scheduler, "queue_cells", lambda cells, only_ready: list(cells))
    monkeypatch.setattr(scheduler, "live_driver_assignments", lambda: {})
    monkeypatch.setattr(scheduler, "unmanaged_event_native_servers", lambda: [])
    monkeypatch.setattr(scheduler, "launch_cell", lambda c, card, slot:
                        observed.append((c["cell_id"], card, slot)) or
                        SimpleNamespace(pid=len(observed)))
    def stop_loop(seconds):
        raise StopIteration
    monkeypatch.setattr(scheduler.time, "sleep", stop_loop)
    args = SimpleNamespace(cards=[1], include_pending=False, max_cells=3,
                           max_drivers_per_card=2)
    with pytest.raises(StopIteration):
        scheduler.run_scheduler(args)
    assert observed == [(first["cell_id"], 1, 0), (second["cell_id"], 1, 1)]


def test_max_cells_remains_bounded_after_queue_rederivation(tmp_path, monkeypatch):
    first = cell(tmp_path, backend="fixture")
    second = dict(first, cell_id="second", cell_dir=str(tmp_path / "second"))
    Path(second["cell_dir"]).mkdir()
    state = {"first_done": False, "launched": []}
    monkeypatch.setattr(scheduler, "unmanaged_event_native_servers", lambda: [])
    monkeypatch.setattr(scheduler, "live_driver_assignments", lambda: {})
    monkeypatch.setattr(scheduler, "ensure_engines", lambda cards: None)
    monkeypatch.setattr(scheduler, "engine_healthy", lambda card: True)
    monkeypatch.setattr(scheduler, "enumerate_cells", lambda: [first, second])
    monkeypatch.setattr(scheduler, "queue_cells", lambda cells, only_ready: list(cells))
    monkeypatch.setattr(scheduler, "cell_done", lambda task:
                        task["cell_id"] == first["cell_id"] and state["first_done"])
    monkeypatch.setattr(scheduler, "attempt_count", lambda task: 0)
    monkeypatch.setattr(scheduler, "reserve_attempt", lambda task: 1)
    monkeypatch.setattr(scheduler.time, "sleep", lambda seconds: None)

    def launch(task, card, slot):
        state["launched"].append(task["cell_id"])
        def poll():
            state["first_done"] = True
            return 0
        return SimpleNamespace(pid=42, poll=poll, returncode=0)

    monkeypatch.setattr(scheduler, "launch_cell", launch)
    args = SimpleNamespace(cards=[1], include_pending=False, max_cells=1,
                           max_drivers_per_card=1)
    assert scheduler.run_scheduler(args) == 0
    assert state["launched"] == [first["cell_id"]]


def test_mixed_driver_protocols_and_overlapping_tasks_prevent_sharing(tmp_path):
    first = cell(tmp_path, backend="fixture")
    first["task_ids"] = ["task"]
    second = dict(first, cell_id="second", cell_dir=str(tmp_path / "second"))
    assert not scheduler.may_share_engine(second, 1, {}, {1: {first["cell_dir"]: first}})
    first["condition"] = second["condition"] = "tracer_history"
    assert not scheduler.may_share_engine(second, 1, {}, {1: {first["cell_dir"]: first}})
    second["task_ids"] = ["other-task"]
    assert not scheduler.may_share_engine(second, 1, {}, {1: {first["cell_dir"]: first}})
    first["condition"] = "compression_full_budget"
    assert not scheduler.may_share_engine(second, 1, {}, {1: {first["cell_dir"]: first}})
    second["condition"] = "compression_full_budget"
    assert scheduler.may_share_engine(second, 1, {}, {1: {first["cell_dir"]: first}})


def test_native_compression_shares_only_disjoint_tasks_without_recovery(tmp_path):
    first = cell(tmp_path, backend="c2kv")
    first["task_ids"] = ["task"]
    second = dict(first, cell_id="second", cell_dir=str(tmp_path / "second"),
                  benchmark_key="bfcl_base", task_ids=["other-task"])
    occupied = {1: {first["cell_dir"]: first}}
    assert scheduler.may_share_engine(second, 1, {}, occupied)
    second.update(benchmark_key="appworld", task_ids=["task"])
    assert not scheduler.may_share_engine(second, 1, {}, occupied)
    second["task_ids"] = ["other-task"]
    second["condition"] = "tracer_history"
    assert not scheduler.may_share_engine(second, 1, {}, occupied)
    second.update(backend="h2o", condition="compression_full_budget")
    assert not scheduler.may_share_engine(second, 1, {}, occupied)


def test_unhealthy_engine_is_not_restarted_under_a_live_driver(tmp_path, monkeypatch):
    orphan = cell(tmp_path, backend="fixture")
    launches = []
    monkeypatch.setattr(scheduler, "live_driver_assignments", lambda:
                        {1: {orphan["cell_dir"]: orphan}})
    monkeypatch.setattr(scheduler, "unmanaged_event_native_servers", lambda: [])
    monkeypatch.setattr(scheduler, "ensure_engines", lambda cards: launches.append(cards))
    monkeypatch.setattr(scheduler, "engine_healthy", lambda card: False)
    monkeypatch.setattr(scheduler, "enumerate_cells", lambda: [])
    monkeypatch.setattr(scheduler.time, "monotonic", iter([0, 121]).__next__)
    def stop_loop(seconds):
        raise StopIteration
    monkeypatch.setattr(scheduler.time, "sleep", stop_loop)
    args = SimpleNamespace(cards=[1], include_pending=False, max_cells=None,
                           max_drivers_per_card=2)
    with pytest.raises(StopIteration):
        scheduler.run_scheduler(args)
    assert launches == []
