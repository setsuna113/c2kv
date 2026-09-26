"""CPU-only checks for concurrent serving runs of prepared paper cells."""

from __future__ import annotations

import json
import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from benchmarks.paper import runner, serving
from benchmarks.paper.racer_matrix import parse_racer_policies, with_racer_methods
from benchmarks.paper.candidate_matrix import with_candidate_methods
from benchmarks.paper.upstream_liveness import UpstreamUnavailable

TASKS = [f"multi_turn_long_context_{index}" for index in range(5)]
PROFILE = Path("out/deployment_profile.json")


def base_config():
    config = json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config["history_kv_budget_tokens"] = 768
    return config


def racer_config():
    return with_racer_methods(base_config(), ("snapkv",), parse_racer_policies("pending_verified"), 768)


def cell(config, cell_id):
    return next(row for row in runner.cells(config) if row["cell_id"] == cell_id)


def value(command, flag):
    return command[command.index(flag) + 1]


class FakeProcess:
    def __init__(self, polls_until_exit, returncode=0):
        self.remaining = polls_until_exit
        self.returncode = returncode
        self.pid = None
        self.terminated = False

    def poll(self):
        if self.terminated:
            return -15
        if self.remaining > 0:
            self.remaining -= 1
            return None
        return self.returncode

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return self.poll()


def test_engine_slot_cap_scales_and_paper_engine_command_stays_single_flight():
    config = base_config()
    full = cell(config, "bfcl_long_context__full")
    paper = runner.server_command(config, Path("sglang"), full["arm"], full["benchmark"])
    scaled = serving.serving_server_command(config, Path("sglang"), full, 4)
    index = paper.index("--max-running-requests") + 1
    assert paper[index] == "1" and scaled[index] == "4"
    assert scaled[:index] + scaled[index + 1:] == paper[:index] + paper[index + 1:]
    assert "--disable-radix-cache" not in scaled  # Full keeps the configured prefix cache


def test_engine_slot_cap_override_is_independent_of_worker_count():
    config = base_config()
    full = cell(config, "bfcl_long_context__full")
    fixed = dict(config, serving_engine_max_running_requests=4)
    for workers in (4, 8):
        command = serving.serving_server_command(fixed, Path("sglang"), full, workers)
        assert value(command, "--max-running-requests") == "4"
    assert value(runner.server_command(fixed, Path("sglang"), full["arm"],
                                       full["benchmark"]), "--max-running-requests") == "1"


@pytest.mark.parametrize("invalid", [None, 0, -1, True, 4.0, "4"])
def test_engine_slot_cap_override_rejects_nonpositive_or_noninteger_values(invalid):
    config = dict(base_config(), serving_engine_max_running_requests=invalid)
    full = cell(config, "bfcl_long_context__full")
    with pytest.raises(ValueError, match="serving_engine_max_running_requests"):
        serving.serving_server_command(config, Path("sglang"), full, 4)


@pytest.mark.parametrize("cell_id", ["bfcl_long_context__full",
                                      "bfcl_long_context__racer_c2kv_pending_verified_b768"])
def test_overlap_schedule_is_serving_only_and_default_command_is_unchanged(cell_id):
    config = with_racer_methods(base_config(), ("c2kv",),
                                parse_racer_policies("pending_verified"), 768)
    selected = cell(config, cell_id)
    source = Path("sglang")
    default = serving.serving_server_command(config, source, selected, 4)
    disabled = dict(config, serving_overlap_schedule=False)
    enabled = dict(config, serving_overlap_schedule=True)
    assert serving.serving_server_command(disabled, source, selected, 4) == default
    assert default.count("--disable-overlap-schedule") == 1
    assert serving.serving_server_command(enabled, source, selected, 4) == [
        part for part in default if part != "--disable-overlap-schedule"]
    paper = runner.server_command(config, source, selected["arm"], selected["benchmark"])
    assert runner.server_command(enabled, source, selected["arm"], selected["benchmark"]) == paper


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", [], {}])
def test_overlap_schedule_rejects_nonboolean_values(invalid):
    config = dict(base_config(), serving_overlap_schedule=invalid)
    full = cell(config, "bfcl_long_context__full")
    with pytest.raises(ValueError, match="serving_overlap_schedule must be a boolean"):
        serving.serving_server_command(config, Path("sglang"), full, 4)


@pytest.mark.parametrize("policy", ["lru", "lfu", "slru", "priority"])
def test_radix_eviction_policy_is_explicit_and_serving_only(policy):
    config = base_config()
    selected = cell(config, "bfcl_long_context__full")
    source = Path("sglang")
    default = serving.serving_server_command(config, source, selected, 4)
    enabled = dict(config, serving_radix_eviction_policy=policy)
    assert "--radix-eviction-policy" not in default
    assert serving.serving_server_command(enabled, source, selected, 4) == [
        *default, "--radix-eviction-policy", policy]
    if policy == "priority":
        assert "--disable-overlap-schedule" in default
        assert "--disable-overlap-schedule" in serving.serving_server_command(
            enabled, source, selected, 4)
    assert runner.server_command(enabled, source, selected["arm"],
                                 selected["benchmark"]) == runner.server_command(
                                     config, source, selected["arm"], selected["benchmark"])


def test_radix_eviction_policy_does_not_enable_disabled_cache_or_overlap():
    config = racer_config()
    selected = cell(config, "bfcl_long_context__racer_snapkv_pending_verified_b768")
    source = Path("sglang")
    default = serving.serving_server_command(config, source, selected, 4)
    command = serving.serving_server_command(
        dict(config, serving_radix_eviction_policy="slru"), source, selected, 4)
    assert "--disable-radix-cache" in default and "--disable-radix-cache" in command
    assert "--disable-overlap-schedule" in default and "--disable-overlap-schedule" in command
    assert command == [*default, "--radix-eviction-policy", "slru"]


@pytest.mark.parametrize("invalid", [None, True, 1, 1.0, [], {}, "", "fifo", "LRU"])
def test_radix_eviction_policy_rejects_invalid_types_and_values(invalid):
    config = dict(base_config(), serving_radix_eviction_policy=invalid)
    selected = cell(config, "bfcl_long_context__full")
    with pytest.raises(ValueError, match="serving_radix_eviction_policy"):
        serving.serving_server_command(config, Path("sglang"), selected, 4)


def test_full_task_command_is_the_single_task_subset_on_its_lane_with_a_local_reset():
    config = base_config()
    full = cell(config, "bfcl_long_context__full")
    task = TASKS[3]
    command = serving.task_command(config, full, task, Path("out/t"), PROFILE, lane=2)
    lane_config = dict(config, proxy_port=config["proxy_port"] + 2)
    expected = runner.run_command(lane_config, dict(full, task_ids=[task]), Path("out/t"), PROFILE)
    assert command == expected + ["--shared-engine"]
    assert value(command, "--proxy-port") == str(config["proxy_port"] + 2)
    assert value(command, "--run-ids") == task
    assert value(command, "--num-workers") == "1"


def test_racer_task_command_is_the_native_command_for_one_task_on_its_lane():
    config = racer_config()
    racer = cell(config, "bfcl_long_context__racer_snapkv_pending_verified_b768")
    command = serving.task_command(config, racer, TASKS[0], Path("out/t"), PROFILE, lane=1)
    lane_config = dict(config, proxy_port=config["proxy_port"] + 1)
    assert command == runner.run_command(lane_config, racer, Path("out/t"), PROFILE) + [
        "--task-ids", TASKS[0]]
    assert value(command, "--proxy-port") == str(config["proxy_port"] + 1)
    assert "--shared-engine" not in command


def run_fake_pool(tmp_path, workers, durations, monitor=lambda: None):
    started, live, peak = [], set(), [0]
    processes = {}

    def popen(command, **kwargs):
        task, lane = command
        assert kwargs["start_new_session"] in (True, False)
        live.add(task)
        peak[0] = max(peak[0], len(live))
        started.append((task, lane))
        process = FakeProcess(durations[task], returncode=1 if task == TASKS[4] else 0)
        original_poll = process.poll

        def poll():
            code = original_poll()
            if code is not None:
                live.discard(task)
            return code

        process.poll = poll
        processes[task] = process
        return process

    rows = serving.run_pool(
        TASKS, lambda task, lane, directory: [task, lane], lambda task: tmp_path / task,
        workers=workers, env={}, cwd=None, monitor=monitor, poll_interval=0, popen=popen)
    return rows, started, peak[0], processes


def test_pool_caps_concurrency_reuses_freed_lanes_and_keeps_task_order(tmp_path, monkeypatch):
    stopped = []
    monkeypatch.setattr(serving, "stop_owned_group", stopped.append)
    durations = dict(zip(TASKS, (3, 0, 0, 2, 1)))
    rows, started, peak, _ = run_fake_pool(tmp_path, 2, durations)
    assert peak == 2
    assert [task for task, _ in started] == TASKS  # official order, dynamic lanes
    assert started[:2] == [(TASKS[0], 0), (TASKS[1], 1)]
    assert {lane for _, lane in started} == {0, 1}
    assert sorted(row["task_index"] for row in rows) == list(range(len(TASKS)))
    assert {row["task_id"]: row["returncode"] for row in rows}[TASKS[4]] == 1
    assert len(stopped) == len(TASKS)  # every finished child's session is reaped
    assert all((tmp_path / task / "child.log").is_file() for task in TASKS)


def test_pool_stops_every_running_child_when_the_engine_disappears(tmp_path, monkeypatch):
    stopped = []
    monkeypatch.setattr(serving, "stop_owned_group", lambda process: stopped.append(process))
    calls = [0]

    def monitor():
        calls[0] += 1
        if calls[0] == 2:
            raise UpstreamUnavailable("engine exited")

    durations = {task: 10 for task in TASKS}
    with pytest.raises(UpstreamUnavailable):
        run_fake_pool(tmp_path, 3, durations, monitor=monitor)
    assert len(stopped) == 3
    assert not (tmp_path / TASKS[3]).exists()


def test_serve_cell_runs_every_task_and_writes_the_throughput_manifest(tmp_path, monkeypatch):
    config = base_config()
    output = tmp_path / "paper"
    plan, profile = runner.prepare(config, output, tmp_path / "sglang")
    full = next(row for row in plan if row["cell_id"] == "bfcl_long_context__full")
    monkeypatch.setattr(serving, "wait_server", lambda process, port: None)
    monkeypatch.setattr(serving, "cleanup_cell_processes", lambda proxy, server: None)
    monkeypatch.setattr(serving, "UpstreamLiveness", lambda url, process: (lambda: None))
    commands = []

    def popen(command, **kwargs):
        commands.append((command, kwargs))
        if "--out" in command:
            out = Path(value(command, "--out"))
            task = value(command, "--run-ids")
            (out / "summary_full.json").write_text(json.dumps({
                "n_scored": 1, "semantic_score": float(TASKS.index(task) % 2)}))
        return FakeProcess(1)

    offset = 20000
    manifest = serving.serve_cell(config, full, output, tmp_path / "sglang", profile,
                                  workers=2, tasks=TASKS, port_offset=offset,
                                  poll_interval=0, popen=popen)
    server_command, server_kwargs = commands[0]
    assert value(server_command, "--max-running-requests") == "2"
    assert value(server_command, "--port") == str(config["server_port"] + offset)
    assert server_kwargs["env"]["C2KV_PAPER_TELEMETRY_LOG"].endswith("server_telemetry.jsonl")
    assert server_kwargs["env"]["C2KV_PAPER_CONCURRENT"] == "1"
    ports = {value(command, "--proxy-port") for command, _ in commands[1:]}
    assert ports == {str(config["proxy_port"] + offset + lane) for lane in (0, 1)}
    assert all("--shared-engine" in command for command, _ in commands[1:])
    directory = output / "serving" / full["cell_id"] / "workers_2"
    assert json.loads((directory / "serving.json").read_text()) == manifest
    assert manifest["n_tasks"] == manifest["n_scored"] == len(TASKS)
    assert manifest["successful_tasks"] == 2
    assert manifest["engine_max_running_requests"] == 2
    assert manifest["measurement_scope"] == "concurrent_serving"
    assert manifest["overlap_schedule"] is False
    started = json.loads((directory / "started.json").read_text())
    assert started["overlap_schedule"] is False
    assert "--disable-overlap-schedule" in started["server_command"]
    assert manifest["status"] == "completed"
    assert manifest["engine_telemetry"]["status"] == "unavailable"
    assert manifest["runtime_scope"] == "first_task_process_start_to_last_observed_process_exit"
    assert [row["task_id"] for row in manifest["tasks"]] == TASKS
    with pytest.raises(FileExistsError):
        serving.serve_cell(config, full, output, tmp_path / "sglang", profile,
                           workers=2, tasks=TASKS, port_offset=offset,
                           poll_interval=0, popen=popen)


@pytest.mark.parametrize("cell_id,tasks,message", [
    ("tau2__full", TASKS, "Serving runs support"),
    ("bfcl_long_context__full", [TASKS[0], TASKS[0]], "unique"),
    ("bfcl_long_context__full", ["../escape"], "cannot name"),
])
def test_serve_cell_rejects_unsupported_scope_before_starting_anything(
        tmp_path, cell_id, tasks, message):
    config = base_config()
    def popen(*args, **kwargs):
        raise AssertionError("nothing may start")
    with pytest.raises(ValueError, match=message):
        serving.serve_cell(config, cell(config, cell_id), tmp_path, tmp_path / "sglang",
                           PROFILE, workers=2, tasks=tasks, popen=popen)
    assert not (tmp_path / "serving").exists()


def test_runner_serve_prepares_the_matrix_and_serves_the_one_selected_cell(tmp_path, monkeypatch):
    from benchmarks.paper import c1

    calls = {}

    def fake_tasks(config, benchmark, requested=None):
        calls["tasks"] = (benchmark, requested)
        return list(requested)

    def fake_serve(config, cell, output, source, profile, *, workers, tasks, port_offset):
        calls["serve"] = (cell["cell_id"], workers, tasks, port_offset, profile)
        return {key: None for key in ("cell_id", "workers", "n_tasks", "n_scored",
                                      "total_runtime_seconds", "tasks_per_hour",
                                      "successful_tasks_per_hour", "status")}

    monkeypatch.setattr(c1, "selected_tasks", fake_tasks)
    monkeypatch.setattr(serving, "serve_cell", fake_serve)
    output = tmp_path / "paper"
    runner.main(["serve", "--output", str(output), "--sglang-source", str(tmp_path / "sglang"),
                 "--history-kv-budget-tokens", "768",
                 "--cells", "bfcl_long_context__full", "--workers", "3",
                 "--serve-tasks", ",".join(TASKS[:2]), "--port-offset", "7"])
    assert calls["tasks"] == ("bfcl_long_context", TASKS[:2])
    assert calls["serve"] == ("bfcl_long_context__full", 3, TASKS[:2], 7,
                              output / "deployment_profile.json")
    assert (output / "commands.json").is_file()


@pytest.mark.parametrize("argv", [
    ["run", "--workers", "2"],
    ["run", "--serve-tasks", TASKS[0]],
    ["serve", "--cells", "bfcl_long_context__full"],
    ["serve", "--cells", "bfcl_long_context__full", "--workers", "0"],
    ["serve", "--cells", "a,b", "--workers", "2"],
    ["serve", "--cells", "bfcl_long_context__full", "--workers", "2", "--stage", "common_prefix"],
])
def test_runner_cli_keeps_serving_options_out_of_the_paper_actions(argv):
    with pytest.raises(SystemExit):
        runner.main(argv + (["--sglang-source", "sglang"] if argv[0] == "serve" else []))


def test_real_children_overlap_and_reuse_lane_ports(tmp_path):
    # Both first-wave children must own distinct listening sockets before the
    # parent releases them. A serial implementation cannot pass this barrier.
    holders = [socket.socket() for _ in range(2)]
    try:
        for holder in holders:
            holder.bind(("127.0.0.1", 0))
        ports = [holder.getsockname()[1] for holder in holders]
    finally:
        for holder in holders:
            holder.close()
    child = tmp_path / "child.py"
    child.write_text('''import json, socket, sys, time
from pathlib import Path
root, task, port = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
with socket.socket() as listener:
    listener.bind(("127.0.0.1", port))
    listener.listen()
    start = time.monotonic_ns()
    (root / (task + ".ready")).touch()
    deadline = time.monotonic() + 10
    while not (root / "release").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("parent did not release the concurrent children")
        time.sleep(0.01)
    time.sleep(0.05)  # Keep the overlap visible on coarse Windows clocks.
    (root / (task + ".json")).write_text(json.dumps({"start": start, "end": time.monotonic_ns()}))
''', encoding="utf-8")
    processes = []
    deadline = time.monotonic() + 15

    def popen(*args, **kwargs):
        process = subprocess.Popen(*args, **kwargs)
        processes.append(process)
        return process

    def monitor():
        assert time.monotonic() < deadline, "real child pool did not complete"
        if all((tmp_path / f"{task}.ready").exists() for task in ("one", "two")):
            (tmp_path / "release").touch()

    rows = serving.run_pool(
        ["one", "two", "three"],
        lambda task, lane, directory: [sys.executable, str(child), str(tmp_path), task, str(ports[lane])],
        lambda task: tmp_path / "tasks" / task,
        workers=2, env=os.environ.copy(), cwd=None, monitor=monitor,
        poll_interval=0.01, popen=popen)
    assert len(rows) == 3 and all(row["returncode"] == 0 for row in rows)
    assert all(process.poll() == 0 for process in processes)
    intervals = {task: json.loads((tmp_path / f"{task}.json").read_text())
                 for task in ("one", "two", "three")}
    assert max(intervals[task]["start"] for task in ("one", "two")) < min(
        intervals[task]["end"] for task in ("one", "two"))
    by_task = {row["task_id"]: row for row in rows}
    predecessor = next(task for task in ("one", "two")
                       if by_task[task]["lane"] == by_task["three"]["lane"])
    assert intervals[predecessor]["end"] <= intervals["three"]["start"]
    assert all((Path(row["output"]) / "process.json").is_file() for row in rows)


def test_real_children_are_stopped_on_monitor_failure(tmp_path):
    processes = []

    def popen(*args, **kwargs):
        process = subprocess.Popen(*args, **kwargs)
        processes.append(process)
        return process

    def fail():
        raise UpstreamUnavailable("test engine disappeared")

    with pytest.raises(UpstreamUnavailable):
        serving.run_pool(
            ["one", "two", "never"],
            lambda task, lane, directory: [sys.executable, "-c", "import time; time.sleep(30)"],
            lambda task: tmp_path / task,
            workers=2, env=os.environ.copy(), cwd=None, monitor=fail,
            poll_interval=0.01, popen=popen)
    assert len(processes) == 2
    assert all(process.poll() is not None for process in processes)
    assert not (tmp_path / "never").exists()


def test_aborted_serving_writes_receipt_without_a_throughput(tmp_path, monkeypatch):
    config = base_config()
    full = cell(config, "bfcl_long_context__full")
    monkeypatch.setattr(serving, "cleanup_cell_processes", lambda proxy, server: None)

    def fail(*args):
        raise UpstreamUnavailable("engine failed before readiness")

    monkeypatch.setattr(serving, "wait_server", fail)
    with pytest.raises(UpstreamUnavailable):
        serving.serve_cell(config, full, tmp_path, tmp_path / "sglang", PROFILE,
                           workers=2, tasks=TASKS, port_offset=21000,
                           popen=lambda *args, **kwargs: FakeProcess(0))
    manifest = json.loads((tmp_path / "serving" / full["cell_id"] / "workers_2" / "serving.json").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["tasks_per_hour"] is None
    assert manifest["successful_tasks_per_hour"] is None
    assert manifest["n_scored"] == 0 and manifest["n_unscored"] == len(TASKS)


def test_summary_uses_monotonic_makespan_and_preserves_invalid_score(tmp_path):
    rows = []
    for index, score in enumerate((1, float("nan"))):
        directory = tmp_path / str(index)
        directory.mkdir()
        (directory / "summary_full.json").write_text(json.dumps({"n_scored": 1, "semantic_score": score}))
        rows.append({"task_index": index, "task_id": str(index), "returncode": 0,
                     "output": str(directory), "start_monotonic_ns": index * 1_000_000_000,
                     "end_monotonic_ns": (index + 2) * 1_000_000_000,
                     "start_unix_ns": 100, "end_unix_ns": 0})
    result = serving.summarize(cell(base_config(), "bfcl_long_context__full"), 2, 2,
                               [9000, 9001], ["0", "1"], rows)
    assert result["total_runtime_seconds"] == 3
    assert result["successful_tasks_per_hour"] == 1200
    assert result["status"] == "completed_with_failures" and result["n_unscored"] == 1
    assert "score_error" in result["tasks"][1]


def test_pool_preserves_completion_before_engine_failure(tmp_path):
    def fail():
        raise UpstreamUnavailable("engine exited beside a completed task")

    with pytest.raises(UpstreamUnavailable):
        serving.run_pool(
            ["done", "running"], lambda task, lane, directory: [task],
            lambda task: tmp_path / task, workers=2, env={}, cwd=None,
            monitor=fail, poll_interval=0,
            popen=lambda command, **kwargs: FakeProcess(0 if command == ["done"] else 100))
    receipt = json.loads((tmp_path / "done" / "process.json").read_text())
    assert receipt["returncode"] == 0 and receipt["task_id"] == "done"


def test_serve_resolves_native_paths_before_changing_child_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = racer_config()
    racer = cell(config, "bfcl_long_context__racer_snapkv_pending_verified_b768")
    monkeypatch.setattr(serving, "wait_server", lambda *args: None)
    monkeypatch.setattr(serving, "cleanup_cell_processes", lambda *args: None)
    monkeypatch.setattr(serving, "UpstreamLiveness", lambda *args, **kwargs: lambda: None)
    commands = []

    def popen(command, **kwargs):
        commands.append((command, kwargs))
        return FakeProcess(0)

    serving.serve_cell(config, racer, Path("relative"), Path("engine"), Path("profile.json"),
                       workers=1, tasks=TASKS[:1], port_offset=21000,
                       poll_interval=0, popen=popen)
    child, kwargs = commands[1]
    assert Path(value(child, "--out")) == tmp_path / "relative" / "serving" / racer["cell_id"] / "workers_1" / "tasks" / TASKS[0]
    assert kwargs["cwd"] == runner.ROOT.parent
    assert Path(kwargs["env"]["C2KV_PAPER_TELEMETRY_LOG"]).is_absolute()


def test_serve_requires_explicit_engine_checkout():
    with pytest.raises(SystemExit):
        runner.main(["serve", "--cells", "bfcl_long_context__full", "--workers", "2"])


def test_native_feature_flags_enable_only_the_native_cell_and_preserve_full():
    config = with_racer_methods(base_config(), ("c2kv",), parse_racer_policies("pending_verified"), 768)
    native = cell(config, "bfcl_long_context__racer_c2kv_pending_verified_b768")
    full = cell(config, "bfcl_long_context__full")
    enabled = dict(config, serving_native_raw_prefix_cache=True, serving_background_extras=True,
                   serving_bulk_cache_lookup=True, serving_cross_turn_prewarm=True)
    before = serving.serving_server_command(config, Path("sglang"), native, 2)
    after = serving.serving_server_command(enabled, Path("sglang"), native, 2)
    assert "--disable-radix-cache" in before
    assert after == [part for part in before if part != "--disable-radix-cache"]
    assert serving.native_serving_features(enabled, native) == {
        "raw_prefix_cache": "raw-prefix-v1",
        "background_extras": "selected-first-response-barrier-v1",
        "bulk_cache_lookup": "bulk-cache-lookup-v1",
        "cross_turn_prewarm": "cross-turn-prewarm-v1",
        "async_compression": None,
    }
    assert serving.serving_server_command(enabled, Path("sglang"), full, 2) == serving.serving_server_command(config, Path("sglang"), full, 2)
    assert all(value is None for value in serving.native_serving_features(enabled, full).values())
    # A prepared normal paper command remains unchanged; only serve opts in.
    assert runner.server_command(enabled, Path("sglang"), native["arm"], native["benchmark"]) == runner.server_command(config, Path("sglang"), native["arm"], native["benchmark"])


def test_native_features_reject_persistent_raw_kv_before_launch():
    config = dict(racer_config(), serving_native_raw_prefix_cache=True)
    with pytest.raises(ValueError, match="C2KV native"):
        serving.native_serving_features(config, cell(config, "bfcl_long_context__racer_snapkv_pending_verified_b768"))


def test_feature_capability_check_rejects_older_engine(monkeypatch):
    requested = {"raw_prefix_cache": "raw-prefix-v1", "background_extras": None}
    monkeypatch.setattr(serving.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"{}"))
    with pytest.raises(RuntimeError, match="does not confirm requested raw_prefix_cache"):
        serving.require_native_serving_features(34000, requested)
    reported = {"c2kv_native_packed": {"serving_features": requested}}
    monkeypatch.setattr(serving.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(json.dumps(reported).encode()))
    assert serving.require_native_serving_features(34000, requested) == requested


def test_prepare_freezes_serving_features_and_regular_run_rejects_them(tmp_path):
    output = tmp_path / "prepared"
    runner.main(["prepare", "--output", str(output), "--native-raw-prefix-cache", "--background-extras",
                 "--bulk-cache-lookup", "--cross-turn-prewarm", "--persistent-runtime",
                 "--incremental-tokenization", "--async-compression", "--history-kv-budget-tokens", "768"])
    resolved = json.loads((output / "config.resolved.json").read_text())
    assert resolved["history_kv_budget_tokens"] == 768
    assert resolved["serving_native_raw_prefix_cache"] is True
    assert resolved["serving_background_extras"] is True
    assert resolved["serving_bulk_cache_lookup"] is True
    assert resolved["serving_cross_turn_prewarm"] is True
    assert resolved["serving_persistent_runtime"] is True
    assert resolved["serving_incremental_tokenization"] is True
    assert resolved["serving_async_compression"] is True
    with pytest.raises(SystemExit):
        runner.main(["run", "--config", str(output / "config.resolved.json")])


def test_normal_paper_environment_cannot_inherit_serving_optimizations(monkeypatch):
    for variable in ("C2KV_NATIVE_RAW_PREFIX_CACHE", "C2KV_NATIVE_BACKGROUND_EXTRAS",
                     "C2KV_NATIVE_BULK_CACHE_LOOKUP", "C2KV_NATIVE_CROSS_TURN_PREWARM",
                     "C2KV_INCREMENTAL_TOKENIZATION", "C2KV_NATIVE_ASYNC_COMPRESSION"):
        monkeypatch.setenv(variable, "1")
    env = runner.paper_env(base_config(), Path("engine"))
    assert env["C2KV_NATIVE_RAW_PREFIX_CACHE"] == env["C2KV_NATIVE_BACKGROUND_EXTRAS"] == "0"
    assert env["C2KV_NATIVE_BULK_CACHE_LOOKUP"] == env["C2KV_NATIVE_CROSS_TURN_PREWARM"] == "0"
    assert env["C2KV_INCREMENTAL_TOKENIZATION"] == "0"
    assert env["C2KV_NATIVE_ASYNC_COMPRESSION"] == "0"


def test_persistent_runtime_accepts_only_actual_c1_v2_b256_cell(tmp_path):
    arm = "c2kv_c1_v2_verified_r8"
    config = runner.with_native_history_budget(
        with_candidate_methods(base_config(), ("c1_v2_verified",)), arm, 256)
    plan, _ = runner.prepare(config, tmp_path / "paper", tmp_path / "sglang")
    by_id = {row["cell_id"]: row for row in plan}
    selected = by_id[f"bfcl_base__{arm}_b256"]
    enabled = dict(config, serving_persistent_runtime=True,
                   serving_incremental_tokenization=True)
    assert serving.persistent_runtime_enabled(enabled, selected)
    command = serving.persistent_lane_command(
        enabled, selected, TASKS[:2], Path("out/lane"), PROFILE, 1,
        Path("out/tasks"))
    assert value(command, "--arm") == arm
    assert value(command, "--history-budget-tokens") == "256"
    assert value(command, "--task-ids") == ",".join(TASKS[:2])
    assert value(command, "--serving-task-output") == str(Path("out/tasks"))
    assert "--persistent-runtime" in command
    assert "--dynamic-serving-lane" not in command
    dynamic = serving.persistent_lane_command(
        enabled, selected, TASKS[:2], Path("out/lane"), PROFILE, 1,
        Path("out/tasks"), dynamic=True)
    assert value(dynamic, "--dynamic-serving-lane") == "1"
    assert serving.dynamic_persistent_runtime_enabled(
        dict(enabled, serving_dynamic_persistent_runtime=True), True)
    with pytest.raises(ValueError, match="requires persistent"):
        serving.dynamic_persistent_runtime_enabled(
            dict(enabled, serving_dynamic_persistent_runtime=True), False)
    for wrong in (by_id[f"bfcl_base__{arm}"], by_id["bfcl_base__full"]):
        with pytest.raises(ValueError, match="C1 v2 verified B256"):
            serving.persistent_runtime_enabled(enabled, wrong)


def test_real_persistent_pool_reuses_lane_process_and_records_cold_span(tmp_path):
    child = tmp_path / "lane.py"
    child.write_text('''import json, os, sys, time
from pathlib import Path
root, tasks = Path(sys.argv[1]), sys.argv[2].split(",")
time.sleep(0.05)
for task in tasks:
    out = root / task
    out.mkdir(parents=True)
    start = time.monotonic_ns()
    (out / "summary_test.json").write_text(json.dumps({"n_scored": 1, "semantic_score": 1}))
    end = time.monotonic_ns()
    (out / "process.json").write_text(json.dumps({
        "task_id": task, "returncode": 0, "start_monotonic_ns": start,
        "end_monotonic_ns": end, "start_unix_ns": time.time_ns(),
        "end_unix_ns": time.time_ns(), "output": str(out), "pid": os.getpid()}))
''', encoding="utf-8")
    tasks = ["one", "two", "three"]
    directory = tmp_path / "serving"
    rows, cold = serving.run_persistent_pool(
        tasks,
        lambda lane_tasks, lane, lane_dir: [
            sys.executable, str(child), str(directory / "tasks"), ",".join(lane_tasks)],
        directory, workers=2, env=os.environ.copy(), cwd=None,
        monitor=lambda: None, poll_interval=0.01)
    assert [row["task_id"] for row in rows] == tasks
    assert [row["lane"] for row in rows] == [0, 1, 0]
    assert rows[0]["pid"] == rows[2]["pid"] != rows[1]["pid"]
    assert cold > 0.05
    assert all((directory / "tasks" / task / "summary_test.json").is_file()
               for task in tasks)


def test_dynamic_persistent_pool_reassigns_to_idle_lane_without_restarting(tmp_path):
    child = tmp_path / "dynamic_lane.py"
    child.write_text('''import json, os, sys, time
from pathlib import Path
lane, tasks = Path(sys.argv[1]), Path(sys.argv[2])
sequence = 0
while True:
    assignment = lane / f"assignment_{sequence}.json"
    if not assignment.exists():
        if (lane / "assignment_stop.requested").exists():
            break
        time.sleep(0.01)
        continue
    item = json.loads(assignment.read_text())
    task = item["task_id"]
    if task == "bad":
        sys.exit(3)
    time.sleep(0.5 if task == "slow" else 0.025)
    out = tasks / task
    out.mkdir(parents=True)
    start = time.monotonic_ns()
    row = {"task_id": task, "returncode": 0, "start_monotonic_ns": start,
           "end_monotonic_ns": time.monotonic_ns(), "output": str(out), "pid": os.getpid()}
    temporary = out / "process.tmp"
    temporary.write_text(json.dumps(row))
    os.replace(temporary, out / "process.json")
    temporary = lane / f"assignment_{sequence}.complete.tmp"
    temporary.write_text(json.dumps(item))
    os.replace(temporary, lane / f"assignment_{sequence}.complete.json")
    sequence += 1
''', encoding="utf-8")
    tasks = ["slow", "fast_a", "fast_b", "fast_c"]
    directory = tmp_path / "serving"
    rows, cold = serving.run_dynamic_persistent_pool(
        tasks,
        lambda _eligible, lane, lane_dir: [
            sys.executable, str(child), str(lane_dir), str(directory / "tasks")],
        directory, workers=2, env=os.environ.copy(), cwd=None,
        monitor=lambda: None, poll_interval=0.01)
    assert [row["task_id"] for row in rows] == tasks
    assert [row["lane"] for row in rows] == [0, 1, 1, 1]
    assert rows[1]["pid"] == rows[2]["pid"] == rows[3]["pid"] != rows[0]["pid"]
    assert cold >= 0.5
    assert len(list((directory / "tasks").glob("*/process.json"))) == len(tasks)

    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="exited 3"):
        serving.run_dynamic_persistent_pool(
            ["bad", "fast", "later"],
            lambda _eligible, lane, lane_dir: [
                sys.executable, str(child), str(lane_dir), str(failed / "tasks")],
            failed, workers=2, env=os.environ.copy(), cwd=None,
            monitor=lambda: None, poll_interval=0.01)
    assert not (failed / "tasks" / "later" / "process.json").exists()


def test_persistent_abort_keeps_completed_task_receipt_without_throughput(tmp_path, monkeypatch):
    arm = "c2kv_c1_v2_verified_r8"
    config = runner.with_native_history_budget(
        with_candidate_methods(base_config(), ("c1_v2_verified",)), arm, 256)
    config["serving_persistent_runtime"] = True
    output = tmp_path / "paper"
    plan, profile = runner.prepare(config, output, tmp_path / "sglang")
    selected = next(row for row in plan if row["cell_id"] == f"bfcl_base__{arm}_b256")
    monkeypatch.setattr(serving, "wait_server", lambda *args: None)
    monkeypatch.setattr(serving, "cleanup_cell_processes", lambda *args: None)
    monkeypatch.setattr(serving, "UpstreamLiveness", lambda *args, **kwargs: lambda: None)

    def fail_pool(tasks, command_for_lane, directory, **kwargs):
        task_dir = directory / "tasks" / tasks[0]
        task_dir.mkdir(parents=True)
        (task_dir / f"summary_{arm}.json").write_text(json.dumps({
            "n_scored": 1, "semantic_score": 0.0}))
        (task_dir / "process.json").write_text(json.dumps({
            "task_id": tasks[0], "task_status": "method_failure",
            "failure": {"kind": "capacity_infeasible"}, "returncode": 0,
            "output": str(task_dir), "start_monotonic_ns": 1,
            "end_monotonic_ns": 2, "start_unix_ns": 1, "end_unix_ns": 2}))
        raise UpstreamUnavailable("test lane failed after one task")

    monkeypatch.setattr(serving, "run_persistent_pool", fail_pool)
    with pytest.raises(UpstreamUnavailable):
        serving.serve_cell(config, selected, output, tmp_path / "sglang", profile,
                           workers=1, tasks=TASKS[:2], port_offset=23000,
                           popen=lambda *args, **kwargs: FakeProcess(0))
    manifest = json.loads((output / "serving" / selected["cell_id"] /
                           "workers_1" / "serving.json").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["n_finished"] == 1 and manifest["n_unscored"] == 1
    assert manifest["n_method_failures"] == 1
    assert manifest["tasks"][0]["task_index"] == 0
    assert manifest["tasks"][0]["lane"] == 0
    assert manifest["tasks"][0]["failure"]["kind"] == "capacity_infeasible"
    assert manifest["total_runtime_seconds"] is None
    assert manifest["tasks_per_hour"] is None


def test_dynamic_persistent_abort_keeps_actual_lane_assignment(tmp_path, monkeypatch):
    arm = "c2kv_c1_v2_verified_r8"
    config = runner.with_native_history_budget(
        with_candidate_methods(base_config(), ("c1_v2_verified",)), arm, 256)
    config.update(serving_persistent_runtime=True, serving_dynamic_persistent_runtime=True)
    output = tmp_path / "paper"
    plan, profile = runner.prepare(config, output, tmp_path / "sglang")
    selected = next(row for row in plan if row["cell_id"] == f"bfcl_base__{arm}_b256")
    monkeypatch.setattr(serving, "wait_server", lambda *args: None)
    monkeypatch.setattr(serving, "cleanup_cell_processes", lambda *args: None)
    monkeypatch.setattr(serving, "UpstreamLiveness", lambda *args, **kwargs: lambda: None)

    def fail_pool(tasks, command_for_lane, directory, **kwargs):
        task_dir = directory / "tasks" / tasks[0]
        task_dir.mkdir(parents=True)
        (task_dir / f"summary_{arm}.json").write_text(json.dumps({
            "n_scored": 1, "semantic_score": 1.0}))
        (task_dir / "process.json").write_text(json.dumps({
            "task_id": tasks[0], "returncode": 0, "task_status": "completed",
            "output": str(task_dir), "start_monotonic_ns": 1,
            "end_monotonic_ns": 2, "start_unix_ns": 1,
            "end_unix_ns": 2, "lane": 1}))
        raise UpstreamUnavailable("test lane failed after one dynamic task")

    monkeypatch.setattr(serving, "run_dynamic_persistent_pool", fail_pool)
    with pytest.raises(UpstreamUnavailable):
        serving.serve_cell(config, selected, output, tmp_path / "sglang", profile,
                           workers=2, tasks=TASKS[:2], port_offset=23000,
                           popen=lambda *args, **kwargs: FakeProcess(0))
    manifest = json.loads((output / "serving" / selected["cell_id"] /
                           "workers_2" / "serving.json").read_text())
    assert manifest["status"] == "aborted"
    assert manifest["dynamic_persistent_runtime"] is True
    assert manifest["tasks"][0]["lane"] == 1
    assert manifest["total_runtime_seconds"] is None


def test_serving_records_features_and_checks_engine_before_children(tmp_path, monkeypatch):
    config = with_racer_methods(base_config(), ("c2kv",), parse_racer_policies("pending_verified"), 768)
    config.update(serving_native_raw_prefix_cache=True, serving_background_extras=True,
                  serving_bulk_cache_lookup=True, serving_cross_turn_prewarm=True,
                  serving_overlap_schedule=True)
    native = cell(config, "bfcl_long_context__racer_c2kv_pending_verified_b768")
    monkeypatch.setattr(serving, "wait_server", lambda *args: None)
    monkeypatch.setattr(serving, "cleanup_cell_processes", lambda *args: None)
    monkeypatch.setattr(serving, "UpstreamLiveness", lambda *args, **kwargs: lambda: None)
    commands = []
    verified = []

    def require(port, features):
        assert len(commands) == 1
        verified.append(features)
        return features

    monkeypatch.setattr(serving, "require_native_serving_features", require)

    def popen(command, **kwargs):
        commands.append((command, kwargs))
        if len(commands) > 1:
            assert verified
            (Path(value(command, "--out")) / f"summary_{native['arm']}.json").write_text(json.dumps({"n_scored": 1, "semantic_score": 1}))
        return FakeProcess(0)

    result = serving.serve_cell(config, native, tmp_path, tmp_path / "engine", PROFILE,
                                workers=1, tasks=TASKS[:1], port_offset=22000,
                                poll_interval=0, popen=popen)
    assert result["status"] == "completed"
    assert result["overlap_schedule"] is True
    assert "--disable-overlap-schedule" not in commands[0][0]
    assert result["native_serving_features"] == result["engine_serving_capabilities"] == verified[0]
    for _, kwargs in commands:
        assert kwargs["env"]["C2KV_NATIVE_RAW_PREFIX_CACHE"] == "1"
        assert kwargs["env"]["C2KV_NATIVE_BACKGROUND_EXTRAS"] == "1"
        assert kwargs["env"]["C2KV_NATIVE_BULK_CACHE_LOOKUP"] == "1"
        assert kwargs["env"]["C2KV_NATIVE_CROSS_TURN_PREWARM"] == "1"
