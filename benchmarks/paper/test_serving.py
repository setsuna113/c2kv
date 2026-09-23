"""CPU-only checks for concurrent serving runs of prepared paper cells."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.paper import runner, serving
from benchmarks.paper.racer_matrix import parse_racer_policies, with_racer_methods
from benchmarks.paper.upstream_liveness import UpstreamUnavailable

TASKS = [f"multi_turn_long_context_{index}" for index in range(5)]
PROFILE = Path("out/deployment_profile.json")


def base_config():
    return json.loads(runner.DEFAULT_CONFIG.read_text(encoding="utf-8"))


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
    ports = {value(command, "--proxy-port") for command, _ in commands[1:]}
    assert ports == {str(config["proxy_port"] + offset + lane) for lane in (0, 1)}
    assert all("--shared-engine" in command for command, _ in commands[1:])
    directory = output / "serving" / full["cell_id"] / "workers_2"
    assert json.loads((directory / "serving.json").read_text()) == manifest
    assert manifest["n_tasks"] == manifest["n_scored"] == len(TASKS)
    assert manifest["successful_tasks"] == 2
    assert manifest["engine_max_running_requests"] == 2
    assert manifest["measurement_scope"] == "concurrent_serving"
    assert manifest["status"] == "completed"
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
        runner.main(argv)
