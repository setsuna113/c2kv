"""CPU process checks for task isolation in the persistent lane protocol."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import urllib.error
import urllib.request

import pytest


RUNTIME = (Path(__file__).resolve().parents[2] / "experiments" /
           "history_system" / "runtime")


CHILD = r'''
import json
import os
from pathlib import Path
import time
from benchmarks.memory_runtime import event_native_server as server
from benchmarks.memory_runtime import event_native_persistent as persistent
from benchmarks.memory_runtime.event_native_api import EventNativeAPI, make_server

class Runner:
    max_generation_calls = 4
    def __init__(self):
        self.generation_calls = 0
    def run(self, payload):
        self.generation_calls += 1
        return {
            "status": "ok", "outer_request_id": payload["outer_request_id"],
            "session_id": payload["session_id"], "decision_key": payload["decision_key"],
            "response": {"role": "assistant", "content": "ok", "finish_reason": "stop"},
            "generation_usage_total": {
                "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

def serve_task(args):
    args.out.mkdir(parents=True, exist_ok=False)
    api = EventNativeAPI(
        Runner(), run_id=args.run_id, model_name=args.model_name,
        view_mode=args.view_mode, max_new_tokens=args.max_new_tokens,
        allowed_task_ids=[args.task_ids], max_decisions=args.max_decisions,
        deadline_monotonic=time.monotonic() + 20,
        steps_path=args.out / "steps.jsonl", benchmark=args.benchmark)
    http = make_server(api, port=0)
    http.timeout = 0.05
    server.save_json(args.out / "ready.json", {
        "pid": os.getpid(), "port": http.server_address[1], "task_id": args.task_ids})
    try:
        while not (args.out / "persistent_task_stop.requested").exists():
            if (args.out / "abort.requested").exists():
                server.save_json(args.out / "final.json", {
                    "status": "stopped", "stop_reason": "signal"})
                return
            http.handle_request()
    finally:
        http.server_close()
    server.save_json(args.out / "final.json", {
        "status": "stopped", "stop_reason": "task_completed",
        "api_health": api.health(), "pid": os.getpid(),
        "cost_summary": {"status": "test"},
        "journal_summary": {"completed": True, "failed": 0, "pending": 0}})

server._serve = serve_task
persistent.run_plan(Path(__import__("sys").argv[1]),
                    dynamic="--dynamic" in __import__("sys").argv)
'''


def _plan(tmp_path, tasks=("one", "two")):
    commands = []
    for task in tasks:
        commands.append([
            sys.executable, "-m", "benchmarks.memory_runtime.event_native_server",
            "--checkpoint", str(tmp_path / "checkpoint"),
            "--out", str(tmp_path / task / "server"),
            "--run-id", "test", "--model-name", "test-model",
            "--view-mode", "static", "--ratio", "8",
            "--max-new-tokens", "8", "--task-ids", task,
            "--max-decisions", "4", "--max-generation-calls", "4",
            "--max-wall-seconds", "20", "--generation-backend", "sglang",
            "--sglang-backend-url", "http://127.0.0.1:1", "--port", "0",
        ])
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(commands), encoding="utf-8")
    return path


def _wait(path, process):
    deadline = time.monotonic() + 15
    while not path.is_file():
        assert process.poll() is None, f"lane exited early: {process.returncode}"
        assert time.monotonic() < deadline, f"timed out waiting for {path}"
        time.sleep(0.02)
    return json.loads(path.read_text(encoding="utf-8"))


def _post(port, task, step):
    body = {
        "model": "test-model", "temperature": 0, "store": False,
        "max_completion_tokens": 8,
        "messages": [{"role": "user", "content": "test"}],
        "c2kv_eval_context": {"benchmark": "bfcl", "task_id": task,
                              "user_turn": 0, "step": step, "attempt": 0},
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def _health(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
        return json.load(response)


def _start(tmp_path, *, dynamic=False, tasks=("one", "two")):
    child = tmp_path / "child.py"
    child.write_text(CHILD, encoding="utf-8")
    env = dict(os.environ, PYTHONPATH=str(RUNTIME))
    command = [sys.executable, str(child), str(_plan(tmp_path, tasks))]
    if dynamic:
        command.append("--dynamic")
    process = subprocess.Popen(command,
                               cwd=RUNTIME, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    return process


def test_real_process_reuses_pid_but_resets_api_and_journals(tmp_path):
    process = _start(tmp_path)
    try:
        (tmp_path / "one").mkdir()
        (tmp_path / "one" / "persistent_task_start.requested").touch()
        first = _wait(tmp_path / "one" / "server" / "ready.json", process)
        assert _health(first["port"])["decisions_reserved"] == 0
        _post(first["port"], "one", 0)
        _post(first["port"], "one", 1)
        assert _health(first["port"])["decisions_reserved"] == 2
        (tmp_path / "one" / "server" / "persistent_task_stop.requested").touch()
        _wait(tmp_path / "one" / "server" / "persistent_task_complete.json", process)
        assert not (tmp_path / "two").exists()
        (tmp_path / "two").mkdir()
        (tmp_path / "two" / "persistent_task_start.requested").touch()
        second = _wait(tmp_path / "two" / "server" / "ready.json", process)
        assert second["pid"] == first["pid"]
        assert _health(second["port"])["decisions_reserved"] == 0
        assert _health(second["port"])["generation_calls_reserved"] == 0
        try:
            _post(second["port"], "one", 0)
        except urllib.error.HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("previous task must not be accepted")
        _post(second["port"], "two", 0)
        (tmp_path / "two" / "server" / "persistent_task_stop.requested").touch()
        _wait(tmp_path / "two" / "server" / "persistent_task_complete.json", process)
        assert process.wait(timeout=5) == 0
        first_steps = (tmp_path / "one" / "server" / "steps.jsonl").read_text().splitlines()
        second_steps = (tmp_path / "two" / "server" / "steps.jsonl").read_text().splitlines()
        assert len(first_steps) == 2 and len(second_steps) == 1
        assert all(json.loads(row)["session_id"] == "bfcl/one/attempt-0" for row in first_steps)
        assert json.loads(second_steps[0])["session_id"] == "bfcl/two/attempt-0"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def test_dynamic_lane_runs_out_of_order_and_leaves_unassigned_task_uncreated(tmp_path):
    process = _start(tmp_path, dynamic=True, tasks=("one", "two", "three"))
    try:
        assert not (tmp_path / "one").exists()
        assert not (tmp_path / "three").exists()
        (tmp_path / "two").mkdir()
        (tmp_path / "two" / "persistent_task_start.requested").touch()
        second = _wait(tmp_path / "two" / "server" / "ready.json", process)
        _post(second["port"], "two", 0)
        (tmp_path / "two" / "server" / "persistent_task_stop.requested").touch()
        second_complete = _wait(
            tmp_path / "two" / "server" / "persistent_task_complete.json", process)
        assert second_complete["dynamic_dispatch"] is True
        assert second_complete["worker_pid"] == second["pid"]

        (tmp_path / "one").mkdir()
        (tmp_path / "one" / "persistent_task_start.requested").touch()
        first = _wait(tmp_path / "one" / "server" / "ready.json", process)
        assert first["pid"] == second["pid"]
        assert _health(first["port"])["decisions_reserved"] == 0
        try:
            _post(first["port"], "two", 0)
        except urllib.error.HTTPError as error:
            assert error.code == 403
        else:
            raise AssertionError("previous task must not be accepted")
        _post(first["port"], "one", 0)
        (tmp_path / "one" / "server" / "persistent_task_stop.requested").touch()
        first_complete = _wait(
            tmp_path / "one" / "server" / "persistent_task_complete.json", process)
        assert first_complete["worker_pid"] == second_complete["worker_pid"]
        assert not (tmp_path / "three").exists()
        (tmp_path / "persistent_lane_stop.requested").touch()
        assert process.wait(timeout=2) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


SHUTDOWN_CHILD = r'''
from pathlib import Path
import sys
import time

marker = Path(sys.argv[1])
while not marker.exists():
    time.sleep(0.01)
time.sleep(float(sys.argv[2]))
raise SystemExit(int(sys.argv[3]))
'''


def _shutdown_lane(tmp_path, *, exit_delay, exit_code):
    from benchmarks.paper import c1 as paper_c1

    delivery = paper_c1.load_delivery()
    lane = delivery.PersistentTaskServer.__new__(delivery.PersistentTaskServer)
    lane.dynamic = True
    lane.plan_path = tmp_path / "persistent_lane_plan.json"
    lane.log = (tmp_path / "persistent_lane.log").open("w", encoding="utf-8")
    lane.process = subprocess.Popen(
        [sys.executable, "-c", SHUTDOWN_CHILD,
         str(tmp_path / "persistent_lane_stop.requested"),
         str(exit_delay), str(exit_code)],
        stdout=lane.log, stderr=subprocess.STDOUT)
    return delivery, lane


def test_dynamic_close_accepts_exit_after_original_two_second_limit(tmp_path):
    _, lane = _shutdown_lane(tmp_path, exit_delay=2.25, exit_code=0)
    try:
        lane.close()
        assert lane.process.returncode == 0
        assert lane.log.closed
        assert (tmp_path / "persistent_lane_stop.requested").is_file()
    finally:
        if lane.process.poll() is None:
            lane.process.kill()
            lane.process.wait(timeout=5)


def test_dynamic_close_reaps_stuck_process_and_reports_timeout(tmp_path, monkeypatch):
    delivery, lane = _shutdown_lane(tmp_path, exit_delay=30, exit_code=0)
    monkeypatch.setattr(delivery, "DYNAMIC_PERSISTENT_CLOSE_TIMEOUT_SECONDS", 0.1)
    with pytest.raises(TimeoutError, match="did not stop within 0.1s"):
        lane.close()
    assert lane.process.poll() is not None
    assert lane.log.closed


def test_dynamic_close_reports_nonzero_exit(tmp_path):
    _, lane = _shutdown_lane(tmp_path, exit_delay=0, exit_code=7)
    with pytest.raises(RuntimeError, match="Dynamic persistent lane exited 7"):
        lane.close()
    assert lane.process.returncode == 7
    assert lane.log.closed


def test_c1_dynamic_assignment_iterator_acknowledges_only_completed_tasks(tmp_path):
    from benchmarks.paper import c1 as paper_c1

    server = SimpleNamespace(process=SimpleNamespace(poll=lambda: None))
    stream = paper_c1._dynamic_task_assignments(tmp_path, ["one", "two"], server)
    (tmp_path / "assignment_0.json").write_text(json.dumps({
        "sequence": 0, "task_index": 1, "task_id": "two"}), encoding="utf-8")
    assert next(stream) == "two"
    assert not (tmp_path / "assignment_0.complete.json").exists()
    (tmp_path / "assignment_1.json").write_text(json.dumps({
        "sequence": 1, "task_index": 0, "task_id": "one"}), encoding="utf-8")
    assert next(stream) == "one"
    assert json.loads((tmp_path / "assignment_0.complete.json").read_text()) == {
        "sequence": 0, "task_index": 1, "task_id": "two"}
    (tmp_path / "assignment_stop.requested").touch()
    assert list(stream) == []
    assert (tmp_path / "assignment_1.complete.json").is_file()


def test_aborted_first_task_cannot_start_next_task(tmp_path):
    process = _start(tmp_path)
    try:
        (tmp_path / "one").mkdir()
        (tmp_path / "one" / "persistent_task_start.requested").touch()
        _wait(tmp_path / "one" / "server" / "ready.json", process)
        (tmp_path / "one" / "server" / "abort.requested").touch()
        assert process.wait(timeout=5) != 0
        assert not (tmp_path / "two" / "server").exists()
        assert not (tmp_path / "one" / "server" / "persistent_task_complete.json").exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.parametrize("dynamic, order", [
    (False, ("one", "two")), (True, ("two", "one")),
])
def test_parent_run_task_owns_output_before_lane_starts_each_task(
        tmp_path, monkeypatch, dynamic, order):
    from benchmarks.paper import c1 as paper_c1

    delivery = paper_c1.load_delivery()
    child = tmp_path / "child.py"
    child.write_text(CHILD, encoding="utf-8")
    worker = tmp_path / "worker.py"
    worker.write_text('''import json, sys, urllib.request
from pathlib import Path
root, task = Path(sys.argv[1]), sys.argv[2]
ready = json.loads((root / "server" / "ready.json").read_text())
body = {"model": "test-model", "temperature": 0, "store": False,
        "max_completion_tokens": 8, "messages": [{"role": "user", "content": "test"}],
        "c2kv_eval_context": {"benchmark": "bfcl", "task_id": task,
                              "user_turn": 0, "step": 0, "attempt": 0}}
request = urllib.request.Request(
    f"http://127.0.0.1:{ready['port']}/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=5) as response:
    assert response.status == 200
(root / "bfcl").mkdir()
(root / "bfcl" / "official_summary.json").write_text(
    json.dumps({"n_scored": 1, "n_generated": 1, "semantic_score": 1.0}))
''', encoding="utf-8")
    lane_root = tmp_path / "lane" / "native"
    lane_root.mkdir(parents=True)
    args = SimpleNamespace(out=lane_root, benchmark="bfcl", task_timeout=10,
                           method="proposed", detector="t02_risk", tool_memory="none")

    def commands(_args, task, _controller):
        out = lane_root / "task_shards" / task / "server"
        server_command = [
            sys.executable, "-m", "benchmarks.memory_runtime.event_native_server",
            "--checkpoint", str(tmp_path / "checkpoint"), "--out", str(out),
            "--run-id", "test", "--model-name", "test-model",
            "--view-mode", "static", "--ratio", "8", "--max-new-tokens", "8",
            "--task-ids", task, "--max-decisions", "4", "--max-generation-calls", "4",
            "--max-wall-seconds", "20", "--generation-backend", "sglang",
            "--sglang-backend-url", "http://127.0.0.1:1", "--port", "0",
        ]
        return server_command, [sys.executable, str(worker), str(out.parent), task]

    monkeypatch.setattr(delivery, "commands_for_task", commands)
    monkeypatch.setattr(delivery, "summarize_task", lambda *args: {"official_score": 1.0})
    monkeypatch.setattr(delivery, "functional_checks", lambda *args: {"required": {"test": True}})
    actual_popen = subprocess.Popen

    def launch(command, **kwargs):
        if len(command) > 3 and command[2] == "benchmarks.memory_runtime.event_native_persistent":
            command = [sys.executable, str(child), command[3],
                       *(["--dynamic"] if "--dynamic" in command else [])]
        return actual_popen(command, **kwargs)

    monkeypatch.setattr(delivery.subprocess, "Popen", launch)
    lane = delivery.PersistentTaskServer(args, ["one", "two"], tmp_path / "controller.json",
                                         dynamic=dynamic)
    try:
        assert not (lane_root / "task_shards" / order[0]).exists()
        first, _ = delivery.run_task(args, order[0], tmp_path / "controller.json",
                                     persistent_server=lane)
        assert first["status"] == "completed"
        assert not (lane_root / "task_shards" / order[1]).exists()
        second, _ = delivery.run_task(args, order[1], tmp_path / "controller.json",
                                      persistent_server=lane)
        assert second["status"] == "completed"
        assert (lane_root / "task_shards" / "one" / "server" / "final.json").is_file()
        assert (lane_root / "task_shards" / "two" / "server" / "final.json").is_file()
    finally:
        lane.close()
