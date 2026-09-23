"""Concurrent task serving on one CUDA engine, for throughput and runtime.

The paper matrix runs single-flight so that per-request telemetry can attribute
peaks to one request. A serving run keeps every task's command unchanged: each
official task runs as its own single-task child (its own proxy, its own
official harness), at most ``workers`` children run at once, and all of them
share one engine whose request-slot cap is scaled by ``workers``. Each running
child binds its lane's proxy port. Per-request peak fields in the engine ledger
are engine-wide under concurrency; the serving manifest records per-task wall
times, exit codes and scores, from which task throughput and total runtime
follow.
"""
from __future__ import annotations

from collections import deque
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .artifact_io import atomic_json
from .process_lifecycle import stop_owned_group, unwind_on_termination
from .runner import (
    ROOT, _guard_checkpoint_serving_layout, _guard_method_actor, _guard_tool_context,
    _unsupported_stage, cleanup_cell_processes, is_native_arm, paper_env,
    require_budget_renderer, run_command, server_command, wait_server, with_port_offset,
)
from .task_subsets import is_subset
from .upstream_liveness import UpstreamLiveness

SCHEMA = "paper-serving-v1"
# Per-task children need a single-task entry point on both the native (c1
# --task-ids) and the proxy (run.py --run-ids) paths; BFCL has both.
SERVING_BENCHMARKS = frozenset({"bfcl_base", "bfcl_long_context"})


def serving_server_command(config, source, cell, workers):
    """The cell's engine command with its request-slot cap scaled by ``workers``."""
    command = server_command(config, source, cell["arm"], cell["benchmark"],
                             tool_checkpoint=cell.get("tool_checkpoint"),
                             tool_memory=cell.get("tool_memory"))
    index = command.index("--max-running-requests") + 1
    command[index] = str(int(command[index]) * workers)
    return command


def lane_ports(config, workers):
    return [config["proxy_port"] + lane for lane in range(workers)]


def task_command(config, cell, task, directory, profile_path, lane):
    """The unchanged single-task command, bound to one lane's proxy port."""
    lane_config = dict(config, proxy_port=config["proxy_port"] + lane)
    if is_native_arm(cell["arm"]):
        return run_command(lane_config, cell, directory, profile_path) + ["--task-ids", task]
    # Every task has its own proxy, so its episode reset must not flush the
    # engine that the other lanes are still using.
    return run_command(lane_config, dict(cell, task_ids=[task]), directory,
                       profile_path) + ["--shared-engine"]


def _validate_task_id(task):
    if (not isinstance(task, str) or not task or task != task.strip() or "," in task
            or task in {".", ".."} or "/" in task or "\\" in task):
        raise ValueError(f"Task ID cannot name a task output directory: {task!r}")


def run_pool(tasks, command_for, directory_for, *, workers, env, cwd, monitor,
             poll_interval=1.0, popen=subprocess.Popen):
    """Run one child per task, at most ``workers`` at once, in task order.

    Each child is started in its own session so an interruption stops exactly
    the children this pool owns; a finished child's session is reaped before
    its lane is reused.
    """
    pending = deque(enumerate(tasks))
    free = list(range(workers))
    running = {}
    rows = []
    try:
        while pending or running:
            while pending and free:
                lane = free.pop(0)
                index, task = pending.popleft()
                directory = directory_for(task)
                directory.mkdir(parents=True, exist_ok=False)
                command = command_for(task, lane, directory)
                log = (directory / "child.log").open("w", encoding="utf-8")
                try:
                    start_unix_ns, start = time.time_ns(), time.monotonic()
                    process = popen(command, env=env, cwd=cwd, stdout=log,
                                    stderr=subprocess.STDOUT,
                                    start_new_session=os.name == "posix")
                except BaseException:
                    log.close()
                    raise
                running[lane] = {"task_index": index, "task_id": task, "lane": lane,
                                 "process": process, "log": log, "start": start,
                                 "start_unix_ns": start_unix_ns, "output": directory}
            time.sleep(poll_interval)
            monitor()
            for lane in sorted(running):
                entry = running[lane]
                code = entry["process"].poll()
                if code is None:
                    continue
                end_unix_ns, end = time.time_ns(), time.monotonic()
                del running[lane]
                try:
                    stop_owned_group(entry["process"])
                finally:
                    entry["log"].close()
                rows.append({"task_index": entry["task_index"], "task_id": entry["task_id"],
                             "lane": lane, "returncode": code,
                             "start_unix_ns": entry["start_unix_ns"], "end_unix_ns": end_unix_ns,
                             "wall_seconds": end - entry["start"],
                             "output": str(entry["output"])})
                free.append(lane)
                free.sort()
    except BaseException as failure:
        for entry in running.values():
            try:
                stop_owned_group(entry["process"])
            except Exception as error:  # keep stopping the remaining lanes
                if hasattr(failure, "add_note"):
                    failure.add_note(f"Additionally, stopping {entry['task_id']} failed: {error}")
            finally:
                entry["log"].close()
        raise
    return rows


def _task_score(row, arm):
    """Attach the task's official score when its child completed normally."""
    path = Path(row["output"]) / f"summary_{arm}.json"
    if row["returncode"] != 0 or not path.is_file():
        return dict(row, n_scored=None, semantic_score=None)
    summary = json.loads(path.read_text(encoding="utf-8"))
    return dict(row, n_scored=summary.get("n_scored", summary.get("n")),
                semantic_score=summary.get("semantic_score"))


def summarize(cell, workers, max_running_requests, ports, tasks, rows):
    rows = sorted((_task_score(row, cell["arm"]) for row in rows), key=lambda row: row["task_index"])
    scored = [row for row in rows if row["n_scored"] == 1 and row["semantic_score"] is not None]
    runtime = ((max(row["end_unix_ns"] for row in rows) - min(row["start_unix_ns"] for row in rows)) / 1e9
               if rows else None)
    hours = runtime / 3600 if runtime else None
    return {
        "schema": SCHEMA, "cell_id": cell["cell_id"], "arm": cell["arm"],
        "benchmark": cell["benchmark"], "workers": workers,
        "engine_max_running_requests": max_running_requests, "lane_proxy_ports": ports,
        "measurement_scope": "concurrent_serving",
        "scope_note": ("Engine per-request peak and duration fields are engine-wide while "
                       "tasks overlap; use per-task wall times and the engine ledger's pool "
                       "occupancy, not per-request peaks."),
        "n_tasks": len(tasks), "n_finished": len(rows),
        "n_failed": sum(row["returncode"] != 0 for row in rows),
        "n_scored": len(scored),
        "successful_tasks": sum(row["semantic_score"] for row in scored),
        "total_runtime_seconds": runtime,
        "tasks_per_hour": len(scored) / hours if hours else None,
        "successful_tasks_per_hour": sum(row["semantic_score"] for row in scored) / hours if hours else None,
        "status": "completed" if len(scored) == len(tasks) else "completed_with_failures",
        "tasks": rows,
    }


@unwind_on_termination
def serve_cell(config, cell, output, source, profile_path, *, workers, tasks,
               port_offset=0, poll_interval=1.0, popen=subprocess.Popen):
    """Serve every task of one prepared closed-loop cell with ``workers`` lanes."""
    from benchmarks.arms import get_arm

    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if cell["benchmark"] not in SERVING_BENCHMARKS:
        raise ValueError(f"Serving runs support {sorted(SERVING_BENCHMARKS)} cells")
    if is_subset(cell):
        raise ValueError("Serving runs take whole prepared cells; pass tasks explicitly")
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("Serving tasks must be a non-empty list of unique IDs")
    for task in tasks:
        _validate_task_id(task)
    unsupported = _unsupported_stage("closed_loop", cell)
    if unsupported:
        raise RuntimeError(f"Unsupported closed_loop for {cell['arm']}: {unsupported}")
    _guard_method_actor(cell)
    _guard_checkpoint_serving_layout(config, cell)
    _guard_tool_context(cell)
    config = with_port_offset(config, port_offset)
    ports = lane_ports(config, workers)
    if config["server_port"] in ports:
        raise ValueError("Lane proxy ports overlap the engine port")
    server_cmd = serving_server_command(config, source, cell, workers)
    max_running = int(server_cmd[server_cmd.index("--max-running-requests") + 1])
    directory = Path(output) / "serving" / cell["cell_id"] / f"workers_{workers}"
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "started.json", {
        "schema": SCHEMA, "cell": cell, "config": config, "workers": workers,
        "tasks": list(tasks), "lane_proxy_ports": ports, "server_command": server_cmd,
        "port_offset": port_offset, "sglang_source": str(source), "time": time.time()},
        exclusive=True)
    for port in (config["server_port"], *ports):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Configured port {port} is already occupied")
    env = paper_env(config, source)
    telemetry_name = ("native_engine_telemetry.jsonl" if is_native_arm(cell["arm"])
                      else "server_telemetry.jsonl")
    env["C2KV_PAPER_TELEMETRY_LOG"] = str(directory / telemetry_name)
    native = is_native_arm(cell["arm"])
    with (directory / "server.log").open("w") as log:
        # Own only this process group. Never stop another experiment's server.
        server = popen(server_cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                       start_new_session=True)
        run_failure = None
        try:
            wait_server(server, config["server_port"])
            monitor = UpstreamLiveness(f"http://127.0.0.1:{config['server_port']}", process=server)
            if get_arm(cell["arm"]).text_history_budget_tokens is not None:
                require_budget_renderer(config["server_port"])
            rows = run_pool(
                tasks,
                lambda task, lane, task_dir: task_command(
                    config, cell, task, task_dir, profile_path, lane),
                lambda task: directory / "tasks" / task,
                workers=workers, env=env, cwd=ROOT.parent if native else None,
                monitor=monitor, poll_interval=poll_interval, popen=popen)
        except BaseException:
            run_failure = sys.exc_info()
        finally:
            try:
                cleanup_cell_processes(None, server)
            except BaseException as error:
                if run_failure is None:
                    raise
                if hasattr(run_failure[1], "add_note"):
                    run_failure[1].add_note(f"Additionally, cleanup failed: {error}")
        if run_failure is not None:
            _, error, traceback = run_failure
            raise error.with_traceback(traceback)
    manifest = summarize(cell, workers, max_running, ports, tasks, rows)
    atomic_json(directory / "serving.json", manifest)
    return manifest
