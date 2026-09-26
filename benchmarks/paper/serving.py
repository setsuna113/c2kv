"""Concurrent task serving on one CUDA engine, for throughput and runtime.

The paper matrix runs single-flight so that per-request telemetry can attribute
peaks to one request. A serving run keeps every task's command unchanged: each
official task runs as its own single-task child (its own proxy, its own
official harness), at most ``workers`` children run at once, and all of them
share one engine whose request-slot cap defaults to scaling by ``workers`` and
can be fixed independently through configuration. Each running
child binds its lane's proxy port. Per-request peak fields in the engine ledger
are engine-wide under concurrency; the serving manifest records per-task wall
times, exit codes and scores, from which task throughput and total runtime
follow.
"""
from __future__ import annotations

from collections import deque
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from .artifact_io import atomic_json
from .process_lifecycle import stop_owned_group, unwind_on_termination
from .runner import (
    ROOT, _guard_checkpoint_serving_layout, _guard_method_actor, _guard_tool_context,
    _unsupported_stage, cleanup_cell_processes, is_native_arm, paper_env,
    require_budget_renderer, run_command, server_command, wait_server, with_port_offset,
)
from .task_subsets import is_subset
from .racer_matrix import is_racer_arm, parse_racer_arm_name
from .serving_measurement import aggregate_engine_ledger
from .async_measurement import summarize_async_compression
from .upstream_liveness import UpstreamLiveness

SCHEMA = "paper-serving-v2"
# Per-task children need a single-task entry point on both the native (c1
# --task-ids) and the proxy (run.py --run-ids) paths; BFCL has both.
SERVING_BENCHMARKS = frozenset({"bfcl_base", "bfcl_long_context"})
FEATURES = {
    "raw_prefix_cache": ("serving_native_raw_prefix_cache", "C2KV_NATIVE_RAW_PREFIX_CACHE", "raw-prefix-v1"),
    "background_extras": ("serving_background_extras", "C2KV_NATIVE_BACKGROUND_EXTRAS", "selected-first-response-barrier-v1"),
    "bulk_cache_lookup": ("serving_bulk_cache_lookup", "C2KV_NATIVE_BULK_CACHE_LOOKUP", "bulk-cache-lookup-v1"),
    "cross_turn_prewarm": ("serving_cross_turn_prewarm", "C2KV_NATIVE_CROSS_TURN_PREWARM", "cross-turn-prewarm-v1"),
    "async_compression": ("serving_async_compression", "C2KV_NATIVE_ASYNC_COMPRESSION", "nonblocking-history-v1"),
}


def native_serving_features(config, cell):
    """Resolve explicit native optimizations, leaving Full controls unchanged."""
    requested = {}
    for name, (key, _, version) in FEATURES.items():
        value = config.get(key, False)
        if type(value) is not bool:
            raise ValueError(f"{key} must be a boolean")
        requested[name] = version if value else None
    if not any(requested.values()) or cell["arm"] == "full":
        return dict.fromkeys(FEATURES)
    arm = cell["arm"]
    if not _v4_c2kv_cell(cell):
        raise ValueError("Native serving optimizations require a v4 C2KV cell")
    return requested


def _v4_c2kv_cell(cell):
    arm = cell.get("arm", "")
    backend = cell.get("racer_backend") or {}
    return (isinstance(arm, str) and arm.startswith("racer_v4_c2kv_")
            and is_racer_arm(arm) and parse_racer_arm_name(arm)[0] == "c2kv"
            and backend.get("schema") == "racer-backend-v4"
            and backend.get("backend") == "c2kv")


def require_native_serving_features(port, requested):
    """Refuse silently ignored feature flags on an older engine checkout."""
    if not any(requested.values()):
        return None
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/model_info", timeout=10) as response:
        info = json.load(response)
    reported = info.get("c2kv_native_packed", {}).get("serving_features", {})
    for feature, version in requested.items():
        if version is not None and reported.get(feature) != version:
            raise RuntimeError(f"Engine does not confirm requested {feature}={version}")
    return reported


def serving_overlap_schedule_enabled(config):
    enabled = config.get("serving_overlap_schedule", False)
    if type(enabled) is not bool:
        raise ValueError("serving_overlap_schedule must be a boolean")
    return enabled


def serving_server_command(config, source, cell, workers):
    """The cell's engine command with an optional fixed request-slot cap."""
    command = server_command(config, source, cell["arm"], cell["benchmark"],
                             tool_checkpoint=cell.get("tool_checkpoint"),
                             tool_memory=cell.get("tool_memory"))
    index = command.index("--max-running-requests") + 1
    if "serving_engine_max_running_requests" in config:
        cap = config["serving_engine_max_running_requests"]
        if type(cap) is not int or cap <= 0:
            raise ValueError("serving_engine_max_running_requests must be a positive integer")
    else:
        cap = int(command[index]) * workers
    command[index] = str(cap)
    if native_serving_features(config, cell)["raw_prefix_cache"] is not None:
        command = [part for part in command if part != "--disable-radix-cache"]
    # The shared serving engine uses the validated decode graph on both arms.
    command = [part for part in command if part != "--disable-cuda-graph"]
    if serving_overlap_schedule_enabled(config):
        command.remove("--disable-overlap-schedule")
    eviction = config.get("serving_radix_eviction_policy")
    if eviction is not None:
        if eviction not in {"lru", "lfu", "slru", "priority"}:
            raise ValueError("serving_radix_eviction_policy must be lru, lfu, slru, or priority")
        command += ["--radix-eviction-policy", eviction]
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


def persistent_runtime_enabled(config, cell):
    value = config.get("serving_persistent_runtime", False)
    if type(value) is not bool:
        raise ValueError("serving_persistent_runtime must be a boolean")
    if value and (not _v4_c2kv_cell(cell)
                  or cell.get("history_budget_tokens") != 256
                  or cell["benchmark"] not in SERVING_BENCHMARKS):
        raise ValueError("Persistent runtime supports v4 C2KV BFCL B256 only")
    return value


def dynamic_persistent_runtime_enabled(config, persistent_runtime):
    value = config.get("serving_dynamic_persistent_runtime", False)
    if type(value) is not bool:
        raise ValueError("serving_dynamic_persistent_runtime must be a boolean")
    if value and not persistent_runtime:
        raise ValueError("Dynamic persistent dispatch requires persistent C1 runtime")
    return value


def persistent_lane_command(config, cell, tasks, directory, profile_path, lane,
                            task_output, *, dynamic=False):
    lane_config = dict(config, proxy_port=config["proxy_port"] + lane)
    command = (run_command(lane_config, cell, directory, profile_path)
               + ["--task-ids", ",".join(tasks), "--persistent-runtime",
                  "--serving-task-output", str(task_output)])
    if dynamic:
        command += ["--dynamic-serving-lane", str(lane)]
    return command


def run_persistent_pool(tasks, command_for_lane, directory, *, workers, env, cwd,
                        monitor, poll_interval=0.1, popen=subprocess.Popen):
    """Run one persistent C1 child per lane and retain per-task receipts."""
    assignments = [tasks[lane::workers] for lane in range(workers)]
    running = {}
    first_start = last_end = None
    try:
        for lane, lane_tasks in enumerate(assignments):
            if not lane_tasks:
                continue
            lane_dir = directory / "lanes" / f"lane_{lane}"
            lane_dir.mkdir(parents=True, exist_ok=False)
            command = command_for_lane(lane_tasks, lane, lane_dir)
            log = (lane_dir / "child.log").open("w", encoding="utf-8")
            try:
                started = time.monotonic_ns()
                first_start = started if first_start is None else min(first_start, started)
                process = popen(command, env=env, cwd=cwd, stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=os.name == "posix")
            except BaseException:
                log.close()
                raise
            running[lane] = (process, log)
            atomic_json(lane_dir / "started.json", {
                "lane": lane, "tasks": lane_tasks, "command": command,
                "start_monotonic_ns": started})
        while running:
            time.sleep(poll_interval)
            for lane, (process, log) in list(running.items()):
                code = process.poll()
                if code is None:
                    continue
                last_end = time.monotonic_ns()
                del running[lane]
                try:
                    stop_owned_group(process)
                finally:
                    log.close()
                atomic_json(directory / "lanes" / f"lane_{lane}" / "process.json",
                            {"lane": lane, "returncode": code,
                             "end_monotonic_ns": last_end})
                if code != 0:
                    raise RuntimeError(f"Persistent C1 lane {lane} exited {code}")
            if running:
                monitor()
    except BaseException as failure:
        for process, log in running.values():
            try:
                stop_owned_group(process)
            except Exception as error:
                if hasattr(failure, "add_note"):
                    failure.add_note(f"Additionally, stopping persistent lane failed: {error}")
            finally:
                log.close()
        raise
    rows = []
    for index, task in enumerate(tasks):
        path = directory / "tasks" / task / "process.json"
        if not path.is_file():
            raise RuntimeError(f"Persistent lane did not record task {task!r}")
        row = json.loads(path.read_text(encoding="utf-8"))
        rows.append(dict(row, task_index=index, lane=index % workers))
    return rows, (last_end - first_start) / 1e9


def run_dynamic_persistent_pool(tasks, command_for_lane, directory, *, workers, env, cwd,
                                monitor, poll_interval=0.1, popen=subprocess.Popen):
    """Dispatch each task to the next idle persistent lane without retries."""
    pending = deque(enumerate(tasks))
    running = {}
    active = {}
    next_sequence = {}
    stopping = set()
    rows = []
    first_start = last_end = None

    def dispatch(lane):
        lane_dir = directory / "lanes" / f"lane_{lane}"
        if pending:
            index, task = pending.popleft()
            sequence = next_sequence[lane]
            next_sequence[lane] += 1
            active[lane] = (sequence, index, task)
            atomic_json(lane_dir / f"assignment_{sequence}.json", {
                "sequence": sequence, "task_index": index, "task_id": task},
                exclusive=True)
        else:
            (lane_dir / "assignment_stop.requested").touch(exist_ok=False)
            stopping.add(lane)

    try:
        for lane in range(min(workers, len(tasks))):
            lane_dir = directory / "lanes" / f"lane_{lane}"
            lane_dir.mkdir(parents=True, exist_ok=False)
            command = command_for_lane(tasks, lane, lane_dir)
            log = (lane_dir / "child.log").open("w", encoding="utf-8")
            try:
                started = time.monotonic_ns()
                first_start = started if first_start is None else min(first_start, started)
                process = popen(command, env=env, cwd=cwd, stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=os.name == "posix")
            except BaseException:
                log.close()
                raise
            running[lane] = (process, log)
            next_sequence[lane] = 0
            atomic_json(lane_dir / "started.json", {
                "lane": lane, "eligible_tasks": tasks, "command": command,
                "dispatch": "dynamic", "start_monotonic_ns": started})
            dispatch(lane)

        while running:
            time.sleep(poll_interval)
            for lane, (process, log) in list(running.items()):
                lane_dir = directory / "lanes" / f"lane_{lane}"
                assigned = active.get(lane)
                if assigned is not None:
                    sequence, index, task = assigned
                    marker = lane_dir / f"assignment_{sequence}.complete.json"
                    if marker.is_file():
                        completed = json.loads(marker.read_text(encoding="utf-8"))
                        if completed != {"sequence": sequence, "task_index": index,
                                         "task_id": task}:
                            raise RuntimeError(f"Persistent C1 lane {lane} returned a different task")
                        path = directory / "tasks" / task / "process.json"
                        row = json.loads(path.read_text(encoding="utf-8"))
                        if row.get("task_id") != task or row.get("returncode") != 0:
                            raise RuntimeError(f"Persistent C1 lane {lane} wrote an invalid task receipt")
                        rows.append(dict(row, task_index=index, lane=lane))
                        del active[lane]
                        if process.poll() is None:
                            dispatch(lane)
                code = process.poll()
                if code is None:
                    continue
                if lane in active or lane not in stopping or code != 0:
                    raise RuntimeError(f"Persistent C1 lane {lane} exited {code} before completion")
                last_end = time.monotonic_ns()
                del running[lane]
                try:
                    stop_owned_group(process)
                finally:
                    log.close()
                atomic_json(lane_dir / "process.json", {
                    "lane": lane, "returncode": code, "end_monotonic_ns": last_end})
            if running:
                monitor()
    except BaseException as failure:
        for process, log in running.values():
            try:
                stop_owned_group(process)
            except Exception as error:
                if hasattr(failure, "add_note"):
                    failure.add_note(f"Additionally, stopping persistent lane failed: {error}")
            finally:
                log.close()
        raise
    if len(rows) != len(tasks):
        raise RuntimeError("Persistent C1 dynamic pool did not complete every task")
    return sorted(rows, key=lambda row: row["task_index"]), (last_end - first_start) / 1e9


def _validate_task_id(task):
    if (not isinstance(task, str) or not task or task != task.strip() or "," in task
            or task in {".", ".."} or "/" in task or "\\" in task):
        raise ValueError(f"Task ID cannot name a task output directory: {task!r}")


def run_pool(tasks, command_for, directory_for, *, workers, env, cwd, monitor,
             poll_interval=0.1, popen=subprocess.Popen):
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
                    start_unix_ns, start = time.time_ns(), time.monotonic_ns()
                    atomic_json(directory / "started.json", {
                        "task_id": task, "lane": lane, "command": command,
                        "start_unix_ns": start_unix_ns, "start_monotonic_ns": start})
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
            for lane in sorted(running):
                entry = running[lane]
                code = entry["process"].poll()
                if code is None:
                    continue
                end_unix_ns, end = time.time_ns(), time.monotonic_ns()
                del running[lane]
                try:
                    stop_owned_group(entry["process"])
                finally:
                    entry["log"].close()
                row = {"task_index": entry["task_index"], "task_id": entry["task_id"],
                             "lane": lane, "returncode": code,
                             "start_unix_ns": entry["start_unix_ns"], "end_unix_ns": end_unix_ns,
                             "start_monotonic_ns": entry["start"], "end_monotonic_ns": end,
                             "wall_seconds": (end - entry["start"]) / 1e9,
                             "output": str(entry["output"])}
                atomic_json(entry["output"] / "process.json", row)
                rows.append(row)
                free.append(lane)
                free.sort()
            # Reap natural completions before a failed liveness check unwinds
            # the pool, so completed tasks keep their process/score receipts.
            if running or pending:
                monitor()
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
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        score = summary.get("semantic_score")
        n = summary.get("n_scored", summary.get("n"))
        if n != 1 or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Expected one scored task and a finite semantic_score in [0, 1]")
    except (OSError, ValueError, AttributeError) as error:
        return dict(row, n_scored=None, semantic_score=None, score_error=str(error))
    return dict(row, n_scored=n, semantic_score=score)


def summarize(cell, workers, max_running_requests, ports, tasks, rows):
    rows = sorted((_task_score(row, cell["arm"]) for row in rows), key=lambda row: row["task_index"])
    scored = [row for row in rows if row["n_scored"] == 1 and row["semantic_score"] is not None]
    runtime = ((max(row["end_monotonic_ns"] for row in rows) - min(row["start_monotonic_ns"] for row in rows)) / 1e9
               if rows else None)
    hours = runtime / 3600 if runtime else None
    return {
        "schema": SCHEMA, "cell_id": cell["cell_id"], "arm": cell["arm"],
        "benchmark": cell["benchmark"], "workers": workers,
        "engine_max_running_requests": max_running_requests, "lane_proxy_ports": ports,
        "measurement_scope": "concurrent_serving",
        "runtime_scope": "first_task_process_start_to_last_observed_process_exit",
        "scope_note": ("Runtime includes per-task proxy/harness startup, tools and scoring; "
                       "engine startup is excluded. This is closed-loop harness throughput, "
                       "not warm request latency. Request durations overlap; pool samples "
                       "are engine-wide and are not per-request or continuous memory peaks."),
        "n_tasks": len(tasks), "n_finished": len(rows),
        "n_failed": sum(row["returncode"] != 0 for row in rows),
        "n_method_failures": sum(row.get("task_status") == "method_failure" for row in rows),
        "n_harness_failures": sum(row.get("task_status") == "harness_failure" for row in rows),
        "n_scored": len(scored),
        "n_unscored": len(tasks) - len(scored),
        "successful_tasks": sum(row["semantic_score"] for row in scored),
        "total_runtime_seconds": runtime,
        "tasks_per_hour": len(scored) / hours if hours else None,
        "successful_tasks_per_hour": sum(row["semantic_score"] for row in scored) / hours if hours else None,
        "status": ("completed" if len(scored) == len(tasks)
                   and all(row.get("task_status") not in {"method_failure", "harness_failure"}
                           for row in rows) else "completed_with_failures"),
        "tasks": rows,
    }


@unwind_on_termination
def serve_cell(config, cell, output, source, profile_path, *, workers, tasks,
               port_offset=0, poll_interval=0.1, popen=subprocess.Popen):
    """Serve every task of one prepared closed-loop cell with ``workers`` lanes."""
    from benchmarks.arms import get_arm, history_kv_spec

    output = Path(output).resolve()
    source = Path(source).resolve()
    profile_path = Path(profile_path).resolve()
    if cell.get("arm") != "full" and not _v4_c2kv_cell(cell):
        raise ValueError("Serving supports only v4 C2KV and Full control cells")
    features = native_serving_features(config, cell)
    persistent_runtime = persistent_runtime_enabled(config, cell)
    dynamic_persistent_runtime = dynamic_persistent_runtime_enabled(config, persistent_runtime)
    incremental = config.get("serving_incremental_tokenization", False)
    if type(incremental) is not bool:
        raise ValueError("serving_incremental_tokenization must be a boolean")
    if incremental and (not _v4_c2kv_cell(cell)
                        or cell.get("history_budget_tokens") != 256):
        raise ValueError("Incremental tokenization supports v4 C2KV B256 only")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if cell["benchmark"] not in SERVING_BENCHMARKS:
        raise ValueError(f"Serving runs support {sorted(SERVING_BENCHMARKS)} cells")
    arm = get_arm(cell["arm"])
    if not is_native_arm(cell["arm"]):
        persistent = arm.history_kv and history_kv_spec(arm)["persistent_session"]
        if not (arm.name == "full" or arm.text_policy or persistent):
            raise ValueError(f"Arm {arm.name!r} does not support a shared engine")
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
    overlap_schedule = serving_overlap_schedule_enabled(config)
    max_running = int(server_cmd[server_cmd.index("--max-running-requests") + 1])
    directory = Path(output) / "serving" / cell["cell_id"] / f"workers_{workers}"
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / "started.json", {
        "schema": SCHEMA, "cell": cell, "config": config, "workers": workers,
        "tasks": list(tasks), "lane_proxy_ports": ports, "server_command": server_cmd,
        "native_serving_features": features,
        "overlap_schedule": overlap_schedule,
        "persistent_runtime": persistent_runtime,
        "dynamic_persistent_runtime": dynamic_persistent_runtime,
        "port_offset": port_offset, "sglang_source": str(source), "time": time.time()},
        exclusive=True)
    for port in (config["server_port"], *ports):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"Configured port {port} is already occupied")
    env = paper_env(config, source)
    # Serve owns these engine and controller switches; a normal paper run keeps
    # its original environment and the Full arm has no gist worker.
    env.update(C2KV_GIST_ASYNC="1" if _v4_c2kv_cell(cell) else "0",
               C2KV_GIST_BATCH_SIZE="1", C2KV_BASE_QUERY_GRAPH="1",
               C2KV_NATIVE_COMPACT_RESPONSE="1",
               C2KV_NATIVE_BULK_FIRST_MISS="1",
               C2KV_NATIVE_PREWARM_FIT_BUDGET="1")
    telemetry_name = ("native_engine_telemetry.jsonl" if is_native_arm(cell["arm"])
                      else "server_telemetry.jsonl")
    env["C2KV_PAPER_TELEMETRY_LOG"] = str(directory / telemetry_name)
    env["C2KV_PAPER_CONCURRENT"] = "1"
    for feature, (_, variable, _) in FEATURES.items():
        env[variable] = "1" if features[feature] is not None else "0"
    env["C2KV_INCREMENTAL_TOKENIZATION"] = "1" if incremental else "0"
    native = is_native_arm(cell["arm"])
    capabilities = None
    with (directory / "server.log").open("w") as log:
        # Own only this process group. Never stop another experiment's server.
        server = popen(server_cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                       start_new_session=True)
        run_failure = None
        try:
            wait_server(server, config["server_port"])
            capabilities = require_native_serving_features(config["server_port"], features)
            monitor = UpstreamLiveness(f"http://127.0.0.1:{config['server_port']}", process=server)
            if get_arm(cell["arm"]).text_history_budget_tokens is not None:
                require_budget_renderer(config["server_port"])
            if persistent_runtime:
                pool = (run_dynamic_persistent_pool if dynamic_persistent_runtime
                        else run_persistent_pool)
                rows, cold_runtime = pool(
                    tasks,
                    lambda lane_tasks, lane, lane_dir: persistent_lane_command(
                        config, cell, lane_tasks, lane_dir, profile_path, lane,
                        directory / "tasks", dynamic=dynamic_persistent_runtime),
                    directory, workers=workers, env=env, cwd=ROOT.parent,
                    monitor=monitor, poll_interval=poll_interval, popen=popen)
            else:
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
                    run_failure = sys.exc_info()
                elif hasattr(run_failure[1], "add_note"):
                    run_failure[1].add_note(f"Additionally, cleanup failed: {error}")
        if run_failure is not None:
            _, error, traceback = run_failure
            # Preserve completed children and the shared ledger even when the
            # engine disappears. An aborted run must not publish a throughput.
            completed = [json.loads(path.read_text(encoding="utf-8"))
                         for path in (directory / "tasks").glob("*/process.json")]
            if persistent_runtime:
                task_indices = {task: index for index, task in enumerate(tasks)}
                completed = [dict(row, task_index=task_indices[row["task_id"]],
                                  lane=(row["lane"] if dynamic_persistent_runtime
                                        else task_indices[row["task_id"]] % workers))
                             for row in completed if row.get("task_id") in task_indices]
            manifest = summarize(cell, workers, max_running, ports, tasks, completed)
            manifest.update(status="aborted", error=f"{type(error).__name__}: {error}",
                            total_runtime_seconds=None, tasks_per_hour=None,
                            successful_tasks_per_hour=None,
                            native_serving_features=features,
                            overlap_schedule=overlap_schedule,
                            persistent_runtime=persistent_runtime,
                            dynamic_persistent_runtime=dynamic_persistent_runtime,
                            incremental_tokenization=incremental,
                            engine_serving_capabilities=capabilities,
                            engine_telemetry=aggregate_engine_ledger(directory / telemetry_name))
            if features['async_compression'] is not None:
                manifest['async_compression_measurement'] = summarize_async_compression(directory)
            atomic_json(directory / "serving.json", manifest)
            raise error.with_traceback(traceback)
    manifest = summarize(cell, workers, max_running, ports, tasks, rows)
    if persistent_runtime:
        manifest["runtime_scope"] = "first_lane_process_start_to_last_lane_process_exit"
        manifest["scope_note"] = (
            "Cold lane runtime includes Python startup, task-local controller initialization, "
            "official harness/tools/scoring, conversion and lane cleanup; shared engine startup "
            "is excluded. Each task has fresh controller, generator, API, session and budgets. "
            f"{'Dynamic' if dynamic_persistent_runtime else 'Static round-robin'} lane "
            "assignment is recorded per task; per-request memory peaks "
            "remain engine-wide under concurrency.")
        manifest["total_runtime_seconds"] = cold_runtime
        hours = cold_runtime / 3600 if cold_runtime else None
        manifest["tasks_per_hour"] = manifest["n_scored"] / hours if hours else None
        manifest["successful_tasks_per_hour"] = manifest["successful_tasks"] / hours if hours else None
    manifest["persistent_runtime"] = persistent_runtime
    manifest["dynamic_persistent_runtime"] = dynamic_persistent_runtime
    manifest["incremental_tokenization"] = incremental
    manifest["native_serving_features"] = features
    manifest["overlap_schedule"] = overlap_schedule
    manifest["engine_serving_capabilities"] = capabilities
    manifest["engine_telemetry"] = aggregate_engine_ledger(directory / telemetry_name)
    if features['async_compression'] is not None:
        manifest['async_compression_measurement'] = summarize_async_compression(directory)
    atomic_json(directory / "serving.json", manifest)
    return manifest
