"""Paper-matrix orchestration for the selected H0/C1000/R1 controller.

The delivered runtime owns packing, retrieval, detector scoring and regeneration.
This module only selects the official tasks and connects their artifacts to the
paper measurement contract.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error

from benchmarks.measurement.telemetry import append_jsonl, canonical_sha256, read_jsonl
from benchmarks.measurement.replay import _paper_measurement
from .candidate_matrix import ARM_TO_VARIANT, SUPPORTED_BENCHMARKS as CANDIDATE_BENCHMARKS
from benchmarks.arms import get_arm
from benchmarks.native_history_budget import NativeHistoryBudget
from .process_lifecycle import defer_termination, unwind_on_termination

ARMS = {"c2kv_c1_t02_r8": 8, "c2kv_c1_t02_r4": 4, "c2kv_c1_off_r8": 8}
ARMS["c2kv_native_r4"] = 4
ARMS.update({arm: 8 for arm in ARM_TO_VARIANT})
ARM = "c2kv_c1_t02_r8"
RATIO = ARMS[ARM]


def select_arm(arm):
    """Bind this process to one native C1 arm (default: the final ratio-8 system)."""
    global ARM, RATIO
    if arm not in ARMS:
        raise ValueError(f"Unknown native C1 arm {arm!r}; expected one of {sorted(ARMS)}")
    ARM, RATIO = arm, ARMS[arm]
    return ARM, RATIO


ROOT = Path(__file__).resolve().parents[2]
DELIVERY = ROOT / "experiments" / "history_system"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_delivery():
    # The delivery's wrapper intentionally uses sibling modules, while its
    # runtime executes in a separate process with its own benchmarks namespace.
    directory = str(DELIVERY)
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import benchmarks
    runtime_benchmarks = str(DELIVERY / "runtime" / "benchmarks")
    if runtime_benchmarks not in benchmarks.__path__:
        benchmarks.__path__.append(runtime_benchmarks)
    spec = importlib.util.spec_from_file_location("paper_c1_delivery", DELIVERY / "run_c1.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def delivery_args(config, benchmark, output, task_ids, delivery):
    settings = config.get("c1", {})
    detector = settings.get("detector", "d3_hybrid")
    benchmark_dir = config["tau2_dir"] if benchmark == "tau2" else config["bfcl_dir"]
    if ARM in ARM_TO_VARIANT and benchmark not in CANDIDATE_BENCHMARKS:
        raise ValueError("candidate arms support bfcl_base, bfcl_long_context, appworld and acebench_agent")
    command = [
        "--method", ("c2kv_native" if ARM == "c2kv_native_r4" else
                     "c2kv_only" if ARM == "c2kv_c1_off_r8" else "proposed"),
        "--checkpoint", config["checkpoint"],
        "--sglang-backend-url", config.get("upstream") or f"http://127.0.0.1:{config['server_port']}",
        "--embedding-model", settings.get("embedding_model", "unused-native-bare"),
        "--embedding-device", settings.get("embedding_device", "cpu"),
        "--selector-threshold", str(settings.get("selector_threshold", 0.5)),
        "--benchmark-dir", benchmark_dir,
        "--bfcl-python", config["bench_python"],
        "--sglang-root", config["sglang_source"],
        "--portable-root", str(ROOT),
        "--port", str(config["proxy_port"]),
        "--task-timeout", str(settings.get("task_timeout", 10800)),
        "--out", str(output),
        "--ratio", str(RATIO),
    ]
    if ARM in ARM_TO_VARIANT:
        command += ["--candidate-algorithm", ARM_TO_VARIANT[ARM]]
    else:
        command += ["--detector", detector]
        if ARM != "c2kv_native_r4" and detector in {"t02_risk", "legacy_prefill"}:
            command += ["--embedding-batch-size", str(settings.get("embedding_batch_size", 1))]
    if "native_history_budget_tokens" in config:
        budget = NativeHistoryBudget(config["native_history_budget_tokens"])
        budget.validate_arm(get_arm(ARM))
        if benchmark not in {"bfcl_base", "bfcl_long_context"}:
            raise ValueError("Native history budget sweep currently supports BFCL only")
        command += budget.cli_args()
    if benchmark == "tau2":
        command += ["--tau2-dir", config["tau2_dir"],
                    "--tau2-python", config.get("tau2_python", config["bench_python"]),
                    "--task-set", config.get("tau2_task_set", "airline"),
                    "--user-base-url", config.get("upstream") or f"http://127.0.0.1:{config['server_port']}"]
        for task_id in task_ids:
            command += ["--tau2-task-id", task_id]
        if config.get("tau2_max_steps") is not None:
            command += ["--tau2-max-steps", str(config["tau2_max_steps"])]
    if config.get("tool_memory"):
        command += ["--tool-memory", config["tool_memory"]]
        if config.get("tool_checkpoint"):
            command += ["--tool-checkpoint", config["tool_checkpoint"]]
        if config.get("tool_budget_tokens") is not None:
            command += ["--tool-budget-tokens", str(config["tool_budget_tokens"])]
    args = delivery.build_parser().parse_args(command)
    args.benchmark = ("acon_appworld" if benchmark == "appworld"
                      else "acebench" if benchmark == "acebench_agent"
                      else "tau2" if benchmark == "tau2" else "bfcl")
    args.task_id = list(task_ids)
    return args


def selected_tasks(config, benchmark, requested=None):
    if benchmark in {"acebench_agent", "toolsandbox", "tau2"}:
        from .native_extra import selected_tasks as extra_tasks
        return extra_tasks(config, benchmark, requested)
    if benchmark == "appworld":
        from .c1_appworld import task_ids
        available = task_ids(config)
    else:
        category = {"bfcl_base": "multi_turn_base", "bfcl_long_context": "multi_turn_long_context"}[benchmark]
        data_dir = Path(config["bfcl_dir"]) / "bfcl_eval" / "data"
        paths = list(data_dir.glob(f"BFCL_*_{category}.json"))
        if len(paths) != 1:
            raise ValueError(f"Expected one official {category} file in {data_dir}, found {paths}")
        content = paths[0].read_text(encoding="utf-8")
        try:
            rows = json.loads(content)
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        available = [row["id"] for row in rows]
    if len(available) != len(set(available)):
        raise ValueError("Official task selection contains duplicate IDs")
    if requested:
        if len(requested) != len(set(requested)) or not set(requested) <= set(available):
            raise ValueError("Requested tasks must be unique IDs in the official split")
        return list(requested)
    return available


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def method_label():
    if ARM == "c2kv_c1_off_r8":
        return "C1 initial allocation (recovery off)"
    if ARM in ARM_TO_VARIANT:
        return f"C2KV {ARM_TO_VARIANT[ARM]}"
    return "C2KV" if ARM == "c2kv_native_r4" else "C2KV+C1"


def summarize_scores(benchmark, receipts):
    if benchmark != "appworld":
        scores = [row["unified_metrics"]["official_score"] for row in receipts]
        failures = [row["task_id"] for row in receipts if row.get("status") == "harness_failure"]
        infeasible = [row["task_id"] for row in receipts if row.get("status") == "method_failure"]
        return {"arm": ARM, "method": method_label(), "ratio": RATIO,
                "n_scored": len(scores), "n": len(scores),
                "semantic_score": sum(scores) / len(scores) if scores else None,
                "n_harness_failures": len(failures), "harness_failure_task_ids": failures,
                "n_method_failures": len(infeasible), "method_failure_task_ids": infeasible,
                "task_rows": receipts, "result_status": "preliminary, n=1"}
    from .c1_appworld import summarize_scores as appworld_scores
    result = appworld_scores(receipts)
    result.update(arm=ARM, ratio=RATIO, method=method_label())
    return result


def prepare_native(config, benchmark, directory, tasks, delivery):
    native = directory / "native"
    native.mkdir(parents=True, exist_ok=True)
    args = delivery_args(config, benchmark, native, tasks, delivery)
    controller, profile = delivery.build_profile(args)
    profile.update(arm=ARM, benchmark=benchmark, task_ids=tasks,
                   paper_actor_model=config["model"],
                   source_delivery=delivery.source_revision(ROOT),
                   comparison=("final system ratio8; bare C2KV ratio4 is not a detector-only ablation"
                               if RATIO == 8 else
                               "ratio-4 ablation of the final system: same controller, same ratio as bare C2KV"))
    if ARM == "c2kv_native_r4":
        profile["comparison"] = "Independent native static gist baseline; not a detector-only C1 ablation"
    elif ARM == "c2kv_c1_off_r8":
        profile["comparison"] = "Same C1 ratio8 initial history allocation, recovery disabled"
    elif ARM in ARM_TO_VARIANT:
        profile["comparison"] = "Explicit ratio-8 candidate; not a legacy C1 or D3 score"
    previous_profile = native / "profile.json"
    if previous_profile.is_file():
        previous = json.loads(previous_profile.read_text(encoding="utf-8"))
        if previous.get("native_history_budget") != profile.get("native_history_budget"):
            raise ValueError("Native history budget changed; use a separate cell output directory")
    profile["sglang_backend_preflight"] = delivery.preflight_sglang_backend(args)
    controller_path = native / "controller.json"
    save(controller_path, controller)
    save(native / "profile.json", profile)
    return native, args, controller_path


def run_closed_loop(config, benchmark, directory, requested=None):
    delivery = load_delivery()
    tasks = selected_tasks(config, benchmark, requested)
    native, args, controller_path = prepare_native(config, benchmark, directory, tasks, delivery)
    receipts = []
    for task in tasks:
        task_root = native / "task_shards" / task
        receipt_path = task_root / "paper_task_result.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") not in ("completed", "harness_failure", "method_failure"):
                raise RuntimeError(f"Task has a previous non-completed result; not rerunning {task}")
        else:
            if task_root.exists():
                raise RuntimeError(f"Task already has execution evidence; not rerunning {task}")
            print(json.dumps({"arm": ARM, "task": task, "status": "running"}), flush=True)
            try:
                if benchmark in {"acebench_agent", "toolsandbox", "tau2"}:
                    from .native_extra import run_task
                    receipt, metrics = run_task(config, benchmark, task, native, delivery, controller_path)
                elif benchmark == "appworld":
                    from .c1_appworld import run_task
                    receipt, metrics = run_task(config, task, native, delivery, controller_path)
                else:
                    receipt, metrics = delivery.run_task(
                        args, task, controller_path, termination_guard=defer_termination)
            except (RuntimeError, subprocess.CalledProcessError) as error:
                final_path = task_root / "server" / "final.json"
                if (final_path.is_file()
                        and json.loads(final_path.read_text(encoding="utf-8")).get("cost_summary_error")):
                    raise
                failure = controller_step_failure(task_root)
                if failure is None:
                    raise
                # Keep the evidence, score the task 0 and go on; the summary
                # carries the counts so the cell is never read as clean.
                status, kind, message = failure
                receipt = {"task_id": task, "status": status,
                           "failure": {"kind": kind, "message": message, "error": str(error)},
                           "qualification": ("harness failure: CUDA OOM in the C1 controller; "
                                             "scored 0, not a model decision") if kind == "cuda_oom"
                           else ("method failure: the controller declared this input infeasible "
                                 "under its budget; scored 0")}
                metrics = {
                    "task_id": task,
                    "official_score": 0.0,
                    "normal_termination": False,
                    "protocol_legal": None,
                    status: kind,
                }
                print(json.dumps({"arm": ARM, "task": task, "status": status, "kind": kind}),
                      flush=True)
            if ARM == "c2kv_native_r4" and receipt.get("status") == "completed":
                import native_bare
                receipt["native_bare_route"] = native_bare.validate_manifest(task_root / "server" / "ready.json")
            receipt["unified_metrics"] = metrics
            save(receipt_path, receipt)
        receipts.append(receipt)
        save(directory / f"summary_{ARM}.json", summarize_scores(benchmark, receipts))
        save(native / "result.json", {"status": "running", "tasks": receipts})
    save(native / "result.json", {"status": "completed", "tasks": receipts})
    return native


# Controller step errors a cell survives with a scored-0 receipt.  cuda_oom is a
# harness limit (the controller process ran out of GPU memory next to the
# server); capacity_infeasible is the method's own admission decision (mandatory
# raw input plus the minimum whole-event gist exceed its declared budget).
TOLERATED_STEP_ERRORS = {"OutOfMemoryError": ("harness_failure", "cuda_oom")}


def _capacity_session_id(task_root):
    """Bind a typed capacity failure to this task's per-task native server."""
    try:
        ready = json.loads((task_root / "server" / "ready.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(ready, dict)
            or ready.get("schema") != "a-event-native-server-v1"
            or ready.get("status") != "ready"
            or ready.get("allowed_task_ids") != [task_root.name]
            or not isinstance(ready.get("benchmark"), str)
            or not ready["benchmark"]):
        return None
    return f"{ready['benchmark']}/{task_root.name}/attempt-0"


def controller_step_failure(task_root):
    """(status, kind, message) for the latest failed controller step, else None."""
    steps = task_root / "server" / "steps.jsonl"
    if not steps.is_file():
        return None
    capacity_session_id = _capacity_session_id(task_root)
    rows = list(read_jsonl(steps))
    for row in rows[-1:]:
        if row.get("status") != "failed":
            return None
        error = row.get("error")
        text = json.dumps(error) if isinstance(error, dict) else str(error or "")
        if (capacity_session_id is not None
                and row.get("schema") in {"a-event-native-exact-step-v1", "a-acebench-event-step-v1"}
                and row.get("status") == "failed"
                and row.get("session_id") == capacity_session_id
                and row.get("failure_kind") == "method_failure"
                and row.get("failure_code") == "c2kv_capacity_infeasible"
                and isinstance(error, dict)
                and error.get("type") == "CapacityInfeasible"):
            return "method_failure", "capacity_infeasible", text[:2000]
        for marker, (status, kind) in TOLERATED_STEP_ERRORS.items():
            if marker in text:
                return status, kind, text[:2000]
    return None


def controller_oom_message(task_root):
    """The CUDA OOM message recorded by the controller's step journal, else None."""
    failure = controller_step_failure(task_root)
    return failure[2] if failure and failure[1] == "cuda_oom" else None


def _controller_process(command, native, task, delivery):
    task_root = native / "task_shards" / task
    task_root.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(delivery.RUNTIME / "python"), str(delivery.RUNTIME)))
    log = (task_root / "controller.log").open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=delivery.RUNTIME, env=env,
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ready = task_root / "server" / "ready.json"
    try:
        deadline = time.monotonic() + 600
        while not ready.is_file():
            if process.poll() is not None:
                raise RuntimeError(f"C1 replay controller exited: {task_root / 'controller.log'}")
            if time.monotonic() > deadline:
                raise TimeoutError("C1 replay controller readiness timeout")
            time.sleep(0.25)
    except BaseException:
        delivery.runner._stop_server(process, task_root / "server.supervisor.json")
        log.close()
        raise
    return process, log, task_root


def replay_task_id(rows, conversation):
    """Keep official episode identity, which is also rendered in evidence IDs."""
    if any("ace_official_task_id" in row for row in rows):
        official_ids = [row.get("ace_official_task_id") for row in rows]
        official_id = official_ids[0]
        if (not isinstance(official_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", official_id)
                or any(value != official_id for value in official_ids)):
            raise ValueError("Recorded ACE prefixes require one consistent official task ID")
        return official_id
    identities = {row["replay_payload"].get("c2kv_measurement_session_id") for row in rows}
    if len(identities) != 1:
        raise ValueError("One recorded conversation has inconsistent episode identities")
    identity = identities.pop()
    if isinstance(identity, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", identity):
        return identity
    return "replay_" + hashlib.sha256(conversation.encode()).hexdigest()[:16]


def replay_sampling_receipt(source, target):
    fields = ("temperature", "top_p", "seed", "max_tokens", "max_completion_tokens")
    source_values = {key: source[key] for key in fields if key in source}
    target_values = {key: target[key] for key in fields if key in target}
    return {
        "source": source_values,
        "target": target_values,
        "changed": source_values != target_values,
        "policy": "recorded Full messages with the target arm's native sampling contract",
    }


def validate_replay_finalization(task_root, *, declared_failure=False):
    """Do not accept successful HTTP responses with invalid persisted costs."""
    final_path = task_root / "server" / "final.json"
    # A declared OOM can terminate the controller before it writes final.json.
    # Keep that existing failed-prefix policy, but never hide a cost error.
    if declared_failure and not final_path.is_file():
        return
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if final.get("cost_summary_error"):
        raise RuntimeError(f"Native replay cost finalization failed; see {final_path}")
    if declared_failure:
        return
    journal = final.get("journal_summary") or {}
    if (final.get("status") != "stopped" or final.get("error")
            or journal.get("failed") or journal.get("pending")
            or not journal.get("completed") or not final.get("cost_summary")):
        raise RuntimeError(f"Native replay controller finalization failed; see {final_path}")


def run_common_prefix(config, benchmark, directory, prefix_path):
    """Teacher-force recorded observations; never execute or score replay drafts."""
    records = [row for row in read_jsonl(prefix_path) if row.get("event_type") == "recorded_prefix"]
    if not records:
        raise ValueError("No recorded Full prefixes")
    groups = {}
    for row in records:
        if row.get("source_arm") != "full" or canonical_sha256(row["replay_payload"]) != row["canonical_sha256"]:
            raise ValueError("Invalid recorded Full-prefix source")
        if config.get("tool_memory") and benchmark in {"acebench_agent", "appworld"}:
            from benchmarks.measurement.replay import validate_tool_source_capture
            validate_tool_source_capture(row["replay_payload"], benchmark)
        key = row.get("conversation_id") or row["replay_payload"].get("c2kv_measurement_session_id")
        if not key:
            raise ValueError("C1 replay requires recorded conversation identity")
        groups.setdefault(str(key), []).append(row)
    delivery = load_delivery()
    tasks = [replay_task_id(rows, key) for key, rows in groups.items()]
    if len(tasks) != len(set(tasks)):
        raise ValueError("Recorded conversations reuse an episode identity")
    if benchmark == "tau2" and set(tasks) != set(selected_tasks(config, benchmark)):
        raise ValueError("tau2 replay must cover exactly the selected official task identities")
    native, args, controller_path = prepare_native(config, benchmark, directory, tasks, delivery)
    sequence = 0
    tolerated_failures = []   # (task, prefix_id, status, kind): failed replay prefixes
    declared_failure_tasks = []  # one server-declared incident per task
    unattempted_prefixes = 0
    for task, rows in zip(tasks, groups.values()):
        if benchmark in {"acebench_agent", "toolsandbox", "tau2"}:
            from .native_extra import controller_command
            command = controller_command(config, benchmark, task, native, delivery, controller_path)
        elif benchmark == "appworld":
            from .c1_appworld import controller_command
            command = controller_command(config, task, native, delivery, controller_path)
        else:
            command, _ = delivery.commands_for_task(args, task, controller_path)
        process, log, task_root = _controller_process(command, native, task, delivery)
        previous_user_turn, turn_step = None, -1
        try:
            if benchmark in {"acebench_agent", "toolsandbox", "tau2"}:
                from .native_extra import validate_ready_manifest
                validate_ready_manifest(config, benchmark, task,
                                        task_root / "server" / "ready.json", controller_path)
            elif ARM == "c2kv_native_r4":
                import native_bare
                native_bare.validate_manifest(task_root / "server" / "ready.json")
            if config.get("tool_memory") and benchmark not in {"acebench_agent", "toolsandbox", "tau2"}:
                from .native_extra import validate_tool_ready
                ready = json.loads((task_root / "server" / "ready.json").read_text(encoding="utf-8"))
                validate_tool_ready(config, ready)
            for step, row in enumerate(rows):
                payload = copy.deepcopy(row["replay_payload"])
                payload["model"] = command[command.index("--model-name") + 1]
                payload.pop("c2kv_measurement_session_id", None)
                if benchmark in {"bfcl_base", "bfcl_long_context"}:
                    # Adapt OpenAI transport aliases without changing the
                    # recorded prompt or generation budget.
                    if payload.get("stream") is False:
                        payload.pop("stream")
                    if "max_tokens" in payload:
                        cap = payload.pop("max_tokens")
                        if payload.get("max_completion_tokens", cap) != cap:
                            raise ValueError("Conflicting recorded generation caps")
                        payload["max_completion_tokens"] = cap
                    payload.setdefault("store", False)
                    # Full recorded BFCL's harness sampling (temperature 0.001); the
                    # native arm serves greedy only and its API rejects anything else.
                    # Replay under the arm's own contract; the original payload is kept
                    # verbatim in the prefix_replay row for traceability.
                    payload["temperature"] = 0.0
                    user_turn = max(0, sum(m.get("role") == "user" for m in payload["messages"]) - 1)
                    turn_step = turn_step + 1 if user_turn == previous_user_turn else 0
                    previous_user_turn = user_turn
                    payload["c2kv_eval_context"] = {
                        "benchmark": "bfcl", "task_id": task, "attempt": 0,
                        "user_turn": user_turn, "step": turn_step,
                    }
                elif benchmark == "acebench_agent":
                    from .native_extra import replay_payload
                    payload = replay_payload(payload, task, step,
                                             tool_memory=bool(config.get("tool_memory")))
                if payload.get("messages") != row["replay_payload"].get("messages"):
                    raise ValueError("Native replay changed the recorded Full message prefix")
                sampling_receipt = replay_sampling_receipt(row["replay_payload"], payload)
                request = urllib.request.Request(
                    f"http://127.0.0.1:{config['proxy_port']}/v1/chat/completions",
                    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                started = time.perf_counter_ns()
                unix = time.time_ns()
                try:
                    with OPENER.open(request, timeout=config["c1"]["task_timeout"]) as response:
                        result = json.load(response)
                except urllib.error.HTTPError as error:
                    detail = error.read().decode("utf-8", "replace")
                    failure = controller_step_failure(task_root)
                    if error.code < 500 and not (
                        error.code == 422 and failure is not None
                        and failure[1] == "capacity_infeasible"
                    ):
                        failure = None
                    if failure is None:
                        raise RuntimeError(f"C1 replay HTTP {error.code}: {detail}") from error
                    # The same declared failures the closed loop scores as zero (CUDA OOM,
                    # CapacityInfeasible) end this task's server; record the prefix and the
                    # rest of the task as failed instead of losing the whole replay cell.
                    status, kind, message = failure
                    declared_failure_tasks.append((task, status, kind))
                    unattempted_prefixes += len(rows) - step - 1
                    for skipped, later in enumerate(rows[step:]):
                        append_jsonl(directory / "prefix_replay.jsonl", {
                            "schema": "c2kv.prefix_replay.v1", "event_type": "prefix_replay",
                            "source_run_id": benchmark + "__full", "target_run_id": benchmark + "__" + ARM,
                            "sequence": sequence, "prefix_id": later["prefix_id"],
                            "canonical_sha256": later["canonical_sha256"], "native_task_id": task,
                            "native_step": step + skipped, "request_id": None,
                            "start_unix_ns": unix if skipped == 0 else None,
                            "duration_ns": (time.perf_counter_ns() - started) if skipped == 0 else None,
                            "http_status": error.code if skipped == 0 else None,
                            "request": later["replay_payload"], "response": None,
                            "source_paper_measurement": _paper_measurement(later.get("source_response")),
                            "sampling_contract": (sampling_receipt if skipped == 0 else None),
                            "replay_attempted": skipped == 0,
                            "error": detail if skipped == 0 else f"task server terminated after {kind}",
                            "failure": {"status": status, "kind": kind, "message": message},
                            "teacher_forced": True, "external_actions_executed": 0,
                        })
                        tolerated_failures.append((task, later["prefix_id"], status, kind))
                        sequence += 1
                    break
                append_jsonl(directory / "prefix_replay.jsonl", {
                    "schema": "c2kv.prefix_replay.v1", "event_type": "prefix_replay",
                    "source_run_id": benchmark + "__full", "target_run_id": benchmark + "__" + ARM,
                    "sequence": sequence, "prefix_id": row["prefix_id"],
                    "canonical_sha256": row["canonical_sha256"], "native_task_id": task,
                    "native_step": step, "request_id": result.get("id"),
                    "start_unix_ns": unix, "duration_ns": time.perf_counter_ns() - started,
                    "http_status": 200, "request": row["replay_payload"], "response": result,
                    "source_paper_measurement": _paper_measurement(row.get("source_response")),
                    "sampling_contract": sampling_receipt,
                    "replay_attempted": True,
                    "error": None, "teacher_forced": True, "external_actions_executed": 0,
                })
                sequence += 1
        finally:
            delivery.runner._stop_server(process, task_root / "server.supervisor.json")
            log.close()
        validate_replay_finalization(
            task_root, declared_failure=any(item[0] == task for item in declared_failure_tasks))
    kinds = {}
    for _, _, status, kind in tolerated_failures:
        kinds[f"{status}/{kind}"] = kinds.get(f"{status}/{kind}", 0) + 1
    save(directory / "replay_summary.json", {
        "prefixes": len(records), "completed": len(records) - len(tolerated_failures),
        "failed": len(tolerated_failures), "failed_tasks": sorted({t for t, *_ in tolerated_failures}),
        "declared_failure_tasks": len(declared_failure_tasks),
        "attempted_prefixes": len(records) - unattempted_prefixes,
        "not_attempted_after_declared_failure": unattempted_prefixes,
        "failure_kinds": kinds,
        "failure_kind_units": "prefixes, including later prefixes not attempted after task failure",
        "policy": "prefixes of a task whose native server declared a tolerated failure (cuda_oom, "
                  "capacity_infeasible) are recorded as failed; any other error fails the cell",
    })
    return native


def apply_tool_cli(config, tool_memory, tool_checkpoint, tool_budget_tokens):
    from benchmarks.toolmemory import parse_tool_memory_spec

    spec = parse_tool_memory_spec(tool_memory)
    if spec is None:
        if tool_checkpoint or tool_budget_tokens is not None:
            raise ValueError("Tool options require active --tool-memory")
        if tool_memory:
            for key in ("tool_memory", "tool_checkpoint", "tool_budget_tokens"):
                config.pop(key, None)
        return
    config["tool_memory"] = tool_memory
    config["tool_checkpoint"] = tool_checkpoint
    if tool_budget_tokens is not None:
        config["tool_budget_tokens"] = tool_budget_tokens


@unwind_on_termination
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=sorted(ARMS), default="c2kv_c1_t02_r8")
    parser.add_argument("--benchmark", choices=("bfcl_base", "bfcl_long_context", "appworld", "acebench_agent", "toolsandbox", "tau2"), required=True)
    parser.add_argument("--stage", choices=("closed_loop", "common_prefix"), default="closed_loop")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--upstream")
    parser.add_argument("--proxy-port", type=int)
    parser.add_argument("--num-workers", type=int, choices=(1,), default=1)
    parser.add_argument("--task-ids", help="comma-separated official IDs for a bounded smoke/subset")
    parser.add_argument("--prefixes", type=Path)
    parser.add_argument("--tool-memory", default="")
    parser.add_argument("--tool-checkpoint", default="")
    parser.add_argument("--tool-budget-tokens", type=int)
    parser.add_argument("--history-budget-tokens", type=int)
    args = parser.parse_args(argv)
    select_arm(args.arm)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["native_arm"] = ARM
    if args.history_budget_tokens is not None:
        budget = NativeHistoryBudget(args.history_budget_tokens)
        budget.validate_arm(get_arm(ARM))
        if args.benchmark not in {"bfcl_base", "bfcl_long_context"}:
            parser.error("Native history budget sweep currently supports BFCL only")
        config["native_history_budget_tokens"] = budget.target_tokens
    apply_tool_cli(config, args.tool_memory, args.tool_checkpoint, args.tool_budget_tokens)
    if args.upstream:
        config["upstream"] = args.upstream
    if args.proxy_port:
        config["proxy_port"] = args.proxy_port
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage == "common_prefix":
        if args.prefixes is None:
            parser.error("common_prefix requires --prefixes")
        native = run_common_prefix(config, args.benchmark, args.out, args.prefixes)
    else:
        native = run_closed_loop(config, args.benchmark, args.out,
                                 args.task_ids.split(",") if args.task_ids else None)
    from benchmarks.measurement.c1 import convert_run
    conversion = convert_run(native, args.out, benchmark=args.benchmark, arm=ARM,
                             replay=args.stage == "common_prefix")
    save(args.out / "c1_conversion.json", conversion)
    print(json.dumps({"arm": ARM, "status": "completed", "output": str(args.out)}), flush=True)


if __name__ == "__main__":
    main()
