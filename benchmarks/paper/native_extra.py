"""Official ACEBench Agent and ToolSandbox tasks for native bare C2KV."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import subprocess
import time
from collections.abc import Mapping

from . import c1_appworld
from .process_lifecycle import run_owned


BENCHMARKS = {"acebench_agent": ("acebench", "acebench-text-actions-v1"),
              "toolsandbox": ("toolsandbox", "openai-single-task-v1")}
TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def _task_id(value):
    task = str(value)
    if task in {".", ".."} or not TASK_ID.fullmatch(task):
        raise ValueError(f"Invalid official task identity: {task!r}")
    return task


def _ace_tasks(config):
    from benchmarks.adapters import acebench_adapter as ace

    source = Path(config["acebench_dir"])
    language = str(config.get("acebench_language", "en"))
    if language not in {"en", "zh"}:
        raise ValueError("ACEBench language must be en or zh")
    tests = ace.expand_categories("agent", ace.load_category_map(source))
    if not tests or any(not test.startswith("agent_") for test in tests):
        raise ValueError("ACEBench agent category must contain only agent tests")
    tasks = []
    for test in tests:
        for row in ace._jsonl(source / "data_all" / f"data_{language}" / f"data_{test}.json"):
            tasks.append(_task_id(row["id"]))
    return tasks


def _toolsandbox_tasks(config):
    """Ask the installed official resolver for exactly the configured suite."""
    source = Path(config["toolsandbox_dir"]).resolve()
    scenarios = config.get("toolsandbox_scenarios") or []
    if scenarios and (not isinstance(scenarios, list) or
                      any(not isinstance(item, str) for item in scenarios)):
        raise ValueError("toolsandbox_scenarios must be a list of names")
    if not scenarios and config.get("toolsandbox_suite") != "full":
        raise ValueError("ToolSandbox paper suite requires full or explicit scenarios")
    script = (
        "import json; from tool_sandbox.cli import resolve_scenarios; "
        "print(json.dumps(sorted(resolve_scenarios(desired_scenario_names=None, "
        "preferred_tool_backend='DEFAULT'))))"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(source), env.get("PYTHONPATH"))))
    completed = run_owned(
        [config.get("toolsandbox_python", config["bench_python"]), "-c", script],
        cwd=source, env=env, check=True, capture_output=True, text=True,
    )
    try:
        available = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Official ToolSandbox resolver did not return JSON") from error
    if not isinstance(available, list):
        raise RuntimeError("Official ToolSandbox resolver returned no scenario list")
    available = [_task_id(item) for item in available]
    if scenarios:
        if len(scenarios) != len(set(scenarios)) or not set(scenarios) <= set(available):
            raise ValueError("Configured ToolSandbox scenarios are not unique official IDs")
        return list(scenarios)
    return available


def selected_tasks(config, benchmark, requested=None):
    if benchmark == "acebench_agent":
        available = _ace_tasks(config)
    elif benchmark == "toolsandbox":
        available = _toolsandbox_tasks(config)
    else:
        raise ValueError(f"Unsupported native bare benchmark: {benchmark}")
    if not available or len(available) != len(set(available)):
        raise ValueError("Official task selection is empty or contains duplicate IDs")
    if requested:
        wanted = [_task_id(item) for item in requested]
        if len(wanted) != len(set(wanted)) or not set(wanted) <= set(available):
            raise ValueError("Requested tasks must be unique IDs in the official split")
        return wanted
    return available


def replay_payload(payload, task, step):
    """Bind a recorded ACE prefix only when it carries real source receipts."""
    _task_id(task)
    if type(step) is not int or step < 0 or not isinstance(payload, Mapping):
        raise ValueError("ACEBench replay requires one nonnegative step and request object")
    source = payload.get("c2kv_ace_source")
    if (not isinstance(source, Mapping)
            or source.get("version") != "acebench-text-actions-v1"
            or not isinstance(source.get("receipts"), list)):
        raise ValueError("ACEBench native replay requires recorded execution receipts")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("ACEBench native replay requires recorded actor messages")
    if (sum(isinstance(row, Mapping) and row.get("role") == "tool" for row in messages)
            != len(source["receipts"])):
        raise ValueError("ACEBench native replay receipt coverage differs from the visible prefix")
    if (payload.get("temperature") != 0.001 or payload.get("top_p") != 1
            or payload.get("max_tokens") != 1000):
        raise ValueError("ACEBench native replay sampling differs from the official scene")
    result = copy.deepcopy(dict(payload))
    if result.get("stream") is False:
        result.pop("stream")
    result.pop("c2kv_measurement_session_id", None)
    result["store"] = False
    result["c2kv_eval_context"] = {
        "benchmark": "acebench", "task_id": task,
        "user_turn": max(0, sum(row.get("role") == "user" for row in messages) - 1),
        "step": step, "attempt": 0,
    }
    return result


def server_command(config, benchmark, task, native, delivery, controller_path):
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unsupported native bare benchmark: {benchmark}")
    from experiments.history_system.native_bare import configure_design

    delivery_root = c1_appworld._delivery_path(delivery)
    design = json.loads((delivery_root / "configs" / "current_algorithm.json").read_text(encoding="utf-8"))
    design = configure_design(design)
    design["runtime"].update(
        sglang_backend_url=c1_appworld._sglang_upstream(config),
        device="cpu", npu_allocator_metrics=False,
    )
    if benchmark == "acebench_agent":
        # ACEBench's scene agent constructors use 1000 even when generate.py
        # receives a larger outer handler cap. Bind the actually sent request.
        design["sampling"]["max_completion_tokens"] = 1000
    runner = c1_appworld._delivery_runner(delivery)
    runtime_name, source_profile = BENCHMARKS[benchmark]
    return runner.server_command(
        design, task_id=_task_id(task),
        checkpoint=str(Path(config["checkpoint"]).resolve()),
        output=str(Path(native).resolve()),
        port=c1_appworld._controller_port(config),
        python=c1_appworld._controller_python(config),
        benchmark=runtime_name, source_profile=source_profile,
    )


controller_command = server_command


def _run_official(config, benchmark, task, task_out, base_url, model):
    user_url = c1_appworld._sglang_upstream(config)
    if benchmark == "acebench_agent":
        from benchmarks.adapters import acebench_adapter as ace

        old = os.environ.get("C2KV_ACE_NATIVE")
        os.environ["C2KV_ACE_NATIVE"] = "1"
        try:
            summary = ace.run_acebench(
                base_url, user_url, task_out / "acebench",
                acebench_dir=Path(config["acebench_dir"]), category="agent",
                language=config.get("acebench_language", "en"),
                model=model, user_model=config["model"], num_threads=1,
                task_ids=task,
                python=config.get("acebench_python", config["bench_python"]),
            )
        finally:
            if old is None:
                os.environ.pop("C2KV_ACE_NATIVE", None)
            else:
                os.environ["C2KV_ACE_NATIVE"] = old
        selected = summary.get("selection") or {}
        selected_ids = [value for source in selected.get("sources", [])
                        for value in source.get("selected_ids", [])]
        if summary.get("n") != 1 or selected_ids != [task]:
            raise RuntimeError("Official ACEBench task selection or scoring differs from the frozen task")
        namespace = "acebench"
        scorer = "ACEBench eval_main.py"
    else:
        from benchmarks.adapters import toolsandbox_adapter as ts

        summary = ts.run_ts(
            base_url, task_out / "toolsandbox", test_mode=False,
            agent=ts.AGENT, user=ts.AGENT, user_base_url=user_url,
            scenarios=[task], benchmark_dir=Path(config["toolsandbox_dir"]),
            python=config.get("toolsandbox_python", config["bench_python"]),
            parallel=1, model=model, user_model=config["model"], expected=1,
        )
        if summary.get("n") != 1 or summary.get("scenario_ids") != [task]:
            raise RuntimeError("Official ToolSandbox did not score the frozen scenario")
        namespace = "toolsandbox"
        scorer = "tool_sandbox official CLI"
    score = summary.get("semantic_score")
    if type(score) not in (int, float):
        raise RuntimeError("Official harness did not return one numeric semantic score")
    official = {
        "schema": "paper-native-extra-official-task-v1",
        "benchmark": namespace, "task_id": task, "official_scorer": scorer,
        "n": 1, "task_rows": [{"task_id": task, "semantic_score": float(score),
                               "normal_termination": True, "protocol_legal": None}],
        "adapter_summary": summary,
    }
    out = task_out / namespace / "official_summary.json"
    out.write_text(json.dumps(official, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return official


def run_task(config, benchmark, task, native, delivery, controller_path):
    """Run one official task against a one-task native event server."""
    task = _task_id(task)
    command = server_command(config, benchmark, task, native, delivery, controller_path)
    delivery_root = c1_appworld._delivery_path(delivery)
    runtime = delivery_root / "runtime"
    runner = c1_appworld._delivery_runner(delivery)
    task_out = Path(native).resolve() / "task_shards" / task
    task_out.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(runtime / "python"), str(runtime), env.get("PYTHONPATH"))))
    timeout = float(config.get("c1", {}).get("task_timeout", 10800))
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    process = None
    try:
        with (task_out / "controller.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=runtime, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            ready = c1_appworld._wait_ready(process, task_out / "server" / "ready.json", deadline)
            official = _run_official(config, benchmark, task, task_out,
                                     str(ready["base_url"]),
                                     command[command.index("--model-name") + 1])
    finally:
        if process is not None:
            runner._stop_server(process, task_out / "server.supervisor.json")
    final_path = task_out / "server" / "final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    journal = final.get("journal_summary") or {}
    if (final.get("status") == "failed" or final.get("cost_summary_error")
            or journal.get("failed") or journal.get("pending")
            or not journal.get("completed") or process.returncode != 0):
        raise RuntimeError(f"Native bare controller finalization failed; see {final_path}")
    run_c1 = c1_appworld._delivery_run_c1(delivery, runner)
    namespace = BENCHMARKS[benchmark][0]
    metrics = run_c1.summarize_task(namespace, task, task_out, official,
                                    time.monotonic() - started)
    acceptance = run_c1.functional_checks("c2kv_native", "disabled", metrics)
    if not all(acceptance["required"].values()):
        raise RuntimeError(f"Native bare functional acceptance failed: {acceptance['required']}")
    return ({"task_id": task, "status": "completed", "official_summary": official,
             "unified_metrics": metrics,
             "qualification": "official single-task native event result"}, metrics)
