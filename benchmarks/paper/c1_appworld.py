"""Run the delivered native C1 controller through the official AppWorld harness.

The ACON code owns the prompt, actor loop, AppWorld state, and official scorer.
The delivered event-native server owns every model decision.  One controller
server is created per frozen AppWorld task identity.
"""
from __future__ import annotations

from experiments.history_system.native_bare import ARM_RATIOS as NATIVE_RATIOS

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

from benchmarks.adapters import acon_adapter as acon
from .process_lifecycle import run_owned


BENCHMARK = "acon_appworld"
SOURCE_PROFILE = "openai-single-task-v1"
DEFAULT_TAG = "c1_t02_r8"
APPWORLD_ACTOR_SAMPLING = {
    "temperature": 0.0,
    "top_p": 1.0,
    "presence_penalty": 0.5,
    "seed": 42,
    "chat_template_kwargs": {"enable_thinking": False},
}


def _runtime(config: Mapping[str, Any]) -> Mapping[str, Any]:
    value = config.get("c1_runtime", {})
    if not isinstance(value, Mapping):
        raise ValueError("config.c1_runtime must be an object")
    return value


def _source_environment(config: Mapping[str, Any]) -> dict[str, str]:
    root = Path(str(config["appworld_root"])).resolve()
    return {
        **os.environ,
        "APPWORLD_ROOT": str(root),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }


def _prepare_appworld_run(
    config: Mapping[str, Any], official_out: Path, split: str, task_id: str,
) -> Path:
    """Bind ACON's prepare helper to the configured immutable AppWorld data root."""
    data_root = Path(str(config["appworld_root"])).resolve()
    dataset = data_root / "data" / "datasets" / f"{split}.txt"
    if not dataset.is_file():
        raise FileNotFoundError(f"Configured AppWorld dataset does not exist: {dataset}")
    previous = os.environ.get("APPWORLD_ROOT")
    os.environ["APPWORLD_ROOT"] = str(data_root)
    try:
        return acon.prepare_appworld_run(
            Path(str(config["acon_dir"])).resolve(), official_out, split, [task_id]
        )
    finally:
        if previous is None:
            os.environ.pop("APPWORLD_ROOT", None)
        else:
            os.environ["APPWORLD_ROOT"] = previous


def task_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Load the exact configured split through AppWorld's official API."""
    split = str(config.get("appworld_split", acon.APPWORLD_DEFAULT_SPLIT))
    python = str(config["appworld_python"])
    cwd = Path(str(config["acon_dir"])).resolve() / "experiments" / "appworld"
    code = (
        "import json,sys; from appworld import load_task_ids; "
        "print(json.dumps(load_task_ids(sys.argv[1])))"
    )
    process = run_owned(
        [python, "-c", code, split], cwd=cwd, env=_source_environment(config),
        check=True, capture_output=True, text=True,
    )
    try:
        values = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("AppWorld load_task_ids did not return JSON") from error
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) or not value for value in values)
        or len(values) != len(set(values))
    ):
        raise RuntimeError("AppWorld split returned invalid or duplicate task IDs")
    return tuple(values)


def _load_module(path: Path, stem: str) -> ModuleType:
    identity = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    name = f"_c2kv_paper_{stem}_{identity}"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _delivery_path(delivery: Path | ModuleType) -> Path:
    if isinstance(delivery, ModuleType):
        return Path(delivery.HERE).resolve()
    return Path(delivery).resolve()


def _delivery_runner(delivery: Path | ModuleType) -> ModuleType:
    root = _delivery_path(delivery)
    module = (
        delivery.runner
        if isinstance(delivery, ModuleType) and hasattr(delivery, "runner")
        else _load_module(root / "runner.py", "c1_runner")
    )
    module.ROOT = (root / "runtime").resolve()
    return module


def _delivery_run_c1(delivery: Path | ModuleType, runner: ModuleType) -> ModuleType:
    """Load run_c1 with its delivery-local absolute imports, without leaking them."""
    if isinstance(delivery, ModuleType) and hasattr(delivery, "summarize_task"):
        return delivery
    delivery = _delivery_path(delivery)
    aliases: dict[str, ModuleType | None] = {}
    modules = {"runner": runner}
    for name in ("current", "evidence_sets", "c1_artifact_binding"):
        modules[name] = _load_module(delivery / f"{name}.py", f"c1_{name}")
    for name, module in modules.items():
        aliases[name] = sys.modules.get(name)
        sys.modules[name] = module
    try:
        return _load_module(delivery / "run_c1.py", "run_c1")
    finally:
        for name, previous in aliases.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


def _controller_port(config: Mapping[str, Any]) -> int:
    value = _runtime(config).get("controller_port", config.get("proxy_port"))
    if type(value) is not int or not 1 <= value <= 65535:
        raise ValueError("C1 controller port must be an integer in [1, 65535]")
    return value


def _sglang_upstream(config: Mapping[str, Any]) -> str:
    runtime = _runtime(config)
    value = (runtime.get("sglang_backend_url") or config.get("upstream")
             or config.get("sglang_backend_url"))
    if value is None:
        port = config.get("server_port")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ValueError("C1 requires c1_runtime.sglang_backend_url or server_port")
        value = f"http://127.0.0.1:{port}"
    value = str(value).rstrip("/")
    if not re.fullmatch(r"http://(?:127\.0\.0\.1|localhost|\[::1\]):[0-9]+", value):
        raise ValueError("C1 SGLang backend must be an explicit loopback HTTP endpoint")
    return value


def _controller_python(config: Mapping[str, Any]) -> str:
    runtime = _runtime(config)
    value = runtime.get("controller_python") or config.get("bench_python")
    if not isinstance(value, str) or not value:
        raise ValueError("C1 requires c1_runtime.controller_python or bench_python")
    return value


def _openai_origin(base_url: str) -> str:
    """Convert the native ready `/v1` URL to the origin ACON expects."""
    value = str(base_url).rstrip("/")
    match = re.fullmatch(
        r"(http://(?:127\.0\.0\.1|localhost|\[::1\]):[0-9]+)(?:/v1)?", value
    )
    if match is None:
        raise ValueError("C1 AppWorld endpoint must be a loopback origin or /v1 base URL")
    return match.group(1)


def apply_native_history_budget(
    config: Mapping[str, Any], design: dict[str, Any],
    delivery: Path | ModuleType, output: Path,
) -> dict[str, Any] | None:
    """Materialize an explicit native history budget as this cell's eval policy.

    Absent a budget the design keeps the frozen policy. The BFCL delivery path
    applies the same override through run_c1.commands_for_task.
    """
    tokens = config.get("native_history_budget_tokens")
    if tokens is None:
        return None
    delivery_root = _delivery_path(delivery)
    budget = _load_module(delivery_root / "history_budget.py", "history_budget")
    receipt = budget.resolve_override(
        tokens, Path(str(config["checkpoint"])), design,
        delivery_root / "runtime", Path(output),
    )
    budget.materialize(receipt)
    design["runtime"]["eval_policy"] = receipt["override_eval_policy_path"]
    return receipt


def _resolved_design(
    config: Mapping[str, Any], delivery: Path | ModuleType, controller_path: Path,
    output: Path | None = None,
) -> dict[str, Any]:
    delivery = _delivery_path(delivery)
    design = json.loads(
        (delivery / "configs" / "current_algorithm.json").read_text(encoding="utf-8")
    )
    c1 = config.get("c1", {})
    if not isinstance(c1, Mapping):
        raise ValueError("config.c1 must be an object")
    if config.get("native_arm") in NATIVE_RATIOS:
        bare = _load_module(delivery / "native_bare.py", "native_bare")
        design = bare.configure_design(design, NATIVE_RATIOS[config["native_arm"]])
        design["runtime"].update(sglang_backend_url=_sglang_upstream(config),
                                 device="cpu", npu_allocator_metrics=False)
        return design
    if (
        design.get("ratio") != 8
        or c1.get("detector", "t02_risk") not in {"t02_risk", "d3_hybrid"}
        or float(c1.get("selector_threshold", 0.5)) != 0.5
        or c1.get("history_variant", "H0") != "H0"
        or c1.get("recovery_rounds", 1) != 1
    ):
        raise ValueError("AppWorld requires H0/C1000/ratio8/R1 with T02 or D3 hybrid")
    detector = c1.get("detector", "t02_risk")
    no_recovery = config.get("native_arm") == "c2kv_c1_off_r8"
    design["candidate_id"] = "c2kv_only" if no_recovery else f"c1_{detector}"
    design["run_id_template"] = f"paper_{design['candidate_id']}_r8_appworld"
    design["runtime"].update(
        controller=str(controller_path.resolve()),
        sglang_backend_url=_sglang_upstream(config),
        device="cpu",
        npu_allocator_metrics=False,
    )
    if no_recovery:
        design["runtime"].pop("shadow_feature_config", None)
    if output is not None:
        apply_native_history_budget(config, design, delivery, output)
    return design


def server_command(
    config: Mapping[str, Any], task_id: str, directory: Path,
    delivery: Path | ModuleType, controller_path: Path,
) -> list[str]:
    """Build the delivered native C1 server command for one AppWorld task."""
    delivery_root = _delivery_path(delivery)
    directory = Path(directory).resolve()
    runner = _delivery_runner(delivery)
    design = _resolved_design(config, delivery_root, Path(controller_path), directory)
    command = runner.server_command(
        design,
        task_id=task_id,
        checkpoint=str(Path(str(config["checkpoint"])).resolve()),
        output=str(directory),
        port=_controller_port(config),
        python=_controller_python(config),
        benchmark=BENCHMARK,
        source_profile=SOURCE_PROFILE,
    )
    if config.get("tool_memory"):
        command.extend(["--tool-memory", str(config["tool_memory"])])
        if config.get("tool_checkpoint"):
            command.extend(["--tool-checkpoint", str(Path(config["tool_checkpoint"]).resolve())])
        if config.get("tool_budget_tokens") is not None:
            command.extend(["--tool-budget-tokens", str(config["tool_budget_tokens"])])
    return command


# The paper replay path uses this name for the same one-task native endpoint.
controller_command = server_command


def _wait_ready(process: subprocess.Popen, ready_path: Path, deadline: float) -> dict[str, Any]:
    while not ready_path.is_file():
        if process.poll() is not None:
            raise RuntimeError(
                f"C1 controller exited {process.returncode} before readiness"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("C1 AppWorld controller readiness timeout")
        time.sleep(0.25)
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    if ready.get("base_url") is None:
        raise RuntimeError("C1 controller ready receipt has no base_url")
    return ready


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"Expected an object at {path}:{number}")
        rows.append(value)
    return rows


def _official_task_outcome(evaluation_path: Path, task_id: str) -> dict[str, Any]:
    """Load the scorer's durable per-task record, including audit details."""
    evaluation = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    individual = evaluation.get("individual") if isinstance(evaluation, Mapping) else None
    if not isinstance(individual, Mapping) or set(individual) != {task_id}:
        raise RuntimeError("Official AppWorld evaluation does not contain exactly the selected task")
    outcome = individual[task_id]
    if not isinstance(outcome, Mapping):
        raise RuntimeError("Official AppWorld task outcome is not an object")
    success = outcome.get("success", outcome.get("passes_fully"))
    if not isinstance(success, bool):
        raise RuntimeError("Official AppWorld task outcome has no strict success boolean")
    return json.loads(json.dumps(outcome, ensure_ascii=False, allow_nan=False))


def _scenario_id(task_id: str) -> str:
    """Mirror appworld.task.task_id_to_generator_id for scorer grouping."""
    untagged = task_id.split(":", 1)[0]
    if untagged.count("_") != 1:
        raise ValueError(f"Invalid official AppWorld task ID: {task_id!r}")
    return untagged.split("_", 1)[0]


def _percentage_average(values: Sequence[float]) -> float:
    """Mirror AppWorld Evaluator._compute_metrics percentage rounding."""
    fraction = round(sum(values) / len(values) if values else 0, 4)
    return round(100 * fraction, 1)


def validate_task_execution(
    task_id: str, *, steps_path: Path, telemetry_path: Path, history_path: Path,
) -> dict[str, Any]:
    """Bind each executed AppWorld action to the native decision's final draft."""
    steps = _read_jsonl(Path(steps_path))
    events = _read_jsonl(Path(telemetry_path))
    decisions = [row for row in events if row.get("event_type") == "decision"]
    actions = [row for row in events if row.get("event_type") == "tool_action"]
    if not steps or len(steps) != len(decisions) or len(decisions) != len(actions):
        raise RuntimeError(
            "AppWorld requires one native step, harness decision, and executed action per turn"
        )
    final_drafts = []
    discarded_drafts = []
    for index, (step, decision, action) in enumerate(zip(steps, decisions, actions)):
        session = step.get("session_id")
        if session != f"{BENCHMARK}/{task_id}/attempt-0":
            raise RuntimeError(f"Native AppWorld task identity mismatch at decision {index}")
        request_id = step.get("outer_request_id")
        if (
            not isinstance(request_id, str)
            or decision.get("decision_request_id") != request_id
            or action.get("decision_request_id") != request_id
        ):
            raise RuntimeError(f"AppWorld native/action request join mismatch at decision {index}")
        traces = step.get("generation_trace") or []
        if not traces:
            raise RuntimeError(f"Native AppWorld decision {index} has no generation trace")
        final = (traces[-1].get("native_draft") or {}).get("text")
        raw = ((decision.get("response") or {}).get("raw_response"))
        parsed = ((decision.get("response") or {}).get("action"))
        if not isinstance(final, str) or raw != final:
            raise RuntimeError(f"AppWorld harness did not receive final native draft {index}")
        if action.get("action") != parsed:
            raise RuntimeError(f"AppWorld executed action differs from final parsed draft {index}")
        if decision.get("episode_id") != task_id or action.get("episode_id") != task_id:
            raise RuntimeError(f"AppWorld telemetry task identity mismatch at decision {index}")
        final_drafts.append(final)
        discarded_drafts.extend(
            (trace.get("native_draft") or {}).get("text")
            for trace in traces[:-1] if trace.get("discarded") is True
        )

    sessions = json.loads(Path(history_path).read_text(encoding="utf-8"))
    messages = [message for session in sessions for message in session]
    assistant = [message.get("content") for message in messages
                 if isinstance(message, Mapping) and message.get("role") == "assistant"]
    if assistant != final_drafts:
        raise RuntimeError("AppWorld LLM history differs from native final drafts")
    assistant_positions = [
        index for index, message in enumerate(messages)
        if isinstance(message, Mapping) and message.get("role") == "assistant"
    ]
    for index in assistant_positions[:-1]:
        observation = messages[index + 1] if index + 1 < len(messages) else None
        if (not isinstance(observation, Mapping)
                or observation.get("role") != "user"
                or not isinstance(observation.get("content"), str)
                or not observation["content"].strip()):
            raise RuntimeError("AppWorld history omitted the next action observation")
    return {
        "schema": "c1-appworld-execution-binding-v1",
        "task_id": task_id,
        "native_decisions": len(steps),
        "executed_actions": len(actions),
        "discarded_drafts_not_executed": len(discarded_drafts),
        "action_observation_history_bound": True,
    }


# The native server's typed 422 as the ACON runner records it in results.json.
CAPACITY_REJECTION = re.compile(
    r"Error code: 422\b.*[\"']code[\"']\s*:\s*[\"']c2kv_capacity_infeasible[\"']")


def capacity_rejection(task_out: Path, run_dir: Path, task_id: str) -> str | None:
    """The method's capacity failure that ended this task, else None.

    The ACON runner turns the server's 422 into an ordinary generation_error.
    Only that exact rejection, bound by the server's typed step for this task,
    is a method failure; every other generation error remains fatal.
    """
    results_path = acon.appworld_task_dir(run_dir, task_id) / "results.json"
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (not isinstance(results, Mapping)
            or results.get("termination_reason") != "generation_error"
            or not CAPACITY_REJECTION.search(str(results.get("error") or ""))):
        return None
    from .c1 import controller_step_failure

    failure = controller_step_failure(Path(task_out))
    if failure is None or failure[:2] != ("method_failure", "capacity_infeasible"):
        return None
    return failure[2]


def _run_official_harness(
    config: Mapping[str, Any], task_id: str, task_out: Path, base_url: str,
    model: str,
) -> tuple[dict[str, Any], Path, Path]:
    official_out = task_out / BENCHMARK
    acon_dir = Path(str(config["acon_dir"])).resolve()
    python = str(config["appworld_python"])
    split = str(config.get("appworld_split", acon.APPWORLD_DEFAULT_SPLIT))
    max_iter = int(config.get("appworld_max_iter", acon.APPWORLD_DEFAULT_MAX_ITER))
    tag = str(_runtime(config).get("appworld_tag", DEFAULT_TAG))
    acon.validate_appworld_runner_patches(acon_dir)
    run_root = _prepare_appworld_run(config, official_out, split, task_id)
    cwd = run_root / "experiments" / "appworld"
    run_dir = acon.appworld_run_dir(run_root, model, tag, split)
    telemetry_path = official_out / "measurement" / "harness_events.jsonl"
    origin = _openai_origin(base_url)
    env = {
        **acon.appworld_runner_env(origin, telemetry_path, run_dir, acon_dir),
        "APPWORLD_ROOT": str(cwd),
    }
    if config.get("tool_memory"):
        env["C2KV_TOOL_CONTEXT_ON"] = "1"
    timeout = float(config.get("c1", {}).get("task_timeout", 10800))
    run_owned(
        acon.appworld_command(python, model, tag, split, max_iter, [task_id]),
        cwd=cwd, env=env, check=True, timeout=timeout,
    )
    if capacity_rejection(task_out, run_dir, task_id) is not None:
        # No official score exists; paper c1.run_closed_loop records the typed
        # step as this task's scored-zero capacity_infeasible method failure.
        raise RuntimeError(f"AppWorld task {task_id} ended by the native capacity rejection")
    acon.validate_appworld_telemetry(telemetry_path, [task_id])
    scorer_env = {**acon.runner_env(origin), "APPWORLD_ROOT": str(cwd)}
    run_owned(
        acon.appworld_evaluate_command(acon._appworld_cli(python), model, tag, split),
        cwd=cwd, env=scorer_env, check=True, timeout=timeout,
    )
    summary = acon.collect_appworld(
        acon.appworld_eval_path(run_root, model, tag, split), run_dir,
        expected=1, expected_ids=[task_id], telemetry_path=telemetry_path,
    )
    evaluation_path = Path(str(summary["evaluation_path"]))
    official_task_outcome = _official_task_outcome(evaluation_path, task_id)
    official_success = official_task_outcome.get(
        "success", official_task_outcome.get("passes_fully")
    )
    if summary.get("semantic_score") != float(official_success):
        raise RuntimeError("Adapter score differs from the official AppWorld task outcome")
    results_path = acon.appworld_task_dir(run_dir, task_id) / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    task_row = {
        "task_id": task_id,
        "semantic_score": summary.get("semantic_score"),
        "normal_termination": results.get("termination_reason") != "error",
        "protocol_legal": None,
        "n_turns": results.get("iterations"),
        "termination": results.get("termination_reason"),
        "agent_reported_success": results.get("success"),
    }
    official = {
        "schema": "c1-appworld-official-summary-v1",
        "benchmark": BENCHMARK,
        "task_id": task_id,
        "official_scorer": "appworld evaluate (state-based unit tests)",
        "n": 1,
        "task_rows": [task_row],
        "official_task_outcome": official_task_outcome,
        "official_aggregate": summary.get("official_aggregate"),
        "adapter_summary": summary,
        "evaluation_path": str(evaluation_path),
    }
    official_path = official_out / "official_summary.json"
    official_path.write_text(json.dumps(official, indent=2) + "\n", encoding="utf-8")
    return official, telemetry_path, run_dir


def run_task(
    config: Mapping[str, Any], task_id: str, directory: Path,
    delivery: Path | ModuleType, controller_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run one frozen task and return the delivery-style receipt and metrics."""
    if not isinstance(task_id, str) or not task_id or "," in task_id:
        raise ValueError("AppWorld C1 requires one explicit task identity")
    directory = Path(directory).resolve()
    delivery_root = _delivery_path(delivery)
    task_out = directory / "task_shards" / task_id
    task_out.mkdir(parents=True, exist_ok=False)
    command = server_command(config, task_id, directory, delivery, controller_path)
    runner = _delivery_runner(delivery)
    env = os.environ.copy()
    runtime = delivery_root / "runtime"
    env["PYTHONPATH"] = os.pathsep.join(
        (str(runtime / "python"), str(runtime), env.get("PYTHONPATH", ""))
    ).rstrip(os.pathsep)
    timeout = float(config.get("c1", {}).get("task_timeout", 10800))
    deadline = time.monotonic() + timeout
    started = time.monotonic()
    process = None
    official = None
    telemetry_path = run_dir = None
    try:
        with (task_out / "controller.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=runtime, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
            ready = _wait_ready(process, task_out / "server" / "ready.json", deadline)
            if config.get("native_arm") in NATIVE_RATIOS:
                from experiments.history_system.native_bare import validate_manifest
                validate_manifest(task_out / "server" / "ready.json",
                                  NATIVE_RATIOS[config["native_arm"]])
            else:
                from .candidate_matrix import ARM_TO_VARIANT
                arm = config.get("native_arm")
                if arm in ARM_TO_VARIANT:
                    from .native_extra import (
                        validate_candidate_identity, validate_loaded_controller,
                        validate_native_budget_policy,
                    )
                    controller = validate_loaded_controller(arm, ready, controller_path)
                    validate_candidate_identity(arm, ARM_TO_VARIANT[arm], controller, ready)
                    validate_native_budget_policy(config, ready, arm)
            if config.get("tool_memory"):
                from .native_extra import validate_tool_ready
                validate_tool_ready(config, ready)
            official, telemetry_path, run_dir = _run_official_harness(
                config, task_id, task_out, str(ready["base_url"]),
                command[command.index("--model-name") + 1],
            )
    finally:
        if process is not None:
            runner._stop_server(process, task_out / "server.supervisor.json")

    final_path = task_out / "server" / "final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    journal = final.get("journal_summary") or {}
    if (
        official is None or telemetry_path is None or run_dir is None
        or final.get("status") == "failed" or final.get("cost_summary_error")
        or journal.get("failed") or journal.get("pending") or not journal.get("completed")
        or process is None or process.returncode != 0
    ):
        raise RuntimeError(f"C1 AppWorld controller finalization failed; see {final_path}")

    task_dir = acon.appworld_task_dir(run_dir, task_id)
    binding = validate_task_execution(
        task_id,
        steps_path=task_out / "server" / "steps.jsonl",
        telemetry_path=telemetry_path,
        history_path=task_dir / acon.HISTORY_FILE,
    )
    run_c1 = _delivery_run_c1(delivery, runner)
    metrics = run_c1.summarize_task(
        BENCHMARK, task_id, task_out, official, time.monotonic() - started,
    )
    from .candidate_matrix import ARM_TO_VARIANT
    candidate = ARM_TO_VARIANT.get(config.get("native_arm"))
    acceptance = run_c1.functional_checks(
        ("c2kv_native" if config.get("native_arm") in NATIVE_RATIOS else
         "c2kv_only" if config.get("native_arm") == "c2kv_c1_off_r8" else "proposed"),
        config.get("c1", {}).get("detector", "t02_risk"), metrics,
        *((candidate,) if candidate is not None else ()),
    )
    if not all(acceptance["required"].values()):
        raise RuntimeError(f"C1 AppWorld functional acceptance failed: {acceptance['required']}")
    metrics["appworld_execution_binding"] = binding
    receipt = {
        "task_id": task_id,
        "status": "completed",
        "official_summary": official,
        "unified_metrics": metrics,
        "appworld_execution_binding": binding,
        "qualification": "official single-task AppWorld state-based result",
    }
    return receipt, metrics


def summarize_scores(receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate official outcomes and the two declared scored-zero failures."""
    rows = []
    scenario_scores: dict[str, list[float]] = defaultdict(list)
    method_failures = []
    harness_failures = []
    for receipt in receipts:
        metrics = receipt.get("unified_metrics")
        if not isinstance(metrics, Mapping) or metrics.get("official_score") is None:
            raise ValueError("Every AppWorld C1 receipt requires an attached official score")
        status = receipt.get("status")
        if status in {"method_failure", "harness_failure"}:
            expected_kind = {
                "method_failure": "capacity_infeasible",
                "harness_failure": "cuda_oom",
            }[status]
            failure = receipt.get("failure")
            task_id = receipt.get("task_id")
            metric_task_id = metrics.get("task_id")
            if (
                not isinstance(task_id, str)
                or not task_id
                or (metric_task_id is not None and metric_task_id != task_id)
            ):
                raise ValueError("AppWorld C1 failure receipt has no stable task identity")
            if (
                not isinstance(failure, Mapping)
                or failure.get("kind") != expected_kind
                or metrics.get(status) != expected_kind
                or float(metrics["official_score"]) != 0.0
            ):
                raise ValueError("AppWorld C1 failure receipt does not match its scored-zero contract")
            if receipt.get("official_summary") is not None:
                raise ValueError("AppWorld C1 driver failure cannot claim an official scorer artifact")
            scenario_id = _scenario_id(task_id)
            scenario_scores[scenario_id].append(0.0)
            (method_failures if status == "method_failure" else harness_failures).append(
                task_id
            )
            rows.append({
                "task_id": task_id,
                "scenario_id": scenario_id,
                "semantic_score": 0.0,
                "official_success": None,
                "difficulty": None,
                "normal_termination": metrics.get("normal_termination", False),
                "protocol_legal": metrics.get("protocol_legal"),
                "score_source": f"{status}_zero",
                "failure_kind": expected_kind,
            })
            continue
        official = receipt.get("official_summary")
        if not isinstance(official, Mapping):
            raise ValueError("Every AppWorld C1 receipt requires its official scorer artifact")
        task_id = metrics.get("task_id")
        if not isinstance(task_id, str) or official.get("task_id") != task_id:
            raise ValueError("AppWorld C1 receipt and official task identities differ")
        outcome = official.get("official_task_outcome")
        if not isinstance(outcome, Mapping):
            raise ValueError("AppWorld C1 receipt lacks the official per-task outcome")
        success = outcome.get("success", outcome.get("passes_fully"))
        if not isinstance(success, bool) or float(success) != float(metrics["official_score"]):
            raise ValueError("AppWorld C1 official outcome differs from its attached score")
        scenario_id = _scenario_id(task_id)
        scenario_scores[scenario_id].append(float(success))
        rows.append({
            "task_id": task_id,
            "scenario_id": scenario_id,
            "semantic_score": metrics["official_score"],
            "official_success": success,
            "difficulty": outcome.get("difficulty"),
            "normal_termination": metrics.get("normal_termination"),
            "protocol_legal": metrics.get("protocol_legal"),
            "score_source": "official_appworld",
            "failure_kind": None,
        })
    strict_scores = [score for scores in scenario_scores.values() for score in scores]
    task_goal_completion = _percentage_average(strict_scores)
    scenario_goal_completion = _percentage_average(
        [min(scores) for scores in scenario_scores.values()]
    )
    official_aggregate = {
        "task_goal_completion": task_goal_completion,
        "scenario_goal_completion": scenario_goal_completion,
    }
    from . import c1 as paper_c1
    return {
        "arm": paper_c1.ARM,
        "method": "C2KV+C1",
        "ratio": paper_c1.RATIO,
        "n_scored": len(rows),
        "n": len(rows),
        "semantic_score": (
            sum(float(row["semantic_score"]) for row in rows) / len(rows)
            if rows else None
        ),
        "task_goal_completion": task_goal_completion,
        "scenario_goal_completion": scenario_goal_completion,
        "official_aggregate": official_aggregate,
        "official_grouping": "appworld.task.task_id_to_generator_id",
        "n_method_failures": len(method_failures),
        "method_failure_task_ids": method_failures,
        "n_harness_failures": len(harness_failures),
        "harness_failure_task_ids": harness_failures,
        "n_official_scored": len(rows) - len(method_failures) - len(harness_failures),
        "failure_score_policy": (
            "capacity_infeasible method failures and CUDA OOM harness failures "
            "are scored zero without claiming an official per-task outcome"
        ),
        "task_rows": rows,
        "result_status": "preliminary, n=1",
        "official_scorer": "appworld evaluate (state-based unit tests)",
    }
