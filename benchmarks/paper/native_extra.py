"""Official ACEBench Agent and ToolSandbox tasks for native C2KV."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
from collections.abc import Mapping
from functools import lru_cache

from experiments.history_system.native_bare import ARM_RATIOS as NATIVE_RATIOS
from . import c1_appworld
from .candidate_matrix import ARM_TO_VARIANT, GOAL_VARIANTS, REPAIR_VARIANTS, VERIFIED_VARIANTS
from .racer_matrix import is_racer_arm, parse_racer_arm_name, racer_config_for_arm
from experiments.history_system.candidate_algorithms import (
    INITIAL_VIEW_VARIANTS, INITIAL_VIEW_VERSION, initial_view_fields,
    PROOF_REGISTRY_VERSION, VERIFIED_VERSION,
    STATIC_EXTENSION_VARIANTS, STATIC_EXTENSION_VERSION,
)
from .process_lifecycle import run_owned
from benchmarks.toolsandbox_suite import selected_scenarios


BENCHMARKS = {"acebench_agent": ("acebench", "acebench-text-actions-v1"),
              "toolsandbox": ("toolsandbox", "openai-single-task-v1"),
              "tau2": ("tau2", "openai-single-task-v1"),
              "appworld": ("acon_appworld", "openai-single-task-v1")}
TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
C1_RATIOS = {"c2kv_c1_t02_r8": 8, "c2kv_c1_t02_r4": 4}


def validate_tool_ready(config, manifest):
    if not config.get("tool_memory"):
        return
    from benchmarks.toolmemory import validate_ready_tool_contract

    validate_ready_tool_contract(
        manifest, config["tool_memory"], config.get("tool_checkpoint"),
        config.get("tool_budget_tokens"))


def arm_identity(config):
    """Resolve the actual native controller without relying on a benchmark label."""
    arm = config.get("native_arm", "c2kv_native_r4")
    if is_racer_arm(arm):
        backend, policy, budget = parse_racer_arm_name(arm)
        racer = racer_config_for_arm(config, arm)
        if racer["history_budget_tokens"] != budget:
            raise ValueError("RACER arm history budget differs from its backend config")
        return {"arm": arm, "ratio": 8, "method": (
                    "c2kv_only" if policy == "off" else "proposed"),
                "detector": "disabled" if policy == "off" or policy in REPAIR_VARIANTS
                            else "t02_risk",
                "candidate_algorithm": policy if policy in ARM_TO_VARIANT.values() else None,
                "model_name": arm, "racer_backend": racer}
    if arm in NATIVE_RATIOS:
        return {"arm": arm, "ratio": NATIVE_RATIOS[arm], "method": "c2kv_native",
                "detector": "disabled", "candidate_algorithm": None,
                "model_name": arm}
    if arm == "c2kv_c1_off_r8":
        return {"arm": arm, "ratio": 8, "method": "c2kv_only",
                "detector": "disabled", "candidate_algorithm": None,
                "model_name": "c2kv_only"}
    if arm in ARM_TO_VARIANT:
        variant = ARM_TO_VARIANT[arm]
        return {"arm": arm, "ratio": 8, "method": "proposed",
                "detector": "disabled" if variant in REPAIR_VARIANTS else "t02_risk",
                "candidate_algorithm": variant,
                "model_name": f"c2kv_{variant}"}
    if arm in C1_RATIOS:
        settings = config.get("c1") or {}
        detector = settings.get("detector", "d3_hybrid")
        if (detector not in {"t02_risk", "d3_hybrid"}
                or settings.get("selector_threshold", 0.5) != 0.5
                or settings.get("history_variant", "H0") != "H0"
                or settings.get("recovery_rounds", 1) != 1):
            raise ValueError("C1 ACEBench/ToolSandbox requires the H0/R1 frozen detector contract")
        return {"arm": arm, "ratio": C1_RATIOS[arm], "method": "proposed",
                "detector": detector, "candidate_algorithm": None,
                "model_name": f"c1_{detector}"}
    raise ValueError(f"Unsupported native ACEBench/ToolSandbox arm: {arm!r}")


def validate_ready_manifest(config, benchmark, task, ready_path, controller_path):
    """Reject a server whose loaded controller differs from the selected arm."""
    identity = arm_identity(config)
    ready_path = Path(ready_path)
    manifest = json.loads(ready_path.read_text(encoding="utf-8"))
    runtime_name, source_profile = BENCHMARKS[benchmark]
    bare = identity["method"] == "c2kv_native"
    expected_view = ("ac_gist_static" if bare else
                     "ac_native_s0_lexical_raw_reserve_failed_operation")
    if (manifest.get("schema") != "a-event-native-server-v1"
            or manifest.get("status") != "ready"
            or manifest.get("benchmark") != runtime_name
            or manifest.get("source_profile") != source_profile
            or manifest.get("allowed_task_ids") != [_task_id(task)]
            or manifest.get("model_name") != identity["model_name"]
            or manifest.get("view_mode") != expected_view
            or manifest.get("ratio") != identity["ratio"]
            or manifest.get("generation_backend") != "sglang"):
        raise RuntimeError(f"Native {identity['arm']} ready manifest differs from its selected arm")
    validate_tool_ready(config, manifest)
    if bare:
        from experiments.history_system.native_bare import validate_manifest

        validate_manifest(ready_path, identity["ratio"])
        return manifest
    controller_path = Path(controller_path).resolve()
    controller_bytes = controller_path.read_bytes()
    controller = json.loads(controller_bytes)
    loaded = manifest.get("s0_controller_contract")
    if (not isinstance(controller, dict) or not isinstance(loaded, Mapping)
            or loaded.get("source") != str(controller_path)
            or loaded.get("sha256") != hashlib.sha256(controller_bytes).hexdigest()
            or loaded.get("config") != controller):
        raise RuntimeError(f"Native {identity['arm']} loaded a different S0 controller")
    racer = identity.get("racer_backend")
    if racer is not None:
        expected_receipt = dict(
            racer, identity=f"racer:{racer['backend']}:{racer['policy']}:b{racer['history_budget_tokens']}",
            quality_validated=False,
        )
        route = manifest.get("route_contract") or {}
        if (controller.get("racer_backend") != racer
                or manifest.get("racer_backend") != expected_receipt
                or route.get("baseline_identity") != expected_receipt["identity"]
                or route.get("history_allocation") != racer["allocation"]
                or route.get("recovery_enabled") is not (racer["policy"] != "off")):
            raise RuntimeError(f"Native {identity['arm']} RACER backend identity differs")
        variant = identity["candidate_algorithm"]
        candidate = controller.get("candidate_algorithm")
        if variant is not None:
            loaded_candidate = manifest.get("candidate_algorithm")
            if (not isinstance(candidate, Mapping) or candidate.get("variant") != variant
                    or not isinstance(loaded_candidate, Mapping)
                    or loaded_candidate.get("variant") != variant):
                raise RuntimeError(f"Native {identity['arm']} RACER candidate identity differs")
        elif racer["policy"] == "off" and any(
                key in controller for key in ("candidate_algorithm", "post_draft_recovery",
                                           "gp_experiments", "d3_hybrid_recovery")):
            raise RuntimeError(f"Native {identity['arm']} unexpectedly enables recovery")
        return manifest
    variant = identity["candidate_algorithm"]
    candidate = controller.get("candidate_algorithm")
    loaded_candidate = manifest.get("candidate_algorithm")
    if variant is not None:
        route = manifest.get("route_contract") or {}
        artifact = candidate.get("risk_artifact") if isinstance(candidate, Mapping) else None
        if variant in REPAIR_VARIANTS:
            valid_config = (isinstance(candidate, Mapping)
                            and dict(candidate) == {"variant": variant})
            version = "c2kv-source-repair-v1"
        else:
            valid_config = (isinstance(candidate, Mapping)
                            and candidate.get("variant") == variant
                            and candidate.get("risk_threshold") == 0.5
                            and isinstance(artifact, Mapping)
                            and artifact.get("model_kind") == "c1_risk_logistic"
                            and (variant not in VERIFIED_VARIANTS
                                 or candidate.get("proof_registry_version") == PROOF_REGISTRY_VERSION))
            view_fields = initial_view_fields(variant)
            valid_config = valid_config and all(candidate.get(key) == value
                                                for key, value in view_fields.items())
            version = (STATIC_EXTENSION_VERSION if variant in STATIC_EXTENSION_VARIANTS else
                       INITIAL_VIEW_VERSION if variant in INITIAL_VIEW_VARIANTS else
                       VERIFIED_VERSION if variant in VERIFIED_VARIANTS else
                       "c2kv-goal-composition-v1" if variant in GOAL_VARIANTS
                       else "c2kv-paper-candidates-v1")
        if (not valid_config or not isinstance(loaded_candidate, Mapping)
                or loaded_candidate.get("variant") != variant
                or loaded_candidate.get("stable_call_ids") is not True
                or any(loaded_candidate.get(key) != value
                       for key, value in initial_view_fields(variant).items())
                or (variant in VERIFIED_VARIANTS
                    and loaded_candidate.get("proof_registry_version") != PROOF_REGISTRY_VERSION)
                or route.get("baseline_identity") != version + ":" + variant
                or route.get("recovery_enabled") is not True
                or route.get("max_generations_per_decision") != 2):
            raise RuntimeError(f"Native {identity['arm']} candidate controller identity differs")
    elif candidate is not None or loaded_candidate is not None:
        raise RuntimeError(f"Native {identity['arm']} detector controller identity differs")
    elif identity["method"] == "c2kv_only":
        if any(key in controller for key in
               ("post_draft_recovery", "gp_experiments", "d3_hybrid_recovery")):
            raise RuntimeError(f"Native {identity['arm']} unexpectedly enables recovery")
    elif identity["detector"] == "d3_hybrid":
        recovery = controller.get("post_draft_recovery")
        if (controller.get("d3_hybrid_recovery") is not True
                or not isinstance(recovery, Mapping)
                or recovery.get("gate") != "prefill_linear_head"
                or "gp_experiments" in controller):
            raise RuntimeError(f"Native {identity['arm']} detector controller identity differs")
    else:
        gp = controller.get("gp_experiments")
        artifact = gp.get("selector_artifact") if isinstance(gp, Mapping) else None
        if (controller.get("d3_hybrid_recovery") is not None
                or not isinstance(gp, Mapping) or gp.get("set_selector") != "risk"
                or not isinstance(artifact, Mapping)
                or artifact.get("model_kind") != "c1_risk_logistic"):
            raise RuntimeError(f"Native {identity['arm']} detector controller identity differs")
    return manifest


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
    scenarios = selected_scenarios(
        config.get("toolsandbox_suite"), config.get("toolsandbox_scenarios"),
        require_paper_suite=True)
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
    if scenarios is not None:
        if not set(scenarios) <= set(available):
            raise ValueError("Configured ToolSandbox scenarios are not unique official IDs")
        return list(scenarios)
    return available


def selected_tasks(config, benchmark, requested=None):
    if benchmark == "acebench_agent":
        available = _ace_tasks(config)
    elif benchmark == "toolsandbox":
        available = _toolsandbox_tasks(config)
    elif benchmark == "tau2":
        from .tau2 import selected_tasks as tau2_tasks
        available = tau2_tasks(config)
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


def replay_payload(payload, task, step, *, tool_memory=False):
    """Bind a recorded ACE prefix only when it carries real source receipts.

    Full records ``c2kv_tool_spans_v1`` for tool-context replays; a raw-tool
    native server rejects that field, so it is forwarded only with tool memory.
    """
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
    if not tool_memory:
        result.pop("c2kv_tool_spans_v1", None)
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
    delivery_root = c1_appworld._delivery_path(delivery)
    identity = arm_identity(config)
    if identity["method"] == "c2kv_native":
        from experiments.history_system.native_bare import configure_design

        design = json.loads((delivery_root / "configs" / "current_algorithm.json").read_text(encoding="utf-8"))
        design = configure_design(design, identity["ratio"])
    else:
        design = json.loads((delivery_root / "configs" / "current_algorithm.json").read_text(encoding="utf-8"))
        design["ratio"] = identity["ratio"]
        design["candidate_id"] = identity["model_name"]
        design["run_id_template"] = f"paper_{identity['arm']}_{identity['detector']}"
        design["runtime"]["controller"] = str(Path(controller_path).resolve())
        if (identity["method"] == "c2kv_only"
                or identity["candidate_algorithm"] in REPAIR_VARIANTS):
            design["runtime"].pop("shadow_feature_config", None)
    c1_appworld.apply_native_generation_timeout(config, design)
    c1_appworld.apply_native_history_budget(config, design, delivery_root, Path(native))
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
    command = runner.server_command(
        design, task_id=_task_id(task),
        checkpoint=str(Path(config["checkpoint"]).resolve()),
        output=str(Path(native).resolve()),
        port=c1_appworld._controller_port(config),
        python=c1_appworld._controller_python(config),
        benchmark=runtime_name, source_profile=source_profile,
    )
    if config.get("tool_memory"):
        command.extend(["--tool-memory", str(config["tool_memory"])])
        if config.get("tool_checkpoint"):
            command.extend(["--tool-checkpoint", str(Path(config["tool_checkpoint"]).resolve())])
        if config.get("tool_budget_tokens") is not None:
            command.extend(["--tool-budget-tokens", str(config["tool_budget_tokens"])])
    return command


controller_command = server_command


@lru_cache(maxsize=8)
def _preflight_controller_tokenizer(python: str, checkpoint: str) -> None:
    """Check the controller interpreter against the served tokenizer before launch."""
    script = (
        "import sys; from transformers import AutoTokenizer; "
        "tok=AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True); "
        "tok.encode('tokenizer preflight')"
    )
    try:
        result = run_owned([python, "-c", script, checkpoint],
                           capture_output=True, text=True)
    except OSError as error:
        raise RuntimeError(f"Native controller Python is unavailable: {python}") from error
    if result.returncode != 0:
        raise RuntimeError(
            f"Native controller Python {python} cannot load the served tokenizer "
            f"from {checkpoint}: {(result.stderr or '').strip()[-1000:]}")


def _run_official(config, benchmark, task, task_out, base_url, model):
    user_url = c1_appworld._sglang_upstream(config)
    if benchmark == "tau2":
        from benchmarks.adapters.tau2_adapter import run_tau2
        from .tau2 import options

        settings = options(config)
        settings.pop("max_tasks")
        settings["task_ids"] = [task]
        summary = run_tau2(
            base_url, user_url, task_out / "tau2",
            tau2_dir=Path(config["tau2_dir"]),
            python=config.get("tau2_python", config["bench_python"]),
            run_name="paper_" + hashlib.sha256(str(task_out.resolve()).encode()).hexdigest()[:20],
            model=model, user_model=config["model"], native=True,
            native_server_dir=task_out / "server", **settings,
        )
        if summary.get("n") != 1 or summary.get("task_ids") != [task]:
            raise RuntimeError("Official tau2 did not score the frozen task")
        namespace = "tau2"
        scorer = "tau2 evaluate-trajs reward_info.reward"
    elif benchmark == "acebench_agent":
        from benchmarks.adapters import acebench_adapter as ace

        old = os.environ.get("C2KV_ACE_NATIVE")
        old_tool = os.environ.get("C2KV_TOOL_CONTEXT_ON")
        os.environ["C2KV_ACE_NATIVE"] = "1"
        if config.get("tool_memory"):
            os.environ["C2KV_TOOL_CONTEXT_ON"] = "1"
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
            if old_tool is None:
                os.environ.pop("C2KV_TOOL_CONTEXT_ON", None)
            else:
                os.environ["C2KV_TOOL_CONTEXT_ON"] = old_tool
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
            native_server_dir=task_out / "server",
        )
        if summary.get("n") != 1 or summary.get("scenario_ids") != [task]:
            raise RuntimeError("Official ToolSandbox did not score the frozen scenario")
        namespace = "toolsandbox"
        scorer = "tool_sandbox official CLI"
    score = summary.get("semantic_score")
    if type(score) not in (int, float):
        raise RuntimeError("Official harness did not return one numeric semantic score")
    normal_termination = True
    if benchmark == "tau2":
        normal_termination = summary["task_rows"][0]["termination"] in {"agent_stop", "user_stop"}
    official = {
        "schema": "paper-native-extra-official-task-v1",
        "benchmark": namespace, "task_id": task, "official_scorer": scorer,
        "n": 1, "task_rows": [{"task_id": task, "semantic_score": float(score),
                               "normal_termination": normal_termination, "protocol_legal": None}],
        "adapter_summary": summary,
    }
    out = task_out / namespace / "official_summary.json"
    out.write_text(json.dumps(official, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return official


def _official_task_failure(official, task):
    summary = official.get("adapter_summary") if isinstance(official, Mapping) else None
    failures = summary.get("task_failures") if isinstance(summary, Mapping) else None
    if not isinstance(failures, Mapping):
        return None
    direct = failures.get(task)
    if direct in {"decision_cap_reached", "generation_cap_reached", "context_overflow"}:
        return direct
    capacity = failures.get("c2kv_capacity_infeasible")
    if capacity == [task]:
        return "c2kv_capacity_infeasible"
    return None


def run_task(config, benchmark, task, native, delivery, controller_path):
    """Run one official task against a one-task native event server."""
    task = _task_id(task)
    _preflight_controller_tokenizer(
        c1_appworld._controller_python(config),
        str(Path(config["checkpoint"]).resolve()))
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
            validate_ready_manifest(config, benchmark, task,
                                    task_out / "server" / "ready.json", controller_path)
            official = _run_official(config, benchmark, task, task_out,
                                     str(ready["base_url"]),
                                     command[command.index("--model-name") + 1])
    finally:
        if process is not None:
            runner._stop_server(process, task_out / "server.supervisor.json")
    final_path = task_out / "server" / "final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    journal = final.get("journal_summary") or {}
    adapter_failure = _official_task_failure(official, task)
    started_count = journal.get("started")
    completed = journal.get("completed")
    zero_generation_failure = adapter_failure in {
        "context_overflow", "c2kv_capacity_infeasible"}
    if (final.get("status") != "stopped" or final.get("cost_summary_error")
            or journal.get("schema") != "a-runtime-attempt-journal-v1"
            or type(started_count) is not int or type(completed) is not int
            or started_count != completed or completed < 0
            or journal.get("failed") != 0 or journal.get("pending") != 0
            or (not zero_generation_failure and completed == 0)
            or not isinstance(final.get("cost_summary"), Mapping)
            or not final["cost_summary"] or process.returncode != 0):
        raise RuntimeError(f"Native bare controller finalization failed; see {final_path}")
    run_c1 = c1_appworld._delivery_run_c1(delivery, runner)
    namespace = BENCHMARKS[benchmark][0]
    metrics = run_c1.summarize_task(namespace, task, task_out, official,
                                    time.monotonic() - started)
    from .c1 import controller_step_failure
    failure = controller_step_failure(task_out)
    if failure and failure[1] in {"decision_cap_reached", "generation_cap_reached"}:
        if (final.get("api_health") or {}).get("terminal_reason") != failure[1]:
            raise RuntimeError(f"Native budget finalization failed; see {final_path}")
        metrics.update(official_score=0.0, normal_termination=False,
                       method_failure=failure[1])
        return ({"task_id": task, "status": "method_failure",
                 "failure": {"kind": failure[1], "message": failure[2]},
                 "official_summary": official, "unified_metrics": metrics,
                 "qualification": "declared native task budget exhausted; scored 0 as a "
                                  "task-local budget failure, not an official reward"}, metrics)
    if adapter_failure == "context_overflow":
        health = final.get("api_health") or {}
        if (failure is not None or health.get("terminal") is not False
                or health.get("terminal_reason") is not None):
            raise RuntimeError(f"Native context-overflow finalization failed; see {final_path}")
        metrics.update(official_score=0.0, normal_termination=False,
                       method_failure="context_overflow")
        return ({"task_id": task, "status": "method_failure",
                 "failure": {"kind": "context_overflow",
                             "message": "agent input exceeded the served model context"},
                 "official_summary": official, "unified_metrics": metrics,
                 "qualification": "typed task-bound agent context overflow; scored 0 as a "
                                  "task-local method failure, not an official reward"}, metrics)
    if adapter_failure == "c2kv_capacity_infeasible":
        if not failure or failure[1] != "capacity_infeasible":
            raise RuntimeError(f"Native capacity finalization failed; see {final_path}")
        metrics.update(official_score=0.0, normal_termination=False,
                       method_failure="capacity_infeasible")
        return ({"task_id": task, "status": "method_failure",
                 "failure": {"kind": "capacity_infeasible", "message": failure[2]},
                 "official_summary": official, "unified_metrics": metrics,
                 "qualification": "declared native capacity failure; scored 0 as a "
                                  "task-local method failure, not an official reward"}, metrics)
    identity = arm_identity(config)
    acceptance = run_c1.functional_checks(
        identity["method"], identity["detector"], metrics,
        identity["candidate_algorithm"],
        **({"racer_backend": identity["racer_backend"]}
           if "racer_backend" in identity else {}))
    if not all(acceptance["required"].values()):
        raise RuntimeError(f"Native {identity['arm']} functional acceptance failed: {acceptance['required']}")
    return ({"task_id": task, "status": "completed", "official_summary": official,
             "unified_metrics": metrics,
             "qualification": "official single-task native event result"}, metrics)
