"""Self-contained CPU validation for the active history-system runtime."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNTIME = HERE / "runtime"
CURRENT = HERE / "configs/current_algorithm.json"
DEFAULT_TESTS = (
    HERE / "test_budget_guard.py",
    Path("benchmarks/memory_runtime/tests/test_same_event_bridge_only.py"),
    Path("benchmarks/memory_runtime/tests/test_event_native_recovery.py"),
    Path("benchmarks/memory_runtime/tests/test_event_native_step.py"),
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_path(reference: str) -> Path:
    relative = Path(reference)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Runtime config path must be contained: {reference}")
    path = (RUNTIME / relative).resolve()
    if RUNTIME.resolve() not in path.parents:
        raise ValueError(f"Runtime config path escapes runtime: {reference}")
    return path


def source_files() -> dict[str, str]:
    return {
        p.relative_to(RUNTIME).as_posix(): sha(p)
        for p in RUNTIME.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
        and ".pytest_cache" not in p.parts and p.suffix != ".pyc"
    }


def current_configs(current: dict) -> tuple[dict[str, dict], dict[str, str]]:
    if current.get("schema") != "a-current-algorithm-v1":
        raise ValueError("Unexpected current algorithm schema")
    if current.get("candidate_id") != "d3_prefill_event" or current.get("ratio") != 8:
        raise ValueError("Current algorithm must bind selected D3 ratio8")
    keys = ("controller", "eval_policy", "eval_capacity", "shadow_feature_config")
    paths = {key: runtime_path(current["runtime"][key]) for key in keys}
    configs = {key: read(path) for key, path in paths.items()}
    recovery = configs["controller"].get("post_draft_recovery", {})
    if (recovery.get("schema") != "a-e1-post-draft-event-recovery-v1"
            or recovery.get("gate") != "prefill_linear_head"
            or not isinstance(recovery.get("prefill_head"), dict)):
        raise ValueError("Current controller is not Prefill-guided recovery")
    if current.get("source", {}).get("controller_sha256") != sha(paths["controller"]):
        raise ValueError("Current controller differs from selected release binding")
    policy = configs["eval_policy"].get("policy", {})
    if (policy.get("history_budget_bytes"), policy.get("workspace_budget_bytes")) != (
            113246208, 113246208):
        raise ValueError("Current policy differs from B0")
    if configs["eval_capacity"].get("capacity") != {
            "max_workspace_tokens": 36864, "max_sequence_tokens": 40960}:
        raise ValueError("Current capacity differs from selected D3")
    if configs["shadow_feature_config"].get("enabled") is not True:
        raise ValueError("Selected D3 requires shadow features")
    return configs, {key: sha(path) for key, path in paths.items()}


def design_for(candidate: str, current: dict, configs: dict[str, dict],
               files: dict[str, str]) -> dict:
    tasks = ["multi_turn_base_0", "multi_turn_long_context_0"]
    return {
        "schema": "a-history-system-candidate-design-v1",
        "status": "development_not_frozen",
        "candidate_id": candidate,
        "run_id_template": "a_history_validation_" + candidate,
        "evaluation_stage": "development_search",
        "launch_authorized": False,
        "acceptance_parameters_frozen": False,
        "task_ids": tasks,
        "limits": {**current["limits"], "tasks": len(tasks)},
        **{key: current[key] for key in (
            "route", "compression_policy", "history_view_protocol", "decode_strategy",
            "prefill_chunk_size", "sampling", "session_cache_policy", "retry_contract",
            "ratio", "checkpoint_selection")},
        "runtime": dict(current["runtime"]),
        "resolved_configs": configs,
        "source_files": files,
        "task_and_scorer_lineage": {
            "bfcl_root_default": "<bfcl-root>", "source_bindings": {}},
        "search_contract": {
            "B0_history_bytes": 113246208,
            "history_token_budget": 768,
            "quality_target": "CPU validation only",
            "speed_is_auxiliary": True,
            "full_task_rollout": True,
            "new_cohort_planned_reference_not_automatic_retry": True,
        },
    }


def main() -> None:
    current = read(CURRENT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default=current["candidate_id"])
    parser.add_argument("--extra-test", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.candidate.replace("_", "").isalnum():
        raise ValueError("Invalid candidate name")
    output = args.output or (
        REPO / "outputs/history_system_search/r001"
        / ("validation." + args.candidate + ".json")
    )
    if output.exists():
        raise FileExistsError(output)

    sys.path[:0] = [str(RUNTIME / "python"), str(RUNTIME)]
    runner = load("active_history_system_runner", HERE / "runner.py")
    runner.ROOT = RUNTIME
    configs, config_hashes = current_configs(current)
    files = source_files()
    design = design_for(args.candidate, current, configs, files)
    runner.validate(design, allow_development=True)
    preview = runner.preview(
        design,
        argparse.Namespace(
            checkpoint=current["checkpoint_selection"]["path"],
            output="<unique-run-output>",
            benchmark_dir="<bfcl-root>",
            python="/home/liuyancheng/envs/sgl/bin/python",
            bfcl_python=current["runtime"]["bfcl_python"],
            port_base=28800,
        ),
    )
    if (preview["whole_task_denominator"] != 2
            or preview["automatic_reruns"] != 0
            or any("--shadow-feature-config" not in row["server"]
                   for row in preview["cells"])):
        raise ValueError("Current D3 static preview changed its runtime contract")

    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join([str(RUNTIME / "python"), str(RUNTIME)]),
        PYTHONDONTWRITEBYTECODE="1",
    )
    tests = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "--import-mode=importlib",
         "--confcutdir=.", "-q", "-p", "no:cacheprovider",
         *(str(path) for path in DEFAULT_TESTS), *args.extra_test],
        cwd=RUNTIME, env=env, capture_output=True, text=True, timeout=60,
    )
    if tests.returncode != 0:
        raise AssertionError(tests.stdout + tests.stderr)

    result = {
        "schema": "a-history-system-cpu-validation-v1",
        "status": "passed",
        "candidate_id": args.candidate,
        "model_calls": 0,
        "scorer_calls": 0,
        "network_calls": 0,
        "current_algorithm": {
            "path": CURRENT.relative_to(REPO).as_posix(),
            "sha256": sha(CURRENT),
            "candidate_id": current["candidate_id"],
            "checkpoint_selection": current["checkpoint_selection"],
        },
        "runtime_config_sha256": config_hashes,
        "runner_sha256": sha(HERE / "runner.py"),
        "checks": {
            "current_d3_config_constructs_and_previews": True,
            "budget_bridge_recovery_step_tests_passed": True,
            "tests_stdout": tests.stdout,
        },
        "runtime_source_files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {key: value for key, value in result.items()
         if key != "runtime_source_files"},
        indent=2,
    ))


if __name__ == "__main__":
    main()
