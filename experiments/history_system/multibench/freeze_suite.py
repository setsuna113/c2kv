"""Freeze one ordered multi-benchmark suite; do not launch models or scorers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
HISTORY_SYSTEM = HERE.parent
REPO = HISTORY_SYSTEM.parents[1]
ACTIVE_RUNTIME = HISTORY_SYSTEM / "runtime"
DEFAULT_CONTROLLER = ACTIVE_RUNTIME / "configs/controller.json"
DEFAULT_EVAL_POLICY = ACTIVE_RUNTIME / "configs/eval_policy.json"
DEFAULT_EVAL_CAPACITY = ACTIVE_RUNTIME / "configs/eval_capacity.json"
DEFAULT_SHADOW_FEATURES = ACTIVE_RUNTIME / "configs/shadow_features.json"
DEFAULT_CHECKPOINT = HISTORY_SYSTEM / "configs/checkpoint.selected.json"
EXCLUDED_NAMES = {"__pycache__", ".pytest_cache"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_runner(path: Path):
    spec = importlib.util.spec_from_file_location("_frozen_multibench_suite_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load suite runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not any(part in EXCLUDED_NAMES for part in path.parts)
        and path.suffix != ".pyc"
    )


def _copy_runtime(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Active runtime does not exist: {source}")
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
    )


def _checkpoint_binding(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Checkpoint binding must be an object")
    required = {
        "status": "selected",
        "selected_arm": "C",
        "selected_step": 1000,
        "ratio": 8,
    }
    for name, expected in required.items():
        if value.get(name) != expected:
            raise ValueError("The suite requires the selected C1000 ratio8 checkpoint")
    path, digest = value.get("path"), value.get("config_sha256")
    if not isinstance(path, str) or not path:
        raise ValueError("Checkpoint path must be explicit")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("Checkpoint config_sha256 must be a lowercase SHA-256 digest")
    return {
        "status": "selected",
        "path": path,
        "config_sha256": digest,
        "selected_arm": "C",
        "selected_step": 1000,
        "ratio": 8,
    }


def _validate_fixed_configs(
    controller: Any, eval_policy: Any, eval_capacity: Any, shadow_features: Any
) -> None:
    if not isinstance(controller, Mapping):
        raise ValueError("Controller configuration must be an object")
    policy = eval_policy.get("policy") if isinstance(eval_policy, Mapping) else None
    if not isinstance(policy, Mapping) or (
        policy.get("history_budget_bytes") != 113246208
        or policy.get("workspace_budget_bytes") != 113246208
        or policy.get("lease_decisions") != 0
    ):
        raise ValueError("Evaluation policy must preserve exact B0=113246208 with no lease")
    capacity = eval_capacity.get("capacity") if isinstance(eval_capacity, Mapping) else None
    if not isinstance(capacity, Mapping) or capacity.get("max_sequence_tokens") != 40960:
        raise ValueError("Evaluation capacity must preserve context40960")
    if not isinstance(shadow_features, Mapping) or shadow_features.get("enabled") is not True:
        raise ValueError("The delivery suite requires explicit enabled shadow feature capture")


def _package_files(submitted: Path) -> dict[str, str]:
    return {
        path.relative_to(submitted).as_posix(): sha256(path)
        for path in _source_files(submitted)
        if path.name != "package.manifest.json"
    }


def _frozen_adapter_smoke(submitted: Path) -> dict[str, Any]:
    runtime = submitted / "runtime"
    required_files = [
        "benchmarks/adapters/base.py",
        "benchmarks/adapters/bfcl_adapter.py",
        "benchmarks/adapters/tau2_adapter.py",
        "benchmarks/adapters/toolsandbox_adapter.py",
        "benchmarks/adapters/acebench_adapter.py",
        "benchmarks/adapters/acon_adapter.py",
        "benchmarks/metrics.py",
        "benchmarks/reqlog.py",
        "benchmarks/proxy.py",
        "benchmarks/terminal_check.py",
    ]
    missing = [name for name in required_files if not (runtime / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Frozen official adapter dependencies are missing: {missing}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(runtime / "python"), str(runtime)]
    )
    environment["PYTHONNOUSERSITE"] = "1"
    help_run = subprocess.run(
        [sys.executable, str(submitted / "official_one_task.py"), "--help"],
        cwd=submitted,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if help_run.returncode != 0:
        raise RuntimeError(
            "Frozen official_one_task --help failed: " + help_run.stderr[-2000:]
        )
    modules = [
        "benchmarks.adapters.bfcl_adapter",
        "benchmarks.adapters.tau2_adapter",
        "benchmarks.adapters.toolsandbox_adapter",
        "benchmarks.adapters.acebench_adapter",
        "benchmarks.adapters.acon_adapter",
    ]
    import_code = (
        "import importlib,json,pathlib; root=pathlib.Path(r'"
        + str(runtime.resolve()).replace("'", "\\'")
        + "').resolve(); names="
        + repr(modules)
        + "; found={}; "
        + "[(lambda m,n: found.__setitem__(n,str(pathlib.Path(m.__file__).resolve())))(importlib.import_module(n),n) for n in names]; "
        + "assert all(root in pathlib.Path(p).parents for p in found.values()); print(json.dumps(found,sort_keys=True))"
    )
    import_run = subprocess.run(
        [sys.executable, "-c", import_code],
        cwd=submitted,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if import_run.returncode != 0:
        raise RuntimeError("Frozen adapter import failed: " + import_run.stderr[-2000:])
    return {
        "schema": "a-history-multibench-frozen-adapter-smoke-v1",
        "status": "passed_cpu_no_model_or_scorer",
        "python": sys.executable,
        "official_help_returncode": help_run.returncode,
        "adapter_modules": json.loads(import_run.stdout),
        "required_files": required_files,
        "model_calls": 0,
        "scorer_calls": 0,
        "network_calls": 0,
    }


def freeze_suite(
    *,
    suite_id: str,
    candidate_id: str,
    tasks_manifest: Path,
    controller_path: Path,
    eval_policy_path: Path,
    eval_capacity_path: Path,
    shadow_feature_path: Path,
    checkpoint_path: Path,
    official_one_task_path: Path,
    runtime_source: Path,
    output: Path,
    server_python: str,
    official_python: str,
    max_decisions_per_task: int = 96,
    server_wall_seconds_per_task: int = 10800,
    official_wall_seconds_per_task: int = 10800,
    server_ready_seconds: int = 900,
    port_base: int = 42000,
    model_name: str | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Frozen suite output already exists: {output}")
    if (
        not isinstance(suite_id, str)
        or not suite_id
        or not isinstance(candidate_id, str)
        or not candidate_id
    ):
        raise ValueError("suite_id and candidate_id must be nonempty")
    for name, value in {
        "server_python": server_python,
        "official_python": official_python,
    }.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be explicit")

    runner = _load_runner(HERE / "suite_runner.py")
    task_manifest = read_json(tasks_manifest)
    tasks = runner.validate_task_manifest(task_manifest)
    checkpoint = _checkpoint_binding(read_json(checkpoint_path))
    controller = read_json(controller_path)
    eval_policy = read_json(eval_policy_path)
    eval_capacity = read_json(eval_capacity_path)
    shadow_features = read_json(shadow_feature_path)
    _validate_fixed_configs(controller, eval_policy, eval_capacity, shadow_features)
    if not official_one_task_path.is_file():
        raise FileNotFoundError(f"Official one-task adapter is missing: {official_one_task_path}")

    submitted = output / "submitted"
    submitted.mkdir(parents=True)
    _copy_runtime(runtime_source, submitted / "runtime")
    shutil.copy2(HERE / "suite_runner.py", submitted / "suite_runner.py")
    shutil.copy2(official_one_task_path, submitted / "official_one_task.py")
    save_json(submitted / "tasks.json", task_manifest)
    save_json(submitted / "configs/controller.json", controller)
    save_json(submitted / "configs/eval_policy.json", eval_policy)
    save_json(submitted / "configs/eval_capacity.json", eval_capacity)
    save_json(submitted / "configs/shadow_features.json", shadow_features)

    suite = {
        "schema": runner.SUITE_SCHEMA,
        "status": "frozen",
        "suite_id": suite_id,
        "candidate_id": candidate_id,
        "task_manifest": "tasks.json",
        "task_manifest_sha256": sha256(submitted / "tasks.json"),
        "fixed_denominator": len(tasks),
        "benchmark_counts": dict(sorted(Counter(row["benchmark"] for row in tasks).items())),
        "max_new_tokens_by_benchmark": dict(runner.TASK_MAX_NEW_TOKENS),
        "checkpoint": checkpoint,
        "interpreters": {"server": server_python, "official": official_python},
        "runtime": {
            "runtime_root": "runtime",
            "model_name": model_name or candidate_id,
            "server_module": "benchmarks.memory_runtime.event_native_server",
            "official_one_task": "official_one_task.py",
            "source_profile": "openai-single-task-v1",
            "view_mode": "ac_native_s0_lexical_raw_reserve_failed_operation",
            "compression_policy": "always-compress-v1",
            "history_view_protocol": "fixed-budget-main",
            "ratio": 8,
            "dtype": "bfloat16",
            "device": "npu:0",
            "prefill_chunk_size": 256,
            "decode_strategy": "incremental",
            "sampling": {"mode": "greedy", "temperature": 0, "seed": 0},
            "no_raw_snapshot": True,
            "npu_allocator_metrics": True,
            "torch_threads": 4,
            "controller": "configs/controller.json",
            "eval_policy": "configs/eval_policy.json",
            "eval_capacity": "configs/eval_capacity.json",
            "shadow_feature_config": "configs/shadow_features.json",
        },
        "limits": {
            "generation_calls_per_task": 96,
            "extraction_calls_per_task": 1152,
            "stage_wall_seconds": 21600,
            "max_decisions_per_task": max_decisions_per_task,
            "server_wall_seconds_per_task": server_wall_seconds_per_task,
            "official_wall_seconds_per_task": official_wall_seconds_per_task,
            "server_ready_seconds": server_ready_seconds,
        },
        "retry_contract": {
            "automatic_reruns": 0,
            "server_start_retries": 0,
            "official_worker_retries": 0,
            "transport_retries": 0,
        },
        "automatic_reruns": 0,
        "execution": {
            "task_order": "manifest_list_order",
            "fresh_server_per_task": True,
            "infra_failure_remains_in_denominator": True,
            "continue_after_task_infra_failure": True,
            "outer_launch_wrapper": (
                "/bin/bash /home/liuyancheng/c2kv-a-runtime-20260907/"
                "native_npu_cost_tf58_v1/launch.sh"
            ),
        },
    }
    save_json(submitted / "suite.json", suite)

    adapter_smoke = _frozen_adapter_smoke(submitted)
    save_json(output / "adapter_smoke.local.json", adapter_smoke)

    package_files = _package_files(submitted)
    package_manifest = {
        "schema": runner.PACKAGE_SCHEMA,
        "suite_id": suite_id,
        "files": package_files,
    }
    save_json(submitted / "package.manifest.json", package_manifest)

    # Load the copied runner and validate only the immutable copied package.
    frozen_runner = _load_runner(submitted / "suite_runner.py")
    frozen_suite = read_json(submitted / "suite.json")
    preview = frozen_runner.preview(frozen_suite, submitted.resolve(), port_base)
    save_json(output / "preview.local.json", preview)

    archive = output / "package.tar.gz"
    archive_names = sorted([*package_files, "package.manifest.json"])
    with tarfile.open(archive, "w:gz") as stream:
        for name in archive_names:
            stream.add(submitted / name, arcname=name, recursive=False)
    receipt = {
        "schema": "a-history-multibench-freeze-receipt-v1",
        "status": "frozen_not_launched",
        "suite_id": suite_id,
        "candidate_id": candidate_id,
        "tasks": len(tasks),
        "fixed_denominator": len(tasks),
        "benchmark_counts": suite["benchmark_counts"],
        "archive": str(archive.resolve()),
        "archive_sha256": sha256(archive),
        "manifest_sha256": sha256(submitted / "package.manifest.json"),
        "suite_sha256": sha256(submitted / "suite.json"),
        "task_manifest_sha256": sha256(submitted / "tasks.json"),
        "runner_sha256": sha256(submitted / "suite_runner.py"),
        "official_one_task_sha256": sha256(submitted / "official_one_task.py"),
        "adapter_smoke_sha256": sha256(output / "adapter_smoke.local.json"),
        "runtime_files": len(_source_files(submitted / "runtime")),
        "package_files": len(package_files),
        "automatic_reruns": 0,
        "model_calls": 0,
        "scorer_calls": 0,
        "network_calls": 0,
    }
    save_json(output / "freeze.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--tasks-manifest", type=Path, required=True)
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--eval-policy", type=Path, default=DEFAULT_EVAL_POLICY)
    parser.add_argument("--eval-capacity", type=Path, default=DEFAULT_EVAL_CAPACITY)
    parser.add_argument("--shadow-feature-config", type=Path, default=DEFAULT_SHADOW_FEATURES)
    parser.add_argument("--checkpoint-binding", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--official-one-task", type=Path, default=HERE / "official_one_task.py")
    parser.add_argument("--runtime-source", type=Path, default=ACTIVE_RUNTIME)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server-python", default="/home/liuyancheng/envs/sgl/bin/python")
    parser.add_argument("--official-python", default="/home/liuyancheng/envs/sgl/bin/python")
    parser.add_argument("--max-decisions-per-task", type=int, default=96)
    parser.add_argument("--server-wall-seconds-per-task", type=int, default=10800)
    parser.add_argument("--official-wall-seconds-per-task", type=int, default=10800)
    parser.add_argument("--server-ready-seconds", type=int, default=900)
    parser.add_argument("--model-name", help="Public served alias; candidate identity remains separate")
    parser.add_argument("--port-base", type=int, default=42000)
    args = parser.parse_args(argv)
    receipt = freeze_suite(
        suite_id=args.suite_id,
        candidate_id=args.candidate_id,
        tasks_manifest=args.tasks_manifest.resolve(),
        controller_path=args.controller.resolve(),
        eval_policy_path=args.eval_policy.resolve(),
        eval_capacity_path=args.eval_capacity.resolve(),
        shadow_feature_path=args.shadow_feature_config.resolve(),
        checkpoint_path=args.checkpoint_binding.resolve(),
        official_one_task_path=args.official_one_task.resolve(),
        runtime_source=args.runtime_source.resolve(),
        output=args.out.resolve(),
        server_python=args.server_python,
        official_python=args.official_python,
        max_decisions_per_task=args.max_decisions_per_task,
        server_wall_seconds_per_task=args.server_wall_seconds_per_task,
        official_wall_seconds_per_task=args.official_wall_seconds_per_task,
        server_ready_seconds=args.server_ready_seconds,
        port_base=args.port_base,
        model_name=args.model_name,
    )
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
