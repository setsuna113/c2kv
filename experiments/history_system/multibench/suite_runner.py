"""Run one immutable multi-benchmark task shard without automatic reruns."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import urlsplit


SUITE_SCHEMA = "a-history-multibench-frozen-suite-v1"
TASK_MANIFEST_SCHEMA = "a-history-multibench-task-manifest-v1"
PACKAGE_SCHEMA = "a-history-multibench-package-v1"
STAGE_SCHEMA = "a-history-multibench-stage-v1"
OFFICIAL_TASK_SCHEMA = "history-system-official-task-v1"
TASK_RESULT_SCHEMA = "history-system-official-result-v1"
SUPPORTED_BENCHMARKS = frozenset(
    {"tau2", "toolsandbox", "acebench", "acon_appworld"}
)
OFFICIAL_TASK_FIELDS = frozenset(
    {
        "schema",
        "benchmark",
        "task_id",
        "benchmark_dir",
        "bench_python",
        "user_base_url",
        "source_binding",
        "max_new_tokens",
        "run_name",
        "split",
        "tag",
        "max_steps",
        "max_iter",
        "category",
        "language",
        "max_dialog_turns",
        "appworld_root",
        "task_key",
    }
)
FORBIDDEN_TASK_FIELDS = frozenset(
    {"gold", "oracle", "target_action", "hidden_state", "expected_score"}
)
CLEANUP_RESERVE_SECONDS = 60.0
TOOLSANDBOX_OFFICIAL_MODEL_NAME = "gpt-4o-2024-05-13"
TASK_MAX_NEW_TOKENS = {
    "tau2": 4096,
    "toolsandbox": 4096,
    "acon_appworld": 2048,
    "acebench": 1200,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_relative(root: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field} must remain inside the frozen package")
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{field} escapes the frozen package")
    return resolved


def _digest(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _validate_source_binding(
    value: Any, required_roots: tuple[str, ...], ordinal: int
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"components"}:
        raise ValueError(f"Task row {ordinal} requires exact source_binding.components")
    components = value.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError(f"Task row {ordinal} requires source components")
    component_paths: set[str] = set()
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            raise ValueError(f"Task row {ordinal} source component {index} must be an object")
        kind, path = component.get("kind"), component.get("path")
        if not isinstance(path, str) or not path or path in component_paths:
            raise ValueError(f"Task row {ordinal} source component paths must be nonempty and unique")
        component_paths.add(path)
        if kind == "git":
            if set(component) != {"path", "kind", "revision", "required_clean"}:
                raise ValueError(f"Task row {ordinal} git source binding fields changed")
            revision = component.get("revision")
            if (
                not isinstance(revision, str)
                or len(revision) != 40
                or any(character not in "0123456789abcdef" for character in revision)
            ):
                raise ValueError(f"Task row {ordinal} git source requires a full object revision")
            if component.get("required_clean") is not True:
                raise ValueError(f"Task row {ordinal} git source must require a clean checkout")
        elif kind == "files":
            if set(component) != {"path", "kind", "files"}:
                raise ValueError(f"Task row {ordinal} file source binding fields changed")
            files = component.get("files")
            if not isinstance(files, Mapping) or not files:
                raise ValueError(f"Task row {ordinal} file source requires hashes")
            for name, digest in files.items():
                relative = Path(name) if isinstance(name, str) else Path("/")
                if (
                    not isinstance(name, str)
                    or not name
                    or relative.is_absolute()
                    or ".." in relative.parts
                ):
                    raise ValueError(f"Task row {ordinal} file source paths must be contained")
                _digest(digest, f"Task row {ordinal} source hash for {name}")
        else:
            raise ValueError(f"Task row {ordinal} source component kind is unsupported")
    missing_roots = [root for root in required_roots if root not in component_paths]
    if missing_roots:
        raise ValueError(
            f"Task row {ordinal} source binding does not cover required roots: {missing_roots!r}"
        )


def validate_task_manifest(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, Mapping) or value.get("schema") != TASK_MANIFEST_SCHEMA:
        raise ValueError("Unexpected multi-benchmark task manifest schema")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task manifest requires a nonempty ordered tasks list")
    if value.get("ordering") != "listed" or value.get("automatic_reruns") != 0:
        raise ValueError("Task order must be frozen and automatic reruns disabled")
    if value.get("fixed_denominator") != len(tasks):
        raise ValueError("Task manifest fixed denominator differs from its rows")
    keys: set[str] = set()
    rows = []
    for ordinal, raw in enumerate(tasks):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Task row {ordinal} must be an object")
        unknown = set(raw) - OFFICIAL_TASK_FIELDS
        if unknown:
            raise ValueError(f"Task row {ordinal} has unsupported fields: {sorted(unknown)!r}")
        if FORBIDDEN_TASK_FIELDS & set(raw):
            raise ValueError(f"Task row {ordinal} contains privileged evaluation fields")
        benchmark, task_id = raw.get("benchmark"), raw.get("task_id")
        if benchmark not in SUPPORTED_BENCHMARKS:
            raise ValueError(f"Task row {ordinal} has an unsupported benchmark")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"Task row {ordinal} requires a nonempty task_id")
        for field in ("schema", "benchmark_dir", "bench_python"):
            if not isinstance(raw.get(field), str) or not raw[field]:
                raise ValueError(f"Task row {ordinal} requires nonempty {field}")
        if not isinstance(raw.get("user_base_url"), str) or (
            benchmark != "acon_appworld" and not raw["user_base_url"]
        ):
            raise ValueError(f"Task row {ordinal} requires its official user endpoint")
        if raw["schema"] != OFFICIAL_TASK_SCHEMA:
            raise ValueError(f"Task row {ordinal} has an unsupported schema")
        expected_cap = TASK_MAX_NEW_TOKENS[benchmark]
        if raw.get("max_new_tokens") != expected_cap:
            raise ValueError(
                f"Task row {ordinal} must freeze {benchmark} max_new_tokens={expected_cap}"
            )
        required_roots = (raw["benchmark_dir"],)
        if benchmark == "acon_appworld" and raw.get("split") != "test_normal":
            raise ValueError("AppWorld task rows must bind split=test_normal")
        if benchmark == "acon_appworld" and raw.get("max_iter", 50) != 50:
            raise ValueError("AppWorld task rows must keep max_iter=50")
        if benchmark == "acon_appworld":
            appworld_root = raw.get("appworld_root")
            if not isinstance(appworld_root, str) or not appworld_root:
                raise ValueError("AppWorld task rows require an explicit appworld_root")
            required_roots += (appworld_root,)
        elif "appworld_root" in raw:
            raise ValueError("appworld_root is valid only for AppWorld task rows")
        _validate_source_binding(raw.get("source_binding"), required_roots, ordinal)
        if benchmark == "tau2" and (
            not isinstance(raw.get("run_name"), str) or not raw["run_name"]
        ):
            raise ValueError("tau2 task rows require an explicit run_name")
        if benchmark == "tau2" and raw.get("max_steps", 200) != 200:
            raise ValueError("tau2 task rows must keep max_steps=200")
        if benchmark == "acebench" and (
            raw.get("category") != "agent" or raw.get("language") != "en"
        ):
            raise ValueError("ACEBench task rows must bind category=agent and language=en")
        if benchmark == "toolsandbox" and "role" in raw:
            raise ValueError("ToolSandbox official task rows do not carry a role field")
        task_key = raw.get("task_key")
        if not isinstance(task_key, str) or not task_key:
            task_key = f"{ordinal:04d}_{benchmark}_{hashlib.sha256(task_id.encode()).hexdigest()[:12]}"
        if (
            any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in task_key)
            or task_key in keys
        ):
            raise ValueError("Task keys must be unique filesystem-safe strings")
        keys.add(task_key)
        row = copy.deepcopy(dict(raw))
        row.update(ordinal=ordinal, task_key=task_key)
        rows.append(row)
    return tuple(rows)


def verify_package(package_root: Path, suite: Mapping[str, Any]) -> dict[str, str]:
    manifest_path = package_root / "package.manifest.json"
    manifest = read_json(manifest_path)
    if not isinstance(manifest, Mapping) or manifest.get("schema") != PACKAGE_SCHEMA:
        raise ValueError("Frozen package manifest is absent or has the wrong schema")
    if manifest.get("suite_id") != suite.get("suite_id"):
        raise ValueError("Package and suite identities differ")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("Package manifest requires a nonempty file inventory")
    observed = {}
    for name, expected in files.items():
        path = _safe_relative(package_root, name, "package file")
        digest = _digest(expected, f"package hash for {name}")
        if not path.is_file():
            raise ValueError(f"Frozen package file is missing: {name}")
        actual = sha256(path)
        if actual != digest:
            raise ValueError(f"Frozen package file changed: {name}")
        observed[name] = actual
    # Transport and execution receipts live beside the immutable uploaded files.
    execution_artifacts = {
        "package.tar.gz", "preview.remote.json", "launch.json",
        "resources.before.txt", "supervisor.log", "terminal_recovery.tar.gz",
    }
    unlisted = sorted(
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file()
        and path != manifest_path
        and path.relative_to(package_root).as_posix() not in execution_artifacts
        and path.relative_to(package_root).parts[0] != "results"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.relative_to(package_root).as_posix() not in files
    )
    if unlisted:
        raise ValueError(f"Frozen package contains unlisted files: {unlisted!r}")
    return observed


def validate_suite(
    suite: Any, package_root: Path, *, verify_files: bool = True
) -> tuple[dict[str, Any], ...]:
    if not isinstance(suite, Mapping) or suite.get("schema") != SUITE_SCHEMA:
        raise ValueError("Unexpected frozen suite schema")
    if suite.get("status") != "frozen" or suite.get("automatic_reruns") != 0:
        raise ValueError("Execution requires a frozen zero-rerun suite")
    if not isinstance(suite.get("suite_id"), str) or not suite["suite_id"]:
        raise ValueError("Frozen suite requires a nonempty suite_id")
    task_path = _safe_relative(package_root, suite.get("task_manifest"), "task_manifest")
    expected_tasks = _digest(suite.get("task_manifest_sha256"), "task_manifest_sha256")
    if sha256(task_path) != expected_tasks:
        raise ValueError("Frozen task manifest changed")
    tasks = validate_task_manifest(read_json(task_path))
    if suite.get("fixed_denominator") != len(tasks):
        raise ValueError("Suite fixed denominator differs from task manifest")

    checkpoint = suite.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Frozen suite requires a checkpoint binding")
    if (
        checkpoint.get("status") != "selected"
        or checkpoint.get("selected_arm") != "C"
        or checkpoint.get("selected_step") != 1000
        or checkpoint.get("ratio") != 8
    ):
        raise ValueError("Suite must keep the selected C1000 ratio8 binding")
    _digest(checkpoint.get("config_sha256"), "checkpoint.config_sha256")
    if not isinstance(checkpoint.get("path"), str) or not checkpoint["path"]:
        raise ValueError("Checkpoint path must be explicit")

    runtime = suite.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("Suite runtime contract is missing")
    exact_runtime = {
        "server_module": "benchmarks.memory_runtime.event_native_server",
        "official_one_task": "official_one_task.py",
        "source_profile": "openai-single-task-v1",
        "view_mode": "ac_native_s0_lexical_raw_reserve_failed_operation",
        "compression_policy": "always-compress-v1",
        "history_view_protocol": "fixed-budget-main",
        "ratio": 8,
        "dtype": "bfloat16",
        "device": "cpu",
        "generation_backend": "sglang",
        "session_cache_policy": "external-sglang-content-addressed-chunks-v1",
        "prefill_chunk_size": 256,
        "decode_strategy": "incremental",
        "no_raw_snapshot": True,
    }
    for name, expected in exact_runtime.items():
        if runtime.get(name) != expected:
            raise ValueError(f"Frozen suite changed runtime.{name}")
    backend_url = runtime.get("sglang_backend_url")
    parsed_backend_url = urlsplit(backend_url) if isinstance(backend_url, str) else None
    if (
        parsed_backend_url is None
        or parsed_backend_url.scheme != "http"
        or not parsed_backend_url.netloc
        or parsed_backend_url.username is not None
        or parsed_backend_url.password is not None
        or parsed_backend_url.path not in {"", "/"}
        or parsed_backend_url.query
        or parsed_backend_url.fragment
    ):
        raise ValueError("Frozen suite requires a bare SGLang backend URL")
    backend_timeout = runtime.get("sglang_timeout_seconds")
    if (type(backend_timeout) not in (int, float)
            or not math.isfinite(backend_timeout) or backend_timeout <= 0):
        raise ValueError("Frozen suite requires a finite SGLang backend timeout")
    if runtime.get("npu_allocator_metrics") is not False:
        raise ValueError("External SGLang generation cannot use process-local allocator metrics")
    if runtime.get("sampling") != {"mode": "greedy", "temperature": 0, "seed": 0}:
        raise ValueError("Frozen suite must use greedy seed0 generation")
    if suite.get("max_new_tokens_by_benchmark") != TASK_MAX_NEW_TOKENS:
        raise ValueError("Frozen suite changed benchmark-specific generation caps")
    for field in (
        "runtime_root",
        "controller",
        "eval_policy",
        "eval_capacity",
        "shadow_feature_config",
        "official_one_task",
    ):
        path = _safe_relative(package_root, runtime.get(field), f"runtime.{field}")
        if not path.exists():
            raise ValueError(f"Frozen runtime dependency is missing: {field}")

    interpreters = suite.get("interpreters")
    if not isinstance(interpreters, Mapping) or any(
        not isinstance(interpreters.get(name), str) or not interpreters[name]
        for name in ("server", "official")
    ):
        raise ValueError("Frozen suite requires explicit server and official interpreters")
    if not isinstance(suite.get("candidate_id"), str) or not suite["candidate_id"]:
        raise ValueError("Frozen suite requires a nonempty candidate_id")

    eval_policy = read_json(
        _safe_relative(package_root, runtime["eval_policy"], "runtime.eval_policy")
    )
    policy = eval_policy.get("policy") if isinstance(eval_policy, Mapping) else None
    if not isinstance(policy, Mapping) or (
        policy.get("history_budget_bytes") != 113246208
        or policy.get("workspace_budget_bytes") != 113246208
        or policy.get("lease_decisions") != 0
    ):
        raise ValueError("Frozen eval policy changed exact B0 or lease policy")
    eval_capacity = read_json(
        _safe_relative(package_root, runtime["eval_capacity"], "runtime.eval_capacity")
    )
    capacity = eval_capacity.get("capacity") if isinstance(eval_capacity, Mapping) else None
    if not isinstance(capacity, Mapping) or capacity.get("max_sequence_tokens") != 40960:
        raise ValueError("Frozen eval capacity changed context40960")
    shadow = read_json(
        _safe_relative(
            package_root,
            runtime["shadow_feature_config"],
            "runtime.shadow_feature_config",
        )
    )
    if not isinstance(shadow, Mapping) or shadow.get("enabled") is not True:
        raise ValueError("Frozen delivery suite requires shadow feature capture")

    limits = suite.get("limits")
    required_limits = {
        "generation_calls_per_task": 96,
        "extraction_calls_per_task": 1152,
        "stage_wall_seconds": 21600,
    }
    if not isinstance(limits, Mapping) or any(
        limits.get(name) != expected for name, expected in required_limits.items()
    ):
        raise ValueError("Frozen suite changed the generation, extraction, or stage cap")
    for name in (
        "max_decisions_per_task",
        "server_wall_seconds_per_task",
        "official_wall_seconds_per_task",
        "server_ready_seconds",
    ):
        value = limits.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"limits.{name} must be positive and finite")
    if limits["max_decisions_per_task"] > limits["generation_calls_per_task"]:
        raise ValueError("Decision cap cannot exceed the shared generation-call cap")
    if suite.get("retry_contract") != {
        "automatic_reruns": 0,
        "server_start_retries": 0,
        "official_worker_retries": 0,
        "transport_retries": 0,
    }:
        raise ValueError("Frozen suite retry contract changed")
    if verify_files:
        verify_package(package_root, suite)
    return tasks


def verify_checkpoint(suite: Mapping[str, Any]) -> dict[str, Any]:
    binding = suite["checkpoint"]
    checkpoint = Path(binding["path"]).resolve()
    config = checkpoint / "config.json"
    if not config.is_file() or sha256(config) != binding["config_sha256"]:
        raise ValueError("Selected C1000 checkpoint config is missing or changed")
    return {
        "path": str(checkpoint),
        "config_sha256": binding["config_sha256"],
        "selected_arm": "C",
        "selected_step": 1000,
        "ratio": 8,
    }


def server_command(
    suite: Mapping[str, Any], package_root: Path, row: Mapping[str, Any],
    output: Path, port: int,
) -> list[str]:
    runtime, limits = suite["runtime"], suite["limits"]
    root = _safe_relative(package_root, runtime["runtime_root"], "runtime_root")
    command = [
        suite["interpreters"]["server"],
        "-m",
        runtime["server_module"],
        "--checkpoint",
        suite["checkpoint"]["path"],
        "--out",
        str(output / "server"),
        "--run-id",
        f"{suite['suite_id']}__{row['task_key']}",
        "--model-name",
        (
            TOOLSANDBOX_OFFICIAL_MODEL_NAME
            if row["benchmark"] == "toolsandbox"
            else runtime.get("model_name", suite["candidate_id"])
        ),
        "--benchmark",
        row["benchmark"],
        "--source-profile",
        runtime["source_profile"],
        "--view-mode",
        runtime["view_mode"],
        "--compression-policy",
        runtime["compression_policy"],
        "--history-view-protocol",
        runtime["history_view_protocol"],
        "--ratio",
        str(runtime["ratio"]),
        "--max-new-tokens",
        str(row["max_new_tokens"]),
        "--decode-strategy",
        runtime["decode_strategy"],
        "--prefill-chunk-size",
        str(runtime["prefill_chunk_size"]),
        "--task-ids",
        row["task_id"],
        "--max-decisions",
        str(limits["max_decisions_per_task"]),
        "--max-generation-calls",
        str(limits["generation_calls_per_task"]),
        "--max-extraction-calls",
        str(limits["extraction_calls_per_task"]),
        "--eval-policy",
        str(_safe_relative(package_root, runtime["eval_policy"], "eval_policy")),
        "--eval-capacity",
        str(_safe_relative(package_root, runtime["eval_capacity"], "eval_capacity")),
        "--s0-config",
        str(_safe_relative(package_root, runtime["controller"], "controller")),
        "--shadow-feature-config",
        str(
            _safe_relative(
                package_root, runtime["shadow_feature_config"], "shadow_feature_config"
            )
        ),
        "--max-wall-seconds",
        str(limits["server_wall_seconds_per_task"]),
        "--generation-backend",
        runtime["generation_backend"],
        "--sglang-backend-url",
        runtime["sglang_backend_url"],
        "--sglang-timeout-seconds",
        str(runtime["sglang_timeout_seconds"]),
        "--device",
        runtime["device"],
        "--dtype",
        runtime["dtype"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--torch-threads",
        str(runtime.get("torch_threads", 4)),
    ]
    if runtime.get("npu_allocator_metrics", False):
        command.append("--npu-allocator-metrics")
    extraction = runtime.get("extraction_policy", "all-eligible")
    if extraction not in ("all-eligible", "retained-gist"):
        raise ValueError("Unsupported frozen extraction policy")
    if extraction != "all-eligible":
        command.extend(["--extraction-policy", extraction])
    command.append("--no-raw-snapshot")
    if root != package_root / "runtime":
        raise ValueError("Frozen runtime root must be package_root/runtime")
    return command


def official_command(
    suite: Mapping[str, Any], package_root: Path, task_path: Path,
    server_manifest: Path, official_out: Path, base_url: str, wall_seconds: float,
) -> list[str]:
    return [
        suite["interpreters"]["official"],
        str(
            _safe_relative(
                package_root,
                suite["runtime"]["official_one_task"],
                "runtime.official_one_task",
            )
        ),
        "--task",
        str(task_path),
        "--base-url",
        base_url,
        "--server-manifest",
        str(server_manifest),
        "--out",
        str(official_out),
        "--max-wall-seconds",
        str(wall_seconds),
    ]


def preview(suite: Mapping[str, Any], package_root: Path, port_base: int) -> dict[str, Any]:
    tasks = validate_suite(suite, package_root)
    _validate_ports(port_base, len(tasks))
    cells = []
    for row in tasks:
        shard = Path("<output>") / "task_shards" / row["task_key"]
        task_path = shard / "task.json"
        cells.append(
            {
                "ordinal": row["ordinal"],
                "benchmark": row["benchmark"],
                "task_id": row["task_id"],
                "task_key": row["task_key"],
                "server": server_command(
                    suite, package_root, row, shard, port_base + row["ordinal"]
                ),
                "official": official_command(
                    suite,
                    package_root,
                    task_path,
                    shard / "server" / "ready.json",
                    shard / "official",
                    f"http://127.0.0.1:{port_base + row['ordinal']}/v1",
                    suite["limits"]["official_wall_seconds_per_task"],
                ),
            }
        )
    return {
        "schema": "a-history-multibench-preview-v1",
        "status": "passed_cpu_static_preview_no_model_or_scorer",
        "suite_id": suite["suite_id"],
        "fixed_denominator": len(tasks),
        "task_order": [row["task_key"] for row in tasks],
        "cells": cells,
        "automatic_reruns": 0,
        "model_calls": 0,
        "scorer_calls": 0,
        "network_calls": 0,
    }


def _validate_ports(port_base: int, count: int) -> None:
    if type(port_base) is not int or not 1024 <= port_base <= 65535 - count + 1:
        raise ValueError("port-base must leave one valid fixed port per task")


def _process_environment(package_root: Path, *, inherit_pythonpath: bool = True) -> dict[str, str]:
    environment = os.environ.copy()
    runtime = package_root / "runtime"
    paths = [str(runtime / "python"), str(runtime)]
    if inherit_pythonpath and environment.get("PYTHONPATH"):
        paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(paths)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    bypasses = []
    for name in ("NO_PROXY", "no_proxy"):
        for value in environment.get(name, "").split(","):
            value = value.strip()
            if value and value not in bypasses:
                bypasses.append(value)
    for value in ("127.0.0.1", "localhost"):
        if value not in bypasses:
            bypasses.append(value)
    loopback_bypass = ",".join(bypasses)
    environment["NO_PROXY"] = loopback_bypass
    environment["no_proxy"] = loopback_bypass
    return environment


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        try:
            handle.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _stop_server(process: subprocess.Popen, supervisor_path: Path) -> int | None:
    child_pid = None
    if supervisor_path.exists():
        try:
            child_pid = read_json(supervisor_path).get("child_pid")
        except Exception:
            child_pid = None
    if process.poll() is not None:
        return process.returncode
    try:
        if os.name == "posix" and isinstance(child_pid, int) and child_pid > 1:
            os.kill(child_pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix" and isinstance(child_pid, int) and child_pid > 1:
                os.kill(child_pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        return process.wait(timeout=10)


def _stop_worker(process: subprocess.Popen) -> int | None:
    if process.poll() is not None:
        return process.returncode
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        return process.wait(timeout=10)


def _artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def _feature_status(value: Any, name: str) -> str:
    if not isinstance(value, Mapping):
        return "missing"
    entry = value.get(name)
    if not isinstance(entry, Mapping) or not isinstance(entry.get("status"), str):
        return "missing"
    return entry["status"]


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _compression_receipt(
    step: Mapping[str, Any], generation: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Extract the same-prefix H/A/S fields used by compression_metrics.py."""
    errors: list[str] = []
    controller = generation.get("controller")
    if not isinstance(controller, Mapping):
        return {}, ["missing_or_invalid_controller"]
    ratio = controller.get("compression_ratio")
    if not isinstance(ratio, Mapping):
        return {}, ["missing_or_invalid_compression_ratio"]
    if ratio.get("schema") != "a-same-prefix-compression-ratio-v1":
        errors.append("unexpected_compression_ratio_schema")
    reference = controller.get("same_prefix_full_reference")
    if not isinstance(reference, Mapping):
        errors.append("missing_or_invalid_same_prefix_full_reference")
        reference = {}

    fields = {
        "full_history_bytes": ratio.get("full_history_bytes"),
        "common_live_bytes": ratio.get("common_live_bytes"),
        "active_history_bytes": ratio.get("active_history_bytes"),
        "active_gist_bytes": ratio.get("active_gist_bytes"),
        "active_raw_history_bytes": ratio.get("active_raw_history_bytes"),
    }
    for name, value in fields.items():
        if not _nonnegative_int(value):
            errors.append(f"invalid_{name}")
    h, s, a = (
        fields["full_history_bytes"],
        fields["common_live_bytes"],
        fields["active_history_bytes"],
    )
    if _nonnegative_int(a) and controller.get("actual_history_bytes") != a:
        errors.append("controller_actual_history_bytes_mismatch")
    if _nonnegative_int(h) and reference.get("full_history_bytes") != h:
        errors.append("same_prefix_full_history_bytes_mismatch")
    if _nonnegative_int(s) and reference.get("common_live_bytes") != s:
        errors.append("same_prefix_common_live_bytes_mismatch")
    if all(
        _nonnegative_int(fields[name])
        for name in ("active_history_bytes", "active_gist_bytes", "active_raw_history_bytes")
    ) and fields["active_gist_bytes"] + fields["active_raw_history_bytes"] != a:
        errors.append("active_history_component_sum_mismatch")

    coverage = controller.get("source_coverage")
    complete_coverage = (
        coverage.get("complete_history_coverage")
        if isinstance(coverage, Mapping)
        else None
    )
    if not isinstance(complete_coverage, bool):
        errors.append("invalid_complete_history_coverage")
    history_reduction = (
        h / a
        if _nonnegative_int(h) and h > 0 and _nonnegative_int(a) and a > 0
        else None
    )
    full_input_reduction = (
        (s + h) / (s + a)
        if history_reduction is not None and _nonnegative_int(s)
        else None
    )
    receipt = {
        "decision_key": step.get("decision_key"),
        "phase": generation.get("phase"),
        "configured_ratio": controller.get("requested_ratio"),
        **fields,
        "full_input_bytes": s + h if _nonnegative_int(s) and _nonnegative_int(h) else None,
        "active_input_bytes": s + a if _nonnegative_int(s) and _nonnegative_int(a) else None,
        "system_active_history_reduction": history_reduction,
        "system_full_input_reduction": full_input_reduction,
        "complete_source_coverage": complete_coverage,
        "observed_b0_violation": _nonnegative_int(a) and a > 113246208,
        "errors": list(errors),
    }
    return receipt, errors


def collect_server_evidence(server_out: Path) -> dict[str, Any]:
    steps_path, final_path = server_out / "steps.jsonl", server_out / "final.json"
    evidence = {
        "server_final": _artifact(final_path),
        "steps": _artifact(steps_path),
        "attempts": _artifact(server_out / "attempts.jsonl"),
        "step_count": 0,
        "generation_count": 0,
        "shadow_feature_count": 0,
        "feature_status_counts": {"prefill": {}, "memgen": {}, "tool_name": {}},
        "compression_receipts": [],
        "compression_receipt_errors": 0,
        "errors": [],
    }
    if final_path.is_file():
        try:
            final = read_json(final_path)
            evidence["server_status"] = final.get("status")
            evidence["server_cost_summary"] = copy.deepcopy(final.get("cost_summary"))
        except Exception as error:
            evidence["errors"].append(f"final.json: {type(error).__name__}: {error}")
    if not steps_path.is_file():
        evidence["errors"].append("steps.jsonl missing")
        return evidence
    for line_number, line in enumerate(steps_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            step = json.loads(line)
        except Exception as error:
            evidence["errors"].append(
                f"steps.jsonl line {line_number}: {type(error).__name__}: {error}"
            )
            continue
        evidence["step_count"] += 1
        for generation in step.get("generation_trace") or ():
            evidence["generation_count"] += 1
            receipt, receipt_errors = _compression_receipt(step, generation)
            evidence["compression_receipts"].append(receipt)
            evidence["compression_receipt_errors"] += len(receipt_errors)
            evidence["errors"].extend(
                f"steps.jsonl line {line_number} generation {evidence['generation_count']}: {error}"
                for error in receipt_errors
            )
            stats = (generation.get("generation") or {}).get("stats") or {}
            features = stats.get("shadow_features")
            if isinstance(features, Mapping):
                evidence["shadow_feature_count"] += 1
            for name in ("prefill", "memgen", "tool_name"):
                status = _feature_status(features, name)
                counts = evidence["feature_status_counts"][name]
                counts[status] = counts.get(status, 0) + 1
    return evidence


def _valid_score(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping) and value:
        return (
            any(item is not None and _valid_score(item) for item in value.values())
            and all(item is None or _valid_score(item) for item in value.values())
        )
    return False


def _official_artifacts(result_path: Path, value: Any) -> tuple[list[dict[str, Any]], list[str]]:
    if not isinstance(value, list):
        return [], ["official_artifacts_not_a_list"]
    output_root = result_path.parent.resolve()
    artifacts: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, Mapping)
            or not {"path", "sha256"} <= set(item)
            or set(item) - {"path", "sha256", "bytes"}
        ):
            errors.append(f"official_artifact_{index}_contract_invalid")
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"official_artifact_{index}_path_invalid")
            continue
        path = Path(raw_path)
        path = path.resolve() if path.is_absolute() else (output_root / path).resolve()
        if path != output_root and output_root not in path.parents:
            errors.append(f"official_artifact_{index}_outside_worker_output")
            continue
        try:
            expected = _digest(item.get("sha256"), f"official artifact {index} hash")
        except ValueError:
            errors.append(f"official_artifact_{index}_hash_invalid")
            continue
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"official_artifact_{index}_missing_or_changed")
            continue
        if "bytes" in item and item["bytes"] != path.stat().st_size:
            errors.append(f"official_artifact_{index}_size_mismatch")
            continue
        artifacts.append(
            {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}
        )
    return artifacts, errors


def read_official_result(
    path: Path,
    row: Mapping[str, Any],
    worker_returncode: int | None,
    expected_max_wall_seconds: float,
) -> dict[str, Any]:
    artifact = _artifact(path)
    if artifact is None:
        return {
            "status": "infra_failed",
            "reason": "official_result_missing",
            "official_score": None,
            "scored": False,
            "artifact": None,
        }
    try:
        result = read_json(path)
    except Exception as error:
        return {
            "status": "infra_failed",
            "reason": f"official_result_invalid_json:{type(error).__name__}",
            "official_score": None,
            "scored": False,
            "artifact": artifact,
        }
    official_artifacts, artifact_errors = _official_artifacts(
        path, result.get("official_artifacts")
    )
    elapsed = result.get("elapsed_seconds")
    maximum = result.get("max_wall_seconds")
    timing_valid = (
        type(elapsed) in (int, float)
        and math.isfinite(float(elapsed))
        and elapsed >= 0
        and type(maximum) in (int, float)
        and math.isfinite(float(maximum))
        and math.isclose(float(maximum), expected_max_wall_seconds, rel_tol=1e-9, abs_tol=1e-6)
    )
    valid = (
        worker_returncode == 0
        and result.get("schema") == TASK_RESULT_SCHEMA
        and result.get("status") == "completed"
        and result.get("scored") is True
        and result.get("benchmark") == row["benchmark"]
        and result.get("task_id") == row["task_id"]
        and _valid_score(result.get("official_score"))
        and isinstance(result.get("benchmark_summary"), Mapping)
        and isinstance(result.get("source_binding"), list)
        and result.get("error") is None
        and timing_valid
        and not artifact_errors
    )
    if not valid:
        return {
            "status": "infra_failed",
            "reason": "official_result_contract_or_worker_failure",
            "official_score": None,
            "scored": False,
            "artifact": artifact,
            "reported_status": result.get("status"),
            "contract_errors": artifact_errors,
        }
    return {
        "status": "official_scored",
        "reason": None,
        "official_score": copy.deepcopy(result["official_score"]),
        "scored": True,
        "artifact": artifact,
        "official_artifacts": official_artifacts,
        "benchmark_summary": copy.deepcopy(result.get("benchmark_summary")),
        "source_binding": copy.deepcopy(result.get("source_binding")),
        "elapsed_seconds": float(elapsed),
        "max_wall_seconds": float(maximum),
    }


def execute_one_task(
    suite: Mapping[str, Any], package_root: Path, row: Mapping[str, Any],
    shard: Path, port: int, stage_deadline: float,
) -> dict[str, Any]:
    started = time.monotonic()
    environment = _process_environment(package_root)
    server_out = shard / "server"
    supervisor_path = shard / "server.supervisor.json"
    ready_path = server_out / "ready.json"
    server_log_path = shard / "server.log"
    official_out = shard / "official"
    official_log_path = shard / "official.log"
    official_result_path = official_out / "result.json"
    server: subprocess.Popen | None = None
    worker: subprocess.Popen | None = None
    server_returncode = worker_returncode = None
    worker_wall_seconds: int | None = None
    reason = None
    if not _port_is_free(port):
        return {
            "outcome": "infra_failed_in_denominator",
            "reason": "fixed_port_unavailable",
            "official_score": None,
            "scored": False,
            "wall_seconds": time.monotonic() - started,
        }
    try:
        with server_log_path.open("x", encoding="utf-8", newline="\n") as log:
            server = subprocess.Popen(
                server_command(suite, package_root, row, shard, port),
                cwd=_safe_relative(
                    package_root, suite["runtime"]["runtime_root"], "runtime_root"
                ),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name == "posix",
            )
        ready_deadline = min(
            stage_deadline - CLEANUP_RESERVE_SECONDS,
            time.monotonic() + suite["limits"]["server_ready_seconds"],
        )
        while (
            not ready_path.exists()
            and server.poll() is None
            and time.monotonic() < ready_deadline
        ):
            time.sleep(0.25)
        if not ready_path.exists():
            reason = "server_not_ready"
        else:
            ready = read_json(ready_path)
            base_url = ready.get("base_url")
            expected_model_name = (
                TOOLSANDBOX_OFFICIAL_MODEL_NAME
                if row["benchmark"] == "toolsandbox"
                else suite["runtime"].get("model_name", suite["candidate_id"])
            )
            if (
                ready.get("status") != "ready"
                or base_url != f"http://127.0.0.1:{port}/v1"
                or ready.get("source_profile") != "openai-single-task-v1"
                or ready.get("allowed_task_ids") != [row["task_id"]]
                or ready.get("model_name") != expected_model_name
                or ready.get("max_new_tokens") != row["max_new_tokens"]
            ):
                reason = "server_ready_contract_mismatch"
            else:
                remaining = min(
                    suite["limits"]["official_wall_seconds_per_task"],
                    stage_deadline - time.monotonic() - CLEANUP_RESERVE_SECONDS,
                )
                if remaining <= 0:
                    reason = "stage_wall_exhausted_before_official_worker"
                else:
                    worker_wall_seconds = max(1, math.floor(remaining))
                    with official_log_path.open("x", encoding="utf-8", newline="\n") as log:
                        worker = subprocess.Popen(
                            official_command(
                                suite,
                                package_root,
                                shard / "task.json",
                                ready_path,
                                official_out,
                                base_url,
                                worker_wall_seconds,
                            ),
                            cwd=package_root,
                            env=_process_environment(package_root, inherit_pythonpath=False),
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=os.name == "posix",
                        )
                        try:
                            worker_returncode = worker.wait(timeout=worker_wall_seconds)
                        except subprocess.TimeoutExpired:
                            reason = "official_worker_wall_timeout"
                            worker_returncode = _stop_worker(worker)
                        worker = None
    except BaseException as error:
        reason = f"{type(error).__name__}: {error}"
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        if worker is not None:
            worker_returncode = _stop_worker(worker)
        if server is not None:
            server_returncode = _stop_server(server, supervisor_path)

    evidence = collect_server_evidence(server_out)
    official = read_official_result(
        official_result_path,
        row,
        worker_returncode,
        worker_wall_seconds or int(suite["limits"]["official_wall_seconds_per_task"]),
    )
    if reason is not None or evidence["errors"]:
        official = {
            **official,
            "status": "infra_failed",
            "scored": False,
            "official_score": None,
            "reason": reason or "server_evidence_incomplete",
        }
    return {
        "outcome": (
            "official_scored" if official["status"] == "official_scored"
            else "infra_failed_in_denominator"
        ),
        "reason": official["reason"],
        "official_score": official["official_score"],
        "scored": official["scored"],
        "official_result": official,
        "server_evidence": evidence,
        "server_returncode": server_returncode,
        "worker_returncode": worker_returncode,
        "server_log": _artifact(server_log_path),
        "official_log": _artifact(official_log_path),
        "wall_seconds": time.monotonic() - started,
    }


def _initial_outcome(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "ordinal": row["ordinal"],
        "benchmark": row["benchmark"],
        "task_id": row["task_id"],
        "task_key": row["task_key"],
        "max_new_tokens": row["max_new_tokens"],
        "in_fixed_denominator": True,
        "outcome": "not_started_in_denominator",
        "official_score": None,
        "scored": False,
    }


def run_suite(
    suite: Mapping[str, Any], package_root: Path, output: Path, port_base: int,
    *, executor: Callable[..., dict[str, Any]] = execute_one_task,
    check_checkpoint: bool = True,
) -> int:
    tasks = validate_suite(suite, package_root)
    _validate_ports(port_base, len(tasks))
    if output.exists():
        raise FileExistsError(f"Suite output already exists; automatic rerun disabled: {output}")
    checkpoint = verify_checkpoint(suite) if check_checkpoint else copy.deepcopy(suite["checkpoint"])
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + suite["limits"]["stage_wall_seconds"]
    record = {
        "schema": STAGE_SCHEMA,
        "status": "running_fixed_manifest",
        "suite_id": suite["suite_id"],
        "candidate_id": suite["candidate_id"],
        "checkpoint": checkpoint,
        "fixed_denominator": len(tasks),
        "task_manifest_sha256": suite["task_manifest_sha256"],
        "task_order": [row["task_key"] for row in tasks],
        "automatic_reruns": 0,
        "task_outcomes": [_initial_outcome(row) for row in tasks],
        "official_scored_tasks": 0,
        "infra_failed_tasks": 0,
        "completed_task_cells": 0,
        "wall_seconds": 0.0,
        "wall_seconds_final": False,
    }
    stage_path = output / "stage.json"
    save_json(stage_path, record)
    try:
        for row in tasks:
            outcome = record["task_outcomes"][row["ordinal"]]
            if time.monotonic() >= deadline - CLEANUP_RESERVE_SECONDS:
                for pending in record["task_outcomes"][row["ordinal"] :]:
                    pending["outcome"] = "not_started_stage_wall_in_denominator"
                record["status"] = "stage_wall_exhausted"
                break
            # A fresh server imports source for every row, so recheck the package
            # immediately before each import rather than trusting launch-time state.
            verify_package(package_root, suite)
            shard = output / "task_shards" / row["task_key"]
            shard.mkdir(parents=True, exist_ok=False)
            official_row = {
                key: copy.deepcopy(value)
                for key, value in row.items()
                if key not in {"ordinal", "task_key"}
            }
            save_json(shard / "task.json", official_row)
            outcome["outcome"] = "running"
            save_json(stage_path, record)
            try:
                result = executor(
                    suite,
                    package_root,
                    row,
                    shard,
                    port_base + row["ordinal"],
                    deadline,
                )
                if not isinstance(result, Mapping) or result.get("outcome") not in {
                    "official_scored",
                    "infra_failed_in_denominator",
                }:
                    raise ValueError("Task executor returned an invalid terminal outcome")
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                result = {
                    "outcome": "infra_failed_in_denominator",
                    "reason": f"executor_exception:{type(error).__name__}:{error}",
                    "official_score": None,
                    "scored": False,
                }
            outcome.update(copy.deepcopy(dict(result)))
            if outcome["outcome"] == "official_scored":
                record["official_scored_tasks"] += 1
            else:
                record["infra_failed_tasks"] += 1
            record["completed_task_cells"] = row["ordinal"] + 1
            record["wall_seconds"] = time.monotonic() - started
            save_json(stage_path, record)
        else:
            record["status"] = (
                "completed_fixed_manifest"
                if record["infra_failed_tasks"] == 0
                else "completed_fixed_manifest_with_infra_failures"
            )
    except BaseException as error:
        record["status"] = "interrupted_without_rerun"
        record["terminal_error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["wall_seconds"] = time.monotonic() - started
        record["wall_seconds_final"] = True
        record["denominator_preserved"] = len(record["task_outcomes"]) == len(tasks)
        record["unscored_in_denominator"] = sum(
            not row["scored"] for row in record["task_outcomes"]
        )
        save_json(stage_path, record)
    if record["status"] == "stage_wall_exhausted":
        return 4
    return 0 if record["infra_failed_tasks"] == 0 else 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preview", "run"))
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port-base", type=int, default=42000)
    parser.add_argument("--preview-output", type=Path)
    args = parser.parse_args(argv)
    package_root = args.package_root.resolve()
    suite_path = args.suite.resolve() if args.suite else package_root / "suite.json"
    suite = read_json(suite_path)
    if args.action == "preview":
        result = preview(suite, package_root, args.port_base)
        if args.preview_output:
            save_json(args.preview_output, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.output is None:
        parser.error("run requires --output")
    return run_suite(suite, package_root, args.output.resolve(), args.port_base)


if __name__ == "__main__":
    raise SystemExit(main())
