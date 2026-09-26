"""Freeze and dispatch the evidence_sets_v1 H0/R1 D20 evaluation.

The package built here has four independent, no-rerun lanes.  Each lane owns
one actor engine, one physical NPU, and one fixed 20-task BFCL denominator.
Remote execution refuses occupied devices, occupied ports, source drift, and
pre-existing output before it starts model work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT_ROOT = REPO / "outputs/history_system_search/evidence_sets_v1"
DEFAULT_SOURCE = OUTPUT_ROOT / "prepared"
DEFAULT_OUTPUT = OUTPUT_ROOT / "eval/prepared_v1"
DEFAULT_REMOTE_ROOT = PurePosixPath(
    "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_v1"
)
REMOTE_SOURCE = PurePosixPath(
    "/home/liuyancheng/c2kv-evidence-sets-20260916/prepared_v2"
)
CHECKPOINT = PurePosixPath(
    "/home/liuyancheng/c2kv-b-final-20260912/checkpoints/"
    "b_history/arm-C/seed-42/checkpoint-1000"
)
BENCHMARK_DIR = PurePosixPath(
    "/home/liuyancheng/benchmarks/gorilla/berkeley-function-call-leaderboard"
)
SGL_PYTHON = "/home/liuyancheng/envs/sgl/bin/python"
BFCL_PYTHON = "/home/liuyancheng/envs/bench/bin/python"
MODEL_PATHS = {
    "embedding": PurePosixPath(
        "/home/liuyancheng/c2kv-evidence-sets-20260916/models/"
        "Qwen3-Embedding-0.6B"
    ),
    "reranker": PurePosixPath(
        "/home/liuyancheng/c2kv-evidence-sets-20260916/models/"
        "Qwen3-Reranker-0.6B"
    ),
    "selector": PurePosixPath("/home/liuyancheng/c2kv/models/Qwen3-4B-Instruct-2507"),
}
LANES = {
    "C0": {
        "selector": "candidate_rule",
        "physical_device": 0,
        "engine_port": 36300,
        "task_port_base": 36400,
        "model_smokes": ("embedding",),
    },
    "C2": {
        "selector": "reranker",
        "physical_device": 1,
        "engine_port": 36310,
        "task_port_base": 36420,
        "model_smokes": ("embedding", "reranker"),
    },
    "C3": {
        "selector": "local_llm",
        "physical_device": 2,
        "engine_port": 36320,
        "task_port_base": 36440,
        "model_smokes": ("embedding", "selector"),
    },
    "C5": {
        "selector": "parameter_source",
        "physical_device": 3,
        "engine_port": 36330,
        "task_port_base": 36460,
        "model_smokes": ("embedding",),
    },
}
TASK_PATTERN = re.compile(r"multi_turn_[a-z0-9_]+_([0-9]+)$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _save(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix != ".pyc"
    }


def _copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
    )


def task_groups(task_ids: Sequence[str]) -> set[int]:
    groups: set[int] = set()
    for task_id in task_ids:
        match = TASK_PATTERN.fullmatch(task_id)
        if match is None:
            raise ValueError(f"Unsupported BFCL task ID: {task_id}")
        groups.add(int(match.group(1)))
    return groups


def validate_task_isolation(d20: Mapping[str, Any], training: Mapping[str, Any]) -> dict[str, Any]:
    d20_ids = d20.get("task_ids")
    training_ids = training.get("task_ids")
    if not isinstance(d20_ids, list) or len(d20_ids) != 20 or len(set(d20_ids)) != 20:
        raise ValueError("D20 must contain exactly 20 unique task IDs")
    if not isinstance(training_ids, list) or not training_ids:
        raise ValueError("T02 training manifest must contain explicit task_ids")
    exact_overlap = sorted(set(d20_ids) & set(training_ids))
    group_overlap = sorted(task_groups(d20_ids) & task_groups(training_ids))
    if exact_overlap or group_overlap:
        raise ValueError(
            f"D20 overlaps T02 training: exact={exact_overlap}, groups={group_overlap}"
        )
    return {
        "d20_task_count": len(d20_ids),
        "d20_groups": sorted(task_groups(d20_ids)),
        "training_task_count": len(training_ids),
        "training_groups": sorted(task_groups(training_ids)),
        "exact_overlap": [],
        "canonical_group_overlap": [],
    }


def lane_specs() -> list[dict[str, Any]]:
    result = []
    for name, raw in LANES.items():
        row = {"name": name, **raw}
        row["model_smokes"] = list(row["model_smokes"])
        row["task_ports"] = list(
            range(row["task_port_base"], row["task_port_base"] + 20)
        )
        result.append(row)
    all_ports = [
        port
        for row in result
        for port in [row["engine_port"], *row["task_ports"]]
    ]
    if len(all_ports) != len(set(all_ports)):
        raise ValueError("Lane ports overlap")
    if {row["physical_device"] for row in result} != {0, 1, 2, 3}:
        raise ValueError("Evidence lanes must bind exactly physical devices 0..3")
    return result


def device_processes(output: str) -> dict[int, list[int]]:
    if "Process id" not in output or "HBM-Usage" not in output:
        raise ValueError("Unknown npu-smi output layout")
    result = {
        int(match.group(1)): []
        for match in re.finditer(r"^\|\s+(\d+)\s+910\w+\s*\|", output, re.M)
    }
    if set(result) != set(range(8)):
        raise ValueError("Expected all eight physical devices")
    for line in output.splitlines():
        match = re.match(r"^\|\s+(\d+)\s+(\d+)\s*\|\s+(\d+)\s*\|", line)
        if match:
            result[int(match.group(1))].append(int(match.group(3)))
    return result


def _static_manifest(package: Path) -> dict[str, Any]:
    excluded = {"static_files.json"}
    files = {
        name: digest
        for name, digest in _tree_hashes(package).items()
        if name not in excluded and not name.startswith("sglang/")
    }
    return {
        "schema": "evidence-sets-eval-static-files-v1",
        "file_count": len(files),
        "files": files,
    }


def _verify_hashes(root: Path, manifest: Mapping[str, Any]) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Source manifest requires a nonempty files mapping")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if root.resolve() not in path.parents:
            raise ValueError(f"Manifest path escapes package: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Frozen source is missing: {path}")
        actual = _sha(path)
        if actual != expected:
            raise RuntimeError(
                f"Frozen source changed: {relative}; expected={expected}, actual={actual}"
            )


def verify_package(package: Path, *, require_sglang: bool = True) -> dict[str, Any]:
    package = package.resolve()
    _verify_hashes(package, _read(package / "static_files.json"))
    sglang = _read(package / "sglang_files.json")
    if require_sglang:
        _verify_hashes(package, sglang)
    return {
        "schema": "evidence-sets-eval-package-verification-v1",
        "status": "passed",
        "package": str(package),
        "static_file_count": _read(package / "static_files.json")["file_count"],
        "sglang_file_count": sglang["file_count"],
        "verified_at": _now(),
    }


def _load_evidence_module(source_history: Path):
    import importlib

    source = str(source_history.resolve())
    sys.path.insert(0, source)
    try:
        module = importlib.import_module("evidence_sets")
    finally:
        sys.path.pop(0)
    if Path(module.__file__).resolve() != (source_history / "evidence_sets.py").resolve():
        raise RuntimeError("Loaded evidence_sets from an unexpected source snapshot")
    return module


def _controller_for_lane(module: Any, runtime: Path, selector: str) -> tuple[dict, dict]:
    gp, _ = module.build_config(
        history="H0",
        selector=selector,
        candidate_pool_size=8,
        selected_evidence_max=4,
        fallback_256=False,
        reserve_tokens=0,
        embedding_model=str(MODEL_PATHS["embedding"]),
        embedding_revision=None,
        embedding_device="npu:0",
        reranker_model=str(MODEL_PATHS["reranker"]),
        reranker_revision=None,
        reranker_device="npu:0",
        selector_model=str(MODEL_PATHS["selector"]),
        selector_revision=None,
        selector_device="npu:0",
    )
    for role in ("embedding", "reranker", "selector"):
        gp["local_models"][role]["dtype"] = "bfloat16"
    gp = module._validate_with_active_runtime(gp)
    sys.path[:0] = [str(runtime / "python"), str(runtime)]
    try:
        from benchmarks.memory_runtime.recovery.experiment_config import (
            configure_controller,
        )
    finally:
        del sys.path[:2]
    base = _read(runtime / "configs/controller.json")
    controller = configure_controller(base, gp)
    return controller, gp


def _design(
    *,
    current: Mapping[str, Any],
    runtime: Path,
    lane: Mapping[str, Any],
    controller: Mapping[str, Any],
    tasks: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    policy = _read(runtime / "configs/eval_policy.json")
    capacity = _read(runtime / "configs/eval_capacity.json")
    candidate_id = f"evidence_sets_v1_h0_{lane['name'].lower()}_r1"
    value = {
        key: current[key]
        for key in (
            "route",
            "compression_policy",
            "history_view_protocol",
            "decode_strategy",
            "prefill_chunk_size",
            "sampling",
            "session_cache_policy",
            "retry_contract",
        )
    }
    value.update(
        {
            "schema": "a-history-system-candidate-design-v1",
            "status": "frozen",
            "state": "frozen",
            "candidate_id": candidate_id,
            "run_id_template": "a_history_evidence_sets_v1_" + lane["name"].lower(),
            "evaluation_stage": "development_search",
            "launch_authorized": True,
            "acceptance_parameters_frozen": False,
            "task_ids": tasks["task_ids"],
            "task_manifest_sha256": _sha(
                HERE / "configs/r001.tasks.json"
            ),
            "limits": {**current["limits"], "tasks": 20},
            "ratio": current["ratio"],
            "checkpoint_selection": current["checkpoint_selection"],
            "runtime": {
                "controller": "configs/controller.json",
                "eval_policy": "configs/eval_policy.json",
                "eval_capacity": "configs/eval_capacity.json",
                "shadow_feature_config": "configs/shadow_features.json",
                **{
                    key: current["runtime"][key]
                    for key in (
                        "server_module",
                        "official_worker_module",
                        "device",
                        "dtype",
                        "bfcl_python",
                        "npu_allocator_metrics",
                    )
                },
                "generation_backend": "sglang",
                "sglang_backend_url": f"http://127.0.0.1:{lane['engine_port']}",
                "sglang_timeout_seconds": 10800,
            },
            "resolved_configs": {
                "controller": controller,
                "eval_policy": policy,
                "eval_capacity": capacity,
            },
            "source_files": _tree_hashes(runtime),
            "task_and_scorer_lineage": {
                "bfcl_root_default": str(BENCHMARK_DIR),
                "source_bindings": {
                    name: {"remote_sha256": digest}
                    for name, digest in lineage["files"].items()
                },
            },
            "search_contract": {
                "B0_history_bytes": 113246208,
                "history_token_budget": 768,
                "full_task_rollout": True,
                "fixed_denominator": "r001_mixed20",
                "history": "H0",
                "selection_protocol": "evidence_sets_v1",
                "recovery_attempts": 1,
                "fallback_unit": None,
                "recovery_reserve_tokens": 0,
                "new_cohort_planned_reference_not_automatic_retry": True,
            },
            "cpu_validation": {
                "source": "prepared_v2 frozen history_system snapshot",
                "note": "Static preview is stored beside this design",
            },
            "cumulative_time_limit_seconds": None,
            "automatic_reruns": 0,
        }
    )
    return value


def _preview_lane(package: Path, lane: Mapping[str, Any]) -> dict[str, Any]:
    lane_root = package / "lanes" / lane["name"]
    command = [
        sys.executable,
        str(package / "history_system/runner.py"),
        "preview",
        "--design",
        str(lane_root / "design.json"),
        "--runtime-root",
        str(lane_root / "runtime"),
        "--checkpoint",
        str(CHECKPOINT),
        "--output",
        str(DEFAULT_REMOTE_ROOT / "lanes" / lane["name"] / "results"),
        "--benchmark-dir",
        str(BENCHMARK_DIR),
        "--python",
        SGL_PYTHON,
        "--bfcl-python",
        BFCL_PYTHON,
        "--port-base",
        str(lane["task_port_base"]),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def prepare(package: Path, *, source: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    package = package.resolve()
    source = source.resolve()
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation package: {package}")
    source_history = source / "history_system"
    required = (
        source / "source_files.json",
        source_history / "runner.py",
        source_history / "evidence_sets.py",
        source_history / "runtime",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Frozen source input is missing: {path}")

    tasks_path = HERE / "configs/r001.tasks.json"
    training_path = OUTPUT_ROOT / "tasks.t02.expanded104.json"
    lineage_path = REPO / "outputs/history_system_search/r001/remote_lineage.json"
    tasks = _read(tasks_path)
    training = _read(training_path)
    isolation = validate_task_isolation(tasks, training)
    lineage = _read(lineage_path)
    current = _read(HERE / "configs/current_algorithm.json")
    if current["checkpoint_selection"]["path"] != str(CHECKPOINT):
        raise RuntimeError("Current checkpoint binding differs from checkpoint-1000 contract")

    package.mkdir(parents=True)
    try:
        (package / "history_system").mkdir()
        shutil.copyfile(source_history / "runner.py", package / "history_system/runner.py")
        shutil.copyfile(Path(__file__), package / "evidence_eval.py")
        shutil.copyfile(tasks_path, package / "tasks.d20.json")
        shutil.copyfile(training_path, package / "tasks.t02.expanded104.json")
        shutil.copyfile(lineage_path, package / "remote_lineage.json")

        module = _load_evidence_module(source_history)
        lane_receipts = []
        for lane in lane_specs():
            lane_root = package / "lanes" / lane["name"]
            runtime = lane_root / "runtime"
            _copy_tree(source_history / "runtime", runtime)
            controller, gp = _controller_for_lane(
                module, runtime, lane["selector"]
            )
            _save(runtime / "configs/controller.json", controller)
            design = _design(
                current=current,
                runtime=runtime,
                lane=lane,
                controller=controller,
                tasks=tasks,
                lineage=lineage,
            )
            _save(lane_root / "gp.json", gp)
            _save(lane_root / "design.json", design)
            _save(lane_root / "tasks.json", tasks)
            _save(lane_root / "lane.json", lane)
            preview = _preview_lane(package, lane)
            _save(lane_root / "preview.json", preview)
            lane_receipts.append(
                {
                    "lane": lane["name"],
                    "selector": lane["selector"],
                    "physical_device": lane["physical_device"],
                    "engine_port": lane["engine_port"],
                    "task_port_base": lane["task_port_base"],
                    "task_count": len(design["task_ids"]),
                    "automatic_reruns": design["automatic_reruns"],
                    "design_sha256": _sha(lane_root / "design.json"),
                }
            )

        source_manifest = _read(source / "source_files.json")
        sglang_files = {
            name: digest
            for name, digest in source_manifest["files"].items()
            if name.startswith("sglang/")
        }
        if not sglang_files:
            raise ValueError("Frozen prepared_v2 source manifest has no SGLang files")
        _save(
            package / "sglang_files.json",
            {
                "schema": "evidence-sets-eval-sglang-files-v1",
                "file_count": len(sglang_files),
                "files": sglang_files,
                "copy_source": str(REMOTE_SOURCE / "sglang"),
            },
        )
        provenance = {
            "schema": "evidence-sets-eval-provenance-v1",
            "prepared_at": _now(),
            "source_snapshot": "prepared_v2",
            "source_manifest_sha256": _sha(source / "source_files.json"),
            "runner_sha256": _sha(source_history / "runner.py"),
            "evidence_sets_sha256": _sha(source_history / "evidence_sets.py"),
            "d20_manifest_sha256": _sha(tasks_path),
            "t02_training_manifest_sha256": _sha(training_path),
            "remote_lineage_sha256": _sha(lineage_path),
            "task_isolation": isolation,
        }
        _save(package / "provenance.json", provenance)
        contract = {
            "schema": "evidence-sets-d20-eval-v1",
            "status": "frozen_launch_authorized",
            "launch_authorized": True,
            "remote_root": str(DEFAULT_REMOTE_ROOT),
            "checkpoint": str(CHECKPOINT),
            "benchmark_dir": str(BENCHMARK_DIR),
            "history": "H0",
            "selection_protocol": "evidence_sets_v1",
            "recovery_attempts": 1,
            "fallback_unit": None,
            "recovery_reserve_tokens": 0,
            "fixed_task_manifest": "tasks.d20.json",
            "per_lane_task_budget": 20,
            "total_task_execution_budget": 80,
            "automatic_retries": 0,
            "automatic_reruns": 0,
            "first_real_task_is_in_fixed_denominator": True,
            "extra_full_task_smoke": False,
            "actor_mem_fraction_static": 0.50,
            "lanes": lane_receipts,
            "reserved_outside_package": {
                "device_4": "spare_for_later_C1_C4",
                "device_5": "Zhuyuhan_do_not_touch",
                "device_6": "T02_owned_by_root",
                "device_7": "forbidden",
            },
        }
        _save(package / "launch_contract.json", contract)
        stage = """#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/liuyancheng/c2kv-evidence-sets-20260916/eval_v1
SOURCE=/home/liuyancheng/c2kv-evidence-sets-20260916/prepared_v2
[[ -d "$ROOT" ]] || { echo "missing uploaded eval_v1" >&2; exit 66; }
[[ ! -e "$ROOT/sglang" ]] || { echo "refusing to replace eval_v1/sglang" >&2; exit 73; }
cp -a "$SOURCE/sglang" "$ROOT/sglang"
exec /home/liuyancheng/envs/sgl/bin/python "$ROOT/evidence_eval.py" verify-package --package "$ROOT"
"""
        stage_path = package / "stage_remote.sh"
        stage_path.write_text(stage, encoding="utf-8", newline="\n")
        _save(package / "static_files.json", _static_manifest(package))
        verification = verify_package(package, require_sglang=False)
        receipt = {
            "schema": "evidence-sets-eval-prepare-v1",
            "status": "prepared",
            "package": str(package),
            "lanes": lane_receipts,
            "task_isolation": isolation,
            "verification": verification,
        }
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        return receipt
    except BaseException:
        if package.exists():
            shutil.rmtree(package)
        raise


def _bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _npu_snapshot() -> tuple[str, dict[int, list[int]]]:
    completed = subprocess.run(
        ["npu-smi", "info"], check=True, capture_output=True, text=True
    )
    return completed.stdout, device_processes(completed.stdout)


def _assert_lane_free(lane: Mapping[str, Any]) -> dict[str, Any]:
    raw, processes = _npu_snapshot()
    device = lane["physical_device"]
    if processes[device]:
        raise RuntimeError(
            f"Physical NPU {device} acquired by PIDs {processes[device]}; refusing lane"
        )
    ports = [lane["engine_port"], *lane["task_ports"]]
    occupied = [port for port in ports if not _bindable(port)]
    if occupied:
        raise RuntimeError(f"Lane {lane['name']} ports are occupied: {occupied}")
    return {
        "checked_at": _now(),
        "physical_device": device,
        "device_processes": processes[device],
        "npu_smi_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "ports": ports,
        "ports_bindable": True,
    }


def _lane_env(package: Path, lane: Mapping[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": str(lane["physical_device"]),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "OMP_NUM_THREADS": "4",
            "HCCL_SOCKET_IFNAME": "lo",
            "GLOO_SOCKET_IFNAME": "lo",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    for name in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ):
        env.pop(name, None)
    runtime = package / "lanes" / lane["name"] / "runtime"
    paths = [str(package / "sglang/python"), str(runtime / "python"), str(runtime)]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def _apply_supervisor_network_env(env: Mapping[str, str]) -> None:
    """Keep supervisor-side localhost probes off any inherited HTTP proxy."""

    for name in (
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "all_proxy",
    ):
        os.environ.pop(name, None)
    os.environ["NO_PROXY"] = env["NO_PROXY"]
    os.environ["no_proxy"] = env["no_proxy"]


def _run_model_smokes(
    package: Path, lane: Mapping[str, Any], env: Mapping[str, str], output: Path
) -> list[dict[str, Any]]:
    runtime = package / "lanes" / lane["name"] / "runtime"
    receipts = []
    for capability in lane["model_smokes"]:
        command = [
            SGL_PYTHON,
            "-m",
            "benchmarks.memory_runtime.recovery.local_selection_models",
            "smoke-" + capability,
            "--model",
            str(MODEL_PATHS[capability]),
            "--device",
            "npu:0",
            "--dtype",
            "bfloat16",
        ]
        if capability == "selector":
            command.append("--full-selector-fixture")
        completed = subprocess.run(
            command,
            cwd=runtime,
            env=dict(env),
            check=True,
            capture_output=True,
            text=True,
        )
        receipt = json.loads(completed.stdout)
        _save(output / f"model_smoke.{capability}.json", receipt)
        receipts.append(
            {
                "capability": capability,
                "status": receipt["status"],
                "artifact_identity": receipt["artifact_identity"],
            }
        )
    return receipts


def _engine_command(package: Path, lane: Mapping[str, Any]) -> list[str]:
    return [
        SGL_PYTHON,
        "-m",
        "sglang.launch_server",
        "--model-path",
        str(CHECKPOINT),
        "--served-model-name",
        "d3-c1000",
        "--model-impl",
        "sglang",
        "--device",
        "npu",
        "--attention-backend",
        "ascend",
        "--dtype",
        "bfloat16",
        "--random-seed",
        "0",
        "--enable-c2kv",
        "--c2kv-gist-type",
        "dynamic-interleave",
        "--c2kv-gist-param",
        "qkv",
        "--c2kv-query-proj",
        "base",
        "--c2kv-pool-fraction",
        "0.05",
        "--c2kv-shadow-feature-layer",
        "-2",
        "--enable-return-hidden-states",
        "--mem-fraction-static",
        "0.50",
        "--max-total-tokens",
        "65536",
        "--context-length",
        "131072",
        "--max-running-requests",
        "1",
        "--page-size",
        "128",
        "--chunked-prefill-size",
        "256",
        "--disable-radix-cache",
        "--disable-overlap-schedule",
        "--disable-cuda-graph",
        "--host",
        "127.0.0.1",
        "--port",
        str(lane["engine_port"]),
    ]


def _get_json(url: str, timeout: float = 10) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(request, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object from {url}")
    return value


def _wait_engine(process: subprocess.Popen, lane: Mapping[str, Any]) -> dict[str, Any]:
    base = f"http://127.0.0.1:{lane['engine_port']}"
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"Actor engine exited during startup with code {code}")
        try:
            with urlopen(base + "/health_generate", timeout=5) as response:
                if response.status == 200:
                    info = _get_json(base + "/model_info")
                    native = info.get("c2kv_native_packed")
                    if info.get("model_path") != str(CHECKPOINT):
                        raise RuntimeError("Engine top-level model_path differs from checkpoint")
                    if not isinstance(native, dict):
                        raise RuntimeError("Engine lacks c2kv_native_packed admission data")
                    binding = native.get("model_binding")
                    if not isinstance(binding, dict) or binding.get("model_path") != str(
                        CHECKPOINT
                    ):
                        raise RuntimeError("Engine native model_binding differs from checkpoint")
                    if not isinstance(native.get("kv_bytes_per_token"), int) or native[
                        "kv_bytes_per_token"
                    ] <= 0:
                        raise RuntimeError("Engine kv_bytes_per_token is invalid")
                    return info
        except (OSError, URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(2)
    raise TimeoutError("Actor engine did not pass health/model_info within 1200 seconds")


def _wrapper_preflight(package: Path, lane: Mapping[str, Any]) -> dict[str, Any]:
    runtime = package / "lanes" / lane["name"] / "runtime"
    sys.path[:0] = [str(runtime / "python"), str(runtime)]
    try:
        from history_memory.sglang_generator import SGLangEventNativeGenerator

        generator = SGLangEventNativeGenerator(
            f"http://127.0.0.1:{lane['engine_port']}",
            expected_model_path=str(CHECKPOINT),
            model_context=131072,
            max_new_tokens=4096,
            max_generation_calls=96,
            max_extraction_calls=1152,
            timeout_seconds=30,
            eos_token_ids=[151645],
            eos_source="evidence_eval_preflight",
        )
        generator._ensure_model_info()
        return {
            "status": "passed",
            "wrapper": "SGLangEventNativeGenerator",
            "model_binding": generator._model_binding,
            "kv_bytes_per_token": generator._kv_bytes_per_token,
        }
    finally:
        del sys.path[:2]


def _runner_command(package: Path, lane: Mapping[str, Any]) -> list[str]:
    lane_root = package / "lanes" / lane["name"]
    return [
        SGL_PYTHON,
        str(package / "history_system/runner.py"),
        "run",
        "--design",
        str(lane_root / "design.json"),
        "--runtime-root",
        str(lane_root / "runtime"),
        "--checkpoint",
        str(CHECKPOINT),
        "--output",
        str(lane_root / "results"),
        "--benchmark-dir",
        str(BENCHMARK_DIR),
        "--python",
        SGL_PYTHON,
        "--bfcl-python",
        BFCL_PYTHON,
        "--port-base",
        str(lane["task_port_base"]),
    ]


def _stop_owned_group(process: subprocess.Popen) -> int | None:
    if process.poll() is not None:
        return process.returncode
    os.killpg(process.pid, signal.SIGTERM)
    try:
        return process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        return process.wait(timeout=10)


def run_lane(package: Path, lane_name: str) -> int:
    package = package.resolve()
    contract = _read(package / "launch_contract.json")
    if not contract.get("launch_authorized"):
        raise RuntimeError("Frozen launch contract is not authorized")
    lane = _read(package / "lanes" / lane_name / "lane.json")
    if lane_name not in LANES or lane != lane_specs()[list(LANES).index(lane_name)]:
        raise RuntimeError("Lane binding differs from frozen evaluator mapping")
    lane_root = package / "lanes" / lane_name
    run_root = lane_root / "run"
    if run_root.exists() or (lane_root / "results").exists():
        raise FileExistsError(f"Refusing automatic rerun of lane {lane_name}")
    run_root.mkdir()
    status_path = run_root / "status.json"
    status: dict[str, Any] = {
        "schema": "evidence-sets-eval-lane-status-v1",
        "lane": lane_name,
        "state": "preflight",
        "started_at": _now(),
        "physical_device": lane["physical_device"],
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }
    _save(status_path, status)
    engine: subprocess.Popen | None = None
    try:
        status["package_verification"] = verify_package(package)
        for remote_path in (CHECKPOINT, BENCHMARK_DIR, *MODEL_PATHS.values()):
            required = Path(remote_path)
            if not required.exists():
                raise FileNotFoundError(f"Required production input is missing: {required}")
        status["resource_preflight"] = _assert_lane_free(lane)
        env = _lane_env(package, lane)
        _apply_supervisor_network_env(env)
        status["model_smokes"] = _run_model_smokes(package, lane, env, run_root)
        status["resource_after_model_smokes"] = _assert_lane_free(lane)
        status["state"] = "starting_actor_engine"
        _save(status_path, status)
        engine_command = _engine_command(package, lane)
        engine_log = (run_root / "engine.log").open("x", encoding="utf-8", newline="\n")
        engine = subprocess.Popen(
            engine_command,
            cwd=package,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=engine_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        status["engine"] = {
            "pid": engine.pid,
            "command": engine_command,
            "log": str(run_root / "engine.log"),
        }
        _save(status_path, status)
        model_info = _wait_engine(engine, lane)
        _save(run_root / "engine.model_info.json", model_info)
        status["wrapper_preflight"] = _wrapper_preflight(package, lane)
        status["state"] = "running_fixed_d20"
        status["engine"]["healthy_at"] = _now()
        _save(status_path, status)
        runner_command = _runner_command(package, lane)
        with (run_root / "runner.log").open("x", encoding="utf-8", newline="\n") as log:
            runner = subprocess.Popen(
                runner_command,
                cwd=package,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            status["runner"] = {
                "pid": runner.pid,
                "command": runner_command,
                "log": str(run_root / "runner.log"),
            }
            _save(status_path, status)
            code = runner.wait()
        manifest_path = lane_root / "results/stage_manifest.json"
        manifest = _read(manifest_path) if manifest_path.is_file() else None
        status["runner"]["returncode"] = code
        status["stage_manifest"] = str(manifest_path) if manifest is not None else None
        status["completed_task_cells"] = (
            manifest.get("completed_task_cells", 0) if manifest is not None else 0
        )
        status["stage_status"] = manifest.get("status") if manifest is not None else None
        status["state"] = "completed" if code == 0 else "failed_no_rerun"
        status["finished_at"] = _now()
        _save(status_path, status)
        return code
    except BaseException as error:
        status["state"] = "failed_no_rerun"
        status["error"] = f"{type(error).__name__}: {error}"
        status["finished_at"] = _now()
        _save(status_path, status)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return 2
    finally:
        if engine is not None:
            try:
                code = _stop_owned_group(engine)
                status.setdefault("engine", {})["cleanup_returncode"] = code
            except Exception as cleanup_error:
                status["engine_cleanup_error"] = (
                    f"{type(cleanup_error).__name__}: {cleanup_error}"
                )
            _save(status_path, status)


def launch(package: Path) -> dict[str, Any]:
    package = package.resolve()
    verification = verify_package(package)
    contract = _read(package / "launch_contract.json")
    if contract.get("total_task_execution_budget") != 80:
        raise RuntimeError("Launch budget differs from frozen 80-task contract")
    lane_rows = []
    for lane in lane_specs():
        lane_root = package / "lanes" / lane["name"]
        if (lane_root / "run").exists() or (lane_root / "results").exists():
            raise FileExistsError(f"Refusing automatic rerun of lane {lane['name']}")
        lane_rows.append(_assert_lane_free(lane))
    launch_root = package / "run"
    if launch_root.exists():
        raise FileExistsError("Launch receipt already exists; refusing duplicate launch")
    launch_root.mkdir()
    children = []
    for lane in lane_specs():
        log_path = launch_root / f"supervisor.{lane['name']}.log"
        log = log_path.open("x", encoding="utf-8", newline="\n")
        command = [
            SGL_PYTHON,
            str(package / "evidence_eval.py"),
            "run-lane",
            "--package",
            str(package),
            "--lane",
            lane["name"],
        ]
        process = subprocess.Popen(
            command,
            cwd=package,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log.close()
        children.append(
            {
                "lane": lane["name"],
                "pid": process.pid,
                "command": command,
                "log": str(log_path),
            }
        )
    receipt = {
        "schema": "evidence-sets-eval-launch-v1",
        "status": "dispatched",
        "launched_at": _now(),
        "verification": verification,
        "resource_preflight": lane_rows,
        "supervisors": children,
        "total_task_execution_budget": 80,
        "automatic_reruns": 0,
    }
    _save(launch_root / "launch.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def status(package: Path) -> dict[str, Any]:
    package = package.resolve()
    lanes = []
    for lane in lane_specs():
        lane_root = package / "lanes" / lane["name"]
        status_path = lane_root / "run/status.json"
        row = _read(status_path) if status_path.is_file() else {
            "lane": lane["name"],
            "state": "not_started",
            "completed_task_cells": 0,
        }
        manifest_path = lane_root / "results/stage_manifest.json"
        if manifest_path.is_file():
            manifest = _read(manifest_path)
            row["stage_status"] = manifest.get("status")
            row["completed_task_cells"] = manifest.get("completed_task_cells", 0)
            row["task_outcomes"] = manifest.get("task_outcomes", [])
        lanes.append(row)
    result = {
        "schema": "evidence-sets-eval-status-v1",
        "observed_at": _now(),
        "lanes": lanes,
        "completed_task_cells": sum(row.get("completed_task_cells", 0) for row in lanes),
        "fixed_budget": 80,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--package", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    verify = sub.add_parser("verify-package")
    verify.add_argument("--package", type=Path, required=True)
    verify.add_argument("--allow-missing-sglang", action="store_true")
    run = sub.add_parser("run-lane")
    run.add_argument("--package", type=Path, required=True)
    run.add_argument("--lane", choices=tuple(LANES), required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--package", type=Path, required=True)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--package", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args.package, source=args.source)
        return 0
    if args.command == "verify-package":
        result = verify_package(
            args.package, require_sglang=not args.allow_missing_sglang
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-lane":
        return run_lane(args.package, args.lane)
    if args.command == "launch":
        launch(args.package)
        return 0
    status(args.package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
