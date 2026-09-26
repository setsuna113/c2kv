"""Launch the fixed eight-task C2 continuation after T02 releases NPU 0."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import evidence_eval as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVAL_ROOT = REPO / "outputs/history_system_search/evidence_sets_v1/eval"
DEFAULT_OUTPUT = EVAL_ROOT / "c2_remaining_v1/prepared_v1"
DEFAULT_SOURCE_PACKAGE = EVAL_ROOT / "failed_repair/prepared_v4"
DEFAULT_REMOTE_ROOT = PurePosixPath(
    "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_c2_remaining_v1"
)
AFTER_STATUS_PATH = PurePosixPath(
    "/home/liuyancheng/c2kv-evidence-sets-20260916/"
    "prepared_v6/workers/npu0/slot.json"
)
ALLOWED_PREDECESSOR_PHASES = ("completed", "completed_no_work")
TASKS = (
    "multi_turn_base_120",
    "multi_turn_long_context_120",
    "multi_turn_base_130",
    "multi_turn_long_context_130",
    "multi_turn_base_170",
    "multi_turn_long_context_170",
    "multi_turn_base_190",
    "multi_turn_long_context_190",
)
LANES = {
    "C2": {
        "selector": "reranker",
        "physical_device": 0,
        "engine_port": 37800,
        "task_port_base": 37900,
        "model_smokes": ("embedding", "reranker"),
        "after_status_path": str(AFTER_STATUS_PATH),
    }
}
ASCEND_SET_ENVS = (
    Path("/usr/local/Ascend/ascend-toolkit/set_env.sh"),
    Path("/usr/local/Ascend/nnal/atb/set_env.sh"),
)
PATCH_RECEIPT = EVAL_ROOT / "extraction_budget_patch_receipt.json"
PATCH_FIXTURE = EVAL_ROOT / "extraction_budget_failure_steps_fixture.jsonl"
LOCAL_SGLANG = REPO.parent / "sglang-c2kv"
PATCHED_CLIENT_RELATIVE = Path("python/history_memory/sglang_generator.py")
PATCHED_SERVER_RELATIVES = (
    Path("python/sglang/srt/mem_cache/c2kv_native_packed.py"),
    Path("python/sglang/srt/entrypoints/http_server.py"),
)


def _ensure_ascend_environment() -> None:
    if os.name == "nt":
        return
    missing = [path for path in ASCEND_SET_ENVS if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Ascend environment scripts are missing: {missing}")
    commands = "; ".join(
        f"source {path} >/dev/null 2>&1" for path in ASCEND_SET_ENVS
    )
    completed = subprocess.run(
        ["/bin/bash", "-c", f"{commands}; env -0"],
        check=True,
        capture_output=True,
    )
    for entry in completed.stdout.split(b"\0"):
        if entry and b"=" in entry:
            name, value = entry.split(b"=", 1)
            os.environ[name.decode()] = value.decode()


def lane_specs() -> list[dict[str, Any]]:
    row = {"name": "C2", **LANES["C2"]}
    row["model_smokes"] = list(row["model_smokes"])
    row["task_ports"] = list(range(row["task_port_base"], row["task_port_base"] + len(TASKS)))
    ports = [row["engine_port"], *row["task_ports"]]
    if len(ports) != len(set(ports)):
        raise RuntimeError("C2 continuation ports overlap")
    return [row]


def _configure() -> None:
    base.DEFAULT_REMOTE_ROOT = DEFAULT_REMOTE_ROOT
    base.LANES = LANES
    base.lane_specs = lane_specs


def _patch_hashes(receipt: Mapping[str, Any]) -> dict[str, str]:
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("Typed budget patch receipt has no file hashes")
    required = {
        "client": "c2kv-a-runtime/experiments/history_system/runtime/"
        "python/history_memory/sglang_generator.py",
        "native": "sglang-c2kv/python/sglang/srt/mem_cache/c2kv_native_packed.py",
        "server": "sglang-c2kv/python/sglang/srt/entrypoints/http_server.py",
        "fixture": "c2kv-a-runtime/outputs/history_system_search/evidence_sets_v1/"
        "eval/extraction_budget_failure_steps_fixture.jsonl",
    }
    values: dict[str, str] = {}
    for role, name in required.items():
        digest = files.get(name)
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"Typed budget patch receipt lacks {role} hash")
        values[role] = digest
    return values


def _verify_patch_sources(receipt: Mapping[str, Any]) -> dict[str, str]:
    hashes = _patch_hashes(receipt)
    paths = {
        "client": HERE / "runtime" / PATCHED_CLIENT_RELATIVE,
        "native": LOCAL_SGLANG / PATCHED_SERVER_RELATIVES[0],
        "server": LOCAL_SGLANG / PATCHED_SERVER_RELATIVES[1],
        "fixture": PATCH_FIXTURE,
    }
    for role, path in paths.items():
        if not path.is_file() or base._sha(path) != hashes[role]:
            raise RuntimeError(f"Typed budget patch source drifted: {role}: {path}")
    return hashes


def prepare(
    package: Path,
    *,
    source_package: Path = DEFAULT_SOURCE_PACKAGE,
) -> dict[str, Any]:
    """Freeze the patched C2 continuation without uploading or launching it."""

    _configure()
    package = package.resolve()
    source_package = source_package.resolve()
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite C2 continuation: {package}")
    for required in (
        source_package / "lanes/C2/design.json",
        source_package / "lanes/C2/runtime",
        source_package / "history_system/runner.py",
        source_package / "sglang_files.json",
        PATCH_RECEIPT,
        PATCH_FIXTURE,
        EVAL_ROOT / "c2_remaining_v1/manifest.json",
        EVAL_ROOT / "failed_repair/c2_extraction_budget_failure.json",
    ):
        if not required.exists():
            raise FileNotFoundError(f"C2 continuation input is missing: {required}")
    patch_receipt = base._read(PATCH_RECEIPT)
    if patch_receipt.get("schema") != "c2kv-extraction-budget-patch-receipt-v1":
        raise RuntimeError("Unexpected typed budget patch receipt schema")
    patch_hashes = _verify_patch_sources(patch_receipt)

    shutil.copytree(
        source_package,
        package,
        ignore=shutil.ignore_patterns("run", "results", "sglang", "__pycache__", "*.pyc"),
    )
    try:
        for stale in (
            "static_files.json",
            "launch_contract.json",
            "provenance.json",
            "stage_remote.sh",
            "evidence_eval_failed_repair.py",
            "evidence_eval_repair.py",
        ):
            (package / stale).unlink(missing_ok=True)
        for stale_manifest in package.glob("tasks.*.json"):
            stale_manifest.unlink()
        for child in tuple((package / "lanes").iterdir()):
            if child.name != "C2":
                shutil.rmtree(child)

        shutil.copyfile(Path(__file__), package / "evidence_eval_c2_remaining.py")
        shutil.copyfile(HERE / "c2_remaining_dispatcher.py", package / "c2_remaining_dispatcher.py")
        shutil.copyfile(PATCH_RECEIPT, package / "payload_budget_patch_receipt.json")
        shutil.copyfile(PATCH_FIXTURE, package / "extraction_budget_failure_steps_fixture.jsonl")
        shutil.copyfile(
            EVAL_ROOT / "c2_remaining_v1/manifest.json",
            package / "c2_remaining_manifest.json",
        )
        shutil.copyfile(
            EVAL_ROOT / "failed_repair/c2_extraction_budget_failure.json",
            package / "c2_source_failure.json",
        )

        lane = lane_specs()[0]
        lane_root = package / "lanes/C2"
        runtime = lane_root / "runtime"
        client_target = runtime / PATCHED_CLIENT_RELATIVE
        shutil.copyfile(HERE / "runtime" / PATCHED_CLIENT_RELATIVE, client_target)
        overlay_root = package / "sglang_overlay"
        for relative in PATCHED_SERVER_RELATIVES:
            target = overlay_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(LOCAL_SGLANG / relative, target)

        task_manifest = {
            "schema": "a-history-system-task-manifest-v1",
            "manifest_id": "D20-C2-remaining8",
            "stage": "development_search_original_tasks_continuation",
            "task_ids": list(TASKS),
            "fixed_denominator": len(TASKS),
            "automatic_retries": 0,
            "automatic_reruns": 0,
        }
        task_manifest_path = package / "tasks.c2_remaining8.json"
        base._save(task_manifest_path, task_manifest)

        design_path = lane_root / "design.json"
        design = copy.deepcopy(base._read(design_path))
        design.update(
            {
                "candidate_id": "evidence_sets_v1_h0_c2_remaining_r1",
                "run_id_template": "a_history_evidence_sets_v1_c2_remaining",
                "task_ids": list(TASKS),
                "task_manifest_sha256": base._sha(task_manifest_path),
                "limits": {**design["limits"], "tasks": len(TASKS)},
                "automatic_reruns": 0,
            }
        )
        design["runtime"]["sglang_backend_url"] = f"http://127.0.0.1:{lane['engine_port']}"
        controller = base._read(runtime / "configs/controller.json")
        gp = controller["gp_experiments"]
        if (
            gp.get("set_selector") != "reranker"
            or gp.get("semantic_query_overflow_policy")
            != "task_head_tail_preserve_draft_v1"
            or gp.get("local_models", {}).get("reranker", {}).get("batch_size") != 1
        ):
            raise RuntimeError("Source package lacks the frozen C2 batch-one contract")
        design["resolved_configs"]["controller"] = controller
        design["search_contract"].update(
            {
                "fixed_denominator": "C2-remaining8-single-attempt-v1",
                "outer_task_isolation": "one runner subprocess per fixed task",
                "typed_budget_terminal": "SGLangExtractionBudgetExhausted",
                "typed_budget_failure_quality": "runtime_failure_in_denominator",
                "typed_budget_failure_retry": 0,
                "unhandled_runtime_failure": "stop_lane",
                "preserved_source_failure_not_rerun": "multi_turn_long_context_100",
            }
        )
        design["source_files"] = base._tree_hashes(runtime)
        base._save(design_path, design)
        base._save(lane_root / "tasks.json", task_manifest)
        base._save(lane_root / "lane.json", lane)
        base._save(lane_root / "preview.json", base._preview_lane(package, lane))

        sglang_manifest_path = package / "sglang_files.json"
        sglang_manifest = base._read(sglang_manifest_path)
        sglang_files = sglang_manifest.get("files")
        if not isinstance(sglang_files, dict):
            raise RuntimeError("Source SGLang manifest changed")
        sglang_files[
            "sglang/python/sglang/srt/mem_cache/c2kv_native_packed.py"
        ] = patch_hashes["native"]
        sglang_files[
            "sglang/python/sglang/srt/entrypoints/http_server.py"
        ] = patch_hashes["server"]
        sglang_manifest["file_count"] = len(sglang_files)
        base._save(sglang_manifest_path, sglang_manifest)

        contract = {
            "schema": "evidence-sets-c2-remaining-eval-v1",
            "status": "frozen_waiting_for_t02_npu0_terminal",
            "launch_authorized": False,
            "launch_gate": {"flag": "--approved-task-executions", "required_value": 8},
            "remote_root": str(DEFAULT_REMOTE_ROOT),
            "checkpoint": str(base.CHECKPOINT),
            "benchmark_dir": str(base.BENCHMARK_DIR),
            "lane": "C2",
            "physical_device": 0,
            "engine_port": lane["engine_port"],
            "task_port_base": lane["task_port_base"],
            "after_status_path": str(AFTER_STATUS_PATH),
            "allowed_predecessor_phases": list(ALLOWED_PREDECESSOR_PHASES),
            "requires_fresh_device_after_predecessor": True,
            "fixed_task_manifest": task_manifest_path.name,
            "task_execution_budget": len(TASKS),
            "task_ids": list(TASKS),
            "preserved_runtime_failure_not_rerun": "multi_turn_long_context_100",
            "typed_budget_error": "SGLangExtractionBudgetExhausted",
            "typed_budget_message_prefix": "C2KV_EXTRACTION_BUDGET_EXHAUSTED:",
            "typed_budget_failure_policy": "runtime_failure_in_denominator_then_continue",
            "other_runtime_failure_policy": "stop_lane",
            "max_extraction_calls_per_task": 1152,
            "max_generation_calls_per_task": 96,
            "semantic_query_overflow_policy": "task_head_tail_preserve_draft_v1",
            "reranker_batch_size": 1,
            "automatic_retries": 0,
            "automatic_reruns": 0,
            "first_real_task_is_in_fixed_denominator": True,
            "extra_full_task_smoke": False,
        }
        base._save(package / "launch_contract.json", contract)
        provenance = {
            "schema": "evidence-sets-c2-remaining-provenance-v1",
            "prepared_at": base._now(),
            "source_package": str(source_package),
            "source_static_manifest_sha256": base._sha(source_package / "static_files.json"),
            "design_sha256": base._sha(design_path),
            "task_manifest_sha256": base._sha(task_manifest_path),
            "runtime_tree_sha256": base._sha(client_target),
            "payload_budget_patch_receipt_sha256": base._sha(
                package / "payload_budget_patch_receipt.json"
            ),
            "classifier_fixture_sha256": base._sha(
                package / "extraction_budget_failure_steps_fixture.jsonl"
            ),
            "source_failure_receipt_sha256": base._sha(package / "c2_source_failure.json"),
            "wrapper_sha256": base._sha(package / "evidence_eval_c2_remaining.py"),
            "dispatcher_sha256": base._sha(package / "c2_remaining_dispatcher.py"),
            "patch_hashes": patch_hashes,
        }
        base._save(package / "provenance.json", provenance)

        stage = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={DEFAULT_REMOTE_ROOT}
SOURCE={base.REMOTE_SOURCE}
[[ -d \"$ROOT\" ]] || {{ echo \"missing uploaded eval_c2_remaining_v1\" >&2; exit 66; }}
[[ ! -e \"$ROOT/sglang\" ]] || {{ echo \"refusing to replace eval_c2_remaining_v1/sglang\" >&2; exit 73; }}
cp -a \"$SOURCE/sglang\" \"$ROOT/sglang\"
cp -a \"$ROOT/sglang_overlay/python/sglang/srt/mem_cache/c2kv_native_packed.py\" \"$ROOT/sglang/python/sglang/srt/mem_cache/c2kv_native_packed.py\"
cp -a \"$ROOT/sglang_overlay/python/sglang/srt/entrypoints/http_server.py\" \"$ROOT/sglang/python/sglang/srt/entrypoints/http_server.py\"
exec {base.SGL_PYTHON} \"$ROOT/evidence_eval_c2_remaining.py\" verify-package --package \"$ROOT\"
"""
        (package / "stage_remote.sh").write_text(stage, encoding="utf-8", newline="\n")
        base._save(package / "static_files.json", base._static_manifest(package))
        verification = verify_package(package, require_sglang=False)
        result = {
            "schema": "evidence-sets-c2-remaining-prepare-v1",
            "status": "frozen_waiting_for_t02_npu0_terminal",
            "package": str(package),
            "remote_root": str(DEFAULT_REMOTE_ROOT),
            "task_ids": list(TASKS),
            "task_execution_budget": len(TASKS),
            "physical_device": 0,
            "after_status_path": str(AFTER_STATUS_PATH),
            "verification": verification,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except BaseException:
        if package.exists():
            shutil.rmtree(package)
        raise


def _runner_command(package: Path, lane: Mapping[str, Any]) -> list[str]:
    lane_root = package / "lanes" / lane["name"]
    return [
        base.SGL_PYTHON,
        str(package / "c2_remaining_dispatcher.py"),
        "run",
        "--design",
        str(lane_root / "design.json"),
        "--runtime-root",
        str(lane_root / "runtime"),
        "--checkpoint",
        str(base.CHECKPOINT),
        "--output",
        str(lane_root / "results"),
        "--benchmark-dir",
        str(base.BENCHMARK_DIR),
        "--python",
        base.SGL_PYTHON,
        "--bfcl-python",
        base.BFCL_PYTHON,
        "--port-base",
        str(lane["task_port_base"]),
    ]


def verify_package(package: Path, *, require_sglang: bool = True) -> dict[str, Any]:
    _configure()
    package = package.resolve()
    result = base.verify_package(package, require_sglang=require_sglang)
    contract = base._read(package / "launch_contract.json")
    lane = base._read(package / "lanes/C2/lane.json")
    design = base._read(package / "lanes/C2/design.json")
    if contract.get("schema") != "evidence-sets-c2-remaining-eval-v1":
        raise RuntimeError("C2 continuation launch schema changed")
    if (
        contract.get("task_execution_budget") != len(TASKS)
        or contract.get("automatic_retries") != 0
        or contract.get("automatic_reruns") != 0
        or contract.get("launch_authorized") is not False
        or contract.get("typed_budget_error") != "SGLangExtractionBudgetExhausted"
    ):
        raise RuntimeError("C2 continuation launch/budget contract changed")
    if lane != lane_specs()[0]:
        raise RuntimeError("C2 continuation lane mapping changed")
    if (
        design.get("task_ids") != list(TASKS)
        or design.get("limits", {}).get("tasks") != len(TASKS)
        or design.get("limits", {}).get("extraction_calls_per_task") != 1152
        or design.get("automatic_reruns") != 0
    ):
        raise RuntimeError("C2 continuation fixed task contract changed")
    gp = design["resolved_configs"]["controller"]["gp_experiments"]
    if (
        gp.get("set_selector") != "reranker"
        or gp.get("semantic_query_overflow_policy")
        != "task_head_tail_preserve_draft_v1"
        or gp.get("local_models", {}).get("reranker", {}).get("batch_size") != 1
    ):
        raise RuntimeError("C2 continuation selector contract changed")
    if not (package / "payload_budget_patch_receipt.json").is_file():
        raise FileNotFoundError("Frozen typed budget patch receipt is missing")
    patch_receipt = base._read(package / "payload_budget_patch_receipt.json")
    patch_hashes = _patch_hashes(patch_receipt)
    patched_paths = {
        "client": package / "lanes/C2/runtime" / PATCHED_CLIENT_RELATIVE,
        "native": package / "sglang_overlay" / PATCHED_SERVER_RELATIVES[0],
        "server": package / "sglang_overlay" / PATCHED_SERVER_RELATIVES[1],
        "fixture": package / "extraction_budget_failure_steps_fixture.jsonl",
    }
    for role, path in patched_paths.items():
        if not path.is_file() or base._sha(path) != patch_hashes[role]:
            raise RuntimeError(f"Frozen typed budget patch drifted: {role}")
    sglang_files = base._read(package / "sglang_files.json").get("files", {})
    if (
        sglang_files.get(
            "sglang/python/sglang/srt/mem_cache/c2kv_native_packed.py"
        )
        != patch_hashes["native"]
        or sglang_files.get(
            "sglang/python/sglang/srt/entrypoints/http_server.py"
        )
        != patch_hashes["server"]
    ):
        raise RuntimeError("Frozen SGLang manifest does not bind the typed budget patch")
    tasks = base._read(package / "tasks.c2_remaining8.json")
    if (
        tasks.get("task_ids") != list(TASKS)
        or tasks.get("fixed_denominator") != len(TASKS)
        or "multi_turn_long_context_100" in tasks.get("task_ids", [])
    ):
        raise RuntimeError("C2 continuation task manifest changed")
    if (
        contract.get("after_status_path") != str(AFTER_STATUS_PATH)
        or contract.get("allowed_predecessor_phases")
        != list(ALLOWED_PREDECESSOR_PHASES)
        or contract.get("requires_fresh_device_after_predecessor") is not True
    ):
        raise RuntimeError("C2 continuation resource dependency changed")
    result.update(
        {
            "lane": "C2",
            "task_ids": list(TASKS),
            "task_execution_budget": len(TASKS),
            "physical_device": 0,
            "after_status_path": str(AFTER_STATUS_PATH),
            "allowed_predecessor_phases": list(ALLOWED_PREDECESSOR_PHASES),
            "automatic_retries": 0,
            "automatic_reruns": 0,
        }
    )
    return result


def _terminal_predecessor_receipt(lane: Mapping[str, Any]) -> dict[str, Any] | None:
    path = Path(lane["after_status_path"])
    if not path.is_file():
        return None
    receipt = base._read(path)
    phase = receipt.get("phase")
    if phase not in ALLOWED_PREDECESSOR_PHASES:
        return None
    if receipt.get("physical_device") != lane["physical_device"]:
        raise RuntimeError("T02 terminal slot device differs from C2 continuation")
    if not isinstance(receipt.get("finished_at_epoch"), (int, float)):
        raise RuntimeError("T02 terminal slot lacks finished_at_epoch")
    return {
        "kind": "t02_worker_slot",
        "path": str(path),
        "sha256": base._sha(path),
        "phase": phase,
        "worker_id": receipt.get("worker_id"),
        "physical_device": receipt["physical_device"],
        "finished_at_epoch": receipt["finished_at_epoch"],
    }


def run_lane(package: Path) -> int:
    _ensure_ascend_environment()
    _configure()
    package = package.resolve()
    approval = base._read(package / "run/launch.json")
    if approval.get("approved_task_executions") != len(TASKS):
        raise RuntimeError("C2 continuation launch receipt does not bind eight tasks")
    original_read = base._read
    original_runner_command = base._runner_command

    def approved_read(path: Path) -> dict[str, Any]:
        value = original_read(path)
        if Path(path).resolve() == (package / "launch_contract.json").resolve():
            value = {**value, "launch_authorized": True}
        return value

    base._read = approved_read
    base._runner_command = _runner_command
    try:
        return base.run_lane(package, "C2")
    finally:
        base._read = original_read
        base._runner_command = original_runner_command


def wait_lane(package: Path) -> int:
    _ensure_ascend_environment()
    _configure()
    package = package.resolve()
    lane = lane_specs()[0]
    relay_root = package / "relay"
    relay_root.mkdir(exist_ok=True)
    relay_path = relay_root / "C2.json"
    if relay_path.exists():
        raise FileExistsError("C2 continuation relay already exists; no automatic restart")
    receipt: dict[str, Any] = {
        "schema": "evidence-sets-c2-remaining-relay-v1",
        "state": "waiting_for_t02_npu0_terminal",
        "supervisor_pid": os.getpid(),
        "physical_device": 0,
        "after_status_path": str(AFTER_STATUS_PATH),
        "allowed_predecessor_phases": list(ALLOWED_PREDECESSOR_PHASES),
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }
    base._save(relay_path, receipt)
    while True:
        dependency = _terminal_predecessor_receipt(lane)
        if dependency is not None:
            break
        time.sleep(20)
    receipt.update(state="waiting_for_fresh_npu0", predecessor_receipt=dependency)
    base._save(relay_path, receipt)
    while True:
        try:
            resource = base._assert_lane_free(lane)
            break
        except RuntimeError as error:
            receipt["last_resource_wait_error"] = str(error)
            base._save(relay_path, receipt)
            time.sleep(20)
    receipt.update(state="starting_c2_remaining", resource_preflight=resource)
    base._save(relay_path, receipt)
    code = run_lane(package)
    receipt.update(
        state="completed" if code == 0 else "failed_no_rerun",
        lane_returncode=code,
        finished_at=base._now(),
    )
    base._save(relay_path, receipt)
    return code


def launch(package: Path, *, approved_task_executions: int) -> dict[str, Any]:
    _ensure_ascend_environment()
    _configure()
    package = package.resolve()
    if approved_task_executions != len(TASKS):
        raise RuntimeError("C2 continuation launch requires exactly eight executions")
    verification = verify_package(package)
    lane_root = package / "lanes/C2"
    if (
        (lane_root / "run").exists()
        or (lane_root / "results").exists()
        or (package / "relay/C2.json").exists()
    ):
        raise FileExistsError("C2 continuation already has run evidence; no rerun")
    launch_root = package / "run"
    if launch_root.exists():
        raise FileExistsError("C2 continuation launch receipt already exists")
    launch_root.mkdir()
    log_path = launch_root / "supervisor.C2.log"
    command = [
        base.SGL_PYTHON,
        str(package / "evidence_eval_c2_remaining.py"),
        "wait-lane",
        "--package",
        str(package),
    ]
    receipt: dict[str, Any] = {
        "schema": "evidence-sets-c2-remaining-launch-v1",
        "status": "dispatching_wait_relay",
        "launched_at": base._now(),
        "approved_task_executions": len(TASKS),
        "verification": verification,
        "supervisor": {"pid": None, "command": command, "log": str(log_path)},
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }
    base._save(launch_root / "launch.json", receipt)
    log = log_path.open("x", encoding="utf-8", newline="\n")
    process = subprocess.Popen(
        command,
        cwd=package,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    receipt["supervisor"]["pid"] = process.pid
    receipt["status"] = "waiting_relay_dispatched"
    base._save(launch_root / "launch.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def status(package: Path) -> dict[str, Any]:
    _configure()
    result = base.status(package)
    relay = package / "relay/C2.json"
    result["relay"] = base._read(relay) if relay.is_file() else {
        "state": "not_dispatched"
    }
    result["task_execution_budget"] = len(TASKS)
    result["automatic_retries"] = 0
    result["automatic_reruns"] = 0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--package", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument(
        "--source-package", type=Path, default=DEFAULT_SOURCE_PACKAGE
    )
    verify = sub.add_parser("verify-package")
    verify.add_argument("--package", type=Path, required=True)
    verify.add_argument("--allow-missing-sglang", action="store_true")
    run = sub.add_parser("run-lane")
    run.add_argument("--package", type=Path, required=True)
    wait = sub.add_parser("wait-lane")
    wait.add_argument("--package", type=Path, required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--package", type=Path, required=True)
    launch_parser.add_argument("--approved-task-executions", type=int, required=True)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--package", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args.package, source_package=args.source_package)
        return 0
    if args.command == "verify-package":
        result = verify_package(args.package, require_sglang=not args.allow_missing_sglang)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-lane":
        return run_lane(args.package)
    if args.command == "wait-lane":
        return wait_lane(args.package)
    if args.command == "launch":
        launch(args.package, approved_task_executions=args.approved_task_executions)
        return 0
    if args.command == "status":
        status(args.package)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
