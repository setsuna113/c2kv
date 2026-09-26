"""Prepare and dispatch the isolated evidence_sets_v1 D20 repair cohort.

The package reuses only the 20 clean cells from eval_v1 by immutable receipt
and executes the other 15 task IDs per lane once. Launch remains gated on the
explicitly approved 34 extra task executions beyond the original 80-start
budget.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import evidence_eval as base


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
EVAL_ROOT = REPO / "outputs/history_system_search/evidence_sets_v1/eval"
DEFAULT_SOURCE_PACKAGE = EVAL_ROOT / "prepared_v1_final"
DEFAULT_OUTPUT = EVAL_ROOT / "repair/prepared_v1"
DEFAULT_REMOTE_ROOT = PurePosixPath(
    "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_repair_v1"
)
ORIGINAL_REMOTE_ROOT = PurePosixPath(
    "/home/liuyancheng/c2kv-evidence-sets-20260916/eval_v1"
)
APPROVED_EXTRA_EXECUTIONS = 34
REPAIR_TASKS = (
    "multi_turn_base_20",
    "multi_turn_base_40",
    "multi_turn_long_context_40",
    "multi_turn_base_60",
    "multi_turn_long_context_60",
    "multi_turn_base_100",
    "multi_turn_long_context_100",
    "multi_turn_base_120",
    "multi_turn_long_context_120",
    "multi_turn_base_130",
    "multi_turn_long_context_130",
    "multi_turn_base_170",
    "multi_turn_long_context_170",
    "multi_turn_base_190",
    "multi_turn_long_context_190",
)
CLEAN_REUSE_TASKS = (
    "multi_turn_base_0",
    "multi_turn_long_context_0",
    "multi_turn_long_context_20",
    "multi_turn_base_50",
    "multi_turn_long_context_50",
)
LANES = {
    "C0": {
        "selector": "candidate_rule",
        "physical_device": 0,
        "engine_port": 36600,
        "task_port_base": 36700,
        "model_smokes": ("embedding",),
    },
    "C2": {
        "selector": "reranker",
        "physical_device": 1,
        "engine_port": 36610,
        "task_port_base": 36720,
        "model_smokes": ("embedding", "reranker"),
    },
    "C3": {
        "selector": "local_llm",
        "physical_device": 2,
        "engine_port": 36620,
        "task_port_base": 36740,
        "model_smokes": ("embedding", "selector"),
    },
    "C5": {
        "selector": "parameter_source",
        "physical_device": 3,
        "engine_port": 36630,
        "task_port_base": 36760,
        "model_smokes": ("embedding",),
    },
}
OVERLAY_RELATIVE = (
    "benchmarks/memory_runtime/recovery/experiment.py",
    "benchmarks/memory_runtime/recovery/experiment_config.py",
    "benchmarks/memory_runtime/recovery/local_selection_models.py",
    "benchmarks/memory_runtime/recovery/set_models.py",
    "benchmarks/memory_runtime/recovery/set_retrieval.py",
    "benchmarks/memory_runtime/recovery/set_selectors.py",
    "benchmarks/memory_runtime/recovery/set_training.py",
    "benchmarks/memory_runtime/tests/test_evidence_sets.py",
    "benchmarks/memory_runtime/tests/test_local_selection_models.py",
    "benchmarks/memory_runtime/tests/test_set_models.py",
)


def lane_specs() -> list[dict[str, Any]]:
    rows = []
    for name, raw in LANES.items():
        row = {"name": name, **raw}
        row["model_smokes"] = list(row["model_smokes"])
        row["task_ports"] = list(
            range(row["task_port_base"], row["task_port_base"] + len(REPAIR_TASKS))
        )
        rows.append(row)
    ports = [
        port for row in rows for port in [row["engine_port"], *row["task_ports"]]
    ]
    if len(ports) != len(set(ports)):
        raise ValueError("Repair lane ports overlap")
    if {row["physical_device"] for row in rows} != {0, 1, 2, 3}:
        raise ValueError("Repair lanes must bind physical devices 0..3")
    return rows


def _configure_base() -> None:
    base.DEFAULT_REMOTE_ROOT = DEFAULT_REMOTE_ROOT
    base.LANES = LANES
    base.lane_specs = lane_specs


def _manifest(identifier: str, stage: str, task_ids: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": "a-history-system-task-manifest-v1",
        "manifest_id": identifier,
        "stage": stage,
        "task_ids": list(task_ids),
        "fixed_denominator": len(task_ids),
        "automatic_reruns": 0,
    }


def _changed_files(before: Mapping[str, str], after: Mapping[str, str]) -> dict[str, Any]:
    names = sorted(set(before) | set(after))
    return {
        name: {"source_sha256": before.get(name), "repair_sha256": after.get(name)}
        for name in names
        if before.get(name) != after.get(name)
    }


def _update_design(
    design: dict[str, Any],
    *,
    package: Path,
    runtime: Path,
    lane: Mapping[str, Any],
    task_manifest: Path,
) -> dict[str, Any]:
    value = copy.deepcopy(design)
    value["candidate_id"] = f"evidence_sets_v1_h0_{lane['name'].lower()}_repair_r1"
    value["run_id_template"] = (
        "a_history_evidence_sets_v1_repair_" + lane["name"].lower()
    )
    value["launch_authorized"] = True
    value["task_ids"] = list(REPAIR_TASKS)
    value["task_manifest_sha256"] = base._sha(task_manifest)
    value["limits"] = {**value["limits"], "tasks": len(REPAIR_TASKS)}
    value["runtime"]["sglang_backend_url"] = (
        f"http://127.0.0.1:{lane['engine_port']}"
    )
    controller_path = runtime / "configs/controller.json"
    controller = base._read(controller_path)
    gp = controller["gp_experiments"]
    gp["semantic_query_overflow_policy"] = "task_head_tail_preserve_draft_v1"
    if lane["name"] == "C3":
        gp["local_models"]["selector"]["max_input_tokens"] = 131056
    base._save(controller_path, controller)
    value["resolved_configs"]["controller"] = controller
    value["source_files"] = base._tree_hashes(runtime)
    value["search_contract"].update(
        {
            "fixed_denominator": "r001_mixed20_repair15_plus_clean5_reuse",
            "repair_task_count": len(REPAIR_TASKS),
            "clean_reuse_task_count": len(CLEAN_REUSE_TASKS),
            "clean_reuse_receipt": str(package / "clean_reuse.remote.json"),
            "semantic_query_overflow_policy": "task_head_tail_preserve_draft_v1",
            "selector_max_input_tokens": (
                131056 if lane["name"] == "C3" else None
            ),
            "original_archive_text_truncated": False,
            "lexical_routes_truncated": False,
        }
    )
    value["cpu_validation"] = {
        "source": "eval_v1 frozen snapshot plus recorded repair overlays",
        "note": "Static preview covers exactly repair15; no model or scorer call.",
    }
    return value


def prepare(
    package: Path,
    *,
    source_package: Path = DEFAULT_SOURCE_PACKAGE,
) -> dict[str, Any]:
    package = package.resolve()
    source_package = source_package.resolve()
    if package.exists():
        raise FileExistsError(f"Refusing to overwrite repair package: {package}")
    for required in (
        source_package / "history_system/runner.py",
        source_package / "sglang_files.json",
        EVAL_ROOT / "runtime_failure_audit.json",
        EVAL_ROOT / "repair/clean_reuse.remote.json",
        EVAL_ROOT / "payload_budget_audit.json",
        EVAL_ROOT / "selector_long_payload_smoke.json",
        EVAL_ROOT / "repair/real_failure_query_audit.json",
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Repair input is missing: {required}")
    _configure_base()
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
        ):
            (package / stale).unlink(missing_ok=True)
        shutil.copyfile(Path(__file__), package / "evidence_eval_repair.py")
        shutil.copyfile(HERE / "runner.py", package / "history_system/runner.py")
        shutil.copyfile(HERE / "evidence_sets.py", package / "history_system/evidence_sets.py")
        shutil.copyfile(
            EVAL_ROOT / "runtime_failure_audit.json",
            package / "runtime_failure_audit.json",
        )
        shutil.copyfile(
            EVAL_ROOT / "repair/clean_reuse.remote.json",
            package / "clean_reuse.remote.json",
        )
        shutil.copyfile(
            EVAL_ROOT / "payload_budget_audit.json",
            package / "payload_budget_audit.json",
        )
        shutil.copyfile(
            EVAL_ROOT / "selector_long_payload_smoke.json",
            package / "selector_long_payload_smoke.json",
        )
        shutil.copyfile(
            EVAL_ROOT / "repair/real_failure_query_audit.json",
            package / "real_failure_query_audit.json",
        )
        repair_manifest = package / "tasks.repair15.json"
        clean_manifest = package / "tasks.clean_reuse5.json"
        base._save(
            repair_manifest,
            _manifest("D20-repair15", "development_search_repair", REPAIR_TASKS),
        )
        base._save(
            clean_manifest,
            _manifest("D20-clean-reuse5", "completed_clean_reuse", CLEAN_REUSE_TASKS),
        )

        overlay_source = HERE / "runtime"
        lane_receipts = []
        runtime_diffs: dict[str, Any] = {}
        for lane in lane_specs():
            lane_root = package / "lanes" / lane["name"]
            runtime = lane_root / "runtime"
            source_runtime = source_package / "lanes" / lane["name"] / "runtime"
            before = base._tree_hashes(source_runtime)
            for relative in OVERLAY_RELATIVE:
                source = overlay_source / relative
                target = runtime / relative
                if not source.is_file():
                    raise FileNotFoundError(f"Repair overlay is missing: {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            old_design = base._read(lane_root / "design.json")
            design = _update_design(
                old_design,
                package=package,
                runtime=runtime,
                lane=lane,
                task_manifest=repair_manifest,
            )
            # _update_design changes controller.json, so freeze the final tree now.
            design["source_files"] = base._tree_hashes(runtime)
            base._save(lane_root / "design.json", design)
            base._save(lane_root / "gp.json", design["resolved_configs"]["controller"]["gp_experiments"])
            base._save(lane_root / "tasks.json", base._read(repair_manifest))
            base._save(lane_root / "lane.json", lane)
            preview = base._preview_lane(package, lane)
            base._save(lane_root / "preview.json", preview)
            after = base._tree_hashes(runtime)
            runtime_diffs[lane["name"]] = {
                "source_runtime": str(source_runtime),
                "repair_runtime": str(runtime),
                "changed_files": _changed_files(before, after),
            }
            lane_receipts.append(
                {
                    "lane": lane["name"],
                    "selector": lane["selector"],
                    "physical_device": lane["physical_device"],
                    "engine_port": lane["engine_port"],
                    "task_port_base": lane["task_port_base"],
                    "task_count": len(REPAIR_TASKS),
                    "task_ids": list(REPAIR_TASKS),
                    "status_path": str(
                        DEFAULT_REMOTE_ROOT / "lanes" / lane["name"] / "run/status.json"
                    ),
                    "design_sha256": base._sha(lane_root / "design.json"),
                }
            )
        base._save(
            package / "source_runtime_diff.json",
            {
                "schema": "evidence-eval-repair-source-diff-v1",
                "overlay_files": list(OVERLAY_RELATIVE),
                "runner": {
                    "source_sha256": base._sha(source_package / "history_system/runner.py"),
                    "repair_sha256": base._sha(package / "history_system/runner.py"),
                },
                "evidence_sets": {
                    "source_sha256": base._read(source_package / "provenance.json").get(
                        "evidence_sets_sha256"
                    ),
                    "repair_sha256": base._sha(package / "history_system/evidence_sets.py"),
                },
                "lanes": runtime_diffs,
            },
        )
        contract = {
            "schema": "evidence-sets-d20-repair-eval-v1",
            "status": "frozen_pending_explicit_extra_budget_approval",
            "launch_authorized": False,
            "launch_gate": {
                "flag": "--approved-extra-task-executions",
                "required_value": APPROVED_EXTRA_EXECUTIONS,
            },
            "remote_root": str(DEFAULT_REMOTE_ROOT),
            "original_eval_root": str(ORIGINAL_REMOTE_ROOT),
            "checkpoint": str(base.CHECKPOINT),
            "benchmark_dir": str(base.BENCHMARK_DIR),
            "history": "H0",
            "selection_protocol": "evidence_sets_v1",
            "semantic_query_overflow_policy": "task_head_tail_preserve_draft_v1",
            "fixed_task_manifest": "tasks.repair15.json",
            "clean_reuse_manifest": "tasks.clean_reuse5.json",
            "clean_reuse_receipt": "clean_reuse.remote.json",
            "runtime_failure_audit": "runtime_failure_audit.json",
            "payload_budget_audit": "payload_budget_audit.json",
            "selector_long_payload_smoke": "selector_long_payload_smoke.json",
            "real_failure_query_audit": "real_failure_query_audit.json",
            "per_lane_task_budget": len(REPAIR_TASKS),
            "total_task_execution_budget": len(REPAIR_TASKS) * len(LANES),
            "original_task_execution_budget": 80,
            "observed_original_task_starts": 54,
            "cumulative_task_starts_after_repair": 114,
            "extra_task_executions_requiring_approval": APPROVED_EXTRA_EXECUTIONS,
            "clean_cells_reused": len(CLEAN_REUSE_TASKS) * len(LANES),
            "failed_or_incomplete_cells_not_reused": 34,
            "automatic_retries": 0,
            "automatic_reruns": 0,
            "first_real_task_is_in_fixed_denominator": True,
            "extra_full_task_smoke": False,
            "actor_mem_fraction_static": 0.50,
            "lanes": lane_receipts,
        }
        base._save(package / "launch_contract.json", contract)
        provenance = {
            "schema": "evidence-sets-eval-repair-provenance-v1",
            "prepared_at": base._now(),
            "source_package": str(source_package),
            "source_static_manifest_sha256": base._sha(source_package / "static_files.json"),
            "runner_sha256": base._sha(package / "history_system/runner.py"),
            "evidence_sets_sha256": base._sha(package / "history_system/evidence_sets.py"),
            "repair_manifest_sha256": base._sha(repair_manifest),
            "clean_manifest_sha256": base._sha(clean_manifest),
            "clean_reuse_receipt_sha256": base._sha(package / "clean_reuse.remote.json"),
            "runtime_failure_audit_sha256": base._sha(package / "runtime_failure_audit.json"),
            "payload_budget_audit_sha256": base._sha(package / "payload_budget_audit.json"),
            "selector_long_payload_smoke_sha256": base._sha(
                package / "selector_long_payload_smoke.json"
            ),
            "real_failure_query_audit_sha256": base._sha(
                package / "real_failure_query_audit.json"
            ),
            "source_runtime_diff_sha256": base._sha(package / "source_runtime_diff.json"),
        }
        base._save(package / "provenance.json", provenance)
        stage = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT={DEFAULT_REMOTE_ROOT}
SOURCE={base.REMOTE_SOURCE}
[[ -d \"$ROOT\" ]] || {{ echo \"missing uploaded eval_repair_v1\" >&2; exit 66; }}
[[ ! -e \"$ROOT/sglang\" ]] || {{ echo \"refusing to replace eval_repair_v1/sglang\" >&2; exit 73; }}
cp -a \"$SOURCE/sglang\" \"$ROOT/sglang\"
exec {base.SGL_PYTHON} \"$ROOT/evidence_eval_repair.py\" verify-package --package \"$ROOT\"
"""
        (package / "stage_remote.sh").write_text(stage, encoding="utf-8", newline="\n")
        base._save(package / "static_files.json", base._static_manifest(package))
        verification = verify_package(package, require_sglang=False)
        result = {
            "schema": "evidence-sets-eval-repair-prepare-v1",
            "status": "prepared_pending_explicit_extra_budget_approval",
            "package": str(package),
            "remote_root": str(DEFAULT_REMOTE_ROOT),
            "lanes": lane_receipts,
            "total_task_execution_budget": 60,
            "extra_task_executions_requiring_approval": APPROVED_EXTRA_EXECUTIONS,
            "verification": verification,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    except BaseException:
        if package.exists():
            shutil.rmtree(package)
        raise


def verify_package(package: Path, *, require_sglang: bool = True) -> dict[str, Any]:
    _configure_base()
    package = package.resolve()
    result = base.verify_package(package, require_sglang=require_sglang)
    contract = base._read(package / "launch_contract.json")
    selector_smoke = base._read(package / "selector_long_payload_smoke.json")
    if (
        selector_smoke.get("schema")
        != "c2kv-selector-long-payload-functional-smoke-v1"
        or selector_smoke.get("status") != "passed"
        or selector_smoke.get("prompt_tokens") != 72338
        or selector_smoke.get("selector_max_input_tokens") != 131056
        or selector_smoke.get("legal_action") is not True
        or selector_smoke.get("post_release", {}).get("process_state")
        != "No process in device."
    ):
        raise RuntimeError("C3 real long-payload selector smoke receipt changed")
    if (
        contract.get("per_lane_task_budget") != len(REPAIR_TASKS)
        or contract.get("total_task_execution_budget") != 60
        or contract.get("extra_task_executions_requiring_approval")
        != APPROVED_EXTRA_EXECUTIONS
        or contract.get("launch_authorized") is not False
    ):
        raise RuntimeError("Repair launch/budget contract changed")
    for lane in lane_specs():
        lane_root = package / "lanes" / lane["name"]
        if base._read(lane_root / "lane.json") != lane:
            raise RuntimeError(f"Repair lane binding changed: {lane['name']}")
        design = base._read(lane_root / "design.json")
        if design.get("task_ids") != list(REPAIR_TASKS):
            raise RuntimeError(f"Repair task list changed: {lane['name']}")
        if design["limits"].get("tasks") != len(REPAIR_TASKS):
            raise RuntimeError(f"Repair lane budget changed: {lane['name']}")
        gp = design["resolved_configs"]["controller"]["gp_experiments"]
        if gp.get("semantic_query_overflow_policy") != "task_head_tail_preserve_draft_v1":
            raise RuntimeError(f"Repair query overflow policy changed: {lane['name']}")
        if lane["name"] == "C3" and gp["local_models"]["selector"].get(
            "max_input_tokens"
        ) != 131056:
            raise RuntimeError("C3 selector long-context cap changed")
    result["repair_task_count_per_lane"] = len(REPAIR_TASKS)
    result["clean_reuse_count_per_lane"] = len(CLEAN_REUSE_TASKS)
    result["launch_authorized"] = False
    return result


def _approval_receipt(package: Path) -> dict[str, Any]:
    path = package / "run/launch.json"
    if not path.is_file():
        raise RuntimeError("Repair lane lacks the explicit launch approval receipt")
    receipt = base._read(path)
    if receipt.get("approved_extra_task_executions") != APPROVED_EXTRA_EXECUTIONS:
        raise RuntimeError("Repair launch approval value differs from 34")
    return receipt


def run_lane(package: Path, lane_name: str) -> int:
    _configure_base()
    package = package.resolve()
    _approval_receipt(package)
    original_read = base._read

    def approved_read(path: Path) -> dict[str, Any]:
        value = original_read(path)
        if Path(path).resolve() == (package / "launch_contract.json").resolve():
            value = {**value, "launch_authorized": True}
        return value

    base._read = approved_read
    try:
        return base.run_lane(package, lane_name)
    finally:
        base._read = original_read


def launch(package: Path, *, approved_extra_task_executions: int) -> dict[str, Any]:
    _configure_base()
    package = package.resolve()
    if approved_extra_task_executions != APPROVED_EXTRA_EXECUTIONS:
        raise RuntimeError(
            "Repair launch requires --approved-extra-task-executions 34"
        )
    verification = verify_package(package)
    lane_rows = []
    for lane in lane_specs():
        lane_root = package / "lanes" / lane["name"]
        if (lane_root / "run").exists() or (lane_root / "results").exists():
            raise FileExistsError(f"Refusing automatic rerun of lane {lane['name']}")
        lane_rows.append(base._assert_lane_free(lane))
    launch_root = package / "run"
    if launch_root.exists():
        raise FileExistsError("Repair launch receipt exists; refusing duplicate launch")
    launch_root.mkdir()
    receipt: dict[str, Any] = {
        "schema": "evidence-sets-eval-repair-launch-v1",
        "status": "dispatching",
        "launched_at": base._now(),
        "approved_extra_task_executions": approved_extra_task_executions,
        "verification": verification,
        "resource_preflight": lane_rows,
        "supervisors": [],
        "total_task_execution_budget": 60,
        "automatic_retries": 0,
        "automatic_reruns": 0,
    }
    base._save(launch_root / "launch.json", receipt)
    for lane in lane_specs():
        log_path = launch_root / f"supervisor.{lane['name']}.log"
        log = log_path.open("x", encoding="utf-8", newline="\n")
        command = [
            base.SGL_PYTHON,
            str(package / "evidence_eval_repair.py"),
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
        receipt["supervisors"].append(
            {
                "lane": lane["name"],
                "pid": process.pid,
                "command": command,
                "log": str(log_path),
            }
        )
    receipt["status"] = "dispatched"
    base._save(launch_root / "launch.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt


def status(package: Path) -> dict[str, Any]:
    _configure_base()
    result = base.status(package)
    result["repair_task_execution_budget"] = 60
    result["clean_cells_reused"] = 20
    result["combined_d20_cells"] = 80
    result.pop("fixed_budget", None)
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
    run.add_argument("--lane", choices=tuple(LANES), required=True)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--package", type=Path, required=True)
    launch_parser.add_argument("--approved-extra-task-executions", type=int, required=True)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--package", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare(args.package, source_package=args.source_package)
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
        launch(
            args.package,
            approved_extra_task_executions=args.approved_extra_task_executions,
        )
        return 0
    if args.command == "status":
        status(args.package)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
