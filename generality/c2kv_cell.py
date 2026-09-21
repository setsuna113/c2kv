"""C2KV-backend cell driver for the generality experiment (runs on ascend03).

One cell = backend=c2kv x working point x condition x benchmark. The driver
launches the frozen T02-era controller server (our forked controller_runtime)
per task against the long-lived NPU engine, runs the official benchmark
worker, and writes per-task receipts with resume support.

Conditions:
  tracer_history          -> method=proposed, detector=t02_risk, cell threshold,
                             eval_policy K + recovery caps B, ratio 4
  recovery_off_same_initial -> method=c2kv_only, eval_policy K, ratio 4
  compression_full_budget  -> method=c2kv_only, eval_policy B (single budget),
                             ratio 4; only labelled competitive after the
                             available_history audit passes
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

try:
    from .bfcl_results import (
        bfcl_row_is_valid,
        collect_bfcl_results,
        completion_receipt,
        ordered_unique,
    )
except ImportError:
    from bfcl_results import (
        bfcl_row_is_valid,
        collect_bfcl_results,
        completion_receipt,
        ordered_unique,
    )

try:
    from .completion_contract import write_cell_status
except ImportError:  # Direct file launch on ascend03.
    from completion_contract import write_cell_status

try:
    from .process_lifecycle import defer_interrupts, interruptible, stop_owned_group, wait_owned_worker
    from .tau2_harness import completed_tau2_result, completed_tau2_task, run_tau2_task
    from .toolsandbox_harness import (
        completed_task as completed_toolsandbox_task,
        official_result as toolsandbox_official_result,
        score_summary as toolsandbox_score_summary,
        worker_command as toolsandbox_worker_command,
    )
    from .upstream_liveness import UpstreamLiveness, UpstreamUnavailable
except ImportError:  # Direct file launch on ascend03.
    from process_lifecycle import defer_interrupts, interruptible, stop_owned_group, wait_owned_worker
    from tau2_harness import completed_tau2_result, completed_tau2_task, run_tau2_task
    from toolsandbox_harness import (
        completed_task as completed_toolsandbox_task,
        official_result as toolsandbox_official_result,
        score_summary as toolsandbox_score_summary,
        worker_command as toolsandbox_worker_command,
    )
    from upstream_liveness import UpstreamLiveness, UpstreamUnavailable

try:
    from .candidate_cell import (
        VARIANTS as CANDIDATE_VARIANTS,
        GOAL_VARIANTS,
        STATIC_VARIANTS,
        STATIC_VERSION,
        VERIFIED_STATIC_VARIANTS,
        VERIFIED_VARIANTS,
        candidate_cell_from_source,
        controller_with_binding as candidate_controller_with_binding,
        static_contract,
    )
except ImportError:  # Direct file launch on ascend03.
    from candidate_cell import (
        VARIANTS as CANDIDATE_VARIANTS,
        GOAL_VARIANTS,
        STATIC_VARIANTS,
        STATIC_VERSION,
        VERIFIED_STATIC_VARIANTS,
        VERIFIED_VARIANTS,
        candidate_cell_from_source,
        controller_with_binding as candidate_controller_with_binding,
        static_contract,
    )

GENERATION_ROOT = Path("/home/liuyancheng/c2kv-generality-20260918")
C1_DELIVERY = GENERATION_ROOT / "src" / "c1_delivery"
RUNTIME = Path(__file__).resolve().parents[1] / "controller_runtime"
sys.path.insert(0, str(C1_DELIVERY))

import current                     # noqa: E402  (c1_delivery parent module)
import evidence_sets              # noqa: E402
from c1_artifact_binding import bind_risk_artifact  # noqa: E402

RISK_ARTIFACT = C1_DELIVERY / "artifacts" / "c1_risk.t02_v1.json"

WORKER_MODULES = {
    "bfcl": "benchmarks.memory_runtime.event_native_bfcl",
    "acon_appworld": "benchmarks.memory_runtime.event_native_appworld",
}


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_static_ready_manifest(cell: dict, ready_path: Path) -> None:
    """Bind a static-view attempt to the frozen controller loaded by the server."""
    variant = cell.get("candidate_algorithm")
    if variant not in STATIC_VARIANTS:
        return
    controller_path = Path(cell["controller_path"]).resolve()
    try:
        controller_bytes = controller_path.read_bytes()
        controller = json.loads(controller_bytes)
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("static candidate ready manifest or controller is unreadable") from error
    loaded = ready.get("s0_controller_contract")
    candidate = ready.get("candidate_algorithm")
    route = ready.get("route_contract")
    contract = static_contract(variant)
    if variant in VERIFIED_STATIC_VARIANTS:
        if (cell.get("proof_registry_version") != contract["proof_registry_version"]
                or not isinstance(controller.get("candidate_algorithm"), dict)
                or controller["candidate_algorithm"].get("proof_registry_version")
                != contract["proof_registry_version"]):
            raise RuntimeError("static candidate ready manifest differs from frozen controller")
    if (not isinstance(loaded, dict) or not isinstance(candidate, dict)
            or not isinstance(route, dict)
            or ready.get("status") != "ready"
            or loaded.get("source") != str(controller_path)
            or loaded.get("sha256") != hashlib.sha256(controller_bytes).hexdigest()
            or loaded.get("config") != controller
            or candidate.get("variant") != variant
            or candidate.get("stable_call_ids") is not True
            or any(candidate.get(key) != value for key, value in contract.items())
            or route.get("baseline_identity") != STATIC_VERSION + ":" + variant
            or route.get("recovery_enabled") is not True
            or route.get("max_generations_per_decision") != 2):
        raise RuntimeError("static candidate ready manifest differs from frozen controller")


def _controller_with_binding(cell: dict) -> tuple[dict, dict | None]:
    """T02 risk selector with the cell's calibrated threshold (fixed weights)."""
    if cell["condition"] == "candidate_algorithm":
        return candidate_controller_with_binding(
            cell,
            base_controller=evidence_sets._base_controller(),
            selected=current.load_config(),
            risk_artifact_path=RISK_ARTIFACT,
            bind_risk_artifact=bind_risk_artifact,
        )
    if cell["condition"] == "tracer_history":
        return _risk_controller_with_binding(cell, float(cell["threshold"]))
    controller = evidence_sets._base_controller()
    controller.pop("post_draft_recovery", None)
    controller.pop("gp_experiments", None)
    return controller, None


def _risk_controller_with_binding(cell: dict, selector_threshold: float) -> tuple[dict, dict]:
    """Construct the frozen risk head; calibration reads its score only."""
    config, _ = evidence_sets.build_config(
            history="H0",
            selector="risk",
            selector_artifact=RISK_ARTIFACT,
            selector_threshold=selector_threshold,
            embedding_model=cell["embedding_model"],
            embedding_device="cpu",
            semantic_query_overflow_policy="task_head_tail_preserve_draft_v1",
    )
    config["local_models"]["embedding"]["dtype"] = "bfloat16"
    config["selector_artifact"], binding = bind_risk_artifact(
        config["selector_artifact"], Path(cell["checkpoint"])
    )
    controller = current._configure_controller(evidence_sets._base_controller(), config)
    # The legacy detector does not enter this frozen risk-selector path.
    gp = controller["gp_experiments"]
    assert gp["set_selector"] == "risk", gp["set_selector"]
    assert gp["selection_protocol"] == "evidence_sets_v1", gp["selection_protocol"]
    assert gp["selector_artifact"]["model_kind"] == "c1_risk_logistic"
    assert (hashlib.sha256(RISK_ARTIFACT.read_bytes()).hexdigest()
            == "18a11f73aa1f7d4b0add86eed66ae9e5e129ea4bdfbe0dfad23faf4f7d2fb4ab")
    assert gp["recovery_reserve_tokens"] == 0, gp["recovery_reserve_tokens"]
    return controller, binding


def calibration_controller_config(cell: dict) -> tuple[dict, dict]:
    """Build a risk-score-only controller without reading a calibrated threshold.

    The required selector threshold of 1.0 is inert: calibration requests set
    recovery_disabled, and the runner calls only the risk predictor.
    """
    if cell["backend"] != "c2kv" or cell["benchmark"] != "bfcl":
        raise ValueError("C2KV calibration requires a C2KV BFCL cell")
    if cell["condition"] != "tracer_history":
        raise ValueError("C2KV calibration uses the tracer history view")
    return _risk_controller_with_binding(cell, 1.0)


def build_controller_config(cell: dict) -> dict:
    controller, binding = _controller_with_binding(cell)
    if binding is not None:
        _write(Path(cell["cell_dir"]) / "risk_artifact_binding.json", binding)
    return controller


def build_eval_policy(cell: dict, budgets: dict) -> dict:
    wp = budgets["working_points"][cell["working_point"]]
    policy = {
        "history_budget_bytes": wp["history_allowance_bytes"],
        "workspace_budget_bytes": wp["history_allowance_bytes"],
        "lease_decisions": 0,
        "max_retrieved_events": 2,
    }
    if cell["condition"] == "tracer_history":
        policy["recovery_history_bytes"] = wp["common_cap_bytes"]
        policy["recovery_workspace_bytes"] = wp["common_cap_bytes"]
    elif cell["condition"] in ("compression_full_budget", "candidate_algorithm"):
        # competitive control: the bare compressor may use the whole common cap
        policy["history_budget_bytes"] = wp["common_cap_bytes"]
        policy["workspace_budget_bytes"] = wp["common_cap_bytes"]
    base = {
        "schema": "a-event-native-eval-policy-v1",
        "policy_id": f"generality-{cell['cell_id']}",
        "policy": policy,
    }
    if "history_budget_tokens" not in cell:
        return base
    override = cell.get("native_history_budget")
    if override is None:
        override = native_history_budget_profile(cell, budgets, base)
    elif override.get("requested_tokens") != cell["history_budget_tokens"]:
        raise ValueError("native history budget profile differs from the cell")
    return copy.deepcopy(override["eval_policy"])


def _json_sha256(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_file_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def native_history_budget_profile(cell: dict, budgets: dict, base: dict | None = None) -> dict:
    """Bind an explicit token cap to this checkpoint's verified KV geometry."""
    tokens = cell.get("history_budget_tokens")
    if type(tokens) is not int or tokens <= 0:
        raise ValueError("history_budget_tokens must be a positive integer")
    if (cell.get("condition") != "candidate_algorithm"
            or cell.get("benchmark") != "bfcl" or cell.get("ratio") != 8):
        raise ValueError("native history budget requires a ratio-8 BFCL candidate cell")
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from controller_runtime.benchmarks.memory_runtime.event_native import (
        inspect_checkpoint, validate_inference_byte_profile,
    )

    checkpoint = Path(cell["checkpoint"])
    checkpoint_config_sha256 = hashlib.sha256(
        (checkpoint / "config.json").read_bytes()).hexdigest()
    checkpoint_profile = inspect_checkpoint(checkpoint)
    bytes_per_token = validate_inference_byte_profile(checkpoint_profile, "bfloat16")
    budget_bytes = tokens * bytes_per_token
    if base is None:
        base_cell = dict(cell)
        base_cell.pop("history_budget_tokens")
        base_cell.pop("native_history_budget", None)
        base = build_eval_policy(base_cell, budgets)
    legacy_id = f"{cell['candidate_source_cell_id']}__candidate_{cell['candidate_algorithm']}"
    base = copy.deepcopy(base)
    base["policy_id"] = f"generality-{legacy_id}"
    base["policy"]["history_budget_bytes"] = budgets["working_points"][
        cell["working_point"]]["common_cap_bytes"]
    base["policy"]["workspace_budget_bytes"] = base["policy"]["history_budget_bytes"]
    override = copy.deepcopy(base)
    override["policy_id"] = f"generality-{cell['cell_id']}"
    override["policy"]["history_budget_bytes"] = budget_bytes
    override["policy"]["workspace_budget_bytes"] = budget_bytes
    return {
        "schema": "c2kv-native-history-budget-override-v1",
        "requested_tokens": tokens,
        "kv_bytes_per_token": bytes_per_token,
        "inference_dtype": "bfloat16",
        "history_budget_bytes": budget_bytes,
        "workspace_budget_bytes": budget_bytes,
        "checkpoint_config_sha256": checkpoint_config_sha256,
        "base_policy_id": base["policy_id"],
        "base_history_budget_bytes": base["policy"]["history_budget_bytes"],
        "base_workspace_budget_bytes": base["policy"]["workspace_budget_bytes"],
        "base_eval_policy_path": None,
        "base_eval_policy_source": "virtual_working_point_candidate_policy",
        "base_eval_policy_sha256": hashlib.sha256(_json_file_bytes(base)).hexdigest(),
        "override_eval_policy_path": str(Path(cell["cell_dir"]) / "eval_policy.json"),
        "override_eval_policy_sha256": hashlib.sha256(_json_file_bytes(override)).hexdigest(),
        "override_policy_sha256": _json_sha256(override),
        "eval_policy": override,
    }


def server_command(cell: dict, task_ids: list[str], out: Path, port: int,
                   *, max_decisions: int | None = None,
                   max_generation_calls: int | None = None,
                   max_extraction_calls: int | None = None,
                   max_wall_seconds: int | None = None) -> list[str]:
    design = current.load_config()
    caps = cell["caps"]
    command = [
        cell["python_sgl"], "-m", "benchmarks.memory_runtime.event_native_server",
        "--checkpoint", cell["checkpoint"],
        "--out", str(out / "server"),
        "--run-id", f"{cell['cell_id']}__b{abs(hash(tuple(task_ids))) % 10**8}",
        "--model-name", cell["model_name"],
        "--benchmark", cell["benchmark"],
        "--source-profile",
        "native-v1" if cell["benchmark"] == "bfcl" else "openai-single-task-v1",
        "--view-mode", design["route"],
        "--compression-policy", design["compression_policy"],
        "--history-view-protocol", design["history_view_protocol"],
        "--ratio", str(cell["ratio"]),
        "--max-new-tokens", str(caps["max_completion_tokens"]),
        "--decode-strategy", design["decode_strategy"],
        "--prefill-chunk-size", str(design["prefill_chunk_size"]),
        "--task-ids", ",".join(task_ids),
        "--max-decisions", str(max_decisions or caps["generation_attempts_per_task"] * len(task_ids)),
        "--max-generation-calls", str(max_generation_calls or caps["generation_attempts_per_task"]),
        "--max-extraction-calls", str(max_extraction_calls or caps["extraction_calls_per_task"]),
        "--eval-policy", str(cell["eval_policy_path"]),
        "--eval-capacity", str(C1_DELIVERY / "runtime/configs/eval_capacity.json"),
        "--s0-config", str(cell["controller_path"]),
        "--max-wall-seconds", str(max_wall_seconds or caps["task_timeout"]),
        "--device", "cpu",
        "--dtype", "bfloat16",
        "--generation-backend", "sglang",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--torch-threads", "4",
        "--sglang-backend-url", cell["sglang_backend_url"],
        "--sglang-timeout-seconds", str(caps["task_timeout"]),
        "--no-raw-snapshot",
    ]
    if (cell["condition"] == "tracer_history"
            or cell.get("candidate_algorithm") in GOAL_VARIANTS + VERIFIED_VARIANTS + STATIC_VARIANTS):
        # The frozen C1 risk head needs its exact prefill hidden-state contract.
        command.extend(["--shadow-feature-config",
                        str(RUNTIME / "configs" / "shadow_features.json")])
    return command


def bfcl_worker_command(cell: dict, task_ids: list[str], out: Path, port: int) -> list[str]:
    # the official worker reads the frozen allowlist from the server manifest
    del task_ids
    return [
        cell["python_bench"], "-m", "benchmarks.memory_runtime.event_native_bfcl",
        "--server-manifest", str(out / "server" / "ready.json"),
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--benchmark-dir", cell["benchmark_dir"],
        "--out", str(out / "bfcl_worker"),
        "--max-wall-seconds", str(caps_of(cell)["task_timeout"] * 200),
    ]


def appworld_worker_command(cell: dict, task_id: str, out: Path, port: int) -> list[str]:
    return [
        cell["python_sgl"], "-m", "benchmarks.memory_runtime.event_native_appworld",
        "--server-manifest", str(out / "server" / "ready.json"),
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--acon-dir", cell["acon_dir"],
        "--appworld-root", cell["appworld_root"],
        "--bench-python", cell["python_appworld"],
        "--out", str(out / "appworld_worker"),
        "--task-id", task_id,
        "--max-iter", "50",
        "--max-wall-seconds", str(caps_of(cell)["task_timeout"]),
    ]


def caps_of(cell: dict) -> dict:
    return cell["caps"]


def worker_command(cell: dict, task_ids: list[str], out: Path, port: int) -> list[str]:
    if cell["benchmark"] == "bfcl":
        return bfcl_worker_command(cell, task_ids, out, port)
    return appworld_worker_command(cell, task_ids[0], out, port)


def validate_chunk(out: Path, task_ids: list[str]) -> tuple[list[str], list[str]]:
    """Split the chunk's official result rows into healthy vs bad tasks.

    BFCL v4 multi_turn rows carry `result` (+ per-turn fields) on success and
    a `traceback` field on inference errors; there is no model_responses
    field in this vintage. Healthy = a row exists and traceback is None.
    Connection errors, terminal controller failures and timeouts all leave a
    traceback and must never be counted as an official zero score.
    """
    import glob
    requested = ordered_unique(task_ids)
    requested_set = set(requested)
    observed: dict[str, bool] = {}
    for path in glob.glob(str(out / "bfcl_worker" / "bfcl" / "result" / "**" / "*.json"),
                          recursive=True):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            task_id = row.get("id")
            if task_id not in requested_set:
                continue
            # Any valid official row completes the task.  Later traceback rows
            # cannot turn an already valid task back into a retry.
            observed[task_id] = observed.get(task_id, False) or bfcl_row_is_valid(
                row, fc_model=True)
    healthy = [task_id for task_id in requested if observed.get(task_id, False)]
    bad = [task_id for task_id in requested if not observed.get(task_id, False)]
    return healthy, bad


def validate_appworld_summary(out: Path, task_id: str) -> dict:
    path = out / "appworld_worker" / "official_summary.json"
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"missing or invalid AppWorld summary for {task_id}") from error
    score = summary.get("semantic_score") if isinstance(summary, dict) else None
    if (not isinstance(summary, dict)
            or summary.get("schema") != "a-event-native-appworld-run-v1"
            or summary.get("status") != "completed"
            or summary.get("task_id") != task_id
            or summary.get("n") != 1
            or not isinstance(score, (int, float)) or isinstance(score, bool)
            or not math.isfinite(score)):
        raise RuntimeError(f"AppWorld summary is not a scored completion for {task_id}")
    return summary


def appworld_task_completed(cell_dir: Path, task_id: str) -> bool:
    for path in (cell_dir / "batches").glob("*/appworld_worker/official_summary.json"):
        try:
            validate_appworld_summary(path.parent.parent, task_id)
        except RuntimeError:
            continue
        return True
    return False


def tau2_task_completed(cell_dir: Path, task_id: str) -> bool:
    return any(_qualified_tau2_result(path, task_id) is not None for path in
               (cell_dir / "batches").glob(f"*/tau2_worker/{task_id}"))


def _qualified_tau2_result(output_root: Path, task_id: str) -> dict | None:
    result = completed_tau2_result(output_root, task_id)
    if result is None:
        return None
    code = result.get("task_failure_kind")
    if code is not None and typed_tau2_budget_cost_finalization(
            output_root.parent.parent, task_id, code)["status"] != "valid":
        return None
    return result


def tau2_score_summary(cell: dict) -> dict:
    cell_dir = Path(cell["cell_dir"])
    rows = []
    pending = []
    for task_id in cell["task_ids"]:
        scored = None
        for path in sorted((cell_dir / "batches").glob(
                f"*/tau2_worker/{task_id}"), reverse=True):
            scored = _qualified_tau2_result(path, task_id)
            if scored is not None:
                break
        if scored is None:
            pending.append(task_id)
        else:
            row = {"task_id": task_id, "semantic_score": scored["semantic_score"],
                   "termination": scored["termination"],
                   "score_source": scored.get("score_source", "official_tau2")}
            if "task_failure_kind" in scored:
                row["task_failure_kind"] = scored["task_failure_kind"]
                row["official_reward"] = scored["official_reward"]
            rows.append(row)
    budget_failures = [row["task_id"] for row in rows if "task_failure_kind" in row]
    return {
        "schema": "c2kv-tau2-score-summary-v1", "cell_id": cell["cell_id"],
        "n_total": len(cell["task_ids"]),
        "n_official_scored": len(rows) - len(budget_failures),
        "n_budget_failures": len(budget_failures),
        "budget_failure_task_ids": budget_failures,
        "n_completed": len(rows),
        "pending_task_ids": pending,
        "semantic_score": (sum(row["semantic_score"] for row in rows) / len(rows)
                           if not pending else None),
        "task_rows": rows,
    }


def cost_finalization(out: Path) -> dict:
    """Keep official task quality separate from the server's persisted costs."""
    final_path = out / "server" / "final.json"
    if not final_path.is_file():
        return {"status": "unavailable", "reason": "missing_final_receipt"}
    try:
        final = json.loads(final_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {"status": "failed", "reason": "invalid_final_receipt"}
    if not isinstance(final, dict):
        return {"status": "failed", "reason": "invalid_final_receipt"}
    if final.get("cost_summary_error"):
        return {"status": "failed", "reason": "cost_summary_error",
                "error": final["cost_summary_error"]}
    if not isinstance(final.get("cost_summary"), dict) or not final["cost_summary"]:
        return {"status": "unavailable", "reason": "missing_cost_summary"}
    return {"status": "valid"}


def typed_tau2_budget_cost_finalization(out: Path, task_id: str, code: str) -> dict:
    """Qualify a declared budget loss only with a clean, task-bound final journal."""
    existing = cost_finalization(out)
    if existing["status"] != "valid":
        return existing
    try:
        final = json.loads((out / "server" / "final.json").read_text(encoding="utf-8"))
        ready = json.loads((out / "server" / "ready.json").read_text(encoding="utf-8"))
        rejections = [json.loads(line) for line in
                      (out / "server" / "budget_rejections.jsonl").read_text(
                          encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return {"status": "failed", "reason": "invalid_final_receipt"}
    journal = final.get("journal_summary")
    health = final.get("api_health")
    if (final.get("status") != "stopped"
            or not isinstance(ready, dict) or not rejections
            or not isinstance(rejections[-1], dict)
            or ready.get("schema") != "a-event-native-server-v1"
            or ready.get("status") != "ready" or ready.get("benchmark") != "tau2"
            or ready.get("allowed_task_ids") != [task_id]
            or not isinstance(ready.get("run_id"), str) or not ready["run_id"]
            or not isinstance(journal, dict) or not isinstance(health, dict)
            or journal.get("schema") != "a-runtime-attempt-journal-v1"
            or type(journal.get("started")) is not int
            or type(journal.get("completed")) is not int
            or journal["started"] != journal["completed"]
            or journal.get("pending") != 0 or journal.get("failed") != 0
            or health.get("allowed_task_ids") != [task_id]
            or health.get("terminal_reason") != code):
        return {"status": "failed", "reason": "unverified_typed_budget_final"}
    rejection = rejections[-1]
    if (rejection.get("schema") != "a-event-native-budget-rejection-v1"
            or rejection.get("run_id") != ready["run_id"]
            or rejection.get("task_id") != task_id
            or rejection.get("session_id") != f"tau2/{task_id}/attempt-0"
            or rejection.get("status_code") != 429 or rejection.get("code") != code):
        return {"status": "failed", "reason": "unverified_typed_budget_rejection"}
    if code == "generation_cap_reached":
        used, cap = health.get("generation_calls_reserved"), health.get("max_generation_calls")
        rejection_used, rejection_cap = (
            rejection.get("generation_calls_reserved"), rejection.get("max_generation_calls"))
    elif code == "decision_cap_reached":
        used, cap = health.get("decisions_reserved"), health.get("max_decisions")
        rejection_used, rejection_cap = (
            rejection.get("decisions_reserved"), rejection.get("max_decisions"))
    else:
        return {"status": "failed", "reason": "unknown_typed_budget_code"}
    if (type(used) is not int or type(cap) is not int or cap <= 0 or used != cap
            or rejection_used != used or rejection_cap != cap):
        return {"status": "failed", "reason": "unverified_typed_budget_cap"}
    return {"status": "valid"}


def appworld_method_failure_evidence(out: Path, task_id: str) -> dict | None:
    """Accept only a durable, task-bound capacity failure from our controller."""
    # A declared capacity failure may precede final.json, but it cannot hide
    # an invalid final receipt or a persisted cost-summary failure.
    if cost_finalization(out)["status"] == "failed":
        return None
    ready_path = out / "server" / "ready.json"
    steps_path = out / "server" / "steps.jsonl"
    try:
        ready = json.loads(ready_path.read_text(encoding="utf-8"))
        if (ready.get("schema") != "a-event-native-server-v1"
                or ready.get("benchmark") != "acon_appworld"
                or ready.get("allowed_task_ids") != [task_id]):
            return None
        rows = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError, AttributeError):
        return None
    session_id = f"acon_appworld/{task_id}/attempt-0"
    failures = [row for row in rows if isinstance(row, dict)
                and row.get("schema") == "a-event-native-exact-step-v1"
                and row.get("status") == "failed"
                and row.get("session_id") == session_id
                and row.get("failure_kind") == "method_failure"
                and row.get("failure_code") == "c2kv_capacity_infeasible"]
    if len(failures) != 1:
        return None
    try:
        return {
            "batch": out.name,
            "server_ready_sha256": hashlib.sha256(ready_path.read_bytes()).hexdigest(),
            "server_steps_sha256": hashlib.sha256(steps_path.read_bytes()).hexdigest(),
            "decision_key": failures[0].get("decision_key"),
        }
    except OSError:
        return None


def appworld_method_failure_receipt(cell_dir: Path, task_id: str) -> dict | None:
    path = cell_dir / "tasks" / task_id / "terminal.json"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (not isinstance(receipt, dict)
            or receipt.get("schema") != "c2kv-appworld-method-failure-v1"
            or receipt.get("task_id") != task_id
            or receipt.get("status") != "method_failure"
            or receipt.get("failure_code") != "c2kv_capacity_infeasible"
            or type(receipt.get("semantic_score")) not in (int, float)
            or receipt.get("semantic_score") != 0.0
            or receipt.get("score_source") != "method_failure_zero"
            or receipt.get("official_summary") is not None):
        return None
    evidence = receipt.get("evidence")
    batch = evidence.get("batch") if isinstance(evidence, dict) else None
    if not isinstance(batch, str) or not batch or Path(batch).name != batch:
        return None
    return (receipt if evidence == appworld_method_failure_evidence(
        cell_dir / "batches" / batch, task_id) else None)


def appworld_score_summary(cell: dict) -> dict:
    """Keep method-failure zeros in the frozen denominator without a fake scorer row."""
    cell_dir = Path(cell["cell_dir"])
    rows = []
    pending = []
    official = method_failures = 0
    scored_by_task = {}
    expected = set(cell["task_ids"])
    for path in sorted((cell_dir / "batches").glob("*/appworld_worker/official_summary.json")):
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
            task_id = candidate.get("task_id") if isinstance(candidate, dict) else None
            if isinstance(task_id, str) and task_id in expected and task_id not in scored_by_task:
                scored_by_task[task_id] = validate_appworld_summary(path.parent.parent,
                                                                     task_id)
        except (OSError, json.JSONDecodeError, RuntimeError):
            continue
    for task_id in cell["task_ids"]:
        scored = scored_by_task.get(task_id)
        if scored is not None:
            official += 1
            rows.append({"task_id": task_id, "semantic_score": scored["semantic_score"],
                         "score_source": "official_appworld"})
        elif appworld_method_failure_receipt(cell_dir, task_id) is not None:
            method_failures += 1
            rows.append({"task_id": task_id, "semantic_score": 0.0,
                         "score_source": "method_failure_zero",
                         "failure_code": "c2kv_capacity_infeasible"})
        else:
            pending.append(task_id)
    return {
        "schema": "c2kv-appworld-score-summary-v1",
        "cell_id": cell["cell_id"], "n_total": len(cell["task_ids"]),
        "score_denominator": len(cell["task_ids"]),
        "n_scored": len(rows), "n_official_scored": official,
        "n_method_failures": method_failures,
        "method_failure_task_ids": [row["task_id"] for row in rows
                                    if row["score_source"] == "method_failure_zero"],
        "pending_task_ids": pending,
        "semantic_score": (sum(row["semantic_score"] for row in rows) / len(cell["task_ids"])
                           if not pending else None),
        "failure_score_policy": "capacity_infeasible method failures score zero without an official scorer outcome",
        "task_rows": rows,
    }


def _runtime_retrieval_cell(cell: dict, out: Path) -> dict:
    """Record per-attempt retrieval placement without rewriting frozen inputs."""
    overrides = {key: cell[key] for key in ("embedding_device", "embedding_batch_size")
                 if key in cell}
    if not overrides:
        return cell
    if cell["condition"] != "tracer_history":
        raise ValueError("retrieval execution overrides require a tracer cell")
    source = Path(cell["controller_path"])
    frozen = source.read_bytes()
    config = json.loads(frozen)
    embedding = config["gp_experiments"]["local_models"]["embedding"]
    if "embedding_device" in overrides:
        embedding["device"] = overrides["embedding_device"]
    if "embedding_batch_size" in overrides:
        embedding["batch_size"] = overrides["embedding_batch_size"]
    path = out / "effective_controller.json"
    raw = (json.dumps(config, indent=2, ensure_ascii=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
    _write(out / "retrieval_execution.json", {
        "frozen_controller": str(source),
        "frozen_controller_sha256": hashlib.sha256(frozen).hexdigest(),
        "effective_controller": str(path),
        "effective_controller_sha256": hashlib.sha256(raw).hexdigest(),
        "overrides": overrides,
        "ascend_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "scope": "execution placement and microbatching only; model and input contract unchanged",
    })
    return {**cell, "controller_path": str(path)}


def run_task(cell: dict, task_ids: list[str], port: int, batch_dirname: str) -> dict:
    """Run one chunk of tasks under a single controller server instance.

    The controller accepts a frozen comma-separated allowlist; the official
    BFCL worker generates/scores exactly that subset. The returned status
    carries per-task health from the official result rows so the caller can
    bisect a poisoned chunk instead of letting one terminal task zero out the
    rest.
    """
    out = Path(cell["cell_dir"]) / "batches" / batch_dirname
    # Attempts are evidence.  Never reuse a name or remove an older attempt:
    # retries legitimately produce duplicate raw rows which canonical rescore
    # resolves without losing provenance.
    out.mkdir(parents=True, exist_ok=False)
    server_cell = _runtime_retrieval_cell(cell, out)
    env = os.environ.copy()
    # Load torch_npu only when the configured retrieval device requires it.
    # The actor stays in the separate SGLang process.
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["PYTHONPATH"] = os.pathsep.join((str(RUNTIME / "python"), str(RUNTIME)))
    env["no_proxy"] = env["NO_PROXY"] = "127.0.0.1,localhost"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        env.pop(k, None)
    worker_env = env.copy()
    if cell["benchmark"] == "toolsandbox":
        # Official ToolSandbox calls RapidAPI through the host proxy. Local
        # agent/user endpoints stay direct via NO_PROXY; only the worker gets
        # these external-network proxy settings, not the controller server.
        for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            if key in os.environ:
                worker_env[key] = os.environ[key]
    if cell["benchmark"] == "bfcl":
        worker_env["PYTHONPATH"] = str(RUNTIME)
    elif cell["benchmark"] == "acon_appworld":
        # event_native_appworld needs: controller_runtime (for the worker
        # itself), benchmarks/ (for adapters/proxy imports), acon/src (for
        # the ACON harness), and paper_harness/benchmarks (for proxy)
        # event_native_appworld imports ``adapters.acon_adapter`` from the
        # paper harness.  The controller runtime also has an ``adapters``
        # package, so the harness directory must precede runtime/benchmarks.
        paper_source = Path(env.get("C2KV_PAPER_SOURCE")
                            or GENERATION_ROOT / "src" / "paper_harness")
        worker_env["PYTHONPATH"] = os.pathsep.join((
            str(paper_source / "benchmarks"),
            str(RUNTIME), str(RUNTIME / "benchmarks"),
            "/home/liuyancheng/baselines/acon/src"))
        worker_env["APPWORLD_ROOT"] = cell.get("appworld_root", "")
    server_log = (out / "controller.log").open("wb")
    worker_log = (out / "benchmark.log").open("wb")
    started = time.monotonic()
    server = worker = None
    status = {"chunk": batch_dirname, "n_tasks": len(task_ids), "status": "started",
              "started_at": started}
    _write(out / "status.json", status)
    healthy, bad = [], task_ids
    tau2_budget_failure = None
    try:
        server = subprocess.Popen(
            server_command(server_cell, task_ids, out, port), cwd=str(RUNTIME), env=env,
            stdout=server_log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
        ready = out / "server" / "ready.json"
        deadline = time.monotonic() + cell["caps"]["task_timeout"] * len(task_ids)
        upstream_monitor = UpstreamLiveness(cell["sglang_backend_url"])
        while not ready.exists():
            upstream_monitor()
            if server.poll() is not None:
                raise RuntimeError(f"controller exited rc={server.returncode}")
            if time.monotonic() > deadline:
                raise TimeoutError("controller readiness timeout")
            time.sleep(2)
        validate_static_ready_manifest(server_cell, ready)
        if cell["benchmark"] == "tau2":
            if len(task_ids) != 1:
                raise ValueError("tau2 requires one frozen task per controller server")
            task_id = task_ids[0]
            receipt = run_tau2_task(
                cell, task_id, f"http://127.0.0.1:{port}",
                cell["sglang_backend_url"], out / "tau2_worker" / task_id,
                native_server_dir=out / "server")
            if receipt["status"] != "completed":
                raise RuntimeError(f"official tau2 worker did not score {task_id}")
            tau2_budget_failure = receipt.get("task_failure_kind")
        elif cell["benchmark"] == "toolsandbox":
            if len(task_ids) != 1:
                raise ValueError("ToolSandbox requires one frozen task per controller server")
            task_id = task_ids[0]
            worker = subprocess.Popen(
                toolsandbox_worker_command(
                    cell, task_id, f"http://127.0.0.1:{port}/v1",
                    cell["sglang_backend_url"].rstrip("/") + "/v1",
                    out / "toolsandbox_worker" / task_id),
                cwd=str(Path(__file__).resolve().parents[1]),
                env=worker_env, stdout=worker_log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            rc = wait_owned_worker(worker, timeout=max(60, deadline - time.monotonic()),
                                   monitor=upstream_monitor)
            if rc != 0 or toolsandbox_official_result(
                    out / "toolsandbox_worker" / task_id / "official_summary.json",
                    task_id) is None:
                raise RuntimeError(f"official ToolSandbox worker did not score {task_id}")
        elif cell["benchmark"] == "acon_appworld":
            # AppWorld: one task per worker invocation (event_native_appworld)
            for task_id in task_ids:
                worker = subprocess.Popen(
                    appworld_worker_command(cell, task_id, out, port),
                    cwd=str(RUNTIME),
                    env=worker_env, stdout=worker_log, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True,
                )
                rc = wait_owned_worker(worker, timeout=max(60, deadline - time.monotonic()),
                                       monitor=upstream_monitor)
                if rc != 0:
                    raise RuntimeError(f"official worker exited rc={rc} for {task_id}")
                validate_appworld_summary(out, task_id)
        else:
            worker = subprocess.Popen(
                bfcl_worker_command(cell, task_ids, out, port), cwd=str(RUNTIME),
                env=worker_env, stdout=worker_log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
            rc = wait_owned_worker(worker, timeout=max(60, deadline - time.monotonic()),
                                       monitor=upstream_monitor)
            if rc != 0:
                raise RuntimeError(f"official worker exited rc={rc}")
        if cell["benchmark"] == "tau2":
            healthy = [task_id for task_id in task_ids if completed_tau2_task(
                out / "tau2_worker" / task_id, task_id)]
            bad = [task_id for task_id in task_ids if task_id not in healthy]
        elif cell["benchmark"] == "toolsandbox":
            healthy = [task_id for task_id in task_ids if toolsandbox_official_result(
                out / "toolsandbox_worker" / task_id / "official_summary.json",
                task_id) is not None]
            bad = [task_id for task_id in task_ids if task_id not in healthy]
        elif cell["benchmark"] == "acon_appworld":
            # AppWorld results are evaluation JSONs, not BFCL result rows;
            # the worker's rc=0 + official_summary.json are the health check
            healthy, bad = task_ids, []
        else:
            healthy, bad = validate_chunk(out, task_ids)
        status.update(status="completed" if not bad else "partial",
                      healthy=healthy, bad=bad,
                      wall_s=time.monotonic() - started)
    except Exception as error:  # infra failure: preserve healthy rows before retry
        if isinstance(error, UpstreamUnavailable):
            status["infra_failure_kind"] = "upstream_unavailable"
        # The official BFCL worker can exit nonzero after writing valid rows.
        # Preserve those outputs so the caller retries only tasks that truly
        # lack a completed official result.
        if cell["benchmark"] == "tau2":
            healthy = [task_id for task_id in task_ids if completed_tau2_task(
                out / "tau2_worker" / task_id, task_id)]
            bad = [task_id for task_id in task_ids if task_id not in healthy]
            status.update(status="partial" if healthy else "failed", healthy=healthy,
                          bad=bad, error=f"{type(error).__name__}: {error}",
                          wall_s=time.monotonic() - started)
        elif cell["benchmark"] == "toolsandbox":
            healthy = [task_id for task_id in task_ids if toolsandbox_official_result(
                out / "toolsandbox_worker" / task_id / "official_summary.json",
                task_id) is not None]
            bad = [task_id for task_id in task_ids if task_id not in healthy]
            status.update(status="partial" if healthy else "failed", healthy=healthy,
                          bad=bad, error=f"{type(error).__name__}: {error}",
                          wall_s=time.monotonic() - started)
        elif cell["benchmark"] == "bfcl":
            healthy, bad = validate_chunk(out, task_ids)
            if healthy:
                status.update(
                    status="partial" if bad else "completed",
                    healthy=healthy,
                    bad=bad,
                    error=f"{type(error).__name__}: {error}",
                    wall_s=time.monotonic() - started,
                )
            else:
                status.update(status="failed", error=f"{type(error).__name__}: {error}",
                              wall_s=time.monotonic() - started)
        else:
            status.update(status="failed", error=f"{type(error).__name__}: {error}",
                          wall_s=time.monotonic() - started)
    finally:
        with defer_interrupts():
            for proc in (worker, server):
                if proc is None:
                    continue
                # The leader may have exited while --serve-child still owns
                # the group. Its Popen PID remains the group ID we created.
                stop_owned_group(proc)
            server_log.close()
            worker_log.close()
    status["cost_finalization"] = cost_finalization(out)
    if tau2_budget_failure is not None:
        status["cost_finalization"] = typed_tau2_budget_cost_finalization(
            out, task_ids[0], tau2_budget_failure)
        if status["cost_finalization"]["status"] != "valid":
            healthy, bad = [], task_ids
            status.update(status="failed", healthy=healthy, bad=bad,
                          error="typed tau2 budget failure lacks a clean final journal")
    _write(out / "done.json" if not bad and status["cost_finalization"]["status"] == "valid"
           else out / "status.json", status)
    return status


def load_cell(cell_json: Path) -> dict:
    cell = json.loads(cell_json.read_text())
    for key, path in (
        ("controller_path", cell.get("controller_path")),
        ("eval_policy_path", cell.get("eval_policy_path")),
    ):
        if key in cell and not Path(cell[key]).is_absolute():
            cell[key] = str((cell_json.parent / cell[key]).resolve())
    return cell


def prepare_cell_files(cell: dict, budgets: dict) -> dict:
    # Launch placement is recorded per attempt, outside the scientific freeze.
    cell = dict(cell)
    execution = {key: cell.pop(key) for key in ("embedding_device", "embedding_batch_size")
                 if key in cell}
    cell_dir = Path(cell["cell_dir"])
    if "history_budget_tokens" in cell:
        cell["native_history_budget"] = native_history_budget_profile(cell, budgets)
    controller, binding = _controller_with_binding(cell)
    policy = build_eval_policy(cell, budgets)
    prepared = dict(cell)
    prepared["controller_path"] = str(cell_dir / "controller.json")
    prepared["eval_policy_path"] = str(cell_dir / "eval_policy.json")
    frozen = {"controller.json": controller, "eval_policy.json": policy,
              "cell.json": prepared}
    if binding is not None:
        frozen["risk_artifact_binding.json"] = binding

    attempts = cell_dir / "batches"
    if attempts.is_dir() and any(attempts.iterdir()):
        for name, expected in frozen.items():
            path = cell_dir / name
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f"existing attempts require frozen {name}") from error
            if name == "cell.json":
                # Engine address/slot and paths to the separately verified
                # controller/policy files are launch routing, not experiment
                # inputs.  Older source manifests omitted the derived paths.
                previous = dict(previous)
                expected = dict(expected)
                for path_field, frozen_name in (
                    ("controller_path", "controller.json"),
                    ("eval_policy_path", "eval_policy.json"),
                ):
                    recorded_path = previous.get(path_field)
                    if recorded_path is not None:
                        if not isinstance(recorded_path, str) or not recorded_path:
                            raise ValueError(f"existing attempts have invalid frozen {path_field}")
                        source = Path(recorded_path)
                        if not source.is_absolute():
                            source = cell_dir / source
                        try:
                            recorded = json.loads(source.read_text(encoding="utf-8"))
                        except (OSError, json.JSONDecodeError) as error:
                            raise ValueError(
                                f"existing attempts require frozen {frozen_name} at recorded path"
                            ) from error
                        if recorded != frozen[frozen_name]:
                            raise ValueError(
                                f"existing attempts use a different frozen {frozen_name} at recorded path"
                            )
                for transient in ("sglang_backend_url", "scheduler_port_slot",
                                  "controller_path", "eval_policy_path"):
                    previous.pop(transient, None)
                    expected.pop(transient, None)
            if previous != expected:
                raise ValueError(f"existing attempts use a different frozen {name}")
        return {**prepared, **execution}

    for name, value in frozen.items():
        if name == "eval_policy.json" and "history_budget_tokens" in cell:
            # Match the recorded file-byte hash on Windows and on ascend03.
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / name).write_bytes(_json_file_bytes(value))
        else:
            _write(cell_dir / name, value)
    return {**prepared, **execution}


def _attempt_name(task_ids: list[str]) -> str:
    digest = hashlib.sha256("\0".join(task_ids).encode("utf-8")).hexdigest()[:10]
    return f"a{time.time_ns()}_p{os.getpid()}_{digest}"


def _write_bfcl_completion(cell: dict, expected_task_ids: list[str]) -> dict:
    cell_dir = Path(cell["cell_dir"])
    completion = collect_bfcl_results(cell_dir, expected_task_ids, fc_model=True)
    receipt = completion_receipt(completion)
    receipt["cell_id"] = cell["cell_id"]
    receipt["generated_at"] = time.time()
    _write(cell_dir / "bfcl_completion.json", receipt)
    _write(cell_dir / "bfcl_refill.json", {
        "schema": "generality-bfcl-refill-v1",
        "cell_id": cell["cell_id"],
        "task_ids": completion["refill_task_ids"],
        "n_tasks": len(completion["refill_task_ids"]),
        "source": "bfcl_completion.json",
        "generated_at": receipt["generated_at"],
    })
    return completion


@interruptible
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=Path, required=True, help="cell.json path")
    parser.add_argument("--budgets", type=Path, required=True, help="resolved budgets json")
    parser.add_argument("--candidate-algorithm", choices=CANDIDATE_VARIANTS,
                        help="explicit ratio-8 candidate run outside the legacy matrix")
    parser.add_argument("--history-budget-tokens", type=int,
                        help="explicit native history/workspace cap for a candidate cell")
    parser.add_argument("--sglang-backend-url",
                        help="existing engine URL for an explicit candidate run")
    parser.add_argument("--task-ids", nargs="*", default=None,
                        help="subset override; default: cell manifest")
    parser.add_argument("--port-base", type=int, default=37200)
    parser.add_argument("--chunk", type=int, default=8,
                        help="tasks per controller server instance")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument(
        "--max-attempts-per-task", type=int, default=4,
        help="bounded attempts in this invocation, including the initial batch",
    )
    parser.add_argument(
        "--audit-results-only", action="store_true",
        help="write exact BFCL completion/refill manifests without launching work",
    )
    args = parser.parse_args(argv)

    if args.max_attempts_per_task < 1:
        parser.error("--max-attempts-per-task must be at least 1")
    if args.history_budget_tokens is not None and args.history_budget_tokens <= 0:
        parser.error("--history-budget-tokens must be positive")
    cell = load_cell(args.cell)
    if args.history_budget_tokens is not None and args.candidate_algorithm is None:
        parser.error("--history-budget-tokens requires --candidate-algorithm")
    if args.candidate_algorithm is not None:
        cell = candidate_cell_from_source(
            cell, args.candidate_algorithm, args.sglang_backend_url,
            history_budget_tokens=args.history_budget_tokens)
    elif args.sglang_backend_url is not None:
        parser.error("--sglang-backend-url requires --candidate-algorithm")
    expected_task_ids = cell["task_ids"]
    if (not isinstance(expected_task_ids, list) or not expected_task_ids
            or any(not isinstance(task_id, str) or not task_id
                   for task_id in expected_task_ids)
            or len(expected_task_ids) != len(set(expected_task_ids))):
        raise ValueError("cell manifest must contain unique nonempty task IDs")
    requested_task_ids = ordered_unique(
        args.task_ids if args.task_ids is not None else expected_task_ids
    )
    unexpected_requested = sorted(set(requested_task_ids) - set(expected_task_ids))
    if unexpected_requested:
        raise ValueError(
            "requested task IDs are outside the frozen cell manifest: "
            + ",".join(unexpected_requested)
        )
    if args.audit_results_only:
        if cell["benchmark"] != "bfcl":
            raise ValueError("--audit-results-only is supported only for BFCL cells")
        completion = _write_bfcl_completion(cell, expected_task_ids)
        print(json.dumps(completion_receipt(completion), ensure_ascii=False))
        return 0

    budgets = json.loads(args.budgets.read_text())
    # Existing attempts freeze the controller, policy, and cell contract;
    # prepare_cell_files rejects drift before writing any of them.
    cell = prepare_cell_files(cell, budgets)
    task_ids = requested_task_ids
    if args.max_tasks is not None:
        task_ids = task_ids[: args.max_tasks]

    # Ordinary OpenAI harness requests carry no task identity. Each server
    # must therefore own one frozen official task.
    if cell["benchmark"] in {"acon_appworld", "tau2", "toolsandbox"}:
        args.chunk = 1

    cell_dir = Path(cell["cell_dir"])
    progress = cell_dir / "progress.jsonl"
    port_counter = [args.port_base]
    attempt_counts = {task_id: 0 for task_id in task_ids}
    valid_task_ids: set[str] = set()
    if cell["benchmark"] == "bfcl":
        completion = _write_bfcl_completion(cell, expected_task_ids)
        valid_task_ids.update(completion["valid_task_ids"])
        task_ids = [task_id for task_id in task_ids if task_id not in valid_task_ids]

    def next_port() -> int:
        """Monotonic port: the scheduler now gives each card 1000 ports, so
        a driver never crosses into a neighbor's range; TIME_WAIT from a
        previous server on the SAME port is handled by the strict probe."""
        import socket
        while True:
            port_counter[0] += 1
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(("127.0.0.1", port_counter[0]))
                    return port_counter[0]
                except OSError:
                    continue

    def run_chunk(chunk: list[str], depth: int) -> None:
        """Run a chunk; on failure or unhealthy tasks, bisect down to singles.

        BFCL tracebacks and missing rows remain retryable; only a structured
        official output completes a task.  Attempts in one invocation are
        bounded by ``--max-attempts-per-task``.
        """
        if not chunk:
            return
        # BFCL resumes from canonical official rows.  AppWorld keeps its
        # existing per-task marker protocol.
        if cell["benchmark"] == "bfcl":
            # A timeout/nonzero worker may finish writing a valid row while
            # run_task is cleaning up its process group.  Refresh from disk
            # before every retry decision so that late output is not rerun.
            refreshed = collect_bfcl_results(cell_dir, expected_task_ids, fc_model=True)
            valid_task_ids.update(refreshed["valid_task_ids"])
            chunk = [
                task_id for task_id in chunk
                if task_id not in valid_task_ids
                and attempt_counts.get(task_id, 0) < args.max_attempts_per_task
            ]
        elif cell["benchmark"] == "tau2":
            chunk = [
                task_id for task_id in chunk
                if not tau2_task_completed(cell_dir, task_id)
                and attempt_counts.get(task_id, 0) < args.max_attempts_per_task
            ]
        elif cell["benchmark"] == "toolsandbox":
            chunk = [
                task_id for task_id in chunk
                if not completed_toolsandbox_task(cell_dir, task_id)
                and attempt_counts.get(task_id, 0) < args.max_attempts_per_task
            ]
        else:
            chunk = [
                task_id for task_id in chunk
                if not appworld_task_completed(cell_dir, task_id)
                and appworld_method_failure_receipt(cell_dir, task_id) is None
                and attempt_counts.get(task_id, 0) < args.max_attempts_per_task
            ]
        if not chunk:
            return
        for task_id in chunk:
            attempt_counts[task_id] = attempt_counts.get(task_id, 0) + 1
        port = next_port()
        name = _attempt_name(chunk)
        result = run_task(cell, chunk, port, name)
        new_method_failure = False
        if (cell["benchmark"] == "acon_appworld" and len(chunk) == 1
                and result["status"] != "completed"
                and not appworld_task_completed(cell_dir, chunk[0])):
            evidence = appworld_method_failure_evidence(cell_dir / "batches" / name,
                                                        chunk[0])
            if evidence is not None:
                terminal_path = cell_dir / "tasks" / chunk[0] / "terminal.json"
                if terminal_path.exists():
                    with (terminal_path.parent / "terminal_history.jsonl").open(
                            "a", encoding="utf-8") as history:
                        history.write(json.dumps({"preserved_at_ns": time.time_ns(),
                            "previous_raw": terminal_path.read_text(encoding="utf-8")}) + "\n")
                _write(terminal_path, {
                    "schema": "c2kv-appworld-method-failure-v1",
                    "task_id": chunk[0], "status": "method_failure",
                    "failure_code": "c2kv_capacity_infeasible",
                    "semantic_score": 0.0, "score_source": "method_failure_zero",
                    "official_summary": None, "evidence": evidence,
                })
                result = {**result, "status": "method_failure",
                          "terminal_kind": "method_failure",
                          "failure_code": "c2kv_capacity_infeasible"}
                new_method_failure = True
        with progress.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        healthy = list(result.get("healthy") or [])
        if result.get("infra_failure_kind") == "upstream_unavailable":
            valid_task_ids.update(healthy)
            if cell["benchmark"] == "bfcl":
                completion = _write_bfcl_completion(cell, expected_task_ids)
                counts = {
                    "n_completed": completion["valid_count"],
                    "n_retryable": len(completion["refill_task_ids"]),
                }
            elif cell["benchmark"] == "tau2":
                summary = tau2_score_summary(cell)
                _write(cell_dir / "tau2_score_summary.json", summary)
                counts = {
                    "n_completed": summary["n_completed"],
                    "n_budget_failures": summary["n_budget_failures"],
                    "n_retryable": len(summary["pending_task_ids"]),
                }
            elif cell["benchmark"] == "toolsandbox":
                summary = toolsandbox_score_summary(cell)
                _write(cell_dir / "toolsandbox_score_summary.json", summary)
                counts = {
                    "n_completed": summary["n_official_scored"],
                    "n_retryable": len(summary["pending_task_ids"]),
                }
            else:
                summary = appworld_score_summary(cell)
                _write(cell_dir / "appworld_score_summary.json", summary)
                counts = {
                    "n_completed": summary["n_official_scored"],
                    "n_terminal": summary["n_method_failures"],
                    "n_retryable": len(summary["pending_task_ids"]),
                }
            write_cell_status(cell, {
                "cell_id": cell["cell_id"], "status": "incomplete",
                "stop_reason": "upstream_unavailable", "error": result.get("error"),
                "n_total": len(expected_task_ids), **counts,
                "finished_at": time.time(),
            })
            raise UpstreamUnavailable(result.get("error") or "Inference service disappeared")
        if new_method_failure:
            print(json.dumps({"cell": cell["cell_id"], "task": chunk[0],
                              "status": "method_failure"}), flush=True)
            return
        if result["status"] == "completed":
            for task_id in healthy:
                valid_task_ids.add(task_id)
                _write(cell_dir / "tasks" / task_id / "done.json",
                       {"task_id": task_id, "status": "completed"})
            print(json.dumps({"cell": cell["cell_id"], "chunk": len(chunk),
                              "status": "ok", "done": healthy and len(healthy)}), flush=True)
            return
        # partial or failed: bisect
        if len(chunk) == 1:
            task_id = chunk[0]
            # A worker exit without a scored task is not evidence of a model
            # failure.  In particular, old AppWorld cells recorded the
            # acon_adapter ImportError as terminal for every task.
            _write(cell_dir / "tasks" / task_id / "retryable.json",
                   {"task_id": task_id,
                    "status": "retryable",
                    "reason": result.get("error") or "unhealthy_result_rows",
                    "chunk_status": result["status"]})
            print(json.dumps({"cell": cell["cell_id"], "task": task_id,
                              "status": "retryable_recorded"}), flush=True)
            return
        # healthy tasks inside a partial chunk still count
        for task_id in healthy:
            valid_task_ids.add(task_id)
            _write(cell_dir / "tasks" / task_id / "done.json",
                   {"task_id": task_id, "status": "completed"})
        remainder = [t for t in chunk if t not in set(healthy)]
        half = max(1, len(remainder) // 2)
        run_chunk(remainder[:half], depth + 1)
        run_chunk(remainder[half:], depth + 1)

    for i in range(0, len(task_ids), args.chunk):
        run_chunk(task_ids[i:i + args.chunk], 0)

    if cell["benchmark"] == "bfcl":
        completion = _write_bfcl_completion(cell, expected_task_ids)
        write_cell_status(cell, {
            "cell_id": cell["cell_id"],
            "status": (
                "complete"
                if completion["valid_count"] == completion["expected_count"]
                else "incomplete"
            ),
            "n_completed": completion["valid_count"],
            "n_valid_unique": completion["valid_count"],
            "n_retryable": len(completion["refill_task_ids"]),
            "n_total": completion["expected_count"],
            "raw_result_rows": completion["total_rows"],
            "duplicate_result_rows": completion["duplicate_rows"],
            "finished_at": time.time(),
        })
    elif cell["benchmark"] == "tau2":
        summary = tau2_score_summary(cell)
        _write(cell_dir / "tau2_score_summary.json", summary)
        write_cell_status(cell, {
            "cell_id": cell["cell_id"],
            "status": "complete" if not summary["pending_task_ids"] else "incomplete",
            "n_completed": summary["n_completed"],
            "n_budget_failures": summary["n_budget_failures"],
            "n_retryable": len(summary["pending_task_ids"]),
            "n_total": len(expected_task_ids),
            "semantic_score": summary["semantic_score"],
            "score_denominator": len(expected_task_ids),
            "score_summary": str(cell_dir / "tau2_score_summary.json"),
            "finished_at": time.time(),
        })
    elif cell["benchmark"] == "toolsandbox":
        summary = toolsandbox_score_summary(cell)
        _write(cell_dir / "toolsandbox_score_summary.json", summary)
        write_cell_status(cell, {
            "cell_id": cell["cell_id"],
            "status": "complete" if not summary["pending_task_ids"] else "incomplete",
            "n_completed": summary["n_official_scored"],
            "n_retryable": len(summary["pending_task_ids"]),
            "n_total": len(expected_task_ids),
            "semantic_score": summary["semantic_score"],
            "score_denominator": summary["score_denominator"],
            "score_summary": str(cell_dir / "toolsandbox_score_summary.json"),
            "finished_at": time.time(),
        })
    else:
        summary = appworld_score_summary(cell)
        _write(cell_dir / "appworld_score_summary.json", summary)
        done = summary["n_official_scored"]
        terminal = summary["n_method_failures"]
        retryable = len(summary["pending_task_ids"])
        write_cell_status(cell, {
            "cell_id": cell["cell_id"],
            "status": "complete" if retryable == 0 else "incomplete",
            "n_completed": done, "n_retryable": retryable,
            "n_terminal": terminal,
            "method_failure_task_ids": summary["method_failure_task_ids"],
            "n_total": len(expected_task_ids),
            "semantic_score": summary["semantic_score"],
            "score_denominator": summary["score_denominator"],
            "score_summary": str(cell_dir / "appworld_score_summary.json"),
            "finished_at": time.time(),
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
