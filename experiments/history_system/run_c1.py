"""Run the selected C1 recovery controller with an official benchmark harness.

The controller/runtime is shared by BFCL, tau2, and ToolSandbox. This file
only owns orchestration: one controller and one official harness worker per
explicit task identity.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import current
import evidence_sets
import runner
from c1_artifact_binding import bind_risk_artifact
from benchmarks.memory_runtime.candidate_algorithms import ALL_VARIANTS
from benchmarks.memory_runtime.racer.config import BackendConfig

RACER_CANDIDATE_POLICIES = tuple(ALL_VARIANTS)


def racer_arm_name(backend: str, policy: str, history_budget_tokens: int) -> str:
    return f"racer_{backend}_{policy}_b{history_budget_tokens}"


def validate_racer_backend(value: Mapping[str, Any]) -> dict:
    """Validate the standalone delivery contract without importing paper orchestration."""
    from dataclasses import asdict

    parsed = BackendConfig.parse(value)
    parsed.history_spec()
    normalized = {"schema": "racer-backend-v1", **asdict(parsed)}
    if dict(value) != normalized:
        raise ValueError("RACER backend config differs from its resolved arm contract")
    return normalized

HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
# Mirrors benchmarks.memory_runtime.tokenization (TOOL_SCHEMA_MODES, DEFAULT_TOOL_SCHEMA);
# the runtime package is importable only from the delivery's own PYTHONPATH.
TOOL_SCHEMA_MODES = ("sglang-full", "raw")
TOOL_SCHEMA_DEFAULT = "sglang-full"
DEFAULT_RISK_ARTIFACT = HERE / "artifacts/c1_risk.t02_v1.json"
DEFAULT_RISK_ARTIFACT_SHA256 = "18a11f73aa1f7d4b0add86eed66ae9e5e129ea4bdfbe0dfad23faf4f7d2fb4ab"
BENCHMARKS = ("bfcl", "tau2", "toolsandbox")
METHODS = ("proposed", "c2kv_only", "c2kv_native")
SOURCE_PROFILE = {
    "bfcl": "native-v1",
    "tau2": "openai-single-task-v1",
    "toolsandbox": "openai-single-task-v1",
}
OFFICIAL_SCORER = {
    "bfcl": "official BFCL generation + checker",
    "tau2": "tau2 run -> tau2 evaluate-trajs -> reward_info.reward",
    "toolsandbox": "tool_sandbox official CLI -> result_summary.json",
}
SUMMARY_FIELDS = (
    "benchmark", "task_id", "status", "official_score", "normal_termination",
    "protocol_legal", "generation_calls", "detector_calls",
    "detector_trigger_count", "recovery_count", "successful_recovery_count",
    "evidence_units_appended", "raw_tokens_restored", "full_history_kv",
    "native_raw_events_restored", "native_raw_prompt_token_delta",
    "native_raw_history_token_delta", "native_active_history_byte_delta",
    "active_history_kv", "kv_retention", "compression_ratio",
    "generation_prefill_tokens", "recovery_prefill_tokens",
    "total_prefill_tokens", "wall_time", "native_generate_requests",
    "prefill_detector_scores", "prefill_detector_unavailable",
    "risk_detector_scores", "risk_detector_unavailable",
    "candidate_decisions", "candidate_variants", "candidate_ratio8",
    "candidate_stable_call_ids", "candidate_budget_passed",
    "racer_backend_identity", "racer_generation_receipts",
    "racer_actual_generation_count", "racer_accounting_passed",
    "racer_generation_backend_match",
    "racer_transaction_receipts", "racer_transactions_complete",
    "racer_actual_cost_complete", "racer_effective_budget_match",
    "racer_max_history_and_evidence_tokens", "racer_kv_bytes_per_token",
    "gist_tokens", "raw_workspace_tokens",
    "gist_cache_hits", "native_packing_present",
)

PREFILL_GATE_UNAVAILABLE_REASONS = frozenset({
    "shadow_features_schema_unavailable",
    "prefill_hidden_unavailable",
    "prefill_layer_mismatch",
    "prefill_position_mismatch",
    "prefill_readout_mismatch",
    "prefill_feature_dimension_mismatch",
    "prefill_hidden_invalid",
})


def save(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def source_revision(path: Path) -> dict:
    """Record source identity without fetching or modifying its worktree."""
    root = path.resolve()
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, text=True, capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--short"],
        check=True, text=True, capture_output=True,
    ).stdout.splitlines()
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "HEAD", "--binary"],
        check=True, capture_output=True,
    ).stdout
    dirty_files = {}
    for row in status:
        relative = row[3:].split(" -> ")[-1]
        candidate = root / relative
        if candidate.is_file():
            dirty_files[relative] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    return {
        "path": str(root), "commit": commit, "dirty": bool(status),
        "status_short": status, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_diff_bytes": len(diff), "dirty_file_sha256": dirty_files,
    }


def _bare_endpoint(value: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    if (parsed.scheme != "http" or not parsed.netloc or parsed.username is not None
            or parsed.password is not None or parsed.query or parsed.fragment):
        raise ValueError(f"expected a bare HTTP endpoint, got {value!r}")
    if parsed.path.rstrip("/") not in {"", "/v1"}:
        raise ValueError(f"expected a bare HTTP endpoint, got {value!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def preflight_sglang_backend(
    args: argparse.Namespace, *, opener=None,
) -> dict[str, Any]:
    """Fail before creating a run directory when the native engine is absent.

    This is capability discovery only; it never submits generation.
    """
    base = _bare_endpoint(args.sglang_backend_url)
    endpoint = f"{base}/model_info"
    opener = opener or urlopen
    try:
        with opener(Request(endpoint, method="GET"), timeout=10) as response:
            info = json.load(response)
    except (HTTPError, URLError, OSError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"C1 SGLang backend is not ready at {endpoint}: "
            f"{type(error).__name__}: {error}. Start the PR #5 SGLang engine "
            "and verify /model_info before running run_c1.py."
        ) from error
    if not isinstance(info, Mapping):
        raise RuntimeError(f"C1 SGLang backend returned invalid /model_info at {endpoint}")
    capability = info.get("c2kv_native_packed")
    if not isinstance(capability, Mapping):
        raise RuntimeError("SGLang /model_info lacks c2kv_native_packed capability")
    binding = capability.get("model_binding")
    if not isinstance(binding, Mapping):
        raise RuntimeError("SGLang native-packed capability lacks model_binding")
    expected_checkpoint = args.checkpoint.resolve()
    observed_paths = [info.get("model_path"), binding.get("model_path")]
    if any(
        not isinstance(path, str) or Path(path).resolve() != expected_checkpoint
        for path in observed_paths
    ):
        raise RuntimeError(
            "SGLang checkpoint binding differs from --checkpoint: "
            f"expected={expected_checkpoint}, observed={observed_paths}"
        )
    checks = {
        "native_generate_enabled": capability.get("enabled") is True,
        "native_generate_endpoint": (
            capability.get("endpoint") == "/v1/c2kv/native_generate"
        ),
        "bfloat16": binding.get("dtype") == "bfloat16",
        "base_query_projection": (
            capability.get("base_query_enforced") is True
            and binding.get("query_projection") == "base"
        ),
        "prefill_feature": (
            isinstance(capability.get("shadow_feature_layer"), int)
            and capability.get("shadow_feature_readout") == "decoder_layer_output"
        ),
        "gist_extraction_packing": (
            binding.get("gist_type") == "dynamic-interleave"
            and binding.get("gist_param") == "qkv"
            and isinstance(capability.get("packing_version"), str)
            and bool(capability.get("packing_version"))
        ),
    }
    if args.method == "c2kv_native" or getattr(args, "candidate_algorithm", None) in {
        "request_contract", "argument_binding", "no_progress"}:
        checks.pop("prefill_feature")
    if not all(checks.values()):
        raise RuntimeError(f"SGLang C1 capability check failed: {checks}")
    return {
        "endpoint": endpoint,
        "checkpoint": str(expected_checkpoint),
        "checks": checks,
        "packing_version": capability["packing_version"],
        "shadow_feature_layer": capability.get("shadow_feature_layer"),
    }


def _build_profile_unbudgeted(args: argparse.Namespace) -> tuple[dict, dict]:
    """Keep D3-hybrid, compatibility Prefill, and trained risk modes distinct."""
    if getattr(args, "candidate_algorithm", None) is not None:
        import candidate_algorithms

        return candidate_algorithms.build_profile(
            args,
            base_controller=evidence_sets._base_controller(),
            selected=current.load_config(),
            risk_artifact_path=DEFAULT_RISK_ARTIFACT,
            risk_artifact_sha256=DEFAULT_RISK_ARTIFACT_SHA256,
            bind_artifact=bind_risk_artifact,
        )
    if args.method == "c2kv_native":
        import native_bare
        if args.selector_artifact is not None or args.ratio not in (4, 8):
            raise ValueError("Native bare C2KV requires ratio4 or ratio8 and no selector artifact")
        actual = hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest()
        if actual != current.load_config()["checkpoint_selection"]["config_sha256"]:
            raise ValueError("Native bare delivery requires the selected C1000 checkpoint")
        return {}, dict(native_bare.profile(args.ratio), checkpoint=str(args.checkpoint.resolve()),
                        checkpoint_config_sha256=actual, automatic_reruns=0)
    if args.method == "c2kv_only":
        if args.selector_artifact is not None:
            raise ValueError("c2kv_only does not accept --selector-artifact")
        controller = evidence_sets._base_controller()
        controller.pop("post_draft_recovery", None)
        controller.pop("gp_experiments", None)
        controller.pop("d3_hybrid_recovery", None)
        selected = current.load_config()
        actual = hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest()
        if actual != selected["checkpoint_selection"]["config_sha256"]:
            raise ValueError("C1 delivery requires the selected C1000 checkpoint config; this runtime is model-bound")
        return controller, {
            "schema": "c1-delivery-profile-v2",
            "method": "c2kv_only",
            "detector": "disabled",
            "algorithm": "C2KV event-native compression only",
            "new_c1_training_claimed": False,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_config_sha256": actual,
            "selection_protocol": None,
            "history_variant": "H0",
            "ratio": effective_ratio(args, selected),
            "controller_sha256": hashlib.sha256(
                json.dumps(controller, sort_keys=True).encode()
            ).hexdigest(),
            "automatic_reruns": 0,
        }
    if args.detector != "t02_risk" and args.selector_artifact is not None:
        raise ValueError("--selector-artifact is only used with --detector t02_risk")
    artifact_path = None
    artifact_sha256 = None
    if args.detector == "t02_risk":
        artifact_path = args.selector_artifact or DEFAULT_RISK_ARTIFACT
        if not artifact_path.is_file():
            raise FileNotFoundError(f"selector_artifact does not exist: {artifact_path}")
        artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if args.selector_artifact is None and artifact_sha256 != DEFAULT_RISK_ARTIFACT_SHA256:
            raise ValueError("Bundled T02 C1 artifact differs from the evaluated release")
    artifact_binding = None
    if args.detector == "d3_hybrid":
        controller = evidence_sets._base_controller()
        recovery = controller.get("post_draft_recovery")
        if not isinstance(recovery, Mapping) or recovery.get("gate") != "prefill_linear_head":
            raise ValueError(
                "d3_hybrid requires the frozen native post_draft_recovery Prefill head"
            )
        controller["d3_hybrid_recovery"] = True
        controller.pop("gp_experiments", None)
    else:
        embedding_batch_size = getattr(args, "embedding_batch_size", 16)
        if type(embedding_batch_size) is not int or embedding_batch_size < 1:
            raise ValueError("embedding batch size must be a positive integer")
        config, _ = evidence_sets.build_config(
            history="H0",
            selector="legacy_prefill" if args.detector == "legacy_prefill" else "risk",
            selector_artifact=artifact_path,
            selector_threshold=args.selector_threshold,
            embedding_model=str(args.embedding_model.resolve()),
            embedding_device=args.embedding_device,
            semantic_query_overflow_policy="task_head_tail_preserve_draft_v1",
        )
        config["local_models"]["embedding"]["dtype"] = "bfloat16"
        config["local_models"]["embedding"]["batch_size"] = embedding_batch_size
        if artifact_path is not None:
            config["selector_artifact"], artifact_binding = bind_risk_artifact(
                config["selector_artifact"], args.checkpoint
            )
        controller = current._configure_controller(evidence_sets._base_controller(), config)
        controller.pop("d3_hybrid_recovery", None)
    selected = current.load_config()
    actual = hashlib.sha256((args.checkpoint / "config.json").read_bytes()).hexdigest()
    if actual != selected["checkpoint_selection"]["config_sha256"]:
        raise ValueError("C1 delivery requires the selected C1000 checkpoint config; this detector is model-bound")
    if args.detector == "d3_hybrid":
        recovery = controller["post_draft_recovery"]
        algorithm = "D3-hybrid complete-event raw recovery"
        selection_protocol = "d3_hybrid_recovery_v1"
        algorithm_contract = {
            "candidate_order": "candidate_first",
            "query": "goal_draft_latest_complete_tool_observation",
            "empty_draft_guard": "empty_text_and_no_legal_tool_call_abstain",
            "revision_policy": (
                "exclude_revision_cancelled_without_global_explicit_revision_abstain"
            ),
            "admission": "first_feasible_complete_event_with_native_b0_repack",
            "presentation": "complete_event_raw",
            "gate": "frozen_prefill_linear_head_after_candidate_feasibility",
            "gate_artifact_sha256": recovery["prefill_head"]["artifact_sha256"],
            "gate_threshold": recovery["prefill_head"]["threshold"],
            "cumulative_recovery_quota": False,
            "recovery_rounds_per_decision": 1,
        }
    else:
        algorithm = (
            "C1 legacy Prefill compatibility"
            if args.detector == "legacy_prefill"
            else "C1 T02 risk"
        )
        selection_protocol = "evidence_sets_v1"
        algorithm_contract = None
    return controller, {
        "schema": "c1-delivery-profile-v2",
        "method": "proposed",
        "detector": args.detector,
        "algorithm": algorithm,
        "algorithm_contract": algorithm_contract,
        "new_c1_training_claimed": args.detector == "t02_risk",
        "selector_artifact": str(artifact_path.resolve()) if artifact_path else None,
        "selector_artifact_sha256": artifact_sha256,
        "selector_artifact_binding": artifact_binding,
        "selector_threshold": args.selector_threshold if artifact_path else None,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_config_sha256": actual,
        "selection_protocol": selection_protocol,
        "history_variant": "H0",
        "ratio": effective_ratio(args, selected),
        "controller_sha256": hashlib.sha256(json.dumps(controller, sort_keys=True).encode()).hexdigest(),
        "automatic_reruns": 0,
    }


def _history_budget_override(args: argparse.Namespace, design: dict | None = None) -> dict | None:
    tokens = getattr(args, "history_budget_tokens", None)
    racer = getattr(args, "racer_backend_config", None)
    if tokens is None and racer is not None:
        tokens = racer["history_budget_tokens"]
    if tokens is None:
        return None
    if racer is None and args.method == "c2kv_native":
        raise ValueError("--history-budget-tokens does not support native bare static packing")
    if racer is None and args.benchmark != "bfcl":
        raise ValueError("--history-budget-tokens currently supports BFCL only")
    import history_budget

    return history_budget.resolve_override(
        tokens, args.checkpoint, design or current.load_config(), RUNTIME, args.out,
    )


def build_profile(args: argparse.Namespace) -> tuple[dict, dict]:
    """Return the selected controller and an optional native budget contract."""
    override = _history_budget_override(args)
    controller, profile = _build_profile_unbudgeted(args)
    racer = getattr(args, "racer_backend_config", None)
    if racer is not None:
        racer = validate_racer_backend(racer)
        controller = dict(controller)
        controller["racer_backend"] = racer
        profile["racer_backend"] = racer
        profile["composition_identity"] = racer_arm_name(
            racer["backend"], racer["policy"], racer["history_budget_tokens"])
        profile["history_budget_tokens"] = racer["history_budget_tokens"]
        profile["calibration_status"] = racer["detector_calibration"]
        profile.pop("ratio", None)
        profile["controller_sha256"] = hashlib.sha256(
            json.dumps(controller, sort_keys=True).encode()).hexdigest()
    if override is not None:
        profile["racer_runtime_budget" if racer is not None else "native_history_budget"] = override
    return controller, profile


def _identities(args: argparse.Namespace) -> list[str]:
    if args.benchmark == "bfcl":
        values = args.task_id
        pattern = r"multi_turn_(?:base|long_context)_[0-9]+"
    elif args.benchmark == "tau2":
        values = args.tau2_task_id
        pattern = r"[A-Za-z0-9_.:-]+"
    else:
        values = args.ts_scenario
        pattern = r"[A-Za-z0-9_.:-]+"
    if not values:
        raise ValueError(f"benchmark={args.benchmark} requires at least one explicit task/scenario ID")
    if any(re.fullmatch(pattern, value) is None for value in values) or len(values) != len(set(values)):
        raise ValueError("task/scenario identities must be unique path-safe official IDs")
    return values


def _benchmark_dir(args: argparse.Namespace) -> Path:
    if args.benchmark == "bfcl":
        if args.benchmark_dir is None or not (args.benchmark_dir / "bfcl_eval").is_dir():
            raise ValueError("--benchmark-dir must contain the official bfcl_eval package")
        return args.benchmark_dir
    if args.benchmark == "tau2":
        if args.tau2_dir is None or not (args.tau2_dir / "src" / "tau2").is_dir():
            raise ValueError("--tau2-dir must contain src/tau2")
        return args.tau2_dir
    if args.toolsandbox_dir is None or not (args.toolsandbox_dir / "tool_sandbox").is_dir():
        raise ValueError("--toolsandbox-dir must contain tool_sandbox")
    return args.toolsandbox_dir


def _controller_endpoint(args: argparse.Namespace) -> str:
    return f"http://127.0.0.1:{args.port}"


def effective_ratio(args: argparse.Namespace, selected: dict) -> int:
    """The delivered ratio (8) unless --ratio asks for the other supported C0 ratio."""
    ratio = getattr(args, "ratio", None)
    return int(ratio) if ratio else int(selected["ratio"])


def _model_name(args: argparse.Namespace) -> str:
    racer = getattr(args, "racer_backend_config", None)
    if racer is not None:
        return racer_arm_name(
            racer["backend"], racer["policy"], racer["history_budget_tokens"])
    if getattr(args, "candidate_algorithm", None) is not None:
        return f"c2kv_{args.candidate_algorithm}"
    if args.method == "c2kv_native":
        import native_bare
        return native_bare.arm_for_ratio(args.ratio)
    return "c2kv_only" if args.method == "c2kv_only" else f"c1_{args.detector}"


def portable_worker_command(args: argparse.Namespace, task: str, task_out: Path) -> list[str]:
    if args.benchmark == "bfcl":
        raise ValueError("BFCL uses the frozen native official worker")
    python = args.tau2_python if args.benchmark == "tau2" else args.toolsandbox_python
    run_identity = hashlib.sha256(
        str(task_out.resolve()).encode("utf-8")
    ).hexdigest()[:12]
    command = [
        python, "-m", "c2kv_eval.portable.c1_worker",
        "--benchmark", args.benchmark,
        "--task-id", task,
        "--agent-base-url", _controller_endpoint(args),
        "--user-base-url", args.user_base_url,
        "--benchmark-dir", str(_benchmark_dir(args).resolve()),
        "--bench-python", python,
        "--out", str(task_out / args.benchmark),
        "--run-name", f"c1_{args.benchmark}_{task}_{run_identity}",
        "--model", _model_name(args),
        "--task-timeout", str(args.task_timeout),
    ]
    if args.benchmark == "tau2":
        command.extend(["--task-set", args.task_set])
        if args.tau2_max_steps is not None:
            command.extend(["--tau2-max-steps", str(args.tau2_max_steps)])
    else:
        command.extend(["--ts-agent", args.ts_agent, "--ts-user", args.ts_user])
    return command


def commands_for_task(args: argparse.Namespace, task: str, controller_path: Path) -> tuple[list[str], list[str]]:
    design = current.load_config()
    override = _history_budget_override(args, design)
    if override is not None:
        import history_budget

        # A normal run has already created --out. A preview keeps it untouched.
        if args.out.is_dir() and not getattr(args, "preview", False):
            history_budget.materialize(override)
        design["runtime"]["eval_policy"] = override["override_eval_policy_path"]
    design["ratio"] = effective_ratio(args, design)
    design["candidate_id"] = _model_name(args)
    design["run_id_template"] = design["candidate_id"]
    design["runtime"].update(controller=str(controller_path), sglang_backend_url=args.sglang_backend_url)
    if args.method == "c2kv_native":
        import native_bare
        design = native_bare.configure_design(design, args.ratio)
    if args.method == "c2kv_only" or getattr(args, "candidate_algorithm", None) in {
        "request_contract", "argument_binding", "no_progress"}:
        design["runtime"].pop("shadow_feature_config", None)
    temporary_controller = None
    if args.method != "c2kv_native" and not controller_path.is_file():
        # Preview validates the exact server argv without creating the requested
        # output directory. The runner reads the controller to validate backend
        # requirements, so give that read a short-lived identical copy.
        controller, _ = build_profile(args)
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        temporary_controller = Path(handle.name)
        try:
            json.dump(controller, handle)
        finally:
            handle.close()
        design["runtime"]["controller"] = str(temporary_controller)
    try:
        server = runner.server_command(
            design, task_id=task, checkpoint=str(args.checkpoint.resolve()),
            output=str(args.out.resolve()), port=args.port, python=sys.executable,
            benchmark=args.benchmark, source_profile=SOURCE_PROFILE[args.benchmark],
        )
    finally:
        if temporary_controller is not None:
            temporary_controller.unlink(missing_ok=True)
    if "--s0-config" in server:
        server[server.index("--s0-config") + 1] = str(controller_path)
    if args.tool_memory != "none":
        server.extend(["--tool-memory", args.tool_memory])
        if args.tool_checkpoint is not None:
            server.extend(["--tool-checkpoint", str(args.tool_checkpoint.resolve())])
        if args.tool_budget_tokens is not None:
            server.extend(["--tool-budget-tokens", str(args.tool_budget_tokens)])
    if getattr(args, "tool_schema", TOOL_SCHEMA_DEFAULT) != TOOL_SCHEMA_DEFAULT:
        # Explicit non-default prologue only; the release server command stays byte-identical.
        server.extend(["--tool-schema", args.tool_schema])
    task_out = args.out / "task_shards" / task
    if args.benchmark == "bfcl":
        worker = runner.worker_command(
            design, task_id=task, output=str(args.out.resolve()),
            benchmark_dir=str(_benchmark_dir(args).resolve()), port=args.port,
            python=args.bfcl_python, max_wall_seconds=args.task_timeout,
        )
    else:
        worker = portable_worker_command(args, task, task_out)
    return server, worker


def _official_summary_path(args: argparse.Namespace, task_out: Path) -> Path:
    return task_out / args.benchmark / "official_summary.json"


def _single_official_row(benchmark: str, summary: Mapping[str, Any]) -> dict:
    if benchmark == "bfcl":
        if summary.get("n_scored") != 1 or summary.get("n_generated") != 1:
            raise RuntimeError("Official BFCL did not generate and score exactly the requested task")
        score = summary.get("semantic_score")
        if score is None:
            score = summary.get("accuracy")
        return {"semantic_score": score, "normal_termination": True, "protocol_legal": None}
    if summary.get("n") != 1:
        raise RuntimeError(f"Official {benchmark} harness did not score exactly one task")
    rows = summary.get("task_rows") or []
    if len(rows) != 1:
        raise RuntimeError(f"Official {benchmark} summary requires exactly one task row")
    return dict(rows[0])


def _walk(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def summarize_task(benchmark: str, task: str, task_out: Path, official: Mapping[str, Any], wall_time: float) -> dict:
    """Join official scores with C1-native telemetry without redefining either."""
    official_row = _single_official_row(benchmark, official)
    records = []
    ready_path = task_out / "server" / "ready.json"
    ready = (json.loads(ready_path.read_text(encoding="utf-8"))
             if ready_path.is_file() else {})
    steps_path = task_out / "server" / "steps.jsonl"
    if steps_path.exists():
        for line in steps_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    traces = [trace for record in records for trace in record.get("generation_trace", [])]
    nodes = list(_walk(records))
    # One durable step record represents one committed decision.  Inspect its
    # final decision object exactly once: recursively walking recovery_rounds
    # would count the same recovery both there and in exact_recovery.
    decisions = [
        record.get("exact_recovery")
        for record in records
        if isinstance(record.get("exact_recovery"), Mapping)
    ]
    gates = [
        decision["gate"]
        for decision in decisions
        if isinstance(decision.get("gate"), Mapping)
    ]
    risk_selections = [
        decision["selection"] for decision in decisions
        if isinstance(decision.get("selection"), Mapping)
        and decision["selection"].get("selector") == "risk"
        and decision["selection"].get("score_semantics") == "current_turn_failure_risk"
    ]
    legacy_gates = [gate for gate in gates if gate.get("type") != "risk"]
    prefill_evaluations = [
        gate for gate in legacy_gates
        if _number(gate.get("score")) is not None
        or gate.get("reason") in PREFILL_GATE_UNAVAILABLE_REASONS
    ]
    prefill_unavailable = sum(
        gate.get("reason") in PREFILL_GATE_UNAVAILABLE_REASONS
        for gate in prefill_evaluations
    )
    risk_scores = sum(
        selection.get("available") is True and _number(selection.get("score")) is not None
        for selection in risk_selections
    )
    risk_unavailable = sum(selection.get("available") is False for selection in risk_selections)
    candidate_decisions = [
        decision for decision in decisions
        if decision.get("version") in {
            "c2kv-paper-candidates-v1", "c2kv-source-repair-v1",
            "c2kv-goal-composition-v1", "c2kv-verified-binding-v1",
            "c2kv-initial-view-composition-v1", "c2kv-static-extension-v1",
            "c2kv-c1-v2-verified-v1"}
    ]
    candidate_traces = [
        trace for trace in traces
        if isinstance(trace.get("controller"), Mapping)
        and isinstance(trace["controller"].get("candidate_algorithm"), Mapping)
    ]
    budget_checks = [
        check for record in records
        for check in record.get("pre_generation_budget_checks", [])
    ]
    recovery_rows = [
        decision for decision in decisions if decision.get("status") == "recover"
    ]
    restored_events = [
        decision["restored_event"]
        for decision in recovery_rows
        if isinstance(decision.get("restored_event"), Mapping)
        and decision["restored_event"].get("representation") == "native_raw_event"
    ]
    generation_calls = sum(
        1 for trace in traces if trace.get("status") in {"started", "pending", "completed", "failed"}
    )
    native_stats = [
        trace.get("generation", {}).get("stats", {})
        for trace in traces
        if isinstance(trace.get("generation"), Mapping)
        and isinstance(trace.get("generation", {}).get("stats"), Mapping)
        and trace.get("generation", {}).get("stats", {}).get("backend")
            == "sglang_c2kv_native_packed"
    ]
    racer_rows = []
    for record in records:
        for trace in record.get("generation_trace", []):
            generation = trace.get("generation")
            stats = generation.get("stats") if isinstance(generation, Mapping) else None
            if (trace.get("status") == "completed" and isinstance(stats, Mapping)
                    and isinstance(stats.get("racer_backend"), Mapping)):
                racer_rows.append((record, trace, stats))
    racer_identities = sorted({
        stats["racer_backend"].get("identity")
        for _, _, stats in racer_rows
        if isinstance(stats["racer_backend"].get("identity"), str)
    })
    racer_accounting_rows = []
    racer_transaction_rows = []
    racer_cost_rows = []
    racer_actual_generation_count = 0
    for record, trace, stats in racer_rows:
        accounting = stats.get("racer_accounting")
        receipt = stats.get("racer_backend")
        budget = receipt.get("history_budget_tokens") if isinstance(receipt, Mapping) else None
        accounting_valid = False
        if isinstance(accounting, Mapping):
            history = accounting.get("active_history_tokens")
            evidence = accounting.get("native_evidence_tokens")
            combined = accounting.get("history_and_evidence_tokens")
            accounting_valid = (
                type(history) is int and history >= 0
                and type(evidence) is int and evidence >= 0
                and type(combined) is int and combined == history + evidence
                and type(budget) is int and combined <= budget
            )
        racer_accounting_rows.append((accounting_valid, accounting))

        report = stats.get("kv_memory_report")
        lifecycle = report.get("history_kv_lifecycle") if isinstance(report, Mapping) else None
        transaction = (lifecycle.get("transaction") if isinstance(lifecycle, Mapping) else None)
        if not isinstance(transaction, Mapping) and isinstance(report, Mapping):
            transaction = report.get("racer_transaction")
        expected_phase = "regenerate" if trace.get("phase") == "regeneration" else "draft"
        racer_transaction_rows.append(
            isinstance(lifecycle, Mapping)
            and lifecycle.get("persistent_session_enabled") is True
            and lifecycle.get("full_history_reprefill_performed") is False
            and isinstance(lifecycle.get("session_id"), str)
            and bool(lifecycle["session_id"])
            and isinstance(transaction, Mapping)
            and transaction.get("decision_id") == record.get("decision_key")
            and transaction.get("phase") == expected_phase
        )

        usage = trace.get("usage")
        token_ids = generation.get("token_ids")
        resident = accounting.get("resident_prompt_tokens") if isinstance(accounting, Mapping) else None
        cost_valid = (
            isinstance(usage, Mapping) and isinstance(token_ids, list)
            and type(resident) is int and resident >= 0
            and type(usage.get("prompt_tokens")) is int
            and type(usage.get("completion_tokens")) is int
            and type(usage.get("total_tokens")) is int
            and usage["prompt_tokens"] == resident
            and usage["completion_tokens"] == len(token_ids)
            and usage["total_tokens"] == resident + len(token_ids)
            and usage == stats.get("racer_served_usage")
        )
        racer_cost_rows.append(cost_valid)
        calls = stats.get("generation_calls")
        if type(calls) is int:
            racer_actual_generation_count += calls

    ready_racer = ready.get("racer_backend") if isinstance(ready, Mapping) else None
    runtime_policy = ready.get("runtime_policy_contract") if isinstance(ready, Mapping) else None
    effective_policy = (runtime_policy.get("effective_policy")
                        if isinstance(runtime_policy, Mapping) else None)
    kv_unit = (effective_policy.get("kv_bytes_per_token")
               if isinstance(effective_policy, Mapping) else None)
    ready_budget = (ready_racer.get("history_budget_tokens")
                    if isinstance(ready_racer, Mapping) else None)
    racer_generation_backend_match = (
        bool(racer_rows) and isinstance(ready_racer, Mapping)
        and all(dict(stats["racer_backend"]) == dict(ready_racer)
                for _, _, stats in racer_rows)
    )
    expected_budget_bytes = (
        ready_budget * kv_unit
        if type(ready_budget) is int and type(kv_unit) is int else None
    )
    racer_effective_budget_match = (
        expected_budget_bytes is not None
        and isinstance(effective_policy, Mapping)
        and effective_policy.get("history_budget_bytes") == expected_budget_bytes
        and effective_policy.get("workspace_budget_bytes") == expected_budget_bytes
    )
    full_bytes = active_bytes = 0.0
    kv_bytes_per_token = None
    generation_prefill = recovery_prefill = 0.0
    for trace in traces:
        if trace.get("status") != "completed":
            continue
        usage = trace.get("usage") or trace.get("generation_usage") or {}
        prompt_tokens = _number(usage.get("prompt_tokens")) or 0.0
        if trace.get("phase") == "regeneration":
            recovery_prefill += prompt_tokens
        else:
            generation_prefill += prompt_tokens
        for node in _walk(trace):
            ratio = node.get("compression_ratio")
            if isinstance(ratio, Mapping):
                full = _number(ratio.get("full_history_bytes"))
                active = _number(ratio.get("active_history_bytes"))
                if full is not None and active is not None:
                    full_bytes += full
                    active_bytes += active
                    unit = _number(node.get("kv_bytes_per_token")) or _number(ratio.get("kv_bytes_per_token"))
                    if unit:
                        kv_bytes_per_token = unit
                    break
    if kv_bytes_per_token is None:
        for node in nodes:
            unit = _number(node.get("kv_bytes_per_token"))
            if unit:
                kv_bytes_per_token = unit
                break
    restored = sum(
        _number(unit.get("token_count")) or 0.0
        for row in recovery_rows for unit in row.get("appended_units") or []
    )
    successful = sum(
        isinstance(record.get("exact_recovery"), Mapping)
        and record["exact_recovery"].get("status") == "recover"
        and any(
            trace.get("phase") == "regeneration"
            and trace.get("status") == "completed"
            for trace in record.get("generation_trace", [])
        )
        for record in records
    )
    score = official_row.get("semantic_score")
    if score is None:
        score = official.get("semantic_score")
    repair_commits = [record["commit_validation"] for record in records
                      if isinstance(record.get("commit_validation"), Mapping)]
    return {
        "benchmark": benchmark, "task_id": task, "status": "completed",
        "official_score": score,
        "normal_termination": official_row.get("normal_termination"),
        "protocol_legal": official_row.get("protocol_legal"),
        "generation_calls": generation_calls,
        "decision_count": len(records),
        "detector_calls": len(prefill_evaluations) + len(risk_selections),
        "detector_trigger_count": sum(gate.get("triggered") is True for gate in gates),
        "recovery_count": len(recovery_rows),
        "successful_recovery_count": successful,
        **({"repair_commit": {
            "checks": len(repair_commits),
            "accepted_regenerations": sum(
                row.get("accepted") is True
                and isinstance(row.get("selected_generation_index"), int)
                and row["selected_generation_index"] > 0 for row in repair_commits),
            "reverted_to_original": sum(
                row.get("accepted") is False and row.get("fallback") == "original"
                for row in repair_commits),
            "source_supported_abstentions": sum(
                row.get("synthetic_abstention") is True for row in repair_commits),
            "recovery_count_semantics": "attempted; does not imply revision accepted",
        }} if repair_commits else {}),
        "evidence_units_appended": sum(int(row.get("appended_unit_count") or 0) for row in recovery_rows),
        "raw_tokens_restored": int(restored),
        "native_raw_events_restored": len(restored_events),
        "native_raw_prompt_token_delta": int(sum(
            _number(event.get("marginal_raw_prompt_tokens")) or 0.0
            for event in restored_events
        )),
        "native_raw_history_token_delta": int(sum(
            _number(event.get("marginal_raw_history_tokens")) or 0.0
            for event in restored_events
        )),
        "native_active_history_byte_delta": int(sum(
            _number(event.get("marginal_active_history_bytes")) or 0.0
            for event in restored_events
        )),
        "full_history_kv": full_bytes / kv_bytes_per_token if kv_bytes_per_token else None,
        "active_history_kv": active_bytes / kv_bytes_per_token if kv_bytes_per_token else None,
        "kv_retention": active_bytes / full_bytes if full_bytes else None,
        "compression_ratio": full_bytes / active_bytes if active_bytes else None,
        "generation_prefill_tokens": generation_prefill,
        "recovery_prefill_tokens": recovery_prefill,
        "total_prefill_tokens": generation_prefill + recovery_prefill,
        "wall_time": wall_time,
        "native_generate_requests": len(native_stats),
        "prefill_detector_scores": sum(
            _number(gate.get("score")) is not None for gate in prefill_evaluations
        ),
        "prefill_detector_unavailable": prefill_unavailable,
        "risk_detector_scores": risk_scores,
        "risk_detector_unavailable": risk_unavailable,
        "candidate_decisions": len(candidate_decisions),
        "candidate_variants": sorted({
            decision.get("variant") for decision in candidate_decisions
            if isinstance(decision.get("variant"), str)
        }),
        "candidate_ratio8": bool(records) and all(record.get("ratio") == 8 for record in records)
            and len(candidate_traces) == len(traces) and all(
                trace["controller"].get("requested_ratio") == 8 for trace in candidate_traces
            ),
        "candidate_stable_call_ids": bool(candidate_traces)
            and len(candidate_traces) == len(traces) and all(
            trace["controller"]["candidate_algorithm"].get("stable_call_ids") is True
            for trace in candidate_traces
        ),
        "candidate_budget_passed": bool(budget_checks)
            and len(budget_checks) == len(traces) and all(
            check.get("status") == "passed" for check in budget_checks
        ),
        "racer_backend_receipt": ready_racer,
        "racer_backend_identity": (
            ready_racer.get("identity") if isinstance(ready_racer, Mapping) else None),
        "racer_generation_receipts": len(racer_rows),
        "racer_actual_generation_count": racer_actual_generation_count,
        "racer_accounting_passed": bool(racer_accounting_rows) and all(
            valid for valid, _ in racer_accounting_rows),
        "racer_generation_backend_match": racer_generation_backend_match,
        "racer_transaction_receipts": sum(racer_transaction_rows),
        "racer_transactions_complete": bool(racer_transaction_rows) and all(racer_transaction_rows),
        "racer_actual_cost_complete": bool(racer_cost_rows) and all(racer_cost_rows),
        "racer_effective_budget_match": racer_effective_budget_match,
        "racer_max_history_and_evidence_tokens": max((
            int(accounting["history_and_evidence_tokens"])
            for valid, accounting in racer_accounting_rows
            if valid and isinstance(accounting, Mapping)
        ), default=None),
        "racer_kv_bytes_per_token": kv_unit,
        "racer_observed_identities": racer_identities,
        "gist_tokens": sum(int(stats.get("gist_tokens") or 0) for stats in native_stats),
        "raw_workspace_tokens": sum(int(stats.get("workspace_tokens") or 0) for stats in native_stats),
        "gist_cache_hits": sum(
            int(stats.get("scope_reused_chunks") or 0)
            + int(stats.get("session_reused_gist_chunks") or 0)
            for stats in native_stats
        ),
        "native_packing_present": bool(native_stats) and any(
            int(stats.get("gist_tokens") or 0) > 0
            and int(stats.get("workspace_tokens") or 0) > 0
            for stats in native_stats
        ),
    }


def write_unified_summary(out: Path, rows: list[dict]) -> None:
    save(out / "unified_summary.json", {"schema": "c1-unified-summary-v1", "rows": rows})
    with (out / "unified_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in SUMMARY_FIELDS} for row in rows)


def functional_checks(method: str, detector: str, telemetry: Mapping[str, Any],
                      candidate_algorithm: str | None = None,
                      racer_backend: Mapping[str, Any] | None = None) -> dict:
    """Separate required runtime behavior from descriptive efficiency telemetry."""

    repair_candidate = candidate_algorithm in {
        "request_contract", "argument_binding", "no_progress"}
    if repair_candidate:
        detector_contract = (telemetry.get("risk_detector_scores", 0) == 0
                             and telemetry.get("risk_detector_unavailable", 0) == 0)
    elif candidate_algorithm is not None:
        detector_contract = (telemetry.get("risk_detector_scores", 0) > 0
                             and telemetry.get("risk_detector_unavailable", 0) == 0)
    elif method in {"c2kv_only", "c2kv_native"}:
        detector_contract = telemetry["detector_calls"] == 0
    elif detector == "t02_risk":
        detector_contract = telemetry["risk_detector_unavailable"] == 0
    elif detector == "d3_hybrid":
        detector_contract = telemetry["prefill_detector_unavailable"] == 0
    else:
        detector_contract = telemetry["prefill_detector_scores"] > 0
    candidate_required = ({
        "candidate_decisions": telemetry.get("candidate_decisions") == telemetry.get("decision_count")
            and telemetry.get("candidate_decisions", 0) > 0,
        "candidate_variant": telemetry.get("candidate_variants") == [candidate_algorithm],
        **({"no_risk_scores": telemetry.get("risk_detector_scores", 0) == 0,
            "no_risk_unavailable": telemetry.get("risk_detector_unavailable", 0) == 0}
           if repair_candidate else {
               "risk_scores": telemetry.get("risk_detector_scores", 0) > 0,
               "risk_available": telemetry.get("risk_detector_unavailable", 0) == 0}),
        **({"ratio8": telemetry.get("candidate_ratio8") is True}
           if racer_backend is None else {}),
        "stable_call_ids": telemetry.get("candidate_stable_call_ids") is True,
        "budget_passed": telemetry.get("candidate_budget_passed") is True,
    } if candidate_algorithm is not None else {})
    racer_required = {}
    persistent_racer = False
    if racer_backend is not None:
        expected_racer = validate_racer_backend(racer_backend)
        expected_identity = "racer:{backend}:{policy}:b{budget}".format(
            backend=expected_racer["backend"], policy=expected_racer["policy"],
            budget=expected_racer["history_budget_tokens"])
        expected_receipt = telemetry.get("racer_backend_receipt")
        receipt_matches = isinstance(expected_receipt, Mapping) and all(
            expected_receipt.get(key) == value for key, value in expected_racer.items())
        persistent_racer = expected_racer["backend"] != "c2kv"
        observed_identity = telemetry.get("racer_observed_identities")
        racer_required = {
            "racer_backend_identity": (
                receipt_matches
                and telemetry.get("racer_backend_identity") == expected_identity
                and (observed_identity == [expected_identity] if persistent_racer
                     else observed_identity in ([], [expected_identity]))
            ),
            "racer_effective_budget": telemetry.get("racer_effective_budget_match") is True,
        }
        if persistent_racer:
            racer_required.update({
                "racer_generation_receipts": (
                    telemetry.get("racer_generation_receipts", 0) > 0
                    and telemetry.get("racer_generation_receipts")
                    == telemetry.get("generation_calls")
                    == telemetry.get("racer_actual_generation_count")
                ),
                "racer_accounting": telemetry.get("racer_accounting_passed") is True,
                "racer_generation_backend": (
                    telemetry.get("racer_generation_backend_match") is True),
                "racer_transactions": (
                    telemetry.get("racer_transactions_complete") is True
                    and telemetry.get("racer_transaction_receipts")
                    == telemetry.get("racer_generation_receipts")
                ),
                "racer_actual_cost": telemetry.get("racer_actual_cost_complete") is True,
            })
        if expected_racer["policy"] == "t02":
            racer_required["racer_detector_scores"] = (
                telemetry.get("risk_detector_scores", 0) > 0
                and telemetry.get("risk_detector_unavailable", 0) == 0)
    return {
        "required": {
            **({} if persistent_racer else {
                "native_generate_requests": telemetry["native_generate_requests"] > 0}),
            "detector_contract": detector_contract,
            **candidate_required,
            **racer_required,
            **({"no_recovery": telemetry.get("recovery_count", 0) == 0,
                "one_generation_per_decision": telemetry.get("generation_calls") == telemetry.get("decision_count")}
               if method == "c2kv_native" else {}),
        },
        "observed": {
            "native_packing_present": telemetry["native_packing_present"] is True,
            "gist_cache_used": telemetry["gist_cache_hits"] > 0,
            "compression_ratio_gt_one": (
                telemetry["compression_ratio"] is not None
                and telemetry["compression_ratio"] > 1.0
            ),
        },
    }


def run_task(args: argparse.Namespace, task: str, controller_path: Path,
             *, termination_guard=None) -> tuple[dict, dict]:
    server_command, worker_command = commands_for_task(args, task, controller_path)
    task_out = args.out / "task_shards" / task
    task_out.mkdir(parents=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(RUNTIME / "python"), str(RUNTIME)))
    worker_env = env.copy()
    if args.benchmark == "bfcl":
        worker_env["PYTHONPATH"] = str(RUNTIME)
    else:
        worker_env["PYTHONPATH"] = os.pathsep.join((str(args.portable_root.resolve()), str(RUNTIME)))
    process = worker = None
    deadline = time.monotonic() + args.task_timeout
    started = time.monotonic()
    summary_path = _official_summary_path(args, task_out)
    try:
        with (task_out / "controller.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                server_command, cwd=RUNTIME, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=os.name == "posix",
            )
            ready_path = task_out / "server" / "ready.json"
            while not ready_path.exists():
                if process.poll() is not None:
                    raise RuntimeError(f"Controller exited {process.returncode}; see {task_out / 'controller.log'}")
                if time.monotonic() >= deadline:
                    raise TimeoutError("Controller readiness timeout")
                time.sleep(1)
            if args.method == "c2kv_native":
                import native_bare
                native_bare.validate_manifest(ready_path, args.ratio)
            if getattr(args, "tool_memory", "none") != "none":
                from benchmarks.memory_runtime.event_native_tool import validate_ready_tool_contract
                validate_ready_tool_contract(
                    json.loads(ready_path.read_text(encoding="utf-8")),
                    args.tool_memory, getattr(args, "tool_checkpoint", None),
                    getattr(args, "tool_budget_tokens", None))
            with (task_out / "benchmark.log").open("w", encoding="utf-8") as bench_log:
                worker = subprocess.Popen(
                    worker_command, cwd=RUNTIME, env=worker_env, stdout=bench_log,
                    stderr=subprocess.STDOUT, start_new_session=os.name == "posix",
                )
                result = worker.wait(timeout=max(1, deadline - time.monotonic()))
            if result:
                raise RuntimeError(f"Official {args.benchmark} worker exited {result}; see {task_out / 'benchmark.log'}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            _single_official_row(args.benchmark, summary)
    finally:
        from contextlib import nullcontext
        with termination_guard() if termination_guard is not None else nullcontext():
            try:
                if worker is not None:
                    runner._stop_bfcl(worker, task_out / args.benchmark / "running.json")
            finally:
                if process is not None:
                    runner._stop_server(process, task_out / "server.supervisor.json")
    final_path = task_out / "server" / "final.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    if (final.get("cost_summary_error") or final.get("status") == "failed"
            or final.get("stop_reason") == "runner_failed" or process.returncode != 0):
        raise RuntimeError(f"Controller finalization failed; see {final_path}")
    journal = final.get("journal_summary") or {}
    if journal.get("failed") or journal.get("pending") or not journal.get("completed"):
        raise RuntimeError(f"Model attempts failed, remain pending, or are missing; see {final_path}")
    telemetry = summarize_task(args.benchmark, task, task_out, summary, time.monotonic() - started)
    acceptance = functional_checks(
        args.method, args.detector, telemetry,
        getattr(args, "candidate_algorithm", None),
        getattr(args, "racer_backend_config", None),
    )
    required = acceptance["required"]
    if not all(required.values()):
        raise RuntimeError(f"C1 functional acceptance failed: {required}; see {task_out / 'server'}")
    return {
        "task_id": task, "status": "completed", "official_summary": summary,
        "unified_metrics": telemetry, "qualification": "official single-task harness result",
    }, telemetry


def _parse_racer_backend(value: str) -> dict:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("RACER backend config must be JSON") from error
    try:
        return validate_racer_backend(loaded)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="bfcl")
    parser.add_argument(
        "--method", choices=METHODS, default="proposed",
        help="proposed enables recovery; c2kv_only keeps the S0 initial policy without recovery; c2kv_native uses static native gist packing without S0",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sglang-backend-url", required=True)
    parser.add_argument("--embedding-model", type=Path)
    parser.add_argument("--embedding-device", default="cpu")
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument(
        "--detector",
        choices=("d3_hybrid", "legacy_prefill", "t02_risk"),
        default="t02_risk",
    )
    parser.add_argument(
        "--candidate-algorithm",
        choices=("static_t02", "turn_c1", "goal_rescue", "dependency_first",
                 "request_contract", "argument_binding", "no_progress",
                 "goal_pending", "goal_source", "goal_progress", "goal_joint",
                 "goal_verified", "pending_verified", "goal_static", "pending_static",
                 "goal_verified_static", "pending_verified_static",
                 "static_verified", "static_action_ledger", "static_verified_v2",
                 "c1_v2_verified"),
        default=None,
    )
    parser.add_argument("--selector-artifact", type=Path,
                        help="Override the bundled, evaluated T02 C1 risk artifact")
    parser.add_argument("--selector-threshold", type=float, default=0.5)
    parser.add_argument("--benchmark-dir", type=Path)
    parser.add_argument("--bfcl-python", default=sys.executable)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--tau2-dir", type=Path)
    parser.add_argument("--tau2-python", default=sys.executable)
    parser.add_argument("--tau2-task-id", action="append", default=[])
    parser.add_argument("--task-set", default="airline")
    parser.add_argument("--tau2-max-steps", type=int)
    parser.add_argument("--toolsandbox-dir", type=Path)
    parser.add_argument("--toolsandbox-python", default=sys.executable)
    parser.add_argument("--ts-scenario", action="append", default=[])
    parser.add_argument("--ts-agent", default="GPT_4_o_2024_05_13")
    parser.add_argument("--ts-user", default="GPT_4_o_2024_05_13")
    parser.add_argument("--user-base-url")
    parser.add_argument("--portable-root", type=Path, default=Path("/home/zhuyuhan/project/bfcl-c2kv"))
    parser.add_argument(
        "--sglang-root", type=Path,
        default=Path("/home/zhuyuhan/project/kvoffload-sglang-c2kv-pr5-runtime"),
        help="SGLang checkout served by --sglang-backend-url; recorded for provenance",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=38810)
    parser.add_argument("--task-timeout", type=int, default=3600)
    parser.add_argument("--ratio", type=int, choices=sorted(runner.SUPPORTED_RATIOS), default=None,
                        help="Override the delivered compression ratio with the other supported C0 ratio (ablation)")
    parser.add_argument("--tool-memory", default="none")
    parser.add_argument("--tool-checkpoint", type=Path)
    parser.add_argument("--tool-budget-tokens", type=int)
    parser.add_argument("--tool-schema", choices=TOOL_SCHEMA_MODES, default=TOOL_SCHEMA_DEFAULT,
                        help="Native tool prologue: sglang-full (release default, matches Full) "
                             "or raw (client tool JSON unchanged)")
    parser.add_argument("--history-budget-tokens", type=int,
                        help="Override both native C2KV history and workspace byte caps using checkpoint KV geometry")
    parser.add_argument("--racer-backend-config", type=_parse_racer_backend,
                        help="resolved racer-backend-v1 modular history/policy composition")
    parser.add_argument("--preview", action="store_true", help="Print commands without model, harness, or network calls")
    return parser


def validate_args(args: argparse.Namespace) -> list[str]:
    budget_tokens = getattr(args, "history_budget_tokens", None)
    if budget_tokens is not None and (type(budget_tokens) is not int or budget_tokens <= 0):
        raise ValueError("--history-budget-tokens must be a positive integer")
    if budget_tokens is not None and args.method == "c2kv_native":
        raise ValueError("--history-budget-tokens does not support native bare static packing")
    if budget_tokens is not None and args.benchmark != "bfcl":
        raise ValueError("--history-budget-tokens currently supports BFCL only")
    racer = getattr(args, "racer_backend_config", None)
    if racer is not None:
        validate_racer_backend(racer)
        if budget_tokens is not None:
            raise ValueError("RACER history budget is frozen in --racer-backend-config")
        if args.ratio not in (None, 8):
            raise ValueError("RACER uses the existing ratio-8 policy factory and an explicit token budget")
        policy = racer["policy"]
        if policy == "off":
            valid_policy_route = args.method == "c2kv_only" and args.candidate_algorithm is None
        elif policy == "t02":
            valid_policy_route = (args.method == "proposed" and args.detector == "t02_risk"
                                  and args.candidate_algorithm is None)
        else:
            valid_policy_route = (args.method == "proposed"
                                  and args.candidate_algorithm == policy
                                  and policy in RACER_CANDIDATE_POLICIES)
        if not valid_policy_route:
            raise ValueError("RACER policy must use its unchanged native C1 policy factory")
    if args.tool_memory == "none":
        if args.tool_checkpoint is not None or args.tool_budget_tokens is not None:
            raise ValueError("Tool options require --tool-memory")
    elif args.tool_memory.startswith("t0:"):
        if args.tool_checkpoint is None or not (args.tool_checkpoint / "config.json").is_file():
            raise ValueError("T0 tool memory requires a local --tool-checkpoint")
    else:
        from benchmarks.memory_runtime.event_native_tool import parse_native_tool_spec

        spec = parse_native_tool_spec(args.tool_memory)
        if spec is None or spec.encoder == "t0" or spec.interface_policy != "schema":
            raise ValueError(
                "Global H2O/SnapKV selection across disjoint visible tool spans is not implemented"
            )
        if args.tool_checkpoint is not None:
            raise ValueError("Raw-KV tool memory does not use --tool-checkpoint")
    if args.tool_budget_tokens is not None and args.tool_budget_tokens <= 0:
        raise ValueError("--tool-budget-tokens must be positive")
    identities = _identities(args)
    _benchmark_dir(args)
    if args.task_timeout <= 0 or not 1 <= args.port <= 65535:
        raise ValueError("task timeout and port must be valid positive values")
    embedding_batch_size = getattr(args, "embedding_batch_size", 16)
    if type(embedding_batch_size) is not int or embedding_batch_size < 1:
        raise ValueError("embedding batch size must be a positive integer")
    if (args.method != "c2kv_native" and getattr(args, "candidate_algorithm", None) is None
            and (args.embedding_model is None or not (args.embedding_model / "config.json").is_file())):
        raise ValueError("embedding model must be a local model directory")
    if not (args.checkpoint / "config.json").is_file():
        raise ValueError("checkpoint must be a local model directory")
    if not (args.sglang_root / ".git").exists():
        raise ValueError("--sglang-root must be the PR #5-compatible SGLang checkout")
    _bare_endpoint(args.sglang_backend_url)
    if args.benchmark != "bfcl":
        if not args.user_base_url:
            raise ValueError("tau2/ToolSandbox require --user-base-url for the raw Full user simulator")
        if _bare_endpoint(args.user_base_url) == _controller_endpoint(args):
            raise ValueError("agent C1 endpoint and raw Full user endpoint must differ")
        if not args.portable_root.is_dir():
            raise ValueError("--portable-root must be the bfcl-c2kv checkout")
    return identities


def create_output_directory(path: Path) -> None:
    """Create a new run root and never resume/overwrite implicitly."""
    path.mkdir(parents=True, exist_ok=False)


def preview(args: argparse.Namespace, identities: list[str], profile: dict) -> dict:
    controller_path = (args.out / "controller.json").resolve()
    cells = []
    for task in identities:
        server, worker = commands_for_task(args, task, controller_path)
        cells.append({
            "task_id": task,
            "C1_server_command": server,
            "benchmark_worker_command": worker,
            "agent_endpoint": _controller_endpoint(args),
            "user_endpoint": args.user_base_url if args.benchmark != "bfcl" else None,
        })
    return profile | {
        "benchmark": args.benchmark,
        "source_profile": SOURCE_PROFILE[args.benchmark],
        "task_or_scenario_ids": identities,
        "official_scorer": OFFICIAL_SCORER[args.benchmark],
        "cells": cells,
        "model_calls": 0,
        "benchmark_calls": 0,
        "network_calls": 0,
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        identities = validate_args(args)
        controller, profile = build_profile(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    profile.update(
        benchmark=args.benchmark, source_profile=SOURCE_PROFILE[args.benchmark],
        task_ids=identities, sglang_backend_url=args.sglang_backend_url,
        benchmark_dir=str(_benchmark_dir(args).resolve()),
        user_base_url=args.user_base_url if args.benchmark != "bfcl" else None,
        official_scorer=OFFICIAL_SCORER[args.benchmark],
        source_revisions={
            "c2kv": source_revision(HERE.parents[1]),
            "kvoffload-sglang-c2kv": source_revision(args.sglang_root),
            "bfcl-c2kv": source_revision(args.portable_root),
        },
    )
    if args.preview:
        print(json.dumps(preview(args, identities, profile), indent=2, ensure_ascii=False))
        return 0
    try:
        profile["sglang_backend_preflight"] = preflight_sglang_backend(args)
    except RuntimeError as error:
        parser.error(str(error))
    create_output_directory(args.out)
    controller_path = (args.out / "controller.json").resolve()
    save(controller_path, controller)
    save(args.out / "profile.json", profile)
    receipt = {"schema": "c1-multibench-delivery-run-v1", "status": "running", "tasks": []}
    metrics: list[dict] = []
    try:
        for task in identities:
            print(json.dumps({"benchmark": args.benchmark, "task": task, "status": "running"}), flush=True)
            task_receipt, task_metrics = run_task(args, task, controller_path)
            receipt["tasks"].append(task_receipt)
            metrics.append(task_metrics)
            save(args.out / "result.json", receipt)
            write_unified_summary(args.out, metrics)
        receipt["status"] = "completed"
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.out / "result.json", receipt)
        write_unified_summary(args.out, metrics)
    print(json.dumps({"status": "completed", "result": str(args.out / "result.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
