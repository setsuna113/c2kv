"""Collect exact terminal D128 runtime-failure evidence without model work.

The collector reads one frozen package filesystem.  It recognizes only the
specific CapacityInfeasible contract emitted by the frozen S0 policy; every
other terminal error remains unknown and makes the receipt ineligible for
downstream operational selection.  Outputs are exclusive-create.
"""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping


SCHEMA = "experiment3-d128-runtime-failure-audit-v1"
SUPPORTED_STAGES = {"H0_R1", "H1_R1"}
SUPPORTED_CONTROLLERS = {"C0", "C5"}
CAPACITY_PREFIX = (
    "Native S0 mandatory raw input and minimum whole-event gist "
    "cannot fit the declared limits:"
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSONL objects: {path}")
    return rows


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": _sha(path),
            "size_bytes": path.stat().st_size}


def _counts(values: Iterable[object]) -> dict[str, int]:
    observed = collections.Counter(str(value) for value in values)
    return dict(sorted(observed.items()))


def _one_result(root: Path) -> Path:
    paths = sorted((root / "bfcl/bfcl/result").rglob("*_result.json"))
    if len(paths) != 1:
        raise ValueError(f"Expected one BFCL result under {root}, found {len(paths)}")
    return paths[0]


def _capacity_classification(
        error: Mapping[str, Any], failed: Mapping[str, Any],
        policy_path: Path, exception_path: Path, *,
        eval_capacity_limit: object, design_capacity_limit: object,
        startup_capacity_limit: object, backend_context: object,
        model_context: object, sequence_values: list[int]) -> dict[str, Any]:
    message = error.get("message")
    policy = policy_path.read_text(encoding="utf-8")
    exceptions = exception_path.read_text(encoding="utf-8")
    markers = (set(re.findall(
        r"(?:history_byte_budget|workspace_byte_budget|physical_sequence_budget):[48]",
        message)) if isinstance(message, str) else set())
    checks = {
        "error_type": error.get("type") == "CapacityInfeasible",
        "exact_message_prefix": (
            isinstance(message, str) and message.startswith(CAPACITY_PREFIX)),
        "physical_sequence_budget_ratio4": "physical_sequence_budget:4" in markers,
        "physical_sequence_budget_ratio8": "physical_sequence_budget:8" in markers,
        "failed_step_zero_generation_attempts": failed.get("generation_attempts") == 0,
        "failed_step_zero_generation_completed": failed.get("generation_completed") == 0,
        "failed_step_empty_generation_trace": failed.get("generation_trace") == [],
        "frozen_policy_contains_raise_contract": (
            "Native S0 mandatory raw input and minimum whole-event gist " in policy
            and "raise CapacityInfeasible(" in policy),
        "frozen_exception_kind_matches": (
            'kind = "capacity_infeasible"' in exceptions
            and "class CapacityInfeasible" in exceptions),
        "positive_equal_frozen_capacity_limits": (
            type(eval_capacity_limit) is int and eval_capacity_limit > 0
            and eval_capacity_limit == design_capacity_limit
            and eval_capacity_limit == startup_capacity_limit),
        "candidate_sequences_exceed_only_eval_capacity": (
            bool(sequence_values)
            and type(eval_capacity_limit) is int
            and all(value > eval_capacity_limit for value in sequence_values)),
        "candidate_sequences_fit_backend_context": (
            bool(sequence_values) and type(backend_context) is int
            and all(value <= backend_context for value in sequence_values)),
        "candidate_sequences_fit_model_context": (
            bool(sequence_values) and type(model_context) is int
            and all(value <= model_context for value in sequence_values)),
        "no_model_physical_context_failure": (
            not isinstance(message, str)
            or "model_physical_context" not in message),
    }
    recognized = all(checks.values())
    if recognized:
        return {
            "status": "audited_in_contract",
            "selection_eligible": True,
            "family": "pre_generation_capacity_infeasible",
            "exception_type": "CapacityInfeasible",
            "source_exception": "CapacityInfeasible",
            "source_semantics": (
                "declared method cannot represent this prefix within its budget"),
            "is_sglang_extraction_budget_exhausted": False,
            "is_sglang_http_failure": False,
            "is_endpoint_or_startup_failure": False,
            "budget_contract_verified": True,
            "budget_contract": {
                "eval_capacity_max_sequence_tokens": eval_capacity_limit,
                "design_resolved_max_sequence_tokens": design_capacity_limit,
                "startup_effective_max_sequence_tokens": startup_capacity_limit,
                "backend_context_length": backend_context,
                "model_context": model_context,
                "candidate_sequence_tokens_min": min(sequence_values),
                "candidate_sequence_tokens_max": max(sequence_values),
                "binding_limit": "eval_capacity_max_sequence_tokens",
            },
            "checks": checks,
        }
    return {
        "status": "unknown_unclassified",
        "selection_eligible": False,
        "family": "unknown_runtime_failure",
        "exception_type": error.get("type"),
        "budget_contract_verified": False,
        "checks": checks,
        "reason": "Terminal runtime error does not match the exact audited capacity contract",
    }


def _frozen_evaluator_path(package: Path, shard: Path,
                           source_provenance_kind: str) -> Path:
    if source_provenance_kind == "remainder_source_binding":
        return package / "evidence_eval.py"
    if source_provenance_kind == "shard_provenance":
        return shard / "evidence_eval.py"
    raise ValueError(f"Unsupported source provenance kind: {source_provenance_kind}")


def _failure(package: Path, shard: Path, lane: Path,
             stage_path: Path, stage_document: Mapping[str, Any],
             outcome: Mapping[str, Any], *, stage_name: str,
             controller: str, source_provenance_path: Path,
             source_provenance_kind: str, source_design_path: Path,
             package_static: Mapping[str, Any],
             package_contract: Mapping[str, Any]) -> dict[str, Any]:
    task_id = outcome.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"Malformed failure outcome in {stage_path}")
    root = lane / "results/task_shards" / task_id
    paths = {
        "steps": root / "server/steps.jsonl",
        "attempts": root / "server/attempts.jsonl",
        "server_final": root / "server/final.json",
        "server_startup": root / "server/startup.json",
        "sglang_http": root / "server/sglang_http.jsonl",
        "server_log": root / "server.log",
        "server_supervisor": root / "server.supervisor.json",
        "bfcl_final": root / "bfcl/final.json",
        "official_summary": root / "bfcl/official_summary.json",
        "bfcl_worker_log": root / "bfcl/worker.log",
        "bfcl_result": _one_result(root),
        "package_static_files": Path(package_static["path"]),
        "package_contract": Path(package_contract["path"]),
        "source_provenance": source_provenance_path,
        "evaluated_design": lane / "design.json",
        "source_design": source_design_path,
        "frozen_eval_capacity": lane / "runtime/configs/eval_capacity.json",
        "frozen_shard_evaluator": _frozen_evaluator_path(
            package, shard, source_provenance_kind),
        "engine_log": lane / "run/engine.log",
        "engine_status": lane / "run/status.json",
        "frozen_capacity_policy_source": (
            lane / "runtime/benchmarks/memory_runtime/event_native_s0_policy.py"),
        "frozen_capacity_exception_source": (
            lane / "runtime/benchmarks/memory_runtime/always_compress.py"),
    }
    steps = _rows(paths["steps"])
    error_rows = [row for row in steps if isinstance(row.get("error"), dict)]
    if len(error_rows) != 1:
        raise ValueError(
            f"{shard.name}/{task_id}: expected one terminal error row, found {len(error_rows)}")
    failed = error_rows[0]
    error = failed["error"]
    traces = [trace for row in steps for trace in row.get("generation_trace", [])]
    http = _rows(paths["sglang_http"])
    responses = [row for row in http if row.get("event") == "response"]
    extractions = [row["response"]["extraction"] for row in responses
                   if isinstance(row.get("response"), dict)
                   and isinstance(row["response"].get("extraction"), dict)]
    server_final = _read(paths["server_final"])
    startup = _read(paths["server_startup"])
    bfcl_final = _read(paths["bfcl_final"])
    official = _read(paths["official_summary"])
    supervisor = _read(paths["server_supervisor"])
    message = error.get("message", "")
    design = _read(paths["evaluated_design"])
    provenance = _read(paths["source_provenance"])
    eval_capacity = _read(paths["frozen_eval_capacity"])
    engine_log = paths["engine_log"].read_text(encoding="utf-8", errors="replace")
    backend_matches = [int(value) for value in re.findall(
        r"(?:context_length|context_len)=(\d+)", engine_log)]
    if not backend_matches or len(set(backend_matches)) != 1:
        backend_context = None
    else:
        backend_context = backend_matches[0]
    top_numbers = {}
    for key in ("history_budget_bytes", "workspace_budget_bytes", "model_context"):
        match = re.search(r"'" + re.escape(key) + r"': (\d+)", message)
        top_numbers[key] = int(match.group(1)) if match else None
    sequence_values = [int(value) for value in re.findall(
        r"'sequence_tokens': (\d+)", message)]
    logical_values = [int(value) for value in re.findall(
        r"'logical_sequence_tokens': (\d+)", message)]
    design_capacity_limit = design.get("resolved_configs", {}).get(
        "eval_capacity", {}).get("capacity", {}).get("max_sequence_tokens")
    eval_capacity_limit = eval_capacity.get("capacity", {}).get(
        "max_sequence_tokens")
    startup_capacity_limit = startup.get("runtime_packing_contract", {}).get(
        "effective_packing", {}).get("max_sequence_tokens")
    classification = _capacity_classification(
        error, failed, paths["frozen_capacity_policy_source"],
        paths["frozen_capacity_exception_source"],
        eval_capacity_limit=eval_capacity_limit,
        design_capacity_limit=design_capacity_limit,
        startup_capacity_limit=startup_capacity_limit,
        backend_context=backend_context,
        model_context=top_numbers["model_context"],
        sequence_values=sequence_values)
    algorithm_id = stage_document.get("candidate_id")
    expected_history = stage_name.split("_", 1)[0]
    if (
        controller not in SUPPORTED_CONTROLLERS
        or not isinstance(algorithm_id, str) or not algorithm_id
        or design.get("candidate_id") != algorithm_id
        or design.get("search_contract", {}).get("history") != expected_history
        or (
            source_provenance_kind == "shard_provenance"
            and provenance.get("controller") != controller
        )
        or (
            source_provenance_kind == "shard_provenance"
            and provenance.get("output_algorithm_id", algorithm_id) != algorithm_id
        )
        or (
            source_provenance_kind == "remainder_source_binding"
            and provenance.get("continuation_lane") != lane.name
        )
    ):
        raise ValueError(
            f"{shard.name}/{task_id}: shard provenance/design/stage identity differs")
    return {
        "stage": stage_name,
        "shard": shard.name,
        "controller": controller,
        "task_id": task_id,
        "source_algorithm_id": algorithm_id,
        "source_binding": {
            "stage": stage_name,
            "controller": controller,
            "task_id": task_id,
            "source_algorithm_id": algorithm_id,
            "package_root": str(package),
            "source_package_static_sha256": package_static["sha256"],
            "source_package_contract_sha256": package_contract["sha256"],
            "shard": shard.name,
            "lane": lane.name,
            "source_provenance_kind": source_provenance_kind,
            "source_provenance_sha256": _sha(paths["source_provenance"]),
            "evaluated_design_sha256": _sha(paths["evaluated_design"]),
            "server_startup_sha256": _sha(paths["server_startup"]),
            "frozen_eval_capacity_sha256": _sha(paths["frozen_eval_capacity"]),
        },
        "quality_status": "runtime_failed",
        "stage_outcome": dict(outcome),
        "classification": classification,
        "failure": {
            "decision_key": failed.get("decision_key"),
            "step_status": failed.get("status"),
            "error": error,
            "error_object_sha256": _json_sha(error),
            "failed_step_generation_attempts": failed.get("generation_attempts"),
            "failed_step_generation_completed": failed.get("generation_completed"),
            "failed_step_generation_trace_count": len(
                failed.get("generation_trace") or []),
            "failed_step_generation_usage_known": failed.get(
                "generation_usage_known"),
        },
        "actual_execution_before_failure": {
            "step_rows": len(steps),
            "step_status_counts": _counts(row.get("status") for row in steps),
            "completed_generation_traces": sum(
                trace.get("status") == "completed" for trace in traces),
            "generation_phase_counts": _counts(
                trace.get("phase") for trace in traces),
            "generation_status_counts": _counts(
                trace.get("status") for trace in traces),
            "attempt_journal_generation_attempts": server_final.get(
                "journal_summary", {}).get("by_kind", {}).get("generation", {}),
            "sglang_http_response_count": len(responses),
            "sglang_http_status_counts": _counts(
                row.get("http_status") for row in responses),
            "successful_response_extraction_model_calls": sum(
                row.get("model_calls", 0) for row in extractions),
            "successful_response_extraction_reported_max_values": sorted(set(
                row["max_extraction_calls"] for row in extractions
                if isinstance(row.get("max_extraction_calls"), int))),
            "all_observed_sglang_responses_http_200": bool(responses) and all(
                row.get("http_status") == 200 for row in responses),
        },
        "declared_budget_and_failure_measurement": {
            "startup": {key: startup.get(key) for key in (
                "max_decisions", "max_generation_calls", "max_new_tokens",
                "max_wall_seconds", "ratio")},
            "failure_history_budget_bytes": top_numbers["history_budget_bytes"],
            "failure_workspace_budget_bytes": top_numbers["workspace_budget_bytes"],
            "failure_model_context": top_numbers["model_context"],
            "frozen_eval_capacity_max_sequence_tokens": eval_capacity_limit,
            "design_resolved_max_sequence_tokens": design_capacity_limit,
            "startup_effective_max_sequence_tokens": startup_capacity_limit,
            "engine_backend_context_length": backend_context,
            "failure_candidate_sequence_tokens_min": (
                min(sequence_values) if sequence_values else None),
            "failure_candidate_sequence_tokens_max": (
                max(sequence_values) if sequence_values else None),
            "failure_candidate_logical_sequence_tokens_min": (
                min(logical_values) if logical_values else None),
            "failure_candidate_logical_sequence_tokens_max": (
                max(logical_values) if logical_values else None),
            "failure_reason_markers": sorted(set(re.findall(
                r"(?:history_byte_budget|workspace_byte_budget|physical_sequence_budget):[48]",
                message))),
        },
        "runtime_vs_official": {
            "runtime_completed": outcome.get("runtime_completed"),
            "server_returncode": outcome.get("server_returncode"),
            "server_final_status": server_final.get("status"),
            "server_stop_reason": server_final.get("stop_reason"),
            "server_supervisor_child_returncode": supervisor.get("child_returncode"),
            "bfcl_worker_returncode": bfcl_final.get("worker_returncode"),
            "bfcl_final_status": bfcl_final.get("status"),
            "official_summary_exists": True,
            "official": {key: official.get(key) for key in (
                "scored", "n", "n_total", "n_scored", "n_generated",
                "correct_count", "semantic_score")},
            "official_zero_imputed": False,
            "official_score_does_not_convert_runtime_failure_to_completed": True,
        },
        "evidence": {"stage_manifest": _binding(stage_path),
                     **{key: _binding(path) for key, path in paths.items()}},
    }


def _work_items(package: Path, contract: Mapping[str, Any], stage_name: str
                ) -> list[dict[str, Any]]:
    """Resolve exact native-shard or continuation-lane package identities."""
    rows = contract.get("shards")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Package contract has no exact shard rows")
    by_id = {row.get("shard_id"): row for row in rows}
    if len(by_id) != len(rows) or None in by_id:
        raise ValueError("Package contract shard IDs are missing or duplicated")
    result: list[dict[str, Any]] = []
    if (package / "shards").is_dir():
        expected_schemas = {
            "H0_R1": "experiment3-d128-expansion-package-v1",
            "H1_R1": "experiment3-h1-r3-package-v1",
        }
        if contract.get("schema") != expected_schemas[stage_name]:
            raise ValueError("Native shard package contract schema/stage differs")
        discovered = sorted(path for path in (package / "shards").iterdir()
                            if path.is_dir())
        if {path.name for path in discovered} != set(by_id):
            raise ValueError("Package contract and native shard directories differ")
        for shard in discovered:
            row = by_id[shard.name]
            if row.get("controller") not in SUPPORTED_CONTROLLERS:
                raise ValueError(f"Unsupported controller in contract: {shard.name}")
            if stage_name == "H1_R1" and row.get("stage") != stage_name:
                raise ValueError(f"H1 shard stage differs: {shard.name}")
            if (row.get("relative_package") != shard.name
                    or row.get("static_files_sha256") != _sha(
                        shard / "static_files.json")):
                raise ValueError(f"Contract shard identity differs: {shard.name}")
            lane = shard / "lanes" / shard.name
            result.append({
                "shard": shard,
                "lane": lane,
                "controller": row["controller"],
                "source_provenance_path": shard / "provenance.json",
                "source_provenance_kind": "shard_provenance",
                "source_design_path": shard / "source_design.json",
            })
        return result
    if not (package / "lanes").is_dir():
        raise FileNotFoundError("Package has neither shards/ nor lanes/")
    if contract.get("schema") != "experiment3-expansion-remainder-v1":
        raise ValueError("Direct-lane package is not an exact remainder contract")
    discovered = sorted(path for path in (package / "lanes").iterdir()
                        if path.is_dir())
    if {path.name for path in discovered} != set(by_id):
        raise ValueError("Remainder contract and continuation lane directories differ")
    for lane in discovered:
        row = by_id[lane.name]
        controller = row.get("controller")
        if controller not in SUPPORTED_CONTROLLERS:
            raise ValueError(f"Unsupported controller in remainder contract: {lane.name}")
        result.append({
            "shard": lane,
            "lane": lane,
            "controller": controller,
            "source_provenance_path": lane / "remainder_source_binding.json",
            "source_provenance_kind": "remainder_source_binding",
            "source_design_path": lane / "source_design.json",
        })
    return result


def collect(package_root: Path, expected_static_sha256: str, *, stage: str,
            package_contract_path: Path) -> dict[str, Any]:
    package = package_root.resolve()
    if stage not in SUPPORTED_STAGES:
        raise ValueError(f"Unsupported D128 stage: {stage}")
    static_path = package / "static_files.json"
    if (not re.fullmatch(r"[0-9a-f]{64}", expected_static_sha256)
            or not static_path.is_file()
            or _sha(static_path) != expected_static_sha256):
        raise ValueError("Frozen package static_files.json differs from expected SHA256")
    contract_path = package_contract_path.resolve()
    try:
        contract_path.relative_to(package)
    except ValueError as error:
        raise ValueError("Package contract must be inside the frozen package") from error
    package_static = _binding(static_path)
    package_contract = _binding(contract_path)
    contract = _read(contract_path)
    failures = []
    snapshot = []
    for item in _work_items(package, contract, stage):
        shard = item["shard"]
        lane = item["lane"]
        stage_path = lane / "results/stage_manifest.json"
        if not stage_path.is_file():
            snapshot.append({"shard": shard.name, "stage_manifest": None})
            continue
        stage_document = _read(stage_path)
        outcomes = stage_document.get("task_outcomes")
        if not isinstance(outcomes, list):
            raise ValueError(f"Malformed task_outcomes: {stage_path}")
        terminal = [row for row in outcomes if isinstance(row, dict)
                    and row.get("outcome") == "runtime_failure_in_denominator"]
        snapshot.append({
            "shard": shard.name,
            "status": stage_document.get("status"),
            "state": stage_document.get("state"),
            "completed_task_cells": stage_document.get("completed_task_cells"),
            "task_cells_started": stage_document.get("task_cells_started"),
            "outcome_counts": _counts(
                row.get("outcome") for row in outcomes if isinstance(row, dict)),
            "runtime_failure_task_ids": [row.get("task_id") for row in terminal],
            "stage_manifest": _binding(stage_path),
        })
        for outcome in terminal:
            if (outcome.get("runtime_completed") is not False
                    or outcome.get("server_returncode") in (None, 0)):
                raise ValueError(
                    f"{shard.name}/{outcome.get('task_id')}: not an actual terminal runtime failure")
            failures.append(_failure(
                package, shard, lane, stage_path, stage_document, outcome,
                stage_name=stage, controller=item["controller"],
                source_provenance_path=item["source_provenance_path"],
                source_provenance_kind=item["source_provenance_kind"],
                source_design_path=item["source_design_path"],
                package_static=package_static, package_contract=package_contract))
    unknown = [row for row in failures
               if row["classification"]["status"] != "audited_in_contract"]
    return {
        "schema": SCHEMA,
        "status": "audited" if not unknown else "contains_unknown_failure",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_mode": (
            "read-only native artifacts; no model calls, reruns, launches, or package mutations"),
        "package": {
            "path": str(package),
            "stage": stage,
            "static_files": package_static,
            "contract": package_contract,
        },
        "scope": {
            "fixed_d128_runtime_outcomes": True,
            "automatic_retry_or_rerun": False,
            "official_scores_preserved": True,
            "official_zero_used_as_clean_runtime_completion": False,
            "observed_failure_count": len(failures),
            "audited_capacity_failure_count": len(failures) - len(unknown),
            "unknown_failure_count": len(unknown),
            "selection_eligible": not unknown,
        },
        "classification_summary": {
            "failure_count": len(failures),
            "families": _counts(
                row["classification"]["family"] for row in failures),
            "all_fail_at_zero_generation_on_failed_step": all(
                row["failure"]["failed_step_generation_completed"] == 0
                for row in failures),
            "all_have_official_summary_but_remain_runtime_failed": all(
                row["runtime_vs_official"]["official_summary_exists"]
                and not row["runtime_vs_official"]["runtime_completed"]
                for row in failures),
        },
        "failures": failures,
        "all_shards_snapshot": snapshot,
        "snapshot_interpretation": (
            "Only failures present in native stage manifests at verified_at_utc are asserted; "
            "running shards may change later."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(SUPPORTED_STAGES), required=True)
    parser.add_argument("--package-contract", type=Path, required=True)
    parser.add_argument("--expected-static-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if str(args.output) == "-":
        receipt = collect(
            args.package_root, args.expected_static_sha256, stage=args.stage,
            package_contract_path=args.package_contract)
        json.dump(receipt, sys.stdout, ensure_ascii=False, indent=2,
                  sort_keys=True, allow_nan=False)
        sys.stdout.write("\n")
        return 0
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    receipt = collect(
        args.package_root, args.expected_static_sha256, stage=args.stage,
        package_contract_path=args.package_contract)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": receipt["status"], "output": str(output),
                      "sha256": _sha(output),
                      "observed_failure_count": len(receipt["failures"])},
                     ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
