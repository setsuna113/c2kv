"""Read-only D20 promotion and D128 continuation planning for Experiment 3.

This command never starts model work. It preserves unresolved runtime outcomes
and reuses the twenty fixed-manifest evidence cells of a promoted, unchanged method.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import evidence_eval_summary as summary
from evidence_eval import task_groups


LANES = ("C0", "C1", "C2", "C3", "C4_turn", "C4_task", "C5")
TRAINED = ("C1", "C4_turn", "C4_task")
REMOTE = "/home/liuyancheng/c2kv-evidence-sets-20260916"
FAILURE_AUDIT_SCHEMA = "experiment3-d20-operational-failure-audit-v1"
OPERATIONAL_SELECTION_SCOPE = "fixed_d20_operational_success_v1"
EXTRACTION_BUDGET_PREFIX = "C2KV_EXTRACTION_BUDGET_EXHAUSTED:"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def manifest_ids(document: dict, count: int, name: str) -> list[str]:
    ids = document.get("task_ids")
    if not isinstance(ids, list) or len(ids) != count or len(set(ids)) != count:
        raise ValueError(f"{name} must contain {count} unique tasks")
    task_groups(ids)
    return ids


def validate_manifests(d20: dict, d128: dict, f128: dict, training: dict) -> list[str]:
    small = manifest_ids(d20, 20, "D20")
    full = manifest_ids(d128, 128, "D128")
    heldout = manifest_ids(f128, 128, "F128")
    train = training.get("task_ids", [])
    if not train or len(set(train)) != len(train):
        raise ValueError("Training manifest must enumerate unique tasks")
    if not set(small) <= set(full):
        raise ValueError("D20 is not a subset of frozen D128")
    if set(full) & set(heldout):
        raise ValueError("D128 and F128 share exact tasks")
    if task_groups(train) & (task_groups(full) | task_groups(heldout)):
        raise ValueError("Training overlaps evaluation task groups")
    return [task for task in full if task not in set(small)]


def collect_trained(host: str) -> dict:
    specs = [{"id": lane, "remote_root": f"{REMOTE}/eval_trained_v4/{lane}"}
             for lane in TRAINED]
    # Follow the verified scheduling receipt instead of reviving the old NPU0 waiter.
    prefix = "SOURCE_SPECS = " + repr(specs) + "\n"
    prefix += f"""
import json
from pathlib import Path
receipt = Path({REMOTE!r}) / 'post_t02_v4/c4_task_npu2.supersession.json'
if receipt.exists():
    replacement = json.loads(receipt.read_text())['replacement']['package']
    if Path(replacement).resolve().parent != (Path({REMOTE!r}) / 'eval_trained_v4').resolve():
        raise ValueError('C4 task replacement escapes its evaluation directory')
    SOURCE_SPECS[-1]['remote_root'] = replacement
parallel_receipt = Path({REMOTE!r}) / 'post_t02_v5/scheduled.json'
if parallel_receipt.exists():
    schedule = json.loads(parallel_receipt.read_text())
    if schedule.get('schema') != 'evidence-post-t02-scheduled-v5':
        raise ValueError('Unexpected parallel schedule schema')
    owners = {{row['name']: row for row in schedule['children']}}
    if set(owners) != {{'C1', 'C4_turn', 'C4_task'}}:
        raise ValueError('Parallel schedule lacks exact trained lanes')
    SOURCE_SPECS = []
    for name in ('C1', 'C4_turn', 'C4_task'):
        root = Path(owners[name]['root']).resolve()
        if root.parent != (Path({REMOTE!r}) / 'eval_trained_v5').resolve():
            raise ValueError('Parallel schedule escapes its frozen directory')
        SOURCE_SPECS.extend({{'id': name + '.part' + str(index),
                             'remote_root': str(root / ('shard' + str(index)))}}
                            for index in range(2))
remaining_root = Path({REMOTE!r}) / 'eval_trained_remaining_v1/C4_turn'
remaining_receipt = remaining_root / 'continuation.json'
if remaining_receipt.exists():
    continuation = json.loads(remaining_receipt.read_text())
    rows = continuation.get('rows', [])
    if (continuation.get('schema') != 'evidence-sets-trained-d20-remaining-v1'
            or continuation.get('lane') != 'C4_turn' or len(rows) != 6
            or continuation.get('total_task_execution_budget') != 6
            or continuation.get('automatic_retries') != 0
            or continuation.get('automatic_reruns') != 0):
        raise ValueError('Unexpected never-started continuation receipt')
    for index, row in enumerate(rows):
        identity = 'C4_turn.remaining' + str(index)
        root = Path(row['root']).resolve()
        if (row.get('id') != identity or row.get('lane') != 'C4_turn'
                or root != (remaining_root / identity).resolve()):
            raise ValueError('Never-started continuation escapes its frozen directory')
        SOURCE_SPECS.append({{'id': identity, 'remote_root': str(root)}})
"""
    code = summary.REMOTE_COLLECTOR.replace(
        'LANES = ("C0", "C2", "C3", "C5")', "LANES = " + repr(TRAINED))
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                             host, "python3", "-"],
                            input=prefix + code, text=True, encoding="utf-8",
                            capture_output=True, check=True)
    return json.loads(result.stdout)


def _overlay_never_started(snapshot: dict, lane: str, cells: list[dict]) -> list[dict]:
    sources = snapshot.get("sources", {})
    continuations = [(key, value) for key, value in sources.items()
                     if key.startswith(lane + ".remaining")]
    if not continuations:
        return cells
    eligible = {}
    for part in range(2):
        owner = f"{lane}.part{part}"
        parent = sources.get(owner, {}).get("lanes", {}).get(lane, {})
        if parent.get("status", {}).get("state") not in ("failed_no_rerun", "completed"):
            continue
        for task, observation in parent.get("task_observations", {}).items():
            if (observation.get("stage_outcome") or {}).get("outcome") == "not_started":
                eligible[task] = owner
    by_task = {cell["task_id"]: dict(cell) for cell in cells}
    continued = set()
    for owner, source in continuations:
        row = source.get("lanes", {}).get(lane)
        tasks = row.get("declared_tasks") if isinstance(row, dict) else None
        if not isinstance(tasks, list) or len(tasks) != 1:
            raise ValueError("Continuation source must declare exactly one original task")
        task = tasks[0]
        if task in continued or task not in eligible or by_task[task]["status"] != "pending":
            raise ValueError("Continuation must cover distinct explicitly never-started tasks only")
        observation = row["task_observations"][task]
        state, reason = summary._task_state(owner, observation)
        by_task[task] = {"task_id": task, "status": state, "status_reason": reason,
                         "quality_source_id": owner,
                         "original_never_started_source_id": eligible[task],
                         "official": observation.get("official_summary") if state == "completed" else None,
                         "measured_cost": summary._cost_fields(observation)}
        continued.add(task)
    if continued != set(eligible):
        raise ValueError("Continuation does not cover the exact terminal parent's never-started remainder")
    return [by_task[cell["task_id"]] for cell in cells]


def trained_cells(snapshot: dict, lane: str, canonical: list[str]) -> list[dict]:
    sources = snapshot.get("sources", {})
    parallel_keys = [f"{lane}.part{index}" for index in range(2)]
    if any(key in sources for key in parallel_keys):
        cells = []
        for index, key in enumerate(parallel_keys):
            expected = canonical[index * 10:(index + 1) * 10]
            source = sources.get(key, {}).get("lanes", {}).get(lane)
            if source is None:
                cells.extend({"task_id": task, "status": "pending", "official": None,
                              "quality_source_id": key,
                              "status_reason": "trained_shard_not_prepared"} for task in expected)
                continue
            if source.get("declared_tasks") != expected:
                raise ValueError(f"{key}: trained shard denominator/order differs from D20")
            for task in expected:
                observation = source["task_observations"][task]
                state, reason = summary._task_state(key, observation)
                cells.append({"task_id": task, "status": state, "status_reason": reason,
                              "quality_source_id": key,
                              "official": observation.get("official_summary") if state == "completed" else None,
                              "measured_cost": summary._cost_fields(observation)})
        return _overlay_never_started(snapshot, lane, cells)
    source = snapshot.get("sources", {}).get(lane, {}).get("lanes", {}).get(lane)
    if source is None:
        return [{"task_id": task, "status": "pending", "official": None,
                 "status_reason": "trained_evaluation_not_prepared"} for task in canonical]
    if source.get("declared_tasks") != canonical:
        raise ValueError(f"{lane}: trained evaluation denominator/order differs from D20")
    cells = []
    for task in canonical:
        observation = source["task_observations"][task]
        state, reason = summary._task_state("trained_eval_v4", observation)
        cells.append({"task_id": task, "status": state, "status_reason": reason,
                      "quality_source_id": lane,
                      "official": observation.get("official_summary") if state == "completed" else None,
                      "measured_cost": summary._cost_fields(observation)})
    return cells


def summarize_lane(cells: list[dict], canonical: list[str]) -> dict:
    if [cell["task_id"] for cell in cells] != canonical:
        raise ValueError("Lane cells must cover the exact ordered D20")
    statuses = {"completed", "runtime_failed", "pending"}
    if any(cell.get("status") not in statuses for cell in cells):
        raise ValueError("Unrecognized task cell status")
    completed = [cell for cell in cells if cell["status"] == "completed"]
    for cell in completed:
        if not summary._official_valid(cell.get("official")):
            raise ValueError("Completed cell has no valid official score")
        if type(cell["official"]["correct_count"]) is not int:
            raise ValueError("Official success must be an integer, not boolean")
    success = [cell["task_id"] for cell in completed if cell["official"]["correct_count"] == 1]
    failures = [cell["task_id"] for cell in cells if cell["status"] == "runtime_failed"]
    pending = [cell["task_id"] for cell in cells if cell["status"] == "pending"]
    return {"completed": len(completed), "runtime_failed": failures, "pending": pending,
            "success_ids": success, "success_count": len(success),
            "success_count_bounds": [len(success), len(success) + len(failures) + len(pending)],
            "success_count_bounds_scope": (
                "descriptive counterfactual if unobserved cells had completed; not a promotion gate"),
            "clean_d20": len(completed) == 20, "all_cells_terminal": not pending,
            "operational_selection_scope": OPERATIONAL_SELECTION_SCOPE,
            "operational_denominator": 20,
            "operational_success_ids": success,
            "operational_success_count": len(success),
            "audited_runtime_failures": [],
            "unaudited_runtime_failures": failures,
            "operational_selection_eligible": not pending and not failures,
            "quality_cells": cells}


def _bound_failure_source(entry: dict) -> dict:
    evidence = entry.get("source_evidence")
    if (not isinstance(evidence, dict) or not isinstance(evidence.get("path"), str)
            or not _valid_sha256(evidence.get("sha256"))):
        raise ValueError("Failure audit source evidence must bind a path and SHA256")
    path = Path(evidence["path"])
    if not path.is_file() or digest(path) != evidence["sha256"]:
        raise ValueError("Failure audit source evidence hash differs")
    document = read(path)
    stage = entry.get("stage_manifest")
    steps = entry.get("steps")
    if (not isinstance(stage, dict) or not isinstance(stage.get("path"), str)
            or not _valid_sha256(stage.get("sha256"))
            or not isinstance(steps, dict) or not isinstance(steps.get("path"), str)
            or not _valid_sha256(steps.get("sha256"))):
        raise ValueError("Failure audit must bind native stage and steps hashes")
    return document


def _validate_historical_budget_failure(entry: dict, document: dict) -> None:
    failure = document.get("failure", {})
    artifacts = document.get("remote_artifact_sha256", {})
    extraction = document.get("extraction_budget", {})
    expected_stage = str(Path(document.get("remote_root", ""))
                         / "lanes/C2/results/stage_manifest.json").replace("\\", "/")
    expected_steps = str(Path(document.get("remote_root", ""))
                         / "lanes/C2/results/task_shards"
                         / entry["task_id"] / "server/steps.jsonl").replace("\\", "/")
    if (
        document.get("schema") != "evidence-sets-c2-extraction-budget-failure-v1"
        or document.get("lane") != entry["lane"]
        or failure.get("task_id") != entry["task_id"]
        or failure.get("stage_outcome") != "runtime_failure_in_denominator"
        or failure.get("runtime_completed") is not False
        or failure.get("official_zero_accepted_for_quality") is not False
        or failure.get("worker_returncode") != 0
        or failure.get("server_returncode") != 1
        or not isinstance(failure.get("engine_error"), str)
        or not failure["engine_error"].startswith(EXTRACTION_BUDGET_PREFIX)
        or extraction.get("physical_extractions_after_failed_attempt")
            != extraction.get("configured_task_cap")
        or entry.get("classification")
            != "historical_validated_extraction_budget_deployment_failure"
        or entry["stage_manifest"].get("path") != expected_stage
        or entry["stage_manifest"].get("sha256") != artifacts.get("stage_manifest.json")
        or entry["steps"].get("path") != expected_steps
        or entry["steps"].get("sha256") != artifacts.get("server/steps.jsonl")
        or entry["steps"].get("final_status") != "failed"
        or entry["steps"].get("decision_key") != failure.get("decision_key")
        or entry["steps"].get("error") != failure.get("step_error")
    ):
        raise ValueError("Historical extraction-budget failure contract differs")


def _validate_typed_budget_failure(entry: dict, document: dict) -> None:
    if (
        document.get("schema") != "experiment3-trained-d20-runtime-failure-v1"
        or document.get("controller") != entry["lane"]
        or document.get("task_id") != entry["task_id"]
        or document.get("status") != "runtime_failed_not_quality_zero"
        or document.get("source_shard_state") != "failed_no_rerun"
        or document.get("model_artifact_unchanged") is not True
        or entry.get("classification")
            != "typed_extraction_budget_deployment_failure"
        or entry["stage_manifest"] != document.get("stage_manifest")
        or entry["steps"].get("path") != document.get("steps", {}).get("path")
        or entry["steps"].get("sha256") != document.get("steps", {}).get("sha256")
        or entry["steps"].get("final_status") != "failed"
        or entry["steps"].get("decision_key") != document.get("decision_key")
        or entry["steps"].get("error") != document.get("error")
        or entry["steps"]["error"].get("type") != "SGLangExtractionBudgetExhausted"
        or not entry["steps"]["error"].get("message", "").startswith(
            EXTRACTION_BUDGET_PREFIX)
    ):
        raise ValueError("Typed extraction-budget failure contract differs")


def validate_operational_failure_audit(audit: dict | None, rows: dict) -> dict:
    """Admit only individually evidenced deployment failures as non-successes."""
    failed = {(lane, task): cell for lane, row in rows.items()
              for task in row["runtime_failed"]
              for cell in row["quality_cells"] if cell["task_id"] == task}
    entries = [] if audit is None else audit.get("entries")
    if audit is not None and (
            audit.get("schema") != FAILURE_AUDIT_SCHEMA
            or audit.get("status") != "audited"
            or audit.get("selection_scope", {}).get("id") != OPERATIONAL_SELECTION_SCOPE
            or audit["selection_scope"].get("fixed_denominator") != 20
            or audit["selection_scope"].get("counterfactual_bounds_used_for_promotion") is not False
            or audit["selection_scope"].get("automatic_retries") != 0
            or audit["selection_scope"].get("automatic_reruns") != 0
            or not isinstance(entries, list)):
        raise ValueError("Unsupported operational failure audit")
    admitted = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Failure audit entries must be objects")
        key = (entry.get("lane"), entry.get("task_id"))
        if key in admitted or key not in failed:
            raise ValueError("Failure audit entry does not bind one current runtime failure")
        cell = failed[key]
        if (entry.get("quality_source_id") != cell.get("quality_source_id")
                or cell.get("official") is not None
                or cell.get("status_reason") != "runtime_failure_in_denominator"
                or entry.get("selection_outcome") != "operational_non_success"
                or entry.get("official_score_status") != "unobserved_runtime_failure"
                or entry.get("official_zero_imputed") is not False
                or entry.get("retry_or_rerun") is not False):
            raise ValueError("Failure audit owner or outcome contract differs")
        document = _bound_failure_source(entry)
        if document.get("schema") == "evidence-sets-c2-extraction-budget-failure-v1":
            _validate_historical_budget_failure(entry, document)
        elif document.get("schema") == "experiment3-trained-d20-runtime-failure-v1":
            _validate_typed_budget_failure(entry, document)
        else:
            raise ValueError("Failure audit source is not an approved documented contract")
        admitted[key] = entry
        cell["operational_selection"] = {
            "scope": OPERATIONAL_SELECTION_SCOPE,
            "outcome": "non_success",
            "official_score_status": "unobserved_runtime_failure",
            "audit_source_sha256": entry["source_evidence"]["sha256"],
            "classification": entry["classification"],
        }
    for lane, row in rows.items():
        admitted_tasks = [task for task in row["runtime_failed"] if (lane, task) in admitted]
        unaudited = [task for task in row["runtime_failed"] if (lane, task) not in admitted]
        row.update(
            audited_runtime_failures=admitted_tasks,
            unaudited_runtime_failures=unaudited,
            operational_selection_eligible=not row["pending"] and not unaudited,
        )
    return {
        "schema": FAILURE_AUDIT_SCHEMA,
        "status": "audited" if audit is not None else "not_provided",
        "selection_scope": OPERATIONAL_SELECTION_SCOPE,
        "success_definition": "runtime completed with valid official correct_count equal to 1",
        "audited_deployment_failure_treatment": (
            "operational non-success with official score unobserved"),
        "counterfactual_bounds_used_for_promotion": False,
        "admitted": [{"lane": lane, "task_id": task,
                      "quality_source_id": failed[(lane, task)].get("quality_source_id"),
                      "classification": entry["classification"]}
                     for (lane, task), entry in admitted.items()],
        "unaudited_runtime_failures": [
            {"lane": lane, "task_id": task,
             "quality_source_id": cell.get("quality_source_id")}
            for (lane, task), cell in failed.items() if (lane, task) not in admitted],
    }


def _cost_measurement(name: str, value: Any, lane: dict) -> dict | None:
    if not isinstance(value, dict):
        return None
    covered = value.get("covered_tasks")
    extra = value.get("extra_generations")
    latency = value.get("selector_seconds")
    evidence = value.get("evidence")
    if (type(covered) is not int or covered <= 0 or covered > 20
            or type(extra) is not int or extra < 0):
        return None
    if latency is not None and (isinstance(latency, bool)
            or not isinstance(latency, (int, float)) or not math.isfinite(latency) or latency < 0):
        return None
    if (not isinstance(evidence, list) or len(evidence) != covered
            or len({row.get("task_id") for row in evidence if isinstance(row, dict)}) != covered):
        return None
    by_task = {cell["task_id"]: cell for cell in lane["quality_cells"]}
    for row in evidence:
        task = row.get("task_id") if isinstance(row, dict) else None
        cell = by_task.get(task)
        if (cell is None or cell.get("status") != "completed"
                or row.get("owner") != cell.get("quality_source_id")
                or row.get("expected_correct_count") != cell["official"]["correct_count"]
                or not _valid_sha256(row.get("steps_sha256"))
                or not _valid_sha256(row.get("official_sha256"))):
            raise ValueError(f"{name}: tie cost evidence differs from current quality owner")
    missing = [task for task in by_task if task not in {row["task_id"] for row in evidence}]
    if covered == 20:
        if missing:
            raise ValueError(f"{name}: full tie cost evidence does not cover D20")
        kind = "full_exact"
    else:
        if set(missing) != set(lane.get("audited_runtime_failures", [])):
            return None
        kind = "partial_lower_bound"
    evidence_sha256 = hashlib.sha256(json.dumps(
        evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    return {"controller": name, "coverage": covered, "measurement": kind,
            "observed_extra_generations": extra,
            "extra_generations_role": "exact" if kind == "full_exact" else "lower_bound",
            "selector_seconds": latency,
            "selector_seconds_admissible_for_exact_extra_tie": kind == "full_exact",
            "missing_audited_runtime_failures": missing,
            "evidence_rows_sha256": evidence_sha256}


def _best_cost_method(group: list[str], costs: dict,
                      lanes: dict) -> tuple[str | None, str | None, dict]:
    measurements = {name: _cost_measurement(name, costs.get(name), lanes[name])
                    for name in group}
    proof = {"criterion": "extra_generations_then_selector_seconds",
             "candidates": [value for value in measurements.values() if value is not None]}
    if any(value is None for value in measurements.values()):
        return None, "waiting_for_measured_tie_cost", proof
    full = [value for value in measurements.values()
            if value["measurement"] == "full_exact"]
    partial = [value for value in measurements.values()
               if value["measurement"] == "partial_lower_bound"]
    if not full:
        return None, "waiting_for_measured_tie_cost", proof
    fewest = min(value["observed_extra_generations"] for value in full)
    if any(value["observed_extra_generations"] <= fewest for value in partial):
        proof["blocked_by_partial_lower_bound_at_or_below_best_full_extra"] = True
        return None, "waiting_for_measured_tie_cost", proof
    tied = [value for value in full if value["observed_extra_generations"] == fewest]
    if len(tied) == 1:
        winner = tied[0]["controller"]
        proof.update(winner=winner,
                     selector_seconds_used=False,
                     decision="unique full exact extra count strictly below every partial lower bound")
        return winner, None, proof
    if any(value["selector_seconds"] is None for value in tied):
        return None, "waiting_for_measured_tie_cost", proof
    fastest = min(value["selector_seconds"] for value in tied)
    tied = [value for value in tied if value["selector_seconds"] == fastest]
    if len(tied) == 1:
        winner = tied[0]["controller"]
        proof.update(winner=winner,
                     selector_seconds_used=True,
                     decision="full exact extra tie resolved by full measured selector seconds")
        return winner, None, proof
    return None, "exact_cost_tie", proof


def select_top_two(lanes: dict, costs: dict | None = None) -> dict:
    """Rank terminal methods by fixed-D20 operational successes."""
    costs = costs or {}
    if set(lanes) != set(LANES):
        raise ValueError("All seven controller lanes are required")
    scope = {"id": OPERATIONAL_SELECTION_SCOPE, "fixed_denominator": 20,
             "success_definition": "runtime completed with valid official correct_count equal to 1",
             "audited_deployment_failure_treatment": (
                 "operational non-success with official score unobserved"),
             "counterfactual_bounds_used_for_promotion": False}
    pending = [name for name in LANES if not lanes[name]["all_cells_terminal"]]
    if pending:
        return {"phase": "waiting_for_d20", "pending_lanes": pending, "selected": [],
                "operational_selection_scope": scope}
    blocked = {name: lanes[name].get("unaudited_runtime_failures", [])
               for name in LANES if lanes[name].get("unaudited_runtime_failures")}
    if blocked:
        return {"phase": "unaudited_runtime_failure_blocks_promotion", "selected": [],
                "blocked_lanes": blocked, "operational_selection_scope": scope}
    eligible = [name for name in LANES if lanes[name].get("operational_selection_eligible")]
    if len(eligible) < 2:
        return {"phase": "insufficient_terminal_methods", "selected": [],
                "operational_selection_scope": scope}
    score_levels = sorted(
        {lanes[name]["operational_success_count"] for name in eligible}, reverse=True)
    selected = []
    reasons = []
    for score in score_levels:
        tied = [name for name in eligible
                if lanes[name]["operational_success_count"] == score]
        remaining = 2 - len(selected)
        if len(tied) <= remaining:
            selected.extend(tied)
        else:
            by_success = {}
            for name in tied:
                by_success.setdefault(
                    tuple(sorted(lanes[name]["operational_success_ids"])), []).append(name)
            groups = list(by_success.values())
            if len(groups) > remaining:
                return {"phase": "complementary_tie_at_cutoff", "selected": [],
                        "certain_higher_ranked": selected, "tied_methods": tied,
                        "success_groups": groups, "operational_selection_scope": scope}
            representatives = []
            for group in groups:
                if len(group) == 1:
                    representatives.append(group[0])
                    continue
                winner, reason, proof = _best_cost_method(group, costs, lanes)
                if reason:
                    return {"phase": reason, "selected": [], "tied_methods": group,
                            "operational_selection_scope": scope,
                            "tie_cost_evidence": proof}
                representatives.append(winner)
                reasons.append({"same_success_set": group, "preferred": winner,
                                "cost_proof": proof})
            selected.extend(representatives)
            # If all tied methods have the same success set, fill the remaining
            # place from that score level before considering a lower score.
            if len(selected) < 2:
                leftover = [name for name in tied if name not in selected]
                winner, reason, proof = _best_cost_method(leftover, costs, lanes)
                if reason:
                    return {"phase": reason, "selected": [], "tied_methods": leftover,
                            "operational_selection_scope": scope,
                            "tie_cost_evidence": proof}
                selected.append(winner)
                reasons.append({"same_success_set": leftover, "preferred": winner,
                                "cost_proof": proof})
        if len(selected) == 2:
            break
    cutoff = min(lanes[name]["operational_success_count"] for name in selected)
    return {"phase": "promotion_ready", "selected": selected, "cutoff_successes": cutoff,
            "tie_resolution": reasons, "operational_selection_scope": scope,
            "audited_runtime_failure_non_successes": [
                {"lane": name, "task_id": task}
                for name in LANES for task in lanes[name]["audited_runtime_failures"]]}


def validate_reuse_audit(audit: dict | None, rows: dict, canonical: list[str]) -> dict:
    result = {"status": "requires_per_task_frozen_runtime_audit",
              "single_frozen_config_d128_claim_allowed": False,
              "required_reports": ["historical_D20", "new108", "combined_with_provenance"],
              "audited_lanes": []}
    if audit is None:
        return result
    if (audit.get("schema") != "evidence-sets-d20-runtime-compatibility-v1"
            or audit.get("conclusion", {}).get("result_reuse_supported") is not True
            or audit.get("target", {}).get("source_id") != "failed_repair_v1"):
        raise ValueError("Unsupported D20 compatibility audit")
    audited = audit.get("lanes", {})
    if not audited or not set(audited) <= {"C0", "C3", "C5"}:
        raise ValueError("Unexpected lanes in D20 compatibility audit")
    for lane, report in audited.items():
        expected = "semantic_nonactivation_compatible_20_of_20"
        if (report.get("classification") != expected
                or audit["conclusion"].get("lanes", {}).get(lane) != expected
                or not rows[lane]["clean_d20"]):
            raise ValueError(f"{lane}: compatibility audit is not a clean D20")
        tasks = report.get("tasks", [])
        if (len(tasks) != 20 or len({item.get("task_id") for item in tasks}) != 20
                or {item.get("task_id") for item in tasks} != set(canonical)):
            raise ValueError(f"{lane}: compatibility audit covers different tasks")
        by_task = {item["task_id"]: item for item in tasks}
        for cell in rows[lane]["quality_cells"]:
            item = by_task[cell["task_id"]]
            if (item.get("quality_source_id") != cell.get("quality_source_id")
                    or item.get("classification") not in (
                        "semantic_nonactivation_compatible", "byte_identical")):
                raise ValueError(f"{lane}: compatibility audit owner or conclusion differs")
    result.update(status="audited_historical_lanes_bound_to_target",
                  audited_lanes=sorted(audited), target=audit["target"],
                  execution_gate="Expansion runner must verify the exact audited target package")
    return result


def build_plan(untrained: dict, trained: dict, d20: dict, d128: dict, f128: dict,
               training: dict, costs: dict | None = None, reuse_audit: dict | None = None,
               failure_audit: dict | None = None) -> dict:
    remaining = validate_manifests(d20, d128, f128, training)
    canonical = d20["task_ids"]
    rows = {}
    for lane in LANES:
        cells = (trained_cells(trained, lane, canonical) if lane in TRAINED
                 else untrained["lanes"][lane]["quality_cells"])
        rows[lane] = summarize_lane(cells, canonical)
        measured = (costs or {}).get(lane)
        if isinstance(measured, dict) and measured.get("covered_tasks") == 20:
            evidence = measured.get("evidence")
            if (not isinstance(evidence, list) or len(evidence) != 20
                    or {item.get("task_id") for item in evidence} != set(canonical)):
                raise ValueError(f"{lane}: tie costs do not cover the same D20")
            by_task = {item["task_id"]: item for item in evidence}
            for cell in cells:
                item = by_task[cell["task_id"]]
                if (cell["status"] != "completed"
                        or item.get("owner") != cell.get("quality_source_id")
                        or item.get("expected_correct_count") != cell["official"]["correct_count"]):
                    raise ValueError(f"{lane}: tie cost owner differs from current quality owner")
    failure_validation = validate_operational_failure_audit(failure_audit, rows)
    reuse = validate_reuse_audit(reuse_audit, rows, canonical)
    promotion = select_top_two(rows, costs)
    manifests = {lane: {"schema": "a-history-system-task-manifest-v1",
                       "manifest_id": f"exp3_{lane}_d128_remaining108",
                       "stage": "development_search", "task_ids": remaining,
                       "reused_task_ids": canonical, "full_task_ids": d128["task_ids"],
                       "historical_runtime_failed_task_ids":
                           rows[lane]["audited_runtime_failures"],
                       "reuse_requires_unchanged_algorithm_and_verified_d20_provenance": True}
                 for lane in promotion["selected"]}
    promoted_with_failures = [lane for lane in promotion["selected"]
                              if rows[lane]["audited_runtime_failures"]]
    return {"schema": "experiment3-expansion-readiness-v1", "quality_label": "preliminary, n=1",
            "launch_authorized": False, "phase": promotion["phase"], "promotion": promotion,
            "lanes": rows, "d128_remaining_per_promoted_method": len(remaining),
            "next_manifests": manifests, "new_execution_count": len(manifests) * len(remaining),
            "proposed_shards": build_shards(manifests) if manifests else [],
            "reuse_validation": reuse, "operational_failure_validation": failure_validation,
            "downstream_runtime_failure_support": {
                "required_for_selected_controllers": promoted_with_failures,
                "clean_d20_and_complete_d128_validators_changed": False,
                "status": "defer_until_actual_selection" if not promoted_with_failures
                          else "requires_follow_up_before_expansion_execution"},
            "f128_used_for_selection": False, "automatic_retries": 0,
            "existing_d128_f128_shared_group_ids": sorted(
                task_groups(d128["task_ids"]) & task_groups(f128["task_ids"])),
            "later_stages": ["Both promoted controllers under H0 and H1",
                             "Calibrate hidden-state-dependent heads before H1 claims",
                             "Leading complete configuration with R3",
                             "Verify result artifacts and release owned accelerator processes"]}


def build_shards(manifests: dict) -> list[dict]:
    """Prepare six disjoint task shards without reserving or starting devices."""
    if len(manifests) != 2:
        raise ValueError("Expansion requires exactly two promoted methods")
    rows = []
    devices = ((0, 1, 2), (3, 4, 6))
    for rank, (lane, manifest) in enumerate(manifests.items()):
        ids = manifest_ids(manifest, 108, f"{lane} remaining D128")
        for shard_index, device in enumerate(devices[rank]):
            rows.append({"shard_id": f"{lane}_part{shard_index}", "controller": lane,
                         "preferred_physical_device": device,
                         "wait_for_device_free_and_verify_owner": True,
                         "task_ids": ids[shard_index::3], "task_budget": 36,
                         "automatic_retries": 0, "launch_authorized": False})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--untrained-summary", type=Path, required=True)
    parser.add_argument("--d20", type=Path, required=True)
    parser.add_argument("--d128", type=Path, required=True)
    parser.add_argument("--f128", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--costs", type=Path)
    parser.add_argument("--reuse-audit", type=Path)
    parser.add_argument("--failure-audit", type=Path)
    parser.add_argument("--ssh-host", default="npu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = read(args.reuse_audit) if args.reuse_audit else None
    failure_audit = read(args.failure_audit) if args.failure_audit else None
    if audit is not None and audit.get("evidence", {}).get("summary_latest", {}).get(
            "sha256") != digest(args.untrained_summary):
        raise ValueError("Compatibility audit binds a different D20 quality summary")
    snapshot = collect_trained(args.ssh_host)
    result = build_plan(read(args.untrained_summary), snapshot, read(args.d20), read(args.d128),
                        read(args.f128), read(args.training), read(args.costs) if args.costs else None,
                        reuse_audit=audit, failure_audit=failure_audit)
    inputs = ("untrained_summary", "d20", "d128", "f128", "training")
    result["inputs"] = {name: {"path": str(getattr(args, name).resolve()),
                               "sha256": digest(getattr(args, name))} for name in inputs}
    if args.costs:
        result["inputs"]["costs"] = {"path": str(args.costs.resolve()), "sha256": digest(args.costs)}
    if args.reuse_audit:
        result["inputs"]["reuse_audit"] = {
            "path": str(args.reuse_audit.resolve()), "sha256": digest(args.reuse_audit)}
    if args.failure_audit:
        result["inputs"]["failure_audit"] = {
            "path": str(args.failure_audit.resolve()), "sha256": digest(args.failure_audit)}
    result["trained_snapshot"] = snapshot
    result["observed_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"phase": result["phase"], "promotion": result["promotion"],
                      "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
