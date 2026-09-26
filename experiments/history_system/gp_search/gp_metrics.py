"""Batch five-metric scoring for GP runs (CPU-only, no model calls).

Metrics (2026-09-16 scoring spec):
 1. whole-task success (official scorer)
 2. consecutive correct progress over user turns (official prefix checkers,
    replaying the committed actions recorded in server/steps.jsonl)
 3. first-failed-turn required-response coverage + state_valid (responses
    obtained by re-executing committed calls with the official executor)
 4. post-recovery outcome grouping (first append+regeneration turn)
 5. cost (generations, tokens, recovery counts, wall; KV/encoder when recorded)

Server usage (bench python has bfcl_eval):
  /home/liuyancheng/envs/bench/bin/python gp_metrics.py --group A
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

RUN_ROOT = Path("/home/liuyancheng/gp_search_v1")
HS = RUN_ROOT / "history_system"
sys.path.insert(0, str(HS / "runtime" / "benchmarks"))
sys.path.insert(0, str(HS))
os.environ.pop("http_proxy", None), os.environ.pop("https_proxy", None)
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from bfcl_gold_recovery import official_prefix_check, _sanitize_namespace  # noqa: E402
from detectors.labels import _default_task_loader  # noqa: E402


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def decode_turns_from_record(inference_log, total_turns):
    """Model call batches per user turn via the labels decoder (CPU-validated
    on this exact format: per-turn dicts with step_N message lists)."""
    from detectors.labels import decoded_turns_from_inference_log
    decoded = decoded_turns_from_inference_log(inference_log)
    turns = [(t.get("decoded_batches") or []) if t.get("complete") else []
             for t in decoded]
    turns += [[]] * (total_turns - len(turns))
    return turns[:total_turns]


def recorded_tool_responses(inference_log, upto_turn):
    """Actual executed tool responses through turn `upto_turn`, in order."""
    responses = []
    if not isinstance(inference_log, list):
        return responses
    turn_dicts = [e for e in inference_log if isinstance(e, dict)]
    for entry in turn_dicts[: upto_turn + 1]:
        for key in sorted(k for k in entry if str(k).startswith("step_")):
            for message in entry[key]:
                if isinstance(message, dict) and message.get("role") == "tool":
                    responses.append(message.get("content"))
    return responses


def _executor():
    checker = importlib.import_module(
        "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker")
    return checker.execute_multi_turn_func_call


def _matching():
    checker = importlib.import_module(
        "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker")
    return checker._is_subsequence_unordered


def execute_ground_truth_responses(ground_truth, turn, test_entry, tag):
    """Execute the official reference calls of one turn in a private sandbox."""
    execute = _executor()
    namespace = _sanitize_namespace(f"c2kv_metrics_gt_{tag}_{uuid.uuid4().hex}")
    initial_config = test_entry["initial_config"]
    involved = test_entry["involved_classes"]
    task_id = test_entry["id"]
    long_context = "long_context" in str(task_id) or "composite" in str(task_id)
    utils = importlib.import_module(
        "bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils")
    before = set(vars(utils))
    try:
        results, _ = execute(
            func_call_list=copy.deepcopy(ground_truth[turn]),
            initial_config=copy.deepcopy(initial_config),
            involved_classes=copy.deepcopy(involved),
            model_name=namespace,
            test_entry_id=task_id,
            long_context=long_context,
            is_evaL_run=True,
        )
        return results
    finally:
        for name in set(vars(utils)) - before:
            if name.startswith(namespace) and name.endswith("_instance"):
                vars(utils).pop(name, None)


def prefix_progress(decoded_turns, ground_truth, test_entry):
    total = len(ground_truth)
    per_turn_valid = []
    failure = None
    for k in range(1, total + 1):
        result = official_prefix_check(decoded_turns[:k], ground_truth[:k], test_entry)
        valid = bool(result.get("valid"))
        per_turn_valid.append(valid)
        if not valid:
            failure = {"turn_index": k - 1, "result": result}
            break
    progress = failure["turn_index"] if failure else total
    return progress, total, per_turn_valid, failure


def coverage_at_failure(failure, inference_log, ground_truth, test_entry, run, task):
    turn = failure["turn_index"]
    main = (failure.get("result") or {}).get("multi_turn") or {}
    error_type = main.get("error_type", "invalid_irrelevance")
    state_valid = error_type != "multi_turn:instance_state_mismatch"
    differences = (main.get("details") or {}).get("differences") \
        if not state_valid else None
    gt_responses = execute_ground_truth_responses(
        ground_truth, turn, test_entry, f"{run}_{task}")
    if not gt_responses:
        return {"coverage": None, "state_valid": state_valid,
                "state_differences": differences, "error_type": error_type,
                "note": "empty required set"}
    actual = recorded_tool_responses(inference_log, turn)
    ok, missing = _matching()(gt_responses, actual)
    def _plain(value):
        return json.loads(json.dumps(value, default=str))
    return {
        "coverage": round(1 - len(missing) / len(gt_responses), 4),
        "matched": len(gt_responses) - len(missing),
        "required": len(gt_responses),
        "missing_sample": [str(m)[:120] for m in list(missing)[:4]],
        "state_valid": state_valid,
        "state_differences": _plain(differences) if differences else None,
        "error_type": error_type,
    }


def first_recovery_turn(steps_path: Path):
    if not steps_path.exists():
        return None
    with steps_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                step = json.loads(line)
            except Exception:
                continue
            rounds = step.get("recovery_rounds")
            if not (isinstance(rounds, list) and any(
                    (r or {}).get("appended_unit_count") for r in rounds
                    if isinstance(r, dict))):
                continue
            trace = step.get("generation_trace") or []
            if any((t or {}).get("phase") == "regeneration" for t in trace
                   if isinstance(t, dict)):
                match = re.match(r"turn-(\d+)", str(step.get("decision_key", "")))
                if match:
                    return int(match.group(1))
    return None


def cost_of_shard(shard: Path, steps_path: Path):
    cost = {"generations": 0, "regenerations": 0, "recovery_appends": 0,
            "usage_total": None, "wall_seconds": None, "kv_peak_bytes": None}
    if steps_path.exists():
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage_known = False
        with steps_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    step = json.loads(line)
                except Exception:
                    continue
                trace = step.get("generation_trace") or []
                cost["generations"] += len(trace)
                cost["regenerations"] += max(0, len(trace) - 1)
                rounds = step.get("recovery_rounds")
                if isinstance(rounds, list):
                    cost["recovery_appends"] += sum(
                        (r or {}).get("appended_unit_count") or 0 for r in rounds
                        if isinstance(r, dict))
                if step.get("generation_usage_known"):
                    total = step.get("generation_usage_total")
                    if isinstance(total, dict):
                        for key in usage:
                            usage[key] += total.get(key) or 0
                        usage_known = True
        if usage_known:
            cost["usage_total"] = usage
    final = read_json(shard / "server" / "final.json") or {}
    inventory = final.get("cost_inventory") or {}
    cost["wall_seconds"] = inventory.get("wall_seconds")
    cost["kv_peak_bytes"] = inventory.get("active_kv_peak_bytes") or \
        inventory.get("kv_peak_bytes")
    return cost


def score_task(run_name, task_id, shard, loader_cache):
    summary = read_json(shard / "bfcl" / "official_summary.json")
    if summary is None:
        return {"task_id": task_id, "status": "no_official_summary"}
    sr = int(summary.get("correct_count", 0)) == int(summary.get("n", 1))
    if task_id not in loader_cache:
        loader_cache[task_id] = _default_task_loader(task_id)
    test_entry, ground_truth = loader_cache[task_id]
    steps_path = shard / "server" / "steps.jsonl"
    from detectors.labels import _result_path, _task_result
    inference_log = []
    result_path = _result_path(shard / "bfcl")
    if result_path is not None:
        task_record = _task_result(result_path, task_id) or {}
        inference_log = task_record.get("inference_log") or []
    decoded = decode_turns_from_record(inference_log, len(ground_truth))
    progress, total, per_turn_valid, failure = prefix_progress(
        decoded, ground_truth, test_entry)
    row = {
        "task_id": task_id,
        "sr": 1 if sr else 0,
        "progress": f"{progress}/{total}",
        "progress_value": round(progress / total, 4),
        "first_failed_turn": failure["turn_index"] if failure else None,
    }
    if failure:
        row["coverage"] = coverage_at_failure(
            failure, inference_log, ground_truth, test_entry, run_name, task_id)
    else:
        row["coverage"] = {"coverage": None, "note": "no failed turn"}
    recovery_turn = first_recovery_turn(steps_path)
    row["first_recovery_turn"] = recovery_turn
    if recovery_turn is not None:
        f = row["first_failed_turn"]
        if f is not None and f < recovery_turn:
            group = "failed_before_recovery"
        elif f is not None and f == recovery_turn:
            group = "recovery_turn_failed"
        elif f is not None:
            group = "passed_then_failed_later"
        else:
            group = ("recovery_turn_passed_whole_task_passed" if sr
                     else "recovery_turn_passed_whole_task_failed")
        row["recovery_group"] = group
    row["cost"] = cost_of_shard(shard, steps_path)
    return row


def score_run(run_dir: Path, loader_cache):
    rows = []
    shards = run_dir / "task_shards"
    for shard in sorted(shards.iterdir()) if shards.exists() else []:
        if not shard.is_dir():
            continue
        try:
            rows.append(score_task(run_dir.name, shard.name, shard, loader_cache))
        except Exception as error:
            rows.append({"task_id": shard.name, "status": "scoring_error",
                         "error": f"{type(error).__name__}: {error}"})
    known = [r for r in rows if r.get("sr") is not None]
    later = [r for r in known if r.get("recovery_group") == "passed_then_failed_later"]
    passed_r = [r for r in known if r.get("recovery_group") in (
        "passed_then_failed_later",
        "recovery_turn_passed_whole_task_passed")]
    return {
        "schema": "a-history-gp-metrics-v1",
        "run": run_dir.name,
        "qualification": "preliminary, n=1",
        "known_pass": sum(r["sr"] for r in known),
        "known_fail": sum(1 - r["sr"] for r in known),
        "unknown": sum(1 for r in rows if r.get("sr") is None),
        "mean_progress": round(
            sum(r["progress_value"] for r in known) / len(known), 4) if known else None,
        "later_failure_rate": (round(len(later) / len(passed_r), 4)
                               if passed_r and len(passed_r) > 0 else None),
        "cost_totals": {
            "generations": sum((r.get("cost") or {}).get("generations") or 0 for r in rows),
            "recovery_appends": sum((r.get("cost") or {}).get("recovery_appends") or 0
                                    for r in rows),
            "wall_seconds": sum((r.get("cost") or {}).get("wall_seconds") or 0
                                for r in rows),
        },
        "tasks": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--group", default=None)
    parser.add_argument("--all-groups", action="store_true")
    args = parser.parse_args()
    targets = list(args.run)
    if args.group or args.all_groups:
        groups = ([args.group] if args.group else sorted(
            p.stem.removeprefix("build.") for p in RUN_ROOT.glob("build.[A-Z].json")))
        for group in groups:
            mappings = (read_json(RUN_ROOT / f"build.{group}.json") or {}
                        ).get("mappings", [])
            for mapping in mappings:
                run_dir = RUN_ROOT / "runs" / Path(mapping["directory"]).name
                record = read_json(run_dir / "stage_manifest.json") or {}
                if record.get("status") == "completed_fixed_manifest":
                    targets.append(run_dir)
    loader_cache = {}
    for run_dir in targets:
        started = time.time()
        metrics = score_run(run_dir.resolve(), loader_cache)
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"run": run_dir.name, "pass": metrics["known_pass"],
                          "mean_progress": metrics["mean_progress"],
                          "seconds": round(time.time() - started, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
