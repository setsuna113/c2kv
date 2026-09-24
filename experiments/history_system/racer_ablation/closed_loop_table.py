"""Closed-loop ablation table (M0-M4) from extract_decisions outputs.

Task success uses the full 200-task denominator: a method capacity failure is an
unfinished task scored 0 and counted separately; harness/scorer failures are
listed and never silently dropped. Differences are paired by task against M1
(RACER-core) with a task-resampling bootstrap interval.

usage: python -m racer_ablation.closed_loop_table OUT_JSON M0.json M1.json M2.json M3.json M4.json
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

from .common import write_json
from .heldout import heldout_tasks

LABELS = ("M0", "M1", "M2", "M3", "M4")
TASKS = [f"multi_turn_long_context_{index}" for index in range(200)]
BOOTSTRAP = 10000
SEED = 20260924


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def paired(reference, other, tasks):
    x = np.array([other[t]["correct"] for t in tasks], dtype=float)
    y = np.array([reference[t]["correct"] for t in tasks], dtype=float)
    diff = x - y
    rng = np.random.default_rng(SEED)
    draws = diff[rng.integers(0, len(diff), (BOOTSTRAP, len(diff)))].mean(axis=1)
    only_other = int(((x == 1) & (y == 0)).sum())
    only_reference = int(((x == 0) & (y == 1)).sum())
    return {"n": len(tasks), "difference": float(diff.mean()),
            "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            "other_only": only_other, "reference_only": only_reference,
            "mcnemar_exact_p": mcnemar_exact(only_other, only_reference)}


def funnel(tasks):
    decisions = [d for task in tasks.values() for d in task["decisions"]]
    keys = ("triggered", "generation_limit", "repack_feasible", "no_feasible_candidate",
            "self_revision_context_limit", "regeneration_attempted", "regeneration_completed",
            "regenerated_action_committed", "parse_fallback", "correction_proposed",
            "correction_verified", "correction_changed")
    return {"decisions": len(decisions), "failed_decisions": sum(d["status"] != "ok" for d in decisions),
            **{key: sum(bool(d[key]) for d in decisions) for key in keys}}


def summarize(extracted):
    tasks = extracted["tasks"]
    missing = [t for t in TASKS if t not in tasks]
    completed = [t for t in TASKS if t in tasks]
    decisions = sum(task["decision_count"] for task in tasks.values())
    history = sum(task["history_kv_tokens_sum"] for task in tasks.values())
    generations = sum(task["history_kv_generations"] for task in tasks.values())
    regenerations = sum(d["regeneration_completed"] for task in tasks.values() for d in task["decisions"])
    return {
        "cell": extracted["cell"], "tasks_present": len(completed), "tasks_missing": missing,
        "correct": sum(tasks[t]["correct"] for t in completed),
        "success_rate_full_denominator": sum(tasks[t]["correct"] for t in completed) / len(TASKS),
        "method_failures": sorted(t for t in completed if tasks[t]["method_failure"]),
        "harness_failures": sorted(t for t in completed if tasks[t]["harness_failure"]),
        "regenerations_per_decision": regenerations / decisions if decisions else None,
        "actor_generations_per_task": sum(tasks[t]["generations_completed"] for t in completed) / len(completed),
        "mean_history_kv_tokens": history / generations if generations else None,
        "history_kv_generations": generations,
        "unified_history_kv_check": sum(abs((task["unified_active_history_kv"] or 0)
                                            - task["history_kv_tokens_sum"]) for task in tasks.values()),
        "prompt_tokens_per_task": sum(tasks[t]["prompt_tokens"] for t in completed) / len(completed),
        "completion_tokens_per_task": sum(tasks[t]["completion_tokens"] for t in completed) / len(completed),
        "self_revision_added_tokens": sum(d["self_revision_added_tokens"] or 0
                                          for task in tasks.values() for d in task["decisions"]),
        "funnel": funnel(tasks),
    }


def correction_effect(full, core):
    changed = {t: sum(d["correction_changed"] for d in full[t]["decisions"]) for t in full}
    full_only = sorted(t for t in TASKS if t in full and t in core
                       and full[t]["correct"] and not core[t]["correct"])
    core_only = sorted(t for t in TASKS if t in full and t in core
                       and core[t]["correct"] and not full[t]["correct"])
    return {"decisions_changed_by_correction": sum(changed.values()),
            "tasks_with_changed_correction": sum(1 for value in changed.values() if value),
            "full_only_correct": full_only, "core_only_correct": core_only,
            "full_only_with_changed_correction": [t for t in full_only if changed.get(t)],
            "core_only_with_changed_correction_in_full": [t for t in core_only if changed.get(t)]}


def main(argv):
    out, paths = argv[0], argv[1:]
    runs = {label: json.load(open(path, encoding="utf-8")) for label, path in zip(LABELS, paths)}
    tables = {label: summarize(run) for label, run in runs.items()}
    reference = runs["M1"]["tasks"]
    common = [t for t in TASKS if all(t in run["tasks"] for run in runs.values())]
    heldout = [t for t in heldout_tasks() if t in common]
    result = {"schema": "racer-ablation-closed-loop-v1", "result_status": "preliminary, n=1",
              "rows": tables, "paired_vs_M1": {}, "paired_vs_M1_heldout_subset": {},
              "common_tasks": len(common), "heldout_tasks": len(heldout),
              "correction_M0_vs_M1": correction_effect(runs["M0"]["tasks"], reference)}
    for label, run in runs.items():
        if label != "M1":
            result["paired_vs_M1"][label] = paired(reference, run["tasks"], common)
            result["paired_vs_M1_heldout_subset"][label] = paired(reference, run["tasks"], heldout)
    write_json(out, result)
    for label in LABELS:
        row = tables[label]
        print(label, row["correct"], "/200", "method_failures", len(row["method_failures"]),
              "harness", len(row["harness_failures"]), "regen/dec", row["regenerations_per_decision"],
              "gen/task", row["actor_generations_per_task"], "histKV", row["mean_history_kv_tokens"])


if __name__ == "__main__":
    main(sys.argv[1:])
