"""Aggregate GP candidate runs into per-config analyses and a group summary."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def step_stats(steps_path: Path) -> dict:
    stats = {
        "decisions": 0,
        "recovery_checks": 0,
        "recovery_triggereds": 0,
        "recovery_rounds_sum": 0,
        "selected_units_sum": 0,
        "appended_units_sum": 0,
        "regeneration_count": 0,
    }
    if not steps_path.exists():
        return stats
    with steps_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                step = json.loads(line)
            except Exception:
                continue
            stats["decisions"] += 1
            checks = step.get("recovery_checks")
            if isinstance(checks, list):
                stats["recovery_checks"] += len(checks)
            rounds = step.get("recovery_rounds")
            recovered = False
            if isinstance(rounds, list):
                for entry in rounds:
                    if not isinstance(entry, dict):
                        continue
                    appended = entry.get("appended_unit_count") or 0
                    selected = entry.get("selected_unit_count") or 0
                    stats["selected_units_sum"] += selected
                    stats["appended_units_sum"] += appended
                    if entry.get("status") == "recover" or appended:
                        stats["recovery_rounds_sum"] += 1
                        recovered = recovered or bool(appended)
            if recovered:
                stats["recovery_triggereds"] += 1
            trace = step.get("generation_trace")
            if isinstance(trace, list) and len(trace) > 1:
                stats["regeneration_count"] += len(trace) - 1
    return stats


def analyse_run(run_dir: Path) -> dict | None:
    manifest = read(run_dir / "stage_manifest.json")
    if manifest is None:
        return None
    per_task = {}
    recovery = {
        "decisions": 0, "recovery_checks": 0, "recovery_triggereds": 0,
        "recovery_rounds_sum": 0, "selected_units_sum": 0,
        "appended_units_sum": 0, "regeneration_count": 0,
    }
    task_walls = {}
    for shard in sorted((run_dir / "task_shards").iterdir()):
        if not shard.is_dir():
            continue
        task_id = shard.name
        summary = read(shard / "bfcl" / "official_summary.json")
        if summary is not None and summary.get("n_scored", 0) > 0:
            passed = int(summary.get("correct_count", 0)) == int(summary.get("n", 1))
            category = summary.get("categories", "")
        else:
            passed = False
            category = ("multi_turn_long_context" if "long_context" in task_id
                        else "multi_turn_base")
        per_task[task_id] = {"passed": passed, "category": category,
                             "official": summary is not None}
        task_stats = step_stats(shard / "server" / "steps.jsonl")
        for key in recovery:
            recovery[key] += task_stats[key]
        final = read(shard / "server" / "final.json") or {}
        cost = final.get("cost_inventory") or {}
        task_walls[task_id] = cost.get("wall_seconds")
    passed_tasks = sorted(t for t, row in per_task.items() if row["passed"])
    return {
        "schema": "a-history-gp-analysis-v1",
        "candidate_id": manifest.get("candidate_id"),
        "status": manifest.get("status"),
        "state": manifest.get("state"),
        "denominator": manifest.get("whole_task_denominator"),
        "score": len(passed_tasks),
        "passed_tasks": passed_tasks,
        "base_pass": sum(1 for row in per_task.values()
                         if row["passed"] and row["category"] == "multi_turn_base"),
        "long_pass": sum(1 for row in per_task.values()
                         if row["passed"] and row["category"] == "multi_turn_long_context"),
        "per_task": per_task,
        "recovery": recovery,
        "task_wall_seconds": task_walls,
        "wall_seconds": manifest.get("wall_seconds"),
        "qualification": "preliminary, n=1",
        "collected_at_epoch": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline", default=None,
                        help="candidate_id used as the delta-N reference")
    parser.add_argument("--out", default=None, help="summary output name (group id)")
    args = parser.parse_args()
    runs_root = args.run_root / "runs"
    analyses = {}
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        analysis = analyse_run(run_dir)
        if analysis is None:
            continue
        analyses[run_dir.name] = analysis
        save(run_dir / "analysis.json", analysis)
    baseline = None
    if args.baseline and args.baseline in analyses:
        baseline = set(analyses[args.baseline]["passed_tasks"])
    rows = []
    for name, analysis in analyses.items():
        row = {
            "run": name,
            "state": analysis["state"],
            "status": analysis.get("status"),
            "score": analysis["score"],
            "base": analysis["base_pass"],
            "long": analysis["long_pass"],
            "delta_n": None,
            "new_pass": None, "new_fail": None,
            "appended_units": analysis["recovery"]["appended_units_sum"],
            "recovery_triggereds": analysis["recovery"]["recovery_triggereds"],
            "regenerations": analysis["recovery"]["regeneration_count"],
            "wall_seconds": analysis["wall_seconds"],
        }
        if baseline is not None and name != args.baseline:
            current = set(analysis["passed_tasks"])
            row["new_pass"] = sorted(current - baseline)
            row["new_fail"] = sorted(baseline - current)
            row["delta_n"] = len(current - baseline) - len(baseline - current)
        rows.append(row)
    rows.sort(key=lambda r: (-r["score"], r["run"]))
    summary = {
        "schema": "a-history-gp-summary-v1",
        "baseline": args.baseline,
        "qualification": "preliminary, n=1",
        "rows": rows,
    }
    out_name = args.out or "summary"
    save(args.run_root / f"{out_name}.json", summary)
    lines = [
        f"# GP summary ({out_name}) — preliminary, n=1",
        "",
        "| run | state | score | base | long | ΔN | appended | trigger | regen | wall_s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        delta = "" if row["delta_n"] is None else str(row["delta_n"])
        wall = "" if row["wall_seconds"] is None else f"{row['wall_seconds']:.0f}"
        lines.append(
            f"| {row['run']} | {row['state']} | {row['score']} | {row['base']} "
            f"| {row['long']} | {delta} | {row['appended_units']} "
            f"| {row['recovery_triggereds']} | {row['regenerations']} | {wall} |")
    (args.run_root / f"{out_name}.md").write_text("\n".join(lines) + "\n",
                                                  encoding="utf-8")
    print(json.dumps({"configs": len(rows),
                      "top": rows[:5]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
