"""Task-level repair oracle for the bench layer (hybrid-x-D combo).

The repair arm is only ever run on the oracle-triggered subset: tasks where
the full arm SUCCEEDED and the compressed base arm FAILED.  This module (a)
derives that eligible id set from two finished runs' artifacts, (b) emits the
queue task file carrying the per-benchmark id restriction, and (c) scores a
finished repair run on the same subset.  Pure stdlib; run on the server (or
against pulled-back bundles).

Per-benchmark success predicate (documented, big-effect reading only):
* tau2  — official reward == 1.0 per task_id (updated_results.json).
* bfcl  — id absent from the score json's failure records (the score file is
          [header, *failure_records]; eligible = base_failures - full_failures).
* ts    — scenario similarity >= 0.5 in result_summary.json (test subset).

Usage:
  python benchmarks/repair_oracle.py eligible -b tau2 \
      --full-run up_tau2_full --base-run hr_tau2_c2kv \
      --out results/bench/oracle_tau2_c2kv.json
  python benchmarks/repair_oracle.py task -b tau2 --arm c2kv_repair \
      --eligible results/bench/oracle_tau2_c2kv.json --name hr_tau2_c2kv_rp \
      --ckpt /home/liuyancheng/checkpoints_upstream/checkpoint-1088 \
      [--out ~/bench_queue/pending/hr_tau2_c2kv_rp.task]
  python benchmarks/repair_oracle.py score -b tau2 \
      --repair-run hr_tau2_c2kv_rp --eligible results/bench/oracle_tau2_c2kv.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

TAU2_SIMS = Path.home() / "benchmarks/tau2/data/simulations"
BFCL_ARCHIVE = Path.home() / "bench_results/bfcl_archive"
TS_RESULTS = Path.home() / "bench_results"
PROXY_LOGS = Path.home() / "bench_logs"
CKPT_1088 = "/home/liuyancheng/checkpoints_upstream/checkpoint-1088"
TS_SUCCESS_SIM = 0.5


def _tau2_scores(run: str) -> Dict[str, Optional[float]]:
    path = TAU2_SIMS / run / "updated_results.json"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        str(sim.get("task_id")): (sim.get("reward_info") or {}).get("reward")
        for sim in data.get("simulations") or []
    }


def _bfcl_fail_ids(run: str) -> List[str]:
    hits = sorted(glob.glob(str(BFCL_ARCHIVE / run / "*_score.json")))
    if not hits:
        raise SystemExit(f"FATAL: no archived score json for run {run!r} under {BFCL_ARCHIVE / run}")
    rows = json.loads(Path(hits[0]).read_text(encoding="utf-8"))
    return [str(row.get("id")) for row in rows[1:] if row.get("id") is not None]


def _ts_scores(run: str) -> Dict[str, Optional[float]]:
    hits = sorted(glob.glob(str(TS_RESULTS / f"task_{run}" / "*" / "result_summary.json")))
    if not hits:
        raise SystemExit(f"FATAL: no result_summary.json for run {run!r} under {TS_RESULTS / f'task_{run}'}")
    data = json.loads(Path(hits[-1]).read_text(encoding="utf-8"))
    return {
        str(r.get("name")): r.get("similarity")
        for r in data.get("per_scenario_results") or []
    }


def _success(benchmark: str, score: Optional[float]) -> bool:
    if score is None:
        return False
    if benchmark == "bfcl":
        return bool(score)
    return float(score) >= (1.0 if benchmark == "tau2" else TS_SUCCESS_SIM)


def eligible_set(benchmark: str, full_run: str, base_run: str) -> Dict[str, Any]:
    """ids where full succeeded and base failed."""
    if benchmark == "tau2":
        full, base = _tau2_scores(full_run), _tau2_scores(base_run)
    elif benchmark == "bfcl":
        full_f, base_f = set(_bfcl_fail_ids(full_run)), set(_bfcl_fail_ids(base_run))
        full, base = (
            {i: 1.0 for i in base_f - full_f},  # full passed (not a failure)
            {i: 0.0 for i in base_f},
        )
    elif benchmark == "ts":
        full, base = _ts_scores(full_run), _ts_scores(base_run)
    else:
        raise SystemExit(f"FATAL: unknown benchmark {benchmark!r}")
    ids = sorted(
        i for i in base
        if not _success(benchmark, base[i])
        and _success(benchmark, full.get(i))
    )
    return {
        "benchmark": benchmark, "full_run": full_run, "base_run": base_run,
        "n_full_scored": len(full), "n_base_scored": len(base),
        "n_eligible": len(ids), "eligible_ids": ids,
    }


def write_task(benchmark: str, arm: str, ckpt: str, name: str,
               eligible: Dict[str, Any], out: Optional[str]) -> str:
    ids = eligible["eligible_ids"]
    lines = [f"BENCH={benchmark}", f"ARM={arm}", f"CKPT={ckpt}"]
    if ids:
        sep = " " if benchmark in ("tau2", "ts") else ","
        lines.append(f'{ "TASK_IDS" if benchmark == "tau2" else "RUN_IDS" if benchmark == "bfcl" else "SCENARIOS" }="{sep.join(ids)}"')
    body = "\n".join(lines) + "\n"
    if out:
        Path(out).write_text(body, encoding="utf-8")
        return out
    return body


def _proxy_costs(run: str) -> Dict[str, Any]:
    path = PROXY_LOGS / f"proxy_task_{run}.jsonl"
    if not path.exists():
        return {"proxy_log": None}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out: Dict[str, Any] = {"proxy_log": str(path), "n_requests": len(rows)}
    for key in ("gist_tokens", "original_tokens", "repair_block_tokens", "wall_sec"):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            out[f"{key}_sum"] = round(sum(vals), 1)
            out[f"{key}_mean"] = round(sum(vals) / len(vals), 1)
    return out


def score_repair(benchmark: str, repair_run: str, eligible: Dict[str, Any]) -> Dict[str, Any]:
    ids = set(eligible["eligible_ids"])
    if benchmark == "tau2":
        scores = _tau2_scores(repair_run)
    elif benchmark == "bfcl":
        fails = set(_bfcl_fail_ids(repair_run))
        scores = {i: (0.0 if i in fails else 1.0) for i in ids}
    elif benchmark == "ts":
        scores = _ts_scores(repair_run)
    else:
        raise SystemExit(f"FATAL: unknown benchmark {benchmark!r}")
    scored = {i: scores.get(i) for i in sorted(ids)}
    rescued = [i for i, s in scored.items() if _success(benchmark, s)]
    missing = [i for i in scored if i not in scores]
    return {
        "benchmark": benchmark, "repair_run": repair_run,
        "n_eligible": len(ids), "n_rescued": len(rescued),
        "rescue_rate": round(len(rescued) / len(ids), 4) if ids else None,
        "rescued_ids": rescued, "unscored_ids": missing,
        "costs": _proxy_costs(repair_run),
        "per_id": scored,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_el = sub.add_parser("eligible", help="compute the oracle subset from full+base runs")
    p_el.add_argument("-b", "--benchmark", required=True, choices=["tau2", "bfcl", "ts"])
    p_el.add_argument("--full-run", required=True)
    p_el.add_argument("--base-run", required=True)
    p_el.add_argument("--out", required=True)
    p_tk = sub.add_parser("task", help="emit the repair-arm queue task file")
    p_tk.add_argument("-b", "--benchmark", required=True, choices=["tau2", "bfcl", "ts"])
    p_tk.add_argument("--arm", required=True)
    p_tk.add_argument("--name", required=True)
    p_tk.add_argument("--ckpt", default=CKPT_1088)
    p_tk.add_argument("--eligible", required=True)
    p_tk.add_argument("--out", default=None, help="task file path; default = print to stdout")
    p_sc = sub.add_parser("score", help="score a repair run on the oracle subset")
    p_sc.add_argument("-b", "--benchmark", required=True, choices=["tau2", "bfcl", "ts"])
    p_sc.add_argument("--repair-run", required=True)
    p_sc.add_argument("--eligible", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "eligible":
        result = eligible_set(args.benchmark, args.full_run, args.base_run)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    elif args.cmd == "task":
        eligible = json.loads(Path(args.eligible).read_text(encoding="utf-8"))
        body = write_task(args.benchmark, args.arm, args.ckpt, args.name, eligible, args.out)
        print(body if not args.out else f"wrote {args.out}")
    else:
        eligible = json.loads(Path(args.eligible).read_text(encoding="utf-8"))
        result = score_repair(args.benchmark, args.repair_run, eligible)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
