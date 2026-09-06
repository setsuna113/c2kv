"""Offline re-evaluation of a killed ToolSandbox run (trajectory salvage).

The ``tool_sandbox`` CLI computes per-scenario similarity in memory and only
writes ``result_summary.json`` when the WHOLE run finishes.  When a run is
killed mid-flight the per-scenario artifacts on disk are already complete
for every finished scenario, so the deterministic evaluation can be
replayed offline:

    ExecutionContext.from_dict(execution_context.json)
    scenario.evaluation.evaluate(ctx, max_turn_count=scenario.max_messages)

Scored set = scenario dirs WITH ``conversation.json`` (the CLI writes the
conversation AFTER evaluate succeeded, so those scenarios finished
play_and_evaluate).  Dirs without it were killed/crashed mid-play and are
counted as missing — never zero-filled (terminal-state semantics).

Usage (benchts venv on the server, from the repo root):

    # validate the replay against a COMPLETED run's result_summary.json
    python benchmarks/ts_salvage_replay.py --validate \
        --run-dir ~/bsa_results/gate_ts/agent_..._09_03_2026_12_36_12 \
        --ts-root ~/benchmarks/ToolSandbox

    # salvage the killed run and write summary_<arm>.json for sg_harvest
    python benchmarks/ts_salvage_replay.py \
        --run-dir ~/bsa_results/matrix2/ts_full_e8f647e/agent_..._09_04_2026_11_43_31 \
        --ts-root ~/benchmarks/ToolSandbox --arm full --n-total 1032 \
        --out-summary ~/bsa_results/matrix2/ts_full_e8f647e/summary_full.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def build_rows(run_dir: Path, ts_root: Path) -> tuple[list, list, list]:
    sys.path.insert(0, str(ts_root))
    from tool_sandbox.cli.utils import resolve_scenarios
    from tool_sandbox.common.execution_context import ExecutionContext
    from tool_sandbox.common.tool_discovery import ToolBackend

    name_to_scenario = resolve_scenarios(None, ToolBackend.DEFAULT)
    rows: list[dict] = []
    missing: list[str] = []       # killed/crashed mid-play (no conversation.json)
    replay_fail: list[tuple] = []
    for d in sorted((run_dir / "trajectories").iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not (d / "conversation.json").exists():
            missing.append(name)
            continue
        ec_path = d / "execution_context.json"
        scenario = name_to_scenario.get(name)
        if scenario is None or not ec_path.exists():
            replay_fail.append((name, "no scenario definition or no execution_context.json"))
            continue
        try:
            ctx = ExecutionContext.from_dict(
                json.loads(ec_path.read_text(encoding="utf-8")))
            er = scenario.evaluation.evaluate(
                execution_context=ctx, max_turn_count=scenario.max_messages)
            rows.append({
                "name": name,
                "categories": [str(c) for c in scenario.categories],
                "traceback": None,
                "exception_type": None,
                "milestone_similarity": er.milestone_similarity,
                "minefield_similarity": er.minefield_similarity,
                "similarity": er.similarity,
                "turn_count": er.turn_count,
            })
        except Exception:
            replay_fail.append((name, traceback.format_exc(limit=3)))
    return rows, missing, replay_fail


_WORKER_STATE: dict = {}


def _eval_one(name: str) -> dict:
    """Pool worker: evaluate a single scenario dir (init via _pool_init)."""
    from tool_sandbox.common.execution_context import ExecutionContext
    n2s = _WORKER_STATE["n2s"]
    d = _WORKER_STATE["traj"] / name
    ec_path = d / "execution_context.json"
    scenario = n2s.get(name)
    if scenario is None or not ec_path.exists():
        return {"name": name, "error": "no scenario definition or no execution_context.json"}
    try:
        ctx = ExecutionContext.from_dict(
            json.loads(ec_path.read_text(encoding="utf-8")))
        er = scenario.evaluation.evaluate(
            execution_context=ctx, max_turn_count=scenario.max_messages)
        return {
            "name": name,
            "categories": [str(c) for c in scenario.categories],
            "traceback": None,
            "exception_type": None,
            "milestone_similarity": er.milestone_similarity,
            "minefield_similarity": er.minefield_similarity,
            "similarity": er.similarity,
            "turn_count": er.turn_count,
        }
    except Exception:
        return {"name": name, "error": traceback.format_exc(limit=3)}


def _pool_init(ts_root: str, traj: str, polars_threads: int) -> None:
    # cap polars threads BEFORE the tool_sandbox import pulls polars in —
    # otherwise workers x all-cores oversubscribes the machine
    os.environ["POLARS_MAX_THREADS"] = str(polars_threads)
    sys.path.insert(0, ts_root)
    from tool_sandbox.cli.utils import resolve_scenarios
    from tool_sandbox.common.tool_discovery import ToolBackend
    _WORKER_STATE["n2s"] = resolve_scenarios(None, ToolBackend.DEFAULT)
    _WORKER_STATE["traj"] = Path(traj)


def build_rows_parallel(run_dir: Path, ts_root: Path, workers: int) -> tuple[list, list, list]:
    """Multiprocess replay — the milestone matcher is O(snapshots^2) on the
    handful of multi-thousand-turn scenarios, so a sequential pass can take
    hours; the original CLI runs scenarios in a pool for the same reason."""
    import multiprocessing as mp
    traj = run_dir / "trajectories"
    scored, missing = [], []
    for d in sorted(traj.iterdir()):
        if not d.is_dir():
            continue
        (scored if (d / "conversation.json").exists() else missing).append(d.name)
    ncpu = os.cpu_count() or 8
    polars_threads = max(2, ncpu // (workers * 2))
    ctx = mp.get_context("spawn")
    rows: list[dict] = []
    replay_fail: list[tuple] = []
    with ctx.Pool(workers, initializer=_pool_init,
                  initargs=(str(ts_root), str(traj), polars_threads)) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_one, scored, chunksize=1), 1):
            if i % 25 == 0:
                print(f"[parallel] {i}/{len(scored)} scenarios evaluated", flush=True)
            if "error" in r:
                replay_fail.append((r["name"], r["error"]))
            else:
                rows.append(r)
    rows.sort(key=lambda r: r["name"])
    return rows, missing, replay_fail


def validate(run_dir: Path, ts_root: Path) -> int:
    """Replay a COMPLETED run and compare with its recorded result_summary."""
    rows, missing, fails = build_rows(run_dir, ts_root)
    recorded = json.loads((run_dir / "result_summary.json").read_text())
    rec = {r["name"]: r for r in recorded.get("per_scenario_results") or []}
    got = {r["name"]: r for r in rows}
    bad = 0
    for name in sorted(set(rec) | set(got)):
        r1, r2 = rec.get(name), got.get(name)
        if r1 is None or r2 is None:
            print(f"ONLY-ONE-SIDE {name}: recorded={r1 is not None} replayed={r2 is not None}")
            bad += 1
            continue
        for k in ("similarity", "milestone_similarity", "minefield_similarity", "turn_count"):
            if r1.get(k) != r2.get(k):
                print(f"MISMATCH {name}.{k}: recorded={r1.get(k)} replayed={r2.get(k)}")
                bad += 1
    for name, tb in fails:
        print(f"REPLAY-FAIL {name}: {tb}")
        bad += 1
    print(f"validate: {len(got)} replayed vs {len(rec)} recorded, "
          f"{len(missing)} without conversation.json, mismatches={bad}")
    return 0 if bad == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="agent_<user>_<agent>_<ts> output dir (holds trajectories/)")
    parser.add_argument("--ts-root", type=Path, required=True,
                        help="ToolSandbox checkout (repo root)")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel evaluation workers (milestone matching is "
                             "quadratic in snapshots; >1 recommended for big runs)")
    parser.add_argument("--arm", default=None, help="arm name for the summary file")
    parser.add_argument("--n-total", type=int, default=None,
                        help="planned scenario count (terminal-state accounting)")
    parser.add_argument("--out-summary", type=Path, default=None,
                        help="write aggregate summary json here (sg_harvest format)")
    parser.add_argument("--out-rows", type=Path, default=None,
                        help="write per-scenario replayed rows json here")
    args = parser.parse_args()

    if args.validate:
        raise SystemExit(validate(args.run_dir, args.ts_root))

    rows, missing, fails = build_rows(args.run_dir, args.ts_root) if args.workers <= 1 \
        else build_rows_parallel(args.run_dir, args.ts_root, args.workers)
    n_total = args.n_total or (len(rows) + len(missing) + len(fails))
    never_started = n_total - len(rows) - len(missing) - len(fails)
    print(f"replayed {len(rows)} scenarios; dir-without-conversation {len(missing)}; "
          f"replay-fail {len(fails)}; never-started {never_started}; planned {n_total}")
    if fails:
        for name, tb in fails[:5]:
            print(f"REPLAY-FAIL {name}: {tb}")
        raise SystemExit("FATAL: replay failures — scored set is not clean")
    if len(rows) + len(missing) + len(fails) > n_total:
        raise SystemExit(
            f"FATAL: terminal-state check failed: {len(rows)} scored + "
            f"{len(missing)} missing + {len(fails)} failed exceeds planned {n_total}")

    if args.out_rows:
        args.out_rows.write_text(
            json.dumps({"per_scenario_results": rows, "missing": missing},
                       indent=2, ensure_ascii=False), encoding="utf-8")

    if args.out_summary:
        repo_bench = Path(__file__).resolve().parent
        sys.path.insert(0, str(repo_bench))
        from metrics import aggregate
        agg = aggregate([{
            "task_id": r["name"],
            "semantic_score": r["similarity"],
            "milestone_similarity": r["milestone_similarity"],
            "minefield_similarity": r["minefield_similarity"],
            "turn_count": r["turn_count"],
        } for r in rows], cluster_key="task_id")
        agg.update({
            "benchmark": "toolsandbox",
            "arm": args.arm,
            "partial": True,
            "n_total": n_total,
            "n_missing_killed": len(missing) + never_started,
            "n_missing_no_conversation": len(missing),
            "n_never_started": never_started,
            "salvage": {
                "method": "offline re-evaluation (ExecutionContext.from_dict + "
                          "scenario.evaluation.evaluate) of trajectories left by the "
                          "killed tool_sandbox CLI; scored = dirs with conversation.json",
                "run_dir": str(args.run_dir),
                "killed": "2026-09-06 ~06:00 +0800, user decision (marathon degenerate "
                          "loops, zero completions for 34h)",
                "per_scenario_rows": str(args.out_rows) if args.out_rows else None,
            },
            "doc_packing": "turn",
            "max_doc_length": 512,
            "max_doc_num": 12,
            "request_log": "logs/proxy_full_34305.jsonl",
        })
        args.out_summary.write_text(
            json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: agg[k] for k in
                          ("n", "semantic_score", "semantic_score_ci95",
                           "n_total", "n_missing_killed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
