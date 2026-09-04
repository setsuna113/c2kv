"""Harvest SGLang-matrix results into the report-5.4 table format.

Aggregates (all under the server's home, run ON the server):
* tau2 arms   : ~/benchmarks/tau2/data/simulations/sg_tau2_<arm>[_a|_b]
                chunk pairs are unioned (chunked runs share one arm);
                mean official reward + perfect count + termination mix
* TS arms     : ~/bench_results/task_sg_ts_<arm>/*/result_summary.json
                mean similarity per scenario
* BFCL arms   : ~/bench_results/bfcl_archive/sg_bfcl_<arm>/*_score.json
                official accuracy + scorer total_count
* cost columns: ~/bench_logs/proxy_task_task_sg_*.jsonl
                kv_resident/pool (sglang_runtime), gist/original ledger,
                wall p50/p90

Output: markdown table + json (--json-out). Chunks with void_/infra-only
content are flagged, never silently averaged.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import Counter
from pathlib import Path

HOME = Path.home()
TAU2 = HOME / "benchmarks/tau2/data/simulations"
TS = HOME / "bench_results"
BFCL = HOME / "bench_results/bfcl_archive"
PROXY = HOME / "bench_logs"

TAU2_ARMS = ["full", "c2kv", "c2kv_rec", "hy3_rec", "c2kv_rp", "hy3_rp"]
TS_ARMS = ["full", "c2kv", "c2kv_rec", "hy3_rec", "c2kv_rp", "hy3_rp"]
BFCL_ARMS = ["full", "c2kv", "c2kv_rec", "hy3_rec", "c2kv_rp", "hy3_rp"]


def tau2_arm(arm: str) -> dict | None:
    sims, notes = [], []
    import glob as _glob
    names = [f"sg_tau2_{arm}", f"sg_tau2_{arm}_a", f"sg_tau2_{arm}_b",
             f"sg_tau2_{arm}_a2"]  # a2 = post-fix rerun of a voided chunk
    # single-task makeup runs (terminal-gate rejects rerun solo) merge in
    names += [Path(x).name for x in
              _glob.glob(str(TAU2 / f"sg_makeup_{arm}_[0-9]*"))]
    for name in names:
        d = TAU2 / name
        if not (d / "updated_results.json").exists():
            continue
        data = json.loads((d / "updated_results.json").read_text())
        sims.extend(data.get("simulations") or [])
        notes.append(name)
    if not sims:
        return None
    rewards = [(s.get("reward_info") or {}).get("reward") for s in sims]
    vals = [r for r in rewards if r is not None]
    term = Counter(s.get("termination_reason") for s in sims)
    return {
        "runs": notes, "n": len(sims),
        "mean_reward": round(st.mean(vals), 4) if vals else None,
        "perfect": sum(1 for v in vals if v == 1.0),
        "infra_errors": term.get("infrastructure_error", 0),
        "termination": dict(term),
    }


def ts_arm(arm: str) -> dict | None:
    summaries = sorted(glob.glob(str(TS / f"task_sg_ts_{arm}" / "*" / "result_summary.json")))
    if not summaries:
        return None
    rows = []
    for path in summaries:
        data = json.loads(Path(path).read_text())
        rows.extend(data.get("per_scenario_results") or [])
    sims = [r.get("similarity") for r in rows if r.get("similarity") is not None]
    return {
        "n": len(rows), "mean_similarity": round(st.mean(sims), 4) if sims else None,
        "errors": sum(1 for r in rows if r.get("traceback")),
    }


def bfcl_arm(arm: str) -> dict | None:
    hits = sorted(glob.glob(str(BFCL / f"sg_bfcl_{arm}" / "*_score.json")))
    if not hits:
        return None
    rows = [json.loads(l) for l in open(hits[-1]) if l.strip()]
    hdr = rows[0]
    return {
        "accuracy": hdr.get("accuracy"),
        "correct": hdr.get("correct_count"),
        "scorer_total": hdr.get("total_count"),
        "failure_rows": len(rows) - 1,
    }


def costs(arm: str) -> dict | None:
    # EXACT names only (audit: the old `*_{arm}*` glob absorbed sibling
    # arms -- c2kv pulled in c2kv_rec/c2kv_rp, full pulled in cd_full --
    # double-counted chunk files and mixed TS/BFCL logs into the "tau2
    # full-run" wall percentiles)
    names = [f"proxy_task_task_sg_tau2_{arm}",
             f"proxy_task_task_sg_tau2_{arm}_a",
             f"proxy_task_task_sg_tau2_{arm}_b",
             f"proxy_task_task_sg_tau2_{arm}_a2"]
    names += [Path(x).stem for x in
              glob.glob(str(PROXY / f"proxy_task_task_sg_makeup_{arm}_[0-9]*.jsonl"))]
    rows = []
    for name in names:
        path = PROXY / f"{name}.jsonl"
        if path.exists():
            rows.extend(json.loads(l) for l in open(path) if l.strip())
    if not rows:
        return None
    ok = [r for r in rows if r.get("status") == "ok"]
    walls = sorted(r["wall_sec"] for r in ok if isinstance(r.get("wall_sec"), (int, float)))
    kv = [r["kv_resident_tokens"] for r in ok if isinstance(r.get("kv_resident_tokens"), int)]
    gist = sum(r.get("gist_tokens") or 0 for r in ok)
    orig = sum(r.get("original_tokens") or 0 for r in ok)
    out = {
        "n_requests": len(rows), "n_ok": len(ok),
        "errors": len(rows) - len(ok),
        "wall_p50": round(walls[len(walls) // 2], 2) if walls else None,
        "wall_p90": round(walls[int(.9 * len(walls))], 2) if walls else None,
        "compression_logical_over_gist": round(orig / gist, 2) if gist else None,
    }
    if kv:
        out["kv_resident_p50"] = st.median(kv)
    return out


MATRIX2_ARMS = ["full", "c2kv", "hybrid",
                "hiagent", "acon_hist", "acon_obs"]
MATRIX2 = HOME / "bsa_results/matrix2"


def matrix2_harvest(json_out: str | None = None) -> dict:
    """Harvest the post-audit-fix matrix2 chain: summary_*.json per run +
    BFCL score dirs (handler c2kv-<dashed arm>) + textarm consumers
    (degenerate rate, compressor usage) from the request logs."""
    report: dict = {}
    print("| benchmark | arm | n | metric | textarm note |")
    print("|---|---|---|---|---|")
    for arm in MATRIX2_ARMS:
        dashed = arm.replace("_", "-")
        for bench, prefix in (("tau2", f"tau2_{arm}_CONTAMINATED"),
                              ("ts", f"ts_{arm}")):
            summary = MATRIX2 / prefix / f"summary_{arm}.json"
            if not summary.exists():
                continue
            d = json.loads(summary.read_text())
            ta = d.get("textarm_summary") or {}
            note = ""
            if ta:
                note = (f"degenerate {ta.get('degenerate_requests')}/"
                        f"{ta.get('textarm_requests')}; compressor "
                        f"{ta.get('compressor_calls')} calls "
                        f"({ta.get('compressor_prompt_tokens')}+"
                        f"{ta.get('compressor_completion_tokens')} tok)")
            metric = (f"reward {d.get('semantic_score')} ci{d.get('semantic_score_ci95')}"
                      if bench == "tau2" else f"sim {d.get('semantic_score')}")
            label = "τ²†CONTAMINATED" if bench == "tau2" else "TS"
            print(f"| {label} | {arm} | {d.get('n')} | {metric} | {note} |")
            report.setdefault(bench, {})[arm] = d
        # bfcl_eval writes scores to <gorilla-root>/score/<model>/... ; the
        # result/<model>/score/... layout only exists in some archive
        # copies — probe BOTH (audit: reading only the latter made BFCL
        # rows silently vanish from the matrix2 table)
        gorr = HOME / ("benchmarks/gorilla/berkeley-function-call-leaderboard")
        score_files = sorted(
            (gorr / f"score/c2kv-{dashed}/multi_turn").glob("*score*.json"))
        score_files += sorted(
            (gorr / f"result/c2kv-{dashed}/score/multi_turn").glob("*score*.json"))
        result_dir = gorr / f"result/c2kv-{dashed}/multi_turn"
        if not score_files and result_dir.exists():
            raise SystemExit(
                f"FATAL: BFCL result rows exist for c2kv-{dashed} but no "
                f"score file was found (score/ and result/*/score/ both "
                f"empty) — the run scored nothing")
        for f in score_files:
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if d.get("total_count") is not None:
                    print(f"| BFCL | {arm} | {d['total_count']} | "
                          f"acc {round(d['accuracy'], 4)} "
                          f"({d['correct_count']}/{d['total_count']}) | |")
                    report.setdefault("bfcl", {})[arm] = d
    if json_out:
        Path(json_out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--matrix2", action="store_true",
                        help="harvest the post-audit matrix2 chain instead "
                             "of the Sep-2 sg matrix")
    args = parser.parse_args()
    if args.matrix2:
        matrix2_harvest(args.json_out)
        return
    report = {"tau2": {}, "ts": {}, "bfcl": {}, "costs": {}}
    print("| benchmark | arm | n | metric | infra/errors | note |")
    print("|---|---|---|---|---|---|")
    for arm in TAU2_ARMS:
        r = tau2_arm(arm)
        if r:
            report["tau2"][arm] = r
            flag = " ⚠infra=%d" % r["infra_errors"] if r["infra_errors"] else ""
            print(f"| τ² | {arm} | {r['n']} | reward {r['mean_reward']} "
                  f"(perfect {r['perfect']}) | {r['infra_errors']}{flag} | {'+'.join(r['runs'])} |")
    for arm in TS_ARMS:
        r = ts_arm(arm)
        if r:
            report["ts"][arm] = r
            print(f"| TS | {arm} | {r['n']} | sim {r['mean_similarity']} | {r['errors']} | |")
    for arm in BFCL_ARMS:
        r = bfcl_arm(arm)
        if r:
            report["bfcl"][arm] = r
            print(f"| BFCL | {arm} | {r['scorer_total']} | acc {r['accuracy']} "
                  f"({r['correct']}/{r['scorer_total']}) | | scorer-caliber |")
    print("\n| arm | requests | ok/err | wall p50/p90 | logical/gist | kv_resident p50 |")
    print("|---|---|---|---|---|---|")
    for arm in TAU2_ARMS:
        c = costs(arm)
        if c:
            report["costs"][arm] = c
            print(f"| {arm} | {c['n_requests']} | {c['n_ok']}/{c['errors']} "
                  f"| {c['wall_p50']}/{c['wall_p90']}s "
                  f"| {c['compression_logical_over_gist']}x "
                  f"| {c.get('kv_resident_p50')} |")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
