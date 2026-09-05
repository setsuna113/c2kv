# -*- coding: utf-8 -*-
"""t34 scoring glue: merge the t34 feature files and orientations, then run
the (runner-maintained) ``t33_score`` winner table on them.

RUNBOOK (local or server, zero GPU)
-----------------------------------
1. Merge every t34 feature file per arm into one frame (join on qid, refuse
   duplicate column names across files):

     python agent/t34_score.py merge --arm c2kv \\
         --inputs results/t34/features_attn_c2kv.jsonl results/t34/features_l1.jsonl ... \\
         --out results/t34/features_t34_c2kv.jsonl
     python agent/t34_score.py merge --arm full  --inputs ... --out results/t34/features_t34_full.jsonl

2. Score with t33_score, with the t34 orientations merged in
   (configs/t34/orientations_*.json are the per-unit pre-declared
   directions; a feature absent from every file is scored with orientation
   +1 by t33_score's default — the merge step warns about those):

     python agent/t34_score.py score --features_c2kv results/t34/features_t34_c2kv.jsonl \\
         --features_full results/t34/features_t34_full.jsonl \\
         --battery_full results/bdf_pilot/d_r2/battery_full.jsonl \\
         --battery_c2kv results/bdf_pilot/d_r2/battery_c2kv.jsonl \\
         --manifest configs/bdf_pilot/d_cw_manifest_r2.json --out_dir results/t34/score

3. Re-derive the verdict column with the prevalence-aware rule
   (``verdict_prevalence_aware``) — a reference implementation of the §4.0
   winner rule with the chance level taken from the evaluation frame, the
   length-control clause enforced as "residualised AP above chance AND not
   below the raw AP by more than `len_tolerance`", and the uncensored clause
   as "direction unchanged" (AUROC on the uncensored slice > 0.5):

     python agent/t34_score.py reverdict --winner_table results/t34/score/winner_table.json \\
         --out results/t34/score/winner_table_t34verdict.json

Notes
-----
* ``t33_score`` is owned by the runner (it is being fixed for the §4.0-4.4
  audit findings); this module never edits it, only calls it.
* Locators are NOT scored here — see agent/t34_locate_score.py (S@k table).
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from t34_common import META_COLS, guard_columns, load_jsonl, prevalence  # noqa: E402


SHARED_ORIENTATIONS_FILE = "orientations_shared.json"
_LAST_ORIENTATION_REPORT: Dict[str, Any] = {}


def orientation_report() -> Dict[str, Any]:
    """Details of the last merged_orientations() call: shared keys that a
    unit file redeclared (ignored, shared file wins) and per-key sources."""
    return dict(_LAST_ORIENTATION_REPORT)


def merged_orientations(config_dir: Optional[Path] = None) -> Dict[str, int]:
    """Union of configs/t34/orientations_*.json.

    ``orientations_shared.json`` is authoritative for the shared control
    scalars it lists; a unit file redeclaring one of those keys is IGNORED for
    that key (recorded in :func:`orientation_report`).  Conflicts between two
    unit files on a non-shared key are errors."""
    global _LAST_ORIENTATION_REPORT
    d = Path(config_dir) if config_dir else _HERE.parent / "configs/t34"
    out: Dict[str, int] = {}
    src: Dict[str, str] = {}
    shared: Dict[str, int] = {}
    shared_path = d / SHARED_ORIENTATIONS_FILE
    if shared_path.exists():
        for k, v in json.loads(shared_path.read_text(encoding="utf-8")).items():
            if not k.startswith("_") and isinstance(v, int) and not isinstance(v, bool):
                shared[k] = int(v)
    redeclared: List[Dict[str, Any]] = []
    for p in sorted(glob.glob(str(d / "orientations_*.json"))):
        if Path(p).name == SHARED_ORIENTATIONS_FILE:
            continue
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        for k, v in data.items():
            if k in shared:
                try:
                    if int(v) != shared[k]:
                        redeclared.append({"key": k, "file": Path(p).name, "unit_value": int(v),
                                           "shared_value": shared[k]})
                except (TypeError, ValueError):
                    pass
                continue
            # metadata entries ("_description", "_rationale", ...) and any
            # non-integer value are documentation, not an orientation
            if k.startswith("_") or isinstance(v, bool) or not isinstance(v, (int, float, str)):
                continue
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
            if v not in (-1, 0, 1):
                raise ValueError(f"{p}: orientation for {k} must be -1/0/1, got {v}")
            if k in out and out[k] != v:
                raise ValueError(f"orientation conflict for {k}: {src[k]}={out[k]} vs {p}={v}")
            out[k] = v
            src[k] = p
    out.update(shared)
    for k in shared:
        src[k] = str(shared_path)
    _LAST_ORIENTATION_REPORT = {"n_shared": len(shared), "redeclared_shared_keys": redeclared,
                                "sources": src}
    return out


def merge_feature_files(paths: Sequence[str], arm: str) -> List[Dict[str, Any]]:
    """Join per-qid feature rows across files; duplicate non-meta columns
    across files are refused (two units must not emit the same name)."""
    by_qid: Dict[str, Dict[str, Any]] = {}
    seen_cols: Dict[str, str] = {}
    for p in paths:
        rows = load_jsonl(p)
        cols = {k for r in rows for k in r} - META_COLS
        for c in cols:
            if c in seen_cols and seen_cols[c] != p:
                raise ValueError(f"column {c!r} emitted by both {seen_cols[c]} and {p}")
            seen_cols[c] = p
        for r in rows:
            if r.get("arm") not in (None, arm):
                raise ValueError(f"{p}: row {r.get('qid')} has arm={r.get('arm')!r}, expected {arm!r}")
            tgt = by_qid.setdefault(r["qid"], {"qid": r["qid"], "arm": arm})
            for k, v in r.items():
                if k in META_COLS:
                    continue
                tgt[k] = v
    guard_columns(sorted(seen_cols), context=f"t34 merged features[{arm}]")
    return [by_qid[q] for q in sorted(by_qid)]


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def verdict_prevalence_aware(entry: Dict[str, Any], *, prevalence_all: float,
                             prevalence_uncensored: Optional[float] = None,
                             len_tolerance: float = 0.05) -> Dict[str, Any]:
    """§4.0 winner rule with the chance level of the EVALUATION frame.

    Clauses (all must hold):
      beats_baseline : coverage > baseline coverage AND precision > baseline
                       precision AND false_resets <= baseline false_resets at
                       the matched fire count (as t33_score computes them);
      above_chance   : AUPRC CI lower bound > prevalence of the frame;
      s0             : ΔAUPRC-vs-S0 CI lower bound > 0 (undefined S0 => the
                       clause is 'undefined', not passed);
      length         : residualised AP > prevalence AND
                       raw AP − residualised AP <= len_tolerance;
      uncensored     : AUROC on the uncensored slice > 0.5 when available,
                       else AP(uncensored) > prevalence_uncensored.
    Returns the per-clause booleans and the verdict string.
    """
    ap_ci = entry.get("auprc_ci") or [None, None]
    d = entry.get("delta_vs_s0") or {}
    base = entry.get("parse_baseline") or {}
    op = entry.get("matched_rate_op") or {}
    ap = entry.get("auprc")
    ap_len = entry.get("auprc_length_controlled")
    ap_unc = entry.get("auprc_uncensored")
    auroc_unc = entry.get("auroc_uncensored")
    clauses = {
        "beats_baseline": bool(op) and bool(base)
        and (op.get("coverage") or 0) > (base.get("coverage") or 0)
        and (op.get("precision") or 0) > (base.get("precision") or 0)
        and (base.get("false_resets") or 0) >= (op.get("false_resets") or 0),
        "above_chance": ap_ci[0] is not None and ap_ci[0] > prevalence_all,
        "s0": (d.get("ci_lo") is not None and d.get("ci_lo") > 0) if d else None,
        "length": ap_len is not None and ap is not None
        and ap_len > prevalence_all and (ap - ap_len) <= len_tolerance,
        "uncensored": (auroc_unc > 0.5) if auroc_unc is not None
        else (ap_unc is not None and prevalence_uncensored is not None and ap_unc > prevalence_uncensored),
    }
    live = all(v is True for v in clauses.values())
    verdict = "LIVE" if live else ("s0-undefined" if clauses["s0"] is None and all(
        v is True for k, v in clauses.items() if k != "s0") else "not-live")
    return {"clauses": clauses, "verdict": verdict, "chance_level": prevalence_all}


def _cmd_merge(args: argparse.Namespace) -> int:
    rows = merge_feature_files(args.inputs, args.arm)
    write_jsonl(Path(args.out), rows)
    ori = merged_orientations()
    cols = sorted({k for r in rows for k in r} - META_COLS)
    missing = [c for c in cols if c not in ori]
    print(json.dumps({"n_rows": len(rows), "n_cols": len(cols),
                      "cols_without_declared_orientation": missing,
                      "redeclared_shared_keys": orientation_report().get("redeclared_shared_keys")},
                     ensure_ascii=False))
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    import t33_score  # runner-owned
    t33_score.ORIENTATIONS.update(merged_orientations())
    argv = ["--features_c2kv", args.features_c2kv, "--features_full", args.features_full,
            "--battery_full", args.battery_full, "--battery_c2kv", args.battery_c2kv,
            "--manifest", args.manifest, "--out_dir", args.out_dir]
    if args.rows_c2kv:
        argv += ["--rows_c2kv", args.rows_c2kv]
    return int(t33_score.main(argv) or 0)


def _cmd_reverdict(args: argparse.Namespace) -> int:
    wt = json.loads(Path(args.winner_table).read_text(encoding="utf-8"))
    n_pos = int((wt.get("census") or {}).get("n_cw") or 0)
    n_neg = int((wt.get("census") or {}).get("n_cc") or 0)
    prev = n_pos / (n_pos + n_neg) if (n_pos + n_neg) else float("nan")
    cens = wt.get("census") or {}
    unc_pos = n_pos - int(cens.get("censored_cw") or 0)
    unc_neg = n_neg - int(cens.get("censored_cc") or 0)
    prev_unc = unc_pos / (unc_pos + unc_neg) if (unc_pos + unc_neg) else None
    out_entries = []
    n_live = 0
    for e in wt.get("entries") or []:
        v = verdict_prevalence_aware(e, prevalence_all=prev, prevalence_uncensored=prev_unc)
        e2 = dict(e)
        e2["t34_verdict"] = v["verdict"]
        e2["t34_clauses"] = v["clauses"]
        n_live += v["verdict"] == "LIVE"
        out_entries.append(e2)
    result = {"chance_level": prev, "chance_level_uncensored": prev_unc,
              "n_live_t34": n_live, "n_live_t33": sum(1 for e in wt.get("entries") or [] if e.get("verdict") == "LIVE"),
              "entries": out_entries}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "entries"}))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--arm", required=True, choices=["c2kv", "full"])
    m.add_argument("--inputs", nargs="+", required=True)
    m.add_argument("--out", required=True)
    m.set_defaults(fn=_cmd_merge)
    s = sub.add_parser("score")
    for k in ("features_c2kv", "features_full", "battery_full", "battery_c2kv", "manifest", "out_dir"):
        s.add_argument(f"--{k}", required=True)
    s.add_argument("--rows_c2kv", default="")
    s.set_defaults(fn=_cmd_score)
    r = sub.add_parser("reverdict")
    r.add_argument("--winner_table", required=True)
    r.add_argument("--out", required=True)
    r.set_defaults(fn=_cmd_reverdict)
    args = parser.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
