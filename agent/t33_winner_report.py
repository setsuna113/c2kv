# -*- coding: utf-8 -*-
"""t33 final two-criteria winner report (post-fix LIVE re-judgment).

The user-facing table reports EXACTLY two scoring criteria:
  1. complete-case AUPRC (+ n_scored, the subset's own prevalence, session-
     clustered bootstrap CI) — no median fills, no mixed row sets;
  2. same-layer S0 twin — for scalar features the same feature recomputed on
     the full arm; for probes the full-arm refit at the SAME layer the c2kv
     arm selected (never each arm's own best layer).

Winner-rule clauses (parse-baseline three-indicator, prevalence CI, S0
delta, LEN increment, uncensored direction) are applied mechanically for the
verdict column; algebraic duplicates of one score are collapsed into score
families before any counting.

Inputs: winner_table.json + probes_{c2kv,full}.json + prefix_control.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t33_labels import build_label_frame, join_arms, load_jsonl  # noqa: E402
from t33_score import ORIENTATIONS, score_feature, verdict, session_clusters  # noqa: E402


def _oof_vec(entry: Dict[str, Any], qids: List[str]) -> Optional[np.ndarray]:
    v = entry.get("oof_scores")
    if v is None or len(v) != len(qids):
        return None
    arr = np.array([np.nan if x is None else float(x) for x in v], dtype=float)
    return arr


def _inject(entry: Dict[str, Any], qids: List[str], frame2: List[Dict[str, Any]],
            qid_index: Dict[str, int], col: str) -> bool:
    """Place an OOF score vector into the frame.  Vectors are either aligned
    with the probe file's qids list, or a SUBSET aligned via oof_row_indices
    (anchor-complete-case rows) — both handled."""
    v = entry.get("oof_scores")
    if v is None:
        return False
    idxs = entry.get("oof_row_indices")
    if idxs is not None and len(idxs) == len(v):
        pairs = [(qids[i], x) for i, x in zip(idxs, v)]
    elif len(v) == len(qids):
        pairs = zip(qids, v)
    else:
        return False
    placed = False
    for qid, x in pairs:
        i = qid_index.get(qid)
        if i is not None and x is not None:
            frame2[i][col] = float(x)
            placed = True
    return placed


def probe_row(
    name: str, frame: List[Dict[str, Any]], qid_index: Dict[str, int],
    c2kv: Dict[str, Any], full: Dict[str, Any],
    sessions: np.ndarray, labels: np.ndarray, keep: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Adjudicate one probe via the same score_feature/verdict machinery by
    injecting its OOF scores (and the full-arm same-configuration twin) as
    pseudo-features on the trigger frame."""
    cq = c2kv.get("qids") or []
    fq = full.get("qids") or []
    c_entry = c2kv.get(name)
    f_entry = full.get(name) if isinstance(full, dict) else None
    if not isinstance(c_entry, dict):
        return None
    frame2 = [dict(r) for r in frame]
    if not _inject(c_entry, cq, frame2, qid_index, "c::probe_" + name):
        return None
    if isinstance(f_entry, dict):
        _inject(f_entry, fq, frame2, qid_index, "s0::probe_" + name)
    ORIENTATIONS["probe_" + name] = 1
    e = score_feature(frame2, "c::probe_" + name, sessions, labels, keep)
    if e is None:
        return None
    e["feature"] = "probe::" + name
    e["verdict"] = verdict(e)
    return e


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner_table", required=True)
    parser.add_argument("--probes_c2kv", required=True)
    parser.add_argument("--probes_full", required=True)
    parser.add_argument("--prefix_control", default="")
    parser.add_argument("--battery_full", required=True)
    parser.add_argument("--battery_c2kv", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--features_c2kv", required=True,
                        help="features_c2kv.jsonl (n_generated/censored for the probe rows)")
    parser.add_argument("--out_md", required=True)
    parser.add_argument("--out_json", required=True)
    args = parser.parse_args(argv)

    wt = json.loads(Path(args.winner_table).read_text(encoding="utf-8"))
    pc = json.loads(Path(args.probes_c2kv).read_text(encoding="utf-8"))
    pf = json.loads(Path(args.probes_full).read_text(encoding="utf-8"))
    prefix_ctl = json.loads(Path(args.prefix_control).read_text(encoding="utf-8")) \
        if args.prefix_control and Path(args.prefix_control).exists() else None

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    label_frame = build_label_frame(
        join_arms(load_jsonl(args.battery_full), load_jsonl(args.battery_c2kv)), manifest)
    lab = {r["qid"]: r for r in label_frame}

    frame = [{
        "qid": r["qid"], "session_id": r["session_id"], "label_cw": r["label_cw"],
        "parse_fail_fire": r["parse_fail_fire"], "censored": None, "n_generated": None,
    } for r in label_frame]
    feats_c = {r["qid"]: r for r in load_jsonl(args.features_c2kv)}
    for r in frame:
        f = feats_c.get(r["qid"]) or {}
        r["n_generated"] = f.get("n_generated")
        r["censored"] = f.get("censored")
    qid_index = {r["qid"]: i for i, r in enumerate(frame)}
    keep = np.array([r["label_cw"] in (0, 1) for r in frame])
    labels = np.array([1 if r["label_cw"] == 1 else 0 for r in frame])
    sessions = session_clusters([r["session_id"] for r in frame])

    sections: Dict[str, Any] = {}
    lines: List[str] = ["# t33 post-fix winner report (two criteria)", ""]

    # ---- Section A: scalar features (from winner_table, families deduped) --
    entries = [e for e in wt.get("entries", []) if not str(e["feature"]).startswith("c::probe_")]
    live = [e for e in entries if e.get("verdict") == "LIVE"]
    sections["scalar"] = {
        "n_features": len(entries),
        "n_live": len(live),
        "live_features": [e["feature"] for e in live],
        "live_families": sorted({e.get("score_family", e["feature"]) for e in live}),
    }
    lines += ["## A. scalar decode/prefix features (complete-case)", ""]
    lines += [f"features={len(entries)}  LIVE rows={len(live)}  "
              f"LIVE score families={len(sections['scalar']['live_families'])}", ""]
    lines += ["| feature | fam | n | prev | AP [CI] | ΔvsS0 [CI] | AP(len-ctl) vs AP(len) | verdict |",
              "|---|---|---|---|---|---|---|---|"]
    for e in sorted(entries, key=lambda x: -(x.get("auprc") or -1)):
        d = e.get("delta_vs_s0") or {}
        fam = e.get("score_family") or "-"
        same = "" if (fam == "-" or fam == e["feature"]) else " (=dup)"
        lines.append(
            f"| {e['feature']} | {fam}{same} | {e.get('n_scored')} | {e.get('eval_prevalence')} | "
            f"{e.get('auprc')} {e.get('auprc_ci')} | {d.get('point')} {d.get('ci_lo')},{d.get('ci_hi')} | "
            f"{e.get('auprc_length_controlled')} vs {e.get('auprc_len_only')} | {e.get('verdict')} |")

    # ---- Section B: probes with the same-layer twin ----
    cq = pc.get("qids") or []
    sel_layers = (pc.get("probe_prefill_layer_select") or {}).get("selected_layers") or []
    layers_grid = (pc.get("probe_prefill_layer_select") or {}).get("layers_grid") or []
    twin_layer = None
    if sel_layers:
        maj_idx = int(np.bincount([int(x) for x in sel_layers]).argmax())
        # selected_layers index INTO layers_grid (the every5 subset), not
        # absolute layer ids — map back before looking up the full arm
        twin_layer = int(layers_grid[maj_idx]) if maj_idx < len(layers_grid) else maj_idx
    lines += ["", "## B. hidden-state probes (same-layer S0 twin)", ""]
    if twin_layer is not None:
        lines += [f"c2kv layer-select chose grid indices {sel_layers} "
                  f"(= absolute layers {[layers_grid[int(x)] for x in sel_layers]}); twin "
                  f"reported at the majority absolute layer {twin_layer} of the FULL "
                  "arm (never each arm's own best).", ""]

    # same-layer twin for probe_prefill: full per-layer oof at twin_layer
    frame2 = [dict(r) for r in frame]
    ls = pc.get("probe_prefill_layer_select") or {}
    cv = ls.get("oof_scores")
    if cv:
        for qid, val in zip(cq, cv):
            i = qid_index.get(qid)
            if i is not None and val is not None:
                frame2[i]["c::probe_prefill_layer_select"] = float(val)
    fl = (pf.get("probe_prefill_per_layer") or {}).get(str(twin_layer)) if twin_layer is not None else None
    if fl and fl.get("oof_scores"):
        for qid, val in zip(pf.get("qids") or [], fl["oof_scores"]):
            i = qid_index.get(qid)
            if i is not None and val is not None:
                frame2[i]["s0::probe_prefill_layer_select"] = float(val)
    ORIENTATIONS["probe_prefill_layer_select"] = 1

    probe_names = ["tool_call_error_name_last", "tool_call_error_last",
                   "kwts_ensemble", "joint_overflow", "alien_arm_b_cw"]
    probe_entries = []
    e = score_feature(frame2, "c::probe_prefill_layer_select", sessions, labels, keep)
    if e is not None:
        e["feature"] = "probe::probe_prefill_layer_select"
        e["same_layer_twin"] = {"layer": twin_layer,
                                "full_arm_auprc_at_layer": (fl or {}).get("auprc")}
        e["verdict"] = verdict(e)
        probe_entries.append(e)
    for name in probe_names:
        pe = probe_row(name, frame, qid_index, pc, pf, sessions, labels, keep)
        if pe is not None:
            probe_entries.append(pe)

    sections["probes"] = {
        "n": len(probe_entries),
        "live": [e["feature"] for e in probe_entries if e.get("verdict") == "LIVE"],
    }
    lines += [f"probes adjudicated={len(probe_entries)}", ""]
    lines += ["| probe | n | prev | AP [CI] | Δvs twin [CI] | verdict | notes |",
              "|---|---|---|---|---|---|---|"]
    for e in probe_entries:
        d = e.get("delta_vs_s0") or {}
        notes = ""
        if e.get("same_layer_twin"):
            notes = f"twin@L{e['same_layer_twin']['layer']} full AP={e['same_layer_twin']['full_arm_auprc_at_layer']}"
        if e.get("delta_vs_s0") is None:
            notes = (notes + " " if notes else "") + "no twin (one-arm only)"
        lines.append(
            f"| {e['feature']} | {e.get('n_scored')} | {e.get('eval_prevalence')} | "
            f"{e.get('auprc')} {e.get('auprc_ci')} | {d.get('point')} {d.get('ci_lo')},{d.get('ci_hi')} | "
            f"{e.get('verdict')} | {notes} |")

    # ---- Section C: 4.1 arm-invariant / S0-undefined adjudication ----
    lines += ["", "## C. 4.1 arm-invariant / S0-undefined prefix families", ""]
    if prefix_ctl:
        lines += ["S0 twin is an identity-zero (identical features both arms) or undefined "
                  "(no gist pass on the full arm) — adjudicated by increment over the cheap "
                  "baseline (length + parse-failure) instead.", ""]
        lines += ["| family | AP(base) | AP(+fam) | Δ [CI] | perm p | verdict |",
                  "|---|---|---|---|---|---|"]
        for ent in prefix_ctl.get("entries", []):
            ci = ent.get("delta_ci") or [None, None]
            lines.append(
                f"| {ent.get('family')} | {ent.get('baseline_auprc')} | {ent.get('plus_feature_auprc')} | "
                f"{ent.get('delta_auprc')} {ci[0]},{ci[1]} | {ent.get('perm_p')} | {ent.get('verdict')} |")
        sections["prefix_41"] = {
            e.get("family"): e.get("verdict") for e in prefix_ctl.get("entries", [])}
    else:
        lines += ["(prefix_control.json not supplied)"]
        sections["prefix_41"] = None

    # ---- amendments ----
    lines += ["", "## prereg amendments (disclosed)", "",
              "- winner clauses compare against each eval subset's own prevalence; the 900-frame 0.1033 is a reference column only",
              "- complete-case scoring per feature (n_scored + own prevalence); no median fill, no full-sequence fallback for span features",
              "- length control residualizes the ORIENTED score; LEN clause = increment over the LEN-only score",
              "- e-CUSUM estimator rewritten to the prereg definition (session-prefix baseline, causal repeat channel)",
              "- FC-UQ SMT mask includes arg-name tokens (class 3); decision token at position 0 (class 1)",
              "- svip args_first reads the first argument VALUE token; the `{`-position readout kept as _syntax",
              "- probes: layer/anchor/C selection in inner folds; same-layer twin; FPR@90TPR from the same OOF model",
              ]

    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(args.out_json).write_text(json.dumps(sections, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    print(f"[t33] winner report -> {args.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
