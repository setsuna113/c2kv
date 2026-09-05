# -*- coding: utf-8 -*-
"""t33 scoring harness (survey 4.0 winner rule) for scalar features.

Everything here is rank/statistic arithmetic — no sklearn needed.  For every
candidate feature (from t33_extract_features, compressed arm; the full arm's
twin is joined as the S0 control) we report:

  AUPRC / AUROC with session-clustered bootstrap CIs (227 clusters);
  matched-fire-rate operating point vs the parse-failure baseline
  (coverage /93, precision, false-reset /68) plus the full sweep;
  Delta-AUPRC vs the S0 twin on the SAME bootstrap resamples;
  length-controlled variant (rank-residualized on n_generated);
  censored / uncensored stratified AUPRC with direction check;
  e-CUSUM mu0 calibration (per-token pooling over C->C) + isolation controls;
  KnowNo |C| histogram + split-half conformal with achieved Beta bound.

Output: winner_table.{json,md} + curves.json in --out_dir.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t33_labels import build_label_frame, census, join_arms, load_jsonl  # noqa: E402

RNG = np.random.default_rng(20260905)
BOOTSTRAP_REPS = 2000
MDE_PP = (17, 25)

META_COLS = {
    "qid", "arm", "stop_reason", "smt_mask_source", "ecusum_a_seq",
    "n_generated", "censored", "session_id", "label_cw", "parse_fail_fire",
    "z_deferral", "censored_at_cap", "censored_at_cap_full", "generate_sec",
}

# Pre-declared orientations (risk direction), from configs/t33/prereg.md.
#   +1: larger value => MORE likely C->W (risk).  -1: larger => safer.
ORIENTATIONS: Dict[str, int] = {
    # 4.2 uncertainty: lower confidence => risk
    "flare_min_p_all": -1, "flare_min_p_span": -1, "flare_min_p_name": -1,
    "fc_max_nll_all": 1, "fc_avg_nll_all": 1, "fc_gnll_all": 1,
    "fc_max_nll_smt": 1, "fc_avg_nll_smt": 1, "fc_gnll_smt": 1,
    "name_region_nll": 1, "args_region_nll": 1,
    "leyline_margin_name_first": -1, "margin_min_all": -1, "margin_mean_all": -1,
    "p1_name_first": -1,
    "kono_pool_mass_top5": -1, "kono_none_mass": 1, "kono_top_pool_prob": -1,
    "kono_emitted_in_pool": -1,
    # 4.3 entropy: higher uncertainty => risk
    "hbar_all": 1, "hbar_name": 1, "hbar_args": 1, "entropy_name_max": 1,
    "entropy_args_max": 1, "entropy_max_span": 1,
    "entropycache_max_all": 1, "entropycache_max_no_eos": 1, "hbar_no_eos": 1,
    "ergo_dh_region": 1,
    "svip_sqrt_h_name_first": 1, "svip_sqrt_h_args_first": 1, "svip_sqrt_h_argvalue_max": 1,
    "confkv_c_min": -1, "confkv_c_mean": -1, "confkv_c_name": -1,
    "ecusum_u_max": 1, "ecusum_u_mean": 1, "ecusum_a_max": 1, "ecusum_a_mean": 1,
    "ecusum_cusum_s_max": 1, "ecusum_cusum_s_shuf_max": 1,
    # 4.4 IC: lower consistency => risk
    "ic_ic_uniform_name_first": -1, "ic_ic_uniform_name_last": -1,
    "ic_ic_lastk_name_first": -1, "ic_ic_lastk_name_last": -1,
    "ic_first_agree_layer_name_first": 1, "ic_first_agree_layer_name_last": 1,
    "ic_margin_final_name_first": -1, "ic_margin_final_name_last": -1,
    # 4.1 prefix stats: more packing / ratio / drops => risk
    "s8_n_docs_kept": 1, "s8_dropped_docs": 1, "s8_packing_sat": 1,
    "s8_doc_tokens_sum": 1, "s8_n_ctx": 1,
    "boundary_max_doc_len": 1, "boundary_mean_doc_len": 1,
    "boundary_longest_doc_pos_frac": 1,
    "gzip_ratio_mean": 1, "gzip_ratio_min": 1, "gzip_ratio_max": 1,
    "surprise_max_k": -1, "surprise_mean_k": -1, "surprise_hit_rate_mean": -1,
    "rung0_dropped_any": 1, "s8_kept_frac": -1,
    "gist_gists_per_doc_mean": -1, "gist_gists_per_doc_max": -1,
    # saturation family: null-expected control, direction not asserted
    # text surface (s_text): truncation-ish => risk
    "text_n_chars": -1, "text_payload_chars": -1, "text_name_len": 0,
    "text_closed_tag": -1, "text_parse_ok": -1, "text_has_tool_call": 0,
    "text_digit_frac": 0, "text_brace_count": 0, "text_distinct_word_frac": 0,
}


# ------------------------------------------------------------------ metrics

def auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """Average precision (step interpolation off; sklearn-equivalent AP)."""
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(1, tp + fp)
    n_pos = int(y.sum())
    if n_pos == 0 or len(y) == n_pos:
        return None
    return float((precision * y).sum() / n_pos)


def session_clusters(sessions: Sequence[str]) -> np.ndarray:
    uniq = {s: i for i, s in enumerate(sorted(set(sessions)))}
    return np.array([uniq[s] for s in sessions])


def clustered_bootstrap(
    scores: np.ndarray, labels: np.ndarray, clusters: np.ndarray,
    metric_fn=auprc, reps: int = BOOTSTRAP_REPS, rng: RNG = RNG,
) -> Tuple[Optional[float], Optional[float]]:
    uniq = np.unique(clusters)
    vals = []
    n_pos_total = int(labels.sum())
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in pick])
        if int(labels[idx].sum()) == 0 or len(idx) - int(labels[idx].sum()) == 0:
            continue
        v = metric_fn(scores[idx], labels[idx])
        if v is not None:
            vals.append(v)
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_delta_bootstrap(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray, clusters: np.ndarray,
    reps: int = BOOTSTRAP_REPS, rng: RNG = RNG,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """CI of AUPRC(a) - AUPRC(b) on the SAME resamples (S0 delta)."""
    uniq = np.unique(clusters)
    point = (auprc(scores_a, labels) or 0.0) - (auprc(scores_b, labels) or 0.0)
    deltas = []
    for _ in range(reps):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(clusters == c)[0] for c in pick])
        if int(labels[idx].sum()) == 0:
            continue
        va, vb = auprc(scores_a[idx], labels[idx]), auprc(scores_b[idx], labels[idx])
        if va is not None and vb is not None:
            deltas.append(va - vb)
    if not deltas:
        return point, None, None
    return point, float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def rank_residualize(x: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Length control: residual of x's ranks on control's ranks (linear)."""
    rx = np.argsort(np.argsort(x)).astype(float)
    rc = np.argsort(np.argsort(control)).astype(float)
    slope, intercept = np.polyfit(rc, rx, 1)
    return rx - (slope * rc + intercept)


def operating_point(scores: np.ndarray, labels: np.ndarray, thresh: float) -> Dict[str, Any]:
    fire = scores >= thresh
    n_fire = int(fire.sum())
    cov = int((fire & (labels == 1)).sum())
    fr = int((fire & (labels == 0)).sum())
    return {
        "threshold": float(thresh),
        "fires": n_fire,
        "coverage": cov, "coverage_of_93": round(cov / max(1, int(labels.sum())), 4),
        "false_resets": fr, "false_reset_of_68": round(fr / max(1, int((labels == 0).sum())), 4),
        "precision": round(cov / n_fire, 4) if n_fire else None,
    }


def sweep(scores: np.ndarray, labels: np.ndarray, n_points: int = 25) -> List[Dict[str, Any]]:
    qs = np.linspace(0.0, 1.0, n_points)
    lo, hi = float(np.nanmin(scores)), float(np.nanmax(scores))
    if lo == hi:
        return []
    thresholds = lo + qs * (hi - lo)
    return [operating_point(scores, labels, t) for t in thresholds]


# ------------------------------------------------------------------ e-CUSUM

def ecusum_finalize(frame_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """mu0 = 90th percentile of per-token a_t pooled over C->C rows; then
    S_n per row; plus the shuffled-order isolation control."""
    cc_seqs = [r["ecusum_a_seq"] for r in frame_rows if r.get("label_cw") == 0 and r.get("ecusum_a_seq")]
    all_seqs = [r.get("ecusum_a_seq") for r in frame_rows]
    if not cc_seqs:
        return {}
    pooled = np.concatenate([np.asarray(s) for s in cc_seqs])
    mu0 = float(np.percentile(pooled, 90))
    lam = 0.5

    def cusum(seq: Sequence[float]) -> float:
        s = 0.0
        best = 0.0
        for a in seq:
            s = max(0.0, s + math.log1p(lam * (a - mu0)))
            best = max(best, s)
        return best

    rng = np.random.default_rng(7)
    for r in frame_rows:
        seq = r.get("ecusum_a_seq")
        if not seq:
            continue
        r["ecusum_cusum_s_max"] = cusum(seq)
        shuffled = list(seq)
        rng.shuffle(shuffled)
        r["ecusum_cusum_s_shuf_max"] = cusum(shuffled)
    return {"mu0": mu0, "lam": lam, "n_cc_token_pool": int(pooled.size)}


# ------------------------------------------------------------------ KnowNo

def knono_report(frame_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in frame_rows if r.get("kono_pool_mass_top5") is not None]
    if not rows:
        return {}
    # |C| histogram at a swept q-hat; split-half session-grouped calibration
    sessions = sorted({r["session_id"] for r in rows})
    rng = np.random.default_rng(11)
    half = set(rng.choice(sessions, size=max(1, len(sessions) // 2), replace=False))
    cal = [r for r in rows if r["session_id"] in half]
    ev = [r for r in rows if r["session_id"] not in half]
    out: Dict[str, Any] = {"n_rows": len(rows), "top5_truncation": True}
    # kappa on calibration: 1 - f_hat of the emitted token is degenerate
    # without full pool probabilities; we report the pool-mass distribution
    # and the |C| histogram under a pool-mass threshold proxy.
    masses = np.array([r["kono_pool_mass_top5"] for r in rows])
    out["pool_mass_quantiles"] = {
        str(q): round(float(np.percentile(masses, q)), 4) for q in (10, 50, 90)
    }
    sizes = []
    for r in ev:
        m = r["kono_pool_mass_top5"]
        sizes.append(1 + int(m < 0.9))  # degenerate |C| proxy; documented
    out["c_size_histogram_ev"] = {str(v): sizes.count(v) for v in sorted(set(sizes))}
    out["note"] = ("top-5 truncated: full pool renormalization needs top_logprobs "
                   "over the pool; |C| is a pool-mass proxy, conformal numbers "
                   "are diagnostic only per prereg")
    return out


# ------------------------------------------------------------------ frame

def build_frame(
    features_c2kv: List[Dict[str, Any]],
    features_full: List[Dict[str, Any]],
    label_frame: List[Dict[str, Any]],
    rows_c2kv: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_qid_c = {r["qid"]: r for r in features_c2kv}
    by_qid_f = {r["qid"]: r for r in features_full}
    by_qid_lab = {r["qid"]: r for r in label_frame}
    by_qid_row = {r["qid"]: r for r in rows_c2kv}
    frame = []
    for qid, lab in by_qid_lab.items():
        c = by_qid_c.get(qid)
        f = by_qid_f.get(qid)
        if c is None or f is None:
            continue
        row: Dict[str, Any] = {
            "qid": qid,
            "session_id": lab["session_id"],
            "label_cw": lab["label_cw"],
            "parse_fail_fire": lab["parse_fail_fire"],
            "censored": c.get("censored"),
            "n_generated": c.get("n_generated"),
            "generate_sec": (by_qid_row.get(qid) or {}).get("generate_sec"),
        }
        for k, v in c.items():
            if k not in META_COLS and v is not None and not isinstance(v, (list, dict)):
                row[f"c::{k}"] = v
        for k, v in f.items():
            if k not in META_COLS and v is not None and not isinstance(v, (list, dict)):
                row[f"s0::{k}"] = v
        if c.get("ecusum_a_seq"):
            row["ecusum_a_seq"] = c["ecusum_a_seq"]
        frame.append(row)
    return frame


def score_feature(
    frame: List[Dict[str, Any]], col: str, sessions: np.ndarray,
    labels: np.ndarray, keep: np.ndarray,
) -> Optional[Dict[str, Any]]:
    vals = np.array([frame[i].get(col) if frame[i].get(col) is not None else np.nan
                     for i in range(len(frame))], dtype=float)
    v = vals[keep]
    y = labels[keep]
    s = sessions[keep]
    if np.all(np.isnan(v)) or len(np.unique(v[~np.isnan(v)])) < 2:
        return None
    med = float(np.nanmedian(v))
    v_filled = np.where(np.isnan(v), med, v)
    orientation = ORIENTATIONS.get(col.split("::")[-1], 1)
    risk = -v_filled if orientation < 0 else v_filled

    ap = auprc(risk, y)
    ar = auroc(risk, y)
    ap_lo, ap_hi = clustered_bootstrap(risk, y, s, auprc)
    ar_lo, ar_hi = clustered_bootstrap(risk, y, s, auroc)

    # S0 twin
    s0_col = col.replace("c::", "s0::", 1)
    delta = None
    if s0_col != col and frame and frame[0].get(s0_col) is not None:
        v0 = np.array([frame[i].get(s0_col) if frame[i].get(s0_col) is not None else np.nan
                       for i in range(len(frame))], dtype=float)[keep]
        if not np.all(np.isnan(v0)) and len(np.unique(v0[~np.isnan(v0)])) >= 2:
            v0f = np.where(np.isnan(v0), float(np.nanmedian(v0)), v0)
            risk0 = -v0f if orientation < 0 else v0f
            d_point, d_lo, d_hi = paired_delta_bootstrap(risk, risk0, y, s)
            delta = {"point": round(d_point, 4), "ci_lo": round(d_lo, 4) if d_lo is not None else None,
                     "ci_hi": round(d_hi, 4) if d_hi is not None else None}

    # length control
    ngen = v_filled  # same-arm n_generated as the control
    control = np.array([frame[i].get("n_generated") or 0 for i in range(len(frame))], dtype=float)[keep]
    resid = rank_residualize(v_filled, control)
    ap_resid = auprc(resid, y)

    # stratified
    cens = np.array([bool(frame[i].get("censored")) for i in range(len(frame))])[keep]
    ap_uncens = auprc(risk[~cens], y[~cens]) if (~cens).sum() > 10 else None
    ap_cens = auprc(risk[cens], y[cens]) if cens.sum() > 10 else None

    # matched-fire-rate operating point vs parse baseline
    pf = np.array([bool(frame[i].get("parse_fail_fire")) for i in range(len(frame))])[keep]
    pf_rate = float(pf.mean())
    thresh = float(np.quantile(risk, 1.0 - pf_rate))
    op = operating_point(risk, y, thresh)
    op["baseline_fire_rate"] = round(pf_rate, 4)

    base_cov = int((pf & (y == 1)).sum())
    base_fr = int((pf & (y == 0)).sum())

    return {
        "feature": col,
        "orientation": orientation,
        "auprc": None if ap is None else round(ap, 4),
        "auprc_ci": [None if ap_lo is None else round(ap_lo, 4), None if ap_hi is None else round(ap_hi, 4)],
        "auroc": None if ar is None else round(ar, 4),
        "auroc_ci": [None if ar_lo is None else round(ar_lo, 4), None if ar_hi is None else round(ar_hi, 4)],
        "base_rate": 0.1033,
        "delta_vs_s0": delta,
        "auprc_length_controlled": None if ap_resid is None else round(ap_resid, 4),
        "auprc_uncensored": None if ap_uncens is None else round(ap_uncens, 4),
        "auprc_censored": None if ap_cens is None else round(ap_cens, 4),
        "matched_rate_op": op,
        "parse_baseline": {"coverage": base_cov, "false_resets": base_fr,
                           "precision": round(base_cov / max(1, int(pf.sum())), 4)},
        "sweep": sweep(risk, y),
    }


def verdict(entry: Dict[str, Any]) -> str:
    """The 4.0 winner rule, applied mechanically."""
    if entry.get("auprc") is None:
        return "n/a"
    ap_lo = (entry["auprc_ci"] or [None, None])[0]
    d = entry.get("delta_vs_s0") or {}
    d_lo = d.get("ci_lo")
    unc = entry.get("auprc_uncensored")
    base = entry.get("parse_baseline") or {}
    op = entry.get("matched_rate_op") or {}
    beats_base = (op.get("coverage", 0) > (base.get("coverage") or 0)
                  and (op.get("precision") or 0) > (base.get("precision") or 0)
                  and (base.get("false_resets") or 0) >= op.get("false_resets", 0))
    above_base_rate = ap_lo is not None and ap_lo > 0.1033
    s0_ok = d_lo is not None and d_lo > 0
    length_ok = (entry.get("auprc_length_controlled") is not None
                 and entry["auprc_length_controlled"] > 0.1033)
    unc_ok = unc is not None and unc > 0.1033
    live = beats_base and above_base_rate and s0_ok and length_ok and unc_ok
    return "LIVE" if live else "not-live"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_c2kv", required=True)
    parser.add_argument("--features_full", required=True)
    parser.add_argument("--battery_full", required=True, help="frozen r2 battery rows")
    parser.add_argument("--battery_c2kv", required=True, help="frozen r2 battery rows")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rows_c2kv", default="", help="RERUN battery rows (for generate_sec cost axis)")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    label_frame = build_label_frame(
        join_arms(load_jsonl(args.battery_full), load_jsonl(args.battery_c2kv)), manifest)
    features_c = load_jsonl(args.features_c2kv)
    features_f = load_jsonl(args.features_full)
    rows_c2kv = load_jsonl(args.rows_c2kv) if args.rows_c2kv else []
    frame = build_frame(features_c, features_f, label_frame, rows_c2kv)

    stats = census(label_frame, manifest)
    cusum_cal = ecusum_finalize(frame)
    knono = knono_report(frame)

    keep = np.array([r["label_cw"] in (0, 1) for r in frame])
    labels = np.array([1 if r["label_cw"] == 1 else 0 for r in frame])[keep]
    sessions = session_clusters([r["session_id"] for r in frame])[keep]

    # baseline row first
    pf = np.array([bool(r.get("parse_fail_fire")) for r in frame])[keep]
    baseline_entry = {
        "feature": "baseline::parse_fail_only",
        "auprc": round(auprc(pf.astype(float), labels), 4),
        "auroc": round(auroc(pf.astype(float), labels), 4),
        "parse_baseline": {"coverage": int((pf & (labels == 1)).sum()),
                           "false_resets": int((pf & (labels == 0)).sum()),
                           "precision": round(float(pf[labels == 1].mean() if pf.any() else 0), 4)},
    }
    # LEN control row
    len_entry = score_feature(frame, "c::n_generated", session_clusters([r["session_id"] for r in frame]),
                              np.array([1 if r["label_cw"] == 1 else 0 for r in frame]), keep)

    cols = sorted({k for r in frame for k in r if k.startswith("c::")})
    entries = []
    for col in cols:
        e = score_feature(frame, col, session_clusters([r["session_id"] for r in frame]),
                          np.array([1 if r["label_cw"] == 1 else 0 for r in frame]), keep)
        if e is not None:
            e["verdict"] = verdict(e)
            entries.append(e)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "census": stats,
        "ecusum_calibration": cusum_cal,
        "kono": knono,
        "baseline": baseline_entry,
        "len_control": ({k: v for k, v in len_entry.items() if k != "sweep"} if len_entry else None),
        "entries": entries,
        "n_frame": len(frame),
        "n_trigger_subset": int(keep.sum()),
        "mde_pp": MDE_PP,
        "bootstrap_reps": BOOTSTRAP_REPS,
    }
    (out_dir / "winner_table.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# t33 winner table (features)", "",
             f"frame={len(frame)} trigger-subset={int(keep.sum())} "
             f"(C->W={int(labels.sum())}, C->C={int((labels==0).sum())})",
             f"baseline parse-fail: {baseline_entry['parse_baseline']}", "",
             "| feature | AUPRC [CI] | AUROC | ΔvsS0 [CI] | AP(len-ctl) | AP(uncens) | verdict |",
             "|---|---|---|---|---|---|---|"]
    def fmt(x):
        return "-" if x is None else str(x)
    for e in sorted(entries, key=lambda x: -(x.get("auprc") or -1)):
        d = e.get("delta_vs_s0") or {}
        lines.append(
            f"| {e['feature']} | {fmt(e['auprc'])} {e['auprc_ci']} | {fmt(e['auroc'])} | "
            f"{fmt(d.get('point'))} {fmt(d.get('ci_lo'))},{fmt(d.get('ci_hi'))} | "
            f"{fmt(e.get('auprc_length_controlled'))} | {fmt(e.get('auprc_uncensored'))} | {e.get('verdict')} |")
    (out_dir / "winner_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[t33] scored {len(entries)} features -> {out_dir}/winner_table.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
