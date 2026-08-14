#!/usr/bin/env python
"""Pre-registered trigger-signal evaluation on the clean S4 logp recompute.

Pure-CPU analysis (numpy only). Combines:
  - the clean teacher-forced logp recompute (agent/recompute_s4_logp.py), and
  - the round-1 S4 arm archives (agent/eval_agent_history_s4_npu.sh), used ONLY
    for behavioral fields (has_tool_call / prediction_tool_name /
    tool_name_match / prediction). The arm files' logp_prefix_* fields are void
    (round-1 scored them after generation, which polluted the prefix KV cache
    in place) and are never read here.

Label (positive = paired action change), per qid over valid A x B rows:
  (A called and B did not call) OR (both called but extracted tool names
  differ -- a name flip). "Called" and tool names come from the same row
  fields / parsing as agent/analyze_s4_forced_prefix.py (helpers copied below
  with attribution, since the eval modules import torch at module top).

Signals (score per qid, a-priori orientation so larger = predicts change):
  S1 = logp_prefix_c2kv, single-ended clean logp (orientation: negated --
       compression damage lowers the forced-prefix logp);
  S2 = (logp_prefix_full - logp_prefix_c2kv) / |logp_prefix_full|,
       length-normalized paired gap (positive when c2kv is worse than full);
  S3 = behavioral fingerprint from the archives only (no logp): univariate
       AUROC per feature plus best-of-S3;
  S0 = same-family full-KV control: logp_prefix_full single-ended (same
       orientation convention as S1) + arm-A/D behavioral features, best-of-S0.

Inference: AUROC with session-cluster bootstrap 95% CIs (2000 reps, seed 0;
self-contained numpy helpers -- stats_paired.py is intentionally NOT used).
Delta-AUROC = best(S1..S3) - best(S0) with a paired cluster-bootstrap CI.

Example:
  python agent/analyze_trigger_signals.py \
    --logp_jsonl outputs/s4_logp_recompute.jsonl \
    --arm_a outputs/s4_armA_full.jsonl --arm_b outputs/s4_armB_c2kv.jsonl \
    --arm_c outputs/s4_armC_c2kv_forced.jsonl --arm_d outputs/s4_armD_full_forced.jsonl \
    --out_prefix outputs/s4_trigger_signals
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # degrade gracefully: reported at runtime, not import time
    np = None  # type: ignore[assignment]

Z_975 = 1.959964  # copied from agent/analyze_s4_forced_prefix.py

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0

# Pre-registered kill criteria (printed verbatim in the report before verdicts).
KILL_BLOCK = (
    "KILL if any of: (i) best candidate AUROC < 0.62; "
    "(ii) its 95% CI lower bound <= 0.5; (iii) \u0394AUROC vs S0 < 0.07"
)
KILL_AUROC_MIN = 0.62
KILL_DELTA_MIN = 0.07


# ---------------------------------------------------------------------------
# Helpers copied with attribution. The eval modules import torch at module
# top, so this CPU-only script copies the small pure functions instead.
# ---------------------------------------------------------------------------


def _has_tool_call(text: str) -> bool:
    """Copied from agent/eval_agent_history_c2kv.py::_has_tool_call."""
    return "<tool_call>" in (text or "") or "Action:" in (text or "")


def _extract_tool_name(text: str) -> Optional[str]:
    """Copied from agent/eval_agent_tool_definition_c2kv.py::_extract_tool_name."""
    if not text:
        return None
    blocks = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, flags=re.S)
    candidates = blocks or [text]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except Exception:
            value = None
        if isinstance(value, dict):
            function = value.get("function") if isinstance(value.get("function"), dict) else {}
            name = (
                value.get("name")
                or value.get("tool_name")
                or value.get("function_name")
                or function.get("name")
            )
            if name:
                return str(name)
    match = re.search(r'"(?:name|tool_name|function_name)"\s*:\s*"([^"]+)"', text)
    if match:
        return match.group(1)
    match = re.search(r"<tool_call>.*?([A-Za-z0-9_.:-]+).*?</tool_call>", text, flags=re.S)
    if match:
        return match.group(1)
    return None


def _load_rows(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Copied from agent/analyze_s4_forced_prefix.py::_load_rows (skipped rows dropped)."""
    if not path:
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("skipped"):
                continue
            rows[row["qid"]] = row
    return rows


def _load_logp_rows(path: str) -> Dict[str, Dict[str, Any]]:
    """Clean recompute rows keyed by qid; only rows with BOTH logps non-null count
    as clean coverage (skipped/partial rows are kept for counts but not scored)."""
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("qid")
            if qid:
                rows[str(qid)] = row
    return rows


def _logp_clean(row: Optional[Dict[str, Any]]) -> bool:
    return (
        row is not None
        and row.get("logp_prefix_c2kv") is not None
        and row.get("logp_prefix_full") is not None
    )


def _wilson(k: int, n: int) -> Tuple[float, float, float]:
    """Copied from agent/analyze_s4_forced_prefix.py::_wilson."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + Z_975**2 / n
    center = (p + Z_975**2 / (2 * n)) / denom
    half = Z_975 * math.sqrt(p * (1 - p) / n + Z_975**2 / (4 * n**2)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def _session_tool_names(dataset_path: str) -> Dict[str, set]:
    """Copied from agent/analyze_s4_forced_prefix.py::_session_tool_names.

    Only used for the optional S3 in-session-schema feature; the import pulls
    the training data source (and transitively torch), so any failure degrades
    to "feature omitted with note".
    """
    try:
        from train_agent_tool_definition_c2kv import (  # noqa: E402
            AgentLLMTracesSource,
            AgentToolDefinitionDataArgs,
            _as_tool_list,
            _canonical_tool_definition,
            _span_attributes,
            _tool_name,
        )
    except Exception as error:
        print(f"[warn] cannot import data source ({error}); S3 in-schema feature unavailable")
        return {}
    try:
        source = AgentLLMTracesSource(AgentToolDefinitionDataArgs(dataset_path=dataset_path))
        sessions: Dict[str, set] = {}
        for session in source.sessions:
            names = set()
            for span in session["spans"]:
                tool_value = _span_attributes(span).get("gen_ai.tool.definitions")
                if not tool_value:
                    continue
                for tool in _as_tool_list(_canonical_tool_definition(tool_value)):
                    name = _tool_name(tool)
                    if name:
                        names.add(str(name))
                break
            sessions[session["session_id"]] = names
        return sessions
    except Exception as error:
        print(f"[warn] session tool names unavailable ({error}); S3 in-schema feature omitted")
        return {}


def _git_commit() -> str:
    """Best-effort git commit hash of this repo; 'unknown' on any failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=Path(__file__).resolve().parents[1],
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Row field accessors: prefer the eval-emitted fields (identical definitions to
# agent/analyze_s4_forced_prefix.py), fall back to the copied parsers.
# ---------------------------------------------------------------------------


def _row_called(row: Dict[str, Any]) -> bool:
    value = row.get("has_tool_call")
    if value is not None:
        return bool(value)
    return _has_tool_call(row.get("prediction", ""))


def _row_tool_name(row: Dict[str, Any]) -> Optional[str]:
    value = row.get("prediction_tool_name")
    if value:
        return str(value)
    return _extract_tool_name(row.get("prediction", ""))


def _paired_action_change_label(a_row: Dict[str, Any], b_row: Dict[str, Any]) -> int:
    """Positive = (A called, B did not) OR (both called, extracted names differ).

    A None extraction on one side counts as "differ" when the other side has a
    name: B produced an unparseable call where A produced tool X, which is an
    action change. (Flagged interpretation: the pre-registration says
    "extracted tool names differ" without specifying the None case.)
    """
    a_called = _row_called(a_row)
    b_called = _row_called(b_row)
    if a_called and not b_called:
        return 1
    if a_called and b_called:
        return 1 if _row_tool_name(a_row) != _row_tool_name(b_row) else 0
    return 0


# ---------------------------------------------------------------------------
# AUROC + session-cluster bootstrap (self-contained; stats_paired.py NOT used).
# ---------------------------------------------------------------------------


def _rankdata(values: "np.ndarray") -> "np.ndarray":
    """Average ranks (1-based) with tie handling, numpy-only (scipy rankdata)."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        ranks[order[index : end + 1]] = 0.5 * (index + end) + 1.0
        index = end + 1
    return ranks


def _auroc(scores: "np.ndarray", labels: "np.ndarray") -> float:
    """Mann-Whitney AUROC; nan when only one class is present."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _cluster_index_draws(
    clusters: "np.ndarray", reps: int, seed: int
) -> Tuple[List["np.ndarray"], "np.ndarray"]:
    """Session-cluster bootstrap: resample unique session ids with replacement
    (duplicates kept, standard cluster bootstrap). Returns (draws, unique)."""
    unique = np.unique(clusters)
    members = {session: np.where(clusters == session)[0] for session in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws.append(np.concatenate([members[session] for session in sampled]))
    return draws, unique


def _auroc_bootstrap(
    scores: "np.ndarray",
    labels: "np.ndarray",
    draws: Sequence["np.ndarray"],
) -> Tuple[float, Optional[List[float]], int]:
    """Point AUROC + percentile 95% CI over pre-drawn cluster-bootstrap indices."""
    point = _auroc(scores, labels)
    values = []
    for idx in draws:
        value = _auroc(scores[idx], labels[idx])
        if not math.isnan(value):
            values.append(value)
    if not values:
        return point, None, 0
    lo, hi = np.percentile(values, [2.5, 97.5])
    return point, [float(lo), float(hi)], len(values)


# A cleaned signal: (oriented scores with nan dropped, matching labels, per-signal
# cluster-bootstrap index draws remapped onto the dropped positions).
CleanSignal = Tuple["np.ndarray", "np.ndarray", List["np.ndarray"]]


def _clean_signal(
    entry: Dict[str, Any],
    labels: "np.ndarray",
    draws: Sequence["np.ndarray"],
) -> CleanSignal:
    values = np.asarray(entry["values"], dtype=float) * entry["orient"]
    mask = ~np.isnan(values)
    return values[mask], labels[mask], draws_for(draws, mask)


def _delta_best_bootstrap(
    candidate_signals: Dict[str, CleanSignal],
    control_signals: Dict[str, CleanSignal],
) -> Tuple[float, Optional[List[float]], int]:
    """Delta-AUROC = best(candidates) - best(controls), paired per bootstrap rep:
    both sides use the same underlying session resamples (per-signal draws only
    remap positions around that signal's own missing entries), and the per-rep
    argmax may pick different signals -- the correct bootstrap of the max
    statistic."""
    cand_points = [_auroc(scores, labels) for scores, labels, _ in candidate_signals.values()]
    ctrl_points = [_auroc(scores, labels) for scores, labels, _ in control_signals.values()]
    cand_points = [value for value in cand_points if not math.isnan(value)]
    ctrl_points = [value for value in ctrl_points if not math.isnan(value)]
    point = (max(cand_points) - max(ctrl_points)) if cand_points and ctrl_points else float("nan")

    def _best(signals: Dict[str, CleanSignal], rep: int) -> Optional[float]:
        values = []
        for scores, labels, sub_draws in signals.values():
            if rep >= len(sub_draws):
                continue
            idx = sub_draws[rep]
            value = _auroc(scores[idx], labels[idx])
            if not math.isnan(value):
                values.append(value)
        return max(values) if values else None

    all_signals = list(candidate_signals.values()) + list(control_signals.values())
    # Signals with no valid draws (all-nan) are already excluded by the caller;
    # ignore any stragglers here instead of zeroing the whole rep loop.
    reps = min((len(sub_draws) for _, _, sub_draws in all_signals if sub_draws), default=0)
    values = []
    for rep in range(reps):
        cand_best = _best(candidate_signals, rep)
        ctrl_best = _best(control_signals, rep)
        if cand_best is not None and ctrl_best is not None:
            values.append(cand_best - ctrl_best)
    if not values:
        return point, None, 0
    lo, hi = np.percentile(values, [2.5, 97.5])
    return point, [float(lo), float(hi)], len(values)


# ---------------------------------------------------------------------------
# Signal assembly.
# ---------------------------------------------------------------------------


def _finite_or_nan(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _build_signals(
    eval_qids: List[str],
    logp_rows: Dict[str, Dict[str, Any]],
    arm_rows: Dict[str, Dict[str, Dict[str, Any]]],
    session_tools: Optional[Dict[str, set]],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Raw per-qid signal values + a-priori orientation (sign applied so that a
    LARGER oriented score predicts the positive label). Signals that cannot be
    computed for a qid stay nan and are dropped per-signal (n reported)."""
    notes: List[str] = []
    raw: Dict[str, List[float]] = {}

    def series(name: str) -> List[float]:
        raw[name] = []
        return raw[name]

    s1 = series("S1.logp_prefix_c2kv")
    s2 = series("S2.norm_logp_gap")
    s3_b_call = series("S3.b_made_call")
    s3_c_match = series("S3.c_forced_tool_name_match")
    s3_b_schema = series("S3.b_prediction_in_session_schema")
    s0_full = series("S0.logp_prefix_full")
    s0_a_call = series("S0.a_made_call")
    s0_d_match = series("S0.d_forced_tool_name_match")

    c_rows = arm_rows.get("C", {})
    d_rows = arm_rows.get("D", {})
    n_s2_undefined = 0
    for qid in eval_qids:
        logp = logp_rows[qid]  # clean coverage guaranteed by the eval-set filter
        logp_c2kv = _finite_or_nan(logp.get("logp_prefix_c2kv"))
        logp_full = _finite_or_nan(logp.get("logp_prefix_full"))
        s1.append(logp_c2kv)
        s0_full.append(logp_full)
        if logp_full == 0.0 or math.isnan(logp_full) or math.isnan(logp_c2kv):
            s2.append(float("nan"))  # |logp_full| == 0 would divide by zero
            n_s2_undefined += 1
        else:
            s2.append((logp_full - logp_c2kv) / abs(logp_full))

        b_row = arm_rows["B"][qid]
        a_row = arm_rows["A"][qid]
        s3_b_call.append(float(_row_called(b_row)))
        c_row = c_rows.get(qid)
        s3_c_match.append(float(bool(c_row.get("tool_name_match"))) if c_row is not None else float("nan"))
        if session_tools is None:
            s3_b_schema.append(float("nan"))
        else:
            tools = session_tools.get(b_row.get("session_id") or "")
            pred_name = _row_tool_name(b_row)
            if not tools or pred_name is None:
                # No session schema or B made no parseable call: feature undefined.
                s3_b_schema.append(float("nan"))
            else:
                s3_b_schema.append(float(pred_name in tools))
        s0_a_call.append(float(_row_called(a_row)))
        d_row = d_rows.get(qid)
        s0_d_match.append(float(bool(d_row.get("tool_name_match"))) if d_row is not None else float("nan"))

    if n_s2_undefined:
        notes.append(f"S2 undefined (nan) for {n_s2_undefined}/{len(eval_qids)} qids (|logp_full|==0 or missing logp)")
    if not c_rows:
        notes.append("arm C rows unavailable; S3.c_forced_tool_name_match omitted")
    if not d_rows:
        notes.append("arm D rows unavailable; S0.d_forced_tool_name_match omitted")
    if session_tools is None:
        notes.append("session tool schema unavailable (no usable --dataset_path); S3.b_prediction_in_session_schema omitted")

    signals: Dict[str, Dict[str, Any]] = {
        # Candidate family (c2kv-specific).
        "S1.logp_prefix_c2kv": {
            "family": "candidate",
            "values": raw["S1.logp_prefix_c2kv"],
            # Compression damage LOWERS the forced-prefix logp -> negate.
            "orient": -1.0,
            "orientation_note": "negated: damage lowers logp_prefix_c2kv",
        },
        "S2.norm_logp_gap": {
            "family": "candidate",
            "values": raw["S2.norm_logp_gap"],
            "orient": 1.0,
            "orientation_note": "positive gap means c2kv scores the forced prefix lower than full KV",
        },
        "S3.b_made_call": {
            "family": "candidate_s3",
            "values": raw["S3.b_made_call"],
            # Under the documented B-arm call collapse, a MISSING call indicates damage.
            "orient": -1.0,
            "orientation_note": "negated: missed B call indicates damage",
        },
        "S3.c_forced_tool_name_match": {
            "family": "candidate_s3",
            "values": raw["S3.c_forced_tool_name_match"],
            "orient": 1.0,
            "orientation_note": "forcing recovers the correct tool when free c2kv generation was damaged",
        },
        "S3.b_prediction_in_session_schema": {
            "family": "candidate_s3",
            "values": raw["S3.b_prediction_in_session_schema"],
            # In-schema is healthy; out-of-schema hallucination indicates damage.
            "orient": -1.0,
            "orientation_note": "negated: out-of-schema B call indicates damage",
        },
        # Same-family full-KV control family.
        "S0.logp_prefix_full": {
            "family": "control_s0",
            "values": raw["S0.logp_prefix_full"],
            "orient": -1.0,
            "orientation_note": "same convention as S1: lower full-KV logp = generically harder sample",
        },
        "S0.a_made_call": {
            "family": "control_s0",
            "values": raw["S0.a_made_call"],
            "orient": -1.0,
            "orientation_note": "negated: same convention as S3.b_made_call",
        },
        "S0.d_forced_tool_name_match": {
            "family": "control_s0",
            "values": raw["S0.d_forced_tool_name_match"],
            "orient": 1.0,
            "orientation_note": "same convention as S3.c_forced_tool_name_match",
        },
    }
    return signals, notes


def _scored_signals(
    signals: Dict[str, Dict[str, Any]],
    labels: "np.ndarray",
    draws: Sequence["np.ndarray"],
) -> Dict[str, Dict[str, Any]]:
    """Per-signal oriented AUROC + cluster-bootstrap CI; undefined (nan) values dropped."""
    scored: Dict[str, Dict[str, Any]] = {}
    for name, spec in signals.items():
        values = np.asarray(spec["values"], dtype=float) * spec["orient"]
        mask = ~np.isnan(values)
        n = int(mask.sum())
        if n == 0:
            scored[name] = {**spec, "n": 0, "n_pos": 0, "auroc": None, "ci95": None, "n_valid_reps": 0}
            continue
        sub_labels = labels[mask]
        sub_values = values[mask]
        point, ci, n_reps = _auroc_bootstrap(sub_values, sub_labels, draws_for(draws, mask))
        scored[name] = {
            **spec,
            "n": n,
            "n_pos": int((sub_labels == 1).sum()),
            "auroc": None if math.isnan(point) else round(point, 4),
            "auroc_raw": point,
            "ci95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
            "n_valid_reps": n_reps,
        }
    return scored


def draws_for(draws: Sequence["np.ndarray"], mask: "np.ndarray") -> List["np.ndarray"]:
    """Remap cluster-bootstrap index draws onto a masked sub-array.

    The draws index the full eval set; for a signal with nan-dropped entries we
    keep draw entries whose position is valid and remap them to sub-array
    positions. Cluster structure is preserved (whole sessions stay in/out).
    """
    remap = np.full(len(mask), -1, dtype=int)
    remap[np.where(mask)[0]] = np.arange(int(mask.sum()))
    sub_draws = []
    for idx in draws:
        sub = remap[idx[mask[idx]]]
        if len(sub):
            sub_draws.append(sub)
    return sub_draws


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


def _fmt_auroc(entry: Dict[str, Any]) -> str:
    if entry.get("auroc") is None:
        return "n/a"
    ci = entry.get("ci95")
    ci_text = f" [{ci[0]:.4f},{ci[1]:.4f}]" if ci else " [n/a]"
    return f"{entry['auroc']:.4f}{ci_text} ({entry['n_pos']}/{entry['n']})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logp_jsonl", required=True, help="Clean recompute rows (agent/recompute_s4_logp.py).")
    parser.add_argument("--arm_a", help="Round-1 arm A jsonl (full, free).")
    parser.add_argument("--arm_b", help="Round-1 arm B jsonl (c2kv@4, free).")
    parser.add_argument("--arm_c", help="Round-1 arm C jsonl (c2kv@4, forced).")
    parser.add_argument("--arm_d", help="Round-1 arm D jsonl (full, forced).")
    parser.add_argument("--out_prefix", required=True, help="Writes <out_prefix>.json and <out_prefix>.md.")
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Optional: enables the S3 in-session-schema feature (omitted with note otherwise).",
    )
    parser.add_argument("--bootstrap_reps", type=int, default=BOOTSTRAP_REPS)
    parser.add_argument("--bootstrap_seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    if np is None:
        sys.exit("analyze_trigger_signals.py requires numpy for the cluster bootstrap (pip install numpy).")

    commit = _git_commit()
    logp_rows = _load_logp_rows(args.logp_jsonl)
    arm_rows = {
        "A": _load_rows(args.arm_a),
        "B": _load_rows(args.arm_b),
        "C": _load_rows(args.arm_c),
        "D": _load_rows(args.arm_d),
    }

    # ---- Step 0 (printed FIRST, before any AUROC): coverage funnel. ----
    a_intersect_b = sorted(set(arm_rows["A"]) & set(arm_rows["B"]))
    eval_qids = [qid for qid in a_intersect_b if _logp_clean(logp_rows.get(qid))]
    labels = [_paired_action_change_label(arm_rows["A"][qid], arm_rows["B"][qid]) for qid in eval_qids]
    n_eval = len(eval_qids)
    n_pos = sum(labels)
    p_prev, lo_prev, hi_prev = _wilson(n_pos, n_eval)
    step0 = {
        "n_A_intersect_B": len(a_intersect_b),
        "n_with_clean_logp": n_eval,
        "clean_coverage": f"{n_eval}/{len(a_intersect_b)}",
        "n_positives": n_pos,
        "prevalence": f"{n_pos}/{n_eval}",
        "prevalence_rate": round(p_prev, 4),
        "prevalence_ci95": [round(lo_prev, 4), round(hi_prev, 4)],
        "git_commit": commit,
    }
    print("=== Step 0: coverage and label prevalence ===")
    print(f"|A∩B| valid qids: {len(a_intersect_b)}")
    print(f"with clean logp coverage (both logps non-null): {n_eval}/{len(a_intersect_b)}")
    print(
        f"positive class (paired action change): {n_pos}/{n_eval}"
        f" = {p_prev:.4f} [{lo_prev:.4f},{hi_prev:.4f}] (Wilson)"
    )
    print(f"git commit: {commit}")
    print()

    report: Dict[str, Any] = {
        "git_commit": commit,
        "inputs": {
            "logp_jsonl": args.logp_jsonl,
            "arm_a": args.arm_a,
            "arm_b": args.arm_b,
            "arm_c": args.arm_c,
            "arm_d": args.arm_d,
        },
        "bootstrap": {"reps": args.bootstrap_reps, "seed": args.bootstrap_seed, "cluster": "session_id"},
        "step0": step0,
        "notes": [],
    }
    if n_eval == 0 or n_pos in (0, n_eval):
        report["notes"].append("degenerate label set (empty or single-class); AUROC undefined everywhere")
        md = [
            "# S4 trigger-signal evaluation",
            "",
            f"git commit: {commit}",
            "",
            "## Step 0: coverage and label prevalence",
            "",
            "| quantity | value |",
            "|---|---|",
            f"| |A∩B| valid qids | {len(a_intersect_b)} |",
            f"| clean logp coverage | {n_eval}/{len(a_intersect_b)} |",
            f"| positive class | {n_pos}/{n_eval} = {p_prev:.4f} [{lo_prev:.4f},{hi_prev:.4f}] (Wilson) |",
            "",
            "Degenerate label set (empty or single-class); AUROC undefined everywhere.",
            "",
        ]
        _write_outputs(args.out_prefix, report, md)
        print("Degenerate label set; wrote skeleton report.")
        return

    labels_arr = np.asarray(labels, dtype=int)
    clusters = np.asarray([
        (logp_rows[qid].get("session_id") or arm_rows["B"][qid].get("session_id") or qid.rsplit(":", 1)[0])
        for qid in eval_qids
    ])
    draws, unique_sessions = _cluster_index_draws(clusters, args.bootstrap_reps, args.bootstrap_seed)
    report["bootstrap"]["n_sessions"] = int(len(unique_sessions))

    session_tools = _session_tool_names(args.dataset_path) if args.dataset_path else None
    signals, notes = _build_signals(eval_qids, logp_rows, arm_rows, session_tools)
    report["notes"].extend(notes)
    scored = _scored_signals(signals, labels_arr, draws)

    def _valid(family: str) -> Dict[str, Dict[str, Any]]:
        return {
            name: entry
            for name, entry in scored.items()
            if entry["family"] == family and entry.get("auroc") is not None
        }

    s3 = _valid("candidate_s3")
    s0 = _valid("control_s0")
    best_s3_name = max(s3, key=lambda name: s3[name]["auroc_raw"]) if s3 else None
    best_s0_name = max(s0, key=lambda name: s0[name]["auroc_raw"]) if s0 else None

    candidates: Dict[str, Dict[str, Any]] = {
        name: entry for name, entry in scored.items() if entry["family"] == "candidate" and entry.get("auroc") is not None
    }
    if best_s3_name is not None:
        candidates[f"S3_best({best_s3_name})"] = s3[best_s3_name]
    best_cand_name = max(candidates, key=lambda name: candidates[name]["auroc_raw"]) if candidates else None

    # Delta-AUROC: best(S1..S3) - best(S0), paired cluster bootstrap.
    # All-nan signals (auroc None) are excluded: they carry no information and
    # their empty draw lists would zero the paired rep loop.
    cand_clean = {
        name: _clean_signal(entry, labels_arr, draws)
        for name, entry in scored.items()
        if entry["family"] in {"candidate", "candidate_s3"} and entry.get("auroc") is not None
    }
    s0_clean = {
        name: _clean_signal(entry, labels_arr, draws)
        for name, entry in s0.items()
        if entry.get("auroc") is not None
    }
    delta_point, delta_ci, delta_reps = _delta_best_bootstrap(cand_clean, s0_clean)

    def _drop_internal(entry: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in entry.items() if key not in {"values", "auroc_raw"}}

    report["signals"] = {name: _drop_internal(entry) for name, entry in scored.items()}
    report["s3_best"] = {"signal": best_s3_name, "auroc": s3[best_s3_name]["auroc"] if best_s3_name else None}
    report["s0_best"] = {"signal": best_s0_name, "auroc": s0[best_s0_name]["auroc"] if best_s0_name else None}
    report["best_candidate"] = {
        "signal": best_cand_name,
        "auroc": candidates[best_cand_name]["auroc"] if best_cand_name else None,
        "ci95": candidates[best_cand_name]["ci95"] if best_cand_name else None,
    }
    report["delta_auroc"] = {
        "point": None if math.isnan(delta_point) else round(delta_point, 4),
        "ci95": [round(delta_ci[0], 4), round(delta_ci[1], 4)] if delta_ci else None,
        "n_valid_reps": delta_reps,
        "definition": "best(S1..S3) - best(S0), paired session-cluster bootstrap",
    }

    # ---- Kill criteria: verbatim block first, then measured values + verdicts. ----
    best_auroc = candidates[best_cand_name]["auroc_raw"] if best_cand_name else float("nan")
    best_ci = candidates[best_cand_name]["ci95"] if best_cand_name else None
    ci_low = best_ci[0] if best_ci else float("nan")
    clauses = [
        {
            "clause": "(i) best candidate AUROC < 0.62",
            "measured": None if math.isnan(best_auroc) else round(best_auroc, 4),
            "threshold": KILL_AUROC_MIN,
            "kills": bool(math.isnan(best_auroc) or best_auroc < KILL_AUROC_MIN),
        },
        {
            "clause": "(ii) its 95% CI lower bound <= 0.5",
            "measured": None if math.isnan(ci_low) else round(ci_low, 4),
            "threshold": 0.5,
            "kills": bool(math.isnan(ci_low) or ci_low <= 0.5),
        },
        {
            "clause": "(iii) ΔAUROC vs S0 < 0.07",
            "measured": None if math.isnan(delta_point) else round(delta_point, 4),
            "threshold": KILL_DELTA_MIN,
            "kills": bool(math.isnan(delta_point) or delta_point < KILL_DELTA_MIN),
        },
    ]
    overall = "KILLED" if any(clause["kills"] for clause in clauses) else "PASS"
    report["kill_criteria"] = {"block": KILL_BLOCK, "clauses": clauses, "overall": overall}

    print("=== Per-signal AUROC (oriented: larger = predicts action change) ===")
    rows_md = ["| signal | family | AUROC [95% CI] (pos/n) |", "|---|---|---|"]
    for name, entry in scored.items():
        rows_md.append(f"| {name} | {entry['family']} | {_fmt_auroc(entry)} |")
        print(f"{name:36s} {_fmt_auroc(entry)}")
    print()
    print(f"best-of-S3: {best_s3_name}  best-of-S0: {best_s0_name}  best candidate: {best_cand_name}")
    print(
        f"ΔAUROC = {report['delta_auroc']['point']} CI {report['delta_auroc']['ci95']}"
        f" ({delta_reps}/{args.bootstrap_reps} valid reps)"
    )
    print()
    print("=== Kill criteria ===")
    print(KILL_BLOCK)
    kill_md = ["| clause | measured | verdict |", "|---|---|---|"]
    for clause in clauses:
        verdict = "KILLED" if clause["kills"] else "PASS"
        kill_md.append(f"| {clause['clause']} | {clause['measured']} | {verdict} |")
        print(f"  {clause['clause']}: measured={clause['measured']} -> {verdict}")
    print(f"overall: {overall}")

    md = [
        "# S4 trigger-signal evaluation",
        "",
        f"git commit: {commit}",
        "",
        "## Step 0: coverage and label prevalence",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| |A∩B| valid qids | {len(a_intersect_b)} |",
        f"| clean logp coverage | {n_eval}/{len(a_intersect_b)} |",
        f"| positive class | {n_pos}/{n_eval} = {p_prev:.4f} [{lo_prev:.4f},{hi_prev:.4f}] (Wilson) |",
        f"| sessions (bootstrap clusters) | {len(unique_sessions)} |",
        "",
        "## Per-signal AUROC",
        "",
        *rows_md,
        "",
        f"best-of-S3: `{best_s3_name}`; best-of-S0: `{best_s0_name}`; best candidate: `{best_cand_name}`",
        "",
        f"ΔAUROC = {report['delta_auroc']['point']} CI {report['delta_auroc']['ci95']} ({delta_reps}/{args.bootstrap_reps} valid reps)",
        "",
        "## Kill criteria",
        "",
        f"> {KILL_BLOCK}",
        "",
        *kill_md,
        "",
        f"**overall: {overall}**",
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in report["notes"]],
        "",
    ]
    _write_outputs(args.out_prefix, report, md)
    print(f"\nwrote {args.out_prefix}.json and {args.out_prefix}.md")


def _write_outputs(out_prefix: str, report: Dict[str, Any], md_lines: List[str]) -> None:
    json_path = Path(f"{out_prefix}.json")
    md_path = Path(f"{out_prefix}.md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
