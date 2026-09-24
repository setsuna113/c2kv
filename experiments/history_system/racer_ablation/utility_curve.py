"""Recovery utility on frozen, task-balanced, prefix-valid decision states.

For each sampled state i: Y0 = current-turn success of M2 (held draft committed,
M2 continues), YR = current-turn success of the probe P (same prefix, risk gate
bypassed once at the state, then M2 policy). Delta_i = YR - Y0. N = states with
both labels defined and a reproduced prefix; F = states of N where P completed
one extra generation. U_pi(k) = (1/N) sum of Delta over the top-k states of F
ranked by pi. Random ranking has the closed-form expectation k/(|F| N) sum_F Delta.
Intervals resample tasks (one state per task) with a fixed seed.

usage: python -m racer_ablation.utility_curve BFCL_DIR STATES_JSON LIGHT_JSON OUT_JSON \
           --m2 ROOT [ROOT ...] --probe ROOT [ROOT ...]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from benchmarks.memory_runtime.recovery.set_models import C1RiskArtifact

from .common import CELLS, draft_tokens, raw_result, steps, task_rows, turn_of, write_json
from .extract_decisions import decision_row
from .light_detector import light_score, light_vector
from .sample_states import terminated_turn
from .turn_scorer import TurnScorer

BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260924
GRID = np.linspace(0.0, 1.0, 101)


def draft_context(record):
    """C1 detector inputs of the held draft, rebuilt from the recorded trace."""
    trace = record["generation_trace"][0]
    shadow = trace["generation"]["stats"].get("shadow_features") or {}
    prefill = shadow.get("prefill") or {}
    draft = trace["native_draft"]
    parse_ok = draft["status"] != "malformed"
    return {
        "prefill_hidden": prefill.get("hidden") if prefill.get("status") == "captured" else None,
        "prefill_contract": {"layer": prefill.get("layer"), "readout": prefill.get("readout"),
                             "position_kind": (prefill.get("position") or {}).get("kind"),
                             "bindings": shadow.get("bindings") or {}},
        "draft_logprobs": list(trace["generation"]["token_logprobs"]),
        "parse_ok": parse_ok, "is_stop": parse_ok and not draft.get("tool_calls"),
    }


def prefix_check(m2_records, probe_records, target):
    """Draft tokens must match M2 at every decision up to and including the target."""
    m2_keys = [record["decision_key"] for record in m2_records]
    if target not in m2_keys:
        return "target_missing_in_m2", None
    probe = {record["decision_key"]: record for record in probe_records}
    for index, record in enumerate(m2_records[: m2_keys.index(target) + 1]):
        other = probe.get(record["decision_key"])
        if other is None:
            return "probe_missing_decision", index
        if draft_tokens(other) != draft_tokens(record):
            return "prefix_not_reproduced", index
    return "reproduced", None


def curve(delta, scores):
    """U(k)/N for k = 0..|F| in descending score order (ties by task id)."""
    order = sorted(range(len(delta)), key=lambda i: (-scores[i], i))
    return np.concatenate([[0.0], np.cumsum([delta[i] for i in order])])


def at_grid(values, n):
    """Interpolate U over x = k/N on a fixed grid; NaN beyond the pool size."""
    x = np.arange(len(values)) / n
    out = np.interp(GRID, x, values / n)
    out[GRID > x[-1] + 1e-12] = np.nan
    return out


def auc_ap(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    positives, negatives = labels.sum(), len(labels) - labels.sum()
    if positives == 0 or negatives == 0:
        return None, None
    ranks = np.argsort(np.argsort(scores)) + 1
    # Average ranks for ties.
    for value in np.unique(scores):
        tie = scores == value
        ranks = ranks.astype(float)
        ranks[tie] = ranks[tie].mean()
    auroc = (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(-scores, kind="stable")
    hits = np.cumsum(labels[order])
    precision = hits / np.arange(1, len(labels) + 1)
    ap = float((precision * labels[order]).sum() / positives)
    return float(auroc), ap


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("bfcl_dir")
    parser.add_argument("states")
    parser.add_argument("light")
    parser.add_argument("out")
    parser.add_argument("--m2", nargs="+", required=True)
    parser.add_argument("--probe", nargs="+", required=True)
    args = parser.parse_args(argv)
    scorer = TurnScorer(args.bfcl_dir)
    states = json.load(open(args.states, encoding="utf-8"))["states"]
    light = json.load(open(args.light, encoding="utf-8"))
    m2_rows, _ = task_rows(args.m2, CELLS["M2"])
    probe_rows, probe_failures = task_rows(args.probe, CELLS["P"])
    artifact = None
    rows, coverage = [], {"target_tasks": len(states), "sampled": 0, "probe_run": 0,
                          "reproduced": 0, "pair_scorable": 0, "recoverable": 0,
                          "reasons": {}}

    def excluded(task, reason, **extra):
        coverage["reasons"][reason] = coverage["reasons"].get(reason, 0) + 1
        rows.append({"task_id": task, "status": reason, **extra})

    for task, state in sorted(states.items()):
        if state.get("status") != "sampled":
            excluded(task, state.get("status", "not_sampled"))
            continue
        coverage["sampled"] += 1
        target, turn = state["decision_key"], turn_of(state["decision_key"])
        if task not in probe_rows:
            excluded(task, "probe_missing", decision_key=target)
            continue
        coverage["probe_run"] += 1
        m2_records, probe_records = steps(m2_rows[task]), steps(probe_rows[task])
        status, index = prefix_check(m2_records, probe_records, target)
        if status != "reproduced":
            excluded(task, status, decision_key=target, mismatch_index=index)
            continue
        coverage["reproduced"] += 1
        y0 = scorer.turn_outcomes(task, raw_result(m2_rows[task]) or [],
                                  terminated_turn=terminated_turn(m2_records))
        yr = scorer.turn_outcomes(task, raw_result(probe_rows[task]) or [],
                                  terminated_turn=terminated_turn(probe_records))
        label0 = y0[turn] if turn < len(y0) else None
        label_r = yr[turn] if turn < len(yr) else None
        target_record = next(r for r in probe_records if r["decision_key"] == target)
        decision = decision_row(target_record)
        context = draft_context(target_record)
        if artifact is None:
            with open(f"{probe_rows[task]['_cell_dir']}/native/controller.json", encoding="utf-8") as handle:
                artifact = C1RiskArtifact(json.load(handle)["candidate_algorithm"]["risk_artifact"])
        offline = artifact.predict_risk(context)
        entry = {"task_id": task, "decision_key": target, "turn": turn,
                 "y0": label0, "yr": label_r, "full_score": decision["risk_score"],
                 "full_score_offline": offline.score if offline.available else None,
                 "light_score": light_score(light, light_vector(context)),
                 "risk_triggered_online": decision["risk_triggered"],
                 "repack_feasible": decision["repack_feasible"],
                 "extra_generation": decision["regeneration_completed"],
                 "regenerated_action_committed": decision["regenerated_action_committed"],
                 "parse_fallback": decision["parse_fallback"]}
        if label0 is None or label_r is None:
            excluded(task, "label_undefined", **entry)
            continue
        coverage["pair_scorable"] += 1
        coverage["recoverable"] += int(entry["extra_generation"])
        rows.append(dict(entry, status="scored", delta=int(label_r) - int(label0)))

    scored = [row for row in rows if row["status"] == "scored"]
    pool = [row for row in scored if row["extra_generation"]]
    n = len(scored)
    result = {"schema": "racer-ablation-utility-v1", "result_status": "preliminary, n=1",
              "coverage": coverage, "states": rows,
              "probe_failures": {key: sorted(value) for key, value in probe_failures.items()},
              "consistency": {
                  "nonpool_delta_nonzero": sum(1 for row in scored
                                               if not row["extra_generation"] and row["delta"] != 0),
                  "max_abs_offline_online_score": max(
                      (abs(row["full_score"] - row["full_score_offline"]) for row in rows
                       if row.get("full_score") is not None
                       and row.get("full_score_offline") is not None), default=None)}}
    if n:
        delta = [row["delta"] for row in pool]
        orders = {"full": [row["full_score"] for row in pool],
                  "light": [row["light_score"] for row in pool]}
        curves = {name: (curve(delta, scores) / n).tolist() for name, scores in orders.items()}
        curves["random_expected"] = [k * sum(delta) / (len(pool) * n) if pool else 0.0
                                     for k in range(len(pool) + 1)]
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        bands = {name: [] for name in ("full", "light", "random_expected")}
        for _ in range(BOOTSTRAP):
            sample = [scored[i] for i in rng.integers(0, n, n)]
            sub = [row for row in sample if row["extra_generation"]]
            d = [row["delta"] for row in sub]
            for name in ("full", "light"):
                key = "full_score" if name == "full" else "light_score"
                bands[name].append(at_grid(curve(d, [row[key] for row in sub]), n))
            bands["random_expected"].append(at_grid(
                np.array([k * sum(d) / len(sub) if sub else 0.0 for k in range(len(sub) + 1)]), n))
        result["curves"] = {"x": [k / n for k in range(len(pool) + 1)], **curves}
        result["bands"] = {"grid": GRID.tolist(), **{
            name: {"lo": np.nanpercentile(np.stack(values), 2.5, axis=0).tolist(),
                   "hi": np.nanpercentile(np.stack(values), 97.5, axis=0).tolist()}
            for name, values in bands.items()}}
        result["transitions"] = {
            scope: {f"{a}->{b}": sum(1 for row in rows_ if (row["y0"], row["yr"]) == (a, b))
                    for a in (False, True) for b in (False, True)}
            for scope, rows_ in (("pool", pool), ("all_scored", scored))}
        risk_labels = [int(not row["y0"]) for row in scored]
        result["supplement_auroc_ap_on_y0_failure"] = {
            name: dict(zip(("auroc", "ap"), auc_ap([row[key] for row in scored], risk_labels)))
            for name, key in (("full", "full_score"), ("light", "light_score"))}
        result["n"], result["pool"] = n, len(pool)
        result["always_endpoint"] = sum(delta) / n
    write_json(args.out, result)
    print(json.dumps({"coverage": coverage, "n": n, "pool": len(pool)}))


if __name__ == "__main__":
    main()
