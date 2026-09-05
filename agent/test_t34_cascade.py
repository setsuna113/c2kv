# -*- coding: utf-8 -*-
"""Tests for agent/t34_cascade.py (U6: 2502.15845 cascade + GCN ceiling).

Run from the worktree root:
    PYTHONIOENCODING=utf-8 python -m pytest agent/test_t34_cascade.py -q

No torch, no GPU, ASCII-only output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
ROOT = _HERE.parent

import t34_cascade as C  # noqa: E402
from t34_common import FrozenAssets, FrozenFrame, session_of  # noqa: E402
from t33_labels import parse_fail_baseline  # noqa: E402


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------

def _call(name="tool_a", args=None):
    payload = json.dumps({"name": name, "arguments": args or {"x": 1}})
    return "Action:\n<tool_call>\n" + payload + "\n</tool_call>"


def _row(qid, *, pred, ratio=8.0, chunks=4, gist=64, prompt=1000, sess=None):
    return {
        "qid": qid, "session_id": sess or session_of(qid), "prediction": pred,
        "actual_compression_ratio": ratio, "doc_chunks": chunks, "gist_tokens": gist,
        "hybrid_top_k": None, "prompt_tokens": prompt,
        "input_tokens": 4000 + prompt, "kept_history_tokens": 100 * chunks,
        "system_prefill_sec": 0.1, "full_prefill_sec": 0.2, "tool_compress_sec": 0.3,
        "blend_sec": 0.05, "generate_sec": 1.0, "generated_tokens": 20,
        # target-side fields the feature path must NOT read:
        "target": "gold text", "target_has_tool_call": True,
        "target_tool_name": "tool_gold", "tool_name_match": False,
        "prediction_tool_name": "tool_a", "exact_match": False,
    }


def _synthetic_frame(n_sessions=12, steps=4, seed=0):
    """A FrozenFrame built by hand (labels alternate, sessions are clusters)."""
    rng = np.random.default_rng(seed)
    pairs, labels = [], []
    for s in range(n_sessions):
        for t in range(steps):
            qid = f"sess{s:02d}:{t}"
            lab = int((s + t) % 3 != 0)          # ~2/3 positives
            bad = bool(rng.random() < (0.6 if lab else 0.2))
            pred = "no call here" if bad else _call("tool_a", {"x": int(rng.integers(0, 5))})
            c = _row(qid, pred=pred, ratio=8.0 + rng.normal(0, 1.0),
                     chunks=int(rng.integers(2, 9)), gist=int(rng.integers(30, 120)),
                     prompt=int(rng.integers(500, 4000)))
            f = _row(qid, pred=_call("tool_gold"))
            f["tool_name_match"] = True
            c["tool_name_match"] = bool(lab == 0)
            pairs.append((f, c))
            labels.append({"qid": qid, "session_id": c["session_id"], "label_cw": lab,
                           "censored_at_cap": bool(t % 2), "censored_at_cap_full": False,
                           "parse_fail_fire": bad, "z_deferral": lab})
    manifest = {"kv_recipe": {"max_new_tokens": 128},
                "cw_qids": [r["qid"] for r in labels if r["label_cw"] == 1]}
    return FrozenFrame(pairs=pairs, labels=labels, manifest=manifest, witness=None)


def _expensive_for(frame, seed=1):
    """Stage-2 table covering BOTH classes (what the NPU step must produce)."""
    rng = np.random.default_rng(seed)
    out = {}
    for r in frame.trigger_subset():
        q = r["qid"]
        changed = bool(rng.random() < (0.7 if r["label_cw"] == 1 else 0.3))
        pred = _call("tool_repaired") if changed else frame.c2kv_by_qid[q]["prediction"]
        out[q] = {"prediction": pred, "d_corr_slice_prefill_sec": 0.5,
                  "d_recompute_prefill_sec": 0.0, "generate_sec": 1.2}
    return out


# ---------------------------------------------------------------------------
# 1. Algorithm 1 semantics
# ---------------------------------------------------------------------------

def test_band_gives_exactly_p_on_distinct_scores():
    """2502.15845 Sec. 5: given t1, t* is set so that exactly a fraction p of
    the calibration scores falls in [t1, t*]."""
    s = np.linspace(0.0, 1.0, 100)          # distinct, t1 at the minimum
    for p in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0):
        t_star, realized, k = C.band_upper_threshold(s, t1=float(s.min()), p=p)
        assert k == round(p * len(s))
        assert realized == pytest.approx(p, abs=1e-12)
        in_band = ((s >= s.min()) & (s <= t_star)).sum()
        assert in_band == k or (k == 0 and in_band == 0)


def test_band_p_zero_degenerates_to_single_threshold():
    s = np.linspace(0, 1, 21)
    t_star, realized, k = C.band_upper_threshold(s, t1=0.5, p=0.0)
    assert k == 0 and realized == 0.0 and t_star < 0.5
    thr = C.CascadeThresholds(t1=0.5, t_star=t_star, t2=0.5, p_target=0.0, p_realized=0.0)
    out = C.apply_cascade(s, thr, [None] * len(s))
    assert not out["escalated"].any()
    assert np.array_equal(out["fire"], s >= 0.5)


def test_algorithm1_three_branches():
    thr = C.CascadeThresholds(t1=0.3, t_star=0.7, t2=0.5, p_target=0.4, p_realized=0.4)
    assert C.cascade_decide(0.1, thr, None) == {
        "fire": False, "escalated": False, "stage": "cheap_negative", "fallback": False}
    assert C.cascade_decide(0.9, thr, None)["stage"] == "cheap_positive"
    assert C.cascade_decide(0.9, thr, None)["fire"] is True
    band_hi = C.cascade_decide(0.5, thr, 1.0)
    assert band_hi == {"fire": True, "escalated": True, "stage": "band", "fallback": False}
    band_lo = C.cascade_decide(0.5, thr, 0.0)
    assert band_lo["fire"] is False and band_lo["escalated"] is True
    missing = C.cascade_decide(0.5, thr, None)
    assert missing["escalated"] is True and missing["fallback"] is True
    assert missing["fire"] is C.BAND_FALLBACK_FIRE


def test_band_edges_are_inclusive_per_figure_1():
    thr = C.CascadeThresholds(t1=0.3, t_star=0.7, t2=0.5, p_target=0.4, p_realized=0.4)
    assert C.cascade_decide(0.3, thr, 1.0)["escalated"] is True     # s == t1 -> band
    assert C.cascade_decide(0.7, thr, 1.0)["escalated"] is True     # s == t* -> band
    assert C.cascade_decide(0.7001, thr, 1.0)["escalated"] is False


def test_band_with_ties_reports_realized_p_not_target():
    s = np.array([0.0] * 5 + [1.0] * 5)
    t_star, realized, k = C.band_upper_threshold(s, t1=0.0, p=0.3)
    assert k == 3
    # ties make the exact fraction unattainable; the realized value is returned
    assert realized == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 2. p-sweep cost monotonicity
# ---------------------------------------------------------------------------

def test_escalation_and_cost_monotone_in_p_at_fixed_t1():
    rng = np.random.default_rng(3)
    s = rng.random(80)
    qids = [f"s{i:02d}:{i%4}" for i in range(80)]
    cost = {"base_sec": {q: 1.0 for q in qids},
            "expensive_sec": {q: 0.5 for q in qids}}
    prev_esc, prev_cost = -1, -1.0
    for p in C.P_GRID:
        t_star, _r, _k = C.band_upper_threshold(s, t1=0.0, p=p)
        thr = C.CascadeThresholds(t1=0.0, t_star=t_star, t2=0.5, p_target=p, p_realized=_r)
        out = C.apply_cascade(s, thr, [1.0] * len(s))
        c = C.cascade_cost(qids, out["escalated"], cost)
        assert c["n_escalated"] >= prev_esc
        assert c["total_gpu_sec"] >= prev_cost - 1e-12
        prev_esc, prev_cost = c["n_escalated"], c["total_gpu_sec"]
    assert prev_esc == len(s)          # p = 1.0 escalates everything above t1


def test_cost_charges_base_for_every_step():
    """DEVIATION: unlike the paper's relative-FLOPs metric, the base generation
    is charged for every step (SPEC 5.6 frozen sum)."""
    rows = [_row("a:1", pred=_call()), _row("b:1", pred=_call())]
    cost = C.gpu_sec_columns(rows, {"a:1": {"d_corr_slice_prefill_sec": 0.4,
                                            "generate_sec": 1.1}})
    assert cost["base_sec"]["a:1"] == pytest.approx(0.1 + 0.2 + 0.3 + 0.05 + 1.0)
    assert cost["expensive_sec"]["b:1"] is None
    c = C.cascade_cost(["a:1", "b:1"], [True, True], cost)
    assert c["n_escalated"] == 2 and c["n_escalated_unpriced"] == 1
    assert c["escalation_gpu_sec"] == pytest.approx(1.5)
    assert c["total_gpu_sec"] == pytest.approx(2 * 1.65 + 1.5)


def test_t2_grid_degenerates_to_one_interior_cut_for_a_binary_stage2():
    """DEVIATION check: s_expensive is binary, so q2 = 0 fires on EVERY
    escalation and any q2 > 0 fires iff the action changed."""
    s = np.linspace(0.0, 1.0, 40)
    exp = [1.0 if i % 3 == 0 else 0.0 for i in range(40)]
    fired = {}
    for q2 in C.Q2_GRID:
        thr = C.thresholds_from_levels(s, exp, q1=0.0, q2=q2, p=1.0)
        out = C.apply_cascade(s, thr, exp)
        assert out["escalated"].all()          # p = 1.0 -> the whole frame is band
        fired[q2] = out["fire"].copy()
    assert fired[0.0].all(), "q2 = 0 must fire on every escalation"
    changed = np.array([v == 1.0 for v in exp])
    for q2 in C.Q2_GRID:
        if q2 > 0.0:
            assert np.array_equal(fired[q2], changed), q2


def test_cascade_thresholds_are_constructible_from_three_absolute_cuts():
    """The wrapper contract for t34_judge: (t1, t*, t2) alone is enough, and the
    unmeasured budget fields stay NaN rather than being invented."""
    thr = C.CascadeThresholds(t1=0.2, t_star=0.8, t2=0.5)
    assert np.isnan(thr.p_target) and np.isnan(thr.p_realized)
    assert C.cascade_decide(0.1, thr, None)["stage"] == "cheap_negative"
    assert C.cascade_decide(0.5, thr, 1.0)["stage"] == "band"
    assert C.cascade_decide(0.9, thr, None)["stage"] == "cheap_positive"


# ---------------------------------------------------------------------------
# 3. selection discipline: the fitter never sees outer-test rows
# ---------------------------------------------------------------------------

def test_nested_cv_fitter_never_sees_outer_test_rows():
    frame = _synthetic_frame()
    sub = frame.trigger_subset()
    qids = [r["qid"] for r in sub]
    y = np.array([r["label_cw"] for r in sub])
    groups = np.array([session_of(q) for q in qids])
    rows = [frame.c2kv_by_qid[q] for q in qids]
    s = C.Stage1Ladder(4).fit(rows).score(rows, [True] * len(rows))
    exp = [1.0 if i % 2 else 0.0 for i in range(len(qids))]
    seen = []

    def spy(cal_scores, cal_y, cal_exp, p, **kw):
        seen.append(np.asarray(cal_scores, dtype=float).copy())
        return C.fit_thresholds(cal_scores, cal_y, cal_exp, p, **kw)

    res = C.cascade_nested_cv(s, y, exp, groups, 0.2, fit_fn=spy)
    assert res["folds"], "no outer folds ran"
    assert len(seen) == len(res["folds"])
    # EXACT check: fold i's calibration vector must be the complement of fold
    # i's test mask -- element by element, not merely "smaller than the frame".
    masks = [m for m in _folds(groups, 5)
             if m.sum() and (~m).sum() and y[~m].sum() not in (0, int((~m).sum()))]
    assert len(masks) == len(seen)
    for m, cal in zip(masks, seen):
        assert len(cal) == int((~m).sum())
        assert np.array_equal(cal, s[~m])
    # sessions are never split across train/test
    for m in _folds(groups, 5):
        assert len(set(groups[m]) & set(groups[~m])) == 0


def _folds(groups, k, seed=20260905):
    from t34_common import grouped_folds
    return grouped_folds(groups, k, seed)


def test_fit_thresholds_reports_combination_count():
    rng = np.random.default_rng(5)
    s = rng.random(60)
    y = (rng.random(60) < 0.5).astype(int)
    exp = [float(v) for v in (rng.random(60) < 0.5)]
    groups = np.array([f"s{i%10}" for i in range(60)])
    thr, rep = C.fit_thresholds(s, y, exp, 0.4, groups=groups)
    assert rep["n_combos"] == len(C.Q1_GRID) * len(C.Q2_GRID)
    assert rep["n_evaluations"] == rep["n_combos"] * rep["inner_folds"]
    assert thr.p_target == 0.4
    assert thr.t_star >= thr.t1 or thr.p_realized == 0.0


def test_outer_fold_thresholds_are_refit_per_fold():
    frame = _synthetic_frame()
    sub = frame.trigger_subset()
    qids = [r["qid"] for r in sub]
    y = np.array([r["label_cw"] for r in sub])
    groups = np.array([session_of(q) for q in qids])
    rows = [frame.c2kv_by_qid[q] for q in qids]
    s = C.Stage1Ladder(4).fit(rows).score(rows, [True] * len(rows))
    exp = _stage2_vector(frame, qids)
    res = C.cascade_nested_cv(s, y, exp, groups, 0.4)
    assert len(res["folds"]) >= 2
    assert res["n_threshold_combos"] == len(C.Q1_GRID) * len(C.Q2_GRID)
    assert res["scored"].all()
    assert res["ladder_refit_per_fold"] is False
    # thresholds are genuinely refit per fold
    assert len({(f["t1"], f["t_star"], f["t2"]) for f in res["folds"]}) > 1


def test_ladder_ecdf_is_refit_on_each_outer_train_fold():
    frame = _synthetic_frame()
    sub = frame.trigger_subset()
    qids = [r["qid"] for r in sub]
    y = np.array([r["label_cw"] for r in sub])
    groups = np.array([session_of(q) for q in qids])
    rows = [frame.c2kv_by_qid[q] for q in qids]
    s = C.Stage1Ladder(4).fit(rows).score(rows, [True] * len(rows))
    exp = _stage2_vector(frame, qids)
    res = C.cascade_nested_cv(s, y, exp, groups, 0.4, rows=rows, n_rungs=4,
                              expects_call=[True] * len(rows))
    assert res["ladder_refit_per_fold"] is True
    assert res["scored"].all()
    assert len(res["folds"]) >= 2


def test_lazy_cascade_calls_the_expensive_stage_only_inside_the_band():
    """The budget dial only means something if stage 2 is lazy (Sec. 5)."""
    s = np.linspace(0.0, 1.0, 50)
    t_star, realized, k = C.band_upper_threshold(s, t1=0.2, p=0.2)
    thr = C.CascadeThresholds(t1=0.2, t_star=t_star, t2=0.5, p_target=0.2,
                              p_realized=realized)
    calls = []

    def expensive(key):
        calls.append(key)
        return 1.0

    out = C.cascade(s, expensive, thr, keys=[f"q{i}" for i in range(len(s))])
    in_band = int(((s >= thr.t1) & (s <= thr.t_star)).sum())
    assert out["n_expensive_calls"] == in_band == len(calls) == k
    assert out["escalated"].sum() == in_band
    assert np.array_equal(out["fire"], (s > thr.t_star) | out["escalated"])
    assert np.isnan(out["s_expensive"][~out["escalated"]]).all()


def _stage2_vector(frame, qids):
    table = _expensive_for(frame)
    return [C.stage2_action_changed(frame.c2kv_by_qid[q]["prediction"],
                                    table[q]["prediction"]) for q in qids]


# ---------------------------------------------------------------------------
# 4. transfer-regime bookkeeping
# ---------------------------------------------------------------------------

def test_two_transfer_regimes_are_bookkept_separately():
    frame = _synthetic_frame()
    sub = frame.trigger_subset()
    qids = [r["qid"] for r in sub]
    y = np.array([r["label_cw"] for r in sub])
    groups = np.array([session_of(q) for q in qids])
    rows = [frame.c2kv_by_qid[q] for q in qids]
    s = C.Stage1Ladder(4).fit(rows).score(rows, [True] * len(rows))
    exp = _stage2_vector(frame, qids)
    prac = C.cascade_nested_cv(s, y, exp, groups, 0.4)
    orac = C.cascade_oracle_regime(s, y, exp, groups, 0.4)
    assert prac["regime"] == "practical_cross_session"
    assert orac["regime"] == "oracle_same_qid"
    assert len(orac["folds"]) == 1 and orac["folds"][0]["n_test"] == len(y)
    assert orac["folds"][0]["inner_folds"] == 0        # no inner CV in the oracle regime
    assert len(prac["folds"]) >= 2
    assert all(f["inner_folds"] == 3 for f in prac["folds"])
    # the oracle regime is the argmax over the SAME grid on the SAME rows
    j_orac = C._youden(orac["fire"], y)
    exhaustive = []
    for q1 in C.Q1_GRID:
        for q2 in C.Q2_GRID:
            thr = C.thresholds_from_levels(s, exp, q1, q2, 0.4)
            exhaustive.append(C._youden(C.apply_cascade(s, thr, exp)["fire"], y))
    assert j_orac == pytest.approx(max(exhaustive))
    # the practical regime is scored on held-out sessions only
    assert prac["scored"].all()


def test_sweep_refuses_p_gt_zero_when_stage2_covers_one_class_only():
    frame = _synthetic_frame()
    sub = frame.trigger_subset()
    partial = {r["qid"]: {"prediction": _call("tool_repaired"),
                          "d_corr_slice_prefill_sec": 0.4, "generate_sec": 1.0}
               for r in sub if r["label_cw"] == 1}          # positives only, like d_corr
    res = C.p_sweep(frame, expensive=partial, n_rungs=1, reps=50)
    assert res["expensive_coverage"]["ok"] is False
    assert [r["p"] for r in res["p_rows"]] == [0.0]
    assert {r["p"] for r in res["refusals"]} == set(C.P_GRID) - {0.0}
    res2 = C.p_sweep(frame, expensive=partial, n_rungs=1, reps=50,
                     allow_partial_expensive=True)
    fams = {r["family"] for r in res2["p_rows"] if r["p"] > 0}
    assert fams == {"diagnostic_partial"}


def test_sweep_refuses_p_gt_zero_with_no_stage2_table_at_all():
    frame = _synthetic_frame()
    res = C.p_sweep(frame, expensive=None, n_rungs=1, reps=50)
    aud = res["expensive_coverage"]
    assert aud["ok"] is False and aud["n_available"] == 0
    assert "untestable" in aud["reason"]
    assert [r["p"] for r in res["p_rows"]] == [0.0]


# ---------------------------------------------------------------------------
# 5. label discipline
# ---------------------------------------------------------------------------

LABEL_KEYS = ("target", "target_has_tool_call", "target_tool_name",
              "tool_name_match", "exact_match", "prediction_tool_name")


def test_feature_path_identical_with_label_fields_deleted():
    """Mechanical assertion: the whole feature path is invariant to deleting
    every target/label field from the row."""
    frame = _synthetic_frame()
    sub = frame.trigger_subset()
    qids = [r["qid"] for r in sub]
    rows = [frame.c2kv_by_qid[q] for q in qids]
    stripped = []
    for r in rows:
        c = dict(r)
        for k in LABEL_KEYS:
            c.pop(k, None)
        stripped.append(c)
    lad_a = C.Stage1Ladder(4).fit(rows)
    lad_b = C.Stage1Ladder(4).fit(stripped)
    a = lad_a.score(rows, [True] * len(rows))
    b = lad_b.score(stripped, [True] * len(stripped))
    assert np.allclose(a, b)
    for r, srow in zip(rows, stripped):
        assert C.prefix_scalars(r) == C.prefix_scalars(srow)
        assert (C.parse_fail_indicator(r.get("prediction"), True)
                == C.parse_fail_indicator(srow.get("prediction"), True))
        assert (C.canonical_action(r.get("prediction"))
                == C.canonical_action(srow.get("prediction")))
    # stage 2 also only compares two compressed-arm emissions
    tbl = _expensive_for(frame)
    for q in qids:
        st = {k: v for k, v in tbl[q].items()}
        assert (C.stage2_action_changed(frame.c2kv_by_qid[q]["prediction"], st["prediction"])
                == C.stage2_action_changed(stripped[qids.index(q)].get("prediction"),
                                           st["prediction"]))


def test_load_expensive_table_drops_scoring_columns():
    p = ROOT / "results" / "t34" / "_test_expensive.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"qid": "a:1", "prediction": _call(),
                             "tool_name_match": True, "target": "gold",
                             "d_corr_slice_prefill_sec": 0.4,
                             "generate_sec": 1.0}) + "\n", encoding="utf-8")
    try:
        tbl = C.load_expensive_table(p)
        assert set(tbl["a:1"]) == {"prediction", *C.EXPENSIVE_COST_FIELDS}
    finally:
        p.unlink()


def test_parse_fail_indicator_matches_the_frozen_baseline_semantics():
    good, bad = _call(), "no tool call at all"
    for pred in (good, bad):
        for expects in (True, False):
            assert (C.parse_fail_indicator(pred, expects)
                    == parse_fail_baseline(pred, expects))
    assert C.expects_call_from_tools([{"name": "t"}]) is True
    assert C.expects_call_from_tools([]) is False


def test_s0_twin_is_label_side_and_named_as_such():
    assert "label" in C.s_cheap_label_s0_twin.__name__
    assert "label" in C.gold_witness_label_node_features.__name__
    assert "label" in C.label_witness_values.__name__
    frame = _synthetic_frame()
    sub = frame.trigger_subset()
    qids = [r["qid"] for r in sub]
    rows = [frame.c2kv_by_qid[q] for q in qids]
    frows = [frame.full_by_qid[q] for q in qids]
    lad = C.Stage1Ladder(2).fit(rows)
    twin = C.s_cheap_label_s0_twin(frows, lad, [True] * len(frows))
    assert twin.shape == (len(qids),)


# ---------------------------------------------------------------------------
# 6. ladder / controls
# ---------------------------------------------------------------------------

def test_rung0_is_exactly_the_parse_failure_baseline():
    frame = _synthetic_frame()
    rows = [frame.c2kv_by_qid[r["qid"]] for r in frame.trigger_subset()]
    s = C.Stage1Ladder(1).fit(rows).score(rows, [True] * len(rows))
    expect = np.array([float(C.parse_fail_indicator(r["prediction"], True)) for r in rows])
    assert np.array_equal(s, expect)


def test_ladder_never_reranks_the_baseline_away():
    frame = _synthetic_frame()
    rows = [frame.c2kv_by_qid[r["qid"]] for r in frame.trigger_subset()]
    lad = C.Stage1Ladder(4).fit(rows)
    s = lad.score(rows, [True] * len(rows))
    fail = np.array([C.parse_fail_indicator(r["prediction"], True) for r in rows])
    assert s[fail].min() > s[~fail].max()


def test_ladder_audit_flags_undefined_and_constant_rungs():
    frame = _synthetic_frame()
    rows = [frame.c2kv_by_qid[r["qid"]] for r in frame.trigger_subset()]
    a = C.Stage1Ladder(len(C.STAGE1_LADDER)).fit(rows).audit()
    assert "s9_ledger_gap" in a["unusable"]          # single frame on this face
    assert "s10_hybrid_tail_frac" in a["unusable"]   # hybrid off -> structurally absent
    assert "s8_compression_ratio" in a["usable"]


def test_ecdf_is_fit_on_calibration_rows_only():
    rows_a = [_row(f"s{i}:0", pred=_call(), ratio=float(i)) for i in range(10)]
    rows_b = [_row(f"s{i}:1", pred=_call(), ratio=100.0 + i) for i in range(10)]
    lad = C.Stage1Ladder(2).fit(rows_a)
    s_b = lad.score(rows_b, [True] * len(rows_b))
    # every out-of-range row saturates the train ECDF at 1.0 -> the test
    # distribution never re-normalises the score
    assert np.allclose(s_b, 0.999)


def test_random_at_matched_rate_precision_is_the_prevalence():
    y = np.array([1] * 93 + [0] * 68)
    r = C.random_at_matched_rate(y, 40, reps=400, seed=7)
    assert r["precision_mean"] == pytest.approx(93 / 161, abs=0.03)


def test_oracle_trigger_ceiling_is_perfect_and_priced():
    y = np.array([1, 0, 1, 0])
    qids = ["a:1", "a:2", "b:1", "b:2"]
    cost = {"base_sec": {q: 1.0 for q in qids}, "expensive_sec": {q: 0.5 for q in qids}}
    row = C.oracle_trigger_ceiling(y, qids, cost)
    assert row["coverage"] == 1.0 and row["precision"] == 1.0 and row["false_reset"] == 0.0
    assert row["family"] == "oracle"
    assert row["cost"]["n_escalated"] == 2


def test_trigger_metrics_use_the_frames_own_denominators():
    y = np.array([1] * 93 + [0] * 68)
    fire = np.zeros(161, dtype=bool)
    fire[:35] = True
    fire[93:93 + 28] = True
    m = C.trigger_metrics(fire, y)
    assert m["coverage"] == pytest.approx(35 / 93)
    assert m["false_reset"] == pytest.approx(28 / 68)
    assert m["precision"] == pytest.approx(35 / 63)
    assert m["prevalence"] == pytest.approx(93 / 161)


def test_winner_rule_is_evaluated_against_chance_s0_and_length():
    """digest 4.0: chance is the EVALUATION-FRAME prevalence, and the S0 /
    length-control clauses are stated on the paired delta's CI lower bound."""
    frame = _synthetic_frame()
    res = C.p_sweep(frame, expensive=None, n_rungs=4, p_grid=(0.0,), reps=60)
    wr = res["winner_rule"]
    assert wr["chance_ap"] == pytest.approx(res["frame"]["prevalence_is_chance_ap"])
    assert wr["chance_ap"] != pytest.approx(0.1033, abs=1e-3)
    assert set(wr["delta_auprc_vs_length_controls"]) == {n for n, _f, _o in C.LENGTH_CONTROLS}
    for blk in (wr["delta_auprc_vs_s0_twin"], *wr["delta_auprc_vs_length_controls"].values()):
        assert blk["ci95"] is None or blk["ci95"][0] <= blk["delta_auprc"] <= blk["ci95"][1]
    tm = wr["three_metrics_at_baseline_fire_count"]
    assert tm["candidate"]["fires"] == tm["baseline"]["fires"]
    assert wr["verdict"] in ("alive", "dead")
    if wr["verdict"] == "dead":
        assert wr["failed_criteria"]


def test_winner_rule_marks_a_perfect_score_alive_and_a_null_score_dead():
    y = np.array([1] * 40 + [0] * 40)
    clusters = np.array([i % 20 for i in range(80)])
    perfect = y.astype(float) + np.linspace(0, 0.01, 80)
    null = np.zeros(80)
    # a real 0/1 rule that fires on half of each class (coverage .5, fr .5)
    weak = np.zeros(80)
    weak[:20] = 1.0
    weak[40:60] = 1.0
    ctl = {"ctl_input_tokens": np.linspace(0, 1, 80)}
    good = C.winner_rule(perfect, y, clusters, baseline=weak, s0_twin=null,
                         controls=ctl, baseline_fires=40, reps=200)
    assert good["verdict"] == "alive" and good["failed_criteria"] == []
    bad = C.winner_rule(null.copy(), y, clusters, baseline=perfect, s0_twin=perfect,
                        controls=ctl, baseline_fires=40, reps=200)
    assert bad["verdict"] == "dead"
    assert "delta_auprc_vs_s0_twin" in bad["failed_criteria"]
    assert "three_metrics_vs_parse_fail_baseline" in bad["failed_criteria"]


# ---------------------------------------------------------------------------
# 7. GCN ceiling
# ---------------------------------------------------------------------------

def _toy_graphs(n=6, nodes=4, feats=3, seed=0):
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(n):
        X = rng.normal(size=(nodes, feats))
        graphs.append({"X": X, "A_hat": C.normalize_adjacency(C.similarity_graph(X))})
    y = np.array([i % 2 for i in range(n)], dtype=float)
    return graphs, y


def test_gcn_gradients_match_finite_differences():
    graphs, y = _toy_graphs(seed=11)
    p = C.gcn_init(graphs[0]["X"].shape[1], 5, seed=2)
    loss, grad = C.gcn_loss_and_grads(p, graphs, y, l2=0.03)
    flat = p.flat()
    gflat = grad.flat()
    eps = 1e-6
    rng = np.random.default_rng(4)
    idx = rng.choice(flat.size, size=min(12, flat.size), replace=False)
    for i in idx:
        up, dn = flat.copy(), flat.copy()
        up[i] += eps
        dn[i] -= eps
        lu, _ = C.gcn_loss_and_grads(p.like(up), graphs, y, l2=0.03)
        ld, _ = C.gcn_loss_and_grads(p.like(dn), graphs, y, l2=0.03)
        num = (lu - ld) / (2 * eps)
        assert abs(num - gflat[i]) <= 1e-5 + 1e-3 * abs(gflat[i]), (i, num, gflat[i])
    assert np.isfinite(loss)


def test_gcn_relu_gradient_is_exercised():
    """Guard against a linear-only path silently passing the gradient check."""
    graphs, y = _toy_graphs(n=8, seed=21)
    p = C.gcn_init(graphs[0]["X"].shape[1], 6, seed=3)
    hit = 0
    for g in graphs:
        pre = g["A_hat"] @ g["X"] @ p.W1 + p.b1
        hit += int((pre <= 0).any() and (pre > 0).any())
    assert hit > 0, "no graph exercised both sides of the relu"


def test_gcn_fit_reduces_the_loss_and_normalisation_is_symmetric():
    graphs, y = _toy_graphs(n=10, seed=5)
    p0 = C.gcn_init(graphs[0]["X"].shape[1], 6, seed=1)
    l0, _ = C.gcn_loss_and_grads(p0, graphs, y)
    p1 = C.gcn_fit(graphs, y, hidden=6, l2=0.0, epochs=200, seed=1)
    l1, _ = C.gcn_loss_and_grads(p1, graphs, y)
    assert l1 < l0
    A = C.similarity_graph(np.array([[0.0], [1.0], [2.0]]))
    Ah = C.normalize_adjacency(A)
    assert np.allclose(Ah, Ah.T)
    assert np.allclose(np.diag(A), 0.0)


def test_gcn_ceiling_runs_session_grouped_and_reports_denominators():
    rng = np.random.default_rng(9)
    qids, y, groups, vectors = [], [], [], {}
    for s in range(10):
        for t in range(3):
            q = f"g{s:02d}:{t}"
            lab = int((s + t) % 2)
            qids.append(q)
            y.append(lab)
            groups.append(f"g{s:02d}")
            base = 1.0 if lab else 0.0
            vectors[q] = (rng.normal(base, 0.5, size=(4, 2))).tolist()
    res = C.gcn_detection_ceiling(vectors, qids, np.array(y), np.array(groups),
                                  hidden_grid=(4,), l2_grid=(1e-2,), epochs=40)
    assert res["n_scored"] == len(qids)
    assert res["prevalence"] == pytest.approx(0.5)
    assert res["auroc"] is not None and 0.0 <= res["auroc"] <= 1.0
    assert res["family"] == "detector"
    assert res["n_clusters"] == 10
    assert all("hidden" in c and "l2" in c for c in res["chosen"])


def test_query_lexical_node_features_are_gold_free():
    docs = ["the access token is abcdefgh12345 for spotify",
            "unrelated block about weather",
            "playlist library listing"]
    q = "show playlist library for spotify"
    v = C.query_lexical_node_features(docs, q)
    assert len(v) == 3 and len(v[0]) == 3
    scores = [row[0] for row in v]
    assert scores[2] > scores[1]                       # "playlist"/"library" hit doc 2
    assert v[0][2] == 0.0 and v[2][2] == pytest.approx(1.0)


def test_gold_witness_node_features_are_absent_without_the_frozen_table():
    frame = _synthetic_frame()
    assert C.gold_witness_label_node_features(frame, "sess00:0") is None


# ---------------------------------------------------------------------------
# 8. end-to-end on the frozen frame + artefacts
# ---------------------------------------------------------------------------

def test_p_sweep_on_the_synthetic_frame_end_to_end(tmp_path):
    frame = _synthetic_frame()
    res = C.p_sweep(frame, expensive=_expensive_for(frame), n_rungs=4,
                    p_grid=(0.0, 0.2, 1.0), reps=50)
    assert res["frame"]["prevalence_is_chance_ap"] == pytest.approx(
        res["frame"]["n_pos"] / res["frame"]["n"])
    assert res["expensive_coverage"]["ok"] is True
    assert [r["p"] for r in res["p_rows"]] == [0.0, 0.2, 1.0]
    for row in res["p_rows"]:
        for regime in ("practical", "oracle"):
            m = row[regime]
            assert m["n_pos"] + m["n_neg"] == m["n"]
            assert m["n_threshold_combos"] == len(C.Q1_GRID) * len(C.Q2_GRID)
            assert m["cost"]["total_gpu_sec"] > 0
    assert res["p_rows"][0]["practical"]["n_escalated"] == 0
    assert res["oracle_trigger"]["coverage"] == 1.0
    assert res["baseline_parse_fail"]["fires"] > 0
    assert res["s0_twin"]["family"] == "control"
    for name, _f, _o in C.LENGTH_CONTROLS:
        assert name in res["length_control"]
        assert res["length_control"][name]["n_scored"] == res["frame"]["n"]
    assert set(res["strata"]) == {"censored_at_cap", "uncensored"}
    n = C.emit_features(frame, tmp_path / "f.jsonl", n_rungs=4,
                        expensive=_expensive_for(frame))
    assert n == res["frame"]["n"]
    rows = [json.loads(l) for l in (tmp_path / "f.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["s9_ledger_gap"] is None            # null kept, no sentinel
    orient = json.loads((ROOT / "configs/t34/orientations_cascade.json").read_text(encoding="utf-8"))
    for k in rows[0]:
        if k in ("qid", "session_id", "arm"):
            continue
        assert k in orient, f"missing orientation for {k}"


def test_frozen_frame_shape_if_present():
    root = ROOT
    if not (root / "results/bdf_pilot/d_r2/battery_c2kv.jsonl").exists():
        pytest.skip("frozen battery not in this worktree")
    frame = FrozenAssets(root).load()
    sub = frame.trigger_subset()
    assert len(sub) == 161
    assert sum(r["label_cw"] for r in sub) == 93
    # EXPECTS_CALL_ALWAYS is exact on this frame (the DEVIATIONS entry claims it)
    assert all(frame.c2kv_by_qid[r["qid"]]["target_has_tool_call"] for r in sub)
    res = C.p_sweep(frame, expensive=None, n_rungs=len(C.STAGE1_LADDER),
                    p_grid=(0.0,), reps=50)
    assert res["frame"]["n"] == 161 and res["frame"]["n_sessions"] == 100
    assert res["frame"]["prevalence_is_chance_ap"] == pytest.approx(93 / 161, abs=1e-9)
    assert res["baseline_parse_fail"]["fires"] == 63
    assert res["baseline_parse_fail"]["coverage_hits"] == 35
    assert res["baseline_parse_fail"]["false_resets"] == 28


def test_ceiling_cli_refuses_without_a_sidecar(tmp_path, capsys):
    if not (ROOT / "results/bdf_pilot/d_r2/battery_c2kv.jsonl").exists():
        pytest.skip("frozen battery not in this worktree")
    rc = C.main(["ceiling", "--root", str(ROOT), "--node_features", "query_lexical",
                 "--out", str(tmp_path / "c.json")])
    assert rc == 2
    assert "sidecar" in capsys.readouterr().out


def test_ceiling_cli_refuses_a_stale_sidecar(tmp_path, capsys):
    if not (ROOT / "configs/bdf_pilot/d_witness_r2.json").exists():
        pytest.skip("frozen witness table not in this worktree")
    frame = FrozenAssets(ROOT).load()
    side = tmp_path / "sidecar_c2kv.jsonl"
    with side.open("w", encoding="utf-8") as fh:
        for r in frame.trigger_subset():
            fh.write(json.dumps({"qid": r["qid"], "session_id": r["session_id"],
                                 "docs": ["fabricated block a", "fabricated block b"],
                                 "query": "what now", "tools": [], "system_prompt": "",
                                 "doc_lengths": [3, 3], "dropped_docs": [],
                                 "kept_history_tokens": 6}) + "\n")
    rc = C.main(["ceiling", "--root", str(ROOT), "--sidecar", str(side),
                 "--node_features", "query_lexical", "--out", str(tmp_path / "c.json")])
    assert rc == 2
    assert "witness" in capsys.readouterr().out


def test_ceiling_cli_refuses_gold_witness_only_availability(tmp_path, capsys):
    if not (ROOT / "configs/bdf_pilot/d_witness_r2.json").exists():
        pytest.skip("frozen witness table not in this worktree")
    rc = C.main(["ceiling", "--root", str(ROOT), "--node_features", "gold_witness_label",
                 "--out", str(tmp_path / "c.json")])
    assert rc == 2                       # 93/161: availability is the label
    assert "availability" in capsys.readouterr().out


def test_orientation_config_matches_the_declared_ladder():
    """The pre-declared config must not drift from the module's own ladder
    orientations -- the config is the artefact a reviewer reads."""
    orient = json.loads((ROOT / "configs/t34/orientations_cascade.json").read_text(
        encoding="utf-8"))
    for name, _sig, ori, _doc in C.STAGE1_LADDER:
        assert orient[name] == ori, (name, orient[name], ori)
    for name, _field, ori in C.LENGTH_CONTROLS:
        assert orient[name] == ori, (name, orient[name], ori)


def test_ladder_counts_neutral_rank_imputations():
    """The one in-score fallback (0.5 for an undefined usable rung) is COUNTED,
    and a rung undefined on every row is excluded outright instead."""
    rows = [_row(f"s{i}:0", pred=_call(), ratio=float(i)) for i in range(8)]
    for r in rows[:3]:
        r["actual_compression_ratio"] = None          # defined for 5/8 rows
    lad = C.Stage1Ladder(2).fit(rows)
    assert "s8_compression_ratio" in lad.usable
    lad.score(rows, [True] * len(rows))
    a = lad.audit()
    assert a["neutral_rank_imputations"]["s8_compression_ratio"] == 3
    assert a["n_scored"] == 8
    assert a["orientations"]["s8_compression_ratio"] == 1
    # a rung undefined on EVERY row is unusable, not imputed
    a2 = C.Stage1Ladder(len(C.STAGE1_LADDER)).fit(rows).audit()
    assert "s9_ledger_gap" in a2["unusable"]
    assert "s9_ledger_gap" not in a2["neutral_rank_imputations"]


def test_winner_rule_checks_the_uncensored_stratum_and_says_when_it_cannot():
    """digest 4.0 clause: the direction must hold on the uncensored subset.
    Unchecked must never read as passed."""
    y = np.array([1] * 40 + [0] * 40)
    clusters = np.array([i % 20 for i in range(80)])
    ctl = {"ctl_input_tokens": np.linspace(0, 1, 80)}
    weak = np.zeros(80)
    weak[:20] = 1.0
    weak[40:60] = 1.0
    null = np.zeros(80)
    # a score that is perfect overall but INVERTED on the uncensored half
    cens = np.array([i % 2 == 0 for i in range(80)])
    flipped = y.astype(float) + np.linspace(0, 0.01, 80)
    flipped[~cens] = 1.0 - flipped[~cens]
    res = C.winner_rule(flipped, y, clusters, baseline=weak, s0_twin=null,
                        controls=ctl, baseline_fires=40, censored=cens, reps=200)
    unc = res["uncensored_stratum_direction"]
    assert unc["checked"] is True and unc["n"] == 40
    assert unc["direction_holds"] is False
    assert "direction_holds_on_uncensored_stratum" in res["failed_criteria"]
    assert res["verdict"] == "dead"
    # omitting the mask leaves the clause UNCHECKED, listed, and non-passing
    res2 = C.winner_rule(flipped, y, clusters, baseline=weak, s0_twin=null,
                         controls=ctl, baseline_fires=40, reps=200)
    assert res2["uncensored_stratum_direction"]["checked"] is False
    assert res2["uncensored_stratum_direction"]["direction_holds"] is None
    assert "direction_holds_on_uncensored_stratum" in res2["unchecked_criteria"]
    assert "conditional" in res2["verdict_note"]


def test_stage2_block_choice_audit_flags_a_gold_block_index(tmp_path):
    """s_expensive must come from a GOLD-FREE block choice; a dump whose block
    index is the witness k makes every p>0 number an oracle number."""
    frame = _synthetic_frame()
    qids = [r["qid"] for r in frame.trigger_subset()][:4]
    frame.witness = {"entries": {q: {"k_witness": 3, "k_median": 1, "score": [0.0] * 4}
                                 for q in qids}}
    gold = tmp_path / "gold.jsonl"
    free = tmp_path / "free.jsonl"
    with gold.open("w", encoding="utf-8") as fh:
        for q in qids:
            fh.write(json.dumps({"qid": q, "prediction": _call(),
                                 "d_corr_doc_index": 3}) + "\n")
    with free.open("w", encoding="utf-8") as fh:
        for q in qids:
            fh.write(json.dumps({"qid": q, "prediction": _call(),
                                 "d_corr_doc_index": 1}) + "\n")
    bad = C.expensive_block_choice_audit(gold, frame)
    assert bad["gold_free"] is False and bad["n_equal_k_witness_gold"] == 4
    assert "k_witness" in bad["explained_by"]
    good = C.expensive_block_choice_audit(free, frame)
    assert good["gold_free"] is True and good["n_equal_k_median_gold_free"] == 4
    # no witness table -> None, never an assumed pass
    assert C.expensive_block_choice_audit(free, None)["gold_free"] is None
    # a dump neither selector explains everywhere -> unknown, not "gold free"
    mixed = tmp_path / "mixed.jsonl"
    with mixed.open("w", encoding="utf-8") as fh:
        for j, q in enumerate(qids):
            fh.write(json.dumps({"qid": q, "prediction": _call(),
                                 "d_corr_doc_index": 1 if j else 2}) + chr(10))
    unk = C.expensive_block_choice_audit(mixed, frame)
    assert unk["gold_free"] is None and unk["explained_by"] is None


def test_frozen_stage2_dump_is_the_gold_free_arm_if_present():
    if not (ROOT / "results/bdf_pilot/d_r2/d_corr.jsonl").exists():
        pytest.skip("frozen D-line repair dump not in this worktree")
    if not (ROOT / "configs/bdf_pilot/d_witness_r2.json").exists():
        pytest.skip("frozen witness table not in this worktree")
    frame = FrozenAssets(ROOT).load()
    a = C.expensive_block_choice_audit(ROOT / "results/bdf_pilot/d_r2/d_corr.jsonl", frame)
    assert a["n_checked_against_witness"] == 93
    assert a["gold_free"] is True                     # k_median, not k_witness
    assert a["n_equal_k_median_gold_free"] == 93
    # 19 rows agree with k_witness by coincidence; that must not flip the verdict
    assert 0 < a["n_equal_k_witness_gold"] < 93


def test_transfer_gap_is_reported_on_the_selection_objective():
    """The gap must be read on the objective the thresholds were CHOSEN for;
    coverage alone can point the other way and misread as transfer failure."""
    frame = _synthetic_frame()
    res = C.p_sweep(frame, expensive=_expensive_for(frame), n_rungs=4,
                    p_grid=(0.0, 0.4), reps=50)
    for row in res["p_rows"]:
        assert row["transfer_gap_primary"] == "transfer_gap_youden"
        assert "transfer_gap_coverage" in row
        for regime in ("practical", "oracle"):
            assert row[regime]["select_by"] == "youden"
        assert row["transfer_gap_youden"] == pytest.approx(
            row["oracle"]["selection_objective"]
            - row["practical"]["selection_objective"])
        # the oracle regime is the argmax of that objective over the same grid
        # ON these very rows, so on the full-frame scoring it cannot be beaten
        # by any single global pair -- but the practical regime uses a per-fold
        # MIXTURE, so only the oracle's own bound is asserted here.
        assert row["oracle"]["selection_objective"] >= C._youden(
            np.zeros(res["frame"]["n"], dtype=bool),
            np.array([r["label_cw"] for r in frame.trigger_subset()]))


def test_deviations_block_is_complete():
    assert len(C.DEVIATIONS) >= 6
    for d in C.DEVIATIONS:
        assert set(d) == {"method", "paper", "what", "why"}
        assert d["what"] and d["why"]
    assert any("2502.15845" in d["paper"] for d in C.DEVIATIONS)


def test_cli_help_and_subcommands():
    for argv in (["--help"], ["sweep", "--help"], ["ceiling", "--help"]):
        with pytest.raises(SystemExit) as e:
            C.main(argv)
        assert e.value.code == 0


def test_orientations_json_agrees_with_code_constants():
    import t34_cascade as K
    merged = K.assert_orientations_consistent()
    for k, v in K.declared_orientations().items():
        assert merged[k] == v


def test_net_coverage_refuses_unmeasured_harm():
    import t34_cascade as K
    assert K.net_coverage(0.5, 0.9, 0.8, 0.1, None)["net_coverage"] is None
    r = K.net_coverage(0.5, 0.9, 0.8, 0.1, 0.5)
    assert abs(r["net_coverage"] - (0.5 * 0.9 * 0.8 - 0.05)) < 1e-12
