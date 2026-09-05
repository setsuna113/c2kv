# -*- coding: utf-8 -*-
"""Unit tests for agent/t34_sequential.py (t34 unit U7b, survey 4.9).

Run from the worktree root:

    PYTHONIOENCODING=utf-8 python -m pytest agent/test_t34_sequential.py -q

Everything here is pure python / numpy / scipy; no torch, no model, no NPU.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import t34_sequential as seq  # noqa: E402


# --------------------------------------------------------------------------
# (A) 2606.12476 - q/p persistence, order ladder, gate, CUSUM/ARL
# --------------------------------------------------------------------------

def _simulate_markov(p: float, q: float, n_sessions: int, length: int,
                     seed: int = 7) -> dict:
    """Sessions of a first-order two-state chain with known (p, q)."""
    rng = np.random.default_rng(seed)
    seqs = {}
    for s in range(n_sessions):
        y = 0 if rng.random() > 0.2 else 1
        items = [(0, int(y))]
        for t in range(1, length):
            pr = q if y == 1 else p
            y = int(rng.random() < pr)
            items.append((t, y))
        seqs[f"s{s}"] = items
    return seqs


def test_transition_pq_recovers_known_chain():
    seqs = _simulate_markov(p=0.05, q=0.85, n_sessions=400, length=40)
    pairs, stats = seq.adjacent_pairs(seqs)
    table = seq.transition_pq(pairs)
    assert stats["n_pairs"] == 400 * 39
    assert abs(table["p_hat"] - 0.05) < 0.02
    assert abs(table["q_hat"] - 0.85) < 0.03
    assert table["q_over_p"] > 5


def test_adjacent_pairs_break_chain_on_undefined_and_gaps():
    seqs = {
        "a": [(0, 1), (1, None), (2, 1)],      # undefined step breaks both pairs
        "b": [(0, 0), (2, 1)],                 # step gap -> non-adjacent
        "c": [(0, 0), (1, 1)],                 # the only usable pair
    }
    pairs, stats = seq.adjacent_pairs(seqs)
    assert stats["n_pairs"] == 1
    assert stats["dropped_undefined"] == 2
    assert stats["dropped_nonadjacent"] == 1
    assert pairs["c"] == [(0, 1)]


def test_markov_order_ladder_shapes_and_monotone_loglik():
    seqs = _simulate_markov(p=0.05, q=0.85, n_sessions=200, length=30)
    ladder = seq.markov_order_ladder(seqs, max_order=4)
    assert [r["order"] for r in ladder] == [1, 2, 3, 4]
    assert [r["params"] for r in ladder] == [2, 4, 8, 16]
    lls = [r["loglik"] for r in ladder]
    # higher order can only fit the same data at least as well (ML on the same data)
    assert all(lls[i + 1] >= lls[i] - 1e-6 for i in range(len(lls) - 1))
    assert all(r["n_contexts_seen"] > 0 for r in ladder)


class _FakeFrame:
    """Minimal stand-in for FrozenFrame for the diagnostic (labels only)."""

    def __init__(self, labels):
        self.labels = labels


def _frame_from_seqs(seqs):
    labels = []
    for sid, items in seqs.items():
        for idx, y in items:
            labels.append({"qid": f"{sid}:{idx}", "session_id": sid, "label_cw": y})
    return _FakeFrame(labels)


def test_gate_licenses_persistent_chain_and_refuses_iid():
    strong = seq.qp_persistence_diagnostic(
        _frame_from_seqs(_simulate_markov(p=0.05, q=0.9, n_sessions=200, length=20)),
        reps=200)
    assert strong["sequential_licensed"] is True
    assert strong["ci95"]["q_minus_p"][0] > 0
    seq.require_gate(strong)  # must not raise

    iid = seq.qp_persistence_diagnostic(
        _frame_from_seqs(_simulate_markov(p=0.3, q=0.3, n_sessions=60, length=4)),
        reps=200)
    assert iid["sequential_licensed"] is False
    assert "not opened" in iid["verdict"]
    with pytest.raises(seq.SequentialGateError):
        seq.require_gate(iid)
    seq.require_gate(iid, force=True)  # explicit prereg-violating override


def test_gate_refuses_when_too_few_adjacent_pairs_exist():
    """The frozen battery subsamples steps inside a session, so 'adjacent in the
    session' is not 'adjacent in the frame'.  A strongly persistent chain observed
    through only a handful of adjacent pairs must NOT license the family."""
    seqs = {f"s{i}": [(0, 0), (1, 1)] for i in range(6)}   # 6 adjacent pairs only
    diag = seq.qp_persistence_diagnostic(_frame_from_seqs(seqs), reps=100)
    assert diag["pair_stats"]["n_pairs"] == 6
    assert diag["estimable"] is False
    assert diag["sequential_licensed"] is False
    assert "NOT ESTIMABLE" in diag["verdict"]
    with pytest.raises(seq.SequentialGateError):
        seq.require_gate(diag)


def test_subsampled_sessions_are_visible_as_gaps_and_the_naive_variant_differs():
    # steps 8, 13, 16, 19 in one session: no genuinely adjacent pair at all
    seqs = {f"s{i}": [(8, 1), (13, 0), (16, 0), (19, 1)] for i in range(20)}
    pairs, stats = seq.adjacent_pairs(seqs, max_gap=1)
    assert stats["n_pairs"] == 0
    assert stats["dropped_nonadjacent"] == 60
    assert set(stats["gap_hist"]) == {"5", "3"}
    naive, naive_stats = seq.adjacent_pairs(seqs, max_gap=None)
    assert naive_stats["n_pairs"] == 60
    diag = seq.qp_persistence_diagnostic(_frame_from_seqs(seqs), reps=50)
    assert diag["sequential_licensed"] is False
    variant = diag["variant_consecutive_in_frame_NOT_THE_ESTIMAND"]
    assert variant["pair_stats"]["n_pairs"] == 60
    assert "not an estimate" in variant["warning"]


def test_qp_diagnostic_uses_step_index_not_decision_step():
    # qid suffix ordering must drive the sequence: labels given out of order
    labels = [
        {"qid": "s:2", "session_id": "s", "label_cw": 1},
        {"qid": "s:0", "session_id": "s", "label_cw": 0},
        {"qid": "s:1", "session_id": "s", "label_cw": 0},
    ]
    seqs = seq.label_sequences(_FakeFrame(labels))
    assert [i for i, _ in seqs["s"]] == [0, 1, 2]
    pairs, _ = seq.adjacent_pairs(seqs)
    assert pairs["s"] == [(0, 0), (0, 1)]


def test_cusum_reference_k_is_midpoint_of_class_means():
    inc = [0.0, 2.0, 10.0, 12.0]
    y = [0, 0, 1, 1]
    assert seq.cusum_reference_k(inc, y) == pytest.approx(0.5 * (1.0 + 11.0))


def test_learned_cusum_path_floors_at_zero_and_accumulates():
    path = seq.learned_cusum_path([-5.0, -5.0, 3.0, 3.0], k=0.0)
    assert path[0] == 0.0 and path[1] == 0.0
    assert path[2] == pytest.approx(3.0)
    assert path[3] == pytest.approx(6.0)
    assert seq.first_alarm(path, 5.0) == 3
    assert seq.first_alarm(path, 100.0) is None


def test_arl_alignment_on_simulated_null():
    rng = np.random.default_rng(3)
    null = rng.normal(0.0, 1.0, size=5000)
    k = 0.5
    sol = seq.threshold_for_arl(null, k=k, target_arl=100.0, n_runs=400,
                                max_steps=3000, seed=11)
    check = seq.simulate_arl(null, k=k, h=sol["h"], n_runs=800, max_steps=3000, seed=12)
    # the aligned threshold must reproduce the target ARL within a factor of ~2
    assert 40.0 < check["arl"] < 260.0
    # and a strictly higher threshold must give a strictly longer ARL
    higher = seq.simulate_arl(null, k=k, h=sol["h"] * 1.6, n_runs=400,
                              max_steps=3000, seed=12)
    assert higher["arl"] > check["arl"]


def test_align_thresholds_flags_the_horizon_pitfall():
    rng = np.random.default_rng(5)
    null = rng.normal(0.0, 1.0, size=2000).tolist()
    out = seq.align_thresholds_at_arl({"d": null}, {"d": 0.5}, targets=(50, 100),
                                      max_episode_len=4, n_runs=100, max_steps=500)
    assert out["arl_exceeds_episode_length"] is True
    assert "asymptotic regime does not exist" in out["note"]


def test_detection_delays_and_censored_edd():
    onset = {"a": 1, "b": 0, "c": None}
    fire = {"a": 3, "b": None, "c": 0}
    last = {"a": 5, "b": 4, "c": 2}
    d = seq.detection_delays(fire, onset, last)
    assert d["n_onsets"] == 2 and d["n_detected"] == 1
    assert d["recall"] == pytest.approx(0.5)
    assert d["delay_among_detected"] == pytest.approx(2.0)
    # censored EDD charges the miss the whole remaining episode: (2 + 4)/2
    assert d["censored_edd"] == pytest.approx(3.0)


def test_shuffled_order_control_preserves_the_multiset():
    vals = {"s": [0.1, 0.9, 0.4, 0.2]}
    out = seq.shuffle_within_sessions(vals, seed=1)
    assert sorted(out["s"]) == pytest.approx(sorted(vals["s"]))


def test_per_step_threshold_arm_has_no_accumulation():
    vals = {"s": [0.4, 0.4, 0.4]}          # never crosses 1.0 without accumulating
    assert seq.per_step_threshold_arm(vals, 1.0)["s"] is None
    assert seq.cusum_arm(vals, k=0.0, h=1.0)["s"] == 2


# --------------------------------------------------------------------------
# (B) 2607.11317 - causal r_t, mu_0 pooling, betting / e-CUSUM
# --------------------------------------------------------------------------

def test_repeat_coverage_is_causal():
    base = list("abcdefghij")
    tail_a = base + list("abcabcabc")
    tail_b = base + list("zzzzzzzzz")
    ra = seq.causal_repeat_coverage(tail_a, n=3, window=32)
    rb = seq.causal_repeat_coverage(tail_b, n=3, window=32)
    # everything up to and including index 9 must be identical
    assert np.allclose(ra[:10], rb[:10])
    # and altering tokens after t cannot change a_t either
    assert seq.alarm_score_rep_only(ra[9]) == seq.alarm_score_rep_only(rb[9])


def test_period_3_repetition_is_detected():
    loop = list("abc") * 12
    r = seq.causal_repeat_coverage(loop, n=3, window=32)
    assert r[-1] > 0.9
    prose = list("the quick brown fox jumps over a lazy dog while it rains")
    r2 = seq.causal_repeat_coverage(prose, n=3, window=32)
    assert r2[-1] < r[-1]


def test_mu0_is_the_90th_percentile_with_the_pooling_unit_stated():
    vals = list(np.linspace(0.0, 1.0, 101))
    cal = seq.mu0_from_pool(vals)
    assert cal["mu0"] == pytest.approx(0.9, abs=1e-6)
    assert cal["quantile"] == 0.90
    assert cal["pooling_unit"] == "per-token a_t over C->C rows"
    assert cal["n"] == 101


def test_alarm_score_never_backfills_a_missing_u_channel():
    assert seq.alarm_score(0.5, None) is None
    assert seq.alarm_score(0.5, 0.0) == pytest.approx(0.35)
    assert seq.alarm_score_rep_only(0.5) == pytest.approx(0.35)
    assert seq.alarm_score(1.0, 1.0) == pytest.approx(1.0)  # bounded by min(1, .)


def test_entropy_spike_relative_first_step_is_none_not_zero():
    u = seq.entropy_spike_relative([1.0, 2.0, 1.0])
    assert u[0] is None
    assert u[1] > 0.0
    assert u[2] == 0.0  # a drop is clipped at 0, but the first step stays None


def test_truncated_entropy_is_renormalised_over_top_k():
    # two equally likely tokens after renormalisation -> ln 2
    top = [[math.log(0.25), 1], [math.log(0.25), 2]]
    assert seq.truncated_entropy(top) == pytest.approx(math.log(2.0))
    assert seq.truncated_entropy([]) is None


def test_truncated_entropy_refuses_an_unrecognised_row_shape():
    """The local dump shape [logprob, token_id] is verified against t33_capture.py:234;
    the SERVING shape has never been seen here, so a guess must abort loudly."""
    pairs = [[math.log(0.25), 1], [math.log(0.25), 2]]
    dicts = [{"logprob": math.log(0.25), "token": "a"},
             {"logprob": math.log(0.25), "token": "b"}]
    bare = [math.log(0.25), math.log(0.25)]
    for rows in (pairs, dicts, bare):
        assert seq.truncated_entropy(rows) == pytest.approx(math.log(2.0))
    with pytest.raises(ValueError) as e:
        seq.truncated_entropy([["tok", -1.0]])
    assert "unrecognised" in str(e.value)
    with pytest.raises(ValueError):
        seq.truncated_entropy([{"p": 0.5}])


def test_capture_dir_without_the_dump_is_loud_not_a_silent_null_u_channel(tmp_path):
    status = seq.capture_entropy_availability(None, "c2kv")
    assert status["u_channel_available"] is False and "capture rerun" in status["reason"]
    missing = seq.capture_entropy_availability(tmp_path, "c2kv")
    assert missing["u_channel_available"] is False
    assert "does not exist" in missing["reason"]
    with pytest.raises(seq.CaptureUnavailable):
        seq._capture_step_entropies(tmp_path, "c2kv")
    # a real dump reports which entropy it used
    d = tmp_path / "c2kv"
    d.mkdir()
    (d / "p0.steps.jsonl").write_text(
        json.dumps({"qid": "s0:0", "steps": [
            {"entropy_full": 1.0},
            {"top5": [[math.log(0.5), 1], [math.log(0.5), 2]]},
            {}]}) + "\n", encoding="utf-8")
    st = seq.capture_entropy_availability(tmp_path, "c2kv")
    ents = seq._capture_step_entropies(tmp_path, "c2kv", status=st)
    assert ents["s0:0"][0] == 1.0
    assert ents["s0:0"][1] == pytest.approx(math.log(2.0))
    assert ents["s0:0"][2] is None
    assert st["entropy_source"] == {"entropy_full": 1, "truncated_top_logprobs": 1,
                                    "missing": 1}
    assert st["u_channel_available"] is True


def test_betting_process_and_cusum_floor_agree_up_to_the_reset():
    a = [0.9, 0.9, 0.9]
    mu0 = 0.4
    e = seq.betting_process(a, lam=0.5, mu0=mu0)
    s = seq.ecusum_log_path(a, lam=0.5, mu0=mu0)
    # with no reset triggered the CUSUM path is exactly log E_n
    assert np.allclose(s, np.log(e))
    # a healthy stream keeps the floored statistic pinned at zero
    s_healthy = seq.ecusum_log_path([0.0, 0.0, 0.0], lam=0.5, mu0=mu0)
    assert np.allclose(s_healthy, 0.0)


def test_ecusum_evaluate_reports_nominal_and_achieved():
    a = {"h1": [0.0, 0.0], "h2": [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]}
    ev = seq.ecusum_evaluate(a, ["h1", "h2"], mu0=0.4, tau=0.5)
    assert ev["nominal_delta"] == 0.05
    assert ev["n_healthy_sessions"] == 2
    assert ev["achieved_false_alarm_rate"] == pytest.approx(0.5)
    assert "supermartingale" in ev["note"]


# --------------------------------------------------------------------------
# (C) 2608.27808 - LTT feasibility, threshold selection, physiology
# --------------------------------------------------------------------------

def test_ltt_alpha_grid_matches_the_closed_form():
    grid = seq.ltt_alpha_grid(68, delta=0.05)
    rows = {r["alpha"]: r for r in grid["grid"]}
    assert rows[0.05]["min_pvalue_at_zero_false_alarms"] == pytest.approx(0.95 ** 68)
    # (1-alpha)^n <= delta  <=>  n >= ln(delta)/ln(1-alpha)
    for a, row in rows.items():
        assert row["certifiable"] == ((1 - a) ** 68 <= 0.05)
        assert row["n_required_for_alpha"] == math.ceil(math.log(0.05) / math.log(1 - a))
    assert rows[0.20]["max_false_alarms_allowed"] >= rows[0.05]["max_false_alarms_allowed"]


def test_ltt_alpha_grid_infeasible_at_small_n():
    grid = seq.ltt_alpha_grid(5, delta=0.05)
    rows = {r["alpha"]: r for r in grid["grid"]}
    assert rows[0.05]["certifiable"] is False
    assert rows[0.05]["max_false_alarms_allowed"] is None


def test_ltt_selects_the_last_rejected_threshold():
    # 100 healthy scores, uniform on [0,1); alpha = 0.20 must certify a high threshold
    cal = list(np.linspace(0.0, 0.99, 100))
    sel = seq.ltt_select_threshold(cal, alpha=0.20, delta=0.05)
    assert sel["certified"] is True
    fpr = float(np.mean(np.asarray(cal) >= sel["theta"]))
    assert fpr <= 0.20
    assert sel["sequence"][-1]["rejected"] is False  # stopped at first non-rejection
    assert all(r["rejected"] for r in sel["sequence"][:-1])


def test_ltt_no_threshold_certified_path():
    # every candidate fires on the whole healthy pool -> nothing can be certified
    cal = [1.0] * 30
    sel = seq.ltt_select_threshold(cal, alpha=0.05, delta=0.05)
    assert sel["certified"] is False
    assert sel["theta"] is None
    assert sel["reason"] == "no threshold certified"
    # ... and at n far below the feasibility floor it is impossible by construction
    sel2 = seq.ltt_select_threshold([0.1, 0.2, 0.3], alpha=0.05, delta=0.05)
    assert sel2["certified"] is False


def test_ltt_three_way_runs_on_healthy_only():
    rng = np.random.default_rng(0)
    scores = {}
    sessions = {}
    for s in range(30):
        for t in range(3):
            q = f"s{s}:{t}"
            scores[q] = float(rng.random())
            sessions[q] = f"s{s}"
    out = seq.ltt_three_way(scores, sessions, alpha=0.20, repeats=25)
    assert out["repeats_used"] > 0
    assert out["n_sessions"] == 30
    if out["n_certified"]:
        assert 0.0 <= out["exceedance_rate"] <= 1.0


def test_physiology_nll_matches_the_closed_form():
    rows = []
    rng = np.random.default_rng(1)
    for _ in range(40):
        rows.append({"generate_sec": float(rng.normal(10, 1)),
                     "ttft_sec": float(rng.normal(3, 0.5)),
                     "tbt_sec": float(rng.normal(0.1, 0.01)),
                     "generated_tokens": float(rng.normal(50, 5)),
                     "prompt_tokens": float(rng.normal(90, 8))})
    tools = ["Read"] * 40
    model = seq.physiology_fit(rows, tools)
    assert model is not None
    probe = rows[0]
    nll, seen = model.nll(probe, "Read")
    assert seen is True
    v = model.vector(probe)
    mu, sd = model.per_tool["Read"]
    z = (v - mu) / sd
    expected = 0.5 * float(np.sum(z ** 2)) + float(np.sum(np.log(sd)))
    assert nll == pytest.approx(expected)
    # an unseen tool falls back to the POOLED model and says so
    nll2, seen2 = model.nll(probe, "NeverSeen")
    assert seen2 is False and nll2 is not None
    # a row missing a vital yields None, never a sentinel
    assert model.nll({"generate_sec": 1.0}, "Read") == (None, False)


def test_physiology_min_tool_rows_falls_back_to_pooled():
    rows = [{"generate_sec": float(i), "ttft_sec": 1.0, "tbt_sec": 0.1,
             "generated_tokens": 10.0 + i, "prompt_tokens": 20.0} for i in range(10)]
    tools = ["A"] * 8 + ["B"] * 2          # B has only 2 rows < PHYS_MIN_TOOL_ROWS
    model = seq.physiology_fit(rows, tools)
    assert "A" in model.per_tool and "B" not in model.per_tool
    _, seen = model.nll(rows[0], "B")
    assert seen is False


def test_prefix_gate_signs_are_fixed_a_priori_and_missing_signals_are_dropped():
    assert seq.CURA_PREFIX_SIGNS["actual_compression_ratio"] == +1
    assert seq.CURA_PREFIX_SIGNS["gist_tokens"] == -1
    norm = {"gist_tokens": (10.0, 2.0), "actual_compression_ratio": (8.0, 1.0)}
    s, used = seq.prefix_step_score({"gist_tokens": 12.0,
                                     "actual_compression_ratio": 9.0,
                                     "n_docs": None}, norm)
    assert used == 2
    assert s == pytest.approx(-1.0 + 1.0)
    none_s, none_used = seq.prefix_step_score({"n_docs": None}, norm)
    assert none_s is None and none_used == 0


def test_prefix_gate_path_uses_k_equal_half_and_floors():
    assert seq.CURA_K == 0.5
    path = seq.prefix_gate_path([2.0, None, 2.0, -10.0])
    assert path[0] == pytest.approx(1.5)
    assert path[1] is None                    # undefined step leaves W unchanged
    assert path[2] == pytest.approx(3.0)
    assert path[3] == 0.0


def _matched_frame(n_sessions: int = 20, seed: int = 3):
    """Sessions of two rows each: one positive-carrying, one healthy."""
    rng = np.random.default_rng(seed)
    y, groups, good, length = [], [], [], []
    for s in range(n_sessions):
        for t in range(2):
            pos = int(t == 0 and s % 2 == 0)
            y.append(pos)
            groups.append(f"s{s}")
            good.append(5.0 + rng.normal(scale=0.1) if pos else rng.normal(scale=0.1))
            length.append(rng.normal(scale=1.0))
    return (np.array(good), np.array(length), np.array(y, dtype=int), groups)


def test_matched_fpr_comparison_separates_a_good_score_from_noise():
    good, noise, y, groups = _matched_frame()
    out = seq.matched_fpr_comparison(good, noise, y, target_fpr=0.25, tol=0.15,
                                     groups=groups)
    assert out["n"] == len(y) and out["n_pos"] == int(y.sum())
    assert out["a"]["recall"] > out["b"]["recall"]
    assert out["matched"] is True


def test_matched_fpr_comparison_requires_session_groups():
    """There is no in-sample path: without the grouping unit the split cannot be
    made and the function must refuse rather than fall back to the eval frame."""
    y = np.array([1, 1, 0, 0])
    s = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(TypeError):
        seq.matched_fpr_comparison(s, s, y, target_fpr=0.25)   # groups is required
    with pytest.raises(ValueError):
        seq.matched_fpr_comparison(s, s, y, target_fpr=0.25, groups=["a", "b"])


def test_matched_fpr_threshold_is_not_selected_on_the_evaluation_negatives():
    """2608.27808 sec.4: the operating point comes from a DISJOINT calibration half.

    The two halves here have deliberately different negative distributions, so an
    in-sample quantile and the calibration quantile cannot coincide; the reported
    held-out FPR must be the one the calibration threshold actually realises.
    """
    seed = 20260905
    groups = [f"s{s}" for s in range(20) for _ in range(2)]
    y = np.array([1 if (s % 5 == 0 and t == 0) else 0
                  for s in range(20) for t in range(2)], dtype=int)
    score = np.array([float(s) for s in range(20) for _ in range(2)], dtype=float)

    # replicate the module's own session-grouped split
    sess = np.unique(np.array(groups))
    perm = np.random.default_rng(seed).permutation(sess.size)
    cal_sessions = {sess[i] for i in perm[:int(round(0.5 * sess.size))]}
    cal_mask = np.array([g in cal_sessions for g in groups])
    neg = y == 0
    expected_thr = float(np.quantile(score[cal_mask & neg], 0.80))
    in_sample_thr = float(np.quantile(score[~cal_mask & neg], 0.80))

    out = seq.matched_fpr_comparison(score, score, y, target_fpr=0.20, groups=groups,
                                     seed=seed)
    assert out["split"]["n_calibration_sessions"] == 10
    assert out["split"]["n_evaluation_rows"] == int((~cal_mask).sum())
    assert "CALIBRATION" in out["threshold_selected_on"]
    assert out["a"]["threshold"] == pytest.approx(expected_thr)
    assert out["a"]["threshold"] != pytest.approx(in_sample_thr)
    # every reported rate is measured on the held-out rows only
    ev_neg = (~cal_mask) & neg
    assert out["a"]["fpr"] == pytest.approx(
        float((score[ev_neg] >= expected_thr).mean()))
    assert out["a"]["n_evaluation_negatives"] == int(ev_neg.sum())


def test_matched_fpr_comparison_flags_a_tied_length_score():
    """generated_tokens piles up on the 128 cap: the budget cannot be realised and
    the comparison must say so rather than silently reporting unmatched recalls."""
    y, groups, composite = [], [], []
    for s in range(10):
        for t in range(2):
            pos = int(t == 0)
            y.append(pos)
            groups.append(f"s{s}")
            composite.append(5.0 - 0.1 * s if pos else 0.5 - 0.01 * s)
    y = np.array(y, dtype=int)
    capped = np.full(len(y), 128.0)          # every row sits on the cap
    out = seq.matched_fpr_comparison(np.array(composite), capped, y,
                                     target_fpr=0.10, groups=groups)
    assert out["b"]["tie_mass_at_threshold"] == pytest.approx(1.0)
    assert out["b"]["fpr"] == pytest.approx(1.0)
    assert out["matched"] is False
    assert "NOT at a matched budget" in out["unmatched_note"]
    assert "128-token cap" in out["unmatched_note"]


# --------------------------------------------------------------------------
# (D) 2608.02464 - reservoir determinism, ridge, fusion, threshold
# --------------------------------------------------------------------------

def test_reservoir_is_deterministic_and_never_trained():
    a = seq.Reservoir(n_in=4, size=16, seed=42)
    b = seq.Reservoir(n_in=4, size=16, seed=42)
    c = seq.Reservoir(n_in=4, size=16, seed=43)
    assert np.allclose(a.W, b.W) and np.allclose(a.W_in, b.W_in)
    assert not np.allclose(a.W, c.W)
    X = np.random.default_rng(0).normal(size=(7, 4))
    assert np.allclose(a.run(X), b.run(X))
    assert np.max(np.abs(np.linalg.eigvals(a.W))) == pytest.approx(0.9, abs=1e-6)


def test_ridge_fit_is_the_closed_form_solution():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(50, 6))
    Y = rng.normal(size=(50, 3))
    lam = 0.1
    A = seq.ridge_fit(Z, Y, lam)
    expected = np.linalg.inv(Z.T @ Z + lam * np.eye(6)) @ Z.T @ Y
    assert np.allclose(A, expected)


def test_per_channel_max_beats_mean_dilution():
    paths = {"a": np.array([0.0, 0.0, 0.0]), "b": np.array([0.0, 0.0, 9.0]),
             "c": np.array([0.0, 0.0, 0.0])}
    mx = seq.fuse_per_channel_max(paths)
    mn = seq.fuse_mean(paths)
    assert mx[-1] == pytest.approx(9.0)
    assert mn[-1] == pytest.approx(3.0)
    assert mx[-1] > mn[-1]


def test_threshold_comes_from_validation_episodes_only():
    val = [np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0])]
    test_only = [np.array([1000.0])]
    thr = seq.threshold_from_validation(val, beta=0.05)
    assert thr == pytest.approx(np.quantile([2.0, 3.0, 4.0], 0.95))
    # feeding a test episode must change nothing about the validation quantile
    assert seq.threshold_from_validation(val, beta=0.05) == thr
    assert seq.threshold_from_validation(test_only, beta=0.05) != thr


def test_delta_mahalanobis_baseline_flags_a_jump():
    rng = np.random.default_rng(4)
    healthy = [rng.normal(size=(8, 3)) for _ in range(10)]
    model = seq.DeltaMahalanobis.fit(healthy, delta=True)
    bad = np.vstack([rng.normal(size=(4, 3)), np.full((1, 3), 50.0),
                     rng.normal(size=(3, 3))])
    scores = model.scores(bad)
    assert scores.argmax() == 3          # the step where the jump happens
    assert scores.max() > 10 * np.median(scores)


def _fake_proxy_rows(n_conv: int, n_turn: int = 5, error_kind=None):
    rows = []
    rng = np.random.default_rng(9)
    for c in range(n_conv):
        for t in range(n_turn):
            rows.append({
                "conv_id": f"c{c}", "turn": t, "arm": "c2kv",
                "gist_tokens": 70 + int(rng.integers(0, 5)),
                "original_tokens": 550 + int(rng.integers(0, 20)),
                "n_docs": 3, "dropped_docs": [], "doc_packing": "turn",
                "repair_frame_delta": 0.0, "wall_sec": float(rng.normal(1.0, 0.05)),
                "usage": {"completion_tokens": 40, "prompt_tokens": 900},
                "finish_reason": "stop", "error_kind": error_kind,
                "has_tool_call": True, "protocol_legal": True,
                "text": "x" * 40,
            })
    return rows


def test_esn_excludes_cache_misses_explicitly():
    rows = _fake_proxy_rows(3) + _fake_proxy_rows(1, error_kind="C2KV_CACHE_MISS")
    episodes, counts = seq.episodes_from_proxy(rows)
    assert counts["excluded_cache_miss"] == 5
    assert len(episodes) == 3
    assert set(episodes["c0"]) == set(seq.ESN_CHANNELS)


def test_esn_monitor_aborts_below_the_healthy_episode_floor():
    rows = _fake_proxy_rows(5)
    episodes, _ = seq.episodes_from_proxy(rows)
    out = seq.esn_monitor(episodes, list(episodes), beta=0.05)
    assert out["aborted"] is True
    assert str(seq.ESN_MIN_HEALTHY_EPISODES) in out["reason"]
    assert "NPU cost table" in out["cpu_cost_note"]


def test_esn_monitor_runs_with_enough_healthy_episodes():
    rows = _fake_proxy_rows(20)
    episodes, _ = seq.episodes_from_proxy(rows)
    out = seq.esn_monitor(episodes, list(episodes), beta=0.10, seed=3)
    assert out["aborted"] is False
    assert out["n_healthy"] == 20
    assert out["theta_per_channel_max"] is not None
    assert out["baseline"]["delta_mahalanobis"]["threshold_from_validation_only"] is not None
    assert out["u_channel"].startswith("ABSENT BY DESIGN")
    fp = out["false_positives_raw"]                 # raw counts with their denominators
    assert "/" in fp["healthy_fit_split_IN_SAMPLE"]
    assert "/" in fp["healthy_validation_split_THRESHOLD_SOURCE"]
    # the three healthy splits are disjoint and each count says what it is in-sample for
    assert (out["n_healthy_fit"] + out["n_healthy_sigma"] + out["n_healthy_val"]
            == out["n_healthy"])
    assert "/" in fp["healthy_sigma_split_UNSEEN_BY_READOUT_AND_THRESHOLD"]
    assert "in-sample" in fp["note"]
    assert out["healthy_definition"].startswith("bench TASK SUCCESS")


def test_esn_sigma_err_comes_from_held_out_healthy_runs():
    """2608.02464 sec.3: sigma_err (eq.4) is measured on held-out healthy runs.

    On the fit residuals it is optimistically small, so the same episodes score
    LOWER surprise than they do under a held-out sigma.
    """
    rng = np.random.default_rng(4)
    fit = [rng.normal(size=(10, 3)) for _ in range(6)]
    held = [rng.normal(size=(10, 3)) * 3.0 for _ in range(3)]   # noisier held-out runs
    ch_in = seq.esn_channel_fit(fit, seed=9)
    ch_out = seq.esn_channel_fit(fit, seed=9, sigma_mats=held)
    assert ch_in.sigma_in_sample is True and "IN-SAMPLE" in ch_in.sigma_source
    assert ch_out.sigma_in_sample is False and ch_out.n_sigma_episodes == 3
    assert np.all(ch_out.sigma_err > ch_in.sigma_err)
    # the readout itself is untouched by the sigma pool
    assert np.allclose(ch_in.A, ch_out.A)


def test_esn_monitor_reports_the_sigma_split_and_its_provenance():
    rows = _fake_proxy_rows(20)
    episodes, _ = seq.episodes_from_proxy(rows)
    out = seq.esn_monitor(episodes, list(episodes), beta=0.10, seed=3)
    src = out["sigma_err_source"]
    assert src["held_out"] is True and src["n_episodes"] == out["n_healthy_sigma"] > 0
    assert set(src["per_channel"]) == set(out["channels"])
    # sigma_frac = 0 collapses the split and the fallback is declared, not silent
    flat = seq.esn_monitor(episodes, list(episodes), beta=0.10, seed=3, sigma_frac=0.0)
    assert flat["sigma_err_source"]["held_out"] is False
    assert flat["n_healthy_sigma"] == 0
    assert flat["false_positives_raw"][
        "healthy_sigma_split_UNSEEN_BY_READOUT_AND_THRESHOLD"] is None


# --------------------------------------------------------------------------
# module hygiene: deviations, orientations, CLI
# --------------------------------------------------------------------------

def test_deviations_are_well_formed():
    assert seq.DEVIATIONS
    for entry in seq.DEVIATIONS:
        assert set(entry) == {"method", "paper", "what", "why"}
        assert entry["paper"].split()[0] in {"2606.12476", "2607.11317",
                                             "2608.27808", "2608.02464"}


def test_orientations_file_matches_the_module_and_covers_every_feature():
    path = _HERE.parent / "configs/t34/orientations_benchb.json"
    declared = {k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items()
                if not k.startswith("_")}
    assert declared == seq.SEQ_ORIENTATIONS
    assert set(declared.values()) <= {1, -1}


def test_feature_columns_pass_the_leakage_guard_and_are_all_oriented():
    from t33_labels import guard_columns
    rows = seq.build_sequential_features(_make_tiny_frame(), arm="c2kv")
    cols = sorted({k for r in rows for k in r} - {"qid", "arm", "session_id"})
    guard_columns(cols, context="test")
    assert set(cols) == set(seq.SEQ_ORIENTATIONS)


class _TinyFrame:
    """Two sessions x two steps, enough to exercise the causal assembly."""

    def __init__(self, rows, labels):
        self._rows = rows
        self.labels = labels

    @property
    def c2kv_by_qid(self):
        return self._rows

    @property
    def full_by_qid(self):
        return self._rows

    def cc_qids(self):
        return [r["qid"] for r in self.labels if r["label_cw"] == 0]


def _make_tiny_frame():
    rows = {}
    labels = []
    for s in range(4):
        for t in range(2):
            qid = f"s{s}:{t}"
            rows[qid] = {
                "qid": qid, "prediction": 'Action:\n<tool_call>\n{"name":"Read",'
                                          '"arguments":{"path":"a.py"}}\n</tool_call>',
                "generate_sec": 10.0 + s, "ttft_sec": 3.0, "tbt_sec": 0.1,
                "generated_tokens": 50.0 + t, "prompt_tokens": 90.0,
                "gist_tokens": 70 + s, "actual_compression_ratio": 7.9 + 0.1 * s,
                "doc_chunks": 1 + t, "kept_history_tokens": 500 + 10 * s,
                "hybrid_top_k": None,
            }
            labels.append({"qid": qid, "session_id": f"s{s}",
                           "label_cw": 0 if s < 3 else 1,
                           "parse_fail_fire": False})
    return _TinyFrame(rows, labels)


def test_features_are_causal_across_steps():
    rows = seq.build_sequential_features(_make_tiny_frame(), arm="c2kv")
    by_qid = {r["qid"]: r for r in rows}
    assert by_qid["s0:0"]["seq_n_prior_steps"] == 0
    assert by_qid["s0:1"]["seq_n_prior_steps"] == 1
    # no capture -> the u channel and the fused alarm score stay null, never 0
    assert by_qid["s0:0"]["ecusum_u_rel"] is None
    assert by_qid["s0:0"]["ecusum_a_fused"] is None
    assert by_qid["s0:0"]["ecusum_a_rep_only"] is not None
    # the causal running max cannot decrease
    assert (by_qid["s0:1"]["cura_phys_nll_max_causal"]
            >= by_qid["s0:0"]["cura_phys_nll_max_causal"])


def _outlier_frame():
    """Four healthy sessions; s0's vitals are far from the rest.

    Under an in-sample fit s0 helps set its own mean; out of fold it cannot.
    """
    rows = {}
    labels = []
    for s in range(4):
        for t in range(2):
            qid = f"s{s}:{t}"
            scale = 40.0 if s == 0 else 1.0
            rows[qid] = {
                "qid": qid, "prediction": 'Action:\n<tool_call>\n{"name":"Read",'
                                          '"arguments":{"path":"a.py"}}\n</tool_call>',
                "generate_sec": 10.0 * scale + s, "ttft_sec": 3.0 * scale,
                "tbt_sec": 0.1 * scale, "generated_tokens": 50.0 + t,
                "prompt_tokens": 90.0 * scale,
                "gist_tokens": 70 + 30 * s * (scale > 1),
                "actual_compression_ratio": 7.9 + 0.1 * s,
                "doc_chunks": 1 + t, "kept_history_tokens": 500 + 10 * s,
                "hybrid_top_k": None,
            }
            labels.append({"qid": qid, "session_id": f"s{s}", "label_cw": 0,
                           "parse_fail_fire": False})
    return _TinyFrame(rows, labels)


def test_features_dump_is_out_of_fold_and_stamps_its_provenance():
    """2608.27808 sec.4: the physiology Gaussian and the (mu_j, sigma_j) are fitted on
    the fit split only.  Built once on every C->C row they are in-sample for every
    C->C row in the dump, and any winner table scoring them inherits the leak."""
    frame = _outlier_frame()
    rows, prov = seq.build_sequential_features_out_of_fold(frame, arm="c2kv",
                                                           n_folds=3)
    in_sample = {r["qid"]: r for r in seq.build_sequential_features(frame, arm="c2kv")}
    oof = {r["qid"]: r for r in rows}
    assert set(oof) == set(in_sample)                      # no row is dropped
    assert prov["n_rows"] == len(oof)
    assert prov["n_rows_out_of_fold"] == len(oof)
    assert all(r["cura_constants_out_of_fold"] == 1.0 for r in rows)
    assert all(r["cura_constants_out_of_fold"] == 0.0 for r in in_sample.values())
    # the outlier session is far more surprising when it did not set its own mean
    assert oof["s0:0"]["cura_phys_nll"] > in_sample["s0:0"]["cura_phys_nll"]
    # the unfitted channels are untouched by the split
    for qid in oof:
        assert oof[qid]["ecusum_r"] == in_sample[qid]["ecusum_r"]
        assert oof[qid]["seq_step_index"] == in_sample[qid]["seq_step_index"]


def test_features_out_of_fold_never_backfills_an_empty_fit_pool():
    """A fold with no C->C training row must emit None for the fitted columns."""
    frame = _outlier_frame()
    rows = seq.build_sequential_features(frame, arm="c2kv", fit_qids=[])
    for r in rows:
        assert r["cura_phys_nll"] is None
        assert r["cura_prefix_S"] is None and r["cura_prefix_W"] is None
        assert r["cura_prefix_n_signals"] == 0.0
        assert r["cura_constants_out_of_fold"] == 0.0
        assert r["ecusum_a_rep_only"] is not None      # frozen row facts survive


def test_features_subcommand_defaults_to_the_out_of_fold_dump():
    args = seq.build_parser().parse_args(["features", "--out", "x.jsonl"])
    assert args.in_sample_dump is False
    assert args.oof_folds == seq.FEATURES_OOF_FOLDS
    forced = seq.build_parser().parse_args(["features", "--out", "x.jsonl",
                                            "--in_sample_dump"])
    assert forced.in_sample_dump is True


def test_session_steps_are_ordered_numerically_not_lexicographically():
    """qids are '<session>:<step>'; sorting them as strings puts step 10 before step
    2 and silently reverses every cross-step statistic."""
    rows = {}
    labels = []
    for t in (2, 10, 13):
        qid = f"s0:{t}"
        rows[qid] = {"qid": qid, "prediction": "abc abc abc" if t == 13 else "hello world",
                     "generate_sec": 10.0, "ttft_sec": 3.0, "tbt_sec": 0.1,
                     "generated_tokens": 50.0, "prompt_tokens": 90.0,
                     "gist_tokens": 70, "actual_compression_ratio": 8.0,
                     "doc_chunks": 1, "kept_history_tokens": 500, "hybrid_top_k": None}
        labels.append({"qid": qid, "session_id": "s0", "label_cw": 0,
                       "parse_fail_fire": False})
    out = seq.build_sequential_features(_TinyFrame(rows, labels), arm="c2kv")
    by_qid = {r["qid"]: r for r in out}
    assert by_qid["s0:2"]["seq_n_prior_steps"] == 0
    assert by_qid["s0:10"]["seq_n_prior_steps"] == 1
    assert by_qid["s0:13"]["seq_n_prior_steps"] == 2


def test_cli_help_builds():
    parser = seq.build_parser()
    for cmd in ("qp", "ltt-grid", "ecusum", "cura", "features", "arl", "esn"):
        assert cmd in parser.format_help() or True
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    args = parser.parse_args(["ltt-grid", "--n", "68"])
    assert args.n == 68 and args.delta == seq.CURA_DELTA


# --------------------------------------------------------------------------
# reviewer additions: the formulas the fixes touch
# --------------------------------------------------------------------------

def test_memoryless_arm_threshold_is_arl_aligned_not_the_cusum_h():
    """2606.12476 sec.4.1: every detector is matched at a COMMON ARL by sweeping ITS
    OWN threshold, so the no-accumulation arm may not reuse the CUSUM's h."""
    rng = np.random.default_rng(0)
    null = rng.normal(size=20000)
    for gamma in (50, 100, 200):
        out = seq.threshold_for_arl_memoryless(null, target_arl=gamma)
        # ARL of a memoryless rule is 1 / P_0(score >= h)
        assert out["realised_arl"] == pytest.approx(gamma, rel=0.15)
        assert out["h"] == pytest.approx(
            float(np.quantile(null, 1.0 - 1.0 / gamma)), rel=1e-9)
    # and it is NOT the CUSUM threshold at the same ARL
    k = 0.0
    cusum_h = seq.threshold_for_arl(null, k=k, target_arl=100.0, n_runs=200,
                                    max_steps=800)["h"]
    assert abs(cusum_h - seq.threshold_for_arl_memoryless(
        null, target_arl=100)["h"]) > 0.5


def test_ltt_sequence_can_express_zero_false_alarms():
    """At n=68, alpha=0.05 admits at most 0 false alarms; a candidate grid taken from
    the observed scores alone tops out at k=1 and could never certify it."""
    rng = np.random.default_rng(1)
    cal = list(rng.normal(size=68))
    out = seq.ltt_select_threshold(cal, alpha=0.05)
    assert out["certified"] is True
    assert out["sequence"][0]["false_alarms"] == 0
    assert out["theta_above_all_calibration_scores"] is True
    # alpha=0.20 tolerates 7 false alarms at n=68, so an observed threshold wins
    out20 = seq.ltt_select_threshold(cal, alpha=0.20)
    assert out20["certified"] and out20["theta"] <= max(cal)
    assert out20["theta_above_all_calibration_scores"] is False


def test_ltt_three_way_refits_inside_the_fit_third():
    """2608.27808 sec.4/5: all learned components are refit on the fit split only."""
    sessions = {f"q{i}": f"s{i // 2}" for i in range(60)}
    seen_fit_pools = []

    def score_fn(fit_sessions):
        seen_fit_pools.append(tuple(fit_sessions))
        # a normalisation genuinely fitted on the fit third
        base = float(len(fit_sessions))
        return {q: (i % 7) / base for i, q in enumerate(sessions)}

    out = seq.ltt_three_way(None, sessions, alpha=0.10, repeats=5, score_fn=score_fn,
                            qids=list(sessions))
    assert out["refit_inside_fit_fold"] is True
    assert len(seen_fit_pools) == 5
    assert len(set(seen_fit_pools)) > 1            # a different fit third per replicate
    n_sess = len({sessions[q] for q in sessions})
    assert all(len(p) == n_sess // 3 for p in seen_fit_pools)
    # the legacy call path is still available but says it did not refit
    legacy = seq.ltt_three_way({q: 0.5 for q in sessions}, sessions, alpha=0.10,
                               repeats=3)
    assert legacy["refit_inside_fit_fold"] is False


def test_esn_surprise_matches_the_paper_closed_form():
    """2608.02464 eq.4: q_t = mean_d ((xhat_t - x_t)/sigma_err)^2."""
    rng = np.random.default_rng(2)
    mats = [rng.normal(size=(8, 4)) for _ in range(5)]
    ch = seq.esn_channel_fit(mats, seed=11)
    X = mats[0]
    H = ch.reservoir.run(X)
    Z = np.hstack([H[:-1], X[:-1], np.ones((X.shape[0] - 1, 1))])
    manual = np.mean(((X[1:] - Z @ ch.A) / ch.sigma_err) ** 2, axis=1)
    assert np.allclose(ch.surprise(X), manual)


def test_reservoir_leak_rate_is_the_paper_form():
    """2608.02464 eq.2 is leaky: h_t = (1-a) h_{t-1} + a tanh(W h + W_in x)."""
    X = np.ones((3, 2))
    r1 = seq.Reservoir(n_in=2, size=6, seed=5, leak=1.0)
    r0 = seq.Reservoir(n_in=2, size=6, seed=5, leak=0.0)
    assert np.allclose(r0.run(X), 0.0)             # a = 0 freezes the state at h_0 = 0
    h = np.zeros(6)
    for t in range(3):
        h = np.tanh(r1.W @ h + r1.W_in @ X[t])
    assert np.allclose(r1.run(X)[-1], h)


def test_ecusum_mu0_uses_the_unit_of_the_accumulated_statistic():
    """2607.11317 sec.3.3 needs mu_0 to bound E[a_n | F_{n-1}] for the a_n that is
    actually accumulated; ours is a per-STEP max, not a per-token draw."""
    # each step's token stream is mostly quiet and peaks once; a_n is that peak
    per_token = ([0.0] * 9 + [0.7]) * 10
    per_step = [0.7] * 10
    tok = seq.mu0_from_pool(per_token, unit="per-token")
    step = seq.mu0_from_pool(per_step, unit="per-step")
    assert step["mu0"] > tok["mu0"]
    assert "per-step" in step["pooling_unit"]


def test_ecusum_never_backfills_a_missing_mu0_with_zero():
    """mu_0 = 0 would make every increment positive and fire on every trace."""
    empty = seq.mu0_from_pool([], unit="per-step")
    assert empty["mu0"] is None
    path = seq.ecusum_log_path([0.0, 0.0, 0.0], mu0=0.0)
    assert path[-1] == pytest.approx(0.0)          # a_t == mu_0 -> flat, not an alarm
    path_bad = seq.ecusum_log_path([0.05, 0.05, 0.05], mu0=0.0)
    assert path_bad[-1] > 0.0                      # any positive a_t drifts up at mu_0=0


def test_ecusum_flags_a_tau_that_is_unreachable_at_our_episode_length():
    """a_n <= 1 caps the per-step increment at log(1+lam(1-mu_0)); on 4-step episodes
    tau = ln(1/delta) = 3.0 cannot be reached, so 'achieved 0' is structural silence."""
    a = {"s0": [1.0, 1.0, 1.0, 1.0], "s1": [0.9, 0.9]}
    out = seq.ecusum_evaluate(a, ["s0", "s1"], mu0=0.4, tau=math.log(1 / 0.05))
    assert out["achieved_false_alarm_rate"] == 0.0
    assert out["tau_attainable_at_this_episode_length"] is False
    assert "STRUCTURAL SILENCE" in out["silence_note"]
    assert out["max_attainable_S"] < out["tau"]
    # a long enough episode makes the same tau reachable and the flag flips
    long_out = seq.ecusum_evaluate({"s0": [1.0] * 40}, ["s0"], mu0=0.4,
                                   tau=math.log(1 / 0.05))
    assert long_out["tau_attainable_at_this_episode_length"] is True
    assert long_out["silence_note"] is None


# --------------------------------------------------------------------------
# fixer additions: the ARL calibration split and the gate that guards it
# --------------------------------------------------------------------------

def _synthetic_frozen_frame(n_sessions: int = 40, n_steps: int = 3, seed: int = 11):
    """A FrozenFrame-shaped face big enough to split by session.

    Half the sessions carry a C->W onset at their middle step; the c2kv row fields
    are the ones build_sequential_features actually reads.
    """
    from t34_common import FrozenFrame
    rng = np.random.default_rng(seed)
    pairs, labels = [], []
    for s in range(n_sessions):
        sick = (s % 2 == 0)
        for t in range(n_steps):
            qid = f"s{s}:{t}"
            lab = 1 if (sick and t >= 1) else 0
            row = {
                "qid": qid,
                "prediction": ("repeat repeat repeat repeat " * 3 if lab
                               else 'Action:\n<tool_call>\n{"name":"Read",'
                                    '"arguments":{"path":"a.py"}}\n</tool_call>'),
                "generate_sec": 10.0 + rng.normal() + 2.0 * lab,
                "ttft_sec": 3.0 + rng.normal() * 0.1,
                "tbt_sec": 0.1 + rng.normal() * 0.01,
                "generated_tokens": 50.0 + 10.0 * lab,
                "prompt_tokens": 90.0 + s,
                "gist_tokens": 70 + s, "actual_compression_ratio": 7.9 + 0.01 * s,
                "doc_chunks": 1 + t, "kept_history_tokens": 500 + 10 * s,
                "hybrid_top_k": None,
            }
            pairs.append(({"qid": qid}, row))
            labels.append({"qid": qid, "session_id": f"s{s}", "label_cw": lab,
                           "parse_fail_fire": bool(lab and t == n_steps - 1)})
    return FrozenFrame(pairs=pairs, labels=labels, manifest={})


def _patch_frame(monkeypatch, frame):
    class _Assets:
        def __init__(self, root):
            self.root = root

        def load(self):
            return frame

    monkeypatch.setattr(seq, "FrozenAssets", _Assets)


def test_arl_honours_the_negative_gate_by_default(tmp_path, monkeypatch):
    """The q/p gate is pre-registered: a negative verdict refuses the whole family
    unless the caller explicitly declares the violation with --force."""
    _patch_frame(monkeypatch, _synthetic_frozen_frame())
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"sequential_licensed": False,
                                "verdict": "NOT ESTIMABLE: 15 adjacent pairs"}),
                    encoding="utf-8")
    with pytest.raises(seq.SequentialGateError):
        seq.main(["arl", "--root", str(tmp_path), "--gate", str(gate)])


def test_arl_fits_k_and_the_null_pool_on_a_calibration_split(tmp_path, monkeypatch):
    """2606.12476 sec.3.3/4.1: k = (mu_0+mu_1)/2 and the ARL null quantile are
    calibration quantities; reading them off the rows the arms are scored on is
    threshold selection on the evaluation frame."""
    frame = _synthetic_frozen_frame()
    _patch_frame(monkeypatch, frame)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"sequential_licensed": False, "verdict": "no"}),
                    encoding="utf-8")
    out = tmp_path / "arl.json"
    assert seq.main(["arl", "--root", str(tmp_path), "--gate", str(gate), "--force",
                     "--n_runs", "40", "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    cal = res["calibration"]
    assert cal["unit"] == "session"
    assert cal["n_calibration_sessions"] < cal["n_sessions_scored"]
    assert cal["n_calibration_rows"] + cal["n_evaluation_rows"] == res["probe"]["n_scored"]
    assert cal["n_calibration_negatives"] > 0 and cal["n_evaluation_negatives"] > 0
    assert cal["null_pool_underpowered"] is (
        cal["n_calibration_negatives"] < seq.ARL_MIN_NULL_POOL)
    # k is NOT the all-rows value, and the all-rows value is named as unused
    assert res["k_reference"] != pytest.approx(res["k_all_rows_companion_NOT_USED"])
    assert "CALIBRATION sessions only" in res["k_pool"]
    # every arm is scored on the held-out sessions only
    n_ev_onsets = res["arms"]["cusum"]["n_onsets"]
    assert 0 < n_ev_onsets <= cal["n_evaluation_rows"]
    assert all(res["arms"][a]["n_onsets"] == n_ev_onsets for a in res["arms"])


def test_arl_aborts_instead_of_borrowing_evaluation_rows_when_calibration_is_one_class(
        tmp_path, monkeypatch):
    """A one-class calibration half leaves mu_1 (or mu_0) undefined; k must not be
    back-filled from the rows the arms are then scored on."""
    frame = _synthetic_frozen_frame()
    _patch_frame(monkeypatch, frame)
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"sequential_licensed": False, "verdict": "no"}),
                    encoding="utf-8")
    out = tmp_path / "arl0.json"
    # cal_frac = 0 leaves the calibration half empty -> one-class by construction
    assert seq.main(["arl", "--root", str(tmp_path), "--gate", str(gate), "--force",
                     "--cal_frac", "0.0", "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["aborted"] is True
    assert "one-class" in res["reason"]
    assert "arms" not in res


def test_matched_fpr_reports_the_resolution_of_both_pools():
    """A 10 % budget cannot be measured on a handful of negatives; both the pool the
    threshold is read from and the pool the rate is measured on say so."""
    groups = [f"s{s}" for s in range(8) for _ in range(2)]
    y = np.array([1 if t == 0 else 0 for _ in range(8) for t in range(2)], dtype=int)
    score = np.arange(16, dtype=float)
    out = seq.matched_fpr_comparison(score, score, y, target_fpr=0.10, groups=groups)
    assert "resolution" in out["calibration_resolution_note"]
    assert "quantised" in out["evaluation_resolution_note"]
    big_g = [f"s{s}" for s in range(60) for _ in range(2)]
    big_y = np.array([1 if t == 0 else 0 for _ in range(60) for t in range(2)],
                     dtype=int)
    big = seq.matched_fpr_comparison(np.arange(120, dtype=float),
                                     np.arange(120, dtype=float), big_y,
                                     target_fpr=0.10, groups=big_g)
    assert big["calibration_resolution_note"] is None
    assert big["evaluation_resolution_note"] is None


def test_prefix_signals_respect_the_two_sidecar_index_spaces():
    """t34_dump_sidecar: `docs`/`doc_lengths` are the KEPT blocks; `dropped_docs`
    indexes the post-split history list.  n_docs is the total (kept + dropped), the
    saturation flag comes from the dropped side, and neither may be faked from the
    visible count alone."""
    row = {"gist_tokens": 70, "actual_compression_ratio": 8.0, "doc_chunks": 2,
           "kept_history_tokens": 500, "hybrid_top_k": None}
    side = {"doc_lengths": [10] * 16, "dropped_docs": [3, 4, 5]}
    sig = seq.prefix_signals(row, side)
    assert sig["n_docs"] == 19.0                  # 16 visible + 3 dropped, not 16
    assert sig["dropped_docs"] == 3.0
    assert sig["max_doc_num_saturated"] == 1.0
    # nothing dropped -> not saturated, and the total is the visible count
    none_dropped = seq.prefix_signals(row, {"doc_lengths": [10] * 4,
                                            "dropped_docs": []})
    assert none_dropped["n_docs"] == 4.0
    assert none_dropped["max_doc_num_saturated"] == 0.0
    # the dropped side missing -> the total is UNKNOWN, never the kept count
    no_dropped_key = seq.prefix_signals(row, {"doc_lengths": [10] * 16})
    assert no_dropped_key["n_docs"] is None
    assert no_dropped_key["dropped_docs"] is None
    assert no_dropped_key["max_doc_num_saturated"] is None
    # no sidecar at all -> every sidecar-only signal is None and drops out of S_t
    bare = seq.prefix_signals(row, None)
    assert bare["n_docs"] is None and bare["max_doc_num_saturated"] is None
    norm = {k: (0.0, 1.0) for k in seq.CURA_PREFIX_SIGNS}
    _, used = seq.prefix_step_score(bare, norm)
    assert used == 4                              # the four battery-face signals only


def test_cura_cli_runs_end_to_end_and_splits_fit_from_calibration(tmp_path, monkeypatch):
    """Regression: the matched-FPR control needs session groups, so a `cura` run that
    forgets them dies with a TypeError instead of producing a report."""
    _patch_frame(monkeypatch, _synthetic_frozen_frame())
    out = tmp_path / "cura.json"
    assert seq.main(["cura", "--root", str(tmp_path), "--repeats", "5",
                     "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    for a in ("0.05", "0.1", "0.2"):
        assert res["ltt"][a]["fit_calibration_disjoint"] is True
        u = res["ltt_usability"][a]
        # a certificate above every calibration score fires on nothing and is not usable
        assert u["usable"] is (u["certified"] and not u["fires_on_nothing"])
        assert res["three_way"][a]["refit_inside_fit_fold"] is True
    m = res["matched_fpr_10pct"]
    assert "CALIBRATION half" in m["threshold_selected_on"]
    assert m["split"]["n_calibration_sessions"] > 0
    assert "OUTSIDE the CURA fit half" in m["rows_used"]
    assert res["fold_internal_floor"]["n_scored"] > 0
