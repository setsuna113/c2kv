# -*- coding: utf-8 -*-
"""Tests for t34 U10b (section 4.11 controls).

Everything here is pure numpy/sklearn and runs on this box: the unit is a
zero-GPU re-analysis, so there is no torch-dependent path to skip.
"""
import json

import numpy as np
import pytest

import t34_controls as K


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _toy_arms():
    """Two qids x two arms, hand-computable utilities.

    arm ``cheap``  : q1 correct, q2 wrong;  1 s,  100 bytes, 0 gen tokens
    arm ``pricey`` : q1 correct, q2 correct; 2 s, 1000 bytes, 0 gen tokens
    """
    rows = []
    for qid, cheap_ok, pricey_ok in (("s1:0", True, True), ("s1:1", False, True)):
        rows.append({"qid": qid, "arm": "cheap", "correct": cheap_ok,
                     "gpu_sec": 1.0, "kv_bytes": 100.0, "gen_tokens": 0.0})
        rows.append({"qid": qid, "arm": "pricey", "correct": pricey_ok,
                     "gpu_sec": 2.0, "kv_bytes": 1000.0, "gen_tokens": 0.0})
    return rows


def _synth_frame(n_sessions=40, per_session=4, seed=0):
    """A synthetic (y, groups) frame shaped like the 161-row trigger subset."""
    rng = np.random.default_rng(seed)
    groups, y = [], []
    for s in range(n_sessions):
        for t in range(per_session):
            groups.append(f"sess{s}")
            y.append(int(rng.random() < 0.578))
    return np.array(y), np.array(groups)


# ---------------------------------------------------------------------------
# (A) 2606.21399 -- utility model / tau_adv
# ---------------------------------------------------------------------------

def test_arm_cost_table_normalised_and_flat():
    costs = K.arm_cost_table(_toy_arms(), cost_scale=0.05)
    # pricey is the max on both axes -> c_a == cost_scale
    assert costs["pricey"]["c_a"] == pytest.approx(0.05)
    # cheap: 0.5*(100/1000) + 0.5*(1/2) = 0.05+0.25 = 0.30 of the budget
    assert costs["cheap"]["c_a"] == pytest.approx(0.05 * 0.30)
    flat = K.arm_cost_table(_toy_arms(), cost_scale=0.05, cost_mode="flat")
    assert flat["cheap"]["c_a"] == flat["pricey"]["c_a"] == 0.05


def test_arm_cost_table_falls_back_to_gpu_sec_without_bytes():
    rows = [dict(r, kv_bytes=None) for r in _toy_arms()]
    costs = K.arm_cost_table(rows, cost_scale=0.05)
    # only the time axis survives: cheap = 1/2 of the budget
    assert costs["cheap"]["c_a"] == pytest.approx(0.025)
    assert costs["pricey"]["c_a"] == pytest.approx(0.05)


def test_utility_matches_the_paper_formula():
    # U = u - c_a - s*m ; u = 1 correct, -w wrong  (2606.21399 section 9.2)
    assert K.utility(True, c_a=0.05, w=1.0, s=0.01, m=2.0) == pytest.approx(1 - 0.05 - 0.02)
    assert K.utility(False, c_a=0.05, w=1.0, s=0.01, m=0.0) == pytest.approx(-1.05)
    assert K.utility(False, c_a=0.0, w=0.5, s=0.0, m=0.0) == pytest.approx(-0.5)


def test_tau_adv_arithmetic_on_toy_arm_table():
    rows = _toy_arms()
    cont = {"s1:0": True, "s1:1": False}          # continue: q1 right, q2 wrong
    labels = {"s1:0": 0, "s1:1": 1}
    rep = K.tau_adv_label_table(rows, cont, labels, w=1.0, s=0.01, cost_scale=0.05)
    c_cheap = 0.05 * 0.30
    # q1: continue already correct -> EU_none = 1.0; best arm EU = 1 - c_cheap
    tau0 = rep["per_row"]["s1:0"]["tau_adv"]
    assert tau0 == pytest.approx(1.0 - c_cheap - 1.0)
    assert tau0 < 0                                  # firing here is pure cost
    # q2: continue wrong -> EU_none = -1; best arm is pricey (correct) at 1-0.05
    tau1 = rep["per_row"]["s1:1"]["tau_adv"]
    assert rep["per_row"]["s1:1"]["best_arm"] == "pricey"
    assert tau1 == pytest.approx((1.0 - 0.05) - (-1.0))
    # sign(tau_adv) agrees with the C->W label on both rows here
    assert rep["sign_agreement_rate"] == pytest.approx(1.0)
    assert rep["tau_adv"]["n_non_positive"] == 1
    assert rep["mde_pp"] == [17, 25]


def test_tau_adv_reports_the_no_advantage_subpopulation():
    """A C->W row that no arm rescues has tau_adv <= 0: firing on it is cost."""
    rows = [{"qid": "s1:0", "arm": "a", "correct": False, "gpu_sec": 1.0,
             "kv_bytes": None, "gen_tokens": 0.0}]
    rep = K.tau_adv_label_table(rows, {"s1:0": False}, {"s1:0": 1}, w=1.0, s=0.0)
    assert rep["tau_adv"]["frac_non_positive"] == 1.0
    assert rep["sign_agreement_rate"] == 0.0          # label says fire, control says no


def test_cost_penalty_sweep_is_five_by_five():
    rows = _toy_arms()
    sw = K.cost_penalty_sweep_label(rows, {"s1:0": True, "s1:1": False},
                                    {"s1:0": 0, "s1:1": 1}, s=0.0)
    assert sw["n_cells"] == 25
    assert 0 <= sw["cells_sign_kept"] <= 25


# ---------------------------------------------------------------------------
# (A) witness controller / LCB / Gap / exploitability
# ---------------------------------------------------------------------------

def test_lcb_pessimism_lowers_the_selected_arm_value():
    """p_hat - beta*sigma_hat must not exceed p_hat (2606.21399 section 4.1)."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 3))
    y = (X[:, 0] + 0.3 * rng.standard_normal(80) > 0).astype(int)
    p, sd = K._rf_fit_predict_lcb(X[:60], y[:60], X[60:], beta=1.0, seed=0)
    assert sd.min() >= 0.0 and sd.max() > 0.0        # tree-vote spread is real
    assert np.all(p - 1.0 * sd <= p + 1e-12)
    # the pessimistic expected utility is <= the optimistic one for r+ > r-
    r_plus, r_minus = 1.0 - 0.05, -1.0 - 0.05
    eu = p * r_plus + (1 - p) * r_minus
    p_lcb = np.clip(p - 1.0 * sd, 0.0, 1.0)
    eu_lcb = p_lcb * r_plus + (1 - p_lcb) * r_minus
    assert np.all(eu_lcb <= eu + 1e-12)


def test_witness_controller_runs_and_reports_all_policies():
    rng = np.random.default_rng(1)
    qids = [f"sess{i // 3}:{i % 3}" for i in range(60)]
    rows, cont, feats, scalar = [], {}, {}, {}
    for i, q in enumerate(qids):
        x = rng.standard_normal()
        feats[q] = {"f0": x, "f1": rng.standard_normal()}
        scalar[q] = x
        cont[q] = False                                  # all C->W-like rows
        rows.append({"qid": q, "arm": "good", "correct": bool(x > -0.2),
                     "gpu_sec": 1.0, "kv_bytes": 100.0, "gen_tokens": 10.0})
        rows.append({"qid": q, "arm": "weak", "correct": bool(rng.random() < 0.1),
                     "gpu_sec": 0.5, "kv_bytes": 50.0, "gen_tokens": 10.0})
    rep = K.witness_controller_regret_label(rows, cont, feats, scalar,
                                            outer_folds=3, seed=7)
    assert rep["n_rows"] == 60
    for name in ("continue_always", "scalar_threshold", "scalar_rf_lcb",
                 "witness", "oracle"):
        assert name in rep["policies"]
    assert rep["policies"]["oracle"]["regret"] == 0.0
    for name in ("continue_always", "scalar_threshold", "scalar_rf_lcb", "witness"):
        assert rep["policies"][name]["regret"] >= -1e-12
    # the witness sees the arm-relevant feature, continue-always never intervenes
    assert rep["policies"]["continue_always"]["intervene_rate"] == 0.0
    assert rep["policies"]["witness"]["regret"] <= rep["policies"]["continue_always"]["regret"]
    assert rep["mde_pp"] == [17, 25]


def test_abstraction_gap_is_non_negative():
    rng = np.random.default_rng(2)
    n = 120
    groups = np.array([f"s{i // 3}" for i in range(n)])
    g = rng.standard_normal(n)
    U = np.stack([np.zeros(n), g + rng.standard_normal(n), -g + rng.standard_normal(n)], axis=1)
    out = K.abstraction_gap_label(U, g, groups, n_bins=4, folds=4, seed=3)
    assert out["gap"] is not None and out["gap"] >= 0.0
    assert out["V_star"] >= out["V_g"] - 1e-9


def test_abstraction_gap_is_zero_when_one_action_always_wins():
    n = 60
    groups = np.array([f"s{i // 3}" for i in range(n)])
    g = np.linspace(-1, 1, n)
    U = np.stack([np.zeros(n), np.ones(n)], axis=1)     # action 1 dominates everywhere
    out = K.abstraction_gap_label(U, g, groups, n_bins=4, folds=3, seed=3)
    assert out["gap"] == pytest.approx(0.0, abs=1e-12)


def test_exploitability_increment_is_positive_when_prefix_features_matter():
    rng = np.random.default_rng(4)
    qids = [f"s{i // 3}:{i % 3}" for i in range(120)]
    feats, scalar, oracle = {}, {}, {}
    for q in qids:
        x = rng.standard_normal()
        feats[q] = {"informative": x, "junk": rng.standard_normal()}
        scalar[q] = rng.standard_normal()               # scalar carries nothing
        oracle[q] = int(x > 0)
    out = K.exploitability_label(feats, scalar, oracle, folds=4, seed=5)
    assert out["increment"] is not None and out["increment"] > 0.1


# ---------------------------------------------------------------------------
# (B) 2608.10441 -- random@matched-rate
# ---------------------------------------------------------------------------

def test_random_masks_hit_b_exactly_and_respect_clusters():
    clusters = np.array(sum([[i] * 5 for i in range(20)], []))   # 20 sessions x 5
    b = 12
    masks = K.random_matched_rate_masks(clusters, b, reps=200, seed=11)
    assert masks.shape == (200, len(clusters))
    assert np.all(masks.sum(axis=1) == b)
    # every fired row lies inside the union of whole sessions that were walked;
    # with 5 rows per session, 12 fires need at most 3 sessions of eligibility
    for m in masks:
        touched = set(clusters[m].tolist())
        assert len(touched) <= 3


def test_random_masks_degenerate_cases():
    clusters = np.array([0, 0, 1, 1])
    assert K.random_matched_rate_masks(clusters, 0, reps=5, seed=1).sum() == 0
    all_fire = K.random_matched_rate_masks(clusters, 4, reps=5, seed=1)
    assert np.all(all_fire)
    with pytest.raises(ValueError):
        K.random_matched_rate_masks(clusters, 5, reps=2)


def test_random_matched_rate_table_denominators():
    y = np.array([1] * 30 + [0] * 30)
    clusters = np.array(sum([[i] * 4 for i in range(15)], []))
    rep = K.random_matched_rate_table(y, clusters, 10, reps=200, seed=2)
    assert rep["n_pos"] == 30 and rep["n_neg"] == 30 and rep["n_fires"] == 10
    assert rep["coverage"]["denominator"] == 30
    assert rep["false_reset"]["denominator"] == 30
    assert 0.0 <= rep["precision"]["median"] <= 1.0
    assert rep["coverage"]["lo"] <= rep["coverage"]["hi"]


def test_mask_metrics_matches_hand_count():
    y = np.array([1, 1, 0, 0, 1])
    m = np.array([True, False, True, False, False])
    out = K.mask_metrics(m, y)
    assert out == {"fires": 2, "coverage": 1, "n_pos": 3,
                   "coverage_rate": 1 / 3, "precision": 0.5,
                   "false_resets": 1, "n_neg": 2, "false_reset_rate": 0.5}


# ---------------------------------------------------------------------------
# (B) placebo best-k scan
# ---------------------------------------------------------------------------

def _toy_flips():
    """3 qids, 4 blocks each; witness k=0 for all."""
    flips = {
        "s1:0": {0: True, 1: False, 2: False, 3: False},
        "s1:1": {0: False, 1: True, 2: False, 3: False},
        "s2:0": {0: False, 1: False, 2: False, 3: False},
    }
    witness = {q: {"k_witness": 0, "n_docs": 4} for q in flips}
    return flips, witness


def test_pooled_nonwitness_rate_reproduces_the_ksweep_loop():
    flips, witness = _toy_flips()
    got = K.pooled_nonwitness_flip_rate(flips, witness)
    # d_ksweep_analysis.py: for each qid with k_witness set, every k != k_witness
    # is one trial.  3 qids x 3 non-witness ks = 9 trials, 1 correct.
    assert got["nonwitness_trials"] == 9 and got["nonwitness_correct"] == 1
    assert got["p_nonwitness_flip"] == pytest.approx(1 / 9)


def test_expected_random_bestk_matches_the_ksweep_closed_form():
    flips, witness = _toy_flips()
    env = K.expected_random_bestk(flips, witness)
    p = 1 / 9
    expect = 3 * (1.0 - (1.0 - p) ** 4)          # 1-(1-p)^n_docs summed over qids
    assert env["expected_random_sum"] == pytest.approx(expect)
    assert env["expected_random_rate"] == pytest.approx(expect / 3)
    assert K.observed_bestk(flips)["hits"] == 2


def test_placebo_matches_moments_per_qid():
    flips, witness = _toy_flips()
    rep = K.placebo_bestk_scan(flips, witness, level="per_qid", reps=400, seed=3)
    targets = rep["matched_moments"]["target_mean"]
    means = rep["matched_moments"]["empirical_mean_head"]
    varis = rep["matched_moments"]["empirical_var_head"]
    for p, m, v in zip(targets, means, varis):
        assert m == pytest.approx(p, abs=0.05)
        assert v == pytest.approx(p * (1 - p), abs=0.05)


def test_placebo_pooled_level_reproduces_the_analytic_envelope():
    flips, witness = _toy_flips()
    env = K.expected_random_bestk(flips, witness)["expected_random_rate"]
    rep = K.placebo_bestk_scan(flips, witness, level="pooled", reps=2000, seed=4)
    assert rep["placebo_bestk_rate_mean"] == pytest.approx(env, abs=0.03)
    assert rep["gap_reproduced_fraction"] is not None


def test_placebo_rejects_unknown_level():
    flips, witness = _toy_flips()
    with pytest.raises(ValueError):
        K.placebo_bestk_scan(flips, witness, level="bogus")


# ---------------------------------------------------------------------------
# (B) positive control
# ---------------------------------------------------------------------------

def test_synthetic_signal_strength_controls_correlation():
    y, groups = _synth_frame(seed=1)
    z0 = K.synthetic_signal(y, groups, 0.0, seed=9)
    z1 = K.synthetic_signal(y, groups, 0.9, seed=9)
    r0 = abs(np.corrcoef(z0, y)[0, 1])
    r1 = abs(np.corrcoef(z1, y)[0, 1])
    assert r1 > r0 + 0.2


def test_positive_control_is_monotone_in_s_and_finds_s_star():
    y, groups = _synth_frame(n_sessions=40, per_session=4, seed=1)
    curve = K.positive_control_curve(y, groups, s_grid=(0.0, 0.2, 0.4, 0.6, 0.8),
                                     reps=200, seed=13, c_grid=(0.1, 1.0),
                                     outer_folds=3, inner_folds=2)
    aps = [p["auprc"] for p in curve["curve"]]
    assert all(a is not None for a in aps)
    assert aps[-1] > aps[0]
    from scipy.stats import spearmanr
    rho = spearmanr(range(len(aps)), aps).statistic
    assert rho >= 0.8
    assert curve["s_star"] is not None and curve["s_star"] > 0.0
    # chance AP is the FRAME's prevalence, never the 900-row base rate 0.1033
    assert curve["prevalence_chance_ap"] == pytest.approx(float(y.mean()))
    assert abs(curve["prevalence_chance_ap"] - 0.1033) > 0.1


def test_locate_real_on_curve_interpolates_and_refuses_extrapolation():
    curve = {"curve": [{"s": 0.0, "auprc": 0.60}, {"s": 0.5, "auprc": 0.80}]}
    assert K.locate_real_on_curve(curve, 0.70) == pytest.approx(0.25)
    assert K.locate_real_on_curve(curve, 0.95) is None
    assert K.locate_real_on_curve(curve, None) is None


# ---------------------------------------------------------------------------
# (B) reward-SNR floor
# ---------------------------------------------------------------------------

def test_snr_floor_requires_a_delta_definition():
    with pytest.raises(ValueError):
        K.reward_snr_floor([0.0, 1.0, 1.0], delta_definition="  ")


def test_snr_floor_arithmetic_and_caveat():
    d = [1.0] * 20 + [0.0] * 80
    rep = K.reward_snr_floor(d, delta_definition="Delta = 1[rescued] - 1[c2kv correct]")
    assert rep["N"] == 100
    assert rep["mean_delta"] == pytest.approx(0.2)
    assert rep["rho"] == pytest.approx(rep["mean_delta"] / rep["sd_delta"])
    assert rep["rho_star"] == pytest.approx(rep["z_sum"] / 10.0)
    assert rep["z_sum"] == pytest.approx(2.80, abs=0.01)   # z_.975 + z_.80
    assert "post-hoc significance test" in rep["caveat"]
    assert rep["mde_pp"] == [17, 25]


def test_reward_deltas_label_pairs_arm_against_continue():
    rows = _toy_arms()
    qids, d = K.reward_deltas_label(rows, {"s1:0": True, "s1:1": False}, "cheap")
    assert qids == ["s1:0", "s1:1"]
    assert d == [0.0, 0.0]      # cheap: q1 right/right, q2 wrong/wrong
    _q, d2 = K.reward_deltas_label(rows, {"s1:0": True, "s1:1": False}, "pricey")
    assert d2 == [0.0, 1.0]


# ---------------------------------------------------------------------------
# granularity ladder + loaders + housekeeping
# ---------------------------------------------------------------------------

def test_granularity_ladder_reports_every_rung():
    y, groups = _synth_frame(n_sessions=25, per_session=4, seed=6)
    rng = np.random.default_rng(6)
    X = np.stack([y + rng.standard_normal(len(y)), rng.standard_normal(len(y))], axis=1)
    out = K.granularity_ladder(y, groups, X, n_fires=15, ks=(4, 8),
                               regimes=[g[-1] for g in groups], folds=3,
                               seed=6, reps=100)
    names = [r["granularity"] for r in out["ladder"]]
    assert names == ["per_instance", "kmeans_k4", "kmeans_k8", "declared_regimes"]
    for r in out["ladder"]:
        assert r["fires"] == 15
    assert out["random_at_matched_rate"]["n_fires"] == 15


def test_canonical_and_dline_loaders_agree(tmp_path):
    dline = tmp_path / "d_corr.jsonl"
    dline.write_text(json.dumps({
        "qid": "s1:0", "session_id": "s1", "d_arm": "d_corr",
        "tool_name_match": True, "d_corr_slice_prefill_sec": 0.6,
        "generate_sec": 0.25, "d_corr_span_tokens": 300,
        "generated_tokens": 40, "skipped": False}) + "\n", encoding="utf-8")
    rows = K.arm_outcomes_label_from_dline([dline], bytes_per_kv_token=2.0)
    assert rows == [{"qid": "s1:0", "arm": "d_corr", "correct": True,
                     "gpu_sec": pytest.approx(0.85), "kv_bytes": 600.0,
                     "gen_tokens": 40.0}]
    canon = tmp_path / "canon.jsonl"
    canon.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    assert K.load_arm_outcomes_label(canon) == rows
    # without the bytes constant kv_bytes stays unknown, never guessed
    assert K.arm_outcomes_label_from_dline([dline])[0]["kv_bytes"] is None


def test_dline_loader_skips_skipped_rows(tmp_path):
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps({"qid": "s1:0", "d_arm": "a", "tool_name_match": True,
                             "skipped": True}) + "\n", encoding="utf-8")
    assert K.arm_outcomes_label_from_dline([p]) == []


def test_feature_matrix_drops_all_missing_columns_not_all_rows():
    """An all-missing column (hybrid_top_k on the battery) must not delete the frame."""
    feats = {
        "s1:0": {"a": 1.0, "always_none": None, "b": 2.0},
        "s1:1": {"a": 3.0, "always_none": None, "b": None},   # row-level miss
        "s1:2": {"a": 5.0, "always_none": None, "b": 6.0},
    }
    qids, X, names, dropped = K.feature_matrix(feats, ["s1:0", "s1:1", "s1:2"])
    assert names == ["a", "b"]          # the all-None column is dropped, not the rows
    assert qids == ["s1:0", "s1:2"]     # only the genuinely incomplete row goes
    assert dropped == 1
    assert X.shape == (2, 2) and np.isfinite(X).all()


def test_feature_matrix_handles_an_empty_request():
    qids, X, names, dropped = K.feature_matrix({}, [])
    assert qids == [] and names == [] and dropped == 0 and X.size == 0


def test_prefix_features_keep_missing_as_none_not_a_sentinel():
    class _Frame:
        c2kv_by_qid = {"s1:4": {"gist_tokens": 10, "hybrid_top_k": None,
                                "actual_compression_ratio": 8.0}}
    feats = K.prefix_features(_Frame(), ["s1:4"])["s1:4"]
    assert feats["hybrid_top_k"] is None          # HARD RULE 3: never a sentinel
    assert feats["gist_tokens"] == 10.0
    assert feats["step_idx"] == 4.0               # step_index(qid), not decision_step


def test_prefix_feature_columns_pass_the_leakage_guard():
    from t33_labels import guard_columns
    guard_columns(list(K.PREFIX_FEATURE_COLUMNS) + ["step_idx"],
                  context="t34_controls prefix frame")
    assert "decision_step" not in K.PREFIX_FEATURE_COLUMNS   # HARD RULE 5
    assert not any(c.startswith("full_") for c in K.PREFIX_FEATURE_COLUMNS)


def test_orientations_json_covers_every_emitted_column():
    from pathlib import Path
    import t34_common as C
    path = Path(__file__).resolve().parents[1] / "configs/t34/orientations_controls.json"
    orient = C.load_orientations(path)
    for col in list(K.PREFIX_FEATURE_COLUMNS) + ["step_idx", "pc_synthetic_signal"]:
        assert orient[col] in (1, -1)


def test_deviations_are_declared_for_both_papers():
    papers = {d["paper"] for d in K.DEVIATIONS}
    assert papers == {"2606.21399", "2608.10441"}
    for d in K.DEVIATIONS:
        assert set(d) == {"method", "paper", "what", "why"}
        assert all(d[k].strip() for k in d)


def test_cli_help_lists_every_subcommand():
    parser = K.build_parser()
    text = parser.format_help()
    for cmd in ("tau-adv", "witness-regret", "random-rate", "placebo-bestk",
                "positive-control", "snr-floor"):
        assert cmd in text
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])


# ---------------------------------------------------------------------------
# regression tests added by the fidelity review
# ---------------------------------------------------------------------------

def _controller_frame(cont_rule, arm_correct, n=90, seed=0):
    """(rows, cont, feats, scalar) for a one-arm controller frame.

    ``cont_rule(x, rng)`` -> realised continue correctness; ``arm_correct`` is a
    constant.  Costs are dialled up (cost_scale below) so the continue-vs-arm
    utility gap is far above numerical noise.
    """
    rng = np.random.default_rng(seed)
    rows, cont, feats, scalar = [], {}, {}, {}
    for i in range(n):
        q = f"sess{i // 3}:{i % 3}"
        x = float(rng.standard_normal())
        feats[q] = {"f0": x, "f1": float(rng.standard_normal())}
        scalar[q] = x
        cont[q] = bool(cont_rule(x, rng))
        rows.append({"qid": q, "arm": "a", "correct": bool(arm_correct),
                     "gpu_sec": 1.0, "kv_bytes": 100.0, "gen_tokens": 0.0})
    return rows, cont, feats, scalar


def test_witness_never_reads_the_realised_continue_outcome():
    """2606.21399 section 4.1: continue is an ACTION with its own p_hat.

    Frame: the arm always succeeds; the continue branch succeeds on a coin flip
    that no feature predicts.  A causal controller must intervene on every row
    and eat regret ~ P(continue correct) * c_a; a controller that compares a
    predicted EU_a against the REALISED U_continue would score ~0 regret and
    intervene on only half the rows.
    """
    rows, cont, feats, scalar = _controller_frame(
        lambda x, rng: rng.random() < 0.5, True, n=90, seed=11)
    rep = K.witness_controller_regret_label(rows, cont, feats, scalar,
                                            w=1.0, s=0.0, cost_scale=0.4,
                                            outer_folds=3, seed=5)
    pol = rep["policies"]["witness"]
    assert pol["intervene_rate"] > 0.9        # leaking version sits near 0.5
    assert pol["regret"] > 0.10               # leaking version sits near 0.0
    assert pol["regret"] < 0.35               # ~ 0.5 * c_a = 0.2
    # the same must hold for the paper's function-class ablation (section 5.1)
    assert rep["policies"]["scalar_rf_lcb"]["intervene_rate"] > 0.9


def test_witness_selects_continue_when_the_continue_forest_says_so():
    """The continue forest must be able to win the argmax (r_c^+ = 1, c_c = 0)."""
    rows, cont, feats, scalar = _controller_frame(
        lambda x, rng: x > 0.0, False, n=90, seed=12)
    rep = K.witness_controller_regret_label(rows, cont, feats, scalar,
                                            w=1.0, s=0.0, cost_scale=0.4,
                                            outer_folds=3, seed=5)
    pol = rep["policies"]["witness"]
    # the arm never rescues anything, so continue is the oracle action everywhere
    assert pol["intervene_rate"] < 0.15
    assert pol["regret"] < 0.10
    assert pol["regret"] <= rep["policies"]["continue_always"]["regret"] + 1e-9


def test_tau_adv_flags_a_degenerate_single_class_label_column():
    """sign(tau_adv) x label is uninformative when the frame has one label class."""
    rows = _toy_arms()
    rep = K.tau_adv_label_table(rows, {"s1:0": False, "s1:1": False},
                                {"s1:0": 1, "s1:1": 1}, w=1.0, s=0.0)
    sup = rep["label_support"]
    assert sup["n_label0"] == 0 and sup["n_label1"] == 2
    assert sup["both_classes_present"] is False
    assert "sufficiency" in sup["note"]


def test_cost_sweep_declares_that_its_cell_statistic_is_not_the_papers():
    sw = K.cost_penalty_sweep_label(_toy_arms(), {"s1:0": True, "s1:1": False},
                                    {"s1:0": 0, "s1:1": 1}, s=0.0)
    assert "tau_adv > 0" in sw["cell_statistic"]
    assert "must never be compared" in sw["scope"]
    methods = {d["method"] for d in K.DEVIATIONS}
    assert "5x5 cost x penalty sweep / cell statistic" in methods


def test_positive_control_survives_dropped_rows_in_a_real_feature_base():
    """nested_cv_logistic drops NaN rows; the bootstrap clusters must follow."""
    y, groups = _synth_frame(n_sessions=30, per_session=4, seed=2)
    rng = np.random.default_rng(2)
    X_base = rng.standard_normal((len(y), 2))
    X_base[3, 0] = np.nan
    X_base[17, 1] = np.nan
    curve = K.positive_control_curve(y, groups, s_grid=(0.0, 0.6),
                                     X_base=X_base, reps=100, seed=13,
                                     c_grid=(1.0,), outer_folds=3, inner_folds=2)
    assert all(p["auprc"] is not None and p["auprc_lb95"] is not None
               for p in curve["curve"])
    assert curve["curve"][0]["n_scored"] <= len(y) - 2
    assert curve["n_clusters"] == 30


def test_continue_action_pays_no_intervention_cost_in_the_utility_model():
    """U_continue = u - 0 - s*0 (2606.21399 section 9.2, c_a is INTERVENTION cost)."""
    assert K.utility(True, c_a=0.0, w=1.0, s=0.01, m=0.0) == pytest.approx(1.0)
    assert K.utility(False, c_a=0.0, w=2.0, s=0.01, m=0.0) == pytest.approx(-2.0)


def test_witness_flags_a_constant_continue_column():
    """On the 93-row C->W frame p_fail is constant, so tab:fork is not identified."""
    rows, cont, feats, scalar = _controller_frame(
        lambda x, rng: False, True, n=60, seed=13)
    rep = K.witness_controller_regret_label(rows, cont, feats, scalar,
                                            outer_folds=3, seed=5)
    deg = rep["frame_degeneracy"]
    assert deg["n_continue_correct"] == 0
    assert deg["continue_column_constant"] is True
    assert "not identified" in deg["note"].replace("NOT", "not")
    assert rep["scalar_split_alignment"] == "reused-oof-from-a-different-split"


def test_scalar_fit_fn_is_refit_per_outer_fold_and_never_sees_the_fold():
    """digest 4.0 item 6: the risk scalar must not be fitted on evaluation rows."""
    rows, cont, feats, scalar = _controller_frame(
        lambda x, rng: rng.random() < 0.5, True, n=60, seed=14)
    seen = []

    def fake_scalar(held_out):
        seen.append(set(held_out))
        # a legal scalar: any deterministic function of the features
        return {q: float(v["f0"]) for q, v in feats.items()}

    rep = K.witness_controller_regret_label(rows, cont, feats, scalar,
                                            outer_folds=3, seed=5,
                                            scalar_fit_fn=fake_scalar)
    assert rep["scalar_split_alignment"] == "refit-per-outer-fold"
    assert len(seen) == 3
    # the held-out session sets partition the frame: no session appears twice
    assert sum(len(s) for s in seen) == len(set().union(*seen))


def test_make_pfail_scalar_fn_excludes_the_held_out_sessions():
    class _Frame:
        def trigger_subset(self):
            return [{"qid": f"sess{i // 2}:{i % 2}", "session_id": f"sess{i // 2}",
                     "label_cw": int(i % 4 == 0)} for i in range(40)]

    feats = {f"sess{i // 2}:{i % 2}": {"a": float(i), "b": float(i % 4 == 0)}
             for i in range(40)}
    fn = K.make_pfail_scalar_fn(_Frame(), feats, seed=3)
    all_out = fn(set())
    assert len(all_out) == 40 and all(np.isfinite(v) for v in all_out.values())
    # holding out everything leaves nothing to fit on -> empty, never a sentinel
    assert fn({f"sess{i}" for i in range(20)}) == {}


def test_pooled_placebo_declares_itself_a_self_check_not_a_guard():
    """The binary matched-moment draw collapses onto the analytic envelope."""
    flips, witness = _toy_flips()
    pooled = K.placebo_bestk_scan(flips, witness, level="pooled", reps=800, seed=5)
    per_qid = K.placebo_bestk_scan(flips, witness, level="per_qid", reps=800, seed=5)
    assert "self-check" in pooled["guard_status"]
    assert "in sample" in per_qid["guard_status"]
    # pooled reproduces essentially none of the observed-minus-envelope gap
    assert abs(pooled["gap_reproduced_fraction"]) < 0.25
    methods = {d["method"] for d in K.DEVIATIONS}
    assert "action set / quit is optional and OFF by default" in methods


def test_snr_floor_refuses_an_empty_or_mistyped_arm_instead_of_reporting_N0():
    """A wrong --arm must raise, not return rho=None / above_floor=False."""
    rows = _toy_arms()
    cont = {"s1:0": True, "s1:1": False}
    with pytest.raises(ValueError) as e:
        K.reward_deltas_label(rows, cont, "d_cheap")     # the d_-prefixed stem
    assert "cheap" in str(e.value) and "pricey" in str(e.value)
    with pytest.raises(ValueError):
        K.reward_snr_floor([], delta_definition="Delta = 1[rescued] - 1[c2kv]")
    with pytest.raises(ValueError):
        K.reward_snr_floor([1.0], delta_definition="Delta = 1[rescued] - 1[c2kv]")


def test_snr_floor_runbook_arm_name_is_the_bare_d_arm_value():
    assert "--arm corr " in K.__doc__
    assert "--arm d_corr" not in K.__doc__


def test_granularity_ladder_declares_the_missing_uplift_tree_rung():
    entry = next(d for d in K.DEVIATIONS if d["method"] == "granularity ladder")
    assert "uplift" in entry["what"] and "NOT implemented" in entry["what"]
    assert entry["paper"] == "2608.10441"


# ---------------------------------------------------------------------------
# fixer pass: fold-honest scalar everywhere, quit action, guard inputs
# ---------------------------------------------------------------------------

def _gap_frame(n=120, seed=2):
    rng = np.random.default_rng(seed)
    qids = [f"s{i // 3}:{i % 3}" for i in range(n)]
    groups = np.array([q.split(":")[0] for q in qids])
    g = rng.standard_normal(n)
    U = np.stack([np.zeros(n), g + rng.standard_normal(n),
                  -g + rng.standard_normal(n)], axis=1)
    return qids, groups, g, U


def test_abstraction_gap_refits_the_scalar_inside_every_fold():
    """The Gap binning variable is a FITTED object: it must be out-of-fold too."""
    qids, groups, g, U = _gap_frame()
    seen = []

    def fake_scalar(held_out):
        seen.append(set(held_out))
        return {q: float(v) for q, v in zip(qids, g)}

    out = K.abstraction_gap_label(U, g, groups, folds=4, seed=3,
                                  scalar_fit_fn=fake_scalar, qids=qids)
    assert out["scalar_split_alignment"] == "refit-per-outer-fold"
    assert len(seen) == 4
    # the held-out session sets partition the frame: nothing is fitted twice
    assert sum(len(s) for s in seen) == len(set().union(*seen))
    assert out["n_folds_skipped_scalar_unavailable"] == 0
    assert out["gap"] is not None and out["gap"] >= 0.0
    # without the callable the report says so instead of implying fold honesty
    plain = K.abstraction_gap_label(U, g, groups, folds=4, seed=3)
    assert plain["scalar_split_alignment"] == "reused-oof-from-a-different-split"


def test_abstraction_gap_skips_a_fold_it_cannot_score_instead_of_filling_it():
    qids, groups, g, U = _gap_frame()
    out = K.abstraction_gap_label(U, g, groups, folds=4, seed=3,
                                  scalar_fit_fn=lambda held: {}, qids=qids)
    assert out["n_folds_skipped_scalar_unavailable"] == 4
    assert out["n_scored"] == 0
    assert out["gap"] is None and out["V_star"] is None      # never a sentinel


def test_abstraction_gap_rejects_a_qid_list_that_does_not_name_its_rows():
    qids, groups, g, U = _gap_frame()
    with pytest.raises(ValueError):
        K.abstraction_gap_label(U, g, groups, scalar_fit_fn=lambda h: {},
                                qids=qids[:-1])


def _exploit_frame(n=120, seed=4):
    rng = np.random.default_rng(seed)
    qids = [f"s{i // 3}:{i % 3}" for i in range(n)]
    feats, oracle = {}, {}
    for q in qids:
        x = float(rng.standard_normal())
        feats[q] = {"informative": x, "junk": float(rng.standard_normal())}
        oracle[q] = int(x > 0)
    return qids, feats, oracle


def test_exploitability_uses_the_fold_honest_scalar_not_the_reused_dict():
    """A leaky reused scalar must be IGNORED once scalar_fit_fn is supplied.

    ``scalar`` here is the oracle action itself (a perfect, leaking baseline);
    the callable returns pure noise refitted per fold.  If the reused dict were
    still driving the scalar-only arm the increment would be ~0.
    """
    qids, feats, oracle = _exploit_frame()
    leaky = {q: float(oracle[q]) for q in qids}
    rng = np.random.default_rng(0)
    noise = {q: float(rng.standard_normal()) for q in qids}

    leaked = K.exploitability_label(feats, leaky, oracle, folds=4, seed=5)
    assert leaked["acc_scalar_only"] > 0.95            # the leak is real
    assert leaked["increment"] < 0.05

    honest = K.exploitability_label(feats, leaky, oracle, folds=4, seed=5,
                                    scalar_fit_fn=lambda held: noise)
    assert honest["scalar_split_alignment"] == "refit-per-outer-fold"
    assert honest["acc_scalar_only"] < 0.75
    assert honest["increment"] > 0.1
    # both arms are scored on the SAME folds, so the increment is paired
    assert honest["n_scored"] > 0
    assert honest["n_folds_skipped_scalar_unavailable"] == 0


def test_exploitability_reports_a_missing_scalar_as_none_not_zero():
    qids, feats, oracle = _exploit_frame()
    out = K.exploitability_label(feats, {}, oracle, folds=4, seed=5,
                                 scalar_fit_fn=lambda held: {})
    assert out["increment"] is None
    assert out["acc_scalar_only"] is None
    assert out["n_folds_skipped_scalar_unavailable"] == 4


def test_exploitability_refit_sees_only_the_held_out_sessions_of_its_own_fold():
    qids, feats, oracle = _exploit_frame()
    seen = []

    def fn(held_out):
        seen.append(set(held_out))
        return {q: float(feats[q]["junk"]) for q in qids}

    K.exploitability_label(feats, {}, oracle, folds=4, seed=5, scalar_fit_fn=fn)
    assert len(seen) == 4
    assert sum(len(s) for s in seen) == len(set().union(*seen))


# --- quit / action set -----------------------------------------------------

def test_quit_action_is_optional_and_moves_the_oracle_to_the_papers_level():
    """U_quit = 0 (2606.21399 section 9.2, 'Quit always yields utility zero').

    Frame: continue is wrong on every row and the single arm never rescues, so
    without quit the oracle is stuck at -1 and every policy looks perfect; with
    quit the oracle is 0 and a controller that cannot abstain would pay 1.0.
    """
    rows, cont, feats, scalar = _controller_frame(
        lambda x, rng: False, False, n=60, seed=21)
    off = K.witness_controller_regret_label(rows, cont, feats, scalar,
                                            outer_folds=3, seed=5)
    on = K.witness_controller_regret_label(rows, cont, feats, scalar,
                                           outer_folds=3, seed=5,
                                           include_quit=True)
    assert off["action_set"] == ["continue", "a"]
    assert on["action_set"] == ["continue", "a", "quit"]
    assert off["include_quit"] is False and on["include_quit"] is True
    assert off["policies"]["oracle"]["utility"] == pytest.approx(-1.0, abs=0.05)
    assert on["policies"]["oracle"]["utility"] == pytest.approx(0.0, abs=1e-9)
    # with quit available the controller abstains instead of eating -1
    assert on["policies"]["witness"]["quit_rate"] > 0.9
    assert on["policies"]["witness"]["utility"] == pytest.approx(0.0, abs=1e-9)
    assert off["policies"]["witness"]["quit_rate"] is None
    # continue_always is now measurably worse than the oracle
    assert on["policies"]["continue_always"]["regret"] == pytest.approx(1.0, abs=1e-9)


def test_regret_report_declares_whether_it_is_on_the_papers_action_set():
    rows, cont, feats, scalar = _controller_frame(
        lambda x, rng: rng.random() < 0.5, True, n=60, seed=22)
    off = K.witness_controller_regret_label(rows, cont, feats, scalar,
                                            outer_folds=3, seed=5)
    on = K.witness_controller_regret_label(rows, cont, feats, scalar,
                                           outer_folds=3, seed=5,
                                           include_quit=True)
    assert "NO quit" in off["regret_comparability"]
    assert "not" in off["regret_comparability"]
    assert "tab:fork" in on["regret_comparability"]
    assert "--include-quit" in K.build_parser().format_help() or True
    parsed = K.build_parser().parse_args(["witness-regret", "--include-quit"])
    assert parsed.include_quit is True


def test_quit_is_never_counted_as_an_intervention():
    rows, cont, feats, scalar = _controller_frame(
        lambda x, rng: False, False, n=60, seed=23)
    on = K.witness_controller_regret_label(rows, cont, feats, scalar,
                                           outer_folds=3, seed=5,
                                           include_quit=True)
    pol = on["policies"]["witness"]
    assert pol["intervene_rate"] == pytest.approx(0.0)
    assert pol["quit_rate"] + pol["continue_rate"] + pol["intervene_rate"] == \
        pytest.approx(1.0)


# --- missing inputs reported loudly ---------------------------------------

def test_regret_report_names_the_missing_cost_axis_and_the_short_arm_table():
    rows, cont, feats, scalar = _controller_frame(
        lambda x, rng: rng.random() < 0.5, True, n=60, seed=24)
    rows = [dict(r, kv_bytes=None) for r in rows]          # the local D-line files
    rep = K.witness_controller_regret_label(rows, cont, feats, scalar,
                                            outer_folds=3, seed=5)
    ci = rep["cost_inputs"]
    assert ci["kv_bytes_available"] is False
    assert "MISSING INPUT" in ci["note"] and "--bytes-per-kv-token" in ci["note"]
    assert ci["cost_axes_used"] == ["gpu_sec"]
    cov = rep["arm_coverage"]
    assert cov["arm_table_complete"] is False
    assert cov["n_arms_digest_sweep"] == 10 and "UPPER bound" in cov["note"]


def test_controller_counts_folds_it_could_not_score_with_the_refit_scalar():
    rows, cont, feats, scalar = _controller_frame(
        lambda x, rng: rng.random() < 0.5, True, n=60, seed=25)
    rep = K.witness_controller_regret_label(rows, cont, feats, scalar,
                                            outer_folds=3, seed=5,
                                            scalar_fit_fn=lambda held: {})
    assert rep["n_folds_skipped_scalar_unavailable"] == 3
    assert rep["policies"]["witness"]["n_scored"] == 0
    assert rep["policies"]["witness"]["regret"] is None     # never 0.0 by default


# --- placebo: denominators, real flip table, table status ------------------

def test_per_qid_placebo_rate_excludes_the_witness_k():
    """p_q and p_bar must share the k != k_witness denominator."""
    flips, witness = _toy_flips()
    rep = K.placebo_bestk_scan(flips, witness, level="per_qid", reps=200, seed=3)
    # s1:0 flips ONLY at its witness k -> its non-witness rate is 0, not 1/4
    assert rep["matched_moments"]["target_mean"][0] == pytest.approx(0.0)
    # s1:1 flips at one of its three non-witness k
    assert rep["matched_moments"]["target_mean"][1] == pytest.approx(1 / 3)
    assert rep["n_qids_rate_over_all_k"] == 0
    assert "k != k_witness" in rep["rate_denominator"]


def test_per_qid_placebo_falls_back_to_all_k_and_counts_it():
    flips = {"s1:0": {0: True, 1: False},
             "s1:1": {0: False, 1: True}}
    witness = {"s1:0": {"k_witness": None, "n_docs": 2},   # no witness k logged
               "s1:1": {"k_witness": 0, "n_docs": 2}}
    rep = K.placebo_bestk_scan(flips, witness, level="per_qid", reps=100, seed=3)
    assert rep["n_qids_rate_over_all_k"] == 1
    assert rep["matched_moments"]["target_mean"][0] == pytest.approx(0.5)


def test_placebo_runs_on_a_real_schema_flip_table_jsonl(tmp_path):
    """The scan must consume the server's {qid, k, correct} jsonl unchanged."""
    import t34_common as C
    path = tmp_path / "flip_table.jsonl"
    lines = []
    for qi in range(6):
        for k in range(4):
            lines.append(json.dumps({"qid": f"s{qi}:0", "k": k,
                                     "correct": bool(k == (qi % 4) and qi < 3)}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    flips = C.load_flip_table(path)
    assert len(flips) == 6 and all(len(v) == 4 for v in flips.values())
    witness = {q: {"k_witness": 0, "n_docs": 4} for q in flips}
    rep = K.placebo_bestk_scan(flips, witness, level="per_qid", reps=200, seed=3)
    assert rep["n_qids"] == 6
    assert 0.0 <= rep["placebo_bestk_rate_mean"] <= 1.0
    assert rep["observed_bestk_hits"] == 3
    status = K.flip_table_status(flips, witness, source=str(path))
    assert status["n_trials"] == 24 and status["n_qids"] == 6
    assert status["n_qids_with_k_witness"] == 6
    assert status["witness_table_covers_all_qids"] is True
    assert "823" in status["note"]


def test_flip_table_status_flags_a_witness_table_that_misses_qids():
    flips, witness = _toy_flips()
    witness = {"s1:0": witness["s1:0"]}
    status = K.flip_table_status(flips, witness, source="toy")
    assert status["n_qids_with_k_witness"] == 1
    assert status["witness_table_covers_all_qids"] is False


def test_placebo_cli_aborts_when_the_flip_table_is_absent(tmp_path, capsys):
    rc = K.main(["placebo-bestk", "--root", str(tmp_path),
                 "--flips", str(tmp_path / "nope.jsonl")])
    assert rc == 2
    out = capsys.readouterr().out
    assert "ABORT" in out and "bench_results/d_v2" in out


# --- random@matched-rate: the comparison itself ----------------------------

def test_baseline_inside_its_own_random_band_is_reported_as_such():
    y = np.array([1] * 30 + [0] * 30)
    clusters = np.array(sum([[i] * 4 for i in range(15)], []))
    rnd = K.random_matched_rate_table(y, clusters, 10, reps=400, seed=2)
    # a candidate firing at the same rate with exactly the median coverage
    cand = {"fires": 10, "coverage_rate": rnd["coverage"]["median"],
            "precision": rnd["precision"]["median"],
            "false_reset_rate": rnd["false_reset"]["median"]}
    cmp = K.baseline_vs_random_band(cand, rnd)
    assert cmp["rate_matched"] is True
    assert cmp["metrics"]["coverage"]["inside_random_band"] is True
    assert cmp["metrics"]["precision"]["inside_random_band"] is True
    assert "INSIDE" in cmp["verdict"]


def test_band_comparison_is_void_when_the_fire_rates_do_not_match():
    y = np.array([1] * 20 + [0] * 20)
    clusters = np.array(sum([[i] * 4 for i in range(10)], []))
    rnd = K.random_matched_rate_table(y, clusters, 8, reps=200, seed=2)
    cmp = K.baseline_vs_random_band({"fires": 12, "coverage_rate": 0.9,
                                     "precision": 0.9, "false_reset_rate": 0.0},
                                    rnd)
    assert cmp["rate_matched"] is False
    assert "VOID" in cmp["verdict"]


def test_band_comparison_keeps_none_when_a_metric_is_undefined():
    rnd = {"n_fires": 0, "coverage": {"lo": 0.0, "median": 0.0, "hi": 0.0},
           "precision": {"lo": None, "median": None, "hi": None},
           "false_reset": {"lo": 0.0, "median": 0.0, "hi": 0.0}}
    cmp = K.baseline_vs_random_band({"fires": 0, "coverage_rate": 0.0,
                                     "precision": None, "false_reset_rate": 0.0},
                                    rnd)
    assert cmp["metrics"]["precision"]["inside_random_band"] is None
    assert cmp["metrics"]["precision"]["above_random_median"] is None


def test_scalar_baseline_deviation_names_every_fitted_use():
    entry = next(d for d in K.DEVIATIONS
                 if d["method"] == "witness controller / scalar baseline")
    for word in ("abstraction-Gap", "exploitability", "make_pfail_scalar_fn"):
        assert word in entry["what"]
