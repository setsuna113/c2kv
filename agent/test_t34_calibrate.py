# -*- coding: utf-8 -*-
"""Tests for agent/t34_calibrate.py (digest 4.11 thresholds & guarantees).

Run:  PYTHONIOENCODING=utf-8 python -m pytest agent/test_t34_calibrate.py -q
from C:/Users/yl998/Documents/programming/c2kv/tmp/t34-migration
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

import t34_calibrate as C  # noqa: E402
from t34_common import FrozenAssets, clopper_pearson  # noqa: E402

ROOT = _HERE.parent
HAVE_ASSETS = (ROOT / "results/bdf_pilot/d_r2/battery_full.jsonl").exists()


# ---------------------------------------------------------------------------
# (A) p-values, slack, fixed sequence, direction reversal
# ---------------------------------------------------------------------------

def test_hoeffding_pvalue_hand_value():
    # n=100, delta=0.3, Ehat=0.2 -> exp(-2*100*0.1^2) = exp(-2)
    assert C.hoeffding_pvalue(0.2, 0.3, 100) == pytest.approx(math.exp(-2.0), rel=1e-12)
    # no slack -> p = 1 (cannot reject)
    assert C.hoeffding_pvalue(0.4, 0.3, 100) == pytest.approx(1.0)
    assert C.hoeffding_pvalue(0.2, 0.3, 0) == pytest.approx(1.0)


def test_h1_bernoulli_kl_hand_value():
    # h1(0.2, 0.3) = 0.2 ln(2/3) + 0.8 ln(0.8/0.7)
    want = 0.2 * math.log(2.0 / 3.0) + 0.8 * math.log(0.8 / 0.7)
    assert C.h1_bernoulli_kl(0.2, 0.3) == pytest.approx(want, rel=1e-12)
    # edge a = 0 -> -log(1-b)
    assert C.h1_bernoulli_kl(0.0, 0.3) == pytest.approx(-math.log(0.7), rel=1e-12)
    assert C.h1_bernoulli_kl(1.0, 0.3) == pytest.approx(math.log(1.0 / 0.3), rel=1e-12)


def test_hoeffding_bentkus_matches_the_two_closed_forms_and_is_tighter():
    from scipy.stats import binom
    n, delta, rhat = 100, 0.3, 0.2
    first = math.exp(-n * C.h1_bernoulli_kl(min(rhat, delta), delta))
    second = math.e * float(binom.cdf(math.ceil(n * rhat), n, delta))
    p_hb = C.hoeffding_bentkus_pvalue(rhat, delta, n)
    assert p_hb == pytest.approx(min(1.0, first, second), rel=1e-12)
    # HB is the tighter (smaller) p-value here, which is why CALM uses it
    assert p_hb < C.hoeffding_pvalue(rhat, delta, n)
    # no slack -> cannot reject with either bound
    assert C.hoeffding_bentkus_pvalue(0.4, 0.3, n) == pytest.approx(1.0)
    # degenerate tolerance is not a licence to certify
    assert C.hoeffding_bentkus_pvalue(0.0, 0.0, n) == pytest.approx(1.0)


def test_hoeffding_slack_rows_vs_sessions():
    """Card 2207.07061: eps=0.05 needs slack sqrt(ln20/(2n)); ~4.1pp at n=900
    rows and ~14.4pp at n=72 session clusters.  Computed, not transcribed."""
    assert C.hoeffding_slack(900, 0.05) == pytest.approx(math.sqrt(math.log(20.0) / 1800.0))
    assert 100 * C.hoeffding_slack(900, 0.05) == pytest.approx(4.1, abs=0.05)
    assert 100 * C.hoeffding_slack(72, 0.05) == pytest.approx(14.4, abs=0.05)
    # session clustering costs an order of magnitude in n and ~3.5x in slack
    assert C.hoeffding_slack(72, 0.05) > 3 * C.hoeffding_slack(900, 0.05)


def test_aggregate_to_units_means_per_session():
    keys, vals = C.aggregate_to_units(["s2", "s1", "s1", "s2"], [1.0, 0.0, 1.0, 0.0])
    assert keys == ["s1", "s2"]
    assert vals.tolist() == [0.5, 0.5]


def test_fixed_sequence_stop_rule_and_prefix():
    # risks increase along the grid; only the first three can be rejected
    grid = [1.0, 0.9, 0.8, 0.7, 0.6]
    risks = [0.00, 0.02, 0.04, 0.40, 0.00]   # note the LAST one would reject if tested
    res = C.fixed_sequence_test(grid, risks, n_units=400, delta=0.20, eps=0.05,
                                pvalue="hoeffding")
    assert res.certified is True
    assert res.lambda_valid == [1.0, 0.9, 0.8]
    assert res.lambda_chosen == 0.8            # last rejected = cheapest certifiable
    assert res.stopped_at == 3
    # the walk STOPS: the trailing point is recorded but never tested
    assert res.rows[-1]["tested"] is False
    assert res.rows[-1]["rejected"] is False


def test_fixed_sequence_certification_failure_is_a_legal_outcome():
    res = C.fixed_sequence_test([1.0, 0.9], [0.5, 0.5], n_units=100, delta=0.10,
                                eps=0.05, pvalue="hb")
    assert res.certified is False
    assert res.lambda_chosen is None           # NO always-valid fallback leg
    assert res.lambda_valid == []
    assert res.stopped_at == 0


def test_fixed_sequence_multi_start_level_and_validation():
    res = C.fixed_sequence_test([1.0], [0.0], n_units=400, delta=0.2, eps=0.05, n_starts=4)
    assert res.eps == 0.05 and res.n_starts == 4
    with pytest.raises(ValueError):
        C.fixed_sequence_test([1.0, 0.9], [0.0], n_units=10, delta=0.1)
    with pytest.raises(ValueError):
        C.fixed_sequence_test([1.0], [0.0], n_units=10, delta=0.1, pvalue="clt")


def test_fst_slack_is_reported_at_the_walking_level_not_the_nominal_eps():
    """The walk rejects at eps / n_starts, so the slack it needs is the slack at
    THAT level (2110.01052 2.3.1); reporting sqrt(ln(1/eps)/(2n)) would understate
    the requirement whenever n_starts > 1."""
    grid = [1.0, 0.5]
    risks = [0.9, 0.9]
    one = C.fixed_sequence_test(grid, risks, n_units=50, delta=0.5, eps=0.05, n_starts=1)
    four = C.fixed_sequence_test(grid, risks, n_units=50, delta=0.5, eps=0.05, n_starts=4)
    assert one.slack_required == pytest.approx(C.hoeffding_slack(50, 0.05))
    assert four.slack_required == pytest.approx(C.hoeffding_slack(50, 0.05 / 4))
    assert four.slack_required > one.slack_required


def test_bonferroni_variant():
    assert C.bonferroni_reject([0.01, 0.02, 0.30], 0.05) == [True, False, False]
    assert C.bonferroni_reject([], 0.05) == []


def _toy_stream(n_sessions=40, per_session=5, seed=0):
    rng = np.random.default_rng(seed)
    sess, score, nofire = [], [], []
    for s in range(n_sessions):
        for k in range(per_session):
            sess.append(f"s{s:03d}")
            risky = (k == 0)
            score.append(1.0 if risky else 0.0)
            nofire.append(1.0 if risky else float(rng.random() < 0.05))
    return sess, np.array(score), np.array(nofire)


def test_certify_fire_rate_direction_is_reversed_and_returns_cheapest():
    sess, score, nofire = _toy_stream()
    fire = np.full(score.size, 0.0)            # perfect repair, for the direction test
    res = C.certify_fire_rate(score, nofire, fire, sess, delta=0.20, eps=0.05,
                              pvalue="hoeffding")
    # the grid's SAFE end is q = 1.0 (always fire), not CALM's lambda = 1
    assert res.rows[0]["lambda"] == pytest.approx(1.0)
    assert res.rows[0]["n_fire"] == score.size
    assert res.rows[0]["emp_risk"] == pytest.approx(0.0)
    assert res.rows[-1]["lambda"] == pytest.approx(0.0)
    assert res.extras["safe_end"].startswith("always_fire")
    assert res.unit == "session" and res.n_units == 40
    # risk is non-decreasing as the fire rate falls -> rejection set is a prefix
    tested = [r["emp_risk"] for r in res.rows if r["tested"]]
    assert all(b >= a - 1e-12 for a, b in zip(tested, tested[1:]))
    assert res.certified is True
    # chosen = the SMALLEST (cheapest) fire rate in the rejection prefix
    assert res.lambda_chosen == pytest.approx(min(res.lambda_valid))
    assert res.lambda_chosen < 1.0
    assert res.extras["certification_buys_nothing"] is False
    # a certificate that only clears at the safe end is flagged: real, but worth zero
    tight = C.certify_fire_rate(score, nofire, fire, sess, delta=0.02, eps=0.05,
                                pvalue="hoeffding")
    if tight.certified:
        assert tight.extras["certification_buys_nothing"] is (tight.lambda_chosen == 1.0)


def test_certify_fire_rate_always_fire_risk_equals_mean_loss_fire():
    """CALM's lambda=1 leg is exact (risk 0); ours is not.  With an inexact
    repair arm the safe end already carries risk, which is the whole reason the
    feasible region is bounded away from delta = 0."""
    sess, score, nofire = _toy_stream()
    fire = np.full(score.size, 0.30)           # 70 % repair success, inexact
    res = C.certify_fire_rate(score, nofire, fire, sess, delta=0.10, eps=0.05)
    assert res.rows[0]["emp_risk"] == pytest.approx(0.30)
    assert res.certified is False               # 0.30 > delta = 0.10 everywhere
    assert res.lambda_chosen is None


def test_certify_fire_rate_flags_its_quantile_grid_and_binary_score():
    """The grid point is an empirical quantile, not a fixed score threshold: the
    departure from 2207.07061's fixed Lambda must be visible in the artifact."""
    rng = np.random.default_rng(5)
    n = 40
    sess = [f"s{i // 4}" for i in range(n)]
    binary = (rng.random(n) < 0.3).astype(float)
    res = C.certify_fire_rate(binary, np.ones(n), np.zeros(n), sess,
                              delta=0.5, step=0.25)
    assert res.extras["grid_is_empirical_quantile"] is True
    assert "APPROXIMATE" in res.extras["grid_note"]
    assert res.extras["score_is_binary"] is True and res.extras["n_distinct_scores"] == 2
    assert res.extras["tie_note"] is not None
    cont = C.certify_fire_rate(rng.random(n), np.ones(n), np.zeros(n), sess,
                               delta=0.5, step=0.25)
    assert cont.extras["score_is_binary"] is False and cont.extras["tie_note"] is None
    assert any(d["what"].startswith("Our grid indexes a FIRE RATE") for d in C.DEVIATIONS)


def test_certify_fire_rate_row_unit_is_flagged_and_bounds_are_checked():
    sess, score, nofire = _toy_stream()
    fire = np.zeros(score.size)
    res = C.certify_fire_rate(score, nofire, fire, sess, delta=0.2, unit="row")
    assert res.extras["iid_assumption_violated"] is True
    assert res.n_units == score.size
    assert res.extras["slack_rows_pp"] < res.extras["slack_sessions_pp"]
    with pytest.raises(ValueError):
        C.certify_fire_rate(score, nofire + 2.0, fire, sess, delta=0.2)


# ---------------------------------------------------------------------------
# (B) UCBs, e-CRC, the abstention floor
# ---------------------------------------------------------------------------

def test_ucb_closed_forms():
    assert C.ucb_hoeffding(0.1, 100, 0.1) == pytest.approx(
        0.1 + math.sqrt(math.log(20.0) / 200.0))
    lg = math.log(20.0)
    assert C.ucb_empirical_bernstein(0.1, 0.09, 100, 0.1) == pytest.approx(
        0.1 + math.sqrt(2 * 0.09 * lg / 100.0) + 7 * lg / (3 * 99.0))
    assert C.ucb_empirical_bernstein(0.1, 0.09, 1, 0.1) == float("inf")
    assert C.ucb_hoeffding(0.1, 0, 0.1) == float("inf")


def _nesting_stream(n=1200, seed=11):
    rng = np.random.default_rng(seed)
    scores = rng.random(n)                       # risk-oriented (+1)
    risks = (rng.random(n) < 0.01 + 0.09 * scores).astype(float)
    return scores, risks


def test_ucb_nesting_holds_strictly_on_low_variance_synthetic_data():
    """2606.29054 Prop. 'Bound Ordering': Lambda*_Hoeff subset Lambda*_Bern
    subset Lambda*_e-CRC.  Built in the low-variance, large-|E_lambda| regime the
    proposition is about, with all three sets STRICTLY different so the
    inclusion is not vacuous."""
    scores, risks = _nesting_stream()
    rep = C.bound_ordering_report(scores, risks, alpha=0.08, delta=0.1,
                                  orientation=1, min_emit=150)
    assert rep["nesting_holds"] is True
    assert (rep["sizes"]["hoeffding"] < rep["sizes"]["bernstein"]
            < rep["sizes"]["ecrc"])


def test_bound_ordering_is_reported_not_asserted():
    """The module must be able to SAY the nesting failed.  On tiny emit sets the
    empirical-Bernstein constant 7log(2/delta)/(3(n-1)) dominates and Hoeffding
    certifies lambdas Bernstein cannot, so the inclusion breaks -- which is why
    bound_ordering_report returns a flag instead of asserting the proposition."""
    scores, risks = _nesting_stream(n=600, seed=11)
    rep = C.bound_ordering_report(scores, risks, alpha=0.20, delta=0.1,
                                  orientation=1, min_emit=10)
    assert rep["hoeffding_subset_bernstein"] is False
    assert rep["nesting_holds"] is False
    assert rep["sizes"]["bernstein"] < rep["sizes"]["hoeffding"]


def test_crc_select_lambda_orientation_and_degeneracy():
    scores, risks = _nesting_stream(n=600, seed=11)
    a = C.crc_select_lambda(scores, risks, alpha=0.20, bound="bernstein", orientation=1)
    b = C.crc_select_lambda(-scores, risks, alpha=0.20, bound="bernstein", orientation=-1)
    assert a["certified"] and b["certified"]
    assert a["lambda_star_reliability"] == pytest.approx(b["lambda_star_reliability"])
    assert a["threshold_in_score_units"] == pytest.approx(-a["lambda_star_reliability"])
    # every non-degenerate grid point respects |E_lambda| >= min_emit
    assert all(r["n_emit"] >= 10 for r in a["grid"] if not r["degenerate"])
    assert a["n_emit"] >= 10
    # the emitted set really does satisfy the risk target empirically
    assert a["risk_on_emitted"] <= 0.20 + 1e-9
    # a target below the base risk cannot be certified at any threshold
    hard = C.crc_select_lambda(scores, risks, alpha=0.001, bound="bernstein")
    assert hard["certified"] is False and hard["threshold_in_score_units"] is None
    assert hard["abstention_rate"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        C.crc_select_lambda(scores, risks, alpha=0.1, bound="clt")


def test_ecrc_false_certification_rate_bounded_by_delta_on_a_null_stream():
    """Ville: under H_0 (E[R] >= alpha) the wealth certifies with prob <= delta."""
    delta, alpha, n_sims, n = 0.1, 0.20, 300, 200
    rng = np.random.default_rng(20260905)
    false_certs = 0
    for _ in range(n_sims):
        r = (rng.random(n) < alpha).astype(float)     # E[R] = alpha, the H_0 boundary
        ok, _w = C.ecrc_certified(r, alpha, delta)
        false_certs += int(ok)
    assert false_certs / n_sims <= delta


def test_ecrc_kappa_is_predictable_and_wealth_grows_under_the_alternative():
    r = np.zeros(200)                              # risk far below alpha
    for kelly in ("approx", "grid"):
        ok, w = C.ecrc_certified(r, alpha=0.20, delta=0.1, kelly=kelly)
        assert ok and w >= 10.0
        # kappa_1 sees no past, so the first bet is the clip floor and W_1 == 1
        assert C.ecrc_wealth(r, 0.20, kelly=kelly)[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        C.ecrc_wealth(r, 0.20, kelly="whole-stream")


def test_ecrc_default_kelly_is_the_papers_printed_closed_form():
    """2606.29054 Appendix Alg. 'e-CRC': mu_hat_j = mean(r_1..r_{j-1}) (j >= 2),
    kappa_j = clip((alpha - mu_hat_j)/(alpha(1-alpha)), 0, 0.5), W_0 = 1."""
    import inspect
    assert inspect.signature(C.ecrc_wealth).parameters["kelly"].default == "paper"
    assert inspect.signature(C.ecrc_certified).parameters["kelly"].default == "paper"
    assert inspect.signature(C.crc_select_lambda).parameters["kelly"].default == "paper"

    alpha = 0.20
    r = np.array([0.0, 1.0, 0.0])
    w = C.ecrc_wealth(r, alpha)                      # default = 'paper'
    # j = 1: no past -> kappa_1 = 0 -> W_1 = W_0 = 1
    assert w[0] == pytest.approx(1.0)
    # j = 2: mu_hat = 0 -> kappa = clip(0.20 / (0.20*0.80)) = clip(1.25) = 0.5
    assert w[1] == pytest.approx(1.0 * (1.0 + 0.5 * (alpha - 1.0)))
    # j = 3: mu_hat = 0.5 -> (0.20 - 0.50)/0.16 < 0 -> clipped to 0 -> wealth frozen
    assert w[2] == pytest.approx(w[1])
    # the bets are genuinely different rules, so the default is not a cosmetic
    # label: at alpha = 0.5 with r = 0.4 the paper's kappa is (0.5-0.4)/0.25 = 0.4
    # while the second-order approximation clips to 0.5.
    r2 = np.array([0.4, 0.4])
    assert C.ecrc_wealth(r2, 0.5)[1] == pytest.approx(1.0 + 0.4 * 0.1)
    assert C.ecrc_wealth(r2, 0.5, kelly="approx")[1] == pytest.approx(1.0 + 0.5 * 0.1)
    with pytest.raises(ValueError):
        C.ecrc_wealth(r, alpha, kelly="whole-stream")


def test_ecrc_paper_kelly_respects_the_null_bound():
    delta, alpha, n_sims, n = 0.1, 0.20, 300, 200
    rng = np.random.default_rng(777)
    false_certs = sum(
        int(C.ecrc_certified((rng.random(n) < alpha).astype(float), alpha, delta)[0])
        for _ in range(n_sims))
    assert false_certs / n_sims <= delta


def test_crc_select_lambda_does_not_report_wealth_as_a_ucb():
    rng = np.random.default_rng(11)
    scores = rng.random(300)
    risks = (rng.random(300) < 0.02).astype(float)
    for bound in ("hoeffding", "bernstein"):
        rows = C.crc_select_lambda(scores, risks, alpha=0.2, bound=bound)["grid"]
        live = [r for r in rows if not r["degenerate"]]
        assert live and all(r["ucb"] is not None and r["wealth"] is None for r in live)
    rows = C.crc_select_lambda(scores, risks, alpha=0.2, bound="ecrc")["grid"]
    live = [r for r in rows if not r["degenerate"]]
    assert live and all(r["ucb"] is None and r["wealth"] is not None for r in live)


def test_ecrc_exact_grid_kelly_also_respects_the_null_bound():
    delta, alpha, n_sims, n = 0.1, 0.20, 60, 120
    rng = np.random.default_rng(4242)
    false_certs = 0
    for _ in range(n_sims):
        r = (rng.random(n) < alpha).astype(float)
        ok, _w = C.ecrc_certified(r, alpha, delta, kelly="grid")
        false_certs += int(ok)
    assert false_certs / n_sims <= delta


def test_abstention_floor_arithmetic_matches_the_card():
    """Card 2606.29054, Reading A (mu = 93/900) and Reading B (mu = 0.7911)."""
    mu_a = 93 / 900
    got = {a: C.abstention_lower_bound(mu_a, a) for a in (0.10, 0.05, 0.02, 0.01)}
    assert 100 * got[0.10] == pytest.approx(0.37, abs=0.01)
    assert 100 * got[0.05] == pytest.approx(5.61, abs=0.01)
    assert 100 * got[0.02] == pytest.approx(8.50, abs=0.01)
    assert 100 * got[0.01] == pytest.approx(9.42, abs=0.01)
    assert 100 * C.abstention_lower_bound(0.7911, 0.10) == pytest.approx(76.8, abs=0.05)
    assert C.abstention_lower_bound(0.05, 0.10) == 0.0        # mu <= alpha -> no floor
    # the finite-sample term dwarfs the floor at loose alpha (card's whole point)
    fs = C.finite_sample_term(900, 0.1)
    assert fs == pytest.approx(math.sqrt(math.log(10.0) / 900.0))
    assert fs == pytest.approx(0.0506, abs=0.0002)
    assert fs > got[0.10] * 10


def test_feasibility_table_flags_where_the_bound_bites():
    tab = C.feasibility_table(93 / 900, [0.10, 0.05, 0.02, 0.01], 900, 0.1)
    flags = {r["alpha"]: r["floor_exceeds_correction"] for r in tab}
    # At alpha = 0.10 the floor (0.37 %) is an order of magnitude below the
    # finite-sample term (5.06 %): the bound says nothing at our n.
    assert flags[0.10] is False
    # alpha = 0.05 sits ON the boundary (floor 5.61 % vs bare rate 5.06 %); the
    # O() constant is unstated in 2606.29054 2.2 (card open question 2), so the
    # 0.5 pp margin is NOT a separation -- the card's prose ("only alpha <= 0.02
    # produces a floor that survives the correction") is the conservative read.
    assert flags[0.05] is True
    assert tab[1]["min_fire_rate"] - tab[1]["finite_sample_term"] < 0.01
    assert flags[0.02] is True and flags[0.01] is True      # here it really bites
    assert tab[3]["min_fire_rate"] > 1.8 * tab[3]["finite_sample_term"]


def test_feasibility_table_refuses_to_call_the_bare_rate_a_separation():
    """2606.29054 2.2 writes the correction as O(sqrt(log(1/delta)/n)); the constant is
    in an unread appendix, so no alpha may be reported as a separation."""
    tab = C.feasibility_table(93 / 900, [0.10, 0.05, 0.02], 900, 0.1)
    for row in tab:
        assert row["separation_claimable"] is None
        assert row["finite_sample_term_is_bare_rate"] is True
        assert "unread" in row["separation_note"]
    assert any("finite-sample term" in d["method"] for d in C.DEVIATIONS)


def test_recommend_bound_rule():
    assert C.recommend_bound(500, 0.0926)["recommended"] == "bernstein"
    assert C.recommend_bound(500, 0.0926)["low_variance_regime"] is True
    assert C.recommend_bound(150, 0.0926)["recommended"] == "ecrc"
    assert C.recommend_bound(500, 0.30)["low_variance_regime"] is False


# ---------------------------------------------------------------------------
# (C) Clopper-Pearson recall gate
# ---------------------------------------------------------------------------

def test_cp_lower_one_sided_agrees_with_the_two_sided_helper_at_2alpha():
    for k, n in ((30, 50), (0, 50), (50, 50), (7, 12)):
        assert C.cp_lower_one_sided(k, n, 0.05) == pytest.approx(
            clopper_pearson(k, n, 0.10)[0], abs=1e-12)
    assert C.cp_lower_one_sided(0, 50) == 0.0
    assert C.cp_lower_one_sided(50, 50, 0.05) == pytest.approx(0.05 ** (1 / 50))
    assert math.isnan(C.cp_lower_one_sided(3, 0))


def test_cp_recall_threshold_monotone_in_budget():
    s = np.arange(50) / 50.0
    taus = [C.cp_recall_threshold(s, t)["tau"] for t in (0.5, 0.7, 0.9)]
    assert taus[0] <= taus[1] <= taus[2]
    assert all(np.isfinite(t) for t in taus)
    # the certified lower bound really does clear the budget
    for t in (0.5, 0.7, 0.9):
        g = C.cp_recall_threshold(s, t)
        assert g["gate_open"] and g["cp_lower"] >= t


def test_cp_recall_threshold_closure_and_abstention():
    s = np.arange(50) / 50.0
    g1 = C.cp_recall_threshold(s, 1.0)
    assert g1["gate_open"] is False and g1["tau"] == float("inf")
    assert "t_r = 1" in g1["reason"]
    # 0.95 is above the n=50 ceiling 0.05^(1/50) = 0.9426 -> abstain, not crash
    g2 = C.cp_recall_threshold(s, 0.95)
    assert g2["gate_open"] is False and "abstain" in g2["reason"]
    g3 = C.cp_recall_threshold([], 0.5)
    assert g3["gate_open"] is False and g3["n_r"] == 0


def test_sample_complexity_and_ceiling_match_the_cards_arithmetic():
    assert C.sample_complexity_n_pos(0.95, 0.05) == 59        # ln.05/ln.95 = 58.4
    assert C.sample_complexity_n_pos(0.98, 0.05) == 149
    assert C.sample_complexity_n_pos(0.99, 0.05) == 299
    assert C.certifiable_recall_ceiling(68, 0.05) == pytest.approx(0.9569, abs=1e-4)
    assert C.certifiable_recall_ceiling(188, 0.05) == pytest.approx(0.9842, abs=1e-4)
    # the session-cluster unit (100 clusters on both the C->C and harm sets)
    assert C.certifiable_recall_ceiling(100, 0.05) == pytest.approx(0.05 ** 0.01)
    assert C.sample_complexity_n_pos(1.5, 0.05) == -1
    assert math.isnan(C.certifiable_recall_ceiling(0))


def test_assert_no_cascade_claim_on_the_battery():
    C.assert_no_cascade_claim(1, "battery")            # single gate is fine
    C.assert_no_cascade_claim(6, "bench")              # rounds exist there
    with pytest.raises(ValueError):
        C.assert_no_cascade_claim(6, "battery")
    with pytest.raises(ValueError):
        C.cascade_thresholds([[0.1], [0.2]], [0.9, 0.9], face="battery")


def test_budget_search_abstains_when_no_margin_feasible_vector_exists():
    cal = [np.arange(40) / 40.0]
    calls = {"n": 0}

    def val_apply(taus):
        calls["n"] += 1
        return {"recall": 0.80, "savings": 0.5}      # never clears 0.95 + 0.02

    out = C.budget_feasibility_search([[0.85], [0.9]], cal, val_apply,
                                      rho_star=0.95, margin=0.02)
    assert out["abstain"] is True and out["deployed"] is None
    assert calls["n"] == 2

    def val_ok(taus):
        return {"recall": 0.99, "savings": 0.3}

    out2 = C.budget_feasibility_search([[0.85], [0.9]], cal, val_ok,
                                       rho_star=0.95, margin=0.02)
    assert out2["abstain"] is False and out2["n_feasible"] == 2


def test_frozen_certificate_counts_survivors_against_every_gate():
    s = np.array([0.1, 0.2, 0.9, 0.95])
    cert = C.frozen_certificate(s, [0.5], alpha_m=0.05)
    assert cert["n_pos"] == 4 and cert["survivors"] == 2
    assert cert["recall_hat"] == pytest.approx(0.5)
    assert cert["cp_lower"] == pytest.approx(C.cp_lower_one_sided(2, 4, 0.05))
    assert C.frozen_certificate([], [0.5])["n_pos"] == 0


# ---------------------------------------------------------------------------
# (D) CRC primitive, weighted CRC, face pair
# ---------------------------------------------------------------------------

def _loss_matrix(n=50, m=11, seed=5):
    rng = np.random.default_rng(seed)
    lam = np.linspace(0.0, 1.0, m)
    base = rng.random(n)
    L = np.clip(base[:, None] - lam[None, :], 0.0, 1.0)   # non-increasing in lambda
    return lam, L


def test_crc_primitive_matches_its_definition():
    lam, L = _loss_matrix()
    out = C.crc_lambda_hat(lam, L, alpha=0.20)
    n = L.shape[0]
    rhat = L.mean(axis=0)
    lhs = n / (n + 1) * rhat + 1.0 / (n + 1)
    want = lam[int(np.where(lhs <= 0.20)[0][0])]
    assert out["lambda_hat"] == pytest.approx(want)
    assert out["loss_monotone_nonincreasing"] is True
    assert out["set_empty"] is False
    # empty feasible set -> lambda_max, exactly as the paper defines it
    tight = C.crc_lambda_hat(lam, L, alpha=-1.0)
    assert tight["set_empty"] is True and tight["lambda_hat"] == pytest.approx(lam[-1])
    with pytest.raises(ValueError):
        C.crc_lambda_hat(lam[::-1], L, alpha=0.2)


def test_weighted_crc_reduces_to_plain_crc_with_equal_weights():
    lam, L = _loss_matrix()
    plain = C.crc_lambda_hat(lam, L, alpha=0.20)
    w1 = C.weighted_crc_lambda_hat(lam, L, np.ones(L.shape[0]), alpha=0.20)
    assert w1["lambda_hat"] == pytest.approx(plain["lambda_hat"])
    assert w1["N_w"] == pytest.approx(float(L.shape[0]))
    assert w1["rhat"] == pytest.approx(plain["rhat"])
    assert w1["weighted"] is True and plain["weighted"] is False
    # a uniform down-weight is NOT a no-op: it shrinks N_w and inflates B/(N_w+1)
    whalf = C.weighted_crc_lambda_hat(lam, L, np.full(L.shape[0], 0.5), alpha=0.20)
    assert whalf["N_w"] == pytest.approx(0.5 * L.shape[0])
    assert whalf["lambda_hat"] >= plain["lambda_hat"]
    with pytest.raises(ValueError):
        C.weighted_crc_lambda_hat(lam, L, np.full(L.shape[0], 1.5), alpha=0.2)
    with pytest.raises(ValueError):
        C.weighted_crc_lambda_hat(lam, L, np.ones(3), alpha=0.2)


def test_weighted_crc_downweights_the_far_cell():
    lam = np.linspace(0.0, 1.0, 11)
    near = np.clip(0.2 - lam[None, :], 0.0, 1.0).repeat(20, axis=0)
    far = np.clip(0.9 - lam[None, :], 0.0, 1.0).repeat(20, axis=0)
    L = np.vstack([near, far])
    cells = ["bench"] * 20 + ["battery"] * 20
    w = C.similarity_weights(cells, "bench", rho=0.1)
    assert w[:20].tolist() == [1.0] * 20 and w[20:].tolist() == [0.1] * 20
    eq = C.crc_lambda_hat(lam, L, alpha=0.10)["lambda_hat"]
    wt = C.weighted_crc_lambda_hat(lam, L, w, alpha=0.10)["lambda_hat"]
    assert wt <= eq          # trusting the near cell buys a smaller threshold
    assert "d_TV" in C.weighted_crc_lambda_hat(lam, L, w, alpha=0.1)["guarantee"]
    with pytest.raises(ValueError):
        C.similarity_weights(cells, "bench", rho=0.0)


def test_selective_loss_matrix_is_monotone_and_orientation_aware():
    scores = np.array([0.9, 0.1, 0.5])
    risks = np.array([1.0, 0.0, 1.0])
    lam = np.linspace(-1.0, 0.0, 11)
    L = C.selective_loss_matrix(scores, risks, lam, orientation=1)
    assert np.all(np.diff(L, axis=1) <= 1e-12)
    assert L[:, -1].max() == 0.0        # nothing emits at lambda_max -> loss 0 <= alpha
    L2 = C.selective_loss_matrix(-scores, risks, lam, orientation=-1)
    assert np.allclose(L, L2)


def test_face_violation_report_is_a_pair():
    rep = C.face_violation_report(0.3, [0.0, 0.0, 1.0], [1.0, 1.0, 1.0], alpha=0.20)
    assert rep["source"]["n"] == 3 and rep["target"]["n"] == 3
    assert rep["source"]["mean_loss"] == pytest.approx(1 / 3)
    assert rep["source"]["violated"] is True and rep["target"]["violated"] is True
    assert "pair_incomplete" not in rep
    empty = C.face_violation_report(0.3, [0.0], [], alpha=0.5)
    assert empty["target"]["n"] == 0 and empty["source"]["violated"] is False
    # JSON-safe: an absent face is None, never NaN, and the pair is flagged incomplete
    assert empty["target"]["mean_loss"] is None and empty["target"]["violated"] is None
    assert "pair_incomplete" in empty
    json.dumps(empty, allow_nan=False)


# ---------------------------------------------------------------------------
# (E) stratified CRC
# ---------------------------------------------------------------------------

def test_derive_n_strata_is_back_derived_never_tuned():
    d = C.derive_n_strata(68, 0.95, 0.05)
    assert d["n_pos_required"] == 59
    assert d["n_strata"] == 1                  # 68 // 59 = 1: 68 C->C rows cannot be cut
    d2 = C.derive_n_strata(900, 0.95, 0.05)
    assert d2["n_strata"] == 8                 # 900 // 59 = 15, capped at max_strata
    d3 = C.derive_n_strata(900, 0.95, 0.05, max_strata=4)
    assert d3["n_strata"] == 4
    d4 = C.derive_n_strata(120, 0.90, 0.05)
    assert d4["n_pos_required"] == C.sample_complexity_n_pos(0.90, 0.05)


def test_derive_n_strata_divides_the_SUCCESS_count_not_the_row_count():
    """2607.06503 3.4.1's n_pos counts successful (must-not-fire) episodes.  On the
    161-row trigger frame only 68 rows are C->C, so dividing 161 would license two
    strata while each holds ~34 successes -- the exact failure digest 4.11 warns
    about.  With the success count supplied the frame yields ONE stratum."""
    assert C.derive_n_strata(161, 0.95, 0.05)["n_strata"] == 2          # naive divisor
    d = C.derive_n_strata(161, 0.95, 0.05, n_pos_units=68)
    assert d["n_strata"] == 1
    assert d["n_pos_units"] == 68 and d["n_units"] == 161
    assert d["derivation_basis"].startswith("success")
    assert C.derive_n_strata(161, 0.95, 0.05)["derivation_basis"].startswith("all units")


def test_stratified_crc_success_mask_drives_the_thin_stratum_fallback():
    """A stratum can be wide in rows and thin in SUCCESSES; the fallback must fire on
    the success count, otherwise conditional_guarantee=True is printed for a stratum
    whose Clopper-Pearson bound cannot support the target."""
    lam = np.linspace(0.0, 1.0, 21)
    n = 240
    strat = np.arange(n, dtype=float)
    L = np.clip(0.5 - lam[None, :], 0.0, 1.0).repeat(n, axis=0)
    succ = np.zeros(n, dtype=bool)
    succ[::4] = True                                   # 60 successes among 240 rows
    loose = C.stratified_crc(strat, lam, L, alpha=0.10, rho_star=0.95, alpha_m=0.05)
    assert loose["n_strata"] == 4 and loose["n_fallback_strata"] == 0   # 240 // 59
    tight = C.stratified_crc(strat, lam, L, alpha=0.10, rho_star=0.95, alpha_m=0.05,
                             success_mask=succ)
    assert tight["derived"]["n_pos_units"] == 60
    assert tight["n_strata"] == 1                      # 60 // 59
    # forcing four strata with the mask: each holds 15 successes < 59 -> all fall back
    forced = C.stratified_crc(strat, lam, L, alpha=0.10, rho_star=0.95, alpha_m=0.05,
                              n_strata=4, success_mask=succ)
    assert forced["n_fallback_strata"] == 4
    for r in forced["strata"]:
        assert r["n"] == 60 and r["n_success"] == 15 and r["n_pos_basis"] == 15
        assert r["conditional_guarantee"] is False
    with pytest.raises(ValueError):
        C.stratified_crc(strat, lam, L, alpha=0.1, success_mask=succ[:3])


def test_stratified_crc_per_stratum_and_pooled():
    lam = np.linspace(0.0, 1.0, 21)
    rng = np.random.default_rng(7)
    n = 400
    strat = rng.random(n)
    base = 0.1 + 0.8 * strat                    # hard rows live in the high stratum
    L = np.clip(base[:, None] - lam[None, :], 0.0, 1.0)
    res = C.stratified_crc(strat, lam, L, alpha=0.10, rho_star=0.90, alpha_m=0.05,
                           stratifier_name="synthetic")
    assert res["n_strata"] == res["derived"]["n_strata"] >= 2
    assert sum(r["n"] for r in res["strata"]) == n
    # a stratum's own threshold tracks its difficulty
    lams = [r["lambda_hat"] for r in res["strata"] if r["n"]]
    assert lams == sorted(lams)
    assert all(r["conditional_guarantee"] for r in res["strata"] if not r["fallback"])
    assert res["pooled_risk"] <= res["pooled_alpha"] + 1e-9
    assert set(res["strata"][0]).issuperset({"stratum", "n", "fallback", "lambda_hat"})


def test_stratified_crc_falls_back_to_the_global_threshold_when_a_stratum_is_thin():
    lam = np.linspace(0.0, 1.0, 21)
    n = 70
    strat = np.arange(n, dtype=float)
    L = np.clip(0.5 - lam[None, :], 0.0, 1.0).repeat(n, axis=0)
    res = C.stratified_crc(strat, lam, L, alpha=0.10, rho_star=0.95, alpha_m=0.05,
                           n_strata=4, stratifier_name="forced")
    assert res["n_fallback_strata"] == 4        # every stratum is below n_pos_required=59
    for r in res["strata"]:
        assert r["fallback"] is True
        assert r["conditional_guarantee"] is False
        assert r["lambda_hat"] == pytest.approx(res["global_lambda_hat"])
    with pytest.raises(ValueError):
        C.stratified_crc(strat[:3], lam, L, alpha=0.1)


# ---------------------------------------------------------------------------
# module hygiene / label discipline
# ---------------------------------------------------------------------------

def test_deviations_block_is_complete_and_well_formed():
    assert len(C.DEVIATIONS) >= 10
    methods = {d["method"] for d in C.DEVIATIONS}
    for needle in ("CALM", "CRC feasibility", "Clopper-Pearson", "Non-exchangeable",
                   "adaptive CRC"):
        assert any(needle in m for m in methods), needle
    for d in C.DEVIATIONS:
        assert set(d) == {"method", "paper", "what", "why"}
        assert all(isinstance(v, str) and v.strip() for v in d.values())


def test_label_side_functions_are_named_label():
    for name in ("battery_label_divergence_c2kv", "battery_label_cw_loss"):
        assert "_label_" in name
        assert "LABEL SIDE" in getattr(C, name).__doc__


def test_orientations_json_declares_the_baseline_score():
    from t34_common import load_orientations
    p = ROOT / "configs/t34/orientations_calib.json"
    assert p.exists()
    ori = load_orientations(p)
    assert ori["parse_fail_fire"] == 1
    assert all(v in (1, -1) for v in ori.values())


def test_json_safe_refuses_to_smuggle_nan_into_an_artifact():
    obj = {"a": float("nan"), "b": [float("inf"), 1.0], "c": np.float64("nan"),
           "d": np.int64(3), "e": np.bool_(True), "f": np.array([1.0, np.nan])}
    safe = C.json_safe(obj)
    assert safe["a"] is None and safe["b"] == [None, 1.0] and safe["c"] is None
    assert safe["d"] == 3 and safe["e"] is True and safe["f"] == [1.0, None]
    json.dumps(safe, allow_nan=False)


def test_cli_help_for_every_subcommand():
    parser = C.build_parser()
    for cmd in ("feasibility", "sample-complexity", "recall-gate", "calm",
                "weighted-crc", "stratified-crc"):
        with pytest.raises(SystemExit) as e:
            parser.parse_args([cmd, "--help"])
        assert e.value.code == 0


# ---------------------------------------------------------------------------
# integration on the frozen assets (present on this box)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAVE_ASSETS, reason="frozen battery not on this box")
def test_frozen_frame_denominators_are_what_the_contract_says():
    frame = FrozenAssets(ROOT).load()
    sub = frame.trigger_subset()
    assert len(frame.labels) == 900
    assert len(sub) == 161
    assert sum(r["label_cw"] for r in sub) == 93
    assert len({r["session_id"] for r in sub}) == 100
    qids = [r["qid"] for r in frame.labels]
    assert C.battery_label_cw_loss(frame, qids).mean() == pytest.approx(93 / 900)
    d = C.battery_label_divergence_c2kv(frame, qids)
    assert set(np.unique(d)).issubset({0.0, 1.0})
    s = C.parse_fail_score(frame, qids)
    assert set(np.unique(s)).issubset({0.0, 1.0})


@pytest.mark.skipif(not HAVE_ASSETS, reason="frozen battery not on this box")
def test_cli_feasibility_and_recall_gate_and_sample_complexity_run():
    import argparse as _ap
    ns = _ap.Namespace(root=str(ROOT), alphas=[0.10, 0.05, 0.02, 0.01], delta=0.1,
                       reps=200, min_emit=10)
    out = C.cmd_feasibility(ns)
    assert out["n_rows"] == 900 and out["n_sessions"] == 227
    a = out["readings"]["A_cw_indicator"]
    assert a["mu"] == pytest.approx(93 / 900)
    assert a["ci95"][0] <= a["mu"] <= a["ci95"][1]
    assert a["n_clusters"] == 227
    assert a["bound_choice"]["low_variance_regime"] is True
    # both units for the finite-sample correction; the clustered one is the honest one
    assert a["finite_sample_term_row_unit"] == pytest.approx(C.finite_sample_term(900, 0.1))
    assert a["finite_sample_term_session_unit"] == pytest.approx(
        C.finite_sample_term(227, 0.1))
    assert a["finite_sample_term_session_unit"] > a["finite_sample_term_row_unit"]
    b = out["readings"]["B_one_minus_tool_name_match"]
    assert b["mu"] > a["mu"]                     # Reading B is the hopeless one
    assert b["floor_table"][0]["min_fire_rate"] > 0.5
    assert out["calibration_split"]["prevalence_subset"] == pytest.approx(93 / 161)

    ns2 = _ap.Namespace(root=str(ROOT), rho_star=[0.90, 0.95, 0.98], alpha_m=0.05)
    sc = C.cmd_sample_complexity(ns2)
    assert sc["units"]["cc_rows"] == 68 and sc["units"]["harm_rows"] == 188
    assert sc["units"]["cc_sessions"] == 46 and sc["units"]["harm_sessions"] == 100
    assert sc["certifiable_recall_ceiling"]["cc_rows"] == pytest.approx(0.9569, abs=1e-4)
    assert sc["certifiable_recall_ceiling"]["harm_rows"] == pytest.approx(0.9842, abs=1e-4)

    ns3 = _ap.Namespace(root=str(ROOT), rho_star=0.95, alpha=0.05, alpha_m=0.05,
                        margin=0.02)
    rg = C.cmd_recall_gate(ns3)
    assert rg["R_g"] == 1 and rg["cascade_claim_forbidden"] is True
    assert rg["frame"]["n_pos"] == 93 and rg["frame"]["n_neg"] == 68
    assert rg["operating_point"]["n_pos"] == 93
    assert "precision" in rg
    # 2607.06503 3.4: the quotable certificate is the one on data tau never saw.
    assert "certificate" not in rg
    assert rg["certificate_to_quote"] == "certificate_split"
    ins, spl = rg["certificate_in_sample"], rg["certificate_split"]
    assert ins["in_sample"] is True and ins["valid_post_selection"] is False
    assert ins["n_pos"] == 68                      # all C->C rows chose tau
    assert spl["in_sample"] is False and spl["valid_post_selection"] is True
    assert 0 < spl["n_pos"] < ins["n_pos"]         # a disjoint half, strictly smaller
    assert spl["ceiling"] < ins["ceiling"]         # halving n_pos costs ceiling
    assert spl["target_above_ceiling"] is True     # 0.95 is unreachable on 29 rows
    assert spl["n_sessions_cal"] + spl["n_sessions_cert"] == 100


@pytest.mark.skipif(not HAVE_ASSETS, reason="frozen battery not on this box")
def test_cli_calm_and_stratified_and_weighted_run_on_the_frozen_frame():
    import argparse as _ap
    ns = _ap.Namespace(root=str(ROOT), delta=0.30, eps=0.05, step=0.05, pvalue="hb",
                       unit="session", repair_outcomes=None, assume_repair_success=0.806)
    out = C.cmd_calm(ns)
    assert out["unit"] == "session" and out["n_units"] == 100
    assert out["assumed_repair_success"] == pytest.approx(0.806)
    assert "ASSUMPTION" in out["assumption_warning"]
    assert len(out["rows"]) == 21 and out["rows"][0]["lambda"] == pytest.approx(1.0)
    assert out["extras"]["slack_sessions_pp"] > out["extras"]["slack_rows_pp"]
    assert len(out["bonferroni_variant"]) == 21

    ns2 = _ap.Namespace(root=str(ROOT), stratifier="actual_compression_ratio",
                        alpha=0.30, rho_star=0.90, alpha_m=0.05)
    st = C.cmd_stratified_crc(ns2)
    assert st["label_free"] is True
    assert sum(r["n"] for r in st["strata"]) == 161
    assert st["stratifier_field"] == "actual_compression_ratio"
    # the n_pos rule is applied to the 68 C->C rows, not to all 161
    assert st["derived"]["n_pos_units"] == 68 and st["derived"]["n_units"] == 161

    ns2b = _ap.Namespace(root=str(ROOT), stratifier="actual_compression_ratio",
                         alpha=0.30, rho_star=0.95, alpha_m=0.05)
    st2 = C.cmd_stratified_crc(ns2b)
    # digest 4.11's own conclusion: at rho* = 0.95 the calibration set cannot be cut
    assert st2["derived"]["n_pos_required"] == 59 and st2["n_strata"] == 1
    # the three metrics are reported per stratum and then pooled, against the
    # stratum's OWN prevalence as chance precision
    assert len(st2["per_stratum_metrics"]) == 1
    m0 = st2["per_stratum_metrics"][0]
    assert m0["n"] == 161
    assert m0["chance_precision"] == pytest.approx(93 / 161)
    assert set(m0["operating_point"]) >= {"coverage", "precision", "false_resets"}
    assert st2["pooled_metrics"]["n_pos"] == 93
    assert st2["pooled_prevalence"] == pytest.approx(93 / 161)
    json.dumps(C.json_safe(st2), allow_nan=False)

    ns3 = _ap.Namespace(root=str(ROOT), alpha=0.30, rho=0.9, target_cell="bench",
                        target_losses=None)
    wc = C.cmd_weighted_crc(ns3)
    assert wc["face_violation_pair"]["target"]["n"] == 0
    assert "pair_incomplete" in wc["face_violation_pair"]
    assert wc["target_rows_available"] == 0
    assert "none claimed" in wc["guarantee_language"]
    json.dumps(wc, allow_nan=False)          # every CLI artifact must be strict JSON


@pytest.mark.skipif(not HAVE_ASSETS, reason="frozen battery not on this box")
def test_calm_rejects_a_missing_repair_input():
    import argparse as _ap
    ns = _ap.Namespace(root=str(ROOT), delta=0.3, eps=0.05, step=0.05, pvalue="hb",
                       unit="session", repair_outcomes=None, assume_repair_success=None)
    with pytest.raises(SystemExit):
        C.cmd_calm(ns)


@pytest.mark.skipif(not HAVE_ASSETS, reason="frozen battery not on this box")
def test_stratifier_must_be_a_label_free_prefix_scalar_present_on_the_row():
    import argparse as _ap
    # tool_name_match IS on the battery row but it is the LABEL: the allowlist must
    # refuse it even though a bare presence check would let it through.
    ns = _ap.Namespace(root=str(ROOT), stratifier="tool_name_match", alpha=0.3,
                       rho_star=0.9, alpha_m=0.05)
    with pytest.raises(SystemExit):
        C.cmd_stratified_crc(ns)
    # dropped_docs is sidecar-only, never silently substituted by a row column
    ns2 = _ap.Namespace(root=str(ROOT), stratifier="dropped_docs", alpha=0.3,
                        rho_star=0.9, alpha_m=0.05)
    with pytest.raises(SystemExit):
        C.cmd_stratified_crc(ns2)
    # every allowlisted name survives the t33 leakage guard
    from t33_labels import guard_columns
    guard_columns(sorted(set(C.PREFIX_SCALARS.values())), context="test")


def test_feasibility_accepts_a_real_valued_score_column(tmp_path):
    import json, subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    if not (root / "results/bdf_pilot/d_r2/battery_full.jsonl").exists():
        pytest.skip("frozen battery not on this box")
    import t34_common as C
    frame = C.FrozenAssets(root).load()
    feats = tmp_path / "f.jsonl"
    with open(feats, "w", encoding="utf-8") as fh:
        for i, r in enumerate(frame.trigger_subset()):
            fh.write(json.dumps({"qid": r["qid"], "x": float(i % 7) / 7.0 if i % 11 else None}) + "\n")
    out = tmp_path / "feas.json"
    cmd = [sys.executable, str(root / "agent/t34_calibrate.py"), "--out", str(out), "feasibility",
           "--root", str(root), "--alphas", "0.10", "--reps", "20", "--score-column", f"{feats}:x:-1"]
    res = subprocess.run(cmd, capture_output=True, text=True, env={**__import__('os').environ, "PYTHONIOENCODING": "utf-8"})
    assert res.returncode == 0, res.stderr[-800:]
    d = json.loads(out.read_text(encoding="utf-8"))
    assert "rows without a value dropped" in json.dumps(d)
