# -*- coding: utf-8 -*-
import numpy as np
import pytest

import t34_common as C


def test_average_precision_tie_invariant_and_matches_sklearn():
    pytest.importorskip("sklearn")
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=60)
    s = np.round(rng.random(60), 1)  # many ties
    ap = C.average_precision(s, y)
    assert ap == pytest.approx(average_precision_score(y, s), abs=1e-12)
    # permuting rows inside tie blocks must not change AP
    perm = rng.permutation(60)
    assert C.average_precision(s[perm], y[perm]) == pytest.approx(ap, abs=1e-12)


def test_average_precision_binary_predictor_closed_form():
    # 35 TP, 28 FP fire; 58 positives not fired: AP = (35/63)*(35/93) + ... distinct-threshold
    y = np.array([1] * 93 + [0] * 68)
    s = np.zeros(161)
    s[:35] = 1.0
    s[93:93 + 28] = 1.0
    ap = C.average_precision(s, y)
    # thresholds: {1.0}: prec 35/63, rec 35/93 ; {0.0}: prec 93/161, rec 1
    expect = (35 / 63) * (35 / 93) + (93 / 161) * (1 - 35 / 93)
    assert ap == pytest.approx(expect, abs=1e-12)


def test_auroc_average_ranks():
    y = np.array([1, 1, 0, 0])
    s = np.array([0.5, 0.5, 0.5, 0.1])
    assert C.auroc(s, y) == pytest.approx(0.75)


def test_clustered_bootstrap_uses_present_clusters_only():
    y = np.array([1, 0, 1, 0, 1, 0])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.3])
    cl = C.session_clusters(["a", "a", "b", "b", "c", "c"])
    lo, hi, n = C.clustered_bootstrap(C.auroc, s, y, cl, reps=50)
    assert n == 3 and lo is not None and hi is not None and lo <= hi


def test_operating_point_denominators():
    y = np.array([1, 1, 0, 0, 1])
    s = np.array([0.9, 0.1, 0.8, 0.2, 0.5])
    op = C.operating_point(s, y, 2)
    assert op["fires"] == 2 and op["coverage"] == 1 and op["false_resets"] == 1
    assert op["n_pos"] == 3 and op["n_neg"] == 2 and op["precision"] == 0.5


def test_locate_table_counts_abstentions_in_denominator():
    hits = {"a": True, "b": False, "c": None, "d": True}
    t = C.locate_table(hits)
    assert t["n"] == 4 and t["hits"] == 2 and t["abstained"] == 1
    assert t["s_at_k"] == 0.5
    assert 0.0 <= t["p_vs_floor"] <= 1.0


def test_exact_binom_and_clopper_pearson():
    assert C.exact_binom_one_sided(71, 93, 0.25) < 1e-6
    lo, hi = C.clopper_pearson(71, 93)
    assert 0.65 < lo < 0.763 < hi < 0.86


def test_inverted_score_control_flags_artefact():
    truth = {f"q{i}": 0 for i in range(20)}
    good = {q: [1.0, 0.2, 0.1] for q in truth}
    out = C.inverted_score_control(good, truth)
    assert out["forward"]["hits"] == 20 and out["inverted"]["hits"] == 0
    assert out["mcnemar_p"] < 1e-4


def test_nested_cv_selects_in_inner_folds_only():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    n = 200
    groups = np.array([f"s{i // 4}" for i in range(n)])
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(int)
    res = C.nested_cv_logistic(X, y, groups, c_grid=(0.01, 1.0), outer_folds=4, inner_folds=3)
    assert res["n_scored"] == n and res["auroc"] > 0.85
    assert all("C" in c and "inner_metric" in c for c in res["chosen"])


def test_nested_cv_drops_nan_rows_and_reports():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 2))
    X[3, 0] = np.nan
    y = (X[:, 1] > 0).astype(int)
    groups = np.array([f"s{i // 2}" for i in range(40)])
    res = C.nested_cv_logistic(X, y, groups, c_grid=(1.0,), outer_folds=3, inner_folds=2)
    assert res["n_dropped_nan"] == 1 and res["n_scored"] == 39


def test_write_features_refuses_leaky_columns(tmp_path):
    with pytest.raises(ValueError):
        C.write_features_jsonl(tmp_path / "f.jsonl", [{"qid": "a", "tool_name_match": 1}], context="t")
    n = C.write_features_jsonl(tmp_path / "f.jsonl", [{"qid": "a", "x": None, "y": 1.0}], context="t")
    assert n == 1
    text = (tmp_path / "f.jsonl").read_text(encoding="utf-8")
    assert '"x": null' in text  # missingness kept visible


def test_first_divergence_index():
    assert C.first_divergence_index([1, 2, 3], [1, 2, 4]) == 2
    assert C.first_divergence_index([1, 2], [1, 2, 3]) is None


def test_session_and_step_helpers():
    assert C.session_of("abc_123:7") == "abc_123" and C.step_index("abc_123:7") == 7


def test_chooser_semantics_match_select_k_star():
    assert C.chooser_argmax([0.0, 0.0]) is None           # no block above zero -> abstain
    assert C.chooser_argmax([0.5, 0.5, 0.1]) == 0         # lowest index on ties
    assert C.chooser_argmax([float("nan"), 0.2]) == 1
    assert C.chooser_argmin([1.0, 1.0]) is None           # constant: nothing to invert
    assert C.chooser_argmin([0.3, 0.1, 0.1]) == 1


def test_inverted_control_abstains_like_the_chooser():
    truth = {"a": 0, "b": 0}
    out = C.inverted_score_control({"a": [0.0, 0.0], "b": [2.0, 1.0]}, truth)
    assert out["forward"]["abstained"] == 1 and out["forward"]["hits"] == 1
    assert out["inverted"]["abstained"] == 1 and out["inverted"]["hits"] == 0


def test_parse_fail_baseline_is_prediction_only_by_default():
    from t33_labels import parse_fail_baseline
    bad = "<tool_call>{\"name\": \"f\", \"arguments\": {"
    assert parse_fail_baseline(bad) is True
    assert parse_fail_baseline(bad, target_has_tool_call=False) is False  # gold-gated variant only
    good = "<tool_call>{\"name\": \"f\", \"arguments\": {}}</tool_call>"
    assert parse_fail_baseline(good) is False


def test_label_frame_keeps_deployable_and_gold_gated_columns_and_they_agree_on_161():
    root = C._HERE.parent
    if not (root / "results/bdf_pilot/d_r2/battery_full.jsonl").exists():
        pytest.skip("frozen battery not on this box")
    frame = C.FrozenAssets(root).load()
    sub = frame.trigger_subset()
    assert all(r["parse_fail_fire"] == r["parse_fail_fire_gold_gated"] for r in sub)
    assert sum(1 for r in sub if r["parse_fail_fire"]) == 63


def test_load_flip_table_accepts_raw_ksweep_rows(tmp_path):
    import json
    from t34_common import load_flip_table
    p = tmp_path / 'sweep.jsonl'
    rows = [
        {'qid': 's:1', 'd_ksweep_k': 0, 'd_corr_doc_index': 0, 'tool_name_match': False, 'skipped': False},
        {'qid': 's:1', 'd_ksweep_k': 2, 'd_corr_doc_index': 2, 'tool_name_match': True, 'skipped': False},
        {'qid': 's:1', 'd_ksweep_k': 3, 'd_corr_doc_index': 3, 'tool_name_match': True, 'skipped': True},
        {'qid': 's:2', 'k': 1, 'correct': True},
    ]
    p.write_text(chr(10).join(json.dumps(r) for r in rows) + chr(10), encoding='utf-8')
    ft = load_flip_table(p)
    assert ft == {'s:1': {0: False, 2: True}, 's:2': {1: True}}
