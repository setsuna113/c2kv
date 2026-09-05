# -*- coding: utf-8 -*-
"""Unit tests for the t33 group-1 fixes (audit 2026-09-05).

Covers: tie-aware AP, midrank residualization determinism, orientation before
residualization, winner-clause prevalences, the causal period>=3 repeat
channel, session-prefix entropy baselines, the SMT six-class mask, the
args-span syntax fix, and the parse-failure baseline's gold-column removal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from t33_score import (  # noqa: E402
    ORIENTATIONS, auprc, auroc, midrank, rank_residualize, score_feature, verdict,
)
from t33_extract_features import (  # noqa: E402
    _repeat_trigram_coverage, session_entropy_baselines, smt_token_set,
)
from t33_labels import guard_columns, parse_fail_baseline  # noqa: E402
from t33_spanmap import parse_tool_call  # noqa: E402

TOOL_CALL_TEXT = (
    "(no content\n\nAction:\n<tool_call>\n"
    '{"name":"get_weather","arguments":{"city":"Paris","unit":"C"}}\n'
    "</tool_call>"
)


# ------------------------------------------------------------------ AUPRC ties

def test_auprc_ties_hand_computed():
    # two tie groups of two; positives split across groups
    scores = np.array([0.9, 0.9, 0.1, 0.1])
    labels = np.array([1, 0, 1, 0])
    # group {0.9}: precision 1/2, one positive; group {0.1}: precision 2/4
    assert auprc(scores, labels) == pytest.approx((0.5 * 1 + 0.5 * 1) / 2)


def test_auprc_ties_do_not_depend_on_row_order():
    # a big tie block must give the same AP regardless of label arrangement
    scores = np.array([1.0] * 20 + [0.0] * 20)
    labels = np.zeros(40, dtype=int)
    labels[:6] = 1
    a = auprc(scores, labels)
    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = rng.permutation(40)
        assert auprc(scores[perm], labels[perm]) == pytest.approx(a)


def test_auprc_matches_sklearn():
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = rng.integers(30, 120)
        scores = rng.normal(size=int(n))
        scores[rng.random(int(n)) < 0.3] = 0.5  # force ties
        labels = rng.random(int(n)) < 0.4
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        assert auprc(scores, labels.astype(int)) == pytest.approx(
            sklearn.average_precision_score(labels.astype(int), scores))


def test_baseline_ap_known_value():
    # 30-fire parse baseline on the 161 subset: 23 C->W fires in the top tie
    # block, 7 C->C fires in the bottom block, 70/61 remainder
    pf = np.zeros(161)
    pf[:23] = 1.0        # fires that are C->W
    pf[100:107] = 1.0    # fires that are C->C
    y = np.zeros(161, dtype=int)
    y[:23] = 1
    y[23:93] = 1
    # fires: 30 (23 pos, 7 neg); positives 93
    # tie group {1.0}: precision 23/30, 23 pos; group {0.0}: precision 93/161, 70 pos
    expected = ((23 / 30) * 23 + (93 / 161) * 70) / 93
    assert auprc(pf, y) == pytest.approx(expected)


# ------------------------------------------------- midrank / residualization

def test_midrank_ties_averaged_and_deterministic():
    x = np.array([3.0, 1.0, 3.0, 2.0, 3.0])
    r = midrank(x)
    # the three 3.0s occupy sorted positions 3,4,5 -> average rank 4
    assert r[0] == pytest.approx(4.0)
    assert r[1] == 1.0 and r[3] == 2.0
    for _ in range(3):
        assert np.allclose(midrank(x), r)


def test_rank_residualize_negation_mirrors():
    # residualizing the negated feature mirrors the residual exactly, so the
    # (v, +1) and (-v, -1) oriented pairs collapse onto the same risk and the
    # same AP(len-ctl) once orientation is applied BEFORE residualization
    rng = np.random.default_rng(7)
    v = rng.normal(size=80)
    control = rng.normal(size=80)
    assert np.allclose(rank_residualize(-v, control), -rank_residualize(v, control))


def test_orientation_applied_before_residualization_in_score_feature():
    rng = np.random.default_rng(11)
    n = 120
    base = rng.normal(size=n)
    control = np.clip(rng.normal(size=n) * 3 + 5, 1, None).astype(int)
    y = (rng.random(n) < 0.4).astype(int)
    frame = []
    for i in range(n):
        frame.append({
            "qid": f"q{i}", "session_id": f"s{i % 10}",
            "c::feat_pos": float(base[i]), "c::feat_neg": float(-base[i]),
            "n_generated": int(control[i]), "censored": i % 3 == 0,
            "parse_fail_fire": i % 5 == 0, "label_cw": int(y[i]),
        })
    keep = np.ones(n, dtype=bool)
    labels = y.copy()
    sessions = np.array([i % 10 for i in range(n)])
    ORIENTATIONS["feat_pos"] = 1
    ORIENTATIONS["feat_neg"] = -1
    e_pos = score_feature(frame, "c::feat_pos", sessions, labels, keep)
    e_neg = score_feature(frame, "c::feat_neg", sessions, labels, keep)
    assert e_pos["auprc"] == pytest.approx(e_neg["auprc"])
    assert e_pos["auprc_length_controlled"] == pytest.approx(
        e_neg["auprc_length_controlled"])
    assert e_pos["n_scored"] == n


def test_score_feature_complete_case_no_median_fill():
    rng = np.random.default_rng(13)
    n = 100
    v = rng.normal(size=n)
    y = (rng.random(n) < 0.5).astype(int)
    frame = [{
        "qid": f"q{i}", "session_id": f"s{i % 8}", "c::f": (
            float(v[i]) if i < 60 else None),
        "n_generated": 50 + i, "censored": i % 4 == 0,
        "parse_fail_fire": i % 6 == 0, "label_cw": int(y[i]),
    } for i in range(n)]
    e = score_feature(frame, "c::f", np.array([i % 8 for i in range(n)]),
                      y.copy(), np.ones(n, dtype=bool))
    assert e["n_scored"] == 60
    assert e["eval_prevalence"] == pytest.approx(float(y[:60].mean()), abs=0.01)


# ------------------------------------------------------------------- verdict

def _entry(**over):
    e = {
        "auprc": 0.7, "auprc_ci": [0.60, 0.8], "orientation": 1,
        "eval_prevalence": 0.578, "delta_vs_s0": {"ci_lo": 0.02},
        "auprc_length_controlled": 0.65, "auprc_len_only": 0.60,
        "auprc_uncensored": 0.60, "uncens_prevalence": 0.5595,
        "matched_rate_op": {"coverage": 40, "precision": 0.7, "false_resets": 10},
        "parse_baseline": {"coverage": 35, "precision": 0.55, "false_resets": 12},
    }
    e.update(over)
    return e


def test_verdict_live_when_all_clauses_pass():
    assert verdict(_entry()) == "LIVE"


def test_verdict_ci_clause_uses_own_prevalence():
    # CI lower 0.55 fails a 0.578-prevalence subset but would have passed the
    # old vacuous 0.1033 threshold
    assert verdict(_entry(auprc_ci=[0.55, 0.8])) == "not-live"
    assert verdict(_entry(auprc_ci=[0.60, 0.8])) == "LIVE"


def test_verdict_len_clause_requires_increment_over_len_only():
    assert verdict(_entry(auprc_length_controlled=0.55, auprc_len_only=0.60)) == "not-live"
    assert verdict(_entry(auprc_length_controlled=0.62, auprc_len_only=0.60)) == "LIVE"


def test_verdict_uncensored_clause_uses_slice_prevalence():
    assert verdict(_entry(auprc_uncensored=0.55, uncens_prevalence=0.5595)) == "not-live"
    assert verdict(_entry(auprc_uncensored=0.58, uncens_prevalence=0.5595)) == "LIVE"


def test_verdict_orientation_zero_is_control():
    assert "control" in verdict(_entry(orientation=0))


# ---------------------------------------------------------- e-CUSUM channels

def test_repeat_channel_detects_period3_cycle():
    ids = [1, 2, 3] * 10
    r = _repeat_trigram_coverage(ids, 16)
    assert max(r) > 0.0, "period-3 cycle must light the repeat channel"
    assert r[10] > 0.0


def test_repeat_channel_causal():
    ids = list(np.random.default_rng(5).integers(0, 50, size=64))
    r1 = _repeat_trigram_coverage(ids, 16)
    mutated = list(ids)
    mutated[40] = 999  # future token change
    r2 = _repeat_trigram_coverage(mutated, 16)
    # r at t only sees trigrams ending <= t, so r[0:38] must be identical
    assert np.allclose(r1[:38], r2[:38])


def test_repeat_channel_zero_for_fresh_sequence():
    ids = list(range(60))  # no repeated trigram at all
    assert max(_repeat_trigram_coverage(ids, 16)) == 0.0


def test_session_entropy_baselines():
    rows = [
        {"qid": "s1:0", "meta": {"session_id": "s1"},
         "steps": [{"entropy_full": 1.0}, {"entropy_full": 2.0}]},
        {"qid": "s1:1", "meta": {"session_id": "s1"},
         "steps": [{"entropy_full": 4.0}]},
        {"qid": "s2:0", "meta": {"session_id": "s2"},
         "steps": [{"entropy_full": 8.0}]},
    ]
    b = session_entropy_baselines(rows)
    assert b["s1:0"] is None          # first row: no history
    assert b["s1:1"] == pytest.approx(1.5)  # mean of s1:0's tokens
    assert b["s2:0"] is None


# ------------------------------------------------------------------ SMT mask

def _offsets_from(text, tok_lens):
    offs, cur = [], 0
    for ln in tok_lens:
        offs.append((cur, cur + ln))
        cur += ln
    assert cur == len(text), (cur, len(text))
    return offs


def test_smt_mask_includes_arg_keys_and_decision_token():
    parsed = parse_tool_call(TOOL_CALL_TEXT)
    # uniform 1-char tokens for simplicity: text chars == tokens
    offs = [(i, i + 1) for i in range(len(TOOL_CALL_TEXT))]
    spans = {"first_tok": 0, "name_first": parsed["name_span"][0]}
    toks = smt_token_set(TOOL_CALL_TEXT, spans, offs)
    # decision token: first generated token
    assert 0 in toks
    # name value chars are in the mask
    for c in range(*parsed["name_span"]):
        assert c in toks
    # argument KEY chars must be included (paper class 3)
    args_text = TOOL_CALL_TEXT[parsed["args_span"][0]:parsed["args_span"][1]]
    key_chars = set()
    i = 0
    while i < len(args_text):
        if args_text[i] == '"':
            j = args_text.find('"', i + 1)
            rest = args_text[j + 1:j + 3].lstrip()
            if rest.startswith(":"):
                key_chars.update(range(parsed["args_span"][0] + i + 1,
                                       parsed["args_span"][0] + j))
            i = j + 1
        else:
            i += 1
    assert key_chars, "test text must contain argument keys"
    assert key_chars.issubset(toks), "class (3) arg-name tokens must be in the mask"


def test_smt_mask_excludes_braces_and_colons():
    parsed = parse_tool_call(TOOL_CALL_TEXT)
    offs = [(i, i + 1) for i in range(len(TOOL_CALL_TEXT))]
    spans = {"first_tok": 0}
    toks = smt_token_set(TOOL_CALL_TEXT, spans, offs)
    brace_positions = {i for i, c in enumerate(TOOL_CALL_TEXT) if c in "{}:"}
    assert not (brace_positions & toks)


# ---------------------------------------------------- spanmap syntax position

def test_args_span_skips_json_brace():
    parsed = parse_tool_call(TOOL_CALL_TEXT)
    cs, _ce = parsed["args_span"]
    assert TOOL_CALL_TEXT[cs] != "{", "args span must start after the brace"
    assert TOOL_CALL_TEXT[cs] == '"', "args span starts at the first key quote"


def test_parse_tool_call_name_and_values():
    parsed = parse_tool_call(TOOL_CALL_TEXT)
    assert parsed["name"] == "get_weather"
    assert parsed["parse_ok"]


# ------------------------------------------------- parse-failure baseline

def test_parse_fail_baseline_no_gold_column():
    # no target_has_tool_call argument anymore; pure compressed-arm signal
    assert parse_fail_baseline(TOOL_CALL_TEXT) is False
    assert parse_fail_baseline("no tool call at all") is True
    assert parse_fail_baseline('<tool_call>\n{"name": "x", "arguments": {') is True


# ------------------------------------------------------------------- guard

def test_guard_columns_leak_specimens():
    with pytest.raises(ValueError):
        guard_columns(["a_made_call", "nice_column"])
    with pytest.raises(ValueError):
        guard_columns(["tool_name_match"])
    guard_columns(["flare_min_p_all", "hbar_name"])  # must not raise
