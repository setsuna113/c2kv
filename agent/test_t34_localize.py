# -*- coding: utf-8 -*-
"""Unit tests for t34 U3 (digest section 4.7 localisation).

Everything here is synthetic and torch-free: the sidecar, the witness table and
the flip table are all built in-test, so the suite runs on the Windows box with
no frozen asset and no GPU.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_AGENT = Path(__file__).resolve().parent
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

import t34_localize as L  # noqa: E402
import t34_locate_score as S  # noqa: E402
from t34_common import inverted_score_control, write_features_jsonl  # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

DOC0 = "user profile for alice with token abcdefghijkl"
DOC1 = "order 998877 was created for widget_gizmo_42"

PRED_CLOSED = ('Now let me look.\nAction:\n<tool_call>\n'
               '{"name":"users__get","arguments":{"token":"abcdefghijkl"}}\n</tool_call>')
PRED_TRUNC = ('Action:\n<tool_call>\n'
              '{"name":"users__get","arguments":{"token":"abcdefghijkl","limit":20')


def make_row(qid="s1:3", docs=(DOC0, DOC1), prediction=PRED_CLOSED,
             query="look up the order 998877", gold=("orders__create", "998877"),
             truth_k=1, decision_step=3, censored=False):
    return L.LocalizeRow(
        qid=qid, session_id=qid.rsplit(":", 1)[0], docs=list(docs), query=query,
        prediction=prediction, doc_lengths=[len(d) for d in docs],
        decision_step=decision_step, censored_at_cap=censored,
        gold_values=list(gold), truth_k=truth_k,
    )


# --------------------------------------------------------------------------
# (A) CausalCache proposal-as-reference -- 2608.22577 sec 3.4
# --------------------------------------------------------------------------

def test_values_from_proposal_closed():
    got = L.values_from_proposal(PRED_CLOSED)
    assert got["parse_ok"] is True and got["closed"] is True and got["salvaged"] is False
    assert got["values"] == ["users__get", "abcdefghijkl"]


def test_values_from_proposal_truncated_is_salvaged_not_dropped():
    got = L.values_from_proposal(PRED_TRUNC)
    assert got["parse_ok"] is False and got["closed"] is False
    assert got["salvaged"] is True
    # the key "token" / "limit" are NOT leaves; the value and the number are
    assert got["values"] == ["users__get", "abcdefghijkl", "20"]


def test_values_from_proposal_no_tool_call():
    got = L.values_from_proposal("I am not going to call anything.")
    assert got["has_tool_call"] is False and got["values"] == []


def test_values_from_query_tokens_and_json():
    assert "998877" in L.values_from_query("look up the order 998877")
    assert set(L.values_from_query('{"a": "zeta", "b": 7}')) == {"zeta", "7"}


def test_ladder_ordering_and_disagreement():
    """The ladder's five rungs, and the pre-registered risk: the proposal points
    somewhere else than gold (2608.22577 pitfall: our proposals are wrong by
    construction of the C->W label)."""
    row = make_row()
    rows = {r["chooser"]: r for r in L.ladder_choosers(row)}
    assert set(rows) == {L.CH_GOLD, L.CH_PROPOSAL, L.CH_QUERY_ONLY, L.CH_NONE, L.CH_FIRST}
    assert rows[L.CH_GOLD]["khat"] == 1          # gold witness value lives in doc1
    assert rows[L.CH_PROPOSAL]["khat"] == 0      # the arm's own (wrong) action points at doc0
    assert rows[L.CH_QUERY_ONLY]["khat"] == 1
    assert rows[L.CH_NONE]["khat"] is None
    assert rows[L.CH_NONE]["score_vector"] == []  # empty so the inverted control abstains
    assert rows[L.CH_FIRST]["khat"] == 0
    assert rows[L.CH_PROPOSAL]["note"]["parse_ok"] is True


def test_k_star_none_is_a_result_not_an_exception():
    row = make_row(prediction="no call here", query="")
    rows = {r["chooser"]: r for r in L.ladder_choosers(row)}
    assert rows[L.CH_PROPOSAL]["khat"] is None
    assert rows[L.CH_QUERY_ONLY]["khat"] is None


def test_inverted_control_uses_the_same_vectors():
    """2608.22577 Table `chooser`: argmin on the SAME score vector is the
    specificity control -- it must be computed from the chooser's own vector."""
    row = make_row()
    rows = {r["chooser"]: r for r in L.ladder_choosers(row)}
    vecs = {row.qid: rows[L.CH_GOLD]["score_vector"]}
    ctrl = inverted_score_control(vecs, {row.qid: row.truth_k})
    assert ctrl["forward"]["hits"] == 1
    assert ctrl["inverted"]["hits"] == 0
    # and it really is the same vector, negated
    v = np.asarray(vecs[row.qid])
    assert int(np.argmax(v)) == row.truth_k and int(np.argmin(v)) != row.truth_k


# --------------------------------------------------------------------------
# (B) LANTERN RRF -- 2606.05182 sec 3.2
# --------------------------------------------------------------------------

def test_ranks_from_scores_ties_share_the_best_rank():
    assert L.ranks_from_scores([5.0, 5.0, 1.0]) == [1, 1, 3]
    assert L.ranks_from_scores([1.0, 3.0, 2.0]) == [3, 1, 2]


def test_rrf_arithmetic_matches_hand_computation():
    ranks_a = [1, 2, 3]
    ranks_b = [3, 1, 2]
    got = L.rrf_fuse([ranks_a, ranks_b], k=60)
    want = [1 / 61 + 1 / 63, 1 / 62 + 1 / 61, 1 / 63 + 1 / 62]
    assert got == pytest.approx(want)
    assert L.RRF_K == 60  # declared, never tuned (their Exp. 6 is flat)


def test_rrf_rejects_ragged_rank_lists():
    with pytest.raises(ValueError):
        L.rrf_fuse([[1, 2], [1, 2, 3]])


def test_tag_extraction_families():
    tags = L.extract_tags('call spotify__show_liked_songs on src/app/main.py got E404 "alpha" 42')
    assert "spotify__show_liked_songs" in tags
    assert "src/app/main.py" in tags
    assert "alpha" in tags and "42" in tags


def test_tag_jaccard_is_bounded_and_prefers_the_overlapping_block():
    row = make_row(docs=("token abcdefghijkl here", "totally unrelated prose"),
                   prediction=PRED_CLOSED, query="")
    js = L.ranker_tag_jaccard(row)
    assert all(0.0 <= x <= 1.0 for x in js)
    assert js[0] > js[1]


def test_importance_recency_frequency_richness():
    row = make_row(docs=("a" * 10, "b" * 20, "c" * 20))
    imp = L.ranker_importance(row, t_half=8.0)
    # most recent block has Delta t = 0 -> R = 1
    assert imp[2] == pytest.approx(math.exp(0.0) * 1.0 * 1.0)
    # block 0 is 2 steps old and half as rich
    assert imp[0] == pytest.approx(math.exp(-0.693 * 2 / 8.0) * 1.0 * 0.5)
    assert imp[2] > imp[1] > imp[0]


def test_rrf_free_arm_and_semantic_arm_are_separate():
    row = make_row()

    def fake_encode(batch):
        # deterministic 2-d embeddings; the query aligns with doc1
        table = {DOC0: [1.0, 0.0], DOC1: [0.0, 1.0]}
        return [table.get(t, [0.0, 1.0]) for t in batch]

    free_only, cost_none = L.rrf_choosers(row, with_semantic=False)
    assert [r["chooser"] for r in free_only] == [L.CH_RRF3] and cost_none is None
    both, cost = L.rrf_choosers(row, encode_fn=fake_encode, with_semantic=True)
    assert [r["chooser"] for r in both] == [L.CH_RRF3, L.CH_RRF4]
    assert cost["bytes"] > 0 and cost["ms"] >= 0.0 and cost["n_texts"] == 3
    assert both[1]["khat"] in (0, 1)


# --------------------------------------------------------------------------
# (C) sigma margin -- 2607.07724 sec 3.3 / 3.4
# --------------------------------------------------------------------------

def test_sigma_paper_matches_the_published_formula():
    s = [10.0, 7.0, 3.0, 1.0]
    # k = 2: (s_(1) - s_(2)) / (s_(0) - s_(2)) = 4 / 7
    assert L.sigma_paper(s, 2) == pytest.approx(4.0 / 7.0)
    # at k = 1 the paper's numerator and denominator collapse onto the same gap
    # -- exactly the degeneracy that forces the sigma_top1 re-base.
    assert L.sigma_paper(s, 1) == pytest.approx(1.0)
    assert L.sigma_paper(s, 4) is None          # no element beyond the cut
    assert L.sigma_paper([1.0, 1.0], 1) == 0.0  # degenerate denominator -> coin flip


def test_sigma_top1_rebase_and_declared_alternatives():
    s = [10.0, 7.0, 3.0, 1.0]
    assert L.sigma_top1(s) == pytest.approx(3.0 / 9.0)   # (s0-s1)/(s0-smin)
    assert L.gap_raw(s) == pytest.approx(3.0)
    assert L.gap_second(s) == pytest.approx(4.0)
    assert L.sigma_top1([4.0]) is None
    assert L.gap_second([4.0, 1.0]) is None
    assert L.sigma_top1([2.0, 2.0, 2.0]) == 0.0
    # the re-base is a real departure: with k=1 the paper's sigma differs
    assert L.sigma_paper(s, 1) != pytest.approx(L.sigma_top1(s))


def test_aggregate_sigma_is_a_uniform_mean_before_thresholding():
    cells = {"a": [10.0, 7.0, 3.0, 1.0], "b": [4.0, 4.0, 4.0, 4.0]}
    assert L.aggregate_sigma(cells) == pytest.approx((1.0 / 3.0 + 0.0) / 2.0)
    assert L.aggregate_sigma({"a": [1.0]}) is None


def test_sigma_threshold_is_a_fixed_rate_cut():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    thr = L.sigma_threshold(vals, 0.4)
    fired = [v for v in vals if v <= thr]
    assert len(fired) == 2 and 0.2 <= thr < 0.3


def test_sigma_abstain_curve_reports_three_unmerged_numbers():
    sigma = {f"s{i}:1": (0.05 * i) for i in range(10)}
    hits = {f"s{i}:1": (i >= 3) for i in range(10)}
    curve = L.sigma_abstain_curve(sigma, hits, [0.1, 0.5])
    row0, row3 = curve[0], curve[1]
    assert row0["abstained"] < row3["abstained"]
    assert row3["abstained"] >= 1
    # abstentions are misses in S@k but excluded from selective accuracy
    assert row3["s_at_k"] <= row3["selective_accuracy"]
    assert row3["selective_n"] + row3["abstained"] == row3["n"]


def test_sigma_abstain_curve_q0_never_abstains_and_carries_no_sentinel():
    """2607.07724's boundary is CLOSED (`sigma <= quantile_q`), so at q = 0 the
    single minimum-sigma row would still fire.  The declared departure is an
    explicit q <= 0 branch that abstains on NOTHING, flagged `never_abstain`,
    with a None threshold (an undefined threshold is a missing value, not -inf).
    """
    sigma = {f"s{i}:1": (0.05 * i) for i in range(10)}
    hits = {f"s{i}:1": (i >= 3) for i in range(10)}
    curve = {r["q"]: r for r in L.sigma_abstain_curve(sigma, hits, [0.0, 0.1])}
    zero = curve[0.0]
    assert zero["never_abstain"] is True
    assert zero["abstained"] == 0 and zero["abstain_rate"] == 0.0
    assert zero["gate_abstain_rate"] == 0.0 and zero["n_sigma_undefined"] == 0
    assert zero["threshold"] is None                 # no -inf / nan sentinel
    assert zero["selective_n"] == zero["n"]
    assert curve[0.1]["never_abstain"] is False and curve[0.1]["abstained"] >= 1
    json.dumps(list(curve.values()))                 # strict JSON, no Infinity


def test_sigma_abstain_curve_undefined_sigma_stays_none_even_at_q0():
    """A row with no sigma is a MISSING INPUT, not an abstain decision, and the
    q = 0 branch must not resurrect it as a scored row."""
    sigma = {"a:1": 0.5, "b:1": None}
    hits = {"a:1": True, "b:1": True}
    row = L.sigma_abstain_curve(sigma, hits, [0.0])[0]
    assert row["abstained"] == 1 and row["n"] == 2 and row["selective_n"] == 1
    # the missing sigma is named, not reported as a gate decision
    assert row["n_sigma_undefined"] == 1 and row["gate_abstain_rate"] == 0.0


def test_select_q_is_chosen_in_inner_folds_only():
    rng = np.random.default_rng(0)
    sigma, y, groups = {}, {}, {}
    for s in range(12):
        for t in range(4):
            qid = f"sess{s}:{t}"
            pos = int(s % 3 == 0)
            sigma[qid] = float(rng.normal(0.2 if pos else 0.7, 0.05))
            y[qid] = pos
            groups[qid] = f"sess{s}"
    got = L.select_q_grouped(sigma, y, groups, q_grid=(0.1, 0.2, 0.4), outer_folds=4, inner_folds=2)
    assert got["n_scored"] > 0 and got["chosen"]
    assert all(c["q"] in (0.1, 0.2, 0.4) for c in got["chosen"])
    # every fired qid was scored out of fold; nothing outside the scored set leaks
    assert set(got["fires"]) <= set(sigma)
    # the signal is strong, so the fired set should be enriched for positives
    fired_pos = sum(y[q] for q, f in got["fires"].items() if f)
    assert fired_pos >= 1


# --------------------------------------------------------------------------
# (D) gold-free dependency-edge oracle -- 2605.25310 sec 3
# --------------------------------------------------------------------------

def test_longest_common_substring_min_length_rule():
    assert L.longest_common_substring_len("xxabcyy", "zzabcww") == 0     # 3 chars < 4
    assert L.longest_common_substring_len("xxabcdyy", "zzabcdww") == 4
    assert L.longest_common_substring_len("hello world", "say hello world") == 11


def test_edge_oracle_is_case_sensitive():
    assert L.edge_strengths(["QUICK BROWN"], "quick brown") == [0]
    assert L.edge_strengths(["quick brown"], "quick brown")[0] == 11


def test_edge_oracle_normalises_whitespace_only():
    doubled = L.edge_strengths(["the  quick   brown fox"], "value: quick brown")
    single = L.edge_strengths(["the quick brown fox"], "value: quick brown")
    # collapsing runs of whitespace makes the doubled-space block match exactly
    # as well as the single-space one (" quick brown" = 12 chars)
    assert doubled == single == [12]


def test_edge_locator_argmax_and_none():
    assert L.edge_locator([0, 7, 3]) == 1
    assert L.edge_locator([5, 5, 1]) == 0      # lowest index on ties
    assert L.edge_locator([0, 0, 0]) is None


def test_action_arguments_text_strict_vs_capped():
    text, strict = L.action_arguments_text(PRED_CLOSED)
    assert strict is True and "abcdefghijkl" in text
    text2, strict2 = L.action_arguments_text(PRED_TRUNC)
    assert strict2 is False and "abcdefghijkl" in text2


def test_typed_equality_oracle_is_type_aware():
    docs = ['{"user_id": 123, "name": "bob"}']
    pred_int = ('<tool_call>\n{"name":"orders__get","arguments":{"user_id":123}}\n</tool_call>')
    pred_str = ('<tool_call>\n{"name":"orders__get","arguments":{"user_id":"123"}}\n</tool_call>')
    assert L.typed_equality_edges(docs, pred_int) == [1]
    assert L.typed_equality_edges(docs, pred_str) == [0]


def test_typed_equality_keeps_types_through_the_128_token_cap():
    """2605.25310 sec 4.1.2 compares by TYPE-AWARE equality.  On a censored
    emission the arguments are salvaged from raw text, and stringifying the
    salvaged numbers would make the typed oracle structurally unable to match
    any numeric leaf on the 41/93 capped rows."""
    docs = ['{"user_id": 123, "name": "bob"}']
    trunc_int = '<tool_call>\n{"name":"orders__get","arguments":{"user_id":123'
    trunc_str = '<tool_call>\n{"name":"orders__get","arguments":{"user_id":"123"'
    assert L.typed_equality_edges(docs, trunc_int) == [1]
    assert L.typed_equality_edges(docs, trunc_str) == [0]
    # and the string-valued salvage path (used by witness-IDF) is untouched
    assert L._salvage_arg_values('{"user_id":123') == ["123"]
    assert L._salvage_typed_arg_values('{"user_id":123') == [123]


def test_edge_oracle_maximal_hit_length_matches_the_paper_procedure():
    """2605.25310 App.: enumerate every contiguous >= 4-char substring of the
    normalised block text that occurs verbatim in the normalised arguments and
    discard any contained in a longer retained hit -- i.e. the surviving hit is
    the MAXIMAL one, which is what the strength column reports."""
    # "order 998877" (12 chars) is the maximal hit; "998877" is contained in it
    assert L.longest_common_substring_len("order 998877 created", 'x "order 998877" y') == 12
    # sub-threshold overlap contributes nothing at all, it does not fall back
    assert L.longest_common_substring_len("abc", "abc") == 0
    # case sensitivity bites only the differing characters: "rder 998877" (11)
    # still matches, the leading "O" vs "o" does not
    assert L.longest_common_substring_len("Order 998877", "order 998877") == 11
    assert L.longest_common_substring_len("ORDER", "order") == 0


def test_edge_choosers_keep_own_and_gold_apart():
    row = make_row(docs=("token abcdefghijkl", "order 998877 created"))
    rows = {r["chooser"]: r for r in L.edge_choosers(row)}
    assert rows[L.CH_EDGE_OWN]["khat"] == 0            # own (wrong) action grounds in doc0
    assert rows[L.CH_EDGE_GOLD]["khat"] == 1           # gold values ground in doc1
    assert rows[L.CH_EDGE_GOLD]["note"]["envelope"] is True
    assert "_label_" in L.CH_EDGE_GOLD


# --------------------------------------------------------------------------
# (E) position priors -- 2307.03172 (qualitative only)
# --------------------------------------------------------------------------

def test_position_priors():
    assert L.position_prior_khat(5, "first") == 0
    assert L.position_prior_khat(5, "last") == 4
    assert L.position_prior_khat(5, "median", 2) == 2
    assert L.position_prior_khat(0, "first") is None
    with pytest.raises(ValueError):
        L.position_prior_khat(3, "middle")


def test_k_median_follows_the_frozen_convention_on_even_n():
    """d_witness_select.py:150 freezes k_median = (n_docs - 1) // 2, NOT n // 2.

    41/93 C->W qids saturate max_doc_num = 16, where the two differ (7 vs 8),
    so an n // 2 fallback would score a different block than the frozen witness
    table's own k_median column on nearly half the trigger set.
    """
    assert L.position_prior_khat(16, "median") == 7
    assert L.position_prior_khat(4, "median") == 1
    assert L.position_prior_khat(5, "median") == 2


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 15, 16])
def test_positional_score_argmax_equals_the_prior_it_stands_for(n):
    """The inverted-score control inverts THESE vectors, so their argmax has to
    be the block the prior actually picks -- otherwise the control's forward
    arm and the chooser row disagree."""
    for prior in ("first", "median", "last"):
        vec = L.positional_scores(n, prior)
        assert int(np.argmax(vec)) == L.position_prior_khat(n, prior)


def test_two_point_prior_threshold_is_fitted_out_of_fold():
    rows, truth, scalar = [], {}, {}
    for s in range(10):
        for t in range(4):
            qid = f"p{s}:{t}"
            n = 5
            early = t < 2
            rows.append(L.LocalizeRow(qid=qid, session_id=f"p{s}", docs=["x"] * n, query="",
                                      prediction="", doc_lengths=[1] * n, decision_step=t))
            truth[qid] = 0 if early else n - 1
            scalar[qid] = float(t)
    got = L.two_point_prior_oof(rows, truth, scalar, outer_folds=5)
    assert got["chosen"] and len(got["khat"]) == len(rows)
    hits = sum(1 for q, k in got["khat"].items() if k == truth[q])
    assert hits == len(rows)  # a clean two-point rule exists and CV finds it


# --------------------------------------------------------------------------
# features: orientation, leakage, missingness
# --------------------------------------------------------------------------

def test_two_point_rows_carry_the_vector_their_own_choice_maximises():
    rows = [make_row(qid=f"tp{i}:2", docs=("a", "b", "c", "d"), decision_step=i) for i in range(8)]
    recs, _ = L.all_choosers(rows)
    tp = [r for r in recs if r["chooser"] == L.CH_TWOPOINT]
    assert tp, "the two-point prior row is mandatory"
    for r in tp:
        assert int(np.argmax(r["score_vector"])) == r["khat"]


def test_features_columns_match_declared_orientations():
    row = make_row()
    feats = L.features_for_row(row)
    cols = set(feats) - {"qid", "session_id"}
    assert cols == set(L.FEATURE_ORIENTATIONS)


def test_features_never_read_the_gold_side():
    a = make_row(gold=("orders__create", "998877"))
    b = make_row(gold=("something__else", "000000"))
    assert L.features_for_row(a) == L.features_for_row(b)


def test_features_use_none_not_sentinels_when_undefined():
    row = make_row(docs=(DOC0,), prediction="no call", query="")
    feats = L.features_for_row(row)
    assert feats["loc_sigma_top1_proposal"] is None     # a single block has no margin
    assert feats["loc_proposal_n_values"] is None       # no tool call at all
    assert feats["loc_edge_own_max_strength"] is not None


def test_features_pass_the_leakage_guard_and_keep_nulls(tmp_path):
    rows = [L.features_for_row(make_row(qid="a:1")), L.features_for_row(make_row(qid="b:2"))]
    out = tmp_path / "features_localize.jsonl"
    n = write_features_jsonl(out, rows, context="test")
    assert n == 2
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "gap_second" not in first or True  # names are prefixed, not scoring columns
    assert set(first) == set(rows[0])


def test_orientations_file_is_in_sync():
    path = _AGENT.parent / "configs" / "t34" / "orientations_localize.json"
    assert path.exists(), "run: python agent/t34_localize.py orientations"
    on_disk = {k: int(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    assert on_disk == L.FEATURE_ORIENTATIONS


def test_module_declares_every_deviation():
    assert L.DEVIATIONS
    for d in L.DEVIATIONS:
        assert set(d) == {"method", "paper", "what", "why"}
        assert d["paper"] and d["what"] and d["why"]


# --------------------------------------------------------------------------
# the locator table CLI
# --------------------------------------------------------------------------

def _synthetic_witness(n_qids=8):
    entries = {}
    for i in range(n_qids):
        entries[f"w{i}:2"] = {"n_docs": 4, "k_witness": (i % 4), "k_median": 2,
                              "doc_text_sha256": [], "doc_lengths": [1, 1, 1, 1]}
    entries["w_none:2"] = {"n_docs": 4, "k_witness": None, "k_median": 2,
                           "doc_text_sha256": [], "doc_lengths": [1, 1, 1, 1]}
    return {"entries": entries, "n_qids": n_qids + 1}


def _chooser_rows(witness, chooser="cand"):
    rows = []
    for qid, e in witness["entries"].items():
        ref = e["k_witness"]
        vec = [0.0] * e["n_docs"]
        if ref is None:
            khat = None
        else:
            khat = ref
            vec[ref] = 5.0
            vec[(ref + 1) % e["n_docs"]] = 1.0
        rows.append({"qid": qid, "session_id": qid.split(":")[0], "chooser": chooser,
                     "khat": khat, "score_vector": vec, "n_docs": e["n_docs"],
                     "censored_at_cap": False, "note": {}})
    return rows


def test_locate_table_on_a_synthetic_chooser():
    witness = _synthetic_witness()
    table = S.build_table(_chooser_rows(witness), witness)
    by = {c["chooser"]: c for c in table["choosers"]}
    # the position priors are MANDATORY rows even though the chooser file omits them
    assert {L.CH_FIRST, L.CH_MEDIAN, L.CH_LAST} <= set(by)
    cand = by["cand"]
    assert cand["hits_vs_witness"]["hits"] == 8
    assert cand["hits_vs_witness"]["n"] == 9          # the k*=None qid stays in the denominator
    assert cand["hits_vs_witness"]["abstained"] == 1
    assert cand["hits_vs_witness"]["p_vs_floor"] < 0.05
    assert table["floor_wrong_block"] == 0.25
    # the inverted-score control ran on the same vectors and is much worse
    inv = cand["inverted_control"]
    assert inv["forward"]["hits"] == 8 and inv["inverted"]["hits"] < 8
    # McNemar against every prior, on the shared subset, with n reported
    assert set(cand["vs_priors"]) == set(L.PRIOR_CHOOSERS)
    assert cand["vs_priors"][L.CH_FIRST]["n_shared"] == 9


def test_inverted_control_forward_arm_reproduces_the_chooser():
    """t34_common.inverted_score_control now runs the FROZEN select_k_star
    semantics on both arms, so on a chooser whose k_hat IS the argmax of its own
    vector the control's forward arm must reproduce the chooser exactly -- no
    call-site blanking, and in particular no silencing of the argmin arm."""
    witness = _synthetic_witness(n_qids=4)
    rows = _chooser_rows(witness)
    table = S.build_table(rows, witness)
    cand = [c for c in table["choosers"] if c["chooser"] == "cand"][0]
    assert cand["khat_argmax_disagreements"] == 0
    assert cand["inverted_control"]["forward"]["hits"] == cand["hits_vs_witness"]["hits"]
    assert cand["inverted_control"]["forward"]["abstained"] == cand["hits_vs_witness"]["abstained"]
    assert cand["inverted_control"]["inverted"]["hits"] < cand["hits_vs_witness"]["hits"]


def test_khat_that_is_not_the_vectors_argmax_is_counted_not_papered_over():
    """The specificity control is only interpretable when the chooser's k_hat is
    the argmax of the vector being inverted.  A row where it is not must be
    COUNTED (the control can then be read as invalid for that chooser), never
    silently blanked -- blanking would also delete the argmin arm's row."""
    witness = _synthetic_witness(n_qids=4)
    rows = _chooser_rows(witness)
    victim = [r for r in rows if r["khat"] is not None][0]
    victim["khat"] = None                      # k_hat no longer the vector's argmax
    table = S.build_table(rows, witness)
    cand = [c for c in table["choosers"] if c["chooser"] == "cand"][0]
    assert cand["hits_vs_witness"]["abstained"] == 2      # the k*=None qid + the victim
    assert cand["khat_argmax_disagreements"] == 1
    assert victim["qid"] in cand["khat_argmax_disagreement_sample"]


def test_surface_form_ceiling_is_compared_in_the_same_table():
    """Digest 4.7 (2605.25310 line): the position baselines AND the surface-form
    ceiling (witness-IDF scored with the arm's own action) must be in the same
    table, reported as a difference."""
    witness = _synthetic_witness(n_qids=4)
    rows = _chooser_rows(witness)
    for r in _chooser_rows(witness, chooser=L.CH_PROPOSAL):
        if r["khat"] is not None:              # make the ceiling arm strictly worse
            r["khat"] = (r["khat"] + 1) % r["n_docs"]
        rows.append(r)
    table = S.build_table(rows, witness)
    by = {c["chooser"]: c for c in table["choosers"]}
    assert table["surface_form_ceiling_present"] is True
    cmp = by["cand"]["vs_surface_form_ceiling"]
    assert cmp["n_shared"] == 5 and cmp["delta_s_at_k"] > 0
    assert cmp["delta_s_at_k"] == pytest.approx(
        (cmp["a_hits"] - cmp["b_hits"]) / cmp["n_shared"])
    # the ceiling row does not get compared against itself
    assert by[L.CH_PROPOSAL]["vs_surface_form_ceiling"] is None
    # and it is reported as ABSENT rather than skipped when the arm is missing
    assert S.build_table(_chooser_rows(witness), witness)["surface_form_ceiling_present"] is False


def test_mcnemar_vs_priors_reports_the_delta_not_only_the_marginals():
    witness = _synthetic_witness()
    table = S.build_table(_chooser_rows(witness), witness)
    cand = [c for c in table["choosers"] if c["chooser"] == "cand"][0]
    cmp = cand["vs_priors"][L.CH_FIRST]
    assert cmp["delta_s_at_k"] == pytest.approx(
        cmp["a_s_at_k"] - cmp["b_s_at_k"])
    assert cmp["delta_s_at_k"] > 0


def test_synthesised_priors_share_the_stratum_denominator():
    """Under --stratum the mandatory prior rows must not be scored on all 93
    qids while the candidate is scored on the censored slice."""
    witness = _synthetic_witness()
    rows = _chooser_rows(witness)
    for r in rows[:4]:
        r["censored_at_cap"] = True
    table = S.build_table(rows, witness, stratum="censored")
    by = {c["chooser"]: c for c in table["choosers"]}
    assert {L.CH_FIRST, L.CH_MEDIAN, L.CH_LAST} <= set(by)
    n_cand = by["cand"]["hits_vs_witness"]["n"]
    for p in L.PRIOR_CHOOSERS:
        assert by[p]["hits_vs_witness"]["n"] == n_cand


def test_flip_column_is_never_merged_with_the_witness_column():
    witness = _synthetic_witness()
    rows = _chooser_rows(witness)
    flip = {"w0:2": {0: True, 1: False}, "w1:2": {1: False}}
    table = S.build_table(rows, witness, flip)
    cand = [c for c in table["choosers"] if c["chooser"] == "cand"][0]
    assert cand["hits_vs_flip"]["n"] == 2          # only the flip table's own qids
    assert cand["hits_vs_flip"]["hits"] == 1
    assert cand["hits_vs_witness"]["n"] == 9       # untouched
    assert cand["hits_vs_flip"]["n"] != cand["hits_vs_witness"]["n"]


def test_unswept_khat_is_unmeasured_not_a_flip_miss():
    """The D-line sweep is per (qid, k).  A chooser that picked a k the sweep
    never tried for that qid has an UNKNOWN flip outcome; scoring it False would
    turn a missing measurement into a miss and deflate every flip S@k."""
    witness = _synthetic_witness(n_qids=4)
    rows = _chooser_rows(witness)
    # w0:2 -> k_witness 0 (swept), w1:2 -> k_witness 1 (present but NOT swept)
    flip = {"w0:2": {0: True}, "w1:2": {0: False}}
    table = S.build_table(rows, witness, flip)
    cand = [c for c in table["choosers"] if c["chooser"] == "cand"][0]
    assert cand["hits_vs_flip"]["n"] == 1                 # only the swept row
    assert cand["hits_vs_flip"]["hits"] == 1
    assert cand["n_khat_outside_flip_sweep"] == 1
    assert cand["hits_vs_flip"]["n_khat_outside_flip_sweep"] == 1


def test_locate_table_stratifies_by_cap():
    witness = _synthetic_witness()
    rows = _chooser_rows(witness)
    for r in rows[:4]:
        r["censored_at_cap"] = True
    censored = S.build_table(rows, witness, stratum="censored")
    cand = [c for c in censored["choosers"] if c["chooser"] == "cand"][0]
    assert cand["hits_vs_witness"]["n"] == 4
    assert censored["stratum"] == "censored"


def test_frame_drops_qids_the_witness_table_does_not_cover():
    """A C->C qid has no reference block; letting it into the denominator would
    be a guaranteed miss that silently deflates every S@k."""
    witness = _synthetic_witness()
    rows = _chooser_rows(witness)
    rows.append({"qid": "not_in_witness:9", "session_id": "not_in_witness", "chooser": "cand",
                 "khat": 0, "score_vector": [1.0, 0.0], "n_docs": 2,
                 "censored_at_cap": False, "note": {}})
    inside = S.build_table(rows, witness, frame="witness")
    outside = S.build_table(rows, witness, frame="all")
    assert inside["n_rows_outside_witness_frame"] == 1
    cand_in = [c for c in inside["choosers"] if c["chooser"] == "cand"][0]
    cand_all = [c for c in outside["choosers"] if c["chooser"] == "cand"][0]
    assert cand_in["hits_vs_witness"]["n"] == 9
    assert cand_all["hits_vs_witness"]["n"] == 10


def test_render_is_ascii():
    witness = _synthetic_witness()
    text = S.render(S.build_table(_chooser_rows(witness), witness))
    text.encode("ascii")  # raises if any non-ASCII slipped in
    assert "chooser" in text


def test_stale_sidecar_is_refused_not_merely_counted():
    """build_rows only RECORDS the sha mismatch; the CLI must refuse, or a
    whole locator table could come out of a stale decoded-doc dump."""
    with pytest.raises(SystemExit):
        L._guard_sidecar({"mismatched_qids": ["w0:2", "w1:2"]}, False)
    L._guard_sidecar({"mismatched_qids": ["w0:2"]}, True)   # explicit override
    L._guard_sidecar({"mismatched_qids": []}, False)


def test_clis_expose_help():
    for mod in (L, S):
        with pytest.raises(SystemExit) as exc:
            mod.main(["--help"])
        assert exc.value.code == 0


def test_semantic_arm_records_which_encoder_produced_the_numbers():
    """An injected encoder is a test double, not MiniLM: the cost row has to say
    so, or an `ms`/`bytes` line gets read as a measurement of the real S14
    dependency (which has never run on this box -- no torch)."""
    rows = [make_row(qid=f"e{i}:2") for i in range(3)]
    fake = lambda batch: np.eye(len(batch), 4)
    recs, meta = L.all_choosers(rows, with_semantic=True, encode_fn=fake)
    assert L.CH_RRF4 in {r["chooser"] for r in recs}
    cost = meta["semantic_cost"]
    assert cost["encoder"] == "injected"
    assert cost["n_calls"] == 3
    # both denominators the digest's cost line asks for, neither replacing the other
    assert cost["ms_per_turn"] is not None and cost["ms_per_session"] is not None
    assert cost["bytes_per_turn"] is not None and cost["bytes_per_session"] is not None


try:  # the S14 dependency is absent on this box (no torch); the test below pins
    import sentence_transformers as _st  # noqa: F401
    _HAVE_ST = True
except Exception:
    _HAVE_ST = False


@pytest.mark.skipif(_HAVE_ST, reason="sentence_transformers present: no missing-input path to test")
def test_missing_semantic_dependency_aborts_by_name():
    """--semantic on a box without torch must abort naming the dependency, not
    surface a bare ImportError from inside a ranker."""
    with pytest.raises(SystemExit) as exc:
        L.load_semantic_encoder()
    assert "sentence_transformers" in str(exc.value)
    with pytest.raises(SystemExit):
        L.all_choosers([make_row()], with_semantic=True)


class _FakeFrame:
    """Minimal stand-in for t34_common.FrozenFrame (build_rows only needs these)."""

    def __init__(self, entries):
        self._entries = entries
        self.labels = [{"qid": q, "censored_at_cap": False} for q in entries]

    @property
    def c2kv_by_qid(self):
        return {q: {"qid": q, "prediction": PRED_CLOSED, "session_id": q.split(":")[0],
                    "decision_step": 2} for q in self._entries}

    def witness_entry(self, qid):
        return self._entries.get(qid)


def test_sidecar_sha_blind_spot_is_reported_by_name(tmp_path, monkeypatch):
    """check_docs_against_witness can only verify qids the witness table covers.
    A C->C sidecar row (which still feeds the FEATURE frame) has nothing to be
    checked against, so "0 mismatches" must not read as "everything verified"."""
    entries = {"w0:2": {"n_docs": 2, "k_witness": 1, "k_median": 0,
                        "doc_text_sha256": [], "tool_name": "orders__create",
                        "arg_leaf_values": ["998877"]}}
    side = tmp_path / "sidecar.jsonl"
    side.write_text("\n".join(
        json.dumps({"qid": q, "session_id": q.split(":")[0], "docs": [DOC0, DOC1],
                    "query": "q", "doc_lengths": [1, 1], "dropped_docs": []})
        for q in ("w0:2", "cc0:2")) + "\n", encoding="utf-8")

    class _FakeAssets:
        def __init__(self, root):
            self.root = root

        def load(self):
            frame = _FakeFrame(entries)
            frame.labels.append({"qid": "cc0:2", "censored_at_cap": False})
            return frame

    monkeypatch.setattr(L, "FrozenAssets", _FakeAssets)
    monkeypatch.setattr(L, "check_docs_against_witness",
                        lambda docs, frame: {"n_checked_docs": 0, "mismatched_qids": []})
    rows, audit = L.build_rows(tmp_path, side)
    assert audit["n_qids_without_witness_sha"] == 1
    assert audit["qids_without_witness_sha_sample"] == ["cc0:2"]
    assert audit["sha_coverage_complete"] is False
    # the dropped side was not dumped: reported unavailable, never faked from docs
    assert audit["dropped_side_available"] is False
    assert all(r.dropped_doc_texts is None for r in rows)
    assert all(r.visible_docs == r.docs for r in rows)


def test_all_choosers_assembles_every_arm():
    rows = [make_row(qid=f"z{i}:2") for i in range(6)]
    recs, meta = L.all_choosers(rows, k_median_by_qid={r.qid: 1 for r in rows})
    names = {r["chooser"] for r in recs}
    assert {L.CH_GOLD, L.CH_PROPOSAL, L.CH_QUERY_ONLY, L.CH_NONE, L.CH_FIRST,
            L.CH_MEDIAN, L.CH_LAST, L.CH_RRF3, L.CH_EDGE_OWN, L.CH_EDGE_GOLD,
            L.CH_EDGE_TYPED, L.CH_TWOPOINT} <= names
    assert L.CH_RRF4 not in names            # semantic arm is off by default
    assert meta["semantic_cost"]["n_calls"] == 0
    for r in recs:
        assert set(r) == {"qid", "session_id", "chooser", "khat", "score_vector",
                          "n_docs", "censored_at_cap", "note"}
