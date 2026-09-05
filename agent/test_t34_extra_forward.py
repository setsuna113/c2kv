# -*- coding: utf-8 -*-
"""Tests for t34 unit U5: t34_extra_forward.py (digest 4.8) and t34_contextcite.py
(digest 4.7).  Everything here runs CPU-only; the torch paths are importorskip'd and
their reduction logic is exercised with synthetic inputs.

Run from the worktree root:
  PYTHONIOENCODING=utf-8 python -m pytest agent/test_t34_extra_forward.py -q
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import t34_common as C
import t34_contextcite as CC
import t34_extra_forward as X
from t33_spanmap import parse_tool_call, spans_from_generation

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers: a tokenizer-free "decode" so spans_from_generation works on CPU
# ---------------------------------------------------------------------------

VOCAB = {}


def _mk_ids(text: str):
    """One token per character, ids stable across the test module."""
    ids = []
    for ch in text:
        VOCAB.setdefault(ch, len(VOCAB) + 1000)
        ids.append(VOCAB[ch])
    return ids


def _decode(ids):
    inv = {v: k for k, v in VOCAB.items()}
    return "".join(inv[i] for i in ids)


CALL_TEXT = 'Action:\n<tool_call>\n{"name":"venmo__search","arguments":{"q":"alice"}}\n</tool_call>'


# ===========================================================================
# (A) VeriCache label pass
# ===========================================================================

def test_vericache_no_divergence_reports_none_and_cap_hit_separately():
    ids = _mk_ids(CALL_TEXT)
    spans = spans_from_generation(_decode, ids)
    row = X.vericache_label_row("s:1", ids, ids, spans, cap_tokens=len(ids))
    assert row["first_div_idx"] is None
    assert row["accept_len"] == len(ids)
    assert row["div_region"] == "none"
    assert row["cap_hit"] is True                      # emitted alongside, never merged
    assert row["div_bucket"] == "after_cap"
    row2 = X.vericache_label_row("s:1", ids, ids, spans, cap_tokens=len(ids) + 5)
    assert row2["cap_hit"] is False and row2["div_bucket"] == "no_divergence"


@pytest.mark.parametrize("region", ["tool_name", "arguments", "closing", "preamble"])
def test_vericache_div_region_uses_the_real_span_map(region):
    ids = _mk_ids(CALL_TEXT)
    spans = spans_from_generation(_decode, ids)
    pick = {
        "tool_name": spans["name_first"] + 1,
        "arguments": spans["args_first"] + 2,
        "preamble": 1,
        "closing": spans["payload_last"] + 2,
    }[region]
    star = list(ids)
    star[pick] = 999999
    row = X.vericache_label_row("s:1", ids, star, spans, cap_tokens=1000)
    assert row["first_div_idx"] == pick
    assert row["accept_len"] == pick
    assert row["div_region"] == region


def test_vericache_unparsed_row_is_not_forced_into_a_span():
    ids = _mk_ids("no call at all")
    spans = spans_from_generation(_decode, ids)
    star = list(ids)
    star[2] = 7
    row = X.vericache_label_row("s:2", ids, star, spans, cap_tokens=1000)
    assert row["div_region"] == "unparsed"
    assert row["div_bucket"] == "other"


def test_vericache_bonus_token_and_compare_length():
    ids = _mk_ids("abc")
    star = ids + [42]
    spans = spans_from_generation(_decode, ids)
    row = X.vericache_label_row("s:3", ids, star, spans, cap_tokens=1000)
    assert row["n_compared"] == 3 and row["bonus_token_id"] == 42


def test_vericache_disagreement_and_census():
    a = [{"qid": "s:1", "first_div_idx": 5, "div_bucket": "name_span", "cap_hit": False,
          "accept_len": 5},
         {"qid": "s:2", "first_div_idx": None, "div_bucket": "no_divergence",
          "cap_hit": False, "accept_len": 10}]
    b = [{"qid": "s:1", "first_div_idx": 6, "div_bucket": "name_span", "cap_hit": False,
          "accept_len": 6},
         {"qid": "s:2", "first_div_idx": None, "div_bucket": "no_divergence",
          "cap_hit": False, "accept_len": 10}]
    rep = X.vericache_label_disagreement(a, b)
    assert rep["n_common"] == 2
    assert rep["index_disagree_rate"] == 0.5
    assert rep["bucket_disagree_rate"] == 0.0        # the documented downgrade target
    assert rep["downgrade_to_bucket"] is True
    assert rep["bf16_shape_rounding_maxabs"] == 0.0078125
    cen = X.vericache_label_census(a)
    assert cen["n"] == 2 and cen["n_no_divergence"] == 1
    assert cen["div_bucket"]["name_span"] == 1


def test_emitted_ids_prefers_capture_and_flags_the_retokenise_caveat():
    cap = {"s:1": {"qid": "s:1", "generated_ids": [1, 2, 3]}}
    ids, src = X.emitted_ids_for_row("s:1", cap, "ignored", lambda t: [9])
    assert (ids, src) == ([1, 2, 3], "capture")
    ids, src = X.emitted_ids_for_row("s:9", cap, "hello", lambda t: [7, 7])
    assert (ids, src) == ([7, 7], "retokenized")
    ids, src = X.emitted_ids_for_row("s:9", cap, "hello", None)
    assert src == "unavailable"


def test_retokenised_ids_carry_a_local_roundtrip_signal():
    """A shifted id sequence shifts first_div_idx.  With a decoder available the
    roundtrip is CHECKED here rather than assumed, so a lossy fallback is visible
    without a real Qwen tokenizer."""
    enc = lambda t: [ord(c) for c in t]
    ok_dec = lambda ids: "".join(chr(i) for i in ids)
    bad_dec = lambda ids: "".join(chr(i) for i in ids) + "!"
    assert X.emitted_ids_for_row("q", {}, "abc", enc)[1] == "retokenized"
    assert X.emitted_ids_for_row("q", {}, "abc", enc, decode_fn=ok_dec)[1] == \
        "retokenized_checked"
    assert X.emitted_ids_for_row("q", {}, "abc", enc, decode_fn=bad_dec)[1] == \
        "retokenized_lossy"
    assert X.emitted_ids_for_row("q", {}, "abc", None)[1] == "unavailable"


def test_census_makes_a_mixed_id_source_visible():
    rows = [
        {"qid": "s:1", "first_div_idx": None, "cap_hit": False, "accept_len": 3,
         "div_region": "none", "div_bucket": "no_divergence", "ids_source": "capture"},
        {"qid": "s:2", "first_div_idx": 1, "cap_hit": False, "accept_len": 1,
         "div_region": "tool_name", "div_bucket": "name_span",
         "ids_source": "retokenized_lossy"},
    ]
    cen = X.vericache_label_census(rows)
    assert cen["by_ids_source"] == {"capture": 1, "retokenized_lossy": 1}
    assert cen["mixed_ids_sources"] is True
    assert cen["n_retokenized_lossy"] == 1
    single = X.vericache_label_census(rows[:1])
    assert single["mixed_ids_sources"] is False


def test_vericache_labels_never_reach_a_feature_frame(tmp_path):
    """The VeriCache columns are labels; the guard must reject a naive attempt to
    write the c2kv arm's tool_name_match beside them (historical leak specimen)."""
    with pytest.raises(ValueError):
        C.write_features_jsonl(tmp_path / "f.jsonl",
                               [{"qid": "s:1", "accept_len": 3, "tool_name_match": 1}],
                               context="leak probe")
    with pytest.raises(ValueError):
        C.write_features_jsonl(tmp_path / "f.jsonl",
                               [{"qid": "s:1", "a_made_call": 1}], context="leak probe")


def test_forward_argmax_needs_torch():
    pytest.importorskip("torch")
    assert callable(X._forward_argmax)


# ===========================================================================
# (B) AsymSpec
# ===========================================================================

def test_jsd_zero_for_identical_and_ln2_for_disjoint():
    p = np.array([[0.2, 0.3, 0.5]])
    assert X.jensen_shannon_nats(p, p)[0] == pytest.approx(0.0, abs=1e-12)
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    assert X.jensen_shannon_nats(a, b)[0] == pytest.approx(math.log(2), abs=1e-12)


def test_jsd_bounded_by_ln2_on_random_logits():
    rng = np.random.default_rng(3)
    a = rng.normal(size=(40, 17)) * 8
    b = rng.normal(size=(40, 17)) * 8
    st = X.asymspec_position_stats(a, b)
    assert st["jsd"].shape == (40,)
    assert np.all(st["jsd"] >= -1e-12)
    assert np.all(st["jsd"] <= X.JSD_MAX_NATS + 1e-12)
    assert np.allclose(st["delta_l1"], np.abs(a - b).sum(axis=-1))


def test_asymspec_position_stats_refuses_shape_mismatch():
    with pytest.raises(ValueError):
        X.asymspec_position_stats(np.zeros((3, 4)), np.zeros((3, 5)))


def test_asymspec_aggregate_name_features_are_none_without_a_name_span():
    st = {"jsd": np.array([0.1, 0.4, 0.2]), "delta_l1": np.array([1.0, 2.0, 3.0])}
    agg = X.asymspec_aggregate(st, name_positions=None)
    assert agg["asym_jsd_max"] == pytest.approx(0.4)
    assert agg["asym_jsd_first"] == pytest.approx(0.1)
    assert agg["asym_jsd_name_max"] is None          # no sentinel, no whole-span fallback
    assert agg["asym_n_name_positions"] == 0
    agg2 = X.asymspec_aggregate(st, name_positions=[1, 2])
    assert agg2["asym_jsd_name_max"] == pytest.approx(0.4)
    assert agg2["asym_jsd_name_mean"] == pytest.approx(0.3)


def test_split_history_blocks_uses_the_post_split_index_base():
    # ``docs`` holds the KEPT rows only; ``dropped_docs`` indexes the post-split list.
    side = {"qid": "s:1", "docs": ["A", "C"], "dropped_docs": [1],
            "dropped_doc_texts": ["B"]}
    lay = X.split_history_blocks(side)
    assert lay["n_split_blocks"] == 3
    assert lay["kept_positions"] == [0, 2]
    assert lay["blocks"] == ["A", "B", "C"]          # original order restored
    assert lay["kept_texts"] == ["A", "C"]           # no kept block deleted
    assert lay["dropped_text_available"] is True


def test_split_history_blocks_flags_missing_dropped_text():
    lay = X.split_history_blocks({"docs": ["A", "C"], "dropped_docs": [1]})
    assert lay["dropped_text_available"] is False
    assert lay["blocks"] == ["A", None, "C"]


def test_split_history_blocks_refuses_an_out_of_range_index():
    with pytest.raises(ValueError):
        X.split_history_blocks({"docs": ["A"], "dropped_docs": [7]})


def test_build_text_views_drops_and_truncates_per_s8_fields():
    # kept rows: "alpha beta gamma" and "eps zeta eta theta"; the dropped block sits
    # between them in the post-split list and its text comes from dropped_doc_texts.
    side = {"qid": "s:1", "docs": ["alpha beta gamma", "eps zeta eta theta"],
            "query": "QQ", "system_prompt": "SYS", "dropped_docs": [1],
            "dropped_doc_texts": ["delta"], "kept_history_tokens": 4}
    v = X.build_text_views(side)
    assert "delta" not in v["x_comp"] and "delta" in v["x_full"]
    # the dropped block is restored in POSITION, between the two kept ones
    assert v["x_full"].index("alpha") < v["x_full"].index("delta") < v["x_full"].index("eps")
    # and no KEPT block is deleted from x_comp by the dropped index
    assert "alpha" in v["x_comp"] and "eps" in v["x_comp"]
    assert v["n_kept_docs"] == 2 and v["n_dropped_docs"] == 1 and v["n_docs"] == 3
    assert v["per_doc_token_budget"] == 2
    assert v["x_comp"].startswith("SYS") and v["x_comp"].endswith("QQ")
    assert "alpha beta" in v["x_comp"] and "gamma" not in v["x_comp"]
    assert v["proxy"] == X.PROXY_STAMP
    assert v["truncation_mode"] == "whitespace"
    assert v["view_complete"] is True


def test_build_text_views_marks_the_row_unusable_without_dropped_text():
    side = {"qid": "s:1", "docs": ["alpha", "beta"], "query": "QQ",
            "system_prompt": "SYS", "dropped_docs": [1], "kept_history_tokens": 2}
    v = X.build_text_views(side)
    assert v["view_complete"] is False
    # both kept blocks survive: the dropped index must not delete one of them
    assert "alpha" in v["x_comp"] and "beta" in v["x_comp"]


def test_missing_dropped_text_degrades_the_row_instead_of_deleting_it():
    """The default U2 dump has no dropped_doc_texts, so SKIPPING every incomplete row
    would delete every row that HAS a drop -- the whole informative subset.  The row is
    scored on the narrower DECLARED axis (truncation) and stamped with its variant."""
    side = {"qid": "s:1", "docs": ["alpha beta gamma", "eps zeta eta"], "query": "QQ",
            "system_prompt": "SYS", "dropped_docs": [1], "kept_history_tokens": 4}
    v = X.build_text_views(side)
    assert v["view_complete"] is False
    assert v["proxy_variant"] == "truncation_only_missing_dropped_text"
    # x_full = the VISIBLE blocks at FULL length; x_comp = the same blocks truncated,
    # so the two views still differ on a real, declared axis.
    assert "gamma" in v["x_full"] and "gamma" not in v["x_comp"]
    assert "alpha" in v["x_comp"] and "eps" in v["x_comp"]
    assert v["x_full"] != v["x_comp"]


def test_proxy_variant_separates_the_three_axis_cases():
    """The three variants carry DIFFERENT estimands and must be stratified, never
    pooled: both axes / no drop at all / drop axis missing."""
    both = X.build_text_views({"docs": ["a b c", "d e f"], "query": "q",
                               "dropped_docs": [1], "dropped_doc_texts": ["x y"],
                               "kept_history_tokens": 4})
    assert both["proxy_variant"] == "drop_and_truncation" and both["view_complete"]
    nodrop = X.build_text_views({"docs": ["a b c"], "query": "q", "dropped_docs": [],
                                 "kept_history_tokens": 2})
    assert nodrop["proxy_variant"] == "truncation_only_no_drop"
    assert nodrop["view_complete"] is True
    degraded = X.build_text_views({"docs": ["a b c"], "query": "q", "dropped_docs": [1],
                                   "kept_history_tokens": 2})
    assert degraded["proxy_variant"] == "truncation_only_missing_dropped_text"


def test_truncate_to_tokens_with_an_injected_tokenizer():
    enc = lambda t: list(t)
    dec = lambda ids: "".join(ids)
    assert X.truncate_to_tokens("abcdef", 3, enc, dec) == "abc"
    assert X.truncate_to_tokens("ab", 9, enc, dec) == "ab"
    assert X.truncate_to_tokens("ab", 0, enc, dec) == ""


class _FakeTok:
    def __init__(self, vocab_size, table=None):
        self.vocab_size = vocab_size
        self._table = table or {}

    def encode(self, text, **kw):
        return self._table.get(text, [len(text)])


def test_assert_drafter_family_refuses_self_and_family_mismatch():
    t4 = _FakeTok(151936)
    with pytest.raises(ValueError, match="killed S2"):
        X.assert_drafter_family(t4, t4, "same/model", "same/model")
    with pytest.raises(ValueError, match="vocab"):
        X.assert_drafter_family(_FakeTok(32000), t4, "d", "t")
    X.assert_drafter_family(_FakeTok(151936), t4, "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B")


def test_score_span_logits_needs_torch():
    pytest.importorskip("torch")
    assert callable(X._score_span_logits)


def test_name_positions_come_from_the_parse_not_a_prefix_heuristic():
    parsed = parse_tool_call(CALL_TEXT)
    cs, _ = parsed["payload_span"]
    pos = X._name_positions_in_span(CALL_TEXT, parsed, lambda t: list(t), cs)
    ns = parsed["name_span"]
    assert pos[0] == ns[0] - cs
    assert len(pos) == ns[1] - ns[0]


# ===========================================================================
# (C) paired-config spread
# ===========================================================================

def test_config_pair_is_deterministic_and_a_real_permutation():
    a = X.config_pair_for_qid("sess:3", 5, "doc_permute")
    b = X.config_pair_for_qid("sess:3", 5, "doc_permute")
    assert a == b
    assert sorted(a["config_b"]["doc_order"]) == list(range(5))
    assert a["resampling"] is False
    c = X.config_pair_for_qid("other:3", 5, "doc_permute")
    assert c["seed"] != a["seed"]
    with pytest.raises(ValueError):
        X.config_pair_for_qid("s:1", 4, "hybrid_top_k", (3, 3))
    with pytest.raises(ValueError):
        X.config_pair_for_qid("s:1", 4, "nonsense")


def test_forbidden_doc_packing_pair_is_refused():
    a, b = X.FORBIDDEN_PACKING_PAIR
    with pytest.raises(ValueError, match="doc packing"):
        X.assert_config_pair_legal(dict(a), dict(b))
    X.assert_config_pair_legal({"hybrid_top_k": 2}, {"hybrid_top_k": 3})


def test_top5_to_dense_and_truncated_jsd():
    top = [[math.log(0.5), 7], [math.log(0.25), 8]]
    d = X.top5_to_dense(top)
    assert sum(d.values()) == pytest.approx(1.0)
    assert d[7] == pytest.approx(2 / 3)
    assert X.truncated_jsd(d, d) == pytest.approx(0.0, abs=1e-12)
    other = X.top5_to_dense([[math.log(0.9), 9]])
    v = X.truncated_jsd(d, other)
    assert 0.0 < v <= X.JSD_MAX_NATS + 1e-12
    assert X.truncated_jsd({}, d) is None


def _fake_capture(qid, text, lp=-0.1, top_shift=0):
    ids = _mk_ids(text)
    spans = spans_from_generation(_decode, ids)
    steps = [{"step": i, "token_id": t, "chosen_logprob": lp,
              "top5": [[math.log(0.6), t + top_shift], [math.log(0.4), t + 1 + top_shift]]}
             for i, t in enumerate(ids)]
    return {"qid": qid, "text": text, "steps": steps,
            "spans": {k: v for k, v in spans.items() if k != "text"}}


def test_spread_row_is_zero_when_the_two_configs_agree():
    rec = _fake_capture("s:1", CALL_TEXT)
    row = X.paired_config_spread_row(rec, dict(rec))
    assert row["spread_action_disagree"] == 0.0
    assert row["spread_name_disagree"] == 0.0
    assert row["spread_span_nll_spread"] == pytest.approx(0.0)
    assert row["spread_logit_jsd_mean"] == pytest.approx(0.0, abs=1e-12)
    assert row["spread_first_token_divergence"] is None
    assert row["logit_spread_truncated"] is True and row["resampling"] is False
    assert row["cost_multiplier"] == X.SPREAD_COST_MULTIPLIER


def test_spread_row_detects_disagreement_and_nll_spread():
    a = _fake_capture("s:1", CALL_TEXT, lp=-0.1)
    b = _fake_capture("s:1", CALL_TEXT.replace("venmo__search", "venmo__lookup"), lp=-0.5)
    row = X.paired_config_spread_row(a, b)
    assert row["spread_action_disagree"] == 1.0
    assert row["spread_name_disagree"] == 1.0
    assert row["spread_span_nll_spread"] > 0
    assert row["spread_first_token_divergence"] is not None
    assert row["spread_mean_output_logprob"] == pytest.approx(-0.1)


def test_span_nll_is_none_without_a_tool_call_span():
    rec = _fake_capture("s:1", "just prose, no call")
    assert X._span_nll(rec) is None
    row = X.paired_config_spread_row(rec, dict(rec))
    assert row["spread_span_nll_spread"] is None       # no whole-output fallback
    assert row["spread_name_disagree"] is None


def test_evicted_mass_control_is_none_not_zero_without_the_dropped_text():
    """2607.21475 mandatory baseline (3).  A row that HAS drops but no plaintext for
    them has an UNDEFINED evicted mass -- reporting 0.0 would read as 'nothing was
    evicted', which is the opposite of the truth."""
    ctl = X.evicted_mass_control_row({"qid": "s:1", "docs": ["aaa", "bbb"],
                                      "dropped_docs": [1], "doc_lengths": [3, 3]})
    assert ctl["baseline_evicted_doc_frac"] == pytest.approx(1 / 3)
    assert ctl["baseline_evicted_char_mass"] is None
    assert ctl["baseline_retained_len_entropy"] == pytest.approx(math.log(2))
    assert ctl["evicted_control_available"] is True


def test_evicted_mass_control_from_a_complete_sidecar():
    ctl = X.evicted_mass_control_row({"qid": "s:1", "docs": ["aaa", "bbb"],
                                      "dropped_docs": [1], "dropped_doc_texts": ["cc"],
                                      "doc_lengths": [3, 3]})
    assert ctl["baseline_evicted_doc_frac"] == pytest.approx(1 / 3)
    assert ctl["baseline_evicted_char_mass"] == pytest.approx(2 / 8)
    assert ctl["baseline_n_dropped_docs"] == 1 and ctl["baseline_n_kept_docs"] == 2


def test_evicted_mass_control_is_zero_when_nothing_was_dropped():
    ctl = X.evicted_mass_control_row({"qid": "s:1", "docs": ["aaa"], "dropped_docs": [],
                                      "doc_lengths": [3]})
    assert ctl["baseline_evicted_doc_frac"] == 0.0
    assert ctl["baseline_evicted_char_mass"] == 0.0
    # one kept block: the retained-entropy control is a degenerate constant -> None
    assert ctl["baseline_retained_len_entropy"] is None


def test_retained_length_entropy_edge_cases():
    assert X.retained_length_entropy(None) is None
    assert X.retained_length_entropy([5]) is None
    assert X.retained_length_entropy([0, 0]) is None
    assert X.retained_length_entropy([1, 1, 1, 1]) == pytest.approx(math.log(4))
    # concentrated mass -> lower entropy than the uniform split
    assert X.retained_length_entropy([90, 10]) < math.log(2)


def test_paired_config_label_frames_match_the_frozen_manifest():
    frame = C.FrozenAssets(ROOT).load()
    fr = X.paired_config_label_frames(frame)
    tr = frame.manifest["transitions"]
    assert fr["prediction"]["n"] == 900
    assert fr["prediction"]["n_pos"] == tr["C->W"] + tr["W->W"] == 712
    assert fr["attribution"]["n"] == 712
    assert fr["attribution"]["n_pos"] == tr["C->W"] == 93


# ===========================================================================
# (D) SelfCheckGPT N=1
# ===========================================================================

def test_string_number_leaves_excludes_bools_and_null():
    vals = X.string_number_leaves({"a": "Alice", "b": 42, "c": True, "d": None,
                                   "e": [1.5, "x"]})
    assert set(vals) == {"Alice", "42", "1.5", "x"}


def test_lexical_grounding_denominators_and_case_folding():
    ref = "the user ALICE asked venmo__search to run"
    out = X.lexical_grounding("venmo__search", {"q": "alice"}, ref)
    assert out["selfcheck_grounding_all"] == pytest.approx(1.0)
    assert out["selfcheck_grounding_args"] == pytest.approx(1.0)
    assert out["selfcheck_n_leaves_args"] == 1
    assert out["selfcheck_name_only_row"] is False
    miss = X.lexical_grounding("venmo__search", {"q": "bob"}, ref)
    assert miss["selfcheck_grounding_args"] == pytest.approx(0.0)
    assert miss["selfcheck_grounding_all"] == pytest.approx(0.5)


def test_lexical_grounding_name_only_row_gives_none_for_the_args_variant():
    out = X.lexical_grounding("venmo__search", {}, "venmo__search")
    assert out["selfcheck_grounding_args"] is None      # never divide by 93
    assert out["selfcheck_n_leaves_args"] == 0
    assert out["selfcheck_name_only_row"] is True
    assert out["selfcheck_grounding_all"] == pytest.approx(1.0)


def test_verbalize_call_template():
    s = X.verbalize_call("venmo__search", {"q": "alice", "n": 2})
    assert s == "The assistant calls venmo__search with q=alice, n=2."
    assert X.verbalize_call("t", {}) == "The assistant calls t with no arguments."
    assert X.verbalize_call(None, {}) is None


def test_contradiction_from_logits_drops_neutral_and_reads_id2label():
    id2label = {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"}
    z = [2.0, 5.0, 1.0]
    got = X.contradiction_from_logits(z, id2label)
    assert got == pytest.approx(math.exp(2.0) / (math.exp(2.0) + math.exp(1.0)))
    # a permuted head ordering must give the same answer
    id2 = {0: "entailment", 1: "neutral", 2: "contradiction"}
    z2 = [1.0, 5.0, 2.0]
    assert X.contradiction_from_logits(z2, id2) == pytest.approx(got)
    assert X.contradiction_from_logits(z, {0: "a", 1: "b", 2: "c"}) is None


def test_selfcheck_row_reads_only_the_compressed_arms_own_text_and_the_sidecar(tmp_path):
    side = {"qid": "sess:4", "docs": ["alice used venmo__search yesterday", "unrelated"],
            "query": "do it again", "dropped_docs": []}
    row = X.selfcheck_row(CALL_TEXT, side, nli_fn=lambda p, h: 0.25)
    assert row["qid"] == "sess:4" and row["session_id"] == "sess"
    assert row["selfcheck_grounding_all"] == pytest.approx(1.0)
    assert row["selfcheck_nli_contradiction"] == pytest.approx(0.25)
    assert row["selfcheck_parse_ok"] is True
    # the emitted columns must survive the leakage guard
    n = C.write_features_jsonl(tmp_path / "sc.jsonl", [row], context="selfcheck test")
    assert n == 1
    # mechanical same-event check: the label function reads tool_name_match off BOTH
    # arms' rows; the feature function is handed neither, and poisoning the sidecar
    # with those fields must not change a single emitted value.
    poisoned = dict(side)
    poisoned.update({"tool_name_match": True, "target": "x", "target_tool_name": "y",
                     "full_prediction": "z"})
    assert X.selfcheck_row(CALL_TEXT, poisoned, nli_fn=lambda p, h: 0.25) == row
    assert not (set(row) & {"tool_name_match", "target", "target_tool_name"})


def test_build_reference_text_uses_every_kept_doc_plus_the_query():
    # ``docs`` is the visible set; ``dropped_docs`` indexes the post-split list, so
    # filtering docs by it would delete a VISIBLE block and understate grounding.
    side = {"docs": ["A", "C"], "dropped_docs": [1], "dropped_doc_texts": ["B"],
            "query": "Q"}
    ref = X.build_reference_text(side)
    assert ref == "A\n\nC\n\nQ"
    assert "B" not in ref                      # the dropped block is NOT visible


def test_build_reference_text_keeps_all_docs_when_a_drop_index_collides():
    side = {"docs": ["alpha", "bravo", "charlie"], "dropped_docs": [1, 3],
            "query": "Q"}
    ref = X.build_reference_text(side)
    for doc in ("alpha", "bravo", "charlie"):
        assert doc in ref


# ===========================================================================
# scoring frame
# ===========================================================================

def test_score_feature_table_uses_the_frames_own_prevalence_and_orientation():
    frame = C.FrozenAssets(ROOT).load()
    subset = frame.trigger_subset()
    assert len(subset) == 161
    rows = []
    for r in subset:
        # a synthetic feature that is perfectly anti-correlated with the label
        rows.append({"qid": r["qid"], "session_id": r["session_id"],
                     "toy_signal": -float(r["label_cw"])})
    rep = X.score_feature_table(rows, frame, {"toy_signal": -1}, reps=50)
    col = rep["columns"]["toy_signal"]
    assert col["n"] == 161 and col["n_pos"] == 93
    assert col["chance_ap"] == pytest.approx(93 / 161)          # 0.578, never 0.1033
    assert col["chance_ap"] != pytest.approx(C.BASE_RATE_900)
    assert col["auroc"] == pytest.approx(1.0)                    # orientation applied
    flipped = X.score_feature_table(rows, frame, {"toy_signal": 1}, reps=50)
    assert flipped["columns"]["toy_signal"]["auroc"] == pytest.approx(0.0)


def test_score_feature_table_skips_constant_columns():
    frame = C.FrozenAssets(ROOT).load()
    rows = [{"qid": r["qid"], "session_id": r["session_id"],
             "toy_constant": 2.0} for r in frame.trigger_subset()]
    rep = X.score_feature_table(rows, frame, {"toy_constant": 1}, reps=10)
    assert rep["columns"]["toy_constant"]["note"] == "constant column, not scored"


def test_score_feature_table_refuses_an_undeclared_orientation():
    frame = C.FrozenAssets(ROOT).load()
    rows = [{"qid": r["qid"], "session_id": r["session_id"],
             "undeclared_signal": float(hash(r["qid"]) % 97)}
            for r in frame.trigger_subset()]
    rep = X.score_feature_table(rows, frame, {}, reps=10)
    col = rep["columns"]["undeclared_signal"]
    assert col["note"] == "no declared orientation, not scored"
    assert "auroc" not in col and "auprc" not in col


def test_score_feature_table_marks_diagnostic_columns():
    frame = C.FrozenAssets(ROOT).load()
    rows = [{"qid": r["qid"], "session_id": r["session_id"],
             "spread_n_common_positions": 7.0} for r in frame.trigger_subset()]
    rep = X.score_feature_table(rows, frame, {"spread_n_common_positions": 1}, reps=10)
    assert (rep["columns"]["spread_n_common_positions"]["note"]
            == "diagnostic column, not a scored feature")


def test_score_feature_table_names_the_missing_mandatory_baselines():
    """2607.21475's card adopts four mandatory baselines verbatim; the spread table
    must not be published with two of them silently absent."""
    frame = C.FrozenAssets(ROOT).load()
    subset = frame.trigger_subset()
    rows = [{"qid": r["qid"], "session_id": r["session_id"],
             "spread_mean_output_logprob": -float(r["label_cw"]) - 0.5}
            for r in subset]
    rep = X.score_feature_table(rows, frame, {"spread_mean_output_logprob": -1}, reps=10)
    led = rep["mandatory_baselines_2607_21475"]
    assert led["mean_output_logprob"]["present"] is True
    assert led["parse_failure_alone"]["present"] is True
    assert led["evicted_score_mass_predicted_weak"]["present"] is False
    assert led["matched_fire_rate_coin"]["present"] is False
    assert "t34_controls" in led["matched_fire_rate_coin"]["owner"]
    # a coin's expected precision is the FRAME's prevalence, never the 900-row 0.1033
    assert (led["matched_fire_rate_coin"]["expected_precision_at_matched_fire_rate"]
            == pytest.approx(93 / 161))
    assert "evicted_score_mass_predicted_weak" in rep["missing_mandatory_baselines"]
    assert rep["spread_table_publishable"] is False

    with_ctl = [dict(r, baseline_evicted_doc_frac=float(i % 5) / 4.0,
                     baseline_retained_len_entropy=float(i % 3) / 2.0)
                for i, r in enumerate(rows)]
    rep2 = X.score_feature_table(
        with_ctl, frame,
        {"spread_mean_output_logprob": -1, "baseline_evicted_doc_frac": 1,
         "baseline_retained_len_entropy": 1}, reps=10)
    led2 = rep2["mandatory_baselines_2607_21475"]
    assert led2["evicted_score_mass_predicted_weak"]["present"] is True
    assert rep2["declared_controls"]["retained_entropy_analogue"]["present"] is True
    # only the coin (another unit's file) is still missing
    assert rep2["missing_mandatory_baselines"] == ["matched_fire_rate_coin"]


def _emitted_feature_rows():
    """Every feature row shape this unit writes, with realistic (non-None) values."""
    st = X.asymspec_position_stats(np.array([[2.0, 0.0, 1.0], [0.5, 1.5, 0.2]]),
                                   np.array([[0.0, 2.0, 1.0], [1.5, 0.5, 0.2]]))
    asym = dict(X.asymspec_aggregate(st, [0]))
    asym["asym_jsd_max_nullctl"] = 0.0
    asym["asym_jsd_mean_nullctl"] = 0.0
    spread = X.paired_config_spread_row(_fake_capture("s:1", CALL_TEXT),
                                        _fake_capture("s:1", CALL_TEXT))
    sc = X.selfcheck_row(CALL_TEXT, {"qid": "s:1", "docs": ["nyc"], "query": "q"},
                         nli_fn=lambda p, h: 0.0)
    ctl = X.evicted_mass_control_row({"qid": "s:1", "docs": ["aaa", "bbb"],
                                      "dropped_docs": [1], "dropped_doc_texts": ["cc"],
                                      "doc_lengths": [3, 3]})
    return [asym, spread, sc, ctl]


def test_declared_orientations_cover_every_scorable_feature_name():
    """Derive the scorable set from ``score_feature_table``'s OWN rules, not from a
    hand-written name filter -- a filter can silently exempt a new column, and the
    scorer would then rank it under a default orientation."""
    orient = C.load_orientations(ROOT / "configs/t34/orientations_extra.json")
    scored = set()
    for row in _emitted_feature_rows():
        for key, val in row.items():
            if key in C.META_COLS or key in X.DIAGNOSTIC_COLS:
                continue
            if isinstance(val, bool) or isinstance(val, str) or val is None:
                continue
            if isinstance(val, (int, float)):
                scored.add(key)
    assert scored, "no scorable feature names produced"
    assert scored <= set(orient), f"undeclared orientations: {sorted(scored - set(orient))}"


def test_the_s0_entropy_control_is_emitted_and_oriented():
    """2608.26004 card, mandatory S0 control (b): the drafter's own single-ended
    entropy on x_comp_tilde -- the 'generic sample difficulty' confound that killed
    S1 at 0.5313.  It must exist as its own column, not be folded into D_i."""
    logits_b = np.array([[0.0, 0.0], [10.0, -10.0]])
    st = X.asymspec_position_stats(np.zeros_like(logits_b), logits_b)
    assert st["entropy_b"][0] == pytest.approx(math.log(2.0))   # uniform row
    assert st["entropy_b"][1] < 1e-6                            # near-deterministic row
    agg = X.asymspec_aggregate(st, [0])
    assert agg["asym_entropy_comp_max"] == pytest.approx(math.log(2.0))
    assert agg["asym_entropy_comp_mean"] == pytest.approx(math.log(2.0) / 2, abs=1e-6)
    orient = C.load_orientations(ROOT / "configs/t34/orientations_extra.json")
    assert orient["asym_entropy_comp_mean"] == 1 and orient["asym_entropy_comp_max"] == 1


def test_action_disagree_is_action_level_not_text_level():
    """2607.21475 E1: action_disagree = 1[action_canonical differs].  Key order and
    payload whitespace are NOT an action change; a different tool name is."""
    a = '<tool_call>\n{"name": "t__a", "arguments": {"x": 1, "y": 2}}\n</tool_call>'
    b = '<tool_call>\n{"arguments": {"y": 2, "x": 1},   "name": "t__a"}\n</tool_call>'
    c = '<tool_call>\n{"name": "t__b", "arguments": {"x": 1, "y": 2}}\n</tool_call>'
    assert X.canonical_action_key(a) == X.canonical_action_key(b)
    assert X.canonical_action_key(a) != X.canonical_action_key(c)
    same = X.paired_config_spread_row(_fake_capture("s:1", a), _fake_capture("s:1", b))
    assert same["spread_action_disagree"] == 0.0
    assert same["spread_text_disagree"] == 1.0       # the text DID differ
    diff = X.paired_config_spread_row(_fake_capture("s:1", a), _fake_capture("s:1", c))
    assert diff["spread_action_disagree"] == 1.0


def test_canonical_action_key_keeps_unparsed_rows_total():
    """An emission with no parseable call still gets a total comparison (falls back to
    the whitespace-normalised text) rather than being declared equal to every other
    unparseable emission."""
    assert X.canonical_action_key("free text one")[0] == "unparsed"
    assert X.canonical_action_key("free text one") != X.canonical_action_key("other text")
    assert X.canonical_action_key("free  text") == X.canonical_action_key("free text")


def test_canonical_action_key_is_diffed_against_the_bench_face_when_it_exists():
    """The local rebuild and benchmarks/proxy.py's action_canonical must induce the SAME
    partition (the estimand is 1[differs], not the key's shape).  The bench proxy in this
    worktree has no such symbol, so this skips loudly instead of pretending it agrees."""
    fn = X.bench_face_action_canonical(ROOT)
    if fn is None:
        pytest.skip("benchmarks/proxy.py in this worktree defines no action_canonical; "
                    "the cross-face diff must run where that proxy version lives")
    a = '<tool_call>\n{"name": "t__a", "arguments": {"x": 1, "y": 2}}\n</tool_call>'
    b = '<tool_call>\n{"arguments": {"y": 2, "x": 1},   "name": "t__a"}\n</tool_call>'
    c = '<tool_call>\n{"name": "t__b", "arguments": {"x": 1, "y": 2}}\n</tool_call>'
    d = "free text, no call at all"
    for u, v in ((a, b), (a, c), (a, d), (d, d), (c, d)):
        assert (fn(u) == fn(v)) == (X.canonical_action_key(u) == X.canonical_action_key(v)), \
            f"bench face and local rebuild disagree on ({u!r}, {v!r})"


def test_scored_columns_carry_their_declared_caveat():
    """A caveat that lives only in the module docstring is not read at the moment the
    column is ranked; the S0 entropy control is on the PROXY view and the null control
    is a determinism sentinel, and the table must say so."""
    frame = C.FrozenAssets(ROOT).load()
    subset = frame.trigger_subset()
    rows = [{"qid": r["qid"], "session_id": r["session_id"],
             "asym_entropy_comp_mean": float(hash(r["qid"]) % 13) / 13.0,
             "spread_logit_jsd_mean": float(hash(r["qid"]) % 7) / 7.0}
            for r in subset]
    rep = X.score_feature_table(rows, frame,
                                {"asym_entropy_comp_mean": 1, "spread_logit_jsd_mean": 1},
                                reps=10)
    note = rep["columns"]["asym_entropy_comp_mean"]["declared_note"]
    assert "PROXY view" in note and "gist view" in note
    assert "OVER-estimates" in rep["columns"]["spread_logit_jsd_mean"]["declared_note"]
    assert set(X.COLUMN_NOTES) <= set(
        C.load_orientations(ROOT / "configs/t34/orientations_extra.json"))


def test_module_declares_its_deviations():
    for mod in (X, CC):
        assert isinstance(mod.DEVIATIONS, list) and mod.DEVIATIONS
        for d in mod.DEVIATIONS:
            assert set(d) == {"method", "paper", "what", "why"}
            assert d["paper"].split()[0][0].isdigit()


# ===========================================================================
# ContextCite: design
# ===========================================================================

def test_enumerate_ablations_is_the_full_lattice():
    vs = CC.enumerate_ablations(3)
    assert len(vs) == 8 and len(set(vs)) == 8
    assert vs[0] == (0, 0, 0) and vs[-1] == (1, 1, 1)
    with pytest.raises(ValueError):
        CC.enumerate_ablations(21)


def test_sample_ablations_is_seeded_and_reproducible():
    a = CC.sample_ablations("sess:7", 16, 32)
    b = CC.sample_ablations("sess:7", 16, 32)
    c = CC.sample_ablations("sess:8", 16, 32)
    assert a == b and a != c
    assert len(a) == 32 and all(len(v) == 16 for v in a)
    assert set(x for v in a for x in v) <= {0, 1}


def test_build_design_switches_at_the_declared_d():
    assert CC.build_design("q:1", 6)["mode"] == "enumerate"
    assert CC.build_design("q:1", 6)["n_ablations"] == 64
    assert CC.build_design("q:1", 7, n=32)["mode"] == "sample"
    assert CC.build_design("q:1", 7, n=32)["n_ablations"] == 32


def test_design_over_the_real_witness_table_has_both_branches():
    w = json.loads((ROOT / "configs/bdf_pilot/d_witness_r2.json").read_text(encoding="utf-8"))
    modes = [CC.build_design(q, int(e["n_docs"]))["mode"] for q, e in w["entries"].items()]
    assert modes.count("enumerate") == 31 and modes.count("sample") == 62


def test_logit_from_logp_matches_the_closed_form():
    for p in (0.001, 0.1, 0.5, 0.9):
        assert CC.logit_from_logp(math.log(p)) == pytest.approx(math.log(p / (1 - p)),
                                                                abs=1e-9)
    assert CC.logit_from_logp(-1e-15) == 36.0


# ===========================================================================
# ContextCite: the regression and the cache-pollution guards
# ===========================================================================

class _CleanScorer:
    """A faithful score_fn: a FRESH prefix per ablation, nothing shared.

    Also detects the failure mode itself: ``kv`` stands in for a KV cache; a clean
    implementation never grows it across calls.
    """

    def __init__(self, true_w, noise=0.0):
        self.true_w = np.asarray(true_w, dtype=float)
        self.kv = []                       # shared object; must never grow
        self.calls = []
        self.noise = noise

    def fingerprint(self):
        return len(self.kv)

    def __call__(self, qid, v):
        prefix = list(v)                   # fresh per call
        assert prefix is not self.kv
        self.calls.append(tuple(v))
        base = -6.0 + float(self.true_w @ np.asarray(v, dtype=float))
        return min(-1e-6, base)


class _PollutingScorer(_CleanScorer):
    """Deliberately reproduces the round-1 bug: ``generate(use_cache=True)`` appends
    prompt+answer KV to a SHARED cache, so every later score drifts."""

    def __call__(self, qid, v):
        self.kv.extend(v)                  # in-place growth of the shared cache
        drift = 0.05 * len(self.kv)
        return min(-1e-6, super().__call__(qid, v) - drift)


def test_run_attribution_recovers_signs_on_a_clean_scorer():
    sc = _CleanScorer([2.0, 0.0, -1.0, 0.0])
    row = CC.run_attribution("sess:1", 4, sc, state_fingerprint=sc.fingerprint)
    assert row["cache_pollution_detected"] is False
    assert row["design_mode"] == "enumerate" and row["n_ablations"] == 16
    w = np.asarray(row["weights"])
    assert w[0] > 0 and w[2] < 0
    assert abs(w[1]) < 1e-6 and abs(w[3]) < 1e-6
    assert row["lam"] in CC.LASSO_LAMBDA_GRID
    assert row["stamp"].startswith("oracle upper envelope")
    # the shared cache never grew: the prefix was fresh per ablation
    assert sc.kv == []
    # the v=0 sentinel was scored first and last
    assert sc.calls[0] == (0, 0, 0, 0) and sc.calls[-1] == (0, 0, 0, 0)


def test_run_attribution_detects_the_round1_cache_pollution_pattern():
    sc = _PollutingScorer([2.0, 0.0, -1.0, 0.0])
    with pytest.raises(CC.CachePollutionError):
        CC.run_attribution("sess:1", 4, sc)          # sentinel alone must catch it
    assert sc.kv, "the polluting fake did grow the shared cache"


def test_run_attribution_state_fingerprint_catches_in_place_growth_immediately():
    sc = _PollutingScorer([1.0, 0.0])
    with pytest.raises(CC.CachePollutionError, match="mutated shared state"):
        CC.run_attribution("sess:1", 2, sc, state_fingerprint=sc.fingerprint)


def test_run_attribution_sampled_branch_is_reproducible():
    sc1 = _CleanScorer([1.0] + [0.0] * 7)
    sc2 = _CleanScorer([1.0] + [0.0] * 7)
    a = CC.run_attribution("sess:5", 8, sc1, n=32)
    b = CC.run_attribution("sess:5", 8, sc2, n=32)
    assert a["design_mode"] == "sample" and a["n_ablations"] == 32
    assert a["weights"] == b["weights"]
    assert a["holdout_idx"] == b["holdout_idx"] and a["fit_idx"] == b["fit_idx"]
    assert set(a["holdout_idx"]) & set(a["fit_idx"]) == set()


def test_metric_vectors_are_scored_on_the_sampled_branch():
    """2409.00729 section 3.2 evaluates top-k drop by RUNNING the model on
    ablate(C, v_top-k).  At d = 8 the uniform sample of 32 essentially never contains
    v = 1 / v = 0 / the top-k vectors, so they must be scored explicitly -- otherwise
    the paper's headline metric is None on the large-d rows."""
    sc = _CleanScorer([3.0, 2.0, 1.0] + [0.0] * 5)
    row = CC.run_attribution("sess:9", 8, sc, n=32)
    assert row["design_mode"] == "sample"
    assert row["n_metric_vectors"] > 0
    scored = {tuple(r["v"]) for r in row["records"] if r["logp"] is not None}
    assert tuple([1] * 8) in scored and tuple([0] * 8) in scored
    # the metric rows are NOT in the fit or the LDS hold-out
    design_idx = {i for i, r in enumerate(row["records"]) if r["role"] == "design"}
    assert set(row["fit_idx"]) <= design_idx
    assert set(row["holdout_idx"]) <= design_idx
    logp_by_v = {tuple(r["v"]): r["logp"] for r in row["records"] if r["logp"] is not None}
    m = CC.top_k_metrics(row["weights"], logp_by_v, ks=(1, 3, 5))
    for k in (1, 3, 5):
        assert m[f"top{k}_drop"] is not None and m[f"top{k}_gain"] is not None


def test_metric_vectors_are_reused_not_rescored_on_an_enumerated_design():
    """The cost cap: an enumerated lattice ALREADY contains v=1, v=0 and every top-k
    vector, so the section 3.2 metrics must cost zero extra score_fn calls there."""
    sc = _CleanScorer([2.0, 1.0, 0.0])
    row = CC.run_attribution("sess:3", 3, sc)
    assert row["design_mode"] == "enumerate"
    assert row["n_metric_vectors"] == 0
    assert row["metric_call_bound"] == 0
    assert len(sc.calls) == 2 + 2 ** 3               # two v=0 sentinels + the lattice
    logp_by_v = {tuple(r["v"]): r["logp"] for r in row["records"]
                 if r["logp"] is not None}
    m = CC.top_k_metrics(row["weights"], logp_by_v, ks=(1, 3))
    assert m["top1_drop"] is not None and m["top3_gain"] is not None


def test_metric_call_bound_is_the_declared_cap():
    assert CC.MAX_METRIC_VECTORS == 8
    assert CC.metric_call_bound(16, "enumerate") == 0
    assert CC.metric_call_bound(16, "sample") == CC.MAX_METRIC_VECTORS
    assert CC.metric_call_bound(1, "sample") == 2      # 2^d binds before the k count
    assert CC.metric_call_bound(2, "sample") == 4
    assert CC.metric_call_bound(0, "sample") == 0


def test_metric_vectors_for_excludes_and_caps():
    w = [3.0, 2.0, 1.0, 0.0]
    full = CC.metric_vectors_for(w, ks=(1, 3))
    kept = CC.metric_vectors_for(w, ks=(1, 3), exclude={(1, 1, 1, 1), (0, 0, 0, 0)})
    assert (1, 1, 1, 1) not in kept and (0, 0, 0, 0) not in kept
    assert len(kept) == len(full) - 2
    assert len(CC.metric_vectors_for(w, ks=(1, 3), cap=2)) == 2
    assert CC.metric_vectors_for(w, ks=(1, 3), cap=0) == []


def test_run_attribution_respects_the_metric_call_cap():
    sc = _CleanScorer([3.0, 2.0, 1.0] + [0.0] * 5)
    row = CC.run_attribution("sess:9", 8, sc, n=32, max_metric_calls=2)
    assert row["max_metric_calls"] == 2
    assert row["n_metric_vectors"] <= 2


def test_sham_control_hook_is_absent_by_default_and_says_so():
    """2409.00729 card pitfall 2: gist->raw restores a LONGER span, so every weight
    must be read against the D-line's equal-length sham arm.  Unbound, the row must
    advertise that it has NO attribution floor rather than look complete."""
    sc = _CleanScorer([2.0, 0.0, -1.0])
    row = CC.run_attribution("sess:4", 3, sc)
    assert row["sham_available"] is False
    assert row["sham_weights"] is None and row["sham_weight_l2"] is None
    assert row["sham_n_calls"] == 0
    assert row["sham_arm"] == CC.D_LINE_SHAM_ARM
    assert row["sham_reference_l2"] == pytest.approx(0.0968)
    rep = CC.report([row], reps=10)
    assert rep["sham_control"]["n_rows_with_sham"] == 0
    assert "NO equal-length attribution floor" in rep["sham_control"]["note"]


def test_sham_control_hook_rescores_the_design_and_reports_a_floor():
    """The hook re-scores the DESIGN points through the sham scorer and fits a second
    LASSO; the sham weights are reported BESIDE the attribution weights and never
    subtracted from them."""
    sc = _CleanScorer([3.0, 0.0, -2.0])
    sham = _CleanScorer([0.0, 0.0, 0.0])     # equal-length neutral span: no information
    row = CC.run_attribution("sess:6", 3, sc, score_fn_sham=sham)
    assert row["sham_available"] is True
    assert row["sham_n_calls"] == row["n_ablations"] == 8
    assert row["sham_weight_l2"] == pytest.approx(0.0, abs=1e-6)
    assert row["attribution_weight_l2"] > row["sham_weight_l2"]
    assert row["weights"] != row["sham_weights"]          # never subtracted / replaced
    assert "never subtract" in row["sham_reading"]
    # the sham value rides on the record, so the artifact replays
    design = [r for r in row["records"] if r["role"] == "design"]
    assert design and all("logp_sham" in r for r in design)
    assert all("logp_sham" not in r for r in row["records"] if r["role"] == "metric")
    rep = CC.report([row], reps=10)
    assert rep["sham_control"]["n_rows_with_sham"] == 1
    assert rep["sham_control"]["d_line_reference_l2"] == pytest.approx(0.0968)


def test_top_k_metrics_are_none_when_k_exceeds_d():
    """A d=2 row must not report the SAME number for top1/top3/top5 -- that would
    make every aggregate over k silently double-count the small-d rows."""
    w = [1.0, 0.5]
    logp_by_v = {(1, 1): -1.0, (0, 0): -6.0, (0, 1): -4.0, (1, 0): -2.0}
    m = CC.top_k_metrics(w, logp_by_v, ks=(1, 3, 5))
    assert m["top1_drop"] is not None
    assert m["top3_drop"] is None and m["top5_drop"] is None
    assert m["top3_gain"] is None and m["top3_blocks"] is None


def test_metric_vectors_for_covers_the_paper_metric_vectors():
    w = [3.0, 2.0, 1.0, 0.0]
    mv = set(CC.metric_vectors_for(w, ks=(1, 3)))
    assert (1, 1, 1, 1) in mv and (0, 0, 0, 0) in mv
    assert (0, 1, 1, 1) in mv and (1, 0, 0, 0) in mv          # k = 1 drop / gain
    assert (0, 0, 0, 1) in mv and (1, 1, 1, 0) in mv          # k = 3 drop / gain


def test_holdout_indices_are_record_indices_even_with_unscored_ablations():
    """report() looks the held-out ablations up in ``records``; when some ablations
    fail to score, the stored indices must still point at the right rows."""
    class _Gappy(_CleanScorer):
        def __call__(self, qid, v):
            if sum(v) == 1:                       # every single-block ablation fails
                self.calls.append(tuple(v))
                return None
            return super().__call__(qid, v)

    sc = _Gappy([2.0, 0.0, -1.0, 0.5])
    row = CC.run_attribution("sess:2", 4, sc)
    recs = row["records"]
    assert row["n_scored"] == len(recs) - 4
    for i in row["fit_idx"] + row["holdout_idx"]:
        assert recs[i]["logp"] is not None
    assert set(row["fit_idx"]) & set(row["holdout_idx"]) == set()
    rep = CC.report([row], ks=(1,), reps=10)
    assert rep["n_qids"] == 1


def test_fit_lasso_declares_its_lambda_selection():
    V = np.array(CC.enumerate_ablations(3), dtype=float)
    y = V @ np.array([3.0, 0.0, -2.0])
    fit = CC.fit_lasso_weights(V, y)
    assert fit["lam"] in CC.LASSO_LAMBDA_GRID
    assert "declared" in fit["lam_selection"]
    assert fit["weights"][0] > 0 > fit["weights"][2]


# ===========================================================================
# ContextCite: metrics
# ===========================================================================

def test_top_k_vectors_and_metric_arithmetic_on_synthetic_f():
    w = [0.5, 3.0, -1.0]
    tv = CC.top_k_vectors(w, 1)
    assert tv["top"] == (1,) and tv["drop"] == (1, 0, 1) and tv["gain"] == (0, 1, 0)
    logp = {(1, 1, 1): -2.0, (1, 0, 1): -5.0, (0, 0, 0): -9.0, (0, 1, 0): -4.0}
    m = CC.top_k_metrics(w, logp, ks=(1,))
    assert m["top1_drop"] == pytest.approx(3.0)     # -2.0 - (-5.0)
    assert m["top1_gain"] == pytest.approx(5.0)     # -4.0 - (-9.0)
    assert m["top1_blocks"] == [1]
    m2 = CC.top_k_metrics(w, {}, ks=(1,))
    assert m2["top1_drop"] is None and m2["top1_gain"] is None   # never imputed


def test_top_k_vectors_break_ties_to_the_lowest_index():
    tv = CC.top_k_vectors([1.0, 1.0, 0.0], 1)
    assert tv["top"] == (0,)


def test_lds_is_spearman_of_actual_vs_predicted_effects():
    w = [1.0, 2.0]
    held = [((0, 0), -9.0), ((1, 0), -8.0), ((0, 1), -7.0), ((1, 1), -6.0)]
    assert CC.lds(w, held) == pytest.approx(1.0)
    held_rev = [(v, -lp) for v, lp in held]
    assert CC.lds(w, held_rev) == pytest.approx(-1.0)
    assert CC.lds(w, held[:2]) is None                   # < 3 points


def test_relevant_sources_uses_the_papers_factor_of_two():
    d = 2
    ones = (1, 1)
    logp = {ones: -1.0, (0, 1): -1.0 - math.log(2) - 0.1, (1, 0): -1.05}
    assert CC.relevant_sources([0.0, 0.0], logp) == 1
    assert CC.relevant_sources([0.0, 0.0], {(1, 1): -1.0}) is None


def test_locate_table_and_inverted_control_on_synthetic_weights():
    rows = [{"qid": "a:1", "weights": [0.1, 5.0, 0.2]},
            {"qid": "b:1", "weights": [4.0, 0.1, 0.2]}]
    witness = {"entries": {"a:1": {"k_witness": 1}, "b:1": {"k_witness": 0}}}
    out = CC.contextcite_label_locate_table(rows, witness)
    assert out["forward"]["hits"] == 2 and out["forward"]["n"] == 2
    assert out["inverted"]["hits"] == 0
    assert out["forward"]["floor"] == C.LOCATE_FLOOR_WRONG_BLOCK
    assert out["forward"]["witness_oracle"] == pytest.approx(71 / 93)


def test_flip_consistency_marks_rows_without_any_flip_as_undefined():
    rows = [{"qid": "a:1", "weights": [0.1, 5.0]}, {"qid": "b:1", "weights": [5.0, 0.1]}]
    flips = {"a:1": {0: False, 1: True}, "b:1": {0: False, 1: False}}
    out = CC.contextcite_label_flip_consistency(rows, flips)
    assert out["hits"] == 1 and out["n"] == 2 and out["abstained"] == 1
    assert out["n_rows_with_any_flip"] == 1


def test_report_end_to_end_on_synthetic_attributions():
    sc = _CleanScorer([2.0, 0.0, -1.0])
    row = CC.run_attribution("sess:1", 3, sc)
    rep = CC.report([row], witness={"entries": {"sess:1": {"k_witness": 0}}},
                    flip_table={"sess:1": {0: True}}, ks=(1,), reps=25)
    assert rep["n_qids"] == 1
    assert rep["design_modes"]["enumerate"] == 1
    assert rep["top1_gain"]["n"] == 1
    assert rep["locate"]["forward"]["hits"] == 1
    assert rep["flip_consistency"]["hits"] == 1
    assert rep["frozen_reference_rows"]["best_k_ceiling"] == (81, 93)
    assert "oracle upper envelope" in rep["stamp"]


# ===========================================================================
# CLIs
# ===========================================================================

@pytest.mark.parametrize("mod", [X, CC])
def test_parsers_build_and_help_does_not_crash(mod, capsys):
    p = mod.build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--help"])
    out = capsys.readouterr().out
    assert "usage" in out.lower()


def test_contextcite_design_cli_freezes_a_plan(tmp_path):
    out = tmp_path / "design.json"
    rc = CC.main(["design", "--witness", str(ROOT / "configs/bdf_pilot/d_witness_r2.json"),
                  "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_qids"] == 93
    assert payload["n_enumerated_rows"] == 31
    assert payload["max_total_score_fn_calls"] > 0
    assert payload["enumerate_max_d"] == CC.ENUMERATE_MAX_D


def test_spread_plan_cli_freezes_a_plan(tmp_path):
    out = tmp_path / "plan.json"
    rc = X.main(["spread-plan", "--witness",
                 str(ROOT / "configs/bdf_pilot/d_witness_r2.json"),
                 "--mode", "doc_permute", "--out", str(out)])
    assert rc == 0
    plan = json.loads(out.read_text(encoding="utf-8"))
    assert plan["resampling"] is False and len(plan["per_qid"]) == 93
    any_qid = next(iter(plan["per_qid"]))
    assert sorted(plan["per_qid"][any_qid]["config_b"]["doc_order"]) == \
        list(range(len(plan["per_qid"][any_qid]["config_b"]["doc_order"])))


def test_vericache_agree_cli(tmp_path, capsys):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(json.dumps({"qid": "s:1", "first_div_idx": 3, "div_bucket": "name_span",
                             "cap_hit": False, "accept_len": 3}) + "\n", encoding="utf-8")
    b.write_text(json.dumps({"qid": "s:1", "first_div_idx": 3, "div_bucket": "name_span",
                             "cap_hit": False, "accept_len": 3}) + "\n", encoding="utf-8")
    rc = X.main(["vericache-agree", "--pass_a", str(a), "--pass_b", str(b)])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["index_disagree_rate"] == 0.0 and rep["downgrade_to_bucket"] is False


def test_selfcheck_cli_end_to_end(tmp_path, capsys):
    side = tmp_path / "sidecar.jsonl"
    bat = tmp_path / "bat.jsonl"
    side.write_text(json.dumps({"qid": "sess:1", "docs": ["alice venmo__search"],
                                "query": "again", "dropped_docs": []}) + "\n",
                    encoding="utf-8")
    bat.write_text(json.dumps({"qid": "sess:1", "prediction": CALL_TEXT}) + "\n",
                   encoding="utf-8")
    out = tmp_path / "feat.jsonl"
    rc = X.main(["selfcheck", "--sidecar", str(side), "--battery_c2kv", str(bat),
                 "--out", str(out)])
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert info["rows"] == 1 and info["cost"]["lexical_gpu_sec"] == 0.0
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["selfcheck_grounding_all"] == pytest.approx(1.0)
    assert row["selfcheck_nli_contradiction"] is None      # kept as null, not imputed


def test_spread_cli_end_to_end(tmp_path, capsys):
    ca = tmp_path / "a.jsonl"
    cb = tmp_path / "b.jsonl"
    rec_a = _fake_capture("sess:1", CALL_TEXT, lp=-0.2)
    rec_b = _fake_capture("sess:1", CALL_TEXT.replace("alice", "bobby"), lp=-0.7)
    ca.write_text(json.dumps(rec_a) + "\n", encoding="utf-8")
    cb.write_text(json.dumps(rec_b) + "\n", encoding="utf-8")
    out = tmp_path / "spread.jsonl"
    rc = X.main(["spread", "--capture_a", str(ca), "--capture_b", str(cb),
                 "--root", str(ROOT), "--out", str(out)])
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert info["estimands"]["prediction"]["n"] == 900
    assert info["estimands"]["attribution"]["n_pos"] == 93
    assert info["cost_multiplier"] == 2.0
    # without --sidecar the mandatory predicted-weak baseline is ABSENT and says so
    assert info["n_rows_with_evicted_control"] == 0
    assert "evicted_score_mass_predicted_weak" in info["missing_mandatory_baselines"]
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["baseline_evicted_doc_frac"] is None      # None, never 0.0
    assert row["baseline_retained_len_entropy"] is None
    assert row["evicted_control_available"] is False


def test_spread_cli_attaches_the_evicted_control_from_the_sidecar(tmp_path, capsys):
    """2607.21475 mandatory baseline (3): the spread table must not be published
    without the dropped_docs / evicted-score-mass predicted-weak control."""
    ca = tmp_path / "a.jsonl"
    cb = tmp_path / "b.jsonl"
    side = tmp_path / "sidecar.jsonl"
    ca.write_text(json.dumps(_fake_capture("sess:1", CALL_TEXT, lp=-0.2)) + "\n",
                  encoding="utf-8")
    cb.write_text(json.dumps(_fake_capture("sess:1", CALL_TEXT, lp=-0.7)) + "\n",
                  encoding="utf-8")
    side.write_text(json.dumps({"qid": "sess:1", "docs": ["aaa", "bbb"],
                                "query": "q", "dropped_docs": [1],
                                "dropped_doc_texts": ["cc"],
                                "doc_lengths": [3, 3]}) + "\n", encoding="utf-8")
    out = tmp_path / "spread.jsonl"
    rc = X.main(["spread", "--capture_a", str(ca), "--capture_b", str(cb),
                 "--sidecar", str(side), "--root", str(ROOT), "--out", str(out)])
    assert rc == 0
    info = json.loads(capsys.readouterr().out)
    assert info["n_rows_with_evicted_control"] == 1
    assert info["missing_mandatory_baselines"] == ["matched_fire_rate_coin"]
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["baseline_evicted_doc_frac"] == pytest.approx(1 / 3)
    assert row["baseline_evicted_char_mass"] == pytest.approx(2 / 8)
    assert row["evicted_control_available"] is True
    assert row["spread_mean_output_logprob"] is not None
