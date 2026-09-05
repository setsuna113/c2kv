# -*- coding: utf-8 -*-
"""Unit tests for t34 U7a (digest 4.9 bench-side rescoring).

Every test runs on synthetic inputs on the Windows box: no torch, no GPU, no
network.  The two tests that touch the frozen battery skip cleanly when the
frozen artefacts are not present.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import t34_bench_signals as B
import t34_common as C

ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------
# bookkeeping
# --------------------------------------------------------------------------

def test_deviations_are_declared_and_complete():
    assert B.DEVIATIONS
    for entry in B.DEVIATIONS:
        assert set(entry) == {"method", "paper", "what", "why"}
        assert all(entry[k].strip() for k in entry)
        assert entry["paper"] in {"2608.11977", "2608.16370", "2608.29685", "2510.07777"}


def test_orientations_config_matches_rationale():
    table = B.declared_orientations()
    assert table, "orientations_bencha.json must exist and be non-empty"
    assert set(table.values()) <= {1, -1}
    for name in table:
        if name.startswith("lsm_"):
            sig = next(s for s in B.BASE_STEP_SIGNALS if name.startswith("lsm_" + s + "_"))
            assert "lsm_%s_*" % sig in B.ORIENTATION_RATIONALE
        else:
            assert name in B.ORIENTATION_RATIONALE
    # every base signal x reducer combination is pre-declared
    for sig in B.BASE_STEP_SIGNALS:
        for red in B.REDUCERS:
            assert B.lsm_feature_name(sig, red) in table


def test_orient_refuses_undeclared_feature():
    with pytest.raises(KeyError):
        B.orient("not_declared_anywhere", np.array([1.0, 2.0]))
    flipped = B.orient("ceq_kept_history_tokens", np.array([1.0, 2.0]))
    assert list(flipped) == [-1.0, -2.0]  # declared -1: more kept history = safer


# --------------------------------------------------------------------------
# (A) BTM -- 2608.11977
# --------------------------------------------------------------------------

def test_observation_classifier_explicit_only():
    cases = {
        "timeout": "Error: request timed out after 30s",
        "rate_limit": "HTTP 429 Too Many Requests",
        "auth_error": "401 Unauthorized: invalid api key",
        "server_error": "Internal Server Error while contacting upstream",
        "schema_drift": "unknown field 'customer_uuid' in request",
        "malformed_response": "could not parse response: invalid JSON at line 1",
    }
    for expect, text in cases.items():
        got = B.classify_observation(text)
        assert got["class"] == expect, (expect, got)
        assert got["channel"] == "explicit"
    clean = B.classify_observation('{"balance": 42.0, "status": "ok"}')
    assert clean["class"] is None and clean["channel"] == "unclassified"
    # silent modes are never emitted by the text classifier, by construction
    for text in ("partial results only", "value may be stale", "the answer is 7"):
        assert B.classify_observation(text)["class"] not in B.SILENT_FAILURE_MODES
    assert len(B.EXPLICIT_FAILURE_MODES) == 6 and len(B.SILENT_FAILURE_MODES) == 3


def _synthetic_flip_table():
    return {
        "s1:1": {0: False, 1: True, 2: False},     # S2', single k
        "s1:2": {0: True, 1: True},                # S2', multi k
        "s2:3": {0: False, 1: False},              # S3'
        "s2:4": {0: False, 1: True},               # S1' when the none arm flips it
    }


def test_solvability_classes_and_census():
    qids = ["s1:1", "s1:2", "s2:3", "s2:4", "s3:5"]
    classes = B.assign_solvability_classes(qids, _synthetic_flip_table(),
                                           none_arm_hits={"s2:4": True, "s1:1": False})
    assert classes["s1:1"]["class"] == "S2'" and classes["s1:1"]["single_k"]
    assert classes["s1:2"]["class"] == "S2'" and not classes["s1:2"]["single_k"]
    assert classes["s2:3"]["class"] == "S3'"
    assert classes["s2:4"]["class"] == "S1'"       # none arm wins over the flip table
    assert classes["s3:5"]["class"] == "S3'" and classes["s3:5"]["flip_table_missing"]
    cen = B.solvability_census(classes)
    assert (cen["n_S1"], cen["n_S2"], cen["n_S3"]) == (1, 2, 2)
    assert cen["n_S2_single_k"] == 1 and cen["n_S2_multi_k"] == 1
    assert cen["denominator_all"] == 5 and cen["denominator_S2"] == 2
    # the "coverage against 81, not 93" denominator is the RECOVERABLE set S1'+S2',
    # which parts ways with |S2'| the moment a none arm flips anything
    assert cen["denominator_recoverable"] == 3
    assert cen["n_flip_table_missing"] == 1


def test_recoverable_denominator_equals_s2_only_when_s1_is_empty():
    qids = ["s1:1", "s1:2", "s2:3"]
    no_none = B.solvability_census(
        B.assign_solvability_classes(qids, _synthetic_flip_table()))
    assert no_none["denominator_recoverable"] == no_none["denominator_S2"] == 2
    with_none = B.solvability_census(
        B.assign_solvability_classes(qids, _synthetic_flip_table(),
                                     none_arm_hits={"s1:2": True}))
    assert with_none["denominator_S2"] == 1
    assert with_none["denominator_recoverable"] == 2      # S1' + S2', not |S2'|


def test_no_none_arm_gives_constructively_empty_s1():
    qids = ["s1:1", "s2:3"]
    classes = B.assign_solvability_classes(qids, _synthetic_flip_table())
    assert B.solvability_census(classes)["n_S1"] == 0
    assert all(v["none_flips"] is False for v in classes.values())


def test_solvability_manifest_freeze_is_deterministic(tmp_path):
    classes = B.assign_solvability_classes(["s1:1", "s2:3"], _synthetic_flip_table())
    a = B.freeze_solvability_manifest(tmp_path / "m.json", classes, {"src": "t"})
    b = B.freeze_solvability_manifest(tmp_path / "m2.json", classes, {"src": "t"})
    assert a["sha256"] == b["sha256"]
    other = dict(classes)
    other["s2:3"] = dict(other["s2:3"], **{"class": "S2'"})
    c = B.freeze_solvability_manifest(tmp_path / "m3.json", other, {"src": "t"})
    assert c["sha256"] != a["sha256"]
    payload = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert payload["census"]["denominator_all"] == 2
    assert payload["paper"] == "2608.11977"


def test_metrics_by_solvability_class_uses_both_denominators():
    qids = ["s1:1", "s1:2", "s2:3", "cc:1", "cc:2"]
    y = [1, 1, 1, 0, 0]
    scores = [0.9, 0.8, 0.1, 0.7, 0.05]
    classes = B.assign_solvability_classes(qids[:3], _synthetic_flip_table())
    out = B.metrics_by_solvability_class(scores, y, qids, classes, n_fires=3)
    assert out["global"]["fires"] == 3 and out["global"]["coverage"] == 2
    assert out["global"]["false_resets"] == 1
    s2row = next(r for r in out["by_class"] if r["class"] == "S2'")
    assert s2row["n_pos_in_class"] == 2 and s2row["coverage"] == 2
    assert s2row["coverage_over_class"] == 1.0
    assert s2row["coverage_over_all_cw"] == pytest.approx(2 / 3)
    assert s2row["coverage_over_S2"] == pytest.approx(1.0)
    assert s2row["coverage_over_recoverable"] == pytest.approx(1.0)


def test_l1_visibility_cut():
    tools = [{"function": {"name": "get_order", "parameters": {
        "required": ["order_id"], "properties": {"order_id": {"type": "string"}}}}}]
    legal = 'Action:\n<tool_call>\n{"name":"get_order","arguments":{"order_id":"a1"}}\n</tool_call>'
    assert B.l1_visibility(legal, tools)["l1_visible"] is False
    assert B.l1_visibility(legal, tools)["protocol_legal"] is True
    assert B.l1_visibility(legal, [])["l1_visible"] is None      # legality not computable
    bad_args = 'Action:\n<tool_call>\n{"name":"get_order","arguments":{}}\n</tool_call>'
    assert B.l1_visibility(bad_args, tools)["l1_visible"] is True
    assert "order_id" in B.l1_visibility(bad_args, tools)["first_violation"]
    assert B.l1_visibility("I will look that up for you.", tools)["l1_visible"] is True
    truncated = 'Action:\n<tool_call>\n{"name":"get_order","arguments":{"order_id":"a'
    assert B.l1_visibility(truncated, tools)["l1_visible"] is True


def test_cap_caveat_blocks_the_cut_when_capped():
    hot = B.cap_caveat([True] * 5 + [False] * 5, 128)
    assert hot["cap_rate"] == 0.5 and hot["explicit_silent_computable"] is False
    assert "128" in hot["caveat"]
    cold = B.cap_caveat([True] + [False] * 19, 512)
    assert cold["explicit_silent_computable"] is True and cold["caveat"] is None
    assert B.cap_caveat([], 128)["explicit_silent_computable"] is None


# --------------------------------------------------------------------------
# (B) IRBench -- 2608.16370
# --------------------------------------------------------------------------

def _partition(**over):
    obj = {"benchmark": "synthetic", "retrieval": ["get_order", "search_product"],
           "execution": ["place_order"], "prose_or_other": ["respond"],
           "retrieval_returns": {}}
    obj.update(over)
    obj["sha256"] = B.partition_sha256(obj)
    return obj


def test_partition_template_on_disk_is_valid_and_empty():
    obj = B.load_tool_partition(ROOT / "configs/t34/tool_partition_template.json")
    rep = B.validate_tool_partition(obj)
    assert rep["ok"] and rep["sha256_ok"] is True
    assert rep["filled"] is False, "the template must ship empty; the runner fills it"
    # structurally valid but NOT a measurement: say so where a runner will see it
    assert rep["ready_for_measurement"] is False
    assert "heuristic" in rep["note"]
    assert B.validate_tool_partition(_partition())["ready_for_measurement"] is True
    assert B.validate_tool_partition(_partition())["note"] is None
    for bucket in B.TOOL_BUCKETS:
        assert bucket in obj


def test_partition_validator_catches_overlap_sha_and_gaps():
    bad = _partition()
    bad["execution"] = bad["execution"] + ["get_order"]
    rep = B.validate_tool_partition(bad)
    assert not rep["ok"] and any("both" in e for e in rep["errors"])
    stale = _partition()
    stale["sha256"] = "0" * 64
    assert B.validate_tool_partition(stale)["sha256_ok"] is False
    gap = B.validate_tool_partition(_partition(), declared_tools=["get_order", "cancel_order"])
    assert gap["unclassified"] == ["cancel_order"] and not gap["ok"]
    wrong_returns = _partition(retrieval_returns={"place_order": ["id"]})
    assert not B.validate_tool_partition(wrong_returns)["ok"]


def _proxy_rows():
    def row(arm, conv, turn, names, **kw):
        r = {"arm": arm, "conv_id": conv, "turn": turn,
             "action": {"tool_calls": [{"name": n, "arguments": {}} for n in names],
                        "text": ""}}
        r.update(kw)
        return r
    return [
        row("full", "c1", 0, ["get_order"]),          # turn 0: excluded
        row("full", "c1", 1, ["get_order"]),
        row("full", "c1", 2, ["place_order"]),
        row("full", "c1", 3, []),                     # prose step
        row("c2kv", "c1", 0, ["get_order"]),
        row("c2kv", "c1", 1, ["get_order", "search_product"]),
        row("c2kv", "c1", 2, ["place_order"]),
        row("c2kv", "c1", 3, ["mystery_tool"]),       # unbucketed
    ]


def test_cost_triples_exclude_turn0_and_keep_prose_bucket():
    triples = B.cost_triples(_proxy_rows(), _partition())
    full = triples[("full", "c1")]
    c2kv = triples[("c2kv", "c1")]
    assert full["n_steps"] == 3 and full["C_R"] == 1 and full["C_E"] == 1
    assert full["C_other"] == 1                       # the prose step, never dropped
    assert c2kv["C_R"] == 2 and c2kv["n_unbucketed"] == 1
    assert full["C_total"] == 3
    # including turn 0 changes the counts -> the exclusion is load-bearing
    with_zero = B.cost_triples(_proxy_rows(), _partition(), min_turn=0)
    assert with_zero[("full", "c1")]["C_R"] == 2


def test_cost_triples_report_missing_actions():
    rows = [{"arm": "a", "conv_id": "c", "turn": 1}]
    rec = B.cost_triples(rows, _partition())[("a", "c")]
    assert rec["n_no_action"] == 1 and rec["C_R"] == 0


def test_cost_triples_enforce_a_fixed_interaction_horizon():
    """2608.16370 Definition 1: the triple is recorded under a FIXED horizon."""
    rows = [{"arm": "a", "conv_id": "c", "turn": t,
             "action": {"tool_calls": [{"name": "get_order"}]}} for t in range(1, 8)]
    unbounded = B.cost_triples(rows, _partition())
    assert unbounded[("a", "c")]["C_R"] == 7
    assert unbounded[("a", "c")]["max_turn_seen"] == 7
    meta = unbounded[("__meta__", "__meta__")]
    assert meta["horizon_declared"] is False and "NOT horizon-bounded" in meta["note"]
    bounded = B.cost_triples(rows, _partition(), max_turn=4)
    assert bounded[("a", "c")]["C_R"] == 4          # turns 1..4 only
    bmeta = bounded[("__meta__", "__meta__")]
    assert bmeta["n_rows_over_horizon"] == 3 and bmeta["max_turn"] == 4
    assert bmeta["note"] is None
    assert B.PAPER_INTERACTION_HORIZON_TURNS == 24


def test_cost_triples_make_a_turnless_log_visible():
    """The short proxy schema has no `turn`; an empty result must say why."""
    rows = [{"arm": "a", "gist_tokens": 10}, {"arm": "a", "gist_tokens": 12}]
    out = B.cost_triples(rows, _partition())
    meta = out[("__meta__", "__meta__")]
    assert meta["n_rows_missing_turn"] == 2 and meta["n_rows"] == 2
    assert [k for k in out if k != ("__meta__", "__meta__")] == []


def test_cost_triples_record_termination_reasons():
    rows = [{"arm": "a", "conv_id": "c", "turn": 1, "finish_reason": "tool_calls",
             "action": {"tool_calls": [{"name": "get_order"}]}},
            {"arm": "a", "conv_id": "c", "turn": 2, "finish_reason": "length",
             "action": {"tool_calls": []}},
            {"arm": "a", "conv_id": "c", "turn": 3, "status": "cache_miss"}]
    rec = B.cost_triples(rows, _partition())[("a", "c")]
    assert rec["termination"] == {"tool_calls": 1, "length": 1, "cache_miss": 1}


def test_wilcoxon_bootstrap_and_holm_bookkeeping():
    diffs = [3.0, 4.0, 2.0, 5.0, 6.0, 3.5]
    w = B.wilcoxon_paired(diffs)
    assert w["n"] == 6 and w["p"] < 0.05
    assert B.wilcoxon_paired([0.0, 0.0])["p"] == 1.0
    ci = B.bootstrap_diff_ci(diffs, reps=500, seed=1)
    assert ci["lo"] < ci["mean"] < ci["hi"] and ci["lo"] > 0
    holm = B.holm_correct({"a": 0.001, "b": 0.04, "c": 0.9}, alpha=0.05)
    assert holm["a"]["rank"] == 1 and holm["a"]["reject"]
    assert holm["b"]["threshold"] == pytest.approx(0.05 / 2)
    assert holm["b"]["reject"] is False        # 0.04 > 0.025 -> step-down stops
    assert holm["c"]["reject"] is False        # monotone: nothing after a failure
    assert holm["a"]["p_adj"] == pytest.approx(0.003)


def test_irbench_report_declares_family_and_flags_post_hoc():
    triples = {}
    for i in range(10):
        conv = "c%d" % i
        triples[("full", conv)] = {"C_R": 10, "C_E": 5, "C_total": 15, "Q": 1.0}
        triples[("c2kv", conv)] = {"C_R": 10 + 3 + i * 0.1, "C_E": 5, "C_total": 18, "Q": 1.0}
    rep = B.irbench_report(triples, arm_a="c2kv", arm_b="full",
                           primary_family=("C_R",), post_hoc=("C_E", "Q"))
    assert rep["primary_family"] == ["C_R"]
    assert rep["metrics"]["C_R"]["post_hoc"] is False
    assert rep["metrics"]["C_E"]["post_hoc"] is True
    assert rep["metrics"]["C_R"]["n_pairs"] == 10
    assert "C_R" in rep["holm"] and "C_E" not in rep["holm"]
    assert rep["metrics"]["C_R"]["bootstrap_mean_diff"]["lo"] > 0
    # a single-metric family applies no correction: the scope must say so
    assert rep["holm_family_scope"].startswith("metrics within")
    assert "SIX retrieval COMPARISONS" in rep["holm_warning"]


def test_irbench_family_holm_runs_across_comparison_cells():
    """2608.16370 section 3.4: the family spans cells, not metrics inside one cell."""
    rng = np.random.default_rng(7)
    triples = {}
    # three arm pairs vs `full`; only the first has a real effect
    for i in range(12):
        conv = "c%d" % i
        triples[("full", conv)] = {"C_R": 10.0 + rng.normal(0, 0.1), "C_E": 5.0}
        triples[("big", conv)] = {"C_R": 16.0 + rng.normal(0, 0.1), "C_E": 5.0}
        triples[("mid", conv)] = {"C_R": 10.2 + rng.normal(0, 0.1), "C_E": 5.0}
        triples[("nil", conv)] = {"C_R": 10.0 + rng.normal(0, 0.1), "C_E": 5.0}
    comparisons = [("big", "full", "C_R"), ("mid", "full", "C_R"), ("nil", "full", "C_R")]
    rep = B.irbench_family_report(triples, comparisons,
                                  post_hoc=[("big", "full", "C_E")])
    assert rep["family_size"] == 3
    for cell in rep["holm"].values():
        assert cell["m"] == 3          # every p is corrected against the whole family
    assert rep["cells"]["big_vs_full::C_R"]["post_hoc"] is False
    assert rep["cells"]["big_vs_full::C_E"]["post_hoc"] is True
    assert "big_vs_full::C_E" not in rep["holm"]        # post-hoc never enters the family
    assert rep["holm"]["big_vs_full::C_R"]["reject"] is True
    assert rep["holm"]["nil_vs_full::C_R"]["reject"] is False
    assert rep["n_survive_holm"] >= 1
    # the meta row cost_triples appends must never become a paired observation
    triples[("__meta__", "__meta__")] = {"meta": True, "C_R": 999.0}
    again = B.irbench_family_report(triples, comparisons)
    assert again["cells"]["big_vs_full::C_R"]["n_pairs"] == 12


def _doc(name, args):
    return "<tool_call>\n%s\n</tool_call>" % json.dumps({"name": name, "arguments": args})


def test_dstar_schema_mode_beats_heuristic_mode():
    schema_part = _partition(retrieval_returns={"get_order": {"order_id": "string",
                                                             "status": "string"}})
    text = _doc("place_order", {"order_id": "a1", "prior_failure_count": 3})
    schema = B.block_r_frac(text, schema_part)
    assert schema["modes"] == ["schema"]
    # name + prior_failure_count are R; order_id is covered by get_order's return
    assert schema["n_values"] == 3 and schema["n_r"] == 2
    assert schema["r_frac"] == pytest.approx(2 / 3)
    heur = B.block_r_frac(text, _partition())
    assert heur["modes"] == ["heuristic"]
    assert heur["r_frac"] is not None
    assert B.block_r_frac("no tool call here", _partition())["r_frac"] is None


def test_block_value_strings_matches_frozen_witness_semantics():
    from d_witness_core import target_values
    text = _doc("get_order", {"order_id": "a1", "n": 2, "ok": True})
    assert B.block_value_strings(text) == target_values(
        "get_order", {"order_id": "a1", "n": 2, "ok": True})
    keyed = dict((k, v) for k, v in B.block_values(text))
    assert keyed["order_id"] == "a1" and keyed["name"] == "get_order"


def test_dstar_features_weighting_and_missing_values():
    """docs = KEPT blocks (compressed, weight = loss frac); dropped text = weight 1."""
    part = _partition(retrieval_returns={"get_order": {"order_id": "string"}})
    kept = [_doc("place_order", {"note": "hand written"}),   # both values R -> 1.0
            "plain prose, no call"]                          # no values
    dropped = [_doc("get_order", {"order_id": "a1"})]        # name R, order_id D -> 0.5
    feats = B.dstar_features_for_qid(kept, part, dropped_docs=[7],
                                     dropped_doc_texts=dropped,
                                     gist_tokens=50, original_tokens=200)
    assert feats["irb_aggregate_loss_frac"] == pytest.approx(0.75)
    assert feats["irb_n_dropped_blocks"] == 1
    assert feats["irb_n_blocks_with_values"] == 2
    assert feats["irb_s_t_scope"] == "kept_and_dropped"
    assert feats["irb_s_t_dropped_only"] == pytest.approx(0.5)
    assert feats["irb_s_t_r_weighted"] == pytest.approx(0.5 + 1.0 * 0.75)
    assert feats["irb_r_frac_dropped_mean"] == pytest.approx(0.5)
    assert feats["irb_r_frac_all_mean"] == pytest.approx((1.0 + 0.5) / 2)
    assert feats["irb_dr_mode"] == "schema"
    empty = B.dstar_features_for_qid(["prose"], part, dropped_docs=[])
    assert empty["irb_s_t_r_weighted"] is None      # None, never a sentinel
    assert empty["irb_s_t_scope"] is None


def test_dstar_diagnostic_columns_carry_no_orientation():
    """The scope / mode columns are bookkeeping, not risk scores."""
    table = B.declared_orientations()
    scored = [k for k in B.DSTAR_FEATURE_KEYS if k not in B.DSTAR_DIAGNOSTIC_COLUMNS]
    assert scored and all(k in table for k in scored)
    assert all(k not in table for k in B.DSTAR_DIAGNOSTIC_COLUMNS)
    for k in B.DSTAR_DIAGNOSTIC_COLUMNS:
        with pytest.raises(KeyError):
            B.orient(k, np.array([1.0]))


def test_dropped_docs_indices_never_select_from_the_kept_docs_list():
    """docs holds ONLY kept blocks; dropped_docs indexes the post-split list.

    The old ``if k in dropped`` over ``enumerate(docs)`` scored VISIBLE blocks as
    dropped, so changing the index values changed s_t.  They must not.
    """
    part = _partition()
    kept = [_doc("get_order", {"order_id": "a1"}), _doc("place_order", {"n": 1})]
    low = B.dstar_features_for_qid(kept, part, dropped_docs=[0, 1],
                                   gist_tokens=50, original_tokens=200)
    high = B.dstar_features_for_qid(kept, part, dropped_docs=[7, 9],
                                    gist_tokens=50, original_tokens=200)
    assert low["irb_s_t_r_weighted"] == high["irb_s_t_r_weighted"]
    assert low["irb_r_frac_all_mean"] == high["irb_r_frac_all_mean"]
    # every kept block is visible, so the kept half is the only half that exists
    assert low["irb_s_t_scope"] == "kept_only_partial"


def test_dstar_dropped_half_is_none_without_dropped_doc_texts():
    part = _partition()
    kept = [_doc("place_order", {"note": "x"})]
    partial = B.dstar_features_for_qid(kept, part, dropped_docs=[0, 4],
                                       gist_tokens=50, original_tokens=200)
    assert partial["irb_dropped_text_available"] is False
    assert partial["irb_s_t_dropped_only"] is None       # never 0, never a docs subset
    assert partial["irb_s_t_scope"] == "kept_only_partial"
    assert partial["irb_n_dropped_blocks"] == 2
    # with nothing dropped at all the dropped half is trivially complete
    none_dropped = B.dstar_features_for_qid(kept, part, dropped_docs=[],
                                            gist_tokens=50, original_tokens=200)
    assert none_dropped["irb_dropped_text_available"] is True
    assert none_dropped["irb_s_t_dropped_only"] == pytest.approx(0.0)
    assert none_dropped["irb_s_t_scope"] == "kept_and_dropped"


def test_dstar_kept_half_is_none_without_an_aggregate_loss_fraction():
    part = _partition()
    kept = [_doc("place_order", {"note": "x"})]
    dropped = [_doc("get_order", {"order_id": "a1"})]
    feats = B.dstar_features_for_qid(kept, part, dropped_docs=[3],
                                     dropped_doc_texts=dropped)
    assert feats["irb_aggregate_loss_frac"] is None
    assert feats["irb_s_t_scope"] == "dropped_only_partial"
    assert feats["irb_s_t_r_weighted"] == feats["irb_s_t_dropped_only"]


# --------------------------------------------------------------------------
# (C) Last Step Matters -- 2608.29685
# --------------------------------------------------------------------------

def test_progress_prefix_lengths_are_causal_and_monotone():
    lens = B.progress_prefix_lengths(20)
    assert len(lens) == B.N_PROGRESS_POINTS
    assert lens[0] == 1 and lens[-1] == 20
    assert all(b >= a for a, b in zip(lens, lens[1:]))
    assert all(1 <= v <= 20 for v in lens)
    # a prefix of length L may only contain steps [0, L): no lookahead
    rows = list(range(20))
    for i, L in enumerate(lens):
        assert max(rows[:L]) <= L - 1


def test_reducers():
    v = [1.0, 5.0, 2.0, 4.0, 3.0, 9.0]
    assert B.reduce_series(v, "mean") == pytest.approx(4.0)
    assert B.reduce_series(v, "running_max") == 9.0
    assert B.reduce_series(v, "running_min") == 1.0
    assert B.reduce_series(v, "last") == 9.0
    assert B.reduce_series(v, "last3") == pytest.approx((4.0 + 3.0 + 9.0) / 3)
    assert B.reduce_series(v, "last5") == pytest.approx(np.mean(v[-5:]))
    assert B.reduce_series([None, None], "mean") is None
    with pytest.raises(ValueError):
        B.reduce_series(v, "median")


def _traj_rows(n_conv=12, n_steps=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    labels = {}
    for i in range(n_conv):
        conv = "conv%02d" % i
        y = int(i % 2 == 0)
        labels[conv] = y
        for t in range(n_steps):
            rows.append({"arm": "c2kv", "conv_id": conv, "turn": t, "ts": float(t),
                         "entropy": float(rng.normal(2.0 + 1.5 * y, 0.05))})
    return rows, labels


def test_backward_truncation_table_reports_absent_signals_and_base_rate():
    rows, labels = _traj_rows()
    convs = B.group_by_conversation(rows, arm="c2kv")
    table = B.backward_truncation_table(convs, labels)
    assert table["n_trajectories_eligible"] == 12
    assert table["signals_available"]["entropy"] is True
    assert table["signals_available"]["verbal_confidence"] is False
    assert table["base_rate"] == pytest.approx(0.5)
    assert table["base_rate_900_reference"] == pytest.approx(C.BASE_RATE_900)
    absent = [r for r in table["rows"] if r["signal"] == "ppl"]
    assert absent and all(r["auroc"] is None and r["reason"] == "signal absent" for r in absent)
    ent = [r for r in table["rows"] if r["signal"] == "entropy" and r["reducer"] == "last"]
    assert len(ent) == B.N_PROGRESS_POINTS
    assert all(r["feature"] == "lsm_entropy_last" and r["orientation"] == 1 for r in ent)
    assert ent[-1]["auroc"] > 0.9          # entropy is +1 oriented and tracks the label
    assert {r["reducer"] for r in table["rows"] if r["signal"] == "entropy"} == set(B.REDUCERS)


def test_backward_truncation_applies_declared_orientation():
    rows, labels = _traj_rows()
    for r in rows:
        # verbal confidence is HIGH when the trajectory succeeds: orientation -1
        r["verbal_confidence"] = 90.0 - 40.0 * labels[r["conv_id"]]
    convs = B.group_by_conversation(rows, arm="c2kv")
    table = B.backward_truncation_table(convs, labels)
    vc = [r for r in table["rows"]
          if r["signal"] == "verbal_confidence" and r["reducer"] == "last"]
    assert vc[-1]["orientation"] == -1
    assert vc[-1]["auroc"] > 0.9, "an un-oriented score would have scored below 0.5"


def test_short_trajectories_are_excluded_and_counted():
    rows, labels = _traj_rows(n_conv=4, n_steps=12)
    short, short_labels = _traj_rows(n_conv=2, n_steps=5, seed=7)
    short = [dict(r, conv_id="short_" + r["conv_id"]) for r in short]
    labels.update({"short_" + k: v for k, v in short_labels.items()})
    convs = B.group_by_conversation(rows + short, arm="c2kv")
    table = B.backward_truncation_table(convs, labels)
    assert table["n_trajectories_total"] == 6
    assert table["n_trajectories_eligible"] == 4
    assert table["min_steps"] == B.MIN_STEPS_FOR_TRUNCATION == 11


def test_combination_classifier_is_grouped_not_stratified():
    rows, labels = _traj_rows(n_conv=16)
    convs = B.group_by_conversation(rows, arm="c2kv")
    res = B.combination_classifier(convs, labels, progress_index=10)
    assert res["cv"].startswith("grouped")
    assert "stratified" in res["paper_cv"]
    assert res["feature_columns"] == ["entropy__%s" % r for r in B.REDUCERS]
    assert res["n_scored"] > 0 and res["auroc"] is not None
    empty = B.combination_classifier(
        {c: [dict(r, entropy=None) for r in rr] for c, rr in convs.items()},
        labels, progress_index=10)
    assert empty["n_features"] == 0 and "no per-step signal" in empty["reason"]


def test_post_divergence_self_convergence_rate():
    rows = [
        {"arm": "a", "conv_id": "c1", "turn": 0, "match": True},
        {"arm": "a", "conv_id": "c1", "turn": 1, "match": False, "diverged_now": True},
        {"arm": "a", "conv_id": "c1", "turn": 2, "match": False},
        {"arm": "a", "conv_id": "c1", "turn": 3, "match": True},      # converged at k=2
        {"arm": "a", "conv_id": "c2", "turn": 0, "match": False, "diverged_now": True},
        {"arm": "a", "conv_id": "c2", "turn": 1, "match": False, "re_diverged": True},
        {"arm": "a", "conv_id": "c2", "turn": 2, "tracking_lost": True},
    ]
    out = B.post_divergence_convergence(B.group_by_conversation(rows), k_max=3)
    assert out["n_divergence_events"] == 2
    assert out["self_convergence"][1]["rate"] == 0.0
    assert out["self_convergence"][2]["rate"] == pytest.approx(0.5)
    assert out["n_re_diverged_rows"] == 1 and out["n_tracking_lost_rows"] == 1


def test_bench_regenerate_rescue_is_full_rollback_not_s1_prime():
    rows = [
        {"arm": "rec", "conv_id": "c1", "turn": 0, "match": True},
        {"arm": "rec", "conv_id": "c1", "turn": 1, "diverged_now": True,
         "repaired": True, "recovered_action_match": False, "repair_fidelity": True},
        {"arm": "rec", "conv_id": "c2", "turn": 0, "diverged_now": True,
         "repaired": True, "recovered_action_match": False, "repair_fidelity": False},
    ]
    out = B.bench_regenerate_rescue(B.group_by_conversation(rows))
    assert out["face"] == "bench"
    assert out["n_divergence_events"] == 2 and out["n_convs_with_divergence"] == 2
    assert out["n_repair_events"] == 2 and out["n_regenerate_changed_action"] == 2
    assert out["n_regenerate_matched_reference"] == 1
    assert out["full_rollback_conv_rescue_rate"] == pytest.approx(0.5)
    # the estimand must be stamped: this is the full-rollback rung, NOT BTM's S1'
    assert out["is_s1_prime"] is False
    assert out["rung"] == "full_rollback_regeneration"
    assert "UPPER BOUND" in out["note"] and "S1'" in out["note"]
    assert B.bench_regenerate_rescue({})["full_rollback_conv_rescue_rate"] is None
    assert "s1_prime_conv_rate" not in out


def test_bench_regenerate_rescue_reports_absent_columns():
    """A log without the recover columns must read as 'absent', not as 0 rescues."""
    bare = [{"arm": "rec", "conv_id": "c1", "turn": t} for t in range(3)]
    out = B.bench_regenerate_rescue(B.group_by_conversation(bare))
    assert out["n_rows"] == 3
    assert out["columns_missing"] == sorted(B._RECOVER_COLUMNS)
    assert out["full_rollback_conv_rescue_rate"] is None
    present = B.bench_regenerate_rescue(B.group_by_conversation(
        [{"arm": "rec", "conv_id": "c1", "turn": 0, "diverged_now": False}]))
    assert present["columns_present"]["diverged_now"] == 1


def test_truncation_table_records_which_label_it_scored():
    rows, labels = _traj_rows(n_conv=4)
    convs = B.group_by_conversation(rows, arm="c2kv")
    assert B.backward_truncation_table(convs, labels)["label_kind"] == "final_step_cw"
    assert B.backward_truncation_table(convs, labels,
                                       label_kind="task_oracle")["label_kind"] == "task_oracle"


# --------------------------------------------------------------------------
# (D) Context Equilibria -- 2510.07777
# --------------------------------------------------------------------------

def _ar1_series(a, b, n_traj=40, n_steps=25, sigma=0.2, seed=3):
    """Z_{t+1} = Z_t + a + b Z_t + eps  ->  dZ = a + b Z."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_traj):
        z = [float(rng.normal(-a / b, 1.0))]
        for _ in range(n_steps - 1):
            z.append(z[-1] + a + b * z[-1] + float(rng.normal(0.0, sigma)))
        out.append(z)
    return out


def test_equilibrium_fit_recovers_known_a_b():
    series = _ar1_series(a=1.0, b=-0.4)
    fit = B.fit_delta_regression(series)
    assert fit["a"] == pytest.approx(1.0, abs=0.08)
    assert fit["b"] == pytest.approx(-0.4, abs=0.05)
    assert fit["z_star"] == pytest.approx(2.5, abs=0.3)
    assert fit["sigma_eta"] == pytest.approx(0.2, abs=0.05)
    assert fit["n_trajectories"] == 40 and fit["n_pairs"] == 40 * 24
    assert B.fit_delta_regression([[1.0]])["b"] is None


def test_white_noise_pre_gate_fails_on_pure_noise():
    rng = np.random.default_rng(11)
    noise = [rng.normal(5.0, 1.0, size=25).tolist() for _ in range(40)]
    gate = B.equilibrium_gate(noise, reps=200)
    assert gate["fit"]["b"] == pytest.approx(-1.0, abs=0.12)
    assert gate["gate"] == "FAIL"
    assert gate["z_star"] is None and gate["sigma_eta"] is None
    assert "mean reversion" in gate["action_on_fail"]
    assert gate["white_noise_null"]["null_b_ci"][0] < -0.8


def test_white_noise_pre_gate_passes_on_real_dynamics():
    gate = B.equilibrium_gate(_ar1_series(a=1.0, b=-0.3), reps=200)
    assert gate["gate"] == "PASS"
    assert gate["b_ci_excludes_minus_one"] and gate["b_outside_white_noise_null"]
    assert gate["z_star"] == pytest.approx(1.0 / 0.3, abs=0.5)


def test_fire_deviation_and_kappa_selected_on_inner_folds():
    assert B.fire_deviation(5.0, 2.0, 1.0, 4.0) is False   # 3.0 <= 4 * 1.0
    assert B.fire_deviation(5.0, 2.0, 1.0, 2.0) is True    # 3.0 >  2 * 1.0
    assert B.fire_deviation(None, 2.0, 1.0, 1.0) is None
    series, labels = {}, {}
    for i in range(24):
        y = int(i % 2 == 0)
        base = _ar1_series(a=1.0, b=-0.4, n_traj=1, n_steps=12, seed=100 + i)[0]
        if y:
            base = base[:-1] + [base[-1] + 6.0]      # positives end far above equilibrium
        series["c%02d" % i] = base
        labels["c%02d" % i] = y
    sel = B.select_kappa(series, labels, kappa_grid=(0.5, 1.0, 2.0, 8.0))
    assert sel["kappa"] in (0.5, 1.0, 2.0, 8.0)
    # the criterion is precision at a fixed fire rate, NOT a ranking metric on the
    # two-valued fired indicator
    assert sel["criterion"] == "precision at a fixed fire rate"
    assert "inner_ap" not in sel and "per_kappa_inner_ap" not in sel
    assert sel["inner_precision"] is not None and sel["inner_precision"] > 0.6
    assert sel["chance_precision"] == pytest.approx(0.5)   # evaluation-frame prevalence
    assert sel["inner_fire_rate"] is not None
    assert sel["deviation_auroc"] is not None and sel["deviation_auroc"] > 0.6
    assert "inner folds" in sel["note"]
    assert B.select_kappa({"a": [1.0]}, {"a": 1})["kappa"] is None


def test_select_kappa_respects_the_declared_fire_rate():
    """A tighter target fire rate must not be met by a kappa that fires more."""
    series, labels = {}, {}
    for i in range(24):
        y = int(i % 4 == 0)
        base = _ar1_series(a=1.0, b=-0.4, n_traj=1, n_steps=12, seed=300 + i)[0]
        if y:
            base = base[:-1] + [base[-1] + 6.0]
        series["c%02d" % i] = base
        labels["c%02d" % i] = y
    grid = (0.25, 0.5, 1.0, 2.0)
    sel = B.select_kappa(series, labels, kappa_grid=grid, target_fire_rate=0.25)
    assert sel["target_fire_rate"] == pytest.approx(0.25)
    assert sel["target_fire_rate_source"] == "caller-supplied"
    assert sel["kappa"] in grid
    if sel["constraint_met"]:
        assert sel["inner_fire_rate"] <= 0.25 + 1e-12
    # the default target is the evaluation-frame prevalence, stated not implied
    default = B.select_kappa(series, labels, kappa_grid=grid)
    assert default["target_fire_rate"] == pytest.approx(0.25)
    assert default["target_fire_rate_source"] == "label prevalence of the supplied groups"
    assert default["chance_precision"] == pytest.approx(0.25)


def test_fit_delta_regression_matches_closed_form_ols():
    """dZ_t = a + b Z_t by OLS, Z* = -a/b, sigma_eta = residual sd (2510.07777 11.2)."""
    series = [[1.0, 3.0, 4.0, 4.5, 7.0, 6.0]]
    z = np.array([1.0, 3.0, 4.0, 4.5, 7.0])
    dz = np.array([2.0, 1.0, 0.5, 2.5, -1.0])
    b_hat = float(((z - z.mean()) * (dz - dz.mean())).sum() / ((z - z.mean()) ** 2).sum())
    a_hat = float(dz.mean() - b_hat * z.mean())
    resid = dz - (a_hat + b_hat * z)
    fit = B.fit_delta_regression(series)
    assert fit["a"] == pytest.approx(a_hat)
    assert fit["b"] == pytest.approx(b_hat)
    assert fit["z_star"] == pytest.approx(-a_hat / b_hat)
    assert fit["sigma_eta"] == pytest.approx(float(np.sqrt(resid @ resid / 3)))
    assert fit["n_pairs"] == 5 and fit["n_trajectories"] == 1


def test_equilibrium_cli_fits_on_the_cc_pool_only(tmp_path, capsys):
    """The digest fixes the fit pool: dZ = a + bZ on the C->C training pool."""
    rows = []
    for i in range(8):
        conv = "c%d" % i
        y = int(i % 2 == 0)
        for t in range(6):
            # positives carry a wildly different ratio, so including them moves the fit
            rows.append({"arm": "c2kv", "conv_id": conv, "turn": t, "ts": float(t),
                         "gist_tokens": 100.0,
                         "original_tokens": 800.0 + (400.0 * y * t)})
    log = tmp_path / "proxy.jsonl"
    log.write_text(chr(10).join(json.dumps(r) for r in rows), encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"c%d" % i: int(i % 2 == 0) for i in range(8)}),
                      encoding="utf-8")

    out_all = tmp_path / "all.json"
    B.main(["equilibrium", "--proxy-log", str(log), "--reps", "50",
            "--out", str(out_all)])
    capsys.readouterr()
    pool_all = json.loads(out_all.read_text(encoding="utf-8"))["fit_pool"]
    assert pool_all["labels_supplied"] is False
    assert pool_all["n_episodes_fitted"] == 8
    assert "later be scored on" in pool_all["warning"]

    out_cc = tmp_path / "cc.json"
    B.main(["equilibrium", "--proxy-log", str(log), "--reps", "50",
            "--labels", str(labels), "--out", str(out_cc)])
    capsys.readouterr()
    cc = json.loads(out_cc.read_text(encoding="utf-8"))
    assert cc["fit_pool"]["labels_supplied"] is True
    assert cc["fit_pool"]["n_episodes_fitted"] == 4
    assert cc["fit_pool"]["n_skipped_positive"] == 4
    assert cc["fit_pool"]["warning"] is None
    # the two pools are genuinely different fits, so the pool choice is load-bearing
    assert (json.loads(out_all.read_text(encoding="utf-8"))["fit"]["b"]
            != cc["fit"]["b"])
    # kappa is wired: unlabelled -> null with a loud reason, labelled -> selected
    all_out = json.loads(out_all.read_text(encoding="utf-8"))
    assert all_out["kappa_selection"]["kappa"] is None
    assert "MISSING INPUT" in all_out["kappa_selection"]["reason"]
    assert all_out["fire_rule"]["rule"] is None
    assert "criterion" in cc["kappa_selection"]
    assert cc["kappa_selection"]["criterion"] == "precision at a fixed fire rate"
    # an incomplete rule always carries the reason it is incomplete
    assert cc["fire_rule"]["rule"] is not None or cc["fire_rule"]["reason"]


def test_equilibrium_cli_completes_the_fire_rule_on_real_dynamics(tmp_path, capsys):
    """With a PASSing gate and labels, the CLI emits kappa and the full fire rule."""
    rows = []
    labels_map = {}
    for i in range(24):
        conv = "c%02d" % i
        y = int(i % 2 == 0)
        z = _ar1_series(a=1.0, b=-0.4, n_traj=1, n_steps=14, seed=700 + i)[0]
        if y:
            z = z[:-1] + [z[-1] + 6.0]
        labels_map[conv] = y
        for t, zt in enumerate(z):
            # ratio = original / gist, so a constant gist makes ratio track z_t
            rows.append({"arm": "c2kv", "conv_id": conv, "turn": t,
                         "gist_tokens": 100.0, "original_tokens": 100.0 * float(zt)})
    log = tmp_path / "proxy.jsonl"
    log.write_text(chr(10).join(json.dumps(r) for r in rows), encoding="utf-8")
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(labels_map), encoding="utf-8")
    out = tmp_path / "eq.json"
    assert B.main(["equilibrium", "--proxy-log", str(log), "--labels", str(labels),
                   "--reps", "200", "--kappa-grid", "0.5,1.0,2.0",
                   "--out", str(out)]) == 0
    capsys.readouterr()
    eq = json.loads(out.read_text(encoding="utf-8"))
    assert eq["gate"] == "PASS"
    sel = eq["kappa_selection"]
    assert sel["kappa"] in (0.5, 1.0, 2.0)
    assert sel["n_groups"] == 24 and sel["chance_precision"] == pytest.approx(0.5)
    assert eq["fire_rule"]["rule"] == "fire iff Z_t - Z* > kappa * sigma_eta"
    assert eq["fire_rule"]["threshold"] == pytest.approx(
        eq["z_star"] + sel["kappa"] * eq["sigma_eta"])
    # the outer fit still saw only the C->C pool
    assert eq["fit_pool"]["n_episodes_fitted"] == 12
    assert eq["fit_pool"]["n_skipped_positive"] == 12


def test_equilibrium_reports_the_four_component_bench_face(tmp_path, capsys):
    """A bench proxy row carries 4 of the 6 Z^pre components -- say so, don't impute."""
    rows = [{"arm": "c2kv", "conv_id": "c0", "turn": t, "gist_tokens": 100.0,
             "original_tokens": 800.0, "dropped_docs": [1], "repair_frame": 3.0}
            for t in range(6)]
    log = tmp_path / "proxy.jsonl"
    log.write_text(chr(10).join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "eq.json"
    assert B.main(["equilibrium", "--proxy-log", str(log), "--reps", "20",
                   "--out", str(out)]) == 0
    capsys.readouterr()
    cov = json.loads(out.read_text(encoding="utf-8"))["z_pre_coverage"]
    assert cov["n_components_declared"] == 6
    assert cov["n_components_present"] == 4
    assert cov["components_absent_from_this_log"] == ["hybrid_tail", "kept_history_tokens"]
    assert cov["bench_face_expected"] == list(B.BENCH_LOG_Z_PRE_COMPONENTS)
    for name in B.BATTERY_ONLY_Z_PRE_COMPONENTS:
        assert cov["components"][name]["battery_only_on_the_bench_face"] is True


def test_equilibrium_aborts_when_the_chosen_z_is_absent(tmp_path, capsys):
    rows = [{"arm": "c2kv", "conv_id": "c0", "turn": t, "gist_tokens": 100.0,
             "original_tokens": 800.0} for t in range(6)]
    log = tmp_path / "proxy.jsonl"
    log.write_text(chr(10).join(json.dumps(r) for r in rows), encoding="utf-8")
    rc = B.main(["equilibrium", "--proxy-log", str(log), "--z", "hybrid_tail",
                 "--reps", "20"])
    printed = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert printed["abort"] is True and "MISSING INPUT" in printed["reason"]
    assert "hybrid_tail" in printed["z_pre_coverage"]["components_absent_from_this_log"]


def test_truncation_table_flags_the_100pct_column_as_post_generation():
    rows, labels = _traj_rows(n_conv=4)
    table = B.backward_truncation_table(B.group_by_conversation(rows, arm="c2kv"), labels)
    assert "POST-generation" in table["causality_note"]
    assert table["rows"][-1]["progress_pct"] == 100


def test_z_pre_components_are_prefix_only_and_none_when_absent():
    row = {"gist_tokens": 100, "original_tokens": 800, "dropped_docs": [1, 2],
           "kept_history_tokens": 512, "hybrid_top_k": 3}
    z = B.z_pre_components(row)
    assert z["compression_ratio"] == pytest.approx(8.0)
    assert z["dropped_docs_n"] == 2.0 and z["hybrid_tail"] == 3.0
    assert z["position_gap"] is None                  # not back-fillable: stays None
    assert set(z) == set(B.Z_PRE_COMPONENTS)
    assert all(v is None for v in B.z_pre_components({}).values())
    # battery-shaped row: doc_tokens is the ORIGINAL size and
    # compressed_history_tokens is the POST-compression count -- confusing the two
    # would report a ratio of 1.0 on every frozen row.
    battery = {"gist_tokens": 70, "compressed_history_tokens": 70, "doc_tokens": 554,
               "kept_history_tokens": 554, "actual_compression_ratio": 7.9143}
    zb = B.z_pre_components(battery)
    assert zb["compression_ratio"] == pytest.approx(7.9143)
    no_measured = dict(battery)
    no_measured.pop("actual_compression_ratio")
    assert B.z_pre_components(no_measured)["compression_ratio"] == pytest.approx(554 / 70)


# --------------------------------------------------------------------------
# feature frames + frozen-asset smoke
# --------------------------------------------------------------------------

def test_feature_writer_rejects_label_leakage(tmp_path):
    rows = [{"qid": "a:1", "session_id": "a", "irb_s_t_r_weighted": 0.5,
             "tool_name_match": True}]
    with pytest.raises(ValueError):
        C.write_features_jsonl(tmp_path / "f.jsonl", rows, context="leak test")
    ok = [{"qid": "a:1", "session_id": "a", "irb_s_t_r_weighted": None}]
    assert C.write_features_jsonl(tmp_path / "g.jsonl", ok, context="ok test") == 1
    written = json.loads((tmp_path / "g.jsonl").read_text(encoding="utf-8").strip())
    assert written["irb_s_t_r_weighted"] is None      # null kept, never imputed


@pytest.mark.skipif(not (ROOT / "results/bdf_pilot/d_r2/battery_c2kv.jsonl").exists(),
                    reason="frozen battery not present")
def test_battery_z_rows_on_frozen_assets(tmp_path):
    frame = C.FrozenAssets(ROOT).load()
    rows = B.battery_z_rows(frame)
    assert len(rows) == 900
    ratios = [r["ceq_compression_ratio"] for r in rows
              if r["ceq_compression_ratio"] is not None]
    assert ratios and min(ratios) > 1.5, "ratio 1.0 means the wrong token field was read"
    table = B.declared_orientations()
    feature_cols = {k for r in rows for k in r if k.startswith("ceq_")}
    assert feature_cols <= set(table), sorted(feature_cols - set(table))
    n = C.write_features_jsonl(tmp_path / "z.jsonl", rows, context="battery z")
    assert n == 900
    assert all(r["form"] == "per_step" for r in rows)
    # the S0 twin on the full arm is structurally degenerate and must say so
    s0 = B.z_pre_s0_report(frame)
    assert s0["arm"] == "full"
    assert set(s0["components"]) == {"ceq_" + c for c in B.Z_PRE_COMPONENTS}
    assert s0["components"]["ceq_gist_tokens"]["usable_as_s0"] is False
    assert "structurally constant" in s0["note"]


@pytest.mark.skipif(not (ROOT / "results/bdf_pilot/d_r2/battery_c2kv.jsonl").exists(),
                    reason="frozen battery not present")
def test_classes_cli_shouts_when_the_flip_table_is_absent(tmp_path, capsys):
    """No flip table on this box: every row is UNMEASURED, not measured-unsolvable."""
    out = tmp_path / "classes.json"
    assert B.main(["classes", "--root", str(ROOT), "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    n = printed["census"]["n_rows"]
    assert printed["census"]["n_flip_table_missing"] == n
    assert printed["provenance"]["n_rows_without_flip_table"] == n
    assert "MISSING INPUT" in printed["provenance"]["flip_table_note"]
    # and the same fact survives into the frozen artefact
    frozen = json.loads(out.read_text(encoding="utf-8"))
    assert all(v["flip_table_missing"] for v in frozen["classes"].values())


@pytest.mark.skipif(not (ROOT / "results/bdf_pilot/d_r2/battery_c2kv.jsonl").exists(),
                    reason="frozen battery not present")
def test_dstar_cli_shouts_when_the_sidecar_has_no_dropped_text(tmp_path, capsys):
    frame = C.FrozenAssets(ROOT).load()
    qids = [r["qid"] for r in frame.trigger_subset()][:5]
    side = tmp_path / "sidecar.jsonl"
    side.write_text(chr(10).join(json.dumps({
        "qid": q,
        "docs": [_doc("get_order", {"order_id": "a1"})],   # KEPT blocks only
        "dropped_docs": [3, 4],                            # post-split indices
    }) for q in qids), encoding="utf-8")
    part = tmp_path / "part.json"
    part.write_text(json.dumps(_partition()), encoding="utf-8")
    out = tmp_path / "feats.jsonl"
    assert B.main(["dstar", "--root", str(ROOT), "--sidecar", str(side),
                   "--partition", str(part), "--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["n_rows_with_dropped_blocks"] == 5
    assert printed["n_rows_missing_dropped_doc_texts"] == 5
    assert printed["dropped_side_available"] is False
    assert "with_dropped_text" in printed["note"]
    assert printed["n_rows_missing_sidecar"] == printed["written"] - 5
    got = {json.loads(line)["qid"]: json.loads(line)
           for line in out.read_text(encoding="utf-8").splitlines()}
    for q in qids:
        assert got[q]["irb_s_t_dropped_only"] is None       # unavailable, not 0
        assert got[q]["irb_s_t_scope"] == "kept_only_partial"
        assert got[q]["irb_n_dropped_blocks"] == 2
    # rows with no sidecar at all carry the full null schema, not a ragged row
    missing = [r for q, r in got.items() if q not in set(qids)]
    assert missing and all(set(B.DSTAR_FEATURE_KEYS) <= set(r) for r in missing)
    assert all(r[k] is None for r in missing for k in B.DSTAR_FEATURE_KEYS)


@pytest.mark.skipif(not (ROOT / "results/bdf_pilot/d_r2/battery_c2kv.jsonl").exists(),
                    reason="frozen battery not present")
def test_l1_cut_and_cap_caveat_on_frozen_trigger_subset():
    frame = C.FrozenAssets(ROOT).load()
    sub = frame.trigger_subset()
    assert len(sub) == 161
    caveat = B.cap_caveat([r["censored_at_cap"] for r in sub], frame.cap_tokens())
    assert caveat["cap_tokens"] == 128
    assert caveat["cap_rate"] is not None
    vis = [B.l1_visibility(frame.c2kv_by_qid[r["qid"]].get("prediction", ""))["l1_visible"]
           for r in sub]
    assert len(vis) == 161 and set(vis) <= {True, False, None}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", ["classes", "l1-cut", "partition-check", "triples",
                                 "dstar", "truncation", "selfconv", "equilibrium",
                                 "battery-z"])
def test_cli_help(cmd, capsys):
    with pytest.raises(SystemExit) as exc:
        B.main([cmd, "--help"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip()


def test_cli_partition_check_roundtrip(tmp_path, capsys):
    path = tmp_path / "p.json"
    path.write_text(json.dumps(_partition()), encoding="utf-8")
    assert B.main(["partition-check", "--partition", str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] and report["n_tools"] == 4
