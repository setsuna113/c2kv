# -*- coding: utf-8 -*-
"""Tests for the §4.12 self-report / judge unit (torch-free)."""

import json

import numpy as np
import pytest

import t34_common as C
import t34_judge as J
import t34_selfreport as SR


# --------------------------------------------------------------------------
# (A) VISTA parsers — messy free text
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expect", [
    ("12000", 12000),
    ("About 12,000 tokens I think.", 12000),
    ("roughly 3k tokens", 3000),
    ("I'd estimate approximately 850 tokens in total.", 850),
    ("Earlier turn 3 holds about 500 tokens.", 500),      # token-adjacent wins
    ("<think>hmm maybe 99</think>Answer: 640", 640),
    ("~1.2k tokens", 1200),
    ("I cannot know that.", None),
    ("", None),
])
def test_parse_integer_answer(text, expect):
    assert SR.parse_integer_answer(text) == expect


@pytest.mark.parametrize("text,n,expect", [
    ("100, 200, 50, 400", 4, [100, 200, 50, 400]),
    # turn indices must not be mistaken for sizes
    ("turn 3: 100 tokens, turn 5: 200 tokens", 2, [100, 200]),
    ("Earlier turn 1 is about 120 tokens and earlier turn 2 about 80 tokens.",
     2, [120, 80]),
    ("roughly 1.2k, 800", 2, [1200, 800]),
    ("I cannot know that.", 4, None),          # unparsed -> None, not padded
    ("120", 4, None),                          # too few -> the row is unparsed
    ("10, 20, 30, 40, 50", 4, [10, 20, 30, 40]),   # declared first-n precedence
])
def test_parse_integer_list_answer_is_per_block(text, n, expect):
    assert SR.parse_integer_list_answer(text, n) == expect


@pytest.mark.parametrize("text,expect", [
    ("A", "A"),
    ("B.", "B"),
    ("I think option b is larger.", "B"),
    ("The answer is A, earlier turn 2.", "A"),
    ("They look the same to me.", None),
])
def test_parse_choice_answer(text, expect):
    assert SR.parse_choice_answer(text) == expect


@pytest.mark.parametrize("text,detected,block", [
    ("Yes, turn 3 was condensed.", True, 2),
    ("no, nothing was removed", False, None),
    ("Yes - block #1 appears to be missing.", True, 0),
    ("Yes, 4", True, 3),
    ("I am not able to tell.", None, None),
])
def test_parse_loss_answer(text, detected, block):
    got = SR.parse_loss_answer(text)
    assert got["detected"] is detected
    assert got["block"] == block


def test_parse_selfcheck_yes_no_and_score():
    got = SR.parse_selfcheck("Yes, it is consistent. Score: 0.85")
    assert got["consistent"] is True and got["score"] == pytest.approx(0.85)
    got = SR.parse_selfcheck("No. 0.1")
    assert got["consistent"] is False and got["score"] == pytest.approx(0.1)
    got = SR.parse_selfcheck("about 70%")
    assert got["score"] == pytest.approx(0.7)
    got = SR.parse_selfcheck("hmm")
    assert got["consistent"] is None and got["score"] is None
    assert got["parse_ok"] is False


# --------------------------------------------------------------------------
# scorers: VISTA's own arithmetic
# --------------------------------------------------------------------------

def test_median_relative_error_arithmetic_and_missing_handling():
    # rel errors 0.5, 0.0, 1.0 -> median 0.5; one unparsed, one bad truth
    res = SR.median_relative_error([(50, 100), (100, 100), (200, 100),
                                    (None, 100), (10, 0)])
    assert res["median_relative_error"] == pytest.approx(0.5)
    assert res["n_scored"] == 3
    assert res["n_unparsed"] == 1
    assert res["n_undefined_truth"] == 1


def test_pairwise_hard_subset_accuracy_matches_vista_rule():
    items = [
        ("A", 100, 60),    # hard (ratio 1.67), correct
        ("B", 100, 60),    # hard, wrong
        ("A", 1000, 10),   # easy (ratio 100), correct
        (None, 100, 90),   # unparsed
        ("A", 50, 50),     # tie -> excluded
    ]
    res = SR.pairwise_hard_subset_accuracy(items)
    assert res["n_all"] == 3 and res["accuracy_all"] == pytest.approx(2 / 3)
    assert res["n_hard_subset"] == 2
    assert res["accuracy_hard_subset"] == pytest.approx(0.5)
    assert res["n_unparsed"] == 1 and res["n_ties_excluded"] == 1


def test_p4_detection_reports_degeneracy_instead_of_a_bogus_p():
    truth = {f"s:{i}": True for i in range(10)}       # constant truth
    detected = {f"s:{i}": True for i in range(10)}
    res = SR.p4_detection_score(detected, truth)
    assert res["degenerate"] is True
    assert res["p_vs_base_rate"] is None
    assert res["accuracy"] == pytest.approx(1.0)


def test_p4_detection_beats_base_rate_when_truth_varies():
    truth = {f"s:{i}": (i % 2 == 0) for i in range(40)}
    detected = dict(truth)                            # perfect
    res = SR.p4_detection_score(detected, truth)
    assert res["degenerate"] is False
    assert res["majority_base_rate"] == pytest.approx(0.5)
    assert res["p_vs_base_rate"] < 0.05


def test_p4_localisation_uses_the_25pct_floor_and_counts_abstentions():
    reported = {"s:1": 0, "s:2": 3, "s:3": None}
    truth = {"s:1": [0], "s:2": [1], "s:3": [2], "s:4": []}
    tab = SR.p4_localisation_table(reported, truth)
    assert tab["floor"] == 0.25
    assert tab["n"] == 3 and tab["hits"] == 1
    assert tab["abstained"] == 1          # s:3 counted in n, reported separately
    assert tab["n_undefined_excluded"] == 1
    assert tab["s_at_k"] == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# ledger: content, placement, leak scan
# --------------------------------------------------------------------------

def test_ledger_builder_refuses_non_s8_fields():
    ok = SR.build_ledger_line({"doc_chunks": 7, "kept_history_tokens": 1420,
                               "gist_tokens": 178, "actual_compression_ratio": 8.0,
                               "dropped_docs": [1, 2], "hybrid_top_k": 4,
                               "repair_frame_delta": 0})
    assert "7" in ok and "1420" in ok and "178" in ok
    assert "2, 3" in ok           # dropped blocks rendered 1-based
    with pytest.raises(ValueError):
        SR.build_ledger_line({"target_tool_name": "book_flight"})
    with pytest.raises(ValueError):
        SR.build_ledger_line({"tool_name_match": True})


def test_ledger_sits_after_history_and_before_the_query():
    history = [{"role": "user", "content": "h1"},
               {"role": "assistant", "content": "h2"},
               {"role": "user", "content": "h3"}]
    ledger = SR.build_ledger_line({"doc_chunks": 3, "gist_tokens": 40})
    query = SR.build_probe_query("P1")
    msgs = SR.assemble_probe_messages(history, query, ledger_line=ledger)
    pos = SR.ledger_position(msgs, ledger)
    assert pos["index"] == len(history)
    assert pos["after_all_history"] is True
    assert pos["before_query"] is True
    assert msgs[-1]["content"] == query
    # and the history prefix is untouched, byte for byte
    assert msgs[: len(history)] == history


def test_current_messages_for_probe_is_the_tail_only():
    tail = SR.current_messages_for_probe("q?", ledger_line="L", role="system")
    assert [m["role"] for m in tail] == ["system", "user"]
    assert SR.current_messages_for_probe("q?") == [{"role": "user", "content": "q?"}]


def test_probe_queries_do_not_leak_context_state():
    for probe, blocks in (("P1", None), ("P2", [0, 1]), ("P3", [0, 2]), ("P4", None)):
        q = SR.build_probe_query(probe, blocks=blocks)
        assert SR.scan_prompt_for_state_leak(q) == []
    leaky = "Your history holds 1420 tokens at compression ratio 8"
    assert SR.scan_prompt_for_state_leak(leaky)


def test_block_indices_render_one_based():
    q = SR.build_probe_query("P3", blocks=[0, 1])
    assert "turn 1" in q and "turn 2" in q


def test_sample_blocks_is_deterministic_per_qid():
    a = SR.sample_blocks("sess:3", 10, 4)
    b = SR.sample_blocks("sess:3", 10, 4)
    assert a == b and len(a) == 4 and a == sorted(a)
    assert set(a) <= set(range(10))
    # a different qid draws its own sample; over many qids they are not all equal
    others = {tuple(SR.sample_blocks(f"sess:{i}", 10, 4)) for i in range(20)}
    assert len(others) > 1
    assert SR.sample_blocks("sess:3", 2, 4) == [0, 1]      # k capped at n_docs
    assert SR.sample_blocks("sess:3", 0, 4) == []


# --------------------------------------------------------------------------
# plan / gate / cost
# --------------------------------------------------------------------------

def _sidecar_row(qid, lens, dropped=()):
    return {"qid": qid, "session_id": qid.rsplit(":", 1)[0],
            "docs": ["d" * n for n in lens], "doc_lengths": list(lens),
            "dropped_docs": list(dropped), "query": "do the thing",
            "tools": [], "system_prompt": "sys", "kept_history_tokens": sum(lens)}


def _battery_row(qid, **kw):
    row = {"qid": qid, "session_id": qid.rsplit(":", 1)[0], "doc_chunks": 4,
           "kept_history_tokens": 400, "gist_tokens": 50,
           "actual_compression_ratio": 8.0, "hybrid_top_k": None,
           "compressed_history_tokens": 50, "prediction": "<tool_call>{}</tool_call>"}
    row.update(kw)
    return row


def test_probe_plan_is_deterministic_and_carries_truths():
    sc = [_sidecar_row("s1:2", [100, 200, 50, 400], dropped=[1])]
    bat = {"s1:2": _battery_row("s1:2")}
    rows_a = SR.probe_plan_rows(sc, bat, condition="minus_ledger")
    rows_b = SR.probe_plan_rows(sc, bat, condition="minus_ledger")
    assert [r["query"] for r in rows_a] == [r["query"] for r in rows_b]
    assert {r["probe"] for r in rows_a} == set(SR.PROBE_IDS)
    p4 = next(r for r in rows_a if r["probe"] == "P4")
    assert p4["truth"]["lost_dropped"] is True
    assert p4["truth"]["dropped_docs"] == [1]
    p3 = next(r for r in rows_a if r["probe"] == "P3")
    assert p3["truth"]["size_a"] in (100, 200, 50, 400)
    assert all(r["ledger"] is None for r in rows_a)
    plus = SR.probe_plan_rows(sc, bat, condition="plus_ledger")
    assert all(r["ledger"] and r["ledger"].startswith("Context state") for r in plus)


def test_score_probe_file_end_to_end_on_synthetic_generations():
    rows = [
        {"qid": "s1:1", "probe": "P1", "condition": "minus_ledger",
         "generation": "about 200 tokens",
         "truth": {"total_tokens": 400, "compressed_history_tokens": 50},
         "generate_sec": 1.0},
        {"qid": "s1:1", "probe": "P2", "condition": "minus_ledger",
         # per-block answers: rel errors |50-100|/100 = 0.5 and
         # |300-200|/200 = 0.5 -> median 0.5 (NOT the sum 350 vs 300)
         "generation": "50 tokens, 300 tokens",
         "truth": {"blocks": [0, 1], "block_tokens": [100, 200]},
         "generate_sec": 1.0},
        {"qid": "s1:1", "probe": "P3", "condition": "minus_ledger",
         "generation": "B", "truth": {"blocks": [0, 1], "size_a": 100,
                                      "size_b": 150}, "generate_sec": 1.0},
        {"qid": "s1:1", "probe": "P4", "condition": "minus_ledger",
         "generation": "Yes, turn 2 was condensed.",
         "truth": {"lost_dropped": True, "lost_any": True, "dropped_docs": [1]},
         "generate_sec": 1.0},
    ]
    res = SR.score_probe_file(rows)
    assert res["P1"]["median_relative_error"] == pytest.approx(0.5)
    # both P1 readings are scored and printed; neither is chosen post hoc
    assert res["P1_vs_compressed_history_tokens"]["median_relative_error"] == \
        pytest.approx(3.0)                                   # |200-50|/50
    assert "pre-compression" in res["P1"]["truth_column"]
    assert res["P2"]["median_relative_error"] == pytest.approx(0.5)
    # the P2 estimand is PER BLOCK (2606.30005 §4), so one request with two
    # sampled blocks contributes two scored pairs, not one
    assert res["P2"]["n_scored"] == 2
    assert res["P2"]["n_requests"] == 1
    assert res["P2"]["estimand"] == "per_block_relative_error"
    assert res["P3"]["accuracy_hard_subset"] == pytest.approx(1.0)
    assert res["P4_localisation"]["hits"] == 1
    assert res["parse_rate"] == pytest.approx(1.0)
    assert res["generate_sec_total"] == pytest.approx(4.0)


def test_the_two_turn_numbering_frames_are_named_in_every_rendered_string():
    """``docs`` is kept-only while ``dropped_docs`` indexes the post-split
    history, so a +ledger prompt carries BOTH frames and must label them."""
    p2 = SR.build_probe_query("P2", blocks=[0, 1])
    p3 = SR.build_probe_query("P3", blocks=[0, 1])
    p4 = SR.build_probe_query("P4")
    assert SR.VISIBLE_FRAME_PREAMBLE in p2 and SR.VISIBLE_FRAME_PREAMBLE in p3
    assert "still see" in p2 and "still see" in p3
    assert "original conversation" in p4
    ledger = SR.build_ledger_line({"dropped_docs": [1, 2]})
    assert "original conversation" in ledger and "2, 3" in ledger
    # the frames are declared, not inferred from the wording
    assert SR.NUMBERING_FRAMES["P2"] == "visible_kept_order"
    assert SR.NUMBERING_FRAMES["P4"] == "original_post_split_order"
    assert SR.NUMBERING_FRAMES["ledger_dropped_docs"] == \
        SR.NUMBERING_FRAMES["P4"]
    # and the +ledger prompt no longer says "turn 2 is gone" and "size of turn
    # 2?" in the same breath without saying which 2 it means
    for q in (p2, p3):
        assert SR.scan_prompt_for_state_leak(q) == []


def test_p2_and_p3_denominators_are_broken_out_per_request():
    """sample_blocks caps k at the number of VISIBLE blocks, so a short row
    contributes fewer than P2_N_BLOCKS pairs and no P3 request at all."""
    rows = [
        {"qid": "s1:1", "probe": "P2", "generation": "50 tokens, 300 tokens",
         "truth": {"blocks": [0, 1], "block_tokens": [100, 200]}},
        {"qid": "s1:2", "probe": "P2", "generation": "120 tokens",
         "truth": {"blocks": [0], "block_tokens": [100]}},
        {"qid": "s1:1", "probe": "P3", "generation": "A",
         "truth": {"blocks": [0, 1], "size_a": 200, "size_b": 100}},
    ]
    res = SR.score_probe_file(rows)
    assert res["P2"]["n_requests"] == 2
    assert res["P2"]["blocks_per_request_histogram"] == {"1": 1, "2": 1}
    assert res["P2"]["n_blocks_asked"] == 3
    assert res["P2"]["n_scored"] == 3
    assert res["P2"]["blocks_per_request_requested"] == SR.P2_N_BLOCKS
    # P3 is asked of fewer qids than P2, and says so
    assert res["P3"]["n_requests"] == 1 and res["P3"]["n_qids"] == 1


def test_pooled_p4_refuses_a_single_arm_and_scores_the_pooled_frame():
    def _p4(qid, arm, gen, lost):
        return {"qid": qid, "probe": "P4", "arm": arm, "generation": gen,
                "truth": {"lost_any": lost, "lost_dropped": lost,
                          "dropped_docs": [0] if lost else []}}

    c2kv = [_p4(f"s{i}:1", "c2kv", "yes, turn 1", True) for i in range(12)]
    # one arm only: the truth is constant, so no number is produced
    solo = SR.pooled_p4_detection(c2kv)
    assert solo["available"] is False and solo["detection"] is None
    assert "--arm full" in solo["reason"]
    assert solo["n_rows_by_arm"] == {"c2kv": 12}
    # the within-file scorer says the same thing where a reader will see it
    single = SR.score_probe_file(c2kv)
    assert single["P4_detection_any_condensed"]["degenerate"] is True
    assert "score-p4-pooled" in single["P4_detection_any_condensed"]["pooled_scorer"]
    # the full arm supplies the negatives, and each row brings its own truth
    full = [_p4(f"s{i}:1", "full", "no, nothing was removed", False)
            for i in range(12)]
    pooled = SR.pooled_p4_detection(c2kv + full)
    assert pooled["available"] is True
    assert pooled["n_rows_by_arm"] == {"c2kv": 12, "full": 12}
    det = pooled["detection"]
    assert det["degenerate"] is False and det["n"] == 24
    assert det["accuracy"] == pytest.approx(1.0)
    assert det["p_vs_base_rate"] < 0.05
    # a full arm that really did drop blocks is scored as it is, not as False
    full_dropped = [_p4("s0:1", "full", "yes, turn 1", True)] + full[1:]
    assert SR.pooled_p4_detection(c2kv + full_dropped)["detection"]["n"] == 24


def test_ledger_gate_fails_closed_when_the_model_is_blind():
    blind = {"parse_rate": 0.9,
             "P1": {"median_relative_error": 0.7},
             "P2": {"median_relative_error": 0.8},
             "P4_detection": {"p_vs_base_rate": 0.4}}
    gate = SR.ledger_gate(blind)
    assert gate["run_plus_ledger"] is False
    assert "not worth an extra LLM call" in gate["verdict"]
    alive = dict(blind, P4_detection={"p_vs_base_rate": 0.001})
    assert SR.ledger_gate(alive)["run_plus_ledger"] is True
    unusable = dict(alive, parse_rate=0.1)
    assert SR.ledger_gate(unusable)["run_plus_ledger"] is False


def test_require_p4_gate_refuses_without_a_result_file(tmp_path):
    with pytest.raises(RuntimeError, match="gated on the VISTA P4"):
        SR.require_p4_gate(None)
    with pytest.raises(RuntimeError, match="gated on the VISTA P4"):
        SR.require_p4_gate(str(tmp_path / "missing.json"))
    failed = tmp_path / "p4.json"
    failed.write_text(json.dumps({"p4_gate": {"passed": False, "reasons": ["x"]}}),
                      encoding="utf-8")
    with pytest.raises(RuntimeError, match="P4 gate FAILED"):
        SR.require_p4_gate(str(failed))
    passed = tmp_path / "p4ok.json"
    passed.write_text(json.dumps({"p4_gate": {"passed": True}}), encoding="utf-8")
    assert SR.require_p4_gate(str(passed))["gated"] is True
    assert SR.require_p4_gate(None, force=True)["forced"] is True


def test_require_p4_gate_falls_back_to_the_p4_only_gate_not_the_ledger_gate(tmp_path):
    """A score file whose +ledger gate passed on the SIZE probes alone must NOT
    license the selfcheck arm: the card licenses it on P4 specifically."""
    size_only = {"parse_rate": 0.95,
                 "P1": {"median_relative_error": 0.05},
                 "P2": {"median_relative_error": 0.05},
                 "P4_detection": {"p_vs_base_rate": 0.9, "degenerate": False}}
    assert SR.ledger_gate(size_only)["run_plus_ledger"] is True   # spend +ledger
    assert SR.p4_selfcheck_gate(size_only)["passed"] is False     # but no arm
    path = tmp_path / "vista.json"
    path.write_text(json.dumps(size_only), encoding="utf-8")
    with pytest.raises(RuntimeError, match="P4 gate FAILED"):
        SR.require_p4_gate(str(path))


def test_p4_selfcheck_gate_reports_degeneracy_as_undetermined():
    degenerate = {"parse_rate": 0.95,
                  "P1": {"median_relative_error": 0.6},
                  "P2": {"median_relative_error": 0.6},
                  "P4_detection": {"p_vs_base_rate": None, "degenerate": True}}
    g = SR.p4_selfcheck_gate(degenerate)
    assert g["undetermined"] is True and g["passed"] is False
    assert any("full-arm S0" in r for r in g["reasons"])
    alive = {"parse_rate": 0.95,
             "P1": {"median_relative_error": 0.6},
             "P2": {"median_relative_error": 0.6},
             "P4_detection": {"p_vs_base_rate": 0.001, "degenerate": False}}
    assert SR.p4_selfcheck_gate(alive)["passed"] is True


def test_missing_sidecar_aborts_loudly_instead_of_planning_nothing(tmp_path):
    """The sidecar is unit U2's deliverable; without it there is no visible
    block length and no dropped-block truth, and nothing here invents one."""
    argv = ["plan-probes", "--sidecar", str(tmp_path / "nope.jsonl"),
            "--battery-c2kv", "c", "--battery-full", "f",
            "--manifest", "m.json", "--out", str(tmp_path / "plan.jsonl")]
    with pytest.raises(SystemExit, match="sidecar"):
        SR.main(argv)


def test_p4_gate_stays_closed_when_the_pooled_file_could_not_be_scored(tmp_path):
    """score-p4-pooled writes ``p4_gate: null`` when only one arm was run; the
    selfcheck arm must not slip through that hole."""
    path = tmp_path / "pooled.json"
    path.write_text(json.dumps({
        "parse_rate": 1.0, "p4_gate": None,
        "P4_detection_pooled": {"available": False, "detection": None}}),
        encoding="utf-8")
    with pytest.raises(RuntimeError, match="P4 gate FAILED"):
        SR.require_p4_gate(str(path))


def test_probe_cost_report_matches_the_preregistered_budget():
    rep = SR.probe_cost_report()
    assert rep["generations"] == 1288
    assert rep["gpu_hours"] == pytest.approx(1288 * 11.1 / 3600.0)


# --------------------------------------------------------------------------
# (B) MemGPT page-in
# --------------------------------------------------------------------------

def _action(name):
    return {"tool_calls": [{"name": name, "arguments": {}}], "text": ""}


def test_intercept_fires_only_on_the_reload_call_and_respects_the_budget():
    state = {"conv": "c1", "reloaded_convs": [], "max_reloads_per_conv": 1}
    miss = SR.intercept(_action("book_flight"), state)
    assert miss["fired"] is False and miss["recovery"] is None

    hit = SR.intercept(_action(SR.RELOAD_TOOL_NAME), state)
    assert hit["fired"] is True
    assert hit["recovery"] == {"placement": "append_tail", "target": "first",
                               "target_spec": "offset:0", "reissue_same_request": True}
    assert hit["state_update"] == {"reloaded_convs": ["c1"]}
    assert state["reloaded_convs"] == []        # pure: nothing mutated

    state2 = {"conv": "c1", "reloaded_convs": ["c1"], "max_reloads_per_conv": 1}
    again = SR.intercept(_action(SR.RELOAD_TOOL_NAME), state2)
    assert again["fired"] is False
    assert again["reason"] == "reload_budget_exhausted"


def test_reload_tool_is_parameterless_and_the_block_variant_is_named_diagnostic():
    schema = SR.reload_tool_schema()
    assert schema["function"]["parameters"]["properties"] == {}
    assert schema["function"]["parameters"]["required"] == []
    diag = SR.reload_tool_schema_block_variant_offline_diagnostic()
    assert "block_id" in diag["function"]["parameters"]["properties"]
    assert "offline_diagnostic" in \
        SR.reload_tool_schema_block_variant_offline_diagnostic.__name__


def test_page_out_counters_are_the_papers_example_defaults():
    assert (SR.MEMGPT_WARNING_FRAC, SR.MEMGPT_FLUSH_FRAC,
            SR.MEMGPT_EVICTION_FRAC) == (0.70, 1.00, 0.50)
    quiet = SR.page_out_signal(600, 1000)
    assert quiet["warning"] is False and quiet["flush"] is False
    warn = SR.page_out_signal(800, 1000)
    assert warn["warning"] is True and warn["flush"] is False
    flush = SR.page_out_signal(1100, 1000)
    assert flush["flush"] is True and flush["eviction_tokens"] == 500


def test_fire_rate_smoke_stopping_rule():
    silent = [{"conv_id": f"c{i}", "turn": 0, "action": _action("book_flight")}
              for i in range(30)]
    res = SR.fire_rate_smoke(silent)
    assert res["n_conversations"] == 30 and res["n_fires"] == 0
    assert res["stop"] is True and "stop" in res["stop_reason"]

    too_few = SR.fire_rate_smoke(silent[:10])
    assert too_few["stop"] is False          # not enough conversations to stop

    with_fire = silent + [{"conv_id": "c30", "turn": 2,
                           "action": _action(SR.RELOAD_TOOL_NAME)}]
    res2 = SR.fire_rate_smoke(with_fire)
    assert res2["n_fires"] == 1 and res2["stop"] is False


def test_page_in_metrics_denominators():
    rows = [
        {"conv_id": "c1", "turn": 3, "action": _action(SR.RELOAD_TOOL_NAME),
         "generate_sec": 1.5},
        {"conv_id": "c2", "turn": 0, "action": _action(SR.RELOAD_TOOL_NAME),
         "generate_sec": 1.0},
        {"conv_id": "c3", "turn": 1, "action": _action("book_flight"),
         "generate_sec": 9.0},
    ]
    res = SR.page_in_metrics(rows, {"c1": [4], "c9": [2]}, cc_convs=["c2", "c3"])
    assert res["n_cw_steps"] == 2
    assert res["coverage"] == pytest.approx(0.5)      # c1 fired at step-1
    assert res["n_fires"] == 2
    assert res["precision"] == pytest.approx(0.0)     # neither fire sits ON a C->W step
    assert res["n_cc_conversations"] == 2
    assert res["false_reset_rate"] == pytest.approx(0.5)
    assert res["cost_generate_sec_2x"] == pytest.approx(5.0)


def test_memory_pressure_notice_documents_the_fingerprint_hazard():
    assert "tracking_lost" in SR.memory_pressure_notice.__doc__
    assert SR.RELOAD_TOOL_NAME in SR.memory_pressure_notice(0.8)


# --------------------------------------------------------------------------
# (C) selfcheck arm
# --------------------------------------------------------------------------

def test_selfcheck_reuses_the_prefix_and_appends_after_the_emitted_call():
    current = [{"role": "user", "content": "book me a flight"}]
    msgs = SR.selfcheck_current_messages(current, "<tool_call>{}</tool_call>")
    assert msgs[: len(current)] == current            # prefix reuse: byte identical
    assert msgs[-2]["role"] == "assistant"
    assert msgs[-1]["content"] == SR.SELFCHECK_QUESTION
    assert SR.SELFCHECK_GEN["max_new_tokens"] == 32
    assert SR.SELFCHECK_GEN["enable_thinking"] is False
    assert SR.SELFCHECK_GEN["temperature"] == 0.0


def test_selfcheck_features_keep_none_and_pass_the_leakage_guard(tmp_path):
    rows = [
        {"qid": "s1:1", "generation": "No. 0.2", "generate_sec": 0.4},
        {"qid": "s1:2", "generation": "Yes, 0.9", "generate_sec": 0.5},
        {"qid": "s2:1", "generation": "???", "generate_sec": 0.3},
    ]
    feats = SR.selfcheck_feature_rows(rows)
    assert feats[0]["selfcheck_risk"] == pytest.approx(0.8)
    assert feats[0]["selfcheck_says_inconsistent"] == 1.0
    assert feats[2]["selfcheck_risk"] is None          # no sentinel fallback
    assert feats[2]["selfcheck_answer_parse_ok"] == 0.0
    out = tmp_path / "f.jsonl"
    n = C.write_features_jsonl(out, feats, context="test")
    assert n == 3
    written = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert written[2]["selfcheck_risk"] is None        # json null, not imputed


def test_cofire_diagnostic_is_the_worth_it_criterion():
    sc = {"a": True, "b": True, "c": False, "d": True}
    pf = {"a": True, "b": False, "c": True, "d": True}
    res = SR.cofire_with_parse_failure(sc, pf)
    assert res["n_selfcheck_fires"] == 3
    assert res["n_cofire"] == 2
    assert res["fraction_of_selfcheck_fires_cofiring"] == pytest.approx(2 / 3)
    assert res["n_selfcheck_only"] == 1

    # an UNPARSED selfcheck answer is not "did not fire": it leaves the frame
    with_undefined = dict(sc, e=None)
    pf2 = dict(pf, e=True)
    res2 = SR.cofire_with_parse_failure(with_undefined, pf2)
    assert res2["n_undefined_selfcheck"] == 1
    assert res2["n_rows"] == 4
    assert res2["n_selfcheck_fires"] == 3
    assert res2["fraction_of_selfcheck_fires_cofiring"] == pytest.approx(2 / 3)


# --------------------------------------------------------------------------
# scoring frame discipline
# --------------------------------------------------------------------------

def _frame_161(rng):
    y = [1] * 93 + [0] * 68
    sess = [f"s{i // 2}" for i in range(161)]
    return y, sess


def test_chance_ap_is_the_subset_prevalence_not_the_900_base_rate():
    y, sess = _frame_161(np.random.default_rng(0))
    scores = list(np.random.default_rng(1).random(161))
    table = SR.score_feature_table(scores, y, sess, name="x", orientation=1, reps=50)
    assert table["n"] == 161 and table["n_pos"] == 93
    assert table["prevalence_chance_ap"] == pytest.approx(93 / 161, abs=1e-9)
    assert table["prevalence_chance_ap"] != pytest.approx(C.BASE_RATE_900)


def test_score_feature_table_drops_undefined_rows_and_says_so():
    y, sess = _frame_161(None)
    scores = [None if i % 10 == 0 else float(i) for i in range(161)]
    table = SR.score_feature_table(scores, y, sess, name="x", orientation=1, reps=20)
    assert table["n_dropped_undefined"] == 17
    assert table["n"] == 161 - 17


def test_s0_label_control_is_a_paired_delta_on_the_same_rows():
    y, sess = _frame_161(None)
    rng = np.random.default_rng(3)
    good = [float(v) + (2.0 if lbl else 0.0) for v, lbl in zip(rng.random(161), y)]
    flat = list(rng.random(161))
    res = SR.selfcheck_label_control_s0(good, flat, y, sess, reps=100)
    assert res["n_scored"] == 161 and res["n_pos"] == 93
    assert res["delta_auprc"] > 0
    assert res["alive_by_s0_rule"] is True
    assert "label" in SR.selfcheck_label_control_s0.__name__


def test_orientations_file_covers_every_emitted_feature():
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "configs/t34/orientations_selfreport.json"
    orients = json.loads(path.read_text(encoding="utf-8"))
    emitted = set()
    emitted.update(k for k in SR.selfcheck_feature_rows(
        [{"qid": "s:1", "generation": "yes 0.5", "generate_sec": 0.1}])[0]
        if k not in C.META_COLS)
    emitted.update(k for k in J.judge_feature_rows(
        [{"qid": "s:1", "judge_score": 0.5, "judge_utility": 1.0,
          "judge_parse_ok": True, "judge_seconds": 0.2,
          "judge_prompt_tokens": 10, "judge_completion_tokens": 3}])[0]
        if k not in C.META_COLS)
    missing = sorted(emitted - set(orients))
    assert missing == [], f"orientations missing for {missing}"
    assert all(v in (1, -1) for v in orients.values())
    assert SR.orientation_of("selfcheck_risk") == 1
    assert SR.orientation_of("selfcheck_answer_parse_ok") == -1
    with pytest.raises(KeyError):
        SR.orientation_of("a_feature_nobody_declared")


def test_plus_ledger_discloses_state_on_purpose_but_the_query_never_does():
    sc = [_sidecar_row("s1:2", [100, 200, 50, 400], dropped=[1])]
    bat = {"s1:2": _battery_row("s1:2")}
    rows = SR.probe_plan_rows(sc, bat, condition="plus_ledger")
    for r in rows:
        assert SR.scan_prompt_for_state_leak(r["query"]) == []
    # the ledger itself is the disclosure: the scan WOULD flag it, by design
    assert SR.scan_prompt_for_state_leak(rows[0]["ledger"])


# --------------------------------------------------------------------------
# judge (2507.21017) with a mocked client
# --------------------------------------------------------------------------

class _MockClient:
    """Stands in for JudgeClient; records the prompts it was given."""

    def __init__(self, replies):
        self.model = "mock-judge"
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return {"text": self.replies.pop(0), "judge_seconds": 0.25,
                "judge_prompt_tokens": 120, "judge_completion_tokens": 30}


def test_mirage_prompt_has_both_steps_and_the_three_categories():
    prompt = J.mirage_prompt("SEGMENT")
    assert "Step 1" in prompt and "Step 2" in prompt
    for word in ("faithful", "incomplete", "hallucinated"):
        assert word in prompt
    assert "harmful_probability" in prompt
    assert "SEGMENT" in prompt
    assert "reference trajectory" in prompt      # judge has no reference


@pytest.mark.parametrize("grade,expect", [
    ("faithful", 1.0), ("incomplete", 0.5), ("hallucinated", 0.0),
    ("Incomplete Action", 0.5), ("partial", 0.5), ("nonsense", None), (None, None),
])
def test_utility_mapping_keeps_the_incomplete_middle_grade(grade, expect):
    assert J.utility_from_grade(grade) == expect


def test_judge_one_uses_the_frozen_generation_contract_and_cost_columns():
    reply = json.dumps({"risk_trigger": "ungrounded arg", "why": "no turn shows it",
                        "category": "hallucinated", "harmful_probability": 0.82,
                        "justification": "..."})
    client = _MockClient([reply])
    row = {"qid": "s1:2", "session_id": "s1", "prediction": "<tool_call>{}</tool_call>",
           "doc_chunks": 3}
    sidecar = {"qid": "s1:2", "docs": ["alpha", "beta"], "dropped_docs": [1],
               "query": "book it"}
    rec = J.judge_one(row, sidecar, client, judge_name="small_judge")
    assert rec["judge_score"] == pytest.approx(0.82)
    assert rec["judge_utility"] == 0.0
    assert rec["judge_seconds"] == 0.25
    assert rec["judge_prompt_tokens"] == 120 and rec["judge_completion_tokens"] == 30
    assert rec["judge_name"] == "small_judge"
    assert J.JUDGE_GEN["temperature"] == 0
    assert J.JUDGE_GEN["max_completion_tokens"] == 256
    assert J.JUDGE_GEN["chat_template_kwargs"]["enable_thinking"] is False
    # the segment carried the compressed prefix summary and the emitted action
    assert "earlier_turn" in client.prompts[0]
    assert "agent_action" in client.prompts[0]


def test_judge_feature_rows_pass_the_leakage_guard(tmp_path):
    feats = J.judge_feature_rows([
        {"qid": "s1:1", "judge_score": 0.7, "judge_utility": 0.5,
         "judge_parse_ok": True, "judge_seconds": 0.2,
         "judge_prompt_tokens": 100, "judge_completion_tokens": 20},
        {"qid": "s1:2", "judge_score": None, "judge_utility": None,
         "judge_parse_ok": False, "judge_seconds": 0.2,
         "judge_prompt_tokens": 100, "judge_completion_tokens": 20},
    ])
    assert feats[0]["judge_utility_risk"] == pytest.approx(0.5)
    assert feats[1]["judge_harmful_probability"] is None
    out = tmp_path / "judge_feats.jsonl"
    assert C.write_features_jsonl(out, feats, context="test judge") == 2


def test_segment_refuses_label_side_fields():
    with pytest.raises(ValueError):
        J.assert_segment_is_feature_side(["prediction", "tool_name_match"])
    with pytest.raises(ValueError):
        J.assert_segment_is_feature_side(["prediction", "target"])
    J.assert_segment_is_feature_side(["qid", "prediction", "doc_chunks"])


def test_compact_segment_marks_dropped_blocks_and_omits_targets():
    row = {"qid": "s1:2", "prediction": "CALL", "doc_chunks": 2,
           "target": "SHOULD NOT APPEAR", "tool_name_match": True}
    sidecar = {"docs": ["one", "two"], "dropped_docs": [0], "query": "q"}
    seg = J.compact_battery_segment(row, sidecar)
    assert "SHOULD NOT APPEAR" not in seg
    assert '"available": false' in seg.lower()
    assert "CALL" in seg


def test_segment_history_follows_the_sidecar_contract():
    """``docs`` holds ONLY the kept blocks; ``dropped_docs`` indexes the
    POST-SPLIT history, not ``docs``.

    So every entry of ``docs`` is visible, the dropped blocks are the ones
    ABSENT from it, and 'enumerate(docs) filtered by dropped_docs' is wrong in
    both directions.
    """
    sc = {"docs": ["kept-A", "kept-B"], "dropped_docs": [1], "query": "q"}
    ents = J.segment_history_entries(sc)
    assert [e["history_index"] for e in ents] == [0, 1, 2]
    # both docs are visible; the missing block is the one that is not in docs
    assert [(e["history_index"], e["available"], e["text"]) for e in ents] == [
        (0, True, "kept-A"), (1, False, None), (2, True, "kept-B")]
    assert J.dropped_text_availability(sc) == {
        "n_dropped_blocks": 1, "n_visible_blocks": 2,
        "dropped_doc_texts_available": False}
    # with --with_dropped_text the dropped side carries its text, still False
    sc2 = dict(sc, dropped_doc_texts=["gone-text"])
    ents2 = J.segment_history_entries(sc2)
    assert ents2[1] == {"history_index": 1, "earlier_turn": 2,
                        "available": False, "text": "gone-text"}
    assert J.dropped_text_availability(sc2)["dropped_doc_texts_available"] is True
    # a row that dropped nothing leaves the question undefined, not False
    assert J.dropped_text_availability({"docs": ["a"], "dropped_docs": []}
                                       )["dropped_doc_texts_available"] is None
    # and a mismatched text list is not zipped up by position
    bad = dict(sc, dropped_doc_texts=["x", "y"])
    assert J.segment_history_entries(bad)[1]["text"] is None
    assert J.dropped_text_availability(bad)["dropped_doc_texts_available"] is False
    # the rendered segment shows the gap AND names its text as unavailable
    seg = J.compact_battery_segment({"qid": "s1:2", "prediction": "CALL"}, sc)
    assert '"earlier_turn": 2, "available": false, "text": null' in seg


def test_parse_judge_response_tolerates_prose_and_keeps_none():
    good = J.parse_judge_response('prefix {"category": "incomplete", '
                                  '"harmful_probability": 0.4} suffix')
    assert good["utility"] == 0.5 and good["harmful_probability"] == 0.4
    assert good["harmful_probability_scale"] == "unit"
    assert good["harmful_probability_raw"] == "0.4"
    assert good["parse_ok"] is True
    bad = J.parse_judge_response("no json here")
    assert bad["harmful_probability"] is None and bad["utility"] is None
    assert bad["parse_ok"] is False


def test_harmful_probability_is_range_checked_and_the_raw_string_is_kept():
    # a bare 82 is NOT silently divided by 100: nothing in the reply says percent
    out = J.parse_judge_response('{"harmful_probability": 82}')
    assert out["harmful_probability"] is None
    assert out["harmful_probability_scale"] == "out_of_range"
    assert out["harmful_probability_raw"] == "82"
    assert out["parse_ok"] is False           # no category either
    # an explicit percent IS rescaled, and says so
    pct = J.parse_judge_response('{"harmful_probability": "82%"}')
    assert pct["harmful_probability"] == pytest.approx(0.82)
    assert pct["harmful_probability_scale"] == "percent_rescaled"
    assert pct["harmful_probability_raw"] == "82%"
    # ... including when the '%' sits outside the JSON number in the raw reply
    near = J.parse_judge_response('the model wrote harmful_probability: 82% here'
                                  ' {"harmful_probability": 82}')
    assert near["harmful_probability"] == pytest.approx(0.82)
    assert near["harmful_probability_scale"] == "percent_rescaled"
    # a percent-looking value above 100, a negative, a bool and a word are None
    for blob in ('{"harmful_probability": "180%"}',
                 '{"harmful_probability": -0.2}',
                 '{"harmful_probability": true}',
                 '{"harmful_probability": "high"}'):
        got = J.parse_judge_response(blob)
        assert got["harmful_probability"] is None, blob
        assert got["harmful_probability_scale"] in J.HP_SCALES, blob
    # the boundaries stay unit-scale
    for v, expect in (("0", 0.0), ("1", 1.0), ("0.5", 0.5), ('"0.5"', 0.5)):
        got = J.parse_judge_response('{"harmful_probability": %s}' % v)
        assert got["harmful_probability"] == pytest.approx(expect)
        assert got["harmful_probability_scale"] == "unit"


def test_judge_rows_carry_the_range_and_input_availability_columns():
    client = _MockClient([json.dumps({"category": "faithful",
                                      "harmful_probability": 82})])
    rec = J.judge_one({"qid": "s1:2", "prediction": "CALL"},
                      {"docs": ["a"], "dropped_docs": [1]}, client)
    assert rec["judge_score"] is None                  # out of range, not 82
    assert rec["judge_score_raw"] == "82"
    assert rec["judge_score_scale"] == "out_of_range"
    assert rec["judge_utility"] == 1.0                 # the category still parsed
    assert rec["sidecar_available"] is True
    assert rec["dropped_doc_texts_available"] is False  # dropped, but no text
    counts = J._scale_counts([rec, {"qid": "x"}])
    assert counts == {"out_of_range": 1, "unrecorded": 1}


def test_us_hr_aggregates_and_flags_that_they_are_not_trigger_metrics():
    res = J.us_hr([1.0, 0.5, 0.0, 0.0, None])
    assert res["US"] == pytest.approx(0.375)
    assert res["HR"] == pytest.approx(0.5)
    assert res["n_graded"] == 4 and res["n_ungraded"] == 1
    assert res["not_a_trigger_metric"] is True


def test_human_validation_sampler_is_seeded():
    qids = [f"s{i}:1" for i in range(300)]
    a = J.sample_human_validation(qids, n=160, seed=7)
    b = J.sample_human_validation(qids, n=160, seed=7)
    c = J.sample_human_validation(qids, n=160, seed=8)
    assert a["qids"] == b["qids"] and a["n_sampled"] == 160
    assert a["qids"] != c["qids"]


# --------------------------------------------------------------------------
# Verify-when-Uncertain cascade (2502.15845 Algorithm 1)
# --------------------------------------------------------------------------

def test_cascade_star_threshold_is_the_budget_dial():
    """Wrapper semantics are t34_cascade's: p=0 puts t* just BELOW t1 (nothing
    escalates) and an over-large p clamps to the largest score, not +inf."""
    s = list(np.linspace(0.0, 1.0, 101))
    assert J.cascade_star_threshold(s, 0.0, 0.0) < 0.0        # empty band
    t_star = J.cascade_star_threshold(s, 0.0, 0.2)
    band = [v for v in s if 0.0 <= v <= t_star]
    assert len(band) == pytest.approx(20, abs=2)
    assert J.cascade_star_threshold(s, 0.0, 1.0) == pytest.approx(1.0)
    # None holes are dropped, not coerced
    assert J.cascade_star_threshold([None, 0.0, 1.0], 0.0, 1.0) == pytest.approx(1.0)


def test_cascade_decide_follows_algorithm_1():
    d = J.cascade_decide(0.1, 0.3, 0.7, 0.9, 0.5)
    assert d["fire"] is False and d["escalated"] is False
    assert d["stage"] == "cheap_negative"
    d = J.cascade_decide(0.9, 0.3, 0.7, None, 0.5)
    assert d["fire"] is True and d["escalated"] is False
    d = J.cascade_decide(0.5, 0.3, 0.7, 0.9, 0.5)
    assert d["fire"] is True and d["escalated"] is True
    d = J.cascade_decide(0.5, 0.3, 0.7, 0.1, 0.5)
    assert d["fire"] is False and d["escalated"] is True
    # a band row with no stage-2 score takes the declared fallback (do NOT
    # fire: the escalation could not be paid for) and is counted as such
    d = J.cascade_decide(0.5, 0.3, 0.7, None, 0.5)
    assert d["fire"] is False and d["stage"] == "band_unavailable"
    assert d["fallback"] is True
    assert J.cascade_decide(None, 0.3, 0.7, 0.9, 0.5)["stage"] == "undefined"


def test_cascade_helpers_are_thin_wrappers_over_the_single_implementation():
    """Algorithm 1 exists once on this branch; these names must delegate."""
    import t34_cascade as K

    assert tuple(J.P_GRID) == tuple(K.P_GRID)
    thr = K.CascadeThresholds(t1=0.3, t_star=0.7, t2=0.5)
    for s, e in ((0.1, 0.9), (0.9, None), (0.5, 0.9), (0.5, 0.1), (0.5, None)):
        assert J.cascade_decide(s, 0.3, 0.7, e, 0.5) == K.cascade_decide(s, thr, e)
    a = J.cascade_apply([0.1, 0.5, None], [0.9, 0.1, 0.9], 0.3, 0.7, 0.5)
    b = K.apply_cascade([0.1, 0.5, np.nan], thr, [0.9, 0.1, 0.9])
    assert list(a["fire"]) == list(b["fire"])
    assert list(a["stage"]) == list(b["stage"])
    assert J.cascade_star_threshold([0.0, 0.5, 1.0], 0.0, 0.5) == \
        pytest.approx(K.band_upper_threshold([0.0, 0.5, 1.0], 0.0, 0.5)[0])
    # and the module says so where a reader will see it
    assert "t34_cascade" in J.__doc__
    assert any("t34_cascade" in d["what"] for d in J.DEVIATIONS)


def test_fit_cascade_selects_thresholds_out_of_fold_and_prices_the_judge():
    rng = np.random.default_rng(11)
    n = 160
    y = [1 if i % 2 == 0 else 0 for i in range(n)]
    groups = [f"s{i // 4}" for i in range(n)]
    stage1 = [1.0 if (lbl and i % 4 == 0) else 0.0 for i, lbl in enumerate(y)]
    stage2 = [float(rng.random() * 0.3 + (0.6 if lbl else 0.0)) for lbl in y]
    secs = [0.5] * n
    res = J.fit_cascade(stage1, stage2, y, groups, p=0.4, stage2_seconds=secs,
                        outer_folds=3, inner_folds=2)
    assert res["n_scored"] > 0
    assert res["n_threshold_pairs_tried"] > 0
    assert res["escalation_fraction"] is not None
    assert res["judge_seconds_spent"] == pytest.approx(
        0.5 * res["escalation_fraction"] * res["n_scored"], rel=1e-6)
    assert len(res["chosen_per_outer_fold"]) >= 1
    assert res["implementation"] == "t34_cascade.cascade_nested_cv"
    assert res["regime"] == "practical_cross_session"
    # the realized band fraction is measured per fold, never assumed
    assert all("p_realized" in f for f in res["chosen_per_outer_fold"])
    assert res["n_band_unavailable"] == 0     # every row has a stage-2 score


def test_cascade_threshold_grids_are_built_inside_the_train_fold():
    """The candidate (t1, t2) values must come from the outer-TRAIN rows only.

    Two frames that agree on every training row but differ wildly on the held-out
    rows would produce different grids — and therefore different chosen
    thresholds — if the grid were quantiled over the whole frame.
    """
    import t34_common as CC

    n = 120
    y = [1 if i % 2 == 0 else 0 for i in range(n)]
    groups = [f"s{i // 4}" for i in range(n)]
    g = np.asarray(groups)
    # the rows the FIRST outer fold holds out, under the same seed/folding
    held = C.grouped_folds(g, 3, 20260905)[0]
    assert held.any() and not held.all()

    base1 = [0.2 + 0.6 * (i % 5) / 4.0 for i in range(n)]
    base2 = [0.1 + 0.8 * (i % 7) / 6.0 for i in range(n)]
    # variant: only the held-out rows' scores move, and they move far
    var1 = [(-5.0 if held[i] else v) for i, v in enumerate(base1)]
    var2 = [(9.0 if held[i] else v) for i, v in enumerate(base2)]

    kw = dict(outer_folds=3, inner_folds=2, seed=20260905)
    a = J.fit_cascade(base1, base2, y, groups, p=0.4, **kw)
    b = J.fit_cascade(var1, var2, y, groups, p=0.4, **kw)
    # fold 0 trains on the untouched rows in both frames, so its chosen
    # thresholds — and the levels and band they came from — must be identical
    fa, fb = a["chosen_per_outer_fold"][0], b["chosen_per_outer_fold"][0]
    for key in ("t1", "t2", "t_star", "q1", "q2", "p_realized"):
        assert fa[key] == pytest.approx(fb[key]), key
    assert CC is C


def test_cascade_sweep_covers_the_preregistered_p_grid():
    y = [1 if i % 3 == 0 else 0 for i in range(60)]
    groups = [f"s{i // 3}" for i in range(60)]
    stage1 = [0.0] * 60
    stage2 = [0.9 if lbl else 0.1 for lbl in y]
    res = J.cascade_sweep(stage1, stage2, y, groups, outer_folds=3, inner_folds=2)
    assert res["p_grid"] == list(J.P_GRID)
    assert len(res["rows"]) == len(J.P_GRID)
    assert res["rows"][0]["p"] == 0.0
    # a binary (here constant) stage-1 signal collapses the band; the sweep
    # says so instead of presenting itself as a threshold study
    assert res["stage1_degenerate_band"] is True
    assert res["rows"][0]["stage1_distinct_values"] == 1
    cont = J.cascade_sweep(list(np.linspace(0, 1, 60)), stage2, y, groups,
                           p_grid=(0.2,), outer_folds=3, inner_folds=2)
    assert cont["stage1_degenerate_band"] is False


def test_cascade_reports_band_rows_it_could_not_pay_for():
    """An escalated row with no judge score does NOT fire and is counted."""
    n = 90
    y = [1 if i % 2 == 0 else 0 for i in range(n)]
    groups = [f"s{i // 3}" for i in range(n)]
    stage1 = list(np.linspace(0.0, 1.0, n))
    stage2 = [None] * n                       # the judge never answered
    res = J.fit_cascade(stage1, stage2, y, groups, p=0.4,
                        outer_folds=3, inner_folds=2)
    assert res["n_band_unavailable"] > 0
    assert res["n_band_unavailable"] <= res["n_scored"]


# --------------------------------------------------------------------------
# probe-mode driver (server): wiring/reduction logic without torch
# --------------------------------------------------------------------------

def test_probe_mode_wiring_without_torch(tmp_path):
    import t34_probe_mode as PM

    ns = type("NS", (), {})()
    ns.arm, ns.ratio, ns.attn_impl = "c2kv", 8, "eager"
    ns.dataset_path, ns.tokenizer_path, ns.model_path = "d", "t", "m"
    ns.base_model, ns.max_new_tokens = "b", 64
    args = PM.build_args(ns)
    assert args.mode == "c2kv" and args.do_sample is False
    assert args.max_new_tokens == 64 and args.override_ratio == 8
    ns.arm = "full"
    assert PM.build_args(ns).mode == "full"

    plan_row = {"qid": "s1:2", "session_id": "s1", "probe": "P1",
                "condition": "minus_ledger", "query": "how many tokens?",
                "ledger": None, "truth": {"total_tokens": 400}}
    metrics = {"prediction": "about 200 tokens", "generate_sec": 1.2,
               "prompt_tokens": 33, "generated_tokens": 5,
               "target": "LEAK", "tool_name_match": True}
    rec = PM.probe_row_record(plan_row, "c2kv", "vista", metrics,
                              metrics["prediction"])
    assert "target" not in rec and "tool_name_match" not in rec
    assert rec["parsed"]["value"] == 200
    assert rec["generate_sec"] == 1.2

    out = tmp_path / "probes.jsonl"
    out.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    assert PM.resume_done(str(out)) == {"s1:2|P1|minus_ledger"}

    sc_plan = PM.selfcheck_plan(["s1:2"], {"s1:2": "<tool_call>{}</tool_call>"})
    assert sc_plan[0]["probe"] == "selfcheck"
    assert sc_plan[0]["query"] == SR.SELFCHECK_QUESTION
    assert PM.parse_for_probe("selfcheck", "yes 0.9")["score"] == pytest.approx(0.9)


def test_probe_mode_selfcheck_gate_refuses_without_p4():
    import t34_probe_mode as PM
    argv = ["--mode", "selfcheck", "--arm", "c2kv", "--model_path", "m",
            "--base_model", "b", "--tokenizer_path", "t", "--dataset_path", "d",
            "--battery_full", "f", "--battery_c2kv", "c", "--manifest", "m.json",
            "--out", "o.jsonl"]
    with pytest.raises(RuntimeError, match="gated on the VISTA P4"):
        PM.main(argv)


def test_every_module_declares_its_deviations():
    import t34_probe_mode as PM
    for mod in (SR, J, PM):
        assert isinstance(mod.DEVIATIONS, list) and mod.DEVIATIONS
        for d in mod.DEVIATIONS:
            assert set(d) == {"method", "paper", "what", "why"}
            assert all(isinstance(v, str) and v for v in d.values())
