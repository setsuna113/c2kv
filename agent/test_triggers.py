# -*- coding: utf-8 -*-
"""Unit tests for agent/triggers.py and agent/t34_dump_sidecar.py (t34 §4.6).

Everything here is synthetic: synthetic tool schemas, synthetic emitted
actions, synthetic decoded docs, synthetic proxy log rows.  No frozen asset is
read except through the two tests that build a tiny in-memory battery/sidecar
pair, so the suite runs on the torch-free Windows box.

Run:
  PYTHONIOENCODING=utf-8 python -m pytest agent/test_triggers.py -q
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

_AGENT = Path(__file__).resolve().parent
if str(_AGENT) not in sys.path:
    sys.path.insert(0, str(_AGENT))

import t34_dump_sidecar as DS  # noqa: E402
import triggers as T  # noqa: E402


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

TOOLS = [
    {"type": "function", "function": {
        "name": "search_users",
        "parameters": {"type": "object", "required": ["query"],
                       "properties": {"query": {"type": "string"},
                                      "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "get_reservation",
        "parameters": {"type": "object", "required": ["reservation_id"],
                       "properties": {"reservation_id": {"type": "string"}}},
        "returns": {"type": "object", "required": ["status"]}}},
]


def call_text(name, args):
    return ("Action:\n<tool_call>\n"
            + json.dumps({"name": name, "arguments": args})
            + "\n</tool_call>")


# --------------------------------------------------------------------------
# module hygiene
# --------------------------------------------------------------------------

def test_deviations_block_is_populated_and_typed():
    assert len(T.DEVIATIONS) >= 10
    for d in T.DEVIATIONS + DS.DEVIATIONS:
        assert set(d) == {"method", "paper", "what", "why"}
        assert all(isinstance(v, str) and v.strip() for v in d.values())


def test_no_import_from_benchmarks_tree():
    src = (_AGENT / "triggers.py").read_text(encoding="utf-8")
    assert "from benchmarks" not in src and "import benchmarks" not in src


# --------------------------------------------------------------------------
# fuzzy backend
# --------------------------------------------------------------------------

def test_fuzz_backend_is_declared_and_stamped():
    backend = T.fuzz_backend()
    assert backend in ("rapidfuzz", "difflib")
    row = T.saag_cascade(call_text("search_users", {"query": "abc"}), TOOLS, "abc")
    assert row["fuzz_backend"] == backend


def test_partial_ratio_bounds_and_monotonicity():
    assert T.partial_ratio("alpha", "alpha") == pytest.approx(100.0)
    assert 0.0 <= T.partial_ratio("alpha", "zzzzzz") <= 100.0
    assert T.partial_ratio("needle", "a haystack with a needle inside") > \
        T.partial_ratio("needle", "a haystack with nothing inside")


def test_w_ratio_and_jaccard():
    assert T.w_ratio("get_user", "get_user") == pytest.approx(100.0)
    assert T.w_ratio("", "x") == 0.0 or T.fuzz_backend() == "rapidfuzz"
    assert T.jaccard("a b", "a b") == 1.0
    assert T.jaccard("a b", "c d") == 0.0


def test_difflib_fallback_truncates_the_haystack():
    # the cap is a declared deviation; assert it is actually applied
    hay = "z" * (T.DIFFLIB_HAYSTACK_CHARS + 50) + "needle"
    assert T._difflib_partial_ratio("needle", hay) < 100.0


# --------------------------------------------------------------------------
# (A) SIEVE three-valued routing
# --------------------------------------------------------------------------

def test_route_pass_when_legal_and_grounded():
    txt = call_text("search_users", {"query": "alice smith", "limit": 5})
    assert T.route(txt, TOOLS, "the user alice smith wrote in", "give me 5") == T.ROUTE_PASS


def test_route_deviation_on_parse_failure():
    assert T.route("just prose, no call", TOOLS, "", "") == T.ROUTE_DEVIATION
    d = T.route_detail("just prose, no call", TOOLS, "", "")
    assert d["reason"] == "no tool call emitted"


def test_route_deviation_on_unknown_name():
    txt = call_text("not_a_tool", {"query": "alice"})
    d = T.route_detail(txt, TOOLS, "alice", "")
    assert d["route"] == T.ROUTE_DEVIATION and "unknown tool name" in d["reason"]


def test_route_deviation_on_missing_required_key():
    txt = call_text("search_users", {"limit": 5})
    d = T.route_detail(txt, TOOLS, "5", "")
    assert d["route"] == T.ROUTE_DEVIATION and "missing required" in d["reason"]


def test_route_deviation_on_type_mismatch():
    txt = call_text("search_users", {"query": "alice", "limit": "five"})
    d = T.route_detail(txt, TOOLS, "alice five", "")
    assert d["route"] == T.ROUTE_DEVIATION and "must be integer" in d["reason"]


def test_route_unresolved_when_a_value_is_not_grounded():
    txt = call_text("search_users", {"query": "zzz-unseen-value"})
    d = T.route_detail(txt, TOOLS, "nothing relevant here", "nor here")
    assert d["route"] == T.ROUTE_UNRESOLVED and d["ungrounded"] == ["query"]


def test_route_pass_with_no_arguments_at_all():
    txt = call_text("search_users", {"query": "q"})
    tools = [{"name": "noargs", "parameters": {}}]
    assert T.route(call_text("noargs", {}), tools, "", "") == T.ROUTE_PASS
    # and the argument-bearing call is still routed on its values
    assert T.route(txt, TOOLS, "q", "") == T.ROUTE_PASS


def test_route_grounds_in_visible_text_union_query():
    txt = call_text("search_users", {"query": "carol"})
    assert T.route(txt, TOOLS, "carol was mentioned earlier", "") == T.ROUTE_PASS
    assert T.route(txt, TOOLS, "", "find carol") == T.ROUTE_PASS


def test_escalation_accounting_shape_and_missing_cells():
    routes = [T.ROUTE_PASS, T.ROUTE_DEVIATION, T.ROUTE_UNRESOLVED, T.ROUTE_PASS]
    acc = T.escalation_accounting(routes, cpu_ms_total=8.0)
    assert acc["fires"] == 2 and acc["fire_rate"] == 0.5
    assert acc["escalation_rate"] == 0.5 and acc["cpu_ms_per_step"] == 2.0
    # cost cells stay None when the caller supplies no cost ledger
    assert acc["escalation_cost_total"] is None
    assert acc["cost_per_rescued_step"] is None
    acc2 = T.escalation_accounting(routes, cost_per_escalation=3.0, rescued_steps=1)
    assert acc2["escalation_cost_total"] == 6.0
    assert acc2["cost_per_rescued_step"] == 6.0


def test_always_on_vs_selective_absorption():
    routes = [T.ROUTE_PASS] * 6 + [T.ROUTE_DEVIATION] * 4
    cmp = T.always_on_vs_selective(routes, cost_per_escalation=1.0)
    assert cmp["always_on"]["fires"] == 10
    assert cmp["selective"]["fires"] == 4
    assert cmp["absorbed_share_of_escalations"] == pytest.approx(0.6)
    assert "utility" in cmp["note"]


def test_crosstab_2x3_denominators_and_silent_cell():
    routes = {"s:1": T.ROUTE_PASS, "s:2": T.ROUTE_DEVIATION,
              "s:3": T.ROUTE_UNRESOLVED, "s:4": T.ROUTE_PASS, "s:5": T.ROUTE_PASS}
    labels = {"s:1": 1, "s:2": 1, "s:3": 0, "s:4": 0, "s:5": None}
    ct = T.crosstab_2x3(routes, labels)
    assert ct["n_cw"] == 2 and ct["n_cc"] == 2 and ct["n_outside_trigger_frame"] == 1
    assert ct["cw_and_pass"] == 1
    assert ct["coverage"] == 0.5 and ct["false_reset"] == 0.5
    assert ct["precision"] == 0.5
    assert sum(sum(v.values()) for v in ct["table"].values()) == 4


def test_dropped_docs_subanalysis_splits_and_preregisters():
    routes = {"a:1": T.ROUTE_PASS, "a:2": T.ROUTE_DEVIATION}
    labels = {"a:1": 1, "a:2": 1}
    dropped = {"a:1": True, "a:2": False}
    out = T.dropped_docs_subanalysis(routes, labels, dropped)
    assert out["dropped_nonempty"]["n_cw"] == 1
    assert out["dropped_empty"]["n_cw"] == 1
    assert "prediction" in out["prereg"]


# --------------------------------------------------------------------------
# (B) SAAG stages
# --------------------------------------------------------------------------

def test_fnem_three_values():
    assert T.fnem(None, ["a"]) == -1
    assert T.fnem("b", ["a"]) == 0
    assert T.fnem("a", ["a"]) == 1


def test_ncs_defined_only_when_fnem_is_zero():
    assert T.ncs("search_users", ["search_users"]) is None      # FNEM = 1
    assert T.ncs(None, ["search_users"]) is None                # FNEM = -1
    v = T.ncs("search_user", ["search_users", "get_reservation"])
    assert v is not None and 0.0 <= v <= 1.0
    # a name sharing no trigram with anything scores 0.0, not None
    assert T.ncs("qqq", ["search_users"]) == 0.0


def test_rpr_and_empty_required_set():
    assert T.rpr({"query": "x"}, ["query"]) == 1.0
    assert T.rpr({}, ["query", "limit"]) == 0.0
    assert T.rpr({"query": "x"}, ["query", "limit"]) == 0.5
    assert T.rpr({"a": 1}, []) == 1.0          # declared: nothing required = covered
    assert T.rpr(None, ["query"]) is None


def test_spr_and_drift_carveout():
    admissible = ["query", "limit"]
    # "qeury" is a near-miss of "query" -> drifted (PDS >= 0.8), not spurious
    assert T.pds("qeury", admissible) >= T.PDS_DRIFT_CUT
    assert T.spr({"qeury": "x"}, admissible) == 0.0
    assert T.drifted_parameters({"qeury": "x"}, admissible) == ["qeury"]
    # a genuinely foreign parameter is spurious
    assert T.spr({"query": "x", "zzzzzzzz": 1}, admissible) == 0.5
    assert T.spr({}, admissible) == 0.0


def test_avem_typed_rules():
    text = "reservation HXK123 for 3 nights, tags: alpha and beta"
    # string on a word boundary, case-insensitive
    assert T.avem({"a": "hxk123"}, text) == 1.0
    # substring that is NOT on a word boundary must not count
    assert T.avem({"a": "XK12"}, text) == 0.0
    # numeric must match a float token
    assert T.avem({"n": 3}, text) == 1.0
    assert T.avem({"n": 7}, text) == 0.0
    # list is element-wise: all elements or nothing
    assert T.avem({"l": ["alpha", "beta"]}, text) == 1.0
    assert T.avem({"l": ["alpha", "gamma"]}, text) == 0.0
    # no values at all -> undefined, never 1.0
    assert T.avem({}, text) is None
    assert T.avem(None, text) is None


def test_qslo_only_over_non_exact_strings():
    text = "reservation HXK123"
    assert T.qslo({"a": "HXK123"}, text) is None      # exact -> excluded
    v = T.qslo({"a": "HXK124"}, text)
    assert v is not None and 0.0 <= v <= 1.0
    assert T.qslo({"n": 5}, text) is None             # numerics are not strings


def test_vhr_assumed_exclusion_is_reported():
    text = "no digits here, only words"
    # a numeric with no numbers in the text is 'assumed' -> EXCLUDED
    out = T.vhr_ctx({"n": 42}, text)
    assert out["n_excluded"] == 1 and out["n_graded"] == 0
    assert out["excluded_frac"] == 1.0
    assert out["vhr_sum"] is None and out["pass"] is None
    # a boolean is also assumed
    assert T.vhr_ctx({"b": True}, text)["n_excluded"] == 1
    # a numeric that fails against a text WITH numbers scores h = 1
    out2 = T.vhr_ctx({"n": 42}, "the number is 7")
    assert out2["vhr_sum"] == 1.0 and out2["n_excluded"] == 0
    # an exact string scores h = 0 and passes
    out3 = T.vhr_ctx({"s": "words"}, text)
    assert out3["vhr_sum"] == 0.0 and out3["pass"] is True


def test_classify_value_list_and_object_rules():
    text = "alpha 5"
    assert T.classify_value(["alpha", 5], text) == T.EXACT
    assert T.classify_value(["alpha", "zeta"], text) == T.PARTIAL
    assert T.classify_value([True, None], text) == T.ASSUMED
    assert T.classify_value({"k": "alpha"}, text) == T.ASSUMED
    assert T.classify_value([], text) == T.ASSUMED


def test_saag_cascade_halts_at_stage_one():
    row = T.saag_cascade(call_text("not_a_tool", {"query": "x"}), TOOLS, "x")
    assert row["stage_reached"] == 1 and row["failed_stage"] == 1
    assert row["fnem"] == 0 and row["ncs"] is not None
    assert row["rpr"] is None and row["avem"] is None


def test_saag_cascade_halts_at_stage_two():
    row = T.saag_cascade(call_text("search_users", {"limit": 1}), TOOLS, "1")
    assert row["stage_reached"] == 2 and row["failed_stage"] == 2
    assert row["rpr"] == 0.0 and row["avem"] is None


def test_saag_cascade_reaches_stage_three_and_stamps():
    row = T.saag_cascade(call_text("search_users", {"query": "alice"}), TOOLS, "alice")
    assert row["stage_reached"] == 3 and row["passed"] is True
    assert row["stamp"] == "transition on trigger set, not full set"


def test_saag_stage_three_undefined_when_nothing_gradable():
    tools = [{"name": "ping", "parameters": {"type": "object", "properties": {"flag": {"type": "boolean"}}}}]
    row = T.saag_cascade(call_text("ping", {"flag": True}), tools, "no digits")
    assert row["stage_reached"] == 3
    assert row["passed"] is None and row["failed_stage"] is None
    assert row["vhr_excluded_frac"] == 1.0


def test_saag_query_only_variant_is_the_declared_control():
    action = call_text("search_users", {"query": "alice"})
    hist = T.saag_cascade(action, TOOLS, "alice appeared in the history")
    q_only = T.saag_cascade_query_only(action, TOOLS, "who is that?")
    assert hist["avem"] == 1.0
    assert q_only["avem"] == 0.0


def test_argument_bearing_and_census_denominator():
    assert T.argument_bearing({"a": "xyz"}) is True
    assert T.argument_bearing({}) is False
    assert T.argument_bearing({"a": True}) is False
    assert T.argument_bearing({"a": "ab"}, min_chars=3) is False
    rows = [("s:1", call_text("search_users", {"query": "alice"}), 1),
            ("s:2", call_text("search_users", {}), 1),
            ("s:3", call_text("search_users", {"query": "bob"}), 0),
            ("s:4", "prose", None)]
    c = T.argument_census(rows)
    assert c["C->W"]["n"] == 2 and c["C->W"]["argument_bearing"] == 1
    assert c["C->C"]["n"] == 1 and c["outside"]["n"] == 1


# --------------------------------------------------------------------------
# (C) must_read_before_write + census + report formats
# --------------------------------------------------------------------------

def test_identifier_shape_is_prereg_and_discriminating():
    assert T.identifier_leaves({"id": "HXK1234"}) == ["HXK1234"]
    assert T.identifier_leaves({"id": "a1b2c3d4e5f6"}) == ["a1b2c3d4e5f6"]
    assert T.identifier_leaves({"id": "mcp__venmo__login"}) == ["mcp__venmo__login"]
    # ordinary words and short tokens are not identifiers
    assert T.identifier_leaves({"w": "reservation"}) == []
    assert T.identifier_leaves({"w": "ab1"}) == []


def test_mrbw_fires_only_on_a_gist_only_identifier():
    gate = T.MustReadBeforeWriteGate()
    gate.note_gisted_text("your reservation HXK1234 is confirmed")
    gate.note_raw_text("please cancel reservation ZZZ9999")
    fired = gate.fire({"reservation_id": "HXK1234"})
    assert fired["fire"] is True and fired["gist_only"] == ["HXK1234"]
    # an identifier the agent also saw raw does not fire
    assert gate.fire({"reservation_id": "ZZZ9999"})["fire"] is False
    # an identifier never seen at all does not fire either (that is the
    # grounding check's job, not this gate's)
    assert gate.fire({"reservation_id": "QQQ0001"})["fire"] is False


def test_mrbw_never_fires_when_every_doc_is_raw():
    gate = T.MustReadBeforeWriteGate()
    gate.note_raw_text("your reservation HXK1234 is confirmed")
    assert gate.fire({"reservation_id": "HXK1234"})["fire"] is False


def test_silent_failure_census_over_proxy_rows():
    rows = [
        {"arm": "c2kv", "status": "ok", "error_kind": None, "finish_reason": "stop"},
        {"arm": "c2kv", "status": "ok", "error_kind": None, "finish_reason": "length"},
        {"arm": "c2kv", "status": "upstream_error", "error_kind": "upstream_error",
         "finish_reason": None},
        {"arm": "full", "status": "ok", "error_kind": None, "finish_reason": "stop"},
    ]
    c = T.silent_failure_census(rows, arm="c2kv")
    assert c["n_rows"] == 3 and c["tool_error"] == 1 and c["parse_failure"] == 1
    assert c["neither"] == 1
    assert "CONTAMINATED" in c["contamination_stamp"]


def test_firing_share_decomposition_reports_both_strata():
    out = T.firing_share_decomposition([True, True, False, False],
                                       [1.0, 0.0, 1.0, 1.0])
    assert out["p_fire"] == 0.5
    assert out["firing_stratum"] == {"n": 2, "mean": 0.5}
    assert out["nonfiring_stratum"] == {"n": 2, "mean": 1.0}
    with pytest.raises(ValueError):
        T.firing_share_decomposition([True], [1.0, 2.0])


def test_per_signal_precision_audit_is_raw_counts_plus_the_or():
    y = [1, 1, 0, 0]
    signals = {"a": [True, False, False, False], "b": [False, True, True, False]}
    out = T.per_signal_precision_audit(signals, y)
    assert out["a"] == {"fires": 1, "true": 1, "false": 0, "precision": 1.0,
                        "role": "candidate"}
    assert out["b"]["false"] == 1 and out["b"]["precision"] == 0.5
    assert out["_OR"]["fires"] == 3 and out["_OR"]["precision"] == pytest.approx(2 / 3)
    assert out["_OR_members"] == ["a", "b"]


def test_the_target_reading_baseline_never_enters_the_or():
    """2607.07405 tab:gate-audit + digest §4.0 item 1: the frozen parse-failure
    baseline is gated on ``target_has_tool_call``, so an OR containing it would
    be a detector that reads the target."""
    y = [1, 1, 0, 0]
    signals = {"a": [True, False, False, False]}
    base = {"frozen_parse_fail_baseline": [False, True, False, False]}
    out = T.per_signal_precision_audit(signals, y, baselines=base)
    assert out["frozen_parse_fail_baseline"]["role"] == "baseline"
    assert out["frozen_parse_fail_baseline"]["fires"] == 1      # still audited alone
    assert out["_OR_members"] == ["a"]                          # but not in the OR
    assert out["_OR"]["fires"] == 1
    # opting it in has to be explicit
    opted = T.per_signal_precision_audit(
        signals, y, baselines=base,
        or_members=["a", "frozen_parse_fail_baseline"])
    assert opted["_OR"]["fires"] == 2
    with pytest.raises(ValueError):
        T.per_signal_precision_audit(signals, y, baselines={"a": [True] * 4})
    with pytest.raises(ValueError):
        T.per_signal_precision_audit(signals, y, or_members=["nope"])


# --------------------------------------------------------------------------
# (D) deterministic checks
# --------------------------------------------------------------------------

def test_required_coverage_against_the_session_schema():
    assert T.required_coverage(call_text("search_users", {"query": "x"}), TOOLS) is True
    assert T.required_coverage(call_text("search_users", {"limit": 1}), TOOLS) is False
    assert T.required_coverage(call_text("nope", {"a": 1}), TOOLS) is False
    assert T.required_coverage("prose", TOOLS) is None       # nothing emitted
    assert T.required_coverage(call_text("x", {}), []) is None  # no pool advertised


def test_tool_contract_uses_the_declared_return_schema():
    msg_ok = {"role": "tool", "content": json.dumps({"status": "confirmed"})}
    msg_bad = {"role": "tool", "content": json.dumps({"other": 1})}
    assert T.tool_contract(msg_ok, "get_reservation", TOOLS) is True
    assert T.tool_contract(msg_bad, "get_reservation", TOOLS) is False


def test_tool_contract_name_typed_heuristic_and_unknown():
    tools = [{"name": "get_thing", "parameters": {}}, {"name": "frobnicate", "parameters": {}}]
    assert T.tool_contract({"role": "tool", "content": "payload"}, "get_thing", tools) is True
    assert T.tool_contract({"role": "tool", "content": ""}, "get_thing", tools) is False
    assert T.tool_contract({"role": "tool", "content": "Error: not found"},
                           "get_thing", tools) is False
    # a tool we cannot type must be unknown (None), never False
    assert T.tool_contract({"role": "tool", "content": "x"}, "frobnicate", tools) is None
    assert T.tool_contract(None, "get_thing", tools) is None


def test_total_consistency_standin_is_named_a_standin():
    doc = T.grounding_standin_for_total_consistency.__doc__
    assert "STAND-IN" in doc and "NOT portable" in doc
    v = T.grounding_standin_for_total_consistency(
        call_text("search_users", {"query": "alice"}), "alice was here")
    assert v == 1.0


def test_false_positive_counts_are_printed_as_a_fraction():
    out = T.false_positive_counts([False] * 5, [True] * 5)
    assert out["as_printed"] == "0/5" and out["rate"] == 0.0


# --------------------------------------------------------------------------
# (E) DART admissibility
# --------------------------------------------------------------------------

def test_gist_channel_is_vacuous_under_left_to_right_compression():
    n = 4
    empty = [k for k in range(n) if not T.gist_extraction_consumers(k, n)]
    assert empty == [n - 1]          # only the last block is conflict-free
    assert T.gist_extraction_consumers(0, n) == [1, 2, 3]


def test_execution_consumers_use_the_models_own_actions():
    docs = ["booking HXK1234 for alice", "weather report", "note about bob"]
    later = [call_text("get_reservation", {"reservation_id": "HXK1234"}),
             call_text("search_users", {"query": "bob"})]
    occ = T.execution_consumers(0, docs, later, mode="occurs")
    arg = T.execution_consumers(0, docs, later, mode="argmax")
    assert 0 in occ and 0 in arg          # step 0 grounds in block 0
    assert 1 not in occ                    # step 1 grounds in block 2
    assert T.execution_consumers(2, docs, later, mode="argmax") == [1]
    with pytest.raises(ValueError):
        T.execution_consumers(0, docs, later, mode="nonsense")


def test_no_committed_conflict_reports_both_channels():
    docs = ["booking HXK1234", "unrelated"]
    later = [call_text("get_reservation", {"reservation_id": "HXK1234"})]
    out = T.no_committed_conflict(0, len(docs), docs, later, committed=[True])
    assert out["channel_a_gist"]["no_conflict"] is False
    assert "vacuous" in out["channel_a_gist"]["bound"]
    assert out["channel_b_occurs"]["no_conflict"] is False
    out2 = T.no_committed_conflict(0, len(docs), docs, later, committed=[False])
    assert out2["channel_b_occurs"]["no_conflict"] is True   # nothing committed


def test_effect_policy_skeleton_has_no_invented_tool_names():
    path = _AGENT.parent / "configs/t34/effect_policy_skeleton.json"
    policy = T.load_effect_policy(path)
    assert policy["irreversible"] == [] and policy["read_only"] == []
    out = T.effect_allowed(["book_flight"], policy)
    assert out["allowed"] is False and out["unknown_effect"] == ["book_flight"]
    # the block must be reported as a MISSING POLICY, not as an effect verdict
    assert policy["is_unfilled_skeleton"] is True and "unfilled_note" in policy
    assert out["policy_unfilled"] is True and "missing input" in out["policy_note"]
    filled = {"irreversible": ["book_flight"], "read_only": ["get_reservation"]}
    assert T.effect_allowed(["get_reservation"], filled)["allowed"] is True
    assert T.effect_allowed(["book_flight"], filled)["blocked_by_effect"] == ["book_flight"]
    assert "policy_unfilled" not in T.effect_allowed(["book_flight"], filled)


def test_admissible_recover_definition_6():
    conflict = {"channel_b_occurs": {"no_conflict": True},
                "channel_a_gist": {"no_conflict": False}}
    effects = {"allowed": True}
    ok = T.admissible_recover(identified=True, checkpoint=2,
                              stable_checkpoints=[1, 2], instance_checkpoints=[2],
                              conflict=conflict, effects=effects)
    assert ok["admissible"] is True and ok["blocked_by"] is None
    bad = T.admissible_recover(identified=True, checkpoint=2,
                               stable_checkpoints=[1, 2], instance_checkpoints=[2],
                               conflict=conflict, effects={"allowed": False})
    assert bad["admissible"] is False and bad["blocked_by"] == ["EffectAllowed"]
    vac = T.admissible_recover(identified=True, checkpoint=2,
                               stable_checkpoints=[1, 2], instance_checkpoints=[2],
                               conflict=conflict, effects=effects,
                               channel="channel_a_gist")
    assert vac["admissible"] is False   # the vacuous bound blocks everything
    with pytest.raises(KeyError):
        T.admissible_recover(identified=True, checkpoint=2, stable_checkpoints=[2],
                             instance_checkpoints=[2], conflict=conflict,
                             effects=effects, channel="nope")


def test_admissibility_panel_shape_without_an_audit():
    events = [{"admissible": True}, {"admissible": False, "blocked_by": ["EffectAllowed"]}]
    panel = T.admissibility_panel(events)
    assert panel["admitted"] == 1 and panel["blocked"] == 1
    assert panel["unsafe_admitted"] is None and panel["false_blocked"] is None
    assert panel["block_reasons"] == {"EffectAllowed": 1}
    audited = T.admissibility_panel([{"admissible": True, "unsafe": False},
                                     {"admissible": False, "ground_truth_ok": True,
                                      "blocked_by": ["Stable"]}])
    assert audited["unsafe_admitted"] == 0 and audited["false_blocked"] == 1


# --------------------------------------------------------------------------
# (F) Tracy's grounding, verbatim and visible
# --------------------------------------------------------------------------

def _tracy_action(name, args):
    return [json.dumps({"name": name, "arguments": args})]


def test_tracy_verbatim_edge_rules():
    msgs = [{"role": "user", "content": "book HXK1234 please"}]
    # parse failure and not an empty-execute response -> 0.0
    assert T.argument_grounding_score_verbatim(["not json"], msgs) == 0.0
    # empty execute response -> 1.0
    assert T.argument_grounding_score_verbatim([], msgs) == 1.0
    assert T.argument_grounding_score_verbatim([""], msgs) == 1.0
    # parsed but no value with len >= 3 -> 1.0 (her rule, kept under her name)
    assert T.argument_grounding_score_verbatim(_tracy_action("t", {"a": "xy"}), msgs) == 1.0
    # ordinary hit rate, lower-case substring
    assert T.argument_grounding_score_verbatim(
        _tracy_action("t", {"a": "HXK1234", "b": "zzzz"}), msgs) == 0.5


def test_tracy_verbatim_uses_only_the_last_twelve_messages_and_three_roles():
    old = [{"role": "user", "content": "HXK1234"}]
    filler = [{"role": "user", "content": "filler"} for _ in range(12)]
    assert T.argument_grounding_score_verbatim(
        _tracy_action("t", {"a": "HXK1234"}), old + filler) == 0.0
    system = [{"role": "system", "content": "HXK1234"}]
    assert T.argument_grounding_score_verbatim(
        _tracy_action("t", {"a": "HXK1234"}), system) == 0.0


def test_grounding_visible_returns_none_instead_of_tracys_one():
    docs = ["booking HXK1234 for alice"]
    assert T.grounding_visible(call_text("search_users", {"query": "alice"}),
                               docs, "") == 1.0
    # no gradable value -> None in the feature frame (NOT 1.0)
    assert T.grounding_visible(call_text("search_users", {}), docs, "") is None
    # word-boundary typing, not raw substring: "lic" is inside "alice" but is
    # not a word-boundary match
    assert T.grounding_visible(call_text("search_users", {"query": "lic"}),
                               docs, "") == 0.0


def test_grounding_dropped_delta_sign_and_none():
    action = call_text("search_users", {"query": "carol"})
    assert T.grounding_dropped_delta(action, ["carol here"], ["nothing"], "") == 1.0
    assert T.grounding_dropped_delta(action, ["nothing"], ["carol here"], "") == -1.0


def test_dropped_states_never_collapse_into_one_none():
    """'nothing was dropped' is a DEFINED cell; 'the text was not dumped' and
    'the count is unknown' are missing inputs (digest §4.6)."""
    action = call_text("search_users", {"query": "carol"})

    # (a) the step dropped nothing: defined, delta == grounded_in_visible
    nothing = T.grounding_dropped_detail(action, ["carol here"], [], "",
                                         n_dropped_blocks=0)
    assert nothing["state"] == T.DROPPED_NO_BLOCKS
    assert nothing["dropped_side_available"] is True
    assert nothing["grounded_in_dropped_docs"] == 0.0 and nothing["delta"] == 1.0

    # (b) blocks were dropped but the dump carried no text: MISSING INPUT
    not_dumped = T.grounding_dropped_detail(action, ["carol here"], None, "",
                                            n_dropped_blocks=3)
    assert not_dumped["state"] == T.DROPPED_TEXT_NOT_DUMPED
    assert not_dumped["dropped_side_available"] is False and not_dumped["delta"] is None

    # (c) neither text nor count: NOT 'nothing was dropped'
    unknown = T.grounding_dropped_detail(action, ["carol here"], None, "")
    assert unknown["state"] == T.DROPPED_COUNT_UNKNOWN
    assert unknown["dropped_side_available"] is False and unknown["delta"] is None
    assert unknown["n_dropped_blocks"] is None

    # (d) the artifact contradicts itself
    bad = T.grounding_dropped_detail(action, ["carol here"], ["one"], "",
                                     n_dropped_blocks=2)
    assert bad["state"] == T.DROPPED_TEXT_COUNT_MISMATCH
    assert bad["dropped_side_available"] is False and bad["delta"] is None

    # (e) no gradable value: both sides undefined, but the side IS available
    empty = T.grounding_dropped_detail(call_text("search_users", {}), ["x"], [], "",
                                       n_dropped_blocks=0)
    assert empty["state"] == T.DROPPED_NO_VALUES and empty["delta"] is None
    assert empty["dropped_side_available"] is True


def test_grounding_strata_reports_defined_denominators():
    scores = {"a:1": 1.0, "a:2": None, "a:3": 0.0}
    labels = {"a:1": 1, "a:2": 1, "a:3": 0}
    dropped = {"a:1": True, "a:2": True, "a:3": False}
    out = T.grounding_strata(scores, labels, dropped)
    assert out["dropped_nonempty/C->W"] == {"n": 2, "n_defined": 1, "mean": 1.0}
    assert out["dropped_empty/C->C"]["n_defined"] == 1


# --------------------------------------------------------------------------
# (G) feature emission
# --------------------------------------------------------------------------

def _synthetic_pair():
    battery = [
        {"qid": "sess1:1", "session_id": "sess1",
         "prediction": call_text("search_users", {"query": "alice smith"})},
        {"qid": "sess1:2", "session_id": "sess1", "prediction": "prose only"},
        {"qid": "sess2:1", "session_id": "sess2",
         "prediction": call_text("get_reservation", {"reservation_id": "HXK1234"})},
    ]
    sidecar = {
        "sess1:1": {"qid": "sess1:1", "docs": ["alice smith wrote in"], "query": "find her",
                    "tools": TOOLS, "dropped_docs": [], "doc_lengths": [5],
                    "system_prompt": "", "kept_history_tokens": 5, "session_id": "sess1"},
        "sess1:2": {"qid": "sess1:2", "docs": ["something"], "query": "q",
                    "tools": TOOLS, "dropped_docs": [1], "doc_lengths": [3],
                    "system_prompt": "", "kept_history_tokens": 3, "session_id": "sess1"},
        "sess2:1": {"qid": "sess2:1", "docs": ["booking HXK1234 confirmed"], "query": "cancel it",
                    "tools": TOOLS, "dropped_docs": [2], "doc_lengths": [6],
                    "system_prompt": "", "kept_history_tokens": 6, "session_id": "sess2"},
    }
    return battery, sidecar


def test_build_feature_rows_columns_and_values():
    battery, sidecar = _synthetic_pair()
    rows, meta = T.build_feature_rows(battery, sidecar, arm="c2kv")
    assert len(rows) == 3
    by = {r["qid"]: r for r in rows}
    assert set(by["sess1:1"]) - {"qid", "session_id"} == set(T.FEATURE_COLUMNS)
    assert by["sess1:1"]["route_fire"] == 0
    assert by["sess1:2"]["route_deviation"] == 1 and by["sess1:2"]["fnem"] == -1
    assert by["sess2:1"]["mrbw_fire"] == 1          # HXK1234 exists only in a gist
    assert by["sess2:1"]["dropped_nonempty"] == 1
    assert by["sess1:1"]["grounding_visible"] == 1.0
    assert by["sess1:1"]["cpu_ms"] > 0
    assert meta["routes"]["sess1:2"] == T.ROUTE_DEVIATION


def test_unparseable_row_gets_tracys_zero_not_her_empty_execute_one():
    battery, sidecar = _synthetic_pair()
    rows, _ = T.build_feature_rows(battery, sidecar)
    by = {r["qid"]: r for r in rows}
    assert by["sess1:2"]["grounding_12msg"] == 0.0
    assert by["sess1:2"]["grounding_visible"] is None


def test_s0_twin_on_the_full_arm_never_fires_the_gate():
    battery, sidecar = _synthetic_pair()
    rows, _ = T.build_feature_rows(battery, sidecar, arm="full")
    assert all(r["mrbw_fire"] == 0 for r in rows)


def test_feature_rows_pass_the_leakage_guard(tmp_path):
    from t34_common import write_features_jsonl
    battery, sidecar = _synthetic_pair()
    rows, _ = T.build_feature_rows(battery, sidecar)
    out = tmp_path / "features_l1.jsonl"
    n = write_features_jsonl(out, rows, context="t34 §4.6 test")
    assert n == 3
    first = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "target" not in first and "tool_name_match" not in first


def test_leakage_guard_still_catches_the_historical_specimens():
    from t33_labels import guard_columns
    for bad in ("a_made_call", "tool_name_match", "full_prediction"):
        with pytest.raises(ValueError):
            guard_columns(list(T.FEATURE_COLUMNS) + [bad], context="t34 §4.6 test")


def test_none_is_kept_as_null_not_a_sentinel(tmp_path):
    from t34_common import write_features_jsonl
    battery = [{"qid": "s:1", "session_id": "s",
                "prediction": call_text("search_users", {})}]
    sidecar = {"s:1": {"qid": "s:1", "docs": ["d"], "query": "q", "tools": TOOLS,
                       "dropped_docs": [], "doc_lengths": [1], "system_prompt": "",
                       "kept_history_tokens": 1, "session_id": "s"}}
    rows, _ = T.build_feature_rows(battery, sidecar)
    assert rows[0]["avem"] is None and rows[0]["grounding_visible"] is None
    out = tmp_path / "f.jsonl"
    write_features_jsonl(out, rows, context="t34 §4.6 test")
    loaded = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert loaded["avem"] is None


def test_orientations_file_matches_the_module_and_covers_every_column():
    path = _AGENT.parent / "configs/t34/orientations_triggers.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == T.ORIENTATIONS
    assert set(T.FEATURE_COLUMNS) <= set(T.ORIENTATIONS)
    assert all(v in (-1, 1) for v in T.ORIENTATIONS.values())
    assert all(isinstance(v, int) for v in on_disk.values())   # must stay flat ints


def test_cli_help_runs():
    with pytest.raises(SystemExit) as exc:
        T.main(["--help"])
    assert exc.value.code == 0
    with pytest.raises(SystemExit) as exc:
        DS.parse_args(["--help"])
    assert exc.value.code == 0


# --------------------------------------------------------------------------
# (H) sidecar dump — pure helpers only (the harness path needs torch)
# --------------------------------------------------------------------------

def test_kept_and_dropped_indices_tail_and_head():
    kept, dropped = DS.kept_and_dropped_indices(20, 16, "tail")
    assert kept == [0] + list(range(5, 20))
    assert dropped == list(range(1, 5))
    assert len(kept) == 16
    kept, dropped = DS.kept_and_dropped_indices(20, 16, "head")
    assert kept == list(range(16)) and dropped == list(range(16, 20))
    assert DS.kept_and_dropped_indices(5, 16, "tail") == (list(range(5)), [])
    assert DS.kept_and_dropped_indices(5, 1, "tail") == ([4], [0, 1, 2, 3])
    with pytest.raises(ValueError):
        DS.kept_and_dropped_indices(20, 16, "middle")


def test_sidecar_row_enforces_the_fixed_schema():
    row = DS.sidecar_row(qid="s:1", session_id="s", docs=["a"], query="q",
                         tools=[], system_prompt="sys", doc_lengths=[3],
                         dropped_docs=[], kept_history_tokens=3)
    assert set(row) == set(DS.SCHEMA_KEYS)
    with_ext = DS.sidecar_row(qid="s:1", session_id="s", docs=["a"], query="q",
                              tools=[], system_prompt="sys", doc_lengths=[3],
                              dropped_docs=[1], kept_history_tokens=3,
                              dropped_doc_texts=["dropped"])
    assert set(with_ext) == set(DS.SCHEMA_KEYS) | {"dropped_doc_texts"}
    with pytest.raises(ValueError):
        DS.validate_row({**row, "surprise": 1})
    with pytest.raises(ValueError):
        DS.validate_row({k: v for k, v in row.items() if k != "docs"})
    with pytest.raises(ValueError):
        DS.sidecar_row(qid="s:1", session_id="s", docs=["a", "b"], query="q",
                       tools=[], system_prompt="", doc_lengths=[3],
                       dropped_docs=[], kept_history_tokens=3)


def test_write_and_reload_sidecar_round_trip(tmp_path):
    rows = [DS.sidecar_row(qid="s:1", session_id="s", docs=["alpha"], query="q",
                           tools=TOOLS, system_prompt="sys", doc_lengths=[4],
                           dropped_docs=[2], kept_history_tokens=4)]
    path = tmp_path / "sidecar_c2kv.jsonl"
    assert DS.write_sidecar(path, rows) == 1
    back = T.load_sidecar(path)
    assert back["s:1"]["docs"] == ["alpha"]
    assert back["s:1"]["dropped_docs"] == [2]
    from t34_common import load_decoded_docs
    assert load_decoded_docs(path) == {"s:1": ["alpha"]}


# --------------------------------------------------------------------------
# fidelity-review additions
# --------------------------------------------------------------------------

def test_vhr_sum_is_arity_confounded_and_the_mean_is_emitted_too():
    """2607.18245 §2.3 defines the PASS criterion on the sum; Table `intrinsic`
    reports VHR as a RATE (0.166-0.336).  Both must reach the feature frame."""
    text = "no match here"
    one = T.vhr_ctx({"a": "zzz"}, text)
    three = T.vhr_ctx({"a": "zzz", "b": "yyy", "c": "xxx"}, text)
    assert one["vhr_sum"] == 1.0 and three["vhr_sum"] == 3.0     # sum grows with arity
    assert one["vhr_mean"] == 1.0 and three["vhr_mean"] == 1.0   # the rate does not
    assert "vhr_ctx" in T.FEATURE_COLUMNS and "vhr_ctx_mean" in T.FEATURE_COLUMNS


def test_query_only_control_shares_the_cascade_denominator():
    """The declared Q-only control must be defined on exactly the rows the
    history version is defined on (2607.18245 §2.1-2.3: stages 1-2 do not read
    the grounding text, so the survivor sets are identical)."""
    battery = [
        # stage-2 failure: missing the required key -> stage 3 undefined on BOTH
        {"qid": "s:1", "session_id": "s", "prediction": call_text("search_users", {"limit": 1})},
        # stage-3 row: grounded in history, not in the query
        {"qid": "s:2", "session_id": "s",
         "prediction": call_text("search_users", {"query": "alice"})},
    ]
    side = {"qid": None, "docs": ["alice wrote in"], "query": "who?", "tools": TOOLS,
            "dropped_docs": [], "doc_lengths": [3], "system_prompt": "",
            "kept_history_tokens": 3, "session_id": "s"}
    sidecar = {"s:1": dict(side, qid="s:1"), "s:2": dict(side, qid="s:2")}
    rows = {r["qid"]: r for r in T.build_feature_rows(battery, sidecar)[0]}
    assert rows["s:1"]["avem"] is None and rows["s:1"]["avem_query_only"] is None
    assert rows["s:2"]["avem"] == 1.0 and rows["s:2"]["avem_query_only"] == 0.0


def test_missing_sidecar_row_is_all_none_not_manufactured():
    """With no sidecar there is no tool pool and no visible text: computing the
    checks anyway would manufacture fnem=0 and an UNRESOLVED route."""
    battery = [{"qid": "s:1", "session_id": "s",
                "prediction": call_text("search_users", {"query": "alice"})}]
    rows, meta = T.build_feature_rows(battery, {})
    assert set(rows[0]) - {"qid", "session_id"} == set(T.FEATURE_COLUMNS)
    assert all(rows[0][c] is None for c in T.FEATURE_COLUMNS)
    assert "s:1" not in meta["routes"]          # never routed as PASS by default


def test_non_candidate_columns_are_named():
    assert set(T.NON_CANDIDATE_COLUMNS) == {"argument_bearing", "censored_at_cap",
                                            "cpu_ms",
                                            "grounding_dropped_side_available"}
    assert set(T.NON_CANDIDATE_COLUMNS) <= set(T.FEATURE_COLUMNS)
    # the declaration must reach the artifact the scorer reads, not just a comment
    _b, sidecar = _synthetic_pair()
    _rows, meta = T.build_feature_rows(_b, sidecar)
    assert meta["non_candidate_columns"] == list(T.NON_CANDIDATE_COLUMNS)


def test_a_of_f_and_the_max_recency_selection_rule():
    """2605.23311 §4.5: A(f) = {c : AdmissibleRecover}, c*(f) = max A(f), and an
    empty A(f) falls back to whole-task rerun."""
    clean = {"channel_b_occurs": {"no_conflict": True}}
    dirty = {"channel_b_occurs": {"no_conflict": False}}
    ok = {"allowed": True}
    conflicts = {0: clean, 1: dirty, 2: clean}
    effects = {0: ok, 1: ok, 2: ok}
    a_of_f = T.admissible_set(identified=True, candidates=[0, 1, 2],
                              stable_checkpoints=[0, 1, 2], instance_checkpoints=[0, 1, 2],
                              conflict_by_checkpoint=conflicts,
                              effects_by_checkpoint=effects)
    assert a_of_f == [0, 2]
    assert T.select_checkpoint(a_of_f)["c_star"] == 2          # recency
    empty = T.select_checkpoint([])
    assert empty["c_star"] is None and "whole-task rerun" in empty["fallback"]
    # a single inadmissible candidate must NOT be reported as "A(f) empty"
    one = T.admissible_recover(identified=True, checkpoint=1, stable_checkpoints=[0, 1],
                               instance_checkpoints=[0, 1], conflict=dirty, effects=ok)
    assert one["admissible"] is False and "fallback" not in one


def test_censoring_stratifier_is_emitted_from_the_arms_own_row():
    """digest §4.0 caliber caveat: at max_new_tokens=128 a DEVIATION route can
    mean 'cut off mid-JSON'.  The stratifier comes from the compressed arm's own
    generated_tokens (never a target field) and is None when the cap is
    unknown."""
    battery, sidecar = _synthetic_pair()
    battery[0] = dict(battery[0], generated_tokens=128)
    battery[1] = dict(battery[1], generated_tokens=40)
    rows = {r["qid"]: r for r in T.build_feature_rows(battery, sidecar,
                                                     cap_tokens=128)[0]}
    assert rows["sess1:1"]["censored_at_cap"] == 1
    assert rows["sess1:2"]["censored_at_cap"] == 0
    assert rows["sess2:1"]["censored_at_cap"] is None      # no generated_tokens
    # no cap supplied -> the column is None everywhere, never 0
    no_cap = {r["qid"]: r for r in T.build_feature_rows(battery, sidecar)[0]}
    assert all(r["censored_at_cap"] is None for r in no_cap.values())
    assert "censored_at_cap" in T.NON_CANDIDATE_COLUMNS


def test_meta_names_every_missing_input_loudly():
    """Every input this box does not have must surface as a NAMED flag beside
    the frame, not as a silently null column."""
    battery, sidecar = _synthetic_pair()
    battery.append({"qid": "sess3:1", "session_id": "sess3", "prediction": "x"})
    _rows, meta = T.build_feature_rows(battery, sidecar, cap_tokens=128)
    da = meta["data_availability"]
    assert da["n_rows_without_sidecar_all_columns_null"] == 1
    assert da["fuzz_backend"] in ("rapidfuzz", "difflib")
    assert da["rapidfuzz_available"] == (da["fuzz_backend"] == "rapidfuzz")
    # sess1:2 and sess2:1 drop blocks and the synthetic sidecar carries no
    # dropped_doc_texts -> the dropped side is a missing INPUT on those rows
    assert da["n_rows_dropped_text_not_dumped"] == 2
    assert da["dropped_side_available_everywhere"] is False
    assert da["censoring_stratifier_available"] is True and da["cap_tokens"] == 128


def test_sidecar_docs_are_all_visible_and_dropped_docs_are_another_index_space():
    """results/t34/sidecar_*.jsonl: ``docs`` holds ONLY kept blocks, so
    filtering it by ``dropped_docs`` deletes text the model DID see.  The
    kept_history_indices bridge is the only legal way to relate the two."""
    row = DS.sidecar_row(qid="s:1", session_id="s",
                         docs=["kept0", "kept5"], query="q", tools=[],
                         system_prompt="", doc_lengths=[2, 2],
                         dropped_docs=[1, 2, 3, 4], kept_history_tokens=4,
                         kept_history_indices=[0, 5])
    assert set(row) == set(DS.SCHEMA_KEYS) | {"kept_history_indices"}
    assert "kept_history_indices" in DS.EXTENSION_KEYS
    assert set(DS.SCHEMA_KEYS) == {"qid", "session_id", "docs", "query", "tools",
                                   "system_prompt", "doc_lengths", "dropped_docs",
                                   "kept_history_tokens"}
    assert T.sidecar_history_map(row) == {0: 0, 5: 1}
    # the naive consumer bug the key exists to prevent
    naive = [d for i, d in enumerate(row["docs"]) if i not in row["dropped_docs"]]
    assert naive != row["docs"]            # it would delete a VISIBLE block
    # a dump without the key cannot be aligned and must say so, not guess
    assert T.sidecar_history_map({k: v for k, v in row.items()
                                  if k != "kept_history_indices"}) is None
    # a bridge that disagrees with docs / dropped_docs is refused outright
    with pytest.raises(ValueError):
        DS.validate_row({**row, "kept_history_indices": [0]})
    with pytest.raises(ValueError):
        DS.validate_row({**row, "kept_history_indices": [0, 1]})
    with pytest.raises(ValueError):
        DS.validate_row({**row, "dropped_doc_texts": ["only one"]})


# --------------------------------------------------------------------------
# report CLI wiring (needs the frozen battery; skipped when it is absent)
# --------------------------------------------------------------------------

_ROOT = _AGENT.parent
_BATTERY = _ROOT / "results/bdf_pilot/d_r2/battery_c2kv.jsonl"


@pytest.mark.skipif(not _BATTERY.exists(), reason="frozen battery not on this box")
def test_report_keeps_one_denominator_and_never_ors_the_baseline(tmp_path):
    """A row with no sidecar carries null features: it must be excluded from
    every audit, not counted as a non-firing detector row; and the frozen
    parse-failure baseline must stay out of the combined `_OR`."""
    from t34_common import FrozenAssets
    frame = FrozenAssets(_ROOT).load()
    cw = [r["qid"] for r in frame.labels if r["label_cw"] == 1][:3]
    cc = [r["qid"] for r in frame.labels if r["label_cw"] == 0][:3]
    fired = set(cw[:2])
    rows, routes = [], {}
    for qid in cw + cc:
        if qid == cw[-1]:                       # the no-sidecar row: all None
            rows.append({"qid": qid, "session_id": T_session(qid),
                         **{c: None for c in T.FEATURE_COLUMNS}})
            continue
        routes[qid] = T.ROUTE_DEVIATION if qid in fired else T.ROUTE_PASS
        rows.append({"qid": qid, "session_id": T_session(qid),
                     **{c: 0 for c in T.FEATURE_COLUMNS},
                     "route_fire": int(qid in fired),
                     "route_deviation": int(qid in fired),
                     "cpu_ms": 1.0, "censored_at_cap": 0})
    feat = tmp_path / "features_l1.jsonl"
    with io.open(feat, "w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (tmp_path / "features_l1.meta.json").write_text(
        json.dumps({"routes": routes, "data_availability": {"probe": True}}),
        encoding="utf-8")
    out = tmp_path / "report.json"
    assert T.main(["--root", str(_ROOT), "report", "--features", str(feat),
                   "--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["n_trigger_rows"] == 6
    assert rep["n_trigger_rows_without_route"] == 1
    audit = rep["per_signal_precision_audit"]
    # the baseline is audited but is not a member of the OR
    assert audit["frozen_parse_fail_baseline"]["role"] == "baseline"
    assert "frozen_parse_fail_baseline" not in audit["_OR_members"]
    # one denominator everywhere: 5 routed rows, not 6
    assert audit["route_any_fire"]["fires"] + audit["route_any_fire"]["true"] >= 0
    assert rep["crosstab_2x3"]["n_cw"] + rep["crosstab_2x3"]["n_cc"] == 5
    assert rep["crosstab_2x3_by_censoring"]["uncensored"]["n_cw"] >= 1
    assert rep["data_availability"] == {"probe": True}
    assert set(rep["non_candidate_columns"]) == set(T.NON_CANDIDATE_COLUMNS)


def T_session(qid):
    return qid.rsplit(":", 1)[0]
