"""reqlog.summarize on a synthetic proxy request log."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reqlog  # noqa: E402


def _rows():
    base = {"status": "ok", "doc_packing": "turn", "c2kv_query_proj": "gist"}
    return [
        {**base, "conv_id": "a", "n_docs": 3, "dropped_docs": 0, "wall_sec": 1.0,
         "gist_tokens": 100, "original_tokens": 800, "kv_resident_tokens": 900},
        {**base, "conv_id": "a", "n_docs": 12, "dropped_docs": 4, "wall_sec": 3.0,
         "gist_tokens": 400, "original_tokens": 3200, "kv_resident_tokens": 2000},
        {**base, "conv_id": "b", "n_docs": 0, "dropped_docs": 0, "wall_sec": 0.5,
         "gist_tokens": 0, "original_tokens": 0},
        {"status": "cache_miss", "error_kind": "cache_miss", "conv_id": "b"},
        {"status": "assemble_error", "error_kind": "assemble_error"},
    ]


def test_summarize_regime_facts(tmp_path):
    log = tmp_path / "proxy.jsonl"
    log.write_text("".join(json.dumps(r) + "\n" for r in _rows()) + "not json\n\n",
                   encoding="utf-8")
    s = reqlog.summarize_file(log)
    assert s["n_requests"] == 5 and s["n_ok"] == 3 and s["n_error"] == 2
    assert s["error_kinds"] == {"cache_miss": 1, "assemble_error": 1}
    assert s["n_conversations"] == 2
    assert s["compressed_requests"] == 2  # n_docs > 0
    assert s["dropped_requests"] == 1 and s["dropped_share"] == pytest.approx(1 / 3)
    assert s["dropped_docs_mean"] == pytest.approx(4 / 3) and s["dropped_docs_max"] == 4
    assert s["n_docs_max"] == 12
    assert s["wall_p50"] == 1.0 and s["wall_p90"] == 3.0
    assert s["logical_over_gist"] == pytest.approx(4000 / 500)
    assert s["kv_resident_p50"] == 2000
    assert s["c2kv_query_proj"] == ["gist"] and s["mixed_query_proj"] is False
    assert s["effective_query_proj_counts"] == {"absent": 3}
    assert s["doc_packing"] == ["turn"]


def test_summarize_flags_mixed_modes_and_missing_file(tmp_path):
    rows = _rows()[:2]
    rows[1] = {**rows[1], "c2kv_query_proj": "base"}
    s = reqlog.summarize(rows)
    assert s["mixed_query_proj"] is True and s["c2kv_query_proj"] == ["base", "gist"]
    empty = reqlog.summarize_file(tmp_path / "missing.jsonl")
    assert empty["n_requests"] == 0 and empty["wall_p50"] is None
    assert empty["dropped_share"] is None and empty["logical_over_gist"] is None


def test_effective_query_proj_counts_are_separate_from_the_flag():
    """B4: ``c2kv_query_proj`` is the server FLAG (one value per run);
    ``c2kv_query_proj_effective`` is the per-request provenance and can differ
    row by row under one flag (a repair-only request flips it).  The two must
    never collapse into one number: ``mixed_query_proj`` stays keyed on the
    flag, and the effective values are counted separately, with ``absent`` for
    a row that carries none."""
    base = {"status": "ok", "c2kv_query_proj": "gist"}
    rows = [
        {**base, "c2kv_query_proj_effective": "gist"},
        {**base, "c2kv_query_proj_effective": "gist"},
        {**base, "c2kv_query_proj_effective": "base"},
        {**base},                                    # server reported nothing
        {"status": "cache_miss", "error_kind": "cache_miss",
         "c2kv_query_proj_effective": "base"},       # not ok -> not counted
    ]
    s = reqlog.summarize(rows)
    assert s["effective_query_proj_counts"] == {"absent": 1, "base": 1, "gist": 2}
    # one FLAG value across the run, so the regime is not mixed even though
    # the effective projection differed per request
    assert s["c2kv_query_proj"] == ["gist"] and s["mixed_query_proj"] is False


def test_effective_query_proj_counts_empty_without_ok_rows():
    s = reqlog.summarize([{"status": "assemble_error"}])
    assert s["effective_query_proj_counts"] == {}


# ---------------------------------------------------------------- cost join

def _log():
    """Two conversations for task A (the id shifts once, as proxy.
    conversation_id does), one for task B, plus noise that must NOT join:
    an error row and a row for a conversation no task claims."""
    ok = {"status": "ok"}
    return [
        {**ok, "conv_id": "a1", "wall_sec": 1.5, "gist_tokens": 10,
         "original_tokens": 100, "n_docs": 0, "dropped_docs": 0},
        {**ok, "conv_id": "a2", "wall_sec": 2.5, "gist_tokens": 40,
         "original_tokens": 400, "n_docs": 7, "dropped_docs": 3},
        {**ok, "conv_id": "b1", "wall_sec": 0.5, "gist_tokens": 5,
         "original_tokens": 50, "n_docs": 2, "dropped_docs": 0},
        {"status": "cache_miss", "error_kind": "cache_miss", "conv_id": "a2",
         "wall_sec": 99.0},
        {**ok, "conv_id": "zz", "wall_sec": 7.0},
    ]


def test_join_by_conversation_fills_cost_columns():
    rows = [{"task_id": "A"}, {"task_id": "B"}, {"task_id": "C"}]
    keys = {"A": ["a1", "a2"], "B": ["b1"], "C": ["c1"]}
    report = reqlog.join_by_conversation(rows, _log(),
                                         lambda r: keys[r["task_id"]])
    # A: two requests, summed; n_docs_max is a MAX, not a sum
    assert rows[0]["n_requests"] == 2
    assert rows[0]["wall_sec"] == pytest.approx(4.0)
    assert rows[0]["gist_tokens"] == 50 and rows[0]["original_tokens"] == 500
    assert rows[0]["n_docs_max"] == 7 and rows[0]["dropped_docs"] == 3
    # B: one request
    assert rows[1]["n_requests"] == 1 and rows[1]["wall_sec"] == pytest.approx(0.5)
    # C: none -> n_requests 0 and NO cost field (aggregate skips missing,
    # it would have averaged in a fake zero)
    assert rows[2]["n_requests"] == 0
    assert set(reqlog.COST_FIELDS) - set(reqlog.SUM_FIELDS) == {"n_docs_max",
                                                                "n_requests"}
    for field in reqlog.COST_FIELDS:
        assert (field in rows[2]) is (field == "n_requests"), field
    assert report["n_rows"] == 3 and report["n_keyed"] == 3
    assert report["n_joined"] == 2
    # the error row never joins; "zz" belongs to no task -> not full coverage
    assert report["n_log_ok"] == 4 and report["n_log_joined"] == 3
    assert report["full_coverage"] is False


def test_join_by_conversation_full_coverage_and_status():
    rows = [{"task_id": "A"}, {"task_id": "B"}]
    log = [r for r in _log() if r.get("conv_id") != "zz"]
    keys = {"A": ["a1", "a2"], "B": ["b1"]}
    report = reqlog.join_by_conversation(rows, log, lambda r: keys[r["task_id"]])
    assert report["full_coverage"] is True
    assert reqlog.cost_join_status(report) == "joined: 2/2 tasks, 3/3 logged requests"


def test_join_by_conversation_partial_and_unjoinable_status():
    rows = [{"task_id": "A"}, {"task_id": "B"}]
    keys = {"A": ["a1", "a2"], "B": ["b1"]}
    partial = reqlog.join_by_conversation(rows, _log(),
                                          lambda r: keys[r["task_id"]])
    assert reqlog.cost_join_status(partial).startswith("partial: 2/2 tasks, 3/4")
    # a WRONG key matches nothing — never a wrong number
    wrong = reqlog.join_by_conversation([{"task_id": "A"}], _log(),
                                        lambda r: ["nope"])
    assert wrong["n_joined"] == 0
    assert reqlog.cost_join_status(wrong).startswith("not joinable: ")


def test_join_by_conversation_never_attributes_an_ambiguous_id():
    rows = [{"task_id": "A"}, {"task_id": "B"}]
    # both tasks claim b1 (e.g. two tasks with an identical opening) — it is
    # attributed to NEITHER
    report = reqlog.join_by_conversation(
        rows, _log(), lambda r: ["a1", "b1"] if r["task_id"] == "A" else ["b1"])
    assert report["ambiguous_conv_ids"] == ["b1"]
    assert rows[0]["n_requests"] == 1 and rows[0]["wall_sec"] == pytest.approx(1.5)
    assert rows[1]["n_requests"] == 0
    assert report["full_coverage"] is False


def test_join_by_conversation_tolerates_unkeyed_rows_and_string_keys():
    rows = [{"task_id": "A"}, {"task_id": "B"}]
    report = reqlog.join_by_conversation(
        rows, _log(), lambda r: "b1" if r["task_id"] == "B" else None)
    assert report["n_keyed"] == 1 and report["n_joined"] == 1
    assert rows[0]["n_requests"] == 0
    assert rows[1]["n_requests"] == 1


def test_join_by_conversation_ignores_non_numeric_and_duplicate_ids():
    rows = [{"task_id": "A"}]
    log = [{"status": "ok", "conv_id": "a1", "wall_sec": None,
            "gist_tokens": True, "n_docs": "7"}]
    reqlog.join_by_conversation(rows, log, lambda r: ["a1", "a1"])
    assert rows[0]["n_requests"] == 1  # the id is not counted twice
    assert "wall_sec" not in rows[0]      # None is not a measurement
    assert "gist_tokens" not in rows[0]   # bool is not a token count
    assert "n_docs_max" not in rows[0]    # "7" is not an int


def test_cost_summary_rolls_up_what_aggregate_does_not_mean():
    """metrics.aggregate means wall/gist/original only and nobody persists
    the rows, so n_requests / n_docs_max / dropped_docs reach an artefact
    ONLY through this block."""
    rows = [{"task_id": "A"}, {"task_id": "B"}, {"task_id": "C"}]
    keys = {"A": ["a1", "a2"], "B": ["b1"], "C": ["c1"]}
    report = reqlog.join_by_conversation(rows, _log(),
                                         lambda r: keys[r["task_id"]])
    summary = reqlog.cost_summary(rows, report)
    # the NUMERIC denominator of the *_mean keys: cost means cover 2 tasks
    # while semantic_score covers all 3
    assert summary["n_cost_joined"] == 2
    assert summary["n_cost_requests"] == 3          # 2 (A) + 1 (B) + 0 (C)
    assert summary["n_docs_max"] == 7               # a MAX over tasks
    assert summary["dropped_docs_total"] == 3       # A dropped 3, B dropped 0
    assert isinstance(summary["n_docs_max"], int)   # counts stay integers
    assert isinstance(summary["dropped_docs_total"], int)
    assert summary["cost_join"] == reqlog.cost_join_status(report)


def test_cost_summary_reports_none_not_zero_when_nothing_matched():
    rows = [{"task_id": "A"}]
    report = reqlog.join_by_conversation(rows, _log(), lambda r: ["nope"])
    summary = reqlog.cost_summary(rows, report)
    assert summary["n_cost_joined"] == 0
    assert summary["n_cost_requests"] == 0  # the join RAN and matched nothing
    # no task carries a doc count: a 0 here would read as "measured zero"
    assert summary["n_docs_max"] is None
    assert summary["dropped_docs_total"] is None


def test_not_joinable_report_carries_its_reason():
    rows = [{"task_id": "A"}, {"task_id": "B"}]
    report = reqlog.not_joinable(rows, "no request log for this run")
    assert report["n_rows"] == 2 and report["n_joined"] == 0
    assert report["full_coverage"] is False
    assert (reqlog.cost_join_status(report)
            == "not joinable: no request log for this run")
    summary = reqlog.cost_summary(rows, report)
    # a join that never ran measured nothing at all
    assert summary["n_cost_requests"] is None
    assert summary["n_docs_max"] is None
    assert summary["dropped_docs_total"] is None
