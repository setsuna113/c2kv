# -*- coding: utf-8 -*-
import json

import pytest

import t34_score as S


def _w(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_merge_joins_on_qid_and_refuses_duplicate_columns(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _w(a, [{"qid": "s:1", "arm": "c2kv", "x": 1.0}, {"qid": "s:2", "arm": "c2kv", "x": None}])
    _w(b, [{"qid": "s:1", "y": 2.0}])
    rows = S.merge_feature_files([str(a), str(b)], "c2kv")
    assert rows[0] == {"qid": "s:1", "arm": "c2kv", "x": 1.0, "y": 2.0}
    assert rows[1]["x"] is None and "y" not in rows[1]
    c = tmp_path / "c.jsonl"
    _w(c, [{"qid": "s:1", "x": 3.0}])
    with pytest.raises(ValueError):
        S.merge_feature_files([str(a), str(c)], "c2kv")


def test_merge_refuses_leaky_columns(tmp_path):
    a = tmp_path / "a.jsonl"
    _w(a, [{"qid": "s:1", "target_tool_name": "f"}])
    with pytest.raises(ValueError):
        S.merge_feature_files([str(a)], "c2kv")


def test_orientation_conflict_detected(tmp_path):
    (tmp_path / "orientations_a.json").write_text('{"f": 1}', encoding="utf-8")
    (tmp_path / "orientations_b.json").write_text('{"f": -1}', encoding="utf-8")
    with pytest.raises(ValueError):
        S.merged_orientations(tmp_path)
    (tmp_path / "orientations_b.json").write_text('{"g": -1}', encoding="utf-8")
    assert S.merged_orientations(tmp_path) == {"f": 1, "g": -1}


def _entry(**kw):
    e = {"auprc": 0.85, "auprc_ci": [0.76, 0.92], "delta_vs_s0": {"ci_lo": 0.12},
         "parse_baseline": {"coverage": 35, "false_resets": 28, "precision": 0.5556},
         "matched_rate_op": {"coverage": 53, "false_resets": 10, "precision": 0.84},
         "auprc_length_controlled": 0.85, "auprc_uncensored": 0.83, "auroc_uncensored": 0.8}
    e.update(kw)
    return e


def test_verdict_uses_frame_prevalence_not_base_rate():
    prev = 93 / 161
    assert S.verdict_prevalence_aware(_entry(), prevalence_all=prev)["verdict"] == "LIVE"
    # CI lower bound above 0.1033 but below the frame prevalence: not live
    v = S.verdict_prevalence_aware(_entry(auprc_ci=[0.50, 0.7]), prevalence_all=prev)
    assert v["clauses"]["above_chance"] is False and v["verdict"] == "not-live"


def test_verdict_length_clause_is_binding():
    prev = 93 / 161
    v = S.verdict_prevalence_aware(_entry(auprc_length_controlled=0.43), prevalence_all=prev)
    assert v["clauses"]["length"] is False
    v = S.verdict_prevalence_aware(_entry(auprc=0.85, auprc_length_controlled=0.70), prevalence_all=prev)
    assert v["clauses"]["length"] is False  # dropped by more than the tolerance


def test_verdict_s0_undefined_is_not_live():
    v = S.verdict_prevalence_aware(_entry(delta_vs_s0=None), prevalence_all=93 / 161)
    assert v["verdict"] == "s0-undefined"


def test_orientation_metadata_entries_are_ignored(tmp_path):
    (tmp_path / "orientations_a.json").write_text(
        '{"_description": "text", "f": 1, "notes": {"x": 1}, "g": "-1"}', encoding="utf-8")
    assert S.merged_orientations(tmp_path) == {"f": 1, "g": -1}
    (tmp_path / "orientations_b.json").write_text('{"h": 5}', encoding="utf-8")
    with pytest.raises(ValueError):
        S.merged_orientations(tmp_path)


def test_shared_orientations_win_over_unit_redeclarations(tmp_path):
    (tmp_path / "orientations_shared.json").write_text('{"_d": "x", "gist_tokens": 1}', encoding="utf-8")
    (tmp_path / "orientations_u.json").write_text('{"gist_tokens": -1, "own": -1}', encoding="utf-8")
    o = S.merged_orientations(tmp_path)
    assert o == {"gist_tokens": 1, "own": -1}
    rep = S.orientation_report()
    assert rep["redeclared_shared_keys"][0]["key"] == "gist_tokens"
