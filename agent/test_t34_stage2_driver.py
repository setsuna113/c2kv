"""Pure-part tests for agent/t34_stage2_driver.py (the torch loop is server-only)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import t34_stage2_driver as S  # noqa: E402


def test_k_median_is_the_frozen_position_heuristic():
    from d_sham_plan import k_star_for
    for n in range(1, 20):
        assert S.k_median(n) == k_star_for(n) == (n - 1) // 2
    with pytest.raises(ValueError):
        S.k_median(0)


def _rows():
    return [{"qid": "a:1", "label_cw": 1}, {"qid": "b:2", "label_cw": 0},
            {"qid": "c:3", "label_cw": 1}, {"qid": "d:4", "label_cw": 0}]


def test_plan_qids_keeps_frame_order_and_filters_by_class():
    assert S.plan_qids(_rows()) == ["a:1", "b:2", "c:3", "d:4"]
    assert S.plan_qids(_rows(), "cw") == ["a:1", "c:3"]
    assert S.plan_qids(_rows(), "cc") == ["b:2", "d:4"]
    assert S.plan_qids(_rows(), "both", max_qids=3) == ["a:1", "b:2", "c:3"]
    assert S.plan_qids(_rows(), "both", qids=["d:4", "a:1"]) == ["a:1", "d:4"]


def test_plan_qids_refuses_rows_outside_the_frame():
    with pytest.raises(ValueError):
        S.plan_qids(_rows(), qids=["zz:9"])
    with pytest.raises(ValueError):
        S.plan_qids(_rows(), "weird")


def test_resume_done_ignores_skipped_and_garbage(tmp_path):
    p = tmp_path / "out.jsonl"
    lines = [json.dumps({"qid": "a:1", "d_corr_doc_index": 3}),
             json.dumps({"qid": "b:2", "d_corr_doc_index": 0, "skipped": True}),
             "not json", ""]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert S.resume_done(p) == {("a:1", 3)}
    assert S.resume_done(tmp_path / "missing.jsonl") == set()


def test_stamp_row_marks_arm_mode_and_block():
    row = S.stamp_row({"qid": "a:1", "prediction": "x"}, 5)
    assert row["d_arm"] == "raw_keepG_kmedian"
    assert row["d_mode"] == "d_raw_keepG"
    assert row["d_ksweep_k"] == 5 and row["d_corr_doc_index"] == 5
    assert row["k_policy"] == "median"


def test_stage2_rows_satisfy_the_cascade_reader_and_audit(tmp_path):
    """The cascade reads {qid, prediction, *_sec} and audits d_corr_doc_index
    against k_median: a stamped row at k_median passes as gold-free."""
    import t34_cascade as CC
    rows = [S.stamp_row({"qid": f"s{i}:1", "prediction": "<tool_call>{}</tool_call>",
                         "d_corr_slice_prefill_sec": 0.03, "d_recompute_prefill_sec": 0.0,
                         "generate_sec": 1.2, "tool_name_match": True}, S.k_median(n))
            for i, n in enumerate((16, 8, 1))]
    p = tmp_path / "stage2.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    table = CC.load_expensive_table(p)
    assert set(table) == {"s0:1", "s1:1", "s2:1"}
    assert "tool_name_match" not in table["s0:1"]          # scoring columns never travel
    assert table["s0:1"]["d_corr_slice_prefill_sec"] == 0.03
    audit = CC.expensive_block_choice_audit(p, frame=None)
    assert audit["n_with_block_index"] == 3


def test_parse_args_defaults_match_the_frozen_recipe():
    a = S.parse_args(["--output_file", "x.jsonl"])
    assert (a.ratio, a.max_doc_length, a.max_doc_num, a.max_new_tokens) == (8, 768, 16, 128)
    assert a.attn_impl == "eager" and a.classes == "both" and a.resume is True
    a2 = S.parse_args(["--output_file", "x.jsonl", "--resume", "false", "--classes", "cc"])
    assert a2.resume is False and a2.classes == "cc"
