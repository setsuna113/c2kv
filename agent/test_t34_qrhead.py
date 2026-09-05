# -*- coding: utf-8 -*-
"""Unit U1b tests: QRHead (2506.09944), Retrieval Head port 0 (2404.15574),
CacheBlend layer-1 deviation (2405.16444).

Everything here runs torch-free.  The two model-dependent code paths get
(a) a torch-gated test and (b) a fake-capture test that exercises the wiring
and the reduction with synthetic tensors.
"""
from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest

import t34_common as C
import t34_cacheblend_l1 as CB
import t34_qrhead as QR
import t34_retrieval_head as RH

ROOT = Path(__file__).resolve().parent.parent
ORIENT_PATH = ROOT / "configs/t34/orientations_qrhead.json"


# ==========================================================================
# (A) QRHead -- Eq. score_perdoc / score_agg, head selection, calibration
# ==========================================================================

def _dense_attention(n_layers=2, n_heads=3, n_q=4, n_keys=11, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random((n_layers, n_heads, n_q, n_keys))
    return a / a.sum(axis=-1, keepdims=True)          # post-softmax rows


def _doc_mass_from(attn, spans):
    out = np.zeros(attn.shape[:3] + (len(spans),))
    for i, (s, e) in enumerate(spans):
        out[..., i] = attn[..., s:e].sum(axis=-1)
    return out


def test_qr_score_perdoc_matches_the_dense_equation():
    """Eq. score_perdoc of 2506.09944 Sec. 3.1, computed densely vs reduced."""
    attn = _dense_attention()
    spans = [(1, 4), (4, 6), (6, 10)]
    rows = list(range(attn.shape[2]))
    dense = QR.qr_score_perdoc_dense(attn, rows, spans)
    reduced = QR.qr_score_perdoc(_doc_mass_from(attn, spans))
    assert dense.shape == (2, 3, 3)
    assert np.allclose(dense, reduced)


def test_qr_score_perdoc_is_the_1_over_q_average_not_a_sum():
    attn = _dense_attention(n_q=5)
    spans = [(1, 4), (4, 6), (6, 10)]
    dm = _doc_mass_from(attn, spans)
    got = QR.qr_score_perdoc(dm)
    assert np.allclose(got, dm.sum(axis=2) / 5)


def test_qr_score_perdoc_rejects_empty_query():
    with pytest.raises(ValueError):
        QR.qr_score_perdoc(np.zeros((2, 2, 0, 3)))
    with pytest.raises(ValueError):
        QR.qr_score_perdoc_dense(_dense_attention(), [], [(0, 2)])


def test_qr_score_agg_sums_over_the_gold_set():
    perdoc = np.arange(2 * 2 * 4, dtype=float).reshape(2, 2, 4)
    agg = QR.qr_score_agg(perdoc, [1, 3])
    assert np.allclose(agg, perdoc[:, :, 1] + perdoc[:, :, 3])
    with pytest.raises(ValueError):
        QR.qr_score_agg(perdoc, [])
    with pytest.raises(ValueError):
        QR.qr_score_agg(perdoc, [9])


def test_head_count_is_1_to_2_percent_of_the_config_grid(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"num_hidden_layers": 36, "num_attention_heads": 32}),
                   encoding="utf-8")
    n_l, n_h = QR.read_head_grid(cfg)
    assert (n_l, n_h) == (36, 32)
    assert QR.heads_from_fraction(n_l, n_h, 0.01) == round(0.01 * 36 * 32)
    assert QR.heads_from_fraction(n_l, n_h, 0.02) == round(0.02 * 36 * 32)
    # both declared bounds, and never zero
    assert QR.heads_from_fraction(2, 2, 0.001) == 1


def test_select_heads_is_deterministic_and_ranked():
    table = np.array([[0.5, 0.9], [0.9, 0.1]])
    assert QR.select_heads(table, 2) == [(0, 1), (1, 0)]   # tie -> (layer, head)
    assert QR.select_heads(table, 1) == [(0, 1)]
    assert QR.head_overlap([(0, 1), (1, 0)], [(1, 0), (1, 1)]) == 1


def test_retriever_scores_average_over_selected_heads_only():
    perdoc = np.zeros((2, 2, 3))
    perdoc[0, 0] = [1.0, 0.0, 0.0]
    perdoc[1, 1] = [0.0, 3.0, 0.0]
    perdoc[0, 1] = [99.0, 99.0, 99.0]          # unselected head must not leak
    r = QR.retriever_scores(perdoc, [(0, 0), (1, 1)])
    assert np.allclose(r, [0.5, 1.5, 0.0])
    with pytest.raises(ValueError):
        QR.retriever_scores(perdoc, [])


def test_calibration_subtracts_the_null_query_score():
    r = np.array([1.0, 2.0, 3.0])
    r0 = np.array([0.5, 2.5, 0.0])
    assert np.allclose(QR.calibrate(r, r0), [0.5, -0.5, 3.0])
    assert QR.NULL_QUERY_TEXT == "N/A"


def test_khat_and_margin_edge_cases():
    assert QR.khat([0.1, 0.9, 0.2]) == 1
    assert QR.khat([1.0, 1.0]) == 0                     # lowest index on ties
    assert QR.khat([]) is None
    assert QR.khat([np.nan, np.nan]) is None
    assert QR.top1_top2_margin([3.0, 1.0, 2.0]) == pytest.approx(1.0)
    assert QR.top1_top2_margin([1.0]) is None           # None, never a sentinel


def test_chooser_domain_makes_the_frozen_chooser_reproduce_khat():
    """The specificity control runs t34_common.chooser_argmax/argmin, which
    abstain when the maximum is <= 0.  chooser_domain has to make that chooser
    agree with the argmax this module actually reports -- including on the
    CALIBRATED arm, whose score R - R(q_null) is signed."""
    for vec in ([0.1, 0.9, 0.2], [-0.4, -0.9, -0.2], [-1.0, -1.0], [0.0, 0.0, 0.0],
                [2.0, np.nan, 5.0], [-3.0]):
        t = QR.chooser_domain(vec)
        assert C.chooser_argmax(t) == QR.khat(vec), vec
    # the inverted arm still picks the minimum, and still abstains on a flat
    # vector (t34_common's own documented rule)
    assert C.chooser_argmin(QR.chooser_domain([-0.4, -0.9, -0.2])) == 1
    assert C.chooser_argmin(QR.chooser_domain([-1.0, -1.0])) is None
    assert QR.chooser_domain([]).size == 0
    assert not np.isfinite(QR.chooser_domain([np.nan, np.nan])).any()


def test_calibrated_locator_control_is_the_same_chooser_as_the_table():
    """Regression: with a raw negative calibrated vector the frozen chooser
    would abstain everywhere and the control's forward arm would not be the
    locator being reported."""
    frame = _witness_frame()
    scored = {
        "sA:1": {"r_unc": np.array([0.1, 0.2, 0.9]),
                 "r_cal": np.array([-0.9, -0.8, -0.1]),   # all negative
                 "profile": {c: 0.2 for c in QR.CLASSES}, "meta": {}},
        "sA:2": {"r_unc": np.array([0.9, 0.2, 0.1]),
                 "r_cal": np.array([-0.1, -0.8, -0.9]),
                 "profile": {c: 0.2 for c in QR.CLASSES}, "meta": {}},
    }
    out = QR.locator_tables(scored, frame)
    assert out["calibrated"]["hits"] == 2
    fwd = out["inverted_control_calibrated"]["forward"]
    assert fwd["hits"] == out["calibrated"]["hits"]
    assert fwd["n"] == out["calibrated"]["n"] == 2
    assert out["inverted_control_calibrated"]["inverted"]["hits"] == 0


def test_locator_control_denominator_matches_the_headline_table():
    """Unprobed rows are abstentions in BOTH tables, never dropped from one."""
    frame = _witness_frame()
    scored = {"sA:1": {"r_unc": np.array([0.1, 0.2, 0.9]), "r_cal": None,
                       "profile": {c: 0.2 for c in QR.CLASSES}, "meta": {}}}
    out = QR.locator_tables(scored, frame)
    assert out["uncalibrated"]["n"] == 2
    assert out["inverted_control_uncalibrated"]["forward"]["n"] == 2
    assert out["inverted_control_uncalibrated"]["forward"]["abstained"] == 1


def test_length_controls_are_reported_in_both_orientations():
    """These covariates have no pre-registered direction; scoring them in one
    arbitrary sign would hide a free baseline that separates the other way."""
    frame = _witness_frame()
    rep = QR.length_control_report(frame, ["sA:1", "sA:2"], np.array([1, 0]))
    for nm in ("n_docs", "kept_history_tokens", "generated_tokens"):
        assert nm in rep
    assert "no winner" in rep["note"]


def test_trigger_scalars_never_use_sentinels():
    s = QR.trigger_scalars([1.0, 0.25])
    assert s["qrhead_r_max"] == pytest.approx(1.0)
    assert s["qrhead_margin_top1_top2"] == pytest.approx(0.75)
    assert s["qrhead_calib_delta"] is None              # no null capture
    s2 = QR.trigger_scalars([1.0, 0.25], [0.4, 0.1])
    assert s2["qrhead_calib_delta"] == pytest.approx(0.4 - 1.0)


def test_class_mass_profile_is_a_normalised_split_over_the_five_classes():
    cm = np.zeros((2, 2, 3, 5))
    cm[0, 0, :, 0] = 1.0        # system_raw (sink)
    cm[0, 0, :, 1] = 3.0        # history_gist
    prof = QR.class_mass_profile(cm, [(0, 0)])
    assert prof["system_raw"] == pytest.approx(0.25)
    assert prof["history_gist"] == pytest.approx(0.75)
    assert sum(prof.values()) == pytest.approx(1.0)
    assert QR.CLASSES == ("system_raw", "history_gist", "history_raw_tail",
                          "current_query", "generated_so_far")


def test_mass_split_diagnostic_separates_cw_from_cc():
    profiles = {
        "s1:1": {c: (0.8 if c == "system_raw" else 0.05) for c in QR.CLASSES},
        "s2:1": {c: (0.1 if c == "system_raw" else 0.225) for c in QR.CLASSES},
    }
    labels = {"s1:1": 1, "s2:1": 0}
    d = QR.mass_split_diagnostic(profiles, labels)
    assert d["cw"]["n"] == 1 and d["cc"]["n"] == 1
    assert d["cw"]["system_raw"]["mean"] == pytest.approx(0.8)
    assert d["cc"]["system_raw"]["mean"] == pytest.approx(0.1)


# --------------------------------------------------------------------------
# detection-set construction and its leakage guard
# --------------------------------------------------------------------------

def _fake_frame(cw=("sA:1",), cc=("sB:1",), extra=("sC:1", "sC:2", "sD:1")):
    labels = ([{"qid": q, "label_cw": 1} for q in cw]
              + [{"qid": q, "label_cw": 0} for q in cc]
              + [{"qid": q, "label_cw": None} for q in extra])
    pairs = []
    for rec in labels:
        pairs.append((
            {"qid": rec["qid"], "target_tool_name": "alpha_tool",
             "target": 'Action:\n<tool_call>\n{"name":"alpha_tool",'
                       '"arguments":{"id":"ZZTOP-4471"}}\n</tool_call>'},
            {"qid": rec["qid"], "prediction": "", "target_has_tool_call": True},
        ))
    return C.FrozenFrame(pairs=pairs, labels=labels, manifest={}, witness=None)


def _fake_sidecar(frame, docs_for):
    out = {}
    for rec in frame.labels:
        q = rec["qid"]
        # sA (C->W) and sB (C->C) are the EVALUATION frame; they share one
        # toolset so that the disjointness constraint has something to bite on.
        is_eval = q.startswith("sA") or q.startswith("sB")
        out[q] = {"qid": q, "docs": docs_for(q),
                  "tools": [{"type": "function", "function": {"name": "alpha_tool"}}]
                  if is_eval else
                  [{"type": "function", "function": {"name": "beta_tool"}}]}
    return out


def test_detection_set_excludes_cw_qids_and_their_sessions():
    frame = _fake_frame()
    side = _fake_sidecar(frame, lambda q: ["nothing here", "ZZTOP-4471 lives here"])
    det = QR.build_detection_set(frame, side, size=10, require_toolset_disjoint=False)
    qids = {r["qid"] for r in det["rows"]}
    assert "sA:1" not in qids
    assert not any(q.startswith("sA") for q in qids)
    assert det["dropped"]["in_cw"] == 1
    QR.assert_detection_set_disjoint(det, frame)


def test_detection_set_also_excludes_the_cc_evaluation_rows_and_sessions():
    """The head table must not be selected on rows the trigger arm scores.

    Digest 4.5 words the constraint as "the 72 C->W sessions"; the evaluation
    frame is 161 rows, so the 68 C->C rows and their sessions are excluded too
    (DEVIATIONS: 'detection-set disjointness')."""
    frame = _fake_frame()
    side = _fake_sidecar(frame, lambda q: ["nothing here", "ZZTOP-4471 lives here"])
    det = QR.build_detection_set(frame, side, size=10, require_toolset_disjoint=False)
    qids = {r["qid"] for r in det["rows"]}
    assert "sB:1" not in qids                       # the C->C evaluation row
    assert not any(q.startswith("sB") for q in qids)  # ... and its session
    assert det["dropped"]["in_eval_frame_cc"] == 1
    assert det["n_eval_qids_excluded"] == 2
    # a roster carrying a C->C evaluation row is rejected mechanically
    with pytest.raises(AssertionError):
        QR.assert_detection_set_disjoint(
            {"rows": [{"qid": "sB:1", "session_id": "sB"}]}, frame)


def test_detection_set_toolset_disjointness_is_enforced_not_relaxed():
    frame = _fake_frame()
    side = _fake_sidecar(frame, lambda q: ["x", "ZZTOP-4471"])
    # every non-eval row here carries beta_tool, the eval session carries
    # alpha_tool -> disjoint, so nothing is dropped for overlap
    det = QR.build_detection_set(frame, side, size=10, require_toolset_disjoint=True)
    assert det["dropped"]["toolset_overlap"] == 0
    # now make one row share the eval toolset
    side["sC:1"]["tools"] = [{"type": "function", "function": {"name": "alpha_tool"}}]
    det2 = QR.build_detection_set(frame, side, size=10, require_toolset_disjoint=True)
    assert det2["dropped"]["toolset_overlap"] == 1
    assert "sC:1" not in {r["qid"] for r in det2["rows"]}


def test_detection_set_drops_rows_without_a_gold_block():
    frame = _fake_frame()
    side = _fake_sidecar(frame, lambda q: ["nothing", "still nothing"])
    det = QR.build_detection_set(frame, side, size=10, require_toolset_disjoint=False)
    assert det["n"] == 0
    assert det["dropped"]["no_gold_block"] >= 1


def test_assert_detection_set_disjoint_is_mechanical():
    frame = _fake_frame()
    bad = {"rows": [{"qid": "sA:1", "session_id": "sA"}]}
    with pytest.raises(AssertionError):
        QR.assert_detection_set_disjoint(bad, frame)


def test_witness_gold_block_label_uses_the_frozen_witness_construction():
    docs = ["chatter", "the id is ZZTOP-4471 and alpha_tool", "alpha_tool"]
    k = QR.witness_gold_block_label(docs, "alpha_tool", {"id": "ZZTOP-4471"})
    assert k == 1
    assert QR.witness_gold_block_label(docs, None, None) is None


# --------------------------------------------------------------------------
# capture store round-trip and head detection over it
# --------------------------------------------------------------------------

def test_capture_roundtrip_and_head_detection(tmp_path):
    rng = np.random.default_rng(3)
    det = {"rows": []}
    cap_dir = tmp_path / "attn"
    # head (1,0) is made the query-focused head: it puts all its mass on the
    # gold block of every detection row.
    for i in range(6):
        qid = f"s{i}:1"
        gold = i % 3
        dm = rng.random((2, 2, 4, 3)) * 0.01
        dm[1, 0, :, gold] += 1.0
        cm = rng.random((2, 2, 4, 5)) * 0.01
        QR.write_capture(cap_dir, qid, doc_mass=dm, class_mass=cm,
                         meta={"query_proj": "gist"})
        det["rows"].append({"qid": qid, "gold_blocks": [gold]})
    got = QR.read_capture(cap_dir, "s0:1")
    assert got["doc_mass"].shape == (2, 2, 4, 3)
    assert got["meta"]["query_proj"] == "gist"
    assert QR.read_capture(cap_dir, "missing:9") is None

    table = QR.detect_heads(cap_dir, det, m=1)
    assert table["heads"] == [[1, 0]]
    assert table["n_detection_rows"] == 6
    assert table["query_proj"] == "gist"
    assert 0 <= table["stability_top_m_overlap_halves"] <= 1


def test_detect_heads_refuses_to_mix_query_proj_modes(tmp_path):
    cap_dir = tmp_path / "attn"
    for i, proj in enumerate(("gist", "base")):
        QR.write_capture(cap_dir, f"s{i}:1",
                         doc_mass=np.ones((1, 1, 2, 2)),
                         class_mass=np.ones((1, 1, 2, 5)),
                         meta={"query_proj": proj})
    det = {"rows": [{"qid": "s0:1", "gold_blocks": [0]},
                    {"qid": "s1:1", "gold_blocks": [0]}]}
    with pytest.raises(RuntimeError, match="query_proj"):
        QR.detect_heads(cap_dir, det, m=1)


# --------------------------------------------------------------------------
# capture_rows wiring against a FAKE unit-U1a module
# --------------------------------------------------------------------------

class _FakeCapture:
    def __init__(self, key_map, query_mode, last_n, layers):
        self.key_map, self.query_mode, self.last_n = key_map, query_mode, last_n
        self.installed = self.removed = 0
        self.n_q = last_n

    def install(self, model):
        self.installed += 1

    def remove(self, model):
        self.removed += 1

    def doc_mass_tensor(self):
        return np.ones((2, 2, self.n_q, 3))

    def class_mass_tensor(self):
        return np.ones((2, 2, self.n_q, 5))


def _fake_u1a():
    mod = types.SimpleNamespace()
    mod.CLASSES = QR.CLASSES
    mod.KeyClassMap = lambda **kw: types.SimpleNamespace(**kw)
    mod.AttentionRowCapture = _FakeCapture
    return mod


def test_capture_rows_wiring_and_query_span_assertion(tmp_path):
    calls = []

    def build_prefix(qid):
        return {
            "model": object(), "query_ids": [1, 2, 3, 4],
            "null_query_ids": [7, 8], "system_len": 5,
            "doc_spans": [(5, 8), (8, 11), (11, 14)],
            "raw_tail_span": None, "query_span": (14, 18),
            "generated_start": 18,
            "forward": lambda ids: calls.append(list(ids)),
        }

    stats = QR.capture_rows([{"qid": "sX:1"}], {}, build_prefix=build_prefix,
                            out_dir=tmp_path, arm="c2kv", query_proj="gist",
                            calibrate_null=True, attention_module=_fake_u1a())
    assert stats["n"] == 1
    assert calls == [[1, 2, 3, 4], [7, 8]]
    cap = QR.read_capture(tmp_path, "sX:1")
    assert cap["doc_mass"].shape == (2, 2, 4, 3)
    assert cap["doc_mass_null"].shape == (2, 2, 2, 3)
    assert cap["meta"]["query_proj"] == "gist"
    assert cap["meta"]["null_query_text"] == "N/A"
    assert stats["qnull_prefill_sec"] >= 0.0


def test_capture_rows_asserts_the_query_span_fits_in_last_n(tmp_path):
    def build_prefix(qid):
        return {"model": object(), "query_ids": [1, 2], "null_query_ids": [7],
                "system_len": 1, "doc_spans": [(1, 3)], "raw_tail_span": None,
                "query_span": (3, 9), "generated_start": 9,   # 6 > last_n = 2
                "forward": lambda ids: None}

    with pytest.raises(AssertionError, match="query span"):
        QR.capture_rows([{"qid": "sX:1"}], {}, build_prefix=build_prefix,
                        out_dir=tmp_path, arm="c2kv", query_proj="gist",
                        attention_module=_fake_u1a())


def test_capture_rows_skips_rows_with_no_prefix(tmp_path):
    stats = QR.capture_rows([{"qid": "sX:1"}], {}, build_prefix=lambda q: None,
                            out_dir=tmp_path, arm="c2kv", query_proj="base",
                            attention_module=_fake_u1a())
    assert stats["n"] == 0 and stats["skipped"] == ["sX:1"]


def test_u1a_is_not_imported_at_module_level():
    """The interface of unit U1a is torch-dependent and is written
    concurrently: importing it eagerly would break this box."""
    import sys
    src = (Path(QR.__file__)).read_text(encoding="utf-8")
    head = src.split("def _load_u1a", 1)[0]
    assert "import t34_attention" not in head
    assert "t34_attention" not in sys.modules or True  # never required here


# --------------------------------------------------------------------------
# locator + feature scoring over the frozen frame
# --------------------------------------------------------------------------

def _witness_frame():
    labels = [{"qid": "sA:1", "label_cw": 1}, {"qid": "sA:2", "label_cw": 1},
              {"qid": "sB:1", "label_cw": 0}, {"qid": "sB:2", "label_cw": 0}]
    pairs = [({"qid": r["qid"]},
              {"qid": r["qid"], "prediction": "Action:\n<tool_call>\n{}\n</tool_call>",
               "target_has_tool_call": True}) for r in labels]
    witness = {"entries": {"sA:1": {"k_witness": 2}, "sA:2": {"k_witness": 0}}}
    return C.FrozenFrame(pairs=pairs, labels=labels, manifest={}, witness=witness)


def test_locator_tables_score_against_the_frozen_floor_with_inversion_control():
    frame = _witness_frame()
    scored = {
        "sA:1": {"r_unc": np.array([0.1, 0.2, 0.9]), "r_cal": None,
                 "profile": {c: 0.2 for c in QR.CLASSES}, "meta": {}},
        "sA:2": {"r_unc": np.array([0.9, 0.2, 0.1]), "r_cal": None,
                 "profile": {c: 0.2 for c in QR.CLASSES}, "meta": {}},
    }
    out = QR.locator_tables(scored, frame)
    t = out["uncalibrated"]
    assert t["hits"] == 2 and t["n"] == 2
    assert t["floor"] == C.LOCATE_FLOOR_WRONG_BLOCK == 0.25
    assert t["witness_oracle"] == pytest.approx(71 / 93)
    inv = out["inverted_control_uncalibrated"]
    assert inv["forward"]["hits"] == 2 and inv["inverted"]["hits"] == 0
    # k_first legacy column: only sA:2 has k*=0
    assert out["legacy_k_first"]["hits"] == 1


def test_locator_abstains_when_the_witness_k_star_is_none():
    frame = _witness_frame()
    frame.witness["entries"]["sA:1"] = {"k_witness": None}
    scored = {"sA:1": {"r_unc": np.array([1.0, 0.0]), "r_cal": None,
                       "profile": {c: 0.2 for c in QR.CLASSES}, "meta": {}}}
    out = QR.locator_tables(scored, frame)
    assert out["uncalibrated"]["abstained"] == 2       # k*=None + missing sA:2
    assert out["uncalibrated"]["n"] == 2               # abstentions stay in n


def test_feature_rows_cover_the_whole_trigger_frame_with_nulls_not_sentinels():
    frame = _witness_frame()
    scored = {"sA:1": {"r_unc": np.array([0.2, 0.8]), "r_cal": np.array([0.1, 0.5]),
                       "profile": {c: 0.2 for c in QR.CLASSES},
                       "meta": {"query_proj": "gist"}}}
    rows = QR.feature_rows(scored, frame, arm="c2kv")
    assert len(rows) == 4                              # 2 C->W + 2 C->C
    by = {r["qid"]: r for r in rows}
    assert by["sA:1"]["qrhead_r_max"] == pytest.approx(0.8)
    assert by["sA:1"]["query_proj"] == "gist"
    assert by["sB:1"]["qrhead_r_max"] is None          # null, not 0.0
    assert set(QR.TRIGGER_FEATURES) <= set(by["sB:1"])


def test_features_pass_the_leakage_guard_and_leak_specimens_do_not(tmp_path):
    frame = _witness_frame()
    rows = QR.feature_rows({}, frame, arm="c2kv")
    n = C.write_features_jsonl(tmp_path / "f.jsonl", rows, context="test")
    assert n == 4
    bad = [dict(r, a_made_call=1) for r in rows]
    with pytest.raises(ValueError, match="a_made_call"):
        C.write_features_jsonl(tmp_path / "bad.jsonl", bad, context="test")
    worse = [dict(r, tool_name_match=1) for r in rows]
    with pytest.raises(ValueError):
        C.write_features_jsonl(tmp_path / "worse.jsonl", worse, context="test")


def test_evaluate_features_uses_the_frame_prevalence_as_chance_ap():
    frame = _witness_frame()
    scored = {}
    for i, q in enumerate(["sA:1", "sA:2", "sB:1", "sB:2"]):
        r = np.array([0.9 - 0.1 * i, 0.1])
        scored[q] = {"r_unc": r, "r_cal": None,
                     "profile": {c: 0.2 for c in QR.CLASSES}, "meta": {}}
    rows = QR.feature_rows(scored, frame, arm="c2kv")
    rep = QR.evaluate_features(rows, frame, C.load_orientations(ORIENT_PATH), reps=50)
    ent = rep["features"]["qrhead_r_max"]
    assert ent["n"] == 4 and ent["n_pos"] == 2
    assert ent["prevalence_chance_ap"] == pytest.approx(0.5)
    assert ent["orientation"] == -1
    assert ent["s0_full_arm"]["note"].startswith("S0 twin undefined")
    assert rep["baseline_parse_fail"]["n"] == 4


def test_evaluate_features_s0_twin_is_a_paired_delta_on_the_same_rows():
    frame = _witness_frame()
    mk = lambda v: {"r_unc": np.array([v, 0.0]), "r_cal": None,
                    "profile": {c: 0.2 for c in QR.CLASSES}, "meta": {}}
    scored = {"sA:1": mk(0.9), "sA:2": mk(0.8), "sB:1": mk(0.2), "sB:2": mk(0.1)}
    s0 = {"sA:1": mk(0.1), "sA:2": mk(0.2), "sB:1": mk(0.8), "sB:2": mk(0.9)}
    rows = QR.feature_rows(scored, frame, arm="c2kv")
    s0_rows = QR.feature_rows(s0, frame, arm="full")
    rep = QR.evaluate_features(rows, frame, C.load_orientations(ORIENT_PATH),
                               s0_rows=s0_rows, reps=50)
    d = rep["features"]["qrhead_r_max"]["s0_full_arm"]
    assert "delta_ap_ci" in d and d["ap"] is not None


# ==========================================================================
# (B) Retrieval Head -- port 0
# ==========================================================================

def test_copy_paste_rule_needs_identity_and_position_and_membership():
    x = [10, 11, 12, 13, 14]          # context; needle occupies [1, 3)
    needle = [11, 12]
    span = (1, 3)
    # generated 11 then 12, both attended at their own needle positions
    got = RH.copy_paste_tokens([1, 2], [11, 12], x, needle, span)
    assert got == [11, 12]
    # criterion (2), position: argmax outside the needle span
    assert RH.copy_paste_tokens([0, 2], [11, 12], x, needle, span) == [12]
    # criterion (2), identity: argmax inside the span but a different token
    assert RH.copy_paste_tokens([2, 2], [11, 12], x, needle, span) == [12]
    # criterion (1), membership: generated token not in the needle
    assert RH.copy_paste_tokens([1, 2], [99, 12], x, needle, span) == [12]


def test_retrieval_score_is_recall_of_the_needle_tokens():
    assert RH.retrieval_score([11, 12], [11, 12]) == pytest.approx(1.0)
    assert RH.retrieval_score([11], [11, 12]) == pytest.approx(0.5)
    assert RH.retrieval_score([], [11, 12]) == pytest.approx(0.0)
    # repeated needle tokens: set semantics is the primary reading
    assert RH.retrieval_score([11], [11, 11, 12]) == pytest.approx(0.5)
    assert RH.retrieval_score([11, 11], [11, 11, 12],
                              count_mode="positions") == pytest.approx(2 / 3)
    assert np.isnan(RH.retrieval_score([1], []))
    with pytest.raises(ValueError):
        RH.retrieval_score([1], [1], count_mode="nope")


def test_trial_head_scores_on_a_synthetic_argmax_array():
    x = [5, 6, 7, 8]
    needle, span = [6, 7], (1, 3)
    gen = [6, 7]
    argmax = np.zeros((2, 2, 2), dtype=int)
    argmax[0, 0] = [1, 2]      # perfect copy head
    argmax[0, 1] = [1, 0]      # half
    argmax[1, 0] = [0, 0]      # nothing
    argmax[1, 1] = [3, 3]      # attends outside the needle
    got = RH.trial_head_scores(argmax, gen, x, needle, span)
    assert got.shape == (2, 2)
    assert np.allclose(got, [[1.0, 0.5], [0.0, 0.0]])


def test_threshold_and_head_list_use_the_papers_unvalidated_constant():
    assert RH.RETRIEVAL_HEAD_THRESHOLD == 0.1
    table = np.array([[0.0, 0.05], [0.3, 0.9]])
    assert RH.retrieval_head_mask(table).tolist() == [[False, False], [True, True]]
    assert RH.head_list(table) == [(1, 1), (1, 0)]
    assert RH.head_list(table, top_m=1) == [(1, 1)]
    rep = RH.sparsity_report(table)
    assert rep["n_above_threshold"] == 2
    assert rep["frac_exactly_zero"] == pytest.approx(0.25)


def test_pearson_criterion_matches_the_papers_own_read():
    a = np.array([[0.0, 0.5], [0.9, 0.1]])
    b = a * 2.0 + 0.01
    r = RH.pearson_head_maps(a, b)
    assert r == pytest.approx(1.0)
    assert r > RH.HEAD_SET_PRESERVED_R == 0.8
    assert RH.pearson_head_maps(np.zeros((2, 2)), a) is None   # None, not 0.0
    with pytest.raises(ValueError):
        RH.pearson_head_maps(a, np.zeros((3, 3)))


def test_pairwise_pearson_reports_every_pair_with_the_verdict():
    maps = {"base": np.array([[0.0, 0.5], [0.9, 0.1]]),
            "fixed_joint": np.array([[0.0, 0.5], [0.9, 0.1]]),
            "ckpt1088": np.array([[0.9, 0.1], [0.0, 0.5]])}
    out = RH.pairwise_pearson(maps)
    assert len(out["pairs"]) == 3
    got = {(p["a"], p["b"]): p for p in out["pairs"]}
    assert got[("base", "fixed_joint")]["head_set_preserved"] is True
    assert out["criterion_preserved_r"] == 0.8
    assert out["criterion_cross_family_r"] == 0.1


def test_aggregate_head_scores_averages_over_trials():
    t1 = np.array([[1.0, 0.0]])
    t2 = np.array([[0.0, 0.0]])
    assert np.allclose(RH.aggregate_head_scores([t1, t2]), [[0.5, 0.0]])
    with pytest.raises(ValueError):
        RH.aggregate_head_scores([])


class _FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [abs(hash(w)) % 1000 for w in text.split()]}


def test_build_needle_context_places_the_needle_at_the_requested_depth():
    tok = _FakeTok()
    spec = RH.NEEDLE_TUPLES[0]
    trial = RH.build_needle_context(tok, spec, 200, 0.5)
    lo, hi = trial["needle_span"]
    assert trial["input_ids"][lo:hi] == trial["needle_ids"]
    assert 0.3 < lo / len(trial["input_ids"]) < 0.7
    assert len(RH.NEEDLE_TUPLES) >= 3
    assert len(RH.DEFAULT_DEPTHS) == 10


def test_build_needle_context_refuses_to_leave_the_battery_range():
    with pytest.raises(ValueError, match="battery operating range"):
        RH.build_needle_context(_FakeTok(), RH.NEEDLE_TUPLES[0],
                                RH.BATTERY_MAX_CONTEXT + 1, 0.5)


def test_write_heatmaps_always_emits_the_raw_arrays(tmp_path):
    maps = {"base": np.array([[0.1, 0.2], [0.3, 0.4]])}
    written = RH.write_heatmaps(maps, tmp_path)
    assert (tmp_path / "base.npy").exists() and (tmp_path / "base.json").exists()
    assert np.allclose(np.load(tmp_path / "base.npy"), maps["base"])
    assert any(w.endswith(".json") for w in written)


def test_retrieval_head_cli_compare_and_head_set(tmp_path, capsys):
    a = tmp_path / "base.json"
    b = tmp_path / "fixed_joint.json"
    a.write_text(json.dumps({"tag": "base",
                             "score_map": [[0.0, 0.5], [0.9, 0.1]]}), encoding="utf-8")
    b.write_text(json.dumps({"tag": "fixed_joint",
                             "score_map": [[0.0, 0.4], [0.8, 0.1]]}), encoding="utf-8")
    rc = RH.main(["compare", "--scores", str(a), "--scores", str(b),
                  "--heatmap-dir", str(tmp_path / "hm"),
                  "--out", str(tmp_path / "cmp.json")])
    assert rc == 0
    rep = json.loads((tmp_path / "cmp.json").read_text(encoding="utf-8"))
    assert rep["pairs"][0]["head_set_preserved"] is True
    assert "residual_risk" in rep and "GIST" in rep["residual_risk"]

    rc = RH.main(["head-set", "--scores", str(b), "--out", str(tmp_path / "hs.json")])
    assert rc == 0
    hs = json.loads((tmp_path / "hs.json").read_text(encoding="utf-8"))
    assert hs["heads"] == [[1, 0], [0, 1]]
    assert hs["threshold"] == 0.1
    assert "consumed_by" in hs


# ==========================================================================
# (C) CacheBlend layer-1 deviation
# ==========================================================================

def test_span_attention_output_matches_a_hand_softmax():
    q = np.array([[1.0, 0.0]])                       # [H=1, d=2]
    k = np.array([[[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]]])   # [1, S=3, 2]
    v = np.array([[[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]]])
    scale = 1.0
    out = CB.span_attention_output(q, k, v, span=(0, 2), scale=scale)
    logits = np.array([1.0, 0.0])
    p = np.exp(logits - logits.max()); p /= p.sum()
    want = p[0] * v[0, 0] + p[1] * v[0, 1]
    assert np.allclose(out[0], want)


def test_span_restriction_changes_the_softmax_denominator():
    q = np.array([[1.0, 0.0]])
    k = np.array([[[1.0, 0.0], [0.0, 1.0], [5.0, 0.0]]])
    v = np.array([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
    inside = CB.span_attention_output(q, k, v, span=(0, 2), scale=1.0)
    whole = CB.span_attention_output(q, k, v, span=None, scale=1.0)
    assert not np.allclose(inside, whole)
    # the span-restricted output only mixes the span's values
    assert inside.sum() == pytest.approx(1.0)


def test_span_attention_output_expands_gqa_and_validates_shapes():
    q = np.ones((4, 2))
    k = np.ones((2, 3, 2))
    v = np.stack([np.full((3, 2), 1.0), np.full((3, 2), 5.0)])
    out = CB.span_attention_output(q, k, v, scale=1.0)
    assert out.shape == (4, 2)
    assert np.allclose(out[:2], 1.0) and np.allclose(out[2:], 5.0)
    with pytest.raises(ValueError):
        CB.span_attention_output(np.ones(2), k, v)
    with pytest.raises(ValueError):
        CB.span_attention_output(q, np.ones((2, 3)), v)
    with pytest.raises(ValueError):
        CB.span_attention_output(q, k, v, span=(1, 1))


def test_deviation_is_the_frobenius_norm_of_the_output_difference():
    a = np.array([[3.0, 0.0]])
    b = np.array([[0.0, 4.0]])
    assert CB.deviation(a, b) == pytest.approx(5.0)
    with pytest.raises(ValueError):
        CB.deviation(a, np.zeros((2, 2)))


def test_block_deviations_are_none_where_the_probe_is_missing():
    q = np.ones((1, 2))
    gk = np.ones((1, 6, 2))
    gv = np.tile(np.arange(6.0).reshape(6, 1), (1, 2))[None, :, :]
    spans = [(0, 2), (2, 4), (4, 6)]
    raw = [(np.ones((1, 2, 2)), np.zeros((1, 2, 2))), None,
           (np.ones((1, 2, 2)), np.zeros((1, 2, 2)))]
    dev = CB.block_deviations(q, gk, gv, spans, raw)
    assert dev[1] is None
    assert dev[0] is not None and dev[2] is not None
    assert dev[2] > dev[0]          # later gist values are further from zero


def test_dev_estimators_are_registered_separately():
    dev = [0.1, None, 0.9, 0.4]
    assert CB.dev_argmax(dev) == 2                     # locator estimand
    s = CB.dev_trigger_scalars(dev)                    # trigger estimand
    assert s["cacheblend_dev_max"] == pytest.approx(0.9)
    assert s["cacheblend_dev_mean"] == pytest.approx(np.mean([0.1, 0.9, 0.4]))
    assert s["cacheblend_dev_margin"] == pytest.approx(0.5)
    assert CB.dev_argmax([None, None]) is None
    assert all(v is None for v in CB.dev_trigger_scalars([None]).values())
    assert CB.dev_trigger_scalars([0.3])["cacheblend_dev_margin"] is None


def test_s0_swap_control_reads_no_compressed_information():
    q = np.ones((1, 2))
    raw = [(np.ones((1, 2, 2)), np.zeros((1, 2, 2))),
           (np.ones((1, 2, 2)), np.ones((1, 2, 2))),
           None]
    dev = CB.s0_swap_control(raw, q, swap=1)
    assert dev[0] == pytest.approx(np.sqrt(2))         # 0 vs 1 over [H=1,d=2]
    assert dev[1] is None                              # swapped partner missing
    assert dev[2] is None


def test_cacheblend_locator_control_uses_the_same_chooser_as_the_table():
    """The inverted control must invert the locator that is reported, on the
    same 93-row denominator."""
    frame = _witness_frame()
    dev_rows = {"sA:1": {"qid": "sA:1", "dev": [0.1, 0.2, 0.9]}}   # k*=2
    out = CB.locator_report(dev_rows, frame)
    assert out["table"]["hits"] == 1 and out["table"]["n"] == 2
    fwd = out["inverted_control"]["forward"]
    assert fwd["hits"] == out["table"]["hits"]
    assert fwd["n"] == out["table"]["n"] == 2          # unprobed row abstains here too
    assert out["inverted_control"]["inverted"]["hits"] == 0
    assert out["n_with_dev"] == 1
    for vec in ([0.1, 0.2, 0.9], [0.0, 0.0], [1.0, np.nan]):
        assert C.chooser_argmax(QR.chooser_domain(vec)) == CB.dev_argmax(vec)


def test_layerwise_spearman_pregate():
    dev_by_layer = [[1.0, 2.0, 3.0, 4.0],
                    [1.1, 2.2, 3.3, 4.4],
                    [4.0, 3.0, 2.0, 1.0]]
    out = CB.layerwise_spearman(dev_by_layer)
    assert out["pairs"][0]["rho"] == pytest.approx(1.0)
    assert out["pairs"][1]["rho"] == pytest.approx(-1.0)
    assert out["mean_rho"] == pytest.approx(0.0)
    assert out["n_pairs_defined"] == 2


def test_layerwise_spearman_drops_undefined_blocks_and_reports_n():
    out = CB.layerwise_spearman([[1.0, None, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    assert out["pairs"][0]["n"] == 3
    out2 = CB.layerwise_spearman([[1.0, None], [1.0, 2.0]])
    assert out2["pairs"][0]["rho"] is None and out2["mean_rho"] is None


def test_pregate_summary_aggregates_rows():
    s = CB.pregate_summary([{"mean_rho": 0.9}, {"mean_rho": 0.5}, {"mean_rho": None}])
    assert s["n_rows"] == 3 and s["n_rows_defined"] == 2
    assert s["mean_rho"] == pytest.approx(0.7)


def test_cacheblend_locator_and_features_over_the_frame(tmp_path):
    frame = _witness_frame()
    dev_rows = {
        "sA:1": {"qid": "sA:1", "dev": [0.1, 0.2, 0.9], "layer1_probe_sec": [0.01] * 3,
                 "layer1_probe_sec_total": 0.03, "probe_layer": 1},
        "sA:2": {"qid": "sA:2", "dev": [0.9, 0.2, 0.1], "layer1_probe_sec": [0.02] * 3,
                 "layer1_probe_sec_total": 0.06, "probe_layer": 1},
        "sB:1": {"qid": "sB:1", "dev": [0.1, 0.1], "layer1_probe_sec": [0.01, 0.01],
                 "layer1_probe_sec_total": 0.02, "probe_layer": 1},
    }
    loc = CB.locator_report(dev_rows, frame)
    assert loc["table"]["hits"] == 2 and loc["table"]["n"] == 2
    assert loc["inverted_control"]["inverted"]["hits"] == 0

    rows = CB.feature_rows(dev_rows, frame, arm="c2kv")
    assert len(rows) == 4
    by = {r["qid"]: r for r in rows}
    assert by["sA:1"]["cacheblend_dev_max"] == pytest.approx(0.9)
    assert by["sB:2"]["cacheblend_dev_max"] is None
    C.write_features_jsonl(tmp_path / "f.jsonl", rows, context="test")

    rep = CB.evaluate(rows, frame, C.load_orientations(ORIENT_PATH), reps=50)
    ent = rep["trigger"] if "trigger" in rep else rep
    e = ent["features"]["cacheblend_dev_max"]
    assert e["orientation"] == 1
    assert e["n"] == 3 and e["n_pos"] == 2
    assert e["prevalence_chance_ap"] == pytest.approx(2 / 3)

    cost = CB.cost_summary(dev_rows)
    assert cost["n_block_probes"] == 8
    assert cost["layer1_probe_sec_per_decision_total"] == pytest.approx(0.11)


def test_cacheblend_score_cli_end_to_end(tmp_path, monkeypatch, capsys):
    frame = _witness_frame()
    monkeypatch.setattr(C.FrozenAssets, "load", lambda self: frame)
    dev = tmp_path / "dev.jsonl"
    with dev.open("w", encoding="utf-8") as fh:
        for qid, vec in (("sA:1", [0.1, 0.2, 0.9]), ("sA:2", [0.9, 0.1, 0.1]),
                         ("sB:1", [0.1, 0.1, 0.1]), ("sB:2", [0.2, 0.1, 0.1])):
            fh.write(json.dumps({"qid": qid, "dev": vec, "probe_layer": 1,
                                 "layer1_probe_sec": [0.01] * 3,
                                 "layer1_probe_sec_total": 0.03}) + "\n")
    rc = CB.main(["score", "--root", str(tmp_path), "--dev", str(dev),
                  "--features", str(tmp_path / "feat.jsonl"),
                  "--report", str(tmp_path / "rep.json"), "--reps", "20"])
    assert rc == 0
    rep = json.loads((tmp_path / "rep.json").read_text(encoding="utf-8"))
    assert rep["locator"]["table"]["floor"] == 0.25
    assert rep["trigger"]["baseline_parse_fail"]["n"] == 4
    assert rep["cost"]["n_block_probes"] == 12
    assert any(d["paper"] == "2405.16444" for d in rep["deviations"])
    assert (tmp_path / "feat.jsonl").exists()


# ==========================================================================
# cross-cutting: orientations, deviations, docstring citations
# ==========================================================================

def test_orientations_file_matches_the_declared_rationales():
    orient = C.load_orientations(ORIENT_PATH)
    declared = dict(QR.ORIENTATION_RATIONALE)
    declared.update(CB.ORIENTATION_RATIONALE)
    assert set(orient) == set(declared)
    for name, (sign, reason) in declared.items():
        assert orient[name] == sign, name
        assert sign in (-1, 1)
        assert len(reason) > 20


def test_every_emitted_feature_has_an_orientation():
    orient = C.load_orientations(ORIENT_PATH)
    for name in QR.TRIGGER_FEATURES:
        assert name in orient
    for name in CB.TRIGGER_FEATURES:
        assert name in orient


def test_feature_names_survive_the_leakage_guard():
    from t33_labels import guard_columns
    guard_columns(list(QR.TRIGGER_FEATURES) + list(CB.TRIGGER_FEATURES),
                  context="orientation names")


def test_every_module_declares_its_deviations():
    for mod, paper in ((QR, "2506.09944"), (RH, "2404.15574"), (CB, "2405.16444")):
        assert isinstance(mod.DEVIATIONS, list) and mod.DEVIATIONS
        for d in mod.DEVIATIONS:
            assert set(d) == {"method", "paper", "what", "why"}
            assert d["paper"] == paper


def test_public_functions_cite_the_paper_they_implement():
    import inspect
    checks = [
        (QR.qr_score_perdoc, "2506.09944"),
        (QR.qr_score_agg, "2506.09944"),
        (QR.retriever_scores, "2506.09944"),
        (QR.calibrate, "2506.09944"),
        (QR.heads_from_fraction, "2506.09944"),
        (RH.copy_paste_tokens, "2404.15574"),
        (RH.retrieval_score, "2404.15574"),
        (RH.retrieval_head_mask, "2404.15574"),
        (RH.pearson_head_maps, "Sec. 3.3"),
        (CB.span_attention_output, "Attn_l"),
        (CB.layerwise_spearman, "2405.16444"),
        (CB.s0_swap_control, "digest 4.5"),
    ]
    for fn, needle in checks:
        doc = inspect.getdoc(fn) or ""
        assert needle in doc, fn.__name__


def test_clis_expose_help():
    for mod in (QR, RH, CB):
        with pytest.raises(SystemExit) as exc:
            mod.main(["--help"])
        assert exc.value.code == 0


# --------------------------------------------------------------------------
# torch-gated smoke test for the partial prefill
# --------------------------------------------------------------------------
def test_partial_prefill_kv_agrees_with_torch_softmax_attention():
    """torch-gated: our numpy span attention must equal torch's own softmax
    attention over the same span (the arithmetic the NPU probe relies on)."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    q = rng.normal(size=(4, 6))
    k = rng.normal(size=(2, 5, 6))
    v = rng.normal(size=(2, 5, 6))
    got = CB.span_attention_output(q, k, v, span=(1, 4), scale=0.5)
    tq = torch.tensor(q)
    tk = torch.tensor(np.repeat(k, 2, axis=0))[:, 1:4, :]
    tv = torch.tensor(np.repeat(v, 2, axis=0))[:, 1:4, :]
    logits = torch.einsum("hd,hsd->hs", tq, tk) * 0.5
    want = torch.einsum("hs,hsd->hd", torch.softmax(logits, dim=-1), tv)
    assert np.allclose(got, want.numpy())


def test_probe_row_reduction_with_a_fake_partial_prefill(monkeypatch):
    """Wiring/reduction test for the NPU probe: the torch call is replaced by a
    synthetic K,V so the per-block loop, the timers and the S0 swap are
    exercised without torch."""
    def fake_prefill(model, ids, positions, probe_layer):
        pos = list(positions)
        assert len(pos) == len(ids)
        # ledger positions must be contiguous and start where ctx says
        assert pos == list(range(pos[0], pos[0] + len(ids)))
        val = float(len(ids))
        return (np.ones((1, len(ids), 2)), np.full((1, len(ids), 2), val), 0.001)

    monkeypatch.setattr(CB, "partial_prefill_kv", fake_prefill)
    ctx = {
        "qid": "sA:1", "arm": "c2kv", "scale": 1.0,
        "doc_ids": [[1, 2], [3, 4, 5], []],
        "offsets": [10, 12, 15],
        "gist_spans": [(0, 1), (1, 2), (2, 3)],
        "gist_k": np.ones((1, 3, 2)),
        "gist_v": np.zeros((1, 3, 2)),
        "q": np.ones((1, 2)),
    }
    row = CB.probe_row(object(), ctx, probe_layer=1, s0_swap=True)
    assert row["qid"] == "sA:1" and row["probe_layer"] == 1
    assert row["dev"][2] is None                    # empty block -> None
    assert row["dev"][0] == pytest.approx(np.sqrt(2 * 2.0 ** 2))
    assert row["dev"][1] == pytest.approx(np.sqrt(2 * 3.0 ** 2))
    assert row["layer1_probe_sec"][2] is None
    assert row["layer1_probe_sec_total"] == pytest.approx(0.002)
    assert row["dev_s0_swap"][0] == pytest.approx(np.sqrt(2 * 1.0 ** 2))
    assert row["dev_s0_swap"][2] is None


class _FakeDecodeCapture:
    """Stand-in for U1a's AttentionRowCapture during the needle decode."""

    def __init__(self, key_map, query_mode, last_n, layers):
        assert query_mode in ("decode", "prefill_last_n")
        self.key_map = key_map
        self.query_mode = query_mode
        self._argmax = None

    def set_argmax(self, arr):
        self._argmax = np.asarray(arr)

    def argmax_tensor(self):
        return self._argmax


def test_run_needle_trial_wiring_with_a_fake_decode(monkeypatch):
    """Wiring/reduction test for the needle detection: a fake U1a capture and a
    fake greedy decode drive the copy-paste rule end to end."""
    fake = types.SimpleNamespace(
        CLASSES=QR.CLASSES,
        KeyClassMap=lambda **kw: types.SimpleNamespace(**kw),
        AttentionRowCapture=_FakeDecodeCapture,
    )
    trial = {"input_ids": [5, 6, 7, 8], "needle_ids": [6, 7],
             "needle_span": (1, 3), "answer": "x"}
    seen = {}

    def decode_fn(tr, capture, max_new_tokens):
        seen["max_new_tokens"] = max_new_tokens
        seen["query_span"] = capture.key_map.query_span
        seen["doc_spans"] = capture.key_map.doc_spans
        capture.set_argmax(np.array([[[1, 2], [0, 0]]]))   # [L=1,H=2,T=2]
        return [6, 7]

    got = RH.run_needle_trial(trial, decode_fn=decode_fn, attention_module=fake,
                              max_new_tokens=32)
    assert got.shape == (1, 2)
    assert np.allclose(got, [[1.0, 0.0]])
    assert seen["max_new_tokens"] == 32
    assert seen["doc_spans"] == [(1, 3)]
    assert seen["query_span"] == (4, 4)


def test_run_needle_trial_truncates_argmax_to_the_generated_length():
    fake = types.SimpleNamespace(
        CLASSES=QR.CLASSES,
        KeyClassMap=lambda **kw: types.SimpleNamespace(**kw),
        AttentionRowCapture=_FakeDecodeCapture,
    )
    trial = {"input_ids": [5, 6, 7], "needle_ids": [6], "needle_span": (1, 2)}

    def decode_fn(tr, capture, max_new_tokens):
        capture.set_argmax(np.array([[[1, 1, 1]]]))       # 3 steps captured
        return [6]                                        # 1 token generated

    got = RH.run_needle_trial(trial, decode_fn=decode_fn, attention_module=fake)
    assert got.shape == (1, 1)
    assert got[0, 0] == pytest.approx(1.0)


# ==========================================================================
# pre-registered controls (digest 4.0 contract + digest 4.5 per-entry)
# ==========================================================================

def test_cap_stratum_uses_the_manifest_cap():
    assert QR.cap_stratum({"generated_tokens": 128}, 128) == "censored"
    assert QR.cap_stratum({"generated_tokens": 127}, 128) == "uncensored"
    assert QR.cap_stratum({}, 128) == "unknown"


def test_length_controls_read_only_free_s8_columns():
    frame = C.FrozenFrame(
        pairs=[({"qid": "sA:1"},
                {"qid": "sA:1", "doc_chunks": 4, "kept_history_tokens": 900,
                 "generated_tokens": 128})],
        labels=[{"qid": "sA:1", "label_cw": 1}], manifest={})
    got = QR.length_control_scores(frame, ["sA:1"])
    assert got["n_docs"][0] == 4
    assert got["kept_history_tokens"][0] == 900
    assert set(got) == {"n_docs", "kept_history_tokens", "generated_tokens"}


def test_random_matched_rate_control_tracks_one_over_n_blocks():
    truth = {f"q{i}": 0 for i in range(40)}
    ctrl = QR.random_locator_control({f"q{i}": 4 for i in range(40)}, truth, reps=400)
    assert ctrl["n"] == 40
    assert ctrl["mean_s_at_k"] == pytest.approx(0.25, abs=0.05)
    assert ctrl["frozen_floor"] == 0.25
    assert QR.random_locator_control({}, {})["mean_s_at_k"] is None


def test_random_matched_rate_control_ignores_abstained_rows():
    ctrl = QR.random_locator_control({"a": 3, "b": 3}, {"a": 1, "b": None}, reps=50)
    assert ctrl["n"] == 1


def test_nested_cv_operating_point_selects_the_rate_out_of_fold():
    rng = np.random.default_rng(5)
    groups = np.repeat(np.arange(20), 5)
    y = (groups % 2 == 0).astype(int)
    s = y + rng.normal(0, 0.05, size=y.shape)          # separable
    out = QR.nested_cv_operating_point(s, y, groups)
    assert out["objective"] == QR.INNER_FOLD_OBJECTIVE
    assert len(out["chosen_rates"]) == 5
    assert out["coverage"] > 0.7 * out["n_pos"]
    assert out["precision"] is not None
    # a pure-noise score must not manufacture coverage
    noise = rng.normal(size=y.shape)
    weak = QR.nested_cv_operating_point(noise, y, groups)
    assert weak["coverage"] <= out["coverage"]


def test_nested_cv_operating_point_reports_both_denominators():
    groups = np.repeat(np.arange(10), 4)
    y = np.tile([1, 0, 0, 0], 10)
    out = QR.nested_cv_operating_point(np.arange(40, dtype=float), y, groups)
    assert out["n_pos"] == 10 and out["n_neg"] == 30
    assert out["false_reset_rate"] is not None
    assert out["per_step_false_fire_rate"] is not None


def test_stratified_metrics_reports_n_and_prevalence_per_stratum():
    s = np.array([0.9, 0.1, 0.8, 0.2])
    y = np.array([1, 0, 1, 0])
    out = QR.stratified_metrics(s, y, ["censored", "censored",
                                       "uncensored", "uncensored"])
    assert set(out) == {"censored", "uncensored"}
    assert out["censored"]["n"] == 2 and out["censored"]["n_pos"] == 1
    assert out["censored"]["prevalence_chance_ap"] == pytest.approx(0.5)


def test_evaluate_features_carries_every_preregistered_control():
    frame = _witness_frame()
    scored = {q: {"r_unc": np.array([0.9 - 0.1 * i, 0.1]), "r_cal": None,
                  "profile": {c: 0.2 for c in QR.CLASSES}, "meta": {}}
              for i, q in enumerate(["sA:1", "sA:2", "sB:1", "sB:2"])}
    rows = QR.feature_rows(scored, frame, arm="c2kv")
    rep = QR.evaluate_features(rows, frame, C.load_orientations(ORIENT_PATH), reps=20)
    ent = rep["features"]["qrhead_r_max"]
    for key in ("nested_cv_operating_point", "by_cap_stratum", "length_controls",
                "s0_full_arm", "operating_point_at_baseline_fires", "ap_ci"):
        assert key in ent, key
    assert rep["cap_tokens"] == 128


def test_cacheblend_evaluate_carries_the_same_controls():
    frame = _witness_frame()
    dev_rows = {q: {"qid": q, "dev": [0.9 - 0.1 * i, 0.1], "probe_layer": 1,
                    "layer1_probe_sec": [0.01, 0.01], "layer1_probe_sec_total": 0.02}
                for i, q in enumerate(["sA:1", "sA:2", "sB:1", "sB:2"])}
    rows = CB.feature_rows(dev_rows, frame, arm="c2kv")
    rep = CB.evaluate(rows, frame, C.load_orientations(ORIENT_PATH), reps=20)
    ent = rep["features"]["cacheblend_dev_max"]
    for key in ("nested_cv_operating_point", "by_cap_stratum", "length_controls",
                "s0_full_arm_swap"):
        assert key in ent, key
    loc = CB.locator_report(dev_rows, frame)
    assert "random_matched_rate" in loc and "legacy_k_first" in loc


# ==========================================================================
# unit U1a interface contract (imported lazily; skipped if U1a is absent)
# ==========================================================================

def test_remove_capture_shim_accepts_both_signatures():
    calls = []

    class _NoArg:
        def remove(self):
            calls.append("noarg")

    class _WithModel:
        def remove(self, model):
            calls.append(("model", model))

    QR._remove_capture(_NoArg(), "M")
    QR._remove_capture(_WithModel(), "M")
    assert calls == ["noarg", ("model", "M")]


def test_u1a_interface_matches_what_this_unit_calls():
    """Drift guard against unit U1a's concurrently written t34_attention."""
    A = pytest.importorskip("t34_attention")
    import inspect
    assert tuple(A.CLASSES) == QR.CLASSES
    kcm = inspect.signature(A.KeyClassMap.__init__).parameters
    assert {"system_len", "doc_spans", "raw_tail_span", "query_span",
            "generated_start"} <= set(kcm)
    cap = inspect.signature(A.AttentionRowCapture.__init__).parameters
    assert {"key_map", "query_mode", "last_n", "layers"} <= set(cap)
    for name in ("install", "remove", "class_mass_tensor", "doc_mass_tensor",
                 "argmax_tensor"):
        assert callable(getattr(A.AttentionRowCapture, name)), name


def test_run_needle_trial_decode_mode_shifts_the_generated_alignment():
    """query_mode="decode" does not capture the prompt forward, so its first
    row emits generated token 1 -- an unshifted read would invent copy events."""
    fake = types.SimpleNamespace(
        CLASSES=QR.CLASSES,
        KeyClassMap=lambda **kw: types.SimpleNamespace(**kw),
        AttentionRowCapture=_FakeDecodeCapture,
    )
    trial = {"input_ids": [5, 6, 7], "needle_ids": [6, 7], "needle_span": (1, 3)}

    def decode_fn(tr, capture, max_new_tokens):
        # one captured row, attending position 2 (token 7)
        capture.set_argmax(np.array([[[2]]]))
        return [6, 7]                       # token 0 was emitted by the prompt

    shifted = RH.run_needle_trial(trial, decode_fn=decode_fn,
                                  attention_module=fake, query_mode="decode")
    assert shifted[0, 0] == pytest.approx(0.5)      # only token 7 counted
    unshifted = RH.run_needle_trial(trial, decode_fn=decode_fn,
                                    attention_module=fake,
                                    query_mode="prefill_last_n")
    assert unshifted[0, 0] == pytest.approx(0.0)    # row 0 vs token 6: no match
    assert RH.NEEDLE_QUERY_MODE == "prefill_last_n"


def test_detect_on_model_loops_the_full_grid_with_a_fake_decode():
    """The Sec. 2 detection loop (tuples x lengths x depths) with no torch."""
    fake = types.SimpleNamespace(
        CLASSES=QR.CLASSES,
        KeyClassMap=lambda **kw: types.SimpleNamespace(**kw),
        AttentionRowCapture=_FakeDecodeCapture,
    )
    seen = []

    def decode_fn(trial, capture, max_new_tokens):
        seen.append((trial["tuple"], trial["length_tokens"], trial["depth"]))
        lo = trial["needle_span"][0]
        capture.set_argmax(np.array([[[lo]]]))
        return [trial["needle_ids"][0]]

    out = RH.detect_on_model(_FakeTok(), decode_fn=decode_fn,
                             lengths=(256, 512), depths=(0.0, 0.5),
                             attention_module=fake)
    assert out["n_trials"] == len(RH.NEEDLE_TUPLES) * 2 * 2 == len(seen)
    assert np.asarray(out["score_map"]).shape == (1, 1)
    assert out["length_grid"] == [256, 512] and out["depths"] == [0.0, 0.5]
    assert out["query_mode"] == RH.NEEDLE_QUERY_MODE
    assert out["sparsity"]["n_heads_total"] == 1


# ==========================================================================
# sidecar -> prefix plan (torch-free half of the capture wiring)
# ==========================================================================

class _PlanTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


#: The generation-prompt suffix this fake chat template appends (the real
#: Qwen template appends an assistant header).
_GEN_SUFFIX = [7, 8]


def _plan_chat_template_ids(tok, msgs, tools=None, keep_bos=False,
                            max_length=None, add_generation_prompt=False):
    ids = [1, 2] + [ord(c) for c in (msgs[0]["content"] or "")]
    if max_length is not None:
        ids = ids[:max_length]
    return ids + (_GEN_SUFFIX if add_generation_prompt else [])


_PLAN_HARNESS = types.SimpleNamespace(_chat_template_ids=_plan_chat_template_ids)


def _plan_rec(docs=("aaa", "bbbb"), query="qq", lengths=None):
    return {"qid": "sA:1", "system_prompt": "S", "tools": [],
            "docs": list(docs), "query": query,
            "doc_lengths": list(lengths) if lengths is not None
            else [len(d) for d in docs],
            "dropped_docs": []}


def test_sidecar_prefix_plan_counts_the_logical_history_length():
    plan = QR.sidecar_prefix_plan(_plan_rec(), _PlanTok(), harness=_PLAN_HARNESS)
    assert plan["doc_lengths"] == [3, 4]
    assert plan["history_logical_len"] == 7      # LOGICAL, not the gist length
    # the sidecar's `query` text is the templated current message WITHOUT the
    # generation prompt; the harness's router appends it, so the plan must too
    assert plan["query_ids"] == [ord("q")] * 2 + _GEN_SUFFIX
    assert plan["n_generation_prompt_tokens"] == len(_GEN_SUFFIX)
    assert plan["system_ids"][:2] == [1, 2]
    assert plan["max_system_length"] == QR.FROZEN_MAX_SYSTEM_LENGTH == 4096


def test_prefix_plan_restores_the_generation_prompt_the_sidecar_drops():
    """agent/t34_dump_sidecar.py decodes _chat_template_ids(current) WITHOUT
    add_generation_prompt, while eval_agent_history_c2kv:2777 builds the query
    span WITH it.  |q| in Eq. score_perdoc and the CacheBlend probe's LAST
    query token both change if the suffix is dropped."""
    assert QR.generation_prompt_ids(_PlanTok(), _PLAN_HARNESS) == _GEN_SUFFIX
    off = QR.sidecar_prefix_plan(_plan_rec(), _PlanTok(), harness=_PLAN_HARNESS,
                                 add_generation_prompt=False)
    on = QR.sidecar_prefix_plan(_plan_rec(), _PlanTok(), harness=_PLAN_HARNESS)
    assert len(on["query_ids"]) == len(off["query_ids"]) + len(_GEN_SUFFIX)
    assert off["n_generation_prompt_tokens"] == 0
    # a template whose generation prompt is not a pure suffix is refused, not
    # silently ignored
    bad = types.SimpleNamespace(
        _chat_template_ids=lambda tok, msgs, tools=None, keep_bos=False,
        max_length=None, add_generation_prompt=False: [9] + [1, 2])
    with pytest.raises(AssertionError, match="generation prompt"):
        QR.generation_prompt_ids(_PlanTok(), bad)


def test_null_query_keeps_the_prompt_structure():
    """2506.09944 Sec. 4.1 swaps only the QUERY.  With a chat-templated prompt
    that means the same turn with "N/A" inside it, not a bare 2-token string."""
    ids = QR.null_query_ids(_PlanTok(), _PLAN_HARNESS)
    assert ids[:2] == [1, 2]                 # the template wrapper survives
    assert ids[-len(_GEN_SUFFIX):] == _GEN_SUFFIX
    assert [ord(c) for c in QR.NULL_QUERY_TEXT] == ids[2:-len(_GEN_SUFFIX)]
    assert QR.NULL_QUERY_TEXT == "N/A"


def test_prefix_plan_truncates_the_system_prefill_at_the_frozen_cap():
    """system_length shifts every ledger offset, so the frozen recipe's
    --max_system_length (4096, read off t34_dump_sidecar._harness_args) has to
    be applied when the system prefix is rebuilt from the sidecar text."""
    rec = _plan_rec()
    rec["system_prompt"] = "S" * 50
    plan = QR.sidecar_prefix_plan(rec, _PlanTok(), harness=_PLAN_HARNESS,
                                  max_system_length=8)
    assert len(plan["system_ids"]) == 8
    assert plan["max_system_length"] == 8


def test_sidecar_prefix_plan_refuses_a_tokenizer_round_trip_that_shifts_spans():
    rec = _plan_rec(lengths=[3, 9])              # sidecar disagrees
    with pytest.raises(AssertionError, match="doc_lengths"):
        QR.sidecar_prefix_plan(rec, _PlanTok(), harness=_PLAN_HARNESS)


def test_sidecar_prefix_plan_refuses_an_empty_query_and_an_oversized_grid():
    with pytest.raises(AssertionError, match="empty query"):
        QR.sidecar_prefix_plan(_plan_rec(query=""), _PlanTok(), harness=_PLAN_HARNESS)
    with pytest.raises(AssertionError, match="max_doc_num"):
        QR.sidecar_prefix_plan(_plan_rec(docs=("a", "b", "c")), _PlanTok(),
                               harness=_PLAN_HARNESS, max_doc_num=2)


def test_sidecar_prefix_plan_truncates_at_max_doc_length():
    rec = _plan_rec(docs=("aaaaa",), lengths=[2])
    plan = QR.sidecar_prefix_plan(rec, _PlanTok(), harness=_PLAN_HARNESS,
                                  max_doc_length=2)
    assert plan["doc_lengths"] == [2]


def test_pad_row_matches_the_harness_grid_convention():
    assert QR._pad_row([1, 2], 4) == [1, 2, -100, -100]
    assert QR._pad_row([1, 2, 3, 4, 5], 3) == [1, 2, 3]


def test_load_sidecar_keys_by_qid(tmp_path):
    p = tmp_path / "sidecar.jsonl"
    p.write_text(json.dumps({"qid": "sA:1", "docs": ["a"]}) + "\n"
                 + json.dumps({"qid": "sB:1", "docs": ["b"]}) + "\n",
                 encoding="utf-8")
    got = CB.load_sidecar(p)
    assert set(got) == {"sA:1", "sB:1"} and got["sA:1"]["docs"] == ["a"]


# ==========================================================================
# fidelity-review additions
# ==========================================================================

class _GrowingCache:
    """Minimal stand-in for the prefix cache: ``update`` appends (which is what
    ``Qwen3Attention.forward`` does even under ``use_cache=False``,
    modeling_qwen3.py:277) and ``crop`` truncates."""

    def __init__(self, length):
        self.length = int(length)

    def get_seq_length(self):
        return self.length

    def update(self, n):
        self.length += int(n)

    def crop(self, length):
        self.length = min(self.length, int(length))


class _UncroppableCache(_GrowingCache):
    crop = None


def test_reset_cache_restores_the_prefix_before_the_q_null_forward():
    """arXiv 2506.09944 Sec. 4.1 swaps ONLY the query: R(q_null, d_i) must be
    measured against the same context as R(q, d_i)."""
    cache = _GrowingCache(100)
    assert QR.reset_cache_to(cache, 100) == 100      # first forward: no-op
    cache.update(17)                                  # the query's K,V land here
    assert cache.get_seq_length() == 117
    assert QR.reset_cache_to(cache, 100) == 100      # q_null sees the prefix only
    assert cache.get_seq_length() == 100


def test_reset_cache_refuses_a_cache_it_cannot_restore():
    cache = _UncroppableCache(100)
    cache.update(5)
    with pytest.raises(RuntimeError, match="crop"):
        QR.reset_cache_to(cache, 100)
    with pytest.raises(RuntimeError, match="shrank"):
        QR.reset_cache_to(_GrowingCache(90), 100)


def test_head_stability_halves_are_session_disjoint(tmp_path):
    """arXiv 2506.09944 Sec. 6.4 measures top-m overlap across DISJOINT detection
    subsets; interleaving rows by index would put one session on both sides."""
    cap_dir = tmp_path / "attn"
    det = {"rows": []}
    # two sessions, two rows each; the two sessions prefer different heads, so
    # a session-disjoint split must find overlap 0 while an index-parity split
    # (which mixes both sessions into both halves) would find overlap 1.
    for sess, head in (("sA", 0), ("sB", 1)):
        for step in (1, 2):
            dm = np.full((1, 2, 2, 2), 0.01)
            dm[0, head, :, 0] += 1.0
            QR.write_capture(cap_dir, f"{sess}:{step}", doc_mass=dm,
                             class_mass=np.ones((1, 2, 2, 5)),
                             meta={"query_proj": "gist"})
            det["rows"].append({"qid": f"{sess}:{step}", "gold_blocks": [0]})
    table = QR.detect_heads(cap_dir, det, m=1)
    assert table["stability_top_m_overlap_halves"] == 0
    assert "session-disjoint" in table["stability_split"]


def test_head_stability_is_undefined_with_one_detection_session(tmp_path):
    cap_dir = tmp_path / "attn"
    det = {"rows": []}
    for step in (1, 2):
        QR.write_capture(cap_dir, f"sA:{step}", doc_mass=np.ones((1, 1, 2, 2)),
                         class_mass=np.ones((1, 1, 2, 5)),
                         meta={"query_proj": "gist"})
        det["rows"].append({"qid": f"sA:{step}", "gold_blocks": [0]})
    table = QR.detect_heads(cap_dir, det, m=1)
    assert table["stability_top_m_overlap_halves"] is None
    assert "fewer than 2 detection sessions" in table["stability_split"]


def test_cacheblend_deviation_is_not_claimed_to_be_the_papers_cad():
    """The statistic lives in OUTPUT space; the paper's CAD is a norm on the
    attention MATRIX and its KVD is per-token K,V.  Neither is shape-compatible
    with a gist span, so the departure has to be declared."""
    entry = [d for d in CB.DEVIATIONS if d["method"] == "deviation statistic"]
    assert entry and "attention OUTPUT" in entry[0]["what"]
    assert "MATRIX" in entry[0]["what"] and "KVD" in entry[0]["what"]
    assert "CAD" not in CB.deviation.__doc__.split("Motivated by")[0]


def test_cacheblend_declares_the_pregate_estimand_shift():
    names = {d["method"] for d in CB.DEVIATIONS}
    assert "Insight-2 pre-gate estimand" in names


def test_cacheblend_evaluate_reports_both_confidence_intervals():
    frame = _witness_frame()
    dev_rows = {q: {"qid": q, "dev": [0.9 - 0.1 * i, 0.1], "probe_layer": 1,
                    "layer1_probe_sec": [0.01], "layer1_probe_sec_total": 0.01}
                for i, q in enumerate(["sA:1", "sA:2", "sB:1", "sB:2"])}
    rows = CB.feature_rows(dev_rows, frame, arm="c2kv")
    rep = CB.evaluate(rows, frame, C.load_orientations(ORIENT_PATH), reps=20)
    ent = rep["features"]["cacheblend_dev_max"]
    assert "ap_ci" in ent and "auroc_ci" in ent


def test_class_mass_coverage_exposes_the_profiles_denominator():
    """The profile renormalises over the CLASSIFIED mass; without the
    denominator a 0.75 gist share could be 0.75 of 2 % of the attention."""
    cm = np.zeros((1, 1, 2, 5))
    cm[0, 0, :, 0] = 0.005      # system_raw
    cm[0, 0, :, 1] = 0.015      # history_gist
    prof = QR.class_mass_profile(cm, [(0, 0)])
    assert prof["history_gist"] == pytest.approx(0.75)
    assert QR.class_mass_coverage(cm, [(0, 0)]) == pytest.approx(0.02)


def test_score_rows_carries_the_coverage_next_to_the_profile(tmp_path):
    cap_dir = tmp_path / "attn"
    cm = np.zeros((1, 1, 2, 5))
    cm[0, 0, :, 1] = 0.5
    QR.write_capture(cap_dir, "sA:1", doc_mass=np.ones((1, 1, 2, 2)),
                     class_mass=cm, meta={"query_proj": "gist"})
    scored = QR.score_rows(cap_dir, [(0, 0)], ["sA:1"])
    assert scored["sA:1"]["class_mass_coverage"] == pytest.approx(0.5)
    assert scored["sA:1"]["profile"]["history_gist"] == pytest.approx(1.0)


def test_calibrated_control_denominator_matches_when_only_some_rows_calibrate():
    """A row captured WITHOUT the q_null forward must still occupy the
    calibrated arm's denominator, exactly as an uncaptured row does."""
    frame = _witness_frame()
    scored = {
        "sA:1": {"r_unc": np.array([0.9, 0.1]), "r_cal": np.array([0.5, -0.2]),
                 "profile": {c: 0.2 for c in QR.CLASSES}, "meta": {}},
        # captured, but WITHOUT the q_null forward
        "sA:2": {"r_unc": np.array([0.2, 0.8]), "r_cal": None,
                 "profile": {c: 0.2 for c in QR.CLASSES}, "meta": {}},
    }
    tables = QR.locator_tables(scored, frame)
    n_head = tables["calibrated"]["n"]
    assert n_head == tables["uncalibrated"]["n"] == 2
    assert tables["calibrated"]["abstained"] == 1
    assert tables["inverted_control_calibrated"]["forward"]["n"] == n_head
    assert tables["inverted_control_uncalibrated"]["forward"]["n"] == n_head
