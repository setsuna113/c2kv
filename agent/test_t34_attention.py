# -*- coding: utf-8 -*-
"""Tests for agent/t34_attention.py (t34 unit U1a, digest section 4.5).

Run from the worktree root:
    PYTHONIOENCODING=utf-8 python -m pytest agent/test_t34_attention.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import t34_attention as A  # noqa: E402
from t34_common import session_of  # noqa: E402

ROOT = _HERE.parent


# ---------------------------------------------------------------------------
# KeyClassMap
# ---------------------------------------------------------------------------

def simple_map(system_len=4, doc_spans=((4, 6), (6, 9)), tail=(9, 11), query=(11, 14), gen=14):
    return A.KeyClassMap(system_len, list(doc_spans), tail, query, gen)


def test_classes_are_the_frozen_cross_unit_contract():
    assert A.CLASSES == ("system_raw", "history_gist", "history_raw_tail",
                         "current_query", "generated_so_far")
    assert A.CLASS_INDEX["history_gist"] == 1


def test_key_class_map_partitions_every_key_exactly_once():
    km = simple_map()
    n_keys = 18
    cid = km.class_ids(n_keys)
    assert cid.shape == (n_keys,)
    assert (cid >= 0).all(), "no key may be left unclassified in a complete map"
    cov = km.coverage(n_keys)
    assert cov["n_unclassified"] == 0
    assert sum(cov[c] for c in A.CLASSES) == n_keys
    assert cov["system_raw"] == 4
    assert cov["history_gist"] == 5
    assert cov["history_raw_tail"] == 2
    assert cov["current_query"] == 3
    assert cov["generated_so_far"] == 4


def test_key_class_map_doc_ids_and_lengths():
    km = simple_map()
    did = km.doc_ids(18)
    assert km.n_docs == 2
    assert did[4] == 0 and did[5] == 0
    assert did[6] == 1 and did[8] == 1
    assert did[3] == -1 and did[9] == -1
    assert km.doc_span_lens().tolist() == [2, 3]


def test_key_class_map_rejects_overlapping_classes():
    with pytest.raises(ValueError, match="overlapping"):
        A.KeyClassMap(4, [(4, 8)], (6, 10), (10, 12), 12)


def test_key_class_map_normalises_overlapping_gist_spans():
    # _gist_spans_from_doc_lengths can hand back end_i == start_{i+1} + 1
    km = A.KeyClassMap(2, [(2, 5), (4, 7)], None, (7, 9), 9)
    assert km.n_span_adjustments == 1
    assert km.doc_spans == [(2, 5), (5, 7)]
    did = km.doc_ids(9)
    assert (did[2:5] == 0).all() and (did[5:7] == 1).all()


def test_gist_spans_mirror_matches_the_harness_when_torch_is_available():
    spans = A.gist_spans_from_doc_lengths([10, 20, 5], 8)
    assert spans[0][0] == 0 and spans[-1][1] <= 8
    try:
        sys.path.insert(0, str(ROOT / "python"))
        from eval_agent_history_c2kv import _gist_spans_from_doc_lengths  # type: ignore
    except Exception:  # torch not installed on the analysis box
        pytest.skip("eval_agent_history_c2kv needs torch")
    for lens, g in (([10, 20, 5], 8), ([1], 4), ([3, 3, 3, 3], 5), ([7, 2], 0)):
        assert A.gist_spans_from_doc_lengths(lens, g) == _gist_spans_from_doc_lengths(lens, g)


def test_from_prefix_builds_gist_spans_on_c2kv_and_raw_spans_on_full():
    km_c = A.KeyClassMap.from_prefix(arm="c2kv", system_length=5, doc_lengths=[10, 10],
                                     gist_tokens=4, query_len=3)
    assert km_c.n_docs == 2
    assert km_c.doc_spans[0][0] == 5 and km_c.doc_spans[-1][1] == 9
    assert km_c.query_span == (9, 12) and km_c.generated_start == 12

    km_f = A.KeyClassMap.from_prefix(arm="full", system_length=5, doc_lengths=[10, 10],
                                     query_len=3)
    assert km_f.doc_spans == [(5, 15), (15, 25)]
    assert km_f.query_span == (25, 28)


# ---------------------------------------------------------------------------
# the reduction: streaming == dense
# ---------------------------------------------------------------------------

def _dense_reference(q, k, km, mask):
    """Independent dense implementation used only by the tests."""
    q = np.asarray(q, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    n_h, n_q, dim = q.shape
    n_kv, n_k, _ = k.shape
    n_rep = n_h // n_kv
    probs = np.zeros((n_h, n_q, n_k), dtype=np.float32)
    scale = np.float32(dim ** -0.5)
    for h in range(n_h):
        kk = k[h // n_rep]
        logits = (q[h] @ kk.T).astype(np.float32) * scale
        if mask is not None:
            logits = logits + mask.astype(np.float32)
        for i in range(n_q):
            row = logits[i]
            fin = np.isfinite(row)
            if not fin.any():
                continue
            m = row[fin].max()
            e = np.where(fin, np.exp(row - m), 0.0)
            probs[h, i] = (e / e.sum()).astype(np.float32)
    cid, did = km.class_ids(n_k), km.doc_ids(n_k)
    class_mass = np.zeros((n_h, n_q, 5), dtype=np.float32)
    doc_mass = np.zeros((n_h, n_q, km.n_docs), dtype=np.float32)
    for c in range(5):
        class_mass[:, :, c] = probs[:, :, cid == c].sum(axis=2)
    for d in range(km.n_docs):
        doc_mass[:, :, d] = probs[:, :, did == d].sum(axis=2)
    return probs, class_mass, doc_mass


def test_streaming_reduction_equals_dense_reference_with_gqa_and_causal_mask():
    rng = np.random.default_rng(7)
    n_h, n_kv, dim = 6, 3, 8
    km = simple_map(system_len=3, doc_spans=((3, 6), (6, 10)), tail=(10, 12),
                    query=(12, 16), gen=16)
    n_k = 20
    n_q = 4
    q = rng.normal(size=(n_h, n_q, dim)).astype(np.float32)
    k = rng.normal(size=(n_kv, n_k, dim)).astype(np.float32)
    # causal: query row i sits at absolute position n_k - n_q + i
    mask = np.zeros((n_q, n_k), dtype=np.float32)
    for i in range(n_q):
        mask[i, (n_k - n_q + i) + 1:] = -np.inf

    got = A.reduce_attention_rows(q, k, km, mask, return_probs=True)
    probs, class_mass, doc_mass = _dense_reference(q, k, km, mask)

    assert np.allclose(got["probs"], probs, atol=1e-6)
    assert np.allclose(got["class_mass"], class_mass, atol=1e-6)
    assert np.allclose(got["doc_mass"], doc_mass, atol=1e-6)
    # probabilities are a partition of unity over the classified keys
    assert np.allclose(got["class_mass"].sum(axis=-1), 1.0, atol=1e-5)
    assert (got["argmax_key"] == probs.argmax(axis=2)).all()
    assert np.allclose(got["sink_mass"], probs[:, :, 0], atol=1e-7)


def test_chunked_reduction_is_bit_identical_to_unchunked():
    rng = np.random.default_rng(11)
    km = simple_map(system_len=2, doc_spans=((2, 5), (5, 7)), tail=None,
                    query=(7, 10), gen=10)
    q = rng.normal(size=(4, 7, 6)).astype(np.float32)
    k = rng.normal(size=(2, 14, 6)).astype(np.float32)
    base = A.reduce_attention_rows(q, k, km)
    for chunk in (1, 2, 3, 7, 100):
        got = A.reduce_attention_rows(q, k, km, query_chunk=chunk)
        for field in ("class_mass", "doc_mass", "argmax_key", "argmax_val", "sink_mass"):
            assert np.array_equal(got[field], base[field]), f"{field} @ chunk={chunk}"
        assert np.allclose(got["doc_entropy"], base["doc_entropy"], equal_nan=True)


def test_reduce_rejects_bad_gqa_and_head_dim():
    km = simple_map()
    with pytest.raises(ValueError, match="GQA"):
        A.reduce_attention_rows(np.zeros((5, 1, 4)), np.zeros((2, 3, 4)), km)
    with pytest.raises(ValueError, match="head dim"):
        A.reduce_attention_rows(np.zeros((4, 1, 4)), np.zeros((2, 3, 5)), km)


def test_doc_entropy_is_nan_when_no_doc_mass():
    km = A.KeyClassMap(2, [(2, 3)], None, (3, 4), 4)
    probs = np.zeros((1, 1, 4), dtype=np.float32)
    probs[0, 0, 0] = 1.0  # all mass on the system prefix
    out = A.reduce_probs(probs, km.class_ids(4), km.doc_ids(4), km.n_docs)
    assert np.isnan(out["doc_entropy"][0, 0])
    assert out["sink_mass"][0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# (B) Lookback Lens
# ---------------------------------------------------------------------------

def test_lookback_ratio_is_length_invariant():
    """2407.07071: each sum is normalised by its OWN key count, so scaling one
    class's key count with a constant per-key alpha leaves LR unchanged."""
    alpha = 0.01
    out = []
    for n_ctx in (10, 40, 200):
        n_gen = 5
        counts = np.zeros((1, 5), dtype=np.int64)
        counts[0, A.CLASS_INDEX["history_gist"]] = n_ctx
        counts[0, A.CLASS_INDEX["generated_so_far"]] = n_gen
        cm = np.zeros((1, 1, 1, 5), dtype=np.float32)
        cm[0, 0, 0, A.CLASS_INDEX["history_gist"]] = alpha * n_ctx
        cm[0, 0, 0, A.CLASS_INDEX["generated_so_far"]] = alpha * n_gen
        lr = A.lookback_ratios(cm, counts)
        out.append(float(lr[0, 0, 0, A.LOOKBACK_RATIOS.index("lr_gist")]))
    assert out == pytest.approx([0.5, 0.5, 0.5], abs=1e-6)


def test_lr_context_pools_the_four_prompt_classes_before_dividing():
    """2407.07071 eq. (1): A(context) = (1/N) * sum over the WHOLE context.

    Our context is the union of four prompt classes, so their mass is pooled and
    divided ONCE by the pooled key count.  Under a uniform attention distribution
    the ratio must be exactly 0.5 no matter how the context is split into classes,
    and no matter how many of those classes are non-empty.  Summing the four
    classes' individual per-key means would give 0.80 with a raw tail and 0.75
    without one - a value driven by class composition, not by attention.
    """
    alpha = 0.002

    def lr_ctx(counts_vec):
        counts = np.asarray([counts_vec], dtype=np.int64)
        cm = np.zeros((1, 1, 1, 5), dtype=np.float32)
        cm[0, 0, 0, :] = alpha * counts[0]
        lr = A.lookback_ratios(cm, counts)
        return float(lr[0, 0, 0, A.LOOKBACK_RATIOS.index("lr_context")])

    # system / gist / raw_tail / query / generated
    assert lr_ctx([100, 32, 20, 8, 40]) == pytest.approx(0.5, abs=1e-6)
    assert lr_ctx([100, 32, 0, 8, 40]) == pytest.approx(0.5, abs=1e-6)   # no hybrid tail
    assert lr_ctx([400, 32, 20, 8, 5]) == pytest.approx(0.5, abs=1e-6)   # long prefix
    # and it is still a MEAN-per-key ratio, not a mass ratio
    counts = np.array([[0, 100, 0, 0, 10]], dtype=np.int64)
    cm = np.zeros((1, 1, 1, 5), dtype=np.float32)
    cm[0, 0, 0, A.CLASS_INDEX["history_gist"]] = 0.9
    cm[0, 0, 0, A.CLASS_INDEX["generated_so_far"]] = 0.1
    lr = A.lookback_ratios(cm, counts)
    assert float(lr[0, 0, 0, A.LOOKBACK_RATIOS.index("lr_context")]) == pytest.approx(
        0.009 / 0.019, abs=1e-6)


def test_lookback_ratio_moves_with_the_per_key_mean_not_the_mass():
    counts = np.array([[0, 100, 0, 0, 10]], dtype=np.int64)
    cm = np.zeros((1, 1, 1, 5), dtype=np.float32)
    cm[0, 0, 0, A.CLASS_INDEX["history_gist"]] = 0.9
    cm[0, 0, 0, A.CLASS_INDEX["generated_so_far"]] = 0.1
    lr = A.lookback_ratios(cm, counts)
    # per-key: 0.009 vs 0.01 -> LR just under 0.5 even though the raw mass is 9:1
    assert float(lr[0, 0, 0, 1]) == pytest.approx(0.009 / 0.019, abs=1e-6)


def test_lookback_undefined_ratios_are_nan_not_zero():
    counts = np.array([[5, 10, 0, 3, 0]], dtype=np.int64)  # no generated keys, no raw tail
    cm = np.zeros((1, 1, 1, 5), dtype=np.float32)
    cm[0, 0, 0, :4] = 0.25
    lr = A.lookback_ratios(cm, counts)
    assert np.isnan(lr[0, 0, 0, A.LOOKBACK_RATIOS.index("lr_context")])
    assert np.isnan(lr[0, 0, 0, A.LOOKBACK_RATIOS.index("lr_gist")])
    assert np.isnan(lr[0, 0, 0, A.LOOKBACK_RATIOS.index("lr_gist_vs_raw")])


def test_lookback_never_folds_heads():
    counts = np.array([[0, 10, 0, 0, 10]], dtype=np.int64)
    cm = np.zeros((2, 3, 1, 5), dtype=np.float32)
    cm[..., A.CLASS_INDEX["history_gist"]] = 0.5
    cm[..., A.CLASS_INDEX["generated_so_far"]] = 0.5
    cm[1, 2, 0, A.CLASS_INDEX["history_gist"]] = 0.9
    cm[1, 2, 0, A.CLASS_INDEX["generated_so_far"]] = 0.1
    lr = A.lookback_ratios(cm, counts)
    assert lr.shape == (2, 3, 1, 3)
    assert lr[0, 0, 0, 1] != lr[1, 2, 0, 1]


def test_tool_call_span_rows_and_span_mean():
    meta = [{"emit_index": None}, {"emit_index": 0}, {"emit_index": 1},
            {"emit_index": 2}, {"emit_index": 3}]
    rows = A.tool_call_span_rows(meta, 1, 2)
    assert rows.tolist() == [2, 3]
    assert A.tool_call_span_rows(meta, None, None).size == 0
    lr = np.arange(2 * 1 * 5 * 3, dtype=np.float32).reshape(2, 1, 5, 3)
    assert A.span_mean_lookback(lr, np.zeros(0, dtype=int)) is None
    got = A.span_mean_lookback(lr, rows)
    assert got.shape == (2, 1, 3)
    assert np.allclose(got, lr[:, :, rows, :].mean(axis=2))


def test_lookback_feature_matrix_drops_span_less_rows_and_names_columns():
    span = {"s:1": np.zeros((2, 2, 3), dtype=np.float32), "s:2": None}
    X, names, kept = A.lookback_feature_matrix(span, ["s:1", "s:2"])
    assert kept == ["s:1"]
    assert X.shape == (1, 12)
    assert names[0] == "lr__l0__h0__lr_context"
    assert names[-1] == "lr__l1__h1__lr_gist_vs_raw"


def test_drop_undefined_columns_reports_and_never_imputes():
    X = np.array([[1.0, np.nan, 3.0], [2.0, np.nan, 4.0]])
    Xk, names, n = A.drop_undefined_columns(X, ["a", "b", "c"])
    assert n == 1 and names == ["a", "c"] and Xk.shape == (2, 2)


def test_topk_coef_selector_uses_only_training_fold_labels():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 30))
    y = (X[:, 0] + 0.2 * rng.normal(size=40) > 0).astype(int)
    sel = A.make_topk_coef_selector(X, y, k=5)
    tr = np.arange(0, 30)
    te = np.arange(30, 40)
    a, b = sel(X[tr], X[te])
    assert a.shape == (30, 5) and b.shape == (10, 5)
    with pytest.raises(RuntimeError, match="not found"):
        sel(np.zeros((1, 30)), X[te])
    with pytest.raises(ValueError, match="byte-identical"):
        A.make_topk_coef_selector(np.zeros((3, 4)), np.array([0, 1, 0]))


def test_layer_subset_selector_picks_a_depth_band():
    names = [f"lr__l{l}__h0__lr_context" for l in range(6)]
    sel = A.layer_subset_selector(names, 1 / 3, 2 / 3)
    X = np.arange(6 * 6, dtype=float).reshape(6, 6)
    a, _ = sel(X, X)
    assert a.shape[1] == 2  # layers 2 and 3


def test_lookback_probe_runs_and_reports_frame_prevalence_not_the_900_base_rate():
    rng = np.random.default_rng(5)
    n = 60
    X = rng.normal(size=(n, 12))
    groups = np.array([f"s{i // 3}" for i in range(n)])
    y = (X[:, 0] > 0).astype(int)
    s8 = rng.normal(size=(n, 3))
    rep = A.lookback_probe(X, [f"lr__l{i // 2}__h0__lr_context" for i in range(12)],
                           y, groups, s8=s8)
    assert rep["attention"]["n_pos"] > 0
    assert rep["attention"]["prevalence_chance_ap"] == pytest.approx(y.mean(), abs=0.1)
    assert rep["attention"]["prevalence_chance_ap"] != pytest.approx(0.1033, abs=1e-4)
    assert "increment_over_s8" in rep
    assert rep["increment_over_s8"]["n"] == rep["increment_over_s8"]["n"]
    assert set(rep["attention"]["operating_points"]) == {
        "fire_rate_0.10", "fire_rate_0.20", "fire_rate_0.30"}


# ---------------------------------------------------------------------------
# (C) Elastic-Cache drift
# ---------------------------------------------------------------------------

def test_score_doc_vector_mirrors_the_three_harness_modes():
    mass = np.array([0.4, 0.2])
    lens = [4, 1]
    assert np.allclose(A.score_doc_vector(mass, lens, "sum"), [0.4, 0.2])
    assert np.allclose(A.score_doc_vector(mass, lens, "mean"), [0.1, 0.2])
    assert np.allclose(A.score_doc_vector(mass, lens, "sqrt_len"), [0.2, 0.2])
    with pytest.raises(ValueError):
        A.score_doc_vector(mass, lens, "softmax")


def test_doc_vector_stats_entropy_and_margin():
    st = A.doc_vector_stats(np.array([0.5, 0.5]))
    assert st["entropy"] == pytest.approx(np.log(2))
    assert st["margin"] == pytest.approx(0.0)
    assert A.doc_vector_stats(np.zeros(3))["entropy"] is None


def test_predecessor_map_uses_step_index_not_insertion_order():
    qids = ["a:10", "a:2", "a:1", "b:5"]
    pm = A.predecessor_map(qids)
    assert pm["a:1"] is None
    assert pm["a:2"] == "a:1"
    assert pm["a:10"] == "a:2"
    assert pm["b:5"] is None
    assert all(session_of(q) in ("a", "b") for q in qids)


def test_cross_step_join_denominator_is_counted_separately():
    qids = ["a:1", "a:2", "a:3", "b:1", "c:7"]
    y = [1, 0, 1, 1, 0]
    den = A.predecessor_denominator(qids, y)
    assert den == {"n_rows": 5, "n_with_predecessor": 2,
                   "n_pos_rows": 3, "n_pos_with_predecessor": 1}


def test_drift_features_none_without_predecessor_and_never_zero():
    d = A.drift_features(np.array([1.0, 0.0]), None)
    assert d["attn_cos_prev_step"] is None and d["attn_drop_maxdoc"] is None
    assert d["n_aligned_docs"] == 0


def test_drift_features_cosine_and_drop():
    prev = np.array([1.0, 0.0, 0.0])
    cur = np.array([0.25, 0.75, 0.0])
    d = A.drift_features(cur, prev)
    assert d["align_policy"] == "block_index"
    assert d["attn_cos_prev_step"] == pytest.approx(0.25 / np.linalg.norm(cur))
    assert d["attn_drop_maxdoc"] == pytest.approx(0.75)


def test_drift_features_align_on_doc_sha_when_available():
    prev = np.array([9.0, 1.0])            # blocks A, B
    cur = np.array([0.0, 3.0, 4.0])        # blocks X, A, B  (history was prepended)
    d = A.drift_features(cur, prev, prev_shas=["A", "B"], cur_shas=["X", "A", "B"])
    assert d["align_policy"] == "sha256"
    assert d["n_aligned_docs"] == 2
    # aligned pair is prev=[9,1] vs cur=[3,4]; top prev doc is A -> 1 - 3/9
    assert d["attn_drop_maxdoc"] == pytest.approx(1 - 3 / 9)

    # both steps hashed but sharing no block: unalignable, reported as such and
    # NOT silently re-joined by block index (which would compare unrelated docs)
    d2 = A.drift_features(cur, prev, prev_shas=["A", "B"], cur_shas=["P", "Q", "R"])
    assert d2["align_policy"] == "sha256_no_overlap"
    assert d2["n_aligned_docs"] == 0
    assert d2["attn_cos_prev_step"] is None and d2["attn_drop_maxdoc"] is None


def test_drift_feature_matrix_exposes_band_and_mode_as_inner_fold_selectors(tmp_path):
    arrays, metas = _write_capture(tmp_path, "c2kv", ["s1:1", "s1:2", "s1:3", "s2:1"])
    qids = ["s1:1", "s1:2", "s1:3", "s2:1"]
    X, names, kept, groups = A.drift_feature_matrix(arrays, metas, qids)
    assert kept == qids
    # 4 statistics x 3 modes x 4 bands, plus one mode-independent gist_frac per band
    assert X.shape == (4, len(A.LAYER_BANDS) * (len(A.SCORE_MODES) * 4 + 1))
    assert names[0] == "entropy__all__sum"
    assert set(groups) == {"all", "band_all", "band_early", "band_middle", "band_late",
                           "mode_sum", "mode_sqrt_len", "mode_mean"}
    assert len(groups["band_all"]) == len(A.SCORE_MODES) * 4 + 1
    # attn_gist_frac is the first of digest 4.5's five Elastic-Cache features
    assert "gist_frac__all" in names
    assert np.isfinite(X[:, names.index("gist_frac__all")]).all()
    # cross-step columns are nan exactly for the rows without a predecessor
    cos = names.index("cos_prev_step__all__sum")
    assert np.isnan(X[qids.index("s1:1"), cos])
    assert np.isnan(X[qids.index("s2:1"), cos])
    assert np.isfinite(X[qids.index("s1:2"), cos])
    sels = A.column_group_selectors(groups)
    a, b = sels["mode_mean"](X, X)
    assert a.shape[1] == len(A.LAYER_BANDS) * 4

    # the pre-registered sha256 alignment policy reaches the drift block too, not
    # only extract_features: with shas that do not overlap there is nothing to
    # align, so the cross-step columns fall back to nan rather than to 0.0
    shas = {q: [f"{q}-a", f"{q}-b"] for q in qids}
    X2, names2, _, _ = A.drift_feature_matrix(arrays, metas, qids, shas)
    assert names2 == names
    assert np.isnan(X2[qids.index("s1:2"), cos])


def test_cap_control_block_is_built_and_undefined_columns_are_dropped(tmp_path):
    """digest 4.5 pre-registers a cap-flip control (generated_tokens,
    finish_reason == 'length') for the Elastic-Cache block."""
    arrays, metas = _write_capture(tmp_path, "c2kv", ["s1:1", "s1:2"])
    Xc, names, n_dropped = A.cap_control_matrix(metas, ["s1:1", "s1:2"])
    assert names == ["ctl_generated_tokens", "ctl_cap_hit"] and n_dropped == 0
    assert Xc.shape == (2, 2)
    assert (Xc[:, 0] == 40).all()          # generated_tokens
    assert (Xc[:, 1] == 1.0).all()         # finish_reason == "length"

    # a row with no finish_reason makes the column undefined -> dropped, counted,
    # never imputed to 0
    metas["s1:2"]["row"]["finish_reason"] = None
    Xc2, names2, n_dropped2 = A.cap_control_matrix(metas, ["s1:1", "s1:2"])
    assert names2 == ["ctl_generated_tokens"] and n_dropped2 == 1
    assert Xc2.shape == (2, 1)


def test_increment_over_control_uses_one_shared_row_subset():
    rng = np.random.default_rng(3)
    n = 60
    y = (rng.random(n) < 0.4).astype(int)
    g = np.array([f"s{i // 3}" for i in range(n)])
    Xc = np.column_stack([rng.normal(size=n), rng.normal(size=n)])
    X = np.column_stack([y + rng.normal(scale=0.6, size=n), rng.normal(size=n)])
    rep = A.increment_over_control(X, Xc, y, g)
    assert rep is not None
    assert rep["n_rows_shared"] == n and len(rep["ci95"]) == 2

    # a nan anywhere shrinks BOTH arms to the same shared subset
    X2 = X.copy()
    X2[:5, 0] = np.nan
    rep2 = A.increment_over_control(X2, Xc, y, g)
    assert rep2["n_rows_shared"] == n - 5
    # too few shared rows -> None, never a 0.0 that reads like "no increment"
    Xc3 = Xc.copy()
    Xc3[5:, 0] = np.nan
    assert A.increment_over_control(X, Xc3, y, g) is None


def test_flip_table_truth_abstains_on_a_multi_flip_qid():
    assert A.flip_table_truth({0: False, 3: True}) == (3, "unique")
    assert A.flip_table_truth({0: True, 3: True}) == (None, "multi")
    assert A.flip_table_truth({0: False}) == (None, "none")
    assert A.flip_table_truth({}) == (None, "none")
    # keys arriving out of order must not change the verdict
    assert A.flip_table_truth({5: True, 1: False}) == (5, "unique")


def test_band_gist_frac_is_nan_not_zero_without_class_mass():
    assert np.isnan(A.band_gist_frac(None, 0.0, 1.0))
    cm = np.zeros((4, 2, 5), dtype=np.float32)
    cm[..., A.CLASS_INDEX["history_gist"]] = 0.25
    cm[..., A.CLASS_INDEX["system_raw"]] = 0.75
    assert A.band_gist_frac(cm, 0.0, 1.0) == pytest.approx(0.25)


def test_bf16_noise_floor_check():
    a = {"q1": np.array([0.5, 0.25]), "q2": np.array([1.0])}
    b = {"q1": np.array([0.5, 0.25 + 0.001]), "q2": np.array([1.0])}
    rep = A.bf16_noise_floor_check(a, b)
    assert rep["n_compared"] == 2 and rep["within_floor"] is True
    assert rep["floor"] == A.BF16_NOISE_FLOOR == 0.0078125
    b["q2"] = np.array([1.0 + 0.02])
    rep = A.bf16_noise_floor_check(a, b)
    assert rep["within_floor"] is False and rep["n_above_floor"] == 1
    assert rep["worst_qid"] == "q2"


# ---------------------------------------------------------------------------
# (D) Retrieval-Head ports 1 / 2
# ---------------------------------------------------------------------------

def _head_set_file(tmp_path, heads):
    p = tmp_path / "heads.json"
    p.write_text(json.dumps({"model": "Qwen3-4B", "checkpoint": "fixed_joint",
                             "threshold": 0.1, "n_heads_total": 8,
                             "heads": heads}), encoding="utf-8")
    return p


def test_load_head_set_and_mask(tmp_path):
    hs = A.load_head_set(_head_set_file(tmp_path, [[0, 1], [1, 0]]))
    assert hs["heads"] == [(0, 1), (1, 0)]
    assert hs["meta"]["threshold"] == 0.1
    m = A.head_mask(2, 2, hs["heads"])
    assert m.tolist() == [[False, True], [True, False]]
    with pytest.raises(ValueError, match="does not intersect"):
        A.head_mask(1, 1, [(9, 9)])
    bad = tmp_path / "empty.json"
    bad.write_text(json.dumps({"heads": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="empty head set"):
        A.load_head_set(bad)


def test_retrieval_head_doc_scores_restricts_to_the_head_subset():
    doc_mass = np.zeros((2, 2, 3), dtype=np.float32)
    doc_mass[0, 0] = [0.1, 0.8, 0.1]     # not selected
    doc_mass[1, 1] = [0.7, 0.1, 0.2]     # selected
    mask = np.array([[False, False], [False, True]])
    v = A.retrieval_head_doc_scores(doc_mass, [1, 1, 1], mask, mode="sum")
    assert int(np.argmax(v)) == 0
    assert np.allclose(v, [0.7, 0.1, 0.2])


def test_retrieval_head_trigger_scalars():
    cm = np.zeros((1, 2, 5), dtype=np.float32)
    cm[0, 0] = [0.5, 0.3, 0.1, 0.05, 0.05]
    cm[0, 1] = [0.0, 0.9, 0.1, 0.0, 0.0]
    sink = np.array([[0.4, 0.01]], dtype=np.float32)
    out = A.retrieval_head_trigger_scalars(cm, sink, np.array([[True, False]]))
    assert out["attn_rh_sink_ratio"] == pytest.approx(0.4)
    assert out["attn_rh_sink_ratio_system"] == pytest.approx(0.5)
    assert out["attn_rh_gist_ratio"] == pytest.approx(0.3 / 0.4)


def test_locator_report_scores_against_the_floor_with_the_inverted_control():
    # three qids; argmax picks the truth in two of them
    vectors = {"s:1": [0.1, 0.9], "s:2": [0.7, 0.2], "s:3": [0.9, 0.1]}
    truth = {"s:1": 1, "s:2": 0, "s:3": 1}
    n_docs = {q: 2 for q in vectors}
    witness = {"s:1": {"k_median": 1}, "s:2": {"k_median": 0}, "s:3": {"k_median": 0}}
    rep = A.retrieval_head_locator_report(vectors, truth, n_docs, witness)
    assert rep["locator"]["hits"] == 2 and rep["locator"]["n"] == 3
    assert rep["locator"]["floor"] == 0.25
    assert rep["locator"]["witness_oracle"] == pytest.approx(71 / 93)
    assert rep["locator"]["bestk_ceiling"] == pytest.approx(81 / 93)
    assert rep["inverted_control"]["inverted"]["hits"] == 1
    assert set(rep["legacy"]) == {"k_first", "k_median", "k_last"}
    assert rep["legacy"]["k_first"]["table"]["hits"] == 1     # only s:2 has truth 0
    assert rep["legacy"]["k_last"]["table"]["hits"] == 2
    assert 0.0 <= rep["legacy"]["k_median"]["mcnemar_p"] <= 1.0


def test_legacy_locator_hits_abstains_when_truth_is_unknown():
    hits = A.legacy_locator_hits({"a:1": None}, {"a:1": 3}, {"a:1": {"k_median": 1}})
    assert hits["k_first"]["a:1"] is None
    assert hits["k_last"]["a:1"] is None
    assert hits["k_median"]["a:1"] is None


# ---------------------------------------------------------------------------
# (E) capture record schema + feature extraction (synthetic rows)
# ---------------------------------------------------------------------------

class _StubCapture:
    """Minimal stand-in for AttentionRowCapture that finalize_row can consume."""

    def __init__(self, cm, dm, sk, en, meta):
        self._cm, self._dm, self._sk, self._en, self._meta = cm, dm, sk, en, meta
        self.gist_path_forwards = 0
        self.errors = {}

    def class_mass_tensor(self):
        return self._cm

    def doc_mass_tensor(self):
        return self._dm

    def sink_tensor(self):
        return self._sk

    def entropy_tensor(self):
        return self._en

    @property
    def row_meta(self):
        return self._meta


def _fake_row(km, n_layers=2, n_heads=2, n_rows=6, seed=0):
    rng = np.random.default_rng(seed)
    n_docs = km.n_docs
    cm = rng.random((n_layers, n_heads, n_rows, 5)).astype(np.float32)
    cm /= cm.sum(axis=-1, keepdims=True)
    dm = rng.random((n_layers, n_heads, n_rows, n_docs)).astype(np.float32) * 0.3
    sk = rng.random((n_layers, n_heads, n_rows)).astype(np.float32) * 0.1
    en = rng.random((n_layers, n_heads, n_rows)).astype(np.float32)
    meta = [{"forward_id": i, "qrow": 0, "emit_index": i, "n_keys": 30 + i, "kind": "decode"}
            for i in range(n_rows)]
    return _StubCapture(cm, dm, sk, en, meta)


GEN_TEXT = 'Action:\n<tool_call>\n{"name":"x__y","arguments":{"a":"b"}}\n</tool_call>'


def _decode_fn(ids):
    # a toy tokenizer: one character per id, ids index into GEN_TEXT
    return "".join(GEN_TEXT[i] for i in ids)


def test_finalize_row_produces_the_documented_on_disk_record(tmp_path):
    km = A.KeyClassMap.from_prefix(arm="c2kv", system_length=4, doc_lengths=[8, 8],
                                   gist_tokens=4, raw_tail_tokens=2, query_len=3)
    cap = _fake_row(km, n_rows=len(GEN_TEXT))
    gen_ids = list(range(len(GEN_TEXT)))
    arrays, meta = A.finalize_row(
        cap, km, qid="sess:3", arm="c2kv", generated_ids=gen_ids, decode_fn=_decode_fn,
        prefix_meta={"system_length": 4, "gist_tokens": 4, "kept_history_tokens": 16,
                     "actual_compression_ratio": 4.0, "doc_chunks": 2, "doc_tokens": 16,
                     "history_length": 6},
        row_meta={"generated_tokens": len(GEN_TEXT), "finish_reason": "stop",
                  "session_id": "sess", "dropped_docs": [1]})
    assert set(arrays) <= set(A.NPZ_FIELDS)
    # the recompute-vs-eager self-check is reported on every meta line, as null
    # when it could not run (here: a stub with no attn_weights), never as 0.0
    assert meta["recompute_max_abs_diff"] is None and meta["recompute_n_checked"] == 0
    assert arrays["lookback"].shape == (2, 2, 3)
    assert arrays["class_mass"].shape == (2, 2, 5)
    assert arrays["doc_mass"].shape == (2, 2, 2)
    assert meta["has_tool_call"] and meta["parse_ok"]
    assert meta["n_span_rows"] > 0
    assert meta["n_docs"] == 2 and meta["n_layers"] == 2 and meta["n_heads"] == 2
    assert meta["prefix"]["gist_tokens"] == 4

    w = A.AttentionCaptureWriter(tmp_path, "c2kv")
    w.add("sess:3", arrays, meta)
    w.close()
    arrs, metas = A.load_capture(tmp_path / "attn_c2kv.npz", tmp_path / "attn_c2kv.jsonl")
    assert set(metas) == {"sess:3"}
    assert np.allclose(arrs["sess:3"]["doc_mass"], arrays["doc_mass"])


def test_finalize_row_without_a_tool_call_span_writes_nothing_rather_than_zero():
    km = A.KeyClassMap.from_prefix(arm="c2kv", system_length=2, doc_lengths=[4],
                                   gist_tokens=2, query_len=2)
    cap = _fake_row(km, n_rows=4)
    arrays, meta = A.finalize_row(
        cap, km, qid="s:1", arm="c2kv", generated_ids=[0, 1, 2, 3],
        decode_fn=lambda ids: "no call here", prefix_meta={}, row_meta=None)
    assert meta["n_span_rows"] == 0
    assert arrays == {}, "an empty span must yield no arrays, not zeros"


def _write_capture(tmp_path, arm, qids, seed=0):
    w = A.AttentionCaptureWriter(tmp_path, arm)
    km = A.KeyClassMap.from_prefix(arm="c2kv", system_length=4, doc_lengths=[8, 8],
                                   gist_tokens=4, raw_tail_tokens=2, query_len=3)
    for i, qid in enumerate(qids):
        cap = _fake_row(km, n_rows=len(GEN_TEXT), seed=seed + i)
        arrays, meta = A.finalize_row(
            cap, km, qid=qid, arm=arm, generated_ids=list(range(len(GEN_TEXT))),
            decode_fn=_decode_fn,
            prefix_meta={"system_length": 4, "gist_tokens": 4, "kept_history_tokens": 16,
                         "actual_compression_ratio": 4.0, "doc_chunks": 2, "doc_tokens": 16},
            row_meta={"generated_tokens": 40, "finish_reason": "length",
                      "session_id": qid.split(":")[0], "dropped_docs": []})
        w.add(qid, arrays, meta)
    w.close()
    return A.load_capture(tmp_path / f"attn_{arm}.npz", tmp_path / f"attn_{arm}.jsonl")


def test_extract_features_scalars_missingness_and_cross_step_join(tmp_path):
    arrays, metas = _write_capture(tmp_path, "c2kv", ["s1:1", "s1:2", "s2:4"])
    rows = A.extract_features(arrays, metas, arm="c2kv")
    by = {r["qid"]: r for r in rows}
    assert set(by) == {"s1:1", "s1:2", "s2:4"}
    # cross-step features exist only for rows with a predecessor IN THE FRAME
    assert by["s1:1"]["attn_cos_prev_step_sum"] is None
    assert by["s1:2"]["attn_cos_prev_step_sum"] is not None
    assert by["s2:4"]["attn_drop_maxdoc_mean"] is None
    assert by["s1:1"]["has_prev_step"] == 0 and by["s1:2"]["has_prev_step"] == 1
    # cap-flip controls
    assert by["s1:1"]["cap_hit"] == 1 and by["s1:1"]["generated_tokens"] == 40
    # S8 control block present
    for c in A.S8_CONTROL_COLUMNS:
        assert c in by["s1:1"]
    # retrieval-head scalars are None without a head set (never a sentinel)
    assert by["s1:1"]["attn_rh_sink_ratio"] is None
    assert A.predecessor_denominator(list(by)) == {"n_rows": 3, "n_with_predecessor": 1}


def test_extract_features_full_arm_nulls_the_gist_specific_columns(tmp_path):
    arrays, metas = _write_capture(tmp_path, "full", ["s1:1"])
    row = A.extract_features(arrays, metas, arm="full")[0]
    assert row["lr_context_mean"] is not None
    assert row["lr_gist_mean"] is None
    assert row["lr_gist_vs_raw_mean"] is None
    assert row["attn_gist_ratio"] is None


def test_features_pass_the_leakage_guard_and_none_survives_as_null(tmp_path):
    from t34_common import write_features_jsonl

    arrays, metas = _write_capture(tmp_path, "c2kv", ["s1:1", "s1:2"])
    rows = A.extract_features(arrays, metas, arm="c2kv")
    out = tmp_path / "features.jsonl"
    assert write_features_jsonl(out, rows, context="test") == 2
    lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["attn_cos_prev_step_sum"] is None
    assert "tool_name_match" not in lines[0] and "target" not in lines[0]

    with pytest.raises(ValueError, match="label-leaking"):
        write_features_jsonl(tmp_path / "bad.jsonl",
                             [dict(rows[0], a_made_call=1)], context="test")


def test_every_written_feature_has_a_declared_orientation(tmp_path):
    arrays, metas = _write_capture(tmp_path, "c2kv", ["s1:1"])
    rows = A.extract_features(arrays, metas, arm="c2kv")
    orient = json.loads((ROOT / "configs/t34/orientations_attn.json").read_text(encoding="utf-8"))
    missing = sorted(k for k in rows[0]
                     if k not in ("qid", "arm", "session_id") and k not in orient)
    assert missing == [], f"undeclared orientation for {missing}"
    for k, v in orient.items():
        if not k.startswith("_"):
            assert v in (1, -1), f"{k} orientation must be +1 or -1"


# ---------------------------------------------------------------------------
# AttentionRowCapture wiring (fake model here; the real patch is torch-skipped)
# ---------------------------------------------------------------------------

class _StubConfig:
    _attn_implementation = "eager"


class _StubAttn:
    def __init__(self, idx):
        self.layer_idx = idx
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        return ("out", None)

    def forward_with_gist(self, **kwargs):
        self.calls += 1
        return ("out", ("gk", "gv"))


class _StubLayer:
    def __init__(self, idx):
        self.self_attn = _StubAttn(idx)


class _StubModel:
    def __init__(self, n=3, impl="eager"):
        cfg = _StubConfig()
        cfg._attn_implementation = impl
        self.config = cfg

        class _Inner:
            pass

        self.model = _Inner()
        self.model.layers = [_StubLayer(i) for i in range(n)]


def test_install_and_remove_restore_the_original_callables():
    km = simple_map()
    model = _StubModel()
    originals = [l.self_attn.forward for l in model.model.layers]
    cap = A.AttentionRowCapture(km)
    cap.install(model)
    assert all("forward" in l.self_attn.__dict__ for l in model.model.layers)
    assert all(l.self_attn.forward is not o for l, o in zip(model.model.layers, originals))
    cap.remove()
    assert all("forward" not in l.self_attn.__dict__ for l in model.model.layers)
    assert all(l.self_attn.forward.__func__ is o.__func__
               for l, o in zip(model.model.layers, originals))


def test_installing_twice_is_refused():
    cap = A.AttentionRowCapture(simple_map())
    cap.install(_StubModel())
    with pytest.raises(RuntimeError, match="already installed"):
        cap.install(_StubModel())
    cap.remove()


def test_install_rejects_non_eager_attention_with_a_clear_message():
    cap = A.AttentionRowCapture(simple_map())
    with pytest.raises(RuntimeError, match="eager"):
        cap.install(_StubModel(impl="npu_fusion_attention"))


def test_gist_path_is_counted_so_a_bypassed_capture_is_loud():
    km = simple_map()
    model = _StubModel(n=2)
    cap = A.AttentionRowCapture(km)
    cap.install(model)
    for layer in model.model.layers:
        layer.self_attn.forward_with_gist(hidden_states=None)
    cap.remove()
    assert cap.gist_path_forwards == 2
    assert all(l.self_attn.calls == 1 for l in model.model.layers)


def test_recompute_self_check_counters_start_empty_and_reset():
    """The recomputed probabilities are compared against eager's own returned
    attn_weights (the deviation that makes the whole capture interpretable).
    The counters must start as None/0 - never as a 0.0 that reads like 'checked
    and identical' - and reset() must clear them between battery rows."""
    cap = A.AttentionRowCapture(simple_map())
    assert cap.recompute_max_abs_diff is None and cap.recompute_n_checked == 0
    assert cap.recompute_tol == pytest.approx(1e-2)
    assert cap.recompute_max_abs_diff_fp32 is None
    cap.recompute_max_abs_diff, cap.recompute_n_checked = 0.5, 7
    cap.reset()
    assert cap.recompute_max_abs_diff is None and cap.recompute_n_checked == 0


def test_forward_wrapper_never_kills_the_row_and_records_the_error():
    km = simple_map()
    model = _StubModel(n=2)
    cap = A.AttentionRowCapture(km)
    cap.install(model)
    for layer in model.model.layers:
        # hidden_states=None makes _capture raise; the original output must survive
        assert layer.self_attn.forward(hidden_states=None)[0] == "out"
    cap.remove()
    assert sum(cap.errors.values()) == 2
    assert cap.class_mass_tensor().size == 0


def test_query_mode_row_selection_and_emit_index_bookkeeping():
    cap = A.AttentionRowCapture(simple_map(), query_mode="prefill_last_n", last_n=2)
    assert cap._select_rows(1).tolist() == [0]
    assert cap._select_rows(5).tolist() == [3, 4]
    cap_dec = A.AttentionRowCapture(simple_map(), query_mode="decode")
    assert cap_dec._select_rows(5) is None

    # fake the bookkeeping a real forward would fill in
    cap._forward_rows = {0: [3, 4], 1: [0], 2: [0]}
    cap._forward_nq = {0: 5, 1: 1, 2: 1}
    cap._forward_nkeys = {0: 20, 1: 21, 2: 22}
    meta = cap.row_meta
    assert [m["emit_index"] for m in meta] == [None, 0, 1, 2]
    assert [m["kind"] for m in meta] == ["prefill", "prefill", "decode", "decode"]


def test_forward_id_advances_only_when_a_layer_repeats():
    cap = A.AttentionRowCapture(simple_map())
    assert cap._new_forward(0, 1) == 0
    assert cap._new_forward(1, 1) == 0
    assert cap._new_forward(2, 1) == 0
    assert cap._new_forward(0, 1) == 1   # layer 0 again -> a new forward began
    assert cap._new_forward(1, 1) == 1


def test_records_and_tensor_accessors_agree():
    cap = A.AttentionRowCapture(simple_map())
    n_h, n_docs = 2, 2
    for layer in (0, 1):
        for fid in (0, 1):
            cap._blocks[(layer, fid)] = {
                "class_mass": np.full((n_h, 1, 5), layer + fid, dtype=np.float32),
                "doc_mass": np.full((n_h, 1, n_docs), layer, dtype=np.float32),
                "argmax_key": np.full((n_h, 1), fid, dtype=np.int64),
                "argmax_val": np.full((n_h, 1), 0.5, dtype=np.float32),
                "doc_entropy": np.full((n_h, 1), 0.1, dtype=np.float32),
                "sink_mass": np.full((n_h, 1), 0.2, dtype=np.float32),
            }
            cap._forward_rows[fid] = [0]
            cap._forward_nq[fid] = 1
            cap._forward_nkeys[fid] = 18 + fid
    cm = cap.class_mass_tensor()
    assert cm.shape == (2, n_h, 2, 5)
    assert cm[1, 0, 1, 0] == pytest.approx(2.0)
    assert cap.doc_mass_tensor().shape == (2, n_h, 2, n_docs)
    assert cap.argmax_tensor().tolist() == [[[0, 1]] * n_h] * 2
    recs = cap.records
    assert len(recs) == 2 * n_h * 2
    assert {r["layer"] for r in recs} == {0, 1}
    assert all(set(r) >= {"layer", "head", "forward_id", "qrow", "class_mass",
                          "doc_mass", "argmax_key", "sink_mass"} for r in recs)


def test_deviations_are_declared_for_every_ported_paper():
    papers = {d["paper"] for d in A.DEVIATIONS}
    assert any(p.startswith("2407.07071") for p in papers)
    assert any(p.startswith("2510.14973") for p in papers)
    assert any(p.startswith("2404.15574") for p in papers)
    for d in A.DEVIATIONS:
        assert set(d) == {"method", "paper", "what", "why"}
        assert d["what"] and d["why"]


def test_cli_help_works_without_torch():
    parser = A.build_parser()
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["--help"])
    assert exc.value.code == 0
    args = parser.parse_args(["extract-features", "--capture", "a.npz", "--meta", "a.jsonl",
                              "--arm", "c2kv", "--out", "o.jsonl"])
    assert args.arm == "c2kv" and args.fn is A._cmd_extract_features


# ---------------------------------------------------------------------------
# real-model patch (skipped without torch)
# ---------------------------------------------------------------------------

try:  # torch is NOT installed on the analysis box; it lives on the NPU server
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch lives on the NPU server only")
def test_real_qwen3_attention_patch_round_trips():  # pragma: no cover - server only
    sys.path.insert(0, str(ROOT / "python"))
    from models.qwen3.modeling_qwen3 import Qwen3Attention  # type: ignore

    before_fwd = Qwen3Attention.forward
    before_gist = Qwen3Attention.forward_with_gist

    class _Cfg:
        _attn_implementation = "eager"

    class _M:
        pass

    model = _M()
    model.config = _Cfg()
    model.model = _M()
    mod = object.__new__(Qwen3Attention)
    mod.layer_idx = 0
    layer = _M()
    layer.self_attn = mod
    model.model.layers = [layer]

    cap = A.AttentionRowCapture(simple_map())
    cap.install(model)
    assert mod.forward is not before_fwd
    cap.remove()
    assert Qwen3Attention.forward is before_fwd
    assert Qwen3Attention.forward_with_gist is before_gist


# ---------------------------------------------------------------------------
# fixer pass: one gist-fraction column, sentinels, contract errors, truth source
# ---------------------------------------------------------------------------

def test_gist_fraction_is_one_column_under_one_declared_orientation(tmp_path):
    """attn_gist_frac and attn_history_gist_frac were the SAME number written
    twice, each with its own orientation entry.  Exactly one survives."""
    arrays, metas = _write_capture(tmp_path, "c2kv", ["s1:1"])
    row = A.extract_features(arrays, metas, arm="c2kv")[0]
    assert "attn_history_gist_frac" not in row
    assert row["attn_gist_frac"] is not None

    # it really is the history_gist class share of the TOTAL mass
    cm = arrays["s1:1"]["class_mass"].astype(float)
    tot = np.nansum(cm, axis=-1)
    frac = cm[..., A.CLASS_INDEX["history_gist"]] / tot
    assert row["attn_gist_frac"] == pytest.approx(float(np.nanmean(frac)), rel=1e-6)

    orient = json.loads((ROOT / "configs/t34/orientations_attn.json").read_text(encoding="utf-8"))
    assert "attn_history_gist_frac" not in orient
    assert orient["attn_gist_frac"] == -1
    # the +1 columns are the ones with the gist-vs-raw-tail denominator, which is
    # a different claim, not a contradiction; the rationale is written down
    assert orient["attn_gist_ratio"] == 1 and orient["attn_rh_gist_ratio"] == 1
    assert "attn_gist_frac" in orient["_sources"]
    assert "attn_gist_frac" in set(A.CLASS_FRAC_COLUMNS.values())


def test_argmax_tensor_marks_missing_rows_with_minus_one_not_int64_min():
    """A layer that fails to capture on one forward leaves NaN in the stack;
    casting that to int64 would give INT64_MIN, which reads like a key index."""
    cap = A.AttentionRowCapture(simple_map())
    n_h = 2
    for (layer, fid) in ((0, 0), (0, 1), (1, 0)):   # layer 1 misses forward 1
        cap._blocks[(layer, fid)] = {
            "class_mass": np.zeros((n_h, 1, 5), dtype=np.float32),
            "doc_mass": np.zeros((n_h, 1, 2), dtype=np.float32),
            "argmax_key": np.full((n_h, 1), 7, dtype=np.int64),
            "argmax_val": np.full((n_h, 1), 0.5, dtype=np.float32),
            "doc_entropy": np.zeros((n_h, 1), dtype=np.float32),
            "sink_mass": np.zeros((n_h, 1), dtype=np.float32),
        }
    cap._forward_rows = {0: [0], 1: [0]}
    cap._forward_nq = {0: 1, 1: 1}
    cap._forward_nkeys = {0: 18, 1: 19}
    am = cap.argmax_tensor()
    valid = cap.argmax_valid_mask()
    assert am.dtype == np.int64
    assert am[0].tolist() == [[7, 7]] * n_h
    assert am[1, :, 1].tolist() == [-1] * n_h
    assert int(am.min()) == -1, "no INT64_MIN sentinel may survive the cast"
    assert valid[1, :, 1].tolist() == [False] * n_h
    assert valid[0].all()


class _FakeTensor:
    """Just enough of a tensor for the shape-contract checks (no torch here)."""

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(shape)


def test_capture_refuses_a_batched_forward_before_it_needs_torch():
    cap = A.AttentionRowCapture(simple_map())
    with pytest.raises(A.CaptureContractError, match="batch-size-1"):
        cap._capture(None, 0, _FakeTensor((2, 5, 8)), ("cos", "sin"), None, {})


def test_batched_forward_lands_in_capture_errors_and_leaves_the_row_empty():
    model = _StubModel(n=2)
    cap = A.AttentionRowCapture(simple_map())
    cap.install(model)
    for layer in model.model.layers:
        assert layer.self_attn.forward(hidden_states=_FakeTensor((4, 5, 8)),
                                       position_embeddings=("cos", "sin"),
                                       past_key_values=None)[0] == "out"
    cap.remove()
    assert cap.errors == {"CaptureContractError": 2}
    assert cap.contract_errors and "batch-size-1" in cap.contract_errors[0]
    assert cap.class_mass_tensor().size == 0


def test_check_key_layout_rejects_a_batched_or_missing_cache():
    keys = _FakeTensor((1, 4, 32, 128))
    assert A.check_key_layout(keys) is keys
    with pytest.raises(A.CaptureContractError, match="cache layout"):
        A.check_key_layout(_FakeTensor((2, 4, 32, 128)))
    with pytest.raises(A.CaptureContractError, match="cache layout"):
        A.check_key_layout(_FakeTensor((4, 32, 128)))
    with pytest.raises(RuntimeError, match="cache keys unavailable"):
        A.check_key_layout(None)


def test_install_refuses_a_batched_instrument():
    cap = A.AttentionRowCapture(simple_map(), expected_batch_size=4)
    with pytest.raises(RuntimeError, match="batch-size-1"):
        cap.install(_StubModel())


def test_capture_gist_path_is_refused_rather_than_silently_empty():
    with pytest.raises(NotImplementedError, match="not implemented"):
        A.AttentionRowCapture(simple_map(), capture_gist_path=True)


def test_locator_truth_uses_one_source_and_abstains_instead_of_mixing():
    flips = {
        "s:1": {2: True, 0: False},          # unique
        "s:2": {1: True, 3: True},           # multi -> abstain
        "s:3": {0: False},                   # never flips -> abstain
    }
    witness = {q: {"k_witness": 0} for q in ("s:1", "s:2", "s:3", "s:4")}
    truth, rep = A.locator_truth(["s:4", "s:3", "s:2", "s:1"], flips, witness)
    assert truth == {"s:1": 2, "s:2": None, "s:3": None, "s:4": None}
    assert rep["truth_source"] == "flip_table"
    assert rep["n_multi_flip"] == 1 and rep["n_no_flip"] == 1
    assert rep["n_missing_from_flip_table"] == 1     # s:4 does NOT fall back to k*
    assert rep["n_with_reference"] == 1

    # no flip table at all -> the witness k* is the source for every qid, declared
    truth2, rep2 = A.locator_truth(["s:1", "s:4"], {}, witness)
    assert truth2 == {"s:1": 0, "s:4": 0}
    assert rep2["truth_source"] == "witness_k_star"
    assert rep2["n_missing_from_flip_table"] == 0


def test_locator_uses_the_frozen_chooser_and_agrees_with_its_own_control():
    # "a:1" has no block scoring above zero -> select_k_star abstains; a bare
    # argmax would have answered it and disagreed with the inverted control.
    vectors = {"a:1": [0.0, -1.0], "a:2": [0.2, 0.9], "a:3": [0.8, 0.1]}
    truth = {"a:1": 0, "a:2": 1, "a:3": 1}
    n_docs = {q: 2 for q in vectors}
    witness = {q: {"k_median": 0} for q in vectors}
    rep = A.retrieval_head_locator_report(vectors, truth, n_docs, witness)
    assert rep["locator"]["abstained"] == 1
    assert rep["locator"]["hits"] == 1 and rep["locator"]["n"] == 3
    fwd = rep["inverted_control"]["forward"]
    assert (rep["locator"]["hits"], rep["locator"]["abstained"]) == (fwd["hits"], fwd["abstained"])


def test_finalize_row_reports_the_emit_index_check_on_the_meta_line():
    km = A.KeyClassMap.from_prefix(arm="c2kv", system_length=4, doc_lengths=[8, 8],
                                   gist_tokens=4, raw_tail_tokens=2, query_len=3)
    cap = _fake_row(km, n_rows=len(GEN_TEXT))
    _arrays, meta = A.finalize_row(
        cap, km, qid="s:1", arm="c2kv", generated_ids=list(range(len(GEN_TEXT))),
        decode_fn=_decode_fn, prefix_meta={}, row_meta=None)
    chk = meta["emit_index_check"]
    assert set(chk) >= {"ok", "problems", "n_emitting_rows", "max_emit_index"}
    # the stub's forwards are all 1-token, so the inferred convention does NOT
    # hold and the meta line says so instead of hiding a one-token offset
    assert chk["ok"] is False
    assert any("prefill" in p for p in chk["problems"])

    # a well-formed prefill + decode chain passes
    good = [{"forward_id": 0, "qrow": 3, "emit_index": 0, "n_keys": 20, "kind": "prefill"},
            {"forward_id": 1, "qrow": 0, "emit_index": 1, "n_keys": 21, "kind": "decode"},
            {"forward_id": 2, "qrow": 0, "emit_index": 2, "n_keys": 22, "kind": "decode"}]
    assert A.emit_index_consistency(good, n_generated=3)["ok"] is True


def _sidecar(tmp_path, rows):
    p = tmp_path / "sidecar_c2kv.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def test_dropped_doc_counts_read_dropped_docs_never_a_filter_over_docs(tmp_path):
    # docs holds ONLY the kept blocks; dropped_docs indexes the POST-SPLIT history
    # list, so its entries can exceed len(docs) - counting must not touch docs.
    p = _sidecar(tmp_path, [
        {"qid": "s1:1", "docs": ["a", "b"], "doc_lengths": [3, 4], "dropped_docs": [0, 5, 6]},
        {"qid": "s1:2", "docs": ["a"], "doc_lengths": [3], "dropped_docs": []},
    ])
    counts = A.dropped_doc_counts_from_sidecar(str(p))
    assert counts == {"s1:1": 3, "s1:2": 0}
    assert A.dropped_doc_counts_from_sidecar(None) is None


def test_n_dropped_docs_is_null_without_the_sidecar_and_filled_with_it(tmp_path):
    arrays, metas = _write_capture(tmp_path, "c2kv", ["s1:1", "s1:2"])
    for m in metas.values():             # the battery row carries no dropped_docs
        m["row"].pop("dropped_docs", None)
        m["prefix"].pop("dropped_docs", None)
    rows = A.extract_features(arrays, metas, arm="c2kv")
    assert all(r["n_dropped_docs"] is None for r in rows), "null, never 0"

    p = _sidecar(tmp_path, [{"qid": "s1:1", "docs": ["a"], "doc_lengths": [2],
                             "dropped_docs": [0, 1]}])
    counts = A.dropped_doc_counts_from_sidecar(str(p))
    rows2 = A.extract_features(arrays, metas, arm="c2kv", dropped_counts=counts)
    by = {r["qid"]: r for r in rows2}
    assert by["s1:1"]["n_dropped_docs"] == 2
    assert by["s1:2"]["n_dropped_docs"] is None   # not in the sidecar -> still null


def test_lookback_probe_names_the_undefined_s8_control_column():
    rng = np.random.default_rng(7)
    n = 60
    X = rng.normal(size=(n, 12))
    groups = np.array([f"s{i // 3}" for i in range(n)])
    y = (X[:, 0] > 0).astype(int)
    s8 = rng.normal(size=(n, len(A.S8_CONTROL_COLUMNS)))
    s8[:, list(A.S8_CONTROL_COLUMNS).index("n_dropped_docs")] = np.nan
    rep = A.lookback_probe(X, [f"lr__l{i // 2}__h0__lr_context" for i in range(12)],
                           y, groups, s8=s8, s8_names=A.S8_CONTROL_COLUMNS)
    assert rep["s8_columns_dropped_undefined"] == ["n_dropped_docs"]
    assert set(rep["s8_columns_used"]) == set(A.S8_CONTROL_COLUMNS) - {"n_dropped_docs"}
    assert rep["n_dropped_undefined_s8_columns"] == 1
