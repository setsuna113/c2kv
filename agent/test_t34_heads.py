# -*- coding: utf-8 -*-
"""Tests for t34_heads (U9, digest S4.10).

Everything here runs torch-free: the heads are numpy with explicit gradients,
so the gradient checks are finite-difference, and the capture/hidden loaders
are exercised against synthetic shards written into tmp_path.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import t34_common as C
import t34_heads as H
from t33_labels import build_label_frame, guard_columns

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _frozen():
    assets = C.FrozenAssets(ROOT)
    if not assets.battery_full.exists():
        pytest.skip("frozen r2 battery not on this box")
    return assets.load()


def _synth_frame(n_sessions: int = 24, steps: int = 4, seed: int = 0):
    """A synthetic FrozenFrame with the same shape as the r2 battery."""
    rng = np.random.default_rng(seed)
    full_rows, c2kv_rows = [], []
    for s in range(n_sessions):
        sess = f"sess{s:03d}"
        for k in range(1, steps + 1):
            qid = f"{sess}:{k}"
            fm = bool(rng.random() < 0.75)
            cm = bool(rng.random() < 0.55)
            row_common = {
                "qid": qid, "session_id": sess, "skipped": False,
                "target_has_tool_call": True, "generated_tokens": int(rng.integers(10, 130)),
                "prediction": '<tool_call>{"name": "a", "arguments": {}}</tool_call>',
            }
            full_rows.append({**row_common, "tool_name_match": fm})
            c2kv_rows.append({**row_common, "tool_name_match": cm})
    pairs = [(f, c) for f, c in zip(full_rows, c2kv_rows)]
    manifest = {"kv_recipe": {"max_new_tokens": 128},
                "cw_qids": sorted(f["qid"] for f, c in pairs
                                  if f["tool_name_match"] and not c["tool_name_match"])}
    labels = build_label_frame(pairs, manifest)
    return C.FrozenFrame(pairs=pairs, labels=labels, manifest=manifest, witness=None)


def _synth_inputs(frame, *, dim: int = 6, n_layers: int = 1, pool: int = 4, seed: int = 1):
    rng = np.random.default_rng(seed)
    qids = [r["qid"] for r in frame.labels]
    sessions = [r["session_id"] for r in frame.labels]
    ids = list(range(100, 100 + pool))
    n = len(qids)
    y = np.array([1.0 if frame.label_by_qid.get(q) == 1 else 0.0 for q in qids])
    hid = rng.normal(size=(n, n_layers, dim)).astype(np.float32)
    hid[:, :, 0] += 1.4 * y[:, None]            # one weakly informative direction
    return H.HeadInputs(
        qids=qids, sessions=sessions, hidden=hid, layers=list(range(n_layers)),
        cand_ids=[list(ids) for _ in range(n)],
        entropy_vocab=rng.random(n) + 0.3 * y,
        margin_top1_top2=rng.random(n),
        generated_tokens=rng.integers(10, 130, size=n).astype(float),
        cap_hit=rng.random(n) < 0.3,
        valid=np.ones(n, dtype=bool),
    )


def _tool(name, props, required=()):
    return {"type": "function", "function": {
        "name": name,
        "parameters": {"type": "object", "properties": props, "required": list(required)}}}


# ---------------------------------------------------------------------------
# ALIEN head -- 2505.15443 S3.3 / S3.4
# ---------------------------------------------------------------------------

def test_alien_init_reproduces_entropy_exactly():
    """theta_init = lm_head rows => U_ALIEN == U_Entropy at init (2505.15443 S3.3)."""
    rng = np.random.default_rng(0)
    dim, pool = 8, 5
    ids = list(range(10, 10 + pool))
    lm = {t: rng.normal(size=dim) for t in ids}
    head = H.AlienHead.from_lm_head(ids, lm, dim)
    Hx = rng.normal(size=(7, dim))
    mask = np.ones((7, pool), dtype=bool)
    # the model's own restricted distribution, computed independently of the head
    W = np.stack([lm[t] for t in ids])
    logits = Hx @ W.T
    p = np.exp(logits - logits.max(1, keepdims=True))
    p = p / p.sum(1, keepdims=True)
    u_ref = -(p * np.log(p)).sum(1) / np.log(pool)
    assert head.score(Hx, mask) == pytest.approx(u_ref, abs=1e-12)


def test_alien_normalisation_by_log_c_and_undefined_below_two():
    p = np.array([[0.25, 0.25, 0.25, 0.25], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    mask = np.array([[True] * 4, [True] * 4, [True, False, False, False]])
    u, raw = H.normalised_entropy(p, mask)
    assert u[0] == pytest.approx(1.0)                 # uniform over C -> exactly 1
    assert raw[0] == pytest.approx(np.log(4))
    assert u[1] == pytest.approx(0.0)                 # one-hot -> 0
    assert np.isnan(u[2])                             # |C| = 1: log C = 0 is undefined


def test_alien_masked_softmax_handles_empty_pool_row():
    logits = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[True, True], [False, False]])
    p = H.masked_softmax(logits, mask)
    assert p[0].sum() == pytest.approx(1.0)
    assert p[1].sum() == 0.0 and np.isfinite(p).all()


def test_alien_gradient_matches_finite_differences():
    rng = np.random.default_rng(3)
    dim, pool, n = 5, 4, 9
    ids = list(range(20, 20 + pool))
    lm = {t: rng.normal(size=dim) * 0.4 for t in ids}
    head = H.AlienHead.from_lm_head(ids, lm, dim)
    head.theta += rng.normal(size=head.theta.shape) * 0.1   # move off the anchor
    head.bias += rng.normal(size=head.bias.shape) * 0.1
    Hx = rng.normal(size=(n, dim))
    mask = np.ones((n, pool), dtype=bool)
    e = (rng.random(n) < 0.5).astype(float)
    u_ent = rng.random(n) * 0.6 + 0.2
    alpha, beta = 0.3, 0.05
    loss, g_theta, g_bias, _ = head.loss_and_grads(Hx, mask, e, u_ent, alpha=alpha, beta=beta)

    eps = 1e-6
    for _ in range(8):
        i = int(rng.integers(pool))
        j = int(rng.integers(dim))
        head.theta[i, j] += eps
        lp, *_ = head.loss_and_grads(Hx, mask, e, u_ent, alpha=alpha, beta=beta)
        head.theta[i, j] -= 2 * eps
        lm_, *_ = head.loss_and_grads(Hx, mask, e, u_ent, alpha=alpha, beta=beta)
        head.theta[i, j] += eps
        assert g_theta[i, j] == pytest.approx((lp - lm_) / (2 * eps), abs=2e-5)
    for i in range(pool):
        head.bias[i] += eps
        lp, *_ = head.loss_and_grads(Hx, mask, e, u_ent, alpha=alpha, beta=beta)
        head.bias[i] -= 2 * eps
        lm_, *_ = head.loss_and_grads(Hx, mask, e, u_ent, alpha=alpha, beta=beta)
        head.bias[i] += eps
        assert g_bias[i] == pytest.approx((lp - lm_) / (2 * eps), abs=2e-5)
    assert loss > 0


def test_alien_reg_denominator_is_n_for_loss_and_gradient_alike():
    """L_reg = (1/N) sum (U_ALIEN - U_Entropy)^2 (2505.15443 S3.4).

    A row whose entropy anchor is undefined (|C| < 2 upstream) must contribute 0
    to BOTH the loss and the gradient.  With ``nanmean`` the loss divided by the
    finite count while the gradient divided by N, so the analytic gradient no
    longer matched the loss it claimed to differentiate.
    """
    rng = np.random.default_rng(23)
    dim, pool, n = 4, 3, 7
    ids = list(range(30, 30 + pool))
    lm = {t: rng.normal(size=dim) * 0.4 for t in ids}
    head = H.AlienHead.from_lm_head(ids, lm, dim)
    head.theta += rng.normal(size=head.theta.shape) * 0.1
    Hx = rng.normal(size=(n, dim))
    mask = np.ones((n, pool), dtype=bool)
    e = (rng.random(n) < 0.5).astype(float)
    u_ent = rng.random(n) * 0.6 + 0.2
    u_ent[2] = np.nan                                   # anchor missing on one row
    alpha, beta = 0.7, 0.0

    loss, g_theta, _, parts = head.loss_and_grads(Hx, mask, e, u_ent, alpha=alpha, beta=beta)
    assert np.isfinite(loss) and np.isfinite(parts["reg"])
    # the same rows with the NaN row's anchor set to its own U (zero residual) must
    # give the identical L_reg -- i.e. the denominator is n, not the finite count
    u_alt = u_ent.copy()
    u_alt[2] = float(head.score(Hx, mask)[2])
    _, _, _, parts_alt = head.loss_and_grads(Hx, mask, e, u_alt, alpha=alpha, beta=beta)
    assert parts["reg"] == pytest.approx(parts_alt["reg"])

    eps = 1e-6
    for _ in range(6):
        i = int(rng.integers(pool))
        j = int(rng.integers(dim))
        head.theta[i, j] += eps
        lp, *_ = head.loss_and_grads(Hx, mask, e, u_ent, alpha=alpha, beta=beta)
        head.theta[i, j] -= 2 * eps
        lm_, *_ = head.loss_and_grads(Hx, mask, e, u_ent, alpha=alpha, beta=beta)
        head.theta[i, j] += eps
        assert g_theta[i, j] == pytest.approx((lp - lm_) / (2 * eps), abs=2e-5)


def test_alien_l2sp_pulls_back_to_theta_init():
    rng = np.random.default_rng(4)
    dim, pool = 4, 3
    ids = [1, 2, 3]
    lm = {t: rng.normal(size=dim) for t in ids}
    head = H.AlienHead.from_lm_head(ids, lm, dim)
    Hx = rng.normal(size=(6, dim))
    mask = np.ones((6, pool), dtype=bool)
    e = np.array([1.0, 0, 1, 0, 1, 0])
    u_ent = head.score(Hx, mask)
    before = float(np.abs(head.theta - head.theta_init).sum())
    head.fit(Hx, mask, e, u_ent, alpha=0.0, beta=100.0, lr=1e-3, epochs=5, seed=0)
    after_big_beta = float(np.abs(head.theta - head.theta_init).sum())
    head2 = H.AlienHead.from_lm_head(ids, lm, dim)
    head2.fit(Hx, mask, e, u_ent, alpha=0.0, beta=0.0, lr=1e-3, epochs=5, seed=0)
    after_no_beta = float(np.abs(head2.theta - head2.theta_init).sum())
    assert before == 0.0
    assert after_big_beta < after_no_beta


def test_alien_fit_actually_reduces_the_bce_term():
    """Guards against an optimiser that silently does nothing."""
    rng = np.random.default_rng(15)
    dim, pool, n = 4, 3, 60
    ids = [1, 2, 3]
    lm = {t: rng.normal(size=dim) * 0.5 for t in ids}
    Hx = rng.normal(size=(n, dim))
    mask = np.ones((n, pool), dtype=bool)
    head0 = H.AlienHead.from_lm_head(ids, lm, dim)
    u0 = head0.score(Hx, mask)
    e = (u0 > np.median(u0)).astype(float)              # learnable by construction
    u_ent = u0.copy()
    before, *_ = head0.loss_and_grads(Hx, mask, e, u_ent, alpha=0.0, beta=0.0)
    head = H.AlienHead.from_lm_head(ids, lm, dim)
    head.fit(Hx, mask, e, u_ent, alpha=0.0, beta=0.0, lr=4e-4, epochs=20, seed=0)
    after, *_ = head.loss_and_grads(Hx, mask, e, u_ent, alpha=0.0, beta=0.0)
    assert after < before
    assert C.auroc(head.score(Hx, mask), e.astype(int)) >= C.auroc(u0, e.astype(int))


def test_memgen_fit_actually_learns_a_separable_problem():
    rng = np.random.default_rng(16)
    n, dim = 80, 3
    y = (np.arange(n) % 2).astype(float)
    Hx = rng.normal(size=(n, dim)) * 0.3
    Hx[:, 0] += 2.0 * y
    head = H.SparsityHead.zeros(dim)
    head.fit(Hx, y, y == 0, lam=0.0, lr=1e-2, epochs=30, seed=0)
    assert C.auroc(head.proba(Hx), y.astype(int)) > 0.9


def test_alien_random_init_control_differs_from_lm_head_init():
    rng = np.random.default_rng(5)
    ids = [7, 8]
    lm = {t: rng.normal(size=3) for t in ids}
    a = H.AlienHead.from_lm_head(ids, lm, 3)
    b = H.AlienHead.random_init(ids, 3, seed=0)
    assert not np.allclose(a.theta, b.theta)


def test_alien_missing_lm_head_row_is_an_error_not_a_zero():
    with pytest.raises(KeyError):
        H.AlienHead.from_lm_head([1, 2], {1: np.zeros(3)}, 3)


# ---------------------------------------------------------------------------
# MemGen sparsity head -- 2509.24704 S4.2
# ---------------------------------------------------------------------------

def test_memgen_gradient_matches_finite_differences():
    rng = np.random.default_rng(6)
    n, dim = 12, 5
    Hx = rng.normal(size=(n, dim))
    y = (rng.random(n) < 0.4).astype(float)
    head = H.SparsityHead(w=rng.normal(size=dim) * 0.3, b=0.2)
    lam, pbar = 2.0, 0.35
    loss, gw, gb, _ = head.loss_and_grads(Hx, y, lam=lam, pbar=pbar)
    eps = 1e-6
    for j in range(dim):
        head.w[j] += eps
        lp, *_ = head.loss_and_grads(Hx, y, lam=lam, pbar=pbar)
        head.w[j] -= 2 * eps
        lm_, *_ = head.loss_and_grads(Hx, y, lam=lam, pbar=pbar)
        head.w[j] += eps
        assert gw[j] == pytest.approx((lp - lm_) / (2 * eps), abs=2e-6)
    head.b += eps
    lp, *_ = head.loss_and_grads(Hx, y, lam=lam, pbar=pbar)
    head.b -= 2 * eps
    lm_, *_ = head.loss_and_grads(Hx, y, lam=lam, pbar=pbar)
    head.b += eps
    assert gb == pytest.approx((lp - lm_) / (2 * eps), abs=2e-6)
    assert loss > 0


def test_memgen_penalty_only_bites_above_pbar():
    Hx = np.zeros((4, 2))
    head = H.SparsityHead.zeros(2)          # p = 0.5 everywhere
    y = np.array([1.0, 1.0, 0.0, 0.0])
    _, _, _, parts_hi = head.loss_and_grads(Hx, y, lam=1.0, pbar=0.9)
    _, _, _, parts_lo = head.loss_and_grads(Hx, y, lam=1.0, pbar=0.1)
    assert parts_hi["penalty"] == 0.0
    assert parts_lo["penalty"] == pytest.approx(0.4)


def test_memgen_pbar_is_the_cc_row_activation_rate():
    """pbar must come from the C->C rows only (2509.24704's high-reward set)."""
    rng = np.random.default_rng(7)
    n, dim = 20, 3
    Hx = rng.normal(size=(n, dim))
    y = np.array([1.0] * 10 + [0.0] * 10)
    cc = y == 0
    head = H.SparsityHead.zeros(dim)
    head.fit(Hx, y, cc, lam=0.0, lr=1e-2, epochs=3, seed=0)
    p = head.proba(Hx)
    assert float(p[cc].mean()) != pytest.approx(float(p.mean()), abs=1e-9)


def test_memgen_requires_lambda_zero_ablation():
    frame = _synth_frame()
    inputs = _synth_inputs(frame)
    with pytest.raises(ValueError, match="lambda=0"):
        H.run_memgen(inputs, frame, lam_grid=(0.1, 1.0), lr_grid=(1e-2,), epochs=1, reps=5)


# ---------------------------------------------------------------------------
# CORA -- 2604.09155 S3.4 (CRC) and S12.5 (blockwise split)
# ---------------------------------------------------------------------------

def test_crc_threshold_matches_brute_force():
    rng = np.random.default_rng(8)
    for trial in range(20):
        n = int(rng.integers(30, 90))
        s = np.round(rng.random(n), 3)
        harm = (rng.random(n) < 0.3).astype(float)
        alpha = float(rng.choice([0.05, 0.1, 0.2]))
        grid = np.linspace(0.0, 1.0, 201).tolist()
        got = H.crc_threshold(s, harm, alpha, grid=grid)
        feasible = [t for t in grid
                    if (float((harm * (s <= t)).sum()) + 1.0) / (n + 1.0) <= alpha]
        want = max(feasible) if feasible else None
        assert got["tau_hat"] == (None if want is None else pytest.approx(want))


def test_crc_default_grid_is_finite_and_json_strict():
    """tau_hat must never be +/-inf: freeze_json would emit invalid JSON."""
    s = np.array([0.2, 0.4, 0.6, 0.8] * 10)
    harm = np.ones(40)                       # every row harmful -> fire on everything
    out = H.crc_threshold(s, harm, 0.05)
    assert out["tau_hat"] is not None and np.isfinite(out["tau_hat"])
    assert out["tau_hat"] < s.min()
    y = (harm > 0).astype(int)
    col = H.cora_columns(s, y, out["tau_hat"])
    assert col["fires"] == 40 and col["executed_harm_rate"] == 0.0
    text = json.dumps({"crc": out, "col": col})
    json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(
        AssertionError(f"non-finite constant {x} in output")))


def test_crc_threshold_is_monotone_and_infeasible_when_n_too_small():
    s = np.array([0.1, 0.2, 0.3])
    harm = np.array([0.0, 0.0, 0.0])
    out = H.crc_threshold(s, harm, 0.05)
    # (0 + 1) / (3 + 1) = 0.25 > 0.05 -> no tau can certify
    assert out["tau_hat"] is None and out["feasible"] is False
    assert out["min_n_for_alpha"] == 19
    # with zero harm and a big enough n the sup is the largest score
    s2 = np.linspace(0, 1, 40)
    out2 = H.crc_threshold(s2, np.zeros(40), 0.05)
    assert out2["tau_hat"] == pytest.approx(1.0)


def test_crc_never_certifies_a_threshold_that_breaks_the_budget():
    rng = np.random.default_rng(9)
    n = 200
    s = rng.random(n)
    harm = (s < 0.4).astype(float)          # harm concentrated at low scores
    out = H.crc_threshold(s, harm, 0.05)
    tau = out["tau_hat"]
    loss = float((harm * (s <= tau)).sum())
    assert (loss + 1.0) / (n + 1.0) <= 0.05


def test_cora_columns_arithmetic():
    s = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1, 0, 1, 0])
    col = H.cora_columns(s, y, 0.5)         # fire on the top two
    assert col["fires"] == 2 and col["fire_rate"] == pytest.approx(0.5)
    assert col["coverage_recall_cw"] == pytest.approx(0.5)
    assert col["coverage_autonomous"] == pytest.approx(0.5)
    assert col["residual_cw"] == 1
    assert col["residual_cw_among_nonfired"] == pytest.approx(0.5)
    assert col["executed_harm_rate"] == pytest.approx(0.25)
    assert col["false_resets"] == 1 and col["false_reset_rate"] == pytest.approx(0.5)


def test_cora_primary_endpoint_has_no_default():
    s = np.linspace(0, 1, 40)
    y = (np.arange(40) % 3 == 0).astype(int)
    idx = list(range(40))
    with pytest.raises(ValueError, match="primary_endpoint"):
        H.cora_ablation_table(s, y, primary_endpoint="whatever", alpha=0.1,
                              cal_idx=idx, test_idx=idx)
    out = H.cora_ablation_table(s, y, primary_endpoint="executed_harm_rate", alpha=0.1,
                                cal_idx=idx, test_idx=idx, static_taus=[0.5],
                                parse_fail_fire=np.zeros(40))
    settings = [r["setting"] for r in out["rows"]]
    assert any(x.startswith("static tau") for x in settings)
    assert any(x.startswith("CRC tau_hat") for x in settings)
    assert any("parse-failure only" in x for x in settings)


def test_block_split_never_splits_a_session_or_a_toolset():
    sessions = [f"s{i//5:02d}" for i in range(100)]
    keys = {f"s{i:02d}": ("K" if i % 4 == 0 else f"T{i}") for i in range(20)}
    out = H.cora_block_split(sessions, toolset_keys=keys, seed=3)
    assign = out["assignment"]
    # a session lands in exactly one split
    for s in set(sessions):
        assert assign[s] in ("train", "cal", "test")
    # every session sharing toolset "K" is in the same split
    shared = [s for s, k in keys.items() if k == "K"]
    assert len({assign[s] for s in shared}) == 1
    assert sum(out["n_rows_by_split"].values()) == len(sessions)


def test_block_split_that_collapses_to_one_block_is_loud_not_a_null_table():
    """One toolset shared by every session merges the whole frame into one block.

    The CRC table would then be computed on n_cal = 0 and come back as a column
    of nulls, which reads like "no signal" instead of "the split is infeasible".
    """
    sessions = [f"s{i:02d}" for i in range(20) for _ in range(3)]
    keys = {f"s{i:02d}": "same-toolset" for i in range(20)}
    split = H.cora_block_split(sessions, toolset_keys=keys, seed=0)
    assert split["n_blocks"] == 1
    assert split["max_block_row_share"] == pytest.approx(1.0)
    assert split["degenerate"] is True
    assert split["n_rows_by_split"]["cal"] == 0
    rng = np.random.default_rng(21)
    n = len(sessions)
    with pytest.raises(ValueError, match="empty calibration or test fold"):
        H.cora_ablation_table(rng.random(n), (rng.random(n) < 0.5).astype(int),
                              primary_endpoint="executed_harm_rate", alpha=0.1,
                              cal_idx=[], test_idx=list(range(n)))


def test_block_split_without_toolset_keys_keeps_sessions_separate_blocks():
    sessions = [f"s{i}" for i in range(12) for _ in range(2)]
    out = H.cora_block_split(sessions, toolset_keys=None, seed=1)
    assert out["n_blocks"] == 12 and out["n_sessions"] == 12


def test_toolset_key_matches_build_joint_split_manifest():
    tools = [_tool("b", {"x": {"type": "string"}}, ["x"]),
             _tool("a", {"y": {"type": "integer"}})]
    try:
        import build_joint_split_manifest as B
    except Exception:                                   # pragma: no cover
        pytest.skip("build_joint_split_manifest not importable (pyarrow missing)")
    assert H.toolset_key(tools) == B._toolset_key(tools)
    # order-invariant, and sensitive to the parameter signature
    assert H.toolset_key(tools) == H.toolset_key(list(reversed(tools)))
    other = [_tool("b", {"x": {"type": "integer"}}, ["x"]),
             _tool("a", {"y": {"type": "integer"}})]
    assert H.toolset_key(tools) != H.toolset_key(other)


# ---------------------------------------------------------------------------
# Self-REF / [RESET] label structures -- 2410.13284 Alg.1, 2409.14586 S3
# ---------------------------------------------------------------------------

def test_wrong_any_arm_denominators_on_the_frozen_battery():
    frame = _frozen()
    wa = H.label_wrong_any(frame)
    assert len(wa) == 900
    assert sum(wa.values()) == 712                      # C->W 93 + W->W 619
    three = H.selfref_label_three_valued(frame)
    assert three["counts"] == {"cw": 93, "ww": 619, "correct": 188}
    pairs = H.reset_label_negative_pairs(frame)
    assert pairs["counts"] == {"cw": 93, "cc": 68, "wc": 120, "ww": 619}
    assert len(pairs["untried_third_class_wc"]) == 120


def test_alpha_downsample_arithmetic_applies_to_ww_only():
    classes = {f"q{i}": "ww" for i in range(100)}
    classes.update({f"p{i}": "cw" for i in range(10)})
    classes.update({f"r{i}": "correct" for i in range(20)})
    out = H.selfref_label_alpha_downsample(classes, 0.25, seed=0)
    assert out["n_ww_total"] == 100 and out["n_ww_kept"] == 25
    assert out["n_cw_kept"] == 10 and out["n_correct_kept"] == 20
    assert out["n_kept"] == 55
    assert out["positive_fraction_after"] == pytest.approx(35 / 55)
    assert H.selfref_label_alpha_downsample(classes, 1.0)["n_kept"] == 130
    assert H.selfref_label_alpha_downsample(classes, 0.0)["n_kept"] == 30
    with pytest.raises(ValueError):
        H.selfref_label_alpha_downsample(classes, 1.5)


def test_alpha_downsample_never_drops_a_cw_row_on_the_frozen_battery():
    frame = _frozen()
    three = H.selfref_label_three_valued(frame)
    out = H.selfref_label_alpha_downsample(three["classes"], 0.1, seed=0)
    assert out["n_cw_kept"] == 93
    assert out["n_ww_kept"] == 61                       # floor(0.1 * 619)
    assert set(frame.cw_qids()).issubset(set(out["kept_qids"]))


def test_reset_positive_prefix_first_divergence():
    a = {"x:1": [1, 2, 3, 4], "x:2": [5, 6], "x:3": [7, 8, 9]}
    b = {"x:1": [1, 2, 9, 4], "x:2": [5, 6, 7], "x:3": [7, 8, 9]}
    out = H.reset_label_positive_prefixes(a, b, ["x:1", "x:2", "x:3"], cap_tokens=4)
    e = out["entries"]
    assert e["x:1"]["first_div"] == 2
    assert e["x:1"]["cap_hit_c2kv"] is True
    assert e["x:2"]["first_div"] is None and e["x:2"]["prefix_only"] is True
    assert e["x:3"]["first_div"] is None and e["x:3"]["prefix_only"] is False
    assert out["n_with_divergence"] == 1


# ---------------------------------------------------------------------------
# metrics the papers add
# ---------------------------------------------------------------------------

def test_aurc_and_oracle_and_ece():
    y = np.array([1, 1, 0, 0, 0, 0])
    perfect = np.array([1.0, 0.9, 0.1, 0.1, 0.1, 0.1])
    assert H.aurc(perfect, y) == pytest.approx(H.oracle_aurc(y))
    worst = -perfect
    assert H.aurc(worst, y) > H.aurc(perfect, y)
    # a perfectly calibrated constant predictor on a balanced set
    p = np.array([0.5] * 4)
    assert H.ece(p, np.array([1, 1, 0, 0])) == pytest.approx(0.0)
    assert H.ece(np.array([0.9, 0.9]), np.array([0, 0])) == pytest.approx(0.9)


def test_risk_coverage_curve_shape():
    cov, risk = H.risk_coverage(np.array([0.1, 0.2, 0.9]), np.array([0, 0, 1]))
    assert cov[-1] == pytest.approx(1.0)
    assert risk[0] == 0.0 and risk[-1] == pytest.approx(1 / 3)


def test_score_row_reports_its_own_denominators():
    y = np.array([1] * 5 + [0] * 5)
    s = np.arange(10, dtype=float)[::-1]
    s[0] = np.nan
    row = H.score_row("x", s, y, [f"s{i//2}" for i in range(10)], reps=20, n_fires=[2])
    assert row["n"] == 9 and row["n_scored_of"] == 10 and row["n_pos"] == 4
    assert row["prevalence"] == pytest.approx(4 / 9)
    assert row["operating_points"][0]["fires"] == 2


# ---------------------------------------------------------------------------
# nested CV plumbing
# ---------------------------------------------------------------------------

def test_nested_cv_head_is_out_of_fold_and_group_respecting():
    rng = np.random.default_rng(11)
    groups = np.array([f"g{i // 4}" for i in range(40)])
    y = (np.arange(40) % 2).astype(float)
    X = rng.normal(size=(40, 2)) + y[:, None]
    seen_train_groups = []

    def fit_predict(tr, te, params):
        seen_train_groups.append((set(groups[tr]), set(groups[te])))
        mu = X[tr].mean(axis=0)
        return {"test": ((X[te] - mu) ** 2).sum(1) * params["k"],
                "train": ((X[tr] - mu) ** 2).sum(1) * params["k"]}

    res = H.nested_cv_head(fit_predict, y_train=y, groups=groups,
                           grid=[{"k": 1.0}, {"k": 2.0}], outer_folds=4, inner_folds=2,
                           fire_rates=(0.25,))
    assert res["n_scored"] == 40
    for tr_g, te_g in seen_train_groups:
        assert not (tr_g & te_g)
    assert res["fires"][0.25].sum() > 0


# ---------------------------------------------------------------------------
# input frame: capture + hidden shard loaders, save/load round trip
# ---------------------------------------------------------------------------

def _write_synth_capture(tmp_path: Path, qids, dim=6, n_layers=3, pool_ids=(101, 102, 103)):
    arm_dir = tmp_path / "capture" / "c2kv"
    arm_dir.mkdir(parents=True)
    rng = np.random.default_rng(12)
    with io.open(arm_dir / "p0.steps.jsonl", "w", encoding="utf-8") as fh:
        for qid in qids:
            steps = [{"step": j, "token_id": j, "chosen_logprob": -0.1,
                      "entropy_full": 0.5 + 0.01 * j,
                      "top5": [[-0.1, 1], [-0.9, 2]]} for j in range(6)]
            fh.write(json.dumps({
                "qid": qid, "generated_ids": [1, 2, 3, 4, 5, 6], "steps": steps,
                "spans": {"name_first": 2, "name_last": 3, "n_generated": 6},
                "anchors": [["name_first", 2]],
                "ic": {"n_candidates": len(pool_ids),
                       "candidate_token_ids": list(pool_ids), "anchors": {}},
            }) + "\n")
    payload = {}
    for qid in qids:
        payload[f"{qid}::anchor_hidden"] = rng.normal(
            size=(n_layers, 1, dim)).astype(np.float16)
        payload[f"{qid}::anchor_labels"] = np.array(["name_first"])
        payload[f"{qid}::anchor_positions"] = np.array([2])
        payload[f"{qid}::anchor_valid"] = np.array([True])
        payload[f"{qid}::layers"] = np.arange(n_layers)
    np.savez_compressed(arm_dir / "p0_0001.hid.npz", **payload)
    return arm_dir


def test_build_head_inputs_from_synthetic_capture(tmp_path):
    frame = _synth_frame(n_sessions=6, steps=3)
    qids = [r["qid"] for r in frame.labels]
    sessions = [r["session_id"] for r in frame.labels]
    arm_dir = _write_synth_capture(tmp_path, qids)
    capture = H.load_capture_steps(arm_dir / "p0.steps.jsonl")
    hid = H.load_anchor_hiddens(arm_dir)
    assert len(capture) == len(qids) and len(hid) == len(qids)
    inputs = H.build_head_inputs(qids, sessions, capture, hid, cap_tokens=128)
    assert inputs.n == len(qids)
    assert inputs.valid.all()
    assert inputs.hidden.shape[1] == 1                  # penultimate slot by default
    assert inputs.entropy_vocab[0] == pytest.approx(0.52)
    assert inputs.margin_top1_top2[0] == pytest.approx(0.8)
    assert inputs.pool_size().tolist() == [3] * len(qids)
    mask = inputs.pool_mask()
    assert mask.shape == (len(qids), 3) and mask.all()
    # round trip
    out = tmp_path / "inputs.npz"
    inputs.save(out)
    back = H.HeadInputs.load(out)
    assert back.qids == inputs.qids and back.cand_ids == inputs.cand_ids
    assert np.allclose(back.hidden, inputs.hidden, atol=1e-3)


def test_build_head_inputs_marks_missing_rows_invalid_not_sentinel(tmp_path):
    frame = _synth_frame(n_sessions=4, steps=2)
    qids = [r["qid"] for r in frame.labels]
    sessions = [r["session_id"] for r in frame.labels]
    arm_dir = _write_synth_capture(tmp_path, qids[:-1])
    capture = H.load_capture_steps(arm_dir / "p0.steps.jsonl")
    hid = H.load_anchor_hiddens(arm_dir)
    inputs = H.build_head_inputs(qids, sessions, capture, hid, cap_tokens=128)
    assert inputs.valid[:-1].all() and not inputs.valid[-1]
    assert np.isnan(inputs.entropy_vocab[-1])           # None kept as NaN, not 0.0
    assert inputs.cand_ids[-1] == []


def test_cap_hit_prefers_the_capture_stop_reason(tmp_path):
    frame = _synth_frame(n_sessions=2, steps=2)
    qids = [r["qid"] for r in frame.labels]
    sessions = [r["session_id"] for r in frame.labels]
    arm_dir = _write_synth_capture(tmp_path, qids)
    capture = H.load_capture_steps(arm_dir / "p0.steps.jsonl")
    hid = H.load_anchor_hiddens(arm_dir)
    capture[qids[0]]["stop_reason"] = "length"
    capture[qids[1]]["stop_reason"] = "eos"
    inputs = H.build_head_inputs(qids, sessions, capture, hid, cap_tokens=128)
    assert bool(inputs.cap_hit[0]) is True      # 6 generated tokens, but stop=length
    assert bool(inputs.cap_hit[1]) is False


def test_cora_split_units_are_the_frozen_session_counts():
    frame = _frozen()
    qids, sessions, y, _ = H._eval_frame(frame)
    assert len(set(sessions)) == 100
    cw_sessions = {C.session_of(q) for q in frame.cw_qids()}
    assert len(cw_sessions) == 72
    split = H.cora_block_split(sessions, toolset_keys=None, seed=1)
    assert split["n_sessions"] == 100 and split["n_blocks"] == 100
    assert sum(split["n_rows_by_split"].values()) == 161
    # no session straddles two splits
    by_session = {}
    for q, s in zip(qids, sessions):
        by_session.setdefault(s, set()).add(split["assignment"][s])
    assert all(len(v) == 1 for v in by_session.values())


def test_lm_head_row_loader_round_trip(tmp_path):
    ids = [5, 9, 11]
    rows = np.arange(9, dtype=np.float32).reshape(3, 3)
    p = tmp_path / "lm.npz"
    np.savez_compressed(p, ids=np.asarray(ids), rows=rows)
    got = H.load_lm_head_rows(p)
    assert sorted(got) == ids and np.allclose(got[9], rows[1])


# ---------------------------------------------------------------------------
# drivers, end to end on synthetic data
# ---------------------------------------------------------------------------

def test_run_alien_table_starts_with_the_bare_entropy_baseline():
    frame = _synth_frame(n_sessions=20, steps=4, seed=2)
    inputs = _synth_inputs(frame, dim=5, pool=4)
    rng = np.random.default_rng(13)
    lm = {t: rng.normal(size=5) for t in inputs.global_ids}
    out = H.run_alien(inputs, frame, lm, alpha_grid=(0.0,), beta_grid=(0.01,),
                      lr_grid=(1e-3,), epochs=2, reps=20)
    rows = out["table"]
    assert rows[0]["row"].startswith("bare entropy")
    assert any(r["row"].startswith("baseline: parse failure") for r in rows)
    assert any(r["row"] == "ALIEN arm_wrongany" for r in rows)
    assert any(r["row"] == "ALIEN arm_cw" for r in rows)
    assert any("Rand CLS" in r["row"] for r in rows)
    # every row is scored on the SAME frame and reports its own denominators
    frame_n = out["frame"]["n"]
    assert out["frame"]["prevalence_is_chance_ap"] == pytest.approx(
        out["frame"]["n_pos"] / frame_n)
    for r in rows:
        assert r["n_scored_of"] == frame_n
    assert out["arm_gap"] is not None
    assert any("|C| == 2" in c for c in out["caveats"])
    assert any("1/log C" in c for c in out["caveats"])
    # the frozen Oracle column the digest asks for sits next to AURC
    assert out["frame"]["oracle_repair_ceiling_hits"] == [75, 93]
    assert out["frame"]["oracle_repair_ceiling"] == pytest.approx(0.8065, abs=1e-4)
    # the head really trained: a grid point was chosen in every outer fold, and the
    # arm's OOF scores are not the init scores
    for arm in ("arm_wrongany", "arm_cw"):
        assert out["arms"][arm]["chosen"], f"{arm}: no fold produced a fit"
        assert all(0.0 <= c["inner_metric"] <= 1.0 for c in out["arms"][arm]["chosen"])
    alien = [r for r in rows if r["row"] == "ALIEN arm_wrongany"][0]
    lens = [r for r in rows if r["row"].startswith("U_Entropy")][0]
    assert alien["auprc"] is not None and lens["auprc"] is not None
    assert alien["aurc"] is not None and alien["aurc_oracle"] <= alien["aurc"]
    assert alien["ece"] is not None
    json.dumps(out["table"])                            # JSON-serialisable


def test_run_alien_scores_the_right_rows_when_some_are_invalid():
    """Regression: with a strict ``usable`` subset the head must still train on,
    and score, the rows nested_cv_head actually selected.

    ``run_alien`` hands ``nested_cv_head`` a SUBSET (``usable``) but every array
    the fit closure indexes (``views``, ``mask_all``, ``e_train``, ``u_ent``) is
    full-length, so the subset-local fold indices have to be mapped back through
    ``sub``.  Before that map existed the head trained on rows 0..len(sub)-1 -
    including rows explicitly marked invalid - and each OOF score landed on a
    different qid than the row it was computed from.
    """
    frame = _synth_frame(n_sessions=20, steps=4, seed=3)
    inputs = _synth_inputs(frame, dim=5, pool=4)
    inputs.valid[:10] = False                      # a strict usable subset
    rng = np.random.default_rng(0)
    lm = {t: rng.normal(size=5) for t in inputs.global_ids}
    view = inputs.layer_view(0)

    fitted, scored = [], []
    orig_fit, orig_score = H.AlienHead.fit, H.AlienHead.score

    def spy_fit(self, Hx, mask, e, u_ent, **kw):
        fitted.append(np.asarray(Hx).copy())
        return orig_fit(self, Hx, mask, e, u_ent, **kw)

    def spy_score(self, Hx, mask):
        u = orig_score(self, Hx, mask)
        scored.append((np.asarray(Hx).copy(), np.asarray(u).copy()))
        return u

    H.AlienHead.fit, H.AlienHead.score = spy_fit, spy_score
    try:
        out = H.run_alien(inputs, frame, lm, alpha_grid=(0.0,), beta_grid=(0.01,),
                          lr_grid=(1e-3,), epochs=1, reps=10)
    finally:
        H.AlienHead.fit, H.AlienHead.score = orig_fit, orig_score

    def which(row):
        return int(np.argmin(np.abs(view - row).sum(axis=1)))

    trained_on = {which(r) for block in fitted for r in block}
    assert trained_on, "the head never fitted"
    assert not (trained_on & set(range(10))),         f"invalid rows leaked into training: {sorted(trained_on & set(range(10)))}"
    assert trained_on <= set(np.where(inputs.valid)[0].tolist())

    # every emitted OOF score must be the head's output ON THAT ROW's own hidden
    oof = out["oof_scores"]["arm_wrongany"]
    assert np.isfinite(oof[10:]).any() and not np.isfinite(oof[:10]).any()
    by_row = {}
    for Hx, u in scored:
        for r, uu in zip(Hx, u):
            by_row.setdefault(which(r), set()).add(round(float(uu), 12))
    for i in np.where(np.isfinite(oof))[0]:
        assert round(float(oof[i]), 12) in by_row.get(int(i), set()),             f"OOF score at row {i} was not computed from row {i}'s hidden state"


def test_rand_cls_control_is_bce_only_as_published():
    """2505.15443 tab:ablation 'Rand CLS. BCE': BCE only, so alpha = beta = 0."""
    frame = _synth_frame(n_sessions=16, steps=4, seed=7)
    inputs = _synth_inputs(frame, dim=4, pool=3)
    rng = np.random.default_rng(5)
    lm = {t: rng.normal(size=4) for t in inputs.global_ids}
    seen = []
    orig = H.AlienHead.random_init

    def spy(global_ids, dim, seed=0):
        return orig(global_ids, dim, seed=seed)

    orig_fit = H.AlienHead.fit
    rand_ids = set()

    def spy_fit(self, Hx, mask, e, u_ent, *, alpha, beta, lr, **kw):
        # a random-init head has theta_init far from any lm_head row
        if not any(np.allclose(self.theta_init[k], lm[t])
                   for k, t in enumerate(inputs.global_ids)):
            seen.append((alpha, beta))
        return orig_fit(self, Hx, mask, e, u_ent, alpha=alpha, beta=beta, lr=lr, **kw)

    H.AlienHead.fit = spy_fit
    try:
        H.run_alien(inputs, frame, lm, alpha_grid=(0.1,), beta_grid=(0.1,),
                    lr_grid=(1e-3,), epochs=1, reps=10)
    finally:
        H.AlienHead.fit = orig_fit
    assert seen, "the Rand CLS control never fitted"
    assert set(seen) == {(0.0, 0.0)}, f"Rand CLS control ran with {sorted(set(seen))}"


def test_run_alien_s0_twin_on_the_full_arm_is_produced():
    frame = _synth_frame(n_sessions=14, steps=4, seed=4)
    inputs = _synth_inputs(frame, dim=4, pool=3, seed=1)
    full = _synth_inputs(frame, dim=4, pool=3, seed=99)
    rng = np.random.default_rng(14)
    lm = {t: rng.normal(size=4) for t in inputs.global_ids}
    out = H.run_alien(inputs, frame, lm, inputs_full=full, alpha_grid=(0.0,),
                      beta_grid=(0.01,), lr_grid=(1e-3,), epochs=1, reps=10)
    assert out["s0_full_arm"] is not None
    assert out["s0_full_arm"]["table"][0]["row"].startswith("bare entropy")
    # digest S4.0 winner rule: the S0 contrast has to be a PAIRED clustered
    # delta-AUPRC with a CI, not two tables printed side by side.
    d = out["s0_full_arm"]["delta_auprc_vs_s0"]
    assert d is not None and d["delta_auprc"] is not None
    assert d["ci95"][0] is not None and d["ci95"][0] <= d["delta_auprc"] <= d["ci95"][1]
    assert d["n_clusters"] >= 2
    json.dumps(out["s0_full_arm"])


def test_run_alien_reports_the_paired_increment_over_the_length_control():
    """digest S4.0: a live signal needs an INCREMENT over the length control."""
    frame = _synth_frame(n_sessions=20, steps=4, seed=2)
    inputs = _synth_inputs(frame, dim=5, pool=4)
    rng = np.random.default_rng(13)
    lm = {t: rng.normal(size=5) for t in inputs.global_ids}
    out = H.run_alien(inputs, frame, lm, alpha_grid=(0.0,), beta_grid=(0.01,),
                      lr_grid=(1e-3,), epochs=1, reps=20)
    inc = out["increment_over_length_control"]
    assert inc is not None and inc["delta_auprc"] is not None
    assert inc["ci95"][0] <= inc["delta_auprc"] <= inc["ci95"][1]
    assert "generated_tokens" in inc["contrast"]
    json.dumps(inc)


def test_run_memgen_reports_false_reset_and_length_control():
    frame = _synth_frame(n_sessions=20, steps=4, seed=5)
    inputs = _synth_inputs(frame, dim=5)
    out = H.run_memgen(inputs, frame, lam_grid=(0.0, 1.0), lr_grid=(1e-2,), epochs=2,
                       fire_rates=(0.1,), reps=20)
    labels = [r["row"] for r in out["table"]]
    assert labels[0].startswith("bare entropy")
    assert "control: generated_tokens + cap_hit (logistic)" in labels
    assert "MemGen sparsity head" in labels
    col = out["false_reset_column"]["0.1"]
    assert col["n_pos"] + col["n_neg"] == out["frame"]["n"]
    assert col["false_resets"] <= col["n_neg"]
    assert out["increment_over_length_control"] is not None
    json.dumps(out["table"])


def test_run_alien_frame_is_the_161_row_subset_on_the_frozen_battery():
    frame = _frozen()
    qids, sessions, y, pf = H._eval_frame(frame)
    assert len(qids) == 161 and int(y.sum()) == 93
    assert C.prevalence(y) == pytest.approx(0.5776, abs=1e-4)
    assert len(set(sessions)) == 100
    assert C.BASE_RATE_900 == pytest.approx(0.1033, abs=1e-4)   # never the chance line here


# ---------------------------------------------------------------------------
# leakage guard, orientations, CLI
# ---------------------------------------------------------------------------

def test_emitted_feature_columns_pass_the_leakage_guard():
    guard_columns(sorted(H.ORIENTATIONS), context="t34 heads")
    assert set(H.ORIENTATIONS.values()) <= {1, -1}
    for col in ("entropy_name_token_vocab", "alien_u_arm_wrongany_oof", "memgen_p_oof",
                "ctrl_generated_tokens", "ctrl_cap_hit", "pool_size_c"):
        assert col in H.ORIENTATIONS


def test_orientations_file_on_disk_matches_the_module():
    p = ROOT / "configs/t34/orientations_heads.json"
    assert p.exists(), "run: python agent/t34_heads.py orientations"
    assert json.loads(p.read_text(encoding="utf-8")) == H.ORIENTATIONS


def test_build_inputs_features_pass_write_features_jsonl(tmp_path):
    frame = _synth_frame(n_sessions=4, steps=2)
    inputs = _synth_inputs(frame, dim=3, pool=3)
    pool = inputs.pool_size()
    rows = [{"qid": q, "session_id": inputs.sessions[i], "arm": "c2kv",
             "entropy_name_token_vocab": float(inputs.entropy_vocab[i]),
             "margin_name_top1_top2": float(inputs.margin_top1_top2[i]),
             "ctrl_generated_tokens": float(inputs.generated_tokens[i]),
             "ctrl_cap_hit": float(inputs.cap_hit[i]),
             "pool_size_c": int(pool[i])}
            for i, q in enumerate(inputs.qids)]
    n = C.write_features_jsonl(tmp_path / "f.jsonl", rows, context="t34 heads test")
    assert n == len(rows)
    got = C.load_jsonl(str(tmp_path / "f.jsonl"))
    assert set(got[0]) - C.META_COLS <= set(H.ORIENTATIONS)


def test_feature_cell_helper_keeps_missing_as_null_not_a_sentinel():
    v = np.array([1.5, np.nan, np.inf])
    assert H._f(v, 0) == pytest.approx(1.5)
    assert H._f(v, 1) is None
    assert H._f(v, 2) is None
    assert H._f(None, 0) is None


def test_a_label_column_is_still_refused_by_the_guard():
    with pytest.raises(ValueError):
        guard_columns(["entropy_name_token_vocab", "a_made_call"], context="t34 heads")
    with pytest.raises(ValueError):
        guard_columns(["tool_name_match"], context="t34 heads")


def test_deviations_are_declared_for_every_migrated_method():
    methods = {d["method"] for d in H.DEVIATIONS}
    assert {"ALIEN head", "MemGen trigger", "CORA", "Self-REF", "[RESET]"} <= methods
    for d in H.DEVIATIONS:
        assert set(d) == {"method", "paper", "what", "why"}
        assert all(d[k].strip() for k in d)


@pytest.mark.parametrize("cmd", ["dump-lm-head", "build-inputs", "alien", "memgen",
                                 "cora", "labels", "orientations"])
def test_cli_subcommand_help(cmd):
    out = subprocess.run([sys.executable, str(ROOT / "agent/t34_heads.py"), cmd, "--help"],
                         capture_output=True, text=True, encoding="utf-8",
                         env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert out.returncode == 0
    assert "usage" in out.stdout


def test_decode_span_scalars_five_columns_and_none_when_span_absent():
    import math
    import t34_heads as H
    steps = [{"entropy_full": 0.5, "chosen_logprob": -0.1, "top5": [[-0.1, 1], [-2.0, 2]]},
             {"entropy_full": 1.5, "chosen_logprob": -0.7, "top5": [[-0.7, 3], [-0.9, 4]]},
             {"entropy_full": 0.2, "chosen_logprob": -0.05, "top5": [[-0.05, 5], [-3.0, 6]]}]
    row = {"steps": steps, "spans": {"name_first": 1, "payload_first": 0, "payload_last": 2}}
    sc = H.decode_span_scalars(row)
    assert sc["entropy_name_token_vocab"] == 1.5
    assert math.isclose(sc["margin_name_top1_top2"], 0.2)
    assert math.isclose(sc["span_entropy_mean"], (0.5 + 1.5 + 0.2) / 3)
    assert sc["span_entropy_max"] == 1.5
    assert math.isclose(sc["span_seq_nll"], 0.85)
    sc2 = H.decode_span_scalars({"steps": steps, "spans": {}})
    assert all(v is None for v in sc2.values())
