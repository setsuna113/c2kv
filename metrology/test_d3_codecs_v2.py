# -*- coding: utf-8 -*-
"""CPU unit tests for d3_codecs_v2: encode->nbytes->decode roundtrips and
the honesty of the bytes axis (packed sizes exact, no int16 fiction).

Run:  pytest metrology/test_d3_codecs_v2.py -v   (needs torch; skips without)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "python", _REPO_ROOT / "python" / "inference",
          _REPO_ROOT / "agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
for _stub in ("inference.mdocdataset", "inference"):
    sys.modules.pop(_stub, None)

from d3_codecs_v2 import (  # noqa: E402
    channel_sensitivity,
    dec_aatc,
    dec_kvtc,
    dec_raw_bf16,
    dec_raw_q4,
    dec_vector_konly,
    enc_aatc,
    enc_kvtc,
    enc_raw_bf16,
    enc_raw_q4,
    enc_vector_konly,
    fit_pca_basis,
    fit_v_regression_heldout,
    waterfill_bits,
)

H, L, D, DV = 8, 64, 128, 128


_FIXED_BASIS_K = None
_FIXED_BASIS_V = None


def _kv(seed=0, lowrank=True):
    """Test blocks: LOW-RANK + noise from ONE FIXED subspace, so a PCA
    basis fitted on held-out blocks actually spans the test block (real KV
    shares structure across blocks; per-seed random subspaces would make
    the offline basis useless by construction)."""
    global _FIXED_BASIS_K, _FIXED_BASIS_V
    g = torch.Generator().manual_seed(seed)
    if _FIXED_BASIS_K is None:
        fg = torch.Generator().manual_seed(12345)
        _FIXED_BASIS_K = torch.randn(16, D, generator=fg)
        _FIXED_BASIS_V = torch.randn(16, DV, generator=fg)
    def mk(basis, dv):
        if lowrank:
            coef = torch.randn(H, L, 16, generator=g)
            x = torch.einsum("hlr,rd->hld", coef, basis)
            return x + 0.05 * torch.randn(H, L, dv, generator=g)
        return torch.randn(H, L, dv, generator=g)
    return mk(_FIXED_BASIS_K, D), mk(_FIXED_BASIS_V, DV) * 0.5


def _q(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(32, L, D, generator=g)


def test_raw_bf16_roundtrip_exact_bytes():
    k, v = _kv()
    p = enc_raw_bf16(k, v)
    k2, v2 = dec_raw_bf16(p)
    assert torch.equal(k2, k.to(torch.float16).to(torch.float32))
    assert p.part_bytes()["k"] == k.numel() * 2
    # S0.3: DEFLATE column exists and is a real compression size
    assert 0 < p.bytes_deflate < p.nbytes + 64


def test_raw_q4_roundtrip_and_packed_size():
    k, v = _kv()
    p = enc_raw_q4(k, v)
    k2, v2 = dec_raw_q4(p)
    assert k2.shape == k.shape
    rel = ((k2 - k).norm() / k.norm()).item()
    assert rel < 0.2, f"q4 relative error too high: {rel}"
    # the packed stream is EXACTLY numel*4 bits — no int16 fiction
    assert p.part_bytes()["k_codes"] == (H * L * D * 4 + 7) // 8
    assert p.part_bytes()["k_codes"] < k.numel() * 2  # smaller than one f16 plane


def test_vector_konly_heldout_vs_selffit():
    train = [_kv(seed=s) for s in range(3)]
    W = fit_v_regression_heldout(train)
    k, v = _kv(seed=9)
    # held-out encode: W NOT billed
    p = enc_vector_konly(k, v, W=W)
    assert "W" not in p.header or p.header["self_fit"] is False
    k2, v2 = dec_vector_konly(p, W=W)
    assert ((k2 - k).norm() / k.norm()).item() < 0.1
    # self-fit bills W and is flagged as a diagnostic upper bound
    ps = enc_vector_konly(k, v, self_fit=True)
    assert ps.header["self_fit"] is True
    assert ps.part_bytes().get("W", 0) > 0
    _, v_self = dec_vector_konly(ps)
    assert ((v_self - v).norm() / v.norm()).item() < ((v2 - v).norm() / v.norm()).item()


def test_kvtc_offline_basis_roundtrip():
    blocks = [_kv(seed=s)[0] for s in range(3)]
    vblocks = [_kv(seed=s)[1] for s in range(3)]
    basis_k = fit_pca_basis(blocks, rank=32)   # (Vh, mu, S)
    basis_v = fit_pca_basis(vblocks, rank=32)
    k, v = _kv(seed=7)
    # fidelity at the DEFAULT budget
    p = enc_kvtc(k, v, basis_k=basis_k, basis_v=basis_v)
    k2, v2 = dec_kvtc(p, basis_k, basis_v, lead_shape=(H, L))
    assert k2.shape == k.shape
    assert ((k2 - k).norm() / k.norm()).item() < 0.3
    assert p.header["self_fit"] is False
    # allocation checks on a TIGHT budget so the DP actually zeroes the
    # trailing components
    pt = enc_kvtc(k, v, basis_k=basis_k, basis_v=basis_v,
                  budget_bytes=(H * L * D * 2) // 16)
    bits_k = pt.header["bits_k"]
    # S0.1: bits follow the singular values (monotone non-increasing) —
    # v1.1 weights were Vh row norms == 1, i.e. constant allocation
    assert len(set(bits_k)) > 1, bits_k
    assert all(bits_k[i] >= bits_k[i + 1] for i in range(len(bits_k) - 1)), bits_k
    # S0.2: trailing components take ZERO bits (the paper's DP behavior)
    assert min(bits_k) == 0, bits_k
    # and the zero-width payload still decodes
    kt, _ = dec_kvtc(pt, basis_k, basis_v, lead_shape=(H, L))
    assert kt.shape == k.shape and torch.isfinite(kt).all()


def test_aatc_budget_matched_and_decode():
    k, v = _kv()
    q = _q()
    ref = enc_kvtc(k, v, basis_k=fit_pca_basis([k], 32), basis_v=fit_pca_basis([v], 32))
    p = enc_aatc(k, v, q, target_bytes=ref.nbytes)
    # S0.2: lo=0 -> bits actually vary with sensitivity and stay in budget
    assert len(set(p.header["sens_bits"])) > 1
    assert sum(p.header["sens_bits"]) * H * L / 8 <= ref.nbytes * 1.02
    # byte-match against the kvtc budget (target_bytes is actually read):
    # packed payload bytes ~ H*L*sum(bits)/8 must land near the target
    packed = sum(n for name, n in p.part_bytes().items() if name.startswith(("k_w", "v_w")))
    assert 0.5 * ref.nbytes <= packed <= 1.3 * ref.nbytes
    k2, v2 = dec_aatc(p, lead_shape=(H, L))
    assert k2.shape == k.shape
    err1 = ((k2 - k).norm() / k.norm()).item()
    # sanity: more budget -> strictly better reconstruction (the width map
    # really drives the allocation); absolute fidelity at ~1.6 bits/element
    # is information-limited and checked on the real-data bench instead
    p2 = enc_aatc(k, v, q, target_bytes=ref.nbytes * 4)
    k3, _ = dec_aatc(p2, lead_shape=(H, L))
    err2 = ((k3 - k).norm() / k.norm()).item()
    assert err2 < err1


def test_channel_sensitivity_shape_and_gqa():
    k, v = _kv()
    q = _q()
    sens = channel_sensitivity(k, q, n_rep=4)
    assert sens.shape == (H, D)
    assert float(sens.min()) >= 0.0


def test_waterfill_meets_budget():
    w = torch.rand(128)
    bits = waterfill_bits(w, 128 * 4)
    assert 128 * 2 <= int(bits.sum()) <= 128 * 8
    assert int(bits.min()) >= 2 and int(bits.max()) <= 8
