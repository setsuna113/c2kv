# -*- coding: utf-8 -*-
"""CPU unit tests for d67_v2 (GRKV closed-form edit + SelKV mass ratio) and
d4_capsules_v2 (k-means ResKV, vote-merged KeepKV, RESA ledgers,
same-bytes equalization).

Run:  pytest metrology/test_d47_v2.py -v   (needs torch; skips without)
"""
from __future__ import annotations

import math
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

from d67_v2 import (  # noqa: E402
    grkv_v_edit,
    grkv_writeback,
    selkv_count_bias,
    selkv_logit_bias,
    selkv_mass_ratio,
)
from d4_capsules_v2 import (  # noqa: E402
    capsule_bytes,
    decode_reskv,
    encode_keepkv,
    encode_resa,
    encode_reskv,
    equalize_r,
)
from d_attn_ext import attention_with_bias, repeat_kv_for_q  # noqa: E402

H, L, G, D, NREP = 8, 32, 4, 128, 4


def _rand(shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


class TestGRKV:
    def test_delta_reduces_output_error(self):
        k_g = _rand((H, G, D), 1)
        v_g = _rand((H, G, D), 2)
        q = _rand((H * NREP, L, D), 3)
        k_r = _rand((H, L, D), 4)
        v_r = _rand((H, L, D), 5)
        dv = grkv_v_edit(k_g, v_g, q, k_r, v_r, n_rep=NREP)
        assert dv.shape == (H, G, D) and torch.isfinite(dv).all()
        scale = 1.0 / math.sqrt(D)
        kx, _ = repeat_kv_for_q(k_g, v_g, NREP)
        kxr, vxr = repeat_kv_for_q(k_r, v_r, NREP)
        mask = torch.ones(L, L, dtype=torch.bool).tril()
        target = torch.softmax(((q @ kxr.transpose(-1, -2)) * scale).masked_fill(~mask, -math.inf), -1) @ vxr
        a_gist = torch.softmax((q @ kx.transpose(-1, -2)) * scale, -1)
        _, vx0 = repeat_kv_for_q(k_g, v_g, NREP)
        e0 = ((a_gist @ vx0 - target).norm() / target.norm()).item()
        _, vx1 = repeat_kv_for_q(k_g, v_g + dv, NREP)
        e1 = ((a_gist @ vx1 - target).norm() / target.norm()).item()
        assert e1 < e0, f"delta_v did not reduce error: {e0} -> {e1}"

    def test_writeback_adds_in_place(self):
        vals = torch.zeros(1, H, 10, D)
        grkv_writeback(vals, 3, 7, torch.ones(H, 4, D))
        assert vals[0, :, 3:7].abs().sum() > 0 and vals[0, :, :3].abs().sum() == 0


class TestSelKV:
    def test_mass_ratio_detects_oversubscribed_gist(self):
        # queries aligned with +e0; a gist token ALIGNED and STRONG absorbs
        # mass (small R), a weak gist token needs compensation (large R)
        k_g = _rand((1, G, D), 10) * 0.05
        k_g[0, 0] = 6.0 * torch.eye(D)[0]      # strong, aligned
        k_g[0, 1] = 0.3 * torch.eye(D)[0]      # weak, aligned
        k_r = _rand((1, L, D), 11) * 0.05      # weak random raw keys
        q = torch.eye(D)[0].reshape(1, 1, D).repeat(NREP, L, 1)
        R = selkv_mass_ratio(k_g, k_r, q, n_rep=NREP)
        assert R.shape == (1, G)
        assert R[0, 1] > R[0, 0], f"weak gist should need more compensation: {R}"

    def test_bias_and_control(self):
        # selkv_mass_ratio now returns LOG-space ratios (geometric mean);
        # selkv_logit_bias consumes them directly
        log_R = torch.tensor([[math.log(2.0), math.log(8.0)]])
        b = selkv_logit_bias(log_R, alpha=0.5)
        assert torch.allclose(b, 0.5 * log_R)
        c = selkv_count_bias(768, 2, alpha=0.5)
        assert torch.allclose(c, torch.full((2,), 0.5 * math.log(768)))


class TestCapsules:
    def test_reskv_logit_bias_not_v_scaling(self):
        k = _rand((1, L, D), 20)
        v = _rand((1, L, D), 21)
        cap = encode_reskv(k, v, r=4)
        ks, vs, bias = decode_reskv(cap)
        assert ks.shape == (1, 4, D)
        # the bias must enter attention as a logit (d_attn_ext), and that
        # must NOT equal scaling V by the counts
        q = _rand((NREP, 6, D), 22)
        via_logit = attention_with_bias(q, ks[0], vs[0], key_logit_bias=bias[0])
        via_v = attention_with_bias(q, ks[0], vs[0] * bias.exp()[0].unsqueeze(-1))
        assert not torch.allclose(via_logit, via_v, atol=1e-4)

    def test_keepkv_merges_and_uses_q(self):
        k = _rand((1, L, D), 30)
        v = _rand((1, L, D), 31)
        q = _rand((NREP, L, D), 32)
        cap = encode_keepkv(k, v, q, r=4)
        c = cap["caps"][0]
        assert c["k"].shape == (4, D)
        assert float(c["votes"].sum()) > 0
        # votes conserve total importance mass
        from d4_capsules_v2 import _attn_importance
        imp = _attn_importance(k, q, NREP, 1.0 / math.sqrt(D))
        assert abs(float(c["votes"].sum()) - float(imp.sum())) < 1e-3

    def test_resa_positive_ledgers(self):
        k = _rand((1, L, D), 40)
        v = _rand((1, L, D), 41)
        cap = encode_resa(k, v)
        assert (cap["z"] > 0).all(), "psi=elu+1 must keep z positive (v1 used identity)"

    def test_equalize_same_bytes(self):
        # small D so a 32-token block can actually reach resa's ledger
        # bytes with r <= L (a real block has L~768 and no such ceiling)
        d = 64
        k = _rand((1, L, d), 50)
        v = _rand((1, L, d), 51)
        q = _rand((NREP, L, d), 52)
        eq = equalize_r(k, v, q)
        b_resa = capsule_bytes(encode_resa(k, v))
        assert eq["reskv_r"] is not None and eq["keepkv_r"] is not None, eq
        b1 = capsule_bytes(encode_reskv(k, v, eq["reskv_r"]))
        b2 = capsule_bytes(encode_keepkv(k, v, q, eq["keepkv_r"]))
        assert abs(b1 - b_resa) <= 0.02 * b_resa or b1 >= b_resa
        assert abs(b2 - b_resa) <= 0.02 * b_resa or b2 >= b_resa
