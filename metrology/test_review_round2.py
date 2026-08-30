# -*- coding: utf-8 -*-
"""Review round-2 regression tests (2026-08-31 findings A/B/C/D/F/G/H)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "python", _REPO_ROOT / "python" / "inference",
          _REPO_ROOT / "agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
for _stub in ("inference.mdocdataset", "inference"):
    sys.modules.pop(_stub, None)

from d_attn_ext import attention_with_bias, masked_attention_with_bias  # noqa: E402
from inference import attn_bias  # noqa: E402
from inference.bits import pack_bits  # noqa: E402
from d3_codecs_v2 import dec_kvtc, enc_kvtc, fit_pca_basis  # noqa: E402
from d67_v2 import selkv_logit_bias, selkv_mass_ratio  # noqa: E402

H, L, D, NREP = 8, 32, 128, 4


def _rand(shape, seed):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


class TestA_masked_denominator:
    def test_no_phantom_mass_from_masked_positions(self):
        q, k, v = _rand((H, L, D), 1), _rand((H, L, D), 2), _rand((H, L, D), 3)
        phi = torch.rand(H, L, 1, generator=torch.Generator().manual_seed(4)) * 0.05
        Hl = _rand((H, L, D), 5) * 0.05
        z = torch.rand(H, L, 1, generator=torch.Generator().manual_seed(6)) * 0.05 + 0.01
        out = masked_attention_with_bias(q, k, v, extra_num=phi * Hl, extra_den=phi * z)
        scale = 1.0 / math.sqrt(D)
        logits = (q @ k.transpose(-1, -2)) * scale
        mask = torch.ones(L, L, dtype=torch.bool).tril()
        logits = logits.masked_fill(~mask, float("-inf"))
        num = torch.exp(logits) @ v
        den = torch.exp(logits).sum(-1, keepdim=True)
        ref = (num + phi * Hl) / (den + phi * z)
        assert torch.allclose(out, ref, atol=1e-4), f"max|d|={(out - ref).abs().max()}"


class TestB_shift_semantics:
    def test_large_logits_no_nan_and_correct(self):
        q, k, v = _rand((H, L, D), 10), _rand((H, L, D), 11), _rand((H, L, D), 12)
        bias = torch.full((H, L), 180.0)
        out = attention_with_bias(q, k, v, key_logit_bias=bias)
        assert torch.isfinite(out).all()
        ref = attention_with_bias(q, k, v)
        assert torch.allclose(out, ref, atol=1e-4)  # constant bias cancels

    def test_extra_terms_in_unshifted_units(self):
        q, k, v = _rand((H, L, D), 20), _rand((H, L, D), 21), _rand((H, L, D), 22)
        bias = torch.full((H, L), 150.0)
        phi = torch.rand(H, L, 1, generator=torch.Generator().manual_seed(23)) * 0.1
        Hl = _rand((H, L, D), 24) * 0.1
        z = torch.rand(H, L, 1, generator=torch.Generator().manual_seed(25)) * 0.1 + 0.01
        out = attention_with_bias(q, k, v, key_logit_bias=bias,
                                  extra_num=phi * Hl, extra_den=phi * z)
        scale = 1.0 / math.sqrt(D)
        logits = ((q @ k.transpose(-1, -2)) * scale + 150.0).double()
        num = torch.exp(logits) @ v.double()
        den = torch.exp(logits).sum(-1, keepdim=True)
        ref = ((num + (phi * Hl).double()) / (den + (phi * z).double())).float()
        assert torch.isfinite(out).all()
        assert torch.allclose(out, ref, atol=1e-3)


class TestC_extra_den_only:
    def test_raises(self):
        q, k, v = _rand((H, L, D), 30), _rand((H, L, D), 31), _rand((H, L, D), 32)
        with pytest.raises(ValueError):
            attention_with_bias(q, k, v, extra_den=torch.ones(H, L))
        with pytest.raises(ValueError):
            masked_attention_with_bias(q, k, v, extra_den=torch.ones(H, L))


class TestD_eager_registry:
    def _replica(self, q, k, v, mask_add, layer_idx=0):
        """mirrors the patched eager path WITH its (B, H, Lq, Lk) shapes"""
        entry = attn_bias.get_entry(layer_idx)
        q4, k4, v4 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        w = q4 @ k4.transpose(-1, -2) / math.sqrt(D)
        if mask_add is not None:
            w = w + mask_add
        if entry is not None:
            w = entry.pre_softmax(w, 1)
        w = torch.softmax(w.float(), dim=-1).to(q.dtype)
        o = w @ v4
        if entry is not None:
            o = entry.post_output(o, q4, 1)
        return o[0]

    def test_empty_registry_bit_identical(self):
        attn_bias.clear()
        q, k, v = _rand((H, L, D), 40), _rand((H, L, D), 41), _rand((H, L, D), 42)
        mask_add = torch.zeros(1, 1, L, L)
        mask_add[..., torch.triu(torch.ones(L, L, dtype=torch.bool), 1)] = float("-inf")
        vanilla = torch.softmax(
            ((q @ k.transpose(-1, -2)) / math.sqrt(D) + mask_add).float(), -1) @ v
        assert torch.equal(self._replica(q, k, v, mask_add), vanilla[0])

    def test_key_bias_entry_matches_reference(self):
        attn_bias.clear()
        q, k, v = _rand((H, L, D), 43), _rand((H, L, D), 44), _rand((H, L, D), 45)
        bias = _rand((H, L), 46)
        attn_bias.set_entries({0: attn_bias.LayerBiasEntry(key_bias=bias)})
        got = self._replica(q, k, v, None)
        attn_bias.clear()
        ref = attention_with_bias(q, k, v, key_logit_bias=bias)
        assert torch.allclose(got, ref, atol=1e-5)

    def test_less_fold_entry_matches_reference(self):
        from d5_v2 import encode_less, build_layer_entries

        attn_bias.clear()
        q, k, v = _rand((H, L, D), 47), _rand((H, L, D), 48), _rand((H, L, D), 49)
        led = encode_less(k, v)
        attn_bias.set_entries(build_layer_entries({0: led}))
        got = self._replica(q, k, v, None)
        attn_bias.clear()
        phi = F.elu(q.float()) + 1.0
        ref = attention_with_bias(
            q, k, v,
            extra_num=torch.einsum("hld,hde->hle", phi, led["H"].float()),
            extra_den=torch.einsum("hld,hd->hl", phi, led["z"].float()))
        assert torch.allclose(got, ref, atol=1e-4), f"max|d|={(got - ref).abs().max()}"


class TestF_centered_pca:
    def test_offset_costs_nothing_extra(self):
        blocks = [_rand((1, 64, D), s) for s in (50, 51, 52)]
        (Vh, mu) = fit_pca_basis(blocks, rank=32)
        (Vv, muv) = fit_pca_basis([_rand((1, 64, D), 53)], rank=32)
        k = _rand((1, 64, D), 54)
        v = _rand((1, 64, D), 55)
        base = enc_kvtc(k, v, basis_k=(Vh, mu), basis_v=(Vv, muv))
        k2, _ = dec_kvtc(base, (Vh, mu), (Vv, muv), lead_shape=(1, 64))
        err0 = ((k2 - k).norm() / k.norm()).item()
        big = k + 100.0
        p1 = enc_kvtc(big, v, basis_k=(Vh, mu), basis_v=(Vv, muv))
        k3, _ = dec_kvtc(p1, (Vh, mu), (Vv, muv), lead_shape=(1, 64))
        err1 = ((k3 - big).norm() / big.norm()).item()
        assert err1 < err0 + 0.15, f"offset cost: {err0} -> {err1}"


class TestG_logspace_average:
    def test_ordering_and_safe_under_outliers(self):
        k_g = _rand((1, 4, D), 60) * 0.05
        k_g[0, 0] = 6.0 * torch.eye(D)[0]
        k_g[0, 1] = 0.3 * torch.eye(D)[0]
        k_r = _rand((1, L, D), 61) * 0.05
        q = torch.eye(D)[0].reshape(1, 1, D).repeat(NREP, L, 1)
        log_R = selkv_mass_ratio(k_g, k_r, q, n_rep=NREP)
        assert torch.isfinite(log_R).all()
        assert log_R[0, 1] > log_R[0, 0]
        b = selkv_logit_bias(log_R, alpha=0.5)
        assert torch.allclose(b, 0.5 * log_R)


class TestH_negative_codes:
    def test_pack_bits_rejects_negative(self):
        with pytest.raises(ValueError):
            pack_bits(torch.tensor([-1, 2]), 4)
