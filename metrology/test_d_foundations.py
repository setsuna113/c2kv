# -*- coding: utf-8 -*-
"""CPU unit tests for the line-A foundations (torch required; skips without).

- bits.py: pack/unpack roundtrip for every width 1..8, odd lengths, exact
  byte sizes (the honest bytes axis), out-of-range rejection
- d_payload.py: serialize/deserialize roundtrip, exact nbytes accounting,
  shared-artifact amortization
- d_attn_ext.py: bias=0 bit-equality with the plain path; softmax(qk+log c)
  == c-weighted softmax and its NON-equivalence to V-scaling (the reason
  the extension point must be a logit bias); LESS num/den folding vs a
  closed-form toy; GQA mapping

Run:  pytest metrology/test_d_foundations.py -v
"""
from __future__ import annotations

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

from inference.bits import pack_bits, packed_bytes, unpack_bits  # noqa: E402
from d_payload import Payload, SharedArtifacts  # noqa: E402
from d_attn_ext import (  # noqa: E402
    attention_with_bias,
    kv_head_of,
    masked_attention_with_bias,
    repeat_kv_for_q,
)


class TestBits:
    @pytest.mark.parametrize("bits", list(range(1, 9)))
    def test_roundtrip_all_widths(self, bits):
        g = torch.Generator().manual_seed(bits)
        codes = torch.randint(0, 1 << bits, (377,), generator=g)
        buf = pack_bits(codes, bits)
        assert buf.numel() == packed_bytes(377, bits) == (377 * bits + 7) // 8
        back = unpack_bits(buf, bits, 377)
        assert torch.equal(codes, back)

    def test_exact_bytes_small(self):
        # 4 codes at 4 bits = exactly 2 bytes; 3 codes at 4 bits = 2 bytes (padded)
        assert pack_bits(torch.tensor([1, 2, 3, 4]), 4).numel() == 2
        assert pack_bits(torch.tensor([1, 2, 3]), 4).numel() == 2

    def test_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            pack_bits(torch.tensor([16]), 4)

    def test_multidim_shape(self):
        codes = torch.randint(0, 8, (4, 10, 6))
        back = unpack_bits(pack_bits(codes, 3), 3, codes.numel()).reshape(4, 10, 6)
        assert torch.equal(codes, back)


class TestPayload:
    def test_roundtrip_and_bytes(self):
        p = Payload("q4_v2")
        codes = torch.randint(0, 16, (8, 100, 32))
        scales = torch.randn(8, 100, 4)
        p.add_packed("k_codes", codes, 4)
        p.add_floats("k_scales", scales)
        assert p.nbytes >= (codes.numel() * 4 + 7) // 8  # stream dominates
        blob = p.serialize()
        q = Payload.deserialize(blob)
        assert torch.equal(q.read_packed("k_codes"), codes)
        back = q.read_floats("k_scales")
        assert back.shape == (8, 100, 4)
        assert torch.equal(back, scales.to(torch.float16))  # side arrays stored f16
        # nbytes == header + packed stream + float array, exactly
        parts = p.part_bytes()
        assert p.nbytes == sum(parts.values())
        assert parts["k_codes"] == (codes.numel() * 4 + 7) // 8
        assert parts["k_scales"] == scales.numel() * 2

    def test_shared_amortization(self):
        sh = SharedArtifacts()
        sh.put_bytes("pca_basis_L0", 10_000)
        sh.put_bytes("pca_basis_L1", 20_000)
        assert sh.total_bytes == 30_000
        assert sh.amortized_bytes(100) == 300.0
        assert sh.amortized_bytes(0) == 30_000.0


class TestAttnExt:
    def _qkv(self, seed=0, h=4, lq=7, lk=11, d=16, dv=8):
        g = torch.Generator().manual_seed(seed)
        return (torch.randn(h, lq, d, generator=g),
                torch.randn(h, lk, d, generator=g),
                torch.randn(h, lk, dv, generator=g))

    def test_zero_bias_bit_equal_plain_path(self):
        q, k, v = self._qkv()
        out = attention_with_bias(q, k, v)
        ref = torch.matmul(
            torch.softmax(torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5), dim=-1).float(),
            v.float())
        assert torch.equal(out, ref.to(out.dtype))

    def test_logit_bias_equals_multiplicative_key_weight(self):
        q, k, v = self._qkv(seed=1)
        g = torch.Generator().manual_seed(2)
        c = torch.rand(q.shape[0], k.shape[-2], generator=g) + 0.5   # (H, Lk)
        out = attention_with_bias(q, k, v, key_logit_bias=c.log())
        # closed form: weights ∝ c_j * exp(qk/sqrt d)
        logits = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        w = torch.softmax(logits + c.log().unsqueeze(1), dim=-1)
        ref = torch.matmul(w, v)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_logit_bias_NOT_equal_to_v_scaling(self):
        # the reason the extension must be a logit bias: c multiplies into
        # the denominator too, V-scaling does not
        q, k, v = self._qkv(seed=3)
        c = torch.rand(q.shape[0], k.shape[-2], generator=torch.Generator().manual_seed(4)) + 0.5
        via_logit = attention_with_bias(q, k, v, key_logit_bias=c.log())
        via_v = attention_with_bias(q, k, v * c.unsqueeze(-1))
        assert not torch.allclose(via_logit, via_v, atol=1e-3)

    def test_less_folding_closed_form(self):
        # o = (sum e^{qk} v + phi(q) H) / (sum e^{qk} + phi(q) z)
        q, k, v = self._qkv(seed=5)
        scale = 1.0 / (q.shape[-1] ** 0.5)
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale
        num = torch.exp(logits) @ v
        den = torch.exp(logits).sum(-1, keepdim=True)
        phi = torch.rand(q.shape[0], q.shape[-2], 1, generator=torch.Generator().manual_seed(6)) + 0.1
        H = torch.randn(q.shape[0], q.shape[-2], v.shape[-1], generator=torch.Generator().manual_seed(7))
        z = torch.rand(q.shape[0], q.shape[-2], 1, generator=torch.Generator().manual_seed(8)) + 0.5
        folded = attention_with_bias(q, k, v, extra_num=phi * H, extra_den=phi * z)
        ref = (num + phi * H) / (den + phi * z)
        assert torch.allclose(folded, ref, atol=1e-5)

    def test_causal_variant_and_gqa_mapping(self):
        q, k, v = self._qkv(seed=9, lq=8, lk=8)
        out = masked_attention_with_bias(q, k, v)
        scale = 1.0 / (q.shape[-1] ** 0.5)
        logits = torch.matmul(q, k.transpose(-1, -2)) * scale
        logits = logits.masked_fill(~torch.ones(8, 8, dtype=torch.bool).tril(), float("-inf"))
        ref = torch.softmax(logits, -1) @ v
        assert torch.allclose(out, ref, atol=1e-6)
        assert kv_head_of(7, 4) == 1 and kv_head_of(8, 4) == 2  # NOT q[:H_kv]
        k2, v2 = repeat_kv_for_q(k[:2], v[:2], 4)
        assert k2.shape[0] == 8 and torch.equal(k2[7], k[:2][1])
