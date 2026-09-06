# -*- coding: utf-8 -*-
"""CPU test for D11 KVSculpt: the optimized r slots must beat a naive
mean-slot capsule of the SAME r on attention-output error, and the
optimization trace must actually descend.

Run:  pytest metrology/test_d11_kvsculpt.py -v   (needs torch; skips without)
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

from d11_kvsculpt import sculpt_block, sculpt_bytes  # noqa: E402
from d_attn_ext import attention_with_bias, masked_attention_with_bias, repeat_kv_for_q  # noqa: E402

H, L, D, NREP, R = 2, 48, 32, 4, 6


def _block(seed=0):
    g = torch.Generator().manual_seed(seed)
    # low-rank-ish structure so r slots can genuinely represent the block
    basis = torch.randn(4, D, generator=g)
    coef = torch.randn(H, L, 4, generator=g)
    k = torch.einsum("hlr,rd->hld", coef, basis) + 0.05 * torch.randn(H, L, D, generator=g)
    v = torch.einsum("hlr,rd->hld", coef, torch.randn(4, D, generator=g))
    q = torch.randn(H * NREP, L, D, generator=g) * 0.5
    return k, v, q


def _attn_out_err(k2, v2, k, v, q):
    kx, vx = repeat_kv_for_q(k, v, NREP)
    k2x, v2x = repeat_kv_for_q(k2, v2, NREP)
    o = masked_attention_with_bias(q, kx, vx)          # teacher: causal block
    o2 = attention_with_bias(q, k2x, v2x)              # student: r slots, no causal
    return ((o2 - o).norm() / o.norm()).item()


def test_sculpt_beats_naive_mean_slots():
    k, v, q = _block(1)
    cap = sculpt_block(k, v, q, R, n_rep=NREP, iters=80)
    # naive r-slot baseline: r evenly spaced mean slots, no optimization
    idx = torch.linspace(0, L - 1, R).long()
    k_mean = k[:, idx]
    v_mean = v[:, idx]
    err_sculpt = _attn_out_err(cap["k"], cap["v"], k, v, q)
    err_naive = _attn_out_err(k_mean, v_mean, k, v, q)
    assert err_sculpt < err_naive, (err_sculpt, err_naive)


def test_trace_descends_and_bytes_honest():
    k, v, q = _block(2)
    cap = sculpt_block(k, v, q, R, n_rep=NREP, iters=60)
    trace = cap["trace_first"]
    assert trace[-1] <= trace[0], trace
    assert sculpt_bytes(cap) == 8 + H * R * D * 2 * 2
