"""Regression test: the sdpa gist path must not NaN on fully-masked rows.

2026-08-26 H200 incident: with --attn_impl sdpa the first real training step
(step 151 after resume) raised ``FloatingPointError: Non-finite loss detected:
nan, attn_impl=sdpa`` on both ranks. Root cause: stock transformers sdpa feeds
the additive mask straight into the fused kernel; a row whose keys are ALL
masked out. Whether that actually NaNs is backend-dependent: torch's CPU math
kernel zeroes such rows, the CUDA fused kernels we run in production do not —
which is exactly why the guard must live in our wrapper instead of relying on
kernel behavior. ``eager_attention_forward`` zeroes such rows (guard added in
490ba48); ``sdpa_attention_forward_guarded`` mirrors that semantics for the
fused path. CPU-only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

from models.qwen3.modeling_qwen3 import (  # noqa: E402
    eager_attention_forward,
    sdpa_attention_forward_guarded,
)


def _module():
    return SimpleNamespace(
        num_key_value_groups=1, attention_dropout=0.0, training=False, is_causal=False
    )


def _case(seq_len=6, fully_masked_row=2, dtype=torch.float32, neg=None):
    torch.manual_seed(0)
    q = torch.randn(1, 2, seq_len, 8, dtype=dtype)
    k = torch.randn(1, 2, seq_len, 8, dtype=dtype)
    v = torch.randn(1, 2, seq_len, 8, dtype=dtype)
    if neg is None:
        neg = torch.finfo(dtype).min
    mask = torch.zeros(1, 1, seq_len, seq_len, dtype=dtype)
    mask[0, 0, fully_masked_row, :] = neg  # row with no allowed key at all
    mask[0, 0, 4, 3:] = neg  # ordinary partially-masked row for realism
    return q, k, v, mask, fully_masked_row


def test_guarded_sdpa_matches_eager():
    for neg in (None, float("-inf")):  # finfo.min and -inf conventions both occur
        q, k, v, mask, fm_row = _case(neg=neg)
        m = _module()
        ref, _ = eager_attention_forward(m, q, k, v, mask, scaling=1.0)
        out, _ = sdpa_attention_forward_guarded(m, q, k, v, mask, scaling=1.0)
        assert torch.isfinite(out).all()
        assert out.shape == ref.shape  # both return [B, Q, H, D]
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)
        # the fully-masked row is exactly zero in both implementations
        assert out[0, fm_row].abs().sum() == 0
        assert ref[0, fm_row].abs().sum() == 0


def test_guarded_sdpa_gradients_match_eager():
    q, k, v, mask, _ = _case()
    m = _module()
    q1, k1, v1 = (t.clone().requires_grad_(True) for t in (q, k, v))
    ref, _ = eager_attention_forward(m, q1, k1, v1, mask, scaling=1.0)
    ref.sum().backward()
    q2, k2, v2 = (t.clone().requires_grad_(True) for t in (q, k, v))
    out, _ = sdpa_attention_forward_guarded(m, q2, k2, v2, mask, scaling=1.0)
    out.sum().backward()
    for ref_t, out_t in ((q1, q2), (k1, k2), (v1, v2)):
        assert torch.isfinite(out_t.grad).all()
        torch.testing.assert_close(ref_t.grad, out_t.grad, rtol=1e-4, atol=1e-5)


def test_guarded_sdpa_passthrough_when_no_fully_masked_row():
    # No fully-masked row -> outputs identical to stock sdpa (guard is a no-op).
    q, k, v, mask, _ = _case()
    mask = mask.clone()
    mask[0, 0, 2, :] = 0.0
    m = _module()
    stock, _ = ALL_ATTENTION_FUNCTIONS["sdpa"](m, q, k, v, mask, scaling=1.0)
    out, _ = sdpa_attention_forward_guarded(m, q, k, v, mask, scaling=1.0)
    torch.testing.assert_close(out, stock, rtol=0, atol=0)
