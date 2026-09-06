"""Regression test: npu_fusion_attention must not lose the causal mask.

transformers 5.x builds causal masks through ALL_MASK_ATTENTION_FUNCTIONS;
an unregistered impl silently gets `None` (bidirectional attention, i.e.
teacher-forced label leakage).  `models.npu_attention` registers the eager
mask factory for "npu_fusion_attention" at import time — this test guards
that registration.  CPU-only: no torch_npu needed (the fusion kernel itself
is not exercised here).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")  # torch-free collection guard (Windows dev box has no torch)

import torch  # noqa: E402
from transformers.configuration_utils import PretrainedConfig  # noqa: E402
from transformers.masking_utils import (  # noqa: E402
    ALL_MASK_ATTENTION_FUNCTIONS,
    create_causal_mask,
)

_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import models.npu_attention  # noqa: E402, F401  (import performs the registration)


def _mask_for(impl: str, seq_len: int = 5):
    config = PretrainedConfig()
    config._attn_implementation = impl
    inputs_embeds = torch.zeros(1, seq_len, 8)
    attention_mask = torch.ones(1, seq_len, dtype=torch.long)
    return create_causal_mask(config, inputs_embeds, attention_mask, past_key_values=None)


def test_registration_present():
    assert "npu_fusion_attention" in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping


def test_causal_mask_not_dropped():
    mask = _mask_for("npu_fusion_attention")
    assert mask is not None, "causal mask dropped for npu_fusion_attention (label leakage!)"
    assert mask.shape == (1, 1, 5, 5)
    # Upper triangle (future) must be hugely negative, lower triangle + diagonal 0.
    # (transformers fills dtype-min, not -inf, so compare against a threshold.)
    assert mask[0, 0, 0, 1] < -1e30
    assert mask[0, 0, 3, 4] < -1e30
    assert mask[0, 0, 4, 0] == 0
    assert mask[0, 0, 2, 2] == 0


def test_to_npu_attention_mask_converts_float_causal_to_bool_drop():
    mask = _mask_for("npu_fusion_attention")
    bool_mask = models.npu_attention._to_npu_attention_mask(mask)
    assert bool_mask.dtype == torch.bool
    # True marks dropped (masked-out) positions: strictly upper triangle.
    assert bool_mask[0, 0, 0, 1].item() is True
    assert bool_mask[0, 0, 1, 0].item() is False
    assert bool_mask[0, 0, 4, 4].item() is False
