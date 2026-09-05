# -*- coding: utf-8 -*-
"""t33 capture switch contract (survey item 4.0-2).

The load-bearing property: with ``capture=None`` the generate kwargs dict is
LITERALLY unchanged — the battery rerun must be the frozen battery plus
instrumentation, not a new path.  With capture on, the ONLY additions are
``output_scores``/``return_dict_in_generate`` (numerics of greedy sampling are
unaffected), and sampling + capture together are refused.

Run on the eval side (torch required):
  python -m pytest agent/test_t33_capture_kwargs.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "python/inference", "agent"):
    _p = str(_REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def decode(self, ids, skip_special_tokens=True):
        return "fake prediction"


class _CapturingModel:
    def __init__(self):
        self.captured = None

    def generate(self, **kwargs):
        import torch

        self.captured = dict(kwargs)
        input_ids = kwargs["input_ids"]
        cont = torch.cat([input_ids, input_ids.new_full((1, 3), 7)], dim=1)
        if not kwargs.get("return_dict_in_generate"):
            return cont

        class _Out:
            sequences = cont
            scores = tuple(torch.zeros(1, 10) for _ in range(3))

        return _Out()


def _call(**overrides):
    import torch

    from eval_agent_tool_definition_c2kv import _generate_from_input_ids

    model = _CapturingModel()
    tokenizer = _FakeTokenizer()
    input_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    result = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=16,
        attn_impl="eager",
        **overrides,
    )
    return model.captured, input_ids, result


def test_default_kwargs_are_unchanged():
    captured, _ids, result = _call()
    assert captured == {
        "input_ids": captured["input_ids"],
        "attention_mask": captured["attention_mask"],
        "max_new_tokens": 16,
        "do_sample": False,
        "pad_token_id": 0,
        "eos_token_id": 2,
        "use_cache": True,
    }
    assert set(captured) == {
        "input_ids", "attention_mask", "max_new_tokens", "do_sample",
        "pad_token_id", "eos_token_id", "use_cache",
    }
    assert isinstance(result, tuple) and len(result) == 4


def test_capture_adds_only_score_kwargs():
    capture: dict = {}
    captured, _ids, result = _call(capture=capture)
    assert set(captured) == {
        "input_ids", "attention_mask", "max_new_tokens", "do_sample",
        "pad_token_id", "eos_token_id", "use_cache",
        "output_scores", "return_dict_in_generate",
    }
    assert captured["output_scores"] is True
    assert captured["return_dict_in_generate"] is True
    assert len(result) == 4  # return shape unchanged


def test_capture_refuses_sampling():
    with pytest.raises(ValueError, match="greedy"):
        _call(do_sample=True, temperature=0.7, capture={})
