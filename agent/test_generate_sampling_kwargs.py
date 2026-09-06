# -*- coding: utf-8 -*-
"""The shared sampling switch on ``_generate_from_input_ids``.

The whole point of the switch is that the DEFAULT path is untouched: every
existing eval calls ``_generate_from_input_ids`` without the sampling
arguments and must keep producing exactly today's ``generate`` kwargs.  These
tests capture the kwargs with a fake ``model.generate`` and compare the dict
literally, then check that ``temperature``/``top_p`` appear only when
``do_sample`` is on.

Also covers the per-row seeding formula shared with the history eval:
``manual_seed((gen_seed * 1_000_003) ^ crc32(f"{qid}:{mode}:{ratio}"))``.

Run from the repo root:
  python -m pytest agent/test_generate_sampling_kwargs.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "python/inference", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def decode(self, ids, skip_special_tokens=True):
        return "fake prediction"


class _CapturingModel:
    """No ``.model`` attribute on purpose: the attn-impl swap is then skipped."""

    def __init__(self):
        self.captured = None

    def generate(self, **kwargs):
        import torch

        self.captured = dict(kwargs)
        input_ids = kwargs["input_ids"]
        return torch.cat([input_ids, input_ids.new_full((1, 3), 7)], dim=1)


def _call(**overrides):
    import torch

    from eval_agent_tool_definition_c2kv import _generate_from_input_ids

    model = _CapturingModel()
    tokenizer = _FakeTokenizer()
    input_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    prediction, latency, generated_tokens, tbt = _generate_from_input_ids(
        model,
        tokenizer,
        input_ids=input_ids,
        max_new_tokens=16,
        attn_impl="eager",
        **overrides,
    )
    assert prediction == "fake prediction"
    assert generated_tokens == 3
    assert latency >= 0.0 and tbt >= 0.0
    return model.captured, input_ids


def test_default_kwargs_are_unchanged():
    import torch

    captured, input_ids = _call()
    assert set(captured) == {
        "input_ids",
        "attention_mask",
        "max_new_tokens",
        "do_sample",
        "pad_token_id",
        "eos_token_id",
        "use_cache",
    }
    assert captured["do_sample"] is False
    assert captured["max_new_tokens"] == 16
    assert captured["pad_token_id"] == 0
    assert captured["eos_token_id"] == 2
    assert captured["use_cache"] is True
    assert torch.equal(captured["input_ids"], input_ids)
    assert torch.equal(captured["attention_mask"], torch.ones_like(input_ids))


def test_default_path_ignores_sampling_values_when_do_sample_is_off():
    # Passing temperature/top_p without --do_sample must NOT change the dict:
    # a stray value in an ops script cannot silently perturb a greedy arm.
    captured, _ = _call(do_sample=False, temperature=0.7, top_p=0.9)
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert captured["do_sample"] is False


def test_do_sample_injects_temperature_and_top_p():
    captured, _ = _call(do_sample=True, temperature=0.7, top_p=0.9)
    assert captured["do_sample"] is True
    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.9


def test_do_sample_with_none_values_injects_nothing():
    captured, _ = _call(do_sample=True, temperature=None, top_p=None)
    assert captured["do_sample"] is True
    assert "temperature" not in captured
    assert "top_p" not in captured
    captured, _ = _call(do_sample=True, top_p=0.95)
    assert "temperature" not in captured
    assert captured["top_p"] == 0.95


def test_optional_generation_inputs_still_pass_through():
    import torch

    captured, _ = _call(
        use_gist=True,
        position_ids=torch.arange(4).unsqueeze(0),
        past_key_values="sentinel-cache",
        do_sample=True,
        temperature=0.3,
    )
    assert captured["use_gist"] is True
    assert captured["past_key_values"] == "sentinel-cache"
    assert captured["position_ids"].shape == (1, 4)
    assert captured["temperature"] == 0.3


def test_sampling_args_are_keyword_only():
    import torch

    from eval_agent_tool_definition_c2kv import _generate_from_input_ids

    with pytest.raises(TypeError):
        # Positional after past_key_values must not silently land on do_sample.
        _generate_from_input_ids(
            _CapturingModel(),
            _FakeTokenizer(),
            torch.tensor([[1, 2]], dtype=torch.long),
            8,
            "eager",
            False,
            None,
            None,
            True,
        )


def test_per_row_seed_formula():
    import zlib
    from types import SimpleNamespace

    import torch

    from eval_joint_next_action_c2kv import _seed_row

    args = SimpleNamespace(gen_seed=3, override_ratio=8)
    example = SimpleNamespace(qid="s0:1")

    def _draw():
        return torch.rand(4).tolist()

    _seed_row(args, example, "c2kv")
    first = _draw()
    _seed_row(args, example, "c2kv")
    assert _draw() == first, "same (gen_seed, qid, mode, ratio) must replay"

    _seed_row(args, SimpleNamespace(qid="s0:2"), "c2kv")
    assert _draw() != first, "a different qid must not reuse the same stream"

    _seed_row(args, example, "full")
    assert _draw() != first, "a different mode must not reuse the same stream"

    _seed_row(SimpleNamespace(gen_seed=4, override_ratio=8), example, "c2kv")
    assert _draw() != first, "a different gen_seed must not reuse the same stream"

    # Formula is the cross-line contract with eval_agent_history_c2kv.
    expected = (3 * 1_000_003) ^ zlib.crc32(b"s0:1:c2kv:8")
    torch.manual_seed(expected)
    reference = _draw()
    _seed_row(args, example, "c2kv")
    assert _draw() == reference
