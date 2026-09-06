# -*- coding: utf-8 -*-
"""D1 gate: RoPE repositioning unit tests (24号 B.4.1 / 判据0).

Changing a chunk boundary changes the INPUT to the gist position bookkeeping,
so every B arm is frozen until these five properties hold.  Nothing here needs
weights or a dataset: a tiny randomly-initialised Qwen3 with the training gist
config is enough, and everything runs on CPU in fp32.

Properties:
1. ``rotate_k_cache_rope(k, delta_pos=0, ...)`` returns its input — the
   NameError regression at python/inference/rope_reposition.py:49.
2. store-then-rotate == direct: an UNROTATED gist K rotated to position p in
   one shot equals the same K rotated at q and then repositioned by p-q;
   the +d/-d round trip is the identity.
3. + 4. position accounting: three unequal-length chunks through
   ``gist_utils.process_context_input_ids`` — every gist's global position id
   must equal ``chunk-local end position + Sum(preceding chunk original
   lengths) + past_length``, on BOTH branches (path1 =
   ``reconstruct_kwargs=None``, gist_utils.py:592-601; path2 =
   ``reconstruct_kwargs={}`` in eval mode, gist_utils.py:673-687).
5. blend/concat consistency: compressing the tool grid at ``system_length``
   and the history grid at ``system_length + tool_doc_tokens`` and
   concatenating gives layer-wise identical K/V to compressing one joint grid
   — the property agent/eval_joint_next_action_c2kv.py:52-60 asserts in prose.

Run from the repo root:
  python -m pytest python/models/test_rope_reposition.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("python", "python/inference"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

RATIO = 8
# Every chunk length in the blend test is a multiple of RATIO on purpose: the
# embed-mean gist residual (gist_utils.py:426-436) averages over the FULL
# padded row, so a partial trailing group would mix grid padding into the last
# gist and the two constructions would differ for a reason that has nothing to
# do with position bookkeeping.
BLEND_TOOL_LENGTHS = (16, 8)
BLEND_HISTORY_LENGTHS = (24, 8, 16)


@pytest.fixture(autouse=True)
def pinned_ratio(monkeypatch):
    """Pin the dynamic-interleave sampler so the tests are deterministic."""
    monkeypatch.setenv("C2KV_GIST_TRAIN_RATIOS", str(RATIO))


def _tiny_model():
    import torch

    from models.qwen3 import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rope_theta=1e6,
        rms_norm_eps=1e-6,
        gist_type="dynamic-interleave",
        gist_param="qkv",
        gist_residual_type="embed-mean",
        gist_overlap=4,
        gist_extra_embed_num=2,
        gist_token_id=0,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def _grid(lengths, width=None, seed=0):
    """(1, num_chunks, width) context grid, -100 padded, right aligned to 0."""
    import torch

    width = width or max(lengths)
    generator = torch.Generator().manual_seed(seed)
    grid = torch.full((1, len(lengths), width), -100, dtype=torch.long)
    for index, length in enumerate(lengths):
        grid[0, index, :length] = torch.randint(
            1, 128, (length,), generator=generator, dtype=torch.long
        )
    return grid


def _expected_gist_positions(lengths, past_length, ratio=RATIO):
    """chunk-local end position + Sum(preceding original lengths) + past_length."""
    rows = []
    prefix = past_length
    for length in lengths:
        rows.append([min((j + 1) * ratio, length) - 1 + prefix
                     for j in range(math.ceil(length / ratio))])
        prefix += length
    return rows


def _wrap_rotary(model):
    """Swap in an nn.Module that records every position_ids it is called with."""
    import torch

    class _Recorder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.calls = []

        def forward(self, x, position_ids):
            self.calls.append(position_ids.detach().clone())
            return self.inner(x, position_ids)

    recorder = _Recorder(model.model.rotary_emb)
    model.model.rotary_emb = recorder
    return recorder


# ---------------------------------------------------------------------------
# 1. zero-delta regression
# ---------------------------------------------------------------------------


def test_zero_delta_returns_input():
    import torch

    from rope_reposition import rotate_k_cache_rope

    k_cache = torch.randn(2, 5, 16)
    result = rotate_k_cache_rope(k_cache, 0, 1e6, "default")
    # Regression: this branch used to reference an undefined `kv_cache`.
    assert result is k_cache
    assert torch.equal(result, k_cache)


# ---------------------------------------------------------------------------
# 2. store-then-rotate == direct
# ---------------------------------------------------------------------------


def test_store_then_rotate_equals_direct():
    import torch

    from models.gist_utils import apply_rotary_pos_emb
    from rope_reposition import rotate_k_cache_rope

    model = _tiny_model()
    rotary = model.model.rotary_emb
    theta = float(model.config.rope_theta)
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    seq_len = 6
    torch.manual_seed(1)
    # Gist K/V are stored WITHOUT RoPE (modeling_qwen3.py:319-320); this is
    # that stored tensor.
    stored = torch.randn(1, num_kv_heads, seq_len, head_dim, dtype=torch.float32)

    def _rotate_at(start):
        positions = torch.arange(start, start + seq_len, dtype=torch.long).unsqueeze(0)
        cos, sin = rotary(stored, positions)
        return apply_rotary_pos_emb(stored, cos, sin)

    target, source = 137, 41
    direct = _rotate_at(target)
    two_step = rotate_k_cache_rope(_rotate_at(source)[0], target - source, theta, "default")
    assert torch.allclose(direct[0], two_step, atol=1e-5, rtol=1e-5)

    # +d then -d is the identity.
    delta = 91
    there = rotate_k_cache_rope(direct[0], delta, theta, "default")
    back = rotate_k_cache_rope(there, -delta, theta, "default")
    assert torch.allclose(back, direct[0], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 3. + 4. position accounting through process_context_input_ids
# ---------------------------------------------------------------------------


def _run_position_accounting(reconstruct_kwargs):
    import torch

    from models.gist_utils import process_context_input_ids

    lengths = [12, 20, 7]
    past_length = 5
    prompt_length = 4

    model = _tiny_model()
    recorder = _wrap_rotary(model)

    system_ids = torch.randint(1, 128, (1, past_length), dtype=torch.long)
    with torch.inference_mode():
        system_out = model(
            system_ids, attention_mask=torch.ones_like(system_ids), use_cache=True
        )
    cache = system_out.past_key_values
    assert cache.get_seq_length() == past_length

    context_input_ids = _grid(lengths, seed=7)
    attention_mask = torch.ones((1, prompt_length), dtype=torch.long)
    position_ids = torch.arange(prompt_length, dtype=torch.long).unsqueeze(0)
    recorder.calls.clear()
    with torch.inference_mode():
        process_context_input_ids(
            model.model,
            context_input_ids,
            cache,
            attention_mask,
            position_ids,
            reconstruct_kwargs=reconstruct_kwargs,
        )
    # The LAST rotary_emb call is the one that rotates the gist keys to their
    # global positions (generate_gist's own calls come first).
    recorded = recorder.calls[-1]
    assert recorded.shape[0] == len(lengths)

    expected = _expected_gist_positions(lengths, past_length)
    for row, row_expected in enumerate(expected):
        got = recorded[row, : len(row_expected)].tolist()
        assert got == row_expected, (
            f"chunk {row}: gist positions {got} != "
            f"chunk-local end + preceding lengths + past_length {row_expected}"
        )
    return recorded


def test_position_accounting_path1():
    # reconstruct_kwargs=None -> the flat/valid-indices branch (:592-601).
    recorded = _run_position_accounting(None)
    assert recorded.tolist()[0][:2] == [12, 16]


def test_position_accounting_path2():
    # reconstruct_kwargs={} in eval mode -> the reshape branch (:673-687).
    # (model.training is False, so no reconstruction loss is attempted.)
    recorded = _run_position_accounting({})
    assert recorded.tolist()[1][:3] == [24, 32, 36]


def test_position_accounting_paths_agree():
    path1 = _run_position_accounting(None)
    path2 = _run_position_accounting({})
    assert path1.tolist() == path2.tolist()


# ---------------------------------------------------------------------------
# 5. blend/concat consistency
# ---------------------------------------------------------------------------


def _compress_grid(model, grid, prefix_length):
    """_compress_docs_to_cache (eval_joint_next_action_c2kv.py:533-590) inline."""
    import torch

    from models import blend_gist_key_values

    valid_mask = grid != -100
    doc_tokens = int(valid_mask.sum().item())
    input_ids = grid.clone()
    input_ids[~valid_mask] = model.model.gist_token_id
    with torch.inference_mode():
        outputs, gist_mask, pos_ids = model.model.generate_gist(
            input_ids=input_ids,
            attention_mask=valid_mask,
            ratio=RATIO,
        )
        cache, _ = blend_gist_key_values(
            model.config,
            [outputs.past_key_values],
            [gist_mask],
            # _concat_gist_key_values mutates the position ids in place.
            [pos_ids.clone()],
            model.model.rotary_emb,
            prefix_length,
        )
    return cache, doc_tokens


def test_blend_concat_consistency():
    import torch

    model = _tiny_model()
    system_length = 5

    tool_grid = _grid(BLEND_TOOL_LENGTHS, seed=11)[0].unsqueeze(0)
    history_grid = _grid(BLEND_HISTORY_LENGTHS, seed=12)[0].unsqueeze(0)
    width = max(tool_grid.shape[-1], history_grid.shape[-1])

    def _widen(grid):
        pad = width - grid.shape[-1]
        if pad <= 0:
            return grid
        return torch.cat([grid, grid.new_full((*grid.shape[:-1], pad), -100)], dim=-1)

    # Two-sided: tool at system_length, history right after the tool tokens.
    tool_cache, tool_doc_tokens = _compress_grid(model, tool_grid[0], system_length)
    assert tool_doc_tokens == sum(BLEND_TOOL_LENGTHS)
    history_cache, _ = _compress_grid(
        model, history_grid[0], system_length + tool_doc_tokens
    )

    # Single joint grid, tool chunks first.
    joint_grid = torch.cat([_widen(tool_grid)[0], _widen(history_grid)[0]], dim=0)
    joint_cache, joint_doc_tokens = _compress_grid(model, joint_grid, system_length)
    assert joint_doc_tokens == sum(BLEND_TOOL_LENGTHS) + sum(BLEND_HISTORY_LENGTHS)

    assert joint_cache.get_seq_length() == (
        tool_cache.get_seq_length() + history_cache.get_seq_length()
    )
    for layer_index, joint_layer in enumerate(joint_cache.layers):
        side_keys = torch.cat(
            [tool_cache.layers[layer_index].keys, history_cache.layers[layer_index].keys],
            dim=-2,
        )
        side_values = torch.cat(
            [tool_cache.layers[layer_index].values, history_cache.layers[layer_index].values],
            dim=-2,
        )
        assert torch.allclose(joint_layer.keys, side_keys, atol=1e-5, rtol=1e-5), (
            f"layer {layer_index}: two-sided blend K != joint-grid K"
        )
        assert torch.allclose(joint_layer.values, side_values, atol=1e-5, rtol=1e-5), (
            f"layer {layer_index}: two-sided blend V != joint-grid V"
        )
