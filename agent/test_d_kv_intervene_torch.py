# -*- coding: utf-8 -*-
"""Torch-backed tests for the task-D KV interventions in
agent/eval_agent_history_c2kv.py.

Everything runs on CPU against a two-layer randomly initialised Qwen3 gist
model and the deterministic whitespace tokenizer (same recipe as
agent/test_eval_joint_next_action_c2kv.py:544 onward).  No weights, no
dataset, no network.

torch is declared optional at module scope; every repo import that pulls
torch happens inside a test or fixture so collection succeeds on machines
without it.

Coverage:
a. _append_span_cache — index selection and the RoPE rotation, plus the
   delta_pos == 0 short-circuit that made the guard necessary;
b. _append_precomputed_span_cache — appends unrotated, empty span is a no-op;
c. the closed-form gist-token count vs. what _build_tool_cache really emits;
d. truncated grid == full grid on the shared rows (the single-variable claim
   behind E-corr+re);
e. the E-corr slice equals the matching slot range of
   _build_sequential_full_prefix;
f. no-gist recompute rebuilds the sequential-full cache bit for bit, and the
   next-token logits agree;
g. _prefill_tokens_with_cache_maybe_gist mask/position semantics when the
   cache holds FEWER slots than there are logical positions;
h. d_sham_mech is an exact identity against plain c2kv;
i. every arm keeps the original decode layout (history_length unchanged).

Run from the repo root:
  python -m pytest agent/test_d_kv_intervene_torch.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "python/inference", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _StubLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _StubCache:
    """Minimal cache surface used by the append helpers (.layers/.keys/.values)."""

    def __init__(self, layers):
        self.layers = list(layers)

    def get_seq_length(self):
        return self.layers[0].keys.shape[-2]


def _harness():
    import eval_agent_history_c2kv as HH  # noqa: PLC0415

    return HH


def _tokenizer():
    from train.train_data_joint import _WhitespaceSelfTestTokenizer  # noqa: PLC0415

    return _WhitespaceSelfTestTokenizer()


def _tiny_model(tokenizer):
    from models.qwen3 import Qwen3Config, Qwen3ForCausalLM  # noqa: PLC0415

    config = Qwen3Config(
        vocab_size=1024,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        gist_type="dynamic-interleave",
        gist_param="qkv",
        gist_residual_type="embed-mean",
        gist_overlap=8,
        gist_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config)
    model.eval()
    return model


def _args():
    HH = _harness()
    argv = [
        "prog",
        "--model", "./checkpoints/tiny",
        "--tokenizer", "./checkpoints/tiny",
        "--device_type", "cpu",
        "--max_doc_length", "32",
        "--max_doc_num", "8",
        "--min_doc_num", "1",
        "--max_history_tokens", "4096",
        "--max_system_length", "64",
        "--max_prompt_tokens", "64",
        "--max_new_tokens", "4",
        "--override_ratio", "4",
        "--system_attn_impl", "eager",
        "--gist_attn_impl", "eager",
        "--generate_attn_impl", "eager",
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        return HH.parse_args()
    finally:
        sys.argv = saved


def _example():
    from train.train_data_multiturn import CompressHistoryExample  # noqa: PLC0415

    turns = [
        "the wind moved sand across the wide flat plain",
        "grains hopped forward and struck the bed again",
        "a small mound grew behind the sheltering stone",
        "the leeward slope reached its resting angle",
        "an avalanche carried grains down to the base",
    ]
    history = []
    for index, text in enumerate(turns):
        history.append({"role": "user" if index % 2 == 0 else "assistant", "content": text})
    return CompressHistoryExample(
        qid="dune:7",
        history_messages=history,
        current_messages=[{"role": "user", "content": "which way does the ridge travel"}],
        answer="downwind along the resultant direction",
        system_prompt="you describe landforms",
        tools=[],
    )


@pytest.fixture(scope="module")
def stack():
    tokenizer = _tokenizer()
    model = _tiny_model(tokenizer)
    return model, tokenizer, _args(), _example()


def _doc_layout(HH, tokenizer, example, args):
    """(doc_ids, offsets, k_star) with offsets in absolute logical positions."""
    history = HH._history_messages(tokenizer, example, args)
    doc_ids = [
        HH._chat_template_ids(tokenizer, [message], max_length=args.max_doc_length)
        for message in history
    ]
    system_ids = HH._chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    offsets = []
    offset = len(system_ids)
    for ids in doc_ids:
        offsets.append(offset)
        offset += len(ids)
    return doc_ids, offsets, (len(doc_ids) - 1) // 2


def _assert_cache_equal(left, right, label):
    for index, (left_layer, right_layer) in enumerate(zip(left.layers, right.layers)):
        assert torch.equal(left_layer.keys, right_layer.keys), (
            f"{label}: layer {index} keys differ, max |d| = "
            f"{(left_layer.keys - right_layer.keys).abs().max().item()}"
        )
        assert torch.equal(left_layer.values, right_layer.values), (
            f"{label}: layer {index} values differ, max |d| = "
            f"{(left_layer.values - right_layer.values).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# a/b. append primitives
# ---------------------------------------------------------------------------


def test_append_span_cache_rotates_and_selects():
    HH = _harness()
    from rope_reposition import rotate_k_cache_rope  # noqa: PLC0415

    torch.manual_seed(1)
    prefix = _StubCache([_StubLayer(torch.randn(1, 2, 5, 8), torch.randn(1, 2, 5, 8))])
    doc_keys = torch.randn(1, 2, 6, 8)
    doc_values = torch.randn(1, 2, 6, 8)
    doc = _StubCache([_StubLayer(doc_keys.clone(), doc_values.clone())])
    model = _tiny_model(_tokenizer())
    theta, rope_type = HH._model_rope_params(model)

    span = [1, 3, 4]
    start = 17
    out = HH._append_span_cache(model, prefix, doc, start, span)
    assert out.layers[0].keys.shape[-2] == 5 + len(span)

    expected_keys = rotate_k_cache_rope(doc_keys[0], start, theta, rope_type)
    index = torch.tensor(span, dtype=torch.long)
    assert torch.equal(
        out.layers[0].keys[0, :, 5:, :], expected_keys.index_select(1, index)
    )
    assert torch.equal(
        out.layers[0].values[0, :, 5:, :], doc_values[0].index_select(1, index)
    )
    # Rotation actually happened: the appended keys are not the raw slice.
    assert not torch.equal(out.layers[0].keys[0, :, 5:, :], doc_keys[0].index_select(1, index))


def test_rotation_is_a_noop_at_position_zero():
    """rope_reposition short-circuits delta_pos == 0, which is why the task-D
    builder asserts doc k* starts after the system prefix."""
    from rope_reposition import rotate_k_cache_rope  # noqa: PLC0415

    torch.manual_seed(2)
    keys = torch.randn(2, 4, 8)
    assert torch.equal(rotate_k_cache_rope(keys, 0, 10000.0, "default"), keys)
    assert not torch.equal(rotate_k_cache_rope(keys, 1, 10000.0, "default"), keys)


def test_append_precomputed_span_cache_does_not_rotate():
    HH = _harness()
    torch.manual_seed(3)
    keys = torch.randn(1, 2, 5, 8)
    values = torch.randn(1, 2, 5, 8)
    prefix = _StubCache([_StubLayer(keys.clone(), values.clone())])
    span_keys = torch.randn(1, 2, 3, 8)
    span_values = torch.randn(1, 2, 3, 8)
    out = HH._append_precomputed_span_cache(prefix, [(span_keys, span_values)])
    assert torch.equal(out.layers[0].keys[..., 5:, :], span_keys)
    assert torch.equal(out.layers[0].values[..., 5:, :], span_values)
    assert torch.equal(out.layers[0].keys[..., :5, :], keys)


def test_append_precomputed_span_cache_empty_is_identity():
    HH = _harness()
    torch.manual_seed(4)
    keys = torch.randn(1, 2, 5, 8)
    values = torch.randn(1, 2, 5, 8)
    prefix = _StubCache([_StubLayer(keys, values)])
    out = HH._append_precomputed_span_cache(prefix, [])
    assert out.layers[0].keys is keys
    assert out.layers[0].values is values


# ---------------------------------------------------------------------------
# c/d. gist accounting and the truncated grid
# ---------------------------------------------------------------------------


def test_gist_count_model_matches_build_tool_cache(stack):
    HH = _harness()
    from eval_agent_tool_definition_c2kv import _build_tool_cache  # noqa: PLC0415

    model, tokenizer, args, example = stack
    doc_ids, _, _ = _doc_layout(HH, tokenizer, example, args)
    grid = HH._grid_from_doc_ids(doc_ids, args.max_doc_length, args.max_doc_num)
    system_ids = HH._chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_cache, system_length, _ = HH._prefill_system(
        model, torch.tensor([system_ids], dtype=torch.long), args.system_attn_impl
    )
    _, tool_length, gist_tokens, _, _, _ = _build_tool_cache(
        model, grid, system_cache, system_length, args.gist_attn_impl, args.override_ratio
    )
    assert tool_length == sum(len(ids) for ids in doc_ids)
    predicted = HH._gist_tokens_for_lengths(
        [len(ids) for ids in doc_ids],
        args.override_ratio,
        model.config.gist_residual_type,
        args.max_doc_length,
    )
    assert predicted == gist_tokens


def test_truncated_grid_gist_matches_full_grid(stack):
    """Grid rows are the compression batch dimension: a doc's gist does not
    depend on the other rows. This is what makes E-corr and E-corr+re share a
    bit-identical upstream."""
    HH = _harness()
    from eval_agent_tool_definition_c2kv import _build_tool_cache  # noqa: PLC0415

    model, tokenizer, args, example = stack
    doc_ids, _, k_star = _doc_layout(HH, tokenizer, example, args)
    system_ids = HH._chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    system_input_ids = torch.tensor([system_ids], dtype=torch.long)

    caches = {}
    for label, ids in (("full", doc_ids), ("trunc", doc_ids[: k_star + 1])):
        system_cache, system_length, _ = HH._prefill_system(
            model, system_input_ids, args.system_attn_impl
        )
        grid = HH._grid_from_doc_ids(ids, args.max_doc_length, args.max_doc_num)
        cache, _, gist_tokens, _, _, _ = _build_tool_cache(
            model, grid, system_cache, system_length, args.gist_attn_impl, args.override_ratio
        )
        caches[label] = (cache, system_length, gist_tokens)

    upstream = HH._gist_tokens_for_lengths(
        [len(ids) for ids in doc_ids[: k_star + 1]],
        args.override_ratio,
        model.config.gist_residual_type,
        args.max_doc_length,
    )
    assert caches["trunc"][2] == upstream
    assert caches["full"][2] > upstream, "the fixture must have downstream docs"

    system_length = caches["full"][1]
    stop = system_length + upstream
    for full_layer, trunc_layer in zip(caches["full"][0].layers, caches["trunc"][0].layers):
        assert torch.equal(
            full_layer.keys[..., :stop, :], trunc_layer.keys[..., :stop, :]
        )
        assert torch.equal(
            full_layer.values[..., :stop, :], trunc_layer.values[..., :stop, :]
        )


# ---------------------------------------------------------------------------
# e. the corr slice
# ---------------------------------------------------------------------------


def test_corr_slice_equals_sequential_full_slots(stack):
    HH = _harness()
    model, tokenizer, args, example = stack
    doc_ids, offsets, k_star = _doc_layout(HH, tokenizer, example, args)
    span_len = len(doc_ids[k_star])

    prefix, skip = HH._build_d_intervene_prefix(model, tokenizer, example, args, "d_corr", None)
    assert skip is None, skip
    assert prefix["d_corr_doc_index"] == k_star
    assert prefix["d_corr_span_tokens"] == span_len

    sequential, skip = HH._build_sequential_full_prefix(model, tokenizer, example, args)
    assert skip is None, skip
    start = offsets[k_star]
    for corr_layer, seq_layer in zip(prefix["cache"].layers, sequential["cache"].layers):
        assert torch.equal(
            corr_layer.keys[..., -span_len:, :], seq_layer.keys[..., start : start + span_len, :]
        )
        assert torch.equal(
            corr_layer.values[..., -span_len:, :],
            seq_layer.values[..., start : start + span_len, :],
        )


def test_corr_all_appends_every_doc(stack):
    HH = _harness()
    model, tokenizer, args, example = stack
    doc_ids, offsets, _ = _doc_layout(HH, tokenizer, example, args)
    doc_tokens = sum(len(ids) for ids in doc_ids)

    prefix, skip = HH._build_d_intervene_prefix(model, tokenizer, example, args, "d_corr_all", None)
    assert skip is None, skip
    assert prefix["d_corr_doc_index"] is None
    assert prefix["d_corr_span_tokens"] == doc_tokens

    sequential, _ = HH._build_sequential_full_prefix(model, tokenizer, example, args)
    start = offsets[0]
    for corr_layer, seq_layer in zip(prefix["cache"].layers, sequential["cache"].layers):
        assert torch.equal(
            corr_layer.keys[..., -doc_tokens:, :],
            seq_layer.keys[..., start : start + doc_tokens, :],
        )


# ---------------------------------------------------------------------------
# f/g. recompute and the maybe_gist mask contract
# ---------------------------------------------------------------------------


def test_no_gist_recompute_rebuilds_sequential_full(stack):
    """Split the sequential prefill at k* and finish it through the
    maybe_gist path: with no gist in the cache, slots == logical positions and
    the result must be bit-identical."""
    HH = _harness()
    model, tokenizer, args, example = stack
    doc_ids, offsets, k_star = _doc_layout(HH, tokenizer, example, args)
    system_ids = HH._chat_template_ids(
        tokenizer,
        [{"role": "system", "content": example.system_prompt}],
        tools=example.tools or None,
        keep_bos=True,
        max_length=args.max_system_length,
    )
    cache, system_length, _ = HH._prefill_system(
        model, torch.tensor([system_ids], dtype=torch.long), args.system_attn_impl
    )
    logical = system_length
    for ids in doc_ids[: k_star + 1]:
        cache, added, _ = HH._prefill_tokens_with_cache(
            model,
            torch.tensor([ids], dtype=torch.long),
            past_key_values=cache,
            past_length=logical,
            attn_impl=args.generate_attn_impl,
        )
        logical += added
    for doc_index in range(k_star + 1, len(doc_ids)):
        cache, added, _ = HH._prefill_tokens_with_cache_maybe_gist(
            model,
            torch.tensor([doc_ids[doc_index]], dtype=torch.long),
            past_key_values=cache,
            past_length=offsets[doc_index],
            attn_impl=args.generate_attn_impl,
            use_gist=False,
        )
        logical += added

    sequential, skip = HH._build_sequential_full_prefix(model, tokenizer, example, args)
    assert skip is None, skip
    assert logical == system_length + sum(len(ids) for ids in doc_ids)
    _assert_cache_equal(cache, sequential["cache"], "no-gist recompute vs sequential full")

    # ... and the next-token distribution agrees.
    prompt_ids = HH._chat_template_ids(
        tokenizer, HH._current_messages(example), add_generation_prompt=True
    )
    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    logits = []
    for built in (cache, sequential["cache"]):
        slots = built.get_seq_length()
        with torch.inference_mode():
            out = model(
                input_ids=prompt,
                attention_mask=torch.ones((1, slots + prompt.shape[1]), dtype=torch.long),
                position_ids=torch.arange(logical, logical + prompt.shape[1]).unsqueeze(0),
                past_key_values=built,
                use_cache=True,
                logits_to_keep=1,
            )
        logits.append(out.logits.clone())
    assert torch.equal(logits[0], logits[1])


def test_maybe_gist_mask_uses_slots_and_positions_use_logical(stack):
    """The gist prefix has FEWER cache slots than logical positions. The mask
    must be sized from the slots, the position ids from the logical offset —
    that is the whole reason the task-D recompute must not use the
    tool-definition eval's same-named helper."""
    HH = _harness()
    model, tokenizer, args, example = stack
    doc_ids, offsets, k_star = _doc_layout(HH, tokenizer, example, args)

    prefix, skip = HH._build_d_intervene_prefix(model, tokenizer, example, args, "d_corr", None)
    assert skip is None, skip
    cache = prefix["cache"]
    slots = cache.get_seq_length()
    logical = offsets[-1] + len(doc_ids[-1])
    assert slots < logical, "the fixture must compress (slots must be fewer than positions)"

    seen = []

    class _Spy:
        def __init__(self, inner):
            object.__setattr__(self, "_inner", inner)

        def __getattr__(self, item):
            return getattr(object.__getattribute__(self, "_inner"), item)

        def __call__(self, **kwargs):
            seen.append(
                {
                    "mask_width": kwargs["attention_mask"].shape[-1],
                    "positions": kwargs["position_ids"][0].tolist(),
                    "has_gist": "use_gist" in kwargs,
                }
            )
            return object.__getattribute__(self, "_inner")(**kwargs)

    doc_index = len(doc_ids) - 1
    ids = doc_ids[doc_index]
    HH._prefill_tokens_with_cache_maybe_gist(
        _Spy(model),
        torch.tensor([ids], dtype=torch.long),
        past_key_values=cache,
        past_length=offsets[doc_index],
        attn_impl=args.generate_attn_impl,
        use_gist=False,
    )
    assert len(seen) == 1
    assert seen[0]["mask_width"] == slots + len(ids)
    assert seen[0]["positions"] == list(range(offsets[doc_index], offsets[doc_index] + len(ids)))
    assert seen[0]["has_gist"] is False
    assert cache.get_seq_length() == slots + len(ids)


def test_corr_recompute_layout_and_accounting(stack):
    HH = _harness()
    model, tokenizer, args, example = stack
    doc_ids, _, k_star = _doc_layout(HH, tokenizer, example, args)
    doc_tokens = sum(len(ids) for ids in doc_ids)
    downstream = doc_ids[k_star + 1 :]

    prefix, skip = HH._build_d_intervene_prefix(
        model, tokenizer, example, args, "d_corr_recompute", None
    )
    assert skip is None, skip
    assert prefix["d_recompute_docs"] == len(downstream)
    assert prefix["d_recompute_tokens"] == sum(len(ids) for ids in downstream)
    assert prefix["d_corr_span_tokens"] == len(doc_ids[k_star])
    assert prefix["history_length"] == doc_tokens
    expected_dropped = HH._gist_tokens_for_lengths(
        [len(ids) for ids in downstream],
        args.override_ratio,
        model.config.gist_residual_type,
        args.max_doc_length,
    )
    assert prefix["d_dropped_gist_tokens"] == expected_dropped
    expected_slots = (
        prefix["system_length"]
        + prefix["gist_tokens"]
        + prefix["d_corr_span_tokens"]
        + prefix["d_recompute_tokens"]
    )
    assert prefix["cache_length"] == expected_slots


# ---------------------------------------------------------------------------
# h/i. guards and layout invariants
# ---------------------------------------------------------------------------


def test_sham_mech_is_an_exact_identity(stack):
    """The mechanical arm extracts the slice and throws it away; the cache it
    returns must be indistinguishable from plain c2kv.  The NPU smoke repeats
    this end to end on generated text."""
    HH = _harness()
    model, tokenizer, args, example = stack
    baseline, skip = HH._build_c2kv_prefix(model, tokenizer, example, args)
    assert skip is None, skip
    mech, skip = HH._build_d_intervene_prefix(model, tokenizer, example, args, "d_sham_mech", None)
    assert skip is None, skip
    assert mech["cache_length"] == baseline["cache_length"]
    assert mech["gist_tokens"] == baseline["gist_tokens"]
    assert mech["d_corr_span_tokens"] == 0
    assert mech["d_sham_tokens"] == 0
    assert mech["d_recompute_tokens"] == 0
    _assert_cache_equal(mech["cache"], baseline["cache"], "d_sham_mech vs c2kv")


def test_sham_arm_injects_the_planned_span(stack):
    HH = _harness()
    model, tokenizer, args, example = stack
    doc_ids, offsets, k_star = _doc_layout(HH, tokenizer, example, args)
    span_len = len(doc_ids[k_star])
    plan = {
        "k_star": k_star,
        "span_len": span_len,
        "sham_token_ids": [7 + (i % 23) for i in range(span_len)],
    }
    prefix, skip = HH._build_d_intervene_prefix(
        model, tokenizer, example, args, "d_sham_neutral", plan
    )
    assert skip is None, skip
    assert prefix["d_sham_tokens"] == span_len
    assert prefix["d_corr_span_tokens"] == 0
    assert prefix["history_length"] == sum(len(ids) for ids in doc_ids)

    corr, _ = HH._build_d_intervene_prefix(model, tokenizer, example, args, "d_corr", None)
    # Equal byte budget by construction: same number of appended slots.
    assert prefix["cache_length"] == corr["cache_length"]
    # ... but a different payload.
    assert not torch.equal(
        prefix["cache"].layers[0].keys[..., -span_len:, :],
        corr["cache"].layers[0].keys[..., -span_len:, :],
    )


@pytest.mark.parametrize("bad", ["length", "k_star"])
def test_sham_plan_mismatch_is_skipped_not_silently_run(stack, bad):
    HH = _harness()
    model, tokenizer, args, example = stack
    doc_ids, _, k_star = _doc_layout(HH, tokenizer, example, args)
    span_len = len(doc_ids[k_star])
    plan = {"k_star": k_star, "span_len": span_len, "sham_token_ids": [9] * span_len}
    if bad == "length":
        plan["sham_token_ids"] = [9] * (span_len + 1)
        expected = "d_sham_length_mismatch"
    else:
        plan["k_star"] = k_star + 1
        expected = "d_plan_k_star_mismatch"
    prefix, skip = HH._build_d_intervene_prefix(
        model, tokenizer, example, args, "d_sham_neutral", plan
    )
    assert prefix is None
    assert skip.startswith(expected)


def test_missing_sham_plan_is_skipped(stack):
    HH = _harness()
    model, tokenizer, args, example = stack
    prefix, skip = HH._build_d_intervene_prefix(
        model, tokenizer, example, args, "d_sham_neutral", None
    )
    assert prefix is None
    assert skip == "d_sham_plan_missing"


def test_every_arm_keeps_the_original_decode_layout(stack):
    """history_length must stay the raw history token count so decode
    positions are identical across arms — the interventions change what is in
    the cache, never where the next token thinks it is."""
    HH = _harness()
    model, tokenizer, args, example = stack
    doc_ids, _, k_star = _doc_layout(HH, tokenizer, example, args)
    doc_tokens = sum(len(ids) for ids in doc_ids)
    baseline, _ = HH._build_c2kv_prefix(model, tokenizer, example, args)
    assert baseline["history_length"] == doc_tokens

    plan = {
        "k_star": k_star,
        "span_len": len(doc_ids[k_star]),
        "sham_token_ids": [11] * len(doc_ids[k_star]),
    }
    for mode in ("d_sham_neutral", "d_corr", "d_corr_recompute", "d_corr_all", "d_sham_mech"):
        prefix, skip = HH._build_d_intervene_prefix(model, tokenizer, example, args, mode, plan)
        assert skip is None, (mode, skip)
        assert prefix["history_length"] == doc_tokens, mode
        assert prefix["system_length"] == baseline["system_length"], mode
        assert prefix["use_gist"] is True, mode


def test_d_intervene_modes_are_wired_into_dispatch():
    HH = _harness()
    assert HH.D_INTERVENE_MODES == {
        "d_sham_neutral",
        "d_corr",
        "d_corr_recompute",
        "d_corr_all",
        "d_sham_mech",
    }
    assert isinstance(HH.D_INTERVENE, dict)
