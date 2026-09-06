# -*- coding: utf-8 -*-
"""CPU smoke tests for agent/bench_d_cost_crossover.py (S4, tests 22-25).

Tiny config (max_doc_length=32, max_doc_num=8 -> 256-token fitted ceiling)
against the two-layer random Qwen3 gist model and the whitespace tokenizer,
same recipe as agent/test_d_kv_intervene_torch.py.  No weights, no dataset,
no network.

Coverage:
22. context construction — in-regime lengths come from the REAL harness fit
    and land on the target exactly; over-ceiling lengths are packed
    fixed-width docs labeled out_of_regime;
23. run_bench mechanics — one JSONL row per (context, path-set, repeat),
    warmup excluded, summary carries only measured aggregates;
24. the bench never imports the d_kv_intervene driver;
25. clone-per-measurement — a mutating model cannot touch the master system
    cache, and disabling the clone trips the self-check assert.

Run from the repo root:
  python -m pytest agent/test_bench_d_cost_crossover.py -v
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import types
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


def _bench():
    import bench_d_cost_crossover as B  # noqa: PLC0415

    return B


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


def _hargs():
    HH = _harness()
    argv = [
        "prog",
        "--model", "./checkpoints/tiny",
        "--tokenizer", "./checkpoints/tiny",
        "--device_type", "cpu",
        "--selection_filter", "none",
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


# Small cycled vocabulary keeps whitespace-tokenizer ids far below the tiny
# model's vocab_size.
_WORDS = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()


def _example(qid="bench:1", n_turns=12, words_per_turn=14):
    from train.train_data_multiturn import CompressHistoryExample  # noqa: PLC0415

    history = []
    for turn in range(n_turns):
        text = " ".join(_WORDS[(turn + index) % len(_WORDS)] for index in range(words_per_turn))
        history.append({"role": "user" if turn % 2 == 0 else "assistant", "content": text})
    return CompressHistoryExample(
        qid=qid,
        history_messages=history,
        current_messages=[{"role": "user", "content": "what happens next"}],
        answer="more of the same",
        system_prompt="you time things",
        tools=[],
    )


@pytest.fixture(scope="module")
def stack():
    tokenizer = _tokenizer()
    model = _tiny_model(tokenizer)
    return model, tokenizer, _hargs()


# ---------------------------------------------------------------------------
# 22. context construction regimes
# ---------------------------------------------------------------------------


def test_bench_context_construction_regimes(stack):
    B = _bench()
    HH = _harness()
    _model, tokenizer, hargs = stack
    ceiling = hargs.max_doc_length * hargs.max_doc_num
    assert ceiling == 256

    # In-regime: target below the fitted ceiling, built via the REAL fit.
    example = _example()
    target = 64
    contexts, n_available = B.build_contexts(tokenizer, [example], target, hargs, n_per_length=3)
    assert n_available == 1, "shortfall must be reported, not padded"
    assert len(contexts) == 1
    context = contexts[0]
    assert context["construction"] == "harness_fit"
    assert context["out_of_regime"] is False
    assert context["target_len"] == target
    assert context["actual_len"] == target
    assert sum(len(ids) for ids in context["_doc_ids"]) == target
    assert 2 <= context["n_docs"] <= hargs.max_doc_num
    assert context["k_star"] == (context["n_docs"] - 1) // 2
    assert context["slice_tokens"] + context["recompute_tokens"] == target
    assert context["source_qid"] == example.qid

    # The docs are the harness fit itself, final doc truncated to land on L.
    history = HH._history_messages(tokenizer, example, hargs)
    fit_ids = [
        HH._chat_template_ids(tokenizer, [message], max_length=hargs.max_doc_length)
        for message in history
    ]
    expected = []
    acc = 0
    for ids in fit_ids:
        if acc >= target:
            break
        take = min(len(ids), target - acc)
        expected.append(list(ids[:take]))
        acc += take
    assert context["_doc_ids"] == expected

    # Out-of-regime: target above the ceiling -> packed fixed-width docs.
    big = _example(qid="bench:2", n_turns=24, words_per_turn=16)
    target = 300
    assert target > ceiling
    contexts, n_available = B.build_contexts(tokenizer, [big], target, hargs, n_per_length=1)
    assert n_available == 1
    context = contexts[0]
    assert context["construction"] == f"packed_{hargs.max_doc_length}"
    assert context["out_of_regime"] is True
    assert context["actual_len"] == target
    assert context["n_docs"] == math.ceil(target / hargs.max_doc_length)
    docs = context["_doc_ids"]
    assert all(len(ids) == hargs.max_doc_length for ids in docs[:-1])
    assert len(docs[-1]) == target - (len(docs) - 1) * hargs.max_doc_length
    assert context["k_star"] == (context["n_docs"] - 1) // 2
    # Packed docs are slices of the full normalized history template.
    full_ids = HH._chat_template_ids(
        tokenizer,
        [m for m in (HH._normal_chat_message(x) for x in big.history_messages) if m.get("content")],
    )
    assert [token for ids in docs for token in ids] == full_ids[:target]

    # An example that cannot furnish the length qualifies nowhere.
    contexts, n_available = B.build_contexts(
        tokenizer, [example], 10_000, hargs, n_per_length=1
    )
    assert contexts == [] and n_available == 0


def test_bench_out_of_regime_grid_overflow_refused(stack):
    """A length whose packed docs need k*+1 > max_doc_num gist rows would
    silently grow the grid past the deployment geometry (_grid_from_doc_ids
    pads without a guard) — refused loudly, never mislabeled as comparable
    to the labeled out-of-regime point."""
    B = _bench()
    _model, tokenizer, hargs = stack
    huge = _example(qid="bench:4", n_turns=80, words_per_turn=16)
    target = 17 * hargs.max_doc_length  # 17 packed docs -> k*+1 = 9 > 8
    with pytest.raises(SystemExit, match="gist grid"):
        B.build_contexts(tokenizer, [huge], target, hargs, n_per_length=1)


def test_bench_cli_rejects_degenerate_protocol_values():
    """--repeats 0 used to die later with an IndexError on length_rows[-1];
    the CLI must refuse degenerate protocol values before loading anything."""
    B = _bench()
    base = ["--dataset_path", "x", "--tokenizer", "y", "--model", "z"]
    with pytest.raises(SystemExit, match="repeats"):
        B.evaluate(B.parse_args(base + ["--repeats", "0"]))
    with pytest.raises(SystemExit, match="warmup"):
        B.evaluate(B.parse_args(base + ["--warmup", "-1"]))
    with pytest.raises(SystemExit, match="n_per_length"):
        B.evaluate(B.parse_args(base + ["--n_per_length", "0"]))


# ---------------------------------------------------------------------------
# 23. run_bench mechanics: rows, warmup exclusion, summary schema
# ---------------------------------------------------------------------------


def test_bench_jsonl_and_summary_schema(stack, tmp_path, monkeypatch):
    B = _bench()
    model, tokenizer, hargs = stack
    calls = {"full": 0, "repair": 0}
    real_full, real_repair = B._measure_full, B._measure_repair

    def spy_full(*args, **kwargs):
        calls["full"] += 1
        return real_full(*args, **kwargs)

    def spy_repair(*args, **kwargs):
        calls["repair"] += 1
        return real_repair(*args, **kwargs)

    monkeypatch.setattr(B, "_measure_full", spy_full)
    monkeypatch.setattr(B, "_measure_repair", spy_repair)

    out_path = tmp_path / "bench_timings.jsonl"
    summary_path = tmp_path / "bench_summary.json"
    repeats, warmup = 2, 2
    summary = B.run_bench(
        model,
        tokenizer,
        [_example()],
        hargs,
        lengths=[64],
        n_per_length=1,
        repeats=repeats,
        warmup=warmup,
        out_path=str(out_path),
        summary_path=str(summary_path),
    )

    rows = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    # One row per (context, path-set, repeat); warmup ran but was excluded.
    assert len(rows) == repeats
    assert calls["full"] == warmup + repeats
    assert calls["repair"] == warmup + repeats

    required = {
        "target_len", "actual_len", "source_qid", "construction", "out_of_regime",
        "n_docs", "mean_doc_len", "k_star", "slice_tokens", "recompute_tokens",
        "repeat", "path_order",
        "full_reprefill_sec", "gist_sec", "slice_prefill_sec", "append_sec",
        "recompute_sec", "repair_marginal_sec", "repair_with_gist_sec",
        "model_path", "device_type", "attn_impl", "torch_version", "timestamp",
    }
    for row in rows:
        assert required <= set(row)
        assert not any(key.startswith("_") for key in row)
        for column in B.TIMING_COLUMNS:
            assert row[column] >= 0.0
        assert row["repair_marginal_sec"] == pytest.approx(
            row["slice_prefill_sec"] + row["append_sec"] + row["recompute_sec"], abs=1e-5
        )
        assert row["repair_with_gist_sec"] == pytest.approx(
            row["repair_marginal_sec"] + row["gist_sec"], abs=1e-5
        )
    assert [row["repeat"] for row in rows] == [0, 1]
    # Interleaved path order: (context_idx + repeat) % 2 flips it.
    assert rows[0]["path_order"] == "full,repair"
    assert rows[1]["path_order"] == "repair,full"

    # Summary: measured aggregates only, no ratios / crossover claims.
    on_disk = json.loads(summary_path.read_text(encoding="utf-8"))
    assert on_disk == summary
    entry = summary["per_length"]["64"]
    assert entry["n"] == 1 and entry["n_available"] == 1
    assert entry["regime"] == "in_regime"
    assert "regime_note" not in entry, "in-regime lengths carry no out-of-regime note"
    assert set(entry["timings"]) == set(B.TIMING_COLUMNS)
    for stats in entry["timings"].values():
        assert set(stats) == {"mean_of_context_medians", "median_of_context_medians"}
    text = json.dumps(summary).lower()
    for claim in ("crossover", "speedup", "faster", "verdict"):
        assert claim not in text
    assert "environment" in summary


def test_bench_out_of_regime_summary_carries_note(stack, tmp_path):
    B = _bench()
    model, tokenizer, hargs = stack
    big = _example(qid="bench:2", n_turns=24, words_per_turn=16)
    summary = B.run_bench(
        model,
        tokenizer,
        [big],
        hargs,
        lengths=[300],
        n_per_length=1,
        repeats=1,
        warmup=0,
        out_path=str(tmp_path / "t.jsonl"),
        summary_path=str(tmp_path / "s.json"),
    )
    entry = summary["per_length"]["300"]
    assert entry["regime"] == "out_of_regime"
    assert entry["construction"] == f"packed_{hargs.max_doc_length}"
    assert "structurally unreachable" in entry["regime_note"]
    rows = [
        json.loads(line)
        for line in (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["out_of_regime"] is True for row in rows)


# ---------------------------------------------------------------------------
# 24. standalone: the bench never imports the driver
# ---------------------------------------------------------------------------


def test_bench_is_standalone():
    saved = {
        name: sys.modules.pop(name)
        for name in ("d_kv_intervene", "bench_d_cost_crossover")
        if name in sys.modules
    }
    try:
        importlib.import_module("bench_d_cost_crossover")
        assert "d_kv_intervene" not in sys.modules
    finally:
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# 25. clone-per-measurement protects the master system cache
# ---------------------------------------------------------------------------


class _StubLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class _StubCache:
    def __init__(self, layers):
        self.layers = list(layers)

    def get_seq_length(self):
        return self.layers[0].keys.shape[-2]


class _AppendingStubModel:
    """In-place-appends to the passed cache, mirroring DynamicCache.update."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.model = types.SimpleNamespace(
            config=types.SimpleNamespace(_attn_implementation="eager")
        )

    def __call__(self, input_ids=None, past_key_values=None, **kwargs):
        added = input_ids.shape[1]
        for layer in past_key_values.layers:
            shape = list(layer.keys.shape)
            shape[-2] = added
            layer.keys = torch.cat([layer.keys, torch.zeros(shape)], dim=-2)
            layer.values = torch.cat([layer.values, torch.zeros(shape)], dim=-2)
        return types.SimpleNamespace(past_key_values=past_key_values)


def test_bench_system_cache_immutability(stack, monkeypatch):
    B = _bench()
    HH = _harness()
    _model, _tokenizer, hargs = stack
    torch.manual_seed(5)
    system_length = 4
    master = _StubCache(
        [_StubLayer(torch.randn(1, 2, system_length, 8), torch.randn(1, 2, system_length, 8))]
    )
    context = {"_doc_ids": [[3, 4, 5], [6, 7]], "k_star": 0}
    stub = _AppendingStubModel()

    timings = B._measure_full(stub, context, master, system_length, hargs)
    assert master.get_seq_length() == system_length, "clone-per-measurement must hold"
    assert timings["full_reprefill_sec"] >= 0.0
    # A second measurement still sees the pristine master.
    B._measure_full(stub, context, master, system_length, hargs)
    assert master.get_seq_length() == system_length

    # The repair path under the same mutating model: the gist build (the one
    # legitimate reader of the master) is stubbed to a read-only clone, so
    # only the bench's own clone discipline protects the master.
    def stub_build_tool_cache(model, grid, system_cache, system_length_, attn_impl, ratio):
        clone = _StubCache(
            [
                _StubLayer(layer.keys.clone(), layer.values.clone())
                for layer in system_cache.layers
            ]
        )
        return clone, None, None, None, 0.0, 0.0

    monkeypatch.setattr(HH, "_build_tool_cache", stub_build_tool_cache)
    timings = B._measure_repair(stub, context, master, system_length, hargs)
    assert master.get_seq_length() == system_length
    assert timings["repair_marginal_sec"] >= 0.0

    with pytest.raises(AssertionError, match="master system cache mutated"):
        B._measure_repair(stub, context, master, system_length, hargs, clone_system=False)
    # the failed run mutated the master; rebuild before the full-path trip
    master = _StubCache(
        [_StubLayer(torch.randn(1, 2, system_length, 8), torch.randn(1, 2, system_length, 8))]
    )
    with pytest.raises(AssertionError, match="master system cache mutated"):
        B._measure_full(stub, context, master, system_length, hargs, clone_system=False)
