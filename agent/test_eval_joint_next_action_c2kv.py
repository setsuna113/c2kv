# -*- coding: utf-8 -*-
"""CPU-only unit tests for agent/eval_joint_next_action_c2kv.py.

No model weights, no real dataset, no network: doc-subset construction is
exercised with the deterministic whitespace tokenizer from
``train.train_data_joint`` (mirroring test_train_data_joint.py), metrics with
canned prediction/target strings, and the CLI with explicit argv lists.

Coverage:
a. import smoke (module + reused helpers importable on CPU);
b. condition -> doc-subset construction: joint / tool_only / history_only
   chunk contents, tool-first flat order, budget allocation (joint tool cap
   vs. tool_only all-slots), tool-token cap skip, bad condition error;
c. metric wiring: exact/tool-name/response-type matches and arg-name/value
   F1 on canned pairs, via the eval's ``_prediction_metrics`` bundle;
d. summary grouping by (condition, mode, ratio) incl. skip accounting;
e. CLI parsing: defaults, --conditions/--compare_modes/--ratios lists,
   --separate validation, --merge_only requirements.

Run from the repo root (local venv has torch/transformers/datasets/pytest):
  pytest agent/test_eval_joint_next_action_c2kv.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The eval driver imports torch at module top, so the whole file is
# torch-gated: without this it ERRORs at collection on a torch-free box
# instead of skipping.
pytest.importorskip("torch")

# Make python/, python/inference/ and agent/ importable when pytest is invoked
# from the repo root (imported entries insert these themselves, but only after
# their own top-level torch/transformers imports resolve).
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "python/inference", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from train.train_data_joint import JointExample, _WhitespaceSelfTestTokenizer  # noqa: E402
from eval_joint_next_action_c2kv import (  # noqa: E402
    CONDITIONS,
    SEPARATE_MODE,
    _common_valid_rows,
    _condition_doc_chunks,
    _doc_grid,
    _flat_doc_ids,
    _prediction_metrics,
    _summarize,
    parse_args,
)


def _example(tool_docs=3, history_docs=3, doc_words=6):
    return JointExample(
        qid="s0:0",
        session_id="s0",
        tool_documents=[
            " ".join(f"tool{i}word{j}" for j in range(doc_words)) for i in range(tool_docs)
        ],
        history_documents=[
            " ".join(f"hist{i}word{j}" for j in range(doc_words)) for i in range(history_docs)
        ],
        current_messages=[{"role": "user", "content": "current question"}],
        answer='Action:\n<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>',
        system_prompt="You are a test agent.",
        subset="test",
    )


def _chunks(tokenizer, example, condition, **overrides):
    tool_chunks, history_chunks, reason, _meta = _chunks_with_meta(
        tokenizer, example, condition, **overrides
    )
    return tool_chunks, history_chunks, reason


def _chunks_with_meta(tokenizer, example, condition, **overrides):
    kwargs = dict(
        max_doc_length=16,
        max_doc_num=4,
        max_tool_chunks=None,
        max_tool_definition_tokens=1000,
        history_selection="tail",
        split_oversized_history_docs=True,
    )
    kwargs.update(overrides)
    return _condition_doc_chunks(tokenizer, example, condition, **kwargs)


# ---------------------------------------------------------------------------
# a. import smoke
# ---------------------------------------------------------------------------


def test_import_smoke():
    import eval_joint_next_action_c2kv as module

    for name in (
        "evaluate",
        "merge_shards",
        "_build_c2kv_prefix",
        "_build_separate_prefix",
        "_build_baseline_prefix",
        "_compress_docs_to_cache",
        "_generate_one",
        "_generate_one_separate",
    ):
        assert hasattr(module, name), name
    # Reused helpers come from the existing evals (import, not copy).
    from eval_agent_history_c2kv import _target_metrics as history_target_metrics

    assert module._target_metrics is history_target_metrics
    assert CONDITIONS == ("joint", "tool_only", "history_only")
    assert SEPARATE_MODE == "c2kv_separate"


# ---------------------------------------------------------------------------
# b. condition -> doc-subset construction
# ---------------------------------------------------------------------------


def test_joint_condition_has_both_sides_tool_first():
    tokenizer = _WhitespaceSelfTestTokenizer()
    example = _example()
    tool_chunks, history_chunks, reason = _chunks(tokenizer, example, "joint")
    assert reason is None
    assert tool_chunks and history_chunks
    flat = _flat_doc_ids(tool_chunks, history_chunks)
    decoded = tokenizer.decode(flat)
    assert decoded.index("tool0word0") < decoded.index("hist0word0")


def test_tool_only_drops_history_and_shares_the_joint_tool_cap():
    tokenizer = _WhitespaceSelfTestTokenizer()
    # 6 tool docs of ~8 tokens each -> more than the shared tool cap of
    # min(2*4//3, 4) = 2 chunks.  Per-side caps: tool_only gets the SAME cap
    # as joint (the G-Q3 fairness constraint), not all max_doc_num slots.
    example = _example(tool_docs=6, history_docs=2, doc_words=6)
    joint_tools, joint_history, _ = _chunks(tokenizer, example, "joint", max_doc_length=16)
    only_tools, only_history, _ = _chunks(tokenizer, example, "tool_only", max_doc_length=16)
    assert only_history == []
    assert len(joint_tools) == 2  # shared cap: min(_default_max_tool_chunks(4)=2, 4)
    assert len(only_tools) == len(joint_tools)
    assert joint_history  # joint fills its constant history budget
    # The legacy flag reproduces the pre-fix uplift (all 4 slots).
    legacy_tools, _, _ = _chunks(
        tokenizer, example, "tool_only", max_doc_length=16, per_side_caps=False
    )
    assert len(legacy_tools) > len(joint_tools)


def test_history_only_drops_tools_and_shares_the_joint_history_budget():
    tokenizer = _WhitespaceSelfTestTokenizer()
    example = _example(tool_docs=2, history_docs=6, doc_words=6)
    joint_tools, joint_history, _ = _chunks(tokenizer, example, "joint", max_doc_length=16)
    only_tools, only_history, _ = _chunks(tokenizer, example, "history_only", max_doc_length=16)
    assert only_tools == []
    assert len(only_history) == len(joint_history)  # constant per-side budget
    assert joint_tools
    legacy_tools, legacy_history, _ = _chunks(
        tokenizer, example, "history_only", max_doc_length=16, per_side_caps=False
    )
    assert legacy_tools == []
    assert len(legacy_history) > len(only_history)  # legacy: all max_doc_num slots


def test_tool_token_cap_skip():
    tokenizer = _WhitespaceSelfTestTokenizer()
    example = _example(tool_docs=3, doc_words=30)
    tool_chunks, history_chunks, reason = _chunks(
        tokenizer, example, "joint", max_tool_definition_tokens=10
    )
    assert tool_chunks is None and history_chunks is None
    assert reason == "tool_definition_tokens>10"


def test_bad_condition_raises():
    tokenizer = _WhitespaceSelfTestTokenizer()
    with pytest.raises(ValueError):
        _chunks(tokenizer, _example(), "tools_and_history")


def test_doc_grid_pads_to_max_doc_length():
    tokenizer = _WhitespaceSelfTestTokenizer()
    example = _example(tool_docs=2, history_docs=2)
    tool_chunks, history_chunks, _ = _chunks(tokenizer, example, "joint", max_doc_length=16)
    grid = _doc_grid([*tool_chunks, *history_chunks], 16)
    assert grid.shape[0] == len(tool_chunks) + len(history_chunks)
    assert grid.shape[1] == 16
    for row, chunk in zip(grid.tolist(), [*tool_chunks, *history_chunks]):
        assert row[: len(chunk)] == list(chunk)
        assert all(token == -100 for token in row[len(chunk) :])


# ---------------------------------------------------------------------------
# c. metric wiring
# ---------------------------------------------------------------------------

_TARGET = 'Action:\n<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>'


def test_metrics_perfect_tool_call_match():
    tokenizer = _WhitespaceSelfTestTokenizer()
    metrics = _prediction_metrics(tokenizer, _TARGET, _TARGET)
    assert metrics["exact_match"] is True
    assert metrics["tool_name_match"] is True
    assert metrics["target_tool_name"] == "get_weather"
    assert metrics["prediction_tool_name"] == "get_weather"
    assert metrics["response_type_match"] is True
    assert metrics["argument_name_f1"] == 1.0
    assert metrics["argument_value_f1"] == 1.0


def test_metrics_wrong_argument_value():
    tokenizer = _WhitespaceSelfTestTokenizer()
    prediction = 'Action:\n<tool_call>\n{"name":"get_weather","arguments":{"city":"London"}}\n</tool_call>'
    metrics = _prediction_metrics(tokenizer, _TARGET, prediction)
    assert metrics["exact_match"] is False
    assert metrics["tool_name_match"] is True
    assert metrics["argument_name_f1"] == 1.0
    assert metrics["argument_value_f1"] == 0.0


def test_metrics_partial_argument_name_overlap():
    tokenizer = _WhitespaceSelfTestTokenizer()
    prediction = (
        'Action:\n<tool_call>\n{"name":"get_weather",'
        '"arguments":{"city":"Paris","units":"c"}}\n</tool_call>'
    )
    metrics = _prediction_metrics(tokenizer, _TARGET, prediction)
    # pred names {city, units}, gold {city}: P=1/2, R=1 -> F1=2/3
    assert metrics["argument_name_f1"] == pytest.approx(2 / 3, abs=1e-3)
    assert metrics["argument_value_f1"] == pytest.approx(2 / 3, abs=1e-3)


def test_metrics_text_vs_tool_call_response_type():
    tokenizer = _WhitespaceSelfTestTokenizer()
    metrics = _prediction_metrics(tokenizer, _TARGET, "The weather in Paris is sunny.")
    assert metrics["response_type_match"] is False
    assert metrics["tool_name_match"] is False
    assert metrics["argument_name_f1"] == 0.0
    assert metrics["argument_value_f1"] == 0.0


def test_metrics_text_target_token_and_rouge():
    tokenizer = _WhitespaceSelfTestTokenizer()
    target = "The weather in Paris is sunny today."
    metrics = _prediction_metrics(tokenizer, target, target)
    assert metrics["exact_match"] is True
    assert metrics["text_token_f1"] == 1.0
    assert metrics["rouge_l_f1"] == 1.0
    assert metrics["non_tool_exact_match"] is True


# ---------------------------------------------------------------------------
# d. summary grouping
# ---------------------------------------------------------------------------


def _canned_row(condition, mode, ratio, **overrides):
    row = {
        "qid": "s:0",
        "condition": condition,
        "mode": mode,
        "ratio": ratio,
        "skipped": False,
        "exact_match": True,
        "tool_name_match": True,
        "has_tool_call": True,
        "response_type_match": True,
        "argument_name_f1": 1.0,
        "argument_value_f1": 1.0,
        "text_token_f1": 1.0,
        "rouge_l_f1": 1.0,
        "doc_tokens": 80,
        "doc_chunks": 2,
        "gist_tokens": 10,
        "compressed_tokens": 10,
        "prompt_tokens": 12,
        "generated_tokens": 8,
        "actual_compression_ratio": 8.0,
        "system_prefill_sec": 0.1,
        "tool_compress_sec": 0.2,
        "full_prefill_sec": 0.0,
        "blend_sec": 0.05,
        "ttft_sec": 0.35,
        "generate_sec": 0.8,
        "tbt_sec": 0.1,
        "total_sec": 1.15,
    }
    row.update(overrides)
    return row


def test_summarize_groups_by_condition_mode_ratio():
    rows = [
        _canned_row("joint", "c2kv", 8),
        _canned_row("joint", "c2kv", 8, exact_match=False, tool_name_match=False),
        _canned_row("tool_only", "c2kv", 8),
        _canned_row("joint", "full", 1, gist_tokens=0, compressed_tokens=80),
        _canned_row("joint", "c2kv", 8, skipped=True, skip_reason="oom"),
    ]
    summaries = _summarize(rows)
    keys = {(entry["condition"], entry["mode"], entry["ratio"]) for entry in summaries}
    assert keys == {("joint", "c2kv", 8), ("tool_only", "c2kv", 8), ("joint", "full", 1)}
    joint = next(
        entry for entry in summaries
        if (entry["condition"], entry["mode"], entry["ratio"]) == ("joint", "c2kv", 8)
    )
    assert joint["num_examples"] == 3
    assert joint["num_valid"] == 2
    assert joint["num_skipped"] == 1
    assert joint["skip_reasons"] == {"oom": 1}
    assert joint["exact_match"] == 0.5
    assert joint["tool_name_accuracy"] == 0.5
    assert joint["response_type_accuracy"] == 1.0
    assert joint["argument_name_f1"] == 1.0
    assert joint["token_weighted_actual_compression_ratio"] == pytest.approx(8.0)
    assert joint["token_weighted_tbt_sec"] == pytest.approx(0.1)
    full = next(
        entry for entry in summaries
        if (entry["condition"], entry["mode"], entry["ratio"]) == ("joint", "full", 1)
    )
    assert full["avg_actual_compression_ratio"] == 8.0  # canned row value
    assert full["token_weighted_actual_compression_ratio"] == pytest.approx(1.0)


def test_common_valid_rows_intersects_per_group():
    rows = [
        _canned_row("joint", "c2kv", 8, qid="a"),
        _canned_row("joint", "c2kv", 8, qid="b"),
        _canned_row("joint", "full", 1, qid="a"),
        _canned_row("joint", "full", 1, qid="b", skipped=True, skip_reason="oom"),
    ]
    common = _common_valid_rows(rows)
    assert {row["qid"] for row in common} == {"a"}
    assert len(common) == 2


# ---------------------------------------------------------------------------
# e. CLI parsing
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    args = parse_args(["--model", "ckpt"])
    assert args.model == "ckpt"
    assert args.condition == "joint"
    assert args.conditions is None  # falls back to --condition at evaluate() time
    assert args.compare_modes == "c2kv,full"
    assert args.ratios == "8"
    assert args.override_ratio == 8
    assert args.split == "eval"
    assert args.split_manifest_name == "subset_disjoint"
    assert args.max_doc_length == 1024
    assert args.max_doc_num == 24
    assert args.max_system_length == 512
    assert args.require_tool_call is True  # matches joint training (gate-eval default)
    assert args.min_doc_num == 2  # matches JointDataset default
    assert args.max_tool_definition_tokens == 32000
    assert args.separate is False
    assert args.separate_generator == "tool"
    assert args.merge_only is False


def test_parse_args_lists_and_conditions():
    args = parse_args([
        "--model", "ckpt",
        "--conditions", "joint,tool_only",
        "--compare_modes", "c2kv,c2kv_untrained,truncate,full",
        "--ratios", "4,8",
        "--split_manifest_file", "manifest.json",
        "--max_examples", "25",
    ])
    assert args.conditions == "joint,tool_only"
    assert args.compare_modes == "c2kv,c2kv_untrained,truncate,full"
    assert args.ratios == "4,8"
    assert args.split_manifest_file == "manifest.json"
    assert args.max_examples == 25


def test_parse_args_separate_needs_no_model():
    args = parse_args([
        "--separate",
        "--checkpoint_tool", "ckpt-tool",
        "--checkpoint_history", "ckpt-hist",
        "--separate_generator", "history",
    ])
    assert args.separate is True
    assert args.checkpoint_tool == "ckpt-tool"
    assert args.checkpoint_history == "ckpt-hist"
    assert args.separate_generator == "history"


def test_parse_args_requires_model_without_separate():
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_merge_only_requires_input_files():
    with pytest.raises(SystemExit):
        parse_args(["--merge_only"])
    args = parse_args(["--merge_only", "--input_files", "a.jsonl", "b.jsonl"])
    assert args.merge_only is True
    assert args.input_files == ["a.jsonl", "b.jsonl"]


# ---------------------------------------------------------------------------
# f. baseline model loading: gist config injection (base config.json carries
#    no gist fields; modeling_qwen3.py:461 asserts without them).
# ---------------------------------------------------------------------------


class _FakeConfigClass:
    captured: dict = {}

    @staticmethod
    def from_pretrained(path, **kwargs):
        from types import SimpleNamespace

        _FakeConfigClass.captured = {"path": path, "kwargs": kwargs}
        return SimpleNamespace(**kwargs)


class _FakeModel:
    def __init__(self):
        from types import SimpleNamespace

        self.model = SimpleNamespace(layers=[])

    def eval(self):
        return self


class _FakeModelClass:
    captured: dict = {}

    @staticmethod
    def from_pretrained(path, config=None, **kwargs):
        _FakeModelClass.captured = {"path": path, "config": config, "kwargs": kwargs}
        return _FakeModel()


def test_load_baseline_model_injects_gist_config(monkeypatch):
    import eval_joint_next_action_c2kv as module

    monkeypatch.setattr(
        module, "get_model_class", lambda path, kind: (_FakeConfigClass, _FakeModelClass)
    )
    tokenizer = _WhitespaceSelfTestTokenizer()
    args = parse_args(["--model", "ckpt", "--base_model", "base", "--compare_modes", "full"])
    module._load_baseline_model(args, tokenizer, "cpu")

    # The model loads from --base_model, not --model.
    assert _FakeModelClass.captured["path"] == "base"
    config = _FakeModelClass.captured["config"]
    # The config handed to the custom class carries the gist fields from the
    # training-recipe defaults (mirrors _gist_compatible_config /
    # model_utils.get_model_and_tokenizer injection).
    assert config.gist_token_id == tokenizer.eos_token_id
    assert config.gist_type == "dynamic-interleave"
    assert config.gist_param == "qkv"
    assert config.gist_residual_type == "embed-mean"
    assert config.gist_overlap == 64


def test_load_baseline_model_falls_back_to_model_path(monkeypatch):
    import eval_joint_next_action_c2kv as module

    monkeypatch.setattr(
        module, "get_model_class", lambda path, kind: (_FakeConfigClass, _FakeModelClass)
    )
    tokenizer = _WhitespaceSelfTestTokenizer()
    args = parse_args(["--model", "ckpt", "--compare_modes", "full"])
    module._load_baseline_model(args, tokenizer, "cpu")
    assert _FakeModelClass.captured["path"] == "ckpt"
    assert _FakeModelClass.captured["config"].gist_token_id == tokenizer.eos_token_id


def _patch_evaluate_boundary(monkeypatch, calls):
    import eval_joint_next_action_c2kv as module

    monkeypatch.setattr(module, "_setup_device", lambda device_type: "cpu")
    monkeypatch.setattr(module, "_load_tokenizer", lambda args: _WhitespaceSelfTestTokenizer())
    monkeypatch.setattr(module, "_load_examples", lambda args: [])

    def _fake_load_model(args, tokenizer, device):
        calls.append(("_load_model", args.model, args.mode))
        return _FakeModel()

    def _fake_load_baseline(args, tokenizer, device):
        calls.append(("_load_baseline_model", args.base_model or args.model, args.mode))
        return _FakeModel()

    monkeypatch.setattr(module, "_load_model", _fake_load_model)
    monkeypatch.setattr(module, "_load_baseline_model", _fake_load_baseline)
    return module


def test_evaluate_routes_baseline_modes_to_injected_loader(monkeypatch, tmp_path):
    calls = []
    module = _patch_evaluate_boundary(monkeypatch, calls)
    args = parse_args([
        "--model", "ckpt",
        "--base_model", "base",
        "--compare_modes", "c2kv,full,truncate,c2kv_untrained",
        "--output_file", str(tmp_path / "out.jsonl"),
    ])
    module.evaluate(args)
    assert calls == [
        ("_load_model", "ckpt", "c2kv"),
        ("_load_baseline_model", "base", "full"),
        ("_load_baseline_model", "base", "truncate"),
        ("_load_baseline_model", "base", "c2kv"),
    ]


def test_evaluate_c2kv_untrained_requires_base_model(monkeypatch, tmp_path):
    calls = []
    module = _patch_evaluate_boundary(monkeypatch, calls)
    args = parse_args([
        "--model", "ckpt",
        "--compare_modes", "c2kv_untrained",
        "--output_file", str(tmp_path / "out.jsonl"),
    ])
    with pytest.raises(ValueError, match="base_model"):
        module.evaluate(args)
    assert calls == []


# ---------------------------------------------------------------------------
# g. NLL equivalence: trainer-style forward vs. the eval c2kv flow.
#
# Regression test for the NPU symptom "eval_loss 0.038 but c2kv generation
# scores zero": teacher-forced NLL of the answer must be the SAME whether the
# context grid is consumed exactly as GistMultiDocTrainer.compute_loss does
# (system KV -> process_context_input_ids -> prompt forward) or as the eval
# driver does (system prefill -> _build_tool_cache/blend -> prefix-KV forward
# with generation position bookkeeping).
# ---------------------------------------------------------------------------


def _tiny_gist_model(tokenizer):
    import torch

    from models.qwen3 import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
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
        gist_token_id=tokenizer.eos_token_id,  # training convention: gist_token_id=eos
        pad_token_id=tokenizer.pad_token_id,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg)
    model.eval()
    return model


_NLL_EXAMPLE = JointExample(
    qid="nll:0",
    session_id="nll",
    tool_documents=[
        "<TOOL>\n<NAME> get_weather\n<DESCRIPTION> Fetch the current weather for one city.\n</TOOL>",
        "<TOOL>\n<NAME> search_files\n<DESCRIPTION> Search files under one directory path.\n</TOOL>",
    ],
    history_documents=[
        "Previous turn\n[User query]\nList the files in /tmp please.\n[Assistant output]\n"
        'Action:\n<tool_call>\n{"name":"search_files","arguments":{"path":"/tmp"}}\n</tool_call>',
        "Previous turn\n[User query]\nfound a.txt and b.txt under /tmp",
    ],
    current_messages=[{"role": "user", "content": "What is the weather in Paris right now?"}],
    answer='Action:\n<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>',
    system_prompt="You are a careful data agent.",
    subset="test",
)


def _trainer_style_answer_nll(
    model,
    tokenizer,
    example,
    *,
    max_doc_length,
    max_doc_num,
    max_system_length,
):
    """Faithful batch-1 replica of GistMultiDocTrainer.compute_loss (trainer.py:219-349)."""
    import torch

    from train.train_data_joint import JointDataset

    features, reason = JointDataset.preprocess_example(
        example,
        tokenizer=tokenizer,
        max_length=512,
        max_doc_length=max_doc_length,
        min_doc_num=1,
        max_doc_num=max_doc_num,
        max_system_length=max_system_length,
        doc_mode="joint",
    )
    assert features is not None, reason

    # trainer._build_system_kv (batch=1: left-pad to uniform width is a no-op)
    system_ids = [token for token in features["system_input_ids"] if token != -100]
    system_tensor = torch.tensor([system_ids], dtype=torch.long)
    system_mask = torch.ones_like(system_tensor)
    with torch.inference_mode():
        system_out = model(
            system_tensor, attention_mask=system_mask, use_cache=True, logits_to_keep=1
        )
    system_kv = system_out.past_key_values
    past_length = len(system_ids)

    # compute_loss: reshape flat grid, trim batch-local doc padding
    context_grid = torch.tensor(features["context_input_ids"], dtype=torch.long).reshape(
        1, -1, max_doc_length
    )
    doc_lengths = (context_grid != -100).sum(dim=2)
    max_active = int(doc_lengths.max().item())
    if 0 < max_active < max_doc_length:
        context_grid = context_grid[:, :, :max_active]
    context_valid = int((context_grid != -100).sum().item())

    # compute_loss: trim input_ids/labels/attention_mask to active length
    real_len = sum(features["attention_mask"])
    input_ids = torch.tensor([features["input_ids"][:real_len]], dtype=torch.long)
    labels = torch.tensor([features["labels"][:real_len]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    input_length = input_ids.shape[1]
    position_ids = (
        torch.arange(input_length, dtype=torch.long).unsqueeze(0) + past_length + context_valid
    )
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=system_kv,
            context_input_ids=context_grid,
            past_attention_mask=system_mask,
            labels=labels,
            use_cache=True,
        )
    return float(out.loss), past_length, context_valid


def _eval_flow_answer_nll(model, tokenizer, example, args):
    """Teacher-forced NLL through the eval driver's prefix + generation bookkeeping."""
    import torch

    import eval_joint_next_action_c2kv as module
    from train.train_data_multiturn import _chat_template_ids

    prefix, skip_reason = module._build_c2kv_prefix(model, tokenizer, example, args)
    assert prefix is not None, skip_reason
    prompt_ids = _chat_template_ids(tokenizer, example.current_messages, add_generation_prompt=True)
    answer_ids = tokenizer.encode(example.answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    full_ids = prompt_ids + answer_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    labels = torch.tensor([[-100] * len(prompt_ids) + answer_ids], dtype=torch.long)
    original_prefix_length = prefix["system_length"] + prefix["doc_length"]
    position_ids = torch.arange(
        original_prefix_length, original_prefix_length + len(full_ids), dtype=torch.long
    ).unsqueeze(0)
    attention_mask = torch.ones((1, prefix["cache_length"] + len(full_ids)), dtype=torch.long)
    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=prefix["cache"],
            labels=labels,
            use_gist=True,  # generation-time convention (prefix was gist-compressed)
            use_cache=True,
        )
    return float(out.loss), prefix["system_length"], prefix["doc_length"]


def test_nll_equivalence_trainer_vs_eval_c2kv(monkeypatch):
    import torch  # noqa: F401

    import eval_joint_next_action_c2kv as module

    # Pin the training-side dynamic ratio to the eval override (8).
    monkeypatch.setenv("C2KV_GIST_TRAIN_RATIOS", "8")
    tokenizer = _WhitespaceSelfTestTokenizer()
    model = _tiny_gist_model(tokenizer)
    max_doc_length = 64
    max_doc_num = 8
    max_system_length = 128

    trainer_nll, past_length, context_valid = _trainer_style_answer_nll(
        model,
        tokenizer,
        _NLL_EXAMPLE,
        max_doc_length=max_doc_length,
        max_doc_num=max_doc_num,
        max_system_length=max_system_length,
    )

    args = parse_args(["--model", "tiny", "--ratios", "8"])
    args.condition = "joint"
    args.row_mode = "c2kv"
    args.max_doc_length = max_doc_length
    args.max_doc_num = max_doc_num
    args.max_tool_chunks = None
    args.max_system_length = max_system_length
    args.max_tool_definition_tokens = 100000
    args.min_doc_num = 1
    args.override_ratio = 8
    args.system_attn_impl = "eager"
    args.gist_attn_impl = "eager"
    args.generate_attn_impl = "eager"

    eval_nll, system_length, doc_length = _eval_flow_answer_nll(
        model, tokenizer, _NLL_EXAMPLE, args
    )

    # Same logical layout on both paths before comparing losses.
    assert system_length == past_length
    assert doc_length == context_valid
    assert eval_nll == pytest.approx(trainer_nll, rel=1e-3)


# ---------------------------------------------------------------------------
# h. current-message normalization + gist-init diagnostic.
# ---------------------------------------------------------------------------


def test_current_prompt_ids_matches_joint_dataset_normalization():
    """Current turn with a tool-role message: the eval prompt must be built
    exactly like JointDataset.preprocess_example builds it (tool -> user,
    empty content dropped unless assistant)."""
    from eval_joint_next_action_c2kv import _current_prompt_ids
    from train.train_data_multiturn import _chat_template_ids, _normal_chat_message

    tokenizer = _WhitespaceSelfTestTokenizer()
    example = JointExample(
        qid="s:0",
        session_id="s",
        tool_documents=["tool doc words"],
        history_documents=["hist doc words"],
        current_messages=[
            {"role": "user", "content": "call the tool"},
            {"role": "tool", "content": "tool result text"},
            {"role": "user", "content": ""},  # dropped by the training-side filter
        ],
        answer='Action:\n<tool_call>\n{"name":"t","arguments":{}}\n</tool_call>',
        system_prompt="sys",
        subset="test",
    )
    expected_current = [
        _normal_chat_message(message)
        for message in example.current_messages
        if message.get("content") or message.get("role") == "assistant"
    ]
    expected = _chat_template_ids(tokenizer, expected_current, add_generation_prompt=True)
    assert _current_prompt_ids(tokenizer, example) == expected
    decoded = tokenizer.decode(expected)
    assert "tool result text" in decoded
    # tail truncation
    truncated = _current_prompt_ids(tokenizer, example, max_prompt_tokens=3)
    assert truncated == expected[-3:]


def test_gist_params_at_init_fraction():
    import torch

    import eval_joint_next_action_c2kv as module

    tokenizer = _WhitespaceSelfTestTokenizer()
    model = _tiny_gist_model(tokenizer)
    # Constructor zero-inits gist projections (gen_gist_proj), so they differ
    # from the random base projections: fraction 0 ("trained/distinct").
    assert module._gist_params_at_init_fraction(model) == 0.0
    # Simulate the init_gist_proj copy (missing checkpoint keys -> base copy).
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.gist_q_proj.weight.copy_(layer.self_attn.q_proj.weight)
            layer.self_attn.gist_k_proj.weight.copy_(layer.self_attn.k_proj.weight)
            layer.self_attn.gist_v_proj.weight.copy_(layer.self_attn.v_proj.weight)
    assert module._gist_params_at_init_fraction(model) == 1.0
    # One layer drifting away (training) drops the fraction below 1.
    with torch.no_grad():
        model.model.layers[0].self_attn.gist_q_proj.weight.add_(1e-3)
    assert module._gist_params_at_init_fraction(model) == 0.5


# ---------------------------------------------------------------------------
# Per-side caps: target meta propagation + merged gist_init aggregation.
# ---------------------------------------------------------------------------


def test_condition_doc_chunks_reports_target_meta():
    tokenizer = _WhitespaceSelfTestTokenizer()
    base = _example(tool_docs=6, history_docs=2, doc_words=6)
    # Target the LAST tool doc: under the shared cap of 2 the pre-fix
    # head-truncation would drop it; the fixed truncation must keep it.
    targeted = JointExample(
        qid=base.qid,
        session_id=base.session_id,
        tool_documents=base.tool_documents,
        history_documents=base.history_documents,
        current_messages=base.current_messages,
        answer=base.answer,
        system_prompt=base.system_prompt,
        subset=base.subset,
        target_tool="tool5",
        target_tool_doc_index=5,
    )
    tool_chunks, _, reason, meta = _chunks_with_meta(
        tokenizer, targeted, "joint", max_doc_length=16
    )
    assert reason is None
    assert meta["target_known"] is True and meta["target_in_grid"] is True
    decoded = tokenizer.decode([token for chunk in tool_chunks for token in chunk])
    assert "tool5word0" in decoded
    _, _, _, legacy_meta = _chunks_with_meta(
        tokenizer, targeted, "joint", max_doc_length=16, per_side_caps=False
    )
    assert legacy_meta["target_in_grid"] is False
    # history_only never reports target flags (no tool side).
    _, _, _, history_meta = _chunks_with_meta(tokenizer, targeted, "history_only")
    assert history_meta["target_in_grid"] is None


def test_merge_shards_aggregates_gist_init_fractions(tmp_path):
    import json as _json
    from types import SimpleNamespace

    from eval_joint_next_action_c2kv import merge_shards

    # merge_shards re-summarizes the merged rows via _summarize, which
    # hard-indexes the metric booleans — the minimal rows must carry them.
    shard_rows = [
        {"qid": "q0", "condition": "joint", "mode": "c2kv", "ratio": 8, "skipped": False,
         "exact_match": False, "tool_name_match": False, "has_tool_call": False,
         "response_type_match": False},
        {"qid": "q1", "condition": "joint", "mode": "c2kv", "ratio": 8, "skipped": False,
         "exact_match": False, "tool_name_match": False, "has_tool_call": False,
         "response_type_match": False},
    ]
    shard_files = []
    for index, fractions in enumerate(({"joint": 0.1}, {"joint": 0.7, "separate_tool": 0.2})):
        shard = tmp_path / f"part{index}.jsonl"
        shard.write_text(
            "\n".join(_json.dumps(row) for row in shard_rows[index : index + 1]) + "\n",
            encoding="utf-8",
        )
        shard.with_suffix(".summary.json").write_text(
            _json.dumps({"gist_init_fractions": fractions}), encoding="utf-8"
        )
        shard_files.append(str(shard))
    args = SimpleNamespace(
        input_files=shard_files,
        output_file=str(tmp_path / "merged.jsonl"),
        model="m",
        base_model=None,
        dataset_path="d",
        split="eval",
        separate=False,
        checkpoint_tool=None,
        checkpoint_history=None,
        separate_generator=None,
    )
    summary = merge_shards(args)
    # Max per key: the pick guard must see the worst shard.
    assert summary["gist_init_fractions"] == {"joint": 0.7, "separate_tool": 0.2}
    merged_summary_path = (tmp_path / "merged.jsonl").with_suffix(".summary.json")
    persisted = _json.loads(merged_summary_path.read_text(encoding="utf-8"))
    assert persisted["gist_init_fractions"] == {"joint": 0.7, "separate_tool": 0.2}
    # No manifest info in any shard summary -> null passthrough, zero missing.
    assert persisted["qid_manifest"] is None
    assert persisted["qid_manifest_missing"] == 0


def test_merge_shards_passes_through_qid_manifest(tmp_path, caplog):
    """b_prereg.md §2 gates paired tables on qid_manifest_missing == 0; the
    count and the manifest path must survive --merge_only onto the merged
    summary (missing = sum over shards, path = first non-null)."""
    import json as _json
    import logging as _logging
    from types import SimpleNamespace

    from eval_joint_next_action_c2kv import merge_shards

    shard_files = []
    manifests = ["configs/bdf_pilot/b_eval200_qids.json", "configs/other_manifest.json"]
    for index, (manifest, missing) in enumerate(zip(manifests, (2, 1))):
        shard = tmp_path / f"part{index}.jsonl"
        shard.write_text(
            _json.dumps({
                "qid": f"q{index}", "condition": "joint", "mode": "c2kv",
                "ratio": 8, "skipped": False,
            }) + "\n",
            encoding="utf-8",
        )
        shard.with_suffix(".summary.json").write_text(
            _json.dumps({"qid_manifest": manifest, "qid_manifest_missing": missing}),
            encoding="utf-8",
        )
        shard_files.append(str(shard))
    args = SimpleNamespace(
        input_files=shard_files,
        output_file=str(tmp_path / "merged.jsonl"),
        model="m",
        base_model=None,
        dataset_path="d",
        split="eval",
        separate=False,
        checkpoint_tool=None,
        checkpoint_history=None,
        separate_generator=None,
    )
    with caplog.at_level(_logging.WARNING):
        summary = merge_shards(args)
    assert summary["qid_manifest"] == manifests[0]  # first non-null
    assert summary["qid_manifest_missing"] == 3     # sum over shards
    log_text = caplog.text
    assert "DIFFERENT QID MANIFESTS" in log_text
    assert "INCOMPLETE" in log_text
    persisted = _json.loads(
        (tmp_path / "merged.jsonl").with_suffix(".summary.json").read_text(encoding="utf-8")
    )
    assert persisted["qid_manifest"] == manifests[0]
    assert persisted["qid_manifest_missing"] == 3


# ---------------------------------------------------------------------------
# Experiment B: chunking policy / delayed compression / frozen qid manifest.
# ---------------------------------------------------------------------------


def _agent_history_example():
    """A JointExample whose history carries unit provenance (action+observation)."""
    from train.train_data_multiturn import (
        _agent_history_turn_docs,
        _agent_history_turn_units,
    )

    messages = [
        {"role": "user", "content": "list the files in /tmp"},
        {
            "role": "assistant",
            "content": 'Action:\n<tool_call>\n{"name":"ls","arguments":{"path":"/tmp"}}\n</tool_call>',
        },
        {"role": "tool", "content": "a.txt b.txt"},
        {"role": "assistant", "content": "there are two files"},
        {"role": "user", "content": "now read a.txt"},
        {
            "role": "assistant",
            "content": 'Action:\n<tool_call>\n{"name":"read","arguments":{"path":"/tmp/a.txt"}}\n</tool_call>',
        },
        {"role": "tool", "content": "hello world"},
    ]
    return JointExample(
        qid="s0:0",
        session_id="s0",
        tool_documents=["<TOOL> ls </TOOL>", "<TOOL> read </TOOL>"],
        history_documents=[doc["content"] for doc in _agent_history_turn_docs(messages)],
        current_messages=[{"role": "user", "content": "and b.txt?"}],
        answer='Action:\n<tool_call>\n{"name":"read","arguments":{"path":"/tmp/b.txt"}}\n</tool_call>',
        system_prompt="You are a test agent.",
        subset="test",
        history_units=_agent_history_turn_units(messages),
    )


def test_condition_doc_chunks_policy_meta():
    tokenizer = _WhitespaceSelfTestTokenizer()
    example = _agent_history_example()
    kwargs = dict(
        max_doc_length=256,
        max_doc_num=8,
        max_tool_chunks=4,
        max_tool_definition_tokens=10000,
        history_selection="tail",
        split_oversized_history_docs=True,
    )
    _, turn_chunks, reason, turn_meta = _condition_doc_chunks(
        tokenizer, example, "joint", chunk_policy="agent-turn", **kwargs
    )
    assert reason is None
    assert turn_meta["chunk_policy"] == "agent-turn"
    assert turn_meta["raw_recent_tokens"] == 0
    assert turn_meta["history_content_tokens"] > 0
    assert turn_meta["history_wrapped_tokens"] == sum(len(c) for c in turn_chunks)

    _, struct_chunks, reason, struct_meta = _condition_doc_chunks(
        tokenizer, example, "joint", chunk_policy="structural", **kwargs
    )
    assert reason is None
    # Same frozen content, more (smaller) chunks: that IS the P6 arm.
    assert struct_meta["history_content_tokens"] == turn_meta["history_content_tokens"]
    assert len(struct_chunks) > len(turn_chunks)
    assert struct_meta["structural_partial_docs"] == 0

    _, fixed_chunks, reason, fixed_meta = _condition_doc_chunks(
        tokenizer, example, "joint", chunk_policy="fixed-1024", **kwargs
    )
    assert reason is None
    assert fixed_meta["history_content_tokens"] == turn_meta["history_content_tokens"]
    assert len(fixed_chunks) >= 1


def test_delay_meta_raw_ids():
    tokenizer = _WhitespaceSelfTestTokenizer()
    example = _agent_history_example()
    kwargs = dict(
        max_doc_length=256,
        max_doc_num=8,
        max_tool_chunks=4,
        max_tool_definition_tokens=10000,
        history_selection="tail",
        split_oversized_history_docs=True,
        chunk_policy="agent-turn",
    )
    _, base_chunks, _, base_meta = _condition_doc_chunks(
        tokenizer, example, "joint", delay_recent_turns=0, **kwargs
    )
    _, delay_chunks, _, delay_meta = _condition_doc_chunks(
        tokenizer, example, "joint", delay_recent_turns=1, **kwargs
    )
    # One turn left the compressed grid and reappeared as raw prompt ids.
    assert len(delay_chunks) == len(base_chunks) - 1
    assert delay_meta["delayed_docs"] == 1
    assert delay_meta["raw_recent_tokens"] == len(delay_meta["raw_history_ids"]) > 0
    assert delay_meta["raw_history_ids"] == base_chunks[-1]
    # Presented tokens are conserved: grid + raw == the agent-turn grid.
    assert delay_meta["history_wrapped_tokens"] == base_meta["history_wrapped_tokens"]
    # The delayed doc really is the LAST turn (its observation text is there,
    # and the first turn's observation is not).
    decoded = tokenizer.decode(delay_meta["raw_history_ids"])
    assert "hello world" in decoded
    assert "b.txt" not in decoded


def test_wrap_truncation_helper_flags_ceiling_hits_only():
    """A wrap truncated by max_length is counted; an exact fit is not."""
    from eval_joint_next_action_c2kv import _wrap_history_message_ids

    tokenizer = _WhitespaceSelfTestTokenizer()
    short = {"role": "user", "content": "a b c"}
    ids, truncated = _wrap_history_message_ids(tokenizer, short, 64)
    assert truncated is False
    assert len(ids) < 64

    long = {"role": "user", "content": " ".join(f"w{i}" for i in range(200))}
    ids, truncated = _wrap_history_message_ids(tokenizer, long, 16)
    assert truncated is True
    # _chat_template_ids grants max_length+1 for a bos it then strips; the
    # whitespace tokenizer emits no bos, so the ceiling lands at 17 here.
    assert len(ids) <= 17

    # Exact fit at the ceiling: same length, but nothing was cut -> no count.
    exact_len = len(_wrap_history_message_ids(tokenizer, short, 64)[0])
    ids, truncated = _wrap_history_message_ids(tokenizer, short, exact_len)
    assert len(ids) == exact_len
    assert truncated is False


def test_wrap_truncated_docs_zero_on_normal_paths_and_forwarded():
    from eval_joint_next_action_c2kv import _chunk_meta_fields

    tokenizer = _WhitespaceSelfTestTokenizer()
    example = _agent_history_example()
    kwargs = dict(
        max_doc_length=256,
        max_doc_num=8,
        max_tool_chunks=4,
        max_tool_definition_tokens=10000,
        history_selection="tail",
        split_oversized_history_docs=True,
    )
    for policy in ("agent-turn", "structural", "fixed-1024"):
        _, _, reason, meta = _condition_doc_chunks(
            tokenizer, example, "joint", chunk_policy=policy, **kwargs
        )
        assert reason is None
        assert meta["wrap_truncated_docs"] == 0, policy
    _, _, _, delay_meta = _condition_doc_chunks(
        tokenizer, example, "joint", chunk_policy="agent-turn",
        delay_recent_turns=1, **kwargs
    )
    assert delay_meta["wrap_truncated_docs"] == 0
    # The counter reaches the per-row output through _chunk_meta_fields.
    assert _chunk_meta_fields(delay_meta)["wrap_truncated_docs"] == 0
    assert _chunk_meta_fields({"wrap_truncated_docs": 3})["wrap_truncated_docs"] == 3
    assert _chunk_meta_fields({})["wrap_truncated_docs"] == 0


def test_parse_args_new_flags(tmp_path):
    args = parse_args(["--model", "ckpt"])
    assert args.chunk_policy == "agent-turn"
    assert args.delay_recent_turns == 0
    assert args.qid_manifest is None
    assert args.do_sample is False
    assert args.temperature is None and args.top_p is None
    assert args.gen_seed == 0

    manifest = tmp_path / "eval200.json"
    manifest.write_text('["a:0", "b:1"]', encoding="utf-8")
    args = parse_args([
        "--model", "ckpt",
        "--chunk_policy", "structural",
        "--delay_recent_turns", "1",
        "--qid_manifest", str(manifest),
        "--do_sample", "true",
        "--temperature", "0.7",
        "--top_p", "0.9",
        "--gen_seed", "3",
    ])
    assert args.chunk_policy == "structural"
    assert args.delay_recent_turns == 1
    assert args.do_sample is True
    assert (args.temperature, args.top_p, args.gen_seed) == (0.7, 0.9, 3)

    with pytest.raises(SystemExit):
        parse_args(["--model", "ckpt", "--chunk_policy", "fixed-512", "--delay_recent_turns", "1"])
    with pytest.raises(SystemExit):
        parse_args(["--model", "ckpt", "--chunk_policy", "natural-paragraph"])


def test_do_sample_requires_an_explicit_temperature():
    """Untraceable sampling must not be launchable.

    Without --temperature, transformers falls back to the checkpoint's
    generation_config while the run summary records temperature=null, so the
    decode configuration cannot be recovered from the artefacts.
    """

    with pytest.raises(SystemExit):
        parse_args(["--model", "ckpt", "--do_sample", "true"])
    with pytest.raises(SystemExit):
        parse_args(["--model", "ckpt", "--do_sample", "true", "--top_p", "0.9"])
    # Explicit temperature is enough; top_p may legitimately stay at the
    # library default of 1.0 (no nucleus truncation).
    args = parse_args(["--model", "ckpt", "--do_sample", "true", "--temperature", "0.7"])
    assert args.do_sample is True and args.temperature == 0.7
    # Greedy runs are unaffected.
    assert parse_args(["--model", "ckpt"]).temperature is None


def test_qid_manifest_filter(tmp_path, monkeypatch):
    from eval_joint_next_action_c2kv import _filter_by_manifest, _load_qid_manifest

    def _stub(qid):
        return JointExample(
            qid=qid,
            session_id=qid.split(":", 1)[0],
            tool_documents=["t"],
            history_documents=["h"],
            current_messages=[{"role": "user", "content": "q"}],
            answer="a",
        )

    examples = [_stub("b:1"), _stub("a:0"), _stub("c:2")]

    json_manifest = tmp_path / "m.json"
    json_manifest.write_text('["a:0", "c:2", "missing:9"]', encoding="utf-8")
    kept, missing = _filter_by_manifest(examples, str(json_manifest))
    # Manifest order, not source order.
    assert [example.qid for example in kept] == ["a:0", "c:2"]
    assert missing == ["missing:9"]

    lines_manifest = tmp_path / "m.txt"
    lines_manifest.write_text("c:2\nb:1\n\n", encoding="utf-8")
    kept, missing = _filter_by_manifest(examples, str(lines_manifest))
    assert [example.qid for example in kept] == ["c:2", "b:1"]
    assert missing == []

    dict_manifest = tmp_path / "m2.json"
    dict_manifest.write_text('{"qids": ["b:1"], "sha256": "deadbeef"}', encoding="utf-8")
    assert _load_qid_manifest(str(dict_manifest)) == ["b:1"]

    # evaluate() records the shortfall on the run summary: a shrunken frozen
    # set must never pass silently.
    module = _patch_evaluate_boundary(monkeypatch, [])
    monkeypatch.setattr(module, "_load_examples", lambda args: list(examples))
    monkeypatch.setattr(module, "_generate_one", lambda *a, **k: {"skipped": True})
    args = parse_args([
        "--model", "ckpt",
        "--qid_manifest", str(json_manifest),
        "--compare_modes", "c2kv",
        "--output_file", str(tmp_path / "out.jsonl"),
    ])
    summary = module.evaluate(args)
    assert summary["qid_manifest"] == str(json_manifest)
    assert summary["qid_manifest_missing"] == 1
    assert summary["num_examples"] == 2
    assert summary["chunk_policy"] == "agent-turn"
    assert summary["delay_recent_turns"] == 0
    assert summary["do_sample"] is False

