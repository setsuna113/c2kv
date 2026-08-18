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


def test_tool_only_drops_history_and_gives_tools_all_slots():
    tokenizer = _WhitespaceSelfTestTokenizer()
    # 6 tool docs of ~8 tokens each -> more than the joint tool cap of
    # min(2*4//3, 4) = 2 chunks but within the tool_only cap of 4 chunks.
    example = _example(tool_docs=6, history_docs=2, doc_words=6)
    joint_tools, joint_history, _ = _chunks(tokenizer, example, "joint", max_doc_length=16)
    only_tools, only_history, _ = _chunks(tokenizer, example, "tool_only", max_doc_length=16)
    assert only_history == []
    assert len(joint_tools) == 2  # joint cap: min(_default_max_tool_chunks(4)=2, 4)
    assert len(only_tools) > len(joint_tools)  # tool_only cap: all 4 slots
    assert joint_history  # joint still fits history in the remaining slots


def test_history_only_drops_tools_and_gives_history_all_slots():
    tokenizer = _WhitespaceSelfTestTokenizer()
    example = _example(tool_docs=2, history_docs=6, doc_words=6)
    joint_tools, joint_history, _ = _chunks(tokenizer, example, "joint", max_doc_length=16)
    only_tools, only_history, _ = _chunks(tokenizer, example, "history_only", max_doc_length=16)
    assert only_tools == []
    assert len(only_history) > len(joint_history)
    assert joint_tools


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
