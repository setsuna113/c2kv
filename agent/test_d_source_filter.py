"""D source filtering happens before tokenizer-heavy history selection."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_agent_history_c2kv as history  # noqa: E402
from train.train_data_multiturn import CompressHistoryExample  # noqa: E402


def _example(qid):
    return CompressHistoryExample(
        qid=qid,
        history_messages=[{"role": "user", "content": "history"}],
        current_messages=[{"role": "user", "content": "current"}],
        answer="answer",
    )


def _args(**overrides):
    values = {
        "dataset_path": "fixture",
        "split": "eval",
        "eval_ratio": 0.1,
        "split_seed": 42,
        "split_manifest_file": None,
        "split_manifest_name": "subset_disjoint",
        "max_samples_per_session": 4,
        "max_source_examples": None,
        "require_tool_call": False,
        "max_input_chars": None,
        "max_answer_chars": None,
        "include_tools": True,
        "prefix_history_doc_num": None,
        "prefix_history_exact": False,
        "selection_filter": "c2kv",
        "max_examples": 0,
        "selected_qids": ["session-b:0"],
        "selected_sessions": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_prefilter_keeps_unrequested_examples_out_of_tokenizer_path(monkeypatch):
    candidates = [_example("session-a:0"), _example("session-b:0")]
    captured = {}

    class FakeSource:
        def __init__(self, path, **kwargs):
            captured.update(kwargs)
            qids = kwargs.get("selected_qids")
            self.records = [example for example in candidates
                            if qids is None or example.qid in set(qids)]

        def __iter__(self):
            return iter(self.records)

    tokenized = []
    monkeypatch.setattr(history, "AgentLLMTracesCompressHistorySource", FakeSource)
    monkeypatch.setattr(
        history, "_build_history_chunks",
        lambda tokenizer, example, args: (
            tokenized.append(example.qid) or (None, None, None, None, None)
        ),
    )

    examples, _ = history._load_examples(_args(), tokenizer=object())
    assert captured["selected_qids"] == ["session-b:0"]
    assert [example.qid for example in examples] == ["session-b:0"]
    assert tokenized == ["session-b:0"]


def test_global_max_examples_uses_full_path_then_filters(monkeypatch):
    candidates = [_example(f"session-{name}:0") for name in ("a", "b", "c")]
    captured = {}

    class FakeSource:
        def __init__(self, path, **kwargs):
            captured.update(kwargs)

        def __iter__(self):
            return iter(candidates)

    tokenized = []
    monkeypatch.setattr(history, "AgentLLMTracesCompressHistorySource", FakeSource)
    monkeypatch.setattr(
        history, "_build_history_chunks",
        lambda tokenizer, example, args: (
            tokenized.append(example.qid) or (None, None, None, None, None)
        ),
    )

    examples, _ = history._load_examples(
        _args(max_examples=2, selected_qids=["session-b:0"]), tokenizer=object())
    assert captured["selected_qids"] is None
    assert tokenized == ["session-a:0", "session-b:0"]
    assert [example.qid for example in examples] == ["session-b:0"]
