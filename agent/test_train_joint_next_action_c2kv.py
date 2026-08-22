# -*- coding: utf-8 -*-
"""CPU-only unit tests for agent/train_joint_next_action_c2kv.py.

Covers the entry's pure-logic pieces (no model, no real dataset, no network):

a. ``_interleave_rows``: deterministic seeded round-robin — strict
   alternation, leftover tail in order, both leaders reachable across seeds,
   output is a permutation of the inputs;
b. ``_take_within_source_token_budget``: greedy prefix take stops once the
   cumulative pre-chunking estimate (tool + history document tokens) reaches
   the budget, crossing example included, achieved estimate returned;
c. ``_apply_example_order_file``: filter + reorder to exactly the file's qid
   list; hard errors on unknown, duplicate, or malformed qid lists and on
   duplicate qids in the loaded examples;
d. ``MinTargetJointDataset``: the tool-definition path's min_target_tokens
   reservation floor — rows whose answer was truncated below the reserved
   floor are dropped, fully-fitting answers are always kept (whitespace
   tokenizer from train.train_data_joint, mirroring test_train_data_joint.py);
e. ``_dump_train_manifest``: effective train qid order + per-subset counts.

Run from the repo root (local venv has torch/transformers/datasets/pytest):
  pytest agent/test_train_joint_next_action_c2kv.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make python/ and agent/ importable when pytest is invoked from the repo root
# (the entry imports gist_args at module top, before its own sys.path fix).
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from train.train_data_joint import JointExample, _WhitespaceSelfTestTokenizer  # noqa: E402
from train_joint_next_action_c2kv import (  # noqa: E402
    JointDataArgs,
    MinTargetJointDataset,
    _apply_example_order_file,
    _dump_train_manifest,
    _interleave_rows,
    _take_within_source_token_budget,
)


def _example(qid, subset="bench", tool_words=3, history_words=4, current_words=8):
    current = " ".join(f"word{i}" for i in range(current_words))
    return JointExample(
        qid=qid,
        session_id=qid.split(":", 1)[0],
        tool_documents=[" ".join(f"tool{i}" for i in range(tool_words))],
        history_documents=[" ".join(f"hist{i}" for i in range(history_words))],
        current_messages=[{"role": "user", "content": current}],
        answer='Action:\n<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>',
        system_prompt="You are a careful data agent.",
        subset=subset,
    )


# ---------------------------------------------------------------------------
# _interleave_rows
# ---------------------------------------------------------------------------


def _rows(tag, n):
    return [{"row": f"{tag}{index}"} for index in range(n)]


def test_interleave_rows_is_deterministic_per_seed():
    tool_rows, history_rows = _rows("t", 5), _rows("h", 5)
    for seed in range(5):
        assert _interleave_rows(tool_rows, history_rows, seed) == _interleave_rows(
            tool_rows, history_rows, seed
        )


def test_interleave_rows_strict_alternation_and_permutation():
    tool_rows, history_rows = _rows("t", 4), _rows("h", 4)
    out = _interleave_rows(tool_rows, history_rows, seed=42)
    assert len(out) == 8
    leader = out[0]["row"][0]
    assert leader in ("t", "h")
    for index, row in enumerate(out):
        expected = leader if index % 2 == 0 else ("h" if leader == "t" else "t")
        assert row["row"].startswith(expected)
    assert sorted(row["row"] for row in out) == sorted(
        row["row"] for row in tool_rows + history_rows
    )


def test_interleave_rows_appends_leftover_tail_in_order():
    out = _interleave_rows(_rows("t", 2), _rows("h", 5), seed=1)
    ids = [row["row"] for row in out]
    assert len(ids) == 7
    # The three leftover history rows keep their relative order at the tail.
    assert ids[-3:] == ["h2", "h3", "h4"]


def test_interleave_rows_both_leaders_reachable():
    leaders = {
        _interleave_rows(_rows("t", 1), _rows("h", 1), seed)[0]["row"]
        for seed in range(10)
    }
    assert leaders == {"t0", "h0"}


# ---------------------------------------------------------------------------
# _take_within_source_token_budget
# ---------------------------------------------------------------------------


def test_budget_greedy_take_stops_at_budget():
    tokenizer = _WhitespaceSelfTestTokenizer()
    examples = [_example(f"s:{i}") for i in range(4)]  # 3 + 4 = 7 tokens each
    kept, achieved = _take_within_source_token_budget(examples, tokenizer, 14)
    assert [example.qid for example in kept] == ["s:0", "s:1"]
    assert achieved == 14
    # Crossing example is included: budget 15 stops after the third example.
    kept, achieved = _take_within_source_token_budget(examples, tokenizer, 15)
    assert [example.qid for example in kept] == ["s:0", "s:1", "s:2"]
    assert achieved == 21


def test_budget_edges():
    tokenizer = _WhitespaceSelfTestTokenizer()
    examples = [_example(f"s:{i}") for i in range(3)]
    kept, achieved = _take_within_source_token_budget(examples, tokenizer, 0)
    assert kept == [] and achieved == 0
    kept, achieved = _take_within_source_token_budget(examples, tokenizer, 10**9)
    assert len(kept) == 3 and achieved == 21


# ---------------------------------------------------------------------------
# _apply_example_order_file
# ---------------------------------------------------------------------------


def _write_order_file(tmp_path, payload):
    path = tmp_path / "order.json"
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
    return str(path)


def test_example_order_file_filters_and_reorders_exactly(tmp_path):
    examples = [_example("s:0"), _example("s:1"), _example("s:2")]
    order_file = _write_order_file(tmp_path, ["s:2", "s:0"])
    ordered = _apply_example_order_file(examples, order_file)
    assert [example.qid for example in ordered] == ["s:2", "s:0"]


def test_example_order_file_hard_error_on_unknown_qid(tmp_path):
    examples = [_example("s:0"), _example("s:1")]
    order_file = _write_order_file(tmp_path, ["s:0", "s:nope"])
    with pytest.raises(ValueError, match="missing"):
        _apply_example_order_file(examples, order_file)


def test_example_order_file_hard_error_on_duplicate_qid(tmp_path):
    examples = [_example("s:0"), _example("s:1")]
    order_file = _write_order_file(tmp_path, ["s:0", "s:1", "s:0"])
    with pytest.raises(ValueError, match="duplicate"):
        _apply_example_order_file(examples, order_file)


def test_example_order_file_hard_error_on_malformed_json(tmp_path):
    examples = [_example("s:0")]
    order_file = _write_order_file(tmp_path, {"s:0": 1})
    with pytest.raises(ValueError, match="JSON list"):
        _apply_example_order_file(examples, order_file)


def test_example_order_file_hard_error_on_duplicate_data_qid(tmp_path):
    examples = [_example("s:0"), _example("s:0")]
    order_file = _write_order_file(tmp_path, ["s:0"])
    with pytest.raises(RuntimeError, match="duplicate qid"):
        _apply_example_order_file(examples, order_file)


# ---------------------------------------------------------------------------
# MinTargetJointDataset (whitespace tokenizer, offline)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tokenizer():
    return _WhitespaceSelfTestTokenizer()


def _dataset(tokenizer, examples, min_target_tokens, **overrides):
    config = dict(
        tokenizer=tokenizer,
        max_length=128,
        max_doc_length=64,
        min_doc_num=2,
        max_doc_num=6,
        max_system_length=96,
        min_target_tokens=min_target_tokens,
    )
    config.update(overrides)
    return MinTargetJointDataset(examples, **config)


def test_min_target_tokens_keeps_fully_fitting_answer(tokenizer):
    example = _example("s:0", tool_words=8, history_words=20)
    dataset = _dataset(tokenizer, [example], min_target_tokens=32)
    assert len(dataset) == 1  # answer (5 supervised tokens) fits in full: kept


def test_min_target_tokens_drops_answer_truncated_below_floor(tokenizer):
    # Tiny max_length + long current turn: the prompt eats the sequence
    # budget and the answer is truncated to a single supervised token.
    example = _example("s:0", tool_words=8, history_words=20, current_words=60)
    config = dict(max_length=24, min_target_tokens=3)
    dataset = _dataset(tokenizer, [example], **config)
    assert len(dataset) == 0
    dataset = _dataset(tokenizer, [example], max_length=24, min_target_tokens=1)
    assert len(dataset) == 1


# ---------------------------------------------------------------------------
# _dump_train_manifest
# ---------------------------------------------------------------------------


def test_dump_train_manifest_records_order_and_counts(tmp_path):
    data_args = JointDataArgs(doc_mode="alternate", split_seed=7)
    train_examples = [
        _example("s:0", subset="a"),
        _example("s:1", subset="a"),
        _example("s:2", subset="b"),
        _example("toucan:uuid-1:u2", subset="toucan:multi-turn"),
    ]
    eval_examples = [_example("e:0", subset="b")]
    path = _dump_train_manifest(
        data_args,
        str(tmp_path),
        train_examples,
        eval_examples,
        interleaved_train_len=6,
        achieved_source_tokens=None,
    )
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    assert path == str(tmp_path / "train_manifest_used.json")
    assert manifest["doc_mode"] == "alternate"
    assert manifest["train_qids"] == ["s:0", "s:1", "s:2", "toucan:uuid-1:u2"]
    assert manifest["train_subset_counts"] == {"a": 2, "b": 1, "toucan:multi-turn": 1}
    # qid-family source counts: bare session:span qids count as "traces".
    assert manifest["train_source_counts"] == {"traces": 3, "toucan": 1}
    assert manifest["interleaved_train_len"] == 6
    assert manifest["eval_qids"] == ["e:0"]
    # Regime string is always recorded (v2 = empty-tool reclaim by default).
    assert manifest["cap_regime"] == "per_side_caps_v2_empty_tool_reclaim"
    assert manifest["legacy_mode_caps"] is False
    legacy_args = JointDataArgs(doc_mode="joint", legacy_mode_caps=True)
    legacy_path = _dump_train_manifest(
        legacy_args, str(tmp_path / "legacy"), train_examples, eval_examples, None, None
    )
    assert json.loads(Path(legacy_path).read_text())["cap_regime"] == "legacy_mode_caps"


def test_dump_train_manifest_skip_and_retention_breakdowns(tmp_path):
    # P1-7/P0-1 manifest audit: per-pass per-family skip counts and the QA
    # retention counters from the history-bearing passes.
    from train.train_data_joint import JointDataset, _WhitespaceSelfTestTokenizer

    qa_example = JointExample(
        qid="qa:hotpotqa:h1",
        session_id="qa:hotpotqa:h1",
        tool_documents=[],
        history_documents=["Document 1 (title: A) some words here", "Document 2 (title: B) more words"],
        current_messages=[{"role": "user", "content": "question?"}],
        answer="yes",
        subset="qa:hotpotqa",
        gold_history_doc_indices=(1,),
    )
    tokenizer = _WhitespaceSelfTestTokenizer()
    kwargs = dict(
        tokenizer=tokenizer, max_length=256, max_doc_length=128, min_doc_num=2,
        max_doc_num=4, max_system_length=128,
    )
    tool_pass = JointDataset([qa_example], doc_mode="tool_only", **kwargs)
    history_pass = JointDataset([qa_example], doc_mode="history_only", **kwargs)
    path = _dump_train_manifest(
        JointDataArgs(doc_mode="alternate"),
        str(tmp_path),
        [qa_example],
        [],
        interleaved_train_len=1,
        achieved_source_tokens=None,
        train_datasets={"tool_only": tool_pass, "history_only": history_pass},
    )
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    # The tool_only pass drops the QA example, attributed per family (P1-7).
    assert manifest["train_skip_counts_by_family"]["tool_only"] == {"qa:doc_num<2": 1}
    assert manifest["train_skip_counts_by_family"]["history_only"] == {}
    # Retention recorded from the history-bearing pass only.
    assert manifest["qa_retention"] == {
        "history_only": {
            "qa_history_doc_retention": {"kept": 2, "total": 2},
            "qa_gold_doc_retention": {"kept": 1, "total": 1},
            "qa_history_truncated_examples_by_subset": {},
        }
    }
