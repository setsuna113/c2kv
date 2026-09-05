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
e. ``_dump_train_manifest``: effective train qid order + per-subset counts,
   plus the ``action_type_counts`` / ``tool_call_target_truncated_skips``
   audit fields.

Run from the repo root (local venv has torch/transformers/datasets/pytest):
  pytest agent/test_train_joint_next_action_c2kv.py -v
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

# The entry pulls python/models (torch) at import time, so the whole module is
# torch-gated: without this the file ERRORs at collection on a torch-free box
# instead of skipping.
pytest.importorskip("torch")

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
    _estimate_source_tokens,
    _interleave_rows,
    _parse_hybrid_tail_choices,
    _take_within_source_token_budget,
    _validate_regime_args,
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
    # Plain-text answer: a tool-call answer now hits the
    # tool_call_target_truncated integrity guard before this floor instead.
    example = dataclasses.replace(
        _example("s:0", tool_words=8, history_words=20, current_words=60),
        answer=" ".join(f"word{i}" for i in range(30)),
    )
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


def test_dump_train_manifest_action_type_and_truncation_audit(tmp_path):
    # Global action-type counts come from the extraction-time tag on each
    # example; the truncation skip is aggregated per pass like the other
    # skip counters.
    from train.train_data_joint import JointDataset

    examples = [
        dataclasses.replace(_example("s:0"), action_type="tool_call"),
        dataclasses.replace(_example("s:1"), action_type="tool_call"),
        _example("s:2"),  # default action_type="other"
    ]
    path = _dump_train_manifest(
        JointDataArgs(doc_mode="joint"), str(tmp_path), examples, [], None, None
    )
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    assert manifest["action_type_counts"] == {"tool_call": 2, "other": 1}
    assert "tool_call_target_truncated_skips" not in manifest  # no datasets passed

    # A tool-call answer that cannot fit the sequence budget is dropped, not
    # truncated, and the skip surfaces per pass.
    big = _example("s:big", current_words=60)
    dataset = JointDataset(
        [big],
        tokenizer=_WhitespaceSelfTestTokenizer(),
        max_length=24,
        max_doc_length=64,
        min_doc_num=2,
        max_doc_num=6,
        max_system_length=96,
    )
    assert dataset.skipped_by_reason == {"tool_call_target_truncated": 1}
    path = _dump_train_manifest(
        JointDataArgs(doc_mode="joint"),
        str(tmp_path / "audit"),
        [_example("s:0")],
        [],
        None,
        None,
        train_datasets={"joint": dataset},
    )
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    assert manifest["tool_call_target_truncated_skips"] == {"joint": 1}


# ---------------------------------------------------------------------------
# Regime-first argument plumbing (tools_in_system / hybrid_tail_choices)
# ---------------------------------------------------------------------------


def test_estimate_source_tokens_excludes_tool_docs_under_tools_in_system():
    tokenizer = _WhitespaceSelfTestTokenizer()
    example = _example("s:0", tool_words=3, history_words=4)
    # Default and explicit-off are unchanged: tool docs + history docs.
    assert _estimate_source_tokens(example, tokenizer) == 7
    assert _estimate_source_tokens(example, tokenizer, tools_in_system=False) == 7
    # tools_in_system never presents the tool documents through the gist path.
    assert _estimate_source_tokens(example, tokenizer, tools_in_system=True) == 4


def test_source_token_budget_honours_tools_in_system():
    tokenizer = _WhitespaceSelfTestTokenizer()
    examples = [_example(f"s:{i}") for i in range(4)]  # 3 tool + 4 history tokens each
    kept, achieved = _take_within_source_token_budget(examples, tokenizer, 8)
    assert [example.qid for example in kept] == ["s:0", "s:1"]
    assert achieved == 14
    kept, achieved = _take_within_source_token_budget(
        examples, tokenizer, 8, tools_in_system=True
    )
    assert [example.qid for example in kept] == ["s:0", "s:1"]
    assert achieved == 8


def test_parse_hybrid_tail_choices():
    assert _parse_hybrid_tail_choices(None) == []
    assert _parse_hybrid_tail_choices("") == []
    assert _parse_hybrid_tail_choices("  ") == []
    assert _parse_hybrid_tail_choices("0,0,1,3,5") == [0, 0, 1, 3, 5]
    assert _parse_hybrid_tail_choices(" 0 , 2 ") == [0, 2]
    with pytest.raises(ValueError, match="non-negative"):
        _parse_hybrid_tail_choices("0,-1")
    with pytest.raises(ValueError, match="integers"):
        _parse_hybrid_tail_choices("0,x")


def test_validate_regime_args_rejects_tools_in_system_outside_history_only():
    for mode in ("joint", "tool_only", "alternate"):
        with pytest.raises(ValueError, match="history_only"):
            _validate_regime_args(JointDataArgs(doc_mode=mode, tools_in_system=True))
    assert _validate_regime_args(JointDataArgs(doc_mode="history_only", tools_in_system=True)) == []
    assert _validate_regime_args(
        JointDataArgs(doc_mode="history_only", tools_in_system=True, hybrid_tail_choices="0,0,1,3,5")
    ) == [0, 0, 1, 3, 5]
    # Defaults stay off.
    assert _validate_regime_args(JointDataArgs()) == []


def test_dump_train_manifest_records_regime_fields(tmp_path):
    from train.train_data_joint import JointDataset

    examples = [_example(f"s:{index}", history_words=60) for index in range(6)]
    dataset = JointDataset(
        examples,
        tokenizer=_WhitespaceSelfTestTokenizer(),
        max_length=512,
        max_doc_length=16,
        min_doc_num=1,
        max_doc_num=6,
        max_tool_chunks=0,
        max_system_length=96,
        doc_mode="history_only",
        hybrid_tail_choices=[0, 1],
    )
    assert len(dataset) == len(examples)
    path = _dump_train_manifest(
        JointDataArgs(
            doc_mode="history_only", tools_in_system=True, hybrid_tail_choices="0,1"
        ),
        str(tmp_path),
        examples,
        [],
        None,
        None,
        train_datasets={"history_only": dataset},
    )
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    assert manifest["tools_in_system"] is True
    assert manifest["hybrid_tail_choices"] == "0,1"
    assert manifest["system_overflow_skips"] == {"history_only": 0}
    counts = manifest["hybrid_tail_k_counts"]["history_only"]
    assert set(counts) == {"0", "1"}  # both pool values realized
    assert sum(counts.values()) == len(dataset)

    # Defaults: knobs recorded as off.
    default_path = _dump_train_manifest(
        JointDataArgs(), str(tmp_path / "default"), examples, [], None, None
    )
    default_manifest = json.loads(Path(default_path).read_text(encoding="utf-8"))
    assert default_manifest["tools_in_system"] is False
    assert default_manifest["hybrid_tail_choices"] is None


def test_dump_train_manifest_records_drawn_k_and_eval_overflow(tmp_path):
    # Review round 2: the realized-k histogram alone cannot distinguish "never
    # drawn" from "every row that drew it was dropped", and the eval dataset --
    # built in the SAME dialect -- had no system_overflow counter anywhere.
    from train.train_data_joint import JointDataset

    tokenizer = _WhitespaceSelfTestTokenizer()
    examples = [_example(f"s:{index}", history_words=60) for index in range(6)]
    kwargs = dict(
        tokenizer=tokenizer,
        max_length=512,
        max_doc_length=16,
        min_doc_num=1,
        max_doc_num=6,
        max_tool_chunks=0,
        max_system_length=96,
        doc_mode="history_only",
    )
    dataset = JointDataset(examples, hybrid_tail_choices=[0, 1], **kwargs)
    # A tiny max_system_length makes every tools_in_system row overflow.
    eval_kwargs = dict(kwargs)
    eval_kwargs["max_system_length"] = 1
    eval_dataset = JointDataset(
        [dataclasses.replace(example, selected_tools=[{"type": "function", "function": {"name": "t"}}])
         for example in examples],
        tools_in_system=True,
        **eval_kwargs,
    )
    assert len(eval_dataset) == 0
    assert eval_dataset.skipped_by_reason == {"system_overflow": len(examples)}

    path = _dump_train_manifest(
        JointDataArgs(doc_mode="history_only", tools_in_system=True, hybrid_tail_choices="0,1"),
        str(tmp_path),
        examples,
        examples,
        None,
        None,
        train_datasets={"history_only": dataset},
        eval_dataset=eval_dataset,
    )
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    drawn = manifest["hybrid_tail_k_drawn_counts"]["history_only"]
    assert set(drawn) == {"0", "1"}
    assert sum(drawn.values()) == len(examples)  # every candidate, not just emitted
    assert manifest["tools_in_system_missing_tools"] == {"history_only": 0}
    assert manifest["num_eval_rows"] == 0
    assert manifest["eval_system_overflow_skips"] == len(examples)
    assert manifest["eval_skip_counts"] == {"system_overflow": len(examples)}

    # Without an eval dataset the eval keys stay absent (unchanged callers).
    plain = _dump_train_manifest(
        JointDataArgs(), str(tmp_path / "plain"), examples, [], None, None,
        train_datasets={"history_only": dataset},
    )
    plain_manifest = json.loads(Path(plain).read_text(encoding="utf-8"))
    assert "num_eval_rows" not in plain_manifest
    assert "eval_system_overflow_skips" not in plain_manifest


def test_dump_train_manifest_counts_tools_in_system_rows_without_tools(tmp_path):
    from train.train_data_joint import JointDataset

    tokenizer = _WhitespaceSelfTestTokenizer()
    with_tools = dataclasses.replace(
        _example("s:0", history_words=20),
        selected_tools=[{"type": "function", "function": {"name": "t"}}],
    )
    without_tools = dataclasses.replace(_example("s:1", history_words=20), selected_tools=None)
    dataset = JointDataset(
        [with_tools, without_tools],
        tokenizer=tokenizer,
        max_length=512,
        max_doc_length=64,
        min_doc_num=1,
        max_doc_num=6,
        max_tool_chunks=0,
        max_system_length=256,
        doc_mode="history_only",
        tools_in_system=True,
    )
    path = _dump_train_manifest(
        JointDataArgs(doc_mode="history_only", tools_in_system=True),
        str(tmp_path),
        [with_tools, without_tools],
        [],
        None,
        None,
        train_datasets={"history_only": dataset},
    )
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    # A bare system prefix under tools_in_system presents NO tools at all: not a
    # skip, but it must be visible in the manifest.
    assert manifest["tools_in_system_missing_tools"] == {"history_only": 1}
