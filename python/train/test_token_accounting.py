# -*- coding: utf-8 -*-
"""CPU-only unit tests for train/token_accounting.py (U_src/P_src/T_tgt).

No real dataset and no network: joint-scan tests build JointExample records
directly and tokenize with the deterministic whitespace fake
``_WhitespaceSelfTestTokenizer`` shipped with train_data_joint; the CLI test
writes a tiny synthetic agent-llm-traces parquet (same schema as
test_train_data_joint.py); official-mode tests only exercise the
missing-data error paths, which need no data files at all.

Coverage:
a. U_src dedups repeated docs across examples (incl. whitespace variants),
   per-subset vs. global dedup semantics;
b. P_src equals the post-chunking grid non-(-100) count on a small
   constructed example whose history doc spans multiple grid slots;
c. T_tgt counts exactly the supervised (answer + EOS) tokens;
d. the --epochs multiplier scales P_src only (U_src/T_tgt unchanged);
e. official-mode missing-data errors are actionable (mdoc sources and
   --data_root hint; agent parquet discovery and --agent_data hint);
f. CLI mode A end-to-end: synthetic parquet -> JSON report on disk.

Run from the repo root (local venv has torch/transformers/datasets/pytest):
  pytest python/train/test_token_accounting.py -v
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

# Make python/ importable when pytest is invoked from the repo root.
_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from train.token_accounting import main, run_official, scan_joint_examples  # noqa: E402
from train.train_data_joint import (  # noqa: E402
    JointDataset,
    JointExample,
    _WhitespaceSelfTestTokenizer,
)


@pytest.fixture(scope="module")
def tokenizer():
    return _WhitespaceSelfTestTokenizer()


# ---------------------------------------------------------------------------
# Synthetic joint examples.
# ---------------------------------------------------------------------------

_TOOL_DOC = (
    "<TOOL>\n<NAMESPACE> weather\n<NAME> get_weather\n"
    "<DESCRIPTION> Fetch the current weather for one city.\n"
    "<PARAMETERS>\n<PARAM name=\"city\" type=\"string\" required=\"true\">\n"
    "</PARAMETERS>\n</TOOL>"
)
_HISTORY_DOC = "Previous turn\n[User query]\nalpha question one\n[Assistant output]\nalpha answer one"
# Same text as _HISTORY_DOC after whitespace normalization -> same sha1.
_HISTORY_DOC_WS_VARIANT = "Previous   turn\n[User query]\nalpha question one\n[Assistant output]\nalpha answer one\n"
_HISTORY_DOC_B = "Previous turn\n[User query]\nbravo question two\n[Assistant output]\nbravo answer two"

_ANSWER = 'Action:\n<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>'


def _example(
    subset="s",
    qid="t:0",
    tool_docs=None,
    history_docs=None,
    answer=_ANSWER,
):
    return JointExample(
        qid=qid,
        session_id="t",
        tool_documents=tool_docs if tool_docs is not None else [_TOOL_DOC],
        history_documents=history_docs if history_docs is not None else [_HISTORY_DOC],
        current_messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        answer=answer,
        system_prompt="You are a careful data agent.",
        subset=subset,
    )


def _normalized_token_count(tokenizer, text):
    return len(tokenizer.encode(" ".join(text.split()), add_special_tokens=False))


# ---------------------------------------------------------------------------
# a. U_src dedup (per-subset and global).
# ---------------------------------------------------------------------------


def test_u_src_dedups_repeated_docs_across_examples(tokenizer):
    ex1 = _example(subset="alpha", qid="t:0", tool_docs=[_TOOL_DOC], history_docs=[_HISTORY_DOC])
    ex2 = _example(
        subset="bravo",
        qid="t:1",
        tool_docs=[_TOOL_DOC],
        history_docs=[_HISTORY_DOC_WS_VARIANT, _HISTORY_DOC_B],
    )
    report = scan_joint_examples([ex1, ex2], tokenizer)

    # Global dedup: the whitespace variant hashes to _HISTORY_DOC; the shared
    # tool doc is counted once across subsets.
    expected_u = (
        _normalized_token_count(tokenizer, _TOOL_DOC)
        + _normalized_token_count(tokenizer, _HISTORY_DOC)
        + _normalized_token_count(tokenizer, _HISTORY_DOC_B)
    )
    assert report["total"]["U_src"] == expected_u
    assert report["total"]["unique_docs"] == 3

    # Per-subset dedup is WITHIN the subset: the shared tool doc counts in
    # both subsets, and inside "bravo" the whitespace variant IS the first
    # occurrence of _HISTORY_DOC — so subset U_sums exceed the global total.
    assert set(report["per_subset"]) == {"alpha", "bravo"}
    assert report["per_subset"]["alpha"]["unique_docs"] == 2
    assert report["per_subset"]["bravo"]["unique_docs"] == 3
    assert report["per_subset"]["alpha"]["U_src"] == (
        _normalized_token_count(tokenizer, _TOOL_DOC) + _normalized_token_count(tokenizer, _HISTORY_DOC)
    )
    assert report["per_subset"]["bravo"]["U_src"] == expected_u
    # Both examples survive preprocessing and emit one row each.
    assert report["total"]["samples"] == 2
    assert report["skipped_rows"] == {}


# ---------------------------------------------------------------------------
# b. P_src equals the post-chunking grid non-(-100) count.
# ---------------------------------------------------------------------------


def test_p_src_equals_post_chunking_grid_count(tokenizer):
    # A history doc long enough to be split across several grid slots.
    long_history = (
        "Previous turn\n[User query]\n"
        + " ".join(f"tok{i}" for i in range(50))
        + "\n[Assistant output]\n"
        + " ".join(f"ans{i}" for i in range(50))
    )
    example = _example(history_docs=[long_history, _HISTORY_DOC_B])
    config = dict(max_length=256, max_doc_length=32, min_doc_num=2, max_doc_num=8, max_system_length=96)

    row, reason = JointDataset.preprocess_example(example, tokenizer=tokenizer, **config)
    assert row is not None, reason
    # Sanity: the long doc really was chunked into multiple grid slots.
    slots = [
        row["context_input_ids"][start * 32 : (start + 1) * 32]
        for start in range(config["max_doc_num"])
    ]
    used_slots = sum(1 for slot in slots if any(token_id != -100 for token_id in slot))
    assert used_slots >= 3  # 1 tool doc + >=2 history chunks

    expected_p = sum(1 for token_id in row["context_input_ids"] if token_id != -100)
    report = scan_joint_examples([example], tokenizer, **config)
    assert report["total"]["samples"] == 1
    assert report["total"]["P_src"] == expected_p
    assert report["total"]["P_src_per_epoch"] == expected_p


# ---------------------------------------------------------------------------
# c. T_tgt counts exactly the supervised tokens.
# ---------------------------------------------------------------------------


def test_t_tgt_counts_supervised_tokens(tokenizer):
    example = _example()
    config = dict(max_length=256, max_doc_length=64, min_doc_num=2, max_doc_num=6, max_system_length=96)
    row, reason = JointDataset.preprocess_example(example, tokenizer=tokenizer, **config)
    assert row is not None, reason
    expected_t = sum(1 for token_id in row["labels"] if token_id != -100)
    # Untruncated: supervised region is exactly answer + EOS.
    assert expected_t == len(tokenizer.encode(example.answer, add_special_tokens=False)) + 1
    report = scan_joint_examples([example], tokenizer, **config)
    assert report["total"]["T_tgt"] == expected_t


# ---------------------------------------------------------------------------
# d. --epochs scales P_src only.
# ---------------------------------------------------------------------------


def test_epochs_multiplier_scales_p_src_only(tokenizer):
    examples = [
        _example(qid="t:0"),
        _example(qid="t:1", history_docs=[_HISTORY_DOC_B, _HISTORY_DOC]),
    ]
    config = dict(max_length=256, max_doc_length=64, min_doc_num=2, max_doc_num=6, max_system_length=96)
    one = scan_joint_examples(examples, tokenizer, epochs=1, **config)
    three = scan_joint_examples(examples, tokenizer, epochs=3, **config)
    assert one["total"]["P_src"] > 0
    assert three["total"]["P_src"] == 3 * one["total"]["P_src"]
    assert three["total"]["P_src_per_epoch"] == one["total"]["P_src"]
    # U_src is epoch-invariant; T_tgt is reported per epoch.
    assert three["total"]["U_src"] == one["total"]["U_src"]
    assert three["total"]["unique_docs"] == one["total"]["unique_docs"]
    assert three["total"]["T_tgt"] == one["total"]["T_tgt"]
    assert three["epochs"] == 3


# ---------------------------------------------------------------------------
# e. Official-mode missing-data errors are actionable.
# ---------------------------------------------------------------------------


def _official_args(tmp_path, **overrides):
    args = argparse.Namespace(
        data_root=str(tmp_path / "datasets"),
        train_data_cleaned=None,
        mdoc_sources="hotpotqa,wikimqa,longmagpie",
        mdoc_num_samples="32768",
        agent_data=None,
        agent_num_samples=130000,
        num_proc=1,
        split_seed=42,
        tokenizer="fake",
        out=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_official_missing_mdoc_data_message(tmp_path, tokenizer):
    (tmp_path / "datasets").mkdir()
    args = _official_args(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        run_official(args, tokenizer)
    message = str(excinfo.value)
    for source in ("hotpotqa", "wikimqa", "longmagpie"):
        assert source in message
    assert "--data_root" in message
    # The error lists the concrete candidate paths that were tried.
    assert "hotpotqa_train_cleaned" in message
    assert "wikimqa_train_cleaned" in message
    assert "longmagpie_cleaned" in message


def test_official_missing_agent_data_message(tmp_path, tokenizer):
    args = _official_args(tmp_path, agent_data=str(tmp_path / "open-swe"))
    with pytest.raises(FileNotFoundError) as excinfo:
        run_official(args, tokenizer)
    message = str(excinfo.value)
    assert "--agent_data" in message
    assert "parquet" in message


# ---------------------------------------------------------------------------
# f. CLI mode A end-to-end on a tiny synthetic traces parquet.
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch the current weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search files under one directory path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
]


def _write_synthetic_traces(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = tmp_path / "agent-llm-traces"
    data_dir.mkdir()
    spans = [
        {
            "span_id": "span-1",
            "start_time": "2026-01-01T00:00:01",
            "status": "ok",
            "attributes": {
                "gen_ai.tool.definitions": json.dumps(_TOOLS),
                "gen_ai.input.messages": json.dumps([
                    {"role": "system", "content": "You are a weather agent."},
                    {"role": "user", "content": "List the files in /tmp please."},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "search_files", "arguments": "{\"path\": \"/tmp\"}"},
                        }],
                    },
                    {"role": "tool", "content": "found a.txt and b.txt"},
                    {"role": "user", "content": "Now get the weather in Paris."},
                ]),
                "gen_ai.output.messages": json.dumps([
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "c2",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"},
                        }],
                    }
                ]),
            },
        }
    ]
    table = pa.table({
        "benchmark": ["weather-bench"],
        "session_id": ["sess-1"],
        "spans": [json.dumps(spans)],
    })
    pq.write_table(table, data_dir / "shard.parquet")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"train_session_ids": ["sess-1"], "eval_session_ids": []}),
        encoding="utf-8",
    )
    return data_dir, manifest


def test_cli_joint_writes_json(tmp_path):
    data_dir, manifest = _write_synthetic_traces(tmp_path)
    out = tmp_path / "report.json"
    main([
        "joint",
        "--dataset_path", str(data_dir),
        "--split_manifest_file", str(manifest),
        "--tokenizer", "fake",
        "--max_length", "256",
        "--max_doc_length", "64",
        "--max_system_length", "96",
        "--max_doc_num", "6",
        "--epochs", "2",
        "--out", str(out),
    ])
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["mode"] == "joint"
    assert report["dataset_path"] == str(data_dir)
    assert report["examples_scanned"] == 1
    assert report["total"]["samples"] == 1
    assert report["total"]["U_src"] > 0
    assert report["total"]["P_src"] > 0
    assert report["total"]["T_tgt"] > 0
    assert report["total"]["P_src"] == 2 * report["total"]["P_src_per_epoch"]
    assert set(report["per_subset"]) == {"weather-bench"}
    assert report["notes"]

    # Deterministic: a second run over the same input yields the same report.
    out2 = tmp_path / "report2.json"
    main([
        "joint",
        "--dataset_path", str(data_dir),
        "--split_manifest_file", str(manifest),
        "--tokenizer", "fake",
        "--max_length", "256",
        "--max_doc_length", "64",
        "--max_system_length", "96",
        "--max_doc_num", "6",
        "--epochs", "2",
        "--out", str(out2),
    ])
    assert json.loads(out2.read_text(encoding="utf-8")) == report
