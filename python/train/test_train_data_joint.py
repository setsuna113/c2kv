# -*- coding: utf-8 -*-
"""CPU-only unit tests for train/train_data_joint.py (true-joint C2KV data).

No real dataset and no network: the source tests build synthetic spans
mirroring the agent-llm-traces parquet schema (``gen_ai.input.messages`` /
``gen_ai.output.messages`` / ``gen_ai.tool.definitions`` JSON strings plus
``start_time`` / ``span_id`` / ``status``) and write them to a tiny parquet
file with pyarrow; the tokenizer is the deterministic whitespace fake
``_WhitespaceSelfTestTokenizer`` shipped with the module for ``--self_test``.

Coverage:
a. source parsing: session/span parsing, span (start_time, span_id) sorting,
   history/current split, per-session tool-document rendering, subset and
   qid propagation, ``max_samples_per_session``;
b. output keys/shapes vs. the two existing datasets (GistMultiDocTrainer
   contract);
c. leakage self-checks: pass on clean input, FAIL when tool text is injected
   into the system prefix or the prompt (negative controls);
d. chronological history order in the flat context grid;
e. doc_mode subsets (joint / tool_only / history_only);
f. label masking boundaries (prompt -100, answer+EOS supervised);
g. tail-biased history truncation and tool/history budget allocation.

Run from the repo root (local venv has torch/transformers/datasets/pytest):
  pytest python/train/test_train_data_joint.py -v
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

# Make python/ importable when pytest is invoked from the repo root.
_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import train.train_data_joint as tdj  # noqa: E402
from train.train_data_joint import (  # noqa: E402
    AgentLLMTracesJointSource,
    JointDataset,
    JointExample,
    _WhitespaceSelfTestTokenizer,
    assert_no_leakage,
)
from train.train_data_multiturn import _chat_template_ids, _pad  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic agent-llm-traces rows (parquet schema mirrors the real dataset).
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


def _span(span_id, start_time, input_messages, output_messages, tools=None):
    attributes = {
        "gen_ai.input.messages": json.dumps(input_messages),
        "gen_ai.output.messages": json.dumps(output_messages),
    }
    if tools is not None:
        attributes["gen_ai.tool.definitions"] = json.dumps(tools)
    return {
        "span_id": span_id,
        "start_time": start_time,
        "status": "ok",
        "attributes": attributes,
    }


def _tool_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _session_one_spans():
    system = {"role": "system", "content": "You are a weather agent."}
    user1 = {"role": "user", "content": "List the files in /tmp please."}
    assistant1 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call("c1", "search_files", {"path": "/tmp"})],
    }
    tool1 = {"role": "tool", "content": "found a.txt and b.txt"}
    user2 = {"role": "user", "content": "Now get the weather in Paris."}
    assistant2 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call("c2", "get_weather", {"city": "Paris"})],
    }
    tool2 = {"role": "tool", "content": "It is rainy in Paris."}
    user3 = {"role": "user", "content": "Thanks, summarize the results."}
    span_early = _span(
        "span-1",
        "2026-01-01T00:00:01",
        [system, user1, assistant1, tool1, user2],
        [assistant2],
        tools=_TOOLS,
    )
    span_late = _span(
        "span-2",
        "2026-01-01T00:00:02",
        [system, user1, assistant1, tool1, user2, assistant2, tool2, user3],
        [{"role": "assistant", "content": "Summary: found a.txt and b.txt; Paris is rainy."}],
    )
    # Deliberately out of chronological order: the source must sort spans.
    return [span_late, span_early]


def _session_two_spans():
    system = {"role": "system", "content": "You are a files agent."}
    user1 = {"role": "user", "content": "Create notes.txt for me."}
    assistant1 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call("c3", "write_file", {"path": "notes.txt"})],
    }
    tool1 = {"role": "tool", "content": "file created"}
    user2 = {"role": "user", "content": "Delete it again."}
    return [
        _span(
            "span-1",
            "2026-01-01T00:00:01",
            [system, user1, assistant1, tool1, user2],
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [_tool_call("c4", "delete_file", {"path": "notes.txt"})],
                }
            ],
            tools=_TOOLS,
        )
    ]


@pytest.fixture()
def synthetic_dataset(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = tmp_path / "agent-llm-traces"
    data_dir.mkdir()
    rows = [
        {
            "benchmark": "weather-bench",
            "session_id": "sess-1",
            "spans": json.dumps(_session_one_spans()),
        },
        {
            "benchmark": "files-bench",
            "session_id": "sess-2",
            "spans": json.dumps(_session_two_spans()),
        },
    ]
    table = pa.table({key: [row[key] for row in rows] for key in rows[0]})
    pq.write_table(table, data_dir / "shard.parquet")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"train_session_ids": ["sess-1", "sess-2"], "eval_session_ids": []}),
        encoding="utf-8",
    )
    return str(data_dir), str(manifest)


# ---------------------------------------------------------------------------
# Source tests.
# ---------------------------------------------------------------------------


def _joint_source(dataset, **kwargs):
    path, manifest = dataset
    kwargs.setdefault("split", "train")
    kwargs.setdefault("split_manifest_file", manifest)
    kwargs.setdefault("canonical_format_prob", 1.0)
    kwargs.setdefault("minified_json_prob", 0.0)
    kwargs.setdefault("shuffle_tools", False)
    return AgentLLMTracesJointSource(path, **kwargs)


def test_source_parses_synthetic_spans(synthetic_dataset):
    source = _joint_source(synthetic_dataset)
    examples = list(source)
    assert [example.qid for example in examples] == ["sess-1:0", "sess-1:1", "sess-2:0"]

    first = examples[0]
    assert first.session_id == "sess-1"
    assert first.subset == "weather-bench"
    assert first.system_prompt == "You are a weather agent."
    assert "get_weather" not in first.system_prompt  # bare system prompt, no schemas

    assert len(first.tool_documents) == 2
    assert first.tool_documents[0].startswith("<TOOL>")
    assert "<NAME> get_weather" in first.tool_documents[0]
    assert "<NAME> search_files" in first.tool_documents[1]

    assert len(first.history_documents) == 2
    assert all(doc.startswith("Previous turn") for doc in first.history_documents)
    assert "List the files in /tmp please." in first.history_documents[0]
    assert "found a.txt and b.txt" in first.history_documents[1]

    assert first.current_messages == [{"role": "user", "content": "Now get the weather in Paris."}]
    assert first.answer == (
        'Action:\n<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>'
    )

    second = examples[1]
    assert len(second.history_documents) == 4  # all turns before the final user message
    assert second.current_messages == [{"role": "user", "content": "Thanks, summarize the results."}]
    assert second.answer == "Summary: found a.txt and b.txt; Paris is rainy."
    assert second.tool_documents == first.tool_documents  # same pool + deterministic rendering

    third = examples[2]
    assert third.subset == "files-bench"
    assert third.system_prompt == "You are a files agent."
    assert third.answer == (
        'Action:\n<tool_call>\n{"name":"delete_file","arguments":{"path":"notes.txt"}}\n</tool_call>'
    )


def test_source_max_samples_per_session(synthetic_dataset):
    source = _joint_source(synthetic_dataset, max_samples_per_session=1)
    examples = list(source)
    assert len(examples) == 2
    assert {example.session_id for example in examples} == {"sess-1", "sess-2"}


# ---------------------------------------------------------------------------
# Preprocess tests (direct JointExample construction).
# ---------------------------------------------------------------------------

_TOOL_DOC_NAMES = ["get_weather", "search_files", "delete_file"]


def _tool_docs(n=2):
    docs = []
    for index in range(n):
        name = _TOOL_DOC_NAMES[index]
        docs.append(
            f"<TOOL>\n<NAMESPACE> ns{index}\n<NAME> {name}\n"
            f"<DESCRIPTION> Synthetic tool number {index} used for leakage probes.\n"
            f"<PARAMETERS>\n<PARAM name=\"arg{index}\" type=\"string\" required=\"true\">\n"
            f"</PARAMETERS>\n</TOOL>"
        )
    return docs


_HISTORY_MARKERS = ["alpha", "bravo", "charlie", "delta", "echo"]


def _history_docs(n=3):
    return [
        f"Previous turn\n[User query]\n{_HISTORY_MARKERS[i]} question number {i}\n"
        f"[Assistant output]\n{_HISTORY_MARKERS[i]} answer number {i}"
        for i in range(n)
    ]


def _example(**overrides):
    base = dict(
        qid="t:0",
        session_id="t",
        tool_documents=_tool_docs(2),
        history_documents=_history_docs(3),
        current_messages=[{"role": "user", "content": "What is the weather in Paris right now?"}],
        answer='Action:\n<tool_call>\n{"name":"get_weather","arguments":{"city":"Paris"}}\n</tool_call>',
        system_prompt="You are a careful data agent.",
        subset="t",
    )
    base.update(overrides)
    return JointExample(**base)


def _config(**overrides):
    base = dict(
        max_length=128,
        max_doc_length=64,
        min_doc_num=2,
        max_doc_num=6,
        max_system_length=96,
        history_selection="tail",
    )
    base.update(overrides)
    return base


def _features(tokenizer, example, reason_ok=True, **config):
    row, reason = JointDataset.preprocess_example(example, tokenizer=tokenizer, **_config(**config))
    if reason_ok:
        assert row is not None, reason
    return row, reason


def _context_text(tokenizer, features):
    real = [token_id for token_id in features["context_input_ids"] if token_id >= 0]
    return " ".join(tokenizer.decode(real, skip_special_tokens=True).split())


@pytest.fixture(scope="module")
def tokenizer():
    return _WhitespaceSelfTestTokenizer()


def test_output_keys_and_shapes(tokenizer):
    example = _example()
    features, _ = _features(tokenizer, example)
    assert set(features) == {
        "system_input_ids",
        "context_input_ids",
        "input_ids",
        "labels",
        "attention_mask",
        "dynamic",
    }
    assert len(features["system_input_ids"]) == 96
    assert len(features["context_input_ids"]) == 6 * 64
    assert len(features["input_ids"]) == 128
    assert len(features["labels"]) == 128
    assert len(features["attention_mask"]) == 128
    assert features["dynamic"] == 0

    dataset = JointDataset([example, _example(qid="t:1")], tokenizer=tokenizer, **_config())
    assert len(dataset) == 2
    assert set(dataset[0]) == set(features)


def test_assert_no_leakage_passes_on_clean_input(tokenizer):
    example = _example()
    features, _ = _features(tokenizer, example)
    assert_no_leakage(example, features, tokenizer)  # must not raise


def test_assert_no_leakage_fails_when_tools_in_system_prompt(tokenizer):
    example = _example()
    features, _ = _features(tokenizer, example)
    leaked_system_ids = _chat_template_ids(
        tokenizer,
        [{
            "role": "system",
            "content": example.system_prompt + "\n" + "\n".join(example.tool_documents),
        }],
        keep_bos=True,
        max_length=96,
    )
    tampered = dict(features)
    tampered["system_input_ids"] = _pad(leaked_system_ids, 96, -100)
    with pytest.raises(AssertionError, match="system_input_ids"):
        assert_no_leakage(example, tampered, tokenizer)


def test_assert_no_leakage_fails_when_tools_in_prompt(tokenizer):
    example = _example(
        current_messages=[{"role": "user", "content": "Explain this tool please.\n" + _tool_docs(1)[0]}]
    )
    features, _ = _features(tokenizer, example)
    with pytest.raises(AssertionError, match="input_ids"):
        assert_no_leakage(example, features, tokenizer)


def test_history_order_in_context_grid(tokenizer):
    example = _example(history_documents=_history_docs(4))
    features, _ = _features(tokenizer, example, doc_mode="history_only", max_doc_num=4)
    context_text = _context_text(tokenizer, features)
    positions = [context_text.find(f"{marker} question") for marker in _HISTORY_MARKERS[:4]]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
    assert_no_leakage(example, features, tokenizer)


def test_doc_mode_subsets(tokenizer):
    example = _example()

    joint, _ = _features(tokenizer, example, doc_mode="joint")
    joint_text = _context_text(tokenizer, joint)
    assert "Tool definition:" in joint_text
    assert "Previous turn" in joint_text

    tool_only, _ = _features(tokenizer, example, doc_mode="tool_only")
    tool_only_text = _context_text(tokenizer, tool_only)
    assert "Tool definition:" in tool_only_text
    assert "Previous turn" not in tool_only_text

    history_only, _ = _features(tokenizer, example, doc_mode="history_only")
    history_only_text = _context_text(tokenizer, history_only)
    assert "Tool definition:" not in history_only_text
    assert "<TOOL>" not in history_only_text
    assert "Previous turn" in history_only_text

    with pytest.raises(ValueError, match="doc_mode"):
        JointDataset.preprocess_example(example, tokenizer=tokenizer, **_config(doc_mode="bogus"))


def test_label_masking_boundaries(tokenizer):
    example = _example()
    features, _ = _features(tokenizer, example)
    expected = tokenizer.encode(example.answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    real_length = sum(features["attention_mask"])
    prompt_length = real_length - len(expected)
    labels = features["labels"]
    assert all(value == -100 for value in labels[:prompt_length])
    assert labels[prompt_length:real_length] == expected
    assert features["input_ids"][prompt_length:real_length] == expected
    assert labels[real_length - 1] == tokenizer.eos_token_id
    assert all(value == -100 for value in labels[real_length:])
    assert_no_leakage(example, features, tokenizer)


def test_tail_biased_history_truncation(tokenizer):
    example = _example(history_documents=_history_docs(5))
    features, _ = _features(
        tokenizer,
        example,
        doc_mode="history_only",
        max_doc_num=3,
        min_doc_num=1,
    )
    context_text = _context_text(tokenizer, features)
    # Tail policy keeps the first doc plus the most recent ones: H1, H4, H5.
    kept = [_HISTORY_MARKERS[i] for i in (0, 3, 4)]
    dropped = [_HISTORY_MARKERS[i] for i in (1, 2)]
    for marker in kept:
        assert f"{marker} question" in context_text
    for marker in dropped:
        assert f"{marker} question" not in context_text
    positions = [context_text.find(f"{marker} question") for marker in kept]
    assert positions == sorted(positions)


def test_tool_chunk_budget_allocation(tokenizer):
    example = _example(tool_documents=_tool_docs(3), history_documents=_history_docs(5))
    features, _ = _features(
        tokenizer,
        example,
        doc_mode="joint",
        max_doc_num=6,
        max_tool_chunks=2,
        min_doc_num=1,
    )
    context_text = _context_text(tokenizer, features)
    # Tools are capped at max_tool_chunks=2; the third tool doc is dropped.
    assert context_text.count("Tool definition:") == 2
    assert "delete_file" not in context_text
    # History gets the remaining 4 slots, tail-biased: H1, H3, H4, H5.
    assert context_text.count("Previous turn") == 4
    kept = [_HISTORY_MARKERS[i] for i in (0, 2, 3, 4)]
    assert f"{_HISTORY_MARKERS[1]} question" not in context_text
    positions = [context_text.find(f"{marker} question") for marker in kept]
    assert all(position >= 0 for position in positions)
    assert positions == sorted(positions)
    # Tool chunks precede history chunks in the flat grid.
    assert context_text.find("Tool definition:") < context_text.find("Previous turn")


def test_tool_definition_token_cap_drops_example(tokenizer):
    example = _example()
    row, reason = _features(
        tokenizer,
        example,
        reason_ok=False,
        max_tool_definition_tokens=10,
    )
    assert row is None
    assert reason.startswith("tool_definition_tokens>")


def test_min_doc_num_drops_example(tokenizer):
    example = _example(tool_documents=[], history_documents=_history_docs(1))
    row, reason = _features(tokenizer, example, reason_ok=False, min_doc_num=2)
    assert row is None
    assert reason == "doc_num<2"


# ---------------------------------------------------------------------------
# Tool-pool selection (per-example bounded pool).
# ---------------------------------------------------------------------------


def _named_tools(names):
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"desc {name}",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def test_first_tool_call_name_schemas():
    # OpenAI tool_calls shape.
    out = json.dumps([
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c1", "get_weather", {"city": "Paris"})]}
    ])
    assert tdj._first_tool_call_name(out) == "get_weather"
    # gen_ai parts shape.
    out = json.dumps([
        {"role": "assistant",
         "parts": [{"type": "tool_call", "name": "search_files", "arguments": {"path": "/tmp"}}]}
    ])
    assert tdj._first_tool_call_name(out) == "search_files"
    # No tool call anywhere.
    assert tdj._first_tool_call_name(json.dumps([{"role": "assistant", "content": "text"}])) is None


def test_select_tools_target_included_capped_deterministic():
    tools = _named_tools([f"ns{i % 10}.tool_{i}" for i in range(100)])
    selected = tdj._select_tools(tools, "ns0.tool_0", random.Random(0), max_tools_per_sample=32)
    assert len(selected) == 32
    assert any(tdj._tool_name(tool) == "ns0.tool_0" for tool in selected)
    again = tdj._select_tools(tools, "ns0.tool_0", random.Random(0), max_tools_per_sample=32)
    assert [tdj._tool_name(t) for t in selected] == [tdj._tool_name(t) for t in again]
    # Pool under the cap: everything kept in declared order.
    small = tools[:10]
    assert [tdj._tool_name(t) for t in tdj._select_tools(small, "ns0.tool_0", random.Random(1), max_tools_per_sample=32)] == [
        tdj._tool_name(t) for t in small
    ]


def test_select_tools_unknown_target_keeps_declared_order():
    tools = _named_tools([f"ns{i % 10}.tool_{i}" for i in range(100)])
    fallback = tdj._select_tools(tools, "missing.tool", random.Random(2), max_tools_per_sample=32)
    assert [tdj._tool_name(t) for t in fallback] == [tdj._tool_name(t) for t in tools[:32]]


def test_session_examples_bounded_pool_and_no_tool_skip(synthetic_dataset):
    source = _joint_source(synthetic_dataset)
    big_tools = _named_tools([f"ns{i % 5}.tool_{i}" for i in range(40)])
    spans = [
        _span(
            "span-1",
            "2026-01-01T00:00:01",
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": None,
                 "tool_calls": [_tool_call("c1", "ns0.tool_0", {})]},
                {"role": "tool", "content": "obs"},
                {"role": "user", "content": "u2"},
            ],
            [{"role": "assistant", "content": None,
              "tool_calls": [_tool_call("c2", "ns3.tool_33", {})]}],
            tools=big_tools,
        )
    ]
    examples = source._session_examples("big", spans, "bench")
    assert len(examples) == 1
    assert len(examples[0].tool_documents) == 32
    # Target sits beyond the leading 32 declared tools: target-inclusive
    # selection must pull it in (declared-order fallback would drop it).
    assert any("<NAME> ns3.tool_33" in doc for doc in examples[0].tool_documents)

    # Session without any tool definitions yields no joint examples.
    no_tool_spans = [
        _span(
            "span-1",
            "2026-01-01T00:00:01",
            [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
            ],
            [{"role": "assistant", "content": "a2"}],
        )
    ]
    assert source._session_examples("none", no_tool_spans, "bench") == []
