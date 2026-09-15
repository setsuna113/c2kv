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
g. tail-biased history truncation and tool/history budget allocation;
h. H200 arm: position-stratified + action-balanced per-session sampling,
   ``action_type`` tagging, ``<think>`` stripping, tool-call target integrity.

Run from the repo root (local venv has torch/transformers/datasets/pytest):
  pytest python/train/test_train_data_joint.py -v
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
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
    _history_chunk_budget,
    _parameter_signature,
    _render_tool_documents,
    _truncation_stress_example,
    assert_no_leakage,
    assert_target_tool_in_grid,
    build_tool_chunks,
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
    # max_tool_chunks=0: give history the whole grid (per-side budgets no
    # longer recycle unused tool slots, so reserve none for tools).
    features, _ = _features(
        tokenizer, example, doc_mode="history_only", max_doc_num=4, max_tool_chunks=0
    )
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
        max_tool_chunks=0,
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


# ---------------------------------------------------------------------------
# Per-side caps + target-preserving truncation (the cap fix).
# ---------------------------------------------------------------------------


def test_parameter_signature_required_null():
    sig = _parameter_signature({
        "function": {
            "name": "t",
            "parameters": {"properties": {"a": {"type": "string"}}, "required": None},
        }
    })
    assert sig and sig[0]["required"] is False


def test_render_tool_documents_reports_target_index_across_variants():
    tools = [
        {"type": "function", "function": {"name": f"tool_{i}", "parameters": {"properties": {}}}}
        for i in range(8)
    ]
    seen_variants = set()
    for seed in range(60):
        docs, target_index = _render_tool_documents(tools, random.Random(seed), target_tool="tool_5")
        assert target_index is not None
        assert "tool_5" in docs[target_index]
        seen_variants.add("canonical" if docs[target_index].startswith("<TOOL>") else "json")
    assert seen_variants == {"canonical", "json"}
    _, missing = _render_tool_documents(tools, random.Random(0), target_tool="absent_tool")
    assert missing is None


def test_build_tool_chunks_keeps_tail_target(tokenizer):
    stress = _truncation_stress_example(num_tools=12, target_index=10)
    chunks, skip, meta = build_tool_chunks(
        tokenizer, stress, "joint",
        max_doc_length=256, max_doc_num=8, max_tool_chunks=None,
        max_tool_definition_tokens=32000, per_side_caps=True,
    )
    assert skip is None
    assert meta["tool_cap"] == 5 and len(chunks) == 5
    assert meta["target_known"] and meta["target_in_grid"] is True
    decoded = [tokenizer.decode(chunk) for chunk in chunks]
    assert any("tool_10" in text for text in decoded)
    kept_ids = [int(text.split("tool_")[1].split()[0].strip(">")) for text in decoded if "tool_" in text]
    assert kept_ids == sorted(kept_ids)  # original relative order preserved


def test_build_tool_chunks_per_side_caps_align_tool_only_with_joint(tokenizer):
    stress = _truncation_stress_example(num_tools=12, target_index=10)
    _, _, meta_joint = build_tool_chunks(
        tokenizer, stress, "joint",
        max_doc_length=256, max_doc_num=8, max_tool_chunks=None,
        max_tool_definition_tokens=32000, per_side_caps=True,
    )
    _, _, meta_tool_only = build_tool_chunks(
        tokenizer, stress, "tool_only",
        max_doc_length=256, max_doc_num=8, max_tool_chunks=None,
        max_tool_definition_tokens=32000, per_side_caps=True,
    )
    assert meta_joint["tool_cap"] == meta_tool_only["tool_cap"] == 5


def test_build_tool_chunks_legacy_reproduces_prefix_behavior(tokenizer):
    stress = _truncation_stress_example(num_tools=12, target_index=10)
    _, _, meta_tool_only = build_tool_chunks(
        tokenizer, stress, "tool_only",
        max_doc_length=256, max_doc_num=8, max_tool_chunks=None,
        max_tool_definition_tokens=32000, per_side_caps=False,
    )
    assert meta_tool_only["tool_cap"] == 8  # legacy: all max_doc_num slots
    chunks, _, meta_joint = build_tool_chunks(
        tokenizer, stress, "joint",
        max_doc_length=256, max_doc_num=8, max_tool_chunks=None,
        max_tool_definition_tokens=32000, per_side_caps=False,
    )
    # Legacy head-truncation drops the tail target (the pre-fix bug, kept
    # reproducible for diffing old runs) and meta must report the drop.
    assert meta_joint["target_in_grid"] is False
    decoded = " ".join(tokenizer.decode(chunk) for chunk in chunks)
    assert "tool_10" not in decoded


def test_history_budget_constant_across_modes_under_per_side_caps():
    for mode in ("joint", "history_only"):
        assert _history_chunk_budget(mode, 24, 16, 3, per_side_caps=True) == 8
    # Legacy recycled spare tool slots / gave history_only the whole grid.
    assert _history_chunk_budget("joint", 24, 16, 3, per_side_caps=False) == 21
    assert _history_chunk_budget("history_only", 24, 16, 0, per_side_caps=False) == 24


def test_assert_target_tool_in_grid_detects_legacy_drop(tokenizer):
    stress = _truncation_stress_example(num_tools=12, target_index=10)
    kwargs = dict(
        tokenizer=tokenizer, max_length=512, max_doc_length=256, min_doc_num=2,
        max_doc_num=8, max_system_length=512, doc_mode="joint",
    )
    fixed, _ = JointDataset.preprocess_example(stress, per_side_caps=True, **kwargs)
    assert_target_tool_in_grid(stress, fixed, tokenizer)  # must pass
    legacy, _ = JointDataset.preprocess_example(stress, per_side_caps=False, **kwargs)
    with pytest.raises(AssertionError, match="target tool document"):
        assert_target_tool_in_grid(stress, legacy, tokenizer)


def test_joint_dataset_target_stats_and_invariant(tokenizer):
    stress = _truncation_stress_example(num_tools=12, target_index=10)
    kwargs = dict(
        max_length=512, max_doc_length=256, min_doc_num=2,
        max_doc_num=8, max_system_length=512, doc_mode="joint",
    )
    dataset = JointDataset([stress], tokenizer, per_side_caps=True, **kwargs)
    assert dataset.target_stats == {
        "target_known": 1, "target_in_grid": 1, "target_truncated_to_cap": 0,
    }
    legacy = JointDataset([stress], tokenizer, per_side_caps=False, **kwargs)
    assert legacy.target_stats == {
        "target_known": 1, "target_in_grid": 0, "target_truncated_to_cap": 0,
    }


def test_oversized_target_doc_truncated_to_cap_not_fatal(tokenizer):
    # Target schema alone chunks into more than tool_cap pieces: retention
    # keeps the cap-filling prefix, flags it, and the constructor invariant
    # must NOT fire — this is a data condition, not a retention regression.
    big_target = JointExample(
        qid="s:big",
        session_id="s",
        tool_documents=[" ".join(f"w{i}" for i in range(200))],
        history_documents=[
            "Previous turn synthetic filler one with enough words to probe cleanly.",
            "Previous turn synthetic filler two with enough words to probe cleanly.",
        ],
        current_messages=[{"role": "user", "content": "call the big tool now"}],
        answer='Action:\n<tool_call>\n{"name":"big","arguments":{}}\n</tool_call>',
        target_tool="big",
        target_tool_doc_index=0,
    )
    chunks, skip, meta = build_tool_chunks(
        tokenizer, big_target, "joint",
        max_doc_length=16, max_doc_num=8, max_tool_chunks=None,
        max_tool_definition_tokens=32000, per_side_caps=True,
    )
    assert skip is None
    assert len(chunks) == meta["tool_cap"] == 5  # cap-filling prefix of the target
    assert meta["target_in_grid"] is False
    assert meta["target_truncated_to_cap"] is True
    dataset = JointDataset(
        [big_target], tokenizer, max_length=512, max_doc_length=16, min_doc_num=1,
        max_doc_num=8, max_system_length=512, doc_mode="joint", per_side_caps=True,
    )  # must not raise
    assert dataset.target_stats == {
        "target_known": 1, "target_in_grid": 0, "target_truncated_to_cap": 1,
    }


def test_zero_tool_cap_is_config_not_retention_failure(tokenizer):
    stress = _truncation_stress_example(num_tools=4, target_index=3)
    dataset = JointDataset(
        [stress], tokenizer, max_length=512, max_doc_length=256, min_doc_num=1,
        max_doc_num=8, max_tool_chunks=0, max_system_length=512,
        doc_mode="joint", per_side_caps=True,
    )  # must not raise: tool side deliberately absent, nothing to retain
    assert dataset.target_stats == {
        "target_known": 0, "target_in_grid": 0, "target_truncated_to_cap": 0,
    }


def test_source_examples_carry_target_doc_index(synthetic_dataset):
    source = _joint_source(synthetic_dataset)
    indexed = 0
    for example in source.records:
        if example.target_tool_doc_index is None:
            # Legitimate: no tool call in the target, or the called name is
            # not among the session's definitions (e.g. delete_file).
            continue
        indexed += 1
        assert example.target_tool
        assert example.target_tool in example.tool_documents[example.target_tool_doc_index]
    assert indexed > 0  # the get_weather example must carry its index


# ---------------------------------------------------------------------------
# P0-1: v2 empty-tool reclaim (the QA family reserves no tool slots) and the
# QA history/gold retention audit counters.
# ---------------------------------------------------------------------------


def _qa_style_example(qid="qa:hotpotqa:x", num_docs=6, gold=None, subset="qa:hotpotqa"):
    """Tool-less QA-style example: documents carry the HotpotQA title prefix."""
    return JointExample(
        qid=qid,
        session_id=qid,
        tool_documents=[],
        history_documents=[
            f"Document {index + 1} (title: Title {index}) " + " ".join(f"w{index}-{j}" for j in range(20))
            for index in range(num_docs)
        ],
        current_messages=[{"role": "user", "content": "Which title answers the question?"}],
        answer="Title 3",
        subset=subset,
        gold_history_doc_indices=gold,
    )


def test_empty_tool_example_reclaims_full_history_grid(tokenizer):
    # Budget helpers: a tool-less example reserves no tool side (v2); the
    # tool-bearing default keeps the v1 constant; legacy is untouched.
    assert _history_chunk_budget("joint", 24, 16, 0, per_side_caps=True, has_tool_documents=False) == 24
    assert _history_chunk_budget("history_only", 24, 16, 0, per_side_caps=True, has_tool_documents=False) == 24
    assert _history_chunk_budget("joint", 24, 16, 0, per_side_caps=True) == 8  # v1 constant, default
    assert _history_chunk_budget("joint", 24, 16, 0, per_side_caps=False) == 24  # legacy recycled already
    assert _history_chunk_budget("history_only", 24, 16, 0, per_side_caps=False) == 24

    example = _qa_style_example(num_docs=10)
    chunks, skip, meta = build_tool_chunks(
        tokenizer, example, "joint",
        max_doc_length=256, max_doc_num=8, max_tool_chunks=None,
        max_tool_definition_tokens=32000, per_side_caps=True,
    )
    assert skip is None and chunks == [] and meta["tool_cap"] == 0
    # tool_only over a tool-less example keeps producing 0 chunks (the
    # alternate-arm skip, recorded per family — see the skip test below).
    chunks, skip, meta = build_tool_chunks(
        tokenizer, example, "tool_only",
        max_doc_length=256, max_doc_num=8, max_tool_chunks=None,
        max_tool_definition_tokens=32000, per_side_caps=True,
    )
    assert skip is None and chunks == [] and meta["tool_cap"] == 0

    # End to end: 10 documents all fit the reclaimed 12-slot grid (under v1
    # the history side would have been capped at 12 - 8 = 4 slots).
    meta_out: dict = {}
    row, reason = JointDataset.preprocess_example(
        example, tokenizer=tokenizer, max_length=512, max_doc_length=256, min_doc_num=2,
        max_doc_num=12, max_system_length=512, doc_mode="joint", per_side_caps=True,
        meta_out=meta_out,
    )
    assert row is not None, reason
    assert meta_out["tool_cap"] == 0
    assert meta_out["history_docs_total"] == 10
    assert meta_out["history_kept_source_indices"] == list(range(10))
    assert_no_leakage(example, row, tokenizer)  # every document present, in order


def test_tool_bearing_example_unaffected_by_empty_tool_reclaim(tokenizer):
    # The v2 reclaim must not touch examples WITH tools: caps, budgets and the
    # emitted row stay exactly at the v1 values.
    stress = _truncation_stress_example(num_tools=12, target_index=10)
    assert _history_chunk_budget("joint", 24, 16, 3, per_side_caps=True, has_tool_documents=True) == 8
    chunks, skip, meta = build_tool_chunks(
        tokenizer, stress, "joint",
        max_doc_length=256, max_doc_num=8, max_tool_chunks=None,
        max_tool_definition_tokens=32000, per_side_caps=True,
    )
    assert skip is None and meta["tool_cap"] == 5 and len(chunks) == 5
    meta_out: dict = {}
    row, reason = JointDataset.preprocess_example(
        stress, tokenizer=tokenizer, max_length=512, max_doc_length=256, min_doc_num=2,
        max_doc_num=8, max_system_length=512, doc_mode="joint", per_side_caps=True,
        meta_out=meta_out,
    )
    assert row is not None, reason
    # v1 split: 5 tool chunks + min(2 history docs, 8-5=3 slots) = 2 history docs.
    assert meta_out["num_tool_chunks"] == 5
    assert meta_out["num_history_docs"] == 2
    assert meta_out["history_kept_source_indices"] == [0, 1]
    assert_no_leakage(stress, row, tokenizer)
    assert_target_tool_in_grid(stress, row, tokenizer)


def test_qa_retention_counters_track_grid_survivors(tokenizer):
    example = _qa_style_example(qid="qa:hotpotqa:g", num_docs=10, gold=(3, 7))
    full = JointDataset(
        [example], tokenizer, max_length=512, max_doc_length=256, min_doc_num=2,
        max_doc_num=12, max_system_length=512, doc_mode="joint",
    )
    assert full.qa_retention_stats == {
        "qa_history_doc_retention": {"kept": 10, "total": 10},
        "qa_gold_doc_retention": {"kept": 2, "total": 2},
        "qa_history_truncated_examples_by_subset": {},
    }
    assert full.skipped_by_family_reason == {}

    # Tight grid (4 slots): tail selection keeps source docs {0, 7, 8, 9} —
    # gold doc 7 survives, gold doc 3 is cut and the counter must show it.
    tight = JointDataset(
        [example], tokenizer, max_length=512, max_doc_length=256, min_doc_num=2,
        max_doc_num=4, max_system_length=512, doc_mode="joint",
    )
    assert tight.qa_retention_stats["qa_history_doc_retention"] == {"kept": 4, "total": 10}
    assert tight.qa_retention_stats["qa_gold_doc_retention"] == {"kept": 1, "total": 2}
    assert tight.qa_retention_stats["qa_history_truncated_examples_by_subset"] == {"qa:hotpotqa": 1}


def test_tool_only_pass_skips_tool_less_qa_counted_by_family(tokenizer):
    # P1-7 visibility: the alternate arm's tool_only pass renders no documents
    # for QA examples and skips them; the skip must be attributable per family.
    example = _qa_style_example(num_docs=4)
    dataset = JointDataset(
        [example], tokenizer, max_length=512, max_doc_length=256, min_doc_num=2,
        max_doc_num=8, max_system_length=512, doc_mode="tool_only",
    )
    assert len(dataset) == 0
    assert dataset.skipped_by_reason == {"doc_num<2": 1}
    assert dataset.skipped_by_family_reason == {"qa:doc_num<2": 1}
    # The joint pass over the same example keeps every document.
    joint = JointDataset(
        [example], tokenizer, max_length=512, max_doc_length=256, min_doc_num=2,
        max_doc_num=8, max_system_length=512, doc_mode="joint",
    )
    assert len(joint) == 1
    assert joint.qa_retention_stats["qa_history_doc_retention"] == {"kept": 4, "total": 4}


# ---------------------------------------------------------------------------
# H200 arm: position-stratified + action-balanced per-session sampling,
# action_type tagging, <think> stripping, tool-call target integrity.
# ---------------------------------------------------------------------------


def _decision_session_spans(num_spans, tool_call_mask=None):
    """One session with num_spans decision points (span ``i`` targets action ``i``).

    The conversation opens with one full exchange so even the first span has
    non-empty history.  ``tool_call_mask[i]`` falsy -> target ``i`` is a
    plain-text answer (no tool call); default is all tool calls.
    """
    conversation = [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "initial request"},
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c-init", "search_files", {"path": "/tmp"})]},
        {"role": "tool", "content": "initial observation"},
    ]
    spans = []
    for index in range(num_spans):
        user = {"role": "user", "content": f"request number {index}"}
        if tool_call_mask is None or tool_call_mask[index]:
            output = [{
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call(f"c{index}", "get_weather", {"city": f"city{index}"})],
            }]
        else:
            output = [{"role": "assistant", "content": f"plain answer number {index}"}]
        spans.append(_span(
            f"span-{index}",
            f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}",
            conversation + [user],
            output,
            tools=_TOOLS if index == 0 else None,
        ))
        conversation = conversation + [user, *output, {"role": "tool", "content": f"observation {index}"}]
    return spans


def _write_traces_dataset(tmp_path, sessions):
    """Write ``[(session_id, spans)]`` rows as one parquet shard + a train-all manifest."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    data_dir = tmp_path / "agent-llm-traces"
    data_dir.mkdir()
    rows = [
        {"benchmark": "strat-bench", "session_id": session_id, "spans": json.dumps(spans)}
        for session_id, spans in sessions
    ]
    table = pa.table({key: [row[key] for row in rows] for key in rows[0]})
    pq.write_table(table, data_dir / "shard.parquet")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({
            "train_session_ids": [session_id for session_id, _ in sessions],
            "eval_session_ids": [],
        }),
        encoding="utf-8",
    )
    return str(data_dir), str(manifest)


def _span_index(qid):
    return int(qid.rsplit(":", 1)[1])


def test_stratified_pick_position_quotas_and_determinism(tmp_path):
    dataset = _write_traces_dataset(tmp_path, [("sess-strat", _decision_session_spans(7))])

    def build():
        source = _joint_source(dataset, max_samples_per_session=4, require_tool_call=False)
        return [example.qid for example in source]

    qids = build()
    assert qids == build()  # identical across two constructions with the same seed
    indices = [_span_index(qid) for qid in qids]
    assert indices == sorted(indices)  # picks returned in chronological order
    # 7 candidates -> thirds [0,1] / [2,3] / [4,5,6] with quota 1/1/2.
    assert len(indices) == 4
    assert sum(index in (0, 1) for index in indices) == 1
    assert sum(index in (2, 3) for index in indices) == 1
    assert sum(index in (4, 5, 6) for index in indices) == 2


def test_stratified_pick_backfills_short_bucket(tmp_path):
    # 6 candidates, k=5: buckets [0,1]/[2,3]/[4,5], quotas 1/1/3 — the late
    # bucket is one short and is backfilled from the middle bucket (late ->
    # middle -> early).
    dataset = _write_traces_dataset(tmp_path, [("sess-backfill", _decision_session_spans(6))])
    source = _joint_source(dataset, max_samples_per_session=5, require_tool_call=False)
    indices = sorted(_span_index(example.qid) for example in source)
    assert len(indices) == 5
    assert indices[1:] == [2, 3, 4, 5]  # middle + late buckets fully taken
    assert indices[0] in (0, 1)


def test_stratified_pick_action_balance_hits_target(tmp_path):
    # Alternating tool/plain targets: every position bucket holds both pools,
    # so the per-session tool_call target round(4 * 0.75) = 3 is met exactly.
    mask = [True, False] * 4  # 8 candidates, half tool_call
    dataset = _write_traces_dataset(tmp_path, [("sess-mix", _decision_session_spans(8, mask))])
    source = _joint_source(
        dataset, max_samples_per_session=4, require_tool_call=False, action_tool_call_frac=0.75,
    )
    examples = list(source)
    assert len(examples) == 4
    assert Counter(example.action_type for example in examples) == {"tool_call": 3, "other": 1}
    # Position quotas still hold: buckets [0,1]/[2,3]/[4..7] -> 1/1/2.
    indices = sorted(_span_index(example.qid) for example in examples)
    assert sum(index in (0, 1) for index in indices) == 1
    assert sum(index in (2, 3) for index in indices) == 1
    assert sum(index in (4, 5, 6, 7) for index in indices) == 2


def test_stratified_pick_action_balance_pool_short_fallback(tmp_path):
    # No "other" candidates at all: the tool pool fills every bucket quota.
    dataset = _write_traces_dataset(tmp_path, [("sess-alltool", _decision_session_spans(7))])
    source = _joint_source(dataset, max_samples_per_session=4, require_tool_call=False)
    examples = list(source)
    assert len(examples) == 4
    assert all(example.action_type == "tool_call" for example in examples)


def test_stratified_pick_text_first_then_tools_does_not_crash(tmp_path):
    # Regression (2026-09-05 audit): "text turns first, tool calls later" made
    # the middle bucket backfill from the other pool, driving other_target
    # negative, and the late bucket then called rng.sample(pool, -1) ->
    # ValueError at dataset load with the launcher default
    # require_tool_call=False.  Every shape must load and return min(k, n).
    shapes = [
        [False, False, True, True, True],
        [False, False, False, True, True, True],
        [False, True, False, True, True, True, True],
        [False, False, True, True, True, True, True, True, True],
    ]
    for mask in shapes:
        name = "sess-neg-" + "".join("t" if m else "o" for m in mask)
        (tmp_path / name).mkdir()
        dataset = _write_traces_dataset(tmp_path / name, [(name, _decision_session_spans(len(mask), mask))])
        source = _joint_source(
            dataset, max_samples_per_session=4, require_tool_call=False, action_tool_call_frac=0.75,
        )
        examples = list(source)
        assert len(examples) == 4, (mask, [e.qid for e in examples])
        indices = [_span_index(example.qid) for example in examples]
        assert indices == sorted(indices)


def test_require_tool_call_keeps_legacy_uniform_pick(tmp_path):
    # Regression: require_tool_call=True must stay bit-identical to the
    # pre-change behavior — tool-call-only candidates, uniform random pick
    # with the same seeded rng (split_seed + 0 for the train split).
    spans = _decision_session_spans(7)
    dataset = _write_traces_dataset(tmp_path, [("sess-legacy", spans)])
    source = _joint_source(dataset, max_samples_per_session=4, require_tool_call=True)
    candidates = source._session_examples("sess-legacy", spans, "strat-bench")
    assert len(candidates) == 7  # every target carries a tool call
    expected = [example.qid for example in random.Random(42).sample(candidates, 4)]
    assert [example.qid for example in source] == expected


def test_session_examples_tag_action_type(tmp_path):
    mask = [True, False]
    dataset = _write_traces_dataset(tmp_path, [("sess-tags", _decision_session_spans(2, mask))])
    source = _joint_source(dataset)
    examples = list(source)
    assert len(examples) == 2  # under the per-session cap: both kept
    assert [example.action_type for example in examples] == ["tool_call", "other"]


def test_strip_think_blocks_helper():
    assert tdj._strip_think_blocks("<think>secret\nreasoning</think>\nFinal answer.") == "Final answer."
    assert tdj._strip_think_blocks("Answer. <think>tail cut by the char cap") == "Answer."
    assert tdj._strip_think_blocks("No think here.") == "No think here."
    # The tool-call target surface is never touched.
    action = 'Action:\n<tool_call>\n{"name":"t","arguments":{}}\n</tool_call>'
    assert tdj._strip_think_blocks(action) == action


def _think_session_spans(output_content):
    conversation = [
        {"role": "system", "content": "You are a test agent."},
        {"role": "user", "content": "initial request"},
        {"role": "assistant", "content": None,
         "tool_calls": [_tool_call("c-init", "search_files", {"path": "/tmp"})]},
        {"role": "tool", "content": "initial observation"},
        {"role": "user", "content": "follow-up request"},
    ]
    return [
        _span(
            "span-0",
            "2026-01-01T00:00:00",
            conversation,
            [{"role": "assistant", "content": output_content}],
            tools=_TOOLS,
        )
    ]


def test_session_examples_strip_inline_think_from_answer(tmp_path):
    spans = _think_session_spans("<think>hidden reasoning</think>Visible answer.")
    dataset = _write_traces_dataset(tmp_path, [("sess-think", spans)])
    examples = list(_joint_source(dataset))
    assert len(examples) == 1
    assert examples[0].answer == "Visible answer."
    assert examples[0].action_type == "other"


def test_session_examples_drop_think_only_answer(tmp_path):
    spans = _think_session_spans("<think>only reasoning, no visible answer</think>")
    dataset = _write_traces_dataset(tmp_path, [("sess-thinkonly", spans)])
    assert list(_joint_source(dataset)) == []


def test_tool_call_answer_over_budget_dropped_not_truncated(tokenizer):
    # Long current turn: after maximal prompt truncation the answer budget is
    # a single token, so the tool-call target can never fit — drop, don't
    # train on a partial tool call.
    example = _example(
        current_messages=[{"role": "user", "content": " ".join(f"filler{i}" for i in range(60))}]
    )
    row, reason = _features(tokenizer, example, reason_ok=False, max_length=24)
    assert row is None
    assert reason == "tool_call_target_truncated"
    dataset = JointDataset([example], tokenizer=tokenizer, **_config(max_length=24))
    assert len(dataset) == 0
    assert dataset.skipped_by_reason == {"tool_call_target_truncated": 1}
    assert dataset.skipped_by_family_reason == {"traces:tool_call_target_truncated": 1}


def test_tool_call_answer_within_budget_kept_intact(tokenizer):
    example = _example()
    features, reason = _features(tokenizer, example)
    assert reason == "ok"
    expected = tokenizer.encode(example.answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    real_length = sum(features["attention_mask"])
    assert features["input_ids"][real_length - len(expected):real_length] == expected
    assert features["labels"][real_length - len(expected):real_length] == expected


def test_plain_answer_over_budget_still_truncated(tokenizer):
    answer = " ".join(f"word{i}" for i in range(200))  # no tool-call markers
    example = _example(answer=answer)
    row, reason = _features(tokenizer, example)
    assert reason == "ok"
    supervised = [value for value in row["labels"] if value != -100]
    full = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    assert len(supervised) < len(full)  # hard-truncated to the budget
    assert supervised == full[: len(supervised)]


# ---------------------------------------------------------------------------
# Regime-first knobs: tools_in_system (raw tool schemas in the system prefix)
# and the hybrid raw history tail.  Both default OFF and must leave every
# existing feature byte-identical when unset.
# ---------------------------------------------------------------------------


def _selected_tools(n=1):
    return [
        {
            "type": "function",
            "function": {
                "name": f"zzmarker_tool_{index}",
                "description": f"zzmarkerdesc{index} synthetic schema for the system prefix probe",
                "parameters": {
                    "type": "object",
                    "properties": {f"zzarg{index}": {"type": "string"}},
                    "required": [f"zzarg{index}"],
                },
            },
        }
        for index in range(n)
    ]


def _system_text(tokenizer, features):
    real = [token_id for token_id in features["system_input_ids"] if token_id >= 0]
    return " ".join(tokenizer.decode(real, skip_special_tokens=True).split())


def _prompt_text(tokenizer, features):
    real = [
        token_id
        for token_id, mask in zip(features["input_ids"], features["attention_mask"])
        if mask
    ]
    return " ".join(tokenizer.decode(real, skip_special_tokens=True).split())


_HYBRID_CONFIG = dict(
    doc_mode="history_only",
    max_doc_num=5,
    max_tool_chunks=0,
    min_doc_num=1,
    max_length=512,
    max_system_length=256,
)


def test_tools_in_system_puts_tools_only_in_the_system_prefix(tokenizer):
    example = _example(selected_tools=_selected_tools(2))
    features, reason = _features(
        tokenizer,
        example,
        doc_mode="history_only",
        tools_in_system=True,
        max_system_length=256,
    )
    assert reason == "ok"
    system_text = _system_text(tokenizer, features)
    assert "<TOOLS>" in system_text
    assert "zzmarker_tool_0" in system_text and "zzmarker_tool_1" in system_text
    # ... and nowhere else: not in the gist grid, not in the ordinary prompt.
    context_text = _context_text(tokenizer, features)
    prompt_text = _prompt_text(tokenizer, features)
    for probe in ("<TOOLS>", "zzmarker_tool_0", "zzmarker_tool_1", "zzmarkerdesc0"):
        assert probe not in context_text
        assert probe not in prompt_text


def test_tools_in_system_off_keeps_the_bare_system_prefix(tokenizer):
    example = _example(selected_tools=_selected_tools(2))
    features, reason = _features(tokenizer, example, doc_mode="history_only")
    assert reason == "ok"
    system_text = _system_text(tokenizer, features)
    assert "<TOOLS>" not in system_text and "zzmarker_tool_0" not in system_text


def test_tools_in_system_gives_history_the_full_grid(tokenizer):
    example = _example(history_documents=_history_docs(5), selected_tools=_selected_tools(1))
    # max_tool_chunks stays at its default (2/3 of max_doc_num), which without
    # tools_in_system reserves tool slots the history side can never use.
    baseline_meta = {}
    _features(
        tokenizer,
        example,
        meta_out=baseline_meta,
        doc_mode="history_only",
        max_doc_num=5,
        max_system_length=256,
    )
    assert baseline_meta["num_history_docs"] == 2  # 5 - min(default 3, 5)

    meta = {}
    features, reason = _features(
        tokenizer,
        example,
        meta_out=meta,
        doc_mode="history_only",
        tools_in_system=True,
        max_doc_num=5,
        max_system_length=256,
    )
    assert reason == "ok"
    assert meta["num_history_docs"] == 5
    context_text = _context_text(tokenizer, features)
    for marker in _HISTORY_MARKERS[:5]:
        assert f"{marker} question" in context_text


def test_tools_in_system_skips_over_long_system_prefix(tokenizer):
    example = _example(selected_tools=_selected_tools(8))
    row, reason = _features(
        tokenizer,
        example,
        reason_ok=False,
        doc_mode="history_only",
        tools_in_system=True,
        max_system_length=32,
    )
    assert row is None
    assert reason == "system_overflow"
    dataset = JointDataset(
        [example],
        tokenizer=tokenizer,
        **_config(doc_mode="history_only", tools_in_system=True, max_system_length=32),
    )
    assert len(dataset) == 0
    assert dataset.skipped_by_reason == {"system_overflow": 1}
    assert dataset.skipped_by_family_reason == {"traces:system_overflow": 1}
    # Never truncated: with room the LAST tool and the closing template marker
    # both survive (HF right-truncation would have deleted them silently).
    features, ok_reason = _features(
        tokenizer,
        example,
        doc_mode="history_only",
        tools_in_system=True,
        max_system_length=4096,
    )
    assert ok_reason == "ok"
    system_text = _system_text(tokenizer, features)
    assert "zzmarker_tool_7" in system_text
    assert "</TOOLS>" in system_text


def test_tools_in_system_requires_history_only(tokenizer):
    example = _example(selected_tools=_selected_tools(1))
    for mode in ("joint", "tool_only"):
        with pytest.raises(ValueError, match="history_only"):
            JointDataset.preprocess_example(
                example, tokenizer=tokenizer, **_config(doc_mode=mode, tools_in_system=True)
            )
        with pytest.raises(ValueError, match="history_only"):
            JointDataset(
                [example], tokenizer=tokenizer, **_config(doc_mode=mode, tools_in_system=True)
            )


def test_new_knobs_at_defaults_are_byte_identical(tokenizer):
    for mode in ("joint", "tool_only", "history_only"):
        example = _example(history_documents=_history_docs(4), selected_tools=_selected_tools(2))
        baseline, reason_a = JointDataset.preprocess_example(
            example, tokenizer=tokenizer, **_config(doc_mode=mode)
        )
        explicit, reason_b = JointDataset.preprocess_example(
            example,
            tokenizer=tokenizer,
            **_config(doc_mode=mode, tools_in_system=False, hybrid_tail_k=0),
        )
        assert reason_a == reason_b
        assert baseline == explicit
        plain = JointDataset([example], tokenizer=tokenizer, **_config(doc_mode=mode))
        knobbed = JointDataset(
            [example],
            tokenizer=tokenizer,
            hybrid_tail_choices=None,
            **_config(doc_mode=mode),
        )
        assert plain.data == knobbed.data
        assert knobbed.hybrid_tail_k_counts == ({0: len(knobbed)} if len(knobbed) else {})


def test_hybrid_tail_moves_last_docs_out_of_the_grid(tokenizer):
    example = _example(history_documents=_history_docs(5))
    meta = {}
    features, reason = _features(
        tokenizer, example, meta_out=meta, hybrid_tail_k=2, **_HYBRID_CONFIG
    )
    assert reason == "ok"
    assert meta["hybrid_tail_k"] == 2
    assert meta["num_history_docs"] == 5
    assert meta["num_compressed_history_docs"] == 3

    context_text = _context_text(tokenizer, features)
    assert "charlie question" in context_text
    assert "delta question" not in context_text
    assert "echo question" not in context_text

    # The raw tail is chat-template rendered and PREPENDED to the prompt.
    expected = []
    for index in (3, 4):
        expected.extend(
            _chat_template_ids(
                tokenizer,
                [{"role": "user", "content": example.history_documents[index]}],
                max_length=_config(**_HYBRID_CONFIG)["max_doc_length"],
            )
        )
    assert features["input_ids"][: len(expected)] == expected
    assert features["labels"][: len(expected)] == [-100] * len(expected)


def test_hybrid_tail_capped_so_min_doc_num_stays_compressed(tokenizer):
    example = _example(history_documents=_history_docs(5))
    meta = {}
    config = dict(_HYBRID_CONFIG)
    config["min_doc_num"] = 2
    features, reason = _features(tokenizer, example, meta_out=meta, hybrid_tail_k=99, **config)
    assert reason == "ok"
    assert meta["hybrid_tail_k"] == 3  # 5 fitted docs - min_doc_num=2
    assert meta["num_compressed_history_docs"] == 2
    context_text = _context_text(tokenizer, features)
    assert "alpha question" in context_text and "bravo question" in context_text
    assert "charlie question" not in context_text


def test_hybrid_tail_k_zero_is_identical_to_no_tail(tokenizer):
    example = _example(history_documents=_history_docs(5))
    baseline, _ = _features(tokenizer, example, **_HYBRID_CONFIG)
    zero, _ = _features(tokenizer, example, hybrid_tail_k=0, **_HYBRID_CONFIG)
    assert baseline == zero


def test_hybrid_tail_choices_drawn_deterministically_from_qid(tokenizer):
    choices = [0, 1, 3]
    examples = [
        _example(qid=f"t:{index}", history_documents=_history_docs(5)) for index in range(12)
    ]
    kwargs = dict(tokenizer=tokenizer, **_config(**_HYBRID_CONFIG))
    first = JointDataset(examples, hybrid_tail_choices=choices, **kwargs)
    again = JointDataset(examples, hybrid_tail_choices=choices, **kwargs)
    reversed_order = JointDataset(list(reversed(examples)), hybrid_tail_choices=choices, **kwargs)
    assert first.data == again.data
    # Order-independent: the draw is seeded by the qid, not by position.
    assert first.hybrid_tail_k_counts == reversed_order.hybrid_tail_k_counts
    assert first.data == list(reversed(reversed_order.data))
    expected = Counter(
        random.Random(f"{example.qid}:hybrid_tail").choice(choices) for example in examples
    )
    assert first.hybrid_tail_k_counts == dict(sorted(expected.items()))
    assert len(first.hybrid_tail_k_counts) > 1  # the pool is actually sampled


def test_hybrid_tail_choices_reject_negative(tokenizer):
    with pytest.raises(ValueError, match="non-negative"):
        JointDataset(
            [_example()], tokenizer=tokenizer, hybrid_tail_choices=[0, -1], **_config()
        )
    with pytest.raises(ValueError, match="non-negative"):
        JointDataset.preprocess_example(
            _example(), tokenizer=tokenizer, hybrid_tail_k=-1, **_config()
        )


# ---------------------------------------------------------------------------
# Regime-first knobs, review round 2: the raw tail must SHORTEN under the
# sequence budget instead of dropping the row, drawn-vs-realized k must be
# visible, and tools_in_system on a tool-less example must be countable.
# ---------------------------------------------------------------------------


def _long_history_docs(n=5, words=40):
    return [
        "Previous turn\n[User query]\n"
        + " ".join([_HISTORY_MARKERS[i]] * words)
        + f"\n[Assistant output]\n{_HISTORY_MARKERS[i]} answer number {i}"
        for i in range(n)
    ]


_LONG_TAIL_CONFIG = dict(
    doc_mode="history_only",
    max_doc_num=8,
    max_tool_chunks=0,
    min_doc_num=1,
    max_doc_length=64,
    max_length=128,
    max_system_length=96,
)


def test_hybrid_tail_over_budget_shortens_instead_of_dropping_the_row(tokenizer):
    # k raw docs of max_doc_length tokens each can be several times max_length.
    # Left-truncating them used to leave no room for the answer, so every
    # large-k row died as tool_call_target_truncated -- silently deleting whole
    # strata of the pool and breaking the paired-arm example set.
    example = _example(history_documents=_long_history_docs(5))
    meta = {}
    features, reason = _features(
        tokenizer, example, meta_out=meta, hybrid_tail_k=5, **_LONG_TAIL_CONFIG
    )
    assert reason == "ok"
    assert 0 < meta["hybrid_tail_k"] < 5
    # The shed docs go BACK into the compressed grid, they are not lost.
    assert meta["num_compressed_history_docs"] + meta["hybrid_tail_k"] == meta["num_history_docs"]
    assert meta["num_history_docs"] == 5

    # The surviving tail is still the CHRONOLOGICAL end of the history and is
    # still a raw prefix of the prompt.
    realized = meta["hybrid_tail_k"]
    expected = []
    for index in range(5 - realized, 5):
        expected.extend(
            _chat_template_ids(
                tokenizer,
                [{"role": "user", "content": example.history_documents[index]}],
                max_length=_LONG_TAIL_CONFIG["max_doc_length"],
            )
        )
    assert features["input_ids"][: len(expected)] == expected
    assert features["labels"][: len(expected)] == [-100] * len(expected)
    # ... and the answer survived intact (the whole point of the shedding).
    assert sum(1 for value in features["labels"] if value != -100) > 0


def test_hybrid_tail_fully_shed_degrades_to_k_zero(tokenizer):
    # Not even ONE raw doc fits: the row must still be emitted, byte-identical
    # to the no-tail rendering, with realized k == 0.
    config = dict(_LONG_TAIL_CONFIG)
    config["max_length"] = 64
    example = _example(history_documents=_long_history_docs(5))
    meta = {}
    features, reason = _features(
        tokenizer, example, meta_out=meta, hybrid_tail_k=5, **config
    )
    assert reason == "ok"
    assert meta["hybrid_tail_k"] == 0
    assert meta["num_compressed_history_docs"] == meta["num_history_docs"]
    baseline, baseline_reason = _features(tokenizer, example, **config)
    assert baseline_reason == "ok"
    assert features == baseline


def test_hybrid_tail_choices_keep_the_same_row_set_as_k_zero(tokenizer):
    examples = [
        _example(qid=f"t:{index}", history_documents=_long_history_docs(5))
        for index in range(24)
    ]
    kwargs = dict(tokenizer=tokenizer, **_config(**_LONG_TAIL_CONFIG))
    plain = JointDataset(examples, **kwargs)
    hybrid = JointDataset(examples, hybrid_tail_choices=[0, 5], **kwargs)
    # The paired comparison only holds if both arms train on the same rows.
    assert len(hybrid) == len(plain) == len(examples)
    assert hybrid.skipped_by_reason == plain.skipped_by_reason == {}
    # Both strata were drawn ...
    assert set(hybrid.hybrid_tail_k_drawn_counts) == {0, 5}
    assert sum(hybrid.hybrid_tail_k_drawn_counts.values()) == len(examples)
    # ... and the k=5 stratum realized as a SHORTENED tail, not as a drop.
    assert sum(hybrid.hybrid_tail_k_counts.values()) == len(examples)
    assert max(hybrid.hybrid_tail_k_counts) > 0


def test_hybrid_tail_drawn_counts_cover_every_candidate(tokenizer):
    choices = [0, 1, 3]
    examples = [
        _example(qid=f"t:{index}", history_documents=_history_docs(5)) for index in range(12)
    ]
    kwargs = dict(tokenizer=tokenizer, **_config(**_HYBRID_CONFIG))
    dataset = JointDataset(examples, hybrid_tail_choices=choices, **kwargs)
    expected = Counter(
        random.Random(f"{example.qid}:hybrid_tail").choice(choices) for example in examples
    )
    # Drawn counts cover EVERY candidate (emitted + skipped), so a stratum lost
    # to skips shows up as drawn > realized instead of vanishing.
    assert dataset.hybrid_tail_k_drawn_counts == dict(sorted(expected.items()))
    assert sum(dataset.hybrid_tail_k_drawn_counts.values()) == len(examples)


def test_hybrid_tail_drawn_counts_default_off(tokenizer):
    dataset = JointDataset([_example()], tokenizer=tokenizer, **_config())
    assert dataset.hybrid_tail_k_drawn_counts == {0: 1}
    assert dataset.tools_in_system_missing_tools == 0


def test_tools_in_system_without_selected_tools_is_lenient_but_counted(tokenizer):
    # QA rows (and any caller that does not populate selected_tools) get a BARE
    # system prefix under tools_in_system -- no tools presented at all.  That is
    # deliberately not a skip, but it must be countable: a non-zero counter on a
    # tools_in_system run means part of the mixture is mistrained.
    example = _example(selected_tools=None)
    features, reason = _features(
        tokenizer,
        example,
        doc_mode="history_only",
        tools_in_system=True,
        max_system_length=256,
    )
    assert reason == "ok"
    assert "<TOOLS>" not in _system_text(tokenizer, features)

    dataset = JointDataset(
        [_example(qid="t:0", selected_tools=None), _example(qid="t:1", selected_tools=_selected_tools(1))],
        tokenizer=tokenizer,
        tools_in_system=True,
        **_config(doc_mode="history_only", max_system_length=256),
    )
    assert dataset.tools_in_system_missing_tools == 1


def test_assert_no_leakage_accepts_declared_regime_knobs(tokenizer):
    example = _example(history_documents=_history_docs(5), selected_tools=_selected_tools(2))
    config = dict(_HYBRID_CONFIG)
    config["max_system_length"] = 256
    meta = {}
    features, reason = _features(
        tokenizer, example, meta_out=meta, tools_in_system=True, hybrid_tail_k=2, **config
    )
    assert reason == "ok"
    # Undeclared: the tool schemas in the system prefix and the raw history tail
    # in input_ids both look like leaks.
    with pytest.raises(AssertionError):
        assert_no_leakage(example, features, tokenizer)
    # Declared: both are the intended dialect and must pass.
    assert_no_leakage(
        example,
        features,
        tokenizer,
        tools_in_system=True,
        hybrid_tail_k=meta["hybrid_tail_k"],
    )


def test_traces_source_examples_carry_selected_tools(synthetic_dataset):
    # A1's primary call site: the traces source is what the regime arm consumes,
    # so selected_tools must be populated THERE, matching the rendered pool.
    source = _joint_source(synthetic_dataset)
    examples = list(source)
    assert examples
    for example in examples:
        assert example.selected_tools is not None
        assert len(example.selected_tools) == len(example.tool_documents)
        for tool in example.selected_tools:
            name = tdj._tool_name(tool)
            assert any(f"<NAME> {name}" in doc for doc in example.tool_documents)


def test_shuffled_system_tools_removes_the_gold_first_oracle():
    """tools_in_system must not put the target tool at position 0 every time.

    _select_tools seeds its pool with target[:1], so the raw return order is a
    positional oracle.  The grid path never sees it (it shuffles its own
    copy), but the system prefix renders selected_tools verbatim.
    """
    tools = [{"type": "function", "function": {"name": f"ns{i // 8}.tool_{i}"}} for i in range(64)]
    first_is_target = 0
    trials = 60
    for span in range(trials):
        selected = tdj._select_tools(
            tools, "ns0.tool_0", random.Random(f"seed:{span}"), max_tools_per_sample=32
        )
        assert tdj._tool_name(selected[0]) == "ns0.tool_0"  # the oracle we are removing
        ordered = tdj._shuffled_system_tools(selected, 42, "sess", span)
        assert sorted(tdj._tool_name(t) for t in ordered) == sorted(
            tdj._tool_name(t) for t in selected
        )
        if tdj._tool_name(ordered[0]) == "ns0.tool_0":
            first_is_target += 1
    # uniform position => ~1/32 of trials; the unfixed code scores trials/trials
    assert first_is_target < trials // 4, first_is_target


def test_shuffled_system_tools_is_deterministic():
    tools = [{"type": "function", "function": {"name": f"t{i}"}} for i in range(40)]
    a = tdj._shuffled_system_tools(tools, 42, "sess", 3)
    b = tdj._shuffled_system_tools(tools, 42, "sess", 3)
    c = tdj._shuffled_system_tools(tools, 42, "sess", 4)
    assert [tdj._tool_name(t) for t in a] == [tdj._tool_name(t) for t in b]
    assert [tdj._tool_name(t) for t in a] != [tdj._tool_name(t) for t in c]
