# -*- coding: utf-8 -*-
"""CPU-only unit tests for train/train_data_joint_multisource.py (G-medium mixture).

Fixtures: ``testdata_gjoint/fixtures.json`` — REAL rows dumped from the server
datasets (truncated but structure-faithful), loaded with ``encoding="utf-8"``.
The conversion cores are pure (parsed dicts in, JointExamples out), so most
tests need no parquet; the source-level tests write tiny parquet/jsonl files
with pyarrow exactly like test_train_data_joint.py does.

Coverage:
a. Toucan: example points (user at i>0 whose following assistant turn has a
   tool call; i==0 excluded), qid/session/subset, answer surface identical to
   ``_render_agent_output_messages`` on the same normalized turn, history
   chronological ``Previous turn`` docs, no tool schemas in system_prompt,
   determinism, ``require_tool_call=False`` keeps text-only turns;
b. Open-SWE: resolved==1 filter, first-action (empty history) skip, Thought+
   Action answer surface, trajectory system prompt, determinism;
c. QA: hotpotqa raw documents, 2wiki [[title, sents]] flattening (plus the
   unparseable-context raw fallback), longmagpie "?" suffix split + skip;
d. source-level (pyarrow): streaming, keep_qids prefilter, max_records,
   max_samples_per_session subsampling, split validation;
e. every produced example survives ``JointDataset.preprocess_example`` with
   the whitespace self-test tokenizer and passes ``assert_no_leakage``.

Run from the repo root (local venv has torch/transformers/datasets/pytest):
  pytest python/train/test_train_data_joint_multisource.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make python/ importable when pytest is invoked from the repo root.
_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from train.train_data_joint import (  # noqa: E402
    JointDataset,
    _WhitespaceSelfTestTokenizer,
    assert_no_leakage,
)
from train.train_data_multiturn import (  # noqa: E402
    _render_agent_output_messages,
)
from train.train_data import DEFAULT_SYSTEM_PROMPT  # noqa: E402
import train.train_data_joint_multisource as tdm  # noqa: E402
from train.train_data_joint_multisource import (  # noqa: E402
    OpenSWEJointSource,
    QADocsJointSource,
    ToucanJointSource,
    hotpotqa_row_to_example,
    longmagpie_row_to_example,
    openswe_row_to_examples,
    qid_source_family,
    split_longmagpie_question,
    toucan_row_to_examples,
    wiki2_row_to_example,
)


_FIXTURES_PATH = Path(__file__).resolve().parent / "testdata_gjoint" / "fixtures.json"


@pytest.fixture(scope="module")
def fixtures():
    # utf-8 explicitly: the rows carry CJK/emoji and the Windows default is GBK.
    return json.loads(_FIXTURES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tokenizer():
    return _WhitespaceSelfTestTokenizer()


# ---------------------------------------------------------------------------
# Toucan conversion core.
# ---------------------------------------------------------------------------


def _toucan_messages(row):
    return json.loads(row["messages"])


def test_toucan_example_points_and_qids(fixtures):
    row = fixtures["toucan"][1]  # user turns at 0, 5, 10, 12
    examples = toucan_row_to_examples(row)
    uuid = row["uuid"]
    # u5's turn (assistant + tool_call) qualifies; u10/u12 have text-only
    # turns; u0 has no history.  Exactly one example.
    assert [example.qid for example in examples] == [f"toucan:{uuid}:u5"]
    example = examples[0]
    assert example.session_id == f"toucan:{uuid}"
    assert example.subset == "toucan:multi-turn"
    assert example.target_tool == "weather-mcp-server-get_alerts"
    assert example.system_prompt == DEFAULT_SYSTEM_PROMPT
    assert example.current_messages == [
        {"role": "user", "content": _toucan_messages(row)[5]["content"]}
    ]


def test_toucan_answer_surface_matches_traces_renderer(fixtures):
    row = fixtures["toucan"][1]
    example = toucan_row_to_examples(row)[0]
    messages = _toucan_messages(row)
    # The same normalized turn rendered by the traces path's renderer must
    # reproduce the example answer byte-for-byte.
    normalized_turn = [
        {"role": "assistant", "content": messages[6]["content"]},
        {
            "role": "assistant",
            "tool_calls": [{"name": "weather-mcp-server-get_alerts", "arguments": {"state": "CO"}}],
        },
    ]
    reference, has_tool_call = _render_agent_output_messages(normalized_turn, None)
    assert has_tool_call
    assert example.answer == reference
    # Byte-level pin of the unified Action surface (minified JSON, name first).
    assert example.answer.endswith(
        'Action:\n<tool_call>\n{"name":"weather-mcp-server-get_alerts",'
        '"arguments":{"state":"CO"}}\n</tool_call>'
    )
    assert example.answer.startswith(messages[6]["content"])


def test_toucan_history_chronological_and_tool_calls_rendered(fixtures):
    row = fixtures["toucan"][1]
    example = toucan_row_to_examples(row)[0]
    messages = _toucan_messages(row)
    docs = example.history_documents
    assert len(docs) == 2
    assert all(doc.startswith("Previous turn") for doc in docs)
    # First turn: original question + assistant text + rendered Action block.
    assert messages[0]["content"] in docs[0]
    assert '"name":"mcp服务-query_weather","arguments":{"city":"Denver"}' in docs[0]
    assert "Action:\n<tool_call>" in docs[0]
    # Second turn doc: the tool result folded into a user section (traces
    # history style), then the assistant summary.
    assert "🌍 Denver, US" in docs[1]
    assert docs[1].index("[User query]") < docs[1].index("[Assistant output]")
    # Chronology across docs.
    joined = "\n".join(docs)
    assert joined.index("current weather in Denver") < joined.index("🌍 Denver, US")


def test_toucan_no_tool_schema_in_system_prompt(fixtures):
    for row in fixtures["toucan"]:
        for example in toucan_row_to_examples(row):
            assert example.system_prompt == DEFAULT_SYSTEM_PROMPT
            assert "<TOOL>" not in example.system_prompt
            for doc in example.tool_documents:
                assert doc not in example.system_prompt


def test_toucan_single_user_row_yields_no_examples(fixtures):
    # Row 0 has exactly one user message (index 0): no non-empty-history point.
    assert toucan_row_to_examples(fixtures["toucan"][0]) == []


def test_toucan_require_tool_call_false_keeps_text_turns(fixtures):
    row = fixtures["toucan"][1]
    uuid = row["uuid"]
    examples = toucan_row_to_examples(row, require_tool_call=False)
    assert [example.qid for example in examples] == [
        f"toucan:{uuid}:u5",
        f"toucan:{uuid}:u10",
        f"toucan:{uuid}:u12",
    ]
    messages = _toucan_messages(row)
    # Text-only turns: the plain assistant text is the answer, no Action block.
    assert examples[1].answer == messages[11]["content"]
    assert examples[2].answer == messages[13]["content"]
    assert examples[1].target_tool is None and examples[1].target_tool_doc_index is None


def test_toucan_determinism(fixtures):
    row = fixtures["toucan"][1]
    assert toucan_row_to_examples(row) == toucan_row_to_examples(row)
    # The tool-render seed is driven by split_seed: a different seed may pick
    # a different render variant but must itself be reproducible.
    assert toucan_row_to_examples(row, split_seed=7) == toucan_row_to_examples(row, split_seed=7)


def test_toucan_tool_docs_and_target_index(fixtures):
    example = toucan_row_to_examples(fixtures["toucan"][1])[0]
    assert len(example.tool_documents) == 3  # 3 schema tools, under the 32 cap
    assert example.target_tool_doc_index is not None
    target_doc = example.tool_documents[example.target_tool_doc_index]
    assert "weather-mcp-server-get_alerts" in target_doc
    names = "\n".join(example.tool_documents)
    for name in ("mcp服务-query_weather", "mcp服务-add", "weather-mcp-server-get_alerts"):
        assert name in names


# ---------------------------------------------------------------------------
# Open-SWE conversion core.
# ---------------------------------------------------------------------------


def test_openswe_resolved_filter(fixtures):
    assert openswe_row_to_examples(fixtures["openswe"][1], subset="openswe:test") == []  # resolved=0
    examples = openswe_row_to_examples(fixtures["openswe"][0], subset="openswe:test")
    assert [example.qid for example in examples] == [
        "openswe:f8853683-87bd-418b-b87f-19ab95c60317:a4",
        "openswe:f8853683-87bd-418b-b87f-19ab95c60317:a6",
    ]
    # a2 (the trajectory's first action) is skipped: no earlier assistant
    # message after the first user message -> history would be empty.
    assert all(not example.qid.endswith(":a2") for example in examples)
    for example in examples:
        assert example.session_id == "openswe:f8853683-87bd-418b-b87f-19ab95c60317"
        assert example.subset == "openswe:test"
        assert example.target_tool == "str_replace_editor"


def test_openswe_answer_surface_matches_traces_renderer(fixtures):
    row = fixtures["openswe"][0]
    examples = openswe_row_to_examples(row, subset="openswe:test")
    action = row["trajectory"][4]
    normalized_action = {
        "role": "assistant",
        "content": action["content"],
        "reasoning_content": action["reasoning_content"],
        "tool_calls": [
            {"name": "str_replace_editor", "arguments": {"command": "view", "path": "/workspace/python-attrs__attrs__1.0/src"}}
        ],
    }
    reference, has_tool_call = _render_agent_output_messages([normalized_action], None)
    assert has_tool_call
    assert examples[0].answer == reference
    # Reasoning is kept as a Thought block; arguments are the parsed object,
    # minified (byte-compatible with the traces answer surface).
    assert examples[0].answer.startswith("Thought:\nLet me explore the source directory")
    assert '"arguments":{"command":"view","path":"/workspace/python-attrs__attrs__1.0/src"}' in examples[0].answer
    assert "Action:\n<tool_call>" in examples[0].answer


def test_openswe_history_and_system_prompt(fixtures):
    row = fixtures["openswe"][0]
    examples = openswe_row_to_examples(row, subset="openswe:test")
    first, second = examples
    # a4 history covers trajectory messages 2..3; a6 covers 2..5.  Turn
    # grouping follows the traces style: a tool result opens a new
    # "[User query]" section and the NEXT assistant action joins its
    # "[Assistant output]", so docs are 2 / 3 (not one doc per message).
    assert len(first.history_documents) == 2
    assert len(second.history_documents) == 3
    assert all(doc.startswith("Previous turn") for doc in second.history_documents)
    assert second.history_documents[0] == first.history_documents[0]
    assert '"name":"str_replace_editor"' in first.history_documents[0]
    assert "Here's the files and directories up to 2 levels deep" in first.history_documents[1]
    # Chronological order across the a6 docs: root action -> root listing ->
    # src action -> src listing.
    joined = "\n".join(second.history_documents)
    positions = [
        joined.index('"path":"/workspace/python-attrs__attrs__1.0"'),
        joined.index("AUTHORS.rst"),
        joined.index('"path":"/workspace/python-attrs__attrs__1.0/src"'),
        joined.index("src/attr/__init__.py"),
    ]
    assert positions == sorted(positions)
    # Current prompt is the FIRST user message for every action of the trace.
    assert first.current_messages == [{"role": "user", "content": row["trajectory"][1]["content"]}]
    assert second.current_messages == first.current_messages
    # System prompt is the trajectory's own system message (no schemas added).
    assert first.system_prompt == row["trajectory"][0]["content"]
    assert "<TOOL>" not in first.system_prompt
    assert "<PARAMETERS>" not in first.system_prompt


def test_openswe_tool_docs_and_target_index(fixtures):
    row = fixtures["openswe"][0]
    example = openswe_row_to_examples(row, subset="openswe:test")[0]
    # The fixture row's tools column ships only execute_bash/think; the called
    # str_replace_editor is NOT among the definitions, so (by the JointExample
    # contract) target_tool is set but target_tool_doc_index is None.
    assert example.target_tool == "str_replace_editor"
    assert example.target_tool_doc_index is None
    assert len(example.tool_documents) == 2
    names = "\n".join(example.tool_documents)
    assert "execute_bash" in names and "think" in names
    # With the target schema present in the tools column, its index is reported.
    import copy

    enriched = copy.deepcopy(row)
    enriched["tools"] = list(row["tools"]) + [
        json.dumps({
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": "Custom editing tool.",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            },
        })
    ]
    enriched_example = openswe_row_to_examples(enriched, subset="openswe:test")[0]
    assert enriched_example.target_tool_doc_index is not None
    assert "str_replace_editor" in enriched_example.tool_documents[enriched_example.target_tool_doc_index]


def test_openswe_determinism(fixtures):
    row = fixtures["openswe"][0]
    assert openswe_row_to_examples(row, subset="openswe:test") == openswe_row_to_examples(row, subset="openswe:test")


# ---------------------------------------------------------------------------
# QA conversion cores.
# ---------------------------------------------------------------------------


def test_hotpotqa_mapping(fixtures):
    rows = fixtures["hotpotqa"]
    examples = [hotpotqa_row_to_example(row, index) for index, row in enumerate(rows)]
    assert all(example is not None for example in examples)
    first = examples[0]
    row = rows[0]
    assert first.qid == f"qa:hotpotqa:{row['_id']}"
    assert first.session_id == first.qid
    assert first.subset == "qa:hotpotqa"
    # Raw document strings, no added prefix.
    assert first.history_documents == list(row["documents"])
    assert all(not doc.startswith("Previous turn") for doc in first.history_documents)
    assert first.current_messages == [{"role": "user", "content": row["question"]}]
    assert first.answer == row["answer"]
    assert first.tool_documents == []
    assert first.target_tool is None and first.target_tool_doc_index is None


def test_wiki2_mapping_native_and_fallback(fixtures):
    # Native [[title, [sentence, ...]], ...] shape (synthetic, mirroring the
    # fixture row's schema): one doc per entry, title line + joined sentences.
    row = {
        "_id": "wiki2-synthetic",
        "question": "Which film came first?",
        "answer": "Move",
        "context": [
            ["Move (1970 film)", ["Move is a 1970 film.", "It was directed by Stuart Rosenberg."]],
            ["Méditerranée (1963 film)", ["Méditerranée is a 1963 French film."]],
        ],
    }
    example = wiki2_row_to_example(row)
    assert example is not None
    assert example.qid == "qa:2wiki:wiki2-synthetic"
    assert example.history_documents == [
        "Move (1970 film)\nMove is a 1970 film. It was directed by Stuart Rosenberg.",
        "Méditerranée (1963 film)\nMéditerranée is a 1963 French film.",
    ]
    assert example.current_messages == [{"role": "user", "content": "Which film came first?"}]
    assert example.answer == "Move"

    # The fixture row's context is a truncated (unparseable) JSON string: the
    # converter falls back to one raw-text document instead of dropping it.
    fixture_example = wiki2_row_to_example(fixtures["wiki2"][0])
    assert fixture_example is not None
    assert fixture_example.qid == f"qa:2wiki:{fixtures['wiki2'][0]['_id']}"
    assert fixture_example.history_documents == [fixtures["wiki2"][0]["context"]]


def test_longmagpie_question_split_rule():
    context, question = split_longmagpie_question(
        "The study fed spiders graphene. Silk was collected before and after the treatment."
        "No incorporation was repaired.Can you summarize the key findings of the study?"
    )
    assert question == "Can you summarize the key findings of the study?"
    assert context.endswith("No incorporation was repaired.")
    # Multiple trailing question sentences form one run.
    context, question = split_longmagpie_question(
        "Some background text here. What happened first? Why did it matter?"
    )
    assert question == "What happened first? Why did it matter?"
    assert context == "Some background text here."
    # No trailing "?" sentence -> None (row skipped).
    assert split_longmagpie_question("All statements. No question at the end.") is None
    assert split_longmagpie_question("") is None
    # A document that is ONLY a question has no context left -> None.
    assert split_longmagpie_question("What is this?") is None


def test_longmagpie_row_mapping_and_skip(fixtures):
    # Synthetic positive row (the fixture row's user content is truncated and
    # carries no "?" suffix, so it exercises the skip path below).
    row = {
        "messages": [
            {"role": "user", "content": "A long document body. It has two sentences.What is the main claim?"},
            {"role": "assistant", "content": "The main claim is X."},
        ]
    }
    example = longmagpie_row_to_example(row, 7)
    assert example is not None
    assert example.qid == "qa:longmagpie:7"
    assert example.session_id == "qa:longmagpie:7"
    assert example.subset == "qa:longmagpie"
    assert example.history_documents == ["A long document body. It has two sentences."]
    assert example.current_messages == [{"role": "user", "content": "What is the main claim?"}]
    assert example.answer == "The main claim is X."
    assert example.tool_documents == []
    # Fixture row: truncated content ends without "?" -> skipped.
    assert longmagpie_row_to_example(fixtures["longmagpie"][0], 0) is None


def test_qid_source_family_mapping():
    assert qid_source_family("sess-1:0") == "traces"
    assert qid_source_family("toucan:abc:u2") == "toucan"
    assert qid_source_family("openswe:traj-1:a4") == "openswe"
    assert qid_source_family("qa:hotpotqa:x") == "qa"
    assert qid_source_family("qa:2wiki:y") == "qa"


# ---------------------------------------------------------------------------
# Source-level tests (pyarrow parquet IO; same fixture style as
# test_train_data_joint.py).
# ---------------------------------------------------------------------------


@pytest.fixture()
def toucan_dataset(tmp_path, fixtures):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "toucan"
    sft = root / "SFT"
    sft.mkdir(parents=True)
    rows = fixtures["toucan"] + [
        # A non-multi-turn row the pushed-down filter must exclude.
        {
            "uuid": "00000000-0000-0000-0000-000000000000",
            "subset_name": "single-turn-original",
            "question": "[]",
            "target_tools": "",
            "tools": fixtures["toucan"][0]["tools"],
            "messages": fixtures["toucan"][0]["messages"],
        }
    ]
    table = pa.table({key: [row[key] for row in rows] for key in rows[0]})
    pq.write_table(table, sft / "train-00000-of-00001.parquet")
    return str(root)


@pytest.fixture()
def openswe_dataset(tmp_path, fixtures):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "open-swe-traces" / "data" / "qwen35_sweagent"
    root.mkdir(parents=True)
    keys = ["instance_id", "repo", "license", "trajectory_id", "trajectory", "tools", "resolved", "hf_dataset_name"]
    table = pa.table({key: [row[key] for row in fixtures["openswe"]] for key in keys})
    pq.write_table(table, root / "shard-0.parquet")
    return str(tmp_path / "open-swe-traces")


def test_toucan_source_streams_and_filters(toucan_dataset, fixtures):
    source = ToucanJointSource(toucan_dataset)
    examples = list(source)
    uuid = fixtures["toucan"][1]["uuid"]
    assert [example.qid for example in examples] == [f"toucan:{uuid}:u5"]
    # Streaming is lazy and deterministic: a second pass yields the same.
    assert list(source) == examples
    assert len(source) == 1  # materializes + caches
    assert list(source) == examples

    # keep_qids prefilter.
    kept = list(ToucanJointSource(toucan_dataset, keep_qids=frozenset({f"toucan:{uuid}:u5"})))
    assert [example.qid for example in kept] == [f"toucan:{uuid}:u5"]
    assert list(ToucanJointSource(toucan_dataset, keep_qids=frozenset({"toucan:other:u1"}))) == []

    # max_records / max_samples_per_session smoke.
    assert list(ToucanJointSource(toucan_dataset, max_records=0)) == []
    assert len(list(ToucanJointSource(toucan_dataset, max_samples_per_session=1))) == 1


def test_openswe_source_streams_and_subsamples(openswe_dataset, fixtures):
    source = OpenSWEJointSource(openswe_dataset)
    examples = list(source)
    assert len(examples) == 2  # resolved==1 row only, actions a4/a6
    assert {example.subset for example in examples} == {"openswe:qwen35_sweagent"}
    assert source.stats["rows_unresolved"] == 1
    # One example per trajectory when sub-sampled.
    assert len(list(OpenSWEJointSource(openswe_dataset, max_samples_per_session=1))) == 1
    keep = examples[0].qid
    kept = list(OpenSWEJointSource(openswe_dataset, keep_qids=frozenset({keep})))
    assert [example.qid for example in kept] == [keep]


def test_qa_source_counts_and_stats(tmp_path, fixtures):
    import pyarrow as pa
    import pyarrow.parquet as pq

    hotpotqa_path = tmp_path / "hotpotqa_train.jsonl"
    hotpotqa_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in fixtures["hotpotqa"]),
        encoding="utf-8",
    )
    wiki2_dir = tmp_path / "wiki2"
    wiki2_dir.mkdir()
    pq.write_table(
        pa.table({key: [row[key] for row in fixtures["wiki2"]] for key in fixtures["wiki2"][0]}),
        wiki2_dir / "train.parquet",
    )
    longmagpie_dir = tmp_path / "longmagpie"
    (longmagpie_dir / "data").mkdir(parents=True)
    lm_rows = fixtures["longmagpie"] + [
        {
            "messages": [
                {"role": "user", "content": "Document body here. More body.What is the point?"},
                {"role": "assistant", "content": "The point is Y."},
            ]
        }
    ]
    pq.write_table(
        pa.table({"messages": [row["messages"] for row in lm_rows]}),
        longmagpie_dir / "data" / "shard-0.parquet",
    )
    source = QADocsJointSource(
        hotpotqa_path=str(hotpotqa_path),
        wiki2_path=str(wiki2_dir),
        longmagpie_path=str(longmagpie_dir),
    )
    examples = list(source)
    qids = [example.qid for example in examples]
    assert qids[:2] == [f"qa:hotpotqa:{row['_id']}" for row in fixtures["hotpotqa"]]
    assert qids[2] == f"qa:2wiki:{fixtures['wiki2'][0]['_id']}"
    # The truncated fixture longmagpie row is skipped (counted); the synthetic
    # one with a "?" suffix is kept with its global row index.
    assert qids[3] == "qa:longmagpie:1"
    assert source.stats["longmagpie_rows"] == 2
    assert source.stats["longmagpie_skipped"] == 1
    assert all(qid_source_family(qid) == "qa" for qid in qids)
    # keep_qids prefilter on a single qa qid.
    kept = list(
        QADocsJointSource(
            hotpotqa_path=str(hotpotqa_path),
            keep_qids=frozenset({qids[0]}),
        )
    )
    assert [example.qid for example in kept] == [qids[0]]


def test_sources_reject_non_train_split(tmp_path, fixtures):
    with pytest.raises(ValueError, match="train-split only"):
        ToucanJointSource(str(tmp_path), split="eval")
    with pytest.raises(ValueError, match="train-split only"):
        OpenSWEJointSource(str(tmp_path), split="eval")
    with pytest.raises(ValueError, match="train-split only"):
        QADocsJointSource(hotpotqa_path=str(tmp_path / "x.jsonl"), split="eval")
    with pytest.raises(ValueError, match="at least one"):
        QADocsJointSource()


# ---------------------------------------------------------------------------
# End-to-end: every produced example preprocesses cleanly (no skip reasons)
# and passes the leakage self-check with the whitespace tokenizer.
# ---------------------------------------------------------------------------


def test_all_fixture_examples_preprocess_without_leakage(fixtures, tokenizer):
    examples = []
    for row in fixtures["toucan"]:
        examples.extend(toucan_row_to_examples(row))
        examples.extend(toucan_row_to_examples(row, require_tool_call=False))
    examples.extend(openswe_row_to_examples(fixtures["openswe"][0], subset="openswe:test"))
    for index, row in enumerate(fixtures["hotpotqa"]):
        examples.append(hotpotqa_row_to_example(row, index))
    examples.append(wiki2_row_to_example(fixtures["wiki2"][0]))
    examples.append(
        longmagpie_row_to_example(
            {
                "messages": [
                    {"role": "user", "content": "A long document body. It has two sentences.What is the main claim?"},
                    {"role": "assistant", "content": "The main claim is X."},
                ]
            },
            0,
        )
    )
    examples = [example for example in examples if example is not None]
    assert examples  # the fixtures must actually exercise every family
    assert {qid_source_family(example.qid) for example in examples} == {"toucan", "openswe", "qa"}
    config = dict(
        tokenizer=tokenizer,
        max_length=4096,
        max_doc_length=1024,
        min_doc_num=1,
        max_doc_num=16,
        max_system_length=1024,
    )
    for example in examples:
        row, reason = JointDataset.preprocess_example(example, **config)
        assert row is not None, f"{example.qid} unexpectedly skipped: {reason}"
        assert_no_leakage(example, row, tokenizer)
