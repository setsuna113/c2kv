"""Real-shaped source bridges and compact paired-corpus integration tests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from history_memory.dataset import iter_decisions
from history_memory.packing import select_view
from history_memory.policy import RuntimeConfig
from history_memory.preparation import (
    PackingConfig,
    PrefixLifecyclePlanner,
    PreparedCorpus,
    SamplingConfig,
    prepare_paired_corpus,
)
from history_memory.sources import (
    SourceRowError,
    adapt_hotpotqa_row,
    load_g_sources,
    normalize_openai_messages,
)


class DeterministicTokenizer:
    vocab_size = 256
    special_tokens_map = {"eos_token": "<eos>"}
    chat_template = "deterministic-test-v1"

    @staticmethod
    def apply_chat_template(
        messages,
        *,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        truncation=False,
    ):
        assert tokenize and not enable_thinking and not truncation
        rendered = []
        if tools:
            rendered.append("<tools>" + json.dumps(tools, sort_keys=True) + "<|end|>")
        for message in messages:
            value = dict(message)
            role = value.pop("role")
            rendered.append(
                f"<{role}>"
                + json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "<|end|>"
            )
        if add_generation_prompt:
            rendered.append("<assistant>")
        return tuple(ord(character) for character in "".join(rendered))


TOKENIZER = DeterministicTokenizer()


def _tool(name="read_file"):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Read a file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }


def _call(call_id: str, name: str, path: str):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": {"path": path}},
    }


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_generic_missing_result_binding_is_unique_or_rejected():
    raw = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("a", "read", "a"), _call("b", "read", "b")],
        },
        {"role": "tool", "name": "read", "content": "ambiguous"},
    ]
    with pytest.raises(SourceRowError, match="ambiguous_tool_result_binding"):
        normalize_openai_messages(raw, namespace="n")

    distinct = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call("a", "read", "a"), _call("b", "stat", "b")],
        },
        {"role": "tool", "name": "stat", "content": "B"},
        {"role": "tool", "name": "read", "content": "A"},
    ]
    audit = Counter()
    normalized = normalize_openai_messages(distinct, namespace="n", audit=audit)
    assert [message["tool_call_id"] for message in normalized[1:]] == ["b", "a"]
    assert audit["tool_result_ids_bound_by_unique_name"] == 2


def test_hotpotqa_accepts_huggingface_context_struct_and_legacy_documents():
    row = adapt_hotpotqa_row(
        {
            "id": "hp-context",
            "context": {
                "title": ["First title", "Second title"],
                "sentences": [
                    ["First sentence.", "Second sentence."],
                    ["Another document."],
                ],
            },
            "question": "Which document?",
            "answer": "The first one.",
        }
    )
    assert [message["content"] for message in row["messages"]] == [
        "First title\nFirst sentence. Second sentence.",
        "Second title\nAnother document.",
        "Which document?",
        "The first one.",
    ]
    assert row["source_metadata"]["normalized_message_sources"][:2] == [
        {"field": "context", "title_index": 0, "sentences_index": 0},
        {"field": "context", "title_index": 1, "sentences_index": 1},
    ]

    legacy = adapt_hotpotqa_row(
        {
            "_id": "hp-documents",
            "documents": ["Legacy document"],
            "question": "What?",
            "answer": "Legacy.",
        }
    )
    assert legacy["messages"][0]["content"] == "Legacy document"
    assert legacy["source_metadata"]["normalized_message_sources"][0] == {
        "field": "documents",
        "index": 0,
    }

    with_empty_context = adapt_hotpotqa_row(
        {
            "id": "hp-empty-context",
            "context": {
                "title": ["", "Retained title"],
                "sentences": [[], ["Retained sentence."]],
            },
            "question": "Retained?",
            "answer": "Yes.",
        }
    )
    assert with_empty_context["source_metadata"]["normalized_message_sources"][0] == {
        "field": "context",
        "title_index": 1,
        "sentences_index": 1,
    }

    with pytest.raises(SourceRowError, match="invalid_hotpotqa_context"):
        adapt_hotpotqa_row(
            {
                "id": "hp-mismatched-context",
                "context": {"title": ["Only title"], "sentences": []},
                "question": "Invalid?",
                "answer": "Yes.",
            }
        )


def test_toucan_limit_counts_multiturn_rows_after_subset_filter(tmp_path):
    toucan = tmp_path / "toucan"
    rows = [
        {"uuid": "single-original", "subset_name": "single-turn-original"},
        {"uuid": "single-diverse", "subset_name": "single-turn-diversify"},
    ]
    rows.extend(
        {
            "uuid": f"multi-{index}",
            "subset_name": "multi-turn",
            "tools": "[]",
            "messages": json.dumps(
                [
                    {"role": "user", "content": f"Question {index}"},
                    {"role": "assistant", "content": f"Answer {index}"},
                ]
            ),
        }
        for index in range(3)
    )
    _write_jsonl(toucan / "SFT" / "train.jsonl", rows)

    loaded = load_g_sources(toucan_path=toucan, max_rows_per_source=2)

    assert [row["session_id"] for row in loaded.rows] == ["multi-0", "multi-1"]
    assert loaded.audit["toucan.rows_scanned"] == 5
    assert loaded.audit["toucan.rows_seen"] == 4
    assert loaded.audit["toucan.skipped.non_multiturn_subset"] == 2
    assert loaded.audit["toucan.sessions_emitted"] == 2
    assert loaded.audit["toucan.truncated_at_max_rows"] == 1


def test_load_g_sources_runs_all_six_file_bridges_and_preserves_provenance(tmp_path):
    traces = tmp_path / "traces"
    trace_rows = [
        {
            "session_id": "trace-good",
            "benchmark": "toolbench_train",
            "spans": [
                {
                    "span_id": "s1",
                    "start_time": "1",
                    "attributes": {
                        "gen_ai.tool.definitions": json.dumps([_tool()]),
                        "gen_ai.input.messages": json.dumps(
                            [{"role": "user", "parts": [{"type": "text", "content": "Read a.txt"}]}]
                        ),
                        "gen_ai.output.messages": json.dumps(
                            [
                                {
                                    "parts": [
                                        {
                                            "type": "tool_call",
                                            "id": "trace-call",
                                            "name": "read_file",
                                            "arguments": {"path": "a.txt"},
                                        }
                                    ]
                                }
                            ]
                        ),
                    },
                },
                {
                    "span_id": "s2",
                    "start_time": "2",
                    "attributes": {
                        "gen_ai.input.messages": json.dumps(
                            [
                                {"role": "user", "parts": [{"type": "text", "content": "Read a.txt"}]},
                                {
                                    "role": "assistant",
                                    "parts": [
                                        {
                                            "type": "tool_call",
                                            "id": "trace-call",
                                            "name": "read_file",
                                            "arguments": {"path": "a.txt"},
                                        }
                                    ],
                                },
                                {"role": "tool", "tool_call_id": "trace-call", "content": "alpha"},
                            ]
                        ),
                        "gen_ai.output.messages": json.dumps([{"content": "Found alpha."}]),
                    },
                },
            ],
        },
        {"session_id": "trace-airline", "benchmark": "tau2_airline", "spans": []},
    ]
    _write_jsonl(traces / "data" / "train.jsonl", trace_rows)
    split_manifest = tmp_path / "split.json"
    split_manifest.write_text(
        json.dumps(
            {
                "taskproxy_disjoint": {
                    "train_session_ids": ["trace-good", "trace-airline"],
                    "eval_session_ids": ["trace-eval"],
                }
            }
        ),
        encoding="utf-8",
    )

    toucan = tmp_path / "toucan"
    _write_jsonl(
        toucan / "SFT" / "train.jsonl",
        [
            {
                "uuid": "toucan-1",
                "subset_name": "multi-turn",
                "tools": json.dumps([_tool("read"), _tool("stat")]),
                "messages": json.dumps(
                    [
                        {"role": "user", "content": "Inspect two files"},
                        {"role": "assistant", "content": "I will inspect them."},
                        {"role": "tool_call", "content": "{'name':'read','arguments':{'path':'a'}}"},
                        {"role": "tool_call", "content": "{'name':'stat','arguments':{'path':'b'}}"},
                        {"role": "tool_response", "content": "A"},
                        {"role": "tool_response", "content": "B"},
                        {"role": "assistant", "content": "Done."},
                    ]
                ),
            }
        ],
    )

    openswe = tmp_path / "openswe"
    _write_jsonl(
        openswe / "data" / "openhands.jsonl",
        [
            {
                "trajectory_id": "swe-1",
                "instance_id": "repo-1",
                "resolved": 1,
                "tools": [json.dumps(_tool("edit"))],
                "trajectory": [
                    {"role": "system", "content": "Edit carefully."},
                    {"role": "user", "content": "Fix it."},
                    {"role": "assistant", "content": None, "tool_calls": [_call("edit-1", "edit", "x.py")]},
                    {"role": "tool", "tool_call_id": "edit-1", "content": "ok"},
                    {"role": "assistant", "content": "Fixed."},
                ],
            },
            {"trajectory_id": "swe-bad", "resolved": 0, "trajectory": []},
        ],
    )

    hotpot = tmp_path / "hotpot"
    _write_jsonl(
        hotpot / "data" / "train.jsonl",
        [{"_id": "hp-1", "documents": ["Doc one", "Doc two"], "question": "Which?", "answer": "One"}],
    )
    wiki = tmp_path / "wiki.jsonl"
    _write_jsonl(
        wiki,
        [{"_id": "w-1", "context": [["Title", ["Sentence one.", "Sentence two."]]], "question": "What?", "answer": "Answer"}],
    )
    longmagpie = tmp_path / "longmagpie"
    _write_jsonl(
        longmagpie / "data" / "shard-a.jsonl",
        [{"messages": [{"role": "user", "content": "A document. More context.What is it?"}, {"role": "assistant", "content": "Context."}]}],
    )

    loaded = load_g_sources(
        traces_path=traces,
        traces_split_manifest=split_manifest,
        toucan_path=toucan,
        openswe_path=openswe,
        hotpotqa_path=hotpot,
        wiki2_path=wiki,
        longmagpie_path=longmagpie,
        max_rows_per_source=20,
    )
    assert {row["source"] for row in loaded.rows} == {
        "agent-llm-traces",
        "toucan",
        "openswe:openhands",
        "qa:hotpotqa",
        "qa:2wiki",
        "qa:longmagpie",
    }
    for row in loaded.rows:
        provenance = row["source_metadata"]["normalized_message_sources"]
        assert len(provenance) == len(row["messages"])
    trace = next(row for row in loaded.rows if row["source"] == "agent-llm-traces")
    assert [message["role"] for message in trace["messages"]] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert trace["messages"][1]["tool_calls"][0]["id"] == "trace-call"
    assert trace["messages"][2]["tool_call_id"] == "trace-call"
    toucan_row = next(row for row in loaded.rows if row["source"] == "toucan")
    assert [message["role"] for message in toucan_row["messages"]] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert loaded.audit["toucan.tool_result_ids_bound_by_source_order"] == 1
    assert loaded.audit["toucan.tool_result_ids_bound_by_single_pending"] == 1
    assert loaded.audit["agent-llm-traces.skipped.excluded_airline"] == 1
    assert loaded.audit["openswe.skipped.unresolved"] == 1


def _recovery_row(session_id="recover"):
    return {
        "session_id": session_id,
        "source": "normalized-agent",
        "split": "train",
        "task_id": session_id,
        "template_id": "tool-recovery",
        "tools": [_tool("read_file")],
        "messages": [
            {"role": "system", "content": "Use files."},
            {"role": "user", "content": "Read alpha.txt"},
            {"role": "assistant", "content": None, "tool_calls": [_call("alpha", "read_file", "alpha.txt")]},
            {"role": "tool", "tool_call_id": "alpha", "content": "alpha payload"},
            {"role": "assistant", "content": "Saved the result."},
            {"role": "user", "content": "Use alpha.txt now."},
            {"role": "assistant", "content": "The alpha payload is ready."},
            {"role": "user", "content": "Continue with an unrelated summary."},
            {"role": "assistant", "content": "Continuing."},
            {"role": "user", "content": "Finish the unrelated summary."},
            {"role": "assistant", "content": "Finished."},
        ],
    }


def _packing(**overrides):
    values = dict(
        ratios=(4, 8),
        recent_tool_events=0,
        max_chunk_tokens=256,
        chunk_overlap=16,
        max_chunks=64,
        max_encoder_tokens=50_000,
        max_system_tokens=10_000,
        max_workspace_tokens=10_000,
        max_target_tokens=10_000,
        max_sequence_tokens=50_000,
    )
    values.update(overrides)
    return PackingConfig(**values)


def _sampling(**overrides):
    values = dict(
        max_rows_per_source=100,
        max_sessions=100,
        max_decisions_per_session=64,
        max_total_decisions=100,
        max_presented_tokens_per_arm=1_000_000,
        qa_target_fraction=0.15,
        repetitions=1,
        seed=42,
    )
    values.update(overrides)
    return SamplingConfig(**values)


def _policy(**overrides):
    values = dict(
        mode="persistent",
        history_budget_bytes=1_000_000,
        workspace_budget_bytes=1_000_000,
        lease_decisions=2,
        max_retrieved_events=2,
    )
    values.update(overrides)
    return RuntimeConfig(**values)


def test_lifecycle_planner_recovers_retains_and_releases_without_target_access():
    decisions = list(iter_decisions(_recovery_row()))
    planner = PrefixLifecyclePlanner(TOKENIZER, _packing(), _policy(), kv_bytes_per_token=1)
    metadata_by_target = {}
    views = {}
    for decision in decisions:
        static = select_view(decision.store, recent_tool_events=0)
        view, metadata = planner(decision, static)
        metadata_by_target[decision.target_dict().get("content")] = metadata
        views[decision.target_dict().get("content")] = (static, view)
    recovered = metadata_by_target["The alpha payload is ready."]
    assert recovered["retrieved_event_ids"]
    assert views["The alpha payload is ready."][0] != views["The alpha payload is ready."][1]
    assert metadata_by_target["Continuing."]["retained_event_ids"]
    assert metadata_by_target["Finished."]["expired_lease_event_ids"]


def test_paired_prepare_loader_integrity_and_grouping(tmp_path):
    output = tmp_path / "prepared"
    manifest = prepare_paired_corpus(
        [_recovery_row()],
        TOKENIZER,
        output,
        packing=_packing(),
        sampling=_sampling(),
        policy_config=_policy(),
        kv_bytes_per_token=1,
    )
    assert set(path.name for path in output.iterdir()) == {
        "manifest.json",
        "sessions.jsonl",
        "paired_decisions.jsonl",
    }
    assert manifest["counts"]["changed_view_pairs"] >= 2
    assert manifest["counts"]["retrieved_event_selections"] >= 1
    assert manifest["counts"]["retained_event_selections"] >= 1
    assert manifest["counts"]["released_event_selections"] >= 1
    assert manifest["counts"]["presented_tokens.arm.C"] <= 1_000_000
    assert manifest["counts"]["presented_tokens.arm.B"] <= 1_000_000
    assert set(manifest["file_integrity"]) == {"sessions.jsonl", "paired_decisions.jsonl"}
    assert manifest["policy"]["source_commit"] == "affe0e3bd29cce06beadd5a67b1e629f8ca77022"

    c = PreparedCorpus(output, TOKENIZER, "C")
    b = PreparedCorpus(output / "manifest.json", TOKENIZER, "B")
    assert len(c) == len(b) > 0
    assert c.decision_ids == b.decision_ids
    assert c.session_keys == b.session_keys
    assert c.group_ids == b.group_ids
    assert all(len(group) == 3 for group in c.group_ids)
    for index in range(len(c)):
        left, right = c[index], b[index]
        assert left.target_ids == right.target_ids
        assert left.ratio == right.ratio
        assert left.repetition_index == right.repetition_index
        assert left.weight == right.weight

    sessions = output / "sessions.jsonl"
    original = sessions.read_text(encoding="utf-8")
    assert "alpha payload" in original
    sessions.write_text(original.replace("alpha payload", "omega payload", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity check failed"):
        PreparedCorpus(output, TOKENIZER, "C")


def test_qa_mix_is_selected_before_expansion_and_reported_after_packing(tmp_path):
    rows = []
    for index in range(17):
        rows.append(
            {
                "session_id": f"agent-{index}",
                "source": "agent",
                "split": "train",
                "task_id": f"agent-{index}",
                "template_id": "simple",
                "messages": [
                    {"role": "user", "content": f"Question {index}"},
                    {"role": "assistant", "content": f"Answer {index}"},
                ],
                "tools": [],
            }
        )
    for index in range(10):
        rows.append(
            {
                "session_id": f"qa-{index}",
                "source": "qa:hotpotqa",
                "split": "train",
                "task_id": f"qa-{index}",
                "template_id": "qa",
                "messages": [
                    {"role": "user", "content": f"Document {index}"},
                    {"role": "user", "content": f"Question {index}"},
                    {"role": "assistant", "content": f"Answer {index}"},
                ],
                "tools": [],
            }
        )
    manifest = prepare_paired_corpus(
        rows,
        TOKENIZER,
        tmp_path / "mix",
        packing=_packing(ratios=(8,)),
        sampling=_sampling(qa_target_fraction=0.15),
        policy_config=_policy(),
        kv_bytes_per_token=1,
        allow_unchanged_b=True,
    )
    assert manifest["counts"]["base_decisions_written"] == 20
    assert manifest["counts"]["base_decisions.source.qa:hotpotqa"] == 3
    assert manifest["statistics"]["actual_qa_base_decision_fraction"] == pytest.approx(0.15)


def test_pair_is_skipped_together_when_one_complete_target_exceeds_cap(tmp_path):
    row = {
        "session_id": "target-caps",
        "source": "agent",
        "split": "train",
        "task_id": "target-caps",
        "template_id": "caps",
        "messages": [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "x" * 300},
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "ok"},
        ],
        "tools": [],
    }
    manifest = prepare_paired_corpus(
        [row],
        TOKENIZER,
        tmp_path / "caps",
        packing=_packing(ratios=(8,), max_target_tokens=80),
        sampling=_sampling(),
        policy_config=_policy(),
        kv_bytes_per_token=1,
        allow_unchanged_b=True,
    )
    assert manifest["counts"]["decision_pairs_skipped.PackingBudgetError"] == 1
    corpus_c = PreparedCorpus(tmp_path / "caps", TOKENIZER, "C")
    corpus_b = PreparedCorpus(tmp_path / "caps", TOKENIZER, "B")
    assert len(corpus_c) == len(corpus_b) == 1
    assert corpus_c[0].target_ids == corpus_b[0].target_ids


def test_presented_token_cap_is_per_arm_and_skips_whole_pairs(tmp_path):
    manifest = prepare_paired_corpus(
        [_recovery_row("token-cap")],
        TOKENIZER,
        tmp_path / "token-cap",
        packing=_packing(ratios=(8,)),
        sampling=_sampling(max_presented_tokens_per_arm=1),
        policy_config=_policy(),
        kv_bytes_per_token=1,
        allow_unchanged_b=True,
    )
    assert manifest["counts"]["decision_pairs_skipped_presented_token_cap"] > 0
    assert manifest["counts"]["presented_tokens.arm.C"] <= 1
    assert manifest["counts"]["presented_tokens.arm.B"] <= 1
    assert len(PreparedCorpus(tmp_path / "token-cap", TOKENIZER, "C")) == len(
        PreparedCorpus(tmp_path / "token-cap", TOKENIZER, "B")
    )


def test_formal_prepare_rejects_changed_b_with_zero_gist_exposure(tmp_path):
    row = {
        "session_id": "all-restored",
        "source": "qa:synthetic",
        "split": "train",
        "task_id": "all-restored",
        "template_id": "all-restored",
        "messages": [
            {"role": "user", "content": "alpha.txt contains the evidence"},
            {"role": "user", "content": "Use alpha.txt evidence"},
            {"role": "assistant", "content": "alpha evidence"},
        ],
        "tools": [],
    }
    with pytest.raises(ValueError, match="zero gist tokens"):
        prepare_paired_corpus(
            [row],
            TOKENIZER,
            tmp_path / "zero-gist",
            packing=_packing(ratios=(8,)),
            sampling=_sampling(),
            policy_config=_policy(),
            kv_bytes_per_token=1,
        )
