from __future__ import annotations

import json
import importlib.util
import sys
import types
from pathlib import Path

import pytest

from history_memory.dataset import iter_decisions
from next_compression.common import sha256_file
from next_compression.tools import (
    TOOL_PROTOCOL_HEAD,
    TOOL_PROTOCOL_TAIL,
    ToolPreparationConfig,
    iter_tool_records,
    pack_tool_memory,
    prepare_tools,
    restore_tool_definitions,
    split_tool_definitions,
    tool_variant_material,
)


class ByteChatTokenizer:
    """A reversible template with the prefix boundaries required by packing."""

    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        truncation=False,
    ):
        assert tokenize is True
        assert enable_thinking is False
        assert truncation is False
        self.calls.append(json.loads(json.dumps({"messages": messages, "tools": tools})))
        parts = []
        if tools is not None:
            parts.append(
                "<tools>"
                + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
                + "</tools>"
            )
        for message in messages:
            role = message["role"]
            if role == "assistant":
                payload = {key: value for key, value in message.items() if key != "role"}
                parts.append(
                    "<assistant>"
                    + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    + "<eos>"
                )
            else:
                parts.append(
                    f"<{role}>"
                    + json.dumps(
                        {key: value for key, value in message.items() if key != "role"},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + f"</{role}>"
                )
        if add_generation_prompt:
            parts.append("<assistant>")
        return [byte + 1 for byte in "".join(parts).encode("utf-8")]

    @staticmethod
    def decode(ids) -> str:
        return bytes(int(token) - 1 for token in ids).decode("utf-8")

    def __len__(self) -> int:
        return 257


def _standard_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "description": "Look up exact weather observations.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The requested city.",
                            "minLength": 2,
                        },
                        "units": {
                            "type": "string",
                            "enum": ["c", "f"],
                        },
                    },
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "computer_use_preview",
            "description": "Operate the visible computer.",
            "display_width": 1280,
            "display_height": 720,
        },
    ]


def _row(*, final_text: bool = False):
    messages = [
        {"role": "system", "content": "SYSTEM_INSTRUCTIONS"},
        {"role": "user", "content": "FIRST_QUERY"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup_weather",
                        "arguments": '{"city":"Oxford"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "OLD_RESULT"},
        {"role": "user", "content": "CURRENT_QUERY"},
        {
            "role": "assistant",
            "content": "Done" if final_text else None,
            **(
                {}
                if final_text
                else {
                    "tool_calls": [
                        {
                            "id": "call-2",
                            "type": "function",
                            "function": {
                                "name": "lookup_weather",
                                "arguments": '{"city":"TARGET_SECRET_ARGUMENT","units":"c"}',
                            },
                        }
                    ]
                }
            ),
        },
    ]
    return {
        "session_id": "session-1",
        "source": "fixture",
        "split": "train",
        "task_id": "task-1",
        "template_id": "template-1",
        "messages": messages,
        "tools": _standard_tools(),
    }


def _memory_text(tokenizer, record):
    memory = record["memory"]
    arrays = [memory["system_input_ids"], memory["workspace_input_ids"]]
    arrays.extend(chunk["token_ids"] for chunk in memory["chunks"])
    return "\n".join(tokenizer.decode(tokens) for tokens in arrays)


def _system_content(tokenizer, record):
    rendered = tokenizer.decode(record["memory"]["system_input_ids"])
    assert rendered.startswith("<system>") and rendered.endswith("</system>")
    payload = rendered[len("<system>") : -len("</system>")]
    return json.loads(payload)["content"]


def test_matched_variants_keep_full_targets_and_do_not_encode_target():
    tokenizer = ByteChatTokenizer()
    row = _row()
    records, audit = prepare_tools([row], tokenizer)

    # Two assistant decisions, each represented at both fixed ratios.
    assert len(records["T0"]) == len(records["T1"]) == 4
    t0_keys = [(record["decision_id"], record["ratio"]) for record in records["T0"]]
    t1_keys = [(record["decision_id"], record["ratio"]) for record in records["T1"]]
    assert t0_keys == t1_keys
    assert [record["target_ids"] for record in records["T0"]] == [
        record["target_ids"] for record in records["T1"]
    ]

    final_t0 = records["T0"][-1]
    final_t1 = records["T1"][-1]
    target_text = tokenizer.decode(final_t0["target_ids"])
    assert "lookup_weather" in target_text
    assert "TARGET_SECRET_ARGUMENT" in target_text
    assert target_text.endswith("<eos>")
    for record in (final_t0, final_t1):
        encoder_text = _memory_text(tokenizer, record)
        assert "TARGET_SECRET_ARGUMENT" not in encoder_text
        assert "SYSTEM_INSTRUCTIONS" in encoder_text
        assert "OLD_RESULT" in encoder_text
        assert "CURRENT_QUERY" in encoder_text

    t0_text = _memory_text(tokenizer, final_t0)
    t1_text = _memory_text(tokenizer, final_t1)
    assert "Look up exact weather observations." in t0_text
    assert "The requested city." in t0_text
    assert "Look up exact weather observations." in t1_text
    assert "The requested city." in t1_text
    # T1 keeps constraints and type information in the native tools prefix.
    t1_system = _system_content(tokenizer, final_t1)
    assert '"type":"computer_use_preview"' in t1_system
    assert '"required":["city"]' in t1_system
    assert '"enum":["c","f"]' in t1_system
    assert '"additionalProperties":false' in t1_system
    assert "Look up exact weather observations." not in t1_system
    # T0 has no native tools prefix; every full definition is an independent doc.
    t0_system = _system_content(tokenizer, final_t0)
    assert "lookup_weather" not in t0_system
    assert "<tools>\n</tools>" in t0_system
    # The generic raw tool-call protocol is identical across variants. Only
    # T1's native schema payload differs inside the tool tags.
    for system_text in (t0_system, t1_system):
        assert TOOL_PROTOCOL_HEAD in system_text
        assert TOOL_PROTOCOL_TAIL in system_text
    assert len({chunk["event_id"] for chunk in final_t0["memory"]["chunks"]}) == 2
    assert audit["decisions.accepted"] == 2
    assert audit["records.written.variant.T0.ratio.8"] == 2
    assert audit["records.written.variant.T1.ratio.12"] == 2


def test_t1_split_is_lossless_and_stable_for_nonstandard_schemas():
    tools = [
        {
            "name": "odd_search",
            "description": "Search a nonstandard backend.",
            "input_schema": {
                "type": "object",
                "properties": {
                    # A parameter literally named description must remain.
                    "description": {
                        "type": "string",
                        "description": "A query field whose name is description.",
                        "pattern": "^[a-z]+$",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    },
                },
                "required": ["description"],
                "oneOf": [{"required": ["limit"]}, {"not": {"required": ["limit"]}}],
                "x-provider-constraint": {"mode": "strict"},
            },
        }
    ]
    first = split_tool_definitions(tools)
    second = split_tool_definitions(tools)
    assert first == second
    native, supplements = first
    assert restore_tool_definitions(native, supplements) == tuple(tools)
    schema = native[0]["input_schema"]
    assert schema["properties"]["description"]["type"] == "string"
    assert schema["properties"]["description"]["pattern"] == "^[a-z]+$"
    assert schema["required"] == ["description"]
    assert schema["oneOf"]
    assert schema["x-provider-constraint"] == {"mode": "strict"}
    assert "description" not in native[0]
    assert "description" not in schema["properties"]["description"]
    assert [item.as_document() for item in supplements] == [
        item.as_document() for item in second[1]
    ]


def test_joint_budget_rejection_never_leaves_an_unmatched_variant():
    row = _row()
    # A large structural enum stays in T1's native prefix, while it also makes
    # the complete T0 definition document exceed this raw tool-token limit.
    row["tools"][0]["function"]["parameters"]["properties"]["city"]["enum"] = [
        f"city-{index}" for index in range(200)
    ]
    tokenizer = ByteChatTokenizer()
    config = ToolPreparationConfig(max_tool_tokens=200)
    stream, audit = iter_tool_records([row], tokenizer, config)
    assert list(stream) == []
    assert audit["decisions.skipped"] == 2
    assert audit["decisions.skipped.reason.tool_tokens_over_limit"] == 2
    assert audit["records.written.variant.T0"] == 0
    assert audit["records.written.variant.T1"] == 0


def test_non_tool_response_and_source_grounded_terminal_stop_are_audited():
    tokenizer = ByteChatTokenizer()
    records, audit = prepare_tools([_row(final_text=True)], tokenizer)
    assert len(records["T0"]) == len(records["T1"]) == 4
    assert audit["decisions.accepted.kind.covered_tool_call"] == 1
    assert audit["decisions.accepted.kind.non_tool_response"] == 1
    assert audit["decisions.source_grounded_terminal_stop"] == 1
    assert audit["decisions.source_grounded_terminal_stop.source.fixture"] == 1
    terminal = records["T0"][-1]
    assert terminal["metadata"]["source_grounded_terminal_stop"] is True
    assert tokenizer.decode(terminal["target_ids"]).endswith("<eos>")


def test_single_turn_no_history_is_kept_and_audited():
    row = _row()
    row["messages"] = [row["messages"][0], row["messages"][4], row["messages"][5]]
    tokenizer = ByteChatTokenizer()
    records, audit = prepare_tools([row], tokenizer)
    assert len(records["T0"]) == len(records["T1"]) == 2
    assert audit["decisions.no_prior_history"] == 1
    assert audit["decisions.no_prior_history.source.fixture"] == 1


def test_pack_helper_matches_preparation_and_limits_reject_without_slicing():
    tokenizer = ByteChatTokenizer()
    decision = list(iter_decisions(_row()))[-1]
    config = ToolPreparationConfig(max_target_tokens=1)
    records, audit = prepare_tools([_row()], tokenizer, config)
    assert records == {"T0": [], "T1": []}
    assert audit["decisions.skipped.reason.target_tokens_over_limit"] == 2

    usable = ToolPreparationConfig()
    direct = pack_tool_memory(
        decision.store, decision.tools, tokenizer, variant="T1", config=usable
    )
    records, _ = prepare_tools([_row()], tokenizer, usable)
    prepared = records["T1"][-1]["memory"]
    assert tuple(prepared["system_input_ids"]) == direct.system_input_ids
    assert tuple(prepared["workspace_input_ids"]) == direct.workspace_input_ids
    assert [tuple(chunk["token_ids"]) for chunk in prepared["chunks"]] == [
        chunk.token_ids for chunk in direct.chunks
    ]


def test_total_sequence_limit_rejects_the_matched_pair_without_slicing():
    tokenizer = ByteChatTokenizer()
    records, audit = prepare_tools(
        [_row()], tokenizer, ToolPreparationConfig(max_sequence_tokens=1)
    )
    assert records == {"T0": [], "T1": []}
    assert (
        audit["decisions.skipped.reason.sequence_tokens_over_limit.T0.ratio8"]
        == 2
    )


def test_per_session_cap_is_chronological_and_audited_without_target_ranking():
    tokenizer = ByteChatTokenizer()
    records, audit = prepare_tools(
        [_row()], tokenizer, ToolPreparationConfig(max_decisions_per_session=1)
    )
    assert len(records["T0"]) == len(records["T1"]) == 2
    assert audit["decisions.candidates"] == 2
    assert audit["decisions.observed"] == 1
    assert audit["decisions.skipped.reason.per_session_limit"] == 1
    assert records["T0"][0]["metadata"]["source_message_index"] == 2
    assert records["T0"][0]["metadata"]["limits"]["max_decisions_per_session"] == 1


def test_t1_rejects_schema_only_tools_jointly_because_no_gist_would_train():
    row = _row()
    row["tools"] = [
        {
            "type": "function",
            "function": {
                "name": "lookup_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    tokenizer = ByteChatTokenizer()
    records, audit = prepare_tools([row], tokenizer)
    assert records == {"T0": [], "T1": []}
    assert audit["decisions.skipped.reason.no_compressible_tool_information"] == 2


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown tool variant"):
        tool_variant_material(_standard_tools(), "T2")


def _write_normalized_source(path: Path, source: str, count: int):
    rows = []
    for index in range(count):
        row = _row()
        row["session_id"] = f"{source}-session-{index}"
        row["source"] = source
        row["task_id"] = f"{source}-task-{index}"
        row["messages"] = [row["messages"][0], row["messages"][4], row["messages"][5]]
        rows.append(row)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "sessions": len(rows),
    }


def test_streaming_cli_round_robins_sources_and_freezes_pair_budget(
    tmp_path, monkeypatch
):
    cli_path = Path(__file__).resolve().parents[2] / "agent" / "prepare_next_tools.py"
    spec = importlib.util.spec_from_file_location("prepare_next_tools_cli", cli_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    source_a = tmp_path / "a.jsonl"
    source_b = tmp_path / "b.jsonl"
    manifest = {
        "schema": cli.NORMALIZED_SCHEMA,
        "sources": {
            "a_src": _write_normalized_source(source_a, "a", 2),
            "b_src": _write_normalized_source(source_b, "b", 2),
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    tokenizer = ByteChatTokenizer()

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(path, local_files_only=True):
            assert path == "fake-tokenizer"
            assert local_files_only is True
            return tokenizer

    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer)
    )
    full_output = tmp_path / "full"
    result = cli.prepare(
        cli.arguments(
            [
                "--normalized-manifest",
                str(manifest_path),
                "--tokenizer",
                "fake-tokenizer",
                "--output-dir",
                str(full_output),
                "--sources",
                "a_src",
                "b_src",
                "--workers",
                "1",
                "--executor",
                "thread",
                "--batch-sessions",
                "2",
            ]
        )
    )
    assert result["records_per_variant"] == {"T0": 8, "T1": 8}
    assert result["audit"]["records.written.variant.T0"] == 8

    train_path = Path(__file__).resolve().parents[2] / "agent" / "train_next_compression.py"
    train_spec = importlib.util.spec_from_file_location(
        "train_next_compression_for_tool_test", train_path
    )
    assert train_spec is not None and train_spec.loader is not None
    train_cli = importlib.util.module_from_spec(train_spec)
    train_spec.loader.exec_module(train_cli)
    corpus, _, render_profile, loss_profile = train_cli.load_corpus(
        types.SimpleNamespace(
            cpu_smoke=False, data_path=str(full_output / "T0"), variant="T0"
        ),
        tokenizer,
    )
    assert len(corpus) == 8
    assert render_profile == cli.TOOL_PACKING_PROFILE
    assert loss_profile == "decision-mean-complete-ce-v1"

    def read_records(output, variant):
        return [
            json.loads(line)
            for line in (output / variant / "records.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    t0 = read_records(full_output, "T0")
    t1 = read_records(full_output, "T1")
    assert [record["source"] for record in t0[::2]] == ["a", "b", "a", "b"]
    assert [record["decision_id"] for record in t0] == [
        record["decision_id"] for record in t1
    ]
    first_pair_tokens = sum(
        record["metadata"]["token_counts"]["presented_encoder_tokens"]
        for record in t0[:2]
    )

    bounded_output = tmp_path / "bounded"
    bounded = cli.prepare(
        cli.arguments(
            [
                "--normalized-manifest",
                str(manifest_path),
                "--tokenizer",
                "fake-tokenizer",
                "--output-dir",
                str(bounded_output),
                "--sources",
                "a_src",
                "b_src",
                "--max-presented-tokens",
                str(first_pair_tokens),
                "--workers",
                "1",
                "--executor",
                "thread",
                "--batch-sessions",
                "2",
            ]
        )
    )
    assert bounded["records_per_variant"] == {"T0": 2, "T1": 2}
    assert bounded["audit"]["decisions.selected"] == 1
    assert bounded["audit"]["budget.rejected_matched_decisions"] == 1
    bounded_t0 = read_records(bounded_output, "T0")
    bounded_t1 = read_records(bounded_output, "T1")
    assert [record["decision_id"] for record in bounded_t0] == [
        record["decision_id"] for record in bounded_t1
    ]
