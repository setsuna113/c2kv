"""Source-bounded tool context: exact producer spans and OFF identity."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_annotations
import toolmemory
from test_toolmemory import FakeTokenizer, fake_extract, memory, write_t0_checkpoint


def test_acebench_annotations_point_only_to_rendered_functions():
    functions = [{"name": "search", "description": "Find records"},
                 {"name": "update", "description": "Update records"}]
    content = "Keep the official action syntax.\nAPIs:\n" + json.dumps(functions, ensure_ascii=False)
    messages = [{"role": "system", "content": content},
                {"role": "user", "content": "search first"}]
    spans = source_annotations.acebench_function_spans(messages, functions)
    assert len(spans) == 2
    assert [content[s["start"]:s["end"]] for s in spans] == [
        json.dumps(function, ensure_ascii=False) for function in functions]
    assert content[:spans[0]["start"]].startswith("Keep the official action syntax")
    resolved = toolmemory.resolve_visible_tool_spans({"messages": messages,
                                                      toolmemory.TOOL_SPANS_FIELD: spans})
    assert [toolmemory.visible_tool_snapshot(span) for span in resolved] == functions


def test_appworld_annotations_require_actual_pure_docs_output():
    code = "print(apis.api_docs.show_api_doc(app_name='music', api_name='search'))"
    assert source_annotations.pure_appworld_doc_action(code) == "appworld.api_docs.show_api_doc"
    assert source_annotations.pure_appworld_doc_action(
        code + "\nprint(apis.music.search())") is None
    output = "{'api_name': 'search', 'parameters': []}"
    messages = [{"role": "system", "content": "Quoted example: " + output},
                {"role": "user", "content": "Find an API."},
                {"role": "assistant", "content": code + "\n# " + output},
                {"role": "user", "content": "Earlier context\n" + output + "\nNext action as Python."}]
    spans = source_annotations.appworld_doc_spans(
        messages, [("appworld.api_docs.show_api_doc", output)])
    assert len(spans) == 1
    span = spans[0]
    assert span["message_index"] == 3
    assert messages[3]["content"][span["start"]:span["end"]] == output
    assert source_annotations.appworld_doc_spans(messages, []) == []


def test_off_annotation_strip_is_identity_and_no_catalog_is_noop(tmp_path):
    payload = {"messages": [{"role": "user", "content": "Run Python code."}],
               "model": "agent", "temperature": 0}
    before = copy.deepcopy(payload)
    assert toolmemory.strip_request_annotations(payload) is payload
    assert payload == before
    assert toolmemory.tool_visibility_status(payload) == "no_visible_definition"
    mem = memory(tmp_path)
    assert mem.plan(payload) is None
    assert mem.stats["skipped_no_tools"] == 1


def test_t0_replaces_only_visible_source_spans_and_keeps_action_dialect(tmp_path):
    doc = "{'api_name': 'search', 'parameters': [{'name': 'query'}]}"
    content = "Use Python statements only.\nObserved docs:\n" + doc + "\nReturn one code block."
    start = content.index(doc)
    payload = {"model": "agent", "messages": [
        {"role": "system", "content": "Generate Python code."},
        {"role": "user", "content": content},
    ], toolmemory.TOOL_SPANS_FIELD: [{"message_index": 1, "start": start,
                                      "end": start + len(doc),
                                      "source": "appworld.api_docs.show_api_doc"}]}
    mem = memory(tmp_path)
    plan = mem.plan(payload)
    assert plan is not None
    assert plan.protocol == ""
    assert toolmemory.with_protocol_system(plan.messages, plan.protocol) == plan.messages
    assert plan.messages[0] == payload["messages"][0]
    placeholder = toolmemory.source_span_placeholder(plan.source_spans[0])
    assert plan.messages[1]["content"] == (
        content[:start] + placeholder + content[start + len(doc):])
    assert plan.info["action_protocol"] == "benchmark_original"
    assert plan.info["n_visible_source_spans"] == 1
    assert plan.info["carrier_anchors"][0]["message_index"] == 1
    assert plan.info["carrier_anchors"][0]["placeholder"] == placeholder
    assert plan.chunks[0].catalog_index == 0
    assert plan.carriers()[0][toolmemory.CARRIER_MARK]["anchor"]["source"] == (
        "appworld.api_docs.show_api_doc")
    forwarded = mem.stage_request(payload, plan)
    assert toolmemory.TOOL_SPANS_FIELD not in forwarded
    assert "c2kv_tools_in_prompt" not in forwarded
    assert payload["messages"][1]["content"] == content


def test_raw_kv_plan_keeps_messages_until_query_bearing_final_assembly():
    content = "API: do_search(query)\nPlease search now."
    payload = {"messages": [{"role": "user", "content": content}],
               toolmemory.TOOL_SPANS_FIELD: [{"message_index": 0, "start": 5,
                                               "end": 21, "source": "acebench.functions"}]}
    spec = toolmemory.parse_tool_memory_spec("h2o:r8")
    plan = toolmemory.plan_visible_tool_memory(payload, spec)
    assert plan.messages == payload["messages"]
    assert not plan.chunks
    assert plan.info["representation"] == "selected_raw_kv"
    assert plan.info["compressed_source_indices"] == [0]
    assert toolmemory.strip_request_annotations(payload) == {"messages": payload["messages"]}
    assert toolmemory.parse_tool_memory_spec("snapkv:r12:hybrid1").encoder == "snapkv"


def test_invalid_or_overlapping_source_intervals_fail_closed():
    payload = {"messages": [{"role": "user", "content": "abcdef"}],
               toolmemory.TOOL_SPANS_FIELD: [
                   {"message_index": 0, "start": 1, "end": 4, "source": "producer"},
                   {"message_index": 0, "start": 3, "end": 5, "source": "producer"},
               ]}
    with pytest.raises(toolmemory.ToolMemoryError, match="overlap"):
        toolmemory.resolve_visible_tool_spans(payload)


def test_dynamic_doc_carriers_append_at_source_and_shift_history_ledger():
    def carrier(key, message_index):
        anchor = {"identity": key, "message_index": message_index}
        return {"role": "user", "content": "", "c2kv_key_hash": key,
                toolmemory.CARRIER_MARK: {"event_id": key, "part_index": 0,
                                           "anchor": anchor}}

    initial = [{"role": "system", "content": "Python action syntax"},
               {"role": "user", "content": "doc A placeholder"},
               {"role": "user", "content": "query one"}]
    def events(messages):
        return [{"message_index": index, "role": message["role"],
                 "phase": "others"} for index, message in enumerate(messages)]
    first, first_counts = toolmemory.insert_carriers(
        initial, {"current_start_out_index": 2,
                  "history_kv_event_messages": events(initial)},
        [carrier("doc-a", 1)], source_out_indices={1: 1})
    assert [message.get("c2kv_key_hash") for message in first] == [
        None, None, "doc-a", None]
    assert first_counts["current_start_out_index"] == 3
    assert [item["message_index"] for item in first_counts["history_kv_event_messages"]] == list(range(4))
    assert first_counts["history_kv_event_messages"][2]["phase"] == "others"

    # The next decision retains the old doc and receives a new official doc.
    second_input = first + [{"role": "tool", "content": "doc B placeholder"},
                            {"role": "user", "content": "query two"}]
    second, second_counts = toolmemory.insert_carriers(
        second_input, {"current_start_out_index": 5,
                       "history_kv_event_messages": events(second_input)},
        [carrier("doc-a", 1), carrier("doc-b", 3)],
        source_out_indices={1: 1, 3: 4})
    assert [message.get("c2kv_key_hash") for message in second] == [
        None, None, "doc-a", None, None, "doc-b", None]
    assert second_counts["current_start_out_index"] == 6
    assert [item["message_index"] for item in second_counts["history_kv_event_messages"]] == list(range(7))
    assert second_counts["history_kv_event_messages"][5]["phase"] == "others"


def test_source_carrier_requires_assembly_mapping():
    carrier = {"role": "user", "content": "", "c2kv_key_hash": "k",
               toolmemory.CARRIER_MARK: {"event_id": "e", "part_index": 0,
                                          "anchor": {"identity": "a", "message_index": 2}}}
    with pytest.raises(toolmemory.ToolMemoryError, match="tool_anchor"):
        toolmemory.insert_carriers([{"role": "system", "content": "s"}], {}, [carrier])


def test_explicit_t0_budget_checks_total_slots_before_extraction(tmp_path):
    spec = toolmemory.parse_tool_memory_spec("t0:r8")
    payload = {"messages": [{"role": "user", "content": "Call search"}],
               "tools": [{"type": "function", "function": {"name": "search"}}]}
    pure = toolmemory.plan_visible_tool_memory(payload, spec, FakeTokenizer())
    assert pure.info["resident_tool_tokens"] == (
        pure.info["protocol_prefix_tokens"] + pure.info["expected_gist_tokens"])
    with pytest.raises(toolmemory.ToolMemoryError, match="tool_budget"):
        toolmemory.enforce_tool_budget(pure, pure.info["resident_tool_tokens"] - 1)
    checkpoint = write_t0_checkpoint(tmp_path / "ckpt")
    calls = []
    def observed_extract(*args):
        calls.append(args)
        return fake_extract(*args)
    mem = toolmemory.ToolMemory(spec, checkpoint, observed_extract,
                                tokenizer=FakeTokenizer(),
                                budget_tokens=pure.info["resident_tool_tokens"] - 1)
    with pytest.raises(toolmemory.ToolMemoryError, match="tool_budget"):
        mem.plan(payload)
    assert calls == []
