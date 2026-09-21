"""Schema split keeps execution data once and encodes descriptive prose only."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toolinterface  # noqa: E402
import toolmemory  # noqa: E402
from test_toolmemory import TOOLS, memory  # noqa: E402


@pytest.mark.parametrize("encoder", toolmemory.ENCODERS)
def test_schema_profile_is_versioned_and_default_is_untouched(encoder):
    default = toolmemory.parse_tool_memory_spec(f"{encoder}:r8:hybrid1")
    split = toolmemory.parse_tool_memory_spec(f"{encoder}:r8:hybrid1:schema")
    assert default.name + "_schema" == split.name
    assert "interface_render_profile" not in default.as_dict()
    assert "description_document_profile" not in default.as_dict()
    assert split.as_dict()["interface_render_profile"] == "tool-schema-split-v3"
    assert split.as_dict()["description_document_profile"] == "tool-description-only-v1"


def test_schema_walk_retains_literals_and_binds_only_prose_to_compact_tree():
    schema = {"type": "object", "title": "Input", "description": "Choose carefully",
              "examples": [{"description": "literal example"}], "properties": {
                  "description": {"type": "string", "description": "User message"},
                  "title": {"type": "string", "default": {"description": "literal default"}},
                  "mode": {"enum": [{"description": "literal enum"}],
                           "const": {"title": "literal const"}},
                  "nested": {"$ref": "#/$defs/Nested"}},
              "required": ["description", "title"], "$defs": {"Nested": {
                  "type": "object", "properties": {"value": {
                      "type": "integer", "description": "Nested meaning"}}}},
              "x-executable": {"description": "extension literal"}}
    tool = {"type": "function", "function": {"name": "act", "description": "Do work",
                                           "strict": True, "parameters": schema}}
    original = copy.deepcopy(tool)
    compact = toolinterface.compact_tool(tool)
    document = toolinterface.description_document(tool, 7)
    assert tool == original
    assert compact["function"]["name"] == "act"
    assert compact["function"]["strict"] is True
    assert "description" not in compact["function"]
    params = compact["function"]["parameters"]
    assert params["required"] == schema["required"]
    assert set(params["properties"]) == set(schema["properties"])
    assert params["properties"]["title"]["default"] == {"description": "literal default"}
    assert params["properties"]["mode"]["enum"] == schema["properties"]["mode"]["enum"]
    assert params["properties"]["mode"]["const"] == schema["properties"]["mode"]["const"]
    assert params["examples"] == schema["examples"]
    assert params["x-executable"] == schema["x-executable"]
    assert document["tool_index"] == 7
    assert {item["text"] for item in document["annotations"]} == {
        "Do work", "Input", "Choose carefully", "User message", "Nested meaning"}
    assert all(all(isinstance(step, int) for step in item["address"])
               for item in document["annotations"])
    assert "act" not in json.dumps(document)
    assert "literal default" not in json.dumps(document)
    # An address is resolved through the retained compact tree, after prose
    # fields were removed; its ordinal never depends on removed siblings.
    for item in document["annotations"]:
        node = compact
        for step in item["address"]:
            node = list(node.values())[step] if isinstance(node, dict) else node[step]
        assert item["field"] not in node


def test_json_span_scanner_distinguishes_identical_annotation_and_literal_values():
    tool = {"name": "act", "description": "first", "parameters": {"type": "object",
            "properties": {"description": {"type": "string", "description": "first"},
                           "mode": {"default": {"description": "first"}}}}}
    text = json.dumps(tool, indent=2)
    paths = {tuple(item["path"]) for item in toolinterface.tool_prose(tool)}
    spans = toolinterface.json_string_value_spans(text, paths)
    assert len(spans) == 2
    assert all(text[span["start"]:span["end"]] == '"first"' for span in spans)
    assert {tuple(span["path"]) for span in spans} == {
        ("description",), ("parameters", "properties", "description", "description")}


def test_t0_split_uses_prose_documents_and_native_once(tmp_path):
    payload = {"messages": [{"role": "user", "content": "get_weather Paris"}],
               "tools": copy.deepcopy(TOOLS)}
    baseline = memory(tmp_path / "baseline", "t0:r8:hybrid1").plan(payload)
    owner = memory(tmp_path / "split", "t0:r8:hybrid1:schema")
    split = owner.plan(payload)
    assert payload["tools"] == TOOLS
    assert split.info["native_indices"] == baseline.info["native_indices"] == [2]
    assert split.info["n_documents"] == 2
    assert split.info["n_protected_interfaces"] == 2
    assert {span["catalog_index"] for span in split.interface_spans} == {0, 1}
    assert split.protocol.count('"name":"get_weather"') == 1
    assert split.protocol.count('"name":"book_flight"') == 1
    assert "Book a flight" not in split.protocol
    assert split.chunks != baseline.chunks
    assert all(chunk.catalog_index in {0, 1} for chunk in split.chunks)
    assert "[tool_index:0]" in split.protocol
    assert "[tool_index:1]" in split.protocol
    for span in split.interface_spans:
        assert split.messages[span["message_index"]]["content"][
            span["start"]:span["end"]] == span["text"]
    owner.plan(payload)
    assert owner.stats["chunk_cache_hits"] == len(split.chunks)


def test_no_prose_makes_no_t0_document_or_gist(tmp_path):
    tool = {"type": "function", "function": {"name": "act", "parameters": {
        "type": "object", "properties": {"x": {"type": "integer", "default": 3}}}}}
    assert toolmemory.description_documents([tool], [0]) == ()
    plan = memory(tmp_path, "t0:r8:schema").plan({
        "messages": [{"role": "user", "content": "act"}], "tools": [tool]})
    assert plan.info["n_documents"] == 0
    assert plan.chunks == []
    assert plan.records == []
    assert plan.info["gist_tokens"] == 0
    assert plan.protocol.count('"name":"act"') == 1


def test_blank_and_non_string_annotations_stay_raw_without_empty_gist(tmp_path):
    tool = {"type": "function", "title": {"literal": 1}, "function": {
        "name": "act", "description": "  ", "parameters": {
            "type": "object", "description": "\n", "properties": {
                "x": {"type": "integer", "title": ["literal"]}}}}}
    assert toolinterface.compact_tool(tool) == tool
    assert toolinterface.description_document(tool, 0) is None
    plan = memory(tmp_path, "t0:r8:schema").plan({
        "messages": [{"role": "user", "content": "act"}], "tools": [tool]})
    assert plan.chunks == []
    assert plan.records == []


def test_source_without_prose_moves_once_without_placeholder_or_gist(tmp_path):
    doc = json.dumps({"name": "lookup", "parameters": {"type": "object",
                      "properties": {"id": {"type": "integer"}}}})
    plan = memory(tmp_path, "t0:r8:schema").plan(_source_payload(doc))
    assert plan.info["n_documents"] == 0
    assert plan.chunks == []
    assert plan.protocol.count('"name":"lookup"') == 1
    assert "[Tool definition" not in plan.messages[1]["content"]
    assert doc not in plan.messages[1]["content"]


def test_opaque_structured_tool_is_full_once_and_not_encoded(tmp_path):
    tool = {"api_name": "opaque_api", "description": "Producer-specific prose",
            "parameters": [{"name": "id", "type": "integer"}]}
    plan = memory(tmp_path, "t0:r8:schema").plan({
        "messages": [{"role": "user", "content": "opaque_api"}], "tools": [tool]})
    assert plan.info["n_documents"] == 0
    assert plan.info["n_interface_fallbacks"] == 1
    assert plan.info["interface_fallback_indices"] == [0]
    assert plan.protocol.count('"api_name":"opaque_api"') == 1
    assert plan.chunks == []


def _source_payload(*docs):
    content = "Before\n" + "\nBetween\n".join(docs) + "\nAfter"
    annotations = []
    cursor = 0
    for doc in docs:
        start = content.index(doc, cursor)
        annotations.append({"message_index": 0, "start": start,
                            "end": start + len(doc), "source": "producer"})
        cursor = start + len(doc)
    return {"messages": [{"role": "user", "content": content}],
            toolmemory.TOOL_SPANS_FIELD: annotations}


def test_source_moves_once_and_opaque_fallback_has_no_gist(tmp_path):
    known = json.dumps({"name": "search", "description": "long prose", "parameters": {
        "type": "object", "properties": {"q": {"type": "string",
                                               "description": "query prose"}}}})
    opaque = "search(query: str) -> records; execute with Python"
    plan = memory(tmp_path, "t0:r8:schema").plan(_source_payload(known, opaque))
    assert plan.info["native_indices"] == [1]
    assert plan.info["n_interface_fallbacks"] == 1
    assert plan.info["interface_fallback_indices"] == [1]
    assert plan.info["n_documents"] == 1
    assert {chunk.catalog_index for chunk in plan.chunks} == {0}
    assert known not in plan.protocol
    assert "long prose" not in plan.protocol
    assert plan.protocol.count(opaque) == 1
    assert known not in plan.messages[1]["content"]
    assert opaque not in plan.messages[1]["content"]
    assert "Before\n" in plan.messages[1]["content"]
    assert "\nBetween\n" in plan.messages[1]["content"]
    assert plan.messages[1]["content"].count("[Tool definition") == 1


@pytest.mark.parametrize("encoder", ["t0", "h2o"])
def test_native_source_is_full_once_after_relocation(tmp_path, encoder):
    doc = json.dumps({"name": "lookup", "description": "Execution detail",
                      "parameters": {"type": "object", "properties": {
                          "query": {"type": "string"}}}})
    owner = memory(tmp_path / encoder, f"{encoder}:r8:hybrid1:schema")
    plan = owner.plan(_source_payload(doc))
    assert plan.info["native_indices"] == [0]
    assert plan.protocol.count(doc) == 1
    assert doc not in plan.messages[1]["content"]
    assert plan.interface_spans[0]["copy_reason"] == "native_source_full"
    assert plan.chunks == []


@pytest.mark.parametrize("method", ["streamingllm", "h2o", "snapkv", "pyramidkv"])
def test_raw_structured_once_and_only_prose_value_spans(tmp_path, method):
    payload = {"tools": copy.deepcopy(TOOLS),
               "messages": [{"role": "user", "content": "get_weather Paris"}]}
    request, plan = memory(tmp_path, f"{method}:r8:hybrid1:schema").prepare_full_history_request(payload)
    hint = request["c2kv_kv_memory_hint"]["tool_kv_eviction"]
    assert plan.protocol.count('"name":"get_weather"') == 1
    assert plan.info["interface_copy_tokens"] == 0
    assert plan.interface_spans == ()
    assert len(hint["schema_spans"]) == len(TOOLS)
    assert {span["schema_index"] for span in hint["schema_spans"]} == set(range(len(TOOLS)))
    assert hint["protected_schema_indices"] == [2]
    for span in hint["schema_spans"]:
        content = request["messages"][span["message_index"]]["content"]
        assert content[span["start"]:span["end"]] == span["text"]
        assert json.loads(span["text"]) in {
            "Book a flight", "Cancel an order", "Weather for a city"}


def test_raw_source_once_and_only_description_values_selectable(tmp_path):
    doc = json.dumps({"name": "lookup", "description": "Find records", "parameters": {
        "type": "object", "properties": {"q": {"type": "string",
                                               "description": "Search query"}}}})
    payload = _source_payload(doc)
    request, plan = memory(tmp_path, "h2o:r8:schema").prepare_full_history_request(payload)
    hint = request["c2kv_kv_memory_hint"]["tool_kv_eviction"]
    assert plan.protocol.count(doc) == 1
    assert doc not in request["messages"][1]["content"]
    assert len(hint["schema_spans"]) == 2
    assert {json.loads(span["text"]) for span in hint["schema_spans"]} == {
        "Find records", "Search query"}
    assert all(span["message_index"] == 0 for span in hint["schema_spans"])
    assert hint["tool_protocol_span"]["text"] == plan.protocol


def test_default_none_keeps_full_document_and_protocol(tmp_path):
    plan = memory(tmp_path, "t0:r8:hybrid1").plan({
        "messages": [{"role": "user", "content": "get_weather"}], "tools": TOOLS})
    assert plan.protocol == toolmemory.protocol_block([TOOLS[2]])
    assert [chunk.catalog_index for chunk in plan.chunks] == [0, 1]
    assert all("tool" in document and document["type"] == "tool_definition"
               for document in toolmemory.t0_documents(TOOLS, [0, 1]))


def test_path_import_finds_sibling_helper_without_benchmarks_on_sys_path():
    script = """
import importlib.util, sys
from pathlib import Path
p = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location('c2kv_shared_toolmemory', p)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module.parse_tool_memory_spec('t0:r8:schema').interface_policy == 'schema'
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(HERE / "toolmemory.py")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
