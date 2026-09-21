"""Raw schema requests with no selectable prose still report tool eviction."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toolmemory  # noqa: E402
from test_toolmemory import memory  # noqa: E402


@pytest.mark.parametrize("method", ["streamingllm", "h2o", "snapkv", "pyramidkv"])
def test_structured_catalog_without_prose_emits_all_protected_noop_hint(tmp_path, method):
    tools = [
        {"type": "function", "function": {"name": "get_value", "parameters": {
            "type": "object", "properties": {"key": {"type": "string"}}}}},
        {"type": "function", "function": {"name": "list_values", "parameters": {
            "type": "object", "properties": {"limit": {"type": "integer"}}}}},
    ]
    owner = memory(tmp_path, f"{method}:r8:uniform:schema")
    request, plan = owner.prepare_full_history_request({
        "messages": [{"role": "user", "content": "List values"}], "tools": tools})

    hint = request["c2kv_kv_memory_hint"]["tool_kv_eviction"]
    assert hint["method"] == method
    assert hint["protected_schema_indices"] == [0, 1]
    assert len(hint["schema_spans"]) == 2
    assert plan.info["raw_noop_schema_indices"] == [0, 1]
    assert plan.info.get("raw_tool_eviction") != "skipped_no_retained_schema"
    for index, span in enumerate(hint["schema_spans"]):
        schema = json.dumps(tools[index], ensure_ascii=False, separators=(",", ":"))
        assert span["schema_index"] == index
        assert span["text"] == schema
        assert span["message_index"] == 0
        assert request["messages"][0]["content"][span["start"]:span["end"]] == schema
        assert plan.protocol.count(schema) == 1
    assert hint["tool_protocol_span"]["text"] == plan.protocol
    assert request["c2kv_tools_in_prompt"] is False


def test_source_only_without_prose_stays_raw_without_eviction_hint(tmp_path):
    source = "run_tool(path: str) -> result"
    content = "Execute: " + source
    owner = memory(tmp_path, "h2o:r8:uniform:schema")
    request, plan = owner.prepare_full_history_request({
        "messages": [{"role": "user", "content": content}],
        toolmemory.TOOL_SPANS_FIELD: [{"message_index": 0,
            "start": content.index(source), "end": len(content), "source": "producer"}]})

    assert "tool_kv_eviction" not in request.get("c2kv_kv_memory_hint", {})
    assert plan.info["raw_tool_eviction"] == "skipped_no_retained_schema"
    assert request["messages"][0]["content"].count(source) == 1
    assert source not in request["messages"][1]["content"]
