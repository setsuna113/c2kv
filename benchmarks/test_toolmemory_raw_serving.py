"""Raw tool methods share the existing request planner and serving wire."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toolmemory
from test_toolmemory import FakeTokenizer, TOOLS, write_t0_checkpoint


@pytest.mark.parametrize("method", ["streamingllm", "h2o", "snapkv", "pyramidkv"])
def test_raw_methods_preserve_history_and_send_exact_schema_spans(tmp_path, method):
    checkpoint = write_t0_checkpoint(tmp_path / "t0")
    payload = {"tools": TOOLS, "messages": [
        {"role": "user", "content": "Earlier request"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "cancel_order 42"},
    ]}
    original = copy.deepcopy(payload)
    def no_extract(*_args):
        raise AssertionError("raw KV must be selected in the serving engine")
    memory = toolmemory.ToolMemory(
        toolmemory.parse_tool_memory_spec(f"{method}:r8:hybrid1"), checkpoint,
        no_extract, tokenizer=FakeTokenizer())
    request, plan = memory.prepare_full_history_request(payload, target_resident_tokens=180)
    assert payload == original
    assert request["messages"][1:] == original["messages"]
    assert request["tools"] == original["tools"]
    assert request["c2kv_tools_in_prompt"] is False
    assert request["c2kv_use_gist_projection"] is False
    config = request["c2kv_kv_memory_hint"]["tool_kv_eviction"]
    assert config["method"] == method
    assert config["protected_schema_indices"] == [1]
    assert config["target_resident_tokens_per_layer"] == 180
    assert len(config["schema_spans"]) == len(TOOLS)
    for span in [*config["schema_spans"], config["tool_protocol_span"]]:
        content = request["messages"][span["message_index"]]["content"]
        assert content[span["start"]:span["end"]] == span["text"]
    assert not any("c2kv_key_hash" in message for message in request["messages"])
    assert plan.info["history_policy"] == "full"


def test_raw_schema_movement_is_rejected(tmp_path):
    checkpoint = write_t0_checkpoint(tmp_path / "t0")
    memory = toolmemory.ToolMemory(toolmemory.parse_tool_memory_spec("h2o:r8"),
                                   checkpoint, None, tokenizer=FakeTokenizer())
    payload = {"tools": TOOLS, "messages": [{"role": "user", "content": "weather"}]}
    plan = memory.plan(payload)
    with pytest.raises(toolmemory.ToolMemoryError, match="raw_schema_assembly"):
        memory.stage_request(payload, plan)


def test_raw_inline_schema_tracks_proxy_inserted_system(tmp_path, monkeypatch):
    import proxy
    from arms import Arm
    checkpoint = write_t0_checkpoint(tmp_path / "t0")
    memory = toolmemory.ToolMemory(toolmemory.parse_tool_memory_spec("h2o:r8"),
                                   checkpoint, None, tokenizer=FakeTokenizer())
    doc = "search(query: str)"
    payload = {"messages": [{"role": "user", "content": "API: " + doc + "\nSearch now"}],
               toolmemory.TOOL_SPANS_FIELD: [{"message_index": 0, "start": 5,
                   "end": 5 + len(doc), "source": "acebench.functions"}]}
    plan = memory.plan(payload)
    monkeypatch.setattr(proxy._TRACE, "tool_plan", plan, raising=False)
    assembled, counts = proxy._assemble_request(plan.messages, Arm(name="full", compress_history=False))
    request = memory.stage_request({**payload, "messages": assembled}, plan)
    span, = request["c2kv_kv_memory_hint"]["tool_kv_eviction"]["schema_spans"]
    assert span["message_index"] == 1
    assert request["messages"][1] == payload["messages"][0]
    assert counts["compressed_records"] == []


def test_full_and_retrieval_use_native_tools_without_extraction(tmp_path):
    checkpoint = write_t0_checkpoint(tmp_path / "t0")
    def no_extract(*_args):
        raise AssertionError("native controls need no gist extraction")
    memory = toolmemory.ToolMemory(toolmemory.parse_tool_memory_spec("t0:r8"),
                                   checkpoint, no_extract, tokenizer=FakeTokenizer())
    payload = {"tools": TOOLS, "messages": [{"role": "user", "content": "weather"}]}
    for native, retrieval in (([0, 1, 2], False), ([2], True)):
        request, plan = memory.prepare_full_history_request(
            payload, native_override=native, retrieval_only=retrieval)
        assert plan.info["native_indices"] == native
        assert plan.chunks == [] and plan.records == []
        assert "tool_kv_eviction" not in request.get("c2kv_kv_memory_hint", {})
        assert request["messages"][-1] == payload["messages"][-1]
