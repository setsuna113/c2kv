"""Executable tool interfaces remain raw while full definitions follow each encoder."""
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
from test_toolmemory import FakeTokenizer, TOOLS, memory, write_t0_checkpoint  # noqa: E402


@pytest.mark.parametrize("encoder", toolmemory.ENCODERS)
@pytest.mark.parametrize("layout", ["", ":uniform", ":hybrid1"])
def test_schema_policy_is_opt_in_and_has_distinct_identity(encoder, layout):
    default = toolmemory.parse_tool_memory_spec(f"{encoder}:r8{layout}")
    protected = toolmemory.parse_tool_memory_spec(f"{encoder}:r8{layout}:schema")
    assert default.interface_policy == "none"
    assert protected.interface_policy == "schema"
    assert protected.name == default.name + "_schema"
    assert protected.as_dict()["interface_policy"] == "schema"
    assert "interface_policy" not in default.as_dict()
    assert default.name == (f"{encoder}_r8_hybrid1" if layout == ":hybrid1"
                            else f"{encoder}_r8")


def test_schema_walk_preserves_executable_keys_literals_and_extensions():
    schema = {
        "type": "object", "title": "annotation", "description": "annotation",
        "examples": [{"description": "annotation"}],
        "properties": {
            "description": {"type": "string", "description": "annotation"},
            "title": {"type": "string", "default": {"description": "literal"}},
            "examples": {"type": "array", "items": {"type": "string", "title": "annotation"}},
            "mode": {"enum": [{"description": "literal", "title": "literal"}],
                     "const": {"examples": ["literal"]}},
            "nested": {"$ref": "#/$defs/Nested"},
        },
        "required": ["description", "title", "examples"],
        "$defs": {"Nested": {"type": "object", "properties": {
            "value": {"type": "integer", "description": "annotation"}}}},
        "oneOf": [{"required": ["description"]}, {"required": ["title"]}],
        "x-executable": {"description": "extension literal"},
    }
    tool = {"type": "function", "function": {"name": "act", "description": "prose",
                                           "strict": True, "parameters": schema}}
    original = copy.deepcopy(tool)
    compact = toolinterface.compact_tool(tool)
    assert tool == original
    assert compact["function"]["name"] == "act"
    assert compact["function"]["strict"] is True
    assert "description" not in compact["function"]
    params = compact["function"]["parameters"]
    assert params["required"] == schema["required"]
    assert set(params["properties"]) == set(schema["properties"])
    assert params["properties"]["title"]["default"] == {"description": "literal"}
    assert params["properties"]["mode"]["enum"] == schema["properties"]["mode"]["enum"]
    assert params["properties"]["mode"]["const"] == schema["properties"]["mode"]["const"]
    assert params["properties"]["nested"]["$ref"] == "#/$defs/Nested"
    assert params["$defs"]["Nested"]["properties"]["value"] == {"type": "integer"}
    assert params["x-executable"] == {"description": "extension literal"}


def test_t0_schema_keeps_original_encoder_chunks_and_counts_copy_once(tmp_path):
    payload = {"messages": [{"role": "user", "content": "get_weather Paris"}],
               "tools": copy.deepcopy(TOOLS)}
    baseline = memory(tmp_path / "baseline", "t0:r8:hybrid1").plan(payload)
    protected_memory = memory(tmp_path / "protected", "t0:r8:hybrid1:schema")
    protected = protected_memory.plan(payload)
    assert payload["tools"] == TOOLS
    assert protected.info["native_indices"] == baseline.info["native_indices"] == [2]
    assert [chunk.token_ids for chunk in protected.chunks] == [
        chunk.token_ids for chunk in baseline.chunks]
    assert [chunk.event_id for chunk in protected.chunks] == [
        chunk.event_id for chunk in baseline.chunks]
    assert protected.info["n_protected_interfaces"] == len(TOOLS)
    assert {span["catalog_index"] for span in protected.interface_spans} == {0, 1, 2}
    assert protected.info["interface_copy_tokens"] > 0
    assert protected.info["resident_tool_tokens"] == (
        protected.info["protocol_prefix_tokens"] + protected.info["gist_tokens"])
    assert protected.info["resident_tool_tokens"] > baseline.info["resident_tool_tokens"]
    assert protected.protocol.count('"name":"get_weather"') == 2
    for span in protected.interface_spans:
        content = protected.messages[span["message_index"]]["content"]
        assert content[span["start"]:span["end"]] == span["text"]
    protected_memory.plan(payload)
    assert protected_memory.stats["chunk_extracts"] == len(protected.chunks)
    assert protected_memory.stats["chunk_cache_hits"] == len(protected.chunks)


def test_t0_schema_budget_includes_interface_before_extraction(tmp_path):
    payload = {"messages": [{"role": "user", "content": "weather"}], "tools": TOOLS}
    spec = toolmemory.parse_tool_memory_spec("t0:r8:schema")
    pure = toolmemory.plan_visible_tool_memory(payload, spec, FakeTokenizer())
    checkpoint = write_t0_checkpoint(tmp_path / "ckpt")
    called = []
    def observed(*args):
        called.append(args)
        raise AssertionError("budget should fail before extract")
    owner = toolmemory.ToolMemory(spec, checkpoint, observed, FakeTokenizer(),
                                  budget_tokens=pure.info["resident_tool_tokens"] - 1)
    with pytest.raises(toolmemory.ToolMemoryError, match="tool_budget"):
        owner.plan(payload)
    assert called == []


def test_visible_source_is_compacted_or_preserved_with_exact_offsets(tmp_path):
    known = json.dumps({"name": "search", "description": "long prose", "parameters": {
        "type": "object", "properties": {"q": {"type": "string", "description": "long prose"}},
        "required": ["q"]}})
    opaque = "search(query: str) -> records; execute with Python"
    content = f"Before\n{known}\nBetween\n{opaque}\nAfter"
    payload = {"messages": [{"role": "user", "content": content}],
               toolmemory.TOOL_SPANS_FIELD: [
                   {"message_index": 0, "start": content.index(known),
                    "end": content.index(known) + len(known), "source": "producer"},
                   {"message_index": 0, "start": content.index(opaque),
                    "end": content.index(opaque) + len(opaque), "source": "producer"}]}
    class StrictTokenizer(FakeTokenizer):
        def native_ids(self, messages, **kwargs):
            assert messages, "native_ids requires a nonempty message list"
            return super().native_ids(messages, **kwargs)
    plan = memory(tmp_path, "t0:r8:schema", tokenizer=StrictTokenizer()).plan(payload)
    assert plan.info["n_protected_interfaces"] == 2
    assert plan.info["n_interface_fallbacks"] == 1
    assert plan.info["interface_fallback_indices"] == [1]
    assert opaque in [span["text"] for span in plan.interface_spans]
    assert "long prose" not in next(span["text"] for span in plan.interface_spans
                                    if span["catalog_index"] == 0)
    assert "Before\n" in plan.messages[1]["content"]
    assert "\nBetween\n" in plan.messages[1]["content"]
    assert plan.chunks[0].catalog_index == 0
    for span in plan.interface_spans:
        assert plan.messages[span["message_index"]]["content"][
            span["start"]:span["end"]] == span["text"]


def test_public_raw_adapter_matches_proxy_owner_without_checkpoint_state(tmp_path):
    payload = {"messages": [{"role": "user", "content": "get_weather Paris"}],
               "tools": TOOLS}
    spec = toolmemory.parse_tool_memory_spec("h2o:r8:hybrid1:schema")
    pure = toolmemory.plan_visible_tool_memory(payload, spec)
    prepared = toolmemory.prepare_raw_tool_plan(payload, pure, FakeTokenizer(),
                                                 budget_tokens=1000)
    owner = memory(tmp_path, "h2o:r8:hybrid1:schema")
    owned = owner.plan(payload)
    assert prepared is pure
    assert prepared.messages == owned.messages
    assert prepared.raw_schema_spans == owned.raw_schema_spans
    assert prepared.interface_spans == owned.interface_spans
    assert prepared.info["matched_resident_tool_tokens"] == (
        owned.info["matched_resident_tool_tokens"])
    assert prepared.info["budget_tokens"] == 1000


@pytest.mark.parametrize("encoder", ["t0", "h2o"])
def test_native_source_topk_has_full_protected_copy_when_history_drops_source(tmp_path, encoder):
    doc = json.dumps({"name": "lookup", "description": "Execution detail must survive",
                      "parameters": {"type": "object", "properties": {
                          "query": {"type": "string"}}, "required": ["query"]}})
    content = "Available: " + doc + "\nUse lookup."
    payload = {"messages": [{"role": "user", "content": content}],
               toolmemory.TOOL_SPANS_FIELD: [{"message_index": 0,
                   "start": len("Available: "), "end": len("Available: ") + len(doc),
                   "source": "producer"}]}
    owner = memory(tmp_path / encoder, f"{encoder}:r8:hybrid1:schema")
    plan = owner.plan(payload)
    assert plan.info["native_indices"] == [0]
    assert plan.info["n_protected_interfaces"] == 1
    assert plan.info["n_native_source_interface_copies"] == 1
    assert plan.info["native_source_interface_copy_indices"] == [0]
    assert plan.info["n_interface_fallbacks"] == 0
    assert plan.info["all_native"] is True
    assert plan.interface_spans[0]["copy_reason"] == "native_source_full"
    assert plan.interface_spans[0]["text"] == doc
    assert "Execution detail must survive" in plan.protocol
    assert plan.messages[1]["content"] == content
    # The source's original message can disappear in a later history view;
    # the protected system copy is complete without it.
    assert plan.messages[0]["content"].count(doc) == 1
    if encoder == "t0":
        assert plan.chunks == []
        assert plan.info["resident_tool_tokens"] == (
            plan.info["protocol_prefix_tokens"] + plan.info["native_source_tokens"])
    else:
        request = owner.stage_request({**payload, "messages": plan.messages}, plan)
        hint = request["c2kv_kv_memory_hint"]["tool_kv_eviction"]
        assert hint["protected_interface_spans"][0]["text"] == doc
        assert hint["protected_schema_indices"] == [0]
        assert hint["schema_spans"][0]["text"] == doc


@pytest.mark.parametrize("method", ["streamingllm", "h2o", "snapkv", "pyramidkv"])
def test_raw_schema_keeps_full_spans_evictable_and_protects_copy(tmp_path, method):
    payload = {"tools": copy.deepcopy(TOOLS),
               "messages": [{"role": "user", "content": "get_weather Paris"}]}
    owner = memory(tmp_path, f"{method}:r8:hybrid1:schema")
    request, plan = owner.prepare_full_history_request(payload)
    hint = request["c2kv_kv_memory_hint"]["tool_kv_eviction"]
    assert hint["protected_schema_indices"] == [2]
    assert len(hint["schema_spans"]) == len(TOOLS)
    assert {span["catalog_index"] for span in hint["protected_interface_spans"]} == {0, 1, 2}
    assert plan.info["n_protected_interfaces"] == len(TOOLS)
    assert plan.info["matched_resident_tool_tokens"] > 0
    assert plan.info["target_resident_tokens_per_layer"] > 0
    for span in [*hint["schema_spans"], *hint["protected_interface_spans"]]:
        content = request["messages"][span["message_index"]]["content"]
        assert content[span["start"]:span["end"]] == span["text"]
    assert all("description" in span["text"] for span in hint["schema_spans"])
    assert all("Book a flight" not in span["text"] for span in hint["protected_interface_spans"])
    assert request["tools"] == TOOLS


def test_raw_hybrid_source_protocol_is_stable_when_native_tool_changes(tmp_path):
    owner = memory(tmp_path, "h2o:r8:hybrid1:schema")
    plans = []
    for query in ("book_flight", "get_weather"):
        plans.append(owner.plan({"tools": copy.deepcopy(TOOLS),
                                 "messages": [{"role": "user", "content": query}]}))
    assert plans[0].info["native_indices"] != plans[1].info["native_indices"]
    assert plans[0].protocol == plans[1].protocol
    assert plans[0].info["interface_render_profile"] == toolmemory.INTERFACE_RENDER_PROFILE
    assert plans[0].info["interface_copy_tokens"] > 0


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
