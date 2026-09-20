"""Schema interface survives the paper proxy's history-method request paths."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import proxy  # noqa: E402
import toolmemory  # noqa: E402
import textarms  # noqa: E402
from arms import Arm, get_arm  # noqa: E402
from test_toolmemory import FakeTokenizer, TOOLS, memory  # noqa: E402


def _annotated_source():
    doc = '{"name":"lookup","description":"Full unchanged execution details","parameters":{"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}}'
    messages = [{"role": "system", "content": "Follow the documented API."},
                {"role": "user", "content": "Old definition: " + doc},
                {"role": "assistant", "content": "Earlier response."},
                {"role": "user", "content": "A later turn."},
                {"role": "assistant", "content": "Later response."},
                {"role": "user", "content": "Lookup the current item."}]
    payload = {"messages": messages, toolmemory.TOOL_SPANS_FIELD: [{
        "message_index": 1, "start": len("Old definition: "),
        "end": len("Old definition: ") + len(doc), "source": "api_docs"}]}
    return payload, doc


def _stage(memory, payload, arm, monkeypatch):
    plan = memory.plan(payload)
    monkeypatch.setattr(proxy._TRACE, "tool_plan", plan, raising=False)
    assembled, counts = proxy._assemble_request(plan.messages, arm)
    request = memory.stage_request({**payload, "messages": assembled}, plan)
    return request, plan, counts


@pytest.mark.parametrize("arm_name", ["acon_hist_ut_co", "hiagent_full"])
@pytest.mark.parametrize("spec", ["t0:r8:hybrid3:schema", "h2o:r8:hybrid3:schema"])
def test_schema_interface_is_preserved_after_text_policy(
    tmp_path, monkeypatch, arm_name, spec,
):
    tools = [*TOOLS, {"type": "function", "function": {
        "name": "lookup_receipt", "description": "Lookup a receipt",
        "parameters": {"type": "object", "properties": {"number": {"type": "string"}}}}}]
    source = {"model": "c2kv-agent", "tools": tools, "messages": [
        {"role": "system", "content": "Use the available tools."},
        {"role": "user", "content": "Earlier request"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "Lookup receipt 42"},
    ]}
    def transform(messages, *args, **kwargs):
        return list(messages), {"policy": arm_name}
    if arm_name == "hiagent_full":
        monkeypatch.setattr(textarms, "hiagent_transform", transform)
    else:
        monkeypatch.setattr(textarms, "acon_transform", transform)
    arm = get_arm(arm_name)
    transformed, stats = proxy._apply_text_arm(source, arm, "case-42")
    assert stats["policy"] == arm_name
    assert transformed["tools"][:len(tools)] == tools
    if arm_name == "hiagent_full":
        assert len(transformed["tools"]) == len(tools) + 1
        assert transformed["tools"][-1]["function"]["name"] == textarms.HIAGENT_RETRIEVE_TOOL_NAME
    selected = memory(tmp_path, spec, FakeTokenizer())
    request, plan, counts = _stage(selected, transformed, arm, monkeypatch)
    assert request["tools"] == transformed["tools"]
    assert request["c2kv_tools_in_prompt"] is False
    assert plan.info["interface_policy"] == "schema"
    assert plan.info["n_protected_interfaces"] >= 1
    assert counts["tool_memory"]["spec"].endswith("_schema")
    system = request["messages"][0]["content"]
    assert "# Executable tool interfaces" in system
    if spec.startswith("t0:"):
        assert any("c2kv_key_hash" in msg for msg in request["messages"])
    else:
        hint = request["c2kv_kv_memory_hint"]["tool_kv_eviction"]
        assert hint["method"] == "h2o"
        assert len(hint["protected_interface_spans"]) >= 1
        for span in hint["protected_interface_spans"]:
            text = request["messages"][span["message_index"]]["content"]
            assert text[span["start"]:span["end"]] == span["text"]


@pytest.mark.parametrize("arm_name", ["full", "c2kv4", "history_kv_h2o_r25_persistent"])
def test_schema_interface_survives_history_assembly(tmp_path, monkeypatch, arm_name):
    monkeypatch.setattr(proxy, "_count_extract_tokens", lambda role, text: len(text))
    monkeypatch.setattr(proxy, "_extract", lambda role, text, ratio, timeout: {
        "key_hash": "history-" + str(len(text)), "gist_len": 2,
        "original_seq_len": len(text)})
    source = {"tools": TOOLS, "messages": [
        {"role": "system", "content": "Follow the tool protocol."},
        {"role": "user", "content": "Earlier request"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "cancel_order 42"},
    ]}
    selected = memory(tmp_path, "t0:r8:hybrid1:schema", FakeTokenizer())
    request, plan, counts = _stage(selected, source, get_arm(arm_name), monkeypatch)
    assert request["c2kv_tools_in_prompt"] is False
    assert plan.info["n_protected_interfaces"] == len(TOOLS)
    assert any("c2kv_key_hash" in message for message in request["messages"])
    assert "# Executable tool interfaces" in request["messages"][0]["content"]
    assert counts["tool_memory"]["spec"] == "t0_r8_hybrid1_schema"


@pytest.mark.parametrize("spec", ["t0:r8:schema", "h2o:r8:schema"])
def test_dropped_source_keeps_protected_interface_and_t0_full_chunks(
    tmp_path, monkeypatch, spec,
):
    payload, doc = _annotated_source()
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    monkeypatch.setattr(proxy, "MAX_DOC_NUM", 1)
    monkeypatch.setattr(proxy, "_count_extract_tokens", lambda role, text: len(text))
    monkeypatch.setattr(proxy, "_extract", lambda role, text, ratio, timeout: {
        "key_hash": "history-" + str(len(text)), "gist_len": 2,
        "original_seq_len": len(text)})
    owner = memory(tmp_path, spec, FakeTokenizer())
    plan = owner.plan(payload)
    monkeypatch.setattr(proxy._TRACE, "tool_plan", plan, raising=False)
    arm = get_arm("c2kv4")
    assembled, counts = proxy._assemble_request(plan.messages, arm)
    request = owner.stage_request({**payload, "messages": assembled}, plan)
    assert counts["dropped_docs"] > 0
    assert "# Executable tool interfaces" in request["messages"][0]["content"]
    assert doc not in request["messages"][0]["content"]
    if spec.startswith("t0:"):
        assert counts["tool_memory"]["source_prefix_fallback_indices"] == [0]
        assert len(plan.chunks) == len(plan.records)
        assert [message["c2kv_key_hash"] for message in request["messages"]
                if message.get("c2kv_region") == "tool"] == [
                    record["key_hash"] for record in plan.records]
        assert all(message.get("c2kv_source_token_count") == len(chunk.token_ids)
                   for message, chunk in zip(
                       (message for message in request["messages"]
                        if message.get("c2kv_region") == "tool"), plan.chunks))
    else:
        assert counts["tool_memory"]["omitted_history_schema_indices"] == [0]
        assert plan.assembled_schema_spans == ()
        assert "tool_kv_eviction" not in request.get("c2kv_kv_memory_hint", {})
        assert plan.info["raw_tool_eviction"] == "skipped_no_retained_schema"


def test_rewritten_source_retains_raw_interface_but_not_stale_offsets(tmp_path, monkeypatch):
    payload, _ = _annotated_source()
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    monkeypatch.setattr(proxy, "MAX_DOC_NUM", 100)
    monkeypatch.setattr(proxy, "_count_extract_tokens", lambda role, text: len(text))
    monkeypatch.setattr(proxy, "_extract", lambda role, text, ratio, timeout: {
        "key_hash": "history-" + str(len(text)), "gist_len": 2,
        "original_seq_len": len(text)})
    owner = memory(tmp_path, "h2o:r8:hybrid1:schema", FakeTokenizer())
    plan = owner.plan(payload)
    monkeypatch.setattr(proxy._TRACE, "tool_plan", plan, raising=False)
    assembled, counts = proxy._assemble_request(plan.messages, get_arm("c2kv4"))
    request = owner.stage_request({**payload, "messages": assembled}, plan)
    assert counts["dropped_docs"] == 0
    assert counts["tool_memory"]["omitted_history_schema_indices"] == [0]
    assert plan.info["native_source_interface_copy_indices"] == [0]
    assert "tool_kv_eviction" not in request.get("c2kv_kv_memory_hint", {})
    assert "Full unchanged execution details" in request["messages"][0]["content"]


def test_joint_history_kv_uses_tool_only_target_and_retained_schema_indices(tmp_path, monkeypatch):
    payload, _ = _annotated_source()
    payload["tools"] = TOOLS
    owner = memory(tmp_path, "h2o:r8:hybrid1:schema", FakeTokenizer())
    plan = owner.plan(payload)
    monkeypatch.setattr(proxy._TRACE, "tool_plan", plan, raising=False)
    arm = Arm(name="history_kv_h2o_r25_persistent", compress_history=False,
              history_kv={"method": "h2o", "retention_ratio": 0.25})
    assembled, counts = proxy._assemble_request(plan.messages, arm)
    request = owner.stage_request({**payload, "messages": assembled}, plan)
    hint = request["c2kv_kv_memory_hint"]["tool_kv_eviction"]
    assert hint["joint_history_assembly"] is True
    assert hint["joint_tool_target_tokens_per_layer"] == plan.info["matched_resident_tool_tokens"]
    assert hint["joint_tool_target_tokens_per_layer"] < hint["target_resident_tokens_per_layer"]
    retained = {span["schema_index"] for span in hint["schema_spans"]}
    assert hint["protected_schema_indices"] == [
        index for index in plan.info["native_indices"] if index in retained]
    assert "history_kv_event_messages" in counts


def test_dropped_top_k_source_does_not_protect_an_unseen_schema(tmp_path, monkeypatch):
    payload, _ = _annotated_source()
    payload["tools"] = TOOLS
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    monkeypatch.setattr(proxy, "MAX_DOC_NUM", 1)
    monkeypatch.setattr(proxy, "_count_extract_tokens", lambda role, text: len(text))
    monkeypatch.setattr(proxy, "_extract", lambda role, text, ratio, timeout: {
        "key_hash": "history-" + str(len(text)), "gist_len": 2,
        "original_seq_len": len(text)})
    owner = memory(tmp_path, "h2o:r8:hybrid1:schema", FakeTokenizer())
    plan = owner.plan(payload)
    assert plan.info["native_indices"] == [len(TOOLS)]
    monkeypatch.setattr(proxy._TRACE, "tool_plan", plan, raising=False)
    assembled, counts = proxy._assemble_request(plan.messages, get_arm("c2kv4"))
    request = owner.stage_request({**payload, "messages": assembled}, plan)
    hint = request["c2kv_kv_memory_hint"]["tool_kv_eviction"]
    assert counts["tool_memory"]["omitted_history_schema_indices"] == [len(TOOLS)]
    assert {span["schema_index"] for span in hint["schema_spans"]} == set(range(len(TOOLS)))
    assert hint["protected_schema_indices"] == []
    assert "Full unchanged execution details" in request["messages"][0]["content"]


@pytest.mark.parametrize("arm_name", ["acon_hist_ut_co", "hiagent_full"])
@pytest.mark.parametrize("spec", ["t0:r8:schema", "h2o:r8:schema"])
def test_text_summary_plans_annotated_tools_before_rewriting_history(
    tmp_path, monkeypatch, arm_name, spec,
):
    payload, doc = _annotated_source()
    arm = get_arm(arm_name)
    owner = memory(tmp_path, spec, FakeTokenizer())
    monkeypatch.setattr(proxy, "TOOL_MEMORY", owner)

    def transform(messages, *_args, **_kwargs):
        assert doc not in "\n".join(message["content"] for message in messages)
        return [messages[0], messages[-1]], {"source_history_transform": arm_name}

    monkeypatch.setattr(textarms, "hiagent_transform" if arm_name == "hiagent_full"
                        else "acon_transform", transform)
    stripped, plan = proxy._text_source_tool_plan(payload, arm)
    assert stripped.get(toolmemory.TOOL_SPANS_FIELD) is None
    assert plan.info["n_visible_source_spans"] == 1
    assert doc not in "\n".join(message["content"] for message in stripped["messages"])
    transformed, stats = proxy._apply_text_arm(stripped, arm, "text-source-test")
    assert stats["source_history_transform"] == arm_name
    finished = proxy._finish_text_source_tool_plan(transformed, plan)
    monkeypatch.setattr(proxy._TRACE, "tool_plan", plan, raising=False)
    assembled, counts = proxy._assemble_request(finished["messages"], arm)
    request = owner.stage_request({**finished, "messages": assembled}, plan)
    actor_content = "\n".join(message.get("content") or ""
                              for message in request["messages"])
    assert doc not in actor_content
    assert "# Executable tool interfaces" in request["messages"][0]["content"]
    assert counts["tool_memory"]["source_history_order"] == "tool_documents_before_text_history"
    assert counts["tool_memory"]["source_history_placement"] == (
        "compressed_source_chunks_at_prefix" if spec.startswith("t0:")
        else "protected_interfaces_at_prefix")
    if arm_name == "hiagent_full":
        names = [(tool.get("function") or {}).get("name") for tool in request["tools"]]
        assert names.count(textarms.HIAGENT_RETRIEVE_TOOL_NAME) == 1
        retrieved = dict(transformed, messages=[
            {"role": "system", "content": "Retrieved goal details. " + transformed["messages"][0]["content"]},
            *transformed["messages"][1:]])
        retried = proxy._finish_text_source_tool_plan(retrieved, plan)
        retry_assembled, retry_counts = proxy._assemble_request(retried["messages"], arm)
        retry_request = owner.stage_request({**retried, "messages": retry_assembled}, plan)
        assert retry_counts["tool_memory"]["source_history_placement"] == (
            "compressed_source_chunks_at_prefix" if spec.startswith("t0:")
            else "protected_interfaces_at_prefix")
        assert retry_request["messages"][0]["content"].count("# Executable tool interfaces") == 1
    if spec.startswith("t0:"):
        assert counts["tool_memory"]["n_source_prefix_fallbacks"] == 1
        assert any(message.get("c2kv_region") == "tool" for message in request["messages"])
        assert [message["c2kv_key_hash"] for message in request["messages"]
                if message.get("c2kv_region") == "tool"] == [
                    record["key_hash"] for record in plan.records]
    else:
        assert counts["tool_memory"]["n_omitted_history_schemas"] == 1
        assert all(span["schema_index"] < plan.info["n_structured_tools"]
                   for span in plan.assembled_schema_spans)
