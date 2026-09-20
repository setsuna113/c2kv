"""Joint raw tool/history requests carry one independent KV policy each."""
from __future__ import annotations

import sys
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import toolmemory  # noqa: E402
import toolmemory_joint  # noqa: E402
from backends.sglang import SglangBackend  # noqa: E402
from test_toolmemory import fake_extract, write_t0_checkpoint  # noqa: E402


class CharacterTokenizer:
    def _load(self):
        return self

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            tools=None, **kwargs):
        rendered = "".join(f"<{message['role']}>" + str(message.get("content") or "")
                           + "</>" for message in messages)
        if add_generation_prompt:
            rendered += "<assistant>"
        return [ord(char) for char in rendered] if tokenize else rendered

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [ord(char) for char in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}

    def native_ids(self, messages, *, tools=None, generation=False):
        return tuple(self.apply_chat_template(messages, tokenize=True,
                     add_generation_prompt=generation, tools=tools))


def _fixture(tmp_path, method="h2o", *, native=False, tool_turn=False,
             visible_source=False, system="Follow instructions.", query="Lookup now"):
    tokenizer = CharacterTokenizer()
    tool = {"type": "function", "function": {
        "name": "lookup", "description": "long prose " * 20,
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "annotation " * 20}},
            "required": ["key"]}}}
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": query}]
    if tool_turn:
        messages[2] = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-1", "type": "function", "function": {
                "name": "lookup", "arguments": '{"key":"old"}'}}]}
        messages.insert(3, {"role": "tool", "tool_call_id": "call-1",
                            "content": "old result"})
    if visible_source:
        source = "run_tool(source: str) -> result; keep this executable signature"
        messages[-1]["content"] += "\n" + source
    payload = {"tools": [tool], "messages": messages}
    if visible_source:
        payload[toolmemory.TOOL_SPANS_FIELD] = [{
            "message_index": len(messages) - 1,
            "start": messages[-1]["content"].index(source),
            "end": messages[-1]["content"].index(source) + len(source),
            "source": "producer"}]
    spec = toolmemory.parse_tool_memory_spec(
        f"{method}:r8:{'hybrid1' if native else 'uniform'}:schema")
    owner = toolmemory.ToolMemory(spec, write_t0_checkpoint(tmp_path / "checkpoint-1034"),
                                  fake_extract, tokenizer=tokenizer, budget_tokens=36864)
    plan = owner.plan(payload)
    plan.info["joint_history_assembly"] = True
    plan.info["joint_tool_target_tokens_per_layer"] = int(
        plan.info["matched_resident_tool_tokens"])
    staged = owner.stage_request({**payload, "messages": plan.messages}, plan)
    arm = SimpleNamespace(name="history", history_kv={"method": "h2o"},
                          kv_reuse=None, constrain_tools=False, repair=False)
    context = {"history_kv": {"spec": {
        "method": "h2o", "backend": "physical_eviction", "target_tokens": 5,
        "retention_ratio": None, "recent_window": 16, "kernel_size": 7,
        "pooling": "avgpool", "h2o_recent_fraction": 0.5},
        "history_out_indices": [1, 2, 3] if tool_turn else [1, 2],
        "history_text": "Earlier question Earlier answer",
        "history_message_count": 4 if tool_turn else 3,
        "history_start_message_count": 1}}
    calls = []

    def post(path, body, timeout):
        calls.append((path, body))
        return {"success": True, "key_hash": "tool-kv", "token_len": body["history_kv_target_tokens"],
                "original_seq_len": len(body["input_ids"]),
                "span_start": body["span_start"], "span_end": body["span_end"],
                "history_kv_method": body["history_kv_method"]}

    backend = SglangBackend(post)
    prepared = backend.prepare_chat(staged, arm, None, context=context)
    return owner, plan, staged, prepared, backend, calls


def test_joint_source_identity_changes_for_prefix_but_not_query(tmp_path):
    digests = []
    for index, (system, query) in enumerate([
        ("Follow instructions.", "Lookup now"),
        ("Follow instructions.", "Lookup a different record"),
        ("Changed instructions.", "Lookup now"),
    ]):
        owner, plan, staged, prepared, backend, _ = _fixture(
            tmp_path / str(index), system=system, query=query)
        final = toolmemory_joint.prepare_joint_raw_tool_history(
            staged, prepared, plan, owner.tokenizer, backend)
        digests.append(final["c2kv_kv_memory_hint"]["joint_tool_memory"][
            "source_protocol_token_sha256"])
    assert digests[0] == digests[1]
    assert digests[0] != digests[2]


@pytest.mark.parametrize("method", ["streamingllm", "h2o", "snapkv", "pyramidkv"])
def test_joint_request_preextracts_tool_and_preserves_history_method(tmp_path, method):
    owner, plan, staged, prepared, backend, calls = _fixture(tmp_path, method)
    assert "tool_kv_eviction" in prepared["c2kv_kv_memory_hint"]
    assert "history_kv_eviction" in prepared["c2kv_kv_memory_hint"]
    final = toolmemory_joint.prepare_joint_raw_tool_history(
        staged, prepared, plan, owner.tokenizer, backend)
    hint = final["c2kv_kv_memory_hint"]
    assert "tool_kv_eviction" not in hint
    assert hint["history_kv_eviction"]["method"] == "h2o"
    assert len(calls) == 1
    path, repair = calls[0]
    assert path == "/v1/c2kv/repair_extract"
    assert repair["history_kv_method"] == method
    assert repair["history_kv_selectable_relative_indices"]
    assert len(repair["history_kv_selectable_relative_indices"]) < (
        repair["span_end"] - repair["span_start"])
    assert repair["history_kv_target_tokens"] >= (
        repair["span_end"] - repair["span_start"]
        - len(repair["history_kv_selectable_relative_indices"]))
    assert final["messages"][1]["c2kv_region"] == "tool"
    assert final["messages"][1]["c2kv_source_token_end"] == repair["span_end"]
    assert hint["history_kv_eviction"]["history_message_count"] == 4
    assert plan.protocol not in final["messages"][0]["content"]
    assert hint["joint_tool_memory"]["resident_tool_tokens"] <= owner.budget_tokens


def test_joint_all_native_keeps_full_raw_protocol_without_repair(tmp_path):
    owner, plan, staged, prepared, backend, calls = _fixture(tmp_path, native=True)
    final = toolmemory_joint.prepare_joint_raw_tool_history(
        staged, prepared, plan, owner.tokenizer, backend)
    assert calls == []
    assert plan.protocol in final["messages"][0]["content"]
    assert "tool_kv_eviction" not in final["c2kv_kv_memory_hint"]
    assert final["c2kv_kv_memory_hint"]["history_kv_eviction"]["method"] == "h2o"
    assert plan.info["selection_backend"] == "raw_full_no_repair"


def test_non_joint_history_keeps_tool_hint_after_backend_shaping(tmp_path):
    owner, plan, staged, prepared, _, _ = _fixture(tmp_path)
    plan.info.pop("joint_history_assembly")
    staged["c2kv_kv_memory_hint"]["tool_kv_eviction"].pop("joint_history_assembly")
    assert prepared["c2kv_kv_memory_hint"]["tool_kv_eviction"] is (
        staged["c2kv_kv_memory_hint"]["tool_kv_eviction"])
    assert "history_kv_eviction" in prepared["c2kv_kv_memory_hint"]


def test_non_joint_c2kv_history_remaps_visible_source_after_gist_carrier(tmp_path):
    _, plan, staged, _, backend, _ = _fixture(tmp_path, visible_source=True)
    staged["messages"] = list(staged["messages"])
    staged["messages"].insert(1, {"role": "user", "content": "",
                                  "c2kv_key_hash": "history-gist"})
    tool_hint = dict(staged["c2kv_kv_memory_hint"]["tool_kv_eviction"])
    tool_hint.pop("joint_history_assembly")
    tool_hint["schema_spans"] = [
        {**span, "message_index": span["message_index"] + (span["message_index"] >= 1)}
        for span in tool_hint["schema_spans"]]
    staged["c2kv_kv_memory_hint"] = {"tool_kv_eviction": tool_hint}
    arm = SimpleNamespace(name="c2kv", history_kv=None, kv_reuse=None,
                          constrain_tools=False, repair=False)
    prepared = backend.prepare_chat(staged, arm, None)
    remapped = prepared["c2kv_kv_memory_hint"]["tool_kv_eviction"]
    source = next(span for span in remapped["schema_spans"] if span["schema_index"] == 1)
    assert source["message_index"] == len(staged["messages"]) - 2
    assert staged["messages"][-1]["content"][source["start"]:source["end"]] == source["text"]
    assert remapped["tool_protocol_span"]["message_index"] == 0


def test_joint_exact_frame_failure_does_not_send_chat(tmp_path):
    owner, plan, staged, prepared, backend, calls = _fixture(tmp_path)
    prepared["messages"][0] = dict(prepared["messages"][0],
                                   content="changed " + prepared["messages"][0]["content"])
    with pytest.raises(toolmemory_joint.JointToolMemoryError,
                       match="PROTOCOL_MOVED_BY_HISTORY"):
        toolmemory_joint.prepare_joint_raw_tool_history(
            staged, prepared, plan, owner.tokenizer, backend)
    assert calls == []


def test_joint_handles_assistant_tool_calls_with_null_content(tmp_path):
    owner, plan, staged, prepared, backend, calls = _fixture(
        tmp_path, tool_turn=True)
    final = toolmemory_joint.prepare_joint_raw_tool_history(
        staged, prepared, plan, owner.tokenizer, backend)
    assert len(calls) == 1
    assert final["messages"][3]["content"] is None
    assert final["messages"][4]["role"] == "tool"
    assert final["messages"][1]["c2kv_region"] == "tool"


def test_joint_counts_retained_opaque_source_outside_protocol_once(tmp_path):
    owner, plan, staged, prepared, backend, calls = _fixture(
        tmp_path, visible_source=True)
    final = toolmemory_joint.prepare_joint_raw_tool_history(
        staged, prepared, plan, owner.tokenizer, backend)
    audit = final["c2kv_kv_memory_hint"]["joint_tool_memory"]
    assert audit["retained_source_span_count"] == 1
    assert audit["retained_source_tool_tokens"] > 0
    assert audit["raw_source_fallback_count"] == 1
    assert audit["resident_tool_tokens_accounting"] == "pre_history_eviction_upper_bound"
    assert audit["active_tool_protocol_tokens"] == audit["selected_protocol_tokens"]
    assert audit["resident_tool_tokens"] == (
        audit["selected_protocol_tokens"] + audit["retained_source_tool_tokens"])
    assert len(calls) == 1


def test_joint_keeps_repair_extract_history_carrier_independent(tmp_path):
    owner, plan, staged, _, backend, calls = _fixture(tmp_path)
    backend.history_kv_extract = lambda *args: {
        "key_hash": "history-kv", "selected_token_count": 4,
        "requested_span_tokens": 32}
    arm = SimpleNamespace(name="history", history_kv={"method": "h2o"},
                          kv_reuse=None, constrain_tools=False, repair=False)
    context = {"history_kv": {"spec": {"method": "h2o", "backend": "repair_extract"},
               "history_out_indices": [1, 2], "history_text": "Earlier question Earlier answer",
               "history_message_count": 3, "history_start_message_count": 1}}
    prepared = backend.prepare_chat(staged, arm, None, context=context)
    final = toolmemory_joint.prepare_joint_raw_tool_history(
        staged, prepared, plan, owner.tokenizer, backend)
    assert "history_kv_eviction" not in final["c2kv_kv_memory_hint"]
    assert final["c2kv_kv_memory_hint"]["history_kv_backend"] == "repair_extract"
    assert final["messages"][1]["c2kv_repair_only_key_hashes"] == ["tool-kv"]
    assert final["messages"][2]["c2kv_repair_only_key_hashes"] == ["history-kv"]
    assert len(calls) == 1


@pytest.mark.parametrize("method", ["streamingllm", "h2o", "snapkv", "pyramidkv"])
def test_joint_c2kv_gist_history_keeps_two_independent_carriers(tmp_path, method):
    owner, plan, staged, _, backend, calls = _fixture(tmp_path, method)
    staged["messages"] = list(staged["messages"])
    staged["messages"].insert(1, {"role": "user", "content": "",
                                  "c2kv_key_hash": "history-gist", "c2kv_region": "history",
                                  "c2kv_source_token_count": 32})
    staged["c2kv_kv_memory_hint"]["tool_kv_eviction"].pop("joint_history_assembly")
    arm = SimpleNamespace(name="c2kv", history_kv=None, kv_reuse=None,
                          constrain_tools=False, repair=False)
    prepared = backend.prepare_chat(staged, arm, None)
    final = toolmemory_joint.prepare_joint_raw_tool_history(
        staged, prepared, plan, owner.tokenizer, backend,
        source_messages=plan.messages)
    assert "tool_kv_eviction" not in final["c2kv_kv_memory_hint"]
    assert final["c2kv_kv_memory_hint"]["joint_tool_memory"]["history_composition"] == "gist_carriers"
    assert final["messages"][1]["c2kv_repair_only_key_hashes"] == ["tool-kv"]
    assert final["messages"][2]["c2kv_key_hash"] == "history-gist"
    assert final["messages"][1]["c2kv_repair_token_start"] < len(
        owner.tokenizer.native_ids(final["messages"][:1]))
    assert plan.protocol not in final["messages"][0]["content"]
    assert len(calls) == 1


@pytest.mark.skipif(not os.getenv("C2KV_REAL_TOKENIZER_CHECKPOINT"),
                    reason="requires an installed checkpoint tokenizer")
@pytest.mark.parametrize("gist_history", [False, True])
def test_joint_real_qwen_token_frame(gist_history):
    tokenizer = toolmemory.NativeTokenizer(Path(os.environ["C2KV_REAL_TOKENIZER_CHECKPOINT"]))
    tool = {"type": "function", "function": {"name": "lookup",
        "description": "Find a record by key. " * 20,
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "Record key. " * 20}},
            "required": ["key"]}}}
    messages = [{"role": "system", "content": "Follow instructions."},
                {"role": "user", "content": "Earlier lookup"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call-1", "type": "function", "function": {
                        "name": "lookup", "arguments": '{"key":"old"}'}}]},
                {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
                {"role": "user", "content": "Lookup the new record"}]
    payload = {"messages": messages, "tools": [tool]}
    spec = toolmemory.parse_tool_memory_spec("h2o:r8:uniform:schema")
    plan = toolmemory.plan_visible_tool_memory(payload, spec)
    toolmemory.prepare_raw_tool_plan(payload, plan, tokenizer, budget_tokens=36864)
    hint = {"method": "h2o", "schema_spans": list(plan.raw_schema_spans),
            "protected_schema_indices": [],
            "tool_protocol_span": plan.info["tool_protocol_span"],
            "protected_interface_spans": list(plan.interface_spans),
            "target_resident_tokens_per_layer": plan.info["target_resident_tokens_per_layer"],
            "joint_history_assembly": not gist_history,
            "joint_tool_target_tokens_per_layer": plan.info["matched_resident_tool_tokens"],
            "recent_window": 16, "kernel_size": 7, "pooling": "avgpool",
            "h2o_recent_fraction": 0.5, "max_resident_tool_tokens": 36864}
    staged = {"messages": list(plan.messages), "tools": [tool],
              "c2kv_tools_in_prompt": False,
              "chat_template_kwargs": {"enable_thinking": False},
              "c2kv_kv_memory_hint": {"tool_kv_eviction": hint}}
    if gist_history:
        staged["messages"].insert(1, {"role": "user", "content": "",
                                      "c2kv_key_hash": "history-gist"})
    calls = []

    def post(path, body, timeout):
        calls.append(body)
        return {"success": True, "key_hash": "tool-kv", "token_len": body["history_kv_target_tokens"],
                "original_seq_len": len(body["input_ids"]),
                "span_start": body["span_start"], "span_end": body["span_end"],
                "history_kv_method": body["history_kv_method"]}

    backend = SglangBackend(post)
    arm = SimpleNamespace(name="c2kv", history_kv=None, kv_reuse=None,
                          constrain_tools=False, repair=False)
    prepared = backend.prepare_chat(staged, arm, None)
    final = toolmemory_joint.prepare_joint_raw_tool_history(
        staged, prepared, plan, tokenizer, backend,
        source_messages=plan.messages if gist_history else None)
    assert "tool_kv_eviction" not in final["c2kv_kv_memory_hint"]
    assert calls and final["messages"][1]["c2kv_source_token_count"] > 0


@pytest.mark.parametrize("retained_source", [False, True])
def test_joint_source_only_protocol_remains_raw_and_counted(tmp_path, retained_source):
    tokenizer = CharacterTokenizer()
    signature = "run_tool(path: str) -> result"
    content = "Current request: " + signature
    payload = {"messages": [{"role": "system", "content": "Use tools."},
                            {"role": "user", "content": "Older question"},
                            {"role": "assistant", "content": "Older answer"},
                            {"role": "user", "content": content}],
               toolmemory.TOOL_SPANS_FIELD: [{"message_index": 3,
                  "start": content.index(signature), "end": len(content),
                  "source": "producer"}]}
    spec = toolmemory.parse_tool_memory_spec("h2o:r8:uniform:schema")
    owner = toolmemory.ToolMemory(spec,
        write_t0_checkpoint(tmp_path / "checkpoint-1034"), fake_extract,
        tokenizer=tokenizer, budget_tokens=36864)
    plan = owner.plan(payload)
    plan.info["joint_history_assembly"] = True
    plan.info["joint_tool_target_tokens_per_layer"] = plan.info["matched_resident_tool_tokens"]
    staged_messages = [dict(message) for message in plan.messages]
    if not retained_source:
        staged_messages[3]["content"] = "Current request"
        plan.assembled_schema_spans = ()
    staged = owner.stage_request({"messages": staged_messages, "tools": []}, plan)
    arm = SimpleNamespace(name="history", history_kv={"method": "h2o"},
                          kv_reuse=None, constrain_tools=False, repair=False)
    history = {"spec": {"method": "h2o", "backend": "physical_eviction",
                        "target_tokens": 5, "retention_ratio": None,
                        "recent_window": 16, "kernel_size": 7,
                        "pooling": "avgpool", "h2o_recent_fraction": 0.5},
               "history_out_indices": [1, 2], "history_text": "Older question Older answer",
               "history_message_count": 3, "history_start_message_count": 1}
    backend = SglangBackend(lambda *_: (_ for _ in ()).throw(AssertionError(
        "source-only catalog must not extract raw tool KV")))
    prepared = backend.prepare_chat(staged, arm, None, context={"history_kv": history})
    final = toolmemory_joint.prepare_joint_raw_tool_history(
        staged, prepared, plan, tokenizer, backend)
    audit = final["c2kv_kv_memory_hint"]["joint_tool_memory"]
    assert plan.protocol in final["messages"][0]["content"]
    assert "tool_kv_eviction" not in final["c2kv_kv_memory_hint"]
    assert audit["selection_backend"] == "raw_full_no_repair"
    assert audit["active_tool_protocol_tokens"] > 0
    assert audit["retained_source_span_count"] == int(retained_source)
    assert audit["resident_tool_tokens"] == (
        audit["active_tool_protocol_tokens"] + audit["retained_source_tool_tokens"])
