"""Focused CPU contracts for persistent RACER tool-plan composition."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from history_memory.packing import MemoryView
from benchmarks.memory_runtime.event_native_tool import (
    ToolRegionController, parse_native_tool_spec, shared_tool_catalog,
)
from benchmarks.memory_runtime.racer.allocator import PersistentMemory
from benchmarks.memory_runtime.racer.generator import PersistentRacerGenerator
from benchmarks.memory_runtime.racer.tools import PersistentToolBinder
from benchmarks.memory_runtime.tests.test_event_native_tool import CharacterTokenizer


def _tool(name: str, description: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {}},
    }}


class PersistentInner:
    kv_bytes_per_token = 1
    max_recovery_rounds = 1
    policy_config = SimpleNamespace(history_budget_bytes=10000,
                                    workspace_budget_bytes=10000)

    def prepare(self, payload, *, ratio, max_new_tokens):
        messages = tuple(copy.deepcopy(list(payload["messages"])))
        memory = PersistentMemory(
            MemoryView((), ()), (), (), (), (),
            source_messages=messages,
            source_tools=tuple(copy.deepcopy(list(payload.get("tools") or ()))),
            history_message_count=len(messages),
            history_start_message_count=1,
            history_budget_tokens=128,
            retained_history_cap=64,
            common_tokens=8,
            full_history_tokens=64,
            backend_identity="h2o",
        )
        return SimpleNamespace(
            memory=memory,
            metadata={"common_raw_prompt_tokens": 8, "actual_history_bytes": 64},
            eligible_chunks=(),
        )

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error):
        return {
            "regenerate": True,
            "memory": replace(prepared.memory, recovery_messages=(
                {"role": "user", "content": "bounded recovery note"},)),
            "metadata": copy.deepcopy(prepared.metadata),
            "decision": {"status": "recover", "reason": "cpu_test"},
        }


class ExtractBackend:
    def __init__(self):
        self.calls = []

    def extract_tokens(self, token_ids, ratio, projection_set):
        self.calls.append((tuple(token_ids), ratio, projection_set))
        return {
            "key_hash": f"tool-{len(self.calls)}",
            "original_seq_len": len(token_ids),
            "gist_len": (len(token_ids) + ratio - 1) // ratio,
        }


class ToolHTTPJournal:
    def __init__(self):
        self.rows = []

    def append(self, value):
        self.rows.append(copy.deepcopy(value))


class ToolNative:
    upstream = "http://localhost:1"
    timeout_seconds = 5
    max_tool_extraction_calls = 1
    max_tool_repair_calls = 1
    tool_extraction_calls_reserved = 0
    tool_repair_calls = 0

    def __init__(self, *, fail=False):
        self.fail = fail
        self.requests = []
        self._http_journal = ToolHTTPJournal()

    def _read_json(self, request, *, label, allow_empty=False):
        body = json.loads(request.data)
        self.requests.append((request.full_url, body))
        if self.fail:
            return {"success": False, "error": "injected failure"}, 200
        return {
            "success": True,
            "key_hash": "tool-http-1",
            "original_seq_len": len(body["token_ids"]),
            "gist_len": (len(body["token_ids"]) + body["compression_ratio"] - 1)
                        // body["compression_ratio"],
        }, 200


def _payload():
    return {
        "session_id": "persistent-tools",
        "decision_key": "turn-0/step-1",
        "messages": [
            {"role": "system", "content": "Use APIs."},
            {"role": "user", "content": "alpha_lookup"},
        ],
        "tools": [
            _tool("alpha_lookup", "alpha records"),
            _tool("beta_archive", "beta records"),
        ],
    }


def test_persistent_binding_freezes_plan_and_remaps_normal_chat_carriers(
    tmp_path, monkeypatch,
):
    tokenizer = CharacterTokenizer()
    backend = ExtractBackend()
    catalog = shared_tool_catalog()
    monkeypatch.setattr(catalog, "load_tool_checkpoint_contract", lambda path, spec: {
        "checkpoint": str(path), "config_sha256": "cpu-contract",
    })
    spec = parse_native_tool_spec(
        "t0:r8:hybrid1:schema:selector=latest_event_topk_v1")
    generator = PersistentRacerGenerator(
        SimpleNamespace(), tokenizer, SimpleNamespace(), backend=backend)
    generator.configure_tool_memory(spec, checkpoint=tmp_path / "checkpoint")
    binder = generator._tool_binder
    controller = ToolRegionController(
        PersistentInner(), tokenizer, spec,
        model_context=10000, generator=generator)

    prepared = controller.prepare(_payload(), ratio=8, max_new_tokens=16)
    first_calls = len(backend.calls)
    assert first_calls == 1
    assert prepared.memory.tool_plan["schema"] == "racer-tool-plan-binding-v1"
    assert prepared.memory.tool_plan["selector"]["native_indices"] == [0]
    assert prepared.metadata["tool_memory"]["persistent_binding"] == prepared.memory.tool_plan
    assert prepared.memory.chunks == ()

    recovered = controller.reconsider(prepared, [], draft_text="retry")
    assert len(backend.calls) == first_calls
    assert recovered["memory"].tool_plan == prepared.memory.tool_plan
    assert recovered["memory"].recovery_messages

    source = list(recovered["memory"].source_messages)
    assembled = [*source, {"role": "user", "content": "internal recovery evidence"}]
    staged, history_end, history_start, index_map = binder.stage_request(
        recovered["memory"],
        {"messages": assembled, "tools": []},
        history_message_count=len(source),
        history_start_message_count=1,
    )
    assert staged["c2kv_tools_in_prompt"] is False
    assert staged["tools"] == _payload()["tools"]
    assert staged["messages"][1]["c2kv_key_hash"] == "tool-1"
    assert history_start == 2 and history_end == len(source) + 1
    assert index_map == {0: 0, 1: 2, 2: 3}


def test_persistent_binding_rejects_changed_source_after_freeze(tmp_path, monkeypatch):
    tokenizer = CharacterTokenizer()
    backend = ExtractBackend()
    catalog = shared_tool_catalog()
    monkeypatch.setattr(catalog, "load_tool_checkpoint_contract", lambda path, spec: {
        "checkpoint": str(path), "config_sha256": "cpu-contract",
    })
    binder = PersistentToolBinder(
        backend, tokenizer, checkpoint=tmp_path / "checkpoint")
    controller = ToolRegionController(
        PersistentInner(), tokenizer,
        parse_native_tool_spec("t0:r8:hybrid1:schema"),
        model_context=10000, generator=SimpleNamespace(
            bind_tool_plan=binder.bind_tool_plan))
    prepared = controller.prepare(_payload(), ratio=8, max_new_tokens=16)
    changed = list(prepared.memory.source_messages)
    changed[-1] = {"role": "user", "content": "changed after selection"}
    with pytest.raises(ValueError, match="moved or changed"):
        binder.stage_request(
            prepared.memory, {"messages": changed, "tools": []},
            history_message_count=len(changed), history_start_message_count=1)


def test_default_transport_bounds_and_journals_actual_tool_extraction(tmp_path, monkeypatch):
    tokenizer = CharacterTokenizer()
    native = ToolNative()
    catalog = shared_tool_catalog()
    monkeypatch.setattr(catalog, "load_tool_checkpoint_contract", lambda path, spec: {
        "checkpoint": str(path), "config_sha256": "cpu-contract",
    })
    generator = PersistentRacerGenerator(
        native, tokenizer, SimpleNamespace(), backend=None)
    generator.configure_tool_memory(
        parse_native_tool_spec("t0:r8:hybrid1:schema"),
        checkpoint=tmp_path / "checkpoint")
    controller = ToolRegionController(
        PersistentInner(), tokenizer,
        parse_native_tool_spec("t0:r8:hybrid1:schema"),
        model_context=10000, generator=generator)

    prepared = controller.prepare(_payload(), ratio=8, max_new_tokens=16)
    transport = prepared.memory.tool_plan["transport"]
    assert transport["attempted_tool_extraction_calls"] == 1
    assert transport["completed_tool_extraction_calls"] == 1
    assert transport["unknown_usage_calls"] == 0
    assert transport["attempts"][0]["key_hash"] == "tool-http-1"
    assert native.tool_extraction_calls_reserved == 1
    assert [row["event"] for row in native._http_journal.rows] == ["request", "response"]

    recovered = controller.reconsider(prepared, [], draft_text="retry")
    assert recovered["memory"].tool_plan["transport"] == transport
    assert len(native.requests) == 1

    from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
    record = {"generation_trace": [{
        "status": "completed",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "generation": {"stats": {"racer_tool_cost": generator._consume_tool_cost(
            "turn-0/step-1")}},
    }]}
    EventNativeDecisionRunner._totals(record)
    assert record["tool_transport_total"]["completed_tool_extraction_calls"] == 1
    assert record["tool_transport_total"]["unknown_usage_calls"] == 0

    changed = _payload()
    changed["decision_key"] = "turn-0/step-2"
    changed["messages"][-1]["content"] = "beta_archive"
    with pytest.raises(Exception, match="cap is exhausted"):
        controller.prepare(changed, ratio=8, max_new_tokens=16)
    assert len(native.requests) == 1


def test_failed_tool_extraction_is_reserved_with_unknown_usage(tmp_path, monkeypatch):
    tokenizer = CharacterTokenizer()
    native = ToolNative(fail=True)
    catalog = shared_tool_catalog()
    monkeypatch.setattr(catalog, "load_tool_checkpoint_contract", lambda path, spec: {
        "checkpoint": str(path), "config_sha256": "cpu-contract",
    })
    generator = PersistentRacerGenerator(
        native, tokenizer, SimpleNamespace(), backend=None)
    generator.configure_tool_memory(
        parse_native_tool_spec("t0:r8:hybrid1:schema"),
        checkpoint=tmp_path / "checkpoint")
    controller = ToolRegionController(
        PersistentInner(), tokenizer,
        parse_native_tool_spec("t0:r8:hybrid1:schema"),
        model_context=10000, generator=generator)

    with pytest.raises(Exception, match=r"c2kv extract\(token_ids, tool\) failed"):
        controller.prepare(_payload(), ratio=8, max_new_tokens=16)
    aggregate = generator.session_cache_info()["tool_transport"]
    assert aggregate["attempted_tool_extraction_calls"] == 1
    assert aggregate["completed_tool_extraction_calls"] == 0
    assert aggregate["unknown_usage_calls"] == 1
    assert native.tool_extraction_calls_reserved == 1
    assert native._http_journal.rows[-1]["usage_scope"] == (
        "unknown after failed or ambiguous submission")


def test_persistent_raw_binding_remaps_protected_schema_after_internal_note():
    tokenizer = CharacterTokenizer()
    backend = ExtractBackend()
    binder = PersistentToolBinder(backend, tokenizer)
    controller = ToolRegionController(
        PersistentInner(), tokenizer,
        parse_native_tool_spec(
            "h2o:r8:hybrid1:schema:selector=last_user_topk_v1"),
        model_context=10000, generator=SimpleNamespace(
            bind_tool_plan=binder.bind_tool_plan))

    prepared = controller.prepare(_payload(), ratio=8, max_new_tokens=16)
    source = list(prepared.memory.source_messages)
    assembled = [
        {"role": "system", "content": "internal history ledger"},
        *source,
    ]
    source_indices = tuple(range(1, len(assembled)))
    staged, history_end, history_start, index_map = binder.stage_request(
        prepared.memory,
        {"messages": assembled, "tools": []},
        history_message_count=len(assembled),
        history_start_message_count=1,
        source_message_indices=source_indices,
    )

    eviction = staged["c2kv_kv_memory_hint"]["tool_kv_eviction"]
    assert eviction["method"] == "h2o"
    assert eviction["protected_schema_indices"] == [0]
    for span in [
        *eviction["schema_spans"],
        *eviction["protected_interface_spans"],
        eviction["tool_protocol_span"],
    ]:
        content = staged["messages"][span["message_index"]]["content"]
        assert content[span["start"]:span["end"]] == span["text"]
    assert staged["tools"] == _payload()["tools"]
    assert (history_end, history_start) == (len(assembled), 1)
    assert index_map == {index: index for index in range(len(assembled))}
    assert backend.calls == []


def test_next_decision_refreshes_from_completed_semantic_event_and_freezes_tools(
    tmp_path, monkeypatch,
):
    tokenizer = CharacterTokenizer()
    backend = ExtractBackend()
    catalog = shared_tool_catalog()
    monkeypatch.setattr(catalog, "load_tool_checkpoint_contract", lambda path, spec: {
        "checkpoint": str(path), "config_sha256": "cpu-contract",
    })
    spec = parse_native_tool_spec(
        "t0:r8:hybrid1:schema:selector=latest_event_topk_v1")
    generator = PersistentRacerGenerator(
        SimpleNamespace(), tokenizer, SimpleNamespace(), backend=backend)
    generator.configure_tool_memory(spec, checkpoint=tmp_path / "checkpoint")
    controller = ToolRegionController(
        PersistentInner(), tokenizer, spec,
        model_context=10000, generator=generator)
    payload = _payload()
    payload["messages"][-1]["content"] = "continue"
    first = controller.prepare(payload, ratio=8, max_new_tokens=16)
    assert first.memory.tool_plan["selector"]["native_indices"] == [0]

    completed = copy.deepcopy(payload)
    completed["decision_key"] = "turn-0/step-2"
    completed["messages"].extend([
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "actual-1", "type": "function",
            "function": {"name": "beta_archive", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "actual-1",
         "content": "beta_archive completed"},
    ])
    second = controller.prepare(completed, ratio=8, max_new_tokens=16)
    assert second.memory.tool_plan["selector"]["native_indices"] == [1]
    assert second.memory.tool_plan["selector"]["selector_latest_io_present"] is True
    assert second.memory.tool_plan["binding_id"] != first.memory.tool_plan["binding_id"]
    assert second.memory.tool_plan["source_tools_sha256"] == first.memory.tool_plan[
        "source_tools_sha256"]

    changed_tools = copy.deepcopy(completed)
    changed_tools["decision_key"] = "turn-0/step-3"
    changed_tools["tools"][0]["function"]["description"] = "changed catalog"
    with pytest.raises(ValueError, match="Tools changed within a session"):
        controller.prepare(changed_tools, ratio=8, max_new_tokens=16)
