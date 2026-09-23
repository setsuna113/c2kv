"""Focused CPU contracts for persistent RACER tool-plan composition."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
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
        self.native_calls = []

    def extract_tokens(self, token_ids, ratio, projection_set):
        self.calls.append((tuple(token_ids), ratio, projection_set))
        return {
            "key_hash": f"tool-{len(self.calls)}",
            "original_seq_len": len(token_ids),
            "gist_len": (len(token_ids) + ratio - 1) // ratio,
        }

    def extract_native_tool_tokens(self, token_ids, *, source_start):
        self.native_calls.append((tuple(token_ids), source_start))
        return {"key_hash": f"native-{len(self.native_calls)}", "token_len": len(token_ids),
                "original_seq_len": len(token_ids), "position_start": source_start,
                "position_end": source_start + len(token_ids), "already_rotated": False}


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
        if "input_ids" in body:
            return {"success": True, "key_hash": "native-http-1",
                    "original_seq_len": len(body["input_ids"]),
                    "token_len": len(body["input_ids"]),
                    "position_start": body["repair_position_ids"][0],
                    "position_end": body["repair_position_ids"][-1] + 1,
                    "already_rotated": False}, 200
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
    assert len(backend.native_calls) == 1
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
    assert staged["messages"][1]["c2kv_repair_only_key_hashes"] == ["native-1"]
    assert staged["messages"][2]["c2kv_key_hash"] == "tool-1"
    assert history_start == 3 and history_end == len(source) + 2
    assert index_map == {0: 0, 1: 3, 2: 4}


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
    assert transport["attempted_tool_repair_calls"] == 1
    assert transport["completed_tool_repair_calls"] == 1
    assert transport["unknown_usage_calls"] == 0
    assert transport["attempts"][0]["key_hash"] == "native-http-1"
    assert native.tool_extraction_calls_reserved == 1
    assert native.tool_repair_calls == 1
    assert [row["event"] for row in native._http_journal.rows] == [
        "request", "response", "request", "response"]

    recovered = controller.reconsider(prepared, [], draft_text="retry")
    assert recovered["memory"].tool_plan["transport"] == transport
    assert len(native.requests) == 2

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
    assert len(native.requests) == 2


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

    with pytest.raises(Exception, match="native tool repair_extract failed"):
        controller.prepare(_payload(), ratio=8, max_new_tokens=16)
    aggregate = generator.session_cache_info()["tool_transport"]
    assert aggregate["attempted_tool_repair_calls"] == 1
    assert aggregate["completed_tool_repair_calls"] == 0
    assert aggregate["unknown_usage_calls"] == 1
    assert native.tool_repair_calls == 1
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


def test_native_gist_swap_keeps_exact_prompt_prefix_and_document_source_frame(tmp_path, monkeypatch):
    tokenizer = CharacterTokenizer()
    backend = ExtractBackend()
    catalog = shared_tool_catalog()
    monkeypatch.setattr(catalog, "load_tool_checkpoint_contract", lambda path, spec: {
        "checkpoint": str(path), "config_sha256": "cpu-contract"})
    selections = []
    select = catalog.tool_selection
    def counted_selection(*args, **kwargs):
        selections.append(1)
        return select(*args, **kwargs)
    monkeypatch.setattr(catalog, "tool_selection", counted_selection)
    spec = parse_native_tool_spec("t0:r8:hybrid1:schema:selector=latest_event_topk_v1")
    generator = PersistentRacerGenerator(SimpleNamespace(), tokenizer, SimpleNamespace(), backend=backend)
    generator.configure_tool_memory(spec, checkpoint=tmp_path / "checkpoint")
    controller = ToolRegionController(PersistentInner(), tokenizer, spec,
                                      model_context=10000, generator=generator)
    payload = _payload()
    payload["messages"][-1]["content"] = "continue"
    first = controller.prepare(payload, ratio=8, max_new_tokens=16)
    first_plan = generator._tool_binder.resolve_plan(first.memory)
    messages, end, start, _ = generator._ledger(first.memory, "draft")
    staged1, end1, start1, _ = generator._tool_binder.stage_request(
        first.memory, {"messages": messages}, history_message_count=end,
        history_start_message_count=start, source_message_indices=generator._source_positions)
    completed = copy.deepcopy(payload)
    completed["decision_key"] = "turn-0/step-2"
    completed["messages"].extend([
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "native-swap", "type": "function", "function": {
                "name": "beta_archive", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "native-swap", "content": "beta_archive completed"},
    ])
    second = controller.prepare(completed, ratio=8, max_new_tokens=16)
    second_plan = generator._tool_binder.resolve_plan(second.memory)
    messages, end, start, _ = generator._ledger(second.memory, "draft")
    staged2, end2, start2, _ = generator._tool_binder.stage_request(
        second.memory, {"messages": messages}, history_message_count=end,
        history_start_message_count=start, source_message_indices=generator._source_positions)
    assert len(selections) == 2
    assert first_plan.info["native_indices"] == [0]
    assert second_plan.info["native_indices"] == [1]
    assert first_plan.info["persistent_render_profile"] == catalog.PERSISTENT_RENDER_PROFILE
    assert first_plan.messages == second_plan.messages[:len(first_plan.messages)]
    assert first_plan.chunks == second_plan.chunks
    assert first_plan.info["source_protocol_token_sha256"] == second_plan.info["source_protocol_token_sha256"]
    import benchmarks
    backend_path = str(Path(__file__).resolve().parents[6] / "benchmarks")
    if backend_path not in benchmarks.__path__:
        benchmarks.__path__.append(backend_path)
    from benchmarks.backends.sglang import SglangBackend
    chat_backend = SglangBackend(lambda *args: {})
    history_spec = {"method": "h2o", "backend": "physical_eviction",
                    "recent_window": 16, "kernel_size": 7, "pooling": "avgpool",
                    "h2o_recent_fraction": 0.5}
    arm = SimpleNamespace(name="h2o", history_kv=history_spec, kv_reuse=None,
                          constrain_tools=False, repair=False)
    for staged, end, start in ((staged1, end1, start1), (staged2, end2, start2)):
        served = chat_backend.prepare_chat(staged, arm, None, context={"history_kv": {
            "spec": history_spec, "session_id": "persistent-tools",
            "history_out_indices": list(range(start, end)),
            "history_message_count": end, "history_start_message_count": start,
            "history_text": json.dumps(staged["messages"][:end]),
        }})
        assert served["c2kv_kv_memory_hint"]["joint_tool_memory"] == (
            staged["c2kv_kv_memory_hint"]["joint_tool_memory"])
    carriers1 = [row for row in staged1["messages"] if row.get("c2kv_region") == "tool"]
    carriers2 = [row for row in staged2["messages"] if row.get("c2kv_region") == "tool"]
    assert [row["c2kv_source_token_count"] for row in carriers1] == [
        row["c2kv_source_token_count"] for row in carriers2]
    visible1 = [row for row in staged1["messages"] if row.get("c2kv_region") != "tool"]
    visible2 = [row for row in staged2["messages"] if row.get("c2kv_region") != "tool"]
    ids1 = tokenizer.apply_chat_template(visible1, tokenize=True)
    assert tokenizer.apply_chat_template(visible2, tokenize=True)[:len(ids1)] == ids1
    assert "c2kv_repair_only_key_hashes" in carriers1[0] and "c2kv_key_hash" in carriers2[0]
    assert "c2kv_key_hash" in carriers1[1] and "c2kv_repair_only_key_hashes" in carriers2[1]
    assert backend.native_calls[0][0] == backend.calls[1][0] == first_plan.chunks[0].token_ids
    assert backend.native_calls[1][0] == backend.calls[0][0] == first_plan.chunks[1].token_ids
    assert backend.native_calls[1][1] == backend.native_calls[0][1] + len(first_plan.chunks[0].token_ids)
    assert first_plan.info["resident_tool_tokens"] == (
        first_plan.info["protocol_prefix_tokens"] + len(first_plan.chunks[0].token_ids)
        + first_plan.records[1]["gist_len"])
    before = (len(backend.calls), len(backend.native_calls))
    controller.reconsider(second, [], draft_text="retry")
    assert len(selections) == 2
    assert before == (len(backend.calls), len(backend.native_calls))


def test_persistent_static_and_dynamic_selectors_share_document_profile(tmp_path, monkeypatch):
    catalog = shared_tool_catalog()
    tokenizer = CharacterTokenizer()
    monkeypatch.setattr(catalog, "load_tool_checkpoint_contract", lambda path, spec: {
        "checkpoint": str(path), "config_sha256": "cpu-contract"})
    stable = []
    for policy in ("last_user_topk_v1", "latest_event_topk_v1"):
        spec = parse_native_tool_spec(f"t0:r8:hybrid1:schema:selector={policy}")
        binder = PersistentToolBinder(ExtractBackend(), tokenizer, checkpoint=tmp_path)
        selected = catalog.plan_visible_tool_memory(_payload(), spec, SimpleNamespace(
            native_ids=lambda messages, **kwargs: tokenizer.apply_chat_template(messages, tokenize=True)))
        stable.append(binder.prepare_plan(selected, _payload()))
    assert stable[0].messages == stable[1].messages
    assert stable[0].chunks == stable[1].chunks
    assert stable[0].info["resident_tool_tokens"] == stable[1].info["resident_tool_tokens"]
    assert stable[0].info["source_protocol_token_sha256"] == stable[1].info["source_protocol_token_sha256"]


def test_persistent_layout_rejects_changed_protected_source_before_extraction(tmp_path, monkeypatch):
    tokenizer = CharacterTokenizer()
    backend = ExtractBackend()
    catalog = shared_tool_catalog()
    monkeypatch.setattr(catalog, "load_tool_checkpoint_contract", lambda path, spec: {
        "checkpoint": str(path), "config_sha256": "cpu-contract"})
    spec = parse_native_tool_spec("t0:r8:hybrid1:schema")
    generator = PersistentRacerGenerator(SimpleNamespace(), tokenizer, SimpleNamespace(), backend=backend)
    generator.configure_tool_memory(spec, checkpoint=tmp_path)
    controller = ToolRegionController(PersistentInner(), tokenizer, spec,
                                      model_context=10000, generator=generator)
    controller.prepare(_payload(), ratio=8, max_new_tokens=16)
    before = (len(backend.calls), len(backend.native_calls))
    changed = _payload()
    changed["decision_key"] = "turn-0/step-2"
    changed["messages"][0]["content"] = "Altered protected instruction."
    with pytest.raises(ValueError, match="source catalog or protected prefix changed"):
        controller.prepare(changed, ratio=8, max_new_tokens=16)
    assert (len(backend.calls), len(backend.native_calls)) == before


def test_native_tool_transport_preserves_token_source_and_rejects_wrong_positions():
    from benchmarks.backends.sglang import SglangBackend
    submitted = []
    def post(path, body, timeout):
        submitted.append((path, body))
        return {"success": True, "key_hash": "native", "original_seq_len": 3,
                "token_len": 3, "position_start": 20, "position_end": 23,
                "already_rotated": False}
    backend = SglangBackend(post)
    backend.extract_native_tool_tokens([11, 12, 13], source_start=20)
    assert submitted == [("/v1/c2kv/repair_extract", {
        "input_ids": [11, 12, 13], "span_start": 0, "span_end": 3,
        "position_offset": 0, "repair_position_ids": [20, 21, 22],
        "raw_kv_position_mode": "pre_rope", "repair_mode": "d_corr",
        "extract_source": "model_prefill"})]
    with pytest.raises(Exception, match="inconsistent source frame"):
        backend.extract_native_tool_tokens([11, 12, 13], source_start=21)
