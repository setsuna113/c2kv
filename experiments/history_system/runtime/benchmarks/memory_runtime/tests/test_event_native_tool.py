"""CPU checks for the optional native visible-tool carrier."""
from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import pytest
import sys
from types import SimpleNamespace

from history_memory.events import EventStore
from history_memory.packing import EncoderChunk, MemoryView, PackedMemory, native_ids
from history_memory.sglang_generator import SGLangEventNativeError, SGLangEventNativeGenerator
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_costs import summarize_event_native_steps
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.event_native_raw import RuntimeMemoryView, render_raw_control_messages
from benchmarks.memory_runtime.event_native_tool import (
    ToolRegionController, parse_native_tool_spec, validate_ready_tool_contract,
)


def test_on_ready_requires_loaded_tool_contract(tmp_path):
    checkpoint = tmp_path / "tool-checkpoint"
    with pytest.raises(RuntimeError, match="tool memory contract"):
        validate_ready_tool_contract({}, "t0:r8", checkpoint)
    loaded = {"tool_memory_contract": {
        "spec": parse_native_tool_spec("t0:r8").as_dict(),
        "tool_budget_tokens": None,
        "checkpoint": {"checkpoint": str(checkpoint)},
    }}
    validate_ready_tool_contract(loaded, "t0:r8", checkpoint)
    validate_ready_tool_contract({}, None)
    with pytest.raises(RuntimeError, match="checkpoint"):
        validate_ready_tool_contract(loaded, "t0:r8", tmp_path / "wrong")


class CharacterTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        rendered = "".join(f"<{message['role']}>{message['content']}" for message in messages)
        if kwargs.get("tokenize"):
            return list(map(ord, rendered))
        return rendered

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)),
                "offset_mapping": [(index, index + 1) for index in range(len(text))]}


class ToolAwareTokenizer(CharacterTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        rendered = super().apply_chat_template(messages, tokenize=False)
        if kwargs.get("tools"):
            rendered = "<tools>" + json.dumps(kwargs["tools"], sort_keys=True) + "</tools>" + rendered
        return list(map(ord, rendered)) if kwargs.get("tokenize") else rendered


class BoundaryMergingTokenizer(CharacterTokenizer):
    _merges = {"[[": 0x110000, "],": 0x110001}
    _reverse = {value: key for key, value in _merges.items()}

    def apply_chat_template(self, messages, **kwargs):
        rendered = super().apply_chat_template(messages, tokenize=False)
        return self(rendered)["input_ids"] if kwargs.get("tokenize") else rendered

    def __call__(self, text, **kwargs):
        ids, offsets = [], []
        cursor = 0
        while cursor < len(text):
            pair = text[cursor:cursor + 2]
            if pair in self._merges:
                ids.append(self._merges[pair])
                offsets.append((cursor, cursor + 2))
                cursor += 2
            else:
                ids.append(ord(text[cursor]))
                offsets.append((cursor, cursor + 1))
                cursor += 1
        return {"input_ids": ids, "offset_mapping": offsets}

    def decode(self, ids, **kwargs):
        return "".join(self._reverse[token] if token in self._reverse else chr(token)
                       for token in ids)


class HistoryController:
    kv_bytes_per_token = 1
    policy_config = SimpleNamespace(history_budget_bytes=10000,
                                    workspace_budget_bytes=10000)
    max_recovery_rounds = 1

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.seen = []

    def prepare(self, payload, *, ratio, max_new_tokens):
        self.seen.append(payload)
        system, user = payload["messages"]
        memory = PackedMemory(
            MemoryView((), ()),
            tuple(self.tokenizer.apply_chat_template([system], tokenize=True)),
            tuple(self.tokenizer.apply_chat_template([user], tokenize=True)),
            (0, 1), (),
        )
        metadata = {"decision_index": 0,
                    "common_raw_prompt_tokens": len(memory.system_input_ids) + len(memory.workspace_input_ids),
                    "actual_history_bytes": 0}
        return SimpleNamespace(memory=memory, metadata=metadata, eligible_chunks=())

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error):
        return {"regenerate": True, "memory": prepared.memory,
                "metadata": prepared.metadata,
                "decision": {"status": "recover", "reason": "cpu_test"}}


class RecordingGenerator:
    def __init__(self):
        self.memories = []
        self.requests = []

    @contextmanager
    def decision_scope(self, **kwargs):
        yield

    def generate(self, memory, **kwargs):
        self.memories.append(memory)
        self.requests.append(kwargs)
        return SimpleNamespace(token_ids=(33,), finish_reason="stop",
                               token_logprobs=(0.0,), stats={"eos_token_ids": ()})

    def session_cache_info(self):
        return {}

    def close_session(self):
        pass


@pytest.mark.parametrize("method", ["streamingllm", "h2o", "snapkv", "pyramidkv"])
def test_schema_policy_keeps_native_interfaces_through_recovery(method):
    tokenizer = CharacterTokenizer()
    inner = HistoryController(tokenizer)
    calls = []
    generator = RecordingGenerator()

    def repair(ids, **kwargs):
        calls.append(kwargs)
        return {"key_hash": "raw-tool-handle", "token_len": kwargs["target_tokens"]}

    generator.repair_tool_span = repair
    controller = ToolRegionController(
        inner, tokenizer, parse_native_tool_spec(f"{method}:r8:hybrid1:schema"),
        model_context=20000, generator=generator)
    tools = [{"type": "function", "function": {
        "name": name, "description": (name + " description ") * 40,
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                       "required": ["path"], "additionalProperties": False}}}
        for name in ("cp", "ls")]
    payload = {"session_id": "protected-tools", "tools": tools,
               "messages": [{"role": "system", "content": "Use APIs."},
                            {"role": "user", "content": "cp a file"}]}
    prepared = controller.prepare(payload, ratio=8, max_new_tokens=4)
    assert inner.seen[0]["tools"] == []
    assert len(calls) == 1
    call = calls[0]
    protocol_tokens = prepared.memory.system_input_ids[call["span_start"]:call["span_end"]]
    protected = tokenizer.decode([token for index, token in enumerate(protocol_tokens)
                                  if index not in call["selectable_relative_indices"]])
    assert '"name":"cp"' in protected and '"name":"ls"' in protected
    assert '"required":["path"]' in protected
    assert prepared.metadata["tool_memory"]["resident_tool_tokens"] == call["target_tokens"]
    recovered = controller.reconsider(prepared, [], draft_text="retry")
    assert len(calls) == 2 and calls[1] == calls[0]
    assert recovered["memory"].raw_tool_segments == prepared.memory.raw_tool_segments


def test_t0_schema_policy_preserves_original_encoder_chunks_and_charges_interfaces():
    tokenizer = CharacterTokenizer()
    payload = {"session_id": "t0-interfaces", "tools": [{"type": "function", "function": {
        "name": "cp", "description": "copy files", "parameters": {
            "type": "object", "properties": {"source": {"type": "string"}},
            "required": ["source"]}}}], "messages": [
                {"role": "system", "content": "Use APIs."},
                {"role": "user", "content": "Copy the file."}]}
    prepared = []
    for spec in ("t0:r8", "t0:r8:schema"):
        controller = ToolRegionController(HistoryController(tokenizer), tokenizer,
            parse_native_tool_spec(spec), model_context=10000, generator=RecordingGenerator())
        prepared.append(controller.prepare(payload, ratio=8, max_new_tokens=4))
    assert prepared[0].memory.chunks == prepared[1].memory.chunks
    assert prepared[1].memory.costs(8)["resident_kv_tokens"] > prepared[0].memory.costs(8)["resident_kv_tokens"]
    assert '"required":["source"]' in tokenizer.decode(prepared[1].memory.system_input_ids)


def test_source_tool_carrier_survives_native_recovery_and_final_trace(tmp_path):
    tokenizer = CharacterTokenizer()
    inner = HistoryController(tokenizer)
    generator = RecordingGenerator()
    controller = ToolRegionController(inner, tokenizer,
                                      parse_native_tool_spec("t0:r8"),
                                      model_context=10000, generator=generator)
    runner = EventNativeDecisionRunner(
        controller, generator, tokenizer, ratio=8, max_new_tokens=4,
        max_generation_calls=2, journal=AttemptJournal(tmp_path / "attempts.jsonl"),
    )
    definition = "api_docs.show_bank() => bank.transfer(amount, recipient)"
    payload = {
        "session_id": "episode-1", "decision_key": "turn-1",
        "messages": [{"role": "system", "content": "Use visible APIs: " + definition},
                     {"role": "user", "content": "Transfer money"}],
        "tools": [],
        "c2kv_tool_spans_v1": [{"message_index": 0, "start": len("Use visible APIs: "),
                                "end": len("Use visible APIs: ") + len(definition),
                                "source": "appworld_api_docs"}],
    }
    result = runner.run(payload)
    assert result["status"] == "ok"
    assert len(generator.memories) == 2
    original = EventStore.from_messages(payload["session_id"], payload["messages"])
    full_view = RuntimeMemoryView(
        gist_event_ids=(),
        raw_event_ids=tuple(event.event_id for event in original.events),
        evidence_event_ids=(), raw_control_layout="full-original-native-v1")
    expected_full = len(native_ids(
        tokenizer, render_raw_control_messages(original, full_view),
        tools=payload["tools"], generation=True))
    assert all(request["paper_whole_full_kv_tokens"] == expected_full
               for request in generator.requests)
    assert all(memory.tool_gist_segments for memory in generator.memories)
    assert generator.memories[0].tool_gist_segments == generator.memories[1].tool_gist_segments
    final = result["generation_trace"][-1]["prepared_input"]
    assert final["tool_gist_segments"]
    assert final["tool_gist_segments"][0]["chunks"][0]["projection_set"] == "tool"
    assert inner.seen[0]["messages"][0]["content"].startswith("Use visible APIs: [Tool definition ")
    assert "c2kv_tool_spans_v1" not in inner.seen[0]


def test_off_spec_does_not_create_tool_controller_or_change_payload():
    assert parse_native_tool_spec(None) is None
    assert parse_native_tool_spec("none") is None


def test_full_denominator_counts_original_structured_tools_before_t0_rewrite():
    tokenizer = ToolAwareTokenizer()
    inner = HistoryController(tokenizer)
    controller = ToolRegionController(
        inner, tokenizer, parse_native_tool_spec("t0:r8"),
        model_context=10000, generator=RecordingGenerator())
    payload = {
        "session_id": "with-tools", "messages": [
            {"role": "system", "content": "Use the API"},
            {"role": "user", "content": "Look up weather"}],
        "tools": [{"type": "function", "function": {
            "name": "get_weather", "description": "Weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    }
    prepared = controller.prepare(payload, ratio=4, max_new_tokens=4)
    full_with_tools = controller._whole_full_tokens(payload)
    full_without_tools = controller._whole_full_tokens(dict(payload, tools=[]))
    assert prepared.metadata["paper_whole_full_kv_tokens"] == full_with_tools
    assert full_with_tools > full_without_tools
    assert inner.seen[0]["tools"] == []


def test_embedded_json_tool_anchors_preserve_adjacent_bpe_text():
    tokenizer = BoundaryMergingTokenizer()
    inner = HistoryController(tokenizer)
    controller = ToolRegionController(
        inner, tokenizer, parse_native_tool_spec("t0:r8"),
        model_context=10000, generator=RecordingGenerator())
    docs = [json.dumps({"name": name, "description": "A visible function"})
            for name in ("turn_on_wifi", "login_device")]
    content = "Available APIs: [" + ", ".join(docs) + "]"
    spans = [{"message_index": 0, "start": content.index(doc),
              "end": content.index(doc) + len(doc), "source": "acebench.functions"}
             for doc in docs]
    payload = {"session_id": "ace-json", "messages": [
        {"role": "system", "content": content},
        {"role": "user", "content": "Enable WiFi"}],
        "tools": [], "c2kv_tool_spans_v1": spans}
    prepared = controller.prepare(payload, ratio=4, max_new_tokens=4)
    rewritten_system = inner.seen[0]["messages"][0]
    before = tokenizer.apply_chat_template([rewritten_system], tokenize=False)
    after = tokenizer.decode(prepared.memory.system_input_ids)
    assert before == after
    assert len(prepared.memory.tool_gist_segments) == 2
    assert "Available APIs: [" in after and ", " in after and after.endswith("]")
    for segment in prepared.memory.tool_gist_segments:
        left, right = segment["token_start"], segment["token_end"]
        assert tokenizer.decode(prepared.memory.system_input_ids[left:right]).startswith(
            "[Tool definition ")


def test_off_native_request_has_the_original_exact_fields(tmp_path):
    generator = SGLangEventNativeGenerator(
        "http://127.0.0.1:30000", expected_model_path=tmp_path,
        model_context=100, max_new_tokens=4, max_generation_calls=1,
        max_extraction_calls=2, timeout_seconds=1,
        eos_token_ids=(0,), eos_source="cpu-test",
    )
    generator._model_binding = {"model_path": str(tmp_path)}
    generator._kv_bytes_per_token = 1
    observed = {}

    def capture(payload, *args, **kwargs):
        observed.update(payload)
        raise RuntimeError("request captured")

    generator._post_native_generate = capture
    memory = PackedMemory(MemoryView((), ()), (11, 12), (13, 14), (0,), ())
    try:
        generator.generate(memory, ratio=8, max_new_tokens=2)
    except RuntimeError as error:
        assert str(error) == "request captured"
    else:
        raise AssertionError("native request was not submitted")
    for dynamic in ("rid", "generation_id"):
        observed.pop(dynamic)
    assert json.dumps(observed, sort_keys=True, separators=(",", ":")) == json.dumps({
        "schema": "c2kv-native-packed-generation-v1",
        "session_id": None,
        "packing_version": "history-event-v1",
        "raw_layout_profile": "event-native-evidence-v1",
        "encoding_scope": "current",
        "system_input_ids": [11, 12],
        "workspace_input_ids": [13, 14],
        "encoder_chunks": [],
        "compression_chunks": [],
        "compression_ratio": 8,
        "max_extraction_calls": 2,
        "sampling_params": {"temperature": 0.0, "sampling_seed": 0,
                            "max_new_tokens": 2, "stop_token_ids": [0]},
        "shadow_features": None,
    }, sort_keys=True, separators=(",", ":"))


def test_on_native_request_sends_raw_full_denominator(tmp_path):
    generator = SGLangEventNativeGenerator(
        "http://127.0.0.1:30000", expected_model_path=tmp_path,
        model_context=100, max_new_tokens=4, max_generation_calls=1,
        max_extraction_calls=2, timeout_seconds=1,
        eos_token_ids=(0,), eos_source="cpu-test",
    )
    generator._model_binding = {"model_path": str(tmp_path)}
    generator._kv_bytes_per_token = 1
    observed = {}

    def capture(payload, *args, **kwargs):
        observed.update(payload)
        raise RuntimeError("request captured")

    generator._post_native_generate = capture
    memory = PackedMemory(MemoryView((), ()), (11, 12), (13, 14), (0,), ())
    with pytest.raises(RuntimeError, match="request captured"):
        generator.generate(memory, ratio=8, max_new_tokens=2,
                           paper_whole_full_kv_tokens=123)
    assert observed["paper_whole_full_kv_tokens"] == 123


def test_ace_anchored_t0_response_uses_physical_prefix_in_cost_summary(tmp_path):
    # Shape and costs from the first successful ACE native T0 response: the
    # logical 784-token system contains 340 placeholder tokens replaced by gist KV.
    spans_and_lengths = (
        (413, 436, 45), (438, 459, 45), (461, 484, 109),
        (486, 507, 93), (509, 533, 67), (535, 557, 93),
        (559, 580, 51), (582, 606, 52), (608, 630, 103),
        (632, 655, 67), (657, 680, 69), (682, 704, 83),
        (706, 730, 63), (732, 754, 161), (756, 781, 72),
    )
    segments = tuple({
        "token_start": start, "token_end": end,
        "chunks": (EncoderChunk(
            event_id=f"ace-visible-tool-{index}", part_index=0,
            source_indices=(index,), source_token_start=0,
            source_token_end=length, token_ids=(100 + index,) * length,
            projection_set="tool", compression_ratio=8),),
    } for index, (start, end, length) in enumerate(spans_and_lengths))
    memory = PackedMemory(MemoryView((), ()), (11,) * 784, (12,) * 87,
                          (0, 1), (), tool_gist_segments=segments)
    assert memory.costs(8)["resident_kv_tokens"] == 684
    generator = SGLangEventNativeGenerator(
        "http://127.0.0.1:30000", expected_model_path=tmp_path,
        model_context=4096, max_new_tokens=4, max_generation_calls=1,
        max_extraction_calls=32, max_tool_extraction_calls=32,
        timeout_seconds=1, eos_token_ids=(0,), eos_source="cpu-test",
    )
    generator._model_binding = {"model_path": str(tmp_path)}
    generator._tool_projection_identity = "ace-tool-projection"
    generator._kv_bytes_per_token = 147456
    selected, extras, _ = generator._prepare_chunks(memory, 8, None)
    rows = [{"chunk_id": chunk["chunk_id"], "handle": chunk["handle"],
             "cache_key": f"cached-{index}", "cache_hit": True,
             "original_seq_len": len(chunk["token_ids"]),
             "gist_len": (len(chunk["token_ids"]) + 7) // 8}
            for index, chunk in enumerate(selected)]
    costs = {
        "system_tokens": 784, "raw_tokens": 87,
        "presented_encoder_tokens": 1173, "gist_tokens": 153,
        "resident_kv_tokens": 684, "system_prefix_kv_tokens": 444,
        "gist_prefix_kv_tokens": 153, "workspace_resident_kv_tokens": 87,
        "system_prefix_kv_logical_bytes": 65470464,
        "gist_prefix_kv_logical_bytes": 22560768,
        "raw_workspace_kv_logical_bytes": 12828672,
        "resident_kv_logical_bytes": 100859904,
        "anchored_tool_source_tokens": 340,
        "anchored_tool_gist_tokens": 153,
    }
    payload = {"rid": "ace-native-1", "generation_id": "draft-1",
               "session_id": "agent_multi_step_0", "max_extraction_calls": 32,
               "max_tool_extraction_calls": 32}
    response = {
        "schema": "c2kv-native-packed-generation-response-v1", **{
            name: payload[name] for name in ("rid", "generation_id", "session_id")},
        "text": "done", "output_ids": [42], "token_logprobs": [0.0],
        "finish_reason": "stop", "encoder_chunks": rows,
        "compression_chunks": [], "costs": costs,
        "extraction": {"requested_chunks": len(rows), "unique_chunks": len(rows),
                       "cache_hits": len(rows), "cache_misses": 0,
                       "model_calls": 0, "history_model_calls": 0,
                       "tool_model_calls": 0, "max_extraction_calls": 32,
                       "max_tool_extraction_calls": 32},
    }
    generation = generator._result_from_response(
        response, payload=payload, memory=memory, selected=selected,
        extras=extras, ratio=8, requested_tokens=4, request_index=1,
        http_status=200, wall_seconds=0.1, scope=None, effective_eos_ids=(0,),
    )
    stats = generation.stats
    assert stats["system_prefix_kv_tokens"] == 444
    assert stats["resident_prefix_kv_tokens"] == 597
    assert stats["resident_prefix_kv_bytes"] == 88031232
    trace = {"phase": "draft", "status": "completed", "discarded": False,
             "attempt_uid": "ace-native-1", "attempt_index": 1,
             "generation": {"stats": stats}}
    summary = summarize_event_native_steps({
        "session_id": payload["session_id"], "decision_key": "turn-0",
        "status": "ok", "generation_trace": [trace],
    })
    assert summary["costs"]["logical_kv_peaks"]["resident_prefix_kv_logical_bytes"]["strict_peak"] == 88031232
    bad = dict(costs, system_prefix_kv_logical_bytes=784 * 147456)
    with pytest.raises(SGLangEventNativeError, match="system_prefix_kv_logical_bytes mismatch"):
        generator._validate_costs(bad, memory, 8)
    bad = dict(costs, workspace_resident_kv_tokens=88)
    with pytest.raises(SGLangEventNativeError, match="physical KV components"):
        generator._validate_costs(bad, memory, 8)


def test_off_cost_response_keeps_logical_prefix_fallback(tmp_path):
    generator = SGLangEventNativeGenerator(
        "http://127.0.0.1:30000", expected_model_path=tmp_path,
        model_context=100, max_new_tokens=4, max_generation_calls=1,
        max_extraction_calls=2, timeout_seconds=1,
        eos_token_ids=(0,), eos_source="cpu-test",
    )
    generator._model_binding = {"model_path": str(tmp_path)}
    generator._kv_bytes_per_token = 16
    memory = PackedMemory(MemoryView((), ()), (11, 12), (13, 14, 15), (0,), ())
    costs = generator._validate_costs({
        "system_tokens": 2, "raw_tokens": 3,
        "presented_encoder_tokens": 0, "gist_tokens": 0,
        "resident_kv_tokens": 5, "system_prefix_kv_logical_bytes": 32,
        "gist_prefix_kv_logical_bytes": 0,
        "raw_workspace_kv_logical_bytes": 48,
        "resident_kv_logical_bytes": 80,
    }, memory, 8)
    assert (costs["system_prefix_kv_tokens"],
            costs["gist_prefix_kv_tokens"],
            costs["workspace_resident_kv_tokens"]) == (2, 0, 3)


def test_t0_native_server_requires_the_loaded_tool_projection(tmp_path):
    tool = tmp_path / "T0"
    contract = {"checkpoint": str(tool), "config_sha256": "a" * 64}
    def generator():
        return SGLangEventNativeGenerator(
            "http://127.0.0.1:30000", expected_model_path=tmp_path,
            model_context=100, max_new_tokens=4, max_generation_calls=1,
            max_extraction_calls=2, timeout_seconds=1,
            eos_token_ids=(0,), eos_source="cpu-test",
            expected_tool_checkpoint_contract=contract,
        )

    native = {"model_binding": {"model_path": str(tmp_path)},
              "kv_bytes_per_token": 1,
              "tool_gist": {"enabled": True, "source": str(tool),
                            "identity": "test-tool-projection-identity",
                            "config_sha256": "a" * 64,
                            "extract_projection_set": "tool"}}
    g = generator()
    g._read_json = lambda *args, **kwargs: ({"model_path": str(tmp_path),
                                             "c2kv_native_packed": native}, 200)
    g._ensure_model_info()
    native["tool_gist"]["config_sha256"] = "b" * 64
    other = generator()
    other._read_json = g._read_json
    with pytest.raises(SGLangEventNativeError, match="config hash differs"):
        other._ensure_model_info()
    native["tool_gist"]["config_sha256"] = "a" * 64
    native["tool_gist"].pop("identity")
    with pytest.raises(SGLangEventNativeError, match="projection identity is missing"):
        other._ensure_model_info()


def test_tool_chunk_handle_matches_the_engine_contract():
    paper_root = Path(__file__).resolve().parents[6]
    sources = [paper_root.parent / name / "python/sglang/srt/mem_cache/c2kv_native_packed.py"
               for name in ("sglang-paper-tool-integration", "engine")]
    engine_source = next((path for path in sources if path.is_file()), None)
    if engine_source is None:
        pytest.skip("shared SGLang engine source is not beside the paper checkout")
    module_spec = importlib.util.spec_from_file_location("c2kv_engine_handle_contract_test", engine_source)
    engine = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = engine
    module_spec.loader.exec_module(engine)

    binding = {"model_path": "/checkpoint", "weight_version": "test"}
    identity = "tool-projection-test-identity"
    generator = object.__new__(SGLangEventNativeGenerator)
    generator._model_binding = binding
    generator._tool_projection_identity = identity
    generator.encoding_scope = "current"
    for projection, ratio in (("tool", 8), (None, 4)):
        chunk = EncoderChunk(
            event_id="visible-tool" if projection else "history-event",
            part_index=0, source_indices=(0,), source_token_start=0,
            source_token_end=4, token_ids=(11, 12, 13, 14),
            projection_set=projection,
            compression_ratio=ratio if projection else None,
        )
        client = generator._chunk_payload(chunk, ratio)
        expected = engine.canonical_chunk_handle(
            client, model_binding=binding, packing_version="history-event-v1",
            encoding_scope="current", compression_ratio=ratio,
            tool_binding={"enabled": True, "identity": identity} if projection else None,
        )
        assert client["handle"] == expected


@pytest.mark.parametrize("spec", ["t0:r8:hybrid2:schema", "h2o:r8:hybrid2:schema"])
def test_native_source_topk_is_protected_even_without_encoder_work(spec):
    tokenizer = BoundaryMergingTokenizer()
    inner = HistoryController(tokenizer)
    definition = "opaque_api(x) => execute exactly"
    content = "Visible: " + definition
    payload = {"session_id": "native-source", "messages": [
        {"role": "system", "content": content},
        {"role": "user", "content": "Execute"}], "tools": [],
        "c2kv_tool_spans_v1": [{"message_index": 0, "start": len("Visible: "),
            "end": len(content), "source": "appworld_api_docs"}]}
    controller = ToolRegionController(inner, tokenizer, parse_native_tool_spec(spec),
        model_context=10000, generator=RecordingGenerator())
    prepared = controller.prepare(payload, ratio=8, max_new_tokens=4)
    assert definition in prepared.plan.protocol
    assert prepared.plan.info["n_native_source_interface_copies"] == 1
    assert not prepared.memory.raw_tool_segments and not prepared.memory.chunks
    assert prepared.metadata["tool_memory"]["resident_tool_tokens"] > len(definition)
    if spec.startswith("h2o"):
        assert prepared.plan.info["selection_backend"] == "raw_full_no_repair"
        assert prepared.plan.info["retained_source_span_count"] == 1
        assert prepared.plan.info["raw_source_fallback_count"] == 0
