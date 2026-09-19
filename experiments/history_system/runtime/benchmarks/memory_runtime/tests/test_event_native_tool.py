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
