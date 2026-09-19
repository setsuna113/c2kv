"""CPU checks for the optional native visible-tool carrier."""
from __future__ import annotations

from contextlib import contextmanager
import json
import pytest
from types import SimpleNamespace

from history_memory.packing import MemoryView, PackedMemory
from history_memory.sglang_generator import SGLangEventNativeError, SGLangEventNativeGenerator
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
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

    @contextmanager
    def decision_scope(self, **kwargs):
        yield

    def generate(self, memory, **kwargs):
        self.memories.append(memory)
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
