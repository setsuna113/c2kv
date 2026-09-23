"""CPU checks for native cross-turn history prewarming."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from history_memory.cross_turn_prewarm import plan_cross_turn_chunks
from history_memory.events import EventStore
from history_memory.packing import encode_scope_chunks
from history_memory.sglang_generator import (
    SGLangEventNativeError, SGLangEventNativeGenerator, SGLangTransportError,
)
from benchmarks.memory_runtime.adapter import raw_source_cutoff
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        rendered = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return list(rendered.encode("utf-8"))


def _plan(messages, *, scope="current", benchmark=None):
    store = EventStore.from_messages("s", messages, benchmark=benchmark)
    return plan_cross_turn_chunks(
        store, Tokenizer(), encoding_scope=scope,
        max_chunk_tokens=768, chunk_overlap=64,
        atomic_unit_token_limit=8192,
        source_cutoff=raw_source_cutoff(messages), benchmark=benchmark,
    )


def _generator(monkeypatch, tmp_path):
    monkeypatch.setenv("C2KV_NATIVE_CROSS_TURN_PREWARM", "1")
    generator = SGLangEventNativeGenerator(
        "http://127.0.0.1:30000", expected_model_path=tmp_path,
        model_context=8192, max_new_tokens=16, max_generation_calls=4,
        max_extraction_calls=4, timeout_seconds=1,
        eos_token_ids=(0,), eos_source="cpu-test",
    )
    generator._model_binding = {"model_path": str(tmp_path)}
    generator._kv_bytes_per_token = 1
    return generator


@pytest.mark.parametrize("scope", ["current", "event"])
def test_current_raw_user_is_exact_next_turn_history_chunk(scope):
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]
    planned = _plan(messages, scope=scope)
    assert planned and {chunk.event_id for chunk in planned} == {"s:m3"}
    next_messages = [*messages, {"role": "assistant", "content": "reply"},
                     {"role": "user", "content": "third"}]
    store = EventStore.from_messages("s", next_messages)
    historical = encode_scope_chunks(
        store, ("s:m3",), Tokenizer(), encoding_scope=scope,
        max_chunk_tokens=768, chunk_overlap=64,
        atomic_unit_token_limit=8192,
    )
    assert planned == historical
    generator = SGLangEventNativeGenerator(
        "http://127.0.0.1:30000", expected_model_path="/checkpoint",
        model_context=8192, max_new_tokens=16, max_generation_calls=4,
        max_extraction_calls=4, timeout_seconds=1,
        eos_token_ids=(0,), eos_source="cpu-test", encoding_scope=scope,
    )
    generator._model_binding = {"model_path": "/checkpoint"}
    assert [generator._chunk_payload(chunk, 8)["handle"] for chunk in planned] == [
        generator._chunk_payload(chunk, 8)["handle"] for chunk in historical]


def test_pending_tool_event_and_appworld_task_packet_are_not_prepared():
    pending = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "work"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call1", "type": "function",
             "function": {"name": "lookup", "arguments": "{}"}}]},
    ]
    assert {chunk.event_id for chunk in _plan(pending)} == {"s:m1"}
    assert _plan(pending, benchmark="acon_appworld") == ()


@pytest.mark.parametrize("next_session", ["s", "other"])
def test_submit_ack_does_not_drain_and_next_scope_charges_cancelled_tail(
    monkeypatch, tmp_path, next_session):
    generator = _generator(monkeypatch, tmp_path)
    chunks = _plan([{"role": "system", "content": "rules"},
                    {"role": "user", "content": "w" * 1000}])
    assert len(chunks) >= 2
    calls = []

    def fake_request(payload):
        calls.append(dict(payload))
        base = {"schema": "c2kv-native-prewarm-response-v1",
                "owner_id": payload["owner_id"], "job_id": payload["job_id"],
                "submitted_chunks": 2}
        if payload["operation"] == "submit":
            return {**base, "status": "queued"}
        handles = generator._prewarm_job["submitted_handles"]
        return {**base, "status": "cancelled", "budget_known": True,
                "completed_chunks": 1, "cancelled_chunks": 1,
                "cache_hits": 0, "model_calls": 1,
                "extraction": {"model_calls": 1, "history_model_calls": 1,
                               "tool_model_calls": 0},
                "results": [{"handle": handles[0], "cache_hit": False}]}

    generator._prewarm_request = fake_request
    submitted = generator.submit_cross_turn_prewarm(
        chunks[:2], ratio=8, session_id="s", outer_request_id="outer")
    assert submitted["status"] == "queued"
    assert [item["operation"] for item in calls] == ["submit"]
    assert generator.extraction_calls_reserved == 0
    with generator.decision_scope(session_id=next_session):
        assert generator.extraction_calls_reserved == 1
        assert generator._prewarm_job is None
    assert [item["operation"] for item in calls] == ["submit", "drain"]
    assert generator.session_cache_info()["cross_turn_prewarm"]["last_receipt"]["status"] == "cancelled"
    assert generator.reconcile_cross_turn_prewarm() is None
    assert generator.extraction_calls_reserved == 1


def test_close_session_cancels_outstanding_job(monkeypatch, tmp_path):
    generator = _generator(monkeypatch, tmp_path)
    chunks = _plan([{"role": "user", "content": "work"}])
    operations = []

    def fake_request(payload):
        operations.append(payload["operation"])
        base = {"schema": "c2kv-native-prewarm-response-v1",
                "owner_id": payload["owner_id"], "job_id": payload["job_id"],
                "submitted_chunks": 1}
        if payload["operation"] == "submit":
            return {**base, "status": "queued"}
        return {**base, "status": "cancelled", "budget_known": True,
                "completed_chunks": 0, "cancelled_chunks": 1,
                "cache_hits": 0, "model_calls": 0,
                "extraction": {"model_calls": 0, "history_model_calls": 0,
                               "tool_model_calls": 0}, "results": []}

    generator._prewarm_request = fake_request
    generator.submit_cross_turn_prewarm(chunks, ratio=8, session_id="s", outer_request_id="outer")
    generator.close_session()
    assert operations == ["submit", "cancel"]
    assert generator._prewarm_job is None


def test_lost_submit_ack_keeps_identity_for_session_cleanup(monkeypatch, tmp_path):
    generator = _generator(monkeypatch, tmp_path)
    chunks = _plan([{"role": "user", "content": "work"}])
    operations = []

    def fake_request(payload):
        operations.append(payload["operation"])
        if payload["operation"] == "submit":
            raise SGLangTransportError("submit ACK lost")
        return {"schema": "c2kv-native-prewarm-response-v1",
                "owner_id": payload["owner_id"], "job_id": payload["job_id"],
                "status": "cancelled", "budget_known": True,
                "submitted_chunks": 1, "completed_chunks": 0,
                "cancelled_chunks": 1, "cache_hits": 0, "model_calls": 0,
                "extraction": {"model_calls": 0, "history_model_calls": 0,
                               "tool_model_calls": 0}, "results": []}

    generator._prewarm_request = fake_request
    with pytest.raises(SGLangTransportError, match="ACK lost"):
        generator.submit_cross_turn_prewarm(
            chunks, ratio=8, session_id="s", outer_request_id="outer")
    assert generator._prewarm_job is not None
    generator.close_session()
    assert operations == ["submit", "cancel"]
    assert generator._prewarm_job is None
    assert generator.session_cache_info()["cross_turn_prewarm"]["budget_known"] is True


def test_already_extracted_handle_is_excluded(monkeypatch, tmp_path):
    generator = _generator(monkeypatch, tmp_path)
    chunks = _plan([{"role": "user", "content": "w" * 1000}])
    first = generator._chunk_payload(chunks[0], 8)["handle"]
    generator._last_decision_extracted_handles = {first}
    seen = []

    def fake_request(payload):
        seen.append(payload)
        return {"schema": "c2kv-native-prewarm-response-v1",
                "owner_id": payload["owner_id"], "job_id": payload["job_id"],
                "status": "queued", "submitted_chunks": len(payload["chunks"])}

    generator._prewarm_request = fake_request
    generator.submit_cross_turn_prewarm(
        chunks[:2], ratio=8, session_id="s", outer_request_id="outer")
    assert len(seen[0]["chunks"]) == 1
    assert seen[0]["chunks"][0]["handle"] != first


def test_unknown_background_budget_blocks_next_decision(monkeypatch, tmp_path):
    generator = _generator(monkeypatch, tmp_path)
    chunks = _plan([{"role": "user", "content": "work"}])

    def fake_request(payload):
        base = {"schema": "c2kv-native-prewarm-response-v1",
                "owner_id": payload["owner_id"], "job_id": payload["job_id"],
                "submitted_chunks": 1}
        if payload["operation"] == "submit":
            return {**base, "status": "queued"}
        return {**base, "status": "failed", "budget_known": False}

    generator._prewarm_request = fake_request
    generator.submit_cross_turn_prewarm(chunks, ratio=8, session_id="s", outer_request_id="outer")
    with pytest.raises(SGLangEventNativeError, match="budget is unknown"):
        with generator.decision_scope(session_id="s"):
            pass
    assert generator.extraction_calls_reserved == 0
    assert generator.session_cache_info()["cross_turn_prewarm"]["budget_known"] is False
    with pytest.raises(SGLangEventNativeError, match="budget is unknown"):
        with generator.decision_scope(session_id="s"):
            pass


def test_default_off_has_no_prewarm_request(monkeypatch, tmp_path):
    monkeypatch.delenv("C2KV_NATIVE_CROSS_TURN_PREWARM", raising=False)
    generator = SGLangEventNativeGenerator(
        "http://127.0.0.1:30000", expected_model_path=tmp_path,
        model_context=8192, max_new_tokens=16, max_generation_calls=4,
        max_extraction_calls=4, timeout_seconds=1,
        eos_token_ids=(0,), eos_source="cpu-test",
    )
    generator._prewarm_request = lambda payload: pytest.fail("prewarm request while disabled")
    chunks = _plan([{"role": "user", "content": "work"}])
    assert generator.submit_cross_turn_prewarm(
        chunks, ratio=8, session_id="s", outer_request_id="outer") is None
    with generator.decision_scope(session_id="s"):
        pass
    generator.close_session()
    assert "cross_turn_prewarm" not in generator.session_cache_info()


def test_enabled_client_requires_engine_feature(monkeypatch, tmp_path):
    generator = _generator(monkeypatch, tmp_path)
    generator._model_binding = None
    native = {"model_binding": {"model_path": str(tmp_path)},
              "kv_bytes_per_token": 1, "serving_features": {}}
    generator._read_json = lambda *args, **kwargs: (
        {"model_path": str(tmp_path), "c2kv_native_packed": native}, 200)
    with pytest.raises(SGLangEventNativeError, match="cross-turn-prewarm-v1 admission"):
        generator._ensure_model_info()
    native["serving_features"]["cross_turn_prewarm"] = "cross-turn-prewarm-v1"
    generator._ensure_model_info()
    assert generator._model_binding == native["model_binding"]


def test_runner_plans_after_commit_without_repreparing_policy():
    messages = [{"role": "system", "content": "rules"},
                {"role": "user", "content": "latest"}]
    class Controller:
        benchmark = "bfcl"
        packing = SimpleNamespace(max_chunk_tokens=768, chunk_overlap=64,
                                  max_encoder_tokens=8192, max_sequence_tokens=8192)

        def prepare(self, *args, **kwargs):
            pytest.fail("speculative controller.prepare is forbidden")

    seen = []
    generator = SimpleNamespace(
        encoding_scope="current", model_context=8192,
        submit_cross_turn_prewarm=lambda chunks, **kwargs:
            seen.append((chunks, kwargs)) or {"status": "queued"},
    )
    runner = object.__new__(EventNativeDecisionRunner)
    runner.controller, runner.generator = Controller(), generator
    runner.tokenizer, runner.ratio = Tokenizer(), 8
    result = runner._submit_cross_turn_prewarm(
        {"session_id": "s", "messages": messages}, None,
        {"outer_request_id": "outer"})
    assert result["status"] == "queued"
    assert seen[0][0] == _plan(messages)
    assert seen[0][1] == {"ratio": 8, "session_id": "s", "outer_request_id": "outer"}
    changed = [messages[0], {"role": "user", "content": "rewritten"}]
    skipped = runner._submit_cross_turn_prewarm(
        {"session_id": "s", "messages": messages},
        SimpleNamespace(plan=SimpleNamespace(messages=changed)),
        {"outer_request_id": "outer"})
    assert skipped["reason"] == "current_source_rendering_changed"
    assert len(seen) == 1
