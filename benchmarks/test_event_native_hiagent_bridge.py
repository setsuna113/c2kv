"""CPU contract tests for the opt-in native HiAgent phase bridge."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import proxy
from arms import get_arm
from backends.event_native_hiagent import (
    EventNativeHiAgentBackend,
    NativeHiAgentBridgeClient,
)
from memory_runtime.attempt_journal import AttemptJournal
from memory_runtime.generation_budget import TaskGenerationBudgetExceeded
from native_hiagent_protocol import (
    POLICY_SAMPLING_SCHEMA,
    RECEIPT_FIELD,
    RECEIPT_SCHEMA,
    validate_policy_sampling,
)


POLICY_SAMPLING = {
    "temperature": 0.001,
    "seed": 42,
    "max_completion_tokens": 4096,
    "top_p": 1.0,
}
CONTEXT = {
    "benchmark": "bfcl",
    "task_id": "multi_turn_base_17",
    "user_turn": 2,
    "step": 3,
    "attempt": 0,
}
MESSAGES = [
    {"role": "system", "content": "Keep native rows."},
    {"role": "assistant", "content": "Subgoal: inspect", "tool_calls": [{
        "id": "call_a",
        "type": "function",
        "function": {"name": "inspect", "arguments": '{"id":17}'},
    }]},
    {
        "role": "tool",
        "name": "inspect",
        "tool_call_id": "call_a",
        "content": "result=17",
    },
    {"role": "user", "content": "Continue."},
]
TOOLS = [{
    "type": "function",
    "function": {
        "name": "inspect",
        "description": "Inspect an item.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
    },
}]


def _payload(**overrides):
    value = {
        "model": "c2kv-agent",
        "messages": MESSAGES,
        "tools": TOOLS,
        "temperature": 0.8,
        "max_completion_tokens": 7,
    }
    value.update(overrides)
    return json.loads(json.dumps(value))


def _receipt(envelope):
    return {
        "schema": RECEIPT_SCHEMA,
        "parent_request_id": envelope["parent_request_id"],
        "official_eval_context": envelope["official_eval_context"],
        "proxy_attempt_uid": envelope["proxy_attempt_uid"],
        "native_attempt_uid": "native-attempt-1",
        "phase": envelope["phase"],
        "call_ordinal": envelope["call_ordinal"],
        "native_session_id": envelope["native_session_id"],
        "native_decision_key": envelope["native_decision_key"],
    }


def _response(envelope):
    return {
        "choices": [{
            "message": {"role": "assistant", "content": "Subgoal: continue"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
        RECEIPT_FIELD: _receipt(envelope),
    }


class _Response:
    status = 200

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.value).encode("utf-8")


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_policy_sampling_is_explicit_and_accepts_configured_temperature(tmp_path):
    with pytest.raises(ValueError, match="missing explicit fields"):
        validate_policy_sampling({"temperature": 0.001})
    path = tmp_path / "policy_sampling.json"
    path.write_text(json.dumps({
        "schema": POLICY_SAMPLING_SCHEMA,
        **POLICY_SAMPLING,
    }), encoding="utf-8")
    client = NativeHiAgentBridgeClient.from_policy_sampling_file(path)
    assert client.policy_sampling == POLICY_SAMPLING


def test_phase_envelopes_preserve_native_rows_and_use_separate_sessions():
    client = NativeHiAgentBridgeClient(POLICY_SAMPLING)
    original = _payload()
    original_snapshot = json.loads(json.dumps(original))
    _, first = client.prepare_call(
        original,
        parent_request_id="proxy-request-1",
        official_eval_context=CONTEXT,
        proxy_attempt_uid="proxy-attempt-1",
        phase="policy",
    )
    _, second = client.prepare_call(
        original,
        parent_request_id="proxy-request-1",
        official_eval_context=CONTEXT,
        proxy_attempt_uid="proxy-attempt-2",
        phase="trajectory_retrieval_policy",
    )
    assert first["model_call"]["messages"] == MESSAGES
    assert first["model_call"]["tools"] == TOOLS
    assert first["model_call"]["sampling"] == POLICY_SAMPLING
    assert first["official_eval_context"] == CONTEXT
    assert [first["call_ordinal"], second["call_ordinal"]] == [1, 2]
    assert first["native_session_id"] != second["native_session_id"]
    assert original == original_snapshot

    compressor = {
        "model": "c2kv-agent",
        "messages": [{"role": "user", "content": "Summarize."}],
        "tools": [],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": 100,
        "stop": ["\n\n"],
    }
    _, third = client.prepare_call(
        compressor,
        parent_request_id="proxy-request-1",
        official_eval_context=CONTEXT,
        proxy_attempt_uid="proxy-attempt-3",
        phase="compressor",
    )
    assert third["model_call"]["sampling"] == {
        "temperature": 0.0,
        "seed": 42,
        "max_completion_tokens": 100,
        "stop": ["\n\n"],
        "top_p": 1.0,
    }


def test_receipt_join_and_backend_are_strictly_opt_in():
    client = NativeHiAgentBridgeClient(POLICY_SAMPLING)
    payload = _payload()
    _, envelope = client.prepare_call(
        payload,
        parent_request_id="proxy-request-1",
        official_eval_context=CONTEXT,
        proxy_attempt_uid="proxy-attempt-1",
        phase="policy",
    )
    result = client.validate_response(_response(envelope), envelope)
    assert result[RECEIPT_FIELD]["official_eval_context"] == CONTEXT
    changed = _response(envelope)
    changed[RECEIPT_FIELD]["call_ordinal"] = 2
    with pytest.raises(ValueError, match="changes call_ordinal"):
        client.validate_response(changed, envelope)

    backend = EventNativeHiAgentBackend(lambda *args: None)
    assert backend.prepare_chat(payload, get_arm("hiagent_full_native"), None) == payload
    with pytest.raises(proxy.BackendError, match="hiagent_full_native"):
        backend.prepare_chat(payload, get_arm("hiagent_full"), None)


def test_proxy_reserves_before_bridge_allocation_and_send(monkeypatch, tmp_path):
    journal_path = tmp_path / "attempts.jsonl"
    client = NativeHiAgentBridgeClient(POLICY_SAMPLING, max_calls_per_task=96)
    captured = []

    class Opener:
        def open(self, request, **kwargs):
            envelope = json.loads(request.data.decode("utf-8"))
            captured.append((request.full_url, envelope))
            return _Response(_response(envelope))

    monkeypatch.setattr(proxy, "NATIVE_HIAGENT_BRIDGE", client)
    monkeypatch.setattr(
        proxy, "GENERATION_BUDGET", proxy.GenerationBudget(None, per_task_limit=1)
    )
    monkeypatch.setattr(proxy, "ATTEMPT_JOURNAL", AttemptJournal(journal_path))
    monkeypatch.setattr(proxy, "TEXTARM_TRACE", None)
    monkeypatch.setattr(proxy, "NO_UPSTREAM_RETRIES", True)
    monkeypatch.setattr(proxy, "UPSTREAM", "http://127.0.0.1:36200")
    monkeypatch.setattr(proxy, "_OPENER", Opener())
    token = proxy._ATTEMPT_REQUEST.set({
        "request_id": "proxy-request-1", "eval_context": CONTEXT,
    })
    try:
        result = proxy._post_json("/v1/chat/completions", _payload(), 1)
        with pytest.raises(TaskGenerationBudgetExceeded):
            proxy._post_json("/v1/chat/completions", _payload(), 1)
    finally:
        proxy._ATTEMPT_REQUEST.reset(token)

    assert result[RECEIPT_FIELD]["call_ordinal"] == 1
    assert client.call_count(CONTEXT) == 1
    assert len(captured) == 1
    assert captured[0][0].endswith("/v1/native-hiagent/completions")
    assert captured[0][1]["model_call"]["messages"] == MESSAGES
    assert [row["event"] for row in _rows(journal_path)] == ["started", "finished"]


def test_http_failure_consumes_reserved_attempt_without_retry(monkeypatch, tmp_path):
    journal_path = tmp_path / "attempts.jsonl"
    client = NativeHiAgentBridgeClient(POLICY_SAMPLING, max_calls_per_task=96)
    captured = []

    class Opener:
        def open(self, request, **kwargs):
            captured.append(json.loads(request.data.decode("utf-8")))
            raise HTTPError(
                request.full_url, 503, "unavailable", {}, io.BytesIO(b"unavailable")
            )

    monkeypatch.setattr(proxy, "NATIVE_HIAGENT_BRIDGE", client)
    monkeypatch.setattr(
        proxy, "GENERATION_BUDGET", proxy.GenerationBudget(None, per_task_limit=1)
    )
    monkeypatch.setattr(proxy, "ATTEMPT_JOURNAL", AttemptJournal(journal_path))
    monkeypatch.setattr(proxy, "TEXTARM_TRACE", None)
    monkeypatch.setattr(proxy, "NO_UPSTREAM_RETRIES", True)
    monkeypatch.setattr(proxy, "UPSTREAM", "http://127.0.0.1:36200")
    monkeypatch.setattr(proxy, "_OPENER", Opener())
    token = proxy._ATTEMPT_REQUEST.set({
        "request_id": "proxy-request-1", "eval_context": CONTEXT,
    })
    try:
        with pytest.raises(proxy.UpstreamError):
            proxy._post_json("/v1/chat/completions", _payload(), 1)
        with pytest.raises(TaskGenerationBudgetExceeded):
            proxy._post_json("/v1/chat/completions", _payload(), 1)
    finally:
        proxy._ATTEMPT_REQUEST.reset(token)

    assert len(captured) == 1
    assert captured[0]["call_ordinal"] == 1
    assert client.call_count(CONTEXT) == 1
    rows = _rows(journal_path)
    assert [row["event"] for row in rows] == ["started", "finished"]
    assert rows[-1]["status"] == "failed"


def test_proxy_to_real_dispatcher_reconciles_auxiliary_and_policy(monkeypatch, tmp_path):
    import threading
    from urllib.request import build_opener, ProxyHandler
    from benchmarks.memory_runtime.event_native_hiagent import make_hiagent_server
    from benchmarks.memory_runtime.tests.test_event_native_hiagent import (
        ACTOR_SAMPLING, EVAL_CONTEXT, MODEL, _make_dispatcher,
    )

    dispatcher, actor, auxiliary, paths = _make_dispatcher(tmp_path)
    client = NativeHiAgentBridgeClient(ACTOR_SAMPLING["policy"])
    sent = []
    server = make_hiagent_server(dispatcher)
    http_opener = build_opener(ProxyHandler({}))
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})

    class Opener:
        def open(self, request, **kwargs):
            envelope = json.loads(request.data.decode("utf-8"))
            sent.append(envelope)
            return http_opener.open(request, **kwargs)

    journal_path = tmp_path / "proxy_attempts.jsonl"
    monkeypatch.setattr(proxy, "NATIVE_HIAGENT_BRIDGE", client)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET",
                        proxy.GenerationBudget(None, per_task_limit=96))
    monkeypatch.setattr(proxy, "ATTEMPT_JOURNAL", AttemptJournal(journal_path))
    monkeypatch.setattr(proxy, "TEXTARM_TRACE", None)
    monkeypatch.setattr(proxy, "NO_UPSTREAM_RETRIES", True)
    monkeypatch.setattr(proxy, "UPSTREAM", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setattr(proxy, "_OPENER", Opener())
    token = proxy._ATTEMPT_REQUEST.set({
        "request_id": "joined-request", "eval_context": EVAL_CONTEXT,
    })
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "Summarize."}]}
    thread.start()
    try:
        with proxy.textarm_phase("compressor"):
            summary = proxy._post_json("/v1/chat/completions", {
                **payload, "max_tokens": 100, "temperature": 0,
                "top_p": 1, "seed": 42, "stop": ["\n\n"],
            }, 1)
        with proxy.textarm_phase("policy"):
            action = proxy._post_json("/v1/chat/completions", payload, 1)
    finally:
        proxy._ATTEMPT_REQUEST.reset(token)
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert summary["choices"][0]["message"]["content"] == "Summary"
    assert summary["usage"]["completion_tokens"] == 2
    assert action["choices"][0]["message"]["content"] == "Done."
    assert sent[0]["native_session_id"] != sent[1]["native_session_id"]
    proxy_starts = [row for row in _rows(journal_path) if row["event"] == "started"]
    native_joins = [row for row in _rows(paths["join"]) if row["event"] == "started"]
    assert len(proxy_starts) == len(native_joins) == len(_rows(paths["steps"])) == 2
    for request, start, joined, response in zip(
        sent, proxy_starts, native_joins, [summary, action]
    ):
        assert request["proxy_attempt_uid"] == start["attempt_uid"] == joined["proxy_attempt_uid"]
        assert joined["native_attempt_uid"] == response[RECEIPT_FIELD]["native_attempt_uid"]
        assert joined["official_eval_context"] == EVAL_CONTEXT


def test_proxy_configuration_requires_explicit_sampling_and_shared_96(tmp_path):
    common = {
        "backend_name": "event_native_hiagent",
        "arm": get_arm("hiagent_full_native"),
        "policy_sampling_path": "",
        "max_generation_attempts_per_task": 96,
        "no_upstream_retries": True,
        "capture_request_views": True,
        "request_log": str(tmp_path / "proxy.jsonl"),
        "memory_runtime_config": "",
        "upstream": "http://127.0.0.1:36200",
    }
    with pytest.raises(ValueError, match="explicit"):
        proxy._configure_native_hiagent_bridge(**common)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({
        "schema": POLICY_SAMPLING_SCHEMA, **POLICY_SAMPLING,
    }), encoding="utf-8")
    common["policy_sampling_path"] = str(path)
    common["max_generation_attempts_per_task"] = 95
    with pytest.raises(ValueError, match="96"):
        proxy._configure_native_hiagent_bridge(**common)
    common["max_generation_attempts_per_task"] = 96
    common["upstream"] = "http://127.0.0.1:36200/v1"
    with pytest.raises(ValueError, match="host-only"):
        proxy._configure_native_hiagent_bridge(**common)
