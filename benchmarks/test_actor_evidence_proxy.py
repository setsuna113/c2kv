"""Only the final application response leaves the actor-evidence proxy."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy
from arms import get_arm
from memory_runtime.actor_evidence import ActorEvidenceRuntime
from memory_runtime.attempt_journal import AttemptJournal
from memory_runtime.native_source_needs import TOOL_NAME
from memory_runtime.tests.test_actor_evidence import count, request
from memory_runtime.tests.test_native_workspace import _setup
from memory_runtime.tests.test_source_needs_runtime import _config, _messages
from test_source_needs_proxy import _Response, _MeasuredBackend


def drive(monkeypatch, tmp_path, *, direct=False, invalid=False, generation_limit=2):
    _setup(monkeypatch, "ac_native_workspace")
    runtime = ActorEvidenceRuntime({**_config("ac_actor_evidence_once"),
        "actor_evidence_schema_token_cap": 20000}, count)
    for name, value in {"MEMORY_RUNTIME": runtime, "STATE": proxy.ProxyState(),
        "BACKEND": _MeasuredBackend(bytes_per_kv_token=1), "ARM": get_arm("c2kv4"),
        "MEMORY_RUNTIME_BYTES_PER_KV_TOKEN": None, "MEMORY_RUNTIME_FATAL_ERROR": None,
        "GENERATION_BUDGET": proxy.GenerationBudget(generation_limit),
        "EXTRACTION_BUDGET": proxy.ExtractionBudget(None), "CAPTURE_REQUEST_VIEWS": True,
        "QUERY_PROJECTION": "base", "UPSTREAM": "http://127.0.0.1:1",
        "ATTEMPT_JOURNAL": AttemptJournal(tmp_path / "attempts.jsonl"),
        "REQUEST_LOG_PATH": str(tmp_path / "proxy.jsonl")}.items():
        monkeypatch.setattr(proxy, name, value)
    posts = []
    final_call = {"id": "final", "type": "function", "function": {
        "name": "cat", "arguments": '{"file_name":"summary.txt"}'}}
    def open_request(req, timeout):
        payload = json.loads(req.data)
        posts.append(payload)
        names = [tool["function"]["name"] for tool in payload["tools"]]
        raw_messages = [message for message in payload["messages"] if not message.get("c2kv_key_hash")]
        if len(posts) == 1 and not direct:
            assert TOOL_NAME in names and "OLD-FILE-LIST" not in json.dumps(raw_messages)
            calls = request(["future:m99" if invalid else "needs-task:m1"], [final_call])["tool_calls"]
            content = "INTERNAL_REQUEST_AND_UNCOMMITTED_APPLICATION_CALL"
        else:
            if len(posts) == 2:
                assert TOOL_NAME not in names and "OLD-FILE-LIST" in json.dumps(raw_messages)
            calls, content = [final_call], "FINAL_APPLICATION_RESPONSE"
        raw_tokens = count([m for m in payload["messages"] if not m.get("c2kv_key_hash")], payload["tools"])
        return _Response({"choices": [{"message": {"content": content, "tool_calls": calls},
            "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": raw_tokens,
            "completion_tokens": 7, "total_tokens": raw_tokens + 7}})
    monkeypatch.setattr(proxy, "_OPENER", SimpleNamespace(open=open_request))
    messages = _messages()
    messages[5]["content"] = "Continue the current goal."
    app_tools = [{"type": "function", "function": {"name": "cat", "parameters": {
        "type": "object", "properties": {"file_name": {"type": "string"}}, "required": ["file_name"]}}}]
    payload = {"model": "test-model", "messages": messages, "tools": app_tools,
        "temperature": 0.001, "seed": 0, "max_completion_tokens": 4096,
        "c2kv_eval_context": {"task_id": "needs-task", "attempt": 0, "user_turn": 1, "step": 1}}
    data = json.dumps(payload).encode()
    handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    handler.rfile, handler.headers = io.BytesIO(data), {"Content-Length": str(len(data))}
    handler.path = "/v1/chat/completions"
    sent = []
    def receive(status, body):
        assert (tmp_path / "proxy.jsonl").is_file()
        sent.append((status, body))
    handler._send_json = receive
    handler.do_POST()
    log = json.loads((tmp_path / "proxy.jsonl").read_text().splitlines()[-1])
    journal = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert log["request_view"]["tools"] == app_tools
    assert log["request_view"]["messages"] == messages
    return sent, posts, log, journal


def test_internal_request_and_mixed_action_are_withheld_until_final_response(monkeypatch, tmp_path):
    sent, posts, log, journal = drive(monkeypatch, tmp_path)
    assert sent[0][0] == 200 and len(posts) == 2
    assert sent[0][1]["choices"][0]["message"]["content"] == "FINAL_APPLICATION_RESPONSE"
    assert [r["phase"] for r in log["generation_trace"]] == ["evidence_request", "action"]
    assert log["generation_trace"][0]["submitted_to_executor"] is False
    assert log["generation_trace"][1]["submitted_to_executor"] is True
    assert all(r["backend_verified"] for r in log["generation_trace"])
    assert log["generation_usage_total"]["completion_tokens"] == 14
    assert log["memory_runtime"]["actor_evidence"]["mixed_application_calls_discarded"] == 1
    assert len(journal) == 4


def test_direct_application_action_uses_one_generation(monkeypatch, tmp_path):
    sent, posts, log, journal = drive(monkeypatch, tmp_path, direct=True)
    assert sent[0][0] == 200 and len(posts) == 1
    assert log["generation_attempts"] == 1 and len(journal) == 2
    assert log["generation_trace"][0]["submitted_to_executor"] is True


def test_invalid_internal_request_never_returns_the_mixed_application_action(monkeypatch, tmp_path):
    sent, posts, log, journal = drive(monkeypatch, tmp_path, invalid=True)
    assert sent[0][0] == 502 and len(posts) == 1
    assert len(journal) == 2 and log["generation_attempts"] == 1


def test_generation_budget_exhaustion_cannot_hide_a_second_call(monkeypatch, tmp_path):
    sent, posts, log, journal = drive(monkeypatch, tmp_path, generation_limit=1)
    assert sent[0][0] == 502 and len(posts) == 1
    assert len(journal) == 2 and log["generation_attempts"] == 1
