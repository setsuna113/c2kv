"""Account predictor and action separately; submit only the action response."""

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy
from arms import get_arm
from memory_runtime.attempt_journal import AttemptJournal
from memory_runtime.source_needs_runtime import SourceNeedsRuntime
from memory_runtime.tests.test_native_workspace import _count, _setup
from memory_runtime.tests.test_source_needs_runtime import _config, _messages
from test_memory_runtime_proxy import _Backend


class _Response:
    def __init__(self, body):
        self.body = body
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False
    def read(self):
        return json.dumps(self.body).encode()


class _MeasuredBackend(_Backend):
    def normalize_response(self, data):
        value = super().normalize_response(data)
        value["cost"]["c2kv_tools_dump"] = "full"
        return value


def _run(monkeypatch, tmp_path, *, fail_prediction=False, generation_limit=2,
         route="ac_native_needs_typed"):
    _setup(monkeypatch, "ac_native_workspace")
    config = _config(route)
    if route.endswith("_state_none"):
        config["state_prompt_token_cap"] = 2000
    runtime = SourceNeedsRuntime(config, _count)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "STATE", proxy.ProxyState())
    monkeypatch.setattr(proxy, "BACKEND", _MeasuredBackend(bytes_per_kv_token=1))
    monkeypatch.setattr(proxy, "ARM", get_arm("full" if route.startswith("raw_") else "c2kv4"))
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_BYTES_PER_KV_TOKEN", None)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_FATAL_ERROR", None)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", proxy.GenerationBudget(generation_limit))
    monkeypatch.setattr(proxy, "EXTRACTION_BUDGET", proxy.ExtractionBudget(None))
    monkeypatch.setattr(proxy, "CAPTURE_REQUEST_VIEWS", True)
    monkeypatch.setattr(proxy, "QUERY_PROJECTION", "base")
    monkeypatch.setattr(proxy, "UPSTREAM", "http://127.0.0.1:1")
    journal = tmp_path / "attempts.jsonl"
    log = tmp_path / "proxy.jsonl"
    monkeypatch.setattr(proxy, "ATTEMPT_JOURNAL", AttemptJournal(journal))
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(log))
    posts = []

    def open_request(request, timeout):
        started = json.loads(journal.read_text().splitlines()[-1])
        assert started["event"] == "started"
        body = json.loads(request.data)
        posts.append(body)
        if runtime.source_needs_strategy in {"typed", "tool"} and len(posts) == 1:
            assert "tool_choice" not in body and "response_format" not in body
            assert body["max_completion_tokens"] == runtime.predictor_completion_token_cap
            assert body["c2kv_use_gist_projection"] is False
            if fail_prediction:
                raise URLError("synthetic prediction transport failure")
            content = '{"needs":[{"kind":"prior_result","source_ids":["needs-task:m1"]}]}'
            if runtime.source_needs_strategy == "tool":
                assert [t["function"]["name"] for t in body["tools"]] == ["request_history_evidence"]
                tool_calls = [{"type": "function", "function": {"name": "request_history_evidence", "arguments": content}}]
                content = None
            else:
                assert "tools" not in body
                tool_calls = None
        else:
            assert "OLD-FILE-LIST" in json.dumps(body["messages"])
            content = "FINAL_ACTION_ONLY"
            tool_calls = [{"id": "final-call", "type": "function", "function": {
                "name": "cat", "arguments": '{"file_name":"summary.txt"}'}}]
        prompt_tokens = _count([m for m in body["messages"] if not m.get("c2kv_key_hash")], body.get("tools"))
        return _Response({"choices": [{"message": {"content": content, "tool_calls": tool_calls},
            "finish_reason": "tool_calls" if tool_calls else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 7,
                      "total_tokens": prompt_tokens + 7}})

    monkeypatch.setattr(proxy, "_OPENER", SimpleNamespace(open=open_request))
    payload = {"model": "test-model", "messages": _messages(), "tools": [],
        "temperature": 0.001, "seed": 0, "max_completion_tokens": 4096,
        "c2kv_eval_context": {"task_id": "needs-task", "attempt": 0, "user_turn": 1, "step": 1}}
    raw = json.dumps(payload).encode()
    handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    handler.rfile = io.BytesIO(raw)
    handler.headers = {"Content-Length": str(len(raw))}
    handler.path = "/v1/chat/completions"
    sent = []
    def capture_response(status, body):
        # A client may terminate the owned proxy immediately after its last
        # response. The complete request trace must already be on disk.
        if status == 200:
            recorded = json.loads(log.read_text().splitlines()[-1])
            assert recorded["status"] == "ok"
            assert len(recorded["generation_trace"]) == len(posts)
        sent.append((status, body))
    handler._send_json = capture_response
    handler.do_POST()
    return sent, posts, json.loads(log.read_text().splitlines()[-1]), [
        json.loads(line) for line in journal.read_text().splitlines()]


def test_two_budgeted_calls_one_returned_action_and_complete_costs(monkeypatch, tmp_path):
    sent, posts, row, journal = _run(monkeypatch, tmp_path)
    assert sent[0][0] == 200 and len(posts) == 2
    assert sent[0][1]["choices"][0]["message"]["content"] == "FINAL_ACTION_ONLY"
    assert [record["phase"] for record in row["generation_trace"]] == ["source_prediction", "action"]
    assert all(record["backend_verified"] for record in row["generation_trace"])
    assert row["generation_trace"][0]["submitted_to_executor"] is False
    assert row["generation_usage_total"]["completion_tokens"] == 14
    assert row["generation_attempts"] == 2
    assert [entry["event"] for entry in journal] == ["started", "finished", "started", "finished"]
    assert len(row["forwarded_request_views"]) == 2


@pytest.mark.parametrize("route", ["ac_native_needs_tool", "raw_native_needs_tool"])
def test_native_request_is_accounted_and_never_submitted_as_an_action(monkeypatch, tmp_path, route):
    sent, posts, row, journal = _run(monkeypatch, tmp_path, route=route)
    assert sent[0][0] == 200 and len(posts) == 2
    assert posts[1]["tools"] == []
    assert sent[0][1]["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "cat"
    assert [record["phase"] for record in row["generation_trace"]] == ["source_prediction", "action"]
    assert row["generation_trace"][0]["submitted_to_executor"] is False
    assert row["generation_trace"][0]["response_view"]["tool_calls"][0]["function"]["name"] == "request_history_evidence"
    assert row["generation_attempts"] == 2 and row["generation_usage_total"]["completion_tokens"] == 14
    assert all(record["backend_verified"] for record in row["generation_trace"])
    assert [entry["event"] for entry in journal] == ["started", "finished", "started", "finished"]


def test_prediction_transport_failure_stops_without_action_or_retry(monkeypatch, tmp_path):
    sent, posts, row, journal = _run(monkeypatch, tmp_path, fail_prediction=True)
    assert sent[0][0] == 502 and len(posts) == 1
    assert row["generation_attempts"] == 1
    assert row["generation_trace"][0]["phase"] == "source_prediction"
    assert row["generation_trace"][0]["status"] == "failed"
    assert row["generation_usage_total"]["prompt_tokens"] is None
    assert [entry["event"] for entry in journal] == ["started", "finished"]


def test_exhausted_generation_budget_does_not_hide_an_unreserved_action(monkeypatch, tmp_path):
    sent, posts, row, journal = _run(monkeypatch, tmp_path, generation_limit=1)
    assert sent[0][0] == 502 and len(posts) == 1
    assert row["generation_attempts"] == 1 and len(row["forwarded_request_views"]) == 1
    assert row["generation_trace"][0]["phase"] == "source_prediction"
    assert [entry["event"] for entry in journal] == ["started", "finished"]


@pytest.mark.parametrize("route", ["ac_native_state_none", "raw_native_state_none"])
def test_state_candidate_uses_one_accounted_action_without_auxiliary_generation(monkeypatch, tmp_path, route):
    sent, posts, row, journal = _run(monkeypatch, tmp_path, route=route, generation_limit=1)
    assert sent[0][0] == 200 and len(posts) == 1
    assert [record["phase"] for record in row["generation_trace"]] == ["action"]
    assert row["generation_trace"][0]["backend_verified"] is True
    assert row["memory_runtime"]["state_bytes"] > 0
    assert row["memory_runtime"]["observed_state"]["admitted_calls"] > 0
    assert row["generation_attempts"] == 1
    assert [entry["event"] for entry in journal] == ["started", "finished"]
