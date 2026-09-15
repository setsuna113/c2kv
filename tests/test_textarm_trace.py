"""Durable transport-level trace tests for HiAgent and ACON model calls."""
from __future__ import annotations

from contextlib import contextmanager
import io
import json
import os
from pathlib import Path
import sys
from urllib.error import URLError

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

import proxy
import textarms
from arms import get_arm
from backends import BackendError
from memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal
from memory_runtime.generation_budget import GenerationBudget
from memory_runtime.extraction_telemetry import ExtractionBudget
from textarm_trace import (
    TextarmTrace, current_textarm_phase, read_textarm_trace, textarm_trace_path,
)


class _Response:
    status = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.body).encode("utf-8")


class _Opener:
    def __init__(self, actions=None, before_open=None):
        self.actions = list(actions or [])
        self.before_open = before_open
        self.calls = []

    def open(self, request, **_kwargs):
        self.calls.append(json.loads(request.data.decode("utf-8")))
        if self.before_open is not None:
            self.before_open()
        action = self.actions.pop(0) if self.actions else _body()
        if isinstance(action, BaseException):
            raise action
        return _Response(action)


class _Backend:
    name = "test"
    wants_request_context = False

    @staticmethod
    def prepare_chat(payload, _arm, _plan):
        return json.loads(json.dumps(payload))

    @staticmethod
    def normalize_response(data):
        if data.get("backend_error"):
            raise BackendError("finish_abort", "synthetic backend abort")
        choice = data["choices"][0]
        message = choice.get("message") or {}
        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "cost": data.get("cost"),
        }


def _body(content="policy answer", *, tool_calls=None, prompt=11, completion=3):
    return {
        "choices": [{
            "message": {"content": content, "tool_calls": tool_calls},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
        "cost": {"kv_resident_tokens": prompt, "bytes_per_kv_token": 8},
    }


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text(
        encoding="utf-8").splitlines()]


def _run_logged_request(monkeypatch, request_log, *, arm, capture, source,
                        transform=None):
    monkeypatch.setattr(proxy, "ARM", arm)
    monkeypatch.setattr(proxy, "BACKEND", _Backend())
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", None)
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(request_log))
    monkeypatch.setattr(proxy, "CAPTURE_REQUEST_VIEWS", capture)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", GenerationBudget(None))
    monkeypatch.setattr(proxy, "EXTRACTION_BUDGET", ExtractionBudget(None))
    monkeypatch.setattr(proxy, "ATTEMPT_JOURNAL", None)
    monkeypatch.setattr(proxy, "TEXTARM_TRACE", None)
    monkeypatch.setattr(proxy, "QUERY_PROJECTION", None)
    monkeypatch.setattr(proxy.STATE, "recover", None)
    monkeypatch.setattr(proxy.STATE, "reference_log_path", "")
    if transform is not None:
        monkeypatch.setattr(proxy, "_apply_text_arm", transform)
    wire = []

    def post(_path, payload, _timeout, retries=0):
        wire.append(json.loads(json.dumps(payload)))
        return _body()

    monkeypatch.setattr(proxy, "_post_json", post)
    raw = json.dumps(source).encode("utf-8")
    handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    handler.path = "/v1/chat/completions"
    handler.headers = {"Content-Length": str(len(raw))}
    handler.rfile = io.BytesIO(raw)
    sent = []
    handler._send_json = lambda code, body: sent.append((code, body))
    handler.do_POST()
    return _rows(request_log)[-1], sent, wire


@contextmanager
def _request(request_id, task_id):
    token = proxy._ATTEMPT_REQUEST.set({
        "request_id": request_id,
        "eval_context": {"benchmark": "bfcl", "task_id": task_id,
                         "attempt": 0, "private": "not-allowlisted"},
    })
    try:
        yield
    finally:
        proxy._ATTEMPT_REQUEST.reset(token)


@pytest.fixture
def configured(monkeypatch, tmp_path):
    request_log = tmp_path / "proxy_hiagent_34100.jsonl"
    journal_path = tmp_path / "attempts_proxy_hiagent_34100.jsonl"
    trace_path = textarm_trace_path(request_log)
    monkeypatch.setattr(proxy, "ARM", get_arm("hiagent_full"))
    monkeypatch.setattr(proxy, "BACKEND", _Backend())
    monkeypatch.setattr(proxy, "UPSTREAM", "http://127.0.0.1:1")
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(request_log))
    monkeypatch.setattr(proxy, "ATTEMPT_JOURNAL", AttemptJournal(journal_path))
    monkeypatch.setattr(proxy, "TEXTARM_TRACE", TextarmTrace(trace_path))
    monkeypatch.setattr(proxy, "CAPTURE_REQUEST_VIEWS", True)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", None)
    monkeypatch.setattr(proxy, "NO_UPSTREAM_RETRIES", True)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", GenerationBudget(20))
    monkeypatch.setattr(proxy, "EXTRACTION_BUDGET", ExtractionBudget(None))
    return journal_path, trace_path


def test_success_fsyncs_start_before_http_and_keeps_full_normalized_cost(
        configured, monkeypatch):
    journal_path, trace_path = configured
    real_fsync = os.fsync
    fsyncs = []

    def observed_fsync(descriptor):
        fsyncs.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", observed_fsync)

    def before_open():
        assert _rows(journal_path)[-1]["status"] == "started"
        assert read_textarm_trace(trace_path)[-1]["status"] == "started"
        assert len(fsyncs) >= 2

    opener = _Opener(before_open=before_open)
    monkeypatch.setattr(proxy, "_OPENER", opener)
    payload = {
        "model": "c2kv-agent",
        "messages": [{"role": "user", "content": "do the task"}],
        "tools": [{"type": "function", "function": {"name": "act"}}],
        "temperature": 0.001,
        "max_completion_tokens": 64,
        "stream": False,
        "private_transport_field": "must-not-be-traced",
    }
    with _request("request-a", "task-a"):
        assert proxy._post_json("/v1/chat/completions", payload, 1)["usage"][
            "completion_tokens"] == 3

    trace = read_textarm_trace(trace_path)
    journal = _rows(journal_path)
    assert [row["event"] for row in trace] == ["started", "finished"]
    assert trace[0]["attempt_uid"] == trace[1]["attempt_uid"] == journal[0][
        "attempt_uid"] == journal[1]["attempt_uid"]
    assert trace[0]["phase"] == "policy"
    assert trace[0]["eval_context"] == {
        "benchmark": "bfcl", "task_id": "task-a", "attempt": 0}
    assert trace[0]["request_view"] == {
        "model": "c2kv-agent",
        "messages": payload["messages"],
        "tools": payload["tools"],
        "sampling": {"temperature": 0.001, "max_completion_tokens": 64},
    }
    assert trace[1]["response_view"] == {
        "content": "policy answer", "tool_calls": None, "finish_reason": "stop"}
    assert trace[1]["usage"] == _body()["usage"]
    assert trace[1]["cost"] == _body()["cost"]
    assert trace[1]["wall_sec"] >= 0
    assert "private_transport_field" not in trace_path.read_text(encoding="utf-8")
    assert len(fsyncs) >= 4


def test_request_log_keeps_original_textarm_source_on_success_and_early_error(
        monkeypatch, tmp_path):
    source_messages = [
        {"role": "system", "content": "source system"},
        {"role": "user", "content": "source request"},
    ]
    source_tools = [{
        "type": "function",
        "function": {"name": "source_tool", "parameters": {"type": "object"}},
    }]
    source = {
        "model": "c2kv-agent",
        "messages": source_messages,
        "tools": source_tools,
        "temperature": 0.25,
        "c2kv_eval_context": {"benchmark": "bfcl", "task_id": "source-task"},
    }
    transformed_messages = [
        {"role": "system", "content": "transformed policy"},
        {"role": "user", "content": "compressed source"},
    ]
    internal_tool = {
        "type": "function",
        "function": {"name": "internal_tool", "parameters": {"type": "object"}},
    }

    def transform(payload, _arm, _conv):
        staged = dict(payload)
        staged["messages"] = transformed_messages
        staged["tools"] = source_tools + [internal_tool]
        return staged, {"n_compressor_calls": 0, "compressor_usage": {}}

    success, sent, wire = _run_logged_request(
        monkeypatch, tmp_path / "success.jsonl", arm=get_arm("acon_obs"),
        capture=True, source=source, transform=transform)
    assert sent[0][0] == 200 and len(wire) == 1
    assert success["request_view"]["messages"] == transformed_messages
    assert success["request_view"]["tools"] == source_tools + [internal_tool]
    assert success["textarm_source_request_view"] == {
        "model": "c2kv-agent",
        "messages": source_messages,
        "tools": source_tools,
        "sampling": {"temperature": 0.25},
    }

    def fail_before_transform(*_args):
        raise proxy.GenerationBudgetExceeded(1)

    early, sent, wire = _run_logged_request(
        monkeypatch, tmp_path / "early.jsonl", arm=get_arm("acon_obs"),
        capture=True, source=source, transform=fail_before_transform)
    assert sent[0][0] == 502 and wire == []
    assert early["status"] == "generation_budget_exhausted"
    assert early["textarm_source_request_view"]["messages"] == source_messages
    assert early["textarm_source_request_view"]["tools"] == source_tools


@pytest.mark.parametrize("arm,capture", [
    (get_arm("acon_obs"), False),
    (get_arm("full"), True),
])
def test_request_log_omits_textarm_source_when_capture_or_textarm_is_disabled(
        monkeypatch, tmp_path, arm, capture):
    source = {
        "model": "c2kv-agent",
        "messages": [{"role": "user", "content": "source request"}],
        "tools": [],
    }

    def passthrough(payload, _arm, _conv):
        return dict(payload), {"n_compressor_calls": 0, "compressor_usage": {}}

    row, sent, wire = _run_logged_request(
        monkeypatch, tmp_path / f"{arm.name}-{capture}.jsonl", arm=arm,
        capture=capture, source=source, transform=passthrough)
    assert sent[0][0] == 200 and len(wire) == 1
    assert "textarm_source_request_view" not in row


def test_compressor_and_internal_retrieval_phases_are_scoped_per_call(
        configured, monkeypatch):
    _, trace_path = configured
    opener = _Opener([
        _body("compressed history"),
        _body("environment action"),
        _body("ordinary policy"),
    ])
    monkeypatch.setattr(proxy, "_OPENER", opener)
    with _request("request-compressor", "task-a"):
        assert proxy._textarm_compress({
            "model": "c2kv-agent", "messages": [{"role": "user", "content": "history"}]
        }) == "compressed history"
    assert current_textarm_phase() == "policy"

    monkeypatch.setattr(
        proxy, "_apply_text_arm",
        lambda payload, _arm, _conv, _ids: (
            payload,
            {"invalid_retrieval_subgoals": [], "compressor_usage": {},
             "n_compressor_calls": 0},
        ),
    )
    retrieval_request = [{
        "id": "retrieve-1",
        "type": "function",
        "function": {"name": textarms.HIAGENT_RETRIEVE_TOOL_NAME,
                     "arguments": '{"subgoal_ids":[1]}'},
    }]
    initial = _body("", tool_calls=retrieval_request)
    with _request("request-retrieval", "task-b"):
        result = proxy._hiagent_retrieval_loop(
            {"model": "c2kv-agent", "messages": []},
            get_arm("hiagent_full"), "conversation", initial,
            {"n_compressor_calls": 0},
            lambda staged: proxy._post_json(
                "/v1/chat/completions", staged, 1),
        )
    assert result["choices"][0]["message"]["content"] == "environment action"
    assert current_textarm_phase() == "policy"

    with _request("request-policy", "task-c"):
        proxy._post_json(
            "/v1/chat/completions",
            {"model": "c2kv-agent", "messages": [{"role": "user", "content": "next"}]},
            1,
        )
    starts = [row for row in read_textarm_trace(trace_path) if row["event"] == "started"]
    assert [(row["phase"], row["eval_context"]["task_id"]) for row in starts] == [
        ("compressor", "task-a"),
        ("trajectory_retrieval_policy", "task-b"),
        ("policy", "task-c"),
    ]


def test_transport_unknown_usage_and_http_200_backend_failure_known_usage(
        configured, monkeypatch):
    journal_path, trace_path = configured
    backend_failure = {
        "backend_error": True,
        "usage": {"prompt_tokens": 19, "completion_tokens": 0, "total_tokens": 19},
    }
    opener = _Opener([backend_failure, URLError("synthetic disconnect")])
    monkeypatch.setattr(proxy, "_OPENER", opener)

    with _request("request-backend", "task-a"):
        raw = proxy._post_json(
            "/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}]}, 1)
    assert raw == backend_failure
    with pytest.raises(BackendError, match="synthetic backend abort"):
        proxy.BACKEND.normalize_response(raw)
    with _request("request-transport", "task-b"):
        with pytest.raises(proxy.UpstreamError, match="synthetic disconnect"):
            proxy._post_json(
                "/v1/chat/completions", {"messages": [{"role": "user", "content": "b"}]}, 1)

    finishes = [row for row in read_textarm_trace(trace_path)
                if row["event"] == "finished"]
    assert finishes[0]["status"] == "failed"
    assert finishes[0]["failure_stage"] == "backend_normalization"
    assert finishes[0]["error_kind"] == "finish_abort"
    assert finishes[0]["usage"] == backend_failure["usage"]
    assert finishes[1]["status"] == "failed"
    assert finishes[1]["failure_stage"] == "transport"
    assert finishes[1]["usage"] is None and finishes[1]["cost"] is None
    journal = summarize_attempt_journal(journal_path)
    assert (journal["completed"], journal["failed"], journal["pending"]) == (1, 1, 0)
    assert journal["finished_usage_totals"] == backend_failure["usage"]


def test_trace_finish_failure_after_http_is_never_retried(configured, monkeypatch):
    journal_path, trace_path = configured

    class BrokenFinishTrace(TextarmTrace):
        def finish(self, *_args, **_kwargs):
            raise OSError("synthetic trace fsync failure")

    opener = _Opener()
    monkeypatch.setattr(proxy, "_OPENER", opener)
    monkeypatch.setattr(proxy, "TEXTARM_TRACE", BrokenFinishTrace(trace_path))
    monkeypatch.setattr(proxy, "NO_UPSTREAM_RETRIES", False)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", GenerationBudget(None))
    with _request("request-a", "task-a"):
        with pytest.raises(OSError, match="trace fsync failure"):
            proxy._post_json(
                "/v1/chat/completions", {"messages": [{"role": "user", "content": "a"}]},
                1, retries=2)

    assert len(opener.calls) == 1
    assert [row["event"] for row in _rows(journal_path)] == ["started", "finished"]
    assert _rows(journal_path)[-1]["status"] == "completed"
    assert [row["event"] for row in read_textarm_trace(trace_path)] == ["started"]


def test_main_leaves_trace_disabled_for_non_text_arm(monkeypatch, tmp_path):
    class Server:
        def __init__(self, *_args):
            pass

        def serve_forever(self):
            return None

    monkeypatch.setattr(proxy, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(proxy, "get_backend", lambda *_args: _Backend())
    # main() is process-oriented and assigns module globals directly.  Record
    # their pre-test values so pytest restores them for later proxy tests.
    for name in (
        "ARM", "BACKEND", "MEMORY_RUNTIME", "UPSTREAM", "REQUEST_LOG_PATH",
        "DOC_PACKING", "MAX_DOC_LENGTH", "MAX_DOC_NUM", "QUERY_PROJECTION",
        "WITNESS_TOKENIZER_PATH", "MEMORY_RUNTIME_BYTES_PER_KV_TOKEN",
        "MEMORY_RUNTIME_FATAL_ERROR", "NO_UPSTREAM_RETRIES",
        "CAPTURE_REQUEST_VIEWS", "GENERATION_BUDGET", "EXTRACTION_BUDGET",
        "ATTEMPT_JOURNAL", "TEXTARM_TRACE",
    ):
        monkeypatch.setattr(proxy, name, getattr(proxy, name))
    monkeypatch.setattr(
        proxy.STATE, "reference_log_path", proxy.STATE.reference_log_path)
    request_log = tmp_path / "proxy_full.jsonl"
    proxy.main([
        "--upstream", "http://127.0.0.1:1",
        "--backend", "sglang",
        "--arm", "full",
        "--port", "39999",
        "--capture-request-views",
        "--request-log", str(request_log),
    ])

    assert proxy.TEXTARM_TRACE is None
    assert not textarm_trace_path(request_log).exists()
