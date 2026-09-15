"""Behavioral cost, failure, and cache tests for text-summary transport."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))

import proxy
from memory_runtime.attempt_journal import summarize_attempt_journal
from memory_runtime.generation_budget import GenerationBudget
from memory_runtime.summary_transport import render_summary


class _Tokenizer:
    bos_token = "<bos>"

    def apply_chat_template(self, messages, **_kwargs):
        return self.bos_token + " ".join(message["content"] for message in messages)

    def encode(self, rendered, **_kwargs):
        return rendered.split()


class _Runtime:
    def __init__(self):
        self.run_id = "summary-test-run"
        self.bytes_per_kv_token = 8
        self.tokenizer = _Tokenizer()
        self.summary_config = {
            "summary_model": "c2kv-agent",
            "summary_prompt_token_cap": 1024,
            "summary_attempts_per_task": 1152,
        }

    @staticmethod
    def _token_counter(messages, tools):
        assert tools is None
        return sum(max(1, len(message["content"].split())) for message in messages)


class _Backend:
    @staticmethod
    def prepare_chat(payload, arm, _repair):
        assert arm.name == "full"
        return dict(payload)

    @staticmethod
    def normalize_response(data):
        choice = data["choices"][0]
        message = choice.get("message") or {}
        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
            "cost": {"bytes_per_kv_token": 8},
        }


class _Response:
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
        payload = json.loads(request.data.decode("utf-8"))
        self.calls.append(payload)
        if self.before_open is not None:
            self.before_open()
        action = self.actions.pop(0) if self.actions else "ok"
        if isinstance(action, BaseException):
            raise action
        expected = _Runtime._token_counter(payload["messages"], None)
        body = {
            "choices": [{
                "message": {"content": "kept item-17 and its observed result"},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": expected,
                "completion_tokens": 6,
                "total_tokens": expected + 6,
            },
        }
        if action == "empty":
            body["choices"][0]["message"]["content"] = ""
        elif action == "bad_finish":
            body["choices"][0]["finish_reason"] = "abort"
        elif action == "bad_usage":
            body["usage"]["prompt_tokens"] = expected + 1
            body["usage"]["total_tokens"] += 1
        return _Response(body)


def _source(label="item-17"):
    return [
        {"role": "user", "content": f"Remember {label}."},
        {"role": "assistant", "content": f"Recorded {label}."},
        {"role": "user", "content": "Continue with the current action."},
    ]


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _paths(tmp_path):
    request_log = tmp_path / "proxy_full_34100.jsonl"
    return (
        request_log,
        tmp_path / "summary_attempts_proxy_full_34100.jsonl",
        tmp_path / "summary_trace_proxy_full_34100.jsonl",
    )


@pytest.fixture
def configured(monkeypatch, tmp_path):
    runtime = _Runtime()
    actor_budget = GenerationBudget(2, per_task_limit=2)
    request_log, journal_path, trace_path = _paths(tmp_path)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_BYTES_PER_KV_TOKEN", None)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_FATAL_ERROR", None)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", actor_budget)
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(request_log))
    monkeypatch.setattr(proxy, "UPSTREAM", "http://127.0.0.1:1")
    monkeypatch.setattr(proxy, "BACKEND", _Backend())
    monkeypatch.setattr(proxy, "MAX_DOC_LENGTH", 512)
    monkeypatch.setattr(proxy, "MAX_DOC_NUM", 12)
    return runtime, actor_budget, journal_path, trace_path


def _render(source, context, request_id):
    token = proxy._ATTEMPT_REQUEST.set({
        "request_id": request_id,
        "eval_context": context,
    })
    try:
        return render_summary(proxy, source, context)
    finally:
        proxy._ATTEMPT_REQUEST.reset(token)


def test_http_success_returns_only_after_usage_and_both_ledgers_are_fsynced(
        configured, monkeypatch):
    runtime, actor_budget, journal_path, trace_path = configured
    real_fsync = os.fsync
    fsynced = []

    def observed_fsync(descriptor):
        fsynced.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", observed_fsync)

    def before_open():
        assert _read_jsonl(journal_path)[-1]["status"] == "started"
        assert _read_jsonl(trace_path)[-1]["status"] == "started"
        assert len(fsynced) >= 2

    opener = _Opener(before_open=before_open)
    monkeypatch.setattr(proxy, "_OPENER", opener)
    result = _render(
        _source(), {"task_id": "task-a", "attempt": 0, "user_turn": 1}, "request-a")

    assert result["producer_calls"] == 1
    assert result["records"][0]["completion_cap"] == 16
    assert len(opener.calls) == 1
    assert opener.calls[0]["max_completion_tokens"] == 16
    assert opener.calls[0]["model"] == runtime.summary_config["summary_model"]
    assert actor_budget.consumed == 0

    summary = summarize_attempt_journal(journal_path)
    assert (summary["started"], summary["completed"], summary["pending"]) == (1, 1, 0)
    expected = runtime._token_counter(opener.calls[0]["messages"], None)
    assert summary["finished_usage_totals"] == {
        "prompt_tokens": expected,
        "completion_tokens": 6,
        "total_tokens": expected + 6,
    }
    trace = _read_jsonl(trace_path)
    assert [row["event"] for row in trace] == ["started", "finished", "lookups"]
    assert trace[1]["status"] == "completed"
    assert trace[1]["attempt_uid"] == trace[0]["attempt_uid"]
    assert trace[1]["usage"] == summary["finished_usage_totals"]
    assert trace[1]["wall_sec"] >= 0
    assert trace[2]["producer_calls"] == 1
    assert journal_path.read_bytes().endswith(b"\n")
    assert trace_path.read_bytes().endswith(b"\n")
    assert len(fsynced) >= 5


@pytest.mark.parametrize("first_action", [OSError("synthetic transport failure"),
                                            "empty", "bad_finish"])
def test_transport_and_invalid_responses_are_not_cached(
        configured, monkeypatch, first_action):
    _, actor_budget, journal_path, trace_path = configured
    opener = _Opener([first_action, "ok"])
    monkeypatch.setattr(proxy, "_OPENER", opener)
    context = {"task_id": "task-a", "attempt": 0}

    with pytest.raises((OSError, ValueError, proxy.MemoryRuntimeError)):
        _render(_source(), context, "request-failed")
    recovered = _render(_source(), context, "request-recovered")
    cached = _render(_source(), context, "request-cache-hit")

    assert len(opener.calls) == 2
    assert recovered["producer_calls"] == 1
    assert cached["producer_calls"] == 0
    assert cached["lookups"] == [{
        "summary_key": recovered["lookups"][0]["summary_key"],
        "client_cache_hit": True,
        "packing_fragment_id": 0,
    }]
    summary = summarize_attempt_journal(journal_path)
    assert (summary["started"], summary["failed"], summary["completed"]) == (2, 1, 1)
    model_finishes = [row for row in _read_jsonl(trace_path)
                      if row["event"] == "finished"]
    assert [row["status"] for row in model_finishes] == ["failed", "completed"]
    assert actor_budget.consumed == 0


def test_cache_identity_is_exact_and_isolated_by_task_and_attempt(
        configured, monkeypatch):
    _, actor_budget, journal_path, trace_path = configured
    opener = _Opener()
    monkeypatch.setattr(proxy, "_OPENER", opener)

    first = _render(_source(), {"task_id": "task-a", "attempt": 0}, "request-a0")
    exact = _render(_source(), {"task_id": "task-a", "attempt": 0}, "request-a0-hit")
    other_attempt = _render(
        _source(), {"task_id": "task-a", "attempt": 1}, "request-a1")
    other_task = _render(
        _source(), {"task_id": "task-b", "attempt": 0}, "request-b0")
    changed_fragment = _render(
        _source("item-18"), {"task_id": "task-a", "attempt": 0}, "request-a0-changed")

    assert [result["producer_calls"] for result in
            (first, exact, other_attempt, other_task, changed_fragment)] == [1, 0, 1, 1, 1]
    assert len(opener.calls) == 4
    assert summarize_attempt_journal(journal_path)["completed"] == 4
    lookups = [row for row in _read_jsonl(trace_path) if row["event"] == "lookups"]
    assert [row["producer_calls"] for row in lookups] == [1, 0, 1, 1, 1]
    assert actor_budget.consumed == 0


def test_prompt_cap_rejects_before_journal_or_http_transport(
        configured, monkeypatch):
    runtime, actor_budget, journal_path, trace_path = configured
    opener = _Opener()
    monkeypatch.setattr(proxy, "_OPENER", opener)
    runtime._token_counter = lambda _messages, _tools: 1025

    with pytest.raises(ValueError, match="prompt exceeds"):
        _render(_source(), {"task_id": "task-a", "attempt": 0}, "request-a")

    assert opener.calls == []
    assert not journal_path.exists()
    assert not trace_path.exists()
    assert actor_budget.consumed == 0


def test_per_task_cap_is_shared_across_attempt_ids_and_rejects_before_transport(
        configured, monkeypatch):
    runtime, actor_budget, journal_path, trace_path = configured
    opener = _Opener()
    monkeypatch.setattr(proxy, "_OPENER", opener)

    _render(_source(), {"task_id": "task-a", "attempt": 0}, "request-a0")
    runtime.summary_config["summary_attempts_per_task"] = 1
    before_journal = journal_path.read_bytes()
    before_trace = trace_path.read_bytes()
    with pytest.raises(ValueError, match="attempt budget exhausted"):
        _render(_source(), {"task_id": "task-a", "attempt": 1}, "request-a1")

    assert len(opener.calls) == 1
    assert journal_path.read_bytes() == before_journal
    assert trace_path.read_bytes() == before_trace
    assert actor_budget.consumed == 0


def test_per_task_cap_does_not_leak_between_tasks(configured, monkeypatch):
    runtime, actor_budget, journal_path, _ = configured
    opener = _Opener()
    monkeypatch.setattr(proxy, "_OPENER", opener)

    _render(_source(), {"task_id": "task-a", "attempt": 0}, "request-a")
    runtime.summary_config["summary_attempts_per_task"] = 1
    _render(_source(), {"task_id": "task-b", "attempt": 0}, "request-b")

    assert len(opener.calls) == 2
    assert summarize_attempt_journal(journal_path)["completed"] == 2
    assert actor_budget.consumed == 0
