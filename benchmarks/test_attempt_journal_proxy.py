"""Durable cost records at the real proxy transport/producer seams."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy
from arms import get_arm
from memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal
from test_exact_recovery_proxy import _ExactRuntime, _synthetic_body
from test_memory_runtime_proxy import _Backend, _payload


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.body).encode("utf-8")


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", proxy.GenerationBudget(2))
    monkeypatch.setattr(proxy, "EXTRACTION_BUDGET", proxy.ExtractionBudget(2))
    monkeypatch.setattr(proxy, "ATTEMPT_JOURNAL", None)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", None)
    monkeypatch.setattr(proxy, "UPSTREAM", "http://127.0.0.1:1")


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("with_trace", [False, True])
def test_extraction_miss_starts_before_producer_hit_is_free_failure_is_finished(
    monkeypatch, tmp_path, with_trace
):
    path = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(proxy, "ATTEMPT_JOURNAL", AttemptJournal(path))
    cache = proxy.ExtractCache()
    key = ("user", "content_hash", 4, "tools_hash")

    def producer():
        assert _rows(path)[-1]["event"] == "started"
        return {"key_hash": "gist", "original_seq_len": 8, "gist_len": 2}

    def failed():
        assert _rows(path)[-1]["event"] == "started"
        raise ValueError("PRIVATE_ERROR_TEXT")

    with proxy.capture_extractions(enabled=with_trace):
        assert cache.get_or_put(key, producer) == cache.get_or_put(key, producer)
        assert len(_rows(path)) == 2
        with pytest.raises(ValueError, match="PRIVATE_ERROR_TEXT"):
            cache.get_or_put(key, failed, force=True)
        with pytest.raises(proxy.ExtractionBudgetExceeded):
            cache.get_or_put(key, failed, force=True)
    rows = _rows(path)
    assert len(rows) == 4
    assert [row["status"] for row in rows if row["event"] == "finished"] == [
        "completed", "failed"]
    assert "PRIVATE_ERROR_TEXT" not in path.read_text(encoding="utf-8")


def test_transport_failure_is_recorded_once_and_cap_rejection_is_not_started(
    monkeypatch, tmp_path
):
    path = tmp_path / "attempts.jsonl"
    monkeypatch.setattr(proxy, "ATTEMPT_JOURNAL", AttemptJournal(path))
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", proxy.GenerationBudget(1))
    calls = []

    class Opener:
        def open(self, *args, **kwargs):
            assert _rows(path)[-1]["event"] == "started"
            calls.append(1)
            raise proxy.URLError("PRIVATE_TRANSPORT_TEXT")

    monkeypatch.setattr(proxy, "_OPENER", Opener())
    with pytest.raises(proxy.UpstreamError):
        proxy._post_json("/v1/chat/completions", {}, 1)
    with pytest.raises(proxy.GenerationBudgetExceeded):
        proxy._post_json("/v1/chat/completions", {}, 1)
    assert calls == [1]
    assert [row["event"] for row in _rows(path)] == ["started", "finished"]
    assert _rows(path)[-1]["status"] == "failed"
    assert "PRIVATE_TRANSPORT_TEXT" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("fail_at", ["start", "finish"])
def test_journal_io_failure_never_retries_transport(monkeypatch, fail_at):
    calls = []

    class Journal:
        def start(self, *args):
            if fail_at == "start":
                raise OSError("journal write failed")
            return object()

        def finish(self, *args, **kwargs):
            raise OSError("journal write failed")

    class Opener:
        def open(self, *args, **kwargs):
            calls.append(1)
            return _Response({"usage": {"completion_tokens": 1}})

    monkeypatch.setattr(proxy, "ATTEMPT_JOURNAL", Journal())
    monkeypatch.setattr(proxy, "_OPENER", Opener())
    with pytest.raises(OSError, match="journal write failed"):
        proxy._post_json("/v1/chat/completions", {}, 1, retries=2)
    assert len(calls) == int(fail_at == "finish")


def _exercise_exact_proxy(directory, interrupt_second=False):
    directory = Path(directory)
    journal_path = directory / "attempts.jsonl"
    request_path = directory / "proxy.jsonl"
    bodies = [
        _synthetic_body("PRIVATE_DRAFT", "draft", {"prompt_tokens": 10, "completion_tokens": 2}),
        _synthetic_body("PRIVATE_FINAL", "final", {"prompt_tokens": 20, "completion_tokens": 3}),
    ]

    class Opener:
        calls = 0

        def open(self, *args, **kwargs):
            self.calls += 1
            assert _rows(journal_path)[-1]["event"] == "started"
            if interrupt_second and self.calls == 2:
                (directory / "second_started").write_text("ready", encoding="utf-8")
                while True:
                    time.sleep(1)
            return _Response(bodies[self.calls - 1])

    raw = json.dumps(_payload()).encode("utf-8")
    handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    handler.rfile = io.BytesIO(raw)
    handler.headers = {"Content-Length": str(len(raw))}
    handler.path = "/v1/chat/completions"
    sent = []
    handler._send_json = lambda code, body: sent.append((code, body))
    with patch.multiple(
        proxy, STATE=proxy.ProxyState(), CACHE=proxy.ExtractCache(),
        GENERATION_BUDGET=proxy.GenerationBudget(2),
        EXTRACTION_BUDGET=proxy.ExtractionBudget(2),
        ATTEMPT_JOURNAL=AttemptJournal(journal_path),
        MEMORY_RUNTIME=_ExactRuntime("gap"), MEMORY_RUNTIME_FATAL_ERROR=None,
        MEMORY_RUNTIME_BYTES_PER_KV_TOKEN=None,
        ARM=get_arm("c2kv4"), BACKEND=_Backend(), UPSTREAM="http://127.0.0.1:1",
        REQUEST_LOG_PATH=str(request_path), _OPENER=Opener(),
    ):
        handler.do_POST()
    return sent


def test_real_exact_transport_records_two_attempts_with_one_request_id(tmp_path):
    sent = _exercise_exact_proxy(tmp_path)
    assert sent[0][0] == 200
    rows = _rows(tmp_path / "attempts.jsonl")
    request = _rows(tmp_path / "proxy.jsonl")[0]
    assert [row["event"] for row in rows] == ["started", "finished", "started", "finished"]
    starts = [row for row in rows if row["event"] == "started"]
    assert [row["attempt_index"] for row in starts] == [1, 2]
    assert {row["request_id"] for row in starts} == {request["request_id"]}
    assert all(row["eval_context"]["task_id"] == request["eval_context"]["task_id"] for row in starts)
    finishes = [row for row in rows if row["event"] == "finished"]
    assert [row["usage"]["completion_tokens"] for row in finishes] == [2, 3]
    assert "PRIVATE_" not in (tmp_path / "attempts.jsonl").read_text(encoding="utf-8")
    assert proxy._ATTEMPT_REQUEST.get() is None


def test_terminated_exact_second_generation_preserves_first_usage_and_pending(tmp_path):
    code = "from test_attempt_journal_proxy import _exercise_exact_proxy; import sys; _exercise_exact_proxy(sys.argv[1], True)"
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parent))
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 15
        while not (tmp_path / "second_started").exists() and time.monotonic() < deadline:
            if child.poll() is not None:
                out, error = child.communicate()
                pytest.fail(f"Child stopped before the pending attempt: {out!r} {error!r}")
            time.sleep(0.02)
        assert (tmp_path / "second_started").exists()
    finally:
        if child.poll() is None:
            child.terminate()
        child.communicate(timeout=5)
    rows = _rows(tmp_path / "attempts.jsonl")
    assert [row["event"] for row in rows] == ["started", "finished", "started"]
    assert rows[1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 2}
    assert not (tmp_path / "proxy.jsonl").exists()
    summary = summarize_attempt_journal(tmp_path / "attempts.jsonl")
    assert summary["started"] == 2
    assert summary["finished"] == 1
    assert summary["pending"] == 1
    assert summary["finished_usage_totals"] == {
        "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": None}
    assert summary["finished_usage_observed_attempts"]["completion_tokens"] == 1


def test_main_enables_journal_without_matching_proxy_log_glob(monkeypatch, tmp_path):
    class Server:
        def __init__(self, *args):
            pass

        def serve_forever(self):
            proxy._start_attempt("generation", 1)

    monkeypatch.setattr(proxy, "ThreadingHTTPServer", Server)
    monkeypatch.setattr(proxy, "get_backend", lambda *args: _Backend())
    proxy.main(["--upstream", "http://127.0.0.1:1", "--arm", "full", "--port", "39999",
                "--request-log", str(tmp_path / "proxy_full_39999.jsonl")])
    assert (tmp_path / "attempts_proxy_full_39999.jsonl").is_file()
    assert list(tmp_path.glob("proxy_*.jsonl")) == []
