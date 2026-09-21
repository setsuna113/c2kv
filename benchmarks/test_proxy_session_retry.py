"""A lost session response must not resubmit a mutating generation."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy
from backends.base import BackendError
from backends.sglang import SglangBackend


@pytest.fixture(autouse=True)
def upstream_url(monkeypatch):
    monkeypatch.setattr(proxy, "UPSTREAM", "http://127.0.0.1:34999")
    monkeypatch.setattr(proxy, "STATE", proxy.ProxyState())


@pytest.mark.parametrize("failure", ["timeout", "http502", "http429"])
def test_advanced_session_is_not_replayed_after_lost_response(monkeypatch, failure):
    submitted = []
    events = []
    cleanups = []

    def open_request(request, timeout):
        submitted.append(json.loads(request.data))
        # The transport cannot tell whether the server committed this request.
        if failure == "timeout":
            raise TimeoutError("response lost after session advanced")
        code = int(failure.removeprefix("http"))
        raise HTTPError(request.full_url, code, "lost response", {},
                        io.BytesIO(b"upstream session outcome unknown"))

    monkeypatch.setattr(proxy, "_OPENER", SimpleNamespace(open=open_request))
    monkeypatch.setattr(proxy, "BACKEND", SimpleNamespace(
        abort_history_request=lambda request_id, session_id, timeout: (
            cleanups.append((request_id, session_id, timeout))
            or {"request_status": "aborted", "session_status": "closed"}
        )))
    monkeypatch.setattr(proxy, "_measurement_event", lambda *args, **kwargs: events.append(kwargs))
    monkeypatch.setattr(proxy.time, "sleep", lambda _: pytest.fail("must not retry a session"))
    payload = {"rid": "attempt-1", "session_params": {"id": "owned-history"},
               "messages": [], "max_tokens": 8}
    with pytest.raises(proxy.UpstreamError):
        proxy._post_json("/v1/chat/completions", payload, 600)
    assert submitted == [payload]
    assert events[0]["attempt"] == 0
    if failure == "timeout":
        assert cleanups == [("attempt-1", "owned-history", proxy.SESSION_CLEANUP_TIMEOUT)]
        assert events[-1]["status"] == "terminal"
    else:
        assert cleanups == []
        assert len(events) == 1


def test_unresolved_timeout_cleanup_poisoned_episode_is_not_restarted(monkeypatch):
    monkeypatch.setattr(proxy, "_OPENER", SimpleNamespace(
        open=lambda request, timeout: (_ for _ in ()).throw(TimeoutError("lost"))))
    monkeypatch.setattr(proxy, "_measurement_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(proxy, "BACKEND", SimpleNamespace(
        supports_episode_reset=True,
        abort_history_request=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("cleanup timeout")),
    ))
    proxy.STATE.active_measurement_session = "episode-attempt-1"
    payload = {"rid": "attempt-1", "session_params": {"id": "owned-history"},
               "messages": [], "max_tokens": 8}

    with pytest.raises(proxy.UpstreamError, match="cleanup failed"):
        proxy._post_json("/v1/chat/completions", payload, 600)
    with pytest.raises(proxy.BackendError, match="episode is terminal"):
        proxy._activate_measurement_session("episode-attempt-1")


def test_new_attempt_boundary_closes_failed_session_before_reuse(monkeypatch):
    closed = []
    opened = []
    cleanups = []
    monkeypatch.setattr(proxy, "_OPENER", SimpleNamespace(
        open=lambda request, timeout: (_ for _ in ()).throw(TimeoutError("lost"))))
    monkeypatch.setattr(proxy, "_measurement_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(proxy, "BACKEND", SimpleNamespace(
        supports_episode_reset=True,
        close_history_session=closed.append,
        abort_history_request=lambda request_id, session_id, timeout: (
            cleanups.append((request_id, session_id, timeout))
            or {"request_status": "completed", "session_status": "closed"}
        ),
        open_history_session=lambda session_id: opened.append(session_id) or session_id,
        flush_cache=lambda timeout: None,
    ))
    proxy.STATE.active_measurement_session = "episode-attempt-1"
    proxy.STATE.history_sessions = {"conversation": "owned-history"}
    payload = {"rid": "request-attempt-1",
               "session_params": {"id": "owned-history"},
               "messages": [], "max_tokens": 8}

    # The first attempt's response is lost after the engine may have completed.
    with pytest.raises(proxy.UpstreamError):
        proxy._post_json("/v1/chat/completions", payload, 600)
    assert cleanups == [(
        "request-attempt-1", "owned-history", proxy.SESSION_CLEANUP_TIMEOUT)]
    assert proxy.STATE.history_sessions == {}
    with pytest.raises(proxy.BackendError, match="episode is terminal"):
        proxy._activate_measurement_session("episode-attempt-1")

    proxy._activate_measurement_session("episode-attempt-2")
    replacement = proxy._history_session_id("conversation")

    # The cleanup endpoint already closed attempt 1.  Attempt 2 opens a fresh
    # engine session; no code path replays the old generation/full history.
    assert closed == []
    assert opened == [replacement]
    assert replacement != "owned-history"
    assert proxy.STATE.history_sessions == {"conversation": replacement}
    assert proxy.STATE.active_measurement_session == "episode-attempt-2"
    assert proxy.STATE.failed_measurement_session is None


def test_sglang_cleanup_requires_exact_non_retried_lifecycle_receipt():
    calls = []

    def post_json(path, payload, timeout, retries=2):
        calls.append((path, payload, timeout, retries))
        return {
            "rid": payload["rid"],
            "session_id": payload["session_id"],
            "request_status": "completed",
            "session_status": "closed",
            "finish_reason": "stop",
        }

    backend = SglangBackend(post_json)
    receipt = backend.abort_history_request("request-1", "session-1", timeout=60.0)

    assert receipt["rid"] == "request-1"
    assert calls == [(
        "/abort_request",
        {"rid": "request-1", "session_id": "session-1",
         "wait_for_completion": True, "close_session": True, "timeout": 60.0},
        65,
        0,
    )]


@pytest.mark.parametrize("wrong_field", ["rid", "session_id"])
def test_sglang_cleanup_rejects_mismatched_lifecycle_identity(wrong_field):
    def post_json(path, payload, timeout, retries=2):
        result = {
            "rid": payload["rid"],
            "session_id": payload["session_id"],
            "request_status": "aborted",
            "session_status": "closed",
            "finish_reason": "abort",
        }
        result[wrong_field] = "some-other-id"
        return result

    backend = SglangBackend(post_json)
    with pytest.raises(BackendError, match="exact terminal request/session"):
        backend.abort_history_request("request-1", "session-1")


def test_stateless_transport_keeps_existing_retry_contract(monkeypatch):
    submitted = []
    delays = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"success":true}'

    def open_request(request, timeout):
        submitted.append(json.loads(request.data))
        if len(submitted) == 1:
            raise TimeoutError("temporary transport failure")
        return Response()

    monkeypatch.setattr(proxy, "_OPENER", SimpleNamespace(open=open_request))
    monkeypatch.setattr(proxy, "_measurement_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(proxy.time, "sleep", delays.append)
    payload = {"messages": [], "max_tokens": 8}
    assert proxy._post_json("/v1/chat/completions", payload, 600) == {"success": True}
    assert submitted == [payload, payload]
    assert delays == [2]
