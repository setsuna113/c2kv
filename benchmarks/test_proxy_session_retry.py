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


@pytest.fixture(autouse=True)
def upstream_url(monkeypatch):
    monkeypatch.setattr(proxy, "UPSTREAM", "http://127.0.0.1:34999")


@pytest.mark.parametrize("failure", ["timeout", "http502", "http429"])
def test_advanced_session_is_not_replayed_after_lost_response(monkeypatch, failure):
    submitted = []
    events = []

    def open_request(request, timeout):
        submitted.append(json.loads(request.data))
        # The transport cannot tell whether the server committed this request.
        if failure == "timeout":
            raise TimeoutError("response lost after session advanced")
        code = int(failure.removeprefix("http"))
        raise HTTPError(request.full_url, code, "lost response", {},
                        io.BytesIO(b"upstream session outcome unknown"))

    monkeypatch.setattr(proxy, "_OPENER", SimpleNamespace(open=open_request))
    monkeypatch.setattr(proxy, "_measurement_event", lambda *args, **kwargs: events.append(kwargs))
    monkeypatch.setattr(proxy.time, "sleep", lambda _: pytest.fail("must not retry a session"))
    payload = {"session_params": {"id": "owned-history"}, "messages": [], "max_tokens": 8}
    with pytest.raises(proxy.UpstreamError):
        proxy._post_json("/v1/chat/completions", payload, 600)
    assert submitted == [payload]
    assert len(events) == 1 and events[0]["attempt"] == 0


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
