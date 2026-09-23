"""A configured generation deadline must not let the engine reap a live session.

box4 s19kv, toolsandbox__commitkv_b512 (paper 98fd2eb, engine 36e969135,
--generation-timeout 3600): the agent's history session was opened with the
engine's 600 s idle timeout, which counts from the start of the session's last
request.  A ~600 s user-simulator turn ran in between, the engine closed the
session at 09:35:33 and the agent's next turn failed with "session id ... does
not exist".  The proxy now opens such sessions without an idle timeout and
still closes them itself; the default deadline keeps the old request bytes.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from benchmarks.backends.sglang import SglangBackend
from benchmarks.model_identity import QWEN3_4B


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reply(handler, value, status=200):
    body = b"" if value is None else json.dumps(value).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _SessionUpstream(BaseHTTPRequestHandler):
    opened = []  # raw /open_session bodies
    closed = []  # /close_session payloads
    chats = []

    def log_message(self, *_args):
        return

    def do_GET(self):  # noqa: N802
        if self.path != "/model_info":
            _reply(self, {"error": "unknown path"}, 404)
            return
        _reply(self, {"model_type": "qwen3", "model_dimensions": QWEN3_4B,
                      "model_path": "/qwen3-4b-session-fixture"})

    def do_POST(self):  # noqa: N802
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        payload = json.loads(raw or b"{}")
        if self.path.startswith("/flush_cache"):
            _reply(self, "Cache flushed. fake")
        elif self.path == "/open_session":
            self.__class__.opened.append(raw)
            _reply(self, payload["session_id"])
        elif self.path == "/close_session":
            self.__class__.closed.append(payload)
            _reply(self, None)
        elif self.path == "/v1/chat/completions":
            self.__class__.chats.append(payload)
            _reply(self, {
                "id": "chat-1", "object": "chat.completion",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "done"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                "metadata": {"persistent_history_session": {
                    "continuation_mode": "exact_generated_prefix",
                    "generated_text": "done",
                }},
            })
        else:
            _reply(self, {"error": "unknown path"}, 404)


def _post(opener, url: str, payload: dict) -> dict:
    request = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with opener.open(request, timeout=10) as response:
            return {"status": response.status, "body": json.load(response)}
    except HTTPError as error:
        return {"status": error.code, "body": json.loads(error.read())}


def _open_bytes(session_id: str, timeout) -> bytes:
    """The exact /open_session body; ``timeout=None`` means no idle timeout."""
    payload = {"capacity_of_str_len": 0, "session_id": session_id, "streaming": True}
    if timeout is not None:
        payload["timeout"] = timeout
    return json.dumps(payload).encode("utf-8")


@pytest.mark.parametrize(("deadline_args", "idle_timeout"), [
    ([], 600.0),                                # historical default
    (["--generation-timeout", "600"], 600.0),   # explicit default: same bytes
    (["--generation-timeout", "3600"], None),   # the s19kv overlay
])
def test_proxy_opens_session_idle_timeout_from_deadline(tmp_path, deadline_args, idle_timeout):
    _SessionUpstream.opened = []
    _SessionUpstream.closed = []
    _SessionUpstream.chats = []
    upstream = ThreadingHTTPServer(("127.0.0.1", _free_port()), _SessionUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    proxy_port = _free_port()
    process = subprocess.Popen([
        sys.executable, str(Path(__file__).with_name("proxy.py")),
        "--upstream", f"http://127.0.0.1:{upstream.server_port}",
        "--backend", "sglang", "--benchmark", "toolsandbox",
        "--arm", "commitkv", "--history-kv-target-tokens", "512",
        "--model-family", "qwen3-4b",
        "--port", str(proxy_port),
        "--request-log", str(tmp_path / "proxy.jsonl"),
        *deadline_args,
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    opener = build_opener(ProxyHandler({}))
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise AssertionError(f"proxy exited during startup: {output}")
            try:
                with socket.create_connection(("127.0.0.1", proxy_port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError("proxy did not become ready")

        base = f"http://127.0.0.1:{proxy_port}"
        episode = {"c2kv_measurement_session_id": "idle-task"}
        reply = _post(opener, base + "/v1/chat/completions", {
            "model": "qwen3-4b", **episode,
            "messages": [{"role": "user", "content": "Use the tools."}]})
        assert reply["status"] == 200, reply

        assert len(_SessionUpstream.opened) == 1
        session_id = json.loads(_SessionUpstream.opened[0])["session_id"]
        assert _SessionUpstream.opened[0] == _open_bytes(session_id, idle_timeout)
        assert [chat["session_params"]["id"] for chat in _SessionUpstream.chats] == [session_id]

        # Without an idle timeout the proxy's own episode close is what frees it.
        closed = _post(opener, base + "/close_measurement_session", episode)
        assert closed == {"status": 200, "body": {"closed_owned_sessions": True}}
        assert _SessionUpstream.closed == [{"session_id": session_id}]
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        upstream.shutdown()
        upstream.server_close()


def _recording_backend():
    calls = []

    def post_json(path, payload, timeout, retries=2):
        calls.append((path, json.dumps(payload).encode("utf-8"), timeout))
        return payload["session_id"]

    return SglangBackend(post_json), calls


def test_open_session_default_call_is_byte_identical():
    backend, calls = _recording_backend()
    assert backend.open_history_session("sess-default") == "sess-default"
    assert backend.open_history_session("sess-300", 300) == "sess-300"
    assert calls == [
        ("/open_session", _open_bytes("sess-default", 600.0), 600),
        ("/open_session", _open_bytes("sess-300", 300.0), 300),
    ]


def test_open_session_without_idle_expiry_keeps_the_request_bound():
    backend, calls = _recording_backend()
    assert backend.open_history_session("sess-owned", expire_idle=False) == "sess-owned"
    assert calls == [("/open_session", _open_bytes("sess-owned", None), 600)]
