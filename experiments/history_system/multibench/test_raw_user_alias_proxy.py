from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest


MODULE_PATH = Path(__file__).with_name("raw_user_alias_proxy.py")
SPEC = importlib.util.spec_from_file_location("raw_user_alias_proxy", MODULE_PATH)
assert SPEC and SPEC.loader
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)


class UpstreamHandler(BaseHTTPRequestHandler):
    bodies: list[dict] = []
    response_status = 200

    def log_message(self, _format, *_args):
        return

    def do_GET(self):  # noqa: N802
        body = json.dumps({"object": "list", "data": [{"id": "c2kv-agent"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.__class__.bodies.append(json.loads(body))
        response = json.dumps({"seen": len(self.__class__.bodies)}).encode()
        self.send_response(self.__class__.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def _start(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


def _request_json(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urlrequest.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlrequest.build_opener(urlrequest.ProxyHandler({})).open(req, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urlerror.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_alias_proxy_only_changes_model_and_never_retries(tmp_path):
    UpstreamHandler.bodies = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    _start(upstream)
    log = tmp_path / "requests.jsonl"
    server = P.AliasServer(
        ("127.0.0.1", 0), P.AliasHandler,
        upstream=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
        upstream_model="c2kv-agent",
        aliases=("candidate", "gpt-4o-2024-05-13"),
        request_log=log,
        timeout=5,
    )
    _start(server)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    payload = {
        "model": "candidate",
        "messages": [{"role": "user", "content": "secret marker"}],
        "tools": [{"type": "function", "function": {"name": "x"}}],
        "temperature": 0.37,
        "presence_penalty": 0.5,
        "seed": 7,
        "stream": False,
    }
    status, _ = _request_json(base + "/v1/chat/completions", payload)
    assert status == 200
    assert len(UpstreamHandler.bodies) == 1
    expected = dict(payload, model="c2kv-agent")
    assert UpstreamHandler.bodies[0] == expected
    log_text = log.read_text(encoding="utf-8")
    assert "secret marker" not in log_text
    row = json.loads(log_text)
    assert row["message_count"] == 1 and row["tool_count"] == 1

    UpstreamHandler.response_status = 503
    status, _ = _request_json(base + "/chat/completions", dict(payload, model="gpt-4o-2024-05-13"))
    assert status == 503
    assert len(UpstreamHandler.bodies) == 2

    status, _ = _request_json(base + "/v1/chat/completions", dict(payload, model="wrong"))
    assert status == 400
    assert len(UpstreamHandler.bodies) == 2
    server.shutdown()
    upstream.shutdown()


def test_alias_proxy_models_and_health(tmp_path):
    UpstreamHandler.response_status = 200
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    _start(upstream)
    server = P.AliasServer(
        ("127.0.0.1", 0), P.AliasHandler,
        upstream=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
        upstream_model="c2kv-agent", aliases=("candidate",),
        request_log=tmp_path / "requests.jsonl", timeout=5,
    )
    _start(server)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    status, models = _request_json(base + "/v1/models")
    assert status == 200
    assert [row["id"] for row in models["data"]] == ["candidate", "c2kv-agent"]
    assert _request_json(base + "/health") == (200, {"status": "ok", "schema": P.SCHEMA})
    server.shutdown()
    upstream.shutdown()
