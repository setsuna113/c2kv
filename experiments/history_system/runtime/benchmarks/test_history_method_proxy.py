from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FakeUpstream(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *_args):
        return

    def do_POST(self):  # noqa: N802
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size))
        self.__class__.requests.append(payload)
        messages = payload.get("messages") or []
        is_compressor = any(
            "append-only agent memory" in str(message.get("content", ""))
            for message in messages if message.get("role") == "system"
        )
        if is_compressor:
            content = "MEMORY: get_weather Cambridge -> 12 C"
        else:
            content = "forwarded"
        body = {
            "id": "fake",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {
                "role": "assistant", "content": content,
            }, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(messages), "completion_tokens": 1},
        }
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _post(url: str, payload: dict) -> dict:
    request = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read())
    except HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise AssertionError(f"proxy returned HTTP {error.code}: {detail}") from error


def test_agentfold_proxy_end_to_end(tmp_path):
    _FakeUpstream.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", _free_port()), _FakeUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    proxy_port = _free_port()
    log_path = tmp_path / "proxy.jsonl"
    proxy = Path(__file__).with_name("proxy.py")
    process = subprocess.Popen([
        sys.executable, str(proxy),
        "--upstream", f"http://127.0.0.1:{upstream.server_port}",
        "--backend", "hfserver", "--benchmark", "acebench",
        "--arm", "agentfold", "--model-family", "qwen3-4b",
        "--port", str(proxy_port), "--request-log", str(log_path),
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise AssertionError(f"proxy exited during startup: {output}")
            try:
                with socket.create_connection(("127.0.0.1", proxy_port), timeout=1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError("proxy did not become ready")

        first = {"model": "qwen3-4b", "messages": [
            {"role": "user", "content": "Find the weather for Cambridge."},
        ]}
        assert _post(f"http://127.0.0.1:{proxy_port}/v1/chat/completions", first)["choices"]

        second = {"model": "qwen3-4b", "messages": [
            {"role": "user", "content": "Find the weather for Cambridge."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-1", "type": "function",
                "function": {"name": "get_weather", "arguments": "{city: Cambridge}"},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "12 C"},
            {"role": "user", "content": "Continue with the result."},
        ]}
        assert _post(f"http://127.0.0.1:{proxy_port}/v1/chat/completions", second)["choices"]
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        upstream.shutdown()
        upstream.server_close()

    compressor_requests = [
        request for request in _FakeUpstream.requests
        if any("append-only agent memory" in str(message.get("content", ""))
               for message in request.get("messages", [])
               if message.get("role") == "system")
    ]
    assert len(compressor_requests) == 1
    forwarded = [
        request for request in _FakeUpstream.requests
        if request.get("messages") and request["messages"][-1].get("content")
        == "Continue with the result."
    ]
    assert forwarded
    assert any("[agentfold granular memory unit]" in str(message.get("content"))
               for message in forwarded[-1]["messages"])
