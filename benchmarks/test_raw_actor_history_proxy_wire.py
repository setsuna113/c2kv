"""Local wire coverage for exact raw-output persistent AgentKV history."""

from __future__ import annotations

import copy
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

from benchmarks.model_identity import QWEN3_4B
from benchmarks.bfcl_response import normalize_native_message


RAW_ONE = '<tool_call>{"name":"lookup","arguments":{"b":2,"a":1}}</tool_call>'
MIXED_RAW_ONE = ('I will look it up.\n\n<tool_call>'
                 '{"name":"lookup","arguments":{"b":2,"a":1}}</tool_call>')
RAW_TWO = '<tool_call>{"name":"finish","arguments":{"ok":true}}</tool_call>'


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reply(handler, value, status=200):
    body = json.dumps(value).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class _ReferenceUpstream(BaseHTTPRequestHandler):
    paths = []
    chats = []
    sessions = []
    model_info_requests = 0
    native_text_output = False
    mixed_tool_call_content = False

    def log_message(self, *_args):
        return

    def do_GET(self):  # noqa: N802
        self.__class__.paths.append(self.path)
        if self.path != "/model_info":
            _reply(self, {"error": "unknown path"}, 404)
            return
        self.__class__.model_info_requests += 1
        _reply(self, {
            "model_type": "qwen3",
            "model_dimensions": QWEN3_4B,
            "model_path": "/qwen3-4b-wire-fixture",
        })

    def do_POST(self):  # noqa: N802
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size) or b"{}")
        self.__class__.paths.append(self.path)
        if self.path.startswith("/flush_cache"):
            _reply(self, "Cache flushed. fake")
            return
        if self.path == "/open_session":
            self.__class__.sessions.append(copy.deepcopy(payload))
            _reply(self, payload["session_id"])
            return
        if self.path != "/v1/chat/completions":
            _reply(self, {"error": "unknown path"}, 404)
            return

        self.__class__.chats.append(copy.deepcopy(payload))
        turn = len(self.__class__.chats)
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"b":2,"a":1}',
                    },
                }],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "finish", "arguments": '{"ok":true}'},
                }],
            },
            {"role": "assistant", "content": "done"},
        ][turn - 1]
        raw = [RAW_ONE, RAW_TWO, "done"][turn - 1]
        if self.__class__.mixed_tool_call_content and turn == 1:
            messages["content"] = "I will look it up."
            raw = MIXED_RAW_ONE
        if self.__class__.native_text_output:
            messages = {"role": "assistant", "content": raw, "tool_calls": []}
        _reply(self, {
            "id": f"chat-{turn}",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": messages,
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2},
            "metadata": {"persistent_history_session": {
                "continuation_mode": "exact_generated_prefix",
                "generated_text": raw,
            }},
        })


def _post(opener, url: str, payload: dict, *, expected_status=200) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener.open(request, timeout=10) as response:
            assert response.status == expected_status
            return json.load(response)
    except HTTPError as error:
        assert error.code == expected_status
        return json.loads(error.read())


@pytest.mark.parametrize("arm_name", ["commitkv", "agentkv"])
@pytest.mark.parametrize(("benchmark", "native_text_output", "mixed_tool_call_content"), [
    ("acebench", False, False),
    ("bfcl", True, False),
    ("toolsandbox", False, True),
])
def test_exact_history_three_turn_wire_replays_raw_text_and_rejects_mismatch(
        tmp_path, arm_name, benchmark, native_text_output, mixed_tool_call_content):
    _ReferenceUpstream.paths = []
    _ReferenceUpstream.chats = []
    _ReferenceUpstream.sessions = []
    _ReferenceUpstream.model_info_requests = 0
    _ReferenceUpstream.native_text_output = native_text_output
    _ReferenceUpstream.mixed_tool_call_content = mixed_tool_call_content
    upstream = ThreadingHTTPServer(("127.0.0.1", _free_port()), _ReferenceUpstream)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()

    proxy_port = _free_port()
    process = subprocess.Popen([
        sys.executable,
        str(Path(__file__).with_name("proxy.py")),
        "--upstream", f"http://127.0.0.1:{upstream.server_port}",
        "--backend", "sglang",
        "--benchmark", benchmark,
        "--arm", arm_name,
        "--model-family", "qwen3-4b",
        "--port", str(proxy_port),
        "--request-log", str(tmp_path / "proxy.jsonl"),
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

        url = f"http://127.0.0.1:{proxy_port}/v1/chat/completions"
        messages = [{"role": "user", "content": "Use the tools."}]
        first_source = copy.deepcopy(messages)
        first = _post(opener, url, {
            "model": "qwen3-4b", "messages": messages,
            "c2kv_measurement_session_id": "raw-wire-task",
        })
        assert messages == first_source

        echo_one = copy.deepcopy(first["choices"][0]["message"])
        if native_text_output:
            echo_one = normalize_native_message(echo_one, first["id"])
        if mixed_tool_call_content:
            # ToolSandbox stores only the executable calls and reconstructs
            # mixed assistant tool-call messages with empty content.
            echo_one["content"] = ""
        echo_one["tool_calls"][0]["function"]["arguments"] = '{ "a": 1, "b": 2 }'
        messages += [echo_one, {
            "role": "tool", "tool_call_id": echo_one["tool_calls"][0]["id"], "content": "lookup-result",
        }]
        second_source = copy.deepcopy(messages)
        second = _post(opener, url, {
            "model": "qwen3-4b", "messages": messages,
            "c2kv_measurement_session_id": "raw-wire-task",
        })
        assert messages == second_source

        echo_two = copy.deepcopy(second["choices"][0]["message"])
        if native_text_output:
            echo_two = normalize_native_message(echo_two, second["id"])
        echo_two["tool_calls"][0]["function"]["arguments"] = '{ "ok" : true }'
        messages += [echo_two, {
            "role": "tool", "tool_call_id": echo_two["tool_calls"][0]["id"], "content": "finish-result",
        }]
        third_source = copy.deepcopy(messages)
        third = _post(opener, url, {
            "model": "qwen3-4b", "messages": messages,
            "c2kv_measurement_session_id": "raw-wire-task",
        })
        assert third["choices"][0]["message"]["content"] == "done"
        assert messages == third_source

        assert len(_ReferenceUpstream.chats) == 3
        assert _ReferenceUpstream.model_info_requests == 1
        assert sum(path.startswith("/flush_cache")
                   for path in _ReferenceUpstream.paths) == 1
        assert len(_ReferenceUpstream.sessions) == 1
        session_ids = {
            chat["session_params"]["id"] for chat in _ReferenceUpstream.chats
        }
        assert session_ids == {_ReferenceUpstream.sessions[0]["session_id"]}

        second_wire = _ReferenceUpstream.chats[1]["messages"]
        third_wire = _ReferenceUpstream.chats[2]["messages"]
        first_raw = MIXED_RAW_ONE if mixed_tool_call_content else RAW_ONE
        assert {"role": "assistant", "content": first_raw} in second_wire
        assert {"role": "assistant", "content": first_raw} in third_wire
        assert {"role": "assistant", "content": RAW_TWO} in third_wire
        assert not any(message.get("tool_calls") for message in second_wire + third_wire)

        changed = copy.deepcopy(messages)
        changed[1]["tool_calls"][0]["function"]["name"] = "different_tool"
        failure = _post(opener, url, {
            "model": "qwen3-4b", "messages": changed,
            "c2kv_measurement_session_id": "raw-wire-task",
        }, expected_status=502)
        assert "unchanged prior assistant action" in failure["error"]
        assert len(_ReferenceUpstream.chats) == 3
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        upstream.shutdown()
        upstream.server_close()
