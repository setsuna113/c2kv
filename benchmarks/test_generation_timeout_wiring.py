"""A frozen --generation-timeout reaches the persistent proxy and the BFCL client.

CPU only: the runner's prepared command is parsed by run.py, run.py starts the
real proxy process, and a fake engine proves which deadline the proxy applies
to one persistent generation. The BFCL leg checks the client read deadline.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from benchmarks import run as bench_run
from benchmarks.model_identity import QWEN3_4B
from benchmarks.paper import runner

REFERENCE_ATTENTION_BFCL_LONG = (
    "bfcl_long_context__commitkv", "bfcl_long_context__agentkv",
    "bfcl_long_context__commitkv_b768", "bfcl_long_context__agentkv_b768",
    "bfcl_long_context__history_kv_pyramidkv_r25_persistent",
)


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    """The same overlay prepared with and without an explicit deadline."""
    result = {}
    for name, extra in (("default", []), ("explicit", ["--generation-timeout", "1800"])):
        output = tmp_path_factory.mktemp(name) / "root"
        runner.main(["prepare", "--output", str(output), *extra,
                     "--history-kv-budget", "commitkv=768", "--history-kv-budget", "agentkv=768"])
        # Compare commands independently of where each root was prepared.
        text = (output / "commands.json").read_text(encoding="utf-8")
        text = text.replace(json.dumps(str(output))[1:-1], "ROOT")
        result[name] = {row["cell_id"]: row for row in json.loads(text)}
    return result


@pytest.mark.parametrize("cell_id", REFERENCE_ATTENTION_BFCL_LONG)
def test_prepared_reference_attention_commands_parse_to_the_frozen_deadline(plans, cell_id):
    explicit = plans["explicit"][cell_id]["command"]
    default = plans["default"][cell_id]["command"]
    assert bench_run.build_parser().parse_args(explicit[2:]).generation_timeout == 1800.0
    assert "--generation-timeout" not in default
    assert bench_run.build_parser().parse_args(default[2:]).generation_timeout == 600.0
    # the deadline is the only difference between the two prepared commands
    position = explicit.index("--generation-timeout")
    assert explicit[:position] + explicit[position + 2:] == default


def test_ordinary_cells_keep_their_commands(plans):
    for cell_id, row in plans["explicit"].items():
        if cell_id not in REFERENCE_ATTENTION_BFCL_LONG and "--generation-timeout" not in row["command"]:
            assert row["command"] == plans["default"][cell_id]["command"]


def test_bfcl_client_read_deadline_keeps_cleanup_headroom(plans, monkeypatch, tmp_path):
    bfcl_adapter = bench_run.ADAPTERS["bfcl"]
    args = bench_run.build_parser().parse_args(
        plans["explicit"]["bfcl_long_context__commitkv_b768"]["command"][2:])
    seen = {}
    monkeypatch.setattr(bfcl_adapter, "run_bfcl",
                        lambda *_args, **kwargs: seen.update(kwargs) or {"n": 0})
    monkeypatch.setattr(bfcl_adapter, "default_bfcl_dir", lambda: str(tmp_path))
    args.out = tmp_path / "cell"
    bfcl_adapter.run(bench_run.build_context(args, tmp_path / "proxy.jsonl"))
    assert seen["request_timeout"] == 1800.0
    timeout = bfcl_adapter.client_kwargs("http://proxy/v1", seen["request_timeout"])["timeout"]
    assert timeout.read == 1890.0 and timeout.connect == 8.0


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _SlowEngine(BaseHTTPRequestHandler):
    delay = 0.0
    aborts = []

    def log_message(self, *_args):
        return

    def _reply(self, value, status=200):
        body = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        self._reply({"model_type": "qwen3", "model_dimensions": QWEN3_4B,
                     "model_path": "/qwen3-4b-timeout-fixture"})

    def do_POST(self):  # noqa: N802
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        if self.path.startswith("/flush_cache"):
            self._reply("Cache flushed. fake")
        elif self.path == "/open_session":
            self._reply(payload["session_id"])
        elif self.path == "/abort_request":
            self.__class__.aborts.append(payload)
            self._reply({"rid": payload["rid"], "session_id": payload["session_id"],
                         "request_status": "aborted", "session_status": "closed"})
        else:
            time.sleep(self.__class__.delay)
            self._reply({"id": "chat-1", "object": "chat.completion",
                         "choices": [{"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": "done"}}],
                         "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                         "metadata": {"persistent_history_session": {
                             "continuation_mode": "exact_generated_prefix",
                             "generated_text": "done"}}})


@pytest.mark.parametrize(("deadline", "delay", "timed_out"), [(1.0, 3.0, True), (6.0, 1.0, False)])
def test_run_py_proxy_applies_the_deadline_to_a_persistent_generation(tmp_path, deadline, delay, timed_out):
    _SlowEngine.delay, _SlowEngine.aborts = delay, []
    engine = ThreadingHTTPServer(("127.0.0.1", _free_port()), _SlowEngine)
    threading.Thread(target=engine.serve_forever, daemon=True).start()
    port = _free_port()
    process, log = bench_run.start_proxy(
        f"http://127.0.0.1:{engine.server_port}", "commitkv", port, tmp_path,
        benchmark="bfcl", doc_packing="message", generation_timeout=deadline)
    try:
        request = Request(f"http://127.0.0.1:{port}/v1/chat/completions", method="POST",
                          headers={"Content-Type": "application/json"},
                          data=json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}],
                                           "c2kv_measurement_session_id": "task-1"}).encode("utf-8"))
        started = time.monotonic()
        try:
            with build_opener(ProxyHandler({})).open(request, timeout=30) as response:
                status = response.status
        except HTTPError as error:
            status = error.code
        elapsed = time.monotonic() - started
    finally:
        bench_run._stop_process(process)
        engine.shutdown()
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    if timed_out:
        assert status == 502 and deadline <= elapsed < delay
        assert rows[-1]["error"] == "upstream 0: timed out"
        assert len(_SlowEngine.aborts) == 1 and _SlowEngine.aborts[0]["close_session"] is True
    else:
        assert status == 200 and elapsed >= delay
        assert rows[-1]["status"] == "ok" and _SlowEngine.aborts == []
