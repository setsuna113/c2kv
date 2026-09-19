"""End-to-end proxy contract of the tool-memory axis against a fake SGLang.

Real proxy process, real Qwen3 tokenizer (the pinned base snapshot in the HF
cache; skipped when absent), fake upstream that records what reaches the
server.  Checks the wire shape the server-side change relies on:
``/v1/c2kv/extract`` gets exact ``token_ids`` with ``projection_set="tool"``,
the chat request keeps ``tools`` but sets ``c2kv_tools_in_prompt=false``, the
leading system message carries the explicit protocol with the native schemas,
and the gist carriers sit right after the system prefix with no proxy-only
fields.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toolmemory  # noqa: E402
import proxy  # noqa: E402

pytest.importorskip("transformers")

SNAPSHOTS = sorted(glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/*/tokenizer.json")))
pytestmark = pytest.mark.skipif(not SNAPSHOTS, reason="pinned Qwen3-4B tokenizer snapshot not cached")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _FakeSGLang(BaseHTTPRequestHandler):
    extracts = []
    chats = []

    def log_message(self, *_args):
        return

    def _reply(self, body):
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self):  # noqa: N802
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size))
        if self.path == "/v1/c2kv/extract":
            self.__class__.extracts.append(payload)
            ids = payload["token_ids"]
            ratio = payload["compression_ratio"]
            key = hashlib.sha256(json.dumps([ids, ratio, payload.get("projection_set")]).encode()).hexdigest()
            self._reply({"key_hash": key, "gist_len": (len(ids) + ratio - 1) // ratio,
                         "original_seq_len": len(ids), "cache_hit": False, "success": True})
            return
        self.__class__.chats.append(payload)
        self._reply({
            "id": "fake", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "metadata": {"sglang_runtime": {"kv_resident_tokens": 10}},
        })


def _post(url, payload):
    request = Request(url, data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except HTTPError as error:
        raise AssertionError(f"proxy returned HTTP {error.code}: {error.read().decode('utf-8', 'replace')}")


def _tool_checkpoint(tmp_path):
    snapshot = Path(SNAPSHOTS[0]).parent
    checkpoint = tmp_path / "checkpoint-1034"
    checkpoint.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if (snapshot / name).exists():
            shutil.copy2(snapshot / name, checkpoint / name)
    (checkpoint / "config.json").write_text(json.dumps({
        "history_memory_compression_domain": "tool", "history_memory_variant": "T0",
        "history_memory_render_profile": toolmemory.RENDER_PROFILE,
        "history_memory_supported_ratios": [8, 12]}), encoding="utf-8")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1034}), encoding="utf-8")
    return checkpoint


TOOLS = [
    {"type": "function", "function": {"name": "book_flight", "description": "Book a flight",
                                      "parameters": {"type": "object", "properties": {"dest": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "cancel_order", "description": "Cancel an order",
                                      "parameters": {"type": "object", "properties": {"id": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}},
]


def test_tool_memory_proxy_end_to_end(tmp_path):
    _FakeSGLang.extracts = []
    _FakeSGLang.chats = []
    upstream = ThreadingHTTPServer(("127.0.0.1", _free_port()), _FakeSGLang)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    checkpoint = _tool_checkpoint(tmp_path)
    proxy_port = _free_port()
    log_path = tmp_path / "proxy.jsonl"
    process = subprocess.Popen([
        sys.executable, str(Path(__file__).with_name("proxy.py")),
        "--upstream", f"http://127.0.0.1:{upstream.server_port}",
        "--backend", "sglang", "--benchmark", "bfcl", "--arm", "full",
        "--port", str(proxy_port), "--request-log", str(log_path),
        "--tool-memory", "t0:r8:hybrid1", "--tool-checkpoint", str(checkpoint),
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.time() + 60
        while time.time() < deadline:
            if process.poll() is not None:
                raise AssertionError(f"proxy exited during startup: {process.stdout.read()}")
            try:
                with socket.create_connection(("127.0.0.1", proxy_port), timeout=1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError("proxy did not become ready")
        payload = {"model": "c2kv-agent", "tools": TOOLS, "messages": [
            {"role": "system", "content": "You are a booking agent."},
            {"role": "user", "content": "cancel_order 42 please"},
        ]}
        answer = _post(f"http://127.0.0.1:{proxy_port}/v1/chat/completions", payload)
        assert answer["choices"][0]["message"]["content"] == "ok"
        info = answer["c2kv_proxy"]["tool_memory"]
        # tools-only request: no compressible remainder -> None, served raw
        bare = {"model": "c2kv-agent", "messages": [{"role": "user", "content": "hi"}]}
        assert _post(f"http://127.0.0.1:{proxy_port}/v1/chat/completions", bare)["c2kv_proxy"]["tool_memory"] is None
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        upstream.shutdown()
        upstream.server_close()

    # ---- extraction wire shape: exact token ids, tool projection set, ratio 8
    assert len(_FakeSGLang.extracts) == 2, _FakeSGLang.extracts
    for extract in _FakeSGLang.extracts:
        assert extract["projection_set"] == "tool" and extract["compression_ratio"] == 8
        assert isinstance(extract["token_ids"], list) and all(isinstance(t, int) for t in extract["token_ids"])
        assert "text" not in extract or extract["text"] == ""
    # the chunks are the T0 document envelopes rendered as user messages
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    snapshots = [toolmemory.tool_snapshot(t) for t in TOOLS]
    remainder = [i for i in range(3) if i != 1]           # cancel_order is the native top-1
    for extract, index in zip(_FakeSGLang.extracts, remainder):
        document = {"type": "tool_definition", "tool_index": index, "tool": snapshots[index]}
        expected = tokenizer.apply_chat_template(
            [{"role": "user", "content": toolmemory.document_envelope(document)}], tokenize=True,
            add_generation_prompt=False, enable_thinking=False, truncation=False)
        if hasattr(expected, "input_ids"):
            expected = expected.input_ids
        assert extract["token_ids"] == list(expected)

    # ---- chat wire shape
    chat = _FakeSGLang.chats[0]
    assert chat["c2kv_tools_in_prompt"] is False
    assert chat["tools"] == TOOLS                          # kept for the server's tool-call parser
    messages = chat["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a booking agent.\n\n" + toolmemory.protocol_block([snapshots[1]])
    carriers = messages[1:3]
    assert [m["c2kv_key_hash"] for m in carriers] == [
        hashlib.sha256(json.dumps([e["token_ids"], 8, "tool"]).encode()).hexdigest()
        for e in _FakeSGLang.extracts]
    assert all(set(m) == {"role", "content", "c2kv_key_hash", "c2kv_ratio"} and m["content"] == "" for m in carriers)
    assert messages[3] == {"role": "user", "content": "cancel_order 42 please"}
    boundary = chat["c2kv_kv_memory_hint"]["paper_measurement"]
    assert (boundary["history_start_message_count"], boundary["history_message_count"]) == (0, 0)
    # the bare request (no tools) only gets the proxy's usual default system
    # prompt; no protocol, no carriers, no c2kv_tools_in_prompt
    bare_messages = _FakeSGLang.chats[1]["messages"]
    assert bare_messages[-1] == {"role": "user", "content": "hi"} and len(bare_messages) == 2
    assert bare_messages[0]["role"] == "system" and toolmemory.TOOL_PROTOCOL_HEAD not in bare_messages[0]["content"]
    assert "c2kv_tools_in_prompt" not in _FakeSGLang.chats[1]

    # ---- accounting
    assert info["n_tools"] == 3 and info["native_indices"] == [1] and info["native_names"] == ["cancel_order"]
    assert info["n_chunks"] == 2 and info["gist_tokens"] == sum(
        (len(e["token_ids"]) + 7) // 8 for e in _FakeSGLang.extracts)
    assert info["raw_tool_prologue_tokens"] > info["resident_tool_tokens"] > 0
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[0]["tool_memory"]["spec"] == "t0_r8_hybrid1" and rows[0]["status"] == "ok"


def test_tool_memory_refuses_budget_adapted_text_arms():
    """Budget-adapted arms preflight the raw actor payload (tool prologue
    included) through the server's chat budget renderer; the proxy refuses the
    combination before touching the checkpoint or the upstream."""
    with pytest.raises(SystemExit, match="budget-adapted"):
        proxy.main([
            "--upstream", "http://127.0.0.1:1", "--backend", "sglang",
            "--arm", "hiagent_full_b4096", "--port", "1",
            "--tool-memory", "t0:r8", "--tool-checkpoint", "/nope"])
