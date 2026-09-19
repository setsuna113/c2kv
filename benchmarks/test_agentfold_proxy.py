"""Local wire test: actor folding and environment action share one response."""
import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler

from benchmarks.model_identity import QWEN3_4B


def test_joint_actor_response_through_proxy(tmp_path):
    seen = []

    class Actor(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, data):
            body = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply({"model_type": "qwen3", "model_dimensions": QWEN3_4B,
                        "model_path": "/qwen3-4b-fixture"})

        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            content = "print(1)" if len(seen) == 1 else (
                '<compress>{"compress_range":[0,0],"compress_text":"one done"}</compress>print(2)')
            self.reply({"choices": [{"message": {"role": "assistant", "content": content},
                                     "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 30, "completion_tokens": 20}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Actor)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("proxy.py")),
        "--arm", "agentfold", "--backend", "hfserver", "--benchmark", "acon_appworld",
        "--upstream", f"http://127.0.0.1:{server.server_port}", "--port", str(port),
        "--request-log", str(tmp_path / "requests.jsonl")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    opener = build_opener(ProxyHandler({}))
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise AssertionError(process.stdout.read())
            try:
                with opener.open(f"http://127.0.0.1:{port}/health", timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        messages = [{"role": "user", "content": "Compute"}]
        for step in range(2):
            request = Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                              data=json.dumps({"model": "same-actor", "messages": messages,
                                               "c2kv_measurement_session_id": "task"}).encode(),
                              headers={"Content-Type": "application/json"})
            with opener.open(request, timeout=10) as r:
                result = json.load(r)
            message = result["choices"][0]["message"]
            assert message["content"] == f"print({step + 1})"
            messages += [message, {"role": "user", "content": str(step + 1)}]
        assert len(seen) == 2
        assert all(req["model"] == "same-actor" for req in seen)
        assert "[Step 0 to 0]" in str(seen[1]["messages"])
    finally:
        process.terminate()
        process.wait(timeout=5)
        server.shutdown()
        server.server_close()
