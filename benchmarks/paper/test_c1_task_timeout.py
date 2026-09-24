"""A BFCL task wall timeout fails only that task and releases its engine session.

SZ BFCL Long b256 racer_v2 PyramidKV c1_v2_verified s2: task
multi_turn_long_context_107 was still inside a generation when run_c1's
10800 s task limit expired. subprocess.TimeoutExpired escaped the per-task
score-0 fallback and aborted the remaining shard. The controller, stopped
mid-request, never closed its persistent engine session.
"""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

from benchmarks.paper import c1 as paper_c1

SESSION = "racer-task-107"


@pytest.fixture
def engine():
    """A stand-in engine that records lifecycle calls."""
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, body))
            reply = (b"" if self.path == "/close_session" else json.dumps({
                "rid": body.get("rid"), "session_id": body.get("session_id"),
                "request_status": "aborted", "session_status": "closed"}).encode())
            self.send_response(200)
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_address[1]}", calls=calls)
    server.shutdown()
    server.server_close()


def _http_rows(*rows):
    return "".join(json.dumps(row) + "\n" for row in rows)


def _request(rid):
    return {"event": "request", "schema": "racer-chat-http-v1", "request": {
        "rid": rid, "c2kv_kv_memory_hint": {"persistent_history_session": {"session_id": SESSION}}}}


def _server_script(tmp_path):
    # Writes the ready file and an in-flight generation, then blocks like a
    # controller waiting on the engine.
    script = tmp_path / "blocked_controller.py"
    script.write_text(
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "server = Path(sys.argv[1])\n"
        "server.mkdir(parents=True, exist_ok=True)\n"
        f"(server / 'sglang_http.jsonl').write_text({_http_rows(_request('attempt-1'))!r})\n"
        "(server / 'ready.json').write_text(json.dumps({'status': 'ready'}))\n"
        "time.sleep(120)\n", encoding="utf-8")
    return script


def test_wall_timeout_is_task_local_and_aborts_the_orphaned_request(tmp_path, monkeypatch, engine):
    delivery = paper_c1.load_delivery()
    cell = tmp_path / "cell"
    native = cell / "native"
    native.mkdir(parents=True)
    script = _server_script(tmp_path)
    args = SimpleNamespace(out=native, benchmark="bfcl", task_timeout=3, method="c2kv_only",
                           tool_memory="none", sglang_backend_url=engine.url)

    def commands(_args, task, _controller):
        server = native / "task_shards" / task / "server"
        return ([sys.executable, str(script), str(server)],
                [sys.executable, "-c", "import time; time.sleep(120)"])

    real_run_task = delivery.run_task

    def run_task(run_args, task, controller, **kwargs):
        if task == "multi_turn_long_context_108":
            (native / "task_shards" / task).mkdir(parents=True)
            metrics = {"task_id": task, "official_score": 1.0, "normal_termination": True}
            return {"task_id": task, "status": "completed", "unified_metrics": metrics}, metrics
        return real_run_task(run_args, task, controller, **kwargs)

    monkeypatch.setattr(delivery, "commands_for_task", commands)
    monkeypatch.setattr(delivery, "run_task", run_task)
    monkeypatch.setattr(paper_c1, "load_delivery", lambda: delivery)
    monkeypatch.setattr(paper_c1, "selected_tasks", lambda *_args: [
        "multi_turn_long_context_107", "multi_turn_long_context_108"])
    monkeypatch.setattr(paper_c1, "prepare_native",
                        lambda *_args: (native, args, tmp_path / "controller.json"))

    assert paper_c1.run_closed_loop({}, "bfcl_long_context", cell) == native
    shards = native / "task_shards"
    failed = json.loads((shards / "multi_turn_long_context_107" / "paper_task_result.json").read_text())
    assert (failed["status"], failed["failure"]["kind"]) == ("harness_failure", "task_timeout")
    assert failed["unified_metrics"]["official_score"] == 0.0
    assert failed["failure"]["timeout_seconds"] == 3
    assert failed["failure"]["engine_cleanup"]["status"] == "aborted_in_flight_request"
    assert engine.calls == [("/abort_request", {
        "rid": "attempt-1", "session_id": SESSION, "wait_for_completion": True,
        "close_session": True, "timeout": 60.0})]
    completed = json.loads((shards / "multi_turn_long_context_108" / "paper_task_result.json").read_text())
    assert completed["status"] == "completed"
    summary = json.loads((cell / f"summary_{paper_c1.ARM}.json").read_text())
    assert summary["harness_failure_task_ids"] == ["multi_turn_long_context_107"]


def test_engine_release_receipts(tmp_path, engine):
    delivery = paper_c1.load_delivery()
    args = SimpleNamespace(sglang_backend_url=engine.url)
    server = tmp_path / "server"
    server.mkdir()
    release = delivery.release_engine_session

    (server / "final.json").write_text(json.dumps({"session_cache_after_close": {"closed": True}}))
    assert release(args, server) == {"status": "closed_by_controller"}
    (server / "final.json").unlink()

    (server / "sglang_http.jsonl").write_text(_http_rows({"event": "request"}))
    assert release(args, server) == {"status": "no_persistent_session"}

    # The last generation answered; only the session is left open. A killed
    # writer may leave its final line cut.
    (server / "sglang_http.jsonl").write_text(
        _http_rows(_request("attempt-1"), {"event": "response", "response": {}}) + '{"event": "req')
    assert release(args, server) == {"status": "closed_idle_session", "session_id": SESSION}
    assert engine.calls[-1] == ("/close_session", {"session_id": SESSION})

    unreachable = SimpleNamespace(sglang_backend_url="http://127.0.0.1:9")
    (server / "sglang_http.jsonl").write_text(_http_rows(_request("attempt-2")))
    receipt = release(unreachable, server, timeout=1)
    assert receipt["status"] == "failed" and receipt["session_id"] == SESSION
