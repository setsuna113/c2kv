"""Native cap evidence survives message-only transport errors without model work."""
import json
import shutil
import time
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from benchmarks.native_budget_failure import native_budget_failure
from benchmarks.paper import c1


def evidence(server, *, code="decision_cap_reached"):
    server.mkdir(parents=True, exist_ok=True)
    ready = {"schema": "a-event-native-server-v1", "status": "ready", "benchmark": "tau2",
             "run_id": "cap-run", "allowed_task_ids": ["9"],
             "max_decisions": 96, "max_generation_calls": 96}
    row = {"schema": "a-event-native-budget-rejection-v1", "run_id": "cap-run",
           "task_id": "9", "session_id": "tau2/9/attempt-0", "status_code": 429,
           "code": code, "decisions_reserved": 96, "max_decisions": 96,
           "generation_calls_reserved": 96, "max_generation_calls": 96}
    (server / "ready.json").write_text(json.dumps(ready), encoding="utf-8")
    (server / "budget_rejections.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return row


@pytest.mark.parametrize("change", [
    {"run_id": "other-run"}, {"task_id": "10"}, {"session_id": "tau2/10/attempt-0"},
    {"status_code": 500}, {"code": "rate_limit_exceeded"}, {"decisions_reserved": 95},
    {"max_decisions": 95}, {"schema": "untrusted"},
])
def test_rejection_must_match_task_run_and_actual_cap(tmp_path, change):
    row = evidence(tmp_path)
    assert native_budget_failure(tmp_path, "9") == "decision_cap_reached"
    (tmp_path / "budget_rejections.jsonl").write_text(json.dumps(row | change) + "\n")
    assert native_budget_failure(tmp_path, "9") is None


def test_full_counters_without_rejection_are_not_a_budget_failure(tmp_path):
    evidence(tmp_path)
    (tmp_path / "budget_rejections.jsonl").unlink()
    (tmp_path / "final.json").write_text(json.dumps({"api_health": {
        "terminal_reason": "decision_cap_reached", "decisions_reserved": 96}}))
    assert native_budget_failure(tmp_path, "9") is None


def test_96_successes_then_message_only_429_is_task_local(tmp_path, monkeypatch):
    from benchmarks.adapters import tau2_adapter
    c1.load_delivery()
    from benchmarks.memory_runtime.attempt_journal import AttemptJournal, summarize_attempt_journal
    from benchmarks.memory_runtime.event_native_api import EventNativeAPI, make_server

    cell = tmp_path / "cell"
    native = cell / "native"
    native.mkdir(parents=True)
    shard = tmp_path / "captured" / "9"
    server_dir = shard / "server"
    evidence(server_dir)
    (server_dir / "budget_rejections.jsonl").unlink()
    journal = AttemptJournal(server_dir / "attempts.jsonl")

    class Runner:
        generation_calls = 0
        max_generation_calls = 96

        def run(self, payload):
            self.generation_calls += 1
            handle = journal.start("generation", self.generation_calls, payload["outer_request_id"],
                                   {"task_id": "9", "decision_id": payload["decision_key"]})
            journal.finish(handle, "completed")
            return {"schema": "a-event-native-exact-step-v1", "status": "ok",
                    **payload, "generation_trace": []}

    runner = Runner()
    api = EventNativeAPI(runner, run_id="cap-run", model_name="model", benchmark="tau2",
                         view_mode="static", max_new_tokens=1, allowed_task_ids=["9"],
                         max_decisions=96, deadline_monotonic=time.monotonic() + 60,
                         steps_path=server_dir / "steps.jsonl")
    api._validate_request = lambda p: (
        {"session_id": "tau2/9/attempt-0", "decision_key": f"turn-0/step-{p['step']}",
         "outer_request_id": f"request-{p['step']}"}, ("9", 0, p["step"]), str(p["step"]))
    api._openai_response = lambda record: {"ok": True}
    server = make_server(api)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        for index in range(97):
            request = Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                              data=json.dumps({"step": index}).encode(),
                              headers={"Content-Type": "application/json"})
            if index < 96:
                with urlopen(request, timeout=5) as response:
                    assert response.status == 200
            else:
                with pytest.raises(HTTPError) as rejected:
                    urlopen(request, timeout=5)
                assert rejected.value.code == 429
                message = json.load(rejected.value)["error"]["message"]
        assert "decision_cap_reached" not in message
        assert runner.generation_calls == api.decisions_reserved == 96
        steps = [json.loads(line) for line in (server_dir / "steps.jsonl").read_text().splitlines()]
        assert len(steps) == 96 and all(row["status"] == "ok" for row in steps)
        costs = summarize_attempt_journal(server_dir / "attempts.jsonl")
        assert costs["completed"] == 96 and costs["failed"] == costs["pending"] == 0
        (server_dir / "final.json").write_text(json.dumps({
            "status": "stopped", "journal_summary": costs, "api_health": api.health()}), encoding="utf-8")
        out = shard / "tau2"
        events = out / "measurement" / "harness_events.jsonl"
        events.parent.mkdir(parents=True)
        from benchmarks.tau2_instrumentation.c2kv_tau2_hook import _exception_details
        stripped = RuntimeError("RateLimitError: " + message)
        stripped.status_code = 429
        stripped.code = "429"  # Actual LiteLLM RateLimitError uses a status alias here.
        details = _exception_details(stripped)
        assert details["api_error_code"] is None
        events.write_text(json.dumps({"event_type": "decision", "episode_id": "9",
                                      "error": details}) + "\n")
        failures = tau2_adapter._declared_task_failures(
            out, [{"task_id": "9", "termination_reason": "infrastructure_error"}], 1,
            native_server_dir=server_dir)
        assert failures == {"9": "decision_cap_reached"}
        monkeypatch.setattr(c1, "load_delivery", lambda: None)
        monkeypatch.setattr(c1, "selected_tasks", lambda *_: ["9", "10"])
        monkeypatch.setattr(c1, "prepare_native", lambda *_: (native, None, None))
        calls = []

        def run_task(config, benchmark, task, *args):
            calls.append(task)
            if task == "9":
                shutil.copytree(shard, native / "task_shards" / task)
                raise RuntimeError("message-only transport exception")
            (native / "task_shards" / task).mkdir()
            return {"task_id": task, "status": "completed"}, {"official_score": 1.0}

        from benchmarks.paper import native_extra
        monkeypatch.setattr(native_extra, "run_task", run_task)
        c1.run_closed_loop({}, "tau2", cell)
        assert calls == ["9", "10"]
        receipt = json.loads((native / "task_shards" / "9" / "paper_task_result.json").read_text())
        assert receipt["failure"]["kind"] == "decision_cap_reached"
        assert receipt["unified_metrics"]["official_score"] == 0.0
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()
