"""A recovery-consuming generation cap remains a typed, durable task outcome."""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from threading import Thread
from types import SimpleNamespace
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime import budget_guard, event_native_server, event_native_step
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_api import EventNativeAPI, EventNativeAPIError, make_server
from benchmarks.memory_runtime.event_native_draft import NativeDraft
from benchmarks.memory_runtime.event_native_step import (
    EventNativeDecisionRunner, EventNativeStepError, GenerationCallCapExceeded,
)


def _api(tmp_path, runner, *, max_decisions=5):
    api = EventNativeAPI(
        runner, run_id="cap-test", model_name="model", view_mode="static",
        max_new_tokens=1, allowed_task_ids=["task"], max_decisions=max_decisions,
        deadline_monotonic=time.monotonic() + 60,
        steps_path=tmp_path / "steps.jsonl",
    )
    api._validate_request = lambda payload: (
        {"session_id": "task", "decision_key": payload["decision_key"],
         "outer_request_id": payload["decision_key"]},
        ("task", 0, int(payload["decision_key"][1:])), payload["decision_key"],
    )
    api._openai_response = lambda record: {"status": record["status"]}
    return api


def test_recovery_uses_last_call_then_next_decision_gets_repeatable_http_429(
        tmp_path, monkeypatch):
    memory = SimpleNamespace(costs=lambda ratio: {"resident_kv_tokens": 1})
    monkeypatch.setattr(event_native_step, "memory_to_dict", lambda value: {})
    monkeypatch.setattr(budget_guard, "history_budget_receipt",
                        lambda *args, **kwargs: {"status": "passed", "errors": []})
    monkeypatch.setattr(event_native_step, "decode_native_generation",
                        lambda *args, **kwargs: NativeDraft("ok", "ok", (), "text", ""))

    class Controller:
        def prepare(self, payload, **kwargs):
            return SimpleNamespace(memory=memory, metadata={"decision_index": 0})

        def reconsider(self, prepared, calls, **kwargs):
            return {"regenerate": True, "memory": memory,
                    "metadata": prepared.metadata,
                    "decision": {"reason": "test_recovery"}}

    class Generator:
        def __init__(self):
            self.calls = 0
            self.closes = 0

        @contextmanager
        def decision_scope(self, *, session_id):
            yield

        def generate(self, memory, *, ratio, max_new_tokens, **kwargs):
            self.calls += 1
            return SimpleNamespace(token_ids=(1,), token_logprobs=(-0.1,),
                                   finish_reason="stop", stats={})

        def close_session(self):
            self.closes += 1

        def session_cache_info(self):
            return {"status": "released"}

    generator = Generator()
    runner = EventNativeDecisionRunner(
        Controller(), generator, object(), ratio=4, max_new_tokens=1,
        max_generation_calls=2, journal=AttemptJournal(tmp_path / "attempts.jsonl"),
    )
    api = _api(tmp_path, runner)
    server = make_server(api)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"

        def post(decision_key):
            body = json.dumps({"decision_key": decision_key}).encode("utf-8")
            request = Request(base + "/v1/chat/completions", data=body,
                              headers={"Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=5) as response:
                    return response.status, json.load(response)
            except HTTPError as error:
                return error.code, json.load(error)

        assert post("d0") == (200, {"status": "ok"})
        for _ in range(2):
            status, response = post("d1")
            assert status == 429
            assert response["error"]["code"] == "generation_cap_reached"
        assert generator.calls == runner.generation_calls == 2
        assert generator.closes >= 1
        health = api.health()
        assert health["decisions_reserved"] == 2 < health["max_decisions"]
        assert health["generation_calls_reserved"] == health["max_generation_calls"] == 2
        assert health["terminal_reason"] == "generation_cap_reached"
        assert event_native_server._stop_for_health(health) is False
        steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
        assert [row["status"] for row in steps] == ["ok", "failed"]
        assert [trace["phase"] for trace in steps[0]["generation_trace"]] == [
            "draft", "regeneration"]
        assert steps[1]["failure_kind"] == "budget_exhausted"
        assert steps[1]["failure_code"] == "generation_cap_reached"
        assert steps[1]["generation_attempts"] == 0
        attempts = [json.loads(line) for line in (tmp_path / "attempts.jsonl").read_text().splitlines()]
        assert len([row for row in attempts if row.get("status") == "started"]) == 2
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_generation_cap_requires_exception_type_and_durable_step(tmp_path, monkeypatch):
    def typed_failure(payload):
        try:
            raise GenerationCallCapExceeded("cap")
        except GenerationCallCapExceeded as cause:
            raise EventNativeStepError("cap", {"status": "failed"}) from cause

    api = _api(tmp_path, SimpleNamespace(run=typed_failure, close=lambda: None))
    monkeypatch.setattr(api, "_append_step", lambda record: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(EventNativeAPIError) as failed:
        api.handle_chat({"decision_key": "d0"})
    assert (failed.value.status_code, failed.value.code) == (500, "steps_write_failed")
    assert api.health()["terminal_reason"] == "steps_write_failed"
    assert event_native_server._stop_for_health(api.health()) is True

    def ordinary_failure(payload):
        raise RuntimeError("Finite generation-call cap exhausted before submission")

    ordinary = _api(tmp_path, SimpleNamespace(run=ordinary_failure, close=lambda: None))
    with pytest.raises(EventNativeAPIError) as failed:
        ordinary.handle_chat({"decision_key": "d0"})
    assert (failed.value.status_code, failed.value.code) == (500, "runner_failed")
    assert ordinary.health()["terminal_reason"] == "runner_failed"
    assert event_native_server._stop_for_health(ordinary.health()) is True
