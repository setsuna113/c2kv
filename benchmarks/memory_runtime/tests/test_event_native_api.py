"""Finite HTTP and validation tests for the event-native OpenAI transport."""
from __future__ import annotations

import copy
import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from benchmarks.memory_runtime.event_native_api import (
    EventNativeAPI,
    EventNativeAPIError,
    make_server,
)
from benchmarks.memory_runtime.event_native_step import EventNativeStepError


def _tool_call(value: str) -> dict:
    return {
        "id": "d1_r1_0",
        "type": "function",
        "function": {"name": "lookup", "arguments": json.dumps({"id": value})},
    }


def _success_record(*, tool_calls=None, content=None) -> dict:
    calls = list(tool_calls or [])
    return {
        "schema": "a-event-native-exact-step-v1",
        "status": "ok",
        "session_id": "internal",
        "decision_key": "internal",
        "generation_trace": [
            {
                "phase": "draft",
                "discarded": True,
                "native_draft": {"text": "PRIVATE-DRAFT"},
                "generation": {"stats": {"extracted_chunks": 2}},
            },
            {
                "phase": "regeneration",
                "discarded": False,
                "native_draft": {"text": "FINAL-DRAFT"},
                "generation": {"stats": {"scope_reused_chunks": 1}},
            },
        ],
        "response": {
            "role": "assistant",
            "content": content,
            "tool_calls": calls,
            "reasoning_content": "final-reasoning",
            "finish_reason": "stop",
        },
        "generation_attempts": 2,
        "generation_usage_total": {
            "prompt_tokens": 30,
            "completion_tokens": 7,
            "total_tokens": 37,
        },
    }


def _failed_record() -> dict:
    return {
        "schema": "a-event-native-exact-step-v1",
        "status": "failed",
        "session_id": "internal",
        "decision_key": "internal",
        "generation_trace": [
            {
                "phase": "draft",
                "status": "completed",
                "discarded": True,
                "native_draft": {"text": "PRIVATE-DRAFT"},
            },
            {"phase": "regeneration", "status": "failed", "usage": None},
        ],
        "response": None,
        "generation_usage_total": {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
        "generation_usage_known": {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13,
        },
        "error": {"type": "RuntimeError", "message": "synthetic failure"},
    }


class FakeRunner:
    max_generation_calls = 20

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.generation_calls = 0
        self.closed = 0

    def close(self):
        self.closed += 1

    def run(self, payload):
        self.calls.append(copy.deepcopy(payload))
        outcome = self.outcomes.pop(0)
        record = outcome.record if isinstance(outcome, EventNativeStepError) else outcome
        self.generation_calls += len(record.get("generation_trace", ()))
        if isinstance(outcome, Exception):
            raise outcome
        return copy.deepcopy(outcome)


def _payload(*, task_id="multi_turn_base_7", user_turn=1, step=2, messages=None):
    return {
        "messages": messages or [{"role": "user", "content": "Use the tool."}],
        "model": "tiny-event-native",
        "temperature": 0,
        "store": False,
        "max_completion_tokens": 32,
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up one item.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            },
        }],
        "seed": 0,
        "c2kv_eval_context": {
            "benchmark": "bfcl",
            "task_id": task_id,
            "user_turn": user_turn,
            "step": step,
            "attempt": 0,
        },
    }


def _api(
    tmp_path,
    runner,
    *,
    max_decisions=4,
    deadline=None,
    run_id="run-9",
    view_mode="capacity_exact_persistent",
):
    return EventNativeAPI(
        runner,
        run_id=run_id,
        model_name="tiny-event-native",
        view_mode=view_mode,
        max_new_tokens=32,
        allowed_task_ids={"multi_turn_base_7", "multi_turn_base_8"},
        max_decisions=max_decisions,
        deadline_monotonic=time.monotonic() + 60 if deadline is None else deadline,
        steps_path=tmp_path / "steps.jsonl",
    )


def _http_json(base_url: str, path: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def _read_steps(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("benchmark", ["bfcl", "acebench"])
def test_frozen_benchmark_namespaces_requests_before_runner_or_dedup(tmp_path, benchmark):
    runner = FakeRunner([_success_record(content="Final text.")])
    api = EventNativeAPI(
        runner, run_id="namespace-check", model_name="tiny-event-native",
        benchmark=benchmark, view_mode="capacity_exact_once", max_new_tokens=32,
        allowed_task_ids=["shared-task"], max_decisions=2,
        deadline_monotonic=time.monotonic() + 60, steps_path=tmp_path / "steps.jsonl",
    )
    payload = _payload(task_id="shared-task", user_turn=0, step=0)
    payload["c2kv_eval_context"]["benchmark"] = benchmark
    response = api.handle_chat(payload)
    assert api.handle_chat(copy.deepcopy(payload)) == response
    assert len(runner.calls) == 1
    assert runner.calls[0]["session_id"] == f"{benchmark}/shared-task/attempt-0"
    assert runner.calls[0]["decision_key"] == "turn-0/step-0"
    assert runner.calls[0]["messages"] == payload["messages"]
    assert api.health()["benchmark"] == benchmark

    other = copy.deepcopy(payload)
    other["c2kv_eval_context"]["benchmark"] = "acebench" if benchmark == "bfcl" else "bfcl"
    with pytest.raises(EventNativeAPIError) as error:
        api.handle_chat(other)
    assert error.value.code == "invalid_benchmark"
    assert len(runner.calls) == 1 and api.decisions_reserved == 1


@pytest.mark.parametrize("benchmark", [None, "unknown", ["acebench"]])
def test_constructor_rejects_unconfigured_benchmark(tmp_path, benchmark):
    with pytest.raises(ValueError, match="benchmark"):
        EventNativeAPI(
            FakeRunner([]), run_id="namespace-check", model_name="tiny-event-native",
            benchmark=benchmark, view_mode="capacity_exact_once", max_new_tokens=32,
            allowed_task_ids=["shared-task"], max_decisions=2,
            deadline_monotonic=time.monotonic() + 60, steps_path=tmp_path / "steps.jsonl",
        )


def test_http_maps_context_returns_only_final_response_and_deduplicates(tmp_path):
    record = _success_record(tool_calls=[_tool_call("final")])
    runner = FakeRunner([record])
    api = _api(tmp_path, runner)
    server = make_server(api)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    payload = _payload()
    try:
        status, health = _http_json(base_url, "/health")
        assert status == 200
        assert health == {
            "schema": "a-event-native-api-health-v1",
            "status": "ok",
            "terminal": False,
            "terminal_reason": None,
            "accepting_new_decisions": True,
            "run_id": "run-9",
            "model_name": "tiny-event-native",
            "benchmark": "bfcl",
            "view_mode": "capacity_exact_persistent",
            "route_contract": {
                "view_mode": "capacity_exact_persistent", "baseline_identity": "C2KV-persistent",
                "recovery_enabled": True, "max_generations_per_decision": 2,
                "legacy_1088_equivalent": False,
            },
            "runtime_policy_contract": None,
            "decode_strategy": None,
            "session_cache_policy": None,
            "max_new_tokens": 32,
            "allowed_task_ids": ["multi_turn_base_7", "multi_turn_base_8"],
            "decisions_reserved": 0,
            "max_decisions": 4,
            "generation_calls_reserved": 0,
            "max_generation_calls": 20,
            "deadline_exceeded": False,
        }

        status, response = _http_json(base_url, "/v1/chat/completions", payload)
        assert status == 200
        assert response["object"] == "chat.completion"
        assert response["model"] == "tiny-event-native"
        assert response["choices"] == [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("final")],
                "reasoning_content": "final-reasoning",
            },
            "finish_reason": "tool_calls",
        }]
        assert response["usage"] == {
            "prompt_tokens": 30,
            "completion_tokens": 7,
            "total_tokens": 37,
        }
        assert "PRIVATE-DRAFT" not in json.dumps(response)
        assert runner.calls == [{
            "session_id": "bfcl/multi_turn_base_7/attempt-0",
            "decision_key": "turn-1/step-2",
            "messages": payload["messages"],
            "tools": payload["tools"],
        }]
        assert _read_steps(tmp_path / "steps.jsonl") == [record]

        duplicate_status, duplicate = _http_json(
            base_url, "/v1/chat/completions", copy.deepcopy(payload)
        )
        assert duplicate_status == 200 and duplicate == response
        assert len(runner.calls) == 1
        assert _read_steps(tmp_path / "steps.jsonl") == [record]

        changed = copy.deepcopy(payload)
        changed["messages"][0]["content"] = "Changed input."
        with pytest.raises(HTTPError) as captured:
            _http_json(base_url, "/v1/chat/completions", changed)
        assert captured.value.code == 409
        assert len(runner.calls) == 1
        assert _read_steps(tmp_path / "steps.jsonl") == [record]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_session_identity_is_stable_across_routes_while_api_state_stays_isolated(
    tmp_path,
) -> None:
    first_runner = FakeRunner(
        [_success_record(content="first"), _success_record(content="other task")]
    )
    second_runner = FakeRunner([_success_record(content="second")])
    first = _api(
        tmp_path / "first",
        first_runner,
        run_id="run-a",
        view_mode="capacity_exact_once",
    )
    second = _api(
        tmp_path / "second",
        second_runner,
        run_id="run-b",
        view_mode="capacity_protect",
    )
    payload = _payload()

    first_response = first.handle_chat(payload)
    second_response = second.handle_chat(copy.deepcopy(payload))
    assert first_runner.calls[0]["session_id"] == second_runner.calls[0][
        "session_id"
    ] == "bfcl/multi_turn_base_7/attempt-0"
    assert first.health()["run_id"] == "run-a"
    assert second.health()["run_id"] == "run-b"
    assert first.health()["view_mode"] == "capacity_exact_once"
    assert second.health()["view_mode"] == "capacity_protect"
    assert first.decisions_reserved == second.decisions_reserved == 1

    # Each API owns its own completion cache.  A duplicate is local to the
    # first server and does not consume either runner a second time.
    assert first.handle_chat(copy.deepcopy(payload)) == first_response
    assert len(first_runner.calls) == len(second_runner.calls) == 1
    assert second_response["choices"][0]["message"]["content"] == "second"

    first.handle_chat(_payload(task_id="multi_turn_base_8", step=0))
    assert first_runner.calls[1]["session_id"] == "bfcl/multi_turn_base_8/attempt-0"
    assert first_runner.calls[1]["session_id"] != first_runner.calls[0]["session_id"]
    assert len(second_runner.calls) == 1


def test_plain_text_uses_null_tool_calls_for_official_consumer_fallback(tmp_path):
    runner = FakeRunner([
        _success_record(tool_calls=[], content="plain final"),
        _success_record(tool_calls=[], content="next final"),
    ])
    api = _api(tmp_path, runner)
    response = api.handle_chat(_payload())
    message = response["choices"][0]["message"]
    assert message["tool_calls"] is None
    assert message["content"] == "plain final"

    # The pinned BFCL consumer iterates tool_calls in a try block and falls
    # back to content on TypeError. An empty list would incorrectly yield [].
    try:
        consumed = [call["function"]["name"] for call in message["tool_calls"]]
    except TypeError:
        consumed = message["content"]
    assert consumed == "plain final"

    # The pinned handler can preserve the returned reasoning_content on the
    # next assistant prefix. It is accepted as observed data; packing owns the
    # choice to omit it from model-visible fields.
    continued = _payload(
        step=3,
        messages=[
            {"role": "user", "content": "first"},
            message,
            {"role": "user", "content": "continue"},
        ],
    )
    next_response = api.handle_chat(continued)
    assert next_response["choices"][0]["message"]["content"] == "next final"
    assert runner.calls[1]["messages"][1]["reasoning_content"] == "final-reasoning"


def test_cap_and_deadline_reject_before_runner_or_step_write(tmp_path):
    runner = FakeRunner([_success_record(content="done")])
    api = _api(tmp_path, runner, max_decisions=1)
    api.handle_chat(_payload())
    with pytest.raises(EventNativeAPIError) as cap_error:
        api.handle_chat(_payload(task_id="multi_turn_base_8", step=0))
    assert (cap_error.value.status_code, cap_error.value.code) == (
        429,
        "decision_cap_reached",
    )
    assert api.health()["terminal"] is True
    assert api.health()["decisions_reserved"] == 1
    assert len(runner.calls) == 1
    assert len(_read_steps(tmp_path / "steps.jsonl")) == 1

    expired_runner = FakeRunner([])
    expired = _api(
        tmp_path / "expired",
        expired_runner,
        deadline=time.monotonic() - 1,
    )
    with pytest.raises(EventNativeAPIError) as deadline_error:
        expired.handle_chat(_payload())
    assert (deadline_error.value.status_code, deadline_error.value.code) == (
        408,
        "deadline_exceeded",
    )
    assert expired.health()["terminal"] is True
    assert expired.health()["decisions_reserved"] == 0
    assert expired_runner.calls == []
    assert not (tmp_path / "expired" / "steps.jsonl").exists()


@pytest.mark.parametrize(
    "mutate, expected_code",
    [
        (lambda value: value.update(stream=False), "stream_unsupported"),
        (lambda value: value.update(c2kv_oracle={"target": "gold"}), "privileged_field"),
        (lambda value: value.update(temperature=0.1), "non_greedy"),
        (lambda value: value.update(max_completion_tokens=32.0), "max_tokens_mismatch"),
        (
            lambda value: value["c2kv_eval_context"].update(attempt=1),
            "attempt_forbidden",
        ),
        (
            lambda value: value["c2kv_eval_context"].update(task_id="unknown"),
            "unknown_task",
        ),
        (
            lambda value: value["messages"][0].update(gold="secret"),
            "unknown_message_field",
        ),
    ],
)
def test_malformed_or_privileged_requests_never_call_runner(
    tmp_path, mutate, expected_code
):
    runner = FakeRunner([])
    api = _api(tmp_path, runner)
    payload = _payload()
    mutate(payload)
    with pytest.raises(EventNativeAPIError) as captured:
        api.handle_chat(payload)
    assert captured.value.code == expected_code
    assert api.decisions_reserved == 0
    assert runner.calls == []
    assert not (tmp_path / "steps.jsonl").exists()


def test_runner_failure_fsyncs_full_trace_then_makes_api_terminal(tmp_path):
    record = _failed_record()
    failure = EventNativeStepError("synthetic failure", record)
    runner = FakeRunner([failure])
    api = _api(tmp_path, runner)

    with pytest.raises(EventNativeAPIError) as captured:
        api.handle_chat(_payload())
    assert (captured.value.status_code, captured.value.code) == (500, "runner_failed")
    assert _read_steps(tmp_path / "steps.jsonl") == [record]
    assert api.health()["terminal"] is True
    assert api.health()["terminal_reason"] == "runner_failed"
    assert api.health()["decisions_reserved"] == 1
    assert api.health()["generation_calls_reserved"] == 2
    assert runner.closed == 1

    with pytest.raises(EventNativeAPIError) as terminal:
        api.handle_chat(_payload())
    assert (terminal.value.status_code, terminal.value.code) == (503, "terminal_failure")
    assert len(runner.calls) == 1
    assert _read_steps(tmp_path / "steps.jsonl") == [record]


def test_step_write_failure_releases_a_successfully_generated_session(tmp_path, monkeypatch):
    runner = FakeRunner([_success_record(content="Done.")])
    api = _api(tmp_path, runner)
    def fail_write(record):
        raise OSError("synthetic step write failure")
    monkeypatch.setattr(api, "_append_step", fail_write)
    with pytest.raises(EventNativeAPIError) as captured:
        api.handle_chat(_payload())
    assert captured.value.code == "steps_write_failed"
    assert runner.closed == 1
    assert api.health()["terminal"] is True


def test_make_server_rejects_non_loopback_binding(tmp_path):
    api = _api(tmp_path, FakeRunner([]))
    with pytest.raises(ValueError, match="loopback"):
        make_server(api, host="0.0.0.0")
