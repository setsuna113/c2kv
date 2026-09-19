import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import benchmarks
from benchmarks import acebench_cli as hook


RUNTIME = Path(__file__).resolve().parents[1] / "experiments" / "history_system" / "runtime"
if str(RUNTIME / "python") not in sys.path:
    sys.path.insert(0, str(RUNTIME / "python"))
if str(RUNTIME / "benchmarks") not in benchmarks.__path__:
    benchmarks.__path__.append(str(RUNTIME / "benchmarks"))

from benchmarks.memory_runtime.acebench_source import build_ace_event_store  # noqa: E402
from benchmarks.memory_runtime.acebench_runtime import AceEventNativeAPI  # noqa: E402
from benchmarks.memory_runtime.event_native_api import EventNativeAPI, EventNativeAPIError  # noqa: E402


def test_native_receipt_is_from_executed_decoder_and_original_observation(tmp_path, monkeypatch):
    monkeypatch.setenv("C2KV_ACE_NATIVE", "1")
    monkeypatch.setenv("C2KV_ACEBENCH_TELEMETRY", str(tmp_path / "harness.jsonl"))
    monkeypatch.setenv("ACEBENCH_AGENT_BASE_URL", "http://agent/v1")
    calls = []
    decoding = hook.decode_wrapper(lambda text: ["Wifi()"])
    executing = hook.executor_wrapper(lambda decoded: (["{}"], {}))

    @hook.execution_wrapper
    def official_execute(_executor, history):
        decoded = decoding(history[-1]["message"])
        results, _ = executing(decoded)
        return {"sender": "execution", "recipient": "agent",
                "message": [json.loads(item) for item in results]}, {}

    def create(_resource, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(c2kv_proxy={"request_id": f"decision-{len(calls)}"})

    request = hook.request_wrapper(create)
    resource = SimpleNamespace(_client=SimpleNamespace(base_url="http://agent/v1"))
    initial = [{"role": "system", "content": "system"},
               {"role": "user", "content": "wifi"}]
    history = [{"sender": "user", "message": "wifi"},
               {"sender": "agent", "message": "[Wifi()]"}]

    @hook.task_wrapper
    def episode(test_id):
        request(resource, messages=initial, model="c2kv_native_r4",
                temperature=0.001, top_p=1, max_tokens=1000)
        observed, _ = official_execute(object(), history)
        visible = [*initial, {"role": "assistant", "content": history[-1]["message"]},
                   {"role": "tool", "content": json.dumps(observed["message"]),
                    "tool_call_id": "acebench-execution-2"}]
        request(resource, messages=visible, model="c2kv_native_r4",
                temperature=0.001, top_p=1, max_tokens=1000)
        return visible

    visible = episode("agent_multi_turn_1")
    assert calls[0]["temperature"] == calls[1]["temperature"] == 0.001
    assert calls[0]["max_tokens"] == calls[1]["max_tokens"] == 1000
    assert "c2kv_measurement_session_id" not in calls[1]["extra_body"]
    source = calls[1]["extra_body"]["c2kv_ace_source"]
    assert source["receipts"] == [{
        "version": "acebench-execution-receipt-v1",
        "agent_history_index": 1, "execution_message_index": 3,
        "decode_status": "ok", "decoded_calls": ["Wifi()"],
        "executor_status": "returned", "executor_return_shape": "list",
        "executor_return_count": 1,
    }]
    assert calls[1]["extra_body"]["c2kv_eval_context"] == {
        "benchmark": "acebench", "task_id": "agent_multi_turn_1",
        "user_turn": 0, "step": 1, "attempt": 0,
    }
    store = build_ace_event_store("acebench/agent_multi_turn_1/attempt-0", visible, source)
    assert [(event.kind, event.complete) for event in store.events][-1] == ("tool_event", True)


def test_native_ace_api_preserves_source_sampling_in_step_receipt():
    api = AceEventNativeAPI.__new__(AceEventNativeAPI)
    api.max_new_tokens = 1000
    api.view_mode = "ac_gist_static"
    original = {"temperature": 0.001, "top_p": 1, "max_tokens": 1000,
                "messages": [{"role": "system", "content": "system"}],
                "c2kv_ace_source": {"version": "acebench-text-actions-v1", "receipts": []}}
    captured = {}

    def accept(_self, payload):
        captured.update(payload)
        api._append_step({"session_id": "acebench/task/attempt-0", "decision_key": "turn-0/step-0"})
        return {"accepted": True}

    with mock.patch.object(EventNativeAPI, "handle_chat", accept), \
         mock.patch.object(EventNativeAPI, "_append_step", lambda _self, row: captured.update(step=row)):
        assert api.handle_chat(original) == {"accepted": True}
    assert original["temperature"] == 0.001
    assert captured["temperature"] == 0
    assert captured["max_completion_tokens"] == 1000
    assert captured["step"]["source_sampling"] == {
        "temperature": 0.001, "top_p": 1, "max_tokens": 1000, "seed": None}
    with pytest.raises(EventNativeAPIError, match="temperature"):
        api.handle_chat({**original, "temperature": 0})


def test_existing_ace_finite_routes_keep_original_request_contract():
    api = AceEventNativeAPI.__new__(AceEventNativeAPI)
    api.view_mode = "capacity_protect"
    payload = {"temperature": 0, "max_completion_tokens": 1000}
    with mock.patch.object(EventNativeAPI, "handle_chat", return_value={"legacy": True}) as old:
        assert api.handle_chat(payload) == {"legacy": True}
    old.assert_called_once_with(payload)
