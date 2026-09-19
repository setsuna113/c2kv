"""The AppWorld measurement hook may annotate a request, but cannot own task identity."""

from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.event_native_api import EventNativeAPI, EventNativeAPIError
from benchmarks.memory_runtime.single_task_harness_api import SingleTaskHarnessAPI


TASK_ID = "3d9a636_1"


def _api(monkeypatch, benchmark="acon_appworld"):
    api = SingleTaskHarnessAPI.__new__(SingleTaskHarnessAPI)
    api.benchmark = benchmark
    api.allowed_task_ids = frozenset({TASK_ID})
    api.max_new_tokens = 2048
    api.runner = SimpleNamespace(controller=SimpleNamespace(gp=object()), max_new_tokens=2048)
    api._wire_identities = {}
    api._wire_mode = None
    api._transport_receipts = {}
    monkeypatch.setattr(EventNativeAPI, "_validate_request", lambda self, payload: None)
    monkeypatch.setattr(EventNativeAPI, "handle_chat", lambda self, payload: payload)
    return api


def _acon_request():
    # ACON vLLM.generate options plus c2kv_appworld_hook's extra_body.
    return {
        "model": "c2kv-event-native",
        "messages": [{"role": "system", "content": "Use AppWorld APIs."},
                     {"role": "user", "content": "Complete the task."}],
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "seed": 42,
        "presence_penalty": 0.5,
        "chat_template_kwargs": {"enable_thinking": False},
        "c2kv_measurement_session_id": TASK_ID,
    }


def test_full_appworld_request_preserves_server_owned_identity(monkeypatch):
    api = _api(monkeypatch)
    normalized = api.handle_chat(_acon_request())
    assert normalized["c2kv_eval_context"] == {
        "benchmark": "acon_appworld",
        "task_id": TASK_ID,
        "user_turn": 0,
        "step": 0,
        "attempt": 0,
    }
    assert "c2kv_measurement_session_id" not in normalized
    assert normalized["temperature"] == 0
    assert normalized["seed"] == 0
    assert normalized["max_completion_tokens"] == 2048


@pytest.mark.parametrize(
    ("measurement", "expected_code"),
    [
        ("another_task", "task_identity_mismatch"),
        (None, "invalid_measurement_session_id"),
        (7, "invalid_measurement_session_id"),
        ("", "invalid_measurement_session_id"),
    ],
)
def test_measurement_metadata_cannot_change_task(monkeypatch, measurement, expected_code):
    api = _api(monkeypatch)
    request = _acon_request()
    request["c2kv_measurement_session_id"] = measurement
    with pytest.raises(EventNativeAPIError) as error:
        api.handle_chat(request)
    assert error.value.code == expected_code
    assert api._wire_identities == {}


def test_other_benchmark_rejects_appworld_measurement_metadata(monkeypatch):
    api = _api(monkeypatch, benchmark="tau2")
    request = _acon_request()
    request["presence_penalty"] = 0
    request.pop("chat_template_kwargs")
    request["seed"] = 0
    with pytest.raises(EventNativeAPIError) as error:
        api.handle_chat(request)
    assert error.value.code == "task_identity_mismatch"


def test_appworld_without_measurement_metadata_uses_server_identity(monkeypatch):
    api = _api(monkeypatch)
    request = _acon_request()
    request.pop("c2kv_measurement_session_id")
    assert api.handle_chat(request)["c2kv_eval_context"]["task_id"] == TASK_ID
