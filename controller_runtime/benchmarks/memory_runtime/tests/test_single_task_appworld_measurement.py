"""The AppWorld measurement hook may annotate a request, but cannot own task identity."""

import json
import time
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.event_native_api import EventNativeAPI, EventNativeAPIError
from benchmarks.memory_runtime.single_task_harness_api import SingleTaskHarnessAPI


TASK_ID = "3d9a636_1"


def _api(monkeypatch, benchmark="acon_appworld", captured=None):
    api = SingleTaskHarnessAPI.__new__(SingleTaskHarnessAPI)
    api.benchmark = benchmark
    api.run_id = "appworld-measurement-test"
    api.model_name = "gen_c2kv_K0_compression_full_budget"
    api.allowed_task_ids = frozenset({TASK_ID})
    api.max_new_tokens = 2048
    api.runner = SimpleNamespace(controller=SimpleNamespace(gp=object()), max_new_tokens=2048)
    api._wire_identities = {}
    api._wire_mode = None
    api._transport_receipts = {}
    def fake_handle_chat(self, payload):
        if captured is not None:
            captured["receipt"] = next(iter(self._transport_receipts.values())).copy()
        return payload
    monkeypatch.setattr(EventNativeAPI, "handle_chat", fake_handle_chat)
    return api


def _acon_request():
    # ACON vLLM.generate options plus c2kv_appworld_hook's extra_body.
    return {
        "model": "gen_c2kv_K0_compression_full_budget",
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
    captured = {}
    api = _api(monkeypatch, captured=captured)
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
    assert captured["receipt"]["normalized_server_fields"] == {
        "temperature": 0, "top_p": 1.0, "presence_penalty": 0.5,
        "seed": 42, "chat_template_kwargs": {"enable_thinking": False},
        "max_completion_tokens": 2048,
    }
    assert captured["receipt"]["normalized_away_fields"] == []


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


def _toolsandbox_request():
    request = _acon_request()
    request.pop("chat_template_kwargs")
    request["presence_penalty"] = 0
    request["seed"] = 0
    request["c2kv_measurement_session_id"] = "opaque-scenario-run-927"
    return request


def test_toolsandbox_opaque_measurement_session_preserves_frozen_task(monkeypatch):
    captured = {}
    api = _api(monkeypatch, benchmark="toolsandbox", captured=captured)
    normalized = api.handle_chat(_toolsandbox_request())
    assert normalized["c2kv_eval_context"]["benchmark"] == "toolsandbox"
    assert normalized["c2kv_eval_context"]["task_id"] == TASK_ID
    assert "c2kv_measurement_session_id" not in normalized
    assert captured["receipt"]["task_identity_source"] == "server"
    assert captured["receipt"]["measurement_task_id_validated"] is False
    assert captured["receipt"]["measurement_session_id_accepted"] is True


@pytest.mark.parametrize("measurement", [None, "", 7])
def test_toolsandbox_rejects_invalid_measurement_session(monkeypatch, measurement):
    api = _api(monkeypatch, benchmark="toolsandbox")
    request = _toolsandbox_request()
    request["c2kv_measurement_session_id"] = measurement
    with pytest.raises(EventNativeAPIError) as error:
        api.handle_chat(request)
    assert error.value.code == "invalid_measurement_session_id"
    assert api._wire_identities == {}


def test_toolsandbox_response_request_id_matches_native_step(monkeypatch):
    api = _api(monkeypatch, benchmark="toolsandbox")
    record = {
        "status": "ok", "outer_request_id": "c1-native-request-4",
        "response": {"role": "assistant", "content": "Done.",
                     "tool_calls": [], "finish_reason": "stop"},
        "generation_usage_total": {
            "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
        },
    }
    response = api._openai_response(record)
    assert response["id"] == record["outer_request_id"]
    assert response["c2kv_proxy"]["request_id"] == record["outer_request_id"]


def test_toolsandbox_real_adapter_injects_missing_runner_request_id(tmp_path):
    seen = {}
    def run(payload):
        seen["outer_request_id"] = payload["outer_request_id"]
        return {
            "status": "ok", "session_id": payload["session_id"],
            "decision_key": payload["decision_key"],
            "generation_trace": [],
            "response": {"role": "assistant", "content": "Done.",
                         "tool_calls": [], "finish_reason": "stop"},
            "generation_usage_total": {
                "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12,
            },
        }

    steps = tmp_path / "steps.jsonl"
    api = SingleTaskHarnessAPI(
        SimpleNamespace(run=run), benchmark="toolsandbox", run_id="ts-native-test",
        model_name="gen_c2kv_K0_compression_full_budget", view_mode="static",
        max_new_tokens=2048, allowed_task_ids=[TASK_ID], max_decisions=2,
        deadline_monotonic=time.monotonic() + 60, steps_path=steps,
    )
    response = api.handle_chat(_toolsandbox_request())
    journal = json.loads(steps.read_text(encoding="utf-8").splitlines()[0])
    request_id = seen["outer_request_id"]
    assert journal["outer_request_id"] == request_id
    assert response["id"] == request_id
    assert response["c2kv_proxy"]["request_id"] == request_id
    assert journal["harness_transport_normalization"]["task_identity_source"] == "server"
    assert journal["harness_transport_normalization"]["measurement_session_id_accepted"] is True
