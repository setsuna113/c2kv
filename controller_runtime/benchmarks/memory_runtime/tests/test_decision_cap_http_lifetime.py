"""A finite model decision budget must remain a typed task outcome."""

import time
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.event_native_api import EventNativeAPI, EventNativeAPIError
from benchmarks.memory_runtime.event_native_server import _stop_for_health


def test_decision_cap_keeps_http_endpoint_for_typed_rejection(tmp_path):
    assert _stop_for_health({"terminal": False, "terminal_reason": None}) is False
    assert _stop_for_health({"terminal": True,
                             "terminal_reason": "decision_cap_reached"}) is False
    assert _stop_for_health({"terminal": True,
                             "terminal_reason": "terminal_failure"}) is True
    api = EventNativeAPI(
        SimpleNamespace(run=lambda payload: {"status": "ok"}),
        run_id="test", model_name="model", view_mode="static", max_new_tokens=1,
        allowed_task_ids=["task"], max_decisions=1,
        deadline_monotonic=time.monotonic() + 60, steps_path=tmp_path / "steps.jsonl")
    api._validate_request = lambda payload: (payload, ("task", 0, payload["step"]),
                                             str(payload["step"]))
    api.decisions_reserved = 1
    assert _stop_for_health(api.health()) is False
    with pytest.raises(EventNativeAPIError) as raised:
        api.handle_chat({"step": 1})
    assert (raised.value.status_code, raised.value.code) == (429, "decision_cap_reached")
