"""The native actor sees the same function schema as the Full SGLang actor."""

import copy
import time
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.event_native_api import EventNativeAPI, EventNativeAPIError


def _api(tmp_path):
    return EventNativeAPI(
        SimpleNamespace(run=lambda payload: None),
        run_id="test", model_name="actor", view_mode="static",
        max_new_tokens=16, allowed_task_ids=["1"], max_decisions=1,
        deadline_monotonic=time.monotonic() + 60,
        steps_path=tmp_path / "steps.jsonl", benchmark="tau2",
    )


def _request(tools):
    return {
        "model": "actor", "temperature": 0, "store": False,
        "max_completion_tokens": 16,
        "c2kv_eval_context": {
            "benchmark": "tau2", "task_id": "1", "user_turn": 0,
            "step": 0, "attempt": 0,
        },
        "messages": [
            {"role": "system", "content": "Policy"},
            {"role": "user", "content": "Look up my reservation"},
        ],
        "tools": tools,
    }


@pytest.mark.parametrize("strict", [None, False, True])
def test_native_tool_prologue_uses_full_sglang_schema_without_mutating_source(tmp_path, strict):
    function = {"name": "lookup", "description": "Look up a reservation",
                "parameters": {"type": "object"}, "response_schema": {"ignored": True}}
    if strict is not None:
        function["strict"] = strict
    request = _request([{"type": "function", "function": function, "extra": "ignored"}])
    source = copy.deepcopy(request)

    runner_payload, _, _ = _api(tmp_path)._validate_request(request)

    assert request == source
    assert runner_payload["messages"] == source["messages"]
    assert runner_payload["tools"] == [{
        "type": "function",
        "function": {
            "description": "Look up a reservation", "name": "lookup",
            "parameters": {"type": "object"},
            "strict": False if strict is None else strict,
        },
    }]


def test_native_tool_prologue_preserves_no_tools_and_rejects_invalid_schema(tmp_path):
    api = _api(tmp_path)
    assert api._validate_request(_request([]))[0]["tools"] == []
    with pytest.raises(EventNativeAPIError) as failed:
        api._validate_request(_request([{"type": "function"}]))
    assert (failed.value.status_code, failed.value.code) == (400, "invalid_tools")
