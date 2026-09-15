"""Native evidence requests preserve the bounded source and execution contract."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from memory_runtime import native_source_needs as native
from memory_runtime.source_needs import build_needs_input
from memory_runtime.tests.test_source_needs import _store
from memory_runtime.tests.test_source_needs_runtime import _messages, _prepare


def _response(source="needs-test:m1", *, name=native.TOOL_NAME):
    return {"content": None, "tool_calls": [{"type": "function", "function": {
        "name": name, "arguments": json.dumps({"needs": [{"kind": "prior_result", "source_ids": [source]}]})}}]}


def test_native_tool_schema_is_counted_and_tracks_the_fitted_source_pool():
    context = build_needs_input(_store(), [])
    def count(messages, tools):
        fitted = json.loads(messages[-1]["content"])
        choices = tools[0]["function"]["parameters"]["properties"]["needs"]["items"]["properties"]["source_ids"]["items"]["enum"]
        assert choices == [entry["source_id"] for entry in fitted["index"]]
        assert "OLD-RAW-MARKER" not in json.dumps([messages, tools])
        return 100 + 50 * len(choices)
    fitted, receipt = native.fit_prediction_input(context, count, max_prompt_tokens=150)
    assert len(fitted["index"]) == 1 and len(context["index"]) == 2
    assert receipt["tool_schema_counted"] and receipt["dropped_source_ids"] == ["needs-test:m1"]
    assert native.parse_prediction(_response(), fitted).status == "invalid_prediction_abstain"
    fitted, receipt = native.fit_prediction_input(context, count, max_prompt_tokens=99)
    assert fitted is None and receipt["status"] == "current_input_exceeds_prompt_cap"


def test_only_the_declared_native_evidence_call_can_request_sources():
    context = build_needs_input(_store(), [])
    assert native.parse_prediction(_response(), context).source_ids == ("needs-test:m1",)
    invalid = [_response("future:m100"), _response(name="delete"),
        {"tool_calls": _response()["tool_calls"] * 2},
        {"tool_calls": [{"function": {"name": native.TOOL_NAME, "arguments": '{"needs":[]}'}}]}]
    for value in invalid:
        assert native.parse_prediction(value, context).status == "invalid_prediction_abstain"
    result = native.parse_prediction({"content": '{"needs":[{"kind":"prior_result","source_ids":["needs-test:m1"]}]}'}, context)
    assert result.status == "no_source_requested" and result.source_ids == ()


@pytest.mark.parametrize("representation", ["ac", "raw"])
def test_native_request_restores_the_same_source_as_the_json_interface(monkeypatch, representation):
    def request(messages, tokens, *, tools):
        assert tools[0]["function"]["name"] == native.TOOL_NAME
        return _response("needs-task:m1")
    out, counts = _prepare(monkeypatch, representation + "_native_needs_tool", _messages(), prediction=request)
    selected = counts["memory_runtime"]["source_needs"]
    assert selected["version"] == native.TOOL_NEEDS_VERSION
    assert selected["admitted_event_ids"] == ["needs-task:m1"]
    assert "OLD-FILE-LIST" in json.dumps(out)
    assert native.TOOL_NAME not in json.dumps(out)
    none, none_counts = _prepare(monkeypatch, representation + "_native_needs_none", _messages())
    empty, empty_counts = _prepare(monkeypatch, representation + "_native_needs_tool", _messages(),
        prediction=lambda *args, **kwargs: {"content": "No additional history is needed.", "tool_calls": None})
    assert empty == none
    assert empty_counts["memory_runtime"]["active_history_bytes"] == none_counts["memory_runtime"]["active_history_bytes"]
