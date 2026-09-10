"""Bounded observable inputs and fail-closed source selection contracts."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "python"))

from history_memory.events import EventStore
from memory_runtime.source_needs import (
    build_needs_input, fit_prediction_input, lexical_source_ids,
    parse_source_request, prediction_messages,
)


def _tool(messages, name, result, *, complete=True):
    call_id = f"call-{len(messages)}"
    messages.append({"role": "assistant", "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": name, "arguments": '{"file_name":"report.txt"}'}}]})
    if complete:
        messages.append({"role": "tool", "tool_call_id": call_id,
                         "content": json.dumps(result)})


def _store():
    messages = [{"role": "user", "content": "Inspect the directory."}]
    _tool(messages, "ls", {"current_directory_content": ["OLD-RAW-MARKER", "report.txt"]})
    _tool(messages, "pwd", {"current_working_directory": "OLD-PATH-MARKER"})
    messages.append({"role": "user", "content": "Review both files."})
    _tool(messages, "cat", {"file_content": "CURRENT-RAW-MARKER"})
    return EventStore.from_messages("needs-test", messages)


def test_predictor_sees_metadata_but_no_older_values():
    store = _store()
    context = build_needs_input(store, [])
    rendered = json.dumps(prediction_messages(context))
    assert "OLD-RAW-MARKER" not in rendered and "OLD-PATH-MARKER" not in rendered
    assert "CURRENT-RAW-MARKER" in rendered
    assert "current_working_directory" in rendered
    assert context["index"][0]["calls"][0]["argument_anchors"] == {"file_name": "report.txt"}
    assert [row["source_id"] for row in context["index"]] == ["needs-test:m3", "needs-test:m1"]


def test_pending_event_is_never_an_older_candidate():
    messages = [m.to_dict() for m in _store().messages]
    _tool(messages, "lookup", None, complete=False)
    context = build_needs_input(EventStore.from_messages("needs-test", messages), [])
    assert all(row["source_id"] not in {"needs-test:m6", "needs-test:m8"}
               for row in context["index"])


def test_fit_drops_oldest_whole_entry_and_does_not_truncate_current_input():
    context = build_needs_input(_store(), [])
    def counter(messages, tools):
        body = json.loads(messages[-1]["content"])
        assert body["recent_tool_event"] == context["recent_tool_event"]
        return 100 + 50 * len(body["index"])
    fitted, receipt = fit_prediction_input(context, counter, max_prompt_tokens=150)
    assert [row["source_id"] for row in fitted["index"]] == ["needs-test:m3"]
    assert receipt["dropped_source_ids"] == ["needs-test:m1"]
    assert len(context["index"]) == 2
    fitted, receipt = fit_prediction_input(context, counter, max_prompt_tokens=99)
    assert fitted is None and receipt["status"] == "current_input_exceeds_prompt_cap"


def test_out_of_pool_or_excess_source_request_abstains_as_a_whole():
    context = build_needs_input(_store(), [])
    for content in (
        '{"needs":[{"kind":"prior_result","source_ids":["needs-test:m1","future:m99"]}]}',
        '{"needs":[{"kind":"action","source_ids":["needs-test:m1"]}]}',
        '{"needs":[],"tool_calls":[{"name":"delete"}]}',
        'The source is needs-test:m1.',
    ):
        result = parse_source_request(content, context)
        assert result.status == "invalid_prediction_abstain" and result.source_ids == ()


def test_validated_sources_and_lexical_control_stay_in_the_same_pool():
    store = _store()
    context = build_needs_input(store, [])
    result = parse_source_request(
        '{"needs":[{"kind":"location_state","source_ids":["needs-test:m3"]},'
        '{"kind":"prior_result","source_ids":["needs-test:m1"]}]}', context)
    assert result.source_ids == ("needs-test:m3", "needs-test:m1")
    assert set(lexical_source_ids(store, context)) <= set(result.source_ids)
    reduced = {**context, "index": context["index"][:1]}
    assert parse_source_request(
        '{"needs":[{"kind":"prior_result","source_ids":["needs-test:m1"]}]}',
        reduced).source_ids == ()
    assert set(lexical_source_ids(store, reduced)) <= {"needs-test:m3"}


def test_null_result_and_error_field_do_not_create_success_claims():
    messages = [{"role": "user", "content": "Continue."}]
    _tool(messages, "write", None)
    _tool(messages, "read", {"error": "private failure details"})
    _tool(messages, "pwd", {"current_working_directory": "."})
    context = build_needs_input(EventStore.from_messages("shape", messages), [])
    results = [entry["results"][0] for entry in context["index"]]
    assert results[0]["error_field_present"] is True
    assert results[1]["shape"] == "null"
    assert "private failure details" not in json.dumps(context)
    assert "success" not in json.dumps(context)


def test_entity_argument_anchors_are_exact_and_bounded():
    messages = [{"role": "user", "content": "Find distances."},
        {"role": "assistant", "tool_calls": [{"id": "old", "function": {
            "name": "lookup", "arguments": json.dumps({"city": "Rivermist", "large": "X" * 200})}}]},
        {"role": "tool", "tool_call_id": "old", "content": '{"zipcode":"RESULT-ONLY-MARKER"}'}]
    _tool(messages, "latest", None)
    context = build_needs_input(EventStore.from_messages("anchor", messages), [])
    assert context["index"][0]["calls"][0]["argument_anchors"] == {"city": "Rivermist"}
    assert "RESULT-ONLY-MARKER" not in json.dumps(context)
