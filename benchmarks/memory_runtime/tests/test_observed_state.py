"""Observed-state source identity, bounded content, and completion semantics."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "python"))

from history_memory.events import EventStore
from memory_runtime.observed_state import build_observed_state, fit_observed_state


def add(messages, name, arguments, result, *, complete=True):
    call_id = f"call-{len(messages)}"
    messages.append({"role": "assistant", "tool_calls": [{"id": call_id,
        "function": {"name": name, "arguments": json.dumps(arguments)}}]})
    if complete:
        messages.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})


def state(messages, **kwargs):
    return build_observed_state(EventStore.from_messages("state-test", messages), **kwargs)


def test_repeated_error_keeps_the_prior_observed_directory_without_declaring_completion():
    messages = [{"role": "user", "content": "Copy the document."}]
    add(messages, "cd", {"folder": "ResearchDocs"}, {"current_working_directory": "ResearchDocs"})
    add(messages, "cd", {"folder": "ResearchDocs"}, {"error": "directory not found"})
    value = state(messages)
    row = value["calls"][0]
    assert row["latest_result"]["source_id"] == "state-test:m3"
    assert row["latest_result"]["result_source_index"] == 4
    assert row["last_non_error_result"]["source_id"] == "state-test:m1"
    assert row["last_non_error_result"]["fields"] == [{"path": ["current_working_directory"], "value": "ResearchDocs"}]
    assert row["observations"] == row["observations_in_goal"] == 2
    assert row["result_changed_since_previous_same_call"] is True
    assert value["goal_completion"] == "unknown"


def test_success_null_empty_and_assistant_prose_do_not_mark_goal_complete():
    messages = [{"role": "user", "content": "Complete both operations."}]
    for result in ({"success": True}, None, "success", {"error": None}):
        add(messages, "write", {}, result)
    messages.append({"role": "assistant", "content": "Everything is complete."})
    value = state(messages)
    assert value["goal_completion"] == "unknown"
    assert value["calls"][0]["latest_result"]["error_field_reported"] is False
    assert value["calls"][0]["latest_result"]["success_flag_reported"] is None
    assert "Everything is complete" not in json.dumps(value)


def test_exact_call_identity_and_goal_scopes_preserve_legitimate_repetition():
    messages = [{"role": "user", "content": "Read both cities."}]
    add(messages, "lookup", {"city": "A"}, {"zip": "11111"})
    add(messages, "lookup", {"city": "B"}, {"zip": "22222"})
    messages.append({"role": "user", "content": "Read A again."})
    add(messages, "lookup", {"city": "A"}, {"zip": "11111"})
    value = state(messages)
    assert len(value["calls"]) == 2
    row = value["calls"][0]
    assert row["observations"] == 2 and row["observations_in_goal"] == 1
    assert row["result_changed_since_previous_same_call"] is False
    assert row["goal_source_id"] == value["goal_source_id"] == "state-test:m5"
    assert value["calls"][1]["arguments"] == {"city": "B"}


def test_pending_parallel_event_contributes_no_unobserved_result():
    messages = [{"role": "user", "content": "Read."},
        {"role": "assistant", "tool_calls": [
            {"id": "a", "function": {"name": "lookup", "arguments": '{"city":"A"}'}},
            {"id": "b", "function": {"name": "lookup", "arguments": '{"city":"B"}'}}]},
        {"role": "tool", "tool_call_id": "b", "content": '{"zip":"22222"}'}]
    assert state(messages)["calls"] == []
    messages.append({"role": "tool", "tool_call_id": "a", "content": '{"zip":"11111"}'})
    rows = {row["arguments"]["city"]: row for row in state(messages)["calls"]}
    assert rows["A"]["latest_result"]["result_source_index"] == 3
    assert rows["B"]["latest_result"]["result_source_index"] == 2
    assert rows["A"]["latest_result"]["fields"][0]["value"] == "11111"


def test_large_arguments_are_omitted_whole_but_never_merge_distinct_calls():
    messages = [{"role": "user", "content": "Inspect."}]
    add(messages, "lookup", {"key": "A" * 300}, {"large": "X" * 300, "small": 7})
    add(messages, "lookup", {"key": "B" * 300}, {"small": 8})
    rows = state(messages)["calls"]
    assert len(rows) == 2 and all(row["arguments_omitted"] for row in rows)
    assert rows[1]["latest_result"]["fields"] == [{"path": ["small"], "value": 7}]
    assert rows[1]["latest_result"]["omitted_fields"] == 1
    assert "XXX" not in json.dumps(rows)


def test_nested_scalar_paths_and_empty_collections_are_exact():
    messages = [{"role": "user", "content": "Inspect grades."}]
    add(messages, "find", {}, {"scores": [100, 95], "missing": [], "meta": {}})
    row = state(messages, max_fields=3)["calls"][0]["latest_result"]
    assert row["fields"] == [{"path": ["scores", 0], "value": 100},
        {"path": ["scores", 1], "value": 95}, {"path": ["missing"], "value": []}]
    assert row["omitted_fields"] == 1


def test_budget_fit_drops_oldest_whole_calls_without_mutating_source_state():
    messages = [{"role": "user", "content": "Inspect."}]
    for number in range(4):
        add(messages, "lookup", {"key": number}, {"value": number})
    original = state(messages, max_calls=3)
    def count(messages, tools):
        value = json.loads(messages[0]["content"].split("\n", 1)[1])
        return 100 + 50 * len(value["calls"])
    fitted, receipt = fit_observed_state(original, count, max_prompt_tokens=150)
    assert fitted["calls"] == original["calls"][:1]
    assert fitted["omitted_distinct_calls"] == 3
    assert len(original["calls"]) == 3 and receipt["prompt_tokens"] == 150
    fitted, receipt = fit_observed_state(original, count, max_prompt_tokens=99)
    assert fitted is None and receipt["status"] == "state_header_exceeds_prompt_cap"


def test_invalid_budgets_are_rejected():
    with pytest.raises(ValueError):
        state([], max_calls=0)
    with pytest.raises(ValueError):
        fit_observed_state(state([]), lambda *_: -1, max_prompt_tokens=100)
