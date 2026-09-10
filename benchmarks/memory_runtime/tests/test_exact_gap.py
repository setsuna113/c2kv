import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
from memory_runtime.adapter import RuntimeAdapter  # Load the shared event interface.
from memory_runtime.exact_gap import detect_exact_source_gap, contains_exact_literal
from history_memory.events import EventStore


def draft(**arguments):
    return [{"id": "draft", "type": "function", "function": {
        "name": "unverified_tool", "arguments": json.dumps(arguments)}}]


def source(extra=()):
    return EventStore.from_messages("task", [
        {"role": "user", "content": "Find the record."},
        {"role": "assistant", "tool_calls": [{"id": "c", "type": "function", "function": {
            "name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": '{"id":"item-17","path":"folder/file.txt"}'},
        *extra,
        {"role": "user", "content": "Use the earlier record."},
    ])


def detect(store, calls, visible=None):
    cutoff = len(store.messages) - 1
    return detect_exact_source_gap(store, visible_source_indices={cutoff} if visible is None else visible,
                                   source_cutoff=cutoff, draft_tool_calls=calls)


def test_unique_complete_source_gap_does_not_judge_the_tool():
    store = source()
    result = detect(store, draft(id="item-17", path="folder/file.txt"))
    assert (result.status, result.event_id) == ("gap", "task:m1")
    assert result.metadata()["judges_action_correctness"] is False
    assert "item-17" not in json.dumps(result.metadata())
    assert detect(store, draft(id="item-17"), visible={2, 3}).status == "no_op"


def test_visible_plain_text_and_partial_event_observation_are_exact_provenance():
    store = source()
    store = EventStore.from_messages("task", [*[m.to_dict() for m in store.messages[:-1]],
                                              {"role": "user", "content": 'Use item-17.'}])
    assert detect(store, draft(id="item-17")).status == "no_op"
    assert detect(source(), draft(id="item-17"), visible={2}).status == "no_op"


def test_no_source_ambiguous_sources_and_distinct_sources_abstain():
    assert detect(source(), draft(id="item-18")).reason == "missing_source"
    repeated = source([{"role": "assistant", "content": 'Earlier value: "item-17"'}])
    assert detect(repeated, draft(id="item-17")).reason == "ambiguous_sources"
    separate = source([{"role": "assistant", "content": 'Other binding: "node-22"'}])
    assert detect(separate, draft(id="item-17", node="node-22")).reason == "multiple_source_events"


def test_incomplete_event_and_metadata_keys_are_not_sources():
    store = EventStore.from_messages("task", [
        {"role": "assistant", "tool_calls": [{"id": "item-17", "type": "function", "function": {
            "name": "lookup", "arguments": '{"path":"folder/file.txt"}'}}]},
        {"role": "user", "content": "Continue."},
    ])
    assert detect(store, draft(path="folder/file.txt")).reason == "missing_source"
    assert detect(store, draft(id="item-17")).reason == "missing_source"
    assert detect(source(), draft(key="id")).reason == "missing_source"


@pytest.mark.parametrize("arguments", ['{"id":', '{"id":"item-17","id":"other"}',
                                         '{"id":NaN}', '[]'])
def test_malformed_arguments_abstain_without_partial_upgrade(arguments):
    calls = draft(id="item-17")
    calls[0]["function"]["arguments"] = arguments
    assert detect(source(), calls).reason == "malformed_arguments"


def test_non_native_text_and_numeric_arguments_cannot_trigger():
    assert detect(source(), None).reason == "no_native_tool_calls"
    assert detect(source(), draft(count=17, enabled=True, empty="")).reason == "no_string_bindings"


def test_literal_match_does_not_normalize_or_match_identifier_substrings():
    assert contains_exact_literal('Use "item-17".', "item-17")
    assert not contains_exact_literal("item-170", "item-17")
    assert not contains_exact_literal("prefix/item-17", "item-17")
    assert not contains_exact_literal("folder/file.txt.bak", "folder/file.txt")
    assert contains_exact_literal("Read folder/file.txt.", "folder/file.txt")
    assert not contains_exact_literal("ITEM-17", "item-17")
    assert not contains_exact_literal("two  words", "two words")


def test_visibility_cannot_reference_a_future_message():
    with pytest.raises(ValueError, match="outside"):
        detect(source(), draft(id="item-17"), visible={999})


@pytest.mark.parametrize("content", [
    json.dumps({"message": "'research_notes.txt' copied to 'archives/research_notes.txt'"}),
    json.dumps([{"message": "'research_notes.txt' copied to 'archives/research_notes.txt'"}]),
    json.dumps("'research_notes.txt' copied to 'archives/research_notes.txt'"),
    [{"type": "text", "text": json.dumps({"message": "Copied to 'archives/research_notes.txt'."})}],
])
def test_literal_in_structured_observation_prose_is_visible_or_retrievable(content):
    messages = [message.to_dict() for message in source().messages]
    messages[2]["content"] = content
    store = EventStore.from_messages("task", messages)
    calls = draft(source="archives/research_notes.txt")
    visible = detect(store, calls, visible={2, 3})
    assert visible.status == "no_op"
    assert visible.bindings[0].visible is True
    assert visible.metadata()["version"] == "exact-source-gap-v2"
    hidden = detect(store, calls)
    assert (hidden.status, hidden.event_id) == ("gap", "task:m1")


@pytest.mark.parametrize("content", [
    {"message": "Copied to prefix/item-17."},
    {"message": "Copied to item-170."},
    {"message": "Copied to ITEM-17."},
    {"item-17": "Only the key contains the requested literal."},
])
def test_structured_observation_keeps_boundaries_case_and_key_exclusion(content):
    messages = [message.to_dict() for message in source().messages]
    messages[2]["content"] = json.dumps(content)
    assert detect(EventStore.from_messages("task", messages), draft(id="item-17")).reason == "missing_source"


def test_structured_content_fix_does_not_search_inside_tool_argument_values():
    messages = [message.to_dict() for message in source().messages]
    messages[1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"note": "Copied item-17."})
    messages[2]["content"] = '{"message":"Done."}'
    assert detect(EventStore.from_messages("task", messages), draft(id="item-17")).reason == "missing_source"
