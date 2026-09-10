from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from history_memory.events import EventStore
from memory_runtime.raw_recency import build_raw_recency_view


DEFAULT_SYSTEM = {"role": "system", "content": "training system"}


def render(source):
    messages = [copy.deepcopy(message) for message in source]
    messages = [
        {"role": "user", "content": message["content"]}
        if message["role"] == "tool" else message
        for message in messages
    ]
    for message in messages:
        if message.get("tool_calls"):
            message["content"] = "Action:" + json.dumps(message.pop("tool_calls"))
    if not any(message["role"] == "system" for message in messages):
        messages.insert(0, copy.deepcopy(DEFAULT_SYSTEM))
    return messages, {"renderer": "training"}


def count(messages, tools):
    del tools
    return len(messages)


def complete_source():
    return [
        {"role": "user", "content": "old"},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
          "function": {"name": "read", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "17"},
        {"role": "user", "content": "use 17"},
        {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function",
          "function": {"name": "write", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": "done"},
    ]


def build(source, budget):
    return build_raw_recency_view(
        EventStore.from_messages("t", source), render, count, [], 1, budget)


def test_all_fit_is_exact_full_training_render_without_evidence_envelope():
    source = complete_source()
    messages, counts, metadata = build(source, 100)
    assert messages == render(source)[0]
    assert metadata["all_history_fits"] is True
    assert metadata["selected_source_indices"] == list(range(len(source)))
    assert metadata["skipped_event_ids"] == []
    assert metadata["evidence_bytes"] == 0
    assert counts["renderer"] == "training"
    assert not any("Historical evidence" in message.get("content", "")
                   for message in messages)


def test_all_fit_short_circuits_nonmonotonic_intermediate_view_costs():
    source = complete_source()

    def nonmonotonic_count(messages, tools):
        del tools
        return {2: 10, 3: 11, 4: 100, 5: 100, 6: 100, 7: 11}[len(messages)]

    messages, _, metadata = build_raw_recency_view(
        EventStore.from_messages("t", source), render, nonmonotonic_count, [], 1, 1)
    assert messages == render(source)[0]
    assert metadata["all_history_fits"] is True
    assert metadata["selected_source_indices"] == list(range(len(source)))
    assert metadata["skipped_event_ids"] == []


def test_cross_cutoff_tool_event_is_mandatory_and_older_events_skip_whole():
    source = complete_source()
    messages, counts, metadata = build(source, 1)
    assert metadata["mandatory_event_ids"] == ["t:m4"]
    assert metadata["selected_event_ids"] == ["t:m4"]
    assert metadata["protected_event_ids"] == ["t:m4"]
    assert metadata["selected_source_indices"] == [4, 5]
    assert metadata["skipped_event_ids"] == ["t:m0", "t:m1", "t:m3"]
    assert metadata["evicted_event_ids"] == metadata["skipped_event_ids"]
    assert messages == render(source[4:])[0]
    assert counts["current_start_out_index"] == 2
    assert counts["history_raw"] == 1
    assert counts["current_raw"] == 1


def test_incomplete_parallel_tool_event_stays_exact_and_reports_missing_call():
    source = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "a", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "b", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "a done"},
    ]
    messages, _, metadata = build(source, 1)
    assert metadata["raw_pending_event_ids"] == ["t:m1"]
    assert metadata["missing_tool_call_ids"] == {"t:m1": ["c2"]}
    assert metadata["selected_source_indices"] == [1, 2]
    assert "c2" in messages[1]["content"]
    assert all(message.get("content") != "b done" for message in messages)


def test_mandatory_history_fails_closed_when_it_exceeds_budget():
    with pytest.raises(ValueError, match="Mandatory complete/current events need"):
        build(complete_source(), 0)
