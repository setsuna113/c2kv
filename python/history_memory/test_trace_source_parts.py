"""Regressions for the OTel records stored in agent-llm-traces."""

import json
from collections import Counter

import pytest

from history_memory.sources import SourceRowError, adapt_agent_trace_row


def trace_row(*, replay_text="(no content)", result_id="call-1"):
    call = {"type": "tool_call", "id": "call-1", "name": "lookup", "arguments": {}}
    user = {"role": "user", "parts": [{"type": "text", "content": "Find it"}]}
    output = {"role": "assistant", "parts": [{"type": "text", "content": "\n\n"}, call]}
    replay = {"role": "assistant", "parts": [{"type": "text", "content": replay_text}, call]}
    response = {"role": "user", "parts": [
        {"type": "tool_call_response", "id": result_id, "result": [{"answer": 42}]},
        {"type": "text", "content": "Explain the result"},
    ]}
    return {"session_id": "trace", "benchmark": "appworld", "spans": [
        {"span_id": "one", "start_time": "1", "attributes": {
            "gen_ai.tool.definitions": [{"name": "lookup", "parameters": {}}],
            "gen_ai.input.messages": [user], "gen_ai.output.messages": [output],
        }},
        {"span_id": "two", "start_time": "2", "attributes": {
            "gen_ai.input.messages": [user, replay, response],
            "gen_ai.output.messages": [{"role": "assistant", "content": "It is 42"}],
        }},
    ]}


def test_otel_tool_response_preserves_result_id_text_and_coordinates():
    audit = Counter()
    result = adapt_agent_trace_row(trace_row(), audit=audit)
    messages = result["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "user", "assistant"]
    assert messages[2]["tool_call_id"] == "call-1"
    assert json.loads(messages[2]["content"]) == [{"answer": 42}]
    assert messages[3]["content"] == "Explain the result"
    assert result["source_metadata"]["normalized_message_sources"][2] == {
        "span_id": "two", "field": "gen_ai.input.messages", "index": 2,
        "part_index": 0, "source_role": "user",
    }
    assert audit["trace_tool_response_parts_promoted"] == 1


def test_otel_tool_response_does_not_bind_wrong_explicit_id():
    with pytest.raises(SourceRowError, match="unmatched_tool_result"):
        adapt_agent_trace_row(trace_row(result_id="wrong"))


def test_otel_replay_still_rejects_substantively_changed_assistant_text():
    with pytest.raises(SourceRowError, match="non_monotone_trace_prefix"):
        adapt_agent_trace_row(trace_row(replay_text="A different answer"))
