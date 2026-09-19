from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import history_methods


def _dialect(message):
    calls = message.get("tool_calls") or []
    return ", ".join(str(call.get("function", {}).get("name", "tool"))
                   for call in calls)


def _compressor(calls):
    def compress(payload):
        calls.append(copy.deepcopy(payload))
        return "MEMORY:" + payload["messages"][1]["content"].splitlines()[-1][:40]
    return compress


def _tool_turn():
    return [
        {"role": "user", "content": "Find the weather for Cambridge."},
        {"role": "assistant", "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "get_weather", "arguments": "{city: Cambridge}"},
        }], "content": None},
        {"role": "tool", "tool_call_id": "call-1", "content": "12 C"},
    ]


def test_length_summary_surrogate_only_folds_newly_closed_turns():
    state = history_methods.HistoryMethodState()
    calls = []
    first = [{"role": "user", "content": "start"}]
    out, stats = history_methods.transform(
        first, state, "length_summary_surrogate", _compressor(calls), _dialect)
    assert out == first
    assert stats["compressed_turns"] == 0

    transcript = first + _tool_turn()[1:] + [{"role": "user", "content": "continue"}]
    out, stats = history_methods.transform(
        transcript, state, "length_summary_surrogate", _compressor(calls), _dialect)
    assert len(calls) == 1
    assert stats["compressed_turns"] == 1
    assert out[-1] == transcript[-1]
    assert out[0]["content"].startswith("[length_summary_surrogate granular memory unit]")
    assert state.raw_archive[0][0]["content"] == "start"

    # Replaying the same growing transcript must not call the compressor again.
    _, repeated = history_methods.transform(
        transcript, state, "length_summary_surrogate", _compressor(calls), _dialect)
    assert len(calls) == 1
    assert repeated["compressed_turns"] == 0


def test_commit_summary_surrogate_waits_for_action_observation_commit():
    state = history_methods.HistoryMethodState()
    calls = []
    base = [{"role": "user", "content": "lookup"}]
    history_methods.transform(base, state, "commit_summary_surrogate", _compressor(calls), _dialect)

    # An assistant answer without a tool observation is retained raw.
    pending = base + [{"role": "assistant", "content": "I need more information."},
                      {"role": "user", "content": "try again"}]
    out, stats = history_methods.transform(
        pending, state, "commit_summary_surrogate", _compressor(calls), _dialect)
    assert not calls
    assert stats["compressed_turns"] == 0
    assert any(message.get("content") == "I need more information."
               for message in out)

    # A complete action-observation turn is compressed exactly once.
    committed = _tool_turn() + [{"role": "user", "content": "next"}]
    state = history_methods.HistoryMethodState()
    out, stats = history_methods.transform(
        committed, state, "commit_summary_surrogate", _compressor(calls), _dialect)
    assert len(calls) == 1
    assert stats["compressed_turns"] == 1
    assert out[0]["content"].startswith("[commit_summary_surrogate commit memory unit]")


def test_lexical_recovery_surrogate_has_one_explicit_lexical_recovery():
    state = history_methods.HistoryMethodState()
    calls = []
    first = _tool_turn() + [{"role": "user", "content": "What should I do next?"}]
    history_methods.transform(first, state, "lexical_recovery_surrogate", _compressor(calls), _dialect)

    query = first + [{"role": "assistant", "content": "done"},
                     {"role": "user", "content": "get_weather Cambridge"}]
    _, stats = history_methods.transform(
        query, state, "lexical_recovery_surrogate", _compressor(calls), _dialect)
    assert stats["recovery"]["attempted"] is True
    assert stats["recovery"]["count"] == 1

    query2 = query + [{"role": "assistant", "content": "done"},
                      {"role": "user", "content": "get_weather again"}]
    _, stats2 = history_methods.transform(
        query2, state, "lexical_recovery_surrogate", _compressor(calls), _dialect)
    assert stats2["recovery"]["attempted"] is False
    assert stats2["recovery"]["count"] == 1


def test_multi_turn_methods_require_qwen3_4b():
    with pytest.raises(ValueError, match="qwen3-4b"):
        history_methods.transform(
            [{"role": "user", "content": "hello"}],
            history_methods.HistoryMethodState(), "length_summary_surrogate", lambda _: "x", _dialect,
            model_family="qwen2-7b",
        )
