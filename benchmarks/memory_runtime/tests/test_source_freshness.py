"""Observable matching boundaries and actual native admission for freshness."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from history_memory.events import EventStore
from memory_runtime.source_freshness import refresh_source_ids
from memory_runtime.tests.test_source_needs_runtime import _prepare


def add(messages, name, arguments, result):
    index = len(messages)
    call_id = "c" + str(index)
    messages.extend([
        {"role": "assistant", "tool_calls": [{"id": call_id, "type": "function",
         "function": {"name": name, "arguments": json.dumps(arguments)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)},
    ])
    return "needs-task:m" + str(index)


def refresh(messages, ids, pool):
    return refresh_source_ids(EventStore.from_messages("needs-task", messages), ids, pool)


def test_full_arguments_and_latest_error_are_preserved_without_success_preference():
    messages = [{"role": "user", "content": "Read status."}]
    old = add(messages, "read", {"file": "a", "size": 1}, {"value": 1})
    distinct = add(messages, "read", {"file": "a", "size": 2}, {"value": 2})
    new = add(messages, "read", {"size": 1, "file": "a"}, {"error": "missing"})
    ids, receipt = refresh(messages, (old, distinct), (old, distinct, new))
    assert ids == (new, distinct)
    assert receipt["decisions"][0]["reason"] == "latest_current_goal_same_call"
    assert refresh(messages, (old, new), (old, new))[0] == (new,)
    assert refresh(messages, (old,), (old, distinct))[0] == (old,)


@pytest.mark.parametrize("barrier", ["cd", "message_login", "trading_logout"])
def test_context_changes_and_previous_goals_keep_original_sources(barrier):
    messages = [{"role": "user", "content": "Read status."}]
    old = add(messages, "read", {}, {"error": "missing"})
    add(messages, barrier, {}, {"success": True})
    new = add(messages, "read", {}, {"value": 1})
    ids, receipt = refresh(messages, (old,), (old, new))
    assert ids == (old,)
    assert receipt["decisions"][0]["reason"] == "context_barrier_between_calls"
    messages.append({"role": "user", "content": "Compare earlier observations."})
    assert refresh(messages, (old,), (old, new))[1]["decisions"][0]["reason"] == "outside_current_goal"


@pytest.mark.parametrize("representation", ["ac", "raw"])
def test_native_admission_charges_refreshed_result_and_preserves_default(monkeypatch, representation):
    messages = [{"role": "user", "content": "Read status of file a."}]
    old = add(messages, "read", {"file": "a"}, {"error": "a missing"})
    new = add(messages, "read", {"file": "a"}, {"value": "X" * 5000})
    add(messages, "pwd", {}, {"directory": "work"})
    # Fix source ranking to exercise admission independently of lexical scoring.
    monkeypatch.setattr("memory_runtime.source_needs_runtime.lexical_source_ids", lambda *_: (old,))
    _, base = _prepare(monkeypatch, representation + "_native_needs_lexical", messages, budget=1000)
    _, fresh = _prepare(monkeypatch, representation + "_native_needs_lexical_fresh", messages, budget=1000)
    base, fresh = base["memory_runtime"], fresh["memory_runtime"]
    assert base["source_needs"]["requested_event_ids"] == [old]
    assert "freshness" not in base["source_needs"]
    assert fresh["source_needs"]["requested_event_ids"] == [new]
    assert fresh["source_needs"]["skipped_for_budget"][0]["event_id"] == new
    assert fresh["source_needs"]["admitted_event_ids"] == []
    assert fresh["active_history_bytes"] <= 1000
    assert fresh["protected_event_ids"] == base["protected_event_ids"]
    assert bool(fresh["gist_tokens"]) == (representation == "ac")
