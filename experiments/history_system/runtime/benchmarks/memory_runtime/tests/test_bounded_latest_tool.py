"""Budgeted protection for the latest complete historical tool event."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from arms import get_arm
from history_memory.events import EventStore
from memory_runtime.source_needs_runtime import SourceNeedsRuntime
from memory_runtime.tests.test_native_workspace import _count, _setup
from memory_runtime.tests.test_source_needs_runtime import _config, _messages
import proxy


ROUTES = (
    "ac_native_needs_lexical_raw_reserve_failed_operation",
    "raw_native_needs_lexical_raw_reserve_failed_operation",
)


def _large_historical_tool_messages():
    return [
        {"role": "user", "content": "Find the archive."},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "lookup", "arguments": '{"query":"archive"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"location":"archive"}'},
        {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function",
            "function": {"name": "inspect", "arguments": '{"query":"work-marker"}'}}]},
        {"role": "tool", "tool_call_id": "c2",
         "content": json.dumps({"payload": "X" * 5000})},
        {"role": "assistant", "content": "I recorded work-marker."},
        {"role": "user", "content": "Use work-marker for the current answer."},
    ]


def _prepare(monkeypatch, route, messages, *, budget=1000, policy=None):
    _setup(monkeypatch, "ac_native_workspace")
    config = _config(route, budget)
    if policy is not None:
        config["latest_complete_tool_protection"] = policy
    runtime = SourceNeedsRuntime(config, _count)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "ARM", get_arm("full" if route.startswith("raw_") else "c2kv4"))
    return proxy._prepare_memory_input(messages, {
        "task_id": "needs-task", "attempt": 0, "user_turn": 1, "step": 1,
    }, [])


@pytest.mark.parametrize("route", ROUTES)
def test_large_latest_historical_tool_is_required_by_default_and_budgeted_by_policy(
        monkeypatch, route):
    source = _large_historical_tool_messages()
    original = copy.deepcopy(source)
    store = EventStore.from_messages("needs-task", source)
    latest = [event for event in store.events
              if event.kind == "tool_event" and event.complete][-1]

    with pytest.raises(proxy.MemoryRuntimeError) as failure:
        _prepare(monkeypatch, route, source)
    assert failure.value.kind == "capacity_infeasible"

    out, counts = _prepare(monkeypatch, route, source, policy="budgeted")
    meta = counts["memory_runtime"]
    receipt = meta["latest_complete_tool_protection"]
    assert source == original
    assert receipt["policy"] == "budgeted"
    assert receipt["status"] == "skipped" and receipt["event_id"] == latest.event_id
    assert receipt["candidate_required_native_bytes"] > receipt["budget_bytes"]
    assert receipt["reason"] == "protected_native_over_budget"
    assert receipt["selector_recomputed_after_skip"] is True
    assert receipt["indexed_after_skip"] is True
    assert latest.event_id not in meta["protected_event_ids"]
    assert latest.event_id in meta["source_needs"]["candidate_event_ids"]
    assert latest.event_id in meta["source_needs"]["requested_event_ids"]
    assert latest.event_id in {
        row["event_id"] for row in meta["source_needs"]["skipped_for_budget"]}
    assert not set(latest.source_indices) <= set(meta["selected_source_indices"])
    assert meta["active_history_bytes"] <= 1000
    assert bool(meta["gist_tokens"]) == route.startswith("ac_")
    assert any(message.get("content") == source[-1]["content"] for message in out)


@pytest.mark.parametrize("route", ROUTES)
def test_small_common_latest_tool_and_goal_are_identical_to_default(monkeypatch, route):
    source = _messages()
    default_out, default_counts = _prepare(monkeypatch, route, source, budget=20000)
    bounded_out, bounded_counts = _prepare(
        monkeypatch, route, source, budget=20000, policy="budgeted")
    assert default_out == bounded_out
    assert "latest_complete_tool_protection" not in default_counts["memory_runtime"]
    receipt = bounded_counts["memory_runtime"]["latest_complete_tool_protection"]
    assert receipt["status"] == "mandatory-common"
    assert receipt["reason"] == "latest_complete_tool_is_in_the_mandatory_common_suffix"
    assert any(message.get("content") == "Review both files." for message in bounded_out)
    assert any("CURRENT-RESULT" in str(message.get("content")) for message in bounded_out)


@pytest.mark.parametrize("route", ROUTES)
def test_small_optional_latest_tool_is_admitted(monkeypatch, route):
    source = _large_historical_tool_messages()
    source[4]["content"] = '{"payload":"work-marker"}'
    out, counts = _prepare(monkeypatch, route, source, budget=20000, policy="budgeted")
    meta = counts["memory_runtime"]
    receipt = meta["latest_complete_tool_protection"]
    assert receipt["status"] == "admitted"
    assert receipt["reason"] == "protected_native_and_required_representation_fit"
    assert receipt["event_id"] in meta["protected_event_ids"]
    assert receipt["event_id"] not in meta["source_needs"]["candidate_event_ids"]
    assert any("work-marker" in str(message.get("content")) for message in out)


def test_c2kv_representation_conflict_can_drop_the_optional_latest_tool(monkeypatch):
    source = _large_historical_tool_messages()
    source[4]["content"] = '{"payload":"work-marker"}'
    _, roomy_counts = _prepare(
        monkeypatch, ROUTES[0], source, budget=20000, policy="required")
    required_native_bytes = roomy_counts["memory_runtime"][
        "latest_complete_tool_protection"]["candidate_required_native_bytes"]

    with pytest.raises(proxy.MemoryRuntimeError) as failure:
        _prepare(monkeypatch, ROUTES[0], source, budget=required_native_bytes)
    assert failure.value.kind == "capacity_infeasible"

    _, bounded_counts = _prepare(
        monkeypatch, ROUTES[0], source, budget=required_native_bytes, policy="budgeted")
    meta = bounded_counts["memory_runtime"]
    receipt = meta["latest_complete_tool_protection"]
    assert receipt["status"] == "skipped"
    assert receipt["reason"] == "required_representation_over_budget"
    assert receipt["candidate_required_native_bytes"] <= required_native_bytes
    assert receipt["candidate_active_history_bytes"] > required_native_bytes
    assert meta["gist_tokens"] > 0 and meta["active_history_bytes"] <= required_native_bytes


@pytest.mark.parametrize("route", ROUTES)
def test_incomplete_historical_event_remains_mandatory(monkeypatch, route):
    source = [
        {"role": "user", "content": "Start the operation."},
        {"role": "assistant", "tool_calls": [{"id": "pending", "type": "function",
            "function": {"name": "pending_call", "arguments": json.dumps({
                "payload": "P" * 5000})}}]},
        {"role": "user", "content": "Continue the current goal."},
    ]
    with pytest.raises(proxy.MemoryRuntimeError) as failure:
        _prepare(monkeypatch, route, source, policy="budgeted")
    assert failure.value.kind == "capacity_infeasible"


@pytest.mark.parametrize("route", ROUTES)
def test_receipt_reports_no_complete_tool_event(monkeypatch, route):
    out, counts = _prepare(
        monkeypatch, route, [{"role": "user", "content": "Start."}],
        policy="budgeted")
    receipt = counts["memory_runtime"]["latest_complete_tool_protection"]
    assert receipt["status"] == "none" and receipt["event_id"] is None
    assert out[-1]["content"] == "Start."


def test_budgeted_policy_is_rejected_outside_the_two_dedicated_routes():
    config = _config("ac_native_needs_lexical")
    config["latest_complete_tool_protection"] = "budgeted"
    with pytest.raises(ValueError, match="dedicated raw-reserve failed-operation route"):
        SourceNeedsRuntime(config, _count)


def test_unknown_latest_complete_tool_policy_is_rejected():
    config = _config(ROUTES[0])
    config["latest_complete_tool_protection"] = "best-effort"
    with pytest.raises(ValueError, match="Unknown latest_complete_tool_protection"):
        SourceNeedsRuntime(config, _count)
