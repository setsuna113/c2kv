"""Shared-controller view contracts using the real legacy message renderer."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks"), str(ROOT / "python")]

from memory_runtime.source_needs_runtime import SourceNeedsRuntime
from memory_runtime.tests.test_native_workspace import _count, _setup
import proxy
from arms import get_arm


def _config(route, budget=20000):
    return {"mode": route, "run_id": "needs-runtime-test", "bytes_per_kv_token": 1,
            "history_budget_bytes": budget, "workspace_budget_bytes": budget,
            "lease_decisions": 0, "max_retrieved_events": 2,
            "compression_policy": "always-compress-v1", "history_view_protocol": "fixed-budget-main",
            "source_index_max_events": 12, "predictor_prompt_token_cap": 20000,
            "predictor_completion_token_cap": 256}


def _messages(old_value="OLD-FILE-LIST"):
    return [
        {"role": "user", "content": "Find the files."},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "ls", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"files": old_value})},
        {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {
            "name": "pwd", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c2", "content": '{"directory":"work"}'},
        {"role": "user", "content": "Review both files."},
        {"role": "assistant", "tool_calls": [{"id": "c3", "type": "function", "function": {
            "name": "cat", "arguments": '{"file_name":"report.txt"}'}}]},
        {"role": "tool", "tool_call_id": "c3", "content": '{"content":"CURRENT-RESULT"}'},
    ]


def _prepare(monkeypatch, route, messages, *, budget=20000, prediction=None, state_cap=2000):
    _setup(monkeypatch, "ac_native_workspace")
    config = _config(route, budget)
    if route.endswith("_state_none"):
        config["state_prompt_token_cap"] = state_cap
    runtime = SourceNeedsRuntime(config, _count)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "ARM", get_arm("full" if route.startswith("raw_") else "c2kv4"))
    return proxy._prepare_memory_input(messages, {"task_id": "needs-task", "attempt": 0,
        "user_turn": 1, "step": 1}, [], source_predictor=prediction)


def _request(messages, tokens):
    body = json.loads(messages[-1]["content"])
    assert "OLD-FILE-LIST" not in messages[-1]["content"]
    assert "CURRENT-RESULT" in messages[-1]["content"]
    assert "needs-task:m1" in {entry["source_id"] for entry in body["index"]}
    return {"content": '{"needs":[{"kind":"prior_result","source_ids":["needs-task:m1"]}]}',
            "tool_calls": None}


def test_shared_prediction_and_protection_with_real_native_source_restoration(monkeypatch):
    source = _messages()
    metadata = []
    for route in ("ac_native_needs_typed", "raw_native_needs_typed"):
        out, counts = _prepare(monkeypatch, route, source, prediction=_request)
        state = counts["memory_runtime"]
        assert state["source_needs"]["admitted_event_ids"] == ["needs-task:m1"]
        assert "OLD-FILE-LIST" in json.dumps(out)
        assert sum("CURRENT-RESULT" in str(message.get("content")) for message in out) == 1
        assert state["active_history_bytes"] <= state["history_budget_bytes"]
        assert not any(message.get("role") == "tool" for message in out)
        assert bool(state["gist_tokens"]) == route.startswith("ac_")
        metadata.append(state)
    assert metadata[0]["protected_event_ids"] == metadata[1]["protected_event_ids"]
    assert metadata[0]["source_needs"]["candidate_event_ids"] == metadata[1]["source_needs"]["candidate_event_ids"]
    assert metadata[0]["common_raw_prompt_tokens"] == metadata[1]["common_raw_prompt_tokens"]
    assert metadata[1]["source_coverage"]["complete_history_coverage"] is True


def test_large_requested_event_is_skipped_without_dropping_gist_or_current_goal(monkeypatch):
    source = _messages("X" * 10000)
    out, counts = _prepare(monkeypatch, "ac_native_needs_typed", source, budget=1000,
                           prediction=_request)
    state = counts["memory_runtime"]
    assert state["source_needs"]["admitted_event_ids"] == []
    assert state["source_needs"]["skipped_for_budget"][0]["event_id"] == "needs-task:m1"
    assert state["gist_tokens"] > 0 and state["active_history_bytes"] <= 1000
    assert any(message.get("content") == "Review both files." for message in out)


def test_raw_fills_freed_capacity_and_equals_full_when_everything_fits(monkeypatch):
    source = _messages()
    out, counts = _prepare(monkeypatch, "raw_native_needs_lexical", source)
    full, _ = proxy._assemble(source, get_arm("full"))
    assert out == full
    assert counts["memory_runtime"]["gist_tokens"] == 0
    assert counts["memory_runtime"]["recency_selected_event_ids"]


def test_no_history_does_not_call_predictor_or_add_gist(monkeypatch):
    def forbidden(*args):
        raise AssertionError("No older candidates exist")
    source = [{"role": "user", "content": "Start."}]
    out, counts = _prepare(monkeypatch, "ac_native_needs_typed", source, prediction=forbidden)
    full, _ = proxy._assemble(source, get_arm("full"))
    assert out == full and counts["memory_runtime"]["gist_tokens"] == 0


def test_predictor_tool_call_is_never_a_source_request(monkeypatch):
    out, counts = _prepare(monkeypatch, "ac_native_needs_typed", _messages(), prediction=lambda *args: {
        "content": '{"needs":[{"kind":"prior_result","source_ids":["needs-task:m1"]}]}',
        "tool_calls": [{"function": {"name": "delete", "arguments": "{}"}}]})
    assert counts["memory_runtime"]["source_needs"]["prediction_status"] == "invalid_prediction_abstain"
    assert counts["memory_runtime"]["retrieved_event_ids"] == []


def test_infeasible_protected_workspace_has_no_full_bypass(monkeypatch):
    with pytest.raises(proxy.MemoryRuntimeError) as failure:
        _prepare(monkeypatch, "ac_native_needs_lexical", _messages(), budget=1)
    assert failure.value.kind == "capacity_infeasible"


@pytest.mark.parametrize("representation", ["ac", "raw"])
def test_no_retrieval_control_matches_empty_request_without_predictor(monkeypatch, representation):
    def forbidden(*args):
        raise AssertionError("The no-retrieval control must not call a predictor")
    source = _messages()
    none, none_counts = _prepare(monkeypatch, representation + "_native_needs_none", source,
                                 prediction=forbidden)
    empty, empty_counts = _prepare(monkeypatch, representation + "_native_needs_typed", source,
        prediction=lambda *args: {"content": '{"needs":[]}'})
    assert none == empty
    assert none_counts["memory_runtime"]["active_history_bytes"] == empty_counts["memory_runtime"]["active_history_bytes"]
    assert none_counts["memory_runtime"]["source_needs"]["prediction_status"] == "no_retrieval_control"
    assert none_counts["memory_runtime"]["retrieved_event_ids"] == []


@pytest.mark.parametrize("representation", ["ac", "raw"])
def test_observed_state_is_visible_and_charged_in_the_shared_history_budget(monkeypatch, representation):
    out, counts = _prepare(monkeypatch, representation + "_native_state_none", _messages())
    state = counts["memory_runtime"]
    receipt = state["observed_state"]
    state_index = receipt["state_out_index"]
    assert state_index is not None and receipt["admitted_calls"] > 0
    assert "OLD-FILE-LIST" in out[state_index]["content"]
    assert '"goal_completion":"unknown"' in out[state_index]["content"]
    raw = [message for message in out if not message.get("c2kv_key_hash")]
    without = [message for index, message in enumerate(out)
               if index != state_index and not message.get("c2kv_key_hash")]
    assert state["state_bytes"] == _count(raw, []) - _count(without, []) > 0
    assert state["evidence_bytes"] == state["native_evidence_bytes"] + state["state_bytes"]
    assert state["active_history_bytes"] <= state["history_budget_bytes"]
    assert state["source_needs"]["requested_event_ids"] == []
    assert state["observed_state"]["state_counts_as_complete_event_coverage"] is False
    assert bool(state["gist_tokens"]) == (representation == "ac")


def test_state_is_removed_before_protected_raw_or_the_required_gist_is_lost(monkeypatch):
    _, reference = _prepare(monkeypatch, "ac_native_needs_none", _messages())
    meta = reference["memory_runtime"]
    budget = meta["evidence_bytes"] + min(block["gist_tokens"] for block in meta["block_refs"])
    expected, _ = _prepare(monkeypatch, "ac_native_needs_none", _messages(), budget=budget)
    out, counts = _prepare(monkeypatch, "ac_native_state_none", _messages(), budget=budget)
    state = counts["memory_runtime"]
    assert out == expected
    assert state["observed_state"]["dropped_for_workspace"]
    assert state["observed_state"]["admitted_calls"] == 0 and state["state_bytes"] == 0
    assert state["gist_tokens"] > 0 and state["active_history_bytes"] <= budget
    assert any(message.get("content") == "Review both files." for message in out)


@pytest.mark.parametrize("representation", ["ac", "raw"])
def test_empty_observed_state_preserves_the_original_full_input(monkeypatch, representation):
    source = [{"role": "user", "content": "Start."}]
    out, counts = _prepare(monkeypatch, representation + "_native_state_none", source)
    full, _ = proxy._assemble(source, get_arm("full"))
    assert out == full and counts["memory_runtime"]["state_bytes"] == 0
    assert counts["memory_runtime"]["observed_state"]["admitted_calls"] == 0
