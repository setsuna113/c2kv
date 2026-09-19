"""CPU regressions for source-bound repair observations and no-progress review."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.events import EventStore  # noqa: E402
from benchmarks.memory_runtime.acebench_source import build_ace_event_store  # noqa: E402
from benchmarks.memory_runtime.candidate_algorithms.no_progress import (  # noqa: E402
    Policy, detect_no_progress,
)
from benchmarks.memory_runtime.candidate_algorithms.observations import (  # noqa: E402
    current_request, failure_reported, operation_records,
)
from benchmarks.memory_runtime.candidate_algorithms.repair_protocol import RepairContext  # noqa: E402


def call(name="lookup", arguments=None, call_id="c0"):
    if arguments is None:
        arguments = {"id": 1}
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


def pair(name="lookup", arguments=None, result=None, call_id="c0"):
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [call(name, arguments, call_id)]},
        {"role": "tool", "tool_call_id": call_id,
         "content": json.dumps(result if result is not None else {"value": 1})},
    ]


def context(store, draft_calls):
    return RepairContext(
        prepared=SimpleNamespace(_store=store),
        draft_tool_calls=tuple(draft_calls), draft_text="", parse_error=None,
        token_counter=lambda rows: len(rows[0]["content"]), token_budget=100_000,
    )


def ace_store(actions, results, *, return_count=None):
    rows = [
        {"role": "system", "content": "Available APIs"},
        {"role": "user", "content": "Look up records."},
        {"role": "assistant", "content": actions},
        {"role": "tool", "tool_call_id": "acebench-execution-2",
         "content": json.dumps(results)},
    ]
    from benchmarks.memory_runtime.acebench_source import parse_acebench_draft
    draft = parse_acebench_draft(actions, call_id_prefix="test")
    decoded = [part.strip() for part in actions.strip()[1:-1].split(",")]
    receipt = {
        "version": "acebench-execution-receipt-v1",
        "execution_message_index": 3,
        "agent_history_index": 1,
        "decode_status": "ok",
        "decoded_calls": decoded,
        "executor_status": "returned",
        "executor_return_shape": "list",
        "executor_return_count": len(draft.tool_calls) if return_count is None else return_count,
    }
    source = {"version": "acebench-text-actions-v1", "receipts": [receipt]}
    return build_ace_event_store("ace", rows, source)


def test_native_records_keep_exact_call_and_result_source_for_parallel_results():
    rows = [{"role": "system", "content": "Tools"},
            {"role": "user", "content": "Find both."},
            {"role": "assistant", "content": None, "tool_calls": [
                call("lookup", {"id": 1}, "first"),
                call("lookup", {"id": 2}, "second"),
            ]},
            {"role": "tool", "tool_call_id": "second", "content": '{"value":2}'},
            {"role": "tool", "tool_call_id": "first", "content": '{"value":1}'},
            {"role": "user", "content": "Now find another."},
            *pair("lookup", {"id": 3}, {"value": 3}, "third")]
    store = EventStore.from_messages("native", rows)
    all_rows = operation_records(store)
    assert [(row.call_source_index, row.result_source_index) for row in all_rows] == [
        (2, 4), (2, 3), (6, 7)]
    assert [row.tool_call_id for row in all_rows] == ["first", "second", "third"]
    goal, current = current_request(store)
    assert goal.event_id == "native:m5"
    assert [row.tool_call_id for row in current] == ["third"]


def test_ace_receipt_backed_text_actions_pair_only_exact_per_call_results():
    store = ace_store("[Lookup(key=1),Lookup(key=2)]", [
        {"value": 1}, "Error during execution: missing key",
    ])
    rows = operation_records(store)
    assert [(row.tool, row.arguments, row.result_source_index) for row in rows] == [
        ("Lookup", {"key": 1}, 3), ("Lookup", {"key": 2}, 3)]
    assert [row.failure_reported for row in rows] == [False, True]
    assert all(row.event_id == "ace:m2" for row in rows)
    assert all(row.call_source_index == 2 for row in rows)

    ambiguous = ace_store("[Lookup(key=1),Lookup(key=2)]", [{"aggregate": 2}])
    assert ambiguous.events[-1].complete is False
    assert operation_records(ambiguous) == ()

    # Generic grouping without the official receipt cannot upgrade text to a
    # call/result observation.
    raw = EventStore.from_messages("ace-raw", [m.to_dict() for m in store.messages])
    assert operation_records(raw) == ()


def test_failure_status_requires_explicit_error_report():
    assert failure_reported("Error during execution: key not found")
    assert failure_reported({"error": "key not found"})
    assert failure_reported({"success": False})
    assert not failure_reported("Reference: Error during execution: key not found")
    assert not failure_reported({"message": "Error during execution: key not found"})
    assert not failure_reported({"error": None, "success": True})


def test_no_progress_detects_repeated_success_but_keeps_stop_legal():
    rows = [{"role": "user", "content": "Check a value."},
            *pair(result={"value": 7}, call_id="one"),
            *pair(result={"value": 7}, call_id="two")]
    store = EventStore.from_messages("success", rows)
    trigger = detect_no_progress(store, [call("other", {"id": 9}, "draft")])
    assert trigger.reason == "repeated_exact_observation"
    assert [row.observation_version for row in trigger.sources] == [1, 2]
    assert detect_no_progress(store, []) is None
    proposal = Policy().propose(context(store, [call("other", {"id": 9}, "draft")]))
    assert proposal is not None
    assert proposal.receipt["goal_completion"] == "unknown"
    verdict = Policy().validate(
        context(store, [call("other", {"id": 9}, "draft")]), proposal,
        [], draft_text="Done")
    assert verdict.accepted and verdict.reason == "stop_is_legal"


def test_failed_repeat_guard_resets_on_changed_argument_state_or_user():
    rows = [{"role": "user", "content": "Look up an item."},
            *pair(result="Error during execution: missing", call_id="one")]
    store = EventStore.from_messages("failure", rows)
    original = call("lookup", {"id": 1}, "draft")
    ctx = context(store, [original])
    proposal = Policy().propose(ctx)
    assert proposal is not None
    assert proposal.reason == "draft_repeats_failed_operation"
    assert '"tool_call_id"' not in proposal.messages[0]["content"]
    assert proposal.receipt["observed_sources"][0]["tool_call_id"] == "one"
    same_action_new_id = context(store, [call("lookup", {"id": 1}, "new-transport-id")])
    assert (Policy().propose(same_action_new_id).receipt["source_state_sha256"]
            == proposal.receipt["source_state_sha256"])
    assert Policy().validate(ctx, proposal, [original], draft_text="").fallback == "stop"
    assert Policy().validate(ctx, proposal, [call("lookup", {"id": 2})],
                             draft_text="").accepted

    new_user = EventStore.from_messages("new-user", [*rows,
        {"role": "user", "content": "Try again."}])
    assert detect_no_progress(new_user, [original]) is None

    changed_state = EventStore.from_messages("changed", [*rows,
        *pair("refresh", {}, {"state": "updated"}, "refresh")])
    assert detect_no_progress(changed_state, [original]) is None


def test_transient_failure_retry_is_legal_and_complete_over_budget_packet_is_kept():
    rows = [{"role": "user", "content": "Look up an item."},
            *pair(result="Error during execution: timeout", call_id="one")]
    store = EventStore.from_messages("transient", rows)
    repeated = call("lookup", {"id": 1}, "new-transport-id")
    assert detect_no_progress(store, [repeated]) is None
    assert Policy().propose(context(store, [repeated])) is None

    twice = EventStore.from_messages("transient-twice", [
        *rows, *pair(result="Error during execution: timeout", call_id="two")])
    ctx = context(twice, [call("other", {"id": 2})])
    proposal = Policy().propose(ctx)
    assert proposal.reason == "repeated_exact_observation"
    assert Policy().validate(ctx, proposal, [repeated], draft_text="").accepted
    oversized = Policy().propose(replace(ctx, token_budget=1))
    assert oversized.receipt["status"] == "packet_over_budget"
    assert oversized.messages == proposal.messages
    assert oversized.guard == proposal.guard


def test_parallel_event_does_not_prove_unchanged_intervening_state():
    rows = [
        {"role": "user", "content": "Run the available actions."},
        {"role": "assistant", "content": None, "tool_calls": [
            call("refresh", {}, "b"),
            call("lookup", {"id": 1}, "a"),
        ]},
        {"role": "tool", "tool_call_id": "a",
         "content": '"Error during execution: missing"'},
        {"role": "tool", "tool_call_id": "b",
         "content": '{"state":"updated"}'},
    ]
    store = EventStore.from_messages("parallel", rows)
    assert detect_no_progress(store, [call("refresh", {})]) is None
    assert detect_no_progress(store, [call("lookup", {"id": 1})]) is None
