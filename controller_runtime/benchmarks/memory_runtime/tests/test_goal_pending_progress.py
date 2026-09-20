"""CPU contracts for Goal-Pending and Goal-Progress policy packets."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.events import EventStore  # noqa: E402
from benchmarks.memory_runtime.candidate_algorithms import goal_pending, goal_progress  # noqa: E402
from benchmarks.memory_runtime.candidate_algorithms.progress import (  # noqa: E402
    operation_records as goal_operation_records, review_messages as original_review_messages,
)
from benchmarks.memory_runtime.candidate_algorithms.repair_protocol import RepairContext  # noqa: E402


def call(name="get_order_details", arguments=None, call_id="draft"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments or {"order_id": "o1"}),
    }}


def pair(name="get_order_details", arguments=None, result=None, call_id="one"):
    return [
        {"role": "assistant", "content": None, "tool_calls": [call(name, arguments, call_id)]},
        {"role": "tool", "tool_call_id": call_id,
         "content": json.dumps(result if result is not None else {"status": "open"})},
    ]


def context(rows, draft_calls=(), *, budget=100_000, draft_text="Done", tools=()):
    store = EventStore.from_messages("goal", rows)
    return RepairContext(SimpleNamespace(_store=store, _tools=list(tools)),
                         tuple(draft_calls), draft_text,
                         None, lambda messages: sum(len(row["content"]) for row in messages),
                         budget)


def original_goal_review(ctx):
    goal, records = goal_operation_records(ctx.prepared._store)
    messages, receipt = original_review_messages(
        ctx.prepared._store, goal, records,
        token_counter=ctx.token_counter, token_budget=ctx.token_budget,
    )
    assert receipt["status"] == "prepared"
    return messages


def packet(message):
    return json.loads(message["content"].split("\n", 1)[1])


def test_pending_preserves_original_goal_review_and_does_not_anchor_stop_prose():
    rows = [{"role": "user", "content": "Look up the item, then place the order."},
            *pair(result={"item": "widget"})]
    ctx = context(rows, draft_text="Everything is complete.")
    original = original_goal_review(ctx)
    proposal = goal_pending.review_messages(ctx, original)
    assert proposal is not None
    assert proposal.messages[:len(original)] == original
    status = packet(proposal.messages[-1])
    assert packet(original[0])["original_request"] == [rows[0]["content"]]
    assert status["operation_statuses"] == [{"result_source_index": 2,
                                               "status": "lookup_returned"}]
    assert "observed_result" not in proposal.messages[-1]["content"]
    assert "original_request" not in proposal.messages[-1]["content"]
    assert status["obligations"][0]["status"] == "requires_model_review"
    assert "Everything is complete" not in proposal.messages[-1]["content"]
    assert proposal.receipt["original_goal_review_preserved"] is True
    assert proposal.receipt["goal_completion"] == "unknown"


def test_pending_distinguishes_explicit_execution_receipts_and_prior_turn_actions():
    rows = [{"role": "user", "content": "Place an order."},
            *pair("place_order", result={"success": True, "order_id": "o1"}),
            {"role": "user", "content": "Place another order."},
            *pair("place_order", result={"order_id": "o2"}, call_id="two"),
            *pair("place_order", result={"success": False}, call_id="three")]
    ctx = context(rows)
    proposal = goal_pending.review_messages(ctx, original_goal_review(ctx))
    assert proposal is not None
    status = packet(proposal.messages[-1])
    assert [row["status"] for row in status["operation_statuses"]] == [
        "execution_completion_unknown", "execution_reported_failure"]
    assert [row["result_source_index"] for row in status["operation_statuses"]] == [5, 7]
    assert "o2" not in proposal.messages[-1]["content"]


def test_pending_deferred_consumer_carries_original_anchor_and_actual_producer_receipt():
    rows = [{"role": "user", "content": "Create a project."},
            *pair("create_project", {"name": "A"}, {"project_id": "p1"}),
            {"role": "user", "content": "Give its status."},
            *pair("lookup_project", {"project_id": "p1"}, {"status": "active"}, "lookup")]
    ctx = context(rows)
    deferred = ({"request_event_id": "goal:m0",
                 "call": call("notify_project", {"project_id": "p1"}, "transport-consumer"),
                 "producer_calls": [call("create_project", {"name": "A"},
                                         "transport-producer")],
                 "unbound_paths": [["project_id"]],
                 "producer": {"tool": "create_project", "result_source_index": 2,
                              "observed_result": {"project_id": "p1"}}},)
    proposal = goal_pending.review_messages(
        ctx, original_goal_review(ctx), deferred_consumers=deferred)
    assert proposal is not None
    status = packet(proposal.messages[-1])
    assert status["deferred_consumers"][0]["original_request_event_id"] == "goal:m0"
    assert status["deferred_consumers"][0]["original_consumer"] == {
        "tool": "notify_project", "arguments": {"project_id": "p1"}}
    assert status["deferred_consumers"][0]["producer_source"] == {
        "result_source_index": 2, "status": "execution_completion_unknown",
        "observed_result": {"project_id": "p1"}}
    assert status["deferred_consumers"][0]["status"] == "requires_model_review"
    assert status["operation_statuses"][0]["status"] == "lookup_returned"
    assert "transport-consumer" not in proposal.messages[-1]["content"]
    assert "transport-producer" not in proposal.messages[-1]["content"]
    assert proposal.receipt["deferred_consumers"] == list(deferred)


def test_pending_abstains_for_call_draft_or_if_original_packet_cannot_fit():
    rows = [{"role": "user", "content": "Check an order."}, *pair()]
    ctx = context(rows)
    original = original_goal_review(ctx)
    assert goal_pending.review_messages(replace(ctx, draft_tool_calls=(call(),)), original) is None
    assert goal_pending.review_messages(replace(ctx, token_budget=ctx.token_counter(original)),
                                        original) is None


def test_pending_compact_supplement_fits_without_duplicating_large_goal_record():
    rows = [{"role": "user", "content": "Inspect this large record and finish."},
            *pair(result={"record": "x" * 1200})]
    ctx = context(rows)
    original = original_goal_review(ctx)
    proposal = goal_pending.review_messages(ctx, original)
    assert proposal is not None
    assert proposal.messages[0] == original[0]
    assert "x" * 1200 not in proposal.messages[-1]["content"]
    assert proposal.receipt["packet_tokens"] < ctx.token_counter(original) + 1200
    tight = goal_pending.review_messages(
        replace(ctx, token_budget=proposal.receipt["packet_tokens"]), original)
    assert tight is not None
    assert tight.messages == proposal.messages


def test_pending_execution_name_coverage_still_requires_explicit_success():
    rows = [{"role": "user", "content": "Run the requested actions."}]
    for number, name in enumerate(("lockDoors", "fillFuelTank", "startEngine",
                                   "pressBrakePedal", "cp", "mkdir")):
        rows.extend(pair(name, {}, {"message": "done"}, f"action-{number}"))
    proposal = goal_pending.review_messages(context(rows),
                                            original_goal_review(context(rows)))
    assert proposal is not None
    assert [row["status"] for row in packet(proposal.messages[-1])["operation_statuses"]] == [
        "execution_completion_unknown"] * 6


def test_pending_deferred_producer_receipt_references_admitted_record():
    rows = [{"role": "user", "content": "Look up a project, then notify it."},
            *pair("lookup_project", {"project_id": "p1"},
                  {"project_id": "p1", "status": "active"}, "lookup")]
    ctx = context(rows)
    original = original_goal_review(ctx)
    deferred = ({"request_event_id": "goal:m0",
                 "call": call("notify_project", {"project_id": "p1"}, "transport-id"),
                 "unbound_paths": [["project_id"]],
                 "producer": {"tool": "lookup_project", "result_source_index": 2,
                              "observed_result": {"project_id": "p1", "status": "active"}}},)
    proposal = goal_pending.review_messages(ctx, original, deferred_consumers=deferred)
    assert proposal is not None
    source = packet(proposal.messages[-1])["deferred_consumers"][0]["producer_source"]
    assert source == {"result_source_index": 2, "status": "lookup_returned"}
    assert "project_id" in original[0]["content"]
    assert "transport-id" not in proposal.messages[-1]["content"]


def test_progress_repeat_and_nonconsecutive_no_new_information_repeat():
    rows = [{"role": "user", "content": "Check the order and proceed."},
            *pair(result={"status": "open"}, call_id="one")]
    ctx = context(rows, (call(call_id="held"),))
    proposal = goal_progress.Policy().propose(ctx)
    assert proposal is not None
    assert proposal.receipt["goal_completion"] == "unknown"
    assert proposal.receipt["omitted_operation_count"] == 0
    assert "Check the order and proceed." in proposal.messages[0]["content"]
    assert '"observed_operations"' in proposal.messages[0]["content"]
    assert proposal.receipt["state_key"]

    later = [*rows, *pair("get_stock_info", {"symbol": "ABC"}, {"price": 2}, "stock"),
             *pair(result={"status": "open"}, call_id="order-again"),
             *pair("get_stock_info", {"symbol": "ABC"}, {"price": 2}, "stock-again")]
    later_proposal = goal_progress.Policy().propose(context(later, (call(),)))
    assert later_proposal is not None
    assert later_proposal.receipt["observed_operation_count"] == 4
    assert later_proposal.receipt["omitted_operation_count"] == 0
    assert len(packet(later_proposal.messages[0])["observed_operations"]) == 4


def test_progress_new_evidence_mutation_new_user_or_unknown_tool_resets_epoch():
    read = [{"role": "user", "content": "Check the order."},
            *pair(result={"status": "open"}, call_id="one")]
    held = (call(),)
    cases = [
        [*read, *pair("get_stock_info", {"symbol": "ABC"},
                      {"price": 2}, "new-evidence")],
        [*read, *pair("cd", {"path": "/tmp"}, {"ok": True}, "cd")],
        [*read, *pair("custom_tool", {}, {"ok": True}, "unknown")],
        [*read, {"role": "user", "content": "Check another order."}],
    ]
    for rows in cases:
        assert goal_progress.Policy().propose(context(rows, held)) is None

    first = goal_progress.Policy().propose(context(read, held))
    repeated = goal_progress.Policy().propose(context(
        [*read, *pair(result={"status": "open"}, call_id="same-result")], held))
    changed = goal_progress.Policy().propose(context(
        [*read, *pair(result={"status": "shipped"}, call_id="changed-result")], held))
    assert first is not None and repeated is not None and changed is not None
    assert first.receipt["state_key"] == repeated.receipt["state_key"]
    assert first.receipt["state_key"] != changed.receipt["state_key"]


def test_progress_budget_reports_complete_admitted_records_and_omissions():
    rows = [{"role": "user", "content": "Inspect the records."}]
    for number in range(4):
        rows.extend(pair("get_order_details", {"order_id": f"o{number}"},
                         {"detail": "x" * 80}, f"read-{number}"))
    held = (call(arguments={"order_id": "o3"}),)
    full_ctx = context(rows, held)
    full = goal_progress.Policy().propose(full_ctx)
    assert full is not None and full.receipt["omitted_operation_count"] == 0
    partial = None
    for budget in range(full.receipt["packet_tokens"] - 1, 0, -1):
        proposal = goal_progress.Policy().propose(replace(full_ctx, token_budget=budget))
        if proposal is not None and proposal.receipt["omitted_operation_count"]:
            partial = proposal
            break
    assert partial is not None
    assert partial.receipt["packet_tokens"] <= partial.receipt["token_budget"]
    assert (len(packet(partial.messages[0])["observed_operations"])
            == partial.receipt["admitted_operation_count"])
    assert (partial.receipt["admitted_operation_count"]
            + partial.receipt["omitted_operation_count"] == 4)


def test_progress_rejects_stop_malformed_and_repeated_read_without_synthetic_stop():
    rows = [{"role": "user", "content": "Check the order, then place the order."},
            *pair(result={"status": "open"}, call_id="one")]
    tools = ({"type": "function", "function": {"name": "place_order",
                                             "description": "Place an order"}},
             {"type": "function", "function": {"name": "send_message",
                                             "description": "Send a message"}},
             {"type": "function", "function": {"name": "get_stock_info",
                                             "description": "Get stock information"}})
    ctx = context(rows, (call(),), tools=tools)
    policy = goal_progress.Policy()
    proposal = policy.propose(ctx)
    assert proposal is not None
    for candidate, parse_error in [((), None), (({"function": {"name": ""}},), None),
                                   ((call(call_id="new-id"),), None), ((), "bad parse")]:
        verdict = policy.validate(ctx, proposal, candidate, draft_text="Done",
                                  parse_error=parse_error)
        assert not verdict.accepted
        assert verdict.fallback == "original"
    assert policy.validate(ctx, proposal, (call("place_order"),),
                           draft_text="").accepted
    assert not policy.validate(ctx, proposal, (call("send_message"),),
                               draft_text="").accepted
    assert not policy.validate(ctx, proposal, (call("get_stock_info"),),
                               draft_text="").accepted
    buy_rows = [{"role": "user", "content": "Check the order, then buy an item."},
                *pair(result={"status": "open"}, call_id="one")]
    buy_tools = ({"type": "function", "function": {
        "name": "place_order", "description": "Place an order to buy an item"}},)
    buy_ctx = context(buy_rows, (call(),), tools=buy_tools)
    buy_proposal = policy.propose(buy_ctx)
    assert buy_proposal is not None
    assert policy.validate(buy_ctx, buy_proposal, (call("place_order"),),
                           draft_text="").accepted


def test_progress_requires_single_pure_read_and_no_incomplete_intervening_event():
    rows = [{"role": "user", "content": "Check the order."}, *pair()]
    assert goal_progress.Policy().propose(context(rows, (call(), call("get_stock_info")))) is None
    assert goal_progress.Policy().propose(context(rows, (call("place_order"),))) is None
    pending = [*rows, {"role": "assistant", "content": None,
                       "tool_calls": [call("place_order", call_id="unreturned")]}]
    assert goal_progress.Policy().propose(context(pending, (call(),))) is None


def test_progress_abstains_for_explicit_polling_and_time_varying_reads():
    polling = [{"role": "user", "content": "Poll the order until it ships."},
               *pair(result={"status": "open"})]
    assert goal_progress.Policy().propose(context(polling, (call(),))) is None
    for name in ("get_current_time", "get_outside_temperature_from_weather_com",
                 "get_outside_temperature_from_google"):
        rows = [{"role": "user", "content": "Check the latest value."},
                *pair(name, {}, {"value": 1})]
        assert goal_progress.Policy().propose(context(rows, (call(name, {}),))) is None
