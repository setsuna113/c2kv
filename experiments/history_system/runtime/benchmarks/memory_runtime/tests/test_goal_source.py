"""Bounded Goal-Source policy behavior at the observed source/commit seam."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.candidate_algorithms.goal_source import Policy  # noqa: E402
from benchmarks.memory_runtime.candidate_algorithms.repair_protocol import RepairContext  # noqa: E402
from history_memory.events import EventStore  # noqa: E402


def _call(name, arguments, call_id="draft"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


def _observed(name, arguments, result, call_id):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            _call(name, arguments, call_id)]},
        {"role": "tool", "tool_call_id": call_id,
         "content": json.dumps(result)},
    ]


def _context(rows, calls, budget=100_000):
    store = EventStore.from_messages("goal-source-test", rows)
    return RepairContext(SimpleNamespace(_store=store), tuple(calls), "", None,
                         lambda messages: len(messages[0]["content"]), budget)


def _packet(proposal):
    return json.loads(proposal.messages[0]["content"].split("\n", 1)[1])


def _wc_rows():
    return [
        {"role": "user", "content": "Inspect the last line."},
        *_observed("tail", {"file_name": "DataSet1.csv", "lines": 1},
                   {"last_lines": "Bob | 10 | 7"}, "tail"),
        {"role": "user", "content": "Count lines, words, and characters in the file."},
        *_observed("wc", {"file_name": "DataSet1.csv", "mode": "l"},
                   {"count": 3, "type": "lines"}, "lines"),
        *_observed("wc", {"file_name": "DataSet1.csv", "mode": "w"},
                   {"count": 16, "type": "words"}, "words"),
        *_observed("wc", {"file_name": "DataSet1.csv", "mode": "c"},
                   {"count": 60, "type": "characters"}, "characters"),
        {"role": "user", "content": "Compute the average of the three numerical values obtained."},
    ]


def test_numeric_reference_selects_complete_previous_request_groups_only():
    context = _context(_wc_rows(), [_call("mean", {"numbers": [5, 10, 7]})])
    proposal = Policy().propose(context)
    assert proposal is not None
    groups = _packet(proposal)["source_groups"]
    assert [group["observed_result"] for group in groups] == [
        {"count": 3, "type": "lines"},
        {"count": 16, "type": "words"},
        {"count": 60, "type": "characters"},
    ]
    assert [group["producer"]["arguments"]["mode"] for group in groups] == [
        "l", "w", "c"]
    assert len(proposal.receipt["source_groups"]) == 3
    assert Policy().validate(context, proposal,
                             [_call("mean", {"numbers": [3, 16, 60]})],
                             draft_text="").accepted
    assert not Policy().validate(context, proposal, [], draft_text="Done").accepted


def test_numeric_packet_does_not_truncate_a_required_group_to_fit_budget():
    context = _context(_wc_rows(), [_call("mean", {"numbers": [5, 10, 7]})])
    proposal = Policy().propose(context)
    assert proposal is not None
    smaller = _context(_wc_rows(), context.draft_tool_calls,
                       budget=proposal.receipt["packet_tokens"] - 1)
    rejected = Policy().propose(smaller)
    assert rejected.receipt["status"] == "packet_exceeds_budget"
    assert _packet(rejected)["source_groups"] == _packet(proposal)["source_groups"]


def test_numeric_reference_does_not_join_different_source_entities():
    rows = [{"role": "user", "content": "Count three things."},
            *_observed("wc", {"file_name": "one.csv", "mode": "l"},
                       {"count": 3, "type": "lines"}, "one"),
            *_observed("wc", {"file_name": "one.csv", "mode": "w"},
                       {"count": 16, "type": "words"}, "two"),
            *_observed("wc", {"file_name": "other.csv", "mode": "c"},
                       {"count": 60, "type": "characters"}, "three"),
            {"role": "user", "content":
             "Compute the average of the three numerical values obtained."}]
    assert Policy().propose(_context(rows, [_call("mean", {"numbers": [5, 10, 7]})])) is None


def test_correct_draft_and_unrelated_grounded_field_do_not_trigger_review():
    assert Policy().propose(_context(
        _wc_rows(), [_call("mean", {"numbers": [3, 16, 60]})])) is None
    rows = [{"role": "user", "content": "Find the message."},
            *_observed("get_message", {"query": "latest"},
                       {"message_id": "MSG-12345"}, "message"),
            {"role": "user", "content": "Delete the selected message."}]
    assert Policy().propose(_context(
        rows, [_call("delete_message", {"message_id": "MSG-12345"})])) is None


def test_exact_literal_patch_keeps_call_shape_and_grounded_other_field():
    rows = [{"role": "user", "content": "Find the message."},
            *_observed("get_message", {"query": "latest"},
                       {"message_id": "MSG-12345"}, "message"),
            {"role": "user", "content":
             "Delete message_id MSG-12345 with access token ABCDE12345."}]
    original = _call("delete_message", {
        "message_id": "MSG-12345", "access_token": "OLD123"})
    candidate = _call("delete_message", {
        "message_id": "WRONG", "access_token": "OLD123"})
    context = _context(rows, [original])
    patched, receipt = Policy().patch_calls(context, [candidate])
    assert len(patched) == 1 and patched[0]["function"]["name"] == "delete_message"
    assert json.loads(patched[0]["function"]["arguments"]) == {
        "message_id": "MSG-12345", "access_token": "ABCDE12345"}
    assert receipt["status"] == "patched"
    assert len(receipt["patches"]) == 2
    assert json.loads(candidate["function"]["arguments"])["access_token"] == "OLD123"


def test_ambiguous_source_and_literal_do_not_patch():
    rows = [{"role": "user", "content": "Find messages."},
            *_observed("get_message", {"query": "first"},
                       {"message_id": "MSG-1"}, "first"),
            *_observed("get_message", {"query": "second"},
                       {"message_id": "MSG-2"}, "second"),
            {"role": "user", "content":
             "Delete the selected message; token is OLD or token is NEW."}]
    draft = _call("delete_message", {"message_id": "MSG-1", "token": "OTHER"})
    changed = _call("delete_message", {"message_id": "GUESS", "token": "OTHER"})
    patched, receipt = Policy().patch_calls(_context(rows, [draft]), [changed])
    assert patched == (changed,)
    assert receipt["status"] == "unchanged"
    extra = _call("archive_message", {"message_id": "MSG-1"}, "extra")
    patched, receipt = Policy().patch_calls(_context(rows, [draft]), [changed, extra])
    assert patched == (changed, extra)
    assert receipt["status"] == "unchanged"


def test_ordinal_requires_one_observable_entity_list():
    rows = [{"role": "user", "content": "List messages."},
            *_observed("list_messages", {}, {"messages": [
                {"message_id": "MSG-1"}, {"message_id": "MSG-2"}]}, "list"),
            {"role": "user", "content": "Delete the second message."}]
    context = _context(rows, [_call("delete_message", {"message_id": "MSG-1"})])
    proposal = Policy().propose(context)
    assert proposal is not None
    assert proposal.receipt["ordinal_conflicts"][0]["requested_value"] == "MSG-2"
    assert not Policy().validate(context, proposal, context.draft_tool_calls,
                                 draft_text="").accepted
    assert Policy().validate(context, proposal,
                             [_call("delete_message", {"message_id": "MSG-2"})],
                             draft_text="").accepted
    ambiguous = _context([*rows[:-1],
                          *_observed("list_messages", {}, {"messages": [
                              {"message_id": "MSG-3"}, {"message_id": "MSG-4"}]}, "new-list"),
                          rows[-1]], context.draft_tool_calls)
    assert Policy().propose(ambiguous) is None


def test_last_scalar_list_and_travel_to_alias_use_observed_order():
    stock_rows = [{"role": "user", "content": "Display stocks in my watchlist."},
                  *_observed("get_watchlist", {}, {"watchlist": ["NVDA", "AAPL"]}, "stocks"),
                  {"role": "user", "content":
                   "Among the stocks listed, the last one looks promising. Buy shares."}]
    stock = _context(stock_rows, [_call("get_stock_info", {"symbol": "NVDA"})])
    proposal = Policy().propose(stock)
    assert proposal is not None
    assert proposal.receipt["ordinal_conflicts"][0]["requested_value"] == "AAPL"

    airport_rows = [{"role": "user", "content":
                     "Calculate a ticket from the first airport to the last airport."},
                    *_observed("list_all_airports", {}, "['RMS', 'SBK', 'BOS']", "airports")]
    airport = _context(airport_rows, [_call("get_flight_cost", {
        "travel_from": "RMS", "travel_to": "SBK", "travel_class": "business"})])
    proposal = Policy().propose(airport)
    assert proposal is not None
    assert proposal.receipt["ordinal_conflicts"][0]["path"] == ["travel_to"]
    assert proposal.receipt["ordinal_conflicts"][0]["requested_value"] == "BOS"
    assert _packet(proposal)["source_groups"][0]["observed_result"] == "['RMS', 'SBK', 'BOS']"
    assert "tool_call_id" not in proposal.messages[0]["content"]


def test_batch_dependency_receipt_preserves_consumer_for_later_goal_review():
    rows = [{"role": "user", "content": "Find the airport, then book the flight."}]
    producer = _call("get_nearest_airport", {"city": "Rivermist"}, "producer")
    consumer = _call("book_flight", {"origin": "RIV", "passengers": 1}, "consumer")
    context = _context(rows, [producer, consumer])
    proposal = Policy().propose(context)
    assert proposal is not None
    deferred = proposal.guard["deferred_consumers"]
    assert deferred == proposal.receipt["deferred_consumers"]
    assert len(deferred) == 1
    assert deferred[0]["call"] == consumer
    assert deferred[0]["producer_calls"] == [producer]
    assert deferred[0]["request_event_id"] == proposal.receipt["request_event_id"]
    assert deferred[0]["unbound_paths"] == [["origin"]]
    assert deferred[0]["binding_status"] == "unresolved_until_producer_observed"
    assert Policy().validate(context, proposal, [producer], draft_text="").accepted
    assert not Policy().validate(context, proposal, [producer, consumer],
                                 draft_text="").accepted
    assert not Policy().validate(context, proposal, [consumer], draft_text="").accepted
    assert not Policy().validate(context, proposal, [producer,
        _call("book_flight", {"origin": "OTHER", "passengers": 1})],
        draft_text="").accepted
    unrelated = _context([{"role": "user", "content":
                          "Find the order, then send a message."}], [
        _call("get_order", {"query": "latest"}),
        _call("send_message", {"message_id": "MSG-1"})])
    assert Policy().propose(unrelated) is None


def test_airport_cost_booking_batch_defers_each_consumer_in_order():
    rows = [{"role": "user", "content":
             "Travel from Rivermist to Los Angeles; double check the flight cost "
             "and then pay for the business class booking."}]
    airport = _call("get_nearest_airport_by_city", {"location": "Rivermist"}, "airport")
    cost = _call("get_flight_cost", {"travel_from": "RIV", "travel_to": "LAX",
                                     "travel_class": "business"}, "cost")
    book = _call("book_flight", {"travel_from": "RIV", "travel_to": "LAX",
                                 "travel_class": "business", "card_id": "1_3456"}, "book")
    context = _context(rows, [airport, cost, book])
    proposal = Policy().propose(context)
    assert proposal is not None
    deferred = proposal.guard["deferred_consumers"]
    assert [entry["call"]["function"]["name"] for entry in deferred] == [
        "get_flight_cost", "book_flight"]
    assert deferred[0]["producer_calls"] == [airport]
    assert deferred[1]["producer_calls"] == [cost]
    assert ["travel_from"] in deferred[0]["unbound_paths"]
    assert ["travel_to"] in deferred[1]["unbound_paths"]
    assert Policy().validate(context, proposal, [airport], draft_text="").accepted
    assert not Policy().validate(context, proposal, [airport, cost], draft_text="").accepted
    assert "\"id\":\"airport\"" not in proposal.messages[0]["content"]


def test_changed_entity_does_not_restore_old_grounded_field():
    rows = [{"role": "user", "content": "Find Alice's message."},
            *_observed("get_message", {"recipient": "Alice"},
                       {"message_id": "MSG-A"}, "alice"),
            {"role": "user", "content": "Send the selected message."}]
    original = _call("send_message", {"recipient": "Alice", "message_id": "MSG-A"})
    switched = _call("send_message", {"recipient": "Bob", "message_id": "MSG-B"})
    patched, receipt = Policy().patch_calls(_context(rows, [original]), [switched])
    assert patched == (switched,)
    assert receipt["status"] == "unchanged"
    missing = _call("send_message", {"recipient": "Alice"})
    patched, receipt = Policy().patch_calls(_context(rows, [original]), [missing])
    assert json.loads(patched[0]["function"]["arguments"])["message_id"] == "MSG-A"
    assert receipt["patches"][0]["reason"] == "same_entity_grounded_field"
    original_without_recipient = _call("send_message", {"message_id": "MSG-A"})
    introduced_recipient = _call("send_message", {
        "recipient": "Bob", "message_id": "MSG-B"})
    patched, receipt = Policy().patch_calls(
        _context(rows, [original_without_recipient]), [introduced_recipient])
    assert patched == (introduced_recipient,)
    assert receipt["status"] == "unchanged"
