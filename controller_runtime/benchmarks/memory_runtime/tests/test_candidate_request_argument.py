"""Source-backed review and commit guards for the two opt-in repair policies."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.candidate_algorithms.argument_binding import (
    Policy as BindingPolicy,
)
from benchmarks.memory_runtime.candidate_algorithms.repair_protocol import RepairContext
from benchmarks.memory_runtime.candidate_algorithms.request_contract import (
    Policy as RequestPolicy,
)
from history_memory.events import EventStore


def _call(call_id, name, arguments, result):
    return [
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        }]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)},
    ]


def _draft(name, arguments):
    return {"type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


def _context(rows, calls=(), *, text="Done"):
    store = EventStore.from_messages("repair", rows)
    return RepairContext(SimpleNamespace(_store=store), tuple(calls), text, None,
                         lambda messages: sum(len(m["content"]) for m in messages),
                         100_000)


def _wire(proposal):
    return json.loads(proposal.messages[0]["content"].split("\n", 1)[1])


def test_request_reviews_ordinary_call_and_stop_after_observed_current_action():
    rows = [{"role": "user", "content": "Find the fare and book a flight."},
            *_call("fare", "get_fare", {"route": "A-B"}, {"fare": 72})]
    policy = RequestPolicy()
    call = policy.propose(_context(rows, [_draft("book_flight", {"fare": 72})]))
    stop = policy.propose(_context(rows))
    assert call is not None and stop is not None
    assert call.receipt["held_action_kind"] == "tool_calls"
    assert stop.receipt["held_action_kind"] == "stop"
    assert len(_wire(call)["observed_operations"]) == 1
    assert call.receipt["completion"] == "unknown"


def test_request_scope_does_not_resurrect_prior_turn_order():
    rows = [{"role": "user", "content": "Buy 50 ZETA shares."},
            *_call("order", "place_order", {"shares": 50, "symbol": "ZETA"},
                   {"order_id": "O1", "success": True}),
            {"role": "user", "content": "Top up my account with $5,000."},
            *_call("fund", "fund_account", {"amount": 5000}, {"success": True})]
    policy = RequestPolicy()
    context = _context(rows)
    proposal = policy.propose(context)
    assert proposal is not None
    assert [row["tool"] for row in _wire(proposal)["observed_operations"]] == ["fund_account"]
    verdict = policy.validate(context, proposal,
                              [_draft("place_order", {"shares": 50, "symbol": "ZETA"})],
                              draft_text="")
    assert not verdict.accepted
    assert verdict.fallback == "original"


def test_explicit_current_user_repeat_is_allowed():
    rows = [{"role": "user", "content": "Buy 50 ZETA shares."},
            *_call("order", "place_order", {"shares": 50, "symbol": "ZETA"},
                   {"success": True}),
            {"role": "user", "content": "Buy another 50 ZETA shares again."},
            *_call("quote", "get_quote", {"symbol": "ZETA"}, {"price": 4})]
    policy = RequestPolicy()
    context = _context(rows)
    proposal = policy.propose(context)
    assert proposal is not None
    assert policy.validate(context, proposal,
                           [_draft("place_order", {"shares": 50, "symbol": "ZETA"})],
                           draft_text="").accepted


def test_repeating_prior_read_is_not_blocked_as_a_side_effect():
    rows = [{"role": "user", "content": "Find the quote."},
            *_call("old", "get_quote", {"symbol": "ZETA"}, {"price": 4}),
            {"role": "user", "content": "Check the account before continuing."},
            *_call("account", "get_account", {}, {"balance": 5000})]
    policy = RequestPolicy()
    context = _context(rows)
    proposal = policy.propose(context)
    assert proposal is not None
    assert policy.validate(context, proposal,
                           [_draft("get_quote", {"symbol": "ZETA"})],
                           draft_text="").accepted


def test_exact_latest_user_literal_corrects_prior_token_and_rejects_wrong_revision():
    rows = [{"role": "user", "content": "Use access token OLD123."},
            *_call("old", "check_token", {"access_token": "OLD123"},
                   {"access_token": "OLD123"}),
            {"role": "user", "content": "Set my budget limit to $1500 using my secure token ABCDE12345."}]
    policy = BindingPolicy()
    context = _context(rows, [_draft("set_budget_limit", {
        "budget_limit": 1500, "access_token": "ABCDEF12345"})])
    proposal = policy.propose(context)
    assert proposal is not None
    assert proposal.receipt["explicit_conflict_count"] == 1
    assert not policy.validate(context, proposal, [_draft("set_budget_limit", {
        "budget_limit": 1500, "access_token": "ABCDEF12345"})], draft_text="").accepted
    assert policy.validate(context, proposal, [_draft("set_budget_limit", {
        "budget_limit": 1500, "access_token": "ABCDE12345"})], draft_text="").accepted


def test_equal_name_field_on_unrelated_entity_does_not_bind():
    rows = [{"role": "user", "content": "Find order details."},
            *_call("order", "get_order", {"query": "open"}, {"id": "ID-12345"}),
            {"role": "user", "content": "Delete the selected message."}]
    context = _context(rows, [_draft("delete_message", {"id": "ID-12345"})])
    assert BindingPolicy().propose(context) is None


def test_unrelated_producer_with_matching_specific_field_does_not_lock_draft():
    rows = [{"role": "user", "content": "Find order details."},
            *_call("order", "get_order", {"query": "open"},
                   {"message_id": "MSG-12345"}),
            {"role": "user", "content": "Delete the selected message."}]
    context = _context(rows, [_draft("delete_message", {"message_id": "MSG-12345"})])
    assert BindingPolicy().propose(context) is None


def test_unassigned_prose_is_not_treated_as_an_exact_literal():
    rows = [{"role": "user", "content": "Check my account balance before proceeding."}]
    context = _context(rows, [_draft("get_balance", {"balance": "before"})])
    assert BindingPolicy().propose(context) is None


def test_output_binding_abstains_across_distinct_entity_queries():
    rows = [{"role": "user", "content": "Find both messages."},
            *_call("first", "get_message", {"query": "first"},
                   {"message_id": "MSG-12345"}),
            *_call("second", "get_message", {"query": "second"},
                   {"message_id": "MSG-12345"}),
            {"role": "user", "content": "Delete the selected message."}]
    context = _context(rows, [_draft("delete_message", {"message_id": "MSG-12345"})])
    assert BindingPolicy().propose(context) is None


def test_generic_amount_result_does_not_lock_revised_calculation():
    rows = [{"role": "user", "content": "Check the account."},
            *_call("balance", "get_account", {}, {"amount": "100"}),
            {"role": "user", "content": "Transfer a calculated amount."}]
    context = _context(rows, [_draft("transfer_account", {"amount": "100"})])
    assert BindingPolicy().propose(context) is None


def test_grounded_held_field_survives_revision():
    rows = [{"role": "user", "content": "Delete the selected message."},
            *_call("lookup", "get_message", {"query": "latest"},
                   {"message_id": "MSG-67410"})]
    policy = BindingPolicy()
    context = _context(rows, [_draft("delete_message", {
        "message_id": "MSG-67410", "force": False})])
    proposal = policy.propose(context)
    assert proposal is not None
    assert proposal.receipt["grounded_held_field_count"] == 1
    assert _wire(proposal)["source_results"][0]["observation_version"] == 1
    assert not policy.validate(context, proposal, [_draft("delete_message", {
        "force": True})], draft_text="").accepted


def test_calculated_values_are_reviewed_without_forcing_numeric_equality():
    rows = [{"role": "user", "content": "Count lines, words and characters."},
            *_call("wc", "count_file", {"path": "data.csv"},
                   {"lines": 3, "words": 16, "characters": 60}),
            {"role": "user", "content": "Compute the average of the three numerical values obtained."}]
    policy = BindingPolicy()
    context = _context(rows, [_draft("calculate", {"values": [3, 16, 60]})])
    proposal = policy.propose(context)
    assert proposal is not None
    assert proposal.receipt["arithmetic_review"]
    assert policy.validate(context, proposal, [_draft("calculate", {
        "values": [6, 32, 120]})], draft_text="").accepted


def test_multi_call_dependency_review_does_not_force_unrelated_serialization():
    rows = [{"role": "user", "content": "Find the airport, then book the flight."}]
    policy = BindingPolicy()
    context = _context(rows, [
        _draft("get_nearest_airport", {"city": "Rivermist"}),
        _draft("book_flight", {"origin": "RIV"}),
    ])
    proposal = policy.propose(context)
    assert proposal is not None
    assert proposal.receipt["same_batch_dependency_review"]
    assert policy.validate(context, proposal, list(context.draft_tool_calls),
                           draft_text="").accepted
    unrelated = _context(rows, [
        _draft("write_note", {"text": "hello"}),
        _draft("send_message", {"text": "world"}),
    ])
    assert policy.propose(unrelated) is None
