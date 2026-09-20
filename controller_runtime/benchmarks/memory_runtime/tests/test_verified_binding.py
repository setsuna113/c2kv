"""Proof and negative-control tests for deterministic verified bindings."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.candidate_algorithms.repair_protocol import RepairContext  # noqa: E402
from benchmarks.memory_runtime.candidate_algorithms.verified_binding import (  # noqa: E402
    PROOF_REGISTRY_VERSION, Policy,
)
from history_memory.events import EventStore  # noqa: E402


def _call(name, arguments, call_id="draft"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


def _tools(*declarations):
    return [{"type": "function", "function": {
        "name": name, "parameters": {"type": "object", "properties": {
            field: {"type": kind} for field, kind in fields.items()}}}}
        for name, fields in declarations]


def _observed(name, arguments, result, call_id="observed"):
    return [{"role": "assistant", "content": "", "tool_calls": [
        _call(name, arguments, call_id)]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)}]


def _context(rows, calls, tools=(), parse_error=None):
    store = EventStore.from_messages("verified-binding-test", rows)
    prepared = SimpleNamespace(_store=store, _tools=tools)
    return RepairContext(prepared, tuple(calls), "", parse_error,
                         lambda messages: len(messages), 100_000)


def _args(call):
    return json.loads(call["function"]["arguments"])


def test_secure_token_and_access_code_have_typed_literal_proofs():
    tools = _tools(("set_budget_limit", {"access_token": "string", "budget_limit": "number"}),
                   ("book_flight", {"access_token": "string", "card_id": "string"}))
    cases = [
        ("I got some substantial savings from the last trip! Now, let's set a budget limit of $1500 for my future travel planning using my secure token ABCDE12345.",
         _call("set_budget_limit", {"access_token": "ABCDEFGH12345", "budget_limit": 1500}),
         "ABCDE12345"),
        ("I need to arrange a business class flight for Robert Trenton from San Francisco to Los Angeles on November 25th 2026. The reservation should be made using his travel card with id card_3487 and access code 1293.",
         _call("book_flight", {"access_token": "abc123xyz", "card_id": "card_3487"}),
         "1293"),
    ]
    for text, held, expected in cases:
        context = _context([{"role": "user", "content": text}], [held], tools)
        proposal = Policy().propose(context)
        assert proposal is not None
        assert proposal.proof_registry_version == PROOF_REGISTRY_VERSION
        assert len(proposal.proofs) == 1
        proof = proposal.proofs[0]
        assert text[slice(*proof.literal_span)] == expected
        assert proof.field_path == ("access_token",)
        corrected, receipt = Policy().apply(context, proposal, [held])
        assert receipt["status"] == "applied"
        assert _args(corrected[0])["access_token"] == expected
        assert Policy().validate(context, proposal, corrected).accepted


def test_quoted_message_preserves_punctuation_and_rejects_template():
    tools = _tools(("send_message", {"receiver_id": "string", "message": "string"}),
                   ("post_tweet", {"content": "string"}))
    request = ("I am Alice. Could you kindly help write a message 'I checked all the details. "
               "Move forward with the plan as we discussed earlier.' to John?")
    held = _call("send_message", {"receiver_id": "USR001",
                                  "message": "I checked all the details. Move forward with the plan as we discussed earlier"})
    context = _context([{"role": "user", "content": request}], [held], tools)
    proposal = Policy().propose(context)
    assert proposal is not None
    assert request[slice(*proposal.proofs[0].literal_span)] == (
        "I checked all the details. Move forward with the plan as we discussed earlier.")
    corrected, _ = Policy().apply(context, proposal, [held])
    assert _args(corrected[0]) == {"receiver_id": "USR001", "message":
                                  "I checked all the details. Move forward with the plan as we discussed earlier."}
    assert Policy().propose(_context([{"role": "user", "content": request}], corrected, tools)) is None
    template = "Post a message 'Price is ${price}.'"
    assert Policy().propose(_context([{"role": "user", "content": template}], [held], tools)) is None


def test_last_stock_uses_complete_watchlist_and_changes_only_symbol():
    tools = _tools(("get_stock_info", {"symbol": "string"}),
                   ("get_watchlist", {}))
    rows = [{"role": "user", "content": "Show my watchlist."},
            *_observed("get_watchlist", {}, {"watchlist": ["NVDA", "AAPL"]}),
            {"role": "user", "content":
             "I observed that among the stocks listed, the last one is showing promising movement. "
             "I'm contemplating acquiring 100 shares at the prevailing market rate."}]
    held = _call("get_stock_info", {"symbol": "NVDA"})
    context = _context(rows, [held], tools)
    proposal = Policy().propose(context)
    assert proposal is not None
    proof = proposal.proofs[0]
    assert proof.kind == "ordinal_list_result"
    assert proof.producer_result_path == ("watchlist", 1)
    assert proof.producer_event_id is not None
    assert proof.producer_observation_version == 1
    corrected, receipt = Policy().apply(context, proposal, [held])
    assert receipt["status"] == "applied"
    assert _args(corrected[0]) == {"symbol": "AAPL"}


def test_first_and_last_airport_use_one_observed_list():
    tools = _tools(("get_flight_cost", {"travel_from": "string", "travel_to": "string",
                                       "travel_class": "string"}),
                   ("list_all_airports", {}))
    rows = [{"role": "user", "content":
             "Calculate the cost of a business class ticket for the first airport on the available list "
             "to the last airport on the same list, on the last day of October 2026."},
            *_observed("list_all_airports", {}, "['RMS', 'PHV', 'BOS']")]
    held = _call("get_flight_cost", {"travel_from": "RMS", "travel_to": "PHV",
                                     "travel_class": "business"})
    context = _context(rows, [held], tools)
    proposal = Policy().propose(context)
    assert proposal is not None
    assert len(proposal.proofs) == 1
    assert proposal.proofs[0].producer_result_path == (2,)
    corrected, _ = Policy().apply(context, proposal, [held])
    assert _args(corrected[0]) == {"travel_from": "RMS", "travel_to": "BOS",
                                   "travel_class": "business"}


def test_negative_controls_do_not_bind_prose_amounts_cities_or_roles():
    tools = _tools(("get_flight_cost", {"travel_from": "string", "travel_to": "string"}),
                   ("get_available_stocks", {"sector": "string"}),
                   ("set_budget_limit", {"budget_limit": "number"}),
                   ("message_login", {"user_id": "string"}),
                   ("send_message", {"receiver_id": "string", "message": "string"}),
                   ("purchase_insurance", {"insurance_cost": "number"}))
    cases = [
        ("Fly from San Francisco to Los Angeles.",
         _call("get_flight_cost", {"travel_from": "SFO", "travel_to": "LAX"})),
        ("The technology sector is booming.",
         _call("get_available_stocks", {"sector": "Technology"})),
        ("Set my budget limit to 20,000 USD.",
         _call("set_budget_limit", {"budget_limit": 20000})),
        ("Log in as USR001 and notify my advisor (user id 'USR003').",
         _call("message_login", {"user_id": "USR001"})),
        ("Send David a message; my user id is 'USR005'.",
         _call("send_message", {"receiver_id": "USR007", "message": "Hello"})),
        ("Purchase insurance offering up to $500 coverage.",
         _call("purchase_insurance", {"insurance_cost": 20})),
    ]
    for request, held in cases:
        assert Policy().propose(_context([{"role": "user", "content": request}],
                                         [held], tools)) is None


def test_proof_tampering_stale_prefix_and_unproved_changes_are_rejected():
    tools = _tools(("set_budget_limit", {"access_token": "string", "budget_limit": "number"}))
    request = {"role": "user", "content": "Set my budget limit using secure token ABCDE12345."}
    held = _call("set_budget_limit", {"access_token": "WRONG1", "budget_limit": 1500})
    context = _context([request], [held], tools)
    policy = Policy()
    proposal = policy.propose(context)
    assert proposal is not None
    forged = replace(proposal, proofs=(replace(proposal.proofs[0], after="MALICIOUS"),))
    assert not policy.validate(context, forged, [held]).accepted
    refused, receipt = policy.apply(context, forged, [held])
    assert receipt["status"] == "refused" and refused == (held,)
    shifted_span = replace(proposal, proofs=(replace(proposal.proofs[0],
                                                     literal_span=(0, 3)),))
    assert not policy.validate(context, shifted_span, [held]).accepted
    corrected, _ = policy.apply(context, proposal, [held])
    changed = _call("set_budget_limit", {"access_token": "ABCDE12345", "budget_limit": 20})
    assert not policy.validate(context, proposal, [changed]).accepted
    assert not policy.validate(context, proposal, [*corrected, held]).accepted
    assert not policy.validate(context, proposal,
                               [_call("book_flight", _args(corrected[0]))]).accepted
    changed_id = [{**corrected[0], "id": "replacement-id"}]
    assert not policy.validate(context, proposal, changed_id).accepted
    assert policy.apply(context, proposal, [{**held, "id": "replacement-id"}])[1]["status"] == "refused"
    assert not policy.apply(context, proposal, [changed])[1]["status"] == "applied"
    later = _context([request, {"role": "user", "content": "Use secure token NEW12345."}],
                     [held], tools)
    assert not policy.validate(later, proposal, corrected).accepted


def test_ambiguous_credentials_malformed_calls_and_missing_schema_abstain():
    tools = _tools(("book_flight", {"access_token": "string"}))
    held = _call("book_flight", {"access_token": "OLD123"})
    request = "Use secure token NEW123 and access code OTHER456."
    assert Policy().propose(_context([{"role": "user", "content": request}], [held], tools)) is None
    assert Policy().propose(_context([{"role": "user", "content":
                                      "Use secure token NEW123."}], [held])) is None
    malformed = {"id": "bad", "type": "function", "function": {
        "name": "book_flight", "arguments": '{"access_token":"ONE","access_token":"TWO"}'}}
    assert Policy().propose(_context([{"role": "user", "content": request}],
                                      [malformed], tools)) is None


def test_credential_label_inside_message_and_wrong_card_identity_abstain():
    tools = _tools(("set_budget_limit", {"access_token": "string", "budget_limit": "number"}),
                   ("book_flight", {"access_token": "string", "card_id": "string"}))
    quoted = "Set the budget limit and send a message 'access token ABC123 is available'."
    held = _call("set_budget_limit", {"access_token": "OLD123", "budget_limit": 1500})
    assert Policy().propose(_context([{"role": "user", "content": quoted}], [held], tools)) is None
    booking = ("Book a flight using card with id card_3487 and access code 1293.")
    wrong = _call("book_flight", {"access_token": "OLD123", "card_id": "card_1"})
    assert Policy().propose(_context([{"role": "user", "content": booking}], [wrong], tools)) is None
    other_action = ("Set my budget limit now. Book a flight using card with id card_3487 "
                    "and access code 1293.")
    assert Policy().propose(_context([{"role": "user", "content": other_action}],
                                      [held], tools)) is None


def test_ordinal_ambiguity_or_intervening_mutation_abstains():
    tools = _tools(("get_stock_info", {"symbol": "string"}), ("get_watchlist", {}))
    held = _call("get_stock_info", {"symbol": "NVDA"})
    prefix = [{"role": "user", "content": "Show my watchlist."},
              *_observed("get_watchlist", {}, {"watchlist": ["NVDA", "AAPL"]}, "first")]
    request = {"role": "user", "content": "Buy the last stock on the watchlist."}
    duplicate = [*prefix, *_observed("get_watchlist", {},
                                    {"watchlist": ["NVDA", "MSFT"]}, "second"), request]
    assert Policy().propose(_context(duplicate, [held], tools)) is None
    mutation = [*prefix, *_observed("add_to_watchlist", {"symbol": "MSFT"},
                                   {"success": True}, "mutate"), request]
    assert Policy().propose(_context(mutation, [held], tools)) is None
    not_last = [*prefix, {"role": "user", "content":
                          "Among the stocks listed, choose the first, not last one."}]
    assert Policy().propose(_context(not_last, [held], tools)) is None
    other_list = [*prefix, *_observed("get_available_stocks", {},
                                     {"stocks": ["GOOG", "MSFT"]}, "other"), request]
    assert Policy().propose(_context(other_list, [held], tools)) is None


def test_ordinal_proof_refuses_tampered_list_path_and_new_source_version():
    tools = _tools(("get_stock_info", {"symbol": "string"}), ("get_watchlist", {}))
    rows = [{"role": "user", "content": "Show the watchlist."},
            *_observed("get_watchlist", {}, {"watchlist": ["NVDA", "AAPL"]}),
            {"role": "user", "content": "Among the stocks listed, the last one looks good."}]
    held = _call("get_stock_info", {"symbol": "NVDA"})
    context = _context(rows, [held], tools)
    proposal = Policy().propose(context)
    assert proposal is not None
    forged = replace(proposal, proofs=(replace(proposal.proofs[0],
                                               producer_result_path=("watchlist", 0)),))
    assert not Policy().validate(context, forged, [held]).accepted
    newer = _context([*rows, *_observed("get_watchlist", {},
                                        {"watchlist": ["NVDA", "MSFT"]}, "newer")],
                     [held], tools)
    assert not Policy().validate(newer, proposal,
                                 [_call("get_stock_info", {"symbol": "AAPL"})]).accepted


def test_declared_enum_excludes_unavailable_source_value():
    tools = _tools(("get_stock_info", {"symbol": "string"}), ("get_watchlist", {}))
    tools[0]["function"]["parameters"]["properties"]["symbol"]["enum"] = ["NVDA"]
    rows = [{"role": "user", "content": "Show my watchlist."},
            *_observed("get_watchlist", {}, {"watchlist": ["NVDA", "AAPL"]}),
            {"role": "user", "content": "Among the stocks listed, the last one looks good."}]
    assert Policy().propose(_context(rows, [_call("get_stock_info", {"symbol": "NVDA"})],
                                      tools)) is None


def test_incomplete_quote_spans_and_card_ids_inside_message_abstain():
    tools = _tools(("send_message", {"message": "string"}),
                   ("book_flight", {"access_token": "string", "card_id": "string"}))
    for request in ("Send the message 'Don't go.'", 'Send the message "Say \\"hello\\"."'):
        held = _call("send_message", {"message": "Keep original"})
        assert Policy().propose(_context([{"role": "user", "content": request}],
                                          [held], tools)) is None
    request = "Book a flight, send message 'card with id card_3487', and use access code 1293."
    held = _call("book_flight", {"access_token": "OLD123", "card_id": "card_3487"})
    assert Policy().propose(_context([{"role": "user", "content": request}],
                                      [held], tools)) is None
