"""CPU tests for the narrow, source-backed Static-ActionLedger policy."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.candidate_algorithms.action_ledger import (  # noqa: E402
    ACTION_LEDGER_VERSION,
    ACTION_RULES_VERSION,
    Policy,
)
from benchmarks.memory_runtime.candidate_algorithms.repair_protocol import RepairContext  # noqa: E402
from history_memory.events import EventStore  # noqa: E402


def _call(name, arguments, call_id="draft", **transport):
    return {**transport, "id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


def _pair(name, arguments, result, call_id="observed"):
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [_call(name, arguments, call_id)]},
        {"role": "tool", "tool_call_id": call_id,
         "content": json.dumps(result)},
    ]


def _tool(name, properties, required):
    return {"type": "function", "function": {
        "name": name,
        "parameters": {
            "type": "object",
            "properties": {
                field: declaration if isinstance(declaration, dict)
                else {"type": declaration}
                for field, declaration in properties.items()
            },
            "required": list(required),
            "additionalProperties": False,
        },
    }}


BOOK = _tool(
    "book_flight",
    {"travel_from": "string", "travel_to": "string"},
    ("travel_from", "travel_to"),
)
SEND = _tool(
    "send_message",
    {"receiver_id": "string", "message": "string"},
    ("receiver_id", "message"),
)
FUND = _tool("fund_account", {"amount": "number"}, ("amount",))
FUND_REAL = {"type": "function", "function": {
    "name": "fund_account", "parameters": {
        "type": "dict", "properties": {"amount": {"type": "float"}},
        "required": ["amount"],
    },
}}
BOOK_REAL = {"type": "function", "function": {
    "name": "book_flight", "parameters": {
        "type": "dict",
        "properties": {field: {"type": "string"} for field in (
            "access_token", "card_id", "travel_date", "travel_from",
            "travel_to", "travel_class",
        )},
        "required": [
            "access_token", "card_id", "travel_date", "travel_from",
            "travel_to", "travel_class",
        ],
    },
}}


def _context(rows, calls=(), *, tools=(), text="", parse_error=None):
    store = EventStore.from_messages("action-ledger-test", rows)
    prepared = SimpleNamespace(_store=store, _tools=tuple(tools))
    return RepairContext(
        prepared, tuple(calls), text, parse_error,
        lambda messages: sum(len(message["content"]) for message in messages),
        100_000,
    )


def test_versions_and_true_fare_condition_produce_one_guarded_stop_repair():
    rows = [
        {"role": "user", "content": (
            "Check the fare from SFO to LAX and, if it is under 100 USD, "
            "book the flight."
        )},
        *_pair(
            "get_flight_cost",
            {"travel_from": "SFO", "travel_to": "LAX"},
            {"fare": 72, "currency": "USD"},
            "fare",
        ),
    ]
    context = _context(rows, tools=(BOOK,))
    policy = Policy()
    assessment = policy.inspect(context)
    assert ACTION_LEDGER_VERSION == "static-action-ledger-v1"
    assert ACTION_RULES_VERSION == "action-ledger-rules-v1"
    assert assessment.receipt["status"] == "ready_unexecuted"
    assert len(assessment.ready) == 1
    ready = assessment.ready[0]
    assert ready.semantic_call() == {
        "tool": "book_flight",
        "arguments": {"travel_from": "SFO", "travel_to": "LAX"},
    }
    assert [(span.field, span.text) for span in ready.source_spans] == [
        ("travel_from", "SFO"), ("travel_to", "LAX")]
    assert any(item["kind"] == "fare_query_result" for item in ready.evidence)

    proposal = policy.propose(context)
    assert proposal is not None
    assert proposal.reason == "static_action_ledger_ready"
    assert '"tool_call_id"' not in proposal.messages[0]["content"]
    candidate = _call(
        "book_flight", {"travel_from": "SFO", "travel_to": "LAX"}, "new-id"
    )
    assert policy.validate(context, proposal, [candidate]).accepted
    wrong = _call(
        "book_flight", {"travel_from": "SFO", "travel_to": "JFK"}, "new-id"
    )
    assert not policy.validate(context, proposal, [wrong]).accepted
    assert not policy.validate(context, proposal, [candidate, candidate]).accepted


def test_false_and_currency_unknown_conditions_never_propose_booking():
    request = {"role": "user", "content": (
        "Check the fare from SFO to LAX and, if it is at most 100 USD, "
        "book the flight."
    )}
    expensive = _context([
        request,
        *_pair("get_flight_cost", {"travel_from": "SFO", "travel_to": "LAX"},
               {"fare": 140, "currency": "USD"}),
    ], tools=(BOOK,))
    assert Policy().inspect(expensive).obligations[0].state == "blocked"
    assert Policy().inspect(expensive).obligations[0].reason == "condition_false"
    assert Policy().propose(expensive) is None

    no_currency = _context([
        request,
        *_pair("get_flight_cost", {"travel_from": "SFO", "travel_to": "LAX"},
               {"fare": 72}),
    ], tools=(BOOK,))
    obligation = Policy().inspect(no_currency).obligations[0]
    assert (obligation.state, obligation.reason) == (
        "unknown", "currency_or_threshold_unknown")
    assert Policy().propose(no_currency) is None

    unsupported_condition = _context([{
        "role": "user",
        "content": "Unless the fare is high, book a flight from SFO to LAX.",
    }], tools=(BOOK,))
    assert Policy().inspect(unsupported_condition).obligations[0].state == "unknown"


def test_explicit_query_prerequisite_must_have_one_complete_matching_result():
    request = {"role": "user", "content": (
        "Find the fare from SFO to LAX and then book the flight."
    )}
    missing = _context([request], tools=(BOOK,))
    row = Policy().inspect(missing).obligations[0]
    assert (row.state, row.reason) == ("blocked", "fare_query_not_completed")

    returned = _context([
        request,
        *_pair("get_flight_cost", {"travel_from": "SFO", "travel_to": "LAX"},
               {"fare": 72, "currency": "USD"}),
    ], tools=(BOOK,))
    assert Policy().inspect(returned).obligations[0].state == "ready_unexecuted"


def test_exact_current_request_success_filters_duplicate_and_preserves_transport():
    request = {"role": "user", "content": "Book a flight from SFO to LAX."}
    rows = [
        request,
        *_pair("book_flight", {"travel_from": "SFO", "travel_to": "LAX"},
               {"success": True, "booking_id": "B-1"}, "booked"),
    ]
    duplicate = _call(
        "book_flight", {"travel_from": "SFO", "travel_to": "LAX"},
        "selected", provider_meta={"slot": 3},
    )
    unrelated = _call("get_weather", {"city": "Paris"}, "lookup", index=9)
    context = _context(rows, [duplicate, unrelated], tools=(BOOK,))
    assessment = Policy().inspect(context)
    assert assessment.obligations[0].state == "completed"
    filtered, receipt = Policy().filter_completed(context, [duplicate, unrelated])
    assert filtered == (unrelated,)
    assert filtered[0] == unrelated
    assert receipt["status"] == "filtered_completed_duplicates"
    assert receipt["removed"][0]["candidate_call_id"] == "selected"
    assert receipt["removed"][0]["success_sources"][0]["result_source_index"] == 2
    assert receipt["assessment"]["obligations"][0]["state"] == "completed"


def test_previous_turn_success_is_not_completion_for_a_new_request():
    rows = [
        {"role": "user", "content": "Book a flight from SFO to LAX."},
        *_pair("book_flight", {"travel_from": "SFO", "travel_to": "LAX"},
               {"success": True}, "old"),
        {"role": "user", "content": "Book a flight from SFO to LAX."},
    ]
    candidate = _call("book_flight", {"travel_from": "SFO", "travel_to": "LAX"})
    context = _context(rows, [candidate], tools=(BOOK,))
    assert Policy().inspect(context).obligations[0].state == "ready_unexecuted"
    filtered, receipt = Policy().filter_completed(context, [candidate])
    assert filtered == (candidate,)
    assert receipt["status"] == "no_proven_completed_duplicate"
    assert receipt["assessment"]["obligations"][0]["state"] == "ready_unexecuted"


def test_explicit_repeat_is_unknown_and_is_never_suppressed():
    rows = [
        {"role": "user", "content": (
            "Book another flight again from SFO to LAX."
        )},
        *_pair("book_flight", {"travel_from": "SFO", "travel_to": "LAX"},
               {"success": True}, "first"),
    ]
    candidate = _call("book_flight", {"travel_from": "SFO", "travel_to": "LAX"})
    context = _context(rows, [candidate], tools=(BOOK,))
    obligation = Policy().inspect(context).obligations[0]
    assert (obligation.state, obligation.reason) == (
        "unknown", "ambiguous_multiplicity")
    assert Policy().filter_completed(context, [candidate])[0] == (candidate,)

    for wording in (
        "Send a message to receiver id 'USR007' saying \"Ping\" every hour.",
        "Send a message to receiver id 'USR007' saying \"Ping\" until acknowledged.",
        "Send an additional message to receiver id 'USR007' saying \"Ping\".",
    ):
        send_args = {"receiver_id": "USR007", "message": "Ping"}
        repeated = _context([
            {"role": "user", "content": wording},
            *_pair("send_message", send_args,
                   {"sent_status": True, "message_id": 1}, "sent"),
        ], [_call("send_message", send_args)], tools=(SEND,))
        assert Policy().inspect(repeated).obligations[0].state == "unknown"
        assert Policy().filter_completed(
            repeated, [_call("send_message", send_args)])[0]


def test_failed_pending_and_ambiguous_receipts_do_not_trigger_retry_or_filter():
    request = {"role": "user", "content": "Book a flight from SFO to LAX."}
    arguments = {"travel_from": "SFO", "travel_to": "LAX"}
    candidate = _call("book_flight", arguments)

    failed = _context([
        request, *_pair("book_flight", arguments, {"success": False}, "failed"),
    ], [candidate], tools=(BOOK,))
    assert Policy().inspect(failed).obligations[0].state == "blocked"
    failed_stop = _context([
        request, *_pair("book_flight", arguments, {"success": False}, "failed"),
    ], tools=(BOOK,))
    assert Policy().propose(failed_stop) is None
    assert Policy().filter_completed(failed, [candidate])[0] == (candidate,)

    pending = _context([
        request,
        {"role": "assistant", "content": None,
         "tool_calls": [_call("book_flight", arguments, "pending")]},
    ], [candidate], tools=(BOOK,))
    assert Policy().inspect(pending).obligations[0].reason == "pending_action_attempt"
    assert Policy().filter_completed(pending, [candidate])[0] == (candidate,)

    ambiguous = _context([
        request, *_pair("book_flight", arguments, {"message": "processed"}, "amb"),
    ], [candidate], tools=(BOOK,))
    row = Policy().inspect(ambiguous).obligations[0]
    assert (row.state, row.reason) == ("unknown", "ambiguous_action_receipt")
    assert Policy().filter_completed(ambiguous, [candidate])[0] == (candidate,)


def test_no_receipt_text_action_is_an_ambiguous_attempt_not_unexecuted():
    rows = [
        {"role": "user", "content": "Book a flight from SFO to LAX."},
        {"role": "assistant", "content": (
            "[book_flight(travel_from='SFO', travel_to='LAX')]"
        )},
    ]
    context = _context(rows, tools=(BOOK,))
    row = Policy().inspect(context).obligations[0]
    assert (row.state, row.reason) == ("unknown", "ambiguous_action_attempt")
    assert Policy().propose(context) is None


def test_schema_mismatch_and_tool_prefix_without_authorization_abstain():
    extra_required = _tool(
        "book_flight",
        {"travel_from": "string", "travel_to": "string", "passenger": "string"},
        ("travel_from", "travel_to", "passenger"),
    )
    mismatch = _context([
        {"role": "user", "content": "Book a flight from SFO to LAX."},
    ], tools=(extra_required,))
    assert Policy().inspect(mismatch).obligations[0].state == "unknown"
    assert Policy().propose(mismatch) is None

    prefix_only = _context([
        {"role": "user", "content": "The string book_flight appears in this note."},
    ], tools=(BOOK,))
    assert Policy().inspect(prefix_only).obligations == ()


def test_quoted_send_binds_receiver_not_sender_and_requires_exact_candidate():
    request = {"role": "user", "content": (
        "Send a message to receiver id 'USR007' saying \"Meet at 5\"."
    )}
    context = _context([request], tools=(SEND,))
    policy = Policy()
    ready = policy.inspect(context).ready[0]
    assert ready.arguments == {"receiver_id": "USR007", "message": "Meet at 5"}
    assert {span.text for span in ready.source_spans} == {"USR007", "Meet at 5"}
    proposal = policy.propose(context)
    assert proposal is not None
    wrong_role = _call(
        "send_message", {"receiver_id": "USR005", "message": "Meet at 5"}
    )
    assert not policy.validate(context, proposal, [wrong_role]).accepted
    correct = _call(
        "send_message", {"receiver_id": "USR007", "message": "Meet at 5"}
    )
    assert policy.validate(context, proposal, [correct]).accepted

    role_ambiguous = _context([{"role": "user", "content": (
        "Send a message from sender id 'USR005' to receiver id 'USR007' "
        "saying \"Meet at 5\"."
    )}], tools=(SEND,))
    assert Policy().inspect(role_ambiguous).ready == ()


def test_ambiguous_recipient_and_incomplete_quote_are_unknown():
    for wording in (
        "Send Alex a message saying 'Hello'.",
        "Send a message to recipient named Alice saying 'Hello'.",
        "Send a message to receiver Alice saying 'Hello'.",
    ):
        ambiguous = _context([{
            "role": "user", "content": wording,
        }], tools=(SEND,))
        assert Policy().inspect(ambiguous).obligations[0].state == "unknown"

    incomplete = _context([{
        "role": "user",
        "content": "Send a message to receiver id 'USR007' saying 'Don't go'.",
    }], tools=(SEND,))
    assert Policy().inspect(incomplete).obligations[0].state == "unknown"


def test_funding_requires_unambiguous_currency_and_registered_semantics():
    ready = _context([{
        "role": "user", "content": "Top up my account with 5,000 USD.",
    }], tools=(FUND,))
    obligation = Policy().inspect(ready).ready[0]
    assert obligation.arguments == {"amount": 5000}
    assert any(item["kind"] == "registered_fixed_currency_semantics"
               for item in obligation.evidence)

    ambiguous = _context([{
        "role": "user", "content": "Top up my account with $5,000.",
    }], tools=(FUND,))
    row = Policy().inspect(ambiguous).obligations[0]
    assert (row.state, row.reason) == (
        "unknown", "unsupported_or_ambiguous_funding_grammar")


def test_real_bfcl_schema_funding_form_is_a_supported_omission_witness():
    context = _context([{
        "role": "user", "content": "Top up my trading account with 5,000 USD.",
    }], tools=(FUND_REAL,))
    policy = Policy()
    obligation = policy.inspect(context).ready[0]
    assert obligation.semantic_call() == {
        "tool": "fund_account", "arguments": {"amount": 5000},
    }
    proposal = policy.propose(context)
    assert proposal is not None
    call = _call("fund_account", {"amount": 5000}, "generated")
    assert policy.validate(context, proposal, [call]).accepted


def test_real_bfcl_booking_schema_uses_completion_only_duplicate_proof():
    request = {"role": "user", "content": (
        "Planning an incredible journey from NYC to Tokyo on December 24th 2026 "
        "in first class! I have my credit card ready with the id card_5678, which "
        "is expiring soon, and I'd like to allocate business expenses wisely. "
        "The cardholder's name matches mine, Michael Thompson, and I can provide "
        "the CVV when needed, which is 456. Just double check how much the flight "
        "costs for me and then could you make the booking using access token "
        "'abc123xyz'?"
    )}
    arguments = {
        "access_token": "abc123xyz", "card_id": "card_5678",
        "travel_date": "2026-12-24", "travel_from": "JFK",
        "travel_to": "HND", "travel_class": "first",
    }
    observed = _context([
        request,
        *_pair("book_flight", arguments, {
            "booking_id": "3426812", "transaction_id": "45451592",
            "booking_status": True, "booking_history": {},
        }, "booked"),
        *_pair("get_flight_cost", {
            "travel_from": "JFK", "travel_to": "HND",
            "travel_date": "2026-12-24", "travel_class": "first",
        }, {"travel_cost_list": [9500.0]}, "read-after-booking"),
    ], tools=(BOOK_REAL,))
    policy = Policy()
    row = policy.inspect(observed).obligations[0]
    assert (row.state, row.arguments) == ("completed", arguments)
    duplicate = _call("book_flight", arguments, "selected")
    filtered, receipt = policy.filter_completed(observed, [duplicate])
    assert filtered == ()
    assert receipt["status"] == "filtered_completed_duplicates"

    # The same broad natural-language request cannot synthesize six arguments
    # when there is no completed operation to back them.
    missing = _context([request], tools=(BOOK_REAL,))
    assert policy.inspect(missing).ready == ()
    assert policy.propose(missing) is None


def test_contradicted_success_and_later_cancellation_do_not_filter_booking():
    request = {"role": "user", "content": "Book a flight from SFO to LAX."}
    arguments = {"travel_from": "SFO", "travel_to": "LAX"}
    duplicate = _call("book_flight", arguments)
    for result in (
        {"booking_status": False, "success": True, "status": "success"},
        {"booking_status": True, "success": False},
    ):
        context = _context([
            request, *_pair("book_flight", arguments, result, "booked"),
        ], [duplicate], tools=(BOOK,))
        assert Policy().inspect(context).obligations[0].state != "completed"
        assert Policy().filter_completed(context, [duplicate])[0] == (duplicate,)

    cancelled = _context([
        request,
        *_pair("book_flight", arguments, {"booking_status": True}, "booked"),
        *_pair("cancel_booking", {"booking_id": "B-1"},
               {"cancel_status": True}, "cancelled"),
    ], [duplicate], tools=(BOOK,))
    assert Policy().inspect(cancelled).obligations[0].state != "completed"
    assert Policy().filter_completed(cancelled, [duplicate])[0] == (duplicate,)

    read_after = _context([
        request,
        *_pair("book_flight", arguments, {"booking_status": True}, "booked"),
        *_pair("get_flight_cost", arguments, {"travel_cost_list": [72.0]}, "read"),
    ], [duplicate], tools=(BOOK,))
    assert Policy().inspect(read_after).obligations[0].state == "completed"
    assert Policy().filter_completed(read_after, [duplicate])[0] == ()


def test_false_sent_status_overrides_generic_success_marker():
    request = {"role": "user", "content": (
        "Send a message to receiver id 'USR007' saying \"Meet at 5\"."
    )}
    arguments = {"receiver_id": "USR007", "message": "Meet at 5"}
    duplicate = _call("send_message", arguments)
    context = _context([
        request,
        *_pair("send_message", arguments,
               {"sent_status": False, "success": True}, "sent"),
    ], [duplicate], tools=(SEND,))
    assert Policy().inspect(context).obligations[0].state != "completed"
    assert Policy().filter_completed(context, [duplicate])[0] == (duplicate,)


def test_negation_questions_quotes_and_extra_conditions_never_become_ready():
    cases = [
        ("Do not book a flight from SFO to LAX.", BOOK),
        ("Should I book a flight from SFO to LAX?", BOOK),
        ('"Book a flight from SFO to LAX."', BOOK),
        ("Book a flight from SFO to LAX only after approval.", BOOK),
        ("Check the fare from SFO to LAX and, if it is under 100 USD, "
         "book the flight if my manager approves.", BOOK),
        ("If approval arrives, send a message to receiver id 'USR007' "
         "saying \"Meet at 5\".", SEND),
        ("If the bank confirms, top up my account with 5,000 USD.", FUND),
    ]
    for request, tool in cases:
        assessment = Policy().inspect(_context(
            [{"role": "user", "content": request}], tools=(tool,)))
        assert assessment.ready == (), request


def test_tampered_proposal_parse_error_and_extra_call_are_rejected():
    context = _context([{
        "role": "user", "content": "Book a flight from SFO to LAX.",
    }], tools=(BOOK,))
    policy = Policy()
    proposal = policy.propose(context)
    assert proposal is not None
    candidate = _call("book_flight", {"travel_from": "SFO", "travel_to": "LAX"})
    proposal.guard["obligation_id"] = "forged"
    assert not policy.validate(context, proposal, [candidate]).accepted

    fresh = policy.propose(context)
    assert fresh is not None
    assert not policy.validate(
        context, fresh, [candidate], parse_error="bad output").accepted
    assert not policy.validate(context, fresh, []).accepted
