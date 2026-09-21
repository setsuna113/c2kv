"""Regression tests for source-proven relation repairs."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.candidate_algorithms.relational_binding import (  # noqa: E402
    PROOF_REGISTRY_VERSION,
    Policy,
)
from benchmarks.memory_runtime.candidate_algorithms.repair_protocol import RepairContext  # noqa: E402
from history_memory.events import EventStore  # noqa: E402


def _call(name, arguments, call_id="draft", **extra):
    return {"id": call_id, "type": "function", **extra, "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}


def _tools(*declarations):
    output = []
    for name, fields in declarations:
        properties = {}
        for field, kind in fields.items():
            properties[field] = ({"type": "array", "items": {"type": "number"}}
                                 if kind == "number_array" else {"type": kind})
        output.append({"type": "function", "function": {
            "name": name,
            "parameters": {"type": "object", "properties": properties},
        }})
    return output


def _observed(name, arguments, result, call_id):
    return [
        {"role": "assistant", "content": "", "tool_calls": [
            _call(name, arguments, call_id),
        ]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)},
    ]


def _pending(name, arguments, call_id="pending"):
    return {"role": "assistant", "content": "", "tool_calls": [
        _call(name, arguments, call_id),
    ]}


def _context(rows, calls, tools=(), parse_error=None):
    store = EventStore.from_messages("relation-test", rows)
    prepared = SimpleNamespace(_store=store, _tools=tools)
    return RepairContext(prepared, tuple(calls), "", parse_error,
                         lambda messages: len(messages), 100_000)


def _args(call):
    value = call["function"]["arguments"]
    return json.loads(value) if isinstance(value, str) else value


def _mean_tools():
    return _tools(("wc", {"file_name": "string", "mode": "string"}),
                  ("mean", {"numbers": "number_array"}),
                  ("lookup", {"key": "string"}),
                  ("echo", {"file_name": "string", "content": "string"}))


def _mean_rows(counts=(3, 16, 60), *, duplicates=True):
    rows = [
        {"role": "user", "content":
         "Use the file named 'DataSet1.csv' for this analysis."},
        {"role": "user", "content":
             "Provide a summary of the lines, words, and characters in the previous file."}]
    for mode, count, label in zip("lwc", counts, ("lines", "words", "characters")):
        rows.extend(_observed("wc", {"file_name": "DataSet1.csv", "mode": mode},
                              {"count": count, "type": label}, f"wc-{mode}"))
        if duplicates and mode in "wc":
            rows.extend(_observed("wc", {"file_name": "DataSet1.csv", "mode": mode},
                                  {"count": count, "type": label}, f"wc-{mode}-repeat"))
    rows.append({"role": "user", "content":
                 "Could you compute the average of the three numerical value obtained? "
                 "Just for my personal use."})
    return rows


def test_real_mean_request_deduplicates_wc_and_preserves_other_call_transport():
    lookup = _call("lookup", {"key": "keep"}, "first", provider="native")
    held = _call("mean", {"numbers": [5, 10, 7]}, "mean", provider="native")
    context = _context(_mean_rows(), [lookup, held], _mean_tools())
    proposal = Policy().propose(context)
    assert proposal is not None
    assert proposal.proof_registry_version == PROOF_REGISTRY_VERSION
    assert proposal.prefix_source_index == len(context.prepared._store.messages) - 1
    assert len(proposal.proofs[0].producers) == 5
    corrected, receipt = Policy().apply(context, proposal, [lookup, held])
    assert receipt["status"] == "applied"
    assert corrected[0] == lookup
    assert corrected[1]["id"] == held["id"]
    assert corrected[1]["provider"] == "native"
    assert isinstance(corrected[1]["function"]["arguments"], str)
    assert _args(corrected[1]) == {"numbers": [3, 16, 60]}
    assert Policy().validate(context, proposal, corrected).accepted


def test_equal_wc_values_are_valid_but_conflicts_and_file_mutations_abstain():
    held = _call("mean", {"numbers": [1, 2, 3]})
    equal = _context(_mean_rows((7, 7, 7)), [held], _mean_tools())
    proposal = Policy().propose(equal)
    assert proposal is not None
    corrected, _ = Policy().apply(equal, proposal, [held])
    assert _args(corrected[0])["numbers"] == [7, 7, 7]

    conflict_rows = _mean_rows(duplicates=False)
    conflict_rows[-1:-1] = _observed(
        "wc", {"file_name": "DataSet1.csv", "mode": "w"},
        {"count": 99, "type": "words"}, "wc-conflict",
    )
    assert Policy().propose(_context(conflict_rows, [held], _mean_tools())) is None

    mutation_rows = _mean_rows(duplicates=False)
    mutation_rows[-1:-1] = _observed(
        "echo", {"file_name": "DataSet1.csv", "content": "changed"},
        None, "mutate",
    )
    assert Policy().propose(_context(mutation_rows, [held], _mean_tools())) is None

    partial_other = _mean_rows(duplicates=False)
    partial_other[-1:-1] = _observed(
        "wc", {"file_name": "Other.csv", "mode": "l"},
        {"count": 8, "type": "lines"}, "other-file",
    )
    assert Policy().propose(_context(partial_other, [held], _mean_tools())) is None

    permutation = _call("mean", {"numbers": [60, 3, 16]})
    assert Policy().propose(_context(_mean_rows(), [permutation], _mean_tools())) is None


def _message_tools():
    return _tools(("send_message", {"receiver_id": "string", "message": "string"}),
                  ("view_messages_sent", {}),
                  ("delete_message", {"receiver_id": "string"}))


def _message_rows():
    return [
        {"role": "user", "content":
         "Please send a note to my friend's user ID USR003 stating, 'Status changed.'"},
        *_observed("send_message", {"receiver_id": "USR003", "message": "Status changed."},
                   {"sent_status": True, "message_id": {"new_id": 67410},
                    "message": "Message sent to 'USR003' successfully."}, "send"),
        {"role": "user", "content": "Would you mind showing recent messages?"},
        *_observed("view_messages_sent", {}, {"messages": {"USR003": ["Status changed."]}},
                   "view"),
        {"role": "user", "content":
         "My friend mentioned something irrelevant. Can you help me delete message id: 67410?"},
    ]


def test_real_message_receipt_repairs_receiver_without_inventing_message_id():
    held = _call("delete_message", {"receiver_id": "USR001"}, provider="native")
    context = _context(_message_rows(), [held], _message_tools())
    proposal = Policy().propose(context)
    assert proposal is not None
    proof = proposal.proofs[0]
    assert proof.producers[0].producer_result_path == ("message_id", "new_id")
    corrected, _ = Policy().apply(context, proposal, [held])
    assert _args(corrected[0]) == {"receiver_id": "USR003"}
    assert corrected[0]["provider"] == "native"


@pytest.mark.parametrize("later", [
    _observed("send_message", {"receiver_id": "USR003", "message": "newer"},
              {"sent_status": True, "message_id": {"new_id": 67411}}, "later-send"),
    _observed("send_message", {"receiver_id": "USR003", "message": "unknown"},
              "malformed receipt", "later-send-malformed"),
    [_pending("send_message", {"receiver_id": "USR003", "message": "pending"})],
    _observed("delete_message", {"receiver_id": "USR003"},
              {"deleted_status": True}, "later-delete"),
])
def test_later_send_or_deletion_makes_message_target_ambiguous(later):
    rows = _message_rows()
    rows[-1:-1] = copy.deepcopy(later)
    held = _call("delete_message", {"receiver_id": "USR001"})
    assert Policy().propose(_context(rows, [held], _message_tools())) is None


def test_malformed_message_receipt_abstains_without_raising():
    rows = _message_rows()
    rows[2]["content"] = json.dumps("not a receipt")
    held = _call("delete_message", {"receiver_id": "USR001"})
    assert Policy().propose(_context(rows, [held], _message_tools())) is None


def _card_tools():
    return _tools(("get_all_credit_cards", {}),
                  ("register_credit_card", {"card_number": "string"}),
                  ("book_flight", {"access_token": "string", "card_id": "string",
                                   "travel_date": "string", "travel_from": "string",
                                   "travel_to": "string", "travel_class": "string"}))


def _card_rows(cards=None):
    cards = cards or {"main_card": {
        "card_number": "1234-5678-9876-5432", "expiry_date": "12/26",
        "cardholder_name": "Michael Zhang", "balance": 50000.0,
    }}
    return [
        {"role": "user", "content":
         "Book a business class ticket from LAX to JFK and charge it to my main credit card."},
        *_observed("get_all_credit_cards", {}, {"credit_card_list": cards}, "cards"),
    ]


def _book(card_id="1234-5678-9876-5432"):
    return _call("book_flight", {
        "access_token": "abc123xyz", "card_id": card_id,
        "travel_date": "2026-10-12", "travel_from": "LAX", "travel_to": "JFK",
        "travel_class": "business",
    }, provider="native")


def test_real_card_snapshot_maps_number_to_authorized_role_key():
    held = _book()
    context = _context(_card_rows(), [held], _card_tools())
    proposal = Policy().propose(context)
    assert proposal is not None
    proof = proposal.proofs[0]
    assert proof.producers[0].producer_result_path == (
        "credit_card_list", "main_card", "card_number")
    corrected, _ = Policy().apply(context, proposal, [held])
    assert _args(corrected[0]) == {
        **_args(held), "card_id": "main_card",
    }
    assert corrected[0]["provider"] == "native"


def test_card_rule_rejects_valid_key_duplicate_number_and_later_mutation():
    assert Policy().propose(_context(_card_rows(), [_book("main_card")], _card_tools())) is None
    duplicate = {
        "main_card": {"card_number": "1234-5678-9876-5432"},
        "backup_card": {"card_number": "1234-5678-9876-5432"},
    }
    assert Policy().propose(_context(_card_rows(duplicate), [_book()], _card_tools())) is None
    mutated = _card_rows()
    mutated.extend(_observed("register_credit_card", {"card_number": "9999"},
                               {"card_id": "new_card"}, "register"))
    assert Policy().propose(_context(mutated, [_book()], _card_tools())) is None


def test_same_batch_source_mutations_abstain():
    mean_batch = [
        _call("echo", {"file_name": "DataSet1.csv", "content": "changed"}, "write"),
        _call("mean", {"numbers": [5, 10, 7]}, "mean"),
    ]
    assert Policy().propose(_context(_mean_rows(), mean_batch, _mean_tools())) is None

    message_batch = [
        _call("send_message", {"receiver_id": "USR003", "message": "newer"}, "new-send"),
        _call("delete_message", {"receiver_id": "USR001"}, "delete"),
    ]
    assert Policy().propose(_context(
        _message_rows(), message_batch, _message_tools(),
    )) is None

    card_batch = [
        _call("register_credit_card", {"card_number": "9999"}, "register"),
        _book(),
    ]
    assert Policy().propose(_context(_card_rows(), card_batch, _card_tools())) is None


@pytest.mark.parametrize("rows,held,tools", [
    ([*_mean_rows()[:-1], {"role": "user", "content":
       "Do not compute the average of the three numerical values obtained."}],
     _call("mean", {"numbers": [5, 10, 7]}), _mean_tools()),
    ([*_message_rows()[:-1], {"role": "user", "content":
       "Do not delete message id: 67410."}],
     _call("delete_message", {"receiver_id": "USR001"}), _message_tools()),
    ([{"role": "user", "content":
       "Do not book a ticket with my main credit card."},
      *_observed("get_all_credit_cards", {}, {"credit_card_list": {"main_card": {
          "card_number": "1234-5678-9876-5432"}}}, "cards-negated")],
     _book(), _card_tools()),
])
def test_negated_requests_abstain(rows, held, tools):
    assert Policy().propose(_context(rows, [held], tools)) is None


def test_stale_tampered_multi_and_unproved_changes_are_rejected():
    held = _book()
    context = _context(_card_rows(), [held], _card_tools())
    policy = Policy()
    proposal = policy.propose(context)
    assert proposal is not None
    forged = replace(proposal, proofs=(replace(proposal.proofs[0], after="backup_card"),))
    assert not policy.validate(context, forged, [held]).accepted
    corrected, _ = policy.apply(context, proposal, [held])
    changed = copy.deepcopy(corrected)
    arguments = _args(changed[0])
    arguments["travel_to"] = "SFO"
    changed[0]["function"]["arguments"] = json.dumps(arguments)
    assert not policy.validate(context, proposal, changed).accepted
    changed_id = copy.deepcopy(corrected)
    changed_id[0]["id"] = "other"
    assert not policy.validate(context, proposal, changed_id).accepted
    assert not policy.validate(context, proposal, [*corrected, held]).accepted

    later = _context([*_card_rows(), {"role": "assistant", "content": "later"}],
                     [held], _card_tools())
    assert not policy.validate(later, proposal, corrected).accepted
    refused, receipt = policy.apply(later, proposal, [held])
    assert refused == (held,) and receipt["status"] == "refused"

    two_books = [_book(), _call("book_flight", _args(held), "second")]
    assert Policy().propose(_context(_card_rows(), two_books, _card_tools())) is None
