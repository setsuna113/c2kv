"""CPU checks for prefill dependency groups and their source-bound budget."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "python")]

from benchmarks.memory_runtime.candidate_algorithms.dependencies import (
    build_dependency_workspace,
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


def _tool(name, slot):
    return {"type": "function", "function": {
        "name": name,
        "parameters": {"type": "object", "properties": {slot: {"type": "string"}}},
    }}


def _wire(messages):
    assert len(messages) == 1 and messages[0]["role"] == "user"
    return json.loads(messages[0]["content"].split("\n", 1)[1])


def _chars(messages):
    return len(messages[0]["content"]) if messages else 0


def test_selects_cross_event_typed_dependency_for_current_goal():
    store = EventStore.from_messages("dependencies", [
        {"role": "user", "content": "Manage orders and invoices."},
        *_call("order-source", "get_order", {"customer": "Ada"},
               {"order_id": "ORD-1234", "status": "ready"}),
        *_call("order-use", "submit_order", {"order_id": "ORD-1234"},
               {"submitted": True}),
        *_call("invoice-source", "get_invoice", {"customer": "Ada"},
               {"invoice_id": "INV-5678", "status": "open"}),
        *_call("invoice-use", "pay_invoice", {"invoice_id": "INV-5678"},
               {"paid": True}),
        {"role": "user", "content": "Finish paying invoice INV-5678."},
    ])
    tools = [_tool("submit_order", "order_id"), _tool("pay_invoice", "invoice_id")]
    messages, receipt = build_dependency_workspace(
        store, tools, token_counter=_chars, token_budget=100_000,
    )
    groups = _wire(messages)["groups"]
    invoice = next(group for group in groups
                   if group["producer"]["tool"] == "get_invoice"
                   and "historical_consumer" in group)
    assert invoice["producer"]["src"]["result_source_index"] == 6
    assert invoice["result_fields"] == [
        {"path": ["invoice_id"], "typed_value": "INV-5678"}
    ]
    assert invoice["historical_consumer"]["tool"] == "pay_invoice"
    assert invoice["historical_consumer"]["matches"][0]["relation"] == "observed_typed_equal"
    assert receipt["query_source_indices"] == [0, 9, 8]
    assert 6 in receipt["source_indices"] and 7 in receipt["source_indices"]


def test_repeated_field_name_does_not_overwrite_older_source_or_version():
    store = EventStore.from_messages("versions", [
        {"role": "user", "content": "Submit order ORD-1234."},
        *_call("lookup-1", "get_order", {"customer": "Ada"},
               {"order_id": "ORD-1234"}),
        *_call("consume", "submit_order", {"order_id": "ORD-1234"},
               {"submitted": True}),
        *_call("lookup-2", "get_order", {"customer": "Ada"},
               {"order_id": "ORD-9876"}),
    ])
    messages, _ = build_dependency_workspace(
        store, [_tool("submit_order", "order_id")],
        token_counter=_chars, token_budget=100_000,
    )
    relation = next(group for group in _wire(messages)["groups"]
                    if "historical_consumer" in group)
    assert relation["producer"]["src"]["result_source_index"] == 2
    assert relation["producer"]["observation_version"] == 1
    assert relation["result_fields"][0]["typed_value"] == "ORD-1234"
    assert relation["historical_consumer"]["src"]["call_source_index"] == 3


def test_array_entities_with_same_field_are_separate_atomic_groups():
    store = EventStore.from_messages("entities", [
        {"role": "user", "content": "Submit the correct order."},
        *_call("lookup", "list_orders", {"customer": "Ada"}, {"orders": [
            {"order_id": "ORD-1A2B"}, {"order_id": "ORD-3C4D"},
        ]}),
        *_call("submit", "submit_order", {"order_id": "ORD-1A2B"},
               {"submitted": True}),
    ])
    messages, receipt = build_dependency_workspace(
        store, [_tool("submit_order", "order_id")],
        token_counter=_chars, token_budget=100_000,
    )
    producer_groups = [group for group in _wire(messages)["groups"]
                       if group["producer"]["tool"] == "list_orders"]
    assert {tuple(group["producer"]["entity_path"]) for group in producer_groups} == {
        ("orders", 0), ("orders", 1),
    }
    assert all(len(group["result_fields"]) == 1 for group in producer_groups)
    linked = next(group for group in producer_groups if "historical_consumer" in group)
    assert linked["result_fields"][0]["typed_value"] == "ORD-1A2B"
    assert any(row["entity_path"] == ["orders", 1]
               for row in receipt["selected_groups"])


def test_whole_group_admission_and_no_unsupported_match():
    store = EventStore.from_messages("budget", [
        {"role": "user", "content": "Pay invoice INV-5678 and submit order ORD-1234."},
        *_call("order-source", "get_order", {"customer": "Ada"},
               {"order_id": "ORD-1234"}),
        *_call("order-use", "submit_order", {"order_id": "ORD-1234"},
               {"submitted": True}),
        *_call("invoice-source", "get_invoice", {"customer": "Ada"},
               {"invoice_id": "INV-5678"}),
        *_call("invoice-use", "pay_invoice", {"invoice_id": "INV-5678"},
               {"paid": True}),
    ])
    tools = [_tool("submit_order", "order_id"), _tool("pay_invoice", "invoice_id")]
    full, full_receipt = build_dependency_workspace(
        store, tools, token_counter=_chars, token_budget=100_000,
    )
    assert len(full_receipt["selected_groups"]) >= 2
    first = _wire(full)["groups"][0]
    single_content = full[0]["content"].split("\n", 1)[0] + "\n" + json.dumps(
        {"v": "dependency-workspace-v1", "groups": [first]},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )
    limited, receipt = build_dependency_workspace(
        store, tools, token_counter=_chars, token_budget=len(single_content),
    )
    assert _wire(limited)["groups"] == [first]
    assert receipt["packet_tokens"] == len(single_content)
    assert any(row["reason"] == "whole_group_over_budget"
               for row in receipt["omitted_groups"])
    empty, empty_receipt = build_dependency_workspace(
        store, tools, token_counter=_chars, token_budget=0,
    )
    assert empty == () and empty_receipt["status"] == "nothing_fits_budget"

    unrelated = EventStore.from_messages("unrelated", [
        {"role": "user", "content": "Find a customer."},
        *_call("lookup", "search", {"query": "Ada"}, {"status": "ready"}),
        *_call("next", "search", {"query": "ready"}, {"matches": []}),
    ])
    no_link, no_link_receipt = build_dependency_workspace(
        unrelated, [_tool("search", "query")],
        token_counter=_chars, token_budget=100_000,
    )
    assert no_link == ()
    assert no_link_receipt["status"] == "no_supported_dependencies"


def test_rejects_invalid_counter_and_budget():
    store = EventStore.from_messages("empty", [{"role": "user", "content": "Hello"}])
    with pytest.raises(ValueError, match="token_budget"):
        build_dependency_workspace(store, [], token_counter=_chars, token_budget=-1)
