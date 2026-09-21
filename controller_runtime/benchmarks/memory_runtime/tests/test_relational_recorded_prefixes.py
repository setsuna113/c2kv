"""Regression tests over the three recorded prefixes that motivated v2 relations."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.candidate_algorithms.observations import (  # noqa: E402
    operation_records,
)
from benchmarks.memory_runtime.candidate_algorithms.relational_binding import (  # noqa: E402
    PROOF_REGISTRY_VERSION,
    Policy,
)
from benchmarks.memory_runtime.candidate_algorithms.repair_protocol import (  # noqa: E402
    RepairContext,
)
from history_memory.events import EventStore  # noqa: E402


FIXTURE = Path(__file__).parent / "fixtures" / "static_verified_v2" / "recorded_prefixes.json"
EXPECTED = {
    "base_15_turn4_step0": ("numbers", [3, 16, 60]),
    "base_149_turn4_step0": ("receiver_id", "USR003"),
    "base_180_turn1_step2": ("card_id", "main_card"),
}
EXPECTED_MESSAGE_COUNTS = {
    "base_15_turn4_step0": 25,
    "base_149_turn4_step0": 29,
    "base_180_turn1_step2": 11,
}
EXPECTED_TOOLS = {
    "base_15_turn4_step0": {"wc", "mean"},
    "base_149_turn4_step0": {"send_message", "view_messages_sent", "delete_message"},
    "base_180_turn1_step2": {"get_all_credit_cards", "book_flight"},
}


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _cases():
    return {case["id"]: case for case in _fixture()["cases"]}


def _context(case, *, messages=None, calls=None):
    store = EventStore.from_messages(
        case["session_id"],
        copy.deepcopy(case["messages"] if messages is None else messages),
        benchmark="bfcl",
    )
    prepared = SimpleNamespace(_store=store, _tools=copy.deepcopy(case["tools"]))
    return RepairContext(
        prepared=prepared,
        draft_tool_calls=tuple(copy.deepcopy(
            case["draft_tool_calls"] if calls is None else calls)),
        draft_text="",
        parse_error=None,
        token_counter=lambda rows: len(rows),
        token_budget=100_000,
    )


def _arguments(call):
    value = call["function"]["arguments"]
    return json.loads(value) if isinstance(value, str) else copy.deepcopy(value)


def _with_arguments(call, arguments):
    output = copy.deepcopy(call)
    output["function"]["arguments"] = json.dumps(
        arguments, ensure_ascii=False, separators=(",", ":"))
    return output


def _observed(name, arguments, result, call_id):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(
                        arguments, ensure_ascii=False, separators=(",", ":")),
                },
            }],
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False),
        },
    ]


def _nested_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _nested_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nested_keys(item)


def test_fixture_contains_only_complete_native_prefixes_and_runtime_inputs():
    data = _fixture()
    assert data["fixture_version"] == "c2kv-recorded-prefixes-v1"
    assert set(_cases()) == set(EXPECTED)
    assert {"state_info", "scorer", "possible_answer", "gold"}.isdisjoint(
        _nested_keys(data))

    for case in data["cases"]:
        assert len(case["messages"]) == EXPECTED_MESSAGE_COUNTS[case["id"]]
        assert case["source"]["prefix_boundary"] == "before_held_decision"
        assert case["decision_key"] == (
            f"turn-{case['source']['cell']['turn']}/step-{case['source']['cell']['step']}")
        assert case["source"]["task_id"] in case["session_id"]
        assert {tool["function"]["name"] for tool in case["tools"]} == EXPECTED_TOOLS[case["id"]]

        call_ids = [
            call["id"]
            for message in case["messages"]
            for call in message.get("tool_calls", ())
        ]
        result_ids = [
            message["tool_call_id"]
            for message in case["messages"]
            if message["role"] == "tool"
        ]
        assert len(call_ids) == len(set(call_ids))
        assert sorted(call_ids) == sorted(result_ids)
        store = EventStore.from_messages(
            case["session_id"], case["messages"], benchmark="bfcl")
        assert all(event.complete for event in store.events)

    wc = [
        row.observed_result
        for row in operation_records(_context(_cases()["base_15_turn4_step0"]).prepared._store)
        if row.tool == "wc"
    ]
    assert wc == [
        {"count": 3, "type": "lines"},
        {"count": 16, "type": "words"},
        {"count": 60, "type": "characters"},
        {"count": 16, "type": "words"},
        {"count": 60, "type": "characters"},
    ]


@pytest.mark.parametrize("case_id", tuple(EXPECTED))
def test_recorded_prefix_repairs_only_the_proven_field(case_id):
    case = _cases()[case_id]
    context = _context(case)
    original = copy.deepcopy(case["draft_tool_calls"])
    before = _arguments(original[0])

    policy = Policy()
    proposal = policy.propose(context)
    assert proposal is not None
    assert proposal.proof_registry_version == PROOF_REGISTRY_VERSION
    assert PROOF_REGISTRY_VERSION == "verified-binding-relations-v2"
    corrected, receipt = policy.apply(context, proposal, original)

    assert receipt["status"] == "applied"
    assert receipt["proof_registry_version"] == PROOF_REGISTRY_VERSION
    assert policy.validate(context, proposal, corrected).accepted
    assert case["draft_tool_calls"] == original
    assert len(corrected) == len(original) == 1
    assert corrected[0]["id"] == original[0]["id"]
    assert corrected[0]["type"] == original[0]["type"]
    assert corrected[0]["function"]["name"] == original[0]["function"]["name"]

    field, expected = EXPECTED[case_id]
    after = _arguments(corrected[0])
    assert after[field] == expected
    assert [key for key in before if before[key] != after[key]] == [field]
    assert {key: value for key, value in after.items() if key != field} == {
        key: value for key, value in before.items() if key != field
    }


@pytest.mark.parametrize("case_id", tuple(EXPECTED))
def test_already_correct_recorded_calls_remain_unchanged(case_id):
    case = _cases()[case_id]
    field, expected = EXPECTED[case_id]
    arguments = _arguments(case["draft_tool_calls"][0])
    arguments[field] = expected
    correct = [_with_arguments(case["draft_tool_calls"][0], arguments)]
    assert Policy().propose(_context(case, calls=correct)) is None


def test_counterfactual_file_mutation_makes_recorded_wc_values_stale():
    case = _cases()["base_15_turn4_step0"]
    messages = copy.deepcopy(case["messages"])
    # Counterfactual: the source file changes after every recorded wc receipt.
    messages[-1:-1] = _observed(
        "echo",
        {"content": "a newer row", "file_name": "DataSet1.csv"},
        "None",
        "counterfactual-b15-late-write",
    )
    assert Policy().propose(_context(case, messages=messages)) is None


def test_counterfactual_late_send_protects_the_requested_message_id():
    case = _cases()["base_149_turn4_step0"]
    messages = copy.deepcopy(case["messages"])
    # Counterfactual: delete_message deletes the latest receiver message, and a
    # later send means receiver_id no longer denotes requested message 67410.
    messages[-1:-1] = _observed(
        "send_message",
        {"receiver_id": "USR003", "message": "A newer message."},
        {
            "sent_status": True,
            "message_id": {"new_id": 99999},
            "message": "Message sent to 'USR003' successfully.",
        },
        "counterfactual-b149-late-send",
    )
    assert Policy().propose(_context(case, messages=messages)) is None


def test_counterfactual_duplicate_card_number_is_ambiguous_and_stales_proof():
    case = _cases()["base_180_turn1_step2"]
    context = _context(case)
    proposal = Policy().propose(context)
    assert proposal is not None
    corrected, _ = Policy().apply(
        context, proposal, copy.deepcopy(case["draft_tool_calls"]))

    messages = copy.deepcopy(case["messages"])
    result = next(
        message for message in messages
        if message.get("tool_call_id") == "b180-t1-s1-c0"
    )
    observed = json.loads(result["content"])
    observed["credit_card_list"]["backup_card"] = copy.deepcopy(
        observed["credit_card_list"]["main_card"])
    result["content"] = json.dumps(observed, ensure_ascii=False)
    ambiguous = _context(case, messages=messages)

    assert Policy().propose(ambiguous) is None
    assert not Policy().validate(ambiguous, proposal, corrected).accepted
