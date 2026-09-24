"""ACEBench receipt sources reach both branches of the C1 v2 capacity gate.

Regression for the RACER v2 ``c1_v2_verified`` ACEBench cells (paper 077559f):
once the incumbent S0 branch raised ``CapacityInfeasible``, the gate prepared
the same request with its source-allocation branch, whose stock request
validator rejected the legitimate ``c2kv_ace_source`` field as privileged and
turned a per-task capacity outcome into a terminal ``runner_failed``.

``FAILED_DECISION`` copies the ACE source and the two committed actor actions of
the failed ``acebench_agent__racer_v2_c2kv_c1_v2_verified_b256`` decision
``acebench/agent_multi_step_1/attempt-0`` ``turn-0/step-2`` (box9 steps.jsonl).
The step log does not store the prompts or execution observations, so the
system/user text and the observation lists are short stand-ins with the same
roles, indices and receipt coverage.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.source_packing import SourceMemoryView

from benchmarks.memory_runtime.acebench_controls import (
    _validate_ace_request,
    build_acebench_controller,
)
from benchmarks.memory_runtime.acebench_source import (
    build_ace_event_store,
    parse_acebench_draft,
)
from benchmarks.memory_runtime.always_compress import (
    ALWAYS_COMPRESSION_POLICY,
    CapacityInfeasible,
)
from benchmarks.memory_runtime.candidate_algorithms.c1_v2 import c1_v2_fields
from benchmarks.memory_runtime.candidate_algorithms.capacity_source_gate import (
    CapacityGatedSourceAllocator,
)
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_s0_policy import (
    S0_CONFIG_DEFAULTS,
    EventNativeS0Controller,
)
from benchmarks.memory_runtime.policy import PolicyInputError
from benchmarks.memory_runtime.tests.test_candidate_allocation import (
    Tokenizer,
    packing,
    policy,
)
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk


_ACTION_0 = (
    "[login_food_platform(username='Eve', password='password123'), "
    "add_food_delivery_order(username='Eve', merchant_name='Domino\\'s', "
    "items=[{'product': 'Super Supreme Pizza', 'quantity': 1}]), "
    "add_reminder(title='Today\\'s Spending', description='Today\\'s spending () yuan', "
    "time='2024-07-15 09:30')]"
)
_ACTION_1 = "[turn_on_wifi(), " + _ACTION_0[1:]
_DECODED_0 = [
    "login_food_platform(username='Eve', password='password123')",
    "add_food_delivery_order(username='Eve', merchant_name=\"Domino's\", "
    "items=[{'product': 'Super Supreme Pizza', 'quantity': 1}])",
    "add_reminder(title=\"Today's Spending\", description=\"Today's spending () yuan\", "
    "time='2024-07-15 09:30')",
]


def _receipt(assistant_index, decoded):
    return {
        "version": "acebench-execution-receipt-v1",
        "agent_history_index": assistant_index - 1,
        "execution_message_index": assistant_index + 1,
        "decode_status": "ok",
        "decoded_calls": list(decoded),
        "executor_status": "returned",
        "executor_return_shape": "list",
        "executor_return_count": len(decoded),
    }


FAILED_DECISION = {
    "session_id": "acebench/agent_multi_step_1/attempt-0",
    "decision_key": "turn-0/step-2",
    "messages": [
        {"role": "system", "content": "Use the visible food, reminder and wifi APIs."},
        {"role": "user", "content": "Order a Super Supreme Pizza from Domino's for Eve."},
        {"role": "assistant", "content": _ACTION_0},
        {"role": "tool", "tool_call_id": "acebench-execution-2",
         "content": json.dumps(["wifi is off"] * 3)},
        {"role": "assistant", "content": _ACTION_1},
        {"role": "tool", "tool_call_id": "acebench-execution-4",
         "content": json.dumps([True] * 4)},
    ],
    "tools": [],
    "c2kv_ace_source": {
        "version": "acebench-text-actions-v1",
        "receipts": [
            _receipt(2, _DECODED_0),
            _receipt(4, ["turn_on_wifi()", *_DECODED_0]),
        ],
    },
}

# A long completed producer makes full incumbent protection infeasible while
# the shared small protection still fits, so the source branch must serve it.
_LONG_CONTENT = " ".join(["note"] * 320)
_LONG_ACTION = "[write_note(title='plan', content='" + _LONG_CONTENT + "')]"
RESCUED_DECISION = {
    "session_id": "acebench/agent_multi_step_0/attempt-0",
    "decision_key": "turn-0/step-1",
    "messages": [
        {"role": "system", "content": "Use the visible note API."},
        {"role": "user", "content": "Write the plan note."},
        {"role": "assistant", "content": _LONG_ACTION},
        {"role": "tool", "tool_call_id": "acebench-execution-2",
         "content": json.dumps([True])},
    ],
    "tools": [],
    "c2kv_ace_source": {
        "version": "acebench-text-actions-v1",
        "receipts": [
            _receipt(2, ["write_note(title='plan', content='" + _LONG_CONTENT + "')"]),
        ],
    },
}

BACKENDS = (None, "c2kv", "pyramidkv", "h2o")


def _racer_backend(name, budget, *, schema="v2"):
    config = {
        "schema": f"racer-backend-{schema}", "backend": name,
        "policy": "c1_v2_verified", "history_budget_tokens": budget,
        "allocation": "racer_s0" if schema == "v2" else "backend_native_persistent",
        "detector_calibration": (
            "reference" if name == "c2kv" else "frozen_c2kv_unvalidated_transfer"),
    }
    if schema == "v2":
        config["mode"] = "on"
    elif schema == "v4":
        config["extra_protection"] = "on"
    else:
        raise ValueError("unsupported test RACER schema")
    if name == "c2kv":
        config["backend_config"] = {"method": "c2kv"}
    return config


def _controller(monkeypatch, backend, budget, *, score=0.2, schema="v2"):
    monkeypatch.setattr(
        "benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
        lambda artifact: Risk(score))
    candidate = {"variant": "c1_v2_verified", "stable_call_ids": True,
                 "recovery_rounds_per_decision": 1, "risk_artifact": {"fixture": True},
                 "risk_threshold": 0.5, **c1_v2_fields("c1_v2_verified")}
    s0_config = {**S0_CONFIG_DEFAULTS, "candidate_algorithm": candidate}
    if backend is not None:
        s0_config["racer_backend"] = _racer_backend(backend, budget, schema=schema)
    return build_acebench_controller(
        Tokenizer(), packing={**packing(), "ratios": [8], "max_target_tokens": 64},
        policy=policy(budget), view_mode=NATIVE_S0_MODE,
        compression_policy=ALWAYS_COMPRESSION_POLICY, s0_config=s0_config)


def _find(controller, kind):
    node = controller
    while not isinstance(node, kind):
        # Follow real composition attributes; wrapper __getattr__ delegation
        # would skip the gate and land on its incumbent branch.
        node = node.__dict__.get("inner") or node.__dict__["base"]
    return node


def _incumbent_s0(gate):
    node = gate.incumbent
    while hasattr(node, "base"):
        node = node.base
    return node


def test_archived_food_order_dict_literal_is_a_complete_ace_event():
    draft = parse_acebench_draft(_ACTION_0, call_id_prefix="held")
    assert draft.status == "tool_calls"
    assert len(draft.tool_calls) == 3
    assert json.loads(draft.tool_calls[1]["function"]["arguments"])["items"] == [
        {"product": "Super Supreme Pizza", "quantity": 1}]
    payload = FAILED_DECISION
    store = build_ace_event_store(
        payload["session_id"], payload["messages"][:4],
        {"version": "acebench-text-actions-v1",
         "receipts": payload["c2kv_ace_source"]["receipts"][:1]})
    event = store.events[2]
    assert event.kind == "tool_event"
    assert event.complete is True
    assert event.source_indices == (2, 3)
    assert len(event.tool_call_ids) == 3
    assert [message.to_dict() for message in store.messages] == payload["messages"][:4]


def test_ace_dict_literal_recurses_through_lists_and_dicts():
    draft = parse_acebench_draft(
        "[send(payload={'outer': [{'inner': {'count': -1, 'ok': None}}]})]",
        call_id_prefix="held")
    assert draft.status == "tool_calls"
    assert json.loads(draft.tool_calls[0]["function"]["arguments"]) == {
        "payload": {"outer": [{"inner": {"count": -1, "ok": None}}]}}


@pytest.mark.parametrize(("action", "reason"), [
    ("[send(payload={1: 'x'})]", "dict_key"),
    ("[send(payload={'a': 1, 'a': 2})]", "duplicate_dict_key"),
    ("[send(payload={**{'a': 1}})]", "dict_unpack"),
    ("[send(payload={'a': helper()})]", "nested_call"),
    ("[send(payload={'a': 1e309})]", "non_finite_number"),
])
def test_ace_dict_literal_rejects_unsafe_values(action, reason):
    draft = parse_acebench_draft(action, call_id_prefix="held")
    assert draft.status == "malformed"
    assert draft.reason == f"unsupported_ace_grammar:{reason}"


def test_v4_ace_s0_gate_uses_the_same_dict_literal_receipt_parser(monkeypatch):
    controller = _controller(monkeypatch, "h2o", 4096, schema="v4")
    leaf = _find(controller, EventNativeS0Controller)
    assert leaf._validate_request.__func__ is _validate_ace_request
    _, _, ace_store, _, _, _ = leaf._validate_request(FAILED_DECISION, 8, 8)
    assert [(event.kind, event.complete) for event in ace_store.events[2:4]] == [
        ("tool_event", True), ("tool_event", True)]
    prepared = controller.prepare(copy.deepcopy(FAILED_DECISION), ratio=8, max_new_tokens=8)
    assert prepared.metadata["racer_backend"]["schema"] == "racer-backend-v4"
    assert prepared.metadata["native_protection_request"]["input_rewritten"] is False
    assert prepared.memory.recovery_messages == ()
    assert [(event.kind, event.complete) for event in prepared._store.events[2:4]] == [
        ("tool_event", True), ("tool_event", True)]


@pytest.mark.parametrize("receipt_change", [
    {"decode_status": "error"},
    {"decoded_calls": ["login_food_platform(username='Mallory')", *_DECODED_0[1:]]},
])
def test_v4_ace_recovery_store_keeps_unverified_execution_opaque(
        monkeypatch, receipt_change):
    payload = copy.deepcopy(FAILED_DECISION)
    payload["c2kv_ace_source"]["receipts"][0].update(receipt_change)
    controller = _controller(monkeypatch, "h2o", 4096, schema="v4")
    prepared = controller.prepare(payload, ratio=8, max_new_tokens=8)
    event = prepared._store.events[2]
    assert event.kind == "acebench_execution_opaque"
    assert event.complete is False
    assert event.source_indices == (2, 3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_ace_source_contract_is_bound_on_both_gate_branches(monkeypatch, backend):
    gate = _find(_controller(monkeypatch, backend, 256), CapacityGatedSourceAllocator)
    branches = (_incumbent_s0(gate), gate.source_allocator)
    assert branches[0] is not branches[1]
    for branch in branches:
        assert isinstance(branch, EventNativeS0Controller)
        assert branch.benchmark == "acebench"
        assert branch._validate_request.__func__ is _validate_ace_request
        assert branch._validate_request.__self__ is branch


@pytest.mark.parametrize("backend", BACKENDS)
def test_failed_ace_decision_is_a_capacity_outcome_not_a_privileged_field(
        monkeypatch, backend):
    controller = _controller(monkeypatch, backend, 24)
    payload = copy.deepcopy(FAILED_DECISION)
    # A receipt that cannot verify the historical execution remains opaque.
    # The archived dict action is now parseable, so it no longer supplies this
    # capacity-failure fixture on every backend by itself.
    payload["c2kv_ace_source"]["receipts"][0]["decode_status"] = "error"
    gate = _find(controller, CapacityGatedSourceAllocator)
    seen = []
    original = gate.source_allocator.prepare

    def observed(payload, **kwargs):
        seen.append(sorted(payload))
        return original(payload, **kwargs)

    monkeypatch.setattr(gate.source_allocator, "prepare", observed)
    # CapacityInfeasible is the per-task method outcome the API records as
    # c2kv_capacity_infeasible; PolicyInputError stopped the whole server.
    with pytest.raises(CapacityInfeasible):
        controller.prepare(payload, ratio=8, max_new_tokens=8)
    assert seen == [sorted(payload)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_ace_capacity_rescue_prepares_and_reviews_receipt_history(monkeypatch, backend):
    controller = _controller(monkeypatch, backend, 128)
    payload = copy.deepcopy(RESCUED_DECISION)
    prepared = controller.prepare(payload, ratio=8, max_new_tokens=8)
    assert payload == RESCUED_DECISION
    assert prepared.metadata["capacity_source_gate"]["phase"] == "initial"
    assert isinstance(prepared.memory.view, SourceMemoryView)
    events = {event.kind: event for event in prepared._store.events}
    assert events["tool_event"].source_indices == (2, 3)
    draft = parse_acebench_draft(_LONG_ACTION, call_id_prefix="held")
    result = controller.reconsider(prepared, list(draft.tool_calls), draft_text=_LONG_ACTION)
    assert result["regenerate"] is False
    assert result["decision"]["variant"] == "c1_v2_verified"


@pytest.mark.parametrize("field", ["target", "gold_answer", "c2kv_ace_official_task_id"])
@pytest.mark.parametrize("backend", ["c2kv", "pyramidkv"])
def test_privileged_fields_stay_forbidden_on_both_gate_branches(
        monkeypatch, backend, field):
    gate = _find(_controller(monkeypatch, backend, 24), CapacityGatedSourceAllocator)
    payload = {**copy.deepcopy(FAILED_DECISION), field: "agent_multi_step_1"}
    for branch in (gate.incumbent, gate.source_allocator):
        with pytest.raises(PolicyInputError, match="privileged request fields"):
            branch.prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
