"""RACER draft capacity with the CommitKV window the draft itself opens.

SZ BFCL Long b256 (racer_v2 CommitKV c1_v2_verified): each draft that brought
a new tool result protected 27-32 pending tokens, while the planner and the
engine assumed the 0 left open by the previous request and admitted evidence
that left only 25.  The engine now reports the window a new tool event opens.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.sglang_generator import SGLangEventNativeError
from history_memory.source_packing import SourceMemoryView

from benchmarks.memory_runtime.backend_capacity import BackendCapacityConstraints, capacity_scope
from benchmarks.memory_runtime.racer.generator import PersistentRacerGenerator
from benchmarks.memory_runtime.tests.test_shared_capacity_protection import (
    BUDGET, FULL_PROTECTION_TOKENS, PENDING_FLOOR, SMALL_PROTECTION_TOKENS, _gate,
)
from benchmarks.memory_runtime.tests.test_source_allocation import request


def _transition(tokens, sources):
    return {"event_message_indices": [0, 1, 2],
            "tool_event": {"tokens": tokens, "positions": list(range(100, 100 + tokens)),
                           "source_message_indices": list(sources)}}


def _generator(current_transition, held_transition=None):
    transport = PersistentRacerGenerator(
        SimpleNamespace(), None, SimpleNamespace(history_budget_tokens=256), backend=SimpleNamespace())
    transport._logical_session_id = "session-a"
    transport._session_id = "racer-session-a"
    transport._decision_key = "decision-1"
    transport._source = [{"role": "user"}, {"role": "assistant"}, {"role": "tool"}]
    transport._source_positions = [0, 1, 2]
    transaction = {"regeneration_mandatory_history": {
        "tokens": 0, "source_message_indices": [], "release": "replaced_source_message"}}
    if held_transition is not None:
        transaction["commitkv_next_transition"] = held_transition
    report = {"racer_transaction": transaction,
              "racer_current_mandatory_history": {
                  "tokens": 0, "source_message_indices": [], "release": "replaced_source_message"}}
    if current_transition is not None:
        report["racer_current_commitkv_next_transition"] = current_transition
    response = {"metadata": {"kv_memory_report": report}}
    transport._held_retention = transport._retention_receipt(response)
    transport._current_retention = transport._current_retention_receipt(response)
    return transport


def test_committed_draft_exposes_the_window_its_tool_result_opens():
    transport = _generator(_transition(30, [1, 2]))
    transport._resolution = "commit"
    constraint = transport.backend_capacity_constraints(
        session_id="session-a", decision_key="decision-2", stage="draft")
    assert constraint.mandatory_history_tokens == 0
    assert constraint.new_source_start == 3
    assert constraint.tool_event_mandatory_history_tokens == 30
    assert constraint.tool_event_mandatory_source_indices == (1, 2)
    assert constraint.provenance == "engine_current_resident_receipt"
    committed_and_result = ["others", "act", "tool", "act", "tool"]
    turn_end = ["others", "act", "tool", "others", "others"]
    assert constraint.minimum_history_tokens((), source_phases=committed_and_result) == 30
    assert constraint.minimum_history_tokens((), source_phases=turn_end) == 0
    assert constraint.minimum_history_tokens((), source_phases=committed_and_result[:3]) == 0
    # Regeneration keeps the held receipt and its release rule unchanged.
    assert transport.backend_capacity_constraints(
        session_id="session-a", decision_key="decision-1", stage="regeneration") is None


def test_discarded_draft_reads_the_held_checkpoint_receipt():
    transport = _generator(_transition(30, [1]), held_transition=_transition(0, []))
    transport._resolution = "discard"
    assert transport.backend_capacity_constraints(
        session_id="session-a", decision_key="decision-2", stage="draft") is None


def test_engines_without_the_receipt_keep_the_previous_constraints():
    transport = _generator(None)
    transport._resolution = "commit"
    assert transport.backend_capacity_constraints(
        session_id="session-a", decision_key="decision-2", stage="draft") is None
    legacy = BackendCapacityConstraints(
        session_id="s", decision_key="d", stage="draft", history_budget_tokens=256,
        mandatory_history_tokens=28, mandatory_source_indices=(3,), release="never",
        provenance="engine_current_resident_receipt")
    assert legacy.minimum_history_tokens((), source_phases=["tool", "tool"]) == 28


def test_malformed_transition_receipt_is_rejected():
    broken = _transition(3, [1])
    broken["tool_event"]["positions"] = [1]
    with pytest.raises(SGLangEventNativeError, match="current lifecycle transition"):
        _generator(broken)


def _lifecycle(payload, *, new_source_start, tool_tokens=PENDING_FLOOR):
    return BackendCapacityConstraints(
        session_id=payload["session_id"], decision_key=payload["decision_key"], stage="draft",
        history_budget_tokens=BUDGET, mandatory_history_tokens=0, mandatory_source_indices=(),
        release="never", provenance="engine_current_resident_receipt",
        new_source_start=new_source_start, tool_event_mandatory_history_tokens=tool_tokens,
        tool_event_mandatory_source_indices=(1,))


@pytest.mark.parametrize("new_source_start", [1, 2])
def test_planner_leaves_room_for_the_window_a_new_tool_result_opens(new_source_start):
    payload = request()
    ordinary = _gate("commitkv").prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    assert ordinary.memory.native_evidence_tokens == FULL_PROTECTION_TOKENS
    assert FULL_PROTECTION_TOKENS + PENDING_FLOOR > BUDGET
    with capacity_scope(_lifecycle(payload, new_source_start=new_source_start)):
        planned = _gate("commitkv").prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    assert isinstance(planned.memory.view, SourceMemoryView)
    assert planned.memory.native_evidence_tokens == SMALL_PROTECTION_TOKENS
    assert planned.memory.retained_history_min_tokens == PENDING_FLOOR
    assert planned.memory.native_evidence_tokens + PENDING_FLOOR <= BUDGET
    assert planned.metadata["backend_capacity_constraints"]["tool_event_mandatory_history_tokens"] == PENDING_FLOOR


def test_planner_is_unchanged_when_no_new_tool_result_arrives():
    payload = request()
    ordinary = _gate("commitkv").prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    with capacity_scope(_lifecycle(payload, new_source_start=3)):
        planned = _gate("commitkv").prepare(copy.deepcopy(payload), ratio=8, max_new_tokens=8)
    assert planned.memory == ordinary.memory
    assert planned.memory.native_evidence_tokens == FULL_PROTECTION_TOKENS
