"""Engine mandatory-history receipts exposed before RACER source admission."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.racer.generator import PersistentRacerGenerator
from benchmarks.memory_runtime.racer.allocator import PersistentMemory
from benchmarks.memory_runtime.racer.tools import _shifted_index_map
from history_memory.packing import MemoryView
from history_memory.sglang_generator import SGLangEventNativeError


def _carrier_removed_source_index(source_index):
    """Use the binder and serving helper for a carrier inserted at wire row 1."""
    engine_root = Path(os.environ.get("RACER_ENGINE_ROOT", PAPER.parent / "sglang-paper"))
    composition = (engine_root / "python" / "sglang" /
                   "srt" / "mem_cache" / "c2kv_composition.py")
    if not composition.exists():
        pytest.skip("matching SGLang composition checkout is unavailable")
    spec = importlib.util.spec_from_file_location("_racer_capacity_composition", composition)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    index_map = _shifted_index_map(3, [1])
    assert index_map == {0: 0, 1: 2, 2: 3}
    hint = {"persistent_history_session": {"recovery_append": {
        "source_message_indices": [index_map[source_index]]}},
        "history_kv_event_messages": [{"message_index": index} for index in range(4)]}
    module.remap_message_metadata(hint, [1], 4)
    return hint["persistent_history_session"]["recovery_append"]["source_message_indices"][0]


def generator():
    transport = PersistentRacerGenerator(
        SimpleNamespace(), None, SimpleNamespace(history_budget_tokens=256),
        backend=SimpleNamespace())
    transport._logical_session_id = "session-a"
    transport._session_id = "racer-session-a"
    transport._decision_key = "decision-1"
    # Receipts use the carrier-removed internal ledger frame.
    transport._source_positions = [0, 1, 2]
    response = {"metadata": {"kv_memory_report": {
        "racer_transaction": {"regeneration_mandatory_history": {
            "tokens": 28, "source_message_indices": [1],
            "release": "replaced_source_message"}},
        "racer_current_mandatory_history": {
            "tokens": 20, "source_message_indices": [2],
            "release": "replaced_source_message"}}}}
    transport._held_retention = transport._retention_receipt(response)
    transport._current_retention = transport._current_retention_receipt(response)
    return transport


def test_regeneration_uses_held_receipt_and_maps_ledger_source_indices():
    transport = generator()
    constraint = transport.backend_capacity_constraints(
        session_id="session-a", decision_key="decision-1", stage="regeneration")
    assert constraint.mandatory_history_tokens == 28
    assert constraint.mandatory_source_indices == (1,)
    assert constraint.minimum_history_tokens((1,)) == 0
    assert constraint.minimum_history_tokens((2,)) == 28
    assert constraint.provenance == "engine_held_checkpoint_receipt"


def test_next_draft_chooses_current_on_commit_and_held_on_discard():
    transport = generator()
    transport._resolution = "commit"
    committed = transport.backend_capacity_constraints(
        session_id="session-a", decision_key="decision-2", stage="draft")
    assert (committed.mandatory_history_tokens, committed.mandatory_source_indices) == (20, (2,))
    assert committed.provenance == "engine_current_resident_receipt"
    assert committed.release == "never"
    assert committed.minimum_history_tokens((2,)) == 20

    transport._resolution = "discard"
    discarded = transport.backend_capacity_constraints(
        session_id="session-a", decision_key="decision-2", stage="draft")
    assert (discarded.mandatory_history_tokens, discarded.mandatory_source_indices) == (28, (1,))
    assert discarded.provenance == "engine_held_checkpoint_receipt"
    assert discarded.release == "never"
    assert discarded.minimum_history_tokens((1,)) == 28


def test_fresh_session_and_missing_receipt_keep_backward_compatibility():
    fresh = PersistentRacerGenerator(
        SimpleNamespace(), None, SimpleNamespace(history_budget_tokens=256),
        backend=SimpleNamespace())
    assert fresh.backend_capacity_constraints(
        session_id="new", decision_key="first", stage="draft") is None
    transport = generator()
    transport._held_retention = None
    assert transport.backend_capacity_constraints(
        session_id="session-a", decision_key="decision-1", stage="regeneration") is None
    transport._resolution = "commit"
    transport._current_retention = None
    assert transport.backend_capacity_constraints(
        session_id="session-a", decision_key="decision-2", stage="draft") is None


def test_stale_regeneration_and_cross_session_receipts_are_rejected():
    transport = generator()
    with pytest.raises(SGLangEventNativeError, match="another decision"):
        transport.backend_capacity_constraints(
            session_id="session-a", decision_key="decision-2", stage="regeneration")
    with pytest.raises(SGLangEventNativeError, match="another session"):
        transport.backend_capacity_constraints(
            session_id="session-b", decision_key="decision-1", stage="regeneration")


def test_known_source_releases_mixed_pending_window_but_unmapped_action_does_not():
    transport = generator()
    response = {"metadata": {"kv_memory_report": {
        "racer_transaction": {"regeneration_mandatory_history": {
            "tokens": 28, "source_message_indices": [1, 3],
            "release": "replaced_source_message"}}}}}
    transport._held_retention = transport._retention_receipt(response)
    constraint = transport.backend_capacity_constraints(
        session_id="session-a", decision_key="decision-1", stage="regeneration")
    assert constraint.mandatory_source_indices == (1,)
    assert constraint.release == "replaced_source_message"
    assert constraint.minimum_history_tokens((1,)) == 0
    assert constraint.minimum_history_tokens((2,)) == 28

    current_response = {"metadata": {"kv_memory_report": {
        "racer_current_mandatory_history": {
            "tokens": 16, "source_message_indices": [3],
            "release": "replaced_source_message"}}}}
    transport._current_retention = transport._current_retention_receipt(current_response)
    transport._resolution = "commit"
    next_draft = transport.backend_capacity_constraints(
        session_id="session-a", decision_key="decision-2", stage="draft")
    assert next_draft.mandatory_history_tokens == 16
    assert next_draft.mandatory_source_indices == ()
    assert next_draft.release == "never"
    assert next_draft.minimum_history_tokens((1, 2)) == 16


def test_binder_carrier_remap_matches_existing_regeneration_preflight():
    transport = generator()
    response = {"metadata": {"kv_memory_report": {
        "racer_transaction": {"regeneration_mandatory_history": {
            "tokens": 28, "source_message_indices": [_carrier_removed_source_index(2)],
            "release": "replaced_source_message"}}}}}
    transport._held_retention = transport._retention_receipt(response)
    constraint = transport.backend_capacity_constraints(
        session_id="session-a", decision_key="decision-1", stage="regeneration")
    assert constraint.mandatory_source_indices == (2,)
    recovered = PersistentMemory(
        MemoryView((), ()), (), (), (), (), recovery_tokens=232,
        native_evidence_source_indices=(2,))
    assert transport.regeneration_capacity(recovered) is None
    unrecovered = PersistentMemory(
        MemoryView((), ()), (), (), (), (), recovery_tokens=232,
        native_evidence_source_indices=(1,))
    rejection = transport.regeneration_capacity(unrecovered)
    assert rejection["status"] == "capacity_exhausted"
    assert rejection["mandatory_source_message_indices"] == [2]
    assert rejection["recovered_source_message_indices"] == [1]
