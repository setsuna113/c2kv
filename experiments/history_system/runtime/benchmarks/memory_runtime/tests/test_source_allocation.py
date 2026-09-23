"""Budget, exact-source and recovery contracts for initial source allocation."""
import copy
import json

import pytest

from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, packing, policy, tool_pair
from benchmarks.memory_runtime.always_compress import CapacityInfeasible
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.candidate_algorithms.repacking import repack
from benchmarks.memory_runtime.recovery.config import E1_RECOVERY_VERSION
from benchmarks.memory_runtime.recovery.orchestrator import EventNativeRecoveryController
from benchmarks.memory_runtime.source_allocation import SourceAllocatedS0Controller


def controller(budget=128, **kwargs):
    return SourceAllocatedS0Controller(Tokenizer(), packing={**packing(), "ratios": [4, 8]},
                                       policy=policy(budget), **kwargs)


def request():
    pair = tool_pair(1, result={"ok": True})
    pair[0]["tool_calls"][0]["function"]["arguments"] = json.dumps({
        "file_name": "report.txt", "content": " ".join(["long"] * 320)})
    return {"session_id": "source-test", "decision_key": "d1", "tools": [],
            "messages": [{"role": "user", "content": "Write the file."}, *pair]}


def wrapped(base):
    return EventNativeRecoveryController(base, {"schema": E1_RECOVERY_VERSION, "gate": "disabled"},
                                         benchmark="bfcl")


def test_completed_producer_is_gist_and_current_result_stays_exact_raw():
    base = controller()
    payload = request()
    original = copy.deepcopy(payload)
    prepared = wrapped(base).prepare(payload, ratio=8, max_new_tokens=8)
    view = prepared.memory.view
    assert view.raw_source_indices == (0, 2)
    assert view.gist_source_indices == (1,)
    assert view.omitted_source_indices == ()
    assert payload == original
    assert prepared.metadata["source_allocation"]["legacy_fallback_invoked"] is False
    assert "capacity_fallback" not in prepared.metadata
    assert prepared.metadata["common_input_source_indices"] == [2]
    assert set(prepared.metadata["per_ratio"]) == {"8"}
    assert prepared.metadata["actual_managed_history_tokens"] <= 128
    assert history_budget_receipt(prepared.memory, prepared.metadata, base,
                                  ratio=8, phase="draft")["status"] == "passed"
    eligible = {(chunk.event_id, chunk.part_index, chunk.token_ids) for chunk in prepared.eligible_chunks}
    assert all((chunk.event_id, chunk.part_index, chunk.token_ids) in eligible for chunk in prepared.memory.chunks)
    measured, metadata, receipt = repack(base, prepared)
    assert measured.memory == prepared.memory
    assert metadata == prepared.metadata
    assert receipt["source_partition_reused"]


def test_small_history_can_stay_all_raw_without_duplicate_gist():
    payload = request()
    payload["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"file_name":"x"}'
    prepared = controller().prepare(payload, ratio=8, max_new_tokens=8)
    assert prepared.memory.view.raw_source_indices == (0, 1, 2)
    assert prepared.memory.chunks == ()
    assert prepared.metadata["actual_gist_tokens"] == 0


def test_current_user_and_pending_call_cannot_be_compressed_to_hide_overflow():
    payload = request()
    payload["messages"][0]["content"] = " ".join(["current"] * 300)
    with pytest.raises(CapacityInfeasible):
        controller().prepare(payload, ratio=8, max_new_tokens=8)
    payload = request()
    payload["messages"].pop()
    prepared = controller().prepare(payload, ratio=8, max_new_tokens=8)
    # A pending producer is part of common live input, remains raw, and is
    # constrained by physical context even when exempt from history bytes.
    assert prepared.memory.view.raw_source_indices == (0, 1)
    with pytest.raises(CapacityInfeasible):
        controller(model_context=64).prepare(payload, ratio=8, max_new_tokens=8)


def test_t02_restores_complete_event_or_abstains_without_changing_source():
    base = controller()
    prepared = wrapped(base).prepare(request(), ratio=8, max_new_tokens=8)
    candidate = prepared.memory.view.gist_event_ids[0]
    measure, metadata, receipt = repack(base, prepared, candidate=candidate)
    assert measure is metadata is None
    assert receipt["status"] == "abstained"
    assert prepared.memory.view.gist_source_indices == (1,)
    roomy = controller(budget=1000)
    # Reallocation under a roomy contract forces every source in that event raw.
    restored, metadata, receipt = repack(roomy, prepared, candidate=candidate)
    assert receipt["status"] == "admitted"
    assert candidate in restored.memory.view.raw_event_ids
    assert {1, 2} <= set(restored.memory.raw_source_indices)
    assert not {1, 2} & set(restored.memory.view.gist_source_indices)
    assert history_budget_receipt(restored.memory, metadata, roomy,
                                  ratio=8, phase="recovery")["status"] == "passed"


def test_same_bridge_is_removed_when_its_event_is_restored_raw(monkeypatch):
    from benchmarks.memory_runtime.source_needs import SourceRequest
    base = controller(budget=1000)
    monkeypatch.setattr(base, "_lexical_request", lambda *args, **kwargs: (
        SourceRequest((), (), "no_lexical_match"), None, {"status": "no_lexical_match"}))
    pair = tool_pair(1, result={"id": "project-1234", "detail": " ".join(["detail"] * 240)})
    pair[0]["tool_calls"][0]["function"] = {
        "name": "create_project", "arguments": '{"project_id":"project-1234"}'}
    payload = {"session_id": "bridge", "decision_key": "d1", "messages": [
        {"role": "user", "content": "Create a project."}, *pair,
        {"role": "user", "content": "Continue."}, *tool_pair(2, result={"value": "other"})],
        "tools": [{"type": "function", "function": {"name": "consume_project", "parameters": {
            "type": "object", "properties": {"id": {"type": "string"}, "project_id": {"type": "string"}},
            "required": ["id", "project_id"]}}}]}
    prepared = wrapped(base).prepare(payload, ratio=8, max_new_tokens=8)
    bridge = prepared.metadata["same_event_reference"]
    assert bridge["status"] == "admitted"
    assert len(prepared.metadata["derived_workspace_prefix_messages"]) == 1
    measure, metadata, receipt = repack(base, prepared, candidate=bridge["source_event_id"])
    assert receipt["status"] == "admitted"
    assert metadata["derived_workspace_prefix_messages"] == []
    assert metadata["same_event_reference"]["status"] == "replaced_by_complete_event_raw"
    assert metadata["same_event_reference"]["source_result_present_in_raw_workspace"]
    assert metadata["actual_managed_history_tokens"] == metadata["per_ratio"]["8"]["history_total_tokens"]
    assert history_budget_receipt(measure.memory, metadata, base, ratio=8, phase="recovery")["status"] == "passed"


def test_interleaved_producers_compare_joint_history_and_context_costs(monkeypatch):
    """A cheap early gist must not hide a feasible joint raw/gist assignment."""
    from dataclasses import replace
    from history_memory.events import EventStore
    base = controller(budget=100)
    first = tool_pair(1, result={"ok": 1})
    second = tool_pair(2, result={"ok": 2})
    rows = [first[0], second[0], second[1], first[1], {"role": "user", "content": "Continue."}]
    store = EventStore.from_messages("interleaved", rows)
    boundary = base._boundary(store, ())
    assert boundary["required"] - boundary["mandatory"] == {0, 1}
    original = base._measure_sources

    def costs(store, tools, raw, gist, mandatory, common, ratio, max_new, **kwargs):
        measured = original(store, tools, raw, gist, mandatory, common, ratio, max_new, **kwargs)
        # Isolate the admission algorithm with two independent source costs:
        # source 0 raw=(60 bytes,10 positions), gist=(10,70);
        # source 1 raw=(100,10), gist=(20,70). Only raw0+gist1 fits
        # both the 100-byte history and 80-position logical limits.
        history = sum(({0: 60, 1: 100}[index] if index in raw else {0: 10, 1: 20}[index])
                      for index in (0, 1) if index in raw or index in gist)
        logical = sum(10 if index in raw else 70 for index in (0, 1) if index in raw or index in gist)
        reasons = tuple(reason for exceeds, reason in ((history > 100, "history_byte_budget:8"),
                                                        (logical > 80, "model_logical_context")) if exceeds)
        return replace(measured, per_ratio={"8": {**measured.per_ratio["8"], "history_bytes": history}},
                       logical_sequence_tokens=logical, reasons=reasons)

    monkeypatch.setattr(base, "_measure_sources", costs)
    selected, receipt = base._allocate(store, (), ratio=8, max_new_tokens=8, boundary=boundary)
    assert selected is not None and not selected.reasons
    assert 0 in selected.memory.raw_source_indices
    assert 1 in selected.memory.view.gist_source_indices
    assert receipt["pruned_allocation_states"] == 0
