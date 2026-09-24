"""CPU contracts for scoped source-faithful native protection units."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.events import EventStore

from benchmarks.memory_runtime.racer.config import BackendConfig
from benchmarks.memory_runtime.racer.allocator import PersistentHistoryAllocator
from benchmarks.memory_runtime.racer.native_protection_v2 import NativeProtectionV2Allocator
from benchmarks.memory_runtime.racer.policies import BackendPolicy
from benchmarks.memory_runtime.racer.generator import PersistentRacerGenerator
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner, EventNativeStepError
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native import memory_to_dict
from benchmarks.memory_runtime.recovery.source import select_source_event
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, packing, policy
from benchmarks.memory_runtime.tests.test_racer_transport import Native, Decoder


def _controller():
    config = BackendConfig.parse({
        "schema": "racer-backend-v4", "backend": "h2o", "policy": "off",
        "history_budget_tokens": 4096, "extra_protection": "on",
        "allocation": "backend_native_persistent", "detector_calibration": "not_used",
    })
    return NativeProtectionV2Allocator(
        Tokenizer(), backend_config=config,
        packing={**packing(), "ratios": [8]}, policy=policy(4096),
    )


def _tool(call_id, content):
    return [{"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {
            "name": "lookup", "arguments": json.dumps({"query": call_id})}}]},
        {"role": "tool", "tool_call_id": call_id, "content": content}]


def _messages():
    records = {"records": [
        {"id": "alpha", "owner": "Mira", "status": "pending", "details": "keep this record"},
        {"id": "bravo", "owner": "Nora", "status": "closed", "details": "unrelated record"},
        {"id": "charlie", "owner": "Omar", "status": "closed", "details": "another record"},
    ]}
    return [{"role": "system", "content": "Use the visible records."},
            {"role": "user", "content": "Find records."},
            *_tool("first", json.dumps(records, indent=2)),
            {"role": "assistant", "content": "The records were found."},
            {"role": "user", "content": "Continue alpha, owned by Mira."}]


def _prepare(controller, messages, decision):
    return controller.prepare({"session_id": "v2-session", "decision_key": decision,
                               "messages": copy.deepcopy(messages), "tools": []},
                              ratio=8, max_new_tokens=32)


def _receipt(prepared, *, admitted=(), full=()):
    memory = prepared.memory
    return {"kv_memory_report": {"racer_native_protection": {
        "schema": "racer-native-protection-v2", "decision_id": prepared.metadata["decision_key"],
        "scope_id": memory.protection_scope_id,
        "event_ids": list(memory.protection_event_ids),
        "unit_ids": [unit["unit_id"] for unit in memory.protection_units],
        "units": [{"unit_id": unit["unit_id"], "event_id": unit["event_id"],
                   "admitted_rows": 2 if unit["unit_id"] in admitted else 0,
                   "total_rows": 2,
                   "status": "admitted" if unit["unit_id"] in admitted else "rejected"}
                  for unit in memory.protection_units],
        "event_coverage": [{"event_id": event_id, "total_rows": 2,
                            "full_rows": 2 if event_id in full else 0,
                            "partial_rows": 0 if event_id in full else 1,
                            "status": "full" if event_id in full else "partial"}
                           for event_id in memory.protection_event_ids],
    }}}


def test_json_unit_is_exact_compact_record_and_input_is_unchanged():
    controller = _controller()
    messages = _messages()
    prepared = _prepare(controller, messages, "d1")
    memory = prepared.memory
    ordinary = PersistentHistoryAllocator(Tokenizer(), backend_config=controller.backend_config,
        packing={**packing(), "ratios": [8]}, policy=policy(4096))
    baseline = _prepare(ordinary, messages, "d1")
    assert memory.source_messages == baseline.memory.source_messages
    assert memory.workspace_input_ids == baseline.memory.workspace_input_ids
    assert "protection_units" not in memory_to_dict(baseline.memory)
    assert memory_to_dict(memory)["protection_units"] == memory.protection_units
    assert memory.recovery_messages == ()
    matching = [unit for unit in memory.protection_units if unit["kind"] == "json_object"
                and any("alpha" in fragment["text"] for fragment in unit["fragments"])]
    assert matching
    assert all(unit["complete_event"] is False for unit in matching)
    for unit in matching:
        for fragment in unit["fragments"]:
            assert fragment["text"] in messages[fragment["source_index"]]["content"]
            assert json.loads(fragment["text"])["id"] == "alpha"
            assert len(fragment["text"]) < len(messages[3]["content"])


def test_lease_is_committed_only_for_admitted_units_and_expires_on_new_user():
    controller = _controller()
    messages = _messages()
    messages = [*messages[:-1], *_tool("other", "Alpha remains pending. Mira owns alpha."),
                {"role": "assistant", "content": "The second lookup finished."}, messages[-1]]
    first = _prepare(controller, messages, "d1")
    units = first.memory.protection_units
    assert len(units) >= 2
    admitted = units[-1]["unit_id"]
    stats = _receipt(first, admitted={admitted})
    controller.observe_native_protection(first, memory=first.memory, stats=stats)
    assert not controller._protection_leases
    controller.commit_native_protection(first, memory=first.memory, stats=stats)
    assert [unit["unit_id"] for unit in controller._protection_leases["v2-session"][1]] == [admitted]

    continued = [*messages, *_tool("second", "Alpha remains pending.")]
    second = _prepare(controller, continued, "d2")
    assert second.memory.protection_scope_id == first.memory.protection_scope_id
    assert second.memory.protection_units[0]["unit_id"] == admitted
    controller.commit_native_protection(second, memory=second.memory,
                                        stats=_receipt(second, admitted={admitted}))

    changed = [*continued, {"role": "user", "content": "Now work on bravo."}]
    third = _prepare(controller, changed, "d3")
    assert third.memory.protection_scope_id != first.memory.protection_scope_id
    assert third.memory.protection_units[0]["unit_id"] != admitted


def test_only_full_event_coverage_removes_recovery_candidate():
    controller = _controller()
    prepared = _prepare(controller, _messages(), "d1")
    event_id = next(unit["event_id"] for unit in prepared.memory.protection_units
                    if unit["kind"] == "json_object")
    wrapper = SimpleNamespace(memory=prepared.memory, metadata=prepared.metadata,
                              _store=EventStore.from_messages("v2-session", _messages()))
    partial = _receipt(prepared, admitted={unit["unit_id"] for unit in prepared.memory.protection_units})
    controller.observe_native_protection(wrapper, memory=prepared.memory, stats=partial)
    selected, _ = select_source_event(wrapper, [], draft_text="alpha Mira")
    assert selected == event_id

    full = _receipt(prepared, full={event_id})
    bad = copy.deepcopy(full)
    event = next(row for row in bad["kv_memory_report"]["racer_native_protection"]["event_coverage"]
                 if row["event_id"] == event_id)
    event["full_rows"] = 0
    with pytest.raises(ValueError, match="every model row"):
        controller.observe_native_protection(wrapper, memory=prepared.memory, stats=bad)
    event["total_rows"] = 0
    with pytest.raises(ValueError, match="every model row"):
        controller.observe_native_protection(wrapper, memory=prepared.memory, stats=bad)
    controller.observe_native_protection(wrapper, memory=prepared.memory, stats=full)
    selected, receipt = select_source_event(wrapper, [], draft_text="alpha Mira")
    assert selected != event_id
    assert event_id not in receipt["ranked_candidate_event_ids"]


def test_selected_recovery_memory_carries_plan_but_failed_selection_does_not_commit():
    controller = _controller()
    prepared = _prepare(controller, _messages(), "d1")
    memory = prepared.memory
    store = EventStore.from_messages("v2-session", _messages())
    candidate = next(event.event_id for event in store.events if event.kind == "tool_event")
    measured = controller._try_measure(store, (), (*memory.view.raw_event_ids, candidate),
                                       memory.view.mandatory_raw_event_ids,
                                       memory.view.gist_event_ids,
                                       prepared.metadata["eligible_extraction"]["eligible_event_ids"],
                                       prepared.metadata["common_raw_prompt_tokens"], 32)
    assert measured.memory.protection_scope_id == memory.protection_scope_id
    assert measured.memory.protection_units == memory.protection_units
    assert not controller._protection_leases
    selected_unit = memory.protection_units[0]["unit_id"]
    selected = _receipt(prepared)
    for outcome in selected["kv_memory_report"]["racer_native_protection"]["units"]:
        if outcome["unit_id"] == selected_unit:
            outcome.update(status="retained", retained_rows=2, coverage_status="full")
    controller.commit_native_protection(prepared, memory=measured.memory, stats=selected)
    assert [unit["unit_id"] for unit in controller._protection_leases["v2-session"][1]] == [selected_unit]
    controller.clear_native_protection()
    assert not controller._active_plans and not controller._protection_leases


class _ReceiptNative(Native):
    def _read_json(self, request, **kwargs):
        result, status = super()._read_json(request, **kwargs)
        if request.full_url.endswith("/v1/chat/completions"):
            session = json.loads(request.data)["c2kv_kv_memory_hint"]["persistent_history_session"]
            plan = session["extra_protection"]
            result["metadata"]["kv_memory_report"]["racer_native_protection"] = {
                "schema": "racer-native-protection-v2", "applied": True,
                "status": "admitted", "decision_id": session["transaction"]["decision_id"],
                "scope_id": plan["scope_id"], "event_ids": plan["event_ids"],
                "unit_ids": plan["unit_ids"],
                "units": [{"unit_id": unit["unit_id"], "event_id": unit["event_id"],
                           "status": "admitted", "admitted_rows": 1, "total_rows": 1,
                           "coverage_status": "full", "retained_rows": 1}
                          for unit in plan["units"]],
                "event_coverage": [{"event_id": event_id, "status": "partial",
                                    "full_rows": 0, "partial_rows": 1, "total_rows": 1}
                                   for event_id in plan["event_ids"]],
            }
        return result, status


def _runner(tmp_path):
    allocator = _controller()
    control = BackendPolicy(allocator, allocator.backend_config, native_allocator=allocator)
    native = _ReceiptNative()
    generator = PersistentRacerGenerator(native, Decoder(), allocator.backend_config)
    runner = EventNativeDecisionRunner(control, generator, Decoder(), ratio=8,
        max_new_tokens=32, max_generation_calls=96,
        journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    return runner, allocator, generator


def test_runner_commits_only_after_selected_backend_resolution(tmp_path):
    payload = {"session_id": "v2-session", "decision_key": "d1",
               "messages": _messages(), "tools": []}
    runner, allocator, _ = _runner(tmp_path / "success")
    result = runner.run(payload)
    assert result["status"] == "ok"
    assert allocator._protection_leases["v2-session"][1]
    runner.close()
    assert not allocator._protection_leases

    failed_runner, failed_allocator, failed_generator = _runner(tmp_path / "failed")
    failed_generator.resolve_decision = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("backend resolution failed"))
    with pytest.raises(EventNativeStepError, match="backend resolution failed"):
        failed_runner.run(payload)
    assert not failed_allocator._protection_leases
