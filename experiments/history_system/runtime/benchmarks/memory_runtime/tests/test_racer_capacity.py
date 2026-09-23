"""Capacity rejection preserves an executable draft and its persistent state."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.candidate_algorithms.controller import CandidateRecoveryController
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner, EventNativeStepError
from benchmarks.memory_runtime.racer.capacity import HistoryCapacityInfeasible
from benchmarks.memory_runtime.racer.generator import PersistentRacerGenerator
from benchmarks.memory_runtime.tests.test_racer_transport import Native, Decoder
from benchmarks.memory_runtime.tests.test_racer_composition import allocator, config
from benchmarks.memory_runtime.tests.test_candidate_allocation import messages
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk


class RejectingNative(Native):
    def __init__(self, phase="regenerate", *, rollback_safe=True):
        super().__init__()
        self.phase, self.rollback_safe = phase, rollback_safe

    def _read_json(self, request, **kwargs):
        response, status = super()._read_json(request, **kwargs)
        if not request.full_url.endswith("/v1/chat/completions"):
            return response, status
        body = json.loads(request.data)
        transaction = body["c2kv_kv_memory_hint"]["persistent_history_session"]["transaction"]
        if transaction["phase"] != self.phase:
            return response, status
        return {"error": {"code": "RACER_CAPACITY_INFEASIBLE", "message": "pending does not fit",
            "capacity": {"schema": "racer-capacity-infeasible-v1",
                "stage": "regeneration" if self.phase == "regenerate" else "draft",
                "decision_id": transaction["decision_id"], "required_tokens": 28,
                "capacity_tokens": 24, "rollback_safe": self.rollback_safe}}}, 422


def runner_for(native, monkeypatch, tmp_path):
    control = CandidateRecoveryController(allocator(), {"variant": "static_t02"}, risk_model=Risk(0.9))
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.select_source_event",
        lambda prepared, *args, **kwargs: (prepared.memory.view.gist_event_ids[0],
            {"ranked_candidate_event_ids": [prepared.memory.view.gist_event_ids[0]]}))
    generator = PersistentRacerGenerator(native, Decoder(), config())
    runner = EventNativeDecisionRunner(control, generator, Decoder(), ratio=8, max_new_tokens=32,
        max_generation_calls=96, journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    return runner, generator, control


def test_rejected_recovery_keeps_draft_state_and_next_step(monkeypatch, tmp_path):
    native = RejectingNative()
    runner, generator, control = runner_for(native, monkeypatch, tmp_path)
    committed = []
    control.commit_memory = lambda prepared, memory: committed.append(memory) or {}
    def unexpected_commit(*args, **kwargs):
        raise AssertionError("Rejected recovery must submit the unmodified initial draft")
    control.validate_commit = unexpected_commit
    control.finalize_commit = unexpected_commit
    payload = {"session_id": "s", "decision_key": "d1", "messages": messages(), "tools": []}
    record = runner.run(payload)
    assert record["status"] == "ok"
    assert record["response"]["content"] == "Done"
    assert record["recovery_skipped"]["reason"] == "capacity"
    assert [row["discarded"] for row in record["generation_trace"]] == [False, True]
    assert record["generation_attempts"] == 2 and record["generation_completed"] == 1
    assert record["generation_usage_total"]["prompt_tokens"] is None
    assert record["generation_usage_known"]["prompt_tokens"] == 52
    assert committed[0].recovery_tokens == 0
    assert generator._messages == list(committed[0].source_messages)
    assert record["backend_commit"]["resolution_on_next_decision"] == "commit"
    assert not native.closed
    next_payload = {**payload, "decision_key": "d2", "recovery_disabled": True,
        "messages": messages() + [{"role": "assistant", "content": "Done"},
                                   {"role": "user", "content": "Continue"}]}
    result = runner.run(next_payload)
    assert result["status"] == "ok"
    assert len(committed) == 2
    assert not any("Historical source evidence" in str(row) for row in generator._messages)
    runner.close()


@pytest.mark.parametrize("phase,rollback_safe", [("draft", True), ("regenerate", False)])
def test_initial_or_unrollbackable_capacity_is_task_failure(monkeypatch, tmp_path, phase, rollback_safe):
    runner, _, _ = runner_for(RejectingNative(phase, rollback_safe=rollback_safe), monkeypatch, tmp_path)
    with pytest.raises(EventNativeStepError) as failed:
        runner.run({"session_id": "s", "decision_key": "d1", "messages": messages(), "tools": []})
    record = failed.value.record
    assert record["failure_kind"] == "method_failure"
    assert record["response"] is None
    assert record["error"]["type"] == "HistoryCapacityInfeasible"
    assert runner._terminal_error is None


@pytest.mark.parametrize("field,value", [("decision_id", "wrong"), ("stage", "draft"),
                                         ("required_tokens", True), ("capacity_tokens", 28)])
def test_rejection_must_be_bound_to_actual_decision(field, value):
    receipt = {"schema": "racer-capacity-infeasible-v1", "stage": "regeneration",
               "decision_id": "d1", "required_tokens": 28, "capacity_tokens": 24,
               "rollback_safe": True}
    receipt[field] = value
    with pytest.raises(ValueError, match="rejection receipt"):
        HistoryCapacityInfeasible.from_response(
            {"error": {"code": "RACER_CAPACITY_INFEASIBLE", "capacity": receipt}},
            decision_id="d1", phase="regeneration")
