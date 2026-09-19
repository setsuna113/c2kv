"""Whole-controller contracts for opt-in T02 recovery and completion review."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from benchmarks.memory_runtime.tests.test_candidate_allocation import (
    Tokenizer, messages, packing, policy, tool_pair,
)
from benchmarks.memory_runtime.candidate_algorithms.allocation import CandidateAllocator
from benchmarks.memory_runtime.candidate_algorithms.controller import CandidateRecoveryController, memory_signature
from benchmarks.memory_runtime.candidate_algorithms.progress import review_request
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.policy import PolicyInputError


class Risk:
    def __init__(self, score):
        self.score = score

    def predict_risk(self, context):
        return SimpleNamespace(available=self.score is not None, score=self.score,
                               reason=None if self.score is not None else "missing test features")


def controller(variant="static_t02", score=0.9):
    geometry = packing()
    geometry["ratios"] = [8]
    base = CandidateAllocator(Tokenizer(), packing=geometry, policy=policy(),
                              variant=variant)
    return CandidateRecoveryController(base, {"variant": variant, "risk_threshold": 0.5},
                                       risk_model=Risk(score))


def prepare(control, rows=None, key="turn-0/step-0"):
    return control.prepare({"session_id": "test", "decision_key": key,
                            "messages": rows or messages(), "tools": []},
                           ratio=8, max_new_tokens=32)


def test_gate_reject_skips_retrieval_and_keeps_exact_draft_view(monkeypatch):
    control = controller(score=0.1)
    prepared = prepare(control)
    def forbidden(*args, **kwargs):
        raise AssertionError("Risk rejection must not retrieve")
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.select_source_event", forbidden)
    result = control.reconsider(prepared, [], draft_text="Done")
    assert not result["regenerate"]
    assert result["memory"] == prepared.memory
    assert result["decision"]["selection"]["score"] == 0.1


def test_gate_unavailable_fails_closed():
    control = controller(score=None)
    with pytest.raises(PolicyInputError, match="risk unavailable"):
        control.reconsider(prepare(control), [], draft_text="Done")


def test_complete_source_replaces_its_gist_and_respects_budget(monkeypatch):
    control = controller()
    prepared = prepare(control)
    candidate = next(iter(prepared.memory.view.gist_event_ids))
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.select_source_event",
                        lambda *args, **kwargs: (candidate, {"ranked_candidate_event_ids": [candidate]}))
    result = control.reconsider(prepared, [], draft_text="Need the historical value")
    assert result["regenerate"]
    assert candidate in result["memory"].view.raw_event_ids
    assert candidate not in result["memory"].view.gist_event_ids
    assert result["metadata"]["eligible_extraction"]["retained_encoder_unit_ids"] == list(
        dict.fromkeys(chunk.event_id for chunk in result["memory"].chunks))
    assert history_budget_receipt(result["memory"], result["metadata"], control,
                                  ratio=8, phase="regeneration")["status"] == "passed"
    assert control.reconsider(prepared, [], draft_text="Need the historical value") == result
    assert control._recovery_counts["test"] == 1
    with pytest.raises(PolicyInputError, match="another held draft"):
        control.reconsider(prepared, [], draft_text="Different")


def test_stop_review_changes_view_even_below_risk_threshold_once_per_observation():
    control = controller("goal_rescue", score=0.1)
    rows = [{"role": "system", "content": "Use tools."},
            {"role": "user", "content": "Find the price, then buy one item."},
            *tool_pair(1, result={"price": 12})]
    prepared = prepare(control, rows)
    result = control.reconsider(prepared, [], draft_text="The price is 12.")
    assert result["regenerate"]
    assert result["decision"]["reason"] == "stop_completion_review"
    assert memory_signature(result["memory"]) != memory_signature(prepared.memory)
    assert result["decision"]["goal_review"]["stop_is_error_label"] is False
    assert history_budget_receipt(result["memory"], result["metadata"], control,
                                  ratio=8, phase="regeneration")["status"] == "passed"
    # Merely repeating prose must not create another review of the same facts.
    again = prepare(control, rows + [{"role": "assistant", "content": "The price is 12."}],
                    key="turn-0/step-1")
    assert not control.reconsider(again, [], draft_text="The price is 12.")["regenerate"]


def test_stop_without_observed_operation_remains_legal():
    control = controller("goal_rescue", score=0.1)
    rows = [{"role": "user", "content": "Which details do you need?"}]
    result = control.reconsider(prepare(control, rows), [], draft_text="Please provide an account.")
    assert not result["regenerate"]


def test_different_successful_actions_are_not_a_failure_loop():
    control = controller("goal_rescue", score=0.1)
    rows = [{"role": "user", "content": "Find values."},
            *tool_pair(1, result={"value": 10}), *tool_pair(2, result={"value": 10})]
    prepared = prepare(control, rows)
    calls = [{"type": "function", "function": {"name": "lookup", "arguments": '{"key":3}'}}]
    reason, *_ = review_request(prepared, calls)
    assert reason is None


def test_generation_limit_abstains_without_risk_computation():
    control = controller()
    prepared = prepare(control)
    control._recovery_counts["test"] = control.required_task_generation_limit
    control.risk_model.predict_risk = lambda context: pytest.fail("cap must precede scoring")
    result = control.reconsider(prepared, [], draft_text="Done")
    assert result["decision"]["reason"] == "shared_task_generation_limit"
    assert not result["regenerate"]


def test_goal_review_preserves_admitted_bridge_and_failed_operation_cue(monkeypatch):
    from benchmarks.memory_runtime.same_event_bridge_only import (
        SAME_EVENT_BRIDGE_ONLY_POLICY,
        SameEventBridgeOnlyS0Controller,
    )
    from benchmarks.memory_runtime.source_needs import SourceRequest

    geometry = packing()
    geometry["ratios"] = [8]
    base = SameEventBridgeOnlyS0Controller(
        Tokenizer(), packing=geometry, policy=policy(),
        same_event_bridge_only_policy=SAME_EVENT_BRIDGE_ONLY_POLICY,
    )
    monkeypatch.setattr(base, "_lexical_request", lambda *args, **kwargs: (
        SourceRequest((), (), "no_lexical_match"), None,
        {"status": "no_lexical_match"},
    ))
    control = CandidateRecoveryController(
        base, {"variant": "goal_rescue", "risk_threshold": 0.5},
        risk_model=Risk(0.1),
    )
    request = {
        "session_id": "bridge-cue",
        "decision_key": "d1",
        "messages": [
            {"role": "system", "content": "Use observed tool results."},
            {"role": "user", "content": "Create a project."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "create-1", "type": "function",
                "function": {"name": "create_project", "arguments": (
                    '{"project_id":"project-1234"}'
                )},
            }]},
            {"role": "tool", "tool_call_id": "create-1", "content": (
                '{"id":"project-1234"}'
            )},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "note-1", "type": "function",
                "function": {"name": "note", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "note-1", "content": (
                '{"value":"other"}'
            )},
            {"role": "user", "content": "Continue."},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "status-1", "type": "function",
                "function": {"name": "status_check", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "status-1", "content": (
                '{"error":"temporary failure"}'
            )},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "consume_project",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "project_id": {"type": "string"},
                    },
                    "required": ["id", "project_id"],
                },
            },
        }],
    }
    prepared = control.prepare(request, ratio=8, max_new_tokens=32)
    initial = prepared.metadata["derived_workspace_prefix_messages"]
    assert prepared.metadata["same_event_reference"]["status"] == "admitted"
    assert prepared.metadata["failed_operation_cue"]["status"] == "admitted"
    assert len(initial) == 2
    result = control.reconsider(prepared, [], draft_text="Done.")
    assert result["regenerate"]
    derived = result["metadata"]["derived_workspace_prefix_messages"]
    assert len(derived) == 3
    assert derived[:2] == initial
    assert result["decision"]["goal_review"]["status"] == "admitted"
    assert history_budget_receipt(
        result["memory"], result["metadata"], control,
        ratio=8, phase="regeneration",
    )["status"] == "passed"


def test_decision_runner_keeps_candidate_call_ids_stable_only_for_new_arms(
    tmp_path, monkeypatch,
):
    from contextlib import contextmanager

    from benchmarks.memory_runtime.attempt_journal import AttemptJournal
    from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
    from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
    from benchmarks.memory_runtime.recovery.config import E1_RECOVERY_VERSION
    from benchmarks.memory_runtime.recovery.orchestrator import (
        EventNativeRecoveryController,
    )

    native_call = '<tool_call>{"name":"lookup","arguments":{"key":"x"}}</tool_call>'

    class DraftTokenizer(Tokenizer):
        def decode(self, token_ids, **kwargs):
            return native_call

    class Generator:
        @contextmanager
        def decision_scope(self, *, session_id=None):
            yield

        def generate(self, memory, *, ratio, max_new_tokens, **kwargs):
            return SimpleNamespace(
                token_ids=(123,), token_logprobs=(-0.1,),
                finish_reason="stop", stats={"eos_token_ids": []},
            )

        def session_cache_info(self):
            return {"status": "empty"}

        def close_session(self):
            return None

    def run(control, label):
        runner = EventNativeDecisionRunner(
            control, Generator(), DraftTokenizer(), ratio=8,
            max_new_tokens=32, max_generation_calls=2,
            journal=AttemptJournal(tmp_path / label / "attempts.jsonl"),
        )
        record = runner.run({
            "session_id": "stable-ids", "decision_key": "d1",
            "messages": messages(), "tools": [],
        })
        return record["response"]["tool_calls"][0]["id"], record

    geometry = packing()
    geometry["ratios"] = [8]

    def candidate(score):
        base = CandidateAllocator(
            DraftTokenizer(), packing=geometry, policy=policy(),
            variant="static_t02",
        )
        return CandidateRecoveryController(
            base, {"variant": "static_t02", "risk_threshold": 0.5},
            risk_model=Risk(score),
        )

    def source(prepared, *args, **kwargs):
        selected = prepared.memory.view.gist_event_ids[0]
        return selected, {"ranked_candidate_event_ids": [selected]}

    monkeypatch.setattr(
        "benchmarks.memory_runtime.candidate_algorithms.controller.select_source_event",
        source,
    )
    candidate_direct_id, candidate_direct = run(candidate(0.1), "candidate-direct")
    candidate_recovered_id, candidate_recovered = run(
        candidate(0.9), "candidate-recovered"
    )
    assert len(candidate_direct["generation_trace"]) == 1
    assert len(candidate_recovered["generation_trace"]) == 2
    assert candidate_direct_id == candidate_recovered_id == "d1_r0_0"

    def legacy(triggered):
        base = EventNativeS0Controller(
            DraftTokenizer(), packing=geometry, policy=policy(),
        )
        control = EventNativeRecoveryController(
            base, {"schema": E1_RECOVERY_VERSION, "gate": "seeded_random"},
            benchmark="bfcl",
        )
        control._gate = lambda prepared: {
            "triggered": triggered, "reason": "test_gate",
        }
        control._select_event = lambda prepared, *args, **kwargs: (
            prepared.memory.view.gist_event_ids[0], {"reason": "test_source"},
        )
        return control

    legacy_direct_id, legacy_direct = run(legacy(False), "legacy-direct")
    legacy_recovered_id, legacy_recovered = run(legacy(True), "legacy-recovered")
    assert len(legacy_direct["generation_trace"]) == 1
    assert len(legacy_recovered["generation_trace"]) == 2
    assert legacy_direct_id == "d1_r0_0"
    assert legacy_recovered_id == "d1_r1_0"
