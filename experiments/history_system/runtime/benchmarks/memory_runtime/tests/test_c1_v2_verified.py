"""C1 v2 composes S0, T02, and Verified without budget-dependent Goal review."""
import copy
import json
from pathlib import Path

import pytest

from benchmarks.memory_runtime.candidate_algorithms.c1_v2 import (
    C1V2VerifiedController, c1_v2_fields, validate_c1_v2_config,
)
from benchmarks.memory_runtime.candidate_algorithms.verified_controller import VerifiedBindingController
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_server import _candidate_ready_contract
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.policy import PolicyInputError
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, packing, policy, tool_pair
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.tests.test_goal_composition import call, payload
from benchmarks.memory_runtime.tests.test_verified_controller import Binding
from benchmarks.memory_runtime.tests.test_static_extensions import proof_payload
from benchmarks.memory_runtime.tests.test_verified_binding import _call


def config():
    return {"variant": "c1_v2_verified", "risk_artifact": {"fixture": True},
            "risk_threshold": 0.5, **c1_v2_fields("c1_v2_verified")}


def controller(*, score=0.1, binding=None, budget=4096, tokenizer=None):
    base = EventNativeS0Controller(tokenizer or Tokenizer(), packing={**packing(), "ratios": [8]},
                                   policy=policy(budget))
    return C1V2VerifiedController(base, config(), risk_model=Risk(score), binding_policy=binding)


def test_completion_review_is_disabled_even_when_large_packet_would_fit():
    c = controller()
    old = VerifiedBindingController(controller().base, {"variant": "pending_verified"},
                                    risk_model=Risk(0.1))
    rows = [{"role": "user", "content": "Find the price and then buy one item."},
            *tool_pair(1, result={"price": 12})]
    p = c.prepare(payload(rows=rows), ratio=8, max_new_tokens=32)
    b = old.prepare(payload(rows=rows), ratio=8, max_new_tokens=32)
    assert p.memory == b.memory
    c._goal_review_request = lambda *args, **kwargs: pytest.fail("Goal must never execute")
    result = c.reconsider(p, [], draft_text="The price is 12.")
    legacy = old.reconsider(b, [], draft_text="The price is 12.")
    assert legacy["regenerate"] and legacy["decision"]["reason"] == "stop_completion_review"
    assert not result["regenerate"] and "goal_review" not in result["decision"]
    assert result["decision"]["completion_review"] is False


@pytest.mark.parametrize("score", [0.1, 0.9])
def test_tool_draft_memory_recovery_and_verified_output_match_frozen_pending(score):
    c = controller(score=score, binding=Binding())
    old = VerifiedBindingController(controller().base, {"variant": "pending_verified"},
                                    risk_model=Risk(score), binding_policy=Binding())
    p = c.prepare(payload(), ratio=8, max_new_tokens=32)
    b = old.prepare(payload(), ratio=8, max_new_tokens=32)
    calls = [call()]
    result = c.reconsider(p, calls, draft_text="lookup")
    legacy = old.reconsider(b, calls, draft_text="lookup")
    assert p.memory == b.memory
    for key in ["memory", "regenerate"]:
        assert result[key] == legacy[key]
    assert result["decision"]["reason"] == legacy["decision"]["reason"]
    for ctrl, prepared in [(c, p), (old, b)]:
        ctrl.validate_commit(prepared, calls, draft_text="lookup")
    actual, receipt = c.finalize_commit(p, calls)
    expected, prior = old.finalize_commit(b, calls)
    assert actual == expected and receipt["changed"] == prior["changed"]


def test_real_frozen_proof_commits_without_goal_context_or_model_call():
    c = controller()
    p = c.prepare(proof_payload(), ratio=8, max_new_tokens=32)
    calls = [_call("set_budget_limit", {"access_token": "wrong", "budget_limit": 1500})]
    result = c.reconsider(p, calls, draft_text="call")
    assert not result["regenerate"] and result["memory"] == p.memory
    assert not hasattr(p, "_goal_context")
    c.validate_commit(p, calls, draft_text="call")
    output, receipt = c.finalize_commit(p, calls)
    assert json.loads(output[0]["function"]["arguments"])["access_token"] == "ABCDE12345"
    assert receipt["changed"] and receipt["additional_generations"] == 0
    assert output[0]["id"] == calls[0]["id"]
    assert c.reconsider(p, calls, draft_text="call") == result
    assert c.finalize_commit(p, calls) == (output, receipt)
    with pytest.raises(PolicyInputError, match="another selected draft"):
        c.finalize_commit(p, [])


def test_real_proof_reaches_served_response_through_existing_runner(tmp_path):
    from benchmarks.memory_runtime.attempt_journal import AttemptJournal
    from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
    from benchmarks.memory_runtime.tests.test_static_extensions import DraftTokenizer, Generator

    text = '<tool_call>' + json.dumps({"name": "set_budget_limit", "arguments": {
        "access_token": "wrong", "budget_limit": 1500}}) + '</tool_call>'
    tokenizer = DraftTokenizer([text])
    c = controller(tokenizer=tokenizer)
    runner = EventNativeDecisionRunner(c, Generator(), tokenizer, ratio=8,
        max_new_tokens=32, max_generation_calls=2,
        journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    record = runner.run(proof_payload())
    assert record["generation_completed"] == 1
    assert record["generation_trace"][0]["native_draft"]["text"] == text
    arguments = json.loads(record["response"]["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"access_token": "ABCDE12345", "budget_limit": 1500}
    assert record["commit_transform"]["changed"]
    assert record["commit_transform"]["additional_generations"] == 0
    assert all(row["status"] == "passed" for row in record["pre_generation_budget_checks"])
    assert runner.run(proof_payload()) == record


@pytest.mark.parametrize("mode", ["no_validation", "parse_error", "changed_selection", "invalid_proof"])
def test_rejected_or_unvalidated_commit_is_unchanged(mode):
    binding = Binding(accept=mode != "invalid_proof")
    c = controller(binding=binding)
    p = c.prepare(payload(), ratio=8, max_new_tokens=32)
    calls = [call()]
    c.reconsider(p, calls, draft_text="lookup")
    if mode != "no_validation":
        c.validate_commit(p, calls, draft_text="lookup",
                          parse_error="bad" if mode == "parse_error" else None)
    if mode == "changed_selection":
        calls = [call(arguments={"key": 17})]
    output, receipt = c.finalize_commit(p, calls)
    assert list(output) == calls and not receipt["changed"]


def test_task_cap_precedes_detector_and_verified_and_foreign_commit_is_rejected():
    binding = Binding()
    c = controller(binding=binding)
    p = c.prepare(payload(), ratio=8, max_new_tokens=32)
    c._recovery_counts[p._store.session_id] = c.required_task_generation_limit
    c.risk_model.predict_risk = lambda context: pytest.fail("cap must precede detector")
    result = c.reconsider(p, [call()], draft_text="lookup")
    assert result["decision"]["reason"] == "shared_task_generation_limit"
    assert binding.proposed == 0
    with pytest.raises(PolicyInputError, match="reviewed decision"):
        controller().validate_commit(p, [call()], draft_text="lookup")


def test_factory_preserves_actual_s0_bridge_and_validates_new_ready_contract(monkeypatch):
    from benchmarks.memory_runtime.candidate_algorithms.tool_event_rescue import ToolEventRescueAllocator

    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
                        lambda artifact: Risk(0.1))
    runtime = Path(__file__).resolve().parents[3]
    frozen = json.loads((runtime / "configs/controller.json").read_text(encoding="utf-8"))
    frozen.pop("post_draft_recovery")
    args = dict(packing={**packing(), "ratios": [8]}, policy=policy(4096),
                view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY)
    new = build_event_native_controller(Tokenizer(), **args,
                                       s0_config={**frozen, "candidate_algorithm": config()})
    assert isinstance(new.base, ToolEventRescueAllocator)
    old = build_event_native_controller(Tokenizer(), **args, s0_config=frozen)
    prepared = new.prepare(payload(), ratio=8, max_new_tokens=32)
    prior = old.prepare(payload(), ratio=8, max_new_tokens=32)
    assert prepared.memory == prior.memory
    assert prepared.metadata["same_event_reference"] == prior.metadata["same_event_reference"]
    identity, baseline = _candidate_ready_contract(config())
    assert baseline == "c2kv-c1-v2-verified-v1:c1_v2_verified"
    for key, value in c1_v2_fields("c1_v2_verified").items():
        assert identity[key] == value
        bad = config()
        bad.pop(key)
        with pytest.raises(ValueError, match="contract mismatch"):
            validate_c1_v2_config(bad)
    bad = config()
    bad["completion_review"] = 0
    with pytest.raises(ValueError, match="completion_review=False"):
        validate_c1_v2_config(bad)
