"""Exercise proof commits at the actual Goal decision/commit boundary."""
from contextlib import contextmanager
from types import SimpleNamespace
import copy
import json

import pytest

from benchmarks.memory_runtime.candidate_algorithms import VERIFIED_VERSION
from benchmarks.memory_runtime.candidate_algorithms.controller import CandidateRecoveryController
from benchmarks.memory_runtime.candidate_algorithms.goal_controller import GoalCompositionController
from benchmarks.memory_runtime.candidate_algorithms.repair_protocol import GuardVerdict
from benchmarks.memory_runtime.candidate_algorithms.verified_controller import VerifiedBindingController
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.policy import PolicyInputError
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, packing, policy, tool_pair
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.tests.test_goal_composition import call, payload


def controller(variant="goal_verified", score=0.1, binding_policy=None, tokenizer=None):
    geometry = packing()
    geometry["ratios"] = [8]
    base = EventNativeS0Controller(tokenizer or Tokenizer(), packing=geometry,
                                   policy=policy(budget=4096))
    return VerifiedBindingController(base, {"variant": variant}, risk_model=Risk(score),
                                     binding_policy=binding_policy)


class Proposal:
    def to_receipt(self):
        return {"fixture": "explicit source proof"}


class Binding:
    def __init__(self, *, propose=True, accept=True):
        self.enabled = propose
        self.accept = accept
        self.proposed = self.applied = 0

    def propose(self, context):
        self.proposed += 1
        return Proposal() if self.enabled else None

    def apply(self, context, proposal, calls):
        self.applied += 1
        corrected = copy.deepcopy(calls)
        corrected[0]["function"]["arguments"] = '{"key":42}'
        return tuple(corrected), {"proof": "fixture"}

    def validate(self, context, proposal, calls):
        return GuardVerdict(self.accept, "fixture_checked" if self.accept else "stale_proof")


@pytest.mark.parametrize("variant", ["goal_verified", "pending_verified"])
def test_no_proposal_preserves_base_memory_review_and_selected_response(variant):
    binding = Binding(propose=False)
    c = controller(variant, binding_policy=binding)
    baseline = (GoalCompositionController(controller().base, {"variant": "goal_pending"},
                                         risk_model=Risk(0.1)) if variant == "pending_verified" else
                CandidateRecoveryController(controller().base, {"variant": "goal_rescue"},
                                            risk_model=Risk(0.1)))
    rows = [{"role": "user", "content": "Find the price and then buy one item."},
            *tool_pair(1, result={"price": 12})]
    data = payload(rows=rows)
    p = c.prepare(data, ratio=8, max_new_tokens=32)
    b = baseline.prepare(data, ratio=8, max_new_tokens=32)
    assert p.memory == b.memory
    result = c.reconsider(p, [], draft_text="The price is 12.")
    old = baseline.reconsider(b, [], draft_text="The price is 12.")
    assert result["memory"] == old["memory"]
    assert result["regenerate"] == old["regenerate"]
    assert result["decision"]["reason"] == old["decision"]["reason"]
    assert binding.proposed == 0
    selected = [call("buy", {"key": 1})]
    c.validate_commit(p, selected, draft_text="buy")
    corrected, receipt = c.finalize_commit(p, selected)
    assert list(corrected) == selected and not receipt["changed"]
    assert binding.applied == 0


def test_goal_risk_recovery_preempts_proof_discovery_and_final_patch():
    binding = Binding()
    c = controller(score=0.9, binding_policy=binding)
    p = c.prepare(payload(), ratio=8, max_new_tokens=32)
    result = c.reconsider(p, [call()], draft_text="lookup")
    assert result["regenerate"] and binding.proposed == 0
    # Even an incorrectly retained proposal must not override Goal recovery.
    p._binding_proposal = Proposal()
    selected = [call(arguments={"key": 17})]
    c.validate_commit(p, selected, draft_text="recovered")
    output, receipt = c.finalize_commit(p, selected)
    assert list(output) == selected and binding.applied == 0
    assert receipt["status"] == "original_goal_recovery_preserved"


def test_proof_commit_requires_selected_draft_acceptance_and_is_idempotent():
    binding = Binding()
    c = controller(binding_policy=binding)
    p = c.prepare(payload(), ratio=8, max_new_tokens=32)
    original = [call()]
    result = c.reconsider(p, original, draft_text="lookup")
    assert not result["regenerate"]
    assert result["decision"]["verified_binding"]["status"] == "proposed"
    assert result["decision"]["version"] == VERIFIED_VERSION
    assert c.reconsider(p, original, draft_text="lookup") == result
    assert binding.proposed == 1
    c.validate_commit(p, original, draft_text="lookup")
    output, receipt = c.finalize_commit(p, original)
    assert json.loads(output[0]["function"]["arguments"]) == {"key": 42}
    assert receipt["changed"] and receipt["additional_generations"] == 0
    assert output[0]["id"] == original[0]["id"]
    assert c.finalize_commit(p, original) == (output, receipt)
    assert binding.applied == 1
    with pytest.raises(PolicyInputError, match="another selected draft"):
        c.finalize_commit(p, [call(arguments={"key": 2})])
    with pytest.raises(PolicyInputError, match="another held draft"):
        c.reconsider(p, [call(arguments={"key": 2})], draft_text="changed")


@pytest.mark.parametrize("mode", ["no_validation", "rejected_validation", "changed_selection", "invalid_proof"])
def test_no_patch_after_rejection_missing_validation_or_changed_selection(mode):
    binding = Binding(accept=mode != "invalid_proof")
    c = controller(binding_policy=binding)
    p = c.prepare(payload(), ratio=8, max_new_tokens=32)
    selected = [call()]
    c.reconsider(p, selected, draft_text="lookup")
    if mode != "no_validation":
        c.validate_commit(p, selected, draft_text="lookup",
                          parse_error="malformed" if mode == "rejected_validation" else None)
    if mode == "changed_selection":
        selected = [call(arguments={"key": 17})]
    output, receipt = c.finalize_commit(p, selected)
    assert list(output) == selected and not receipt["changed"]
    assert binding.applied == (1 if mode == "invalid_proof" else 0)


def test_task_cap_and_registry_version_preserve_the_existing_contract():
    binding = Binding()
    c = controller(binding_policy=binding)
    p = c.prepare(payload(), ratio=8, max_new_tokens=32)
    c._recovery_counts["composed"] = c.required_task_generation_limit
    result = c.reconsider(p, [call()], draft_text="lookup")
    assert not result["regenerate"] and binding.proposed == 0
    assert result["decision"]["verified_binding"]["status"] == "task_generation_limit_preserved"
    with pytest.raises(ValueError, match="registry version mismatch"):
        VerifiedBindingController(c.base, {"variant": "goal_verified", "proof_registry_version": "old"},
                                  risk_model=Risk(0.1))


def test_actual_runner_keeps_generated_text_and_cost_separate_from_proven_commit(tmp_path):
    class DraftTokenizer(Tokenizer):
        def decode(self, ids, **kwargs):
            return '<tool_call>{"name":"lookup","arguments":{"key":99}}</tool_call>'

    class Generator:
        @contextmanager
        def decision_scope(self, *, session_id=None):
            yield

        def generate(self, memory, **kwargs):
            return SimpleNamespace(token_ids=(100,), token_logprobs=(-0.1,),
                                   finish_reason="stop", stats={"eos_token_ids": []})

        def session_cache_info(self):
            return {"status": "empty"}

        def close_session(self):
            pass

    binding = Binding()
    tokenizer = DraftTokenizer()
    c = controller(binding_policy=binding, tokenizer=tokenizer)
    runner = EventNativeDecisionRunner(c, Generator(), tokenizer, ratio=8, max_new_tokens=32,
        max_generation_calls=2, journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    record = runner.run(payload())
    assert record["generation_completed"] == 1
    assert record["generation_usage_total"]["completion_tokens"] == 1
    assert record["commit_transform"]["changed"]
    assert record["commit_transform"]["model_generation_unmodified"]
    assert '"key":99' in record["generation_trace"][0]["native_draft"]["text"]
    assert json.loads(record["response"]["tool_calls"][0]["function"]["arguments"]) == {"key": 42}
    assert all(row["status"] == "passed" for row in record["pre_generation_budget_checks"])
    assert runner.run(payload()) == record
    assert binding.applied == 1
