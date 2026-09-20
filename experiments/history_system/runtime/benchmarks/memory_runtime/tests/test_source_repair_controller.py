"""Observable proposal admission and final commit behavior, including costs."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, messages, packing, policy
from benchmarks.memory_runtime.candidate_algorithms.repair_controller import RepairController
from benchmarks.memory_runtime.candidate_algorithms.repair_protocol import RepairProposal, GuardVerdict
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.policy import PolicyInputError


class ProposalPolicy:
    def __init__(self, verdict=GuardVerdict(True, "supported"), oversized=False):
        self.verdict = verdict
        self.oversized = oversized
        self.calls = 0

    def propose(self, context):
        self.calls += 1
        content = "Source-backed repair review" if not self.oversized else "x " * 10000
        return RepairProposal("source_review", ({"role": "user", "content": content},),
                              {"source_event_ids": [context.prepared._store.events[-1].event_id]})

    def validate(self, context, proposal, candidate_calls, *, draft_text, parse_error=None):
        return self.verdict


def control(proposer=None, tokenizer=None):
    geometry = packing()
    geometry["ratios"] = [8]
    base = EventNativeS0Controller(tokenizer or Tokenizer(), packing=geometry, policy=policy(budget=4096))
    return RepairController(base, {"variant": "request_contract"}, policy=proposer or ProposalPolicy())


def payload(key="d1"):
    return {"session_id": "repair", "decision_key": key, "messages": messages(), "tools": []}


def test_admission_is_measured_idempotent_and_has_no_detector_dependency():
    proposer = ProposalPolicy()
    controller = control(proposer)
    prepared = controller.prepare(payload(), ratio=8, max_new_tokens=32)
    result = controller.reconsider(prepared, [], draft_text="Done")
    assert result["regenerate"]
    assert "risk" not in result["decision"]["selection"]["selector"]
    assert history_budget_receipt(result["memory"], result["metadata"], controller,
                                 ratio=8, phase="regeneration")["status"] == "passed"
    assert controller.reconsider(prepared, [], draft_text="Done") == result
    assert proposer.calls == 1
    with pytest.raises(PolicyInputError, match="another held draft"):
        controller.reconsider(prepared, [], draft_text="Changed")
    again = controller.prepare(payload("d2"), ratio=8, max_new_tokens=32)
    assert controller.reconsider(again, [], draft_text="Different prose")["decision"]["reason"] == "source_action_state_already_reviewed"


def test_oversized_proposal_abstains_without_changing_view():
    controller = control(ProposalPolicy(oversized=True))
    prepared = controller.prepare(payload(), ratio=8, max_new_tokens=32)
    result = controller.reconsider(prepared, [], draft_text="Done")
    assert not result["regenerate"]
    assert result["memory"] == prepared.memory
    assert result["decision"]["reason"] == "repair_packet_exceeds_budget"


def test_generation_cap_precedes_policy():
    proposer = ProposalPolicy()
    controller = control(proposer)
    prepared = controller.prepare(payload(), ratio=8, max_new_tokens=32)
    controller._recovery_counts["repair"] = controller.required_task_generation_limit
    assert not controller.reconsider(prepared, [], draft_text="Done")["regenerate"]
    assert proposer.calls == 0


@pytest.mark.parametrize("accepted,fallback,selected", [(True, "original", 1),
                                                       (False, "original", 0),
                                                       (False, "stop", None)])
def test_commit_hook_keeps_both_generation_costs_and_only_selected_action(tmp_path, accepted, fallback, selected):
    class DraftTokenizer(Tokenizer):
        def decode(self, token_ids, **kwargs):
            value = "original" if token_ids[0] == 100 else "revised"
            return '<tool_call>{"name":"lookup","arguments":{"key":"' + value + '"}}</tool_call>'

    class Generator:
        count = 0

        @contextmanager
        def decision_scope(self, *, session_id=None):
            yield

        def generate(self, memory, **kwargs):
            self.count += 1
            return SimpleNamespace(token_ids=(99 + self.count,), token_logprobs=(-0.1,),
                                   finish_reason="stop", stats={"eos_token_ids": []})

        def session_cache_info(self):
            return {"status": "empty"}

        def close_session(self):
            pass

    tokenizer = DraftTokenizer()
    controller = control(ProposalPolicy(GuardVerdict(accepted, "fixture", fallback)), tokenizer)
    runner = EventNativeDecisionRunner(controller, Generator(), tokenizer, ratio=8,
                                      max_new_tokens=32, max_generation_calls=2,
                                      journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    record = runner.run(payload())
    assert record["generation_completed"] == 2
    assert record["generation_usage_total"]["completion_tokens"] == 2
    assert record["commit_validation"]["selected_generation_index"] == selected
    assert [row["discarded"] for row in record["generation_trace"]] == [selected != 0, selected != 1]
    if selected is None:
        assert record["response"]["tool_calls"] == []
        assert record["commit_validation"]["synthetic_abstention"]
    else:
        calls = record["response"]["tool_calls"]
        assert calls[0]["id"] == "d1_r0_0"
        assert ("original" if selected == 0 else "revised") in calls[0]["function"]["arguments"]
