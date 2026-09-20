"""Goal branch priority, bounded composition, continuation, and commit provenance."""
from contextlib import contextmanager
from types import SimpleNamespace
import copy
import json

import pytest

from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, messages, packing, policy, tool_pair
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.candidate_algorithms.controller import CandidateRecoveryController
from benchmarks.memory_runtime.candidate_algorithms.goal_controller import GoalCompositionController
from benchmarks.memory_runtime.candidate_algorithms.goal_commit import corrected_draft
from benchmarks.memory_runtime.candidate_algorithms.repair_protocol import RepairProposal, GuardVerdict
from benchmarks.memory_runtime.event_native_s0_policy import EventNativeS0Controller
from benchmarks.memory_runtime.event_native_draft import parse_native_draft
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.policy import PolicyInputError


def call(name="lookup", arguments=None, call_id="fixture"):
    return {"id": call_id, "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments or {"key": 99})}}


def payload(key="d1", rows=None):
    return {"session_id": "composed", "decision_key": key, "messages": rows or messages(), "tools": []}


def control(variant="goal_source", score=0.1, policies=(), tokenizer=None):
    geometry = packing()
    geometry["ratios"] = [8]
    base = EventNativeS0Controller(tokenizer or Tokenizer(), packing=geometry, policy=policy(budget=4096))
    return GoalCompositionController(base, {"variant": variant}, risk_model=Risk(score), policies=policies)


class Extra:
    def __init__(self, oversized=False):
        self.count = 0
        self.oversized = oversized

    def propose(self, context):
        self.count += 1
        return RepairProposal("fixture_extra", ({"role": "user", "content":
            "review recorded source" if not self.oversized else "x " * 100000},), {"state_key": "epoch1"})

    def validate(self, *args, **kwargs):
        return GuardVerdict(False, "fixture_keep_original")


@pytest.mark.parametrize("variant", ["goal_pending", "goal_source", "goal_progress", "goal_joint"])
def test_first_view_and_original_goal_stop_branch_are_preserved(variant):
    composed = control(variant)
    baseline = CandidateRecoveryController(control().base, {"variant": "goal_rescue"}, risk_model=Risk(0.1))
    rows = [{"role": "user", "content": "Find the price then buy one item."}, *tool_pair(1, result={"price": 12})]
    data = payload(rows=rows)
    p = composed.prepare(data, ratio=8, max_new_tokens=32)
    b = baseline.prepare(data, ratio=8, max_new_tokens=32)
    assert p.memory == b.memory
    result = composed.reconsider(p, [], draft_text="The price is 12.")
    original = baseline.reconsider(b, [], draft_text="The price is 12.")
    assert original["regenerate"] and result["regenerate"]
    assert result["decision"]["backbone"]["reason"] == original["decision"]["reason"]
    assert result["decision"]["additional_policy"] is None
    assert composed._recovery_counts["composed"] == 1
    if variant in {"goal_source", "goal_progress"}:
        assert result["memory"] == original["memory"]


def test_original_risk_recovery_preempts_every_additional_policy():
    extra = Extra()
    c = control(score=0.9, policies=(("source", extra),))
    p = c.prepare(payload(), ratio=8, max_new_tokens=32)
    result = c.reconsider(p, [call()], draft_text="lookup")
    assert result["regenerate"]
    assert result["decision"]["backbone"]["reason"] == "risk_triggered_complete_event_replaced"
    assert extra.count == 0


def test_extra_uses_only_unused_slot_and_preserves_budget_and_idempotence():
    first, second = Extra(), Extra()
    c = control(policies=(("source", first), ("progress", second)))
    p = c.prepare(payload(), ratio=8, max_new_tokens=32)
    result = c.reconsider(p, [call()], draft_text="lookup")
    assert result["regenerate"] and result["decision"]["additional_policy"] == "source"
    assert first.count == 1 and second.count == 0
    assert c._recovery_counts["composed"] == 1
    assert history_budget_receipt(result["memory"], result["metadata"], c,
                                  ratio=8, phase="regeneration")["status"] == "passed"
    assert result == c.reconsider(p, [call()], draft_text="lookup")
    with pytest.raises(PolicyInputError, match="another held draft"):
        c.reconsider(p, [call(arguments={"key": 1})], draft_text="changed")
    p2 = c.prepare(payload("d2"), ratio=8, max_new_tokens=32)
    c.policies = (("source", first),)
    assert not c.reconsider(p2, [call()], draft_text="lookup")["regenerate"]


def test_cap_and_infeasible_extra_keep_original_goal_abstention():
    extra = Extra(oversized=True)
    c = control(policies=(("source", extra),))
    p = c.prepare(payload(), ratio=8, max_new_tokens=32)
    result = c.reconsider(p, [call()], draft_text="lookup")
    assert not result["regenerate"] and result["memory"] == p.memory
    assert result["decision"]["additional_attempts"][0]["status"] == "packet_exceeds_budget"
    p2 = c.prepare(payload("d2"), ratio=8, max_new_tokens=32)
    c._recovery_counts["composed"] = c.required_task_generation_limit
    result = c.reconsider(p2, [call()], draft_text="lookup")
    assert result["decision"]["reason"] == "shared_task_generation_limit"
    assert extra.count == 1


def test_field_transform_keeps_names_ids_and_parseable_native_response():
    draft = parse_native_draft('<tool_call>{"name":"send","arguments":{"recipient":"wrong"}}</tool_call>', call_id_prefix="stable")
    patched = copy.deepcopy(draft.tool_calls)
    patched[0]["function"]["arguments"] = '{"recipient":"correct"}'
    result = corrected_draft(draft, patched, benchmark="bfcl")
    assert result.tool_calls[0]["id"] == draft.tool_calls[0]["id"]
    assert "correct" in result.text and "wrong" in draft.text
    patched[0]["function"]["name"] = "delete"
    with pytest.raises(ValueError, match="identity"):
        corrected_draft(draft, patched, benchmark="bfcl")


def test_step_journals_patch_without_rewriting_generation_or_cost(tmp_path):
    class DraftTokenizer(Tokenizer):
        def decode(self, ids, **kwargs):
            return '<tool_call>{"name":"send","arguments":{"recipient":"wrong"}}</tool_call>'

    class PatchPolicy:
        def propose(self, context):
            return None

        def patch_calls(self, context, calls):
            result = copy.deepcopy(calls)
            result[0]["function"]["arguments"] = '{"recipient":"correct"}'
            return result, {"source": "fixture_current_user"}

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

    tokenizer = DraftTokenizer()
    c = control(policies=(("source", PatchPolicy()),), tokenizer=tokenizer)
    runner = EventNativeDecisionRunner(c, Generator(), tokenizer, ratio=8, max_new_tokens=32,
        max_generation_calls=2, journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    record = runner.run(payload())
    assert record["generation_completed"] == 1
    assert record["generation_usage_total"]["completion_tokens"] == 1
    assert record["commit_transform"]["changed"]
    assert record["commit_transform"]["model_generation_unmodified"]
    assert '"correct"' in record["response"]["tool_calls"][0]["function"]["arguments"]
    assert "wrong" in record["generation_trace"][0]["native_draft"]["text"]
    assert runner.run(payload()) == record


def test_deferred_consumer_requires_real_producer_and_expires_after_actual_attempt():
    c = control()
    rows = [{"role": "user", "content": "Look up the airport, then book the flight."}]
    p = c.prepare(payload(rows=rows), ratio=8, max_new_tokens=32)
    producer = call("get_airport", {"city": "Example"}, "p")
    consumer = call("book_flight", {"travel_from": "UNKNOWN", "traveler": "Alice"}, "c")
    goal = p._store.events[-1]
    p._goal_proposal = RepairProposal("dependency", (), {}, guard={"deferred_consumers": [{
        "request_event_id": goal.event_id, "call": consumer, "producer_calls": [producer],
        "unbound_paths": [["travel_from"]]}]})
    c.finalize_commit(p, [producer])
    assert c._active_deferred(p) == ()
    rows += [{"role": "assistant", "content": None, "tool_calls": [producer]},
             {"role": "tool", "tool_call_id": "p", "content": '{"airport":"EXA"}'}]
    p2 = c.prepare(payload("d2", rows), ratio=8, max_new_tokens=32)
    active = c._active_deferred(p2)
    assert len(active) == 1 and active[0]["producer"]["observed_result"] == {"airport": "EXA"}
    resolved = call("book_flight", {"travel_from": "EXA", "traveler": "Alice"}, "c2")
    rows += [{"role": "assistant", "content": None, "tool_calls": [resolved]},
             {"role": "tool", "tool_call_id": "c2", "content": '{"success":true}'}]
    p3 = c.prepare(payload("d3", rows), ratio=8, max_new_tokens=32)
    assert c._active_deferred(p3) == ()
    assert c._deferred["composed"] == []


def test_deferred_consumer_does_not_survive_new_user_request():
    c = control()
    rows = [{"role": "user", "content": "Look up the airport, then book the flight."}]
    p = c.prepare(payload(rows=rows), ratio=8, max_new_tokens=32)
    c._deferred["composed"] = [{"request_event_id": p._store.events[-1].event_id,
        "after_source_index": 0, "call": call("book_flight"), "producer_calls": [call()]}]
    rows += [{"role": "assistant", "content": "I will look it up."},
             {"role": "user", "content": "Cancel that request."}]
    p2 = c.prepare(payload("d2", rows), ratio=8, max_new_tokens=32)
    assert c._active_deferred(p2) == ()


def test_chained_deferred_consumer_activates_on_resolved_intermediate_producer():
    c = control()
    rows = [{"role": "user", "content": "Find the airport then the price then book."}]
    p = c.prepare(payload(rows=rows), ratio=8, max_new_tokens=32)
    airport = call("get_airport", {"city": "Example"}, "airport")
    cost = call("get_flight_cost", {"travel_from": "UNKNOWN"}, "cost")
    booking = call("book_flight", {"travel_from": "UNKNOWN"}, "booking")
    p._goal_proposal = RepairProposal("dependency", (), {}, guard={"deferred_consumers": [
        {"request_event_id": p._store.events[-1].event_id, "call": cost,
         "producer_calls": [airport], "unbound_paths": [["travel_from"]]},
        {"request_event_id": p._store.events[-1].event_id, "call": booking,
         "producer_calls": [cost], "unbound_paths": [["travel_from"]]}]})
    _, receipt = c.finalize_commit(p, [airport])
    assert len(receipt["deferred_consumers_registered"]) == 2
    rows += [{"role": "assistant", "content": None, "tool_calls": [airport]},
             {"role": "tool", "tool_call_id": "airport", "content": '{"airport":"EXA"}'}]
    p2 = c.prepare(payload("d2", rows), ratio=8, max_new_tokens=32)
    assert [row["call"]["function"]["name"] for row in c._active_deferred(p2)] == ["get_flight_cost"]
    actual_cost = call("get_flight_cost", {"travel_from": "EXA"}, "actual-cost")
    rows += [{"role": "assistant", "content": None, "tool_calls": [actual_cost]},
             {"role": "tool", "tool_call_id": "actual-cost", "content": '{"price":123}'}]
    p3 = c.prepare(payload("d3", rows), ratio=8, max_new_tokens=32)
    active = c._active_deferred(p3)
    assert [row["call"]["function"]["name"] for row in active] == ["book_flight"]
    assert active[0]["producer"]["arguments"] == {"travel_from": "EXA"}


def test_ace_commit_text_and_calls_agree_after_field_patch():
    from benchmarks.memory_runtime.acebench_source import parse_acebench_draft
    draft = parse_acebench_draft("[send(recipient='old')]", call_id_prefix="ace")
    patched = copy.deepcopy(draft.tool_calls)
    patched[0]["function"]["arguments"] = '{"recipient":"new"}'
    result = corrected_draft(draft, patched, benchmark="acebench")
    reparsed = parse_acebench_draft(result.content, call_id_prefix="ace")
    assert reparsed.tool_calls == result.tool_calls
