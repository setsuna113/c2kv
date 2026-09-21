"""Static backbone parity and source-proven changes at the actual commit boundary."""
from contextlib import contextmanager
from types import SimpleNamespace
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.candidate_algorithms import STATIC_EXTENSION_VARIANTS, STATIC_EXTENSION_VERSION
from benchmarks.memory_runtime.candidate_algorithms.allocation import CandidateAllocator
from benchmarks.memory_runtime.candidate_algorithms.controller import CandidateRecoveryController
from benchmarks.memory_runtime.candidate_algorithms.static_extensions import (
    StaticVerifiedController, StaticActionLedgerController, extension_fields,
)
from benchmarks.memory_runtime.event_native_controls import build_event_native_controller
from benchmarks.memory_runtime.event_native_server import _candidate_ready_contract
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.always_compress import ALWAYS_COMPRESSION_POLICY
from benchmarks.memory_runtime.event_native_always import NATIVE_S0_MODE
from benchmarks.memory_runtime.event_native_s0_policy import S0_CONFIG_DEFAULTS
from benchmarks.memory_runtime.tests.test_candidate_allocation import Tokenizer, packing, policy, messages, tool_pair
from benchmarks.memory_runtime.tests.test_candidate_recovery import Risk
from benchmarks.memory_runtime.tests.test_verified_binding import _call, _tools, _observed
from benchmarks.memory_runtime.budget_guard import history_budget_receipt
from benchmarks.memory_runtime.policy import PolicyInputError


def config(variant):
    return {"variant": variant, "risk_artifact": {"fixture": True},
            "risk_threshold": 0.5, **extension_fields(variant)}


def controller(variant="static_verified", *, score=0.1, tokenizer=None, **kwargs):
    geometry = {**packing(), "ratios": [8]}
    base = CandidateAllocator(tokenizer or Tokenizer(), packing=geometry,
                              policy=policy(4096), variant="static_t02")
    cls = StaticVerifiedController if variant == "static_verified" else StaticActionLedgerController
    return cls(base, config(variant), risk_model=Risk(score), **kwargs)


def payload(rows=None, tools=(), key="turn-0/step-0"):
    return {"session_id": "static-extension", "decision_key": key,
            "messages": copy.deepcopy(messages() if rows is None else rows), "tools": list(tools)}


@pytest.mark.parametrize("variant", STATIC_EXTENSION_VARIANTS)
@pytest.mark.parametrize("score", [0.1, 0.9])
def test_original_static_allocation_and_event_recovery_are_preserved(variant, score, monkeypatch):
    c = controller(variant, score=score)
    baseline = CandidateRecoveryController(controller().base, {"variant": "static_t02"}, risk_model=Risk(score))
    p = c.prepare(payload(), ratio=8, max_new_tokens=32)
    b = baseline.prepare(payload(), ratio=8, max_new_tokens=32)
    assert p.memory == b.memory
    event = next(iter(b.memory.view.gist_event_ids))
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.select_source_event",
                        lambda *args, **kwargs: (event, {"ranked_candidate_event_ids": [event]}))
    old = baseline.reconsider(b, [], draft_text="Done")
    result = c.reconsider(p, [], draft_text="Done")
    assert (result["memory"], result["regenerate"], result["decision"]["reason"]) == (
        old["memory"], old["regenerate"], old["decision"]["reason"])
    assert c.reconsider(p, [], draft_text="Done") == result
    assert result["metadata"]["candidate_algorithm"]["variant"] == variant
    assert result["decision"]["backbone"]["variant"] == "static_t02"
    assert history_budget_receipt(result["memory"], result["metadata"], c,
                                  ratio=8, phase="regeneration")["status"] == "passed"
    with pytest.raises(PolicyInputError, match="another held draft"):
        c.reconsider(p, [], draft_text="Different")


@pytest.mark.parametrize("variant", STATIC_EXTENSION_VARIANTS)
def test_factory_and_ready_contract_require_exact_policy_identity(variant, monkeypatch):
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.C1RiskArtifact",
                        lambda artifact: Risk(0.1))
    frozen = json.loads((ROOT / "configs" / "controller.json").read_text(encoding="utf8"))
    for key in ("gp_experiments", "post_draft_recovery", "d3_hybrid_recovery"):
        frozen.pop(key, None)
    c = build_event_native_controller(Tokenizer(), packing={**packing(), "ratios": [8]},
        policy=policy(4096), view_mode=NATIVE_S0_MODE, compression_policy=ALWAYS_COMPRESSION_POLICY,
        s0_config={**frozen, "candidate_algorithm": config(variant)})
    assert c.prepare(payload(), ratio=8, max_new_tokens=32).memory == controller().prepare(
        payload(), ratio=8, max_new_tokens=32).memory
    identity, baseline = _candidate_ready_contract(config(variant))
    assert baseline == STATIC_EXTENSION_VERSION + ":" + variant
    for key, value in extension_fields(variant).items():
        assert identity[key] == value
        bad = config(variant)
        bad.pop(key)
        with pytest.raises(ValueError, match="contract mismatch"):
            _candidate_ready_contract(bad)


def proof_payload():
    return payload([{"role": "user", "content":
        "Set a budget limit of $1500 using my secure token ABCDE12345."}],
        _tools(("set_budget_limit", {"access_token": "string", "budget_limit": "number"})))


def test_real_verified_field_proof_changes_only_the_abstained_selected_call():
    c = controller()
    p = c.prepare(proof_payload(), ratio=8, max_new_tokens=32)
    calls = [_call("set_budget_limit", {"access_token": "wrong", "budget_limit": 1500})]
    result = c.reconsider(p, calls, draft_text="call")
    assert not result["regenerate"] and result["memory"] == p.memory
    c.validate_commit(p, calls, draft_text="call")
    output, receipt = c.finalize_commit(p, calls)
    assert json.loads(output[0]["function"]["arguments"]) == {
        "access_token": "ABCDE12345", "budget_limit": 1500}
    assert output[0]["id"] == calls[0]["id"]
    assert receipt["changed"] and receipt["additional_generations"] == 0
    assert c.finalize_commit(p, calls) == (output, receipt)
    with pytest.raises(PolicyInputError, match="another selected draft"):
        c.finalize_commit(p, [])


@pytest.mark.parametrize("variant", STATIC_EXTENSION_VARIANTS)
def test_task_generation_cap_stays_a_hard_boundary(variant):
    c = controller(variant)
    p = c.prepare(proof_payload(), ratio=8, max_new_tokens=32)
    c._recovery_counts["static-extension"] = c.required_task_generation_limit
    c.risk_model.predict_risk = lambda context: pytest.fail("cap must precede scoring")
    assert c.reconsider(p, [], draft_text="Done")["decision"]["reason"] == "shared_task_generation_limit"


class DraftTokenizer(Tokenizer):
    def __init__(self, texts):
        self.texts = texts

    def decode(self, ids, **kwargs):
        return self.texts[ids[0] - 100]


class Generator:
    def __init__(self):
        self.calls = 0

    @contextmanager
    def decision_scope(self, *, session_id=None):
        yield

    def generate(self, memory, **kwargs):
        token = 100 + self.calls
        self.calls += 1
        return SimpleNamespace(token_ids=(token,), token_logprobs=(-0.1,),
                               finish_reason="stop", stats={"eos_token_ids": []})

    def session_cache_info(self):
        return {"status": "empty"}

    def close_session(self):
        pass


def run_drafts(variant, data, texts, tmp_path):
    tokenizer = DraftTokenizer(texts)
    c = controller(variant, tokenizer=tokenizer)
    runner = EventNativeDecisionRunner(c, Generator(), tokenizer, ratio=8, max_new_tokens=32,
        max_generation_calls=2, journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    return runner.run(data)


def test_runner_preserves_generated_text_and_cost_when_verified_commit_changes(tmp_path):
    text = '<tool_call>{"name":"set_budget_limit","arguments":{"access_token":"wrong","budget_limit":1500}}</tool_call>'
    record = run_drafts("static_verified", proof_payload(), [text], tmp_path)
    assert record["generation_completed"] == 1
    assert record["generation_trace"][0]["native_draft"]["text"] == text
    assert json.loads(record["response"]["tool_calls"][0]["function"]["arguments"])["access_token"] == "ABCDE12345"
    assert record["commit_transform"]["changed"]
    assert record["commit_transform"]["model_generation_unmodified"]


def ledger_payload(*, completed=False, fare=72):
    tools = _tools(("book_flight", {"travel_from": "string", "travel_to": "string"}))
    tools[0]["function"]["parameters"]["required"] = ["travel_from", "travel_to"]
    args = {"travel_from": "SFO", "travel_to": "LAX"}
    rows = [{"role": "user", "content":
        "Check the fare from SFO to LAX and, if it is under 100 USD, book the flight."},
        *_observed("get_flight_cost", args, {"fare": fare, "currency": "USD"}, "fare")]
    if completed:
        rows.extend(_observed("book_flight", args, {"success": True}, "booking"))
    return payload(rows, tools)


@pytest.mark.parametrize("second,accepted", [
    ('<tool_call>{"name":"book_flight","arguments":{"travel_from":"SFO","travel_to":"LAX"}}</tool_call>', True),
    ('<tool_call>{"name":"book_flight","arguments":{"travel_from":"LAX","travel_to":"SFO"}}</tool_call>', False),
    ("Done", False),
])
def test_stop_review_commits_only_the_exact_ready_action(tmp_path, second, accepted):
    record = run_drafts("static_action_ledger", ledger_payload(), ["Done", second], tmp_path)
    assert record["generation_completed"] == 2
    assert record["commit_validation"]["accepted"] is accepted
    assert bool(record["response"]["tool_calls"]) is accepted
    assert all(row["status"] == "passed" for row in record["pre_generation_budget_checks"])
    draft, revised = record["generation_trace"]
    for field in ("raw_event_ids", "gist_event_ids"):
        assert draft["controller"][field] == revised["controller"][field]
    if not accepted:
        assert record["response"]["content"] == "Done"


def test_false_condition_keeps_stop_without_extra_generation(tmp_path):
    record = run_drafts("static_action_ledger", ledger_payload(fare=150), ["Done"], tmp_path)
    assert record["generation_completed"] == 1
    assert record["response"]["content"] == "Done"


def test_completed_duplicate_is_removed_at_actual_runner_commit(tmp_path):
    text = '<tool_call>{"name":"book_flight","arguments":{"travel_from":"SFO","travel_to":"LAX"}}</tool_call>'
    record = run_drafts("static_action_ledger", ledger_payload(completed=True), [text], tmp_path)
    assert record["generation_completed"] == 1
    assert record["generation_trace"][0]["native_draft"]["text"] == text
    assert record["response"]["tool_calls"] == []
    assert record["response"]["native_parse_status"] == "text"
    assert record["commit_transform"]["changed"]
    assert record["commit_transform"]["removed"][0]["success_sources"]


def test_ledger_packet_cannot_evict_static_history_to_fit(monkeypatch):
    c = controller("static_action_ledger")
    p = c.prepare(ledger_payload(), ratio=8, max_new_tokens=32)
    monkeypatch.setattr(c.base, "_try_measure", lambda *args, **kwargs: None)
    result = c.reconsider(p, [], draft_text="Done")
    assert not result["regenerate"] and result["memory"] == p.memory
    assert result["decision"]["action_ledger"]["status"] == "packet_not_admitted_without_eviction"


def test_same_obligation_is_not_reviewed_again_after_only_prose_changes():
    c = controller("static_action_ledger")
    data = ledger_payload()
    p = c.prepare(data, ratio=8, max_new_tokens=32)
    assert c.reconsider(p, [], draft_text="Done")["regenerate"]
    data["messages"].append({"role": "assistant", "content": "Done"})
    data["decision_key"] = "turn-0/step-1"
    p = c.prepare(data, ratio=8, max_new_tokens=32)
    result = c.reconsider(p, [], draft_text="Done")
    assert not result["regenerate"]
    assert result["decision"]["action_ledger"]["status"] == "witness_already_reviewed"


def test_filtered_commit_preserves_other_actions_and_rejects_invention():
    from benchmarks.memory_runtime.candidate_algorithms.static_commit import render_filtered_commit
    from benchmarks.memory_runtime.event_native_draft import NativeDraft
    completed = _call("book_flight", {"travel_from": "SFO", "travel_to": "LAX"})
    retained = _call("lookup", {"key": 3}, "retained")
    draft = NativeDraft("original", "stale narration", (completed, retained), "tool_calls", "native")
    output = render_filtered_commit(draft, (retained,), benchmark="bfcl")
    assert output.tool_calls == (retained,)
    assert "book_flight" not in output.text
    assert render_filtered_commit(draft, (), benchmark="acebench").text == "Finish conversation"
    with pytest.raises(ValueError, match="remove unchanged calls"):
        render_filtered_commit(draft, (_call("invented", {}),), benchmark="bfcl")
