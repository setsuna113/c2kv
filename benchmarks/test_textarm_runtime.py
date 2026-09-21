"""Exercise internal HiAgent retrieval without executing its meta-tool outside the proxy."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy
import textarms
from arms import get_arm


def _original():
    return {"model": "c2kv-agent", "messages": [
        {"role": "user", "content": "Compare the two accounts."},
        {"role": "assistant", "content": "Subgoal: inspect first account"},
        {"role": "tool", "content": "account A secret detail: 193", "tool_call_id": "call_a"},
        {"role": "assistant", "content": "Subgoal: inspect second account"},
        {"role": "user", "content": "Now compare."},
    ]}


def _retrieval(ids):
    return {"choices": [{"message": {"role": "assistant", "tool_calls": [
        {"id": "meta", "type": "function", "function": {
            "name": textarms.HIAGENT_RETRIEVE_TOOL_NAME,
            "arguments": json.dumps({"subgoal_ids": ids})}}]}}],
        "usage": {"prompt_tokens": 71, "completion_tokens": 9}}


def test_retrieval_reveals_trajectory_and_charges_extra_generation(monkeypatch):
    textarms.reset_state()
    monkeypatch.setattr(proxy, "_textarm_compress", lambda *a, **k: "Account inspected.")
    original = _original()
    arm = get_arm("hiagent_full")
    staged, stats = proxy._apply_text_arm(original, arm, "test")
    assert "193" not in json.dumps(staged["messages"])
    assert staged["tools"][-1]["function"]["name"] == textarms.HIAGENT_RETRIEVE_TOOL_NAME
    sent = []
    final = {"choices": [{"message": {"role": "assistant", "content": "A has 193."}}]}
    def send(payload):
        sent.append(payload)
        return final
    result = proxy._hiagent_retrieval_loop(original, arm, "test", _retrieval([1]), stats, send)
    assert result == final
    assert len(sent) == 1 and "193" in json.dumps(sent[0]["messages"])
    assert stats["retrieval_usage"] == {"calls": 1, "prompt_tokens": 71, "completion_tokens": 9}
    assert stats["retrieved_subgoals"] == [1]
    assert "tools" not in original


def test_repeated_revealed_trajectory_gets_bounded_feedback_and_remains_metered(
        monkeypatch):
    textarms.reset_state()
    monkeypatch.setattr(proxy, "_textarm_compress", lambda *a, **k: "Account inspected.")
    original = _original()
    arm = get_arm("hiagent_full")
    _, stats = proxy._apply_text_arm(original, arm, "test")
    sent = []
    final = {"choices": [{"message": {"role": "assistant", "content": "A has 193."}}]}

    def send(payload):
        sent.append(payload)
        return _retrieval([1]) if len(sent) == 1 else final

    result = proxy._hiagent_retrieval_loop(
        original, arm, "test", _retrieval([1]), stats, send)

    assert result is final
    assert len(sent) == 2
    assert all(json.dumps(payload["messages"]).count("account A secret detail: 193") == 1
               for payload in sent)
    assert "already_revealed" in json.dumps(sent[1]["messages"])
    assert stats["retrieval_usage"] == {
        "calls": 2, "prompt_tokens": 142, "completion_tokens": 18}
    assert stats["duplicate_retrieval_attempts"] == [{
        "requested_subgoals": [1],
        "already_revealed_subgoals": [1],
        "unavailable_subgoals": [],
        "feedback_reason": "already_revealed",
    }]
    assert stats["retrieved_subgoals"] == [1]
    assert "tools" not in original


def test_repeated_retrieval_does_not_expand_four_round_limit(monkeypatch):
    arm = get_arm("hiagent_full")

    def apply(payload, arm, conv, ids=None, retrieval_feedback=None):
        return {"messages": [{"role": "user", "content": retrieval_feedback or "revealed"}]}, {
            "retrieved_subgoals": [1], "invalid_retrieval_subgoals": [],
            "n_compressor_calls": 0, "compressor_usage": {},
        }

    sent = []
    stats = {"n_compressor_calls": 0, "compressor_usage": {}}
    monkeypatch.setattr(proxy, "_apply_text_arm", apply)
    with pytest.raises(ValueError, match="exceeded four internal trajectory retrieval rounds"):
        proxy._hiagent_retrieval_loop(
            {}, arm, "test", _retrieval([1]), stats,
            lambda payload: sent.append(payload) or _retrieval([1]))
    assert len(sent) == 4
    assert stats["retrieval_usage"] == {
        "calls": 5, "prompt_tokens": 355, "completion_tokens": 45}
    assert len(stats["duplicate_retrieval_attempts"]) == 4


def test_nonexistent_retrieval_is_not_leaked_to_executor(monkeypatch):
    textarms.reset_state()
    monkeypatch.setattr(proxy, "_textarm_compress", lambda *a, **k: "Account inspected.")
    original = _original()
    arm = get_arm("hiagent_full")
    _, stats = proxy._apply_text_arm(original, arm, "test")
    sent = []
    final = {"choices": [{"message": {"role": "assistant", "content": "Continue."}}]}
    def send(payload):
        sent.append(payload)
        return final
    assert proxy._hiagent_retrieval_loop(
        original, arm, "test", _retrieval([99]), stats, send) is final
    assert len(sent) == 1
    assert "subgoal_unavailable" in json.dumps(sent[0]["messages"])
    assert "account A secret detail" not in json.dumps(sent[0]["messages"])
    assert stats["invalid_retrieval_attempts"] == [{
        "requested_subgoals": [99], "invalid_subgoals": [99]}]


@pytest.mark.parametrize("policy,mode,guideline", [
    ("acon_hist_ut_co", "hist", "ut_co"), ("acon_obs_ut", "obs", "ut"),
    ("acon_hist", "hist", "base")])
def test_acon_guideline_reaches_compressor(monkeypatch, policy, mode, guideline):
    seen = {}
    def transform(messages, *args, **kwargs):
        seen.update(kwargs)
        return messages, {}
    monkeypatch.setattr(textarms, "acon_transform", transform)
    proxy._apply_text_arm(_original(), get_arm(policy), "test")
    assert seen["mode"] == mode and seen["guideline"] == guideline
