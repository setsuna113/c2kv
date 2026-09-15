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
    return {"choices": [{"message": {"role": "assistant",
        "content": "Subgoal: compare the two accounts", "tool_calls": [
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
    final_call = {"id": "environment", "type": "function", "function": {
        "name": "report_comparison", "arguments": json.dumps({"value": 193})}}
    final = {"choices": [{"message": {"role": "assistant",
        "content": "Subgoal: report the comparison",
        "tool_calls": [final_call]}}]}

    def send(payload):
        sent.append(payload)
        return final

    result = proxy._hiagent_retrieval_loop(original, arm, "test", _retrieval([1]), stats, send)
    assert result == final
    assert len(sent) == 1 and "193" in json.dumps(sent[0]["messages"])
    assert result["choices"][0]["message"]["tool_calls"] == [final_call]
    assert stats["retrieval_usage"] == {"calls": 1, "prompt_tokens": 71, "completion_tokens": 9}
    assert stats["retrieved_subgoals"] == [1]
    assert "tools" not in original


def test_nonexistent_retrieval_is_not_leaked_to_executor(monkeypatch):
    textarms.reset_state()
    monkeypatch.setattr(proxy, "_textarm_compress", lambda *a, **k: "Account inspected.")
    original = _original()
    arm = get_arm("hiagent_full")
    _, stats = proxy._apply_text_arm(original, arm, "test")
    with pytest.raises(ValueError, match="nonexistent"):
        proxy._hiagent_retrieval_loop(original, arm, "test", _retrieval([99]), stats,
                                      lambda payload: pytest.fail("invalid retrieval must not generate"))


def test_hiagent_full_native_preserves_current_and_retrieved_tool_rows(monkeypatch):
    textarms.reset_state()
    summaries = []

    def compress(payload, *args, **kwargs):
        summaries.append(payload)
        return "First account inspected."

    monkeypatch.setattr(proxy, "_textarm_compress", compress)
    source_tools = [{
        "type": "function",
        "function": {
            "name": "inspect_account",
            "description": "Inspect one account.",
            "parameters": {
                "type": "object",
                "properties": {"account": {"type": "string"}},
                "required": ["account"],
            },
        },
    }]
    messages = [
        {"role": "system", "content": "System contract."},
        {"role": "user", "content": "Compare the accounts."},
        {"role": "assistant", "content": "Subgoal: inspect first account", "tool_calls": [{
            "id": "call_first", "type": "function", "function": {
                "name": "inspect_account", "arguments": '{"account":"first"}'},
        }]},
        {"role": "tool", "name": "inspect_account", "tool_call_id": "call_first",
         "content": "first=193"},
        {"role": "assistant", "content": "Subgoal: inspect second account", "tool_calls": [{
            "id": "call_second", "type": "function", "function": {
                "name": "inspect_account", "arguments": '{"account":"second"}'},
        }]},
        {"role": "tool", "name": "inspect_account", "tool_call_id": "call_second",
         "content": "second=211"},
        {"role": "user", "content": "Report the comparison."},
    ]
    original = {"model": "c2kv-agent", "messages": messages, "tools": source_tools}
    original_snapshot = json.loads(json.dumps(original))
    arm = get_arm("hiagent_full_native")

    assert arm.text_policy == "hiagent_full"
    assert arm.compress_history is False
    assert arm.native_messages is True
    assert get_arm("hiagent_full").native_messages is False

    summarized, stats = proxy._apply_text_arm(original, arm, "test")
    assembled, counts = proxy._assemble(summarized["messages"], arm)

    assert len(summaries) == 1
    assert stats["n_compressor_calls"] == 1
    assert any(message.get("role") == "user"
               and message.get("content") ==
               "Subgoal 1: inspect first account\nSummary: First account inspected."
               for message in assembled)
    assert not any(message.get("tool_call_id") == "call_first" for message in assembled)
    current_rows = [message for message in messages if message.get("tool_call_id") == "call_second"
                    or any(call.get("id") == "call_second"
                           for call in message.get("tool_calls") or [])]
    current_start = next(index for index, message in enumerate(assembled)
                         if any(call.get("id") == "call_second"
                                for call in message.get("tool_calls") or []))
    assert assembled[current_start:current_start + 2] == current_rows
    assert assembled[-1] == messages[-1]
    assert counts["doc_packing"] == "native"
    assert counts["gist_tokens"] == 0
    assert summarized["tools"][:-1] == source_tools
    assert summarized["tools"][-1]["function"]["name"] == "hiagent_retrieve"

    retrieved, retrieved_stats = proxy._apply_text_arm(
        original, arm, "test", retrieve_subgoals=[1])
    retrieved_assembled, retrieved_counts = proxy._assemble(retrieved["messages"], arm)
    retrieved_start = next(index for index, message in enumerate(retrieved_assembled)
                           if any(call.get("id") == "call_first"
                                  for call in message.get("tool_calls") or []))
    assert retrieved_assembled[retrieved_start:retrieved_start + 2] == messages[2:4]
    assert retrieved_stats["retrieved_subgoals"] == [1]
    assert retrieved_stats["n_compressor_calls"] == 0
    assert retrieved_counts["doc_packing"] == "native"
    assert original == original_snapshot


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
