import copy

import pytest

from benchmarks.raw_actor_history import RawActorHistory


def response(message, text):
    return {"choices": [{"message": message}], "metadata": {
        "persistent_history_session": {"continuation_mode": "exact_generated_prefix",
                                       "generated_text": text}}}


def test_structured_echo_uses_original_actor_text_without_mutating_harness_history():
    state = RawActorHistory()
    source = [{"role": "user", "content": "task"}]
    assistant = {"role": "assistant", "content": None, "tool_calls": [{
        "id": "c1", "function": {"name": "lookup", "arguments": '{"b":2,"a":1}'}}]}
    original_text = '<tool_call>{"name":"lookup","arguments":{"b":2,"a":1}}</tool_call>'
    state.commit(source, response(assistant, original_text))
    echo = copy.deepcopy(assistant)
    echo["tool_calls"][0]["function"]["arguments"] = '{ "a": 1, "b": 2 }'
    next_source = [*source, echo, {"role": "tool", "content": "result", "tool_call_id": "c1"}]
    prepared = state.prepare(next_source)
    assert prepared[1] == {"role": "assistant", "content": original_text}
    assert "tool_calls" in next_source[1]
    state.commit(next_source, response({"role": "assistant", "content": "done"}, "done"))
    third = state.prepare([*next_source, {"role": "assistant", "content": "done"},
                           {"role": "user", "content": "continue"}])
    assert third[1]["content"] == original_text
    assert third[3]["content"] == "done"


def test_modified_action_and_missing_receipt_are_rejected():
    state = RawActorHistory()
    with pytest.raises(ValueError, match="receipt"):
        state.commit([], {"choices": [{"message": {"content": "answer"}}]})
    source = [{"role": "user", "content": "task"}]
    state.commit(source, response({"role": "assistant", "content": "answer"}, "answer"))
    with pytest.raises(ValueError, match="unchanged"):
        state.prepare([*source, {"role": "assistant", "content": "changed"}])
    assert state.prepare([*source, {"role": "assistant", "content": "answer"}])[1]["content"] == "answer"
