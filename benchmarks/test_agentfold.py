import copy

import pytest

from benchmarks import agentfold


def response(content, calls=None):
    return {"choices": [{"message": {"role": "assistant", "content": content,
                                     "tool_calls": calls or []}}],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10}}


def test_actor_selects_deep_range_without_reloading_original_events():
    state = agentfold.State()
    messages = [{"role": "user", "content": "Find the answer"}]
    agentfold.prepare({"messages": messages}, state, code_actions=True)
    first = agentfold.finish(response("print('secret first action')"), state, code_actions=True)
    messages += [first["choices"][0]["message"], {"role": "user", "content": "SECRET_RAW_0"}]
    staged, _ = agentfold.prepare({"messages": messages}, state, code_actions=True)
    assert "SECRET_RAW_0" in str(staged)
    second = agentfold.finish(response('<compress>{"compress_range":[0,0],"compress_text":"summary 0"}</compress>print(2)'), state, code_actions=True)
    assert second["choices"][0]["message"]["content"] == "print(2)"
    assert second["usage"] == first["usage"]
    messages += [second["choices"][0]["message"], {"role": "user", "content": "SECRET_RAW_1"}]
    staged, _ = agentfold.prepare({"messages": messages}, state, code_actions=True)
    assert "SECRET_RAW_0" not in str(staged)
    third = agentfold.finish(response('<compress>{"compress_range":[0,1],"compress_text":"merged"}</compress>print(3)'), state, code_actions=True)
    assert state.steps == [{"start": 0, "end": 1, "content": "merged"}]
    messages += [third["choices"][0]["message"], {"role": "user", "content": "done"}]
    staged, _ = agentfold.prepare({"messages": messages}, state, code_actions=True)
    assert "SECRET_RAW" not in str(staged)


def test_parallel_tool_batch_is_one_action_and_requires_all_feedback():
    state = agentfold.State()
    messages = [{"role": "user", "content": "query"}]
    agentfold.prepare({"messages": messages}, state)
    calls = [{"id": f"c{i}", "type": "function", "function": {
        "name": "lookup", "arguments": "{}"}} for i in range(2)]
    data = agentfold.finish(response("", calls), state)
    messages += [data["choices"][0]["message"],
                 {"role": "tool", "tool_call_id": "c0", "content": "a"}]
    with pytest.raises(ValueError, match="all results"):
        agentfold.prepare({"messages": messages}, state)
    messages += [{"role": "tool", "tool_call_id": "c1", "content": "b"}]
    agentfold.prepare({"messages": messages}, state)
    assert state.next_step == 1
    with pytest.raises(ValueError, match="exactly one"):
        agentfold.finish(response("no directive"), state)
    assert state.steps[0]["end"] == 0


def test_native_tool_ids_survive_multiple_turns_and_folded_raw_history_stays_hidden():
    state = agentfold.State()
    messages = [{"role": "user", "content": "query"}]
    agentfold.prepare({"messages": messages}, state)
    first_calls = [{"id": "call-0", "type": "function", "function": {
        "name": "lookup", "arguments": '{"q":"SECRET_NATIVE_0"}'}}]
    first = agentfold.finish(response(None, first_calls), state)
    assert first["choices"][0]["message"]["tool_calls"] == first_calls
    messages += [first["choices"][0]["message"],
                 {"role": "tool", "tool_call_id": "call-0",
                  "content": "SECRET_RESULT_0"}]
    staged, _ = agentfold.prepare({"messages": messages}, state)
    assert "call-0" in str(staged)
    assert "SECRET_RESULT_0" in str(staged)

    second_calls = [{"id": "call-1", "type": "function", "function": {
        "name": "lookup", "arguments": '{"q":"next"}'}}]
    second = agentfold.finish(response(
        '<compress>{"compress_range":[0,0],"compress_text":"kept call-0 outcome"}</compress>',
        second_calls), state)
    assert second["choices"][0]["message"]["tool_calls"] == second_calls
    messages += [second["choices"][0]["message"],
                 {"role": "tool", "tool_call_id": "call-1",
                  "content": "SECRET_RESULT_1"}]
    staged, _ = agentfold.prepare({"messages": messages}, state)
    assert "SECRET_NATIVE_0" not in str(staged)
    assert "SECRET_RESULT_0" not in str(staged)
    assert "kept call-0 outcome" in str(staged)
    assert "call-1" in str(staged)
    assert "SECRET_RESULT_1" in str(staged)


def test_tool_batch_ids_are_complete_unique_and_prepare_is_transactional():
    for calls, error in [
        ([{"type": "function", "function": {"name": "x", "arguments": "{}"}}],
         "every native tool call ID"),
        ([{"id": "same", "type": "function", "function": {"name": "x", "arguments": "{}"}},
          {"id": "same", "type": "function", "function": {"name": "y", "arguments": "{}"}}],
         "unique native tool call IDs"),
    ]:
        state = agentfold.State()
        agentfold.prepare({"messages": [{"role": "user", "content": "q"}]}, state)
        before = copy.deepcopy(state)
        with pytest.raises(ValueError, match=error):
            agentfold.finish(response(None, calls), state)
        assert state == before

    state = agentfold.State()
    messages = [{"role": "user", "content": "q"}]
    agentfold.prepare({"messages": messages}, state)
    calls = [{"id": "c0", "type": "function", "function": {
        "name": "x", "arguments": "{}"}},
        {"id": "c1", "type": "function", "function": {
            "name": "y", "arguments": "{}"}}]
    returned = agentfold.finish(response(None, calls), state)
    before = copy.deepcopy(state)
    malformed = messages + [returned["choices"][0]["message"],
        {"role": "tool", "tool_call_id": "c0", "content": "a"},
        {"role": "tool", "tool_call_id": "c0", "content": "duplicate"},
        {"role": "tool", "tool_call_id": "c1", "content": "b"}]
    with pytest.raises(ValueError, match="all results"):
        agentfold.prepare({"messages": malformed}, state)
    assert state == before


def test_invalid_fold_is_fail_closed_without_partial_state_advance():
    state = agentfold.State(
        steps=[{"start": 0, "end": 0, "content": "raw"}],
        next_step=1,
    )
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="JSON object"):
        agentfold.finish(response("<compress>[]</compress>print(1)"), state,
                         code_actions=True)
    assert state == before
    with pytest.raises(ValueError, match="must precede AppWorld code"):
        agentfold.finish(response(
            'print(1)<compress>{"compress_range":[0,0],"compress_text":"s"}</compress>'),
            state, code_actions=True)
    assert state == before


def test_cannot_fold_inside_an_existing_summary_or_use_future_range():
    state = agentfold.State(steps=[{"start": 0, "end": 2, "content": "summary"},
                                   {"start": 3, "end": 3, "content": "latest"}])
    for bounds in ([1, 3], [0, 4], [0, 2]):
        import json
        text = '<compress>' + json.dumps({"compress_range": bounds,
                                         "compress_text": "bad"}) + '</compress>'
        with pytest.raises(ValueError, match="whole suffix"):
            agentfold.finish(response(text), state)
