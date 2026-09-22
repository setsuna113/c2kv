import copy

import pytest

from benchmarks.raw_actor_history import RawActorHistory
from benchmarks.bfcl_response import normalize_native_message


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


@pytest.mark.parametrize("text", [
    'I will inspect the file.\n\n<tool_call>\n{"name":"head","arguments":{"file_name":"report.txt","lines":1}}\n</tool_call>',
    '<tool_call>{"name":"head","arguments":{"lines":1}}</tool_call>\n'
    '<tool_call>{"name":"sort","arguments":{}}</tool_call>',
])
def test_bfcl_native_echo_restores_raw_text_and_rejects_changed_actions(text):
    state = RawActorHistory()
    source = [{"role": "user", "content": "task"}]
    message = {"role": "assistant", "content": text, "tool_calls": []}
    reply = dict(response(message, text), id="fixture-response")
    frozen = copy.deepcopy(reply)
    state.commit(source, reply, benchmark="bfcl")
    echo = normalize_native_message(message, reply["id"])
    assert echo["tool_calls"][0]["id"] == "bfcl_native_fixture-response_0"
    prepared = state.prepare([*source, echo, {"role": "tool", "content": "result"}])
    assert prepared[1] == {"role": "assistant", "content": text}
    assert reply == frozen
    for field, value in (("content", "Changed prose"), ("id", "different-id"),
                         ("name", "different_tool"), ("arguments", '{"lines":2}')):
        changed = copy.deepcopy(echo)
        if field == "content":
            changed[field] = value
        elif field == "id":
            changed["tool_calls"][0][field] = value
        else:
            changed["tool_calls"][0]["function"][field] = value
        with pytest.raises(ValueError, match="unchanged"):
            state.prepare([*source, changed])


def test_bfcl_malformed_draft_is_not_repaired_and_other_benchmarks_keep_raw_echo():
    text = '<tool_call>{"name":"head","arguments":{"lines":1}}'
    message = {"role": "assistant", "content": text}
    assert normalize_native_message(message, "r") == message
    text += '</tool_call>'
    message = dict(message, content=text)
    state = RawActorHistory()
    state.commit([], dict(response(message, text), id="r"), benchmark="acebench")
    assert state.prepare([message]) == [message]
    with pytest.raises(ValueError, match="unchanged"):
        state.prepare([normalize_native_message(message, "r")])


def test_toolsandbox_mixed_text_tool_call_accepts_only_canonical_action_echo():
    state = RawActorHistory()
    source = [{"role": "user", "content": "Turn off low battery mode."}]
    assistant = {"role": "assistant", "content": "I'll turn it off.", "tool_calls": [{
        "id": "call-1", "type": "function",
        "function": {"name": "set_low_battery_mode_status",
                     "arguments": '{"on":false}'},
    }]}
    raw_text = ("I'll turn it off.\n\n<tool_call>\n"
                '{"name":"set_low_battery_mode_status","arguments":{"on":false}}\n'
                "</tool_call>")
    state.commit(source, response(assistant, raw_text), benchmark="toolsandbox")

    # ToolSandbox stores the executable call and reconstructs this canonical
    # assistant message, so mixed prose from the response is unavailable.
    echo = copy.deepcopy(assistant)
    echo["content"] = ""
    echo["tool_calls"][0]["function"]["arguments"] = '{ "on" : false }'
    prepared = state.prepare([*source, echo, {
        "role": "tool", "tool_call_id": "call-1", "content": "None",
    }])
    assert prepared[1] == {"role": "assistant", "content": raw_text}

    for field, value in (("content", "changed prose"), ("id", "call-2"),
                         ("name", "set_wifi_status"),
                         ("arguments", '{"on":true}')):
        changed = copy.deepcopy(echo)
        if field == "content":
            changed["content"] = value
        elif field == "id":
            changed["tool_calls"][0][field] = value
        else:
            changed["tool_calls"][0]["function"][field] = value
        with pytest.raises(ValueError, match="unchanged"):
            state.prepare([*source, changed])


def test_toolsandbox_plain_assistant_text_must_stay_unchanged():
    state = RawActorHistory()
    source = [{"role": "user", "content": "What happened?"}]
    assistant = {"role": "assistant", "content": "Location service is disabled."}
    state.commit(source, response(assistant, assistant["content"]),
                 benchmark="toolsandbox")
    assert state.prepare([*source, copy.deepcopy(assistant)])[1] == assistant
    with pytest.raises(ValueError, match="unchanged"):
        state.prepare([*source, {"role": "assistant", "content": "It is enabled."}])
