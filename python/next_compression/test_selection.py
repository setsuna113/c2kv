import copy

import pytest


def test_malformed_function_payload_is_rejected():
    from next_compression.selection import parse_calls
    with pytest.raises(ValueError):
        parse_calls('<tool_call>{"function":"invalid"}</tool_call>')

from next_compression.selection import choose_candidates, parse_calls, score_records


def rows(correct=True):
    call = '<tool_call>{"name":"f","arguments":{"id":"a"}}</tool_call>'
    return [{"decision_id": key, "ratio": ratio, "target_text": target,
             "generated_text": target if correct else "wrong", "uniform_ce": 2.0}
            for ratio in (8, 12) for key, target in (("call", call), ("text", "Done"))]


def test_parser_preserves_json_strings_and_rejects_partial_call():
    text = '<tool_call>{"name":"f","arguments":{"text":"</tool_call>"}}</tool_call>'
    assert parse_calls(text)[0]["arguments"]["text"] == "</tool_call>"
    with pytest.raises(ValueError):
        parse_calls(text[:-3])


def test_wrong_arguments_and_false_call_cannot_be_reported_as_success():
    values = rows()
    values[0]["generated_text"] = values[0]["target_text"].replace('"a"', '"b"')
    values[1]["generated_text"] = values[0]["target_text"]
    result = score_records(values[:2])
    assert result["strict_ordered_tool_call_accuracy"] == 0
    assert result["false_tool_call_rate"] == 1


def test_selection_keeps_ratio_winner_and_final_and_rejects_protocol_drift():
    entries = [{"variant": "H1", "checkpoint": f"checkpoint-{step}", "step": step,
                "records": rows(correct), "contract": {"dev": "frozen", "max_new_tokens": 512}}
               for step, correct in ((500, True), (726, False))]
    result = choose_candidates(entries)["H1"]["return_candidates"]
    assert {item["checkpoint"] for item in result} == {"checkpoint-500", "checkpoint-726"}
    altered = copy.deepcopy(entries)
    altered[1]["contract"]["max_new_tokens"] = 1024
    with pytest.raises(ValueError, match="protocols"):
        choose_candidates(altered)


def test_explicit_gold_calls_override_quoted_examples_in_prose():
    values = rows()[:2]
    values[1]["target_text"] = 'Example: <tool_call> placeholder </tool_call>'
    values[1]["gold_tool_calls"] = []
    assert score_records(values)["tool_decisions"] == 1
