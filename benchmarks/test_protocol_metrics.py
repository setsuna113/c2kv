"""Protocol validation must account for every emitted call fragment."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import protocol_columns_for_turn


TOOLS = [{"type": "function", "function": {
    "name": "lookup", "parameters": {"type": "object", "properties": {}}
}}]


def test_valid_call_does_not_hide_unterminated_second_call():
    result = protocol_columns_for_turn({"content": (
        '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
        '<tool_call>{"name":"lookup"'
    )}, TOOLS)
    assert result["n_tool_calls"] == 1
    assert result["protocol_legal"] is False
    assert result["first_violation"] == "unterminated <tool_call> syntax"


def test_complete_call_and_plain_turn_end_remain_legal():
    result = protocol_columns_for_turn({"content": (
        '<tool_call>{"name":"lookup","arguments":{}}</tool_call>Done.'
    )}, TOOLS)
    assert result["protocol_legal"] is True
    assert protocol_columns_for_turn({"content": "Done."}, TOOLS)["protocol_legal"] is True


def test_unknown_tool_pool_does_not_hide_broken_syntax():
    complete = '<tool_call>{"name":"lookup","arguments":{}}</tool_call>'
    assert protocol_columns_for_turn({"content": complete}, [])["protocol_legal"] is None
    broken = protocol_columns_for_turn({"content": complete + '<tool_call>{'}, [])
    assert broken["protocol_legal"] is False
    assert broken["first_violation"] == "unterminated <tool_call> syntax"
