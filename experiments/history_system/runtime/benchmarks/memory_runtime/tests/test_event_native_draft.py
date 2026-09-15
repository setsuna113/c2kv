"""Native action parsing preserves the complete model-produced draft."""
import json
from types import SimpleNamespace

import pytest

from benchmarks.memory_runtime.event_native_draft import decode_native_generation, parse_native_draft


def parse(text):
    return parse_native_draft(text, call_id_prefix='d3_r0')


def block(name='unverified_tool', arguments=None):
    return '<tool_call>' + json.dumps({'name': name, 'arguments': arguments or {'path': 'x.txt'}}) + '</tool_call>'


def test_parallel_native_calls_keep_order_values_and_unique_ids():
    result = parse('Before. ' + block(arguments={'x':'item-17','n':4,'nested':{'ok':True}})
                   + '\n' + block('second', {'path':'folder/y'}) + ' After.')
    assert result.status == 'tool_calls' and len(result.tool_calls) == 2
    assert [call['id'] for call in result.tool_calls] == ['d3_r0_0', 'd3_r0_1']
    assert [call['function']['name'] for call in result.tool_calls] == ['unverified_tool', 'second']
    assert json.loads(result.tool_calls[0]['function']['arguments']) == {'x':'item-17','n':4,'nested':{'ok':True}}
    assert result.content == 'Before. \n After.'


@pytest.mark.parametrize('suffix', [
    '<tool_call>{"name":"bad","arguments":{}}',
    '<tool_call>{"name":"bad","arguments":{"x":1,"x":2}}</tool_call>',
    '<tool_call>{"name":"bad","arguments":{"x":NaN}}</tool_call>',
    '<tool_call>{"name":"bad","arguments":[]}</tool_call>',
    '</tool_call>',
])
def test_invalid_later_block_never_returns_partial_executable_calls(suffix):
    result = parse(block() + suffix)
    assert result.status == 'malformed' and result.tool_calls == ()
    assert result.text == block() + suffix


def test_plain_json_and_thinking_are_not_native_actions():
    plain = parse('{"name":"read","arguments":{"path":"x"}}')
    assert plain.status == 'text' and plain.tool_calls == ()
    text = '<think>' + block('imagined') + '</think>\n' + block('actual')
    result = parse(text)
    assert len(result.tool_calls) == 1 and result.tool_calls[0]['function']['name'] == 'actual'
    assert result.reasoning_content == block('imagined')
    assert parse('<think>' + block()).status == 'malformed'


def test_decode_removes_terminal_eos_without_dropping_tool_protocol_tokens():
    class Tokenizer:
        def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            assert ids == [11, 12]
            assert skip_special_tokens is False and clean_up_tokenization_spaces is False
            return block()
    result = SimpleNamespace(token_ids=(11,12,99), finish_reason='stop', stats={'eos_token_ids':[99]})
    assert decode_native_generation(Tokenizer(), result, call_id_prefix='d1_r0').status == 'tool_calls'
