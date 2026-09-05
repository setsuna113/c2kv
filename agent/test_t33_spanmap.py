# -*- coding: utf-8 -*-
"""Unit tests for agent/t33_spanmap.py (survey item 4.0-3).

Pure stdlib — runs anywhere.  Run from the repo root:
  python -m pytest agent/test_t33_spanmap.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("agent",):
    _p = str(_REPO_ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from t33_spanmap import (  # noqa: E402
    char_range_to_token,
    parse_tool_call,
    spans_from_generation,
    token_char_offsets,
)

CLOSED = (
    "Let me check that.\n\nAction:\n<tool_call>\n"
    '{"name":"mcp__env__search_users","arguments":{"access_token":"tok_abc","page":4}}\n'
    "</tool_call>\n"
)

TRUNC_NO_CLOSE_BALANCED = (
    "Action:\n<tool_call>\n"
    '{"name":"mcp__env__search_friends","arguments":{"access_token":"tok_abc"}}'
)

TRUNC_JSON_UNTERMINATED = (
    "Action:\n<tool_call>\n"
    '{"name":"mcp__env__search_users","arguments":{"access_token":"eyJhbGciOi'
)

NO_CALL = "I have enough information to answer directly. The answer is 42."


def test_closed_call_parses_strictly():
    p = parse_tool_call(CLOSED)
    assert p["has_tool_call"] and p["closed"] and p["parse_ok"]
    assert p["name"] == "mcp__env__search_users"
    assert p["arguments"] == {"access_token": "tok_abc", "page": 4}
    name_cs = CLOSED.index("mcp__env__search_users")
    assert p["name_span"] == (name_cs, name_cs + len("mcp__env__search_users"))
    assert p["json_error"] is None


def test_balanced_json_without_closing_tag_is_parseable_and_unclosed():
    p = parse_tool_call(TRUNC_NO_CLOSE_BALANCED)
    assert p["has_tool_call"] and not p["closed"]
    assert p["parse_ok"] and p["name"] == "mcp__env__search_friends"
    assert p["json_error"] is None


def test_unterminated_json_is_lenient_not_dropped():
    p = parse_tool_call(TRUNC_JSON_UNTERMINATED)
    assert p["has_tool_call"] and not p["closed"]
    assert not p["parse_ok"]
    assert p["json_error"] == "unterminated_json"
    # the name span must still be recorded (regex path)
    assert p["name_span"] is not None
    assert TRUNC_JSON_UNTERMINATED[p["name_span"][0]:p["name_span"][1]] == "mcp__env__search_users"
    assert p["args_span"] is not None


def test_no_tool_call():
    p = parse_tool_call(NO_CALL)
    assert not p["has_tool_call"] and not p["parse_ok"]
    assert p["name_span"] is None and p["json_span"] is None


def test_object_without_name_fails_strictly():
    p = parse_tool_call('<tool_call>\n{"arguments":{"a":1}}')
    assert p["has_tool_call"] and not p["parse_ok"]
    assert p["json_error"] == "name_missing_or_not_str"


def _fake_decoder(tokens):
    def decode(ids):
        return "".join(tokens[i] for i in ids)
    return decode


def test_token_char_offsets_exact():
    tokens = ["Hello", " ", "wor", "ld"]
    offs = token_char_offsets(_fake_decoder(tokens), list(range(4)))
    assert offs == [(0, 5), (5, 6), (6, 9), (9, 11)]


def test_char_range_to_token():
    offs = [(0, 5), (5, 6), (6, 9), (9, 11)]
    assert char_range_to_token(offs, 0, 5) == (0, 0)
    assert char_range_to_token(offs, 6, 11) == (2, 3)
    assert char_range_to_token(offs, 3, 8) == (0, 2)
    assert char_range_to_token(offs, 99, 100) is None


def test_spans_from_generation_end_to_end():
    pieces = ["Let me ", "check that.\n\nAction:\n", "<tool_call>\n", '{"name":"mcp__env__',
              'search_users","arguments":', '{"access_token":', '"tok_abc"', "}}\n",
              "</tool_call>\n"]
    decode = _fake_decoder(pieces)
    ids = list(range(len(pieces)))
    rec = spans_from_generation(decode, ids)
    expected_text = "".join(pieces)
    assert rec["n_generated"] == len(pieces)
    assert rec["text"] == expected_text
    assert rec["parse_ok"] and rec["closed"]
    # name value chars live across tokens 3..4 ('{"name":"mcp__env__' contains the
    # value start; 'search_users",...' contains the rest)
    assert rec["name_first"] == 3 and rec["name_last"] == 4
    assert rec["args_first"] is not None and rec["args_last"] is not None
    assert rec["last_tok"] == len(pieces) - 1
    assert rec["penult_tok"] == len(pieces) - 2


def test_spans_from_generation_unterminated_records_none_not_crash():
    # name string is closed, arguments object is unterminated (cap censoring)
    pieces = ["Action:\n<tool_call>\n", '{"name":"mcp__', '_x"', ',"arguments":{"a":"yy']
    rec = spans_from_generation(_fake_decoder(pieces), list(range(len(pieces))))
    assert not rec["closed"] and not rec["parse_ok"]
    assert rec["name_first"] is not None  # lenient span still mapped
    assert rec["args_first"] is not None
