# -*- coding: utf-8 -*-
"""CPU unit test for agent/d_strict_metric.py v2 (B14-B17).

Pins the four fixed behaviors:
- B14: the name comes from the harness parser chain (unclosed-block
  fallback recovers names the old closed-tag-only parser dropped), so
  strict ⊆ tool_name_match by construction
- B16: a truncated FIRST block must not swallow a well-formed second block
- B15: "arguments unparseable" vs "arguments null" are separate states;
  null-vs-null matches, unparseable-vs-anything never matches
- B17: files are read as utf-8 (ensure_ascii=False round-trip)

Run:  pytest metrology/test_strict_metric_v2.py -v   (torch-free)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "agent"))

from d_strict_metric import (  # noqa: E402
    ARGS_NULL,
    ARGS_OK,
    ARGS_UNPARSEABLE,
    compute_arm,
    parse_tool_call,
    strict_action_match,
)

GOOD = '{"name": "book_flight", "arguments": {"city": "Paris", "n": 2}}'
UNCLOSED = '<tool_call>\n{"name": "book_flight", "arguments": {"city": "Par'


class TestNameFromHarnessParser:
    def test_unclosed_block_name_recovered(self):
        parsed = parse_tool_call(f"{UNCLOSED}")
        assert parsed is not None and parsed["name"] == "book_flight"
        # arguments unverifiable -> strict must NOT claim a match
        assert not strict_action_match(f"{UNCLOSED}", f"<tool_call>{GOOD}</tool_call>")

    def test_closed_block_full_parse(self):
        text = f"<tool_call>{GOOD}</tool_call>"
        parsed = parse_tool_call(text)
        assert parsed["name"] == "book_flight"
        assert parsed["args_state"] == ARGS_OK
        assert json.loads(parsed["args_canonical"]) == {"city": "Paris", "n": 2}

    def test_strict_never_exceeds_name_match(self):
        # same name, different args -> name-level match, strict False
        a = '<tool_call>{"name": "f", "arguments": {"x": 1}}</tool_call>'
        b = '<tool_call>{"name": "f", "arguments": {"x": 2}}</tool_call>'
        assert strict_action_match(a, b) is False
        assert strict_action_match(a, a) is True


class TestTruncatedFirstBlock:
    def test_second_block_recovered(self):
        # the pathological v1 case: truncated first block swallowed the second
        text = (
            '<tool_call>\n{"name": "search", "argu'
            '<tool_call>\n{"name": "book", "arguments": {"city": "Paris"}}</tool_call>'
        )
        parsed = parse_tool_call(text)
        # name from the harness chain: first JSON-ish candidate is the whole
        # malformed span; the fallback regex finds the first "name" key
        assert parsed is not None
        assert parsed["args_state"] == ARGS_OK
        assert json.loads(parsed["args_canonical"]) == {"city": "Paris"}


class TestArgumentStates:
    def test_null_vs_null_is_a_match(self):
        a = '<tool_call>{"name": "f", "arguments": null}</tool_call>'
        assert strict_action_match(a, a) is True

    def test_unparseable_never_matches(self):
        a = '<tool_call>{"name": "f", "arguments": "{not json"}</tool_call>'
        b = '<tool_call>{"name": "f", "arguments": "{not json"}</tool_call>'
        parsed = parse_tool_call(a)
        assert parsed["args_state"] == ARGS_UNPARSEABLE
        assert strict_action_match(a, b) is False

    def test_null_vs_value_not_a_match(self):
        a = '<tool_call>{"name": "f", "arguments": null}</tool_call>'
        b = '<tool_call>{"name": "f", "arguments": {}}</tool_call>'
        assert strict_action_match(a, b) is False

    def test_explicit_null_state(self):
        parsed = parse_tool_call('<tool_call>{"name": "f", "arguments": null}</tool_call>')
        assert parsed["args_state"] == ARGS_NULL


class TestUtf8Reading:
    def test_compute_arm_reads_utf8_and_nests(self, tmp_path):
        rows = [
            {"qid": "q1", "prediction": '<tool_call>{"name": "搜券", "arguments": {"kw": "空调"}}</tool_call>',
             "target": '<tool_call>{"name": "搜券", "arguments": {"kw": "空调"}}</tool_call>',
             "tool_name_match": True, "skipped": False},
            {"qid": "q2", "prediction": '<tool_call>\n{"name": "book_flight", "argu',
             "target": '<tool_call>{"name": "book_flight", "arguments": {"city": "Paris"}}</tool_call>',
             "tool_name_match": True, "skipped": False},
        ]
        p = tmp_path / "arm.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
        result = compute_arm(p)
        assert result["n_scored"] == 2
        assert result["tool_name_match"] == 2
        assert result["strict_action_match"] == 1  # q1 exact; q2 name-only
        assert result["strict_leq_tool_name"] is True
        assert result["pred_args_states"].get("ok") == 1
        assert result["pred_args_states"].get("no_block") == 1
