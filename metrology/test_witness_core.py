# -*- coding: utf-8 -*-
"""CPU unit test for agent/d_witness_core.py (prereg v2.2 frozen algorithm).

Covers the semantics the prereg froze:
- None result: no value occurs anywhere -> k_star is None (not an exception)
- short values (<8 chars) need word boundaries: "en" must NOT match inside
  "length"; "0" must NOT match inside "2001"
- 1/df cancellation: values present in every doc add the same 1/n to each
  block and cannot change the argmax
- unique witness: a value present in exactly one doc adds 1.0 to that doc
- ties resolve to the lowest index
- leaves(): JSON literal forms (true/false/null), dedup, no empty strings
- k_median (computed by the selector) vs k_witness independence is the
  selector's business; here only the core math is pinned

Run:  pytest metrology/test_witness_core.py -v   (torch-free)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "agent"))

from d_witness_core import (  # noqa: E402
    leaves,
    occurs,
    select_k_star,
    target_values,
    witness_scores,
)


class TestOccurs:
    def test_long_value_substring(self):
        assert occurs("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9", "token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9 end")

    def test_short_value_word_boundary(self):
        assert occurs("en", "lang en used")
        assert not occurs("en", "the length of tenure")  # inside words
        assert not occurs("en", "locale: token.en")  # preceded by a dot
        assert not occurs("en", "entries")  # followed by a word char

    def test_short_number_not_inside_longer_number(self):
        assert occurs("0", "offset 0 end")
        assert not occurs("0", "year 2001")  # would match "0" inside "2001" without the guard

    def test_regex_metacharacters_escaped(self):
        assert occurs("a.b(c)", "call a.b(c) now")  # literal match, not regex semantics
        assert not occurs("a.b(c)", "axbYcZ now")


class TestLeavesAndValues:
    def test_json_literal_forms(self):
        assert sorted(leaves({"a": True, "b": False, "c": None, "d": 3, "e": 1.5, "f": "s"})) == [
            "1.5", "3", "false", "null", "s", "true",
        ]

    def test_nested(self):
        assert leaves({"x": [{"y": "deep"}, 7]}) == ["deep", "7"]

    def test_target_values_dedup_and_drop_empty(self):
        vals = target_values("tool", {"q": "same", "r": "same", "s": "", "n": 1})
        assert vals == ["tool", "same", "1"]

    def test_keys_not_included(self):
        assert "query" not in leaves({"query": "Leprous band"})


class TestSelection:
    def test_none_when_no_witness(self):
        texts = ["alpha beta", "gamma delta"]
        values = ["zzz-not-present-anywhere"]
        assert select_k_star(texts, values) is None

    def test_unique_witness_wins(self):
        texts = [
            "user asked about trains; token abcdefgh123 was issued",
            "the meeting notes mention Sydney opera; token abcdefgh123 again",
            "shipping label includes the same long token abcdefgh123",
        ]
        values = ["Sydney", "abcdefgh123"]
        df, score = witness_scores(texts, values)
        assert df == {"Sydney": 1, "abcdefgh123": 3}
        # doc0 gets 1/3, doc1 gets 1/3 + 1.0, doc2 gets 1/3
        assert select_k_star(texts, values) == 1

    def test_everywhere_value_cancels(self):
        # "en" occurs in every doc (same 1/n each), "Sydney" only in doc 2
        texts = ["lang en a", "lang en b", "lang en Sydney"]
        values = ["en", "Sydney"]
        df, _ = witness_scores(texts, values)
        assert df["en"] == 3
        assert select_k_star(texts, values) == 2

    def test_tie_resolves_to_first_index(self):
        texts = ["has alpha", "has beta", "neither"]
        assert select_k_star(texts, ["alpha"]) == 0
        assert select_k_star(texts, ["beta"]) == 1
        # two equally-unique witnesses in different docs: first-scanned max wins
        assert select_k_star(texts, ["alpha", "beta"]) == 0

    def test_score_is_sum_of_inverse_df(self):
        texts = ["v1 v2", "v1", "v2"]
        df, score = witness_scores(texts, ["v1", "v2"])
        assert df == {"v1": 2, "v2": 2}
        assert score == [1.0, 0.5, 0.5]
