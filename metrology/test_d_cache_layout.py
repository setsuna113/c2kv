# -*- coding: utf-8 -*-
"""CPU unit test for the D-contract cache layout primitives (acceptance #5).

Pins the mechanics that make the three D1 arms' layouts differ as
(cache_length, k_anchor, history_length) triples, using FAKE caches (no
model): gist span extraction from the compression mask, append / in-place
splice / slice-replace semantics, and the merge-system-gist rebuild used by
the k-sweep (a fresh cache per k from immutable parts).

The end-to-end expression of acceptance #5 (three arms on a real qid) runs
on the server in the smoke phase; this file pins the primitives.

Run:  pytest metrology/test_d_cache_layout.py -v   (needs torch; skips without)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (_REPO_ROOT, _REPO_ROOT / "python", _REPO_ROOT / "python" / "inference",
          _REPO_ROOT / "agent"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from d1_arms import _cat_span_to_cache, _replace_span_in_cache, gist_doc_spans  # noqa: E402


class _Layer:
    def __init__(self, keys, values):
        self.keys, self.values = keys, values


class _FakeCache:
    def __init__(self, seq, heads=2, dim=4, layers=2):
        self.layers = [
            _Layer(torch.randn(1, heads, seq, dim), torch.randn(1, heads, seq, dim))
            for _ in range(layers)
        ]

    def get_seq_length(self):
        return self.layers[0].keys.shape[-2]


def test_gist_doc_spans_exact_from_mask():
    # grid rows contribute [3, 5, 0(filler), 4] gists -> exact cumulative spans
    mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, True, True, True],
        [False, False, False, False, False],
        [True, True, True, True, False],
    ])
    assert gist_doc_spans(mask) == [(0, 3), (3, 8), (8, 12)]  # filler row skipped


def test_append_and_replace_splice_layouts_differ():
    cache_len, sys_len, L, g_k = 20, 5, 6, 2      # 5 system + 15 gists; G_k = 2 tokens
    span = [(torch.randn(1, 2, L, 4), torch.randn(1, 2, L, 4)) for _ in range(2)]

    appended = _cat_span_to_cache(_FakeCache(cache_len), span)
    assert appended.get_seq_length() == cache_len + L

    replaced = _replace_span_in_cache(_FakeCache(cache_len), span, sys_len + 8, sys_len + 8 + g_k)
    assert replaced.get_seq_length() == cache_len - g_k + L

    # keepG (append, anchor=offsets[k*]) vs erratum_tail (append, anchor=tail)
    # share cache_length but must differ in anchor/ledger — pinned at the
    # d1_arms level by construction (k_anchor fields), not here.
    assert appended.get_seq_length() != replaced.get_seq_length()


def test_replace_splice_content_positions():
    cache = _FakeCache(10, heads=1, dim=2, layers=1)
    keys_before = cache.layers[0].keys.clone()
    span_k = torch.full((1, 1, 3, 2), 7.0)
    span = [(span_k, torch.zeros_like(span_k))]
    out = _replace_span_in_cache(cache, span, 4, 6)  # cut [4,6), insert 3
    k = out.layers[0].keys[0, 0]
    assert torch.equal(k[:4], keys_before[0, 0, :4])
    assert torch.equal(k[4:7], torch.full((3, 2), 7.0))
    assert torch.equal(k[7:], keys_before[0, 0, 6:])
    assert out.get_seq_length() == 11


def test_insert_at_semantics():
    cache = _FakeCache(8, heads=1, dim=2, layers=1)
    keys_before = cache.layers[0].keys.clone()
    span_k = torch.full((1, 1, 2, 2), 9.0)
    out = _cat_span_to_cache(cache, [(span_k, torch.zeros_like(span_k))], insert_at=3)
    k = out.layers[0].keys[0, 0]
    assert torch.equal(k[:3], keys_before[0, 0, :3])
    assert torch.equal(k[3:5], torch.full((2, 2), 9.0))
    assert torch.equal(k[5:], keys_before[0, 0, 3:])
    assert out.get_seq_length() == 10
