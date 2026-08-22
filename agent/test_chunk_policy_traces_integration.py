# -*- coding: utf-8 -*-
"""Env-gated integration test: the four B arms on REAL traces + REAL tokenizer.

The unit tests run on a whitespace fake; this one is the only place the
chunking policies meet real agent-llm-traces text and a real BPE tokenizer, so
it is where the two things that can only be measured show up:

1. **content identity** — every arm must present the SAME frozen content
   (``content_tokens`` identical across arms, by construction of the frozen
   stream; this test is the construction's regression guard on real data);
2. **gist declaration early warning** — Sum(ceil(len/ratio)) over each arm's
   chunks is what the NPU run will report as ``avg_gist_tokens``.  The 5%
   deviation rule (判据1) can therefore be checked BEFORE burning NPU hours.
   The number is printed, never asserted: this is a warning instrument, not a
   gate (the gate lives in ``analyze_b_pilot.py`` on the real rows).

Gated on two env vars; both missing -> skip (torch is NOT needed):

  C2KV_TRACES_DIR     agent-llm-traces dir holding data/train-*.parquet
  C2KV_TOKENIZER_DIR  local HF dir with tokenizer.json + tokenizer_config.json
  C2KV_TRACES_MIN_EXAMPLES  optional, default 50

  C2KV_TRACES_DIR=.../agent-llm-traces C2KV_TOKENIZER_DIR=.../snapshots/<sha> \
      python -m pytest agent/test_chunk_policy_traces_integration.py -v -s
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _sub in ("python", "agent"):
    _path = str(_REPO_ROOT / _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

_TRACES_DIR = os.environ.get("C2KV_TRACES_DIR")
_TOKENIZER_DIR = os.environ.get("C2KV_TOKENIZER_DIR")
_MIN_EXAMPLES = int(os.environ.get("C2KV_TRACES_MIN_EXAMPLES", "50"))

pytestmark = pytest.mark.skipif(
    not (_TRACES_DIR and _TOKENIZER_DIR),
    reason="set C2KV_TRACES_DIR and C2KV_TOKENIZER_DIR to run the real-data chunking check",
)

ARMS = {
    "P-fixed": ("fixed-1024", 0),
    "P-turn": ("agent-turn", 0),
    "P-struct": ("structural", 0),
    "P-delay": ("agent-turn", 1),
}
RATIO = 8
MAX_DOC_LENGTH = 1024
MAX_DOC_NUM = 24
MAX_TOOL_CHUNKS = 16


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(_TOKENIZER_DIR, local_files_only=True)


@pytest.fixture(scope="module")
def examples():
    from train.train_data_joint import AgentLLMTracesJointSource

    source = AgentLLMTracesJointSource(
        _TRACES_DIR,
        split="train",
        max_samples_per_session=2,
        max_records=max(_MIN_EXAMPLES * 2, 100),
        require_tool_call=True,
        max_input_chars=200_000,
    )
    records = [example for example in source if example.history_documents]
    if len(records) < _MIN_EXAMPLES:
        pytest.skip(f"only {len(records)} usable examples under {_TRACES_DIR}")
    return records[:_MIN_EXAMPLES]


def _arm_stats(tokenizer, examples, chunk_policy, delay_recent_turns):
    from train.train_data_joint import build_history_chunks
    from train.train_data_multiturn import _chat_template_ids

    stats = {
        "content_tokens": [],
        "chunk_count": [],
        "wrapped_tokens": [],
        "raw_recent_tokens": [],
        "gist_tokens": [],
        "structural_fallback_docs": 0,
        "structural_partial_docs": 0,
    }
    for example in examples:
        kept, delayed, meta = build_history_chunks(
            tokenizer,
            example,
            "joint",
            max_doc_length=MAX_DOC_LENGTH,
            max_doc_num=MAX_DOC_NUM,
            max_tool_chunks=MAX_TOOL_CHUNKS,
            num_tool_chunks=MAX_TOOL_CHUNKS,
            per_side_caps=True,
            history_selection="tail",
            split_oversized_history_docs=True,
            chunk_policy=chunk_policy,
            delay_recent_turns=delay_recent_turns,
            # This test is the eval-side accounting rehearsal, so it opts into
            # the content-token measurement the trainer path deliberately skips.
            need_content_tokens=True,
        )
        chunk_ids = [
            _chat_template_ids(tokenizer, [message], max_length=MAX_DOC_LENGTH)
            for message in kept
        ]
        raw_ids = [
            token
            for message in delayed
            for token in _chat_template_ids(tokenizer, [message], max_length=MAX_DOC_LENGTH)
        ]
        stats["content_tokens"].append(meta["content_tokens"])
        stats["chunk_count"].append(len(chunk_ids))
        stats["wrapped_tokens"].append(sum(len(ids) for ids in chunk_ids) + len(raw_ids))
        stats["raw_recent_tokens"].append(len(raw_ids))
        # What the NPU run will report as gist_tokens for this arm: the
        # dynamic-interleave extractor emits ceil(len/ratio) gists per chunk.
        stats["gist_tokens"].append(sum(math.ceil(len(ids) / RATIO) for ids in chunk_ids))
        stats["structural_fallback_docs"] += meta.get("structural_fallback_docs", 0)
        stats["structural_partial_docs"] += meta.get("structural_partial_docs", 0)
    return stats


def test_four_arms_share_the_frozen_content(tokenizer, examples):
    per_arm = {
        name: _arm_stats(tokenizer, examples, policy, delay)
        for name, (policy, delay) in ARMS.items()
    }
    reference = per_arm["P-fixed"]["content_tokens"]
    for name, stats in per_arm.items():
        assert stats["content_tokens"] == reference, (
            f"arm {name} does not present the frozen content stream — the "
            "content-freeze construction regressed"
        )

    def _mean(values):
        return sum(values) / len(values) if values else 0.0

    reference_gist = _mean(per_arm["P-fixed"]["gist_tokens"])
    print(f"\n=== B pilot gist declaration early warning (n={len(examples)}, ratio={RATIO}) ===")
    print(
        f"{'arm':<10} {'chunks':>7} {'content_tok':>12} {'wrapped_tok':>12} "
        f"{'raw_recent':>11} {'sum_ceil_len/8':>15} {'dev_vs_P-fixed':>15}"
    )
    for name, stats in per_arm.items():
        gist = _mean(stats["gist_tokens"])
        deviation = (gist - reference_gist) / reference_gist if reference_gist else 0.0
        # Mirror analyze_b_pilot._gist_declaration exactly: an arm that holds a
        # recent turn back as raw tokens spends that budget in its own
        # raw_recent column, so its lower grid-gist count is by construction
        # and NOT a failed declaration.  Without this the one arm that is
        # designed to be exempt is the only one the table flags.
        exempt = _mean(stats["raw_recent_tokens"]) > 0
        if exempt:
            note = "  (EXEMPT: delayed arm, raw recent turn is a separate cost column)"
        elif abs(deviation) > 0.05:
            note = "  <-- would VOID (判据1 >5%)"
        else:
            note = ""
        print(
            f"{name:<10} {_mean(stats['chunk_count']):>7.2f} "
            f"{_mean(stats['content_tokens']):>12.1f} {_mean(stats['wrapped_tokens']):>12.1f} "
            f"{_mean(stats['raw_recent_tokens']):>11.1f} {gist:>15.1f} "
            f"{deviation * 100:>14.2f}%"
            + note
        )
    print(
        "structural: fallback_docs="
        f"{per_arm['P-struct']['structural_fallback_docs']} "
        f"partial_docs={per_arm['P-struct']['structural_partial_docs']}"
    )
    print("NOTE: printed only. The 5% rule is enforced in analyze_b_pilot.py on real rows.")


def test_delay_arm_conserves_presented_tokens(tokenizer, examples):
    turn = _arm_stats(tokenizer, examples, "agent-turn", 0)
    delay = _arm_stats(tokenizer, examples, "agent-turn", 1)
    # The delayed turn leaves the grid and returns raw: nothing is dropped.
    assert delay["wrapped_tokens"] == turn["wrapped_tokens"]
    assert sum(delay["raw_recent_tokens"]) > 0
    assert sum(delay["chunk_count"]) < sum(turn["chunk_count"])
