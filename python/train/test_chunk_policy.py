# -*- coding: utf-8 -*-
"""CPU-only, torch-free unit tests for the B-line chunking policies.

Covers ``python/train/chunk_policy.py`` plus the two pieces of plumbing it
depends on: ``train_data_multiturn._agent_history_turn_units`` /
``_flatten_turn_units`` and ``train_data_joint.build_history_chunks``.

Coverage:
a. ``parse_chunk_policy`` mapping and rejection of unknown values;
b. provenance equivalence: ``fit_history_with_provenance`` reproduces
   ``_fit_reused_history`` text-for-text (with the real primitives, and with
   pure whitespace fakes to prove the injection contract);
c. unit/renderer invariant: ``_flatten_turn_units(_agent_history_turn_units
   (m)[i]) == _agent_history_turn_docs(m)[i]["content"]``;
d. ``fixed``: content identity of the re-cut stream, window sizing;
e. ``structural``: action+observation atomic pairs never split, ``num_parts>1``
   shards degrade to pass-through and are counted, oversized-block fallback;
f. ``split_delayed``: turn granularity, k=0 identity, fixed+delay ValueError;
g. ``build_history_chunks`` fast path is bit-identical to today's
   ``_fit_reused_history`` call.

Run from the repo root (no torch needed):
  python -m pytest python/train/test_chunk_policy.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make python/ importable when pytest is invoked from the repo root.
_PYTHON_DIR = Path(__file__).resolve().parents[1]
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

from train.chunk_policy import (  # noqa: E402
    CHUNK_POLICIES,
    FROZEN_JOIN,
    FrozenDoc,
    _pair_units,
    apply_policy,
    fit_history_with_provenance,
    parse_chunk_policy,
    split_delayed,
)
from train.train_data_joint import (  # noqa: E402
    JointExample,
    _WhitespaceSelfTestTokenizer,
    _decode_fn,
    _encode_fn,
    build_history_chunks,
)
from train.train_data_multiturn import (  # noqa: E402
    _agent_history_turn_docs,
    _agent_history_turn_units,
    _fit_reused_history,
    _flatten_turn_units,
    _message_token_length,
    _select_history,
    _split_message_to_fit,
)


@pytest.fixture()
def tokenizer():
    return _WhitespaceSelfTestTokenizer()


# ---------------------------------------------------------------------------
# Pure whitespace fakes: no tokenizer object at all, so the injection contract
# is exercised without any repo primitive.
# ---------------------------------------------------------------------------


def _fake_token_len(_tokenizer, message):
    # +2 stands in for the chat-template wrapper tokens.
    return len(str(message.get("content") or "").split()) + 2


def _fake_split(_tokenizer, message, max_doc_length):
    words = str(message.get("content") or "").split()
    budget = max(1, max_doc_length - 2)
    if len(words) + 2 <= max_doc_length:
        return [dict(message)]
    return [
        {"role": message["role"], "content": " ".join(words[start : start + budget])}
        for start in range(0, len(words), budget)
    ]


def _fake_encode(_tokenizer, text):
    return list(str(text).split())


def _fake_decode(_tokenizer, ids):
    return " ".join(ids)


# ---------------------------------------------------------------------------
# a. policy parsing
# ---------------------------------------------------------------------------


def test_parse_chunk_policy():
    assert parse_chunk_policy("fixed-256") == ("fixed", 256)
    assert parse_chunk_policy("fixed-512") == ("fixed", 512)
    assert parse_chunk_policy("fixed-1024") == ("fixed", 1024)
    assert parse_chunk_policy("agent-turn") == ("agent-turn", None)
    assert parse_chunk_policy("structural") == ("structural", None)
    assert set(CHUNK_POLICIES) == {
        "fixed-256",
        "fixed-512",
        "fixed-1024",
        "agent-turn",
        "structural",
    }
    with pytest.raises(ValueError, match="Unsupported chunk_policy"):
        parse_chunk_policy("natural-paragraph")


# ---------------------------------------------------------------------------
# b. provenance equivalence with _fit_reused_history
# ---------------------------------------------------------------------------


def _raw_history(num_docs=5, words=6, long_index=None, long_words=200):
    docs = []
    for index in range(num_docs):
        count = long_words if index == long_index else words
        docs.append(
            {
                "role": "user",
                "content": " ".join(f"turn{index}word{position}" for position in range(count)),
            }
        )
    return docs


@pytest.mark.parametrize("split_oversized", [True, False])
@pytest.mark.parametrize("policy", ["tail", "head"])
@pytest.mark.parametrize("max_doc_num", [2, 3, 10])
def test_provenance_texts_match_fit_reused_history(
    tokenizer, split_oversized, policy, max_doc_num
):
    raw_history = _raw_history(num_docs=5, words=6, long_index=2, long_words=90)
    expected = _fit_reused_history(
        tokenizer,
        raw_history,
        max_doc_length=32,
        max_doc_num=max_doc_num,
        policy=policy,
        split_oversized_history_docs=split_oversized,
    )
    frozen = fit_history_with_provenance(
        tokenizer,
        raw_history,
        max_doc_length=32,
        max_doc_num=max_doc_num,
        policy=policy,
        split_oversized_history_docs=split_oversized,
        split_fn=_split_message_to_fit,
        select_fn=_select_history,
        token_len_fn=_message_token_length,
    )
    assert [doc.text for doc in frozen] == [message["content"] for message in expected]
    # Provenance is coherent: turn indices are in range and shard bookkeeping
    # agrees with the split that produced them.
    for doc in frozen:
        assert 0 <= doc.turn_index < len(raw_history)
        assert 0 <= doc.part_index < doc.num_parts


def test_provenance_with_pure_fakes_tracks_shards():
    """No tokenizer object, no repo primitive: everything injected."""
    raw_history = [
        {"role": "user", "content": "a b c"},
        {"role": "user", "content": " ".join(f"w{i}" for i in range(20))},
    ]
    frozen = fit_history_with_provenance(
        None,
        raw_history,
        max_doc_length=8,
        max_doc_num=99,
        policy="tail",
        split_oversized_history_docs=True,
        split_fn=_fake_split,
        select_fn=_select_history,
        token_len_fn=_fake_token_len,
    )
    assert [doc.turn_index for doc in frozen] == [0, 1, 1, 1, 1]
    assert [doc.num_parts for doc in frozen] == [1, 4, 4, 4, 4]
    assert [doc.part_index for doc in frozen] == [0, 0, 1, 2, 3]

    # split_oversized=False drops the oversized doc entirely (the
    # _fit_reused_history else-branch), keeping provenance for the survivor.
    frozen = fit_history_with_provenance(
        None,
        raw_history,
        max_doc_length=8,
        max_doc_num=99,
        policy="tail",
        split_oversized_history_docs=False,
        split_fn=_fake_split,
        select_fn=_select_history,
        token_len_fn=_fake_token_len,
    )
    assert [doc.turn_index for doc in frozen] == [0]


def test_provenance_requires_slicing_select_fn():
    def _rebuilding_select(messages, max_doc_num, policy):
        return [{"role": m["role"], "content": m["content"]} for m in messages]

    with pytest.raises(ValueError, match="provenance"):
        fit_history_with_provenance(
            None,
            [{"role": "user", "content": "a b"}],
            max_doc_length=8,
            max_doc_num=4,
            policy="tail",
            split_oversized_history_docs=True,
            split_fn=_fake_split,
            select_fn=_rebuilding_select,
            token_len_fn=_fake_token_len,
        )


# ---------------------------------------------------------------------------
# c. unit/renderer invariant
# ---------------------------------------------------------------------------


_AGENT_MESSAGES = [
    {"role": "user", "content": "List the files in /tmp please."},
    {
        "role": "assistant",
        "content": 'Thought:\nlook it up\n\nAction:\n<tool_call>\n{"name":"ls","arguments":{"path":"/tmp"}}\n</tool_call>',
    },
    {"role": "tool", "content": "a.txt\nb.txt"},
    {"role": "assistant", "content": "Found two files."},
    {"role": "user", "content": "Now read a.txt"},
    {
        "role": "assistant",
        "content": 'Action:\n<tool_call>\n{"name":"read","arguments":{"path":"/tmp/a.txt"}}\n</tool_call>',
    },
    {"role": "observation", "content": "hello world"},
    {"role": "user", "content": ""},  # skipped: empty non-assistant
]


def test_flatten_turn_units_reproduces_turn_docs():
    docs = _agent_history_turn_docs(_AGENT_MESSAGES)
    units = _agent_history_turn_units(_AGENT_MESSAGES)
    assert len(docs) == len(units)
    for index, doc in enumerate(docs):
        assert _flatten_turn_units(units[index]) == doc["content"]


def test_turn_units_kinds():
    units = _agent_history_turn_units(_AGENT_MESSAGES)
    assert [unit["kind"] for unit in units[0]] == ["user", "assistant", "tool", "assistant"]
    assert [unit["kind"] for unit in units[1]] == ["user", "assistant", "observation"]


def test_pair_units_binds_action_to_observation():
    units = _agent_history_turn_units(_AGENT_MESSAGES)
    blocks = _pair_units(units[0])
    # user | (action assistant + tool) | plain assistant
    assert [[unit["kind"] for unit in block] for block in blocks] == [
        ["user"],
        ["assistant", "tool"],
        ["assistant"],
    ]
    blocks = _pair_units(units[1])
    assert [[unit["kind"] for unit in block] for block in blocks] == [
        ["user"],
        ["assistant", "observation"],
    ]


def test_pair_units_plain_assistant_does_not_swallow_observation():
    units = [
        {"kind": "assistant", "role": "assistant", "text": "just talking"},
        {"kind": "tool", "role": "tool", "text": "stray observation"},
    ]
    assert [[unit["kind"] for unit in block] for block in _pair_units(units)] == [
        ["assistant"],
        ["tool"],
    ]


# ---------------------------------------------------------------------------
# d. fixed policy
# ---------------------------------------------------------------------------


def _frozen(texts):
    return [
        FrozenDoc(text=text, turn_index=index, part_index=0, num_parts=1)
        for index, text in enumerate(texts)
    ]


def test_fixed_policy_content_identity(tokenizer):
    texts = [" ".join(f"d{d}w{w}" for w in range(30)) for d in range(4)]
    frozen = _frozen(texts)
    messages, meta = apply_policy(
        tokenizer, frozen, [], "fixed", 16, 1024, _encode_fn, _decode_fn,
        token_len_fn=_message_token_length,
    )
    # Every window but the last is exactly the window size; concatenating the
    # windows reproduces the frozen token stream.
    window = meta["fixed_window_tokens"]
    assert window == 16
    stream = _encode_fn(tokenizer, FROZEN_JOIN.join(texts))
    rebuilt = []
    for message in messages:
        rebuilt.extend(_encode_fn(tokenizer, message["content"]))
    assert rebuilt == stream
    assert len(messages) == (len(stream) + window - 1) // window
    assert meta["doc_turn_indices"] == [None] * len(messages)


def test_fixed_window_is_clamped_to_the_wrapped_doc_budget(tokenizer):
    frozen = _frozen([" ".join(f"w{i}" for i in range(50))])
    _, meta = apply_policy(
        tokenizer, frozen, [], "fixed", 1024, 32, _encode_fn, _decode_fn,
        token_len_fn=_message_token_length,
    )
    overhead = _message_token_length(tokenizer, {"role": "user", "content": ""})
    assert meta["fixed_window_tokens"] == 32 - overhead - 8


def test_content_tokens_identical_across_policies(tokenizer):
    messages = _AGENT_MESSAGES
    docs = _agent_history_turn_docs(messages)
    units = _agent_history_turn_units(messages)
    frozen = _frozen([doc["content"] for doc in docs])
    seen = set()
    for policy in CHUNK_POLICIES:
        kind, chunk_size = parse_chunk_policy(policy)
        _, meta = apply_policy(
            tokenizer, frozen, units, kind, chunk_size, 1024, _encode_fn, _decode_fn,
            token_len_fn=_message_token_length,
            flatten_fn=_flatten_turn_units,
            split_fn=_split_message_to_fit,
            need_content_tokens=True,
        )
        seen.add(meta["content_tokens"])
    assert len(seen) == 1 and seen != {0}


def test_content_tokens_are_none_unless_requested(tokenizer):
    """Not-measured must be distinguishable from a measured zero."""
    frozen = _frozen([" ".join(f"w{i}" for i in range(20))])
    for policy in CHUNK_POLICIES:
        kind, chunk_size = parse_chunk_policy(policy)
        _, meta = apply_policy(
            tokenizer, frozen, [], kind, chunk_size, 1024, _encode_fn, _decode_fn,
            token_len_fn=_message_token_length,
            flatten_fn=_flatten_turn_units,
            split_fn=_split_message_to_fit,
        )
        assert meta["content_tokens"] is None, policy
        assert meta["policy_content_tokens"] is None, policy


# ---------------------------------------------------------------------------
# e. structural policy
# ---------------------------------------------------------------------------


def test_structural_splits_turns_at_atomic_blocks(tokenizer):
    docs = _agent_history_turn_docs(_AGENT_MESSAGES)
    units = _agent_history_turn_units(_AGENT_MESSAGES)
    frozen = _frozen([doc["content"] for doc in docs])
    messages, meta = apply_policy(
        tokenizer, frozen, units, "structural", None, 1024, _encode_fn, _decode_fn,
        token_len_fn=_message_token_length,
        flatten_fn=_flatten_turn_units,
        split_fn=_split_message_to_fit,
    )
    # turn 0 -> 3 blocks, turn 1 -> 2 blocks.
    assert len(messages) == 5
    assert meta["structural_repacked_docs"] == 2
    assert meta["structural_partial_docs"] == 0
    assert meta["structural_fallback_docs"] == 0
    assert meta["doc_turn_indices"] == [0, 0, 0, 1, 1]
    # The action and its observation stay in ONE doc; no doc holds the action
    # without the observation.
    action_docs = [m["content"] for m in messages if "<tool_call>" in m["content"]]
    assert len(action_docs) == 2
    assert "a.txt" in action_docs[0] and "hello world" in action_docs[1]


def test_structural_passes_through_sharded_docs(tokenizer):
    frozen = [
        FrozenDoc(text="shard one text", turn_index=0, part_index=0, num_parts=2),
        FrozenDoc(text="shard two text", turn_index=0, part_index=1, num_parts=2),
    ]
    units = _agent_history_turn_units(_AGENT_MESSAGES)
    messages, meta = apply_policy(
        tokenizer, frozen, units, "structural", None, 1024, _encode_fn, _decode_fn,
        token_len_fn=_message_token_length,
        flatten_fn=_flatten_turn_units,
        split_fn=_split_message_to_fit,
    )
    assert [m["content"] for m in messages] == ["shard one text", "shard two text"]
    assert meta["structural_partial_docs"] == 2
    assert meta["structural_repacked_docs"] == 0


def test_structural_single_block_turn_is_passthrough(tokenizer):
    messages_in = [{"role": "user", "content": "only a user query"}]
    docs = _agent_history_turn_docs(messages_in)
    units = _agent_history_turn_units(messages_in)
    frozen = _frozen([doc["content"] for doc in docs])
    messages, meta = apply_policy(
        tokenizer, frozen, units, "structural", None, 1024, _encode_fn, _decode_fn,
        token_len_fn=_message_token_length,
        flatten_fn=_flatten_turn_units,
        split_fn=_split_message_to_fit,
    )
    assert [m["content"] for m in messages] == [docs[0]["content"]]
    assert meta["structural_passthrough_docs"] == 1


def test_structural_oversized_block_falls_back_to_split_fn(tokenizer):
    """An atomic block wider than max_doc_length is split and counted."""
    long_observation = " ".join(f"obs{i}" for i in range(200))
    messages_in = [
        {"role": "user", "content": "short question"},
        {"role": "assistant", "content": 'Action:\n<tool_call>\n{"name":"ls"}\n</tool_call>'},
        {"role": "tool", "content": long_observation},
    ]
    docs = _agent_history_turn_docs(messages_in)
    units = _agent_history_turn_units(messages_in)
    # num_parts=1 is asserted by construction here: the point is that the
    # per-block budget can still be exceeded when a single observation is huge.
    frozen = _frozen([doc["content"] for doc in docs])
    messages, meta = apply_policy(
        tokenizer, frozen, units, "structural", None, 64, _encode_fn, _decode_fn,
        token_len_fn=_message_token_length,
        flatten_fn=_flatten_turn_units,
        split_fn=_split_message_to_fit,
    )
    assert meta["structural_fallback_docs"] == 1
    assert len(messages) > 2
    for message in messages:
        assert _message_token_length(tokenizer, message) <= 64


def test_structural_without_units_is_identity(tokenizer):
    docs = _agent_history_turn_docs(_AGENT_MESSAGES)
    frozen = _frozen([doc["content"] for doc in docs])
    messages, meta = apply_policy(
        tokenizer, frozen, [], "structural", None, 1024, _encode_fn, _decode_fn,
        token_len_fn=_message_token_length,
        flatten_fn=_flatten_turn_units,
        split_fn=_split_message_to_fit,
    )
    assert [m["content"] for m in messages] == [doc["content"] for doc in docs]
    assert meta["structural_passthrough_docs"] == len(docs)


# ---------------------------------------------------------------------------
# f. split_delayed
# ---------------------------------------------------------------------------


def test_split_delayed_is_turn_granular():
    frozen = [
        FrozenDoc(text="t0a", turn_index=0, part_index=0, num_parts=2),
        FrozenDoc(text="t0b", turn_index=0, part_index=1, num_parts=2),
        FrozenDoc(text="t1", turn_index=1, part_index=0, num_parts=1),
        FrozenDoc(text="t2a", turn_index=2, part_index=0, num_parts=2),
        FrozenDoc(text="t2b", turn_index=2, part_index=1, num_parts=2),
    ]
    docs = [{"role": "user", "content": doc.text} for doc in frozen]
    kept, delayed = split_delayed(docs, frozen, 1)
    assert [m["content"] for m in kept] == ["t0a", "t0b", "t1"]
    assert [m["content"] for m in delayed] == ["t2a", "t2b"]
    kept, delayed = split_delayed(docs, frozen, 2)
    assert [m["content"] for m in kept] == ["t0a", "t0b"]
    assert [m["content"] for m in delayed] == ["t1", "t2a", "t2b"]


def test_split_delayed_k_zero_is_identity():
    frozen = _frozen(["a", "b"])
    docs = [{"role": "user", "content": doc.text} for doc in frozen]
    kept, delayed = split_delayed(docs, frozen, 0)
    assert kept == docs and delayed == []


def test_split_delayed_rejects_missing_turn_provenance():
    frozen = _frozen(["a", "b"])
    docs = [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}]
    with pytest.raises(ValueError, match="fixed"):
        split_delayed(docs, frozen, 1, doc_turn_indices=[None, None])
    with pytest.raises(ValueError, match="non-negative"):
        split_delayed(docs, frozen, -1)


def test_build_history_chunks_rejects_fixed_plus_delay(tokenizer):
    example = _joint_example()
    with pytest.raises(ValueError, match="delay_recent_turns"):
        build_history_chunks(
            tokenizer,
            example,
            "joint",
            max_doc_length=64,
            max_doc_num=8,
            max_tool_chunks=4,
            num_tool_chunks=1,
            per_side_caps=True,
            history_selection="tail",
            split_oversized_history_docs=True,
            chunk_policy="fixed-256",
            delay_recent_turns=1,
        )


# ---------------------------------------------------------------------------
# g. build_history_chunks: fast path + delayed wiring
# ---------------------------------------------------------------------------


def _joint_example():
    docs = _agent_history_turn_docs(_AGENT_MESSAGES)
    units = _agent_history_turn_units(_AGENT_MESSAGES)
    return JointExample(
        qid="s0:0",
        session_id="s0",
        tool_documents=["<TOOL> ls </TOOL>", "<TOOL> read </TOOL>"],
        history_documents=[doc["content"] for doc in docs],
        current_messages=[{"role": "user", "content": "and b.txt?"}],
        answer='Action:\n<tool_call>\n{"name":"read","arguments":{"path":"/tmp/b.txt"}}\n</tool_call>',
        system_prompt="You are a test agent.",
        subset="test",
        history_units=units,
    )


def _kwargs(**overrides):
    kwargs = dict(
        max_doc_length=1024,
        max_doc_num=8,
        max_tool_chunks=4,
        num_tool_chunks=2,
        per_side_caps=True,
        history_selection="tail",
        split_oversized_history_docs=True,
    )
    kwargs.update(overrides)
    return kwargs


def test_fast_path_is_bit_identical_to_fit_reused_history(tokenizer):
    example = _joint_example()
    kept, delayed, meta = build_history_chunks(
        tokenizer, example, "joint", **_kwargs(need_content_tokens=True)
    )
    expected = _fit_reused_history(
        tokenizer,
        [{"role": "user", "content": text} for text in example.history_documents],
        max_doc_length=1024,
        max_doc_num=4,  # _history_chunk_budget(joint, 8, 4, 2, per_side_caps=True)
        policy="tail",
        split_oversized_history_docs=True,
    )
    assert kept == expected
    assert delayed == []
    assert meta["chunk_policy"] == "agent-turn"
    assert meta["history_chunk_count"] == len(expected)
    assert meta["content_tokens"] > 0


def test_agent_turn_with_delay_moves_the_last_turn_out(tokenizer):
    example = _joint_example()
    kept, delayed, meta = build_history_chunks(
        tokenizer, example, "joint", **_kwargs(delay_recent_turns=1)
    )
    baseline, _, _ = build_history_chunks(tokenizer, example, "joint", **_kwargs())
    assert kept + delayed == baseline
    assert len(delayed) == 1
    assert meta["delayed_docs"] == 1


def test_structural_arm_produces_more_chunks_at_the_same_content(tokenizer):
    example = _joint_example()
    turn_kept, _, turn_meta = build_history_chunks(
        tokenizer, example, "joint", **_kwargs(need_content_tokens=True)
    )
    struct_kept, _, struct_meta = build_history_chunks(
        tokenizer, example, "joint",
        **_kwargs(chunk_policy="structural", need_content_tokens=True),
    )
    assert struct_meta["content_tokens"] == turn_meta["content_tokens"]
    assert len(struct_kept) > len(turn_kept)


def test_tool_only_mode_returns_no_history(tokenizer):
    example = _joint_example()
    kept, delayed, meta = build_history_chunks(
        tokenizer, example, "tool_only", **_kwargs(chunk_policy="structural")
    )
    assert kept == [] and delayed == []
    assert meta["history_chunk_count"] == 0


class _CountingTokenizer(_WhitespaceSelfTestTokenizer):
    """Records every text handed to ``encode`` (the content-token accounting)."""

    def __init__(self):
        super().__init__()
        self.encoded: list = []

    def encode(self, text, add_special_tokens=False):
        self.encoded.append(text)
        return super().encode(text, add_special_tokens=add_special_tokens)


@pytest.mark.parametrize("policy", ["agent-turn", "structural", "fixed-1024"])
def test_trainer_path_pays_no_extra_encode_for_content_tokens(policy):
    """The trainer never measures content_tokens, so it must never encode for it.

    Guards the regression this flag exists to prevent: an unconditional
    ``_encode_fn(tokenizer, join(history))`` in the fast path doubled the
    history-side tokenization cost of every dataset build.
    """

    example = _joint_example()
    off = _CountingTokenizer()
    kept_off, _, meta_off = build_history_chunks(
        off, example, "joint", **_kwargs(chunk_policy=policy)
    )
    on = _CountingTokenizer()
    kept_on, _, meta_on = build_history_chunks(
        on, example, "joint", **_kwargs(chunk_policy=policy, need_content_tokens=True)
    )
    # Identical chunking either way: the flag only controls accounting.
    assert kept_off == kept_on
    assert meta_off["content_tokens"] is None
    assert meta_on["content_tokens"] > 0
    assert len(off.encoded) < len(on.encoded)
    if policy == "agent-turn":
        # Fast path: the accounting encode is the join of the kept doc texts,
        # and it is the ONLY thing the flag adds there.
        stream = FROZEN_JOIN.join(message["content"] for message in kept_on)
        assert stream in on.encoded
        assert stream not in off.encoded
        assert len(on.encoded) == len(off.encoded) + 1


def test_history_units_missing_degrades_to_agent_turn(tokenizer):
    example = _joint_example()
    stripped = JointExample(
        qid=example.qid,
        session_id=example.session_id,
        tool_documents=example.tool_documents,
        history_documents=example.history_documents,
        current_messages=example.current_messages,
        answer=example.answer,
        system_prompt=example.system_prompt,
        subset=example.subset,
        history_units=None,
    )
    struct_kept, _, _ = build_history_chunks(
        tokenizer, stripped, "joint", **_kwargs(chunk_policy="structural")
    )
    turn_kept, _, _ = build_history_chunks(tokenizer, stripped, "joint", **_kwargs())
    assert struct_kept == turn_kept
