"""Chunking-policy primitives for the B-line pilot (experiment B, 24号 B.4.3).

设计约束（照 24号 B.4.2「内容集冻结」）：**所有臂共享同一份已冻结的
history 文本流**——先跑现役 P-turn 的 ``_fit_reused_history`` 选择/切分，把
结果连同 provenance 冻结下来，其它 policy 只在这份文本流上重新划边界。这样
P-turn 臂字节不变（in-distribution 参照零回归），跨臂内容同一性由构造保证。

This module is deliberately **torch-free**: it imports stdlib only.  Everything
that needs a tokenizer or a repo primitive is injected by the caller
(``split_fn`` / ``select_fn`` / ``token_len_fn`` / ``encode_fn`` / ``decode_fn`` /
``flatten_fn``).  Production callers inject
``train_data_multiturn._split_message_to_fit`` / ``_select_history`` /
``_message_token_length`` / ``_flatten_turn_units``; tests inject whitespace
fakes.  ``train_data_joint.build_history_chunks`` is the single production
entry point.

Policies
--------
``fixed-{256,512,1024}``
    Ignore turn boundaries: join the frozen doc texts with a blank line,
    encode once, cut fixed token windows, decode each window back to text.
``agent-turn``
    Identity on the frozen docs (today's production behaviour).
``structural``
    Re-cut each frozen doc at atomic-block boundaries, where an atomic block
    is ``user`` alone, a plain ``assistant`` message alone, or an
    ``assistant`` message that emits an action (``Action:`` / ``<tool_call>``)
    together with every tool/observation unit that follows it.  A frozen doc
    that is itself a shard of an oversized turn (``num_parts > 1``) has no
    unit-level provenance left and passes through unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

CHUNK_POLICIES = ("fixed-256", "fixed-512", "fixed-1024", "agent-turn", "structural")

# Separator used when the frozen docs are flattened into one token stream
# (``fixed`` policies and the ``content_tokens`` accounting).  Same separator
# on every policy so ``content_tokens`` is comparable across arms.
FROZEN_JOIN = "\n\n"

# Markers that make an assistant unit an "action" unit whose observation must
# not be split away from it.  Mirrors the marker set
# ``train_data_multiturn._render_agent_output_messages`` uses to set
# ``has_tool_call`` (:614-618), minus the loose free-text ones.
ACTION_MARKERS = ("Action:", "<tool_call>")

# Safety margin (tokens) subtracted from the fixed window on top of the
# measured chat-template overhead: decode(encode(x)) is not length-preserving
# for a real BPE tokenizer, so a window sized exactly to the budget can still
# overflow ``max_doc_length`` after re-wrapping.
FIXED_WINDOW_MARGIN = 8

_PROVENANCE_KEY = "_chunk_policy_provenance"

Message = Dict[str, Any]
Unit = Dict[str, str]


def parse_chunk_policy(value: str) -> Tuple[str, Optional[int]]:
    """``"fixed-512"`` -> ``("fixed", 512)``; ``"structural"`` -> ``("structural", None)``."""
    if value not in CHUNK_POLICIES:
        raise ValueError(f"Unsupported chunk_policy {value!r}; choose from {CHUNK_POLICIES}")
    if value.startswith("fixed-"):
        return "fixed", int(value.split("-", 1)[1])
    return value, None


@dataclass(frozen=True)
class FrozenDoc:
    """One document of the frozen content stream, with its turn provenance.

    ``turn_index`` indexes the ``raw_history`` list handed to
    ``fit_history_with_provenance`` (and, in lockstep, the ``history_units``
    list handed to ``apply_policy``).  ``part_index``/``num_parts`` describe
    the oversized-doc split: ``num_parts == 1`` means the whole turn fit in
    ``max_doc_length`` and its unit-level provenance is still usable.
    """

    text: str
    turn_index: int
    part_index: int
    num_parts: int


def fit_history_with_provenance(
    tokenizer: Any,
    raw_history: Sequence[Message],
    *,
    max_doc_length: int,
    max_doc_num: int,
    policy: str,
    split_oversized_history_docs: bool,
    split_fn: Callable[..., List[Message]],
    select_fn: Callable[..., List[Message]],
    token_len_fn: Callable[..., int],
) -> List[FrozenDoc]:
    """``_fit_reused_history`` with turn provenance attached to every output doc.

    Invariant (locked by ``test_chunk_policy.py``)::

        [d.text for d in fit_history_with_provenance(...)]
        == [m["content"] for m in _fit_reused_history(<same args>)]

    The provenance survives ``select_fn`` because the carrier messages handed
    to it keep an extra key; ``_select_history`` only slices its input, so the
    very same dict objects come back.
    """

    carriers: List[Message] = []
    provenance: List[Tuple[int, int, int]] = []
    if split_oversized_history_docs:
        for turn_index, message in enumerate(raw_history):
            parts = list(split_fn(tokenizer, message, max_doc_length))
            for part_index, part in enumerate(parts):
                carriers.append(dict(part))
                provenance.append((turn_index, part_index, len(parts)))
    else:
        for turn_index, message in enumerate(raw_history):
            if token_len_fn(tokenizer, message) <= max_doc_length:
                carriers.append(dict(message))
                provenance.append((turn_index, 0, 1))
    for slot, carrier in enumerate(carriers):
        carrier[_PROVENANCE_KEY] = slot

    selected = select_fn(carriers, max_doc_num=max_doc_num, policy=policy)
    frozen: List[FrozenDoc] = []
    for message in selected:
        slot = message.get(_PROVENANCE_KEY)
        if slot is None:
            raise ValueError(
                "select_fn dropped the provenance key; it must slice its input rather "
                "than rebuild the messages"
            )
        turn_index, part_index, num_parts = provenance[slot]
        frozen.append(
            FrozenDoc(
                text=str(message.get("content") or ""),
                turn_index=turn_index,
                part_index=part_index,
                num_parts=num_parts,
            )
        )
    return frozen


def _has_action_marker(text: str) -> bool:
    return any(marker in text for marker in ACTION_MARKERS)


def _pair_units(units: Sequence[Unit]) -> List[List[Unit]]:
    """Group turn units into atomic blocks that ``structural`` must not split.

    ``user`` alone; a plain ``assistant`` alone; an action-emitting
    ``assistant`` bound to every immediately following tool/observation unit.
    """

    blocks: List[List[Unit]] = []
    index = 0
    total = len(units)
    while index < total:
        unit = units[index]
        kind = unit.get("kind")
        if kind == "assistant" and _has_action_marker(str(unit.get("text") or "")):
            block = [unit]
            probe = index + 1
            while probe < total and units[probe].get("kind") not in ("user", "assistant"):
                block.append(units[probe])
                probe += 1
            blocks.append(block)
            index = probe
            continue
        blocks.append([unit])
        index += 1
    return blocks


def _content_tokens(
    tokenizer: Any,
    texts: Sequence[str],
    encode_fn: Callable[[Any, str], Sequence[int]],
) -> int:
    if not texts:
        return 0
    return len(encode_fn(tokenizer, FROZEN_JOIN.join(texts)))


def apply_policy(
    tokenizer: Any,
    frozen_docs: Sequence[FrozenDoc],
    history_units: Sequence[Sequence[Unit]],
    kind: str,
    chunk_size: Optional[int],
    max_doc_length: int,
    encode_fn: Callable[[Any, str], Sequence[int]],
    decode_fn: Callable[[Any, Sequence[int]], str],
    *,
    token_len_fn: Optional[Callable[..., int]] = None,
    flatten_fn: Optional[Callable[[Sequence[Unit]], str]] = None,
    split_fn: Optional[Callable[..., List[Message]]] = None,
    need_content_tokens: bool = False,
) -> Tuple[List[Message], Dict[str, Any]]:
    """Re-cut the frozen content stream under ``kind``.

    ``token_len_fn`` / ``flatten_fn`` / ``split_fn`` are keyword-only
    extensions of the spec'd signature (the spec named the three injected
    callbacks on ``fit_history_with_provenance`` only): ``fixed`` needs the
    chat-template overhead to size its window so the wrapped chunk still fits
    ``max_doc_length``, and ``structural`` needs the turn renderer plus the
    oversized-block fallback.  Every one of them is optional and the branch
    degrades to a pass-through when it is missing.

    ``need_content_tokens`` is opt-in accounting, NOT free: measuring it costs
    one extra encode of the whole frozen stream (up to ~24k tokens per
    example) plus one encode per emitted chunk, which roughly doubles the
    history-side tokenization cost of a dataset build.  When it is False both
    ``content_tokens`` and ``policy_content_tokens`` are ``None`` — "not
    measured", deliberately distinguishable from a measured 0.  Only the eval
    driver (presented-token / gist-declaration checks) needs the numbers.

    ``meta["doc_turn_indices"]`` is aligned with the returned messages
    (``None`` where the policy destroyed turn provenance) and is what
    ``split_delayed`` consumes.
    """

    texts = [doc.text for doc in frozen_docs]
    meta: Dict[str, Any] = {
        "chunk_kind": kind,
        "chunk_size": chunk_size,
        "content_tokens": None,
        "policy_content_tokens": None,
        "structural_fallback_docs": 0,
        "structural_partial_docs": 0,
        "structural_repacked_docs": 0,
        "structural_passthrough_docs": 0,
        "fixed_window_tokens": None,
        "doc_turn_indices": [],
    }
    messages: List[Message] = []
    turn_indices: List[Optional[int]] = []

    def _emit(text: str, turn_index: Optional[int]) -> None:
        if not text:
            return
        messages.append({"role": "user", "content": text})
        turn_indices.append(turn_index)

    if kind == "agent-turn":
        for doc in frozen_docs:
            _emit(doc.text, doc.turn_index)
    elif kind == "fixed":
        if not chunk_size or chunk_size <= 0:
            raise ValueError(f"fixed chunk policy needs a positive chunk_size, got {chunk_size!r}")
        window = chunk_size
        if token_len_fn is not None:
            overhead = token_len_fn(tokenizer, {"role": "user", "content": ""})
            window = min(window, max_doc_length - overhead - FIXED_WINDOW_MARGIN)
        window = max(1, window)
        meta["fixed_window_tokens"] = window
        stream = FROZEN_JOIN.join(texts)
        ids = list(encode_fn(tokenizer, stream)) if stream else []
        # Same join, same encode as ``_content_tokens``: reuse it instead of
        # paying a second full-stream encode for the accounting.
        if need_content_tokens:
            meta["content_tokens"] = len(ids)
        for start in range(0, len(ids), window):
            _emit(decode_fn(tokenizer, ids[start : start + window]), None)
    elif kind == "structural":
        for doc in frozen_docs:
            units = (
                list(history_units[doc.turn_index])
                if 0 <= doc.turn_index < len(history_units)
                else []
            )
            if doc.num_parts != 1:
                # A shard of an oversized turn: unit provenance no longer maps
                # onto this text, so the shard passes through untouched.
                meta["structural_partial_docs"] += 1
                _emit(doc.text, doc.turn_index)
                continue
            blocks = _pair_units(units) if (units and flatten_fn is not None) else []
            if len(blocks) <= 1:
                meta["structural_passthrough_docs"] += 1
                _emit(doc.text, doc.turn_index)
                continue
            meta["structural_repacked_docs"] += 1
            for block in blocks:
                text = flatten_fn(block)
                if not text:
                    continue
                message = {"role": "user", "content": text}
                oversized = (
                    token_len_fn is not None
                    and split_fn is not None
                    and token_len_fn(tokenizer, message) > max_doc_length
                )
                if oversized:
                    meta["structural_fallback_docs"] += 1
                    for part in split_fn(tokenizer, message, max_doc_length):
                        _emit(str(part.get("content") or ""), doc.turn_index)
                else:
                    _emit(text, doc.turn_index)
    else:
        raise ValueError(f"Unsupported chunk policy kind: {kind!r}")

    meta["doc_turn_indices"] = turn_indices
    if need_content_tokens:
        if meta["content_tokens"] is None:  # not already taken by the fixed branch
            meta["content_tokens"] = _content_tokens(tokenizer, texts, encode_fn)
        meta["policy_content_tokens"] = sum(
            len(encode_fn(tokenizer, message["content"])) for message in messages
        )
    return messages, meta


def split_delayed(
    policy_docs: Sequence[Message],
    frozen_docs: Sequence[FrozenDoc],
    delay_recent_turns: int,
    *,
    doc_turn_indices: Optional[Sequence[Optional[int]]] = None,
) -> Tuple[List[Message], List[Message]]:
    """Hold back the docs of the last ``k`` distinct turns (turn granularity).

    ``k == 0`` is the identity.  The ``fixed`` policies have no turn
    provenance left, so any ``k > 0`` on them raises — bytes-matched delayed
    accounting (24号 B.4.2) needs to know which turn each doc came from.
    """

    if delay_recent_turns < 0:
        raise ValueError(f"delay_recent_turns must be non-negative, got {delay_recent_turns}")
    if delay_recent_turns == 0:
        return list(policy_docs), []
    if doc_turn_indices is None:
        if len(policy_docs) != len(frozen_docs):
            raise ValueError(
                "split_delayed needs doc_turn_indices when the policy changed the doc count "
                f"({len(policy_docs)} docs vs {len(frozen_docs)} frozen docs)"
            )
        doc_turn_indices = [doc.turn_index for doc in frozen_docs]
    if len(doc_turn_indices) != len(policy_docs):
        raise ValueError(
            f"doc_turn_indices length {len(doc_turn_indices)} != policy doc count {len(policy_docs)}"
        )
    if any(index is None for index in doc_turn_indices):
        raise ValueError(
            "delay_recent_turns > 0 requires turn provenance; the 'fixed' chunk policies "
            "discard turn boundaries"
        )
    distinct = sorted({int(index) for index in doc_turn_indices})
    delayed_turns = set(distinct[-delay_recent_turns:])
    kept: List[Message] = []
    delayed: List[Message] = []
    for message, index in zip(policy_docs, doc_turn_indices):
        (delayed if int(index) in delayed_turns else kept).append(message)
    return kept, delayed
