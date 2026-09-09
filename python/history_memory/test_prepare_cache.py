"""Focused contracts for the opt-in preparation tokenization cache."""

from __future__ import annotations

import json

import pytest

from history_memory.events import EventStore
from history_memory.packing import (
    MemoryView,
    PackingCache,
    encode_event_chunks,
    native_ids,
    pack_memory,
    pack_target,
)


class CountingNonAdditiveTokenizer:
    """A template whose combined rendering cannot be derived per message."""

    def __init__(self) -> None:
        self.calls = 0

    def apply_chat_template(
        self,
        messages,
        *,
        tools=None,
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
        truncation=False,
    ):
        assert tokenize is True
        assert enable_thinking is False
        assert truncation is False
        self.calls += 1
        rendered = []
        if tools is not None:
            rendered.append(
                "<tools>"
                + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
                + "</tools>"
            )
        workspace_start = 0
        while (
            workspace_start < len(messages)
            and messages[workspace_start]["role"] == "system"
        ):
            rendered.append(
                "<system>"
                + json.dumps(
                    messages[workspace_start],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "</system>"
            )
            workspace_start += 1
        workspace = messages[workspace_start:]
        ordinary = [message for message in workspace if message["role"] != "assistant"]
        assistants = [message for message in workspace if message["role"] == "assistant"]
        rendered.append(f"<combined:{len(ordinary)}>")
        rendered.append(
            json.dumps(ordinary, ensure_ascii=False, separators=(",", ":"))
        )
        for assistant in assistants:
            rendered.append("<assistant>")
            rendered.append(
                json.dumps(assistant, ensure_ascii=False, separators=(",", ":"))
            )
        if add_generation_prompt:
            rendered.append("<assistant>")
        return tuple(ord(character) for character in "".join(rendered))


def _call(call_id: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "weather", "arguments": '{"city":"Cambridge"}'},
    }


def _messages(result: str = "sunny") -> list[dict[str, object]]:
    return [
        {"role": "system", "content": "Use the tools."},
        {"role": "user", "content": "Check the weather."},
        {"role": "assistant", "content": None, "tool_calls": [_call("call-1")]},
        {"role": "tool", "tool_call_id": "call-1", "content": result},
        {"role": "user", "content": "Keep that evidence."},
        {"role": "assistant", "content": "I will retain it."},
        {"role": "user", "content": "What should I do?"},
    ]


def _mixed_view() -> MemoryView:
    return MemoryView(
        gist_event_ids=("s:m1", "s:m5"),
        raw_event_ids=("s:m0", "s:m2", "s:m4", "s:m6"),
        evidence_event_ids=("s:m2",),
    )


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Look up weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


def test_cached_pack_and_target_are_exact_for_combined_template_and_evidence():
    store = EventStore.from_messages("s", _messages())
    view = _mixed_view()
    target = {"role": "assistant", "content": "Take an umbrella."}

    uncached_tokenizer = CountingNonAdditiveTokenizer()
    expected_memory = pack_memory(store, view, uncached_tokenizer, tools=TOOLS)
    expected_target = pack_target(uncached_tokenizer, target)

    tokenizer = CountingNonAdditiveTokenizer()
    cache = PackingCache("s")
    actual_memory = pack_memory(store, view, tokenizer, tools=TOOLS, cache=cache)
    actual_target = pack_target(tokenizer, target, cache=cache)
    first_call_count = tokenizer.calls

    assert actual_memory == expected_memory
    assert actual_target == expected_target
    assert actual_memory.costs(8)["raw_gist_overlap_events"] == 0
    assert pack_memory(store, view, tokenizer, tools=TOOLS, cache=cache) == expected_memory
    assert pack_target(tokenizer, target, cache=cache) == expected_target
    assert tokenizer.calls == first_call_count

    info = cache.info()
    assert info.event_hits == len(view.gist_event_ids)
    assert info.native_hits == 5
    assert info.event_entries == len(view.gist_event_ids)
    assert info.native_entries == 5


def test_native_cache_distinguishes_content_tools_generation_and_tokenizer():
    tokenizer = CountingNonAdditiveTokenizer()
    cache = PackingCache("s", max_native_entries=16)
    messages = [{"role": "user", "content": "hello"}]

    plain = native_ids(tokenizer, messages, cache=cache)
    assert native_ids(tokenizer, messages, cache=cache) == plain
    assert tokenizer.calls == 1

    generated = native_ids(tokenizer, messages, generation=True, cache=cache)
    with_tools = native_ids(tokenizer, messages, tools=[], cache=cache)
    changed = native_ids(
        tokenizer, [{"role": "user", "content": "changed"}], cache=cache
    )
    assert len({plain, generated, with_tools, changed}) == 4
    assert tokenizer.calls == 4

    other_tokenizer = CountingNonAdditiveTokenizer()
    assert native_ids(other_tokenizer, messages, cache=cache) == plain
    assert other_tokenizer.calls == 1
    assert cache.info().native_hits == 1
    assert cache.info().native_misses == 5


def test_event_cache_reuses_immutable_prefix_and_rejects_stale_id_only_hit():
    tokenizer = CountingNonAdditiveTokenizer()
    cache = PackingCache("s")
    original = EventStore.from_messages("s", _messages("sunny"))
    extended = EventStore.from_messages(
        "s", _messages("sunny") + [{"role": "assistant", "content": "Done."}]
    )
    changed = EventStore.from_messages("s", _messages("rainy"))

    original_chunks = encode_event_chunks(original, "s:m2", tokenizer, cache=cache)
    assert encode_event_chunks(extended, "s:m2", tokenizer, cache=cache) == original_chunks
    assert tokenizer.calls == 1

    changed_chunks = encode_event_chunks(changed, "s:m2", tokenizer, cache=cache)
    assert changed_chunks != original_chunks
    assert tokenizer.calls == 2
    assert cache.info().event_hits == 1
    assert cache.info().event_misses == 2

    other_session = EventStore.from_messages("other", _messages())
    with pytest.raises(ValueError, match="belongs to session"):
        encode_event_chunks(other_session, "other:m2", tokenizer, cache=cache)


def test_cache_bounds_evict_lru_entries_and_skip_oversize_values():
    store = EventStore.from_messages("s", _messages())
    tokenizer = CountingNonAdditiveTokenizer()
    cache = PackingCache(
        "s",
        max_event_entries=1,
        max_event_tokens=100_000,
        max_native_entries=1,
        max_native_tokens=100_000,
    )

    encode_event_chunks(store, "s:m1", tokenizer, cache=cache)
    encode_event_chunks(store, "s:m4", tokenizer, cache=cache)
    encode_event_chunks(store, "s:m1", tokenizer, cache=cache)
    native_ids(tokenizer, [{"role": "user", "content": "one"}], cache=cache)
    native_ids(tokenizer, [{"role": "user", "content": "two"}], cache=cache)
    native_ids(tokenizer, [{"role": "user", "content": "one"}], cache=cache)

    info = cache.info()
    assert info.event_entries == 1
    assert info.event_evictions == 2
    assert info.event_tokens <= cache.max_event_tokens
    assert info.native_entries == 1
    assert info.native_evictions == 2
    assert info.native_tokens <= cache.max_native_tokens

    too_small = PackingCache(
        "s", max_event_tokens=1, max_native_tokens=1
    )
    before = tokenizer.calls
    for _ in range(2):
        encode_event_chunks(store, "s:m1", tokenizer, cache=too_small)
        native_ids(tokenizer, [{"role": "user", "content": "large"}], cache=too_small)
    assert tokenizer.calls == before + 4
    small_info = too_small.info()
    assert small_info.event_entries == 0
    assert small_info.event_misses == 2
    assert small_info.event_skips == 2
    assert small_info.native_entries == 0
    assert small_info.native_misses == 2
    assert small_info.native_skips == 2


def test_event_chunk_parameters_are_part_of_the_cache_key():
    store = EventStore.from_messages("s", _messages())
    tokenizer = CountingNonAdditiveTokenizer()
    cache = PackingCache("s")

    first = encode_event_chunks(
        store, "s:m2", tokenizer, max_chunk_tokens=80, chunk_overlap=8, cache=cache
    )
    second = encode_event_chunks(
        store, "s:m2", tokenizer, max_chunk_tokens=81, chunk_overlap=8, cache=cache
    )
    assert first != second
    assert tokenizer.calls == 2
    assert cache.info().event_misses == 2
