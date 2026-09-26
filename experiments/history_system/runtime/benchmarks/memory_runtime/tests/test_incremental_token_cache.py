"""Exact tokenization reuse across complete event packs and changed views."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.events import EventStore
from history_memory.packing import MemoryView, native_ids, pack_memory
from history_memory.token_cache import NativeTokenCache


class CountingTokenizer:
    chat_template = "template-v1"

    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, *, tools=None,
                            add_generation_prompt=False, **kwargs):
        self.calls += 1
        prefix = "".join(
            f"<{message['role']}>" + json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            + "<end>" for message in messages if message["role"] == "system"
        )
        if tools:
            prefix += "<tools>" + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
        text = prefix + "".join(
            f"<{message['role']}>" + json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            + "<end>" for message in messages if message["role"] != "system"
        )
        text = self.chat_template + text if not messages else text.replace("<", f"<{self.chat_template}:")
        if add_generation_prompt:
            text += "<assistant>"
        # A boundary merge makes fragment-wise token concatenation incorrect.
        ids = []
        index = 0
        while index < len(text):
            if text[index:index + 2] == "}{":
                ids.append(0x110000)
                index += 2
            else:
                ids.append(ord(text[index]))
                index += 1
        return ids


def _rows():
    return [
        {"role": "system", "content": "Use the tool."},
        {"role": "user", "content": "old request"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call-1", "type": "function",
            "function": {"name": "lookup", "arguments": '{"key":"old"}'},
        }]},
        {"role": "tool", "tool_call_id": "call-1", "content": "old result"},
        {"role": "user", "content": "new request"},
    ]


def _view(store, gist):
    gist = set(gist)
    return MemoryView(
        gist_event_ids=tuple(event.event_id for event in store.events
                             if event.event_id in gist),
        raw_event_ids=tuple(event.event_id for event in store.events
                            if event.event_id not in gist),
    )


def test_append_edit_truncate_and_repack_match_uncached_full_outputs():
    tokenizer = CountingTokenizer()
    cache = NativeTokenCache(tokenizer, max_entries=64, max_token_ids=100_000)
    cache.set_session("s")
    rows = _rows()
    first = EventStore.from_messages("s", rows[:4])
    appended = EventStore.from_messages("s", rows)
    edited_rows = copy.deepcopy(rows)
    edited_rows[3]["content"] = "corrected result"
    edited = EventStore.from_messages("s", edited_rows)
    truncated = EventStore.from_messages("s", rows[:2])

    cases = [
        (first, ("s:m1",)),
        (appended, ("s:m1", "s:m2")),
        (appended, ("s:m2",)),
        (edited, ("s:m1", "s:m2")),
        (truncated, ()),
        (appended, ("s:m1", "s:m2")),
    ]
    for store, gist in cases:
        view = _view(store, gist)
        cached = pack_memory(store, view, tokenizer, tools=[{"function": {"name": "lookup"}}],
                             max_chunk_tokens=80, chunk_overlap=10, token_cache=cache)
        fresh = pack_memory(store, view, tokenizer, tools=[{"function": {"name": "lookup"}}],
                            max_chunk_tokens=80, chunk_overlap=10)
        assert cached == fresh
        assert cached.costs(8) == fresh.costs(8)
    assert cache.info()["hits"] > 0


def test_key_covers_tools_generation_visible_content_and_template_change():
    tokenizer = CountingTokenizer()
    cache = NativeTokenCache(tokenizer, max_entries=8, max_token_ids=10_000)
    cache.set_session("s")
    messages = [{"role": "user", "content": "same"}]
    plain = native_ids(tokenizer, messages, token_cache=cache)
    assert native_ids(tokenizer, messages, token_cache=cache) == plain
    assert cache.info()["hits"] == 1
    with_tools = native_ids(tokenizer, messages, tools=[{"name": "search"}], token_cache=cache)
    generated = native_ids(tokenizer, messages, generation=True, token_cache=cache)
    changed = native_ids(tokenizer, [{"role": "user", "content": "changed"}],
                         token_cache=cache)
    assert len({plain, with_tools, generated, changed}) == 4
    tokenizer.chat_template = "template-v2"
    after_change = native_ids(tokenizer, messages, token_cache=cache)
    assert after_change != plain
    assert cache.info()["entries"] == 1
    assert cache.info()["clears"] == 2  # session activation and config change


def test_session_switch_and_lru_bounds_clear_reused_inputs():
    tokenizer = CountingTokenizer()
    cache = NativeTokenCache(tokenizer, max_entries=2, max_token_ids=1000)
    cache.set_session("first")
    for content in ("a", "b", "c"):
        native_ids(tokenizer, [{"role": "user", "content": content}], token_cache=cache)
    assert cache.info()["entries"] == 2
    assert cache.info()["evictions"] == 1
    cache.set_session("second")
    assert cache.info()["entries"] == 0
    before = tokenizer.calls
    native_ids(tokenizer, [{"role": "user", "content": "c"}], token_cache=cache)
    assert tokenizer.calls == before + 1


def test_rendering_scope_checks_config_once_even_when_nested():
    tokenizer = CountingTokenizer()
    cache = NativeTokenCache(tokenizer)
    cache.set_session("s")
    messages = [{"role": "user", "content": "same"}]
    original_config_key = cache._config_key
    config_checks = 0

    def counted_config_key():
        nonlocal config_checks
        config_checks += 1
        return original_config_key()

    cache._config_key = counted_config_key
    with cache.rendering_scope():
        first = native_ids(tokenizer, messages, token_cache=cache)
        with cache.rendering_scope():
            assert native_ids(tokenizer, messages, token_cache=cache) == first
        assert native_ids(tokenizer, messages, token_cache=cache) == first
        assert config_checks == 1
    assert cache.info()["hits"] == 2
    assert native_ids(tokenizer, messages, token_cache=cache) == first
    assert config_checks == 2


def test_rendering_scope_invalidates_between_configurations():
    tokenizer = CountingTokenizer()
    cache = NativeTokenCache(tokenizer)
    cache.set_session("s")
    messages = [{"role": "user", "content": "same"}]
    with cache.rendering_scope():
        first = native_ids(tokenizer, messages, token_cache=cache)
    tokenizer.chat_template = "template-v2"
    with cache.rendering_scope():
        second = native_ids(tokenizer, messages, token_cache=cache)
        assert native_ids(tokenizer, messages, token_cache=cache) == second
    assert second != first
    assert cache.info()["entries"] == 1
    assert cache.info()["clears"] == 2  # session activation and config change


def test_rendering_scope_restores_per_call_checks_after_exception():
    tokenizer = CountingTokenizer()
    cache = NativeTokenCache(tokenizer)
    cache.set_session("s")
    messages = [{"role": "user", "content": "same"}]
    with pytest.raises(RuntimeError, match="interrupt"):
        with cache.rendering_scope():
            first = native_ids(tokenizer, messages, token_cache=cache)
            raise RuntimeError("interrupt")
    tokenizer.chat_template = "template-v2"
    second = native_ids(tokenizer, messages, token_cache=cache)
    assert second != first
    assert cache.info()["entries"] == 1
    assert cache.info()["clears"] == 2
