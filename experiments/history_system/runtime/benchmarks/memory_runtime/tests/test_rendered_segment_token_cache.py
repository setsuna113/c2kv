"""A real ByteLevel BPE tokenizer checks exact rendered-segment reuse."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python"), str(Path(__file__).parent)]

from history_memory.packing import native_ids
from history_memory.token_cache import NativeTokenCache
from test_prologue_token_cache import _full, _tokenizer, TEMPLATE


PARALLEL_TEMPLATE = TEMPLATE.replace(
    "message.content + '<|im_end|>\\n'",
    "(message.content or '') + (message.tool_calls | tojson if message.tool_calls is defined else '') + '<|im_end|>\\n'",
)


def test_append_and_rewrite_reuse_only_unchanged_rendered_segments(monkeypatch):
    tokenizer = _tokenizer()
    cache = NativeTokenCache(tokenizer, max_token_ids=1_000_000)
    cache.set_session("s")
    tools = [{"name": "lookup", "description": "raw schema"}]
    system = {"role": "system", "content": "Answer with tools."}
    old = {"role": "user", "content": "first question α"}
    reply = {"role": "assistant", "content": "first answer"}
    next_user = {"role": "user", "content": "second question"}
    first = [system, old, reply, next_user]
    assert native_ids(tokenizer, first, tools=tools, generation=True,
                      token_cache=cache) == _full(tokenizer, first, tools=tools, generation=True)

    encoded = []
    original = tokenizer.encode

    def record(value, **kwargs):
        encoded.append(value)
        return original(value, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", record)
    appended = [*first, {"role": "assistant", "content": "second answer"},
                {"role": "user", "content": "third question"}]
    assert native_ids(tokenizer, appended, tools=tools, generation=True,
                      token_cache=cache) == _full(tokenizer, appended, tools=tools, generation=True)
    assert encoded
    assert all("first question α" not in value and "first answer" not in value
               for value in encoded)
    assert cache.info()["segment_hits"] > 0

    encoded.clear()
    rewritten = [system, {"role": "user", "content": "corrected question β"},
                 reply, next_user]
    assert native_ids(tokenizer, rewritten, tools=tools, generation=True,
                      token_cache=cache) == _full(tokenizer, rewritten, tools=tools, generation=True)
    assert any("corrected question β" in value for value in encoded)


def test_complete_render_handles_parallel_tool_calls_and_tool_schema_changes():
    tokenizer = _tokenizer()
    tokenizer.chat_template = PARALLEL_TEMPLATE
    cache = NativeTokenCache(tokenizer, max_token_ids=1_000_000)
    cache.set_session("parallel")
    system = {"role": "system", "content": "Use both tools."}
    assistant = {"role": "assistant", "content": None, "tool_calls": [
        {"function": {"name": "lookup", "arguments": '{"a":1}'}},
        {"function": {"name": "lookup", "arguments": '{"b":2}'}}]}
    messages = [system, {"role": "user", "content": "find two"}, assistant,
                {"role": "tool", "content": "first result"},
                {"role": "tool", "content": "second result"},
                {"role": "user", "content": "next"}]
    variants = [
        (messages, [{"name": "lookup", "description": "old"}]),
        ([*messages[:-1], {"role": "user", "content": "next changed"}],
         [{"name": "lookup", "description": "old"}]),
        (messages, [{"name": "lookup", "description": "new"}]),
    ]
    for source, tools in variants:
        assert native_ids(tokenizer, source, tools=tools, generation=True,
                          token_cache=cache) == _full(tokenizer, source, tools=tools,
                                                       generation=True)
    assert cache.info()["segment_hits"] > 0


def test_segment_entries_share_bounds_and_clear_on_session_or_template_change():
    tokenizer = _tokenizer()
    cache = NativeTokenCache(tokenizer, max_entries=4, max_token_ids=300)
    cache.set_session("one")
    system = {"role": "system", "content": "instructions"}
    for index in range(8):
        messages = [system, {"role": "user", "content": f"old {index}"},
                    {"role": "assistant", "content": "answer"},
                    {"role": "user", "content": f"current {index}"}]
        assert native_ids(tokenizer, messages, generation=True,
                          token_cache=cache) == _full(tokenizer, messages, generation=True)
        assert cache.info()["entries"] <= 4
        assert cache.info()["token_ids"] <= 300
    assert cache.info()["evictions"] > 0
    cache.set_session("two")
    assert cache.info()["entries"] == 0
    tokenizer.chat_template += " "
    messages = [system, {"role": "user", "content": "new"}]
    assert native_ids(tokenizer, messages, token_cache=cache) == _full(tokenizer, messages)
    assert cache.info()["clears"] >= 3


def test_unsupported_profile_keeps_original_full_tokenization(monkeypatch):
    import tokenizers

    tokenizer = _tokenizer()
    tokenizer.backend_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()
    cache = NativeTokenCache(tokenizer)
    cache.set_session("unsupported")
    calls = []
    original = tokenizer.apply_chat_template

    def record(messages, **kwargs):
        calls.append(kwargs["tokenize"])
        return original(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", record)
    messages = [{"role": "system", "content": "instructions"},
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"}]
    assert native_ids(tokenizer, messages, token_cache=cache) == _full(tokenizer, messages)
    assert calls == [True, True]
    assert not any(key[0] == "segment" for key in cache._entries)
