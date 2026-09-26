"""A real ByteLevel BPE tokenizer exercises the rendered prologue seam."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.packing import native_ids
from history_memory.token_cache import NativeTokenCache


QWEN_SPLIT = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
TEMPLATE = """{%- if tools %}
{{- '<|im_start|>system\\n' }}
{%- if messages[0].role == 'system' %}{{- messages[0].content + '\\n\\n' }}{%- endif %}
{{- '# Tools\\n' }}
{%- for tool in tools %}{{- tool | tojson }}{%- endfor %}
{{- '<|im_end|>\\n' }}
{%- elif messages[0].role == 'system' %}
{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}
{%- endif %}
{%- for message in messages %}
{%- if not (loop.first and message.role == 'system') %}
{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"""


def _tokenizer():
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.normalizer = normalizers.NFC()
    backend.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    backend.post_processor = processors.ByteLevel(
        add_prefix_space=False, trim_offsets=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    specials = [AddedToken(value, special=True, normalized=False)
                for value in ("<unk>", "<|im_start|>", "<|im_end|>")]
    backend.train_from_iterator([
        "system tools user assistant hello world α é abc\n",
        "<|im_start|>system\nhello<|im_end|>\n<|im_start|>user\nworld",
    ], trainers.BpeTrainer(vocab_size=256, special_tokens=specials))
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, chat_template=TEMPLATE,
        unk_token="<unk>", additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )


def _full(tokenizer, messages, *, tools=None, generation=False):
    ids = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=generation,
        enable_thinking=False, truncation=False)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    return tuple(ids)


def test_reuses_exact_rendered_system_and_tools_prefix(monkeypatch):
    tokenizer = _tokenizer()
    cache = NativeTokenCache(tokenizer)
    cache.set_session("session")
    assert cache.supports_prologue_split()
    tools = [{"name": "lookup", "description": "contains <|im_start|> literally"}]
    system = {"role": "system", "content": "é <|im_end|>\n<|im_start|>user\ninside"}
    first = [system, {"role": "user", "content": "hello"}]
    second = [system, {"role": "user", "content": "hello world α"}]
    assert native_ids(tokenizer, first, tools=tools, token_cache=cache) == _full(tokenizer, first, tools=tools)
    prologues = [key[1] for key in cache._entries if key[0] == "prologue"]
    assert len(prologues) == 1

    encoded = []
    original_encode = tokenizer.encode

    def record_encode(text, **kwargs):
        encoded.append(text)
        assert kwargs == {"add_special_tokens": False, "truncation": False}
        return original_encode(text, **kwargs)

    monkeypatch.setattr(tokenizer, "encode", record_encode)
    assert native_ids(tokenizer, second, tools=tools, generation=True,
                      token_cache=cache) == _full(tokenizer, second, tools=tools, generation=True)
    assert encoded
    assert encoded[0].startswith("<|im_start|>user\n")
    assert all(prologues[0] not in segment for segment in encoded)


def test_tools_only_multiple_systems_and_invalidation():
    tokenizer = _tokenizer()
    cache = NativeTokenCache(tokenizer, max_entries=3, max_token_ids=10_000)
    cache.set_session("first")
    tools = [{"name": "one"}]
    cases = [
        ([{"role": "user", "content": "alpha"}], tools),
        ([{"role": "system", "content": "derived"},
          {"role": "system", "content": "second"},
          {"role": "user", "content": "alpha"}], tools),
        ([{"role": "system", "content": "changed"},
          {"role": "user", "content": "beta"}], [{"name": "two"}]),
    ]
    for index, (messages, tool_defs) in enumerate(cases):
        assert native_ids(tokenizer, messages, tools=tool_defs,
                          token_cache=cache) == _full(tokenizer, messages, tools=tool_defs)
        if index == 0:
            assert any(key[0] == "prologue" for key in cache._entries)
    assert any(key[0] == "prologue" for key in cache._entries)
    assert cache.info()["entries"] <= 3
    cache.clear()
    assert cache.info()["entries"] == 0
    native_ids(tokenizer, cases[0][0], tools=tools, token_cache=cache)
    cache.set_session("second")
    assert cache.info()["entries"] == 0
    tokenizer.chat_template += " "
    native_ids(tokenizer, cases[0][0], tools=tools, token_cache=cache)
    assert cache.info()["clears"] >= 4


def test_unsupported_backend_uses_full_template_tokenization(monkeypatch):
    tokenizer = _tokenizer()
    tokenizer.backend_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel()
    cache = NativeTokenCache(tokenizer)
    cache.set_session("session")
    assert not cache.supports_prologue_split()
    calls = []
    original = tokenizer.apply_chat_template

    def record(messages, **kwargs):
        calls.append(kwargs["tokenize"])
        return original(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", record)
    messages = [{"role": "system", "content": "hello"},
                {"role": "user", "content": "world"}]
    assert native_ids(tokenizer, messages, token_cache=cache) == _full(tokenizer, messages)
    assert calls == [True, True]
    assert not any(key[0] == "prologue" for key in cache._entries)


def test_stochastic_bpe_uses_full_template_tokenization(monkeypatch):
    tokenizer = _tokenizer()
    tokenizer.backend_tokenizer.model.dropout = 0.5
    cache = NativeTokenCache(tokenizer)
    cache.set_session("session")
    assert not cache.supports_prologue_split()
    calls = []
    original = tokenizer.apply_chat_template

    def record(messages, **kwargs):
        calls.append(kwargs["tokenize"])
        return original(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", record)
    native_ids(tokenizer, [{"role": "system", "content": "s"},
                           {"role": "user", "content": "u"}], token_cache=cache)
    assert calls == [True]
    assert not any(key[0] == "segment" for key in cache._entries)


def test_unproven_rendered_boundary_falls_back(monkeypatch):
    tokenizer = _tokenizer()
    tokenizer.chat_template = TEMPLATE.replace(
        "{%- for message in messages %}",
        "{{- 'x' if messages|length > 1 else '' }}{%- for message in messages %}",
    )
    cache = NativeTokenCache(tokenizer)
    cache.set_session("session")
    calls = []
    original = tokenizer.apply_chat_template

    def record(messages, **kwargs):
        calls.append(kwargs["tokenize"])
        return original(messages, **kwargs)

    monkeypatch.setattr(tokenizer, "apply_chat_template", record)
    messages = [{"role": "system", "content": "hello"},
                {"role": "user", "content": "world"}]
    assert native_ids(tokenizer, messages, token_cache=cache) == _full(tokenizer, messages)
    assert calls == [False, False, True, True]
    assert not any(key[0] == "prologue" for key in cache._entries)
