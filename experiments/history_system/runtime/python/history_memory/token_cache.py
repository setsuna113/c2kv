"""Bounded, session-local memo for exact native chat-template tokenization."""

from __future__ import annotations

import json
import hashlib
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Mapping, Sequence


class NativeTokenCache:
    """Cache complete template calls and proven rendered prologues.

    The tokenizer is owned by one controller. Observable rendering settings are
    checked on every lookup outside an explicit synchronous rendering scope,
    and once on entry to its outermost scope. The tokenizer must remain
    unchanged while that scope is active; a detected change clears the memo.
    """

    def __init__(self, tokenizer: Any, *, max_entries: int = 512,
                 max_token_ids: int = 200_000) -> None:
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        if type(max_token_ids) is not int or max_token_ids <= 0:
            raise ValueError("max_token_ids must be a positive integer")
        self.tokenizer = tokenizer
        self.max_entries = max_entries
        self.max_token_ids = max_token_ids
        self.session_id: str | None = None
        self._entries: OrderedDict[tuple[str, str], tuple[int, ...]] = OrderedDict()
        self._token_ids = 0
        self._config = self._config_key()
        self._prologue_supported: bool | None = None
        self._scope_depth = 0
        self._scope_config: str | None = None
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.clears = 0
        self.skips = 0
        self.segment_hits = 0
        self.segment_misses = 0

    def _config_key(self) -> str | None:
        try:
            return json.dumps({
                "chat_template": getattr(self.tokenizer, "chat_template", None),
                "special_tokens_map": getattr(self.tokenizer, "special_tokens_map", None),
                "added_tokens_encoder": getattr(self.tokenizer, "added_tokens_encoder", None),
                "init_kwargs": getattr(self.tokenizer, "init_kwargs", None),
            }, ensure_ascii=False, allow_nan=False, sort_keys=True,
                default=self._tokenizer_json)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tokenizer_json(value: Any) -> dict[str, Any]:
        # Hugging Face init_kwargs includes tokenizers.AddedToken instances.
        # Describe every rendering-relevant field instead of using repr(),
        # whose format is not a stable configuration identity.
        try:
            from tokenizers import AddedToken
        except ImportError:
            AddedToken = ()
        if isinstance(value, AddedToken):
            return {
                "type": "AddedToken", "content": value.content,
                "single_word": value.single_word, "lstrip": value.lstrip,
                "rstrip": value.rstrip, "normalized": value.normalized,
                "special": value.special,
            }
        raise TypeError(f"Unsupported tokenizer config value: {type(value).__name__}")

    def _input_key(self, messages: Sequence[Mapping[str, Any]], *,
                   tools: Any, generation: bool) -> str:
        source = json.dumps({
            "messages": list(messages), "tools": tools,
            "generation": generation,
            "tokenize": True, "enable_thinking": False,
            "truncation": False,
        }, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        return hashlib.sha256(source.encode("utf-8")).hexdigest()

    @contextmanager
    def rendering_scope(self):
        """Check tokenizer rendering once for a synchronous, immutable-config scope.

        The tokenizer must not change until the outermost scope exits.
        """
        if self._scope_depth == 0:
            config = self._config_key()
            if config != self._config:
                self.clear()
                self._config = config
            self._scope_config = config
        self._scope_depth += 1
        try:
            yield
        finally:
            self._scope_depth -= 1
            if self._scope_depth == 0:
                self._scope_config = None

    def clear(self) -> None:
        self._entries.clear()
        self._token_ids = 0
        self._prologue_supported = None
        self.clears += 1

    def set_session(self, session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a nonempty string")
        if session_id != self.session_id:
            self.clear()
            self.session_id = session_id

    def get(self, tokenizer: Any, messages: Sequence[Mapping[str, Any]],
            *, tools: Any, generation: bool) -> tuple[int, ...] | None:
        if tokenizer is not self.tokenizer:
            raise ValueError("NativeTokenCache belongs to another tokenizer")
        config = self._scope_config if self._scope_depth else self._config_key()
        if config != self._config:
            self.clear()
            self._config = config
        if self.session_id is None or config is None:
            self.skips += 1
            return None
        try:
            key = self._input_key(messages, tools=tools, generation=generation)
        except (TypeError, ValueError):
            self.skips += 1
            return None
        entry_key = ("full", key)
        ids = self._entries.get(entry_key)
        if ids is not None:
            self._entries.move_to_end(entry_key)
            self.hits += 1
            return ids
        self.misses += 1
        return None

    def put(self, tokenizer: Any, messages: Sequence[Mapping[str, Any]],
            ids: tuple[int, ...], *, tools: Any, generation: bool) -> None:
        if tokenizer is not self.tokenizer:
            raise ValueError("NativeTokenCache belongs to another tokenizer")
        if self.session_id is None or self._config is None or len(ids) > self.max_token_ids:
            return
        config = self._scope_config if self._scope_depth else self._config_key()
        if config != self._config:
            self.clear()
            self._config = config
            return
        try:
            key = self._input_key(messages, tools=tools, generation=generation)
        except (TypeError, ValueError):
            return
        self._put_entry(("full", key), ids)

    def _put_entry(self, key: tuple[str, str], ids: tuple[int, ...]) -> None:
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._token_ids -= len(previous)
        self._entries[key] = ids
        self._token_ids += len(ids)
        while len(self._entries) > self.max_entries or self._token_ids > self.max_token_ids:
            _, removed = self._entries.popitem(last=False)
            self._token_ids -= len(removed)
            self.evictions += 1

    def supports_prologue_split(self) -> bool:
        """Only the verified Qwen ByteLevel BPE profile has a safe special-token seam."""
        if self.session_id is None or self._config is None:
            return False
        if self._prologue_supported is not None:
            return self._prologue_supported
        try:
            backend = json.loads(self.tokenizer.backend_tokenizer.to_str())
            split = backend["pre_tokenizer"]
            pieces = split["pretokenizers"]
            added = {item["content"]: item for item in backend["added_tokens"]}
            markers = ("<|im_start|>", "<|im_end|>")
            marker_ok = all(
                added[marker]["special"] is True
                and added[marker]["normalized"] is False
                and added[marker]["single_word"] is False
                and added[marker]["lstrip"] is False
                and added[marker]["rstrip"] is False
                for marker in markers
            )
            other_tokens_ok = all(
                not item["lstrip"] and not item["rstrip"]
                and not any(marker in item["content"] for marker in markers)
                and "\n<" not in item["content"]
                for item in backend["added_tokens"]
                if item["content"] not in markers
            )
            self._prologue_supported = bool(
                self.tokenizer.is_fast
                and callable(self.tokenizer.encode)
                and backend["model"]["type"] == "BPE"
                and backend["model"].get("dropout") in (None, 0, 0.0)
                and backend["normalizer"] == {"type": "NFC"}
                and split["type"] == "Sequence"
                and len(pieces) == 2
                and pieces[0] == {
                    "type": "Split",
                    "pattern": {"Regex":
                        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},
                    "behavior": "Isolated", "invert": False,
                }
                and pieces[1] == {
                    "type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": True, "use_regex": False,
                }
                and backend["post_processor"] == {
                    "type": "ByteLevel", "add_prefix_space": False,
                    "trim_offsets": False, "use_regex": False,
                }
                and marker_ok and other_tokens_ok
            )
        except (AttributeError, KeyError, TypeError, ValueError):
            self._prologue_supported = False
        return self._prologue_supported

    def get_prologue(self, rendered: str) -> tuple[int, ...] | None:
        key = ("prologue", rendered)
        ids = self._entries.get(key)
        if ids is not None:
            self._entries.move_to_end(key)
        return ids

    def put_prologue(self, rendered: str, ids: tuple[int, ...]) -> None:
        if self.session_id is not None and self._config is not None and len(ids) <= self.max_token_ids:
            self._put_entry(("prologue", rendered), ids)

    def get_segment(self, rendered: str) -> tuple[int, ...] | None:
        ids = self._entries.get(("segment", rendered))
        if ids is None:
            self.segment_misses += 1
        else:
            self._entries.move_to_end(("segment", rendered))
            self.segment_hits += 1
        return ids

    def put_segment(self, rendered: str, ids: tuple[int, ...]) -> None:
        if self.session_id is not None and self._config is not None and len(ids) <= self.max_token_ids:
            self._put_entry(("segment", rendered), ids)

    def info(self) -> dict[str, int | str | None]:
        return {
            "session_id": self.session_id,
            "entries": len(self._entries), "token_ids": self._token_ids,
            "max_entries": self.max_entries, "max_token_ids": self.max_token_ids,
            "hits": self.hits, "misses": self.misses,
            "segment_hits": self.segment_hits, "segment_misses": self.segment_misses,
            "evictions": self.evictions, "clears": self.clears,
            "skips": self.skips,
        }
