"""Replay the actor's exact text after validating structured API echoes.

Tool-call JSON serialization is not a KV-preserving operation. The serving
receipt supplies the original generated text; only its semantic API echo is
accepted from a harness, and the server checks token-prefix identity again.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field


def _signature(message):
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments") or "{}"
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                pass
        calls.append((call.get("id"), function.get("name"), arguments))
    return message.get("role", "assistant"), message.get("content") or "", calls


@dataclass
class RawActorHistory:
    turns: dict = field(default_factory=dict)

    def prepare(self, messages):
        result = copy.deepcopy(messages)
        for index, (expected, raw_text) in self.turns.items():
            if index >= len(result) or _signature(result[index]) != expected:
                raise ValueError("Exact KV continuation requires the unchanged prior assistant action")
            result[index]["content"] = raw_text
            result[index].pop("tool_calls", None)
        return result

    def commit(self, source_messages, response, *, benchmark=None):
        receipt = (response.get("metadata") or {}).get("persistent_history_session") or {}
        if receipt.get("continuation_mode") != "exact_generated_prefix":
            raise ValueError("Exact KV method requires an exact-generated-prefix serving receipt")
        raw_text = receipt.get("generated_text")
        if not isinstance(raw_text, str):
            raise ValueError("Exact KV serving receipt lacks raw generated text")
        message = response["choices"][0]["message"]
        if benchmark == "bfcl":
            from benchmarks.bfcl_response import normalize_native_message

            message = normalize_native_message(message, response["id"])
        index = len(source_messages)
        if index in self.turns:
            raise ValueError("Exact KV conversation cannot overwrite a committed actor turn")
        self.turns[index] = (_signature(message), raw_text)


_states = {}


def state_for(conversation):
    return _states.setdefault(conversation, RawActorHistory())


def reset_state():
    _states.clear()
