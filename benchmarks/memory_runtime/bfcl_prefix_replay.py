"""Strict frozen-prefix gate for BFCL continuation experiments.

The gate deliberately does not construct SDK response objects or call a live
endpoint.  A handler asks ``query`` before each request: a frozen prefix yields
an independent copy of its stored assistant message, the one verified branch
request yields ``None``, and every later request also yields ``None``.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from typing import Any


REPLAY_SCHEMA = "bfcl-frozen-prefix-replay-v1"


class FrozenPrefixReplayError(RuntimeError):
    """A request does not belong to the admitted frozen continuation."""


def _json_value(value: Any, *, label: str) -> None:
    """Reject Python values that would weaken exact JSON comparison."""
    if value is None or isinstance(value, (str, bool)):
        return
    if type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise FrozenPrefixReplayError(f"{label} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise FrozenPrefixReplayError(f"{label} has a non-string object key")
            _json_value(item, label=f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _json_value(item, label=f"{label}[{index}]")
        return
    raise FrozenPrefixReplayError(
        f"{label} must contain JSON-native values, got {type(value).__name__}"
    )


def _freeze_json(value: Any, *, label: str) -> tuple[str, Any]:
    _json_value(value, label=label)
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return encoded, json.loads(encoded)


def _same_json(expected: str, actual: Any, *, label: str) -> None:
    actual_encoded, _ = _freeze_json(actual, label=label)
    if actual_encoded != expected:
        raise FrozenPrefixReplayError(f"frozen prefix mismatch for {label}")


def _context_key(context: Mapping[str, Any], *, task_id: str, label: str) -> tuple[int, int]:
    required = ("benchmark", "task_id", "user_turn", "step", "attempt")
    if set(context) != set(required):
        raise FrozenPrefixReplayError(
            f"{label} context must contain exactly {', '.join(required)}"
        )
    if type(context["benchmark"]) is not str or context["benchmark"] != "bfcl":
        raise FrozenPrefixReplayError(f"{label} context requires benchmark bfcl")
    if type(context["task_id"]) is not str or context["task_id"] != task_id:
        raise FrozenPrefixReplayError(f"{label} context has a different task_id")
    if type(context["attempt"]) is not int or context["attempt"] != 0:
        raise FrozenPrefixReplayError(f"{label} context requires integer attempt 0")
    for field in ("user_turn", "step"):
        if type(context[field]) is not int or context[field] < 0:
            raise FrozenPrefixReplayError(f"{label} context requires nonnegative integer {field}")
    return context["user_turn"], context["step"]


def _entry(value: Any, *, task_id: str, label: str, assistant: bool) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise FrozenPrefixReplayError(f"{label} must be an object")
    required = {"context", "request_messages", "request_tools"}
    if assistant:
        required.add("assistant_message")
    missing = sorted(required - set(value))
    if missing:
        raise FrozenPrefixReplayError(f"{label} is missing {', '.join(missing)}")
    context = value["context"]
    messages = value["request_messages"]
    tools = value["request_tools"]
    if not isinstance(context, Mapping) or not isinstance(messages, list) or not isinstance(tools, list):
        raise FrozenPrefixReplayError(f"{label} has invalid context, messages, or tools")
    _context_key(context, task_id=task_id, label=label)
    _, frozen = _freeze_json(
        {
            "context": context,
            "request_messages": messages,
            "request_tools": tools,
            **({"assistant_message": value["assistant_message"]} if assistant else {}),
        },
        label=label,
    )
    if assistant:
        message = frozen["assistant_message"]
        if not isinstance(message, dict) or message.get("role") != "assistant":
            raise FrozenPrefixReplayError(f"{label} assistant_message must be an assistant object")
    return frozen


class FrozenPrefixReplay:
    """Yield a finite stored prefix, then permanently delegate to live generation.

    ``query`` returns a copied assistant message during the prefix and ``None``
    once the exact branch request has been observed.  The caller owns SDK-object
    construction and the actual live request after receiving ``None``.
    """

    def __init__(self, bundle: Mapping[str, Any]) -> None:
        if not isinstance(bundle, Mapping):
            raise FrozenPrefixReplayError("frozen replay bundle must be an object")
        task_id = bundle.get("task_id")
        replay = bundle.get("replay")
        branch = bundle.get("branch")
        if type(task_id) is not str or not task_id:
            raise FrozenPrefixReplayError("frozen replay bundle needs a nonempty task_id")
        if not isinstance(replay, list):
            raise FrozenPrefixReplayError("frozen replay bundle needs replay list")
        self.task_id = task_id
        self._replay = [
            _entry(value, task_id=task_id, label=f"replay[{index}]", assistant=True)
            for index, value in enumerate(replay)
        ]
        self._branch = _entry(branch, task_id=task_id, label="branch", assistant=False)
        keys = [
            _context_key(item["context"], task_id=task_id, label=f"replay[{index}]")
            for index, item in enumerate(self._replay)
        ]
        branch_key = _context_key(self._branch["context"], task_id=task_id, label="branch")
        if any(later <= earlier for earlier, later in zip(keys, keys[1:])):
            raise FrozenPrefixReplayError("replay contexts must move strictly forward")
        if keys and branch_key <= keys[-1]:
            raise FrozenPrefixReplayError("branch context must follow every replay context")
        self._index = 0
        self._branch_reached = False
        self._last_context: tuple[int, int] | None = None

    @property
    def replayed_requests(self) -> int:
        return self._index

    @property
    def branch_reached(self) -> bool:
        return self._branch_reached

    def receipt(self) -> dict[str, Any]:
        """Return a JSON-native progress receipt without exposing future requests."""
        return {
            "schema": REPLAY_SCHEMA,
            "task_id": self.task_id,
            "replayed_requests": self._index,
            "branch_reached": self._branch_reached,
            "source_verified": self._branch_reached and self._index == len(self._replay),
        }

    def _validate_live_context(self, context: Any) -> tuple[int, int]:
        if not isinstance(context, Mapping):
            raise FrozenPrefixReplayError("query context must be an object")
        key = _context_key(context, task_id=self.task_id, label="query")
        if self._last_context is not None and key <= self._last_context:
            raise FrozenPrefixReplayError("query context must move strictly forward")
        return key

    def _match(self, entry: Mapping[str, Any], context: Any, messages: Any, tools: Any, *, label: str) -> None:
        _same_json(_freeze_json(entry["context"], label=f"{label}.context")[0], context, label=f"{label}.context")
        _same_json(_freeze_json(entry["request_messages"], label=f"{label}.messages")[0], messages, label=f"{label}.messages")
        _same_json(_freeze_json(entry["request_tools"], label=f"{label}.tools")[0], tools, label=f"{label}.tools")

    def query(self, context: Mapping[str, Any], messages: list[Any], tools: list[Any]) -> dict[str, Any] | None:
        """Validate one BFCL request and return a frozen response, or ``None`` for live."""
        key = self._validate_live_context(context)
        _freeze_json(messages, label="query.messages")
        _freeze_json(tools, label="query.tools")

        if self._index < len(self._replay):
            entry = self._replay[self._index]
            self._match(entry, context, messages, tools, label=f"replay[{self._index}]")
            self._last_context = key
            self._index += 1
            return copy.deepcopy(entry["assistant_message"])

        if not self._branch_reached:
            self._match(self._branch, context, messages, tools, label="branch")
            self._last_context = key
            self._branch_reached = True
            return None

        self._last_context = key
        return None
