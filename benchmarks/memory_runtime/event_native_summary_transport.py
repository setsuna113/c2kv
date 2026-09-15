"""Durable event-native generation transport for historical text summaries.

The caller owns the auxiliary ``EventNativeGenerator``.  It must be distinct
from the actor generator even when both share one model runtime, and their
calls must be serialized by the caller.  This transport never closes or
resets either generator and never touches actor generation accounting.
"""
from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from history_memory.events import EventStore
from history_memory.packing import (
    MemoryView,
    PackedMemory,
    PackingBudgetError,
    pack_memory,
)

from .attempt_journal import AttemptJournal
from .event_native import memory_to_dict
from .event_native_draft import NATIVE_DRAFT_VERSION, decode_native_generation


SCHEMA = "a-event-native-summary-transport-v1"
PROMPT_TOKEN_CAP = 1024
ATTEMPTS_PER_TASK = 1152
MIN_COMPLETION_TOKENS = 16
MAX_COMPLETION_TOKENS = 128


def _json_snapshot(value: Any, *, field: str) -> Any:
    """Return a detached finite JSON value suitable for a durable trace."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field} must contain only finite JSON values") from error


class EventNativeSummaryTransport:
    """One-call native generator compatible with ``SummaryRenderer.generate``.

    Input and decoded model text are written only to the caller-selected local
    trace.  The shared :class:`AttemptJournal` remains content-free.
    """

    def __init__(
        self,
        generator: Any,
        tokenizer: Any,
        *,
        ratio: int,
        deadline_monotonic: float,
        journal: AttemptJournal,
        trace_path: str | Path,
        parent_request_id: Callable[[], str] | None = None,
    ) -> None:
        if not callable(getattr(generator, "generate", None)):
            raise TypeError("generator must expose generate")
        if not callable(getattr(generator, "decision_scope", None)):
            raise TypeError("generator must expose decision_scope")
        if type(ratio) is not int or ratio <= 0:
            raise ValueError("ratio must be a positive integer")
        if (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(float(deadline_monotonic))
        ):
            raise ValueError("deadline_monotonic must be finite")
        if not isinstance(journal, AttemptJournal):
            raise TypeError("a durable AttemptJournal is required")
        if parent_request_id is not None and not callable(parent_request_id):
            raise TypeError("parent_request_id must be callable or None")
        trace = Path(trace_path)
        if not trace.name:
            raise ValueError("trace_path must include a file name")
        if trace.resolve() == journal.path.resolve():
            raise ValueError("trace_path and AttemptJournal path must be distinct")

        self.generator = generator
        self.tokenizer = tokenizer
        self.ratio = ratio
        self.deadline_monotonic = float(deadline_monotonic)
        self.journal = journal
        self.trace_path = trace
        self.parent_request_id = parent_request_id
        self._attempts: dict[str, int] = {}
        self._attempt_lock = threading.Lock()
        self._trace_lock = threading.Lock()

    def _append_trace(self, record: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with self._trace_lock:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
            descriptor = os.open(self.trace_path, flags, 0o600)
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("summary trace append made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _request_id(self, context: Mapping[str, Any]) -> str:
        context_value = context.get("parent_request_id", context.get("request_id"))
        provided = self.parent_request_id() if self.parent_request_id is not None else None
        if provided is not None and context_value is not None and provided != context_value:
            raise ValueError("parent request identity disagrees with eval context")
        value = provided if provided is not None else context_value
        if not isinstance(value, str) or not value:
            raise ValueError("summary generation requires a parent request identity")
        return value

    @staticmethod
    def _task_id(context: Mapping[str, Any]) -> str:
        value = context.get("task_id")
        if not isinstance(value, str) or not value:
            raise ValueError("summary generation requires context.task_id")
        return value

    def _check_deadline(self) -> None:
        if time.monotonic() >= self.deadline_monotonic:
            raise TimeoutError("event-native summary deadline expired before generation")

    def _memory(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        summary_key: str,
    ) -> PackedMemory:
        store = EventStore.from_messages(f"summary:{summary_key}", messages)
        view = MemoryView(
            gist_event_ids=(),
            raw_event_ids=tuple(event.event_id for event in store.events),
            evidence_event_ids=(),
        )
        try:
            memory = pack_memory(
                store,
                view,
                self.tokenizer,
                tools=None,
                max_chunks=0,
                max_raw_tokens=PROMPT_TOKEN_CAP,
            )
        except PackingBudgetError as error:
            raise ValueError("native summary prompt exceeds its fixed cap") from error
        if memory.chunks or memory.view.gist_event_ids:
            raise RuntimeError("native summary input must remain raw-only")
        return memory

    def _reserve_attempt(
        self,
        *,
        task_id: str,
        request_id: str,
        context: Mapping[str, Any],
    ):
        with self._attempt_lock:
            index = self._attempts.get(task_id, 0) + 1
            if index > ATTEMPTS_PER_TASK:
                raise ValueError("event-native summary attempt budget exhausted")
            handle = self.journal.start("generation", index, request_id, context)
            self._attempts[task_id] = index
            return handle

    def __call__(
        self,
        messages: Sequence[Mapping[str, Any]],
        limit: int,
        context: Mapping[str, Any],
        key: str,
    ) -> dict[str, Any]:
        """Generate one bounded plain-text summary without retry or repair."""

        if not isinstance(context, Mapping):
            raise TypeError("context must be a mapping")
        if not isinstance(key, str) or not key:
            raise ValueError("summary key must be a nonempty string")
        if type(limit) is not int or not MIN_COMPLETION_TOKENS <= limit <= MAX_COMPLETION_TOKENS:
            raise ValueError(
                f"summary completion limit must be in "
                f"[{MIN_COMPLETION_TOKENS}, {MAX_COMPLETION_TOKENS}]"
            )
        context_snapshot = _json_snapshot(dict(context), field="context")
        messages_snapshot = _json_snapshot(list(messages), field="messages")
        task_id = self._task_id(context_snapshot)
        request_id = self._request_id(context_snapshot)
        self._check_deadline()
        memory = self._memory(messages_snapshot, summary_key=key)
        prompt_tokens = memory.costs(self.ratio)["resident_kv_tokens"]
        if prompt_tokens > PROMPT_TOKEN_CAP:
            raise ValueError("native summary prompt exceeds its fixed cap")

        started = time.monotonic()
        handle = None
        result = None
        usage = None
        draft = None
        start_record = None
        submitted = False
        try:
            with self.generator.decision_scope(session_id=None):
                # The scope can do local validation; reject an elapsed stage
                # immediately before reserving or submitting model work.
                self._check_deadline()
                handle = self._reserve_attempt(
                    task_id=task_id,
                    request_id=request_id,
                    context=context_snapshot,
                )
                start_record = {
                    "schema": SCHEMA,
                    "event": "started",
                    "status": "started",
                    "attempt_uid": handle.attempt_uid,
                    "attempt_index": handle.attempt_index,
                    "parent_request_id": request_id,
                    "eval_context": context_snapshot,
                    "summary_key": key,
                    "prompt_token_cap": PROMPT_TOKEN_CAP,
                    "completion_token_cap": limit,
                    "expected_prompt_tokens": prompt_tokens,
                    "decode": "greedy-native",
                    "actor_generation": False,
                    "automatic_retries": 0,
                    "submitted_to_generator": False,
                    # Source and output text are confined to this private local trace.
                    "request_messages": messages_snapshot,
                    "packed_memory": memory_to_dict(memory),
                }
                self._append_trace(start_record)
                submitted = True
                result = self.generator.generate(
                    memory,
                    ratio=self.ratio,
                    max_new_tokens=limit,
                    trace_context={
                        "attempt_uid": handle.attempt_uid,
                        "session_id": None,
                        "decision_key": key,
                        "phase": "summary",
                    },
                )
                completion_tokens = len(result.token_ids)
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }

            # Scope exit finalizes and releases auxiliary cache state.  Copy
            # stats and measure elapsed time only after that boundary.
            if not isinstance(result.stats, Mapping):
                raise ValueError("native summary generation stats must be a mapping")
            stats = _json_snapshot(dict(result.stats), field="generation stats")
            draft = decode_native_generation(
                self.tokenizer,
                result,
                call_id_prefix=f"summary_{handle.attempt_uid}",
            )
            if result.finish_reason not in ("stop", "length"):
                raise ValueError("native summary returned an unsupported finish reason")
            if usage["completion_tokens"] > limit:
                raise ValueError("native summary exceeded its completion token cap")
            if draft.status == "tool_calls":
                raise ValueError("native summary returned a tool call")
            if draft.status == "malformed":
                raise ValueError("native summary returned malformed native output")
            if draft.status != "text" or not isinstance(draft.content, str) or not draft.content.strip():
                raise ValueError("native summary returned no usable plain text")

            wall_sec = time.monotonic() - started
            response = {
                "content": draft.content,
                "finish_reason": result.finish_reason,
                "usage": usage,
                "attempt_uid": handle.attempt_uid,
                "wall_sec": wall_sec,
            }
            self._append_trace({
                **start_record,
                "event": "finished",
                "status": "completed",
                "submitted_to_generator": True,
                "usage": usage,
                "stats": stats,
                "native_draft": {
                    "version": NATIVE_DRAFT_VERSION,
                    **asdict(draft),
                },
                "response_view": copy.deepcopy(response),
                "wall_sec": wall_sec,
            })
            self.journal.finish(handle, "completed", usage=usage)
            return response
        except BaseException as error:
            if handle is None:
                raise
            wall_sec = time.monotonic() - started
            partial = getattr(self.generator, "last_generation_trace", None)
            if not (
                isinstance(partial, Mapping)
                and partial.get("attempt_uid") == handle.attempt_uid
            ):
                partial = None
            failure = {
                **(start_record or {
                    "schema": SCHEMA,
                    "attempt_uid": handle.attempt_uid,
                    "attempt_index": handle.attempt_index,
                    "parent_request_id": request_id,
                    "eval_context": context_snapshot,
                    "summary_key": key,
                }),
                "event": "finished",
                "status": "failed",
                "submitted_to_generator": submitted,
                "finish_reason": getattr(result, "finish_reason", None),
                "usage": usage,
                "stats": (
                    _json_snapshot(dict(result.stats), field="generation stats")
                    if result is not None and isinstance(result.stats, Mapping)
                    else None
                ),
                "native_draft": (
                    {"version": NATIVE_DRAFT_VERSION, **asdict(draft)}
                    if draft is not None
                    else None
                ),
                "partial_failure_trace": (
                    _json_snapshot(dict(partial), field="partial failure trace")
                    if partial is not None
                    else None
                ),
                "error_type": type(error).__name__,
                "wall_sec": wall_sec,
            }
            try:
                self._append_trace(failure)
            finally:
                self.journal.finish(handle, "failed", usage=usage)
            raise


__all__ = [
    "ATTEMPTS_PER_TASK",
    "EventNativeSummaryTransport",
    "MAX_COMPLETION_TOKENS",
    "MIN_COMPLETION_TOKENS",
    "PROMPT_TOKEN_CAP",
    "SCHEMA",
]
