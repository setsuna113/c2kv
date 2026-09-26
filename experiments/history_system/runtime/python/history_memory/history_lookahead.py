"""Bounded CPU preparation of chunks from an already observed source prefix."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any, Mapping, Sequence

from .cross_turn_prewarm import plan_cross_turn_chunks
from .events import EventStore
from .packing import EncoderChunk


def _message_identity(message: Mapping[str, Any]) -> str:
    return json.dumps(message, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class _Pending:
    session_id: str
    prefix: tuple[str, ...]
    prefix_sha256: str
    submitted_perf_ns: int
    future: Future


class HistoryLookahead:
    """Prepare at most one observed-history job without queuing another one.

    The tokenizer is shared for read-only use. No controller state or token cache
    is touched by the worker. The caller must consume a finished job with poll;
    unfinished work is never awaited by submit or poll.
    """

    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        self._executor = ThreadPoolExecutor(max_workers=1,
                                             thread_name_prefix="history-lookahead")
        self._lock = Lock()
        self._pending: _Pending | None = None
        self._closed = False

    def submit(
        self, *, session_id: str, messages: Sequence[Mapping[str, Any]],
        benchmark: str | None, encoding_scope: str, max_chunk_tokens: int,
        chunk_overlap: int, atomic_unit_token_limit: int, source_cutoff: int,
    ) -> dict[str, Any]:
        """Snapshot visible input and schedule one CPU-only chunk plan."""
        with self._lock:
            if self._closed:
                return {"status": "skipped", "reason": "closed"}
            pending = self._pending
            if pending is not None:
                if not pending.future.done():
                    return {"status": "skipped", "reason": "worker_busy"}
                if pending.session_id == session_id:
                    return {"status": "skipped", "reason": "result_unconsumed"}
                # A completed result from another session can never be used here.
                self._pending = None
            try:
                snapshot = copy.deepcopy(tuple(messages))
                prefix = tuple(_message_identity(message) for message in snapshot)
            except Exception as error:
                return {"status": "skipped", "reason": "snapshot_unavailable",
                        "error": f"{type(error).__name__}: {error}"}
            digest = hashlib.sha256(json.dumps(prefix, ensure_ascii=False,
                                               separators=(",", ":")).encode("utf-8")).hexdigest()
            submitted_perf_ns = time.perf_counter_ns()
            try:
                future = self._executor.submit(
                    self._compute, session_id, snapshot, benchmark, encoding_scope,
                    max_chunk_tokens, chunk_overlap, atomic_unit_token_limit,
                    source_cutoff,
                )
            except RuntimeError as error:
                return {"status": "skipped", "reason": "worker_unavailable",
                        "error": f"{type(error).__name__}: {error}"}
            self._pending = _Pending(session_id, prefix, digest,
                                     submitted_perf_ns, future)
            return {"status": "queued", "source_prefix_sha256": digest,
                    "source_message_count": len(prefix),
                    "submitted_perf_ns": submitted_perf_ns}

    def poll(self, *, session_id: str,
             messages: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
        """Consume a finished job, retaining only an exact append-only prefix."""
        with self._lock:
            pending = self._pending
            if pending is None or not pending.future.done():
                return None
            self._pending = None

        timing: dict[str, int | None] = {
            "submitted_perf_ns": pending.submitted_perf_ns,
            "started_perf_ns": None,
            "finished_perf_ns": None,
            "compute_duration_ns": None,
        }
        base = {"chunks": (), "source_prefix_sha256": pending.prefix_sha256,
                "source_message_count": len(pending.prefix), "timing": timing}
        try:
            chunks, started_perf_ns, finished_perf_ns, error = pending.future.result()
        except Exception as error:
            return {**base, "status": "failed", "reason": "worker_exception",
                    "error": f"{type(error).__name__}: {error}"}
        timing.update(started_perf_ns=started_perf_ns,
                      finished_perf_ns=finished_perf_ns,
                      compute_duration_ns=finished_perf_ns - started_perf_ns)
        if error is not None:
            return {**base, "status": "failed", "reason": "worker_exception",
                    "error": error}
        if session_id != pending.session_id:
            return {**base, "status": "discarded", "reason": "session_changed"}
        try:
            current_prefix = tuple(_message_identity(messages[index])
                                   for index in range(len(pending.prefix)))
        except (IndexError, KeyError, TypeError, ValueError, OverflowError):
            return {**base, "status": "discarded", "reason": "source_prefix_changed"}
        if current_prefix != pending.prefix:
            return {**base, "status": "discarded", "reason": "source_prefix_changed"}
        return {**base, "status": "completed", "chunks": chunks}

    def close(self) -> None:
        """Drain the one CPU worker and discard any unconsumed result."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pending = None
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _compute(
        self, session_id: str, messages: tuple[Mapping[str, Any], ...],
        benchmark: str | None, encoding_scope: str, max_chunk_tokens: int,
        chunk_overlap: int, atomic_unit_token_limit: int, source_cutoff: int,
    ) -> tuple[tuple[EncoderChunk, ...], int, int, str | None]:
        started_perf_ns = time.perf_counter_ns()
        try:
            store = EventStore.from_messages(session_id, messages, benchmark=benchmark)
            chunks = plan_cross_turn_chunks(
                store, self._tokenizer, benchmark=benchmark,
                encoding_scope=encoding_scope, max_chunk_tokens=max_chunk_tokens,
                chunk_overlap=chunk_overlap,
                atomic_unit_token_limit=atomic_unit_token_limit,
                source_cutoff=source_cutoff,
            )
            error = None
        except Exception as caught:
            chunks = ()
            error = f"{type(caught).__name__}: {caught}"
        return chunks, started_perf_ns, time.perf_counter_ns(), error
