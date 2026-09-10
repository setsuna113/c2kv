"""Request-scoped, content-free telemetry for client extraction lookups."""

from __future__ import annotations

import argparse
import copy
import math
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_CURRENT_TRACE: ContextVar[ExtractionTrace | None] = ContextVar(
    "memory_runtime_extraction_trace", default=None)
_CURRENT_SOURCES: ContextVar[tuple[int, ...]] = ContextVar(
    "memory_runtime_extraction_sources", default=())


def positive_extraction_limit(value: str) -> int:
    """Parse a strictly positive extraction-attempt limit for argparse."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


class ExtractionBudgetExceeded(RuntimeError):
    """Raised before the producer runs when the process cap is exhausted."""

    kind = "extraction_budget_exhausted"

    def __init__(self, limit: int):
        super().__init__(f"extraction attempt budget exhausted ({limit}/{limit})")
        self.limit = limit


class RequestExtractionBudget:
    """The extraction indices owned by one proxy request."""

    def __init__(self, budget: "ExtractionBudget", consumed_before: int) -> None:
        self.budget = budget
        self.consumed_before = consumed_before
        self.attempt_indices: list[int] = []

    def metadata(self) -> dict[str, Any] | None:
        if not self.budget.enabled:
            return None
        return {
            "limit": self.budget.limit,
            "consumed_before": self.consumed_before,
            "consumed_after": self.budget.consumed,
            "attempt_indices": list(self.attempt_indices),
        }


class ExtractionBudget:
    """Process-wide atomic reservations made immediately before producers."""

    def __init__(self, limit: int | None) -> None:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0
        ):
            raise ValueError("extraction attempt limit must be a positive integer")
        self.limit = limit
        self._consumed = 0
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def enabled(self) -> bool:
        return self.limit is not None

    @property
    def consumed(self) -> int:
        with self._lock:
            return self._consumed

    def reserve(self) -> int:
        with self._lock:
            if self.limit is not None and self._consumed >= self.limit:
                raise ExtractionBudgetExceeded(self.limit)
            self._consumed += 1
            index = self._consumed
        scope = self.current_request()
        if scope is not None:
            scope.attempt_indices.append(index)
        return index

    def current_request(self) -> RequestExtractionBudget | None:
        stack = getattr(self._local, "request_stack", ())
        return stack[-1] if stack else None

    @contextmanager
    def request_scope(self):
        record = RequestExtractionBudget(self, self.consumed)
        stack = getattr(self._local, "request_stack", None)
        if stack is None:
            stack = []
            self._local.request_stack = stack
        stack.append(record)
        try:
            yield record
        finally:
            popped = stack.pop()
            assert popped is record


def _seconds(value: Any, label: str) -> float:
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{label} must be finite and nonnegative")
    return float(value)


def _result_fields(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {"key_hash": None, "original_seq_len": None, "gist_len": None}
    if not isinstance(result, Mapping):
        raise TypeError("extraction result must be a mapping or None")
    return {key: copy.deepcopy(result.get(key))
            for key in ("key_hash", "original_seq_len", "gist_len")}


class ExtractionTrace:
    """Mutable request-local recorder whose snapshots are detached values."""

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record_lookup(
        self,
        cache_key,
        *,
        cache_hit: bool,
        force: bool,
        lookup_wall_sec: float,
        producer_wall_sec: float,
        result: Mapping[str, Any] | None = None,
        error_type: str | None = None,
        producer_called: bool | None = None,
        budget_attempt_index: int | None = None,
    ) -> None:
        if not isinstance(cache_key, tuple) or len(cache_key) != 4:
            raise ValueError("cache_key must be (role, content_hash, ratio, tools_hash)")
        if type(cache_hit) is not bool or type(force) is not bool:
            raise TypeError("cache_hit and force must be booleans")
        if error_type is not None and not isinstance(error_type, str):
            raise TypeError("error_type must be a string or None")
        if producer_called is None:
            producer_called = not cache_hit
        if type(producer_called) is not bool or (cache_hit and producer_called):
            raise ValueError("cache hits cannot call the extraction producer")
        if budget_attempt_index is not None and (
            type(budget_attempt_index) is not int or budget_attempt_index <= 0
        ):
            raise ValueError("budget_attempt_index must be a positive integer or None")
        event = {
            "cache_key": {
                "role": copy.deepcopy(cache_key[0]),
                "content_hash": copy.deepcopy(cache_key[1]),
                "ratio": copy.deepcopy(cache_key[2]),
                "tools_hash": copy.deepcopy(cache_key[3]),
            },
            "client_cache_hit": cache_hit,
            "force": force,
            "producer_called": producer_called,
            "producer_succeeded": producer_called and error_type is None,
            "producer_failed": producer_called and error_type is not None,
            "producer_blocked": not cache_hit and not producer_called,
            "budget_attempt_index": budget_attempt_index,
            "lookup_wall_sec": _seconds(lookup_wall_sec, "lookup_wall_sec"),
            "producer_wall_sec": _seconds(producer_wall_sec, "producer_wall_sec"),
            "error_type": error_type,
            "source_indices": list(_CURRENT_SOURCES.get()),
            **_result_fields(result),
            "server_cache_hit": None,
            "actual_extraction_prefill_tokens": None,
        }
        with self._lock:
            event["lookup_index"] = len(self._events)
            self._events.append(event)

    def snapshot(
        self,
        *,
        block_refs: Sequence[Mapping[str, Any]],
        forwarded_requests: Sequence[Sequence[str]],
        request_status: str,
    ) -> dict[str, Any]:
        with self._lock:
            events = copy.deepcopy(self._events)
        blocks = [{key: copy.deepcopy(block.get(key))
                   for key in ("key_hash", "source_indices", "gist_tokens")}
                  for block in block_refs]
        forwarded = [list(keys) for keys in forwarded_requests]

        links: dict[Any, dict[str, Any]] = {}

        def link(key_hash):
            return links.setdefault(key_hash, {
                "key_hash": key_hash,
                "lookup_event_indices": [],
                "retained_block_indices": [],
                "forwarded_occurrences": [],
            })

        for index, event in enumerate(events):
            if event["key_hash"] is not None:
                link(event["key_hash"])["lookup_event_indices"].append(index)
        for index, block in enumerate(blocks):
            link(block.get("key_hash"))["retained_block_indices"].append(index)
        for request_index, keys in enumerate(forwarded):
            for message_index, key_hash in enumerate(keys):
                link(key_hash)["forwarded_occurrences"].append({
                    "request_index": request_index,
                    "gist_message_index": message_index,
                })
        producer_events = [event for event in events if event["producer_called"]]
        producer_successes = [
            event for event in producer_events if event["producer_succeeded"]]
        summary = {
            "lookups": len(events),
            "client_cache_hits": sum(event["client_cache_hit"] for event in events),
            "producer_calls": len(producer_events),
            "producer_successes": len(producer_successes),
            "producer_failures": sum(event["producer_failed"] for event in events),
            "producer_blocked": sum(event["producer_blocked"] for event in events),
            "lookup_wall_sec": sum(event["lookup_wall_sec"] for event in events),
            "producer_wall_sec": sum(event["producer_wall_sec"] for event in events),
            "producer_response_original_seq_len_sum": sum(
                event["original_seq_len"] for event in producer_successes
                if type(event["original_seq_len"]) is int),
            "retained_blocks": len(blocks),
            "forwarded_requests": len(forwarded),
            "forwarded_key_occurrences": sum(len(keys) for keys in forwarded),
            "server_cache_hit": None,
            "actual_extraction_prefill_tokens": None,
        }
        return copy.deepcopy({
            "schema": "a-runtime-extraction-telemetry-v1",
            "request_status": request_status,
            "events": events,
            "block_refs": blocks,
            "forwarded_requests": forwarded,
            "key_links": list(links.values()),
            "summary": summary,
            "producer_calls_scope": (
                "client producer invocations, not general HTTP transport attempts"),
            "source_indices_scope": (
                "native source-message group provenance, not exact token spans"),
            "producer_response_original_seq_len_sum_scope": (
                "producer response metadata, not actual extraction prefill tokens"),
            "unsupported": {
                "server_cache_hit": "current backend extract response has no explicit field",
                "actual_extraction_prefill_tokens": (
                    "current backend extract response has no explicit field"),
            },
        })


def current_extraction_trace() -> ExtractionTrace | None:
    return _CURRENT_TRACE.get()


@contextmanager
def capture_extractions(enabled: bool = True):
    trace = ExtractionTrace() if enabled else None
    token = _CURRENT_TRACE.set(trace)
    try:
        yield trace
    finally:
        _CURRENT_TRACE.reset(token)


@contextmanager
def extraction_sources(source_indices: Iterable[int]):
    if current_extraction_trace() is None:
        yield
        return
    indices = tuple(source_indices)
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("source_indices must contain nonnegative integers")
    token = _CURRENT_SOURCES.set(indices)
    try:
        yield
    finally:
        _CURRENT_SOURCES.reset(token)


def validate_extraction_budget_rows(rows: Sequence[Mapping[str, Any]], limit: int) -> int:
    """Validate one sequential proxy's request and producer ledgers."""
    consumed = 0
    for request_index, row in enumerate(rows, 1):
        budget = row.get("extraction_budget")
        if not isinstance(budget, Mapping) or budget.get("limit") != limit:
            raise ValueError(
                f"Request {request_index} lacks its frozen process extraction cap")
        indices = budget.get("attempt_indices")
        if not isinstance(indices, list) or any(type(value) is not int for value in indices):
            raise ValueError("Invalid extraction attempt indices")
        after = consumed + len(indices)
        if (budget.get("consumed_before") != consumed
                or budget.get("consumed_after") != after
                or indices != list(range(consumed + 1, after + 1))
                or after > limit):
            raise ValueError("Extraction cap ledger is not sequential or exceeds its cap")
        telemetry = row.get("extraction_telemetry")
        if not isinstance(telemetry, Mapping):
            raise ValueError(f"Request {request_index} lacks extraction telemetry")
        events = telemetry.get("events")
        summary = telemetry.get("summary")
        if not isinstance(events, list) or not isinstance(summary, Mapping):
            raise ValueError("Extraction telemetry lacks events or summary")
        producer_events = [event for event in events
                           if isinstance(event, Mapping) and event.get("producer_called") is True]
        event_indices = [event.get("budget_attempt_index") for event in producer_events]
        if (summary.get("producer_calls") != len(producer_events)
                or len(indices) != summary.get("producer_calls")
                or event_indices != indices):
            raise ValueError("Extraction telemetry disagrees with the cap ledger")
        if any(event.get("budget_attempt_index") is not None
               for event in events if isinstance(event, Mapping)
               and event.get("client_cache_hit") is True):
            raise ValueError("Extraction cache hits must not consume the cap")
        consumed = after
    return consumed
