"""Small dependency-free JSONL telemetry primitives.

All timestamps used for cross-component joins are Unix nanoseconds.  Durations
are measured from ``perf_counter_ns`` in the process that owns the event.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional


SCHEMA = "c2kv.measurement.event.v1"
_locks: Dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def append_jsonl(path: "str | os.PathLike[str]", row: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    key = str(target.resolve())
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock, target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: "str | os.PathLike[str]") -> Iterator[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_number}")
            yield row


_episode = contextvars.ContextVar("c2kv_measurement_episode", default=None)
_last_decision = contextvars.ContextVar("c2kv_measurement_last_decision", default=None)


class HarnessTelemetry:
    """Emit benchmark episode, model-decision, and external-tool events.

    The context variables make the API safe for BFCL/AppWorld worker threads.
    A decision id is copied onto later tool events so offline aggregation can
    join each committed action to the complete proxy-side decision chain.
    """

    def __init__(self, path: "str | os.PathLike[str]", benchmark: str):
        self.path = str(path)
        self.benchmark = benchmark

    def _emit(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        row = {
            "schema": SCHEMA,
            "event_type": event_type,
            "benchmark": self.benchmark,
            "unix_ns": time.time_ns(),
            **fields,
        }
        append_jsonl(self.path, row)
        return row

    @contextmanager
    def episode(self, episode_id: str, metadata: Optional[Dict[str, Any]] = None):
        value = {
            "episode_id": str(episode_id),
            "episode_instance_id": uuid.uuid4().hex,
            "episode_metadata": metadata or {},
        }
        episode_token = _episode.set(value)
        decision_token = _last_decision.set(None)
        start_perf = time.perf_counter_ns()
        start_unix = time.time_ns()
        self._emit("episode_start", **value, start_unix_ns=start_unix)
        status = "ok"
        error = None
        try:
            yield value
        except BaseException as exc:
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            end_unix = time.time_ns()
            self._emit(
                "episode_end", **value, status=status, error=error,
                start_unix_ns=start_unix, end_unix_ns=end_unix,
                duration_ns=time.perf_counter_ns() - start_perf,
            )
            _last_decision.reset(decision_token)
            _episode.reset(episode_token)

    def record_decision(
        self,
        *,
        request_id: Optional[str],
        start_unix_ns: int,
        duration_ns: int,
        response: Any = None,
        error: Optional[str] = None,
    ) -> str:
        decision_id = request_id or f"client-{uuid.uuid4().hex}"
        _last_decision.set(decision_id)
        context = dict(_episode.get() or {})
        self._emit(
            "decision", **context, decision_request_id=decision_id,
            start_unix_ns=start_unix_ns,
            end_unix_ns=start_unix_ns + duration_ns,
            duration_ns=duration_ns, response=response, error=error,
        )
        return decision_id

    def record_action(
        self,
        *,
        action: Any,
        outcome: Any,
        start_unix_ns: int,
        duration_ns: int,
        action_index: Optional[int] = None,
        status: str = "ok",
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = dict(_episode.get() or {})
        return self._emit(
            "tool_action", **context,
            decision_request_id=_last_decision.get(),
            action_index=action_index, action=action, outcome=outcome,
            start_unix_ns=start_unix_ns,
            end_unix_ns=start_unix_ns + duration_ns,
            duration_ns=duration_ns, status=status, error=error,
            metadata=metadata or {},
        )


def current_episode() -> Optional[Dict[str, Any]]:
    value = _episode.get()
    return dict(value) if value else None


def last_decision_id() -> Optional[str]:
    return _last_decision.get()
