"""Durable allowlisted model traces for text-policy arms."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import threading
import time


SCHEMA = "a-textarm-model-trace-v1"
PHASES = frozenset({"compressor", "trajectory_retrieval_policy", "policy"})
STATUSES = frozenset({"completed", "failed"})
USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
_PHASE = ContextVar("textarm_model_phase", default="policy")


def textarm_trace_path(request_log):
    path = Path(request_log)
    if not str(request_log) or not path.name:
        raise ValueError("request_log must include a file name")
    return path.with_name("textarm_trace_" + path.name)


def current_textarm_phase():
    return _PHASE.get()


@contextmanager
def textarm_phase(phase):
    if phase not in PHASES:
        raise ValueError(f"textarm phase must be one of {sorted(PHASES)}")
    token = _PHASE.set(phase)
    try:
        yield
    finally:
        _PHASE.reset(token)


def _json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False))


def _request_view(value):
    if not isinstance(value, Mapping):
        raise TypeError("request_view must be a mapping")
    if not set(value) <= {"model", "messages", "tools", "sampling"}:
        raise ValueError("request_view contains fields outside the allowlist")
    if "messages" in value and not isinstance(value["messages"], list):
        raise TypeError("request_view.messages must be a list")
    if "tools" in value and not isinstance(value["tools"], list):
        raise TypeError("request_view.tools must be a list")
    if not isinstance(value.get("sampling", {}), Mapping):
        raise TypeError("request_view.sampling must be a mapping")
    return _json_copy(value)


def _usage(value):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("usage must be a mapping or None")
    result = {}
    for field in USAGE_FIELDS:
        item = value.get(field)
        if item is None:
            continue
        if type(item) not in (int, float) or item < 0:
            raise ValueError(f"usage.{field} must be nonnegative numeric data")
        result[field] = item
    return result


def _response_view(value):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("response_view must be a mapping or None")
    return _json_copy({field: value.get(field) for field in
                       ("content", "tool_calls", "finish_reason")})


def _cost(value):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("cost must be a mapping or None")
    return _json_copy(value)


@dataclass(frozen=True)
class TextarmTraceHandle:
    attempt_uid: str
    attempt_index: int
    request_id: str
    eval_context: dict
    phase: str


class TextarmTrace:
    """Append one fsynced start and terminal record per text-arm chat call."""

    def __init__(self, path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _append(self, record):
        encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                   + "\n").encode("utf-8")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
            descriptor = os.open(self.path, flags, 0o600)
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("textarm trace append made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def start(self, attempt, phase, request_view):
        if attempt is None or not isinstance(getattr(attempt, "attempt_uid", None), str):
            raise TypeError("textarm trace requires an AttemptJournal handle")
        if phase not in PHASES:
            raise ValueError(f"textarm phase must be one of {sorted(PHASES)}")
        handle = TextarmTraceHandle(
            attempt_uid=attempt.attempt_uid,
            attempt_index=attempt.attempt_index,
            request_id=attempt.request_id,
            eval_context=_json_copy(attempt.eval_context),
            phase=phase,
        )
        self._append({
            "schema": SCHEMA,
            "event": "started",
            "status": "started",
            "attempt_uid": handle.attempt_uid,
            "attempt_index": handle.attempt_index,
            "request_id": handle.request_id,
            "eval_context": handle.eval_context,
            "phase": handle.phase,
            "request_view": _request_view(request_view),
            "timestamp_unix_ns": time.time_ns(),
            "scope": "fsynced immediately before the HTTP transport call",
        })
        return handle

    def finish(self, handle, status, *, response_view=None, usage=None, cost=None,
               wall_sec, failure_stage=None, error_type=None, error_kind=None,
               error_detail=None, http_status=None):
        if not isinstance(handle, TextarmTraceHandle):
            raise TypeError("handle must be a TextarmTraceHandle")
        if status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}")
        if type(wall_sec) not in (int, float) or not math.isfinite(wall_sec) \
                or wall_sec < 0:
            raise ValueError("wall_sec must be finite and nonnegative")
        if status == "completed" and any(value is not None for value in
                                          (failure_stage, error_type, error_kind, error_detail)):
            raise ValueError("completed trace cannot carry failure fields")
        record = {
            "schema": SCHEMA,
            "event": "finished",
            "status": status,
            "attempt_uid": handle.attempt_uid,
            "attempt_index": handle.attempt_index,
            "request_id": handle.request_id,
            "eval_context": handle.eval_context,
            "phase": handle.phase,
            "response_view": _response_view(response_view),
            "usage": _usage(usage),
            "cost": _cost(cost),
            "wall_sec": float(wall_sec),
            "failure_stage": failure_stage,
            "error_type": error_type,
            "error_kind": error_kind,
            "error_detail": error_detail,
            "http_status": http_status,
            "timestamp_unix_ns": time.time_ns(),
            "usage_scope": "backend-normalized when available; null means unknown",
            "status_scope": (
                "completed means HTTP JSON normalized by the backend; failed records "
                "transport, JSON, or backend-normalization failure"
            ),
        }
        self._append(record)


def read_textarm_trace(path):
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("textarm trace has a partial trailing record")
    records = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"textarm trace line {line_number} is invalid JSON") from error
        if not isinstance(record, dict) or record.get("schema") != SCHEMA:
            raise ValueError(f"textarm trace line {line_number} has invalid schema")
        records.append(record)
    return records
