"""Raw-ID-only SGLang generation for the ACEBench Full-original control.

This bridge deliberately owns no server-side cache.  It only POSTs one
pre-tokenized, contiguous Full-original prefix to SGLang's ``/generate``
endpoint and records the response that endpoint returned.  It neither sends
C2KV request fields nor asks SGLang to flush, release, or retain a session.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener


RAW_SGLANG_SCHEMA = "a-acebench-full-raw-sglang-v1"
RAW_HTTP_JOURNAL_SCHEMA = "a-acebench-full-raw-sglang-http-v1"
FULL_ORIGINAL_LAYOUT = "full-original-native-v1"
EXTERNAL_CACHE_POLICY = "external-sglang-cache-unknown-v1"


@dataclass(frozen=True)
class FullRawGenerationResult:
    """The subset of ``EventNativeGenerationResult`` used by finite runners.

    This local data class keeps the raw bridge independent of ``torch`` and
    the in-process C2KV runtime.  It intentionally has the same observable
    fields consumed by ``EventNativeDecisionRunner``.
    """

    token_ids: tuple[int, ...]
    finish_reason: str
    token_logprobs: tuple[float, ...]
    stats: dict[str, Any]


class FullRawSGLangError(RuntimeError):
    """A terminal one-request transport or response-contract failure."""


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _json_snapshot(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a JSON object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be JSON-native and finite") from error
    if not isinstance(decoded, dict):  # Kept defensive despite the Mapping check.
        raise TypeError(f"{name} must be a JSON object")
    return decoded


def _generate_url(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("upstream must be a nonempty http://... SGLang endpoint")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "http"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") not in {"", "/generate"}
    ):
        raise ValueError("upstream must be an http://... base or /generate URL without credentials or query")
    return (value.rstrip("/") if parsed.path.rstrip("/") == "/generate"
            else value.rstrip("/") + "/generate")


def _token_ids(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be a sequence of token IDs")
    result = tuple(value)
    if any(type(token) is not int or token < 0 for token in result):
        raise ValueError(f"{name} must contain nonnegative integer token IDs")
    return result


def _eos_ids(value: Any) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise TypeError("eos_token_ids must be an integer or a sequence of integers")
    if isinstance(value, int):
        values = (value,)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        values = tuple(value)
    else:
        raise TypeError("eos_token_ids must be an integer or a sequence of integers")
    if not values or any(type(token) is not int or token < 0 for token in values):
        raise ValueError("eos_token_ids must contain nonnegative integer token IDs")
    return tuple(sorted(set(values)))


def _finish_reason(value: Any) -> tuple[str, Any]:
    if isinstance(value, str) and value:
        return value, value
    if isinstance(value, Mapping) and isinstance(value.get("type"), str) and value["type"]:
        return value["type"], copy.deepcopy(dict(value))
    raise FullRawSGLangError("SGLang response meta_info.finish_reason must be a nonempty string or type object")


class _RawHTTPJournal:
    """Append raw request/response pairs with a flush and fsync per record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.name:
            raise ValueError("journal_path must include a file name")
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        try:
            encoded = (
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ValueError("raw HTTP journal record is not JSON-native") from error
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
            descriptor = os.open(self.path, flags, 0o600)
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("raw HTTP journal append made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


class SGLangFullRawGenerator:
    """A finite SGLang ``/generate`` adapter for full-original raw prefixes."""

    session_cache_policy = EXTERNAL_CACHE_POLICY

    def __init__(
        self,
        upstream: str,
        *,
        model_context: int,
        max_new_tokens: int,
        max_generation_calls: int,
        timeout_seconds: float,
        eos_token_ids: int | Sequence[int],
        eos_source: str,
        journal_path: str | Path | None = None,
        sampling_params: Mapping[str, Any] | None = None,
        max_response_bytes: int = 4 * 1024 * 1024,
        opener: Any | None = None,
    ) -> None:
        self.generate_url = _generate_url(upstream)
        self.model_context = _positive_int(model_context, "model_context")
        self.sampling_params = _json_snapshot(
            {"temperature": 0, "top_p": 1, "sampling_seed": 0}
            if sampling_params is None
            else sampling_params,
            "sampling_params",
        )
        if "max_new_tokens" in self.sampling_params:
            raise ValueError("sampling_params must omit max_new_tokens; generate supplies the finite cap")
        self.max_new_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        self.max_generation_calls = _positive_int(max_generation_calls, "max_generation_calls")
        self.timeout_seconds = _positive_finite(timeout_seconds, "timeout_seconds")
        self.eos_token_ids = _eos_ids(eos_token_ids)
        if not isinstance(eos_source, str) or not eos_source:
            raise ValueError("eos_source must be a nonempty string")
        self.eos_source = eos_source
        self.max_response_bytes = _positive_int(max_response_bytes, "max_response_bytes")
        self._http_journal = (
            _RawHTTPJournal(journal_path) if journal_path is not None else None
        )
        self._opener = opener or build_opener(ProxyHandler({}))
        if not callable(getattr(self._opener, "open", None)):
            raise TypeError("opener must expose open(request, timeout=...)")
        self._requests_submitted = 0
        self._active_session_id: str | None = None

    @contextmanager
    def decision_scope(self, *, session_id: str | None = None) -> Iterator[None]:
        """Give the finite runner a scope without creating an SGLang session."""

        if self._active_session_id is not None:
            raise RuntimeError("decision_scope is not reentrant")
        if session_id is not None and (not isinstance(session_id, str) or not session_id):
            raise ValueError("session_id must be None or a nonempty string")
        self._active_session_id = session_id or "<unscoped>"
        try:
            yield
        finally:
            self._active_session_id = None

    def close_session(self) -> None:
        """Do not call a remote flush/release endpoint for an unowned cache."""

        if self._active_session_id is not None:
            raise RuntimeError("cannot close a session inside its active decision_scope")

    def session_cache_info(self) -> dict[str, Any]:
        """Report only this client's lack of ownership, not cache-state guesses."""

        return {
            "policy": self.session_cache_policy,
            "external_sglang_cache_state": "unknown",
            "client_owns_external_cache": False,
            "flush_or_release_attempted": False,
            "active_decision_scope": self._active_session_id is not None,
            "requests_submitted": self._requests_submitted,
            "maximum_generation_calls": self.max_generation_calls,
            "scope": (
                "SGLang may share cache state outside this client; the raw bridge "
                "does not inspect, reserve, release, or flush it."
            ),
        }

    def generate(
        self,
        memory: Any,
        *,
        ratio: int,
        max_new_tokens: int,
        trace_context: Mapping[str, Any] | None = None,
    ) -> FullRawGenerationResult:
        """Submit exactly one validated Full-original prefix with no retry."""

        _positive_int(ratio, "ratio")
        requested_tokens = _positive_int(max_new_tokens, "max_new_tokens")
        if requested_tokens > self.max_new_tokens:
            raise ValueError("max_new_tokens exceeds this raw bridge's finite cap")
        if trace_context is not None and not isinstance(trace_context, Mapping):
            raise TypeError("trace_context must be a mapping or None")
        input_ids = self._full_original_ids(memory)
        if len(input_ids) + requested_tokens > self.model_context:
            raise ValueError("Full-original raw prompt plus completion exceeds this bridge's model context")
        if self._requests_submitted >= self.max_generation_calls:
            raise RuntimeError("raw SGLang request cap exhausted; automatic retry is disabled")

        payload = {
            "input_ids": list(input_ids),
            "sampling_params": {
                **copy.deepcopy(self.sampling_params),
                "max_new_tokens": requested_tokens,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
            "top_logprobs_num": 0,
            "return_text_in_logprobs": False,
            "stream": False,
        }
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        request_index = self._requests_submitted + 1
        started = time.perf_counter()
        started_ns = time.time_ns()
        raw_response: Any = None
        http_status: int | None = None
        if self._http_journal is not None:
            self._http_journal.append(
                {
                    "schema": RAW_HTTP_JOURNAL_SCHEMA,
                    "event": "request",
                    "status": "started",
                    "request_index": request_index,
                    "timestamp_unix_ns": started_ns,
                    "url": self.generate_url,
                    "timeout_seconds": self.timeout_seconds,
                    "retries": 0,
                    "request": copy.deepcopy(payload),
                }
            )
        self._requests_submitted += 1
        try:
            request = Request(
                self.generate_url,
                data=body,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                status_value = getattr(response, "status", None)
                http_status = int(response.getcode() if status_value is None else status_value)
                raw = response.read(self.max_response_bytes + 1)
            if len(raw) > self.max_response_bytes:
                raise FullRawSGLangError("SGLang response exceeds this bridge's finite byte cap")
            try:
                raw_response = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise FullRawSGLangError("SGLang response is not valid UTF-8 JSON") from error
            result = self._result_from_response(
                raw_response,
                input_ids=input_ids,
                http_status=http_status,
                request_index=request_index,
                wall_seconds=time.perf_counter() - started,
            )
        except HTTPError as error:
            http_status = int(error.code)
            try:
                raw = error.read(self.max_response_bytes + 1)
                if len(raw) <= self.max_response_bytes:
                    raw_response = json.loads(raw.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                raw_response = None
            raise FullRawSGLangError(f"SGLang /generate returned HTTP {http_status}") from error
        except FullRawSGLangError:
            raise
        except (OSError, TimeoutError, ValueError) as error:
            raise FullRawSGLangError("SGLang /generate transport failed") from error
        finally:
            if self._http_journal is not None:
                record = {
                    "schema": RAW_HTTP_JOURNAL_SCHEMA,
                    "event": "response",
                    "status": "completed" if "result" in locals() else "failed",
                    "request_index": request_index,
                    "timestamp_unix_ns": time.time_ns(),
                    "http_status": http_status,
                    "retries": 0,
                    "wall_seconds": time.perf_counter() - started,
                    "response": raw_response,
                }
                if "result" not in locals():
                    record["error_type"] = "FullRawSGLangError"
                self._http_journal.append(record)
        return result

    def _full_original_ids(self, memory: Any) -> tuple[int, ...]:
        view = getattr(memory, "view", None)
        if getattr(view, "raw_control_layout", None) != FULL_ORIGINAL_LAYOUT:
            raise ValueError("raw SGLang bridge accepts only full-original-native-v1 memory")
        for name in ("gist_event_ids", "evidence_event_ids", "omitted_event_ids"):
            value = getattr(view, name, None)
            if tuple(value or ()):
                raise ValueError(f"Full-original raw bridge rejects {name}")
        raw_event_ids = tuple(getattr(view, "raw_event_ids", ()))
        mandatory_raw_event_ids = tuple(getattr(view, "mandatory_raw_event_ids", ()))
        if not raw_event_ids or mandatory_raw_event_ids != raw_event_ids:
            raise ValueError("Full-original raw bridge requires every visible event as mandatory raw")
        chunks = tuple(getattr(memory, "chunks", ()))
        if chunks:
            raise ValueError("Full-original raw bridge rejects encoded chunks or gist state")
        raw_source_indices = _token_ids(
            getattr(memory, "raw_source_indices", None), "memory.raw_source_indices"
        )
        if raw_source_indices != tuple(range(len(raw_source_indices))):
            raise ValueError("Full-original raw bridge rejects noncontiguous source indices")
        system_ids = _token_ids(
            getattr(memory, "system_input_ids", None), "memory.system_input_ids"
        )
        workspace_ids = _token_ids(
            getattr(memory, "workspace_input_ids", None), "memory.workspace_input_ids"
        )
        input_ids = system_ids + workspace_ids
        if not input_ids:
            raise ValueError("Full-original raw bridge requires a nonempty raw prefix")
        return input_ids

    def _result_from_response(
        self,
        response: Any,
        *,
        input_ids: tuple[int, ...],
        http_status: int,
        request_index: int,
        wall_seconds: float,
    ) -> FullRawGenerationResult:
        if not isinstance(response, Mapping):
            raise FullRawSGLangError("SGLang response must be a JSON object")
        if not isinstance(response.get("text"), str):
            raise FullRawSGLangError("SGLang response lacks text")
        output_ids = _token_ids(response.get("output_ids"), "SGLang response output_ids")
        meta = response.get("meta_info")
        if not isinstance(meta, Mapping):
            raise FullRawSGLangError("SGLang response lacks meta_info")
        prompt_tokens = meta.get("prompt_tokens")
        completion_tokens = meta.get("completion_tokens")
        if type(prompt_tokens) is not int or prompt_tokens != len(input_ids):
            raise FullRawSGLangError("SGLang reported prompt_tokens disagree with submitted raw IDs")
        if type(completion_tokens) is not int or completion_tokens != len(output_ids):
            raise FullRawSGLangError("SGLang reported completion_tokens disagree with output_ids")
        logprob_rows = meta.get("output_token_logprobs")
        if not isinstance(logprob_rows, Sequence) or isinstance(logprob_rows, (str, bytes, bytearray)):
            raise FullRawSGLangError("SGLang response lacks output_token_logprobs")
        if meta.get("output_token_logprobs_length") != len(output_ids) or len(logprob_rows) != len(output_ids):
            raise FullRawSGLangError("SGLang output_token_logprobs count disagrees with output_ids")
        logprobs: list[float] = []
        for index, (token_id, row) in enumerate(zip(output_ids, logprob_rows, strict=True)):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)) or len(row) != 3:
                raise FullRawSGLangError("SGLang output_token_logprobs has an invalid row")
            value, observed_token_id, _text = row
            if type(observed_token_id) is not int or observed_token_id != token_id:
                raise FullRawSGLangError(
                    f"SGLang output_token_logprobs token ID disagrees at index {index}"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise FullRawSGLangError("SGLang output_token_logprobs contains a non-finite value")
            logprobs.append(float(value))
        finish_reason, raw_finish_reason = _finish_reason(meta.get("finish_reason"))
        stats = {
            "schema": RAW_SGLANG_SCHEMA,
            "transport": {
                "generate_url": self.generate_url,
                "http_status": http_status,
                "request_index": request_index,
                "wall_seconds": wall_seconds,
                "timeout_seconds": self.timeout_seconds,
                "retries": 0,
            },
            "input": {
                "prompt_tokens": len(input_ids),
                "input_ids_sha256": hashlib.sha256(
                    json.dumps(list(input_ids), separators=(",", ":")).encode("ascii")
                ).hexdigest(),
            },
            "output": {
                "completion_tokens": len(output_ids),
                "server_finish_reason": raw_finish_reason,
                "server_e2e_latency": meta.get("e2e_latency"),
            },
            "eos_token_ids": list(self.eos_token_ids),
            "eos_source": self.eos_source,
            "external_sglang_cache": {
                "state": "unknown",
                "client_owns": False,
                "flush_or_release_attempted": False,
            },
            "scope": (
                "One direct Full-original raw-ID SGLang request. No C2KV request field, "
                "gist/KV injection, cache acquisition, release, or flush was performed."
            ),
        }
        return FullRawGenerationResult(
            token_ids=output_ids,
            finish_reason=finish_reason,
            token_logprobs=tuple(logprobs),
            stats=stats,
        )


__all__ = [
    "EXTERNAL_CACHE_POLICY",
    "FULL_ORIGINAL_LAYOUT",
    "FullRawGenerationResult",
    "FullRawSGLangError",
    "SGLangFullRawGenerator",
    "RAW_HTTP_JOURNAL_SCHEMA",
    "RAW_SGLANG_SCHEMA",
]
