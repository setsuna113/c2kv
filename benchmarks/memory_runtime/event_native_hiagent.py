"""Server-side phase dispatcher for opt-in native HiAgent model calls.

The dispatcher bridges the shared native HiAgent wire envelope to two
existing ``full_original`` decision runners.  Actor and compressor generators
must be distinct objects backed by the same loaded runtime.  Calls are
serialized, use an independent session identity, and close their selected
generator cache before returning.
"""
from __future__ import annotations

import copy
import ipaddress
import json
import math
import os
import socket
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from benchmarks.native_hiagent_protocol import (
    ENDPOINT_PATH,
    RECEIPT_FIELD,
    RECEIPT_SCHEMA,
    validate_call_envelope,
    validate_policy_sampling,
)

from .attempt_journal import AttemptHandle, AttemptJournal
from .event_native_controls import build_event_native_controller
from .event_native_step import EventNativeDecisionRunner


SCHEMA = "a-event-native-hiagent-dispatch-v1"
COMPRESSOR_MAX_COMPLETION_TOKENS = 100
COMPRESSOR_STOP = ("\n\n",)
COMPRESSOR_SAMPLING = {
    "temperature": 0.0,
    "seed": 42,
    "max_completion_tokens": COMPRESSOR_MAX_COMPLETION_TOKENS,
    "stop": list(COMPRESSOR_STOP),
    "top_p": 1.0,
}
ACTOR_PHASES = frozenset({"policy", "trajectory_retrieval_policy"})
_ACTIVE_CALL: ContextVar[dict[str, Any] | None] = ContextVar(
    "event_native_hiagent_call", default=None
)


class EventNativeHiAgentError(ValueError):
    """One stable request or server failure returned by the HTTP adapter."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.error_type = error_type


class FirstSubstringStopMatcher:
    """Stop after the first decoded occurrence of any configured substring."""

    def __init__(self, tokenizer: Any, stop_strings: Sequence[str]) -> None:
        if not callable(getattr(tokenizer, "decode", None)):
            raise TypeError("tokenizer must expose decode")
        if (
            isinstance(stop_strings, (str, bytes, bytearray))
            or not isinstance(stop_strings, Sequence)
            or not stop_strings
            or any(not isinstance(value, str) or not value for value in stop_strings)
        ):
            raise ValueError("stop_strings must contain nonempty strings")
        self.tokenizer = tokenizer
        self.stop_strings = tuple(stop_strings)
        self.first_match: tuple[int, str] | None = None

    def __call__(self, token_ids: tuple[int, ...]) -> bool:
        text = self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        match = _first_stop(text, self.stop_strings)
        if match is None:
            return False
        self.first_match = match
        return True


class _PhaseBoundGenerator:
    """Bind per-call stop strings without changing the existing runner."""

    def __init__(self, generator: Any, tokenizer: Any) -> None:
        self._generator = generator
        self._tokenizer = tokenizer
        self._matcher: FirstSubstringStopMatcher | None = None
        self._bound = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._generator, name)

    @property
    def underlying_generator(self) -> Any:
        return self._generator

    @property
    def matcher(self) -> FirstSubstringStopMatcher | None:
        return self._matcher

    @contextmanager
    def bind_stop_strings(self, stop_strings: Sequence[str]) -> Any:
        if self._bound:
            raise RuntimeError("phase stop binding is not reentrant")
        self._bound = True
        self._matcher = (
            FirstSubstringStopMatcher(self._tokenizer, stop_strings)
            if stop_strings
            else None
        )
        try:
            yield
        finally:
            self._bound = False

    def generate(self, memory: Any, **kwargs: Any) -> Any:
        if self._matcher is not None:
            if "token_prefix_stop" in kwargs:
                raise RuntimeError("runner supplied a second token-prefix stop callback")
            kwargs["token_prefix_stop"] = self._matcher
        return self._generator.generate(memory, **kwargs)


@dataclass(frozen=True)
class _JoinCall:
    generator_role: str
    parent_request_id: str
    official_eval_context: dict[str, Any]
    proxy_attempt_uid: str
    phase: str
    call_ordinal: int
    native_session_id: str
    native_decision_key: str

    def record(self) -> dict[str, Any]:
        return {
            "generator_role": self.generator_role,
            "parent_request_id": self.parent_request_id,
            "official_eval_context": copy.deepcopy(self.official_eval_context),
            "official_task_id": self.official_eval_context["task_id"],
            "proxy_attempt_uid": self.proxy_attempt_uid,
            "phase": self.phase,
            "call_ordinal": self.call_ordinal,
            "native_session_id": self.native_session_id,
            "native_decision_key": self.native_decision_key,
        }


class _JoinWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.name:
            raise ValueError("join_path must include a file name")
        self._lock = threading.Lock()

    def append(self, record: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0)
            descriptor = os.open(self.path, flags, 0o600)
            try:
                remaining = memoryview(encoded)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:
                        raise OSError("native HiAgent join append made no progress")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)


class _JoiningAttemptJournal(AttemptJournal):
    """Preserve AttemptJournal identity while fsyncing the cross-layer join."""

    def __init__(
        self,
        journal: AttemptJournal,
        join_writer: _JoinWriter,
        *,
        generator_role: str,
    ) -> None:
        if not isinstance(journal, AttemptJournal):
            raise TypeError("journal must be an AttemptJournal")
        if generator_role not in {"actor", "auxiliary"}:
            raise ValueError("generator_role must be actor or auxiliary")
        super().__init__(journal.path)
        self._journal = journal
        self._join_writer = join_writer
        self._generator_role = generator_role

    def start(
        self,
        kind: str,
        attempt_index: int,
        request_id: str,
        eval_context: Mapping[str, Any],
    ) -> AttemptHandle:
        call = _ACTIVE_CALL.get()
        if call is None or call["generator_role"] != self._generator_role:
            raise RuntimeError("native HiAgent journal start lacks its active call identity")
        handle = self._journal.start(kind, attempt_index, request_id, eval_context)
        self._join_writer.append({
            "schema": SCHEMA,
            "event": "started",
            "status": "started",
            **call,
            "native_attempt_uid": handle.attempt_uid,
            "native_attempt_index": handle.attempt_index,
            "timestamp_unix_ns": time.time_ns(),
            "scope": "fsynced after native attempt reservation and before generator submission",
        })
        return handle

    def finish(
        self,
        handle: AttemptHandle,
        status: str,
        *,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        call = _ACTIVE_CALL.get()
        if call is None or call["generator_role"] != self._generator_role:
            raise RuntimeError("native HiAgent journal finish lacks its active call identity")
        self._journal.finish(handle, status, usage=usage)
        self._join_writer.append({
            "schema": SCHEMA,
            "event": "finished",
            "status": status,
            **call,
            "native_attempt_uid": handle.attempt_uid,
            "native_attempt_index": handle.attempt_index,
            "usage": copy.deepcopy(dict(usage)) if usage is not None else None,
            "timestamp_unix_ns": time.time_ns(),
        })


class EventNativeHiAgentDispatcher:
    """Validate, serialize, dispatch, account, and return one phase call."""

    def __init__(
        self,
        actor_runner: EventNativeDecisionRunner,
        auxiliary_runner: EventNativeDecisionRunner,
        tokenizer: Any,
        *,
        model_name: str,
        benchmark: str,
        allowed_task_ids: Sequence[str],
        actor_phase_sampling: Mapping[str, Mapping[str, Any]],
        deadline_monotonic: float,
        join_path: str | Path,
        steps_path: str | Path,
    ) -> None:
        if not isinstance(actor_runner, EventNativeDecisionRunner):
            raise TypeError("actor_runner must be an EventNativeDecisionRunner")
        if not isinstance(auxiliary_runner, EventNativeDecisionRunner):
            raise TypeError("auxiliary_runner must be an EventNativeDecisionRunner")
        if actor_runner is auxiliary_runner:
            raise ValueError("actor and auxiliary runners must be distinct")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a nonempty string")
        if not isinstance(benchmark, str) or not benchmark:
            raise ValueError("benchmark must be a nonempty string")
        allowed = frozenset(allowed_task_ids)
        if not allowed or any(not isinstance(value, str) or not value for value in allowed):
            raise ValueError("allowed_task_ids must contain nonempty strings")
        if (
            isinstance(deadline_monotonic, bool)
            or not isinstance(deadline_monotonic, (int, float))
            or not math.isfinite(float(deadline_monotonic))
        ):
            raise ValueError("deadline_monotonic must be finite")
        if not isinstance(actor_phase_sampling, Mapping) or set(actor_phase_sampling) != ACTOR_PHASES:
            raise ValueError("actor_phase_sampling must explicitly configure both actor phases")
        sampling = {
            phase: validate_policy_sampling(actor_phase_sampling[phase])
            for phase in sorted(ACTOR_PHASES)
        }
        for phase, config in sampling.items():
            validate_native_greedy_sampling(config, phase=phase)
        actor_caps = {config["max_completion_tokens"] for config in sampling.values()}
        if len(actor_caps) != 1 or actor_caps != {actor_runner.max_new_tokens}:
            raise ValueError("both actor phase caps must match the actor runner cap")
        if auxiliary_runner.max_new_tokens != COMPRESSOR_MAX_COMPLETION_TOKENS:
            raise ValueError("auxiliary runner must use the 100-token compressor cap")

        self.actor_runner = actor_runner
        self.auxiliary_runner = auxiliary_runner
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.benchmark = benchmark
        self.allowed_task_ids = allowed
        self.actor_phase_sampling = sampling
        self.deadline_monotonic = float(deadline_monotonic)
        self.join_path = Path(join_path)
        self.steps_path = Path(steps_path)
        if not self.steps_path.name:
            raise ValueError("steps_path must include a file name")
        if self.steps_path.resolve() == self.join_path.resolve():
            raise ValueError("steps_path and join_path must be distinct")
        self._steps_writer = _JoinWriter(self.steps_path)
        self._lock = threading.Lock()
        self._completed: dict[tuple[str, str, int], tuple[str, dict[str, Any]]] = {}
        self._next_ordinal: dict[tuple[str, str], int] = {}
        self._task_failures: dict[tuple[str, str], EventNativeHiAgentError] = {}
        self._terminal_failure: dict[str, str] | None = None

    def handle(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        try:
            envelope = validate_call_envelope(payload)
        except (TypeError, ValueError) as error:
            raise EventNativeHiAgentError(400, "invalid_envelope", str(error)) from error
        context = envelope["official_eval_context"]
        if context["benchmark"] != self.benchmark:
            raise EventNativeHiAgentError(400, "benchmark_mismatch", "benchmark is not served here")
        if context["task_id"] not in self.allowed_task_ids:
            raise EventNativeHiAgentError(403, "unknown_task", "official task is not allowed")
        model_call = envelope["model_call"]
        if model_call["model"] != self.model_name:
            raise EventNativeHiAgentError(400, "model_mismatch", "model is not served here")
        signature = json.dumps(
            envelope,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        identity = (
            envelope["parent_request_id"],
            envelope["proxy_attempt_uid"],
            envelope["call_ordinal"],
        )
        task_key = (context["benchmark"], context["task_id"])

        with self._lock:
            if self._terminal_failure is not None:
                raise EventNativeHiAgentError(
                    503,
                    "terminal_failure",
                    "native HiAgent dispatcher stopped after a terminal call failure",
                )
            cached = self._completed.get(identity)
            if cached is not None:
                if cached[0] != signature:
                    raise EventNativeHiAgentError(
                        409,
                        "call_conflict",
                        "phase call identity was reused with different input",
                    )
                return copy.deepcopy(cached[1])
            prior_task_failure = self._task_failures.get(task_key)
            if prior_task_failure is not None:
                raise EventNativeHiAgentError(
                    prior_task_failure.status_code,
                    prior_task_failure.code,
                    str(prior_task_failure),
                    error_type=prior_task_failure.error_type,
                )
            if time.monotonic() >= self.deadline_monotonic:
                raise EventNativeHiAgentError(408, "deadline_exceeded", "stage deadline expired")
            expected_ordinal = self._next_ordinal.get(task_key, 1)
            if envelope["call_ordinal"] != expected_ordinal:
                raise EventNativeHiAgentError(
                    409,
                    "call_ordinal_mismatch",
                    f"expected call_ordinal {expected_ordinal}",
                )

            phase = envelope["phase"]
            runner = self.auxiliary_runner if phase == "compressor" else self.actor_runner
            stop_strings = self._validate_sampling(phase, model_call["sampling"], runner)
            role = "auxiliary" if phase == "compressor" else "actor"
            join = _JoinCall(
                generator_role=role,
                parent_request_id=envelope["parent_request_id"],
                official_eval_context=context,
                proxy_attempt_uid=envelope["proxy_attempt_uid"],
                phase=phase,
                call_ordinal=envelope["call_ordinal"],
                native_session_id=envelope["native_session_id"],
                native_decision_key=envelope["native_decision_key"],
            ).record()
            runner_payload = {
                "session_id": envelope["native_session_id"],
                "decision_key": envelope["native_decision_key"],
                "messages": copy.deepcopy(model_call["messages"]),
                "tools": copy.deepcopy(model_call["tools"]),
            }
            generator = runner.generator
            if not isinstance(generator, _PhaseBoundGenerator):
                raise RuntimeError("native HiAgent runner lacks its phase-bound generator")
            context_token = _ACTIVE_CALL.set(join)
            record = None
            runner_error = None
            try:
                with generator.bind_stop_strings(stop_strings):
                    record = runner.run(runner_payload)
            except Exception as error:
                runner_error = error
                record = getattr(error, "record", None)
                if not isinstance(record, Mapping):
                    record = {
                        "schema": "a-event-native-hiagent-runner-failure-v1",
                        "status": "failed",
                        "session_id": envelope["native_session_id"],
                        "decision_key": envelope["native_decision_key"],
                        "generation_trace": [],
                        "response": None,
                        "error": {"type": type(error).__name__, "message": str(error)},
                    }
            finally:
                try:
                    runner.close()
                finally:
                    _ACTIVE_CALL.reset(context_token)
            persisted_record = copy.deepcopy(dict(record))
            persisted_record["session_cache_after_dispatch_close"] = copy.deepcopy(
                generator.session_cache_info()
            )
            try:
                self._append_step(envelope, persisted_record)
            except Exception as error:
                self._terminal_failure = {
                    "code": "steps_write_failed",
                    "type": type(error).__name__,
                }
                raise EventNativeHiAgentError(
                    500,
                    "steps_write_failed",
                    "native HiAgent runner record could not be persisted",
                ) from error
            if runner_error is not None:
                classified = _classify_runner_failure(record, runner_error)
                if classified.code == "capacity_rejected":
                    self._task_failures[task_key] = classified
                    # EventNativeDecisionRunner is terminal by default. A
                    # capacity rejection is task-local and has already closed
                    # this call's cache, so keep later official tasks runnable
                    # while forbidding another call for the rejected task.
                    runner._terminal_error = None
                else:
                    self._terminal_failure = {
                        "code": classified.code,
                        "type": classified.error_type or type(runner_error).__name__,
                    }
                raise classified from runner_error
            try:
                response = self._response(record, envelope, generator.matcher)
            except Exception as error:
                self._terminal_failure = {
                    "code": "invalid_runner_record",
                    "type": type(error).__name__,
                }
                raise EventNativeHiAgentError(
                    500,
                    "invalid_runner_record",
                    "native HiAgent runner returned an invalid record",
                ) from error
            self._next_ordinal[task_key] = expected_ordinal + 1
            self._completed[identity] = (signature, copy.deepcopy(response))
            return response

    def _append_step(
        self,
        envelope: Mapping[str, Any],
        runner_record: Mapping[str, Any],
    ) -> None:
        self._steps_writer.append({
            "schema": "a-event-native-hiagent-step-v1",
            "parent_request_id": envelope["parent_request_id"],
            "official_eval_context": copy.deepcopy(envelope["official_eval_context"]),
            "official_task_id": envelope["official_eval_context"]["task_id"],
            "proxy_attempt_uid": envelope["proxy_attempt_uid"],
            "phase": envelope["phase"],
            "generator_role": (
                "auxiliary" if envelope["phase"] == "compressor" else "actor"
            ),
            "call_ordinal": envelope["call_ordinal"],
            "native_session_id": envelope["native_session_id"],
            "native_decision_key": envelope["native_decision_key"],
            "runner_record": copy.deepcopy(dict(runner_record)),
        })

    def _validate_sampling(
        self,
        phase: str,
        value: Mapping[str, Any],
        runner: EventNativeDecisionRunner,
    ) -> tuple[str, ...]:
        sampling = copy.deepcopy(dict(value))
        if phase == "compressor":
            if sampling != COMPRESSOR_SAMPLING:
                raise EventNativeHiAgentError(
                    400,
                    "compressor_sampling_mismatch",
                    "compressor must preserve greedy0, top_p=1, seed=42, cap=100 and stop=['\\n\\n']",
                )
            return COMPRESSOR_STOP
        expected = self.actor_phase_sampling[phase]
        if sampling != expected:
            raise EventNativeHiAgentError(
                400,
                "actor_sampling_mismatch",
                "actor phase sampling differs from its explicit server configuration",
            )
        if sampling["max_completion_tokens"] != runner.max_new_tokens:
            raise RuntimeError("actor runner cap changed after dispatcher construction")
        return ()

    def _response(
        self,
        record: Mapping[str, Any],
        envelope: Mapping[str, Any],
        matcher: FirstSubstringStopMatcher | None,
    ) -> dict[str, Any]:
        if record.get("status") != "ok":
            raise RuntimeError("native HiAgent runner returned a non-success record")
        trace = record.get("generation_trace")
        if not isinstance(trace, list) or len(trace) != 1:
            raise RuntimeError("full_original native HiAgent call must use one generation")
        native_attempt_uid = trace[0].get("attempt_uid")
        if not isinstance(native_attempt_uid, str) or not native_attempt_uid:
            raise RuntimeError("native HiAgent runner record lacks its attempt UID")
        final = record.get("response")
        if not isinstance(final, Mapping):
            raise RuntimeError("native HiAgent runner record lacks its response")
        content = final.get("content")
        if content is not None and not isinstance(content, str):
            raise RuntimeError("native HiAgent response content must be text or null")
        tool_calls = final.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise RuntimeError("native HiAgent response tool_calls must be a list")
        finish_reason = "tool_calls" if tool_calls else final.get("finish_reason")
        if envelope["phase"] == "compressor":
            if tool_calls:
                raise RuntimeError("native HiAgent compressor returned a tool call")
            content = _trim_at_first_stop(content or "", COMPRESSOR_STOP)
            if matcher is not None and matcher.first_match is not None:
                finish_reason = "stop"
        usage = record.get("generation_usage_total")
        if not isinstance(usage, Mapping) or any(
            type(usage.get(field)) is not int or usage[field] < 0
            for field in ("prompt_tokens", "completion_tokens", "total_tokens")
        ):
            raise RuntimeError("native HiAgent runner usage is unavailable")
        receipt = {
            "schema": RECEIPT_SCHEMA,
            "parent_request_id": envelope["parent_request_id"],
            "official_eval_context": copy.deepcopy(envelope["official_eval_context"]),
            "proxy_attempt_uid": envelope["proxy_attempt_uid"],
            "native_attempt_uid": native_attempt_uid,
            "phase": envelope["phase"],
            "call_ordinal": envelope["call_ordinal"],
            "native_session_id": envelope["native_session_id"],
            "native_decision_key": envelope["native_decision_key"],
        }
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": copy.deepcopy(tool_calls) if tool_calls else None,
                    "reasoning_content": final.get("reasoning_content"),
                },
                "finish_reason": finish_reason,
            }],
            "usage": {field: usage[field] for field in (
                "prompt_tokens", "completion_tokens", "total_tokens"
            )},
            RECEIPT_FIELD: receipt,
        }

    def health(self) -> dict[str, Any]:
        return {
            "schema": "a-event-native-hiagent-health-v1",
            "status": "terminal" if self._terminal_failure is not None else "ok",
            "terminal_failure": copy.deepcopy(self._terminal_failure),
            "benchmark": self.benchmark,
            "model_name": self.model_name,
            "allowed_task_ids": sorted(self.allowed_task_ids),
            "completed_calls": len(self._completed),
            "capacity_rejected_tasks": len(self._task_failures),
            "deadline_exceeded": time.monotonic() >= self.deadline_monotonic,
            "actor_generation_calls": self.actor_runner.generation_calls,
            "auxiliary_generation_calls": self.auxiliary_runner.generation_calls,
            "join_path": str(self.join_path),
            "steps_path": str(self.steps_path),
        }


def build_event_native_hiagent_dispatcher(
    tokenizer: Any,
    *,
    actor_generator: Any,
    auxiliary_generator: Any,
    packing: Mapping[str, Any],
    policy: Mapping[str, Any],
    ratio: int,
    model_name: str,
    benchmark: str,
    allowed_task_ids: Sequence[str],
    actor_phase_sampling: Mapping[str, Mapping[str, Any]],
    actor_max_generation_calls: int,
    auxiliary_max_generation_calls: int,
    deadline_monotonic: float,
    actor_journal: AttemptJournal,
    auxiliary_journal: AttemptJournal,
    join_path: str | Path,
    steps_path: str | Path,
    model_context: int | None = None,
) -> EventNativeHiAgentDispatcher:
    """Build two real full-original runners over one shared loaded runtime."""

    if actor_generator is auxiliary_generator:
        raise ValueError("actor and auxiliary generators must be distinct")
    actor_runtime = getattr(actor_generator, "runtime", None)
    auxiliary_runtime = getattr(auxiliary_generator, "runtime", None)
    if actor_runtime is None or actor_runtime is not auxiliary_runtime:
        raise ValueError("actor and auxiliary generators must share one loaded runtime")
    if not isinstance(actor_journal, AttemptJournal):
        raise TypeError("actor_journal must be an AttemptJournal")
    if not isinstance(auxiliary_journal, AttemptJournal):
        raise TypeError("auxiliary_journal must be an AttemptJournal")
    if actor_journal.path.resolve() == auxiliary_journal.path.resolve():
        raise ValueError("actor and auxiliary attempt journals must be distinct")
    writer = _JoinWriter(join_path)
    if writer.path.resolve() in {
        actor_journal.path.resolve(), auxiliary_journal.path.resolve()
    }:
        raise ValueError("join_path must be distinct from both attempt journals")
    resolved_steps = Path(steps_path).resolve()
    if resolved_steps in {
        writer.path.resolve(), actor_journal.path.resolve(),
        auxiliary_journal.path.resolve(),
    }:
        raise ValueError("steps_path must be distinct from journals and join_path")
    if not isinstance(actor_phase_sampling, Mapping) or set(actor_phase_sampling) != ACTOR_PHASES:
        raise ValueError("actor_phase_sampling must explicitly configure both actor phases")
    normalized_sampling = {
        phase: validate_policy_sampling(actor_phase_sampling[phase])
        for phase in ACTOR_PHASES
    }
    caps = {value["max_completion_tokens"] for value in normalized_sampling.values()}
    if len(caps) != 1:
        raise ValueError("policy and retrieval must share one explicit completion cap")
    actor_cap = next(iter(caps))
    actor = _PhaseBoundGenerator(actor_generator, tokenizer)
    auxiliary = _PhaseBoundGenerator(auxiliary_generator, tokenizer)
    actor_runner = EventNativeDecisionRunner(
        build_event_native_controller(
            tokenizer,
            packing=packing,
            policy=policy,
            view_mode="full_original",
            model_context=model_context,
        ),
        actor,
        tokenizer,
        ratio=ratio,
        max_new_tokens=actor_cap,
        max_generation_calls=actor_max_generation_calls,
        journal=_JoiningAttemptJournal(actor_journal, writer, generator_role="actor"),
    )
    auxiliary_runner = EventNativeDecisionRunner(
        build_event_native_controller(
            tokenizer,
            packing=packing,
            policy=policy,
            view_mode="full_original",
            model_context=model_context,
        ),
        auxiliary,
        tokenizer,
        ratio=ratio,
        max_new_tokens=COMPRESSOR_MAX_COMPLETION_TOKENS,
        max_generation_calls=auxiliary_max_generation_calls,
        journal=_JoiningAttemptJournal(
            auxiliary_journal, writer, generator_role="auxiliary"
        ),
    )
    return EventNativeHiAgentDispatcher(
        actor_runner,
        auxiliary_runner,
        tokenizer,
        model_name=model_name,
        benchmark=benchmark,
        allowed_task_ids=allowed_task_ids,
        actor_phase_sampling=normalized_sampling,
        deadline_monotonic=deadline_monotonic,
        join_path=writer.path,
        steps_path=steps_path,
    )


def _first_stop(text: str, stop_strings: Sequence[str]) -> tuple[int, str] | None:
    matches = [
        (text.find(stop), order, stop)
        for order, stop in enumerate(stop_strings)
        if text.find(stop) >= 0
    ]
    if not matches:
        return None
    position, _, stop = min(matches)
    return position, stop


def _trim_at_first_stop(text: str, stop_strings: Sequence[str]) -> str:
    match = _first_stop(text, stop_strings)
    return text if match is None else text[:match[0]]


def validate_native_greedy_sampling(
    sampling: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    """Reject sampling fields that the current greedy generator cannot honor."""

    normalized = validate_policy_sampling(sampling)
    if float(normalized["temperature"]) != 0.0:
        raise ValueError(
            f"{phase} requests temperature={normalized['temperature']!r}; "
            "the native generator currently supports greedy temperature=0 only"
        )
    if "top_p" in normalized and float(normalized["top_p"]) != 1.0:
        raise ValueError(f"{phase} top_p must be 1 under the current greedy generator")
    if "min_p" in normalized and float(normalized["min_p"]) != 0.0:
        raise ValueError(f"{phase} min_p must be 0 under the current greedy generator")
    if "top_k" in normalized and normalized["top_k"] != 0:
        raise ValueError(f"{phase} top_k must be 0 under the current greedy generator")


def _classify_runner_failure(
    record: Mapping[str, Any],
    error: Exception,
) -> EventNativeHiAgentError:
    failure = record.get("error")
    error_type = failure.get("type") if isinstance(failure, Mapping) else None
    message = failure.get("message") if isinstance(failure, Mapping) else None
    if error_type in {"PackingBudgetError", "CapacityInfeasible"}:
        safe_message = (
            message
            if isinstance(message, str) and message
            else "native HiAgent input exceeds the configured model capacity"
        )
        return EventNativeHiAgentError(
            422,
            "capacity_rejected",
            safe_message,
            error_type=error_type,
        )
    return EventNativeHiAgentError(
        500,
        "runner_failed",
        "native HiAgent phase generation failed",
        error_type=error_type or type(error).__name__,
    )


def _is_loopback(host: str) -> bool:
    if not isinstance(host, str) or not host:
        return False
    try:
        addresses = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        return bool(addresses) and all(
            ipaddress.ip_address(address[4][0]).is_loopback for address in addresses
        )
    except (OSError, ValueError):
        return False


def make_hiagent_server(
    dispatcher: EventNativeHiAgentDispatcher,
    host: str = "127.0.0.1",
    port: int = 0,
) -> HTTPServer:
    """Construct the serial loopback endpoint; the caller owns its lifecycle."""

    if not isinstance(dispatcher, EventNativeHiAgentDispatcher):
        raise TypeError("dispatcher must be an EventNativeHiAgentDispatcher")
    if not _is_loopback(host):
        raise ValueError("native HiAgent host must resolve only to loopback addresses")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("port must be an integer in [0, 65535]")

    class Handler(BaseHTTPRequestHandler):
        server_version = "EventNativeHiAgent/1"

        def do_GET(self) -> None:
            if urlsplit(self.path).path != "/health":
                self._send_json(404, _error("not_found", "Not found"))
                return
            self._send_json(200, dispatcher.health())

        def do_POST(self) -> None:
            if urlsplit(self.path).path != ENDPOINT_PATH:
                self._send_json(404, _error("not_found", "Not found"))
                return
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].lower()
            if content_type.strip() != "application/json":
                self._send_json(415, _error("invalid_content_type", "Use application/json"))
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
                if length < 0:
                    raise ValueError
            except ValueError:
                self._send_json(411, _error("invalid_content_length", "Content-Length required"))
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._send_json(400, _error("invalid_json", "Body is not valid JSON"))
                return
            try:
                response = dispatcher.handle(payload)
            except EventNativeHiAgentError as error:
                self._send_json(
                    error.status_code,
                    _error(error.code, str(error), error_type=error.error_type),
                )
                return
            self._send_json(200, response)

        def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
            body = json.dumps(
                payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: Any) -> None:
            return

    class LoopbackServer(HTTPServer):
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    server = LoopbackServer((host, port), Handler)
    server.event_native_hiagent = dispatcher  # type: ignore[attr-defined]
    return server


def _error(
    code: str,
    message: str,
    *,
    error_type: str | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "message": message,
            "type": error_type or "invalid_request_error",
            "code": code,
        }
    }


__all__ = [
    "ACTOR_PHASES",
    "COMPRESSOR_MAX_COMPLETION_TOKENS",
    "COMPRESSOR_SAMPLING",
    "COMPRESSOR_STOP",
    "EventNativeHiAgentDispatcher",
    "EventNativeHiAgentError",
    "FirstSubstringStopMatcher",
    "SCHEMA",
    "build_event_native_hiagent_dispatcher",
    "make_hiagent_server",
    "validate_native_greedy_sampling",
]
