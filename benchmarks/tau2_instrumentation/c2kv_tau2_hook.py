"""Agent-only task identity for the official, unmodified tau2 CLI.

The task context is created in tau2's per-task worker thread and injected only
through LLMAgent's ``agent_response`` call. User-simulator and judge calls keep
their original kwargs and go to their independently configured endpoint.
"""
from __future__ import annotations

import contextvars
import os
import re
import time
from collections.abc import Mapping
from typing import Any

from measurement.telemetry import HarnessTelemetry, current_episode


TELEMETRY_ENV = "C2KV_TAU2_TELEMETRY_PATH"
NATIVE_ENV = "C2KV_TAU2_NATIVE"
_task_id = contextvars.ContextVar("c2kv_tau2_task_id", default=None)
_proxy_request_id = contextvars.ContextVar("c2kv_tau2_proxy_request_id", default=None)
_installed = False
_ERROR_CODE = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_TRANSIENT_TRANSPORT_ERRORS = frozenset({
    ("openai", "APIConnectionError"),
    ("httpx", "ConnectError"),
    ("httpx", "ReadError"),
    ("httpx", "RemoteProtocolError"),
    ("httpx", "WriteError"),
    ("httpcore", "ConnectError"),
    ("httpcore", "ReadError"),
    ("httpcore", "RemoteProtocolError"),
    ("httpcore", "WriteError"),
})


def _safe_getattr(value: Any, name: str) -> Any:
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _error_code(value: Any) -> str | None:
    """Read only a bounded API code from a decoded error object."""
    if not isinstance(value, Mapping):
        return None
    candidates = [value.get("code")]
    nested = value.get("error")
    if isinstance(nested, Mapping):
        candidates.append(nested.get("code"))
    for candidate in candidates:
        if isinstance(candidate, str) and _ERROR_CODE.fullmatch(candidate):
            return candidate
    return None


def _exception_chain(error: BaseException) -> list[BaseException]:
    chain = []
    current: BaseException | None = error
    seen = set()
    for _ in range(8):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        chain.append(current)
        current = _safe_getattr(current, "__cause__") or _safe_getattr(
            current, "__context__")
    return chain


def _transient_transport_cause(error: BaseException) -> dict[str, str] | None:
    """Identify a real connection failure without matching exception text."""
    chain = _exception_chain(error)
    for current in chain:
        status = _safe_getattr(current, "status_code")
        response_status = _safe_getattr(_safe_getattr(current, "response"),
                                        "status_code")
        if any(type(value) is int and 400 <= value < 500
               for value in (status, response_status)):
            return None
    for current in chain:
        identity = (type(current).__module__.split(".", 1)[0],
                    type(current).__name__)
        if identity in _TRANSIENT_TRANSPORT_ERRORS:
            return {"module": identity[0], "exception_type": identity[1]}
    return None


def _exception_details(error: BaseException) -> dict[str, Any]:
    """Capture status/code without serializing request data, headers, or bodies."""
    status_code = None
    mapped_codes = []
    direct_codes = []
    current: BaseException | None = error
    seen = set()
    for _ in range(4):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        status = _safe_getattr(current, "status_code")
        if status_code is None and type(status) is int and 100 <= status <= 599:
            status_code = status
        direct_code = _safe_getattr(current, "code")
        if isinstance(direct_code, str) and _ERROR_CODE.fullmatch(direct_code):
            direct_codes.append(direct_code)
        sources = [_safe_getattr(current, "body"), _safe_getattr(current, "response")]
        response = sources[-1]
        response_status = _safe_getattr(response, "status_code")
        if status_code is None and type(response_status) is int and 100 <= response_status <= 599:
            status_code = response_status
        if response is not None and not isinstance(response, Mapping):
            decode = _safe_getattr(response, "json")
            if callable(decode):
                try:
                    sources.append(decode())
                except Exception:
                    pass
        for source in sources:
            code = _error_code(source)
            if code is not None:
                mapped_codes.append(code)
        current = current.__cause__ or current.__context__
    codes = mapped_codes + direct_codes
    api_error_code = next(
        (code for code in codes
         if not (code.isdigit() and status_code is not None
                 and int(code) == status_code)),
        None,
    )
    try:
        message = str(error)
    except Exception:
        message = type(error).__name__
    if len(message) > 2000:
        message = message[:1997] + "..."
    return {
        "exception_type": type(error).__name__,
        "message": message,
        "status_code": status_code,
        "api_error_code": api_error_code,
    }


def _request_id(response: Any) -> str | None:
    extra = getattr(response, "model_extra", None)
    if isinstance(extra, dict):
        proxy = extra.get("c2kv_proxy")
        if isinstance(proxy, dict) and proxy.get("request_id"):
            return str(proxy["request_id"])
    raw = getattr(response, "id", None)
    return str(raw) if raw else None


def install() -> bool:
    """Patch the stable tau2 worker and agent-call seams once per process."""
    global _installed
    path = os.environ.get(TELEMETRY_ENV)
    if not path:
        return False
    if _installed:
        return True

    from tau2.agent import llm_agent
    from tau2.environment.environment import Environment
    from tau2.runner import batch
    from tau2.user import user_simulator
    from tau2.utils import llm_utils

    telemetry = HarnessTelemetry(path, "tau2")
    native = os.environ.get(NATIVE_ENV) == "1"
    original_task = batch.run_single_task
    original_generate = llm_agent.generate
    original_completion = llm_utils.completion
    original_user_generate = user_simulator.generate
    original_tool = Environment.get_response

    def run_single_task(config, task, **kwargs):
        identity = str(task.id)
        token = _task_id.set(identity)
        try:
            with telemetry.episode(identity, {"task_id": identity,
                                              "seed": kwargs.get("seed")}):
                return original_task(config, task, **kwargs)
        finally:
            _task_id.reset(token)

    def completion(*args, **kwargs):
        result = original_completion(*args, **kwargs)
        if _task_id.get() is not None:
            _proxy_request_id.set(_request_id(result))
        return result

    def user_generate(*args, **kwargs):
        if kwargs.get("call_name") != "user_simulator_response":
            return original_user_generate(*args, **kwargs)
        try:
            return original_user_generate(*args, **kwargs)
        except Exception as exc:
            transport = _transient_transport_cause(exc)
            if transport is None:
                raise
            telemetry._emit(
                "user_simulator_retry", **(current_episode() or {}),
                call_name="user_simulator_response", retry_index=1,
                retry_limit=1, reason="transient_transport",
                transport_cause=transport, error=_exception_details(exc),
            )
            return original_user_generate(*args, **kwargs)

    def agent_generate(*args, **kwargs):
        if kwargs.get("call_name") != "agent_response":
            return original_generate(*args, **kwargs)
        identity = _task_id.get()
        if identity is None:
            raise RuntimeError("tau2 agent request has no official task context")
        kwargs = dict(kwargs)
        if not native:
            extra = dict(kwargs.get("extra_body") or {})
            extra["c2kv_measurement_session_id"] = identity
            kwargs["extra_body"] = extra
        _proxy_request_id.set(None)
        start_unix = time.time_ns()
        start_perf = time.perf_counter_ns()
        result = None
        error = None
        try:
            result = original_generate(*args, **kwargs)
            return result
        except BaseException as exc:
            error = _exception_details(exc)
            raise
        finally:
            telemetry.record_decision(
                request_id=_proxy_request_id.get(),
                start_unix_ns=start_unix,
                duration_ns=time.perf_counter_ns() - start_perf,
                response=result.model_dump(mode="json") if hasattr(result, "model_dump") else None,
                error=error,
            )

    def get_response(self, message):
        # The user simulator may call its own tools; record only actions made
        # by the evaluated agent, whose proxy decision is the cost denominator.
        if message.requestor != "assistant":
            return original_tool(self, message)
        start_unix = time.time_ns()
        start_perf = time.perf_counter_ns()
        outcome = None
        error = None
        try:
            outcome = original_tool(self, message)
            return outcome
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            telemetry.record_action(
                action=message.model_dump(mode="json"),
                outcome=outcome.model_dump(mode="json") if hasattr(outcome, "model_dump") else None,
                start_unix_ns=start_unix,
                duration_ns=time.perf_counter_ns() - start_perf,
                status="error" if error or getattr(outcome, "error", False) else "ok",
                error=error,
            )

    batch.run_single_task = run_single_task
    llm_utils.completion = completion
    llm_agent.generate = agent_generate
    user_simulator.generate = user_generate
    Environment.get_response = get_response
    _installed = True
    return True
