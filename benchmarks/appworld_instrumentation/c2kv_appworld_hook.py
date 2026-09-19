"""Runtime-only AppWorld telemetry for the unmodified ACON runner.

The adapter adds this directory to ``PYTHONPATH`` so ``sitecustomize`` calls
``install`` before ACON imports its runner.  The hook deliberately wraps the
smallest stable boundaries: one agent decision and the exact
``AppWorld.execute`` call that commits an action.
"""
from __future__ import annotations

import contextvars
import os
import sys
import time
from typing import Any, Optional

from measurement.telemetry import HarnessTelemetry, current_episode
from source_annotations import appworld_doc_spans, pure_appworld_doc_action


TELEMETRY_ENV = "C2KV_APPWORLD_TELEMETRY_PATH"
RUN_DIR_ENV = "C2KV_APPWORLD_RUN_DIR"
TOOL_CONTEXT_ENV = "C2KV_TOOL_CONTEXT_ON"
_INSTALLED = False
_response_request_id = contextvars.ContextVar(
    "c2kv_appworld_response_request_id", default=None)
_visible_doc_outputs = contextvars.ContextVar(
    "c2kv_appworld_visible_doc_outputs", default=())
_clock_gettime_ns = getattr(time, "clock_gettime_ns", None)
_clock_monotonic_raw = getattr(time, "CLOCK_MONOTONIC_RAW", None)
_clock_realtime = getattr(time, "CLOCK_REALTIME", None)
_fallback_monotonic_ns = time.perf_counter_ns
_fallback_realtime_ns = time.time_ns


def _monotonic_raw_ns() -> int:
    if _clock_gettime_ns is not None and _clock_monotonic_raw is not None:
        return _clock_gettime_ns(_clock_monotonic_raw)
    return _fallback_monotonic_ns()


def _realtime_ns() -> int:
    if _clock_gettime_ns is not None and _clock_realtime is not None:
        return _clock_gettime_ns(_clock_realtime)
    return _fallback_realtime_ns()


def _proxy_request_id(response: Any) -> Optional[str]:
    """Prefer the proxy join id, falling back to the OpenAI response id."""
    extra = getattr(response, "model_extra", None)
    if extra is None and isinstance(response, dict):
        extra = response
    if isinstance(extra, dict):
        proxy = extra.get("c2kv_proxy")
        if isinstance(proxy, dict) and proxy.get("request_id"):
            return str(proxy["request_id"])
    response_id = (
        response.get("id") if isinstance(response, dict)
        else getattr(response, "id", None)
    )
    return str(response_id) if response_id else None


def _close_episode(instance: Any, exc_info=(None, None, None)) -> None:
    context = getattr(instance, "_c2kv_episode_context", None)
    if context is None:
        return
    delattr(instance, "_c2kv_episode_context")
    context.__exit__(*exc_info)


def install() -> bool:
    """Install telemetry wrappers once; return false when telemetry is off."""
    global _INSTALLED
    path = os.environ.get(TELEMETRY_ENV)
    if not path:
        return False
    if _INSTALLED:
        return True

    from productive_agents.agents.unified_agent import UnifiedAgent
    from productive_agents.env.appworld.env import AppWorldEnv

    telemetry = HarnessTelemetry(
        path, "appworld", unix_ns=_realtime_ns, monotonic_ns=_monotonic_raw_ns,
    )

    # ACON's vLLM.generate returns only message.content.  Capture the proxy id
    # from the OpenAI response before ACON discards model_extra.
    try:
        from openai.resources.chat.completions import Completions
    except ImportError:
        Completions = None
    if Completions is not None and not getattr(Completions.create, "_c2kv_wrapped", False):
        original_create = Completions.create

        def measured_create(self, *args, **kwargs):
            episode = current_episode()
            if episode is not None or os.environ.get(TOOL_CONTEXT_ENV) == "1":
                extra_body = dict(kwargs.get("extra_body") or {})
                if episode is not None:
                    extra_body["c2kv_measurement_session_id"] = episode["episode_id"]
                if os.environ.get(TOOL_CONTEXT_ENV) == "1":
                    messages = kwargs.get("messages") or (args[0] if args else [])
                    spans = appworld_doc_spans(messages, _visible_doc_outputs.get())
                    if spans:
                        extra_body["c2kv_tool_spans_v1"] = spans
                kwargs["extra_body"] = extra_body
            response = original_create(self, *args, **kwargs)
            _response_request_id.set(_proxy_request_id(response))
            return response

        measured_create._c2kv_wrapped = True
        Completions.create = measured_create

    original_reset = AppWorldEnv.reset
    original_close = AppWorldEnv.close
    original_forward = UnifiedAgent.forward

    def measured_reset(self, seed=None, task_id=None, **kwargs):
        _close_episode(self)
        _visible_doc_outputs.set(())
        metadata = {
            "experiment_name": str(getattr(self, "experiment_name", "")),
            "max_interactions": getattr(getattr(self, "config", None),
                                        "max_interactions", None),
            "raw_task_dir": (
                str(os.path.join(os.environ[RUN_DIR_ENV], f"task_{task_id}"))
                if os.environ.get(RUN_DIR_ENV) else None
            ),
        }
        context = telemetry.episode(str(task_id), metadata=metadata)
        context.__enter__()
        self._c2kv_episode_context = context
        try:
            return original_reset(self, seed=seed, task_id=task_id, **kwargs)
        except BaseException:
            _close_episode(self, sys.exc_info())
            raise

    def measured_close(self):
        try:
            return original_close(self)
        except BaseException:
            exc_info = sys.exc_info()
            _close_episode(self, exc_info)
            raise
        finally:
            if sys.exc_info()[0] is None:
                _close_episode(self)

    def measured_execute(self, code):
        action = self._clean_code(code)
        doc_source = pure_appworld_doc_action(action)
        start_unix = _realtime_ns()
        start_perf = _monotonic_raw_ns()
        try:
            outcome = self.world.execute(action)
        except BaseException as exc:
            telemetry.record_action(
                action=action, outcome=None, start_unix_ns=start_unix,
                duration_ns=_monotonic_raw_ns() - start_perf,
                action_index=getattr(self, "num_interactions", None),
                status="error", error=f"{type(exc).__name__}: {exc}",
                metadata={"raw_action": code} if action != code else None,
            )
            raise
        telemetry.record_action(
            action=action, outcome=outcome, start_unix_ns=start_unix,
            duration_ns=_monotonic_raw_ns() - start_perf,
            action_index=getattr(self, "num_interactions", None), status="ok",
            metadata={"raw_action": code} if action != code else None,
        )
        if doc_source and isinstance(outcome, str) and outcome:
            _visible_doc_outputs.set((*_visible_doc_outputs.get(), (doc_source, outcome)))
        return outcome

    def measured_forward(self, prompt):
        token = _response_request_id.set(None)
        start_unix = _realtime_ns()
        start_perf = _monotonic_raw_ns()
        try:
            output = original_forward(self, prompt)
        except BaseException as exc:
            telemetry.record_decision(
                request_id=_response_request_id.get(),
                start_unix_ns=start_unix,
                duration_ns=_monotonic_raw_ns() - start_perf,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        else:
            telemetry.record_decision(
                request_id=_response_request_id.get(),
                start_unix_ns=start_unix,
                duration_ns=_monotonic_raw_ns() - start_perf,
                response={
                    "raw_response": getattr(output, "response", None),
                    "action": getattr(output, "action", None),
                },
            )
            return output
        finally:
            _response_request_id.reset(token)

    measured_reset._c2kv_wrapped = True
    measured_close._c2kv_wrapped = True
    measured_execute._c2kv_wrapped = True
    measured_forward._c2kv_wrapped = True
    AppWorldEnv.reset = measured_reset
    AppWorldEnv.close = measured_close
    AppWorldEnv._execute_code = measured_execute
    UnifiedAgent.forward = measured_forward
    _INSTALLED = True
    return True
