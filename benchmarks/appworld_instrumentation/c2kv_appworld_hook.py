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


TELEMETRY_ENV = "C2KV_APPWORLD_TELEMETRY_PATH"
RUN_DIR_ENV = "C2KV_APPWORLD_RUN_DIR"
_INSTALLED = False
_response_request_id = contextvars.ContextVar(
    "c2kv_appworld_response_request_id", default=None)


def _proxy_request_id(response: Any) -> Optional[str]:
    """Return only the request id explicitly attached by the C2KV proxy."""
    extra = getattr(response, "model_extra", None)
    if extra is None and isinstance(response, dict):
        extra = response
    if not isinstance(extra, dict):
        return None
    proxy = extra.get("c2kv_proxy")
    if not isinstance(proxy, dict) or not proxy.get("request_id"):
        return None
    return str(proxy["request_id"])


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

    telemetry = HarnessTelemetry(path, "appworld")

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
            if episode is not None:
                extra_body = dict(kwargs.get("extra_body") or {})
                extra_body["c2kv_measurement_session_id"] = episode["episode_id"]
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
        start_unix = time.time_ns()
        start_perf = time.perf_counter_ns()
        try:
            outcome = self.world.execute(action)
        except BaseException as exc:
            telemetry.record_action(
                action=action, outcome=None, start_unix_ns=start_unix,
                duration_ns=time.perf_counter_ns() - start_perf,
                action_index=getattr(self, "num_interactions", None),
                status="error", error=f"{type(exc).__name__}: {exc}",
                metadata={"raw_action": code} if action != code else None,
            )
            raise
        telemetry.record_action(
            action=action, outcome=outcome, start_unix_ns=start_unix,
            duration_ns=time.perf_counter_ns() - start_perf,
            action_index=getattr(self, "num_interactions", None), status="ok",
            metadata={"raw_action": code} if action != code else None,
        )
        return outcome

    def measured_forward(self, prompt):
        token = _response_request_id.set(None)
        start_unix = time.time_ns()
        start_perf = time.perf_counter_ns()
        try:
            output = original_forward(self, prompt)
        except BaseException as exc:
            telemetry.record_decision(
                request_id=_response_request_id.get(),
                start_unix_ns=start_unix,
                duration_ns=time.perf_counter_ns() - start_perf,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        else:
            telemetry.record_decision(
                request_id=_response_request_id.get(),
                start_unix_ns=start_unix,
                duration_ns=time.perf_counter_ns() - start_perf,
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
