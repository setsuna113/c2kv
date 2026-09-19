"""Instrument official ACEBench execution without changing its scorer."""
from __future__ import annotations
import functools
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import threading
import time
import uuid

_local = threading.local()
_lock = threading.Lock()


def emit(event, **fields):
    path = Path(os.environ["C2KV_ACEBENCH_TELEMETRY"])
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"schema": "c2kv.harness.telemetry.v1", "event_type": event,
           "benchmark": "acebench", "unix_ns": time.time_ns(),
           "episode_id": getattr(_local, "task", None),
           "episode_instance_id": getattr(_local, "session", None),
           "task_id": getattr(_local, "task", None),
           "session_id": getattr(_local, "session", None), **fields}
    with _lock, path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def task_wrapper(fn):
    signature = inspect.signature(fn)
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        task = str(bound.arguments["test_id"])
        _local.task, _local.session = task, f"acebench:{task}:{uuid.uuid4().hex}"
        _local.request = None
        started = time.perf_counter_ns()
        start_unix = time.time_ns()
        status = "failed"
        emit("episode_start", start_unix_ns=start_unix)
        try:
            value = fn(*args, **kwargs)
            status = "completed"
            return value
        finally:
            emit("episode_end", status=status, start_unix_ns=start_unix,
                 end_unix_ns=time.time_ns(), duration_ns=time.perf_counter_ns()-started)
            _local.task = _local.session = _local.request = None
    return wrapped


def execution_wrapper(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if not getattr(_local, "request", None):
            raise RuntimeError("ACEBench action has no measured agent decision")
        started = time.perf_counter_ns()
        start_unix = time.time_ns()
        status = "failed"
        try:
            value = fn(*args, **kwargs)
            status = "completed"
            return value
        finally:
            emit("tool_action", decision_request_id=_local.request,
                 action_id=uuid.uuid4().hex, status=status,
                 start_unix_ns=start_unix, end_unix_ns=time.time_ns(),
                 duration_ns=time.perf_counter_ns()-started)
    return wrapped


def request_wrapper(fn):
    @functools.wraps(fn)
    def wrapped(resource, *args, **kwargs):
        endpoint = str(resource._client.base_url).rstrip("/")
        agent = os.environ["ACEBENCH_AGENT_BASE_URL"].rstrip("/")
        measured = getattr(_local, "session", None) and endpoint == agent
        if measured:
            kwargs["extra_body"] = {**(kwargs.get("extra_body") or {}),
                                     "c2kv_measurement_session_id": _local.session}
        started, start_unix = time.perf_counter_ns(), time.time_ns()
        result = fn(resource, *args, **kwargs)
        if measured:
            proxy = getattr(result, "c2kv_proxy", None)
            if proxy is None:
                proxy = (getattr(result, "model_extra", None) or {}).get("c2kv_proxy")
            if not isinstance(proxy, dict) or not proxy.get("request_id"):
                raise RuntimeError("ACEBench agent response lacks proxy request identity")
            _local.request = proxy["request_id"]
            emit("decision", decision_request_id=_local.request,
                 start_unix_ns=start_unix, end_unix_ns=time.time_ns(),
                 duration_ns=time.perf_counter_ns()-started)
        return result
    return wrapped


def main():
    harness = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(harness))
    from model_inference import apimodel_inference as module
    from openai.resources.chat.completions import Completions
    for name in ("multi_turn_inference", "multi_step_inference"):
        setattr(module.APIModelInference, name, task_wrapper(getattr(module.APIModelInference, name)))
    for cls in (module.EXECUTION, module.EXECUTION_STEP):
        cls.respond = execution_wrapper(cls.respond)
    Completions.create = request_wrapper(Completions.create)
    sys.argv = [str(harness / "generate.py"), *sys.argv[2:]]
    runpy.run_path(str(harness / "generate.py"), run_name="__main__")


if __name__ == "__main__":
    main()
