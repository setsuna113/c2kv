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

ACE_SOURCE_VERSION = "acebench-text-actions-v1"
ACE_RECEIPT_VERSION = "acebench-execution-receipt-v1"

_local = threading.local()
_lock = threading.Lock()


def native_mode():
    return os.environ.get("C2KV_ACE_NATIVE") == "1"


def record_source_mode():
    return os.environ.get("C2KV_ACE_RECORD_SOURCE") == "1"


def receipt_mode():
    return native_mode() or record_source_mode()


def inference_wrapper(fn):
    """Retain the complete official ID while upstream passes a suffix to scenes."""
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        previous = getattr(_local, "official_task", None)
        _local.official_task = str(bound.arguments["id"])
        try:
            return fn(*args, **kwargs)
        finally:
            _local.official_task = previous

    return wrapped


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
        task = (getattr(_local, "official_task", None) if native_mode()
                else None) or str(bound.arguments["test_id"])
        _local.task, _local.session = task, f"acebench:{task}:{uuid.uuid4().hex}"
        _local.source_official_task = getattr(_local, "official_task", None)
        _local.request = None
        _local.receipts = []
        _local.decision_keys = {}
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
            _local.source_official_task = None
            _local.receipts = _local.decision_keys = None
    return wrapped


def decode_wrapper(fn):
    """Observe the official decoder output without running it twice."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        receipt = getattr(_local, "execution_receipt", None)
        try:
            decoded = fn(*args, **kwargs)
        except BaseException:
            if receipt is not None:
                receipt.update(decode_status="error", decoded_calls=None)
            raise
        if receipt is not None:
            if native_mode():
                receipt.update(decode_status="ok", decoded_calls=list(decoded))
            elif isinstance(decoded, list) and all(isinstance(call, str) for call in decoded):
                receipt.update(decode_status="ok", decoded_calls=list(decoded))
            else:
                # Recording must not consume a generator or change the
                # official decoder's return value to inspect it.
                receipt.update(decode_status="invalid", decoded_calls=None)
        return decoded
    return wrapped


def executor_wrapper(fn):
    """Observe the official executor return shape and cardinality."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        result = fn(*args, **kwargs)
        receipt = getattr(_local, "execution_receipt", None)
        if receipt is not None:
            values = result[0] if isinstance(result, tuple) and result else None
            receipt.update(
                executor_status="returned",
                executor_return_shape="list" if isinstance(values, list) else "non_list",
                executor_return_count=len(values) if isinstance(values, list) else None,
            )
        return result
    return wrapped


def install_native_receipts(turn_module, step_module):
    for module, cls in ((turn_module, turn_module.EXECUTION),
                        (step_module, step_module.EXECUTION_STEP)):
        cls.decode_function_list = decode_wrapper(cls.decode_function_list)
        module.execute_agent_func_call = executor_wrapper(module.execute_agent_func_call)


def execution_wrapper(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        if not getattr(_local, "request", None):
            raise RuntimeError("ACEBench action has no measured agent decision")
        started = time.perf_counter_ns()
        start_unix = time.time_ns()
        status = "failed"
        receipt = None
        previous_receipt = getattr(_local, "execution_receipt", None)
        if receipt_mode():
            history = args[1] if len(args) > 1 else kwargs.get("history")
            valid_history = (isinstance(history, list) and bool(history)
                             and isinstance(history[-1], dict)
                             and history[-1].get("sender") == "agent")
            if native_mode() and not valid_history:
                raise RuntimeError("ACEBench execution has no official agent predecessor")
            receipt = {
                "version": ACE_RECEIPT_VERSION,
                "agent_history_index": len(history) - 1 if valid_history else None,
                "execution_message_index": len(history) + 1 if valid_history else None,
                "decode_status": "error",
                "decoded_calls": None,
                "executor_status": "not_called",
                "executor_return_shape": None,
                "executor_return_count": None,
            }
            _local.execution_receipt = receipt
        try:
            value = fn(*args, **kwargs)
            if receipt is not None:
                message = value[0] if isinstance(value, tuple) and value else None
                valid_message = (isinstance(message, dict)
                                 and message.get("sender") == "execution")
                if native_mode() and not valid_message:
                    raise RuntimeError("ACEBench official executor returned no execution message")
                if not valid_message or not valid_history:
                    receipt["decode_status"] = "invalid"
                _local.receipts.append(receipt)
            status = "completed"
            return value
        finally:
            _local.execution_receipt = previous_receipt
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
        if endpoint == agent:
            # A lost reply may follow a committed stateful generation.
            resource._client.max_retries = 0
        if measured:
            extra = dict(kwargs.get("extra_body") or {})
            if native_mode():
                messages = kwargs.get("messages")
                if not isinstance(messages, list) or not messages:
                    raise RuntimeError("Native ACEBench needs official role-preserving messages")
                if sum(row.get("role") == "tool" for row in messages) != len(_local.receipts):
                    raise RuntimeError("ACEBench execution receipts do not cover visible tool observations")
                if kwargs.get("top_p") != 1 or kwargs.get("temperature") != 0.001:
                    raise RuntimeError("Native ACEBench actor sampling differs from the official scene")
                if kwargs.get("max_tokens") != 1000:
                    raise RuntimeError("Native ACEBench actor cap differs from the official scene")
                task = getattr(_local, "task", None)
                if not isinstance(task, str) or not task:
                    raise RuntimeError("Native ACEBench has no official task identity")
                signature = json.dumps(messages, sort_keys=True, ensure_ascii=False)
                step = _local.decision_keys.setdefault(signature, len(_local.decision_keys))
                extra["c2kv_eval_context"] = {
                    "benchmark": "acebench", "task_id": task,
                    "user_turn": max(0, sum(row.get("role") == "user" for row in messages) - 1),
                    "step": step, "attempt": 0,
                }
                extra["c2kv_ace_source"] = {
                    "version": ACE_SOURCE_VERSION, "receipts": list(_local.receipts),
                }
            else:
                extra["c2kv_measurement_session_id"] = _local.session
                if record_source_mode():
                    official_task = getattr(_local, "source_official_task", None)
                    extra["c2kv_ace_official_task_id"] = official_task
                    extra["c2kv_ace_source"] = {
                        "version": ACE_SOURCE_VERSION, "receipts": list(_local.receipts),
                    }
            kwargs["extra_body"] = extra
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
    from model_inference.multi_turn import execution_role as turn_execution
    from model_inference.multi_step import execution_role_step as step_execution
    from openai.resources.chat.completions import Completions
    if receipt_mode():
        module.APIModelInference.inference = inference_wrapper(module.APIModelInference.inference)
        install_native_receipts(turn_execution, step_execution)
    for name in ("multi_turn_inference", "multi_step_inference"):
        setattr(module.APIModelInference, name, task_wrapper(getattr(module.APIModelInference, name)))
    for cls in (module.EXECUTION, module.EXECUTION_STEP):
        cls.respond = execution_wrapper(cls.respond)
    Completions.create = request_wrapper(Completions.create)
    sys.argv = [str(harness / "generate.py"), *sys.argv[2:]]
    runpy.run_path(str(harness / "generate.py"), run_name="__main__")


if __name__ == "__main__":
    main()
