"""Account auxiliary summary generations separately from actor generations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from urllib.request import Request

from .attempt_journal import AttemptJournal
from .text_summary import SummaryRenderer, VERSION, digest


class SummaryTransport:
    def __init__(self, proxy, runtime):
        self.proxy, self.runtime = proxy, runtime
        self.config = runtime.summary_config
        if (self.config.get("summary_model") != "c2kv-agent"
                or self.config.get("summary_prompt_token_cap") != 1024
                or self.config.get("summary_attempts_per_task") != 1152):
            raise ValueError("Unexpected text-summary model or resource contract")
        if not proxy.REQUEST_LOG_PATH:
            raise ValueError("Summary generation requires durable request logging")
        base = Path(proxy.REQUEST_LOG_PATH)
        self.journal = AttemptJournal(base.with_name("summary_attempts_" + base.name))
        self.trace_path = base.with_name("summary_trace_" + base.name)
        self.attempts = {}

    def append(self, record):
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_path.open("a", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            output.flush()
            os.fsync(output.fileno())

    def __call__(self, messages, limit, context, key):
        proxy, runtime = self.proxy, self.runtime
        expected = runtime._token_counter(messages, None)
        if expected > self.config["summary_prompt_token_cap"]:
            raise ValueError("Summary prompt exceeds its fixed cap")
        identity = (runtime.run_id, context["task_id"])
        index = self.attempts.get(identity, 0) + 1
        if index > self.config["summary_attempts_per_task"]:
            raise ValueError("Summary model attempt budget exhausted")
        request_id = (proxy._ATTEMPT_REQUEST.get() or {}).get("request_id")
        if not request_id:
            raise ValueError("Summary generation lacks a parent request identity")
        payload = {"model": self.config["summary_model"], "messages": messages,
            "temperature": 0.001, "seed": 0, "max_completion_tokens": limit,
            "stream": False, "c2kv_use_gist_projection": False}
        payload = proxy.BACKEND.prepare_chat(payload, proxy.get_arm("full"), None)
        handle = self.journal.start("generation", index, request_id, context)
        self.attempts[identity] = index
        record = {"schema": "a-text-summary-transport-v1", "event": "started",
            "version": VERSION, "attempt_uid": handle.attempt_uid,
            "request_id": request_id, "eval_context": context, "summary_key": key,
            "summary_attempt_index": index, "status": "started",
            "expected_prompt_tokens": expected, "request_view": payload,
            "request_sha256": digest(payload), "actor_generation": False,
            "submitted_to_executor": False, "automatic_retries": 0}
        self.append(record)
        started = time.perf_counter()
        usage = None
        try:
            request = Request(proxy.UPSTREAM.rstrip("/") + "/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            # This deliberately bypasses actor-only counters in proxy._post_json.
            # Its independent cap plus the actor cap bounds total model calls;
            # the enclosing fixed-stage supervisor bounds their combined wall.
            with proxy._OPENER.open(request, timeout=600) as response:
                data = json.loads(response.read().decode("utf-8"))
            usage = data.get("usage")
            normalized = proxy.BACKEND.normalize_response(data)
            verification = {"memory_runtime": {
                "bytes_per_kv_token": runtime.bytes_per_kv_token,
                "total_raw_prompt_tokens": expected}}
            proxy._verify_memory_runtime_kv_bytes(verification, normalized)
            content = normalized.get("content")
            if (normalized.get("tool_calls") or not isinstance(content, str) or not content.strip()
                    or normalized.get("finish_reason") not in ("stop", "length")):
                raise ValueError("Summary response is not usable bounded plain text")
            usage = normalized.get("usage")
            if (not isinstance(usage, dict) or usage.get("prompt_tokens") != expected
                    or type(usage.get("completion_tokens")) is not int
                    or not 0 <= usage["completion_tokens"] <= limit):
                raise ValueError("Summary response usage violates its measured token contract")
            result = {"content": content, "finish_reason": normalized["finish_reason"],
                "usage": usage, "attempt_uid": handle.attempt_uid,
                "wall_sec": time.perf_counter() - started}
            self.append({**record, "event": "finished", "status": "completed",
                "usage": usage, "response_view": result,
                "cost": normalized.get("cost"), "backend_verification": verification,
                "wall_sec": result["wall_sec"]})
        except BaseException as error:
            self.append({**record, "event": "finished", "status": "failed",
                "usage": usage, "error_type": type(error).__name__,
                "wall_sec": time.perf_counter() - started})
            self.journal.finish(handle, "failed", usage=usage)
            raise
        self.journal.finish(handle, "completed", usage=usage)
        return result


def render_summary(proxy, source, context):
    runtime = proxy.MEMORY_RUNTIME
    if not hasattr(runtime, "summary_renderer"):
        runtime.summary_transport = SummaryTransport(proxy, runtime)
        runtime.summary_renderer = SummaryRenderer(proxy, runtime.tokenizer, runtime.summary_transport)
    context = {**context, "run_id": runtime.run_id}
    result = runtime.summary_renderer.render(source, context)
    runtime.summary_transport.append({"schema": "a-text-summary-lookups-v1",
        "event": "lookups", "eval_context": context,
        "request_id": (proxy._ATTEMPT_REQUEST.get() or {}).get("request_id"),
        "lookups": result["lookups"], "producer_calls": result["producer_calls"],
        "wall_sec": result["wall_sec"]})
    return result
