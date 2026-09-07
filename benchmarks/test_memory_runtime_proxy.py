"""Proxy seam tests for the opt-in A memory runtime."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy
from arms import Arm, get_arm


def test_wrong_raw_token_count_stops_future_requests(monkeypatch):
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", object())
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_BYTES_PER_KV_TOKEN", None)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_FATAL_ERROR", None)
    counts = {"memory_runtime": {"bytes_per_kv_token": 147456, "total_raw_prompt_tokens": 2}}
    normalized = {"cost": {"bytes_per_kv_token": 147456}, "usage": {"prompt_tokens": 4910}}
    with pytest.raises(proxy.MemoryRuntimeError, match="raw tokenizer count"):
        proxy._verify_memory_runtime_kv_bytes(counts, normalized)
    assert counts["memory_runtime"]["raw_prompt_tokens_verified_by_backend"] is False
    with pytest.raises(proxy.MemoryRuntimeError, match="raw tokenizer count"):
        proxy._check_memory_runtime_fatal()


def _counts():
    return {
        "system_raw": 1,
        "history_raw": 0,
        "current_raw": 1,
        "compressed": 0,
        "gist_tokens": 0,
        "original_tokens": 0,
        "n_gist_messages": 0,
        "compressed_records": [],
        "doc_packing": "message",
        "n_docs": 0,
        "dropped_docs": 0,
        "current_start_out_index": 1,
        "history_packed_original_tokens": None,
        "history_dropped_original_tokens": None,
        "history_packed_candidate_doc_count": None,
        "history_retained_fraction": None,
    }


def _payload(**extra):
    value = {
        "model": "m",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
        "c2kv_eval_context": {
            "run_id": "run-a",
            "task_id": "task-1",
            "attempt_id": "attempt-1",
            "decision_id": "decision-1",
        },
    }
    value.update(extra)
    return value


def _body():
    return {
        "choices": [{
            "message": {"content": "ok", "tool_calls": None},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1},
    }


class _Backend:
    name = "fake"
    wants_request_context = False
    needs_repair_plan = False

    def __init__(self, bytes_per_kv_token=147456):
        self.bytes_per_kv_token = bytes_per_kv_token

    def prepare_chat(self, payload, arm, plan):
        return dict(payload)

    def normalize_response(self, data):
        choice = data["choices"][0]
        message = choice["message"]
        cost = {}
        if self.bytes_per_kv_token is not None:
            cost["bytes_per_kv_token"] = self.bytes_per_kv_token
        return {
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage") or {},
            "cost": cost,
        }


class _Runtime:
    def __init__(self, mode="full_shared", expected_bytes=147456, order=None):
        self.mode = mode
        self.expected_bytes = expected_bytes
        self.order = order
        self.calls = []

    def apply(self, messages, assembled, counts, eval_context, tools):
        if self.order is not None:
            self.order.append("runtime")
        self.calls.append({
            "messages": messages,
            "assembled": assembled,
            "eval_context": eval_context,
            "tools": tools,
        })
        if not {"task_id", "attempt_id", "decision_id"} <= eval_context.keys():
            raise ValueError("missing explicit eval context")
        out = [dict(message) for message in assembled]
        out[-1]["content"] = str(out[-1].get("content") or "") + " [runtime]"
        updated = dict(counts)
        updated["memory_runtime"] = {
            "mode": self.mode,
            "bytes_per_kv_token": self.expected_bytes,
            "byte_geometry_verified_by_backend": False,
        }
        return out, updated


@pytest.fixture(autouse=True)
def _reset_runtime(monkeypatch):
    monkeypatch.setattr(proxy, "STATE", proxy.ProxyState())
    monkeypatch.setattr(proxy, "CACHE", proxy.ExtractCache())
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", None)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_BYTES_PER_KV_TOKEN", None)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_FATAL_ERROR", None)
    monkeypatch.setattr(proxy, "NO_UPSTREAM_RETRIES", False)
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", "")


def _drive(monkeypatch, runtime, arm, backend, payload, responses,
           request_log_path=""):
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "ARM", arm)
    monkeypatch.setattr(proxy, "BACKEND", backend)
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", request_log_path)
    posts = []

    def fake_post(path, body, timeout, retries=2):
        posts.append({"path": path, "body": body, "retries": retries})
        response = responses[len(posts) - 1]
        if isinstance(response, BaseException):
            raise response
        return response

    monkeypatch.setattr(proxy, "_post_json", fake_post)
    raw = json.dumps(payload).encode("utf-8")
    handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    handler.rfile = io.BytesIO(raw)
    handler.headers = {"Content-Length": str(len(raw))}
    handler.path = "/v1/chat/completions"
    sent = []
    handler._send_json = lambda code, obj: sent.append((code, obj))
    handler.do_POST()
    return sent, posts


@pytest.mark.parametrize("mode", [
    "legacy", "protect", "recover_once", "persistent",
])
def test_c2kv_runtime_modes_require_plain_c2kv4(mode):
    runtime = _Runtime(mode=mode)
    proxy._validate_memory_runtime_arm(runtime, get_arm("c2kv4"))
    with pytest.raises(proxy.MemoryRuntimeError, match="requires plain arm 'c2kv4'"):
        proxy._validate_memory_runtime_arm(runtime, get_arm("c2kv"))


@pytest.mark.parametrize("mode", ["no_gist", "full_shared"])
def test_full_runtime_modes_require_training_renderer_full(mode):
    runtime = _Runtime(mode=mode)
    proxy._validate_memory_runtime_arm(runtime, get_arm("full"))
    with pytest.raises(proxy.MemoryRuntimeError, match="requires plain arm 'full'"):
        proxy._validate_memory_runtime_arm(runtime, get_arm("full_native"))


def test_runtime_rejects_conflicting_arm_fields_even_with_allowed_name():
    conflicting = Arm(
        name="c2kv4", compress_history=True, ratio=4, hybrid_top_k=1)
    with pytest.raises(proxy.MemoryRuntimeError, match="hybrid_top_k"):
        proxy._validate_memory_runtime_arm(_Runtime(mode="protect"), conflicting)


def test_hook_runs_after_assemble_before_repair_and_disables_chat_retry(
        monkeypatch, tmp_path):
    order = []
    runtime = _Runtime(order=order)
    original_assemble = proxy._assemble

    def assemble(messages, arm):
        order.append("assemble")
        return original_assemble(messages, arm)

    def repair(messages, arm, counts, tools, out_messages):
        order.append("repair")
        assert counts["memory_runtime"]["mode"] == "full_shared"
        assert out_messages[-1]["content"].endswith("[runtime]")
        return None

    monkeypatch.setattr(proxy, "_assemble", assemble)
    monkeypatch.setattr(proxy, "plan_repair", repair)
    log_path = tmp_path / "requests.jsonl"
    sent, posts = _drive(
        monkeypatch, runtime, get_arm("full"), _Backend(), _payload(), [_body()],
        str(log_path),
    )
    order.append("done")

    assert order == ["assemble", "runtime", "repair", "done"]
    assert [post["retries"] for post in posts] == [0]
    assert posts[0]["body"]["messages"][-1]["content"].endswith("[runtime]")
    assert runtime.calls[0]["eval_context"]["task_id"] == "task-1"
    assert runtime.calls[0]["tools"] == []
    assert sent[0][0] == 200
    metadata = sent[0][1]["c2kv_proxy"]["memory_runtime"]
    assert metadata["byte_geometry_verified_by_backend"] is True
    assert metadata["backend_bytes_per_kv_token"] == 147456
    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert row["memory_runtime"] == metadata


def test_legacy_path_keeps_transport_retry_default_and_has_no_runtime_metadata(
        monkeypatch):
    sent, posts = _drive(
        monkeypatch, None, get_arm("full"), _Backend(), _payload(), [_body()])
    assert [post["retries"] for post in posts] == [2]
    assert sent[0][0] == 200
    assert "memory_runtime" not in sent[0][1]["c2kv_proxy"]


def test_no_upstream_retries_control_disables_transport_and_cachemiss_retry(
        monkeypatch):
    monkeypatch.setattr(proxy, "NO_UPSTREAM_RETRIES", True)
    sent, posts = _drive(
        monkeypatch, None, get_arm("full"), _Backend(), _payload(),
        [proxy.CacheMiss("C2KV_CACHE_MISS")],
    )
    assert sent[0][0] == 502
    assert len(posts) == 1 and posts[0]["retries"] == 0
    assert "cache miss" in sent[0][1]["error"]


@pytest.mark.parametrize("runtime,no_retries", [
    (_Runtime(), False),
    (None, True),
])
def test_retry_gate_also_covers_extract_transport_calls(
        monkeypatch, runtime, no_retries):
    attempts = []

    class Opener:
        def open(self, request, timeout):
            attempts.append(request)
            raise proxy.URLError("offline")

    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "NO_UPSTREAM_RETRIES", no_retries)
    monkeypatch.setattr(proxy, "UPSTREAM", "http://127.0.0.1:1")
    monkeypatch.setattr(proxy, "_OPENER", Opener())
    monkeypatch.setattr(proxy.time, "sleep", lambda seconds: pytest.fail(
        "retry gate must not back off"))
    with pytest.raises(proxy.UpstreamError):
        # Use the default retries=2 exactly as backend.extract does.
        proxy._post_json("/v1/c2kv/extract", {"text": "history"}, 1)
    assert len(attempts) == 1


@pytest.mark.parametrize("forbidden", [
    {"c2kv_oracle": {}},
    {"c2kv_repair": {"policy": "first"}},
    {"messages": [{
        "role": "user", "content": "hello",
        "c2kv_repair_only_key_hashes": ["raw"],
    }]},
])
def test_runtime_rejects_privileged_or_repair_payloads_before_apply(
        monkeypatch, forbidden):
    runtime = _Runtime()
    payload = _payload(**forbidden)
    sent, posts = _drive(
        monkeypatch, runtime, get_arm("full"), _Backend(), payload, [_body()])
    assert sent[0][0] == 502
    assert "cannot be combined" in sent[0][1]["error"]
    assert runtime.calls == []
    assert posts == []


def test_adapter_eval_context_error_is_a_runtime_error_without_upstream_call(
        monkeypatch, tmp_path):
    runtime = _Runtime()
    payload = _payload()
    payload["c2kv_eval_context"].pop("attempt_id")
    log_path = tmp_path / "requests.jsonl"
    sent, posts = _drive(
        monkeypatch, runtime, get_arm("full"), _Backend(), payload, [_body()],
        str(log_path),
    )
    assert sent[0][0] == 502
    assert posts == []
    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert row["status"] == "memory_runtime_error"
    assert row["memory_runtime"]["byte_geometry_verified_by_backend"] is False


def test_cache_miss_is_logged_and_not_reextracted_or_retried_in_runtime(
        monkeypatch, tmp_path):
    runtime = _Runtime(mode="legacy")
    assembled = [{
        "role": "user", "content": "old", "c2kv_key_hash": "gist",
        "c2kv_ratio": 4,
    }, {"role": "user", "content": "now"}]
    counts = _counts()
    counts.update({
        "compressed": 1,
        "gist_tokens": 2,
        "original_tokens": 8,
        "n_gist_messages": 1,
        "n_docs": 1,
        "current_start_out_index": 1,
        "compressed_records": [{
            "role": "user", "content": "old", "out_index": 0,
            "record": {"key_hash": "gist", "gist_len": 2,
                       "original_seq_len": 8},
        }],
    })
    monkeypatch.setattr(proxy, "_assemble", lambda messages, arm: (assembled, counts))
    monkeypatch.setattr(
        proxy, "_extract",
        lambda *args, **kwargs: pytest.fail("runtime must not re-extract on cache miss"),
    )
    log_path = tmp_path / "requests.jsonl"
    sent, posts = _drive(
        monkeypatch, runtime, get_arm("c2kv4"), _Backend(), _payload(),
        [proxy.CacheMiss("C2KV_CACHE_MISS")], str(log_path),
    )
    assert sent[0][0] == 502
    assert len(posts) == 1 and posts[0]["retries"] == 0
    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert row["status"] == "cache_miss"


def test_kv_byte_mismatch_fails_closed_and_latches_runtime_error(
        monkeypatch, tmp_path):
    runtime = _Runtime(expected_bytes=8192)
    log_path = tmp_path / "requests.jsonl"
    sent, posts = _drive(
        monkeypatch, runtime, get_arm("full"), _Backend(147456), _payload(),
        [_body()], str(log_path),
    )
    assert sent[0][0] == 502
    assert len(posts) == 1
    row = json.loads(log_path.read_text(encoding="utf-8").strip())
    metadata = row["memory_runtime"]
    assert row["status"] == "memory_runtime_error"
    assert metadata["byte_geometry_verified_by_backend"] is False
    assert metadata["byte_geometry_verification"] == "mismatch"
    assert metadata["backend_bytes_per_kv_token"] == 147456
    assert proxy.MEMORY_RUNTIME_FATAL_ERROR

    # Once the process observes a geometry disagreement, later requests fail
    # before the stateful adapter or transport can apply the decision twice.
    sent_again, posts_again = _drive(
        monkeypatch, runtime, get_arm("full"), _Backend(147456), _payload(),
        [_body()], str(log_path),
    )
    assert sent_again[0][0] == 502
    assert posts_again == []
    assert len(runtime.calls) == 1


def test_missing_backend_geometry_stays_explicitly_unverified(monkeypatch):
    runtime = _Runtime(expected_bytes=147456)
    sent, _ = _drive(
        monkeypatch, runtime, get_arm("full"), _Backend(None), _payload(), [_body()])
    assert sent[0][0] == 200
    metadata = sent[0][1]["c2kv_proxy"]["memory_runtime"]
    assert metadata["byte_geometry_verified_by_backend"] is False
    assert metadata["byte_geometry_verification"] == "pending_backend_cost"


def test_main_loads_runtime_flags_and_validates_mode_arm(monkeypatch):
    runtime = _Runtime(mode="full_shared")
    loaded = []
    served = []

    monkeypatch.setattr(
        proxy, "_load_memory_runtime",
        lambda config, tokenizer: loaded.append((config, tokenizer)) or runtime,
    )
    monkeypatch.setattr(proxy, "get_backend", lambda name, post: _Backend())

    class Server:
        def __init__(self, address, handler):
            served.append((address, handler))

        def serve_forever(self):
            served.append("served")

    monkeypatch.setattr(proxy, "ThreadingHTTPServer", Server)
    proxy.main([
        "--upstream", "http://127.0.0.1:1",
        "--backend", "sglang",
        "--arm", "full",
        "--port", "39999",
        "--memory-runtime-config", "runtime.json",
        "--memory-tokenizer", "tokenizer",
    ])
    assert loaded == [("runtime.json", "tokenizer")]
    assert proxy.MEMORY_RUNTIME is runtime
    assert served[-1] == "served"


def test_main_requires_runtime_config_and_tokenizer_together():
    with pytest.raises(SystemExit):
        proxy.main([
            "--upstream", "http://127.0.0.1:1",
            "--arm", "full",
            "--port", "39999",
            "--memory-runtime-config", "runtime.json",
        ])
