"""Proxy seam tests for the opt-in A memory runtime."""

from __future__ import annotations

import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy
from arms import Arm, get_arm


def test_raw_recency_hook_reuses_full_renderer_before_system_insertion(monkeypatch):
    from memory_runtime.adapter import RuntimeAdapter

    source = [
        {"role": "user", "content": "Read the file"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call1", "type": "function", "function": {
                "name": "read", "arguments": '{"file":"note.txt"}'}}]},
        {"role": "tool", "tool_call_id": "call1", "content": "observed text"},
    ]
    runtime = RuntimeAdapter(dict(mode="raw_recency", run_id="r", bytes_per_kv_token=1,
                                  history_budget_bytes=5000, workspace_budget_bytes=1),
                             lambda messages, tools: len(json.dumps(messages)))
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    proxy._validate_memory_runtime_arm(runtime, get_arm("full"))
    full, original_counts = proxy._assemble(source, get_arm("full"))
    selected, counts = proxy._apply_memory_runtime(
        source, full, original_counts, {"task_id": "t", "attempt": 0, "decision_id": "d"}, [])
    assert selected == full
    assert selected[0]["role"] == "system"
    assert selected[-1] == {"role": "user", "content": "observed text"}
    assert all("tool_calls" not in message for message in selected)
    assert counts["memory_runtime"]["evidence_bytes"] == 0
    assert counts["memory_runtime"]["workspace_budget_applies"] is False
    assert counts["memory_runtime"]["mode"] == "raw_recency"


def test_capacity_gate_precedes_extraction_and_preserves_full_backend_payload(monkeypatch):
    from memory_runtime.adapter import RuntimeAdapter
    from backends.sglang import SglangBackend

    source = [{"role": "user", "content": "old query"},
              {"role": "assistant", "content": "old answer"},
              {"role": "user", "content": "current query"}]
    runtime = RuntimeAdapter(dict(mode="capacity_protect", run_id="r", bytes_per_kv_token=1,
                                  history_budget_bytes=5000, workspace_budget_bytes=2500),
                             lambda messages, tools: len(json.dumps(messages)))
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "ARM", get_arm("c2kv4"))

    def forbidden_extract(*args, **kwargs):
        raise AssertionError("Gate must run before extraction")

    monkeypatch.setattr(proxy, "_extract", forbidden_extract)
    full, _ = proxy._assemble(source, get_arm("full"))
    selected, counts = proxy._prepare_memory_input(
        source, {"task_id": "t", "attempt": 0, "decision_id": "d"}, [])
    assert selected == full
    assert counts["memory_runtime"]["capacity_gate"]["compression_activated"] is False
    backend = SglangBackend(forbidden_extract)
    payload = {"messages": selected, "tools": [], "c2kv_use_gist_projection": False}
    assert backend.prepare_chat(payload, get_arm("c2kv4"), None) == backend.prepare_chat(
        {**payload, "messages": full}, get_arm("full"), None)


def test_full_capacity_aux_never_extracts_and_preserves_every_full_message(monkeypatch):
    from memory_runtime.adapter import RuntimeAdapter

    source = [{"role": "user", "content": "earlier instruction"},
              {"role": "assistant", "content": "earlier work " * 1000},
              {"role": "user", "content": "Read note.txt"},
              {"role": "assistant", "content": None, "tool_calls": [
                  {"id": "c", "type": "function", "function": {
                      "name": "read", "arguments": '{"file":"note.txt"}'}}]},
              {"role": "tool", "tool_call_id": "c", "content": "observed text"}]
    runtime = RuntimeAdapter(dict(mode="full_capacity_aux", run_id="r", bytes_per_kv_token=1,
                                  history_budget_bytes=3000, workspace_budget_bytes=2500),
                             lambda messages, tools: len(json.dumps(messages)))
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "ARM", get_arm("full"))

    def forbidden_extract(*args, **kwargs):
        raise AssertionError("Full auxiliary control must never extract gist")

    monkeypatch.setattr(proxy, "_extract", forbidden_extract)
    full, _ = proxy._assemble(source, get_arm("full"))
    out, counts = proxy._prepare_memory_input(
        source, {"task_id": "t", "attempt": 0, "decision_id": "d"}, [])
    meta = counts["memory_runtime"]
    evidence_index = meta["evidence_out_index"]
    assert meta["auxiliary_gate"]["auxiliary_activated"] is True
    assert out[:evidence_index] + out[evidence_index + 1:] == full
    assert meta["retrieved_event_ids"] == []
    assert meta["active_history_bytes"] > 3000
    assert meta["gist_tokens"] == 0


def test_trace_observes_real_memo_hit_and_failed_forced_producer(monkeypatch):
    class Backend:
        def __init__(self):
            self.calls = 0

        def extract(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise ValueError("forced extraction failed")
            return {"key_hash": "gist", "original_seq_len": 8, "gist_len": 2}

    backend = Backend()
    monkeypatch.setattr(proxy, "BACKEND", backend)
    with proxy.capture_extractions() as trace:
        with proxy.extraction_sources([0, 1]):
            original = proxy._extract("user", "history", 4)
        with proxy.extraction_sources([3]):
            cached = proxy._extract("user", "history", 4)
            with pytest.raises(ValueError, match="forced extraction failed"):
                proxy._extract("user", "history", 4, force=True)
        snapshot = trace.snapshot(block_refs=[], forwarded_requests=[], request_status="error")
    assert backend.calls == 2
    assert cached is original
    assert original == {"key_hash": "gist", "original_seq_len": 8, "gist_len": 2}
    events = snapshot["events"]
    assert [event["client_cache_hit"] for event in events] == [False, True, False]
    assert [event["producer_called"] for event in events] == [True, False, True]
    assert [event["source_indices"] for event in events] == [[0, 1], [3], [3]]
    assert events[-1]["error_type"] == "ValueError"
    assert proxy.current_extraction_trace() is None


def test_extraction_budget_reserves_before_real_producer_and_counts_failure(monkeypatch):
    budget = proxy.ExtractionBudget(2)
    monkeypatch.setattr(proxy, "EXTRACTION_BUDGET", budget)
    cache = proxy.ExtractCache()
    calls = []

    def successful():
        calls.append("success")
        return {"key_hash": "gist", "original_seq_len": 8, "gist_len": 2}

    def failed():
        calls.append("failed")
        raise ValueError("producer failed")

    with budget.request_scope() as request_budget, proxy.capture_extractions() as trace:
        assert cache.get_or_put(("user", "a", 4, "tools"), successful)["key_hash"] == "gist"
        assert cache.get_or_put(("user", "a", 4, "tools"), successful)["key_hash"] == "gist"
        with pytest.raises(ValueError, match="producer failed"):
            cache.get_or_put(("user", "b", 4, "tools"), failed)
        with pytest.raises(proxy.ExtractionBudgetExceeded):
            cache.get_or_put(("user", "c", 4, "tools"), successful)
        snapshot = trace.snapshot(block_refs=[], forwarded_requests=[], request_status="error")
    assert calls == ["success", "failed"]
    assert request_budget.attempt_indices == [1, 2]
    assert [event["budget_attempt_index"] for event in snapshot["events"]] == [1, None, 2, None]
    assert snapshot["summary"]["producer_calls"] == 2
    assert snapshot["summary"]["producer_failures"] == 1
    assert snapshot["summary"]["producer_blocked"] == 1


def test_trace_captures_split_candidates_without_changing_assembly(monkeypatch):
    class Backend:
        def extract(self, text, role, ratio, **kwargs):
            return {"key_hash": proxy._digest([role, text]),
                    "original_seq_len": len(text), "gist_len": max(1, len(text) // ratio)}

    monkeypatch.setattr(proxy, "BACKEND", Backend())
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    monkeypatch.setattr(proxy, "MAX_DOC_LENGTH", 96)
    monkeypatch.setattr(proxy, "MAX_DOC_NUM", 1)
    source = [{"role": "user", "content": "history " * 40},
              {"role": "assistant", "content": "answer " * 30},
              {"role": "user", "content": "current"}]
    expected, _ = proxy._assemble(source, get_arm("c2kv4"))
    monkeypatch.setattr(proxy, "CACHE", proxy.ExtractCache())
    with proxy.capture_extractions() as trace:
        actual, counts = proxy._assemble(source, get_arm("c2kv4"))
        snapshot = trace.snapshot(block_refs=[], forwarded_requests=[[
            message["c2kv_key_hash"] for message in actual if message.get("c2kv_key_hash")]],
            request_status="ok")
    assert actual == expected
    assert counts["n_docs"] == 1
    assert len(snapshot["events"]) > counts["history_packed_candidate_doc_count"]
    assert all(event["source_indices"] == [0, 1] for event in snapshot["events"])


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
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", proxy.GenerationBudget(None))
    monkeypatch.setattr(proxy, "EXTRACTION_BUDGET", proxy.ExtractionBudget(None))
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", "")


def _drive(monkeypatch, runtime, arm, backend, payload, responses,
           request_log_path="", send_callback=None):
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "ARM", arm)
    monkeypatch.setattr(proxy, "BACKEND", backend)
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", request_log_path)
    posts = []

    def fake_post(path, body, timeout, retries=2):
        proxy._reserve_generation_attempt(path)
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
    def send(code, obj):
        sent.append((code, obj))
        if send_callback is not None:
            send_callback(code, obj)

    handler._send_json = send
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


@pytest.mark.parametrize("mode", ["no_gist", "full_shared", "full_capacity_aux"])
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
    assert "generation_budget" not in sent[0][1]["c2kv_proxy"]


def test_generation_budget_exhausts_across_full_requests(monkeypatch, tmp_path):
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", proxy.GenerationBudget(1))
    log_path = tmp_path / "generation_budget.jsonl"

    first_sent, first_posts = _drive(
        monkeypatch, None, get_arm("full"), _Backend(), _payload(), [_body()],
        str(log_path),
    )
    second_sent, second_posts = _drive(
        monkeypatch, None, get_arm("full"), _Backend(), _payload(), [_body()],
        str(log_path),
    )

    expected_first = {
        "limit": 1,
        "consumed_before": 0,
        "consumed_after": 1,
        "attempt_indices": [1],
    }
    expected_second = {
        "limit": 1,
        "consumed_before": 1,
        "consumed_after": 1,
        "attempt_indices": [],
    }
    assert len(first_posts) == 1 and first_sent[0][0] == 200
    assert first_sent[0][1]["c2kv_proxy"]["generation_budget"] == expected_first
    assert second_posts == [] and second_sent[0][0] == 502
    assert second_sent[0][1]["generation_budget"] == expected_second
    assert "generation attempt budget exhausted" in second_sent[0][1]["error"]
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [row["generation_budget"] for row in rows] == [expected_first, expected_second]
    assert rows[1]["status"] == "generation_budget_exhausted"


def test_extraction_budget_snapshot_is_frozen_before_success_response(monkeypatch, tmp_path):
    budget = proxy.ExtractionBudget(2)
    monkeypatch.setattr(proxy, "EXTRACTION_BUDGET", budget)

    class ReservingRuntime(_Runtime):
        def apply(self, *args, **kwargs):
            budget.reserve()
            return super().apply(*args, **kwargs)

    def next_request_reserves_after_response(_code, _body):
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(budget.reserve).result() == 2

    log_path = tmp_path / "extraction_budget.jsonl"
    sent, _ = _drive(
        monkeypatch, ReservingRuntime(), get_arm("full"), _Backend(),
        _payload(), [_body()], str(log_path),
        send_callback=next_request_reserves_after_response,
    )
    expected = {
        "limit": 2, "consumed_before": 0, "consumed_after": 1,
        "attempt_indices": [1],
    }
    assert sent[0][1]["c2kv_proxy"]["extraction_budget"] == expected
    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert row["extraction_budget"] == expected


def test_chat_network_failure_consumes_budget_and_disables_retry(monkeypatch):
    budget = proxy.GenerationBudget(1)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", budget)
    monkeypatch.setattr(proxy, "UPSTREAM", "http://127.0.0.1:1")
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", None)
    monkeypatch.setattr(proxy, "NO_UPSTREAM_RETRIES", False)
    attempts = []

    class Opener:
        def open(self, request, timeout):
            attempts.append((request, timeout))
            raise OSError("synthetic network failure")

    monkeypatch.setattr(proxy, "_OPENER", Opener())
    monkeypatch.setattr(proxy.time, "sleep", lambda seconds: pytest.fail(
        f"budgeted transport must not back off: {seconds}"))

    with pytest.raises(proxy.UpstreamError):
        proxy._post_json("/v1/chat/completions", {"messages": []}, 1, retries=2)
    assert len(attempts) == 1 and budget.consumed == 1
    with pytest.raises(proxy.GenerationBudgetExceeded):
        proxy._post_json("/v1/chat/completions", {"messages": []}, 1, retries=2)
    assert len(attempts) == 1 and budget.consumed == 1


def test_extraction_transport_does_not_consume_generation_budget(monkeypatch):
    budget = proxy.GenerationBudget(1)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", budget)
    monkeypatch.setattr(proxy, "UPSTREAM", "http://127.0.0.1:1")
    opened = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{}'

    class Opener:
        def open(self, request, timeout):
            opened.append(request.full_url)
            return Response()

    monkeypatch.setattr(proxy, "_OPENER", Opener())
    proxy._post_json("/v1/c2kv/extract", {"text": "history"}, 1)
    assert budget.consumed == 0
    proxy._post_json("/v1/chat/completions", {"messages": []}, 1)
    assert budget.consumed == 1
    assert len(opened) == 2


def test_request_trace_logs_empty_full_and_resets_scope(monkeypatch, tmp_path):
    log_path = tmp_path / "requests.jsonl"
    sent, posts = _drive(monkeypatch, _Runtime(), get_arm("full"), _Backend(),
                         _payload(), [_body()], str(log_path))
    assert sent[0][0] == 200 and len(posts) == 1
    row = json.loads(log_path.read_text())
    assert row["extraction_telemetry"]["events"] == []
    assert proxy.current_extraction_trace() is None


def test_request_trace_keeps_failed_extraction_before_assembly_returns(monkeypatch, tmp_path):
    class Backend(_Backend):
        def extract(self, *args, **kwargs):
            raise proxy.BackendError("extract_failed", "no extraction response")

    payload = _payload(messages=[
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"}])
    log_path = tmp_path / "requests.jsonl"
    sent, posts = _drive(monkeypatch, _Runtime(mode="protect"), get_arm("c2kv4"),
                         Backend(), payload, [], str(log_path))
    assert sent[0][0] == 502 and posts == []
    row = json.loads(log_path.read_text())
    assert row["status"] == "extract_failed"
    event, = row["extraction_telemetry"]["events"]
    assert event["producer_called"] and not event["client_cache_hit"]
    assert event["error_type"] == "BackendError"
    assert event["source_indices"] == [0, 1]
    assert proxy.current_extraction_trace() is None


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


def test_missing_backend_geometry_stops_the_runtime(monkeypatch):
    runtime = _Runtime(expected_bytes=147456)
    sent, _ = _drive(
        monkeypatch, runtime, get_arm("full"), _Backend(None), _payload(), [_body()])
    assert sent[0][0] == 502
    assert "omitted bytes_per_kv_token" in sent[0][1]["error"]
    with pytest.raises(proxy.MemoryRuntimeError, match="omitted bytes_per_kv_token"):
        proxy._check_memory_runtime_fatal()


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
