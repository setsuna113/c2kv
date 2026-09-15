"""Synthetic seam tests for the exact-recovery proxy generation path."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy
from arms import get_arm
from memory_runtime.tests.test_exact_adapter import (
    _adapter,
    _compressed_fixture,
    _context,
    _count,
    _draft,
    _full,
    _source,
)
from test_memory_runtime_proxy import _Backend, _counts, _drive, _payload


DISCARDED_CONTENT = "SYNTHETIC_DISCARDED_DRAFT_CONTENT"
DISCARDED_CALL_ID = "SYNTHETIC_DISCARDED_DRAFT_TOOL_CALL"
FINAL_CONTENT = "SYNTHETIC_FINAL_CONTENT"


def _synthetic_body(content, call_id, usage):
    tool_calls = None if call_id is None else [{
        "id": call_id,
        "index": 0,
        "type": "function",
        "function": {
            "name": "synthetic_tool",
            "arguments": '{"file_name":"synthetic.txt"}',
        },
    }]
    return {
        "choices": [{
            "message": {"content": content, "tool_calls": tool_calls},
            "finish_reason": "tool_calls" if tool_calls else "stop",
        }],
        "usage": usage,
    }


@pytest.fixture
def synthetic_bodies():
    return (
        _synthetic_body(
            DISCARDED_CONTENT,
            DISCARDED_CALL_ID,
            {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        ),
        _synthetic_body(
            FINAL_CONTENT,
            "SYNTHETIC_FINAL_TOOL_CALL",
            {"prompt_tokens": 20, "completion_tokens": 3, "total_tokens": 23},
        ),
    )


class _ExactRuntime:
    """Request-local fake that isolates the proxy's two-generation seam."""

    mode = "capacity_exact_once"
    supports_exact_recovery = True

    def __init__(self, decision_status="gap", expected_bytes=147456):
        self.decision_status = decision_status
        self.expected_bytes = expected_bytes
        self.prepare_calls = []
        self.reconsider_calls = []
        self.prepared = object()
        self.messages = None
        self.counts = None

    def prepare_exact(
        self,
        messages,
        assembled,
        counts,
        eval_context,
        tools,
        *,
        render_compressed,
    ):
        self.prepare_calls.append({
            "messages": copy.deepcopy(messages),
            "eval_context": copy.deepcopy(eval_context),
            "tools": copy.deepcopy(tools),
            "render_compressed": render_compressed,
        })
        self.messages = copy.deepcopy(assembled)
        # Reuse the established proxy-test count contract, then preserve values
        # produced by the real Full renderer at this seam.
        self.counts = _counts()
        self.counts.update(copy.deepcopy(counts))
        self.counts["memory_runtime"] = {
            "mode": self.mode,
            "bytes_per_kv_token": self.expected_bytes,
            "byte_geometry_verified_by_backend": False,
            "exact_recovery": {
                "status": "pending_synthetic_draft",
                "synthetic": True,
            },
        }
        return self.messages, self.counts, self.prepared

    def reconsider(self, prepared, draft_tool_calls):
        assert prepared is self.prepared
        self.reconsider_calls.append(copy.deepcopy(draft_tool_calls))
        regenerate = self.decision_status == "gap"
        reason = {
            "gap": "missing_unique_complete_source",
            "no_op": "all_bindings_visible",
            "abstain": "missing_source",
        }[self.decision_status]
        self.counts["memory_runtime"]["exact_recovery"] = {
            "status": self.decision_status,
            "reason": reason,
            "upgrade_count": int(regenerate),
            "regeneration_allowed": regenerate,
            "synthetic": True,
        }
        if regenerate:
            self.messages = copy.deepcopy(self.messages)
            self.messages[-1]["content"] = (
                str(self.messages[-1].get("content") or "")
                + " [SYNTHETIC_EXACT_UPGRADE]"
            )
        return {
            "regenerate": regenerate,
            "messages": self.messages,
            "counts": self.counts,
            "decision": self.counts["memory_runtime"]["exact_recovery"],
        }


@pytest.fixture(autouse=True)
def _reset_exact_proxy_state(monkeypatch):
    monkeypatch.setattr(proxy, "STATE", proxy.ProxyState())
    monkeypatch.setattr(proxy, "CACHE", proxy.ExtractCache())
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", None)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_BYTES_PER_KV_TOKEN", None)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_FATAL_ERROR", None)
    monkeypatch.setattr(proxy, "NO_UPSTREAM_RETRIES", False)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", proxy.GenerationBudget(None))
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", "")
    monkeypatch.setattr(proxy, "CAPTURE_REQUEST_VIEWS", True)


def _read_log(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    return rows[0]


def test_synthetic_gap_returns_only_regenerated_action_and_accounts_both_calls(
    monkeypatch, tmp_path, synthetic_bodies
):
    runtime = _ExactRuntime("gap")
    draft, final = synthetic_bodies
    expected_final_choices = copy.deepcopy(final["choices"])
    draft_tool_calls = copy.deepcopy(draft["choices"][0]["message"]["tool_calls"])
    log_path = tmp_path / "synthetic_exact_success.jsonl"

    sent, posts = _drive(
        monkeypatch,
        runtime,
        get_arm("c2kv4"),
        _Backend(),
        _payload(),
        [draft, final],
        str(log_path),
    )

    assert len(posts) == 2
    assert [post["retries"] for post in posts] == [0, 0]
    assert posts[0]["body"]["messages"] != posts[1]["body"]["messages"]
    assert runtime.reconsider_calls == [draft_tool_calls]
    assert len(sent) == 1 and sent[0][0] == 200
    response = sent[0][1]
    assert response["choices"] == expected_final_choices
    response_json = json.dumps(response, ensure_ascii=False, sort_keys=True)
    assert DISCARDED_CONTENT not in response_json
    assert DISCARDED_CALL_ID not in response_json
    assert response["c2kv_proxy"]["generation_attempts"] == 2
    assert response["c2kv_proxy"]["generation_completed"] == 2
    assert response["c2kv_proxy"]["generation_usage_total"] == {
        "completion_tokens": 5,
        "prompt_tokens": 30,
        "total_tokens": 35,
    }

    row = _read_log(log_path)
    assert row["status"] == "ok"
    assert row["usage"] == {"prompt_tokens": 20, "completion_tokens": 3,
                            "total_tokens": 23}
    assert row["generation_usage_total"] == {
        "completion_tokens": 5,
        "prompt_tokens": 30,
        "total_tokens": 35,
    }
    draft_record, final_record = row["generation_trace"]
    assert draft_record["phase"] == "draft" and draft_record["discarded"] is True
    assert final_record["phase"] == "regeneration" and final_record["discarded"] is False
    assert draft_record["response_view"] == {
        "content": DISCARDED_CONTENT,
        "tool_calls": draft_tool_calls,
    }
    assert final_record["response_view"]["content"] == FINAL_CONTENT
    assert row["response_view"] == final_record["response_view"]
    assert all(record["backend_verified"] for record in row["generation_trace"])


@pytest.mark.parametrize("mode", ["capacity_exact_once", "ac_acquire_for_next"])
def test_synthetic_actual_adapter_gap_regenerates_once_without_draft_leak(
    monkeypatch, tmp_path, synthetic_bodies, mode
):
    runtime = _adapter(mode)
    source = _source()

    class ActualAdapterBackend(_Backend):
        def __init__(self):
            super().__init__(bytes_per_kv_token=1)
            self.pending_prompt_tokens = []
            self.measured_prompt_tokens = []

        def prepare_chat(self, payload, arm, plan):
            prepared = super().prepare_chat(payload, arm, plan)
            raw_messages = [
                message
                for message in prepared["messages"]
                if not message.get("c2kv_key_hash")
            ]
            measured = _count(raw_messages, prepared.get("tools"))
            self.pending_prompt_tokens.append(measured)
            self.measured_prompt_tokens.append(measured)
            return prepared

        def normalize_response(self, data):
            normalized = super().normalize_response(data)
            prompt_tokens = self.pending_prompt_tokens.pop(0)
            usage = dict(normalized["usage"])
            usage["prompt_tokens"] = prompt_tokens
            usage["total_tokens"] = prompt_tokens + usage["completion_tokens"]
            normalized["usage"] = usage
            normalized["cost"]["c2kv_tools_dump"] = "full"
            return normalized

    def assemble(messages, arm):
        assert messages == source
        if arm.name == "full":
            rendered, fixture_counts = _full(messages)
        else:
            assert arm.name == "c2kv4"
            rendered, fixture_counts = _compressed_fixture()
        counts = _counts()
        counts.update(copy.deepcopy(fixture_counts))
        return copy.deepcopy(rendered), counts

    monkeypatch.setattr(proxy, "_assemble", assemble)
    draft, final = synthetic_bodies
    draft["choices"][0]["message"]["tool_calls"] = _draft()
    expected_final_choices = copy.deepcopy(final["choices"])
    log_path = tmp_path / "synthetic_actual_adapter_success.jsonl"
    payload = _payload(messages=source, c2kv_eval_context=_context("d1"))
    backend = ActualAdapterBackend()

    sent, posts = _drive(
        monkeypatch,
        runtime,
        get_arm("c2kv4"),
        backend,
        payload,
        [draft, final],
        str(log_path),
    )

    if mode == "ac_acquire_for_next":
        assert len(posts) == 1 and posts[0]["retries"] == 0
        assert len(sent) == 1 and sent[0][0] == 200
        assert sent[0][1]["choices"] == draft["choices"]
        row = _read_log(log_path)
        assert len(row["generation_trace"]) == 1
        consumed = row["generation_trace"][0]
        assert consumed["discarded"] is False
        assert consumed["memory_runtime"]["selected_event_ids"] == ["synthetic:m3"]
        assert row["memory_runtime"]["selected_event_ids"] == ["synthetic:m3"]
        assert row["memory_runtime"]["exact_recovery"]["deferred_lease_acquisition_count"] == 1
        assert row["generation_usage_total"]["completion_tokens"] == 2
        assert row["generation_usage_total"]["prompt_tokens"] == backend.measured_prompt_tokens[0]
        return
    assert len(posts) == 2
    assert [post["retries"] for post in posts] == [0, 0]
    assert backend.measured_prompt_tokens == [500, 600]
    assert len(sent) == 1 and sent[0][0] == 200
    assert sent[0][1]["choices"] == expected_final_choices
    response_json = json.dumps(sent[0][1], ensure_ascii=False, sort_keys=True)
    assert DISCARDED_CONTENT not in response_json
    assert "unsubmitted-draft" not in response_json
    exact = sent[0][1]["c2kv_proxy"]["memory_runtime"]["exact_recovery"]
    assert exact["status"] == "gap"
    assert exact["candidate_event_id"] == "synthetic:m0"
    assert exact["upgrade_count"] == 1
    assert exact["regeneration_allowed"] is True

    row = _read_log(log_path)
    assert row["generation_usage_total"] == {
        "completion_tokens": 5,
        "prompt_tokens": sum(backend.measured_prompt_tokens),
        "total_tokens": sum(backend.measured_prompt_tokens) + 5,
    }
    draft_record, final_record = row["generation_trace"]
    assert draft_record["discarded"] is True
    assert draft_record["response_view"]["tool_calls"] == _draft()
    assert final_record["response_view"]["content"] == FINAL_CONTENT
    assert draft_record["memory_runtime"]["selected_event_ids"] == ["synthetic:m3"]
    assert final_record["memory_runtime"]["selected_event_ids"] == [
        "synthetic:m0",
        "synthetic:m3",
    ]


@pytest.mark.parametrize(
    ("decision_status", "reason"),
    [("no_op", "all_bindings_visible"), ("abstain", "missing_source")],
)
def test_synthetic_no_op_and_abstain_use_one_generation(
    monkeypatch, tmp_path, decision_status, reason
):
    runtime = _ExactRuntime(decision_status)
    only = _synthetic_body(
        f"SYNTHETIC_{decision_status.upper()}_FINAL",
        f"SYNTHETIC_{decision_status.upper()}_CALL",
        {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    )
    log_path = tmp_path / f"synthetic_{decision_status}.jsonl"

    sent, posts = _drive(
        monkeypatch,
        runtime,
        get_arm("c2kv4"),
        _Backend(),
        _payload(),
        [only],
        str(log_path),
    )

    assert len(posts) == 1 and posts[0]["retries"] == 0
    assert len(runtime.reconsider_calls) == 1
    assert len(sent) == 1 and sent[0][0] == 200
    assert sent[0][1]["choices"] == only["choices"]
    assert sent[0][1]["c2kv_proxy"]["generation_attempts"] == 1
    row = _read_log(log_path)
    assert len(row["generation_trace"]) == 1
    assert row["generation_trace"][0]["discarded"] is False
    assert row["memory_runtime"]["exact_recovery"]["status"] == decision_status
    assert row["memory_runtime"]["exact_recovery"]["reason"] == reason


def test_synthetic_first_geometry_failure_stops_before_reconsideration(
    monkeypatch, tmp_path, synthetic_bodies
):
    runtime = _ExactRuntime("gap", expected_bytes=8192)
    draft, _ = synthetic_bodies
    log_path = tmp_path / "synthetic_first_geometry_failure.jsonl"

    sent, posts = _drive(
        monkeypatch,
        runtime,
        get_arm("c2kv4"),
        _Backend(bytes_per_kv_token=147456),
        _payload(),
        [draft],
        str(log_path),
    )

    assert len(posts) == 1 and posts[0]["retries"] == 0
    assert runtime.reconsider_calls == []
    assert len(sent) == 1 and sent[0][0] == 502
    assert "bytes_per_kv_token disagrees" in sent[0][1]["error"]
    assert DISCARDED_CONTENT not in json.dumps(sent[0][1], ensure_ascii=False)
    row = _read_log(log_path)
    assert row["status"] == "memory_runtime_error"
    assert row["generation_attempts"] == 1
    assert row["generation_trace"][0]["backend_verified"] is False
    assert row["generation_trace"][0]["response_view"]["content"] == DISCARDED_CONTENT
    assert row["memory_runtime"]["byte_geometry_verified_by_backend"] is False


def test_synthetic_second_generation_failure_returns_only_502_and_keeps_draft_cost(
    monkeypatch, tmp_path, synthetic_bodies
):
    runtime = _ExactRuntime("gap")
    draft, _ = synthetic_bodies
    log_path = tmp_path / "synthetic_second_generation_failure.jsonl"

    sent, posts = _drive(
        monkeypatch,
        runtime,
        get_arm("c2kv4"),
        _Backend(),
        _payload(),
        [draft, proxy.UpstreamError(503, "SYNTHETIC_REGENERATION_FAILURE")],
        str(log_path),
    )

    assert len(posts) == 2
    assert [post["retries"] for post in posts] == [0, 0]
    assert len(runtime.reconsider_calls) == 1
    assert len(sent) == 1 and sent[0][0] == 502
    error_response = sent[0][1]
    assert set(error_response) == {"error"}
    response_json = json.dumps(error_response, ensure_ascii=False, sort_keys=True)
    assert DISCARDED_CONTENT not in response_json
    assert DISCARDED_CALL_ID not in response_json

    row = _read_log(log_path)
    assert row["status"] == "upstream_error"
    assert row["usage"] is None and row["response_view"] is None
    assert row["generation_attempts"] == 2
    assert row["generation_completed"] == 1
    assert len(row["sampling_forwarded"]) == 2
    assert len(row["forwarded_request_views"]) == 2
    assert len(row["extraction_telemetry"]["forwarded_requests"]) == 2
    assert row["generation_usage_total"] == {
        "completion_tokens": None,
        "prompt_tokens": None,
        "total_tokens": None,
    }
    draft_record, failed_record = row["generation_trace"]
    assert draft_record["discarded"] is True
    assert draft_record["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
    }
    assert draft_record["response_view"]["content"] == DISCARDED_CONTENT
    assert failed_record["phase"] == "regeneration"
    assert failed_record["status"] == "failed"
    assert failed_record["backend_verified"] is False
    assert failed_record["error_type"] == "UpstreamError"
    assert "usage" not in failed_record and "response_view" not in failed_record


def test_generation_budget_rejects_exact_regeneration_without_fake_trace(
    monkeypatch, tmp_path, synthetic_bodies
):
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", proxy.GenerationBudget(1))
    runtime = _ExactRuntime("gap")
    draft, unused_final = synthetic_bodies
    log_path = tmp_path / "budget_rejects_regeneration.jsonl"

    sent, posts = _drive(
        monkeypatch,
        runtime,
        get_arm("c2kv4"),
        _Backend(),
        _payload(),
        [draft, unused_final],
        str(log_path),
    )

    budget = {
        "limit": 1,
        "consumed_before": 0,
        "consumed_after": 1,
        "attempt_indices": [1],
    }
    assert len(posts) == 1
    assert len(sent) == 1 and sent[0][0] == 502
    assert sent[0][1]["generation_budget"] == budget
    assert DISCARDED_CONTENT not in json.dumps(sent[0][1], ensure_ascii=False)
    row = _read_log(log_path)
    assert row["status"] == "generation_budget_exhausted"
    assert row["generation_budget"] == budget
    assert row["generation_attempts"] == 1
    assert row["generation_completed"] == 1
    assert len(row["sampling_forwarded"]) == 1
    assert len(row["forwarded_request_views"]) == 1
    assert len(row["extraction_telemetry"]["forwarded_requests"]) == 1
    assert row["generation_usage_total"] == {
        "completion_tokens": 2,
        "prompt_tokens": 10,
        "total_tokens": 12,
    }
    only_record, = row["generation_trace"]
    assert only_record["phase"] == "draft"
    assert only_record["forwarded_request_index"] == 0
    assert only_record["status"] == "completed"
    assert only_record["discarded"] is True
