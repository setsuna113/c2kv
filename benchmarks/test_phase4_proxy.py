"""Proxy seams for the optional Phase4 final-response commit hook."""
from __future__ import annotations

import copy
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import proxy
from arms import get_arm


def _body(call_id):
    calls = [{
        "id": call_id,
        "type": "function",
        "function": {"name": "read", "arguments": json.dumps({"path": call_id})},
    }]
    return {
        "choices": [{
            "message": {"content": f"response:{call_id}", "tool_calls": calls},
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
    }


def _payload():
    return {
        "model": "fake",
        "messages": [{"role": "user", "content": "read the final path"}],
        "tools": [],
        "c2kv_eval_context": {
            "task_id": "phase4-task", "run_id": "phase4-run",
            "attempt_id": 0, "decision_id": "phase4-decision",
        },
    }


class _Backend:
    name = "fake"
    wants_request_context = False
    needs_repair_plan = False

    def prepare_chat(self, payload, arm, plan):
        return dict(payload)

    def normalize_response(self, data):
        choice = data["choices"][0]
        return {
            "content": choice["message"].get("content"),
            "tool_calls": choice["message"].get("tool_calls"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data["usage"],
            "cost": {"bytes_per_kv_token": 147456},
        }


class _LegacyExactRuntime:
    """A minimal existing exact adapter with no final-response hook."""

    mode = "capacity_exact_persistent"
    supports_exact_recovery = True

    def __init__(self, *, regenerate):
        self.regenerate = regenerate
        self.prepared = object()
        self.drafts = []

    def prepare_exact(self, messages, assembled, counts, eval_context, tools, *, render_compressed):
        updated = copy.deepcopy(counts)
        updated["memory_runtime"] = {
            "mode": self.mode,
            "bytes_per_kv_token": 147456,
            "byte_geometry_verified_by_backend": False,
        }
        return copy.deepcopy(assembled), updated, self.prepared

    def reconsider(self, prepared, draft_tool_calls):
        assert prepared is self.prepared
        self.drafts.append(copy.deepcopy(draft_tool_calls))
        messages = [{"role": "user", "content": "read the final path"}]
        return {
            "regenerate": self.regenerate,
            "messages": messages,
            "counts": {
                "gist_tokens": 0,
                "original_tokens": 1,
                "history_packed_original_tokens": 0,
                "history_dropped_original_tokens": 0,
                "history_packed_candidate_doc_count": 0,
                "history_retained_fraction": 1.0,
                "n_gist_messages": 0,
                "memory_runtime": {
                    "mode": self.mode,
                    "bytes_per_kv_token": 147456,
                    "byte_geometry_verified_by_backend": False,
                },
            },
        }


class _Phase4ExactRuntime(_LegacyExactRuntime):
    def __init__(self, *, regenerate):
        super().__init__(regenerate=regenerate)
        self.final_calls = []

    def commit_final(self, prepared, tool_calls):
        assert prepared is self.prepared
        self.final_calls.append(copy.deepcopy(tool_calls))
        return {
            "version": "a-phase4-dev-policy-v1",
            "final_call_ids": [call["id"] for call in tool_calls or []],
            "extra_generation": 0,
        }


@pytest.fixture(autouse=True)
def _reset_proxy(monkeypatch):
    monkeypatch.setattr(proxy, "STATE", proxy.ProxyState())
    monkeypatch.setattr(proxy, "CACHE", proxy.ExtractCache())
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", None)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_BYTES_PER_KV_TOKEN", None)
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME_FATAL_ERROR", None)
    monkeypatch.setattr(proxy, "NO_UPSTREAM_RETRIES", False)
    monkeypatch.setattr(proxy, "CAPTURE_REQUEST_VIEWS", False)
    monkeypatch.setattr(proxy, "GENERATION_BUDGET", proxy.GenerationBudget(None))
    monkeypatch.setattr(proxy, "EXTRACTION_BUDGET", proxy.ExtractionBudget(None))
    monkeypatch.setattr(proxy, "ATTEMPT_JOURNAL", None)
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", "")


def _drive(monkeypatch, runtime, bodies, request_log_path=""):
    monkeypatch.setattr(proxy, "MEMORY_RUNTIME", runtime)
    monkeypatch.setattr(proxy, "ARM", get_arm("c2kv4"))
    monkeypatch.setattr(proxy, "BACKEND", _Backend())
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(request_log_path))
    posts = []

    def fake_post(path, body, timeout, retries=2):
        proxy._reserve_generation_attempt(path)
        posts.append({"path": path, "body": body, "retries": retries})
        return bodies[len(posts) - 1]

    monkeypatch.setattr(proxy, "_post_json", fake_post)
    raw = json.dumps(_payload()).encode("utf-8")
    handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    handler.rfile = io.BytesIO(raw)
    handler.headers = {"Content-Length": str(len(raw))}
    handler.path = "/v1/chat/completions"
    sent = []
    handler._send_json = lambda code, body: sent.append((code, body))
    handler.do_POST()
    return sent, posts


def test_runtime_factory_selects_phase4_only_when_policy_field_is_present(tmp_path, monkeypatch):
    phase_config = tmp_path / "phase4.json"
    phase_config.write_text(json.dumps({"phase4_policy": {"trigger": "off"}}), encoding="utf-8")
    legacy_config = tmp_path / "legacy.json"
    legacy_config.write_text(json.dumps({"mode": "capacity_exact_persistent"}), encoding="utf-8")
    calls = []

    class PhaseFactory:
        @classmethod
        def from_config(cls, config, tokenizer):
            calls.append(("phase4", config, tokenizer))
            return "phase4-runtime"

    class LegacyFactory:
        @classmethod
        def from_config(cls, config, tokenizer):
            calls.append(("legacy", config, tokenizer))
            return "legacy-runtime"

    modules = {
        "benchmarks.memory_runtime.phase4_policy": SimpleNamespace(Phase4RuntimeAdapter=PhaseFactory),
        "benchmarks.memory_runtime.adapter": SimpleNamespace(RuntimeAdapter=LegacyFactory),
    }
    monkeypatch.setattr(proxy.importlib, "import_module", lambda name: modules[name])

    assert proxy._load_memory_runtime(str(phase_config), "phase-tokenizer") == "phase4-runtime"
    assert proxy._load_memory_runtime(str(legacy_config), "legacy-tokenizer") == "legacy-runtime"
    assert calls == [
        ("phase4", str(phase_config), "phase-tokenizer"),
        ("legacy", str(legacy_config), "legacy-tokenizer"),
    ]


def test_phase4_commit_uses_the_only_final_normalized_response_before_log_and_send(
        tmp_path, monkeypatch):
    runtime = _Phase4ExactRuntime(regenerate=False)
    log_path = tmp_path / "one-call.jsonl"
    sent, posts = _drive(monkeypatch, runtime, [_body("only-final")], log_path)

    assert len(posts) == 1 and sent[0][0] == 200
    final_calls = sent[0][1]["choices"][0]["message"]["tool_calls"]
    assert runtime.drafts == [final_calls]
    assert runtime.final_calls == [final_calls]
    metadata = sent[0][1]["c2kv_proxy"]["memory_runtime"]["phase4_final"]
    assert metadata == {
        "version": "a-phase4-dev-policy-v1",
        "final_call_ids": ["only-final"],
        "extra_generation": 0,
    }
    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert row["memory_runtime"]["phase4_final"] == metadata


def test_phase4_commit_uses_regenerated_final_not_discarded_draft(tmp_path, monkeypatch):
    runtime = _Phase4ExactRuntime(regenerate=True)
    log_path = tmp_path / "two-call.jsonl"
    sent, posts = _drive(
        monkeypatch, runtime, [_body("discarded-draft"), _body("regenerated-final")], log_path)

    assert len(posts) == 2 and sent[0][0] == 200
    assert runtime.drafts[0][0]["id"] == "discarded-draft"
    assert runtime.final_calls[0][0]["id"] == "regenerated-final"
    metadata = sent[0][1]["c2kv_proxy"]["memory_runtime"]["phase4_final"]
    assert metadata["final_call_ids"] == ["regenerated-final"]
    row = json.loads(log_path.read_text(encoding="utf-8"))
    assert row["memory_runtime"]["phase4_final"] == metadata


def test_existing_exact_runtime_without_commit_hook_stays_compatible(monkeypatch):
    runtime = _LegacyExactRuntime(regenerate=False)
    sent, posts = _drive(monkeypatch, runtime, [_body("legacy-final")])

    assert len(posts) == 1 and sent[0][0] == 200
    assert runtime.drafts[0][0]["id"] == "legacy-final"
    assert "phase4_final" not in sent[0][1]["c2kv_proxy"]["memory_runtime"]
