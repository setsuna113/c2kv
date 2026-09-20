"""Budget admission and every-retrieval wire guard across the actual proxy seam."""
import copy
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hiagent_budget
import proxy
import textarms
from arms import get_arm
from backends.sglang import SglangBackend
from bfcl_completion import completion_kind


def retrieval(ids):
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "tool_calls": [{"id": "meta", "type": "function",
        "function": {"name": textarms.HIAGENT_RETRIEVE_TOOL_NAME,
                     "arguments": json.dumps({"subgoal_ids": ids})}}]}}],
        "usage": {"prompt_tokens": 71, "completion_tokens": 9}}


def test_retrieval_denied_then_actor_continues_and_failed_trial_is_metered(monkeypatch):
    arm = get_arm("hiagent_full_b768")
    calls = []
    final = {"choices": [{"message": {"content": "Continue without retrieval."}}]}

    def apply(payload, arm, conv, ids=None, retrieval_feedback=None):
        calls.append((ids, retrieval_feedback))
        if ids == [1]:
            raise hiagent_budget.RetrievalBudgetExceeded("complete trajectory too large", {
                "budget": {"limit": 768, "history_after": 900, "passed": False},
                "n_compressor_calls": 1,
                "compressor_usage": {"calls": 1, "prompt_tokens": 300,
                                     "completion_tokens": 20, "wall_sec": 0.1}})
        assert ids == [] and "budget_unavailable" in retrieval_feedback
        return {"messages": [{"role": "user", "content": retrieval_feedback}]}, {
            "budget": {"limit": 768, "history_after": 80, "passed": True},
            "n_compressor_calls": 0, "compressor_usage": {}, "retrieved_subgoals": []}

    monkeypatch.setattr(proxy, "_apply_text_arm", apply)
    stats = {"n_compressor_calls": 0, "compressor_usage": {}}
    result = proxy._hiagent_retrieval_loop({}, arm, "episode", retrieval([1]), stats,
                                          lambda staged: final)
    assert result is final
    assert stats["retrieval_budget_denials"][0]["requested_subgoals"] == [1]
    assert stats["retrieved_subgoals"] == []
    assert stats["compressor_usage"]["prompt_tokens"] == 300
    assert stats["retrieval_usage"] == {"calls": 1, "prompt_tokens": 71, "completion_tokens": 9}
    assert len(calls) == 2


def test_nonexistent_completed_subgoal_gets_bounded_internal_feedback(monkeypatch):
    arm = get_arm("hiagent_full_b768")
    calls = []
    final = {"choices": [{"message": {"content": "I will continue the task."}}]}

    def apply(payload, arm, conv, ids=None, retrieval_feedback=None):
        calls.append((ids, retrieval_feedback))
        assert ids == ([1] if retrieval_feedback is None else [])
        return {"messages": [{"role": "user", "content": retrieval_feedback or "task"}]}, {
            "retrieved_subgoals": [],
            "invalid_retrieval_subgoals": [1] if retrieval_feedback is None else [],
            "n_compressor_calls": 0, "compressor_usage": {},
        }

    monkeypatch.setattr(proxy, "_apply_text_arm", apply)
    stats = {"n_compressor_calls": 0, "compressor_usage": {}}
    response = proxy._hiagent_retrieval_loop(
        {}, arm, "episode", retrieval([1]), stats, lambda staged:
        final if "subgoal_unavailable" in staged["messages"][-1]["content"]
        else pytest.fail("missing internal feedback"))
    assert response is final
    assert calls == [([1], None), ([], hiagent_budget.INVALID_SUBGOAL_FEEDBACK)]
    assert stats["invalid_retrieval_attempts"] == [{
        "requested_subgoals": [1], "invalid_subgoals": [1]}]
    assert stats["retrieval_usage"]["calls"] == 1


def test_token_preflight_includes_internal_tool_and_original_is_unchanged(monkeypatch):
    sent = []
    def post(path, payload, timeout):
        sent.append((path, payload))
        return {"success": True, "server_tokenized": True, "history_tokens": 8,
                "prompt_tokens": 40, "history_start": 1, "history_end": 9}
    monkeypatch.setattr(proxy, "BACKEND", SglangBackend(post))
    monkeypatch.setattr(proxy, "BENCHMARK", "bfcl")
    original = {"model": "model", "messages": [
        {"role": "user", "content": "task"}, {"role": "assistant", "content": "old reply"},
        {"role": "user", "content": "continue"}],
        "tools": [{"type": "function", "function": {"name": "inspect", "parameters": {}}}]}
    saved = copy.deepcopy(original)
    staged, stats = proxy._apply_text_arm(original, get_arm("hiagent_full_b768"), "episode")
    assert original == saved
    assert stats["budget"]["passed"]
    assert sent and all(path == "/v1/c2kv/chat_budget" for path, _ in sent)
    assert all(wire["tools"][-1]["function"]["name"] == "hiagent_retrieve" for _, wire in sent)
    assert staged["tools"][-1]["function"]["name"] == "hiagent_retrieve"


def test_handler_rebuilds_boundary_and_guards_each_retrieval(monkeypatch, tmp_path):
    arm = get_arm("hiagent_full_b768")
    monkeypatch.setattr(proxy, "ARM", arm)
    monkeypatch.setattr(proxy, "BENCHMARK", "bfcl")
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.setattr(proxy, "PREFIX_LOG_PATH", None)
    monkeypatch.setattr(proxy, "TELEMETRY_LOG_PATH", None)
    monkeypatch.setattr(proxy.STATE, "recover", None)
    monkeypatch.setattr(proxy.STATE, "reference_log_path", None)
    wires, checks = [], []

    def post(path, payload, timeout):
        span = payload["c2kv_kv_memory_hint"]["paper_measurement"]
        if path == "/v1/c2kv/chat_budget":
            checks.append((span["history_start_message_count"], span["history_message_count"]))
            return {"success": True, "server_tokenized": True, "history_tokens": 30,
                    "prompt_tokens": 100, "history_start": 20, "history_end": 50}
        wires.append(copy.deepcopy(payload))
        if len(wires) == 1:
            return retrieval([1])
        return {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": "Finished."}}], "usage": {"prompt_tokens": 100}}

    monkeypatch.setattr(proxy, "BACKEND", SglangBackend(post))
    monkeypatch.setattr(proxy, "_post_json", post)

    def apply(payload, arm, conv, ids=None, **kwargs):
        history = [{"role": "user", "content": "summary"}]
        if ids:
            history = [{"role": "user", "content": "full start"},
                       {"role": "assistant", "content": "old action"},
                       {"role": "user", "content": "old result"}]
        out = dict(payload, messages=[{"role": "system", "content": "system"}] + history + [
            {"role": "user", "content": "current"}])
        return out, {"history_indices": list(range(1, len(history) + 1)),
                     "budget": {"limit": 768, "history_after": 30, "passed": True},
                     "n_compressor_calls": 0, "compressor_usage": {}}

    monkeypatch.setattr(proxy, "_apply_text_arm", apply)
    body = json.dumps({"model": "model", "messages": [{"role": "user", "content": "task"}]}).encode()
    handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    handler.path = "/v1/chat/completions"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    response = {}
    handler._send_json = lambda code, obj: response.update(code=code, obj=obj)
    handler.do_POST()
    assert response["code"] == 200, response
    assert checks == [(1, 2), (1, 4)]
    assert len(wires) == 2
    row = json.loads((tmp_path / "requests.jsonl").read_text())
    calls = row["textarm"]["actor_budget_calls"]
    assert [r["phase"] for r in calls] == ["generation", "hiagent_retrieval_generation"]
    assert [r["actor_payload_sha256"] for r in calls] == [proxy.canonical_sha256(w) for w in wires]


def test_invalid_retrieval_feedback_and_evidence_reach_request_log(monkeypatch, tmp_path):
    arm = get_arm("hiagent_full")
    monkeypatch.setattr(proxy, "ARM", arm)
    monkeypatch.setattr(proxy, "BENCHMARK", "toolsandbox")
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(tmp_path / "requests.jsonl"))
    monkeypatch.setattr(proxy, "PREFIX_LOG_PATH", None)
    monkeypatch.setattr(proxy, "TELEMETRY_LOG_PATH", None)
    monkeypatch.setattr(proxy.STATE, "recover", None)
    monkeypatch.setattr(proxy.STATE, "reference_log_path", None)
    sent = []

    def post(path, payload, timeout):
        sent.append(copy.deepcopy(payload))
        if len(sent) == 1:
            return retrieval([1])
        return {"choices": [{"finish_reason": "stop", "message": {
            "role": "assistant", "content": "I can proceed."}}], "usage": {"prompt_tokens": 100}}

    def apply(payload, arm, conv, ids=None, retrieval_feedback=None):
        invalid = ids == [1]
        messages = [{"role": "user", "content": "task"}]
        if retrieval_feedback:
            messages.append({"role": "user", "content": retrieval_feedback})
        return dict(payload, messages=messages), {
            "retrieved_subgoals": [], "invalid_retrieval_subgoals": [1] if invalid else [],
            "n_compressor_calls": 0, "compressor_usage": {},
        }

    monkeypatch.setattr(proxy, "BACKEND", SglangBackend(post))
    monkeypatch.setattr(proxy, "_post_json", post)
    monkeypatch.setattr(proxy, "_apply_text_arm", apply)
    body = json.dumps({"model": "model", "messages": [{"role": "user", "content": "task"}]}).encode()
    handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    handler.path = "/v1/chat/completions"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    response = {}
    handler._send_json = lambda code, obj: response.update(code=code, obj=obj)
    handler.do_POST()
    assert response["code"] == 200
    assert len(sent) == 2 and "subgoal_unavailable" in json.dumps(sent[1]["messages"])
    assert response["obj"]["choices"][0]["message"]["content"] == "I can proceed."
    row = json.loads((tmp_path / "requests.jsonl").read_text())
    assert row["textarm"]["invalid_retrieval_attempts"] == [{
        "requested_subgoals": [1], "invalid_subgoals": [1]}]
    assert row["textarm"]["retrieval_usage"]["calls"] == 1


def test_fixed_floor_is_terminal_but_retrieval_denial_is_not():
    assert completion_kind({"result": [], "traceback": json.dumps({"error": {
        "code": "hiagent_history_budget_exceeded"}})}) == "hiagent_history_budget_exceeded"
    assert completion_kind({"result": [], "traceback": "budget_unavailable"}) == "incomplete"
    assert completion_kind({"result": [], "traceback": "chat_budget_tokenize_failed"}) == "incomplete"


def test_hiagent_budget_requires_live_sglang_renderer():
    from capabilities import preflight
    unsupported = preflight("bfcl", "hiagent_full_b768", "hfserver")
    assert "hiagent_budget_sglang_backend" in {item.code for item in unsupported.errors}
