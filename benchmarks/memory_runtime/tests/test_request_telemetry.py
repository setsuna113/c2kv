import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))
import proxy


def test_capture_allowlist_preserves_exact_content_without_transport_fields():
    payload = {
        "model": "test-model", "messages": [{"role": "user", "content": "query"}],
        "tools": [], "temperature": 0.001, "seed": 0,
        "api_key": "not-captured", "headers": {"Authorization": "not-captured"},
    }
    captured = proxy._captured_request_view(payload)
    payload["messages"][0]["content"] = "mutated"
    assert captured["messages"][0]["content"] == "query"
    assert captured["sampling"] == {"temperature": 0.001, "seed": 0}
    assert "api_key" not in captured and "headers" not in captured


def test_log_keeps_forwarded_sampling_and_opt_in_request_view(monkeypatch, tmp_path):
    path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(path))
    monkeypatch.setattr(proxy, "CAPTURE_REQUEST_VIEWS", True)
    handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    handler.eval_context = {"task_id": "task1"}
    payload = {"messages": [{"role": "user", "content": "query"}],
               "temperature": 0.001, "seed": 0, "max_completion_tokens": 512}
    handler.forwarded_sampling = [proxy._sampling_fields(payload)]
    handler.forwarded_request_views = [proxy._captured_request_view(payload)]
    handler._log_request(payload, {"content": "answer", "tool_calls": [], "usage": {}}, {})
    row = json.loads(path.read_text())
    assert row["sampling_request"] == row["sampling_forwarded"][0]
    assert row["request_view"]["messages"] == payload["messages"]
    assert row["response_view"]["content"] == "answer"


def test_sampling_log_does_not_invent_missing_seed():
    assert proxy._sampling_fields({"temperature": 0}) == {"temperature": 0}
