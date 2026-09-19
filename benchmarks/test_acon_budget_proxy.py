"""Integration boundaries that can otherwise bypass an ACON history budget."""
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acon_budget
import proxy
from arms import get_arm
from backends.sglang import SglangBackend
from bfcl_completion import completion_kind
from benchmarks.paper.runner import server_command, with_acon_budget, cells, DEFAULT_CONFIG, prepare


def test_budget_arm_identity_and_original_remain_distinct():
    assert get_arm("acon_hist_ut_co").text_history_budget_tokens is None
    assert get_arm("acon_hist_ut_co_b768").text_history_budget_tokens == 768
    for name in ("acon_hist_ut_co_b0", "acon_hist_ut_co_b0768"):
        with pytest.raises(ValueError):
            get_arm(name)


def test_harness_preflight_recognizes_budget_arm():
    from capabilities import preflight
    result = preflight("bfcl", "acon_hist_ut_co_b768", "sglang")
    assert not {"known_arm", "valid_arm", "acon_budget_sglang_backend"} & {
        item.code for item in result.errors}
    unsupported = preflight("bfcl", "acon_hist_ut_co_b768", "hfserver")
    assert "acon_budget_sglang_backend" in {item.code for item in unsupported.errors}


def test_budget_matrix_uses_renderer_and_original_cache_policy(tmp_path):
    original = json.loads(DEFAULT_CONFIG.read_text())
    config = with_acon_budget(original, 768)
    added = [cell for cell in cells(config) if cell["arm"] == "acon_hist_ut_co_b768"]
    assert {cell["benchmark"] for cell in added} == {"bfcl_base", "bfcl_long_context", "acebench_agent"}
    assert all(cell["history_budget_tokens"] == 768 for cell in added)
    command = server_command(config, Path("engine"), "acon_hist_ut_co_b768")
    assert command[2] == "benchmarks.paper.budget_server"
    assert "--disable-radix-cache" not in command
    assert server_command(config, Path("engine"), "acon_hist_ut_co")[2] == "sglang.launch_server"
    assert not any(row["arm"] == "acon_hist_ut_co_b768" for row in original["methods"])
    plan, profile = prepare(config, tmp_path / "output", tmp_path / "engine")
    assert profile.is_file()
    assert "history_budget_tokens" in (profile.parent / "matrix.csv").read_text().splitlines()[0]
    assert any(row["arm"] == "acon_hist_ut_co_b768" for row in plan)


def test_summary_is_history_even_without_an_assistant_in_output():
    messages = [
        {"role": "user", "content": "task\n<HISTORY_SUMMARY>saved</HISTORY_SUMMARY>"},
        {"role": "tool", "content": "latest observation"},
        {"role": "user", "content": "continue"},
    ]
    assembled, counts = proxy._assemble(messages, get_arm("acon_hist_ut_co_b768"))
    assert proxy._history_cutoff(assembled) == 0
    counts["acon_budget_history_boundary"] = proxy._acon_budget_boundary(messages, assembled, [0])
    assert proxy._paper_history_message_boundary(assembled, counts) == (1, 2)


def test_preflight_uses_assembled_tools_and_counts_real_payload(monkeypatch):
    acon_budget.reset_state()
    sent = []

    def post(path, payload, timeout):
        sent.append((path, payload))
        return {"success": True, "server_tokenized": True, "history_tokens": 3,
                "prompt_tokens": 10, "history_start": 2, "history_end": 5}

    monkeypatch.setattr(proxy, "BACKEND", SglangBackend(post))
    payload = {"model": "model", "c2kv_measurement_session_id": "episode",
               "messages": [{"role": "user", "content": "task"},
                            {"role": "assistant", "tool_calls": [{"type": "function", "function": {
                                "name": "inspect", "arguments": "{}"}}]},
                            {"role": "tool", "content": "result"}],
               "tools": [{"type": "function", "function": {"name": "inspect", "parameters": {}}}]}
    out, stats = proxy._apply_text_arm(payload, get_arm("acon_hist_ut_co_b768"), "episode")
    assert out["messages"] == payload["messages"]
    assert stats["budget"]["passed"] and stats["compressor_usage"]["calls"] == 0
    path, wire = sent[0]
    assert path == "/v1/c2kv/chat_budget"
    assert "c2kv_measurement_session_id" not in wire
    assert wire["tools"] == payload["tools"]
    assert wire["messages"][0]["role"] == "system"
    assert "tool_calls" not in wire["messages"][2]
    assert "Action:" in wire["messages"][2]["content"]
    assert wire["chat_template_kwargs"]["enable_thinking"] is False


def test_budget_failure_is_typed_terminal_and_does_not_call_actor(monkeypatch, tmp_path):
    log = tmp_path / "requests.jsonl"
    monkeypatch.setattr(proxy, "ARM", get_arm("acon_hist_ut_co_b768"))
    monkeypatch.setattr(proxy, "BACKEND", SimpleNamespace(name="fake"))
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(log))

    def fail(*args):
        raise acon_budget.BudgetExceeded("fixed raw history", {"budget": {
            "limit": 768, "history_after": 900, "passed": False}})

    monkeypatch.setattr(proxy, "_apply_text_arm", fail)
    monkeypatch.setattr(proxy, "_post_json", lambda *args: pytest.fail("must not call actor"))
    body = json.dumps({"model": "model", "messages": [{"role": "user", "content": "task"}]}).encode()
    handler = proxy.ProxyHandler.__new__(proxy.ProxyHandler)
    handler.path = "/v1/chat/completions"
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = io.BytesIO(body)
    sent = {}
    handler._send_json = lambda code, obj: sent.update(code=code, obj=obj)
    handler.do_POST()
    assert sent["code"] == 422
    assert sent["obj"]["error"]["code"] == "acon_history_budget_exceeded"
    row = json.loads(log.read_text())
    assert row["status"] == "acon_history_budget_exceeded"
    assert completion_kind({"result": [], "traceback": json.dumps(sent["obj"])}) == "acon_history_budget_exceeded"
    assert completion_kind({"result": [], "traceback": "upstream 502"}) == "incomplete"
