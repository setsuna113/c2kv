"""ACEBench text-action roles through the shared ACON/HiAgent budget seam."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import acon_budget  # noqa: E402
import hiagent_budget  # noqa: E402
import proxy  # noqa: E402
from arms import get_arm  # noqa: E402
from backends.sglang import SglangBackend  # noqa: E402


@pytest.mark.parametrize("arm_name,policy", [
    ("acon_hist_ut_co_b768", "acon"),
    ("hiagent_full_b768", "hiagent"),
])
def test_ace_roles_and_execution_observation_count_in_actor_history(monkeypatch, arm_name, policy):
    acon_budget.reset_state()
    hiagent_budget.reset_state()
    monkeypatch.setattr(proxy, "BENCHMARK", "acebench")
    seen = []

    def post(path, wire, timeout):
        assert path == "/v1/c2kv/chat_budget"
        boundary = wire["c2kv_kv_memory_hint"]["paper_measurement"]
        start = boundary["history_start_message_count"]
        end = boundary["history_message_count"]
        roles = [item["role"] for item in wire["messages"]]
        seen.append((roles, start, end, wire))
        assert all(item["role"] != "tool" for item in wire["messages"])
        observation = "[{\"result\": \"found\"}]"
        if observation in [item.get("content") for item in wire["messages"]]:
            assert start == 1 and end >= 5
            assert roles[:4] == ["system", "user", "assistant", "user"]
            assert wire["messages"][3]["content"] == observation
        history_tokens = (end - start) * 20
        return {"success": True, "server_tokenized": True,
                "history_tokens": history_tokens, "prompt_tokens": history_tokens + 20,
                "history_start": 10 if history_tokens else 0,
                "history_end": 10 + history_tokens if history_tokens else 0}

    monkeypatch.setattr(proxy, "BACKEND", SglangBackend(post))
    original = {"model": "model", "messages": [
        {"role": "system", "content": "Below is the list of APIs you can use:\n[{'name': 'search'}]"},
        {"role": "user", "content": "Find a record."},
        {"role": "assistant", "content": "search(query='record')"},
        {"role": "tool", "tool_call_id": "acebench-execution-2",
         "content": "[{\"result\": \"found\"}]"},
        {"role": "assistant", "content": "The result was found."},
        {"role": "user", "content": "Continue."},
    ]}
    out, stats = proxy._apply_text_arm(original, get_arm(arm_name), "ace-role-budget")
    assert seen and stats["policy"] == policy
    assert any("[{\"result\": \"found\"}]" in [item.get("content") for item in wire["messages"]]
               for _, _, _, wire in seen)
    assert stats["budget"]["limit"] == 768 and stats["budget"]["passed"]
    assert stats["budget"]["history_before"] == 80
    assert stats["compressor_usage"]["calls"] == 0
    assert original["messages"][3]["role"] == "tool"
    assert out["messages"][-1] == original["messages"][-1]
    assert "search" in seen[0][3]["messages"][0]["content"]
    if policy == "hiagent":
        assert stats["environment_action_format"] == "python_content"
        assert out["tools"][-1]["function"]["name"] == "hiagent_retrieve"
    else:
        assert "tools" not in out


def test_hiagent_subgoal_comment_remains_parseable_as_ace_python_expression():
    # ACEBench's execution decoder wraps the response in brackets, strips the
    # brackets, then parses the result in eval mode. Python comments do not
    # change the call expression that reaches its evaluator.
    action = "# Subgoal: find the matching record\nsearch(query='record')"
    wrapped = "[" + action + "]"
    parsed = ast.parse(wrapped.strip("[]'"), mode="eval")
    assert isinstance(parsed.body, ast.Call)
