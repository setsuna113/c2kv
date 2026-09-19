"""Check the registered bundled handler against the installed BFCL decoder."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from benchmarks.adapters import bfcl_adapter


def _completion(content, tool_calls=None):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate({
        "id": "bundled-handler-test", "object": "chat.completion", "created": 0,
        "model": "c2kv-agent", "choices": [{
            "index": 0, "finish_reason": "tool_calls" if tool_calls else "stop",
            "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    })


def test_registered_fc_handler_normalizes_no_call_for_real_bfcl(
        tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("bfcl_eval")
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

    if hasattr(bfcl_adapter, "_install_timed_executor"):
        monkeypatch.setattr(bfcl_adapter, "_install_timed_executor", lambda _: None)
    name = "c2kv-bundled-handler-test"
    kwargs = {"handler_name": name}
    if "harness_telemetry_path" in inspect.signature(bfcl_adapter.install_handler).parameters:
        kwargs["harness_telemetry_path"] = tmp_path / "events.jsonl"
    bfcl_adapter.install_handler("http://127.0.0.1:1/v1", **kwargs)
    try:
        handler_type = MODEL_CONFIG_MAPPING[name].model_handler
        handler = handler_type.__new__(handler_type)
        handler.is_fc_model = True
        for content in ("No tool is needed.", None):
            parsed = handler._parse_query_response_FC(_completion(content))
            assert parsed["model_responses"] == []
            assert parsed["model_responses_message_for_chat_history"].content == content
            assert handler.decode_execute(parsed["model_responses"], False) == []
            assert handler.decode_ast(parsed["model_responses"], None, False) == []

        call = {"id": "call_1", "type": "function", "function": {
            "name": "lookup", "arguments": '{"city":"X"}'}}
        parsed = handler._parse_query_response_FC(_completion(None, [call]))
        assert parsed["model_responses"] == [{"lookup": '{"city":"X"}'}]
        assert handler.decode_execute(parsed["model_responses"], False) == [
            "lookup(city='X')"]
    finally:
        MODEL_CONFIG_MAPPING.pop(name, None)
