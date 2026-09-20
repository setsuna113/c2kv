"""Exercise the registered C2KV handler against the installed BFCL decoder."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adapters import bfcl_adapter


def _completion(content: str | None, tool_calls=None):
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate({
        "id": "bfcl-handler-test-response",
        "object": "chat.completion",
        "created": 0,
        "model": "c2kv-agent",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "message": {"role": "assistant", "content": content,
                        "tool_calls": tool_calls},
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                  "total_tokens": 2},
    })


class C2KVBFCLHandlerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
        except ImportError as error:
            raise unittest.SkipTest(f"BFCL installation unavailable: {error}")
        cls.model_config_mapping = MODEL_CONFIG_MAPPING

    def setUp(self):
        from bfcl_eval.model_handler import base_handler

        original_executor = base_handler.execute_multi_turn_func_call
        self.addCleanup(setattr, base_handler, "execute_multi_turn_func_call",
                        original_executor)
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        target = Path(self.tempdir.name) / "harness_events.jsonl"
        patcher = patch.object(bfcl_adapter, "HARNESS_TELEMETRY_PATH", target)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.name = "c2kv-handler-test"
        bfcl_adapter.install_handler("http://127.0.0.1:1/v1",
                                     handler_name=self.name)
        self.addCleanup(self.model_config_mapping.pop, self.name, None)
        registered = self.model_config_mapping[self.name].model_handler
        self.handler = registered.__new__(registered)
        self.handler.is_fc_model = True
        self.handler.temperature = 0.0

    def test_no_tool_call_is_empty_fc_response_for_real_bfcl_decoder(self):
        for content in ("No tool is needed.", None):
            with self.subTest(content=content):
                parsed = self.handler._parse_query_response_FC(
                    _completion(content))

                self.assertEqual(parsed["model_responses"], [])
                self.assertEqual(
                    parsed["model_responses_message_for_chat_history"].content,
                    content)
                self.assertEqual(
                    self.handler.decode_execute(parsed["model_responses"], False), [])
                self.assertEqual(
                    self.handler.decode_ast(parsed["model_responses"], None, False), [])

    def test_native_tool_text_reaches_real_bfcl_decoder(self):
        response = _completion(
            '<tool_call>{"name":"lookup","arguments":{"city":"X"}}</tool_call>')
        self.handler.client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)))

        normalized, _ = self.handler._query_FC({"message": [], "tools": []})
        parsed = self.handler._parse_query_response_FC(normalized)

        self.assertEqual(parsed["model_responses"], [
            {"lookup": '{"city":"X"}'}])
        self.assertEqual(
            self.handler.decode_execute(parsed["model_responses"], False),
            ["lookup(city='X')"])
        self.assertEqual(
            self.handler.decode_ast(parsed["model_responses"], None, False),
            [{"lookup": {"city": "X"}}])

    def test_native_handler_echo_preserves_exact_kv_continuation(self):
        from benchmarks.raw_actor_history import RawActorHistory

        text = 'I will inspect the file.\n<tool_call>{"name":"head","arguments":{"file_name":"report.txt","lines":1}}</tool_call>'
        response = _completion(text)
        source = [{"role": "user", "content": "Read the first line."}]
        wire = response.model_dump()
        wire["metadata"] = {"persistent_history_session": {
            "continuation_mode": "exact_generated_prefix", "generated_text": text}}
        state = RawActorHistory()
        state.commit(source, wire, benchmark="bfcl")
        self.handler.client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: response)))
        normalized, _ = self.handler._query_FC({"message": source, "tools": []})
        parsed = self.handler._parse_query_response_FC(normalized)
        history = self.handler._add_assistant_message_FC({"message": list(source)}, parsed)
        echo = history["message"][-1].model_dump()
        self.assertEqual(echo["content"], "I will inspect the file.")
        restored = state.prepare([*source, echo, {"role": "tool", "content": "First line"}])
        self.assertEqual(restored[1]["content"], text)
        self.assertNotIn("tool_calls", restored[1])


if __name__ == "__main__":
    unittest.main()
