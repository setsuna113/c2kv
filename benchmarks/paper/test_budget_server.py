"""CPU-only checks for the paper-owned chat budget preflight."""

import copy
from types import SimpleNamespace

import pytest

from benchmarks.paper.budget_server import measure_chat_budget


class FakeRequest:
    def __init__(self, messages, start=2, end=3):
        self.messages = messages
        self.chat_template_kwargs = {
            "enable_thinking": False,
            "reasoning_effort": "low",
        }
        self.reasoning_effort = None
        self.c2kv_kv_memory_hint = {
            "paper_measurement": {
                "history_start_message_count": start,
                "history_message_count": end,
            }
        }

    def model_copy(self, deep=False):
        assert deep
        return copy.deepcopy(self)


class FakeServingChat:
    def __init__(self):
        self.tokenizer_manager = SimpleNamespace(
            model_config=SimpleNamespace(is_multimodal=False)
        )
        self.is_gpt_oss = False
        self.processed = None
        self.resolved = None

    def _process_messages(self, request, is_multimodal):
        assert is_multimodal is False
        self.processed = request
        return SimpleNamespace(prompt_ids=list(range(30)))

    def _resolve_paper_history_token_count(self, request, prompt_ids):
        self.resolved = request
        assert prompt_ids == list(range(30))
        config = request.c2kv_kv_memory_hint["paper_measurement"]
        assert config["history_start_message_count"] == 2
        assert config["history_message_count"] == 3
        config.update(history_start=11, history_end=20, server_tokenized=True)
        return 9


def test_chat_budget_uses_full_actor_request_and_keeps_summary_in_history():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "common task packet"},
        {"role": "user", "content": "<HISTORY_SUMMARY>dynamic</HISTORY_SUMMARY>"},
        {"role": "user", "content": "current observation"},
    ]
    request = FakeRequest(messages)
    serving_chat = FakeServingChat()

    result = measure_chat_budget(serving_chat, request)

    assert result == {
        "success": True,
        "history_tokens": 9,
        "prompt_tokens": 30,
        "server_tokenized": True,
        "history_start": 11,
        "history_end": 20,
    }
    assert serving_chat.processed is serving_chat.resolved
    assert serving_chat.processed.messages == messages
    assert serving_chat.processed.reasoning_effort == "low"
    assert serving_chat.processed.chat_template_kwargs == {"enable_thinking": False}
    assert request.reasoning_effort is None
    assert request.chat_template_kwargs["reasoning_effort"] == "low"
    assert "history_start" not in request.c2kv_kv_memory_hint["paper_measurement"]


@pytest.mark.parametrize("config", [{}, {"history_message_count": 3}, None])
def test_chat_budget_rejects_missing_history_boundary(config):
    request = FakeRequest([])
    request.c2kv_kv_memory_hint = {"paper_measurement": config}
    serving_chat = FakeServingChat()

    with pytest.raises(ValueError, match="paper_measurement"):
        measure_chat_budget(serving_chat, request)
    assert serving_chat.processed is None


def test_chat_budget_rejects_multimodal_without_rendering():
    request = FakeRequest([])
    serving_chat = FakeServingChat()
    serving_chat.tokenizer_manager.model_config.is_multimodal = True

    with pytest.raises(ValueError, match="text-only"):
        measure_chat_budget(serving_chat, request)
    assert serving_chat.processed is None


def test_chat_budget_rejects_missing_prompt_ids():
    request = FakeRequest([])
    serving_chat = FakeServingChat()
    serving_chat._process_messages = lambda *_: SimpleNamespace(prompt_ids="not IDs")

    with pytest.raises(ValueError, match="prompt token IDs"):
        measure_chat_budget(serving_chat, request)
