"""CPU-only contract tests for ops/textarm_smoke.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ops import textarm_smoke as smoke  # noqa: E402


def test_synthetic_histories_cross_real_compressor_thresholds():
    history = smoke.acon_history_messages()
    prefix = history[1:-2]
    assert smoke.textarms._message_chars(prefix) > smoke.textarms.ACON_HISTORY_THRESHOLD_CHARS
    observation = smoke.acon_observation_messages()[3]["content"]
    assert len(observation) > smoke.textarms.ACON_OBS_THRESHOLD_CHARS


def test_hiagent_fixture_has_two_completed_segments_and_hidden_marker():
    messages = smoke.hiagent_messages()
    calls = []

    def compress(_payload):
        calls.append(1)
        return "summary"

    out, stats = smoke.textarms.hiagent_transform(
        messages, compress, lambda message: str(message.get("tool_calls")),
        variant="full")
    assert stats["n_segments"] == 3 and stats["n_summarized"] == 2
    assert len(calls) == 2
    assert smoke.RETRIEVAL_SECRET not in str(out)
    assert smoke.USER_SENTINEL in str(out)


def test_policy_payload_uses_proxy_training_dialect(monkeypatch):
    staged = {"model": "c2kv-agent", "messages": [
        {"role": "user", "content": "request"},
        {"role": "tool", "content": "observation", "tool_call_id": "x"},
    ]}
    wire = smoke._policy_payload(staged, smoke.get_arm("full"))
    assert all(message["role"] != "tool" for message in wire["messages"])
    assert wire["temperature"] == 0.0
    assert wire["chat_template_kwargs"] == {"enable_thinking": False}
