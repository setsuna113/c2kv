"""Assembly-layer tests for the arm-aware proxy.

These pin the two properties a serving number depends on:

1. an assistant message whose action lives in ``tool_calls`` must reach
   ``/v1/c2kv/extract`` with that action in the text -- the server POPS every
   annotated message out of the prompt, so anything the proxy fails to put in
   the document is simply gone from the model's history;
2. ``--doc-packing turn`` must reproduce the trainer's turn documents
   byte-for-byte, since that is the segment shape ``doc_mode=history_only``
   checkpoints are trained on.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import proxy  # noqa: E402
from arms import get_arm  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_proxy(monkeypatch):
    monkeypatch.setattr(proxy, "DOC_PACKING", "message")
    monkeypatch.setattr(proxy, "MAX_DOCS", 0)
    monkeypatch.setattr(proxy, "CACHE", proxy.ExtractCache())
    calls = []

    def fake_extract(role, content, ratio, timeout):
        calls.append({"role": role, "content": content, "ratio": ratio})
        return {"key_hash": f"h{len(calls)}", "gist_len": 4, "original_seq_len": 32}

    monkeypatch.setattr(proxy, "_extract", fake_extract)
    return calls


CONVERSATION = [
    {"role": "system", "content": "you are an agent"},
    {"role": "user", "content": "book a flight"},
    {"role": "assistant", "content": None,
     "tool_calls": [{"type": "function",
                     "function": {"name": "search_flights",
                                  "arguments": '{"city": "LHR"}'}}]},
    {"role": "tool", "content": "3 flights found"},
    {"role": "assistant", "content": "I found three."},
    {"role": "user", "content": "take the cheapest"},
]


def test_tool_call_action_survives_compression(_reset_proxy):
    calls = _reset_proxy
    out, gist, original, n_gist, dropped = proxy._assemble(
        CONVERSATION, get_arm("c2kv"), 5
    )
    assert dropped == 0
    compressed_texts = [call["content"] for call in calls]
    assert any("search_flights" in text for text in compressed_texts), compressed_texts
    assert any('LHR' in text for text in compressed_texts), compressed_texts
    assert any("<tool_call>" in text for text in compressed_texts), compressed_texts
    # never hand the extractor an empty document
    assert all(text.strip() for text in compressed_texts)
    assert n_gist == len(calls) > 0
    assert gist == 4 * n_gist and original == 32 * n_gist
    # system stays raw and first; the current turn stays raw and last
    assert out[0] == CONVERSATION[0]
    assert out[-1] == CONVERSATION[-1]
    assert all("c2kv_key_hash" in m for m in out[1:-1])


def test_turn_packing_matches_trainer_documents(monkeypatch, _reset_proxy):
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    calls = _reset_proxy
    proxy._assemble(CONVERSATION, get_arm("c2kv"), 5)
    assert [call["role"] for call in calls] == ["user"] * len(calls)
    joined = "\n---\n".join(call["content"] for call in calls)
    assert "Previous turn" in joined
    assert "[User query]" in joined and "book a flight" in joined
    assert "[Assistant output]" in joined
    assert "Action:" in joined and "search_flights" in joined
    assert "3 flights found" in joined


def test_turn_documents_equal_trainer_implementation():
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(HERE.parent / "python"))
    from train.train_data_multiturn import (  # noqa: E402
        _agent_history_turn_docs,
        _normal_agent_message,
    )

    history = CONVERSATION[:-1]
    expected = [
        doc["content"]
        for doc in _agent_history_turn_docs(
            [m for m in (_normal_agent_message(x) for x in history)
             if m is not None and m.get("role") != "system"]
        )
    ]
    assert proxy._turn_documents([m for m in history if m["role"] != "system"]) == expected


def test_max_docs_keeps_the_tail(monkeypatch, _reset_proxy):
    monkeypatch.setattr(proxy, "MAX_DOCS", 1)
    calls = _reset_proxy
    _, _, _, n_gist, dropped = proxy._assemble(CONVERSATION, get_arm("c2kv"), 5)
    assert n_gist == 1 and dropped == 3
    assert "I found three." in calls[0]["content"]


def test_hybrid_tail_stays_raw(_reset_proxy):
    calls = _reset_proxy
    out, _, _, n_gist, _ = proxy._assemble(CONVERSATION, get_arm("hybrid"), 5)
    arm = get_arm("hybrid")
    assert arm.hybrid_top_k > 0
    raw_texts = [m.get("content") for m in out if "c2kv_key_hash" not in m]
    assert "I found three." in raw_texts
    assert all("I found three." not in call["content"] for call in calls)


def test_full_arm_is_a_passthrough(_reset_proxy):
    out, gist, original, n_gist, dropped = proxy._assemble(
        CONVERSATION, get_arm("full"), 5
    )
    assert out == CONVERSATION
    assert (gist, original, n_gist, dropped) == (0, 0, 0, 0)
    assert _reset_proxy == []
