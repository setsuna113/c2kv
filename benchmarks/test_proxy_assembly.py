"""Assembly-layer tests for the arm-aware proxy.

These pin the properties a serving number depends on:

1. an assistant message whose action lives in ``tool_calls`` must reach
   ``/v1/c2kv/extract`` with that action in the text, rendered exactly as the
   trainer renders it -- the server POPS every annotated message out of the
   prompt, so anything the proxy fails to put in the document is simply gone
   from the model's history;
2. ``--doc-packing turn`` must reproduce the trainer's turn documents
   byte-for-byte, since that is the segment shape ``doc_mode=history_only``
   checkpoints are trained on;
3. the trainer's split-then-cap geometry (``max_doc_length`` then
   ``max_doc_num`` with the tail policy that keeps document 0);
4. a request the proxy could not serve must leave a row in the request log,
   otherwise an aborted task is indistinguishable from a wrong answer.
"""
import inspect
import io
import json
import sys
from pathlib import Path
from urllib.error import URLError

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import proxy  # noqa: E402
import run as run_module  # noqa: E402
from arms import get_arm  # noqa: E402

# Captured before any fixture patches the module constants.
SHIPPED_DEFAULTS = (proxy.DOC_PACKING, proxy.MAX_DOCS, proxy.MAX_DOC_LENGTH)


@pytest.fixture(autouse=True)
def _reset_proxy(monkeypatch):
    monkeypatch.setattr(proxy, "DOC_PACKING", "message")
    monkeypatch.setattr(proxy, "MAX_DOCS", 0)
    monkeypatch.setattr(proxy, "MAX_DOC_LENGTH", 0)
    monkeypatch.setattr(proxy, "CACHE", proxy.ExtractCache())
    calls = []

    def fake_extract(role, content, ratio, timeout):
        calls.append({"role": role, "content": content, "ratio": ratio})
        return {"key_hash": f"h{len(calls)}", "gist_len": 4, "original_seq_len": 32}

    monkeypatch.setattr(proxy, "_extract", fake_extract)
    return calls


# OpenAI wire form: function.arguments is a JSON *string*.
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

# The form the trainer's agent-llm-traces carry: arguments is already an object.
CONVERSATION_TRACE = [
    dict(message, tool_calls=[{"type": "function",
                              "function": {"name": "search_flights",
                                           "arguments": {"city": "LHR"}}}])
    if message.get("tool_calls") else dict(message)
    for message in CONVERSATION
]

RENDERED_CALL = (
    '<tool_call>\n{"name":"search_flights","arguments":{"city":"LHR"}}\n</tool_call>'
)


def test_tool_call_action_survives_compression(_reset_proxy):
    calls = _reset_proxy
    out, stats = proxy._assemble(CONVERSATION, get_arm("c2kv"), 5)
    assert stats["dropped_docs"] == 0 and stats["n_split"] == 0
    compressed_texts = [call["content"] for call in calls]
    # the OpenAI JSON-string arguments must be parsed back into an object, so
    # the document is byte-identical to the trainer's rendering
    assert any(RENDERED_CALL in text for text in compressed_texts), compressed_texts
    # never hand the extractor an empty document
    assert all(text.strip() for text in compressed_texts)
    assert stats["n_docs"] == len(calls) > 0
    assert stats["gist_tokens"] == 4 * stats["n_docs"]
    assert stats["original_tokens"] == 32 * stats["n_docs"]
    # system stays raw and first; the current turn stays raw and last
    assert out[0] == CONVERSATION[0]
    assert out[-1] == CONVERSATION[-1]
    assert all("c2kv_key_hash" in m for m in out[1:-1])


def test_malformed_arguments_string_falls_back_to_the_raw_string():
    rendered = proxy._render_tool_calls(
        [{"type": "function",
          "function": {"name": "f", "arguments": "not json"}}]
    )
    assert rendered == '<tool_call>\n{"name":"f","arguments":"not json"}\n</tool_call>'


def test_turn_packing_matches_trainer_documents(monkeypatch, _reset_proxy):
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    calls = _reset_proxy
    proxy._assemble(CONVERSATION, get_arm("c2kv"), 5)
    assert [call["role"] for call in calls] == ["user"] * len(calls)
    joined = "\n---\n".join(call["content"] for call in calls)
    assert "Previous turn" in joined
    assert "[User query]" in joined and "book a flight" in joined
    assert "[Assistant output]" in joined
    assert "Action:" in joined and RENDERED_CALL in joined
    assert "3 flights found" in joined


def test_turn_documents_equal_trainer_implementation():
    pytest.importorskip("pyarrow")
    sys.path.insert(0, str(HERE.parent / "python"))
    from train.train_data_multiturn import (  # noqa: E402
        _agent_history_turn_docs,
        _normal_agent_message,
    )

    # The proxy is fed the OpenAI wire form (arguments as a JSON string); the
    # trainer is fed the trace form (arguments already an object).  Both must
    # produce the same documents -- that equality is the whole point.
    history = CONVERSATION[:-1]
    trainer_history = CONVERSATION_TRACE[:-1]
    expected = [
        doc["content"]
        for doc in _agent_history_turn_docs(
            [m for m in (_normal_agent_message(x) for x in trainer_history)
             if m is not None and m.get("role") != "system"]
        )
    ]
    assert proxy._turn_documents([m for m in history if m["role"] != "system"]) == expected


def test_shipped_defaults_are_the_trained_geometry():
    assert SHIPPED_DEFAULTS == ("turn", 16, 768)
    signature = inspect.signature(run_module.start_proxy)
    assert signature.parameters["doc_packing"].default == "turn"
    assert signature.parameters["max_docs"].default == 16
    assert signature.parameters["max_doc_length"].default == 768


def _turn_conversation(n_turns: int):
    messages = [{"role": "system", "content": "you are an agent"}]
    for index in range(n_turns):
        messages.append({"role": "user", "content": f"q{index}"})
        messages.append({"role": "assistant", "content": f"a{index}"})
    messages.append({"role": "user", "content": "current question"})
    return messages


def test_max_docs_keeps_doc_zero_and_the_tail(monkeypatch, _reset_proxy):
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    monkeypatch.setattr(proxy, "MAX_DOCS", 3)
    calls = _reset_proxy
    out, stats = proxy._assemble(_turn_conversation(5), get_arm("c2kv"), 5)
    assert stats["n_docs"] == 3 and stats["dropped_docs"] == 2
    # every candidate turn doc is extracted; only the kept ones are referenced
    assert len(calls) == 5
    compressed = [message["content"] for message in out if "c2kv_key_hash" in message]
    assert len(compressed) == 3
    # doc0 (the task-defining opening turn) plus the two newest, in order
    assert "q0" in compressed[0]
    assert "q3" in compressed[1]
    assert "q4" in compressed[2]
    assert all("q1" not in text and "q2" not in text for text in compressed)


def test_max_docs_of_one_keeps_only_the_newest(monkeypatch, _reset_proxy):
    monkeypatch.setattr(proxy, "MAX_DOCS", 1)
    calls = _reset_proxy
    _, stats = proxy._assemble(CONVERSATION, get_arm("c2kv"), 5)
    assert stats["n_docs"] == 1 and stats["dropped_docs"] == 3
    assert "I found three." in calls[-1]["content"]


def test_fit_doc_splits_an_oversized_document():
    text = "".join(f"line {index:03d} " + "x" * 30 + "\n" for index in range(100))
    seen = []

    def extract_fn(role, content, ratio):
        seen.append(content)
        # one token per character: an exact, monotone stand-in for the server
        return {"key_hash": f"h{len(seen)}", "gist_len": max(1, len(content) // 8),
                "original_seq_len": len(content)}

    pieces = proxy._fit_doc(text, 8, extract_fn, 768)
    assert len(pieces) >= 3
    assert all(int(record["original_seq_len"]) <= 768 for _, record in pieces)
    assert "".join(piece for piece, _ in pieces) == text


def test_split_runs_before_the_cap_and_is_reported(monkeypatch, _reset_proxy):
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    monkeypatch.setattr(proxy, "MAX_DOC_LENGTH", 100)
    monkeypatch.setattr(proxy, "MAX_DOCS", 0)
    calls = _reset_proxy

    def fake_extract(role, content, ratio, timeout):
        calls.append({"role": role, "content": content, "ratio": ratio})
        return {"key_hash": f"h{len(calls)}", "gist_len": 4,
                "original_seq_len": len(content)}

    monkeypatch.setattr(proxy, "_extract", fake_extract)
    long_line = "\n".join("y" * 60 for _ in range(20))
    messages = [
        {"role": "user", "content": long_line},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "now what?"},
    ]
    _, stats = proxy._assemble(messages, get_arm("c2kv"), 5)
    assert stats["n_split"] == 1
    assert stats["n_docs"] > 1


def test_hybrid_tail_stays_raw(_reset_proxy):
    calls = _reset_proxy
    out, stats = proxy._assemble(CONVERSATION, get_arm("hybrid"), 5)
    arm = get_arm("hybrid")
    assert arm.hybrid_top_k > 0
    raw_texts = [m.get("content") for m in out if "c2kv_key_hash" not in m]
    assert "I found three." in raw_texts
    assert all("I found three." not in call["content"] for call in calls)


def test_full_arm_is_a_passthrough(_reset_proxy):
    out, stats = proxy._assemble(CONVERSATION, get_arm("full"), 5)
    assert out == CONVERSATION
    assert stats == {
        "gist_tokens": 0, "original_tokens": 0,
        "n_docs": 0, "n_split": 0, "dropped_docs": 0,
    }
    assert _reset_proxy == []


class _StubHandler(proxy.ProxyHandler):
    """do_POST driver: no socket, records what would have been sent."""

    def __init__(self, payload):
        raw = json.dumps(payload).encode("utf-8")
        self.path = "/v1/chat/completions"
        self.headers = {"Content-Length": str(len(raw))}
        self.rfile = io.BytesIO(raw)
        self.sent = []

    def _send_json(self, code, obj):
        self.sent.append((code, obj))


def _log_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_assembly_failure_is_logged(monkeypatch, tmp_path, _reset_proxy):
    log_path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(proxy, "ARM", get_arm("c2kv"))
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(log_path))

    def boom(*args, **kwargs):
        raise RuntimeError("c2kv extract failed: pool miss")

    monkeypatch.setattr(proxy, "_assemble", boom)
    handler = _StubHandler({"messages": CONVERSATION, "tools": []})
    handler.do_POST()
    assert handler.sent[0][0] == 502
    rows = _log_rows(log_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "assembly_failed"
    assert "pool miss" in rows[0]["error"]


def test_upstream_failure_is_logged(monkeypatch, tmp_path, _reset_proxy):
    log_path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(proxy, "ARM", get_arm("c2kv"))
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(log_path))

    def boom(*args, **kwargs):
        raise URLError("connection reset")

    monkeypatch.setattr(proxy, "_http_json", boom)
    handler = _StubHandler({"messages": CONVERSATION, "tools": []})
    handler.do_POST()
    assert handler.sent[0][0] == 502
    rows = _log_rows(log_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "upstream_failed"
    assert "connection reset" in rows[0]["error"]
    # the assembly that did succeed is still accounted for
    assert rows[0]["n_docs"] > 0


def test_ok_row_records_the_segmentation_regime(monkeypatch, tmp_path, _reset_proxy):
    log_path = tmp_path / "requests.jsonl"
    monkeypatch.setattr(proxy, "ARM", get_arm("c2kv"))
    monkeypatch.setattr(proxy, "REQUEST_LOG_PATH", str(log_path))
    monkeypatch.setattr(proxy, "DOC_PACKING", "turn")
    monkeypatch.setattr(proxy, "MAX_DOCS", 16)
    monkeypatch.setattr(proxy, "MAX_DOC_LENGTH", 768)
    monkeypatch.setattr(
        proxy, "_http_json",
        lambda *args, **kwargs: {"choices": [{"finish_reason": "stop"}],
                                 "usage": {"prompt_tokens": 10}},
    )
    handler = _StubHandler({"messages": CONVERSATION, "tools": []})
    handler.do_POST()
    code, body = handler.sent[0]
    assert code == 200
    block = body["c2kv_proxy"]
    for key in ("doc_packing", "max_docs", "max_doc_length", "n_docs",
                "n_split", "dropped_docs"):
        assert key in block, key
    assert (block["doc_packing"], block["max_docs"], block["max_doc_length"]) == (
        "turn", 16, 768
    )
    row = _log_rows(log_path)[0]
    assert row["status"] == "ok"
    assert (row["doc_packing"], row["max_docs"], row["max_doc_length"]) == (
        "turn", 16, 768
    )
    for key in ("n_docs", "n_split", "dropped_docs"):
        assert key in row, key
