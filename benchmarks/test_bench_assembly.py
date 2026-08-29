"""Guards for the bench arm assembly: training dialect, hybrid tail, repair.

torch-free -- only proxy._assemble, arms and dialect are exercised, with the
/v1/c2kv/extract call stubbed.

Run: python -m pytest benchmarks/test_bench_assembly.py -v
"""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import proxy  # noqa: E402
from arms import ARMS, Arm, get_arm  # noqa: E402
from dialect import message_text  # noqa: E402


def _tool_call(name="search_flights", arguments='{"from":"SFO"}'):
    return {"id": "c1", "type": "function",
            "function": {"name": name, "arguments": arguments}}


@pytest.fixture
def captured(monkeypatch):
    """Record every (role, text) the proxy would hand to /v1/c2kv/extract."""
    seen = []

    def fake_extract(role, content, ratio, timeout):
        seen.append((role, content))
        return {"key_hash": f"h{len(seen)}", "gist_len": 4, "original_seq_len": 32}

    monkeypatch.setattr(proxy, "_extract", fake_extract)
    return seen


# --------------------------------------------------------------------------
# the training dialect: an assistant action must survive compression
# --------------------------------------------------------------------------

def test_assistant_tool_call_is_rendered_not_deleted(captured):
    """content=None + tool_calls used to hash the literal string '""'.

    OpenAI assistant turns carry the action in `tool_calls`, not `content`, so
    reading content alone deleted every action in the compressed history -- the
    c2kv arm was measuring compression *plus* total loss of the agent's own
    moves.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "book a flight"},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call()]},
        {"role": "tool", "content": '{"flights": []}'},
        {"role": "user", "content": "and a hotel"},
    ]
    proxy._assemble(messages, get_arm("c2kv"), 1)
    texts = [text for _role, text in captured]
    assert texts, "nothing was compressed"
    assert not any(text == '""' for text in texts), f"action deleted: {texts}"
    assert any("search_flights" in text for text in texts)
    assert any(text.startswith("Action:\n<tool_call>") for text in texts)


def test_rendered_content_matches_hf_server_dialect():
    """The proxy and hf_server must produce the same string for one message."""
    message = {"role": "assistant", "content": "Looking.", "tool_calls": [_tool_call()]}
    assert message_text(message) == (
        'Looking.\n\nAction:\n<tool_call>\n'
        '{"name":"search_flights","arguments":{"from":"SFO"}}\n</tool_call>'
    )


def test_compressed_message_drops_tool_calls(captured):
    """Otherwise hf_server renders the action a second time on a cache miss."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call()]},
        {"role": "tool", "content": "ok"},
        {"role": "user", "content": "next"},
    ]
    out, _gist, _orig, _n = proxy._assemble(messages, get_arm("c2kv"), 1)
    for message in out:
        if "c2kv_key_hash" in message:
            assert "tool_calls" not in message
            assert message["content"] != '""'


def test_full_arm_passes_messages_through_untouched(captured):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call()]},
        {"role": "user", "content": "next"},
    ]
    out, _g, _o, n_gist = proxy._assemble(messages, get_arm("full"), 1)
    assert n_gist == 0 and not captured
    assert out == messages


# --------------------------------------------------------------------------
# hybrid tail
# --------------------------------------------------------------------------

def _history_roles(messages, arm):
    """(compressed, raw) role lists for the history portion."""
    cutoff = proxy._history_cutoff(messages)
    out, _g, _o, _n = proxy._assemble(messages, arm, 1)
    compressed = [m["role"] for m in out[:cutoff] if "c2kv_key_hash" in m]
    raw = [m["role"] for m in out[:cutoff] if "c2kv_key_hash" not in m]
    return compressed, raw


@pytest.fixture
def conversation():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("a1")]},
        {"role": "tool", "content": "t1"},
        {"role": "assistant", "content": "a1 text"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("a2")]},
        {"role": "tool", "content": "t2"},
    ]


def test_hybrid_keeps_exactly_k_tail_messages_raw(conversation, captured):
    """hybrid_top_k counts MESSAGES; k=3 is about 1.5 agent turns."""
    for k, arm_name in ((1, "hybrid1"), (3, "hybrid"), (5, "hybrid5")):
        captured.clear()
        compressed, raw = _history_roles(conversation, get_arm(arm_name))
        cutoff = proxy._history_cutoff(conversation)
        # system is always raw and never counted against k
        assert raw[0] == "system"
        n_raw_history = len(raw) - 1
        assert n_raw_history == min(k, cutoff - 1), (
            f"k={k}: kept {n_raw_history} raw, expected {min(k, cutoff - 1)}"
        )
        assert len(compressed) == cutoff - 1 - n_raw_history


def test_hybrid_never_compresses_the_system_prompt(conversation, captured):
    for arm_name in ("c2kv", "hybrid1", "hybrid", "hybrid5"):
        captured.clear()
        proxy._assemble(conversation, get_arm(arm_name), 1)
        assert not any(role == "system" for role, _ in captured)


def test_hybrid_degenerates_to_full_on_short_history(captured):
    """k >= history length leaves nothing compressed.

    Worth pinning: such tasks are indistinguishable from the full arm, so they
    carry no information about compression and must not be read as hybrid wins.
    """
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    _out, _g, _o, n_gist = proxy._assemble(messages, get_arm("hybrid5"), 1)
    assert n_gist == 0
    assert not captured


# --------------------------------------------------------------------------
# repair arms must not be able to masquerade as compression arms
# --------------------------------------------------------------------------

def test_repair_arm_is_rejected_until_implemented():
    """proxy._assemble never reads Arm.repair, so such an arm would run as a
    plain c2kv arm and its rows would be labelled with the repair arm's name."""
    with pytest.raises(NotImplementedError) as exc:
        Arm(name="corr_first", compress_history=True,
            repair={"kind": "corr", "block": 0}).validate()
    assert "repair" in str(exc.value)


def test_registered_arms_all_validate():
    for name in ARMS:
        get_arm(name)


def test_plan_block1_arms_are_registered():
    for name in ("full", "c2kv", "hybrid1", "hybrid", "hybrid5"):
        assert get_arm(name).name == name
    assert [get_arm(n).hybrid_top_k for n in ("hybrid1", "hybrid", "hybrid5")] == [1, 3, 5]
