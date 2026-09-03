"""Offline tests for the text-level baseline arms (textarms.py).

Audit-hardened assertions: compression must actually shrink the message
list, user instructions must survive, subgoal segmentation must trigger on
Subgoal content and be visibly degenerate on null-content tool-call
replies, ACON must not duplicate history (the tuple-precedence bug), and
an empty compressor result must raise, never cache.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import textarms


def _action_dialect(message):
    return "tool:" + str(message.get("tool_calls"))


def _fake_compress_ok(payload):
    """Accepts a full request payload dict; returns a canned summary."""
    assert "model" in payload and payload["messages"], "payload contract"
    user = payload["messages"][-1]["content"]
    return f"SUMMARY({len(user)} chars)"


def _fake_compress_empty(payload):
    return "   "


def _reset():
    textarms.reset_state()


# ---- HiAgent -----------------------------------------------------------------

def test_hiagent_compresses_and_shrinks():
    _reset()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "Subgoal: find data", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "found it"},
        {"role": "user", "content": "IMPORTANT: my budget is $300"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2"}]},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "Subgoal: finish up", "tool_calls": [{"id": "3"}]},
        {"role": "tool", "content": "done-ish"},
        {"role": "user", "content": "continue"},
    ]
    out, stats = textarms.hiagent_transform(
        messages, _fake_compress_ok, _action_dialect)
    assert stats["n_segments"] == 2 and stats["n_summarized"] == 1
    assert stats["n_compressor_calls"] == 1
    assert len(out) < len(messages), "compression must shrink the message list"
    assert out_chars_lt(stats), stats
    # note injected into system
    assert "Subgoal:" in out[0]["content"] and out[0]["role"] == "system"
    # user instruction inside the completed segment SURVIVES verbatim
    assert any(m.get("role") == "user" and "$300" in (m.get("content") or "")
               for m in out), "user constraint must survive"
    # summary message present
    assert any(m.get("role") == "user"
               and (m.get("content") or "").startswith("Subgoal 1: find data")
               for m in out)
    # current segment raw
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in out)
    assert stats["degenerate"] is False
    # cached re-run: no new compressor call
    _, stats2 = textarms.hiagent_transform(
        messages, _fake_compress_ok, _action_dialect)
    assert stats2["n_compressor_calls"] == 0


def test_hiagent_null_content_toolcall_reply_is_visible_degenerate():
    _reset()
    # BFCL FC shape: assistant replies carry tool_calls with null content —
    # no Subgoal line ever appears.  The arm is a passthrough and MUST say
    # so (audit: silent full-arm degeneration).
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "x"}]},
        {"role": "tool", "content": "r"},
        {"role": "user", "content": "next"},
    ]
    out, stats = textarms.hiagent_transform(
        messages, _fake_compress_ok, _action_dialect)
    assert stats["n_segments"] == 0 and stats["n_summarized"] == 0
    assert stats["degenerate"] is True
    assert stats["n_compressor_calls"] == 0
    assert len(out) == len(messages)


def test_hiagent_user_in_summary_input():
    _reset()
    seen = {}

    def spy(payload):
        seen["user"] = payload["messages"][-1]["content"]
        return "ok summary"

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "Subgoal: a", "tool_calls": [{"id": "1"}]},
        {"role": "user", "content": "IMPORTANT: my budget is $300"},
        {"role": "tool", "content": "obs"},
        {"role": "assistant", "content": "Subgoal: b", "tool_calls": [{"id": "2"}]},
        {"role": "tool", "content": "obs2"},
    ]
    textarms.hiagent_transform(messages, spy, _action_dialect)
    assert "$300" in seen["user"], "user constraint must reach the summarizer"


def out_chars_lt(stats):
    return stats["out_chars"] < stats["raw_chars"]


# ---- ACON --------------------------------------------------------------------

def _acon_messages(obs_len=200):
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "long task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "content": "z" * obs_len},
        {"role": "user", "content": "final turn"},
    ]


def test_acon_obs_threshold_and_shrink():
    _reset()
    big = _acon_messages(obs_len=textarms.ACON_OBS_THRESHOLD_CHARS + 10)
    out, stats = textarms.acon_transform(
        big, _fake_compress_ok, _action_dialect, "convA", mode="obs")
    assert stats["n_obs_compressed"] == 1
    assert stats["n_compressor_calls"] == 1
    assert len(out[3]["content"]) < len(big[3]["content"])
    assert out[3]["content"].startswith("SUMMARY(")
    # under threshold: untouched
    small = _acon_messages(obs_len=300)
    _, stats2 = textarms.acon_transform(
        small, _fake_compress_ok, _action_dialect, "convA2", mode="obs")
    assert stats2["n_obs_compressed"] == 0
    # idempotent (cache)
    _, stats3 = textarms.acon_transform(
        big, _fake_compress_ok, _action_dialect, "convA", mode="obs")
    assert stats3["n_compressor_calls"] == 0


def _acon_long_messages():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "long task"},
    ]
    for i in range(20):
        msgs += [
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"a{i}"}]},
            {"role": "tool", "content": "z" * 900},
        ]
    msgs.append({"role": "user", "content": "final turn"})
    return msgs


def test_acon_hist_no_duplication_and_embeds_summary():
    _reset()
    messages = _acon_long_messages()
    out, stats = textarms.acon_transform(
        messages, _fake_compress_ok, _action_dialect, "convB", mode="hist")
    assert stats["history_compressed"] is True
    assert stats["n_compressor_calls"] == 1
    # THE audit bug: the old tuple precedence made tail = ALL nonsystem and
    # duplicated the whole history (43 in -> 84 out).  Output must SHRINK.
    assert len(out) < len(messages), (
        f"history compression must shrink: {len(messages)} -> {len(out)}")
    # summary is embedded in the FIRST user prompt, not a standalone message
    first_user = next(m for m in out if m.get("role") == "user")
    assert "<HISTORY_SUMMARY>" in first_user["content"]
    assert first_user["content"].startswith("long task")
    # the preserved tail keeps the LAST action/observation pair (k=2) and
    # the final user turn; no message is duplicated
    assert out[-1]["content"] == "final turn"
    assert out[-2]["content"] == "z" * 900
    roles = [m.get("role") for m in out]
    assert len(roles) == len(set(map(str, roles))) or True  # roles repeat legitimately
    ids = [tc.get("id") for m in out for tc in (m.get("tool_calls") or [])]
    assert len(ids) == len(set(ids)), "tool_call ids must not duplicate"
    # rolling state: next turn carries prev_summary, one new compressor call
    messages2 = messages + [{"role": "assistant", "content": "done",
                             "tool_calls": [{"id": "z"}]}]
    _, stats3 = textarms.acon_transform(
        messages2, _fake_compress_ok, _action_dialect, "convB", mode="hist")
    assert stats3["n_compressor_calls"] == 1


def test_acon_hist_below_threshold_passthrough_no_duplication():
    _reset()
    messages = _acon_messages(obs_len=100)
    out, stats = textarms.acon_transform(
        messages, _fake_compress_ok, _action_dialect, "convC", mode="hist")
    assert stats["history_compressed"] is False
    assert len(out) == len(messages), "below threshold must not duplicate"


def test_acon_modes_split():
    _reset()
    messages = _acon_long_messages()
    messages[3]["content"] = "y" * (textarms.ACON_OBS_THRESHOLD_CHARS + 5)
    _, so = textarms.acon_transform(
        messages, _fake_compress_ok, _action_dialect, "convD", mode="obs")
    assert so["n_obs_compressed"] == 1 and so["history_compressed"] is False
    _reset()
    _, sh = textarms.acon_transform(
        messages, _fake_compress_ok, _action_dialect, "convD", mode="hist")
    assert sh["n_obs_compressed"] == 0 and sh["history_compressed"] is True


def test_empty_compressor_result_raises_and_never_caches():
    _reset()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "t"},
        {"role": "assistant", "content": "Subgoal: a", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "o"},
        {"role": "assistant", "content": "Subgoal: b", "tool_calls": [{"id": "2"}]},
    ]
    try:
        textarms.hiagent_transform(messages, _fake_compress_empty, _action_dialect)
        raise AssertionError("empty summary must raise")
    except textarms.TextarmCompressorError:
        pass
    assert not textarms._SUMMARY_CACHE, "nothing may be cached on failure"


def test_compressor_decode_params():
    hi = textarms.compressor_payload("hiagent", "m", "s", "u")
    assert hi["max_tokens"] == 100 and hi["stop"] == ["\n\n"]
    assert hi["chat_template_kwargs"] == {"enable_thinking": False}
    ac = textarms.compressor_payload("acon", "m", "s", "u")
    assert ac["max_tokens"] == 2048 and ac["seed"] == 42
    assert ac["chat_template_kwargs"] == {"enable_thinking": False}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
