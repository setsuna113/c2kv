"""Offline tests for the text-level baseline arms (textarms.py).

No live backend: the compressor callback is a fake that echoes a canned
summary, so segmentation / caching / threshold semantics are checked
without the server.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import textarms


def _action_dialect(message):
    return "tool:" + str(message.get("tool_calls"))


def _fake_compress(log):
    def compress(system, user):
        log.append((system, user))
        return "SUMMARY(" + system[:24] + "|" + str(len(user)) + " chars)"
    return compress


def _conv(messages):
    m = [msg for msg in messages if msg.get("role") == "assistant"]
    segs = 0
    for msg in m:
        if (msg.get("content") or "").strip().startswith("Subgoal:"):
            segs += 1
    return segs


def test_hiagent_segments_and_caches():
    log = []
    compress = _fake_compress(log)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "do the thing"},
        {"role": "assistant", "content": "Subgoal: find data", "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "found it"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "2"}]},
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "Subgoal: finish up", "tool_calls": [{"id": "3"}]},
        {"role": "tool", "content": "done-ish"},
        {"role": "user", "content": "continue"},
    ]
    out, stats = textarms.hiagent_transform(messages, compress, _action_dialect)
    assert stats["policy"] == "hiagent"
    assert stats["n_segments"] == 2 and stats["n_summarized"] == 1
    assert stats["n_compressor_calls"] == 1
    # note injected into system
    assert "Subgoal:" in out[0]["content"] and out[0]["role"] == "system"
    # completed segment replaced by one summary message (after the pre block)
    assert out[1]["content"] == "do the thing" and out[1]["role"] == "user"
    assert out[2]["content"].startswith("Subgoal 1: find data")
    assert "SUMMARY(" in out[2]["content"]
    # current segment stays raw (assistant tool_calls kept)
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in out)
    # re-run: cached, no new compressor call
    _, stats2 = textarms.hiagent_transform(messages, compress, _action_dialect)
    assert stats2["n_compressor_calls"] == 0


def test_hiagent_single_segment_passthrough():
    log = []
    out, stats = textarms.hiagent_transform(
        [{"role": "user", "content": "q"},
         {"role": "assistant", "content": "plain", "tool_calls": [{"id": "x"}]}],
        _fake_compress(log), _action_dialect)
    # no Subgoal declaration yet: 0 segments, everything passes through raw
    assert stats["n_segments"] == 0 and stats["n_summarized"] == 0
    assert stats["n_compressor_calls"] == 0
    assert len(log) == 0
    assert [m["content"] for m in out if m.get("role") == "user"] == ["q"]


def test_acon_obs_threshold():
    log = []
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "content": "x" * 300},            # under threshold
        {"role": "assistant", "content": "", "tool_calls": [{"id": "b"}]},
        {"role": "tool", "content": "y" * (textarms.ACON_OBS_THRESHOLD_CHARS + 10)},
        {"role": "user", "content": "next"},
    ]
    out, stats = textarms.acon_transform(messages, _fake_compress(log),
                                         _action_dialect, "convA")
    assert stats["n_obs_compressed"] == 1
    assert stats["n_compressor_calls"] == 1
    tool_contents = [m["content"] for m in out if m.get("role") == "tool"]
    assert len(tool_contents[0]) == 300          # untouched
    assert tool_contents[1].startswith("SUMMARY(")  # refined
    assert stats["history_compressed"] is False  # below history threshold
    # idempotent: same input -> cache hit, no new call
    _, stats2 = textarms.acon_transform(messages, _fake_compress(log),
                                        _action_dialect, "convA")
    assert stats2["n_compressor_calls"] == 0


def test_acon_history_summary_and_rolling_state():
    log = []
    base = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "long task"},
    ]
    # exceed the history threshold in the prefix, keep last turn; each tool
    # observation stays UNDER the obs threshold (1024 chars) so only the
    # history summarizer fires
    filler = [
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "content": "z" * 900},
    ] * 20
    messages = base + filler + [{"role": "user", "content": "final turn"}]
    out, stats = textarms.acon_transform(messages, _fake_compress(log),
                                         _action_dialect, "convB")
    assert stats["history_compressed"] is True
    assert stats["n_compressor_calls"] == 1
    # system raw + summary + preserved tail, no raw prefix messages
    assert out[0]["role"] == "system"
    assert out[1]["content"].startswith("[CONTEXT SUMMARY]")
    assert out[-1]["content"] == "final turn"
    # prev_summary carried on the next turn (rolling), new compressor call
    messages2 = messages + [{"role": "assistant", "content": "done",
                             "tool_calls": [{"id": "z"}]}]
    _, stats3 = textarms.acon_transform(messages2, _fake_compress(log),
                                        _action_dialect, "convB")
    assert stats3["history_compressed"] is True
    assert stats3["n_compressor_calls"] == 1  # summary for the new prefix only
    # the second call must have seen the previous summary as prev_summary
    assert any("SUMMARY(" in user for _, user in log)


def test_acon_state_reset_on_new_conversation():
    log = []
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
        {"role": "tool", "content": "w" * 6000},
        {"role": "user", "content": "turn"},
    ]
    _, s1 = textarms.acon_transform(messages, _fake_compress(log),
                                    _action_dialect, "convC")
    # same content but a different conversation id: cache still prevents a
    # duplicate call (content-keyed), state tracked separately
    _, s2 = textarms.acon_transform(messages, _fake_compress(log),
                                    _action_dialect, "convD")
    assert s1["n_compressor_calls"] == 1 and s2["n_compressor_calls"] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
    print("ALL PASS")
