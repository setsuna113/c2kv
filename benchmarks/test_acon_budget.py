"""Behavioral contract for the bounded ACON history arm."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import acon_budget
import textarms


def _measure(messages, history_indices):
    """Exact test serialization: one character is one test token."""
    assert history_indices == list(range(
        history_indices[0], history_indices[-1] + 1)) if history_indices else True
    return sum(len(str(messages[i].get("content") or "")) +
               len(json.dumps(messages[i].get("tool_calls") or []))
               for i in history_indices)


def _action(message):
    return "tool:" + json.dumps(message.get("tool_calls") or [])


def _messages(observation="O" * 90):
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "a0"},
        {"role": "tool", "content": observation},
        {"role": "assistant", "content": "a1"},
        {"role": "tool", "content": "P" * 90},
        {"role": "assistant", "content": "last action"},
        {"role": "user", "content": "CURRENT"},
    ]


def _call(messages, compress, *, conv="one", budget=100, cutoff=None,
          model="actor", guideline="ut_co", packet=False):
    if cutoff is None:
        cutoff = len(messages) - 1
    return acon_budget.transform(
        messages, compress, _action, conv, model=model,
        budget_tokens=budget, history_cutoff=cutoff, measure=_measure,
        preserve_task_packet=packet, guideline=guideline)


def setup_function():
    acon_budget.reset_state()


def test_undercap_raw_passes_without_compressor_or_rewrite():
    messages = _messages()
    out, stats = _call(messages, lambda _: pytest.fail("compressor called"),
                       budget=1000)
    assert out == messages
    assert stats["n_compressor_calls"] == 0
    assert stats["history_compressed"] is False
    assert stats["budget"]["attempts"] == 0
    assert stats["budget"]["history_after"] == _measure(
        out, stats["history_indices"])


def test_overflow_below_original_threshold_folds_history_and_keeps_raw_tail():
    messages = _messages()
    assert textarms._message_chars(messages) < textarms.ACON_HISTORY_THRESHOLD_CHARS
    calls = []

    def compress(payload):
        calls.append(payload)
        return "short summary"

    out, stats = _call(messages, compress)
    assert len(calls) == 1
    assert calls[0]["model"] == "actor"
    assert calls[0]["max_tokens"] <= 100
    assert "Summary Compression Rules:" in calls[0]["messages"][-1]["content"]
    assert stats["history_compressed"] is True
    assert stats["budget"]["passed"] is True
    assert _measure(out, stats["history_indices"]) <= 100
    assert out[1]["content"].startswith("task\n<HISTORY_SUMMARY>")
    assert out[-2:] == messages[-2:]
    assert "O" * 90 not in json.dumps(out)


def test_rolling_summary_shows_new_raw_then_folds_all_new_messages():
    prompts = []

    def compress(payload):
        prompts.append(payload["messages"][-1]["content"])
        return f"summary{len(prompts)}"

    first = _messages()
    _, stats1 = _call(first, compress, budget=100)
    assert stats1["n_compressor_calls"] == 1
    second = first + [
        {"role": "assistant", "content": "next action"},
        {"role": "tool", "content": "N" * 60},
    ]
    out2, stats2 = _call(second, compress, budget=100, cutoff=9)
    assert stats2["n_compressor_calls"] == 0
    assert stats2["history_compressed"] is False
    assert any(m.get("content") == "CURRENT" for m in out2)
    assert any(m.get("content") == "next action" for m in out2)

    third = second + [
        {"role": "assistant", "content": "final action"},
        {"role": "tool", "content": "latest"},
    ]
    out3, stats3 = _call(third, compress, budget=100, cutoff=11)
    assert stats3["n_compressor_calls"] == 1
    assert "[PREVIOUS SUMMARY] summary1" in prompts[-1]
    assert "CURRENT" in prompts[-1] and "N" * 60 in prompts[-1]
    assert "O" * 90 not in prompts[-1]
    assert out3[-2:] == third[-2:]


def test_valid_previous_summary_survives_a_raw_undercap_measurement():
    messages = _messages()
    _call(messages, lambda _: "short", budget=100)
    out, stats = acon_budget.transform(
        messages, lambda _: pytest.fail("compressor called"), _action,
        "one", model="actor", budget_tokens=100,
        history_cutoff=len(messages) - 1,
        measure=lambda view, indices: _measure(view, indices) // 3)
    assert stats["budget"]["history_before"] <= 100
    assert stats["n_compressor_calls"] == 0
    assert "<HISTORY_SUMMARY>" in out[1]["content"]


def test_previous_summary_can_be_recompressed_without_new_foldable_raw():
    summaries = iter(["X" * 25, "tiny"])
    prompts = []

    def compress(payload):
        prompts.append(payload["messages"][-1]["content"])
        return next(summaries)

    messages = _messages()
    _call(messages, compress, budget=100)
    larger_tail = _messages()
    larger_tail[-2]["content"] = "R" * 35
    out, stats = _call(larger_tail, compress, budget=100)
    assert stats["history_compressed"] is True
    assert stats["n_compressor_calls"] == 1
    assert "[PREVIOUS SUMMARY] " + "X" * 25 in prompts[-1]
    assert out[-2] == larger_tail[-2]
    assert stats["budget"]["passed"] is True


def test_final_candidate_is_remeasured_and_retried_with_tighter_cap():
    caps = []

    def compress(payload):
        caps.append(payload["max_tokens"])
        return "X" * 100 if len(caps) == 1 else "small"

    out, stats = _call(_messages(), compress, budget=100)
    assert len(caps) == 2 and caps[1] < caps[0]
    assert stats["budget"]["attempts"] == 2
    assert stats["budget"]["history_after"] == _measure(
        out, stats["history_indices"])
    assert stats["budget"]["passed"] is True


def test_synthetic_summary_is_history_when_output_has_no_assistant():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "tool", "content": "O" * 160},
        {"role": "assistant", "content": "action"},
        {"role": "user", "content": "earlier observation"},
        {"role": "user", "content": "CURRENT"},
    ]
    out, stats = _call(messages, lambda _: "short", budget=90, cutoff=4)
    assert all(m["role"] != "assistant" for m in out)
    assert stats["history_indices"] == [1]
    assert out[2]["content"] == "earlier observation"
    assert stats["budget"]["history_after"] == _measure(
        out, stats["history_indices"])


def test_appworld_task_packet_is_exact_and_summary_is_charged():
    messages = _messages()
    messages[1]["content"] = "TASK PACKET " + "T" * 120
    out, stats = _call(messages, lambda _: "short", packet=True, budget=90)
    assert out[1] == messages[1]
    assert out[2]["content"].startswith("<HISTORY_SUMMARY>")
    assert stats["history_indices"][0] == 2
    assert 1 not in stats["history_indices"]
    assert _measure(out, stats["history_indices"]) > 0
    assert stats["budget"]["passed"] is True


def test_required_tail_over_budget_fails_without_state_commit():
    messages = _messages()
    messages[-2]["content"] = "F" * 200
    calls = []
    with pytest.raises(acon_budget.BudgetExceeded) as raised:
        _call(messages, lambda payload: calls.append(payload) or "short",
              budget=100)
    assert raised.value.kind == "acon_history_budget_exceeded"
    assert raised.value.receipt["reason"] == "fixed_history_exceeds_budget"
    assert raised.value.budget["passed"] is False
    assert calls == []
    assert acon_budget._STATE == {}


def test_three_failed_attempts_leave_state_uncommitted():
    caps = []

    def compress(payload):
        caps.append(payload["max_tokens"])
        return "X" * 200

    with pytest.raises(acon_budget.BudgetExceeded) as raised:
        _call(_messages(), compress, budget=100)
    assert len(caps) == 3 and caps == sorted(caps, reverse=True)
    assert raised.value.receipt["reason"] == "compression_attempts_exhausted"
    assert raised.value.budget["attempts"] == 3
    assert acon_budget._STATE == {}


def test_budget_model_guideline_and_prompt_isolate_cache_and_state():
    calls = []
    original_acon_state = dict(textarms._ACON_STATE)

    def compress(payload):
        calls.append(payload)
        return "short"

    messages = _messages()
    _call(messages, compress, conv="shared", budget=100)
    _call(messages, compress, conv="shared", budget=100)
    assert len(calls) == 1
    _call(messages, compress, conv="shared", budget=101)
    _call(messages, compress, conv="shared", budget=100, model="other")
    _call(messages, compress, conv="shared", budget=100, guideline="ut")
    changed = _messages("Z" * 90)
    _call(changed, compress, conv="shared", budget=100)
    assert len(calls) == 5
    assert "Z" * 90 in calls[-1]["messages"][-1]["content"]
    assert textarms._ACON_STATE == original_acon_state
