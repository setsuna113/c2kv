"""Behavioral contract for exact-budget HiAgent history."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hiagent_budget


def _measure(messages, history_indices):
    assert history_indices == list(range(
        history_indices[0], history_indices[-1] + 1)) if history_indices else True
    return sum(7 + len(str(messages[i].get("content") or "")) +
               len(json.dumps(messages[i].get("tool_calls") or []))
               for i in history_indices)


def _action(message):
    return json.dumps(message.get("tool_calls") or [], sort_keys=True)


def _call(messages, *, budget=1000, cutoff=None, compress=None,
          measure=_measure, packet=False, model="actor", conv="episode",
          retrieved=None, feedback=None, variant="summary"):
    if cutoff is None:
        cutoff = len(messages) - 1
    if compress is None:
        compress = lambda _: "ok"
    return hiagent_budget.transform(
        messages, compress, _action, conv, model=model,
        budget_tokens=budget, history_cutoff=cutoff, measure=measure,
        preserve_task_packet=packet, retrieved_subgoals=retrieved,
        retrieval_feedback=feedback, variant=variant)


def _call_message(call_id, subgoal=None):
    return {"role": "assistant",
            "content": None if subgoal is None else f"Subgoal: {subgoal}",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": "act", "arguments": "{}"}}]}


def _result(call_id, content):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def setup_function():
    hiagent_budget.reset_state()


def test_no_subgoal_passthrough_is_still_fifo_bounded():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first user"},
        {"role": "assistant", "content": "old action"},
        {"role": "tool", "content": "old observation"},
        {"role": "user", "content": "CURRENT"},
    ]
    full, full_stats = _call(messages)
    assert full[1:] == messages[1:]
    assert full_stats["degenerate"] is True
    assert full_stats["n_compressor_calls"] == 0
    budget = _measure(full, full_stats["history_indices"]) - 10
    out, stats = _call(messages, budget=budget,
                       compress=lambda _: pytest.fail("no subgoal must not summarize"))
    assert stats["budget"]["passed"] is True
    assert stats["budget"]["evicted_records"] >= 1
    assert out[-1] == messages[-1]
    assert "first user" not in json.dumps(out)
    assert _measure(out, stats["history_indices"]) <= budget


def test_active_subgoal_text_survives_long_trajectory_fifo_eviction():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        _call_message("c1", "search"), _result("c1", "O" * 150),
        _call_message("c2"), _result("c2", "P" * 150),
        {"role": "user", "content": "CURRENT " + "C" * 500},
    ]
    out, stats = _call(messages, budget=45)
    assert stats["budget"]["passed"] is True
    assert _measure(out, stats["history_indices"]) <= 45
    assert out[-1] == messages[-1]
    assert {"role": "assistant", "content": "Subgoal: search"} in out
    assert "O" * 150 not in json.dumps(out)
    assert "P" * 150 not in json.dumps(out)


def test_exact_measure_includes_per_message_wrapper_and_synthetic_summary():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "Subgoal: first"},
        {"role": "tool", "content": "observation" * 6},
        {"role": "assistant", "content": "Subgoal: current"},
        {"role": "user", "content": "CURRENT"},
    ]
    out, stats = _call(messages, budget=90)
    assert stats["n_summarized"] >= 1
    assert stats["budget"]["history_before"] == _measure(messages, [1, 2, 3, 4])
    assert stats["budget"]["history_after"] == _measure(out, stats["history_indices"])
    assert stats["budget"]["history_after"] <= 90
    assert all(out[index]["role"] != "system" for index in stats["history_indices"])
    assert any("Summary: ok" in str(out[index].get("content"))
               for index in stats["history_indices"])


def test_complete_retrieval_pinned_and_invalid_id_reported():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "Subgoal: first"},
        {"role": "tool", "content": "FIRST-RAW"},
        {"role": "assistant", "content": "Subgoal: second"},
        {"role": "tool", "content": "SECOND-RAW"},
        {"role": "assistant", "content": "Subgoal: current"},
        {"role": "user", "content": "CURRENT"},
    ]
    out, stats = _call(messages, variant="full", retrieved=[1, 99])
    rendered = json.dumps(out)
    assert "FIRST-RAW" in rendered and "SECOND-RAW" not in rendered
    assert stats["retrieved_subgoals"] == [1]
    assert stats["invalid_retrieval_subgoals"] == [99]
    assert stats["n_summarized"] == 1
    assert "hiagent_retrieve" in out[0]["content"]
    assert out[-1] == messages[-1]


def test_oversize_full_retrieval_raises_distinct_receipt_without_trimming():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "Subgoal: first"},
        {"role": "tool", "content": "R" * 200},
        {"role": "assistant", "content": "Subgoal: current"},
        {"role": "user", "content": "CURRENT"},
    ]
    with pytest.raises(hiagent_budget.RetrievalBudgetExceeded) as raised:
        _call(messages, budget=50, variant="full", retrieved=[1],
              compress=lambda _: pytest.fail("fixed floor checked first"))
    error = raised.value
    assert error.kind == "hiagent_retrieval_budget_exceeded"
    assert error.receipt["retrieved_subgoals"] == [1]
    assert error.budget["passed"] is False
    assert error.budget["fixed_tokens"] > 50


def test_subgoal_ids_are_stable_after_eviction_and_retrieval_uses_archive():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "Subgoal: one"},
        {"role": "tool", "content": "FIRST-TRAJECTORY"},
        {"role": "assistant", "content": "Subgoal: two"},
        {"role": "tool", "content": "SECOND-TRAJECTORY"},
        {"role": "assistant", "content": "Subgoal: three"},
        {"role": "tool", "content": "THIRD-TRAJECTORY"},
        {"role": "user", "content": "CURRENT"},
    ]
    evicted, stats = _call(messages, budget=45)
    assert stats["budget"]["evicted_records"] > 0
    assert "FIRST-TRAJECTORY" not in json.dumps(evicted)
    restored, retrieval_stats = _call(messages, budget=1000, variant="full",
                                      retrieved=[1])
    assert retrieval_stats["retrieved_subgoals"] == [1]
    assert "FIRST-TRAJECTORY" in json.dumps(restored)
    assert any("Subgoal 2: two" in str(m.get("content")) for m in restored)


def test_native_tool_call_and_matching_result_are_evicted_atomically():
    messages = [
        {"role": "user", "content": "task"},
        _call_message("old"), _result("old", "OLD-RESULT"),
        {"role": "assistant", "content": "recent"},
        {"role": "user", "content": "CURRENT"},
    ]
    out, stats = _call(messages, budget=25)
    rendered = json.dumps(out)
    assert '"old"' not in rendered
    assert "OLD-RESULT" not in rendered
    assert stats["budget"]["passed"] is True


def test_tool_call_crossing_cutoff_is_pinned_with_current_result():
    messages = [
        {"role": "user", "content": "task"},
        _call_message("live"),
        _result("live", "CURRENT-RESULT"),
    ]
    out, stats = _call(messages, budget=300, cutoff=2)
    assert out[-2:] == messages[-2:]
    assert stats["budget"]["fixed_tokens"] > 0
    with pytest.raises(hiagent_budget.BudgetExceeded) as raised:
        _call(messages, budget=20, cutoff=2)
    assert raised.value.kind == "hiagent_history_budget_exceeded"
    assert raised.value.receipt["reason"] == "fixed_history_exceeds_budget"


def test_task_packet_excluded_but_other_user_turns_and_feedback_charged():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "TASK PACKET " + "T" * 200},
        {"role": "user", "content": "historical user"},
        {"role": "assistant", "content": "Subgoal: current"},
        {"role": "user", "content": "CURRENT"},
    ]
    out, stats = _call(messages, packet=True, feedback="budget_unavailable")
    assert out[1] == messages[1]
    assert 1 not in stats["history_indices"]
    assert out[-1] == messages[-1]
    assert any("budget_unavailable" in str(out[i].get("content"))
               for i in stats["history_indices"])
    assert stats["budget"]["history_after"] == _measure(out, stats["history_indices"])


def test_summary_cache_is_scoped_to_model_and_trajectory():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "Subgoal: first"},
        {"role": "tool", "content": "observation"},
        {"role": "assistant", "content": "Subgoal: current"},
        {"role": "user", "content": "CURRENT"},
    ]
    calls = []

    def compress(payload):
        calls.append(payload)
        return "ok"

    _, first = _call(messages, compress=compress, model="one")
    _, again = _call(messages, compress=compress, model="one", conv="other")
    _, second = _call(messages, compress=compress, model="two")
    assert first["n_compressor_calls"] == 1
    assert again["n_compressor_calls"] == 0
    assert second["n_compressor_calls"] == 1
    assert [call["model"] for call in calls] == ["one", "two"]
    assert all(call["max_tokens"] == 100 and call["stop"] == ["\n\n"]
               for call in calls)
    hiagent_budget.reset_state()
    _, reset = _call(messages, compress=compress, model="one")
    assert reset["n_compressor_calls"] == 1


def test_fixed_feedback_floor_without_retrieval_raises_generic_budget_error():
    messages = [{"role": "user", "content": "CURRENT"}]
    with pytest.raises(hiagent_budget.BudgetExceeded) as raised:
        _call(messages, budget=10, cutoff=0, feedback="X" * 100)
    assert not isinstance(raised.value, hiagent_budget.RetrievalBudgetExceeded)
    assert raised.value.kind == "hiagent_history_budget_exceeded"
    assert raised.value.budget["fixed_tokens"] > 10


def test_full_arm_reserves_denial_feedback_before_admitting_initial_view():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "Subgoal: first"},
        {"role": "tool", "content": "R" * 200},
        {"role": "assistant", "content": "Subgoal: current"},
        {"role": "user", "content": "CURRENT"},
    ]
    with pytest.raises(hiagent_budget.BudgetExceeded) as floor_error:
        _call(messages, budget=1, variant="full")
    floor_reserved = floor_error.value.budget["history_with_reserved_feedback"]
    _, unconstrained = _call(messages, budget=1000, variant="full")
    full_reserved = unconstrained["budget"]["history_with_reserved_feedback"]
    assert full_reserved > floor_reserved
    budget = floor_reserved + (full_reserved - floor_reserved) // 2
    initial, initial_stats = _call(messages, budget=budget, variant="full")
    assert initial_stats["budget"]["evicted_records"] > 0
    assert initial_stats["budget"]["history_after"] <= budget
    assert initial_stats["budget"]["history_with_reserved_feedback"] <= budget
    assert initial_stats["budget"]["reserve_passed"] is True
    assert initial_stats["budget"]["reserved_feedback_tokens"] > 0
    assert "budget_unavailable" not in json.dumps(initial)
    with pytest.raises(hiagent_budget.RetrievalBudgetExceeded):
        _call(messages, budget=budget, variant="full", retrieved=[1])
    continuation, continued_stats = _call(
        messages, budget=budget, variant="full",
        feedback=hiagent_budget.BUDGET_UNAVAILABLE_FEEDBACK)
    assert continued_stats["budget"]["passed"] is True
    assert continued_stats["budget"]["history_after"] <= budget
    assert continued_stats["budget"]["reserved_feedback_tokens"] == 0
    assert continuation[-1] == messages[-1]
    assert any(hiagent_budget.BUDGET_UNAVAILABLE_FEEDBACK in str(m.get("content"))
               for m in continuation)


def test_full_arm_reserves_already_revealed_feedback():
    messages = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "Subgoal: first"},
        {"role": "tool", "content": "R" * 30},
        {"role": "assistant", "content": "Subgoal: current"},
        {"role": "user", "content": "CURRENT"},
    ]
    _, initial = _call(messages, budget=1000, variant="full", retrieved=[1])
    continued, feedback = _call(
        messages, budget=1000, variant="full",
        retrieved=[1], feedback=hiagent_budget.ALREADY_REVEALED_FEEDBACK)

    assert initial["budget"]["history_with_reserved_feedback"] >= feedback["budget"]["history_after"]
    assert any(hiagent_budget.ALREADY_REVEALED_FEEDBACK in str(message.get("content"))
               for message in continued)


def test_full_arm_fails_initially_if_feedback_reserve_floor_does_not_fit():
    messages = [{"role": "user", "content": "CURRENT"}]
    with pytest.raises(hiagent_budget.BudgetExceeded) as raised:
        _call(messages, budget=10, cutoff=0, variant="full")
    assert not isinstance(raised.value, hiagent_budget.RetrievalBudgetExceeded)
    assert raised.value.receipt["reason"] == "fixed_history_exceeds_budget"
    assert raised.value.budget["fixed_tokens"] <= 10
    assert raised.value.budget["history_with_reserved_feedback"] > 10
