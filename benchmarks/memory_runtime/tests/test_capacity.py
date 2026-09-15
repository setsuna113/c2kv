from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "benchmarks"))

from memory_runtime.capacity import measure_full_history


def test_full_history_charges_tool_return_but_not_system_or_current_query():
    full = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "old query"},
        {"role": "assistant", "content": "Action: read"},
        {"role": "user", "content": "tool return"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "current query"},
    ]
    tools = [{"type": "function", "function": {"name": "read"}}]
    calls = []

    def count(messages, actual_tools):
        calls.append((messages, actual_tools))
        return len(messages)

    measured = measure_full_history(
        full, {"current_start_out_index": 5}, count, tools, 8)
    assert measured == {
        "total_raw_prompt_tokens": 6,
        "common_raw_prompt_tokens": 2,
        "raw_history_tokens": 4,
        "active_history_bytes": 32,
    }
    assert [message["content"] for message in calls[1][0]] == [
        "instructions", "current query"]
    assert calls[0][1] is tools and calls[1][1] is tools


@pytest.mark.parametrize("cutoff", [-1, 3, 1.0, True, None])
def test_rejects_invalid_full_boundary(cutoff):
    with pytest.raises(ValueError, match="current_start_out_index"):
        measure_full_history(
            [{"role": "user", "content": "q"}],
            {"current_start_out_index": cutoff}, len, [], 1)


def test_rejects_gist_carrier_and_negative_history_delta():
    with pytest.raises(ValueError, match="gist carriers"):
        measure_full_history(
            [{"role": "user", "content": "old", "c2kv_key_hash": "k"}],
            {"current_start_out_index": 1}, lambda messages, tools: 1, [], 1)

    counts = iter([1, 2])
    with pytest.raises(ValueError, match="history token delta became negative"):
        measure_full_history(
            [{"role": "user", "content": "old"}],
            {"current_start_out_index": 1}, lambda messages, tools: next(counts), [], 1)

    with pytest.raises(ValueError, match="nonnegative integer"):
        measure_full_history(
            [{"role": "user", "content": "old"}],
            {"current_start_out_index": 1}, lambda messages, tools: -1, [], 1)
