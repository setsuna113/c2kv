"""No-model contracts for the finite BFCL frozen-prefix replay gate."""
from __future__ import annotations

import copy

import pytest

from benchmarks.memory_runtime.bfcl_prefix_replay import (
    FrozenPrefixReplay,
    FrozenPrefixReplayError,
)


TASK = "multi_turn_base_16"
TOOLS = [{"type": "function", "function": {"name": "cp", "parameters": {}}}]


def _context(
    step: int,
    *,
    task_id: str = TASK,
    attempt: int = 0,
    benchmark: str = "bfcl",
):
    return {
        "benchmark": benchmark,
        "task_id": task_id,
        "user_turn": 0,
        "step": step,
        "attempt": attempt,
    }


def _messages(step: int):
    messages = [{"role": "user", "content": "copy the report"}]
    if step:
        messages.extend([
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call-prefix", "type": "function",
                "function": {"name": "cp", "arguments": '{"src":"report"}'},
            }], "reasoning_content": None},
            {"role": "tool", "tool_call_id": "call-prefix", "content": "copied"},
        ])
    return messages


def _assistant():
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call-prefix",
            "type": "function",
            "function": {"name": "cp", "arguments": '{"src":"report"}'},
        }],
        "reasoning_content": None,
    }


def _bundle():
    return {
        "task_id": TASK,
        "replay": [{
            "context": _context(0),
            "request_messages": _messages(0),
            "request_tools": TOOLS,
            "assistant_message": _assistant(),
        }],
        "branch": {
            "context": _context(1),
            "request_messages": _messages(1),
            "request_tools": TOOLS,
        },
    }


def _drive(gate, context, messages, tools, live_calls):
    answer = gate.query(context, messages, tools)
    if answer is None:
        live_calls.append((context, messages, tools))
    return answer


@pytest.mark.parametrize(
    "mutation",
    [
        lambda messages, tools: messages[-1].update(content="different observation"),
        lambda messages, tools: messages[-1].update(tool_call_id="call-substituted"),
        lambda messages, tools: tools[0]["function"].update(name="mv"),
    ],
    ids=["observation", "tool-call-id", "tool-schema"],
)
def test_prefix_mismatch_never_reaches_live_callback(mutation):
    gate = FrozenPrefixReplay(_bundle())
    live_calls = []
    assert _drive(gate, _context(0), _messages(0), copy.deepcopy(TOOLS), live_calls) == _assistant()

    messages, tools = _messages(1), copy.deepcopy(TOOLS)
    mutation(messages, tools)
    with pytest.raises(FrozenPrefixReplayError, match="mismatch"):
        _drive(gate, _context(1), messages, tools, live_calls)
    assert live_calls == []
    assert gate.receipt() == {
        "schema": "bfcl-frozen-prefix-replay-v1",
        "task_id": TASK,
        "replayed_requests": 1,
        "branch_reached": False,
        "source_verified": False,
    }


def test_branch_returns_none_once_and_postfork_never_replays():
    bundle = _bundle()
    gate = FrozenPrefixReplay(bundle)
    first = gate.query(_context(0), _messages(0), copy.deepcopy(TOOLS))
    assert first == _assistant()
    first["tool_calls"][0]["id"] = "caller-mutation"

    live_calls = []
    assert _drive(gate, _context(1), _messages(1), copy.deepcopy(TOOLS), live_calls) is None
    assert len(live_calls) == 1
    assert _drive(gate, _context(2), _messages(1), copy.deepcopy(TOOLS), live_calls) is None
    assert len(live_calls) == 2
    assert gate.receipt()["replayed_requests"] == 1
    assert gate.receipt()["branch_reached"] is True
    assert gate.receipt()["source_verified"] is True
    assert bundle["replay"][0]["assistant_message"]["tool_calls"][0]["id"] == "call-prefix"


def test_postbranch_query_rejects_a_different_benchmark():
    gate = FrozenPrefixReplay(_bundle())
    gate.query(_context(0), _messages(0), copy.deepcopy(TOOLS))
    assert gate.query(_context(1), _messages(1), copy.deepcopy(TOOLS)) is None

    with pytest.raises(FrozenPrefixReplayError, match="benchmark bfcl"):
        gate.query(
            _context(2, benchmark="another-benchmark"),
            _messages(1),
            copy.deepcopy(TOOLS),
        )


@pytest.mark.parametrize(
    "context",
    [
        _context(1, task_id="multi_turn_base_165"),
        _context(1, attempt=1),
        {**_context(1), "step": True},
    ],
    ids=["different-task", "nonzero-attempt", "bool-step"],
)
def test_branch_rejects_wrong_task_attempt_and_bool_int_alias(context):
    gate = FrozenPrefixReplay(_bundle())
    gate.query(_context(0), _messages(0), copy.deepcopy(TOOLS))
    with pytest.raises(FrozenPrefixReplayError):
        gate.query(context, _messages(1), copy.deepcopy(TOOLS))
    assert gate.branch_reached is False


def test_prefix_value_types_are_json_strict_not_python_equality():
    bundle = _bundle()
    bundle["replay"][0]["request_messages"][0]["count"] = 1
    gate = FrozenPrefixReplay(bundle)
    messages = _messages(0)
    messages[0]["count"] = True
    with pytest.raises(FrozenPrefixReplayError, match="mismatch"):
        gate.query(_context(0), messages, copy.deepcopy(TOOLS))


def test_constructor_rejects_replay_after_or_at_branch_context():
    bundle = _bundle()
    bundle["branch"]["context"] = _context(0)
    with pytest.raises(FrozenPrefixReplayError, match="branch context"):
        FrozenPrefixReplay(bundle)
