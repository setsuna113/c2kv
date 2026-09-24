"""ToolSandbox's rebuilt tool-call echo must not abort a persistent RACER cell.

box4 p2off2, toolsandbox__racer_{commitkv,h2o,pyramidkv}_off_b256 at
add_reminder_content_and_date_and_time_alt_3_distraction_tools: the draft emitted
prose plus a datetime_info_to_timestamp call, ToolSandbox echoed the call with
empty content, and the next decision failed "RACER next source does not echo
the committed action".
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from history_memory.packing import MemoryView
from benchmarks.memory_runtime import budget_guard, event_native_step
from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_draft import NativeDraft
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner, EventNativeStepError
from benchmarks.memory_runtime.racer.allocator import PersistentMemory
from benchmarks.memory_runtime.racer.generator import PersistentRacerGenerator
from benchmarks.memory_runtime.tests.test_racer_regeneration_capacity import Backend, Engine

PROSE = "I'll convert that date and time to a POSIX timestamp first."
RAW = PROSE + "\n<tool_call>raw generated call</tool_call>"
CALL = {"id": "d1_r0_0", "type": "function", "function": {
    "name": "datetime_info_to_timestamp", "arguments": '{"year":2024,"month":3}'}}
FIRST = [{"role": "system", "content": "Use the tools."},
         {"role": "user", "content": "Remind me to buy chocolate milk."}]
TOOL = {"role": "tool", "content": "1711126800.0", "name": "datetime_info_to_timestamp",
        "tool_call_id": "d1_r0_0"}


def _echo(arguments=None):
    """The assistant message ToolSandbox sends back: calls only, empty content."""
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": "d1_r0_0", "type": "function", "function": {
            "name": "datetime_info_to_timestamp",
            "arguments": arguments or {"year": 2024, "month": 3}}}]}


class Controller:
    def prepare(self, payload, *, ratio, max_new_tokens):
        messages = tuple(copy.deepcopy(payload["messages"]))
        memory = PersistentMemory(
            MemoryView((), ()), (), (), (), (), source_messages=messages,
            history_message_count=len(messages), history_start_message_count=1,
            history_budget_tokens=256, retained_history_cap=64, common_tokens=8,
            full_history_tokens=64, backend_identity="h2o")
        return SimpleNamespace(memory=memory, metadata={"decision_index": 0})

    def reconsider(self, prepared, draft_tool_calls, *, draft_text, parse_error):
        return {"regenerate": False, "memory": prepared.memory,
                "metadata": prepared.metadata,
                "decision": {"status": "abstain", "reason": "recovery_off"}}


def _runner(tmp_path, monkeypatch, benchmark):
    monkeypatch.setattr(event_native_step, "memory_to_dict", lambda value: {})
    monkeypatch.setattr(budget_guard, "history_budget_receipt",
                        lambda *args, **kwargs: {"status": "passed", "errors": []})
    monkeypatch.setattr(event_native_step, "decode_native_generation", lambda *args, **kwargs: (
        NativeDraft(RAW, PROSE, (copy.deepcopy(CALL),), "tool_calls", "complete_native_calls")))
    engine = Engine(None)
    config = SimpleNamespace(
        allocation="backend_native_persistent", history_budget_tokens=256,
        history_spec=lambda target: {"method": "h2o", "backend": "physical_eviction",
                                     "target_tokens": target},
        receipt=lambda: {"schema": "racer-backend-v2", "identity": "racer:v2:h2o:bare:off:b256"})
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: RAW)
    generator = PersistentRacerGenerator(engine, tokenizer, config, backend=Backend(),
                                         benchmark=benchmark)
    runner = EventNativeDecisionRunner(
        Controller(), generator, tokenizer, ratio=8, max_new_tokens=16,
        max_generation_calls=8, journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    return runner, engine


def _second(runner, echo):
    return runner.run({"session_id": "ts", "decision_key": "turn-0/step-1",
                       "messages": [*FIRST, echo, TOOL]})


def test_toolsandbox_calls_only_echo_commits_the_raw_generated_action(tmp_path, monkeypatch):
    runner, engine = _runner(tmp_path, monkeypatch, "toolsandbox")
    first = runner.run({"session_id": "ts", "decision_key": "turn-0/step-0", "messages": FIRST})
    assert first["response"]["content"] == PROSE
    assert first["backend_commit"]["resolution_on_next_decision"] == "commit"
    second = _second(runner, _echo())
    assert second["status"] == "ok"
    sent = engine.chats[-1]
    assert sent["c2kv_kv_memory_hint"]["persistent_history_session"]["transaction"] == {
        "decision_id": "turn-0/step-1", "phase": "draft", "resolution": "commit"}
    # The engine continues from the exact generated action, prose included.
    assert {"role": "assistant", "content": RAW} in sent["messages"]


@pytest.mark.parametrize(("benchmark", "echo"), [
    (None, _echo()),                                      # other harnesses keep the strict echo
    ("bfcl", _echo()),
    ("toolsandbox", _echo({"year": 2025, "month": 3})),   # a changed call still fails closed
])
def test_other_echo_mismatches_still_fail_closed(tmp_path, monkeypatch, benchmark, echo):
    runner, _ = _runner(tmp_path, monkeypatch, benchmark)
    runner.run({"session_id": "ts", "decision_key": "turn-0/step-0", "messages": FIRST})
    with pytest.raises(EventNativeStepError, match="does not echo the committed action"):
        _second(runner, echo)
