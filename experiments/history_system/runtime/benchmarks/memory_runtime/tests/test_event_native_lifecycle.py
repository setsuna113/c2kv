"""Model-free lifecycle seams spanning events, rebuilt views, and decisions."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


RUNTIME_ROOT = Path(__file__).resolve().parents[3]
HISTORY_PYTHON = RUNTIME_ROOT / "python"
if str(HISTORY_PYTHON) not in sys.path:
    sys.path.insert(0, str(HISTORY_PYTHON))

from benchmarks.memory_runtime.attempt_journal import (
    AttemptJournal,
    summarize_attempt_journal,
)
from benchmarks.memory_runtime.event_native_exact_policy import (
    EventNativeExactController,
)
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.tests.test_event_native_policy import (
    Tokenizer as NativeTokenizer,
    contracts,
)
from benchmarks.memory_runtime.tests.test_event_native_step import Generator, tool
from history_memory.events import EventStore


class Tokenizer(NativeTokenizer):
    def decode(self, ids, **kwargs):
        assert kwargs["skip_special_tokens"] is False
        return "".join(map(chr, ids))


def _native_call(name: str, arguments: dict[str, str]) -> str:
    value = {"name": name, "arguments": arguments}
    return "<tool_call>" + json.dumps(value, separators=(",", ":")) + "</tool_call>"


def _assistant_message(step: dict) -> dict:
    response = step["response"]
    return {
        "role": "assistant",
        "content": response["content"],
        "tool_calls": copy.deepcopy(response["tool_calls"]),
    }


def _make_runner(tmp_path, outputs, *, history_budget_bytes: int = 1_000_000):
    packing, policy = contracts()
    packing["max_target_tokens"] = 256
    policy["history_budget_bytes"] = history_budget_bytes
    journal_path = tmp_path / "attempts.jsonl"
    generator = Generator(journal_path, outputs)
    controller = EventNativeExactController(
        Tokenizer(),
        packing=packing,
        policy=policy,
        mode="capacity_exact_persistent",
        model_context=packing["max_sequence_tokens"],
    )
    runner = EventNativeDecisionRunner(
        controller,
        generator,
        Tokenizer(),
        ratio=8,
        max_new_tokens=256,
        max_generation_calls=len(outputs),
        journal=AttemptJournal(journal_path),
    )
    return runner, generator, journal_path


def _payload(session_id: str, decision_key: str, messages: list[dict]) -> dict:
    return {
        "session_id": session_id,
        "decision_key": decision_key,
        "messages": copy.deepcopy(messages),
        "tools": [],
    }


def _represented_source_indices(memory) -> set[int]:
    return set(memory.raw_source_indices) | {
        source_index
        for chunk in memory.chunks
        for source_index in chunk.source_indices
    }


def _workspace_text(memory) -> str:
    return "".join(map(chr, memory.workspace_input_ids))


def _recovery_prefix() -> list[dict]:
    return [
        {"role": "system", "content": "Use visible source records."},
        {"role": "user", "content": "Start."},
        {
            "role": "assistant",
            "content": "Archived record item-17 has value violet.",
        },
        {"role": "assistant", "content": "Background alpha " + "A" * 800},
        {"role": "assistant", "content": "Background beta " + "B" * 800},
        {"role": "assistant", "content": "Background gamma " + "C" * 800},
        {"role": "user", "content": "Continue with the record."},
    ]


def test_native_success_and_error_results_are_completed_in_the_next_view(tmp_path):
    outputs = [
        _native_call("read_file", {"path": "ok.txt"}),
        _native_call("read_file", {"path": "missing.txt"}),
        "Both outcomes observed.",
    ]
    runner, generator, journal_path = _make_runner(tmp_path, outputs)
    session_id = "lifecycle-success-error"
    messages = [
        {"role": "system", "content": "Inspect files."},
        {"role": "user", "content": "Read ok.txt."},
    ]

    success_call = runner.run(_payload(session_id, "d1", messages))
    success_message = _assistant_message(success_call)
    success_id = success_message["tool_calls"][0]["id"]
    assert success_message["tool_calls"][0]["function"]["name"] == "read_file"
    messages.extend(
        [
            success_message,
            {
                "role": "tool",
                "tool_call_id": success_id,
                "content": '{"status":"success","content":"alpha"}',
            },
            {"role": "user", "content": "Now read missing.txt."},
        ]
    )

    error_call = runner.run(_payload(session_id, "d2", messages))
    error_message = _assistant_message(error_call)
    error_id = error_message["tool_calls"][0]["id"]
    messages.extend(
        [
            error_message,
            {
                "role": "tool",
                "tool_call_id": error_id,
                "content": '{"status":"error","error":"not found"}',
            },
            {"role": "user", "content": "Summarize both outcomes."},
        ]
    )

    final = runner.run(_payload(session_id, "d3", messages))
    store = EventStore.from_messages(session_id, messages)
    tool_events = [event for event in store.events if event.kind == "tool_event"]
    assert [(event.complete, event.source_indices) for event in tool_events] == [
        (True, (2, 3)),
        (True, (5, 6)),
    ]
    assert [
        store.event_messages(event.event_id)[-1].to_dict()["content"]
        for event in tool_events
    ] == [
        '{"status":"success","content":"alpha"}',
        '{"status":"error","error":"not found"}',
    ]
    assert final["generation_trace"][0]["controller"]["capacity_gate"][
        "full_identity_bypass"
    ]
    assert _represented_source_indices(generator.inputs[-1]) == set(range(len(messages)))
    rendered = _workspace_text(generator.inputs[-1])
    assert "alpha" in rendered and "not found" in rendered
    journal = summarize_attempt_journal(journal_path)
    assert (journal["started"], journal["completed"]) == (3, 3)
    assert journal["failed"] == journal["pending"] == 0
    assert journal["finished_usage_totals"]["completion_tokens"] == sum(
        map(len, outputs)
    )


def test_parallel_reverse_results_bind_by_call_id_in_the_rebuilt_view(tmp_path):
    parallel = _native_call("lookup", {"id": "alpha"}) + _native_call(
        "lookup", {"id": "beta"}
    )
    runner, generator, journal_path = _make_runner(
        tmp_path, [parallel, "Reverse arrivals observed."]
    )
    session_id = "lifecycle-parallel-reverse"
    messages = [
        {"role": "system", "content": "Keep call identity."},
        {"role": "user", "content": "Look up alpha and beta in parallel."},
    ]
    first = runner.run(_payload(session_id, "d1", messages))
    assistant = _assistant_message(first)
    alpha_id, beta_id = [call["id"] for call in assistant["tool_calls"]]
    messages.extend(
        [
            assistant,
            {
                "role": "tool",
                "tool_call_id": beta_id,
                "content": "beta-result-arrived-first",
            },
            {
                "role": "tool",
                "tool_call_id": alpha_id,
                "content": "alpha-result-arrived-second",
            },
            {"role": "user", "content": "Use both results."},
        ]
    )
    second = runner.run(_payload(session_id, "d2", messages))

    store = EventStore.from_messages(session_id, messages)
    event = next(event for event in store.events if event.kind == "tool_event")
    assert event.complete and event.tool_call_ids == (alpha_id, beta_id)
    assert event.source_indices == (2, 3, 4)
    result_by_id = {
        message.to_dict()["tool_call_id"]: message.to_dict()["content"]
        for message in store.event_messages(event.event_id)[1:]
    }
    assert result_by_id == {
        beta_id: "beta-result-arrived-first",
        alpha_id: "alpha-result-arrived-second",
    }
    assert _represented_source_indices(generator.inputs[-1]) == set(range(len(messages)))
    rendered = _workspace_text(generator.inputs[-1])
    assert rendered.index("beta-result-arrived-first") < rendered.index(
        "alpha-result-arrived-second"
    )
    assert second["status"] == "ok"
    journal = summarize_attempt_journal(journal_path)
    assert (journal["completed"], journal["failed"], journal["pending"]) == (2, 0, 0)


def test_appended_revision_preserves_prefix_and_cancels_recovered_lease(tmp_path):
    outputs = [tool("item-17"), "Original requirement noted.", "Revision noted."]
    runner, generator, journal_path = _make_runner(
        tmp_path,
        outputs,
        history_budget_bytes=1800 * 64,
    )
    session_id = "lifecycle-revision"
    messages = _recovery_prefix()
    frozen_prefix = copy.deepcopy(messages)
    first = runner.run(_payload(session_id, "d1", messages))
    assert first["exact_recovery"]["candidate_event_id"] == f"{session_id}:m2"
    assert first["exact_recovery"]["upgrade_count"] == 1
    messages.extend(
        [
            _assistant_message(first),
            {
                "role": "user",
                "content": (
                    "Actually, replace the archive requirement with folder revised-22."
                ),
            },
        ]
    )
    second = runner.run(_payload(session_id, "d2", messages))

    assert messages[: len(frozen_prefix)] == frozen_prefix
    old_store = EventStore.from_messages(session_id, frozen_prefix)
    new_store = EventStore.from_messages(session_id, messages)
    assert new_store.events[: len(old_store.events)] == old_store.events
    controller = second["generation_trace"][0]["controller"]
    assert controller["revision_cancelled_event_ids"] == [f"{session_id}:m2"]
    assert controller["retained_event_ids"] == []
    latest_user_index = len(messages) - 1
    assert latest_user_index in generator.inputs[-1].raw_source_indices
    assert "Actually, replace the archive requirement" in _workspace_text(
        generator.inputs[-1]
    )
    journal = summarize_attempt_journal(journal_path)
    assert (journal["completed"], journal["failed"], journal["pending"]) == (3, 0, 0)


def test_clean_recovery_result_enters_the_next_decision(tmp_path):
    outputs = [tool("item-17"), tool("archive"), "Archive result observed."]
    runner, generator, journal_path = _make_runner(
        tmp_path,
        outputs,
        history_budget_bytes=1800 * 64,
    )
    session_id = "lifecycle-recovery-result"
    messages = _recovery_prefix()
    recovered = runner.run(_payload(session_id, "d1", messages))
    final_action = _assistant_message(recovered)
    call_id = final_action["tool_calls"][0]["id"]
    messages.extend(
        [
            final_action,
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": '{"status":"success","created":"archive"}',
            },
            {"role": "user", "content": "Report the created archive result."},
        ]
    )
    continued = runner.run(_payload(session_id, "d2", messages))

    assert recovered["status"] == continued["status"] == "ok"
    assert recovered["generation_attempts"] == 2
    assert recovered["generation_trace"][0]["discarded"] is True
    assert recovered["exact_recovery"]["candidate_event_id"] == f"{session_id}:m2"
    store = EventStore.from_messages(session_id, messages)
    final_tool_event = next(
        event
        for event in store.events
        if event.kind == "tool_event" and call_id in event.tool_call_ids
    )
    assert final_tool_event.complete and final_tool_event.source_indices == (7, 8)
    assert set(final_tool_event.source_indices) <= _represented_source_indices(
        generator.inputs[-1]
    )
    controller = continued["generation_trace"][0]["controller"]
    assert f"{session_id}:m2" in controller["retained_event_ids"]
    assert continued["response"]["content"] == "Archive result observed."
    journal = summarize_attempt_journal(journal_path)
    assert (journal["started"], journal["completed"]) == (3, 3)
    assert journal["failed"] == journal["pending"] == 0
