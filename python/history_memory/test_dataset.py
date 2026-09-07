"""CPU-only tests for observable decision and paired-view construction."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

from history_memory.dataset import (
    LifecycleSelection,
    build_paired_records,
    build_static_records,
    iter_decisions,
)
from history_memory.packing import MemoryView


_ARGUMENTS = '{"city":"München", "units": "C"}'
_STORE_ID = '["fixture-source","session-1"]'


def _row(*, split: str = "train", future: str = "final") -> dict:
    return {
        "session_id": "session-1",
        "source": "fixture-source",
        "split": split,
        "task_id": "task-1",
        "template_id": "template-1",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                    "x-visible-extension": {"keep": True},
                },
            }
        ],
        "messages": [
            {"role": "system", "content": "Use tools.", "reward": 99},
            {"role": "user", "content": "Weather?", "expected_answer": "hidden"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": _ARGUMENTS},
                    }
                ],
                "grader_score": 1,
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "18 C",
                "latency_ms": 20,
            },
            {"role": "user", "content": "And tomorrow?", "gold": "hidden"},
            {"role": "assistant", "content": future, "reward": 1},
        ],
    }


def test_decisions_preserve_roles_tools_and_exact_target_arguments():
    decisions = tuple(iter_decisions(_row()))

    assert len(decisions) == 2
    assert decisions[0].source_message_index == 2
    assert decisions[0].decision_index == 0
    assert decisions[0].target.to_dict()["tool_calls"][0]["function"]["arguments"] == _ARGUMENTS
    assert "grader_score" not in decisions[0].target.to_dict()

    later = decisions[1]
    assert [message.role for message in later.prefix] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert later.store.event(f"{_STORE_ID}:m2").source_indices == (2, 3)
    assert later.store.event(f"{_STORE_ID}:m2").complete is True
    assert "reward" not in later.prefix[0].to_dict()
    assert "expected_answer" not in later.prefix[1].to_dict()
    assert "latency_ms" not in later.prefix[3].to_dict()
    assert "gold" not in later.prefix[4].to_dict()
    assert decisions[0].tools[0]["function"]["x-visible-extension"] == {"keep": True}


def test_tool_results_are_never_targets_and_empty_assistant_is_skipped():
    row = _row()
    row["messages"].insert(4, {"role": "assistant", "content": None, "tool_calls": []})
    decisions = tuple(iter_decisions(row))

    assert len(decisions) == 2
    assert all(decision.target.role == "assistant" for decision in decisions)
    assert {decision.source_message_index for decision in decisions} == {2, 6}


def test_future_change_cannot_change_earlier_planner_input():
    class CapturingPlanner:
        def __init__(self):
            self.inputs = []

        def __call__(self, store, decision_index):
            self.inputs.append(
                (
                    decision_index,
                    tuple(message.to_dict() for message in store.messages),
                    tuple(
                        (event.event_id, event.kind, event.source_indices, event.complete)
                        for event in store.events
                    ),
                )
            )
            return LifecycleSelection()

    first = CapturingPlanner()
    second = CapturingPlanner()
    build_paired_records([_row(future="version A")], first)
    build_paired_records([_row(future="version B")], second)

    assert first.inputs[0] == second.inputs[0]
    assert first.inputs[0][0] == 0
    assert [message["role"] for message in first.inputs[0][1]] == ["system", "user"]
    assert all(
        set(message) <= {"role", "content", "name", "tool_call_id", "tool_calls"}
        for _, messages, _ in first.inputs
        for message in messages
    )
    assert all("version A" not in json.dumps(item, ensure_ascii=False) for item in first.inputs[:-1])


def test_paired_records_match_targets_weights_and_repetitions_and_call_once():
    class Planner:
        def __init__(self):
            self.calls = 0

        def __call__(self, store, decision_index):
            self.calls += 1
            return LifecycleSelection()

    planner = Planner()
    records = build_paired_records([_row()], planner, repetitions=3)

    assert planner.calls == 2
    assert len(records) == 12
    grouped = defaultdict(list)
    for record in records:
        grouped[(record.decision_id, record.repetition_index)].append(record)
    assert len(grouped) == 6
    for pair in grouped.values():
        assert {record.arm for record in pair} == {"C", "B"}
        assert {record.weight for record in pair} == {1.0}
        assert pair[0].target.to_dict() == pair[1].target.to_dict()


def test_stateful_planner_can_recover_retain_then_evict_across_prefixes():
    row = {
        "session_id": "life",
        "source": "fixture-source",
        "split": "train",
        "task_id": "life-task",
        "template_id": "life-template",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old evidence"},
            {"role": "assistant", "content": "answer zero"},
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "answer one"},
            {"role": "user", "content": "question two"},
            {"role": "assistant", "content": "answer two"},
        ],
    }

    class LeasePlanner:
        def __init__(self):
            self.live = set()
            self.transitions = []

        def __call__(self, store, decision_index):
            old_evidence = '["fixture-source","life"]:m1'
            if decision_index == 0:
                self.live.add(old_evidence)
                self.transitions.append("recover")
            elif decision_index == 1:
                self.transitions.append("retain")
            else:
                self.live.remove(old_evidence)
                self.transitions.append("evict")
            return LifecycleSelection(restored_event_ids=tuple(sorted(self.live)))

    planner = LeasePlanner()
    records = build_paired_records([row], planner)
    lifecycle = [record for record in records if record.arm == "B"]

    assert planner.transitions == ["recover", "retain", "evict"]
    old_evidence = '["fixture-source","life"]:m1'
    assert old_evidence in lifecycle[0].view.raw_event_ids
    assert old_evidence in lifecycle[1].view.raw_event_ids
    assert old_evidence in lifecycle[2].view.gist_event_ids


def test_direct_lifecycle_view_is_validated_against_current_prefix():
    def invalid_planner(store, decision_index):
        return MemoryView(gist_event_ids=(), raw_event_ids=("future:m99",))

    with pytest.raises(ValueError, match="View must cover exactly.*unknown"):
        build_paired_records([_row()], invalid_planner)


def test_event_identity_is_scoped_by_source_family():
    def source_row(source, split):
        return {
            "session_id": "s1",
            "source": source,
            "split": split,
            "task_id": "same-task",
            "template_id": "same-template",
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        }

    records = build_static_records(
        [source_row("family-a", "train"), source_row("family-b", "eval")]
    )

    assert [record.decision.session_id for record in records] == ["s1", "s1"]
    assert records[0].decision.store.session_id == '["family-a","s1"]'
    assert records[1].decision.store.session_id == '["family-b","s1"]'
    assert records[0].decision.store.events[0].event_id != records[1].decision.store.events[0].event_id


def test_legacy_function_call_is_rejected_instead_of_silently_dropped():
    row = _row()
    row["messages"][2] = {
        "role": "assistant",
        "content": None,
        "function_call": {"name": "weather", "arguments": _ARGUMENTS},
    }
    # Make the unsupported message a visible prefix for a later target.
    row["messages"][3:5] = []

    with pytest.raises(ValueError, match="Legacy function_call"):
        tuple(iter_decisions(row))


def test_split_leakage_is_rejected_before_any_event_expansion():
    malformed = {
        "session_id": "s-train",
        "source": "source",
        "split": "train",
        "task_id": "task-train",
        "template_id": "shared-template",
        # Expanding this row would fail on the unmatched tool result.
        "messages": [
            {"role": "tool", "tool_call_id": "missing", "content": "bad"},
            {"role": "assistant", "content": "target"},
        ],
    }
    conflicting = {
        "session_id": "s-eval",
        "source": "source",
        "split": "eval",
        "task_id": "task-eval",
        "template_id": "shared-template",
        "messages": [{"role": "assistant", "content": "target"}],
    }

    with pytest.raises(ValueError, match="Group split leakage.*template_id"):
        build_static_records([malformed, conflicting])


def test_split_is_required_even_for_direct_decision_iteration():
    row = _row()
    del row["split"]
    with pytest.raises(ValueError, match="split.*before expansion"):
        tuple(iter_decisions(row))


def test_static_cli_exports_visible_json_without_argument_rewrite(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "decisions.jsonl"
    input_path.write_text(json.dumps(_row(), ensure_ascii=False) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "agent" / "build_history_memory_data.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )
    summary = json.loads(completed.stdout)
    exported = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert summary["decisions_written"] == 2
    assert {record["arm"] for record in exported} == {"C"}
    assert exported[0]["event_store_session_id"] == _STORE_ID
    assert exported[0]["target"]["tool_calls"][0]["function"]["arguments"] == _ARGUMENTS
    assert "grader_score" not in exported[0]["target"]
    assert exported[0]["tools"] == _row()["tools"]
