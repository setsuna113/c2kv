"""Recorded-prefix proofs reach the selected response without model generations."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from benchmarks.memory_runtime.tests.test_static_extensions import (
    controller, payload, DraftTokenizer, Generator,
)
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner
from benchmarks.memory_runtime.attempt_journal import AttemptJournal


def case(task):
    path = Path(__file__).parent / "fixtures/static_verified_v2/recorded_prefixes.json"
    return next(row for row in json.loads(path.read_text(encoding="utf8"))["cases"]
                if row["id"].startswith(task + "_"))


def native_text(calls):
    return "\n".join('<tool_call>' + json.dumps({
        "name": row["function"]["name"],
        "arguments": json.loads(row["function"]["arguments"]),
    }) + '</tool_call>' for row in calls)


@pytest.mark.parametrize("task,field,expected", [
    ("base_15", "numbers", [3, 16, 60]),
    ("base_149", "receiver_id", "USR003"),
    ("base_180", "card_id", "main_card"),
])
def test_recorded_relation_reaches_final_runner_response(tmp_path, task, field, expected):
    row = case(task)
    text = native_text(row["draft_tool_calls"])
    tokenizer = DraftTokenizer([text])
    c = controller("static_verified_v2", tokenizer=tokenizer)
    runner = EventNativeDecisionRunner(c, Generator(), tokenizer, ratio=8,
        max_new_tokens=32, max_generation_calls=2,
        journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    record = runner.run(payload(row["messages"], row["tools"]))
    assert record["generation_completed"] == 1
    assert record["generation_trace"][0]["native_draft"]["text"] == text
    calls = record["response"]["tool_calls"]
    assert len(calls) == len(row["draft_tool_calls"])
    for original, output in zip(row["draft_tool_calls"], calls):
        assert original["function"]["name"] == output["function"]["name"]
        wanted = json.loads(original["function"]["arguments"])
        wanted[field] = expected
        assert json.loads(output["function"]["arguments"]) == wanted
    receipt = record["commit_transform"]
    assert receipt["changed"] and receipt["model_generation_unmodified"]
    assert receipt["additional_generations"] == 0
    assert receipt["additional_model_workspace_tokens"] == 0
    assert receipt["proof_registry_version"] == "verified-binding-relations-v2"


def test_relation_context_binds_regenerated_selected_call(tmp_path, monkeypatch):
    row = case("base_15")
    held = copy.deepcopy(row["draft_tool_calls"])
    held[0]["function"]["arguments"] = json.dumps({"numbers": [1, 1, 1]})
    texts = [native_text(held), native_text(row["draft_tool_calls"])]
    tokenizer = DraftTokenizer(texts)
    c = controller("static_verified_v2", score=0.9, tokenizer=tokenizer)
    data = payload(row["messages"], row["tools"])
    prepared = c.prepare(data, ratio=8, max_new_tokens=32)
    event = next(iter(prepared.memory.view.gist_event_ids))
    monkeypatch.setattr("benchmarks.memory_runtime.candidate_algorithms.controller.select_source_event",
        lambda *args, **kwargs: (event, {"ranked_candidate_event_ids": [event]}))
    runner = EventNativeDecisionRunner(c, Generator(), tokenizer, ratio=8,
        max_new_tokens=32, max_generation_calls=2,
        journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    record = runner.run(data)
    assert record["generation_completed"] == 2
    assert record["generation_trace"][1]["native_draft"]["text"] == texts[1]
    assert record["commit_transform"]["base_verification"]["status"] == "original_static_commit_preserved"
    assert record["commit_transform"]["status"] == "relational_binding_committed"
    assert json.loads(record["response"]["tool_calls"][0]["function"]["arguments"])["numbers"] == [3, 16, 60]
