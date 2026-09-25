"""End-to-end guards for native tool drafts and visible catalog membership."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from benchmarks.memory_runtime.attempt_journal import AttemptJournal
from benchmarks.memory_runtime.event_native_step import EventNativeDecisionRunner, EventNativeStepError
from benchmarks.memory_runtime.tests.test_goal_composition import call
from benchmarks.memory_runtime.tests.test_tool_recovery import TextGenerator, payload, wrapped
from benchmarks.memory_runtime.tests.test_verified_controller import Binding


def tool_block(name, arguments=None):
    return "<tool_call>" + json.dumps({
        "name": name, "arguments": arguments or {"option": "fuel"},
    }) + "</tool_call>"


def run_with_texts(tmp_path, texts, *, mode="draft-full-raw", cap=2, binding=None, data=None):
    outer, _, tokenizer = wrapped(mode=mode, binding=binding)
    generator = TextGenerator(texts)
    outer.generator = generator
    runner = EventNativeDecisionRunner(
        outer, generator, tokenizer, ratio=8, max_new_tokens=32,
        max_generation_calls=cap, journal=AttemptJournal(tmp_path / "attempts.jsonl"))
    return runner, outer, generator, data or payload()


def assert_invalid_action(error, *, completed):
    record = error.value.record
    assert record["status"] == "failed"
    assert record["response"] is None
    assert record["failure_kind"] == "method_failure"
    assert record["failure_code"] == "invalid_tool_action"
    assert record["generation_completed"] == completed
    assert all(trace["discarded"] for trace in record["generation_trace"])
    return record


def test_unknown_trading_tool_restores_catalog_and_regenerates_valid_call(tmp_path):
    data = payload()
    data["tools"][0]["function"].update(
        name="trading_login", description="Authenticate before trading.")
    data["tools"][1]["function"].update(
        name="get_stock_info", description="Read current stock information.")
    runner, _, generator, data = run_with_texts(
        tmp_path, [tool_block("trading_get_stock_info"), tool_block("get_stock_info")],
        data=data)

    record = runner.run(data)

    receipt = record["exact_recovery"]["tool_recovery"]
    assert receipt["schema"] == "c2kv-tool-recovery-v2"
    assert receipt["triggered"] and receipt["reason"] == "unknown_tool_name"
    assert receipt["unknown_tool_names"] == ["trading_get_stock_info"]
    assert receipt["restored_tool_indices"] == [0, 1]
    assert record["generation_completed"] == 2
    assert len(generator.memories) == 2
    assert "Authenticate before trading" in runner.tokenizer.decode(
        generator.memories[1].system_input_ids)
    assert "Read current stock information" in runner.tokenizer.decode(
        generator.memories[1].system_input_ids)
    assert record["response"]["tool_calls"][0]["function"]["name"] == "get_stock_info"
    assert record["tool_commit_validation"]["accepted"]
    assert record["tool_commit_final_validation"]["accepted"]


def test_explicit_malformed_native_tool_block_triggers_raw_recovery(tmp_path):
    runner, _, _, data = run_with_texts(
        tmp_path, ['<tool_call>{broken</tool_call>', tool_block("lookupZipcode")])

    record = runner.run(data)

    receipt = record["exact_recovery"]["tool_recovery"]
    assert receipt["schema"] == "c2kv-tool-recovery-v2"
    assert receipt["triggered"] and receipt["reason"] == "malformed_tool_call"
    assert receipt["restored_tool_indices"] == [0, 1]
    assert record["generation_trace"][0]["native_draft"]["status"] == "malformed"
    assert record["response"]["tool_calls"][0]["function"]["name"] == "lookupZipcode"


@pytest.mark.parametrize("second", [tool_block("another_invented_api"),
                                         '<tool_call>{broken</tool_call>'])
def test_two_invalid_drafts_fail_without_executable_response(tmp_path, second):
    runner, _, _, data = run_with_texts(
        tmp_path, [tool_block("invented_api"), second])

    with pytest.raises(EventNativeStepError) as error:
        runner.run(data)

    record = assert_invalid_action(error, completed=2)
    assert record["tool_commit_validation"]["accepted"] is False
    assert record["tool_commit_validation"]["original_draft"]["accepted"] is False


def test_invalid_regeneration_falls_back_to_valid_original_and_restores_proof(tmp_path):
    binding = Binding()
    runner, _, _, data = run_with_texts(
        tmp_path, [tool_block("displayCarStatus", {"option": "fuelLevel"}),
                   tool_block("invented_api")], binding=binding)

    record = runner.run(data)

    assert record["generation_completed"] == 2
    assert record["tool_commit_validation"]["fallback"] == "original"
    assert record["tool_commit_validation"]["selected_generation_index"] == 0
    assert record["commit_restore"]["restored_original_draft_proof"]
    assert record["commit_transform"]["changed"] and binding.applied == 1
    assert record["response"]["tool_calls"][0]["function"]["name"] == "displayCarStatus"
    assert json.loads(record["response"]["tool_calls"][0]["function"]["arguments"]) == {"key": 42}
    assert record["tool_commit_final_validation"]["accepted"]


@pytest.mark.parametrize("capacity_rejected", [False, True])
def test_unknown_call_cannot_pass_generation_or_capacity_rejection(tmp_path, monkeypatch,
                                                                   capacity_rejected):
    runner, _, generator, data = run_with_texts(
        tmp_path, [tool_block("invented_api"), tool_block("displayCarStatus")],
        cap=2 if capacity_rejected else 1)
    if capacity_rejected:
        monkeypatch.setattr(runner, "_regeneration_capacity",
                            lambda memory: {"reason": "fixture_capacity"})

    with pytest.raises(EventNativeStepError) as error:
        runner.run(data)

    record = assert_invalid_action(error, completed=1)
    assert len(generator.memories) == 1
    assert record["tool_commit_validation"]["accepted"] is False
    if capacity_rejected:
        assert record["recovery_skipped"]["reason"] == "capacity"
    else:
        assert record["exact_recovery"]["reason"] == "shared_task_generation_limit"


def test_mode_none_preserves_legacy_unknown_call_path(tmp_path):
    runner, _, _, data = run_with_texts(
        tmp_path, [tool_block("invented_api")], mode="none")

    record = runner.run(data)

    assert record["generation_completed"] == 1
    assert record["response"]["tool_calls"][0]["function"]["name"] == "invented_api"
    assert "tool_commit_validation" not in record
    assert "tool_commit_final_validation" not in record


def test_plain_text_does_not_trigger_tool_recovery(tmp_path):
    runner, _, _, data = run_with_texts(tmp_path, ["Done."])

    record = runner.run(data)

    assert record["generation_completed"] == 1
    assert record["response"]["content"] == "Done."
    assert record["response"]["tool_calls"] == []
    assert not record["exact_recovery"]["tool_recovery"]["triggered"]
    assert record["tool_commit_final_validation"]["accepted"]


def test_unfinished_thinking_is_not_tool_call_syntax_trigger(tmp_path):
    runner, _, generator, data = run_with_texts(tmp_path, ["<think>still thinking"])

    with pytest.raises(EventNativeStepError) as error:
        runner.run(data)

    record = assert_invalid_action(error, completed=1)
    receipt = record["exact_recovery"]["tool_recovery"]
    assert not receipt["triggered"]
    assert receipt["reason"] != "malformed_tool_call"
    assert len(generator.memories) == 1


def test_batch_with_known_and_unknown_call_is_rejected_as_a_whole(tmp_path):
    draft = tool_block("displayCarStatus") + tool_block("invented_api")
    runner, _, _, data = run_with_texts(tmp_path, [draft], cap=1)

    with pytest.raises(EventNativeStepError) as error:
        runner.run(data)

    record = assert_invalid_action(error, completed=1)
    assert record["generation_trace"][0]["native_draft"]["status"] == "tool_calls"
    assert len(record["generation_trace"][0]["native_draft"]["tool_calls"]) == 2
    assert record["tool_commit_validation"]["unknown_tool_names"] == ["invented_api"]


def test_final_commit_transform_cannot_introduce_fake_tool_name(tmp_path, monkeypatch):
    runner, outer, _, data = run_with_texts(
        tmp_path, [tool_block("displayCarStatus")], cap=1)
    monkeypatch.setattr(outer, "finalize_commit", lambda prepared, calls: (
        tuple(calls), {"changed": True, "status": "fixture_transform"}))
    monkeypatch.setattr(outer, "render_commit", lambda draft, calls, receipt: replace(
        draft, tool_calls=(call("invented_api"),)), raising=False)

    with pytest.raises(EventNativeStepError) as error:
        runner.run(data)

    record = assert_invalid_action(error, completed=1)
    assert record["tool_commit_validation"]["accepted"]
    assert record["tool_commit_final_validation"]["accepted"] is False
    assert record["tool_commit_final_validation"]["unknown_tool_names"] == ["invented_api"]
