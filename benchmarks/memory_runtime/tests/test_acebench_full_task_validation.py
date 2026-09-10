"""Synthetic no-model checks for the ACEBench Full-task artifact validator."""
from __future__ import annotations

import json

import pytest

from benchmarks.memory_runtime.acebench_full_task_validation import validate_run_artifacts
from benchmarks.memory_runtime.attempt_journal import AttemptJournal


def _write(path, value, *, lines=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if lines:
        path.write_text("\n".join(json.dumps(row) for row in value), encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")


def _fixture(tmp_path, *, finish=True):
    task_id, model = "agent_multi_step_19", "c2kv-agent"
    session = f"acebench/{task_id}/attempt-0"
    action = "[SearchMessages(query='Ada')]"
    final = "finish conversation" if finish else "still working"
    receipt = {
        "version": "acebench-execution-receipt-v1",
        "agent_history_index": 1,
        "execution_message_index": 3,
        "decode_status": "ok",
        "decoded_calls": ["SearchMessages(query='Ada')"],
        "executor_status": "returned",
        "executor_return_shape": "list",
        "executor_return_count": 1,
    }
    dialogue = [
        {"sender": "user", "recipient": "agent", "message": "Find Ada"},
        {"sender": "agent", "recipient": "execution", "message": action},
        {"sender": "execution", "recipient": "agent", "message": [{"id": "m1"}],
         "c2kv_acebench_execution": receipt},
        {"sender": "agent", "recipient": "user", "message": final},
    ]
    _write(tmp_path / "dialogue.json", dialogue)
    _write(tmp_path / "result_all/result_en" / model / "data_agent_multi_step_result.json",
           [{"id": task_id, "result": [], "process": [action]}], lines=True)
    _write(tmp_path / "score_all/score_en" / model / "data_agent_multi_step_score.json",
           [{"end_to_end_accuracy": 0, "process_accuracy": 0.5,
             "correct_count": 0, "total_count": 1}], lines=True)

    journal = AttemptJournal(tmp_path / "attempts.jsonl")
    steps, raw = [], []
    for index, (content, source_receipts) in enumerate(((action, []), (final, [receipt])), 1):
        decision = f"turn-0/step-{index - 1}"
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        handle = journal.start("generation", index, json.dumps([session, decision], separators=(",", ":")),
                               {"task_id": session, "decision_id": decision})
        journal.finish(handle, "completed", usage=usage)
        trace = {
            "status": "completed", "discarded": False,
            "attempt_uid": handle.attempt_uid, "attempt_index": index,
            "prepared_input": {"system_input_ids": [10, index], "workspace_input_ids": [12]},
            "usage": usage,
            "generation": {"token_ids": [20, index]},
        }
        steps.append({
            "schema": "a-acebench-event-step-v1", "status": "ok",
            "session_id": session, "decision_key": decision,
            "source_profile": "acebench-text-actions-v1",
            "ace_source": {"version": "acebench-text-actions-v1", "receipts": source_receipts},
            "response": {"role": "assistant", "content": content},
            "generation_trace": [trace], "generation_attempts": 1, "generation_completed": 1,
            "generation_usage_total": usage,
        })
        raw.extend([
            {"schema": "a-acebench-full-raw-sglang-http-v1", "event": "request", "status": "started",
             "request_index": index, "retries": 0,
             "request": {"input_ids": [10, index, 12]}},
            {"schema": "a-acebench-full-raw-sglang-http-v1", "event": "response", "status": "completed",
             "request_index": index, "retries": 0,
             "response": {"output_ids": [20, index],
                          "meta_info": {"prompt_tokens": 3, "completion_tokens": 2}}},
        ])
    _write(tmp_path / "steps.jsonl", steps, lines=True)
    _write(tmp_path / "raw_http.jsonl", raw, lines=True)
    design = {"task_id": task_id, "model_name": model,
              "max_dialog_turns": 4 if finish else 3, "max_generation_calls": 2}
    return design


@pytest.mark.parametrize("finish", [True, False])
def test_validates_finished_and_iteration_limited_runs(tmp_path, finish):
    result = validate_run_artifacts(tmp_path, _fixture(tmp_path, finish=finish))
    assert result["acceptance"] is finish
    assert result["checks"]["iteration_limit_reached"] is (not finish)
    assert result["cost_totals"] == {
        "dialogue_iterations": 3, "agent_turns": 2, "execution_turns": 1,
        "generation_calls": 2, "prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10,
    }


def test_rejects_a_broken_raw_input_join(tmp_path):
    design = _fixture(tmp_path)
    rows = [json.loads(line) for line in (tmp_path / "raw_http.jsonl").read_text().splitlines()]
    rows[0]["request"]["input_ids"] = [999]
    _write(tmp_path / "raw_http.jsonl", rows, lines=True)
    with pytest.raises(ValueError, match="does not join prepared memory"):
        validate_run_artifacts(tmp_path, design)
