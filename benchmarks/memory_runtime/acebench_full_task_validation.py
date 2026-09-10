"""Offline joins for one ACEBench Full-original task run."""
from __future__ import annotations

import json
from pathlib import Path

from .attempt_journal import read_attempt_journal, summarize_attempt_journal


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _rows(path):
    path = Path(path)
    _need(path.is_file(), f"missing artifact: {path}")
    try:
        result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSONL artifact: {path}") from error
    _need(result, f"empty artifact: {path}")
    return result


def validate_run_artifacts(run: Path, design: dict, *, score_root: Path | None = None) -> dict:
    """Validate frozen single-task artifacts without reading gold content."""
    run = Path(run)
    task_id, model = design["task_id"], design["model_name"]
    max_turns, max_calls = design["max_dialog_turns"], design["max_generation_calls"]
    _need(isinstance(task_id, str) and task_id, "invalid design task_id")
    _need(isinstance(model, str) and model, "invalid design model_name")
    _need(type(max_turns) is int and max_turns > 0, "invalid max_dialog_turns")
    _need(type(max_calls) is int and max_calls > 0, "invalid max_generation_calls")

    result = _rows(run / "result_all/result_en" / model / "data_agent_multi_step_result.json")
    _need(len(result) == 1 and result[0].get("id") == task_id, "official result is not the task singleton")
    _need(isinstance(result[0].get("process"), list), "official result process is malformed")
    score_path = (Path(score_root) if score_root is not None else run / "score_all/score_en") / model / "data_agent_multi_step_score.json"
    score = _rows(score_path)
    _need(score[0].get("total_count") == 1, "official score total_count is not one")

    dialogue_path = run / "dialogue.json"
    _need(dialogue_path.is_file(), "missing dialogue.json")
    try:
        dialogue = json.loads(dialogue_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid dialogue.json") from error
    _need(isinstance(dialogue, list) and dialogue and all(isinstance(x, dict) for x in dialogue),
          "dialogue is malformed")
    _need(dialogue[0].get("sender") == "user", "dialogue does not start with the user")
    turns, receipts, process = [], [], []
    user_turn, step_index = -1, 0
    receipt_fields = {"version", "agent_history_index", "execution_message_index", "decode_status",
                      "decoded_calls", "executor_status", "executor_return_shape", "executor_return_count"}
    for index, message in enumerate(dialogue):
        sender = message.get("sender")
        _need(sender in {"user", "agent", "execution"}, f"dialogue sender {index} is invalid")
        if index:
            _need(sender == ("agent" if index % 2 else "execution"), "dialogue roles do not alternate")
        if sender == "user":
            user_turn, step_index = user_turn + 1, 0
        elif sender == "agent":
            _need(isinstance(message.get("message"), str), f"agent message {index} is not text")
            turns.append((message, f"turn-{user_turn}/step-{step_index}", list(receipts)))
            step_index += 1
        else:
            _need(index and dialogue[index - 1].get("sender") == "agent", "execution does not follow agent")
            receipt = message.get("c2kv_acebench_execution")
            _need(isinstance(receipt, dict) and set(receipt) == receipt_fields, "execution receipt is malformed")
            _need(receipt["version"] == "acebench-execution-receipt-v1", "execution receipt version differs")
            _need(receipt["agent_history_index"] == index - 1, "receipt agent index differs")
            _need(receipt["execution_message_index"] == index + 1, "receipt execution index differs")
            receipts.append(receipt)
            process.append(dialogue[index - 1]["message"])
    _need(result[0]["process"] == process, "official result process does not join dialogue actions")
    iterations = len(dialogue) - 1
    _need(iterations <= max_turns, "dialogue exceeds max_dialog_turns")
    last = dialogue[-1]
    finish = (len(dialogue) > 3 and last.get("sender") == "agent" and last.get("recipient") == "user"
              and isinstance(last.get("message"), str) and "finish conversation" in last["message"])
    iteration_limit = not finish and iterations == max_turns

    steps = _rows(run / "steps.jsonl")
    raw = _rows(run / "raw_http.jsonl")
    journal = read_attempt_journal(run / "attempts.jsonl")
    attempts = journal["records"]
    _need(not journal["truncated_tail"], "attempt journal has a truncated tail")
    _need(len(steps) == len(turns) and 0 < len(steps) <= max_calls, "step/agent-turn count differs")
    _need(len(raw) == len(attempts) == 2 * len(steps), "HTTP/attempt pair count differs")
    session = f"acebench/{task_id}/attempt-0"
    prompt_tokens = completion_tokens = 0
    seen_uids, seen_requests = set(), set()
    for call_index, (step, turn) in enumerate(zip(steps, turns, strict=True), 1):
        message, decision, prior_receipts = turn
        _need(step.get("schema") == "a-acebench-event-step-v1" and step.get("status") == "ok",
              f"step {call_index} is incomplete")
        _need(step.get("session_id") == session and step.get("decision_key") == decision,
              f"step {call_index} task/decision identity differs")
        source = step.get("ace_source")
        _need(isinstance(source, dict) and source.get("version") == "acebench-text-actions-v1"
              and source.get("receipts") == prior_receipts, f"step {call_index} source receipts differ")
        _need(step.get("response", {}).get("content") == message["message"],
              f"step {call_index} content does not join dialogue")
        traces = step.get("generation_trace")
        _need(isinstance(traces, list) and len(traces) == 1, f"step {call_index} is not one Full call")
        trace = traces[0]
        _need(trace.get("status") == "completed" and trace.get("discarded") is False,
              f"step {call_index} generation is incomplete")
        _need(step.get("generation_attempts") == step.get("generation_completed") == 1,
              f"step {call_index} generation totals differ")

        request, response = raw[2 * call_index - 2], raw[2 * call_index - 1]
        _need(request.get("schema") == response.get("schema") == "a-acebench-full-raw-sglang-http-v1",
              f"HTTP pair {call_index} schema differs")
        _need(request.get("event") == "request" and request.get("status") == "started"
              and response.get("event") == "response" and response.get("status") == "completed",
              f"HTTP pair {call_index} is incomplete")
        _need(request.get("request_index") == response.get("request_index") == call_index,
              f"HTTP pair {call_index} index is not contiguous")
        _need(request.get("retries") == response.get("retries") == 0, f"HTTP pair {call_index} retried")
        submitted = request.get("request", {}).get("input_ids")
        prepared = trace.get("prepared_input", {})
        _need(isinstance(submitted, list)
              and submitted == prepared.get("system_input_ids", []) + prepared.get("workspace_input_ids", []),
              f"HTTP input {call_index} does not join prepared memory")
        output = response.get("response", {}).get("output_ids")
        _need(isinstance(output, list) and output == trace.get("generation", {}).get("token_ids"),
              f"HTTP output {call_index} does not join generation trace")
        meta = response.get("response", {}).get("meta_info", {})
        prompt, completion = meta.get("prompt_tokens"), meta.get("completion_tokens")
        _need(type(prompt) is int and prompt == len(submitted), f"HTTP prompt cost {call_index} differs")
        _need(type(completion) is int and completion == len(output), f"HTTP completion cost {call_index} differs")
        usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
        _need(trace.get("usage") == step.get("generation_usage_total") == usage,
              f"step cost {call_index} differs from HTTP")
        prompt_tokens, completion_tokens = prompt_tokens + prompt, completion_tokens + completion

        started, finished = attempts[2 * call_index - 2], attempts[2 * call_index - 1]
        keys = ("attempt_uid", "kind", "attempt_index", "request_id", "eval_context")
        _need(all(started.get(k) == finished.get(k) for k in keys), f"attempt pair {call_index} differs")
        uid, request_id = started.get("attempt_uid"), started.get("request_id")
        _need(uid not in seen_uids and request_id not in seen_requests, f"attempt {call_index} is duplicate")
        seen_uids.add(uid)
        seen_requests.add(request_id)
        _need(started.get("event") == "started" and started.get("status") == "started"
              and finished.get("event") == "finished" and finished.get("status") == "completed",
              f"attempt pair {call_index} is incomplete")
        _need(started.get("kind") == "generation" and started.get("attempt_index") == call_index,
              f"attempt {call_index} index differs")
        _need(trace.get("attempt_uid") == uid and trace.get("attempt_index") == call_index,
              f"attempt {call_index} does not join generation trace")
        try:
            decoded_request = json.loads(request_id)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"attempt {call_index} request_id is malformed") from error
        _need(decoded_request == [session, decision], f"attempt {call_index} request identity differs")
        _need(started.get("eval_context") == {"task_id": session, "decision_id": decision},
              f"attempt {call_index} task/decision context differs")
        _need(finished.get("usage") == usage, f"attempt {call_index} cost differs")

    total_tokens = prompt_tokens + completion_tokens
    summary = summarize_attempt_journal(run / "attempts.jsonl")
    count = len(steps)
    _need((summary["started"], summary["finished"], summary["completed"], summary["failed"], summary["pending"])
          == (count, count, count, 0, 0), "attempt summary totals differ")
    _need(summary["finished_usage_totals"] == {"prompt_tokens": prompt_tokens,
          "completion_tokens": completion_tokens, "total_tokens": total_tokens}, "attempt costs differ")
    return {"schema": "a-ace-full-task-artifact-validation-v1", "task_id": task_id,
            "official_score_path": str(score_path), "official_score_header": score[0],
            "acceptance": finish,
            "checks": {"official_result_singleton": True, "official_score_total_count": 1,
                       "artifact_joins": True, "actual_finish_marker": finish,
                       "iteration_limit_reached": iteration_limit},
            "cost_totals": {"dialogue_iterations": iterations, "agent_turns": len(turns),
                            "execution_turns": len(receipts), "generation_calls": count,
                            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                            "total_tokens": total_tokens}}


__all__ = ["validate_run_artifacts"]
