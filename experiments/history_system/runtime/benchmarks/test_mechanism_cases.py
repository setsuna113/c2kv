"""Focused artifact-contract tests for mechanism_cases.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mechanism_cases as audit


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def _action(name: str, arguments_raw: str):
    return {"role": "assistant", "content": [{name: arguments_raw}]}


def _result(task_id: str, second_action=None):
    first_action = _action("first", '{"z": 1}')
    second_action = second_action or _action("second", '{"x": 2}')
    return {
        "id": task_id,
        "result": [],
        "inference_log": [
            [],
            {
                "begin_of_turn_query": [{"role": "user", "content": f"query {task_id}"}],
                "step_0": [first_action, {"role": "tool", "content": "ok"}],
                "step_1": [second_action, {"role": "tool", "content": "done"}],
            },
            [],
        ],
    }


def _run(root: Path, name: str, task_ids, proxy_rows, results=None,
         success_ids=()):
    run = root / name
    _json(run / "test_case_ids_to_generate.json", {audit.CATEGORY: task_ids})
    handler = "c2kv-full" if name == "full_r1" else "c2kv-c2kv4"
    result_path = run / "result" / handler / "multi_turn" \
        / "BFCL_v4_multi_turn_base_result.json"
    result_rows = results if results is not None else [_result(task_id) for task_id in task_ids]
    _jsonl(result_path, result_rows)
    details = [
        {"id": task_id, "valid": False}
        for task_id in task_ids if task_id not in success_ids
    ]
    _jsonl(
        run / "score" / handler / "multi_turn" / "BFCL_v4_multi_turn_base_score.json",
        [{"accuracy": len(success_ids) / len(task_ids),
          "correct_count": len(success_ids), "total_count": len(task_ids)}, *details],
    )
    _jsonl(run / "logs" / f"proxy_{name}.jsonl", proxy_rows)
    _json(run / "command.json", ["python", "run.py", "--model", "fixture-model"])
    _json(run / "checkpoint_profile.resolved.json", {
        "profile_fingerprint": f"profile-{name}",
        "checkpoint": {"name": "checkpoint-fixture", "config_sha256": "a" * 64},
    })
    return run


def _proxy(task_id: str, step: int, fp: str, attempt: int = 0):
    return {
        "status": "ok", "fp": fp,
        "eval_context": {
            "benchmark": "bfcl", "task_id": task_id,
            "user_turn": 0, "step": step, "attempt": attempt,
        },
    }


def _tools(path: Path, task_ids):
    _json(path, {"bfcl_git_sha": audit.PINNED_BFCL_SHA, "tasks": [
        {"task_id": task_id, "tools": [{"type": "function", "function": {"name": "x"}}]}
        for task_id in task_ids
    ]})


def test_missing_result_id_is_rejected_instead_of_silently_joined(tmp_path):
    task_ids = ["multi_turn_base_0", "multi_turn_base_5"]
    rows = []
    for index, task_id in enumerate(task_ids):
        first_fp = audit._messages_fingerprint([
            {"role": "user", "content": f"query {task_id}"}
        ])
        rows.extend([_proxy(task_id, 0, first_fp),
                     _proxy(task_id, 1, (str(index + 2) * 64)[:64])])
    full = _run(tmp_path, "full_r1", task_ids, rows,
                results=[_result(task_ids[0])], success_ids=())
    c2kv = _run(tmp_path, "c2kv4", task_ids, rows, success_ids=())
    tools = tmp_path / "tools.json"
    _tools(tools, task_ids)
    with pytest.raises(audit.AuditError, match="result id mismatch"):
        audit.collect(full, c2kv, tools)


def test_common_state_filter_and_parallel_action_order_are_preserved(tmp_path):
    task_ids = ["multi_turn_base_0", "multi_turn_base_5", "multi_turn_base_10"]
    shared_second = "b" * 64
    full_rows = []
    c2kv_rows = []
    for task_id in task_ids:
        first_fp = audit._messages_fingerprint([
            {"role": "user", "content": f"query {task_id}"}
        ])
        first = _proxy(task_id, 0, first_fp)
        first["n_messages"] = 1
        full_rows.append(first)
        c2kv_rows.append(dict(first))
    full_rows.extend([
        _proxy(task_ids[0], 1, shared_second),
        _proxy(task_ids[1], 1, "c" * 64),
        _proxy(task_ids[2], 1, "d" * 64),
    ])
    c2kv_rows.extend([
        _proxy(task_ids[0], 1, shared_second),
        _proxy(task_ids[1], 1, "e" * 64),
        _proxy(task_ids[2], 1, "d" * 64, attempt=1),
    ])
    parallel = {
        "role": "assistant",
        "content": [
            {"alpha": '{"z": [3, 2, 1], "nested": {"b": 2, "a": 1}}'},
            {"beta": '{"flag": true, "value": 7}'},
        ],
    }
    full_results = [_result(task_id, parallel if task_id == task_ids[0] else None)
                    for task_id in task_ids]
    c2kv_results = [_result(task_id, parallel if task_id == task_ids[0] else None)
                    for task_id in task_ids]
    full = _run(tmp_path, "full_r1", task_ids, full_rows, full_results,
                success_ids=(task_ids[0],))
    c2kv = _run(tmp_path, "c2kv4", task_ids, c2kv_rows, c2kv_results,
                 success_ids=())
    tools = tmp_path / "tools.json"
    _tools(tools, task_ids)

    result = audit.collect(full, c2kv, tools)
    assert [case["task_id"] for case in result["cases"]] == [task_ids[0]]
    assert result["summary"]["first_turn_step_1"]["context_match"] == 2
    assert result["summary"]["first_turn_step_1"]["fp_match"] == 1
    calls = result["cases"][0]["actions"]["full_r1"]["tool_calls"]
    assert [call["name"] for call in calls] == ["alpha", "beta"]
    assert calls[0]["arguments_raw"].startswith('{"z":')
    assert calls[0]["arguments"] == {
        "z": [3, 2, 1], "nested": {"b": 2, "a": 1},
    }
    assert result["cases"][0]["full_task_success"] is True
    assert result["cases"][0]["request"]["status"] == "missing_exact_replay_request"
    assert result["cases"][0]["request"]["payload"] is None
    assert len(result["first_requests"]) == 3
    assert all(item["request"]["status"] == "exact"
               for item in result["first_requests"])
    constructed = result["constructed_cases"][0]
    assert constructed["request"]["status"] == "constructed_from_logged_transition_v1"
    messages = constructed["request"]["payload"]["messages"]
    assert messages[1]["content"] == ""
    assert messages[1]["tool_calls"][0]["id"] == "c2kv_constructed_0_r1_0"
    assert messages[2]["tool_call_id"] == messages[1]["tool_calls"][0]["id"]
