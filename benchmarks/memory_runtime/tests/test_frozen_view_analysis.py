import copy

import pytest

from benchmarks.memory_runtime.frozen_view_analysis import classify_response, summarize_responses, next_action


TOOLS = [{"type": "function", "function": {"name": "ls"}},
         {"type": "function", "function": {"name": "cd"}}]


def _body(name="ls", arguments="{}", finish="tool_calls"):
    return {"choices": [{"finish_reason": finish, "message": {
        "content": "A plan is not itself an action.",
        "tool_calls": [{"id": "call1", "type": "function", "function": {
            "name": name, "arguments": arguments}}]}}]}


def test_native_signature_ignores_ids_and_json_key_order_but_keeps_action_identity():
    first = _body("cd", '{"folder":"archive","extra":true}')
    second = _body("cd", '{"extra":true,"folder":"archive"}')
    second["choices"][0]["message"]["tool_calls"][0]["id"] = "call99"
    a = classify_response(first, TOOLS)
    b = classify_response(second, TOOLS)
    assert a["native_continuation"] is b["native_continuation"] is True
    assert a["signature"] == b["signature"]
    assert a["signature"] != classify_response(_body(), TOOLS)["signature"]


@pytest.mark.parametrize("name,arguments", [("undeclared", "{}"), ("ls", "[]"),
                                           ("ls", "not json"), ("ls", '{"x":NaN}')])
def test_malformed_or_undeclared_native_calls_do_not_count(name, arguments):
    result = classify_response(_body(name, arguments), TOOLS)
    assert result["native_present"] is True
    assert result["native_continuation"] is False
    assert result["category"] == "malformed_native_call"


def test_mixed_valid_invalid_native_list_is_reported_as_malformed():
    response = _body()
    response["choices"][0]["message"]["tool_calls"].append({"type": "function"})
    assert classify_response(response, TOOLS)["category"] == "malformed_native_call"


def test_truncation_is_censored_even_when_a_partial_native_call_is_visible():
    result = classify_response(_body(arguments="{", finish="length"), TOOLS)
    assert result["native_continuation"] is None
    assert result["category"] == "truncated_generation"


def test_tool_text_is_not_a_native_action_and_call_order_is_preserved():
    response = _body()
    response["choices"][0]["message"]["tool_calls"] = None
    response["choices"][0]["message"]["content"] = '<tool_call>{"name":"ls","arguments":{}}</tool_call>'
    assert classify_response(response, TOOLS)["category"] == "no_native_call"
    ordered = _body()
    ordered["choices"][0]["message"]["tool_calls"] += _body("cd", '{"folder":"archive"}')["choices"][0]["message"]["tool_calls"]
    reversed_order = copy.deepcopy(ordered)
    reversed_order["choices"][0]["message"]["tool_calls"].reverse()
    assert classify_response(ordered, TOOLS)["signature"] != classify_response(reversed_order, TOOLS)["signature"]


def test_action_identity_drift_does_not_hide_stable_native_continuation():
    rows = [{"classification": classify_response(_body(name), TOOLS)} for name in ("ls", "cd", "ls", "cd")]
    summary = summarize_responses(rows)
    assert summary["native_continuation_count"] == 4
    assert summary["primary_all_continue"] is True
    assert len(summary["action_signatures"]) == 2


def test_censored_repeat_cannot_support_an_all_stop_pattern():
    ordinary = _body()
    ordinary["choices"][0]["message"]["tool_calls"] = []
    rows = [{"classification": classify_response(ordinary, TOOLS)},
            {"classification": classify_response(_body(finish="length"), TOOLS)}]
    summary = summarize_responses(rows)
    assert summary["scorable_cells"] == 1
    assert summary["censored_or_malformed_response_cells"] == 1
    assert summary["primary_all_stop"] is False


def test_identity_drift_allows_view_diagnosis_but_missing_repeats_do_not():
    continuation = summarize_responses([
        {"classification": classify_response(_body(name), TOOLS)}
        for name in ("ls", "cd", "ls", "cd")])
    stopped = _body()
    stopped["choices"][0]["message"]["tool_calls"] = []
    no_continuation = summarize_responses([
        {"classification": classify_response(stopped, TOOLS)} for _ in range(4)])
    groups = {key: continuation for key in ("negative:A", "negative:C", "main:A", "main:B", "main:D")}
    groups["main:C"] = no_continuation
    assert next_action(groups, True)["decision"] == "inspect_c2kv_layout_positions_and_serving"
    assert next_action(groups, False)["decision"] == "incomplete_no_refill"


def test_directory_analysis_matches_durable_attempts_and_rejects_reordered_cells(tmp_path):
    from benchmarks.memory_runtime.frozen_view_analysis import analyze_directory
    from benchmarks.memory_runtime import frozen_view_probe as probe
    from benchmarks.memory_runtime.tests.test_frozen_view_probe import _prepared_artifact, _Opener

    prepared = _prepared_artifact()
    prepared_path = tmp_path / "prepared.json"
    probe._save(prepared_path, prepared)
    root = tmp_path / "live"
    probe.execute_prepared(prepared, prepared_path, "http://synthetic", root, opener=_Opener())
    result = analyze_directory(root, prepared_path)
    assert result["complete_valid"] is True
    assert result["validation_errors"] == []
    assert result["attempt_journal"]["by_kind"]["generation"]["started"] == 24
    path = root / "cells.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    broken = analyze_directory(root, prepared_path)
    assert broken["complete_valid"] is False
    assert any("frozen order" in error for error in broken["validation_errors"])
    assert broken["next_action"]["decision"] == "incomplete_no_refill"


def test_directory_analysis_preserves_partial_failure_without_refill(tmp_path):
    import pytest
    from benchmarks.memory_runtime.frozen_view_analysis import analyze_directory
    from benchmarks.memory_runtime import frozen_view_probe as probe
    from benchmarks.memory_runtime.tests.test_frozen_view_probe import _prepared_artifact, _Opener

    prepared = _prepared_artifact()
    prepared_path = tmp_path / "prepared.json"
    probe._save(prepared_path, prepared)
    root = tmp_path / "live"
    with pytest.raises(probe.LiveAttemptError):
        probe.execute_prepared(prepared, prepared_path, "http://synthetic", root,
                               opener=_Opener(fail_generation=2))
    result = analyze_directory(root, prepared_path)
    assert result["complete_valid"] is False
    assert result["validation_errors"] == []
    assert len(result["cells"]) == 2
    assert result["cells"][-1]["classification"]["category"] == "transport_or_backend_failure"
    assert result["next_action"]["decision"] == "incomplete_no_refill"
