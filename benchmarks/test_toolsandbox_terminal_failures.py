"""Separate actor-selected invalid retrievals from unscored harness crashes."""

import json
import hashlib

import pytest

from benchmarks.adapters import toolsandbox_adapter as ts


def _result(path, scenarios):
    folder = path / "agent_run"
    folder.mkdir(exist_ok=True)
    (folder / "result_summary.json").write_text(
        json.dumps({"per_scenario_results": scenarios}), encoding="utf-8")


def test_invalid_hiagent_retrieval_is_task_local_and_preserves_other_scores(tmp_path):
    _result(tmp_path, [
        {"name": "complete", "similarity": 1.0, "traceback": None},
        {"name": "bad_retrieval", "similarity": None,
         "traceback": "APIError: HiAgent requested nonexistent completed subgoals: [1]"},
    ])
    result = ts.collect(tmp_path)
    assert result["n"] == 2
    assert result["semantic_score"] == 0.5
    assert result["task_failures"] == {
        "hiagent_invalid_retrieval": ["bad_retrieval"]}
    assert result["scenario_ids"] == ["bad_retrieval", "complete"]


def test_unknown_crash_is_incomplete_even_after_completed_scenarios(tmp_path):
    _result(tmp_path, [
        {"name": "complete", "similarity": 1.0},
        {"name": "server_down", "traceback": "HTTP 502 connection refused"},
    ])
    with pytest.raises(SystemExit, match="1 scenario\\(s\\) crashed"):
        ts.collect(tmp_path)


def test_generic_502_needs_last_typed_proxy_error_for_that_scenario(tmp_path):
    from benchmarks.measurement.telemetry import append_jsonl

    _result(tmp_path, [
        {"name": "bad_retrieval", "traceback": "APIStatusError: HTTP 502"},
        {"name": "server_down", "traceback": "APIStatusError: HTTP 502"},
    ])
    events = tmp_path / "measurement" / "harness_events.jsonl"
    for scenario, instance in (("bad_retrieval", "one"), ("server_down", "two")):
        append_jsonl(events, {"event_type": "episode_start", "episode_id": scenario,
                              "episode_instance_id": instance})
    log = tmp_path / "logs" / "proxy_hiagent_34100.jsonl"
    for instance, status, error in (
        ("one", "upstream_error", "HiAgent requested nonexistent completed subgoals: [1]"),
        ("two", "upstream_error", "connection refused"),
    ):
        conv = hashlib.sha256(json.dumps(
            ["measurement_session", instance], separators=(",", ":")).encode()).hexdigest()
        append_jsonl(log, {"conv_id": conv, "status": status, "error": error})
    assert ts._invalid_retrieval_scenarios(tmp_path) == {"bad_retrieval"}
    with pytest.raises(SystemExit, match="server_down"):
        ts.collect(tmp_path)
    _result(tmp_path, [{"name": "bad_retrieval", "traceback": "APIStatusError: HTTP 502"}])
    result = ts.collect(tmp_path)
    assert result["semantic_score"] == 0.0
    assert result["task_failures"] == {"hiagent_invalid_retrieval": ["bad_retrieval"]}
