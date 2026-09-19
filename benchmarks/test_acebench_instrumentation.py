import json
from types import SimpleNamespace
import pytest
from benchmarks import acebench_cli as hook


def test_official_task_request_action_join_and_user_isolation(tmp_path, monkeypatch):
    output = tmp_path / "harness.jsonl"
    monkeypatch.setenv("C2KV_ACEBENCH_TELEMETRY", str(output))
    monkeypatch.setenv("ACEBENCH_AGENT_BASE_URL", "http://agent/v1")
    requests = []
    def create(resource, **kwargs):
        requests.append(kwargs)
        return SimpleNamespace(c2kv_proxy={"request_id": "request-1"})
    call = hook.request_wrapper(create)
    execute = hook.execution_wrapper(lambda: "official execution result")
    @hook.task_wrapper
    def episode(test_id):
        call(SimpleNamespace(_client=SimpleNamespace(base_url="http://user/v1")), messages=[])
        call(SimpleNamespace(_client=SimpleNamespace(base_url="http://agent/v1/")), messages=[])
        return execute()
    assert episode("agent_multi_turn_1") == "official execution result"
    assert "extra_body" not in requests[0]
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert requests[1]["extra_body"]["c2kv_measurement_session_id"] == rows[0]["session_id"]
    action = next(row for row in rows if row["event_type"] == "tool_action")
    assert action["task_id"] == "agent_multi_turn_1"
    assert action["decision_request_id"] == "request-1"
    assert action["episode_id"] == action["task_id"]
    assert action["episode_instance_id"] == action["session_id"]
    assert action["end_unix_ns"] >= action["start_unix_ns"]
    assert action["duration_ns"] >= 0
    assert rows[-1]["event_type"] == "episode_end"
    assert hook._local.session is None


def test_failed_execution_is_counted_and_original_error_propagates(tmp_path, monkeypatch):
    monkeypatch.setenv("C2KV_ACEBENCH_TELEMETRY", str(tmp_path / "events"))
    @hook.execution_wrapper
    def fail():
        raise ValueError("official failure")
    @hook.task_wrapper
    def episode(test_id):
        hook._local.request = "decision"
        return fail()
    with pytest.raises(ValueError, match="official failure"):
        episode("task")
    rows = [json.loads(line) for line in (tmp_path / "events").read_text().splitlines()]
    assert next(row for row in rows if row["event_type"] == "tool_action")["status"] == "failed"
