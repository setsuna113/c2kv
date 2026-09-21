from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from toolsandbox_cli import install_instrumentation, install_rapidapi_http_status
from measurement.telemetry import HarnessTelemetry


@pytest.mark.parametrize("status", [200, 403, 429])
def test_rapidapi_status_wrapper_keeps_official_response_and_redacts_request(
        tmp_path, status):
    observed = {}
    response = SimpleNamespace(status_code=status, json=lambda: {"official": True})
    def get(*args, **kwargs):
        observed.update(kwargs)
        return response
    shared_requests = SimpleNamespace(get=get, post=object())
    tools = SimpleNamespace(requests=shared_requests)
    path = tmp_path / "rapidapi_http_status.jsonl"
    install_rapidapi_http_status(tools, path)
    assert tools.requests is not shared_requests
    assert shared_requests.get is get
    actual = tools.requests.get(
        url="https://example.rapidapi.com/private/path?token=url-secret",
        headers={"X-RapidAPI-Key": "header-secret"}, params={"q": "param-secret"})
    assert actual is response
    assert actual.json() == {"official": True}
    assert observed["timeout"] == 30
    assert observed["headers"]["X-RapidAPI-Key"] == "header-secret"
    logged = path.read_text(encoding="utf-8")
    assert json.loads(logged) == {"event_type": "rapidapi_http",
                                  "host": "example.rapidapi.com", "status_code": status}
    assert all(secret not in logged for secret in
               ("url-secret", "header-secret", "param-secret", "/private/path"))


def test_rapidapi_transport_error_records_no_url_or_headers(tmp_path):
    def get(*args, **kwargs):
        raise TimeoutError("private URL and key")
    tools = SimpleNamespace(requests=SimpleNamespace(get=get))
    path = tmp_path / "rapidapi_http_status.jsonl"
    install_rapidapi_http_status(tools, path)
    with pytest.raises(TimeoutError):
        tools.requests.get(url="https://example.rapidapi.com/private?key=secret",
                           headers={"X-RapidAPI-Key": "secret"})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "event_type": "rapidapi_http", "host": "example.rapidapi.com",
        "status_code": None,
    }


def test_runtime_instrumentation_joins_scenario_request_and_action(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    telemetry = HarnessTelemetry(path, "toolsandbox",
                                 unix_ns=iter(range(100, 1000)).__next__,
                                 monotonic_ns=iter(range(1000, 2000)).__next__)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://agent/v1")

    class Completions:
        def __init__(self):
            self._client = SimpleNamespace(base_url="http://agent/v1")

        def create(self, **kwargs):
            assert kwargs["extra_body"]["c2kv_measurement_session_id"]
            return SimpleNamespace(model_extra={"c2kv_proxy": {"request_id": "proxy-r1"}})

    class ExecutionModule:
        @staticmethod
        def respond_to_messages_set_all_order_permutations(context, messages, role_type):
            return [SimpleNamespace(openai_tool_call_id="call-1", content="ok",
                                    tool_call_exception=None)]

    agent_role = "agent"

    class Scenario:
        def play_and_evaluate(self, roles, output_directory, scenario_name):
            Completions().create(extra_body={})
            message = SimpleNamespace(sender=agent_role, content="tool()",
                                      openai_tool_call_id="call-1")
            ExecutionModule.respond_to_messages_set_all_order_permutations(
                None, [message], "execution")
            return "official-result"

    install_instrumentation(telemetry, Scenario, Completions,
                            ExecutionModule, agent_role)
    assert Scenario().play_and_evaluate({}, tmp_path, "scenario-1") == "official-result"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in rows] == [
        "episode_start", "decision", "tool_action", "episode_end"]
    decision = rows[1]
    action = rows[2]
    assert decision["episode_id"] == action["episode_id"] == "scenario-1"
    assert decision["decision_request_id"] == "proxy-r1"
    assert action["decision_request_id"] == "proxy-r1"
    assert action["action"] == "tool()"


def test_runtime_instrumentation_rejects_openai_response_id_without_proxy_identity(
        tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    telemetry = HarnessTelemetry(path, "toolsandbox")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://agent/v1")

    class Completions:
        def __init__(self):
            self._client = SimpleNamespace(base_url="http://agent/v1")

        def create(self, **kwargs):
            assert kwargs["extra_body"]["c2kv_measurement_session_id"]
            return SimpleNamespace(id="chatcmpl-is-not-a-proxy-request-id", model_extra={})

    class ExecutionModule:
        @staticmethod
        def respond_to_messages_set_all_order_permutations(context, messages, role_type):
            raise AssertionError("execution must not run after an unjoinable decision")

    class Scenario:
        def play_and_evaluate(self, roles, output_directory, scenario_name):
            return Completions().create(extra_body={})

    install_instrumentation(telemetry, Scenario, Completions, ExecutionModule, "agent")
    with pytest.raises(RuntimeError, match="lacks proxy request identity"):
        Scenario().play_and_evaluate({}, tmp_path, "scenario-missing-id")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event_type"] for row in rows] == ["episode_start", "episode_end"]
    assert rows[-1]["status"] == "error"
